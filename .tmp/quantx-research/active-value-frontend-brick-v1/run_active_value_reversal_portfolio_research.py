from __future__ import annotations

import json
import math
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
PANEL_PATH = ROOT / "panel_all_a_20130101_20260715.parquet"
ACTIVE_VALUE_PATH = Path("data/derived/active_value/daily.parquet")
OUT_JSON = ROOT / "active_value_reversal_portfolio_research.json"
OUT_MD = ROOT / "active_value_reversal_portfolio_research.md"
OUT_TRADES = ROOT / "active_value_reversal_portfolio_trades.parquet"

TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2026-07-15")
INITIAL_CASH = 1_000_000.0
BUY_SLIPPAGE = 0.003
SELL_SLIPPAGE = 0.001
COMMISSION = 0.0005
STAMP_TAX = 0.0005
MIN_FEE = 5.0
MIN_LISTING_DAYS = 120

PANEL_COLUMNS = [
    "instrument",
    "datetime",
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amount",
    "$vwap",
    "ret1",
    "ret3",
    "ret5",
    "vol_ratio",
    "amplitude_pct",
    "body_pct",
    "close_pos",
    "ma5_rel",
    "ma10_rel",
    "ma20_rel",
    "ma50_rel",
    "frontend_brick",
    "frontend_delta",
    "frontend_delta_prev",
    "red_run7",
    "amount_rank_pct",
    "ret1_rank_pct",
    "ret3_rank_pct",
    "ret5_rank_pct",
    "brick_rank_pct",
    "delta_rank_pct",
    "vol_ratio_rank_pct",
    "amplitude_rank_pct",
    "is_mainboard",
    "is_chinext",
    "is_star",
    "is_current_risk_name",
    "st_like_limit_history",
]

FEATURE_COLUMNS = [
    "reversal_value",
    "frontend_brick",
    "ret1",
    "ret3",
    "ret5",
    "amount_rank_pct",
    "ret1_rank_pct",
    "ret3_rank_pct",
    "ret5_rank_pct",
    "brick_rank_pct",
    "delta_rank_pct",
    "vol_ratio",
    "vol_ratio_rank_pct",
    "amplitude_pct",
    "amplitude_rank_pct",
    "body_pct",
    "close_pos",
    "ma5_rel",
    "ma10_rel",
    "ma20_rel",
    "ma50_rel",
    "active_ret1",
    "active_ret2",
    "is_mainboard",
    "is_chinext",
    "is_star",
    "listing_days",
]


@dataclass
class ResearchResult:
    ok: bool
    input_panel: str
    active_value_diagnostics: dict
    candidate_diagnostics: dict
    quantile_diagnostics: list[dict]
    replay_grid: list[dict]
    attribution: list[dict]
    ml_diagnostics: dict
    output_trades: str
    notes: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Research active-value + frontend-brick reversal portfolios.")
    parser.add_argument("--with-permutation-importance", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()

    panel = load_panel(PANEL_PATH)
    panel = add_trade_state(panel)
    active = load_active_value_daily(ACTIVE_VALUE_PATH)
    panel = attach_active_value_daily(panel, active, "active_core_amount")
    panel = add_forward_returns(panel)
    active_candidates = build_candidates(panel, require_reversal=False)
    candidates = active_candidates[active_candidates["is_brick_reversal"]].copy()

    active_diag = summarize_active_variants(active)
    candidate_diag = summarize_candidates(candidates)
    quantiles = build_quantile_diagnostics(candidates)

    replay_rows: list[dict] = []
    replay_results: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    replay_context = build_replay_context(panel)
    for config in replay_configs(full_grid=args.full_grid):
        pool = candidates if config["candidate_pool"] == "brick" else active_candidates
        signals = select_signals(pool, **config)
        replay = replay_portfolio(replay_context, signals, config)
        replay_results.append(replay)
        replay_rows.append(replay["summary"])
        if not replay["trades"].empty:
            trade_frames.append(replay["trades"].assign(config=replay["summary"]["config"]))

    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
    else:
        trades = pd.DataFrame()
    trades.to_parquet(OUT_TRADES, index=False)

    attribution = build_attribution(replay_results, panel)
    ml_diag = run_ml_diagnostic(candidates, with_permutation_importance=args.with_permutation_importance)

    result = ResearchResult(
        ok=True,
        input_panel=str(PANEL_PATH),
        active_value_diagnostics=active_diag,
        candidate_diagnostics=candidate_diag,
        quantile_diagnostics=quantiles,
        replay_grid=replay_rows,
        attribution=attribution,
        ml_diagnostics=ml_diag,
        output_trades=str(OUT_TRADES),
        notes=[
            f"Active-value data is read from {ACTIVE_VALUE_PATH} and is no longer recomputed inside this research script.",
            "active_core_amount is the primary high-quality local 0AMV proxy: sum amount for mainboard, non-ST, non-ST-like, non-new, non-suspended tradable rows.",
            "The main signal window is active_strong_up_day + reversal_prev_down, with reversal_value=frontend_delta.",
            "Signals are generated on T, buys are attempted at T+1 open with slippage, commission, ST/new-stock/suspension/one-price-limit-up filters.",
            "Fixed-hold labels and replay both use T+1 open entry and T+6/T+11 open exit for 5/10 held trading days.",
            "Attribution regresses saved daily NAV returns on equal-weight market and cross-sectional momentum factors; the active-only versus brick-gated momentum pair is the brick ablation.",
            "ML diagnostics are OOS-only research checks and are not yet wired into the formal QuantX strategy engine. Permutation importance is optional because it is slow on the full panel.",
        ],
    )
    OUT_JSON.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=PANEL_COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    for col in frame.columns:
        if col not in {"instrument", "datetime"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_active_value_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing active-value data: {path}. Run `conda run -n test python -m quantx.tools.build_active_value_data --end 2026-07-15` first."
        )
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise ValueError(f"Active-value data lacks date column: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def add_trade_state(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("instrument", sort=False)
    out["prev_close"] = grouped["$close"].shift(1)
    out["first_date"] = grouped["datetime"].transform("min")
    out["listing_days"] = grouped.cumcount().astype("int32")
    out["one_price_limit_up"] = (
        (out["$high"].sub(out["$low"]).abs() <= 1e-6)
        & (out["$open"].sub(out["$close"]).abs() <= 1e-6)
        & (out["prev_close"] > 0)
        & (out["$close"] / out["prev_close"] - 1.0 >= 0.045)
    )
    out["one_price_limit_down"] = (
        (out["$high"].sub(out["$low"]).abs() <= 1e-6)
        & (out["$open"].sub(out["$close"]).abs() <= 1e-6)
        & (out["prev_close"] > 0)
        & (out["$close"] / out["prev_close"] - 1.0 <= -0.045)
    )
    out["basic_tradable"] = (
        (out["$volume"] > 0)
        & (out["$amount"] > 0)
        & (out["$open"] > 0)
        & (out["$close"] > 0)
    )
    out["clean_universe"] = (
        out["basic_tradable"].fillna(False)
        & (~out["is_current_risk_name"].fillna(False).astype(bool))
        & (~out["st_like_limit_history"].fillna(False).astype(bool))
        & (out["listing_days"] >= MIN_LISTING_DAYS)
    )
    return out


def attach_active_value_daily(frame: pd.DataFrame, active: pd.DataFrame, variant: str) -> pd.DataFrame:
    cols = [
        "date",
        variant,
        f"{variant}_ret1",
        f"{variant}_ret2",
        f"{variant}_ma10",
        f"{variant}_strong_up_day",
        f"{variant}_above_ma10",
    ]
    missing = [col for col in cols if col not in active.columns]
    if missing:
        raise ValueError(f"Active-value data missing required columns: {missing}")
    active_cols = active[cols].rename(columns={"date": "datetime"})
    out = frame.merge(active_cols, on="datetime", how="left")
    out = out.rename(
        columns={
            variant: "active_value_hq",
            f"{variant}_ret1": "active_ret1",
            f"{variant}_ret2": "active_ret2",
            f"{variant}_ma10": "active_ma10",
            f"{variant}_strong_up_day": "active_strong_up_day",
            f"{variant}_above_ma10": "active_above_ma10",
        }
    )
    return out


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("instrument", sort=False)
    out["next_open"] = grouped["$open"].shift(-1)
    out["next_volume"] = grouped["$volume"].shift(-1)
    out["next_amount"] = grouped["$amount"].shift(-1)
    out["next_high"] = grouped["$high"].shift(-1)
    out["next_low"] = grouped["$low"].shift(-1)
    out["next_close"] = grouped["$close"].shift(-1)
    out["next_one_price_limit_up"] = grouped["one_price_limit_up"].shift(-1)
    out["next_clean_universe"] = grouped["clean_universe"].shift(-1)
    for horizon in (5, 10):
        exit_open = grouped["$open"].shift(-(horizon + 1))
        out[f"exit_open_{horizon}"] = exit_open
        out[f"fwd{horizon}"] = exit_open / out["next_open"] - 1.0
        entry_cost = out["next_open"] * (1 + BUY_SLIPPAGE) * (1 + COMMISSION)
        exit_proceeds = exit_open * (1 - SELL_SLIPPAGE) * (1 - COMMISSION - STAMP_TAX)
        out[f"fwd{horizon}_net"] = exit_proceeds / entry_cost - 1.0
    return out


def build_candidates(panel: pd.DataFrame, *, require_reversal: bool = True) -> pd.DataFrame:
    is_reversal = (panel["frontend_delta_prev"] < 0) & (panel["frontend_delta"] > 0)
    signal = (
        panel["active_strong_up_day"].fillna(False).astype(bool)
        & panel["clean_universe"].fillna(False)
        & (panel["amount_rank_pct"] >= 0.35)
        & (panel["amplitude_pct"] < 0.22)
    )
    if require_reversal:
        signal &= is_reversal
    cols = [
        "instrument",
        "datetime",
        "active_value_hq",
        "active_ret1",
        "active_ret2",
        "active_above_ma10",
        "frontend_brick",
        "frontend_delta",
        "frontend_delta_prev",
        "ret1",
        "ret3",
        "ret5",
        "vol_ratio",
        "amplitude_pct",
        "body_pct",
        "close_pos",
        "ma5_rel",
        "ma10_rel",
        "ma20_rel",
        "ma50_rel",
        "amount_rank_pct",
        "ret1_rank_pct",
        "ret3_rank_pct",
        "ret5_rank_pct",
        "brick_rank_pct",
        "delta_rank_pct",
        "vol_ratio_rank_pct",
        "amplitude_rank_pct",
        "is_mainboard",
        "is_chinext",
        "is_star",
        "listing_days",
        "fwd5",
        "fwd10",
        "fwd5_net",
        "fwd10_net",
        "next_open",
        "next_volume",
        "next_amount",
        "next_high",
        "next_low",
        "next_close",
        "next_one_price_limit_up",
        "next_clean_universe",
        "exit_open_5",
        "exit_open_10",
    ]
    out = panel.loc[signal, cols].copy()
    out["is_brick_reversal"] = is_reversal.loc[signal].astype(bool).to_numpy()
    out["reversal_value"] = out["frontend_delta"]
    out["stock_ret4"] = out["ret1"] >= 0.04
    out["entry_one_price_limit_up"] = out["next_one_price_limit_up"].eq(True)
    out["entry_zero_volume_or_amount"] = (out["next_volume"] <= 0) | (out["next_amount"] <= 0)
    out["entry_tradable"] = (
        (~out["entry_one_price_limit_up"])
        & (~out["entry_zero_volume_or_amount"].fillna(False))
        & out["next_clean_universe"].eq(True)
        & (out["next_open"] > 0)
    )
    out["rule_score"] = (
        0.35 * out["delta_rank_pct"]
        + 0.25 * out["ret1_rank_pct"]
        + 0.20 * out["brick_rank_pct"]
        + 0.20 * out["amount_rank_pct"]
    )
    out["momentum_score"] = 0.65 * out["ret1_rank_pct"] + 0.35 * out["amount_rank_pct"]
    rng = np.random.default_rng(20260715)
    out["random_score"] = rng.random(len(out))
    return out.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def replay_configs(*, full_grid: bool) -> list[dict]:
    configs = []
    topks = (5, 10, 20) if full_grid else (10,)
    ret4_options = (False, True) if full_grid else (False,)
    for topk in topks:
        for require_ret4 in ret4_options:
            base = {"topk": topk, "require_ret4": require_ret4, "candidate_pool": "brick"}
            configs.append({**base, "name": f"rule_top{topk}_fixed5_ret4_{int(require_ret4)}", "score_col": "rule_score", "mode": "fixed", "hold_days": 5, "force_market_exit": False})
            configs.append({**base, "name": f"rule_top{topk}_fixed10_ret4_{int(require_ret4)}", "score_col": "rule_score", "mode": "fixed", "hold_days": 10, "force_market_exit": False})
            configs.append({**base, "name": f"rule_top{topk}_brick_exit_ret4_{int(require_ret4)}", "score_col": "rule_score", "mode": "brick", "hold_days": 10, "force_market_exit": False})
            configs.append({**base, "name": f"rule_top{topk}_brick_market_exit_ret4_{int(require_ret4)}", "score_col": "rule_score", "mode": "brick", "hold_days": 10, "force_market_exit": True})
    configs.extend([
        {"name": "brick_momentum_top10_fixed10", "topk": 10, "score_col": "momentum_score", "mode": "fixed", "hold_days": 10, "require_ret4": False, "force_market_exit": False, "candidate_pool": "brick"},
        {"name": "active_momentum_top10_fixed10", "topk": 10, "score_col": "momentum_score", "mode": "fixed", "hold_days": 10, "require_ret4": False, "force_market_exit": False, "candidate_pool": "active"},
        {"name": "active_random_top10_fixed10", "topk": 10, "score_col": "random_score", "mode": "fixed", "hold_days": 10, "require_ret4": False, "force_market_exit": False, "candidate_pool": "active"},
    ])
    return configs


def select_signals(candidates: pd.DataFrame, *, topk: int, score_col: str, require_ret4: bool, **_: object) -> pd.DataFrame:
    data = candidates[candidates["entry_tradable"].fillna(False)].copy()
    if require_ret4:
        data = data[data["stock_ret4"].fillna(False)]
    data = data.dropna(subset=[score_col, "datetime", "instrument"])
    data = data[data["datetime"] >= OOS_START]
    return data.sort_values(["datetime", score_col], ascending=[True, False]).groupby("datetime", as_index=False).head(int(topk))


def build_replay_context(panel: pd.DataFrame) -> dict:
    panel_oos = panel[(panel["datetime"] >= OOS_START) & (panel["datetime"] <= END)].copy()
    calendar = pd.DatetimeIndex(sorted(panel_oos["datetime"].dropna().unique()))
    by_symbol = {symbol: day.reset_index(drop=True) for symbol, day in panel.groupby("instrument", sort=False)}
    date_arrays = {symbol: pd.to_datetime(day["datetime"]).to_numpy() for symbol, day in by_symbol.items()}
    active_by_date = panel.drop_duplicates("datetime").set_index("datetime")[["active_above_ma10"]]
    return {"calendar": calendar, "by_symbol": by_symbol, "date_arrays": date_arrays, "active_by_date": active_by_date}


def replay_portfolio(context: dict, signals: pd.DataFrame, config: dict) -> dict:
    calendar = context["calendar"]
    by_symbol = context["by_symbol"]
    date_arrays = context["date_arrays"]
    active_by_date = context["active_by_date"]
    market_state_by_date = context.get("market_state_by_date")
    signals_by_date = {pd.Timestamp(k): g.sort_values(config["score_col"], ascending=False) for k, g in signals.groupby("datetime")}

    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    nav_rows = []
    trades = []
    blocked = {"missing_bar": 0, "zero_volume_or_amount": 0, "one_price_limit_up": 0, "already_held": 0, "cash_or_lot": 0, "one_price_limit_down_sell_delay": 0}

    for date in calendar:
        for symbol in list(positions):
            hist = by_symbol.get(symbol)
            pos = find_pos(date_arrays, symbol, date)
            if hist is None or pos is None or pos <= positions[symbol]["buy_pos"]:
                continue
            decision = pos - 1
            holding_days = pos - positions[symbol]["buy_pos"]
            decision_close = float(hist.loc[decision, "$close"])
            position = positions[symbol]
            if np.isfinite(decision_close) and decision_close > 0:
                position["peak_close"] = max(float(position["peak_close"]), decision_close)
            reason = sell_reason(
                hist,
                decision,
                holding_days,
                config,
                active_by_date,
                position=position,
                market_state_by_date=market_state_by_date,
            )
            if reason:
                bar = hist.loc[pos]
                if bool(bar.get("one_price_limit_down", False)):
                    blocked["one_price_limit_down_sell_delay"] += 1
                    continue
                open_price = float(bar["$open"])
                if np.isfinite(open_price) and open_price > 0:
                    qty = positions[symbol]["qty"]
                    price = open_price * (1 - SELL_SLIPPAGE)
                    gross = qty * price
                    fee = max(gross * COMMISSION, MIN_FEE) + gross * STAMP_TAX
                    cash += gross - fee
                    position = positions[symbol]
                    net_proceeds = gross - fee
                    realized_return = net_proceeds / position["entry_cost"] - 1.0
                    expected_return = position.get("expected_return")
                    trades.append({
                        "date": date,
                        "signal_date": position["signal_date"],
                        "action": "SELL",
                        "symbol": symbol,
                        "price": price,
                        "qty": qty,
                        "reason": reason,
                        "fee": fee,
                        "holding_days": holding_days,
                        "expected_holding_days": position.get("expected_holding_days"),
                        "realized_return": realized_return,
                        "expected_return": expected_return,
                        "label_error": realized_return - expected_return if expected_return is not None and np.isfinite(expected_return) else np.nan,
                    })
                    del positions[symbol]

        prev_idx = calendar.get_loc(date) - 1
        if prev_idx >= 0:
            signal_date = calendar[prev_idx]
            picks = signals_by_date.get(signal_date)
            if picks is not None:
                slots = max(0, int(config["topk"]) - len(positions))
                buy_list = []
                for row in picks.itertuples(index=False):
                    if len(buy_list) >= slots:
                        break
                    symbol = str(row.instrument)
                    if symbol in positions:
                        blocked["already_held"] += 1
                        continue
                    hist = by_symbol.get(symbol)
                    pos = find_pos(date_arrays, symbol, date)
                    if hist is None or pos is None:
                        blocked["missing_bar"] += 1
                        continue
                    bar = hist.loc[pos]
                    if not bool(bar.get("basic_tradable", False)):
                        blocked["zero_volume_or_amount"] += 1
                        continue
                    if bool(bar.get("one_price_limit_up", False)):
                        blocked["one_price_limit_up"] += 1
                        continue
                    buy_list.append((symbol, pos, float(bar["$open"]), row))
                if buy_list:
                    per_cash = cash * 0.98 / len(buy_list)
                    for symbol, pos, open_price, signal_row in buy_list:
                        buy_price = open_price * (1 + BUY_SLIPPAGE)
                        qty = int(per_cash / buy_price / 100) * 100
                        if qty <= 0:
                            blocked["cash_or_lot"] += 1
                            continue
                        gross = qty * buy_price
                        fee = max(gross * COMMISSION, MIN_FEE)
                        if cash < gross + fee:
                            blocked["cash_or_lot"] += 1
                            continue
                        cash -= gross + fee
                        expected_return = getattr(signal_row, f"fwd{config['hold_days']}_net", np.nan) if config["mode"] == "fixed" else np.nan
                        positions[symbol] = {
                            "qty": qty,
                            "buy_price": buy_price,
                            "buy_date": date,
                            "buy_pos": pos,
                            "signal_date": signal_date,
                            "entry_cost": gross + fee,
                            "expected_return": expected_return,
                            "expected_holding_days": int(config["hold_days"]) if config["mode"] == "fixed" else None,
                            "peak_close": buy_price,
                        }
                        trades.append({"date": date, "signal_date": signal_date, "action": "BUY", "symbol": symbol, "price": buy_price, "qty": qty, "reason": config["name"], "fee": fee, "holding_days": 0, "expected_holding_days": positions[symbol]["expected_holding_days"], "realized_return": np.nan, "expected_return": expected_return, "label_error": np.nan})

        total = cash
        invested_value = 0.0
        for symbol, posn in positions.items():
            hist = by_symbol.get(symbol)
            pos = find_pos(date_arrays, symbol, date)
            if hist is not None and pos is not None:
                close = float(hist.loc[pos, "$close"])
                if np.isfinite(close) and close > 0:
                    market_value = posn["qty"] * close
                    invested_value += market_value
                    total += market_value
        nav_rows.append({
            "date": date,
            "total_value": total,
            "cash": cash,
            "invested_value": invested_value,
            "gross_exposure": invested_value / total if total > 0 else 0.0,
            "positions": len(positions),
        })

    nav = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trades)
    summary = summarize_nav(nav, trades_df, blocked, config)
    return {"summary": summary, "nav": nav, "trades": trades_df}


def sell_reason(
    hist: pd.DataFrame,
    decision: int,
    holding_days: int,
    config: dict,
    active_by_date: pd.DataFrame,
    *,
    position: dict | None = None,
    market_state_by_date: pd.DataFrame | None = None,
) -> str | None:
    if config["mode"] == "fixed" and holding_days >= int(config["hold_days"]):
        return f"fixed{config['hold_days']}"
    if config["mode"] == "brick":
        if holding_days >= 1 and float(hist.loc[decision, "frontend_delta"]) < 0:
            return "green"
        if float(hist.loc[decision, "red_run7"]) >= 7:
            return "7red"
        if bool(config.get("force_market_exit")):
            date = pd.Timestamp(hist.loc[decision, "datetime"])
            if date in active_by_date.index and not bool(active_by_date.loc[date, "active_above_ma10"]):
                return "active_ma10_break"
        if holding_days >= int(config["hold_days"]):
            return f"max_hold_{config['hold_days']}"
    if config["mode"] == "risk" and position is not None:
        close = float(hist.loc[decision, "$close"])
        buy_price = float(position["buy_price"])
        peak_close = float(position.get("peak_close", close))
        if holding_days >= 1 and config.get("stop_loss") is not None:
            if close / buy_price - 1.0 <= -float(config["stop_loss"]):
                return "stop_loss"
        if holding_days >= 1 and config.get("take_profit") is not None:
            if close / buy_price - 1.0 >= float(config["take_profit"]):
                return "take_profit"
        if holding_days >= 1 and config.get("trailing_drawdown") is not None:
            activation = float(config.get("trailing_activation", 0.0))
            if peak_close / buy_price - 1.0 >= activation:
                if close / peak_close - 1.0 <= -float(config["trailing_drawdown"]):
                    return "trailing_stop"
        date = pd.Timestamp(hist.loc[decision, "datetime"])
        if holding_days >= 1 and market_state_by_date is not None and date in market_state_by_date.index:
            state = market_state_by_date.loc[date]
            if bool(config.get("exit_breadth")) and not bool(state.get("breadth_healthy", False)):
                return "breadth_break"
            if bool(config.get("exit_amv")) and not bool(state.get("amv_quantile_wave", False)):
                return "amv_wave_end"
        if holding_days >= int(config["hold_days"]):
            return f"max_hold_{config['hold_days']}"
    return None


def summarize_nav(nav: pd.DataFrame, trades: pd.DataFrame, blocked: dict, config: dict) -> dict:
    if nav.empty:
        return {"config": config["name"], "total_return": None}
    nav = nav.copy()
    nav["ret"] = nav["total_value"].pct_change(fill_method=None).fillna(0.0)
    nav["cummax"] = nav["total_value"].cummax()
    nav["drawdown"] = nav["total_value"] / nav["cummax"] - 1.0
    years = {}
    nav["year"] = pd.to_datetime(nav["date"]).dt.year
    for year, group in nav.groupby("year"):
        years[str(year)] = float(group.iloc[-1]["total_value"] / group.iloc[0]["total_value"] - 1.0)
    buys = trades[trades["action"] == "BUY"] if not trades.empty else pd.DataFrame()
    sells = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()
    return {
        "config": config["name"],
        "topk": int(config["topk"]),
        "score_col": config["score_col"],
        "mode": config["mode"],
        "require_ret4": bool(config["require_ret4"]),
        "force_market_exit": bool(config["force_market_exit"]),
        "candidate_pool": config["candidate_pool"],
        "final_value": float(nav.iloc[-1]["total_value"]),
        "total_return": float(nav.iloc[-1]["total_value"] / INITIAL_CASH - 1.0),
        "max_drawdown": float(nav["drawdown"].min()),
        "daily_mean_return": float(nav["ret"].mean()),
        "daily_vol": float(nav["ret"].std()),
        "sharpe_annualized": safe_sharpe(nav["ret"]),
        "yearly_return": years,
        "avg_positions": float(nav["positions"].mean()),
        "avg_gross_exposure": safe_mean(nav.get("gross_exposure")),
        "median_positions": float(nav["positions"].median()),
        "buys": int(len(buys)),
        "sells": int(len(sells)),
        "avg_holding_days": safe_mean(sells.get("holding_days")),
        "sell_reasons": sells["reason"].value_counts().to_dict() if not sells.empty else {},
        "fixed_label_audit": summarize_label_audit(sells),
        "blocked": blocked,
    }


def summarize_label_audit(sells: pd.DataFrame) -> dict:
    if sells.empty or "label_error" not in sells:
        return {"rows": 0, "mae": None, "max_abs_error": None, "correlation": None}
    candidates = sells.dropna(subset=["realized_return", "expected_return", "label_error"])
    delayed = candidates[candidates["holding_days"] != candidates["expected_holding_days"]]
    audit = candidates[candidates["holding_days"] == candidates["expected_holding_days"]]
    if audit.empty:
        return {"rows": 0, "delayed_exit_rows": int(len(delayed)), "mae": None, "max_abs_error": None, "correlation": None}
    return {
        "rows": int(len(audit)),
        "delayed_exit_rows": int(len(delayed)),
        "mae": safe_mean(audit["label_error"].abs()),
        "max_abs_error": float(audit["label_error"].abs().max()),
        "correlation": safe_corr(audit["realized_return"], audit["expected_return"]),
    }


def build_quantile_diagnostics(candidates: pd.DataFrame) -> list[dict]:
    records = []
    data = candidates[candidates["datetime"] >= OOS_START].dropna(subset=["reversal_value", "fwd5_net", "fwd10_net"])
    for require_ret4, sample in [(False, data), (True, data[data["stock_ret4"].fillna(False)])]:
        if sample.empty or sample["reversal_value"].nunique() < 5:
            continue
        sample = sample.copy()
        sample["bucket"] = pd.qcut(sample["reversal_value"], 5, duplicates="drop")
        for bucket, group in sample.groupby("bucket", observed=True):
            for target in ("fwd5_net", "fwd10_net"):
                records.append({
                    "sample": "ret1_ge_4" if require_ret4 else "all",
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "reversal_value_mean": safe_mean(group["reversal_value"]),
                    "target": target,
                    "mean": safe_mean(group[target]),
                    "median": safe_median(group[target]),
                    "win": safe_mean(group[target] > 0),
                })
    return records


def build_attribution(replay_results: list[dict], panel: pd.DataFrame) -> list[dict]:
    factors = build_daily_factors(panel)
    rows = []
    nav_returns: dict[str, pd.DataFrame] = {}
    for replay in replay_results:
        summary = replay["summary"]
        nav_columns = ["date", "total_value"]
        if "gross_exposure" in replay["nav"].columns:
            nav_columns.append("gross_exposure")
        nav = replay["nav"][nav_columns].copy()
        nav["portfolio_ret"] = nav["total_value"].pct_change(fill_method=None)
        nav_returns[summary["config"]] = nav[["date", "portfolio_ret"]]
        sample = nav.merge(factors, left_on="date", right_on="datetime", how="inner").dropna(
            subset=["portfolio_ret", "market_ret", "momentum_factor"]
        )
        regression = factor_regression(sample)
        exposure_regression = factor_regression_with_exposure(sample)
        invested_sample = sample[sample["gross_exposure"] > 0.01] if "gross_exposure" in sample.columns else sample
        invested_regression = factor_regression(invested_sample)
        rows.append({
            "config": summary["config"],
            "candidate_pool": summary.get("candidate_pool"),
            "total_return": summary.get("total_return"),
            "avg_positions": summary.get("avg_positions"),
            **regression,
            "exposure_matched": exposure_regression,
            "invested_days_only": invested_regression,
            "interpretation_hint": attribution_hint(summary),
        })

    active_name = "active_momentum_top10_fixed10"
    brick_name = "brick_momentum_top10_fixed10"
    if active_name in nav_returns and brick_name in nav_returns:
        pair = nav_returns[brick_name].merge(nav_returns[active_name], on="date", suffixes=("_brick", "_active")).dropna()
        diff = pair["portfolio_ret_brick"] - pair["portfolio_ret_active"]
        rows.append({
            "config": "brick_gate_paired_ablation",
            "brick_config": brick_name,
            "control_config": active_name,
            "days": int(len(diff)),
            "mean_daily_increment": safe_mean(diff),
            "annualized_increment": float(diff.mean() * 252) if len(diff) else None,
            "positive_day_ratio": safe_mean(diff > 0),
            "paired_t_stat": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))) if len(diff) > 2 and diff.std(ddof=1) > 0 else None,
            "interpretation_hint": "incremental_effect_of_requiring_brick_reversal_with_same_momentum_rank",
        })
    return rows


def build_daily_factors(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel[(panel["datetime"] >= OOS_START) & (panel["datetime"] <= END)].dropna(subset=["ret1", "ret5"]).copy()
    data = data[data["clean_universe"].fillna(False)]
    data["momentum_bucket"] = data.groupby("datetime")["ret5"].rank(pct=True, method="average")
    market = data.groupby("datetime", as_index=False).agg(market_ret=("ret1", "mean"))
    winners = data[data["momentum_bucket"] >= 0.8].groupby("datetime")["ret1"].mean()
    losers = data[data["momentum_bucket"] <= 0.2].groupby("datetime")["ret1"].mean()
    momentum = winners.sub(losers).rename("momentum_factor").reset_index()
    return market.merge(momentum, on="datetime", how="inner")


def factor_regression(sample: pd.DataFrame) -> dict:
    if len(sample) < 20:
        return {"observations": int(len(sample)), "alpha_daily": None, "alpha_annualized": None, "market_beta": None, "momentum_beta": None, "r_squared": None}
    y = sample["portfolio_ret"].to_numpy(dtype=float)
    x = np.column_stack([
        np.ones(len(sample)),
        sample["market_ret"].to_numpy(dtype=float),
        sample["momentum_factor"].to_numpy(dtype=float),
    ])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    total_ss = float(np.square(y - y.mean()).sum())
    residual_ss = float(np.square(y - fitted).sum())
    return {
        "observations": int(len(sample)),
        "alpha_daily": float(coef[0]),
        "alpha_annualized": float((1 + coef[0]) ** 252 - 1) if coef[0] > -1 else None,
        "market_beta": float(coef[1]),
        "momentum_beta": float(coef[2]),
        "r_squared": 1.0 - residual_ss / total_ss if total_ss > 0 else None,
    }


def factor_regression_with_exposure(sample: pd.DataFrame) -> dict:
    if "gross_exposure" not in sample.columns:
        return factor_regression(sample)
    data = sample.copy()
    data["gross_exposure"] = pd.to_numeric(data["gross_exposure"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    data["market_ret"] = data["market_ret"] * data["gross_exposure"]
    data["momentum_factor"] = data["momentum_factor"] * data["gross_exposure"]
    result = factor_regression(data)
    result["mean_gross_exposure"] = safe_mean(data["gross_exposure"])
    return result


def attribution_hint(row: dict) -> str:
    config = str(row.get("config", ""))
    if "momentum" in config:
        return "momentum_only_baseline"
    if "random" in config:
        return "random_reversal_baseline"
    return "brick_reversal_rule"


def run_ml_diagnostic(candidates: pd.DataFrame, *, with_permutation_importance: bool = False) -> dict:
    data = candidates.dropna(subset=["fwd10_net", *FEATURE_COLUMNS]).copy()
    train = data[data["datetime"] <= TRAIN_END]
    oos = data[data["datetime"] >= OOS_START]
    if train.empty or oos.empty:
        return {"ok": False, "reason": "empty_train_or_oos"}
    y = 0.5 * train["fwd5_net"].fillna(train["fwd10_net"]) + 0.5 * train["fwd10_net"]
    weights = 1.0 + y.clip(lower=0, upper=0.20) * 20.0 + (y > 0).astype(float) * 2.0
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=20260715),
    )
    model.fit(train[FEATURE_COLUMNS], y, histgradientboostingregressor__sample_weight=weights)
    oos = oos.copy()
    oos["ml_score"] = model.predict(oos[FEATURE_COLUMNS])
    rule_top = oos.sort_values(["datetime", "rule_score"], ascending=[True, False]).groupby("datetime", as_index=False).head(10)
    ml_top = oos.sort_values(["datetime", "ml_score"], ascending=[True, False]).groupby("datetime", as_index=False).head(10)
    if not with_permutation_importance:
        feature_importance = [{"skipped": "pass --with-permutation-importance to compute slow permutation importance"}]
    else:
        try:
            pi_sample = oos.sample(min(20000, len(oos)), random_state=20260715)
            importance = permutation_importance(model, pi_sample[FEATURE_COLUMNS], pi_sample["fwd10_net"], n_repeats=2, random_state=20260715, n_jobs=1)
            feature_importance = sorted(
                [{"feature": feature, "importance": float(score)} for feature, score in zip(FEATURE_COLUMNS, importance.importances_mean)],
                key=lambda row: row["importance"],
                reverse=True,
            )[:15]
        except Exception as exc:  # noqa: BLE001 - diagnostic must not break the research run.
            feature_importance = [{"error": str(exc)}]
    return {
        "ok": True,
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "rule_top10": describe_returns(rule_top, "fwd10_net"),
        "ml_top10": describe_returns(ml_top, "fwd10_net"),
        "ml_top10_yearly": yearly_returns(ml_top, "fwd10_net"),
        "rule_top10_yearly": yearly_returns(rule_top, "fwd10_net"),
        "feature_importance_top15": feature_importance,
    }


def summarize_active_variants(active: pd.DataFrame) -> dict:
    out = {"dates": int(active["date"].nunique()), "source": str(ACTIVE_VALUE_PATH)}
    for name in ["active_all_amount", "active_tradable_amount", "active_mainboard_amount", "active_core_amount", "active_vwap_value"]:
        strong_col = f"{name}_strong_up_day"
        above_col = f"{name}_above_ma10"
        out[name] = {
            "last": safe_last(active[name]),
            "strong_days": int(active[strong_col].fillna(False).sum()) if strong_col in active else None,
            "above_ma10_days": int(active[above_col].fillna(False).sum()) if above_col in active else None,
            "ret1_mean": safe_mean(active[f"{name}_ret1"]),
            "ret1_std": safe_std(active[f"{name}_ret1"]),
            "corr_with_all_amount": safe_corr(active[name], active["active_all_amount"]),
        }
    return out


def summarize_candidates(candidates: pd.DataFrame) -> dict:
    train = candidates[candidates["datetime"] <= TRAIN_END]
    oos = candidates[candidates["datetime"] >= OOS_START]
    return {
        "rows": int(len(candidates)),
        "dates": int(candidates["datetime"].nunique()) if len(candidates) else 0,
        "train": describe_returns(train, "fwd10_net"),
        "oos_fwd5": describe_returns(oos, "fwd5_net"),
        "oos_fwd10": describe_returns(oos, "fwd10_net"),
        "oos_ret1_ge_4_fwd10": describe_returns(oos[oos["stock_ret4"].fillna(False)], "fwd10_net"),
        "entry_tradable_ratio": safe_mean(candidates["entry_tradable"]),
        "entry_one_price_limit_up_ratio": safe_mean(candidates["entry_one_price_limit_up"]),
        "entry_zero_volume_or_amount_ratio": safe_mean(candidates["entry_zero_volume_or_amount"]),
    }


def describe_returns(frame: pd.DataFrame, target: str) -> dict:
    if frame is None or frame.empty or target not in frame:
        return {"rows": 0, "dates": 0, "mean": None, "median": None, "win": None}
    values = pd.to_numeric(frame[target], errors="coerce")
    return {
        "rows": int(values.notna().sum()),
        "dates": int(frame.loc[values.notna(), "datetime"].nunique()) if "datetime" in frame else 0,
        "mean": safe_mean(values),
        "median": safe_median(values),
        "win": safe_mean(values > 0),
    }


def yearly_returns(frame: pd.DataFrame, target: str) -> dict:
    if frame.empty or target not in frame:
        return {}
    out = {}
    for year, group in frame.groupby(pd.to_datetime(frame["datetime"]).dt.year):
        out[str(year)] = describe_returns(group, target)
    return out


def render_markdown(result: ResearchResult) -> str:
    lines = [
        "# Active Value Reversal Portfolio Research",
        "",
        f"Input panel: `{result.input_panel}`",
        f"Trades: `{result.output_trades}`",
        "",
        "## Active Value Variants",
        "",
    ]
    for name, payload in result.active_value_diagnostics.items():
        if not isinstance(payload, dict):
            lines.append(f"- {name}: {payload}")
            continue
        lines.append(f"- {name}: strong_days={payload.get('strong_days')}, corr_with_all={fmt(payload.get('corr_with_all_amount'))}, last={fmt(payload.get('last'))}")
    lines.extend(["", "## Candidates", ""])
    for key, value in result.candidate_diagnostics.items():
        lines.append(f"- {key}: {fmt(value)}")
    lines.extend(["", "## Replay Grid", ""])
    for row in sorted(result.replay_grid, key=lambda x: (x.get("total_return") is None, -(x.get("total_return") or -999)))[:20]:
        lines.append(
            f"- {row.get('config')}: return={fmt(row.get('total_return'))}, mdd={fmt(row.get('max_drawdown'))}, "
            f"sharpe={fmt(row.get('sharpe_annualized'))}, buys={row.get('buys')}, avg_pos={fmt(row.get('avg_positions'))}, yearly={row.get('yearly_return')}"
        )
        if row.get("fixed_label_audit", {}).get("rows"):
            lines.append(f"  label_audit={fmt(row.get('fixed_label_audit'))}")
    lines.extend(["", "## Attribution", ""])
    for row in result.attribution:
        lines.append(f"- {row.get('config')}: {fmt(row)}")
    lines.extend(["", "## Quantiles", ""])
    for row in result.quantile_diagnostics[:40]:
        lines.append(f"- {row['sample']} / {row['target']} / {row['bucket']}: rows={row['rows']}, value={fmt(row['reversal_value_mean'])}, mean={fmt(row['mean'])}, win={fmt(row['win'])}")
    lines.extend(["", "## ML", ""])
    for key, value in result.ml_diagnostics.items():
        lines.append(f"- {key}: {fmt(value)}")
    lines.extend(["", "## Notes", ""])
    for note in result.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def find_pos(date_arrays: dict[str, np.ndarray], symbol: str, date: pd.Timestamp) -> int | None:
    dates = date_arrays.get(symbol)
    if dates is None:
        return None
    idx = int(np.searchsorted(dates, np.datetime64(date), side="left"))
    if idx >= len(dates) or pd.Timestamp(dates[idx]) != date:
        return None
    return idx


def safe_mean(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.mean()) if len(series) else None


def safe_median(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.median()) if len(series) else None


def safe_std(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.std()) if len(series) else None


def safe_corr(left, right) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return None
    return float(frame["left"].corr(frame["right"]))


def safe_last(values) -> float | None:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.iloc[-1]) if len(series) else None


def safe_sharpe(values) -> float | None:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 3 or series.std() == 0 or not np.isfinite(series.std()):
        return None
    return float(series.mean() / series.std() * math.sqrt(252))


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, list):
        return json.dumps(value[:10], ensure_ascii=False, default=str)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
