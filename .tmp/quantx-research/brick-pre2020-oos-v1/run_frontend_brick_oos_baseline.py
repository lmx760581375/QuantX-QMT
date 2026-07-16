from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_backtest import load_a_share_symbols, load_config, load_symbols


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline")
PROVIDER = Path("data/qlib_data_fixed")
META = Path("data/meta/quantx_meta.sqlite")
START = "2016-01-01"
TRAIN_END = "2024-12-31"
OOS_START = "2025-01-02"
END = "2026-07-15"
TOPK = 7
MAX_POSITIONS = 10

FEATURE_COLUMNS = [
    "frontend_brick",
    "frontend_prev",
    "frontend_delta",
    "frontend_delta_prev",
    "frontend_decline_sum_prev5",
    "frontend_decline_sum_incl5",
    "frontend_max_consec_down_prev5",
    "frontend_range_from_5d_min",
    "brick_rank_pct",
    "delta_rank_pct",
    "decline_rank_pct",
    "range_rank_pct",
    "ret1",
    "ret2",
    "ret3",
    "ret5",
    "ret10",
    "ret1_rank_pct",
    "ret3_rank_pct",
    "ret5_rank_pct",
    "amount_rank_pct",
    "amount20_rank_pct",
    "vol_ratio",
    "vol_ratio_rank_pct",
    "amplitude_pct",
    "amplitude_rank_pct",
    "body_pct",
    "close_pos",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "open_gap_pct",
    "ma5_rel",
    "ma10_rel",
    "ma20_rel",
    "ma50_rel",
    "ma100_rel",
    "trend_ma20_50",
    "trend_ma50_100",
    "day_brick_z",
    "day_delta_z",
    "day_ret1_z",
    "day_amount_z",
    "is_mainboard",
    "is_chinext",
    "is_star",
]


@dataclass
class BaselineResult:
    ok: bool
    root: str
    candidate_path: str
    score_path: str
    config_path: str
    run_dir: str
    summary: dict
    yearly: dict
    position_diagnostic: dict
    trade_audit: dict
    label_diagnostic: dict
    feature_columns: list[str]
    notes: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict OOS frontend-brick non-distilled baseline.")
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default=START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--universe", default="all_a")
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    write_progress(root, "start", vars(args))

    provider = Path(args.provider_uri)
    symbols = load_a_share_symbols(provider, args.start, args.end, universe=args.universe)
    write_progress(root, "symbols_loaded", {"symbols": len(symbols)})
    reader = QlibBinReader(provider)
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"], args.start, args.end)
    quote = normalize_quote(quote)
    risk_names = load_current_risk_names()
    write_progress(root, "quote_loaded", {"rows": int(len(quote)), "risk_names": len(risk_names)})

    panel = build_panel(quote, risk_names)
    write_progress(root, "panel_built", {"rows": int(len(panel)), "candidate_rows": int(panel["candidate"].sum())})
    candidates = add_forward_labels(panel.loc[panel["candidate"]].copy(), panel)
    candidate_path = root / "frontend_candidates_with_labels.parquet"
    candidates.to_parquet(candidate_path, index=False)
    label_diag = summarize_candidates(candidates, args.train_end, args.oos_start)
    write_progress(root, "labels_built", label_diag)

    scored, model_diag = train_and_score(candidates, train_end=args.train_end, oos_start=args.oos_start, topk=args.topk)
    score_path = root / "frontend_oos_scores.parquet"
    scored[["date", "instrument", "score"]].to_parquet(score_path, index=False)
    write_progress(root, "scores_written", {"rows": int(len(scored)), **model_diag})

    config_path = write_backtest_config(root, provider, score_path, args.oos_start, args.end, args.topk, args.max_positions)
    summary = run_backtest(config_path, root)
    run_dir = Path(summary["run_dir"])
    yearly, position_diag = analyze_run(run_dir)
    quote_for_audit = quote.reorder_levels(["datetime", "instrument"]).sort_index()
    trade_audit = audit_trades(run_dir, quote_for_audit, risk_names)

    result = BaselineResult(
        ok=True,
        root=str(root),
        candidate_path=str(candidate_path),
        score_path=str(score_path),
        config_path=str(config_path),
        run_dir=str(run_dir),
        summary=summary,
        yearly=yearly,
        position_diagnostic=position_diag,
        trade_audit=trade_audit,
        label_diagnostic={**label_diag, **model_diag},
        feature_columns=FEATURE_COLUMNS,
        notes=[
            "No public Top5 trades, public score, or public rank are used for training.",
            "Candidate formula uses frontend brick height and delta recovered from the public K-line frontend.",
            "Train labels require signal date and simulated exit date <= train_end; 2025-2026 labels are diagnostic only.",
            "Entry labels penalize next-open one-price limit-up, zero volume/amount, and open gap >=3%; backtest also skips limit-up and open gap >=3%.",
            "Current ST/delisting names are filtered; historical ST is approximated by 5% limit-rate behavior from OHLCV.",
        ],
    )
    out = root / "frontend_oos_baseline_result.json"
    out.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def write_progress(root: Path, step: str, payload: dict) -> None:
    row = {"ts": datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(), "elapsed_epoch": time.time(), "step": step, **payload}
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"progress": step, **payload}, ensure_ascii=False, default=str), flush=True)


def normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    if list(quote.index.names) != ["instrument", "datetime"]:
        quote = quote.reorder_levels(["instrument", "datetime"])
    quote = quote.sort_index()
    for col in quote.columns:
        quote[col] = pd.to_numeric(quote[col], errors="coerce").astype("float32")
    return quote


def load_current_risk_names() -> set[str]:
    risk: set[str] = set()
    if not META.exists():
        return risk
    with sqlite3.connect(META) as conn:
        for symbol, name in conn.execute("select symbol, name from security_master"):
            upper = str(name or "").upper()
            if "ST" in upper or "退" in upper:
                risk.add(normalize_symbol(str(symbol)))
    return risk


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value:
        code, exchange = value.split(".", 1)
        return f"{exchange}{code}"
    return value


def build_panel(quote: pd.DataFrame, risk_names: set[str]) -> pd.DataFrame:
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    for window in (1, 2, 3, 5, 10):
        frame[f"ret{window}"] = grouped["$close"].pct_change(window, fill_method=None)
    for window in (5, 10, 20, 50, 100):
        frame[f"ma{window}"] = grouped["$close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 3)).mean())
        frame[f"ma{window}_rel"] = frame["$close"] / frame[f"ma{window}"] - 1.0
    frame["trend_ma20_50"] = frame["ma20"] / frame["ma50"] - 1.0
    frame["trend_ma50_100"] = frame["ma50"] / frame["ma100"] - 1.0
    frame["amount20"] = grouped["$amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol20"] = grouped["$volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol_ratio"] = frame["$volume"] / frame["vol20"]
    prev_close = grouped["$close"].shift(1)
    frame["amplitude_pct"] = (frame["$high"] - frame["$low"]) / prev_close
    frame["body_pct"] = (frame["$close"] - frame["$open"]) / prev_close
    high_low = (frame["$high"] - frame["$low"]).replace(0, np.nan)
    frame["close_pos"] = (frame["$close"] - frame["$low"]) / high_low
    frame["upper_shadow_pct"] = (frame["$high"] - frame[["$open", "$close"]].max(axis=1)) / prev_close
    frame["lower_shadow_pct"] = (frame[["$open", "$close"]].min(axis=1) - frame["$low"]) / prev_close
    frame["open_gap_pct"] = frame["$open"] / prev_close - 1.0

    for col in ["frontend_brick", "frontend_var2_base", "frontend_var4", "frontend_var5_base"]:
        frame[col] = np.nan
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        brick, var2, var4, var5 = frontend_brick(
            frame.loc[loc, "$high"].to_numpy(dtype="float64"),
            frame.loc[loc, "$low"].to_numpy(dtype="float64"),
            frame.loc[loc, "$close"].to_numpy(dtype="float64"),
        )
        frame.loc[loc, "frontend_brick"] = brick
        frame.loc[loc, "frontend_var2_base"] = var2
        frame.loc[loc, "frontend_var4"] = var4
        frame.loc[loc, "frontend_var5_base"] = var5
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    frame["frontend_prev"] = grouped["frontend_brick"].shift(1)
    frame["frontend_delta"] = frame["frontend_brick"] - frame["frontend_prev"]
    frame["frontend_delta_prev"] = grouped["frontend_delta"].shift(1)
    down = (-frame["frontend_delta"].clip(upper=0)).fillna(0.0)
    frame["frontend_decline_sum_incl5"] = down.groupby(frame["instrument"]).transform(lambda s: s.rolling(5, min_periods=1).sum())
    frame["frontend_decline_sum_prev5"] = down.groupby(frame["instrument"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    frame["frontend_max_consec_down_prev5"] = rolling_max_consec_down(frame)
    frame["frontend_range_from_5d_min"] = frame["frontend_brick"] - grouped["frontend_brick"].transform(lambda s: s.rolling(5, min_periods=1).min())

    for col, name, asc in [
        ("frontend_brick", "brick_rank_pct", True),
        ("frontend_delta", "delta_rank_pct", True),
        ("frontend_decline_sum_prev5", "decline_rank_pct", True),
        ("frontend_range_from_5d_min", "range_rank_pct", True),
        ("ret1", "ret1_rank_pct", True),
        ("ret3", "ret3_rank_pct", True),
        ("ret5", "ret5_rank_pct", True),
        ("$amount", "amount_rank_pct", True),
        ("amount20", "amount20_rank_pct", True),
        ("vol_ratio", "vol_ratio_rank_pct", True),
        ("amplitude_pct", "amplitude_rank_pct", True),
    ]:
        frame[name] = frame.groupby("datetime")[col].rank(pct=True, ascending=asc)
    for col, out in [("frontend_brick", "day_brick_z"), ("frontend_delta", "day_delta_z"), ("ret1", "day_ret1_z"), ("$amount", "day_amount_z")]:
        mean = frame.groupby("datetime")[col].transform("mean")
        std = frame.groupby("datetime")[col].transform("std")
        frame[out] = (frame[col] - mean) / std.replace(0, np.nan)

    inst = frame["instrument"].astype(str)
    frame["is_star"] = inst.str.startswith("SH688").astype(float)
    frame["is_chinext"] = (inst.str.startswith("SZ300") | inst.str.startswith("SZ301")).astype(float)
    frame["is_mainboard"] = ((frame["is_star"] == 0) & (frame["is_chinext"] == 0)).astype(float)
    frame["is_current_risk_name"] = frame["instrument"].isin(risk_names)

    abs_ret = frame.groupby("instrument")["$close"].pct_change(fill_method=None).abs()
    five = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) >= 3
    ) & (
        ten.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) == 0
    )
    one_price_up = (
        (frame["$high"].sub(frame["$low"]).abs() <= 1e-6)
        & (frame["$close"].div(prev_close) - 1.0 >= 0.045)
    )
    core = (frame["frontend_delta_prev"] < 0) & (frame["frontend_delta"] > 0)
    hard_filter = (
        (frame["brick_rank_pct"] >= 0.60)
        & (frame["frontend_delta"] >= 3.0)
        & (frame["amount_rank_pct"] >= 0.50)
    )
    branch = (
        ((frame["ret1_rank_pct"] >= 0.90) & (frame["amount_rank_pct"] >= 0.60) & (frame["frontend_brick"] >= 52.0))
        | hard_filter
    )
    frame["candidate"] = (
        core
        & branch
        & (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
        & (frame["amplitude_pct"] < 0.18)
        & (~frame["is_current_risk_name"].fillna(False))
        & (~frame["st_like_limit_history"].fillna(False))
        & (~one_price_up.fillna(False))
    )
    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def frontend_brick(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    brick = np.zeros(n, dtype="float64")
    var2 = np.zeros(n, dtype="float64")
    var4 = np.zeros(n, dtype="float64")
    var5 = np.zeros(n, dtype="float64")
    prev_var2 = prev_var4 = prev_var5 = None
    for t in range(n):
        start = max(0, t - 3)
        high_window = high[start : t + 1]
        low_window = low[start : t + 1]
        if np.isnan(high_window).all() or np.isnan(low_window).all():
            hh = ll = np.nan
        else:
            hh = np.nanmax(high_window)
            ll = np.nanmin(low_window)
        rng = hh - ll
        if not np.isfinite(rng) or rng == 0 or not np.isfinite(close[t]):
            low_side = high_side = 0.0
        else:
            low_side = (hh - close[t]) / rng * 100.0 - 90.0
            high_side = (close[t] - ll) / rng * 100.0
        prev_var2 = low_side if prev_var2 is None else (low_side + 3.0 * prev_var2) / 4.0
        prev_var4 = high_side if prev_var4 is None else (high_side + 5.0 * prev_var4) / 6.0
        prev_var5 = prev_var4 if prev_var5 is None else (prev_var4 + 5.0 * prev_var5) / 6.0
        var2[t], var4[t], var5[t] = prev_var2, prev_var4, prev_var5
        diff = prev_var5 - prev_var2
        brick[t] = diff - 4.0 if diff > 4.0 else 0.0
    return brick, var2, var4, var5


def rolling_max_consec_down(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float32")
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        delta = frame.loc[loc, "frontend_delta"].to_numpy(dtype="float64")
        down = np.where(np.isfinite(delta), delta < 0, False)
        values = np.zeros(len(delta), dtype="float32")
        for i in range(len(delta)):
            end = i
            start = max(0, end - 5)
            run = best = 0
            for flag in down[start:end]:
                if flag:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            values[i] = best
        out.loc[loc] = values
    return out


def add_forward_labels(candidates: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    panel_by_symbol = {symbol: day.reset_index(drop=True) for symbol, day in panel.groupby("instrument", sort=False)}
    panel_pos = panel.groupby("instrument", sort=False).cumcount()
    candidates = candidates.copy()
    candidates["__panel_pos"] = panel_pos.loc[candidates.index].to_numpy(dtype="int64")
    rows = []
    for record in candidates.to_dict("records"):
        hist = panel_by_symbol[str(record["instrument"])]
        pos = int(record.pop("__panel_pos"))
        if pos < 0 or pos >= len(hist):
            continue
        label = dynamic_exit_label(hist, pos)
        record.update(label)
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["label_return_rank_pct"] = out.groupby("datetime")["forward_return"].rank(pct=True)
    out["label_risk_rank_pct"] = out.groupby("datetime")["risk_adjusted_return"].rank(pct=True)
    out["label_blend"] = 0.65 * out["forward_return"] + 0.20 * out["label_return_rank_pct"] + 0.15 * np.tanh(out["risk_adjusted_return"] / 2.0)
    out["label_blend_rank_pct"] = out.groupby("datetime")["label_blend"].rank(pct=True)
    out["label_top10"] = out["label_blend_rank_pct"] >= 0.90
    out["label_top20"] = out["label_blend_rank_pct"] >= 0.80
    out["date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    return out


def dynamic_exit_label(hist: pd.DataFrame, pos: int) -> dict:
    buy_idx = pos + 1
    if buy_idx >= len(hist):
        return empty_label()
    buy_open = float(hist.loc[buy_idx, "$open"])
    preclose = float(hist.loc[pos, "$close"])
    if not np.isfinite(buy_open) or buy_open <= 0 or not np.isfinite(preclose) or preclose <= 0:
        return empty_label()
    buy_price = buy_open * 1.003
    next_high = float(hist.loc[buy_idx, "$high"])
    next_low = float(hist.loc[buy_idx, "$low"])
    next_close = float(hist.loc[buy_idx, "$close"])
    next_volume = float(hist.loc[buy_idx, "$volume"])
    next_amount = float(hist.loc[buy_idx, "$amount"])
    one_price = abs(next_high - next_low) <= 1e-6 and abs(buy_open - next_close) <= 1e-6 and next_close / preclose - 1.0 >= 0.045
    untradable = one_price or next_volume <= 0 or next_amount <= 0 or buy_open / preclose - 1.0 >= 0.03 - 1e-12
    exit_idx = min(buy_idx + 7, len(hist) - 1)
    reason = "max_hold_7d"
    for e in range(buy_idx + 1, min(buy_idx + 7, len(hist) - 1) + 1):
        decision = e - 1
        holding_days = e - buy_idx
        pnl_close = float(hist.loc[decision, "$close"]) / buy_price - 1.0
        if holding_days >= 1 and pnl_close <= -0.07:
            exit_idx = e
            reason = "stop_loss_7pct"
            break
        if holding_days >= 1 and float(hist.loc[decision, "frontend_delta"]) < 0:
            exit_idx = e
            reason = "brick_delta_down"
            break
    sell_open = float(hist.loc[exit_idx, "$open"])
    ret = sell_open / buy_price - 1.0 if np.isfinite(sell_open) and sell_open > 0 else np.nan
    close_path = hist.loc[buy_idx:exit_idx, "$close"].astype(float)
    mfe = float(close_path.max() / buy_price - 1.0) if len(close_path) else np.nan
    mae = float(close_path.min() / buy_price - 1.0) if len(close_path) else np.nan
    risk_adj = ret / (abs(mae) + 0.02) if np.isfinite(ret) and np.isfinite(mae) else np.nan
    if untradable and np.isfinite(ret):
        ret = min(ret, -0.20)
        risk_adj = min(risk_adj, -5.0) if np.isfinite(risk_adj) else -5.0
    return {
        "buy_date": hist.loc[buy_idx, "datetime"],
        "exit_date": hist.loc[exit_idx, "datetime"],
        "forward_return": ret,
        "mfe_close": mfe,
        "mae_close": mae,
        "risk_adjusted_return": risk_adj,
        "exit_reason_label": reason,
        "entry_one_price_limit_up": bool(one_price),
        "entry_zero_volume_or_amount": bool(next_volume <= 0 or next_amount <= 0),
        "entry_open_gap_ge_3pct": bool(buy_open / preclose - 1.0 >= 0.03 - 1e-12),
    }


def empty_label() -> dict:
    return {
        "buy_date": pd.NaT,
        "exit_date": pd.NaT,
        "forward_return": np.nan,
        "mfe_close": np.nan,
        "mae_close": np.nan,
        "risk_adjusted_return": np.nan,
        "exit_reason_label": "missing",
        "entry_one_price_limit_up": False,
        "entry_zero_volume_or_amount": False,
        "entry_open_gap_ge_3pct": False,
    }


def summarize_candidates(candidates: pd.DataFrame, train_end: str, oos_start: str) -> dict:
    dates = pd.to_datetime(candidates["datetime"])
    exits = pd.to_datetime(candidates["exit_date"])
    train = candidates[(dates <= pd.Timestamp(train_end)) & (exits <= pd.Timestamp(train_end))]
    oos = candidates[dates >= pd.Timestamp(oos_start)]
    return {
        "rows": int(len(candidates)),
        "dates": int(candidates["datetime"].nunique()) if len(candidates) else 0,
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "train_mean_forward_return": safe_mean(train.get("forward_return")),
        "oos_mean_forward_return_diagnostic": safe_mean(oos.get("forward_return")),
        "train_positive_ratio": safe_mean(train["forward_return"] > 0) if len(train) else None,
        "oos_positive_ratio_diagnostic": safe_mean(oos["forward_return"] > 0) if len(oos) else None,
        "entry_untradable_ratios": {
            "one_price_limit_up": safe_mean(candidates.get("entry_one_price_limit_up")),
            "zero_volume_or_amount": safe_mean(candidates.get("entry_zero_volume_or_amount")),
            "open_gap_ge_3pct": safe_mean(candidates.get("entry_open_gap_ge_3pct")),
        },
    }


def train_and_score(candidates: pd.DataFrame, *, train_end: str, oos_start: str, topk: int) -> tuple[pd.DataFrame, dict]:
    data = candidates.dropna(subset=["forward_return", "label_blend_rank_pct", *FEATURE_COLUMNS]).copy()
    dates = pd.to_datetime(data["datetime"])
    exits = pd.to_datetime(data["exit_date"])
    train = data[(dates <= pd.Timestamp(train_end)) & (exits <= pd.Timestamp(train_end))].copy()
    oos = data[dates >= pd.Timestamp(oos_start)].copy()
    if train.empty or oos.empty:
        raise ValueError("empty train or OOS candidates")
    weights = 1.0 + train["label_top20"].astype(float) * 5.0 + train["label_top10"].astype(float) * 8.0 + train["forward_return"].clip(lower=0, upper=0.25) * 20.0
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.08, random_state=20260715),
    )
    model.fit(train[FEATURE_COLUMNS], train["label_blend_rank_pct"], histgradientboostingregressor__sample_weight=weights)
    oos["score"] = model.predict(oos[FEATURE_COLUMNS])
    top = oos.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime", as_index=False).head(topk)
    diag = {
        "train_rows_used": int(len(train)),
        "oos_rows_scored": int(len(oos)),
        "oos_signal_dates": int(oos["datetime"].nunique()),
        "oos_topk_rows": int(len(top)),
        "oos_topk_mean_forward_return_diagnostic": safe_mean(top["forward_return"]),
        "oos_topk_median_forward_return_diagnostic": float(top["forward_return"].median()) if len(top) else None,
        "oos_topk_positive_ratio_diagnostic": safe_mean(top["forward_return"] > 0) if len(top) else None,
        "oos_topk_yearly_mean_forward_return_diagnostic": {str(year): safe_mean(group["forward_return"]) for year, group in top.groupby(pd.to_datetime(top["datetime"]).dt.year)},
    }
    scored = oos.copy()
    scored["date"] = pd.to_datetime(scored["datetime"]).dt.strftime("%Y-%m-%d")
    return scored, diag


def write_backtest_config(root: Path, provider: Path, score_path: Path, start: str, end: str, topk: int, max_positions: int) -> Path:
    config = {
        "name": "frontend_brick_oos_baseline_v1",
        "version": "2026-07-15",
        "description": "Non-distilled frontend brick OOS baseline. Uses external scores trained through 2024.",
        "data": {"provider_uri": str(provider), "universe": "external_score", "start": start, "end": end, "look_back_days": 260},
        "fields": {"open": "$open", "high": "$high", "low": "$low", "close": "$close", "volume": "$volume", "amount": "$amount", "vwap": "$vwap"},
        "factors": {},
        "signals": {},
        "selector": {"mode": "external_score", "path": str(score_path), "date_col": "date", "instrument_col": "instrument", "score_col": "score", "lag": 1, "sort": "score_desc", "topk": topk, "reason": "frontend_brick_oos", "candidate_limit": 30},
        "rebalance": {"type": "equal_weight", "max_positions": max_positions, "buy_only_new_positions": False},
        "execution": {
            "deal_price": "open",
            "cash_use_ratio": 0.98,
            "sell_rules": [
                {"name": "stop_loss_7pct", "when": "holding_days >= 1 and pnl_pct <= -0.07", "action": "sell_all"},
                {"name": "trail_peak", "when": "holding_days >= 2 and peak_pnl_pct >= 0.10 and drawdown_from_peak <= -0.06", "action": "sell_all"},
                {"name": "max_hold_7d", "when": "holding_days >= 7", "action": "sell_all"},
            ],
            "buy": {"sizing": "cash_equal", "lot_size": 100, "skip_if_holding": True, "skip_limit_up": True, "max_open_gap_pct": 0.03, "reuse_sell_cash": True},
        },
        "cost": {"commission_rate": 0.0005, "min_commission": 5.0, "stamp_tax_rate": 0.0001, "stamp_tax_on_buy": False, "transfer_fee_rate": 0.0, "slippage": 0.0, "buy_slippage": 0.003, "sell_slippage": 0.0},
        "engine": {"init_cash": 1000000, "validate_trading_rules": True, "deal_price": "open", "max_workers": 1, "legacy_cost_price": False, "auto_adjust_buy_quantity": True, "precompute_signals": False},
    }
    path = root / "frontend_oos_backtest.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_backtest(config_path: Path, root: Path) -> dict:
    cmd = ["/Users/mingxiaoli/anaconda3/envs/test/bin/python", "-m", "quantx.tools.run_backtest", "--config", str(config_path), "--output-dir", str(root / "runs"), "--run-id", "frontend_brick_oos_top7_maxpos10", "--json"]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    (root / "backtest_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (root / "backtest_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return json.loads(proc.stdout[start:])


def analyze_run(run_dir: Path) -> tuple[dict, dict]:
    nav = pd.DataFrame(json.loads((run_dir / "daily_nav.json").read_text()))
    nav["date"] = pd.to_datetime(nav["date"])
    yearly = {}
    for year, group in nav.groupby(nav["date"].dt.year):
        start_value = float(group.iloc[0]["total_value"])
        end_value = float(group.iloc[-1]["total_value"])
        yearly[str(year)] = {"start_value": start_value, "end_value": end_value, "return": end_value / start_value - 1.0, "days": int(len(group))}
    positions = pd.DataFrame(json.loads((run_dir / "positions.json").read_text()))
    if len(positions):
        positions["date"] = pd.to_datetime(positions["date"])
        pos_count = positions.groupby("date")["symbol"].nunique()
        avg_pos = float(pos_count.mean())
        med_pos = float(pos_count.median())
    else:
        avg_pos = med_pos = 0.0
    closed = pd.DataFrame(json.loads((run_dir / "closed_positions.json").read_text()))
    holding = pd.to_numeric(closed.get("holding_days", pd.Series(dtype=float)), errors="coerce") if len(closed) else pd.Series(dtype=float)
    return yearly, {"avg_position_count": avg_pos, "median_position_count": med_pos, "avg_holding_days": safe_mean(holding), "median_holding_days": float(holding.median()) if len(holding) else 0.0}


def audit_trades(run_dir: Path, quote: pd.DataFrame, risk_names: set[str]) -> dict:
    trades = json.loads((run_dir / "trades.json").read_text())
    buys = [t for t in trades if t.get("action") == "BUY" and not t.get("reject_reason")]
    violations = {"current_st_or_delist_name": 0, "missing_bar": 0, "zero_volume_or_amount": 0, "one_price_limit_up": 0, "open_gap_ge_3pct": 0}
    buy_counts: dict[str, int] = {}
    for trade in buys:
        symbol = str(trade["symbol"])
        date = pd.Timestamp(trade["date"])
        buy_counts[symbol] = buy_counts.get(symbol, 0) + 1
        if symbol in risk_names:
            violations["current_st_or_delist_name"] += 1
        key = (date, symbol)
        if key not in quote.index:
            violations["missing_bar"] += 1
            continue
        row = quote.loc[key]
        if float(row.get("$volume", np.nan) or 0) <= 0 or float(row.get("$amount", np.nan) or 0) <= 0:
            violations["zero_volume_or_amount"] += 1
        try:
            hist = quote.xs(symbol, level="instrument")
            pos = hist.index.get_loc(date)
            preclose = float(hist.iloc[pos - 1]["$close"]) if isinstance(pos, int) and pos > 0 else np.nan
        except Exception:
            preclose = np.nan
        if np.isfinite(preclose) and preclose > 0:
            open_, high, low, close = (float(row[f]) for f in ("$open", "$high", "$low", "$close"))
            if max(abs(high - low), abs(open_ - close)) <= 1e-6 and close / preclose - 1.0 >= 0.045:
                violations["one_price_limit_up"] += 1
            if open_ / preclose - 1.0 >= 0.03 - 1e-12:
                violations["open_gap_ge_3pct"] += 1
    return {"effective_buys": len(buys), "trade_rows": len(trades), "reject_rows": sum(1 for t in trades if t.get("reject_reason")), "violations": violations, "unique_buy_symbols": len(buy_counts), "top_repeat_buys": sorted(buy_counts.items(), key=lambda item: item[1], reverse=True)[:20]}


def safe_mean(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values).astype(float) if not isinstance(values, pd.Series) else pd.to_numeric(values, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.mean()) if len(series) else None


if __name__ == "__main__":
    raise SystemExit(main())
