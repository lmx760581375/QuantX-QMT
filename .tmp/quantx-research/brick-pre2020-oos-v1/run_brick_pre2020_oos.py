"""Brick pre-2020 strict OOS experiment.

Train labels end at 2019-12-31. Scores and account replay start at 2020-01-02.
All artifacts are written under .tmp/quantx-research.
"""

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
from quantx.core.factor_runtime.ops import brick_chart
from quantx.tools.run_backtest import load_a_share_symbols


ARTIFACT_ID = "brick_pre2020_forward_label_oos_v1"
ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
PROVIDER = Path("data/qlib_data_fixed")
META = Path("data/meta/quantx_meta.sqlite")
START = "2010-01-04"
TRAIN_END = "2019-12-31"
OOS_START = "2020-01-02"
END = "2026-07-15"
TOPK = 5
MAX_POSITIONS = 10


FEATURE_COLUMNS = [
    "brick",
    "brick_prev",
    "brick_growth",
    "brick_strength",
    "prior_green_bars",
    "red_risk_7",
    "ret1",
    "ret2",
    "ret3",
    "ret5",
    "ret10",
    "ret1_rank",
    "ret3_rank",
    "ret5_rank",
    "ret10_rank",
    "vol_ratio",
    "vol_rank",
    "amount20_rank",
    "liquidity_rank",
    "low_liq_rank",
    "amplitude",
    "intraday_pos",
    "ma5_rel",
    "ma10_rel",
    "ma20_rel",
    "ma50_rel",
    "ma100_rel",
    "ma150_rel",
    "trend_ma50_100",
    "trend_ma100_150",
    "green_to_red",
    "ret1_x_brick",
    "ret1_x_vol",
    "low_liq_x_brick",
    "price_x_brick",
    "day_brick_z",
    "day_ret1_z",
    "day_vol_z",
    "day_liq_z",
    "day_price_z",
    "is_sh",
    "is_sz",
    "code_bucket",
]


@dataclass
class ExperimentResult:
    ok: bool
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default=START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    write_progress(root, "start", {"artifact_id": ARTIFACT_ID, "train_end": args.train_end, "oos_start": args.oos_start})

    provider = Path(args.provider_uri)
    symbols = load_a_share_symbols(provider, args.start, args.end, universe="all_mainboard")
    write_progress(root, "symbols_loaded", {"symbol_count": len(symbols)})
    reader = QlibBinReader(provider)
    quote = reader.features(
        symbols,
        ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"],
        args.start,
        args.end,
    )
    quote = normalize_quote(quote)
    risk_names = load_current_risk_names()
    write_progress(root, "quote_loaded", {"rows": int(len(quote)), "risk_name_count": len(risk_names)})

    panel = build_panel(quote, risk_names)
    write_progress(root, "panel_built", {"rows": int(len(panel)), "columns": int(len(panel.columns))})

    candidates = panel.loc[panel["candidate"]].copy()
    candidates = add_forward_labels(candidates, panel)
    candidates_path = root / "candidates_with_labels.parquet"
    candidates.to_parquet(candidates_path, index=False)
    write_progress(root, "labels_built", summarize_candidates(candidates, args.train_end, args.oos_start))

    scored, model_diag = train_and_score(candidates, train_end=args.train_end, oos_start=args.oos_start)
    score_path = root / "scores.parquet"
    scored[["date", "instrument", "score"]].to_parquet(score_path, index=False)
    write_progress(root, "scores_written", {"rows": int(len(scored)), "path": str(score_path), **model_diag})

    config_path = write_backtest_config(root, score_path, args.oos_start, args.end)
    summary = run_backtest(config_path, root)
    run_dir = Path(summary["run_dir"])
    quote_for_audit = quote.reorder_levels(["datetime", "instrument"]).sort_index()
    trade_audit = audit_trades(run_dir, quote_for_audit, risk_names)
    yearly, position_diag = analyze_run(run_dir)

    result = ExperimentResult(
        ok=True,
        score_path=str(score_path),
        config_path=str(config_path),
        run_dir=str(run_dir),
        summary=summary,
        yearly=yearly,
        position_diagnostic=position_diag,
        trade_audit=trade_audit,
        label_diagnostic=model_diag,
        feature_columns=FEATURE_COLUMNS,
        notes=[
            "Strict OOS: model labels are trained only from signals whose dynamic-exit exit_date <= 2019-12-31.",
            "OOS account replay starts at 2020-01-02 and uses no OOS forward labels for training or threshold selection.",
            "Selector uses raw daily Top5 scores with no OOS score_floor search.",
            "Current ST/delisting names are excluded from score generation; historical ST is proxied by past 5% limit-rate behavior.",
        ],
    )
    (root / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def write_progress(root: Path, step: str, payload: dict) -> None:
    row = {
        "ts": datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        "elapsed_epoch": time.time(),
        "step": step,
        **payload,
    }
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

    frame["ret1"] = grouped["$close"].pct_change(fill_method=None)
    for window in (2, 3, 5, 10):
        frame[f"ret{window}"] = grouped["$close"].pct_change(window, fill_method=None)
    for window in (5, 10, 20, 50, 100, 150):
        frame[f"ma{window}"] = grouped["$close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 3)).mean())
    frame["amount20"] = grouped["$amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol20"] = grouped["$volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol_ratio"] = frame["$volume"] / frame["vol20"]
    frame["amplitude"] = frame["$high"] / frame["$low"] - 1.0
    frame["intraday_pos"] = (frame["$close"] - frame["$low"]) / (frame["$high"] - frame["$low"]).replace(0, np.nan)
    for window in (5, 10, 20, 50, 100, 150):
        frame[f"ma{window}_rel"] = frame["$close"] / frame[f"ma{window}"] - 1.0
    frame["trend_ma50_100"] = frame["ma50"] / frame["ma100"] - 1.0
    frame["trend_ma100_150"] = frame["ma100"] / frame["ma150"] - 1.0

    frame["brick"] = np.nan
    for symbol, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        frame.loc[loc, "brick"] = brick_chart(
            frame.loc[loc, "$high"].to_numpy(dtype="float32"),
            frame.loc[loc, "$low"].to_numpy(dtype="float32"),
            frame.loc[loc, "$close"].to_numpy(dtype="float32"),
            8, 3, 12, 12, 8, 92, 114, 1, 1, 1,
        )
    frame["brick_prev"] = grouped["brick"].shift(1)
    frame["brick_growth"] = frame["brick"] / (frame["brick_prev"].abs() + 1e-6)
    frame["brick_strength"] = frame["brick"].clip(lower=0)
    frame["green_to_red"] = (frame["brick"] > 0) & (frame["brick_prev"] < 0)
    frame["red_risk"] = frame["brick"] > 0
    frame["red_risk_7"] = grouped["red_risk"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    frame["prior_green_bars"] = prior_consecutive_by_group(frame, "brick", lambda x: x < 0)

    for col in ["ret1", "ret3", "ret5", "ret10", "vol_ratio", "amount20", "$close"]:
        rank_col = {
            "$close": "price_rank_low",
            "amount20": "amount20_rank",
        }.get(col, f"{col}_rank")
        ascending = col == "$close"
        frame[rank_col] = frame.groupby("datetime")[col].rank(pct=True, ascending=ascending)
    frame["vol_rank"] = frame["vol_ratio_rank"]
    frame["liquidity_rank"] = frame["amount20_rank"]
    frame["low_liq_rank"] = 1.0 - frame["amount20_rank"]

    for col, out in [
        ("brick", "day_brick_z"),
        ("ret1", "day_ret1_z"),
        ("vol_ratio", "day_vol_z"),
        ("amount20", "day_liq_z"),
        ("$close", "day_price_z"),
    ]:
        mean = frame.groupby("datetime")[col].transform("mean")
        std = frame.groupby("datetime")[col].transform("std")
        frame[out] = (frame[col] - mean) / std.replace(0, np.nan)

    abs_ret = frame.groupby("instrument")["$close"].pct_change(fill_method=None).abs()
    five = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five.groupby(frame["instrument"]).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True) >= 2
    ) & (
        ten.groupby(frame["instrument"]).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True) == 0
    )

    market = frame.groupby("datetime").agg(active_value=("$amount", "sum"))
    market["active_ret1"] = market["active_value"].pct_change(fill_method=None)
    market["active_ret2"] = market["active_value"].pct_change(2, fill_method=None)
    market["active_ma5"] = market["active_value"].rolling(5, min_periods=3).mean()
    market["active_gate"] = active_value_gate(market)
    frame = frame.merge(market[["active_value", "active_ret1", "active_ret2", "active_gate"]].reset_index(), on="datetime", how="left")

    frame["ret1_x_brick"] = frame["ret1"] * frame["brick_strength"]
    frame["ret1_x_vol"] = frame["ret1"] * frame["vol_ratio"]
    frame["low_liq_x_brick"] = frame["low_liq_rank"] * frame["brick_strength"]
    frame["price_x_brick"] = frame["price_rank_low"] * frame["brick_strength"]
    frame["is_sh"] = frame["instrument"].str.startswith("SH").astype(int)
    frame["is_sz"] = frame["instrument"].str.startswith("SZ").astype(int)
    frame["code_bucket"] = pd.to_numeric(frame["instrument"].str.extract(r"(\d{3})", expand=False), errors="coerce").fillna(0) / 999.0
    frame["is_current_risk_name"] = frame["instrument"].isin(risk_names)

    current_one_price_up = (
        (frame["$high"].sub(frame["$low"]).abs() <= 1e-6)
        & (frame["$close"] / grouped["$close"].shift(1) - 1.0 >= 0.045)
    )
    frame["candidate"] = (
        frame["active_gate"].fillna(False)
        & (frame["brick"] > 0)
        & (frame["brick_prev"] < 1.5)
        & (frame["ret1_rank"] >= 0.35)
        & (frame["vol_rank"] >= 0.15)
        & (frame["amplitude"] < 0.18)
        & (frame["liquidity_rank"] <= 0.95)
        & (~frame["st_like_limit_history"].fillna(False))
        & (~frame["is_current_risk_name"])
        & (~current_one_price_up.fillna(False))
        & (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
    )
    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def prior_consecutive_by_group(frame: pd.DataFrame, col: str, predicate) -> pd.Series:
    out = pd.Series(0, index=frame.index, dtype="int16")
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        values = predicate(frame.loc[loc, col].to_numpy())
        counts = np.zeros(len(values), dtype="int16")
        run = 0
        for i, value in enumerate(values):
            counts[i] = run
            run = run + 1 if bool(value) else 0
        out.loc[loc] = counts
    return out


def active_value_gate(market: pd.DataFrame) -> pd.Series:
    active = []
    in_wave = False
    for row in market.itertuples():
        start = (pd.notna(row.active_ret1) and row.active_ret1 >= 0.04) or (pd.notna(row.active_ret2) and row.active_ret2 >= 0.04)
        if start:
            in_wave = True
        elif in_wave and pd.notna(row.active_ma5) and row.active_value < row.active_ma5:
            in_wave = False
        active.append(in_wave)
    return pd.Series(active, index=market.index)


def add_forward_labels(candidates: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    panel_by_symbol = {symbol: day.reset_index(drop=True) for symbol, day in panel.groupby("instrument", sort=False)}
    rows = []
    for row in candidates.itertuples(index=False):
        hist = panel_by_symbol[str(row.instrument)]
        pos_arr = hist.index[hist["datetime"].eq(row.datetime)].to_numpy()
        if len(pos_arr) == 0:
            continue
        pos = int(pos_arr[0])
        label = dynamic_exit_label(hist, pos)
        record = row._asdict()
        record.update(label)
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["label_return_rank_pct"] = out.groupby("datetime")["forward_return"].rank(pct=True)
    out["label_risk_rank_pct"] = out.groupby("datetime")["risk_adjusted_return"].rank(pct=True)
    out["label_blend"] = 0.75 * out["forward_return"] + 0.25 * np.tanh(out["risk_adjusted_return"] / 2.0)
    out["label_blend_rank_pct"] = out.groupby("datetime")["label_blend"].rank(pct=True)
    out["label_top10"] = out["label_blend_rank_pct"] >= 0.90
    out["label_top20"] = out["label_blend_rank_pct"] >= 0.80
    out["label_rank_blend"] = 0.50 * out["label_blend_rank_pct"] + 0.30 * out["label_return_rank_pct"] + 0.20 * out["label_risk_rank_pct"]
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
    exit_idx = min(buy_idx + 10, len(hist) - 1)
    reason = "max_hold_10d"
    for e in range(buy_idx + 1, min(buy_idx + 10, len(hist) - 1) + 1):
        decision = e - 1
        holding_days = e - buy_idx
        if holding_days >= 1 and float(hist.loc[decision, "brick"]) < 0:
            exit_idx = e
            reason = "green"
            break
        if float(hist.loc[decision, "red_risk_7"]) >= 7:
            exit_idx = e
            reason = "7red"
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
    train = candidates[(pd.to_datetime(candidates["datetime"]) <= pd.Timestamp(train_end)) & (pd.to_datetime(candidates["exit_date"]) <= pd.Timestamp(train_end))]
    oos = candidates[pd.to_datetime(candidates["datetime"]) >= pd.Timestamp(oos_start)]
    return {
        "rows": int(len(candidates)),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "train_mean_forward_return": float(train["forward_return"].mean()) if len(train) else None,
        "oos_mean_forward_return_diagnostic": float(oos["forward_return"].mean()) if len(oos) else None,
        "train_positive_ratio": float((train["forward_return"] > 0).mean()) if len(train) else None,
        "oos_positive_ratio_diagnostic": float((oos["forward_return"] > 0).mean()) if len(oos) else None,
    }


def train_and_score(candidates: pd.DataFrame, *, train_end: str, oos_start: str) -> tuple[pd.DataFrame, dict]:
    candidates = candidates.dropna(subset=["forward_return", "label_rank_blend", *FEATURE_COLUMNS]).copy()
    dates = pd.to_datetime(candidates["datetime"])
    exit_dates = pd.to_datetime(candidates["exit_date"])
    train = candidates[(dates <= pd.Timestamp(train_end)) & (exit_dates <= pd.Timestamp(train_end))].copy()
    oos = candidates[dates >= pd.Timestamp(oos_start)].copy()
    if train.empty or oos.empty:
        raise ValueError("empty train or OOS candidates")
    weights = (
        1.0
        + train["label_top20"].astype(float) * 6.0
        + train["label_top10"].astype(float) * 10.0
        + (train["forward_return"] > 0).astype(float) * 2.0
        + train["forward_return"].clip(lower=0, upper=0.25) * 20.0
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.06,
            random_state=20260715,
        ),
    )
    model.fit(train[FEATURE_COLUMNS], train["label_rank_blend"], histgradientboostingregressor__sample_weight=weights)
    oos["score"] = model.predict(oos[FEATURE_COLUMNS])
    top = oos.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime", as_index=False).head(TOPK)
    diag = {
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "oos_signal_dates": int(oos["datetime"].nunique()),
        "oos_top5_rows": int(len(top)),
        "train_mean_forward_return": float(train["forward_return"].mean()),
        "oos_top5_mean_forward_return_diagnostic": float(top["forward_return"].mean()) if len(top) else None,
        "oos_top5_median_forward_return_diagnostic": float(top["forward_return"].median()) if len(top) else None,
        "oos_top5_positive_ratio_diagnostic": float((top["forward_return"] > 0).mean()) if len(top) else None,
        "oos_top5_yearly_mean_forward_return_diagnostic": {
            str(year): float(group["forward_return"].mean())
            for year, group in top.groupby(pd.to_datetime(top["datetime"]).dt.year)
        },
    }
    scored = oos.rename(columns={"datetime": "date"})
    return scored, diag


def write_backtest_config(root: Path, score_path: Path, start: str, end: str) -> Path:
    config = {
        "name": ARTIFACT_ID,
        "version": "2026-07-15",
        "description": "Strict pre-2020 trained Brick forward-label OOS replay from 2020 onward. No OOS score_floor search.",
        "data": {"provider_uri": str(PROVIDER), "universe": "external_score", "start": start, "end": end, "look_back_days": 720},
        "fields": {"open": "$open", "high": "$high", "low": "$low", "close": "$close", "volume": "$volume", "vwap": "$vwap"},
        "factors": {
            "brick": "BrickChart(high, low, close, 8, 3, 12, 12, 8, 92, 114, 1, 1, 1)",
            "red_risk": "brick > 0",
            "red_risk_7": "Sum(red_risk, 7)",
        },
        "signals": {"green_exit": "brick < 0", "red7_exit": "red_risk_7 >= 7"},
        "selector": {
            "mode": "external_score",
            "path": str(score_path),
            "date_col": "date",
            "instrument_col": "instrument",
            "score_col": "score",
            "lag": 1,
            "sort": "score_desc",
            "topk": TOPK,
            "reason": "brick_pre2020_oos",
            "candidate_limit": 20,
        },
        "rebalance": {"type": "equal_weight", "max_positions": MAX_POSITIONS, "buy_only_new_positions": False},
        "execution": {
            "deal_price": "open",
            "cash_use_ratio": 0.98,
            "sell_rules": [
                {"name": "green", "when": "holding_days >= 1 and green_exit", "action": "sell_all"},
                {"name": "7red", "when": "red7_exit", "action": "sell_all"},
                {"name": "max_hold_10d", "when": "holding_days >= 10", "action": "sell_all"},
            ],
            "buy": {"sizing": "cash_equal", "lot_size": 100, "skip_if_holding": True, "skip_limit_up": True, "max_open_gap_pct": 0.03, "reuse_sell_cash": True},
        },
        "cost": {"commission_rate": 0.0005, "min_commission": 5.0, "stamp_tax_rate": 0.0001, "stamp_tax_on_buy": False, "transfer_fee_rate": 0.0, "slippage": 0.0, "buy_slippage": 0.003, "sell_slippage": 0.0},
        "engine": {"init_cash": 1000000, "validate_trading_rules": True, "deal_price": "open", "max_workers": 1, "legacy_cost_price": False, "auto_adjust_buy_quantity": True, "precompute_signals": False},
    }
    path = root / "backtest_config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_backtest(config_path: Path, root: Path) -> dict:
    cmd = [
        "/Users/mingxiaoli/anaconda3/envs/test/bin/python",
        "-m", "quantx.tools.run_backtest",
        "--config", str(config_path),
        "--output-dir", str(root / "runs"),
        "--run-id", f"{ARTIFACT_ID}_top5_maxpos10",
        "--json",
    ]
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
    nav = nav.sort_values("date")
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
    holding = {
        "avg_holding_days": float(pd.to_numeric(closed.get("holding_days", pd.Series(dtype=float)), errors="coerce").mean()) if len(closed) else 0.0,
        "median_holding_days": float(pd.to_numeric(closed.get("holding_days", pd.Series(dtype=float)), errors="coerce").median()) if len(closed) else 0.0,
    }
    return yearly, {"avg_position_count": avg_pos, "median_position_count": med_pos, **holding}


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
    return {
        "effective_buys": len(buys),
        "trade_rows": len(trades),
        "reject_rows": sum(1 for t in trades if t.get("reject_reason")),
        "violations": violations,
        "unique_buy_symbols": len(buy_counts),
        "top_repeat_buys": sorted(buy_counts.items(), key=lambda item: item[1], reverse=True)[:20],
    }


if __name__ == "__main__":
    raise SystemExit(main())
