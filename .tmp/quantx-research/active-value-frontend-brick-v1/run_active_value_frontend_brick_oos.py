from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_backtest import load_a_share_symbols

ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
PROVIDER = Path("data/qlib_data_fixed")
META = Path("data/meta/quantx_meta.sqlite")
FRONTEND_SCRIPT = Path(".tmp/quantx-research/brick-pre2020-oos-v1/run_frontend_brick_oos_baseline.py")
START = "2013-01-01"
TRAIN_END = "2024-12-31"
OOS_START = "2025-01-02"
END = "2026-07-15"

FEATURE_COLUMNS = [
    "frontend_brick",
    "frontend_prev",
    "frontend_delta",
    "frontend_delta_prev",
    "red_run7",
    "active_ret1",
    "active_ret2",
    "active_days_since_start",
    "ret1",
    "ret2",
    "ret3",
    "ret5",
    "ret10",
    "ret1_rank_pct",
    "ret3_rank_pct",
    "ret5_rank_pct",
    "brick_rank_pct",
    "delta_rank_pct",
    "amount_rank_pct",
    "amount20_rank_pct",
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
class Result:
    ok: bool
    root: str
    candidate_path: str
    score_path: str
    summary: dict
    yearly: dict
    position_diagnostic: dict
    trade_audit: dict
    label_diagnostic: dict
    feature_columns: list[str]
    notes: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Active-value regime plus aligned frontend-brick OOS reproduction.")
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default=START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--universe", default="all_a")
    parser.add_argument("--candidate-mode", choices=["active_start2", "active_wave"], default="active_start2")
    parser.add_argument("--stage", choices=["panel", "labels", "score", "backtest", "all"], default="all")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    write_progress(root, "start", vars(args))

    panel_path = root / f"panel_{args.universe}_{compact_date(args.start)}_{compact_date(args.end)}.parquet"
    candidate_path = root / f"{args.candidate_mode}_candidates_with_labels.parquet"
    score_path = root / f"{args.candidate_mode}_oos_scores.parquet"

    panel = load_or_build_panel(args, root, panel_path)
    if args.stage == "panel":
        report = build_research_report(args, root, panel_path, candidate_path, score_path, panel=panel)
        write_report_files(root, args.candidate_mode, args.stage, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    candidates = load_or_build_candidates(args, root, candidate_path, panel)
    label_diag = summarize_candidates(candidates, args.train_end, args.oos_start)
    rule_diag = summarize_rule_topk(candidates)
    segment_diag = summarize_segments(candidates, args.train_end, args.oos_start)
    write_progress(root, "labels_ready", {**label_diag, "rule_topk_keys": list(rule_diag)})
    if args.stage == "labels":
        report = build_research_report(
            args,
            root,
            panel_path,
            candidate_path,
            score_path,
            panel=panel,
            label_diag=label_diag,
            rule_diag=rule_diag,
            segment_diag=segment_diag,
        )
        write_report_files(root, args.candidate_mode, args.stage, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    scored, model_diag = load_or_build_scores(args, root, score_path, candidates)
    write_progress(root, "scores_ready", {"rows": int(len(scored)), **model_diag})
    if args.stage == "score":
        report = build_research_report(
            args,
            root,
            panel_path,
            candidate_path,
            score_path,
            panel=panel,
            label_diag=label_diag,
            rule_diag=rule_diag,
            segment_diag=segment_diag,
            model_diag=model_diag,
        )
        write_report_files(root, args.candidate_mode, args.stage, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    bt = run_custom_backtest(scored, panel, risk_names=load_current_risk_names(), start=args.oos_start, end=args.end, topk=args.topk, max_positions=args.max_positions)
    result = build_result(args, root, candidate_path, score_path, bt, label_diag, rule_diag, segment_diag, model_diag)
    out = root / f"{args.candidate_mode}_result.json"
    out.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report = build_research_report(
        args,
        root,
        panel_path,
        candidate_path,
        score_path,
        panel=panel,
        label_diag=label_diag,
        rule_diag=rule_diag,
        segment_diag=segment_diag,
        model_diag=model_diag,
        backtest=bt,
    )
    write_report_files(root, args.candidate_mode, args.stage, report)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def load_or_build_panel(args: argparse.Namespace, root: Path, panel_path: Path) -> pd.DataFrame:
    if panel_path.exists() and not args.refresh_cache:
        panel = pd.read_parquet(panel_path)
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        write_progress(root, "panel_cache_loaded", {"path": str(panel_path), "rows": int(len(panel))})
        return panel

    frontend = load_frontend_module()
    provider = Path(args.provider_uri)
    symbols = load_a_share_symbols(provider, args.start, args.end, universe=args.universe)
    reader = QlibBinReader(provider)
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"], args.start, args.end)
    quote = normalize_quote(quote)
    risk_names = load_current_risk_names()
    write_progress(root, "quote_loaded", {"symbols": len(symbols), "rows": int(len(quote)), "risk_names": len(risk_names)})

    panel = build_panel(quote, risk_names, frontend.frontend_brick)
    panel.to_parquet(panel_path, index=False)
    write_progress(root, "panel_built", {
        "path": str(panel_path),
        "rows": int(len(panel)),
        "candidate_active_start2": int(panel["candidate_active_start2"].sum()),
        "candidate_active_wave": int(panel["candidate_active_wave"].sum()),
    })
    return panel


def load_or_build_candidates(args: argparse.Namespace, root: Path, candidate_path: Path, panel: pd.DataFrame) -> pd.DataFrame:
    if candidate_path.exists() and not args.refresh_cache:
        candidates = pd.read_parquet(candidate_path)
        for col in ("datetime", "buy_date", "exit_date"):
            if col in candidates.columns:
                candidates[col] = pd.to_datetime(candidates[col])
        write_progress(root, "candidate_cache_loaded", {"path": str(candidate_path), "rows": int(len(candidates))})
        return candidates

    candidate_col = f"candidate_{args.candidate_mode}"
    candidates = add_forward_labels(panel.loc[panel[candidate_col]].copy(), panel)
    candidates.to_parquet(candidate_path, index=False)
    write_progress(root, "labels_built", {"path": str(candidate_path), "rows": int(len(candidates))})
    return candidates


def load_or_build_scores(args: argparse.Namespace, root: Path, score_path: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if score_path.exists() and not args.refresh_cache:
        scored = pd.read_parquet(score_path)
        if "datetime" not in scored.columns and "date" in scored.columns:
            scored["datetime"] = pd.to_datetime(scored["date"])
        elif "datetime" in scored.columns:
            scored["datetime"] = pd.to_datetime(scored["datetime"])
        write_progress(root, "score_cache_loaded", {"path": str(score_path), "rows": int(len(scored))})
        return scored, summarize_scored_cache(scored, candidates, args.topk)

    scored, model_diag = train_and_score(candidates, args.train_end, args.oos_start, args.topk)
    score_cols = [col for col in ["date", "datetime", "instrument", "score", "rule_score", "forward_return"] if col in scored.columns]
    scored[score_cols].to_parquet(score_path, index=False)
    write_progress(root, "scores_written", {"path": str(score_path), "rows": int(len(scored)), **model_diag})
    return scored, model_diag


def build_result(
    args: argparse.Namespace,
    root: Path,
    candidate_path: Path,
    score_path: Path,
    bt: dict,
    label_diag: dict,
    rule_diag: dict,
    segment_diag: dict,
    model_diag: dict,
) -> Result:
    return Result(
        ok=True,
        root=str(root),
        candidate_path=str(candidate_path),
        score_path=str(score_path),
        summary=bt["summary"],
        yearly=bt["yearly"],
        position_diagnostic=bt["position_diagnostic"],
        trade_audit=bt["trade_audit"],
        label_diagnostic={**label_diag, "rule_topk": rule_diag, "segments": segment_diag, **model_diag},
        feature_columns=FEATURE_COLUMNS,
        notes=[
            "Active-value proxy: market active value=sum($amount); start when 1d or 2d change >=4%; exit when active value falls below MA10 simple moving average.",
            "candidate_mode=active_start2 keeps only the active-value start day and the next two trading days; active_wave keeps the full active wave.",
            "Brick formula is the aligned frontend formula recovered from the public K-line frontend, not QuantX BrickChart delta.",
            "Red brick is frontend_delta > 0; green sell is frontend_delta < 0; 7red sell is seven consecutive red deltas.",
            "No public Top5 trades, public rank, or public score are used for training.",
            "Backtest is a local research account replay because the formal factor engine does not yet expose frontend_brick as a reusable operator.",
        ],
    )


def compact_date(value: str) -> str:
    return str(value).replace("-", "")


def load_frontend_module():
    spec = importlib.util.spec_from_file_location("frontend_brick_oos_baseline", FRONTEND_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FRONTEND_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def build_panel(quote: pd.DataFrame, risk_names: set[str], frontend_brick_func) -> pd.DataFrame:
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    prev_close = grouped["$close"].shift(1)
    for window in (1, 2, 3, 5, 10):
        frame[f"ret{window}"] = grouped["$close"].pct_change(window, fill_method=None)
    for window in (5, 10, 20, 50, 100):
        ma = grouped["$close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 3)).mean())
        frame[f"ma{window}"] = ma
        frame[f"ma{window}_rel"] = frame["$close"] / ma - 1.0
    frame["trend_ma20_50"] = frame["ma20"] / frame["ma50"] - 1.0
    frame["trend_ma50_100"] = frame["ma50"] / frame["ma100"] - 1.0
    frame["amount20"] = grouped["$amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol20"] = grouped["$volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol_ratio"] = frame["$volume"] / frame["vol20"]
    frame["amplitude_pct"] = (frame["$high"] - frame["$low"]) / prev_close
    frame["body_pct"] = (frame["$close"] - frame["$open"]) / prev_close
    high_low = (frame["$high"] - frame["$low"]).replace(0, np.nan)
    frame["close_pos"] = (frame["$close"] - frame["$low"]) / high_low
    frame["open_gap_pct"] = frame["$open"] / prev_close - 1.0

    frame["frontend_brick"] = np.nan
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        brick, _, _, _ = frontend_brick_func(
            frame.loc[loc, "$high"].to_numpy(dtype="float64"),
            frame.loc[loc, "$low"].to_numpy(dtype="float64"),
            frame.loc[loc, "$close"].to_numpy(dtype="float64"),
        )
        frame.loc[loc, "frontend_brick"] = brick
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    frame["frontend_prev"] = grouped["frontend_brick"].shift(1)
    frame["frontend_delta"] = frame["frontend_brick"] - frame["frontend_prev"]
    frame["frontend_delta_prev"] = grouped["frontend_delta"].shift(1)
    frame["red_brick"] = frame["frontend_delta"] > 0
    frame["red_run7"] = grouped["red_brick"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    market = frame.groupby("datetime").agg(active_value=("$amount", "sum"))
    market["active_ret1"] = market["active_value"].pct_change(fill_method=None)
    market["active_ret2"] = market["active_value"].pct_change(2, fill_method=None)
    market["active_ma10"] = market["active_value"].rolling(10, min_periods=5).mean()
    market["active_gate"], market["active_days_since_start"] = active_value_gate(market)
    frame = frame.merge(market[["active_value", "active_ret1", "active_ret2", "active_gate", "active_days_since_start"]].reset_index(), on="datetime", how="left")

    for col, name, asc in [
        ("frontend_brick", "brick_rank_pct", True),
        ("frontend_delta", "delta_rank_pct", True),
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

    abs_ret = grouped["$close"].pct_change(fill_method=None).abs()
    five = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) >= 3
    ) & (ten.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) == 0)
    current_one_price_up = (
        (frame["$high"].sub(frame["$low"]).abs() <= 1e-6)
        & (frame["$close"].div(prev_close) - 1.0 >= 0.045)
    )
    tradable = (
        (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
        & (~frame["is_current_risk_name"].fillna(False))
        & (~frame["st_like_limit_history"].fillna(False))
        & (~current_one_price_up.fillna(False))
    )
    brick_signal = (
        (frame["frontend_delta"] > 0)
        & (frame["ret1"] >= 0.04)
        & (frame["amount_rank_pct"] >= 0.35)
        & (frame["amplitude_pct"] < 0.22)
    )
    frame["candidate_active_start2"] = tradable & brick_signal & frame["active_gate"].fillna(False) & (frame["active_days_since_start"] <= 2)
    frame["candidate_active_wave"] = tradable & brick_signal & frame["active_gate"].fillna(False)
    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def active_value_gate(market: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    active = []
    days = []
    in_wave = False
    day_count = -1
    for row in market.itertuples():
        start = (pd.notna(row.active_ret1) and row.active_ret1 >= 0.04) or (pd.notna(row.active_ret2) and row.active_ret2 >= 0.04)
        if start:
            in_wave = True
            day_count = 0
        elif in_wave and pd.notna(row.active_ma10) and row.active_value < row.active_ma10:
            in_wave = False
            day_count = -1
        elif in_wave:
            day_count += 1
        active.append(in_wave)
        days.append(day_count if in_wave else np.nan)
    return pd.Series(active, index=market.index), pd.Series(days, index=market.index, dtype="float64")


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
        record.update(dynamic_exit_label(hist, pos))
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
    untradable = one_price or next_volume <= 0 or next_amount <= 0
    exit_idx = min(buy_idx + 10, len(hist) - 1)
    reason = "max_hold_10d"
    for e in range(buy_idx + 1, min(buy_idx + 10, len(hist) - 1) + 1):
        decision = e - 1
        holding_days = e - buy_idx
        if holding_days >= 1 and float(hist.loc[decision, "frontend_delta"]) < 0:
            exit_idx = e
            reason = "green"
            break
        if float(hist.loc[decision, "red_run7"]) >= 7:
            exit_idx = e
            reason = "7red"
            break
    sell_open = float(hist.loc[exit_idx, "$open"])
    ret = sell_open / buy_price - 1.0 if np.isfinite(sell_open) and sell_open > 0 else np.nan
    close_path = hist.iloc[buy_idx : exit_idx + 1]["$close"].astype(float)
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
            "open_gap_ge_3pct_diagnostic_only": safe_mean(candidates.get("entry_open_gap_ge_3pct")),
        },
    }


def train_and_score(candidates: pd.DataFrame, train_end: str, oos_start: str, topk: int) -> tuple[pd.DataFrame, dict]:
    data = candidates.dropna(subset=["forward_return", "label_rank_blend", *FEATURE_COLUMNS]).copy()
    dates = pd.to_datetime(data["datetime"])
    exits = pd.to_datetime(data["exit_date"])
    train = data[(dates <= pd.Timestamp(train_end)) & (exits <= pd.Timestamp(train_end))].copy()
    oos = data[dates >= pd.Timestamp(oos_start)].copy()
    if train.empty or oos.empty:
        raise ValueError("empty train or OOS candidates")
    weights = 1.0 + train["label_top20"].astype(float) * 6.0 + train["label_top10"].astype(float) * 10.0 + (train["forward_return"] > 0).astype(float) * 2.0 + train["forward_return"].clip(lower=0, upper=0.25) * 20.0
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.06, random_state=20260715),
    )
    model.fit(train[FEATURE_COLUMNS], train["label_rank_blend"], histgradientboostingregressor__sample_weight=weights)
    oos["score"] = model.predict(oos[FEATURE_COLUMNS])
    oos["rule_score"] = 0.35 * oos["ret1_rank_pct"] + 0.25 * oos["delta_rank_pct"] + 0.20 * oos["brick_rank_pct"] + 0.20 * oos["amount_rank_pct"]
    top = oos.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime", as_index=False).head(topk)
    rule_top = oos.sort_values(["datetime", "rule_score"], ascending=[True, False]).groupby("datetime", as_index=False).head(topk)
    diag = {
        "train_rows_used": int(len(train)),
        "oos_rows_scored": int(len(oos)),
        "oos_signal_dates": int(oos["datetime"].nunique()),
        "model_topk_mean_forward_return_diagnostic": safe_mean(top["forward_return"]),
        "model_topk_positive_ratio_diagnostic": safe_mean(top["forward_return"] > 0),
        "model_topk_yearly_mean_forward_return_diagnostic": {str(year): safe_mean(group["forward_return"]) for year, group in top.groupby(pd.to_datetime(top["datetime"]).dt.year)},
        "rule_topk_mean_forward_return_diagnostic": safe_mean(rule_top["forward_return"]),
        "rule_topk_positive_ratio_diagnostic": safe_mean(rule_top["forward_return"] > 0),
        "rule_topk_yearly_mean_forward_return_diagnostic": {str(year): safe_mean(group["forward_return"]) for year, group in rule_top.groupby(pd.to_datetime(rule_top["datetime"]).dt.year)},
    }
    oos["date"] = pd.to_datetime(oos["datetime"]).dt.strftime("%Y-%m-%d")
    return oos, diag


def summarize_rule_topk(candidates: pd.DataFrame, topks: Iterable[int] = (5, 10, 20)) -> dict:
    data = candidates.dropna(subset=["forward_return", "ret1_rank_pct", "delta_rank_pct", "brick_rank_pct", "amount_rank_pct"]).copy()
    if data.empty:
        return {}
    data["rule_score"] = 0.35 * data["ret1_rank_pct"] + 0.25 * data["delta_rank_pct"] + 0.20 * data["brick_rank_pct"] + 0.20 * data["amount_rank_pct"]
    out = {}
    for topk in topks:
        top = data.sort_values(["datetime", "rule_score"], ascending=[True, False]).groupby("datetime", as_index=False).head(int(topk))
        out[f"top{topk}"] = describe_label_frame(top)
    return out


def summarize_segments(candidates: pd.DataFrame, train_end: str, oos_start: str) -> dict:
    if candidates.empty:
        return {}
    data = candidates.copy()
    dates = pd.to_datetime(data["datetime"])
    exits = pd.to_datetime(data["exit_date"])
    masks = {
        "train": (dates <= pd.Timestamp(train_end)) & (exits <= pd.Timestamp(train_end)),
        "oos": dates >= pd.Timestamp(oos_start),
        "2025": dates.dt.year == 2025,
        "2026": dates.dt.year == 2026,
    }
    segment_cols = {
        "ret1": "ret1",
        "frontend_delta": "frontend_delta",
        "frontend_brick": "frontend_brick",
        "amount_rank_pct": "amount_rank_pct",
    }
    out = {}
    for segment_name, mask in masks.items():
        segment = data.loc[mask].copy()
        out[segment_name] = {"overall": describe_label_frame(segment), "buckets": {}}
        for label, col in segment_cols.items():
            out[segment_name]["buckets"][label] = bucket_summary(segment, col)
    return out


def bucket_summary(frame: pd.DataFrame, col: str, bins: int = 5) -> list[dict]:
    if frame.empty or col not in frame.columns:
        return []
    data = frame[[col, "forward_return"]].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col, "forward_return"])
    if data.empty or data[col].nunique() < 2:
        return []
    try:
        data["bucket"] = pd.qcut(data[col], q=min(int(bins), data[col].nunique()), duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bucket, group in data.groupby("bucket", observed=True):
        rows.append({
            "bucket": str(bucket),
            "rows": int(len(group)),
            "mean_forward_return": safe_mean(group["forward_return"]),
            "median_forward_return": safe_median(group["forward_return"]),
            "positive_ratio": safe_mean(group["forward_return"] > 0),
        })
    return rows


def describe_label_frame(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {"rows": 0, "dates": 0, "mean_forward_return": None, "median_forward_return": None, "positive_ratio": None}
    return {
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()) if "datetime" in frame.columns else 0,
        "mean_forward_return": safe_mean(frame.get("forward_return")),
        "median_forward_return": safe_median(frame.get("forward_return")),
        "positive_ratio": safe_mean(frame["forward_return"] > 0) if "forward_return" in frame.columns else None,
    }


def summarize_scored_cache(scored: pd.DataFrame, candidates: pd.DataFrame, topk: int) -> dict:
    if "forward_return" not in scored.columns:
        keys = ["date", "instrument"]
        enriched = scored.merge(candidates[["date", "instrument", "forward_return"]], on=keys, how="left")
    else:
        enriched = scored
    diag = {"oos_rows_scored": int(len(enriched)), "oos_signal_dates": int(enriched["datetime"].nunique()) if "datetime" in enriched else 0}
    if "score" in enriched.columns:
        top = enriched.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime", as_index=False).head(int(topk))
        diag["model_topk_mean_forward_return_diagnostic"] = safe_mean(top.get("forward_return"))
        diag["model_topk_positive_ratio_diagnostic"] = safe_mean(top["forward_return"] > 0) if "forward_return" in top else None
    if "rule_score" in enriched.columns:
        rule_top = enriched.sort_values(["datetime", "rule_score"], ascending=[True, False]).groupby("datetime", as_index=False).head(int(topk))
        diag["rule_topk_mean_forward_return_diagnostic"] = safe_mean(rule_top.get("forward_return"))
        diag["rule_topk_positive_ratio_diagnostic"] = safe_mean(rule_top["forward_return"] > 0) if "forward_return" in rule_top else None
    return diag


def build_research_report(
    args: argparse.Namespace,
    root: Path,
    panel_path: Path,
    candidate_path: Path,
    score_path: Path,
    *,
    panel: pd.DataFrame | None = None,
    label_diag: dict | None = None,
    rule_diag: dict | None = None,
    segment_diag: dict | None = None,
    model_diag: dict | None = None,
    backtest: dict | None = None,
) -> dict:
    panel_diag = {}
    if panel is not None and not panel.empty:
        panel_diag = {
            "rows": int(len(panel)),
            "dates": int(panel["datetime"].nunique()),
            "candidate_active_start2": int(panel["candidate_active_start2"].sum()) if "candidate_active_start2" in panel else None,
            "candidate_active_wave": int(panel["candidate_active_wave"].sum()) if "candidate_active_wave" in panel else None,
            "active_signal_dates": int(panel.loc[panel.get("active_gate", False).fillna(False), "datetime"].nunique()) if "active_gate" in panel else None,
        }
    return {
        "created_at": datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        "stage": args.stage,
        "candidate_mode": args.candidate_mode,
        "config": {key: value for key, value in vars(args).items() if key not in {"refresh_cache"}},
        "paths": {
            "root": str(root),
            "panel": str(panel_path),
            "candidates": str(candidate_path),
            "scores": str(score_path),
        },
        "panel": panel_diag,
        "labels": label_diag or {},
        "rule_topk": rule_diag or {},
        "segments": segment_diag or {},
        "model": model_diag or {},
        "backtest": backtest or {},
        "notes": [
            "Active-value proxy is sum($amount), used as 0AMV close approximation until true 0AMV OHLC is available.",
            "active_start2 means active-value start day through the next two trading days.",
            "Rule score is 0.35*ret1_rank + 0.25*frontend_delta_rank + 0.20*frontend_brick_rank + 0.20*amount_rank.",
            "Label returns use T signal, T+1 open buy with dynamic frontend-brick exits.",
        ],
    }


def write_report_files(root: Path, candidate_mode: str, stage: str, report: dict) -> None:
    json_path = root / f"{candidate_mode}_{stage}_diagnostic.json"
    md_path = root / f"{candidate_mode}_{stage}_diagnostic.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict) -> str:
    lines = [
        f"# Active Value Frontend Brick Diagnostic ({report.get('candidate_mode')}, {report.get('stage')})",
        "",
        f"Created: {report.get('created_at')}",
        "",
        "## Panel",
        "",
    ]
    for key, value in (report.get("panel") or {}).items():
        lines.append(f"- {key}: {fmt_value(value)}")
    lines.extend(["", "## Labels", ""])
    for key, value in (report.get("labels") or {}).items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {fmt_value(value)}")
    entry = (report.get("labels") or {}).get("entry_untradable_ratios") or {}
    if entry:
        lines.append("- entry_untradable_ratios: " + ", ".join(f"{k}={fmt_value(v)}" for k, v in entry.items()))
    lines.extend(["", "## Rule TopK", ""])
    for key, value in (report.get("rule_topk") or {}).items():
        lines.append(f"- {key}: rows={value.get('rows')}, dates={value.get('dates')}, mean={fmt_value(value.get('mean_forward_return'))}, median={fmt_value(value.get('median_forward_return'))}, win={fmt_value(value.get('positive_ratio'))}")
    model = report.get("model") or {}
    if model:
        lines.extend(["", "## Model", ""])
        for key, value in model.items():
            lines.append(f"- {key}: {fmt_value(value)}")
    bt = report.get("backtest") or {}
    if bt:
        lines.extend(["", "## Backtest", ""])
        for section in ("summary", "position_diagnostic", "trade_audit"):
            if section in bt:
                lines.append(f"### {section}")
                for key, value in bt[section].items():
                    lines.append(f"- {key}: {fmt_value(value)}")
    lines.extend(["", "## Segment Buckets", ""])
    for segment, payload in (report.get("segments") or {}).items():
        overall = payload.get("overall", {})
        lines.append(f"### {segment}: rows={overall.get('rows')}, mean={fmt_value(overall.get('mean_forward_return'))}, win={fmt_value(overall.get('positive_ratio'))}")
        for bucket_name, rows in (payload.get("buckets") or {}).items():
            lines.append(f"- {bucket_name}")
            for row in rows:
                lines.append(f"  - {row['bucket']}: rows={row['rows']}, mean={fmt_value(row['mean_forward_return'])}, median={fmt_value(row['median_forward_return'])}, win={fmt_value(row['positive_ratio'])}")
    lines.append("")
    return "\n".join(lines)


def fmt_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_custom_backtest(scored: pd.DataFrame, panel: pd.DataFrame, risk_names: set[str], start: str, end: str, topk: int, max_positions: int) -> dict:
    panel_by_symbol = {symbol: day.reset_index(drop=True) for symbol, day in panel.groupby("instrument", sort=False)}
    date_arrays = {symbol: pd.to_datetime(day["datetime"]).to_numpy() for symbol, day in panel_by_symbol.items()}
    calendar = pd.DatetimeIndex(sorted(panel[(panel["datetime"] >= pd.Timestamp(start)) & (panel["datetime"] <= pd.Timestamp(end))]["datetime"].dropna().unique()))
    score_by_date = {pd.Timestamp(k): g.sort_values("score", ascending=False) for k, g in scored.groupby(pd.to_datetime(scored["datetime"]))}
    cash = 1_000_000.0
    positions: dict[str, dict] = {}
    trades = []
    nav_rows = []
    position_rows = []
    violations = {"current_st_or_delist_name": 0, "missing_bar": 0, "zero_volume_or_amount": 0, "one_price_limit_up": 0}
    for date in calendar:
        # Sell at today's open using yesterday's known frontend-brick state.
        for symbol in list(positions):
            hist = panel_by_symbol.get(symbol)
            pos = find_pos(date_arrays, symbol, date)
            if hist is None or pos is None or pos <= positions[symbol]["buy_pos"]:
                continue
            decision = pos - 1
            holding_days = pos - positions[symbol]["buy_pos"]
            reason = None
            if holding_days >= 1 and float(hist.loc[decision, "frontend_delta"]) < 0:
                reason = "green"
            elif float(hist.loc[decision, "red_run7"]) >= 7:
                reason = "7red"
            elif holding_days >= 10:
                reason = "max_hold_10d"
            if reason:
                open_price = float(hist.loc[pos, "$open"])
                if np.isfinite(open_price) and open_price > 0:
                    qty = positions[symbol]["qty"]
                    gross = qty * open_price
                    fee = max(gross * 0.0005, 5.0) + gross * 0.0001
                    cash += gross - fee
                    trades.append({"date": str(date.date()), "action": "SELL", "symbol": symbol, "price": open_price, "qty": qty, "reason": reason, "fee": fee})
                    del positions[symbol]

        signal_idx = calendar.get_loc(date) - 1
        if signal_idx >= 0:
            signal_date = calendar[signal_idx]
            picks = score_by_date.get(signal_date)
            if picks is not None:
                slots = max_positions - len(positions)
                buy_candidates = [row for row in picks.itertuples(index=False) if str(row.instrument) not in positions][: max(topk * 3, topk)]
                buy_list = []
                for row in buy_candidates:
                    if len(buy_list) >= min(topk, slots):
                        break
                    symbol = str(row.instrument)
                    if symbol in risk_names:
                        violations["current_st_or_delist_name"] += 1
                        continue
                    hist = panel_by_symbol.get(symbol)
                    pos = find_pos(date_arrays, symbol, date)
                    if hist is None or pos is None or pos <= 0:
                        violations["missing_bar"] += 1
                        continue
                    bar = hist.loc[pos]
                    preclose = float(hist.loc[pos - 1, "$close"])
                    open_price = float(bar["$open"])
                    high = float(bar["$high"])
                    low = float(bar["$low"])
                    close = float(bar["$close"])
                    volume = float(bar["$volume"])
                    amount = float(bar["$amount"])
                    if volume <= 0 or amount <= 0 or not np.isfinite(open_price) or open_price <= 0:
                        violations["zero_volume_or_amount"] += 1
                        continue
                    if abs(high - low) <= 1e-6 and abs(open_price - close) <= 1e-6 and preclose > 0 and close / preclose - 1.0 >= 0.045:
                        violations["one_price_limit_up"] += 1
                        continue
                    buy_list.append((symbol, pos, open_price))
                if buy_list and cash > 0:
                    per_cash = cash * 0.98 / len(buy_list)
                    for symbol, pos, open_price in buy_list:
                        buy_price = open_price * 1.003
                        qty = int(per_cash / buy_price / 100) * 100
                        if qty <= 0:
                            continue
                        gross = qty * buy_price
                        fee = max(gross * 0.0005, 5.0)
                        if cash < gross + fee:
                            continue
                        cash -= gross + fee
                        positions[symbol] = {"qty": qty, "buy_price": buy_price, "buy_date": date, "buy_pos": pos}
                        trades.append({"date": str(date.date()), "action": "BUY", "symbol": symbol, "price": buy_price, "qty": qty, "reason": "active_value_frontend_brick", "fee": fee})

        total = cash
        for symbol, posn in positions.items():
            hist = panel_by_symbol.get(symbol)
            pos = find_pos(date_arrays, symbol, date)
            if hist is not None and pos is not None:
                close = float(hist.loc[pos, "$close"])
                if np.isfinite(close) and close > 0:
                    total += posn["qty"] * close
                    position_rows.append({"date": str(date.date()), "symbol": symbol, "qty": posn["qty"], "market_value": posn["qty"] * close})
        nav_rows.append({"date": str(date.date()), "total_value": total, "cash": cash, "positions": len(positions)})
    nav = pd.DataFrame(nav_rows)
    final_value = float(nav.iloc[-1]["total_value"]) if len(nav) else 1_000_000.0
    nav["cummax"] = nav["total_value"].cummax()
    nav["drawdown"] = nav["total_value"] / nav["cummax"] - 1.0
    yearly = {}
    nav["year"] = pd.to_datetime(nav["date"]).dt.year
    for year, group in nav.groupby("year"):
        yearly[str(year)] = {"start_value": float(group.iloc[0]["total_value"]), "end_value": float(group.iloc[-1]["total_value"]), "return": float(group.iloc[-1]["total_value"] / group.iloc[0]["total_value"] - 1.0), "days": int(len(group))}
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]
    pos_counts = pd.DataFrame(position_rows).groupby("date")["symbol"].nunique() if position_rows else pd.Series(dtype=float)
    return {
        "summary": {"final_value": final_value, "total_return": final_value / 1_000_000.0 - 1.0, "max_drawdown": float(nav["drawdown"].min()) if len(nav) else 0.0, "trades": len(trades), "buys": len(buys), "sells": len(sells), "final_cash": cash, "final_positions": len(positions)},
        "yearly": yearly,
        "position_diagnostic": {"avg_position_count": float(pos_counts.mean()) if len(pos_counts) else 0.0, "median_position_count": float(pos_counts.median()) if len(pos_counts) else 0.0},
        "trade_audit": {"violations": violations, "effective_buys": len(buys), "sell_reasons": pd.Series([t["reason"] for t in sells]).value_counts().to_dict() if sells else {}},
    }


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


if __name__ == "__main__":
    raise SystemExit(main())
