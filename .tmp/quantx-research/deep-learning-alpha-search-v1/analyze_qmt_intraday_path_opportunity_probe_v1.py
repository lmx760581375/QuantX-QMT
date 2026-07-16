from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "qmt_intraday_path_opportunity_probe_v1_summary.json"

START = "2026-01-01"
END = "2026-07-10"
PERIOD = "5m"
POOL_SIZE = 500
FETCH_TOP = 60
TOP_K = 20
INTERVAL = 5
SEED = 20260714

FEATURE_DIRECTIONS = {
    "intraday_return": True,
    "close_pos_range": True,
    "vwap_support": True,
    "low_to_close_recovery": True,
    "high_to_close_fade": True,
    "tail_return": True,
    "tail_volume_share": True,
    "second_half_return": True,
    "first_half_return": True,
    "drawdown_from_high": True,
    "range_amp": False,
    "volume_concentration_top20pct": False,
    "morning_volume_share": False,
    "late_volume_share": True,
    "smooth_path_score": True,
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CA52 = load_module("crowded_trend_for_intraday_path_probe", ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py")
LIM = load_module("qmt_limit_order_minute_for_intraday_path_probe", ROOT / "analyze_qmt_minute_limit_order_event_world_model_v1.py")
RT = CA52.RT
POS = CA52.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def future_return(market: dict[str, Any], idx: int, col: int) -> float | None:
    return CA52.future_return(market, idx, col, INTERVAL)


def intraday_path_features(frame: pd.DataFrame, preclose: float) -> dict[str, float]:
    if frame.empty or not np.isfinite(preclose) or preclose <= 0:
        return {"minute_available": 0.0}
    frame = frame.sort_values("datetime")
    close = pd.to_numeric(frame.get("close"), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high"), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low"), errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(frame.get("open"), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    amount = pd.to_numeric(frame.get("amount"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(close)
    if n < 8 or not np.isfinite(open_[0]) or open_[0] <= 0 or not np.isfinite(close[-1]):
        return {"minute_available": 0.0}
    total_vol = float(np.nansum(volume))
    total_amt = float(np.nansum(amount))
    vwap = total_amt / total_vol / 100.0 if total_vol > 0 and total_amt > 0 else float(np.nanmean(close))
    day_high = float(np.nanmax(high)) if np.isfinite(high).any() else np.nan
    day_low = float(np.nanmin(low)) if np.isfinite(low).any() else np.nan
    last = float(close[-1])
    mid = max(1, n // 2)
    tail_n = min(6, n)
    top_count = max(1, int(math.ceil(n * 0.2)))
    running_high = np.maximum.accumulate(np.where(np.isfinite(high), high, np.nan))
    dd = np.divide(low, running_high, out=np.full_like(low, np.nan), where=np.isfinite(low) & np.isfinite(running_high) & (running_high > 0)) - 1.0
    returns = np.diff(np.log(np.where(np.isfinite(close) & (close > 0), close, np.nan)))
    smooth = 0.0
    if np.isfinite(returns).sum() > 3:
        smooth = float(np.nanmean(returns) / (np.nanstd(returns) + 1e-6))
    return {
        "minute_available": 1.0,
        "intraday_return": float(last / open_[0] - 1.0),
        "close_return": float(last / preclose - 1.0),
        "first_half_return": float(close[mid - 1] / open_[0] - 1.0) if np.isfinite(close[mid - 1]) else 0.0,
        "second_half_return": float(last / close[mid - 1] - 1.0) if np.isfinite(close[mid - 1]) and close[mid - 1] > 0 else 0.0,
        "tail_return": float(last / close[-tail_n] - 1.0) if np.isfinite(close[-tail_n]) and close[-tail_n] > 0 else 0.0,
        "range_amp": float(day_high / day_low - 1.0) if np.isfinite(day_high) and np.isfinite(day_low) and day_low > 0 else 0.0,
        "close_pos_range": float((last - day_low) / (day_high - day_low)) if np.isfinite(day_high) and np.isfinite(day_low) and day_high > day_low else 0.5,
        "vwap_support": float(last / vwap - 1.0) if np.isfinite(vwap) and vwap > 0 else 0.0,
        "low_to_close_recovery": float(last / day_low - 1.0) if np.isfinite(day_low) and day_low > 0 else 0.0,
        "high_to_close_fade": float(last / day_high - 1.0) if np.isfinite(day_high) and day_high > 0 else 0.0,
        "drawdown_from_high": float(np.nanmin(dd)) if np.isfinite(dd).any() else 0.0,
        "tail_volume_share": float(np.nansum(volume[-tail_n:]) / total_vol) if total_vol > 0 else 0.0,
        "morning_volume_share": float(np.nansum(volume[:mid]) / total_vol) if total_vol > 0 else 0.0,
        "late_volume_share": float(np.nansum(volume[mid:]) / total_vol) if total_vol > 0 else 0.0,
        "volume_concentration_top20pct": float(np.nansum(np.sort(volume)[-top_count:]) / total_vol) if total_vol > 0 else 0.0,
        "smooth_path_score": smooth,
    }


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    fetch = {"requested": 0, "available": 0, "empty": 0, "errors": 0, "cached": 0}
    with LIM.QMTClient(max_retries=1, fill_data=False) as client:
        for session in CA52.CVR.sessions(market, START, END, INTERVAL):
            idx = market["date_index"][session]
            cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
            labels = []
            kept_cols = []
            for col in cols:
                y = future_return(market, idx, int(col))
                if y is None:
                    continue
                labels.append(y)
                kept_cols.append(int(col))
            if len(labels) < TOP_K:
                continue
            labels_arr = np.asarray(labels, dtype=float)
            kept = np.asarray(kept_cols, dtype=int)
            scores = CA52.candidate_scores(CA52.feature_frame(market, idx, kept))
            score = scores["mid_trend_volume_not_extreme"]
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(score[j]), str(market["symbols"][kept[j]])))[:FETCH_TOP]
            for rank, j in enumerate(order, 1):
                col = int(kept[j])
                symbol = market["symbols"][col]
                preclose = float(market["arrays"]["close"][idx - 1, col])
                try:
                    path = LIM.cache_path(symbol, session, PERIOD)
                    if path.exists():
                        fetch["cached"] += 1
                    fetch["requested"] += 1
                    minute = LIM.load_or_fetch_minute(client, symbol, session, PERIOD)
                    day_frame = minute[minute["date"] == session] if not minute.empty and "date" in minute.columns else minute
                    feats = intraday_path_features(day_frame, preclose)
                    if feats.get("minute_available", 0.0) > 0:
                        fetch["available"] += 1
                    else:
                        fetch["empty"] += 1
                except Exception:
                    fetch["errors"] += 1
                    feats = {"minute_available": 0.0}
                rows.append({
                    "date": session,
                    "symbol": symbol,
                    "rank_in_candidate": rank,
                    "base_score": float(score[j]),
                    "future_return": float(labels_arr[j]),
                    "features": feats,
                })
    return rows, fetch


def session_feature_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["features"].get("minute_available", 0.0) > 0:
            by_session.setdefault(row["date"], []).append(row)
    out: dict[str, Any] = {}
    for feature, high_good in FEATURE_DIRECTIONS.items():
        top_returns = []
        bottom_returns = []
        rank_ics = []
        for session_rows in by_session.values():
            if len(session_rows) < TOP_K:
                continue
            ordered = sorted(session_rows, key=lambda row: (float(row["features"].get(feature, 0.0)), row["symbol"]), reverse=high_good)
            top_returns.append(safe_mean([row["future_return"] for row in ordered[:TOP_K]]))
            bottom_returns.append(safe_mean([row["future_return"] for row in ordered[-TOP_K:]]))
            values = np.asarray([row["features"].get(feature, 0.0) for row in session_rows], dtype=float)
            labels = np.asarray([row["future_return"] for row in session_rows], dtype=float)
            rank_ics.append(OP_rank_ic(values if high_good else -values, labels))
        out[feature] = {
            "sessions": len(top_returns),
            "top20_mean": safe_mean(top_returns),
            "bottom20_mean": safe_mean(bottom_returns),
            "spread": safe_mean(top_returns) - safe_mean(bottom_returns),
            "mean_rank_ic": safe_mean(rank_ics),
            "high_good": high_good,
        }
    return dict(sorted(out.items(), key=lambda item: -item[1]["top20_mean"]))


def OP_rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    x = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def composite_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["features"].get("minute_available", 0.0) > 0:
            by_session.setdefault(row["date"], []).append(row)
    variants = {
        "tail_vwap_recovery": {"tail_return": 0.30, "vwap_support": 0.30, "close_pos_range": 0.20, "low_to_close_recovery": 0.20},
        "smooth_late_accumulation": {"smooth_path_score": 0.25, "second_half_return": 0.30, "late_volume_share": 0.20, "volume_concentration_top20pct": -0.25},
        "anti_intraday_exhaustion": {"high_to_close_fade": 0.35, "drawdown_from_high": 0.25, "range_amp": -0.20, "volume_concentration_top20pct": -0.20},
        "intraday_path_composite": {"tail_return": 0.20, "vwap_support": 0.20, "close_pos_range": 0.20, "second_half_return": 0.20, "volume_concentration_top20pct": -0.20},
    }
    out = {}
    for name, weights in variants.items():
        top_returns = []
        rank_ics = []
        for session_rows in by_session.values():
            if len(session_rows) < TOP_K:
                continue
            feature_scores = []
            for feature, weight in weights.items():
                values = pd.Series([row["features"].get(feature, 0.0) for row in session_rows]).rank(pct=True).fillna(0.5).to_numpy(dtype=float) - 0.5
                feature_scores.append(weight * values)
            score = np.sum(np.vstack(feature_scores), axis=0)
            order = np.argsort(-score, kind="mergesort")[:TOP_K]
            labels = np.asarray([row["future_return"] for row in session_rows], dtype=float)
            top_returns.append(safe_mean(labels[order]))
            rank_ics.append(OP_rank_ic(score, labels))
        out[name] = {"sessions": len(top_returns), "top20_mean": safe_mean(top_returns), "mean_rank_ic": safe_mean(rank_ics), "weights": weights}
    return dict(sorted(out.items(), key=lambda item: -item[1]["top20_mean"]))


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    rows, fetch = build_rows(market, amount20_arr)
    available = [row for row in rows if row["features"].get("minute_available", 0.0) > 0]
    baseline_by_session: dict[str, list[float]] = {}
    for row in rows:
        if row["rank_in_candidate"] <= TOP_K:
            baseline_by_session.setdefault(row["date"], []).append(row["future_return"])
    baseline_top20 = safe_mean([safe_mean(values) for values in baseline_by_session.values()])
    feature_scan = session_feature_scan(rows)
    composite = composite_scan(rows)
    best_feature = next(iter(feature_scan.items())) if feature_scan else (None, {})
    best_composite = next(iter(composite.items())) if composite else (None, {})
    verdict = "qmt_intraday_path_opportunity_probe_not_enough"
    if best_composite[1].get("top20_mean", 0.0) > baseline_top20 + 0.01 and best_composite[1].get("mean_rank_ic", 0.0) > 0.05:
        verdict = "qmt_intraday_path_candidate_needs_dev_expansion"
    out = {
        "experiment": "qmt_intraday_path_opportunity_probe_v1",
        "method": "qmt_5m_t_day_path_probe_inside_mid_trend_volume_not_extreme_2026_candidates",
        "params": {"start": START, "end": END, "period": PERIOD, "pool_size": POOL_SIZE, "fetch_top": FETCH_TOP, "top_k": TOP_K, "interval": INTERVAL, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round57_summary": sha256(ROOT / "market_phase_contrastive_world_model_v1_summary.json")},
        "fetch": fetch,
        "rows": len(rows),
        "available_rows": len(available),
        "sessions": len({row["date"] for row in rows}),
        "available_sessions": len({row["date"] for row in available}),
        "baseline_mid_trend_volume_top20_label": baseline_top20,
        "feature_scan": feature_scan,
        "composite_scan": composite,
        "best_feature": {"name": best_feature[0], "metrics": best_feature[1]},
        "best_composite": {"name": best_composite[0], "metrics": best_composite[1]},
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "fetch": fetch, "rows": len(rows), "available_rows": len(available), "baseline": baseline_top20, "best_feature": out["best_feature"], "best_composite": out["best_composite"], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
