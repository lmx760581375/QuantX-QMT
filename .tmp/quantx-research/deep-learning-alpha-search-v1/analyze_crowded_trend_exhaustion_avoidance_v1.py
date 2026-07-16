from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "crowded_trend_exhaustion_avoidance_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
POOL_SIZE = 500
TOP_KS = [10, 20]
INTERVALS = [3, 5]

CANDIDATES = [
    "mid_trend_not_extreme",
    "mid_trend_volume_not_extreme",
    "mid_trend_pullback_recover",
    "mid_trend_low_crowding",
    "mid_trend_vwap_support",
    "avoid_extreme_trend_low_range",
    "avoid_extreme_trend_composite",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_crowded_avoidance", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def finite_rank(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(clean[np.isfinite(clean)])) if np.isfinite(clean).any() else 0.0
    order = np.argsort(np.where(np.isfinite(clean), clean, fill), kind="mergesort")
    ranks = np.empty(len(clean), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(clean), endpoint=True) if len(clean) > 1 else 0.5
    return ranks


def centered_rank(values: np.ndarray) -> np.ndarray:
    return finite_rank(values) - 0.5


def mid_preference(rank01: np.ndarray, center: float = 0.68, width: float = 0.22) -> np.ndarray:
    return -np.abs(rank01 - center) / max(width, 1e-6)


def window(arr: np.ndarray, end_idx: int, cols: np.ndarray, length: int) -> np.ndarray:
    start = max(0, end_idx - length + 1)
    return arr[start:end_idx + 1, :][:, cols].astype(float)


def future_return(market: dict[str, Any], idx: int, col: int, interval: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + interval + 1
    if exit_idx >= len(market["dates"]):
        return None
    arrays = market["arrays"]
    entry = float(arrays["open"][entry_idx, col])
    exit_ = float(arrays["open"][exit_idx, col])
    close_t = float(arrays["close"][idx, col])
    if not np.isfinite(entry) or not np.isfinite(exit_) or not np.isfinite(close_t) or entry <= 0 or exit_ <= 0 or close_t <= 0:
        return None
    if abs(entry / close_t - 1.0) > 0.095:
        return None
    return exit_ / entry - 1.0


def feature_frame(market: dict[str, Any], idx: int, cols: np.ndarray) -> dict[str, np.ndarray]:
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    high = arrays["high"].astype(float)
    low = arrays["low"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    c0 = close[idx, cols]
    c5 = close[idx - 5, cols] if idx >= 5 else np.full(len(cols), np.nan)
    c20 = close[idx - 20, cols] if idx >= 20 else np.full(len(cols), np.nan)
    c60 = close[idx - 60, cols] if idx >= 60 else np.full(len(cols), np.nan)
    ret5 = safe_div(c0, c5) - 1.0
    ret20 = safe_div(c0, c20) - 1.0
    ret60 = safe_div(c0, c60) - 1.0
    high5 = window(high, idx, cols, 5)
    low5 = window(low, idx, cols, 5)
    high20 = window(high, idx, cols, 20)
    low20 = window(low, idx, cols, 20)
    high20_max = np.nanmax(high20, axis=0)
    low5_min = np.nanmin(low5, axis=0)
    high5_max = np.nanmax(high5, axis=0)
    range5 = np.nanmean(safe_div(high5, low5) - 1.0, axis=0)
    range20 = np.nanmean(safe_div(high20, low20) - 1.0, axis=0)
    vol5 = np.nanmean(window(volume, idx, cols, 5), axis=0)
    vol20 = np.nanmean(window(volume, idx, cols, 20), axis=0)
    vol60 = np.nanmean(window(volume, idx, cols, 60), axis=0)
    vwap5 = np.nanmean(window(vwap, idx, cols, 5), axis=0)
    vwap20 = np.nanmean(window(vwap, idx, cols, 20), axis=0)
    amount = np.where(np.isfinite(vwap[idx, cols]) & (vwap[idx, cols] > 0), vwap[idx, cols], c0) * volume[idx, cols]
    return {
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "vol5_vs_20": np.log(np.where(safe_div(vol5, vol20) > 0, safe_div(vol5, vol20), np.nan)),
        "vol20_vs_60": np.log(np.where(safe_div(vol20, vol60) > 0, safe_div(vol20, vol60), np.nan)),
        "range5": range5,
        "range20": range20,
        "near_high20": safe_div(c0, high20_max) - 1.0,
        "close_vwap5": safe_div(c0, vwap5) - 1.0,
        "close_vwap20": safe_div(c0, vwap20) - 1.0,
        "low_to_close5": safe_div(c0, low5_min) - 1.0,
        "upper_shadow5": safe_div(high5_max, c0) - 1.0,
        "amount_rank": finite_rank(amount),
    }


def candidate_scores(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rank = {name: finite_rank(values) for name, values in frame.items()}
    cr = {name: values - 0.5 for name, values in rank.items()}
    extreme_trend = np.maximum.reduce([rank["ret60"], rank["ret20"], rank["vol20_vs_60"], rank["vol5_vs_20"]])
    exhaustion = 0.30 * rank["ret60"] + 0.25 * rank["ret20"] + 0.20 * rank["vol20_vs_60"] + 0.15 * rank["vol5_vs_20"] + 0.10 * rank["close_vwap20"]
    mid_trend = 0.55 * mid_preference(rank["ret60"], 0.68) + 0.45 * mid_preference(rank["ret20"], 0.66)
    return {
        "mid_trend_not_extreme": mid_trend - 0.45 * rank["vol5_vs_20"] - 0.25 * rank["close_vwap20"],
        "mid_trend_volume_not_extreme": mid_trend + 0.25 * mid_preference(rank["vol20_vs_60"], 0.62) - 0.35 * rank["vol5_vs_20"],
        "mid_trend_pullback_recover": mid_trend - 0.25 * cr["ret5"] + 0.25 * cr["low_to_close5"] - 0.25 * rank["upper_shadow5"],
        "mid_trend_low_crowding": mid_trend - 0.35 * exhaustion - 0.20 * rank["amount_rank"],
        "mid_trend_vwap_support": mid_trend + 0.25 * cr["close_vwap20"] - 0.30 * rank["vol5_vs_20"] - 0.20 * rank["range20"],
        "avoid_extreme_trend_low_range": 0.55 * mid_trend - 0.30 * extreme_trend - 0.25 * rank["range20"],
        "avoid_extreme_trend_composite": 0.45 * mid_trend - 0.30 * exhaustion + 0.20 * cr["low_to_close5"] - 0.20 * rank["upper_shadow5"],
    }


def session_records(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, interval: int) -> list[dict[str, Any]]:
    rows = []
    for session in CVR.sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE]
        if len(cols) < max(TOP_KS):
            continue
        labels = []
        kept_cols = []
        for col in cols:
            y = future_return(market, idx, col, interval)
            if y is None:
                continue
            labels.append(y)
            kept_cols.append(col)
        if len(labels) < max(TOP_KS):
            continue
        labels_arr = np.asarray(labels, dtype=float)
        kept = np.asarray(kept_cols, dtype=int)
        scores = candidate_scores(feature_frame(market, idx, kept))
        candidates = {}
        for name, score in scores.items():
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(score[j]), str(market["symbols"][kept[j]])))
            candidates[name] = {f"top{k}": safe_mean(labels_arr[order[:k]]) for k in TOP_KS}
        tail = np.sort(labels_arr)[::-1]
        rows.append({
            "date": session,
            "year": session[:4],
            "candidate_top10": {name: item["top10"] for name, item in candidates.items()},
            "candidate_top20": {name: item["top20"] for name, item in candidates.items()},
            "oracle_top20": safe_mean(tail[:20]),
        })
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out: dict[str, Any] = {
        "sessions": len(rows),
        "oracle_top20": mean(row["oracle_top20"] for row in rows),
        "candidate_top10": {name: mean(row["candidate_top10"][name] for row in rows) for name in CANDIDATES},
        "candidate_top20": {name: mean(row["candidate_top20"][name] for row in rows) for name in CANDIDATES},
    }
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate(part) for year, part in sorted(years.items()) if len(part) != len(rows)}
    return out


def flatten(dev: dict[str, Any], forward: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for name in CANDIDATES:
        year_values = [item["candidate_top20"][name] for item in dev.get("by_year", {}).values()]
        worst = min(year_values) if year_values else 0.0
        dev_value = dev["candidate_top20"][name]
        fwd_value = forward["candidate_top20"][name]
        rows.append({
            "interval": interval,
            "candidate": name,
            "dev_top10": dev["candidate_top10"][name],
            "dev_top20": dev_value,
            "dev_worst_year_top20": worst,
            "forward_top10": forward["candidate_top10"][name],
            "forward_top20": fwd_value,
            "retention": fwd_value / dev_value if abs(dev_value) > 1e-9 else 0.0,
            "score": dev_value + 0.75 * fwd_value + 0.45 * worst,
        })
    return sorted(rows, key=lambda row: -row["score"])


def run_interval(market: dict[str, Any], amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    dev_rows = session_records(market, amount20_arr, DEV_START, DEV_END, interval)
    fwd_rows = session_records(market, amount20_arr, FWD_START, FWD_END, interval)
    dev = aggregate(dev_rows)
    forward = aggregate(fwd_rows)
    return {"dev": dev, "forward": forward, "leaderboard": flatten(dev, forward, interval), "sessions": {"dev": len(dev_rows), "forward": len(fwd_rows)}}


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    intervals = {str(interval): run_interval(market, amount20_arr, interval) for interval in INTERVALS}
    leaderboard = sorted([row for item in intervals.values() for row in item["leaderboard"]], key=lambda row: -row["score"])
    best = leaderboard[0] if leaderboard else {}
    verdict = "crowded_trend_exhaustion_avoidance_not_enough"
    if best and best["dev_top20"] > 0.012 and best["dev_worst_year_top20"] > 0.0 and best["forward_top20"] > 0.010:
        verdict = "crowded_trend_exhaustion_avoidance_candidate_needs_replay"
    out = {
        "experiment": "crowded_trend_exhaustion_avoidance_v1",
        "method": "avoid_extreme_trend_volume_exhaustion_and_test_mid_trend_candidates",
        "params": {
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_ks": TOP_KS,
            "intervals": INTERVALS,
            "candidates": CANDIDATES,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "bifurcation_summary": sha256(ROOT / "trend_continuation_bifurcation_diagnostic_v1_summary.json") if (ROOT / "trend_continuation_bifurcation_diagnostic_v1_summary.json").exists() else None,
        },
        "intervals": intervals,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "leaderboard_top12": leaderboard[:12], "sessions": {k: v["sessions"] for k, v in intervals.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
