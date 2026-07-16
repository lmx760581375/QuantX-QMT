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
OUT = ROOT / "right_tail_event_precursor_mining_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
POOL_SIZE = 500
TOP_KS = [10, 20]
INTERVALS = [3, 5]
RIGHT_TAIL_K = 20

PRECURSORS = [
    "ret5",
    "ret20",
    "ret60",
    "ret20_minus_ret60",
    "pullback5_after_ret20",
    "near_high20",
    "near_high60",
    "range5",
    "range20",
    "compression_5_vs_20",
    "vol5_vs_20",
    "vol20_vs_60",
    "close_vwap5",
    "close_vwap20",
    "amount_rank",
    "low_to_close5",
    "high_to_close5",
]

CANDIDATES = [
    "compression_uptrend",
    "pullback_near_high_absorption",
    "volume_dry_uptrend",
    "volume_expand_reclaim",
    "quiet_ret20_leader",
    "low_to_close_recovery_trend",
    "anti_chase_near_high",
    "right_tail_precursor_composite",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_right_tail_precursor", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def finite_rank(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(clean[np.isfinite(clean)])) if np.isfinite(clean).any() else 0.0
    order = np.argsort(np.where(np.isfinite(clean), clean, fill), kind="mergesort")
    ranks = np.empty(len(clean), dtype=float)
    ranks[order] = np.linspace(-0.5, 0.5, len(clean), endpoint=True) if len(clean) > 1 else 0.0
    return ranks


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


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


def precursor_frame(market: dict[str, Any], idx: int, cols: np.ndarray) -> dict[str, np.ndarray]:
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
    close5 = window(close, idx, cols, 5)
    close20 = window(close, idx, cols, 20)
    close60 = window(close, idx, cols, 60)
    high20 = np.nanmax(window(high, idx, cols, 20), axis=0)
    high60 = np.nanmax(window(high, idx, cols, 60), axis=0)
    low5 = np.nanmin(window(low, idx, cols, 5), axis=0)
    high5 = np.nanmax(window(high, idx, cols, 5), axis=0)
    range5 = np.nanmean(safe_div(window(high, idx, cols, 5), window(low, idx, cols, 5)) - 1.0, axis=0)
    range20 = np.nanmean(safe_div(window(high, idx, cols, 20), window(low, idx, cols, 20)) - 1.0, axis=0)
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
        "ret20_minus_ret60": ret20 - ret60,
        "pullback5_after_ret20": ret20 - ret5,
        "near_high20": safe_div(c0, high20) - 1.0,
        "near_high60": safe_div(c0, high60) - 1.0,
        "range5": range5,
        "range20": range20,
        "compression_5_vs_20": -safe_div(range5, range20),
        "vol5_vs_20": np.log(np.where(safe_div(vol5, vol20) > 0, safe_div(vol5, vol20), np.nan)),
        "vol20_vs_60": np.log(np.where(safe_div(vol20, vol60) > 0, safe_div(vol20, vol60), np.nan)),
        "close_vwap5": safe_div(c0, vwap5) - 1.0,
        "close_vwap20": safe_div(c0, vwap20) - 1.0,
        "amount_rank": finite_rank(amount),
        "low_to_close5": safe_div(c0, low5) - 1.0,
        "high_to_close5": safe_div(c0, high5) - 1.0,
    }


def candidate_scores(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r = {name: finite_rank(values) for name, values in frame.items()}
    return {
        "compression_uptrend": 0.55 * r["ret20"] + 0.35 * r["compression_5_vs_20"] - 0.20 * r["range20"],
        "pullback_near_high_absorption": 0.45 * r["ret20"] + 0.35 * r["pullback5_after_ret20"] + 0.30 * r["near_high20"] - 0.20 * r["vol5_vs_20"],
        "volume_dry_uptrend": 0.55 * r["ret20"] - 0.35 * r["vol5_vs_20"] - 0.25 * r["range5"] + 0.20 * r["near_high60"],
        "volume_expand_reclaim": 0.35 * r["ret20"] + 0.35 * r["vol5_vs_20"] + 0.30 * r["low_to_close5"] - 0.25 * r["high_to_close5"],
        "quiet_ret20_leader": 0.65 * r["ret20"] - 0.45 * r["range20"] - 0.20 * r["amount_rank"],
        "low_to_close_recovery_trend": 0.45 * r["low_to_close5"] + 0.35 * r["ret20"] - 0.25 * r["ret5"] - 0.15 * r["range5"],
        "anti_chase_near_high": 0.45 * r["near_high20"] + 0.35 * r["ret20"] - 0.45 * r["ret5"] - 0.25 * r["close_vwap5"],
        "right_tail_precursor_composite": 0.25 * r["ret20"] + 0.20 * r["near_high20"] + 0.20 * r["compression_5_vs_20"] + 0.20 * r["low_to_close5"] - 0.20 * r["ret5"] - 0.15 * r["range20"],
    }


def session_records(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, interval: int) -> list[dict[str, Any]]:
    rows = []
    for session in CVR.sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE]
        if len(cols) < RIGHT_TAIL_K:
            continue
        labels = []
        kept_cols = []
        for col in cols:
            y = future_return(market, idx, col, interval)
            if y is None:
                continue
            labels.append(y)
            kept_cols.append(col)
        if len(labels) < RIGHT_TAIL_K:
            continue
        labels_arr = np.asarray(labels, dtype=float)
        kept = np.asarray(kept_cols, dtype=int)
        frame = precursor_frame(market, idx, kept)
        scores = candidate_scores(frame)
        candidates = {}
        for name, score in scores.items():
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(score[j]), str(market["symbols"][kept[j]])))
            candidates[name] = {f"top{k}": safe_mean(labels_arr[order[:k]]) for k in TOP_KS}
        tail_order = np.argsort(labels_arr)[::-1]
        bottom_order = np.argsort(labels_arr)
        contrast = {}
        tail_idx = tail_order[:RIGHT_TAIL_K]
        bottom_idx = bottom_order[:RIGHT_TAIL_K]
        for name in PRECURSORS:
            values = frame[name]
            contrast[name] = {
                "tail_mean": safe_mean(values[tail_idx]),
                "pool_mean": safe_mean(values),
                "bottom_mean": safe_mean(values[bottom_idx]),
                "tail_minus_pool": safe_mean(values[tail_idx]) - safe_mean(values),
                "tail_minus_bottom": safe_mean(values[tail_idx]) - safe_mean(values[bottom_idx]),
            }
        rows.append({
            "date": session,
            "year": session[:4],
            "candidate_top10": {name: item["top10"] for name, item in candidates.items()},
            "candidate_top20": {name: item["top20"] for name, item in candidates.items()},
            "oracle_top20": safe_mean(labels_arr[tail_idx]),
            "oracle_bottom20": safe_mean(labels_arr[bottom_idx]),
            "contrast": contrast,
        })
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out: dict[str, Any] = {
        "sessions": len(rows),
        "oracle_top20": mean(row["oracle_top20"] for row in rows),
        "oracle_bottom20": mean(row["oracle_bottom20"] for row in rows),
        "candidate_top10": {name: mean(row["candidate_top10"][name] for row in rows) for name in CANDIDATES},
        "candidate_top20": {name: mean(row["candidate_top20"][name] for row in rows) for name in CANDIDATES},
        "precursor_contrast": {},
    }
    for name in PRECURSORS:
        out["precursor_contrast"][name] = {
            key: mean(row["contrast"][name][key] for row in rows)
            for key in ["tail_mean", "pool_mean", "bottom_mean", "tail_minus_pool", "tail_minus_bottom"]
        }
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate(part) for year, part in sorted(years.items()) if len(part) != len(rows)}
    return out


def flatten_candidates(dev: dict[str, Any], forward: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for name in CANDIDATES:
        dev_value = dev["candidate_top20"][name]
        fwd_value = forward["candidate_top20"][name]
        year_values = [item["candidate_top20"][name] for item in dev.get("by_year", {}).values()]
        worst = min(year_values) if year_values else 0.0
        rows.append({
            "interval": interval,
            "candidate": name,
            "dev_top10": dev["candidate_top10"][name],
            "dev_top20": dev_value,
            "dev_worst_year_top20": worst,
            "forward_top10": forward["candidate_top10"][name],
            "forward_top20": fwd_value,
            "retention": fwd_value / dev_value if abs(dev_value) > 1e-9 else 0.0,
            "score": dev_value + 0.7 * fwd_value + 0.4 * worst,
        })
    return sorted(rows, key=lambda row: -row["score"])


def flatten_contrasts(dev: dict[str, Any], forward: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for name in PRECURSORS:
        d = dev["precursor_contrast"][name]
        f = forward["precursor_contrast"][name]
        rows.append({
            "interval": interval,
            "precursor": name,
            "dev_tail_minus_pool": d["tail_minus_pool"],
            "dev_tail_minus_bottom": d["tail_minus_bottom"],
            "forward_tail_minus_pool": f["tail_minus_pool"],
            "forward_tail_minus_bottom": f["tail_minus_bottom"],
            "score": abs(d["tail_minus_pool"]) + 0.5 * abs(f["tail_minus_pool"]),
        })
    return sorted(rows, key=lambda row: -row["score"])


def run_interval(market: dict[str, Any], amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    dev_rows = session_records(market, amount20_arr, DEV_START, DEV_END, interval)
    fwd_rows = session_records(market, amount20_arr, FWD_START, FWD_END, interval)
    dev = aggregate(dev_rows)
    forward = aggregate(fwd_rows)
    return {
        "dev": dev,
        "forward": forward,
        "candidate_leaderboard": flatten_candidates(dev, forward, interval),
        "contrast_leaderboard": flatten_contrasts(dev, forward, interval),
        "sessions": {"dev": len(dev_rows), "forward": len(fwd_rows)},
    }


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    intervals = {str(interval): run_interval(market, amount20_arr, interval) for interval in INTERVALS}
    candidate_leaderboard = sorted([row for item in intervals.values() for row in item["candidate_leaderboard"]], key=lambda row: -row["score"])
    contrast_leaderboard = sorted([row for item in intervals.values() for row in item["contrast_leaderboard"]], key=lambda row: -row["score"])
    best = candidate_leaderboard[0] if candidate_leaderboard else {}
    verdict = "right_tail_event_precursor_not_enough"
    if best and best["dev_top20"] > 0.012 and best["dev_worst_year_top20"] > 0.0 and best["forward_top20"] > 0.010:
        verdict = "right_tail_event_precursor_candidate_needs_model_probe"
    out = {
        "experiment": "right_tail_event_precursor_mining_v1",
        "method": "mine_future_right_tail_path_precursors_then_test_handcrafted_event_scores",
        "params": {
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_ks": TOP_KS,
            "intervals": INTERVALS,
            "right_tail_k": RIGHT_TAIL_K,
            "precursors": PRECURSORS,
            "candidates": CANDIDATES,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "market_state_summary": sha256(ROOT / "cross_sectional_market_state_path_tail_diagnostic_v1_summary.json") if (ROOT / "cross_sectional_market_state_path_tail_diagnostic_v1_summary.json").exists() else None,
        },
        "intervals": intervals,
        "candidate_leaderboard": candidate_leaderboard,
        "contrast_leaderboard": contrast_leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "candidate_top12": candidate_leaderboard[:12], "contrast_top12": contrast_leaderboard[:12], "sessions": {k: v["sessions"] for k, v in intervals.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
