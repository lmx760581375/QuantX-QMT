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
OUT = ROOT / "trend_exhaustion_regime_boundary_scan_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
POOL_SIZE = 500
TOP_K = 20
INTERVALS = [3, 5]
CANDIDATES = [
    "avoid_extreme_trend_low_range",
    "mid_trend_not_extreme",
    "mid_trend_volume_not_extreme",
    "mid_trend_low_crowding",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CA52 = load_module("crowded_trend_exhaustion_avoidance_v1_for_regime_scan", ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py")
RT = CA52.RT
POS = CA52.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) else 0.0


def q(values: np.ndarray, prob: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, prob)) if len(arr) else 0.0


def market_state(market: dict[str, Any], idx: int, pool_cols: np.ndarray, score_by_candidate: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    c0 = close[idx]
    ret5 = safe_div(c0, close[idx - 5]) - 1.0
    ret20 = safe_div(c0, close[idx - 20]) - 1.0
    ret60 = safe_div(c0, close[idx - 60]) - 1.0
    v20 = np.nanmean(volume[idx - 19:idx + 1], axis=0)
    v60 = np.nanmean(volume[idx - 59:idx + 1], axis=0)
    vol20_vs_60 = np.log(np.where(safe_div(v20, v60) > 0, safe_div(v20, v60), np.nan))
    daily = safe_div(close[idx - 19:idx + 1], close[idx - 20:idx]) - 1.0
    valid = np.isfinite(ret20)
    pool_ret20 = ret20[pool_cols]
    pool_ret60 = ret60[pool_cols]
    pool_vol = vol20_vs_60[pool_cols]
    score_stack = np.vstack([score_by_candidate[name] for name in CANDIDATES])
    best_score = np.nanmax(score_stack, axis=0)
    return {
        "mkt_breadth5": float(np.nanmean(ret5[valid] > 0)),
        "mkt_breadth20": float(np.nanmean(ret20[valid] > 0)),
        "mkt_median_ret20": float(np.nanmedian(ret20[valid])),
        "mkt_median_ret60": float(np.nanmedian(ret60[valid])),
        "mkt_dispersion20": safe_std(ret20[valid]),
        "mkt_vol20": float(np.nanmedian(np.nanstd(daily[:, valid], axis=0))),
        "mkt_vol20_vs_60": float(np.nanmedian(vol20_vs_60[valid])),
        "pool_median_ret20": float(np.nanmedian(pool_ret20)),
        "pool_median_ret60": float(np.nanmedian(pool_ret60)),
        "pool_ret20_p80": q(pool_ret20, 0.80),
        "pool_ret60_p80": q(pool_ret60, 0.80),
        "pool_vol20_vs_60_median": float(np.nanmedian(pool_vol)),
        "pool_score_best_mean": safe_mean(best_score),
        "pool_score_best_p90": q(best_score, 0.90),
        "pool_score_dispersion": safe_std(best_score),
    }


def one_session(market: dict[str, Any], amount20_arr: np.ndarray, session: str, interval: int) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    cols = RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE]
    if len(cols) < TOP_K:
        return None
    labels = []
    kept_cols = []
    for col in cols:
        y = CA52.future_return(market, idx, int(col), interval)
        if y is None:
            continue
        labels.append(y)
        kept_cols.append(int(col))
    if len(labels) < TOP_K:
        return None
    labels_arr = np.asarray(labels, dtype=float)
    kept = np.asarray(kept_cols, dtype=int)
    scores = CA52.candidate_scores(CA52.feature_frame(market, idx, kept))
    candidate_top20 = {}
    for name in CANDIDATES:
        score = scores[name]
        order = sorted(range(len(labels_arr)), key=lambda j: (-float(score[j]), str(market["symbols"][kept[j]])))
        candidate_top20[name] = safe_mean(labels_arr[order[:TOP_K]])
    return {
        "date": session,
        "year": session[:4],
        "interval": interval,
        "candidate_top20": candidate_top20,
        "oracle_top20": safe_mean(np.sort(labels_arr)[::-1][:TOP_K]),
        "state": market_state(market, idx, kept, scores),
    }


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, interval: int) -> list[dict[str, Any]]:
    rows = []
    for session in CA52.CVR.sessions(market, start, end, interval):
        row = one_session(market, amount20_arr, session, interval)
        if row is not None:
            rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out = {
        "sessions": len(rows),
        "oracle_top20": mean(row["oracle_top20"] for row in rows),
        "candidate_top20": {name: mean(row["candidate_top20"][name] for row in rows) for name in CANDIDATES},
    }
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate_rows(part) for year, part in sorted(years.items()) if len(part) != len(rows)}
    return out


def bucket_name(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value >= hi:
        return "high"
    return "mid"


def bucket_scan(dev_rows: list[dict[str, Any]], fwd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not dev_rows:
        return []
    feature_names = list(dev_rows[0]["state"].keys())
    scans = []
    for feature in feature_names:
        vals = np.asarray([row["state"][feature] for row in dev_rows], dtype=float)
        lo, hi = q(vals, 0.30), q(vals, 0.70)
        for bucket in ["low", "mid", "high"]:
            dev_part = [row for row in dev_rows if bucket_name(float(row["state"][feature]), lo, hi) == bucket]
            fwd_part = [row for row in fwd_rows if bucket_name(float(row["state"][feature]), lo, hi) == bucket]
            if len(dev_part) < 20 or len(fwd_part) < 4:
                continue
            dev_ag = aggregate_rows(dev_part)
            fwd_ag = aggregate_rows(fwd_part)
            for candidate in CANDIDATES:
                year_values = [item["candidate_top20"][candidate] for item in dev_ag.get("by_year", {}).values()]
                scans.append({
                    "feature": feature,
                    "bucket": bucket,
                    "thresholds_from_dev": [lo, hi],
                    "candidate": candidate,
                    "dev_sessions": len(dev_part),
                    "forward_sessions": len(fwd_part),
                    "dev_top20": dev_ag["candidate_top20"][candidate],
                    "dev_worst_year_top20": min(year_values) if year_values else 0.0,
                    "forward_top20": fwd_ag["candidate_top20"][candidate],
                    "dev_oracle_top20": dev_ag["oracle_top20"],
                    "forward_oracle_top20": fwd_ag["oracle_top20"],
                    "score": dev_ag["candidate_top20"][candidate] + 0.75 * fwd_ag["candidate_top20"][candidate] + 0.45 * (min(year_values) if year_values else 0.0),
                })
    return sorted(scans, key=lambda item: -item["score"])


def state_contrast(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    if not rows:
        return {}
    returns = np.asarray([row["candidate_top20"][candidate] for row in rows], dtype=float)
    good_cut = q(returns, 0.70)
    bad_cut = q(returns, 0.30)
    good = [row for row in rows if row["candidate_top20"][candidate] >= good_cut]
    bad = [row for row in rows if row["candidate_top20"][candidate] <= bad_cut]
    out = {}
    for feature in rows[0]["state"]:
        out[feature] = {
            "good_mean": safe_mean([row["state"][feature] for row in good]),
            "bad_mean": safe_mean([row["state"][feature] for row in bad]),
            "good_minus_bad": safe_mean([row["state"][feature] for row in good]) - safe_mean([row["state"][feature] for row in bad]),
        }
    return dict(sorted(out.items(), key=lambda item: -abs(item[1]["good_minus_bad"])))


def run_interval(market: dict[str, Any], amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    dev_rows = build_rows(market, amount20_arr, DEV_START, DEV_END, interval)
    fwd_rows = build_rows(market, amount20_arr, FWD_START, FWD_END, interval)
    dev = aggregate_rows(dev_rows)
    forward = aggregate_rows(fwd_rows)
    scans = bucket_scan(dev_rows, fwd_rows)
    contrasts = {candidate: {"dev": state_contrast(dev_rows, candidate), "forward": state_contrast(fwd_rows, candidate)} for candidate in CANDIDATES}
    best = scans[0] if scans else {}
    return {
        "sessions": {"dev": len(dev_rows), "forward": len(fwd_rows)},
        "dev": dev,
        "forward": forward,
        "bucket_leaderboard": scans,
        "best_bucket": best,
        "state_contrast": contrasts,
    }


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    intervals = {str(interval): run_interval(market, amount20_arr, interval) for interval in INTERVALS}
    leaderboard = sorted([row | {"interval": int(interval)} for interval, item in intervals.items() for row in item["bucket_leaderboard"]], key=lambda item: -item["score"])
    best = leaderboard[0] if leaderboard else {}
    verdict = "trend_exhaustion_regime_boundary_not_enough"
    if best and best["dev_top20"] > 0.005 and best["dev_worst_year_top20"] > 0.0 and best["forward_top20"] > 0.005:
        verdict = "trend_exhaustion_regime_boundary_candidate_needs_replay"
    out = {
        "experiment": "trend_exhaustion_regime_boundary_scan_v1",
        "method": "dev_threshold_bucket_scan_for_mid_trend_anti_crowding_regime_boundary",
        "params": {
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_k": TOP_K,
            "intervals": INTERVALS,
            "candidates": CANDIDATES,
            "bucket_thresholds": "q30_q70_fit_on_dev_apply_to_forward",
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round52_script": sha256(ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py"),
            "round52_summary": sha256(ROOT / "crowded_trend_exhaustion_avoidance_v1_summary.json"),
        },
        "intervals": intervals,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "best": best,
        "leaderboard_top12": leaderboard[:12],
        "sessions": {k: v["sessions"] for k, v in intervals.items()},
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
