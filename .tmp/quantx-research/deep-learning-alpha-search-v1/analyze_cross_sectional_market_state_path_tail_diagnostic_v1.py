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


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_market_state_path_tail_diagnostic_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
SEQ_LEN = 60
POOL_SIZE = 500
TOP_KS = [10, 20]
INTERVALS = [3, 5]
STATE_BUCKETS = 4
MIN_BUCKET_SESSIONS = 12

STATE_FEATURES = [
    "market_ret20_median",
    "market_ret60_median",
    "breadth_ret5_pos",
    "breadth_ret20_pos",
    "near_high20_rate",
    "disp_ret5",
    "disp_ret20",
    "tail_spread20",
    "crowding_hot_vol_rate",
    "close_vwap_neg_rate",
    "quiet_trend_rate",
]

CANDIDATES = [
    "rank_amount_pos",
    "ret20_momentum",
    "pullback_from_high20",
    "low_vol_uptrend",
    "close_vwap_neg",
    "anti_crowding_quiet_trend",
    "elastic_low_liquidity_proxy",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_market_state_tail", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def safe_mean(values: np.ndarray | list[float]) -> float:
    arr = finite_values(np.asarray(values, dtype=float))
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_std(values: np.ndarray) -> float:
    arr = finite_values(values)
    return float(np.std(arr)) if len(arr) else 0.0


def safe_quantile(values: np.ndarray, q: float) -> float:
    arr = finite_values(values)
    return float(np.quantile(arr, q)) if len(arr) else 0.0


def finite_rank(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(clean[np.isfinite(clean)])) if np.isfinite(clean).any() else 0.0
    order = np.argsort(np.where(np.isfinite(clean), clean, fill), kind="mergesort")
    ranks = np.empty(len(clean), dtype=float)
    ranks[order] = np.linspace(-0.5, 0.5, len(clean), endpoint=True) if len(clean) > 1 else 0.0
    return ranks


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


def state_features(values: np.ndarray) -> dict[str, float]:
    day_range = values[:, 3]
    close_vwap = values[:, 4]
    ret5 = values[:, 6]
    ret20 = values[:, 7]
    ret60 = values[:, 8]
    rank_ret20 = values[:, 9]
    rank_amount = values[:, 10]
    high20_pos = values[:, 11] if values.shape[1] > 11 else rank_ret20
    hot = (ret5 > safe_quantile(ret5, 0.8)) & (day_range > safe_quantile(day_range, 0.8))
    quiet_trend = (ret20 > safe_quantile(ret20, 0.6)) & (day_range < safe_quantile(day_range, 0.5)) & (rank_amount > safe_quantile(rank_amount, 0.4))
    return {
        "market_ret20_median": safe_quantile(ret20, 0.5),
        "market_ret60_median": safe_quantile(ret60, 0.5),
        "breadth_ret5_pos": float(np.mean(ret5 > 0.0)),
        "breadth_ret20_pos": float(np.mean(ret20 > 0.0)),
        "near_high20_rate": float(np.mean(high20_pos > safe_quantile(high20_pos, 0.7))),
        "disp_ret5": safe_std(ret5),
        "disp_ret20": safe_std(ret20),
        "tail_spread20": safe_quantile(ret20, 0.9) - safe_quantile(ret20, 0.1),
        "crowding_hot_vol_rate": float(np.mean(hot)),
        "close_vwap_neg_rate": float(np.mean(close_vwap < 0.0)),
        "quiet_trend_rate": float(np.mean(quiet_trend)),
    }


def candidate_scores(values: np.ndarray) -> dict[str, np.ndarray]:
    day_range = values[:, 3]
    close_vwap = values[:, 4]
    ret5 = values[:, 6]
    ret20 = values[:, 7]
    ret60 = values[:, 8]
    rank_amount = values[:, 10]
    high20_pos = values[:, 11] if values.shape[1] > 11 else ret20
    return {
        "rank_amount_pos": rank_amount,
        "ret20_momentum": ret20 + 0.25 * ret60,
        "pullback_from_high20": finite_rank(high20_pos) - 0.50 * finite_rank(ret5),
        "low_vol_uptrend": finite_rank(ret20) - 0.65 * finite_rank(day_range),
        "close_vwap_neg": -close_vwap,
        "anti_crowding_quiet_trend": finite_rank(ret20) - 0.45 * finite_rank(ret5) - 0.45 * finite_rank(day_range) - 0.20 * finite_rank(close_vwap),
        "elastic_low_liquidity_proxy": -0.55 * finite_rank(rank_amount) + 0.40 * finite_rank(ret20) - 0.30 * finite_rank(ret5),
    }


def label_stats(labels: np.ndarray) -> dict[str, float]:
    arr = finite_values(labels)
    if len(arr) == 0:
        return {"mean": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "top10": 0.0, "top20": 0.0, "top50": 0.0, "bottom20": 0.0, "top20_minus_bottom20": 0.0}
    ordered = np.sort(arr)[::-1]
    bottom = np.sort(arr)
    top10 = safe_mean(ordered[: min(10, len(ordered))])
    top20 = safe_mean(ordered[: min(20, len(ordered))])
    top50 = safe_mean(ordered[: min(50, len(ordered))])
    bottom20 = safe_mean(bottom[: min(20, len(bottom))])
    return {
        "mean": safe_mean(arr),
        "p90": safe_quantile(arr, 0.90),
        "p95": safe_quantile(arr, 0.95),
        "p99": safe_quantile(arr, 0.99),
        "top10": top10,
        "top20": top20,
        "top50": top50,
        "bottom20": bottom20,
        "top20_minus_bottom20": top20 - bottom20,
    }


def build_sessions(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, interval: int) -> list[dict[str, Any]]:
    rows = []
    for session in CVR.sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE]
        if len(cols) < max(TOP_KS):
            continue
        values = feats[idx - 19:idx + 1, cols, :].mean(axis=0)
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
        kept_positions = [list(cols).index(col) for col in kept_cols]
        kept_values = values[kept_positions, :]
        labels_arr = np.asarray(labels, dtype=float)
        scores = candidate_scores(kept_values)
        candidates = {}
        for name, score in scores.items():
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(score[j]), str(market["symbols"][kept_cols[j]])))
            candidates[name] = {f"top{k}": safe_mean(labels_arr[order[:k]]) for k in TOP_KS}
        rows.append({
            "date": session,
            "year": session[:4],
            "state": state_features(kept_values),
            "label_stats": label_stats(labels_arr),
            "candidates": candidates,
            "sample_count": len(labels_arr),
        })
    return rows


def bucket_edges(rows: list[dict[str, Any]], feature: str) -> np.ndarray:
    values = np.asarray([row["state"][feature] for row in rows], dtype=float)
    if len(values) == 0:
        return np.asarray([], dtype=float)
    quantiles = [i / STATE_BUCKETS for i in range(1, STATE_BUCKETS)]
    return np.asarray([safe_quantile(values, q) for q in quantiles], dtype=float)


def bucket_codes(rows: list[dict[str, Any]], feature: str, edges: np.ndarray | None = None) -> dict[str, int]:
    values = np.asarray([row["state"][feature] for row in rows], dtype=float)
    if edges is None:
        edges = bucket_edges(rows, feature)
    codes = np.searchsorted(edges, values, side="right").astype(int)
    codes = np.clip(codes, 0, STATE_BUCKETS - 1)
    return {rows[i]["date"]: int(codes[i]) for i in range(len(rows))}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out: dict[str, Any] = {"sessions": len(rows), "avg_sample_count": mean(row["sample_count"] for row in rows)}
    for key in ["top10", "top20", "top50", "p95", "p99", "top20_minus_bottom20"]:
        out[f"oracle_{key}"] = mean(row["label_stats"][key] for row in rows)
    out["candidate_top10"] = {name: mean(row["candidates"][name]["top10"] for row in rows) for name in CANDIDATES}
    out["candidate_top20"] = {name: mean(row["candidates"][name]["top20"] for row in rows) for name in CANDIDATES}
    out["best_candidate_top20"] = max(out["candidate_top20"].items(), key=lambda item: item[1])
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate_rows(part) for year, part in sorted(years.items()) if year != rows[0]["year"] or len(part) != len(rows)}
    return out


def state_bucket_report(dev_rows: list[dict[str, Any]], fwd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for feature in STATE_FEATURES:
        edges = bucket_edges(dev_rows, feature)
        dev_codes = bucket_codes(dev_rows, feature, edges)
        fwd_codes = bucket_codes(fwd_rows, feature, edges)
        buckets = {}
        for code in range(STATE_BUCKETS):
            dev_part = [row for row in dev_rows if dev_codes[row["date"]] == code]
            fwd_part = [row for row in fwd_rows if fwd_codes[row["date"]] == code]
            buckets[str(code)] = {"dev": aggregate_rows(dev_part), "forward": aggregate_rows(fwd_part)}
        report[feature] = {"edges": [float(x) for x in edges], "buckets": buckets}
    return report


def flatten_state_edges(bucket_report: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for feature, payload in bucket_report.items():
        buckets = payload["buckets"]
        for code, item in buckets.items():
            dev = item["dev"]
            fwd = item["forward"]
            if dev.get("sessions", 0) < MIN_BUCKET_SESSIONS or fwd.get("sessions", 0) < 4:
                continue
            dev_best_name, dev_best_value = dev["best_candidate_top20"]
            fwd_value = fwd["candidate_top20"].get(dev_best_name, 0.0)
            dev_year_values = [year_item.get("candidate_top20", {}).get(dev_best_name, 0.0) for year_item in dev.get("by_year", {}).values() if year_item.get("sessions", 0) > 0]
            worst_year = min(dev_year_values) if dev_year_values else 0.0
            retention = fwd_value / dev_best_value if abs(dev_best_value) > 1e-9 else 0.0
            rows.append({
                "interval": interval,
                "state_feature": feature,
                "bucket": int(code),
                "dev_sessions": dev["sessions"],
                "forward_sessions": fwd["sessions"],
                "dev_oracle_top20": dev["oracle_top20"],
                "forward_oracle_top20": fwd["oracle_top20"],
                "best_candidate": dev_best_name,
                "dev_candidate_top20": dev_best_value,
                "forward_candidate_top20": fwd_value,
                "dev_worst_year_candidate_top20": worst_year,
                "forward_retention": retention,
                "score": dev_best_value + 0.70 * fwd_value + 0.25 * worst_year + 0.15 * min(retention, 2.0) + 0.10 * dev["oracle_top20"],
            })
    return sorted(rows, key=lambda row: -row["score"])


def run_interval(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    dev_rows = build_sessions(market, feats, amount20_arr, DEV_START, DEV_END, interval)
    fwd_rows = build_sessions(market, feats, amount20_arr, FWD_START, FWD_END, interval)
    buckets = state_bucket_report(dev_rows, fwd_rows)
    edges = flatten_state_edges(buckets, interval)
    return {
        "dev": aggregate_rows(dev_rows),
        "forward": aggregate_rows(fwd_rows),
        "state_buckets": buckets,
        "state_edges": edges,
        "sessions": {"dev": len(dev_rows), "forward": len(fwd_rows)},
    }


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    intervals = {str(interval): run_interval(market, feats, amount20_arr, interval) for interval in INTERVALS}
    all_edges = sorted([edge for item in intervals.values() for edge in item["state_edges"]], key=lambda row: -row["score"])
    verdict = "market_state_tail_diagnostic_not_enough"
    best = all_edges[0] if all_edges else {}
    if best and best["dev_candidate_top20"] > 0.012 and best["forward_candidate_top20"] > 0.010 and best["dev_worst_year_candidate_top20"] > 0.0 and best["forward_retention"] > 0.5:
        verdict = "market_state_tail_candidate_needs_conditioned_ranker"
    out = {
        "experiment": "cross_sectional_market_state_path_tail_diagnostic_v1",
        "method": "market_state_bucket_right_tail_and_candidate_soil_diagnostic_no_training",
        "params": {
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "seq_len": SEQ_LEN,
            "pool_size": POOL_SIZE,
            "top_ks": TOP_KS,
            "intervals": INTERVALS,
            "state_features": STATE_FEATURES,
            "state_buckets": STATE_BUCKETS,
            "candidates": CANDIDATES,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "close_vwap_replay_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py"),
            "residual_summary": sha256(ROOT / "cross_sectional_residual_reversal_structure_scan_v1_summary.json") if (ROOT / "cross_sectional_residual_reversal_structure_scan_v1_summary.json").exists() else None,
        },
        "intervals": intervals,
        "leaderboard": all_edges,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "leaderboard_top12": all_edges[:12], "sessions": {k: v["sessions"] for k, v in intervals.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
