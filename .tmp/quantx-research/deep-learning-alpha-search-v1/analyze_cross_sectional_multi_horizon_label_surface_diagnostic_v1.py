from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_multi_horizon_label_surface_diagnostic_v1_summary.json"

START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
SEQ_LEN = 60
POOL_SIZE = 500
TOP_K = 10
SESSION_STEP = 5
HORIZONS = [3, 5, 10]
TOP_Q = 0.80
BOTTOM_Q = 0.20


FEATURE_NAMES = [
    "ret1",
    "open_gap",
    "intraday",
    "day_range",
    "close_vwap",
    "log_vol_ratio",
    "ret5",
    "ret20",
    "rank_ret1",
    "rank_ret20",
    "rank_amount",
]

FEATURE_DIRECTIONS = {
    "ret1": [1, -1],
    "intraday": [1, -1],
    "day_range": [1, -1],
    "close_vwap": [1, -1],
    "log_vol_ratio": [1, -1],
    "ret5": [1, -1],
    "ret20": [1, -1],
    "rank_ret1": [1, -1],
    "rank_ret20": [1, -1],
    "rank_amount": [1, -1],
    "low_vol_uptrend": [1],
    "panic_exhaustion": [1],
    "quiet_pullback": [1],
    "liquid_momentum": [1],
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RT = load_module("year_invariant_rank_transformer_v1_for_horizon_diag", ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py")
POS = RT.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def due_sessions(market: dict[str, Any], start: str, end: str, max_horizon: int) -> list[str]:
    dates = market["dates"]
    out = []
    for day in dates:
        idx = market["date_index"][day]
        if start <= day <= end and idx >= SEQ_LEN + 20 and idx + max_horizon + 1 < len(dates):
            out.append(day)
    return out[::SESSION_STEP]


def future_return_horizon(market: dict[str, Any], idx: int, col: int, horizon: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + horizon + 1
    arrays = market["arrays"]
    entry = float(arrays["open"][entry_idx, col])
    exit_ = float(arrays["open"][exit_idx, col])
    close_t = float(arrays["close"][idx, col])
    if not np.isfinite(entry) or not np.isfinite(exit_) or not np.isfinite(close_t) or entry <= 0 or exit_ <= 0 or close_t <= 0:
        return None
    if abs(entry / close_t - 1.0) > 0.095:
        return None
    return exit_ / entry - 1.0


def rank_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) < TOP_K or np.std(scores) <= 1e-12 or np.std(labels) <= 1e-12:
        return 0.0
    sr = pd.Series(scores).rank().to_numpy(dtype=float)
    yr = pd.Series(labels).rank().to_numpy(dtype=float)
    return float(np.corrcoef(sr, yr)[0, 1])


def candidate_scores(view: np.ndarray) -> dict[str, np.ndarray]:
    by_name = {name: view[:, i] for i, name in enumerate(FEATURE_NAMES)}
    scores = dict(by_name)
    scores["low_vol_uptrend"] = by_name["rank_ret20"] - 0.50 * by_name["day_range"] - 0.20 * by_name["log_vol_ratio"]
    scores["panic_exhaustion"] = -by_name["rank_ret20"] + by_name["day_range"] + by_name["log_vol_ratio"]
    scores["quiet_pullback"] = -by_name["rank_ret1"] + 0.40 * by_name["rank_ret20"] - 0.30 * by_name["day_range"]
    scores["liquid_momentum"] = by_name["rank_ret20"] + 0.35 * by_name["rank_amount"] + 0.20 * by_name["ret5"]
    return scores


def empty_candidate() -> dict[str, Any]:
    return {"top10_labels": [], "ics": [], "sessions": 0}


def summarize_candidate(values: dict[str, Any]) -> dict[str, float]:
    labels = np.asarray(values["top10_labels"], dtype=float)
    ics = np.asarray(values["ics"], dtype=float)
    return {
        "sessions": int(values["sessions"]),
        "mean_top10_label": float(np.mean(labels)) if len(labels) else 0.0,
        "median_top10_label": float(np.median(labels)) if len(labels) else 0.0,
        "mean_rank_ic": float(np.mean(ics)) if len(ics) else 0.0,
        "median_rank_ic": float(np.median(ics)) if len(ics) else 0.0,
        "positive_top10_ratio": float(np.mean(labels > 0.0)) if len(labels) else 0.0,
    }


def summarize_year_labels(labels: list[float]) -> dict[str, float]:
    arr = np.asarray(labels, dtype=float)
    if not len(arr):
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "top20_mean": 0.0, "bottom20_mean": 0.0}
    hi = float(np.quantile(arr, TOP_Q))
    lo = float(np.quantile(arr, BOTTOM_Q))
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "top20_mean": float(np.mean(arr[arr >= hi])),
        "bottom20_mean": float(np.mean(arr[arr <= lo])),
    }


def stability_score(years: dict[str, float], forward_value: float) -> float:
    arr = np.asarray(list(years.values()), dtype=float)
    if not len(arr):
        return -1e9
    return float(np.mean(arr) - 0.80 * np.std(arr) + 0.40 * np.min(arr) + 0.35 * forward_value)


def build_surface(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray) -> dict[str, Any]:
    max_horizon = max(HORIZONS)
    sessions = due_sessions(market, START, FWD_END, max_horizon)
    data: dict[str, Any] = {
        str(h): {
            "label_by_year": {},
            "candidates": {"last": {}, "mean5": {}, "mean20": {}},
        }
        for h in HORIZONS
    }
    for h in HORIZONS:
        for view in ("last", "mean5", "mean20"):
            for name, directions in FEATURE_DIRECTIONS.items():
                for direction in directions:
                    key = f"{name}_{'pos' if direction > 0 else 'neg'}"
                    data[str(h)]["candidates"][view][key] = {}

    for session in sessions:
        idx = market["date_index"][session]
        year = session[:4]
        cols = RT.pool_for_session(market, amount20_arr, idx)
        if len(cols) < TOP_K:
            continue
        x_last = feats[idx, cols, :]
        x_mean5 = feats[idx - 4:idx + 1, cols, :].mean(axis=0)
        x_mean20 = feats[idx - 19:idx + 1, cols, :].mean(axis=0)
        views = {"last": x_last, "mean5": x_mean5, "mean20": x_mean20}
        for h in HORIZONS:
            labels = []
            keep = []
            for j, col in enumerate(cols):
                ret = future_return_horizon(market, idx, col, h)
                if ret is None:
                    continue
                labels.append(ret)
                keep.append(j)
            if len(labels) < TOP_K:
                continue
            y = np.asarray(labels, dtype=float)
            data_h = data[str(h)]
            data_h["label_by_year"].setdefault(year, []).extend(float(v) for v in y)
            keep_idx = np.asarray(keep, dtype=int)
            for view_name, raw_view in views.items():
                scores_by_name = candidate_scores(raw_view[keep_idx])
                for name, directions in FEATURE_DIRECTIONS.items():
                    base = scores_by_name[name]
                    for direction in directions:
                        key = f"{name}_{'pos' if direction > 0 else 'neg'}"
                        bucket = data_h["candidates"][view_name][key].setdefault(year, empty_candidate())
                        score = base * float(direction)
                        ordered = np.argsort(-score, kind="mergesort")[:TOP_K]
                        bucket["top10_labels"].append(float(np.mean(y[ordered])))
                        bucket["ics"].append(rank_ic(score, y))
                        bucket["sessions"] += 1
    return data


def summarize_surface(surface: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon, hdata in surface.items():
        label_by_year = {year: summarize_year_labels(values) for year, values in sorted(hdata["label_by_year"].items())}
        candidates: dict[str, Any] = {}
        leaderboard = []
        for view, cands in hdata["candidates"].items():
            candidates[view] = {}
            for key, by_year in cands.items():
                yearly = {year: summarize_candidate(values) for year, values in sorted(by_year.items())}
                dev_values = {year: row["mean_top10_label"] for year, row in yearly.items() if year <= "2025"}
                fwd_value = yearly.get("2026", {}).get("mean_top10_label", 0.0)
                score = stability_score(dev_values, fwd_value)
                candidates[view][key] = {"by_year": yearly, "stability_score": score}
                leaderboard.append({"horizon": int(horizon), "view": view, "candidate": key, "stability_score": score, "dev_year_values": dev_values, "forward_top10_label": fwd_value, "forward_rank_ic": yearly.get("2026", {}).get("mean_rank_ic", 0.0)})
        out[horizon] = {"label_by_year": label_by_year, "candidates": candidates, "leaderboard": sorted(leaderboard, key=lambda row: -row["stability_score"])[:20]}
    flat = [row for h in out.values() for row in h["leaderboard"]]
    out["global_leaderboard"] = sorted(flat, key=lambda row: -row["stability_score"])[:30]
    best = out["global_leaderboard"][0] if out["global_leaderboard"] else {}
    verdict = "multi_horizon_label_surface_no_stable_daily_signal"
    if best and best.get("forward_top10_label", 0.0) > 0 and min(best.get("dev_year_values", {"x": -1}).values()) > 0:
        verdict = "multi_horizon_label_surface_has_stable_candidate_needs_model_probe"
    out["verdict"] = verdict
    return out


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    surface = build_surface(market, feats, amount20_arr)
    summary = summarize_surface(surface)
    out = {
        "experiment": "cross_sectional_multi_horizon_label_surface_diagnostic_v1",
        "method": "no_training_horizon_3_5_10_label_surface_and_feature_direction_stability_scan_pool500_due5",
        "params": {"start": START, "dev_end": DEV_END, "forward": [FWD_START, FWD_END], "seq_len": SEQ_LEN, "pool_size": POOL_SIZE, "top_k": TOP_K, "session_step": SESSION_STEP, "horizons": HORIZONS, "feature_names": FEATURE_NAMES, "feature_directions": FEATURE_DIRECTIONS},
        "inputs_sha256": {"script": sha256(Path(__file__)), "rank_transformer_script": sha256(ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py"), "rank_transformer_summary": sha256(ROOT / "cross_sectional_year_invariant_rank_transformer_v1_summary.json") if (ROOT / "cross_sectional_year_invariant_rank_transformer_v1_summary.json").exists() else None},
        "summary": summary,
        "verdict": summary["verdict"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": out["verdict"], "global_leaderboard_top10": summary["global_leaderboard"][:10], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
