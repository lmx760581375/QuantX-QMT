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
OUT = ROOT / "cross_sectional_lead_lag_diffusion_probe_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

INTERVAL = 5
POOL_SIZE = 500
TOP_K_REPLAY = 10
TOP_KS = (10, 20, 30)
NEIGHBOR_KS = (10, 20, 40)
RANDOM_TRIALS = 200
SEED = 20260714

CANDIDATES = (
    "neighbor20_lead_self5_lag",
    "neighbor5_accel_self_rest",
    "cluster_confirmed_leader",
    "diffusion_pressure_composite",
    "anti_crowded_cluster_lag",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_for_lead_lag_diffusion", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
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
    ranks[order] = np.linspace(-0.5, 0.5, len(clean), endpoint=True) if len(clean) > 1 else 0.0
    return ranks


def future_return(market: dict[str, Any], idx: int, col: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + INTERVAL + 1
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


def ret(close: np.ndarray, idx: int, cols: np.ndarray, horizon: int) -> np.ndarray:
    if idx < horizon:
        return np.full(len(cols), np.nan)
    return safe_div(close[idx, cols], close[idx - horizon, cols]) - 1.0


def corr_matrix_from_returns(close: np.ndarray, idx: int, cols: np.ndarray, length: int) -> np.ndarray:
    start = idx - length
    prev = close[start:idx, :][:, cols]
    cur = close[start + 1:idx + 1, :][:, cols]
    x = safe_div(cur, prev) - 1.0
    x = np.where(np.isfinite(x), x, np.nan)
    col_mean = np.nanmean(x, axis=0)
    x = np.where(np.isfinite(x), x, col_mean[None, :])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = np.divide(x, std, out=np.zeros_like(x), where=std > 1e-12)
    sim = (x.T @ x) / max(1, x.shape[0] - 1)
    np.fill_diagonal(sim, -np.inf)
    return sim


def neighbor_mean(values: np.ndarray, sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    strength = np.full(n, np.nan, dtype=float)
    for i in range(n):
        row = sim[i]
        finite = np.flatnonzero(np.isfinite(row) & (row > 0.05))
        if len(finite) == 0:
            continue
        take = finite[np.argsort(row[finite])[-k:]]
        weights = np.maximum(row[take], 0.0)
        vals = values[take]
        ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if ok.any():
            out[i] = float(np.sum(vals[ok] * weights[ok]) / np.sum(weights[ok]))
            strength[i] = float(np.mean(weights[ok]))
    return out, strength


def session_scores(market: dict[str, Any], idx: int, cols: np.ndarray, neighbor_k: int) -> dict[str, np.ndarray]:
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    high = arrays["high"].astype(float)
    low = arrays["low"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    sim20 = corr_matrix_from_returns(close, idx, cols, 20)
    sim60 = corr_matrix_from_returns(close, idx, cols, 60)
    sim = 0.65 * np.nan_to_num(sim20, nan=-np.inf, neginf=-np.inf) + 0.35 * np.nan_to_num(sim60, nan=-np.inf, neginf=-np.inf)
    r5 = ret(close, idx, cols, 5)
    r20 = ret(close, idx, cols, 20)
    r60 = ret(close, idx, cols, 60)
    neigh5, strength = neighbor_mean(r5, sim, neighbor_k)
    neigh20, _ = neighbor_mean(r20, sim, neighbor_k)
    neigh60, _ = neighbor_mean(r60, sim, neighbor_k)
    range5 = np.nanmean(safe_div(high[idx - 4:idx + 1, :][:, cols], low[idx - 4:idx + 1, :][:, cols]) - 1.0, axis=0)
    vol5 = np.nanmean(volume[idx - 4:idx + 1, :][:, cols], axis=0)
    vol20 = np.nanmean(volume[idx - 19:idx + 1, :][:, cols], axis=0)
    close_vwap = safe_div(close[idx, cols], vwap[idx, cols]) - 1.0
    amount20 = np.nanmean(np.where(np.isfinite(vwap[idx - 19:idx + 1, :][:, cols]) & (vwap[idx - 19:idx + 1, :][:, cols] > 0), vwap[idx - 19:idx + 1, :][:, cols], close[idx - 19:idx + 1, :][:, cols]) * volume[idx - 19:idx + 1, :][:, cols], axis=0)
    raw = {
        "self_r5": r5,
        "self_r20": r20,
        "self_r60": r60,
        "neigh_r5": neigh5,
        "neigh_r20": neigh20,
        "neigh_r60": neigh60,
        "lead5": neigh5 - r5,
        "lead20": neigh20 - r20,
        "strength": strength,
        "range5": range5,
        "vol5_vs20": np.log(np.where(safe_div(vol5, vol20) > 0, safe_div(vol5, vol20), np.nan)),
        "close_vwap": close_vwap,
        "amount_rank": finite_rank(amount20),
    }
    rr = {name: finite_rank(value) for name, value in raw.items()}
    return {
        "neighbor20_lead_self5_lag": 0.34 * rr["neigh_r20"] + 0.28 * rr["lead20"] - 0.18 * rr["self_r5"] + 0.12 * rr["strength"] + 0.08 * rr["amount_rank"],
        "neighbor5_accel_self_rest": 0.34 * rr["neigh_r5"] + 0.24 * rr["lead5"] + 0.18 * rr["neigh_r20"] - 0.12 * rr["range5"] - 0.12 * rr["vol5_vs20"],
        "cluster_confirmed_leader": 0.28 * rr["self_r20"] + 0.26 * rr["neigh_r20"] + 0.18 * rr["self_r5"] + 0.14 * rr["neigh_r5"] + 0.08 * rr["strength"] + 0.06 * rr["amount_rank"],
        "diffusion_pressure_composite": 0.26 * rr["neigh_r20"] + 0.22 * rr["lead20"] + 0.20 * rr["neigh_r5"] + 0.14 * rr["lead5"] + 0.10 * rr["strength"] + 0.08 * rr["close_vwap"],
        "anti_crowded_cluster_lag": 0.30 * rr["neigh_r20"] + 0.22 * rr["lead20"] - 0.20 * rr["self_r60"] - 0.14 * rr["vol5_vs20"] - 0.08 * rr["range5"] + 0.06 * rr["amount_rank"],
    }


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, neighbor_k: int) -> dict[str, Any]:
    rows = []
    selections = {name: {} for name in CANDIDATES}
    pools: dict[str, list[str]] = {}
    missing = 0
    for session in CVR.sessions(market, start, end, INTERVAL):
        idx = market["date_index"][session]
        if idx < 100:
            continue
        cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
        if len(cols) < 80:
            continue
        y_by_local = {}
        for j, col in enumerate(cols):
            y = future_return(market, idx, int(col))
            if y is None:
                missing += 1
                continue
            y_by_local[j] = float(y)
        if len(y_by_local) < 80:
            continue
        pools[session] = [str(market["symbols"][cols[j]]) for j in y_by_local]
        scores = session_scores(market, idx, cols, neighbor_k)
        stat = {"session": session, "year": session[:4], "candidates": {}}
        valid_locals = list(y_by_local)
        labels = np.asarray([y_by_local[j] for j in valid_locals], dtype=float)
        for name, score_all in scores.items():
            ordered = sorted(valid_locals, key=lambda j: (-float(score_all[j]), str(market["symbols"][cols[j]])))
            picks = [str(market["symbols"][cols[j]]) for j in ordered[:TOP_K_REPLAY]]
            selections[name][session] = picks
            y_order = np.asarray([y_by_local[j] for j in ordered], dtype=float)
            stat["candidates"][name] = {f"top{k}": safe_mean(y_order[:k]) for k in TOP_KS}
        rows.append(stat)
    return {"rows": rows, "selections": selections, "pools": pools, "missing_future": missing}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out = {"sessions": len(rows), "candidates": {}}
    for name in CANDIDATES:
        out["candidates"][name] = {f"top{k}": safe_mean([row["candidates"][name][f"top{k}"] for row in rows]) for k in TOP_KS}
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate(part) for year, part in sorted(years.items()) if len(part) != len(rows)}
    return out


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K_REPLAY, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K_REPLAY}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def eval_split(split: dict[str, Any], market: dict[str, Any], start: str, end: str, random_trials: int, seed_offset: int) -> dict[str, Any]:
    stats = aggregate(split["rows"])
    replay = {name: CVR.replay_interval(split["selections"][name], market, start, end, INTERVAL)["summary"] for name in CANDIDATES}
    rng = np.random.default_rng(SEED + seed_offset)
    random_values = [CVR.replay_interval(random_selections(split["pools"], rng), market, start, end, INTERVAL)["summary"]["final_multiple"] for _ in range(random_trials)]
    return {"stats": stats, "replay": replay, "random_pool_top10": percentile_summary(random_values), "sessions": len(split["rows"]), "missing_future": split["missing_future"]}


def run_neighbor_k(market: dict[str, Any], amount20_arr: np.ndarray, neighbor_k: int) -> dict[str, Any]:
    train = build_split(market, amount20_arr, TRAIN_START, TRAIN_END, neighbor_k)
    valid = build_split(market, amount20_arr, VALID_START, VALID_END, neighbor_k)
    dev = build_split(market, amount20_arr, DEV_START, DEV_END, neighbor_k)
    forward = build_split(market, amount20_arr, FWD_START, FWD_END, neighbor_k)
    return {
        "neighbor_k": neighbor_k,
        "train": eval_split(train, market, TRAIN_START, TRAIN_END, 30, neighbor_k + 10),
        "valid": eval_split(valid, market, VALID_START, VALID_END, 50, neighbor_k + 20),
        "dev": eval_split(dev, market, DEV_START, DEV_END, 50, neighbor_k + 30),
        "forward": eval_split(forward, market, FWD_START, FWD_END, RANDOM_TRIALS, neighbor_k + 40),
    }


def flatten(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, item in results.items():
        nk = item["neighbor_k"]
        for name in CANDIDATES:
            train_stat = item["train"]["stats"]["candidates"][name]
            valid_stat = item["valid"]["stats"]["candidates"][name]
            valid_replay = item["valid"]["replay"][name]
            dev_replay = item["dev"]["replay"][name]
            forward_replay = item["forward"]["replay"][name]
            year_values = dev_replay.get("annual_returns", {})
            rows.append({
                "key": key,
                "neighbor_k": nk,
                "candidate": name,
                "train_top20": train_stat["top20"],
                "valid_top20": valid_stat["top20"],
                "valid_multiple": valid_replay["final_multiple"],
                "dev_multiple": dev_replay["final_multiple"],
                "dev_all_years_positive": dev_replay["all_years_positive"],
                "dev_worst_year": min(year_values.values(), default=0.0),
                "forward_multiple": forward_replay["final_multiple"],
                "forward_remove_best_3": forward_replay["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "forward_avg_position": forward_replay["avg_position_count"],
                "forward_random_p95": item["forward"]["random_pool_top10"]["p95"],
                "forward_beats_random_p95": forward_replay["final_multiple"] > item["forward"]["random_pool_top10"]["p95"],
                "score": math.log(max(valid_replay["final_multiple"], 1e-9)) + 0.25 * valid_stat["top20"] + 0.15 * train_stat["top20"],
            })
    return sorted(rows, key=lambda row: -row["score"])


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    results = {f"neighbor{neighbor_k}": run_neighbor_k(market, amount20_arr, neighbor_k) for neighbor_k in NEIGHBOR_KS}
    leaderboard = flatten(results)
    selected = leaderboard[0] if leaderboard else {}
    verdict = "cross_sectional_lead_lag_diffusion_not_enough"
    if (
        selected
        and selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > selected["forward_random_p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "cross_sectional_lead_lag_diffusion_candidate_needs_modeling"
    out = {
        "experiment": "cross_sectional_lead_lag_diffusion_probe_v1",
        "method": "dynamic_correlation_neighbor_lead_lag_diffusion_soil_probe_without_external_groups",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "interval": INTERVAL, "pool_size": POOL_SIZE, "neighbor_ks": NEIGHBOR_KS, "top_k_replay": TOP_K_REPLAY, "random_trials": RANDOM_TRIALS, "seed": SEED, "candidates": CANDIDATES},
        "inputs_sha256": {"script": sha256(Path(__file__)), "cvr_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")},
        "results": results,
        "leaderboard": leaderboard,
        "selected_by_valid": selected,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected_by_valid": selected, "leaderboard_top12": leaderboard[:12], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
