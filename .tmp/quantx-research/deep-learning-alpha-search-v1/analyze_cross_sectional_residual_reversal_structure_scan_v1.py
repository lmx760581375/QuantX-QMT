from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_residual_reversal_structure_scan_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_K = 10
POOL_SIZE = 500
RANDOM_TRIALS = 300
SEED = 20260714
INTERVALS = [3, 5]
VIEWS = ["mean20", "mean5", "last"]
GROUPS = ["liquidity", "volatility", "momentum20", "reversal5", "crowding"]
VARIANTS = [
    "global_close_vwap_neg",
    "liquidity_bucket_close_vwap_resid",
    "volatility_bucket_close_vwap_resid",
    "momentum20_bucket_close_vwap_resid",
    "reversal5_bucket_close_vwap_resid",
    "crowding_bucket_close_vwap_resid",
    "multi_bucket_close_vwap_resid_mean",
    "close_vwap_resid_low_range",
    "close_vwap_resid_liquid_low_range",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_residual_scan", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_rank(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    order = np.argsort(np.where(np.isfinite(clean), clean, np.nanmedian(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 0.0), kind="mergesort")
    ranks = np.empty(len(clean), dtype=float)
    ranks[order] = np.linspace(-0.5, 0.5, len(clean), endpoint=True) if len(clean) > 1 else 0.0
    return ranks


def quantile_codes(values: np.ndarray, buckets: int = 5) -> np.ndarray:
    ranks = finite_rank(values)
    codes = np.floor((ranks + 0.5) * buckets).astype(int)
    return np.clip(codes, 0, buckets - 1)


def bucket_residual(base_score: np.ndarray, group_values: np.ndarray, buckets: int = 5) -> np.ndarray:
    base = np.asarray(base_score, dtype=float)
    codes = quantile_codes(group_values, buckets=buckets)
    out = np.zeros(len(base), dtype=float)
    for code in range(buckets):
        mask = codes == code
        if not np.any(mask):
            continue
        local = base[mask]
        center = float(np.nanmean(local)) if np.isfinite(local).any() else 0.0
        out[mask] = local - center
    return out


def score_variants(view: np.ndarray) -> dict[str, np.ndarray]:
    close_vwap = view[:, 4]
    day_range = view[:, 3]
    ret5 = view[:, 6]
    ret20 = view[:, 7]
    rank_amount = view[:, 10]
    rank_ret20 = view[:, 9]
    base = -close_vwap
    groups = {
        "liquidity": rank_amount,
        "volatility": day_range,
        "momentum20": ret20,
        "reversal5": ret5,
        "crowding": 0.45 * finite_rank(rank_amount) + 0.35 * finite_rank(rank_ret20) + 0.20 * finite_rank(day_range),
    }
    residuals = {name: bucket_residual(base, values) for name, values in groups.items()}
    multi = np.mean(np.vstack([residuals[name] for name in GROUPS]), axis=0)
    return {
        "global_close_vwap_neg": base,
        "liquidity_bucket_close_vwap_resid": residuals["liquidity"],
        "volatility_bucket_close_vwap_resid": residuals["volatility"],
        "momentum20_bucket_close_vwap_resid": residuals["momentum20"],
        "reversal5_bucket_close_vwap_resid": residuals["reversal5"],
        "crowding_bucket_close_vwap_resid": residuals["crowding"],
        "multi_bucket_close_vwap_resid_mean": multi,
        "close_vwap_resid_low_range": multi - 0.25 * finite_rank(day_range),
        "close_vwap_resid_liquid_low_range": multi - 0.20 * finite_rank(day_range) + 0.15 * finite_rank(rank_amount),
    }


def build_selections(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, interval: int) -> dict[str, Any]:
    selections = {view: {variant: {} for variant in VARIANTS} for view in VIEWS}
    pools: dict[str, list[str]] = {}
    for session in CVR.sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)
        if len(cols) < TOP_K:
            continue
        cols = cols[:POOL_SIZE]
        pools[session] = [market["symbols"][col] for col in cols]
        views = {
            "last": feats[idx, cols, :],
            "mean5": feats[idx - 4:idx + 1, cols, :].mean(axis=0),
            "mean20": feats[idx - 19:idx + 1, cols, :].mean(axis=0),
        }
        for view_name, values in views.items():
            scores = score_variants(values)
            for variant, score in scores.items():
                ordered = sorted(range(len(cols)), key=lambda j: (-float(score[j]), str(market["symbols"][cols[j]])))[:TOP_K]
                selections[view_name][variant][session] = [market["symbols"][cols[j]] for j in ordered]
    return {"selections": selections, "pools": pools}


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def run_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, interval: int) -> dict[str, Any]:
    built = build_selections(market, feats, amount20_arr, start, end, interval)
    results = {}
    for view, by_variant in built["selections"].items():
        results[view] = {}
        for variant, selection in by_variant.items():
            results[view][variant] = CVR.replay_interval(selection, market, start, end, interval)["summary"]
    rng = np.random.default_rng(SEED + interval + (1 if start.startswith("2026") else 0))
    random_values = [CVR.replay_interval(random_selections(built["pools"], rng), market, start, end, interval)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return {"results": results, "random_pool500_top10": percentile_summary(random_values), "sessions": len(built["pools"])}


def flatten_leaderboard(dev: dict[str, Any], forward: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for view in VIEWS:
        for variant in VARIANTS:
            dev_summary = dev["results"][view][variant]
            fwd_summary = forward["results"][view][variant]
            min_year = min(dev_summary["annual_returns"].values(), default=-1.0)
            rows.append({
                "interval": interval,
                "view": view,
                "variant": variant,
                "dev_multiple": dev_summary["final_multiple"],
                "dev_annual_returns": dev_summary["annual_returns"],
                "dev_all_years_positive": dev_summary["all_years_positive"],
                "dev_min_year_return": min_year,
                "dev_remove_best_3": dev_summary["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "dev_max_drawdown": dev_summary["max_drawdown"],
                "forward_multiple": fwd_summary["final_multiple"],
                "forward_total_return": fwd_summary["total_return"],
                "forward_remove_best_3": fwd_summary["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "forward_max_drawdown": fwd_summary["max_drawdown"],
                "forward_beats_random_p95": fwd_summary["final_multiple"] > forward["random_pool500_top10"]["p95"],
                "score": math.log(max(dev_summary["final_multiple"], 1e-9)) + 0.75 * math.log(max(fwd_summary["final_multiple"], 1e-9)) + 0.50 * min_year - abs(dev_summary["max_drawdown"]) + 0.20 * math.log(max(fwd_summary["remove_best_period_multiples"].get("remove_best_3", 1e-9), 1e-9)),
            })
    return sorted(rows, key=lambda row: -row["score"])


def best_by_variant(leaderboard: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in leaderboard:
        out.setdefault(row["variant"], row)
    return out


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    intervals = {}
    leaderboard = []
    for interval in INTERVALS:
        dev = run_split(market, feats, amount20_arr, DEV_START, DEV_END, interval)
        forward = run_split(market, feats, amount20_arr, FWD_START, FWD_END, interval)
        intervals[str(interval)] = {"dev": dev, "forward": forward}
        leaderboard.extend(flatten_leaderboard(dev, forward, interval))
    leaderboard = sorted(leaderboard, key=lambda row: -row["score"])
    best = leaderboard[0]
    baseline_rows = [row for row in leaderboard if row["variant"] == "global_close_vwap_neg" and row["view"] == "mean20"]
    baseline_rows = sorted(baseline_rows, key=lambda row: -row["score"])
    verdict = "residual_reversal_structure_not_enough"
    if best["variant"] != "global_close_vwap_neg" and best["dev_all_years_positive"] and best["forward_beats_random_p95"] and best["forward_multiple"] > 1.15 and best["forward_remove_best_3"] > 1.0 and best["dev_multiple"] >= 3.0:
        verdict = "residual_reversal_structure_candidate_needs_model_probe"
    out = {
        "experiment": "cross_sectional_residual_reversal_structure_scan_v1",
        "method": "bucket_residual_close_vwap_reversal_scan_with_real_replay_random_remove_best",
        "params": {
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_k": TOP_K,
            "intervals": INTERVALS,
            "views": VIEWS,
            "groups": GROUPS,
            "variants": VARIANTS,
            "random_trials": RANDOM_TRIALS,
            "seed": SEED,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "close_vwap_replay_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py"),
            "close_vwap_replay_summary": sha256(ROOT / "cross_sectional_close_vwap_reversion_replay_v1_summary.json") if (ROOT / "cross_sectional_close_vwap_reversion_replay_v1_summary.json").exists() else None,
            "regime_model_summary": sha256(ROOT / "cross_sectional_close_vwap_regime_model_probe_v1_summary.json") if (ROOT / "cross_sectional_close_vwap_regime_model_probe_v1_summary.json").exists() else None,
        },
        "intervals": intervals,
        "leaderboard": leaderboard,
        "best": best,
        "best_by_variant": best_by_variant(leaderboard),
        "baseline_mean20_global_rows": baseline_rows,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "baseline_mean20_global_rows": baseline_rows, "leaderboard_top12": leaderboard[:12], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
