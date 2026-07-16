from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
R74_PATH = ROOT / "analyze_cross_sectional_daily_path_memory_residual_v1.py"
OUT = ROOT / "liquidity_tier_daily_path_residual_v1_summary.json"

SEED = 20260714
EPOCHS = 24
RANDOM_TRIALS = 160

TIERS = {
    "pool_top500": (0, 500),
    "pool_501_1000": (500, 1000),
    "pool_1001_2000": (1000, 2000),
    "pool_2001_3500": (2000, 3500),
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R74 = load_module("daily_path_residual_v1_for_liquidity_tier", R74_PATH)
POS = R74.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def amount_order_for_session(market: dict[str, Any], amount20_arr: np.ndarray, idx: int) -> list[int]:
    arrays = market["arrays"]
    valid = (
        np.isfinite(arrays["open"][idx]) & (arrays["open"][idx] > 0)
        & np.isfinite(arrays["close"][idx]) & (arrays["close"][idx] > 0)
        & np.isfinite(arrays["volume"][idx]) & (arrays["volume"][idx] > 0)
        & np.isfinite(amount20_arr[idx]) & (amount20_arr[idx] > 0)
    )
    cols = np.flatnonzero(valid)
    return [int(col) for col in sorted(cols, key=lambda col: (-float(amount20_arr[idx, col]), str(market["symbols"][col])))]


def build_split_tier(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, lo: int, hi: int) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    raw_y: list[float] = []
    label_rank: list[float] = []
    sessions: list[str] = []
    symbols: list[str] = []
    base_scores: list[float] = []
    null_vars: list[tuple[float, float, float, float]] = []
    pools: dict[str, list[str]] = {}
    diagnostics = {"sessions_considered": 0, "sessions_kept": 0, "future_return_missing": 0, "candidate_rows": 0}
    for session in R74.RT.due_sessions(market, start, end):
        idx = market["date_index"][session]
        if idx < R74.SEQ_LEN + 20:
            continue
        diagnostics["sessions_considered"] += 1
        ordered = amount_order_for_session(market, amount20_arr, idx)
        cols = ordered[lo:hi]
        if len(cols) < max(R74.TOP_KS):
            continue
        x_all, base_all = R74.build_path_features(feats, idx, cols)
        records = []
        for local_pos, col in enumerate(cols):
            y = R74.RT.future_return(market, idx, col)
            if y is None:
                diagnostics["future_return_missing"] += 1
                continue
            records.append((local_pos, col, float(y)))
        if len(records) < max(R74.TOP_KS):
            continue
        diagnostics["sessions_kept"] += 1
        diagnostics["candidate_rows"] += len(records)
        returns = np.asarray([r[2] for r in records], dtype=float)
        ranks = R74.pd.Series(returns).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        pools[session] = [str(market["symbols"][col]) for _, col, _ in records]
        for rank_value, (local_pos, col, y) in zip(ranks, records, strict=True):
            x = x_all[local_pos]
            xs.append(x)
            raw_y.append(y)
            label_rank.append(float(rank_value))
            sessions.append(session)
            symbols.append(str(market["symbols"][col]))
            base_scores.append(float(base_all[local_pos]))
            last = feats[idx, col]
            null_vars.append((float(last[10]), float(last[9]), float(last[3]), float(last[4])))
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "raw_y": np.asarray(raw_y, dtype=np.float32),
        "label_rank": np.asarray(label_rank, dtype=np.float32),
        "sessions": sessions,
        "symbols": symbols,
        "base_score": np.asarray(base_scores, dtype=np.float32),
        "null_vars": np.asarray(null_vars, dtype=np.float32),
        "pools": pools,
        "diagnostics": diagnostics,
    }


def replay(split: dict[str, Any], scores: np.ndarray, market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return POS.replay(R74.selections_from_scores(split, scores), market, start, end)["summary"]


def random_summary(split: dict[str, Any], market: dict[str, Any], start: str, end: str, trials: int) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [POS.replay(R74.random_selections(split["pools"], rng), market, start, end)["summary"]["final_multiple"] for _ in range(trials)]
    return R74.percentile_summary(values)


def evaluate_tier(name: str, bounds: tuple[int, int], market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray) -> dict[str, Any]:
    lo, hi = bounds
    train = build_split_tier(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END, lo, hi)
    valid = build_split_tier(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END, lo, hi)
    dev = R74.merge_splits(train, valid)
    forward = build_split_tier(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END, lo, hi)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    if len(train["label_rank"]) == 0 or len(valid["label_rank"]) == 0 or len(forward["label_rank"]) == 0:
        return {"tier": name, "bounds": bounds, "error": "empty_split", "diagnostics": {k: v.get("diagnostics", {}) for k, v in splits.items()}}
    null_scores, null_meta = R74.build_null(train, splits)
    old_epochs = R74.EPOCHS
    R74.EPOCHS = EPOCHS
    try:
        model, history = R74.train_residual(train, valid, null_scores)
    finally:
        R74.EPOCHS = old_epochs
    resid = {split_name: R74.predict(model, split["x"]) for split_name, split in splits.items()}
    metrics = {split_name: R74.metric_from_scores(split, resid[split_name]) for split_name, split in splits.items()}
    base_metrics = {split_name: R74.metric_from_scores(split, split["base_score"]) for split_name, split in splits.items()}
    selected_replay = {
        "dev": replay(dev, resid["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": replay(forward, resid["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    base_replay = {
        "dev": replay(dev, dev["base_score"], market, R74.DEV_START, R74.DEV_END),
        "forward": replay(forward, forward["base_score"], market, R74.FWD_START, R74.FWD_END),
    }
    fwd_random = random_summary(forward, market, R74.FWD_START, R74.FWD_END, RANDOM_TRIALS)
    return {
        "tier": name,
        "bounds": bounds,
        "sample_counts": {split_name: int(len(split["label_rank"])) for split_name, split in splits.items()},
        "session_counts": {split_name: int(len(set(split["sessions"]))) for split_name, split in splits.items()},
        "diagnostics": {split_name: split.get("diagnostics", {}) for split_name, split in splits.items()},
        "null_meta": null_meta,
        "history_tail": history[-5:],
        "metrics": metrics,
        "base_metrics": base_metrics,
        "selected_replay": selected_replay,
        "base_replay": base_replay,
        "forward_random": fwd_random,
        "gates": {
            "dev_all_years_positive": selected_replay["dev"]["all_years_positive"],
            "forward_beats_base": selected_replay["forward"]["final_multiple"] > base_replay["forward"]["final_multiple"],
            "forward_beats_random_p90": selected_replay["forward"]["final_multiple"] > fwd_random["p90"],
            "forward_beats_random_p95": selected_replay["forward"]["final_multiple"] > fwd_random["p95"],
            "forward_remove_best_3_positive": selected_replay["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0,
            "avg_position_gt5": selected_replay["forward"]["avg_position_count"] > 5,
        },
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    results = [evaluate_tier(name, bounds, market, feats, amount20_arr) for name, bounds in TIERS.items()]
    leaderboard = sorted(
        [
            {
                "tier": item["tier"],
                "bounds": item["bounds"],
                "dev_multiple": item.get("selected_replay", {}).get("dev", {}).get("final_multiple", 0.0),
                "dev_all_years_positive": item.get("selected_replay", {}).get("dev", {}).get("all_years_positive", False),
                "forward_multiple": item.get("selected_replay", {}).get("forward", {}).get("final_multiple", 0.0),
                "forward_remove_best_3": item.get("selected_replay", {}).get("forward", {}).get("remove_best_period_multiples", {}).get("remove_best_3", 0.0),
                "forward_top10": item.get("metrics", {}).get("forward", {}).get("top_returns", {}).get("top10", 0.0),
                "forward_top20": item.get("metrics", {}).get("forward", {}).get("top_returns", {}).get("top20", 0.0),
                "forward_ic": item.get("metrics", {}).get("forward", {}).get("mean_rank_ic", 0.0),
                "forward_avg_position": item.get("selected_replay", {}).get("forward", {}).get("avg_position_count", 0.0),
                "forward_random_p90": item.get("forward_random", {}).get("p90", 0.0),
                "forward_random_p95": item.get("forward_random", {}).get("p95", 0.0),
                "gates": item.get("gates", {}),
            }
            for item in results
            if "error" not in item
        ],
        key=lambda row: (row["dev_all_years_positive"], row["forward_multiple"], row["forward_remove_best_3"]),
        reverse=True,
    )
    verdict = "liquidity_tier_daily_path_residual_not_enough"
    best = leaderboard[0] if leaderboard else {}
    if best and best["dev_all_years_positive"] and best["forward_multiple"] > best["forward_random_p95"] and best["forward_remove_best_3"] > 1.0 and best["forward_avg_position"] > 5:
        verdict = "liquidity_tier_daily_path_residual_candidate_needs_walkforward"
    out = {
        "experiment": "liquidity_tier_daily_path_residual_v1",
        "method": "same_round74_daily_path_residual_across_amount20_liquidity_tiers",
        "params": {"tiers": TIERS, "epochs": EPOCHS, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round74_script": sha256(R74_PATH), "round77_summary": sha256(ROOT / "daily_path_opportunity_filter_v1_summary.json")},
        "leaderboard": leaderboard,
        "results": results,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "leaderboard": leaderboard, "best": best, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
