from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "daily_limit_up_break_board_state_scan_v1_summary.json"

CANDIDATES = (
    "first_limit_follow_through",
    "two_board_continuation",
    "break_board_reclaim",
    "failed_board_low_absorb",
    "post_limit_volume_dry_hold",
    "limit_pullback_relaunch",
    "limit_state_composite",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R79 = load_module("daily_extreme_event_rebound_continuation_for_limit_state", ROOT / "analyze_daily_extreme_event_rebound_continuation_v1.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return R79.safe_div(a, b) - 1.0


def w(arr: np.ndarray, idx: int, cols: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    return R79.window(arr, idx, cols, length, offset=offset)


def limit_event_frame(market: dict[str, Any], idx: int, cols: np.ndarray) -> dict[str, np.ndarray]:
    arrays = market["arrays"]
    open_ = arrays["open"].astype(float)
    high = arrays["high"].astype(float)
    low = arrays["low"].astype(float)
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    c0 = close[idx, cols]
    o0 = open_[idx, cols]
    h0 = high[idx, cols]
    l0 = low[idx, cols]
    c1 = close[idx - 1, cols]
    c2 = close[idx - 2, cols]
    c3 = close[idx - 3, cols]
    c5 = close[idx - 5, cols]
    c10 = close[idx - 10, cols]
    c20 = close[idx - 20, cols]
    ret1 = pct(c0, c1)
    ret2 = pct(c0, c2)
    ret3 = pct(c0, c3)
    ret5 = pct(c0, c5)
    ret10 = pct(c0, c10)
    ret20 = pct(c0, c20)
    close_limit = ret1 >= 0.092
    high_limit = pct(h0, c1) >= 0.092
    failed_board = high_limit & ~close_limit
    prev_ret1 = pct(c1, c2)
    prev2_ret1 = pct(c2, c3)
    prev_limit = prev_ret1 >= 0.092
    prev2_limit = prev2_ret1 >= 0.092
    limit2 = close_limit & prev_limit
    limit3 = close_limit & prev_limit & prev2_limit
    recent_limit_count = np.sum(pct(close[idx - 4:idx + 1, :][:, cols], close[idx - 5:idx, :][:, cols]) >= 0.092, axis=0).astype(float)
    recent_failed_count = np.sum(pct(high[idx - 4:idx + 1, :][:, cols], close[idx - 5:idx, :][:, cols]) >= 0.092, axis=0).astype(float) - recent_limit_count
    high20_prev = np.nanmax(w(high, idx, cols, 20, offset=1), axis=0)
    high60 = np.nanmax(w(high, idx, cols, 60), axis=0)
    low5 = np.nanmin(w(low, idx, cols, 5), axis=0)
    low10 = np.nanmin(w(low, idx, cols, 10), axis=0)
    vol5 = np.nanmean(w(volume, idx, cols, 5), axis=0)
    vol20 = np.nanmean(w(volume, idx, cols, 20), axis=0)
    amount = np.where(np.isfinite(vwap[idx, cols]) & (vwap[idx, cols] > 0), vwap[idx, cols], c0) * volume[idx, cols]
    amount20 = np.nanmean(np.where(np.isfinite(w(vwap, idx, cols, 20)) & (w(vwap, idx, cols, 20) > 0), w(vwap, idx, cols, 20), w(close, idx, cols, 20)) * w(volume, idx, cols, 20), axis=0)
    return {
        "ret1": ret1,
        "ret2": ret2,
        "ret3": ret3,
        "ret5": ret5,
        "ret10": ret10,
        "ret20": ret20,
        "intraday": pct(c0, o0),
        "open_gap": pct(o0, c1),
        "day_range": pct(h0, l0),
        "close_vwap": pct(c0, vwap[idx, cols]),
        "lower_recover": pct(c0, l0),
        "upper_giveback": pct(c0, h0),
        "near_high20": pct(c0, high20_prev),
        "near_high60": pct(c0, high60),
        "drawup_low5": pct(c0, low5),
        "drawup_low10": pct(c0, low10),
        "vol5_vs_20": np.log(np.where(R79.safe_div(vol5, vol20) > 0, R79.safe_div(vol5, vol20), np.nan)),
        "amount_today_vs20": np.log(np.where(R79.safe_div(amount, amount20) > 0, R79.safe_div(amount, amount20), np.nan)),
        "amount_rank": R79.finite_rank(amount20),
        "close_limit": close_limit.astype(float),
        "high_limit": high_limit.astype(float),
        "failed_board": failed_board.astype(float),
        "prev_limit": prev_limit.astype(float),
        "prev2_limit": prev2_limit.astype(float),
        "limit2": limit2.astype(float),
        "limit3": limit3.astype(float),
        "recent_limit_count": recent_limit_count,
        "recent_failed_count": recent_failed_count,
    }


def limit_candidate_scores(frame: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = {name: R79.finite_rank(values) for name, values in frame.items()}
    close_limit = frame["close_limit"] > 0.5
    high_limit = frame["high_limit"] > 0.5
    failed_board = frame["failed_board"] > 0.5
    prev_limit = frame["prev_limit"] > 0.5
    prev2_limit = frame["prev2_limit"] > 0.5
    masks = {
        "first_limit_follow_through": close_limit & ~prev_limit & (frame["ret20"] > -0.10) & (frame["amount_today_vs20"] > -0.20),
        "two_board_continuation": close_limit & prev_limit & ~prev2_limit & (frame["open_gap"] < 0.07),
        "break_board_reclaim": failed_board & prev_limit & (frame["upper_giveback"] > -0.055) & (frame["close_vwap"] > -0.025),
        "failed_board_low_absorb": failed_board & (frame["lower_recover"] > 0.035) & (frame["amount_today_vs20"] > 0.10),
        "post_limit_volume_dry_hold": prev_limit & ~close_limit & (frame["ret1"] > -0.055) & (frame["near_high20"] > -0.09) & (frame["vol5_vs_20"] < 0.10),
        "limit_pullback_relaunch": (frame["recent_limit_count"] >= 1.0) & (frame["ret5"] < 0.04) & (frame["ret5"] > -0.12) & high_limit & (frame["lower_recover"] > 0.025),
    }
    scores = {
        "first_limit_follow_through": 0.32 * r["ret1"] + 0.22 * r["amount_today_vs20"] + 0.18 * r["near_high20"] + 0.16 * r["close_vwap"] + 0.12 * r["amount_rank"],
        "two_board_continuation": 0.34 * r["ret2"] + 0.20 * r["amount_today_vs20"] + 0.18 * r["near_high20"] - 0.16 * r["open_gap"] + 0.12 * r["amount_rank"],
        "break_board_reclaim": 0.30 * r["upper_giveback"] + 0.24 * r["close_vwap"] + 0.18 * r["amount_today_vs20"] + 0.16 * r["near_high20"] + 0.12 * r["lower_recover"],
        "failed_board_low_absorb": 0.30 * r["lower_recover"] + 0.24 * r["amount_today_vs20"] + 0.18 * r["upper_giveback"] + 0.16 * r["intraday"] + 0.12 * r["amount_rank"],
        "post_limit_volume_dry_hold": 0.26 * r["near_high20"] - 0.24 * r["vol5_vs_20"] + 0.20 * r["ret1"] + 0.18 * r["close_vwap"] + 0.12 * r["amount_rank"],
        "limit_pullback_relaunch": 0.26 * r["recent_limit_count"] + 0.24 * r["high_limit"] - 0.18 * r["ret5"] + 0.18 * r["lower_recover"] + 0.14 * r["amount_today_vs20"],
    }
    composite_mask = np.zeros_like(frame["ret1"], dtype=bool)
    for mask in masks.values():
        composite_mask |= mask
    scores["limit_state_composite"] = 0.18 * scores["first_limit_follow_through"] + 0.18 * scores["two_board_continuation"] + 0.18 * scores["break_board_reclaim"] + 0.16 * scores["failed_board_low_absorb"] + 0.16 * scores["post_limit_volume_dry_hold"] + 0.14 * scores["limit_pullback_relaunch"]
    masks["limit_state_composite"] = composite_mask
    return {name: (scores[name], masks[name]) for name in CANDIDATES}


def main() -> None:
    started = time.perf_counter()
    R79.CANDIDATES = CANDIDATES
    R79.event_frame = limit_event_frame
    R79.candidate_scores = limit_candidate_scores
    market = R79.POS.load_market()
    amount20_arr = R79.RT.amount20(market)
    results = {}
    for pool_size in R79.POOL_SIZES:
        for interval in R79.INTERVALS:
            key = f"pool{pool_size}_interval{interval}"
            results[key] = R79.run_cell(market, amount20_arr, pool_size, interval)
    leaderboard = R79.flatten(results)
    selected = leaderboard[0] if leaderboard else {}
    verdict = "daily_limit_up_break_board_state_not_enough"
    if (
        selected
        and selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > selected["forward_random_p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "daily_limit_up_break_board_state_candidate_needs_modeling"
    out = {
        "experiment": "daily_limit_up_break_board_state_scan_v1",
        "method": "daily_limit_up_break_board_state_approximation_soil_scan_reusing_round79_strict_replay",
        "params": {
            "train": [R79.TRAIN_START, R79.TRAIN_END],
            "valid": [R79.VALID_START, R79.VALID_END],
            "dev": [R79.DEV_START, R79.DEV_END],
            "forward": [R79.FWD_START, R79.FWD_END],
            "pool_sizes": R79.POOL_SIZES,
            "intervals": R79.INTERVALS,
            "top_k_replay": R79.TOP_K_REPLAY,
            "random_trials_forward": R79.RANDOM_TRIALS,
            "seed": R79.SEED,
            "candidates": CANDIDATES,
            "limit_close_threshold": 0.092,
            "limit_high_threshold": 0.092,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "round79_script": sha256(ROOT / "analyze_daily_extreme_event_rebound_continuation_v1.py")},
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
