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
OUT = ROOT / "daily_extreme_event_rebound_continuation_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

POOL_SIZES = (500, 1000)
INTERVALS = (3, 5)
TOP_KS = (10, 20)
TOP_K_REPLAY = 10
RANDOM_TRIALS = 200
SEED = 20260714

CANDIDATES = (
    "limit_break_reclaim_like",
    "failed_upper_shadow_reclaim",
    "volume_dry_after_breakout",
    "panic_rebound_liquidity_recover",
    "multi_day_acceleration_rest",
    "absorption_near_high_pullback",
    "event_composite",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_for_daily_extreme_event", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
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


def window(arr: np.ndarray, end_idx: int, cols: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    end = end_idx - offset
    start = max(0, end - length + 1)
    if end < 0:
        return np.full((0, len(cols)), np.nan)
    return arr[start:end + 1, :][:, cols].astype(float)


def pool_cols(market: dict[str, Any], amount20_arr: np.ndarray, idx: int, pool_size: int) -> list[int]:
    arrays = market["arrays"]
    valid = (
        np.isfinite(arrays["open"][idx]) & (arrays["open"][idx] > 0)
        & np.isfinite(arrays["close"][idx]) & (arrays["close"][idx] > 0)
        & np.isfinite(arrays["volume"][idx]) & (arrays["volume"][idx] > 0)
        & np.isfinite(amount20_arr[idx]) & (amount20_arr[idx] > 0)
    )
    cols = np.flatnonzero(valid)
    ordered = sorted(cols, key=lambda col: (-float(amount20_arr[idx, col]), str(market["symbols"][col])))[:pool_size]
    return [int(col) for col in ordered]


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


def event_frame(market: dict[str, Any], idx: int, cols: np.ndarray) -> dict[str, np.ndarray]:
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
    c3 = close[idx - 3, cols]
    c5 = close[idx - 5, cols]
    c10 = close[idx - 10, cols]
    c20 = close[idx - 20, cols]
    c60 = close[idx - 60, cols]
    ret1 = safe_div(c0, c1) - 1.0
    ret3 = safe_div(c0, c3) - 1.0
    ret5 = safe_div(c0, c5) - 1.0
    ret10 = safe_div(c0, c10) - 1.0
    ret20 = safe_div(c0, c20) - 1.0
    ret60 = safe_div(c0, c60) - 1.0
    open_gap = safe_div(o0, c1) - 1.0
    intraday = safe_div(c0, o0) - 1.0
    day_range = safe_div(h0, l0) - 1.0
    upper_shadow = safe_div(h0, np.maximum(o0, c0)) - 1.0
    lower_recover = safe_div(c0, l0) - 1.0
    high20_prev = np.nanmax(window(high, idx, cols, 20, offset=1), axis=0)
    high60 = np.nanmax(window(high, idx, cols, 60), axis=0)
    low10 = np.nanmin(window(low, idx, cols, 10), axis=0)
    low20 = np.nanmin(window(low, idx, cols, 20), axis=0)
    close5 = window(close, idx, cols, 5)
    high5 = window(high, idx, cols, 5)
    low5 = window(low, idx, cols, 5)
    volume5 = window(volume, idx, cols, 5)
    volume20 = window(volume, idx, cols, 20)
    amount = np.where(np.isfinite(vwap[idx, cols]) & (vwap[idx, cols] > 0), vwap[idx, cols], c0) * volume[idx, cols]
    amount20 = np.nanmean(np.where(np.isfinite(window(vwap, idx, cols, 20)) & (window(vwap, idx, cols, 20) > 0), window(vwap, idx, cols, 20), window(close, idx, cols, 20)) * volume20, axis=0)
    vol5 = np.nanmean(volume5, axis=0)
    vol20 = np.nanmean(volume20, axis=0)
    range5 = np.nanmean(safe_div(high5, low5) - 1.0, axis=0)
    limit_like_recent = np.nanmax(safe_div(close[idx - 4:idx + 1, :][:, cols], np.vstack([close[idx - 5:idx, :][:, cols]])).astype(float) - 1.0, axis=0)
    upper3_prev = np.nanmax(safe_div(high[idx - 3:idx, :][:, cols], np.maximum(open_[idx - 3:idx, :][:, cols], close[idx - 3:idx, :][:, cols])) - 1.0, axis=0)
    prev_failed = np.nanmax(safe_div(close[idx - 3:idx, :][:, cols], high[idx - 3:idx, :][:, cols]) - 1.0, axis=0)
    return {
        "ret1": ret1,
        "ret3": ret3,
        "ret5": ret5,
        "ret10": ret10,
        "ret20": ret20,
        "ret60": ret60,
        "open_gap": open_gap,
        "intraday": intraday,
        "day_range": day_range,
        "upper_shadow": upper_shadow,
        "lower_recover": lower_recover,
        "near_high20": safe_div(c0, high20_prev) - 1.0,
        "near_high60": safe_div(c0, high60) - 1.0,
        "drawup_from_low10": safe_div(c0, low10) - 1.0,
        "drawup_from_low20": safe_div(c0, low20) - 1.0,
        "close_vwap": safe_div(c0, vwap[idx, cols]) - 1.0,
        "vol5_vs_20": np.log(np.where(safe_div(vol5, vol20) > 0, safe_div(vol5, vol20), np.nan)),
        "amount_rank": finite_rank(amount20),
        "range5": range5,
        "limit_like_recent": limit_like_recent,
        "upper3_prev": upper3_prev,
        "prev_failed": prev_failed,
        "amount_today_vs20": np.log(np.where(safe_div(amount, amount20) > 0, safe_div(amount, amount20), np.nan)),
    }


def candidate_scores(frame: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = {name: finite_rank(values) for name, values in frame.items()}
    masks = {
        "limit_break_reclaim_like": (frame["limit_like_recent"] > 0.075) & (frame["near_high20"] > -0.035) & (frame["intraday"] > -0.025),
        "failed_upper_shadow_reclaim": (frame["upper3_prev"] > 0.045) & (frame["prev_failed"] < -0.025) & (frame["ret3"] > -0.015) & (frame["close_vwap"] > -0.015),
        "volume_dry_after_breakout": (frame["ret20"] > 0.08) & (frame["near_high20"] > -0.055) & (frame["vol5_vs_20"] < -0.15) & (frame["range5"] < 0.055),
        "panic_rebound_liquidity_recover": (frame["ret5"] < -0.045) & (frame["lower_recover"] > 0.035) & (frame["amount_today_vs20"] > 0.15),
        "multi_day_acceleration_rest": (frame["ret10"] > 0.10) & (frame["ret3"] > -0.025) & (frame["vol5_vs_20"] < 0.25) & (frame["near_high20"] > -0.06),
        "absorption_near_high_pullback": (frame["ret20"] > 0.08) & (frame["ret5"] < 0.03) & (frame["ret5"] > -0.08) & (frame["near_high60"] > -0.08) & (frame["lower_recover"] > 0.02),
    }
    scores = {
        "limit_break_reclaim_like": 0.28 * r["limit_like_recent"] + 0.24 * r["near_high20"] + 0.20 * r["lower_recover"] + 0.18 * r["amount_today_vs20"] - 0.10 * r["open_gap"],
        "failed_upper_shadow_reclaim": 0.28 * r["upper3_prev"] + 0.26 * r["ret3"] + 0.20 * r["close_vwap"] + 0.16 * r["lower_recover"] - 0.10 * r["upper_shadow"],
        "volume_dry_after_breakout": 0.32 * r["ret20"] + 0.22 * r["near_high20"] - 0.22 * r["vol5_vs_20"] - 0.16 * r["range5"] + 0.08 * r["amount_rank"],
        "panic_rebound_liquidity_recover": -0.26 * r["ret5"] + 0.30 * r["lower_recover"] + 0.24 * r["amount_today_vs20"] + 0.14 * r["intraday"] + 0.06 * r["amount_rank"],
        "multi_day_acceleration_rest": 0.30 * r["ret10"] + 0.20 * r["ret20"] + 0.18 * r["near_high20"] - 0.14 * r["vol5_vs_20"] - 0.10 * r["day_range"] + 0.08 * r["amount_rank"],
        "absorption_near_high_pullback": 0.26 * r["ret20"] - 0.22 * r["ret5"] + 0.22 * r["near_high60"] + 0.18 * r["lower_recover"] - 0.12 * r["vol5_vs_20"],
    }
    composite_mask = np.zeros_like(next(iter(scores.values())), dtype=bool)
    for mask in masks.values():
        composite_mask |= mask
    composite_score = 0.22 * scores["limit_break_reclaim_like"] + 0.18 * scores["failed_upper_shadow_reclaim"] + 0.18 * scores["volume_dry_after_breakout"] + 0.16 * scores["panic_rebound_liquidity_recover"] + 0.14 * scores["multi_day_acceleration_rest"] + 0.12 * scores["absorption_near_high_pullback"]
    masks["event_composite"] = composite_mask
    scores["event_composite"] = composite_score
    return {name: (scores[name], masks[name]) for name in CANDIDATES}


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, interval: int, pool_size: int) -> dict[str, Any]:
    selections = {name: {} for name in CANDIDATES}
    pools: dict[str, list[str]] = {}
    rows = []
    missing_future = 0
    for session in CVR.sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = np.asarray(pool_cols(market, amount20_arr, idx, pool_size), dtype=int)
        if len(cols) < max(TOP_KS):
            continue
        frame = event_frame(market, idx, cols)
        scored = candidate_scores(frame)
        y_by_col: dict[int, float] = {}
        for col in cols:
            y = future_return(market, idx, int(col), interval)
            if y is None:
                missing_future += 1
                continue
            y_by_col[int(col)] = float(y)
        if len(y_by_col) < max(TOP_KS):
            continue
        pools[session] = [str(market["symbols"][col]) for col in y_by_col]
        cand_stats = {}
        for name, (score, mask) in scored.items():
            local = [j for j, col in enumerate(cols) if int(col) in y_by_col and bool(mask[j])]
            ordered = sorted(local, key=lambda j: (-float(score[j]), str(market["symbols"][cols[j]])))
            picks = [str(market["symbols"][cols[j]]) for j in ordered[:TOP_K_REPLAY]]
            selections[name][session] = picks
            labels = np.asarray([y_by_col[int(cols[j])] for j in ordered], dtype=float)
            cand_stats[name] = {
                "event_count": len(local),
                "selected_count": len(picks),
                "top10": safe_mean(labels[:10]) if len(labels) else 0.0,
                "top20": safe_mean(labels[:20]) if len(labels) else 0.0,
            }
        rows.append({"session": session, "year": session[:4], "candidates": cand_stats})
    return {"selections": selections, "pools": pools, "rows": rows, "missing_future": missing_future, "sessions": len(rows)}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sessions": 0}
    out: dict[str, Any] = {"sessions": len(rows), "candidates": {}}
    for name in CANDIDATES:
        event_counts = [row["candidates"][name]["event_count"] for row in rows]
        selected_counts = [row["candidates"][name]["selected_count"] for row in rows]
        out["candidates"][name] = {
            "avg_event_count": safe_mean(event_counts),
            "median_event_count": float(np.median(event_counts)) if event_counts else 0.0,
            "active_rate_top10": safe_mean([count >= 10 for count in event_counts]),
            "avg_selected_count": safe_mean(selected_counts),
            "top10_mean": safe_mean([row["candidates"][name]["top10"] for row in rows]),
            "top20_mean": safe_mean([row["candidates"][name]["top20"] for row in rows]),
        }
    years: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        years.setdefault(row["year"], []).append(row)
    out["by_year"] = {year: aggregate_rows(part) for year, part in sorted(years.items()) if len(part) != len(rows)}
    return out


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator, active_counts: dict[str, int]) -> dict[str, list[str]]:
    out = {}
    for session, pool in pools.items():
        count = min(TOP_K_REPLAY, max(0, int(active_counts.get(session, TOP_K_REPLAY))), len(pool))
        if count > 0:
            out[session] = [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=count, replace=False)]
    return out


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def active_counts(selection: dict[str, list[str]]) -> dict[str, int]:
    return {session: len(items) for session, items in selection.items()}


def split_eval(split: dict[str, Any], market: dict[str, Any], start: str, end: str, interval: int, random_trials: int, seed_offset: int) -> dict[str, Any]:
    rows_summary = aggregate_rows(split["rows"])
    replays = {}
    randoms = {}
    rng = np.random.default_rng(SEED + seed_offset)
    for name in CANDIDATES:
        selection = split["selections"][name]
        replays[name] = CVR.replay_interval(selection, market, start, end, interval)["summary"]
        random_values = [CVR.replay_interval(random_selections(split["pools"], rng, active_counts(selection)), market, start, end, interval)["summary"]["final_multiple"] for _ in range(random_trials)]
        randoms[name] = percentile_summary(random_values)
    return {"session_stats": rows_summary, "replay": replays, "random_matched_count": randoms, "sessions": split["sessions"], "missing_future": split["missing_future"]}


def flatten(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, item in results.items():
        pool_size = int(item["pool_size"])
        interval = int(item["interval"])
        for name in CANDIDATES:
            train_stat = item["train"]["session_stats"]["candidates"][name]
            valid_stat = item["valid"]["session_stats"]["candidates"][name]
            dev_replay = item["dev"]["replay"][name]
            fwd_replay = item["forward"]["replay"][name]
            valid_replay = item["valid"]["replay"][name]
            fwd_random = item["forward"]["random_matched_count"][name]
            annual_values = dev_replay.get("annual_returns", {})
            rows.append({
                "key": key,
                "pool_size": pool_size,
                "interval": interval,
                "candidate": name,
                "train_top20": train_stat["top20_mean"],
                "valid_top20": valid_stat["top20_mean"],
                "train_avg_event_count": train_stat["avg_event_count"],
                "valid_avg_event_count": valid_stat["avg_event_count"],
                "valid_active_rate_top10": valid_stat["active_rate_top10"],
                "valid_multiple": valid_replay["final_multiple"],
                "dev_multiple": dev_replay["final_multiple"],
                "dev_all_years_positive": dev_replay["all_years_positive"],
                "dev_worst_year": min(annual_values.values(), default=0.0),
                "dev_max_drawdown": dev_replay["max_drawdown"],
                "forward_multiple": fwd_replay["final_multiple"],
                "forward_remove_best_3": fwd_replay["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "forward_avg_position": fwd_replay["avg_position_count"],
                "forward_max_drawdown": fwd_replay["max_drawdown"],
                "forward_random_p90": fwd_random["p90"],
                "forward_random_p95": fwd_random["p95"],
                "forward_beats_random_p95": fwd_replay["final_multiple"] > fwd_random["p95"],
                "score": math.log(max(valid_replay["final_multiple"], 1e-9)) + 0.30 * valid_stat["top20_mean"] + 0.20 * train_stat["top20_mean"] + 0.10 * min(valid_stat["avg_event_count"], 10.0) / 10.0,
            })
    return sorted(rows, key=lambda row: -row["score"])


def run_cell(market: dict[str, Any], amount20_arr: np.ndarray, pool_size: int, interval: int) -> dict[str, Any]:
    train = build_split(market, amount20_arr, TRAIN_START, TRAIN_END, interval, pool_size)
    valid = build_split(market, amount20_arr, VALID_START, VALID_END, interval, pool_size)
    dev = build_split(market, amount20_arr, DEV_START, DEV_END, interval, pool_size)
    forward = build_split(market, amount20_arr, FWD_START, FWD_END, interval, pool_size)
    return {
        "pool_size": pool_size,
        "interval": interval,
        "train": split_eval(train, market, TRAIN_START, TRAIN_END, interval, 30, pool_size + interval + 10),
        "valid": split_eval(valid, market, VALID_START, VALID_END, interval, 50, pool_size + interval + 20),
        "dev": split_eval(dev, market, DEV_START, DEV_END, interval, 50, pool_size + interval + 30),
        "forward": split_eval(forward, market, FWD_START, FWD_END, interval, RANDOM_TRIALS, pool_size + interval + 40),
    }


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    results = {}
    for pool_size in POOL_SIZES:
        for interval in INTERVALS:
            key = f"pool{pool_size}_interval{interval}"
            results[key] = run_cell(market, amount20_arr, pool_size, interval)
    leaderboard = flatten(results)
    selected = leaderboard[0] if leaderboard else {}
    verdict = "daily_extreme_event_rebound_continuation_not_enough"
    if (
        selected
        and selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > selected["forward_random_p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "daily_extreme_event_rebound_continuation_candidate_needs_modeling"
    out = {
        "experiment": "daily_extreme_event_rebound_continuation_v1",
        "method": "daily_event_state_soil_scan_with_train_valid_selection_dev_forward_replay_random_remove_best",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_sizes": POOL_SIZES,
            "intervals": INTERVALS,
            "top_ks": TOP_KS,
            "top_k_replay": TOP_K_REPLAY,
            "random_trials_forward": RANDOM_TRIALS,
            "seed": SEED,
            "candidates": CANDIDATES,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "cvr_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py"), "rt_script": sha256(ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py")},
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
