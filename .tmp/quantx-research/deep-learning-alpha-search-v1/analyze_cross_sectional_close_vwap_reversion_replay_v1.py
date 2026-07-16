from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import math
import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_close_vwap_reversion_replay_v1_summary.json"

DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
SEQ_LEN = 60
POOL_SIZE = 500
TOP_K = 10
RANDOM_TRIALS = 300
SEED = 20260714
INTERVALS = [3, 5]
VIEWS = ["last", "mean5", "mean20"]
VARIANTS = [
    "close_vwap_neg",
    "close_vwap_neg_rank_amount_pos",
    "close_vwap_neg_low_range",
    "close_vwap_neg_low_range_liquid",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RT = load_module("year_invariant_rank_transformer_v1_for_close_vwap_replay", ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py")
POS = RT.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_returns(curve: list[dict[str, Any]]) -> dict[str, float]:
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in curve:
        by_year.setdefault(row["date"][:4], []).append(row)
    return {year: rows[-1]["nav"] / rows[0]["nav_before"] - 1.0 for year, rows in sorted(by_year.items()) if rows[0]["nav_before"] > 0}


def summarize(curve: list[dict[str, Any]], trades: list[dict[str, Any]], rejects: dict[str, int]) -> dict[str, Any]:
    returns = [row["daily_return"] for row in curve]
    final = float(curve[-1]["nav"]) if curve else 1.0
    years = annual_returns(curve)
    n_years = max(1e-9, len(curve) / 242.0)
    annual_return = final ** (1.0 / n_years) - 1.0
    vol = pstdev(returns) * math.sqrt(252.0) if len(returns) > 1 else 0.0
    period_returns = sorted([float(row.get("period_return", 0.0)) for row in trades], reverse=True)
    remove = {}
    for n in (1, 3, 5, 10):
        kept = period_returns[n:]
        remove[f"remove_best_{n}"] = float(np.prod([1.0 + value for value in kept])) if kept else 1.0
    return {
        "final_multiple": final,
        "total_return": final - 1.0,
        "annual_return": annual_return,
        "annual_volatility": vol,
        "sharpe": annual_return / vol if vol > 0 else 0.0,
        "max_drawdown": min((row["drawdown"] for row in curve), default=0.0),
        "annual_returns": years,
        "all_years_positive": all(value > 0 for value in years.values()),
        "avg_position_count": mean(row["position_count"] for row in curve) if curve else 0.0,
        "max_position_count": max((row["position_count"] for row in curve), default=0),
        "trade_count": len(trades),
        "days": len(curve),
        "rejects": rejects,
        "remove_best_period_multiples": remove,
    }


def replay_interval(selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str, interval: int) -> dict[str, Any]:
    dates = market["dates"]
    arrays = market["arrays"]
    symbol_index = market["symbol_index"]
    active = sorted(day for day in selections if start <= day <= end)
    if not active:
        return {"summary": summarize([], [], {}), "trades_tail": []}
    start_idx = market["date_index"][active[0]] + 1
    end_bound = max(i for i, day in enumerate(dates) if day <= end)
    end_idx = min(end_bound, market["date_index"][active[-1]] + interval + 1)
    nav = 1.0
    peak = 1.0
    positions: dict[str, float] = {}
    last_rebalance_idx: int | None = None
    active_trade: int | None = None
    curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rejects = {"candidate_total": 0, "selected_total": 0, "missing_or_suspended": 0, "missing_preclose": 0, "price_jump_abs_gt_9p5": 0, "empty_rebalances": 0}
    for idx in range(start_idx, end_idx + 1):
        date = dates[idx]
        nav_before = nav
        if idx > start_idx and positions:
            daily_ret = 0.0
            for sym, weight in positions.items():
                col = symbol_index.get(sym)
                if col is None:
                    continue
                prev_open = float(arrays["open"][idx - 1, col])
                cur_open = float(arrays["open"][idx, col])
                if np.isfinite(prev_open) and np.isfinite(cur_open) and prev_open > 0 and cur_open > 0:
                    daily_ret += weight * (cur_open / prev_open - 1.0)
            nav *= 1.0 + daily_ret
        signal_date = dates[idx - 1]
        records = selections.get(signal_date)
        if records and (last_rebalance_idx is None or idx - last_rebalance_idx >= interval):
            if active_trade is not None and trades[active_trade].get("end_nav") is None:
                trades[active_trade]["end_nav"] = nav
                start_nav = trades[active_trade]["start_nav"]
                trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
            picked = []
            for sym in records[:TOP_K]:
                rejects["candidate_total"] += 1
                ok, reason = POS.tradable(sym, idx, idx - 1, market)
                if ok:
                    picked.append(sym)
                else:
                    rejects[reason] += 1
            target = {sym: 1.0 / len(picked) for sym in picked} if picked else {}
            if not target:
                rejects["empty_rebalances"] += 1
            rejects["selected_total"] += len(target)
            universe = set(target) | set(positions)
            buy = sum(max(0.0, target.get(sym, 0.0) - positions.get(sym, 0.0)) for sym in universe)
            sell = sum(max(0.0, positions.get(sym, 0.0) - target.get(sym, 0.0)) for sym in universe)
            nav *= max(0.0, 1.0 - buy * 0.00052 - sell * 0.00102)
            trades.append({"date": date, "signal_date": signal_date, "count": len(target), "start_nav": nav, "end_nav": None})
            active_trade = len(trades) - 1
            positions = target
            last_rebalance_idx = idx
        peak = max(peak, nav)
        curve.append({"date": date, "nav_before": nav_before, "nav": nav, "daily_return": nav / nav_before - 1.0 if nav_before > 0 else 0.0, "drawdown": nav / peak - 1.0, "position_count": len(positions)})
    if active_trade is not None and trades[active_trade].get("end_nav") is None:
        trades[active_trade]["end_nav"] = nav
        start_nav = trades[active_trade]["start_nav"]
        trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
    return {"summary": summarize(curve, trades, rejects), "trades_tail": trades[-5:]}


def sessions(market: dict[str, Any], start: str, end: str, interval: int) -> list[str]:
    out = []
    for day in market["dates"]:
        idx = market["date_index"][day]
        if start <= day <= end and idx >= SEQ_LEN + 20 and idx + interval + 1 < len(market["dates"]):
            out.append(day)
    return out[::interval]


def score_rows(view: np.ndarray) -> dict[str, np.ndarray]:
    close_vwap = view[:, 4]
    day_range = view[:, 3]
    rank_amount = view[:, 10]
    return {
        "close_vwap_neg": -close_vwap,
        "close_vwap_neg_rank_amount_pos": -close_vwap + 0.35 * rank_amount,
        "close_vwap_neg_low_range": -close_vwap - 0.35 * day_range,
        "close_vwap_neg_low_range_liquid": -close_vwap - 0.25 * day_range + 0.25 * rank_amount,
    }


def build_selections(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, interval: int) -> dict[str, Any]:
    selections = {view: {variant: {} for variant in VARIANTS} for view in VIEWS}
    pools: dict[str, list[str]] = {}
    for session in sessions(market, start, end, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)
        if len(cols) < TOP_K:
            continue
        pools[session] = [market["symbols"][col] for col in cols]
        views = {
            "last": feats[idx, cols, :],
            "mean5": feats[idx - 4:idx + 1, cols, :].mean(axis=0),
            "mean20": feats[idx - 19:idx + 1, cols, :].mean(axis=0),
        }
        for view_name, values in views.items():
            scores = score_rows(values)
            for variant, score in scores.items():
                ordered = sorted(range(len(cols)), key=lambda j: (-float(score[j]), str(market["symbols"][cols[j]])))[:TOP_K]
                selections[view_name][variant][session] = [market["symbols"][cols[j]] for j in ordered]
    return {"selections": selections, "pools": pools}


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def run_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, interval: int) -> dict[str, Any]:
    built = build_selections(market, feats, amount20_arr, start, end, interval)
    results = {}
    for view, by_variant in built["selections"].items():
        results[view] = {}
        for variant, selection in by_variant.items():
            results[view][variant] = replay_interval(selection, market, start, end, interval)["summary"]
    rng = np.random.default_rng(SEED + interval + (1 if start.startswith("2026") else 0))
    random_values = [replay_interval(random_selections(built["pools"], rng), market, start, end, interval)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return {"results": results, "random_pool500_top10": percentile_summary(random_values), "sessions": len(built["pools"])}


def flatten_leaderboard(dev: dict[str, Any], forward: dict[str, Any], interval: int) -> list[dict[str, Any]]:
    rows = []
    for view in VIEWS:
        for variant in VARIANTS:
            dev_summary = dev["results"][view][variant]
            fwd_summary = forward["results"][view][variant]
            rows.append({
                "interval": interval,
                "view": view,
                "variant": variant,
                "dev_multiple": dev_summary["final_multiple"],
                "dev_annual_returns": dev_summary["annual_returns"],
                "dev_all_years_positive": dev_summary["all_years_positive"],
                "dev_remove_best_3": dev_summary["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "dev_max_drawdown": dev_summary["max_drawdown"],
                "forward_multiple": fwd_summary["final_multiple"],
                "forward_total_return": fwd_summary["total_return"],
                "forward_remove_best_3": fwd_summary["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "forward_max_drawdown": fwd_summary["max_drawdown"],
                "forward_beats_random_p95": fwd_summary["final_multiple"] > forward["random_pool500_top10"]["p95"],
                "score": math.log(max(dev_summary["final_multiple"], 1e-9)) + 0.70 * math.log(max(fwd_summary["final_multiple"], 1e-9)) + 0.20 * min(dev_summary["annual_returns"].values(), default=-1.0) - abs(dev_summary["max_drawdown"]),
            })
    return sorted(rows, key=lambda row: -row["score"])


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
    verdict = "close_vwap_reversion_replay_not_enough"
    if best["dev_multiple"] >= 20.0 and best["dev_all_years_positive"] and best["forward_beats_random_p95"] and best["forward_multiple"] > 1.2 and best["forward_remove_best_3"] > 1.0:
        verdict = "close_vwap_reversion_replay_candidate_needs_modeling"
    out = {
        "experiment": "cross_sectional_close_vwap_reversion_replay_v1",
        "method": "real_replay_for_3d_5d_close_vwap_negative_reversion_soil_with_cost_random_remove_best",
        "params": {"dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "seq_len": SEQ_LEN, "pool_size": POOL_SIZE, "top_k": TOP_K, "intervals": INTERVALS, "views": VIEWS, "variants": VARIANTS, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "horizon_surface_script": sha256(ROOT / "analyze_cross_sectional_multi_horizon_label_surface_diagnostic_v1.py"), "horizon_surface_summary": sha256(ROOT / "cross_sectional_multi_horizon_label_surface_diagnostic_v1_summary.json") if (ROOT / "cross_sectional_multi_horizon_label_surface_diagnostic_v1_summary.json").exists() else None},
        "intervals": intervals,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "leaderboard_top8": leaderboard[:8], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
