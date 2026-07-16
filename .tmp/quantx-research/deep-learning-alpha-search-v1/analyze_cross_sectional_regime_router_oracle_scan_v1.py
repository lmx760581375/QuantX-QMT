from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_regime_router_oracle_scan_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_K = 10
REBALANCE_INTERVAL = 5

BASE_VARIANTS = (
    "low_vol_uptrend",
    "vol_compression_breakout",
    "panic_exhaustion",
    "inverse_liquidity_momentum",
    "quiet_pullback",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POS = load_module("multihorizon_v1", ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py")
INV = load_module("inverse_crowding_v1", ROOT / "analyze_cross_sectional_inverse_crowding_reversal_scan_v1.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_period_return(market: dict[str, Any], session: str, symbols: list[str]) -> float:
    signal_idx = market["date_index"][session]
    entry_idx = signal_idx + 1
    if entry_idx >= len(market["dates"]):
        return 0.0
    exit_idx = min(len(market["dates"]) - 1, signal_idx + REBALANCE_INTERVAL + 1)
    picked = []
    for sym in symbols[:TOP_K]:
        ok, _ = POS.tradable(sym, entry_idx, signal_idx, market)
        if ok:
            picked.append(sym)
    if not picked:
        return 0.0
    rets = []
    for sym in picked:
        col = market["symbol_index"][sym]
        entry = float(market["arrays"]["open"][entry_idx, col])
        exit_ = float(market["arrays"]["open"][exit_idx, col])
        if np.isfinite(entry) and np.isfinite(exit_) and entry > 0 and exit_ > 0:
            rets.append(exit_ / entry - 1.0)
    if not rets:
        return 0.0
    return float(np.mean(rets) - 0.00052 - 0.00102)


def rank_rows(rows: list[dict[str, Any]], variant: str) -> list[str]:
    key = f"score_{variant}"
    ranked = sorted(rows, key=lambda row: (-float(row.get(key, 0.0)), str(row["symbol"])))[:TOP_K]
    return [row["symbol"] for row in ranked]


def regime_features(market: dict[str, Any], session: str) -> dict[str, float]:
    idx = market["date_index"][session]
    close = market["arrays"]["close"]
    volume = market["arrays"]["volume"]
    ret5 = POS.safe_div(close[idx], close[idx - 5]) - 1.0
    ret20 = POS.safe_div(close[idx], close[idx - 20]) - 1.0
    ret60 = POS.safe_div(close[idx], close[idx - 60]) - 1.0
    daily_ret20 = POS.safe_div(close[idx - 19:idx + 1], close[idx - 20:idx]) - 1.0
    vol20 = np.nanstd(daily_ret20, axis=0)
    vol_ma20 = np.nanmean(volume[idx - 19:idx + 1], axis=0)
    vol_ratio20 = POS.safe_div(volume[idx], vol_ma20)
    valid = np.isfinite(ret20)
    return {
        "median_ret5": float(np.nanmedian(ret5[valid])),
        "median_ret20": float(np.nanmedian(ret20[valid])),
        "median_ret60": float(np.nanmedian(ret60[valid])),
        "breadth20": float(np.nanmean(ret20[valid] > 0)),
        "breadth5": float(np.nanmean(ret5[valid] > 0)),
        "median_vol20": float(np.nanmedian(vol20[valid])),
        "dispersion20": float(np.nanstd(ret20[valid])),
        "median_vol_ratio20": float(np.nanmedian(vol_ratio20[valid])),
    }


def build_tables(market: dict[str, Any]) -> dict[str, Any]:
    dates = market["dates"]
    sessions = [day for day in dates if DEV_START <= day <= FWD_END and market["date_index"][day] >= 80 and market["date_index"][day] + 1 < len(dates)]
    due5 = sessions[::REBALANCE_INTERVAL]
    selections: dict[str, dict[str, list[str]]] = {variant: {} for variant in BASE_VARIANTS}
    period_returns: dict[str, dict[str, float]] = {variant: {} for variant in BASE_VARIANTS}
    regimes: dict[str, dict[str, float]] = {}
    for session in due5:
        pos_rows = POS.feature_rows(market, session)
        inv_rows = INV.feature_rows(market, session)
        row_source = {
            "low_vol_uptrend": pos_rows,
            "vol_compression_breakout": pos_rows,
            "panic_exhaustion": inv_rows,
            "inverse_liquidity_momentum": inv_rows,
            "quiet_pullback": inv_rows,
        }
        for variant in BASE_VARIANTS:
            symbols = rank_rows(row_source[variant], variant)
            selections[variant][session] = symbols
            period_returns[variant][session] = safe_period_return(market, session, symbols)
        regimes[session] = regime_features(market, session)
    return {"sessions": due5, "selections": selections, "period_returns": period_returns, "regimes": regimes}


def decisions_to_selections(decisions: dict[str, str], selections_by_variant: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    out = {}
    for session, decision in decisions.items():
        out[session] = [] if decision == "cash" else selections_by_variant[decision][session]
    return out


def replay_router(selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    dates = market["dates"]
    arrays = market["arrays"]
    symbol_index = market["symbol_index"]
    active = sorted(day for day in selections if start <= day <= end)
    if not active:
        return POS.summarize([], [], {})
    start_idx = market["date_index"][active[0]] + 1
    end_idx = min(len(dates) - 1, market["date_index"][active[-1]] + 1)
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
        if signal_date in selections and (last_rebalance_idx is None or idx - last_rebalance_idx >= REBALANCE_INTERVAL):
            if active_trade is not None and trades[active_trade].get("end_nav") is None:
                trades[active_trade]["end_nav"] = nav
                start_nav = trades[active_trade]["start_nav"]
                trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
            picked = []
            for sym in selections[signal_date][:TOP_K]:
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
    return POS.summarize(curve, trades, rejects)


def split_sessions(sessions: list[str], start: str, end: str) -> list[str]:
    return [session for session in sessions if start <= session <= end]


def oracle_decisions(sessions: list[str], period_returns: dict[str, dict[str, float]], allow_cash: bool) -> dict[str, str]:
    decisions = {}
    for session in sessions:
        best_variant = max(BASE_VARIANTS, key=lambda variant: period_returns[variant][session])
        if allow_cash and period_returns[best_variant][session] <= 0:
            decisions[session] = "cash"
        else:
            decisions[session] = best_variant
    return decisions


def rolling_best_decisions(sessions: list[str], period_returns: dict[str, dict[str, float]], lookback: int, allow_cash: bool) -> dict[str, str]:
    decisions = {}
    for i, session in enumerate(sessions):
        hist = sessions[max(0, i - lookback):i]
        if len(hist) < max(4, lookback // 3):
            decisions[session] = "low_vol_uptrend"
            continue
        means = {variant: float(np.mean([period_returns[variant][day] for day in hist])) for variant in BASE_VARIANTS}
        best_variant = max(BASE_VARIANTS, key=lambda variant: means[variant])
        decisions[session] = "cash" if allow_cash and means[best_variant] <= 0 else best_variant
    return decisions


def grid_rule_decisions(sessions: list[str], regimes: dict[str, dict[str, float]], params: dict[str, Any]) -> dict[str, str]:
    decisions = {}
    for session in sessions:
        r = regimes[session]
        if r["breadth20"] >= params["high_breadth"] and r["median_ret20"] >= params["high_ret20"]:
            decision = "low_vol_uptrend"
        elif r["breadth20"] <= params["low_breadth"] and r["median_vol20"] >= params["high_vol20"]:
            decision = "panic_exhaustion"
        elif r["median_vol_ratio20"] <= params["low_vol_ratio"]:
            decision = "quiet_pullback"
        else:
            decision = params["else_decision"]
        decisions[session] = decision
    return decisions


def choose_grid_rule(train_sessions: list[str], regimes: dict[str, dict[str, float]], period_returns: dict[str, dict[str, float]]) -> dict[str, Any]:
    best = None
    high_vol_values = [0.020, 0.024, 0.028]
    for high_breadth in (0.48, 0.52, 0.56, 0.60):
        for low_breadth in (0.36, 0.40, 0.44, 0.48):
            for high_ret20 in (-0.02, 0.0, 0.02):
                for high_vol20 in high_vol_values:
                    for low_vol_ratio in (0.75, 0.90, 1.05):
                        for else_decision in ("cash", "vol_compression_breakout", "inverse_liquidity_momentum"):
                            params = {"high_breadth": high_breadth, "low_breadth": low_breadth, "high_ret20": high_ret20, "high_vol20": high_vol20, "low_vol_ratio": low_vol_ratio, "else_decision": else_decision}
                            decisions = grid_rule_decisions(train_sessions, regimes, params)
                            value = float(np.prod([1.0 + (0.0 if decisions[s] == "cash" else period_returns[decisions[s]][s]) for s in train_sessions]))
                            penalty = 0.01 * sum(1 for s in train_sessions if decisions[s] == "cash") / max(1, len(train_sessions))
                            score = value - penalty
                            if best is None or score > best["score"]:
                                best = {"score": score, "train_period_multiple": value, "params": params}
    assert best is not None
    return best


def summarize_decisions(decisions: dict[str, str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for decision in decisions.values():
        counts[decision] = counts.get(decision, 0) + 1
    total = max(1, len(decisions))
    return {"counts": counts, "weights": {key: value / total for key, value in sorted(counts.items())}}


def eval_router(name: str, decisions: dict[str, str], tables: dict[str, Any], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    sessions = split_sessions(tables["sessions"], start, end)
    split_decisions = {session: decisions[session] for session in sessions if session in decisions}
    selections = decisions_to_selections(split_decisions, tables["selections"])
    return {"summary": replay_router(selections, market, start, end), "decisions": summarize_decisions(split_decisions), "name": name}


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    tables = build_tables(market)
    sessions = tables["sessions"]
    train_sessions = split_sessions(sessions, TRAIN_START, TRAIN_END)
    dev_sessions = split_sessions(sessions, DEV_START, DEV_END)
    all_decision_sessions = sessions
    grid_choice = choose_grid_rule(train_sessions, tables["regimes"], tables["period_returns"])
    routers = {
        "oracle_best_no_cash": oracle_decisions(all_decision_sessions, tables["period_returns"], allow_cash=False),
        "oracle_best_with_cash": oracle_decisions(all_decision_sessions, tables["period_returns"], allow_cash=True),
        "rolling_best_12_no_cash": rolling_best_decisions(all_decision_sessions, tables["period_returns"], lookback=12, allow_cash=False),
        "rolling_best_12_with_cash": rolling_best_decisions(all_decision_sessions, tables["period_returns"], lookback=12, allow_cash=True),
        "rolling_best_24_no_cash": rolling_best_decisions(all_decision_sessions, tables["period_returns"], lookback=24, allow_cash=False),
        "grid_regime_router": grid_rule_decisions(all_decision_sessions, tables["regimes"], grid_choice["params"]),
    }
    splits = {"train": (TRAIN_START, TRAIN_END), "valid": (VALID_START, VALID_END), "dev": (DEV_START, DEV_END), "forward": (FWD_START, FWD_END)}
    results = {
        router: {split: eval_router(router, decisions, tables, market, start, end) for split, (start, end) in splits.items()}
        for router, decisions in routers.items()
    }
    base_results = {
        variant: {split: POS.replay(tables["selections"][variant], market, start, end)["summary"] for split, (start, end) in splits.items()}
        for variant in BASE_VARIANTS
    }
    best_forward_router = max(routers, key=lambda name: results[name]["forward"]["summary"]["final_multiple"])
    verdict = "regime_router_not_learnable_or_not_thick"
    best_summary = results[best_forward_router]["forward"]["summary"]
    if (
        best_forward_router not in ("oracle_best_no_cash", "oracle_best_with_cash")
        and results[best_forward_router]["dev"]["summary"]["final_multiple"] >= 20.0
        and results[best_forward_router]["dev"]["summary"]["all_years_positive"]
        and best_summary["final_multiple"] > 1.2
        and best_summary["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0
    ):
        verdict = "regime_router_candidate_needs_modeling"
    out = {
        "experiment": "cross_sectional_regime_router_oracle_scan_v1",
        "method": "pool500_static_variant_oracle_vs_past_only_rolling_and_low_dof_regime_router",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "base_variants": BASE_VARIANTS, "top_k": TOP_K, "rebalance_interval": REBALANCE_INTERVAL},
        "inputs_sha256": {"script": sha256(Path(__file__)), "positive_script": sha256(ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py"), "inverse_script": sha256(ROOT / "analyze_cross_sectional_inverse_crowding_reversal_scan_v1.py")},
        "sessions": {key: len(split_sessions(sessions, start, end)) for key, (start, end) in splits.items()},
        "grid_choice": grid_choice,
        "base_results": base_results,
        "router_results": results,
        "best_forward_router": best_forward_router,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_forward_router": best_forward_router, "grid_choice": grid_choice, "best_forward_summary": best_summary, "router_forward": {name: results[name]["forward"]["summary"] for name in routers}}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
