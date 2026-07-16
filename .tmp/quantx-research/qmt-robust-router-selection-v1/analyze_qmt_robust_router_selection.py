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
ROOT = REPO_ROOT / ".tmp/quantx-research"
SOURCE_ROOT = ROOT / "deep-learning-alpha-search-v1"
OUT_DIR = ROOT / "qmt-robust-router-selection-v1"
OUT = OUT_DIR / "qmt_robust_router_selection_2021_2026_diagnostic.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

LOOKBACKS = (4, 8, 12, 24, 36, 48)
HALF_LIVES = (4.0, 8.0, 12.0, 24.0, 36.0)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R36 = load_module("regime_router_v1_exp123", SOURCE_ROOT / "analyze_cross_sectional_regime_router_oracle_scan_v1.py")
POS = R36.POS
BASE_VARIANTS = tuple(R36.BASE_VARIANTS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sessions(sessions: list[str], start: str, end: str) -> list[str]:
    return [session for session in sessions if start <= session <= end]


def exit_dates(market: dict[str, Any], sessions: list[str]) -> dict[str, str]:
    out = {}
    dates = market["dates"]
    for session in sessions:
        idx = market["date_index"][session]
        exit_idx = min(len(dates) - 1, idx + R36.REBALANCE_INTERVAL + 1)
        out[session] = dates[exit_idx]
    return out


def completed_history(sessions: list[str], current: str, exits: dict[str, str]) -> list[str]:
    return [session for session in sessions if session < current and exits[session] <= current]


def fixed_decisions(sessions: list[str], variant: str) -> dict[str, str]:
    return {session: variant for session in sessions}


def exit_aware_rolling_best(sessions: list[str], period_returns: dict[str, dict[str, float]], exits: dict[str, str], lookback: int, allow_cash: bool) -> dict[str, str]:
    decisions = {}
    for session in sessions:
        hist = completed_history(sessions, session, exits)[-lookback:]
        if len(hist) < max(3, lookback // 3):
            decisions[session] = "low_vol_uptrend"
            continue
        means = {variant: float(np.mean([period_returns[variant][day] for day in hist])) for variant in BASE_VARIANTS}
        best = max(BASE_VARIANTS, key=lambda variant: means[variant])
        decisions[session] = "cash" if allow_cash and means[best] <= 0 else best
    return decisions


def exit_aware_ewma_best(sessions: list[str], period_returns: dict[str, dict[str, float]], exits: dict[str, str], half_life: float, allow_cash: bool) -> dict[str, str]:
    decisions = {}
    for session in sessions:
        hist = completed_history(sessions, session, exits)
        if len(hist) < 4:
            decisions[session] = "low_vol_uptrend"
            continue
        scores = {}
        for variant in BASE_VARIANTS:
            values = np.asarray([period_returns[variant][day] for day in hist], dtype=float)
            ages = np.arange(len(values) - 1, -1, -1, dtype=float)
            weights = 0.5 ** (ages / half_life)
            scores[variant] = float(np.average(values, weights=weights))
        best = max(BASE_VARIANTS, key=lambda variant: scores[variant])
        decisions[session] = "cash" if allow_cash and scores[best] <= 0 else best
    return decisions


def regime_grid_decisions(sessions: list[str], regimes: dict[str, dict[str, float]], params: dict[str, Any]) -> dict[str, str]:
    return R36.grid_rule_decisions(sessions, regimes, params)


def grid_param_space() -> list[dict[str, Any]]:
    rows = []
    for high_breadth in (0.48, 0.52, 0.56, 0.60):
        for low_breadth in (0.36, 0.40, 0.44, 0.48):
            for high_ret20 in (-0.02, 0.0, 0.02):
                for high_vol20 in (0.020, 0.024, 0.028):
                    for low_vol_ratio in (0.75, 0.90, 1.05):
                        for else_decision in ("cash", "vol_compression_breakout", "inverse_liquidity_momentum"):
                            rows.append({
                                "high_breadth": high_breadth,
                                "low_breadth": low_breadth,
                                "high_ret20": high_ret20,
                                "high_vol20": high_vol20,
                                "low_vol_ratio": low_vol_ratio,
                                "else_decision": else_decision,
                            })
    return rows


def build_candidates(tables: dict[str, Any], exits: dict[str, str]) -> dict[str, dict[str, str]]:
    sessions = tables["sessions"]
    candidates: dict[str, dict[str, str]] = {}
    for variant in BASE_VARIANTS:
        candidates[f"fixed::{variant}"] = fixed_decisions(sessions, variant)
    for lookback in LOOKBACKS:
        for allow_cash in (False, True):
            suffix = "cash" if allow_cash else "nocash"
            candidates[f"exit_rolling::{lookback}::{suffix}"] = exit_aware_rolling_best(sessions, tables["period_returns"], exits, lookback, allow_cash)
    for half_life in HALF_LIVES:
        for allow_cash in (False, True):
            suffix = "cash" if allow_cash else "nocash"
            candidates[f"exit_ewma::{half_life:g}::{suffix}"] = exit_aware_ewma_best(sessions, tables["period_returns"], exits, half_life, allow_cash)
    train_sessions = split_sessions(sessions, TRAIN_START, TRAIN_END)
    scored_grids = []
    for params in grid_param_space():
        decisions = regime_grid_decisions(sessions, tables["regimes"], params)
        yearly = period_year_returns(decisions, tables["period_returns"], train_sessions)
        values = list(yearly.values())
        min_year = min(values) if values else -1.0
        mean_year = float(np.mean(values)) if values else -1.0
        cash_ratio = sum(1 for day in train_sessions if decisions[day] == "cash") / max(1, len(train_sessions))
        score = min_year + 0.25 * mean_year - 0.02 * cash_ratio
        scored_grids.append((score, params, decisions, min_year, mean_year))
    scored_grids.sort(key=lambda item: item[0], reverse=True)
    for idx, (_, params, decisions, _, _) in enumerate(scored_grids[:20]):
        candidates[f"robust_grid::{idx:02d}"] = decisions
    return candidates


def period_year_returns(decisions: dict[str, str], period_returns: dict[str, dict[str, float]], sessions: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for session in sessions:
        decision = decisions[session]
        value = 0.0 if decision == "cash" else period_returns[decision][session]
        grouped.setdefault(session[:4], []).append(value)
    return {year: float(np.prod([1.0 + value for value in values]) - 1.0) for year, values in sorted(grouped.items())}


def evaluate_candidate(name: str, decisions: dict[str, str], tables: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "valid": (VALID_START, VALID_END),
        "dev": (DEV_START, DEV_END),
        "forward": (FWD_START, FWD_END),
    }
    evaluated = {split: R36.eval_router(name, decisions, tables, market, start, end) for split, (start, end) in splits.items()}
    return {"name": name, "splits": evaluated, "decision_counts": R36.summarize_decisions(decisions)}


def robust_score(item: dict[str, Any]) -> float:
    train = item["splits"]["train"]["summary"]
    valid = item["splits"]["valid"]["summary"]
    train_years = list(train.get("annual_returns", {}).values())
    min_train_year = min(train_years) if train_years else -1.0
    valid_return = float(valid.get("total_return", -1.0))
    train_remove3 = float(train.get("remove_best_period_multiples", {}).get("remove_best_3", 0.0)) - 1.0
    valid_remove3 = float(valid.get("remove_best_period_multiples", {}).get("remove_best_3", 0.0)) - 1.0
    drawdown_penalty = abs(float(train.get("max_drawdown", 0.0))) + abs(float(valid.get("max_drawdown", 0.0)))
    cash_penalty = 0.0
    avg_pos = min(float(train.get("avg_position_count", 0.0)), float(valid.get("avg_position_count", 0.0)))
    if avg_pos < 5.0:
        cash_penalty += 1.0
    robust_floor = min(min_train_year, valid_return, train_remove3, valid_remove3)
    log_growth = math.log(max(float(train.get("final_multiple", 1e-9)), 1e-9)) + math.log(max(float(valid.get("final_multiple", 1e-9)), 1e-9))
    return 5.0 * robust_floor + 0.25 * log_growth - 0.50 * drawdown_penalty - cash_penalty


def compact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "robust_score": float(item["robust_score"]),
        "decision_counts": item["decision_counts"],
        "splits": {
            split: {
                "summary": data["summary"],
                "decisions": data["decisions"],
            }
            for split, data in item["splits"].items()
        },
    }


def main() -> int:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market = POS.load_market()
    tables = R36.build_tables(market)
    exits = exit_dates(market, tables["sessions"])
    candidates = build_candidates(tables, exits)
    evaluated = []
    for name, decisions in candidates.items():
        item = evaluate_candidate(name, decisions, tables, market)
        item["robust_score"] = robust_score(item)
        evaluated.append(item)
    evaluated.sort(key=lambda item: item["robust_score"], reverse=True)
    best_robust = evaluated[0]
    best_forward = max(evaluated, key=lambda item: item["splits"]["forward"]["summary"]["final_multiple"])
    verdict = "robust_router_selection_rejected"
    fwd = best_robust["splits"]["forward"]["summary"]
    dev = best_robust["splits"]["dev"]["summary"]
    if dev["final_multiple"] >= 20.0 and dev["all_years_positive"] and fwd["final_multiple"] > 1.2 and fwd["avg_position_count"] > 5 and fwd["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0:
        verdict = "robust_router_selection_candidate_needs_formal_replay"
    out = {
        "experiment": "qmt_robust_router_selection_v1",
        "method": "exit_date_aware_robust_selection_over_existing_qmt_only_router_candidates",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "base_variants": BASE_VARIANTS,
            "lookbacks": LOOKBACKS,
            "half_lives": HALF_LIVES,
            "candidate_count": len(evaluated),
        },
        "causality": "Router history is exit-date aware: at signal T it only uses variant outcomes whose open-to-open exit_date <= T. Candidate selection uses train/valid robustness only; forward is not used for selection. Base variants use QMT daily OHLCV/VWAP only.",
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "regime_router_script": sha256(SOURCE_ROOT / "analyze_cross_sectional_regime_router_oracle_scan_v1.py"),
            "positive_script": sha256(SOURCE_ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py"),
            "inverse_script": sha256(SOURCE_ROOT / "analyze_cross_sectional_inverse_crowding_reversal_scan_v1.py"),
        },
        "sessions": {"train": len(split_sessions(tables["sessions"], TRAIN_START, TRAIN_END)), "valid": len(split_sessions(tables["sessions"], VALID_START, VALID_END)), "dev": len(split_sessions(tables["sessions"], DEV_START, DEV_END)), "forward": len(split_sessions(tables["sessions"], FWD_START, FWD_END))},
        "best_robust": compact(best_robust),
        "best_forward_diagnostic_not_for_selection": compact(best_forward),
        "leaderboard": [compact(item) for item in evaluated[:12]],
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "best_robust": brief(best_robust),
        "best_forward_diagnostic_not_for_selection": brief(best_forward),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def brief(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "robust_score": float(item["robust_score"]),
        "train": item["splits"]["train"]["summary"],
        "valid": item["splits"]["valid"]["summary"],
        "dev": item["splits"]["dev"]["summary"],
        "forward": item["splits"]["forward"]["summary"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
