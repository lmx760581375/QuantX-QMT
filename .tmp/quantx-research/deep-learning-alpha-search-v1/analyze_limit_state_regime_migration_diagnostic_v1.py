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
OUT = ROOT / "limit_state_regime_migration_diagnostic_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

CANDIDATE = "post_limit_volume_dry_hold"
CELLS = ((500, 5), (1000, 5))
TOP_K = 10
RANDOM_TRIALS = 200
SEED = 20260714
FEATURES = (
    "mkt_ret5_mean",
    "mkt_ret20_mean",
    "mkt_ret60_mean",
    "mkt_up20_ratio",
    "mkt_limit_density",
    "mkt_high_limit_density",
    "mkt_recent_limit_density",
    "mkt_failed_board_density",
    "event_count",
    "event_score_mean",
    "event_score_dispersion",
)
QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R80 = load_module("daily_limit_up_break_board_state_for_regime_diag", ROOT / "analyze_daily_limit_up_break_board_state_scan_v1.py")
R79 = R80.R79
CVR = R79.CVR
RT = R79.RT
POS = R79.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) else 0.0


def pct(close: np.ndarray, idx: int, horizon: int, cols: np.ndarray) -> np.ndarray:
    if idx < horizon:
        return np.full(len(cols), np.nan)
    return R79.safe_div(close[idx, cols], close[idx - horizon, cols]) - 1.0


def row_for_session(market: dict[str, Any], amount20_arr: np.ndarray, session: str, pool_size: int, interval: int) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    cols = np.asarray(R79.pool_cols(market, amount20_arr, idx, pool_size), dtype=int)
    if len(cols) < TOP_K:
        return None
    frame = R80.limit_event_frame(market, idx, cols)
    score, mask = R80.limit_candidate_scores(frame)[CANDIDATE]
    records = []
    for j, col in enumerate(cols):
        if not bool(mask[j]):
            continue
        y = R79.future_return(market, idx, int(col), interval)
        if y is None:
            continue
        records.append((j, int(col), float(score[j]), float(y)))
    ordered = sorted(records, key=lambda item: (-item[2], str(market["symbols"][item[1]])))
    picks = [str(market["symbols"][col]) for _, col, _, _ in ordered[:TOP_K]]
    returns = [item[3] for item in ordered[:TOP_K]]
    event_scores = [item[2] for item in ordered]
    close = market["arrays"]["close"].astype(float)
    ret5 = pct(close, idx, 5, cols)
    ret20 = pct(close, idx, 20, cols)
    ret60 = pct(close, idx, 60, cols)
    features = {
        "mkt_ret5_mean": safe_mean(ret5),
        "mkt_ret20_mean": safe_mean(ret20),
        "mkt_ret60_mean": safe_mean(ret60),
        "mkt_up20_ratio": safe_mean(ret20 > 0.0),
        "mkt_limit_density": safe_mean(frame["close_limit"] > 0.5),
        "mkt_high_limit_density": safe_mean(frame["high_limit"] > 0.5),
        "mkt_recent_limit_density": safe_mean(frame["recent_limit_count"] >= 1.0),
        "mkt_failed_board_density": safe_mean(frame["failed_board"] > 0.5),
        "event_count": float(len(records)),
        "event_score_mean": safe_mean(event_scores),
        "event_score_dispersion": safe_std(event_scores),
    }
    return {
        "session": session,
        "year": session[:4],
        "selection": picks,
        "selected_count": len(picks),
        "top10_mean": safe_mean(returns),
        **features,
    }


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, pool_size: int, interval: int) -> list[dict[str, Any]]:
    rows = []
    for session in CVR.sessions(market, start, end, interval):
        row = row_for_session(market, amount20_arr, session, pool_size, interval)
        if row is not None:
            rows.append(row)
    return rows


def selections(rows: list[dict[str, Any]], active: dict[str, bool] | None = None) -> dict[str, list[str]]:
    out = {}
    for row in rows:
        if active is not None and not active.get(row["session"], False):
            continue
        if row["selection"]:
            out[row["session"]] = list(row["selection"])
    return out


def random_selections(rows: list[dict[str, Any]], rng: np.random.Generator, active: dict[str, bool]) -> dict[str, list[str]]:
    out = {}
    for row in rows:
        if not active.get(row["session"], False):
            continue
        pool = row["selection"]
        if not pool:
            continue
        size = min(TOP_K, len(pool))
        out[row["session"]] = [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=size, replace=False)]
    return out


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def aggregate_rows(rows: list[dict[str, Any]], active: dict[str, bool] | None = None) -> dict[str, Any]:
    used = [row for row in rows if active is None or active.get(row["session"], False)]
    if not rows:
        return {"sessions": 0, "active_sessions": 0}
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in used:
        by_year.setdefault(row["year"], []).append(row)
    return {
        "sessions": len(rows),
        "active_sessions": len(used),
        "active_rate": len(used) / len(rows),
        "avg_selected_count": safe_mean([row["selected_count"] for row in used]),
        "avg_top10_mean": safe_mean([row["top10_mean"] for row in used]),
        "avg_event_count": safe_mean([row["event_count"] for row in used]),
        "feature_mean": {name: safe_mean([row[name] for row in used]) for name in FEATURES},
        "by_year": {
            year: {
                "sessions": len(part),
                "avg_selected_count": safe_mean([row["selected_count"] for row in part]),
                "avg_top10_mean": safe_mean([row["top10_mean"] for row in part]),
                "avg_event_count": safe_mean([row["event_count"] for row in part]),
                "feature_mean": {name: safe_mean([row[name] for row in part]) for name in FEATURES},
            }
            for year, part in sorted(by_year.items())
        },
    }


def gate_from_threshold(rows: list[dict[str, Any]], feature: str, op: str, threshold: float) -> dict[str, bool]:
    if op == "high":
        return {row["session"]: float(row[feature]) >= threshold for row in rows}
    return {row["session"]: float(row[feature]) <= threshold for row in rows}


def eval_gate(rows: list[dict[str, Any]], market: dict[str, Any], start: str, end: str, interval: int, active: dict[str, bool]) -> dict[str, Any]:
    replay = CVR.replay_interval(selections(rows, active), market, start, end, interval)["summary"]
    stats = aggregate_rows(rows, active)
    return {"replay": replay, "stats": stats}


def run_cell(market: dict[str, Any], amount20_arr: np.ndarray, pool_size: int, interval: int) -> dict[str, Any]:
    split_rows = {
        "train": build_rows(market, amount20_arr, TRAIN_START, TRAIN_END, pool_size, interval),
        "valid": build_rows(market, amount20_arr, VALID_START, VALID_END, pool_size, interval),
        "dev": build_rows(market, amount20_arr, DEV_START, DEV_END, pool_size, interval),
        "forward": build_rows(market, amount20_arr, FWD_START, FWD_END, pool_size, interval),
    }
    baseline_active = {name: {row["session"]: True for row in rows} for name, rows in split_rows.items()}
    baseline = {
        "train": eval_gate(split_rows["train"], market, TRAIN_START, TRAIN_END, interval, baseline_active["train"]),
        "valid": eval_gate(split_rows["valid"], market, VALID_START, VALID_END, interval, baseline_active["valid"]),
        "dev": eval_gate(split_rows["dev"], market, DEV_START, DEV_END, interval, baseline_active["dev"]),
        "forward": eval_gate(split_rows["forward"], market, FWD_START, FWD_END, interval, baseline_active["forward"]),
    }
    gates = []
    for feature in FEATURES:
        values = np.asarray([row[feature] for row in split_rows["train"]], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 10 or np.nanstd(values) <= 1e-12:
            continue
        for q in QUANTILES:
            threshold = float(np.quantile(values, q))
            for op in ("high", "low"):
                name = f"{feature}_{op}_q{q:.2f}"
                active = {split: gate_from_threshold(rows, feature, op, threshold) for split, rows in split_rows.items()}
                ev = {
                    "train": eval_gate(split_rows["train"], market, TRAIN_START, TRAIN_END, interval, active["train"]),
                    "valid": eval_gate(split_rows["valid"], market, VALID_START, VALID_END, interval, active["valid"]),
                    "dev": eval_gate(split_rows["dev"], market, DEV_START, DEV_END, interval, active["dev"]),
                    "forward": eval_gate(split_rows["forward"], market, FWD_START, FWD_END, interval, active["forward"]),
                }
                gates.append({
                    "name": name,
                    "feature": feature,
                    "op": op,
                    "q": q,
                    "threshold": threshold,
                    "evals": ev,
                    "valid_score": math.log(max(ev["valid"]["replay"]["final_multiple"], 1e-9)) + 0.20 * ev["valid"]["stats"]["active_rate"] + 0.10 * ev["train"]["stats"]["active_rate"],
                })
    gates = sorted(gates, key=lambda item: -item["valid_score"])
    selected = gates[0] if gates else None
    forward_random = None
    if selected is not None:
        active_forward = gate_from_threshold(split_rows["forward"], selected["feature"], selected["op"], selected["threshold"])
        rng = np.random.default_rng(SEED + pool_size + interval)
        values = [CVR.replay_interval(random_selections(split_rows["forward"], rng, active_forward), market, FWD_START, FWD_END, interval)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
        forward_random = percentile_summary(values)
    return {
        "pool_size": pool_size,
        "interval": interval,
        "row_counts": {name: len(rows) for name, rows in split_rows.items()},
        "baseline": baseline,
        "gates": gates[:40],
        "selected_by_valid": selected,
        "selected_forward_random": forward_random,
    }


def flatten(cells: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, cell in cells.items():
        selected = cell.get("selected_by_valid")
        if selected is None:
            continue
        ev = selected["evals"]
        fwd = ev["forward"]["replay"]
        dev = ev["dev"]["replay"]
        rows.append({
            "key": key,
            "pool_size": cell["pool_size"],
            "interval": cell["interval"],
            "gate": selected["name"],
            "feature": selected["feature"],
            "op": selected["op"],
            "threshold": selected["threshold"],
            "train_multiple": ev["train"]["replay"]["final_multiple"],
            "valid_multiple": ev["valid"]["replay"]["final_multiple"],
            "dev_multiple": dev["final_multiple"],
            "dev_all_years_positive": dev["all_years_positive"],
            "forward_multiple": fwd["final_multiple"],
            "forward_remove_best_3": fwd["remove_best_period_multiples"].get("remove_best_3", 0.0),
            "forward_avg_position": fwd["avg_position_count"],
            "forward_random_p95": cell["selected_forward_random"]["p95"] if cell.get("selected_forward_random") else 0.0,
            "train_active_rate": ev["train"]["stats"]["active_rate"],
            "valid_active_rate": ev["valid"]["stats"]["active_rate"],
            "forward_active_rate": ev["forward"]["stats"]["active_rate"],
            "score": selected["valid_score"],
        })
    return sorted(rows, key=lambda row: -row["score"])


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    cells = {f"pool{pool}_interval{interval}": run_cell(market, amount20_arr, pool, interval) for pool, interval in CELLS}
    leaderboard = flatten(cells)
    best = leaderboard[0] if leaderboard else {}
    verdict = "limit_state_regime_migration_no_stable_router"
    if (
        best
        and best["dev_multiple"] >= 20.0
        and best["dev_all_years_positive"]
        and best["forward_multiple"] > best["forward_random_p95"]
        and best["forward_remove_best_3"] > 1.0
        and best["forward_avg_position"] > 5
    ):
        verdict = "limit_state_regime_migration_candidate_router_needs_modeling"
    out = {
        "experiment": "limit_state_regime_migration_diagnostic_v1",
        "method": "diagnose_post_limit_volume_dry_hold_regime_migration_with_train_quantile_gates",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "candidate": CANDIDATE,
            "cells": CELLS,
            "features": FEATURES,
            "quantiles": QUANTILES,
            "top_k": TOP_K,
            "random_trials": RANDOM_TRIALS,
            "seed": SEED,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "round80_script": sha256(ROOT / "analyze_daily_limit_up_break_board_state_scan_v1.py"), "round80_summary": sha256(ROOT / "daily_limit_up_break_board_state_scan_v1_summary.json")},
        "cells": cells,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "leaderboard": leaderboard, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
