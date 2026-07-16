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
R74_PATH = ROOT / "analyze_cross_sectional_daily_path_memory_residual_v1.py"
OUT = ROOT / "daily_path_opportunity_filter_v1_summary.json"

ACTIVE_RATE_MIN = 0.50
RANDOM_TRIALS = 200
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R74 = load_module("daily_path_residual_v1_for_opportunity_filter", R74_PATH)
POS = R74.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


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


def replay_with_cash(all_signal_days: list[str], selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    dates = market["dates"]
    arrays = market["arrays"]
    symbol_index = market["symbol_index"]
    active = sorted(day for day in all_signal_days if start <= day <= end)
    if not active:
        return {"summary": summarize([], [], {}), "trades_tail": []}
    start_idx = market["date_index"][active[0]] + 1
    end_idx = min(len(dates) - 1, market["date_index"][active[-1]] + 1)
    nav = 1.0
    peak = 1.0
    positions: dict[str, float] = {}
    active_trade: int | None = None
    signal_days = set(active)
    curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rejects = {"candidate_total": 0, "selected_total": 0, "missing_or_suspended": 0, "missing_preclose": 0, "price_jump_abs_gt_9p5": 0, "cash_rebalances": 0}
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
        if signal_date in signal_days:
            if active_trade is not None and trades[active_trade].get("end_nav") is None:
                trades[active_trade]["end_nav"] = nav
                start_nav = trades[active_trade]["start_nav"]
                trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
            records = selections.get(signal_date, [])
            picked = []
            for sym in records[: R74.TOP_K_REPLAY]:
                rejects["candidate_total"] += 1
                ok, reason = POS.tradable(sym, idx, idx - 1, market)
                if ok:
                    picked.append(sym)
                else:
                    rejects[reason] += 1
            target = {sym: 1.0 / len(picked) for sym in picked} if picked else {}
            if not target:
                rejects["cash_rebalances"] += 1
            rejects["selected_total"] += len(target)
            universe = set(target) | set(positions)
            buy = sum(max(0.0, target.get(sym, 0.0) - positions.get(sym, 0.0)) for sym in universe)
            sell = sum(max(0.0, positions.get(sym, 0.0) - target.get(sym, 0.0)) for sym in universe)
            nav *= max(0.0, 1.0 - buy * 0.00052 - sell * 0.00102)
            trades.append({"date": date, "signal_date": signal_date, "count": len(target), "start_nav": nav, "end_nav": None})
            active_trade = len(trades) - 1
            positions = target
        peak = max(peak, nav)
        curve.append({"date": date, "nav_before": nav_before, "nav": nav, "daily_return": nav / nav_before - 1.0 if nav_before > 0 else 0.0, "drawdown": nav / peak - 1.0, "position_count": len(positions)})
    if active_trade is not None and trades[active_trade].get("end_nav") is None:
        trades[active_trade]["end_nav"] = nav
        start_nav = trades[active_trade]["start_nav"]
        trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
    return {"summary": summarize(curve, trades, rejects), "trades_tail": trades[-5:]}


def session_rows(split: dict[str, Any], scores: np.ndarray) -> list[dict[str, Any]]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    rows = []
    for session, indices in sorted(by_session.items()):
        idx = np.asarray(indices, dtype=int)
        score = scores[idx]
        y = split["raw_y"][idx]
        x = split["x"][idx]
        order = np.argsort(-score, kind="mergesort")
        top10 = order[:10]
        top20 = order[:20]
        rows.append(
            {
                "session": session,
                "top10_return": safe_mean(y[top10]),
                "top20_return": safe_mean(y[top20]),
                "score_mean_top10": safe_mean(score[top10]),
                "score_gap_10_30": safe_mean(score[top10]) - safe_mean(score[order[10:30]]),
                "score_dispersion": float(np.nanstd(score)),
                "mkt_ret1_mean": safe_mean(x[:, 0]),
                "mkt_range_mean": safe_mean(x[:, 3]),
                "mkt_vwap_mean": safe_mean(x[:, 4]),
                "mkt_ret20_mean": safe_mean(x[:, 7]),
                "mkt_liq_mean": safe_mean(x[:, 67]),
                "selection": [str(split["symbols"][idx[j]]) for j in top10],
            }
        )
    return rows


def threshold_gate(rows: list[dict[str, Any]], feature: str, quantile: float, train_rows: list[dict[str, Any]], direction: str) -> tuple[dict[str, bool], float]:
    values = np.asarray([float(row[feature]) for row in train_rows], dtype=float)
    threshold = float(np.nanquantile(values, quantile))
    if direction == "high":
        return {row["session"]: float(row[feature]) >= threshold for row in rows}, threshold
    return {row["session"]: float(row[feature]) <= threshold for row in rows}, threshold


def evaluate_gate(name: str, gate: dict[str, bool], rows: list[dict[str, Any]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    all_days = [row["session"] for row in rows]
    selections = {row["session"]: row["selection"] for row in rows if gate.get(row["session"], False)}
    active_rate = len(selections) / max(1, len(rows))
    active_returns = [row["top10_return"] for row in rows if gate.get(row["session"], False)]
    inactive_returns = [row["top10_return"] for row in rows if not gate.get(row["session"], False)]
    replay = replay_with_cash(all_days, selections, market, start, end)["summary"]
    return {
        "gate": name,
        "active_rate": active_rate,
        "active_top10_mean": safe_mean(active_returns),
        "inactive_top10_mean": safe_mean(inactive_returns),
        "active_minus_inactive": safe_mean(active_returns) - safe_mean(inactive_returns),
        "replay": replay,
    }


def random_cash_summary(rows: list[dict[str, Any]], market: dict[str, Any], start: str, end: str, active_rate: float) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    all_days = [row["session"] for row in rows]
    values = []
    n_active = int(round(len(rows) * active_rate))
    for _ in range(RANDOM_TRIALS):
        picked = set(rng.choice(np.asarray(all_days, dtype=object), size=max(1, n_active), replace=False))
        selections = {row["session"]: row["selection"] for row in rows if row["session"] in picked}
        values.append(replay_with_cash(all_days, selections, market, start, end)["summary"]["final_multiple"])
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    train = R74.build_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    valid = R74.build_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    dev = R74.merge_splits(train, valid)
    forward = R74.build_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    null_scores, null_meta = R74.build_null(train, splits)
    model, history = R74.train_residual(train, valid, null_scores)
    scores = {name: R74.predict(model, split["x"]) for name, split in splits.items()}
    rows = {name: session_rows(split, scores[name]) for name, split in splits.items()}
    baseline = {
        "dev": evaluate_gate("all_active", {row["session"]: True for row in rows["dev"]}, rows["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": evaluate_gate("all_active", {row["session"]: True for row in rows["forward"]}, rows["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    gate_specs: list[tuple[str, str, float, str]] = [("all_active", "score_dispersion", 0.0, "high")]
    for feature, direction in [
        ("score_gap_10_30", "high"),
        ("score_dispersion", "high"),
        ("score_mean_top10", "high"),
        ("mkt_ret20_mean", "high"),
        ("mkt_range_mean", "low"),
        ("mkt_vwap_mean", "high"),
    ]:
        for q in (0.20, 0.35, 0.50):
            gate_specs.append((f"{feature}_{direction}_q{q:.2f}", feature, q, direction))
    scan = []
    for name, feature, q, direction in gate_specs:
        if name == "all_active":
            gates = {split: {row["session"]: True for row in rows[split]} for split in rows}
            threshold = None
        else:
            gates = {}
            threshold = None
            for split in rows:
                gate, threshold = threshold_gate(rows[split], feature, q, rows["train"], direction)
                gates[split] = gate
        evals = {
            "train": evaluate_gate(name, gates["train"], rows["train"], market, R74.TRAIN_START, R74.TRAIN_END),
            "valid": evaluate_gate(name, gates["valid"], rows["valid"], market, R74.VALID_START, R74.VALID_END),
            "dev": evaluate_gate(name, gates["dev"], rows["dev"], market, R74.DEV_START, R74.DEV_END),
            "forward": evaluate_gate(name, gates["forward"], rows["forward"], market, R74.FWD_START, R74.FWD_END),
        }
        valid = evals["valid"]
        train_eval = evals["train"]
        score = -1e9
        if valid["active_rate"] >= ACTIVE_RATE_MIN and train_eval["active_rate"] >= ACTIVE_RATE_MIN:
            score = (
                math.log(max(valid["replay"]["final_multiple"], 1e-9))
                + 0.50 * valid["active_minus_inactive"]
                - abs(valid["replay"]["max_drawdown"])
                + 0.20 * math.log(max(train_eval["replay"]["final_multiple"], 1e-9))
            )
        scan.append({"gate": name, "feature": feature, "quantile": q, "direction": direction, "threshold": threshold, "score": score, "evals": evals})
    scan = sorted(scan, key=lambda item: item["score"], reverse=True)
    selected = scan[0]
    forward_random = random_cash_summary(rows["forward"], market, R74.FWD_START, R74.FWD_END, selected["evals"]["forward"]["active_rate"])
    verdict = "daily_path_opportunity_filter_not_enough"
    fwd = selected["evals"]["forward"]["replay"]
    if (
        selected["evals"]["dev"]["replay"]["all_years_positive"]
        and fwd["final_multiple"] > baseline["forward"]["replay"]["final_multiple"]
        and fwd["final_multiple"] > forward_random["p90"]
        and fwd["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0
        and fwd["avg_position_count"] > 5
    ):
        verdict = "daily_path_opportunity_filter_candidate_needs_walkforward"
    out = {
        "experiment": "daily_path_opportunity_filter_v1",
        "method": "session_level_opportunity_filter_for_round74_real_path_scores_cash_when_inactive",
        "params": {"active_rate_min": ACTIVE_RATE_MIN, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round74_script": sha256(R74_PATH), "round76_summary": sha256(ROOT / "daily_path_right_tail_pairwise_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "null_meta": null_meta,
        "history_tail": history[-6:],
        "baseline": baseline,
        "leaderboard": scan[:12],
        "selected": selected,
        "forward_random_same_active_rate": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected": selected, "baseline": baseline, "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
