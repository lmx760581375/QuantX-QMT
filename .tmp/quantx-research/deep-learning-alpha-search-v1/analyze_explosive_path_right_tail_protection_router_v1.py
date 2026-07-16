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
R86_PATH = ROOT / "analyze_explosive_path_contrastive_event_model_v1.py"
OUT = ROOT / "explosive_path_right_tail_protection_router_v1_summary.json"

SEED = 20260714
LABEL_VARIANT = "loose_explosive_holdable"
TEMPERATURE = 0.07
RANDOM_TRIALS = 200
STATIC_MODES = ("base_only", "model_only", "base_plus_w0.25", "base_plus_w0.50", "base_plus_w0.75", "base_plus_w1.00")
GATE_MODES = ("model_only", "base_plus_w0.50", "base_plus_w1.00")
QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)
FEATURES = (
    "model_base_rank_corr",
    "model_base_top20_overlap",
    "model_rank_dispersion",
    "model_top10_gap_to_mid",
    "model_top10_base_rank_mean",
    "base_top10_model_rank_mean",
    "mkt_ret20_mean",
    "mkt_ret60_mean",
    "mkt_up20_ratio",
    "amount_top20_share",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R86 = load_module("explosive_path_event_for_right_tail_router", R86_PATH)
R83 = R86.R83
R74 = R86.R74


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def rank01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
    order = np.argsort(np.where(np.isfinite(arr), arr, fill), kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.linspace(-0.5, 0.5, len(arr), endpoint=True) if len(arr) > 1 else 0.0
    return ranks


def session_indices(split: dict[str, Any]) -> dict[str, np.ndarray]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    return {session: np.asarray(indices, dtype=int) for session, indices in sorted(by_session.items())}


def session_rank_scores(split: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=float)
    for indices in session_indices(split).values():
        out[indices] = rank01(scores[indices])
    return out


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (den > 0))


def market_features(market: dict[str, Any], session: str, symbols: list[str]) -> dict[str, float]:
    idx = market["date_index"][session]
    cols = np.asarray([market["symbol_index"][sym] for sym in symbols if sym in market["symbol_index"]], dtype=int)
    if len(cols) == 0 or idx < 60:
        return {"mkt_ret20_mean": 0.0, "mkt_ret60_mean": 0.0, "mkt_up20_ratio": 0.0, "amount_top20_share": 0.0}
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    ret20 = safe_div(close[idx, cols], close[idx - 20, cols]) - 1.0
    ret60 = safe_div(close[idx, cols], close[idx - 60, cols]) - 1.0
    price = np.where(np.isfinite(vwap[idx, cols]) & (vwap[idx, cols] > 0), vwap[idx, cols], close[idx, cols])
    amount = np.where(np.isfinite(price) & np.isfinite(volume[idx, cols]), price * volume[idx, cols], np.nan)
    amount_sorted = np.sort(np.nan_to_num(amount, nan=0.0))[::-1]
    top_share = float(np.sum(amount_sorted[:20]) / np.sum(amount_sorted)) if np.sum(amount_sorted) > 0 else 0.0
    return {
        "mkt_ret20_mean": safe_mean(ret20),
        "mkt_ret60_mean": safe_mean(ret60),
        "mkt_up20_ratio": safe_mean(ret20 > 0.0),
        "amount_top20_share": top_share,
    }


def build_event_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str) -> dict[str, Any]:
    raw = R74.build_split(market, feats, amount20_arr, start, end)
    return R86.filter_valid_path(R86.enrich_split(market, raw, LABEL_VARIANT))


def train_scores(splits: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    raw_x = {name: R83.transform_x(split["x"], "real_path") for name, split in splits.items()}
    mu, sd = R83.standardizer(raw_x["train"])
    x_by_split = {name: R83.apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
    model, history = R86.train_model(x_by_split["train"], x_by_split["valid"], splits["train"], splits["valid"], TEMPERATURE, SEED + int(TEMPERATURE * 1000))
    return {name: R86.predict(model, x_by_split[name]) for name in splits}, history


def build_session_rows(market: dict[str, Any], split: dict[str, Any], base_rank: np.ndarray, model_rank: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for session, indices in session_indices(split).items():
        b = base_rank[indices]
        m = model_rank[indices]
        corr = float(np.corrcoef(b, m)[0, 1]) if len(indices) > 3 and np.std(b) > 1e-12 and np.std(m) > 1e-12 else 0.0
        base_order = np.argsort(-b, kind="mergesort")
        model_order = np.argsort(-m, kind="mergesort")
        top_base = set(base_order[:20])
        top_model = set(model_order[:20])
        mid_end = min(80, len(model_order))
        symbols = [split["symbols"][int(i)] for i in indices]
        rows.append({
            "session": session,
            "model_base_rank_corr": corr,
            "model_base_top20_overlap": len(top_base & top_model) / 20.0,
            "model_rank_dispersion": float(np.std(m)),
            "model_top10_gap_to_mid": safe_mean(m[model_order[:10]]) - safe_mean(m[model_order[20:mid_end]]),
            "model_top10_base_rank_mean": safe_mean(b[model_order[:10]]),
            "base_top10_model_rank_mean": safe_mean(m[base_order[:10]]),
            **market_features(market, session, symbols),
        })
    return rows


def score_for_mode(base_rank: np.ndarray, model_rank: np.ndarray, mode: str) -> np.ndarray:
    if mode == "base_only":
        return base_rank.copy()
    if mode == "model_only":
        return model_rank.copy()
    if mode.startswith("base_plus_w"):
        weight = float(mode.removeprefix("base_plus_w"))
        return base_rank + weight * model_rank
    raise ValueError(mode)


def combine_rule(split: dict[str, Any], base_rank: np.ndarray, model_rank: np.ndarray, rows: list[dict[str, Any]], rule: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(base_rank), dtype=float)
    row_by_session = {row["session"]: row for row in rows}
    for session, indices in session_indices(split).items():
        if rule["kind"] == "static":
            mode = rule["mode"]
        else:
            row = row_by_session[session]
            active = row[rule["feature"]] >= rule["threshold"] if rule["op"] == "high" else row[rule["feature"]] <= rule["threshold"]
            mode = rule["active_mode"] if active else "base_only"
        out[indices] = score_for_mode(base_rank[indices], model_rank[indices], mode)
    return out


def selection_score(ev: dict[str, Any]) -> float:
    replay = ev["replay"]
    ret = ev["return_metrics"]["top_returns"]
    event = ev["event_metrics"]
    rb3 = replay["remove_best_period_multiples"].get("remove_best_3", 1.0)
    log_mult = float(np.log(max(1e-9, replay["final_multiple"])))
    dd = abs(min(0.0, replay.get("max_drawdown", 0.0)))
    return float(1.0 * ret["top10"] + 0.6 * ret["top20"] + 0.08 * event["event_hit_rate"]["top10"] - 0.12 * event["event_fail_rate"]["top10"] + 0.055 * log_mult + 0.025 * max(0.0, rb3 - 1.0) - 0.10 * dd)


def evaluate_split(split: dict[str, Any], scores: np.ndarray, market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return {
        "return_metrics": R74.metric_from_scores(split, scores),
        "event_metrics": R86.event_metrics(split, scores),
        "event_pair": R86.event_pair_auc(scores, split),
        "replay": R74.replay_summary(split, scores, market, start, end),
    }


def compact_rule(rule: dict[str, Any]) -> str:
    if rule["kind"] == "static":
        return f"static_{rule['mode']}"
    return f"{rule['feature']}_{rule['op']}_q{rule['quantile']:.2f}_{rule['active_mode']}"


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def main() -> None:
    started = time.perf_counter()
    R86.set_seed(SEED)
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    train = build_event_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    valid = build_event_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    dev = R86.merge_splits(train, valid)
    forward = build_event_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    model_scores, history = train_scores(splits)
    model_rank = {name: session_rank_scores(split, model_scores[name]) for name, split in splits.items()}
    base_rank = {name: session_rank_scores(split, split["base_score"]) for name, split in splits.items()}
    rows_by_split = {name: build_session_rows(market, split, base_rank[name], model_rank[name]) for name, split in splits.items()}

    rules = [{"kind": "static", "mode": mode} for mode in STATIC_MODES]
    for feature in FEATURES:
        values = np.asarray([row[feature] for row in rows_by_split["train"]], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 20 or np.std(values) <= 1e-12:
            continue
        for q in QUANTILES:
            threshold = float(np.quantile(values, q))
            for op in ("high", "low"):
                for active_mode in GATE_MODES:
                    rules.append({"kind": "gate", "feature": feature, "op": op, "threshold": threshold, "quantile": q, "active_mode": active_mode})

    results = []
    for rule in rules:
        scores = {name: combine_rule(split, base_rank[name], model_rank[name], rows_by_split[name], rule) for name, split in splits.items()}
        ev = {
            "train": evaluate_split(train, scores["train"], market, R74.TRAIN_START, R74.TRAIN_END),
            "valid": evaluate_split(valid, scores["valid"], market, R74.VALID_START, R74.VALID_END),
            "dev": evaluate_split(dev, scores["dev"], market, R74.DEV_START, R74.DEV_END),
            "forward": evaluate_split(forward, scores["forward"], market, R74.FWD_START, R74.FWD_END),
        }
        train_score = selection_score(ev["train"])
        valid_score = selection_score(ev["valid"])
        combined = 0.35 * train_score + 0.65 * valid_score
        active = {}
        if rule["kind"] == "gate":
            for name, rows in rows_by_split.items():
                flags = [(row[rule["feature"]] >= rule["threshold"] if rule["op"] == "high" else row[rule["feature"]] <= rule["threshold"]) for row in rows]
                active[name] = safe_mean(flags)
        else:
            active = {name: 1.0 for name in splits}
        results.append({"rule": rule, "rule_name": compact_rule(rule), "train_score": train_score, "valid_score": valid_score, "combined_score": combined, "active_rate": active, "eval": ev})

    def row(item: dict[str, Any]) -> dict[str, Any]:
        ev = item["eval"]
        return {
            "rule": item["rule_name"],
            "combined_score": item["combined_score"],
            "train_score": item["train_score"],
            "valid_score": item["valid_score"],
            "active_rate": item["active_rate"],
            "dev_multiple": ev["dev"]["replay"]["final_multiple"],
            "dev_all_years_positive": ev["dev"]["replay"]["all_years_positive"],
            "dev_annual_returns": ev["dev"]["replay"]["annual_returns"],
            "forward_multiple": ev["forward"]["replay"]["final_multiple"],
            "forward_remove_best_3": ev["forward"]["replay"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
            "forward_avg_position": ev["forward"]["replay"]["avg_position_count"],
            "forward_top10_ret5": ev["forward"]["return_metrics"]["top_returns"]["top10"],
            "forward_top20_ret5": ev["forward"]["return_metrics"]["top_returns"]["top20"],
            "forward_hit": ev["forward"]["event_metrics"]["event_hit_rate"]["top10"],
            "forward_fail": ev["forward"]["event_metrics"]["event_fail_rate"]["top10"],
        }

    leaderboard = sorted([row(item) for item in results], key=lambda x: x["combined_score"], reverse=True)
    selected = leaderboard[0]
    best_forward = max(leaderboard, key=lambda x: x["forward_multiple"])
    forward_random = random_summary(forward, market)
    verdict = "explosive_path_right_tail_protection_router_not_enough"
    if selected["dev_multiple"] >= 20.0 and selected["dev_all_years_positive"] and selected["forward_multiple"] > forward_random["p95"] and selected["forward_remove_best_3"] > 1.0 and selected["forward_avg_position"] > 5:
        verdict = "explosive_path_right_tail_protection_router_candidate_needs_walkforward"
    out = {
        "experiment": "explosive_path_right_tail_protection_router_v1",
        "method": "train_loose_t0_07_explosive_path_contrastive_model_then_route_between_base_model_blends_using_train_quantile_session_state_gates_valid_selected_forward_evaluated",
        "params": {"label_variant": LABEL_VARIANT, "temperature": TEMPERATURE, "static_modes": STATIC_MODES, "gate_modes": GATE_MODES, "quantiles": QUANTILES, "features": FEATURES, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round86_script": sha256(R86_PATH), "round87_summary": sha256(ROOT / "explosive_path_failure_aware_contrastive_selector_v1_summary.json")},
        "sample_counts": {name: int(len(split["event_label"])) for name, split in splits.items()},
        "history_tail": history[-6:],
        "leaderboard": leaderboard,
        "selected_by_train_valid_router_score": selected,
        "best_by_forward_multiple": best_forward,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected": selected, "best_by_forward_multiple": best_forward, "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
