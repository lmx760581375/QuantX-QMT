from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
R86_PATH = ROOT / "analyze_explosive_path_contrastive_event_model_v1.py"
OUT = ROOT / "explosive_path_liquidity_tier_domain_shift_v1_summary.json"

SEED = 20260714
LABEL_VARIANT = "loose_explosive_holdable"
TEMPERATURE = 0.07
RANDOM_TRIALS = 160
POOL_DOMAINS = {
    "top500": (0, 500),
    "top1000": (0, 1000),
    "top1500": (0, 1500),
    "tier500_1000": (500, 1000),
    "tier1000_1500": (1000, 1500),
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R86 = load_module("explosive_path_event_for_liquidity_tier_domain", R86_PATH)
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


def ordered_liquidity_cols(market: dict[str, Any], amount20_arr: np.ndarray, idx: int) -> list[int]:
    arrays = market["arrays"]
    valid = (
        np.isfinite(arrays["open"][idx]) & (arrays["open"][idx] > 0)
        & np.isfinite(arrays["close"][idx]) & (arrays["close"][idx] > 0)
        & np.isfinite(arrays["volume"][idx]) & (arrays["volume"][idx] > 0)
        & np.isfinite(amount20_arr[idx]) & (amount20_arr[idx] > 0)
    )
    cols = np.flatnonzero(valid)
    return [int(col) for col in sorted(cols, key=lambda col: (-float(amount20_arr[idx, col]), str(market["symbols"][col])))]


def build_domain_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str, domain: str) -> dict[str, Any]:
    lo, hi = POOL_DOMAINS[domain]
    xs: list[np.ndarray] = []
    raw_y: list[float] = []
    label_rank: list[float] = []
    event_label: list[int] = []
    sessions: list[str] = []
    symbols: list[str] = []
    base_scores: list[float] = []
    pools: dict[str, list[str]] = {}
    stats_rows: dict[str, list[float]] = {key: [] for key in (
        "future_ret5_open",
        "future_ret10_open",
        "future_close10_from_entry",
        "future_max_high10",
        "future_max_open10",
        "future_min_open10",
        "future_min_low10",
        "future_giveback_high_to_exit",
        "future_open_path_vol10",
        "future_path_efficiency10",
    )}
    cfg = R86.LABEL_VARIANTS[LABEL_VARIANT]
    for session in R74.RT.due_sessions(market, start, end):
        idx = market["date_index"][session]
        if idx < R74.SEQ_LEN + 20:
            continue
        ordered = ordered_liquidity_cols(market, amount20_arr, idx)
        cols = ordered[lo:hi]
        if len(cols) < 30:
            continue
        x_all, base_all = R74.build_path_features(feats, idx, cols)
        records = []
        for local_pos, col in enumerate(cols):
            y = R74.RT.future_return(market, idx, col)
            stats = R86.path_stats(market, session, str(market["symbols"][col]))
            if y is None or stats is None or not all(np.isfinite(v) for v in stats.values()):
                continue
            records.append((local_pos, col, float(y), stats))
        if len(records) < 30:
            continue
        returns = np.asarray([r[2] for r in records], dtype=float)
        ranks = pd.Series(returns).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        pools[session] = [str(market["symbols"][col]) for _, col, _, _ in records]
        for rank_value, (local_pos, col, y, stats) in zip(ranks, records, strict=True):
            xs.append(x_all[local_pos])
            raw_y.append(y)
            label_rank.append(float(rank_value))
            event_label.append(R86.event_label(stats, cfg))
            sessions.append(session)
            symbols.append(str(market["symbols"][col]))
            base_scores.append(float(base_all[local_pos]))
            for key in stats_rows:
                stats_rows[key].append(float(stats[key]))
    out: dict[str, Any] = {
        "x": np.asarray(xs, dtype=np.float32),
        "raw_y": np.asarray(raw_y, dtype=np.float32),
        "label_rank": np.asarray(label_rank, dtype=np.float32),
        "event_label": np.asarray(event_label, dtype=np.int8),
        "sessions": sessions,
        "symbols": symbols,
        "base_score": np.asarray(base_scores, dtype=np.float32),
        "pools": pools,
    }
    for key, values in stats_rows.items():
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def merge_splits(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    out = {
        "sessions": train["sessions"] + valid["sessions"],
        "symbols": train["symbols"] + valid["symbols"],
        "pools": {**train["pools"], **valid["pools"]},
    }
    for key, value in train.items():
        if key in out or key in {"sessions", "symbols", "pools"}:
            continue
        if isinstance(value, np.ndarray):
            out[key] = np.concatenate([value, valid[key]], axis=0)
    return out


def train_domain_scores(splits: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    raw_x = {name: R83.transform_x(split["x"], "real_path") for name, split in splits.items()}
    mu, sd = R83.standardizer(raw_x["train"])
    x_by_split = {name: R83.apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
    model, history = R86.train_model(x_by_split["train"], x_by_split["valid"], splits["train"], splits["valid"], TEMPERATURE, SEED + int(TEMPERATURE * 1000))
    return {name: R86.predict(model, x_by_split[name]) for name in splits}, history


def evaluate(split: dict[str, Any], scores: np.ndarray, market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return {
        "return_metrics": R74.metric_from_scores(split, scores),
        "event_metrics": R86.event_metrics(split, scores),
        "event_pair": R86.event_pair_auc(scores, split),
        "replay": R74.replay_summary(split, scores, market, start, end),
    }


def event_soil(split: dict[str, Any]) -> dict[str, Any]:
    labels = split["event_label"]
    return {
        "samples": int(len(labels)),
        "sessions": int(len(set(split["sessions"]))),
        "positive_rate": safe_mean(labels == 1),
        "negative_rate": safe_mean(labels == -1),
        "ret5_mean": safe_mean(split["raw_y"]),
        "future_max_high10_mean": safe_mean(split["future_max_high10"]),
        "future_ret10_open_mean": safe_mean(split["future_ret10_open"]),
    }


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def compact(domain: str, evaluated: dict[str, Any], soil: dict[str, Any], random_fwd: dict[str, float]) -> dict[str, Any]:
    valid = evaluated["valid"]
    dev = evaluated["dev"]
    forward = evaluated["forward"]
    return {
        "domain": domain,
        "valid_top10_ret5": valid["return_metrics"]["top_returns"]["top10"],
        "valid_top20_ret5": valid["return_metrics"]["top_returns"]["top20"],
        "valid_hit": valid["event_metrics"]["event_hit_rate"]["top10"],
        "valid_fail": valid["event_metrics"]["event_fail_rate"]["top10"],
        "dev_multiple": dev["replay"]["final_multiple"],
        "dev_all_years_positive": dev["replay"]["all_years_positive"],
        "dev_annual_returns": dev["replay"]["annual_returns"],
        "forward_multiple": forward["replay"]["final_multiple"],
        "forward_top10_ret5": forward["return_metrics"]["top_returns"]["top10"],
        "forward_top20_ret5": forward["return_metrics"]["top_returns"]["top20"],
        "forward_hit": forward["event_metrics"]["event_hit_rate"]["top10"],
        "forward_fail": forward["event_metrics"]["event_fail_rate"]["top10"],
        "forward_remove_best_3": forward["replay"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
        "forward_avg_position": forward["replay"]["avg_position_count"],
        "forward_random_p95": random_fwd["p95"],
        "train_soil": soil["train"],
        "valid_soil": soil["valid"],
        "forward_soil": soil["forward"],
        "selection_score": 0.45 * valid["return_metrics"]["top_returns"]["top10"] + 0.30 * valid["return_metrics"]["top_returns"]["top20"] + 0.15 * valid["event_metrics"]["event_hit_rate"]["top10"] - 0.10 * valid["event_metrics"]["event_fail_rate"]["top10"],
    }


def main() -> None:
    started = time.perf_counter()
    R86.set_seed(SEED)
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    results = []
    full_results = {}
    for domain in POOL_DOMAINS:
        train = build_domain_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END, domain)
        valid = build_domain_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END, domain)
        forward = build_domain_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END, domain)
        dev = merge_splits(train, valid)
        splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
        scores, history = train_domain_scores(splits)
        evaluated = {
            "train": evaluate(train, scores["train"], market, R74.TRAIN_START, R74.TRAIN_END),
            "valid": evaluate(valid, scores["valid"], market, R74.VALID_START, R74.VALID_END),
            "dev": evaluate(dev, scores["dev"], market, R74.DEV_START, R74.DEV_END),
            "forward": evaluate(forward, scores["forward"], market, R74.FWD_START, R74.FWD_END),
        }
        soil = {name: event_soil(split) for name, split in splits.items()}
        rand = random_summary(forward, market)
        row = compact(domain, evaluated, soil, rand)
        results.append(row)
        full_results[domain] = {"history_tail": history[-6:], "soil": soil, "evaluated": evaluated, "forward_random": rand}
    leaderboard = sorted(results, key=lambda row: row["selection_score"], reverse=True)
    selected = leaderboard[0]
    best_forward = max(leaderboard, key=lambda row: row["forward_multiple"])
    verdict = "explosive_path_liquidity_tier_domain_shift_not_enough"
    if selected["dev_multiple"] >= 20.0 and selected["dev_all_years_positive"] and selected["forward_multiple"] > selected["forward_random_p95"] and selected["forward_remove_best_3"] > 1.0 and selected["forward_avg_position"] > 5:
        verdict = "explosive_path_liquidity_tier_domain_shift_candidate_needs_walkforward"
    out = {
        "experiment": "explosive_path_liquidity_tier_domain_shift_v1",
        "method": "train_same_loose_t0_07_explosive_path_contrastive_model_per_liquidity_domain_and_compare_event_to_return_conversion_under_valid_selection_forward_evaluation",
        "params": {"label_variant": LABEL_VARIANT, "temperature": TEMPERATURE, "pool_domains": POOL_DOMAINS, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round86_script": sha256(R86_PATH), "round88_summary": sha256(ROOT / "explosive_path_right_tail_protection_router_v1_summary.json")},
        "leaderboard": leaderboard,
        "selected_by_valid_domain_score": selected,
        "best_by_forward_multiple": best_forward,
        "results": full_results,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected": selected, "best_by_forward_multiple": best_forward, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
