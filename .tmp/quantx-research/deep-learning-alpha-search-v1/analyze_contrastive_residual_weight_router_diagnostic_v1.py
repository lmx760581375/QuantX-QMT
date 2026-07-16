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
R83_PATH = ROOT / "analyze_contrastive_pairwise_diffusion_ranker_v1.py"
OUT = ROOT / "contrastive_residual_weight_router_diagnostic_v1_summary.json"

TEMPERATURE = 0.15
HIGH_WEIGHTS = (0.2, 0.3, 0.4, 0.6, 1.0)
STATIC_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)
RANDOM_TRIALS = 200
SEED = 20260714
FEATURES = (
    "model_base_rank_corr",
    "model_base_top20_overlap",
    "model_rank_dispersion",
    "base_rank_dispersion",
    "model_top10_mean_rank",
    "base_top10_model_rank_mean",
    "mkt_ret20_mean",
    "mkt_ret60_mean",
    "mkt_up20_ratio",
    "mkt_amount_rank_top_mean",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R83 = load_module("contrastive_pairwise_ranker_for_router_diag", R83_PATH)
R74 = R83.R74


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def market_features(market: dict[str, Any], session: str, symbols: list[str]) -> dict[str, float]:
    idx = market["date_index"][session]
    cols = np.asarray([market["symbol_index"][sym] for sym in symbols if sym in market["symbol_index"]], dtype=int)
    if len(cols) == 0 or idx < 60:
        return {"mkt_ret20_mean": 0.0, "mkt_ret60_mean": 0.0, "mkt_up20_ratio": 0.0, "mkt_amount_rank_top_mean": 0.0}
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    ret20 = safe_div(close[idx, cols], close[idx - 20, cols]) - 1.0
    ret60 = safe_div(close[idx, cols], close[idx - 60, cols]) - 1.0
    amount20 = np.nanmean(np.where(np.isfinite(vwap[idx - 19:idx + 1, :][:, cols]) & (vwap[idx - 19:idx + 1, :][:, cols] > 0), vwap[idx - 19:idx + 1, :][:, cols], close[idx - 19:idx + 1, :][:, cols]) * volume[idx - 19:idx + 1, :][:, cols], axis=0)
    return {
        "mkt_ret20_mean": safe_mean(ret20),
        "mkt_ret60_mean": safe_mean(ret60),
        "mkt_up20_ratio": safe_mean(ret20 > 0.0),
        "mkt_amount_rank_top_mean": safe_mean(rank01(amount20)[-min(20, len(amount20)):]),
    }


def train_model_scores(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_x = {name: R83.transform_x(split["x"], "real_path") for name, split in splits.items()}
    mu, sd = R83.standardizer(raw_x["train"])
    x_by_split = {name: R83.apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
    model, history = R83.train_model(x_by_split["train"], x_by_split["valid"], splits["train"], splits["valid"], TEMPERATURE, SEED + int(TEMPERATURE * 1000))
    scores = {name: R83.predict(model, x_by_split[name]) for name in splits}
    return {"scores": scores, "history_tail": history[-8:]}


def build_session_rows(market: dict[str, Any], split: dict[str, Any], base_rank: np.ndarray, model_rank: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for session, indices in session_indices(split).items():
        b = base_rank[indices]
        m = model_rank[indices]
        corr = 0.0
        if len(indices) > 3 and np.std(b) > 1e-12 and np.std(m) > 1e-12:
            corr = float(np.corrcoef(b, m)[0, 1])
        top_base = set(indices[np.argsort(-b)[:20]])
        top_model_local = np.argsort(-m)[:20]
        top_model = set(indices[top_model_local])
        top_base_local = np.argsort(-b)[:20]
        symbols = [split["symbols"][int(i)] for i in indices]
        feats = {
            "model_base_rank_corr": corr,
            "model_base_top20_overlap": len(top_base & top_model) / 20.0,
            "model_rank_dispersion": float(np.std(m)),
            "base_rank_dispersion": float(np.std(b)),
            "model_top10_mean_rank": safe_mean(m[np.argsort(-m)[:10]]),
            "base_top10_model_rank_mean": safe_mean(m[top_base_local[:10]]),
            **market_features(market, session, symbols),
        }
        rows.append({"session": session, "year": session[:4], **feats})
    return rows


def combine_scores(split: dict[str, Any], base_rank: np.ndarray, model_rank: np.ndarray, weight_by_session: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(model_rank), dtype=float)
    for session, indices in session_indices(split).items():
        weight = float(weight_by_session.get(session, 0.0))
        out[indices] = base_rank[indices] + weight * model_rank[indices]
    return out


def constant_weight(rows: list[dict[str, Any]], weight: float) -> dict[str, float]:
    return {row["session"]: weight for row in rows}


def gate_weight(rows: list[dict[str, Any]], feature: str, op: str, threshold: float, high_weight: float) -> dict[str, float]:
    out = {}
    for row in rows:
        active = row[feature] >= threshold if op == "high" else row[feature] <= threshold
        out[row["session"]] = high_weight if active else 0.0
    return out


def eval_weight_rule(splits: dict[str, dict[str, Any]], base_rank: dict[str, np.ndarray], model_rank: dict[str, np.ndarray], rows_by_split: dict[str, list[dict[str, Any]]], market: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    score_by_split = {}
    weight_by_split = {}
    for name, rows in rows_by_split.items():
        if rule["kind"] == "static":
            weights = constant_weight(rows, rule["weight"])
        else:
            weights = gate_weight(rows, rule["feature"], rule["op"], rule["threshold"], rule["high_weight"])
        weight_by_split[name] = weights
        score_by_split[name] = combine_scores(splits[name], base_rank[name], model_rank[name], weights)
    metrics = {name: R74.metric_from_scores(splits[name], score_by_split[name]) for name in splits}
    replay = {
        "dev": R74.replay_summary(splits["dev"], score_by_split["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": R74.replay_summary(splits["forward"], score_by_split["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    active = {
        name: {
            "avg_weight": safe_mean(list(weights.values())),
            "active_rate": safe_mean([w > 0.0 for w in weights.values()]),
        }
        for name, weights in weight_by_split.items()
    }
    return {"metrics": metrics, "replay": replay, "active": active, "weights": weight_by_split}


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def main() -> None:
    started = time.perf_counter()
    R83.set_seed(SEED)
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    train = R74.build_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    valid = R74.build_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    dev = R74.merge_splits(train, valid)
    forward = R74.build_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    trained = train_model_scores(splits)
    model_rank = {name: session_rank_scores(split, trained["scores"][name]) for name, split in splits.items()}
    base_rank = {name: session_rank_scores(split, split["base_score"]) for name, split in splits.items()}
    rows_by_split = {name: build_session_rows(market, split, base_rank[name], model_rank[name]) for name, split in splits.items()}

    rules = [{"kind": "static", "name": f"static_w{w}", "weight": w} for w in STATIC_WEIGHTS]
    for feature in FEATURES:
        values = np.asarray([row[feature] for row in rows_by_split["train"]], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 20 or np.std(values) <= 1e-12:
            continue
        for q in QUANTILES:
            threshold = float(np.quantile(values, q))
            for op in ("high", "low"):
                for high_weight in HIGH_WEIGHTS:
                    rules.append({"kind": "gate", "name": f"{feature}_{op}_q{q:.2f}_w{high_weight}", "feature": feature, "op": op, "threshold": threshold, "high_weight": high_weight})

    results = []
    for rule in rules:
        ev = eval_weight_rule(splits, base_rank, model_rank, rows_by_split, market, rule)
        results.append({
            "rule": rule,
            "eval": ev,
            "valid_score": ev["metrics"]["valid"]["top_returns"]["top20"] + 0.30 * ev["metrics"]["valid"]["top_returns"]["top10"] + 0.08 * ev["metrics"]["valid"]["mean_rank_ic"] + 0.02 * ev["active"]["valid"]["avg_weight"],
        })
    results = sorted(results, key=lambda item: item["valid_score"], reverse=True)
    leaderboard = []
    for item in results:
        ev = item["eval"]
        leaderboard.append({
            "rule": item["rule"],
            "valid_score": item["valid_score"],
            "train_top20": ev["metrics"]["train"]["top_returns"]["top20"],
            "valid_top20": ev["metrics"]["valid"]["top_returns"]["top20"],
            "valid_top10": ev["metrics"]["valid"]["top_returns"]["top10"],
            "valid_ic": ev["metrics"]["valid"]["mean_rank_ic"],
            "dev_top20": ev["metrics"]["dev"]["top_returns"]["top20"],
            "dev_by_year_top20": ev["metrics"]["dev"]["by_year_top20"],
            "dev_multiple": ev["replay"]["dev"]["final_multiple"],
            "dev_all_years_positive": ev["replay"]["dev"]["all_years_positive"],
            "forward_top10": ev["metrics"]["forward"]["top_returns"]["top10"],
            "forward_top20": ev["metrics"]["forward"]["top_returns"]["top20"],
            "forward_ic": ev["metrics"]["forward"]["mean_rank_ic"],
            "forward_multiple": ev["replay"]["forward"]["final_multiple"],
            "forward_remove_best_3": ev["replay"]["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
            "forward_avg_position": ev["replay"]["forward"]["avg_position_count"],
            "avg_weight": {name: ev["active"][name]["avg_weight"] for name in ev["active"]},
            "active_rate": {name: ev["active"][name]["active_rate"] for name in ev["active"]},
        })
    selected = leaderboard[0]
    forward_random = random_summary(forward, market)
    verdict = "contrastive_residual_weight_router_no_stable_selector"
    if (
        selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > forward_random["p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "contrastive_residual_weight_router_candidate_needs_walkforward"
    out = {
        "experiment": "contrastive_residual_weight_router_diagnostic_v1",
        "method": "train_quantile_session_state_router_for_base_daily_plus_contrastive_residual_weight",
        "params": {"temperature": TEMPERATURE, "static_weights": STATIC_WEIGHTS, "high_weights": HIGH_WEIGHTS, "quantiles": QUANTILES, "features": FEATURES, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round83_script": sha256(R83_PATH), "round84_summary": sha256(ROOT / "contrastive_pairwise_anchor_residual_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "model_history_tail": trained["history_tail"],
        "leaderboard": leaderboard,
        "selected_by_valid_router": selected,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected_by_valid_router": selected, "leaderboard_top12": leaderboard[:12], "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
