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
OUT = ROOT / "contrastive_pairwise_anchor_residual_v1_summary.json"

TEMPERATURES = (0.15, 0.30)
ANCHORS = ("base_daily", "null_only", "model_only")
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
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


R83 = load_module("contrastive_pairwise_ranker_for_anchor_residual", R83_PATH)
R74 = R83.R74


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
    order = np.argsort(np.where(np.isfinite(arr), arr, fill), kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.linspace(-0.5, 0.5, len(arr), endpoint=True) if len(arr) > 1 else 0.0
    return ranks


def session_rank_scores(split: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=float)
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    for indices in by_session.values():
        idx = np.asarray(indices, dtype=int)
        out[idx] = rank01(scores[idx])
    return out


def pair_auc_by_split(scores: np.ndarray, split: dict[str, Any]) -> dict[str, float]:
    return R83.pairwise_auc(scores, split)


def combined_scores(split: dict[str, Any], anchor_scores: np.ndarray, model_scores: np.ndarray, anchor: str, weight: float) -> np.ndarray:
    model_rank = session_rank_scores(split, model_scores)
    if anchor == "model_only":
        return model_rank
    anchor_rank = session_rank_scores(split, anchor_scores)
    return anchor_rank + weight * model_rank


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def train_contrastive_models(splits: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw_x = {name: R83.transform_x(split["x"], "real_path") for name, split in splits.items()}
    mu, sd = R83.standardizer(raw_x["train"])
    x_by_split = {name: R83.apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
    models = {}
    for temp in TEMPERATURES:
        model, history = R83.train_model(x_by_split["train"], x_by_split["valid"], splits["train"], splits["valid"], temp, SEED + int(temp * 1000))
        scores = {name: R83.predict(model, x_by_split[name]) for name in splits}
        models[f"real_path_t{temp}"] = {"temperature": temp, "history_tail": history[-8:], "scores": scores, "pair": {name: pair_auc_by_split(score, splits[name]) for name, score in scores.items()}}
    return models


def eval_score(split: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    out = R74.metric_from_scores(split, scores)
    out["pair"] = pair_auc_by_split(scores, split)
    return out


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
    null_scores, null_meta = R74.build_null(train, splits)
    models = train_contrastive_models(splits)
    anchor_by_name = {
        "base_daily": {name: split["base_score"] for name, split in splits.items()},
        "null_only": null_scores,
    }
    rows = []
    for model_name, model_info in models.items():
        model_scores = model_info["scores"]
        for anchor in ANCHORS:
            weights = (0.0,) if anchor == "model_only" else WEIGHTS
            for weight in weights:
                score_by_split = {}
                for split_name, split in splits.items():
                    anchor_scores = np.zeros_like(model_scores[split_name]) if anchor == "model_only" else anchor_by_name[anchor][split_name]
                    score_by_split[split_name] = combined_scores(split, anchor_scores, model_scores[split_name], anchor, weight)
                metrics = {name: eval_score(split, score_by_split[name]) for name, split in splits.items()}
                replay = {
                    "dev": R74.replay_summary(dev, score_by_split["dev"], market, R74.DEV_START, R74.DEV_END),
                    "forward": R74.replay_summary(forward, score_by_split["forward"], market, R74.FWD_START, R74.FWD_END),
                }
                rows.append({
                    "model": model_name,
                    "temperature": model_info["temperature"],
                    "anchor": anchor,
                    "weight": weight,
                    "metrics": metrics,
                    "replay": replay,
                    "valid_score": 0.35 * metrics["valid"]["top_returns"]["top10"] + 0.30 * metrics["valid"]["top_returns"]["top20"] + 0.20 * metrics["valid"]["pair"]["pair_auc"] + 0.15 * metrics["valid"]["mean_rank_ic"],
                })
    rows = sorted(rows, key=lambda row: row["valid_score"], reverse=True)
    leaderboard = [
        {
            "model": row["model"],
            "temperature": row["temperature"],
            "anchor": row["anchor"],
            "weight": row["weight"],
            "valid_top10": row["metrics"]["valid"]["top_returns"]["top10"],
            "valid_top20": row["metrics"]["valid"]["top_returns"]["top20"],
            "valid_pair_auc": row["metrics"]["valid"]["pair"]["pair_auc"],
            "valid_ic": row["metrics"]["valid"]["mean_rank_ic"],
            "dev_top20": row["metrics"]["dev"]["top_returns"]["top20"],
            "dev_by_year_top20": row["metrics"]["dev"]["by_year_top20"],
            "dev_multiple": row["replay"]["dev"]["final_multiple"],
            "dev_all_years_positive": row["replay"]["dev"]["all_years_positive"],
            "forward_top10": row["metrics"]["forward"]["top_returns"]["top10"],
            "forward_top20": row["metrics"]["forward"]["top_returns"]["top20"],
            "forward_pair_auc": row["metrics"]["forward"]["pair"]["pair_auc"],
            "forward_ic": row["metrics"]["forward"]["mean_rank_ic"],
            "forward_multiple": row["replay"]["forward"]["final_multiple"],
            "forward_remove_best_3": row["replay"]["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
            "forward_avg_position": row["replay"]["forward"]["avg_position_count"],
            "valid_score": row["valid_score"],
        }
        for row in rows
    ]
    selected = leaderboard[0]
    forward_random = random_summary(forward, market)
    verdict = "contrastive_pairwise_anchor_residual_not_enough"
    if (
        selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > forward_random["p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "contrastive_pairwise_anchor_residual_candidate_needs_walkforward"
    out = {
        "experiment": "contrastive_pairwise_anchor_residual_v1",
        "method": "valid_selected_base_or_null_anchor_plus_contrastive_pairwise_residual_rank",
        "params": {"temperatures": TEMPERATURES, "anchors": ANCHORS, "weights": WEIGHTS, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round83_script": sha256(R83_PATH), "round83_summary": sha256(ROOT / "contrastive_pairwise_diffusion_ranker_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "null_meta": null_meta,
        "model_history": {name: {"temperature": item["temperature"], "history_tail": item["history_tail"], "pair": item["pair"]} for name, item in models.items()},
        "leaderboard": leaderboard,
        "selected_by_valid_anchor": selected,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected_by_valid_anchor": selected, "leaderboard_top12": leaderboard[:12], "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
