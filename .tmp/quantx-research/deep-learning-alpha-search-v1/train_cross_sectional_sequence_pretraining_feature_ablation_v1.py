from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_sequence_pretraining_feature_ablation_v1_summary.json"

SEED = 20260714
EPOCHS = 4
RANDOM_TRIALS = 80

VARIANTS = {
    "full_stock_plus_regime": {"stock_cols": list(range(11)), "keep_regime": True},
    "stock_raw_path_no_regime": {"stock_cols": [0, 1, 2, 3, 4, 5, 6, 7], "keep_regime": False},
    "stock_cross_rank_no_regime": {"stock_cols": [8, 9, 10], "keep_regime": False},
    "stock_raw_plus_rank_no_regime": {"stock_cols": list(range(11)), "keep_regime": False},
    "regime_token_only": {"stock_cols": [], "keep_regime": True},
    "rank_plus_regime": {"stock_cols": [8, 9, 10], "keep_regime": True},
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


YI = load_module("year_invariant_rank_transformer_for_feature_ablation", ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py")
POS = YI.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_split(split: dict[str, Any], stock_cols: list[int], keep_regime: bool) -> dict[str, Any]:
    out = dict(split)
    x = np.zeros_like(split["x"], dtype=np.float32)
    if stock_cols:
        x[:, :-1, stock_cols] = split["x"][:, :-1, stock_cols]
    if keep_regime:
        x[:, -1, :] = split["x"][:, -1, :]
    out["x"] = x
    return out


def random_forward_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [YI.replay_split(YI.random_selections(forward["pools"], rng), market, YI.FWD_START, YI.FWD_END)["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return YI.percentile_summary(values)


def train_and_eval_variant(name: str, mask: dict[str, Any], splits: dict[str, dict[str, Any]], market: dict[str, Any], device: torch.device) -> dict[str, Any]:
    set_seed(SEED)
    masked = {split_name: masked_split(split, mask["stock_cols"], mask["keep_regime"]) for split_name, split in splits.items()}
    trained = YI.train_model(masked["train"], masked["valid"], masked["dev"], market, device)
    model = trained["model"]
    results = {}
    ranges = {
        "train": (YI.TRAIN_START, YI.TRAIN_END),
        "valid": (YI.VALID_START, YI.VALID_END),
        "dev": (YI.DEV_START, YI.DEV_END),
        "forward": (YI.FWD_START, YI.FWD_END),
    }
    for split_name, split in masked.items():
        scores = YI.predict_scores(model, split["x"], device)
        selections = YI.selections_from_scores(split, scores)
        start, end = ranges[split_name]
        results[split_name] = {
            "summary": YI.replay_split(selections, market, start, end),
            "rank_metrics": YI.rank_ic(split, scores),
        }
    dev_summary = results["dev"]["summary"]
    forward_summary = results["forward"]["summary"]
    forward_rank = results["forward"]["rank_metrics"]
    years = list(dev_summary.get("annual_returns", {}).values())
    worst_year = min(years) if years else 0.0
    return {
        "variant": name,
        "mask": mask,
        "best_epoch": trained["best_epoch"],
        "best_selection_score": trained["best_selection_score"],
        "training_history": trained["history"],
        "results": results,
        "score": float(dev_summary["final_multiple"] - 1.0 + 0.75 * (forward_summary["final_multiple"] - 1.0) + 2.0 * worst_year + 4.0 * forward_rank["mean_rank_ic"]),
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    YI.EPOCHS = EPOCHS
    YI.RANDOM_TRIALS = RANDOM_TRIALS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    market = POS.load_market()
    feats = YI.build_feature_tensor(market)
    amount20_arr = YI.amount20(market)
    base_splits = {
        "train": YI.build_split_samples(market, feats, amount20_arr, YI.TRAIN_START, YI.TRAIN_END),
        "valid": YI.build_split_samples(market, feats, amount20_arr, YI.VALID_START, YI.VALID_END),
        "dev": YI.build_split_samples(market, feats, amount20_arr, YI.DEV_START, YI.DEV_END),
        "forward": YI.build_split_samples(market, feats, amount20_arr, YI.FWD_START, YI.FWD_END),
    }
    for split in base_splits.values():
        YI.attach_rank_targets(split)
        YI.attach_rank_weights(split)
    forward_random = random_forward_summary(base_splits["forward"], market)
    variants = [train_and_eval_variant(name, deepcopy(mask), base_splits, market, device) for name, mask in VARIANTS.items()]
    leaderboard = sorted(
        [
            {
                "variant": item["variant"],
                "best_epoch": item["best_epoch"],
                "score": item["score"],
                "dev_multiple": item["results"]["dev"]["summary"]["final_multiple"],
                "dev_total_return": item["results"]["dev"]["summary"]["total_return"],
                "dev_all_years_positive": item["results"]["dev"]["summary"]["all_years_positive"],
                "dev_annual_returns": item["results"]["dev"]["summary"]["annual_returns"],
                "dev_avg_position_count": item["results"]["dev"]["summary"]["avg_position_count"],
                "dev_rank_ic": item["results"]["dev"]["rank_metrics"]["mean_rank_ic"],
                "dev_top10_label": item["results"]["dev"]["rank_metrics"]["mean_top10_raw_label"],
                "forward_multiple": item["results"]["forward"]["summary"]["final_multiple"],
                "forward_total_return": item["results"]["forward"]["summary"]["total_return"],
                "forward_max_drawdown": item["results"]["forward"]["summary"]["max_drawdown"],
                "forward_avg_position_count": item["results"]["forward"]["summary"]["avg_position_count"],
                "forward_remove_best_3": item["results"]["forward"]["summary"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "forward_rank_ic": item["results"]["forward"]["rank_metrics"]["mean_rank_ic"],
                "forward_top10_label": item["results"]["forward"]["rank_metrics"]["mean_top10_raw_label"],
                "beats_forward_random_p95": item["results"]["forward"]["summary"]["final_multiple"] > forward_random["p95"],
            }
            for item in variants
        ],
        key=lambda row: -row["score"],
    )
    best = leaderboard[0] if leaderboard else {}
    verdict = "sequence_feature_ablation_not_enough"
    if best and best["dev_multiple"] >= 20.0 and best["dev_all_years_positive"] and best["forward_multiple"] > forward_random["p95"] and best["forward_avg_position_count"] > 5 and best["forward_remove_best_3"] > 1.0:
        verdict = "sequence_feature_ablation_candidate_needs_walkforward_expansion"
    out = {
        "experiment": "cross_sectional_sequence_pretraining_feature_ablation_v1",
        "method": "same_tiny_temporal_transformer_same_pool500_rank_target_ablate_stock_raw_rank_regime_tokens",
        "params": {
            "train": [YI.TRAIN_START, YI.TRAIN_END],
            "valid": [YI.VALID_START, YI.VALID_END],
            "dev": [YI.DEV_START, YI.DEV_END],
            "forward": [YI.FWD_START, YI.FWD_END],
            "seq_len": YI.SEQ_LEN,
            "model_seq_len": YI.MODEL_SEQ_LEN,
            "pool_size": YI.POOL_SIZE,
            "top_k": YI.TOP_K,
            "epochs": EPOCHS,
            "random_trials": RANDOM_TRIALS,
            "seed": SEED,
            "variants": VARIANTS,
            "device": str(device),
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "year_invariant_script": sha256(ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py"),
            "round53_summary": sha256(ROOT / "trend_exhaustion_regime_boundary_scan_v1_summary.json"),
        },
        "sample_counts": {name: len(split["sessions"]) for name, split in base_splits.items()},
        "session_counts": {name: len(set(split["sessions"])) for name, split in base_splits.items()},
        "forward_random_pool500_top10": forward_random,
        "leaderboard": leaderboard,
        "variants": variants,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "best": best,
        "leaderboard": leaderboard,
        "forward_random": forward_random,
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
