from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
R74_PATH = ROOT / "analyze_cross_sectional_daily_path_memory_residual_v1.py"
OUT = ROOT / "daily_path_right_tail_pairwise_v1_summary.json"

SEED = 20260714
EPOCHS = 24
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4
TOP_THRESHOLD = 0.30
BOTTOM_THRESHOLD = -0.20


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R74 = load_module("daily_path_residual_v1_for_right_tail", R74_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def transform_x(x: np.ndarray, variant: str) -> np.ndarray:
    out = x.copy()
    if variant == "real_path":
        return out
    if variant == "base_liq_only":
        keep = [10, 66, 67]
        mask = np.zeros(out.shape[1], dtype=bool)
        mask[keep] = True
        out[:, ~mask] = 0.0
        return out
    raise ValueError(variant)


def session_groups(split: dict[str, Any]) -> list[np.ndarray]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(by_session.items())]


def train_targets(train: dict[str, Any], null_scores: np.ndarray, objective: str) -> tuple[np.ndarray, np.ndarray]:
    residual = (train["label_rank"] - null_scores).astype(np.float32)
    if objective == "pointwise_residual":
        weight = np.ones_like(residual, dtype=np.float32)
        weight[np.abs(train["label_rank"]) >= TOP_THRESHOLD] = 2.0
        return residual, weight
    if objective == "binary_top_tail":
        target = (train["label_rank"] >= TOP_THRESHOLD).astype(np.float32)
        weight = np.full_like(target, 0.35, dtype=np.float32)
        weight[train["label_rank"] >= TOP_THRESHOLD] = 2.5
        weight[train["label_rank"] <= BOTTOM_THRESHOLD] = 1.0
        return target, weight
    if objective == "top_minus_bottom":
        target = np.zeros_like(residual, dtype=np.float32)
        target[train["label_rank"] >= TOP_THRESHOLD] = 1.0
        target[train["label_rank"] <= BOTTOM_THRESHOLD] = -1.0
        weight = np.full_like(target, 0.25, dtype=np.float32)
        weight[target != 0.0] = 2.0
        return target, weight
    raise ValueError(objective)


def loss_for(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "binary_top_tail":
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    else:
        loss = nn.functional.smooth_l1_loss(pred, target, beta=0.10, reduction="none")
    return (loss * weight).mean()


def train_model(train_x: np.ndarray, valid_x: np.ndarray, train_target: np.ndarray, train_weight: np.ndarray, valid: dict[str, Any], objective: str) -> tuple[R74.ResidualMLP, list[dict[str, float]]]:
    set_seed(SEED)
    model = R74.ResidualMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ds = TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_target.astype(np.float32)), torch.from_numpy(train_weight.astype(np.float32)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e18
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by, bw in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_for(pred, by, bw, objective)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        valid_scores = R74.predict(model, valid_x.astype(np.float32))
        valid_metric = R74.metric_from_scores(valid, valid_scores)
        score = 0.55 * valid_metric["top_returns"]["top10"] + 0.35 * valid_metric["top_returns"]["top20"] + 0.10 * valid_metric["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_right_tail_score": score, "valid_top10": valid_metric["top_returns"]["top10"], "valid_top20": valid_metric["top_returns"]["top20"], "valid_ic": valid_metric["mean_rank_ic"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, x_by_split: dict[str, np.ndarray], splits: dict[str, dict[str, Any]], market: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    scores = {}
    for name, split in splits.items():
        score = R74.predict(model, x_by_split[name].astype(np.float32))
        scores[name] = score
        metrics[name] = R74.metric_from_scores(split, score)
    replay = {
        "dev": R74.replay_summary(splits["dev"], scores["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": R74.replay_summary(splits["forward"], scores["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    return {"metrics": metrics, "replay": replay}


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(R74.RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    train = R74.build_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    valid = R74.build_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    dev = R74.merge_splits(train, valid)
    forward = R74.build_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    null_scores, null_meta = R74.build_null(train, splits)
    configs = []
    for x_variant in ["real_path", "base_liq_only"]:
        for objective in ["pointwise_residual", "binary_top_tail", "top_minus_bottom"]:
            configs.append((x_variant, objective))
    results = []
    for x_variant, objective in configs:
        x_by_split = {name: transform_x(split["x"], x_variant) for name, split in splits.items()}
        target, weight = train_targets(train, null_scores["train"], objective)
        model, history = train_model(x_by_split["train"], x_by_split["valid"], target, weight, valid, objective)
        evaluated = evaluate_model(model, x_by_split, splits, market)
        row = {
            "config": f"{x_variant}_{objective}",
            "x_variant": x_variant,
            "objective": objective,
            "history_tail": history[-6:],
            "metrics": evaluated["metrics"],
            "replay": evaluated["replay"],
        }
        results.append(row)
    leaderboard = sorted(
        [
            {
                "config": item["config"],
                "x_variant": item["x_variant"],
                "objective": item["objective"],
                "valid_top10": item["metrics"]["valid"]["top_returns"]["top10"],
                "valid_top20": item["metrics"]["valid"]["top_returns"]["top20"],
                "valid_ic": item["metrics"]["valid"]["mean_rank_ic"],
                "dev_top20": item["metrics"]["dev"]["top_returns"]["top20"],
                "forward_top10": item["metrics"]["forward"]["top_returns"]["top10"],
                "forward_top20": item["metrics"]["forward"]["top_returns"]["top20"],
                "forward_ic": item["metrics"]["forward"]["mean_rank_ic"],
                "dev_multiple": item["replay"]["dev"]["final_multiple"],
                "dev_all_years_positive": item["replay"]["dev"]["all_years_positive"],
                "forward_multiple": item["replay"]["forward"]["final_multiple"],
                "forward_remove_best_3": item["replay"]["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
            }
            for item in results
        ],
        key=lambda row: 0.55 * row["valid_top10"] + 0.35 * row["valid_top20"] + 0.10 * row["valid_ic"],
        reverse=True,
    )
    forward_random = random_summary(forward, market)
    selected = leaderboard[0]
    best_real = max((row for row in leaderboard if row["x_variant"] == "real_path"), key=lambda row: row["forward_multiple"])
    best_base = max((row for row in leaderboard if row["x_variant"] == "base_liq_only"), key=lambda row: row["forward_multiple"])
    verdict = "daily_path_right_tail_pairwise_not_enough"
    if (
        best_real["forward_multiple"] > best_base["forward_multiple"]
        and best_real["forward_multiple"] > forward_random["p95"]
        and best_real["forward_remove_best_3"] > 1.0
        and best_real["dev_all_years_positive"]
    ):
        verdict = "daily_path_right_tail_pairwise_candidate_needs_walkforward"
    out = {
        "experiment": "daily_path_right_tail_pairwise_v1",
        "method": "daily_60d_path_right_tail_objectives_pointwise_binary_top_tail_top_minus_bottom_vs_base_liq_only",
        "params": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "top_threshold": TOP_THRESHOLD, "bottom_threshold": BOTTOM_THRESHOLD, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round74_script": sha256(R74_PATH), "round75_summary": sha256(ROOT / "daily_path_memory_residual_placebo_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "null_meta": null_meta,
        "leaderboard": leaderboard,
        "results": results,
        "selected_by_valid_right_tail": selected,
        "best_real_by_forward_multiple": best_real,
        "best_base_by_forward_multiple": best_base,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "leaderboard": leaderboard, "best_real_by_forward_multiple": best_real, "best_base_by_forward_multiple": best_base, "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
