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
OUT = ROOT / "daily_path_memory_residual_placebo_v1_summary.json"

SEED = 20260714
EPOCHS = 22
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R74 = load_module("daily_path_residual_v1_for_placebo", R74_PATH)


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
    if variant == "no_path_stats":
        out[:, 11:66] = 0.0
        return out
    if variant == "base_liq_only":
        keep = [10, 66, 67]
        mask = np.zeros(out.shape[1], dtype=bool)
        mask[keep] = True
        out[:, ~mask] = 0.0
        return out
    if variant == "last_day_only":
        out[:, 11:66] = 0.0
        out[:, 66:68] = 0.0
        return out
    raise ValueError(variant)


def shifted_target(train: dict[str, Any], base_target: np.ndarray) -> np.ndarray:
    by_session: dict[str, np.ndarray] = {}
    for i, session in enumerate(train["sessions"]):
        by_session.setdefault(session, []).append(i)
    by_session_arr = {session: np.asarray(indices, dtype=int) for session, indices in by_session.items()}
    sessions = sorted(by_session_arr)
    out = base_target.copy()
    for pos, session in enumerate(sessions[:-1]):
        cur = by_session_arr[session]
        nxt = by_session_arr[sessions[pos + 1]]
        n = min(len(cur), len(nxt))
        out[cur[:n]] = base_target[nxt[:n]]
        if len(cur) > n:
            out[cur[n:]] = float(np.mean(base_target[nxt]))
    return out.astype(np.float32)


def train_model(train_x: np.ndarray, valid_x: np.ndarray, train_target: np.ndarray, valid: dict[str, Any]) -> tuple[R74.ResidualMLP, list[dict[str, float]]]:
    set_seed(SEED)
    model = R74.ResidualMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_target.astype(np.float32))), batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e18
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = nn.functional.smooth_l1_loss(pred, by, beta=0.10)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        valid_scores = R74.predict(model, valid_x.astype(np.float32))
        valid_metric = R74.metric_from_scores(valid, valid_scores)
        score = valid_metric["top_returns"]["top20"] + 0.25 * valid_metric["top_returns"]["top10"] + 0.005 * valid_metric["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_score": score, "valid_top20": valid_metric["top_returns"]["top20"], "valid_top10": valid_metric["top_returns"]["top10"], "valid_ic": valid_metric["mean_rank_ic"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(R74.RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def evaluate_variant(name: str, train: dict[str, Any], valid: dict[str, Any], dev: dict[str, Any], forward: dict[str, Any], null_scores: dict[str, np.ndarray], market: dict[str, Any]) -> dict[str, Any]:
    base_target = (train["label_rank"] - null_scores["train"]).astype(np.float32)
    rng = np.random.default_rng(SEED + 17)
    if name == "permuted_train_residual_label":
        train_target = rng.permutation(base_target).astype(np.float32)
        x_variant = "real_path"
    elif name == "next_session_shift_label":
        train_target = shifted_target(train, base_target)
        x_variant = "real_path"
    else:
        train_target = base_target
        x_variant = name

    train_x = transform_x(train["x"], x_variant)
    valid_x = transform_x(valid["x"], x_variant)
    dev_x = transform_x(dev["x"], x_variant)
    forward_x = transform_x(forward["x"], x_variant)
    model, history = train_model(train_x, valid_x, train_target, valid)
    split_x = {"train": train_x, "valid": valid_x, "dev": dev_x, "forward": forward_x}
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    metrics = {}
    scores = {}
    for split_name, split in splits.items():
        score = R74.predict(model, split_x[split_name].astype(np.float32))
        scores[split_name] = score
        metrics[split_name] = R74.metric_from_scores(split, score)
    replay = {
        "dev": R74.replay_summary(dev, scores["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": R74.replay_summary(forward, scores["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    return {"variant": name, "x_variant": x_variant, "metrics": metrics, "replay": replay, "history_tail": history[-6:]}


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
    variants = ["real_path", "no_path_stats", "last_day_only", "base_liq_only", "permuted_train_residual_label", "next_session_shift_label"]
    results = [evaluate_variant(name, train, valid, dev, forward, null_scores, market) for name in variants]
    leaderboard = sorted(
        [
            {
                "variant": item["variant"],
                "valid_top20": item["metrics"]["valid"]["top_returns"]["top20"],
                "valid_top10": item["metrics"]["valid"]["top_returns"]["top10"],
                "valid_ic": item["metrics"]["valid"]["mean_rank_ic"],
                "dev_top20": item["metrics"]["dev"]["top_returns"]["top20"],
                "forward_top20": item["metrics"]["forward"]["top_returns"]["top20"],
                "forward_top10": item["metrics"]["forward"]["top_returns"]["top10"],
                "forward_ic": item["metrics"]["forward"]["mean_rank_ic"],
                "forward_multiple": item["replay"]["forward"]["final_multiple"],
                "forward_remove_best_3": item["replay"]["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "dev_multiple": item["replay"]["dev"]["final_multiple"],
                "dev_all_years_positive": item["replay"]["dev"]["all_years_positive"],
            }
            for item in results
        ],
        key=lambda row: row["valid_top20"] + 0.25 * row["valid_top10"] + 0.005 * row["valid_ic"],
        reverse=True,
    )
    real = next(row for row in leaderboard if row["variant"] == "real_path")
    placebo_best = max((row for row in leaderboard if row["variant"] != "real_path"), key=lambda row: row["forward_top20"])
    verdict = "daily_path_memory_residual_placebo_inconclusive"
    if real["valid_top20"] > placebo_best["valid_top20"] + 0.001 and real["forward_top20"] > placebo_best["forward_top20"] + 0.001:
        verdict = "daily_path_memory_residual_real_path_survives_placebo_needs_right_tail"
    elif placebo_best["forward_top20"] >= real["forward_top20"] or placebo_best["valid_top20"] >= real["valid_top20"]:
        verdict = "daily_path_memory_residual_placebo_too_strong"
    out = {
        "experiment": "daily_path_memory_residual_placebo_v1",
        "method": "same_split_null_eval_as_round74_real_path_vs_feature_and_label_placebos",
        "params": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED, "variants": variants},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round74_script": sha256(R74_PATH), "round74_summary": sha256(ROOT / "cross_sectional_daily_path_memory_residual_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "null_meta": null_meta,
        "leaderboard": leaderboard,
        "results": results,
        "real_row": real,
        "best_placebo_by_forward_top20": placebo_best,
        "forward_random_pool500_top10": random_summary(forward, market),
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "leaderboard": leaderboard, "real_row": real, "best_placebo_by_forward_top20": placebo_best, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
