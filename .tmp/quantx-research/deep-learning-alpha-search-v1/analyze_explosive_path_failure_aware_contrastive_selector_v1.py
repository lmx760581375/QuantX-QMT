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
R86_PATH = ROOT / "analyze_explosive_path_contrastive_event_model_v1.py"
OUT = ROOT / "explosive_path_failure_aware_contrastive_selector_v1_summary.json"

SEED = 20260714
LABEL_VARIANT = "loose_explosive_holdable"
EPOCHS = 18
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4
MAX_POS_PER_SESSION = 48
MAX_NEG_PER_SESSION = 48
MAX_PAIRS_PER_SESSION = 1600
RANDOM_TRIALS = 200

CONFIGS = (
    {"name": "base_logistic_t0.07", "temperature": 0.07, "failed_weight": 0.0, "crash_weight": 0.0, "focal_gamma": 0.0},
    {"name": "base_logistic_t0.15", "temperature": 0.15, "failed_weight": 0.0, "crash_weight": 0.0, "focal_gamma": 0.0},
    {"name": "failed_w2_logistic_t0.07", "temperature": 0.07, "failed_weight": 2.0, "crash_weight": 0.0, "focal_gamma": 0.0},
    {"name": "failed_w3_logistic_t0.07", "temperature": 0.07, "failed_weight": 3.0, "crash_weight": 0.0, "focal_gamma": 0.0},
    {"name": "failed_w2_crash_w1_logistic_t0.07", "temperature": 0.07, "failed_weight": 2.0, "crash_weight": 1.0, "focal_gamma": 0.0},
    {"name": "focal_g1_t0.07", "temperature": 0.07, "failed_weight": 0.0, "crash_weight": 0.0, "focal_gamma": 1.0},
    {"name": "failed_w2_focal_g1_t0.07", "temperature": 0.07, "failed_weight": 2.0, "crash_weight": 0.0, "focal_gamma": 1.0},
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R86 = load_module("explosive_path_event_for_failure_aware_selector", R86_PATH)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def sample_failure_weights(split: dict[str, Any], cfg: dict[str, float]) -> np.ndarray:
    label_cfg = R86.LABEL_VARIANTS[LABEL_VARIANT]
    labels = split["event_label"]
    failed_spike = (
        (split["future_max_high10"] >= label_cfg["neg_failed_max_high10"])
        & (split["future_ret10_open"] <= label_cfg["neg_failed_ret10"])
        & (split["future_giveback_high_to_exit"] >= label_cfg["neg_failed_giveback"])
    )
    crash = (split["future_ret10_open"] <= label_cfg["neg_ret10"]) | (split["future_min_open10"] <= label_cfg["neg_min_open10"])
    weights = np.ones(len(labels), dtype=np.float32)
    neg = labels == -1
    weights[neg & failed_spike] += float(cfg["failed_weight"])
    weights[neg & crash] += float(cfg["crash_weight"])
    return weights


def event_pair_indices(split: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_all: list[int] = []
    neg_all: list[int] = []
    labels = split["event_label"]
    for indices in R86.session_indices(split):
        pos = indices[labels[indices] == 1]
        neg = indices[labels[indices] == -1]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if len(pos) > MAX_POS_PER_SESSION:
            pos = rng.choice(pos, size=MAX_POS_PER_SESSION, replace=False)
        if len(neg) > MAX_NEG_PER_SESSION:
            neg = rng.choice(neg, size=MAX_NEG_PER_SESSION, replace=False)
        pairs = [(int(p), int(n)) for p in pos for n in neg]
        if len(pairs) > MAX_PAIRS_PER_SESSION:
            take = rng.choice(len(pairs), size=MAX_PAIRS_PER_SESSION, replace=False)
            pairs = [pairs[int(i)] for i in take]
        for p, n in pairs:
            pos_all.append(p)
            neg_all.append(n)
    return np.asarray(pos_all, dtype=np.int64), np.asarray(neg_all, dtype=np.int64)


class EventScorerMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=8192, shuffle=False)
    with torch.no_grad():
        for (bx,) in loader:
            outs.append(model(bx).detach().cpu().numpy())
    return np.concatenate(outs).astype(float) if outs else np.asarray([], dtype=float)


def selection_score(evaluated: dict[str, Any]) -> float:
    ret = evaluated["return_metrics"]["top_returns"]
    event = evaluated["event_metrics"]
    replay = evaluated["replay"]
    pair_auc = evaluated["event_pair"]["event_pair_auc"]
    log_mult = float(np.log(max(1e-9, replay["final_multiple"])))
    remove_best = replay["remove_best_period_multiples"].get("remove_best_3", 1.0)
    dd_penalty = abs(min(0.0, replay.get("max_drawdown", 0.0)))
    return float(
        0.20 * event["event_hit_rate"]["top10"]
        - 0.36 * event["event_fail_rate"]["top10"]
        + 1.20 * ret["top_returns" if False else "top10"]
        + 0.80 * ret["top20"]
        + 0.08 * pair_auc
        + 0.055 * log_mult
        + 0.025 * max(0.0, remove_best - 1.0)
        - 0.12 * dd_penalty
    )


def train_model(train_x: np.ndarray, valid_x: np.ndarray, train: dict[str, Any], valid: dict[str, Any], cfg: dict[str, Any], seed: int) -> tuple[EventScorerMLP, list[dict[str, float]]]:
    set_seed(seed)
    model = EventScorerMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_idx, neg_idx = event_pair_indices(train, seed)
    if len(pos_idx) == 0:
        raise RuntimeError("no event pairs built from train split")
    sample_weights = sample_failure_weights(train, cfg)
    pair_weights = sample_weights[neg_idx].astype(np.float32)
    ds = TensorDataset(torch.from_numpy(train_x[pos_idx]), torch.from_numpy(train_x[neg_idx]), torch.from_numpy(pair_weights))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e18
    history: list[dict[str, float]] = []
    temperature = float(cfg["temperature"])
    gamma = float(cfg["focal_gamma"])
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx_pos, bx_neg, bw in loader:
            opt.zero_grad(set_to_none=True)
            margin = (model(bx_pos) - model(bx_neg)) / temperature
            loss_each = nn.functional.softplus(-margin)
            if gamma > 0:
                hard = torch.sigmoid(-margin).pow(gamma)
                loss_each = loss_each * hard
            loss = (loss_each * bw).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        valid_scores = predict(model, valid_x)
        valid_eval = evaluate_split(valid, valid_scores, R74.VALID_START, R74.VALID_END, None)
        score = selection_score(valid_eval)
        history.append({
            "epoch": float(epoch),
            "loss": safe_mean(losses),
            "valid_selection_score": score,
            "valid_top10_ret5": valid_eval["return_metrics"]["top_returns"]["top10"],
            "valid_top20_ret5": valid_eval["return_metrics"]["top_returns"]["top20"],
            "valid_top10_event_hit": valid_eval["event_metrics"]["event_hit_rate"]["top10"],
            "valid_top10_event_fail": valid_eval["event_metrics"]["event_fail_rate"]["top10"],
            "valid_event_pair_auc": valid_eval["event_pair"]["event_pair_auc"],
        })
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_split(split: dict[str, Any], scores: np.ndarray, start: str, end: str, market: dict[str, Any] | None) -> dict[str, Any]:
    replay = {"final_multiple": 1.0, "max_drawdown": 0.0, "remove_best_period_multiples": {"remove_best_3": 1.0}}
    if market is not None:
        replay = R74.replay_summary(split, scores, market, start, end)
    return {
        "return_metrics": R74.metric_from_scores(split, scores),
        "event_metrics": R86.event_metrics(split, scores),
        "event_pair": R86.event_pair_auc(scores, split),
        "replay": replay,
    }


def transform_splits(train: dict[str, Any], other: dict[str, dict[str, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_train_x = R83.transform_x(train["x"], "real_path")
    mu, sd = R83.standardizer(raw_train_x)
    out = {name: R83.apply_standardizer(R83.transform_x(split["x"], "real_path"), mu, sd) for name, split in other.items()}
    return R83.apply_standardizer(raw_train_x, mu, sd), out


def build_event_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str) -> dict[str, Any]:
    raw = R74.build_split(market, feats, amount20_arr, start, end)
    return R86.filter_valid_path(R86.enrich_split(market, raw, LABEL_VARIANT))


def fit_and_eval_config(market: dict[str, Any], splits: dict[str, dict[str, Any]], cfg: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    train_x, x_by_split = transform_splits(splits["train"], splits)
    model, history = train_model(train_x, x_by_split["valid"], splits["train"], splits["valid"], cfg, SEED + seed_offset)
    scores = {name: predict(model, x_by_split[name]) for name in splits}
    evaluated = {
        "train": evaluate_split(splits["train"], scores["train"], R74.TRAIN_START, R74.TRAIN_END, market),
        "valid": evaluate_split(splits["valid"], scores["valid"], R74.VALID_START, R74.VALID_END, market),
        "dev": evaluate_split(splits["dev"], scores["dev"], R74.DEV_START, R74.DEV_END, market),
        "forward": evaluate_split(splits["forward"], scores["forward"], R74.FWD_START, R74.FWD_END, market),
    }
    return {"history_tail": history[-6:], "eval": evaluated}


def fit_inner_config(market: dict[str, Any], train: dict[str, Any], inner_valid: dict[str, Any], cfg: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    train_x, x_by_split = transform_splits(train, {"inner_valid": inner_valid})
    model, history = train_model(train_x, x_by_split["inner_valid"], train, inner_valid, cfg, SEED + 50000 + seed_offset)
    scores = predict(model, x_by_split["inner_valid"])
    evaluated = evaluate_split(inner_valid, scores, "2024-01-01", "2024-12-31", market)
    return {"history_tail": history[-6:], "eval": evaluated}


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def compact_row(item: dict[str, Any]) -> dict[str, Any]:
    valid = item["full"]["eval"]["valid"]
    inner = item["inner"]["eval"]
    dev = item["full"]["eval"]["dev"]
    forward = item["full"]["eval"]["forward"]
    return {
        "config": item["config"]["name"],
        "temperature": item["config"]["temperature"],
        "failed_weight": item["config"]["failed_weight"],
        "crash_weight": item["config"]["crash_weight"],
        "focal_gamma": item["config"]["focal_gamma"],
        "valid_selection_score": selection_score(valid),
        "inner_2024_selection_score": selection_score(inner),
        "combined_selection_score": item["combined_selection_score"],
        "valid_top10_ret5": valid["return_metrics"]["top_returns"]["top10"],
        "valid_top10_event_hit": valid["event_metrics"]["event_hit_rate"]["top10"],
        "valid_top10_event_fail": valid["event_metrics"]["event_fail_rate"]["top10"],
        "valid_multiple": valid["replay"]["final_multiple"],
        "valid_remove_best_3": valid["replay"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
        "inner_2024_top10_ret5": inner["return_metrics"]["top_returns"]["top10"],
        "inner_2024_top10_event_hit": inner["event_metrics"]["event_hit_rate"]["top10"],
        "inner_2024_top10_event_fail": inner["event_metrics"]["event_fail_rate"]["top10"],
        "inner_2024_multiple": inner["replay"]["final_multiple"],
        "dev_multiple": dev["replay"]["final_multiple"],
        "dev_all_years_positive": dev["replay"]["all_years_positive"],
        "dev_annual_returns": dev["replay"]["annual_returns"],
        "forward_multiple": forward["replay"]["final_multiple"],
        "forward_top10_ret5": forward["return_metrics"]["top_returns"]["top10"],
        "forward_top20_ret5": forward["return_metrics"]["top_returns"]["top20"],
        "forward_top10_event_hit": forward["event_metrics"]["event_hit_rate"]["top10"],
        "forward_top10_event_fail": forward["event_metrics"]["event_fail_rate"]["top10"],
        "forward_event_pair_auc": forward["event_pair"]["event_pair_auc"],
        "forward_remove_best_3": forward["replay"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
        "forward_avg_position": forward["replay"]["avg_position_count"],
        "forward_max_drawdown": forward["replay"]["max_drawdown"],
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)

    train_full = build_event_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    valid = build_event_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    dev = R86.merge_splits(train_full, valid)
    forward = build_event_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    inner_train = build_event_split(market, feats, amount20_arr, "2021-01-01", "2023-12-31")
    inner_valid = build_event_split(market, feats, amount20_arr, "2024-01-01", "2024-12-31")
    full_splits = {"train": train_full, "valid": valid, "dev": dev, "forward": forward}

    results = []
    for i, cfg in enumerate(CONFIGS):
        inner = fit_inner_config(market, inner_train, inner_valid, cfg, i * 1000)
        full = fit_and_eval_config(market, full_splits, cfg, i * 1000 + 101)
        valid_score = selection_score(full["eval"]["valid"])
        inner_score = selection_score(inner["eval"])
        combined = 0.62 * valid_score + 0.38 * inner_score
        results.append({"config": dict(cfg), "inner": inner, "full": full, "combined_selection_score": combined})

    leaderboard = sorted([compact_row(item) for item in results], key=lambda row: row["combined_selection_score"], reverse=True)
    selected = leaderboard[0]
    best_forward = max(leaderboard, key=lambda row: row["forward_multiple"])
    best_forward_low_fail = max(leaderboard, key=lambda row: row["forward_top10_ret5"] - 0.02 * row["forward_top10_event_fail"])
    forward_random = random_summary(forward, market)
    verdict = "explosive_path_failure_aware_selector_not_enough"
    if (
        selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > forward_random["p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "explosive_path_failure_aware_selector_candidate_needs_walkforward"
    out = {
        "experiment": "explosive_path_failure_aware_contrastive_selector_v1",
        "method": "failure_weighted_and_focal_same_session_contrastive_models_for_loose_holdable_explosive_path_events_with_2024_inner_validation_and_2025_validation_selection",
        "params": {
            "label_variant": LABEL_VARIANT,
            "configs": CONFIGS,
            "train": [R74.TRAIN_START, R74.TRAIN_END],
            "valid": [R74.VALID_START, R74.VALID_END],
            "inner_train": ["2021-01-01", "2023-12-31"],
            "inner_valid": ["2024-01-01", "2024-12-31"],
            "forward": [R74.FWD_START, R74.FWD_END],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "round86_script": sha256(R86_PATH), "round86_summary": sha256(ROOT / "explosive_path_contrastive_event_model_v1_summary.json")},
        "sample_counts": {name: int(len(split["event_label"])) for name, split in {**full_splits, "inner_train": inner_train, "inner_valid": inner_valid}.items()},
        "event_soil": {name: R86.event_soil(split) for name, split in {**full_splits, "inner_train": inner_train, "inner_valid": inner_valid}.items()},
        "leaderboard": leaderboard,
        "selected_by_combined_inner2024_valid2025": selected,
        "best_by_forward_multiple": best_forward,
        "best_by_forward_low_fail_proxy": best_forward_low_fail,
        "results": results,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected": selected, "best_by_forward_multiple": best_forward, "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
