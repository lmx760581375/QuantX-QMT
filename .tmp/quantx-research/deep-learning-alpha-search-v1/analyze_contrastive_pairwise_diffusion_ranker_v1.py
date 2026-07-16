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
OUT = ROOT / "contrastive_pairwise_diffusion_ranker_v1_summary.json"

SEED = 20260714
EPOCHS = 28
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4
POS_Q = 0.80
NEG_Q = 0.20
MAX_POS_PER_SESSION = 40
MAX_NEG_PER_SESSION = 40
MAX_PAIRS_PER_SESSION = 1200
TEMPERATURES = (0.07, 0.15, 0.30)
X_VARIANTS = ("real_path", "base_liq_only")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R74 = load_module("daily_path_residual_v1_for_contrastive_pairwise", R74_PATH)


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


def standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu) / sd).astype(np.float32)


def session_indices(split: dict[str, Any]) -> list[np.ndarray]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(by_session.items())]


def pair_indices(split: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_all: list[int] = []
    neg_all: list[int] = []
    for indices in session_indices(split):
        y = split["raw_y"][indices]
        if len(y) < 30:
            continue
        hi = float(np.quantile(y, POS_Q))
        lo = float(np.quantile(y, NEG_Q))
        pos = indices[y >= hi]
        neg = indices[y <= lo]
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


def pairwise_auc(scores: np.ndarray, split: dict[str, Any]) -> dict[str, float]:
    aucs = []
    margins = []
    for indices in session_indices(split):
        y = split["raw_y"][indices]
        if len(y) < 30:
            continue
        hi = float(np.quantile(y, POS_Q))
        lo = float(np.quantile(y, NEG_Q))
        pos = indices[y >= hi]
        neg = indices[y <= lo]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = scores[pos][:, None] - scores[neg][None, :]
        aucs.append(float(np.mean(diff > 0.0) + 0.5 * np.mean(diff == 0.0)))
        margins.append(float(np.mean(diff)))
    return {"pair_auc": safe_mean(aucs), "pair_margin": safe_mean(margins), "sessions": len(aucs)}


class ScorerMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (bx,) in loader:
            outs.append(model(bx).detach().cpu().numpy())
    return np.concatenate(outs).astype(float) if outs else np.asarray([], dtype=float)


def train_model(train_x: np.ndarray, valid_x: np.ndarray, train_split: dict[str, Any], valid_split: dict[str, Any], temperature: float, seed: int) -> tuple[ScorerMLP, list[dict[str, float]]]:
    set_seed(seed)
    model = ScorerMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_idx, neg_idx = pair_indices(train_split, seed)
    ds = TensorDataset(torch.from_numpy(train_x[pos_idx]), torch.from_numpy(train_x[neg_idx]))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e18
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx_pos, bx_neg in loader:
            opt.zero_grad(set_to_none=True)
            margin = (model(bx_pos) - model(bx_neg)) / temperature
            loss = nn.functional.softplus(-margin).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        valid_scores = predict(model, valid_x)
        valid_metric = R74.metric_from_scores(valid_split, valid_scores)
        valid_pair = pairwise_auc(valid_scores, valid_split)
        score = 0.38 * valid_metric["top_returns"]["top10"] + 0.32 * valid_metric["top_returns"]["top20"] + 0.20 * valid_pair["pair_auc"] + 0.10 * valid_metric["mean_rank_ic"]
        history.append({
            "epoch": float(epoch),
            "loss": safe_mean(losses),
            "valid_score": score,
            "valid_top10": valid_metric["top_returns"]["top10"],
            "valid_top20": valid_metric["top_returns"]["top20"],
            "valid_ic": valid_metric["mean_rank_ic"],
            "valid_pair_auc": valid_pair["pair_auc"],
            "valid_pair_margin": valid_pair["pair_margin"],
        })
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, x_by_split: dict[str, np.ndarray], splits: dict[str, dict[str, Any]], market: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    scores = {}
    pair = {}
    for name, split in splits.items():
        score = predict(model, x_by_split[name])
        scores[name] = score
        metrics[name] = R74.metric_from_scores(split, score)
        pair[name] = pairwise_auc(score, split)
    replay = {
        "dev": R74.replay_summary(splits["dev"], scores["dev"], market, R74.DEV_START, R74.DEV_END),
        "forward": R74.replay_summary(splits["forward"], scores["forward"], market, R74.FWD_START, R74.FWD_END),
    }
    return {"metrics": metrics, "pair": pair, "replay": replay}


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(R74.RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def compact_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": item["config"],
        "x_variant": item["x_variant"],
        "temperature": item["temperature"],
        "valid_top10": item["metrics"]["valid"]["top_returns"]["top10"],
        "valid_top20": item["metrics"]["valid"]["top_returns"]["top20"],
        "valid_ic": item["metrics"]["valid"]["mean_rank_ic"],
        "valid_pair_auc": item["pair"]["valid"]["pair_auc"],
        "dev_top20": item["metrics"]["dev"]["top_returns"]["top20"],
        "dev_by_year_top20": item["metrics"]["dev"]["by_year_top20"],
        "forward_top10": item["metrics"]["forward"]["top_returns"]["top10"],
        "forward_top20": item["metrics"]["forward"]["top_returns"]["top20"],
        "forward_ic": item["metrics"]["forward"]["mean_rank_ic"],
        "forward_pair_auc": item["pair"]["forward"]["pair_auc"],
        "dev_multiple": item["replay"]["dev"]["final_multiple"],
        "dev_all_years_positive": item["replay"]["dev"]["all_years_positive"],
        "forward_multiple": item["replay"]["forward"]["final_multiple"],
        "forward_remove_best_3": item["replay"]["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
        "forward_avg_position": item["replay"]["forward"]["avg_position_count"],
    }


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
    results = []
    for x_variant in X_VARIANTS:
        raw_x = {name: transform_x(split["x"], x_variant) for name, split in splits.items()}
        mu, sd = standardizer(raw_x["train"])
        x_by_split = {name: apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
        for temperature in TEMPERATURES:
            model, history = train_model(x_by_split["train"], x_by_split["valid"], train, valid, temperature, SEED + int(temperature * 1000) + (0 if x_variant == "real_path" else 10000))
            evaluated = evaluate_model(model, x_by_split, splits, market)
            results.append({
                "config": f"{x_variant}_pairwise_logistic_t{temperature}",
                "x_variant": x_variant,
                "temperature": temperature,
                "history_tail": history[-8:],
                "metrics": evaluated["metrics"],
                "pair": evaluated["pair"],
                "replay": evaluated["replay"],
            })
    leaderboard = sorted([compact_row(item) for item in results], key=lambda row: 0.38 * row["valid_top10"] + 0.32 * row["valid_top20"] + 0.20 * row["valid_pair_auc"] + 0.10 * row["valid_ic"], reverse=True)
    selected = leaderboard[0]
    best_real = max((row for row in leaderboard if row["x_variant"] == "real_path"), key=lambda row: row["forward_multiple"])
    best_base = max((row for row in leaderboard if row["x_variant"] == "base_liq_only"), key=lambda row: row["forward_multiple"])
    forward_random = random_summary(forward, market)
    verdict = "contrastive_pairwise_diffusion_ranker_not_enough"
    if (
        selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > forward_random["p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "contrastive_pairwise_diffusion_ranker_candidate_needs_walkforward"
    out = {
        "experiment": "contrastive_pairwise_diffusion_ranker_v1",
        "method": "same_session_top_bottom_pairwise_logistic_contrastive_scorer_on_daily_path_features",
        "params": {"train": [R74.TRAIN_START, R74.TRAIN_END], "valid": [R74.VALID_START, R74.VALID_END], "dev": [R74.DEV_START, R74.DEV_END], "forward": [R74.FWD_START, R74.FWD_END], "pos_q": POS_Q, "neg_q": NEG_Q, "max_pos_per_session": MAX_POS_PER_SESSION, "max_neg_per_session": MAX_NEG_PER_SESSION, "max_pairs_per_session": MAX_PAIRS_PER_SESSION, "temperatures": TEMPERATURES, "x_variants": X_VARIANTS, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round74_script": sha256(R74_PATH), "round82_summary": sha256(ROOT / "cross_sectional_lead_lag_diffusion_probe_v1_summary.json")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "leaderboard": leaderboard,
        "selected_by_valid_contrastive": selected,
        "best_real_by_forward_multiple": best_real,
        "best_base_by_forward_multiple": best_base,
        "results": results,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "selected_by_valid_contrastive": selected, "best_real_by_forward_multiple": best_real, "best_base_by_forward_multiple": best_base, "forward_random": forward_random, "leaderboard": leaderboard, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
