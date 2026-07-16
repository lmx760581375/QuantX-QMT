from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
CACHE = ROOT / "session_tensor_cache_v1.npz"
OUT = ROOT / "cross_sectional_temporal_rank_transition_transformer_v1_summary.json"

SEED = 20260714
TOP_KS = [10, 20, 30]
LAGS = [2, 3, 4, 6, 8, 12]
EPOCHS = 80
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 2e-4
ANCHOR_WEIGHTS = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


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


def rank_labels(y: np.ndarray) -> np.ndarray:
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty_like(y, dtype=np.float32)
    ranks[order] = (np.arange(len(y), dtype=np.float32) + 1.0) / len(y) - 0.5
    return ranks


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True, dtype=np.float32) if n > 1 else np.asarray([0.5], dtype=np.float32)


def one_hot(values: np.ndarray, vocab: list[str]) -> np.ndarray:
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    index = {v: i for i, v in enumerate(vocab)}
    for i, v in enumerate(values):
        j = index.get(str(v))
        if j is not None:
            out[i, j] = 1.0
    return out


def build_dataset(cache: dict[str, np.ndarray]) -> dict[str, Any]:
    y = cache["y"].astype(np.float32)
    buckets = cache["buckets"]
    regimes = cache["regimes"]
    layers = cache["layers"]
    market_x = cache["market_x"].astype(np.float32)
    bucket_vocab = sorted({str(v) for v in buckets.reshape(-1)})
    regime_vocab = sorted({str(v) for v in regimes.reshape(-1)})
    layer_vocab = sorted({str(v) for v in layers.reshape(-1)})
    rows = []
    labels = []
    indices = []
    max_lag = max(LAGS)
    pos = np.linspace(1.0, 0.0, y.shape[1], endpoint=True, dtype=np.float32)[:, None]
    for i in range(max_lag, len(y)):
        token_parts = [pos]
        token_parts.append(one_hot(buckets[i], bucket_vocab))
        token_parts.append(one_hot(layers[i], layer_vocab))
        regime_oh = one_hot(np.asarray([regimes[i]] * y.shape[1]), regime_vocab)
        token_parts.append(regime_oh)
        token_parts.append(np.repeat(market_x[i][None, :], y.shape[1], axis=0))

        # Lag 1 is intentionally excluded: because sessions rebalance every five trading days,
        # the immediately previous session's full forward return is not fully observable yet.
        lag_values = []
        for lag in LAGS:
            lag_values.append(y[i - lag][:, None])
            lag_values.append(rank_labels(y[i - lag])[:, None])
        lag_stack = np.concatenate(lag_values, axis=1)
        roll4 = np.mean(np.stack([y[i - lag] for lag in [2, 3, 4]], axis=1), axis=1, keepdims=True)
        roll8 = np.mean(np.stack([y[i - lag] for lag in [2, 3, 4, 6, 8]], axis=1), axis=1, keepdims=True)
        token_parts.extend([np.clip(lag_stack, -0.20, 0.20) / 0.20, np.clip(roll4, -0.20, 0.20) / 0.20, np.clip(roll8, -0.20, 0.20) / 0.20])
        x = np.concatenate(token_parts, axis=1).astype(np.float32)
        rows.append(x)
        labels.append(rank_labels(y[i]))
        indices.append(i)
    x = np.stack(rows).astype(np.float32)
    label = np.stack(labels).astype(np.float32)
    idx = np.asarray(indices, dtype=int)
    train_mask = cache["split"][idx] == "train"
    mu = x[train_mask].reshape(-1, x.shape[-1]).mean(axis=0)
    sd = x[train_mask].reshape(-1, x.shape[-1]).std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    x = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return {"x": x, "label": label, "idx": idx, "feature_dim": x.shape[-1], "bucket_vocab": bucket_vocab, "regime_vocab": regime_vocab, "layer_vocab": layer_vocab, "mu": mu, "sd": sd}


class RankTransitionTransformer(nn.Module):
    def __init__(self, feature_dim: int, width: int = 48, layers: int = 2, heads: int = 4, dropout: float = 0.12) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.pos = nn.Parameter(torch.zeros(1, 60, width))
        block = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 3, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)
        return self.out(h).squeeze(-1)


def train_model(ds: dict[str, Any], cache: dict[str, np.ndarray]) -> tuple[RankTransitionTransformer, np.ndarray, list[dict[str, float]]]:
    set_seed(SEED)
    device = torch.device("cpu")
    x = ds["x"]
    labels = ds["label"]
    idx = ds["idx"]
    train_mask = cache["split"][idx] == "train"
    valid_mask = cache["split"][idx] == "valid"
    model = RankTransitionTransformer(ds["feature_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_x = torch.from_numpy(x[train_mask])
    train_y = torch.from_numpy(labels[train_mask])
    edge_weight = 0.55 + 2.5 * (np.abs(labels[train_mask]) >= 0.30).astype(np.float32)
    train_w = torch.from_numpy(edge_weight)
    loader = DataLoader(TensorDataset(train_x, train_y, train_w), batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by, bw in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx.to(device))
            loss = (((pred - by.to(device)) ** 2) * bw.to(device)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pred_all = predict_scores(model, x)
        valid_eval = evaluate_scores(cache, ds, pred_all, "valid", "pure", -1.0)
        score = valid_eval["model_top"]["top20"] + 0.20 * valid_eval["model_top"]["top10"] + 0.002 * valid_eval["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_score": float(score), "valid_top20": valid_eval["model_top"]["top20"], "valid_top10": valid_eval["model_top"]["top10"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict_scores(model, x)
    return model, pred, history


def predict_scores(model: RankTransitionTransformer, x: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            out.append(model(torch.from_numpy(x[start:start + 64])).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def rank_ic(score: np.ndarray, y: np.ndarray) -> float:
    a = rank_labels(score)
    b = rank_labels(y)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score_from_pred(pred: np.ndarray, mode: str, weight: float) -> np.ndarray:
    model_rank = rank_labels(pred)
    if mode == "pure":
        return model_rank
    return anchor_score(len(pred)) + weight * model_rank


def evaluate_scores(cache: dict[str, np.ndarray], ds: dict[str, Any], pred: np.ndarray, split_name: str, mode: str, weight: float) -> dict[str, Any]:
    idx = ds["idx"]
    mask = cache["split"][idx] == split_name
    top = {k: [] for k in TOP_KS}
    base = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    overlap = []
    for row, session_idx in zip(np.flatnonzero(mask), idx[mask], strict=True):
        y = cache["y"][session_idx]
        score = score_from_pred(pred[row], mode, weight)
        order = np.argsort(-score, kind="mergesort")
        base_order = np.arange(len(y))
        ics.append(rank_ic(score, y))
        overlap.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[:k]))
            if k == 20:
                by_year.setdefault(str(cache["years"][session_idx]), []).append(value)
    return {"sessions": int(np.sum(mask)), "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()}, "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()}, "mean_rank_ic": safe_mean(ics), "top20_overlap_with_base": safe_mean(overlap), "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())}}


def main() -> None:
    started = time.perf_counter()
    cache = {k: v for k, v in np.load(CACHE, allow_pickle=False).items()}
    ds = build_dataset(cache)
    model, pred, history = train_model(ds, cache)
    rows = []
    for mode, weight in [("pure", -1.0)] + [("anchor", w) for w in ANCHOR_WEIGHTS]:
        evals = {split: evaluate_scores(cache, ds, pred, split, mode, weight) for split in ["train", "valid", "forward"]}
        dev = merge_eval(evals["train"], evals["valid"])
        valid = evals["valid"]
        score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"] - 0.001 * max(0.0, 0.85 - valid["top20_overlap_with_base"])
        rows.append({"mode": mode, "weight": weight, "valid_score": score, "train": evals["train"], "valid": valid, "dev": dev, "forward": evals["forward"]})
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    verdict = "cross_sectional_temporal_rank_transition_transformer_not_enough"
    dev_years = selected["dev"]["by_year_model_top20"]
    if selected["forward"]["model_top"]["top20"] > selected["forward"]["base_top"]["top20"] + 0.004 and selected["forward"]["model_top"]["top10"] >= selected["forward"]["base_top"]["top10"] * 0.95 and min(dev_years.values()) > 0.0:
        verdict = "cross_sectional_temporal_rank_transition_transformer_candidate_needs_replay"
    out = {"experiment": "cross_sectional_temporal_rank_transition_transformer_v1", "method": "rank_slot_temporal_transformer_lag2plus_no_recent_label_leakage", "params": {"lags": LAGS, "epochs": EPOCHS, "top_ks": TOP_KS, "anchor_weights": ANCHOR_WEIGHTS}, "inputs_sha256": {"script": sha256(Path(__file__)), "cache": sha256(CACHE), "round69_summary": sha256(ROOT / "session_phase_hot_continuation_meta_model_v1_summary.json")}, "sample_counts": {name: int(np.sum(cache["split"][ds["idx"]] == name)) for name in ["train", "valid", "forward"]}, "dataset": {"feature_dim": ds["feature_dim"], "bucket_vocab": ds["bucket_vocab"], "regime_vocab": ds["regime_vocab"], "layer_vocab": ds["layer_vocab"]}, "history_tail": history[-10:], "leaderboard": rows, "selected": selected, "verdict": verdict, "elapsed_seconds": time.perf_counter() - started}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "selected": selected, "leaderboard_top": rows[:8], "history_tail": history[-5:], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


def merge_eval(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    sessions = train["sessions"] + valid["sessions"]
    def wavg(key: str, sub: str) -> float:
        return (train[key][sub] * train["sessions"] + valid[key][sub] * valid["sessions"]) / sessions if sessions else 0.0
    return {"sessions": sessions, "model_top": {k: wavg("model_top", k) for k in train["model_top"]}, "base_top": {k: wavg("base_top", k) for k in train["base_top"]}, "mean_rank_ic": (train["mean_rank_ic"] * train["sessions"] + valid["mean_rank_ic"] * valid["sessions"]) / sessions if sessions else 0.0, "top20_overlap_with_base": (train["top20_overlap_with_base"] * train["sessions"] + valid["top20_overlap_with_base"] * valid["sessions"]) / sessions if sessions else 0.0, "by_year_model_top20": {**train["by_year_model_top20"], **valid["by_year_model_top20"]}}


if __name__ == "__main__":
    main()
