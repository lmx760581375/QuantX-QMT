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
R70_PATH = ROOT / "analyze_cross_sectional_temporal_rank_transition_transformer_v1.py"
CACHE = ROOT / "session_tensor_cache_v1.npz"
OUT = ROOT / "rank_transition_slot_null_residual_alpha_v1_summary.json"

SEED = 20260714
EPOCHS = 50
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 2e-4
RESID_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]
NULL_SHRINK = 25.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R70 = load_module("rank_transition_v1_for_slot_null", R70_PATH)


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


def token_key(cache: dict[str, np.ndarray], session_idx: int, pos: int, level: str) -> tuple[str, ...]:
    layer = str(cache["layers"][session_idx, pos])
    bucket = str(cache["buckets"][session_idx, pos])
    regime = str(cache["regimes"][session_idx])
    slot10 = f"slot{pos // 10}"
    if level == "slot_layer_bucket_regime":
        return (slot10, layer, bucket, regime)
    if level == "slot_layer_bucket":
        return (slot10, layer, bucket)
    if level == "slot_layer":
        return (slot10, layer)
    return (slot10,)


def build_slot_null(cache: dict[str, np.ndarray], ds: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    idx = ds["idx"]
    labels = ds["label"]
    train_rows = np.flatnonzero(cache["split"][idx] == "train")
    global_mean = float(np.mean(labels[train_rows]))
    stats: dict[str, dict[tuple[str, ...], list[float]]] = {level: {} for level in ["slot_layer_bucket_regime", "slot_layer_bucket", "slot_layer", "slot"]}
    for row in train_rows:
        session_idx = int(idx[row])
        for pos in range(labels.shape[1]):
            value = float(labels[row, pos])
            for level in stats:
                stats[level].setdefault(token_key(cache, session_idx, pos, level), []).append(value)
    means: dict[str, dict[tuple[str, ...], tuple[float, int]]] = {}
    for level, d in stats.items():
        means[level] = {k: (safe_mean(v), len(v)) for k, v in d.items()}
    null = np.zeros_like(labels, dtype=np.float32)
    for row, session_idx in enumerate(idx):
        for pos in range(labels.shape[1]):
            pred = global_mean
            for level in ["slot_layer_bucket_regime", "slot_layer_bucket", "slot_layer", "slot"]:
                item = means[level].get(token_key(cache, int(session_idx), pos, level))
                if item is None:
                    continue
                mean, count = item
                if count >= 5:
                    weight = count / (count + NULL_SHRINK)
                    pred = weight * mean + (1.0 - weight) * global_mean
                    break
            null[row, pos] = pred
    meta = {"global_mean": global_mean, "levels": {level: len(d) for level, d in means.items()}, "shrink": NULL_SHRINK}
    return null, meta


class ResidualMLP(nn.Module):
    def __init__(self, feature_dim: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(0.10),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_residual(ds: dict[str, Any], cache: dict[str, np.ndarray], null: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    set_seed(SEED)
    x = ds["x"].astype(np.float32)
    target = (ds["label"] - null).astype(np.float32)
    idx = ds["idx"]
    train_mask = cache["split"][idx] == "train"
    model = ResidualMLP(ds["feature_dim"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_x = torch.from_numpy(x[train_mask].reshape(-1, x.shape[-1]))
    train_y = torch.from_numpy(target[train_mask].reshape(-1))
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE * 60, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = ((pred - by) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        resid = predict_residual(model, x)
        valid = evaluate_combined(cache, ds, null, resid, "valid", 1.0, "null_plus_residual")
        score = valid["model_top"]["top20"] + 0.2 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_score": score, "valid_top20": valid["model_top"]["top20"], "valid_top10": valid["model_top"]["top10"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict_residual(model, x), history


def predict_residual(model: ResidualMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(flat), 4096):
            out.append(model(torch.from_numpy(flat[start:start + 4096])).cpu().numpy())
    return np.concatenate(out).reshape(x.shape[0], x.shape[1]).astype(np.float32)


def score_for(null: np.ndarray, resid: np.ndarray, row: int, weight: float, mode: str) -> np.ndarray:
    if mode == "null_only":
        return R70.rank_labels(null[row])
    if mode == "residual_only":
        return R70.rank_labels(resid[row])
    return R70.rank_labels(null[row] + weight * resid[row])


def evaluate_combined(cache: dict[str, np.ndarray], ds: dict[str, Any], null: np.ndarray, resid: np.ndarray, split_name: str, weight: float, mode: str) -> dict[str, Any]:
    idx = ds["idx"]
    mask = cache["split"][idx] == split_name
    top = {k: [] for k in R70.TOP_KS}
    base = {k: [] for k in R70.TOP_KS}
    ics = []
    overlap = []
    by_year: dict[str, list[float]] = {}
    for row, session_idx in zip(np.flatnonzero(mask), idx[mask], strict=True):
        y = cache["y"][session_idx]
        score = score_for(null, resid, row, weight, mode)
        order = np.argsort(-score, kind="mergesort")
        base_order = np.arange(len(y))
        ics.append(R70.rank_ic(score, y))
        overlap.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in R70.TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[:k]))
            if k == 20:
                by_year.setdefault(str(cache["years"][session_idx]), []).append(value)
    return {"sessions": int(np.sum(mask)), "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()}, "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()}, "mean_rank_ic": safe_mean(ics), "top20_overlap_with_base": safe_mean(overlap), "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())}}


def main() -> None:
    started = time.perf_counter()
    cache = {k: v for k, v in np.load(CACHE, allow_pickle=False).items()}
    ds = R70.build_dataset(cache)
    null, null_meta = build_slot_null(cache, ds)
    resid, history = train_residual(ds, cache, null)
    rows = []
    for mode, weight in [("null_only", 0.0), ("residual_only", 1.0)] + [("null_plus_residual", w) for w in RESID_WEIGHTS]:
        evals = {split: evaluate_combined(cache, ds, null, resid, split, weight, mode) for split in ["train", "valid", "forward"]}
        dev = R70.merge_eval(evals["train"], evals["valid"])
        valid = evals["valid"]
        score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"]
        rows.append({"mode": mode, "weight": weight, "valid_score": score, "train": evals["train"], "valid": valid, "dev": dev, "forward": evals["forward"]})
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    null_row = next(r for r in rows if r["mode"] == "null_only")
    verdict = "rank_transition_slot_null_residual_alpha_not_enough"
    if selected["mode"] == "null_plus_residual" and selected["forward"]["model_top"]["top20"] > null_row["forward"]["model_top"]["top20"] + 0.003 and selected["valid"]["model_top"]["top20"] > null_row["valid"]["model_top"]["top20"]:
        verdict = "rank_transition_slot_null_residual_alpha_candidate_needs_replay"
    out = {
        "experiment": "rank_transition_slot_null_residual_alpha_v1",
        "method": "train_only_slot_bucket_market_null_plus_residual_mlp",
        "params": {"epochs": EPOCHS, "resid_weights": RESID_WEIGHTS, "null_shrink": NULL_SHRINK},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round71_summary": sha256(ROOT / "rank_transition_leakage_placebo_diagnostic_v1_summary.json"), "cache": sha256(CACHE)},
        "sample_counts": {name: int(np.sum(cache["split"][ds["idx"]] == name)) for name in ["train", "valid", "forward"]},
        "null_meta": null_meta,
        "history_tail": history[-10:],
        "leaderboard": rows,
        "selected": selected,
        "null_only": null_row,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "selected": selected, "null_only": null_row, "leaderboard_top": rows[:8], "history_tail": history[-5:], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
