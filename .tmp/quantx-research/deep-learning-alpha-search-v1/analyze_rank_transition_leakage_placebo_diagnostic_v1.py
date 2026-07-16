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
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
R70_PATH = ROOT / "analyze_cross_sectional_temporal_rank_transition_transformer_v1.py"
CACHE = ROOT / "session_tensor_cache_v1.npz"
OUT = ROOT / "rank_transition_leakage_placebo_diagnostic_v1_summary.json"

SEED = 20260714
EPOCHS = 35
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 2e-4
LAG_COL_START = 25


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R70 = load_module("rank_transition_v1_for_placebo", R70_PATH)


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


def train_variant(ds: dict[str, Any], cache: dict[str, np.ndarray], x: np.ndarray, label: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    set_seed(seed)
    model = R70.RankTransitionTransformer(ds["feature_dim"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx = ds["idx"]
    train_mask = cache["split"][idx] == "train"
    train_x = torch.from_numpy(x[train_mask].astype(np.float32))
    train_y = torch.from_numpy(label[train_mask].astype(np.float32))
    edge_weight = 0.55 + 2.5 * (np.abs(label[train_mask]) >= 0.30).astype(np.float32)
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
            pred = model(bx)
            loss = (((pred - by) ** 2) * bw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        pred_all = R70.predict_scores(model, x)
        valid_eval = R70.evaluate_scores(cache, ds | {"x": x, "label": label}, pred_all, "valid", "pure", -1.0)
        score = valid_eval["model_top"]["top20"] + 0.20 * valid_eval["model_top"]["top10"] + 0.002 * valid_eval["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_score": float(score), "valid_top20": valid_eval["model_top"]["top20"], "valid_top10": valid_eval["model_top"]["top10"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return R70.predict_scores(model, x), history


def make_variant(ds: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    x = ds["x"].copy()
    label = ds["label"].copy()
    rng = np.random.default_rng(SEED + 71)
    if name == "real_lag_payoff":
        return x, label
    if name == "no_lag_payoff":
        x[:, :, LAG_COL_START:] = 0.0
        return x, label
    if name == "permuted_lag_payoff":
        order = rng.permutation(x.shape[0])
        x[:, :, LAG_COL_START:] = x[order, :, LAG_COL_START:]
        return x, label
    if name == "slot_bucket_market_only":
        x[:, :, LAG_COL_START:] = 0.0
        # keep current position/bucket/layer/regime/market only
        return x, label
    if name == "train_label_next_session_shift":
        shifted = label.copy()
        shifted[:-1] = label[1:]
        return x, shifted
    raise ValueError(name)


def evaluate_variant(cache: dict[str, np.ndarray], ds: dict[str, Any], pred: np.ndarray) -> dict[str, Any]:
    rows = []
    for mode, weight in [("pure", -1.0), ("anchor", 0.35), ("anchor", 0.50), ("anchor", 0.75)]:
        evals = {split: R70.evaluate_scores(cache, ds, pred, split, mode, weight) for split in ["train", "valid", "forward"]}
        dev = R70.merge_eval(evals["train"], evals["valid"])
        valid = evals["valid"]
        score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"]
        rows.append({"mode": mode, "weight": weight, "valid_score": score, "train": evals["train"], "valid": valid, "dev": dev, "forward": evals["forward"]})
    return sorted(rows, key=lambda r: r["valid_score"], reverse=True)[0]


def main() -> None:
    started = time.perf_counter()
    cache = {k: v for k, v in np.load(CACHE, allow_pickle=False).items()}
    ds = R70.build_dataset(cache)
    variants = ["real_lag_payoff", "no_lag_payoff", "permuted_lag_payoff", "train_label_next_session_shift"]
    results = []
    for offset, name in enumerate(variants):
        x, label = make_variant(ds, name)
        pred, history = train_variant(ds, cache, x, label, SEED + offset)
        best = evaluate_variant(cache, ds | {"x": x, "label": label}, pred)
        results.append({"variant": name, "selected": best, "history_tail": history[-5:]})
    real = next(r for r in results if r["variant"] == "real_lag_payoff")
    no_lag = next(r for r in results if r["variant"] == "no_lag_payoff")
    perm = next(r for r in results if r["variant"] == "permuted_lag_payoff")
    verdict = "rank_transition_placebo_inconclusive"
    real_fwd = real["selected"]["forward"]["model_top"]["top20"]
    placebo_best = max(no_lag["selected"]["forward"]["model_top"]["top20"], perm["selected"]["forward"]["model_top"]["top20"])
    if real_fwd > placebo_best + 0.002 and real["selected"]["valid"]["model_top"]["top20"] > no_lag["selected"]["valid"]["model_top"]["top20"]:
        verdict = "rank_transition_lag_payoff_has_incremental_signal"
    elif placebo_best >= real_fwd - 0.001:
        verdict = "rank_transition_placebo_too_strong_possible_slot_bias"
    out = {
        "experiment": "rank_transition_leakage_placebo_diagnostic_v1",
        "method": "rank_slot_transformer_lag_ablation_permutation_and_label_shift_placebo",
        "params": {"epochs": EPOCHS, "variants": variants, "lag_col_start": LAG_COL_START},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round70_script": sha256(R70_PATH), "round70_summary": sha256(ROOT / "cross_sectional_temporal_rank_transition_transformer_v1_summary.json"), "cache": sha256(CACHE)},
        "sample_counts": {name: int(np.sum(cache["split"][ds["idx"]] == name)) for name in ["train", "valid", "forward"]},
        "results": results,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "results": results, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
