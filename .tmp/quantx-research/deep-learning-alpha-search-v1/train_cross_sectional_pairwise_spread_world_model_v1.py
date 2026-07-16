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
OUT = ROOT / "cross_sectional_pairwise_spread_world_model_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

INTERVAL = 5
POOL_SIZE = 500
DOMAIN_SIZE = 60
TOP_KS = [10, 20]
SEED = 20260714
EPOCHS = 8
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CA52 = load_module("crowded_trend_for_pairwise_spread", ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py")
RT = CA52.RT
POS = CA52.POS


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


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    y = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def session_table(market: dict[str, Any], amount20_arr: np.ndarray, session: str) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in cols:
        y = CA52.future_return(market, idx, int(col), INTERVAL)
        if y is not None:
            labels.append(float(y))
            kept_cols.append(int(col))
    if len(labels) < DOMAIN_SIZE:
        return None
    kept = np.asarray(kept_cols, dtype=int)
    labels_arr = np.asarray(labels, dtype=float)
    frame = CA52.feature_frame(market, idx, kept)
    scores = CA52.candidate_scores(frame)
    base_score = scores["mid_trend_volume_not_extreme"]
    order = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score[j]), str(market["symbols"][kept[j]])))[:DOMAIN_SIZE], dtype=int)
    feature_names = sorted(frame.keys())
    x = np.vstack([np.asarray(frame[name], dtype=float)[order] for name in feature_names]).T
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = labels_arr[order].astype(np.float32)
    symbols = [market["symbols"][kept[j]] for j in order]
    base_order = np.arange(len(order))
    return {"date": session, "year": session[:4], "x": x, "y": y, "symbols": symbols, "base_order": base_order, "feature_names": feature_names}


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str) -> list[dict[str, Any]]:
    rows = []
    for session in CA52.CVR.sessions(market, start, end, INTERVAL):
        item = session_table(market, amount20_arr, session)
        if item is not None:
            rows.append(item)
    return rows


def pair_dataset(sessions: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for item in sessions:
        x = item["x"]
        y = item["y"]
        n = len(y)
        for i in range(n):
            for j in range(i + 1, n):
                if y[i] == y[j]:
                    continue
                sign = 1.0 if y[i] > y[j] else 0.0
                xs.append(x[i] - x[j])
                ys.append(sign)
                xs.append(x[j] - x[i])
                ys.append(1.0 - sign)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return tuple(((x - mu) / sd).astype(np.float32) for x in (train_x, *others))


class PairwiseMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray) -> dict[str, Any]:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PairwiseMLP(train_x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True)
    valid_tensor = torch.from_numpy(valid_x).to(device)
    history = []
    best_state = None
    best_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(valid_tensor)).detach().cpu().numpy()
        acc = float(np.mean((p >= 0.5) == (valid_y >= 0.5))) if len(valid_y) else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_pair_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"model": model, "device": device, "history": history, "best_epoch": max(history, key=lambda r: r["valid_pair_acc"])["epoch"], "best_valid_pair_acc": best_acc}


def score_session(model_info: dict[str, Any], item: dict[str, Any], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = item["x"]
    n = len(x)
    wins = np.zeros(n, dtype=float)
    pairs = []
    ij = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(x[i] - x[j])
            ij.append((i, j))
    if not pairs:
        return wins
    px = ((np.asarray(pairs, dtype=np.float32) - mu) / sd).astype(np.float32)
    model = model_info["model"]
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(px).to(model_info["device"]))).detach().cpu().numpy()
    for (i, j), p in zip(ij, prob):
        wins[i] += float(p)
        wins[j] += 1.0 - float(p)
    return wins


def evaluate(sessions: list[dict[str, Any]], model_info: dict[str, Any], mu: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    model_top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        score = score_session(model_info, item, mu, sd)
        y = item["y"]
        order = np.argsort(-score, kind="mergesort")
        base_order = item["base_order"]
        ics.append(rank_ic(score, y))
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            model_top[k].append(value)
            base_top[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(sessions),
        "model_top": {f"top{k}": safe_mean(v) for k, v in model_top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "mean_rank_ic": safe_mean(ics),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    splits = {
        "train": build_split(market, amount20_arr, TRAIN_START, TRAIN_END),
        "valid": build_split(market, amount20_arr, VALID_START, VALID_END),
        "dev": build_split(market, amount20_arr, DEV_START, DEV_END),
        "forward": build_split(market, amount20_arr, FWD_START, FWD_END),
    }
    train_x, train_y = pair_dataset(splits["train"])
    valid_x, valid_y = pair_dataset(splits["valid"])
    train_xs, valid_xs = standardize(train_x, valid_x)
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    model_info = train_model(train_xs, train_y, valid_xs, valid_y)
    results = {name: evaluate(items, model_info, mu, sd) for name, items in splits.items()}
    verdict = "pairwise_spread_world_model_not_enough"
    fwd = results["forward"]
    dev = results["dev"]
    dev_years = dev["by_year_model_top20"]
    if fwd["model_top"]["top20"] > fwd["base_top"]["top20"] + 0.003 and min(dev_years.values()) > 0.0 and dev["model_top"]["top20"] > dev["base_top"]["top20"]:
        verdict = "pairwise_spread_world_model_candidate_needs_replay"
    out = {
        "experiment": "cross_sectional_pairwise_spread_world_model_v1",
        "method": "pairwise_candidate_internal_relative_win_model_on_mid_trend_volume_domain",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "interval": INTERVAL, "pool_size": POOL_SIZE, "domain_size": DOMAIN_SIZE, "top_ks": TOP_KS, "epochs": EPOCHS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round58_summary": sha256(ROOT / "qmt_intraday_path_opportunity_probe_v1_summary.json")},
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "pair_counts": {"train": len(train_y), "valid": len(valid_y)},
        "feature_names": splits["train"][0]["feature_names"] if splits["train"] else [],
        "training_history": model_info["history"],
        "best_epoch": model_info["best_epoch"],
        "best_valid_pair_acc": model_info["best_valid_pair_acc"],
        "results": results,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_epoch": out["best_epoch"], "best_valid_pair_acc": out["best_valid_pair_acc"], "sample_counts": out["sample_counts"], "pair_counts": out["pair_counts"], "results": results, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
