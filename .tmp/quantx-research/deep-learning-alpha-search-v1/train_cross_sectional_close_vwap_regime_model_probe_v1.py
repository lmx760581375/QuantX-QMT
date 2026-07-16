from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_close_vwap_regime_model_probe_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

INTERVAL = 3
SEQ_LEN = 60
POOL_SIZE = 500
CANDIDATE_POOL = 120
TOP_K = 10
EPOCHS = 8
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_TRIALS = 300
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_regime_model", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def market_context(view: np.ndarray) -> np.ndarray:
    close_vwap = view[:, 4]
    day_range = view[:, 3]
    ret5 = view[:, 6]
    ret20 = view[:, 7]
    rank_ret20 = view[:, 9]
    rank_amount = view[:, 10]
    return np.asarray(
        [
            np.nanmean(close_vwap),
            np.nanstd(close_vwap),
            np.nanquantile(close_vwap, 0.2),
            np.nanquantile(close_vwap, 0.8),
            np.nanmean(day_range),
            np.nanstd(day_range),
            np.nanmean(ret5),
            np.nanmean(ret20),
            np.nanmean(rank_ret20 > 0.0),
            np.nanmean(rank_amount > 0.0),
            np.nanmean((close_vwap < 0.0) & (day_range < np.nanquantile(day_range, 0.5))),
        ],
        dtype=np.float32,
    )


def future_return(market: dict[str, Any], idx: int, col: int, interval: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + interval + 1
    arrays = market["arrays"]
    entry = float(arrays["open"][entry_idx, col])
    exit_ = float(arrays["open"][exit_idx, col])
    close_t = float(arrays["close"][idx, col])
    if not np.isfinite(entry) or not np.isfinite(exit_) or not np.isfinite(close_t) or entry <= 0 or exit_ <= 0 or close_t <= 0:
        return None
    if abs(entry / close_t - 1.0) > 0.095:
        return None
    return exit_ / entry - 1.0


def build_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    sessions: list[str] = []
    symbols: list[str] = []
    pools: dict[str, list[str]] = {}
    for session in CVR.sessions(market, start, end, INTERVAL):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)
        if len(cols) < TOP_K:
            continue
        last = feats[idx, cols, :]
        mean5 = feats[idx - 4:idx + 1, cols, :].mean(axis=0)
        mean20 = feats[idx - 19:idx + 1, cols, :].mean(axis=0)
        daily_scores = CVR.score_rows(mean20)["close_vwap_neg"]
        ordered = sorted(range(len(cols)), key=lambda j: (-float(daily_scores[j]), str(market["symbols"][cols[j]])))[:CANDIDATE_POOL]
        ctx = market_context(mean20)
        session_records = []
        for rank_pos, j in enumerate(ordered):
            col = cols[j]
            y = future_return(market, idx, col, INTERVAL)
            if y is None:
                continue
            rank_feature = np.asarray([rank_pos / max(1, len(ordered) - 1), daily_scores[j]], dtype=np.float32)
            x = np.concatenate([last[j], mean5[j], mean20[j], ctx, rank_feature]).astype(np.float32)
            session_records.append((market["symbols"][col], x, float(y)))
        if len(session_records) < TOP_K:
            continue
        pools[session] = [record[0] for record in session_records]
        for sym, x, y in session_records:
            xs.append(x)
            ys.append(y)
            sessions.append(session)
            symbols.append(sym)
    split = {"x": np.asarray(xs, dtype=np.float32), "y": np.asarray(ys, dtype=np.float32), "sessions": sessions, "symbols": symbols, "pools": pools}
    attach_targets(split)
    return split


def attach_targets(split: dict[str, Any]) -> None:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    target = np.zeros(len(split["sessions"]), dtype=np.float32)
    weight = np.ones(len(split["sessions"]), dtype=np.float32)
    y = split["y"]
    for indices in by_session.values():
        ranks = pd.Series(y[indices]).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        target[indices] = ranks
        for i in indices:
            weight[i] = 1.5 if abs(float(target[i])) >= 0.30 else 0.7
        weight[indices] *= len(indices) / max(1e-9, float(np.sum(weight[indices])))
    split["target"] = target
    split["weight"] = weight


class SmallMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            outs.append(model(batch_x.to(device)).detach().cpu().numpy())
    return np.concatenate(outs) if outs else np.asarray([], dtype=float)


def selections_from_scores(split: dict[str, Any], scores: np.ndarray) -> dict[str, list[str]]:
    by_session: dict[str, list[tuple[str, float]]] = {}
    for session, sym, score in zip(split["sessions"], split["symbols"], scores):
        by_session.setdefault(session, []).append((sym, float(score)))
    return {session: [sym for sym, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:TOP_K]] for session, items in by_session.items()}


def rank_metrics(split: dict[str, Any], scores: np.ndarray) -> dict[str, float]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    ics = []
    top_labels = []
    for indices in by_session.values():
        if len(indices) < TOP_K:
            continue
        pred_rank = pd.Series(scores[indices]).rank().to_numpy(dtype=float)
        y_rank = pd.Series(split["y"][indices]).rank().to_numpy(dtype=float)
        if np.std(pred_rank) > 0 and np.std(y_rank) > 0:
            ics.append(float(np.corrcoef(pred_rank, y_rank)[0, 1]))
        picked = sorted(indices, key=lambda i: -float(scores[i]))[:TOP_K]
        top_labels.append(float(np.mean(split["y"][picked])))
    return {"mean_rank_ic": float(np.mean(ics)) if ics else 0.0, "median_rank_ic": float(np.median(ics)) if ics else 0.0, "mean_top10_label": float(np.mean(top_labels)) if top_labels else 0.0, "sessions": len(by_session)}


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def train_model(train: dict[str, Any], valid: dict[str, Any], market: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model = SmallMLP(train["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.10)
    dataset = TensorDataset(torch.from_numpy(train["x"]), torch.from_numpy(train["target"]), torch.from_numpy(train["weight"]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    history = []
    best_state = None
    best_valid = -1e18
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = (loss_fn(pred, batch_y) * batch_w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        valid_scores = predict(model, valid["x"], device)
        valid_sel = selections_from_scores(valid, valid_scores)
        valid_summary = CVR.replay_interval(valid_sel, market, VALID_START, VALID_END, INTERVAL)["summary"]
        valid_rank = rank_metrics(valid, valid_scores)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_multiple": valid_summary["final_multiple"], "valid_rank_metrics": valid_rank}
        history.append(row)
        score = math.log(max(valid_summary["final_multiple"], 1e-9)) + 0.5 * valid_rank["mean_rank_ic"]
        if score > best_valid:
            best_valid = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    best_row = max(history, key=lambda row: math.log(max(row["valid_multiple"], 1e-9)) + 0.5 * row["valid_rank_metrics"]["mean_rank_ic"])
    return {"model": model, "history": history, "best_epoch": best_row["epoch"]}


def evaluate_split(model: nn.Module, split: dict[str, Any], market: dict[str, Any], start: str, end: str, device: torch.device) -> dict[str, Any]:
    scores = predict(model, split["x"], device)
    selections = selections_from_scores(split, scores)
    return {"summary": CVR.replay_interval(selections, market, start, end, INTERVAL)["summary"], "rank_metrics": rank_metrics(split, scores)}


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    train = build_split(market, feats, amount20_arr, TRAIN_START, TRAIN_END)
    valid = build_split(market, feats, amount20_arr, VALID_START, VALID_END)
    dev = build_split(market, feats, amount20_arr, DEV_START, DEV_END)
    forward = build_split(market, feats, amount20_arr, FWD_START, FWD_END)
    trained = train_model(train, valid, market, device)
    model = trained["model"]
    split_data = {"train": (train, TRAIN_START, TRAIN_END), "valid": (valid, VALID_START, VALID_END), "dev": (dev, DEV_START, DEV_END), "forward": (forward, FWD_START, FWD_END)}
    results = {name: evaluate_split(model, split, market, start, end, device) for name, (split, start, end) in split_data.items()}
    baseline = CVR.run_split(market, feats, amount20_arr, FWD_START, FWD_END, INTERVAL)["results"]["mean20"]["close_vwap_neg"]
    rng = np.random.default_rng(SEED)
    random_values = [CVR.replay_interval(random_selections(forward["pools"], rng), market, FWD_START, FWD_END, INTERVAL)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = percentile_summary(random_values)
    fwd = results["forward"]["summary"]
    dev_summary = results["dev"]["summary"]
    verdict = "close_vwap_regime_model_not_enough"
    if dev_summary["all_years_positive"] and fwd["final_multiple"] > random_summary["p95"] and fwd["final_multiple"] > baseline["final_multiple"] and fwd["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0:
        verdict = "close_vwap_regime_model_candidate_needs_expanded_validation"
    out = {
        "experiment": "cross_sectional_close_vwap_regime_model_probe_v1",
        "method": "small_mlp_on_close_vwap_candidate_pool_with_stock_and_cross_sectional_context_features_interval3",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "interval": INTERVAL, "pool_size": POOL_SIZE, "candidate_pool": CANDIDATE_POOL, "top_k": TOP_K, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "random_trials": RANDOM_TRIALS, "seed": SEED, "device": str(device)},
        "inputs_sha256": {"script": sha256(Path(__file__)), "close_vwap_replay_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py"), "qmt_microstructure_summary": sha256(ROOT / "qmt_intraday_close_vwap_reversion_microstructure_probe_v1_summary.json") if (ROOT / "qmt_intraday_close_vwap_reversion_microstructure_probe_v1_summary.json").exists() else None},
        "sample_counts": {name: len(split["sessions"]) for name, (split, _, _) in split_data.items()},
        "session_counts": {name: len(set(split["sessions"])) for name, (split, _, _) in split_data.items()},
        "training_history": trained["history"],
        "best_epoch": trained["best_epoch"],
        "results": results,
        "forward_daily_close_vwap_baseline": baseline,
        "forward_random_candidate_pool_top10": random_summary,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_epoch": trained["best_epoch"], "results": results, "forward_baseline": baseline, "forward_random": random_summary, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
