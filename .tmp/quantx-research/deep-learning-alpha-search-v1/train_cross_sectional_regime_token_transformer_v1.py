from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_regime_token_transformer_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

SEQ_LEN = 60
MODEL_SEQ_LEN = SEQ_LEN + 1
POOL_SIZE = 500
TOP_K = 10
REBALANCE_INTERVAL = 5
RANDOM_TRIALS = 200
SEED = 20260714
EPOCHS = 8
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
TOP_QUANTILE = 0.80
BOTTOM_QUANTILE = 0.20
TOP_WEIGHT = 3.0
BOTTOM_WEIGHT = 2.0
MID_WEIGHT = 0.35
REGIME_FEATURE_NAMES = [
    "mkt_ret1_mean",
    "mkt_ret5_mean",
    "mkt_ret20_mean",
    "mkt_rank_ret20_spread",
    "mkt_day_range_mean",
    "mkt_close_vwap_mean",
    "mkt_log_vol_mean",
    "mkt_up1_ratio",
    "mkt_up5_ratio",
    "mkt_high_mom_high_range",
    "mkt_mom_vol_corr",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POS = load_module("multihorizon_v1_for_token_probe", ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rank_matrix(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[0]):
        ranked = pd.Series(values[i]).replace([np.inf, -np.inf], np.nan).rank(pct=True, method="average").fillna(0.5)
        out[i] = ranked.to_numpy(dtype=np.float32) - 0.5
    return out


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float32)
    for i in range(window - 1, values.shape[0]):
        out[i] = np.nanmean(values[i - window + 1:i + 1], axis=0)
    return out


def clip_scale(values: np.ndarray, limit: float) -> np.ndarray:
    return (np.clip(values, -limit, limit) / limit).astype(np.float32)


def build_feature_tensor(market: dict[str, Any]) -> np.ndarray:
    arrays = market["arrays"]
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]
    vwap = arrays["vwap"]
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    ret1 = POS.safe_div(close, prev_close) - 1.0
    open_gap = POS.safe_div(open_, prev_close) - 1.0
    intraday = POS.safe_div(close, open_) - 1.0
    day_range = POS.safe_div(high, low) - 1.0
    close_vwap = POS.safe_div(close, vwap) - 1.0
    vol_ma20 = rolling_nanmean(volume, 20)
    vol_ratio = POS.safe_div(volume, vol_ma20)
    ret5 = POS.safe_div(close, np.vstack([np.full((5, close.shape[1]), np.nan), close[:-5]])) - 1.0
    ret20 = POS.safe_div(close, np.vstack([np.full((20, close.shape[1]), np.nan), close[:-20]])) - 1.0
    log_vol = np.log(np.where(np.isfinite(vol_ratio) & (vol_ratio > 0), vol_ratio, np.nan))
    rank_ret1 = rank_matrix(ret1)
    rank_ret20 = rank_matrix(ret20)
    amount = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close) * volume
    rank_amount = rank_matrix(amount)
    features = np.stack(
        [
            clip_scale(ret1, 0.12),
            clip_scale(open_gap, 0.10),
            clip_scale(intraday, 0.10),
            clip_scale(day_range, 0.12),
            clip_scale(close_vwap, 0.08),
            clip_scale(log_vol, 3.0),
            clip_scale(ret5, 0.25),
            clip_scale(ret20, 0.60),
            rank_ret1,
            rank_ret20,
            rank_amount,
        ],
        axis=2,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def regime_tokens_from_pool(feats: np.ndarray, cols: list[int], idx: int) -> np.ndarray:
    last = feats[idx, cols, :]
    mean5 = feats[idx - 4:idx + 1, cols, :].mean(axis=0)
    ret1 = last[:, 0]
    day_range = last[:, 3]
    close_vwap = last[:, 4]
    log_vol = last[:, 5]
    ret5 = mean5[:, 6]
    ret20 = mean5[:, 7]
    rank_ret20 = last[:, 9]
    rank_amount = last[:, 10]
    hi = rank_ret20 >= np.nanquantile(rank_ret20, 0.8)
    lo = rank_ret20 <= np.nanquantile(rank_ret20, 0.2)
    corr = 0.0
    if np.std(rank_ret20) > 1e-9 and np.std(rank_amount) > 1e-9:
        corr = float(np.corrcoef(rank_ret20, rank_amount)[0, 1])
    values = np.asarray(
        [
            float(np.nanmean(ret1)),
            float(np.nanmean(ret5)),
            float(np.nanmean(ret20)),
            float(np.nanmean(rank_ret20[hi]) - np.nanmean(rank_ret20[lo])) if np.any(hi) and np.any(lo) else 0.0,
            float(np.nanmean(day_range)),
            float(np.nanmean(close_vwap)),
            float(np.nanmean(log_vol)),
            float(np.nanmean(ret1 > 0.0)),
            float(np.nanmean(ret5 > 0.0)),
            float(np.nanmean(day_range[hi])) if np.any(hi) else 0.0,
            corr,
        ],
        dtype=np.float32,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    token = np.zeros(feats.shape[2], dtype=np.float32)
    token[: len(values)] = np.clip(values, -1.0, 1.0)
    return token


def amount20(market: dict[str, Any]) -> np.ndarray:
    arrays = market["arrays"]
    price = np.where(np.isfinite(arrays["vwap"]) & (arrays["vwap"] > 0), arrays["vwap"], arrays["close"])
    return rolling_nanmean(price * arrays["volume"], 20)


def split_sessions(market: dict[str, Any], start: str, end: str) -> list[str]:
    dates = market["dates"]
    return [day for day in dates if start <= day <= end and market["date_index"][day] >= SEQ_LEN + 20 and market["date_index"][day] + REBALANCE_INTERVAL + 1 < len(dates)]


def due_sessions(market: dict[str, Any], start: str, end: str) -> list[str]:
    return split_sessions(market, start, end)[::REBALANCE_INTERVAL]


def pool_for_session(market: dict[str, Any], amount20_arr: np.ndarray, idx: int) -> list[int]:
    arrays = market["arrays"]
    valid = (
        np.isfinite(arrays["open"][idx]) & (arrays["open"][idx] > 0)
        & np.isfinite(arrays["close"][idx]) & (arrays["close"][idx] > 0)
        & np.isfinite(arrays["volume"][idx]) & (arrays["volume"][idx] > 0)
        & np.isfinite(amount20_arr[idx]) & (amount20_arr[idx] > 0)
    )
    cols = np.flatnonzero(valid)
    ordered = sorted(cols, key=lambda col: (-float(amount20_arr[idx, col]), str(market["symbols"][col])))[:POOL_SIZE]
    return [int(col) for col in ordered]


def future_return(market: dict[str, Any], idx: int, col: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + REBALANCE_INTERVAL + 1
    entry = float(market["arrays"]["open"][entry_idx, col])
    exit_ = float(market["arrays"]["open"][exit_idx, col])
    close_t = float(market["arrays"]["close"][idx, col])
    if not np.isfinite(entry) or not np.isfinite(exit_) or not np.isfinite(close_t) or entry <= 0 or exit_ <= 0 or close_t <= 0:
        return None
    if abs(entry / close_t - 1.0) > 0.095:
        return None
    return exit_ / entry - 1.0


def build_split_samples(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    raw_y: list[float] = []
    sessions: list[str] = []
    symbols: list[str] = []
    pools: dict[str, list[str]] = {}
    labels_by_session: dict[str, list[float]] = {}
    for session in due_sessions(market, start, end):
        idx = market["date_index"][session]
        cols = pool_for_session(market, amount20_arr, idx)
        regime_token = regime_tokens_from_pool(feats, cols, idx)
        session_records = []
        pool_symbols = [market["symbols"][col] for col in cols]
        pools[session] = pool_symbols
        for col in cols:
            y = future_return(market, idx, col)
            if y is None:
                continue
            x = feats[idx - SEQ_LEN + 1:idx + 1, col, :]
            if x.shape[0] != SEQ_LEN:
                continue
            x = np.vstack([x, regime_token[None, :]]).astype(np.float32)
            session_records.append((col, x, float(y)))
        if len(session_records) < TOP_K:
            continue
        returns = np.asarray([record[2] for record in session_records], dtype=float)
        ranks = pd.Series(returns).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        labels_by_session[session] = returns.tolist()
        for rank_label, (col, x, y) in zip(ranks, session_records):
            xs.append(x)
            raw_y.append(y)
            sessions.append(session)
            symbols.append(market["symbols"][col])
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y_rank": np.asarray(raw_y, dtype=np.float32),
        "target_rank": None,
        "sessions": sessions,
        "symbols": symbols,
        "pools": pools,
        "labels_by_session": labels_by_session,
    }


def attach_rank_targets(split: dict[str, Any]) -> None:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    target = np.zeros(len(split["sessions"]), dtype=np.float32)
    raw = split["y_rank"]
    for indices in by_session.values():
        ranks = pd.Series(raw[indices]).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        target[indices] = ranks
    split["target_rank"] = target


def attach_tail_targets(split: dict[str, Any]) -> None:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    target = np.zeros(len(split["sessions"]), dtype=np.float32)
    weight = np.full(len(split["sessions"]), MID_WEIGHT, dtype=np.float32)
    raw = split["y_rank"]
    for indices in by_session.values():
        returns = raw[indices]
        hi = float(np.quantile(returns, TOP_QUANTILE))
        lo = float(np.quantile(returns, BOTTOM_QUANTILE))
        for i in indices:
            if raw[i] >= hi:
                target[i] = 1.0
                weight[i] = TOP_WEIGHT
            elif raw[i] <= lo:
                target[i] = 0.0
                weight[i] = BOTTOM_WEIGHT
            else:
                target[i] = 0.0
                weight[i] = MID_WEIGHT
        scale = len(indices) / max(1e-9, float(np.sum(weight[indices])))
        weight[indices] *= scale
    split["target_tail"] = target
    split["sample_weight"] = weight


class TemporalTokenTransformer(nn.Module):
    def __init__(self, feature_dim: int, seq_len: int, d_model: int = 32, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=96, dropout=0.10, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos
        enc = self.encoder(z)
        pooled = torch.cat([enc[:, -1], enc.mean(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)


def predict_scores(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            outs.append(model(batch_x.to(device)).detach().cpu().numpy())
    return np.concatenate(outs) if outs else np.asarray([], dtype=float)


def selections_from_scores(split: dict[str, Any], scores: np.ndarray) -> dict[str, list[str]]:
    by_session: dict[str, list[tuple[str, float]]] = {}
    for session, symbol, score in zip(split["sessions"], split["symbols"], scores):
        by_session.setdefault(session, []).append((symbol, float(score)))
    return {session: [sym for sym, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:TOP_K]] for session, items in by_session.items()}


def rank_ic(split: dict[str, Any], scores: np.ndarray) -> dict[str, float]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    ics = []
    top10_labels = []
    for indices in by_session.values():
        if len(indices) < TOP_K:
            continue
        p = pd.Series(scores[indices]).rank().to_numpy(dtype=float)
        y = pd.Series(split["y_rank"][indices]).rank().to_numpy(dtype=float)
        if np.std(p) > 0 and np.std(y) > 0:
            ics.append(float(np.corrcoef(p, y)[0, 1]))
        ordered = sorted(indices, key=lambda i: -float(scores[i]))[:TOP_K]
        top10_labels.append(float(np.mean(split["y_rank"][ordered])))
    return {"mean_rank_ic": float(np.mean(ics)) if ics else 0.0, "median_rank_ic": float(np.median(ics)) if ics else 0.0, "mean_top10_raw_label": float(np.mean(top10_labels)) if top10_labels else 0.0, "sessions": len(by_session)}


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def replay_split(selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return POS.replay(selections, market, start, end)["summary"]


def train_model(train: dict[str, Any], valid: dict[str, Any], market: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model = TemporalTokenTransformer(train["x"].shape[2], MODEL_SEQ_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    dataset = TensorDataset(torch.from_numpy(train["x"]), torch.from_numpy(train["target_tail"]), torch.from_numpy(train["sample_weight"]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    history = []
    best_state = None
    best_valid_multiple = -1.0
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
        valid_scores = predict_scores(model, valid["x"], device)
        valid_sel = selections_from_scores(valid, valid_scores)
        valid_summary = replay_split(valid_sel, market, VALID_START, VALID_END)
        valid_ic = rank_ic(valid, valid_scores)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_multiple": valid_summary["final_multiple"], "valid_total_return": valid_summary["total_return"], "valid_rank_ic": valid_ic}
        history.append(row)
        if valid_summary["final_multiple"] > best_valid_multiple:
            best_valid_multiple = valid_summary["final_multiple"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": max(history, key=lambda row: row["valid_multiple"])["epoch"]}


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    market = POS.load_market()
    feats = build_feature_tensor(market)
    amount20_arr = amount20(market)
    train = build_split_samples(market, feats, amount20_arr, TRAIN_START, TRAIN_END)
    valid = build_split_samples(market, feats, amount20_arr, VALID_START, VALID_END)
    dev = build_split_samples(market, feats, amount20_arr, DEV_START, DEV_END)
    forward = build_split_samples(market, feats, amount20_arr, FWD_START, FWD_END)
    for split in (train, valid, dev, forward):
        attach_rank_targets(split)
        attach_tail_targets(split)
    trained = train_model(train, valid, market, device)
    model = trained["model"]
    split_data = {"train": (train, TRAIN_START, TRAIN_END), "valid": (valid, VALID_START, VALID_END), "dev": (dev, DEV_START, DEV_END), "forward": (forward, FWD_START, FWD_END)}
    results = {}
    for name, (split, start, end) in split_data.items():
        scores = predict_scores(model, split["x"], device)
        selections = selections_from_scores(split, scores)
        results[name] = {"summary": replay_split(selections, market, start, end), "rank_metrics": rank_ic(split, scores)}
    rng = np.random.default_rng(SEED)
    random_values = [replay_split(random_selections(forward["pools"], rng), market, FWD_START, FWD_END)["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = percentile_summary(random_values)
    fwd = results["forward"]["summary"]
    dev_summary = results["dev"]["summary"]
    verdict = "regime_token_transformer_not_enough"
    if dev_summary["final_multiple"] >= 20.0 and dev_summary["all_years_positive"] and fwd["final_multiple"] > random_summary["p95"] and fwd["final_multiple"] > 1.2 and fwd["avg_position_count"] > 5 and fwd["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0:
        verdict = "regime_token_transformer_candidate_needs_walkforward_expansion"
    out = {
        "experiment": "cross_sectional_regime_token_transformer_v1",
        "method": "small_temporal_transformer_stock_token_plus_cross_sectional_regime_token_to_next_5d_right_tail_weighted_bce_pool500_top10",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "seq_len": SEQ_LEN, "model_seq_len": MODEL_SEQ_LEN, "pool_size": POOL_SIZE, "top_k": TOP_K, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "device": str(device), "random_trials": RANDOM_TRIALS, "seed": SEED, "top_quantile": TOP_QUANTILE, "bottom_quantile": BOTTOM_QUANTILE, "top_weight": TOP_WEIGHT, "bottom_weight": BOTTOM_WEIGHT, "mid_weight": MID_WEIGHT, "regime_feature_names": REGIME_FEATURE_NAMES},
        "inputs_sha256": {"script": sha256(Path(__file__)), "positive_script": sha256(ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py")},
        "sample_counts": {name: len(split["sessions"]) for name, (split, _, _) in split_data.items()},
        "session_counts": {name: len(set(split["sessions"])) for name, (split, _, _) in split_data.items()},
        "training_history": trained["history"],
        "best_epoch_by_valid_multiple": trained["best_epoch"],
        "results": results,
        "forward_random_pool500_top10": random_summary,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_epoch": trained["best_epoch"], "results": results, "forward_random": random_summary, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
