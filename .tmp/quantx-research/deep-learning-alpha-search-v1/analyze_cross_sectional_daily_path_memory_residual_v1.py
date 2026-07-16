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
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_daily_path_memory_residual_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

SEQ_LEN = 60
POOL_SIZE = 500
TOP_KS = (10, 20, 30)
TOP_K_REPLAY = 10
EPOCHS = 35
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4
NULL_SHRINK = 30.0
RESID_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]
RANDOM_TRIALS = 200
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RT = load_module("year_invariant_rank_transformer_for_daily_path_residual", ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py")
POS = RT.POS


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


def rank_ic(score: np.ndarray, label: np.ndarray) -> float:
    x = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(label).rank(method="average").to_numpy(dtype=float)
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def qbin(value: float, bins: np.ndarray) -> int:
    return int(np.searchsorted(bins, value, side="right"))


def base_score_from_last(last: np.ndarray) -> np.ndarray:
    return 0.25 * last[:, 10] + 0.25 * last[:, 9] + 0.20 * last[:, 6] + 0.15 * last[:, 4] - 0.15 * last[:, 3]


def build_path_features(feats: np.ndarray, idx: int, cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    window = feats[idx - SEQ_LEN + 1:idx + 1, cols, :]
    last = window[-1]
    mean5 = window[-5:].mean(axis=0)
    mean20 = window[-20:].mean(axis=0)
    mean60 = window.mean(axis=0)
    std20 = window[-20:].std(axis=0)
    drift = mean5 - mean20
    base = base_score_from_last(last)[:, None]
    liquidity_layer = last[:, 10:11]
    return np.concatenate([last, mean5, mean20, mean60, std20, drift, base, liquidity_layer], axis=1).astype(np.float32), base[:, 0].astype(np.float32)


def build_split(market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, start: str, end: str) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    raw_y: list[float] = []
    label_rank: list[float] = []
    sessions: list[str] = []
    symbols: list[str] = []
    base_scores: list[float] = []
    null_vars: list[tuple[float, float, float, float]] = []
    pools: dict[str, list[str]] = {}
    for session in RT.due_sessions(market, start, end):
        idx = market["date_index"][session]
        if idx < SEQ_LEN + 20:
            continue
        cols = RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE]
        if len(cols) < max(TOP_KS):
            continue
        x_all, base_all = build_path_features(feats, idx, cols)
        records = []
        for local_pos, col in enumerate(cols):
            y = RT.future_return(market, idx, col)
            if y is None:
                continue
            records.append((local_pos, col, float(y)))
        if len(records) < max(TOP_KS):
            continue
        returns = np.asarray([r[2] for r in records], dtype=float)
        ranks = pd.Series(returns).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        pools[session] = [str(market["symbols"][col]) for _, col, _ in records]
        for rank_value, (local_pos, col, y) in zip(ranks, records, strict=True):
            x = x_all[local_pos]
            xs.append(x)
            raw_y.append(y)
            label_rank.append(float(rank_value))
            sessions.append(session)
            symbols.append(str(market["symbols"][col]))
            base_scores.append(float(base_all[local_pos]))
            last = feats[idx, col]
            null_vars.append((float(last[10]), float(last[9]), float(last[3]), float(last[4])))
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "raw_y": np.asarray(raw_y, dtype=np.float32),
        "label_rank": np.asarray(label_rank, dtype=np.float32),
        "sessions": sessions,
        "symbols": symbols,
        "base_score": np.asarray(base_scores, dtype=np.float32),
        "null_vars": np.asarray(null_vars, dtype=np.float32),
        "pools": pools,
    }


def merge_splits(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": np.concatenate([train["x"], valid["x"]], axis=0),
        "raw_y": np.concatenate([train["raw_y"], valid["raw_y"]], axis=0),
        "label_rank": np.concatenate([train["label_rank"], valid["label_rank"]], axis=0),
        "sessions": train["sessions"] + valid["sessions"],
        "symbols": train["symbols"] + valid["symbols"],
        "base_score": np.concatenate([train["base_score"], valid["base_score"]], axis=0),
        "null_vars": np.concatenate([train["null_vars"], valid["null_vars"]], axis=0),
        "pools": {**train["pools"], **valid["pools"]},
    }


def build_null(train: dict[str, Any], splits: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_vars = train["null_vars"]
    bins = [np.quantile(train_vars[:, i], [0.2, 0.4, 0.6, 0.8]) for i in range(train_vars.shape[1])]
    global_mean = float(np.mean(train["label_rank"]))
    levels = ["liq_ret_vol_vwap", "liq_ret_vol", "liq_ret", "liq"]
    stats: dict[str, dict[tuple[int, ...], list[float]]] = {level: {} for level in levels}
    for values, y in zip(train_vars, train["label_rank"], strict=True):
        qs = tuple(qbin(float(values[i]), bins[i]) for i in range(len(bins)))
        for level, key in zip(levels, [qs, qs[:3], qs[:2], qs[:1]], strict=True):
            stats[level].setdefault(key, []).append(float(y))
    means = {level: {key: (safe_mean(vals), len(vals)) for key, vals in data.items()} for level, data in stats.items()}
    null_scores: dict[str, np.ndarray] = {}
    for split_name, split in splits.items():
        pred = np.full(len(split["label_rank"]), global_mean, dtype=np.float32)
        for i, values in enumerate(split["null_vars"]):
            qs = tuple(qbin(float(values[j]), bins[j]) for j in range(len(bins)))
            for level, key in zip(levels, [qs, qs[:3], qs[:2], qs[:1]], strict=True):
                item = means[level].get(key)
                if item is None:
                    continue
                mean_value, count = item
                if count >= 8:
                    w = count / (count + NULL_SHRINK)
                    pred[i] = float(w * mean_value + (1.0 - w) * global_mean)
                    break
        null_scores[split_name] = pred
    meta = {"global_mean": global_mean, "levels": {level: len(data) for level, data in means.items()}, "shrink": NULL_SHRINK, "bins": [b.tolist() for b in bins]}
    return null_scores, meta


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 96),
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
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            out.append(model(torch.from_numpy(x[start:start + 8192])).cpu().numpy())
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def train_residual(train: dict[str, Any], valid: dict[str, Any], null_scores: dict[str, np.ndarray]) -> tuple[ResidualMLP, list[dict[str, float]]]:
    set_seed(SEED)
    model = ResidualMLP(train["x"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    target = (train["label_rank"] - null_scores["train"]).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train["x"]), torch.from_numpy(target)), batch_size=BATCH_SIZE, shuffle=True)
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
        valid_resid = predict(model, valid["x"])
        valid_metrics = metric_from_scores(valid, valid_resid)
        score = valid_metrics["top_returns"]["top20"] + 0.25 * valid_metrics["top_returns"]["top10"] + 0.005 * valid_metrics["mean_rank_ic"]
        history.append({"epoch": float(epoch), "loss": safe_mean(losses), "valid_score": score, "valid_top20": valid_metrics["top_returns"]["top20"], "valid_top10": valid_metrics["top_returns"]["top10"], "valid_ic": valid_metrics["mean_rank_ic"]})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def metric_from_scores(split: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    overlap = []
    by_year: dict[str, list[float]] = {}
    for session, indices in sorted(by_session.items()):
        if len(indices) < max(TOP_KS):
            continue
        idx_arr = np.asarray(indices, dtype=int)
        y = split["raw_y"][idx_arr]
        score = scores[idx_arr]
        base_score = split["base_score"][idx_arr]
        order = np.argsort(-score, kind="mergesort")
        base_order = np.argsort(-base_score, kind="mergesort")
        ics.append(rank_ic(score, y))
        overlap.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base_top[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(session[:4], []).append(value)
    return {"sessions": len(by_session), "top_returns": {f"top{k}": safe_mean(values) for k, values in top.items()}, "base_top_returns": {f"top{k}": safe_mean(values) for k, values in base_top.items()}, "mean_rank_ic": safe_mean(ics), "median_rank_ic": float(np.median(ics)) if ics else 0.0, "top20_overlap_with_base": safe_mean(overlap), "by_year_top20": {year: safe_mean(values) for year, values in sorted(by_year.items())}}


def selections_from_scores(split: dict[str, Any], scores: np.ndarray, top_k: int = TOP_K_REPLAY) -> dict[str, list[str]]:
    by_session: dict[str, list[tuple[str, float]]] = {}
    for session, symbol, score in zip(split["sessions"], split["symbols"], scores, strict=True):
        by_session.setdefault(session, []).append((symbol, float(score)))
    return {session: [sym for sym, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]] for session, items in by_session.items()}


def replay_summary(split: dict[str, Any], scores: np.ndarray, market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return POS.replay(selections_from_scores(split, scores), market, start, end)["summary"]


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K_REPLAY, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K_REPLAY}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def evaluate_mode(split: dict[str, Any], mode: str, weight: float, null: np.ndarray, resid: np.ndarray) -> np.ndarray:
    if mode == "base_daily":
        return split["base_score"]
    if mode == "null_only":
        return null
    if mode == "residual_only":
        return resid
    return null + weight * resid


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    train = build_split(market, feats, amount20_arr, TRAIN_START, TRAIN_END)
    valid = build_split(market, feats, amount20_arr, VALID_START, VALID_END)
    dev = merge_splits(train, valid)
    forward = build_split(market, feats, amount20_arr, FWD_START, FWD_END)
    splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
    null_scores, null_meta = build_null(train, splits)
    model, history = train_residual(train, valid, null_scores)
    resid_scores = {name: predict(model, split["x"]) for name, split in splits.items()}
    rows = []
    configs = [("base_daily", 0.0), ("null_only", 0.0), ("residual_only", 1.0)] + [("null_plus_residual", w) for w in RESID_WEIGHTS]
    for mode, weight in configs:
        row: dict[str, Any] = {"mode": mode, "weight": weight}
        for name, split in splits.items():
            scores = evaluate_mode(split, mode, weight, null_scores[name], resid_scores[name])
            row[name] = metric_from_scores(split, scores)
        row["valid_score"] = row["valid"]["top_returns"]["top20"] + 0.25 * row["valid"]["top_returns"]["top10"] + 0.005 * row["valid"]["mean_rank_ic"]
        rows.append(row)
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    replay = {}
    for name, split, start, end in [("dev", dev, DEV_START, DEV_END), ("forward", forward, FWD_START, FWD_END)]:
        scores = evaluate_mode(split, selected["mode"], selected["weight"], null_scores[name], resid_scores[name])
        replay[name] = replay_summary(split, scores, market, start, end)
    base_replay = {
        "dev": replay_summary(dev, dev["base_score"], market, DEV_START, DEV_END),
        "forward": replay_summary(forward, forward["base_score"], market, FWD_START, FWD_END),
    }
    rng = np.random.default_rng(SEED)
    random_values = [POS.replay(random_selections(forward["pools"], rng), market, FWD_START, FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = percentile_summary(random_values)
    verdict = "cross_sectional_daily_path_memory_residual_not_enough"
    if (
        replay["dev"]["final_multiple"] >= 20.0
        and replay["dev"]["all_years_positive"]
        and replay["forward"]["final_multiple"] > random_summary["p95"]
        and replay["forward"]["final_multiple"] > base_replay["forward"]["final_multiple"]
        and replay["forward"]["avg_position_count"] > 5
        and replay["forward"]["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0
    ):
        verdict = "cross_sectional_daily_path_memory_residual_candidate_needs_walkforward"
    out = {
        "experiment": "cross_sectional_daily_path_memory_residual_v1",
        "method": "stock_level_daily_60d_path_compressed_features_train_only_liq_momentum_vol_vwap_null_plus_residual_mlp_next_5d_rank",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "seq_len": SEQ_LEN, "pool_size": POOL_SIZE, "top_ks": TOP_KS, "top_k_replay": TOP_K_REPLAY, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "null_shrink": NULL_SHRINK, "resid_weights": RESID_WEIGHTS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "rt_script": sha256(ROOT / "train_cross_sectional_year_invariant_rank_transformer_v1.py"), "pos_script": sha256(ROOT / "analyze_cross_sectional_multihorizon_soil_scan_v1.py")},
        "sample_counts": {name: int(len(split["label_rank"])) for name, split in splits.items()},
        "session_counts": {name: int(len(set(split["sessions"]))) for name, split in splits.items()},
        "null_meta": null_meta,
        "history_tail": history[-8:],
        "leaderboard": rows,
        "selected": selected,
        "selected_replay": replay,
        "base_replay": base_replay,
        "forward_random_pool500_top10": random_summary,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "session_counts": out["session_counts"], "selected": selected, "selected_replay": replay, "base_replay": base_replay, "forward_random": random_summary, "history_tail": history[-5:], "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
