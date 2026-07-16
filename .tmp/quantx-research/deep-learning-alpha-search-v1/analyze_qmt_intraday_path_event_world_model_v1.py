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
OUT = ROOT / "qmt_intraday_path_event_world_model_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

PERIOD = "5m"
INTERVAL = 5
POOL_SIZE = 500
FETCH_TOP = 50
TOP_KS = (10, 20)
MIN_TRAIN_SESSIONS = 18
MIN_VALID_SESSIONS = 6
MIN_FORWARD_SESSIONS = 10
MIN_TRAIN_ROWS = 700
MIN_VALID_ROWS = 200
MAX_TRAIN_SESSIONS_PER_YEAR = 8
MAX_VALID_SESSIONS = 10
MAX_FORWARD_SESSIONS = 24
EPOCHS = 35
BATCH_SIZE = 2048
LR = 8e-4
WEIGHT_DECAY = 2e-4
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


PATH_PROBE = load_module("qmt_intraday_path_probe_for_event_world_model", ROOT / "analyze_qmt_intraday_path_opportunity_probe_v1.py")
CVR = PATH_PROBE.CA52.CVR
RT = PATH_PROBE.RT
POS = PATH_PROBE.POS
LIM = PATH_PROBE.LIM


FEATURES = [
    "base_score",
    "rank_in_candidate",
    "minute_available",
    "intraday_return",
    "close_return",
    "first_half_return",
    "second_half_return",
    "tail_return",
    "range_amp",
    "close_pos_range",
    "vwap_support",
    "low_to_close_recovery",
    "high_to_close_fade",
    "drawdown_from_high",
    "tail_volume_share",
    "morning_volume_share",
    "late_volume_share",
    "volume_concentration_top20pct",
    "smooth_path_score",
]

EVENTS = {
    "daily_base": {"base_score": 1.0, "rank_in_candidate": -0.15},
    "tail_vwap_recovery": {"base_score": 0.25, "tail_return": 0.25, "vwap_support": 0.25, "low_to_close_recovery": 0.15, "close_pos_range": 0.10},
    "late_absorption": {"base_score": 0.20, "second_half_return": 0.25, "late_volume_share": 0.20, "vwap_support": 0.20, "volume_concentration_top20pct": -0.15},
    "anti_exhaustion_path": {"base_score": 0.20, "high_to_close_fade": 0.25, "drawdown_from_high": 0.20, "range_amp": -0.20, "volume_concentration_top20pct": -0.15},
    "path_quality_blend": {"base_score": 0.20, "tail_return": 0.20, "vwap_support": 0.20, "close_pos_range": 0.15, "second_half_return": 0.15, "smooth_path_score": 0.10},
}


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
    x = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pick_sessions(market: dict[str, Any], start: str, end: str, max_sessions: int | None, by_year: bool = False) -> list[str]:
    sessions = CVR.sessions(market, start, end, INTERVAL)
    if max_sessions is None or len(sessions) <= max_sessions:
        return sessions
    if by_year:
        picked: list[str] = []
        by_year_sessions: dict[str, list[str]] = {}
        for day in sessions:
            by_year_sessions.setdefault(day[:4], []).append(day)
        for year, values in sorted(by_year_sessions.items()):
            limit = min(MAX_TRAIN_SESSIONS_PER_YEAR, len(values))
            indices = np.linspace(0, len(values) - 1, limit).round().astype(int)
            picked.extend(values[int(i)] for i in indices)
        return sorted(set(picked))
    indices = np.linspace(0, len(sessions) - 1, max_sessions).round().astype(int)
    return sorted({sessions[int(i)] for i in indices})


def candidate_order(market: dict[str, Any], amount20_arr: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in cols:
        y = PATH_PROBE.future_return(market, idx, int(col))
        if y is None:
            continue
        labels.append(float(y))
        kept_cols.append(int(col))
    if len(labels) < max(TOP_KS):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    kept = np.asarray(kept_cols, dtype=int)
    scores = PATH_PROBE.CA52.candidate_scores(PATH_PROBE.CA52.feature_frame(market, idx, kept))["mid_trend_volume_not_extreme"]
    order = sorted(range(len(kept)), key=lambda j: (-float(scores[j]), str(market["symbols"][kept[j]])))[:FETCH_TOP]
    return kept[np.asarray(order, dtype=int)], scores[np.asarray(order, dtype=int)]


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, split_name: str, sessions: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fetch = {"requested": 0, "cached": 0, "available": 0, "empty": 0, "errors": 0}
    with LIM.QMTClient(max_retries=1, fill_data=False) as client:
        for session in sessions:
            idx = market["date_index"][session]
            cols, base_scores = candidate_order(market, amount20_arr, idx)
            if len(cols) < max(TOP_KS):
                continue
            for rank_pos, (col, base_score) in enumerate(zip(cols, base_scores, strict=True), 1):
                symbol = str(market["symbols"][int(col)])
                y = PATH_PROBE.future_return(market, idx, int(col))
                if y is None:
                    continue
                preclose = float(market["arrays"]["close"][idx - 1, int(col)])
                try:
                    path = LIM.cache_path(symbol, session, PERIOD)
                    if path.exists():
                        fetch["cached"] += 1
                    fetch["requested"] += 1
                    minute = LIM.load_or_fetch_minute(client, symbol, session, PERIOD)
                    day_frame = minute[minute["date"] == session] if not minute.empty and "date" in minute.columns else minute
                    minute_features = PATH_PROBE.intraday_path_features(day_frame, preclose)
                    if minute_features.get("minute_available", 0.0) > 0:
                        fetch["available"] += 1
                    else:
                        fetch["empty"] += 1
                except Exception:
                    fetch["errors"] += 1
                    minute_features = {"minute_available": 0.0}
                row = {
                    "split": split_name,
                    "date": session,
                    "symbol": symbol,
                    "future_return": float(y),
                    "base_score": float(base_score),
                    "rank_in_candidate": float((rank_pos - 1) / max(1, FETCH_TOP - 1)),
                }
                row.update({key: float(minute_features.get(key, 0.0)) for key in FEATURES if key not in row})
                rows.append(row)
    return {"name": split_name, "sessions": sessions, "rows": rows, "fetch": fetch}


def rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    x = np.asarray([[float(row.get(feature, 0.0)) for feature in FEATURES] for row in rows], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([float(row["future_return"]) for row in rows], dtype=np.float32)
    sessions = [str(row["date"]) for row in rows]
    symbols = [str(row["symbol"]) for row in rows]
    return x, y, sessions, symbols


def attach_session_rank_target(rows: list[dict[str, Any]]) -> np.ndarray:
    target = np.zeros(len(rows), dtype=np.float32)
    by_session: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_session.setdefault(str(row["date"]), []).append(i)
    labels = np.asarray([float(row["future_return"]) for row in rows], dtype=float)
    for indices in by_session.values():
        ranks = pd.Series(labels[indices]).rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
        target[indices] = ranks
    return target


def event_scores(rows: list[dict[str, Any]], weights: dict[str, float]) -> np.ndarray:
    by_session: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_session.setdefault(str(row["date"]), []).append(i)
    scores = np.zeros(len(rows), dtype=float)
    for indices in by_session.values():
        parts = []
        for feature, weight in weights.items():
            values = pd.Series([float(rows[i].get(feature, 0.0)) for i in indices]).rank(pct=True).fillna(0.5).to_numpy(dtype=float) - 0.5
            parts.append(weight * values)
        if parts:
            scores[indices] = np.sum(np.vstack(parts), axis=0)
    return scores


def selections_from_scores(rows: list[dict[str, Any]], scores: np.ndarray, top_k: int = 10) -> dict[str, list[str]]:
    by_session: dict[str, list[tuple[str, float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        by_session.setdefault(str(row["date"]), []).append((str(row["symbol"]), float(score)))
    return {session: [sym for sym, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]] for session, items in by_session.items()}


def metric_from_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    by_session: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_session.setdefault(str(row["date"]), []).append(i)
    labels = np.asarray([float(row["future_return"]) for row in rows], dtype=float)
    top_returns = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    for session, indices in sorted(by_session.items()):
        if len(indices) < max(TOP_KS):
            continue
        session_scores = scores[indices]
        session_labels = labels[indices]
        order = np.argsort(-session_scores, kind="mergesort")
        ics.append(rank_ic(session_scores, session_labels))
        for k in TOP_KS:
            value = safe_mean(session_labels[order[:k]])
            top_returns[k].append(value)
            if k == 20:
                by_year.setdefault(session[:4], []).append(value)
    return {
        "sessions": len(by_session),
        "mean_rank_ic": safe_mean(ics),
        "median_rank_ic": float(np.median(ics)) if ics else 0.0,
        "top_returns": {f"top{k}": safe_mean(values) for k, values in top_returns.items()},
        "by_year_top20": {year: safe_mean(values) for year, values in sorted(by_year.items())},
    }


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


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            out.append(model(torch.from_numpy(x[start:start + 4096])).cpu().numpy())
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def train_mlp(train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], fwd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    set_seed(SEED)
    train_x, _, _, _ = rows_to_arrays(train_rows)
    valid_x, _, _, _ = rows_to_arrays(valid_rows)
    fwd_x, _, _, _ = rows_to_arrays(fwd_rows)
    target = attach_session_rank_target(train_rows)
    model = SmallMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=0.10)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(target)), batch_size=BATCH_SIZE, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        valid_score = predict(model, valid_x)
        valid_metric = metric_from_scores(valid_rows, valid_score)
        score = valid_metric["top_returns"]["top20"] + 0.25 * valid_metric["top_returns"]["top10"] + 0.005 * valid_metric["mean_rank_ic"]
        history.append({"epoch": epoch, "loss": safe_mean(losses), "valid_score": score, "valid": valid_metric})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history_tail": history[-8:],
        "scores": {
            "train": predict(model, train_x).tolist(),
            "valid": predict(model, valid_x).tolist(),
            "forward": predict(model, fwd_x).tolist(),
        },
    }


def random_pool_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_session: dict[str, list[str]] = {}
    label_map = {(str(row["date"]), str(row["symbol"])): float(row["future_return"]) for row in rows}
    for row in rows:
        by_session.setdefault(str(row["date"]), []).append(str(row["symbol"]))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RANDOM_TRIALS):
        session_values = []
        for session, symbols in by_session.items():
            if len(symbols) < 20:
                continue
            picked = rng.choice(np.asarray(symbols, dtype=object), size=20, replace=False)
            session_values.append(safe_mean([label_map[(session, str(sym))] for sym in picked]))
        values.append(safe_mean(session_values))
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    rows = split["rows"]
    return {
        "planned_sessions": len(split["sessions"]),
        "sessions_with_rows": len({row["date"] for row in rows}),
        "rows": len(rows),
        "available_rows": sum(1 for row in rows if float(row.get("minute_available", 0.0)) > 0),
        "fetch": split["fetch"],
    }


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    train_sessions = pick_sessions(market, TRAIN_START, TRAIN_END, None, by_year=True)
    valid_sessions = pick_sessions(market, VALID_START, VALID_END, MAX_VALID_SESSIONS)
    fwd_sessions = pick_sessions(market, FWD_START, FWD_END, MAX_FORWARD_SESSIONS)

    splits = {
        "train": build_split(market, amount20_arr, "train", train_sessions),
        "valid": build_split(market, amount20_arr, "valid", valid_sessions),
        "forward": build_split(market, amount20_arr, "forward", fwd_sessions),
    }
    coverage = {name: summarize_split(split) for name, split in splits.items()}
    can_train = (
        coverage["train"]["sessions_with_rows"] >= MIN_TRAIN_SESSIONS
        and coverage["valid"]["sessions_with_rows"] >= MIN_VALID_SESSIONS
        and coverage["forward"]["sessions_with_rows"] >= MIN_FORWARD_SESSIONS
        and coverage["train"]["available_rows"] >= MIN_TRAIN_ROWS
        and coverage["valid"]["available_rows"] >= MIN_VALID_ROWS
    )

    leaderboard = []
    for name, weights in EVENTS.items():
        row = {"model": name, "kind": "event_rule"}
        for split_name, split in splits.items():
            scores = event_scores(split["rows"], weights)
            row[split_name] = metric_from_scores(split["rows"], scores)
        row["valid_score"] = row["valid"]["top_returns"]["top20"] + 0.25 * row["valid"]["top_returns"]["top10"] + 0.005 * row["valid"]["mean_rank_ic"]
        leaderboard.append(row)

    mlp_result: dict[str, Any] | None = None
    if can_train:
        mlp_result = train_mlp(splits["train"]["rows"], splits["valid"]["rows"], splits["forward"]["rows"])
        mlp_row = {"model": "small_mlp_path_event", "kind": "mlp"}
        for split_name in ["train", "valid", "forward"]:
            scores = np.asarray(mlp_result["scores"][split_name], dtype=float)
            mlp_row[split_name] = metric_from_scores(splits[split_name]["rows"], scores)
        mlp_row["valid_score"] = mlp_row["valid"]["top_returns"]["top20"] + 0.25 * mlp_row["valid"]["top_returns"]["top10"] + 0.005 * mlp_row["valid"]["mean_rank_ic"]
        leaderboard.append(mlp_row)

    leaderboard = sorted(leaderboard, key=lambda row: row["valid_score"], reverse=True)
    selected = leaderboard[0] if leaderboard else {}
    forward_random = random_pool_summary(splits["forward"]["rows"])
    verdict = "qmt_intraday_path_event_world_model_not_enough"
    if not can_train:
        verdict = "qmt_intraday_path_event_world_model_insufficient_dev_minute_coverage"
    elif (
        selected.get("forward", {}).get("top_returns", {}).get("top20", 0.0) > forward_random["p95"]
        and selected.get("valid", {}).get("top_returns", {}).get("top20", 0.0) > 0.0
        and selected.get("forward", {}).get("mean_rank_ic", 0.0) > 0.03
    ):
        verdict = "qmt_intraday_path_event_world_model_candidate_needs_full_replay"

    out = {
        "experiment": "qmt_intraday_path_event_world_model_v1",
        "method": "qmt_5m_path_event_tokens_with_coverage_gate_and_small_mlp_if_train_valid_available",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "forward": [FWD_START, FWD_END],
            "period": PERIOD,
            "interval": INTERVAL,
            "pool_size": POOL_SIZE,
            "fetch_top": FETCH_TOP,
            "top_ks": TOP_KS,
            "min_train_sessions": MIN_TRAIN_SESSIONS,
            "min_valid_sessions": MIN_VALID_SESSIONS,
            "min_forward_sessions": MIN_FORWARD_SESSIONS,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_valid_rows": MIN_VALID_ROWS,
            "seed": SEED,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "path_probe_script": sha256(ROOT / "analyze_qmt_intraday_path_opportunity_probe_v1.py")},
        "features": FEATURES,
        "events": EVENTS,
        "coverage": coverage,
        "can_train": can_train,
        "forward_random_top20_label": forward_random,
        "leaderboard": leaderboard,
        "selected": selected,
        "mlp_history_tail": mlp_result["history_tail"] if mlp_result else None,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "coverage": coverage, "can_train": can_train, "selected": selected, "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
