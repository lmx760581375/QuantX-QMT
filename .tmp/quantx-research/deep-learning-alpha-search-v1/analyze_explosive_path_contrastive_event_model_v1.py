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
R83_PATH = ROOT / "analyze_contrastive_pairwise_diffusion_ranker_v1.py"
OUT = ROOT / "explosive_path_contrastive_event_model_v1_summary.json"

SEED = 20260714
EPOCHS = 22
BATCH_SIZE = 4096
LR = 8e-4
WEIGHT_DECAY = 2e-4
MAX_POS_PER_SESSION = 48
MAX_NEG_PER_SESSION = 48
MAX_PAIRS_PER_SESSION = 1600
TEMPERATURES = (0.07, 0.15, 0.30)
RANDOM_TRIALS = 200

LABEL_VARIANTS: dict[str, dict[str, float]] = {
    "loose_explosive_holdable": {
        "pos_max_high10": 0.10,
        "pos_ret10": 0.045,
        "pos_min_open10": -0.085,
        "pos_giveback": 0.115,
        "neg_ret10": -0.050,
        "neg_min_open10": -0.105,
        "neg_failed_max_high10": 0.075,
        "neg_failed_ret10": 0.010,
        "neg_failed_giveback": 0.090,
    },
    "strict_explosive_holdable": {
        "pos_max_high10": 0.14,
        "pos_ret10": 0.070,
        "pos_min_open10": -0.075,
        "pos_giveback": 0.105,
        "neg_ret10": -0.055,
        "neg_min_open10": -0.105,
        "neg_failed_max_high10": 0.090,
        "neg_failed_ret10": 0.015,
        "neg_failed_giveback": 0.100,
    },
    "smooth_explosive_holdable": {
        "pos_max_high10": 0.115,
        "pos_ret10": 0.060,
        "pos_min_open10": -0.060,
        "pos_giveback": 0.085,
        "neg_ret10": -0.045,
        "neg_min_open10": -0.090,
        "neg_failed_max_high10": 0.080,
        "neg_failed_ret10": 0.020,
        "neg_failed_giveback": 0.075,
    },
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R83 = load_module("contrastive_pairwise_ranker_for_explosive_path_event", R83_PATH)
R74 = R83.R74


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


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num / den)


def path_stats(market: dict[str, Any], session: str, symbol: str) -> dict[str, float] | None:
    idx = market["date_index"][session]
    col = market["symbol_index"].get(symbol)
    if col is None or idx + 11 >= len(market["dates"]):
        return None
    arrays = market["arrays"]
    entry = float(arrays["open"][idx + 1, col])
    close_t = float(arrays["close"][idx, col])
    if not np.isfinite(entry) or not np.isfinite(close_t) or entry <= 0 or close_t <= 0:
        return None
    if abs(entry / close_t - 1.0) > 0.095:
        return None
    open_path = arrays["open"][idx + 1:idx + 12, col].astype(float)
    high_path = arrays["high"][idx + 1:idx + 11, col].astype(float)
    low_path = arrays["low"][idx + 1:idx + 11, col].astype(float)
    close_path = arrays["close"][idx + 1:idx + 11, col].astype(float)
    if len(open_path) < 11 or len(high_path) < 10 or not np.isfinite(open_path[[0, 5, 10]]).all():
        return None
    ret_open_path = open_path / entry - 1.0
    max_high10 = safe_div(float(np.nanmax(high_path)), entry) - 1.0
    max_open10 = safe_div(float(np.nanmax(open_path)), entry) - 1.0
    min_open10 = safe_div(float(np.nanmin(open_path[1:])), entry) - 1.0
    min_low10 = safe_div(float(np.nanmin(low_path)), entry) - 1.0
    ret5 = safe_div(float(open_path[5]), entry) - 1.0
    ret10 = safe_div(float(open_path[10]), entry) - 1.0
    close10 = safe_div(float(close_path[-1]), entry) - 1.0 if len(close_path) else float("nan")
    giveback = max_high10 - ret10 if np.isfinite(max_high10) and np.isfinite(ret10) else float("nan")
    open_vol = float(np.nanstd(ret_open_path[1:])) if len(ret_open_path) > 1 else 0.0
    efficiency = ret10 / max_high10 if np.isfinite(ret10) and np.isfinite(max_high10) and max_high10 > 1e-9 else 0.0
    return {
        "future_ret5_open": float(ret5),
        "future_ret10_open": float(ret10),
        "future_close10_from_entry": float(close10),
        "future_max_high10": float(max_high10),
        "future_max_open10": float(max_open10),
        "future_min_open10": float(min_open10),
        "future_min_low10": float(min_low10),
        "future_giveback_high_to_exit": float(giveback),
        "future_open_path_vol10": open_vol,
        "future_path_efficiency10": float(efficiency),
    }


def event_label(stats: dict[str, float], cfg: dict[str, float]) -> int:
    pos = (
        stats["future_max_high10"] >= cfg["pos_max_high10"]
        and stats["future_ret10_open"] >= cfg["pos_ret10"]
        and stats["future_min_open10"] >= cfg["pos_min_open10"]
        and stats["future_giveback_high_to_exit"] <= cfg["pos_giveback"]
    )
    failed_spike = (
        stats["future_max_high10"] >= cfg["neg_failed_max_high10"]
        and stats["future_ret10_open"] <= cfg["neg_failed_ret10"]
        and stats["future_giveback_high_to_exit"] >= cfg["neg_failed_giveback"]
    )
    neg = stats["future_ret10_open"] <= cfg["neg_ret10"] or stats["future_min_open10"] <= cfg["neg_min_open10"] or failed_spike
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def enrich_split(market: dict[str, Any], split: dict[str, Any], label_variant: str) -> dict[str, Any]:
    cfg = LABEL_VARIANTS[label_variant]
    labels = []
    valid_mask = []
    stats_rows: dict[str, list[float]] = {
        "future_ret5_open": [],
        "future_ret10_open": [],
        "future_close10_from_entry": [],
        "future_max_high10": [],
        "future_max_open10": [],
        "future_min_open10": [],
        "future_min_low10": [],
        "future_giveback_high_to_exit": [],
        "future_open_path_vol10": [],
        "future_path_efficiency10": [],
    }
    for session, symbol in zip(split["sessions"], split["symbols"], strict=True):
        stats = path_stats(market, session, symbol)
        ok = stats is not None and all(np.isfinite(v) for v in stats.values())
        valid_mask.append(ok)
        if ok and stats is not None:
            labels.append(event_label(stats, cfg))
            for key in stats_rows:
                stats_rows[key].append(float(stats[key]))
        else:
            labels.append(0)
            for key in stats_rows:
                stats_rows[key].append(float("nan"))
    out = dict(split)
    out["event_label"] = np.asarray(labels, dtype=np.int8)
    out["path_valid"] = np.asarray(valid_mask, dtype=bool)
    for key, values in stats_rows.items():
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def filter_valid_path(split: dict[str, Any]) -> dict[str, Any]:
    mask = split["path_valid"]
    pools: dict[str, list[str]] = {}
    keep_indices = np.flatnonzero(mask)
    sessions = [split["sessions"][int(i)] for i in keep_indices]
    symbols = [split["symbols"][int(i)] for i in keep_indices]
    for session, symbol in zip(sessions, symbols, strict=True):
        pools.setdefault(session, []).append(symbol)
    out: dict[str, Any] = {"sessions": sessions, "symbols": symbols, "pools": pools}
    for key, value in split.items():
        if key in {"sessions", "symbols", "pools"}:
            continue
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def merge_splits(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sessions": train["sessions"] + valid["sessions"],
        "symbols": train["symbols"] + valid["symbols"],
        "pools": {**train["pools"], **valid["pools"]},
    }
    for key, value in train.items():
        if key in out or key in {"sessions", "symbols", "pools"}:
            continue
        if isinstance(value, np.ndarray):
            out[key] = np.concatenate([value, valid[key]], axis=0)
    return out


def session_indices(split: dict[str, Any]) -> list[np.ndarray]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(by_session.items())]


def event_pair_indices(split: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_all: list[int] = []
    neg_all: list[int] = []
    labels = split["event_label"]
    for indices in session_indices(split):
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


def event_pair_auc(scores: np.ndarray, split: dict[str, Any]) -> dict[str, float]:
    aucs = []
    margins = []
    labels = split["event_label"]
    for indices in session_indices(split):
        pos = indices[labels[indices] == 1]
        neg = indices[labels[indices] == -1]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = scores[pos][:, None] - scores[neg][None, :]
        aucs.append(float(np.mean(diff > 0.0) + 0.5 * np.mean(diff == 0.0)))
        margins.append(float(np.mean(diff)))
    return {"event_pair_auc": safe_mean(aucs), "event_pair_margin": safe_mean(margins), "sessions": len(aucs)}


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


def event_metrics(split: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    rows = {"top10": [], "top20": []}
    hit = {"top10": [], "top20": []}
    fail = {"top10": [], "top20": []}
    ret10 = {"top10": [], "top20": []}
    max_high = {"top10": [], "top20": []}
    by_year_top20_hit: dict[str, list[float]] = {}
    for session, indices_list in sorted(by_session.items()):
        if len(indices_list) < 20:
            continue
        indices = np.asarray(indices_list, dtype=int)
        order = np.argsort(-scores[indices], kind="mergesort")
        labels = split["event_label"][indices]
        for k in (10, 20):
            take = order[:k]
            key = f"top{k}"
            rows[key].append(float(len(take)))
            hit_value = safe_mean(labels[take] == 1)
            fail_value = safe_mean(labels[take] == -1)
            hit[key].append(hit_value)
            fail[key].append(fail_value)
            ret10[key].append(safe_mean(split["future_ret10_open"][indices][take]))
            max_high[key].append(safe_mean(split["future_max_high10"][indices][take]))
            if k == 20:
                by_year_top20_hit.setdefault(session[:4], []).append(hit_value)
    return {
        "event_hit_rate": {key: safe_mean(values) for key, values in hit.items()},
        "event_fail_rate": {key: safe_mean(values) for key, values in fail.items()},
        "future_ret10_open": {key: safe_mean(values) for key, values in ret10.items()},
        "future_max_high10": {key: safe_mean(values) for key, values in max_high.items()},
        "by_year_top20_event_hit": {year: safe_mean(values) for year, values in sorted(by_year_top20_hit.items())},
    }


def train_model(train_x: np.ndarray, valid_x: np.ndarray, train: dict[str, Any], valid: dict[str, Any], temperature: float, seed: int) -> tuple[EventScorerMLP, list[dict[str, float]]]:
    set_seed(seed)
    model = EventScorerMLP(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_idx, neg_idx = event_pair_indices(train, seed)
    if len(pos_idx) == 0:
        raise RuntimeError("no event pairs built from train split")
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
        ret_metric = R74.metric_from_scores(valid, valid_scores)
        ev_metric = event_metrics(valid, valid_scores)
        pair = event_pair_auc(valid_scores, valid)
        score = (
            0.28 * ev_metric["event_hit_rate"]["top10"]
            - 0.18 * ev_metric["event_fail_rate"]["top10"]
            + 0.22 * ret_metric["top_returns"]["top10"]
            + 0.18 * ret_metric["top_returns"]["top20"]
            + 0.14 * pair["event_pair_auc"]
        )
        history.append({
            "epoch": float(epoch),
            "loss": safe_mean(losses),
            "valid_score": score,
            "valid_top10_ret5": ret_metric["top_returns"]["top10"],
            "valid_top20_ret5": ret_metric["top_returns"]["top20"],
            "valid_top10_event_hit": ev_metric["event_hit_rate"]["top10"],
            "valid_top10_event_fail": ev_metric["event_fail_rate"]["top10"],
            "valid_event_pair_auc": pair["event_pair_auc"],
        })
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_scores(split: dict[str, Any], scores: np.ndarray, market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return {
        "return_metrics": R74.metric_from_scores(split, scores),
        "event_metrics": event_metrics(split, scores),
        "event_pair": event_pair_auc(scores, split),
        "replay": R74.replay_summary(split, scores, market, start, end),
    }


def event_soil(split: dict[str, Any]) -> dict[str, Any]:
    labels = split["event_label"]
    out = {
        "samples": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == -1)),
        "neutral": int(np.sum(labels == 0)),
        "positive_rate": safe_mean(labels == 1),
        "negative_rate": safe_mean(labels == -1),
    }
    by_year: dict[str, list[int]] = {}
    for session, label in zip(split["sessions"], labels, strict=True):
        by_year.setdefault(session[:4], []).append(int(label))
    out["by_year"] = {
        year: {
            "samples": len(values),
            "positive_rate": safe_mean(np.asarray(values) == 1),
            "negative_rate": safe_mean(np.asarray(values) == -1),
        }
        for year, values in sorted(by_year.items())
    }
    return out


def random_summary(forward: dict[str, Any], market: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = [R74.POS.replay(R74.random_selections(forward["pools"], rng), market, R74.FWD_START, R74.FWD_END)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    return R74.percentile_summary(values)


def compact_result(item: dict[str, Any]) -> dict[str, Any]:
    valid = item["eval"]["valid"]
    dev = item["eval"]["dev"]
    forward = item["eval"]["forward"]
    return {
        "config": item["config"],
        "label_variant": item["label_variant"],
        "temperature": item["temperature"],
        "valid_score": item["valid_score"],
        "valid_top10_ret5": valid["return_metrics"]["top_returns"]["top10"],
        "valid_top20_ret5": valid["return_metrics"]["top_returns"]["top20"],
        "valid_top10_event_hit": valid["event_metrics"]["event_hit_rate"]["top10"],
        "valid_top10_event_fail": valid["event_metrics"]["event_fail_rate"]["top10"],
        "valid_event_pair_auc": valid["event_pair"]["event_pair_auc"],
        "dev_multiple": dev["replay"]["final_multiple"],
        "dev_all_years_positive": dev["replay"]["all_years_positive"],
        "dev_by_year_top20_ret5": dev["return_metrics"]["by_year_top20"],
        "dev_top20_event_hit_by_year": dev["event_metrics"]["by_year_top20_event_hit"],
        "forward_multiple": forward["replay"]["final_multiple"],
        "forward_top10_ret5": forward["return_metrics"]["top_returns"]["top10"],
        "forward_top20_ret5": forward["return_metrics"]["top_returns"]["top20"],
        "forward_top10_event_hit": forward["event_metrics"]["event_hit_rate"]["top10"],
        "forward_top10_event_fail": forward["event_metrics"]["event_fail_rate"]["top10"],
        "forward_event_pair_auc": forward["event_pair"]["event_pair_auc"],
        "forward_remove_best_3": forward["replay"]["remove_best_period_multiples"].get("remove_best_3", 0.0),
        "forward_avg_position": forward["replay"]["avg_position_count"],
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = R74.POS.load_market()
    feats = R74.RT.build_feature_tensor(market)
    amount20_arr = R74.RT.amount20(market)
    raw_train = R74.build_split(market, feats, amount20_arr, R74.TRAIN_START, R74.TRAIN_END)
    raw_valid = R74.build_split(market, feats, amount20_arr, R74.VALID_START, R74.VALID_END)
    raw_forward = R74.build_split(market, feats, amount20_arr, R74.FWD_START, R74.FWD_END)
    results = []
    soil_by_variant = {}
    for label_variant in LABEL_VARIANTS:
        train = filter_valid_path(enrich_split(market, raw_train, label_variant))
        valid = filter_valid_path(enrich_split(market, raw_valid, label_variant))
        forward = filter_valid_path(enrich_split(market, raw_forward, label_variant))
        dev = merge_splits(train, valid)
        splits = {"train": train, "valid": valid, "dev": dev, "forward": forward}
        soil_by_variant[label_variant] = {name: event_soil(split) for name, split in splits.items()}
        raw_x = {name: R83.transform_x(split["x"], "real_path") for name, split in splits.items()}
        mu, sd = R83.standardizer(raw_x["train"])
        x_by_split = {name: R83.apply_standardizer(x, mu, sd) for name, x in raw_x.items()}
        for temperature in TEMPERATURES:
            seed = SEED + int(temperature * 1000) + 100000 * (1 + list(LABEL_VARIANTS).index(label_variant))
            model, history = train_model(x_by_split["train"], x_by_split["valid"], train, valid, temperature, seed)
            scores = {name: predict(model, x_by_split[name]) for name in splits}
            evaluated = {
                "train": evaluate_scores(train, scores["train"], market, R74.TRAIN_START, R74.TRAIN_END),
                "valid": evaluate_scores(valid, scores["valid"], market, R74.VALID_START, R74.VALID_END),
                "dev": evaluate_scores(dev, scores["dev"], market, R74.DEV_START, R74.DEV_END),
                "forward": evaluate_scores(forward, scores["forward"], market, R74.FWD_START, R74.FWD_END),
            }
            valid_eval = evaluated["valid"]
            valid_score = (
                0.28 * valid_eval["event_metrics"]["event_hit_rate"]["top10"]
                - 0.18 * valid_eval["event_metrics"]["event_fail_rate"]["top10"]
                + 0.22 * valid_eval["return_metrics"]["top_returns"]["top10"]
                + 0.18 * valid_eval["return_metrics"]["top_returns"]["top20"]
                + 0.14 * valid_eval["event_pair"]["event_pair_auc"]
            )
            results.append({
                "config": f"{label_variant}_pairwise_event_t{temperature}",
                "label_variant": label_variant,
                "temperature": temperature,
                "valid_score": valid_score,
                "history_tail": history[-8:],
                "eval": evaluated,
            })
    leaderboard = sorted([compact_result(item) for item in results], key=lambda row: row["valid_score"], reverse=True)
    selected = leaderboard[0]
    best_forward = max(leaderboard, key=lambda row: row["forward_multiple"])
    best_forward_hit = max(leaderboard, key=lambda row: row["forward_top10_event_hit"])
    forward_for_random = filter_valid_path(enrich_split(market, raw_forward, selected["label_variant"]))
    forward_random = random_summary(forward_for_random, market)
    verdict = "explosive_path_contrastive_event_model_not_enough"
    if (
        selected["dev_multiple"] >= 20.0
        and selected["dev_all_years_positive"]
        and selected["forward_multiple"] > forward_random["p95"]
        and selected["forward_remove_best_3"] > 1.0
        and selected["forward_avg_position"] > 5
    ):
        verdict = "explosive_path_contrastive_event_model_candidate_needs_walkforward"
    out = {
        "experiment": "explosive_path_contrastive_event_model_v1",
        "method": "same_session_positive_negative_pairwise_contrastive_model_for_future_10d_holdable_explosive_path_events_on_daily_60d_path_features",
        "params": {
            "train": [R74.TRAIN_START, R74.TRAIN_END],
            "valid": [R74.VALID_START, R74.VALID_END],
            "dev": [R74.DEV_START, R74.DEV_END],
            "forward": [R74.FWD_START, R74.FWD_END],
            "label_variants": LABEL_VARIANTS,
            "temperatures": TEMPERATURES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_pos_per_session": MAX_POS_PER_SESSION,
            "max_neg_per_session": MAX_NEG_PER_SESSION,
            "max_pairs_per_session": MAX_PAIRS_PER_SESSION,
            "seed": SEED,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round83_script": sha256(R83_PATH),
            "round74_script": sha256(R83.R74_PATH),
        },
        "raw_sample_counts": {"train": len(raw_train["label_rank"]), "valid": len(raw_valid["label_rank"]), "forward": len(raw_forward["label_rank"])},
        "event_soil": soil_by_variant,
        "leaderboard": leaderboard,
        "selected_by_valid_event_score": selected,
        "best_by_forward_multiple": best_forward,
        "best_by_forward_event_hit": best_forward_hit,
        "results": results,
        "forward_random_pool500_top10": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "selected_by_valid_event_score": selected,
        "best_by_forward_multiple": best_forward,
        "best_by_forward_event_hit": best_forward_hit,
        "forward_random": forward_random,
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
