from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_opportunity_first_world_model_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

POOL_SIZE = 500
TOP_K = 20
INTERVAL = 5
SEED = 20260714
EPOCHS = 120
LR = 8e-4
WEIGHT_DECAY = 1e-3
HIDDEN = 32

CANDIDATES = [
    "avoid_extreme_trend_low_range",
    "mid_trend_not_extreme",
    "mid_trend_volume_not_extreme",
    "mid_trend_low_crowding",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CA52 = load_module("crowded_trend_exhaustion_avoidance_for_opportunity_first", ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py")
RT = CA52.RT
POS = CA52.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def clean_stat(values: np.ndarray, fn: str) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "std":
        return float(np.std(arr))
    if fn.startswith("q"):
        return float(np.quantile(arr, float(fn[1:]) / 100.0))
    raise ValueError(fn)


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 4:
        return 0.0
    xr = np.argsort(np.argsort(x[mask], kind="mergesort"), kind="mergesort").astype(float)
    yr = np.argsort(np.argsort(y[mask], kind="mergesort"), kind="mergesort").astype(float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def future_return(market: dict[str, Any], idx: int, col: int) -> float | None:
    return CA52.future_return(market, idx, col, INTERVAL)


def feature_vector(market: dict[str, Any], idx: int, cols: np.ndarray, candidate_scores: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    high = arrays["high"].astype(float)
    low = arrays["low"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    ret1 = safe_div(close[idx], close[idx - 1]) - 1.0
    ret5 = safe_div(close[idx], close[idx - 5]) - 1.0
    ret20 = safe_div(close[idx], close[idx - 20]) - 1.0
    ret60 = safe_div(close[idx], close[idx - 60]) - 1.0
    day_range = safe_div(high[idx], low[idx]) - 1.0
    close_vwap = safe_div(close[idx], vwap[idx]) - 1.0
    vol20 = np.nanmean(volume[idx - 19:idx + 1], axis=0)
    vol60 = np.nanmean(volume[idx - 59:idx + 1], axis=0)
    vol20_vs_60 = np.log(np.where(safe_div(vol20, vol60) > 0, safe_div(vol20, vol60), np.nan))
    amount = np.where(np.isfinite(vwap[idx]) & (vwap[idx] > 0), vwap[idx], close[idx]) * volume[idx]
    pool = {
        "ret1": ret1[cols],
        "ret5": ret5[cols],
        "ret20": ret20[cols],
        "ret60": ret60[cols],
        "range": day_range[cols],
        "close_vwap": close_vwap[cols],
        "vol20_vs_60": vol20_vs_60[cols],
        "amount": amount[cols],
    }
    valid = np.isfinite(ret20)
    out: dict[str, float] = {
        "mkt_breadth1": float(np.nanmean(ret1[valid] > 0)),
        "mkt_breadth5": float(np.nanmean(ret5[valid] > 0)),
        "mkt_breadth20": float(np.nanmean(ret20[valid] > 0)),
        "mkt_ret20_mean": clean_stat(ret20[valid], "mean"),
        "mkt_ret20_std": clean_stat(ret20[valid], "std"),
        "mkt_ret60_mean": clean_stat(ret60[valid], "mean"),
        "mkt_range_mean": clean_stat(day_range[valid], "mean"),
        "mkt_vol20_vs_60_mean": clean_stat(vol20_vs_60[valid], "mean"),
    }
    for name, values in pool.items():
        for stat in ["mean", "std", "q10", "q30", "q50", "q70", "q90"]:
            out[f"pool_{name}_{stat}"] = clean_stat(values, stat)
        out[f"pool_{name}_tail_spread"] = clean_stat(values, "q90") - clean_stat(values, "q10")
    score_stack = np.vstack([candidate_scores[name] for name in CANDIDATES])
    best_score = np.nanmax(score_stack, axis=0)
    out["candidate_score_best_mean"] = clean_stat(best_score, "mean")
    out["candidate_score_best_std"] = clean_stat(best_score, "std")
    out["candidate_score_best_q90"] = clean_stat(best_score, "q90")
    for name in CANDIDATES:
        out[f"score_{name}_mean"] = clean_stat(candidate_scores[name], "mean")
        out[f"score_{name}_std"] = clean_stat(candidate_scores[name], "std")
        out[f"score_{name}_q90"] = clean_stat(candidate_scores[name], "q90")
    return out


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str) -> list[dict[str, Any]]:
    rows = []
    for session in CA52.CVR.sessions(market, start, end, INTERVAL):
        idx = market["date_index"][session]
        cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
        if len(cols) < TOP_K:
            continue
        labels = []
        kept_cols = []
        for col in cols:
            y = future_return(market, idx, int(col))
            if y is None:
                continue
            labels.append(y)
            kept_cols.append(int(col))
        if len(labels) < TOP_K:
            continue
        kept = np.asarray(kept_cols, dtype=int)
        labels_arr = np.asarray(labels, dtype=float)
        scores = CA52.candidate_scores(CA52.feature_frame(market, idx, kept))
        candidate_returns = {}
        for name in CANDIDATES:
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(scores[name][j]), str(market["symbols"][kept[j]])))
            candidate_returns[name] = float(np.mean(labels_arr[order[:TOP_K]]))
        sorted_y = np.sort(labels_arr)[::-1]
        rows.append({
            "date": session,
            "year": session[:4],
            "x_dict": feature_vector(market, idx, kept, scores),
            "labels": {
                "pool_mean": float(np.mean(labels_arr)),
                "pool_median": float(np.median(labels_arr)),
                "pool_positive_ratio": float(np.mean(labels_arr > 0)),
                "oracle_top20": float(np.mean(sorted_y[:TOP_K])),
                "oracle_top50": float(np.mean(sorted_y[:50])),
                "bottom20": float(np.mean(sorted_y[-TOP_K:])),
                "tail_spread20": float(np.mean(sorted_y[:TOP_K]) - np.mean(sorted_y[-TOP_K:])),
                "candidate_best_top20": max(candidate_returns.values()),
            },
            "candidate_top20": candidate_returns,
        })
    return rows


def rows_to_arrays(rows: list[dict[str, Any]], feature_names: list[str], target_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[row["x_dict"].get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([[row["labels"][name] for name in target_names] for row in rows], dtype=np.float32)
    return x, y


class OpportunityMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = np.nanmean(train_x, axis=0)
    sd = np.nanstd(train_x, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return tuple(np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for x in (train_x, *others))


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray, target_names: list[str]) -> dict[str, Any]:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    y_mu = train_y.mean(axis=0)
    y_sd = train_y.std(axis=0)
    y_sd = np.where(y_sd > 1e-8, y_sd, 1.0)
    train_yt = ((train_y - y_mu) / y_sd).astype(np.float32)
    valid_yt = ((valid_y - y_mu) / y_sd).astype(np.float32)
    model = OpportunityMLP(train_x.shape[1], train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_yt).to(device)
    vx = torch.from_numpy(valid_x).to(device)
    best_state = None
    best_score = -1e18
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(tx)
        loss = loss_fn(pred, ty)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            valid_pred = model(vx).detach().cpu().numpy() * y_sd + y_mu
        score = 0.55 * rank_ic(valid_pred[:, target_names.index("pool_mean")], valid_y[:, target_names.index("pool_mean")])
        score += 0.30 * rank_ic(valid_pred[:, target_names.index("candidate_best_top20")], valid_y[:, target_names.index("candidate_best_top20")])
        score += 0.15 * rank_ic(valid_pred[:, target_names.index("oracle_top50")], valid_y[:, target_names.index("oracle_top50")])
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "valid_selection_score": score})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"model": model, "device": device, "y_mu": y_mu, "y_sd": y_sd, "history": history, "best_epoch": max(history, key=lambda row: row["valid_selection_score"])["epoch"], "best_valid_score": best_score}


def predict(model_info: dict[str, Any], x: np.ndarray) -> np.ndarray:
    model = model_info["model"]
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(model_info["device"])).detach().cpu().numpy()
    return pred * model_info["y_sd"] + model_info["y_mu"]


def aggregate_values(rows: list[dict[str, Any]], indices: list[int], pred: np.ndarray | None, target_names: list[str]) -> dict[str, Any]:
    if not indices:
        return {"sessions": 0}
    out: dict[str, Any] = {"sessions": len(indices)}
    for name in ["pool_mean", "oracle_top20", "oracle_top50", "tail_spread20", "candidate_best_top20"]:
        out[name] = mean(rows[i]["labels"][name] for i in indices)
    out["candidate_top20"] = {name: mean(rows[i]["candidate_top20"][name] for i in indices) for name in CANDIDATES}
    if pred is not None:
        out["predicted"] = {name: mean(float(pred[i, target_names.index(name)]) for i in indices) for name in target_names}
    years: dict[str, list[int]] = {}
    for i in indices:
        years.setdefault(rows[i]["year"], []).append(i)
    out["by_year"] = {year: aggregate_values(rows, part, pred, target_names) for year, part in sorted(years.items()) if len(part) != len(indices)}
    return out


def split_metrics(rows: list[dict[str, Any]], pred: np.ndarray, target_names: list[str]) -> dict[str, Any]:
    actual = {name: np.asarray([row["labels"][name] for row in rows], dtype=float) for name in target_names}
    pred_map = {name: pred[:, i] for i, name in enumerate(target_names)}
    metrics = {name: {"rank_ic": rank_ic(pred_map[name], actual[name]), "pearson": float(np.corrcoef(pred_map[name], actual[name])[0, 1]) if len(rows) > 3 and np.std(pred_map[name]) > 1e-12 and np.std(actual[name]) > 1e-12 else 0.0} for name in target_names}
    opportunity_score = 0.55 * pred_map["pool_mean"] + 0.30 * pred_map["candidate_best_top20"] + 0.15 * pred_map["oracle_top50"]
    order = np.argsort(-opportunity_score, kind="mergesort")
    n = len(rows)
    buckets = {
        "top30pct": order[: max(1, int(round(n * 0.30)))].tolist(),
        "mid40pct": order[max(1, int(round(n * 0.30))): max(1, int(round(n * 0.70)))].tolist(),
        "bottom30pct": order[max(1, int(round(n * 0.70))):].tolist(),
    }
    return {
        "target_metrics": metrics,
        "opportunity_score_rank_ic_pool_mean": rank_ic(opportunity_score, actual["pool_mean"]),
        "opportunity_score_rank_ic_candidate_best": rank_ic(opportunity_score, actual["candidate_best_top20"]),
        "buckets": {name: aggregate_values(rows, indices, pred, target_names) for name, indices in buckets.items()},
        "all": aggregate_values(rows, list(range(len(rows))), pred, target_names),
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    train_rows = build_rows(market, amount20_arr, TRAIN_START, TRAIN_END)
    valid_rows = build_rows(market, amount20_arr, VALID_START, VALID_END)
    dev_rows = build_rows(market, amount20_arr, DEV_START, DEV_END)
    fwd_rows = build_rows(market, amount20_arr, FWD_START, FWD_END)
    feature_names = sorted(train_rows[0]["x_dict"].keys())
    target_names = ["pool_mean", "oracle_top50", "candidate_best_top20", "oracle_top20", "tail_spread20"]
    train_x, train_y = rows_to_arrays(train_rows, feature_names, target_names)
    valid_x, valid_y = rows_to_arrays(valid_rows, feature_names, target_names)
    dev_x, dev_y = rows_to_arrays(dev_rows, feature_names, target_names)
    fwd_x, fwd_y = rows_to_arrays(fwd_rows, feature_names, target_names)
    train_xs, valid_xs, dev_xs, fwd_xs = standardize(train_x, valid_x, dev_x, fwd_x)
    model_info = train_model(train_xs, train_y, valid_xs, valid_y, target_names)
    preds = {
        "train": predict(model_info, train_xs),
        "valid": predict(model_info, valid_xs),
        "dev": predict(model_info, dev_xs),
        "forward": predict(model_info, fwd_xs),
    }
    split_rows = {"train": train_rows, "valid": valid_rows, "dev": dev_rows, "forward": fwd_rows}
    results = {name: split_metrics(rows, preds[name], target_names) for name, rows in split_rows.items()}
    best_fwd_candidate = max(CANDIDATES, key=lambda name: results["forward"]["buckets"]["top30pct"]["candidate_top20"][name])
    best = {
        "forward_top30_candidate": best_fwd_candidate,
        "forward_top30_candidate_top20": results["forward"]["buckets"]["top30pct"]["candidate_top20"][best_fwd_candidate],
        "forward_all_candidate_top20": results["forward"]["all"]["candidate_top20"][best_fwd_candidate],
        "dev_top30_candidate_top20": results["dev"]["buckets"]["top30pct"]["candidate_top20"][best_fwd_candidate],
        "dev_all_candidate_top20": results["dev"]["all"]["candidate_top20"][best_fwd_candidate],
        "forward_top30_sessions": results["forward"]["buckets"]["top30pct"]["sessions"],
    }
    verdict = "opportunity_first_world_model_not_enough"
    if best["dev_top30_candidate_top20"] > best["dev_all_candidate_top20"] + 0.004 and best["forward_top30_candidate_top20"] > best["forward_all_candidate_top20"] + 0.004 and results["forward"]["opportunity_score_rank_ic_candidate_best"] > 0.15:
        verdict = "opportunity_first_world_model_candidate_needs_gated_replay"
    out = {
        "experiment": "cross_sectional_opportunity_first_world_model_v1",
        "method": "session_level_tiny_mlp_predict_future_pool_opportunity_before_stock_ranking",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_k": TOP_K,
            "interval": INTERVAL,
            "epochs": EPOCHS,
            "hidden": HIDDEN,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "feature_count": len(feature_names),
            "target_names": target_names,
            "candidate_names": CANDIDATES,
            "device": str(model_info["device"]),
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round52_script": sha256(ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py"),
            "round54_summary": sha256(ROOT / "cross_sectional_sequence_pretraining_feature_ablation_v1_summary.json"),
        },
        "sample_counts": {name: len(rows) for name, rows in split_rows.items()},
        "training_history": model_info["history"],
        "best_epoch": model_info["best_epoch"],
        "best_valid_score": model_info["best_valid_score"],
        "feature_names": feature_names,
        "results": results,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "best_epoch": model_info["best_epoch"],
        "best": best,
        "sample_counts": out["sample_counts"],
        "forward_metrics": results["forward"]["target_metrics"],
        "forward_buckets": results["forward"]["buckets"],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
