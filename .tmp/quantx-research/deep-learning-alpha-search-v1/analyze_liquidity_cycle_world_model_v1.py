from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
G60_PATH = ROOT / "train_group_diffusion_world_model_v1.py"
OUT = ROOT / "liquidity_cycle_world_model_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
DOMAIN_SIZE = 60
TOP_KS = [10, 20]
WEIGHTS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_liquidity_cycle", G60_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def window(arr: np.ndarray, end_idx: int, cols: np.ndarray, length: int) -> np.ndarray:
    start = max(0, end_idx - length + 1)
    return arr[start:end_idx + 1, :][:, cols].astype(float)


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    return G60.rank_ic(a, b)


def rolling_mean_2d(arr: np.ndarray, idx: int, cols: np.ndarray, length: int) -> np.ndarray:
    return np.nanmean(window(arr, idx, cols, length), axis=0)


def group_liquidity_features(amount: np.ndarray, close: np.ndarray, idx: int, pool_cols: np.ndarray, domain_cols: np.ndarray, groups: dict[str, Any]) -> np.ndarray:
    pool_pos = {int(col): pos for pos, col in enumerate(pool_cols)}
    amt5 = rolling_mean_2d(amount, idx, pool_cols, 5)
    amt20 = rolling_mean_2d(amount, idx, pool_cols, 20)
    amt60 = rolling_mean_2d(amount, idx, pool_cols, 60)
    amt120 = rolling_mean_2d(amount, idx, pool_cols, 120)
    c0 = close[idx, pool_cols]
    c20 = close[idx - 20, pool_cols] if idx >= 20 else np.full(len(pool_cols), np.nan)
    c60 = close[idx - 60, pool_cols] if idx >= 60 else np.full(len(pool_cols), np.nan)
    ret20 = safe_div(c0, c20) - 1.0
    ret60 = safe_div(c0, c60) - 1.0

    total_amt20 = np.nansum(amt20)
    industry = groups["industry"]
    concept = groups["concept"]
    rows = []
    for col in domain_cols:
        pos = pool_pos[int(col)]
        masks = []
        ind = int(industry[int(col)])
        if ind >= 0:
            masks.append(industry[pool_cols] == ind)
        if concept.shape[1]:
            cids = np.flatnonzero(concept[int(col)])[:16]
            for cid in cids:
                masks.append(concept[pool_cols, cid])

        stats = []
        for mask in masks:
            if int(np.sum(mask)) < 3:
                continue
            ratio20_120 = safe_div(amt20[mask], amt120[mask])
            ratio5_20 = safe_div(amt5[mask], amt20[mask])
            group_amt20 = np.nansum(amt20[mask])
            group_amt120 = np.nansum(amt120[mask])
            stats.append([
                float(group_amt20 / total_amt20) if total_amt20 > 0 else 0.0,
                float(group_amt20 / group_amt120) if np.isfinite(group_amt120) and group_amt120 > 0 else 0.0,
                safe_mean(ratio20_120),
                safe_mean(ratio5_20),
                float(np.nanmean(ratio20_120 > 1.2)),
                float(np.nanmean(ratio20_120 > 1.5)),
                safe_mean(ret20[mask]),
                safe_mean(ret60[mask]),
                float(np.nanmean(ret20[mask] > 0.0)),
                float(np.nanmean(ret60[mask] > 0.0)),
                float(np.nanstd(ret20[mask])),
            ])
        if stats:
            arr = np.asarray(stats, dtype=float)
            best = arr[np.nanargmax(arr[:, 1])]
            mean = np.nanmean(arr, axis=0)
            group_values = list(best) + [float(mean[1]), float(mean[4]), float(mean[6]), float(len(stats) / 17.0)]
        else:
            group_values = [0.0] * 15

        self_amt20_120 = float(amt20[pos] / amt120[pos]) if np.isfinite(amt20[pos]) and np.isfinite(amt120[pos]) and amt120[pos] > 0 else 0.0
        self_amt5_20 = float(amt5[pos] / amt20[pos]) if np.isfinite(amt5[pos]) and np.isfinite(amt20[pos]) and amt20[pos] > 0 else 0.0
        self_share20 = float(amt20[pos] / total_amt20) if total_amt20 > 0 and np.isfinite(amt20[pos]) else 0.0
        rows.append([self_amt20_120, self_amt5_20, self_share20, float(ret20[pos]) if np.isfinite(ret20[pos]) else 0.0, float(ret60[pos]) if np.isfinite(ret60[pos]) else 0.0] + group_values)
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def session_table(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], session: str) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    amount = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close) * volume
    amount = np.where(np.isfinite(amount) & (amount > 0), amount, np.nan)
    pool_cols = np.asarray(G60.RT.pool_for_session(market, amount20_arr, idx)[:G60.POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in pool_cols:
        y = G60.CA52.future_return(market, idx, int(col), G60.INTERVAL)
        if y is not None:
            labels.append(float(y))
            kept_cols.append(int(col))
    if len(labels) < DOMAIN_SIZE:
        return None
    kept = np.asarray(kept_cols, dtype=int)
    labels_arr = np.asarray(labels, dtype=float)
    frame = G60.CA52.feature_frame(market, idx, kept)
    base_score = G60.CA52.candidate_scores(frame)["mid_trend_volume_not_extreme"]
    order = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score[j]), str(market["symbols"][kept[j]])))[:DOMAIN_SIZE], dtype=int)
    domain_cols = kept[order]
    liq = group_liquidity_features(amount, close, idx, kept, domain_cols, groups)
    y = labels_arr[order].astype(np.float32)
    return {"date": session, "year": session[:4], "liquidity_x": liq, "y": y, "base_order": np.arange(len(y))}


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
    rows = []
    for session in G60.CA52.CVR.sessions(market, start, end, G60.INTERVAL):
        item = session_table(market, amount20_arr, groups, session)
        if item is not None:
            rows.append(item)
    return rows


def train_linear(train: list[dict[str, Any]], valid: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.vstack([item["liquidity_x"] for item in train]).astype(float)
    y = np.concatenate([G60.finite_rank(item["y"]) for item in train]).astype(float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    xs = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    xtx = xs.T @ xs + np.eye(xs.shape[1]) * 5.0
    coef = np.linalg.solve(xtx, xs.T @ (y - 0.5))
    return {"mu": mu, "sd": sd, "coef": coef}


def score_item(item: dict[str, Any], model: dict[str, Any], mode: str, weight: float) -> np.ndarray:
    residual = np.nan_to_num((item["liquidity_x"] - model["mu"]) / model["sd"], nan=0.0, posinf=0.0, neginf=0.0) @ model["coef"]
    residual = G60.finite_rank(residual) - 0.5
    if mode == "pure":
        return residual
    return anchor_score(len(residual)) + weight * residual


def evaluate(sessions: list[dict[str, Any]], model: dict[str, Any], mode: str, weight: float) -> dict[str, Any]:
    top = {k: [] for k in TOP_KS}
    base = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        score = score_item(item, model, mode, weight)
        order = np.argsort(-score, kind="mergesort")
        y = item["y"]
        ics.append(rank_ic(score, y))
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[item["base_order"][:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {"sessions": len(sessions), "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()}, "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()}, "mean_rank_ic": safe_mean(ics), "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())}}


def main() -> None:
    started = time.perf_counter()
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": build_split(market, amount20_arr, groups, TRAIN_START, TRAIN_END),
        "valid": build_split(market, amount20_arr, groups, VALID_START, VALID_END),
        "dev": build_split(market, amount20_arr, groups, DEV_START, DEV_END),
        "forward": build_split(market, amount20_arr, groups, FWD_START, FWD_END),
    }
    model = train_linear(splits["train"], splits["valid"])
    rows = []
    for mode, weight in [("pure", -1.0)] + [("anchor", w) for w in WEIGHTS]:
        evals = {name: evaluate(items, model, mode, weight) for name, items in splits.items()}
        valid = evals["valid"]
        score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"]
        rows.append({"mode": mode, "weight": weight, "valid_score": score, "evals": evals})
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    verdict = "liquidity_cycle_world_model_not_enough"
    dev_years = selected["evals"]["dev"]["by_year_model_top20"]
    if selected["evals"]["forward"]["model_top"]["top20"] > selected["evals"]["forward"]["base_top"]["top20"] + 0.003 and selected["evals"]["dev"]["model_top"]["top20"] > selected["evals"]["dev"]["base_top"]["top20"] and min(dev_years.values()) > 0.0:
        verdict = "liquidity_cycle_world_model_candidate_needs_replay"
    out = {"experiment": "liquidity_cycle_world_model_v1", "method": "ridge_linear_liquidity_cycle_features_inside_mid_trend_volume_top60", "params": {"weights": WEIGHTS, "domain_size": DOMAIN_SIZE, "train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "forward": [FWD_START, FWD_END]}, "inputs_sha256": {"script": sha256(Path(__file__)), "round64_summary": sha256(ROOT / "right_tail_prototype_pool_expansion_v2_stock_only_summary.json")}, "sample_counts": {name: len(items) for name, items in splits.items()}, "leaderboard": [{"mode": r["mode"], "weight": r["weight"], "valid_score": r["valid_score"], "valid": r["evals"]["valid"], "dev": r["evals"]["dev"], "forward": r["evals"]["forward"]} for r in rows], "selected": selected, "verdict": verdict, "elapsed_seconds": time.perf_counter() - started}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "leaderboard_top": out["leaderboard"][:5], "selected": {"mode": selected["mode"], "weight": selected["weight"], "evals": selected["evals"]}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
