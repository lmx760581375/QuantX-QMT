from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "group_residual_anchor_world_model_v1_summary.json"
G60_PATH = ROOT / "train_group_diffusion_world_model_v1.py"

WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
TOP_KS = [10, 20]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_residual_anchor", G60_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def rank01(values: np.ndarray) -> np.ndarray:
    return G60.finite_rank(np.asarray(values, dtype=float))


def train_residual_model(splits: dict[str, list[dict[str, Any]]], use_group: bool) -> dict[str, Any]:
    train_x, train_y = G60.stack_rows(splits["train"], use_group)
    valid_x, valid_y = G60.stack_rows(splits["valid"], use_group)
    mu, sd = G60.standard_params(train_x)
    train_xs = ((train_x - mu) / sd).astype(np.float32)
    valid_xs = ((valid_x - mu) / sd).astype(np.float32)
    model_info = G60.train_model(train_xs, train_y, valid_xs, valid_y)
    return {"model_info": model_info, "mu": mu, "sd": sd, "use_group": use_group, "valid_y": valid_y}


def score_residual(item: dict[str, Any], trained: dict[str, Any]) -> np.ndarray:
    raw = G60.score_item(item, trained["model_info"], trained["mu"], trained["sd"], bool(trained["use_group"]))
    return rank01(raw) - 0.5


def score_anchor(n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray([0.5], dtype=float)
    return np.linspace(1.0, 0.0, n, endpoint=True)


def evaluate_weight(sessions: list[dict[str, Any]], trained: dict[str, Any], weight: float) -> dict[str, Any]:
    top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    changed = []
    for item in sessions:
        y = item["y"]
        base = score_anchor(len(y))
        residual = score_residual(item, trained)
        score = base + weight * residual
        order = np.argsort(-score, kind="mergesort")
        base_order = item["base_order"]
        ics.append(G60.rank_ic(score, y))
        changed.append(1.0 - len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base_top[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(sessions),
        "weight": weight,
        "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "mean_rank_ic": safe_mean(ics),
        "mean_top20_turnover_vs_base": safe_mean(changed),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def evaluate_grid(splits: dict[str, list[dict[str, Any]]], trained: dict[str, Any]) -> dict[str, Any]:
    grid = {str(w): {split: evaluate_weight(items, trained, w) for split, items in splits.items()} for w in WEIGHTS}
    valid_rows = []
    for w in WEIGHTS:
        item = grid[str(w)]["valid"]
        # Valid chooses the residual only if it improves Top20 without destroying Top10 too much.
        valid_rows.append({
            "weight": w,
            "valid_top20": item["model_top"]["top20"],
            "valid_top10": item["model_top"]["top10"],
            "valid_rank_ic": item["mean_rank_ic"],
            "turnover": item["mean_top20_turnover_vs_base"],
            "score": item["model_top"]["top20"] + 0.35 * item["model_top"]["top10"] - 0.002 * item["mean_top20_turnover_vs_base"],
        })
    best = max(valid_rows, key=lambda row: (row["score"], -row["turnover"], -row["weight"]))
    return {"grid": grid, "valid_selection": best, "selected": grid[str(best["weight"])]}


def main() -> None:
    started = time.perf_counter()
    G60.set_seed(G60.SEED)
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": G60.build_split(market, amount20_arr, groups, G60.TRAIN_START, G60.TRAIN_END),
        "valid": G60.build_split(market, amount20_arr, groups, G60.VALID_START, G60.VALID_END),
        "dev": G60.build_split(market, amount20_arr, groups, G60.DEV_START, G60.DEV_END),
        "forward": G60.build_split(market, amount20_arr, groups, G60.FWD_START, G60.FWD_END),
    }
    stock_only = train_residual_model(splits, False)
    stock_plus_group = train_residual_model(splits, True)
    variants = {
        "stock_only_residual": evaluate_grid(splits, stock_only),
        "stock_plus_group_residual": evaluate_grid(splits, stock_plus_group),
    }
    selected = variants["stock_plus_group_residual"]["selected"]
    verdict = "group_residual_anchor_world_model_not_enough"
    dev_years = selected["dev"]["by_year_model_top20"]
    if (
        selected["forward"]["model_top"]["top20"] > selected["forward"]["base_top"]["top20"] + 0.002
        and selected["dev"]["model_top"]["top20"] > selected["dev"]["base_top"]["top20"]
        and min(dev_years.values()) > 0.0
        and selected["forward"]["mean_top20_turnover_vs_base"] < 0.35
    ):
        verdict = "group_residual_anchor_world_model_candidate_needs_replay"
    out = {
        "experiment": "group_residual_anchor_world_model_v1",
        "method": "base_candidate_rank_anchor_plus_valid_selected_tiny_model_residual",
        "params": {"weights": WEIGHTS, "interval": G60.INTERVAL, "pool_size": G60.POOL_SIZE, "domain_size": G60.DOMAIN_SIZE, "train": [G60.TRAIN_START, G60.TRAIN_END], "valid": [G60.VALID_START, G60.VALID_END], "forward": [G60.FWD_START, G60.FWD_END]},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round60_script": sha256(G60_PATH), "round60_summary": sha256(ROOT / "group_diffusion_world_model_v1_summary.json")},
        "group_coverage": {key: value for key, value in groups.items() if key not in {"industry", "concept"}},
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "model_training": {
            "stock_only": {"best_epoch": stock_only["model_info"]["best_epoch"], "best_valid_row_rank_ic": stock_only["model_info"]["best_valid_row_rank_ic"]},
            "stock_plus_group": {"best_epoch": stock_plus_group["model_info"]["best_epoch"], "best_valid_row_rank_ic": stock_plus_group["model_info"]["best_valid_row_rank_ic"]},
        },
        "variants": variants,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "model_training": out["model_training"],
        "selection": {name: value["valid_selection"] for name, value in variants.items()},
        "selected_results": {name: value["selected"] for name, value in variants.items()},
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
