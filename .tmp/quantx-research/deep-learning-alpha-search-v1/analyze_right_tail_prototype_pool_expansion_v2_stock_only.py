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
G60_PATH = ROOT / "train_group_diffusion_world_model_v1.py"
R62_PATH = ROOT / "analyze_right_tail_prototype_world_model_v1.py"
OUT = ROOT / "right_tail_prototype_pool_expansion_v2_stock_only_summary.json"

EXPAND_POOL_SIZES = [200, 500]
MIN_EFFECTIVE = 120
CONFIGS = [
    {"winner_n": 10, "loser_n": 10, "k": 16, "mode": "pure", "weight": -1.0},
    {"winner_n": 10, "loser_n": 20, "k": 16, "mode": "pure", "weight": -1.0},
    {"winner_n": 10, "loser_n": 10, "k": 16, "mode": "anchor", "weight": 0.2},
    {"winner_n": 10, "loser_n": 10, "k": 16, "mode": "anchor", "weight": 0.4},
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_pool_expansion_stock_only", G60_PATH)
R62 = load_module("right_tail_prototype_for_pool_expansion_stock_only", R62_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def session_table(market: dict[str, Any], amount20_arr: np.ndarray, session: str, expand_size: int) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    pool_cols = np.asarray(G60.RT.pool_for_session(market, amount20_arr, idx)[:G60.POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in pool_cols:
        y = G60.CA52.future_return(market, idx, int(col), G60.INTERVAL)
        if y is not None:
            labels.append(float(y))
            kept_cols.append(int(col))
    if len(labels) < MIN_EFFECTIVE:
        return None
    kept = np.asarray(kept_cols, dtype=int)
    labels_arr = np.asarray(labels, dtype=float)
    frame = G60.CA52.feature_frame(market, idx, kept)
    base_score = G60.CA52.candidate_scores(frame)["mid_trend_volume_not_extreme"]
    base_order_all = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score[j]), str(market["symbols"][kept[j]]))), dtype=int)
    effective_size = min(expand_size, len(base_order_all))
    order = base_order_all[:effective_size]
    if len(order) < MIN_EFFECTIVE:
        return None
    base_top60_cols = set(int(kept[j]) for j in base_order_all[:60])
    domain_cols = kept[order]
    feature_names = sorted(frame.keys())
    stock_x = np.vstack([G60.finite_rank(np.asarray(frame[name], dtype=float))[order] for name in feature_names]).T
    y = labels_arr[order].astype(np.float32)
    outside = np.asarray([0.0 if int(col) in base_top60_cols else 1.0 for col in domain_cols], dtype=np.float32)
    return {
        "date": session,
        "year": session[:4],
        "stock_x": np.nan_to_num(stock_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "group_x": np.zeros((len(y), 0), dtype=np.float32),
        "y": y,
        "base_order": np.arange(len(y)),
        "outside_base_top60": outside,
        "feature_names": feature_names,
        "expand_size": expand_size,
        "effective_size": int(effective_size),
        "valid_label_count": int(len(labels_arr)),
    }


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str, expand_size: int) -> list[dict[str, Any]]:
    rows = []
    for session in G60.CA52.CVR.sessions(market, start, end, G60.INTERVAL):
        item = session_table(market, amount20_arr, session, expand_size)
        if item is not None:
            rows.append(item)
    return rows


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def prototype_score(item: dict[str, Any], scorer: dict[str, Any]) -> np.ndarray:
    return R62.prototype_score(item, scorer["mu"], scorer["sd"], False, scorer["winner_proto"], scorer["loser_proto"])


def evaluate(sessions: list[dict[str, Any]], scorer: dict[str, Any], mode: str, weight: float) -> dict[str, Any]:
    top = {10: [], 20: [], 30: []}
    base = {10: [], 20: [], 30: []}
    outside = {10: [], 20: [], 30: []}
    ics = []
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        y = item["y"]
        residual = prototype_score(item, scorer)
        score = residual if mode == "pure" else anchor_score(len(y)) + weight * residual
        order = np.argsort(-score, kind="mergesort")
        ics.append(G60.rank_ic(score, y))
        for k in [10, 20, 30]:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[item["base_order"][:k]]))
            outside[k].append(safe_mean(item["outside_base_top60"][order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()},
        "outside_base_top60_share": {f"top{k}": safe_mean(v) for k, v in outside.items()},
        "mean_rank_ic": safe_mean(ics),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
        "sessions": len(sessions),
    }


def build_scorer(train_sessions: list[dict[str, Any]], winner_n: int, loser_n: int, k: int) -> dict[str, Any]:
    mu, sd = R62.standard_params_from_sessions(train_sessions, False)
    winners, losers = R62.collect_tails(train_sessions, mu, sd, False, winner_n, loser_n)
    return {"mu": mu, "sd": sd, "winner_proto": R62.kmeans(winners, k), "loser_proto": R62.kmeans(losers, k), "tail_counts": {"winners": int(len(winners)), "losers": int(len(losers))}}


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "top10": result["model_top"]["top10"],
        "top20": result["model_top"]["top20"],
        "top30": result["model_top"]["top30"],
        "base_top20": result["base_top"]["top20"],
        "rank_ic": result["mean_rank_ic"],
        "outside20": result["outside_base_top60_share"]["top20"],
        "by_year_top20": result["by_year_model_top20"],
    }


def size_stats(items: list[dict[str, Any]]) -> dict[str, float]:
    sizes = np.asarray([item["effective_size"] for item in items], dtype=float)
    labels = np.asarray([item["valid_label_count"] for item in items], dtype=float)
    return {"sessions": int(len(items)), "effective_min": int(np.min(sizes)), "effective_mean": float(np.mean(sizes)), "effective_max": int(np.max(sizes)), "label_mean": float(np.mean(labels))} if len(items) else {"sessions": 0}


def main() -> None:
    started = time.perf_counter()
    market = G60.POS.load_market()
    amount20_arr = G60.RT.amount20(market)
    by_expand = {}
    all_rows = []
    for expand_size in EXPAND_POOL_SIZES:
        splits = {
            "train": build_split(market, amount20_arr, G60.TRAIN_START, G60.TRAIN_END, expand_size),
            "valid": build_split(market, amount20_arr, G60.VALID_START, G60.VALID_END, expand_size),
            "dev": build_split(market, amount20_arr, G60.DEV_START, G60.DEV_END, expand_size),
            "forward": build_split(market, amount20_arr, G60.FWD_START, G60.FWD_END, expand_size),
        }
        rows = []
        for cfg in CONFIGS:
            scorer = build_scorer(splits["train"], int(cfg["winner_n"]), int(cfg["loser_n"]), int(cfg["k"]))
            evals = {name: evaluate(items, scorer, str(cfg["mode"]), float(cfg["weight"])) for name, items in splits.items()}
            valid = evals["valid"]
            score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.001 * valid["mean_rank_ic"] + 0.0005 * valid["outside_base_top60_share"]["top20"]
            row = {"expand_size": expand_size, "config": cfg, "valid_score": score, "tail_counts": scorer["tail_counts"], "train": compact(evals["train"]), "valid": compact(evals["valid"]), "dev": compact(evals["dev"]), "forward": compact(evals["forward"])}
            rows.append(row)
            all_rows.append(row)
        rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
        by_expand[str(expand_size)] = {"sample_counts": {name: len(items) for name, items in splits.items()}, "size_stats": {name: size_stats(items) for name, items in splits.items()}, "leaderboard": rows, "selected": rows[0]}
    selected = max(all_rows, key=lambda r: r["valid_score"])
    verdict = "right_tail_prototype_pool_expansion_v2_stock_only_not_enough"
    dev_years = selected["dev"]["by_year_top20"]
    if selected["forward"]["top20"] > selected["forward"]["base_top20"] + 0.003 and selected["dev"]["top20"] > selected["dev"]["base_top20"] and min(dev_years.values()) > 0 and selected["forward"]["outside20"] > 0.10:
        verdict = "right_tail_prototype_pool_expansion_v2_stock_only_candidate_needs_replay"
    out = {"experiment": "right_tail_prototype_pool_expansion_v2_stock_only", "method": "flexible_pool_expansion_stock_only_no_group_feature_fast_diagnostic", "params": {"expand_pool_sizes": EXPAND_POOL_SIZES, "min_effective": MIN_EFFECTIVE, "configs": CONFIGS}, "inputs_sha256": {"script": sha256(Path(__file__)), "round62_summary": sha256(ROOT / "right_tail_prototype_world_model_v1_summary.json")}, "by_expand": by_expand, "selected": selected, "verdict": verdict, "elapsed_seconds": time.perf_counter() - started}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": {k: v["sample_counts"] for k, v in by_expand.items()}, "size_stats": {k: v["size_stats"] for k, v in by_expand.items()}, "selected": selected, "leaderboards": {k: v["leaderboard"] for k, v in by_expand.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
