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
V1_PATH = ROOT / "analyze_right_tail_prototype_pool_expansion_v1.py"
OUT = ROOT / "right_tail_prototype_pool_expansion_v2_summary.json"

EXPAND_POOL_SIZES = [120, 200, 500]
MIN_EFFECTIVE = 120
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("right_tail_pool_expansion_v1_for_v2", V1_PATH)
G60 = V1.G60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_table_flexible(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], session: str, expand_size: int) -> dict[str, Any] | None:
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
    base_score_all = G60.CA52.candidate_scores(frame)["mid_trend_volume_not_extreme"]
    base_full_order = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score_all[j]), str(market["symbols"][kept[j]]))), dtype=int)
    effective_size = min(expand_size, len(base_full_order))
    domain_order = base_full_order[:effective_size]
    if len(domain_order) < MIN_EFFECTIVE:
        return None

    base_top60_cols = set(int(kept[j]) for j in base_full_order[:V1.BASE_DOMAIN_SIZE])
    domain_cols = kept[domain_order]
    feature_names = sorted(frame.keys())
    stock_x = np.vstack([G60.finite_rank(np.asarray(frame[name], dtype=float))[domain_order] for name in feature_names]).T
    group_x = G60.group_feature_matrix(market, idx, kept, domain_cols, groups)
    y = labels_arr[domain_order].astype(np.float32)
    outside_base_top60 = np.asarray([0.0 if int(col) in base_top60_cols else 1.0 for col in domain_cols], dtype=np.float32)
    return {
        "date": session,
        "year": session[:4],
        "stock_x": np.nan_to_num(stock_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "group_x": group_x,
        "y": y,
        "base_order": np.arange(len(y)),
        "outside_base_top60": outside_base_top60,
        "feature_names": feature_names,
        "expand_size": expand_size,
        "effective_size": int(effective_size),
        "valid_label_count": int(len(labels_arr)),
    }


def build_split_flexible(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], start: str, end: str, expand_size: int) -> list[dict[str, Any]]:
    rows = []
    for session in G60.CA52.CVR.sessions(market, start, end, G60.INTERVAL):
        item = session_table_flexible(market, amount20_arr, groups, session, expand_size)
        if item is not None:
            rows.append(item)
    return rows


def split_size_stats(items: list[dict[str, Any]]) -> dict[str, float]:
    sizes = np.asarray([item["effective_size"] for item in items], dtype=float)
    labels = np.asarray([item["valid_label_count"] for item in items], dtype=float)
    if len(sizes) == 0:
        return {"sessions": 0, "effective_min": 0, "effective_mean": 0, "effective_max": 0, "label_mean": 0}
    return {
        "sessions": int(len(items)),
        "effective_min": int(np.min(sizes)),
        "effective_mean": float(np.mean(sizes)),
        "effective_max": int(np.max(sizes)),
        "label_mean": float(np.mean(labels)),
    }


def main() -> None:
    started = time.perf_counter()
    G60.set_seed(SEED)
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    by_expand = {}
    for expand_size in EXPAND_POOL_SIZES:
        splits = {
            "train": build_split_flexible(market, amount20_arr, groups, G60.TRAIN_START, G60.TRAIN_END, expand_size),
            "valid": build_split_flexible(market, amount20_arr, groups, G60.VALID_START, G60.VALID_END, expand_size),
            "dev": build_split_flexible(market, amount20_arr, groups, G60.DEV_START, G60.DEV_END, expand_size),
            "forward": build_split_flexible(market, amount20_arr, groups, G60.FWD_START, G60.FWD_END, expand_size),
        }
        by_expand[str(expand_size)] = {
            "sample_counts": {name: len(items) for name, items in splits.items()},
            "size_stats": {name: split_size_stats(items) for name, items in splits.items()},
            **V1.run_for_expand(splits, expand_size),
        }

    selected = max((item["selected"] for item in by_expand.values()), key=lambda payload: payload["selection"]["valid_score"])
    selected_evals = selected["evals"]
    verdict = "right_tail_prototype_pool_expansion_v2_not_enough"
    dev_years = selected_evals["dev"]["by_year_model_top20"]
    if (
        selected_evals["forward"]["model_top"]["top20"] > selected_evals["forward"]["base_top"]["top20"] + 0.003
        and selected_evals["dev"]["model_top"]["top20"] > selected_evals["dev"]["base_top"]["top20"]
        and min(dev_years.values()) > 0.0
        and selected_evals["forward"]["outside_base_top60_share"]["top20"] > 0.10
    ):
        verdict = "right_tail_prototype_pool_expansion_v2_candidate_needs_replay"

    out = {
        "experiment": "right_tail_prototype_pool_expansion_v2",
        "method": "prototype_pool_expansion_with_flexible_effective_pool_size_min120",
        "params": {"expand_pool_sizes": EXPAND_POOL_SIZES, "min_effective": MIN_EFFECTIVE, "base_domain_size": V1.BASE_DOMAIN_SIZE, "k_values": V1.K_VALUES, "tail_specs": V1.TAIL_SPECS, "weights": V1.WEIGHTS, "interval": G60.INTERVAL, "pool_size": G60.POOL_SIZE},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round63_script": sha256(V1_PATH), "round63_summary": sha256(ROOT / "right_tail_prototype_pool_expansion_v1_summary.json")},
        "group_coverage": {key: value for key, value in groups.items() if key not in {"industry", "concept"}},
        "by_expand": by_expand,
        "selected": selected,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "selected_meta": selected["scorer_meta"],
        "selected_config": selected["selection"],
        "selected_results": selected["evals"],
        "sample_counts": {k: v["sample_counts"] for k, v in by_expand.items()},
        "size_stats": {k: v["size_stats"] for k, v in by_expand.items()},
        "leaderboards": {k: v["leaderboard_top12"][:5] for k, v in by_expand.items()},
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
