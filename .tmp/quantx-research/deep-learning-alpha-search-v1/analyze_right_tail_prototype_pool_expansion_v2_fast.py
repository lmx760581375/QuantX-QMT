from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
V2_PATH = ROOT / "analyze_right_tail_prototype_pool_expansion_v2.py"
OUT = ROOT / "right_tail_prototype_pool_expansion_v2_fast_summary.json"

EXPAND_POOL_SIZES = [200, 500]
CONFIGS = [
    {"use_group": True, "winner_n": 10, "loser_n": 10, "k": 16, "mode": "pure", "weight": -1.0},
    {"use_group": True, "winner_n": 10, "loser_n": 20, "k": 16, "mode": "pure", "weight": -1.0},
    {"use_group": True, "winner_n": 10, "loser_n": 10, "k": 16, "mode": "anchor", "weight": 0.2},
    {"use_group": True, "winner_n": 10, "loser_n": 10, "k": 16, "mode": "anchor", "weight": 0.4},
    {"use_group": False, "winner_n": 10, "loser_n": 10, "k": 16, "mode": "anchor", "weight": 0.4},
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_module("right_tail_pool_expansion_v2_for_fast", V2_PATH)
V1 = V2.V1
G60 = V2.G60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def run_config(splits: dict[str, list[dict[str, Any]]], cfg: dict[str, Any]) -> dict[str, Any]:
    scorer = V1.build_scorer(splits["train"], bool(cfg["use_group"]), int(cfg["winner_n"]), int(cfg["loser_n"]), int(cfg["k"]))
    evals = {name: V1.evaluate(items, scorer, str(cfg["mode"]), float(cfg["weight"])) for name, items in splits.items()}
    valid = evals["valid"]
    score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.001 * valid["mean_rank_ic"] + 0.0005 * valid["outside_base_top60_share"]["top20"]
    return {
        "config": cfg,
        "valid_score": score,
        "tail_counts": scorer["tail_counts"],
        "train": compact(evals["train"]),
        "valid": compact(evals["valid"]),
        "dev": compact(evals["dev"]),
        "forward": compact(evals["forward"]),
    }


def main() -> None:
    started = time.perf_counter()
    G60.set_seed(V2.SEED)
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    by_expand = {}
    all_rows = []
    for expand_size in EXPAND_POOL_SIZES:
        splits = {
            "train": V2.build_split_flexible(market, amount20_arr, groups, G60.TRAIN_START, G60.TRAIN_END, expand_size),
            "valid": V2.build_split_flexible(market, amount20_arr, groups, G60.VALID_START, G60.VALID_END, expand_size),
            "dev": V2.build_split_flexible(market, amount20_arr, groups, G60.DEV_START, G60.DEV_END, expand_size),
            "forward": V2.build_split_flexible(market, amount20_arr, groups, G60.FWD_START, G60.FWD_END, expand_size),
        }
        rows = []
        for cfg in CONFIGS:
            row = {"expand_size": expand_size, **run_config(splits, cfg)}
            rows.append(row)
            all_rows.append(row)
        rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
        by_expand[str(expand_size)] = {
            "sample_counts": {name: len(items) for name, items in splits.items()},
            "size_stats": {name: V2.split_size_stats(items) for name, items in splits.items()},
            "leaderboard": rows,
            "selected": rows[0],
        }
    selected = max(all_rows, key=lambda r: r["valid_score"])
    verdict = "right_tail_prototype_pool_expansion_v2_fast_not_enough"
    dev_years = selected["dev"]["by_year_top20"]
    if (
        selected["forward"]["top20"] > selected["forward"]["base_top20"] + 0.003
        and selected["dev"]["top20"] > selected["dev"]["base_top20"]
        and min(dev_years.values()) > 0.0
        and selected["forward"]["outside20"] > 0.10
    ):
        verdict = "right_tail_prototype_pool_expansion_v2_fast_candidate_needs_replay"
    out = {
        "experiment": "right_tail_prototype_pool_expansion_v2_fast",
        "method": "flexible_pool_expansion_min120_fast_diagnostic_subset",
        "params": {"expand_pool_sizes": EXPAND_POOL_SIZES, "configs": CONFIGS, "min_effective": V2.MIN_EFFECTIVE},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round64_full_script": sha256(V2_PATH), "round63_summary": sha256(ROOT / "right_tail_prototype_pool_expansion_v1_summary.json")},
        "by_expand": by_expand,
        "selected": selected,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": {k: v["sample_counts"] for k, v in by_expand.items()}, "size_stats": {k: v["size_stats"] for k, v in by_expand.items()}, "selected": selected, "leaderboards": {k: v["leaderboard"] for k, v in by_expand.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
