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
OUT = ROOT / "right_tail_prototype_pool_expansion_v1_summary.json"

TOP_KS = [10, 20, 30]
BASE_DOMAIN_SIZE = 60
EXPAND_POOL_SIZES = [120, 200, 500]
K_VALUES = [8, 16]
TAIL_SPECS = [(10, 10), (10, 20)]
WEIGHTS = [0.0, 0.10, 0.20, 0.30, 0.40]
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_pool_expansion", G60_PATH)
R62 = load_module("right_tail_prototype_for_pool_expansion", R62_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def session_table(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], session: str, expand_size: int) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    pool_cols = np.asarray(G60.RT.pool_for_session(market, amount20_arr, idx)[:G60.POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in pool_cols:
        y = G60.CA52.future_return(market, idx, int(col), G60.INTERVAL)
        if y is not None:
            labels.append(float(y))
            kept_cols.append(int(col))
    if len(labels) < expand_size:
        return None
    kept = np.asarray(kept_cols, dtype=int)
    labels_arr = np.asarray(labels, dtype=float)
    frame = G60.CA52.feature_frame(market, idx, kept)
    base_score_all = G60.CA52.candidate_scores(frame)["mid_trend_volume_not_extreme"]
    base_full_order = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score_all[j]), str(market["symbols"][kept[j]]))), dtype=int)
    domain_order = base_full_order[:expand_size]
    base_top60_cols = set(int(kept[j]) for j in base_full_order[:BASE_DOMAIN_SIZE])
    domain_cols = kept[domain_order]
    feature_names = sorted(frame.keys())
    stock_x = np.vstack([G60.finite_rank(np.asarray(frame[name], dtype=float))[domain_order] for name in feature_names]).T
    group_x = G60.group_feature_matrix(market, idx, kept, domain_cols, groups)
    y = labels_arr[domain_order].astype(np.float32)
    base_order = np.arange(len(y))
    outside_base_top60 = np.asarray([0.0 if int(col) in base_top60_cols else 1.0 for col in domain_cols], dtype=np.float32)
    return {
        "date": session,
        "year": session[:4],
        "stock_x": np.nan_to_num(stock_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "group_x": group_x,
        "y": y,
        "base_order": base_order,
        "outside_base_top60": outside_base_top60,
        "feature_names": feature_names,
        "expand_size": expand_size,
    }


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], start: str, end: str, expand_size: int) -> list[dict[str, Any]]:
    rows = []
    for session in G60.CA52.CVR.sessions(market, start, end, G60.INTERVAL):
        item = session_table(market, amount20_arr, groups, session, expand_size)
        if item is not None:
            rows.append(item)
    return rows


def build_scorer(train_sessions: list[dict[str, Any]], use_group: bool, winner_n: int, loser_n: int, k: int) -> dict[str, Any]:
    mu, sd = R62.standard_params_from_sessions(train_sessions, use_group)
    winners, losers = R62.collect_tails(train_sessions, mu, sd, use_group, winner_n, loser_n)
    return {
        "use_group": use_group,
        "winner_n": winner_n,
        "loser_n": loser_n,
        "k": k,
        "mu": mu,
        "sd": sd,
        "winner_proto": R62.kmeans(winners, k),
        "loser_proto": R62.kmeans(losers, k),
        "tail_counts": {"winners": int(len(winners)), "losers": int(len(losers))},
    }


def prototype_residual(item: dict[str, Any], scorer: dict[str, Any]) -> np.ndarray:
    return R62.prototype_score(item, scorer["mu"], scorer["sd"], scorer["use_group"], scorer["winner_proto"], scorer["loser_proto"])


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def evaluate(sessions: list[dict[str, Any]], scorer: dict[str, Any], mode: str, weight: float) -> dict[str, Any]:
    model_top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    outside = {k: [] for k in TOP_KS}
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        y = item["y"]
        residual = prototype_residual(item, scorer)
        if mode == "pure":
            score = residual
        else:
            score = anchor_score(len(y)) + weight * residual
        order = np.argsort(-score, kind="mergesort")
        base_order = item["base_order"]
        ics.append(G60.rank_ic(score, y))
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            model_top[k].append(value)
            base_top[k].append(safe_mean(y[base_order[:k]]))
            outside[k].append(safe_mean(item["outside_base_top60"][order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(sessions),
        "mode": mode,
        "weight": weight,
        "model_top": {f"top{k}": safe_mean(v) for k, v in model_top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "outside_base_top60_share": {f"top{k}": safe_mean(v) for k, v in outside.items()},
        "mean_rank_ic": safe_mean(ics),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "top10": result["model_top"]["top10"],
        "top20": result["model_top"]["top20"],
        "top30": result["model_top"]["top30"],
        "rank_ic": result["mean_rank_ic"],
        "outside20": result["outside_base_top60_share"]["top20"],
    }


def run_for_expand(splits: dict[str, list[dict[str, Any]]], expand_size: int) -> dict[str, Any]:
    rows = []
    selected = None
    for use_group in [True, False]:
        for winner_n, loser_n in TAIL_SPECS:
            for k in K_VALUES:
                scorer = build_scorer(splits["train"], use_group, winner_n, loser_n, k)
                configs = [("pure", -1.0)] + [("anchor", w) for w in WEIGHTS]
                for mode, weight in configs:
                    evals = {name: evaluate(items, scorer, mode, weight) for name, items in splits.items()}
                    valid = evals["valid"]
                    # Reward valid return, but lightly reward expansion so we do not just reproduce top60.
                    valid_score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.001 * valid["mean_rank_ic"] + 0.0005 * valid["outside_base_top60_share"]["top20"]
                    row = {
                        "expand_size": expand_size,
                        "use_group": use_group,
                        "winner_n": winner_n,
                        "loser_n": loser_n,
                        "k": k,
                        "mode": mode,
                        "weight": weight,
                        "valid_score": valid_score,
                        "valid": compact(valid),
                        "dev": compact(evals["dev"]),
                        "forward": compact(evals["forward"]),
                    }
                    rows.append(row)
                    if selected is None or valid_score > selected["selection"]["valid_score"]:
                        selected = {"selection": row, "evals": evals, "scorer_meta": {"expand_size": expand_size, "use_group": use_group, "winner_n": winner_n, "loser_n": loser_n, "k": k, "tail_counts": scorer["tail_counts"]}}
    assert selected is not None
    return {"leaderboard_top12": sorted(rows, key=lambda r: r["valid_score"], reverse=True)[:12], "selected": selected}


def main() -> None:
    started = time.perf_counter()
    G60.set_seed(SEED)
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    by_expand = {}
    for expand_size in EXPAND_POOL_SIZES:
        splits = {
            "train": build_split(market, amount20_arr, groups, G60.TRAIN_START, G60.TRAIN_END, expand_size),
            "valid": build_split(market, amount20_arr, groups, G60.VALID_START, G60.VALID_END, expand_size),
            "dev": build_split(market, amount20_arr, groups, G60.DEV_START, G60.DEV_END, expand_size),
            "forward": build_split(market, amount20_arr, groups, G60.FWD_START, G60.FWD_END, expand_size),
        }
        by_expand[str(expand_size)] = {"sample_counts": {name: len(items) for name, items in splits.items()}, **run_for_expand(splits, expand_size)}
    selected = max((item["selected"] for item in by_expand.values()), key=lambda payload: payload["selection"]["valid_score"])
    selected_evals = selected["evals"]
    verdict = "right_tail_prototype_pool_expansion_not_enough"
    dev_years = selected_evals["dev"]["by_year_model_top20"]
    if (
        selected_evals["forward"]["model_top"]["top20"] > selected_evals["forward"]["base_top"]["top20"] + 0.003
        and selected_evals["dev"]["model_top"]["top20"] > selected_evals["dev"]["base_top"]["top20"]
        and min(dev_years.values()) > 0.0
        and selected_evals["forward"]["outside_base_top60_share"]["top20"] > 0.10
    ):
        verdict = "right_tail_prototype_pool_expansion_candidate_needs_replay"
    out = {
        "experiment": "right_tail_prototype_pool_expansion_v1",
        "method": "train_only_winner_loser_prototypes_rank_expanded_base_candidate_pool",
        "params": {"expand_pool_sizes": EXPAND_POOL_SIZES, "base_domain_size": BASE_DOMAIN_SIZE, "k_values": K_VALUES, "tail_specs": TAIL_SPECS, "weights": WEIGHTS, "interval": G60.INTERVAL, "pool_size": G60.POOL_SIZE},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round62_script": sha256(R62_PATH), "round62_summary": sha256(ROOT / "right_tail_prototype_world_model_v1_summary.json")},
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
        "leaderboards": {k: v["leaderboard_top12"][:5] for k, v in by_expand.items()},
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
