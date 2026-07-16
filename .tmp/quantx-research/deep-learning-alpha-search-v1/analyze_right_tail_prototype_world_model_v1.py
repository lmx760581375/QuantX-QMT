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
OUT = ROOT / "right_tail_prototype_world_model_v1_summary.json"

TOP_KS = [10, 20]
SEED = 20260714
K_VALUES = [4, 8, 16]
TAIL_SPECS = [(5, 10), (10, 10), (10, 20)]
WEIGHTS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_right_tail_prototype", G60_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def feature_matrix(item: dict[str, Any], use_group: bool) -> np.ndarray:
    return item["stock_x"] if not use_group else np.hstack([item["stock_x"], item["group_x"]])


def standard_params_from_sessions(sessions: list[dict[str, Any]], use_group: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([feature_matrix(item, use_group) for item in sessions]).astype(np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    return mu.astype(np.float32), sd


def standardized_x(item: dict[str, Any], mu: np.ndarray, sd: np.ndarray, use_group: bool) -> np.ndarray:
    return ((feature_matrix(item, use_group).astype(np.float32) - mu) / sd).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def kmeans(x: np.ndarray, k: int, iterations: int = 20) -> np.ndarray:
    x = normalize_rows(x.astype(np.float32))
    if len(x) <= k:
        return x.copy()
    # Deterministic farthest-point initialization keeps this small experiment stable.
    centers = [int(np.argmax(np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)))]
    while len(centers) < k:
        sim = x @ x[centers].T
        dist = 1.0 - np.max(sim, axis=1)
        centers.append(int(np.argmax(dist)))
    c = x[centers].copy()
    for _ in range(iterations):
        labels = np.argmax(x @ c.T, axis=1)
        new_c = []
        for j in range(k):
            part = x[labels == j]
            new_c.append(part.mean(axis=0) if len(part) else c[j])
        c = normalize_rows(np.asarray(new_c, dtype=np.float32))
    return c


def collect_tails(sessions: list[dict[str, Any]], mu: np.ndarray, sd: np.ndarray, use_group: bool, winner_n: int, loser_n: int) -> tuple[np.ndarray, np.ndarray]:
    winners = []
    losers = []
    for item in sessions:
        x = standardized_x(item, mu, sd, use_group)
        y = item["y"]
        order = np.argsort(-y, kind="mergesort")
        winners.append(x[order[:winner_n]])
        losers.append(x[order[-loser_n:]])
    return np.vstack(winners).astype(np.float32), np.vstack(losers).astype(np.float32)


def prototype_score(item: dict[str, Any], mu: np.ndarray, sd: np.ndarray, use_group: bool, winner_proto: np.ndarray, loser_proto: np.ndarray) -> np.ndarray:
    x = normalize_rows(standardized_x(item, mu, sd, use_group))
    win = np.max(x @ winner_proto.T, axis=1)
    lose = np.max(x @ loser_proto.T, axis=1)
    return G60.finite_rank(win - lose) - 0.5


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def evaluate(sessions: list[dict[str, Any]], scorer: dict[str, Any], weight: float) -> dict[str, Any]:
    model_top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    changed = []
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        y = item["y"]
        residual = prototype_score(item, scorer["mu"], scorer["sd"], scorer["use_group"], scorer["winner_proto"], scorer["loser_proto"])
        score = residual if weight < 0 else anchor_score(len(y)) + weight * residual
        order = np.argsort(-score, kind="mergesort")
        base_order = item["base_order"]
        ics.append(G60.rank_ic(score, y))
        changed.append(1.0 - len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            model_top[k].append(value)
            base_top[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(sessions),
        "weight": weight,
        "model_top": {f"top{k}": safe_mean(v) for k, v in model_top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "mean_rank_ic": safe_mean(ics),
        "mean_top20_turnover_vs_base": safe_mean(changed),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def build_scorer(train_sessions: list[dict[str, Any]], use_group: bool, winner_n: int, loser_n: int, k: int) -> dict[str, Any]:
    mu, sd = standard_params_from_sessions(train_sessions, use_group)
    winners, losers = collect_tails(train_sessions, mu, sd, use_group, winner_n, loser_n)
    return {
        "use_group": use_group,
        "winner_n": winner_n,
        "loser_n": loser_n,
        "k": k,
        "mu": mu,
        "sd": sd,
        "winner_proto": kmeans(winners, k),
        "loser_proto": kmeans(losers, k),
        "tail_counts": {"winners": int(len(winners)), "losers": int(len(losers))},
    }


def compact_eval(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "top10": result["model_top"]["top10"],
        "top20": result["model_top"]["top20"],
        "rank_ic": result["mean_rank_ic"],
        "turnover": result["mean_top20_turnover_vs_base"],
    }


def run_grid(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    selected_payload = None
    for use_group in [False, True]:
        for winner_n, loser_n in TAIL_SPECS:
            for k in K_VALUES:
                scorer = build_scorer(splits["train"], use_group, winner_n, loser_n, k)
                for mode, weight in [("pure", -1.0)] + [("anchor", w) for w in WEIGHTS]:
                    evals = {name: evaluate(items, scorer, weight) for name, items in splits.items()}
                    valid = evals["valid"]
                    score = valid["model_top"]["top20"] + 0.35 * valid["model_top"]["top10"] + 0.002 * valid["mean_rank_ic"] - 0.0015 * valid["mean_top20_turnover_vs_base"]
                    row = {
                        "use_group": use_group,
                        "winner_n": winner_n,
                        "loser_n": loser_n,
                        "k": k,
                        "mode": mode,
                        "weight": weight,
                        "valid_score": score,
                        "valid": compact_eval(valid),
                        "dev": compact_eval(evals["dev"]),
                        "forward": compact_eval(evals["forward"]),
                    }
                    rows.append(row)
                    if selected_payload is None or score > selected_payload["selection"]["valid_score"]:
                        selected_payload = {"selection": row, "evals": evals, "scorer_meta": {"use_group": use_group, "winner_n": winner_n, "loser_n": loser_n, "k": k, "tail_counts": scorer["tail_counts"]}}
    assert selected_payload is not None
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    return {"leaderboard_top20": rows[:20], "selected": selected_payload}


def main() -> None:
    started = time.perf_counter()
    G60.set_seed(SEED)
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": G60.build_split(market, amount20_arr, groups, G60.TRAIN_START, G60.TRAIN_END),
        "valid": G60.build_split(market, amount20_arr, groups, G60.VALID_START, G60.VALID_END),
        "dev": G60.build_split(market, amount20_arr, groups, G60.DEV_START, G60.DEV_END),
        "forward": G60.build_split(market, amount20_arr, groups, G60.FWD_START, G60.FWD_END),
    }
    grid = run_grid(splits)
    selected = grid["selected"]["evals"]
    verdict = "right_tail_prototype_world_model_not_enough"
    dev_years = selected["dev"]["by_year_model_top20"]
    if (
        selected["forward"]["model_top"]["top20"] > selected["forward"]["base_top"]["top20"] + 0.003
        and selected["dev"]["model_top"]["top20"] > selected["dev"]["base_top"]["top20"]
        and min(dev_years.values()) > 0.0
    ):
        verdict = "right_tail_prototype_world_model_candidate_needs_replay"
    out = {
        "experiment": "right_tail_prototype_world_model_v1",
        "method": "train_only_winner_loser_prototypes_with_valid_selected_anchor_residual",
        "params": {"k_values": K_VALUES, "tail_specs": TAIL_SPECS, "weights": WEIGHTS, "interval": G60.INTERVAL, "pool_size": G60.POOL_SIZE, "domain_size": G60.DOMAIN_SIZE, "train": [G60.TRAIN_START, G60.TRAIN_END], "valid": [G60.VALID_START, G60.VALID_END], "forward": [G60.FWD_START, G60.FWD_END]},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round61_summary": sha256(ROOT / "group_residual_anchor_world_model_v1_summary.json"), "round60_script": sha256(G60_PATH)},
        "group_coverage": {key: value for key, value in groups.items() if key not in {"industry", "concept"}},
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "leaderboard_top20": grid["leaderboard_top20"],
        "selected": grid["selected"],
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "selected_meta": out["selected"]["scorer_meta"],
        "selected_config": out["selected"]["selection"],
        "selected_results": out["selected"]["evals"],
        "leaderboard_top8": out["leaderboard_top20"][:8],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
