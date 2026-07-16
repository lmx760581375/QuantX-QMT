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
L65_PATH = ROOT / "analyze_liquidity_cycle_world_model_v1.py"
OUT = ROOT / "liquidity_cycle_causal_state_bucket_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_KS = [10, 20, 30]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L65 = load_module("liquidity_cycle_world_model_v1_for_bucket", L65_PATH)
G60 = L65.G60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def qmap(train: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    x = np.vstack([item["liquidity_x"] for item in train]).astype(float)
    names = {
        "self_amt20_120": 0,
        "self_amt5_20": 1,
        "self_ret20": 3,
        "self_ret60": 4,
        "group_share20": 5,
        "group_amt20_120": 6,
        "group_mean_amt20_120": 7,
        "group_amt5_20": 8,
        "group_width12": 9,
        "group_width15": 10,
        "group_ret20": 11,
        "group_ret60": 12,
        "group_up20_width": 13,
        "group_up60_width": 14,
        "group_ret20_std": 15,
        "group_alt_amt20_120": 16,
        "group_alt_width12": 17,
        "group_alt_ret20": 18,
    }
    out: dict[str, dict[str, float]] = {}
    for name, idx in names.items():
        col = x[:, idx]
        col = col[np.isfinite(col)]
        out[name] = {f"q{q}": float(np.quantile(col, q / 100.0)) for q in [15, 20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 85, 90]}
    return out


def bucket_for_row(row: np.ndarray, qs: dict[str, dict[str, float]]) -> str:
    self_amt20_120 = float(row[0])
    self_amt5_20 = float(row[1])
    self_ret20 = float(row[3])
    group_amt20_120 = max(float(row[6]), float(row[7]), float(row[16]))
    group_amt5_20 = float(row[8])
    group_width12 = max(float(row[9]), float(row[17]))
    group_width15 = float(row[10])
    group_ret20 = max(float(row[11]), float(row[18]))
    group_up20 = float(row[13])

    pulse = (
        group_amt20_120 >= qs["group_amt20_120"]["q65"]
        and group_width12 >= qs["group_width12"]["q60"]
    )
    strong_pulse = (
        group_amt20_120 >= qs["group_amt20_120"]["q75"]
        and group_width15 >= qs["group_width15"]["q60"]
    )
    fading_short = (
        group_amt5_20 <= qs["group_amt5_20"]["q35"]
        or row[1] <= qs["self_amt5_20"]["q35"]
    )
    overheated_price = (
        self_ret20 >= qs["self_ret20"]["q85"]
        or group_ret20 >= qs["group_ret20"]["q85"]
    )
    fresh_price = (
        self_ret20 <= qs["self_ret20"]["q75"]
        and group_ret20 <= qs["group_ret20"]["q80"]
    )
    short_confirm = (
        group_amt5_20 >= qs["group_amt5_20"]["q40"]
        and self_amt20_120 >= qs["self_amt20_120"]["q50"]
    )

    if strong_pulse and overheated_price and fading_short:
        return "pulse_exhausted"
    if strong_pulse and group_up20 >= qs["group_up20_width"]["q65"] and not fading_short:
        return "pulse_accelerating"
    if pulse and fresh_price and short_confirm:
        return "pulse_diffusing_fresh"
    if pulse and fading_short:
        return "pulse_fading"
    if group_amt20_120 <= qs["group_amt20_120"]["q40"] or group_width12 <= qs["group_width12"]["q35"]:
        return "no_pulse"
    return "neutral"


def add_buckets(splits: dict[str, list[dict[str, Any]]], qs: dict[str, dict[str, float]]) -> None:
    for items in splits.values():
        for item in items:
            item["bucket"] = [bucket_for_row(row, qs) for row in item["liquidity_x"]]


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def residual_score(item: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    residual = np.nan_to_num((item["liquidity_x"] - model["mu"]) / model["sd"], nan=0.0, posinf=0.0, neginf=0.0) @ model["coef"]
    return G60.finite_rank(residual) - 0.5


def score_item(item: dict[str, Any], model: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    base = anchor_score(len(item["y"]))
    residual = residual_score(item, model)
    buckets = np.asarray(item["bucket"], dtype=object)
    active = np.isin(buckets, cfg["active_buckets"]).astype(float)
    avoid = np.isin(buckets, cfg["avoid_buckets"]).astype(float)
    if cfg["mode"] == "active_residual":
        return base + cfg["weight"] * residual * active - cfg["avoid_penalty"] * avoid
    if cfg["mode"] == "positive_active_residual":
        return base + cfg["weight"] * np.maximum(residual, 0.0) * active - cfg["avoid_penalty"] * avoid
    if cfg["mode"] == "state_gate_only":
        return base + cfg["gate_bonus"] * active - cfg["avoid_penalty"] * avoid
    raise ValueError(cfg["mode"])


def evaluate(sessions: list[dict[str, Any]], model: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    top = {k: [] for k in TOP_KS}
    base = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    bucket_counts: dict[str, int] = {}
    top20_bucket_counts: dict[str, int] = {}
    bucket_labels: dict[str, list[float]] = {}
    top20_overlap = []
    for item in sessions:
        score = score_item(item, model, cfg)
        order = np.argsort(-score, kind="mergesort")
        y = item["y"]
        base_order = item["base_order"]
        ics.append(G60.rank_ic(score, y))
        for b, label in zip(item["bucket"], y, strict=True):
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
            bucket_labels.setdefault(b, []).append(float(label))
        for b in np.asarray(item["bucket"], dtype=object)[order[:20]]:
            top20_bucket_counts[str(b)] = top20_bucket_counts.get(str(b), 0) + 1
        top20_overlap.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    total_top20 = sum(top20_bucket_counts.values()) or 1
    return {
        "sessions": len(sessions),
        "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()},
        "mean_rank_ic": safe_mean(ics),
        "top20_overlap_with_base": safe_mean(top20_overlap),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "top20_bucket_share": {k: float(v / total_top20) for k, v in sorted(top20_bucket_counts.items())},
        "bucket_label_mean": {k: safe_mean(v) for k, v in sorted(bucket_labels.items())},
    }


def configs() -> list[dict[str, Any]]:
    active_sets = [
        ["pulse_diffusing_fresh"],
        ["pulse_diffusing_fresh", "pulse_accelerating"],
        ["pulse_diffusing_fresh", "neutral"],
    ]
    avoid_sets = [
        ["pulse_exhausted", "pulse_fading"],
        ["pulse_exhausted"],
        ["pulse_fading"],
        [],
    ]
    rows: list[dict[str, Any]] = []
    for active in active_sets:
        for avoid in avoid_sets:
            for weight in [0.15, 0.30, 0.45, 0.60]:
                for penalty in [0.0, 0.05, 0.10, 0.20]:
                    rows.append({"mode": "active_residual", "active_buckets": active, "avoid_buckets": avoid, "weight": weight, "avoid_penalty": penalty, "gate_bonus": 0.0})
                    rows.append({"mode": "positive_active_residual", "active_buckets": active, "avoid_buckets": avoid, "weight": weight, "avoid_penalty": penalty, "gate_bonus": 0.0})
            for bonus in [0.03, 0.06, 0.10, 0.15]:
                for penalty in [0.0, 0.05, 0.10, 0.20]:
                    rows.append({"mode": "state_gate_only", "active_buckets": active, "avoid_buckets": avoid, "weight": 0.0, "avoid_penalty": penalty, "gate_bonus": bonus})
    rows.append({"mode": "state_gate_only", "active_buckets": [], "avoid_buckets": [], "weight": 0.0, "avoid_penalty": 0.0, "gate_bonus": 0.0})
    return rows


def main() -> None:
    started = time.perf_counter()
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": L65.build_split(market, amount20_arr, groups, TRAIN_START, TRAIN_END),
        "valid": L65.build_split(market, amount20_arr, groups, VALID_START, VALID_END),
        "dev": L65.build_split(market, amount20_arr, groups, DEV_START, DEV_END),
        "forward": L65.build_split(market, amount20_arr, groups, FWD_START, FWD_END),
    }
    model = L65.train_linear(splits["train"], splits["valid"])
    thresholds = qmap(splits["train"])
    add_buckets(splits, thresholds)
    rows = []
    for cfg in configs():
        evals = {name: evaluate(items, model, cfg) for name, items in splits.items()}
        valid = evals["valid"]
        dev = evals["dev"]
        min_dev_year = min(dev["by_year_model_top20"].values()) if dev["by_year_model_top20"] else -1.0
        # valid is primary; small penalties discourage Top10 damage and unstable years without seeing forward.
        score = (
            valid["model_top"]["top20"]
            + 0.20 * valid["model_top"]["top10"]
            + 0.0015 * valid["mean_rank_ic"]
            + 0.10 * min(0.0, min_dev_year)
        )
        rows.append({"config": cfg, "valid_score": score, "valid": valid, "dev": dev, "forward": evals["forward"]})
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    base = next(r for r in rows if r["config"]["mode"] == "state_gate_only" and not r["config"]["active_buckets"] and not r["config"]["avoid_buckets"])
    verdict = "liquidity_cycle_causal_state_bucket_not_enough"
    dev_years = selected["dev"]["by_year_model_top20"]
    if (
        selected["forward"]["model_top"]["top20"] > base["forward"]["model_top"]["top20"] + 0.003
        and selected["forward"]["model_top"]["top10"] >= base["forward"]["model_top"]["top10"] * 0.9
        and selected["dev"]["model_top"]["top20"] > base["dev"]["model_top"]["top20"]
        and min(dev_years.values()) > 0.0
    ):
        verdict = "liquidity_cycle_causal_state_bucket_candidate_needs_replay"
    out = {
        "experiment": "liquidity_cycle_causal_state_bucket_v1",
        "method": "train_only_threshold_liquidity_state_bucket_gate_inside_mid_trend_volume_top60",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "forward": [FWD_START, FWD_END],
            "top_ks": TOP_KS,
            "config_count": len(rows),
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round65_script": sha256(L65_PATH),
            "round65_summary": sha256(ROOT / "liquidity_cycle_world_model_v1_summary.json"),
        },
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "thresholds": thresholds,
        "base": base,
        "selected": selected,
        "leaderboard": rows[:30],
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "selected": {"config": selected["config"], "valid": selected["valid"], "dev": selected["dev"], "forward": selected["forward"]},
        "base": {"valid": base["valid"], "dev": base["dev"], "forward": base["forward"]},
        "leaderboard_top": rows[:8],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
