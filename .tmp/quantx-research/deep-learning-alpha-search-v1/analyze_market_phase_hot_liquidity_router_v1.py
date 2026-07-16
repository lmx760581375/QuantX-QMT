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
L67_PATH = ROOT / "analyze_liquidity_state_regime_flip_diagnostic_v1.py"
OUT = ROOT / "market_phase_hot_liquidity_router_v1_summary.json"

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


L67 = load_module("liquidity_state_regime_flip_diagnostic_v1_for_router", L67_PATH)
L66 = L67.L66
L65 = L67.L65
G60 = L67.G60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def anchor_score(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, endpoint=True) if n > 1 else np.asarray([0.5])


def rank_layer(pos: int) -> str:
    if pos < 10:
        return "rank01_10"
    if pos < 20:
        return "rank11_20"
    if pos < 40:
        return "rank21_40"
    return "rank41_60"


def configs() -> list[dict[str, Any]]:
    regime_sets = [
        ["market_up_liq_up"],
        ["market_up_liq_up", "liq_up_high_dispersion"],
        ["market_up_liq_up", "market_down_price_only"],
        ["market_up_liq_up", "market_down_liq_down"],
        ["market_up_liq_up", "market_down_liq_down", "market_down_price_only"],
        ["liq_up_high_dispersion"],
        ["market_down_liq_down", "market_down_price_only"],
    ]
    bucket_sets = [
        ["pulse_accelerating"],
        ["pulse_exhausted"],
        ["pulse_accelerating", "pulse_exhausted"],
    ]
    layer_sets = [
        ["rank21_40"],
        ["rank41_60"],
        ["rank21_40", "rank41_60"],
        ["rank11_20", "rank21_40", "rank41_60"],
    ]
    rows = [{"mode": "base", "active_regimes": [], "hot_buckets": [], "rank_layers": [], "boost": 0.0, "cool_penalty": 0.0}]
    for regimes in regime_sets:
        for buckets in bucket_sets:
            for layers in layer_sets:
                for boost in [0.03, 0.06, 0.10, 0.15, 0.25, 0.40]:
                    rows.append({"mode": "hot_boost", "active_regimes": regimes, "hot_buckets": buckets, "rank_layers": layers, "boost": boost, "cool_penalty": 0.0})
                    for cool_penalty in [0.03, 0.06, 0.10]:
                        rows.append({"mode": "hot_boost_cool_penalty", "active_regimes": regimes, "hot_buckets": buckets, "rank_layers": layers, "boost": boost, "cool_penalty": cool_penalty})
    return rows


def score_item(item: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    score = anchor_score(len(item["y"])).astype(float)
    if cfg["mode"] == "base":
        return score
    active_regime = item["market_regime"] in set(cfg["active_regimes"])
    buckets = np.asarray(item["bucket"], dtype=object)
    layers = np.asarray([rank_layer(i) for i in range(len(buckets))], dtype=object)
    hot = np.isin(buckets, cfg["hot_buckets"]) & np.isin(layers, cfg["rank_layers"])
    if active_regime:
        score = score + cfg["boost"] * hot.astype(float)
    elif cfg["cool_penalty"] > 0:
        score = score - cfg["cool_penalty"] * hot.astype(float)
    return score


def evaluate(items: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    top = {k: [] for k in TOP_KS}
    base = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    overlap20 = []
    promoted_selected = []
    promoted_pool = []
    active_sessions = 0
    for item in items:
        score = score_item(item, cfg)
        order = np.argsort(-score, kind="mergesort")
        y = item["y"]
        base_order = item["base_order"]
        buckets = np.asarray(item["bucket"], dtype=object)
        layers = np.asarray([rank_layer(i) for i in range(len(buckets))], dtype=object)
        hot = np.isin(buckets, cfg["hot_buckets"]) & np.isin(layers, cfg["rank_layers"])
        if cfg["mode"] != "base" and item["market_regime"] in set(cfg["active_regimes"]):
            active_sessions += 1
            promoted_pool.append(float(np.mean(hot)))
            promoted_selected.append(float(np.mean(hot[order[:20]])))
        ics.append(G60.rank_ic(score, y))
        overlap20.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            top[k].append(value)
            base[k].append(safe_mean(y[base_order[:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(items),
        "active_sessions": active_sessions,
        "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base.items()},
        "mean_rank_ic": safe_mean(ics),
        "top20_overlap_with_base": safe_mean(overlap20),
        "promoted_pool_share_active": safe_mean(promoted_pool),
        "promoted_top20_share_active": safe_mean(promoted_selected),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def prepare_splits() -> dict[str, list[dict[str, Any]]]:
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": L65.build_split(market, amount20_arr, groups, TRAIN_START, TRAIN_END),
        "valid": L65.build_split(market, amount20_arr, groups, VALID_START, VALID_END),
        "dev": L65.build_split(market, amount20_arr, groups, DEV_START, DEV_END),
        "forward": L65.build_split(market, amount20_arr, groups, FWD_START, FWD_END),
    }
    bucket_thresholds = L66.qmap(splits["train"])
    L66.add_buckets(splits, bucket_thresholds)
    L67.add_market_features(market, splits)
    regime_thresholds = L67.market_thresholds(splits["train"])
    L67.add_market_regime(splits, regime_thresholds)
    return splits


def main() -> None:
    started = time.perf_counter()
    splits = prepare_splits()
    rows = []
    for cfg in configs():
        evals = {name: evaluate(items, cfg) for name, items in splits.items()}
        valid = evals["valid"]
        dev = evals["dev"]
        min_dev_year = min(dev["by_year_model_top20"].values()) if dev["by_year_model_top20"] else -1.0
        # 只用 valid/dev 选择：收益优先，惩罚年度恶化和过度换手，避免人为把 rank41-60 大量拉上来。
        score = (
            valid["model_top"]["top20"]
            + 0.25 * valid["model_top"]["top10"]
            + 0.0015 * valid["mean_rank_ic"]
            + 0.15 * min(0.0, min_dev_year)
            - 0.001 * max(0.0, 0.90 - valid["top20_overlap_with_base"])
        )
        rows.append({"config": cfg, "valid_score": score, "valid": valid, "dev": dev, "forward": evals["forward"]})
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    base = next(r for r in rows if r["config"]["mode"] == "base")
    selected = rows[0]
    verdict = "market_phase_hot_liquidity_router_not_enough"
    dev_years = selected["dev"]["by_year_model_top20"]
    if (
        selected["forward"]["model_top"]["top20"] > base["forward"]["model_top"]["top20"] + 0.004
        and selected["forward"]["model_top"]["top10"] >= base["forward"]["model_top"]["top10"] * 0.95
        and selected["dev"]["model_top"]["top20"] > base["dev"]["model_top"]["top20"]
        and min(dev_years.values()) > 0.0
    ):
        verdict = "market_phase_hot_liquidity_router_candidate_needs_replay"
    out = {
        "experiment": "market_phase_hot_liquidity_router_v1",
        "method": "train_valid_selected_market_regime_hot_liquidity_rank_layer_boost",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "forward": [FWD_START, FWD_END],
            "top_ks": TOP_KS,
            "config_count": len(rows),
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round67_script": sha256(L67_PATH),
            "round67_summary": sha256(ROOT / "liquidity_state_regime_flip_diagnostic_v1_summary.json"),
        },
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "base": base,
        "selected": selected,
        "leaderboard": rows[:60],
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "base": {"valid": base["valid"], "dev": base["dev"], "forward": base["forward"]},
        "selected": {"config": selected["config"], "valid": selected["valid"], "dev": selected["dev"], "forward": selected["forward"]},
        "leaderboard_top": rows[:10],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
