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
L66_PATH = ROOT / "analyze_liquidity_cycle_causal_state_bucket_v1.py"
OUT = ROOT / "liquidity_state_regime_flip_diagnostic_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L66 = load_module("liquidity_cycle_causal_state_bucket_v1_for_flip", L66_PATH)
L65 = L66.L65
G60 = L66.G60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def rank_layer(pos: int) -> str:
    if pos < 10:
        return "rank01_10"
    if pos < 20:
        return "rank11_20"
    if pos < 40:
        return "rank21_40"
    return "rank41_60"


def add_market_features(market: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> None:
    close = market["arrays"]["close"].astype(float)
    volume = market["arrays"]["volume"].astype(float)
    vwap = market["arrays"]["vwap"].astype(float)
    amount = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close) * volume
    amount = np.where(np.isfinite(amount) & (amount > 0), amount, np.nan)
    for items in splits.values():
        for item in items:
            idx = market["date_index"][item["date"]]
            cols = np.flatnonzero(np.isfinite(close[idx]) & (close[idx] > 0))
            feats = {
                "market_ret20": 0.0,
                "market_ret60": 0.0,
                "breadth20": 0.0,
                "breadth60": 0.0,
                "amount20_120": 0.0,
                "amount5_20": 0.0,
                "disp20": 0.0,
            }
            if len(cols) and idx >= 120:
                c0 = close[idx, cols]
                r20 = safe_div(c0, close[idx - 20, cols]) - 1.0
                r60 = safe_div(c0, close[idx - 60, cols]) - 1.0
                amt5 = np.nanmean(amount[idx - 4:idx + 1, :][:, cols], axis=0)
                amt20 = np.nanmean(amount[idx - 19:idx + 1, :][:, cols], axis=0)
                amt120 = np.nanmean(amount[idx - 119:idx + 1, :][:, cols], axis=0)
                feats = {
                    "market_ret20": safe_mean(r20),
                    "market_ret60": safe_mean(r60),
                    "breadth20": float(np.nanmean(r20 > 0.0)),
                    "breadth60": float(np.nanmean(r60 > 0.0)),
                    "amount20_120": safe_mean(safe_div(amt20, amt120)),
                    "amount5_20": safe_mean(safe_div(amt5, amt20)),
                    "disp20": float(np.nanstd(r20[np.isfinite(r20)])) if np.any(np.isfinite(r20)) else 0.0,
                }
            item["market_features"] = feats


def market_thresholds(train: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = list(train[0]["market_features"].keys())
    out: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([item["market_features"][name] for item in train], dtype=float)
        values = values[np.isfinite(values)]
        out[name] = {f"q{q}": float(np.quantile(values, q / 100.0)) for q in [25, 35, 40, 50, 60, 65, 75]}
    return out


def classify_market(item: dict[str, Any], qs: dict[str, dict[str, float]]) -> str:
    f = item["market_features"]
    trend_up = f["market_ret20"] >= qs["market_ret20"]["q60"] and f["breadth20"] >= qs["breadth20"]["q60"]
    trend_down = f["market_ret20"] <= qs["market_ret20"]["q40"] and f["breadth20"] <= qs["breadth20"]["q40"]
    liquidity_up = f["amount20_120"] >= qs["amount20_120"]["q60"] or f["amount5_20"] >= qs["amount5_20"]["q65"]
    liquidity_down = f["amount20_120"] <= qs["amount20_120"]["q35"] and f["amount5_20"] <= qs["amount5_20"]["q50"]
    high_disp = f["disp20"] >= qs["disp20"]["q65"]
    if trend_up and liquidity_up:
        return "market_up_liq_up"
    if trend_up:
        return "market_up_price_only"
    if trend_down and liquidity_down:
        return "market_down_liq_down"
    if trend_down:
        return "market_down_price_only"
    if liquidity_up and high_disp:
        return "liq_up_high_dispersion"
    if liquidity_up:
        return "liq_up_neutral_price"
    return "neutral_market"


def add_market_regime(splits: dict[str, list[dict[str, Any]]], qs: dict[str, dict[str, float]]) -> None:
    for items in splits.values():
        for item in items:
            item["market_regime"] = classify_market(item, qs)


def add_top20_flags(items: list[dict[str, Any]]) -> None:
    for item in items:
        base_top20 = set(item["base_order"][:20])
        item["base_top20_flags"] = [i in base_top20 for i in range(len(item["y"]))]


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    out = []
    for combo, vals in groups.items():
        labels = [float(v["label"]) for v in vals]
        top20_labels = [float(v["label"]) for v in vals if v["base_top20"]]
        out.append({
            **{k: combo[i] for i, k in enumerate(keys)},
            "count": len(vals),
            "label_mean": safe_mean(labels),
            "top20_count": len(top20_labels),
            "top20_label_mean": safe_mean(top20_labels),
            "right_tail_rate": float(np.mean(np.asarray(labels) >= np.quantile(labels, 0.8))) if len(labels) >= 5 else 0.0,
        })
    return sorted(out, key=lambda r: tuple(r[k] for k in keys))


def flatten(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split, items in splits.items():
        rows = []
        for item in items:
            for pos, (bucket, label, top20) in enumerate(zip(item["bucket"], item["y"], item["base_top20_flags"], strict=True)):
                rows.append({
                    "split": split,
                    "date": item["date"],
                    "year": item["year"],
                    "bucket": str(bucket),
                    "market_regime": str(item["market_regime"]),
                    "rank_layer": rank_layer(pos),
                    "base_top20": bool(top20),
                    "label": float(label),
                })
        out[split] = rows
    return out


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_year_bucket": aggregate(rows, ["year", "bucket"]),
        "by_year_bucket_rank_layer": aggregate(rows, ["year", "bucket", "rank_layer"]),
        "by_year_market_bucket": aggregate(rows, ["year", "market_regime", "bucket"]),
        "by_market_bucket": aggregate(rows, ["market_regime", "bucket"]),
        "by_bucket_rank_layer": aggregate(rows, ["bucket", "rank_layer"]),
        "by_bucket": aggregate(rows, ["bucket"]),
    }


def pivot_bucket_year(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for row in aggregate(rows, ["year", "bucket"]):
        table.setdefault(row["bucket"], {})[row["year"]] = row["label_mean"]
    return table


def top_records(records: list[dict[str, Any]], metric: str, min_count: int = 20, n: int = 20) -> list[dict[str, Any]]:
    kept = [r for r in records if int(r["count"]) >= min_count]
    return sorted(kept, key=lambda r: float(r[metric]), reverse=True)[:n]


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
    bucket_thresholds = L66.qmap(splits["train"])
    L66.add_buckets(splits, bucket_thresholds)
    add_market_features(market, splits)
    regime_thresholds = market_thresholds(splits["train"])
    add_market_regime(splits, regime_thresholds)
    for items in splits.values():
        add_top20_flags(items)
    flat = flatten(splits)
    summaries = {split: summarize_split(rows) for split, rows in flat.items()}
    dev_rows = flat["dev"]
    fwd_rows = flat["forward"]
    forward_hot = [r for r in summaries["forward"]["by_bucket"] if r["bucket"] in {"pulse_exhausted", "pulse_accelerating"}]
    dev_hot = [r for r in summaries["dev"]["by_bucket"] if r["bucket"] in {"pulse_exhausted", "pulse_accelerating"}]
    verdict = "liquidity_state_regime_flip_confirms_market_phase_dependency"
    hot_fwd = safe_mean([r["label_mean"] for r in forward_hot])
    hot_dev = safe_mean([r["label_mean"] for r in dev_hot])
    if hot_fwd <= 0.005 or hot_dev >= 0.0:
        verdict = "liquidity_state_regime_flip_inconclusive"
    out = {
        "experiment": "liquidity_state_regime_flip_diagnostic_v1",
        "method": "bucket_return_flip_by_year_market_regime_and_base_rank_layer",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "forward": [FWD_START, FWD_END],
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round66_script": sha256(L66_PATH),
            "round66_summary": sha256(ROOT / "liquidity_cycle_causal_state_bucket_v1_summary.json"),
        },
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "row_counts": {split: len(rows) for split, rows in flat.items()},
        "bucket_thresholds": bucket_thresholds,
        "market_regime_thresholds": regime_thresholds,
        "bucket_year_pivot_dev": pivot_bucket_year(dev_rows),
        "bucket_year_pivot_forward": pivot_bucket_year(fwd_rows),
        "top_forward_market_bucket": top_records(summaries["forward"]["by_market_bucket"], "label_mean", min_count=10),
        "top_dev_market_bucket": top_records(summaries["dev"]["by_market_bucket"], "label_mean", min_count=50),
        "top_forward_bucket_rank_layer": top_records(summaries["forward"]["by_bucket_rank_layer"], "label_mean", min_count=10),
        "top_dev_bucket_rank_layer": top_records(summaries["dev"]["by_bucket_rank_layer"], "label_mean", min_count=50),
        "summaries": summaries,
        "diagnostics": {
            "hot_bucket_forward_mean": hot_fwd,
            "hot_bucket_dev_mean": hot_dev,
            "forward_hot_buckets": forward_hot,
            "dev_hot_buckets": dev_hot,
        },
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "row_counts": out["row_counts"],
        "bucket_year_pivot_dev": out["bucket_year_pivot_dev"],
        "bucket_year_pivot_forward": out["bucket_year_pivot_forward"],
        "top_forward_market_bucket": out["top_forward_market_bucket"][:10],
        "top_forward_bucket_rank_layer": out["top_forward_bucket_rank_layer"][:10],
        "diagnostics": out["diagnostics"],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
