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
OUT = ROOT / "session_phase_hot_continuation_meta_model_v1_summary.json"
CACHE = ROOT / "session_tensor_cache_v1.npz"
CACHE_META = ROOT / "session_tensor_cache_v1_meta.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_KS = [10, 20, 30]
HOT_BUCKETS = {"pulse_accelerating", "pulse_exhausted"}
HOT_LAYERS = {"rank11_20", "rank21_40", "rank41_60"}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L67 = load_module("liquidity_state_regime_flip_diagnostic_v1_for_meta", L67_PATH)
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


def split_for_date(date: str) -> str:
    if TRAIN_START <= date <= TRAIN_END:
        return "train"
    if VALID_START <= date <= VALID_END:
        return "valid"
    if FWD_START <= date <= FWD_END:
        return "forward"
    return "other"


def build_cache() -> dict[str, Any]:
    market = G60.POS.load_market()
    groups = G60.load_groups(list(market["symbols"]))
    amount20_arr = G60.RT.amount20(market)
    splits = {
        "train": L65.build_split(market, amount20_arr, groups, TRAIN_START, TRAIN_END),
        "valid": L65.build_split(market, amount20_arr, groups, VALID_START, VALID_END),
        "forward": L65.build_split(market, amount20_arr, groups, FWD_START, FWD_END),
    }
    train_for_thresholds = splits["train"]
    bucket_thresholds = L66.qmap(train_for_thresholds)
    L66.add_buckets(splits, bucket_thresholds)
    L67.add_market_features(market, splits)
    regime_thresholds = L67.market_thresholds(train_for_thresholds)
    L67.add_market_regime(splits, regime_thresholds)
    items = sorted(splits["train"] + splits["valid"] + splits["forward"], key=lambda x: x["date"])
    feature_names = list(items[0]["market_features"].keys())
    dates = np.asarray([item["date"] for item in items], dtype="U10")
    years = np.asarray([item["year"] for item in items], dtype="U4")
    split = np.asarray([split_for_date(item["date"]) for item in items], dtype="U8")
    y = np.vstack([item["y"] for item in items]).astype(np.float32)
    buckets = np.asarray([item["bucket"] for item in items], dtype="U32")
    regimes = np.asarray([item["market_regime"] for item in items], dtype="U32")
    market_x = np.asarray([[item["market_features"][name] for name in feature_names] for item in items], dtype=np.float32)
    layers = np.asarray([[rank_layer(i) for i in range(y.shape[1])] for _ in items], dtype="U16")
    np.savez_compressed(CACHE, dates=dates, years=years, split=split, y=y, buckets=buckets, regimes=regimes, market_x=market_x, layers=layers, feature_names=np.asarray(feature_names, dtype="U32"))
    meta = {
        "created_by": Path(__file__).name,
        "cache": str(CACHE),
        "date_range": [str(dates[0]), str(dates[-1])],
        "sample_counts": {name: int(np.sum(split == name)) for name in ["train", "valid", "forward"]},
        "feature_names": feature_names,
        "inputs_sha256": {"script": sha256(Path(__file__)), "round67_script": sha256(L67_PATH), "round67_summary": sha256(ROOT / "liquidity_state_regime_flip_diagnostic_v1_summary.json")},
        "bucket_thresholds": bucket_thresholds,
        "market_regime_thresholds": regime_thresholds,
    }
    CACHE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return load_cache()


def load_cache() -> dict[str, Any]:
    if not CACHE.exists():
        return build_cache()
    data = np.load(CACHE, allow_pickle=False)
    return {k: data[k] for k in data.files}


def hot_mask_for_session(buckets: np.ndarray, layers: np.ndarray) -> np.ndarray:
    return np.isin(buckets, list(HOT_BUCKETS)) & np.isin(layers, list(HOT_LAYERS))


def score_with_hot(y: np.ndarray, buckets: np.ndarray, layers: np.ndarray, boost: float, allow: bool) -> np.ndarray:
    score = anchor_score(len(y)).astype(float)
    if allow:
        score += boost * hot_mask_for_session(buckets, layers).astype(float)
    return score


def top_return(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="mergesort")[:k]
    return safe_mean(y[order])


def base_score(n: int) -> np.ndarray:
    return anchor_score(n)


def session_delta(y: np.ndarray, buckets: np.ndarray, layers: np.ndarray, boost: float) -> dict[str, float]:
    base = base_score(len(y))
    hot = score_with_hot(y, buckets, layers, boost, True)
    return {f"delta_top{k}": top_return(y, hot, k) - top_return(y, base, k) for k in TOP_KS}


def rolling(values: list[float], idx: int, length: int) -> float:
    start = max(0, idx - length)
    return safe_mean(values[start:idx]) if idx > start else 0.0


def build_meta_dataset(cache: dict[str, Any], boost: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = cache["y"]
    buckets = cache["buckets"]
    layers = cache["layers"]
    market_x = cache["market_x"].astype(float)
    regimes = cache["regimes"]
    records = []
    past_base20: list[float] = []
    past_hot20: list[float] = []
    past_delta20: list[float] = []
    past_hot_share: list[float] = []
    for i in range(len(y)):
        hot_mask = hot_mask_for_session(buckets[i], layers[i])
        hot_share_all = float(np.mean(hot_mask))
        hot_share_top20 = float(np.mean(hot_mask[:20]))
        base = base_score(y.shape[1])
        delta = session_delta(y[i], buckets[i], layers[i], boost)
        hot_top20 = top_return(y[i], score_with_hot(y[i], buckets[i], layers[i], boost, True), 20)
        base_top20 = top_return(y[i], base, 20)
        x = list(market_x[i])
        x += [hot_share_all, hot_share_top20, float(regimes[i] == "market_up_liq_up"), float(regimes[i] == "liq_up_high_dispersion"), float(regimes[i] == "market_down_liq_down"), float(regimes[i] == "market_down_price_only")]
        for w in [4, 8, 16, 32]:
            x += [rolling(past_base20, i, w), rolling(past_hot20, i, w), rolling(past_delta20, i, w), rolling(past_hot_share, i, w)]
        label = float(delta["delta_top20"] > 0.001 and delta["delta_top10"] > -0.003)
        records.append({"idx": i, "date": str(cache["dates"][i]), "year": str(cache["years"][i]), "split": str(cache["split"][i]), "delta": delta, "base_top20": base_top20, "hot_top20": hot_top20, "hot_share_all": hot_share_all, "hot_share_top20": hot_share_top20, "label": label})
        past_base20.append(base_top20)
        past_hot20.append(hot_top20)
        past_delta20.append(delta["delta_top20"])
        past_hot_share.append(hot_share_all)
    return np.asarray([r for r in [rec for rec in []]], dtype=float), np.asarray([], dtype=float), records


def feature_matrix(cache: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    y = cache["y"]
    buckets = cache["buckets"]
    layers = cache["layers"]
    market_x = cache["market_x"].astype(float)
    regimes = cache["regimes"]
    past_base20: list[float] = []
    past_hot20: list[float] = []
    past_delta20: list[float] = []
    past_hot_share: list[float] = []
    rows = []
    for i in range(len(y)):
        hot_mask = hot_mask_for_session(buckets[i], layers[i])
        hot_share_all = float(np.mean(hot_mask))
        hot_share_top20 = float(np.mean(hot_mask[:20]))
        x = list(market_x[i])
        x += [hot_share_all, hot_share_top20, float(regimes[i] == "market_up_liq_up"), float(regimes[i] == "liq_up_high_dispersion"), float(regimes[i] == "market_down_liq_down"), float(regimes[i] == "market_down_price_only")]
        for w in [4, 8, 16, 32]:
            x += [rolling(past_base20, i, w), rolling(past_hot20, i, w), rolling(past_delta20, i, w), rolling(past_hot_share, i, w)]
        rows.append(x)
        past_base20.append(records[i]["base_top20"])
        past_hot20.append(records[i]["hot_top20"])
        past_delta20.append(records[i]["delta"]["delta_top20"])
        past_hot_share.append(records[i]["hot_share_all"])
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def train_ridge_classifier(x: np.ndarray, y: np.ndarray, mask: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    xt = x[mask].astype(float)
    yt = y[mask].astype(float)
    mu = np.mean(xt, axis=0)
    sd = np.std(xt, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    xs = (xt - mu) / sd
    xb = np.column_stack([np.ones(len(xs)), xs])
    coef = np.linalg.solve(xb.T @ xb + np.eye(xb.shape[1]) * alpha, xb.T @ (yt - np.mean(yt)))
    intercept = float(np.mean(yt) + coef[0])
    return {"mu": mu, "sd": sd, "coef": coef[1:], "intercept": np.asarray([intercept], dtype=float)}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (x - model["mu"]) / model["sd"]
    raw = xs @ model["coef"] + float(model["intercept"][0])
    return np.clip(raw, 0.0, 1.0)


def evaluate(cache: dict[str, Any], records: list[dict[str, Any]], prob: np.ndarray, threshold: float, boost: float, split_name: str) -> dict[str, Any]:
    mask = cache["split"] == split_name
    top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    hot_top = {k: [] for k in TOP_KS}
    by_year: dict[str, list[float]] = {}
    allowed = []
    overlap = []
    for i in np.flatnonzero(mask):
        allow = bool(prob[i] >= threshold)
        score = score_with_hot(cache["y"][i], cache["buckets"][i], cache["layers"][i], boost, allow)
        base = base_score(cache["y"].shape[1])
        hot = score_with_hot(cache["y"][i], cache["buckets"][i], cache["layers"][i], boost, True)
        order = np.argsort(-score, kind="mergesort")
        base_order = np.argsort(-base, kind="mergesort")
        overlap.append(len(set(order[:20]).intersection(set(base_order[:20]))) / 20.0)
        allowed.append(float(allow))
        for k in TOP_KS:
            v = safe_mean(cache["y"][i, order[:k]])
            top[k].append(v)
            base_top[k].append(top_return(cache["y"][i], base, k))
            hot_top[k].append(top_return(cache["y"][i], hot, k))
            if k == 20:
                by_year.setdefault(str(cache["years"][i]), []).append(v)
    return {
        "sessions": int(np.sum(mask)),
        "threshold": threshold,
        "allow_rate": safe_mean(allowed),
        "model_top": {f"top{k}": safe_mean(v) for k, v in top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "always_hot_top": {f"top{k}": safe_mean(v) for k, v in hot_top.items()},
        "top20_overlap_with_base": safe_mean(overlap),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def run_for_boost(cache: dict[str, Any], boost: float) -> dict[str, Any]:
    _, _, records = build_meta_dataset(cache, boost)
    x = feature_matrix(cache, records)
    labels = np.asarray([r["label"] for r in records], dtype=float)
    train_mask = cache["split"] == "train"
    valid_mask = cache["split"] == "valid"
    models = []
    for alpha in [1.0, 3.0, 10.0, 30.0, 100.0]:
        model = train_ridge_classifier(x, labels, train_mask, alpha)
        prob = predict(model, x)
        for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            valid = evaluate(cache, records, prob, th, boost, "valid")
            dev_years = evaluate(cache, records, prob, th, boost, "train")["by_year_model_top20"]
            min_train_year = min(dev_years.values()) if dev_years else -1.0
            score = valid["model_top"]["top20"] + 0.25 * valid["model_top"]["top10"] + 0.10 * min(0.0, min_train_year) - 0.001 * max(0.0, 0.90 - valid["top20_overlap_with_base"])
            models.append({"boost": boost, "alpha": alpha, "threshold": th, "valid_score": score, "model": model, "prob": prob, "valid": valid})
    selected = sorted(models, key=lambda r: r["valid_score"], reverse=True)[0]
    prob = selected["prob"]
    result = {
        "boost": boost,
        "alpha": selected["alpha"],
        "threshold": selected["threshold"],
        "valid_score": selected["valid_score"],
        "train": evaluate(cache, records, prob, selected["threshold"], boost, "train"),
        "valid": evaluate(cache, records, prob, selected["threshold"], boost, "valid"),
        "forward": evaluate(cache, records, prob, selected["threshold"], boost, "forward"),
        "label_rate": {name: safe_mean(labels[cache["split"] == name]) for name in ["train", "valid", "forward"]},
    }
    return result


def main() -> None:
    started = time.perf_counter()
    cache = load_cache()
    rows = [run_for_boost(cache, b) for b in [0.10, 0.15, 0.25, 0.40]]
    rows = sorted(rows, key=lambda r: r["valid_score"], reverse=True)
    selected = rows[0]
    base = selected["valid"]["base_top"]
    verdict = "session_phase_hot_continuation_meta_model_not_enough"
    dev_years = selected["train"]["by_year_model_top20"] | selected["valid"]["by_year_model_top20"]
    if (
        selected["forward"]["model_top"]["top20"] > selected["forward"]["base_top"]["top20"] + 0.004
        and selected["forward"]["model_top"]["top10"] >= selected["forward"]["base_top"]["top10"] * 0.95
        and min(dev_years.values()) > 0.0
    ):
        verdict = "session_phase_hot_continuation_meta_model_candidate_needs_replay"
    out = {
        "experiment": "session_phase_hot_continuation_meta_model_v1",
        "method": "cached_session_tensor_ridge_meta_router_for_hot_continuation",
        "params": {"boosts": [0.10, 0.15, 0.25, 0.40], "top_ks": TOP_KS, "cache": str(CACHE)},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round67_script": sha256(L67_PATH), "round68_summary": sha256(ROOT / "market_phase_hot_liquidity_router_v1_summary.json"), "cache": sha256(CACHE)},
        "sample_counts": {name: int(np.sum(cache["split"] == name)) for name in ["train", "valid", "forward"]},
        "cache_meta": json.loads(CACHE_META.read_text()) if CACHE_META.exists() else {},
        "leaderboard": rows,
        "selected": selected,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "sample_counts": out["sample_counts"], "selected": selected, "leaderboard": rows, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


if __name__ == "__main__":
    main()
