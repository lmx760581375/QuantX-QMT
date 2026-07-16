from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor


BASE_ANALYZER = Path(".tmp/quantx-research/qmt-factor-expert-timing-v1/analyze_qmt_factor_expert_timing.py")
TOPKS = (10, 15, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_base()
    panel = base.build_panel(Path(args.provider), Path(args.raw_dir), args.start, args.end, args.load_start, args.horizon, args.sample_step)
    panel = add_features(panel, list(base.EXPERTS))
    detail, folds, importances = walk_forward(panel, args.min_train_rows)
    result = summarize(detail)
    result["folds"] = folds
    result["feature_importance"] = summarize_importance(importances)
    result["config"] = {
        "provider": args.provider,
        "raw_dir": args.raw_dir,
        "start": args.start,
        "end": args.end,
        "load_start": args.load_start,
        "horizon": args.horizon,
        "sample_step": args.sample_step,
        "panel_rows": int(len(panel)),
        "sessions": int(panel["session"].nunique()) if not panel.empty else 0,
        "features": feature_columns(list(base.EXPERTS)),
        "causality": "Features are QMT daily expert ranks through completed T only. Labels use T+1 open to T+6 open. Annual folds train only on years strictly before evaluated year.",
        "scope": "QMT retrievable daily market data only; no news, announcements, LHB, ETF flow, northbound, financing, or event/message data.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def load_base():
    spec = importlib.util.spec_from_file_location("qmt_factor_expert_timing", BASE_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_columns(experts: list[str]) -> list[str]:
    interactions = [
        "low_vol20_x_trend60",
        "low_vol20_x_amount_leader",
        "quiet_trend_x_amount_leader",
        "trend60_x_amount_leader",
        "trend60_x_volume_surge",
        "pullback20_x_amount_leader",
        "pullback20_x_low_vol20",
        "near_high20_x_volume_surge",
        "breakout20_x_amount_leader",
        "vwap_reclaim_x_close_strength",
        "range_expansion_x_volume_surge",
        "liquidity_dispersion_x_trend60",
    ]
    return experts + interactions


def add_features(panel: pd.DataFrame, experts: list[str]) -> pd.DataFrame:
    out = panel.copy()
    pairs = {
        "low_vol20_x_trend60": ("low_vol20", "trend60"),
        "low_vol20_x_amount_leader": ("low_vol20", "amount_leader"),
        "quiet_trend_x_amount_leader": ("quiet_trend", "amount_leader"),
        "trend60_x_amount_leader": ("trend60", "amount_leader"),
        "trend60_x_volume_surge": ("trend60", "volume_surge"),
        "pullback20_x_amount_leader": ("pullback20", "amount_leader"),
        "pullback20_x_low_vol20": ("pullback20", "low_vol20"),
        "near_high20_x_volume_surge": ("near_high20", "volume_surge"),
        "breakout20_x_amount_leader": ("breakout20", "amount_leader"),
        "vwap_reclaim_x_close_strength": ("vwap_reclaim", "close_strength"),
        "range_expansion_x_volume_surge": ("range_expansion", "volume_surge"),
        "liquidity_dispersion_x_trend60": ("liquidity_dispersion_leader", "trend60"),
    }
    for name, (a, b) in pairs.items():
        out[name] = out[a].astype(float) * out[b].astype(float)
    for session, idx in out.groupby("session").groups.items():
        labels = out.loc[idx, "label5"]
        out.loc[idx, "label_quintile"] = pd.qcut(labels.rank(method="first"), 5, labels=False).astype(int).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, int]], list[dict[str, float]]]:
    frames: list[pd.DataFrame] = []
    folds: list[dict[str, int]] = []
    importances: list[dict[str, float]] = []
    experts = [col for col in panel.columns if col not in {"session", "exit_date", "year", "instrument", "label5", "raw5", "label_quintile"}]
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        scored, imp = fit_predict(train, test, experts, seed=int(year))
        frames.append(evaluate(scored))
        importances.append(imp)
        fold = {"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())}
        folds.append(fold)
        print(json.dumps(fold, ensure_ascii=False), flush=True)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), folds, importances


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    reg = LGBMRegressor(n_estimators=260, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=6, verbosity=-1)
    ranker = LGBMRanker(objective="lambdarank", n_estimators=220, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=400, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=5.0, random_state=seed + 17, n_jobs=6, verbosity=-1)
    reg.fit(train[features], train["label5"])
    sorted_train = train.sort_values(["session", "instrument"])
    groups = sorted_train.groupby("session", sort=False).size().to_numpy(dtype=int)
    ranker.fit(sorted_train[features], sorted_train["label_quintile"].astype(int), group=groups)
    out = test.copy()
    out["score_reg"] = reg.predict(out[features])
    out["score_rank"] = ranker.predict(out[features])
    out["rank_reg"] = out.groupby("session")["score_reg"].rank(pct=True, method="first")
    out["rank_rank"] = out.groupby("session")["score_rank"].rank(pct=True, method="first")
    out["score_blend"] = 0.5 * out["rank_reg"] + 0.5 * out["rank_rank"]
    imp = {f"reg::{name}": float(value) for name, value in zip(features, reg.feature_importances_)}
    imp.update({f"rank::{name}": float(value) for name, value in zip(features, ranker.feature_importances_)})
    return out, imp


def evaluate(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = (("reg", "score_reg"), ("rank", "score_rank"), ("blend", "score_blend"))
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants:
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                if len(selected) < topk:
                    continue
                rows.append({"session": session, "year": int(str(session)[:4]), "variant": variant, "topk": int(topk), "mean_label5": float(selected["label5"].mean()), "mean_raw5": float(selected["raw5"].mean()), "positive_label5": bool(selected["label5"].mean() > 0), "count": int(len(selected))})
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"daily_row_count": int(len(detail)), "results": {}}
    if detail.empty:
        return result
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        key = f"{variant}::top{int(topk)}"
        result["results"][key] = {"daily_count": int(len(group)), "mean_daily_label5": float(group["mean_label5"].mean()), "mean_daily_raw5": float(group["mean_raw5"].mean()), "positive_label5_ratio": float(group["positive_label5"].mean()), "avg_selected_count": float(group["count"].mean()), "by_year": {str(int(y)): {"mean_daily_label5": float(yg["mean_label5"].mean()), "mean_daily_raw5": float(yg["mean_raw5"].mean()), "positive_label5_ratio": float(yg["positive_label5"].mean())} for y, yg in group.groupby("year")}}
    return result


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, float | str]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:50]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    rows = []
    for key, value in result.get("results", {}).items():
        if key.endswith("::top20"):
            rows.append((float(value["mean_daily_label5"]), float(value["mean_daily_raw5"]), key, value["by_year"]))
    rows.sort(key=lambda item: item[0], reverse=True)
    print(json.dumps({"checksum": checksum, "daily_row_count": result.get("daily_row_count"), "top20_results": rows, "top_features": result.get("feature_importance", [])[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
