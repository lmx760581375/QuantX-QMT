"""Daily market-state gate for the range-regime right-tail experiment.

The gate is evaluated by trading day, not by stock row. It reuses the second-stage
tail predictions and writes only under ./tmp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TOPN_VALUES = (5, 20, 50)
FEATURE_DIRECTIONS = {
    "p_bad_top20_mean": "le",
    "p_bad_top20_q75": "le",
    "p_bad_top20_max": "le",
    "amount_slope": "ge",
    "breadth_gap": "ge",
    "raw_candidate_count_mean": "le",
    "candidate_count": "le",
    "tail_score_gap_top20": "ge",
    "dispersion_tail_score": "ge",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tmp/market_regime_weak_to_strong_alpha_v1/configs/experiment.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(cfg["output_root"])
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    run_gate_experiment(output_root)


def run_gate_experiment(output_root: Path) -> None:
    daily = build_daily_panel(output_root)
    daily.to_csv(output_root / "runs" / "gate_daily_panel.csv", index=False)

    baseline_rows = []
    for topn in TOPN_VALUES:
        for year in range(2022, 2027):
            period = "forward" if year == 2026 else "oos"
            year_frame = daily[daily["year"] == year]
            row = summary_row(year_frame, "no_gate", topn, total_day_count=len(year_frame))
            row.update({"period": period, "year": year})
            baseline_rows.append(row)
        oos_frame = daily[daily["year"].between(2022, 2025)]
        row = summary_row(oos_frame, "no_gate", topn, total_day_count=len(oos_frame))
        row.update({"period": "oos_2022_2025", "year": "2022-2025"})
        baseline_rows.append(row)
        forward_frame = daily[daily["year"] == 2026]
        row = summary_row(forward_frame, "no_gate", topn, total_day_count=len(forward_frame))
        row.update({"period": "forward_summary", "year": 2026})
        baseline_rows.append(row)
    threshold = threshold_grid(daily)
    model_summary, importance = model_gate(daily)
    summary = pd.concat([pd.DataFrame(baseline_rows), threshold, model_summary], ignore_index=True)

    summary.to_csv(output_root / "runs" / "gate_summary.csv", index=False)
    threshold.to_csv(output_root / "runs" / "gate_threshold_grid.csv", index=False)
    importance.to_csv(output_root / "runs" / "gate_feature_importance.csv", index=False)
    print(f"[gate] wrote daily={len(daily)} summary={len(summary)} threshold={len(threshold)}", flush=True)


def build_daily_panel(output_root: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(output_root / "runs" / "tail_predictions.parquet")
    predictions = predictions[
        (predictions["layer"] == "wide")
        & (predictions["target_name"] == "tail_top20")
        & (predictions["bad_name"] == "bad_loss3")
    ].copy()
    predictions["signal_time"] = pd.to_datetime(predictions["signal_time"])
    pred = (
        predictions.groupby(["signal_time", "instrument", "layer", "prediction_year"], as_index=False)
        .agg(ret_5d=("ret_5d", "first"), p_tail=("p_tail", "mean"), p_bad=("p_bad", "mean"), tradable_rank=("tradable_rank", "first"))
    )
    pred["tail_score"] = pred["p_tail"] - 1.5 * pred["p_bad"]

    candidate_cols = [
        "signal_time",
        "instrument",
        "layer",
        "amount_slope",
        "breadth_gap",
        "raw_candidate_count",
        "candidate_rank",
        "layer_score",
    ]
    candidates = pd.read_parquet(output_root / "data" / "candidate_panel.parquet", columns=candidate_cols)
    candidates = candidates[(candidates["layer"] == "wide")].copy()
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"])
    pred = pred.merge(candidates, on=["signal_time", "instrument", "layer"], how="left")

    rows = []
    for session, group in pred.groupby("signal_time", sort=True):
        group = group.sort_values("tail_score", ascending=False)
        row: dict[str, Any] = {
            "signal_time": session.strftime("%Y-%m-%d"),
            "year": int(group["prediction_year"].iloc[0]),
            "candidate_count": int(len(group)),
            "amount_slope": first_float(group["amount_slope"]),
            "breadth_gap": first_float(group["breadth_gap"]),
            "raw_candidate_count_mean": mean_float(group["raw_candidate_count"]),
            "raw_candidate_count_max": max_float(group["raw_candidate_count"]),
            "layer_score_mean": mean_float(group["layer_score"]),
            "layer_score_q75": quantile_float(group["layer_score"], 0.75),
            "layer_score_max": max_float(group["layer_score"]),
            "candidate_rank_mean": mean_float(group["candidate_rank"]),
            "p_tail_all_mean": mean_float(group["p_tail"]),
            "p_tail_all_q75": quantile_float(group["p_tail"], 0.75),
            "p_tail_all_max": max_float(group["p_tail"]),
            "p_bad_all_mean": mean_float(group["p_bad"]),
            "p_bad_all_q75": quantile_float(group["p_bad"], 0.75),
            "p_bad_all_max": max_float(group["p_bad"]),
            "tail_score_all_mean": mean_float(group["tail_score"]),
            "tail_score_all_q10": quantile_float(group["tail_score"], 0.10),
            "tail_score_all_q90": quantile_float(group["tail_score"], 0.90),
            "tail_score_all_std": std_float(group["tail_score"]),
        }
        row["dispersion_tail_score"] = row["tail_score_all_q90"] - row["tail_score_all_q10"]
        for topn in TOPN_VALUES:
            selected = group.head(topn)
            ret_col = f"daily_ret_top{topn}"
            row[ret_col] = mean_float(selected["ret_5d"])
            row[f"daily_positive_top{topn}"] = bool(row[ret_col] > 0) if np.isfinite(row[ret_col]) else False
            row[f"daily_bad0_top{topn}"] = bool(row[ret_col] < 0) if np.isfinite(row[ret_col]) else False
            row[f"daily_bad1_top{topn}"] = bool(row[ret_col] < -0.01) if np.isfinite(row[ret_col]) else False
            row[f"top{topn}_count"] = int(len(selected))
            for prefix, col in (("p_tail", "p_tail"), ("p_bad", "p_bad"), ("tail_score", "tail_score")):
                row[f"{prefix}_top{topn}_mean"] = mean_float(selected[col])
                row[f"{prefix}_top{topn}_q75"] = quantile_float(selected[col], 0.75)
                row[f"{prefix}_top{topn}_max"] = max_float(selected[col])
                row[f"{prefix}_top{topn}_min"] = min_float(selected[col])
            row[f"p_bad_gap_top{topn}"] = row["p_bad_all_mean"] - row[f"p_bad_top{topn}_mean"]
            row[f"tail_score_gap_top{topn}"] = row[f"tail_score_top{topn}_mean"] - row["tail_score_all_mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def threshold_grid(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quantiles = (0.3, 0.5, 0.7)
    for feature, direction in FEATURE_DIRECTIONS.items():
        for q in quantiles:
            rows.extend(evaluate_expanding_threshold(daily, [(feature, direction, q)], f"threshold:{feature}:{direction}:q{q}"))
    combos = [
        [("p_bad_top20_mean", "le", 0.5), ("breadth_gap", "ge", 0.5)],
        [("p_bad_top20_q75", "le", 0.7), ("amount_slope", "ge", 0.3)],
        [("tail_score_gap_top20", "ge", 0.5), ("p_bad_top20_mean", "le", 0.7)],
        [("raw_candidate_count_mean", "le", 0.7), ("breadth_gap", "ge", 0.3)],
    ]
    for idx, combo in enumerate(combos, start=1):
        name = "combo" + str(idx) + ":" + "+".join(f"{f}{d}q{q}" for f, d, q in combo)
        rows.extend(evaluate_expanding_threshold(daily, combo, name))
    return pd.DataFrame(rows)


def evaluate_expanding_threshold(daily: pd.DataFrame, rules: list[tuple[str, str, float]], name: str) -> list[dict[str, Any]]:
    rows = []
    for pred_year in range(2022, 2027):
        train = daily[daily["year"] < pred_year]
        test = daily[daily["year"] == pred_year]
        if train.empty or test.empty:
            continue
        keep = pd.Series(True, index=test.index)
        threshold_values = []
        for feature, direction, quantile in rules:
            threshold = float(train[feature].quantile(quantile))
            threshold_values.append(f"{feature}{direction}{threshold:.6g}")
            if direction == "le":
                keep &= test[feature] <= threshold
            else:
                keep &= test[feature] >= threshold
        selected = test[keep]
        period = "forward" if pred_year == 2026 else "oos"
        for topn in TOPN_VALUES:
            row = summary_row(selected, name, topn, total_day_count=len(test))
            row.update({"period": period, "year": pred_year, "thresholds": ";".join(threshold_values)})
            rows.append(row)
    rows.extend(aggregate_year_rows(rows, name))
    return rows


def model_gate(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMClassifier

    feature_cols = gate_feature_columns(daily)
    params = {
        "n_estimators": 120,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 12,
        "reg_lambda": 0.5,
        "n_jobs": 4,
        "verbosity": -1,
        "class_weight": "balanced",
    }
    keep_quantiles = (0.3, 0.5, 0.7, 0.9)
    rows = []
    importance_rows = []
    for pred_year in range(2022, 2027):
        train = daily[daily["year"] < pred_year].copy()
        test = daily[daily["year"] == pred_year].copy()
        if train.empty or test.empty:
            continue
        train["bad_day"] = (train["daily_ret_top20"] < -0.01).astype(int)
        test["bad_day"] = (test["daily_ret_top20"] < -0.01).astype(int)
        x_train, medians = feature_matrix(train, feature_cols)
        x_test, _ = feature_matrix(test, feature_cols, medians)
        test = test.copy()
        if train["bad_day"].nunique() < 2:
            constant_score = float(train["bad_day"].iloc[0])
            train_score = np.full(len(train), constant_score, dtype=float)
            test["p_bad_day"] = constant_score
        else:
            model = LGBMClassifier(random_state=20260715 + pred_year, **params)
            model.fit(x_train, train["bad_day"].to_numpy())
            train_score = model.predict_proba(x_train)[:, 1]
            test["p_bad_day"] = model.predict_proba(x_test)[:, 1]
            importance_rows.extend(feature_importance_rows(model, feature_cols, pred_year))
        period = "forward" if pred_year == 2026 else "oos"
        for q in keep_quantiles:
            threshold = float(np.quantile(train_score, q))
            selected = test[test["p_bad_day"] <= threshold]
            for topn in TOPN_VALUES:
                row = summary_row(selected, f"lgbm_bad_day_le_train_q{q}", topn, total_day_count=len(test))
                row.update({"period": period, "year": pred_year, "thresholds": f"p_bad_day<={threshold:.6g}"})
                rows.append(row)
    rows.extend(aggregate_year_rows(rows, "lgbm"))
    return pd.DataFrame(rows), pd.DataFrame(importance_rows)


def aggregate_year_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out = []
    for (method, topn, period), group in frame.groupby(["method", "topn", "period"], sort=True):
        if period == "oos":
            valid = group[group["year"].between(2022, 2025)]
            years = valid["year"].nunique()
            if years == 0:
                continue
            row = {
                "method": method,
                "topn": int(topn),
                "period": "oos_2022_2025",
                "year": "2022-2025",
                "trade_day_count": int(valid["trade_day_count"].sum()),
                "total_day_count": int(valid["total_day_count"].sum()),
                "coverage_ratio": weighted_mean(valid, "coverage_ratio", "total_day_count"),
                "daily_mean_return": weighted_mean(valid, "daily_mean_return", "trade_day_count"),
                "positive_day_rate": weighted_mean(valid, "positive_day_rate", "trade_day_count"),
                "bad1_day_rate": weighted_mean(valid, "bad1_day_rate", "trade_day_count"),
                "worst_year_return": float(valid["daily_mean_return"].min()),
                "pos_years": int((valid["daily_mean_return"] > 0).sum()),
                "max_losing_streak": int(valid["max_losing_streak"].max()),
                "thresholds": prefix,
            }
            out.append(row)
    return out


def summary_row(frame: pd.DataFrame, method: str, topn: int, total_day_count: int | None = None) -> dict[str, Any]:
    ret_col = f"daily_ret_top{topn}"
    values = pd.to_numeric(frame[ret_col], errors="coerce").dropna() if not frame.empty else pd.Series(dtype=float)
    total = int(len(frame) if total_day_count is None else total_day_count)
    return {
        "method": method,
        "topn": int(topn),
        "period": "summary",
        "year": "all",
        "trade_day_count": int(len(values)),
        "total_day_count": total,
        "coverage_ratio": None if total == 0 else float(len(values) / total),
        "daily_mean_return": mean_or_none(values),
        "median_return": quantile_or_none(values, 0.5),
        "positive_day_rate": mean_or_none(values > 0),
        "bad0_day_rate": mean_or_none(values < 0),
        "bad1_day_rate": mean_or_none(values < -0.01),
        "max_losing_streak": max_losing_streak(values),
        "worst_year_return": None,
        "pos_years": None,
        "thresholds": "",
    }


def gate_feature_columns(daily: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("daily_ret_", "daily_positive_", "daily_bad0_", "daily_bad1_")
    excluded = {"signal_time", "year"}
    return [
        col
        for col in daily.columns
        if col not in excluded
        and not col.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(daily[col])
        and np.isfinite(pd.to_numeric(daily[col], errors="coerce")).any()
    ]


def feature_matrix(frame: pd.DataFrame, cols: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, cols].to_numpy(dtype=float, copy=True)
    if medians is None:
        medians = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    return values, medians


def feature_importance_rows(model: Any, features: list[str], pred_year: int) -> list[dict[str, Any]]:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    return [
        {"prediction_year": pred_year, "feature": feature, "gain": float(g), "split": int(s)}
        for feature, g, s in zip(features, gain, split)
    ]


def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    valid = frame[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return None
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def max_losing_streak(values: pd.Series) -> int:
    streak = 0
    best = 0
    for value in values:
        if value < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def first_float(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def mean_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def std_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").std())


def min_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").min())


def max_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").max())


def quantile_float(series: pd.Series, q: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(q))


def mean_or_none(values: Any) -> float | None:
    series = pd.Series(values).dropna()
    return None if series.empty else float(series.mean())


def quantile_or_none(values: Any, q: float) -> float | None:
    series = pd.Series(values).dropna()
    return None if series.empty else float(series.quantile(q))


if __name__ == "__main__":
    main()
