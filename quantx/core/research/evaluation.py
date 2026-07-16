"""Leakage-neutral evaluation for out-of-sample cross-sectional predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def aggregate_seed_metrics(metrics_by_seed: Mapping[int, Mapping[str, object]]) -> dict:
    """Aggregate scalar score metrics without merging the underlying predictions."""
    if not metrics_by_seed:
        raise ValueError("At least one seed result is required")
    common_keys = set.intersection(*(set(metrics) for metrics in metrics_by_seed.values()))
    aggregate = {}
    for key in sorted(common_keys):
        values = []
        for seed in sorted(metrics_by_seed):
            value = metrics_by_seed[seed][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                values = []
                break
            if not np.isfinite(value):
                values = []
                break
            values.append(float(value))
        if values:
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "worst": float(np.min(values)),
                "by_seed": {str(seed): float(metrics_by_seed[seed][key]) for seed in sorted(metrics_by_seed)},
            }
    return {"seed_count": len(metrics_by_seed), "metrics": aggregate}


def evaluate_predictions(frame: pd.DataFrame, *, quantiles: int = 5) -> dict:
    """Evaluate only stored OOS scores against theoretical labels."""
    required = {"score", "theoretical_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction evaluation missing columns: {sorted(missing)}")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")

    numeric = frame.copy()
    numeric["score"] = pd.to_numeric(numeric["score"], errors="coerce")
    numeric["theoretical_label"] = pd.to_numeric(numeric["theoretical_label"], errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = numeric[["score", "theoretical_label"]].dropna()
    total_count = len(numeric)
    score_count = int(numeric["score"].notna().sum())
    label_count = int(numeric["theoretical_label"].notna().sum())

    pooled_ic = _correlation(valid["score"], valid["theoretical_label"])
    pooled_rank_ic = _rank_correlation(valid["score"], valid["theoretical_label"])
    daily = _daily_correlations(valid)
    quantile_returns = _quantile_returns(valid, quantiles)

    result = {
        "sample_count": int(len(valid)),
        "total_prediction_rows": int(total_count),
        "prediction_coverage": _ratio(score_count, total_count),
        "label_coverage": _ratio(label_count, total_count),
        "joint_coverage": _ratio(len(valid), total_count),
        "ic": pooled_ic,
        "rank_ic": pooled_rank_ic,
        "daily_ic_mean": _mean_or_none(daily["ic"]),
        "daily_rank_ic_mean": _mean_or_none(daily["rank_ic"]),
        "icir": _information_ratio(daily["ic"]),
        "rank_icir": _information_ratio(daily["rank_ic"]),
        "evaluated_session_count": int(len(daily)),
        "score_mean": _mean_or_none(valid["score"]),
        "label_mean": _mean_or_none(valid["theoretical_label"]),
        "quantile_returns": quantile_returns,
        "by_fold": _group_metrics(numeric, "fold_id"),
        "by_year": _year_metrics(numeric),
    }
    return result


def _daily_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in frame.groupby(level="signal_time", sort=True):
        if len(group) < 2:
            continue
        ic = _correlation(group["score"], group["theoretical_label"])
        rank_ic = _rank_correlation(group["score"], group["theoretical_label"])
        if ic is not None or rank_ic is not None:
            rows.append({"signal_time": session, "ic": ic, "rank_ic": rank_ic})
    if not rows:
        return pd.DataFrame(columns=["ic", "rank_ic"])
    return pd.DataFrame(rows).set_index("signal_time")


def _quantile_returns(frame: pd.DataFrame, quantiles: int) -> dict:
    buckets: dict[int, list[float]] = {index: [] for index in range(1, quantiles + 1)}
    for _, group in frame.groupby(level="signal_time", sort=True):
        if len(group) < 2 or group["score"].nunique() < 2:
            continue
        bucket_count = min(quantiles, len(group), group["score"].nunique())
        ranks = group["score"].rank(method="first")
        assigned = pd.qcut(ranks, q=bucket_count, labels=False, duplicates="drop")
        for bucket, values in group.assign(_bucket=assigned).groupby("_bucket"):
            normalized_bucket = (
                1 if bucket_count == 1 else 1 + round(int(bucket) * (quantiles - 1) / (bucket_count - 1))
            )
            buckets[normalized_bucket].append(float(values["theoretical_label"].mean()))
    means = {str(bucket): _mean_or_none(values) for bucket, values in buckets.items()}
    bottom = means["1"]
    top = means[str(quantiles)]
    return {
        "bucket_count": quantiles,
        "mean_return_by_bucket": means,
        "top_minus_bottom": None if top is None or bottom is None else float(top - bottom),
    }


def _group_metrics(frame: pd.DataFrame, column: str) -> dict:
    if column not in frame.columns:
        return {}
    return {str(group_name): _compact_metrics(group) for group_name, group in frame.groupby(column, sort=True)}


def _year_metrics(frame: pd.DataFrame) -> dict:
    years = pd.DatetimeIndex(frame.index.get_level_values("signal_time")).year
    return {str(year): _compact_metrics(frame[years == year]) for year in sorted(set(int(value) for value in years))}


def _compact_metrics(frame: pd.DataFrame) -> dict:
    valid = frame[["score", "theoretical_label"]].dropna()
    daily = _daily_correlations(valid)
    return {
        "sample_count": int(len(valid)),
        "ic": _correlation(valid["score"], valid["theoretical_label"]),
        "rank_ic": _rank_correlation(valid["score"], valid["theoretical_label"]),
        "daily_ic_mean": _mean_or_none(daily["ic"]),
        "daily_rank_ic_mean": _mean_or_none(daily["rank_ic"]),
        "icir": _information_ratio(daily["ic"]),
        "rank_icir": _information_ratio(daily["rank_ic"]),
    }


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    return _finite_or_none(left.corr(right))


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    return _correlation(left.rank(method="average"), right.rank(method="average"))


def _information_ratio(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    standard_deviation = float(clean.std(ddof=1))
    if standard_deviation <= 1e-12:
        return None
    return float(clean.mean() / standard_deviation * math.sqrt(252.0))


def _mean_or_none(values) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return None
    return _finite_or_none(clean.mean())


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _finite_or_none(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
