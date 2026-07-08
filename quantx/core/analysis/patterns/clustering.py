"""Clustering and similar-sample helpers for trade patterns."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .features import train_feature_columns


def run_cluster_analysis(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Cluster trade windows with available train-time features."""
    if features.empty:
        return pd.DataFrame(), {"ok": False, "reason": "empty features"}
    if importlib.util.find_spec("sklearn") is None:
        return pd.DataFrame(), {"ok": False, "reason": "scikit-learn is not installed"}

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = train_feature_columns(features)
    if not columns:
        return pd.DataFrame(), {"ok": False, "reason": "no train feature columns"}
    sample_count = len(features)
    cluster_count = min(max(2, int(n_clusters)), sample_count)
    if sample_count < 2:
        return pd.DataFrame(), {"ok": False, "reason": "not enough samples"}

    x = features[columns]
    transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    matrix = transformer.fit_transform(x)
    pca_dims = min(2, matrix.shape[1], matrix.shape[0])
    coords = PCA(n_components=pca_dims, random_state=random_state).fit_transform(matrix)
    if pca_dims == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    clusters = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(matrix)

    result = features[["sample_id", "symbol", "entry_date", "exit_date", "return", "max_favorable_excursion", "max_adverse_excursion"]].copy()
    result["cluster_id"] = [f"pattern_cluster_{int(label) + 1:03d}" for label in clusters]
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]
    if not labels.empty:
        keep = [
            column
            for column in [
                "sample_id",
                "outcome_label",
                "behavior_label",
                "binary_success",
                "early_confirmed",
                "early_failed",
                "opportunity_label",
                "had_opportunity",
                "fade_label",
                "faded_after_peak",
                "efficient_capture",
                "first_3d_return",
                "first_5d_return",
                "drawdown_after_peak",
            ]
            if column in labels.columns
        ]
        result = result.merge(labels[keep], on="sample_id", how="left")

    summary = _cluster_summary(result, features, columns)
    return result, {"ok": True, "feature_columns": columns, "clusters": summary}


def _cluster_summary(cluster_rows: pd.DataFrame, features: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    merged = cluster_rows[["sample_id", "cluster_id"]].merge(features[["sample_id", *columns]], on="sample_id", how="left")
    global_means = merged[columns].mean(numeric_only=True)
    rows: List[Dict[str, Any]] = []
    for cluster_id, group in cluster_rows.groupby("cluster_id"):
        feature_group = merged[merged["cluster_id"] == cluster_id]
        cluster_means = feature_group[columns].mean(numeric_only=True)
        diff = (cluster_means - global_means).abs().sort_values(ascending=False).head(8)
        rows.append({
            "cluster_id": cluster_id,
            "sample_count": int(len(group)),
            "success_rate": float(group["binary_success"].mean()) if "binary_success" in group else None,
            "avg_return": float(group["return"].mean()),
            "median_return": float(group["return"].median()),
            "avg_mfe": float(group["max_favorable_excursion"].mean()),
            "avg_mae": float(group["max_adverse_excursion"].mean()),
            "early_confirm_rate": float(group["early_confirmed"].mean()) if "early_confirmed" in group else None,
            "opportunity_rate": float(group["had_opportunity"].mean()) if "had_opportunity" in group else None,
            "fade_after_peak_rate": float(group["faded_after_peak"].mean()) if "faded_after_peak" in group else None,
            "efficient_capture_rate": float(group["efficient_capture"].mean()) if "efficient_capture" in group else None,
            "top_features": [{"feature": name, "mean": float(cluster_means[name]), "global_mean": float(global_means[name])} for name in diff.index],
            "description": _describe_cluster(cluster_means),
            "representative_samples": group.sort_values("return", ascending=False).head(10)["sample_id"].tolist(),
            "counter_examples": group.sort_values("return", ascending=True).head(10)["sample_id"].tolist(),
        })
    return sorted(rows, key=lambda item: item["cluster_id"])


def _describe_cluster(means: pd.Series) -> str:
    parts: List[str] = []
    if means.get("pre_20_volatility", 0) < 0.025:
        parts.append("低波动")
    if means.get("pre_20_return", 0) > 0.08:
        parts.append("前期上涨")
    elif means.get("pre_20_return", 0) < -0.05:
        parts.append("前期回调")
    if means.get("entry_volume_ratio_20", 0) > 1.8:
        parts.append("放量")
    elif means.get("entry_volume_ratio_20", 0) < 0.8:
        parts.append("缩量")
    if means.get("entry_close_position", 0.5) > 0.75:
        parts.append("强收盘")
    if means.get("entry_upper_shadow_ratio", 0) > 0.35:
        parts.append("上影偏长")
    if means.get("entry_distance_to_ma20", 0) > 0.12:
        parts.append("远离MA20")
    return "_".join(parts) if parts else "待命名形态"
