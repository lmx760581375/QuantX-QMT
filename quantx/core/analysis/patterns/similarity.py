"""Similar trade-window retrieval for built pattern analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .features import train_feature_columns
from .io import read_table


def find_similar_samples(
    analysis_dir: str | Path,
    sample_id: str,
    top_k: int = 20,
    same_cluster_only: bool = False,
) -> Dict[str, Any]:
    """Find historical samples closest to a sample in train-time feature space."""
    analysis_path = Path(analysis_dir)
    features = read_table(analysis_path / "window_features.parquet")
    labels = _optional_table(analysis_path / "labels.parquet")
    clusters = _optional_table(analysis_path / "cluster_results.parquet")
    if features.empty:
        raise ValueError(f"No window features found under {analysis_path}")
    if sample_id not in set(features["sample_id"].astype(str)):
        raise ValueError(f"sample_id not found: {sample_id}")

    data = features.copy()
    data["sample_id"] = data["sample_id"].astype(str)
    if not labels.empty:
        labels = labels.copy()
        labels["sample_id"] = labels["sample_id"].astype(str)
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
        data = data.merge(labels[keep], on="sample_id", how="left")
    if not clusters.empty:
        clusters = clusters.copy()
        clusters["sample_id"] = clusters["sample_id"].astype(str)
        keep = [column for column in ["sample_id", "cluster_id", "pca_x", "pca_y"] if column in clusters.columns]
        data = data.merge(clusters[keep], on="sample_id", how="left")

    target = data[data["sample_id"] == sample_id].iloc[0]
    candidates = data[data["sample_id"] != sample_id].copy()
    if same_cluster_only and "cluster_id" in data.columns and pd.notna(target.get("cluster_id")):
        candidates = candidates[candidates["cluster_id"] == target.get("cluster_id")]
    columns = train_feature_columns(features)
    if not columns:
        raise ValueError("No train-time feature columns available for similarity search")
    if candidates.empty:
        return {"sample_id": sample_id, "target": _row_summary(target), "top_k": top_k, "matches": []}

    matrix = features.set_index(features["sample_id"].astype(str))[columns].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median(numeric_only=True)
    matrix = matrix.fillna(medians).fillna(0.0)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    scaled = (matrix - means) / stds
    target_vector = scaled.loc[sample_id]
    candidate_ids = candidates["sample_id"].astype(str).tolist()
    distances = ((scaled.loc[candidate_ids] - target_vector) ** 2).sum(axis=1).pow(0.5)
    ordered_ids = distances.sort_values().head(max(int(top_k), 1)).index.tolist()

    by_id = data.set_index("sample_id", drop=False)
    matches: List[Dict[str, Any]] = []
    for match_id in ordered_ids:
        row = by_id.loc[match_id]
        item = _row_summary(row)
        item["distance"] = float(distances.loc[match_id])
        matches.append(item)
    return {
        "sample_id": sample_id,
        "top_k": int(top_k),
        "same_cluster_only": bool(same_cluster_only),
        "feature_count": len(columns),
        "target": _row_summary(target),
        "matches": matches,
    }


def _optional_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _row_summary(row: pd.Series) -> Dict[str, Any]:
    fields = [
        "sample_id",
        "run_id",
        "strategy_name",
        "symbol",
        "entry_date",
        "exit_date",
        "return",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "outcome_label",
        "behavior_label",
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
        "cluster_id",
    ]
    result: Dict[str, Any] = {}
    for field in fields:
        if field in row.index:
            value = row[field]
            if pd.isna(value):
                result[field] = None
            elif isinstance(value, (bool, np.bool_)):
                result[field] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                result[field] = value.item()
            else:
                result[field] = str(value) if field.endswith("date") else value
    return result
