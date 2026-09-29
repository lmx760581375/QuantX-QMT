"""Shared, auditable market-score artifact contract for QuantX research."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ARTIFACT_KIND = "quantx_market_score_artifact_v1"
CORE_COLUMNS = ("signal_date", "instrument")


class MarketScoreArtifactError(ValueError):
    """Raised when an artifact cannot be safely consumed by QuantX."""


def parse_score_columns(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(item).strip() for item in value]
    out = [item for item in values if item]
    if not out:
        raise MarketScoreArtifactError("At least one score column is required")
    if len(set(out)) != len(out):
        raise MarketScoreArtifactError(f"Duplicate score columns: {out}")
    return out


def normalize_score_frame(frame: pd.DataFrame, score_columns: str | Iterable[str]) -> pd.DataFrame:
    """Return a QuantX-safe score frame and reject silent key corruption."""

    score_columns = parse_score_columns(score_columns)
    required = [*CORE_COLUMNS, *score_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MarketScoreArtifactError(f"Score artifact is missing required columns: {missing}")
    if frame.empty:
        raise MarketScoreArtifactError("Score artifact has no rows")

    out = frame.copy()
    dates = pd.to_datetime(out["signal_date"], errors="coerce")
    if dates.isna().any():
        bad = out.loc[dates.isna(), "signal_date"].head(5).tolist()
        raise MarketScoreArtifactError(f"Invalid signal_date values: {bad}")
    out["signal_date"] = dates.dt.strftime("%Y-%m-%d")
    out["instrument"] = out["instrument"].astype(str).str.strip()
    if out["instrument"].eq("").any():
        raise MarketScoreArtifactError("Score artifact contains empty instrument identifiers")
    if out.duplicated(list(CORE_COLUMNS)).any():
        examples = out.loc[out.duplicated(list(CORE_COLUMNS), keep=False), list(CORE_COLUMNS)].head(5)
        raise MarketScoreArtifactError(f"Duplicate score keys detected: {examples.to_dict(orient='records')}")

    for column in score_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float32)
        if not out[column].notna().any():
            raise MarketScoreArtifactError(f"Score column has no usable values: {column}")
    return out


def artifact_manifest_path(score_path: str | Path) -> Path:
    return Path(score_path).with_suffix(".json")


def summarize_score_frame(frame: pd.DataFrame, score_columns: str | Iterable[str]) -> dict[str, Any]:
    score_columns = parse_score_columns(score_columns)
    coverage = {}
    for column in score_columns:
        usable = frame.loc[frame[column].notna(), ["signal_date", "instrument", column]]
        coverage[column] = {
            "rows": int(len(usable)),
            "dates": int(usable["signal_date"].nunique()),
            "instruments": int(usable["instrument"].nunique()),
            "start": str(usable["signal_date"].min()) if not usable.empty else None,
            "end": str(usable["signal_date"].max()) if not usable.empty else None,
        }
    return {
        "rows": int(len(frame)),
        "dates": int(frame["signal_date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "start": str(frame["signal_date"].min()),
        "end": str(frame["signal_date"].max()),
        "score_coverage": coverage,
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_market_score_artifact(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    score_columns: str | Iterable[str],
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a validated parquet and its mandatory manifest atomically."""

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".parquet", ".pq"}:
        raise MarketScoreArtifactError(f"Score artifact must be parquet: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing score artifact: {output}")

    normalized = normalize_score_frame(frame, score_columns)
    columns = parse_score_columns(score_columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    normalized.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)

    manifest = {
        "kind": ARTIFACT_KIND,
        "schema_version": 1,
        "status": "exported",
        "score_path": str(output),
        "score_columns": columns,
        "core_columns": list(CORE_COLUMNS),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "file_size_bytes": int(output.stat().st_size),
        **summarize_score_frame(normalized, columns),
        "metadata": metadata or {},
    }
    _atomic_json_write(artifact_manifest_path(output), manifest)
    return manifest


def validate_market_score_artifact(
    score_path: str | Path,
    *,
    score_columns: str | Iterable[str] | None = None,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Validate a persisted artifact before it is allowed into QuantX."""

    path = Path(score_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing score artifact: {path}")
    manifest_path = artifact_manifest_path(path)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif require_manifest:
        raise FileNotFoundError(f"Missing mandatory score manifest: {manifest_path}")

    requested_columns = score_columns is not None
    declared_columns = score_columns or manifest.get("score_columns")
    if not declared_columns:
        raise MarketScoreArtifactError("Score columns were not supplied and are absent from the manifest")
    columns = parse_score_columns(declared_columns)
    frame = pd.read_parquet(path)
    normalized = normalize_score_frame(frame, columns)
    summary = summarize_score_frame(normalized, columns)
    if manifest:
        if manifest.get("kind") != ARTIFACT_KIND:
            raise MarketScoreArtifactError(f"Unexpected artifact kind: {manifest.get('kind')}")
        if int(manifest.get("rows", -1)) != summary["rows"]:
            raise MarketScoreArtifactError("Manifest row count does not match parquet")
        manifest_columns = set(manifest.get("score_columns") or [])
        if requested_columns:
            if not set(columns).issubset(manifest_columns):
                raise MarketScoreArtifactError("Manifest does not declare every requested score column")
        elif list(manifest.get("score_columns") or []) != columns:
            raise MarketScoreArtifactError("Manifest score columns do not match parquet")
    return {
        "ok": True,
        "score_path": str(path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "score_columns": columns,
        **summary,
    }


def merge_market_score_artifacts(
    historical_path: str | Path,
    incremental_path: str | Path,
    output_path: str | Path,
    *,
    cutoff_date: str,
    score_columns: str | Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """按固定日期边界合并历史与增量 score，并重新生成完整 manifest。"""

    historical_validation = validate_market_score_artifact(
        historical_path,
        score_columns=score_columns,
        require_manifest=True,
    )
    columns = list(historical_validation["score_columns"])
    incremental_validation = validate_market_score_artifact(
        incremental_path,
        score_columns=columns,
        require_manifest=True,
    )
    cutoff = pd.Timestamp(cutoff_date).strftime("%Y-%m-%d")
    required = [*CORE_COLUMNS, *columns]
    historical_all = normalize_score_frame(pd.read_parquet(historical_path, columns=required), columns)
    incremental_all = normalize_score_frame(pd.read_parquet(incremental_path, columns=required), columns)
    overlap_dates = sorted(
        set(historical_all["signal_date"])
        & set(incremental_all.loc[incremental_all["signal_date"] <= cutoff, "signal_date"])
    )
    overlap_validation = None
    if overlap_dates:
        historical_overlap = historical_all.loc[historical_all["signal_date"].isin(overlap_dates)]
        incremental_overlap = incremental_all.loc[incremental_all["signal_date"].isin(overlap_dates)]
        joined = historical_overlap.merge(
            incremental_overlap,
            on=list(CORE_COLUMNS),
            suffixes=("_historical", "_incremental"),
            validate="one_to_one",
        )
        primary = columns[0]
        daily_spearman = []
        top5_overlap = []
        for _, day in joined.groupby("signal_date", sort=True):
            left = day[f"{primary}_historical"]
            right = day[f"{primary}_incremental"]
            correlation = left.corr(right, method="spearman")
            if correlation is not None and np.isfinite(correlation):
                daily_spearman.append(float(correlation))
            historical_top5 = set(day.nlargest(5, f"{primary}_historical")["instrument"])
            incremental_top5 = set(day.nlargest(5, f"{primary}_incremental")["instrument"])
            top5_overlap.append(len(historical_top5 & incremental_top5))
        overlap_validation = {
            "start": overlap_dates[0],
            "end": overlap_dates[-1],
            "dates": len(overlap_dates),
            "matched_rows": int(len(joined)),
            "mean_spearman": float(np.mean(daily_spearman)) if daily_spearman else None,
            "minimum_top5_overlap": min(top5_overlap) if top5_overlap else None,
        }
    historical = historical_all.loc[historical_all["signal_date"] <= cutoff]
    incremental = incremental_all.loc[incremental_all["signal_date"] > cutoff]
    if historical.empty:
        raise MarketScoreArtifactError(f"Historical score has no rows through cutoff {cutoff}")
    if incremental.empty:
        raise MarketScoreArtifactError(f"Incremental score has no rows after cutoff {cutoff}")
    merged = pd.concat([historical, incremental], ignore_index=True)
    merged = merged.sort_values(list(CORE_COLUMNS), kind="mergesort").reset_index(drop=True)
    return write_market_score_artifact(
        merged,
        output_path,
        score_columns=columns,
        overwrite=overwrite,
        metadata={
            "formal_status": "score_merged_ready_for_quantx",
            "merge_contract": (
                f"historical artifact through {cutoff}; "
                f"incremental label-free inference after {cutoff}"
            ),
            "overlap_validation": overlap_validation,
            "historical_artifact": historical_validation,
            "incremental_artifact": incremental_validation,
        },
    )
