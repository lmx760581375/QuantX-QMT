"""Load yearly-partitioned Kronos sample tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetPaths
from .storage import read_json


def years_between(start: str, end: str) -> list[int]:
    start_year = int(str(start)[:4])
    end_year = int(str(end)[:4])
    return list(range(start_year, end_year + 1))


def sample_years_for_ranges(ranges: list[tuple[str, str]]) -> list[int]:
    years: set[int] = set()
    for start, end in ranges:
        years.update(years_between(start, end))
    return sorted(years)


def load_sample_index(
    paths: DatasetPaths,
    *,
    years: list[int] | tuple[int, ...] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_years = [int(year) for year in years] if years is not None else None
    yearly_dir = paths.data / "sample_index_v1_by_year"
    if yearly_dir.exists():
        files = yearly_sample_files(paths, selected_years)
        if files:
            return pd.concat((pd.read_parquet(path, columns=columns) for path in files), ignore_index=True)
    return pd.read_parquet(paths.sample_index, columns=columns)


def load_label_diag(
    paths: DatasetPaths,
    *,
    years: list[int] | tuple[int, ...] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_years = [int(year) for year in years] if years is not None else None
    yearly_dir = paths.data / "label_diag_tb_h5_tp006_sl004_by_year"
    if yearly_dir.exists():
        files = yearly_label_diag_files(paths, selected_years)
        if files:
            return pd.concat((pd.read_parquet(path, columns=columns) for path in files), ignore_index=True)
    return pd.read_parquet(paths.label_diag, columns=columns)


def yearly_sample_files(paths: DatasetPaths, years: list[int] | None = None) -> list[Path]:
    manifest_path = paths.data / "sample_tables_by_year_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        files = manifest.get("sample_index_by_year", {}).get("files", {})
        selected = sorted(int(year) for year in files) if years is None else [int(year) for year in years]
        return [Path(files[str(year)]["path"]) for year in selected if str(year) in files]
    yearly_dir = paths.data / "sample_index_v1_by_year"
    if not yearly_dir.exists():
        return []
    if years is None:
        return sorted(yearly_dir.glob("sample_index_v1_*.parquet"))
    return [yearly_dir / f"sample_index_v1_{int(year)}.parquet" for year in years if (yearly_dir / f"sample_index_v1_{int(year)}.parquet").exists()]


def yearly_label_diag_files(paths: DatasetPaths, years: list[int] | None = None) -> list[Path]:
    manifest_path = paths.data / "sample_tables_by_year_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        files = manifest.get("label_diag_by_year", {}).get("files", {})
        selected = sorted(int(year) for year in files) if years is None else [int(year) for year in years]
        return [Path(files[str(year)]["path"]) for year in selected if str(year) in files]
    yearly_dir = paths.data / "label_diag_tb_h5_tp006_sl004_by_year"
    if not yearly_dir.exists():
        return []
    if years is None:
        return sorted(yearly_dir.glob("label_diag_tb_h5_tp006_sl004_*.parquet"))
    return [
        yearly_dir / f"label_diag_tb_h5_tp006_sl004_{int(year)}.parquet"
        for year in years
        if (yearly_dir / f"label_diag_tb_h5_tp006_sl004_{int(year)}.parquet").exists()
    ]


def years_from_frame_dates(frame: pd.DataFrame, column: str = "signal_date") -> list[int]:
    if frame.empty:
        return []
    return sorted({int(str(value)[:4]) for value in frame[column].dropna().unique()})
