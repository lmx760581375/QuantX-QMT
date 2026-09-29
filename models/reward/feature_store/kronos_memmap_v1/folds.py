"""Fold construction and purge logic for Kronos memmap experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import LABEL_HORIZON, DatasetPaths
from .storage import now_shanghai, read_json, write_json


FOLD_INDEX_COLUMNS = (
    "row_id",
    "instrument_idx",
    "shard_id",
    "local_instrument_idx",
    "date_idx",
    "instrument",
    "signal_date",
    "label_end_date",
    "year",
    "label",
    "board",
    "asset_bucket",
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    mode: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    prediction_start: str
    prediction_end: str
    validation_select_start: str
    validation_select_end: str
    validation_calibration_start: str
    validation_calibration_end: str
    horizon: int = LABEL_HORIZON

    def to_json(self, counts: dict[str, int] | None = None, purged: dict[str, int] | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = dict(counts or {})
        payload["purged"] = dict(purged or {})
        payload["created_at"] = now_shanghai()
        return payload


OFFICIAL_FOLDS = (
    ("fold_2021", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("fold_2022", "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("fold_2023", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("fold_2024", "2016-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("fold_2025", "2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("forward_2026", "2016-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01", "2026-07-15"),
)


def official_specs(sample: pd.DataFrame) -> list[FoldSpec]:
    specs = []
    dates = sorted(str(value) for value in sample["signal_date"].dropna().unique())
    for fold_id, train_start, train_end, val_start, val_end, pred_start, pred_end in OFFICIAL_FOLDS:
        val_dates = [date for date in dates if val_start <= date <= val_end]
        if len(val_dates) < 4:
            select_start, select_end, cal_start, cal_end = val_start, val_end, val_end, val_end
        else:
            cut = max(1, int(len(val_dates) * 0.70))
            select_start = val_dates[0]
            select_end = val_dates[cut - 1]
            cal_start = val_dates[min(cut + LABEL_HORIZON, len(val_dates) - 1)]
            cal_end = val_dates[-1]
        specs.append(
            FoldSpec(
                fold_id=fold_id,
                mode="official",
                train_start=train_start,
                train_end=train_end,
                validation_start=val_start,
                validation_end=val_end,
                prediction_start=pred_start,
                prediction_end=pred_end,
                validation_select_start=select_start,
                validation_select_end=select_end,
                validation_calibration_start=cal_start,
                validation_calibration_end=cal_end,
            )
        )
    return specs


def auto_smoke_spec(sample: pd.DataFrame, *, fold_id: str = "smoke_auto") -> FoldSpec:
    dates = sorted(str(value) for value in sample["signal_date"].dropna().unique())
    if len(dates) < 12:
        raise ValueError("auto smoke fold needs at least 12 signal dates")
    n = len(dates)
    train_end_idx = max(0, int(n * 0.60) - 1)
    val_select_start_idx = min(n - 1, train_end_idx + 1)
    val_select_end_idx = max(val_select_start_idx, int(n * 0.78) - 1)
    cal_start_idx = min(n - 1, val_select_end_idx + 1 + LABEL_HORIZON)
    cal_end_idx = max(cal_start_idx, int(n * 0.90) - 1)
    pred_start_idx = min(n - 1, cal_end_idx + 1)
    return FoldSpec(
        fold_id=fold_id,
        mode="auto_smoke",
        train_start=dates[0],
        train_end=dates[train_end_idx],
        validation_start=dates[val_select_start_idx],
        validation_end=dates[cal_end_idx],
        prediction_start=dates[pred_start_idx],
        prediction_end=dates[-1],
        validation_select_start=dates[val_select_start_idx],
        validation_select_end=dates[val_select_end_idx],
        validation_calibration_start=dates[cal_start_idx],
        validation_calibration_end=dates[cal_end_idx],
    )


def split_rows(sample: pd.DataFrame, spec: FoldSpec) -> dict[str, pd.DataFrame]:
    frame = sample.copy()
    signal = frame["signal_date"].astype(str)
    label_end = frame["label_end_date"].astype(str)
    train_base = signal.between(spec.train_start, spec.train_end)
    validation_base = signal.between(spec.validation_start, spec.validation_end)
    validation_select_base = signal.between(spec.validation_select_start, spec.validation_select_end)
    validation_calibration_base = signal.between(spec.validation_calibration_start, spec.validation_calibration_end)
    prediction_base = signal.between(spec.prediction_start, spec.prediction_end)
    train = frame.loc[train_base & (label_end < spec.validation_start)].copy()
    validation = frame.loc[validation_base & (label_end < spec.prediction_start)].copy()
    validation_select = frame.loc[validation_select_base & (label_end < spec.validation_calibration_start)].copy()
    validation_calibration = frame.loc[validation_calibration_base & (label_end < spec.prediction_start)].copy()
    prediction = frame.loc[prediction_base & (label_end <= spec.prediction_end)].copy()
    return {
        "train": train,
        "validation": validation,
        "validation_select": validation_select,
        "validation_calibration": validation_calibration,
        "prediction": prediction,
    }


def purge_counts(sample: pd.DataFrame, spec: FoldSpec, splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    signal = sample["signal_date"].astype(str)
    label_end = sample["label_end_date"].astype(str)
    bases = {
        "train": signal.between(spec.train_start, spec.train_end),
        "validation": signal.between(spec.validation_start, spec.validation_end),
        "validation_select": signal.between(spec.validation_select_start, spec.validation_select_end),
        "validation_calibration": signal.between(spec.validation_calibration_start, spec.validation_calibration_end),
        "prediction": signal.between(spec.prediction_start, spec.prediction_end),
    }
    kept = {name: set(rows["row_id"].astype(int).tolist()) for name, rows in splits.items()}
    counts: dict[str, int] = {}
    for name, mask in bases.items():
        base_ids = set(sample.loc[mask, "row_id"].astype(int).tolist())
        counts[name] = int(len(base_ids - kept[name]))
    return counts


def write_fold(paths: DatasetPaths, sample: pd.DataFrame, spec: FoldSpec) -> Path:
    splits = split_rows(sample, spec)
    counts = {name: int(len(rows)) for name, rows in splits.items()}
    purged = purge_counts(sample, spec, splits)
    payload = spec.to_json(counts=counts, purged=purged)
    index_paths = write_fold_indices(paths, spec, splits)
    payload["index_paths"] = index_paths
    payload["label_end_max"] = {
        name: str(rows["label_end_date"].max()) if not rows.empty else None for name, rows in splits.items()
    }
    paths.folds.mkdir(parents=True, exist_ok=True)
    output = paths.folds / f"{spec.fold_id}.json"
    write_json(output, payload)
    return output


def write_fold_indices(paths: DatasetPaths, spec: FoldSpec, splits: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Persist small, purged row-reference tables for direct training reads.

    These files intentionally contain metadata only. The 60-session input is
    still sliced dynamically from the immutable memmap feature stores.
    """
    fold_dir = paths.folds / spec.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, rows in splits.items():
        output = fold_dir / f"{name}.parquet"
        rows.to_parquet(output, index=False, compression="zstd")
        outputs[name] = str(output)
    return outputs


def stream_write_fold_indices(
    paths: DatasetPaths,
    spec: FoldSpec,
    source_files: list[Path],
    *,
    batch_size: int = 500_000,
) -> tuple[dict[str, str], dict[str, int], dict[str, int], dict[str, str | None]]:
    """Create purged split indices without materializing the source panel.

    A source row may belong to several validation views, so each batch is
    filtered independently and appended to each relevant narrow index.  The
    sequential memmaps remain the sole owner of feature-window data.
    """
    if not source_files:
        raise FileNotFoundError("No sample-index parquet files available for fold construction")
    fold_dir = paths.folds / spec.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: fold_dir / f"{name}.parquet" for name in _split_names()}
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {name: 0 for name in _split_names()}
    purged = {name: 0 for name in _split_names()}
    label_end_max: dict[str, str | None] = {name: None for name in _split_names()}
    parquet_files = [pq.ParquetFile(path) for path in source_files]
    try:
        for parquet_file in parquet_files:
            for batch in parquet_file.iter_batches(columns=list(FOLD_INDEX_COLUMNS), batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                for name, base, kept in _split_filters(table, spec).values():
                    base_rows = table.filter(base)
                    selected = table.filter(kept)
                    purged[name] += int(base_rows.num_rows - selected.num_rows)
                    if selected.num_rows == 0:
                        continue
                    writer = writers.get(name)
                    if writer is None:
                        writer = pq.ParquetWriter(outputs[name], selected.schema, compression="zstd")
                        writers[name] = writer
                    writer.write_table(selected)
                    counts[name] += int(selected.num_rows)
                    maximum = pc.max(selected["label_end_date"]).as_py()
                    if maximum is not None and (label_end_max[name] is None or str(maximum) > label_end_max[name]):
                        label_end_max[name] = str(maximum)
    finally:
        for writer in writers.values():
            writer.close()
    for name, output in outputs.items():
        if name not in writers:
            empty = pa.table({column: pa.array([], type=_empty_arrow_type(column)) for column in FOLD_INDEX_COLUMNS})
            pq.write_table(empty, output, compression="zstd")
    return ({name: str(path) for name, path in outputs.items()}, counts, purged, label_end_max)


def _split_names() -> tuple[str, ...]:
    return ("train", "validation", "validation_select", "validation_calibration", "prediction")


def _split_filters(table: pa.Table, spec: FoldSpec) -> dict[str, tuple[str, pa.Array, pa.Array]]:
    signal = table["signal_date"]
    label_end = table["label_end_date"]

    def between(values: pa.ChunkedArray, start: str, end: str) -> pa.Array:
        return pc.and_(pc.greater_equal(values, start), pc.less_equal(values, end))

    def selected(start: str, end: str, cutoff: str, inclusive: bool = False) -> tuple[pa.Array, pa.Array]:
        base = between(signal, start, end)
        end_ok = pc.less_equal(label_end, cutoff) if inclusive else pc.less(label_end, cutoff)
        return base, pc.and_(base, end_ok)

    train_base, train = selected(spec.train_start, spec.train_end, spec.validation_start)
    validation_base, validation = selected(spec.validation_start, spec.validation_end, spec.prediction_start)
    select_base, select = selected(
        spec.validation_select_start,
        spec.validation_select_end,
        spec.validation_calibration_start,
    )
    calibration_base, calibration = selected(
        spec.validation_calibration_start,
        spec.validation_calibration_end,
        spec.prediction_start,
    )
    prediction_base, prediction = selected(spec.prediction_start, spec.prediction_end, spec.prediction_end, inclusive=True)
    return {
        "train": ("train", train_base, train),
        "validation": ("validation", validation_base, validation),
        "validation_select": ("validation_select", select_base, select),
        "validation_calibration": ("validation_calibration", calibration_base, calibration),
        "prediction": ("prediction", prediction_base, prediction),
    }


def _empty_arrow_type(column: str) -> pa.DataType:
    types = {
        "row_id": pa.int64(),
        "instrument_idx": pa.int32(),
        "shard_id": pa.int16(),
        "local_instrument_idx": pa.int32(),
        "date_idx": pa.int32(),
        "instrument": pa.string(),
        "signal_date": pa.string(),
        "label_end_date": pa.string(),
        "year": pa.int16(),
        "label": pa.int8(),
        "board": pa.string(),
        "asset_bucket": pa.int8(),
    }
    return types[column]


def load_fold(path: str | Path) -> FoldSpec:
    payload = read_json(Path(path))
    return FoldSpec(
        fold_id=str(payload["fold_id"]),
        mode=str(payload["mode"]),
        train_start=str(payload["train_start"]),
        train_end=str(payload["train_end"]),
        validation_start=str(payload["validation_start"]),
        validation_end=str(payload["validation_end"]),
        prediction_start=str(payload["prediction_start"]),
        prediction_end=str(payload["prediction_end"]),
        validation_select_start=str(payload["validation_select_start"]),
        validation_select_end=str(payload["validation_select_end"]),
        validation_calibration_start=str(payload["validation_calibration_start"]),
        validation_calibration_end=str(payload["validation_calibration_end"]),
        horizon=int(payload.get("horizon", LABEL_HORIZON)),
    )


def validate_split_rows(rows: pd.DataFrame, split: dict[str, Any], name: str) -> None:
    """Reject a corrupt fold index before it is used for training or scoring."""
    if rows.empty:
        raise ValueError(f"Fold split {name} is empty")
    required = {"signal_date", "label_end_date", "instrument_idx", "shard_id", "local_instrument_idx", "date_idx", "label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Fold split {name} lacks required columns: {sorted(missing)}")
    signal = rows["signal_date"].astype(str)
    label_end = rows["label_end_date"].astype(str)
    bounds = {
        "train": ("train_start", "train_end", "validation_start", False),
        "validation": ("validation_start", "validation_end", "prediction_start", False),
        "validation_select": ("validation_select_start", "validation_select_end", "validation_calibration_start", False),
        "validation_calibration": ("validation_calibration_start", "validation_calibration_end", "prediction_start", False),
        "prediction": ("prediction_start", "prediction_end", "prediction_end", True),
    }
    if name not in bounds:
        raise ValueError(f"Unknown fold split: {name}")
    start_key, end_key, cutoff_key, inclusive = bounds[name]
    in_range = signal.between(str(split[start_key]), str(split[end_key]))
    end_ok = label_end <= str(split[cutoff_key]) if inclusive else label_end < str(split[cutoff_key])
    if not bool((in_range & end_ok).all()):
        bad = rows.loc[~(in_range & end_ok), ["signal_date", "label_end_date"]].head(3).to_dict("records")
        raise ValueError(f"Fold split {name} violates signal/label_end purge bounds: {bad}")
