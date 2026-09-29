"""Build a label-free all-market inference index from Kronos feature memmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.reward.feature_store.prepare_features_impl import (
    load_universe_mask,
    price_state_valid,
    rolling_count_2d,
)
from models.reward.feature_store.kronos_memmap_v1.config import (
    HISTORY_VALID_MIN_ROWS,
    RAW_OHLCV_FIELDS,
    RAW_STATE_FIELDS,
    SEQ_LEN,
)
from models.reward.feature_store.kronos_memmap_v1.execution import ExecutionRulesV1

from .common import WTSPaths, source_memmaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kronos-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = source_memmaps(Path(args.kronos_root).resolve())
    paths = WTSPaths(Path(args.output_root).resolve())
    paths.data.mkdir(parents=True, exist_ok=True)
    fold_dir = paths.data / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    output = fold_dir / "latest_inference.parquet"
    fold_json = fold_dir / "latest_inference.json"
    generated_paths = [
        output,
        fold_json,
        paths.target_meta,
    ]
    existing = [path for path in generated_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite inference artifacts without --force: "
            + ", ".join(str(path) for path in existing)
        )

    instruments = source.instruments.sort_values("instrument_idx").reset_index(drop=True)
    calendar = source.calendar.reset_index(drop=True)
    dates = pd.to_datetime(calendar["date"])
    n_inst = len(instruments)
    n_dates = len(calendar)
    universe = load_universe_mask(source.paths, (n_inst, n_dates))
    base_valid = price_state_valid(source.raw_ohlcv, source.raw_state)
    history_valid = rolling_count_2d(base_valid, SEQ_LEN) >= HISTORY_VALID_MIN_ROWS
    rules = ExecutionRulesV1()
    open_idx = RAW_OHLCV_FIELDS.index("open")
    high_idx = RAW_OHLCV_FIELDS.index("high")
    low_idx = RAW_OHLCV_FIELDS.index("low")
    close_idx = RAW_OHLCV_FIELDS.index("close")
    volume_idx = RAW_OHLCV_FIELDS.index("volume")
    state_tradable = RAW_STATE_FIELDS.index("tradestatus")
    state_st = RAW_STATE_FIELDS.index("is_st")
    state_listed = RAW_STATE_FIELDS.index("has_raw_row")
    shard_lookup = {}
    for item in source.feature_manifest["shards"]:
        start_idx = int(item["instrument_start"])
        end_idx = int(item["instrument_end"])
        for local_idx, inst_idx in enumerate(range(start_idx, end_idx)):
            shard_lookup[int(inst_idx)] = (int(item["shard_id"]), int(local_idx))

    selected_dates = [
        idx for idx, date in enumerate(dates)
        if str(args.start) <= date.strftime("%Y-%m-%d") <= str(args.end)
        and idx >= SEQ_LEN - 1
        and idx + 1 < n_dates
    ]
    writer: pq.ParquetWriter | None = None
    row_count = 0
    used_dates = 0
    first_date = None
    last_date = None
    try:
        for sequence, date_idx in enumerate(selected_dates, start=1):
            entry_idx = date_idx + 1
            preclose = np.asarray(source.raw_ohlcv[:, date_idx, close_idx], dtype=float)
            entry_open = np.asarray(source.raw_ohlcv[:, entry_idx, open_idx], dtype=float)
            entry_high = np.asarray(source.raw_ohlcv[:, entry_idx, high_idx], dtype=float)
            entry_low = np.asarray(source.raw_ohlcv[:, entry_idx, low_idx], dtype=float)
            entry_close = np.asarray(source.raw_ohlcv[:, entry_idx, close_idx], dtype=float)
            entry_volume = np.asarray(source.raw_ohlcv[:, entry_idx, volume_idx], dtype=float)
            entry_state = np.asarray(source.raw_state[:, entry_idx, :])
            limit_rates = np.asarray([
                0.05
                if entry_state[idx, state_st] == 1
                else rules.limit_rate(str(instruments.iloc[idx]["instrument"]))
                for idx in range(n_inst)
            ])
            one_price = (
                np.isfinite(entry_open)
                & np.isfinite(entry_high)
                & np.isfinite(entry_low)
                & np.isfinite(entry_close)
                & (np.abs(entry_high - entry_low) <= rules.one_price_eps)
                & (np.abs(entry_open - entry_close) <= rules.one_price_eps)
                & ((entry_close / preclose - 1.0) >= 0.045 - rules.limit_eps)
            )
            open_limit_up = (
                np.isfinite(preclose)
                & (preclose > 0)
                & (entry_open >= preclose * (1.0 + limit_rates) - rules.limit_eps)
            )
            valid = (
                universe[:, date_idx]
                & history_valid[:, date_idx]
                & (entry_state[:, state_tradable] == 1)
                & (entry_state[:, state_listed] == 1)
                & (entry_state[:, state_st] == 0)
                & np.isfinite(entry_open)
                & (entry_open > 0)
                & np.isfinite(entry_close)
                & (entry_close > 0)
                & np.isfinite(entry_volume)
                & (entry_volume > 0)
                & ~one_price
                & ~open_limit_up
            )
            selected = np.flatnonzero(valid)
            if not len(selected):
                continue
            signal_date = dates[date_idx].strftime("%Y-%m-%d")
            instrument_rows = instruments.iloc[selected]
            count = len(selected)
            frame = pd.DataFrame({
                "row_id": np.arange(row_count, row_count + count, dtype=np.int64),
                "target_row": np.arange(row_count, row_count + count, dtype=np.int64),
                "instrument_idx": selected.astype(np.int64),
                "shard_id": [shard_lookup[int(index)][0] for index in selected],
                "local_instrument_idx": [shard_lookup[int(index)][1] for index in selected],
                "date_idx": np.full(count, int(date_idx), dtype=np.int64),
                "instrument": instrument_rows["instrument"].astype(str).to_numpy(),
                "signal_date": np.full(count, signal_date, dtype=object),
                "year": np.full(count, int(dates[date_idx].year), dtype=np.int16),
                "entry_feasible": np.ones(count, dtype=bool),
                "label_execution_invalid": np.zeros(count, dtype=bool),
                "board": instrument_rows["board"].astype(str).to_numpy(),
                "asset_bucket": instrument_rows["asset_bucket"].astype(np.int8).to_numpy(),
                "wts_rank": np.full(count, 2, dtype=np.int32),
                "wts_rank_pct": np.full(count, 0.5, dtype=np.float32),
                "wts_buyable_rank": np.full(count, 2, dtype=np.int32),
                "wts_score": np.zeros(count, dtype=np.float32),
                "wts_score_z": np.zeros(count, dtype=np.float32),
                "wts_score_pct": np.full(count, 0.5, dtype=np.float32),
                "wts_selector_where": np.ones(count, dtype=bool),
                "wts_pool_size": np.full(count, 4, dtype=np.int32),
                "topn": np.full(count, 4, dtype=np.int32),
                "buyable": np.ones(count, dtype=bool),
                "buy_block_reason": np.full(count, "", dtype=object),
            })
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += count
            used_dates += 1
            first_date = first_date or signal_date
            last_date = signal_date
            if sequence == 1 or sequence % max(1, int(args.progress_every)) == 0 or sequence == len(selected_dates):
                print(
                    f"[inference-index] dates={sequence}/{len(selected_dates)} "
                    f"date={signal_date} rows={row_count}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        raise RuntimeError(f"No valid inference rows in {args.start} through {args.end}")

    paths.target_meta.write_text(json.dumps({
        "shape": [row_count, 0],
        "dtype": "float32",
        "inference_only": True,
        "future_targets_available": False,
    }, indent=2), encoding="utf-8")
    fold_json.write_text(json.dumps({
        "fold_id": "latest_inference",
        "index_paths": {"prediction": str(output.resolve())},
        "inference_only": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "rows": row_count,
        "dates": used_dates,
        "start": first_date,
        "end": last_date,
        "fold": str(fold_json),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
