"""从同源 Qlib 与 BaoStock 数据构建 Reward 模型的统一特征存储。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .kronos_memmap_v1.config import (
    BOARD_TO_CODE,
    DATA_DIR,
    DEFAULT_PROVIDER_URI,
    DEFAULT_RAW_STOCK_DIR,
    DEFAULT_SHARD_SIZE,
    DERIVED_FEATURE_FIELDS,
    HISTORY_VALID_MIN_ROWS,
    LABEL_HORIZON,
    LABEL_INVALID_BITS,
    MARKET_FEATURE_FIELDS,
    MIN_HISTORY_DAYS,
    RAW_OHLCV_FIELDS,
    RAW_STATE_FIELDS,
    RAW_TRADE_FIELDS,
    SEQ_LEN,
    STOP_LOSS,
    TAKE_PROFIT,
    UNIVERSE_REASON_BITS,
    DatasetPaths,
    board_of,
    schema_hash,
    stable_asset_bucket,
)
from .kronos_memmap_v1.execution import ExecutionRulesV1
from .kronos_memmap_v1.qlib_io import read_calendar, read_day_bin_field, read_instruments
from .kronos_memmap_v1.storage import (
    cheap_tree_fingerprint,
    ensure_dirs,
    file_fingerprint,
    file_sha256,
    memmap_meta,
    now_shanghai,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["raw", "universe", "features", "labels", "inference", "all"],
        help="inference builds raw, universe, and feature tensors without future labels.",
    )
    parser.add_argument("--output-root", default=str(DATA_DIR.parent))
    parser.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
    parser.add_argument("--raw-stock-dir", default=str(DEFAULT_RAW_STOCK_DIR))
    parser.add_argument("--start", help="Optional YYYY-MM-DD calendar start for smoke runs.")
    parser.add_argument("--end", help="Optional YYYY-MM-DD calendar end for smoke runs.")
    parser.add_argument("--limit-instruments", type=int, help="Optional instrument limit for smoke runs.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--force", action="store_true", help="Overwrite artifacts for the selected stage.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = DatasetPaths(Path(args.output_root).resolve())
    ensure_dirs(paths.data, paths.reports, paths.runs)
    if args.stage in {"raw", "inference", "all"}:
        build_raw(args, paths)
    if args.stage in {"universe", "inference", "all"}:
        build_universe(args, paths)
    if args.stage in {"features", "inference", "all"}:
        build_features(args, paths)
    if args.stage in {"labels", "all"}:
        build_labels(args, paths)
    print(json.dumps({"ok": True, "stage": args.stage, "root": str(paths.root)}, ensure_ascii=False, sort_keys=True))
    return 0


def build_raw(args: argparse.Namespace, paths: DatasetPaths) -> None:
    provider = Path(args.provider_uri).resolve()
    raw_stock_dir = Path(args.raw_stock_dir).resolve()
    if not provider.exists():
        raise FileNotFoundError(f"Missing provider: {provider}")
    if not raw_stock_dir.exists():
        raise FileNotFoundError(f"Missing raw stock dir: {raw_stock_dir}")
    required = [
        paths.calendar,
        paths.instruments,
        paths.raw_ohlcv,
        paths.raw_state,
        paths.raw_trade,
        paths.raw_ohlcv_meta,
        paths.raw_state_meta,
        paths.raw_trade_meta,
        paths.manifest,
    ]
    _guard_outputs(required, args.force)

    calendar_full = read_calendar(provider)
    calendar = read_calendar(provider, start=args.start, end=args.end)
    if calendar.empty:
        raise ValueError("Selected calendar is empty")
    instruments = prepare_instruments(provider, limit=args.limit_instruments)
    if instruments.empty:
        raise ValueError("No supported stock instruments selected")

    calendar_frame = pd.DataFrame(
        {
            "date_idx": np.arange(len(calendar), dtype=np.int32),
            "date": [day.strftime("%Y-%m-%d") for day in calendar],
            "year": calendar.year.astype(np.int16),
        }
    )
    calendar_frame.to_parquet(paths.calendar, index=False)
    instruments.to_parquet(paths.instruments, index=False)

    n_inst = int(len(instruments))
    n_dates = int(len(calendar))
    raw_ohlcv = np.memmap(paths.raw_ohlcv, dtype=np.float32, mode="w+", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
    raw_ohlcv[:] = np.nan
    raw_state = np.memmap(paths.raw_state, dtype=np.uint8, mode="w+", shape=(n_inst, n_dates, len(RAW_STATE_FIELDS)))
    raw_state[:] = 0
    raw_trade = np.memmap(paths.raw_trade, dtype=np.float32, mode="w+", shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)))
    raw_trade[:] = np.nan

    date_to_idx = {day.strftime("%Y-%m-%d"): idx for idx, day in enumerate(calendar)}
    for row in instruments.itertuples(index=False):
        inst_idx = int(row.instrument_idx)
        instrument = str(row.instrument)
        for field_idx, field in enumerate(RAW_OHLCV_FIELDS):
            raw_ohlcv[inst_idx, :, field_idx] = read_day_bin_field(
                provider, instrument, field, calendar_full, calendar
            )
        load_raw_state_trade(
            raw_stock_dir / f"{instrument}.csv",
            instrument,
            date_to_idx,
            raw_state[inst_idx],
            raw_trade[inst_idx],
        )
        if inst_idx == 0 or (inst_idx + 1) % 250 == 0 or inst_idx + 1 == n_inst:
            print(f"raw progress {inst_idx + 1}/{n_inst} {instrument}", flush=True)
    raw_ohlcv.flush()
    raw_state.flush()
    raw_trade.flush()

    common = {
        "provider_uri": str(provider),
        "raw_stock_dir": str(raw_stock_dir),
        "provider_calendar": file_fingerprint(provider / "calendars" / "day.txt"),
        "provider_instruments": file_fingerprint(provider / "instruments" / "all.txt"),
        "raw_stock_tree": cheap_tree_fingerprint(raw_stock_dir, ("*.csv",)),
        "instrument_count": n_inst,
        "date_count": n_dates,
        "start": calendar[0].strftime("%Y-%m-%d"),
        "end": calendar[-1].strftime("%Y-%m-%d"),
        "limit_instruments": args.limit_instruments,
    }
    write_json(
        paths.raw_ohlcv_meta,
        memmap_meta(
            path=paths.raw_ohlcv,
            shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)),
            dtype="float32",
            axes=("instrument_idx", "date_idx", "raw_ohlcv_field"),
            fields=RAW_OHLCV_FIELDS,
            extra=common | {"feature_schema_hash": schema_hash(RAW_OHLCV_FIELDS)},
        ),
    )
    write_json(
        paths.raw_state_meta,
        memmap_meta(
            path=paths.raw_state,
            shape=(n_inst, n_dates, len(RAW_STATE_FIELDS)),
            dtype="uint8",
            axes=("instrument_idx", "date_idx", "raw_state_field"),
            fields=RAW_STATE_FIELDS,
            extra=common | {"state_schema_hash": schema_hash(RAW_STATE_FIELDS)},
        ),
    )
    write_json(
        paths.raw_trade_meta,
        memmap_meta(
            path=paths.raw_trade,
            shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)),
            dtype="float32",
            axes=("instrument_idx", "date_idx", "raw_trade_field"),
            fields=RAW_TRADE_FIELDS,
            extra=common | {"trade_schema_hash": schema_hash(RAW_TRADE_FIELDS)},
        ),
    )
    write_json(
        paths.manifest,
        {
            "format": "kronos_memmap_classification_v1",
            "format_version": 1,
            "created_at": now_shanghai(),
            "paths": {
                "calendar": str(paths.calendar),
                "instruments": str(paths.instruments),
                "raw_ohlcv": str(paths.raw_ohlcv),
                "raw_state": str(paths.raw_state),
                "raw_trade": str(paths.raw_trade),
            },
            "config": common,
        },
    )


def prepare_instruments(provider: Path, *, limit: int | None) -> pd.DataFrame:
    frame = read_instruments(provider)
    frame["instrument"] = frame["instrument"].astype(str)
    frame["board"] = frame["instrument"].map(board_of)
    frame["board_code"] = frame["board"].map(BOARD_TO_CODE).astype(np.int8)
    frame["is_index"] = frame["board"].eq("index")
    frame["asset_bucket"] = frame["instrument"].map(stable_asset_bucket).astype(np.int8)
    supported = frame["board"].isin(["mainboard_sh", "mainboard_sz", "chinext", "star"])
    frame = frame.loc[supported & ~frame["is_index"]].copy()
    frame = frame.sort_values("instrument").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    frame.insert(0, "instrument_idx", np.arange(len(frame), dtype=np.int32))
    frame["asset_bucket"] = frame["asset_bucket"].astype(np.int8)
    return frame[
        [
            "instrument_idx",
            "instrument",
            "start_date",
            "end_date",
            "board",
            "board_code",
            "is_index",
            "asset_bucket",
        ]
    ]


def load_raw_state_trade(
    csv_path: Path,
    instrument: str,
    date_to_idx: dict[str, int],
    state_out: np.ndarray,
    trade_out: np.ndarray,
) -> None:
    del instrument
    if not csv_path.exists():
        return
    def usecols(column: str) -> bool:
        return column in {"date", "code", "amount", "turnover", "turn", "tradestatus", "is_st", "isST"}

    frame = pd.read_csv(csv_path, usecols=usecols)
    if frame.empty or "date" not in frame:
        return
    frame["date"] = frame["date"].astype(str)
    mapped = frame["date"].map(date_to_idx)
    frame = frame.loc[mapped.notna()].copy()
    if frame.empty:
        return
    idx = mapped.loc[frame.index].astype(int).to_numpy()
    state_out[idx, 2] = 1
    if "tradestatus" in frame:
        state_out[idx, 0] = pd.to_numeric(frame["tradestatus"], errors="coerce").fillna(0).astype(np.uint8).to_numpy()
    if "is_st" in frame:
        state_out[idx, 1] = pd.to_numeric(frame["is_st"], errors="coerce").fillna(0).astype(np.uint8).to_numpy()
    elif "isST" in frame:
        state_out[idx, 1] = pd.to_numeric(frame["isST"], errors="coerce").fillna(0).astype(np.uint8).to_numpy()
    if "amount" in frame:
        trade_out[idx, 0] = pd.to_numeric(frame["amount"], errors="coerce").astype(np.float32).to_numpy()
    turnover_column = "turnover" if "turnover" in frame else "turn"
    if turnover_column in frame:
        trade_out[idx, 1] = pd.to_numeric(frame[turnover_column], errors="coerce").astype(np.float32).to_numpy()


def build_universe(args: argparse.Namespace, paths: DatasetPaths) -> None:
    _guard_outputs([paths.universe_daily, paths.universe_meta], args.force)
    instruments = pd.read_parquet(paths.instruments)
    calendar = pd.read_parquet(paths.calendar)
    n_inst = len(instruments)
    n_dates = len(calendar)
    raw_ohlcv = np.memmap(paths.raw_ohlcv, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
    raw_state = np.memmap(paths.raw_state, dtype=np.uint8, mode="r", shape=(n_inst, n_dates, len(RAW_STATE_FIELDS)))
    open_ = raw_ohlcv[:, :, 0]
    high = raw_ohlcv[:, :, 1]
    low = raw_ohlcv[:, :, 2]
    close = raw_ohlcv[:, :, 3]
    volume = raw_ohlcv[:, :, 4]
    vwap = raw_ohlcv[:, :, 5]
    price_valid = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(volume)
        & np.isfinite(vwap)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
        & (vwap > 0)
        & (high >= low)
    )
    volume_valid = np.isfinite(volume) & (volume > 0)
    state_valid = (raw_state[:, :, 0] == 1) & (raw_state[:, :, 1] == 0) & (raw_state[:, :, 2] == 1)
    trade_count = np.cumsum(price_valid & volume_valid & state_valid, axis=1)

    reason = np.zeros((n_inst, n_dates), dtype=np.uint16)
    board = instruments["board"].astype(str).to_numpy()
    is_index = instruments["is_index"].astype(bool).to_numpy()
    unsupported = ~np.isin(board, ["mainboard_sh", "mainboard_sz", "chinext", "star"])
    reason[unsupported, :] |= UNIVERSE_REASON_BITS["unsupported_board"]
    reason[is_index, :] |= UNIVERSE_REASON_BITS["is_index"]
    reason[trade_count < MIN_HISTORY_DAYS] |= UNIVERSE_REASON_BITS["history_lt_80"]
    reason[raw_state[:, :, 2] != 1] |= UNIVERSE_REASON_BITS["missing_raw_row"]
    reason[raw_state[:, :, 0] != 1] |= UNIVERSE_REASON_BITS["tradestatus_not_1"]
    reason[raw_state[:, :, 1] != 0] |= UNIVERSE_REASON_BITS["is_st"]
    reason[~price_valid] |= UNIVERSE_REASON_BITS["invalid_ohlcv"]
    reason[~volume_valid] |= UNIVERSE_REASON_BITS["zero_volume"]
    in_universe = reason == 0

    writer: pq.ParquetWriter | None = None
    try:
        block = 128
        for start in range(0, n_dates, block):
            end = min(n_dates, start + block)
            width = end - start
            table = pa.table(
                {
                    "date_idx": np.repeat(np.arange(start, end, dtype=np.int32), n_inst),
                    "instrument_idx": np.tile(np.arange(n_inst, dtype=np.int32), width),
                    "in_universe": in_universe[:, start:end].T.reshape(-1),
                    "reason_code": reason[:, start:end].T.reshape(-1).astype(np.uint16),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(paths.universe_daily, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    counts = {
        name: int(((reason & bit) != 0).sum())
        for name, bit in UNIVERSE_REASON_BITS.items()
    }
    write_json(
        paths.universe_meta,
        {
            "path": str(paths.universe_daily),
            "created_at": now_shanghai(),
            "shape": [int(n_inst), int(n_dates)],
            "reason_bits": UNIVERSE_REASON_BITS,
            "counts": counts,
            "in_universe_count": int(in_universe.sum()),
            "calendar_sha256": file_sha256(paths.calendar),
            "instruments_sha256": file_sha256(paths.instruments),
        },
    )


def build_features(args: argparse.Namespace, paths: DatasetPaths) -> None:
    _guard_outputs([paths.features_manifest, paths.market, paths.market_meta], args.force)
    instruments = pd.read_parquet(paths.instruments)
    calendar = pd.read_parquet(paths.calendar)
    n_inst = len(instruments)
    n_dates = len(calendar)
    raw_ohlcv = np.memmap(paths.raw_ohlcv, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
    raw_trade = np.memmap(paths.raw_trade, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)))
    universe = load_universe_mask(paths, (n_inst, n_dates))
    shard_size = int(args.shard_size)
    shards = create_feature_shards(paths, n_inst, n_dates, shard_size, force=args.force)
    shard_maps = open_feature_shards(shards, mode="r+")
    try:
        for row in instruments.itertuples(index=False):
            inst_idx = int(row.instrument_idx)
            shard_id, local_idx = shard_for_inst(shards, inst_idx)
            derived = compute_derived_features(raw_ohlcv[inst_idx], raw_trade[inst_idx])
            finite = np.isfinite(derived)
            if finite.any() and float(np.nanmax(np.abs(derived[finite]))) > np.finfo(np.float16).max:
                raise OverflowError(f"float16 overflow in derived features for {row.instrument}")
            shard_maps[shard_id][local_idx, :, :32] = derived.astype(np.float16)
            if inst_idx == 0 or (inst_idx + 1) % 250 == 0 or inst_idx + 1 == n_inst:
                print(f"feature ts progress {inst_idx + 1}/{n_inst} {row.instrument}", flush=True)
        for mmap in shard_maps.values():
            mmap.flush()
        market = np.memmap(paths.market, dtype=np.float32, mode="w+", shape=(n_dates, len(MARKET_FEATURE_FIELDS)))
        market[:] = np.nan
        fill_rank_and_market(raw_ohlcv, raw_trade, universe, shards, shard_maps, market)
        market.flush()
    finally:
        for mmap in shard_maps.values():
            mmap.flush()
    write_feature_manifest(paths, shards, n_inst, n_dates)
    write_json(
        paths.market_meta,
        memmap_meta(
            path=paths.market,
            shape=(n_dates, len(MARKET_FEATURE_FIELDS)),
            dtype="float32",
            axes=("date_idx", "market_feature"),
            fields=MARKET_FEATURE_FIELDS,
            extra={
                "feature_schema_hash": schema_hash(MARKET_FEATURE_FIELDS),
                "universe_sha256": file_sha256(paths.universe_daily),
                "calendar_sha256": file_sha256(paths.calendar),
            },
        ),
    )


def create_feature_shards(paths: DatasetPaths, n_inst: int, n_dates: int, shard_size: int, *, force: bool) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for shard_id, start in enumerate(range(0, n_inst, shard_size)):
        end = min(n_inst, start + shard_size)
        path = paths.data / f"features_derived_v1_float16.part-{shard_id:03d}.mmap"
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
        mmap = np.memmap(path, dtype=np.float16, mode="w+", shape=(end - start, n_dates, len(DERIVED_FEATURE_FIELDS)))
        mmap[:] = np.nan
        mmap.flush()
        planned.append(
            {
                "shard_id": int(shard_id),
                "path": str(path),
                "instrument_start": int(start),
                "instrument_end": int(end),
                "shape": [int(end - start), int(n_dates), int(len(DERIVED_FEATURE_FIELDS))],
                "dtype": "float16",
            }
        )
    return planned


def open_feature_shards(shards: list[dict[str, Any]], *, mode: str) -> dict[int, np.memmap]:
    out: dict[int, np.memmap] = {}
    for shard in shards:
        out[int(shard["shard_id"])] = np.memmap(
            shard["path"],
            dtype=np.float16,
            mode=mode,
            shape=tuple(int(v) for v in shard["shape"]),
        )
    return out


def write_feature_manifest(paths: DatasetPaths, shards: list[dict[str, Any]], n_inst: int, n_dates: int) -> None:
    enriched = []
    for shard in shards:
        item = dict(shard)
        item["sha256"] = file_sha256(Path(item["path"]))
        enriched.append(item)
    write_json(
        paths.features_manifest,
        {
            "format": "kronos_features_derived_v1",
            "created_at": now_shanghai(),
            "logical_shape": [int(n_inst), int(n_dates), int(len(DERIVED_FEATURE_FIELDS))],
            "dtype": "float16",
            "axes": ["instrument_idx", "date_idx", "derived_feature"],
            "fields": list(DERIVED_FEATURE_FIELDS),
            "feature_schema_hash": schema_hash(DERIVED_FEATURE_FIELDS),
            "shards": enriched,
            "raw_ohlcv_sha256": file_sha256(paths.raw_ohlcv_meta),
            "raw_trade_sha256": file_sha256(paths.raw_trade_meta),
            "universe_sha256": file_sha256(paths.universe_daily),
        },
    )


def compute_derived_features(raw: np.ndarray, trade: np.ndarray) -> np.ndarray:
    open_ = raw[:, 0].astype(float)
    high = raw[:, 1].astype(float)
    low = raw[:, 2].astype(float)
    close = raw[:, 3].astype(float)
    volume = raw[:, 4].astype(float)
    amount = trade[:, 0].astype(float)
    out = np.full((len(raw), 32), np.nan, dtype=np.float32)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ret1 = close / prev_close - 1.0
        out[:, 0] = ret1
        out[:, 1] = open_ / prev_close - 1.0
        out[:, 2] = close / open_ - 1.0
        out[:, 3] = high / open_ - 1.0
        out[:, 4] = low / open_ - 1.0
        range_hl = high / low - 1.0
        out[:, 5] = range_hl
        out[:, 6] = (close - low) / np.maximum(high - low, 1e-12)
    out[:, 7] = rolling_zscore(amount, 20)
    ret_series = pd.Series(ret1)
    for base, window in zip((8, 16, 24), (5, 20, 60), strict=True):
        out[:, base] = safe_div(close, np.roll(close, window)) - 1.0
        out[:window, base] = np.nan
        out[:, base + 1] = ret_series.rolling(window, min_periods=window).std(ddof=0).to_numpy(dtype=np.float32)
        out[:, base + 2] = rolling_max_drawdown(close, window)
        high_w = pd.Series(high).rolling(window, min_periods=window).max().to_numpy(dtype=np.float32)
        low_w = pd.Series(low).rolling(window, min_periods=window).min().to_numpy(dtype=np.float32)
        out[:, base + 3] = safe_div(close, high_w) - 1.0
        out[:, base + 4] = safe_div(close, low_w) - 1.0
        out[:, base + 5] = rolling_zscore(volume, window)
        out[:, base + 6] = rolling_zscore(amount, window)
        out[:, base + 7] = rolling_zscore(range_hl, window)
    return out


def rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values.astype(float))
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return ((series - mean) / std.replace(0, np.nan)).to_numpy(dtype=np.float32)


def rolling_max_drawdown(close: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float32)
    if len(close) < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(close.astype(float), window_shape=window)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    if not valid.any():
        return out
    peaks = np.maximum.accumulate(windows[valid], axis=1)
    drawdowns = np.min(windows[valid] / peaks - 1.0, axis=1).astype(np.float32)
    target = out[window - 1 :]
    target[valid] = drawdowns
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(b) & (b != 0))


def fill_rank_and_market(
    raw_ohlcv: np.memmap,
    raw_trade: np.memmap,
    universe: np.ndarray,
    shards: list[dict[str, Any]],
    shard_maps: dict[int, np.memmap],
    market: np.memmap,
) -> None:
    n_inst, n_dates = universe.shape
    amount_sums = np.full(n_dates, np.nan, dtype=np.float64)
    volume_sums = np.full(n_dates, np.nan, dtype=np.float64)
    rank_sources = [
        ("ret_close_1d_rank", "derived", 0),
        ("rolling_return_5_rank", "derived", 8),
        ("rolling_return_20_rank", "derived", 16),
        ("rolling_return_60_rank", "derived", 24),
        ("volatility_20_rank", "derived", 17),
        ("amount_rank", "raw_trade", 0),
        ("distance_to_high_20_rank", "derived", 19),
        ("distance_to_high_60_rank", "derived", 27),
        ("distance_to_low_20_rank", "derived", 20),
        ("range_hl_rank", "derived", 5),
    ]
    for date_idx in range(n_dates):
        day_mask = universe[:, date_idx]
        source_values: dict[int, np.ndarray] = {}
        for _, source, col in rank_sources:
            if source != "derived":
                continue
            values = np.full(n_inst, np.nan, dtype=np.float32)
            for shard in shards:
                start = int(shard["instrument_start"])
                end = int(shard["instrument_end"])
                values[start:end] = np.asarray(shard_maps[int(shard["shard_id"])][:, date_idx, col], dtype=np.float32)
            source_values[col] = values
        rank_outputs = []
        for _, source, col in rank_sources:
            values = raw_trade[:, date_idx, col].astype(float) if source == "raw_trade" else source_values[col]
            rank_outputs.append(rank_pct(values, day_mask))
        for shard in shards:
            shard_id = int(shard["shard_id"])
            start = int(shard["instrument_start"])
            end = int(shard["instrument_end"])
            block = np.vstack([rank[start:end] for rank in rank_outputs]).T.astype(np.float16)
            shard_maps[shard_id][:, date_idx, 32:42] = block
        market[date_idx] = market_features_for_day(raw_ohlcv, raw_trade, source_values[0], day_mask, date_idx)
        amount_sums[date_idx] = market[date_idx, 14]
        volume_sums[date_idx] = market[date_idx, 16]
        if date_idx == 0 or (date_idx + 1) % 250 == 0 or date_idx + 1 == n_dates:
            print(f"rank/market progress {date_idx + 1}/{n_dates}", flush=True)
    market[:, 15] = rolling_zscore(amount_sums, 20)
    market[:, 17] = rolling_zscore(volume_sums, 20)
    for mmap in shard_maps.values():
        mmap.flush()


def rank_pct(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    selected = mask & np.isfinite(values)
    count = int(selected.sum())
    if count == 0:
        return out
    order = np.argsort(values[selected], kind="mergesort")
    ranks = np.empty(count, dtype=np.float32)
    ranks[order] = (np.arange(count, dtype=np.float32) + 1.0) / float(count)
    out[np.flatnonzero(selected)] = ranks
    return out


def market_features_for_day(
    raw_ohlcv: np.memmap,
    raw_trade: np.memmap,
    ret1: np.ndarray,
    mask: np.ndarray,
    date_idx: int,
) -> np.ndarray:
    out = np.full(len(MARKET_FEATURE_FIELDS), np.nan, dtype=np.float32)
    selected = mask & np.isfinite(ret1)
    values = ret1[selected]
    count = int(values.size)
    if count == 0:
        return out
    out[0] = float(np.mean(values > 0))
    out[1] = float(np.mean(values < 0))
    out[2] = float(np.mean(values == 0))
    out[3] = float(np.mean(values))
    out[4] = float(np.median(values))
    out[5] = float(np.std(values))
    bins = [
        values < -0.10,
        (values >= -0.10) & (values < -0.05),
        (values >= -0.05) & (values < -0.01),
        (values >= -0.01) & (values < 0),
        (values > 0) & (values <= 0.01),
        (values > 0.01) & (values <= 0.05),
        (values > 0.05) & (values <= 0.10),
        values > 0.10,
    ]
    for offset, condition in enumerate(bins, start=6):
        out[offset] = float(np.mean(condition))
    amount = raw_trade[:, date_idx, 0].astype(float)
    volume = raw_ohlcv[:, date_idx, 4].astype(float)
    amount_values = amount[mask & np.isfinite(amount) & (amount > 0)]
    volume_values = volume[mask & np.isfinite(volume) & (volume > 0)]
    out[14] = float(np.sum(amount_values)) if amount_values.size else 0.0
    out[16] = float(np.sum(volume_values)) if volume_values.size else 0.0
    if amount_values.size and out[14] > 0:
        top_n = max(1, int(math.ceil(amount_values.size * 0.10)))
        out[18] = float(np.sum(np.sort(amount_values)[-top_n:]) / out[14])
    out[19] = float(np.percentile(values, 90) - np.percentile(values, 10))
    return out


def build_labels(args: argparse.Namespace, paths: DatasetPaths) -> None:
    _guard_outputs([paths.labels, paths.labels_meta, paths.valid_mask, paths.valid_mask_meta, paths.label_diag, paths.sample_index], args.force)
    instruments = pd.read_parquet(paths.instruments)
    calendar = pd.read_parquet(paths.calendar)
    n_inst = len(instruments)
    n_dates = len(calendar)
    raw_ohlcv = np.memmap(paths.raw_ohlcv, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
    raw_state = np.memmap(paths.raw_state, dtype=np.uint8, mode="r", shape=(n_inst, n_dates, len(RAW_STATE_FIELDS)))
    raw_trade = np.memmap(paths.raw_trade, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)))
    del raw_trade
    universe = load_universe_mask(paths, (n_inst, n_dates))
    shard_lookup = load_shard_lookup(paths)
    labels = np.memmap(paths.labels, dtype=np.int8, mode="w+", shape=(n_inst, n_dates))
    labels[:] = -1
    valid_mask = np.memmap(paths.valid_mask, dtype=np.uint8, mode="w+", shape=(n_inst, n_dates))
    valid_mask[:] = 0
    base_valid = price_state_valid(raw_ohlcv, raw_state)
    history_valid = rolling_count_2d(base_valid, SEQ_LEN) >= HISTORY_VALID_MIN_ROWS
    rules = ExecutionRulesV1()
    invalid_counts = {name: 0 for name in LABEL_INVALID_BITS}
    class_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    dates = calendar["date"].astype(str).to_numpy()
    years = calendar["year"].to_numpy(dtype=np.int16)
    sample_writer: pq.ParquetWriter | None = None
    diag_writer: pq.ParquetWriter | None = None
    row_id = 0
    try:
        for inst_row in instruments.itertuples(index=False):
            inst_idx = int(inst_row.instrument_idx)
            instrument = str(inst_row.instrument)
            shard_id, local_idx = shard_lookup[inst_idx]
            result = label_instrument_vectorized(
                raw_ohlcv[inst_idx],
                raw_state[inst_idx],
                universe[inst_idx],
                history_valid[inst_idx],
                rules,
                instrument,
            )
            labels[inst_idx] = result["labels"]
            valid_mask[inst_idx] = result["valid_mask"].astype(np.uint8)
            for name, value in result["invalid_counts"].items():
                invalid_counts[name] += int(value)
            for label, value in result["class_counts"].items():
                class_counts[int(label)] += int(value)
            valid_idx = np.flatnonzero(result["valid_mask"])
            if valid_idx.size:
                sample = pd.DataFrame(
                    {
                        "row_id": np.arange(row_id, row_id + valid_idx.size, dtype=np.int64),
                        "instrument_idx": np.full(valid_idx.size, inst_idx, dtype=np.int32),
                        "shard_id": np.full(valid_idx.size, shard_id, dtype=np.int16),
                        "local_instrument_idx": np.full(valid_idx.size, local_idx, dtype=np.int32),
                        "date_idx": valid_idx.astype(np.int32),
                        "instrument": np.full(valid_idx.size, instrument, dtype=object),
                        "signal_date": dates[valid_idx],
                        "label_end_date": dates[valid_idx + LABEL_HORIZON],
                        "year": years[valid_idx],
                        "label": result["labels"][valid_idx].astype(np.int8),
                        "entry_feasible": np.ones(valid_idx.size, dtype=bool),
                        "label_execution_invalid": np.zeros(valid_idx.size, dtype=bool),
                        "board": np.full(valid_idx.size, str(inst_row.board), dtype=object),
                        "asset_bucket": np.full(valid_idx.size, int(inst_row.asset_bucket), dtype=np.int8),
                    }
                )
                diag = pd.DataFrame(
                    {
                        "instrument_idx": np.full(valid_idx.size, inst_idx, dtype=np.int32),
                        "date_idx": valid_idx.astype(np.int32),
                        "instrument": np.full(valid_idx.size, instrument, dtype=object),
                        "signal_date": dates[valid_idx],
                        "entry_date": dates[valid_idx + 1],
                        "label_end_date": dates[valid_idx + LABEL_HORIZON],
                        "label": result["labels"][valid_idx].astype(np.int8),
                        "first_hit_day": result["first_hit_day"][valid_idx].astype(np.int8),
                        "entry_feasible": np.ones(valid_idx.size, dtype=bool),
                        "first_hit_exit_feasible": np.ones(valid_idx.size, dtype=bool),
                        "label_execution_invalid": np.zeros(valid_idx.size, dtype=bool),
                        "label_execution_rule_hash": np.full(valid_idx.size, rules.rule_hash, dtype=object),
                        "max_up_5d": result["max_up_5d"][valid_idx].astype(np.float32),
                        "max_down_5d": result["max_down_5d"][valid_idx].astype(np.float32),
                        "future_open_to_close_5d": result["future_open_to_close_5d"][valid_idx].astype(np.float32),
                        "future_open_to_open_5d": result["future_open_to_open_5d"][valid_idx].astype(np.float32),
                        "tie_hit": result["tie_hit"][valid_idx].astype(bool),
                        "board": np.full(valid_idx.size, str(inst_row.board), dtype=object),
                        "asset_bucket": np.full(valid_idx.size, int(inst_row.asset_bucket), dtype=np.int8),
                    }
                )
                sample_writer = write_parquet_batch(paths.sample_index, sample, sample_writer)
                diag_writer = write_parquet_batch(paths.label_diag, diag, diag_writer)
                row_id += int(valid_idx.size)
            if inst_idx == 0 or (inst_idx + 1) % 250 == 0 or inst_idx + 1 == n_inst:
                print(f"label progress {inst_idx + 1}/{n_inst} {instrument}, rows={row_id}", flush=True)
    finally:
        if sample_writer is not None:
            sample_writer.close()
        if diag_writer is not None:
            diag_writer.close()
    if row_id == 0:
        empty_sample_index().to_parquet(paths.sample_index, index=False)
        empty_label_diag().to_parquet(paths.label_diag, index=False)
    labels.flush()
    valid_mask.flush()
    class_counts[-1] = int(n_inst * n_dates - int(row_id))
    write_json(
        paths.labels_meta,
        memmap_meta(
            path=paths.labels,
            shape=(n_inst, n_dates),
            dtype="int8",
            axes=("instrument_idx", "date_idx"),
            fields=("label",),
            extra={
                "label_schema_hash": schema_hash(("SELL", "HOLD", "BUY")),
                "horizon": LABEL_HORIZON,
                "take_profit": TAKE_PROFIT,
                "stop_loss": STOP_LOSS,
                "execution_rule_hash": rules.rule_hash,
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "invalid_counts": invalid_counts,
                "sample_index_rows": int(row_id),
            },
        ),
    )
    write_json(
        paths.valid_mask_meta,
        memmap_meta(
            path=paths.valid_mask,
            shape=(n_inst, n_dates),
            dtype="uint8",
            axes=("instrument_idx", "date_idx"),
            fields=("valid_mask",),
            extra={
                "valid_count": int(valid_mask.sum()),
                "labels_sha256": file_sha256(paths.labels_meta),
                "universe_sha256": file_sha256(paths.universe_daily),
            },
        ),
    )


def write_parquet_batch(path: Path, frame: pd.DataFrame, writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def empty_sample_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": pd.Series(dtype="int64"),
            "instrument_idx": pd.Series(dtype="int32"),
            "shard_id": pd.Series(dtype="int16"),
            "local_instrument_idx": pd.Series(dtype="int32"),
            "date_idx": pd.Series(dtype="int32"),
            "instrument": pd.Series(dtype="object"),
            "signal_date": pd.Series(dtype="object"),
            "label_end_date": pd.Series(dtype="object"),
            "year": pd.Series(dtype="int16"),
            "label": pd.Series(dtype="int8"),
            "entry_feasible": pd.Series(dtype="bool"),
            "label_execution_invalid": pd.Series(dtype="bool"),
            "board": pd.Series(dtype="object"),
            "asset_bucket": pd.Series(dtype="int8"),
        }
    )


def empty_label_diag() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_idx": pd.Series(dtype="int32"),
            "date_idx": pd.Series(dtype="int32"),
            "instrument": pd.Series(dtype="object"),
            "signal_date": pd.Series(dtype="object"),
            "entry_date": pd.Series(dtype="object"),
            "label_end_date": pd.Series(dtype="object"),
            "label": pd.Series(dtype="int8"),
            "first_hit_day": pd.Series(dtype="int8"),
            "entry_feasible": pd.Series(dtype="bool"),
            "first_hit_exit_feasible": pd.Series(dtype="bool"),
            "label_execution_invalid": pd.Series(dtype="bool"),
            "label_execution_rule_hash": pd.Series(dtype="object"),
            "max_up_5d": pd.Series(dtype="float32"),
            "max_down_5d": pd.Series(dtype="float32"),
            "future_open_to_close_5d": pd.Series(dtype="float32"),
            "future_open_to_open_5d": pd.Series(dtype="float32"),
            "tie_hit": pd.Series(dtype="bool"),
            "board": pd.Series(dtype="object"),
            "asset_bucket": pd.Series(dtype="int8"),
        }
    )


def label_instrument_vectorized(
    raw: np.ndarray,
    state: np.ndarray,
    universe: np.ndarray,
    history_valid: np.ndarray,
    rules: ExecutionRulesV1,
    instrument: str,
) -> dict[str, Any]:
    n_dates = int(raw.shape[0])
    labels = np.full(n_dates, -1, dtype=np.int8)
    valid_mask = np.zeros(n_dates, dtype=bool)
    first_hit_day = np.zeros(n_dates, dtype=np.int8)
    tie_hit = np.zeros(n_dates, dtype=bool)
    max_up = np.full(n_dates, np.nan, dtype=np.float32)
    max_down = np.full(n_dates, np.nan, dtype=np.float32)
    future_open_to_close = np.full(n_dates, np.nan, dtype=np.float32)
    future_open_to_open = np.full(n_dates, np.nan, dtype=np.float32)
    invalid_code = np.zeros(n_dates, dtype=np.uint16)
    max_t = n_dates - LABEL_HORIZON
    if max_t <= 0:
        invalid_code[:] |= LABEL_INVALID_BITS["near_calendar_end"]
        return label_result_payload(labels, valid_mask, first_hit_day, tie_hit, max_up, max_down, future_open_to_close, future_open_to_open, invalid_code)

    date_idx = np.arange(max_t, dtype=np.int32)
    invalid_code[: min(SEQ_LEN - 1, n_dates)] |= LABEL_INVALID_BITS["window_too_early"]
    invalid_code[max_t:] |= LABEL_INVALID_BITS["near_calendar_end"]
    invalid_code[np.flatnonzero(~universe[:max_t])] |= LABEL_INVALID_BITS["not_in_universe"]
    invalid_code[np.flatnonzero(~history_valid[:max_t])] |= LABEL_INVALID_BITS["history_invalid"]

    open_ = raw[:, 0].astype(float)
    high = raw[:, 1].astype(float)
    low = raw[:, 2].astype(float)
    close = raw[:, 3].astype(float)
    volume = raw[:, 4].astype(float)
    entry_idx = date_idx + 1
    entry_open = open_[entry_idx]
    entry_high = high[entry_idx]
    entry_low = low[entry_idx]
    entry_close = close[entry_idx]
    entry_volume = volume[entry_idx]
    entry_state = state[entry_idx]
    preclose = close[date_idx]
    limit_rate = np.where(entry_state[:, 1] == 1, 0.05, rules.limit_rate(instrument))
    entry_base_ok = (
        (entry_state[:, 2] == 1)
        & (entry_state[:, 0] == 1)
        & (entry_state[:, 1] == 0)
        & np.isfinite(entry_open)
        & (entry_open > 0)
        & np.isfinite(entry_close)
        & (entry_close > 0)
        & np.isfinite(entry_volume)
        & (entry_volume > 0)
    )
    one_price_up = (
        one_price_bar_vector(entry_open, entry_high, entry_low, entry_close, rules.one_price_eps)
        & (safe_div(entry_close, preclose) - 1.0 >= 0.045 - rules.limit_eps)
    )
    open_limit_up = np.isfinite(preclose) & (preclose > 0) & (entry_open >= preclose * (1.0 + limit_rate) - rules.limit_eps)
    entry_ok = entry_base_ok & ~one_price_up & ~open_limit_up
    prelim = invalid_code[:max_t] == 0
    invalid_code[np.flatnonzero(prelim & ~entry_ok)] |= LABEL_INVALID_BITS["entry_not_feasible"]

    high_win = np.lib.stride_tricks.sliding_window_view(high[1:], window_shape=LABEL_HORIZON)
    low_win = np.lib.stride_tricks.sliding_window_view(low[1:], window_shape=LABEL_HORIZON)
    open_win = np.lib.stride_tricks.sliding_window_view(open_[1:], window_shape=LABEL_HORIZON)
    close_win = np.lib.stride_tricks.sliding_window_view(close[1:], window_shape=LABEL_HORIZON)
    path_valid = (
        np.isfinite(entry_open)
        & (entry_open > 0)
        & np.isfinite(high_win).all(axis=1)
        & np.isfinite(low_win).all(axis=1)
        & (high_win > 0).all(axis=1)
        & (low_win > 0).all(axis=1)
    )
    prelim = invalid_code[:max_t] == 0
    invalid_code[np.flatnonzero(prelim & ~path_valid)] |= LABEL_INVALID_BITS["future_path_invalid"]
    active = invalid_code[:max_t] == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        up_path = high_win / entry_open[:, np.newaxis] - 1.0
        down_path = low_win / entry_open[:, np.newaxis] - 1.0
        future_close = close_win[:, -1] / entry_open - 1.0
        future_open = open_win[:, -1] / entry_open - 1.0
    local_label = np.ones(max_t, dtype=np.int8)
    local_first_hit = np.zeros(max_t, dtype=np.int8)
    local_tie = np.zeros(max_t, dtype=bool)
    for offset in range(LABEL_HORIZON):
        unresolved = active & (local_first_hit == 0)
        up_hit = up_path[:, offset] >= TAKE_PROFIT
        down_hit = down_path[:, offset] <= -STOP_LOSS
        hit = unresolved & (up_hit | down_hit)
        if not hit.any():
            continue
        hit_pos = np.flatnonzero(hit)
        local_first_hit[hit_pos] = offset + 1
        both = up_hit[hit_pos] & down_hit[hit_pos]
        sell = down_hit[hit_pos] & ~up_hit[hit_pos]
        buy = up_hit[hit_pos] & ~down_hit[hit_pos]
        local_label[hit_pos[both]] = 1
        local_tie[hit_pos[both]] = True
        local_label[hit_pos[sell]] = 0
        local_label[hit_pos[buy]] = 2
        exit_idx = hit_pos + offset + 1
        exit_ok = can_sell_vector(raw, state, rules, instrument, exit_idx)
        bad = hit_pos[~exit_ok]
        if bad.size:
            invalid_code[bad] |= LABEL_INVALID_BITS["exit_not_feasible"]
    active = invalid_code[:max_t] == 0
    valid_dates = date_idx[active]
    labels[valid_dates] = local_label[active]
    valid_mask[valid_dates] = True
    first_hit_day[valid_dates] = local_first_hit[active]
    tie_hit[valid_dates] = local_tie[active]
    if path_valid.any():
        max_up[:max_t][path_valid] = np.nanmax(up_path[path_valid], axis=1).astype(np.float32)
        max_down[:max_t][path_valid] = np.nanmin(down_path[path_valid], axis=1).astype(np.float32)
    future_open_to_close[:max_t] = future_close.astype(np.float32)
    future_open_to_open[:max_t] = future_open.astype(np.float32)
    return label_result_payload(labels, valid_mask, first_hit_day, tie_hit, max_up, max_down, future_open_to_close, future_open_to_open, invalid_code)


def label_result_payload(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    first_hit_day: np.ndarray,
    tie_hit: np.ndarray,
    max_up: np.ndarray,
    max_down: np.ndarray,
    future_open_to_close: np.ndarray,
    future_open_to_open: np.ndarray,
    invalid_code: np.ndarray,
) -> dict[str, Any]:
    invalid_counts = {
        name: int(((invalid_code & bit) != 0).sum())
        for name, bit in LABEL_INVALID_BITS.items()
    }
    class_counts = {
        int(label): int((labels[valid_mask] == label).sum())
        for label in (0, 1, 2)
    }
    return {
        "labels": labels,
        "valid_mask": valid_mask,
        "first_hit_day": first_hit_day,
        "tie_hit": tie_hit,
        "max_up_5d": max_up,
        "max_down_5d": max_down,
        "future_open_to_close_5d": future_open_to_close,
        "future_open_to_open_5d": future_open_to_open,
        "invalid_counts": invalid_counts,
        "class_counts": class_counts,
    }


def one_price_bar_vector(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, eps: float) -> np.ndarray:
    return (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
        & (np.maximum(np.abs(high - low), np.abs(open_ - close)) <= eps)
    )


def can_sell_vector(raw: np.ndarray, state: np.ndarray, rules: ExecutionRulesV1, instrument: str, date_idx: np.ndarray) -> np.ndarray:
    open_ = raw[date_idx, 0].astype(float)
    high = raw[date_idx, 1].astype(float)
    low = raw[date_idx, 2].astype(float)
    close = raw[date_idx, 3].astype(float)
    volume = raw[date_idx, 4].astype(float)
    preclose = raw[date_idx - 1, 3].astype(float)
    row_state = state[date_idx]
    limit_rate = np.where(row_state[:, 1] == 1, 0.05, rules.limit_rate(instrument))
    base_ok = (
        (row_state[:, 2] == 1)
        & (row_state[:, 0] == 1)
        & (row_state[:, 1] == 0)
        & np.isfinite(open_)
        & (open_ > 0)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(volume)
        & (volume > 0)
    )
    one_price_down = one_price_bar_vector(open_, high, low, close, rules.one_price_eps) & (
        safe_div(close, preclose) - 1.0 <= -0.045 + rules.limit_eps
    )
    limit_down = np.isfinite(preclose) & (preclose > 0) & (close <= preclose * (1.0 - limit_rate) + rules.limit_eps)
    return base_ok & ~one_price_down & ~limit_down


def label_one_sample(
    raw_ohlcv: np.memmap,
    raw_state: np.memmap,
    rules: ExecutionRulesV1,
    instrument: str,
    inst_idx: int,
    date_idx: int,
) -> dict[str, Any]:
    entry_idx = date_idx + 1
    end_idx = date_idx + LABEL_HORIZON
    entry_open = float(raw_ohlcv[inst_idx, entry_idx, 0])
    entry_state = raw_state[inst_idx, entry_idx]
    can_buy, buy_reason = rules.can_buy(
        instrument,
        open_=entry_open,
        high=float(raw_ohlcv[inst_idx, entry_idx, 1]),
        low=float(raw_ohlcv[inst_idx, entry_idx, 2]),
        close=float(raw_ohlcv[inst_idx, entry_idx, 3]),
        preclose=float(raw_ohlcv[inst_idx, date_idx, 3]),
        volume=float(raw_ohlcv[inst_idx, entry_idx, 4]),
        tradestatus=int(entry_state[0]),
        is_st=int(entry_state[1]),
        has_raw_row=int(entry_state[2]),
    )
    if not can_buy:
        return {"invalid_code": LABEL_INVALID_BITS["entry_not_feasible"], "reason": buy_reason}
    highs = raw_ohlcv[inst_idx, entry_idx : end_idx + 1, 1].astype(float)
    lows = raw_ohlcv[inst_idx, entry_idx : end_idx + 1, 2].astype(float)
    closes = raw_ohlcv[inst_idx, entry_idx : end_idx + 1, 3].astype(float)
    opens = raw_ohlcv[inst_idx, entry_idx : end_idx + 1, 0].astype(float)
    if (
        not np.isfinite(entry_open)
        or entry_open <= 0
        or not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or np.any(highs <= 0)
        or np.any(lows <= 0)
    ):
        return {"invalid_code": LABEL_INVALID_BITS["future_path_invalid"], "reason": "invalid_future_path"}
    up_path = highs / entry_open - 1.0
    down_path = lows / entry_open - 1.0
    label = 1
    first_hit_day = 0
    tie_hit = False
    for offset in range(LABEL_HORIZON):
        up_hit = up_path[offset] >= TAKE_PROFIT
        down_hit = down_path[offset] <= -STOP_LOSS
        if not up_hit and not down_hit:
            continue
        first_hit_day = offset + 1
        if up_hit and down_hit:
            label = 1
            tie_hit = True
        elif down_hit:
            label = 0
        else:
            label = 2
        exit_idx = date_idx + first_hit_day
        exit_state = raw_state[inst_idx, exit_idx]
        can_sell, sell_reason = rules.can_sell(
            instrument,
            open_=float(raw_ohlcv[inst_idx, exit_idx, 0]),
            high=float(raw_ohlcv[inst_idx, exit_idx, 1]),
            low=float(raw_ohlcv[inst_idx, exit_idx, 2]),
            close=float(raw_ohlcv[inst_idx, exit_idx, 3]),
            preclose=float(raw_ohlcv[inst_idx, exit_idx - 1, 3]),
            volume=float(raw_ohlcv[inst_idx, exit_idx, 4]),
            tradestatus=int(exit_state[0]),
            is_st=int(exit_state[1]),
            has_raw_row=int(exit_state[2]),
        )
        if not can_sell:
            return {"invalid_code": LABEL_INVALID_BITS["exit_not_feasible"], "reason": sell_reason}
        break
    return {
        "invalid_code": 0,
        "label": int(label),
        "first_hit_day": int(first_hit_day),
        "tie_hit": bool(tie_hit),
        "max_up_5d": float(np.max(up_path)),
        "max_down_5d": float(np.min(down_path)),
        "future_open_to_close_5d": float(closes[-1] / entry_open - 1.0),
        "future_open_to_open_5d": float(opens[-1] / entry_open - 1.0),
    }


def price_state_valid(raw_ohlcv: np.memmap, raw_state: np.memmap) -> np.ndarray:
    open_ = raw_ohlcv[:, :, 0]
    high = raw_ohlcv[:, :, 1]
    low = raw_ohlcv[:, :, 2]
    close = raw_ohlcv[:, :, 3]
    volume = raw_ohlcv[:, :, 4]
    vwap = raw_ohlcv[:, :, 5]
    return (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(volume)
        & np.isfinite(vwap)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
        & (vwap > 0)
        & (high >= low)
        & (volume > 0)
        & (raw_state[:, :, 0] == 1)
        & (raw_state[:, :, 1] == 0)
        & (raw_state[:, :, 2] == 1)
    )


def rolling_count_2d(mask: np.ndarray, window: int) -> np.ndarray:
    values = mask.astype(np.int16)
    cumsum = np.cumsum(values, axis=1)
    out = cumsum.copy()
    if window < values.shape[1]:
        out[:, window:] = cumsum[:, window:] - cumsum[:, :-window]
    return out


def add_invalid_counts(counts: dict[str, int], invalid_code: int) -> None:
    for name, bit in LABEL_INVALID_BITS.items():
        if invalid_code & bit:
            counts[name] += 1


def shard_for_inst(shards: list[dict[str, Any]], inst_idx: int) -> tuple[int, int]:
    for shard in shards:
        start = int(shard["instrument_start"])
        end = int(shard["instrument_end"])
        if start <= inst_idx < end:
            return int(shard["shard_id"]), int(inst_idx - start)
    raise KeyError(f"No shard for instrument_idx={inst_idx}")


def load_shard_lookup(paths: DatasetPaths) -> dict[int, tuple[int, int]]:
    manifest = read_json(paths.features_manifest)
    lookup: dict[int, tuple[int, int]] = {}
    for shard in manifest["shards"]:
        start = int(shard["instrument_start"])
        end = int(shard["instrument_end"])
        for inst_idx in range(start, end):
            lookup[inst_idx] = (int(shard["shard_id"]), int(inst_idx - start))
    return lookup


def load_universe_mask(paths: DatasetPaths, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    parquet = pq.ParquetFile(paths.universe_daily)
    for batch in parquet.iter_batches(columns=["date_idx", "instrument_idx", "in_universe"], batch_size=1_000_000):
        table = pa.Table.from_batches([batch])
        date_idx = table["date_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        inst_idx = table["instrument_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        values = table["in_universe"].to_numpy(zero_copy_only=False).astype(bool)
        out[inst_idx, date_idx] = values
    return out


def _guard_outputs(paths: list[Path], force: bool) -> None:
    if force:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing artifacts without --force: " + ", ".join(existing[:5]))


if __name__ == "__main__":
    raise SystemExit(main())
