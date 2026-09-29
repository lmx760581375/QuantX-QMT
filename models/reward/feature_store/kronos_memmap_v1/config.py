"""Reward 模型统一特征存储的 schema 与路径。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "workdirs" / "feature_store"
DATA_DIR = EXPERIMENT_ROOT / "data"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
RUNS_DIR = EXPERIMENT_ROOT / "runs"

DEFAULT_PROVIDER_URI = REPO_ROOT / "data" / "qlib_data_fixed"
DEFAULT_RAW_STOCK_DIR = REPO_ROOT / "data" / "raw" / "baostock" / "stocks"
DEFAULT_SECURITY_MASTER = REPO_ROOT / "data" / "meta" / "snapshots" / "security_master.csv"

SEQ_LEN = 60
MIN_HISTORY_DAYS = 80
HISTORY_VALID_MIN_ROWS = 55
LABEL_HORIZON = 5
TAKE_PROFIT = 0.06
STOP_LOSS = 0.04
DEFAULT_SHARD_SIZE = 512

RAW_OHLCV_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "change", "factor")
RAW_TRADE_FIELDS = ("amount", "turnover")
RAW_STATE_FIELDS = ("tradestatus", "is_st", "has_raw_row", "reserved")

DERIVED_TS_FIELDS = (
    "ret_close_1d",
    "ret_open_1d",
    "ret_intraday",
    "high_open",
    "low_open",
    "range_hl",
    "close_position",
    "amount_zscore_20",
    "rolling_return_5",
    "rolling_volatility_5",
    "rolling_max_drawdown_5",
    "distance_to_high_5",
    "distance_to_low_5",
    "volume_zscore_5",
    "amount_zscore_5",
    "range_zscore_5",
    "rolling_return_20",
    "rolling_volatility_20",
    "rolling_max_drawdown_20",
    "distance_to_high_20",
    "distance_to_low_20",
    "volume_zscore_20",
    "amount_zscore_20_window",
    "range_zscore_20",
    "rolling_return_60",
    "rolling_volatility_60",
    "rolling_max_drawdown_60",
    "distance_to_high_60",
    "distance_to_low_60",
    "volume_zscore_60",
    "amount_zscore_60",
    "range_zscore_60",
)

RANK_FIELDS = (
    "ret_close_1d_rank",
    "rolling_return_5_rank",
    "rolling_return_20_rank",
    "rolling_return_60_rank",
    "volatility_20_rank",
    "amount_rank",
    "distance_to_high_20_rank",
    "distance_to_high_60_rank",
    "distance_to_low_20_rank",
    "range_hl_rank",
)

DERIVED_FEATURE_FIELDS = (*DERIVED_TS_FIELDS, *RANK_FIELDS)

MARKET_FEATURE_FIELDS = (
    "up_ratio",
    "down_ratio",
    "flat_ratio",
    "ret_close_1d_mean",
    "ret_close_1d_median",
    "ret_close_1d_std",
    "ret_lt_minus_10_ratio",
    "ret_minus_10_to_minus_5_ratio",
    "ret_minus_5_to_minus_1_ratio",
    "ret_minus_1_to_0_ratio",
    "ret_0_to_1_ratio",
    "ret_1_to_5_ratio",
    "ret_5_to_10_ratio",
    "ret_gt_10_ratio",
    "amount_sum",
    "amount_zscore_20",
    "volume_sum",
    "volume_zscore_20",
    "amount_top10_share",
    "ret_p90_minus_p10",
)

BOARD_TO_CODE = {
    "mainboard_sh": 0,
    "mainboard_sz": 1,
    "chinext": 2,
    "star": 3,
    "index": 4,
    "other": 5,
}
CODE_TO_BOARD = {value: key for key, value in BOARD_TO_CODE.items()}

UNIVERSE_REASON_BITS = {
    "unsupported_board": 1 << 0,
    "is_index": 1 << 1,
    "history_lt_80": 1 << 2,
    "missing_raw_row": 1 << 3,
    "tradestatus_not_1": 1 << 4,
    "is_st": 1 << 5,
    "invalid_ohlcv": 1 << 6,
    "zero_volume": 1 << 7,
}

LABEL_INVALID_BITS = {
    "not_in_universe": 1 << 0,
    "history_invalid": 1 << 1,
    "entry_not_feasible": 1 << 2,
    "future_path_invalid": 1 << 3,
    "exit_not_feasible": 1 << 4,
    "window_too_early": 1 << 5,
    "near_calendar_end": 1 << 6,
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path = EXPERIMENT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.json"

    @property
    def calendar(self) -> Path:
        return self.data / "calendar.parquet"

    @property
    def instruments(self) -> Path:
        return self.data / "instruments.parquet"

    @property
    def raw_ohlcv(self) -> Path:
        return self.data / "raw_ohlcv_float32.mmap"

    @property
    def raw_ohlcv_meta(self) -> Path:
        return self.data / "raw_ohlcv_meta.json"

    @property
    def raw_state(self) -> Path:
        return self.data / "raw_state_uint8.mmap"

    @property
    def raw_state_meta(self) -> Path:
        return self.data / "raw_state_meta.json"

    @property
    def raw_trade(self) -> Path:
        return self.data / "raw_trade_float32.mmap"

    @property
    def raw_trade_meta(self) -> Path:
        return self.data / "raw_trade_meta.json"

    @property
    def universe_daily(self) -> Path:
        return self.data / "universe_daily_v1.parquet"

    @property
    def universe_meta(self) -> Path:
        return self.data / "universe_daily_v1_meta.json"

    @property
    def features_manifest(self) -> Path:
        return self.data / "features_derived_v1_manifest.json"

    @property
    def market(self) -> Path:
        return self.data / "market_cross_section_v1_float32.mmap"

    @property
    def market_meta(self) -> Path:
        return self.data / "market_cross_section_v1_meta.json"

    @property
    def labels(self) -> Path:
        return self.data / "labels_tb_h5_tp006_sl004_int8.mmap"

    @property
    def labels_meta(self) -> Path:
        return self.data / "labels_tb_h5_tp006_sl004_meta.json"

    @property
    def label_diag(self) -> Path:
        return self.data / "label_diag_tb_h5_tp006_sl004.parquet"

    @property
    def valid_mask(self) -> Path:
        return self.data / "valid_mask_v1_uint8.mmap"

    @property
    def valid_mask_meta(self) -> Path:
        return self.data / "valid_mask_v1_meta.json"

    @property
    def sample_index(self) -> Path:
        return self.data / "sample_index_v1.parquet"

    @property
    def folds(self) -> Path:
        return self.data / "folds"

    @property
    def scalers(self) -> Path:
        return self.data / "scalers"


def board_of(instrument: str) -> str:
    if instrument.startswith("SH60"):
        return "mainboard_sh"
    if instrument.startswith("SZ00"):
        return "mainboard_sz"
    if instrument.startswith("SZ30"):
        return "chinext"
    if instrument.startswith("SH68"):
        return "star"
    if instrument.startswith("SZ399"):
        return "index"
    return "other"


def stable_asset_bucket(instrument: str, buckets: int = 5) -> int:
    digest = hashlib.sha256(instrument.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(buckets)


def schema_hash(fields: Iterable[str]) -> str:
    payload = json.dumps(tuple(fields), ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


MODEL_INPUT_SCHEMA_HASH = schema_hash(
    tuple(f"raw:{field}" for field in (*RAW_OHLCV_FIELDS, *RAW_TRADE_FIELDS))
    + tuple(f"derived:{field}" for field in DERIVED_FEATURE_FIELDS)
    + tuple(f"market:{field}" for field in MARKET_FEATURE_FIELDS)
    + ("label:SELL", "label:HOLD", "label:BUY", "seq_len:60")
)
