"""数据层

提供数据源抽象、BaoStock 接入、Qlib 格式转换、数据同步等功能。
"""

from .base import DataSource, OHLCV_FIELDS, EXTENDED_FIELDS, QLIB_FIELDS
from .baostock_client import BaoStockClient
from .baostock_source import BaoStockDataSource, BaoStockConfig
from .factory import create_data_source
from .qmt_client import QMTClient
from .qmt_source import QMTConfig, QMTDataSource
from .calendar import TradingCalendar
from .adjuster import Adjuster
from .converter import BaostockToQlibConverter
from .incremental_sync import AdjustmentAwareIncrementalUpdater, IncrementalUpdateReport, SymbolUpdateResult
from .repository import LocalDataRepository
from .sync_service import DataSyncService, SyncReport
from .validator import DataValidator
from .cache import DataCache

__all__ = [
    "DataSource",
    "OHLCV_FIELDS",
    "EXTENDED_FIELDS",
    "QLIB_FIELDS",
    "BaoStockClient",
    "BaoStockDataSource",
    "BaoStockConfig",
    "QMTClient",
    "QMTDataSource",
    "QMTConfig",
    "create_data_source",
    "TradingCalendar",
    "Adjuster",
    "BaostockToQlibConverter",
    "AdjustmentAwareIncrementalUpdater",
    "IncrementalUpdateReport",
    "SymbolUpdateResult",
    "LocalDataRepository",
    "DataSyncService",
    "SyncReport",
    "DataValidator",
    "DataCache",
]
