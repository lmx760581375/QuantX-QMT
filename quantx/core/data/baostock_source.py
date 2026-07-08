"""BaoStock 数据源实现

实现 DataSource ABC，封装 BaoStockClient + LocalDataRepository + Converter，
对外提供统一的数据访问接口。这是策略层和回测引擎访问数据的唯一入口。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from quantx.core.qlib_import import import_qlib

from .base import DataSource
from .baostock_client import BaoStockClient
from .calendar import TradingCalendar
from .converter import BaostockToQlibConverter
from .qlib_reader import QlibBinReader
from .repository import LocalDataRepository
from .sync_service import DataSyncService

logger = logging.getLogger(__name__)


@dataclass
class BaoStockConfig:
    """BaoStock 数据源配置"""
    data_root: str = "data/raw/baostock"
    qlib_dir: str = "data/qlib_data"
    adjust: str = "forward"  # forward / none
    workers: int = 8
    pause_seconds: float = 0.5
    max_retries: int = 3
    auto_update: bool = True


class QlibInitializer:
    """Qlib 全局初始化器（单例模式，确保只初始化一次）"""
    _initialized: bool = False
    _provider_uri: str = ""

    @classmethod
    def init(cls, provider_uri: str, region: str = "cn") -> None:
        if not cls._initialized or cls._provider_uri != provider_uri:
            qlib = import_qlib()
            qlib.init(provider_uri=provider_uri, region=region)
            cls._initialized = True
            cls._provider_uri = provider_uri
            logger.info(f"Qlib initialized: provider_uri={provider_uri}, region={region}")

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized


class BaoStockDataSource(DataSource):
    """BaoStock 数据源实现

    实现 DataSource ABC，内部使用：
    - BaoStockClient: API 调用
    - LocalDataRepository: 本地 CSV 存储
    - DataSyncService: 数据同步
    - BaostockToQlibConverter: Qlib 格式转换
    - D.features(): Qlib 数据读取
    """

    def __init__(self, config: Optional[BaoStockConfig] = None):
        self.config = config or BaoStockConfig()
        self.client = BaoStockClient(
            pause_seconds=self.config.pause_seconds,
            max_retries=self.config.max_retries,
        )
        self.repository = LocalDataRepository(self.config.data_root)
        self.sync_service = DataSyncService(
            self.client, self.repository,
            workers=self.config.workers,
            pause_seconds=self.config.pause_seconds,
        )
        self.converter = BaostockToQlibConverter(
            qlib_dir=self.config.qlib_dir,
            csv_dir=str(Path(self.config.data_root) / "stocks"),
        )
        self.calendar: Optional[TradingCalendar] = None

    # ============================================================
    # DataSource 接口实现
    # ============================================================

    def get_daily_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        adjust: str = "forward",
    ) -> pd.DataFrame:
        """获取日线行情数据

        从 Qlib 二进制数据读取，如果数据未就绪则自动触发同步+转换。
        """
        # 确保 Qlib 数据就绪
        self._ensure_ready(start_date, end_date, symbols)

        if fields is None:
            qlib_fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$change"]
        else:
            qlib_fields = [f"${f}" for f in fields]

        try:
            QlibInitializer.init(self.config.qlib_dir, "cn")
            from qlib.data import D

            df = D.features(symbols, qlib_fields, start_date, end_date, freq="day")
        except Exception as e:
            logger.warning("Falling back to built-in qlib bin reader for BaoStock data: %s", e)
            df = QlibBinReader(self.config.qlib_dir).features(symbols, qlib_fields, start_date, end_date)
        # 去掉 $ 前缀
        rename_map = {f"${f}": f for f in ["open", "high", "low", "close", "volume", "vwap", "change"]}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    def get_dividend_factors(self, symbols: List[str]) -> pd.DataFrame:
        """获取复权因子"""
        all_factors = []
        with BaoStockClient(pause_seconds=self.config.pause_seconds) as client:
            for sym in symbols:
                baostock_sym = BaoStockClient.to_baostock_format(sym)
                df = client.query_dividend_data(baostock_sym, "2010-01-01", "2030-12-31")
                if not df.empty:
                    df["symbol"] = sym
                    all_factors.append(df)
        if not all_factors:
            return pd.DataFrame()
        return pd.concat(all_factors, ignore_index=True)

    def get_stock_list(self, date: Optional[str] = None) -> List[str]:
        """获取股票列表"""
        if date is not None:
            # 从 Qlib instruments 读取
            try:
                from qlib.data import D
                return D.instruments("all")
            except Exception:
                pass
        # 从本地仓库读取
        return self.repository.get_stock_codes()

    def get_benchmark(
        self, benchmark: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取基准指数数据"""
        self._ensure_ready(start_date, end_date, [benchmark])

        try:
            QlibInitializer.init(self.config.qlib_dir, "cn")
            from qlib.data import D

            return D.features([benchmark], ["$close"], start_date, end_date, freq="day")
        except Exception as exc:
            logger.warning("Falling back to built-in qlib bin reader for benchmark %s: %s", benchmark, exc)
            return QlibBinReader(self.config.qlib_dir).features([benchmark], ["$close"], start_date, end_date)

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        if self.calendar is None:
            self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)
        return self.calendar.get_trading_days(start_date, end_date)

    def is_ready(self, start_date: str, end_date: str) -> bool:
        """检查 Qlib 数据是否就绪"""
        return self.converter.check_coverage(start_date, end_date)

    # ============================================================
    # 高层操作
    # ============================================================

    def sync_data(
        self, symbols: Optional[List[str]] = None, start: str = "2015-01-01", end: Optional[str] = None
    ) -> None:
        """同步数据：拉取 + 转换

        Args:
            symbols: 股票列表，None 表示全量
            start: 起始日期
            end: 结束日期，None 表示今天
        """
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")

        if symbols is None:
            symbols = self.repository.get_stock_codes()

        if not symbols:
            # 首次使用：先获取全量股票列表
            logger.info("No local data found. Fetching stock list first.")
            symbols = self._fetch_stock_list()

        # 1. 同步数据到 CSV
        self.sync_service.sync_full(symbols, start, end)

        # 2. 转换为 Qlib 格式
        self.converter.convert_all(start, end, symbols)

        # 3. 初始化日历
        self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)

    def sync_incremental(self) -> None:
        """增量同步"""
        symbols = self.repository.get_stock_codes()
        if not symbols:
            logger.warning("No stocks to sync. Run sync_data first.")
            return

        # 1. 增量同步 CSV
        report = self.sync_service.sync_incremental(symbols)

        # 2. 增量转换
        if report.updated_stocks:
            self.converter.convert_incremental(report.updated_stocks)

    # ============================================================
    # 内部方法
    # ============================================================

    def _ensure_ready(self, start_date: str, end_date: str, symbols: List[str]) -> None:
        """确保 Qlib 数据就绪，如果未就绪则自动同步"""
        if self.is_ready(start_date, end_date):
            return

        if self.config.auto_update:
            logger.info(f"Data not ready for {start_date}~{end_date}, auto-syncing...")
            # 检查本地 CSV 是否已有数据
            existing = self.repository.get_stock_codes()
            if existing:
                self.sync_service.ensure_coverage(symbols, start_date, end_date)
                self.converter.convert_all(start_date, end_date, symbols)
            else:
                self.sync_data(symbols, start_date, end_date)
        else:
            raise RuntimeError(
                f"Data not ready for {start_date}~{end_date}. "
                "Run sync_data() first or set auto_update=True."
            )

        # 初始化 Qlib
        QlibInitializer.init(self.config.qlib_dir, "cn")

        # 初始化日历
        if self.calendar is None:
            self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)

    def _fetch_stock_list(self) -> List[str]:
        """从 BaoStock 获取全量股票列表"""
        with BaoStockClient(pause_seconds=self.config.pause_seconds) as client:
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            stocks_df = client.query_all_stocks(today)
            if stocks_df.empty:
                raise RuntimeError("Failed to fetch stock list from BaoStock")

            # 保存股票列表
            self.repository.save_universe(stocks_df)

            # 返回所有 A 股代码（排除 B 股、指数等）
            codes = []
            for _, row in stocks_df.iterrows():
                code = row["code"]
                # A 股：sh.60xxxx (主板), sh.68xxxx (科创板), sz.00xxxx (主板), sz.30xxxx (创业板)
                if code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3"):
                    codes.append(BaoStockClient._normalize_symbol(code))
            return codes
