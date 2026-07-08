"""QMT data source backed by the installed xtquant package."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import DataSource
from .baostock_source import QlibInitializer
from .calendar import TradingCalendar
from .converter import BaostockToQlibConverter
from .qmt_client import QMTClient, normalize_symbol
from .repository import LocalDataRepository
from .sync_service import SyncReport

logger = logging.getLogger(__name__)


@dataclass
class QMTConfig:
    data_root: str = "data/raw/qmt"
    qlib_dir: str = "data/qlib_data_fixed"
    adjust: str = "forward"
    sectors: tuple[str, ...] = ("沪深A股",)
    dividend_type: str = "front_ratio"
    auto_update: bool = True
    pause_seconds: float = 0.0
    max_retries: int = 3


class QMTDataSource(DataSource):
    """DataSource implementation that ingests market data from QMT xtdata."""

    def __init__(self, config: Optional[QMTConfig] = None):
        self.config = config or QMTConfig()
        self.client = QMTClient(
            dividend_type=self.config.dividend_type,
            pause_seconds=self.config.pause_seconds,
            max_retries=self.config.max_retries,
        )
        self.repository = LocalDataRepository(self.config.data_root)
        self.converter = BaostockToQlibConverter(
            qlib_dir=self.config.qlib_dir,
            csv_dir=str(Path(self.config.data_root) / "stocks"),
        )
        self.calendar: Optional[TradingCalendar] = None

    def get_daily_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        adjust: str = "forward",
    ) -> pd.DataFrame:
        normalized = [normalize_symbol(symbol) for symbol in symbols]
        self._ensure_ready(start_date, end_date, normalized)
        QlibInitializer.init(self.config.qlib_dir, "cn")

        from qlib.data import D

        qlib_fields = _qlib_fields(fields)
        frame = D.features(normalized, qlib_fields, start_date, end_date, freq="day")
        return _strip_qlib_prefix(frame)

    def get_dividend_factors(self, symbols: List[str]) -> pd.DataFrame:
        frames = []
        with self._new_client() as client:
            for symbol in symbols:
                frame = client.query_dividend_data(symbol)
                if not frame.empty:
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_stock_list(self, date: Optional[str] = None) -> List[str]:
        local = self.repository.get_stock_codes()
        if local:
            return local
        with self._new_client() as client:
            symbols = client.get_stock_codes(self.config.sectors)
        if symbols:
            self.repository.save_universe(pd.DataFrame({"code": symbols}))
        return symbols

    def get_benchmark(self, benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
        normalized = normalize_symbol(benchmark)
        self._ensure_ready(start_date, end_date, [normalized])
        QlibInitializer.init(self.config.qlib_dir, "cn")

        from qlib.data import D

        try:
            return D.features([normalized], ["$close"], start_date, end_date, freq="day")
        except Exception:
            logger.warning("Benchmark %s not available from QMT provider", benchmark)
            return pd.DataFrame()

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        if self.calendar is None:
            self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)
        days = self.calendar.get_trading_days(start_date, end_date)
        if days:
            return days
        with self._new_client() as client:
            return client.get_trading_dates(start_date, end_date)

    def is_ready(self, start_date: str, end_date: str) -> bool:
        return self.converter.check_coverage(start_date, end_date)

    def sync_data(
        self,
        symbols: Optional[List[str]] = None,
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> SyncReport:
        end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
        symbol_list = [normalize_symbol(symbol) for symbol in symbols] if symbols else self.get_stock_list()
        if not symbol_list:
            raise RuntimeError("QMT returned no stock symbols")
        report = self._sync_symbols(symbol_list, start, end, replace=True)
        if report.updated_stocks:
            self.converter.convert_all(start, end, report.updated_stocks)
            self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)
        return report

    def sync_incremental(self, symbols: Optional[List[str]] = None, end: Optional[str] = None) -> SyncReport:
        end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
        symbol_list = [normalize_symbol(symbol) for symbol in symbols] if symbols else self.repository.get_stock_codes()
        if not symbol_list:
            return self.sync_data(start="2015-01-01", end=end)

        report = SyncReport(total_symbols=len(symbol_list), end_time=end)
        with self._new_client() as client:
            for symbol in symbol_list:
                start = self.repository.get_last_date(symbol) or "2015-01-01"
                frame = client.query_history_k_data_with_retry(symbol, start, end)
                if frame.empty:
                    report.failed_count += 1
                    report.failed_symbols.append(symbol)
                    continue
                self.repository.save_symbol(symbol, frame)
                report.synced_count += 1
                report.updated_stocks.append(symbol)
        if report.updated_stocks:
            self.converter.convert_incremental(report.updated_stocks)
        return report

    def _ensure_ready(self, start_date: str, end_date: str, symbols: List[str]) -> None:
        if self.is_ready(start_date, end_date):
            return
        if not self.config.auto_update:
            raise RuntimeError(
                f"QMT data not ready for {start_date}~{end_date}. "
                "Run sync_data() first or set auto_update=True."
            )
        existing = set(self.repository.get_stock_codes())
        target = symbols if symbols else sorted(existing)
        if not target:
            target = self.get_stock_list()
        if existing:
            self._sync_symbols(target, start_date, end_date, replace=False)
            self.converter.convert_all(start_date, end_date, target)
        else:
            self.sync_data(target, start_date, end_date)
        self.calendar = TradingCalendar(provider_uri=self.config.qlib_dir)

    def _sync_symbols(self, symbols: List[str], start: str, end: str, replace: bool) -> SyncReport:
        report = SyncReport(total_symbols=len(symbols), start_time=start, end_time=end)
        with self._new_client() as client:
            for symbol in symbols:
                frame = client.query_history_k_data_with_retry(symbol, start, end)
                if frame.empty:
                    report.failed_count += 1
                    report.failed_symbols.append(symbol)
                    continue
                if replace:
                    self.repository.replace_symbol(symbol, frame)
                else:
                    self.repository.save_symbol(symbol, frame)
                report.synced_count += 1
                report.updated_stocks.append(symbol)
        return report

    def _new_client(self) -> QMTClient:
        return QMTClient(
            dividend_type=self.config.dividend_type,
            pause_seconds=self.config.pause_seconds,
            max_retries=self.config.max_retries,
        )


def qmt_config_from_mapping(data: dict | None) -> QMTConfig:
    raw = dict(data or {})
    allowed = {field.name for field in fields(QMTConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    if "sectors" in values and isinstance(values["sectors"], list):
        values["sectors"] = tuple(values["sectors"])
    return QMTConfig(**values)


def _qlib_fields(fields: Optional[List[str]]) -> List[str]:
    if fields is None:
        return ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$change"]
    return [field if str(field).startswith("$") else f"${field}" for field in fields]


def _strip_qlib_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {column: str(column).lstrip("$") for column in frame.columns if str(column).startswith("$")}
    return frame.rename(columns=rename)
