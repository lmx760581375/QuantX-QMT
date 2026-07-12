"""QMT data source backed by a remote xqshare service."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import DataSource
from .baostock_source import QlibInitializer
from .calendar import TradingCalendar
from .converter import BaostockToQlibConverter
from .qmt_client import QMTClient, normalize_symbol
from .qlib_reader import QlibBinReader
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
    workers: int = 8
    fill_data: bool = True
    env_file: str | None = None


class QMTDataSource(DataSource):
    """DataSource implementation that ingests market data from QMT xtdata."""

    def __init__(self, config: Optional[QMTConfig] = None):
        self.config = config or QMTConfig()
        self.client = QMTClient(
            dividend_type=self.config.dividend_type,
            fill_data=self.config.fill_data,
            pause_seconds=self.config.pause_seconds,
            max_retries=self.config.max_retries,
            env_file=self.config.env_file,
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
        qlib_fields = _qlib_fields(fields)
        frame = self._load_qlib_features(normalized, qlib_fields, start_date, end_date)
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

        try:
            return self._load_qlib_features([normalized], ["$close"], start_date, end_date)
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

        requests = [(symbol, self.repository.get_last_date(symbol) or "2015-01-01", end) for symbol in symbol_list]
        report = self._sync_requests(requests, replace=False)
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
        requests = [(symbol, start, end) for symbol in symbols]
        return self._sync_requests(requests, replace=replace)

    def _sync_requests(self, requests: List[tuple[str, str, str]], replace: bool) -> SyncReport:
        start_time = min((start for _, start, _ in requests), default="")
        end_time = max((end for _, _, end in requests), default="")
        report = SyncReport(total_symbols=len(requests), start_time=start_time, end_time=end_time)
        if not requests:
            return report

        download_ok, download_failed = self._download_requests(requests)
        if download_failed:
            report.failed_count += len(download_failed)
            report.failed_symbols.extend(download_failed)
        if download_ok:
            ok_symbols = set(download_ok)
            requests = [request for request in requests if request[0] in ok_symbols]
        else:
            logger.warning("No QMT downloads succeeded for %s requested symbols", report.total_symbols)
            report.failed_count = report.total_symbols
            report.failed_symbols = [symbol for symbol, _, _ in requests]
            return report

        workers = max(1, int(self.config.workers))
        if workers == 1:
            for symbol, start, end in requests:
                frame = self._fetch_symbol(symbol, start, end)
                self._record_sync_result(report, symbol, frame, replace=replace)
            return report

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol, symbol, start, end): symbol
                for symbol, start, end in requests
            }
            done = 0
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    frame = future.result()
                except Exception as exc:  # pragma: no cover - provider errors are environment-specific.
                    logger.warning("QMT sync failed for %s: %s", symbol, exc)
                    frame = pd.DataFrame()
                self._record_sync_result(report, symbol, frame, replace=replace)
                done += 1
                if done == 1 or done % 50 == 0 or done == report.total_symbols:
                    logger.info(
                        "QMT sync progress %s/%s, synced=%s, failed=%s, latest=%s",
                        done,
                        report.total_symbols,
                        report.synced_count,
                        report.failed_count,
                        symbol,
                    )
        return report

    def _download_requests(self, requests: List[tuple[str, str, str]]) -> tuple[List[str], List[str]]:
        workers = max(1, int(self.config.workers))
        ok: list[str] = []
        failed: list[str] = []
        logger.info("QMT download batch start total=%s workers=%s", len(requests), workers)

        if workers == 1:
            for idx, (symbol, start, end) in enumerate(requests, start=1):
                if self._download_symbol(symbol, start, end):
                    ok.append(symbol)
                else:
                    failed.append(symbol)
                if idx == 1 or idx % 50 == 0 or idx == len(requests):
                    logger.info("QMT download progress %s/%s, ok=%s, failed=%s", idx, len(requests), len(ok), len(failed))
            logger.info("QMT download batch done ok=%s failed=%s", len(ok), len(failed))
            return ok, failed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._download_symbol, symbol, start, end): symbol
                for symbol, start, end in requests
            }
            done = 0
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    success = bool(future.result())
                except Exception as exc:  # pragma: no cover - provider errors are environment-specific.
                    logger.warning("QMT download failed for %s: %s", symbol, exc)
                    success = False
                if success:
                    ok.append(symbol)
                else:
                    failed.append(symbol)
                done += 1
                if done == 1 or done % 50 == 0 or done == len(requests):
                    logger.info("QMT download progress %s/%s, ok=%s, failed=%s, latest=%s", done, len(requests), len(ok), len(failed), symbol)

        if failed:
            logger.warning("QMT download failed symbols count=%s sample=%s", len(failed), failed[:20])
        logger.info("QMT download batch done ok=%s failed=%s", len(ok), len(failed))
        return ok, failed

    def _download_symbol(self, symbol: str, start: str, end: str) -> bool:
        with self._new_client() as client:
            return client.download_history_data(symbol, start, end)

    def _fetch_symbol(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        with self._new_client() as client:
            return client.query_history_k_data_with_retry(symbol, start, end, download_first=False)

    def _record_sync_result(self, report: SyncReport, symbol: str, frame: pd.DataFrame, replace: bool) -> None:
        if frame.empty:
            report.failed_count += 1
            report.failed_symbols.append(symbol)
            return
        if replace:
            self.repository.replace_symbol(symbol, frame)
        else:
            self.repository.save_symbol(symbol, frame)
        report.synced_count += 1
        report.updated_stocks.append(symbol)

    def _new_client(self) -> QMTClient:
        return QMTClient(
            dividend_type=self.config.dividend_type,
            fill_data=self.config.fill_data,
            pause_seconds=self.config.pause_seconds,
            max_retries=self.config.max_retries,
            env_file=self.config.env_file,
        )

    def _load_qlib_features(self, symbols: List[str], fields: List[str], start: str, end: str) -> pd.DataFrame:
        try:
            QlibInitializer.init(self.config.qlib_dir, "cn")
            from qlib.data import D

            return D.features(symbols, fields, start, end, freq="day")
        except Exception as exc:
            logger.warning("Falling back to built-in qlib bin reader for QMT data: %s", exc)
            return QlibBinReader(self.config.qlib_dir).features(symbols, fields, start, end)


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
