"""数据同步服务

从 BaoStock 拉取数据并同步到本地仓库，支持全量/增量模式。
多线程分片拉取，失败重试，ensure_coverage 检查间隙。
"""

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import pandas as pd

from .baostock_client import BaoStockClient
from .repository import LocalDataRepository

logger = logging.getLogger(__name__)

# BaoStock 非线程安全，全局锁保护 API 调用
_BS_LOCK = threading.Lock()


@dataclass
class SyncReport:
    """同步报告"""
    total_symbols: int = 0
    synced_count: int = 0
    failed_count: int = 0
    failed_symbols: List[str] = field(default_factory=list)
    updated_stocks: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""


class DataSyncService:
    """数据同步服务

    负责从 BaoStock 拉取数据并同步到本地 CSV 仓库。
    支持全量同步、增量同步、ensure_coverage。
    """

    def __init__(
        self,
        client: BaoStockClient,
        repository: LocalDataRepository,
        workers: int = 8,
        pause_seconds: float = 0.5,
        max_retries: int = 3,
        socket_timeout: float = 30.0,
    ):
        self.client = client
        self.repository = repository
        self.workers = workers
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.socket_timeout = float(socket_timeout)

    # ============================================================
    # 全量同步
    # ============================================================

    def sync_full(
        self,
        symbols: List[str],
        start: str,
        end: str,
        progress_callback: Callable[[SyncReport], None] | None = None,
    ) -> SyncReport:
        """全量同步所有股票数据

        Args:
            symbols: 股票代码列表
            start: 起始日期
            end: 结束日期
        """
        report = SyncReport(
            total_symbols=len(symbols),
            start_time=start,
            end_time=end,
        )
        logger.info(f"Starting full sync: {len(symbols)} symbols, {start} ~ {end}")

        with BaoStockClient(
            pause_seconds=self.pause_seconds,
            max_retries=self.max_retries,
            socket_timeout=self.socket_timeout,
        ) as client:
            self.client = client  # 使用新的 client（已登录）
            failed = self._sync_batch(
                symbols,
                start,
                end,
                report,
                progress_callback=progress_callback,
            )

        # 失败重试
        if failed:
            logger.warning(f"Retrying {len(failed)} failed symbols")
            retry_report = SyncReport(
                total_symbols=len(failed),
                start_time=start,
                end_time=end,
            )
            with BaoStockClient(
                pause_seconds=self.pause_seconds * 2,
                max_retries=self.max_retries,
                socket_timeout=self.socket_timeout,
            ) as client:
                self.client = client
                remaining = self._sync_batch(
                    failed,
                    start,
                    end,
                    retry_report,
                    progress_callback=progress_callback,
                )
            report.synced_count += retry_report.synced_count
            report.updated_stocks.extend(retry_report.updated_stocks)
            report.failed_symbols = remaining
            report.failed_count = len(remaining)
            if progress_callback is not None:
                progress_callback(report)

        logger.info(
            f"Full sync complete: {report.synced_count} synced, "
            f"{report.failed_count} failed"
        )
        return report

    # ============================================================
    # 增量同步
    # ============================================================

    def sync_incremental(self, symbols: List[str]) -> SyncReport:
        """增量同步：只拉取每个股票的新数据

        Args:
            symbols: 股票代码列表
        """
        report = SyncReport(total_symbols=len(symbols))
        logger.info(f"Starting incremental sync: {len(symbols)} symbols")

        with BaoStockClient(
            pause_seconds=self.pause_seconds,
            max_retries=self.max_retries,
            socket_timeout=self.socket_timeout,
        ) as client:
            self.client = client

            for symbol in symbols:
                try:
                    last_date = self.repository.get_last_date(symbol)
                    if last_date is None:
                        # 没有本地数据，全量拉取（最近 10 年）
                        start = "2015-01-01"
                        df = self.client.query_history_k_data_with_retry(
                            BaoStockClient.to_baostock_format(symbol),
                            start,
                            pd.Timestamp.now().strftime("%Y-%m-%d"),
                        )
                    else:
                        # 只拉取新数据
                        df = self.client.query_history_k_data_with_retry(
                            BaoStockClient.to_baostock_format(symbol),
                            last_date,
                            pd.Timestamp.now().strftime("%Y-%m-%d"),
                        )

                    if not df.empty:
                        self.repository.save_symbol(symbol, df)
                        report.synced_count += 1
                        report.updated_stocks.append(symbol)
                    else:
                        report.failed_count += 1
                        report.failed_symbols.append(symbol)

                except Exception as e:
                    logger.error(f"Failed to sync {symbol}: {e}")
                    report.failed_count += 1
                    report.failed_symbols.append(symbol)

        logger.info(f"Incremental sync complete: {report.synced_count} synced")
        return report

    # ============================================================
    # 覆盖范围检查
    # ============================================================

    def ensure_coverage(
        self, symbols: List[str], start: str, end: str
    ) -> SyncReport:
        """检查数据覆盖范围，填补间隙

        1. 检查每个股票的首尾日期是否覆盖 [start, end]
        2. 对缺失的股票和日期范围进行同步
        """
        report = SyncReport(total_symbols=len(symbols))
        logger.info(f"Checking coverage: {len(symbols)} symbols, {start} ~ {end}")

        coverage = self.repository.get_coverage_report(start, end)

        # 完全缺失的股票：全量同步
        if coverage["missing_stocks"]:
            logger.info(f"Syncing {len(coverage['missing_stocks'])} missing stocks")
            self.sync_full(coverage["missing_stocks"], start, end)

        # 检查每个股票的日期范围
        need_sync = []
        for sym in symbols:
            first = self.repository.get_first_date(sym)
            last = self.repository.get_last_date(sym)
            if first is None or last is None:
                need_sync.append(sym)
            elif first > start or last < end:
                need_sync.append(sym)

        if need_sync:
            logger.info(f"Syncing {len(need_sync)} stocks with incomplete coverage")
            self.sync_full(need_sync, start, end)

        report.synced_count = len(need_sync) + len(coverage["missing_stocks"])
        return report

    # ============================================================
    # 内部方法
    # ============================================================

    def _sync_batch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        report: SyncReport,
        progress_callback: Callable[[SyncReport], None] | None = None,
    ) -> List[str]:
        """批量同步，限制在途任务数量，避免一个卡住请求拖垮整个队列。"""
        failed = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            pending = {}
            symbol_iter = iter(symbols)

            def submit_next() -> None:
                try:
                    symbol = next(symbol_iter)
                except StopIteration:
                    return
                future = executor.submit(self._sync_single, symbol, start, end)
                pending[future] = symbol

            for _ in range(min(max(1, int(self.workers)), len(symbols))):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    symbol = pending.pop(future)
                    try:
                        df, success = future.result()
                        if success and not df.empty:
                            self.repository.save_symbol(symbol, df)
                            report.synced_count += 1
                            report.updated_stocks.append(symbol)
                        else:
                            failed.append(symbol)
                            report.failed_count += 1
                            report.failed_symbols.append(symbol)
                    except Exception as e:
                        logger.error(f"Sync failed for {symbol}: {e}")
                        failed.append(symbol)
                        report.failed_count += 1
                        report.failed_symbols.append(symbol)
                    if progress_callback is not None:
                        progress_callback(report)
                    submit_next()

        return failed

    def _sync_single(
        self, symbol: str, start: str, end: str
    ) -> Tuple[pd.DataFrame, bool]:
        """同步单只股票"""
        baostock_sym = BaoStockClient.to_baostock_format(symbol)
        with _BS_LOCK:  # BaoStock 非线程安全
            df = self.client.query_history_k_data_with_retry(
                baostock_sym, start, end
            )
        return df, not df.empty
