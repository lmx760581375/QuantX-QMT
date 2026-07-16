"""回测上下文

提供策略在回测中访问数据的统一接口。
从 Qlib 加载行情 + 因子数据，管理交易日历和信号矩阵。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd

from .exchange import AStockExchange
from .types import Signal

logger = logging.getLogger(__name__)


def _quote_fields_from_aliases(field_aliases: Optional[Dict[str, str]]) -> List[str]:
    fields: List[str] = []
    for source in (field_aliases or {}).values():
        text = str(source).strip()
        if not text:
            continue
        if text.startswith("$"):
            fields.append(text)
        elif text.replace("_", "").isalnum():
            fields.append(f"${text}")
    return fields

LOOKBACK_COVERAGE_TOLERANCE_DAYS = 10


def compute_load_start(start: str, look_back_days: int) -> str:
    if look_back_days and look_back_days > 0:
        return (pd.Timestamp(start) - timedelta(days=int(look_back_days))).strftime("%Y-%m-%d")
    return str(start)


def validate_trading_calendar_coverage(
    dates: List[str] | pd.DatetimeIndex,
    *,
    start: str,
    end: str,
    load_start: str,
    look_back_days: int,
    provider_uri: str = "",
    tolerance_days: int = LOOKBACK_COVERAGE_TOLERANCE_DAYS,
) -> None:
    """Fail fast when provider calendar cannot cover the configured warmup window."""
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values()
    if date_index.empty:
        raise RuntimeError(
            f"Qlib provider {provider_uri or '<unknown>'} has no trading calendar rows "
            f"for requested range {load_start}~{end}."
        )

    first_date = date_index[0]
    required = pd.Timestamp(load_start)
    if first_date > required + pd.Timedelta(days=tolerance_days):
        gap_days = int((first_date - required).days)
        raise RuntimeError(
            f"Qlib provider {provider_uri or '<unknown>'} does not cover data.look_back_days. "
            f"backtest_start={start}, end={end}, look_back_days={look_back_days}, "
            f"required_load_start={load_start}, first_available_date={first_date.strftime('%Y-%m-%d')}, "
            f"calendar_gap_days={gap_days}. Rebuild or sync the provider from required_load_start "
            "or earlier before running this config."
        )


class BacktestContext:
    """回测上下文

    策略通过此对象访问行情数据、因子数据和预计算信号。
    """

    def __init__(self, exchange: AStockExchange):
        self.exchange = exchange
        self.market_data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.market_panel = None
        self.factor_runtime = None
        self.trade_dates: List[str] = []
        self._date_idx: int = 0
        self._signal_matrix: Dict[str, List[Signal]] = defaultdict(list)
        self._selection_candidates: Dict[str, Dict[str, Any]] = {}
        self._daily_selection_candidates: Dict[str, Dict[str, Any]] = {}
        self._selection_lock = Lock()

    # ============================================================
    # 数据加载
    # ============================================================

    def load_data(
        self,
        symbols: List[str],
        start: str,
        end: str,
        factor_names: Optional[List[str]] = None,
        look_back_days: int = 400,
        field_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """从 Qlib 加载行情数据和因子数据

        Args:
            symbols: 股票列表
            start: 回测起始日期
            end: 回测结束日期
            factor_names: 因子表达式列表（如 ['Ref($close, -5)/$close - 1']）
            look_back_days: 回看天数，用于计算指标（如 MA200）
        """
        load_start = compute_load_start(start, look_back_days)
        coverage_dates = self.exchange.get_trading_dates(load_start, end)
        validate_trading_calendar_coverage(
            coverage_dates,
            start=start,
            end=end,
            load_start=load_start,
            look_back_days=look_back_days,
            provider_uri=getattr(self.exchange, "provider_uri", ""),
        )

        extra_fields = _quote_fields_from_aliases(field_aliases)
        self.exchange.load_quote_data(symbols, load_start, end, extra_fields=extra_fields)

        if self.exchange.quote is not None and not self.exchange.quote.empty:
            from quantx.core.factor_runtime import FactorRuntime, MarketPanel

            self.market_panel = MarketPanel.from_frame(self.exchange.quote, aliases=field_aliases)
            self.factor_runtime = FactorRuntime(self.market_panel)

        if factor_names:
            import qlib
            from qlib.data import D

            self.factor_data = D.features(symbols, factor_names, start, end, freq="day")

        # 构建交易日历
        self.trade_dates = self._build_trade_dates(start, end)

    def _build_trade_dates(self, start: str, end: str) -> List[str]:
        """从 Qlib 日历获取交易日列表"""
        return self.exchange.get_trading_dates(start, end)

    # ============================================================
    # 数据查询
    # ============================================================

    def get_current_data(self, date: str) -> pd.DataFrame:
        """获取当日横截面数据"""
        if self.exchange.quote is None:
            return pd.DataFrame()
        try:
            return self.exchange.quote.loc[pd.Timestamp(date)]
        except (KeyError, IndexError):
            return pd.DataFrame()

    def get_factor_data(self, date: str) -> Optional[pd.DataFrame]:
        """获取当日因子数据"""
        if self.factor_data is None:
            return None
        try:
            return self.factor_data.loc[pd.Timestamp(date)]
        except (KeyError, IndexError):
            return None

    def get_history(
        self, symbol: str, field: str, start: str, end: str
    ) -> pd.Series:
        """获取单股票历史数据"""
        if self.exchange.quote is None:
            return pd.Series()
        return self.exchange.quote.loc[
            pd.IndexSlice[pd.Timestamp(start):pd.Timestamp(end), symbol], field
        ]

    # ============================================================
    # 信号矩阵
    # ============================================================

    def set_stock_signals(self, date: str, signals: List[Signal]) -> None:
        """存储预计算信号"""
        self._signal_matrix[date] = signals

    def get_stock_signals(self, date: str) -> List[Signal]:
        """获取当日预计算信号"""
        return self._signal_matrix.get(date, [])

    def record_selection_candidates(self, date: str, detail: Dict[str, Any]) -> None:
        """Store selector diagnostics for an execution date."""
        with self._selection_lock:
            self._selection_candidates[date] = dict(detail)

    def get_selection_candidates(self) -> List[Dict[str, Any]]:
        """Return selector diagnostics sorted by execution date."""
        with self._selection_lock:
            return [
                self._selection_candidates[date]
                for date in sorted(self._selection_candidates)
            ]

    def record_daily_selection_candidates(self, date: str, detail: Dict[str, Any]) -> None:
        """Store same-day selector diagnostics for visualization and next-session plans."""
        with self._selection_lock:
            self._daily_selection_candidates[date] = dict(detail)

    def get_daily_selection_candidates(self) -> List[Dict[str, Any]]:
        """Return same-day selector diagnostics sorted by signal date."""
        with self._selection_lock:
            return [
                self._daily_selection_candidates[date]
                for date in sorted(self._daily_selection_candidates)
            ]

    # ============================================================
    # 回测迭代
    # ============================================================

    def next(self) -> str:
        """推进到下一个交易日"""
        if self._date_idx >= len(self.trade_dates):
            raise StopIteration("No more trading days")
        date = self.trade_dates[self._date_idx]
        self._date_idx += 1
        return date

    def is_finished(self) -> bool:
        """回测是否结束"""
        return self._date_idx >= len(self.trade_dates)

    def reset(self) -> None:
        """重置迭代器"""
        self._date_idx = 0
