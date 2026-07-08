"""A 股交易日历

从 Qlib 数据目录的 calendars/day.txt 读取交易日列表，
提供交易日查询、日期偏移等工具方法。
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


class TradingCalendar:
    """A 股交易日历

    从 Qlib 数据目录读取 calendars/day.txt，提供交易日查询接口。
    也支持从 list 直接构建（用于测试）。
    """

    def __init__(self, provider_uri: Optional[str] = None, dates: Optional[List[str]] = None):
        """
        Args:
            provider_uri: Qlib 数据目录路径，从 calendars/day.txt 读取
            dates: 直接传入交易日列表（用于测试）
        """
        if dates is not None:
            self._dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
        elif provider_uri is not None:
            calendar_path = Path(provider_uri) / "calendars" / "day.txt"
            if calendar_path.exists():
                self._dates = pd.DatetimeIndex(
                    pd.read_csv(calendar_path, header=None, names=["date"])["date"]
                )
            else:
                raise FileNotFoundError(
                    f"Calendar file not found: {calendar_path}. "
                    "Please run data conversion first."
                )
        else:
            raise ValueError("Either provider_uri or dates must be provided")

    def is_trading_day(self, date: str) -> bool:
        """判断是否为交易日"""
        return pd.Timestamp(date) in self._dates

    def get_trading_days(self, start: str, end: str) -> List[str]:
        """获取日期范围内的所有交易日"""
        mask = (self._dates >= pd.Timestamp(start)) & (self._dates <= pd.Timestamp(end))
        return self._dates[mask].strftime("%Y-%m-%d").tolist()

    def get_next_trading_day(self, date: str, n: int = 1) -> str:
        """获取 N 个交易日后的日期"""
        idx = self._dates.get_indexer([pd.Timestamp(date)], method="bfill")[0]
        target_idx = idx + n
        if target_idx >= len(self._dates):
            raise IndexError(f"No trading day {n} days after {date}")
        return self._dates[target_idx].strftime("%Y-%m-%d")

    def get_prev_trading_day(self, date: str, n: int = 1) -> str:
        """获取 N 个交易日前的日期"""
        idx = self._dates.get_indexer([pd.Timestamp(date)], method="ffill")[0]
        target_idx = idx - n
        if target_idx < 0:
            raise IndexError(f"No trading day {n} days before {date}")
        return self._dates[target_idx].strftime("%Y-%m-%d")

    def count_trading_days(self, start: str, end: str) -> int:
        """计算两个日期之间的交易日数量"""
        return len(self.get_trading_days(start, end))

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        """返回所有交易日"""
        return self._dates

    def __len__(self) -> int:
        return len(self._dates)

    def __contains__(self, date: str) -> bool:
        return self.is_trading_day(date)