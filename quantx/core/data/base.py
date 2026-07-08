"""数据源抽象基类

定义所有数据源必须实现的接口。后续所有模块依赖此接口，不依赖具体实现。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class DataSource(ABC):
    """数据源抽象基类

    所有数据源（BaoStock、Tushare、AKShare等）必须实现此接口。
    策略层和回测引擎通过此接口访问数据，不直接依赖具体数据源。
    """

    @abstractmethod
    def get_daily_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        adjust: str = "forward",
    ) -> pd.DataFrame:
        """获取日线行情数据

        Args:
            symbols: 股票代码列表，格式 SH600519 / SZ000001
            start_date: 起始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD
            fields: 需要获取的字段列表，默认全部
            adjust: 复权方式 forward/none/backward

        Returns:
            (date, symbol) MultiIndex DataFrame，列名: open/high/low/close/volume/amount/vwap
        """
        ...

    @abstractmethod
    def get_dividend_factors(self, symbols: List[str]) -> pd.DataFrame:
        """获取复权因子

        Args:
            symbols: 股票代码列表

        Returns:
            DataFrame，包含除权除息信息
        """
        ...

    @abstractmethod
    def get_stock_list(self, date: Optional[str] = None) -> List[str]:
        """获取指定日期的可交易股票列表

        Args:
            date: 日期，None 表示获取当前最新股票列表

        Returns:
            股票代码列表
        """
        ...

    @abstractmethod
    def get_benchmark(
        self, benchmark: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取基准指数数据

        Args:
            benchmark: 指数代码，如 SH000300（沪深300）
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            (date, symbol) MultiIndex DataFrame，包含 close 列
        """
        ...

    @abstractmethod
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表

        Args:
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            日期字符串列表，格式 YYYY-MM-DD
        """
        ...

    @abstractmethod
    def is_ready(self, start_date: str, end_date: str) -> bool:
        """检查数据是否就绪（Qlib 数据目录是否覆盖所需日期范围）"""
        ...


# 标准字段定义
OHLCV_FIELDS = ["open", "high", "low", "close", "volume"]
EXTENDED_FIELDS = OHLCV_FIELDS + ["amount", "vwap", "change", "factor", "preclose"]
QLIB_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor", "$change"]