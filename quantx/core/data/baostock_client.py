"""BaoStock API 封装

封装 BaoStock 的 HTTP API，提供可靠的数据获取接口。
包含登录/登出上下文管理器、自动日期回溯、字段重命名、重试逻辑。
"""

import logging
import io
import socket
import time
from contextlib import contextmanager, redirect_stdout
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# BaoStock 字段名 → 标准化字段名
FIELD_RENAME_MAP = {
    "date": "date",
    "code": "code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "preclose",
    "volume": "volume",
    "amount": "amount",
    "turn": "turnover",
    "tradestatus": "tradestatus",
    "pctChg": "pct_chg",
    "isST": "is_st",
}


class BaoStockClient:
    """BaoStock API 客户端

    封装 baostock 的登录/登出、数据查询、重试逻辑。
    使用上下文管理器确保登出。

    Usage:
        with BaoStockClient() as client:
            data = client.query_history_k_data("sh.600519", "2020-01-01", "2025-12-31")
    """

    def __init__(self, pause_seconds: float = 0.5, max_retries: int = 3, socket_timeout: float = 30.0):
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.socket_timeout = socket_timeout
        self._logged_in = False
        self._previous_socket_timeout = None

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *args):
        self.logout()

    def login(self) -> bool:
        """登录 BaoStock"""
        import baostock as bs

        self._previous_socket_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.socket_timeout)
        try:
            with redirect_stdout(io.StringIO()):
                result = bs.login()
        except Exception:
            self._restore_socket_timeout()
            raise
        if result.error_code != "0":
            self._restore_socket_timeout()
            raise ConnectionError(f"BaoStock login failed: {result.error_msg}")
        self._logged_in = True
        logger.info("BaoStock login successful")
        return True

    def logout(self) -> bool:
        """登出 BaoStock"""
        if self._logged_in:
            import baostock as bs

            with redirect_stdout(io.StringIO()):
                bs.logout()
            self._logged_in = False
            self._restore_socket_timeout()
            logger.info("BaoStock logout")
        return True

    def _restore_socket_timeout(self) -> None:
        socket.setdefaulttimeout(self._previous_socket_timeout)
        self._previous_socket_timeout = None

    # ============================================================
    # 股票列表
    # ============================================================

    def query_all_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取全量股票列表

        Args:
            date: 指定日期，None 为当前

        Returns:
            DataFrame with columns: code, code_name, ipoDate, outDate, type, status
        """
        import baostock as bs

        self._ensure_login()
        # 必须传日期，最新版 BaoStock 无 date 返回空
        if date is None:
            date = "2025-12-31"  # 已知可用的最近日期
        with redirect_stdout(io.StringIO()):
            result = bs.query_all_stock(day=date)

        if result.error_code != "0":
            raise RuntimeError(f"query_all_stock failed: {result.error_msg}")

        data = []
        while result.next():
            data.append(result.get_row_data())

        # 如果当天返回空，回退到 2025-12-31
        if not data and date != "2025-12-31":
            logger.warning(f"No stocks for {date}, retrying with 2025-12-31")
            with redirect_stdout(io.StringIO()):
                result = bs.query_all_stock(day="2025-12-31")
            while result.next():
                data.append(result.get_row_data())
        return pd.DataFrame(data, columns=result.fields)

    def get_stock_codes(self, date: Optional[str] = None) -> List[str]:
        """获取股票代码列表（Qlib 格式: SH600519）"""
        df = self.query_all_stocks(date)
        codes = []
        for _, row in df.iterrows():
            code = row["code"]
            # BaoStock 格式: sh.600519 → Qlib 格式: SH600519
            normalized = self._normalize_symbol(code)
            codes.append(normalized)
        return codes

    # ============================================================
    # 日线数据
    # ============================================================

    def query_history_k_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
        adjustflag: str = "2",  # 默认前复权。3=不复权（配合 query_adjust_factor 使用）
    ) -> pd.DataFrame:
        """获取日线 K 线数据

        Args:
            symbol: 股票代码，BaoStock 格式 sh.600519
            start_date: 起始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            fields: 需要获取的字段，默认全量。如 "date,code,open,high,low,close,volume"
            adjustflag: 复权标志 1=后复权 2=前复权 3=不复权
        """
        import baostock as bs

        self._ensure_login()
        time.sleep(self.pause_seconds)  # API 限流

        if fields is None:
            fields = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"

        with redirect_stdout(io.StringIO()):
            result = bs.query_history_k_data_plus(
                symbol,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )

        if result.error_code != "0":
            logger.warning(f"query_history_k_data failed for {symbol}: {result.error_msg}")
            return pd.DataFrame()

        data = []
        while result.next():
            data.append(result.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=result.fields)
        return self._process_k_data(df)

    def query_history_k_data_with_retry(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """带重试的 K 线数据查询"""
        for attempt in range(self.max_retries):
            try:
                df = self.query_history_k_data(symbol, start_date, end_date, **kwargs)
                if not df.empty:
                    return df
                if attempt < self.max_retries - 1:
                    time.sleep(self.pause_seconds * (attempt + 1))
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.pause_seconds * (attempt + 1))
        return pd.DataFrame()

    # ============================================================
    # 除权除息数据
    # ============================================================

    def query_adjust_factor(
        self, symbol: str, start_date: str = "2015-01-01", end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取复权因子

        Returns:
            DataFrame with columns: code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor
            - foreAdjustFactor: 前复权因子（累积，最新日期为 1.0）
            - backAdjustFactor: 后复权因子
        """
        import baostock as bs

        self._ensure_login()
        time.sleep(self.pause_seconds)

        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        with redirect_stdout(io.StringIO()):
            result = bs.query_adjust_factor(code=symbol, start_date=start_date, end_date=end_date)

        if result.error_code != "0":
            logger.warning(f"query_adjust_factor failed for {symbol}: {result.error_msg}")
            return pd.DataFrame()

        data = []
        while result.next():
            data.append(result.get_row_data())

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=result.fields)
        df["foreAdjustFactor"] = pd.to_numeric(df["foreAdjustFactor"], errors="coerce")
        df["backAdjustFactor"] = pd.to_numeric(df["backAdjustFactor"], errors="coerce")
        return df

    def query_dividend_data(self, symbol: str, year: str) -> pd.DataFrame:
        """获取除权除息数据"""
        import baostock as bs

        self._ensure_login()
        time.sleep(self.pause_seconds)

        fields = "code,dividPreDate,stockBonus,stockGift,allotNum,allotPrice,interest"
        result = bs.query_dividend_data(symbol, year=start_date[:4], yearType="report")

        if result.error_code != "0":
            return pd.DataFrame()

        data = []
        while result.next():
            data.append(result.get_row_data())

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data, columns=result.fields)

    # ============================================================
    # 交易日历
    # ============================================================

    def query_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        import baostock as bs

        self._ensure_login()
        result = bs.query_trade_dates(start_date=start_date, end_date=end_date)

        if result.error_code != "0":
            raise RuntimeError(f"query_trade_dates failed: {result.error_msg}")

        dates = []
        while result.next():
            row = result.get_row_data()
            if row[1] == "1":  # is_trading_day
                dates.append(row[0])
        return dates

    # ============================================================
    # 内部方法
    # ============================================================

    def _ensure_login(self):
        """确保已登录"""
        if not self._logged_in:
            self.login()

    def _process_k_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理 K 线数据：类型转换、字段重命名"""
        if df.empty:
            return df

        # 字段重命名
        df = df.rename(columns=FIELD_RENAME_MAP)

        # 数值类型转换
        numeric_cols = [
            "open", "high", "low", "close", "preclose",
            "volume", "amount", "turnover", "pct_chg",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 日期格式化
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df

    @staticmethod
    def _normalize_symbol(code: str) -> str:
        """股票代码格式统一：BaoStock sh.600519 → Qlib SH600519"""
        code = code.replace(".", "").upper()
        if code.startswith("SH") or code.startswith("SZ") or code.startswith("BJ"):
            return code
        if code.startswith("6"):
            return f"SH{code}"
        return f"SZ{code}"

    @staticmethod
    def to_baostock_format(symbol: str) -> str:
        """Qlib SH600519 → BaoStock sh.600519"""
        if symbol.startswith("SH"):
            return f"sh.{symbol[2:]}"
        elif symbol.startswith("SZ"):
            return f"sz.{symbol[2:]}"
        elif symbol.startswith("BJ"):
            return f"bj.{symbol[2:]}"
        return symbol
