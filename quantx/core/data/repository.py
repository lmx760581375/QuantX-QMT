"""本地 CSV 数据仓库

管理本地 CSV 文件存储，按股票拆分存储日线数据。
支持 upsert 去重合并、批量加载、覆盖范围查询。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# CSV 列定义
STOCK_CSV_COLUMNS = [
    "date", "code", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turnover", "pct_chg", "is_st",
]


class LocalDataRepository:
    """本地 CSV 数据仓库

    存储布局:
        data/raw/baostock/
        ├── stocks/
        │   ├── SH600519.csv
        │   └── SZ000001.csv
        └── universe/
            └── all_stocks.csv
    """

    def __init__(self, data_root: str = "data/raw/baostock"):
        self.data_root = Path(data_root)
        self.stocks_dir = self.data_root / "stocks"
        self.universe_dir = self.data_root / "universe"
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.universe_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 股票数据读写
    # ============================================================

    def save_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        """保存单只股票数据（upsert 去重合并）

        Args:
            symbol: 股票代码，如 SH600519
            df: 日线数据 DataFrame
        """
        if df.empty:
            return

        filepath = self._symbol_path(symbol)

        if filepath.exists():
            existing = pd.read_csv(filepath, dtype={"code": str})
            # 合并去重：按 date 去重，新数据覆盖旧数据
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date")
            combined.to_csv(filepath, index=False)
        else:
            df_sorted = df.sort_values("date")
            df_sorted.to_csv(filepath, index=False)

        logger.debug(f"Saved {symbol}: {len(df)} rows -> {filepath}")

    def replace_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        """Replace a single stock CSV with a fully refreshed dataframe."""
        if df.empty:
            return
        filepath = self._symbol_path(symbol)
        df_sorted = df.sort_values("date")
        df_sorted.to_csv(filepath, index=False)
        logger.debug(f"Replaced {symbol}: {len(df)} rows -> {filepath}")

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """加载单只股票数据"""
        filepath = self._symbol_path(symbol)
        if not filepath.exists():
            return pd.DataFrame()
        return pd.read_csv(filepath, dtype={"code": str}, parse_dates=["date"])

    def load_all(
        self, start: str, end: str, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """批量加载股票数据，返回 MultiIndex DataFrame

        Args:
            start: 起始日期
            end: 结束日期
            symbols: 股票列表，None 表示加载所有

        Returns:
            (date, symbol) MultiIndex DataFrame
        """
        if symbols is None:
            symbols = self._list_all_symbols()

        frames = []
        for sym in symbols:
            df = self.load_symbol(sym)
            if df.empty:
                continue
            # 添加 symbol 列确保一致
            df["symbol"] = sym
            # 按日期范围过滤
            mask = (df["date"] >= start) & (df["date"] <= end)
            frames.append(df[mask])

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # 确保 date 是 datetime
        result["date"] = pd.to_datetime(result["date"])
        return result.set_index(["date", "symbol"]).sort_index()

    def get_last_date(self, symbol: str) -> Optional[str]:
        """获取某股票最新数据日期"""
        df = self.load_symbol(symbol)
        if df.empty:
            return None
        return pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")

    def get_first_date(self, symbol: str) -> Optional[str]:
        """获取某股票最早数据日期"""
        df = self.load_symbol(symbol)
        if df.empty:
            return None
        return pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d")

    # ============================================================
    # 股票列表
    # ============================================================

    def save_universe(self, stocks_df: pd.DataFrame) -> None:
        """保存股票列表"""
        filepath = self.universe_dir / "all_stocks.csv"
        stocks_df.to_csv(filepath, index=False)

    def load_universe(self) -> pd.DataFrame:
        """加载股票列表"""
        filepath = self.universe_dir / "all_stocks.csv"
        if not filepath.exists():
            return pd.DataFrame()
        return pd.read_csv(filepath, dtype={"code": str})

    def get_stock_codes(self) -> List[str]:
        """获取所有已存储的股票代码"""
        return self._list_all_symbols()

    # ============================================================
    # 覆盖范围
    # ============================================================

    def get_coverage_report(self, start: str, end: str) -> Dict:
        """生成数据覆盖报告

        Returns:
            dict with keys: total_stocks, stocks_with_data, missing_stocks, date_range
        """
        symbols = self._list_all_symbols()
        stocks_with_data = 0
        missing_stocks = []
        earliest_date = None
        latest_date = None

        for sym in symbols:
            last = self.get_last_date(sym)
            first = self.get_first_date(sym)
            if last is not None and first is not None:
                stocks_with_data += 1
                if earliest_date is None or first < earliest_date:
                    earliest_date = first
                if latest_date is None or last > latest_date:
                    latest_date = last
            else:
                missing_stocks.append(sym)

        return {
            "total_stocks": len(symbols),
            "stocks_with_data": stocks_with_data,
            "missing_stocks": missing_stocks,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "requested_range": f"{start} ~ {end}",
        }

    # ============================================================
    # 内部方法
    # ============================================================

    def _symbol_path(self, symbol: str) -> Path:
        """获取股票 CSV 文件路径"""
        return self.stocks_dir / f"{symbol}.csv"

    def _list_all_symbols(self) -> List[str]:
        """列出所有已存储的股票代码"""
        if not self.stocks_dir.exists():
            return []
        return sorted([
            f.stem for f in self.stocks_dir.glob("*.csv")
            if f.stem  # 跳过空文件名
        ])
