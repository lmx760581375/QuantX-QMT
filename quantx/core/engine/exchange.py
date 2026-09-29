"""A 股行情数据提供者

通过 Qlib D.features() 加载行情数据，提供查询接口。
不负责订单校验（校验在 executor.py 中）。
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .board import BoardManager
from .types import OrderAction

logger = logging.getLogger(__name__)


class AStockExchange:
    """A 股行情数据提供者"""

    def __init__(
        self,
        board: Optional[BoardManager] = None,
        provider_uri: str = "data/qlib_data",
    ):
        self.board = board or BoardManager()
        self.provider_uri = provider_uri
        self.quote: Optional[pd.DataFrame] = None
        self._initialized = False
        self.deal_price: str = "open"
        self.buy_deal_price: Optional[str] = None
        self.sell_deal_price: Optional[str] = None
        self._historical_st: set[tuple[str, str]] = set()

    def set_execution_prices(self, buy: Optional[str] = None, sell: Optional[str] = None) -> None:
        """分别设置买入和卖出成交价字段，并保留旧 deal_price 作为回退。"""
        self.buy_deal_price = buy
        self.sell_deal_price = sell

    def set_historical_st_path(self, path: str | os.PathLike[str]) -> None:
        """加载逐日历史 ST 状态，用于计算当日真实涨跌停比例。"""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Historical ST file does not exist: {source}")
        data = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
        required = {"ts_code", "trade_date"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Historical ST file is missing columns: {sorted(missing)}")
        if "is_st" in data.columns:
            is_st = pd.to_numeric(data["is_st"], errors="coerce").fillna(0).astype(bool)
            data = data.loc[is_st].copy()
        raw_dates = data["trade_date"].astype(str).str.strip()
        compact = raw_dates.str.fullmatch(r"\d{8}")
        dates = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
        if compact.any():
            dates.loc[compact] = pd.to_datetime(
                raw_dates.loc[compact], format="%Y%m%d", errors="coerce"
            )
        if (~compact).any():
            dates.loc[~compact] = pd.to_datetime(raw_dates.loc[~compact], errors="coerce")
        dates = dates.dt.strftime("%Y-%m-%d")
        codes = data["ts_code"].astype(str).str.upper().map(self._normalize_ts_code)
        valid = dates.notna() & codes.notna()
        self._historical_st = set(zip(dates[valid], codes[valid]))

    @staticmethod
    def _normalize_ts_code(value: str) -> Optional[str]:
        text = str(value).strip().upper()
        if "." in text:
            code, exchange = text.split(".", 1)
            if exchange in {"SH", "SZ", "BJ"} and code.isdigit():
                return f"{exchange}{code}"
        if text.startswith(("SH", "SZ", "BJ")) and text[2:].isdigit():
            return text
        return None

    def is_historical_st(self, symbol: str, date: str) -> bool:
        return (str(date)[:10], str(symbol).upper()) in self._historical_st

    def _init_qlib(self) -> None:
        if self._initialized:
            return
        qlib = self._import_qlib()
        qlib.init(provider_uri=self.provider_uri, region="cn")
        self._initialized = True

    @staticmethod
    def _import_qlib():
        """Prefer the installed qlib package over this repo's uncompiled qlib source tree."""
        repo_root = Path(__file__).resolve().parents[3]
        local_qlib = repo_root / "qlib"

        def is_local_qlib_module(module) -> bool:
            module_file = getattr(module, "__file__", None)
            if not module_file:
                return False
            try:
                return local_qlib.resolve() in Path(module_file).resolve().parents
            except OSError:
                return False

        def is_blocked_path(path: str) -> bool:
            resolved = Path(path or os.getcwd()).resolve()
            return resolved == repo_root.resolve() or resolved == local_qlib.resolve()

        existing = sys.modules.get("qlib")
        if existing is not None and is_local_qlib_module(existing):
            for name in list(sys.modules):
                if name == "qlib" or name.startswith("qlib."):
                    del sys.modules[name]

        original_path = list(sys.path)
        sys.path = [p for p in original_path if not is_blocked_path(p)]
        try:
            qlib = __import__("qlib")
            if is_local_qlib_module(qlib):
                raise ImportError("local qlib source tree was imported")
            return qlib
        except ImportError:
            sys.path = original_path
            return __import__("qlib")
        finally:
            sys.path = original_path

    def load_quote_data(self, symbols: List[str], start: str, end: str) -> None:
        self._init_qlib()
        from qlib.data import D
        fields = ["$open", "$high", "$low", "$close", "$volume", "$change", "$factor", "$vwap"]
        quote = D.features(symbols, fields, start, end, freq="day")
        self.quote = self._normalize_quote_index(quote)

    @staticmethod
    def _normalize_quote_index(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize Qlib output to QuantX's internal (datetime, instrument) index."""
        if data is None or data.empty:
            return data
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            return data

        names = list(data.index.names)
        if names == ["datetime", "instrument"]:
            result = data.sort_index()
        elif "datetime" in names and "instrument" in names:
            result = data.reorder_levels(["datetime", "instrument"]).sort_index()
        else:
            # Qlib defaults to (instrument, datetime). Keep this as a defensive fallback.
            result = data.swaplevel(0, 1).sort_index()
        result.index = result.index.set_names(["datetime", "instrument"])
        return result

    def load_quote_from_raw(self, data: pd.DataFrame) -> None:
        """直接设置行情数据（用于测试/Mock）"""
        self.quote = self._normalize_quote_index(data)

    def get_trading_dates(self, start: str, end: str) -> List[str]:
        """获取交易日列表"""
        self._init_qlib()
        from qlib.data import D
        cal = D.calendar(start_time=start, end_time=end)
        return pd.to_datetime(cal).strftime("%Y-%m-%d").tolist()

    def get_deal_price(
        self, symbol: str, date: str, direction: OrderAction = OrderAction.BUY
    ) -> Optional[float]:
        """获取成交价格。默认 T 日开盘价，可配置为 close/vwap。"""
        if self.quote is None:
            return None
        try:
            selected_price = self.buy_deal_price
            if direction == OrderAction.SELL:
                selected_price = self.sell_deal_price
            field = f"${(selected_price or self.deal_price).lstrip('$')}"
            price = self.quote.loc[(pd.Timestamp(date), symbol), field]
            if pd.isna(price) or price <= 0:
                return None
            return float(price)
        except (KeyError, IndexError):
            return None

    def get_close(self, symbol: str, date: str) -> Optional[float]:
        """获取收盘价"""
        if self.quote is None:
            return None
        try:
            price = self.quote.loc[(pd.Timestamp(date), symbol), "$close"]
            if pd.isna(price):
                return None
            return float(price)
        except (KeyError, IndexError):
            return None

    def get_preclose(self, symbol: str, date: str) -> Optional[float]:
        """获取前收盘价"""
        if self.quote is None:
            return None
        try:
            sym_data = self.quote.xs(symbol, level="instrument")
            dates = sym_data.index
            pos = dates.get_loc(pd.Timestamp(date))
            if isinstance(pos, int) and pos > 0:
                return float(sym_data.iloc[pos - 1]["$close"])
        except (KeyError, IndexError):
            pass
        return None

    def get_change(self, symbol: str, date: str) -> Optional[float]:
        """获取涨跌幅"""
        if self.quote is None:
            return None
        try:
            val = self.quote.loc[(pd.Timestamp(date), symbol), "$change"]
            if pd.isna(val):
                return None
            return float(val)
        except (KeyError, IndexError):
            return None

    def is_stock_suspended(self, symbol: str, date: str) -> bool:
        """判断是否停牌"""
        if self.quote is None:
            return True
        try:
            row = self.quote.loc[(pd.Timestamp(date), symbol)]
            if pd.isna(row["$close"]):
                return True
            vol = row.get("$volume", 0)
            return vol is None or pd.isna(vol) or vol <= 0
        except (KeyError, IndexError):
            return True

    def is_limit_up(self, symbol: str, date: str) -> bool:
        change = self.get_change(symbol, date)
        if change is None:
            return False
        return change >= self.get_limit_rate(symbol, date) - 1e-6

    def is_limit_down(self, symbol: str, date: str) -> bool:
        change = self.get_change(symbol, date)
        if change is None:
            return False
        return change <= -self.get_limit_rate(symbol, date) + 1e-6

    def get_limit_rate(self, symbol: str, date: str) -> float:
        if self.is_historical_st(symbol, date):
            return 0.05
        if str(symbol).upper().startswith("SZ30") and str(date)[:10] < "2020-08-24":
            return 0.10
        return float(self.board.get_limit_up_rate(symbol))

    def is_deal_price_limit_up(
        self,
        symbol: str,
        date: str,
        direction: OrderAction = OrderAction.BUY,
    ) -> bool:
        price = self.get_deal_price(symbol, date, direction)
        preclose = self.get_preclose(symbol, date)
        if price is None or preclose is None or preclose <= 0:
            return False
        return price / preclose - 1 >= self.get_limit_rate(symbol, date) - 1e-6

    def is_one_side_limit_up(self, symbol: str, date: str) -> bool:
        if not self.is_limit_up(symbol, date):
            return False
        try:
            row = self.quote.loc[(pd.Timestamp(date), symbol)]
            if abs(row["$high"] - row["$low"]) > 1e-6:
                return False
            if abs(row["$open"] - row["$close"]) > 1e-6:
                return False
            vol = row.get("$volume", 0)
            if vol is not None and not pd.isna(vol) and vol > 0:
                return False
            return True
        except (KeyError, IndexError):
            return False

    def is_one_side_limit_down(self, symbol: str, date: str) -> bool:
        if not self.is_limit_down(symbol, date):
            return False
        try:
            row = self.quote.loc[(pd.Timestamp(date), symbol)]
            if abs(row["$high"] - row["$low"]) > 1e-6:
                return False
            if abs(row["$open"] - row["$close"]) > 1e-6:
                return False
            vol = row.get("$volume", 0)
            if vol is not None and not pd.isna(vol) and vol > 0:
                return False
            return True
        except (KeyError, IndexError):
            return False

    def is_stock_tradable(
        self, symbol: str, date: str, direction: Optional[int] = None
    ) -> bool:
        if self.is_stock_suspended(symbol, date):
            return False
        if direction == OrderAction.BUY.value:
            if self.is_one_side_limit_up(symbol, date):
                return False
            return not self.is_limit_up(symbol, date)
        elif direction == OrderAction.SELL.value:
            if self.is_one_side_limit_down(symbol, date):
                return False
            return not self.is_limit_down(symbol, date)
        return True
