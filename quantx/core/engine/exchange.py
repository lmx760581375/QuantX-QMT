"""A 股行情数据提供者

通过 Qlib D.features() 加载行情数据，提供查询接口。
不负责订单校验（校验在 executor.py 中）。
"""

import logging
from typing import List, Optional

import pandas as pd

from quantx.core.qlib_import import import_qlib
from quantx.core.data.qlib_reader import QlibBinReader

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
        self._reader: Optional[QlibBinReader] = None
        self.deal_price: str = "open"

    def _init_qlib(self) -> None:
        if self._initialized:
            return
        try:
            qlib = self._import_qlib()
            qlib.init(provider_uri=self.provider_uri, region="cn")
        except Exception as exc:
            logger.warning("Falling back to built-in qlib bin reader: %s", exc)
            self._reader = QlibBinReader(self.provider_uri)
        self._initialized = True

    @staticmethod
    def _import_qlib():
        return import_qlib()

    def load_quote_data(self, symbols: List[str], start: str, end: str) -> None:
        self._init_qlib()
        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$change", "$factor", "$vwap"]
        if self._reader is not None:
            quote = self._reader.features(symbols, fields, start, end)
        else:
            try:
                from qlib.data import D

                quote = D.features(symbols, fields, start, end, freq="day")
            except Exception as exc:
                logger.warning("Falling back to built-in qlib bin reader after D.features failed: %s", exc)
                quote = self._load_with_builtin_reader(symbols, fields, start, end)
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
        if self._reader is not None:
            cal = self._reader.calendar(start, end)
        else:
            try:
                from qlib.data import D

                cal = D.calendar(start_time=start, end_time=end)
            except Exception as exc:
                logger.warning("Falling back to built-in qlib bin calendar after D.calendar failed: %s", exc)
                self._reader = QlibBinReader(self.provider_uri)
                cal = self._reader.calendar(start, end)
        return pd.to_datetime(cal).strftime("%Y-%m-%d").tolist()

    def _load_with_builtin_reader(self, symbols: List[str], fields: List[str], start: str, end: str) -> pd.DataFrame:
        self._reader = QlibBinReader(self.provider_uri)
        return self._reader.features(symbols, fields, start, end)

    def get_deal_price(
        self, symbol: str, date: str, direction: OrderAction = OrderAction.BUY
    ) -> Optional[float]:
        """获取成交价格。默认 T 日开盘价，可配置为 close/vwap。"""
        if self.quote is None:
            return None
        try:
            field = f"${self.deal_price.lstrip('$')}"
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
        return change >= self.board.get_limit_up_rate(symbol) - 1e-6

    def is_limit_down(self, symbol: str, date: str) -> bool:
        change = self.get_change(symbol, date)
        if change is None:
            return False
        return change <= -self.board.get_limit_down_rate(symbol) + 1e-6

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
