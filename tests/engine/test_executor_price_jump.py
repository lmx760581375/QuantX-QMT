"""Regression coverage for configurable buy-side price-jump protection."""

import pandas as pd
import pytest

from quantx.core.engine.board import BoardManager
from quantx.core.engine.exchange import AStockExchange
from quantx.core.engine.executor import Executor
from quantx.core.engine.types import Order, OrderAction


class _Exchange:
    def __init__(self, change: float):
        self.board = BoardManager()
        self.change = change

    def is_stock_suspended(self, symbol, date):
        return False

    def is_one_side_limit_up(self, symbol, date):
        return False

    def is_limit_up(self, symbol, date):
        return False

    def is_one_side_limit_down(self, symbol, date):
        return False

    def is_limit_down(self, symbol, date):
        return False

    def get_preclose(self, symbol, date):
        return 100.0

    def get_deal_price(self, symbol, date, direction=None):
        return 100.0 * (1.0 + self.change)


def _buy(symbol: str) -> Order:
    return Order(symbol=symbol, action=OrderAction.BUY, price=1.0, quantity=100, date="2021-01-04")


def test_default_price_jump_limit_preserves_existing_9p5pct_rejection():
    exchange = _Exchange(0.10)

    assert Executor()._validate(_buy("SZ300001"), account=None, exchange=exchange) == "price_jump"


def test_board_limit_allows_sub_limit_20pct_board_move():
    exchange = _Exchange(0.10)

    assert Executor(price_jump_limit="board_limit")._validate(
        _buy("SZ300001"), account=None, exchange=exchange
    ) is None


def test_exchange_uses_directional_execution_prices():
    exchange = AStockExchange()
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-01-04"), "SZ000001")],
        names=["datetime", "instrument"],
    )
    exchange.load_quote_from_raw(
        pd.DataFrame({"$open": [9.0], "$close": [10.0]}, index=index)
    )
    exchange.set_execution_prices(buy="close", sell="open")

    assert exchange.get_deal_price("SZ000001", "2021-01-04", OrderAction.BUY) == 10.0
    assert exchange.get_deal_price("SZ000001", "2021-01-04", OrderAction.SELL) == 9.0


def test_exchange_uses_historical_st_rate_for_limit_up(tmp_path):
    st_path = tmp_path / "historical_st.csv"
    st_path.write_text(
        "ts_code,trade_date,is_st\n"
        "600238.SH,2021-01-04,1\n"
        "600239.SH,2021-01-04,0\n",
        encoding="utf-8",
    )
    exchange = AStockExchange()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2021-01-04"), "SH600238"),
            (pd.Timestamp("2021-01-04"), "SH600239"),
        ],
        names=["datetime", "instrument"],
    )
    exchange.load_quote_from_raw(
        pd.DataFrame({"$change": [0.05, 0.05]}, index=index)
    )
    exchange.set_historical_st_path(st_path)

    assert exchange.is_limit_up("SH600238", "2021-01-04") is True
    assert exchange.is_limit_up("SH600239", "2021-01-04") is False


def test_exchange_parses_compact_historical_st_date(tmp_path):
    st_path = tmp_path / "historical_st.csv"
    st_path.write_text(
        "ts_code,trade_date,is_st\n600238.SH,20210104,1\n",
        encoding="utf-8",
    )
    exchange = AStockExchange()
    exchange.set_historical_st_path(st_path)

    assert exchange.is_historical_st("SH600238", "2021-01-04") is True


def test_buy_limit_check_uses_open_when_buy_deal_price_is_open():
    exchange = AStockExchange()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2021-01-03"), "SH600000"),
            (pd.Timestamp("2021-01-04"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    exchange.load_quote_from_raw(
        pd.DataFrame(
            {
                "$open": [10.0, 10.0],
                "$high": [10.0, 11.0],
                "$low": [10.0, 10.0],
                "$close": [10.0, 11.0],
                "$change": [0.0, 0.10],
                "$volume": [1000.0, 1000.0],
            },
            index=index,
        )
    )
    exchange.set_execution_prices(buy="open", sell="close")

    assert exchange.is_limit_up("SH600000", "2021-01-04") is True
    assert exchange.is_deal_price_limit_up(
        "SH600000", "2021-01-04", OrderAction.BUY
    ) is False
    assert Executor(price_jump_limit="board_limit")._validate(
        _buy("SH600000"), account=None, exchange=exchange
    ) is None


def test_buy_limit_check_rejects_open_at_limit():
    exchange = AStockExchange()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2021-01-03"), "SH600000"),
            (pd.Timestamp("2021-01-04"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    exchange.load_quote_from_raw(
        pd.DataFrame(
            {
                "$open": [10.0, 11.0],
                "$high": [10.0, 11.1],
                "$low": [10.0, 10.9],
                "$close": [10.0, 10.8],
                "$change": [0.0, 0.08],
                "$volume": [1000.0, 1000.0],
            },
            index=index,
        )
    )
    exchange.set_execution_prices(buy="open", sell="close")

    assert exchange.is_limit_up("SH600000", "2021-01-04") is False
    assert exchange.is_deal_price_limit_up(
        "SH600000", "2021-01-04", OrderAction.BUY
    ) is True
    assert (
        Executor(price_jump_limit="board_limit")._validate(
            _buy("SH600000"), account=None, exchange=exchange
        )
        == "limit_up"
    )


def test_chinext_limit_rate_is_date_aware():
    exchange = AStockExchange()

    assert exchange.get_limit_rate("SZ300001", "2020-08-21") == pytest.approx(0.10)
    assert exchange.get_limit_rate("SZ300001", "2020-08-24") == pytest.approx(0.20)
