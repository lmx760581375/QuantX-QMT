"""A-share exchange tradability regression tests."""

import pandas as pd

from quantx.core.engine.exchange import AStockExchange
from quantx.core.engine.types import OrderAction


def test_exchange_rejects_buy_on_one_side_limit_up_with_volume():
    exchange = AStockExchange()
    quote = pd.DataFrame(
        {
            "$open": [10.0, 10.5],
            "$high": [10.0, 10.5],
            "$low": [10.0, 10.5],
            "$close": [10.0, 10.5],
            "$volume": [1000.0, 1000.0],
            "$amount": [10_000.0, 10_500.0],
            "$change": [0.0, 0.05],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-07-10"), "SZ000001"), (pd.Timestamp("2026-07-13"), "SZ000001")],
            names=["datetime", "instrument"],
        ),
    )
    exchange.load_quote_from_raw(quote)

    assert exchange.is_one_side_limit_up("SZ000001", "2026-07-13") is True
    assert exchange.is_stock_tradable("SZ000001", "2026-07-13", OrderAction.BUY.value) is False
    assert exchange.is_stock_tradable("SZ000001", "2026-07-13", OrderAction.SELL.value) is True


def test_exchange_rejects_sell_on_one_side_limit_down_with_volume():
    exchange = AStockExchange()
    quote = pd.DataFrame(
        {
            "$open": [10.0, 9.5],
            "$high": [10.0, 9.5],
            "$low": [10.0, 9.5],
            "$close": [10.0, 9.5],
            "$volume": [1000.0, 1000.0],
            "$amount": [10_000.0, 9_500.0],
            "$change": [0.0, -0.05],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-07-10"), "SZ000001"), (pd.Timestamp("2026-07-13"), "SZ000001")],
            names=["datetime", "instrument"],
        ),
    )
    exchange.load_quote_from_raw(quote)

    assert exchange.is_one_side_limit_down("SZ000001", "2026-07-13") is True
    assert exchange.is_stock_tradable("SZ000001", "2026-07-13", OrderAction.SELL.value) is False
    assert exchange.is_stock_tradable("SZ000001", "2026-07-13", OrderAction.BUY.value) is True
