from __future__ import annotations

import pandas as pd

from quantx.core.engine.cost import TransactionCost
from quantx.core.engine.exchange import AStockExchange
from quantx.core.engine.types import OrderAction
from quantx.core.strategy.base import (
    AccountSnapshot,
    PolicyState,
    PositionSnapshot,
    Signal,
    StockSelection,
    WeightAllocation,
)
from quantx.core.strategy.config_strategy import EqualWeightRebalance, RuleExecution


def _state(*, holding_days: int, one_word_up: bool, limit_up: bool | None = None) -> PolicyState:
    symbol = "SH600000"
    is_limit_up = one_word_up if limit_up is None else limit_up
    price = 11.0 if is_limit_up else 10.5
    market = pd.DataFrame(
        {
            "$open": [price],
            "$high": [price if one_word_up else (11.2 if is_limit_up else 10.8)],
            "$low": [price if one_word_up else 10.2],
            "$close": [price],
            "$change": [0.10 if is_limit_up else 0.05],
            "$volume": [0.0 if one_word_up else 1000.0],
        },
        index=[symbol],
    )
    exchange = AStockExchange()
    quote = market.copy()
    quote.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-01-05"), symbol)],
        names=["datetime", "instrument"],
    )
    exchange.load_quote_from_raw(quote)
    exchange.set_execution_prices(buy="open", sell="close")
    context = type("Context", (), {"exchange": exchange, "factor_runtime": None})()
    return PolicyState(
        date="2021-01-05",
        market_data=market,
        account=AccountSnapshot(cash=90_000.0, total_value=101_000.0),
        positions={
            symbol: PositionSnapshot(
                symbol=symbol,
                quantity=1_000,
                avg_cost=10.0,
                market_value=price * 1_000,
                holding_days=holding_days,
            )
        },
        context=context,
    )


def _execution() -> RuleExecution:
    return RuleExecution(
        sell_rules=[
            {"name": "force_exit_10d", "when": "holding_days >= 10", "action": "sell_all"},
            {
                "name": "h2_close_unless_one_side_limit_up",
                "when": "holding_days >= 1 and not one_side_limit_up",
                "action": "sell_all",
            },
        ],
        cost=TransactionCost(
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            slippage=0.0,
        ),
        buy_deal_price="open",
        sell_deal_price="close",
    )


def test_h2_exit_waits_on_one_word_limit_up_then_forces_at_day_10():
    held_one_day = _execution().act(
        _state(holding_days=1, one_word_up=True),
        WeightAllocation(weights={}),
    )
    forced = _execution().act(
        _state(holding_days=10, one_word_up=True),
        WeightAllocation(weights={}),
    )

    assert held_one_day.orders == []
    assert len(forced.orders) == 1
    assert forced.orders[0].action == OrderAction.SELL
    assert forced.orders[0].price == 11.0


def test_h2_exit_sells_at_close_on_the_day_after_entry():
    orders = _execution().act(
        _state(holding_days=1, one_word_up=False),
        WeightAllocation(weights={}),
    )

    assert len(orders.orders) == 1
    assert orders.orders[0].action == OrderAction.SELL
    assert orders.orders[0].price == 10.5


def test_limit_up_state_can_hold_a_non_one_word_closing_limit():
    execution = RuleExecution(
        sell_rules=[
            {"name": "force_exit_10d", "when": "holding_days >= 10", "action": "sell_all"},
            {
                "name": "sell_unless_limit_up",
                "when": "holding_days >= 1 and not limit_up",
                "action": "sell_all",
            },
        ],
        cost=TransactionCost(
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            slippage=0.0,
        ),
        buy_deal_price="open",
        sell_deal_price="close",
    )

    orders = execution.act(
        _state(holding_days=1, one_word_up=False, limit_up=True),
        WeightAllocation(weights={}),
    )

    assert orders.orders == []


def test_top5_with_ten_slots_allocates_ten_percent_of_equity_per_new_name():
    rebalance = EqualWeightRebalance(
        max_positions=10,
        buy_only_new_positions=True,
        weight_scope="portfolio_target",
    )
    selection = StockSelection(
        signals=[
            Signal(symbol=f"SH60000{index}", score=float(10 - index))
            for index in range(5)
        ]
    )
    state = PolicyState(
        date="2021-01-05",
        market_data=pd.DataFrame(),
        account=AccountSnapshot(cash=1_000_000.0, total_value=1_000_000.0),
    )

    allocation = rebalance.act(state, selection)

    assert set(allocation.weights.values()) == {0.1}
