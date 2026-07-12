"""Regression tests for avoiding full-panel lookups during phase 2."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quantx.core.engine.account import Account
from quantx.core.engine.cost import TransactionCost
from quantx.core.strategy.base import AccountSnapshot, PolicyState, WeightAllocation
from quantx.core.strategy.config_strategy import RuleExecution


class _UnexpectedExchangeLookup:
    def get_close(self, symbol, date):
        raise AssertionError("full quote table should not be queried when daily data is available")


def test_account_uses_daily_market_data_for_position_valuation():
    account = Account(
        init_cash=100_000,
        cost=TransactionCost(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0),
    )
    account.buy("SZ000001", 10.0, 1000, "2021-01-04")
    market_data = pd.DataFrame({"$close": [11.0]}, index=pd.Index(["SZ000001"], name="instrument"))

    account.update_daily_balance(
        "2021-01-05",
        _UnexpectedExchangeLookup(),
        market_data=market_data,
    )

    assert account.positions["SZ000001"].market_value == pytest.approx(11_000.0)


def test_rule_execution_gets_price_from_daily_market_data():
    execution = RuleExecution(
        sell_rules=[],
        cost=TransactionCost(),
        deal_price="close",
    )
    market_data = pd.DataFrame({"$close": [11.25]}, index=pd.Index(["SZ000001"], name="instrument"))
    state = PolicyState(
        date="2021-01-05",
        market_data=market_data,
        context=SimpleNamespace(exchange=SimpleNamespace(quote=None)),
    )

    assert execution._get_price(state, "SZ000001") == pytest.approx(11.25)


def test_rule_execution_reuses_previewed_sell_decisions():
    class CountingExecution(RuleExecution):
        def __init__(self):
            super().__init__(sell_rules=[], cost=TransactionCost(), deal_price="close")
            self.decision_calls = 0

        def _iter_sell_decisions(self, state):
            self.decision_calls += 1
            return iter([])

    execution = CountingExecution()
    state = PolicyState(
        date="2021-01-05",
        market_data=pd.DataFrame(),
        account=AccountSnapshot(cash=100_000.0, total_value=100_000.0),
    )

    execution.preview_sell_symbols(state)
    execution.act(state, WeightAllocation(weights={}))

    assert execution.decision_calls == 1
