"""Account accounting regression tests."""

import pytest

from quantx.core.engine.account import Account
from quantx.core.engine.cost import TransactionCost
from quantx.core.engine.types import OrderAction


class _CloseExchange:
    def __init__(self, closes):
        self.closes = closes

    def get_close(self, symbol, date):
        return self.closes.get((symbol, date))


def test_trade_cost_components_are_reported_separately():
    cost = TransactionCost(
        commission_rate=0.001,
        min_commission=1.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.0001,
        stamp_tax_on_buy=True,
    )
    account = Account(init_cash=100_000, cost=cost)

    buy = account.buy("SH600000", 10.0, 1000, "2021-01-04")
    sell = account.sell("SH600000", 11.0, 1000, "2021-01-05", reason="take_profit")

    assert buy.commission == pytest.approx(10.0)
    assert buy.stamp_tax == pytest.approx(5.0)
    assert buy.transfer_fee == pytest.approx(1.0)
    assert buy.total_cost == pytest.approx(16.0)
    assert sell.commission == pytest.approx(11.0)
    assert sell.stamp_tax == pytest.approx(5.5)
    assert sell.transfer_fee == pytest.approx(1.1)
    assert sell.total_cost == pytest.approx(17.6)
    assert sell.reason == "take_profit"


def test_transaction_cost_supports_asymmetric_slippage():
    cost = TransactionCost(slippage=0.01, buy_slippage=0.003, sell_slippage=0.0)

    assert cost.apply_slippage(10.0, OrderAction.BUY) == pytest.approx(10.03)
    assert cost.apply_slippage(10.0, OrderAction.SELL) == pytest.approx(10.0)


def test_daily_return_uses_previous_day_total_value():
    account = Account(init_cash=100_000, cost=TransactionCost(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0, slippage=0.0))
    account.buy("SZ000001", 10.0, 1000, "2021-01-04")

    account.update_daily_balance("2021-01-04", _CloseExchange({("SZ000001", "2021-01-04"): 10.0}))
    account.update_daily_balance("2021-01-05", _CloseExchange({("SZ000001", "2021-01-05"): 11.0}))

    assert account.daily_snapshots[0].daily_return == pytest.approx(0.0)
    assert account.daily_snapshots[1].daily_return == pytest.approx(0.01)


def test_partial_sell_updates_remaining_market_value():
    account = Account(
        init_cash=100_000,
        cost=TransactionCost(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0, slippage=0.0),
    )
    account.buy("SZ000001", 10.0, 1000, "2021-01-04")
    account.update_daily_balance("2021-01-04", _CloseExchange({("SZ000001", "2021-01-04"): 12.0}))

    account.sell("SZ000001", 12.0, 500, "2021-01-05")

    pos = account.positions["SZ000001"]
    assert pos.quantity == 500
    assert pos.market_value == pytest.approx(6000.0)
    assert account.get_total_value() == pytest.approx(102000.0)


def test_account_tracks_drawdown_from_equity_peak():
    account = Account(init_cash=100_000, cost=TransactionCost(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0, slippage=0.0))
    account.buy("SZ000001", 10.0, 1000, "2021-01-04")

    account.update_daily_balance("2021-01-04", _CloseExchange({("SZ000001", "2021-01-04"): 12.0}))
    account.update_daily_balance("2021-01-05", _CloseExchange({("SZ000001", "2021-01-05"): 11.0}))

    assert account.latest_drawdown == pytest.approx(101_000 / 102_000 - 1)


def test_account_freezes_early_hold_return_context():
    account = Account(
        init_cash=100_000,
        cost=TransactionCost(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0, slippage=0.0),
    )
    account.buy("SZ000001", 10.0, 1000, "2021-01-04", context={"entry_pre_5_return": -0.01})

    closes = {
        ("SZ000001", "2021-01-04"): 10.0,
        ("SZ000001", "2021-01-05"): 10.3,
        ("SZ000001", "2021-01-06"): 10.5,
        ("SZ000001", "2021-01-07"): 10.1,
        ("SZ000001", "2021-01-08"): 10.8,
        ("SZ000001", "2021-01-11"): 11.0,
        ("SZ000001", "2021-01-12"): 11.2,
        ("SZ000001", "2021-01-13"): 11.4,
        ("SZ000001", "2021-01-14"): 11.6,
        ("SZ000001", "2021-01-15"): 11.8,
    }
    for _, date in sorted(closes):
        account.update_daily_balance(date, _CloseExchange(closes))

    context = account.positions["SZ000001"].context
    assert context["entry_pre_5_return"] == pytest.approx(-0.01)
    assert context["hold_first_3d_return"] == pytest.approx(0.05)
    assert context["hold_first_5d_return"] == pytest.approx(0.08)
    assert context["hold_first_10d_return"] == pytest.approx(0.18)
