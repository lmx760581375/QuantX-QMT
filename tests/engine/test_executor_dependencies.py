"""Dependent order execution tests."""

from types import SimpleNamespace

from quantx.core.engine.executor import Executor
from quantx.core.engine.types import Order, OrderAction, Trade


def test_buy_is_rejected_when_required_sell_does_not_fill(monkeypatch):
    executor = Executor(validate_trading_rules=False)
    account = SimpleNamespace(trades=[])
    calls = []

    def fake_execute(order, _account, _exchange):
        calls.append((order.symbol, order.action))
        if order.action == OrderAction.SELL and order.symbol == "OLD":
            return None
        return Trade(order.symbol, order.action, order.price, order.quantity, order.date)

    monkeypatch.setattr(executor, "execute", fake_execute)
    orders = [
        Order("OLD", OrderAction.SELL, 1.0, 100, "2026-07-10"),
        Order(
            "NEW",
            OrderAction.BUY,
            1.0,
            100,
            "2026-07-10",
            depends_on_sells=["OLD"],
        ),
    ]

    trades = executor.execute_batch(orders, account, SimpleNamespace())

    assert calls == [("OLD", OrderAction.SELL)]
    assert trades == []
    assert account.trades[-1].symbol == "NEW"
    assert account.trades[-1].reject_reason == "sell_dependency_failed"
