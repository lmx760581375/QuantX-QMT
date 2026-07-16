"""Legacy strategy adapters preserve the existing call contract."""

from quantx.core.decision.adapters.legacy_policy import (
    LEGACY_COMPAT_MODE,
    LegacyCompositeStrategyAdapter,
    LegacyRiskExecutionAdapter,
)
from quantx.core.strategy.base import OrderList, StockSelection, WeightAllocation


class _LegacyStrategy:
    precompute_stock_signals = True

    def __init__(self):
        self.calls = []

    def on_init(self, context):
        self.calls.append(("init", context))

    def on_finish(self, context):
        self.calls.append(("finish", context))

    def get_stock_signal(self, state):
        self.calls.append(("select", state))
        return StockSelection(signals=[])

    def get_trade_signal(self, state, selection):
        self.calls.append(("trade", state, selection))
        return OrderList(orders=[])


class _LegacyExecution:
    def __init__(self):
        self.calls = []

    def prepare(self, context):
        self.calls.append(("prepare", context))

    def preview_sell_symbols(self, state):
        self.calls.append(("preview", state))
        return ["SZ000001"]

    def act(self, state, allocation):
        self.calls.append(("act", state, allocation))
        return OrderList(orders=[])


def test_legacy_composite_adapter_delegates_without_changing_results():
    legacy = _LegacyStrategy()
    adapter = LegacyCompositeStrategyAdapter(legacy)
    state = object()
    context = object()

    adapter.on_init(context)
    selection = adapter.get_stock_signal(state)
    orders = adapter.get_trade_signal(state, selection)
    adapter.on_finish(context)

    assert adapter.execution_mode == LEGACY_COMPAT_MODE
    assert adapter.precompute_stock_signals is True
    assert isinstance(selection, StockSelection)
    assert isinstance(orders, OrderList)
    assert [call[0] for call in legacy.calls] == ["init", "select", "trade", "finish"]


def test_legacy_risk_execution_adapter_delegates_stateful_execution():
    legacy = _LegacyExecution()
    adapter = LegacyRiskExecutionAdapter(legacy)
    context = object()
    state = object()
    allocation = WeightAllocation(weights={"SZ000001": 1.0})

    adapter.prepare(context)
    assert adapter.preview_sell_symbols(state) == ["SZ000001"]
    assert adapter.act(state, allocation).orders == []
    assert adapter.execution_mode == LEGACY_COMPAT_MODE
    assert [call[0] for call in legacy.calls] == ["prepare", "preview", "act"]
