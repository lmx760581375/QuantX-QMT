"""Explicit wrappers for strategy behavior that still uses PolicyState."""

from __future__ import annotations


LEGACY_COMPAT_MODE = "legacy_compat"


class LegacyCompositeStrategyAdapter:
    """Expose an existing composite strategy unchanged during migration."""

    execution_mode = LEGACY_COMPAT_MODE

    def __init__(self, strategy):
        self.strategy = strategy

    @property
    def precompute_stock_signals(self) -> bool:
        return bool(getattr(self.strategy, "precompute_stock_signals", False))

    def on_init(self, context):
        return self.strategy.on_init(context)

    def on_finish(self, context):
        return self.strategy.on_finish(context)

    def get_stock_signal(self, state):
        return self.strategy.get_stock_signal(state)

    def get_trade_signal(self, state, selection):
        return self.strategy.get_trade_signal(state, selection)


class LegacyRiskExecutionAdapter:
    """Mark stateful RuleExecution-like policies as migration-only behavior."""

    execution_mode = LEGACY_COMPAT_MODE

    def __init__(self, execution):
        self.execution = execution

    def prepare(self, context):
        prepare = getattr(self.execution, "prepare", None)
        if callable(prepare):
            return prepare(context)
        return None

    def preview_sell_symbols(self, state):
        preview = getattr(self.execution, "preview_sell_symbols", None)
        if not callable(preview):
            return []
        return preview(state)

    def act(self, state, allocation):
        return self.execution.act(state, allocation)
