"""Backtest engine execution-loop regression tests."""

import pandas as pd

from quantx.core.engine.engine import BacktestConfig, BacktestEngine
from quantx.core.strategy.base import OrderList, StockSelection


class _FakeExchange:
    deal_price = "close"

    def load_quote_data(self, *args, **kwargs):
        pass


class _FakeContext:
    def __init__(self, exchange):
        self.exchange = exchange
        self.trade_dates = []
        self._signals = {}
        self._signal_matrix = self._signals
        self._idx = 0

    def load_data(self, *args, **kwargs):
        self.trade_dates = ["2021-01-04", "2021-01-05", "2021-01-06"]

    def set_stock_signals(self, date, signals):
        self._signals[date] = signals

    def get_stock_signals(self, date):
        return self._signals.get(date, [])

    def get_current_data(self, date):
        return pd.DataFrame()

    def get_factor_data(self, date):
        return None

    def next(self):
        date = self.trade_dates[self._idx]
        self._idx += 1
        return date

    def is_finished(self):
        return self._idx >= len(self.trade_dates)


class _FakeAccount:
    def __init__(self, *args, **kwargs):
        self.cash = 1_000_000.0
        self.latest_daily_return = 0.0
        self.cumulative_return = 0.0
        self.latest_drawdown = 0.0
        self.positions = {}
        self.trades = []
        self.daily_snapshots = []

    def get_total_value(self):
        return self.cash

    def update_daily_balance(self, date, exchange):
        self.daily_snapshots.append(type("Snapshot", (), {
            "date": date,
            "total_value": self.cash,
            "positions": {},
        })())
        self.cash -= 10.0


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def execute_batch(self, orders, account, exchange):
        return []


class _EmptySignalStrategy:
    def __init__(self):
        self.trade_dates = []

    def on_init(self, context):
        pass

    def on_finish(self, context):
        pass

    def get_stock_signal(self, state):
        return StockSelection(signals=[])

    def get_trade_signal(self, state, selection):
        assert isinstance(selection, StockSelection)
        assert selection.signals == []
        self.trade_dates.append(state.date)
        return OrderList(orders=[])


class _AccountAwareSignalStrategy:
    precompute_stock_signals = False

    def __init__(self):
        self.cash_values = []

    def on_init(self, context):
        pass

    def on_finish(self, context):
        pass

    def get_stock_signal(self, state):
        self.cash_values.append(state.account.cash)
        return StockSelection(signals=[])

    def get_trade_signal(self, state, selection):
        return OrderList(orders=[])


class _PrecomputedSignalStrategy(_AccountAwareSignalStrategy):
    precompute_stock_signals = True


def test_engine_runs_trade_logic_on_empty_selection_days(monkeypatch):
    import quantx.core.engine.engine as engine_mod

    monkeypatch.setattr(engine_mod, "BoardManager", lambda: object())
    monkeypatch.setattr(engine_mod, "AStockExchange", lambda board, provider_uri: _FakeExchange())
    monkeypatch.setattr(engine_mod, "BacktestContext", _FakeContext)
    monkeypatch.setattr(engine_mod, "Account", _FakeAccount)
    monkeypatch.setattr(engine_mod, "Executor", _FakeExecutor)

    strategy = _EmptySignalStrategy()
    config = BacktestConfig(start_date="2021-01-04", end_date="2021-01-06", max_workers=1)

    BacktestEngine(config).run(strategy, ["SZ000001"])

    assert strategy.trade_dates == ["2021-01-04", "2021-01-05", "2021-01-06"]


def test_engine_defaults_to_daily_signal_calculation_for_stateful_strategies(monkeypatch):
    import quantx.core.engine.engine as engine_mod

    monkeypatch.setattr(engine_mod, "BoardManager", lambda: object())
    monkeypatch.setattr(engine_mod, "AStockExchange", lambda board, provider_uri: _FakeExchange())
    monkeypatch.setattr(engine_mod, "BacktestContext", _FakeContext)
    monkeypatch.setattr(engine_mod, "Account", _FakeAccount)
    monkeypatch.setattr(engine_mod, "Executor", _FakeExecutor)

    strategy = _AccountAwareSignalStrategy()
    config = BacktestConfig(start_date="2021-01-04", end_date="2021-01-06", max_workers=1)

    BacktestEngine(config).run(strategy, ["SZ000001"])

    assert strategy.cash_values == [1_000_000.0, 999_990.0, 999_980.0]


def test_engine_precomputes_only_when_strategy_opts_in(monkeypatch):
    import quantx.core.engine.engine as engine_mod

    monkeypatch.setattr(engine_mod, "BoardManager", lambda: object())
    monkeypatch.setattr(engine_mod, "AStockExchange", lambda board, provider_uri: _FakeExchange())
    monkeypatch.setattr(engine_mod, "BacktestContext", _FakeContext)
    monkeypatch.setattr(engine_mod, "Account", _FakeAccount)
    monkeypatch.setattr(engine_mod, "Executor", _FakeExecutor)

    strategy = _PrecomputedSignalStrategy()
    config = BacktestConfig(start_date="2021-01-04", end_date="2021-01-06", max_workers=1)

    BacktestEngine(config).run(strategy, ["SZ000001"])

    assert strategy.cash_values == [1_000_000.0, 1_000_000.0, 1_000_000.0]
