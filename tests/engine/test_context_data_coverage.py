"""Backtest context data coverage validation tests."""

import pytest

from quantx.core.engine.context import BacktestContext


class _CalendarExchange:
    provider_uri = "fake_provider"

    def __init__(self, dates):
        self.dates = dates
        self.load_calls = []
        self.quote = None

    def get_trading_dates(self, start, end):
        return [date for date in self.dates if start <= date <= end]

    def load_quote_data(self, symbols, start, end):
        self.load_calls.append((symbols, start, end))


def test_context_rejects_provider_missing_lookback_window():
    exchange = _CalendarExchange(["2020-01-10", "2020-01-13", "2020-01-14"])
    context = BacktestContext(exchange)

    with pytest.raises(RuntimeError, match="required_load_start=2019-12-11"):
        context.load_data(["SH600000"], "2020-01-10", "2020-01-14", look_back_days=30)

    assert exchange.load_calls == []


def test_context_allows_calendar_gap_for_non_trading_days():
    exchange = _CalendarExchange(["2020-01-06", "2020-01-07", "2020-01-08"])
    context = BacktestContext(exchange)

    context.load_data(["SH600000"], "2020-01-06", "2020-01-08", look_back_days=2)

    assert exchange.load_calls == [(["SH600000"], "2020-01-04", "2020-01-08")]
