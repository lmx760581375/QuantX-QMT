"""Unified-engine Wufu ETF strategy tests."""

from types import SimpleNamespace

import pandas as pd

from quantx.core.engine.cost import TransactionCost
from quantx.core.strategy.base import PolicyState
from quantx.core.strategy.factory import build_strategy, explain_strategy
from quantx.strategies.market_regime_rotation import MarketRegimeRotationSelector


def test_strategy_factory_builds_wufu_composite_strategy():
    config = {
        "name": "wufu_qmt",
        "strategy": {"type": "market_regime_rotation", "params": {"choppy_confirm_days": 3}},
        "execution": {"deal_price": "open"},
    }

    strategy = build_strategy(config, TransactionCost())
    explain = explain_strategy(config)

    assert isinstance(strategy.selector, MarketRegimeRotationSelector)
    assert strategy.precompute_stock_signals is False
    assert explain["strategy_type"] == "market_regime_rotation"
    assert explain["selector"]["signal_lag"] == 1


def test_wufu_selector_uses_previous_session_close(monkeypatch):
    selector = MarketRegimeRotationSelector({"choppy_confirm_days": 1})
    selector.bars = {"SH510300": pd.DataFrame()}
    selector.close = pd.DataFrame(
        {"SH510300": [1.0, 1.1]},
        index=pd.to_datetime(["2026-07-09", "2026-07-10"]),
    )
    selector.regime_close = selector.close.rename(columns={"SH510300": "沪深300"})
    captured = {}

    def fake_select_target(date, *args, **kwargs):
        captured["date"] = date
        return "SH510300", [], {"filtered_count": 1}

    monkeypatch.setattr("quantx.strategies.market_regime_rotation._select_target", fake_select_target)
    monkeypatch.setattr("quantx.strategies.market_regime_rotation.resolve_regime", lambda *args: "正常期")
    context = SimpleNamespace(
        trade_dates=["2026-07-09", "2026-07-10"],
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(date="2026-07-10", market_data=pd.DataFrame(), context=context)

    selection = selector.act(state)

    assert captured["date"] == pd.Timestamp("2026-07-09")
    assert [signal.symbol for signal in selection.signals] == ["SH510300"]
