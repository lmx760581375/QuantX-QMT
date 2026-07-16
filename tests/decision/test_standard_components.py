"""Standard portfolio, risk, and order-planning component tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import DecisionClock, MarketTime, SessionPhase
from quantx.core.decision.components import (
    MarketRegimeRiskOverlay,
    NoOpRiskOverlay,
    ScoreWeightedPortfolioConstructor,
    StandardRiskOverlay,
    TopKBufferedEqualWeightConstructor,
    TopKEqualWeightConstructor,
)
from quantx.core.decision.order_planner import StandardOrderPlanner
from quantx.core.decision.types import (
    AlphaSnapshot,
    AlphaValue,
    ExecutionContext,
    FeatureBatch,
    FinalTargetPortfolio,
    MarketObservation,
    PendingRebalance,
    PortfolioState,
    TradabilityState,
)
from quantx.core.engine.types import OrderAction
from quantx.core.strategy.base import AccountSnapshot, PositionSnapshot


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day: int, hour: int, phase: SessionPhase) -> MarketTime:
    return MarketTime(
        session=f"2026-07-{day:02d}",
        phase=phase,
        timestamp=datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def market_and_portfolio(drawdown: float = 0.0):
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    market = MarketObservation(
        clock=DecisionClock(signal_time, signal_time, signal_time, market_time(13, 9, SessionPhase.OPEN)),
        universe=("SZ000001", "SH600000", "SZ000002"),
        features=FeatureBatch(signal_time, (), (), "schema-1", signal_time),
    )
    portfolio = PortfolioState(
        as_of=signal_time,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0, drawdown=drawdown),
        positions={},
    )
    alpha = AlphaSnapshot(
        signal_time=signal_time,
        earliest_execution_time=market.clock.execute_not_before,
        values=(
            AlphaValue("SH600000", 0.3),
            AlphaValue("SZ000001", 0.9),
            AlphaValue("SZ000002", -0.2),
        ),
        source_id="ridge-v1:2026-07-10",
    )
    return market, portfolio, alpha


def test_same_alpha_supports_equal_and_score_weighted_portfolios():
    market, portfolio, alpha = market_and_portfolio()

    equal = TopKEqualWeightConstructor(top_k=2, gross_exposure=0.8).construct(market, portfolio, alpha)
    weighted = ScoreWeightedPortfolioConstructor(top_k=2, gross_exposure=0.8).construct(market, portfolio, alpha)

    assert dict(equal.weights) == {"SZ000001": 0.4, "SH600000": 0.4}
    assert equal.cash_weight == pytest.approx(0.2)
    assert weighted.weights["SZ000001"] > weighted.weights["SH600000"]
    assert abs(sum(weighted.weights.values()) + weighted.cash_weight - 1.0) < 1e-12


def test_standard_risk_caps_names_and_scales_exposure_on_drawdown():
    market, portfolio, alpha = market_and_portfolio(drawdown=-0.25)
    raw = TopKEqualWeightConstructor(top_k=2).construct(market, portfolio, alpha)

    final = StandardRiskOverlay(
        max_position_weight=0.3,
        max_account_drawdown=0.2,
        defensive_gross_exposure=0.4,
    ).apply(market, portfolio, raw)

    assert max(final.weights.values()) <= 0.3
    assert sum(final.weights.values()) <= 0.4 + 1e-12
    assert "account_drawdown" in final.applied_risk_rules
    assert "max_position_weight" in final.applied_risk_rules
    assert NoOpRiskOverlay().apply(market, portfolio, raw).weights == raw.weights


def test_buffered_topk_retains_existing_holding_inside_rank_buffer():
    market, _, alpha = market_and_portfolio()
    existing = PositionSnapshot("SZ000002", 100, 10.0, 1_000.0)
    portfolio = PortfolioState(
        market.clock.signal_time,
        AccountSnapshot(cash=9_000.0, total_value=10_000.0),
        {"SZ000002": existing},
    )

    target = TopKBufferedEqualWeightConstructor(top_k=2, rank_buffer=3).construct(market, portfolio, alpha)

    assert target.weights == {"SZ000002": 0.5, "SZ000001": 0.5}
    assert target.metadata["retained_count"] == 1


def test_market_regime_risk_scales_target_when_breadth_is_weak():
    market, portfolio, alpha = market_and_portfolio()
    market = MarketObservation(
        market.clock,
        market.universe,
        FeatureBatch(
            market.clock.signal_time,
            (),
            {"market_ret20": -0.05, "market_ret60": -0.02, "breadth20": 0.3},
            "schema-1",
            market.clock.signal_time,
        ),
    )
    raw = TopKEqualWeightConstructor(top_k=2, gross_exposure=0.9).construct(market, portfolio, alpha)

    final = MarketRegimeRiskOverlay(
        max_position_weight=0.5,
        max_gross_exposure=0.9,
        risk_off_gross_exposure=0.3,
    ).apply(market, portfolio, raw)

    assert sum(final.weights.values()) == pytest.approx(0.3)
    assert "market_regime" in final.applied_risk_rules


def test_market_regime_can_move_risk_off_capital_to_defensive_instrument():
    market, portfolio, alpha = market_and_portfolio()
    market = MarketObservation(
        market.clock,
        (*market.universe, "SH511880"),
        FeatureBatch(
            market.clock.signal_time,
            (),
            {"market_ret20": -0.05, "market_ret60": -0.10, "breadth20": 0.2},
            "schema-1",
            market.clock.signal_time,
        ),
    )
    raw = TopKEqualWeightConstructor(top_k=2, gross_exposure=0.9).construct(market, portfolio, alpha)

    final = MarketRegimeRiskOverlay(
        max_position_weight=0.5,
        max_gross_exposure=0.9,
        risk_off_gross_exposure=0.0,
        defensive_instrument="SH511880",
        risk_off_total_exposure=0.9,
    ).apply(market, portfolio, raw)

    assert final.weights == {"SH511880": pytest.approx(0.9)}
    assert final.cash_weight == pytest.approx(0.1)


def test_standard_order_planner_converts_full_target_to_sell_then_buy_orders():
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    target = FinalTargetPortfolio(
        decision_time=signal_time,
        earliest_execution_time=execution_time,
        weights={"SZ000001": 1.0},
        cash_weight=0.0,
        source_signal_id="ridge-v1:2026-07-10",
    )
    pending = PendingRebalance.from_target(
        target,
        rebalance_id="rebalance-1",
        account_id="research",
        strategy_instance_id="ridge-top1",
    )
    old_position = PositionSnapshot("SH600000", 100, 10.0, 1_000.0)
    execution = ExecutionContext(
        clock=execution_time,
        open_prices={"SH600000": 10.0, "SZ000001": 20.0},
        tradability={
            "SH600000": TradabilityState(True, True),
            "SZ000001": TradabilityState(True, True),
        },
        account=AccountSnapshot(cash=9_000.0, total_value=10_000.0),
        positions={"SH600000": old_position},
    )

    orders = StandardOrderPlanner(lot_size=100).create_orders(execution, pending).orders

    assert [(order.symbol, order.action, order.quantity) for order in orders] == [
        ("SH600000", OrderAction.SELL, 100),
        ("SZ000001", OrderAction.BUY, 500),
    ]
    assert orders[1].depends_on_sells == ["SH600000"]
    assert orders[1].context["strict_execution"] is True


def test_standard_order_planner_does_not_emit_blocked_side():
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    target = FinalTargetPortfolio(
        signal_time,
        execution_time,
        {"SZ000001": 1.0},
        0.0,
        "signal-1",
    )
    pending = PendingRebalance.from_target(
        target,
        rebalance_id="rebalance-1",
        account_id="research",
        strategy_instance_id="ridge-top1",
    )
    execution = ExecutionContext(
        clock=execution_time,
        open_prices={"SZ000001": 20.0},
        tradability={"SZ000001": TradabilityState(False, True, reason="limit_up")},
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
    )

    assert StandardOrderPlanner().create_orders(execution, pending).orders == []
