"""Decision pipeline and component registry tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import DecisionClock, MarketTime, SessionPhase
from quantx.core.decision.pipeline import DecisionPipeline, ExecutionPipeline
from quantx.core.decision.registry import ComponentCapabilities, ComponentRegistry
from quantx.core.decision.types import (
    AlphaSnapshot,
    AlphaValue,
    ExecutionContext,
    FeatureBatch,
    FinalTargetPortfolio,
    MarketObservation,
    PortfolioState,
    RawTargetPortfolio,
    TradabilityState,
)
from quantx.core.strategy.base import AccountSnapshot, OrderList


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day: int, hour: int, phase: SessionPhase) -> MarketTime:
    return MarketTime(
        session=f"2026-07-{day:02d}",
        phase=phase,
        timestamp=datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def observation_and_portfolio():
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    observation = MarketObservation(
        clock=DecisionClock(
            observation_end=signal_time,
            data_available_at=signal_time,
            signal_time=signal_time,
            execute_not_before=execution_time,
        ),
        universe=("SZ000001", "SH600000"),
        features=FeatureBatch(
            as_of=signal_time,
            instruments=("SZ000001", "SH600000"),
            values=((1.0,), (2.0,)),
            schema_id="feature-schema-1",
            available_at=signal_time,
        ),
    )
    portfolio = PortfolioState(
        as_of=signal_time,
        account=AccountSnapshot(cash=100_000.0, total_value=100_000.0),
        positions={},
    )
    return observation, portfolio


class _Alpha:
    def predict(self, market):
        return AlphaSnapshot(
            signal_time=market.clock.signal_time,
            earliest_execution_time=market.clock.execute_not_before,
            values=(AlphaValue("SZ000001", 1.0), AlphaValue("SH600000", 0.5)),
            source_id="alpha-1",
        )


class _Portfolio:
    def construct(self, market, portfolio, alpha):
        return RawTargetPortfolio(
            decision_time=alpha.signal_time,
            earliest_execution_time=alpha.earliest_execution_time,
            weights={"SZ000001": 0.5, "SH600000": 0.5},
            cash_weight=0.0,
            source_signal_id=alpha.source_id,
        )


class _Risk:
    def apply(self, market, portfolio, raw_target):
        return FinalTargetPortfolio.from_raw(raw_target, applied_risk_rules=("noop",))


class _Planner:
    def __init__(self):
        self.calls = []

    def create_orders(self, execution, pending):
        self.calls.append((execution, pending))
        return OrderList(orders=[])


class _Executor:
    def __init__(self):
        self.calls = []

    def execute_batch(self, orders, account, exchange):
        self.calls.append((orders, account, exchange))
        return ["executed"]


class _EarlyPortfolio(_Portfolio):
    def construct(self, market, portfolio, alpha):
        return RawTargetPortfolio(
            decision_time=alpha.signal_time,
            earliest_execution_time=market.clock.signal_time,
            weights={"SZ000001": 1.0},
            cash_weight=0.0,
            source_signal_id=alpha.source_id,
        )


def test_decision_pipeline_produces_pending_rebalance():
    market, portfolio = observation_and_portfolio()
    pipeline = DecisionPipeline(_Alpha(), _Portfolio(), _Risk())

    pending = pipeline.decide(
        market,
        portfolio,
        account_id="research",
        strategy_instance_id="model-a",
    )

    assert pending.created_at == market.clock.signal_time
    assert pending.execute_not_before == market.clock.execute_not_before
    assert pending.target.applied_risk_rules == ("noop",)
    assert pending.account_id == "research"

    same_pending = pipeline.decide(
        market,
        portfolio,
        account_id="research",
        strategy_instance_id="model-a",
    )
    assert same_pending.rebalance_id == pending.rebalance_id


def test_decision_pipeline_rejects_a_component_that_bypasses_execution_clock():
    market, portfolio = observation_and_portfolio()
    pipeline = DecisionPipeline(_Alpha(), _EarlyPortfolio(), _Risk())

    with pytest.raises(ValueError, match="after decision_time"):
        pipeline.decide(
            market,
            portfolio,
            account_id="research",
            strategy_instance_id="model-a",
        )


def test_execution_pipeline_rejects_early_execution_and_runs_when_due():
    market, portfolio = observation_and_portfolio()
    pending = DecisionPipeline(_Alpha(), _Portfolio(), _Risk()).decide(
        market,
        portfolio,
        account_id="research",
        strategy_instance_id="model-a",
    )
    planner = _Planner()
    executor = _Executor()
    pipeline = ExecutionPipeline(planner, executor)
    account = object()
    exchange = object()

    early_context = ExecutionContext(
        clock=market_time(10, 9, SessionPhase.OPEN),
        open_prices={"SZ000001": 10.0},
        tradability={"SZ000001": TradabilityState(can_buy=True, can_sell=True)},
        account=portfolio.account,
        positions={},
    )
    with pytest.raises(ValueError, match="not ready"):
        pipeline.execute(early_context, pending, account, exchange)
    assert planner.calls == []

    due_context = ExecutionContext(
        clock=market.clock.execute_not_before,
        open_prices={"SZ000001": 10.0},
        tradability={"SZ000001": TradabilityState(can_buy=True, can_sell=True)},
        account=portfolio.account,
        positions={},
    )
    assert pipeline.execute(due_context, pending, account, exchange) == ["executed"]
    assert len(planner.calls) == 1
    assert len(executor.calls) == 1


def test_component_registry_is_explicit_and_rejects_duplicate_registration():
    registry = ComponentRegistry()
    capabilities = ComponentCapabilities(
        needs_account_state=False,
        supports_batch_prediction=True,
        supports_online_update=False,
        output_type="AlphaSnapshot",
    )
    registry.register("alpha", "predictions", lambda config: ("alpha", config), capabilities)

    component = registry.build("alpha", "predictions", {"artifact": "fold-1"})
    assert component == ("alpha", {"artifact": "fold-1"})
    assert registry.capabilities("alpha", "predictions") == capabilities

    with pytest.raises(ValueError, match="already registered"):
        registry.register("alpha", "predictions", lambda config: None, capabilities)

    with pytest.raises(KeyError, match="Unknown component"):
        registry.build("alpha", "missing", {})
