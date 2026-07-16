"""RL allocation and shared simulation-kernel tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quantx.core.decision.clock import DecisionClock, MarketTime, SessionPhase
from quantx.core.decision.components import StandardRiskOverlay, TopKEqualWeightConstructor
from quantx.core.decision.types import (
    AlphaSnapshot,
    AlphaValue,
    FeatureBatch,
    MarketObservation,
    PortfolioState,
    TradabilityState,
)
from quantx.core.engine.cost import TransactionCost
from quantx.core.research.rl import QuantXAllocationEnv, RLAllocationPolicy, ReplayStep
from quantx.core.strategy.base import AccountSnapshot


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day, hour, phase):
    return MarketTime(
        f"2026-07-{day:02d}",
        phase,
        datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def observation():
    signal = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    return MarketObservation(
        DecisionClock(signal, signal, signal, market_time(13, 9, SessionPhase.OPEN)),
        ("A", "B"),
        FeatureBatch(signal, ("A", "B"), np.array([[1.0], [2.0]]), "schema", signal),
    )


def test_fixed_rl_action_matches_equal_weight_raw_target():
    market = observation()
    portfolio = PortfolioState(
        market.clock.signal_time,
        AccountSnapshot(10_000.0, 10_000.0),
        {},
    )
    alpha = AlphaSnapshot(
        market.clock.signal_time,
        market.clock.execute_not_before,
        (AlphaValue("A", 1.0), AlphaValue("B", 0.9)),
        "alpha",
    )

    equal = TopKEqualWeightConstructor(top_k=2).construct(market, portfolio, alpha)
    rl = RLAllocationPolicy().construct_action(market, portfolio, np.array([0.5, 0.5]))

    assert rl.weights == equal.weights
    assert rl.cash_weight == equal.cash_weight


def test_rl_environment_uses_standard_orders_executor_costs_and_risk():
    market = observation()
    step = ReplayStep(
        market=market,
        open_prices={"A": 10.0, "B": 10.0},
        close_prices={"A": 11.0, "B": 11.0},
        previous_closes={"A": 10.0, "B": 10.0},
        tradability={"A": TradabilityState(True, True), "B": TradabilityState(True, True)},
    )
    cost = TransactionCost(
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage=0.0,
    )
    env = QuantXAllocationEnv(
        [step],
        init_cash=10_000.0,
        cost=cost,
        risk=StandardRiskOverlay(max_position_weight=0.4, max_gross_exposure=1.0),
        lot_size=100,
    )

    initial, _ = env.reset(seed=7)
    next_observation, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]))

    assert initial.shape == next_observation.shape
    assert reward == pytest.approx(0.08)
    assert terminated is True
    assert truncated is False
    assert info["total_value"] == pytest.approx(10_800.0)
    assert info["applied_risk_rules"] == ("max_position_weight",)
    assert len(info["trades"]) == 2

    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError:
        return
    check_env(env, skip_render_check=True)
