"""RL allocation adapter backed by QuantX's real execution semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None

from quantx.core.decision.order_planner import StandardOrderPlanner
from quantx.core.decision.types import (
    ExecutionContext,
    FinalTargetPortfolio,
    MarketObservation,
    PendingRebalance,
    PortfolioState,
    RawTargetPortfolio,
    TradabilityState,
)
from quantx.core.engine.account import Account
from quantx.core.engine.executor import Executor
from quantx.core.engine.simulation import SimulationKernel
from quantx.core.strategy.base import AccountSnapshot, PositionSnapshot


_GymBase = gym.Env if gym is not None else object


@dataclass(frozen=True)
class ReplayStep:
    market: MarketObservation
    open_prices: dict[str, float]
    close_prices: dict[str, float]
    previous_closes: dict[str, float]
    tradability: dict[str, TradabilityState]


class RLAllocationPolicy:
    def construct_action(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        action: np.ndarray,
    ) -> RawTargetPortfolio:
        del portfolio
        values = np.asarray(action, dtype=float).reshape(-1)
        if len(values) != len(market.universe):
            raise ValueError("RL action size must match the decision universe")
        if not np.isfinite(values).all():
            raise ValueError("RL action must be finite")
        values = np.clip(values, 0.0, None)
        gross = float(values.sum())
        if gross > 1.0:
            values = values / gross
            gross = 1.0
        weights = {instrument: float(weight) for instrument, weight in zip(market.universe, values) if weight > 0}
        return RawTargetPortfolio(
            decision_time=market.clock.signal_time,
            earliest_execution_time=market.clock.execute_not_before,
            weights=weights,
            cash_weight=1.0 - gross,
            source_signal_id=f"rl:{market.clock.signal_time.session}",
            metadata={"constructor": "rl_allocation"},
        )


class QuantXAllocationEnv(_GymBase):
    metadata = {"render_modes": []}

    def __init__(
        self,
        steps: Sequence[ReplayStep],
        *,
        init_cash: float,
        cost,
        risk,
        lot_size: int = 100,
    ):
        if not steps:
            raise ValueError("RL replay requires at least one step")
        self.steps = tuple(steps)
        self.init_cash = float(init_cash)
        self.cost = cost
        self.risk = risk
        self.policy = RLAllocationPolicy()
        self.planner = StandardOrderPlanner(lot_size=lot_size)
        self.universe = self.steps[0].market.universe
        if any(step.market.universe != self.universe for step in self.steps):
            raise ValueError("RL replay currently requires a stable universe")
        self.account = None
        self.exchange = None
        self.kernel = None
        self.index = 0
        if gym is not None:
            feature_size = int(np.asarray(self.steps[0].market.features.values).size)
            observation_size = feature_size + len(self.universe) + 1
            self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(self.universe),), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        self.account = Account(init_cash=self.init_cash, cost=self.cost)
        self.exchange = _ReplayExchange()
        self.kernel = SimulationKernel(
            planner=self.planner,
            executor=Executor(cost=self.cost, validate_trading_rules=True),
        )
        self.index = 0
        return self._observation(self.steps[0]), {}

    def step(self, action):
        if self.account is None or self.kernel is None or self.exchange is None:
            raise RuntimeError("reset() must be called before step()")
        replay = self.steps[self.index]
        portfolio = self._portfolio_state(replay)
        raw = self.policy.construct_action(replay.market, portfolio, np.asarray(action))
        final = self.risk.apply(replay.market, portfolio, raw)
        pending = PendingRebalance.from_target(
            final,
            rebalance_id=_rebalance_id(final),
            account_id="rl-training",
            strategy_instance_id="rl-policy",
        )
        execution = ExecutionContext(
            clock=replay.market.clock.execute_not_before,
            open_prices=replay.open_prices,
            tradability=replay.tradability,
            account=portfolio.account,
            positions=portfolio.positions,
        )
        self.exchange.set_step(replay)
        result = self.kernel.execute(
            execution,
            pending,
            account=self.account,
            exchange=self.exchange,
            close_prices=replay.close_prices,
        )
        self.index += 1
        terminated = self.index >= len(self.steps)
        next_replay = self.steps[-1] if terminated else self.steps[self.index]
        info = {
            "total_value": result.total_value,
            "trades": result.trades,
            "orders": result.orders,
            "applied_risk_rules": final.applied_risk_rules,
        }
        return self._observation(next_replay), result.reward, terminated, False, info

    def _portfolio_state(self, replay: ReplayStep) -> PortfolioState:
        total_value = float(self.account.get_total_value())
        positions = {
            symbol: PositionSnapshot(
                symbol=symbol,
                quantity=position.quantity,
                avg_cost=position.avg_cost,
                market_value=position.market_value,
                weight=position.market_value / total_value if total_value > 0 else 0.0,
                holding_days=position.holding_days,
                highest_price=position.highest_price,
                lowest_price=position.lowest_price,
                initial_quantity=position.initial_quantity,
                context=dict(position.context),
            )
            for symbol, position in self.account.positions.items()
        }
        return PortfolioState(
            as_of=replay.market.clock.signal_time,
            account=AccountSnapshot(
                cash=self.account.cash,
                total_value=total_value,
                daily_return=self.account.latest_daily_return,
                cumulative_return=self.account.cumulative_return,
                drawdown=self.account.latest_drawdown,
            ),
            positions=positions,
        )

    def _observation(self, replay: ReplayStep) -> np.ndarray:
        market_values = np.asarray(replay.market.features.values, dtype=float).reshape(-1)
        if self.account is None:
            cash_weight = 1.0
            position_weights = np.zeros(len(self.universe))
        else:
            total = float(self.account.get_total_value())
            cash_weight = self.account.cash / total if total > 0 else 0.0
            position_weights = np.asarray(
                [
                    self.account.positions.get(symbol).market_value / total
                    if symbol in self.account.positions and total > 0
                    else 0.0
                    for symbol in self.universe
                ]
            )
        return np.concatenate([market_values, position_weights, [cash_weight]]).astype(np.float32)


class _ReplayExchange:
    def __init__(self):
        self.step = None

    def set_step(self, step: ReplayStep) -> None:
        self.step = step

    def get_deal_price(self, symbol, date, direction=None):
        del date, direction
        return self.step.open_prices.get(symbol)

    def get_preclose(self, symbol, date):
        del date
        return self.step.previous_closes.get(symbol)

    def get_close(self, symbol, date):
        del date
        return self.step.close_prices.get(symbol)


def _rebalance_id(target: FinalTargetPortfolio) -> str:
    identity = "|".join(
        [
            target.source_signal_id,
            target.decision_time.timestamp.isoformat(),
            *(f"{symbol}={weight:.16g}" for symbol, weight in sorted(target.weights.items())),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
