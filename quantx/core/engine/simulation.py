"""Shared strict-event simulation step used by backtests and RL replay."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantx.core.decision.types import ExecutionContext, PendingRebalance


@dataclass(frozen=True)
class SimulationStepResult:
    orders: tuple
    trades: tuple
    previous_value: float
    total_value: float
    reward: float


class SimulationKernel:
    def __init__(self, *, planner, executor=None):
        self.planner = planner
        self.executor = executor

    def plan(self, execution: ExecutionContext, pending: PendingRebalance):
        return self.planner.create_orders(execution, pending)

    def execute(
        self,
        execution: ExecutionContext,
        pending: PendingRebalance,
        *,
        account,
        exchange,
        close_prices: dict[str, float],
    ) -> SimulationStepResult:
        if self.executor is None:
            raise ValueError("SimulationKernel requires an Executor for execute()")
        previous_value = float(account.get_total_value())
        order_list = self.plan(execution, pending)
        trades = self.executor.execute_batch(order_list.orders, account, exchange)
        market_data = pd.DataFrame({"$close": close_prices})
        account.update_daily_balance(execution.clock.session, exchange, market_data=market_data)
        total_value = float(account.get_total_value())
        reward = total_value / previous_value - 1.0 if previous_value > 0 else 0.0
        return SimulationStepResult(
            orders=tuple(order_list.orders),
            trades=tuple(trades),
            previous_value=previous_value,
            total_value=total_value,
            reward=reward,
        )
