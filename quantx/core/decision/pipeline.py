"""Composition of decision and strict execution components."""

from __future__ import annotations

import hashlib

from .alpha import AlphaModel
from .execution import OrderPlanner
from .portfolio import PortfolioConstructor
from .risk import RiskOverlay
from .types import ExecutionContext, FinalTargetPortfolio, MarketObservation, PendingRebalance, PortfolioState


class DecisionPipeline:
    def __init__(self, alpha: AlphaModel, portfolio: PortfolioConstructor, risk: RiskOverlay):
        self.alpha = alpha
        self.portfolio = portfolio
        self.risk = risk

    def decide(
        self,
        market: MarketObservation,
        portfolio_state: PortfolioState,
        *,
        account_id: str,
        strategy_instance_id: str,
        rebalance_id: str | None = None,
    ) -> PendingRebalance:
        alpha = self.alpha.predict(market)
        if alpha.signal_time != market.clock.signal_time:
            raise ValueError("AlphaSnapshot signal_time does not match the decision clock")
        if alpha.earliest_execution_time < market.clock.execute_not_before:
            raise ValueError("AlphaSnapshot attempts execution before the decision clock allows")
        raw = self.portfolio.construct(market, portfolio_state, alpha)
        if raw.decision_time != alpha.signal_time:
            raise ValueError("RawTargetPortfolio decision_time does not match AlphaSnapshot")
        if raw.earliest_execution_time != alpha.earliest_execution_time:
            raise ValueError("RawTargetPortfolio execution time does not match AlphaSnapshot")
        if raw.source_signal_id != alpha.source_id:
            raise ValueError("RawTargetPortfolio source does not match AlphaSnapshot")
        raw.require_universe(market.universe)
        final = self.risk.apply(market, portfolio_state, raw)
        if final.decision_time != raw.decision_time:
            raise ValueError("FinalTargetPortfolio decision_time does not match raw target")
        if final.earliest_execution_time != raw.earliest_execution_time:
            raise ValueError("FinalTargetPortfolio execution time does not match raw target")
        if final.source_signal_id != raw.source_signal_id:
            raise ValueError("FinalTargetPortfolio source does not match raw target")
        final.require_universe(market.universe)
        return PendingRebalance.from_target(
            final,
            rebalance_id=rebalance_id
            or self._rebalance_id(final, account_id=account_id, strategy_instance_id=strategy_instance_id),
            account_id=account_id,
            strategy_instance_id=strategy_instance_id,
        )

    @staticmethod
    def _rebalance_id(
        final: FinalTargetPortfolio,
        *,
        account_id: str,
        strategy_instance_id: str,
    ) -> str:
        target_weights = ",".join(f"{symbol}={weight:.16g}" for symbol, weight in sorted(final.weights.items()))
        identity = "|".join(
            (
                account_id,
                strategy_instance_id,
                final.source_signal_id,
                final.decision_time.timestamp.isoformat(),
                final.earliest_execution_time.timestamp.isoformat(),
                target_weights,
                f"cash={final.cash_weight:.16g}",
                ",".join(final.applied_risk_rules),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


class ExecutionPipeline:
    def __init__(self, planner: OrderPlanner, executor):
        self.planner = planner
        self.executor = executor

    def execute(self, context: ExecutionContext, pending: PendingRebalance, account, exchange):
        if context.clock < pending.execute_not_before:
            raise ValueError("PendingRebalance is not ready for execution")
        if pending.expires_at is not None and context.clock > pending.expires_at:
            raise ValueError("PendingRebalance has expired")
        order_list = self.planner.create_orders(context, pending)
        return self.executor.execute_batch(order_list.orders, account, exchange)
