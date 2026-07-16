"""Immutable values exchanged between research, decision, and execution layers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .clock import DecisionClock, MarketTime, SessionPhase


def _readonly(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def _require_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FeatureBatch:
    as_of: MarketTime
    instruments: tuple[str, ...]
    values: Any
    schema_id: str
    available_at: MarketTime
    valid_mask: Any | None = None

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("FeatureBatch schema_id is required")
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("FeatureBatch instruments contain duplicates")
        if self.available_at > self.as_of:
            raise ValueError("FeatureBatch cannot be available after as_of")


@dataclass(frozen=True)
class MarketObservation:
    clock: DecisionClock
    universe: tuple[str, ...]
    features: FeatureBatch

    def __post_init__(self) -> None:
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("MarketObservation universe contains duplicates")
        outside = set(self.features.instruments) - set(self.universe)
        if outside:
            raise ValueError(f"Feature instruments are outside the decision universe: {sorted(outside)}")
        if self.features.as_of > self.clock.observation_end:
            raise ValueError("FeatureBatch as_of is after observation_end")
        if self.features.available_at > self.clock.signal_time:
            raise ValueError("FeatureBatch is not available at signal_time")


@dataclass(frozen=True)
class PortfolioState:
    as_of: MarketTime
    account: Any
    positions: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly(self.positions))


@dataclass(frozen=True)
class TradabilityState:
    can_buy: bool
    can_sell: bool
    reason: str = ""


@dataclass(frozen=True)
class ExecutionContext:
    clock: MarketTime
    open_prices: Mapping[str, float]
    tradability: Mapping[str, TradabilityState]
    account: Any
    positions: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.clock.phase != SessionPhase.OPEN:
            raise ValueError("ExecutionContext currently supports OPEN execution only")
        prices = {
            symbol: _require_finite(price, f"open price for {symbol}") for symbol, price in self.open_prices.items()
        }
        if any(price <= 0 for price in prices.values()):
            raise ValueError("Execution prices must be positive")
        object.__setattr__(self, "open_prices", _readonly(prices))
        object.__setattr__(self, "tradability", _readonly(self.tradability))
        object.__setattr__(self, "positions", _readonly(self.positions))

    def price(self, instrument: str) -> float:
        try:
            return float(self.open_prices[instrument])
        except KeyError as exc:
            raise KeyError(f"No execution price for {instrument}") from exc


@dataclass(frozen=True)
class AlphaValue:
    instrument: str
    score: float
    rank: int | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not self.instrument:
            raise ValueError("AlphaValue instrument is required")
        object.__setattr__(self, "score", _require_finite(self.score, "AlphaValue score"))
        if self.rank is not None and self.rank < 1:
            raise ValueError("AlphaValue rank must be positive")
        if self.uncertainty is not None:
            uncertainty = _require_finite(self.uncertainty, "AlphaValue uncertainty")
            if uncertainty < 0:
                raise ValueError("AlphaValue uncertainty cannot be negative")
            object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True)
class AlphaSnapshot:
    signal_time: MarketTime
    earliest_execution_time: MarketTime
    values: tuple[AlphaValue, ...]
    source_id: str
    horizon_sessions: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.earliest_execution_time <= self.signal_time:
            raise ValueError("earliest_execution_time must be after signal_time")
        if not self.source_id:
            raise ValueError("AlphaSnapshot source_id is required")
        instruments = [value.instrument for value in self.values]
        if len(instruments) != len(set(instruments)):
            raise ValueError("AlphaSnapshot contains duplicate instruments")
        if self.horizon_sessions is not None and self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def _validate_target(target: Any) -> None:
    if target.earliest_execution_time <= target.decision_time:
        raise ValueError("earliest_execution_time must be after decision_time")
    if not target.source_signal_id:
        raise ValueError("source_signal_id is required")
    weights = {symbol: _require_finite(weight, f"weight for {symbol}") for symbol, weight in target.weights.items()}
    if any(not symbol for symbol in weights):
        raise ValueError("Target instrument cannot be empty")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Long-only target weights cannot be negative")
    cash_weight = _require_finite(target.cash_weight, "cash_weight")
    if cash_weight < 0:
        raise ValueError("cash_weight cannot be negative")
    if not math.isclose(sum(weights.values()) + cash_weight, 1.0, abs_tol=1e-8):
        raise ValueError("Target weights and cash_weight must sum to 1")
    object.__setattr__(target, "weights", _readonly(weights))
    object.__setattr__(target, "cash_weight", cash_weight)
    object.__setattr__(target, "metadata", _readonly(target.metadata))


class _UniverseValidatedTarget:
    weights: Mapping[str, float]

    def require_universe(self, universe: Sequence[str]) -> None:
        outside = set(self.weights) - set(universe)
        if outside:
            raise ValueError(f"Target instruments are outside the decision universe: {sorted(outside)}")


@dataclass(frozen=True)
class RawTargetPortfolio(_UniverseValidatedTarget):
    decision_time: MarketTime
    earliest_execution_time: MarketTime
    weights: Mapping[str, float]
    cash_weight: float
    source_signal_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_target(self)


@dataclass(frozen=True)
class FinalTargetPortfolio(_UniverseValidatedTarget):
    decision_time: MarketTime
    earliest_execution_time: MarketTime
    weights: Mapping[str, float]
    cash_weight: float
    source_signal_id: str
    applied_risk_rules: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_target(self)

    @classmethod
    def from_raw(
        cls,
        raw: RawTargetPortfolio,
        *,
        weights: Mapping[str, float] | None = None,
        cash_weight: float | None = None,
        applied_risk_rules: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "FinalTargetPortfolio":
        return cls(
            decision_time=raw.decision_time,
            earliest_execution_time=raw.earliest_execution_time,
            weights=raw.weights if weights is None else weights,
            cash_weight=raw.cash_weight if cash_weight is None else cash_weight,
            source_signal_id=raw.source_signal_id,
            applied_risk_rules=applied_risk_rules,
            metadata=raw.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True)
class PendingRebalance:
    rebalance_id: str
    account_id: str
    strategy_instance_id: str
    created_at: MarketTime
    execute_not_before: MarketTime
    expires_at: MarketTime | None
    target: FinalTargetPortfolio
    source_signal_id: str

    def __post_init__(self) -> None:
        if not self.rebalance_id or not self.account_id or not self.strategy_instance_id:
            raise ValueError("PendingRebalance identifiers are required")
        if self.created_at >= self.execute_not_before:
            raise ValueError("PendingRebalance execute_not_before must be after created_at")
        if self.expires_at is not None and self.expires_at < self.execute_not_before:
            raise ValueError("PendingRebalance cannot expire before execute_not_before")
        if self.created_at != self.target.decision_time:
            raise ValueError("PendingRebalance created_at must match target decision_time")
        if self.execute_not_before != self.target.earliest_execution_time:
            raise ValueError("PendingRebalance execution time must match target")
        if self.source_signal_id != self.target.source_signal_id:
            raise ValueError("PendingRebalance source_signal_id must match target")

    @classmethod
    def from_target(
        cls,
        target: FinalTargetPortfolio,
        *,
        rebalance_id: str,
        account_id: str,
        strategy_instance_id: str,
        expires_at: MarketTime | None = None,
    ) -> "PendingRebalance":
        return cls(
            rebalance_id=rebalance_id,
            account_id=account_id,
            strategy_instance_id=strategy_instance_id,
            created_at=target.decision_time,
            execute_not_before=target.earliest_execution_time,
            expires_at=expires_at,
            target=target,
            source_signal_id=target.source_signal_id,
        )
