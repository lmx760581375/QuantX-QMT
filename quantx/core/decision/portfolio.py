"""Portfolio construction protocol."""

from typing import Protocol

from .types import AlphaSnapshot, MarketObservation, PortfolioState, RawTargetPortfolio


class PortfolioConstructor(Protocol):
    def construct(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        alpha: AlphaSnapshot,
    ) -> RawTargetPortfolio: ...
