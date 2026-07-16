"""Portfolio risk overlay protocol."""

from typing import Protocol

from .types import FinalTargetPortfolio, MarketObservation, PortfolioState, RawTargetPortfolio


class RiskOverlay(Protocol):
    def apply(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        raw_target: RawTargetPortfolio,
    ) -> FinalTargetPortfolio: ...
