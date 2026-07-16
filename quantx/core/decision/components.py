"""Built-in portfolio construction and risk components."""

from __future__ import annotations

from .types import AlphaSnapshot, FinalTargetPortfolio, MarketObservation, PortfolioState, RawTargetPortfolio


def _ranked(alpha: AlphaSnapshot, top_k: int, score_floor: float | None) -> list:
    values = sorted(alpha.values, key=lambda value: (-value.score, value.instrument))
    if score_floor is not None:
        values = [value for value in values if value.score >= score_floor]
    return values[:top_k]


def _eligible(alpha: AlphaSnapshot, score_floor: float | None) -> list:
    values = sorted(alpha.values, key=lambda value: (-value.score, value.instrument))
    return values if score_floor is None else [value for value in values if value.score >= score_floor]


class TopKEqualWeightConstructor:
    def __init__(self, top_k: int, gross_exposure: float = 1.0, score_floor: float | None = None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= gross_exposure <= 1:
            raise ValueError("gross_exposure must be between 0 and 1")
        self.top_k = int(top_k)
        self.gross_exposure = float(gross_exposure)
        self.score_floor = score_floor

    def construct(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        alpha: AlphaSnapshot,
    ) -> RawTargetPortfolio:
        del portfolio
        selected = _ranked(alpha, self.top_k, self.score_floor)
        weight = self.gross_exposure / len(selected) if selected else 0.0
        weights = {value.instrument: weight for value in selected}
        return RawTargetPortfolio(
            decision_time=alpha.signal_time,
            earliest_execution_time=alpha.earliest_execution_time,
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            source_signal_id=alpha.source_id,
            metadata={"constructor": "topk_equal", "top_k": self.top_k},
        )


class TopKBufferedEqualWeightConstructor:
    """Retain current holdings inside a wider rank buffer to reduce turnover."""

    def __init__(
        self,
        top_k: int,
        rank_buffer: int,
        gross_exposure: float = 1.0,
        score_floor: float | None = None,
    ):
        if top_k < 1 or rank_buffer < top_k:
            raise ValueError("rank_buffer must be greater than or equal to positive top_k")
        if not 0 <= gross_exposure <= 1:
            raise ValueError("gross_exposure must be between 0 and 1")
        self.top_k = int(top_k)
        self.rank_buffer = int(rank_buffer)
        self.gross_exposure = float(gross_exposure)
        self.score_floor = score_floor

    def construct(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        alpha: AlphaSnapshot,
    ) -> RawTargetPortfolio:
        del market
        ranked = _eligible(alpha, self.score_floor)
        buffered_symbols = {value.instrument for value in ranked[: self.rank_buffer]}
        retained = [value for value in ranked[: self.rank_buffer] if value.instrument in portfolio.positions][
            : self.top_k
        ]
        selected_symbols = {value.instrument for value in retained}
        selected = list(retained)
        for value in ranked:
            if len(selected) >= self.top_k:
                break
            if value.instrument not in selected_symbols:
                selected.append(value)
                selected_symbols.add(value.instrument)
        weight = self.gross_exposure / len(selected) if selected else 0.0
        weights = {value.instrument: weight for value in selected}
        return RawTargetPortfolio(
            decision_time=alpha.signal_time,
            earliest_execution_time=alpha.earliest_execution_time,
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            source_signal_id=alpha.source_id,
            metadata={
                "constructor": "topk_buffer_equal",
                "top_k": self.top_k,
                "rank_buffer": self.rank_buffer,
                "retained_count": len(retained),
                "buffered_holding_count": len(set(portfolio.positions).intersection(buffered_symbols)),
            },
        )


class ScoreWeightedPortfolioConstructor:
    def __init__(self, top_k: int, gross_exposure: float = 1.0, score_floor: float | None = None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= gross_exposure <= 1:
            raise ValueError("gross_exposure must be between 0 and 1")
        self.top_k = int(top_k)
        self.gross_exposure = float(gross_exposure)
        self.score_floor = score_floor

    def construct(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        alpha: AlphaSnapshot,
    ) -> RawTargetPortfolio:
        del market, portfolio
        selected = _ranked(alpha, self.top_k, self.score_floor)
        positive_scores = [max(0.0, value.score) for value in selected]
        total = sum(positive_scores)
        if selected and total <= 0:
            positive_scores = [1.0] * len(selected)
            total = float(len(selected))
        weights = {
            value.instrument: self.gross_exposure * score / total for value, score in zip(selected, positive_scores)
        }
        return RawTargetPortfolio(
            decision_time=alpha.signal_time,
            earliest_execution_time=alpha.earliest_execution_time,
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            source_signal_id=alpha.source_id,
            metadata={"constructor": "score_weighted", "top_k": self.top_k},
        )


class NoOpRiskOverlay:
    def apply(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        raw_target: RawTargetPortfolio,
    ) -> FinalTargetPortfolio:
        del market, portfolio
        return FinalTargetPortfolio.from_raw(raw_target)


class StandardRiskOverlay:
    def __init__(
        self,
        *,
        max_position_weight: float = 1.0,
        max_gross_exposure: float = 1.0,
        max_account_drawdown: float | None = None,
        defensive_gross_exposure: float = 0.5,
    ):
        if not 0 < max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 <= max_gross_exposure <= 1 or not 0 <= defensive_gross_exposure <= 1:
            raise ValueError("gross exposure limits must be between 0 and 1")
        if max_account_drawdown is not None and not 0 < max_account_drawdown < 1:
            raise ValueError("max_account_drawdown must be in (0, 1)")
        self.max_position_weight = float(max_position_weight)
        self.max_gross_exposure = float(max_gross_exposure)
        self.max_account_drawdown = max_account_drawdown
        self.defensive_gross_exposure = float(defensive_gross_exposure)

    def apply(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        raw_target: RawTargetPortfolio,
    ) -> FinalTargetPortfolio:
        del market
        rules: list[str] = []
        weights = {
            symbol: min(float(weight), self.max_position_weight) for symbol, weight in raw_target.weights.items()
        }
        if any(weights[symbol] < raw_target.weights[symbol] for symbol in weights):
            rules.append("max_position_weight")
        target_gross = min(sum(weights.values()), self.max_gross_exposure)
        if sum(weights.values()) > self.max_gross_exposure:
            rules.append("max_gross_exposure")
        drawdown = float(getattr(portfolio.account, "drawdown", 0.0) or 0.0)
        if self.max_account_drawdown is not None and drawdown <= -self.max_account_drawdown:
            target_gross = min(target_gross, self.defensive_gross_exposure)
            rules.append("account_drawdown")
        current_gross = sum(weights.values())
        if current_gross > 0 and current_gross > target_gross:
            scale = target_gross / current_gross
            weights = {symbol: weight * scale for symbol, weight in weights.items()}
        return FinalTargetPortfolio.from_raw(
            raw_target,
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            applied_risk_rules=tuple(rules),
        )


class MarketRegimeRiskOverlay(StandardRiskOverlay):
    """Scale exposure using broad-market features observable at signal time."""

    def __init__(
        self,
        *,
        risk_off_gross_exposure: float = 0.35,
        min_market_ret20: float = -0.03,
        min_market_ret60: float = -0.08,
        min_breadth20: float = 0.4,
        defensive_instrument: str | None = None,
        risk_off_total_exposure: float | None = None,
        **standard_kwargs,
    ):
        super().__init__(**standard_kwargs)
        if not 0 <= risk_off_gross_exposure <= 1:
            raise ValueError("risk_off_gross_exposure must be between 0 and 1")
        if not 0 <= min_breadth20 <= 1:
            raise ValueError("min_breadth20 must be between 0 and 1")
        self.risk_off_gross_exposure = float(risk_off_gross_exposure)
        self.min_market_ret20 = float(min_market_ret20)
        self.min_market_ret60 = float(min_market_ret60)
        self.min_breadth20 = float(min_breadth20)
        self.defensive_instrument = str(defensive_instrument) if defensive_instrument else None
        self.risk_off_total_exposure = (
            float(risk_off_total_exposure) if risk_off_total_exposure is not None else self.risk_off_gross_exposure
        )
        if not self.risk_off_gross_exposure <= self.risk_off_total_exposure <= 1:
            raise ValueError("risk_off_total_exposure must cover stock exposure and stay within 1")
        if self.risk_off_total_exposure > self.risk_off_gross_exposure and not self.defensive_instrument:
            raise ValueError("risk_off_total_exposure above stock exposure requires defensive_instrument")

    def apply(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        raw_target: RawTargetPortfolio,
    ) -> FinalTargetPortfolio:
        standard = super().apply(market, portfolio, raw_target)
        values = market.features.values if isinstance(market.features.values, dict) else {}
        checks = (
            values.get("market_ret20", float("inf")) >= self.min_market_ret20,
            values.get("market_ret60", float("inf")) >= self.min_market_ret60,
            values.get("breadth20", 1.0) >= self.min_breadth20,
        )
        if all(checks):
            return standard
        current_gross = sum(standard.weights.values())
        target_gross = min(current_gross, self.risk_off_gross_exposure)
        scale = target_gross / current_gross if current_gross > 0 else 0.0
        weights = {symbol: scaled for symbol, weight in standard.weights.items() if (scaled := weight * scale) > 0}
        if self.defensive_instrument:
            defensive_weight = self.risk_off_total_exposure - sum(weights.values())
            if defensive_weight > 0:
                weights[self.defensive_instrument] = defensive_weight
        return FinalTargetPortfolio.from_raw(
            raw_target,
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            applied_risk_rules=(*standard.applied_risk_rules, "market_regime"),
            metadata={**dict(raw_target.metadata), "market_regime_features": dict(values)},
        )
