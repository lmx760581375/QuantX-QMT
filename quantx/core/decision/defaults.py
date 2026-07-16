"""Explicit registration of built-in decision components."""

from __future__ import annotations

from .components import (
    MarketRegimeRiskOverlay,
    NoOpRiskOverlay,
    ScoreWeightedPortfolioConstructor,
    StandardRiskOverlay,
    TopKBufferedEqualWeightConstructor,
    TopKEqualWeightConstructor,
)
from .order_planner import StandardOrderPlanner
from .precomputed import PrecomputedAlphaModel
from .predictions import PredictionStore
from .registry import ComponentCapabilities, ComponentRegistry


def build_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register(
        "alpha",
        "predictions",
        _build_predictions,
        ComponentCapabilities(False, True, False, "AlphaSnapshot"),
    )
    registry.register(
        "portfolio",
        "topk_equal",
        lambda cfg: TopKEqualWeightConstructor(
            top_k=int(cfg.get("top_k", 20)),
            gross_exposure=float(cfg.get("gross_exposure", 1.0)),
            score_floor=float(cfg["score_floor"]) if cfg.get("score_floor") is not None else None,
        ),
        ComponentCapabilities(True, False, False, "RawTargetPortfolio"),
    )
    registry.register(
        "portfolio",
        "topk_buffer",
        lambda cfg: TopKBufferedEqualWeightConstructor(
            top_k=int(cfg.get("top_k", 20)),
            rank_buffer=int(cfg.get("rank_buffer", 40)),
            gross_exposure=float(cfg.get("gross_exposure", 1.0)),
            score_floor=float(cfg["score_floor"]) if cfg.get("score_floor") is not None else None,
        ),
        ComponentCapabilities(True, False, False, "RawTargetPortfolio"),
    )
    registry.register(
        "portfolio",
        "score_weighted",
        lambda cfg: ScoreWeightedPortfolioConstructor(
            top_k=int(cfg.get("top_k", 20)),
            gross_exposure=float(cfg.get("gross_exposure", 1.0)),
            score_floor=float(cfg["score_floor"]) if cfg.get("score_floor") is not None else None,
        ),
        ComponentCapabilities(True, False, False, "RawTargetPortfolio"),
    )
    registry.register(
        "risk",
        "noop",
        lambda cfg: NoOpRiskOverlay(),
        ComponentCapabilities(True, False, False, "FinalTargetPortfolio"),
    )
    registry.register(
        "risk",
        "market_regime",
        lambda cfg: MarketRegimeRiskOverlay(
            max_position_weight=float(cfg.get("max_position_weight", 1.0)),
            max_gross_exposure=float(cfg.get("max_gross_exposure", 1.0)),
            max_account_drawdown=(
                float(cfg["max_account_drawdown"]) if cfg.get("max_account_drawdown") is not None else None
            ),
            defensive_gross_exposure=float(cfg.get("defensive_gross_exposure", 0.5)),
            risk_off_gross_exposure=float(cfg.get("risk_off_gross_exposure", 0.35)),
            min_market_ret20=float(cfg.get("min_market_ret20", -0.03)),
            min_market_ret60=float(cfg.get("min_market_ret60", -0.08)),
            min_breadth20=float(cfg.get("min_breadth20", 0.4)),
            defensive_instrument=cfg.get("defensive_instrument"),
            risk_off_total_exposure=(
                float(cfg["risk_off_total_exposure"]) if cfg.get("risk_off_total_exposure") is not None else None
            ),
        ),
        ComponentCapabilities(True, False, False, "FinalTargetPortfolio"),
    )
    registry.register(
        "risk",
        "standard",
        lambda cfg: StandardRiskOverlay(
            max_position_weight=float(cfg.get("max_position_weight", 1.0)),
            max_gross_exposure=float(cfg.get("max_gross_exposure", 1.0)),
            max_account_drawdown=(
                float(cfg["max_account_drawdown"]) if cfg.get("max_account_drawdown") is not None else None
            ),
            defensive_gross_exposure=float(cfg.get("defensive_gross_exposure", 0.5)),
        ),
        ComponentCapabilities(True, False, False, "FinalTargetPortfolio"),
    )
    registry.register(
        "order_planner",
        "standard",
        lambda cfg: StandardOrderPlanner(lot_size=int(cfg.get("lot_size", 100))),
        ComponentCapabilities(True, False, False, "OrderList"),
    )
    return registry


def _build_predictions(config: dict):
    store = PredictionStore.load(
        config["path"],
        expected_checksum=config.get("checksum"),
    )
    return PrecomputedAlphaModel(
        store,
        artifact_id=str(config["artifact_id"]),
        feature_schema_hash=str(config["feature_schema_hash"]),
    )
