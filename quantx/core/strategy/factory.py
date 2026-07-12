"""Strategy construction for every standard QuantX backtest."""

from __future__ import annotations

from typing import Any, Dict

from quantx.core.engine.cost import TransactionCost
from quantx.core.strategy.config_strategy import build_formula_strategy, explain_strategy_config


def strategy_type(config: Dict[str, Any]) -> str:
    strategy = config.get("strategy") or {}
    return str(strategy.get("type") or "formula").strip().lower()


def build_strategy(config: Dict[str, Any], cost: TransactionCost):
    kind = strategy_type(config)
    if kind == "formula":
        return build_formula_strategy(config, cost)
    if kind == "market_regime_rotation":
        from quantx.strategies.market_regime_rotation import create_market_regime_rotation_strategy

        return create_market_regime_rotation_strategy(config, cost)
    raise ValueError(f"Unsupported strategy.type: {kind}")


def explain_strategy(config: Dict[str, Any]) -> Dict[str, Any]:
    kind = strategy_type(config)
    if kind == "formula":
        result = explain_strategy_config(config)
        result["strategy_type"] = kind
        return result
    if kind == "market_regime_rotation":
        params = dict((config.get("strategy") or {}).get("params") or {})
        return {
            "name": config.get("name"),
            "version": config.get("version"),
            "strategy_type": kind,
            "formula_count": 0,
            "formula_order": [],
            "dependencies": {},
            "selector": {
                "type": "market_regime_rotation",
                "signal_lag": 1,
                "topk": 1,
                "choppy_confirm_days": int(params.get("choppy_confirm_days", 3)),
            },
            "rebalance": {"type": "target_weight", "max_positions": 1},
            "execution": {
                "deal_price": str((config.get("execution") or {}).get("deal_price", "open")),
                "exit_unselected": True,
            },
        }
    raise ValueError(f"Unsupported strategy.type: {kind}")
