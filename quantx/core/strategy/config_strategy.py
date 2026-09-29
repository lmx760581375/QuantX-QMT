"""Config-driven formula strategy primitives."""

from __future__ import annotations

import ast
import json
import operator as py_operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from quantx.core.engine.cost import TransactionCost
from quantx.core.engine.types import Order, OrderAction
from quantx.core.factor_runtime import FormulaError
from quantx.core.factor_runtime.operators import DEFAULT_REGISTRY
from quantx.core.strategy.base import (
    CompositeStrategy,
    ExecutionStrategy,
    OrderList,
    PolicyState,
    RebalanceStrategy,
    Signal,
    StockSelection,
    StockSelector,
    WeightAllocation,
)


class ConfigStrategyError(ValueError):
    """Raised when a config-driven strategy is invalid."""


@dataclass(frozen=True)
class SellRule:
    name: str
    when: str
    action: str = "sell_all"
    position_pct: Optional[float] = None


@dataclass(frozen=True)
class PositionLimitRule:
    when: str
    value: int


@dataclass(frozen=True)
class CashUseRatioRule:
    when: str
    value: float


@dataclass(frozen=True)
class CompiledStrategySpec:
    """Static compile result for a config-driven strategy."""

    name: str
    version: Any
    fields: Dict[str, str]
    formulas: Dict[str, str]
    formula_order: List[str]
    dependencies: Dict[str, List[str]]
    selector: Dict[str, Any]
    rebalance: Dict[str, Any]
    execution: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "fields": self.fields,
            "formula_count": len(self.formulas),
            "formula_order": self.formula_order,
            "dependencies": self.dependencies,
            "selector": self.selector,
            "rebalance": self.rebalance,
            "execution": self.execution,
        }


STATEFUL_RULE_NAMES = {
    "is_holding",
    "price",
    "cost_price",
    "avg_cost",
    "holding_days",
    "position_qty",
    "initial_position_qty",
    "remaining_position_pct",
    "position_market_value",
    "position_weight",
    "pnl_pct",
    "peak_price",
    "trough_price",
    "peak_pnl_pct",
    "trough_pnl_pct",
    "drawdown_from_peak",
    "cash",
    "total_value",
    "account_drawdown",
    "limit_up",
    "one_side_limit_up",
}

HOLD_CONTEXT_RULE_NAMES = {
    "hold_first_3d_return",
    "hold_first_5d_return",
    "hold_first_10d_return",
}

WATCHLIST_RULE_NAMES = {
    "watchlist_age",
    "watchlist_pnl",
    "watchlist_peak_pnl",
    "watchlist_trough_pnl",
    "watchlist_drawdown_from_peak",
    "setup_score",
}

DEFAULT_FIELD_NAMES = {"open", "high", "low", "close", "volume", "vwap", "change"}
DEFAULT_INDUSTRY_CSV = Path("data/meta/snapshots/industry_membership.csv")
DEFAULT_SECTOR_CSV = Path("data/meta/snapshots/sector_membership.csv")


def _is_position_context_rule_name(name: str) -> bool:
    return name.startswith("entry_") or name in HOLD_CONTEXT_RULE_NAMES


def _unknown_scalar_names(
    names: Set[str],
    scalar_sources: Set[str],
    funcs: Set[str],
    *,
    allow_position_context: bool = False,
) -> Set[str]:
    unknown = names - scalar_sources - funcs
    if allow_position_context:
        unknown = {
            name for name in unknown
            if name not in HOLD_CONTEXT_RULE_NAMES
            and not (name.startswith("entry_") and name[len("entry_"):] in scalar_sources)
        }
    return unknown


class ScalarRuleEvaluator:
    """Small safe evaluator for stateful scalar trading rules."""

    def __init__(self, funcs: Optional[Dict[str, Any]] = None):
        self.funcs = funcs or {
            "abs": abs,
            "Abs": abs,
            "Max": max,
            "Min": min,
            "Maximum": max,
            "Minimum": min,
        }

    def names(self, expr: str) -> Set[str]:
        tree = self._parse(expr)
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def evaluate(self, expr: str, values: Dict[str, Any]) -> Any:
        tree = self._parse(expr)
        return self._eval_node(tree.body, values)

    def _parse(self, expr: str) -> ast.Expression:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(str(exc)) from exc
        self._validate_ast(tree)
        return tree

    def _validate_ast(self, tree: ast.AST) -> None:
        forbidden = (
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.GeneratorExp,
        )
        for node in ast.walk(tree):
            if isinstance(node, forbidden):
                raise FormulaError(f"Unsupported rule syntax: {type(node).__name__}")
            if isinstance(node, (ast.BitAnd, ast.BitOr, ast.Invert)):
                raise FormulaError("Use and/or/not or And()/Or()/Not() instead of &, |, or ~")

    def _eval_node(self, node: ast.AST, values: Dict[str, Any]):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            raise FormulaError(f"Unknown rule variable: {node.id}")
        if isinstance(node, ast.BinOp):
            op = _SCALAR_BIN_OPS.get(type(node.op))
            if op is None:
                raise FormulaError(f"Unsupported binary operator: {type(node.op).__name__}")
            return op(self._eval_node(node.left, values), self._eval_node(node.right, values))
        if isinstance(node, ast.UnaryOp):
            op = _SCALAR_UNARY_OPS.get(type(node.op))
            if op is None:
                raise FormulaError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(self._eval_node(node.operand, values))
        if isinstance(node, ast.BoolOp):
            vals = [bool(self._eval_node(value, values)) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            raise FormulaError(f"Unsupported boolean operator: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, values)
            for op_node, comparator in zip(node.ops, node.comparators):
                op = _SCALAR_CMP_OPS.get(type(op_node))
                if op is None:
                    raise FormulaError(f"Unsupported comparison operator: {type(op_node).__name__}")
                right = self._eval_node(comparator, values)
                if not op(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("Only registered scalar functions are supported")
            if node.keywords:
                raise FormulaError("Keyword arguments are not supported in scalar rules")
            name = node.func.id
            if name not in self.funcs:
                raise FormulaError(f"Unknown scalar function: {name}")
            return self.funcs[name](*[self._eval_node(arg, values) for arg in node.args])
        raise FormulaError(f"Unsupported rule syntax: {type(node).__name__}")


def compile_strategy_config(config: Dict[str, Any]) -> CompiledStrategySpec:
    """Validate and explain a config-driven strategy without loading market data."""
    if not isinstance(config, dict):
        raise ConfigStrategyError("Config must be a mapping")

    fields = _normalize_fields(config.get("fields") or {})
    available_fields = set(DEFAULT_FIELD_NAMES) | set(fields)
    factors = _normalize_formula_mapping(config.get("factors") or {}, "factors")
    groups = _normalize_group_config(config.get("groups") or {})
    group_factors = _normalize_formula_mapping(config.get("group_factors") or {}, "group_factors")
    signals = _normalize_formula_mapping(config.get("signals") or {}, "signals")
    formulas = {**factors, **group_factors, **signals}
    available_fields |= set(groups)

    selector_cfg = dict(config.get("selector") or {})
    rebalance_cfg = dict(config.get("rebalance") or {})
    execution_cfg = dict(config.get("execution") or {})
    _validate_selector_config(selector_cfg)
    _validate_rebalance_config(rebalance_cfg, formulas, available_fields)
    _validate_execution_config(execution_cfg, formulas, available_fields)
    if (
        (execution_cfg.get("buy") or {}).get("sizing", "cash_equal") == "target_weight"
        and rebalance_cfg.get("weight_scope", "available_slots") != "portfolio_target"
    ):
        raise ConfigStrategyError(
            "buy.sizing=target_weight requires rebalance.weight_scope=portfolio_target"
        )

    selector_formulas = {
        "__selector_where": str(selector_cfg.get("where")),
        "__selector_score": str(selector_cfg.get("score", 0)),
    }
    all_formulas = {**formulas, **selector_formulas}
    order, dependencies = _plan_formula_order(all_formulas, available_fields)

    selector_deps = set(dependencies.get("__selector_where", [])) | set(dependencies.get("__selector_score", []))
    if selector_cfg.get("mode", "precomputed") in {"precomputed", "watchlist"} and selector_deps & STATEFUL_RULE_NAMES:
        raise ConfigStrategyError(
            f"selector.mode={selector_cfg.get('mode', 'precomputed')} cannot reference account/position state: "
            f"{sorted(selector_deps & STATEFUL_RULE_NAMES)}"
        )
    if selector_cfg.get("mode", "precomputed") == "watchlist":
        evaluator = ScalarRuleEvaluator()
        confirm_names = evaluator.names(str(selector_cfg.get("confirm")))
        scalar_sources = set(formulas) | available_fields | WATCHLIST_RULE_NAMES
        setup_names = {name for name in confirm_names if name.startswith("setup_")}
        setup_unknown = {
            name for name in setup_names
            if name[len("setup_"):] not in scalar_sources
        }
        unknown = confirm_names - scalar_sources - setup_names - set(evaluator.funcs)
        if unknown or setup_unknown:
            raise ConfigStrategyError(
                f"Watchlist confirm rule references unknown variables: {sorted(unknown | setup_unknown)}"
            )

    return CompiledStrategySpec(
        name=str(config.get("name") or ""),
        version=config.get("version"),
        fields=fields,
        formulas=formulas,
        formula_order=[name for name in order if not name.startswith("__selector_")],
        dependencies={name: deps for name, deps in dependencies.items() if not name.startswith("__selector_")},
        selector={
            "mode": selector_cfg.get("mode", "precomputed"),
            "where": selector_cfg.get("where"),
            "score": selector_cfg.get("score", 0),
            "external_score": selector_cfg.get("external_score") or None,
            "lag": int(selector_cfg.get("lag", 1)),
            "sort": selector_cfg.get("sort", "input_order"),
            "topk": selector_cfg.get("topk"),
            "wrap_first_signal": bool(selector_cfg.get("wrap_first_signal", False)),
            "confirm": selector_cfg.get("confirm"),
            "watchlist": selector_cfg.get("watchlist") or {},
        },
        rebalance={
            "type": rebalance_cfg.get("type", "equal_weight"),
            "max_positions": int(rebalance_cfg.get("max_positions", 10)),
            "max_positions_rules": rebalance_cfg.get("max_positions_rules") or [],
            "cash_use_ratio": float(rebalance_cfg.get("cash_use_ratio", 0.99)),
            "cash_use_ratio_rules": rebalance_cfg.get("cash_use_ratio_rules") or [],
            "weight_scope": rebalance_cfg.get("weight_scope", "available_slots"),
            "buy_only_new_positions": bool(rebalance_cfg.get("buy_only_new_positions", True)),
            "exclude_partial_positions_from_slots": bool(
                rebalance_cfg.get("exclude_partial_positions_from_slots", False)
            ),
            "rank_weights": rebalance_cfg.get("rank_weights") or [],
        },
        execution={
            "deal_price": execution_cfg.get("deal_price", "open"),
            "buy_deal_price": execution_cfg.get("buy_deal_price"),
            "sell_deal_price": execution_cfg.get("sell_deal_price"),
            "price_jump_limit": execution_cfg.get("price_jump_limit", 0.095),
            "max_position_weight": execution_cfg.get("max_position_weight"),
            "trim_overweight_positions": bool(
                execution_cfg.get("trim_overweight_positions", False)
            ),
            "sell_rules": execution_cfg.get("sell_rules") or [],
            "buy": execution_cfg.get("buy") or {},
            "position_manager": execution_cfg.get("position_manager") or None,
        },
    )


def explain_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return compile_strategy_config(config).to_dict()


def _normalize_fields(raw: Dict[str, Any]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    if not isinstance(raw, dict):
        raise ConfigStrategyError("fields must be a mapping")
    for alias, source in raw.items():
        alias_name = str(alias).lstrip("$")
        source_name = str(source)
        if not source_name.startswith("$"):
            source_name = f"${source_name.lstrip('$')}"
        if not alias_name.isidentifier():
            raise ConfigStrategyError(f"Field alias must be a valid identifier: {alias}")
        fields[alias_name] = source_name
    return fields


def _normalize_formula_mapping(raw: Dict[str, Any], section: str) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigStrategyError(f"{section} must be a mapping")
    formulas: Dict[str, str] = {}
    for name, expr in raw.items():
        clean_name = str(name)
        if not clean_name.isidentifier():
            raise ConfigStrategyError(f"{section}.{name} must be a valid identifier")
        formulas[clean_name] = str(expr)
    return formulas


def _normalize_group_config(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ConfigStrategyError("groups must be a mapping")
    groups: Dict[str, Dict[str, Any]] = {}
    for name, cfg in raw.items():
        clean_name = str(name)
        if not clean_name.isidentifier():
            raise ConfigStrategyError(f"groups.{name} must be a valid identifier")
        if isinstance(cfg, str):
            cfg = {"source": cfg}
        if not isinstance(cfg, dict):
            raise ConfigStrategyError(f"groups.{name} must be a mapping")
        source = str(cfg.get("source") or "")
        if source not in {"meta.industry_l1", "industry_l1", "meta.concept", "concept"}:
            raise ConfigStrategyError(f"Unsupported groups.{name}.source: {source}")
        groups[clean_name] = dict(cfg)
    return groups


def _validate_selector_config(selector_cfg: Dict[str, Any]) -> None:
    if not selector_cfg.get("where"):
        raise ConfigStrategyError("selector.where is required")
    mode = selector_cfg.get("mode", "precomputed")
    if mode not in {"precomputed", "watchlist"}:
        raise ConfigStrategyError(f"Unsupported selector.mode: {mode}")
    if mode == "watchlist" and not selector_cfg.get("confirm"):
        raise ConfigStrategyError("selector.confirm is required when selector.mode=watchlist")
    lag = int(selector_cfg.get("lag", 1))
    if lag < 1:
        raise ConfigStrategyError("selector.lag must be >= 1 to avoid same-day lookahead trading")
    sort = selector_cfg.get("sort", "input_order")
    if sort not in {"input_order", "none", "score_desc", "score_asc"}:
        raise ConfigStrategyError(f"Unsupported selector.sort: {sort}")
    external_score = selector_cfg.get("external_score")
    if external_score is not None:
        if not isinstance(external_score, dict):
            raise ConfigStrategyError("selector.external_score must be a mapping")
        path = str(external_score.get("path") or "")
        if not path:
            raise ConfigStrategyError("selector.external_score.path is required")
        missing = str(external_score.get("missing", "drop"))
        if missing not in {"drop", "keep_original"}:
            raise ConfigStrategyError(f"Unsupported selector.external_score.missing: {missing}")
        require_artifact_manifest = external_score.get("require_artifact_manifest", False)
        if not isinstance(require_artifact_manifest, bool):
            raise ConfigStrategyError("selector.external_score.require_artifact_manifest must be boolean")


def _validate_rebalance_config(
    rebalance_cfg: Dict[str, Any],
    formulas: Optional[Dict[str, str]] = None,
    available_fields: Optional[Set[str]] = None,
) -> None:
    if rebalance_cfg.get("type", "equal_weight") != "equal_weight":
        raise ConfigStrategyError(f"Unsupported rebalance.type: {rebalance_cfg.get('type')}")
    if int(rebalance_cfg.get("max_positions", 10)) <= 0:
        raise ConfigStrategyError("rebalance.max_positions must be positive")
    weight_scope = str(rebalance_cfg.get("weight_scope", "available_slots"))
    if weight_scope not in {"available_slots", "portfolio_target"}:
        raise ConfigStrategyError("rebalance.weight_scope must be 'available_slots' or 'portfolio_target'")
    exclude_partial = rebalance_cfg.get("exclude_partial_positions_from_slots", False)
    if not isinstance(exclude_partial, bool):
        raise ConfigStrategyError(
            "rebalance.exclude_partial_positions_from_slots must be boolean"
        )
    cash_use_ratio = float(rebalance_cfg.get("cash_use_ratio", 0.99))
    if not 0.0 <= cash_use_ratio <= 1.0:
        raise ConfigStrategyError("rebalance.cash_use_ratio must be within [0, 1]")
    rank_weights = rebalance_cfg.get("rank_weights") or []
    if rank_weights:
        if not isinstance(rank_weights, list):
            raise ConfigStrategyError("rebalance.rank_weights must be a list")
        if any(float(value) < 0 for value in rank_weights):
            raise ConfigStrategyError("rebalance.rank_weights must be non-negative")
    evaluator = ScalarRuleEvaluator()
    scalar_sources = set(formulas or {}) | set(available_fields or DEFAULT_FIELD_NAMES) | STATEFUL_RULE_NAMES
    for raw_rule in rebalance_cfg.get("max_positions_rules") or []:
        rule = raw_rule if isinstance(raw_rule, PositionLimitRule) else PositionLimitRule(**raw_rule)
        if int(rule.value) <= 0:
            raise ConfigStrategyError("rebalance.max_positions_rules.value must be positive")
        unknown = _unknown_scalar_names(evaluator.names(rule.when), scalar_sources, set(evaluator.funcs))
        if unknown:
            raise ConfigStrategyError(
                f"Rebalance max_positions rule references unknown variables: {sorted(unknown)}"
            )
    for raw_rule in rebalance_cfg.get("cash_use_ratio_rules") or []:
        rule = raw_rule if isinstance(raw_rule, CashUseRatioRule) else CashUseRatioRule(**raw_rule)
        if not 0.0 <= float(rule.value) <= 1.0:
            raise ConfigStrategyError("rebalance.cash_use_ratio_rules.value must be within [0, 1]")
        unknown = _unknown_scalar_names(evaluator.names(rule.when), scalar_sources, set(evaluator.funcs))
        if unknown:
            raise ConfigStrategyError(
                f"Rebalance cash_use_ratio rule references unknown variables: {sorted(unknown)}"
            )


def _validate_execution_config(
    execution_cfg: Dict[str, Any],
    formulas: Dict[str, str],
    available_fields: Set[str],
) -> None:
    deal_price = str(execution_cfg.get("deal_price", "open")).lstrip("$")
    if deal_price not in available_fields and deal_price not in formulas:
        raise ConfigStrategyError(
            f"execution.deal_price references unknown field or formula: {deal_price}"
        )
    for key in ("buy_deal_price", "sell_deal_price"):
        value = execution_cfg.get(key)
        if value is None:
            continue
        price = str(value).lstrip("$")
        if price not in available_fields and price not in formulas:
            raise ConfigStrategyError(f"execution.{key} references unknown field or formula: {price}")
    buy_cfg = execution_cfg.get("buy") or {}
    price_jump_limit = execution_cfg.get("price_jump_limit", 0.095)
    if price_jump_limit != "board_limit":
        if isinstance(price_jump_limit, bool):
            raise ConfigStrategyError("execution.price_jump_limit must be a positive number or 'board_limit'")
        try:
            if float(price_jump_limit) <= 0.0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ConfigStrategyError("execution.price_jump_limit must be a positive number or 'board_limit'") from exc
    position_manager_cfg = execution_cfg.get("position_manager")
    if position_manager_cfg is not None:
        if not isinstance(position_manager_cfg, dict):
            raise ConfigStrategyError("execution.position_manager must be a mapping")
        if str(position_manager_cfg.get("type")) != "python_plugin":
            raise ConfigStrategyError("execution.position_manager.type must be python_plugin")
        factory = str(position_manager_cfg.get("factory") or "")
        if ":" not in factory:
            raise ConfigStrategyError("execution.position_manager.factory must be module:callable")
        if not str(position_manager_cfg.get("python_path") or ""):
            raise ConfigStrategyError("execution.position_manager.python_path is required")
        if not isinstance(position_manager_cfg.get("config") or {}, dict):
            raise ConfigStrategyError("execution.position_manager.config must be a mapping")
    if buy_cfg.get("sizing", "cash_equal") != "cash_equal":
        if buy_cfg.get("sizing") != "target_weight":
            raise ConfigStrategyError(f"Unsupported buy.sizing: {buy_cfg.get('sizing')}")
    max_position_weight = execution_cfg.get("max_position_weight")
    if max_position_weight is not None:
        if isinstance(max_position_weight, bool):
            raise ConfigStrategyError("execution.max_position_weight must be within (0, 1]")
        try:
            value = float(max_position_weight)
        except (TypeError, ValueError) as exc:
            raise ConfigStrategyError("execution.max_position_weight must be within (0, 1]") from exc
        if not 0.0 < value <= 1.0:
            raise ConfigStrategyError("execution.max_position_weight must be within (0, 1]")
    trim_overweight_positions = execution_cfg.get("trim_overweight_positions", False)
    if not isinstance(trim_overweight_positions, bool):
        raise ConfigStrategyError("execution.trim_overweight_positions must be boolean")
    evaluator = ScalarRuleEvaluator()
    scalar_sources = set(formulas) | available_fields | STATEFUL_RULE_NAMES
    if buy_cfg.get("when"):
        unknown = _unknown_scalar_names(evaluator.names(str(buy_cfg["when"])), scalar_sources, set(evaluator.funcs))
        if unknown:
            raise ConfigStrategyError(f"Buy rule references unknown variables: {sorted(unknown)}")
    if buy_cfg.get("add_existing_when"):
        unknown = _unknown_scalar_names(evaluator.names(str(buy_cfg["add_existing_when"])), scalar_sources, set(evaluator.funcs))
        if unknown:
            raise ConfigStrategyError(f"Buy add_existing_when rule references unknown variables: {sorted(unknown)}")
    for rule in execution_cfg.get("sell_rules") or []:
        sell_rule = rule if isinstance(rule, SellRule) else SellRule(**rule)
        if sell_rule.action not in {"sell_all", "sell_to_position_pct"}:
            raise ConfigStrategyError(f"Unsupported sell rule action: {sell_rule.action}")
        if sell_rule.action == "sell_to_position_pct":
            if sell_rule.position_pct is None:
                raise ConfigStrategyError(f"Sell rule {sell_rule.name} requires position_pct")
            position_pct = float(sell_rule.position_pct)
            if position_pct < 0 or position_pct >= 1:
                raise ConfigStrategyError(
                    f"Sell rule {sell_rule.name} position_pct must be >= 0 and < 1"
                )
        unknown = _unknown_scalar_names(
            evaluator.names(sell_rule.when),
            scalar_sources,
            set(evaluator.funcs),
            allow_position_context=True,
        )
        if unknown:
            raise ConfigStrategyError(f"Sell rule {sell_rule.name} references unknown variables: {sorted(unknown)}")


def _plan_formula_order(
    formulas: Dict[str, str],
    available_fields: Set[str],
) -> tuple[List[str], Dict[str, List[str]]]:
    pending = dict(formulas)
    resolved = set(available_fields)
    operator_names = set(DEFAULT_REGISTRY.names())
    dependencies: Dict[str, List[str]] = {}
    order: List[str] = []
    while pending:
        progressed = False
        for name, expr in list(pending.items()):
            deps = _names_in_expr(expr) - operator_names
            dependencies[name] = sorted(deps)
            unknown = deps - resolved - set(pending)
            if unknown:
                raise ConfigStrategyError(f"Formula {name} references unknown variables: {sorted(unknown)}")
            if deps <= resolved:
                order.append(name)
                resolved.add(name)
                del pending[name]
                progressed = True
        if not progressed:
            raise ConfigStrategyError(f"Formula dependency cycle: {sorted(pending)}")
    return order, dependencies


def _names_in_expr(expr: str) -> Set[str]:
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as exc:
        raise ConfigStrategyError(str(exc)) from exc
    ScalarRuleEvaluator()._validate_ast(tree)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


_SCALAR_BIN_OPS = {
    ast.Add: py_operator.add,
    ast.Sub: py_operator.sub,
    ast.Mult: py_operator.mul,
    ast.Div: py_operator.truediv,
    ast.Mod: py_operator.mod,
}

_SCALAR_UNARY_OPS = {
    ast.USub: py_operator.neg,
    ast.UAdd: py_operator.pos,
    ast.Not: py_operator.not_,
}

_SCALAR_CMP_OPS = {
    ast.Gt: py_operator.gt,
    ast.GtE: py_operator.ge,
    ast.Lt: py_operator.lt,
    ast.LtE: py_operator.le,
    ast.Eq: py_operator.eq,
    ast.NotEq: py_operator.ne,
}


class FormulaSelector(StockSelector):
    """Selector that computes full-market formulas and emits daily stock signals."""

    def __init__(
        self,
        formulas: Dict[str, str],
        where: str,
        score: str | int | float = 0,
        lag: int = 1,
        wrap_first_signal: bool = False,
        sort: str = "input_order",
        topk: Optional[int] = None,
        reason: str = "config_buy",
        candidate_limit: int = 20,
        external_score: Optional[Dict[str, Any]] = None,
    ):
        self.formulas = dict(formulas)
        self.where_expr = str(where)
        self.score_expr = str(score)
        self.lag = int(lag)
        if self.lag < 1:
            raise ConfigStrategyError("selector.lag must be >= 1 to avoid same-day lookahead trading")
        self.wrap_first_signal = bool(wrap_first_signal)
        self.sort = sort
        self.topk = topk
        self.reason = reason
        self.candidate_limit = int(candidate_limit)
        self.external_score_cfg = dict(external_score or {})
        self.external_scores: Optional[pd.Series] = None
        self.where_name = "__selector_where"
        self.score_name = "__selector_score"

    def prepare(self, context):
        runtime = getattr(context, "factor_runtime", None)
        if runtime is None:
            raise ConfigStrategyError("FormulaSelector requires BacktestContext.factor_runtime")
        formulas = dict(self.formulas)
        formulas[self.where_name] = self.where_expr
        formulas[self.score_name] = self.score_expr
        runtime.compute_formulas(formulas)
        self.external_scores = _load_external_score_series(self.external_score_cfg) if self.external_score_cfg else None
        self._record_daily_candidates(context, runtime)

    def act(self, state: PolicyState) -> StockSelection:
        runtime = getattr(state.context, "factor_runtime", None)
        if runtime is None:
            raise ConfigStrategyError("FormulaSelector requires BacktestContext.factor_runtime")
        signal_date = self._signal_date(state)
        if signal_date is None:
            self._record_candidates(state, None, None, [], [])
            return StockSelection(signals=[])

        section, raw_selected, sorted_selected, selected = self._candidate_frames(runtime, signal_date)

        signals = [
            Signal(symbol=str(symbol), score=_safe_float(row[self.score_name]), reason=self.reason)
            for symbol, row in selected.iterrows()
        ]
        self._record_candidates(state, signal_date, section, raw_selected, sorted_selected, selected)
        return StockSelection(signals=signals)

    def _candidate_frames(self, runtime, signal_date: str):
        try:
            section = runtime.get_cross_section([self.where_name, self.score_name], signal_date)
        except KeyError as exc:
            raise ConfigStrategyError(f"Selector factors are unavailable for {signal_date}") from exc

        mask = _to_bool_series(section[self.where_name])
        raw_selected = section.loc[mask].copy()
        if raw_selected.empty:
            return section, raw_selected, raw_selected, raw_selected
        raw_selected = self._apply_external_score(raw_selected, signal_date)
        if raw_selected.empty:
            return section, raw_selected, raw_selected, raw_selected

        if self.sort == "score_desc":
            sorted_selected = raw_selected.sort_values(self.score_name, ascending=False)
        elif self.sort == "score_asc":
            sorted_selected = raw_selected.sort_values(self.score_name, ascending=True)
        elif self.sort in {"input_order", "none"}:
            sorted_selected = raw_selected
        else:
            raise ConfigStrategyError(f"Unsupported selector sort: {self.sort}")

        final_selected = sorted_selected
        if self.topk is not None:
            final_selected = sorted_selected.head(int(self.topk))
        return section, raw_selected, sorted_selected, final_selected

    def _apply_external_score(self, frame: pd.DataFrame, signal_date: str) -> pd.DataFrame:
        if self.external_scores is None:
            return frame
        if frame.empty:
            return frame
        date_key = _normalize_external_score_date(signal_date)
        keys = pd.MultiIndex.from_arrays(
            [[date_key] * len(frame), [str(symbol) for symbol in frame.index]],
            names=["date", "instrument"],
        )
        scores = self.external_scores.reindex(keys).to_numpy(dtype=float)
        out = frame.copy()
        out[self.score_name] = scores
        missing = str(self.external_score_cfg.get("missing", "drop"))
        if missing == "drop":
            return out[pd.notna(out[self.score_name])]
        if missing == "keep_original":
            original = pd.to_numeric(frame[self.score_name], errors="coerce")
            out[self.score_name] = pd.to_numeric(out[self.score_name], errors="coerce").fillna(original)
            return out
        raise ConfigStrategyError(f"Unsupported selector.external_score.missing: {missing}")

    def _record_daily_candidates(self, context, runtime) -> None:
        recorder = getattr(context, "record_daily_selection_candidates", None)
        if not callable(recorder):
            return
        for signal_date in list(getattr(context, "trade_dates", [])):
            section, raw_selected, sorted_selected, final_selected = self._candidate_frames(runtime, signal_date)
            detail = self._candidate_detail(
                date=signal_date,
                signal_date=signal_date,
                section=section,
                raw_selected=raw_selected,
                sorted_selected=sorted_selected,
                final_selected=final_selected,
            )
            detail["mode"] = "daily_signal"
            detail["execution_lag"] = self.lag
            detail["for_next_session"] = True
            recorder(signal_date, detail)

    def _record_candidates(
        self,
        state: PolicyState,
        signal_date: Optional[str],
        section: Optional[pd.DataFrame],
        raw_selected: Any,
        sorted_selected: Any,
        final_selected: Any = None,
    ) -> None:
        context = getattr(state, "context", None)
        recorder = getattr(context, "record_selection_candidates", None) if context is not None else None
        if not callable(recorder):
            return
        detail = self._candidate_detail(
            date=state.date,
            signal_date=signal_date,
            section=section,
            raw_selected=raw_selected,
            sorted_selected=sorted_selected,
            final_selected=final_selected,
        )
        recorder(state.date, detail)

    def _candidate_detail(
        self,
        date: str,
        signal_date: Optional[str],
        section: Optional[pd.DataFrame],
        raw_selected: Any,
        sorted_selected: Any,
        final_selected: Any = None,
    ) -> Dict[str, Any]:
        final_selected = sorted_selected if final_selected is None else final_selected
        raw_count = int(len(raw_selected)) if hasattr(raw_selected, "__len__") else 0
        return {
            "date": date,
            "signal_date": signal_date,
            "where": self.where_expr,
            "score": self.score_expr,
            "sort": self.sort,
            "topk": self.topk,
            "lag": self.lag,
            "wrap_first_signal": self.wrap_first_signal,
            "external_score": self.external_score_cfg or None,
            "universe_count": int(len(section)) if section is not None else 0,
            "raw_candidate_count": raw_count,
            "selected_count": int(len(final_selected)) if hasattr(final_selected, "__len__") else 0,
            "raw_candidates": self._candidate_rows(raw_selected, self.candidate_limit),
            "selected_candidates": self._candidate_rows(final_selected, self.candidate_limit),
            "reason": self.reason,
        }

    def _candidate_rows(self, frame: Any, limit: int) -> List[Dict[str, Any]]:
        if frame is None or not hasattr(frame, "head"):
            return []
        rows = []
        try:
            view = frame.head(max(0, int(limit)))
            for symbol, row in view.iterrows():
                rows.append({
                    "symbol": str(symbol),
                    "score": _safe_float(row.get(self.score_name)),
                    "where": bool(_safe_float(row.get(self.where_name))),
                })
        except Exception:
            return []
        return rows

    def _signal_date(self, state: PolicyState) -> Optional[str]:
        dates = list(getattr(state.context, "trade_dates", []))
        if not dates:
            return None
        try:
            idx = dates.index(state.date)
        except ValueError:
            return None
        signal_idx = idx - self.lag
        if signal_idx >= 0:
            return dates[signal_idx]
        if self.wrap_first_signal and dates:
            return dates[signal_idx]
        return None


class GroupAwareFormulaSelector(FormulaSelector):
    """Formula selector that exposes static metadata groups to formulas."""

    def __init__(self, *args, groups: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups or {}

    def prepare(self, context):
        runtime = getattr(context, "factor_runtime", None)
        if runtime is None:
            raise ConfigStrategyError("GroupAwareFormulaSelector requires BacktestContext.factor_runtime")
        _inject_runtime_groups(runtime, self.groups)
        super().prepare(context)


class WatchlistFormulaSelector(StockSelector):
    """Stateful selector that turns setup signals into later confirmed entries."""

    def __init__(
        self,
        formulas: Dict[str, str],
        setup: str,
        confirm: str,
        score: str | int | float = 0,
        lag: int = 1,
        max_age: int = 12,
        min_age: int = 1,
        sort: str = "input_order",
        topk: Optional[int] = None,
        reason: str = "watchlist_buy",
        candidate_limit: int = 20,
        remove_on_confirm: bool = True,
        groups: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.formulas = dict(formulas)
        self.setup_expr = str(setup)
        self.confirm_expr = str(confirm)
        self.score_expr = str(score)
        self.lag = int(lag)
        if self.lag < 1:
            raise ConfigStrategyError("selector.lag must be >= 1 to avoid same-day lookahead trading")
        self.max_age = int(max_age)
        self.min_age = int(min_age)
        if self.max_age < 1:
            raise ConfigStrategyError("selector.watchlist.max_age must be >= 1")
        if self.min_age < 0:
            raise ConfigStrategyError("selector.watchlist.min_age must be >= 0")
        if self.min_age > self.max_age:
            raise ConfigStrategyError("selector.watchlist.min_age must be <= max_age")
        self.sort = sort
        self.topk = topk
        self.reason = reason
        self.candidate_limit = int(candidate_limit)
        self.remove_on_confirm = bool(remove_on_confirm)
        self.groups = groups or {}
        self.setup_name = "__watchlist_setup"
        self.score_name = "__watchlist_score"
        self.evaluator = ScalarRuleEvaluator()
        self.confirm_names = self.evaluator.names(self.confirm_expr)
        self._watchlist: Dict[str, Dict[str, Any]] = {}

    def prepare(self, context):
        runtime = getattr(context, "factor_runtime", None)
        if runtime is None:
            raise ConfigStrategyError("WatchlistFormulaSelector requires BacktestContext.factor_runtime")
        _inject_runtime_groups(runtime, self.groups)
        formulas = dict(self.formulas)
        formulas[self.setup_name] = self.setup_expr
        formulas[self.score_name] = self.score_expr
        runtime.compute_formulas(formulas)
        self._watchlist.clear()

    def act(self, state: PolicyState) -> StockSelection:
        runtime = getattr(state.context, "factor_runtime", None)
        if runtime is None:
            raise ConfigStrategyError("WatchlistFormulaSelector requires BacktestContext.factor_runtime")
        dates = list(getattr(state.context, "trade_dates", []))
        current_idx = self._date_index(dates, state.date)
        if current_idx is None:
            self._record_candidates(state, None, [], [], 0, 0)
            return StockSelection(signals=[])

        self._add_setups(runtime, dates, current_idx)
        expired = self._drop_expired(current_idx)

        if not self._watchlist:
            self._record_candidates(state, None, [], [], 0, expired)
            return StockSelection(signals=[])

        factor_names = self._confirm_factor_names(runtime)
        section = runtime.get_cross_section([self.score_name, *factor_names], state.date)
        rows = []
        for symbol, item in self._watchlist.items():
            age = current_idx - int(item["setup_idx"])
            if age < self.min_age or age > self.max_age or symbol not in section.index:
                continue
            values = self._confirm_values(runtime, section, state.date, symbol, item, age)
            if not bool(self.evaluator.evaluate(self.confirm_expr, values)):
                continue
            rows.append({
                "symbol": symbol,
                "score": _safe_float(section.at[symbol, self.score_name]),
                "setup_date": item["setup_date"],
                "age": age,
                "watchlist_pnl": values.get("watchlist_pnl"),
                "watchlist_peak_pnl": values.get("watchlist_peak_pnl"),
                "where": True,
            })

        sorted_rows = self._sort_rows(rows)
        selected_rows = sorted_rows[: int(self.topk)] if self.topk is not None else sorted_rows
        if self.remove_on_confirm:
            for row in selected_rows:
                self._watchlist.pop(row["symbol"], None)

        self._record_candidates(state, state.date, rows, selected_rows, len(self._watchlist), expired)
        return StockSelection(signals=[
            Signal(symbol=row["symbol"], score=row["score"], reason=self.reason)
            for row in selected_rows
        ])

    def _add_setups(self, runtime, dates: List[str], current_idx: int) -> None:
        setup_idx = current_idx - self.lag
        if setup_idx < 0:
            return
        setup_date = dates[setup_idx]
        setup_factor_names = self._setup_factor_names(runtime)
        section = runtime.get_cross_section([self.setup_name, self.score_name, *setup_factor_names], setup_date)
        mask = _to_bool_series(section[self.setup_name])
        selected = section.loc[mask]
        if self.sort == "score_desc":
            selected = selected.sort_values(self.score_name, ascending=False)
        elif self.sort == "score_asc":
            selected = selected.sort_values(self.score_name, ascending=True)
        elif self.sort not in {"input_order", "none"}:
            raise ConfigStrategyError(f"Unsupported selector sort: {self.sort}")
        if self.topk is not None:
            selected = selected.head(int(self.topk))
        for symbol, row in selected.iterrows():
            sym = str(symbol)
            if sym not in self._watchlist:
                setup_values = {
                    f"setup_{name}": _safe_float(row[name])
                    for name in setup_factor_names
                    if name in row
                }
                self._watchlist[sym] = {
                    "setup_date": setup_date,
                    "setup_idx": setup_idx,
                    "setup_score": _safe_float(row[self.score_name]),
                    "setup_price": _safe_float(row.get("close")),
                    "setup_values": setup_values,
                }

    def _confirm_factor_names(self, runtime) -> List[str]:
        names = []
        for name in self.confirm_names | {self.score_name}:
            if name in runtime.values:
                names.append(name)
        if "close" in runtime.values:
            names.append("close")
        return sorted(set(names))

    def _setup_factor_names(self, runtime) -> List[str]:
        names = ["close"] if "close" in runtime.values else []
        for name in self.confirm_names:
            if name.startswith("setup_") and name[len("setup_"):] in runtime.values:
                names.append(name[len("setup_"):])
        return sorted(set(names))

    def _confirm_values(self, runtime, section: pd.DataFrame, date: str, symbol: str, item: Dict[str, Any], age: int) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if symbol in section.index:
            values.update(section.loc[symbol].to_dict())
        values.update(item.get("setup_values") or {})

        setup_price = _safe_float(item.get("setup_price"))
        cur_price = _safe_float(values.get("close"))
        peak_pnl = np.nan
        trough_pnl = np.nan
        if setup_price and setup_price > 0 and cur_price is not None:
            setup_idx = int(item["setup_idx"])
            current_idx = runtime.panel.date_index(date)
            sym_idx = runtime.panel.instruments.get_loc(symbol)
            close = np.asarray(runtime.values["close"])[setup_idx: current_idx + 1, sym_idx]
            valid_close = close[~pd.isna(close)]
            if len(valid_close):
                peak_pnl = float(np.nanmax(valid_close) / setup_price - 1)
                trough_pnl = float(np.nanmin(valid_close) / setup_price - 1)
        watchlist_pnl = cur_price / setup_price - 1 if setup_price and setup_price > 0 and cur_price is not None else np.nan
        watchlist_drawdown_from_peak = np.nan
        if not pd.isna(watchlist_pnl) and not pd.isna(peak_pnl):
            peak_price_ratio = 1.0 + peak_pnl
            if peak_price_ratio > 0:
                watchlist_drawdown_from_peak = (1.0 + watchlist_pnl) / peak_price_ratio - 1.0
        values.update({
            "watchlist_age": age,
            "watchlist_pnl": watchlist_pnl,
            "watchlist_peak_pnl": peak_pnl,
            "watchlist_trough_pnl": trough_pnl,
            "watchlist_drawdown_from_peak": watchlist_drawdown_from_peak,
            "setup_score": _safe_float(item.get("setup_score")),
        })
        return values

    def _drop_expired(self, current_idx: int) -> int:
        expired = [
            symbol for symbol, item in self._watchlist.items()
            if current_idx - int(item["setup_idx"]) > self.max_age
        ]
        for symbol in expired:
            self._watchlist.pop(symbol, None)
        return len(expired)

    def _sort_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.sort == "score_desc":
            return sorted(rows, key=lambda row: row["score"], reverse=True)
        if self.sort == "score_asc":
            return sorted(rows, key=lambda row: row["score"])
        if self.sort in {"input_order", "none"}:
            return list(rows)
        raise ConfigStrategyError(f"Unsupported selector sort: {self.sort}")

    def _record_candidates(
        self,
        state: PolicyState,
        signal_date: Optional[str],
        raw_rows: List[Dict[str, Any]],
        selected_rows: List[Dict[str, Any]],
        watchlist_size: int,
        expired_count: int,
    ) -> None:
        context = getattr(state, "context", None)
        recorder = getattr(context, "record_selection_candidates", None) if context is not None else None
        if not callable(recorder):
            return
        detail = {
            "date": state.date,
            "signal_date": signal_date,
            "mode": "watchlist",
            "where": self.setup_expr,
            "confirm": self.confirm_expr,
            "score": self.score_expr,
            "sort": self.sort,
            "topk": self.topk,
            "lag": self.lag,
            "watchlist_max_age": self.max_age,
            "watchlist_min_age": self.min_age,
            "watchlist_size": watchlist_size,
            "expired_count": expired_count,
            "raw_candidate_count": len(raw_rows),
            "selected_count": len(selected_rows),
            "raw_candidates": raw_rows[: self.candidate_limit],
            "selected_candidates": selected_rows[: self.candidate_limit],
            "reason": self.reason,
        }
        recorder(state.date, detail)

    def _date_index(self, dates: List[str], date: str) -> Optional[int]:
        try:
            return dates.index(date)
        except ValueError:
            return None


class EqualWeightRebalance(RebalanceStrategy):
    """Equal-weight rebalance over available slots or portfolio target slots."""

    def __init__(
        self,
        max_positions: int,
        buy_only_new_positions: bool = True,
        max_positions_rules: Sequence[PositionLimitRule | Dict[str, Any]] | None = None,
        rank_weights: Sequence[float] | None = None,
        weight_scope: str = "available_slots",
        exclude_partial_positions_from_slots: bool = False,
    ):
        self.max_positions = int(max_positions)
        self.buy_only_new_positions = bool(buy_only_new_positions)
        self.rank_weights = [float(value) for value in (rank_weights or [])]
        self.weight_scope = str(weight_scope)
        self.exclude_partial_positions_from_slots = bool(exclude_partial_positions_from_slots)
        if any(value < 0 for value in self.rank_weights):
            raise ConfigStrategyError("rebalance.rank_weights must be non-negative")
        if self.weight_scope not in {"available_slots", "portfolio_target"}:
            raise ConfigStrategyError("rebalance.weight_scope must be 'available_slots' or 'portfolio_target'")
        self.max_positions_rules = [
            rule if isinstance(rule, PositionLimitRule) else PositionLimitRule(**rule)
            for rule in (max_positions_rules or [])
        ]
        self.evaluator = ScalarRuleEvaluator()
        self.rule_names = set().union(
            *(self.evaluator.names(rule.when) for rule in self.max_positions_rules)
        ) if self.max_positions_rules else set()
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        if not selection.signals:
            return WeightAllocation(weights={})

        current_holdings = {
            sym for sym, pos in state.positions.items()
            if pos.quantity > 0
        }
        planned_sell_symbols = set(state.extra.get("planned_sell_symbols", set()))
        slot_exempt_symbols: Set[str] = set()
        if self.exclude_partial_positions_from_slots:
            slot_exempt_symbols.update(
                state.extra.get("planned_slot_exempt_symbols", set())
            )
            slot_exempt_symbols.update(
                sym
                for sym, pos in state.positions.items()
                if pos.quantity > 0
                and pos.initial_quantity > 0
                and pos.quantity < pos.initial_quantity
            )
        current_holdings_for_slots = current_holdings - planned_sell_symbols - slot_exempt_symbols
        max_positions = self._max_positions_for_state(state)
        if self.buy_only_new_positions:
            available_slots = max(0, max_positions - len(current_holdings_for_slots))
            if available_slots <= 0:
                return WeightAllocation(weights={})
            selected = [s for s in selection.signals if s.symbol not in current_holdings][:available_slots]
        else:
            selected = []
            new_count = 0
            available_new_slots = max(0, max_positions - len(current_holdings_for_slots))
            for signal in selection.signals:
                if signal.symbol in current_holdings:
                    selected.append(signal)
                    continue
                if new_count < available_new_slots:
                    selected.append(signal)
                    new_count += 1
        if not selected:
            return WeightAllocation(weights={})

        weights = self._weights_for_selected(len(selected), max_positions)
        return WeightAllocation(weights={s.symbol: weights[idx] for idx, s in enumerate(selected)})

    def _weights_for_selected(self, count: int, max_positions: int) -> List[float]:
        if count <= 0:
            return []
        if not self.rank_weights:
            normalized = [1.0 / count] * count
        else:
            raw = [
                self.rank_weights[idx] if idx < len(self.rank_weights) else self.rank_weights[-1]
                for idx in range(count)
            ]
            total = sum(raw)
            normalized = [1.0 / count] * count if total <= 0 else [value / total for value in raw]
        if self.weight_scope == "available_slots":
            return normalized
        portfolio_budget = min(1.0, float(count) / float(max(1, max_positions)))
        return [portfolio_budget * value for value in normalized]

    def _max_positions_for_state(self, state: PolicyState) -> int:
        limit = self.max_positions
        if not self.max_positions_rules:
            return limit
        values = self._rule_values(state)
        for rule in self.max_positions_rules:
            if bool(self.evaluator.evaluate(rule.when, values)):
                limit = int(rule.value)
                break
        return limit

    def _rule_values(self, state: PolicyState) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            for key in state.market_data.columns:
                clean_key = str(key).lstrip("$")
                values[clean_key] = _safe_float(pd.to_numeric(state.market_data[key], errors="coerce").mean())

        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        factor_names = [name for name in self.rule_names if runtime is not None and name in runtime.values]
        if factor_names:
            factors = self._factor_section(runtime, state.date, factor_names)
            for name in factor_names:
                values[name] = _first_numeric_value(factors[name])

        values.update({
            "cash": state.account.cash if state.account is not None else 0.0,
            "total_value": state.account.total_value if state.account is not None else 0.0,
            "account_drawdown": state.account.drawdown if state.account is not None else 0.0,
        })
        return values

    def _factor_section(self, runtime, date: str, names: Iterable[str]) -> pd.DataFrame:
        key = f"{date}|{','.join(sorted(names))}"
        if key not in self._factor_cache:
            self._factor_cache[key] = runtime.get_cross_section(sorted(names), date)
        return self._factor_cache[key]


class RuleExecution(ExecutionStrategy):
    """Execution policy driven by stateful sell rules and configurable buy sizing."""

    def __init__(
        self,
        sell_rules: Sequence[SellRule | Dict[str, Any]],
        cost: TransactionCost,
        deal_price: str = "open",
        buy_deal_price: str | None = None,
        sell_deal_price: str | None = None,
        cash_use_ratio: float = 0.99,
        sizing: str = "cash_equal",
        lot_size: int = 100,
        skip_if_holding: bool = True,
        skip_limit_up: bool = False,
        reuse_sell_cash: bool = False,
        cash_use_ratio_rules: Sequence[CashUseRatioRule | Dict[str, Any]] | None = None,
        max_position_weight: float | None = None,
        trim_overweight_positions: bool = False,
        buy_when: str | None = None,
        add_existing_when: str | None = None,
    ):
        self.sell_rules = [
            rule if isinstance(rule, SellRule) else SellRule(**rule)
            for rule in sell_rules
        ]
        self.cost = cost
        self.deal_price = deal_price
        self.buy_deal_price = buy_deal_price or deal_price
        self.sell_deal_price = sell_deal_price or deal_price
        self.cash_use_ratio = float(cash_use_ratio)
        self.sizing = sizing
        self.lot_size = int(lot_size)
        self.skip_if_holding = bool(skip_if_holding)
        self.skip_limit_up = bool(skip_limit_up)
        self.reuse_sell_cash = bool(reuse_sell_cash)
        self.max_position_weight = (
            float(max_position_weight) if max_position_weight is not None else None
        )
        if self.max_position_weight is not None and not 0.0 < self.max_position_weight <= 1.0:
            raise ConfigStrategyError("execution.max_position_weight must be within (0, 1]")
        self.trim_overweight_positions = bool(trim_overweight_positions)
        self.cash_use_ratio_rules = [
            rule if isinstance(rule, CashUseRatioRule) else CashUseRatioRule(**rule)
            for rule in (cash_use_ratio_rules or [])
        ]
        self.buy_when = str(buy_when) if buy_when else ""
        self.add_existing_when = str(add_existing_when) if add_existing_when else ""
        self.evaluator = ScalarRuleEvaluator()
        sell_rule_names = set().union(*(self.evaluator.names(rule.when) for rule in self.sell_rules)) if self.sell_rules else set()
        buy_rule_names = self.evaluator.names(self.buy_when) if self.buy_when else set()
        add_rule_names = self.evaluator.names(self.add_existing_when) if self.add_existing_when else set()
        self.cash_use_rule_names = set().union(
            *(self.evaluator.names(rule.when) for rule in self.cash_use_ratio_rules)
        ) if self.cash_use_ratio_rules else set()
        self.rule_names = sell_rule_names | buy_rule_names | add_rule_names | self.cash_use_rule_names
        self.buy_rule_names = buy_rule_names | add_rule_names
        self.entry_snapshot_names = {
            name[len("entry_"):]
            for name in self.rule_names
            if name.startswith("entry_") and len(name) > len("entry_")
        }
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def prepare(self, context):
        prices = {
            "buy": self.buy_deal_price,
            "sell": self.sell_deal_price,
        }
        exchange = getattr(context, "exchange", None)
        quote = getattr(exchange, "quote", None) if exchange is not None else None
        runtime = getattr(context, "factor_runtime", None)
        if quote is None:
            return
        for price_name in prices.values():
            field = f"${price_name.lstrip('$')}"
            if field in quote.columns:
                continue
            if runtime is None or price_name.lstrip("$") not in runtime.values:
                raise ConfigStrategyError(
                    f"execution price is not available as quote field or computed formula: {price_name}"
                )
            matrix = np.asarray(runtime.values[price_name.lstrip("$")])
            if matrix.ndim != 2:
                raise ConfigStrategyError(
                    f"execution price formula must produce a [date, symbol] matrix: {price_name}"
                )
            if matrix.shape != (len(runtime.panel.dates), len(runtime.panel.instruments)):
                raise ConfigStrategyError(
                    f"execution price formula shape mismatch: {price_name}"
                )

            wide = pd.DataFrame(matrix, index=runtime.panel.dates, columns=runtime.panel.instruments)
            series = wide.stack(future_stack=True)
            series.index = series.index.set_names(["datetime", "instrument"])
            quote[field] = series.reindex(quote.index).astype(float)

    def act(self, state: PolicyState, allocation: WeightAllocation) -> OrderList:
        orders: List[Order] = []
        date = state.date
        position_plan = state.extra.get("position_plan")
        planned_symbols = set(getattr(position_plan, "sell_ratios", {}) or {})
        plan_metadata = dict(getattr(position_plan, "metadata", {}) or {})
        plan_sell_reasons = dict(plan_metadata.get("sell_reasons") or {})
        plan_entry_reasons = dict(plan_metadata.get("entry_reasons") or {})
        plan_entry_contexts = dict(plan_metadata.get("entry_contexts") or {})

        if position_plan is not None:
            for sym, raw_ratio in dict(position_plan.sell_ratios).items():
                pos = state.positions.get(sym)
                price = self._get_price(state, sym, OrderAction.SELL)
                if pos is None or price is None or price <= 0:
                    continue
                ratio = min(1.0, max(0.0, float(raw_ratio)))
                quantity = (int(pos.quantity * ratio) // self.lot_size) * self.lot_size
                if quantity <= 0:
                    continue
                orders.append(Order(
                    symbol=sym,
                    action=OrderAction.SELL,
                    price=price,
                    quantity=quantity,
                    date=date,
                    reason=str(plan_sell_reasons.get(sym, "position_manager_sell")),
                ))

        if position_plan is None or not bool(getattr(position_plan, "replace_sell_rules", True)):
            for sym, pos, cur_price, reason, quantity, is_full_exit in self._iter_sell_decisions(state):
                if sym in planned_symbols or not reason:
                    continue
                orders.append(Order(
                    symbol=sym,
                    action=OrderAction.SELL,
                    price=cur_price,
                    quantity=quantity,
                    date=date,
                    reason=reason,
                ))

        full_exit_symbols = {
            order.symbol for order in orders
            if order.action == OrderAction.SELL and self._is_full_exit_order(state, order)
        }
        current_holdings = {
            sym for sym, pos in state.positions.items()
            if pos.quantity > 0 and sym not in full_exit_symbols
        }
        buyable = [
            sym for sym in allocation.weights
            if not self.skip_if_holding or sym not in current_holdings
        ]
        add_existing_symbols: Set[str] = set()
        for sym in current_holdings:
            if sym in buyable or not self.add_existing_when:
                continue
            price = self._get_price(state, sym, OrderAction.BUY)
            if price is None or price <= 0:
                continue
            if self._add_existing_rule_passes(state, sym, price):
                buyable.append(sym)
                add_existing_symbols.add(sym)
        if not buyable:
            return OrderList(orders=orders)
        if self.sizing not in {"cash_equal", "target_weight"}:
            raise ConfigStrategyError(f"Unsupported buy sizing: {self.sizing}")
        if state.account is None:
            return OrderList(orders=orders)

        cash_base = state.account.cash
        if self.reuse_sell_cash:
            cash_base += sum(
                self._estimate_sell_proceeds(order.symbol, order.price, order.quantity)
                for order in orders
                if order.action == OrderAction.SELL
            )
        cash_use_ratio = (
            min(1.0, max(0.0, float(position_plan.cash_deploy_ratio)))
            if position_plan is not None
            else self._cash_use_ratio_for_state(state)
        )
        raw_weights = {sym: max(0.0, float(allocation.weights.get(sym, 0.0))) for sym in buyable}
        for sym in buyable:
            if sym not in allocation.weights:
                raw_weights[sym] = 1.0
        total_weight = sum(raw_weights.values())
        if total_weight <= 0 and self.sizing == "cash_equal":
            raw_weights = {sym: 1.0 for sym in buyable}
            total_weight = float(len(buyable))

        for sym in buyable:
            price = self._get_price(state, sym, OrderAction.BUY)
            if price is None or price <= 0:
                continue
            if sym not in add_existing_symbols and self.buy_when and not self._buy_rule_passes(state, sym, price):
                continue
            if self.skip_limit_up and self._is_limit_up(state, sym):
                continue
            if self.sizing == "cash_equal":
                cash_per_stock = cash_base * cash_use_ratio * raw_weights[sym] / total_weight
            else:
                target_weight = raw_weights[sym]
                if target_weight <= 0:
                    continue
                target_value = state.account.total_value * cash_use_ratio * target_weight
                if self.max_position_weight is not None:
                    target_value = min(
                        target_value,
                        state.account.total_value * self.max_position_weight,
                    )
                current_position = state.positions.get(sym)
                current_value = (
                    current_position.quantity * price
                    if current_position is not None and current_position.quantity > 0
                    else 0.0
                )
                cash_per_stock = max(0.0, target_value - current_value)
            qty = int(cash_per_stock / price)
            qty = (qty // self.lot_size) * self.lot_size
            if qty > 0:
                entry_context = self._entry_context(state, sym)
                extra_context = plan_entry_contexts.get(sym) if position_plan is not None else None
                if isinstance(extra_context, dict):
                    entry_context.update(extra_context)
                orders.append(Order(
                    symbol=sym,
                    action=OrderAction.BUY,
                    price=price,
                    quantity=qty,
                    date=date,
                    reason=str(plan_entry_reasons.get(sym, "config_buy")),
                    context=entry_context,
                ))

        return OrderList(orders=orders)

    def _buy_rule_passes(self, state: PolicyState, symbol: str, price: float) -> bool:
        try:
            return bool(self.evaluator.evaluate(self.buy_when, self._buy_rule_values(state, symbol, price)))
        except Exception:
            return False

    def _cash_use_ratio_for_state(self, state: PolicyState) -> float:
        values = self._cash_use_rule_values(state)
        for rule in self.cash_use_ratio_rules:
            if bool(self.evaluator.evaluate(rule.when, values)):
                return float(rule.value)
        return self.cash_use_ratio

    def _cash_use_rule_values(self, state: PolicyState) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            for key in state.market_data.columns:
                clean_key = str(key).lstrip("$")
                values[clean_key] = _safe_float(pd.to_numeric(state.market_data[key], errors="coerce").mean())

        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        factor_names = [name for name in self.cash_use_rule_names if runtime is not None and name in runtime.values]
        if factor_names:
            factors = self._factor_section(runtime, state.date, factor_names)
            for name in factor_names:
                values[name] = _first_numeric_value(factors[name])

        values.update({
            "cash": state.account.cash if state.account is not None else 0.0,
            "total_value": state.account.total_value if state.account is not None else 0.0,
            "account_drawdown": state.account.drawdown if state.account is not None else 0.0,
        })
        return values

    def _add_existing_rule_passes(self, state: PolicyState, symbol: str, price: float) -> bool:
        try:
            return bool(self.evaluator.evaluate(self.add_existing_when, self._buy_rule_values(state, symbol, price)))
        except Exception:
            return False

    def _buy_rule_values(self, state: PolicyState, symbol: str, price: float) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            try:
                row = state.market_data.loc[symbol]
                for key, value in row.items():
                    values[str(key).lstrip("$")] = value
            except Exception:
                pass

        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        factor_names = [name for name in self.buy_rule_names if runtime is not None and name in runtime.values]
        if factor_names:
            factors = self._factor_section(runtime, state.date, factor_names)
            if symbol in factors.index:
                values.update(factors.loc[symbol].to_dict())

        pos = state.positions.get(symbol)
        is_holding = bool(pos is not None and pos.quantity > 0)
        if is_holding:
            peak_price = pos.highest_price if pos.highest_price > 0 else price
            trough_price = pos.lowest_price if pos.lowest_price > 0 else price
            avg_cost = pos.avg_cost
            pnl_pct = price / avg_cost - 1 if avg_cost > 0 else 0.0
            peak_pnl_pct = peak_price / avg_cost - 1 if avg_cost > 0 else 0.0
            trough_pnl_pct = trough_price / avg_cost - 1 if avg_cost > 0 else 0.0
            drawdown_from_peak = price / peak_price - 1 if peak_price > 0 else 0.0
            position_values = {
                "is_holding": True,
                "cost_price": avg_cost,
                "avg_cost": avg_cost,
                "holding_days": pos.holding_days,
                "position_qty": pos.quantity,
                "initial_position_qty": pos.initial_quantity or pos.quantity,
                "remaining_position_pct": pos.quantity / (pos.initial_quantity or pos.quantity),
                "position_market_value": pos.market_value,
                "position_weight": pos.weight,
                "pnl_pct": pnl_pct,
                "peak_price": peak_price,
                "trough_price": trough_price,
                "peak_pnl_pct": peak_pnl_pct,
                "trough_pnl_pct": trough_pnl_pct,
                "drawdown_from_peak": drawdown_from_peak,
            }
        else:
            position_values = {
                "is_holding": False,
                "cost_price": 0.0,
                "avg_cost": 0.0,
                "holding_days": 0,
                "position_qty": 0,
                "initial_position_qty": 0,
                "remaining_position_pct": 0.0,
                "position_market_value": 0.0,
                "position_weight": 0.0,
                "pnl_pct": 0.0,
                "peak_price": price,
                "trough_price": price,
                "peak_pnl_pct": 0.0,
                "trough_pnl_pct": 0.0,
                "drawdown_from_peak": 0.0,
            }

        values.update({
            "price": price,
            "cash": state.account.cash if state.account is not None else 0.0,
            "total_value": state.account.total_value if state.account is not None else 0.0,
            "account_drawdown": state.account.drawdown if state.account is not None else 0.0,
        })
        values.update(self._limit_state_values(state, symbol))
        values.update(self._position_context_values(pos))
        values.update(position_values)
        return values

    def preview_sell_symbols(self, state: PolicyState) -> List[str]:
        return [
            sym for sym, _, _, reason, _, is_full_exit in self._iter_sell_decisions(state)
            if reason and is_full_exit
        ]

    def preview_slot_exempt_symbols(self, state: PolicyState) -> List[str]:
        return [
            sym for sym, _, _, reason, _, is_full_exit in self._iter_sell_decisions(state)
            if reason and not is_full_exit
        ]

    def _iter_sell_decisions(self, state: PolicyState):
        for sym, pos in list(state.positions.items()):
            if pos.quantity <= 0:
                continue
            cur_price = self._get_price(state, sym, OrderAction.SELL)
            if cur_price is None or cur_price <= 0 or pos.avg_cost <= 0:
                continue

            values = self._rule_values(state, sym, pos, cur_price)
            reason = None
            quantity = 0
            is_full_exit = False
            for rule in self.sell_rules:
                if rule.action not in {"sell_all", "sell_to_position_pct"}:
                    raise ConfigStrategyError(f"Unsupported sell rule action: {rule.action}")
                if bool(self.evaluator.evaluate(rule.when, values)):
                    quantity = self._sell_quantity_for_rule(rule, pos)
                    if quantity <= 0:
                        continue
                    reason = rule.name
                    is_full_exit = quantity >= pos.quantity
                    break
            if reason is None and self.trim_overweight_positions:
                quantity = self._position_weight_cap_quantity(state, pos, cur_price)
                if quantity > 0:
                    reason = "position_weight_cap"
                    is_full_exit = quantity >= pos.quantity
            yield sym, pos, cur_price, reason, quantity, is_full_exit

    def _position_weight_cap_quantity(self, state: PolicyState, pos, price: float) -> int:
        if self.max_position_weight is None or state.account is None or state.account.total_value <= 0:
            return 0
        cap_value = state.account.total_value * self.max_position_weight
        current_value = pos.quantity * price
        if current_value <= cap_value:
            return 0
        target_quantity = (int(cap_value / price) // self.lot_size) * self.lot_size
        return max(0, ((pos.quantity - target_quantity) // self.lot_size) * self.lot_size)

    def _rule_values(self, state: PolicyState, symbol: str, pos, cur_price: float) -> Dict[str, Any]:
        values: Dict[str, Any] = {}

        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            try:
                row = state.market_data.loc[symbol]
                for key, value in row.items():
                    clean_key = str(key).lstrip("$")
                    values[clean_key] = value
            except Exception:
                pass

        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        factor_names = [name for name in self.rule_names if runtime is not None and name in runtime.values]
        if factor_names:
            factors = self._factor_section(runtime, state.date, factor_names)
            if symbol in factors.index:
                values.update(factors.loc[symbol].to_dict())

        total_value = state.account.total_value if state.account is not None else 0.0
        initial_quantity = pos.initial_quantity or pos.quantity
        values.update({
            "price": cur_price,
            "cost_price": pos.avg_cost,
            "avg_cost": pos.avg_cost,
            "holding_days": pos.holding_days,
            "position_qty": pos.quantity,
            "initial_position_qty": initial_quantity,
            "remaining_position_pct": pos.quantity / initial_quantity if initial_quantity > 0 else 0.0,
            "position_market_value": pos.market_value,
            "position_weight": pos.weight,
            "pnl_pct": cur_price / pos.avg_cost - 1,
            "peak_price": pos.highest_price if pos.highest_price > 0 else cur_price,
            "trough_price": pos.lowest_price if pos.lowest_price > 0 else cur_price,
            "peak_pnl_pct": (pos.highest_price if pos.highest_price > 0 else cur_price) / pos.avg_cost - 1,
            "trough_pnl_pct": (pos.lowest_price if pos.lowest_price > 0 else cur_price) / pos.avg_cost - 1,
            "drawdown_from_peak": cur_price / (pos.highest_price if pos.highest_price > 0 else cur_price) - 1,
            "cash": state.account.cash if state.account is not None else 0.0,
            "total_value": total_value,
            "account_drawdown": state.account.drawdown if state.account is not None else 0.0,
        })
        values.update(self._limit_state_values(state, symbol))
        values.update(self._position_context_values(pos))
        return values

    @staticmethod
    def _limit_state_values(state: PolicyState, symbol: str) -> Dict[str, bool]:
        exchange = getattr(getattr(state, "context", None), "exchange", None)
        if exchange is None:
            return {"limit_up": False, "one_side_limit_up": False}
        try:
            limit_up = bool(exchange.is_limit_up(symbol, state.date))
        except Exception:
            limit_up = False
        try:
            one_side_limit_up = bool(exchange.is_one_side_limit_up(symbol, state.date))
        except Exception:
            one_side_limit_up = False
        return {
            "limit_up": limit_up,
            "one_side_limit_up": one_side_limit_up,
        }

    def _entry_context(self, state: PolicyState, symbol: str) -> Dict[str, Any]:
        if not self.entry_snapshot_names:
            return {}
        values: Dict[str, Any] = {}
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            try:
                row = state.market_data.loc[symbol]
                for key, value in row.items():
                    clean_key = str(key).lstrip("$")
                    if clean_key in self.entry_snapshot_names:
                        values[f"entry_{clean_key}"] = value
            except Exception:
                pass

        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        factor_names = [name for name in self.entry_snapshot_names if runtime is not None and name in runtime.values]
        if factor_names:
            factors = self._factor_section(runtime, state.date, factor_names)
            if symbol in factors.index:
                for key, value in factors.loc[symbol].to_dict().items():
                    values[f"entry_{key}"] = value
        return values

    def _position_context_values(self, pos) -> Dict[str, Any]:
        context = dict(getattr(pos, "context", {}) or {})
        for name in HOLD_CONTEXT_RULE_NAMES:
            context.setdefault(name, np.nan)
        return context

    def _sell_quantity_for_rule(self, rule: SellRule, pos) -> int:
        if rule.action == "sell_all":
            return int(pos.quantity)
        if rule.action == "sell_to_position_pct":
            initial_quantity = int(pos.initial_quantity or pos.quantity)
            target_quantity = initial_quantity * float(rule.position_pct)
            raw_quantity = max(0, int(pos.quantity - target_quantity))
            return (raw_quantity // self.lot_size) * self.lot_size
        raise ConfigStrategyError(f"Unsupported sell rule action: {rule.action}")

    def _is_full_exit_order(self, state: PolicyState, order: Order) -> bool:
        pos = state.positions.get(order.symbol)
        return bool(pos is not None and order.quantity >= pos.quantity)

    def _factor_section(self, runtime, date: str, names: Iterable[str]) -> pd.DataFrame:
        key = f"{date}|{','.join(sorted(names))}"
        if key not in self._factor_cache:
            self._factor_cache[key] = runtime.get_cross_section(sorted(names), date)
        return self._factor_cache[key]

    def _get_price(
        self,
        state: PolicyState,
        symbol: str,
        action: OrderAction = OrderAction.BUY,
    ) -> Optional[float]:
        if state.context is None or state.context.exchange.quote is None:
            return None
        try:
            deal_price = self.buy_deal_price if action == OrderAction.BUY else self.sell_deal_price
            field = f"${deal_price.lstrip('$')}"
            price = state.context.exchange.quote.loc[(pd.Timestamp(state.date), symbol), field]
            return float(price) if not pd.isna(price) else None
        except Exception:
            return None

    def _is_limit_up(self, state: PolicyState, symbol: str) -> bool:
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            try:
                row = state.market_data.loc[symbol]
                if "limit_up" in row:
                    return bool(row["limit_up"] == 1)
            except Exception:
                pass
        return bool(self._limit_state_values(state, symbol)["limit_up"])

    def _estimate_sell_proceeds(self, symbol: str, price: float, quantity: int) -> float:
        return price * quantity - self.cost.calculate_sell_cost(symbol, price, quantity)


def build_formula_strategy(config: Dict[str, Any], cost: TransactionCost) -> CompositeStrategy:
    compile_strategy_config(config)
    factors = config.get("factors") or {}
    group_factors = config.get("group_factors") or {}
    signals = config.get("signals") or {}
    formulas = {**factors, **group_factors, **signals}
    groups = _normalize_group_config(config.get("groups") or {})

    selector_cfg = config.get("selector") or {}
    rebalance_cfg = config.get("rebalance") or {}
    execution_cfg = config.get("execution") or {}
    buy_cfg = execution_cfg.get("buy") or {}
    if not selector_cfg.get("where"):
        raise ConfigStrategyError("selector.where is required")
    if rebalance_cfg.get("type", "equal_weight") != "equal_weight":
        raise ConfigStrategyError(f"Unsupported rebalance.type: {rebalance_cfg.get('type')}")

    selector_mode = selector_cfg.get("mode", "precomputed")
    if selector_mode == "watchlist":
        watchlist_cfg = selector_cfg.get("watchlist") or {}
        selector = WatchlistFormulaSelector(
            formulas=formulas,
            setup=selector_cfg.get("where"),
            confirm=selector_cfg.get("confirm"),
            score=selector_cfg.get("score", 0),
            lag=selector_cfg.get("lag", 1),
            max_age=watchlist_cfg.get("max_age", 12),
            min_age=watchlist_cfg.get("min_age", 1),
            sort=selector_cfg.get("sort", "input_order"),
            topk=selector_cfg.get("topk"),
            reason=selector_cfg.get("reason", "watchlist_buy"),
            candidate_limit=selector_cfg.get("candidate_limit", 20),
            remove_on_confirm=watchlist_cfg.get("remove_on_confirm", True),
            groups=groups,
        )
    else:
        selector_cls = GroupAwareFormulaSelector if groups else FormulaSelector
        selector = selector_cls(
            formulas=formulas,
            where=selector_cfg.get("where"),
            score=selector_cfg.get("score", 0),
            lag=selector_cfg.get("lag", 1),
            wrap_first_signal=selector_cfg.get("wrap_first_signal", False),
            sort=selector_cfg.get("sort", "input_order"),
            topk=selector_cfg.get("topk"),
            reason=selector_cfg.get("reason", "config_buy"),
            candidate_limit=selector_cfg.get("candidate_limit", 20),
            external_score=selector_cfg.get("external_score"),
            **({"groups": groups} if groups else {}),
        )
    rebalance = EqualWeightRebalance(
        max_positions=rebalance_cfg.get("max_positions", 10),
        buy_only_new_positions=rebalance_cfg.get("buy_only_new_positions", True),
        max_positions_rules=rebalance_cfg.get("max_positions_rules") or [],
        rank_weights=rebalance_cfg.get("rank_weights") or [],
        weight_scope=rebalance_cfg.get("weight_scope", "available_slots"),
        exclude_partial_positions_from_slots=rebalance_cfg.get(
            "exclude_partial_positions_from_slots", False
        ),
    )
    execution = RuleExecution(
        sell_rules=execution_cfg.get("sell_rules") or [],
        cost=cost,
        deal_price=execution_cfg.get("deal_price", "open"),
        buy_deal_price=execution_cfg.get("buy_deal_price"),
        sell_deal_price=execution_cfg.get("sell_deal_price"),
        cash_use_ratio=execution_cfg.get("cash_use_ratio", rebalance_cfg.get("cash_use_ratio", 0.99)),
        sizing=buy_cfg.get("sizing", "cash_equal"),
        lot_size=buy_cfg.get("lot_size", 100),
        skip_if_holding=buy_cfg.get("skip_if_holding", True),
        skip_limit_up=buy_cfg.get("skip_limit_up", False),
        reuse_sell_cash=buy_cfg.get("reuse_sell_cash", False),
        cash_use_ratio_rules=rebalance_cfg.get("cash_use_ratio_rules") or [],
        max_position_weight=execution_cfg.get("max_position_weight"),
        trim_overweight_positions=execution_cfg.get("trim_overweight_positions", False),
        buy_when=buy_cfg.get("when"),
        add_existing_when=buy_cfg.get("add_existing_when"),
    )
    position_manager = None
    if execution_cfg.get("position_manager"):
        from .position_manager_loader import build_position_manager

        try:
            position_manager = build_position_manager(execution_cfg.get("position_manager"))
        except Exception as exc:
            raise ConfigStrategyError(f"Failed to build execution.position_manager: {exc}") from exc
    return CompositeStrategy(
        selector=selector,
        rebalance=rebalance,
        execution=execution,
        position_manager=position_manager,
        precompute_stock_signals=(selector_mode != "watchlist"),
    )


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric != 0


def _normalize_external_score_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception as exc:
        raise ConfigStrategyError(f"Invalid selector.external_score date: {value}") from exc


def _validate_external_score_manifest(path: Path, config: Dict[str, Any]) -> None:
    if not bool(config.get("require_artifact_manifest", False)):
        return
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists():
        raise ConfigStrategyError(f"External score artifact manifest is required: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigStrategyError(f"Invalid external score artifact manifest: {manifest_path}") from exc
    if manifest.get("kind") != "quantx_market_score_artifact_v1":
        raise ConfigStrategyError(f"Unsupported external score artifact kind: {manifest.get('kind')}")
    score_col = str(config.get("score_col", "score"))
    if score_col not in set(manifest.get("score_columns") or []):
        raise ConfigStrategyError(f"External score manifest does not declare score_col={score_col!r}")
    if int(manifest.get("rows", 0)) <= 0:
        raise ConfigStrategyError("External score manifest reports no rows")


def _load_external_score_series(config: Dict[str, Any]) -> pd.Series:
    path = Path(str(config.get("path") or "")).expanduser()
    if not path.exists():
        raise ConfigStrategyError(f"Missing selector.external_score.path: {path}")
    _validate_external_score_manifest(path, config)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ConfigStrategyError(f"Unsupported selector.external_score file type: {path.suffix}")

    date_col = str(config.get("date_col", "signal_date"))
    instrument_col = str(config.get("instrument_col", "instrument"))
    score_col = str(config.get("score_col", "score"))
    missing_cols = [col for col in (date_col, instrument_col, score_col) if col not in frame.columns]
    if missing_cols:
        raise ConfigStrategyError(f"External score file missing columns: {missing_cols}")

    out = pd.DataFrame({
        "date": pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
        "instrument": frame[instrument_col].astype(str),
        "score": pd.to_numeric(frame[score_col], errors="coerce"),
    }).dropna(subset=["date", "instrument", "score"])
    if out.empty:
        raise ConfigStrategyError("External score file has no usable rows")
    out = out.drop_duplicates(["date", "instrument"], keep="last")
    return pd.Series(
        out["score"].to_numpy(dtype=float),
        index=pd.MultiIndex.from_frame(out[["date", "instrument"]]),
        name="external_score",
    )


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def _first_numeric_value(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return _safe_float(numeric.iloc[0])


def _inject_runtime_groups(runtime, groups: Dict[str, Dict[str, Any]]) -> None:
    if not groups:
        return
    instruments = [str(sym) for sym in runtime.panel.instruments]
    for name, cfg in groups.items():
        source = str(cfg.get("source") or "")
        if source in {"meta.industry_l1", "industry_l1"}:
            runtime.values[name] = _load_industry_group(instruments, cfg)
        elif source in {"meta.concept", "concept"}:
            runtime.values[name] = _load_concept_membership(instruments, cfg)
        else:
            raise ConfigStrategyError(f"Unsupported groups.{name}.source: {source}")


def _load_industry_group(instruments: Sequence[str], cfg: Dict[str, Any]) -> np.ndarray:
    path = Path(cfg.get("path") or DEFAULT_INDUSTRY_CSV)
    if not path.exists():
        raise ConfigStrategyError(f"Industry group CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _first_existing_column(frame, ["symbol", "股票代码", "code"])
    group_col = _first_existing_column(frame, ["industry_code", "industry_name", "板块代码", "板块名称", "industry"])
    mapping = {
        _normalize_symbol_for_group(row[symbol_col]): str(row[group_col])
        for _, row in frame.iterrows()
        if pd.notna(row.get(symbol_col)) and pd.notna(row.get(group_col))
    }
    labels = [mapping.get(_normalize_symbol_for_group(sym)) for sym in instruments]
    return _encode_labels(labels)


def _load_concept_membership(instruments: Sequence[str], cfg: Dict[str, Any]) -> np.ndarray:
    path = Path(cfg.get("path") or DEFAULT_SECTOR_CSV)
    if not path.exists():
        raise ConfigStrategyError(f"Concept group CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str)
    if "sector_type" in frame.columns:
        frame = frame[frame["sector_type"].fillna("concept") == str(cfg.get("sector_type", "concept"))]
    symbol_col = _first_existing_column(frame, ["symbol", "股票代码", "code"])
    group_col = _first_existing_column(frame, ["sector_code", "sector_name", "板块代码", "板块名称", "concept"])
    if frame.empty:
        return np.zeros((len(instruments), 0), dtype=bool)
    by_symbol: Dict[str, Set[str]] = {}
    group_names: Set[str] = set()
    for _, row in frame.iterrows():
        if pd.isna(row.get(symbol_col)) or pd.isna(row.get(group_col)):
            continue
        symbol = _normalize_symbol_for_group(row[symbol_col])
        group = str(row[group_col])
        by_symbol.setdefault(symbol, set()).add(group)
        group_names.add(group)
    ordered_groups = sorted(group_names)
    group_index = {group: idx for idx, group in enumerate(ordered_groups)}
    out = np.zeros((len(instruments), len(ordered_groups)), dtype=bool)
    for inst_idx, symbol in enumerate(instruments):
        for group in by_symbol.get(_normalize_symbol_for_group(symbol), set()):
            out[inst_idx, group_index[group]] = True
    return out


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ConfigStrategyError(f"CSV is missing required columns, expected one of: {list(names)}")


def _encode_labels(labels: Sequence[Optional[str]]) -> np.ndarray:
    clean = [str(label) if label not in {None, "", "nan"} and not pd.isna(label) else None for label in labels]
    mapping = {label: idx for idx, label in enumerate(sorted({label for label in clean if label is not None}))}
    return np.asarray([mapping.get(label, -1) if label is not None else -1 for label in clean], dtype=np.int32)


def _normalize_symbol_for_group(value: Any) -> str:
    text = str(value).strip()
    if "." in text:
        left, right = text.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{left.upper()}{right}"
        if right.upper() in {"SH", "SZ", "BJ"}:
            return f"{right.upper()}{left}"
    upper = text.upper()
    if upper.startswith(("SH", "SZ", "BJ")):
        return upper
    if len(upper) == 6 and upper.isdigit():
        if upper.startswith("6"):
            return f"SH{upper}"
        if upper.startswith(("0", "3")):
            return f"SZ{upper}"
        if upper.startswith(("4", "8", "9")):
            return f"BJ{upper}"
    return upper
