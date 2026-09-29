"""Formula runtime for dynamic full-market factor computation."""

from __future__ import annotations

import ast
import operator as py_operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd

from .operators import DEFAULT_REGISTRY, OperatorRegistry
from .panel import MarketPanel


class FormulaError(ValueError):
    """Raised when a formula is invalid or unsafe."""


_PARAM_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class FactorRuntime:
    panel: MarketPanel
    registry: OperatorRegistry = DEFAULT_REGISTRY
    values: Dict[str, Any] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    profiles: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.values.update(self.panel.fields)

    def compute_formulas(self, formulas: Dict[str, str], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = params or {}
        expanded = {name: self._expand_params(expr, params) for name, expr in formulas.items()}
        order = self._plan(expanded)
        for name in order:
            self.values[name] = self._eval_formula(expanded[name])
            self.formulas[name] = expanded[name]
        return {name: self.values[name] for name in expanded}

    def get_matrix(self, factor_name: str):
        return self.values[factor_name]

    def get_cross_section(self, factor_names: str | Iterable[str], date) -> pd.DataFrame:
        names = [factor_names] if isinstance(factor_names, str) else list(factor_names)
        return self.panel.cross_section({name: self.values[name] for name in names}, date)

    def explain(self, factor_name: str) -> List[str]:
        if factor_name not in self.formulas:
            raise KeyError(factor_name)
        lines: List[str] = []
        seen: Set[str] = set()

        def walk(name: str, depth: int):
            if name in seen:
                lines.append(f"{'  ' * depth}{name} (cached)")
                return
            seen.add(name)
            expr = self.formulas.get(name)
            if expr is None:
                lines.append(f"{'  ' * depth}{name} [input]")
                return
            lines.append(f"{'  ' * depth}{name} = {expr}")
            for dep in sorted(self._names_in_expr(expr) & set(self.formulas)):
                walk(dep, depth + 1)

        walk(factor_name, 0)
        return lines

    def profile(self) -> Dict[str, float]:
        return dict(self.profiles)

    def _expand_params(self, expr: str, params: Dict[str, Any]) -> str:
        def repl(match):
            key = match.group(1)
            if key not in params:
                raise FormulaError(f"Missing formula parameter: {key}")
            return repr(params[key])
        return _PARAM_RE.sub(repl, expr)

    def _plan(self, formulas: Dict[str, str]) -> List[str]:
        pending = dict(formulas)
        resolved = set(self.values)
        order: List[str] = []
        while pending:
            progressed = False
            for name, expr in list(pending.items()):
                deps = self._names_in_expr(expr) - self._operator_names()
                unknown = deps - resolved - set(pending)
                if unknown:
                    raise FormulaError(f"Formula {name} references unknown variables: {sorted(unknown)}")
                if deps <= resolved:
                    order.append(name)
                    resolved.add(name)
                    del pending[name]
                    progressed = True
            if not progressed:
                raise FormulaError(f"Formula dependency cycle: {sorted(pending)}")
        return order

    def _eval_formula(self, expr: str):
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(str(exc)) from exc
        self._validate_ast(tree)
        return self._eval_node(tree.body)

    def _names_in_expr(self, expr: str) -> Set[str]:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(str(exc)) from exc
        self._validate_ast(tree)
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _operator_names(self) -> Set[str]:
        return set(self.registry.names())

    def _validate_ast(self, tree: ast.AST) -> None:
        forbidden = (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
        for node in ast.walk(tree):
            if isinstance(node, forbidden):
                raise FormulaError(f"Unsupported formula syntax: {type(node).__name__}")
            if isinstance(node, (ast.BitAnd, ast.BitOr, ast.Invert)):
                raise FormulaError("Use and/or/not or And()/Or()/Not() instead of &, |, or ~")

    def _eval_node(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.values:
                return self.values[node.id]
            raise FormulaError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise FormulaError(f"Unsupported binary operator: {type(node.op).__name__}")
            left, right = _broadcast_pair(self._eval_node(node.left), self._eval_node(node.right))
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise FormulaError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise FormulaError(f"Unsupported boolean operator: {type(node.op).__name__}")
            values = [np.asarray(self._eval_node(v), dtype=bool) for v in node.values]
            op = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = values[0]
            for value in values[1:]:
                out, value = _broadcast_pair(out, value)
                out = op(out, value)
            return out
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            out = None
            for op, comp in zip(node.ops, node.comparators):
                cmp_op = _CMP_OPS.get(type(op))
                if cmp_op is None:
                    raise FormulaError(f"Unsupported comparison operator: {type(op).__name__}")
                right = self._eval_node(comp)
                left, right = _broadcast_pair(left, right)
                cur = cmp_op(left, right)
                out = cur if out is None else np.logical_and(out, cur)
                left = right
            return out
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("Only registered function calls are supported")
            name = node.func.id
            try:
                op = self.registry.get(name)
            except KeyError as exc:
                raise FormulaError(str(exc)) from exc
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise FormulaError("Expanded keyword arguments are not supported")
                kwargs[kw.arg] = self._eval_node(kw.value)
            return op.func(*args, **kwargs)
        raise FormulaError(f"Unsupported formula syntax: {type(node).__name__}")


_BIN_OPS = {
    ast.Add: py_operator.add,
    ast.Sub: py_operator.sub,
    ast.Mult: py_operator.mul,
    ast.Div: py_operator.truediv,
    ast.Mod: py_operator.mod,
}

_UNARY_OPS = {
    ast.USub: py_operator.neg,
    ast.UAdd: py_operator.pos,
    ast.Not: np.logical_not,
}

_CMP_OPS = {
    ast.Gt: py_operator.gt,
    ast.GtE: py_operator.ge,
    ast.Lt: py_operator.lt,
    ast.LtE: py_operator.le,
    ast.Eq: py_operator.eq,
    ast.NotEq: py_operator.ne,
}


def _broadcast_pair(left: Any, right: Any):
    """Broadcast daily market vectors against [date, instrument] matrices."""
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.ndim == 2 and right_arr.ndim == 1 and left_arr.shape[0] == right_arr.shape[0]:
        return left, right_arr[:, None]
    if right_arr.ndim == 2 and left_arr.ndim == 1 and right_arr.shape[0] == left_arr.shape[0]:
        return left_arr[:, None], right
    return left, right
