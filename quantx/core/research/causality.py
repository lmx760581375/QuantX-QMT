"""Static read-window analysis for FactorRuntime expressions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

from quantx.core.factor_runtime.operators import DEFAULT_REGISTRY

from .specs import FeatureSpec


@dataclass(frozen=True)
class ReadWindow:
    min_offset_sessions: int | None
    max_offset_sessions: int

    def merge(self, other: "ReadWindow") -> "ReadWindow":
        minimum = (
            None
            if self.min_offset_sessions is None or other.min_offset_sessions is None
            else min(self.min_offset_sessions, other.min_offset_sessions)
        )
        return ReadWindow(minimum, max(self.max_offset_sessions, other.max_offset_sessions))

    def shift(self, sessions: int) -> "ReadWindow":
        minimum = None if self.min_offset_sessions is None else self.min_offset_sessions + sessions
        return ReadWindow(minimum, self.max_offset_sessions + sessions)

    def rolling(self, sessions: int) -> "ReadWindow":
        minimum = None if self.min_offset_sessions is None else self.min_offset_sessions - sessions + 1
        return ReadWindow(minimum, self.max_offset_sessions)


class FeatureCausalityValidator:
    def analyze(self, spec: FeatureSpec) -> Mapping[str, ReadWindow]:
        resolved = {
            **{field: ReadWindow(0, 0) for field in spec.raw_fields},
            **{name: ReadWindow(0, 0) for name in spec.groups},
        }
        pending = dict(spec.expressions)
        while pending:
            progressed = False
            for name, expression in list(pending.items()):
                tree = ast.parse(expression, mode="eval")
                names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                dependencies = names - set(DEFAULT_REGISTRY.names())
                unknown = dependencies - set(resolved) - set(pending)
                if unknown:
                    raise ValueError(f"Feature {name} references unknown inputs: {sorted(unknown)}")
                if dependencies <= set(resolved):
                    resolved[name] = self._window(tree.body, resolved)
                    del pending[name]
                    progressed = True
            if not progressed:
                raise ValueError(f"Feature dependency cycle: {sorted(pending)}")
        return {name: resolved[name] for name in spec.expressions}

    def require_causal(self, spec: FeatureSpec) -> None:
        windows = self.analyze(spec)
        leaking = {name: window for name, window in windows.items() if window.max_offset_sessions > 0}
        if leaking:
            detail = ", ".join(f"{name}=+{window.max_offset_sessions}" for name, window in leaking.items())
            raise ValueError(f"Feature expressions read future data: {detail}")

    def _window(self, node: ast.AST, resolved: Mapping[str, ReadWindow]) -> ReadWindow:
        if isinstance(node, ast.Constant):
            return ReadWindow(0, 0)
        if isinstance(node, ast.Name):
            if node.id not in resolved:
                raise ValueError(f"Unresolved feature input: {node.id}")
            return resolved[node.id]
        if isinstance(node, ast.UnaryOp):
            return self._window(node.operand, resolved)
        if isinstance(node, ast.BinOp):
            return self._window(node.left, resolved).merge(self._window(node.right, resolved))
        if isinstance(node, ast.BoolOp):
            return self._merge_nodes(node.values, resolved)
        if isinstance(node, ast.Compare):
            return self._merge_nodes([node.left, *node.comparators], resolved)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "Ref":
                return self._window(node.args[0], resolved).shift(-self._constant_int(node.args[1]))
            if name in {"Min", "Max", "Mean", "Sum", "RollingQuantile"}:
                return self._window(node.args[0], resolved).rolling(self._constant_int(node.args[1]))
            if name in {"EMA", "SMA_TDX", "ExpandingQuantile", "SuperTrend"}:
                base = self._merge_nodes(node.args[:1] if name != "SuperTrend" else node.args[:3], resolved)
                return ReadWindow(None, base.max_offset_sessions)
            if name == "ATR":
                return self._merge_nodes(node.args[:3], resolved).rolling(self._constant_int(node.args[3]))
            if name in {"Cross", "Filter"}:
                return self._merge_nodes(node.args, resolved).rolling(2)
            try:
                operator = DEFAULT_REGISTRY.get(name)
            except KeyError as exc:
                raise ValueError(f"Unknown feature operator: {name}") from exc
            if operator.kind == "time_series":
                raise ValueError(f"Time-series operator {name} must declare a read-window rule")
            return self._merge_nodes(node.args, resolved)
        raise ValueError(f"Unsupported feature syntax for causality analysis: {type(node).__name__}")

    def _merge_nodes(self, nodes, resolved: Mapping[str, ReadWindow]) -> ReadWindow:
        windows = [self._window(node, resolved) for node in nodes if not isinstance(node, ast.Constant)]
        if not windows:
            return ReadWindow(0, 0)
        result = windows[0]
        for window in windows[1:]:
            result = result.merge(window)
        return result

    @staticmethod
    def _constant_int(node: ast.AST) -> int:
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            sign = -1
            node = node.operand
        if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
            raise ValueError("Window and Ref offsets must be numeric constants")
        value = sign * int(node.value)
        if value == 0:
            return 0
        return value
