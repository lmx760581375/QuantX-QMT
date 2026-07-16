"""Explicit component registration without arbitrary config imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ComponentCapabilities:
    needs_account_state: bool
    supports_batch_prediction: bool
    supports_online_update: bool
    output_type: str
    required_optional_extra: str | None = None


@dataclass(frozen=True)
class _Registration:
    factory: Callable[[dict[str, Any]], Any]
    capabilities: ComponentCapabilities


class ComponentRegistry:
    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], _Registration] = {}

    def register(
        self,
        component_type: str,
        name: str,
        factory: Callable[[dict[str, Any]], Any],
        capabilities: ComponentCapabilities,
    ) -> None:
        key = self._key(component_type, name)
        if key in self._registrations:
            raise ValueError(f"Component {component_type}:{name} is already registered")
        self._registrations[key] = _Registration(factory=factory, capabilities=capabilities)

    def build(self, component_type: str, name: str, config: dict[str, Any]) -> Any:
        return self._get(component_type, name).factory(dict(config))

    def capabilities(self, component_type: str, name: str) -> ComponentCapabilities:
        return self._get(component_type, name).capabilities

    def names(self, component_type: str) -> tuple[str, ...]:
        return tuple(sorted(name for kind, name in self._registrations if kind == component_type))

    def _get(self, component_type: str, name: str) -> _Registration:
        key = self._key(component_type, name)
        try:
            return self._registrations[key]
        except KeyError as exc:
            available = ", ".join(self.names(component_type)) or "<none>"
            raise KeyError(
                f"Unknown component {component_type}:{name}; available {component_type} components: {available}"
            ) from exc

    @staticmethod
    def _key(component_type: str, name: str) -> tuple[str, str]:
        if not component_type or not name:
            raise ValueError("Component type and name are required")
        return component_type, name
