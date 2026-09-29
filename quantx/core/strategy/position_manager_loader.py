"""Explicit loader for optional post-selection position-manager plugins."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

from .base import PositionManager


def build_position_manager(config: Dict[str, Any] | None) -> PositionManager | None:
    """Load an explicitly declared plugin without coupling QuantX to a model.

    Strategy YAML is trusted local configuration.  Requiring both the plugin
    directory and an ``module:factory`` entrypoint makes the runtime dependency
    visible in the saved resolved config rather than implicit in PYTHONPATH.
    """
    if not config:
        return None
    if str(config.get("type")) != "python_plugin":
        raise ValueError("execution.position_manager.type must be python_plugin")
    factory_spec = str(config.get("factory") or "")
    if ":" not in factory_spec:
        raise ValueError("execution.position_manager.factory must be module:callable")
    plugin_path = Path(str(config.get("python_path") or "")).expanduser().resolve()
    if not plugin_path.is_dir():
        raise ValueError(f"execution.position_manager.python_path is not a directory: {plugin_path}")
    module_name, factory_name = factory_spec.split(":", 1)
    if str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"Position-manager factory is not callable: {factory_spec}")
    manager = factory(dict(config.get("config") or {}))
    if not isinstance(manager, PositionManager):
        raise TypeError(f"Position-manager factory must return PositionManager, got {type(manager).__name__}")
    return manager
