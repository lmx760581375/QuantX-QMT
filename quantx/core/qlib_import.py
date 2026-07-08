"""Helpers for importing a usable Qlib package."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def import_qlib():
    """Import Qlib, preferring an installed package and falling back to bundled source."""
    repo_root = Path(__file__).resolve().parents[2]
    bundled_source_root = repo_root / "qlib"
    last_error: Exception | None = None

    existing = sys.modules.get("qlib")
    if _is_usable_qlib(existing):
        return existing
    if existing is not None:
        _clear_qlib_modules()

    original_path = list(sys.path)
    sys.path = [path for path in original_path if not _is_blocked_path(path, repo_root, bundled_source_root)]
    try:
        try:
            qlib = importlib.import_module("qlib")
            if _is_usable_qlib(qlib):
                return qlib
            _clear_qlib_modules()
        except Exception as exc:
            last_error = exc
            _clear_qlib_modules()
    finally:
        sys.path = original_path

    if bundled_source_root.exists():
        _clear_qlib_modules()
        sys.path = [str(bundled_source_root)] + [
            path for path in original_path if not _is_blocked_path(path, repo_root, bundled_source_root)
        ]
        try:
            try:
                qlib = importlib.import_module("qlib")
                if _is_usable_qlib(qlib):
                    return qlib
            except Exception as exc:
                last_error = exc
                _clear_qlib_modules()
        finally:
            sys.path = original_path

    raise ImportError("Unable to import a usable qlib package with qlib.init().") from last_error


def _is_usable_qlib(module) -> bool:
    return module is not None and callable(getattr(module, "init", None))


def _clear_qlib_modules() -> None:
    for name in list(sys.modules):
        if name == "qlib" or name.startswith("qlib."):
            del sys.modules[name]


def _is_blocked_path(path: str, repo_root: Path, bundled_source_root: Path) -> bool:
    try:
        resolved = Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return resolved == repo_root.resolve() or resolved == bundled_source_root.resolve()
