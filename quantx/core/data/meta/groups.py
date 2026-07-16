"""Load static metadata groups as FactorRuntime inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .store import normalize_symbol


DEFAULT_INDUSTRY_CSV = Path("data/meta/snapshots/industry_membership.csv")
DEFAULT_SECTOR_CSV = Path("data/meta/snapshots/sector_membership.csv")


def load_meta_groups(instruments: Sequence[str], groups: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Return group arrays keyed by config group name.

    Industry groups are encoded as one integer code per instrument. Concept
    groups are encoded as a multi-membership boolean matrix with shape
    ``[instrument, concept]``.
    """
    if not groups:
        return {}
    normalized = [normalize_symbol(str(symbol)) for symbol in instruments]
    loaded: dict[str, np.ndarray] = {}
    for name, raw_cfg in groups.items():
        cfg = dict(raw_cfg or {})
        source = str(cfg.get("source") or "")
        if source in {"meta.industry_l1", "industry_l1"}:
            loaded[str(name)] = load_industry_group(normalized, cfg)
        elif source in {"meta.concept", "concept"}:
            loaded[str(name)] = load_concept_membership(normalized, cfg)
        else:
            raise ValueError(f"Unsupported group source for {name}: {source}")
    return loaded


def group_config_identity(groups: Mapping[str, Mapping[str, Any]]) -> str:
    """Stable identity for group configs, including local CSV checksums."""
    payload: dict[str, dict[str, Any]] = {}
    for name, raw_cfg in sorted((groups or {}).items()):
        cfg = dict(raw_cfg or {})
        path = _group_path(cfg)
        item = {key: value for key, value in cfg.items() if key != "path"}
        item["path"] = str(path)
        item["sha256"] = _file_sha256(path) if path.exists() else None
        payload[str(name)] = item
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def load_industry_group(instruments: Sequence[str], cfg: Mapping[str, Any]) -> np.ndarray:
    path = Path(cfg.get("path") or DEFAULT_INDUSTRY_CSV)
    if not path.exists():
        raise ValueError(f"Industry group CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _first_existing_column(frame, ["symbol", "股票代码", "code"])
    group_col = _first_existing_column(frame, ["industry_code", "industry_name", "板块代码", "板块名称", "industry"])
    mapping = {
        normalize_symbol(row[symbol_col]): str(row[group_col])
        for _, row in frame.iterrows()
        if pd.notna(row.get(symbol_col)) and pd.notna(row.get(group_col))
    }
    labels = [mapping.get(normalize_symbol(symbol)) for symbol in instruments]
    return _encode_labels(labels)


def load_concept_membership(instruments: Sequence[str], cfg: Mapping[str, Any]) -> np.ndarray:
    path = Path(cfg.get("path") or DEFAULT_SECTOR_CSV)
    if not path.exists():
        raise ValueError(f"Concept group CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str)
    if "sector_type" in frame.columns:
        frame = frame[frame["sector_type"].fillna("concept") == str(cfg.get("sector_type", "concept"))]
    if frame.empty:
        return np.zeros((len(instruments), 0), dtype=bool)
    symbol_col = _first_existing_column(frame, ["symbol", "股票代码", "code"])
    group_col = _first_existing_column(frame, ["sector_code", "sector_name", "板块代码", "板块名称", "concept"])
    by_symbol: dict[str, set[str]] = {}
    group_names: set[str] = set()
    for _, row in frame.iterrows():
        if pd.isna(row.get(symbol_col)) or pd.isna(row.get(group_col)):
            continue
        symbol = normalize_symbol(row[symbol_col])
        group = str(row[group_col])
        by_symbol.setdefault(symbol, set()).add(group)
        group_names.add(group)
    ordered_groups = sorted(group_names)
    group_index = {group: index for index, group in enumerate(ordered_groups)}
    out = np.zeros((len(instruments), len(ordered_groups)), dtype=bool)
    for inst_index, symbol in enumerate(instruments):
        for group in by_symbol.get(normalize_symbol(symbol), set()):
            out[inst_index, group_index[group]] = True
    return out


def _group_path(cfg: Mapping[str, Any]) -> Path:
    source = str(cfg.get("source") or "")
    if source in {"meta.industry_l1", "industry_l1"}:
        return Path(cfg.get("path") or DEFAULT_INDUSTRY_CSV)
    if source in {"meta.concept", "concept"}:
        return Path(cfg.get("path") or DEFAULT_SECTOR_CSV)
    return Path(cfg.get("path") or "")


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"CSV is missing required columns, expected one of: {list(names)}")


def _encode_labels(labels: Sequence[str | None]) -> np.ndarray:
    clean = [str(label) if label not in {None, "", "nan"} and not pd.isna(label) else None for label in labels]
    mapping = {label: index for index, label in enumerate(sorted({label for label in clean if label is not None}))}
    return np.asarray([mapping.get(label, -1) if label is not None else -1 for label in clean], dtype=np.int32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
