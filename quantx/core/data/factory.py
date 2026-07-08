"""Data source factory."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .baostock_source import BaoStockConfig, BaoStockDataSource
from .base import DataSource
from .qmt_source import QMTDataSource, qmt_config_from_mapping


def create_data_source(config: Mapping[str, Any] | None = None) -> DataSource:
    data = _data_config(config)
    source = str(data.get("source") or data.get("type") or "qmt").lower()
    if source in {"qmt", "xtquant", "myquant"}:
        return QMTDataSource(qmt_config_from_mapping(data))
    if source in {"baostock", "bao_stock"}:
        return BaoStockDataSource(_baostock_config_from_mapping(data))
    raise ValueError(f"Unsupported data source: {source}")


def _data_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    data = dict(config)
    nested = data.get("data")
    if isinstance(nested, Mapping):
        return dict(nested)
    return data


def _baostock_config_from_mapping(data: Mapping[str, Any]) -> BaoStockConfig:
    allowed = {field.name for field in fields(BaoStockConfig)}
    values = {key: value for key, value in data.items() if key in allowed}
    return BaoStockConfig(**values)
