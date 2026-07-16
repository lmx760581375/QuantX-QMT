"""Compatibility adapters for incremental migration to decision contracts."""

from .legacy_policy import LegacyCompositeStrategyAdapter, LegacyRiskExecutionAdapter

__all__ = ["LegacyCompositeStrategyAdapter", "LegacyRiskExecutionAdapter"]
