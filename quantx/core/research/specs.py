"""Serializable specifications for features, labels, and research datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class _StableSpec:
    @property
    def spec_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class FeatureSpec(_StableSpec):
    name: str
    raw_fields: tuple[str, ...]
    expressions: Mapping[str, str]
    lookback_sessions: int
    normalization: str = "none"
    missing_policy: str = "preserve_with_mask"
    require_causal: bool = True
    groups: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.raw_fields:
            raise ValueError("FeatureSpec name and raw_fields are required")
        if self.lookback_sessions < 0:
            raise ValueError("lookback_sessions cannot be negative")
        object.__setattr__(self, "raw_fields", tuple(str(field).lstrip("$") for field in self.raw_fields))
        object.__setattr__(self, "expressions", dict(self.expressions))
        object.__setattr__(self, "groups", {str(key): dict(value) for key, value in self.groups.items()})


@dataclass(frozen=True)
class LabelSpec(_StableSpec):
    name: str
    horizon_sessions: int
    entry_price: str = "next_open"
    exit_price: str = "future_open"
    benchmark: str | None = None
    excess_return: bool = False
    winsorize: tuple[float, float] | None = None
    missing_exit_policy: str = "mark_missing"
    execution_diagnostic: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.horizon_sessions < 1:
            raise ValueError("LabelSpec name and positive horizon_sessions are required")
        if self.entry_price != "next_open" or self.exit_price != "future_open":
            raise ValueError("The first research implementation requires next_open/future_open labels")
        if self.missing_exit_policy not in {"mark_missing", "drop_after_audit"}:
            raise ValueError("Unsupported missing_exit_policy")
        if self.excess_return and not self.benchmark:
            raise ValueError("excess_return labels require a benchmark")
        if self.benchmark is not None:
            object.__setattr__(self, "benchmark", str(self.benchmark).upper())
        if self.winsorize is not None:
            lower, upper = (float(value) for value in self.winsorize)
            if not 0 <= lower < upper <= 1:
                raise ValueError("winsorize must contain quantiles within [0, 1]")
            object.__setattr__(self, "winsorize", (lower, upper))


@dataclass(frozen=True)
class DatasetSpec(_StableSpec):
    data_version_id: str
    provider_uri: str
    universe: str
    start: str
    end: str
    features: FeatureSpec
    label: LabelSpec
    universe_version_id: str
    physical_snapshot_id: str | None = None
    sample_filter: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_version_id or not self.provider_uri or not self.universe_version_id:
            raise ValueError("DatasetSpec data identities are required")
        object.__setattr__(self, "sample_filter", dict(self.sample_filter))
