"""Stable feature schema identity independent of model implementation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    dtypes: tuple[str, ...]
    lookback_sessions: int
    frequency: str
    max_lookahead_sessions: int
    causality_hash: str
    schema_hash: str

    @classmethod
    def create(
        cls,
        *,
        names: tuple[str, ...],
        dtypes: tuple[str, ...],
        lookback_sessions: int,
        frequency: str,
        read_windows: Mapping[str, tuple[int | None, int]],
    ) -> "FeatureSchema":
        if len(names) != len(dtypes):
            raise ValueError("Feature names and dtypes must have equal length")
        if len(set(names)) != len(names):
            raise ValueError("Feature schema names must be unique")
        causality_payload = {name: [window[0], window[1]] for name, window in sorted(read_windows.items())}
        causality_hash = _hash(causality_payload)
        max_lookahead = max((window[1] for window in read_windows.values()), default=0)
        schema_payload = {
            "names": names,
            "dtypes": dtypes,
            "lookback_sessions": lookback_sessions,
            "frequency": frequency,
            "max_lookahead_sessions": max_lookahead,
            "causality_hash": causality_hash,
        }
        schema_hash = _hash(schema_payload)
        return cls(
            names=names,
            dtypes=dtypes,
            lookback_sessions=lookback_sessions,
            frequency=frequency,
            max_lookahead_sessions=max_lookahead,
            causality_hash=causality_hash,
            schema_hash=schema_hash,
        )


def _hash(payload) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
