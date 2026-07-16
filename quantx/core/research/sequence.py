"""Causal raw-window feature construction for sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowBatch:
    sample_index: tuple[tuple[pd.Timestamp, str], ...]
    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: tuple[str, ...]
    lookback_sessions: int


class RawWindowFeatureBuilder:
    def __init__(self, lookback_sessions: int):
        if lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        self.lookback_sessions = int(lookback_sessions)

    def transform(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        *,
        history: pd.DataFrame | None = None,
    ) -> WindowBatch:
        if frame.index.names != ["signal_time", "instrument"]:
            raise ValueError("RawWindowFeatureBuilder requires signal_time/instrument index")
        source = frame
        if history is not None and not history.empty:
            source = pd.concat([history, frame])
            source = source[~source.index.duplicated(keep="last")].sort_index()
        output_index = tuple((pd.Timestamp(date), str(symbol)) for date, symbol in frame.index)
        windows: dict[tuple[pd.Timestamp, str], np.ndarray] = {}
        for instrument in source.index.get_level_values("instrument").unique():
            group = source.xs(instrument, level="instrument").sort_index()
            values = group.loc[:, feature_columns].to_numpy(dtype=float)
            dates = pd.DatetimeIndex(group.index)
            for position, date in enumerate(dates):
                start = max(0, position - self.lookback_sessions + 1)
                available = values[start : position + 1]
                window = np.full((self.lookback_sessions, len(feature_columns)), np.nan, dtype=float)
                window[-len(available) :] = available
                windows[(pd.Timestamp(date), str(instrument))] = window
        values = np.stack([windows[index] for index in output_index]) if output_index else np.empty((0, 0, 0))
        valid_mask = np.isfinite(values).all(axis=2) if len(values) else np.empty((0, 0), dtype=bool)
        return WindowBatch(
            sample_index=output_index,
            values=values,
            valid_mask=valid_mask,
            feature_names=tuple(feature_columns),
            lookback_sessions=self.lookback_sessions,
        )
