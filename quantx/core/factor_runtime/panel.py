"""Market panel representation for matrix factor computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketPanel:
    """Full-market data stored as field matrices with shape ``[T, N]``."""

    dates: pd.DatetimeIndex
    instruments: pd.Index
    fields: Dict[str, np.ndarray]

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        fields: Optional[Iterable[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        dtype=np.float32,
    ) -> "MarketPanel":
        if data is None or data.empty:
            return cls(pd.DatetimeIndex([]), pd.Index([]), {})
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("MarketPanel requires a MultiIndex DataFrame")

        frame = data.copy()
        names = list(frame.index.names)
        if names != ["datetime", "instrument"]:
            if "datetime" in names and "instrument" in names:
                frame = frame.reorder_levels(["datetime", "instrument"])
            else:
                frame = frame.swaplevel(0, 1)
            frame = frame.sort_index()
            frame.index = frame.index.set_names(["datetime", "instrument"])

        dates = pd.DatetimeIndex(frame.index.get_level_values("datetime").unique()).sort_values()
        instruments = pd.Index(frame.index.get_level_values("instrument").unique()).sort_values()

        wanted = list(fields) if fields is not None else list(frame.columns)
        matrices: Dict[str, np.ndarray] = {}
        for field in wanted:
            if field not in frame.columns:
                continue
            name = field[1:] if field.startswith("$") else field
            wide = frame[field].unstack("instrument").reindex(index=dates, columns=instruments)
            matrices[name] = wide.to_numpy(dtype=dtype, copy=True)
        for alias, source in (aliases or {}).items():
            source_name = str(source)
            if not source_name.startswith("$"):
                source_name = f"${source_name}"
            if source_name not in frame.columns:
                raise ValueError(f"Field alias {alias} references missing source field: {source}")
            clean_alias = str(alias).lstrip("$")
            wide = frame[source_name].unstack("instrument").reindex(index=dates, columns=instruments)
            matrices[clean_alias] = wide.to_numpy(dtype=dtype, copy=True)
        return cls(dates=dates, instruments=instruments, fields=matrices)

    def get(self, field: str) -> np.ndarray:
        name = field[1:] if field.startswith("$") else field
        return self.fields[name]

    def has(self, field: str) -> bool:
        name = field[1:] if field.startswith("$") else field
        return name in self.fields

    def date_index(self, date) -> int:
        idx = self.dates.get_loc(pd.Timestamp(date))
        if not isinstance(idx, int):
            raise KeyError(date)
        return idx

    def cross_section(self, values: Dict[str, np.ndarray], date) -> pd.DataFrame:
        idx = self.date_index(date)
        data = {}
        for name, value in values.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                data[name] = np.full(len(self.instruments), arr.item())
            elif arr.ndim == 1:
                if len(arr) == len(self.instruments):
                    data[name] = arr
                else:
                    data[name] = np.full(len(self.instruments), arr[idx])
            else:
                data[name] = arr[idx]
        return pd.DataFrame(data, index=self.instruments)
