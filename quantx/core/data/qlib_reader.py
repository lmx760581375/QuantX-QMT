"""Minimal reader for QuantX Qlib-style binary market data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QlibBinReader:
    """Read the Qlib binary files produced by ``BaostockToQlibConverter``."""

    def __init__(self, provider_uri: str | Path, freq: str = "day"):
        self.provider_uri = Path(provider_uri)
        self.freq = freq
        self._calendar: pd.DatetimeIndex | None = None

    def calendar(self, start_time: str | None = None, end_time: str | None = None) -> pd.DatetimeIndex:
        dates = self._read_calendar()
        if start_time:
            dates = dates[dates >= pd.Timestamp(start_time)]
        if end_time:
            dates = dates[dates <= pd.Timestamp(end_time)]
        return dates

    def features(
        self,
        symbols: Iterable[str],
        fields: Iterable[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        dates = self.calendar(start_time, end_time)
        frames: list[pd.DataFrame] = []
        field_list = [str(field) for field in fields]
        symbol_list = [str(symbol) for symbol in symbols]
        logger.info(
            "Qlib bin read start symbols=%s fields=%s start=%s end=%s",
            len(symbol_list),
            len(field_list),
            start_time,
            end_time,
        )
        for idx, symbol in enumerate(symbol_list, start=1):
            frame = self._read_symbol(str(symbol), field_list, dates)
            if not frame.empty:
                frames.append(frame)
            if idx == 1 or idx % 100 == 0 or idx == len(symbol_list):
                logger.info(
                    "Qlib bin read progress %s/%s, loaded=%s, latest=%s",
                    idx,
                    len(symbol_list),
                    len(frames),
                    symbol,
                )
        if not frames:
            return pd.DataFrame(columns=field_list)
        result = pd.concat(frames).sort_index()
        logger.info("Qlib bin read done rows=%s columns=%s", len(result), len(result.columns))
        return result

    def _read_calendar(self) -> pd.DatetimeIndex:
        if self._calendar is not None:
            return self._calendar
        calendar_path = self.provider_uri / "calendars" / f"{self.freq}.txt"
        if not calendar_path.exists():
            raise FileNotFoundError(f"Missing qlib calendar file: {calendar_path}")
        values = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._calendar = pd.DatetimeIndex(pd.to_datetime(values), name="datetime")
        return self._calendar

    def _read_symbol(self, symbol: str, fields: List[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        symbol_dir = self.provider_uri / "features" / symbol.lower()
        if not symbol_dir.exists():
            return pd.DataFrame()
        columns = {}
        for field in fields:
            values = self._read_field(symbol_dir, field, dates)
            if values is not None:
                columns[field] = values
        if not columns:
            return pd.DataFrame()
        frame = pd.DataFrame(columns, index=dates)
        frame["instrument"] = symbol
        frame.index.name = "datetime"
        return frame.set_index("instrument", append=True).reorder_levels(["instrument", "datetime"])

    def _read_field(self, symbol_dir: Path, field: str, dates: pd.DatetimeIndex) -> np.ndarray | None:
        field_name = field.lower().lstrip("$")
        bin_path = symbol_dir / f"{field_name}.{self.freq}.bin"
        if not bin_path.exists():
            return None
        raw = np.fromfile(str(bin_path), dtype="<f4")
        if raw.size <= 1:
            return None
        start_idx = int(raw[0])
        values = raw[1:]
        calendar = self._read_calendar()
        out = np.full(len(dates), np.nan, dtype=np.float32)
        if len(dates) == 0:
            return out
        first = int(calendar.searchsorted(dates[0], side="left"))
        last = int(calendar.searchsorted(dates[-1], side="right")) - 1
        value_start = max(first, start_idx)
        value_end = min(last, start_idx + len(values) - 1)
        if value_end < value_start:
            return out
        out_start = value_start - first
        out_end = out_start + (value_end - value_start + 1)
        raw_start = value_start - start_idx
        raw_end = raw_start + (value_end - value_start + 1)
        out[out_start:out_end] = values[raw_start:raw_end]
        return out
