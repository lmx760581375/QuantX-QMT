"""Read QuantX Qlib-style day.bin files without materializing the full panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_calendar(provider_uri: Path, *, start: str | None = None, end: str | None = None) -> pd.DatetimeIndex:
    path = provider_uri / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing calendar: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dates = pd.DatetimeIndex(pd.to_datetime(values), name="date")
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    return dates


def read_instruments(provider_uri: Path) -> pd.DataFrame:
    path = provider_uri / "instruments" / "all.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing instruments: {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            rows.append({"instrument": parts[0], "start_date": parts[1], "end_date": parts[2]})
        elif len(parts) == 1:
            rows.append({"instrument": parts[0], "start_date": "", "end_date": ""})
    return pd.DataFrame(rows)


def read_day_bin_field(
    provider_uri: Path,
    instrument: str,
    field: str,
    calendar_full: pd.DatetimeIndex,
    calendar_selected: pd.DatetimeIndex,
) -> np.ndarray:
    symbol_dir = provider_uri / "features" / instrument.lower()
    path = symbol_dir / f"{field.lower().lstrip('$')}.day.bin"
    out = np.full(len(calendar_selected), np.nan, dtype=np.float32)
    if not path.exists() or len(calendar_selected) == 0:
        return out
    raw = np.fromfile(str(path), dtype="<f4")
    if raw.size <= 1:
        return out
    start_idx = int(raw[0])
    values = raw[1:]
    selected_start = int(calendar_full.searchsorted(calendar_selected[0], side="left"))
    selected_end = int(calendar_full.searchsorted(calendar_selected[-1], side="right")) - 1
    value_start = max(selected_start, start_idx)
    value_end = min(selected_end, start_idx + len(values) - 1)
    if value_end < value_start:
        return out
    out_start = value_start - selected_start
    out_end = out_start + (value_end - value_start + 1)
    raw_start = value_start - start_idx
    raw_end = raw_start + (value_end - value_start + 1)
    out[out_start:out_end] = values[raw_start:raw_end]
    return out


def baostock_code_to_instrument(value: str) -> str:
    text = str(value).strip()
    if "." in text:
        prefix, code = text.split(".", 1)
        return prefix.upper() + code
    if text.startswith(("SH", "SZ", "BJ")):
        return text
    if text.startswith("6"):
        return "SH" + text
    return "SZ" + text
