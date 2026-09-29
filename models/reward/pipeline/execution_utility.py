"""Reward 训练兼容的执行效用标签组件。

The research contract is intentionally narrow:

* submit at ``T+1`` close under the active QuantX buy constraints;
* exit at ``T+H`` close under the matching QuantX sell constraints;
* keep only rows executable at both fixed endpoints;
* rank those rows by round-trip net utility within each signal-date.

This module does not simulate portfolio cash, replacement, or delayed sells.
Those portfolio-level effects remain the responsibility of the final QuantX
backtest.  The endpoint label is only a leakage-safe per-stock training target.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import RAW_OHLCV_FIELDS, WTSPaths, WeakToStrongPathDataset, source_memmaps, target_memmap


BUY_COST_FACTOR = 1.0 - 0.0015  # 10bp slippage + 5bp commission.
SELL_COST_FACTOR = 1.0 - 0.0016  # 10bp slippage + 5bp commission + 1bp stamp tax.
EPS = 1.0e-6
TREND_LABELS = ("up", "range", "down")


def net_utility_from_return(gross_return: np.ndarray | float) -> np.ndarray:
    """Convert a gross close-to-close return into the fixed QuantX net utility."""

    values = np.asarray(gross_return, dtype=np.float32)
    return ((1.0 + values) * BUY_COST_FACTOR * SELL_COST_FACTOR - 1.0).astype(np.float32, copy=False)


def _limit_rate(symbols: np.ndarray) -> np.ndarray:
    values = np.full(len(symbols), 0.10, dtype=np.float32)
    symbols = np.asarray(symbols, dtype=str)
    values[np.char.startswith(symbols, "SZ30") | np.char.startswith(symbols, "SH68")] = 0.20
    values[np.char.startswith(symbols, "BJ")] = 0.30
    return values


def _trade_fields(raw: np.ndarray, instrument_idx: np.ndarray, date_idx: np.ndarray) -> dict[str, np.ndarray]:
    fields = {name: RAW_OHLCV_FIELDS.index(name) for name in ("open", "high", "low", "close", "volume", "change")}
    return {
        name: np.asarray(raw[instrument_idx, date_idx, field], dtype=np.float32)
        for name, field in fields.items()
    }


def _suspended(values: dict[str, np.ndarray]) -> np.ndarray:
    return (~np.isfinite(values["close"])) | (~np.isfinite(values["volume"])) | (values["volume"] <= 0.0)


def _one_side_limit(values: dict[str, np.ndarray], limit: np.ndarray, *, up: bool) -> np.ndarray:
    change = values["change"]
    at_limit = np.isfinite(change) & (
        change >= limit - EPS if up else change <= -limit + EPS
    )
    return (
        at_limit
        & np.isfinite(values["open"])
        & np.isfinite(values["high"])
        & np.isfinite(values["low"])
        & np.isfinite(values["close"])
        & (np.abs(values["high"] - values["low"]) <= EPS)
        & (np.abs(values["open"] - values["close"]) <= EPS)
        & ((~np.isfinite(values["volume"])) | (values["volume"] <= 0.0))
    )


def _entry_reject_reasons(
    values: dict[str, np.ndarray],
    preclose: np.ndarray,
    limit: np.ndarray,
    price_jump_limit: float,
) -> np.ndarray:
    suspended = _suspended(values)
    limit_up = np.isfinite(values["change"]) & (values["change"] >= limit - EPS)
    one_side_limit_up = _one_side_limit(values, limit, up=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        jump_abs = np.abs(values["close"] / preclose - 1.0)
    price_jump = (
        np.isfinite(preclose)
        & (preclose > 0.0)
        & np.isfinite(values["close"])
        & (values["close"] > 0.0)
        & (jump_abs > float(price_jump_limit))
    )
    return np.select(
        (suspended, one_side_limit_up, limit_up, price_jump),
        ("suspended", "one_side_limit_up", "limit_up", "price_jump"),
        default="",
    ).astype(str)


def _exit_reject_reasons(values: dict[str, np.ndarray], limit: np.ndarray) -> np.ndarray:
    suspended = _suspended(values)
    limit_down = np.isfinite(values["change"]) & (values["change"] <= -limit + EPS)
    one_side_limit_down = _one_side_limit(values, limit, up=False)
    return np.select(
        (suspended, one_side_limit_down, limit_down),
        ("suspended", "one_side_limit_down", "limit_down"),
        default="",
    ).astype(str)


def build_execution_utility_labels(
    rows: pd.DataFrame,
    *,
    root: str | Path,
    kronos_root: str | Path,
    rank_horizon: int,
    price_jump_limit: float,
) -> pd.DataFrame:
    """Build endpoint-executable H-day net-utility labels for immutable rows."""

    required = {"row_id", "target_row", "instrument_idx", "date_idx", "instrument", "signal_date"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"Candidate rows missing required columns: {sorted(missing)}")
    if int(rank_horizon) < 1:
        raise ValueError("rank_horizon must be positive")
    if float(price_jump_limit) <= 0.0:
        raise ValueError("price_jump_limit must be positive")

    root = Path(root).expanduser().resolve()
    source = source_memmaps(kronos_root)
    raw = source.raw_ohlcv
    instrument_idx = rows["instrument_idx"].to_numpy(dtype=np.int64)
    signal_idx = rows["date_idx"].to_numpy(dtype=np.int64)
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + int(rank_horizon)
    if np.any(signal_idx < 0) or np.any(exit_idx >= raw.shape[1]):
        raise ValueError("Candidate index contains an entry or exit date outside the raw market calendar")

    limit = _limit_rate(rows["instrument"].to_numpy(dtype=str))
    entry_values = _trade_fields(raw, instrument_idx, entry_idx)
    exit_values = _trade_fields(raw, instrument_idx, exit_idx)
    close_field = RAW_OHLCV_FIELDS.index("close")
    preclose = np.asarray(raw[instrument_idx, signal_idx, close_field], dtype=np.float32)
    entry_reject = _entry_reject_reasons(entry_values, preclose, limit, float(price_jump_limit))
    exit_reject = _exit_reject_reasons(exit_values, limit)

    target = target_memmap(WTSPaths(root), "abs", mode="r")
    target_rows = rows["target_row"].to_numpy(dtype=np.int64)
    offset = int(rank_horizon) - 1
    if offset >= target.shape[1]:
        raise ValueError(f"rank_horizon={rank_horizon} exceeds target width={target.shape[1]}")
    gross_return = np.asarray(target[target_rows, offset], dtype=np.float32)
    utility = net_utility_from_return(gross_return)
    eligible = (entry_reject == "") & (exit_reject == "") & np.isfinite(gross_return)

    output = rows.loc[
        :, ["row_id", "target_row", "instrument_idx", "date_idx", "instrument", "signal_date"]
    ].copy()
    output["entry_date"] = pd.to_datetime(source.calendar.iloc[entry_idx]["date"].to_numpy()).strftime("%Y-%m-%d")
    output["exit_date"] = pd.to_datetime(source.calendar.iloc[exit_idx]["date"].to_numpy()).strftime("%Y-%m-%d")
    output["entry_fillable"] = entry_reject == ""
    output["exit_fillable"] = exit_reject == ""
    output["entry_reject_reason"] = entry_reject
    output["exit_reject_reason"] = exit_reject
    output["gross_return"] = gross_return
    output["net_utility"] = np.where(eligible, utility, np.nan).astype(np.float32)
    output["eligible"] = eligible
    output["rank_horizon"] = np.int16(rank_horizon)
    output["price_jump_limit"] = np.float32(price_jump_limit)
    return output


def execution_label_summary(labels: pd.DataFrame) -> dict[str, Any]:
    """Summarize a label artifact without changing its data."""

    entry = labels["entry_reject_reason"].replace("", "ok").value_counts().sort_index()
    exit_ = labels["exit_reject_reason"].replace("", "ok").value_counts().sort_index()
    eligible = labels.loc[labels["eligible"], "net_utility"]
    return {
        "rows": int(len(labels)),
        "eligible_rows": int(labels["eligible"].sum()),
        "eligible_ratio": float(labels["eligible"].mean()),
        "entry_reason_counts": {str(key): int(value) for key, value in entry.items()},
        "exit_reason_counts": {str(key): int(value) for key, value in exit_.items()},
        "net_utility": {
            "mean": float(eligible.mean()) if not eligible.empty else None,
            "median": float(eligible.median()) if not eligible.empty else None,
        },
    }


@dataclass(frozen=True)
class UtilityPairDay:
    date: str
    date_code: int
    good_positions: np.ndarray
    bad_positions_by_good: tuple[np.ndarray, ...]
    labels: np.ndarray


class ExecutionUtilitySameDatePairDataset(Dataset):
    """All-up/range/down same-date pairs ranked by executable net utility."""

    _MASK64 = (1 << 64) - 1

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        labels: pd.DataFrame,
        scaler,
        input_mode: str,
        trend_threshold: float,
        min_utility_gap: float,
        bad_per_good: int,
        max_open_feature_shards: int,
        seed: int,
    ):
        if float(trend_threshold) <= 0.0:
            raise ValueError("trend_threshold must be positive")
        if float(min_utility_gap) <= 0.0:
            raise ValueError("min_utility_gap must be positive")
        if int(bad_per_good) < 1:
            raise ValueError("bad_per_good must be positive")
        required = {"row_id", "eligible", "net_utility"}
        missing = required - set(labels.columns)
        if missing:
            raise KeyError(f"Execution utility labels missing required columns: {sorted(missing)}")

        ordered = rows.reset_index(drop=True).copy()
        joined = ordered.loc[:, ["row_id", "signal_date"]].merge(
            labels.loc[:, ["row_id", "eligible", "net_utility"]],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        if joined["eligible"].isna().any():
            raise RuntimeError(f"Execution utility labels do not cover {int(joined['eligible'].isna().sum())} split rows")
        self.base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=ordered,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
        )
        self.utilities = joined["net_utility"].to_numpy(dtype=np.float32)
        self.eligible = joined["eligible"].to_numpy(dtype=bool)
        self.trend_threshold = float(trend_threshold)
        self.min_utility_gap = float(min_utility_gap)
        self.bad_per_good = int(bad_per_good)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)
        self._days, build_stats = self._build_days()
        if not self._days:
            raise RuntimeError("Execution utility pair construction produced no eligible same-date pairs")
        self._ends = np.cumsum(
            np.asarray([len(day.good_positions) * self.bad_per_good for day in self._days], dtype=np.int64)
        )
        self.stats = {
            "pair_mode": "execution_utility_all_up",
            "source_rows": int(len(ordered)),
            "eligible_rows": int(self.eligible.sum()),
            "trend_threshold": self.trend_threshold,
            "min_utility_gap": self.min_utility_gap,
            "bad_per_good": self.bad_per_good,
            "days_with_pairs": int(len(self._days)),
            "pairs_per_epoch": int(self._ends[-1]),
            **build_stats,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (int(value) + 0x9E3779B97F4A7C15) & ExecutionUtilitySameDatePairDataset._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ExecutionUtilitySameDatePairDataset._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ExecutionUtilitySameDatePairDataset._MASK64
        return (value ^ (value >> 31)) & ExecutionUtilitySameDatePairDataset._MASK64

    def _build_days(self) -> tuple[list[UtilityPairDay], dict[str, int]]:
        frame = self.base.rows.loc[:, ["signal_date"]].copy()
        frame["position"] = np.arange(len(frame), dtype=np.int64)
        frame["utility"] = self.utilities
        frame["eligible"] = self.eligible
        days: list[UtilityPairDay] = []
        stats = {
            "dates_seen": 0,
            "dates_without_eligible_rows": 0,
            "eligible_up_rows": 0,
            "eligible_range_rows": 0,
            "eligible_down_rows": 0,
            "rows_without_lower_utility_bad": 0,
        }
        for date_code, (date, group) in enumerate(frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            positions = group["position"].to_numpy(dtype=np.int64)
            utility = group["utility"].to_numpy(dtype=np.float32)
            valid = group["eligible"].to_numpy(dtype=bool) & np.isfinite(utility)
            positions = positions[valid]
            utility = utility[valid]
            if len(positions) < 2:
                stats["dates_without_eligible_rows"] += 1
                continue
            labels = np.full(len(positions), 1, dtype=np.int64)
            labels[utility >= self.trend_threshold] = 0
            labels[utility <= -self.trend_threshold] = 2
            stats["eligible_up_rows"] += int((labels == 0).sum())
            stats["eligible_range_rows"] += int((labels == 1).sum())
            stats["eligible_down_rows"] += int((labels == 2).sum())

            pools: dict[int, np.ndarray] = {}
            for local_position, (position, value) in enumerate(zip(positions, utility, strict=True)):
                bad = positions[utility < value - self.min_utility_gap]
                if len(bad):
                    pools[int(position)] = np.asarray(bad, dtype=np.int64)
                else:
                    stats["rows_without_lower_utility_bad"] += 1
            up_positions = [int(position) for position, label in zip(positions, labels, strict=True) if label == 0 and int(position) in pools]
            if not up_positions:
                continue

            selected_positions: list[int] = list(up_positions)
            selected_labels: list[int] = [0] * len(up_positions)
            # Preserve the existing all-up contract: every eligible up remains,
            # while range/down are deterministically cycled to equal cardinality.
            for label in (1, 2):
                candidates = [int(position) for position, item in zip(positions, labels, strict=True) if item == label and int(position) in pools]
                if not candidates:
                    candidates = up_positions
                start = int(self._mix64(self.seed + int(date_code) * 17 + label) % len(candidates))
                selected_positions.extend(candidates[(start + offset) % len(candidates)] for offset in range(len(up_positions)))
                selected_labels.extend([label] * len(up_positions))
            good_positions = np.asarray(selected_positions, dtype=np.int64)
            bad_positions = tuple(pools[int(position)] for position in good_positions)
            days.append(
                UtilityPairDay(
                    date=str(date),
                    date_code=int(date_code),
                    good_positions=good_positions,
                    bad_positions_by_good=bad_positions,
                    labels=np.asarray(selected_labels, dtype=np.int64),
                )
            )
        return days, stats

    def __len__(self) -> int:
        return int(self._ends[-1])

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _resolve_pair(self, index: int) -> tuple[UtilityPairDay, int, int, int]:
        day_index = int(np.searchsorted(self._ends, int(index), side="right"))
        if day_index >= len(self._days):
            raise IndexError(index)
        day = self._days[day_index]
        before = 0 if day_index == 0 else int(self._ends[day_index - 1])
        local = int(index) - before
        good_index = local // self.bad_per_good
        bad_draw = local % self.bad_per_good
        good_position = int(day.good_positions[good_index])
        bad_pool = day.bad_positions_by_good[good_index]
        seed = self._mix64(
            self.seed
            + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9
            + int(day.date_code) * 0x94D049BB133111EB
            + good_position * 0xD1B54A32D192ED03
        )
        bad_position = int(bad_pool[(int(seed) + int(bad_draw)) % len(bad_pool)])
        return day, good_index, good_position, bad_position

    def pair_metadata(self, index: int) -> dict[str, Any]:
        day, good_index, good_position, bad_position = self._resolve_pair(index)
        return {
            "signal_date": day.date,
            "date_code": day.date_code,
            "label": TREND_LABELS[int(day.labels[good_index])],
            "good_position": good_position,
            "bad_position": bad_position,
            "good_utility": float(self.utilities[good_position]),
            "bad_utility": float(self.utilities[bad_position]),
            "eligible_bad_pool": int(len(day.bad_positions_by_good[good_index])),
        }

    def __getitem__(self, index: int):
        day, good_index, good_position, bad_position = self._resolve_pair(index)
        good_utility = float(self.utilities[good_position])
        bad_utility = float(self.utilities[bad_position])
        if not bad_utility < good_utility - self.min_utility_gap:
            raise RuntimeError("Execution utility pair violates its strict same-date net-utility gap")
        good = self.base[good_position]
        bad = self.base[bad_position]
        return (
            good[0],
            good[1],
            good[2],
            bad[0],
            bad[1],
            bad[2],
            torch.tensor(good_utility, dtype=torch.float32),
            torch.tensor(bad_utility, dtype=torch.float32),
            torch.tensor(int(day.labels[good_index]), dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
        )
