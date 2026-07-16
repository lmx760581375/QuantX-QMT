"""Point-in-time instrument membership and integrity auditing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class UniverseRecord:
    instrument: str
    effective_from: pd.Timestamp
    effective_to: pd.Timestamp


@dataclass(frozen=True)
class UniverseAuditThresholds:
    max_missing_list_date_ratio: float = 0.001
    max_member_without_quote_ratio: float = 0.01
    require_delisted_history: bool = True
    require_historical_constituents: bool = False
    max_unexplained_daily_count_jump: float = 0.05
    require_quote_coverage: bool = False
    require_st_status: bool = False
    require_suspension_status: bool = False


@dataclass(frozen=True)
class UniverseAuditReport:
    passed: bool
    sessions: int
    symbol_count: int
    delisted_symbol_count: int
    missing_list_date_ratio: float
    max_daily_count_jump: float
    member_without_quote_ratio: float | None
    st_status_declared: bool
    suspension_status_declared: bool
    historical_constituents_declared: bool
    failures: tuple[str, ...]


class PointInTimeUniverseProvider:
    def __init__(self, records: Iterable[UniverseRecord]):
        self.records = tuple(sorted(records, key=lambda row: (row.instrument, row.effective_from)))
        if not self.records:
            raise ValueError("Point-in-time universe cannot be empty")
        self.version_id = self._version_id()

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, str, str]]) -> "PointInTimeUniverseProvider":
        return cls(
            UniverseRecord(str(symbol), pd.Timestamp(start), pd.Timestamp(end)) for symbol, start, end in records
        )

    @classmethod
    def from_qlib_instruments(cls, path: str | Path) -> "PointInTimeUniverseProvider":
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                records.append((parts[0], parts[1], parts[2]))
        return cls.from_records(records)

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return tuple(sorted({record.instrument for record in self.records}))

    def subset(self, symbols: Iterable[str]) -> "PointInTimeUniverseProvider":
        wanted = {str(symbol) for symbol in symbols}
        records = [record for record in self.records if record.instrument in wanted]
        if not records:
            raise ValueError("Universe subset has no matching instruments")
        return PointInTimeUniverseProvider(records)

    def members(self, session: str | pd.Timestamp) -> tuple[str, ...]:
        date = pd.Timestamp(session)
        return tuple(
            sorted(record.instrument for record in self.records if record.effective_from <= date <= record.effective_to)
        )

    def audit(
        self,
        sessions: Sequence[str | pd.Timestamp],
        thresholds: UniverseAuditThresholds | None = None,
        *,
        quoted_members_by_session: Mapping[str, Iterable[str]] | None = None,
        st_status_declared: bool = False,
        suspension_status_declared: bool = False,
        historical_constituents_declared: bool = False,
    ) -> UniverseAuditReport:
        thresholds = thresholds or UniverseAuditThresholds()
        dates = [pd.Timestamp(session) for session in sessions]
        counts = [len(self.members(date)) for date in dates]
        jumps = []
        for previous_date, current_date, previous_count, current_count in zip(dates, dates[1:], counts, counts[1:]):
            listings = sum(previous_date < record.effective_from <= current_date for record in self.records)
            delistings = sum(previous_date <= record.effective_to < current_date for record in self.records)
            expected_delta = listings - delistings
            unexplained_delta = (current_count - previous_count) - expected_delta
            jumps.append(abs(unexplained_delta) / max(previous_count, 1))
        max_jump = max(jumps, default=0.0)
        missing_dates = sum(pd.isna(record.effective_from) or pd.isna(record.effective_to) for record in self.records)
        missing_ratio = missing_dates / len(self.records)
        last_session = max(dates) if dates else max(record.effective_to for record in self.records)
        delisted = {record.instrument for record in self.records if record.effective_to < last_session}
        missing_quote_ratio = self._missing_quote_ratio(dates, quoted_members_by_session)
        failures = []
        if missing_ratio > thresholds.max_missing_list_date_ratio:
            failures.append("missing_list_date_ratio")
        if thresholds.require_delisted_history and not delisted:
            failures.append("missing_delisted_history")
        if max_jump > thresholds.max_unexplained_daily_count_jump:
            failures.append("daily_member_count_jump")
        if thresholds.require_quote_coverage and missing_quote_ratio is None:
            failures.append("quote_coverage_not_measured")
        elif missing_quote_ratio is not None and missing_quote_ratio > thresholds.max_member_without_quote_ratio:
            failures.append("member_without_quote_ratio")
        if thresholds.require_st_status and not st_status_declared:
            failures.append("st_status_not_declared")
        if thresholds.require_suspension_status and not suspension_status_declared:
            failures.append("suspension_status_not_declared")
        if thresholds.require_historical_constituents and not historical_constituents_declared:
            failures.append("historical_constituents_not_declared")
        return UniverseAuditReport(
            passed=not failures,
            sessions=len(dates),
            symbol_count=len(self.all_symbols),
            delisted_symbol_count=len(delisted),
            missing_list_date_ratio=missing_ratio,
            max_daily_count_jump=max_jump,
            member_without_quote_ratio=missing_quote_ratio,
            st_status_declared=st_status_declared,
            suspension_status_declared=suspension_status_declared,
            historical_constituents_declared=historical_constituents_declared,
            failures=tuple(failures),
        )

    def _missing_quote_ratio(
        self,
        dates: Sequence[pd.Timestamp],
        quoted_members_by_session: Mapping[str, Iterable[str]] | None,
    ) -> float | None:
        if quoted_members_by_session is None:
            return None
        member_count = 0
        missing_count = 0
        normalized_quotes = {
            pd.Timestamp(session).strftime("%Y-%m-%d"): set(symbols)
            for session, symbols in quoted_members_by_session.items()
        }
        for date in dates:
            members = set(self.members(date))
            quotes = normalized_quotes.get(date.strftime("%Y-%m-%d"), set())
            member_count += len(members)
            missing_count += len(members.difference(quotes))
        return missing_count / member_count if member_count else 0.0

    def _version_id(self) -> str:
        text = "\n".join(
            f"{record.instrument}|{record.effective_from.date()}|{record.effective_to.date()}"
            for record in self.records
        )
        return f"universe:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"
