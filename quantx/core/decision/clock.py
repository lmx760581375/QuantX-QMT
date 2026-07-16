"""Market-session clocks used to enforce decision and execution ordering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import total_ordering


class SessionPhase(str, Enum):
    PRE_OPEN = "pre_open"
    OPEN = "open"
    INTRADAY = "intraday"
    CLOSE = "close"
    AFTER_CLOSE = "after_close"


_PHASE_ORDER = {
    SessionPhase.PRE_OPEN: 0,
    SessionPhase.OPEN: 1,
    SessionPhase.INTRADAY: 2,
    SessionPhase.CLOSE: 3,
    SessionPhase.AFTER_CLOSE: 4,
}


@total_ordering
@dataclass(frozen=True)
class MarketTime:
    session: str
    phase: SessionPhase
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("MarketTime timestamp must be timezone-aware")
        timezone_name = (
            getattr(self.timestamp.tzinfo, "key", None)
            or getattr(self.timestamp.tzinfo, "zone", None)
            or str(self.timestamp.tzinfo)
        )
        if timezone_name != "Asia/Shanghai":
            raise ValueError("MarketTime timestamp must use Asia/Shanghai timezone")
        try:
            session_date = datetime.strptime(self.session, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid market session: {self.session}") from exc
        if session_date != self.timestamp.date():
            raise ValueError("MarketTime session must match timestamp date")

    def _sort_key(self) -> tuple[datetime, int]:
        return self.timestamp, _PHASE_ORDER[self.phase]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MarketTime):
            return NotImplemented
        return self._sort_key() < other._sort_key()


@dataclass(frozen=True)
class DecisionClock:
    observation_end: MarketTime
    data_available_at: MarketTime
    signal_time: MarketTime
    execute_not_before: MarketTime

    def __post_init__(self) -> None:
        if self.observation_end > self.data_available_at:
            raise ValueError("observation_end cannot be after data_available_at")
        if self.data_available_at > self.signal_time:
            raise ValueError("data_available_at cannot be after signal_time")
        if self.signal_time >= self.execute_not_before:
            raise ValueError("execute_not_before must be after signal_time")
