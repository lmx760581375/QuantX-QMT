"""Array-level execution feasibility rules for labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


BUY = 1
SELL = -1


@dataclass(frozen=True)
class ExecutionRulesV1:
    """Rules aligned with QuantX daily open execution constraints.

    The rule is intentionally pure and array-friendly so label generation can
    reuse the same behavior without constructing a backtest Exchange object for
    every instrument/date pair.
    """

    one_price_eps: float = 1e-6
    limit_eps: float = 1e-6

    @property
    def rule_hash(self) -> str:
        payload = {
            "name": "kronos_execution_rules_v1",
            "one_price_eps": self.one_price_eps,
            "limit_eps": self.limit_eps,
            "buy_blocks": ["suspended", "one_side_limit_up", "open_limit_up"],
            "sell_blocks": ["suspended", "one_side_limit_down", "close_limit_down"],
            "board_limits": {
                "SH60": 0.10,
                "SZ00": 0.10,
                "SZ30": 0.20,
                "SH68": 0.20,
                "other": 0.10,
            },
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(content).hexdigest()

    def limit_rate(self, instrument: str, *, is_st: bool = False) -> float:
        if is_st:
            return 0.05
        if instrument.startswith(("SZ30", "SH68")):
            return 0.20
        if instrument.startswith("BJ"):
            return 0.30
        return 0.10

    def is_one_price_bar(self, open_: float, high: float, low: float, close: float) -> bool:
        values = (open_, high, low, close)
        if any(not np.isfinite(v) or float(v) <= 0 for v in values):
            return False
        return max(abs(float(high) - float(low)), abs(float(open_) - float(close))) <= self.one_price_eps

    def can_buy(
        self,
        instrument: str,
        *,
        open_: float,
        high: float,
        low: float,
        close: float,
        preclose: float,
        volume: float,
        tradestatus: int,
        is_st: int,
        has_raw_row: int,
    ) -> tuple[bool, str]:
        base_ok, reason = self._base_bar_ok(open_, close, volume, tradestatus, is_st, has_raw_row)
        if not base_ok:
            return False, reason
        limit_rate = self.limit_rate(instrument, is_st=bool(is_st))
        if self.is_one_price_bar(open_, high, low, close) and _ret(close, preclose) >= 0.045 - self.limit_eps:
            return False, "one_side_limit_up"
        if np.isfinite(preclose) and preclose > 0 and float(open_) >= preclose * (1.0 + limit_rate) - self.limit_eps:
            return False, "limit_up"
        return True, "ok"

    def can_sell(
        self,
        instrument: str,
        *,
        open_: float,
        high: float,
        low: float,
        close: float,
        preclose: float,
        volume: float,
        tradestatus: int,
        is_st: int,
        has_raw_row: int,
    ) -> tuple[bool, str]:
        base_ok, reason = self._base_bar_ok(open_, close, volume, tradestatus, is_st, has_raw_row)
        if not base_ok:
            return False, reason
        limit_rate = self.limit_rate(instrument, is_st=bool(is_st))
        if self.is_one_price_bar(open_, high, low, close) and _ret(close, preclose) <= -0.045 + self.limit_eps:
            return False, "one_side_limit_down"
        if np.isfinite(preclose) and preclose > 0 and float(close) <= preclose * (1.0 - limit_rate) + self.limit_eps:
            return False, "limit_down"
        return True, "ok"

    @staticmethod
    def _base_bar_ok(open_: float, close: float, volume: float, tradestatus: int, is_st: int, has_raw_row: int) -> tuple[bool, str]:
        if int(has_raw_row) != 1:
            return False, "missing_raw_row"
        if int(tradestatus) != 1:
            return False, "suspended"
        if int(is_st) != 0:
            return False, "is_st"
        if not np.isfinite(open_) or float(open_) <= 0:
            return False, "invalid_open"
        if not np.isfinite(close) or float(close) <= 0:
            return False, "invalid_close"
        if not np.isfinite(volume) or float(volume) <= 0:
            return False, "zero_volume"
        return True, "ok"


def _ret(value: float, base: float) -> float:
    if not np.isfinite(value) or not np.isfinite(base) or float(base) <= 0:
        return 0.0
    return float(value) / float(base) - 1.0
