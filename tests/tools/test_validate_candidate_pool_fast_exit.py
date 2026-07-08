"""Candidate-pool fast-exit validation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantx.tools.validate_candidate_pool_fast_exit import (
    FastExitRule,
    _load_candidate_rows,
    _soft_exit_reason,
    validate_candidates,
)


class FakeDailyBarLoader:
    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame]):
        self.bars_by_symbol = bars_by_symbol
        self.preloaded: tuple[list[str], str | None, str | None] | None = None

    def preload_symbols(self, symbols, start=None, end=None):
        self.preloaded = (sorted(symbols), start, end)

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        return self.bars_by_symbol.get(symbol, pd.DataFrame()).copy()


def test_validate_candidates_finds_fast_exit_on_virtual_candidate():
    bars = _bars_with_weak_early_path("SZ000001")
    loader = FakeDailyBarLoader({"SZ000001": bars})

    rows = validate_candidates(
        [{"symbol": "SZ000001", "signal_date": "2020-12-31", "score": 1.2}],
        loader,
        FastExitRule(),
        max_holding_days=45,
    )

    assert loader.preloaded is not None
    assert len(rows) == 1
    row = rows[0]
    assert row["fast_triggered"] is True
    assert row["fast_exit_reason"] == "fast_exit"
    assert row["fast_holding_days"] == 19
    assert row["baseline_exit_reason"] == "stop_loss_89permil"
    assert row["delta_return"] > 0


def test_load_candidate_rows_deduplicates_selected_candidates(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "daily_selection_candidates.json").write_text(
        """
        [
          {"date": "2021-01-01", "selected_candidates": [{"symbol": "SZ000001", "score": 1.0}, {"symbol": "SZ000001", "score": 2.0}]},
          {"date": "2021-01-04", "selected_candidates": [{"symbol": "SZ000001", "score": 3.0}]}
        ]
        """.strip(),
        encoding="utf-8",
    )

    rows = _load_candidate_rows(run_dir, source="selected", max_rows=None)

    assert rows == [
        {"symbol": "SZ000001", "signal_date": "2021-01-01", "score": 1.0},
        {"symbol": "SZ000001", "signal_date": "2021-01-04", "score": 3.0},
    ]


def test_soft_exit_uses_drawdown_from_peak():
    assert _soft_exit_reason(6, close_pnl=0.08, peak_pnl=0.22) == "trail_peak15_dd10"
    assert _soft_exit_reason(6, close_pnl=0.11, peak_pnl=0.22) is None


def _bars_with_weak_early_path(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=50)
    closes = [100.0]
    for i in range(1, 50):
        if i <= 3:
            close = 98.0
        elif i <= 10:
            close = 95.0
        elif i <= 20:
            close = 95.0
        elif i <= 25:
            close = 92.0
        else:
            close = 90.5
        closes.append(close)
    rows = []
    for date, close in zip(dates, closes):
        rows.append({
            "date": date,
            "symbol": symbol,
            "open": close,
            "high": close * 1.005,
            "low": 91.5 if date == dates[12] else close * 0.995,
            "close": close,
            "volume": 1000000,
        })
    return pd.DataFrame(rows)
