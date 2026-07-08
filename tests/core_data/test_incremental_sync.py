"""Adjustment-aware incremental update tests."""

from __future__ import annotations

import pandas as pd

from quantx.core.data.incremental_sync import AdjustmentAwareIncrementalUpdater
from quantx.core.data.repository import LocalDataRepository


def test_incremental_update_merges_new_rows_when_overlap_aligned(tmp_path):
    repo = LocalDataRepository(tmp_path / "raw")
    repo.replace_symbol("SZ000001", _bars("SZ000001", ["2021-01-01", "2021-01-02", "2021-01-03"], [10.0, 11.0, 12.0]))
    converter = _FakeConverter()
    updater = AdjustmentAwareIncrementalUpdater(
        repo,
        converter,
        client_factory=lambda: _FakeClient({
            ("sz.000001", "2021-01-02", "2021-01-04"): _bars(
                "SZ000001", ["2021-01-02", "2021-01-03", "2021-01-04"], [11.0, 12.0, 13.0],
                precloses=[10.0, 11.0, 12.0],
            ),
        }),
        overlap_days=2,
    )

    report = updater.update(symbols=["SZ000001"], end="2021-01-04")

    loaded = repo.load_symbol("SZ000001")
    assert report.ok is True
    assert report.symbol_results[0].status == "incremental"
    assert report.symbol_results[0].adjustment_changed is False
    assert repo.get_last_date("SZ000001") == "2021-01-04"
    assert loaded["close"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert converter.updated_symbols == ["SZ000001"]


def test_incremental_update_full_refreshes_symbol_when_overlap_adjustment_changed(tmp_path):
    repo = LocalDataRepository(tmp_path / "raw")
    repo.replace_symbol("SZ000001", _bars("SZ000001", ["2021-01-01", "2021-01-02", "2021-01-03"], [10.0, 11.0, 12.0]))
    converter = _FakeConverter()
    updater = AdjustmentAwareIncrementalUpdater(
        repo,
        converter,
        client_factory=lambda: _FakeClient({
            ("sz.000001", "2021-01-02", "2021-01-04"): _bars(
                "SZ000001", ["2021-01-02", "2021-01-03", "2021-01-04"], [10.5, 11.5, 12.5],
                precloses=[9.5, 10.5, 11.5],
            ),
            ("sz.000001", "2021-01-01", "2021-01-04"): _bars(
                "SZ000001", ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"], [9.5, 10.5, 11.5, 12.5],
                precloses=[9.5, 9.5, 10.5, 11.5],
            ),
        }),
        overlap_days=2,
    )

    report = updater.update(symbols=["SZ000001"], end="2021-01-04")

    loaded = repo.load_symbol("SZ000001")
    assert report.ok is True
    assert report.symbol_results[0].status == "full_refresh"
    assert report.symbol_results[0].adjustment_changed is True
    assert "close" in report.symbol_results[0].mismatch_columns
    assert loaded["close"].tolist() == [9.5, 10.5, 11.5, 12.5]
    assert converter.updated_symbols == ["SZ000001"]


def test_incremental_update_returns_structured_failure_when_client_login_fails(tmp_path):
    repo = LocalDataRepository(tmp_path / "raw")
    repo.replace_symbol("SZ000001", _bars("SZ000001", ["2021-01-01"], [10.0]))
    converter = _FakeConverter()
    updater = AdjustmentAwareIncrementalUpdater(
        repo,
        converter,
        client_factory=lambda: _FailingClient(),
        overlap_days=2,
    )

    report = updater.update(symbols=["SZ000001"], end="2021-01-04", dry_run=True)

    assert report.ok is False
    assert report.failed_symbols == ["SZ000001"]
    assert report.symbol_results[0].status == "failed"
    assert "client error" in report.symbol_results[0].message
    assert converter.updated_symbols == []


def test_incremental_update_skips_when_end_date_before_local_overlap(tmp_path):
    repo = LocalDataRepository(tmp_path / "raw")
    repo.replace_symbol("SZ000001", _bars("SZ000001", ["2021-01-10", "2021-01-11", "2021-01-12"], [10.0, 11.0, 12.0]))
    converter = _FakeConverter()
    client = _UnexpectedQueryClient()
    updater = AdjustmentAwareIncrementalUpdater(
        repo,
        converter,
        client_factory=lambda: client,
        overlap_days=2,
    )

    report = updater.update(symbols=["SZ000001"], end="2021-01-01", dry_run=True)

    assert report.ok is True
    assert report.unchanged_symbols == ["SZ000001"]
    assert report.symbol_results[0].status == "unchanged"
    assert "earlier than local overlap" in report.symbol_results[0].message
    assert client.queries == []
    assert converter.updated_symbols == []


def test_incremental_update_stops_before_request_budget_is_exceeded(tmp_path):
    repo = LocalDataRepository(tmp_path / "raw")
    repo.replace_symbol("SZ000001", _bars("SZ000001", ["2021-01-01"], [10.0]))
    repo.replace_symbol("SZ000002", _bars("SZ000002", ["2021-01-01"], [10.0]))
    converter = _FakeConverter()
    client = _UnexpectedQueryClient()
    updater = AdjustmentAwareIncrementalUpdater(
        repo,
        converter,
        client_factory=lambda: client,
        overlap_days=1,
        max_requests=0,
    )

    report = updater.update(symbols=["SZ000001", "SZ000002"], end="2021-01-04")

    assert report.ok is False
    assert report.budget_exhausted is True
    assert report.request_count == 0
    assert report.skipped_symbols == ["SZ000001", "SZ000002"]
    assert client.queries == []
    assert converter.updated_symbols == []


def _bars(symbol, dates, closes, precloses=None):
    rows = []
    for idx, (date, close) in enumerate(zip(dates, closes)):
        preclose = precloses[idx] if precloses is not None else (closes[idx - 1] if idx else close)
        rows.append({
            "date": date,
            "code": symbol,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "preclose": preclose,
            "volume": 1000,
            "amount": close * 1000 * 100,
            "turnover": 0.01,
            "pct_chg": 0.0,
            "is_st": 0,
        })
    return pd.DataFrame(rows)


class _FakeClient:
    def __init__(self, responses):
        self.responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def query_history_k_data_with_retry(self, symbol, start_date, end_date):
        return self.responses.get((symbol, start_date, end_date), pd.DataFrame())


class _FailingClient:
    def __enter__(self):
        raise ConnectionError("login failed")

    def __exit__(self, *args):
        return False


class _UnexpectedQueryClient:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def query_history_k_data_with_retry(self, *args):
        self.queries.append(args)
        raise AssertionError("client should not be queried")


class _FakeConverter:
    def __init__(self):
        self.updated_symbols = []

    def convert_incremental(self, symbols):
        self.updated_symbols = list(symbols)
