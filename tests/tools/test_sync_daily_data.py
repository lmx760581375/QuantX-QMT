"""sync_daily_data CLI tests."""

from quantx.tools import sync_daily_data


def test_sync_daily_data_cli_wires_adjustment_aware_updater(monkeypatch, tmp_path, capsys):
    calls = {}

    class FakeRepository:
        def __init__(self, raw_dir):
            calls["raw_dir"] = raw_dir

    class FakeConverter:
        def __init__(self, qlib_dir, csv_dir):
            calls["qlib_dir"] = qlib_dir
            calls["csv_dir"] = csv_dir

    class FakeReport:
        ok = True

        def to_dict(self):
            return {
                "ok": True,
                "total_symbols": 1,
                "updated_symbols": [],
                "full_refresh_symbols": [],
                "unchanged_symbols": ["SZ000001"],
                "skipped_symbols": [],
                "failed_symbols": [],
                "dry_run": True,
                "request_count": 1,
                "max_requests": 321,
                "budget_exhausted": False,
            }

    class FakeUpdater:
        def __init__(self, repository, converter, client_factory, overlap_days, tolerance, full_refresh_start, max_requests):
            calls["overlap_days"] = overlap_days
            calls["tolerance"] = tolerance
            calls["full_refresh_start"] = full_refresh_start
            calls["max_requests"] = max_requests

        def update(self, symbols, end, limit, dry_run, progress_callback):
            calls["update"] = {
                "symbols": symbols,
                "end": end,
                "limit": limit,
                "dry_run": dry_run,
                "progress_callback": progress_callback,
            }
            return FakeReport()

    monkeypatch.setattr(sync_daily_data, "LocalDataRepository", FakeRepository)
    monkeypatch.setattr(sync_daily_data, "BaostockToQlibConverter", FakeConverter)
    monkeypatch.setattr(sync_daily_data, "AdjustmentAwareIncrementalUpdater", FakeUpdater)

    rc = sync_daily_data.main([
        "--provider-uri",
        str(tmp_path / "qlib"),
        "--raw-dir",
        str(tmp_path / "raw"),
        "--symbols",
        "SZ000001",
        "--limit",
        "1",
        "--end-date",
        "2026-07-06",
        "--overlap-days",
        "120",
        "--tolerance",
        "0.00001",
        "--full-refresh-start",
        "2015-01-01",
        "--max-requests",
        "321",
        "--dry-run",
        "--json",
    ])

    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out
    assert calls["qlib_dir"] == str(tmp_path / "qlib")
    assert calls["csv_dir"] == str(tmp_path / "raw" / "stocks")
    assert calls["raw_dir"] == str(tmp_path / "raw")
    assert calls["overlap_days"] == 120
    assert calls["tolerance"] == 1e-5
    assert calls["full_refresh_start"] == "2015-01-01"
    assert calls["max_requests"] == 321
    assert calls["update"]["symbols"] == ["SZ000001"]
    assert calls["update"]["end"] == "2026-07-06"
    assert calls["update"]["limit"] == 1
    assert calls["update"]["dry_run"] is True
    assert callable(calls["update"]["progress_callback"])
