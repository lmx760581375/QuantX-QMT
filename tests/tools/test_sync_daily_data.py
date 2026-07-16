"""sync_daily_data CLI tests."""

import json
from types import SimpleNamespace

from quantx.tools import sync_daily_data


class _FakeProviderLock:
    def __init__(self, provider_uri, shared=False):
        self.provider_uri = provider_uri
        self.shared = shared

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDataVersion:
    def to_dict(self):
        return {"version": "fake"}


class _FakeDataVersionResolver:
    def resolve(self, *args, **kwargs):
        return _FakeDataVersion()


def test_load_symbols_from_standard_strategy_config(tmp_path):
    config = tmp_path / "strategy.yaml"
    config.write_text(
        "data:\n"
        "  provider_uri: data/qlib_data_fixed\n"
        "  universe: wufu_etf\n"
        "  start: 2016-01-04\n"
        "  end: latest\n",
        encoding="utf-8",
    )

    symbols = sync_daily_data._load_symbols(SimpleNamespace(
        strategy_config=str(config),
        symbols=None,
        symbol_file=None,
    ))

    assert symbols is not None
    assert "SH518880" in symbols
    assert "SH511880" in symbols


def test_resolve_turnover_start_uses_metadata_overlap(tmp_path):
    provider = tmp_path / "qlib"
    metadata_dir = provider / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "turnover_features.json").write_text(json.dumps({"end": "2026-07-15"}), encoding="utf-8")

    start = sync_daily_data._resolve_turnover_start(
        str(provider),
        explicit_start=None,
        full_refresh_start="2010-01-01",
        end="2026-07-16",
        overlap_days=90,
    )

    assert start == "2026-04-16"


def test_resolve_turnover_start_honors_explicit_start(tmp_path):
    start = sync_daily_data._resolve_turnover_start(
        str(tmp_path / "qlib"),
        explicit_start="2026-07-01",
        full_refresh_start="2010-01-01",
        end="2026-07-16",
        overlap_days=90,
    )

    assert start == "2026-07-01"


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


def test_sync_daily_data_qmt_updates_turnover_features_by_default(monkeypatch, tmp_path, capsys):
    calls = {"turnover": []}

    class FakeRepository:
        def __init__(self, raw_dir):
            pass

    class FakeConverter:
        def __init__(self, qlib_dir, csv_dir):
            pass

    class FakeReport:
        ok = True

        def to_dict(self):
            return {
                "ok": True,
                "total_symbols": 1,
                "updated_symbols": ["SH600000"],
                "full_refresh_symbols": [],
                "unchanged_symbols": [],
                "skipped_symbols": [],
                "failed_symbols": [],
                "dry_run": False,
                "request_count": 1,
                "max_requests": 321,
                "budget_exhausted": False,
            }

    class FakeUpdater:
        def __init__(self, **kwargs):
            pass

        def update(self, symbols, end, limit, dry_run, progress_callback):
            return FakeReport()

    def fake_build_turnover_features(**kwargs):
        calls["turnover"].append(kwargs)
        return {"ok": True, "written_symbols": 1}

    import quantx.tools.build_turnover_features as turnover_module

    monkeypatch.setattr(sync_daily_data, "LocalDataRepository", FakeRepository)
    monkeypatch.setattr(sync_daily_data, "BaostockToQlibConverter", FakeConverter)
    monkeypatch.setattr(sync_daily_data, "AdjustmentAwareIncrementalUpdater", FakeUpdater)
    monkeypatch.setattr(sync_daily_data, "ProviderLock", _FakeProviderLock)
    monkeypatch.setattr(sync_daily_data, "DataVersionResolver", _FakeDataVersionResolver)
    monkeypatch.setattr(sync_daily_data, "write_data_version_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(turnover_module, "build_turnover_features", fake_build_turnover_features)

    provider_uri = tmp_path / "qlib"
    rc = sync_daily_data.main([
        "--source",
        "qmt",
        "--provider-uri",
        str(provider_uri),
        "--raw-dir",
        str(tmp_path / "raw"),
        "--symbols",
        "SH600000",
        "--limit",
        "1",
        "--end-date",
        "2026-07-15",
        "--max-requests",
        "321",
        "--json",
    ])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"turnover_features"' in output
    assert len(calls["turnover"]) == 1
    assert calls["turnover"][0]["provider_uri"] == provider_uri
    assert calls["turnover"][0]["output_provider_uri"] == provider_uri
    assert calls["turnover"][0]["end"] == "2026-07-15"


def test_sync_daily_data_can_skip_default_qmt_turnover_update(monkeypatch, tmp_path, capsys):
    calls = {"turnover": []}

    class FakeRepository:
        def __init__(self, raw_dir):
            pass

    class FakeConverter:
        def __init__(self, qlib_dir, csv_dir):
            pass

    class FakeReport:
        ok = True

        def to_dict(self):
            return {
                "ok": True,
                "total_symbols": 1,
                "updated_symbols": ["SH600000"],
                "full_refresh_symbols": [],
                "unchanged_symbols": [],
                "skipped_symbols": [],
                "failed_symbols": [],
                "dry_run": False,
                "request_count": 1,
                "max_requests": 321,
                "budget_exhausted": False,
            }

    class FakeUpdater:
        def __init__(self, **kwargs):
            pass

        def update(self, symbols, end, limit, dry_run, progress_callback):
            return FakeReport()

    import quantx.tools.build_turnover_features as turnover_module

    monkeypatch.setattr(sync_daily_data, "LocalDataRepository", FakeRepository)
    monkeypatch.setattr(sync_daily_data, "BaostockToQlibConverter", FakeConverter)
    monkeypatch.setattr(sync_daily_data, "AdjustmentAwareIncrementalUpdater", FakeUpdater)
    monkeypatch.setattr(sync_daily_data, "ProviderLock", _FakeProviderLock)
    monkeypatch.setattr(sync_daily_data, "DataVersionResolver", _FakeDataVersionResolver)
    monkeypatch.setattr(sync_daily_data, "write_data_version_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(turnover_module, "build_turnover_features", lambda **kwargs: calls["turnover"].append(kwargs))

    rc = sync_daily_data.main([
        "--source",
        "qmt",
        "--provider-uri",
        str(tmp_path / "qlib"),
        "--raw-dir",
        str(tmp_path / "raw"),
        "--symbols",
        "SH600000",
        "--skip-turnover-features",
        "--json",
    ])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"turnover_features"' not in output
    assert calls["turnover"] == []
