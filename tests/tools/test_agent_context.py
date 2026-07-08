"""Agent context CLI tests."""

from quantx.tools.agent_context import (
    build_parser,
    cmd_data_update,
    cmd_latest_run,
    cmd_meta,
    cmd_metrics_compute,
    cmd_metrics_list,
    cmd_report,
    cmd_status,
    cmd_validate_config,
)


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_agent_context_status_reports_project():
    result = cmd_status(Args(root="."))

    assert result["ok"] is True
    assert result["has_run_backtest"] is True
    assert result["has_agent_context"] is True


def test_agent_context_parser_exposes_data_update():
    parser = build_parser()
    args = parser.parse_args(["data-update", "--dry-run", "--limit", "1"])

    assert args.command == "data-update"
    assert args.dry_run is True
    assert args.limit == 1


def test_agent_context_data_update_uses_adjustment_aware_updater(monkeypatch, tmp_path):
    calls = {}

    class FakeRepository:
        def __init__(self, raw_dir):
            calls["raw_dir"] = raw_dir

    class FakeConverter:
        def __init__(self, qlib_dir, csv_dir):
            calls["qlib_dir"] = qlib_dir
            calls["csv_dir"] = csv_dir

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
            return Args(to_dict=lambda: {
                "ok": True,
                "total_symbols": 1,
                "updated_symbols": [],
                "full_refresh_symbols": [],
                "unchanged_symbols": ["SH600000"],
                "failed_symbols": [],
                "dry_run": dry_run,
            })

    monkeypatch.setattr("quantx.tools.agent_context.LocalDataRepository", FakeRepository)
    monkeypatch.setattr("quantx.tools.agent_context.BaostockToQlibConverter", FakeConverter)
    monkeypatch.setattr("quantx.tools.agent_context.AdjustmentAwareIncrementalUpdater", FakeUpdater)

    result = cmd_data_update(Args(
        root=str(tmp_path),
        provider_uri="provider",
        raw_dir="raw",
        symbols=["SH600000"],
        end_date="2026-07-06",
        limit=1,
        overlap_days=120,
        tolerance=1e-5,
        full_refresh_start="2015-01-01",
        pause_seconds=0.1,
        socket_timeout=3,
        max_requests=123,
        progress_every=1,
        symbol_file=None,
        dry_run=True,
    ))

    assert result["ok"] is True
    assert calls["qlib_dir"] == str(tmp_path / "provider")
    assert calls["csv_dir"] == str(tmp_path / "raw" / "stocks")
    assert calls["raw_dir"] == str(tmp_path / "raw")
    assert calls["overlap_days"] == 120
    assert calls["tolerance"] == 1e-5
    assert calls["full_refresh_start"] == "2015-01-01"
    assert calls["max_requests"] == 123
    assert calls["update"]["symbols"] == ["SH600000"]
    assert calls["update"]["end"] == "2026-07-06"
    assert calls["update"]["limit"] == 1
    assert calls["update"]["dry_run"] is True
    assert callable(calls["update"]["progress_callback"])


def test_agent_context_meta_reads_name_and_industry():
    result = cmd_meta(Args(root=".", meta_uri="data/meta/quantx_meta.sqlite", symbols=["SH600000", "SZ920010"]))

    assert result["ok"] is True
    assert result["symbols"][0]["name"] == "浦发银行"
    assert result["symbols"][0]["industry_name"] == "银行"
    assert result["symbols"][1]["industry_name"] == "燃气"


def test_agent_context_validate_config_smoke():
    result = cmd_validate_config(
        Args(root=".", config="configs/strategies/shuijiao_legacy.yaml", symbol_limit=3)
    )

    assert result["ok"] is True
    assert result["data"]["symbols"] == 3


def test_agent_context_latest_report_and_metrics():
    latest = cmd_latest_run(Args(root="."))

    assert latest["ok"] is True
    report = cmd_report(Args(root=".", run_id="latest"))
    assert report["ok"] is True
    assert report["run_id"] == latest["run_id"]
    assert report["top_traded_symbols"]

    specs = cmd_metrics_list(Args(root="."))
    assert any(metric["name"] == "sortino" for metric in specs["metrics"])

    metrics = cmd_metrics_compute(Args(root=".", run_id="latest", include=["sortino", "calmar"], init_cash=None))
    assert metrics["ok"] is True
    assert set(metrics["metrics"]) == {"sortino", "calmar"}
