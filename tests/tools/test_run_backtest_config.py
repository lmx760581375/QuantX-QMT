"""Config runner tests."""

import pytest

from quantx.core.engine import TransactionCost
from quantx.core.engine.engine import BacktestConfig
from quantx.tools.run_backtest import (
    build_backtest_config,
    dry_run_config,
    load_symbols,
    resolve_config_dates,
    validate_backtest_data_coverage,
)


def test_dry_run_config_compiles_shuijiao_yaml():
    result = dry_run_config("configs/strategies/shuijiao_legacy.yaml", symbol_limit=5)

    assert result["ok"] is True
    assert result["strategy"]["name"] == "shuijiao_legacy_formula"
    assert "buy_signal" in result["strategy"]["formula_order"]
    assert result["data"]["symbols"] == 5
    assert result["engine"]["error_policy"] == "fail_fast"


def test_load_symbols_supports_mainboard_universe(tmp_path):
    provider = tmp_path / "provider"
    instruments = provider / "instruments"
    instruments.mkdir(parents=True)
    (instruments / "all.txt").write_text(
        "SH600000 2000-01-01 2099-12-31\n"
        "SH688001 2000-01-01 2099-12-31\n"
        "SZ000001 2000-01-01 2099-12-31\n"
        "SZ399299 2000-01-01 2099-12-31\n"
        "SZ300001 2000-01-01 2099-12-31\n"
        "SZ301001 2000-01-01 2099-12-31\n",
        encoding="utf-8",
    )

    config = {
        "data": {
            "provider_uri": str(provider),
            "universe": "all_mainboard",
            "start": "2021-01-01",
            "end": "2021-12-31",
        }
    }

    assert load_symbols(config) == ["SH600000", "SZ000001"]


def test_load_symbols_excludes_indices_from_all_a(tmp_path):
    provider = tmp_path / "provider"
    instruments = provider / "instruments"
    instruments.mkdir(parents=True)
    (instruments / "all.txt").write_text(
        "SZ399299 2000-01-01 2099-12-31\n"
        "SH000300 2000-01-01 2099-12-31\n"
        "SZ300001 2000-01-01 2099-12-31\n"
        "SH688001 2000-01-01 2099-12-31\n",
        encoding="utf-8",
    )

    config = {
        "data": {
            "provider_uri": str(provider),
            "universe": "all_a",
            "start": "2021-01-01",
            "end": "2021-12-31",
        }
    }

    assert load_symbols(config) == ["SZ300001", "SH688001"]


def test_build_backtest_config_prefers_execution_deal_price():
    cfg = build_backtest_config({
        "data": {
            "provider_uri": "data/qlib_data_fixed",
            "start": "2021-01-01",
            "end": "2021-12-31",
        },
        "execution": {"deal_price": "hlc3"},
        "engine": {"deal_price": "close"},
    }, TransactionCost())

    assert cfg.deal_price == "hlc3"


def test_resolve_config_dates_supports_latest_end(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2021-01-04\n2021-01-05\n", encoding="utf-8")

    config = {
        "data": {
            "provider_uri": str(provider),
            "start": "2021-01-01",
            "end": "latest",
        }
    }

    resolved = resolve_config_dates(config)
    cfg = build_backtest_config(resolved, TransactionCost())

    assert resolved["data"]["end"] == "2021-01-05"
    assert cfg.end_date == "2021-01-05"


def test_validate_backtest_data_coverage_rejects_missing_lookback(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2020-01-10\n2020-01-13\n2020-01-14\n",
        encoding="utf-8",
    )

    cfg = BacktestConfig(
        provider_uri=str(provider),
        start_date="2020-01-10",
        end_date="2020-01-14",
        look_back_days=30,
    )

    with pytest.raises(RuntimeError, match="look_back_days"):
        validate_backtest_data_coverage(cfg)


def test_validate_backtest_data_coverage_allows_weekend_gap(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2020-01-06\n2020-01-07\n2020-01-08\n",
        encoding="utf-8",
    )

    cfg = BacktestConfig(
        provider_uri=str(provider),
        start_date="2020-01-06",
        end_date="2020-01-08",
        look_back_days=2,
    )

    coverage = validate_backtest_data_coverage(cfg)

    assert coverage["load_start"] == "2020-01-04"
    assert coverage["first_available_date"] == "2020-01-06"
