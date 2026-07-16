"""Config runner tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.predictions import PredictionRecord, PredictionStore
from quantx.core.engine import TransactionCost
from quantx.core.engine.engine import BacktestConfig
from quantx.tools.run_backtest import (
    build_cost,
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


def test_load_symbols_supports_named_wufu_etf_universe():
    config = {
        "data": {
            "provider_uri": "data/qlib_data_fixed",
            "universe": "wufu_etf",
            "start": "2021-01-01",
            "end": "2021-12-31",
        }
    }

    symbols = load_symbols(config)

    assert "SH518880" in symbols
    assert "SH513100" in symbols
    assert "SH511880" in symbols
    assert len(symbols) == len(set(symbols))


def test_load_symbols_can_resolve_frozen_prediction_universe(tmp_path):
    signal_time = MarketTime(
        "2024-01-02",
        SessionPhase.AFTER_CLOSE,
        datetime(2024, 1, 2, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    path = tmp_path / "predictions.json"
    checksum = PredictionStore(
        [
            PredictionRecord(signal_time, "SZ000001", 0.2, 5, "model", "fold", "schema"),
            PredictionRecord(signal_time, "SH600000", 0.1, 5, "model", "fold", "schema"),
        ]
    ).write(path)
    config = {
        "data": {"universe": "prediction_store", "extra_symbols": ["SH511880"]},
        "strategy": {"alpha": {"path": str(path), "checksum": checksum}},
    }

    assert load_symbols(config) == ["SH600000", "SZ000001", "SH511880"]


def test_load_symbols_can_limit_prediction_universe_to_daily_topk(tmp_path):
    first_time = MarketTime(
        "2024-01-02",
        SessionPhase.AFTER_CLOSE,
        datetime(2024, 1, 2, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    second_time = MarketTime(
        "2024-01-03",
        SessionPhase.AFTER_CLOSE,
        datetime(2024, 1, 3, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    path = tmp_path / "predictions.json"
    PredictionStore([
        PredictionRecord(first_time, "SZ000001", 0.9, 5, "model", "fold", "schema"),
        PredictionRecord(first_time, "SH600000", 0.8, 5, "model", "fold", "schema"),
        PredictionRecord(second_time, "SZ000002", 0.7, 5, "model", "fold", "schema"),
        PredictionRecord(second_time, "SH600001", 0.6, 5, "model", "fold", "schema"),
    ]).write(path)
    config = {
        "data": {"universe": "prediction_store", "prediction_pool_topk": 1, "extra_symbols": ["SH511880"]},
        "strategy": {"alpha": {"path": str(path), "artifact_id": "model"}},
    }

    assert load_symbols(config) == ["SZ000001", "SZ000002", "SH511880"]


def test_load_symbols_can_resolve_external_score_universe(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "date,instrument,score\n"
        "2025-01-02,SZ000001,0.2\n"
        "2025-01-02,SH600000,0.1\n"
        "2025-01-03,SZ000001,0.3\n",
        encoding="utf-8",
    )
    config = {
        "data": {"universe": "external_score", "extra_symbols": ["SH511880"]},
        "selector": {"path": str(path), "instrument_col": "instrument"},
    }

    assert load_symbols(config) == ["SH600000", "SZ000001", "SH511880"]


def test_load_symbols_external_score_universe_uses_selector_filters(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "date,instrument,score\n"
        "2025-01-02,SZ000001,0.9\n"
        "2025-01-02,SH600000,0.8\n"
        "2025-01-02,SZ000002,0.1\n"
        "2025-01-03,SZ000003,0.7\n",
        encoding="utf-8",
    )
    config = {
        "data": {"universe": "external_score"},
        "selector": {
            "path": str(path),
            "date_col": "date",
            "instrument_col": "instrument",
            "score_col": "score",
            "score_floor": 0.2,
            "sort": "score_desc",
            "topk": 1,
        },
    }

    assert load_symbols(config) == ["SZ000001", "SZ000003"]


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


def test_build_cost_accepts_asymmetric_slippage():
    cost = build_cost({
        "cost": {
            "slippage": 0.01,
            "buy_slippage": 0.003,
            "sell_slippage": 0.0,
        }
    })

    assert cost.slippage == 0.01
    assert cost.buy_slippage == 0.003
    assert cost.sell_slippage == 0.0


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
