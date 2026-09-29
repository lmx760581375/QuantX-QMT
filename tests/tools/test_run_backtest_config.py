"""Config runner tests."""

from quantx.core.engine import TransactionCost
from quantx.core.strategy.config_strategy import explain_strategy_config
from quantx.tools.run_backtest import build_backtest_config, load_config, load_symbols, resolve_config_dates


def test_dry_run_config_compiles_weak_to_strong_yaml():
    result = explain_strategy_config(load_config("configs/strategies/weak_to_strong.yaml"))

    assert result["name"] == "weak_to_strong_pos5_common_cost_2020_2026_mainboard"
    assert "buy_signal" in result["formula_order"]
    assert result["rebalance"]["max_positions"] == 5


def test_dry_run_config_compiles_reward_h15_yaml():
    result = explain_strategy_config(load_config("configs/strategies/reward_h15.yaml"))

    assert result["name"] == "reward_h7_top5_h15_only_2020_2026_all_a"
    assert result["selector"]["external_score"]["score_col"] == "reward_return_topn_risk_score_7d"
    assert result["rebalance"]["max_positions"] == 5


def test_dry_run_config_compiles_reward_h7_runner_yaml():
    result = explain_strategy_config(load_config("configs/strategies/reward_h7_runner.yaml"))

    assert result["name"] == "reward_v1_epoch009_top5_runner"
    assert result["selector"]["external_score"]["score_col"] == "reward_score_7d"
    assert result["rebalance"]["max_positions"] == 5
    assert result["rebalance"]["exclude_partial_positions_from_slots"] is True
    assert result["execution"]["trim_overweight_positions"] is False


def test_dry_run_config_compiles_reward_h2_event_yaml():
    result = explain_strategy_config(load_config("configs/strategies/reward_h2_event.yaml"))

    assert result["name"] == "reward_h2_event_epoch018_top5_no_high_gap"
    assert result["selector"]["external_score"]["score_col"] == "reward_score_2d"
    assert result["rebalance"]["max_positions"] == 10
    assert result["execution"]["buy_deal_price"] == "open"
    assert result["execution"]["sell_deal_price"] == "close"


def test_dry_run_config_compiles_weak_to_strong_pos3_yaml():
    result = explain_strategy_config(load_config("configs/strategies/weak_to_strong_pos3.yaml"))

    assert result["name"] == "weak_to_strong_pos3_common_cost_2020_2026_mainboard"
    assert result["selector"]["topk"] == 4
    assert result["rebalance"]["max_positions"] == 3


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


def test_load_symbols_supports_all_a_without_star_market(tmp_path):
    provider = tmp_path / "provider"
    instruments = provider / "instruments"
    instruments.mkdir(parents=True)
    (instruments / "all.txt").write_text(
        "SH600000 2000-01-01 2099-12-31\n"
        "SH688001 2000-01-01 2099-12-31\n"
        "SZ000001 2000-01-01 2099-12-31\n"
        "SZ300001 2000-01-01 2099-12-31\n"
        "SZ301001 2000-01-01 2099-12-31\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "provider_uri": str(provider),
            "universe": "all_a_no_star",
            "start": "2021-01-01",
            "end": "2021-12-31",
        }
    }

    assert load_symbols(config) == ["SH600000", "SZ000001", "SZ300001", "SZ301001"]


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


def test_build_backtest_config_preserves_board_aware_price_jump_limit():
    cfg = build_backtest_config({
        "data": {
            "provider_uri": "data/qlib_data_fixed",
            "start": "2021-01-01",
            "end": "2021-12-31",
        },
        "execution": {"price_jump_limit": "board_limit"},
    }, TransactionCost())

    assert cfg.price_jump_limit == "board_limit"


def test_build_backtest_config_accepts_historical_st_and_directional_prices():
    cfg = build_backtest_config({
        "data": {
            "provider_uri": "data/qlib_data_fixed",
            "start": "2021-01-01",
            "end": "2021-12-31",
        },
        "execution": {
            "buy_deal_price": "open",
            "sell_deal_price": "close",
        },
        "engine": {
            "historical_st_path": "artifacts/reference/historical_st/historical_st_daily.parquet",
        },
    }, TransactionCost())

    assert cfg.buy_deal_price == "open"
    assert cfg.sell_deal_price == "close"
    assert cfg.historical_st_path.endswith("historical_st_daily.parquet")


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
