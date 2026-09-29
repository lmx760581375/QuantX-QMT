"""Config-driven strategy unit tests."""

import pandas as pd
import pytest

from quantx.core.factor_runtime import FactorRuntime, MarketPanel
from quantx.core.strategy.base import (
    AccountSnapshot,
    PolicyState,
    PositionPlan,
    PositionSnapshot,
    Signal,
    StockSelection,
    WeightAllocation,
)
from quantx.core.engine.cost import TransactionCost
from quantx.core.strategy.config_strategy import (
    ConfigStrategyError,
    EqualWeightRebalance,
    FormulaSelector,
    RuleExecution,
    ScalarRuleEvaluator,
    WatchlistFormulaSelector,
    build_formula_strategy,
    compile_strategy_config,
)
class _Context:
    def __init__(self, runtime):
        self.factor_runtime = runtime
        self.trade_dates = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
        self._daily_selection_candidates = {}

    def record_daily_selection_candidates(self, date, detail):
        self._daily_selection_candidates[date] = dict(detail)

    def get_daily_selection_candidates(self):
        return [
            self._daily_selection_candidates[date]
            for date in sorted(self._daily_selection_candidates)
        ]


def _runtime():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0],
        "$high": [1.1, 2.1, 3.1, 1.1, 4.1, 5.1, 6.1, 1.1],
        "$low": [0.9, 1.9, 2.9, 0.9, 3.9, 4.9, 5.9, 0.9],
    }, index=index)
    return FactorRuntime(MarketPanel.from_frame(frame))


def test_formula_selector_uses_config_formula_with_lag():
    runtime = _runtime()
    context = _Context(runtime)
    selector = FormulaSelector(
        formulas={"buy_signal": "close > 2", "score": "close"},
        where="buy_signal",
        score="score",
        lag=1,
        sort="score_desc",
    )
    selector.prepare(context)

    selection = selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))

    assert [signal.symbol for signal in selection.signals] == ["SZ000001"]
    assert selection.signals[0].score == 3.0


def test_formula_selector_records_daily_candidates_for_visualization():
    runtime = _runtime()
    context = _Context(runtime)
    selector = FormulaSelector(
        formulas={"buy_signal": "close > 2", "score": "close"},
        where="buy_signal",
        score="score",
        lag=1,
        sort="score_desc",
        topk=1,
    )

    selector.prepare(context)

    rows = context.get_daily_selection_candidates()
    assert [row["date"] for row in rows] == context.trade_dates
    assert rows[-1]["mode"] == "daily_signal"
    assert rows[-1]["date"] == "2021-01-07"
    assert rows[-1]["signal_date"] == "2021-01-07"
    assert rows[-1]["raw_candidate_count"] == 1
    assert rows[-1]["selected_candidates"][0]["symbol"] == "SZ000001"


def test_formula_selector_external_score_overrides_formula_sort(tmp_path):
    score_path = tmp_path / "scores.csv"
    pd.DataFrame([
        {"signal_date": "2021-01-05", "instrument": "SZ000001", "ml_score": 0.10},
        {"signal_date": "2021-01-05", "instrument": "SZ000002", "ml_score": 0.90},
    ]).to_csv(score_path, index=False)
    runtime = _runtime()
    context = _Context(runtime)
    selector = FormulaSelector(
        formulas={"buy_signal": "close > 0", "score": "close"},
        where="buy_signal",
        score="score",
        lag=1,
        sort="score_desc",
        external_score={"path": str(score_path), "score_col": "ml_score", "missing": "drop"},
    )
    selector.prepare(context)

    selection = selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))

    assert [signal.symbol for signal in selection.signals] == ["SZ000002", "SZ000001"]
    assert [signal.score for signal in selection.signals] == pytest.approx([0.90, 0.10])


def test_formula_selector_external_score_missing_drop_filters_candidates(tmp_path):
    score_path = tmp_path / "scores.csv"
    pd.DataFrame([
        {"signal_date": "2021-01-05", "instrument": "SZ000002", "ml_score": 0.90},
    ]).to_csv(score_path, index=False)
    runtime = _runtime()
    context = _Context(runtime)
    selector = FormulaSelector(
        formulas={"buy_signal": "close > 0", "score": "close"},
        where="buy_signal",
        score="score",
        lag=1,
        sort="score_desc",
        external_score={"path": str(score_path), "score_col": "ml_score", "missing": "drop"},
    )
    selector.prepare(context)

    selection = selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))

    assert [signal.symbol for signal in selection.signals] == ["SZ000002"]
    assert selection.signals[0].score == pytest.approx(0.90)


def test_compile_strategy_config_rejects_lag_zero():
    with pytest.raises(ConfigStrategyError, match="selector.lag must be >= 1"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "lag": 0},
            "execution": {},
        })


def test_formula_strategy_loads_explicit_position_manager_plugin(tmp_path):
    (tmp_path / "test_position_plugin.py").write_text(
        "from quantx.core.strategy.base import PositionManager, PositionPlan\n"
        "class TestManager(PositionManager):\n"
        "    def act(self, state, selection):\n"
        "        return PositionPlan()\n"
        "def build_position_manager(config):\n"
        "    return TestManager()\n",
        encoding="utf-8",
    )
    config = {
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 0"},
        "selector": {"where": "buy_signal", "score": "close", "topk": 1},
        "rebalance": {"max_positions": 1},
        "execution": {
            "deal_price": "close",
            "position_manager": {
                "type": "python_plugin",
                "python_path": str(tmp_path),
                "factory": "test_position_plugin:build_position_manager",
                "config": {},
            },
        },
    }

    strategy = build_formula_strategy(config, TransactionCost())

    assert type(strategy.position_manager).__name__ == "TestManager"


def test_formula_selector_rejects_lag_zero():
    with pytest.raises(ConfigStrategyError, match="selector.lag must be >= 1"):
        FormulaSelector(
            formulas={"buy_signal": "close > 1"},
            where="buy_signal",
            lag=0,
        )


def test_watchlist_selector_waits_for_confirmation_after_setup():
    runtime = _runtime()
    context = _Context(runtime)
    selector = WatchlistFormulaSelector(
        formulas={"setup": "close > 2", "confirm": "close >= 4", "score": "close"},
        setup="setup",
        confirm="confirm",
        score="score",
        lag=1,
        max_age=2,
        min_age=1,
        sort="score_desc",
        topk=1,
    )
    selector.prepare(context)

    first = selector.act(PolicyState(date="2021-01-05", market_data=pd.DataFrame(), context=context))
    second = selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))

    assert first.signals == []
    assert [signal.symbol for signal in second.signals] == ["SZ000001"]
    assert second.signals[0].score == 4.0


def test_watchlist_selector_drops_expired_setups():
    runtime = _runtime()
    context = _Context(runtime)
    selector = WatchlistFormulaSelector(
        formulas={"setup": "close > 2", "confirm": "close >= 4", "score": "close"},
        setup="setup",
        confirm="confirm",
        score="score",
        lag=1,
        max_age=1,
        min_age=1,
        sort="score_desc",
        topk=1,
    )
    selector.prepare(context)

    selector.act(PolicyState(date="2021-01-05", market_data=pd.DataFrame(), context=context))
    selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))
    selection = selector.act(PolicyState(date="2021-01-07", market_data=pd.DataFrame(), context=context))

    assert selection.signals == []


def test_watchlist_selector_confirm_can_use_watchlist_pnl_state():
    runtime = _runtime()
    context = _Context(runtime)
    selector = WatchlistFormulaSelector(
        formulas={"setup": "close > 2", "score": "close"},
        setup="setup",
        confirm="watchlist_age >= 1 and watchlist_peak_pnl > 0.20 and watchlist_pnl > 0",
        score="score",
        lag=1,
        max_age=2,
        min_age=1,
        sort="score_desc",
        topk=1,
    )
    selector.prepare(context)

    selector.act(PolicyState(date="2021-01-05", market_data=pd.DataFrame(), context=context))
    selection = selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))

    assert [signal.symbol for signal in selection.signals] == ["SZ000001"]


def test_watchlist_drawdown_from_peak_uses_peak_relative_ratio():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [1.0, 10.0, 20.0, 15.0],
        "$high": [1.1, 10.1, 20.1, 15.1],
        "$low": [0.9, 9.9, 19.9, 14.9],
    }, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    context = _Context(runtime)
    selector = WatchlistFormulaSelector(
        formulas={"setup": "close > 2", "score": "close"},
        setup="setup",
        confirm="watchlist_age >= 2 and watchlist_drawdown_from_peak > -0.30 and watchlist_drawdown_from_peak < -0.20",
        score="score",
        lag=1,
        max_age=3,
        min_age=1,
        sort="score_desc",
        topk=1,
    )
    selector.prepare(context)

    selector.act(PolicyState(date="2021-01-05", market_data=pd.DataFrame(), context=context))
    selector.act(PolicyState(date="2021-01-06", market_data=pd.DataFrame(), context=context))
    selection = selector.act(PolicyState(date="2021-01-07", market_data=pd.DataFrame(), context=context))

    assert [signal.symbol for signal in selection.signals] == ["SZ000001"]


def test_build_formula_strategy_watchlist_disables_precompute():
    strategy = build_formula_strategy({
        "name": "watchlist_demo",
        "fields": {"close": "$close"},
        "signals": {"setup": "close > 1", "confirm": "close > 2"},
        "selector": {
            "mode": "watchlist",
            "where": "setup",
            "confirm": "confirm",
            "score": "close",
            "lag": 1,
            "watchlist": {"max_age": 3},
        },
        "rebalance": {"type": "equal_weight", "max_positions": 1},
        "execution": {},
    }, cost=None)

    assert strategy.precompute_stock_signals is False


def test_compile_strategy_config_requires_watchlist_confirm():
    with pytest.raises(ConfigStrategyError, match="selector.confirm is required"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"setup": "close > 1"},
            "selector": {"mode": "watchlist", "where": "setup"},
            "execution": {},
        })


def test_scalar_rule_evaluator_supports_position_state():
    evaluator = ScalarRuleEvaluator()

    assert evaluator.evaluate("pnl_pct > 0.20 and not sell_signal", {
        "pnl_pct": 0.21,
        "sell_signal": False,
    })


def test_compile_strategy_config_returns_formula_dag():
    spec = compile_strategy_config({
        "name": "demo",
        "fields": {"my_close": "$close"},
        "factors": {"ma": "Mean(my_close, 2)"},
        "signals": {"buy_signal": "my_close > ma"},
        "selector": {"where": "buy_signal", "score": "my_close", "sort": "score_asc"},
        "rebalance": {"type": "equal_weight", "max_positions": 3},
        "execution": {
            "sell_rules": [{"name": "take_profit", "when": "pnl_pct > 0.1", "action": "sell_all"}],
            "buy": {"sizing": "cash_equal"},
        },
    })

    assert spec.fields == {"my_close": "$close"}
    assert spec.formula_order == ["ma", "buy_signal"]
    assert spec.dependencies["buy_signal"] == ["ma", "my_close"]
    assert spec.selector["sort"] == "score_asc"


def test_compile_strategy_config_preserves_external_score():
    external_score = {
        "path": "/tmp/model_scores.csv",
        "date_col": "signal_date",
        "instrument_col": "instrument",
        "score_col": "ml_score",
        "missing": "drop",
    }
    spec = compile_strategy_config({
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 1"},
        "selector": {"where": "buy_signal", "score": "close", "external_score": external_score},
        "execution": {},
    })

    assert spec.selector["external_score"] == external_score


def test_compile_strategy_config_accepts_partial_sell_rule():
    spec = compile_strategy_config({
        "name": "demo",
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 1"},
        "selector": {"where": "buy_signal", "score": "close"},
        "execution": {
            "sell_rules": [{
                "name": "scale_half",
                "when": "pnl_pct > 0.12 and remaining_position_pct > 0.50",
                "action": "sell_to_position_pct",
                "position_pct": 0.5,
            }],
        },
    })

    assert spec.execution["sell_rules"][0]["action"] == "sell_to_position_pct"


def test_compile_strategy_config_rejects_partial_sell_without_position_pct():
    with pytest.raises(ConfigStrategyError, match="requires position_pct"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "score": "close"},
            "execution": {
                "sell_rules": [{
                    "name": "scale_half",
                    "when": "pnl_pct > 0.12",
                    "action": "sell_to_position_pct",
                }],
            },
        })


def test_compile_strategy_config_preserves_rank_weights():
    spec = compile_strategy_config({
        "name": "rank_weight_demo",
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 1"},
        "selector": {"where": "buy_signal", "score": "close"},
        "rebalance": {
            "type": "equal_weight",
            "max_positions": 3,
            "rank_weights": [1.0, 0.5, 0.25],
        },
        "execution": {},
    })

    assert spec.rebalance["rank_weights"] == [1.0, 0.5, 0.25]


def test_compile_strategy_config_accepts_portfolio_target_sizing():
    spec = compile_strategy_config({
        "name": "portfolio_target_demo",
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 1"},
        "selector": {"where": "buy_signal", "score": "close"},
        "rebalance": {
            "type": "equal_weight",
            "max_positions": 10,
            "weight_scope": "portfolio_target",
        },
        "execution": {
            "max_position_weight": 0.12,
            "buy": {"sizing": "target_weight"},
        },
    })

    assert spec.rebalance["weight_scope"] == "portfolio_target"
    assert spec.execution["max_position_weight"] == 0.12


def test_compile_strategy_config_accepts_partial_position_slot_exemption():
    spec = compile_strategy_config({
        "name": "partial_runner_demo",
        "fields": {"close": "$close"},
        "signals": {"buy_signal": "close > 1"},
        "selector": {"where": "buy_signal", "score": "close"},
        "rebalance": {
            "type": "equal_weight",
            "max_positions": 5,
            "exclude_partial_positions_from_slots": True,
        },
        "execution": {},
    })

    assert spec.rebalance["exclude_partial_positions_from_slots"] is True


def test_compile_strategy_config_rejects_non_boolean_partial_position_slot_exemption():
    with pytest.raises(
        ConfigStrategyError,
        match="exclude_partial_positions_from_slots must be boolean",
    ):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "score": "close"},
            "rebalance": {
                "max_positions": 5,
                "exclude_partial_positions_from_slots": "yes",
            },
            "execution": {},
        })


def test_equal_weight_rebalance_can_exclude_partial_position_from_slots():
    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=300,
                initial_quantity=1000,
                avg_cost=10.0,
                market_value=3000.0,
            ),
        },
    )
    selection = StockSelection(signals=[
        Signal(symbol="SZ000002", score=2.0),
        Signal(symbol="SZ000003", score=1.0),
    ])

    default_rebalance = EqualWeightRebalance(
        max_positions=1,
        weight_scope="portfolio_target",
    )
    runner_rebalance = EqualWeightRebalance(
        max_positions=1,
        weight_scope="portfolio_target",
        exclude_partial_positions_from_slots=True,
    )

    assert default_rebalance.act(state, selection).weights == {}
    assert runner_rebalance.act(state, selection).weights == {"SZ000002": 1.0}


def test_equal_weight_rebalance_can_release_slot_for_planned_partial_exit():
    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=1000,
                initial_quantity=1000,
                avg_cost=10.0,
                market_value=10000.0,
            ),
        },
        extra={"planned_slot_exempt_symbols": {"SZ000001"}},
    )
    selection = StockSelection(signals=[Signal(symbol="SZ000002", score=1.0)])
    rebalance = EqualWeightRebalance(
        max_positions=1,
        weight_scope="portfolio_target",
        exclude_partial_positions_from_slots=True,
    )

    assert rebalance.act(state, selection).weights == {"SZ000002": 1.0}


def test_compile_strategy_config_rejects_target_sizing_without_portfolio_scope():
    with pytest.raises(ConfigStrategyError, match="requires rebalance.weight_scope=portfolio_target"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "score": "close"},
            "rebalance": {"type": "equal_weight", "max_positions": 10},
            "execution": {"buy": {"sizing": "target_weight"}},
        })


def test_compile_strategy_config_rejects_negative_rank_weights():
    with pytest.raises(ConfigStrategyError, match="rebalance.rank_weights must be non-negative"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "score": "close"},
            "rebalance": {
                "type": "equal_weight",
                "max_positions": 2,
                "rank_weights": [1.0, -0.1],
            },
            "execution": {},
        })


def test_equal_weight_rebalance_can_use_rank_tier_weights():
    rebalance = EqualWeightRebalance(max_positions=4, rank_weights=[1.0, 1.0, 0.25])
    state = PolicyState(date="2021-01-04", market_data=pd.DataFrame())
    selection = StockSelection(signals=[
        Signal("SZ000001", 4.0),
        Signal("SZ000002", 3.0),
        Signal("SZ000003", 2.0),
        Signal("SZ000004", 1.0),
    ])

    allocation = rebalance.act(state, selection)

    assert allocation.weights == pytest.approx({
        "SZ000001": 1.0 / 2.5,
        "SZ000002": 1.0 / 2.5,
        "SZ000003": 0.25 / 2.5,
        "SZ000004": 0.25 / 2.5,
    })


def test_equal_weight_rebalance_portfolio_target_uses_total_slot_count():
    rebalance = EqualWeightRebalance(max_positions=10, weight_scope="portfolio_target")
    positions = {
        f"SZ00000{i}": PositionSnapshot(
            f"SZ00000{i}", quantity=100, avg_cost=10.0, market_value=1000.0,
        )
        for i in range(1, 10)
    }
    state = PolicyState(date="2021-01-04", market_data=pd.DataFrame(), positions=positions)
    selection = StockSelection(signals=[Signal("SZ000010", 1.0)])

    allocation = rebalance.act(state, selection)

    assert allocation.weights == pytest.approx({"SZ000010": 0.10})


def test_equal_weight_rebalance_can_use_account_drawdown_rule():
    rebalance = EqualWeightRebalance(
        max_positions=5,
        max_positions_rules=[{"when": "account_drawdown > -0.10", "value": 3}],
    )
    selection = StockSelection(signals=[
        Signal("SZ000001", 3.0),
        Signal("SZ000002", 2.0),
        Signal("SZ000003", 1.0),
    ])
    positions = {
        "SH600000": PositionSnapshot("SH600000", quantity=100, avg_cost=10.0, market_value=1000.0),
        "SH600001": PositionSnapshot("SH600001", quantity=100, avg_cost=10.0, market_value=1000.0),
    }

    near_high = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0, drawdown=-0.05),
        positions=positions,
    )
    deep_drawdown = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0, drawdown=-0.20),
        positions=positions,
    )

    assert len(rebalance.act(near_high, selection).weights) == 1
    assert len(rebalance.act(deep_drawdown, selection).weights) == 3


def test_build_formula_strategy_injects_industry_group_from_config(tmp_path):
    industry_csv = tmp_path / "industry.csv"
    pd.DataFrame([
        {"symbol": "SZ000001", "industry_code": "bank"},
        {"symbol": "SZ000002", "industry_code": "tech"},
    ]).to_csv(industry_csv, index=False)
    runtime = _runtime()
    context = _Context(runtime)
    strategy = build_formula_strategy({
        "fields": {"close": "$close"},
        "groups": {"industry_l1": {"source": "meta.industry_l1", "path": str(industry_csv)}},
        "group_factors": {"industry_rank": "GroupRank(close, industry_l1)"},
        "signals": {"buy_signal": "industry_rank > 0"},
        "selector": {"where": "buy_signal", "score": "industry_rank", "lag": 1},
        "rebalance": {"type": "equal_weight", "max_positions": 2},
        "execution": {},
    }, cost=None)

    strategy.selector.prepare(context)

    assert "industry_l1" in runtime.values
    assert "industry_rank" in runtime.values


def test_build_formula_strategy_injects_concept_membership_from_config(tmp_path):
    concept_csv = tmp_path / "sector.csv"
    pd.DataFrame([
        {"symbol": "SZ000001", "sector_type": "concept", "sector_code": "BK1"},
        {"symbol": "SZ000001", "sector_type": "concept", "sector_code": "BK2"},
        {"symbol": "SZ000002", "sector_type": "concept", "sector_code": "BK2"},
    ]).to_csv(concept_csv, index=False)
    runtime = _runtime()
    context = _Context(runtime)
    strategy = build_formula_strategy({
        "fields": {"close": "$close"},
        "groups": {"concept": {"source": "meta.concept", "path": str(concept_csv)}},
        "group_factors": {"concept_strength": "GroupMean(close, concept, agg='max')"},
        "signals": {"buy_signal": "concept_strength > 0"},
        "selector": {"where": "buy_signal", "score": "concept_strength", "lag": 1},
        "rebalance": {"type": "equal_weight", "max_positions": 2},
        "execution": {},
    }, cost=None)

    strategy.selector.prepare(context)

    assert runtime.values["concept"].shape == (2, 2)
    assert "concept_strength" in runtime.values


def test_compile_strategy_config_rejects_unknown_formula_variable():
    with pytest.raises(ConfigStrategyError, match="unknown variables"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "missing_factor > close"},
            "selector": {"where": "buy_signal"},
            "execution": {},
        })


def test_compile_strategy_config_rejects_unknown_deal_price():
    with pytest.raises(ConfigStrategyError, match="execution.deal_price"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal"},
            "execution": {"deal_price": "missing_price"},
        })


def test_compile_strategy_config_rejects_unknown_buy_rule_variable():
    with pytest.raises(ConfigStrategyError, match="Buy rule references unknown variables"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal"},
            "execution": {"buy": {"when": "missing_factor > 0"}},
        })


def test_compile_strategy_config_rejects_unknown_add_existing_rule_variable():
    with pytest.raises(ConfigStrategyError, match="Buy add_existing_when rule references unknown variables"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal"},
            "execution": {"buy": {"add_existing_when": "missing_factor > 0"}},
        })


def test_rule_execution_buy_when_filters_current_day_buys():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    market_data = pd.DataFrame({
        "$close": [10.0, 10.0],
        "$open": [9.0, 11.0],
    }, index=["SZ000001", "SZ000002"])
    state = PolicyState(
        date="2021-01-04",
        market_data=market_data,
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        context=_RuntimeContext(),
    )
    execution = RuleExecution([], cost=None, deal_price="close", buy_when="close > open")

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 0.5, "SZ000002": 0.5}))

    assert [order.symbol for order in orders.orders] == ["SZ000001"]


def test_rule_execution_add_existing_when_scans_holdings_outside_allocation():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    positions = {
        "SZ000001": PositionSnapshot(
            "SZ000001",
            quantity=1000,
            avg_cost=8.0,
            market_value=10000.0,
            weight=0.10,
            holding_days=10,
            highest_price=11.0,
            lowest_price=8.0,
        ),
        "SZ000002": PositionSnapshot(
            "SZ000002",
            quantity=1000,
            avg_cost=10.0,
            market_value=10000.0,
            weight=0.10,
            holding_days=3,
            highest_price=10.2,
            lowest_price=9.8,
        ),
    }
    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0, 10.0]}, index=["SZ000001", "SZ000002"]),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        positions=positions,
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [],
        cost=None,
        deal_price="close",
        buy_when="not is_holding",
        add_existing_when="holding_days >= 5 and peak_pnl_pct >= 0.20 and position_weight < 0.30",
    )

    orders = execution.act(state, WeightAllocation(weights={}))

    assert [order.symbol for order in orders.orders] == ["SZ000001"]


def test_rebalance_allows_existing_holding_adds_when_slots_are_full():
    positions = {
        "SZ000001": PositionSnapshot("SZ000001", quantity=1000, avg_cost=10.0, market_value=10000.0),
        "SZ000002": PositionSnapshot("SZ000002", quantity=1000, avg_cost=10.0, market_value=10000.0),
    }
    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(),
        positions=positions,
    )
    selection = StockSelection(signals=[
        Signal("SZ000001", score=3.0),
        Signal("SZ000003", score=2.0),
        Signal("SZ000002", score=1.0),
    ])
    rebalance = EqualWeightRebalance(max_positions=2, buy_only_new_positions=False)

    allocation = rebalance.act(state, selection)

    assert set(allocation.weights) == {"SZ000001", "SZ000002"}


def test_rule_execution_buy_when_can_gate_adds_to_existing_positions():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002", "SZ000003"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    positions = {
        "SZ000001": PositionSnapshot(
            "SZ000001",
            quantity=1000,
            avg_cost=8.0,
            market_value=10000.0,
            weight=0.10,
            holding_days=8,
            highest_price=11.0,
            lowest_price=8.0,
        ),
        "SZ000002": PositionSnapshot(
            "SZ000002",
            quantity=1000,
            avg_cost=10.0,
            market_value=10000.0,
            weight=0.10,
            holding_days=3,
            highest_price=10.2,
            lowest_price=9.8,
        ),
    }
    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0, 10.0, 10.0]}, index=["SZ000001", "SZ000002", "SZ000003"]),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        positions=positions,
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [],
        cost=None,
        deal_price="close",
        skip_if_holding=False,
        buy_when=(
            "not is_holding or "
            "(holding_days >= 5 and peak_pnl_pct >= 0.20 and "
            "drawdown_from_peak > -0.10 and position_weight < 0.30)"
        ),
    )

    orders = execution.act(
        state,
        WeightAllocation(weights={"SZ000001": 1.0, "SZ000002": 1.0, "SZ000003": 1.0}),
    )

    assert [order.symbol for order in orders.orders] == ["SZ000001", "SZ000003"]


def test_rule_execution_uses_allocation_weights_for_buy_sizing():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0, 10.0]}, index=["SZ000001", "SZ000002"]),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        context=_RuntimeContext(),
    )
    execution = RuleExecution([], cost=None, deal_price="close", cash_use_ratio=1.0)

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 0.8, "SZ000002": 0.2}))

    quantities = {order.symbol: order.quantity for order in orders.orders}
    assert quantities == {"SZ000001": 8000, "SZ000002": 2000}


def test_rule_execution_target_weight_sizes_replacement_from_total_value():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0, 10.0]}, index=["SZ000001", "SZ000002"]),
        account=AccountSnapshot(cash=0.0, total_value=100000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001", quantity=1000, avg_cost=10.0, market_value=10000.0,
                holding_days=20,
            ),
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{"name": "time_stop", "when": "holding_days >= 20", "action": "sell_all"}],
        cost=TransactionCost(commission_rate=0.0, stamp_tax_rate=0.0, slippage=0.0),
        deal_price="close",
        sizing="target_weight",
        cash_use_ratio=1.0,
        reuse_sell_cash=True,
    )

    orders = execution.act(state, WeightAllocation(weights={"SZ000002": 0.10}))

    assert [(order.action.name, order.symbol, order.quantity) for order in orders.orders] == [
        ("SELL", "SZ000001", 1000),
        ("BUY", "SZ000002", 1000),
    ]


def test_rule_execution_max_position_weight_does_not_trim_existing_position_by_default():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=100000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001", quantity=2000, avg_cost=10.0, market_value=20000.0,
            ),
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution([], cost=None, deal_price="close", max_position_weight=0.12)

    orders = execution.act(state, WeightAllocation(weights={}))

    assert orders.orders == []


def test_rule_execution_can_explicitly_trim_excess_position():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=100000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001", quantity=2000, avg_cost=10.0, market_value=20000.0,
            ),
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [],
        cost=None,
        deal_price="close",
        max_position_weight=0.12,
        trim_overweight_positions=True,
    )

    orders = execution.act(state, WeightAllocation(weights={}))

    assert [(order.action.name, order.symbol, order.quantity, order.reason) for order in orders.orders] == [
        ("SELL", "SZ000001", 800, "position_weight_cap"),
    ]


def test_position_plan_full_exit_releases_slot_and_reuses_sell_cash():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0, 10.0]}, index=["SZ000001", "SZ000002"]),
        account=AccountSnapshot(cash=0.0, total_value=10000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001", quantity=1000, avg_cost=10.0, market_value=10000.0,
            ),
        },
        context=_RuntimeContext(),
    )
    state.extra["position_plan"] = PositionPlan(
        sell_ratios={"SZ000001": 1.0},
        entry_weights={"SZ000002": 1.0},
        cash_deploy_ratio=1.0,
    )
    execution = RuleExecution(
        [],
        cost=TransactionCost(commission_rate=0.0, stamp_tax_rate=0.0, slippage=0.0),
        deal_price="close",
        reuse_sell_cash=True,
    )

    orders = execution.act(state, WeightAllocation(weights={"SZ000002": 1.0}))

    assert [(order.action.name, order.symbol) for order in orders.orders] == [
        ("SELL", "SZ000001"),
        ("BUY", "SZ000002"),
    ]
    assert orders.orders[0].quantity == 1000
    assert orders.orders[1].quantity == 900


def test_position_plan_partial_exit_does_not_release_a_slot_or_rebuy_holding():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=10000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001", quantity=1000, avg_cost=10.0, market_value=10000.0,
            ),
        },
        context=_RuntimeContext(),
    )
    state.extra["position_plan"] = PositionPlan(
        sell_ratios={"SZ000001": 0.50},
        entry_weights={"SZ000001": 1.0},
        cash_deploy_ratio=1.0,
    )
    execution = RuleExecution([], cost=TransactionCost(), deal_price="close", reuse_sell_cash=True)

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 1.0}))

    assert [(order.action.name, order.symbol, order.quantity) for order in orders.orders] == [
        ("SELL", "SZ000001", 500),
    ]


def test_rule_execution_cash_use_ratio_rules_scale_new_buys():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame(
            {"$close": [10.0, 10.0], "market_breadth20": [0.30, 0.30]},
            index=["SZ000001", "SZ000002"],
        ),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [],
        cost=None,
        deal_price="close",
        cash_use_ratio=1.0,
        cash_use_ratio_rules=[{"when": "market_breadth20 < 0.35", "value": 0.40}],
    )

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 0.5, "SZ000002": 0.5}))

    quantities = {order.symbol: order.quantity for order in orders.orders}
    assert quantities == {"SZ000001": 2000, "SZ000002": 2000}


def test_rule_execution_sell_rule_can_use_position_peak_state():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [11.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [11.0]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=11000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=1000,
                avg_cost=10.0,
                market_value=11000.0,
                holding_days=6,
                highest_price=13.0,
                lowest_price=9.5,
            )
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{"name": "trail_peak", "when": "peak_pnl_pct > 0.20 and drawdown_from_peak < -0.10"}],
        cost=None,
        deal_price="close",
    )

    orders = execution.act(state, WeightAllocation(weights={}))

    assert len(orders.orders) == 1
    assert orders.orders[0].symbol == "SZ000001"
    assert orders.orders[0].reason == "trail_peak"


def test_rule_execution_partial_sell_to_position_pct_uses_initial_quantity():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [11.3]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [11.3]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=11300.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=1000,
                avg_cost=10.0,
                market_value=11300.0,
                holding_days=6,
                highest_price=11.3,
                lowest_price=10.0,
                initial_quantity=1000,
            )
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{
            "name": "scale_half_12pct",
            "when": "pnl_pct > 0.12 and remaining_position_pct > 0.50",
            "action": "sell_to_position_pct",
            "position_pct": 0.5,
        }],
        cost=None,
        deal_price="close",
    )

    orders = execution.act(state, WeightAllocation(weights={}))

    assert len(orders.orders) == 1
    assert orders.orders[0].symbol == "SZ000001"
    assert orders.orders[0].quantity == 500
    assert orders.orders[0].reason == "scale_half_12pct"
    assert execution.preview_sell_symbols(state) == []


def test_rule_execution_partial_sell_does_not_rebuy_same_holding():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [11.3, 10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [11.3, 10.0]}, index=["SZ000001", "SZ000002"]),
        account=AccountSnapshot(cash=100000.0, total_value=111300.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=1000,
                avg_cost=10.0,
                market_value=11300.0,
                holding_days=6,
                highest_price=11.3,
                lowest_price=10.0,
                initial_quantity=1000,
            )
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{
            "name": "scale_half_12pct",
            "when": "pnl_pct > 0.12 and remaining_position_pct > 0.50",
            "action": "sell_to_position_pct",
            "position_pct": 0.5,
        }],
        cost=None,
        deal_price="close",
        cash_use_ratio=1.0,
    )

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 0.5, "SZ000002": 0.5}))

    assert [(order.action.name, order.symbol, order.quantity) for order in orders.orders] == [
        ("SELL", "SZ000001", 500),
        ("BUY", "SZ000002", 10000),
    ]


def test_rule_execution_prepare_adds_formula_deal_price_to_quote():
    runtime = _runtime()
    runtime.compute_formulas({"hlc3": "(high + low + close) / 3"})
    quote = pd.DataFrame({
        "$close": runtime.panel.get("close").reshape(-1),
    }, index=pd.MultiIndex.from_product(
        [runtime.panel.dates, runtime.panel.instruments],
        names=["datetime", "instrument"],
    ))

    class _Exchange:
        def __init__(self):
            self.quote = quote.copy()

    class _RuntimeContext:
        def __init__(self):
            self.factor_runtime = runtime
            self.exchange = _Exchange()

    context = _RuntimeContext()
    execution = RuleExecution([], cost=None, deal_price="hlc3")

    execution.prepare(context)

    assert "$hlc3" in context.exchange.quote.columns
    assert context.exchange.quote.loc[(pd.Timestamp("2021-01-04"), "SZ000001"), "$hlc3"] == pytest.approx(1.0)
    assert context.exchange.quote.loc[(pd.Timestamp("2021-01-06"), "SZ000002"), "$hlc3"] == pytest.approx(5.0)


def test_compile_strategy_config_accepts_position_context_sell_rules():
    spec = compile_strategy_config({
        "fields": {"close": "$close"},
        "signals": {
            "buy_signal": "close > 1",
            "pre_5_return": "close / Ref(close, 5) - 1",
        },
        "selector": {"where": "buy_signal", "score": "close"},
        "execution": {
            "sell_rules": [{
                "name": "context_stop_wait",
                "when": "entry_pre_5_return <= 0 and hold_first_3d_return >= -0.01 and pnl_pct < -0.08",
            }],
        },
    })

    assert spec.execution["sell_rules"][0]["name"] == "context_stop_wait"


def test_compile_strategy_config_rejects_unknown_entry_context_base():
    with pytest.raises(ConfigStrategyError, match="unknown variables"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 1"},
            "selector": {"where": "buy_signal", "score": "close"},
            "execution": {
                "sell_rules": [{"name": "bad_context", "when": "entry_missing_factor > 0"}],
            },
        })


def test_rule_execution_snapshots_entry_context_on_buy_orders():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [10.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

        def is_limit_up(self, symbol, date):
            return False

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-04",
        market_data=pd.DataFrame({"$close": [10.0], "pre_5_return": [-0.02]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=100000.0, total_value=100000.0),
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{"name": "future_context_rule", "when": "entry_pre_5_return <= 0 and pnl_pct < -0.08"}],
        cost=None,
        deal_price="close",
    )

    orders = execution.act(state, WeightAllocation(weights={"SZ000001": 1.0}))

    assert len(orders.orders) == 1
    assert orders.orders[0].action.name == "BUY"
    assert orders.orders[0].context["entry_pre_5_return"] == pytest.approx(-0.02)


def test_rule_execution_sell_rule_can_use_position_context():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-10"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    quote = pd.DataFrame({"$close": [9.0]}, index=index)

    class _Exchange:
        def __init__(self):
            self.quote = quote

    class _RuntimeContext:
        def __init__(self):
            self.exchange = _Exchange()

    state = PolicyState(
        date="2021-01-10",
        market_data=pd.DataFrame({"$close": [9.0]}, index=["SZ000001"]),
        account=AccountSnapshot(cash=0.0, total_value=9000.0),
        positions={
            "SZ000001": PositionSnapshot(
                symbol="SZ000001",
                quantity=1000,
                avg_cost=10.0,
                market_value=9000.0,
                holding_days=6,
                highest_price=10.5,
                lowest_price=9.0,
                initial_quantity=1000,
                context={
                    "entry_pre_5_return": -0.02,
                    "entry_pre_20_above_ma20_ratio": 1.0,
                    "hold_first_3d_return": 0.01,
                },
            )
        },
        context=_RuntimeContext(),
    )
    execution = RuleExecution(
        [{
            "name": "context_stop_loss_wait",
            "when": "pnl_pct < -0.08 and entry_pre_5_return <= 0 and entry_pre_20_above_ma20_ratio >= 1 and hold_first_3d_return >= -0.01",
        }],
        cost=None,
        deal_price="close",
    )

    orders = execution.act(state, WeightAllocation(weights={}))

    assert [(order.action.name, order.symbol, order.reason) for order in orders.orders] == [
        ("SELL", "SZ000001", "context_stop_loss_wait")
    ]


def test_compile_strategy_config_rejects_bitwise_boolean_ops():
    with pytest.raises(Exception, match="Use and/or/not"):
        compile_strategy_config({
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "(close > 1) & (close < 2)"},
            "selector": {"where": "buy_signal"},
            "execution": {},
        })
