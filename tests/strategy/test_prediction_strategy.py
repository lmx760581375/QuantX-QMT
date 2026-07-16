"""Frozen-prediction strategy integration with the standard strategy contract."""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.predictions import PredictionRecord, PredictionStore
from quantx.core.engine.cost import TransactionCost
from quantx.core.engine.engine import BacktestConfig, BacktestEngine
from quantx.core.engine.types import OrderAction
from quantx.core.strategy.base import AccountSnapshot, PolicyState
from quantx.core.strategy.base import PositionSnapshot
from quantx.core.strategy.factory import build_strategy, explain_strategy


def write_predictions(path):
    signal_time = MarketTime(
        "2026-07-10",
        SessionPhase.AFTER_CLOSE,
        datetime(2026, 7, 10, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    PredictionStore(
        [
            PredictionRecord(signal_time, "SZ000001", 0.9, 5, "ridge-v1", "fold-1", "schema-1"),
            PredictionRecord(signal_time, "SH600000", 0.4, 5, "ridge-v1", "fold-1", "schema-1"),
        ]
    ).write(path)


def config(path):
    return {
        "name": "prediction_test",
        "strategy": {
            "type": "decision_pipeline",
            "alpha": {
                "type": "predictions",
                "path": str(path),
                "artifact_id": "ridge-v1",
                "feature_schema_hash": "schema-1",
            },
            "portfolio": {"type": "topk_equal", "top_k": 1},
            "risk": {"type": "noop"},
            "order_planner": {"type": "standard", "lot_size": 100},
        },
        "execution": {"deal_price": "open"},
    }


def config_with_exit(path):
    cfg = config(path)
    cfg["strategy"]["exit"] = {"max_loss_pct": -0.10}
    return cfg


def config_with_buy_filter(path, **buy_filter):
    cfg = config(path)
    cfg["strategy"]["buy_filter"] = buy_filter
    return cfg


def test_prediction_strategy_emits_prior_session_signal_and_open_order(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config(path), TransactionCost())
    quote = pd.DataFrame(
        {"$open": [20.0, 10.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(board=SimpleNamespace()),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    selection = strategy.get_stock_signal(state)
    orders = strategy.get_trade_signal(state, selection).orders

    assert [(signal.symbol, signal.signal_date) for signal in selection.signals] == [
        ("SZ000001", "2026-07-10"),
        ("SH600000", "2026-07-10"),
    ]
    assert len(orders) == 1
    assert orders[0].symbol == "SZ000001"
    assert orders[0].action == OrderAction.BUY
    assert orders[0].date == "2026-07-13"


def test_prediction_strategy_explain_references_frozen_artifact(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)

    explanation = explain_strategy(config(path))

    assert explanation["strategy_type"] == "decision_pipeline"
    assert explanation["artifacts"] == ["ridge-v1"]
    assert explanation["execution"]["event_mode"] == "strict_event"


def test_prediction_strategy_skips_predictions_inside_rebalance_interval(tmp_path):
    path = tmp_path / "predictions.json"
    first_time = MarketTime(
        "2026-07-10",
        SessionPhase.AFTER_CLOSE,
        datetime(2026, 7, 10, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    second_time = MarketTime(
        "2026-07-13",
        SessionPhase.AFTER_CLOSE,
        datetime(2026, 7, 13, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    PredictionStore(
        [
            PredictionRecord(first_time, "SZ000001", 0.9, 5, "ridge-v1", "fold-1", "schema-1"),
            PredictionRecord(second_time, "SZ000001", 0.8, 5, "ridge-v1", "fold-1", "schema-1"),
        ]
    ).write(path)
    strategy_config = config(path)
    strategy_config["strategy"]["rebalance_interval_sessions"] = 5
    strategy = build_strategy(strategy_config, TransactionCost())
    quote = pd.DataFrame({"$open": [20.0]}, index=["SZ000001"])
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13", "2026-07-14"],
        exchange=SimpleNamespace(board=SimpleNamespace()),
        record_selection_candidates=lambda *args: None,
    )
    strategy.on_init(context)

    first_state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )
    second_state = PolicyState(
        date="2026-07-14",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    first_orders = strategy.get_trade_signal(first_state, strategy.get_stock_signal(first_state))
    second_orders = strategy.get_trade_signal(second_state, strategy.get_stock_signal(second_state))

    assert len(first_orders.orders) == 1
    assert second_orders.orders == []


def test_prediction_strategy_marks_missing_prediction_session(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config(path), TransactionCost())
    captured = []
    quote = pd.DataFrame({"$open": [20.0]}, index=["SZ000001"])
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13", "2026-07-14"],
        exchange=SimpleNamespace(board=SimpleNamespace()),
        record_selection_candidates=lambda _date, detail: captured.append(detail),
    )
    state = PolicyState(
        date="2026-07-14",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    selection = strategy.get_stock_signal(state)

    assert selection.signals == []
    assert captured[-1]["signal_date"] == "2026-07-13"
    assert captured[-1]["status"] == "missing_prediction"
    assert captured[-1]["latest_available_signal_date"] == "2026-07-10"


def test_prediction_strategy_does_not_trade_zero_volume_placeholder_bar(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config(path), TransactionCost())
    quote = pd.DataFrame(
        {"$open": [20.0, 10.0], "$volume": [0.0, 1000.0], "$amount": [0.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(board=SimpleNamespace()),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_does_not_buy_one_side_limit_up_bar(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config(path), TransactionCost())
    quote = pd.DataFrame(
        {
            "$open": [20.0, 10.0],
            "$high": [20.0, 10.2],
            "$low": [20.0, 9.8],
            "$close": [20.0, 10.1],
            "$volume": [1000.0, 1000.0],
            "$amount": [20_000.0, 10_000.0],
        },
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(
            board=SimpleNamespace(),
            get_preclose=lambda symbol, date: 19.0 if symbol == "SZ000001" else 10.0,
        ),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_excludes_st_symbols_from_new_buys(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(
        config_with_buy_filter(path, exclude_st=True, meta_store_uri=str(tmp_path / "missing.sqlite")),
        TransactionCost(),
    )
    quote = pd.DataFrame(
        {"$open": [20.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [20_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(board=SimpleNamespace(), get_preclose=lambda symbol, date: 19.0),
        st_symbols={"SZ000001"},
        record_selection_candidates=lambda *args: None,
    )
    strategy.on_init(context)
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_excludes_delisting_name_from_meta_store(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    from quantx.core.data.meta import MetaStore

    meta_uri = tmp_path / "meta.sqlite"
    MetaStore(meta_uri).upsert_security_master(pd.DataFrame([
        {"symbol": "SZ000001", "name": "国华退"},
        {"symbol": "SH600000", "name": "浦发银行"},
    ]))
    strategy = build_strategy(
        config_with_buy_filter(path, exclude_st=True, meta_store_uri=str(meta_uri)),
        TransactionCost(),
    )
    quote = pd.DataFrame(
        {"$open": [20.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [20_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(board=SimpleNamespace(), get_preclose=lambda symbol, date: 19.0),
        record_selection_candidates=lambda *args: None,
    )
    strategy.on_init(context)
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_does_not_buy_open_gap_above_threshold(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config_with_buy_filter(path, max_buy_open_gap_pct=0.03), TransactionCost())
    quote = pd.DataFrame(
        {"$open": [20.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [20_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(
            board=SimpleNamespace(),
            get_preclose=lambda symbol, date: 19.0 if symbol == "SZ000001" else 10.0,
        ),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_excludes_st_like_five_pct_limit_history(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(
        config_with_buy_filter(
            path,
            exclude_st_like_limit_rate=True,
            st_like_lookback_sessions=3,
            st_like_min_limit_hits=2,
        ),
        TransactionCost(),
    )
    dates = ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13"]
    quote_index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates[:3]), ["SZ000001", "SH600000"]],
        names=["datetime", "instrument"],
    )
    exchange_quote = pd.DataFrame(
        {
            "$close": [10.0, 10.0, 10.5, 10.1, 11.025, 10.2],
        },
        index=quote_index,
    )
    quote = pd.DataFrame(
        {"$open": [11.2, 10.0], "$volume": [1000.0, 1000.0], "$amount": [11_200.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=dates,
        exchange=SimpleNamespace(
            quote=exchange_quote,
            board=SimpleNamespace(),
            get_preclose=lambda symbol, date: 11.025 if symbol == "SZ000001" else 10.2,
        ),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SH600000"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_does_not_treat_ten_pct_history_as_st_like(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(
        config_with_buy_filter(
            path,
            exclude_st_like_limit_rate=True,
            st_like_lookback_sessions=3,
            st_like_min_limit_hits=2,
        ),
        TransactionCost(),
    )
    dates = ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13"]
    quote_index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates[:3]), ["SZ000001", "SH600000"]],
        names=["datetime", "instrument"],
    )
    exchange_quote = pd.DataFrame(
        {
            "$close": [10.0, 10.0, 11.0, 10.1, 12.1, 10.2],
        },
        index=quote_index,
    )
    quote = pd.DataFrame(
        {"$open": [12.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [12_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=dates,
        exchange=SimpleNamespace(
            quote=exchange_quote,
            board=SimpleNamespace(),
            get_preclose=lambda symbol, date: 12.1 if symbol == "SZ000001" else 10.2,
        ),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=10_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SZ000001"
    assert orders[0].action == OrderAction.BUY


def test_prediction_strategy_sell_cooldown_blocks_immediate_rebuy(tmp_path):
    path = tmp_path / "predictions.json"
    first_time = MarketTime(
        "2026-07-10",
        SessionPhase.AFTER_CLOSE,
        datetime(2026, 7, 10, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    second_time = MarketTime(
        "2026-07-13",
        SessionPhase.AFTER_CLOSE,
        datetime(2026, 7, 13, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    PredictionStore(
        [
            PredictionRecord(first_time, "SZ000001", 0.9, 5, "ridge-v1", "fold-1", "schema-1"),
            PredictionRecord(first_time, "SH600000", 0.4, 5, "ridge-v1", "fold-1", "schema-1"),
            PredictionRecord(second_time, "SZ000001", 0.9, 5, "ridge-v1", "fold-1", "schema-1"),
            PredictionRecord(second_time, "SH600000", 0.4, 5, "ridge-v1", "fold-1", "schema-1"),
        ]
    ).write(path)
    cfg = config_with_exit(path)
    cfg["strategy"]["buy_filter"] = {"cooldown_sessions_after_sell": 5}
    strategy = build_strategy(cfg, TransactionCost())
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13", "2026-07-14"],
        exchange=SimpleNamespace(board=SimpleNamespace(), get_preclose=lambda symbol, date: 19.0),
        record_selection_candidates=lambda *args: None,
    )
    strategy.on_init(context)
    first_quote = pd.DataFrame(
        {"$open": [18.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [18_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    first_state = PolicyState(
        date="2026-07-13",
        market_data=first_quote,
        account=AccountSnapshot(cash=1_000.0, total_value=10_000.0),
        positions={"SZ000001": PositionSnapshot("SZ000001", 100, 20.0, 1_800.0, holding_days=3)},
        context=context,
    )

    first_orders = strategy.get_trade_signal(first_state, strategy.get_stock_signal(first_state)).orders

    assert [(order.symbol, order.action) for order in first_orders] == [
        ("SZ000001", OrderAction.SELL),
        ("SH600000", OrderAction.BUY),
    ]

    second_quote = pd.DataFrame(
        {"$open": [18.0, 10.0], "$volume": [1000.0, 1000.0], "$amount": [18_000.0, 10_000.0]},
        index=["SZ000001", "SH600000"],
    )
    second_state = PolicyState(
        date="2026-07-14",
        market_data=second_quote,
        account=AccountSnapshot(cash=1_000.0, total_value=10_000.0),
        positions={},
        context=context,
    )

    second_orders = strategy.get_trade_signal(second_state, strategy.get_stock_signal(second_state)).orders

    assert [order.symbol for order in second_orders if order.action == OrderAction.BUY] == ["SH600000"]


def test_prediction_strategy_sells_stop_loss_even_without_new_prediction(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config_with_exit(path), TransactionCost())
    quote = pd.DataFrame(
        {"$open": [18.0], "$volume": [1000.0], "$amount": [18_000.0]},
        index=["SZ000001"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13", "2026-07-14"],
        exchange=SimpleNamespace(board=SimpleNamespace(), get_preclose=lambda symbol, date: 19.0),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-14",
        market_data=quote,
        account=AccountSnapshot(cash=1_000.0, total_value=10_000.0),
        positions={"SZ000001": PositionSnapshot("SZ000001", 100, 20.0, 1_800.0, holding_days=3)},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert len(orders) == 1
    assert orders[0].symbol == "SZ000001"
    assert orders[0].action == OrderAction.SELL
    assert orders[0].reason.startswith("stop_loss")


def test_prediction_strategy_refills_topk_after_stop_loss(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config_with_exit(path), TransactionCost())
    quote = pd.DataFrame(
        {
            "$open": [18.0, 10.0],
            "$volume": [1000.0, 1000.0],
            "$amount": [18_000.0, 10_000.0],
        },
        index=["SZ000001", "SH600000"],
    )
    context = SimpleNamespace(
        trade_dates=["2026-07-10", "2026-07-13"],
        exchange=SimpleNamespace(
            board=SimpleNamespace(),
            get_preclose=lambda symbol, date: 19.0 if symbol == "SZ000001" else 10.0,
        ),
        record_selection_candidates=lambda *args: None,
    )
    state = PolicyState(
        date="2026-07-13",
        market_data=quote,
        account=AccountSnapshot(cash=1_000.0, total_value=10_000.0),
        positions={"SZ000001": PositionSnapshot("SZ000001", 100, 20.0, 1_800.0, holding_days=3)},
        context=context,
    )

    orders = strategy.get_trade_signal(state, strategy.get_stock_signal(state)).orders

    assert [(order.symbol, order.action) for order in orders] == [
        ("SZ000001", OrderAction.SELL),
        ("SH600000", OrderAction.BUY),
    ]
    assert orders[0].reason.startswith("stop_loss")
    assert orders[1].depends_on_sells == ["SZ000001"]


class _EngineExchange:
    def __init__(self):
        self.deal_price = "open"
        self.board = SimpleNamespace(
            get_limit_up_rate=lambda symbol: 0.1,
            get_limit_down_rate=lambda symbol: 0.1,
        )
        self.prices = {
            "2026-07-10": 19.0,
            "2026-07-13": 20.0,
            "2026-07-14": 21.0,
        }

    def get_deal_price(self, symbol, date, direction=None):
        return self.prices.get(date)

    def get_preclose(self, symbol, date):
        dates = list(self.prices)
        index = dates.index(date)
        return self.prices[dates[index - 1]] if index > 0 else self.prices[date]

    def get_close(self, symbol, date):
        return self.prices.get(date)


class _EngineContext:
    def __init__(self, exchange):
        self.exchange = exchange
        self.trade_dates = []
        self._index = 0
        self._candidates = []

    def load_data(self, *args, **kwargs):
        self.trade_dates = ["2026-07-10", "2026-07-13", "2026-07-14"]

    def next(self):
        date = self.trade_dates[self._index]
        self._index += 1
        return date

    def is_finished(self):
        return self._index >= len(self.trade_dates)

    def get_current_data(self, date):
        price = self.exchange.prices[date]
        return pd.DataFrame({"$open": [price], "$close": [price]}, index=["SZ000001"])

    def get_factor_data(self, date):
        return None

    def record_selection_candidates(self, date, detail):
        self._candidates.append(detail)

    def get_selection_candidates(self):
        return self._candidates

    def get_daily_selection_candidates(self):
        return []


def test_standard_backtest_engine_executes_frozen_prediction_at_next_open(tmp_path, monkeypatch):
    import quantx.core.engine.engine as engine_module

    path = tmp_path / "predictions.json"
    write_predictions(path)
    strategy = build_strategy(config(path), TransactionCost(slippage=0.0))
    exchange = _EngineExchange()
    monkeypatch.setattr(engine_module, "BoardManager", lambda: object())
    monkeypatch.setattr(engine_module, "AStockExchange", lambda board, provider_uri: exchange)
    monkeypatch.setattr(engine_module, "BacktestContext", _EngineContext)
    result = BacktestEngine(
        BacktestConfig(
            start_date="2026-07-10",
            end_date="2026-07-14",
            cost=TransactionCost(slippage=0.0),
            validate_trading_rules=True,
        )
    ).run(strategy, ["SZ000001"])

    buys = [trade for trade in result.trades if trade.action == OrderAction.BUY and not trade.reject_reason]
    assert len(buys) == 1
    assert buys[0].date == "2026-07-13"
    assert buys[0].price == 20.0
    assert result.selection_candidates[0]["signal_date"] == "2026-07-10"
