"""Frozen-prediction decision pipeline for the standard BacktestEngine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.decision.clock import DecisionClock, MarketTime, SessionPhase
from quantx.core.decision.defaults import build_default_registry
from quantx.core.decision.pipeline import DecisionPipeline
from quantx.core.decision.queue import PendingRebalanceQueue
from quantx.core.decision.types import (
    AlphaSnapshot,
    ExecutionContext,
    FeatureBatch,
    FinalTargetPortfolio,
    MarketObservation,
    PendingRebalance,
    PortfolioState,
    TradabilityState,
)
from quantx.core.data.meta.store import MetaStore
from quantx.core.engine.types import Order, OrderAction
from quantx.core.engine.simulation import SimulationKernel
from quantx.core.strategy.base import OrderList, PolicyState, Signal, StockSelection


SHANGHAI = ZoneInfo("Asia/Shanghai")


class PredictionDecisionStrategy:
    """Adapter that schedules T predictions for execution at T+1 OPEN."""

    precompute_stock_signals = False
    execution_mode = "strict_event"

    def __init__(
        self,
        *,
        alpha,
        portfolio,
        risk,
        order_planner,
        account_id: str,
        strategy_instance_id: str,
        rebalance_interval_sessions: int = 1,
        max_loss_pct: float | None = None,
        max_loss_days: int = 0,
        trailing_stop_pct: float = 0.0,
        trailing_activate_profit: float = 0.0,
        exclude_st: bool = False,
        meta_store_uri: str = "data/meta/quantx_meta.sqlite",
        exclude_st_like_limit_rate: bool = False,
        st_like_lookback_sessions: int = 60,
        st_like_min_limit_hits: int = 2,
        max_buy_open_gap_pct: float | None = None,
        cooldown_sessions_after_sell: int = 0,
    ):
        self.alpha = alpha
        self.pipeline = DecisionPipeline(alpha, portfolio, risk)
        self.order_planner = order_planner
        self.simulation = SimulationKernel(planner=order_planner)
        self.account_id = account_id
        self.strategy_instance_id = strategy_instance_id
        if rebalance_interval_sessions < 1:
            raise ValueError("rebalance_interval_sessions must be positive")
        self.rebalance_interval_sessions = int(rebalance_interval_sessions)
        self.max_loss_pct = float(max_loss_pct) if max_loss_pct is not None else None
        self.max_loss_days = int(max_loss_days or 0)
        self.trailing_stop_pct = float(trailing_stop_pct or 0.0)
        self.trailing_activate_profit = float(trailing_activate_profit or 0.0)
        self.exclude_st = bool(exclude_st)
        self.meta_store_uri = str(meta_store_uri)
        self.exclude_st_like_limit_rate = bool(exclude_st_like_limit_rate)
        self.st_like_lookback_sessions = int(st_like_lookback_sessions or 60)
        self.st_like_min_limit_hits = int(st_like_min_limit_hits or 2)
        self.max_buy_open_gap_pct = (
            float(max_buy_open_gap_pct) if max_buy_open_gap_pct is not None else None
        )
        self.cooldown_sessions_after_sell = int(cooldown_sessions_after_sell or 0)
        self._last_rebalance_signal_index: int | None = None
        self._st_symbols: set[str] = set()
        self._st_like_cache: dict[tuple[str, int], bool] = {}
        self._st_like_frame: pd.DataFrame | None = None
        self._preclose_frame: pd.DataFrame | None = None
        self._cooldown_until_execution_index: dict[str, int] = {}
        self.pending = PendingRebalanceQueue()
        self.context = None

    def on_init(self, context) -> None:
        self.context = context
        self._st_symbols = self._load_st_symbols(context)
        self._preclose_frame = self._build_preclose_frame(context)
        self._st_like_frame = self._build_st_like_frame(context)
        self.alpha.prepare(context)

    def on_finish(self, context) -> None:
        del context

    def get_stock_signal(self, state: PolicyState) -> StockSelection:
        observation = self._observation(state)
        if observation is None:
            return StockSelection(signals=[])
        alpha = self.alpha.predict(observation)
        state.extra["decision_observation"] = observation
        store = getattr(self.alpha, "store", None)
        records = store.records_for(
            observation.clock.signal_time.session,
            artifact_id=getattr(self.alpha, "artifact_id", None),
        ) if store is not None else ()
        state.extra["has_prediction_event"] = bool(records)
        signals = [
            Signal(
                symbol=value.instrument,
                score=value.score,
                reason=f"prediction:{alpha.source_id}",
                signal_date=alpha.signal_time.session,
            )
            for value in alpha.values
        ]
        recorder = getattr(state.context, "record_selection_candidates", None)
        if callable(recorder):
            recorder(
                state.date,
                {
                    "date": state.date,
                    "signal_date": alpha.signal_time.session,
                    "mode": "frozen_predictions",
                    "artifact_id": alpha.metadata.get("artifact_id"),
                    "status": "available" if records else "missing_prediction",
                    "latest_available_signal_date": _latest_signal_session(
                        store,
                        artifact_id=getattr(self.alpha, "artifact_id", None),
                    ),
                    "raw_candidate_count": len(signals),
                    "selected_count": len(signals),
                    "raw_candidates": [{"symbol": signal.symbol, "score": signal.score} for signal in signals[:100]],
                    "selected_candidates": [
                        {"symbol": signal.symbol, "score": signal.score} for signal in signals[:100]
                    ],
                },
            )
        return StockSelection(signals=signals)

    def get_trade_signal(self, state: PolicyState, selection: StockSelection) -> OrderList:
        observation = state.extra.get("decision_observation") or self._observation(state)
        if observation is None:
            return OrderList(orders=[])
        store = getattr(self.alpha, "store", None)
        records = store.records_for(
            observation.clock.signal_time.session,
            artifact_id=getattr(self.alpha, "artifact_id", None),
        )
        execution = self._execution_context(state, symbols=set(state.positions) | {record.instrument for record in records})
        stop_reasons = self._stop_exit_reasons(state, execution)
        if not records:
            execution_index = self._execution_index(state.date, state.context)
            orders = self._stop_sell_orders(state, execution, stop_reasons)
            self._remember_sell_cooldowns(orders, execution_index)
            return OrderList(orders=orders)
        signal_index = self._signal_index(observation.clock.signal_time.session, state.context)
        if (
            self._last_rebalance_signal_index is not None
            and signal_index - self._last_rebalance_signal_index < self.rebalance_interval_sessions
            and not stop_reasons
        ):
            return OrderList(orders=[])
        portfolio_state = PortfolioState(
            as_of=observation.clock.signal_time,
            account=state.account,
            positions=state.positions,
        )
        execution_index = self._execution_index(state.date, state.context)
        exclude_symbols = set(stop_reasons)
        exclude_symbols.update(self._buy_filter_exclusions(state, execution, execution_index))
        pending = self._decide(observation, portfolio_state, exclude_symbols=exclude_symbols)
        self.pending.put(pending)
        self._last_rebalance_signal_index = signal_index
        execution = self._execution_context(state, symbols=set(state.positions) | set(pending.target.weights))
        ready = self.pending.pop_ready(
            execution.clock,
            account_id=self.account_id,
            strategy_instance_id=self.strategy_instance_id,
        )
        if len(ready) != 1:
            raise ValueError(f"Expected one ready rebalance, got {len(ready)}")
        orders = self.simulation.plan(execution, ready[0]).orders
        for order in orders:
            if order.action == OrderAction.SELL and order.symbol in stop_reasons:
                order.reason = stop_reasons[order.symbol]
        self._remember_sell_cooldowns(orders, execution_index)
        return OrderList(orders=orders)

    def _decide(
        self,
        observation: MarketObservation,
        portfolio_state: PortfolioState,
        *,
        exclude_symbols: set[str] | None = None,
    ) -> PendingRebalance:
        if not exclude_symbols:
            return self.pipeline.decide(
                observation,
                portfolio_state,
                account_id=self.account_id,
                strategy_instance_id=self.strategy_instance_id,
            )
        alpha = self.alpha.predict(observation)
        filtered_alpha = AlphaSnapshot(
            signal_time=alpha.signal_time,
            earliest_execution_time=alpha.earliest_execution_time,
            values=tuple(value for value in alpha.values if value.instrument not in exclude_symbols),
            source_id=alpha.source_id,
            horizon_sessions=alpha.horizon_sessions,
            metadata=dict(alpha.metadata),
        )
        raw = self.pipeline.portfolio.construct(observation, portfolio_state, filtered_alpha)
        final = self.pipeline.risk.apply(observation, portfolio_state, raw)
        return PendingRebalance.from_target(
            final,
            rebalance_id=self.pipeline._rebalance_id(
                final,
                account_id=self.account_id,
                strategy_instance_id=self.strategy_instance_id,
            ),
            account_id=self.account_id,
            strategy_instance_id=self.strategy_instance_id,
        )

    def _stop_exit_reasons(self, state: PolicyState, execution: ExecutionContext) -> dict[str, str]:
        reasons: dict[str, str] = {}
        if self.max_loss_pct is None and self.max_loss_days <= 0 and self.trailing_stop_pct <= 0:
            return reasons
        for symbol, position in state.positions.items():
            if symbol not in execution.open_prices:
                continue
            tradability = execution.tradability.get(symbol)
            if tradability is not None and not tradability.can_sell:
                continue
            avg_cost = float(getattr(position, "avg_cost", 0.0) or 0.0)
            if avg_cost <= 0:
                continue
            price = execution.price(symbol)
            pnl_pct = price / avg_cost - 1.0
            if self.max_loss_pct is not None and pnl_pct <= self.max_loss_pct + 1e-12:
                reasons[symbol] = f"stop_loss_{pnl_pct:.1%}"
                continue
            holding_days = int(getattr(position, "holding_days", 0) or 0)
            if self.max_loss_days > 0 and pnl_pct <= 0 and holding_days >= self.max_loss_days:
                reasons[symbol] = f"time_stop_{holding_days}d"
                continue
            peak = float(getattr(position, "highest_price", 0.0) or 0.0)
            if self.trailing_stop_pct > 0 and peak > 0:
                peak_gain = peak / avg_cost - 1.0
                pullback = price / peak - 1.0
                if peak_gain >= self.trailing_activate_profit and pullback <= -self.trailing_stop_pct:
                    reasons[symbol] = f"trailing_stop_{pullback:.1%}"
        return reasons

    @staticmethod
    def _stop_sell_orders(state: PolicyState, execution: ExecutionContext, stop_reasons: dict[str, str]) -> list[Order]:
        orders: list[Order] = []
        for symbol, reason in sorted(stop_reasons.items()):
            position = state.positions.get(symbol)
            if position is None or int(position.quantity or 0) <= 0:
                continue
            orders.append(Order(
                symbol=symbol,
                action=OrderAction.SELL,
                price=execution.price(symbol),
                quantity=int(position.quantity),
                date=state.date,
                reason=reason,
                context={
                    "strict_execution": True,
                    "can_buy": False,
                    "can_sell": True,
                    "tradability_reason": reason,
                },
            ))
        return orders

    def _signal_index(self, signal_session: str, context=None) -> int:
        dates = list(getattr(context or self.context, "trade_dates", []))
        try:
            return dates.index(signal_session)
        except ValueError as exc:
            raise ValueError(f"Signal session is outside the trading calendar: {signal_session}") from exc

    def _execution_index(self, session: str, context=None) -> int:
        dates = list(getattr(context or self.context, "trade_dates", []))
        try:
            return dates.index(session)
        except ValueError as exc:
            raise ValueError(f"Execution session is outside the trading calendar: {session}") from exc

    def _buy_filter_exclusions(
        self,
        state: PolicyState,
        execution: ExecutionContext,
        execution_index: int,
    ) -> set[str]:
        excluded: set[str] = set()
        current_positions = set(state.positions)
        for symbol in set(execution.open_prices) | set(execution.tradability) | set(self._st_symbols):
            if self.exclude_st and symbol in self._st_symbols:
                excluded.add(symbol)
                continue
            if (
                self.exclude_st_like_limit_rate
                and symbol not in current_positions
                and self._is_st_like_by_limit_history(state, symbol, execution_index)
            ):
                excluded.add(symbol)
                continue
            if symbol not in current_positions and self._is_in_sell_cooldown(symbol, execution_index):
                excluded.add(symbol)
                continue
            tradability = execution.tradability.get(symbol)
            if symbol not in current_positions and tradability is not None and not tradability.can_buy:
                excluded.add(symbol)
        return excluded

    def _is_in_sell_cooldown(self, symbol: str, execution_index: int) -> bool:
        until_index = self._cooldown_until_execution_index.get(symbol)
        if until_index is None:
            return False
        if execution_index <= until_index:
            return True
        self._cooldown_until_execution_index.pop(symbol, None)
        return False

    def _remember_sell_cooldowns(self, orders: list[Order], execution_index: int) -> None:
        if self.cooldown_sessions_after_sell <= 0:
            return
        until_index = execution_index + self.cooldown_sessions_after_sell
        for order in orders:
            if order.action == OrderAction.SELL:
                self._cooldown_until_execution_index[order.symbol] = until_index

    def _is_st_like_by_limit_history(self, state: PolicyState, symbol: str, execution_index: int) -> bool:
        if execution_index <= 0 or self.st_like_lookback_sessions < 1:
            return False
        if self._st_like_frame is not None:
            dates = list(getattr(state.context, "trade_dates", []))
            if execution_index >= len(dates):
                return False
            session = pd.Timestamp(dates[execution_index])
            try:
                return bool(self._st_like_frame.at[session, symbol])
            except (KeyError, ValueError):
                return False
        cache_key = (symbol, execution_index)
        if cache_key in self._st_like_cache:
            return self._st_like_cache[cache_key]
        result = self._compute_st_like_by_limit_history(state, symbol, execution_index)
        self._st_like_cache[cache_key] = result
        return result

    def _compute_st_like_by_limit_history(self, state: PolicyState, symbol: str, execution_index: int) -> bool:
        dates = list(getattr(state.context, "trade_dates", []))
        if not dates:
            return False
        end_index = execution_index - 1
        start_index = max(0, end_index - self.st_like_lookback_sessions + 1)
        selected_dates = pd.DatetimeIndex(pd.to_datetime(dates[start_index : end_index + 1]))
        if len(selected_dates) < self.st_like_min_limit_hits:
            return False
        quote = getattr(getattr(state.context, "exchange", None), "quote", None)
        if quote is None or quote.empty or not isinstance(quote.index, pd.MultiIndex):
            return False
        try:
            history = quote.xs(symbol, level="instrument")
        except (KeyError, ValueError):
            return False
        history = history.reindex(selected_dates)
        if "$close" not in history.columns:
            return False
        close = pd.to_numeric(history["$close"], errors="coerce")
        if "$preclose" in history.columns:
            preclose = pd.to_numeric(history["$preclose"], errors="coerce")
        else:
            preclose = close.shift(1)
        daily_ret = close / preclose - 1.0
        daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).dropna()
        if daily_ret.empty:
            return False
        abs_ret = daily_ret.abs()
        five_pct_hits = int(((abs_ret >= 0.045) & (abs_ret <= 0.055)).sum())
        ten_pct_hits = int((abs_ret >= 0.075).sum())
        return five_pct_hits >= self.st_like_min_limit_hits and ten_pct_hits == 0

    def _build_st_like_frame(self, context) -> pd.DataFrame | None:
        if not self.exclude_st_like_limit_rate or self.st_like_lookback_sessions < 1:
            return None
        quote = getattr(getattr(context, "exchange", None), "quote", None)
        if quote is None or quote.empty or not isinstance(quote.index, pd.MultiIndex):
            return None
        if "$close" not in quote.columns:
            return None
        try:
            close = pd.to_numeric(quote["$close"], errors="coerce").unstack("instrument")
        except (KeyError, ValueError):
            return None
        dates = pd.DatetimeIndex(pd.to_datetime(list(getattr(context, "trade_dates", []))))
        if dates.empty:
            return None
        quote_dates = pd.DatetimeIndex(quote.index.get_level_values("datetime").unique()).sort_values()
        index_dates = quote_dates.union(dates).sort_values()
        close = close.reindex(index_dates)
        if "$preclose" in quote.columns:
            try:
                preclose = pd.to_numeric(quote["$preclose"], errors="coerce").unstack("instrument").reindex(index_dates)
            except (KeyError, ValueError):
                preclose = close.shift(1)
        else:
            preclose = close.shift(1)
        daily_ret = (close / preclose - 1.0).replace([np.inf, -np.inf], np.nan)
        abs_ret = daily_ret.abs()
        five_pct_hits = ((abs_ret >= 0.045) & (abs_ret <= 0.055)).rolling(
            self.st_like_lookback_sessions,
            min_periods=1,
        ).sum()
        ten_pct_hits = (abs_ret >= 0.075).rolling(self.st_like_lookback_sessions, min_periods=1).sum()
        st_like_as_of_close = (five_pct_hits >= self.st_like_min_limit_hits) & (ten_pct_hits == 0)
        return st_like_as_of_close.shift(1).reindex(dates).eq(True)

    def _observation(self, state: PolicyState) -> MarketObservation | None:
        dates = list(getattr(state.context, "trade_dates", []))
        try:
            execution_index = dates.index(state.date)
        except ValueError:
            return None
        if execution_index < 1:
            return None
        signal_session = str(dates[execution_index - 1])
        records = self.alpha.store.records_for(signal_session, artifact_id=self.alpha.artifact_id)
        if records:
            signal_time = records[0].signal_time
        else:
            signal_time = _market_time(signal_session, 15, 0, SessionPhase.AFTER_CLOSE)
        execute_not_before = _market_time(state.date, 9, 30, SessionPhase.OPEN)
        universe = tuple(
            sorted(
                set(str(symbol) for symbol in getattr(state.market_data, "index", []))
                | set(state.positions)
                | {record.instrument for record in records}
            )
        )
        features = FeatureBatch(
            as_of=signal_time,
            instruments=(),
            values=self._market_features(signal_session),
            schema_id=self.alpha.feature_schema_hash,
            available_at=signal_time,
        )
        return MarketObservation(
            clock=DecisionClock(signal_time, signal_time, signal_time, execute_not_before),
            universe=universe,
            features=features,
        )

    def _market_features(self, signal_session: str) -> dict[str, float]:
        panel = getattr(self.context, "market_panel", None)
        if panel is None or not panel.has("close"):
            return {}
        try:
            index = panel.date_index(signal_session)
        except KeyError:
            return {}
        closes = np.asarray(panel.get("close"), dtype=float)
        current = closes[index]
        features: dict[str, float] = {}
        for horizon in (20, 60):
            if index < horizon:
                continue
            previous = closes[index - horizon]
            valid = np.isfinite(current) & np.isfinite(previous) & (previous > 0)
            if valid.any():
                features[f"market_ret{horizon}"] = float(np.median(current[valid] / previous[valid] - 1.0))
        if index >= 19:
            window = closes[index - 19 : index + 1]
            finite = np.isfinite(window)
            counts = finite.sum(axis=0)
            moving_average = np.divide(
                np.nansum(window, axis=0),
                counts,
                out=np.full(closes.shape[1], np.nan, dtype=float),
                where=counts > 0,
            )
            valid = np.isfinite(current) & np.isfinite(moving_average) & (moving_average > 0)
            if valid.any():
                features["breadth20"] = float(np.mean(current[valid] > moving_average[valid]))
        return features

    def _execution_context(self, state: PolicyState, symbols: set[str] | None = None) -> ExecutionContext:
        prices: dict[str, float] = {}
        tradability: dict[str, TradabilityState] = {}
        data = state.market_data
        if data is not None and not data.empty and "$open" in data.columns:
            for symbol, row in data.iterrows():
                symbol = str(symbol)
                if symbols is not None and symbol not in symbols:
                    continue
                value = row["$open"]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                if pd.isna(value) or float(value) <= 0:
                    continue
                price = float(value)
                if _is_untradable_bar(row):
                    tradability[symbol] = TradabilityState(can_buy=False, can_sell=False, reason="suspended")
                    continue
                prices[symbol] = price
                tradability[symbol] = self._tradability(state, symbol, price, row)
        return ExecutionContext(
            clock=_market_time(state.date, 9, 30, SessionPhase.OPEN),
            open_prices=prices,
            tradability=tradability,
            account=state.account,
            positions=state.positions,
        )

    def _tradability(self, state: PolicyState, symbol: str, open_price: float, row=None) -> TradabilityState:
        exchange = getattr(state.context, "exchange", None)
        preclose = self._preclose(symbol, state.date)
        if preclose is None:
            get_preclose = getattr(exchange, "get_preclose", None)
            preclose = get_preclose(symbol, state.date) if callable(get_preclose) else None
        if _is_one_price_limit_bar(row, preclose, direction="up"):
            return TradabilityState(can_buy=False, can_sell=True, reason="one_side_limit_up")
        if _is_one_price_limit_bar(row, preclose, direction="down"):
            return TradabilityState(can_buy=True, can_sell=False, reason="one_side_limit_down")
        board = getattr(exchange, "board", None)
        up_rate = getattr(board, "get_limit_up_rate", None)
        down_rate = getattr(board, "get_limit_down_rate", None)
        if preclose and preclose > 0 and callable(up_rate) and callable(down_rate):
            if open_price >= preclose * (1 + float(up_rate(symbol)) - 1e-6):
                return TradabilityState(can_buy=False, can_sell=True, reason="limit_up")
            if open_price <= preclose * (1 - float(down_rate(symbol)) + 1e-6):
                return TradabilityState(can_buy=True, can_sell=False, reason="limit_down")
        if (
            self.max_buy_open_gap_pct is not None
            and preclose is not None
            and preclose > 0
            and open_price / preclose - 1.0 >= self.max_buy_open_gap_pct - 1e-12
        ):
            return TradabilityState(can_buy=False, can_sell=True, reason="buy_open_gap")
        return TradabilityState(can_buy=True, can_sell=True)

    def _preclose(self, symbol: str, date: str) -> float | None:
        if self._preclose_frame is None:
            return None
        try:
            value = self._preclose_frame.at[pd.Timestamp(date), symbol]
        except (KeyError, ValueError):
            return None
        if pd.isna(value) or float(value) <= 0:
            return None
        return float(value)

    def _build_preclose_frame(self, context) -> pd.DataFrame | None:
        quote = getattr(getattr(context, "exchange", None), "quote", None)
        if quote is None or quote.empty or not isinstance(quote.index, pd.MultiIndex):
            return None
        dates = pd.DatetimeIndex(pd.to_datetime(list(getattr(context, "trade_dates", []))))
        if dates.empty:
            return None
        quote_dates = pd.DatetimeIndex(quote.index.get_level_values("datetime").unique()).sort_values()
        index_dates = quote_dates.union(dates).sort_values()
        if "$preclose" in quote.columns:
            try:
                return pd.to_numeric(quote["$preclose"], errors="coerce").unstack("instrument").reindex(dates)
            except (KeyError, ValueError):
                return None
        if "$close" not in quote.columns:
            return None
        try:
            close = pd.to_numeric(quote["$close"], errors="coerce").unstack("instrument").reindex(index_dates)
        except (KeyError, ValueError):
            return None
        return close.shift(1).reindex(dates)

    def _load_st_symbols(self, context) -> set[str]:
        if not self.exclude_st:
            return set()
        symbols = set(_st_symbols_from_context(context))
        path = Path(self.meta_store_uri)
        if path.exists():
            try:
                meta = MetaStore(path).get_symbol_meta()
            except Exception:
                meta = None
            if meta is not None and not meta.empty and "name" in meta.columns:
                names = meta["name"].fillna("").astype(str).str.upper()
                risk_name = names.str.contains("ST", regex=False) | names.str.contains("退", regex=False)
                symbols.update(str(symbol) for symbol in meta.loc[risk_name, "symbol"])
        return symbols


def create_prediction_decision_strategy(config: Dict[str, Any]):
    strategy_cfg = dict(config.get("strategy") or {})
    if str((config.get("execution") or {}).get("deal_price", "open")).lstrip("$") != "open":
        raise ValueError("decision_pipeline currently requires execution.deal_price=open")
    registry = build_default_registry()
    alpha_cfg = dict(strategy_cfg.get("alpha") or {})
    portfolio_cfg = dict(strategy_cfg.get("portfolio") or {})
    risk_cfg = dict(strategy_cfg.get("risk") or {"type": "noop"})
    planner_cfg = dict(strategy_cfg.get("order_planner") or {"type": "standard"})
    alpha = registry.build("alpha", str(alpha_cfg.pop("type", "predictions")), alpha_cfg)
    portfolio = registry.build("portfolio", str(portfolio_cfg.pop("type", "topk_equal")), portfolio_cfg)
    risk = registry.build("risk", str(risk_cfg.pop("type", "noop")), risk_cfg)
    planner = registry.build("order_planner", str(planner_cfg.pop("type", "standard")), planner_cfg)
    exit_cfg = dict(strategy_cfg.get("exit") or strategy_cfg.get("sell") or {})
    buy_filter_cfg = dict(strategy_cfg.get("buy_filter") or strategy_cfg.get("buy_filters") or {})
    return PredictionDecisionStrategy(
        alpha=alpha,
        portfolio=portfolio,
        risk=risk,
        order_planner=planner,
        account_id=str(strategy_cfg.get("account_id", "backtest")),
        strategy_instance_id=str(strategy_cfg.get("instance_id", config.get("name", "decision_pipeline"))),
        rebalance_interval_sessions=int(strategy_cfg.get("rebalance_interval_sessions", 1)),
        max_loss_pct=(float(exit_cfg["max_loss_pct"]) if exit_cfg.get("max_loss_pct") is not None else None),
        max_loss_days=int(exit_cfg.get("max_loss_days", 0) or 0),
        trailing_stop_pct=float(exit_cfg.get("trailing_stop_pct", 0.0) or 0.0),
        trailing_activate_profit=float(exit_cfg.get("trailing_activate_profit", 0.0) or 0.0),
        exclude_st=bool(buy_filter_cfg.get("exclude_st", False)),
        meta_store_uri=str(buy_filter_cfg.get("meta_store_uri", "data/meta/quantx_meta.sqlite")),
        exclude_st_like_limit_rate=bool(buy_filter_cfg.get("exclude_st_like_limit_rate", False)),
        st_like_lookback_sessions=int(buy_filter_cfg.get("st_like_lookback_sessions", 60) or 60),
        st_like_min_limit_hits=int(buy_filter_cfg.get("st_like_min_limit_hits", 2) or 2),
        max_buy_open_gap_pct=(
            float(buy_filter_cfg["max_buy_open_gap_pct"])
            if buy_filter_cfg.get("max_buy_open_gap_pct") is not None
            else None
        ),
        cooldown_sessions_after_sell=int(buy_filter_cfg.get("cooldown_sessions_after_sell", 0) or 0),
    )


def explain_prediction_decision_strategy(config: Dict[str, Any]) -> Dict[str, Any]:
    strategy_cfg = dict(config.get("strategy") or {})
    alpha_cfg = dict(strategy_cfg.get("alpha") or {})
    portfolio_cfg = dict(strategy_cfg.get("portfolio") or {})
    risk_cfg = dict(strategy_cfg.get("risk") or {"type": "noop"})
    planner_cfg = dict(strategy_cfg.get("order_planner") or {"type": "standard"})
    return {
        "name": config.get("name"),
        "version": config.get("version"),
        "strategy_type": "decision_pipeline",
        "artifacts": [str(alpha_cfg.get("artifact_id"))] if alpha_cfg.get("artifact_id") else [],
        "alpha": {key: value for key, value in alpha_cfg.items() if key not in {"path", "checksum"}},
        "portfolio": portfolio_cfg,
        "rebalance_interval_sessions": int(strategy_cfg.get("rebalance_interval_sessions", 1)),
        "buy_filter": dict(strategy_cfg.get("buy_filter") or strategy_cfg.get("buy_filters") or {}),
        "risk": risk_cfg,
        "execution": {
            "event_mode": "strict_event",
            "deal_price": "open",
            "order_planner": planner_cfg,
        },
    }


def _market_time(session: str, hour: int, minute: int, phase: SessionPhase) -> MarketTime:
    day = datetime.strptime(str(session), "%Y-%m-%d")
    return MarketTime(
        session=str(session),
        phase=phase,
        timestamp=day.replace(hour=hour, minute=minute, tzinfo=SHANGHAI),
    )


def _latest_signal_session(store, *, artifact_id: str | None = None) -> str | None:
    if store is None:
        return None
    sessions = [
        session
        for session in getattr(store, "signal_sessions", ())
        if store.records_for(session, artifact_id=artifact_id)
    ]
    return sessions[-1] if sessions else None


def _st_symbols_from_context(context) -> set[str]:
    symbols = getattr(context, "st_symbols", None)
    if symbols is None:
        return set()
    return {str(symbol) for symbol in symbols}


def _is_untradable_bar(row) -> bool:
    """Treat QMT/Qlib zero-volume placeholder bars as non-executable."""
    for field in ("$volume", "$amount"):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if pd.isna(value) or float(value) <= 0:
            return True
    return False


def _is_one_price_limit_bar(row, preclose: float | None, *, direction: str) -> bool:
    if row is None or preclose is None or preclose <= 0:
        return False
    values = [_row_float(row, field) for field in ("$open", "$high", "$low", "$close")]
    if any(value is None or value <= 0 for value in values):
        return False
    open_price, high, low, close = values
    if max(abs(high - low), abs(open_price - close)) > 1e-6:
        return False
    bar_return = close / preclose - 1.0
    if direction == "up":
        return bar_return >= 0.045
    if direction == "down":
        return bar_return <= -0.045
    return False


def _row_float(row, field: str) -> float | None:
    if field not in row:
        return None
    value = row[field]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if pd.isna(value):
        return None
    return float(value)
