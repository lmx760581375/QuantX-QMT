"""Market-regime-driven rotation policies for the standard backtest engine."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantx.core.engine.cost import TransactionCost
from quantx.core.engine.types import Order, OrderAction
from quantx.core.strategy.base import (
    CompositeStrategy,
    ExecutionStrategy,
    OrderList,
    PolicyState,
    RebalanceStrategy,
    Signal,
    StockSelection,
    StockSelector,
    WeightAllocation,
)
from quantx.strategies.market_regime_rotation_model import (
    REGIME_PROXIES,
    _apply_choppy_confirmation,
    _close_table,
    _select_target,
    parse_args as parse_legacy_args,
    resolve_regime,
)


class MarketRegimeRotationSelector(StockSelector):
    """Select one asset from prior-session closes under a configured regime model."""

    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.regime_proxies = dict(self.params.pop("regime_proxies", REGIME_PROXIES))
        self.cfg = parse_legacy_args([])
        for key, value in self.params.items():
            if not hasattr(self.cfg, key):
                raise ValueError(f"Unsupported Wufu parameter: {key}")
            setattr(self.cfg, key, value)
        self.bars: dict[str, pd.DataFrame] = {}
        self.close = pd.DataFrame()
        self.regime_close = pd.DataFrame()
        self.regime_state: dict[str, Any] = {}
        self.choppy_state: dict[str, Any] = {}

    def prepare(self, context) -> None:
        quote = getattr(getattr(context, "exchange", None), "quote", None)
        if quote is None or quote.empty:
            return
        frame = quote.reset_index().rename(columns={
            "datetime": "date",
            "instrument": "code",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$amount": "amount",
        })
        frame["date"] = pd.to_datetime(frame["date"])
        if "amount" not in frame:
            frame["amount"] = frame["close"] * frame["volume"] * 100.0
        self.bars = {
            str(symbol): rows.sort_values("date").reset_index(drop=True)
            for symbol, rows in frame.groupby("code", sort=False)
            if rows["close"].notna().sum() >= 30
        }
        self.close = _close_table(self.bars)
        regime_columns: dict[str, pd.Series] = {}
        for name, candidates in self.regime_proxies.items():
            symbol = next((item for item in candidates if item in self.close.columns), None)
            if symbol:
                regime_columns[name] = self.close[symbol]
        self.regime_close = pd.DataFrame(regime_columns).sort_index()

    def act(self, state: PolicyState) -> StockSelection:
        context = state.context
        dates = list(getattr(context, "trade_dates", []))
        try:
            date_index = dates.index(state.date)
        except ValueError:
            return StockSelection(signals=[])
        if date_index <= 0 or not self.bars or self.close.empty:
            return StockSelection(signals=[])

        signal_date = pd.Timestamp(dates[date_index - 1])
        regime = resolve_regime(signal_date, self.regime_close, self.regime_state, self.cfg)
        current_holding = next((symbol for symbol, pos in state.positions.items() if pos.quantity > 0), None)
        target, top_metrics, diagnostics = _select_target(
            signal_date,
            regime,
            self.bars,
            self.close,
            current_holding,
            self.cfg,
        )
        target, confirm = _apply_choppy_confirmation(
            target,
            regime,
            current_holding,
            self.cfg,
            self.choppy_state,
            self.cfg.defensive_etf in self.bars,
        )
        diagnostics.update(confirm)
        recorder = getattr(context, "record_selection_candidates", None)
        if callable(recorder):
            recorder(state.date, {
                "date": state.date,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "mode": "market_regime_rotation",
                "regime": regime,
                "target": target,
                "raw_candidate_count": len(top_metrics),
                "selected_count": 1 if target else 0,
                "raw_candidates": top_metrics[:20],
                "selected_candidates": ([{"symbol": target, "score": 1.0, "regime": regime}] if target else []),
                **diagnostics,
            })
        signals = [Signal(
            symbol=target,
            score=1.0,
            reason=f"wufu:{signal_date:%Y-%m-%d}",
            signal_date=signal_date.strftime("%Y-%m-%d"),
        )] if target else []
        return StockSelection(signals=signals)


class TargetRotationRebalance(RebalanceStrategy):
    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        if not selection.signals:
            return WeightAllocation(weights={})
        return WeightAllocation(weights={selection.signals[0].symbol: 1.0})


class TargetRotationExecution(ExecutionStrategy):
    """Rotate the whole portfolio at the configured daily deal price."""

    def __init__(self, deal_price: str = "open", cash_use_ratio: float = 0.99, lot_size: int = 100):
        self.deal_price = str(deal_price).lstrip("$")
        self.cash_use_ratio = float(cash_use_ratio)
        self.lot_size = int(lot_size)

    def act(self, state: PolicyState, allocation: WeightAllocation) -> OrderList:
        target = next(iter(allocation.weights), None)
        orders: list[Order] = []
        required_sells: list[str] = []
        for symbol, position in state.positions.items():
            if position.quantity > 0 and symbol != target:
                price = self._price(state, symbol)
                if price is not None:
                    required_sells.append(symbol)
                    orders.append(Order(
                        symbol=symbol,
                        action=OrderAction.SELL,
                        price=price,
                        quantity=position.quantity,
                        date=state.date,
                        reason="rotation_sell",
                    ))
        if target and target not in state.positions and state.account is not None:
            price = self._price(state, target)
            if price is not None and price > 0:
                quantity = int(state.account.total_value * self.cash_use_ratio / price)
                quantity = quantity // self.lot_size * self.lot_size
                if quantity > 0:
                    orders.append(Order(
                        symbol=target,
                        action=OrderAction.BUY,
                        price=price,
                        quantity=quantity,
                        date=state.date,
                        reason="rotation_buy",
                        depends_on_sells=required_sells,
                    ))
        return OrderList(orders=orders)

    def _price(self, state: PolicyState, symbol: str) -> float | None:
        data = state.market_data
        if data is None or data.empty or symbol not in data.index:
            return None
        value = data.loc[symbol].get(f"${self.deal_price}")
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return None if pd.isna(value) or float(value) <= 0 else float(value)


def create_market_regime_rotation_strategy(
    config: Dict[str, Any],
    cost: TransactionCost | None = None,
) -> CompositeStrategy:
    del cost
    strategy_cfg = config.get("strategy") or {}
    execution_cfg = config.get("execution") or {}
    return CompositeStrategy(
        selector=MarketRegimeRotationSelector(strategy_cfg.get("params") or {}),
        rebalance=TargetRotationRebalance(),
        execution=TargetRotationExecution(
            deal_price=execution_cfg.get("deal_price", "open"),
            cash_use_ratio=execution_cfg.get("cash_use_ratio", 0.99),
            lot_size=execution_cfg.get("lot_size", 100),
        ),
        precompute_stock_signals=False,
    )
