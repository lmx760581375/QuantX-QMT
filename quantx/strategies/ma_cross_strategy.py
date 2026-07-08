"""示例：双均线选股策略

使用 5 日均线上穿 20 日均线作为买入信号。
"""

import numpy as np
import pandas as pd

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
from quantx.core.engine.types import Order, OrderAction


class MACrossSelector(StockSelector):
    """双均线选股策略：MA5 上穿 MA20 买入"""

    def __init__(self, top_k: int = 20, ma_short: int = 5, ma_long: int = 20):
        self.top_k = top_k
        self.ma_short = ma_short
        self.ma_long = ma_long

    def act(self, state: PolicyState) -> StockSelection:
        signals = []
        df = state.market_data

        if df.empty:
            return StockSelection(signals=[])

        for symbol in df.index.get_level_values("symbol").unique() if isinstance(df.index, pd.MultiIndex) else df.index:
            try:
                close = self._get_close_series(state, symbol)
                if close is None or len(close) < self.ma_long + 1:
                    continue

                ma_s = close.rolling(self.ma_short).mean()
                ma_l = close.rolling(self.ma_long).mean()

                # 金叉：MA5 上穿 MA20
                if ma_s.iloc[-2] <= ma_l.iloc[-2] and ma_s.iloc[-1] > ma_l.iloc[-1]:
                    score = (ma_s.iloc[-1] / ma_l.iloc[-1] - 1) * 100
                    signals.append(Signal(symbol=symbol, score=score, reason="ma_cross"))
            except Exception:
                continue

        signals.sort(key=lambda s: s.score, reverse=True)
        return StockSelection(signals=signals[:self.top_k])

    def _get_close_series(self, state: PolicyState, symbol: str) -> pd.Series:
        """获取收盘价序列"""
        if state.context is not None:
            try:
                return state.context.exchange.quote.loc[
                    pd.IndexSlice[:, symbol], "$close"
                ].droplevel("instrument")
            except Exception:
                pass
        return None


class EqualWeightRebalance(RebalanceStrategy):
    """等权调仓"""

    def __init__(self, max_positions: int = 20):
        self.max_positions = max_positions

    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        selected = selection.signals[:self.max_positions]
        if not selected:
            return WeightAllocation(weights={})
        weight = 1.0 / len(selected)
        return WeightAllocation(weights={s.symbol: weight for s in selected})


class OpenPriceExecution(ExecutionStrategy):
    """以 T 日开盘价执行，先卖后买"""

    def __init__(self, slippage: float = 0.001):
        self.slippage = slippage

    def act(self, state: PolicyState, allocation: WeightAllocation) -> OrderList:
        orders = []
        date = state.date

        # 当前持仓权重
        current_weights = {sym: pos.weight for sym, pos in state.positions.items()}

        # 卖出：当前持有但不在目标中
        for sym, cur_w in current_weights.items():
            tgt_w = allocation.weights.get(sym, 0.0)
            if tgt_w == 0.0:
                pos = state.positions[sym]
                orders.append(Order(
                    symbol=sym, action=OrderAction.SELL,
                    price=self._get_price(state, sym),
                    quantity=pos.quantity, date=date,
                    reason="rebalance_sell",
                ))

        # 买入：目标权重 > 0 的股票
        if state.account and state.account.total_value > 0:
            for sym, tgt_w in allocation.weights.items():
                cur_w = current_weights.get(sym, 0.0)
                if tgt_w > cur_w + 0.02:
                    target_value = state.account.total_value * tgt_w
                    price = self._get_price(state, sym)
                    if price and price > 0:
                        qty = int(target_value / price)
                        qty = (qty // 100) * 100
                        if qty > 0:
                            orders.append(Order(
                                symbol=sym, action=OrderAction.BUY,
                                price=price, quantity=qty, date=date,
                                reason="rebalance_buy",
                            ))

        return OrderList(orders=orders)

    def _get_price(self, state: PolicyState, symbol: str) -> float:
        """获取成交价格（开盘价）"""
        try:
            if isinstance(state.market_data.index, pd.MultiIndex):
                price = state.market_data.loc[(state.date, symbol), "$open"]
                return float(price) if not pd.isna(price) else 0.0
            return 0.0
        except Exception:
            return 0.0


def create_strategy(top_k: int = 20, max_positions: int = 20) -> CompositeStrategy:
    """创建双均线策略"""
    return CompositeStrategy(
        selector=MACrossSelector(top_k=top_k),
        rebalance=EqualWeightRebalance(max_positions=max_positions),
        execution=OpenPriceExecution(),
    )