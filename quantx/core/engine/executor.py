"""订单执行器

接收策略生成的订单，进行 A 股规则校验后执行，返回成交结果。
校验规则委托 exchange 查询行情状态，T+1 检查委托 account。
"""

import logging
from typing import List, Optional

from .cost import TransactionCost
from .exchange import AStockExchange
from .types import Order, OrderAction, Trade

logger = logging.getLogger(__name__)


class Executor:
    """订单执行器

    校验规则（按优先级）：
    1. 停牌检查 → 不可交易
    2. 涨跌停检查 → 涨停不买入，跌停不卖出
    3. 一字板检查 → 买卖都不可
    4. T+1 检查 → 当日买入不可卖出
    5. 价格跳变保护 → 偏差 > 9.5% 跳过买入
    6. 现金检查 → 自动调整买入数量
    7. 持仓检查 → 卖出数量不超过持仓
    """

    def __init__(self, cost: Optional[TransactionCost] = None, validate_trading_rules: bool = True):
        self.cost = cost or TransactionCost()
        self.validate_trading_rules = validate_trading_rules

    def execute(
        self, order: Order, account, exchange: AStockExchange
    ) -> Optional[Trade]:
        """执行订单，返回 Trade 或 None（被拒绝）"""
        # 校验
        if self.validate_trading_rules:
            reject = self._validate(order, account, exchange)
            if reject:
                trade = Trade(
                    symbol=order.symbol,
                    action=order.action,
                    price=order.price,
                    quantity=0,
                    date=order.date,
                    reject_reason=reject,
                    reason=order.reason,
                )
                account.trades.append(trade)
                return None

        # 获取成交价
        deal_price = exchange.get_deal_price(order.symbol, order.date, order.action)
        if deal_price is None or deal_price <= 0:
            return None

        # 应用滑点
        deal_price = self.cost.apply_slippage(deal_price, order.action)

        # 执行
        if order.action == OrderAction.BUY:
            return account.buy(
                order.symbol,
                deal_price,
                order.quantity,
                order.date,
                reason=order.reason,
                context=order.context,
            )
        else:
            return account.sell(order.symbol, deal_price, order.quantity, order.date, reason=order.reason)

    def _validate(
        self, order: Order, account, exchange: AStockExchange
    ) -> Optional[str]:
        """校验订单，返回拒绝原因或 None"""
        symbol = order.symbol
        date = order.date

        # 1. 停牌检查
        if exchange.is_stock_suspended(symbol, date):
            return "suspended"

        # 2. 涨跌停 + 一字板
        if order.action == OrderAction.BUY:
            if exchange.is_one_side_limit_up(symbol, date):
                return "one_side_limit_up"
            if exchange.is_limit_up(symbol, date):
                return "limit_up"
        else:
            if exchange.is_one_side_limit_down(symbol, date):
                return "one_side_limit_down"
            if exchange.is_limit_down(symbol, date):
                return "limit_down"

        # 3. 价格跳变保护（买入时检查）
        if order.action == OrderAction.BUY:
            preclose = exchange.get_preclose(symbol, date)
            if preclose and preclose > 0:
                open_price = exchange.get_deal_price(symbol, date)
                if open_price and abs(open_price / preclose - 1) > 0.095:
                    return "price_jump"

        # 4. T+1 检查（卖出时）
        if order.action == OrderAction.SELL:
            if not account.can_sell(symbol, date):
                return "t1_restriction"

        # 5. 持仓检查（卖出时）
        if order.action == OrderAction.SELL:
            pos = account.get_position(symbol)
            if pos is None or pos.quantity <= 0:
                return "no_position"

        return None

    def execute_batch(
        self, orders: List[Order], account, exchange: AStockExchange
    ) -> List[Trade]:
        """批量执行订单（先卖后买）"""
        sell_orders = [o for o in orders if o.action == OrderAction.SELL]
        buy_orders = [o for o in orders if o.action == OrderAction.BUY]

        trades = []
        for order in sell_orders:
            trade = self.execute(order, account, exchange)
            if trade:
                trades.append(trade)
        for order in buy_orders:
            trade = self.execute(order, account, exchange)
            if trade:
                trades.append(trade)
        return trades
