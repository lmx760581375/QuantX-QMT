"""Convert a final target portfolio into standard QuantX orders at OPEN."""

from __future__ import annotations

from quantx.core.engine.types import Order, OrderAction
from quantx.core.strategy.base import OrderList

from .types import ExecutionContext, PendingRebalance


class StandardOrderPlanner:
    def __init__(self, lot_size: int = 100):
        if lot_size < 1:
            raise ValueError("lot_size must be positive")
        self.lot_size = int(lot_size)

    def create_orders(self, execution: ExecutionContext, pending: PendingRebalance) -> OrderList:
        if execution.clock < pending.execute_not_before:
            raise ValueError("PendingRebalance is not ready")
        if pending.expires_at is not None and execution.clock > pending.expires_at:
            raise ValueError("PendingRebalance has expired")
        target = pending.target
        account_value = float(execution.account.total_value)
        sells: list[Order] = []
        buys: list[Order] = []
        symbols = set(execution.positions) | set(target.weights)
        for symbol in sorted(symbols):
            if symbol not in execution.open_prices:
                continue
            price = execution.price(symbol)
            current = execution.positions.get(symbol)
            current_quantity = int(getattr(current, "quantity", 0) or 0)
            target_value = account_value * float(target.weights.get(symbol, 0.0))
            target_quantity = int(target_value / price) // self.lot_size * self.lot_size
            delta = target_quantity - current_quantity
            tradability = execution.tradability.get(symbol)
            if delta < 0 and (tradability is None or tradability.can_sell):
                quantity = current_quantity if target_quantity == 0 else (-delta // self.lot_size * self.lot_size)
                if quantity > 0:
                    sells.append(self._order(symbol, OrderAction.SELL, price, quantity, execution, pending))
            elif delta > 0 and (tradability is None or tradability.can_buy):
                quantity = delta // self.lot_size * self.lot_size
                if quantity > 0:
                    buys.append(self._order(symbol, OrderAction.BUY, price, quantity, execution, pending))
        sell_symbols = [order.symbol for order in sells]
        buy_notional = sum(order.price * order.quantity for order in buys)
        if buy_notional > float(execution.account.cash) and sell_symbols:
            for order in buys:
                order.depends_on_sells = list(sell_symbols)
        return OrderList(orders=[*sells, *buys])

    @staticmethod
    def _order(
        symbol: str,
        action: OrderAction,
        price: float,
        quantity: int,
        execution: ExecutionContext,
        pending: PendingRebalance,
    ) -> Order:
        tradability = execution.tradability.get(symbol)
        return Order(
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            date=execution.clock.session,
            reason=f"target_rebalance:{pending.source_signal_id}",
            context={
                "strict_execution": True,
                "can_buy": True if tradability is None else tradability.can_buy,
                "can_sell": True if tradability is None else tradability.can_sell,
                "tradability_reason": "" if tradability is None else tradability.reason,
                "rebalance_id": pending.rebalance_id,
                "source_signal_id": pending.source_signal_id,
            },
        )
