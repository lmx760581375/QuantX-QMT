"""账户管理

管理回测账户的现金、持仓、交易记录和每日净值。
"""

import logging
from typing import Any, Dict, List, Optional

from .cost import TransactionCost
from .types import DailySnapshot, OrderAction, Position, Trade

logger = logging.getLogger(__name__)


class Account:
    """回测账户"""

    def __init__(
        self,
        init_cash: float = 1_000_000,
        cost: Optional[TransactionCost] = None,
        legacy_cost_price: bool = False,
        auto_adjust_buy_quantity: bool = True,
    ):
        self.init_cash = init_cash
        self.cost = cost or TransactionCost()
        self.legacy_cost_price = legacy_cost_price
        self.auto_adjust_buy_quantity = auto_adjust_buy_quantity
        self.cash = init_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.latest_daily_return: float = 0.0
        self.cumulative_return: float = 0.0
        self.latest_drawdown: float = 0.0
        self._last_total_value: float = init_cash
        self._peak_total_value: float = init_cash

    # ============================================================
    # 买卖操作
    # ============================================================

    def buy(
        self,
        symbol: str,
        price: float,
        quantity: int,
        date: str,
        reason: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Trade]:
        """买入股票"""
        if quantity <= 0:
            return None

        # 按手取整（100 股）
        quantity = (quantity // 100) * 100
        if quantity == 0:
            return None

        # 计算成本
        cost_components = self.cost.calculate_buy_cost_components(symbol, price, quantity)
        total_fee = sum(cost_components.values())
        total_cost = price * quantity + total_fee

        if total_cost > self.cash and self.auto_adjust_buy_quantity:
            # 自动调整买入数量
            if price <= 0:
                return None
            max_quantity = int(self.cash / price)
            quantity = (max_quantity // 100) * 100
            while quantity > 0:
                cost_components = self.cost.calculate_buy_cost_components(symbol, price, quantity)
                total_fee = sum(cost_components.values())
                total_cost = price * quantity + total_fee
                if total_cost <= self.cash:
                    break
                quantity -= 100
            if quantity == 0:
                return None

        if total_cost > self.cash:
            return None

        # 扣款
        self.cash -= total_cost

        # 更新持仓
        if symbol in self.positions:
            old = self.positions[symbol]
            total_qty = old.quantity + quantity
            if self.legacy_cost_price:
                old.avg_cost = total_cost / quantity
            else:
                old.avg_cost = (old.avg_cost * old.quantity + price * quantity) / total_qty
            old.quantity = total_qty
            old.initial_quantity = max(old.initial_quantity or 0, total_qty)
            old.buy_date = date
            if context:
                old.context.update(context)
        else:
            avg_cost = total_cost / quantity if self.legacy_cost_price else price
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                buy_date=date,
                highest_price=price,
                lowest_price=price,
                initial_quantity=quantity,
                context=dict(context or {}),
            )

        trade = Trade(
            symbol=symbol, action=OrderAction.BUY, price=price,
            quantity=quantity, date=date,
            commission=cost_components["commission"],
            stamp_tax=cost_components["stamp_tax"],
            transfer_fee=cost_components["transfer_fee"],
            reason=reason,
        )
        self.trades.append(trade)
        return trade

    def sell(self, symbol: str, price: float, quantity: int, date: str, reason: str = "") -> Optional[Trade]:
        """卖出股票"""
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return None

        quantity = min(quantity, pos.quantity)
        quantity = (quantity // 100) * 100
        if quantity == 0:
            return None

        trade_amount = price * quantity
        cost_components = self.cost.calculate_sell_cost_components(symbol, price, quantity)
        cost_amount = sum(cost_components.values())
        self.cash += trade_amount - cost_amount

        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[symbol]
        else:
            pos.market_value = price * pos.quantity

        trade = Trade(
            symbol=symbol, action=OrderAction.SELL, price=price,
            quantity=quantity, date=date,
            commission=cost_components["commission"],
            stamp_tax=cost_components["stamp_tax"],
            transfer_fee=cost_components["transfer_fee"],
            reason=reason,
        )
        self.trades.append(trade)
        return trade

    # ============================================================
    # T+1 检查
    # ============================================================

    def can_sell(self, symbol: str, date: str) -> bool:
        """检查是否可卖出（T+1 规则）"""
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return False
        return pos.buy_date != date

    # ============================================================
    # 每日更新
    # ============================================================

    def update_daily_balance(self, date: str, exchange) -> None:
        """每日收盘更新持仓市值"""
        prev_value = self._last_total_value

        for sym, pos in self.positions.items():
            close = exchange.get_close(sym, date)
            if close is not None:
                pos.market_value = close * pos.quantity
                if pos.highest_price <= 0:
                    pos.highest_price = close
                else:
                    pos.highest_price = max(pos.highest_price, close)
                if pos.lowest_price <= 0:
                    pos.lowest_price = close
                else:
                    pos.lowest_price = min(pos.lowest_price, close)
                self._update_position_context(pos, close)
            pos.holding_days += 1

        total = self.get_total_value()
        self._peak_total_value = max(self._peak_total_value, total)
        self.latest_drawdown = (
            total / self._peak_total_value - 1 if self._peak_total_value > 0 else 0.0
        )
        self.latest_daily_return = (
            (total - prev_value) / prev_value if prev_value > 0 else 0.0
        )
        self.cumulative_return = total / self.init_cash - 1

        self.daily_snapshots.append(DailySnapshot(
            date=date,
            cash=self.cash,
            total_value=total,
            positions={k: Position(
                symbol=v.symbol, quantity=v.quantity,
                avg_cost=v.avg_cost, buy_date=v.buy_date,
                market_value=v.market_value, holding_days=v.holding_days,
                highest_price=v.highest_price, lowest_price=v.lowest_price,
                initial_quantity=v.initial_quantity,
                context=dict(v.context),
            ) for k, v in self.positions.items()},
            daily_return=self.latest_daily_return,
            cumulative_return=self.cumulative_return,
        ))
        self._last_total_value = total

    # ============================================================
    # 查询
    # ============================================================

    def get_total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_position_weight(self, symbol: str) -> float:
        total = self.get_total_value()
        if total <= 0:
            return 0.0
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return pos.market_value / total

    def apply_external_cash_flow(self, amount: float) -> None:
        """应用不属于策略盈亏的外部现金流，并同步收益与回撤基线。"""
        amount = float(amount)
        if self.cash + amount < -1.0e-8:
            raise ValueError("External cash flow would make account cash negative")
        self.cash += amount
        self._last_total_value += amount
        self._peak_total_value = max(0.0, self._peak_total_value + amount)

    def _update_position_context(self, pos: Position, close: float) -> None:
        if pos.avg_cost <= 0 or close <= 0:
            return
        current_return = close / pos.avg_cost - 1
        next_holding_days = pos.holding_days + 1
        for days in (3, 5, 10):
            key = f"hold_first_{days}d_return"
            if next_holding_days >= days and key not in pos.context:
                pos.context[key] = current_return
