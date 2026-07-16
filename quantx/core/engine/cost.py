"""交易成本模型

精确计算 A 股交易的各项成本。
所有参数通过 YAML 配置注入。
"""

from dataclasses import dataclass
from typing import Dict

from .types import OrderAction


@dataclass
class TransactionCost:
    """A 股交易成本模型"""

    commission_rate: float = 0.0003     # 佣金费率（万三）
    min_commission: float = 5.0         # 最低佣金（元）
    stamp_tax_rate: float = 0.0005      # 印花税（万五，仅卖出）
    transfer_fee_rate: float = 0.00002  # 过户费（十万分之二，仅沪市）
    slippage: float = 0.001             # 滑点比例（0.1%）
    buy_slippage: float | None = None   # 买入侧滑点；未设置时使用 slippage
    sell_slippage: float | None = None  # 卖出侧滑点；未设置时使用 slippage
    stamp_tax_on_buy: bool = False      # 兼容旧回测：买入也扣印花税

    def calculate_buy_cost(
        self, symbol: str, price: float, quantity: int
    ) -> float:
        """计算买入总成本"""
        return sum(self.calculate_buy_cost_components(symbol, price, quantity).values())

    def calculate_buy_cost_components(
        self, symbol: str, price: float, quantity: int
    ) -> Dict[str, float]:
        """计算买入成本明细。"""
        trade_amount = price * quantity
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        stamp_tax = trade_amount * self.stamp_tax_rate if self.stamp_tax_on_buy else 0.0
        transfer_fee = trade_amount * self.transfer_fee_rate if symbol.startswith("SH") else 0.0
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
        }

    def calculate_sell_cost(
        self, symbol: str, price: float, quantity: int
    ) -> float:
        """计算卖出总成本"""
        return sum(self.calculate_sell_cost_components(symbol, price, quantity).values())

    def calculate_sell_cost_components(
        self, symbol: str, price: float, quantity: int
    ) -> Dict[str, float]:
        """计算卖出成本明细。"""
        trade_amount = price * quantity
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        stamp_tax = trade_amount * self.stamp_tax_rate
        transfer_fee = trade_amount * self.transfer_fee_rate if symbol.startswith("SH") else 0.0
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
        }

    def calculate_cost(
        self, symbol: str, action: OrderAction, price: float, quantity: int
    ) -> float:
        """计算交易总成本"""
        if action == OrderAction.BUY:
            return self.calculate_buy_cost(symbol, price, quantity)
        else:
            return self.calculate_sell_cost(symbol, price, quantity)

    def apply_slippage(self, price: float, action: OrderAction) -> float:
        """应用滑点：买入加价，卖出降价"""
        if action == OrderAction.BUY:
            slippage = self.slippage if self.buy_slippage is None else self.buy_slippage
            return price * (1 + slippage)
        else:
            slippage = self.slippage if self.sell_slippage is None else self.sell_slippage
            return price * (1 - slippage)
