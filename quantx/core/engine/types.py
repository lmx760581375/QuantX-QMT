"""回测引擎数据类型定义

定义回测引擎中使用的所有核心数据类型。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderAction(Enum):
    BUY = 1
    SELL = 2


@dataclass
class Order:
    """订单：策略生成的买卖指令"""
    symbol: str            # 股票代码
    action: OrderAction    # BUY / SELL
    price: float           # 委托价格
    quantity: int          # 委托数量（股）
    date: str              # 委托日期
    reason: str = ""       # 下单原因（调试用）
    context: Dict[str, Any] = field(default_factory=dict)  # 下单时固化的策略上下文


@dataclass
class Trade:
    """成交记录：实际成交的订单"""
    symbol: str
    action: OrderAction
    price: float           # 成交价格
    quantity: int          # 成交数量
    date: str
    commission: float = 0.0   # 佣金
    stamp_tax: float = 0.0    # 印花税
    transfer_fee: float = 0.0 # 过户费
    slippage_cost: float = 0.0  # 滑点成本
    reject_reason: str = ""  # 如果被拒绝，记录原因
    reason: str = ""       # 下单原因（调试用）

    @property
    def total_cost(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee + self.slippage_cost

    @property
    def trade_value(self) -> float:
        return self.price * self.quantity


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: int          # 持仓数量
    avg_cost: float        # 持仓均价（摊薄后）
    buy_date: str          # 最近买入日期（用于 T+1 判断）
    market_value: float = 0.0  # 市值
    holding_days: int = 0  # 已持有天数
    highest_price: float = 0.0  # 持仓以来最高收盘价
    lowest_price: float = 0.0  # 持仓以来最低收盘价
    initial_quantity: int = 0  # 建仓以来的基准股数，用于分批卖出规则
    context: Dict[str, Any] = field(default_factory=dict)  # 入场/持仓早期固化特征


@dataclass
class Signal:
    """选股信号"""
    symbol: str
    date: str
    score: float           # 信号分数，越高越优先
    selected: bool = True  # 是否选中
    reason: str = ""       # 选股原因（调试用）


@dataclass
class DailySnapshot:
    """每日账户快照"""
    date: str
    cash: float
    total_value: float
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_return: float = 0.0          # 当日收益率
    cumulative_return: float = 0.0     # 累计收益率
    benchmark_return: float = 0.0      # 基准当日收益率
