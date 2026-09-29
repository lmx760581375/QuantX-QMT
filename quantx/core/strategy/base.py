"""策略层基础类型

定义 Policy 所需的核心数据类：PolicyState, StockSelection, WeightAllocation, OrderList。
也定义三个 Policy 抽象接口：StockSelector, RebalanceStrategy, ExecutionStrategy。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# State 数据类
# ============================================================

@dataclass
class AccountSnapshot:
    """账户快照"""
    cash: float
    total_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    drawdown: float = 0.0


@dataclass
class PositionSnapshot:
    """持仓快照"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    weight: float = 0.0
    holding_days: int = 0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    initial_quantity: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyState:
    """统一的 Policy 状态，规则策略和模型策略共用

    规则策略：通过 context 访问历史数据 (context.get_history_by_range)
    模型策略：通过 factor_data 和 to_array() 获取固定维度特征
    """
    date: str
    market_data: pd.DataFrame
    factor_data: Optional[pd.DataFrame] = None
    account: Optional[AccountSnapshot] = None
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    context: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_array(self, symbols: List[str]) -> np.ndarray:
        """将 state 转为模型可消费的 numpy 数组"""
        raise NotImplementedError("to_array not implemented")


# ============================================================
# Action 数据类
# ============================================================

@dataclass
class Signal:
    """选股信号"""
    symbol: str
    score: float
    reason: str = ""


@dataclass
class StockSelection:
    """选股策略的输出"""
    signals: List[Signal]


@dataclass
class WeightAllocation:
    """调仓策略的输出"""
    weights: Dict[str, float]


@dataclass
class OrderList:
    """执行策略的输出"""
    orders: List[Any]


@dataclass
class PositionPlan:
    """Stateful position-management decision produced after raw selection.

    ``entry_weights`` is restricted to the selector's current candidates. The
    executor remains responsible for all order sizing, validation and fills.
    """

    sell_ratios: Dict[str, float] = field(default_factory=dict)
    entry_weights: Dict[str, float] = field(default_factory=dict)
    cash_deploy_ratio: float = 1.0
    replace_sell_rules: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_exit_symbols(self) -> set[str]:
        return {symbol for symbol, ratio in self.sell_ratios.items() if float(ratio) >= 1.0 - 1e-9}


# ============================================================
# 三个 Policy 抽象接口
# ============================================================

class StockSelector(ABC):
    """选股策略：选哪些股票，打多少分"""

    @abstractmethod
    def act(self, state: PolicyState) -> StockSelection:
        ...


class RebalanceStrategy(ABC):
    """调仓策略：每只股票配多少权重"""

    @abstractmethod
    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        ...


class ExecutionStrategy(ABC):
    """执行策略：怎么把权重差变成订单"""

    @abstractmethod
    def act(self, state: PolicyState, allocation: WeightAllocation) -> OrderList:
        ...


class PositionManager(ABC):
    """Optional post-selection policy for dynamic portfolio decisions."""

    @abstractmethod
    def act(self, state: PolicyState, selection: StockSelection) -> PositionPlan:
        ...


# ============================================================
# 组合策略
# ============================================================

class CompositeStrategy:
    """将三个 Policy 组合成一个完整策略"""

    precompute_stock_signals = False

    def __init__(
        self,
        selector: StockSelector,
        rebalance: RebalanceStrategy,
        execution: ExecutionStrategy,
        position_manager: Optional[PositionManager] = None,
        precompute_stock_signals: bool = False,
    ):
        self.selector = selector
        self.rebalance = rebalance
        self.execution = execution
        self.position_manager = position_manager
        self.precompute_stock_signals = bool(precompute_stock_signals)

    def on_init(self, context):
        self.prepare(context)

    def prepare(self, context):
        bind_selector = getattr(self.position_manager, "bind_selector", None)
        if callable(bind_selector):
            bind_selector(self.selector)
        for policy in (self.selector, self.rebalance, self.execution, self.position_manager):
            prepare = getattr(policy, "prepare", None)
            if callable(prepare):
                prepare(context)

    def on_finish(self, context):
        for policy in (self.selector, self.rebalance, self.execution, self.position_manager):
            finish = getattr(policy, "on_finish", None)
            if callable(finish):
                finish(context)

    def get_stock_signal(self, state: PolicyState) -> StockSelection:
        return self.selector.act(state)

    def get_trade_signal(self, state: PolicyState, selection: StockSelection) -> OrderList:
        preview_slot_exempt = getattr(self.execution, "preview_slot_exempt_symbols", None)
        if callable(preview_slot_exempt):
            state.extra["planned_slot_exempt_symbols"] = set(preview_slot_exempt(state))
        if self.position_manager is not None:
            preview = getattr(self.execution, "preview_sell_symbols", None)
            if callable(preview):
                # A residual position manager needs to know which holdings
                # the original strategy will exit today.  It must not mask a
                # baseline risk exit with a partial policy sale.
                state.extra["baseline_rule_sell_symbols"] = set(preview(state))
                state.extra["planned_sell_symbols"] = set(state.extra["baseline_rule_sell_symbols"])
            state.extra["baseline_allocation"] = self.rebalance.act(state, selection)
            plan = self.position_manager.act(state, selection)
            state.extra["position_plan"] = plan
            state.extra["planned_sell_symbols"] = set(plan.full_exit_symbols)
            allocation = WeightAllocation(weights=dict(plan.entry_weights))
            return self.execution.act(state, allocation)
        preview = getattr(self.execution, "preview_sell_symbols", None)
        if callable(preview):
            state.extra["planned_sell_symbols"] = set(preview(state))
        allocation = self.rebalance.act(state, selection)
        return self.execution.act(state, allocation)
