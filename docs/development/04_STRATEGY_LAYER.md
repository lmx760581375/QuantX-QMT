# 策略层开发计划（Policy 架构 v2.0）

> **版本**: v2.0 | **日期**: 2026-06-25 | **预计工期**: 5-7 天 | **依赖**: 回测引擎 (03_BACKTEST_ENGINE.md)

---

## 1. 核心设计理念

回测框架的核心是 **三个独立的 Policy 接口**，每个 Policy 都可以用**规则**或**模型**实现：

```
选股策略 (StockSelector)     → 选哪些股票？打多少分？
调仓策略 (RebalanceStrategy) → 每只股票配多少权重？什么时候调仓？
执行策略 (ExecutionStrategy)  → 怎么把权重差变成订单？用什么价格？
```

**统一接口**：`act(state) -> action`，规则策略和模型策略共用同一个接口。

```
┌───────────────────────────────────────────────────────────────┐
│                     Policy 统一接口                            │
│                                                               │
│   state: PolicyState ──────► Policy.act() ──────► action      │
│                                                               │
│   规则策略:                   模型策略:                         │
│   act() 内直接写逻辑           act() 内调用 model.predict()      │
│   def act(self, state):       def act(self, state):           │
│       if state.close > ...        return self.model(state)    │
│           return Signal(...)                                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 文件清单

```
quantx/core/strategy/
├── __init__.py              # 模块入口
├── base.py                  # 三个 Policy 抽象接口 + State/Action 数据类
├── selector.py              # 选股策略实现（规则 + 模型）
├── rebalance.py             # 调仓策略实现（规则 + 模型）
├── execution.py             # 执行策略实现（规则 + 模型）
├── composite.py             # 组合策略（将三个 Policy 组合成完整策略）
├── loader.py                # 策略加载器
├── templates.py             # 声明式策略模板（YAML 配置）
└── validator.py             # 策略接口校验

quantx/strategies/           # 用户策略目录
├── __init__.py
├── rule_based/              # 规则策略示例
│   ├── ma_cross.py          # 双均线选股
│   ├── momentum.py          # 动量选股
│   └── equal_weight.py      # 等权调仓
└── model_based/             # 模型策略示例（Phase 3）
    └── rl_selector.py       # RL 选股策略
```

---

## 3. 核心接口定义 (`base.py`)

### 3.1 State 和 Action 数据类

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd

# ============================================================
# State: Policy 可以观察的所有信息
# ============================================================

@dataclass
class AccountSnapshot:
    """账户快照（可序列化，供模型输入）"""
    cash: float
    total_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0

@dataclass
class PositionSnapshot:
    """持仓快照（可序列化）"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    weight: float             # 占组合的比例
    holding_days: int         # 已持有天数
    unrealized_pnl: float     # 浮动盈亏

@dataclass
class PolicyState:
    """统一的 Policy 状态，规则策略和模型策略共用

    规则策略：通过 context 访问历史数据 (context.get_history_by_range)
    模型策略：通过 factor_data 和 to_array() 获取固定维度特征
    """
    date: str
    market_data: pd.DataFrame          # (symbol, OHLCV fields) 当日横截面
    factor_data: Optional[pd.DataFrame] # (symbol, factor_values) 因子值（含历史衍生特征）
    account: AccountSnapshot           # 账户快照
    positions: Dict[str, PositionSnapshot]  # 当前持仓
    context: Any = None                # 回测上下文（规则策略用于访问历史数据）
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展字段

    def to_array(self, symbols: List[str]) -> np.ndarray:
        """将 state 转为模型可消费的 numpy 数组（固定维度）
        只使用 factor_data + account + positions，不依赖 context
        """
        ...

    def to_dict(self) -> dict:
        """将 state 转为可序列化的 dict（供日志/调试）"""
        ...


# ============================================================
# Action: Policy 的决策输出
# ============================================================

@dataclass
class Signal:
    """选股信号"""
    symbol: str
    score: float          # 越高越优先
    reason: str = ""      # 调试用

@dataclass
class StockSelection:
    """选股策略的输出"""
    signals: List[Signal]  # 所有候选股票及其分数

@dataclass
class WeightAllocation:
    """调仓策略的输出"""
    weights: Dict[str, float]  # {symbol: target_weight}，权重之和 = 1.0

@dataclass
class OrderList:
    """执行策略的输出"""
    orders: List[Order]  # 按执行顺序排列（先卖后买）
```

### 3.2 三个 Policy 抽象接口

```python
from abc import ABC, abstractmethod

class StockSelector(ABC):
    """选股策略：选哪些股票，打多少分"""

    @abstractmethod
    def act(self, state: PolicyState) -> StockSelection:
        """
        输入: 当日市场状态
        输出: 候选股票及其分数
        规则: 直接写 if/else 逻辑
        模型: 调用 model.predict(state.to_array())
        """
        ...


class RebalanceStrategy(ABC):
    """调仓策略：每只股票配多少权重"""

    @abstractmethod
    def act(
        self, state: PolicyState, selection: StockSelection
    ) -> WeightAllocation:
        """
        输入: 当日状态 + 选股结果
        输出: 目标权重 {symbol: weight}
        规则: 等权/分数加权/市值加权
        模型: 调用 model.optimize(state.to_array(), selection)
        """
        ...


class ExecutionStrategy(ABC):
    """执行策略：怎么把权重差变成订单"""

    @abstractmethod
    def act(
        self, state: PolicyState, allocation: WeightAllocation
    ) -> OrderList:
        """
        输入: 当日状态 + 目标权重
        输出: 买卖订单列表
        规则: 计算当前权重 vs 目标权重的差异 → 生成 Order
        模型: RL agent 输出最优执行方案
        """
        ...
```

### 3.3 Policy 接口与 RL 的兼容性

```python
class ModelBasedPolicy(ABC):
    """模型驱动的 Policy 基类，提供训练/推理接口"""

    @abstractmethod
    def act(self, state: PolicyState) -> Any:
        """推理：给定 state，返回 action"""
        ...

    @abstractmethod
    def update(self, batch: List[Tuple[PolicyState, Any, float, PolicyState]]):
        """训练：用 (state, action, reward, next_state) 批量更新模型"""
        ...

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        ...

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        ...
```

**RL 训练循环**：回测引擎暴露 `env.step()` 接口，RL Agent 调用 `env.step(action)` 获取 `(next_state, reward, done)`，积累经验后调用 `policy.update(batch)` 更新模型。

---

## 4. 规则策略实现

### 4.1 `selector.py` — 选股策略

```python
class TopKSelector(StockSelector):
    """规则：按信号分数排序，选 Top K 只"""
    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def act(self, state: PolicyState) -> StockSelection:
        signals = self._compute_signals(state)
        signals.sort(key=lambda s: s.score, reverse=True)
        return StockSelection(signals=signals[:self.top_k])

    def _compute_signals(self, state: PolicyState) -> List[Signal]:
        # 从因子数据中计算每只股票的分数
        ...


class ThresholdSelector(StockSelector):
    """规则：分数超过阈值即选中"""
    def __init__(self, min_score: float = 0.5):
        self.min_score = min_score

    def act(self, state: PolicyState) -> StockSelection:
        signals = self._compute_signals(state)
        return StockSelection(
            signals=[s for s in signals if s.score >= self.min_score]
        )


class ExpressionSelector(StockSelector):
    """规则：用 Qlib 表达式定义选股条件"""
    def __init__(self, ranking_expr: str, top_k: int, filters: List[str]):
        self.ranking_expr = ranking_expr  # e.g. "Ref($close, -5) / $close"
        self.top_k = top_k
        self.filters = filters  # e.g. ["$volume > 0", "$close > 5"]

    def act(self, state: PolicyState) -> StockSelection:
        ...
```

### 4.2 `rebalance.py` — 调仓策略

```python
class EqualWeightRebalance(RebalanceStrategy):
    """规则：等权分配"""
    def __init__(self, max_positions: int = 20, threshold: float = 0.05):
        self.max_positions = max_positions
        self.threshold = threshold

    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        selected = selection.signals[:self.max_positions]
        weight = 1.0 / len(selected)
        return WeightAllocation(weights={s.symbol: weight for s in selected})


class ScoreWeightRebalance(RebalanceStrategy):
    """规则：按信号分数加权"""
    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        total = sum(s.score for s in selection.signals if s.score > 0)
        if total <= 0:
            return WeightAllocation(weights={})
        return WeightAllocation(weights={
            s.symbol: s.score / total
            for s in selection.signals if s.score > 0
        })
```

### 4.3 `execution.py` — 执行策略

```python
class DefaultExecution(ExecutionStrategy):
    """规则：以 T 日开盘价执行，先卖后买，按手取整"""
    def __init__(self, slippage: float = 0.001):
        self.slippage = slippage

    def act(
        self, state: PolicyState, allocation: WeightAllocation
    ) -> OrderList:
        current_weights = {
            sym: pos.weight
            for sym, pos in state.positions.items()
        }
        orders = []

        # 1. 卖出：当前持有但目标权重为 0，或权重偏高的股票
        for sym, cur_w in current_weights.items():
            tgt_w = allocation.weights.get(sym, 0.0)
            if tgt_w == 0.0 or cur_w > tgt_w + 0.02:
                # 生成卖出订单
                orders.append(Order(
                    symbol=sym, action=OrderAction.SELL,
                    quantity=self._calc_quantity(sym, cur_w, state),
                    price=state.market_data.loc[sym, 'open'],
                    date=state.date,
                ))

        # 2. 买入：目标权重 > 0 的股票，按目标权重分配资金
        for sym, tgt_w in allocation.weights.items():
            cur_w = current_weights.get(sym, 0.0)
            if tgt_w > cur_w + 0.02:
                orders.append(Order(
                    symbol=sym, action=OrderAction.BUY,
                    quantity=self._calc_buy_quantity(sym, tgt_w, state),
                    price=state.market_data.loc[sym, 'open'],
                    date=state.date,
                ))

        return OrderList(orders=orders)
```

---

## 5. 模型策略实现（Phase 3 启用）

```python
class RLSelector(StockSelector, ModelBasedPolicy):
    """RL 选股策略：神经网络输出每只股票的分数"""
    def __init__(self, model: nn.Module, symbols: List[str]):
        self.model = model
        self.symbols = symbols

    def act(self, state: PolicyState) -> StockSelection:
        # 将 state 转为模型输入
        x = state.to_array(self.symbols)
        # 模型输出每只股票的分数
        scores = self.model(x).detach().numpy()
        signals = [
            Signal(symbol=sym, score=float(scores[i]))
            for i, sym in enumerate(self.symbols)
        ]
        return StockSelection(signals=signals)

    def update(self, batch):
        # RL 训练：用 (state, action, reward, next_state) 更新模型
        ...


class RLRebalancer(RebalanceStrategy, ModelBasedPolicy):
    """RL 调仓策略：神经网络输出最优权重"""
    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        symbols = [s.symbol for s in selection.signals]
        x = state.to_array(symbols)
        weights = self.model(x).detach().numpy()
        weights = softmax(weights)  # 归一化
        return WeightAllocation(weights={
            sym: float(w) for sym, w in zip(symbols, weights)
        })
```

---

## 6. 组合策略 (`composite.py`)

```python
class CompositeStrategy:
    """
    将三个 Policy 组合成一个完整策略。
    引擎只需要调用 get_stock_signal() 和 get_trade_signal()。
    """

    def __init__(
        self,
        selector: StockSelector,
        rebalance: RebalanceStrategy,
        execution: ExecutionStrategy,
    ):
        self.selector = selector
        self.rebalance = rebalance
        self.execution = execution

    def on_init(self, context):
        """回测初始化"""
        pass

    def on_finish(self, context):
        """回测结束"""
        pass

    def get_stock_signal(self, state: PolicyState) -> StockSelection:
        """Phase 1: 选股"""
        return self.selector.act(state)

    def get_trade_signal(
        self, state: PolicyState, selection: StockSelection
    ) -> OrderList:
        """Phase 2: 调仓 + 生成订单"""
        allocation = self.rebalance.act(state, selection)
        return self.execution.act(state, allocation)
```

---

## 7. 回测引擎中的使用

```python
# engine.py
class BacktestEngine:
    def run(self, strategy: CompositeStrategy, start, end, symbols, config):
        # ... 初始化 exchange, account, context ...

        # Phase 1: 并行预计算选股信号
        for date in context.trade_dates:
            state = self._build_state(date, context, account)
            selection = strategy.get_stock_signal(state)
            context.set_stock_signals(date, selection)

        # Phase 2: 顺序执行交易
        while not context.is_finished():
            date = context.next()
            state = self._build_state(date, context, account)
            selection = context.get_stock_signals(date)
            orders = strategy.get_trade_signal(state, selection)

            for order in orders.orders:
                executor.execute(order, account, exchange)
            account.update_daily_balance(date, exchange)

    def _build_state(self, date, context, account) -> PolicyState:
        """构建 PolicyState"""
        return PolicyState(
            date=date,
            market_data=context.get_current_data(date),
            factor_data=context.get_factor_data(date),
            context=context,  # 规则策略用于访问历史数据
            account=AccountSnapshot(
                cash=account.cash,
                total_value=account.get_total_value(),
            ),
            positions={
                sym: PositionSnapshot(
                    symbol=sym, quantity=pos.quantity,
                    avg_cost=pos.avg_cost, market_value=pos.market_value,
                    weight=pos.market_value / account.get_total_value(),
                    holding_days=pos.holding_days,
                    unrealized_pnl=pos.market_value - pos.avg_cost * pos.quantity,
                )
                for sym, pos in account.positions.items()
            },
        )
```

---

## 8. RL 训练环境接口

```python
class BacktestEnv:
    """将回测引擎包装为 OpenAI Gym 风格的 RL 环境"""

    def __init__(self, engine: BacktestEngine, config: BacktestConfig):
        self.engine = engine
        self.config = config
        self._state = None
        self._done = False

    def reset(self) -> PolicyState:
        """重置环境，返回初始 state"""
        self.engine._init()
        self._state = self.engine._build_state(...)
        self._done = False
        return self._state

    def step(self, action) -> Tuple[PolicyState, float, bool]:
        """执行一步，返回 (next_state, reward, done)"""
        # 1. 将 action 转为订单
        orders = self._action_to_orders(action)
        # 2. 执行订单
        for order in orders:
            self.engine.executor.execute(order, ...)
        # 3. 更新账户
        self.engine.account.update_daily_balance(...)
        # 4. 计算 reward（日收益率、Sharpe 等）
        reward = self.engine.account.latest_daily_return
        # 5. 构建 next_state
        self._state = self.engine._build_state(...)
        self._done = self.engine.context.is_finished()
        return self._state, reward, self._done

    def _action_to_orders(self, action):
        """将模型输出的 action 转为 OrderList"""
        ...
```

**RL 训练伪代码**：
```python
env = BacktestEnv(engine, config)
policy = RLSelector(model, symbols)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = policy.act(state)       # 模型推理
        next_state, reward, done = env.step(action)
        policy.replay_buffer.append(state, action, reward, next_state)
        state = next_state

    # 回合结束后更新模型
    policy.update(policy.replay_buffer.sample(batch_size=64))
```

---

## 9. 配置方式

```yaml
# configs/strategies/ma_cross_composite.yaml
strategy:
  type: composite
  selector:
    type: ExpressionSelector
    ranking_expr: "Ref($close, -5) / $close"
    top_k: 20
    filters:
      - "$volume > 0"
      - "$close > 5"
  rebalance:
    type: EqualWeightRebalance
    max_positions: 20
    threshold: 0.05
  execution:
    type: DefaultExecution
    slippage: 0.001
```

```yaml
# configs/strategies/rl_composite.yaml (Phase 3)
strategy:
  type: composite
  selector:
    type: RLSelector
    model_path: "models/selector_v1.pt"
    symbols: "all"
  rebalance:
    type: EqualWeightRebalance
    max_positions: 20
  execution:
    type: DefaultExecution
    slippage: 0.001
```

---

## 10. 开发顺序

```
Day 1-2: 核心接口定义
  ├── base.py (PolicyState, StockSelection, WeightAllocation, OrderList)
  ├── base.py (StockSelector, RebalanceStrategy, ExecutionStrategy ABC)
  ├── base.py (ModelBasedPolicy ABC)
  └── composite.py (CompositeStrategy)

Day 3-4: 规则策略实现
  ├── selector.py (TopKSelector, ThresholdSelector, ExpressionSelector)
  ├── rebalance.py (EqualWeightRebalance, ScoreWeightRebalance)
  └── execution.py (DefaultExecution)

Day 5: 加载器 + 校验
  ├── loader.py (支持加载 CompositeStrategy 配置)
  └── validator.py (校验三个 Policy 接口)

Day 6-7: 示例策略 + 测试
  ├── strategies/rule_based/ma_cross.py
  ├── strategies/rule_based/momentum.py
  ├── 与引擎集成测试
  └── 合成数据端到端测试
```

---

## 11. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 接口粒度 | 三个独立 Policy | 选股/调仓/执行是三个独立关注点，各自可独立演进 |
| 统一接口 | `act(state) -> action` | 规则和模型共用同一个接口，切换只需改实现 |
| State 设计 | `PolicyState` dataclass | 可序列化（供模型输入），可扩展（extra 字段） |
| 模型兼容 | `ModelBasedPolicy` ABC | 提供 train/update/save/load 接口，RL agent 可直接实现 |
| 组合方式 | `CompositeStrategy` | 将三个 Policy 组合，引擎只需面对一个策略对象 |
| RL 环境 | `BacktestEnv` (Gym 风格) | `reset()/step()` 标准接口，兼容主流 RL 框架 |

---

## 12. 验收标准

- [ ] 三个 Policy 接口可通过 YAML 配置独立替换
- [ ] 规则策略和模型策略共用同一个接口，无特殊处理
- [ ] `PolicyState` 可序列化为 numpy 数组（供模型输入）
- [ ] `CompositeStrategy` 正确组合三个 Policy
- [ ] 合成数据端到端回测通过（规则策略）
- [ ] `BacktestEnv` 接口符合 Gym 规范（Phase 3）