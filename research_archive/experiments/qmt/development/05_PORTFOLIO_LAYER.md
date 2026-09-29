# 组合层开发计划（已合并到策略层）

> **状态**: 已废弃，合并到 [04_STRATEGY_LAYER.md](04_STRATEGY_LAYER.md)
> **原因**: 调仓策略（RebalanceStrategy）已作为 Policy 接口的一部分，在策略层中统一管理

---

组合层原本负责「持仓管理和资金分配」，在 Policy 架构中已拆分为两个独立的 Policy：

| 原组合层职责 | 新位置 | 说明 |
|-------------|--------|------|
| 权重分配（等权/分数加权） | `strategy/rebalance.py` → `RebalanceStrategy` | 调仓策略，可规则/可模型 |
| 再平衡/换手率控制 | `strategy/rebalance.py` → `RebalanceStrategy.act()` | 调仓策略内部实现 |
| 买卖执行 | `strategy/execution.py` → `ExecutionStrategy` | 执行策略，可规则/可模型 |

**不再需要独立的 `core/portfolio/` 目录。**

详见 [04_STRATEGY_LAYER.md](04_STRATEGY_LAYER.md) 第 3-5 节。

---

## 1. 目标

组合层负责持仓管理和资金分配：
1. 将策略生成的选股信号转化为目标权重
2. 计算当前持仓与目标持仓的差异，生成调仓方案
3. 控制换手率，避免过度交易

---

## 2. 文件清单

```
quantx/core/portfolio/
├── __init__.py              # 模块入口
├── base.py                  # 组合抽象基类
├── equal_weight.py          # 等权组合（Phase 1）
├── score_weight.py          # 信号分数加权组合
├── market_cap_weight.py     # 市值加权组合
├── rebalance.py             # 再平衡逻辑（核心）
└── optimizer.py             # 组合优化器（Phase 2：风险平价/均值方差）
```

---

## 2.2 各文件详细说明

### 2.2.1 `base.py` — 组合抽象基类

```python
class Portfolio(ABC):
    @abstractmethod
    def allocate(
        self, signals: List[Signal], account: Account, context: BacktestContext
    ) -> Dict[str, float]:
        """根据信号和账户状态，返回 {symbol: target_weight} 目标权重字典"""
        ...
```

### 2.2.2 `equal_weight.py` — 等权组合（Phase 1 默认实现）

```python
class EqualWeightPortfolio(Portfolio):
    def __init__(self, max_positions: int = 20, rebalance_threshold: float = 0.05):
        self.max_positions = max_positions
        self.rebalance_threshold = rebalance_threshold  # 权重偏离阈值

    def allocate(self, signals, account, context):
        # 按分数排序，取前 max_positions 只
        sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
        selected = sorted_signals[:self.max_positions]
        weight = 1.0 / len(selected)
        return {s.symbol: weight for s in selected}
```

### 2.2.3 `score_weight.py` — 信号分数加权

```python
class ScoreWeightPortfolio(Portfolio):
    def allocate(self, signals, account, context):
        total_score = sum(s.score for s in signals if s.score > 0)
        if total_score <= 0:
            return {}
        return {s.symbol: s.score / total_score for s in signals if s.score > 0}
```

### 2.2.4 `rebalance.py` — 再平衡逻辑（核心）

**功能**：计算当前持仓与目标权重的差异，生成调仓方案，控制换手率。

**核心类**：`Rebalancer`

```python
class Rebalancer:
    def __init__(self, threshold: float = 0.05, max_turnover: float = 0.5):
        """
        threshold: 权重偏离阈值，超过此值才触发调仓
        max_turnover: 最大换手率，限制单日买卖总额
        """
        self.threshold = threshold
        self.max_turnover = max_turnover

    def compute_rebalance(
        self,
        current_weights: Dict[str, float],   # 当前持仓权重
        target_weights: Dict[str, float],     # 目标持仓权重
        account: Account,
        context: BacktestContext,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        返回 (to_sell: {symbol: weight}, to_buy: {symbol: weight})
        """
        to_sell = {}
        to_buy = {}

        # 1. 找出需要卖出的股票（当前持有但不在目标中，或权重偏高）
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        for sym in all_symbols:
            cur = current_weights.get(sym, 0.0)
            tgt = target_weights.get(sym, 0.0)
            diff = cur - tgt

            if diff > self.threshold:
                to_sell[sym] = diff  # 需要减持
            elif diff < -self.threshold:
                to_buy[sym] = -diff  # 需要增持

        # 2. 控制换手率：如果总调仓量超过 max_turnover，按比例缩减
        total_turnover = sum(to_sell.values()) + sum(to_buy.values())
        if total_turnover > self.max_turnover:
            scale = self.max_turnover / total_turnover
            to_sell = {k: v * scale for k, v in to_sell.items()}
            to_buy = {k: v * scale for k, v in to_buy.items()}

        # 3. 先卖后买：确保卖出金额 >= 买入金额
        sell_value = sum(account.get_total_value() * w for w in to_sell.values())
        buy_value = sum(account.get_total_value() * w for w in to_buy.values())
        if buy_value > sell_value * 0.95:
            # 买入金额不能超过卖出释放的资金（考虑交易成本）
            scale = sell_value * 0.95 / buy_value
            to_buy = {k: v * scale for k, v in to_buy.items()}

        return to_sell, to_buy
```

**开发要点**：
- 约 80 行新代码
- 再平衡阈值控制换手率：只在权重偏离超过阈值时才调仓
- 先卖后买：确保卖出释放的资金足够支付买入
- 最大换手率限制：防止单日交易过多（如限制不超过总资产的 50%）
- 注意：权重计算基于 `account.total_value`，不是 `account.cash`

---

## 3. 开发顺序

```
Day 1: base.py + equal_weight.py + rebalance.py
Day 2: score_weight.py + market_cap_weight.py
Day 3: 测试
```

---

## 4. 验收标准

- [ ] 等权组合权重之和 = 1.0
- [ ] 分数加权组合按分数比例分配资金
- [ ] 持仓数量不超过 max_positions
- [ ] 再平衡只在权重偏离超过阈值时触发
- [ ] 换手率不超过 max_turnover 限制
- [ ] 卖出释放资金 >= 买入所需资金（考虑交易成本）
