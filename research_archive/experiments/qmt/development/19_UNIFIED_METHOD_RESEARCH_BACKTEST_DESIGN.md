# QuantX 统一研究方法与回测接入框架设计

> **版本**: v0.2.0
> **日期**: 2026-07-12
> **状态**: 评审修订版，尚未实现
> **适用范围**: 规则策略、传统机器学习、深度学习、强化学习、自动因子挖掘
> **依赖**: 当前 QMT/Qlib 数据层、FactorRuntime、Policy 回测引擎
> **取代范围**: 本文取代 `04_STRATEGY_LAYER.md` 中把训练能力直接放入 `Policy`、通过 `PolicyState.to_array()` 接入模型的早期设想；现有规则 Policy 行为继续兼容。

## 1. 文档目的

QuantX 当前已经具备统一的数据读取、因子计算、账户、撮合、成本、策略配置和回测报告能力，但模型研究仍缺少稳定边界。新增一种方法时，容易出现以下问题：

1. 为 LightGBM、Transformer、RL 分别编写不同的回测入口。
2. 把训练逻辑塞入回测循环，导致训练、推理和成交耦合。
3. 模型直接访问完整 `BacktestContext`，难以证明没有读取未来数据。
4. 特征、标签、训练区间、模型文件和回测结果无法形成可复现关系。
5. 规则策略输出股票，RL 输出权重，执行模型输出订单，强行使用同一个 `StockSelector` 接口会失真。

本文定义一套逐步落地的统一架构，使不同技术只需要适配稳定的决策协议，就可以复用同一套数据、训练验证、组合构建和回测系统。

最终目标不是让所有算法继承同一个万能基类，而是让所有算法至少输出以下三类结果之一：

```text
1. AlphaSnapshot     股票评分/预测
2. Raw/FinalTargetPortfolio   风险处理前后的目标持仓权重
3. OrderIntent       订单意图
```

只要一种技术能够输出其中一种结果，就能够接入 QuantX。其余阶段由已有或可配置组件补齐。

这里的“订单意图”不是在信号时点提前使用未来价格计算出的最终订单。决策阶段先产生带最早执行时间的 `PendingRebalance`，到真实执行事件发生后，`OrderPlanner` 才能读取当时可用的价格和交易状态生成订单。

## 2. 核心结论

### 2.1 保留什么

以下现有能力保留并继续作为唯一事实来源：

- QMT/xqshare 数据同步和本地 Qlib provider。
- `QlibBinReader`、`MarketPanel` 和 `FactorRuntime`。
- `Account`、`AStockExchange`、`Executor` 和交易成本模型。
- A 股 T+1、涨跌停、停牌、手数、手续费等交易规则。
- 标准 `run_backtest` 和标准 run artifacts。
- 当前严格的“信号必须早于交易”校验。

### 2.2 新增什么

新增两个相互隔离的平面：

```text
研究平面 Research Plane
  负责数据集、标签、时间切分、训练、验证、模型和样本外预测。

决策平面 Decision Plane
  负责把已知数据或冻结预测转换为经过风险处理的目标组合和待执行调仓。

执行平面 Execution Plane
  在目标成交时点读取当时已知的执行行情，把待执行调仓转换为订单并交给统一回测引擎。
```

### 2.3 最重要的边界

默认模式下，训练不发生在回测撮合循环中。

```text
训练/验证
  -> 生成冻结模型和严格样本外预测
  -> 标准回测读取预测
  -> 构建组合并成交
```

在线学习和 RL 在线更新是后续显式模式，不能复用普通的并行信号预计算路径。

## 3. 设计目标与非目标

### 3.1 设计目标

1. 同一回测引擎运行公式、ML、DL 和 RL 策略。
2. 复用当前 QMT/Qlib 数据层构建训练集、验证集和测试集。
3. 每个预测都能追溯到数据版本、特征版本、模型版本和训练 fold。
4. 默认强制 walk-forward，禁止随机切分时间序列。
5. 把未来数据防护从开发约定升级为类型、数据视图和运行时校验。
6. 模型框架可选，核心包不强依赖 LightGBM、PyTorch、Gymnasium 或 Qlib 完整安装。
7. 同一份预测可以复用不同组合和执行规则，避免重复训练。
8. 现有公式策略和五福策略可渐进迁移，不要求一次性重写。
9. 标准报告同时覆盖交易绩效和模型绩效。
10. 支持离线训练、冻结推理、滚动训练，后续可扩展在线学习。
11. 决策时点与执行时点使用不同的数据视图，订单必须经过待执行队列。
12. 风险决策与订单机械执行分离，避免订单规划层隐式覆盖模型目标。

### 3.2 非目标

1. 第一阶段不实现分布式训练平台。
2. 第一阶段不引入 MLflow、Ray、Airflow 等重型基础设施。
3. 不允许配置文件直接 import 任意 Python 路径。
4. 不承诺任意模型都能产生盈利，只保证接入、验证和审计方式统一。
5. 不用深度学习或 RL 重写现有撮合引擎。
6. 第一阶段只支持日线决策；接口保留时间戳以便后续扩展分钟数据。

## 4. 当前架构与缺口

### 4.1 当前调用链

```text
BacktestEngine
  -> strategy.get_stock_signal(state)
  -> StockSelection
  -> RebalanceStrategy.act(...)
  -> WeightAllocation
  -> ExecutionStrategy.act(...)
  -> OrderList
  -> Executor.execute_batch(...)
```

这个结构已经正确分离选股、调仓和执行，问题主要集中在模型训练和输入输出协议。

### 4.2 具体缺口

| 位置 | 当前状态 | 问题 |
|------|----------|------|
| `PolicyState.to_array()` | 未实现 | 没有特征名、顺序、dtype、归一化、缺失值和时点语义 |
| `StockSelection` | 只保存入选股票 | 无法保存全市场分数、置信度、模型 ID 和预测 horizon |
| `strategy.factory` | `if strategy.type` | 每新增方法都修改核心工厂，缺少组件注册机制 |
| `BacktestContext` | 暴露完整行情和 FactorRuntime | 模型可能绕过 as-of 边界读取未来数据；成交日 state 还包含当日完整 OHLC |
| Phase 1 | 对所有日期并行调用策略 | 不适合滚动训练、在线模型和依赖账户状态的模型 |
| run artifacts | 只保存交易和候选 | 缺少模型 manifest、预测、fold 指标、IC 和特征重要性 |
| 数据股票池 | 当前主要是静态 symbols | 训练可能产生生存者偏差，需逐日可交易股票池 |
| `RuleExecution` | 同时承担止损、止盈、持仓退出和订单生成 | 风险决策与执行机制耦合，目标权重可能被执行层隐式覆盖 |

## 5. 总体架构

### 5.1 两个平面

```text
┌──────────────────────── Research Plane ────────────────────────┐
│                                                                │
│ DataSource/QlibBinReader                                       │
│      │                                                         │
│      ├── UniverseProvider ── point-in-time universe            │
│      ├── FeatureBuilder  ── FeatureBatch                       │
│      └── LabelBuilder    ── LabelBatch                         │
│                 │                                              │
│                 ▼                                              │
│       DatasetBuilder + WalkForwardSplitter                     │
│                 │                                              │
│                 ▼                                              │
│       Trainer/ModelAdapter                                     │
│                 │                                              │
│                 ├── ModelArtifact                              │
│                 ├── FoldMetrics                                │
│                 └── OOS PredictionStore                        │
└───────────────────────────┬────────────────────────────────────┘
                            │ frozen OOS predictions/model
                            ▼
┌──────────────────────── Decision Plane ────────────────────────┐
│ MarketObservation                                              │
│      │                                                         │
│      ▼                                                         │
│ AlphaModel ──────────────> AlphaSnapshot                       │
│      │                                                         │
│      ▼                                                         │
│ PortfolioConstructor ────> RawTargetPortfolio                  │
│      │                                                         │
│      ▼                                                         │
│ RiskOverlay ─────────────> FinalTargetPortfolio                │
│      │                                                         │
│      ▼                                                         │
│ PendingRebalanceQueue                                          │
└───────────────────────────┬────────────────────────────────────┘
                            │ execute_not_before
                            ▼
┌──────────────────────── Execution Plane ───────────────────────┐
│ ExecutionContext(open/tradability only at execution time)      │
│      + PendingRebalance                                        │
│      -> OrderPlanner -> OrderList                              │
└───────────────────────────┬────────────────────────────────────┘
                            ▼
             Account + Exchange + Executor + BacktestEngine
```

### 5.2 为什么训练与回测分开

分开后可以做到：

- 一次训练，多次测试 Top10/Top20/Top50 和不同风控。
- 明确每一条预测是否属于样本外。
- 回测失败时不必重新训练。
- 模型性能与组合性能可以分别诊断。
- 训练依赖和回测依赖可以隔离。
- 同一份预测可以在现有可视化层展示。

训练与回测并非永远不能同进程，而是必须由外层 `ResearchRunner` 明确编排，不能由普通策略在 `act()` 中隐式训练。

## 6. 建议目录结构

```text
quantx/core/
  decision/
    __init__.py
    types.py                 # MarketObservation/PortfolioState/Alpha/Target
    clock.py                 # DecisionClock/ExecutionClock/SessionPhase
    alpha.py                 # AlphaModel Protocol
    portfolio.py             # PortfolioConstructor Protocol
    risk.py                  # RiskOverlay Protocol
    execution.py             # PendingRebalance/OrderPlanner/现有适配器
    queue.py                 # 待执行调仓队列
    pipeline.py              # DecisionPipeline
    registry.py              # 组件注册表
    adapters/
      formula.py             # 现有公式 selector 适配器
      predictions.py         # 冻结预测适配器
      legacy_policy.py       # CompositeStrategy 兼容层

  research/
    __init__.py
    specs.py                 # FeatureSpec/LabelSpec/DatasetSpec
    dataset.py               # DatasetBuilder
    data_version.py          # 逻辑 DataVersion/同步 manifest/训练期读锁
    snapshot.py              # promotion/精确复现时的可选物理快照
    universe.py              # PointInTimeUniverseProvider
    causality.py             # FeatureCausalityValidator
    split.py                 # WalkForwardSplitter/Purge/Embargo
    transform.py             # 仅在训练集 fit 的预处理流水线
    trainer.py               # Trainer Protocol/ResearchRunner
    artifacts.py             # ModelArtifact/Manifest/ArtifactStore
    predictions.py           # PredictionStore
    evaluation.py            # IC/RankIC/decay/fold metrics
    adapters/
      sklearn.py
      lightgbm.py
      pytorch.py
      qlib.py
      rl.py

quantx/models/               # 可选的具体模型实现，不进入 core 契约
  tabular/
  sequence/
  rl/

quantx/tools/
  run_research.py            # 训练、validation、OOS 预测
  run_backtest.py            # 继续作为唯一标准交易回测入口

configs/research/            # 确认有效后才进入项目
configs/strategies/
```

`core` 只定义协议和稳定数据类型；LightGBM、PyTorch 等具体实现放在 adapter 或 `quantx/models`。

## 7. 时间与可用性契约

### 7.1 为什么只有 `date` 不够

当前日线规则使用 `signal_date < execution_date`，已经可以阻止“当日收盘信号按当日开盘成交”。模型框架还需要区分：

- 行情事件发生时间。
- 数据真正可用时间。
- 特征计算截止时间。
- 信号产生时间。
- 最早允许成交时间。
- 标签结束时间。

### 7.2 市场阶段和统一时间类型

日线数据不能只用一个虚构的时间戳表示所有状态。接口同时保留交易 session 和 session phase：

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class SessionPhase(str, Enum):
    PRE_OPEN = "pre_open"
    OPEN = "open"
    INTRADAY = "intraday"
    CLOSE = "close"
    AFTER_CLOSE = "after_close"

@dataclass(frozen=True)
class MarketTime:
    session: str
    phase: SessionPhase
    timestamp: datetime

@dataclass(frozen=True)
class DecisionClock:
    observation_end: MarketTime
    data_available_at: MarketTime
    signal_time: MarketTime
    execute_not_before: MarketTime
```

所有时间戳使用 `Asia/Shanghai`，不能混用 naive datetime。数据源无法证明精确发布时间时，以保守 session phase 表示，不伪造早于真实可用时间的时间戳。

日线默认语义：

```text
T 日 OHLCV 完整数据：T 日 AFTER_CLOSE 才可用
信号时间：T 日 AFTER_CLOSE
最早成交：T+1 OPEN
```

任何预测必须满足：

```text
max(feature.available_at) <= signal_time < execute_not_before
```

训练标签还必须满足：

```text
label.end_time <= fold.training_information_cutoff
```

### 7.3 决策视图与账户视图分离

AlphaModel 不能直接得到完整 `BacktestContext`，也不能默认读取账户。它只读取市场本身：

```python
@dataclass(frozen=True)
class MarketObservation:
    clock: DecisionClock
    universe: tuple[str, ...]
    features: "FeatureBatch"

@dataclass(frozen=True)
class PortfolioState:
    as_of: MarketTime
    account: "AccountSnapshot"
    positions: "Mapping[str, PositionSnapshot]"
```

这样同一市场状态产生的 Alpha 与账户无关，可以批量预计算、缓存，并复用不同的组合策略。确实需要账户信息的 RL 模型应接入 `PortfolioConstructor` 或 `RiskOverlay`，而不是伪装成 AlphaModel。

`MarketObservation` 不提供：

- 任意结束日期的 `get_history()`。
- 完整未来市场矩阵。
- LabelProvider。
- 测试集真实收益。
- 现金和持仓。

规则策略兼容层可以暂时继续读取 `PolicyState`，新模型默认只能读取 `MarketObservation`。

### 7.4 决策事件与执行事件

决策阶段不直接返回最终订单，而是创建待执行调仓：

```python
@dataclass(frozen=True)
class PendingRebalance:
    rebalance_id: str
    created_at: MarketTime
    execute_not_before: MarketTime
    expires_at: MarketTime | None
    target: "FinalTargetPortfolio"
    source_signal_id: str
```

到执行事件发生时，引擎构建严格字段白名单的执行上下文：

```python
@dataclass(frozen=True)
class ExecutionContext:
    clock: MarketTime
    open_prices: "Mapping[str, float]"
    tradability: "Mapping[str, TradabilityState]"
    account: "AccountSnapshot"
    positions: "Mapping[str, PositionSnapshot]"
```

日线 `OPEN` 执行上下文不得包含当日 high、low、close、volume 等尚未完成字段。若执行策略需要这些字段，只能选择 `CLOSE` 或更晚的明确成交阶段。

事件顺序：

```text
T AFTER_CLOSE:
  MarketObservation -> Alpha -> RawTarget -> RiskOverlay
  -> PendingRebalance(created_at=T, execute_not_before=T+1 OPEN)

T+1 OPEN:
  ExecutionContext(open/tradability only) + PendingRebalance
  -> OrderPlanner -> Executor
```

引擎必须验证：

```text
pending.created_at < execution_context.clock
pending.execute_not_before <= execution_context.clock
pending.expires_at is None or execution_context.clock <= pending.expires_at
```

同一账户、同一执行阶段出现多个 pending rebalance 时不能静默叠加。队列键使用 `(account_id, strategy_instance_id, execute_not_before)`，默认 `replace_previous`：同一策略较新的完整目标替换较旧目标；跨策略组合必须先由上层 PortfolioAggregator 合成一个账户级最终目标。其他 merge policy 必须显式注册并记录在 run artifacts。

### 7.5 迁移期执行语义

迁移期明确区分两个模式：

```text
legacy_compat
  仅用于证明重构没有意外改变旧策略代码路径。
  可以保留旧 RuleExecution 的历史语义，但报告必须标记 legacy。
  不允许用于新模型、promotion 或生产模型结论。

strict_event
  决策事件严格早于执行事件。
  执行上下文使用 phase 字段白名单。
  所有新规则、ML、DL、RL 和正式 promotion 必须使用。
```

旧策略若存在“用当日 close 判断后又按同一 close 成交”的规则，迁移到 strict_event 后结果发生变化是预期行为，不能为了保持收益一致而放松时间契约。应分别保留 legacy regression baseline 和 strict migrated baseline。

## 8. 数据层复用设计

### 8.1 数据来源保持不变

训练和回测使用同一个 `provider_uri`：

```text
data/qlib_data_fixed
```

数据读取优先级：

1. 本地 `QlibBinReader`，保证 Mac 环境不依赖可编译的完整 Qlib。
2. 可用时使用 Qlib `D.features()` adapter。
3. `QMTDataSource` 只负责同步和补齐，不应在普通训练过程中逐样本联网。

`provider_uri` 是存储位置，不是数据版本。变化的数据可以正常用于训练和 validation：追加新交易日、补齐缺失值或修正历史数据都属于正常数据治理。要求不是永久冻结 provider，而是让一次实验知道自己读到的是哪个逻辑版本，并保证单次运行期间读取口径稳定。

### 8.2 逻辑 DataVersion 与按需物理快照

当前 QMT 默认使用 `front_ratio`，新增公司行动后，前复权可能重写完整历史价格。因此“相同 provider 路径 + 相同日期”不保证数据相同。

```python
@dataclass(frozen=True)
class DataVersion:
    version_id: str
    provider_uri: str
    created_at: datetime
    calendar_hash: str
    instruments_hash: str
    feature_manifest_hash: str
    universe_version_id: str
    adjustment_mode: str
    corporate_action_version: str | None
    source_sync_id: str
    revision_policy: str
    physical_snapshot_id: str | None
```

第一阶段普通训练和 validation 的最低要求：

1. 数据同步完成后生成 sync manifest。
2. 记录日历、instruments 和字段文件的分块摘要。
3. 记录复权类型和公司行动数据版本。
4. ResearchRunner 获取 provider 读锁后解析逻辑 `version_id`，训练结束前同步任务不得修改同一目录。
5. provider 更新后生成新的 `version_id`，旧 artifact 仍保留它使用的版本和 manifest 引用。
6. `version_id` 变化自动使 feature/fold/model cache 失效。

普通探索不复制整份 Qlib 数据。只有以下情况才创建只读目录、硬链接或 copy-on-write 的物理不可变快照，并把 `physical_snapshot_id` 写入 DataVersion：

- 创建 promotion candidate 并准备 sealed holdout。
- 需要跨机器或长期精确复现实验。
- 数据同步与研究任务无法通过读写锁隔离。

逻辑版本保证“知道用了什么数据”，训练期读锁保证“这次运行中数据不变化”，物理快照保证“未来还能逐字节恢复”。三者解决的问题不同，不应把最昂贵的物理快照作为每次探索的前置条件。长期方案仍应保存不复权价格和公司行动事件，以便按研究时点构造可审计复权数据。

前复权并不必然让所有收益率特征失效，但未来公司行动会改变历史绝对价格、跨股票比较和数据 checksum；因此任何基于当前最新前复权数据的结果都必须标记为 `revised_adjusted_history`，不能声称是当时可直接观察到的原始价格快照。

### 8.3 DatasetBuilder 不复制数据源

`DatasetBuilder` 是现有数据层之上的只读适配器：

```python
class MarketDataReader(Protocol):
    def features(
        self,
        symbols: Sequence[str],
        fields: Sequence[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame: ...

    def calendar(self, start_time: str, end_time: str) -> Sequence[str]: ...
```

现有 `QlibBinReader` 可以直接适配这个协议。

### 8.4 特征复用

两种特征来源统一为 `FeatureBatch`：

1. `FactorRuntimeFeatureBuilder`
   - 复用当前公式 DAG、`MarketPanel` 和横截面算子。
   - 适合 Alpha158 风格价量特征和人工因子。
2. `RawWindowFeatureBuilder`
   - 输出 `[sample, lookback, feature]` 序列。
   - 适合 LSTM、TCN、Transformer、MASTER。

```python
@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    dtypes: tuple[str, ...]
    lookback_sessions: int
    frequency: str
    max_lookahead_sessions: int
    causality_hash: str
    schema_hash: str

@dataclass(frozen=True)
class FeatureBatch:
    as_of: datetime
    instruments: tuple[str, ...]
    values: Any
    schema: FeatureSchema
    available_at: datetime
    valid_mask: Any | None = None
```

模型不得自行猜测列顺序；模型 artifact 的 `schema_hash` 必须与推理批次一致。

### 8.5 特征因果校验

所有 FeatureBuilder 必须给出每个输出字段相对信号时点的读取范围。FactorRuntime 算子在 registry 中声明时间语义：

```text
close              -> [0, 0]
Ref(x, 5)          -> shift(input_range, -5)
Mean(x, 20)        -> [input_min - 19, input_max]
EMA(x, 20)         -> [history_start, input_max]
CSRank(x)          -> input_range，不改变时间范围
Ref(x, -1)         -> [1, 1]，作为正式特征时拒绝
```

```python
@dataclass(frozen=True)
class ReadWindow:
    min_offset_sessions: int
    max_offset_sessions: int

class FeatureCausalityValidator(Protocol):
    def analyze(self, spec: FeatureSpec) -> Mapping[str, ReadWindow]: ...

    def require_causal(self, spec: FeatureSpec) -> None: ...
```

正式 feature 的递归读取窗口必须满足：

```text
max_offset_sessions <= 0
```

LabelBuilder 使用独立的算子命名空间，允许正向读取。FeatureBuilder 不得 import LabelProvider，也不得通过自定义 Python 函数绕过读取窗口声明。无法静态分析的自定义特征必须实现受审计的 `read_window()`，否则拒绝进入正式研究。

除静态检查外，还保留 mutation test：修改信号日之后的行情，历史特征值和预测必须逐值不变。

### 8.6 Point-in-time 股票池

这是进入 ML 前的前置条件。

```python
class UniverseProvider(Protocol):
    def members(self, session: str) -> Sequence[str]: ...
```

股票池至少考虑：

- 上市日期和退市日期。
- 当日是否存在行情。
- 停牌和交易状态。
- ST/风险警示状态，如策略选择排除。
- 板块归属。
- 指数成分历史，如使用 CSI300/CSI500。

当前 `QMTDataSource.get_stock_list(date)` 实际上主要返回本地当前股票列表，不能直接作为 point-in-time universe。实现 ML 研究前需要补齐历史退市股票和逐日成员关系，否则会产生生存者偏差。

Point-in-time universe 不能只靠随机抽查验收。每个 DataVersion 必须输出并校验：

```text
每日成员数量及异常跳变
每日新增上市/退市数量
上市日期缺失率
退市日期和退市行情覆盖率
universe 成员无行情比例
ST/停牌状态覆盖率
指数成分历史覆盖率（使用指数池时）
```

缺少历史退市股票时，`pit_all_mainboard` 必须标记为不可用于正式 ML promotion。第一批模型实验应优先使用具备历史成分记录的指数池；如果历史成分也不完整，只能作为 exploratory run，不能进入 promotion gate。

每日样本 eligibility 必须完全由信号时点已知信息决定。不能因为第二天买不到、未来退市或未来标签缺失而提前修改 T 日股票池。

门槛必须写入配置并进入 dataset manifest，例如：

```yaml
universe_audit:
  max_missing_list_date_ratio: 0.001
  max_member_without_quote_ratio: 0.01
  require_delisted_history: true
  require_historical_constituents: true
  max_unexplained_daily_count_jump: 0.05
```

任一门槛失败时，正式模式 fail-fast；exploratory 模式可以继续，但 artifact 和报告必须显著标记 `universe_integrity=failed`。

### 8.7 元数据时点

当前名称、行业和概念 snapshot 适合展示，但不能默认作为历史训练特征。只有包含生效区间的数据才能进入模型：

```text
symbol, group_id, effective_from, effective_to, source, snapshot_date
```

没有历史版本时，行业/概念只能用于当前展示或明确标记的研究诊断，不能用于正式样本外训练。

## 9. 特征、标签和数据集定义

### 9.1 FeatureSpec

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    builder: str
    expressions: Mapping[str, str]
    raw_fields: tuple[str, ...]
    lookback_sessions: int
    normalization: str
    missing_policy: str
    require_causal: bool = True
```

推荐第一版特征族：

- 1/5/10/20/60/120 日收益。
- 残差动量和市场相对强度。
- 距离 20/60/120 日均线和阶段高低点。
- 实现波动率、下行波动率、振幅和 gap。
- 成交量均值、量比、量价背离和流动性。
- 横截面 rank/z-score。

### 9.2 LabelSpec

标签必须明确入场和退出价格，而不是只写“未来收益”：

```python
@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon_sessions: int
    entry_price: str          # next_open
    exit_price: str           # future_open
    benchmark: str | None
    excess_return: bool
    winsorize: tuple[float, float] | None
    missing_exit_policy: str
    execution_diagnostic: bool
```

例如信号日为 T、horizon=10：

```text
entry = T+1 open
exit  = T+11 open
label = exit / entry - 1
```

如使用超额收益：

```text
label = stock_return - benchmark_return
```

标签只在研究平面可见，决策平面不得 import 或持有 LabelProvider。

### 9.3 理论标签与可成交结果分离

在 T 日生成样本时，系统不知道 T+1 是否停牌、一字涨停、无有效开盘价或成交量为零。不能先查看 T+1 的可交易状态再决定是否保留 T 日样本，否则会用未来信息筛选训练集。

每个样本分别保存：

```text
theoretical_label
  按 LabelSpec 计算的理论未来收益；不存在必要价格时明确为 missing。

execution_status
  标准回测在 T+1 的 FILLED/REJECTED/SUSPENDED/LIMIT_UP/NO_QUOTE 状态。

realized_strategy_return
  组合层和执行层考虑未成交、现金、滑点和成本后的真实贡献。
```

`execution_status` 和 `realized_strategy_return` 是评估结果，不是 T 日模型特征。LabelBuilder 不得根据未来状态静默删除样本。标签缺失必须按 `missing_exit_policy` 显式处理并报告缺失率及其股票分布。

第一版推荐：

```text
模型训练目标：theoretical_label
模型 IC 评估：theoretical_label + label coverage
策略有效性：标准 BacktestEngine 的实际成交和组合收益
```

如后续研究可成交概率，应单独训练 `TradabilityModel`，其输入仍只能使用 T 日已知特征。

### 9.4 DatasetSpec

```python
@dataclass(frozen=True)
class DatasetSpec:
    data_version_id: str
    physical_snapshot_id: str | None
    provider_uri: str
    universe: str
    start: str
    end: str
    features: FeatureSpec
    label: LabelSpec
    sample_filter: Mapping[str, Any]
    universe_snapshot_id: str
```

`data_version_id` 由以下信息产生 hash：

- provider 日历文件摘要。
- instruments 文件摘要。
- 特征字段文件摘要或同步批次 ID。
- universe snapshot/version。
- 复权方式。
- 公司行动版本和同步批次 ID。

## 10. 时间切分与 validation

### 10.1 禁止随机切分

股票样本在同一天高度相关，随机拆行会让同一市场状态同时进入 train 和 validation。统一框架只允许时间序列切分器。

### 10.2 Anchored walk-forward

第一版默认采用扩展窗口：

```text
Fold 1: train 2016-2019, valid 2020, development_oos 2021
Fold 2: train 2016-2020, valid 2021, development_oos 2022
Fold 3: train 2016-2021, valid 2022, development_oos 2023
Fold 4: train 2016-2022, valid 2023, development_oos 2024
```

也支持固定长度 rolling window，但必须在配置中显式声明。

### 10.3 三级评估和最终 holdout

只要研究者查看某个区间结果并据此修改特征、模型、TopK 或风控，该区间就已经参与开发，不能继续称为最终 test。框架明确区分：

```text
development_oos
  可反复运行的 walk-forward 区间，用于研究和淘汰方案。

sealed_holdout
  在特征、模型、超参数、组合和执行配置全部冻结后运行一次。
  运行前只保存加密或不可变的 experiment lock，不展示期间结果。

live_shadow
  从方案冻结日之后积累的模拟实盘，任何历史回测都不能替代。
```

推荐生命周期：

```text
1. 使用 development_oos 研究。
2. 生成 ExperimentLock，包含全部 config 和 artifact hash。
3. 执行 sealed_holdout，结果写入只读评估记录。
4. 无论 holdout 好坏，都不能回头修改后再次声称使用同一 holdout。
5. 修改方案后创建新的 research generation，并依赖新的 live_shadow 证明。
```

由于项目已经观察过 2025-2026 的市场和策略表现，这些年份最多是弱封存历史评估，不是完全未见数据。当前项目最终可信证据必须包含从模型冻结日开始的 live shadow。

ResearchRunner 应记录：

```text
evaluation_tier: development_oos | sealed_holdout | live_shadow
research_generation: integer
experiment_lock_hash: sha256
result_viewed_at: timestamp
```

`ExperimentLock` 是 sealed holdout 的写前锁，至少包含：

```python
@dataclass(frozen=True)
class ExperimentLock:
    research_generation: int
    normalized_research_config_hash: str
    normalized_backtest_config_hash: str
    data_version_id: str
    physical_snapshot_id: str
    universe_snapshot_id: str
    feature_schema_hash: str
    causality_hash: str
    label_spec_hash: str
    model_artifact_checksums: tuple[str, ...]
    seed_set: tuple[int, ...]
    portfolio_config_hash: str
    risk_config_hash: str
    order_planner_config_hash: str
    code_fingerprint: str
    holdout_range: tuple[str, str]
    created_at: datetime
```

创建 lock 后，上述任一字段变化都必须创建新的 research generation。sealed holdout runner 使用 append-only ledger 记录 lock hash、开始时间、结束时间和结果 URI；同一 lock hash 已有记录时拒绝再次运行。

### 10.4 Purge 和 embargo

如果标签 horizon 为 20 日，训练集末尾样本的标签可能跨入 validation。切分器必须删除这些样本：

```text
train sample label_end < validation_start
validation sample label_end < development_oos_start
```

`purge_sessions` 默认由 label horizon 和 execution lag 自动计算，不能默认为 0。

`embargo_sessions` 用于减少相邻区间高度重叠，默认可设为 horizon，后续通过配置调整。

### 10.5 预处理 fit 边界

以下操作只能在 train 上 fit：

- 均值/标准差。
- median/MAD。
- 缺失值填充值。
- PCA/autoencoder。
- 特征选择。
- winsorize 阈值。
- 超参数优化。

validation、development OOS、sealed holdout 和 live shadow 只能调用 `transform()`。

横截面当日 rank/z-score 属于当日可观测变换，可以按当日计算，但不能使用未来日期统计量。

## 11. Trainer 与模型适配器

### 11.1 Trainer 协议

```python
class Trainer(Protocol):
    method_type: str

    def fit(
        self,
        train: "ResearchDataset",
        validation: "ResearchDataset",
        context: "TrainingContext",
    ) -> "ModelArtifact": ...

    def predict(
        self,
        artifact: "ModelArtifact",
        dataset: "ResearchDataset",
    ) -> "PredictionFrame": ...
```

`Trainer` 不直接调用 `BacktestEngine`。validation 阶段可由 `ResearchRunner` 把预测交给快速组合评估器，但最终 promotion 必须走标准回测引擎。

### 11.2 适配器职责

| Adapter | 输入形状 | 输出 | 典型实现 |
|---------|----------|------|----------|
| sklearn | `[samples, features]` | score | Linear/RandomForest |
| LightGBM | `[samples, features]` | score | regression/rank objective |
| PyTorch tabular | `[samples, features]` | score | MLP/TabNet |
| PyTorch sequence | `[samples, time, features]` | score | LSTM/TCN/Transformer |
| Qlib | Qlib DatasetH | score | LGBModel/DoubleEnsemble/TRA/HIST |
| RL | environment trajectory | action/weight | PPO/SAC/离线 RL |

框架不要求模型对象实现统一的 `fit()` 签名，由 adapter 消化框架差异。

### 11.3 模型产物

```python
@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    method_type: str
    model_uri: str
    model_format: str
    model_checksum: str
    loader_name: str
    loader_version: str
    trust_level: str
    manifest_uri: str
    feature_schema_hash: str
    data_version_id: str
    physical_snapshot_id: str | None
    fold_id: str
    created_at: datetime
```

每个 artifact 目录：

```text
artifacts/research/<experiment_id>/
  experiment.yaml
  dataset_manifest.json
  folds/
    fold_2021/
      manifest.json
      model.<adapter-format>
      preprocessing.<adapter-format>
      validation_metrics.json
      development_oos_predictions.parquet
      feature_importance.json
    fold_2022/
      ...
  oos_predictions.parquet
  aggregate_metrics.json
  logs.txt
```

正式项目仍遵循已有约定：研究中的模型、config 和 run 先放 `/tmp`；通过 promotion gate 后才复制到项目 artifacts/configs。

模型和预处理器加载都具有代码执行风险。LightGBM text/JSON、ONNX、受控 Torch `state_dict`、skops 等可审计格式优先；pickle/joblib 只允许加载本机生成且 manifest、checksum、adapter allowlist 全部匹配的可信 artifact。任何从网络下载或来源不明的 pickle 都不得加载。

## 12. Model Manifest

`manifest.json` 至少包含：

```json
{
  "artifact_id": "lgbm_alpha158_fold_2024_abcd1234",
  "method_type": "lightgbm",
  "code_commit": "git-sha",
  "dirty_diff_hash": "sha256:clean-or-diff-hash",
  "code_fingerprint": "sha256:commit-plus-dirty-diff",
  "data_version_id": "qmt-sync-20260712-abcd1234",
  "physical_snapshot_id": null,
  "feature_schema_hash": "sha256:...",
  "universe_version": "pit-mainboard-v1",
  "label": {
    "horizon_sessions": 10,
    "entry_price": "next_open",
    "exit_price": "future_open",
    "excess_return": true
  },
  "train_range": ["2016-01-01", "2022-12-31"],
  "validation_range": ["2023-01-01", "2023-12-31"],
  "prediction_range": ["2024-01-01", "2024-12-31"],
  "evaluation_tier": "development_oos",
  "research_generation": 1,
  "purge_sessions": 11,
  "embargo_sessions": 10,
  "seed": 7,
  "model": {
    "format": "lightgbm_text",
    "checksum": "sha256:...",
    "loader_name": "quantx.lightgbm",
    "loader_version": "1",
    "trust_level": "local_verified"
  },
  "dependencies": {
    "python": "3.x",
    "lightgbm": "x.y.z"
  }
}
```

缺少 manifest 的模型不得进入标准回测或生产信号流程。

`code_commit` 不能表示未提交修改。工作区干净时 `dirty_diff_hash` 使用固定 `clean` 值；工作区有修改时，对规范化 `git diff` 和未跟踪研究代码计算 hash。无法形成稳定 code fingerprint 的 artifact 只能用于 exploratory run。

## 13. PredictionStore

### 13.1 标准预测表

```text
signal_time
signal_session
instrument
score
rank
uncertainty          optional
prediction_horizon
artifact_id
fold_id
feature_schema_hash
```

联合主键：

```text
(signal_time, instrument, artifact_id)
```

### 13.2 OOS 约束

每条预测写入时验证：

```text
signal_session belongs to artifact.prediction_range
artifact.train_range ends before prediction information boundary
feature_schema_hash matches artifact
prediction is unique
```

拼接多个 fold 时，如果同一日期出现多个预测，默认报错，不能静默选最新模型。

### 13.3 为什么先保存预测再回测

同一份预测可以运行：

- Top10/20/50。
- 等权、分数权重、风险平价。
- 每日/每周调仓。
- 不同换手约束。
- 不同止损和市场风险开关。

这样可以区分“模型没有预测力”和“组合/执行设计不合理”。

## 14. 决策平面协议

### 14.1 AlphaSnapshot

```python
@dataclass(frozen=True)
class AlphaValue:
    instrument: str
    score: float
    rank: int | None = None
    uncertainty: float | None = None

@dataclass(frozen=True)
class AlphaSnapshot:
    signal_time: datetime
    earliest_execution_time: datetime
    values: tuple[AlphaValue, ...]
    source_id: str
    horizon_sessions: int | None
    metadata: Mapping[str, Any]
```

公式策略和模型策略都输出完整评分；是否选择 TopK 由组合层决定。为了兼容现有代码，适配器可以把 `AlphaSnapshot` 转成 `StockSelection`。

### 14.2 AlphaModel

```python
class AlphaModel(Protocol):
    def prepare(self, context: "InferenceContext") -> None: ...

    def predict(self, observation: MarketObservation) -> AlphaSnapshot: ...

    def describe(self) -> Mapping[str, Any]: ...
```

主要实现：

- `FormulaAlphaModel`
- `PrecomputedAlphaModel`
- `SklearnAlphaModel`
- `TorchAlphaModel`
- `QlibAlphaModel`

第一版回测只要求 `PrecomputedAlphaModel`；直接在线加载 sklearn/torch 模型是后续便利能力。

### 14.3 RawTargetPortfolio 与 FinalTargetPortfolio

```python
@dataclass(frozen=True)
class RawTargetPortfolio:
    decision_time: datetime
    earliest_execution_time: datetime
    weights: Mapping[str, float]
    cash_weight: float
    source_signal_id: str
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class FinalTargetPortfolio:
    decision_time: datetime
    earliest_execution_time: datetime
    weights: Mapping[str, float]
    cash_weight: float
    source_signal_id: str
    applied_risk_rules: tuple[str, ...]
    metadata: Mapping[str, Any]
```

`RawTargetPortfolio` 表示模型或组合构建器的原始意图；`FinalTargetPortfolio` 是经过止损、市场状态、集中度、组合回撤和监管约束后的唯一可执行目标。

两种目标都执行运行时校验：

- 所有权重有限且非 NaN。
- long-only 模式权重不小于 0。
- `sum(weights) + cash_weight` 在允许误差内等于 1。
- 单票、行业和组合上限满足配置。
- 标的属于当日 universe。

### 14.4 PortfolioConstructor

```python
class PortfolioConstructor(Protocol):
    def construct(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        alpha: AlphaSnapshot,
    ) -> RawTargetPortfolio: ...
```

具体实现：

- `TopKEqualWeightPolicy`
- `ScoreWeightedPolicy`
- `RiskParityPolicy`
- `ConstrainedOptimizerPolicy`
- `RegimeRotationPolicy`
- `RLAllocationPolicy`

现有 `EqualWeightRebalance` 可以通过 adapter 实现该协议。

### 14.5 RiskOverlay

```python
class RiskOverlay(Protocol):
    def apply(
        self,
        market: MarketObservation,
        portfolio: PortfolioState,
        raw_target: RawTargetPortfolio,
    ) -> FinalTargetPortfolio: ...
```

RiskOverlay 负责：

- 单票、行业和风格暴露上限。
- 市场 regime 风险开关。
- 账户回撤降仓。
- 个股止损、止盈、移动止盈和持仓时间退出。
- T+1 不能卖出的持仓约束提示。
- 模型置信度不足时保留现金。

风险规则必须通过修改目标权重表达，例如止损等价于把该股票最终目标权重设为 0。它不能直接生成卖单，实际可卖数量仍由执行阶段确定。

RiskOverlay 按固定优先级组合，不能依赖配置遍历的偶然顺序：

```text
合规/账户硬限制
-> 市场和组合风险限制
-> 个股退出规则
-> 组合构建器原始目标
```

如果 T+1、停牌或跌停导致最终目标暂时无法实现，OrderPlanner 记录 `target_execution_gap` 和拒绝原因，账户保留实际持仓；下一决策事件重新应用 RiskOverlay。风险层不能伪造已经完成的清仓。

现有 `RuleExecution` 中的 sell rules 在 Phase 1 通过 `LegacyRiskExecutionAdapter` 保持原行为；目标架构应逐步把这些规则迁移到 RiskOverlay，避免长期双重决策。

### 14.6 PendingRebalance 与 OrderPlanner

决策阶段把 FinalTargetPortfolio 包装成 `PendingRebalance` 放入队列，不读取下一交易日价格。

```python
class OrderPlanner(Protocol):
    def create_orders(
        self,
        execution: ExecutionContext,
        pending: PendingRebalance,
    ) -> "OrderList": ...
```

OrderPlanner 只负责：

- 根据执行时点价格把目标权重转换为股数。
- 计算当前仓位与目标仓位的差。
- 处理手数、现金预算和卖出资金复用。
- 根据当时停牌、涨跌停和报价状态生成订单或拒绝原因。
- 把订单交给 Executor 做最终规则校验。

RL 执行模型也必须输出标准订单意图，最终订单仍由 `Executor` 校验，不能绕过交易规则。

### 14.7 DecisionPipeline 与 ExecutionPipeline

```python
class DecisionPipeline:
    def __init__(self, alpha, portfolio, risk):
        self.alpha = alpha
        self.portfolio = portfolio
        self.risk = risk

    def decide(
        self,
        market: MarketObservation,
        portfolio_state: PortfolioState,
    ) -> PendingRebalance:
        alpha = self.alpha.predict(market)
        raw = self.portfolio.construct(market, portfolio_state, alpha)
        final = self.risk.apply(market, portfolio_state, raw)
        return PendingRebalance.from_target(final)

class ExecutionPipeline:
    def __init__(self, planner, executor):
        self.planner = planner
        self.executor = executor

    def execute(self, context, pending):
        orders = self.planner.create_orders(context, pending)
        return self.executor.execute_batch(orders)
```

引擎调度 DecisionEvent 和 ExecutionEvent，Alpha 来源仍对 Account/Exchange/Executor 透明。

## 15. 各类技术如何接入

### 15.1 现有公式策略

```text
FactorRuntime formulas
  -> FormulaAlphaModel
  -> AlphaSnapshot
  -> TopK/EqualWeight PortfolioConstructor
  -> RiskOverlay
  -> PendingRebalance
  -> StandardOrderPlanner
```

迁移初期允许 `LegacyCompositeStrategyAdapter` 直接包裹现有 `CompositeStrategy`，确保行为完全一致。

### 15.2 sklearn/LightGBM

训练：

```text
FeatureBatch -> tabular DataFrame -> Trainer.fit -> model.bin
```

推理：

```text
FeatureBatch -> predict -> score per instrument -> AlphaSnapshot
```

第一轮推荐同时训练：

- Linear/Ridge baseline。
- LightGBM regression。
- LightGBM ranking，如数据组织验证后启用。
- DoubleEnsemble 作为第二阶段基准。

### 15.3 深度学习

深度学习不需要新的回测接口，只需新的 FeatureBuilder 和 Trainer：

```text
RawWindowFeatureBuilder
  -> [batch, time, feature]
  -> TorchTrainer
  -> TorchAlphaModel/PredictionStore
  -> AlphaSnapshot
```

必须与相同标签、相同 fold、相同组合策略下的 LightGBM 比较，不能使用不同口径证明提升。

### 15.4 Qlib 模型

Qlib adapter 负责：

- 把当前 provider 组装成 Qlib DatasetH。
- 映射 FeatureSpec/LabelSpec。
- 调用 Qlib model fit/predict。
- 把 Qlib prediction Series 转换为 PredictionStore。

Qlib 自带回测只用于模型开发诊断，最终交易绩效必须回到 QuantX 标准引擎，保证 A 股成本和时序口径一致。

### 15.5 AlphaGen/RD-Agent/自动因子挖掘

自动研究工具不直接获得“正式策略”权限。输出流程：

```text
自动生成公式/模型
  -> 注册为候选 FeatureSpec/Trainer config
  -> 固定 walk-forward
  -> 写入临时 artifact
  -> promotion gate
```

研究 agent 不得反复读取 sealed holdout 结果继续优化。sealed holdout 必须在 ExperimentLock 后一次性运行，之后只能依赖按时间推进的 live shadow。

## 16. RL 专项设计

### 16.1 RL 不等于新的回测系统

RL 训练需要 `reset/step` 环境，但底层仍复用：

- `Account`
- `AStockExchange`
- `Executor`
- `TransactionCost`
- 交易日历和数据视图

应提取可单步推进的 `SimulationKernel`，由 `BacktestEngine` 和 `QuantTradingEnv` 共同调用，而不是复制撮合逻辑。

```text
BacktestEngine.run() -> SimulationKernel.step(decision)
RL Env.step(action)  -> SimulationKernel.step(decision)
```

### 16.2 环境协议

```python
class QuantTradingEnv:
    def reset(self, seed=None, options=None):
        return observation, info

    def step(self, action):
        return observation, reward, terminated, truncated, info
```

### 16.3 三种 RL 接入方式

1. Alpha RL
   - action 表示股票评分或选择。
   - 输出适配成 `AlphaSnapshot`。
2. Allocation RL
   - 输入 Alpha、风险和持仓。
   - action 表示目标权重。
   - 输出 `RawTargetPortfolio`，之后仍经过统一 RiskOverlay。
3. Execution RL
   - 输入目标组合、盘口/分钟状态。
   - action 表示拆单比例、价格和时间。
   - 输出 `OrderIntent`。

当前只有日线数据，第一批 RL 实验最多适合 Allocation RL。Execution RL 需要分钟或盘口数据。

### 16.4 Reward

reward 必须在成本后计算，默认建议：

```text
portfolio_return_after_cost
- turnover_penalty
- drawdown_penalty
- constraint_violation_penalty
```

reward 配置必须写入 manifest。不能只使用最终累计收益，否则训练不稳定且难诊断。

### 16.5 RL validation

- 训练 episode 只能来自 train 区间。
- 超参数和 early stopping 只看 validation。
- sealed holdout 和 live shadow 环境冻结 agent，不允许梯度更新和 replay buffer 学习。
- 多随机种子报告均值、标准差和最差结果。
- 与简单等权/风险开关基线比较。

## 17. 组件注册与配置加载

### 17.1 注册表

核心工厂不再为每种方法增加 `if`：

```python
registry.register("alpha", "formula", build_formula_alpha)
registry.register("alpha", "predictions", build_prediction_alpha)
registry.register("portfolio", "topk_equal", build_topk_equal)
registry.register("portfolio", "rl", build_rl_allocator)
registry.register("risk", "standard", build_standard_risk)
registry.register("order_planner", "standard", build_standard_order_planner)
```

注册项声明 capability：

```python
@dataclass(frozen=True)
class ComponentCapabilities:
    needs_account_state: bool
    supports_batch_prediction: bool
    supports_online_update: bool
    output_type: str
    required_optional_extra: str | None
```

引擎根据 capability 决定是否允许批量预计算，而不是依赖模糊的 `precompute_stock_signals` 布尔值。

### 17.2 插件边界

第一阶段插件必须在代码中显式注册。后续可以支持 Python entry points，但配置不能填写任意 module/class 路径，避免：

- 配置执行任意代码。
- 无法追踪依赖。
- 模型升级后旧 run 无法复现。

## 18. 配置设计

### 18.1 研究配置

```yaml
name: mainboard_lgbm_10d_v1
version: 1

data:
  provider_uri: data/qlib_data_fixed
  data_version: latest_synced
  physical_snapshot: promotion_only
  universe: pit_all_mainboard
  start: 2016-01-04
  end: latest

features:
  builder: factor_runtime
  require_causal: true
  lookback_sessions: 160
  expressions:
    ret5: close / Ref(close, 5) - 1
    ret20: close / Ref(close, 20) - 1
    vol20: Std(close / Ref(close, 1) - 1, 20)
    distance_ma60: close / Mean(close, 60) - 1
  normalization: daily_cross_section_robust_zscore
  missing_policy: preserve_with_mask

label:
  horizon_sessions: 10
  entry_price: next_open
  exit_price: future_open
  benchmark: SH000905
  excess_return: true
  missing_exit_policy: mark_missing
  execution_diagnostic: true

validation:
  type: anchored_yearly_walk_forward
  train_start: 2016-01-04
  first_validation_year: 2020
  first_development_oos_year: 2021
  development_oos_end: 2024-12-31
  sealed_holdout:
    start: 2025-01-01
    end: 2026-07-10
    require_experiment_lock: true
  live_shadow:
    enabled: true
  purge_sessions: auto
  embargo_sessions: 10

model:
  type: lightgbm
  objective: regression
  seed: 7
  params:
    learning_rate: 0.03
    num_leaves: 31

output:
  root: /tmp/quantx-research/mainboard_lgbm_10d_v1
```

### 18.2 回测配置

```yaml
name: mainboard_lgbm_top20_v1
version: 1

strategy:
  type: decision_pipeline

  alpha:
    type: predictions
    artifact: /tmp/quantx-research/mainboard_lgbm_10d_v1

  portfolio:
    type: topk_equal
    topk: 20
    max_weight: 0.05
    cash_use_ratio: 0.98
    rebalance: weekly

  risk:
    type: standard
    max_account_drawdown: 0.20
    max_industry_weight: 0.20
    stop_loss: null

  order_planner:
    type: standard
    execute_at: next_open
    lot_size: 100
    skip_limit_up: true

data:
  provider_uri: data/qlib_data_fixed
  data_version_id: qmt-sync-20260712-abcd1234
  physical_snapshot_id: null
  universe: pit_all_mainboard
  start: 2021-01-04
  end: latest

cost:
  commission_rate: 0.0005
  min_commission: 5
  stamp_tax_rate: 0.0005
  slippage: 0.0002
```

研究配置和回测配置分开，防止修改组合参数时意外重新训练或修改模型训练口径。

## 19. ResearchRunner 工作流

```text
1. validate config
2. acquire provider read lock and resolve/verify logical DataVersion
3. validate point-in-time universe coverage
4. run FeatureCausalityValidator
5. build FeatureBatch and LabelBatch
6. create purged walk-forward development folds
7. for each development fold:
     fit preprocessing on train
     fit model on train
     tune/early-stop on validation
     freeze artifact
     predict development_oos
     write fold artifact
8. concatenate development OOS predictions
9. compute IC/RankIC/decay/stability
10. call standard run_backtest with prediction strategy
11. when explicitly requested for promotion, create physical snapshot and immutable ExperimentLock
12. run sealed holdout exactly once against the locked experiment
13. register the frozen model/config for live shadow
14. write aggregate research report with evaluation tiers separated
```

第 10 和 12 步使用标准回测引擎，不能用训练框架自带的简化收益计算替代最终结果。development OOS、sealed holdout 和 live shadow 指标不得合并成一个看似更长的“test”收益。

## 20. 评估与报告

### 20.0 预期 CLI 生命周期

实现后的标准使用方式：

```bash
# 只校验数据覆盖、股票池、特征、标签和 fold，不训练
python -m quantx.tools.run_research \
  --config configs/research/mainboard_lgbm_10d_v1.yaml \
  --dry-run --json

# 训练所有 walk-forward fold，并生成严格 OOS predictions
python -m quantx.tools.run_research \
  --config configs/research/mainboard_lgbm_10d_v1.yaml \
  --output-dir /tmp/quantx-research \
  --json

# 标准回测只消费已生成的 artifact
python -m quantx.tools.run_backtest \
  --config /tmp/quantx-research/mainboard_lgbm_top20_v1.yaml \
  --output-dir /tmp/quantx-research/runs \
  --json
```

`run_research` 负责训练和模型验证，`run_backtest` 仍是唯一正式交易回测入口。

### 20.1 模型指标

- Pearson IC。
- Rank IC。
- ICIR/Rank ICIR。
- IC decay：1/5/10/20 日。
- Top/Bottom quantile return。
- coverage 和缺失率。
- theoretical label coverage、T+1 execution status 分布和未成交率。
- 不同年份、行业、市值/流动性桶的稳定性。
- 多 seed 均值、标准差和最差值。

### 20.2 组合指标

继续使用标准指标，并增加：

- turnover。
- 平均资金利用率。
- 最近 30/60 日资金利用率。
- 行业和单票集中度。
- 收益贡献 Top10 股票/行业。
- 相对 benchmark 的超额收益和 tracking error。
- 不同 fold 的年度收益。
- target execution gap、pending 过期率和各拒单原因占比。

### 20.3 新增 run artifacts

```text
runs/<run_id>/
  existing files...
  model_manifest.json
  prediction_summary.json
  fold_metrics.json
  ic_timeseries.json
  exposure_summary.json
  dataset_snapshot.json
  decision_targets.json
  risk_overlay_events.json
  pending_rebalances.json
  execution_gaps.json
```

完整逐股预测保存在 research artifact，通过 ID 引用，不默认复制进每个 run，避免 run 目录膨胀。

## 21. 可复现性与缓存

### 21.1 Experiment ID

```text
experiment_id = hash(
  normalized_config
  + code_fingerprint(commit + dirty_diff_hash)
  + data_version_id
  + feature_schema_hash
  + causality_hash
  + dependency_versions
)
```

### 21.2 缓存层级

```text
raw market data        logical data_version_id (+ optional physical_snapshot_id)
feature cache          data_version_id + feature_schema_hash + causality_hash
fold dataset cache     feature cache + split spec
model cache            fold dataset + model config + seed
prediction cache       model artifact + inference dataset
backtest cache         prediction + portfolio/risk/order-planner config
```

缓存命中必须校验 manifest 和 checksum，不能只判断文件存在。对 5000 多只股票逐次重新 hash 全部 bin 成本较高，因此分块 hash 在数据同步时生成；普通研究只验证逻辑版本、sync manifest 和抽样文件完整性，正式 promotion 创建物理快照并执行全量完整性校验。

### 21.3 随机性

所有支持随机性的 Trainer 必须：

- 接收 seed。
- 记录 seed。
- promotion 前至少运行多个 seed。
- 报告分散度，不能只保留最好 seed。

## 22. 依赖管理

核心安装保持轻量：

```toml
[project.optional-dependencies]
ml = ["scikit-learn", "lightgbm"]
dl = ["torch"]
rl = ["gymnasium", "stable-baselines3"]
qlib-models = ["pyqlib"]
research = ["pyarrow", "scikit-learn", "lightgbm"]
```

组件加载时，如果缺少 optional extra，应给出明确错误：

```text
LightGBMAlphaModel requires QuantX optional dependency group 'ml'.
Install with: pip install -e '.[ml]'
```

不能让未使用 ML 的普通规则回测因为没有安装 LightGBM/PyTorch 而失败。

## 23. 错误处理与安全约束

默认 fail-fast：

- feature schema 不匹配。
- 特征因果分析发现未来读取或无法确定读取窗口。
- artifact checksum 不匹配。
- artifact loader、格式或 trust level 不允许。
- DataVersion manifest 与当前 provider 不一致，或训练期读锁失效。
- fold 日期重叠。
- OOS prediction 重复。
- 股票不属于 point-in-time universe。
- signal time 不早于 execution time。
- development OOS/sealed holdout/live shadow 期间模型发生未声明的 update。
- 未创建 ExperimentLock 就请求 sealed holdout。
- 同一 research generation 重复运行 sealed holdout。
- 执行上下文暴露了当前阶段尚不可用的行情字段。
- PendingRebalance 尚未到 execute_not_before 或已经过期。
- 预测包含 NaN/inf 且策略未配置处理方式。
- 权重和不合法。

可以被配置为降级的情况：

- 个别股票特征不足，按 valid mask 排除。
- 个别日期预测 coverage 低于阈值，选择持有现金。
- 模型服务暂时不可用，生产模式使用最后一份已验证 artifact；回测模式不得静默回退。

## 24. 与现有引擎的迁移方案

### Phase 0：数据可信度前置修复

目标：ML 训练数据不存在明显生存者偏差和时点污染。

工作：

- 建立逻辑 DataVersion、sync manifest 和 provider 训练期读锁。
- 为 promotion/精确复现提供按需物理快照，不要求每次探索复制数据。
- 明确前复权历史修订语义及公司行动版本。
- 建立 point-in-time universe。
- 补齐退市股票或明确研究区间限制。
- 保存 DataVersion manifest 和 universe fingerprint。
- 历史行业/概念未补齐前禁止作为模型特征。

验收：同一 ResearchRunner 持锁期间 provider 不可修改；数据修订后产生新 version ID 且旧 cache 不命中；promotion 可生成可校验的物理快照。股票池自动报告上市/退市、无行情、ST/停牌和指数成分覆盖，所有配置门槛通过。随机抽查只作为补充证据。

### Phase 1：决策协议和兼容层

目标：不改变现有策略结果，引入新类型。

工作：

- 新增 `MarketObservation`、`PortfolioState`、`ExecutionContext`。
- 新增 `AlphaSnapshot`、`RawTargetPortfolio`、`FinalTargetPortfolio`。
- 新增 `RiskOverlay`、`PendingRebalance`、`OrderPlanner` 和待执行队列。
- 新增 component registry。
- 实现 `LegacyCompositeStrategyAdapter`。
- 实现 `LegacyRiskExecutionAdapter`，保留旧策略行为但明确为迁移组件。

验收：弱转强、shuijiao、五福结果与迁移前完全一致；next-open ExecutionContext 读取 high/low/close 必须失败；未到执行时点的 pending rebalance 不得成交。

### Phase 2：冻结预测回测

目标：标准回测可以消费外部样本外预测。

工作：

- 实现 `PredictionStore`。
- 实现 `PrecomputedAlphaModel`。
- 实现 TopK equal-weight PortfolioConstructor、NoOp/Standard RiskOverlay。
- 扩展标准报告引用 artifact。

验收：手工构造预测可以生成 PendingRebalance 并在严格 T+1 开盘事件成交；同日预测成交和执行阶段读取未来字段必须失败。

### Phase 3：研究数据集和 LightGBM

目标：当前 QMT 数据可以完成训练、validation、OOS 预测和标准回测闭环。

工作：

- DatasetBuilder、LabelBuilder、WalkForwardSplitter。
- FeatureCausalityValidator。
- Ridge/LightGBM Trainer。
- IC、development OOS、sealed holdout 和 live shadow 分层报告。
- `run_research` CLI。

验收：完成 development walk-forward 并可重复得到相同结果；ExperimentLock 后 sealed holdout 只能运行一次；模型冻结后可以启动 live shadow。

### Phase 4：深度学习 adapter

目标：不修改回测引擎即可接入 PyTorch 模型。

工作：

- RawWindowFeatureBuilder。
- TorchTrainer 和 checkpoint manifest。
- batch inference adapter。

验收：一个最小 LSTM/MLP 与 LightGBM 使用完全相同的 fold 和回测口径。

### Phase 5：RL 环境

目标：训练环境复用真实交易语义。

工作：

- 从引擎提取 `SimulationKernel.step()`。
- Gymnasium adapter。
- RL allocation policy。
- 多 seed validation。

验收：固定 action 的 RL policy 与等权 PortfolioConstructor 产生一致 raw target 和净值；RiskOverlay 仍统一生效；sealed holdout 环境冻结参数和 replay buffer。

### Phase 6：在线学习和生产

目标：支持按计划更新模型，不破坏回测可复现性。

工作：

- Model promotion registry。
- champion/challenger。
- 数据漂移监控。
- 定期重训和回滚。

该阶段不属于第一轮实现。

## 25. 测试计划

### 25.1 单元测试

- DataVersion hash、sync manifest、训练期读写锁和版本变更校验。
- promotion 物理快照完整性校验。
- Point-in-time universe 覆盖率门槛。
- FeatureSchema hash 稳定。
- FeatureCausalityValidator 拒绝 `Ref(x, -1)` 和未声明读取窗口的自定义特征。
- Label 的 T+1 open/horizon 计算准确。
- 理论标签、execution status 和实际成交结果不混用。
- Purge 删除跨边界标签。
- train-only transformer 不读取 validation。
- PredictionStore 拒绝重复和区间外预测。
- AlphaSnapshot 时间校验。
- Raw/FinalTargetPortfolio 权重约束。
- RiskOverlay 把止损表达为目标权重变化。
- PendingRebalance 时间和过期校验。
- OPEN ExecutionContext 不包含 high/low/close。
- artifact loader allowlist、checksum 和 trust level。
- registry 缺少依赖时错误清晰。

### 25.2 集成测试

- QlibBinReader -> DatasetBuilder -> LightGBM -> predictions。
- predictions -> DecisionPipeline -> PendingQueue -> ExecutionPipeline -> BacktestEngine。
- Formula adapter 与旧 CompositeStrategy 一致。
- 同一预测使用 Top10/Top20 时无需重新训练。
- 同一 raw target 经过不同 RiskOverlay 时无需重新预测。
- 模型 sealed holdout/live shadow 期间不能 update。
- sealed holdout 在同一 ExperimentLock 下不能重复运行。

### 25.3 泄漏测试

1. 修改未来行情，历史日期特征和预测必须不变。
2. 修改 development OOS/holdout 标签，训练模型 checksum 必须不变。
3. 把 signal_time 改成 execution_time，回测必须失败。
4. 使用当前行业快照回填历史，dataset validation 必须拒绝。
5. 人为制造跨 fold label，splitter 必须 purge。
6. 公式使用负数 Ref 或 centered future window，causality validator 必须拒绝。
7. 在 OPEN execution event 注入当日 close，字段白名单必须拒绝。
8. 根据 T+1 是否涨停预先删除 T 日样本，dataset audit 必须识别 eligibility 使用未来字段。
9. 更新前复权历史后必须生成新 DataVersion，且不能命中旧 feature cache；已创建的 promotion 物理快照仍可独立校验。
10. 保持 Git commit 不变但修改特征代码，code fingerprint 和 artifact ID 必须变化。

### 25.4 Golden tests

保留固定小型市场数据：

```text
3 symbols x 40 sessions
```

用于验证特征、标签、fold、预测、权重、订单和净值的完整链路，避免大型真实数据让测试难以定位。

## 26. Promotion Gate

技术接入成功不等于策略有效。模型进入项目策略库前必须满足：

1. 全部结果是严格 OOS，并明确 evaluation tier。
2. 多数 development OOS fold 为正，不依赖单一年份。
3. Rank IC、收益和换手在相邻 horizon/TopK 下稳定。
4. 多 seed 结果稳定。
5. 成本后收益为正。
6. 收益不集中在少数股票、行业或一段行情。
7. 比简单线性、LightGBM、动量和等权基线有明确增益。
8. 严格时序和 point-in-time universe 检查通过。
9. 模型、预测、配置和 run artifacts 可完整复现。
10. 未通过前，模型和 run 只保存在 `/tmp`。
11. 方案生成不可变 ExperimentLock 后才允许运行 sealed holdout。
12. 历史 holdout 通过后仍需 live shadow；项目已观察过的历史年份不得包装成完全未见 test。
13. DataVersion、promotion 物理快照、复权版本、artifact format/checksum/trust level 全部可验证。

## 27. 关键设计取舍

### 27.1 不使用万能 ModelBasedPolicy

拒绝原因：

- 监督学习输出 score。
- 组合优化输出 weight。
- 执行模型输出 order intent。
- RL 可能位于任一阶段。

按输出契约分层比按算法名称分层更稳定。

### 27.2 不默认在回测中训练

拒绝原因：

- 难以审计训练截止时间。
- 重复训练导致回测慢且不可复现。
- 无法复用预测比较组合策略。
- 容易把已经查看过的 OOS/holdout 反馈用于继续调参。

### 27.3 不直接使用 Qlib 回测作为正式结果

Qlib 适合作为模型和数据集 adapter，但 QuantX 已有统一的 A 股交易语义。正式结果回到 QuantX，避免成本、涨跌停、T+1 和成交口径分裂。

### 27.4 不让模型直接访问 BacktestContext

AlphaModel 只读取 `MarketObservation`，组合与风险层另行读取 `PortfolioState`，执行层只读取当前阶段允许字段的 `ExecutionContext`。这是防止未来数据泄漏、保证预测可缓存，也是后续支持实盘推理的基础。

### 27.5 不让 OrderPlanner 决定风险目标

止损、止盈、时间退出、组合回撤和暴露约束属于 RiskOverlay。OrderPlanner 只负责把 FinalTargetPortfolio 转成可提交订单。现有 RuleExecution 通过兼容 adapter 保持结果，但不能作为新模型架构的长期边界。

### 27.6 不把可变 provider 路径当成数据版本

QMT 前复权和增量同步可能重写历史。变化的数据可以用于训练和 validation，但 artifact 必须引用逻辑 DataVersion，单次运行必须持有 provider 读锁；仅记录 `provider_uri` 不足以比较实验。物理不可变快照只在 promotion、sealed holdout 或明确要求精确复现时创建。

## 28. 第一轮最小实现切片

第一轮不实现完整 DL/RL，只实现能够证明架构正确的垂直闭环：

```text
1. Logical DataVersion + sync manifest + provider read lock；promotion 时按需创建物理快照
2. PointInTimeUniverseProvider + 自动覆盖率审计
3. FeatureSpec + FeatureCausalityValidator
4. LabelSpec + 理论/执行结果分离
5. DatasetBuilder + AnchoredWalkForwardSplitter + purge
6. Ridge 和 LightGBM Trainer
7. 安全 ModelArtifact + PredictionStore
8. PrecomputedAlphaModel
9. TopKEqualWeightConstructor + StandardRiskOverlay
10. PendingRebalanceQueue + StandardOrderPlanner
11. LegacyComposite/RuleExecution adapters
12. 标准 BacktestEngine 双事件调度
13. development OOS + sealed holdout + live shadow 报告
```

第一个实验建议：

```text
Universe: 有历史成员记录的 point-in-time 指数池；主板池待退市覆盖通过后启用
Features: 日线价量和横截面特征
Labels: 5/10/20 日 next-open excess return
Models: Ridge, LightGBM
Portfolio: Top20/Top50, 单票上限 5%
Validation: development walk-forward + ExperimentLock + sealed holdout + live shadow
Execution: T+1 open, 标准成本
```

完成该切片后：

- 新增 Transformer 只需要 FeatureBuilder + Trainer adapter。
- 新增 Qlib DoubleEnsemble 只需要 Qlib adapter。
- 新增 AlphaGen 只需要候选因子导入器。
- 新增 RL allocation 只需要 Env + PortfolioConstructor adapter，仍复用 RiskOverlay。
- 回测引擎不再为模型类型修改。

## 29. 完成定义

本架构不能以“新增了几个抽象类”为完成标准。第一阶段完成必须同时证明：

1. 现有规则策略没有非预期变化；如旧结果依赖未来字段或不一致成交语义，必须记录差异原因并锁定修正后的严格基线，不能为了对齐旧收益恢复泄漏行为。
2. 当前 QMT/Qlib 数据成功构建训练和 validation 数据。
3. DataVersion、训练期读锁和 point-in-time universe 通过自动完整性门槛；promotion 物理快照可验证。
4. 特征因果分析、walk-forward 和 purge 有自动化测试。
5. LightGBM 生成严格 development OOS 预测。
6. 标准回测通过 PendingRebalance 强制决策/执行双时钟。
7. 同一预测可复用至少两个 PortfolioConstructor 和 RiskOverlay。
8. ExperimentLock、sealed holdout 和 live shadow 生命周期可执行。
9. 模型 manifest、安全加载、fold 指标和交易报告可以追溯。
10. 不安装 ML/DL/RL extras 时，现有规则回测仍正常运行。

满足以上条件后，QuantX 才具备“容易接入各种技术，同时统一训练验证和回测”的基础能力。

## 30. 当前实现与验证状态

截至 2026-07-12，`qmt-mac` 分支已经实现本文 Phase 0-6 的基础设施：

- QMT/Qlib 单一数据层、逻辑 DataVersion、provider 读写锁、同步 manifest 和 promotion 物理快照。
- PIT 股票池、行情覆盖审计，以及 ST、停牌、历史指数成分数据来源声明门禁。
- 因果特征、next-open/future-open 标签、benchmark 超额收益、train-only winsorize、purge/embargo walk-forward。
- Ridge、LightGBM、Torch MLP/LSTM、RL allocation environment，以及安全 artifact loader。
- fold PredictionContract、严格 OOS PredictionStore、逐日 IC/RankIC、ICIR、覆盖率、分位收益、fold/年度和多 seed 聚合。
- PredictionStore 到统一 DecisionPipeline、PendingRebalance、T+1 OPEN 和标准 BacktestEngine 的闭环。
- ExperimentLock、一次性 sealed holdout ledger、live shadow、champion/challenger、回滚、漂移监控和 Promotion Gate。

真实 QMT smoke 结果只证明技术闭环，不代表策略有效：

| 模型 | OOS 预测数 | RankIC | 标准回测收益 |
|---|---:|---:|---:|
| Ridge | 4,850 | 0.008805 | -31.48% |
| LightGBM | 4,850 | 0.055638 | -25.38% |
| Torch MLP | 4,850 | -0.016379 | -43.90% |

这些临时实验、配置和 runs 均保留在 `/tmp/quantx-research`，未进入项目策略库。当前 Qlib 固定库没有独立的历史 ST/停牌字段，因此可以继续 exploratory 训练和 validation，但在配置可审计的 PIT 状态来源之前，`formal` 数据门禁应失败，模型不得 promotion。

规则策略回归中，五福严格对齐旧基线 `1052.09%`。弱转强选股序列一致，但旧基线使用执行日 close 因子，并允许依赖卖出失败后继续买入；严格引擎改为上一交易日因子并以 `sell_dependency_failed` 拒绝依赖买单，收益由旧口径 `3845.73%` 变为严格口径 `2886.81%`。该差异属于时序和成交语义修正，不能恢复旧行为来追求数字一致。
