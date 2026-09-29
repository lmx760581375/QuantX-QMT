# Config Driven Strategy Design

本文档定义 QuantX 下一阶段策略层重构方案：把策略从“专门 Python class”重构为“由 config 完整描述的策略语言”。

核心目标：

```text
Python 只提供通用原子能力。
Config 完整描述策略逻辑。
Runtime 根据 config 生成选股、调仓、执行和回测结果。
```

这意味着 `shuijiao` 不应该长期依赖 `quantx/strategies/shuijiao_strategy.py` 这种专门策略代码，也不应该在 config 中依赖 `ShuijiaoBuy(...)`、`ShuijiaoTrend(...)` 这类策略专属黑盒函数。

`shuijiao` 应该只是一个示例 config。

## 0. 当前实现状态

本轮已经先落地 Phase 0，目标是把策略语言和回测正确性的地基打稳：

1. `FactorRuntime` 默认 fail-fast，并在执行公式时重新校验 AST。
2. 公式语言拒绝 `&`、`|`、`~`，推荐 `and/or/not` 或 `And/Or/Not`。
3. `shuijiao` 当前主路径已经展开为通用公式 DAG，不再通过 `ShuijiaoBuy(...)`、`ShuijiaoTrend(...)` 生成信号。
4. `BacktestResult` 会记录 `signal_errors`；默认 `error_policy=fail_fast`。
5. `run_shuijiao_qlib_diagnostic.py` 暴露关键兼容开关，方便区分“严格旧源码口径”和“旧 outputs artifact 口径”。

已继续完成：

1. 最小可用 YAML/config runner：`python -m quantx.tools.run_backtest --config ...`。
2. 通用 `FormulaSelector + EqualWeightRebalance + RuleExecution`。
3. `configs/strategies/shuijiao_legacy.yaml` 已用完整公式展开 `buy_signal/sell_signal`。
4. `shuijiao_legacy.yaml` 已复现当前 qlib 严格旧源码口径：`total_return=68.98%`。
5. `myquant_shuijiao_best_auto` 已用 YAML 复现为通用公式策略，并完成 exact-universe/no-lookback 对齐审计：QuantX 当前 `total_return=205.06%`，myquant 参考 `186.47%`，剩余差异优先归因为数据/复权/冷启动/旧 artifact 口径。
6. 回测引擎已增加 `precompute_signals` 安全边界：YAML 公式策略可声明预计算，未声明的 Python 策略默认逐日计算，避免状态依赖选股被 Phase 1 并行预计算污染。

仍未完成：

1. 策略专属 operator 还留在 `DEFAULT_REGISTRY` 中，后续应迁到 extension registry。
2. `fields/factors/signals/selector/rebalance/execution` 的完整 Pydantic spec validation 还未实现。
3. Macro 展开、explain 报告和输出落盘还未实现。
4. 交易规则语言第一版已可执行标量规则，但还没有完整变量类型表和 stateful/precomputed 编译报告。
5. exact-universe 目前仍使用临时 YAML；长期应支持 `universe_file` 或 committed universe snapshot，避免把几千个 symbol 手写进策略文件。

### 0.1 本轮审计后新增的正确性边界

两阶段引擎的 Phase 1 信号预计算能带来很大速度收益，但它只能用于纯行情/纯因子 selector。原因是 Phase 1 会在交易执行前计算所有日期的 selection，此时账户、持仓、累计收益、回撤、动态风控状态都还不是对应日期的真实状态。

因此策略必须按阶段拆分变量：

```text
precomputed selector:
  可以引用 market/factor/cross_section 变量。
  不可以引用 cash/positions/pnl_pct/holding_days/drawdown 等账户状态。

sequential rebalance/execution:
  可以引用当日真实 account/position 状态。
  必须逐日顺序执行。
```

当前实现约束：

1. YAML `FormulaSelector` 会检查 `selector.where` 和 `selector.score`，禁止引用账户/持仓状态变量。
2. YAML 公式策略由 `CompositeStrategy(precompute_stock_signals=True)` 明确声明可以预计算。
3. 普通 Python 策略默认 `precompute_stock_signals=False`，除非策略作者显式声明安全。
4. `BacktestConfig.precompute_signals` 可作为引擎级 override，但只应用于明确审计过的策略。

这条边界必须写进 explain/report。后续如果要支持账户状态参与选股，应该新增 `selector.mode: sequential`，而不是偷偷在 precomputed 模式下允许状态变量。

## 1. 当前问题

目前 QuantX 的状态：

```text
数据层: Qlib + AStockExchange + BacktestContext
因子层: MarketPanel + FactorRuntime + OperatorRegistry
策略层: Python class, 例如 ShuijiaoSelector / ShuijiaoRebalance / ShuijiaoExecution
运行入口: 针对 shuijiao 的专门 Python script
```

这个状态有两个优点：

1. 数据层和策略层已经基本解耦。
2. 因子计算已经可以在全市场矩阵上快速运行。

但也有明显问题：

1. 每个策略都需要专门写 Python 文件。
2. 选股、调仓、买卖逻辑散落在 Python class 中，不利于生成、审计和复现。
3. `run_shuijiao_qlib_diagnostic.py` 这种专门脚本不应该成为长期正式入口。
4. 如果 config 只写 `buy_signal: ShuijiaoBuy(...)`，本质上只是把策略黑盒函数藏到了 operator 里，没有真正做到策略配置化。

最终目标应该是：

```bash
python -m quantx.tools.run_backtest --config configs/strategies/shuijiao.yaml
python -m quantx.tools.run_backtest --config configs/strategies/my_new_alpha.yaml
```

## 2. 核心原则

### 2.1 Config 必须完整定义信号生成逻辑

任何被 selector、rebalance、execution 引用的变量，都必须满足下面之一：

1. 是系统内置行情字段，例如 `open`、`high`、`low`、`close`、`volume`、`vwap`、`change`。
2. 是系统内置账户/持仓状态变量，例如 `cash`、`total_value`、`cost_price`、`holding_days`、`pnl_pct`、`position_qty`。
3. 在 config 的 `factors` 或 `signals` 中用公式显式定义。

也就是说，下面这种 config 不完整：

```yaml
selector:
  where: buy_signal
```

因为它没有说明 `buy_signal` 怎么生成。

正确方式是：

```yaml
signals:
  buy_signal: Cross(ma20, ma60) and (close > ma60)

selector:
  where: buy_signal
```

### 2.2 不依赖策略专属黑盒 operator

下面这些 operator 不适合作为长期正式策略语言的核心依赖：

```text
ShuijiaoMidline
ShuijiaoTrend
ShuijiaoBuy
ShuijiaoReadySell
```

它们的问题是：

1. 策略逻辑隐藏在 Python 代码里。
2. config 不能独立解释完整策略。
3. 自动生成 config 时，必须提前知道这些 Python 函数存在。
4. 其他策略无法自然复用。

代码层应该尽量只保留通用原子算子：

```text
Ref
Min
Max
Mean
EMA
SMA_TDX
Cross
Filter
Maximum
Minimum
Where
Abs
CSMean
CSCount
CSRank
CSPctRank
TopK
```

策略专属公式可以作为示例、模板或 macro 存在，但正式运行前必须能展开为完整公式 DAG。

### 2.3 Config 是策略资产

一份策略 config 应该足以回答：

```text
用什么数据？
用什么股票池？
计算哪些因子？
买入信号怎么生成？
卖出信号怎么生成？
如何筛选股票？
如何调仓？
如何生成订单？
交易成本和交易规则是什么？
输出哪些结果？
```

如果必须打开某个策略 Python 文件才能知道买卖逻辑，这个策略就还没有完成配置化。

## 3. 目标 Config 结构

建议标准策略 config 包含以下部分：

```yaml
name: shuijiao_formula
version: 1

data:
  provider_uri: data/qlib_data
  universe: all_a
  start: 2021-01-04
  end: 2025-10-17
  look_back_days: 100

fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume
  vwap: $vwap
  change: $change

factors:
  ma20: Mean(close, 20)
  ma60: Mean(close, 60)

signals:
  buy_signal: Cross(ma20, ma60) and (close > ma60)
  sell_signal: Cross(ma60, ma20)

selector:
  where: buy_signal
  score: 0
  sort: input_order
  topk: null

rebalance:
  type: equal_weight
  max_positions: 10
  cash_use_ratio: 0.99
  buy_only_new_positions: true

execution:
  deal_price: close
  sell_rules:
    - name: signal_sell
      when: sell_signal
      action: sell_all
    - name: take_profit
      when: close / cost_price - 1 > 0.20
      action: sell_all
    - name: stop_loss
      when: close / cost_price - 1 < -0.10
      action: sell_all
  buy:
    sizing: cash_equal
    lot_size: 100
    skip_if_holding: true

cost:
  commission_rate: 0.0005
  min_commission: 5.0
  stamp_tax_rate: 0.0001
  stamp_tax_on_buy: true
  transfer_fee_rate: 0.0
  slippage: 0.0

engine:
  validate_trading_rules: false
  deal_price: close
  max_workers: 1

output:
  save_trades: true
  save_daily_nav: true
  format: json
```

## 4. Shuijiao 公式展开示例

`shuijiao` 不应该长期写成：

```yaml
signals:
  buy_signal: ShuijiaoBuy(trend_line, close, midline)
```

而应该把信号公式完整展开。

示例：

```yaml
name: shuijiao_formula

data:
  provider_uri: data/qlib_data
  universe: all_a
  start: 2021-01-04
  end: 2025-10-17
  look_back_days: 100

fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume
  vwap: $vwap
  change: $change

factors:
  h1: Maximum(high, Ref(close, 1))
  l1: Minimum(low, Ref(close, 1))
  p1: h1 - l1
  resistance: l1 + p1 * 7 / 8
  support: l1 + p1 * 0.5 / 8
  midline: (support + resistance) / 2

  llv_low: Min(low, 55)
  hhv_high: Max(high, 55)
  rsv: (close - llv_low) / (hhv_high - llv_low + 1e-10) * 100
  sma1: SMA_TDX(rsv, 5, 1)
  sma2: SMA_TDX(sma1, 3, 1)
  v11: 3 * sma1 - 2 * sma2
  trend_line: EMA(v11, 3)

  market_health: CSMean(trend_line > 20)

signals:
  trend_prev: Ref(trend_line, 1)
  bb1: (trend_prev < 11) and (trend_prev > 6) and Cross(trend_line, 11)
  bb2: (trend_prev < 6) and (trend_prev > 3) and Cross(trend_line, 6)
  bb3: (trend_prev < 3) and (trend_prev > 1) and Cross(trend_line, 3)
  bb4: (trend_prev < 1) and (trend_prev > 0) and Cross(trend_line, 1)
  bb5: (trend_prev < 0) and Cross(trend_line, 0)

  buy_signal: (bb1 or bb2 or bb3 or bb4 or bb5) and (close < midline)
  sell_signal: (trend_line > 89) and Filter(trend_line > 89, 15) and (close > midline)

selector:
  where: buy_signal
  score: 0
  sort: input_order

rebalance:
  type: equal_weight
  max_positions: 10
  cash_use_ratio: 0.99

execution:
  deal_price: close
  sell_rules:
    - name: signal_sell
      when: sell_signal
      action: sell_all
    - name: take_profit
      when: close / cost_price - 1 > 0.20
      action: sell_all
    - name: stop_loss
      when: close / cost_price - 1 < -0.10
      action: sell_all
```

这份 config 的关键价值是：

```text
不需要 ShuijiaoBuy 这种策略专属 Python operator。
buy_signal 和 sell_signal 的来源完全可见。
FactorRuntime 可以按 DAG 自动排序并计算。
Explain 工具可以展示完整信号链路。
```

## 5. 语言层设计

### 5.1 因子公式语言

因子公式语言运行在全市场矩阵上，输入输出通常是 `[T, N]` 或 `[T]`。

适用范围：

```text
fields
factors
signals 中不依赖账户状态的部分
```

示例：

```yaml
factors:
  ma20: Mean(close, 20)
  trend_strength: close / Mean(close, 60) - 1
  market_health: CSMean(trend_strength > 0)
```

### 5.2 交易规则语言

交易规则语言运行在每日交易阶段，除了行情和因子，还能访问账户/持仓状态。

适用范围：

```text
execution.sell_rules
rebalance.dynamic_rules
risk controls
```

内置状态变量建议包括：

```text
cost_price
holding_days
position_qty
position_weight
pnl_pct
cash
total_value
peak_price
drawdown
```

示例：

```yaml
execution:
  sell_rules:
    - name: take_profit
      when: pnl_pct > 0.20
      action: sell_all
    - name: stop_loss
      when: pnl_pct < -0.10
      action: sell_all
```

### 5.3 统一 AST 安全执行

因子公式和交易规则都应走安全 AST evaluator，而不是 Python `eval`。

禁止语法：

```text
Attribute
Subscript
Lambda
Comprehension
Import
Function definition
Arbitrary method call
```

允许语法：

```text
Name
Constant
UnaryOp
BinOp
BoolOp
Compare
Call to registered operator
```

这样既能支持类 qlib 公式，也能避免 config 执行任意 Python 代码。

### 5.4 布尔语法约束

第一版公式语言推荐只使用：

```text
and
or
not
```

不推荐在 config 中使用：

```text
&
|
~
```

原因是 Python AST 中 `&` / `|` 的优先级高于比较运算，用户写出下面这种公式时，语义很容易和预期不一致：

```yaml
where: trend_line > 20 & close > midline
```

它不是：

```text
(trend_line > 20) and (close > midline)
```

而会按 Python 表达式优先级变成接近：

```text
trend_line > (20 & close) > midline
```

因此第一版实现必须做以下限制：

```text
1. Config 示例全部使用 and/or/not。
2. Validator 默认拒绝 ast.BitAnd / ast.BitOr / ast.Invert。
3. 如果未来要支持 &/|/~，必须只允许每个比较表达式已经显式加括号的场景，并提供清晰错误提示。
4. 更稳妥的替代方案是提供 And(a, b, ...)、Or(a, b, ...)、Not(x) 三个注册算子。
```

推荐写法：

```yaml
signals:
  buy_signal: (bb1 or bb2 or bb3) and (close < midline)
```

或：

```yaml
signals:
  buy_signal: And(Or(bb1, bb2, bb3), close < midline)
```

### 5.5 公式安全执行的实现要求

安全 AST evaluator 不能只在依赖分析阶段校验语法。实现时必须满足：

```text
1. _eval_formula() 内部必须再次调用 _validate_ast(tree)。
2. 任何不支持的 AST node 必须抛 FormulaError，不能漏出 KeyError / TypeError。
3. 函数调用只能调用 OperatorRegistry 中注册过的 operator。
4. OperatorRegistry 应区分 core operator 和 strategy extension operator。
5. Config runner 默认只启用 core operator。
```

这样可以避免后续新增 debug eval、rule eval、macro eval 时绕过安全校验。

### 5.6 变量 shape/type 合约

公式语言必须明确每个变量的形状和可用阶段，否则 config 策略会在广播、对齐、状态变量访问上产生隐形 bug。

建议第一版定义以下类型：

```text
matrix[T, N]
  全市场日频矩阵，例如 close/high/low/trend_line/buy_signal。

series[T]
  市场级时间序列，例如 market_health、portfolio_drawdown_index。

cross_section[N]
  某一交易日的股票横截面，通常是 matrix[T, N] 在 date 上切片。

scalar
  标量参数或账户级值，例如 cash、total_value。

position_state[H]
  当前持仓集合上的状态，例如 cost_price、holding_days、pnl_pct，其中 H 是当前持仓数。
```

可用阶段：

```text
fields/factors/signals:
  只能使用 matrix[T,N]、series[T]、scalar parameter。
  禁止使用 position_state 和账户状态。

selector.where / selector.score:
  默认只能使用 matrix/series 的当日横截面。
  如果需要账户状态，必须显式声明 selector.mode: stateful，并关闭 Phase 1 并行预计算。

rebalance.dynamic_rules:
  可使用 series[T]、account scalar、portfolio state。

execution.sell_rules:
  可使用当日行情、当日信号、position_state、account scalar。
```

这条规则是配置化策略正确性的核心边界。

### 5.7 预计算与有状态规则的边界

QuantX 当前引擎有 Phase 1 并行预计算选股信号。这个设计很快，但只适用于纯市场数据公式。

必须硬性区分：

```text
Precomputable:
  fields
  factors
  signals
  selector.where/score 中只依赖行情、因子和市场级 series 的部分

Stateful:
  引用 cash、total_value、position_qty、cost_price、holding_days、pnl_pct、drawdown 的规则
```

实现要求：

```text
1. StrategySpec validation 必须扫描每个表达式的变量依赖。
2. 如果 Phase 1 并行预计算开启，selector.where/score 禁止引用账户和持仓状态变量。
3. 如果 selector 需要状态变量，必须走顺序模式：selector.mode: stateful。
4. execution.sell_rules 默认顺序执行，不能预计算成静态信号，除非它只依赖 market/factor。
5. Explain 报告必须标记每个表达式是 precomputed 还是 stateful。
```

示例：

```yaml
selector:
  mode: precomputed
  where: buy_signal and (market_health > 0.22)
```

下面这个必须拒绝，除非显式改为 `stateful`：

```yaml
selector:
  mode: precomputed
  where: buy_signal and (cash > 100000)
```

### 5.8 错误处理必须 fail-fast

配置化策略的默认错误处理必须是 fail-fast。

原因：公式拼错、变量未定义、某一天计算异常，如果只是记日志并继续，会让错误策略变成“空信号回测”，收益结果看似正常但没有意义。

实现要求：

```text
1. 默认任何公式错误、信号生成错误、规则执行错误都应终止回测。
2. 如果研究场景需要跳过错误，必须显式配置 continue_on_error: true。
3. continue_on_error 必须在输出报告中醒目标记，并记录跳过日期、表达式和错误信息。
```

建议配置：

```yaml
engine:
  error_policy: fail_fast
```

或：

```yaml
engine:
  error_policy: continue
  max_error_days: 3
```

### 5.9 策略专属 operator 的隔离策略

虽然长期目标是不依赖 `ShuijiaoBuy` 这类黑盒 operator，但迁移期可能仍需要保留它们用于对齐或兼容。

隔离方案：

```text
1. DEFAULT_REGISTRY 只注册 core operator。
2. 策略专属 operator 放到独立 extension module，例如 quantx.strategies.extensions.shuijiao。
3. Config 必须显式声明 extensions 才能使用这些 operator。
4. run_backtest --explain 必须标出哪些公式用了 extension operator。
5. 新策略默认不得依赖 extension operator，除非是临时迁移 config。
```

示例：

```yaml
extensions:
  - quantx.strategies.extensions.shuijiao_legacy
```

这能避免策略黑盒混入通用语言核心。

### 5.10 字段别名与数据加载字段的边界

`fields` 不应该被当成普通公式表达式处理，因为 `$open` 不是合法 Python identifier。

第一版建议：

```yaml
fields:
  open: $open
  high: $high
  low: $low
  close: $close
```

语义是：

```text
从 Qlib 加载 $open 字段，并在 runtime 中注册为 open。
```

实现要求：

```text
1. fields 只允许 Qlib 字段名或已有 input 字段名，不走公式 AST。
2. factors/signals/rules 只能引用别名后的 open/high/low/close。
3. 如果 fields 中定义了别名冲突，例如 close 覆盖已有 factor 名，必须报错。
4. 数据加载阶段根据 fields 自动决定 D.features 需要读取哪些字段。
```

### 5.11 Pure preview 与执行副作用隔离

调仓阶段可能需要知道“今天计划卖出哪些股票”，以便释放仓位名额。但 preview 不能改变执行状态。

实现要求：

```text
1. SellRuleEngine 必须提供纯函数 preview_sell_symbols(state)，不得更新 peak_price 等状态。
2. RuleExecution.act() 才允许提交状态变更。
3. trailing stop、peak price 这类状态必须有明确 update 时机。
4. 单元测试必须覆盖 preview 调用前后 execution state 不变。
```

否则会出现预览阶段提前更新移动止盈状态，导致真实执行逻辑偏移。

## 6. Macro 机制

可以支持 macro，但 macro 必须可展开，不能成为不可审计黑盒。

示例：

```yaml
macros:
  tdx_trend:
    params: [close, high, low, period]
    formulas:
      llv_low: Min(low, period)
      hhv_high: Max(high, period)
      rsv: (close - llv_low) / (hhv_high - llv_low + 1e-10) * 100
      sma1: SMA_TDX(rsv, 5, 1)
      sma2: SMA_TDX(sma1, 3, 1)
      output: EMA(3 * sma1 - 2 * sma2, 3)

factors:
  trend_line: macro.tdx_trend(close, high, low, 55)
```

运行前必须展开成：

```yaml
factors:
  trend_line__llv_low: Min(low, 55)
  trend_line__hhv_high: Max(high, 55)
  trend_line__rsv: (close - trend_line__llv_low) / (trend_line__hhv_high - trend_line__llv_low + 1e-10) * 100
  trend_line__sma1: SMA_TDX(trend_line__rsv, 5, 1)
  trend_line__sma2: SMA_TDX(trend_line__sma1, 3, 1)
  trend_line: EMA(3 * trend_line__sma1 - 2 * trend_line__sma2, 3)
```

报告中必须能看到 macro 展开后的完整公式链。

## 7. 重构后的代码结构

建议新增模块：

```text
quantx/core/spec/
  strategy_spec.py
  data_spec.py
  field_spec.py
  formula_spec.py
  selector_spec.py
  rebalance_spec.py
  execution_spec.py
  cost_spec.py
  output_spec.py

quantx/core/strategy/config/
  config_strategy.py
  formula_selector.py
  config_rebalancer.py
  rule_execution.py
  rule_context.py
  strategy_factory.py

quantx/core/runtime/
  expression.py
  macro_expander.py
  state_variables.py

quantx/tools/
  run_backtest.py

configs/strategies/
  shuijiao_formula.yaml
  examples/
```

已有的 `quantx/core/factor_runtime/` 保留，并继续作为矩阵公式计算后端。

## 8. Runtime 流程

目标流程：

```text
1. Load YAML config
2. Validate with StrategySpec
3. Expand macros
4. Build BacktestConfig
5. Load Qlib data through BacktestContext
6. Build MarketPanel
7. FactorRuntime computes fields + factors + signals
8. FormulaSelector generates daily selections
9. ConfigRebalancer generates target allocations
10. RuleExecution generates orders
11. Executor validates/fills orders
12. Account updates daily NAV
13. Reporter writes trades/nav/metrics/explain
```

核心对象关系：

```text
StrategySpec
  -> StrategyFactory
    -> ConfigDrivenStrategy
      -> FormulaSelector
      -> ConfigRebalancer
      -> RuleExecution
```

### 8.1 高效实现管线

为了保持现在全市场回测十几秒到几十秒级别的性能，config runner 不能把所有规则都逐股票逐日 Python 循环执行。

建议管线拆成三段：

```text
Stage A: Static compile
  - 读取 YAML
  - Pydantic 校验 schema
  - 展开 macro
  - 解析所有表达式 AST
  - 生成依赖 DAG
  - 标记每个表达式的 variable class 和 execution stage

Stage B: Matrix precompute
  - 加载 Qlib 字段
  - 构建 MarketPanel
  - 计算 fields/factors/signals
  - 计算 selector 中 precomputed 的 where/score
  - 输出 daily candidate matrix 或 daily selection cache

Stage C: Stateful daily loop
  - 每日读取 precomputed selection
  - 根据当前 account/positions 构造 RuleContext
  - 纯函数 preview 当日 sell symbols
  - Rebalance 生成目标买入集合
  - Execution 生成 orders
  - Executor 成交
  - Account 更新 NAV 和 position state
```

这样可以保留高速矩阵化因子计算，同时把账户相关逻辑限定在必要的每日循环里。

### 8.2 Compile 阶段必须产出依赖元数据

`StrategySpec` 编译后不应该只是一个 dict，而应生成可审计的 `CompiledStrategy`。

建议结构：

```text
CompiledStrategy
  fields: Dict[str, FieldBinding]
  formulas: Dict[str, CompiledFormula]
  selector: CompiledSelector
  rebalance: CompiledRebalance
  execution: CompiledExecution
  dependency_graph: FormulaDAG
  variable_table: Dict[str, VariableInfo]
```

`VariableInfo` 至少包含：

```text
name
kind: field / factor / signal / account_state / position_state / parameter
shape: matrix / series / cross_section / scalar / position_state
stage: precompute / daily_stateful
source: config path
```

编译阶段必须提前发现：

```text
未定义变量
循环依赖
字段别名冲突
使用了未启用 extension operator
precomputed 表达式引用了 stateful 变量
不安全 AST 语法
不推荐的 BitAnd/BitOr/Invert
```

### 8.3 Selection cache 的形态

为了避免 Phase 2 每日重复计算 selector，`FormulaSelector` 可以在 Stage B 产出每日候选缓存。

建议形态：

```text
SelectionCache
  dates: List[str]
  selections: Dict[str, List[Signal]]
  debug_columns: optional factor snapshots
```

如果 selector 是 `mode: precomputed`，Phase 2 直接读取缓存。

如果 selector 是 `mode: stateful`，Phase 2 每日调用 selector，并且该模式默认不能并行。

### 8.4 Stateful RuleContext

交易规则不能直接访问 `Account` 和 `Position` 对象，应该通过只读 `RuleContext` 暴露安全变量。

建议：

```text
RuleContext(date, symbol)
  market:
    close, open, high, low, vwap, change
  factors:
    buy_signal, sell_signal, score, ...
  position:
    quantity, cost_price, holding_days, pnl_pct, peak_price
  account:
    cash, total_value, drawdown
```

所有 sell rule 表达式都在这个上下文中求值。

好处：

```text
1. 避免规则直接修改账户状态。
2. 方便 explain 单只股票为什么卖出。
3. 方便后续把 rule evaluator 做成向量化或批量评估。
```

## 9. Selector 原子化设计

Selector 不应该关心具体策略名字，只根据 config 和 runtime 工作。

建议第一版支持：

```yaml
selector:
  where: buy_signal and (market_health > 0.22)
  score: trend_line + ema_diff * 20
  sort: score_desc
  topk: 30
  exclude_current_positions: false
```

字段含义：

```text
where: 布尔过滤公式
score: 排序分数公式
sort: input_order / score_desc / score_asc / field_desc / field_asc
topk: 最多输出多少个候选
exclude_current_positions: 是否排除已持仓
```

## 10. Rebalance 原子化设计

第一版支持：

```yaml
rebalance:
  type: equal_weight
  max_positions: 10
  cash_use_ratio: 0.99
  buy_only_new_positions: true
```

动态仓位：

```yaml
rebalance:
  type: equal_weight
  max_positions:
    default: 30
    rules:
      - when: market_health < 0.16
        value: 6
      - when: market_health < 0.22
        value: 12
  cash_use_ratio: 0.99
```

这里的 `when` 走交易规则语言或 `[T]` 市场级因子。

## 11. Execution 原子化设计

Execution 拆成：

```text
SellRuleEngine
BuyOrderBuilder
TradeValidator integration
```

配置示例：

```yaml
execution:
  deal_price: close

  sell_rules:
    - name: signal_sell
      when: sell_signal
      action: sell_all

    - name: take_profit
      when: pnl_pct > 0.20
      action: sell_all

    - name: stop_loss
      when: pnl_pct < -0.10
      action: sell_all

  buy:
    sizing: cash_equal
    lot_size: 100
    skip_if_holding: true
```

第一版买入 sizing 支持：

```text
cash_equal
target_weight
fixed_cash
fixed_shares
score_weight
```

第一版卖出 action 支持：

```text
sell_all
sell_pct
sell_shares
```

## 12. Explain 和可审计性

配置化策略必须提供 explain 输出。

建议命令：

```bash
python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_formula.yaml --explain
```

输出应包括：

```text
1. 使用的数据源和股票池
2. 所有 fields/factors/signals 的 DAG 排序
3. 每个 selector where/score 的展开公式
4. 每个 sell_rule 的公式
5. macro 展开结果
6. 回测参数和成本参数
7. 首 N 个交易日的选股和订单 trace
```

示例：

```text
buy_signal
  = (bb1 or bb2 or bb3 or bb4 or bb5) and (close < midline)
  bb1
    = (trend_prev < 11) and (trend_prev > 6) and Cross(trend_line, 11)
  trend_prev
    = Ref(trend_line, 1)
  trend_line
    = EMA(v11, 3)
```

这对调试自动生成 config 非常重要。

## 13. 迁移计划

### Phase 0: 先补地基，修复当前已知风险

目标：在写 config runner 之前，把公式 runtime 和引擎边界修稳。

必须修复：

```text
1. FactorRuntime._eval_formula() 内部补 _validate_ast(tree)。
2. FormulaRuntime 对不支持的 BinOp/UnaryOp/Compare/Call 统一抛 FormulaError。
3. DSL validator 默认拒绝 BitAnd / BitOr / Invert，或引导使用 And/Or/Not。
4. BacktestEngine Phase 1 错误处理改为 fail-fast，除非显式配置 continue。
5. preview_sell_symbols 必须纯函数化，不能更新 trailing stop / peak_price 状态。
6. 将 strategy extension operator 从 DEFAULT_REGISTRY 中隔离出来。
```

验收：

```text
pytest tests/factor_runtime
新增 DSL safety tests
新增 preview no-side-effect test
新增 engine fail-fast test
shuijiao qlib diagnostic 结果不回退
```

### Phase 1: StrategySpec + Compile

目标：能读取 YAML，并编译成 `CompiledStrategy`，但暂时不要求完全替代现有 Python 策略。

交付：

```text
quantx/core/spec/*
YAML loader
Pydantic schema
Macro placeholder schema
VariableInfo / CompiledFormula
Dependency scanner
Precompute/stateful dependency validator
```

验收：

```text
1. 未定义变量会在启动前报错。
2. 循环依赖会在启动前报错。
3. selector.mode=precomputed 引用 cash 会报错。
4. fields 中 $close 能绑定为 close。
5. 使用未启用 extension operator 会报错。
```

### Phase 2: Config Runner 骨架

目标：提供统一正式入口。

交付：

```text
quantx.tools.run_backtest
--config
--explain
--dry-run
--output-dir
BacktestConfig builder
data provider builder
cost builder
```

验收：

```bash
python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_formula.yaml --dry-run
python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_formula.yaml --explain
```

`--dry-run` 必须只做 config 编译、数据覆盖检查和公式 DAG explain，不跑交易。

### Phase 3: FormulaSelector

目标：移除 `ShuijiaoSelector` 中选股过滤和打分的硬编码。

交付：

```text
FormulaSelector
where/score/sort/topk
SelectionCache
input_order / score_desc / score_asc
selector explain
```

验收：

```text
1. 使用完整展开公式的 shuijiao config 能生成每日 selection。
2. selection 数量和现有 ShuijiaoSelector 在同数据同参数下可对齐。
3. selector 不引用账户状态时可走 Phase 1 并行预计算。
4. selector 引用账户状态时必须显式 stateful，并顺序执行。
```

### Phase 4: ConfigRebalancer

目标：移除 `ShuijiaoRebalance` 专属逻辑。

交付：

```text
ConfigRebalancer
equal_weight
max_positions
cash_use_ratio
dynamic max_positions rules
planned_sell_symbols support
```

验收：

```text
1. max_positions=10 时交易次数与当前 shuijiao diagnostic 对齐。
2. planned_sell_symbols 能释放当日卖出仓位。
3. preview 不修改 execution state。
4. dynamic max_positions 可由 market_health 控制。
```

### Phase 5: RuleExecution

目标：移除 `ShuijiaoExecution` 专属逻辑。

交付：

```text
RuleExecution
SellRuleEngine
BuyOrderBuilder
RuleContext
state variables: pnl_pct / holding_days / cost_price / peak_price
buy sizing: cash_equal / target_weight / fixed_cash / fixed_shares
sell action: sell_all / sell_pct / sell_shares
```

验收：

```text
1. sell_signal / take_profit / stop_loss 可完全由 config 定义。
2. RuleExecution 生成的订单与当前 ShuijiaoExecution 在 legacy 参数下对齐。
3. trailing stop 预览无副作用。
4. 每笔卖出能输出触发规则名。
```

### Phase 6: Macro 和模板

目标：支持公式复用，但保持可展开和可审计。

交付：

```text
macro_expander
macro namespace
macro explain
公式展开快照
```

验收：

```text
1. macro 展开后不依赖黑盒 Python。
2. --explain 能展示展开后的完整 DAG。
3. macro 参数错误会在 compile 阶段报错。
```

### Phase 7: 移除 shuijiao 专门入口

目标：`shuijiao` 从 Python 策略迁移为 config 示例。

交付：

```text
configs/strategies/shuijiao_formula.yaml
configs/strategies/shuijiao_legacy_align.yaml
删除或弃用 run_shuijiao_qlib_diagnostic.py
shuijiao_strategy.py 仅保留兼容 wrapper 或删除
```

验收：

```text
python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_formula.yaml
```

能输出和当前 qlib diagnostic 同级别的收益、交易数、耗时和 explain 报告。

## 14. 最终期望状态

完成后，`shuijiao` 策略不再需要专门的完整 Python 策略文件。

理想状态：

```text
quantx/strategies/shuijiao_strategy.py
  -> 可删除，或仅保留迁移兼容 wrapper

configs/strategies/shuijiao_formula.yaml
  -> 策略真实定义

quantx/core/factor_runtime/operators.py
  -> 只保留通用基础算子

quantx/tools/run_backtest.py
  -> 统一正式运行入口
```

策略系统从：

```text
写 Python 策略 class
```

升级为：

```text
写 config 策略语言
```

这会更适合批量实验、自动生成策略、参数搜索和审计复现。

## 15. 当前代码已知问题与对应修改方向

本节记录从当前代码 review 中发现、会影响配置化落地的问题。

### 15.1 FactorRuntime 安全校验不应只依赖依赖分析

当前风险：

```text
_names_in_expr() 会调用 _validate_ast()
_eval_formula() 当前没有再次调用 _validate_ast()
```

虽然 `compute_formulas()` 常规路径会先 `_plan()`，但后续一旦新增 rule evaluator、debug eval 或单表达式 eval，就可能绕过校验。

修改方向：

```text
1. 在 _eval_formula() 中 parse 后立即 _validate_ast(tree)。
2. 对 _BIN_OPS[type(node.op)] 等直接索引改成显式 if / get，并抛 FormulaError。
3. 增加测试：Attribute/Subscript/ListComp/unknown op 都必须被拒绝。
```

### 15.2 布尔表达式优先级要在 validator 中拦截

当前风险：

```text
FactorRuntime 支持 ast.BitAnd / ast.BitOr
但 config 用户容易写出 a > b & c > d 这种错误语义
```

修改方向：

```text
1. 第一版禁用 BitAnd / BitOr / Invert。
2. 增加 And / Or / Not operator。
3. 所有文档示例使用 and/or/not 或 And/Or/Not。
4. 如果用户使用 &/|/~，报错信息要提示使用 and/or/not。
```

### 15.3 BacktestEngine Phase 1 应默认 fail-fast

当前风险：

```text
Phase 1 signal exception 只 logger.error，然后继续跑。
```

配置化后，公式错误可能被吞掉，变成空信号或缺失日期回测。

修改方向：

```text
1. BacktestConfig 增加 error_policy: fail_fast / continue。
2. fail_fast 下任何 signal exception 立即 raise。
3. continue 下记录 error_dates，并写入 BacktestResult。
4. run_backtest 输出中显示 error_policy 和 error count。
```

### 15.4 preview_sell_symbols 必须无副作用

当前风险：

```text
ShuijiaoExecution.preview_sell_symbols()
  -> _iter_sell_decisions()
    -> trailing stop 分支会更新 _peak_prices
```

这会导致调仓预览阶段改变执行状态。

修改方向：

```text
1. 将 sell decision 拆成 decide_sell(state, symbol, mutate=False)。
2. preview 调用 mutate=False。
3. act 调用 mutate=True。
4. 对 trailing stop 增加 preview no-side-effect 单元测试。
```

### 15.5 lookback 语义要明确

当前风险：

```text
BacktestConfig.look_back_days 当前按自然日 timedelta 回退。
用户可能理解为交易日 bar 数。
```

修改方向：

```text
1. Config 中区分 lookback_calendar_days 和 lookback_bars。
2. 如果配置 lookback_bars，则通过 Qlib calendar 取前 N 个交易日。
3. 旧 myquant 对齐配置可以继续使用 lookback_calendar_days: 100。
```

### 15.6 strategy extension operator 从默认 registry 移出

当前风险：

```text
Shuijiao* operator 注册在默认 operator registry。
这和“不依赖策略专属黑盒”的设计目标冲突。
```

修改方向：

```text
1. DEFAULT_REGISTRY 只保留 core operator。
2. 建立 ExtensionRegistry。
3. shuijiao legacy operator 移到 quantx.strategies.extensions.shuijiao_legacy。
4. Config 中显式声明 extensions 才能加载。
```

### 15.7 fields 映射不能走普通公式 AST

当前风险：

```text
fields:
  close: $close
```

`$close` 不是合法 Python name，不能交给公式 parser。

修改方向：

```text
1. FieldSpec 单独解析。
2. DataLoader 根据 fields 决定 Qlib D.features 字段。
3. MarketPanel 只暴露 alias 后变量，例如 close。
4. factors/signals/rules 只能引用 alias。
```
