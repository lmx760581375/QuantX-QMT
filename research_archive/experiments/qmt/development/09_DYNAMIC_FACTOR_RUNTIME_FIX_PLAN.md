# QuantX 回测修正与动态高速因子运行方案

> **日期**: 2026-07-04
> **背景**: 新版 QuantX 已将基础行情读取切到 Qlib，但当前回测结果尚不能与旧版 `myquant-strategy-baseline` 对齐；同时策略因子经常调整，不希望依赖长期预计算落盘来提速。
> **目标**: 修正当前回测链路中的正确性问题，并建设一个“本次回测内动态高速计算”的因子运行层，让几千只股票的横截面/市场面因子计算本身足够快。

---

## 1. 总体目标

当前诉求不是把每个策略因子提前算好落盘，而是：

1. Qlib 负责快速读取基础行情。
2. QuantX 在每次回测启动时，根据当前策略和参数即时计算因子。
3. 因子计算采用全市场矩阵化/向量化方式，避免按日期、按股票循环。
4. 策略公式和参数可以频繁调整，调整后重新计算，但仍然足够快。
5. 回测成交、账户、交易规则继续由 QuantX 自研引擎控制，便于贴合 A 股和旧版 myquant 逻辑。

目标运行形态：

```text
Qlib binary data
  -> D.features 批量读取 OHLCV
  -> 标准化为 (datetime, instrument) MultiIndex
  -> 转为内存矩阵 close/high/low/open/volume: [T, N]
  -> 本次回测内动态计算策略因子: [T, N] / [T]
  -> 策略按日期读取当天横截面信号
  -> QuantX engine 顺序执行交易
```

这里的“动态计算”可以使用 run-level 内存缓存，避免同一次回测内重复算同一个中间结果；但不以长期落盘预计算作为主要加速方式。

### 1.1 本轮落地状态

已完成：

1. 修复 Qlib bin 写入格式，并增加 converter 回归测试。
2. `AStockExchange` 统一 `(datetime, instrument)` index，并优先导入 conda 环境中已安装的 qlib。
3. `BacktestContext` 在加载行情后构建 `MarketPanel + FactorRuntime`。
4. `shuijiao` 核心指标改为展开公式 DAG，不再在主路径依赖 `ShuijiaoBuy/ShuijiaoTrend` 黑盒算子。
5. 公式运行时补充 `And/Or/Not`，默认拒绝 `&/|/~`，公式错误默认 fail-fast。
6. 修复 Phase 1 信号 list 与 Phase 2 `StockSelection` 的衔接。
7. 修复旧 myquant 兼容口径中的涨停买入检查、同日卖出现金复用 sizing、预览卖出副作用等执行层问题。

仍需继续：

1. 正式 YAML/config runner。
2. 交易规则语言与状态变量 validator。
3. strategy extension operator 与 core operator 拆分。
4. lookback 从 calendar day 改成 trading day 的明确语义。
5. 进一步解释 qlib 数据口径与旧 outputs artifact 的收益差异。

---

## 2. 当前已发现的关键问题

### 2.1 Qlib bin 写入格式疑似错误

位置：`quantx/quantx/core/data/converter.py`

当前 `_write_features()` 将每个字段写成二维 pair：

```python
np.column_stack([date_indices[valid], values[valid]]).astype(np.float32).tofile(str(bin_path))
```

这会生成类似：

```text
date_idx_0, value_0, date_idx_1, value_1, ...
```

但 Qlib day bin 的常见格式是：

```text
start_index, value_0, value_1, value_2, ...
```

抽样验证时，`SZ000008` 从 Qlib 读取的数据出现了不可能的价格：一天正常，一天变成 `1297.0`、`1298.0` 这类日历 index 值，说明 date index 被当成行情值读出来了。

影响：

- OHLCV 错乱。
- 技术指标全部失真。
- 涨跌停、成交价、账户估值全部不可信。
- 新旧回测无法比较。

修正方向：

1. 按 Qlib `dump_bin.py` 的格式重写 converter。
2. 每个字段 bin 文件只写一次起始日历 index，然后连续写 value 序列。
3. 对缺失日期按全局 calendar 对齐，可以填 `NaN`，但不要把 date index 与 value 交替写入。
4. 重建 `quantx/data/qlib_data`。
5. 增加数据健康检查：随机抽样比对 raw CSV 与 `D.features()` 返回值。

验收标准：

- 随机抽样 100 只股票、100 个日期，`raw CSV close/open/high/low` 与 `D.features()` 结果一致，允许浮点误差。
- 不再出现 `1297.0` 这种日历 index 进入价格字段的情况。
- `$change` 与 `(close - preclose) / preclose` 一致。

### 2.2 Qlib MultiIndex 顺序与 QuantX 查询逻辑不一致

位置：

- `quantx/quantx/core/engine/exchange.py`
- `quantx/quantx/core/engine/context.py`
- `quantx/quantx/strategies/shuijiao_strategy.py`

Qlib `D.features()` 默认返回 index：

```text
(instrument, datetime)
```

但当前 QuantX 多处按以下方式查询：

```python
self.quote.loc[(pd.Timestamp(date), symbol), "$open"]
self.quote.loc[pd.Timestamp(date)]
```

也就是期望 index 为：

```text
(datetime, instrument)
```

影响：

- `get_current_data()` 可能取不到当天横截面。
- `get_deal_price()` 取不到成交价。
- `get_close()` 取不到估值价。
- `is_limit_up()` / `is_limit_down()` 逻辑失效。
- 回测可能大量无成交或错误成交。

修正方向：

在 `AStockExchange.load_quote_data()` 里统一标准化：

```python
self.quote = D.features(...)
self.quote = self.quote.swaplevel("instrument", "datetime").sort_index()
self.quote.index.names = ["datetime", "instrument"]
```

并要求 QuantX 内部统一使用 `(datetime, instrument)`。

验收标准：

- `context.get_current_data("2021-01-04")` 返回当天所有股票横截面。
- `exchange.get_deal_price("SZ000008", "2021-01-04")` 返回 `$open`。
- `exchange.get_close("SZ000008", "2021-01-04")` 返回 `$close`。

### 2.3 Shuijiao 指标 fallback 代码不可用

位置：`quantx/quantx/strategies/shuijiao_strategy.py`

当前 `_get_indicators()` 中：

```python
return self._compute_indicators(state, date)
"""计算所有技术指标"""
...
```

后面的指标计算代码位于 `return` 之后，不属于任何真正的方法体；同时类中没有可用的 `_compute_indicators()` 定义。

影响：

- 如果 `context._shuijiao_indicators` 不存在，选股阶段会报错。
- Phase 1 可能每天失败，信号为空。

修正方向：

1. 移除当前不可达代码。
2. 不建议恢复为“按日期 fallback 实时计算”。
3. 改为接入新的动态因子运行层：策略只从 `context.factor_runtime` 或 `context.factor_view` 获取当天因子横截面。
4. 若没有准备好所需因子，直接抛出清晰错误，而不是静默返回空信号。

### 2.4 Shuijiao 默认参数缺失

位置：`quantx/quantx/strategies/shuijiao_strategy.py`

`ShuijiaoSelector.act()` 使用：

```python
p["enable_market_health_entry"]
```

但 `ShuijiaoSelector.DEFAULT_PARAMS` 中没有该 key。

影响：

- 不显式传参时触发 `KeyError`。
- Phase 1 选股失败。

修正方向：

- 在 `ShuijiaoSelector.DEFAULT_PARAMS` 中补齐 `enable_market_health_entry`。
- 参数应分层管理，避免 selector/rebalance/execution 三个 policy 接收彼此无关参数后产生隐式依赖。

### 2.5 新旧 Shuijiao 策略 setting 不一致

旧版 `myquant-strategy-baseline` 的 shuijiao 逻辑：

- 选股：上一交易日 `buy_signal == 1`。
- 无 RSI/KDJ/BBI 二次过滤。
- 无 market health 入场过滤。
- 最大持仓 `100`。
- 用当天 `close` 成交。
- 止盈 `0.20`。
- 止损 `-0.10`。
- 交易成本来自 `configs/test_shuijiao.py`。

新版 QuantX 默认逻辑：

- 加了 RSI/KDJ/BBI 过滤。
- 加了 market health 过滤。
- 加了 score 排序。
- 最大持仓默认 `30`，弱市动态降为 `12` 或 `6`。
- 用 `$open` 成交。
- 止盈 `0.22`。
- 止损 `-0.09`。
- 开启 momentum sell、hard risk off。
- 成本参数与旧版不一致。

影响：

即使数据完全正确，新旧收益率和选股结果也不会类似。

修正方向：

新增 `legacy_myquant_compatible` 配置模式，用于对齐验证：

```text
selector:
  enable_market_health_entry: false
  enable_rsi_filter: false
  enable_kdj_filter: false
  enable_bbi_filter: false
  enable_trend_guard: false
  enable_relax_entry_filters: false

rebalance:
  max_positions_base: 100
  enable_dynamic_positions: false
  enable_hard_risk_off: false

execution:
  deal_price: close
  max_profit_pct: 0.20
  max_loss_pct: -0.10
  enable_momentum_sell: false
  enable_hard_risk_off: false
```

成本参数对齐旧版：

```text
commission_rate: 0.0005
min_commission: 5.0
stamp_tax_rate: 0.0001
slippage: 0.0001
transfer_fee_rate: 0.0 或明确旧版未计过户费
```

### 2.6 当前 lookback 处理不足

旧版 pkl 在 `2021-01-04` 已经有完整指标，因为缓存构建时使用了 `look_back_days=100`。新版 `BacktestContext.load_data()` 虽然参数里有 `look_back_days`，但实际调用 `exchange.load_quote_data(symbols, start, end)` 时没有向前扩展起始日期。

影响：

- 回测开头一段 rolling 指标和旧版必然不一致。
- `trend_line`、BBI、RSI、KDJ 等起始值失真。

修正方向：

1. 加载行情时使用 `load_start = calendar.shift(start, -lookback)`。
2. 因子计算使用 `load_start ~ end`。
3. 交易和绩效统计只从 `start ~ end` 开始。
4. 在动态因子运行层中区分 `data_dates` 与 `trade_dates`。

---

## 3. 不采用长期预计算落盘作为核心方案

本方案刻意避免把策略因子长期预计算为文件作为主要加速手段，原因：

1. 策略因子和参数经常调整，长期缓存很容易失效。
2. 复杂策略会有大量参数组合，落盘缓存管理成本高。
3. 横截面因子常依赖当日全市场状态，缓存粒度和版本管理复杂。
4. 调研/迭代阶段更需要“修改公式后立刻重新跑”。

允许的缓存边界：

- 基础行情由 Qlib binary 存储，这是数据层缓存。
- 本次回测进程内可以缓存中间矩阵，例如 `llv_55`、`hhv_55`、`ema_12`。
- 可选 debug 模式下落盘本次因子结果用于对账，但不作为核心性能机制。

不建议的方式：

- 每次策略改参数都去维护一套落盘因子文件。
- 在策略 `get_stock_signal(date)` 内按日期逐日计算 rolling 指标。
- 对每只股票单独 pandas rolling 后再拼接，作为主路径。

---

## 4. 目标架构：动态高速因子运行层

新增模块建议：

```text
quantx/core/factor_runtime/
├── __init__.py
├── panel.py              # MultiIndex DataFrame <-> 矩阵 Panel
├── runtime.py            # FactorRuntime 主入口
├── ops.py                # 高性能 rolling/cross/ema/sma/filter 等算子
├── spec.py               # 因子计算声明 FactorSpec
├── formula.py            # 类 Qlib 公式层入口：解析公式并提交 runtime 执行
├── parser.py             # 公式 lexer/parser，生成 AST
├── ast.py                # 公式 AST 节点定义
├── planner.py            # AST -> DAG -> 执行计划
├── operators.py          # 公式算子注册表，映射到底层矩阵算子
├── result.py             # FactorResult / FactorView
└── validators.py         # 因子结果校验和对账工具
```

### 4.1 Panel 数据结构

将 qlib 读取的 `(datetime, instrument)` MultiIndex DataFrame 转为矩阵：

```python
class MarketPanel:
    dates: pd.DatetimeIndex
    instruments: pd.Index
    fields: dict[str, np.ndarray]  # field -> float32 array, shape [T, N]
```

基础字段：

```text
open:   [T, N]
high:   [T, N]
low:    [T, N]
close:  [T, N]
volume: [T, N]
change: [T, N]
```

优势：

- 横截面计算是 `axis=1`。
- 时间序列 rolling 是 `axis=0`。
- 几千只股票可以一次性计算，不再拆成几千个 DataFrame。
- 因子公式改了，只需要重新执行本次内存计算。

### 4.2 FactorRuntime 接口

```python
class FactorRuntime:
    def __init__(self, panel: MarketPanel, config: RuntimeConfig):
        self.panel = panel
        self.memory_cache = {}

    def compute(self, spec: FactorSpec) -> FactorResult:
        """根据当前策略因子声明即时计算，不依赖长期落盘预计算。"""

    def get_cross_section(self, factor_name: str, date: pd.Timestamp) -> pd.DataFrame:
        """返回某日所有股票的因子横截面。"""

    def get_matrix(self, factor_name: str) -> np.ndarray:
        """返回完整 [T, N] 因子矩阵。"""
```

### 4.3 FactorSpec 设计

策略声明需要什么，而不是自己在 `act()` 中循环计算：

```python
@dataclass
class FactorSpec:
    name: str
    params: dict
    inputs: list[str]
    outputs: list[str]
    compute_fn: Callable[[MarketPanel, dict], dict[str, np.ndarray]]
```

Shuijiao 示例：

```python
ShuijiaoFactorSpec(
    params={"trend_period": 55},
    outputs=[
        "trend_line",
        "buy_signal",
        "ready_sell",
        "ema12",
        "ema26",
        "bbi",
        "rsi14",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "market_health",
    ],
)
```

### 4.4 高性能算子

优先级：

1. NumPy 向量化。
2. bottleneck 处理 rolling min/max/mean。
3. numba 处理递归类指标，例如通达信 SMA、EMA、FILTER。
4. 必要时再考虑 Cython 或接入 Qlib custom operator。

需要实现的基础算子：

```text
rolling_min(x, window, axis=0)
rolling_max(x, window, axis=0)
rolling_mean(x, window, axis=0)
ema(x, span, axis=0)
tdx_sma(x, n, m, axis=0)
cross(a, b)
filter_signal(cond, n, axis=0)
rank_cs(x, axis=1)
zscore_cs(x, axis=1)
percentile_cs(x, axis=1)
```

Shuijiao 中最关键的是：

```text
LLV(low, 55)
HHV(high, 55)
SMA(rsv, 5, 1)
EMA(v11, 3)
CROSS(trend, threshold)
FILTER(trend > 89, 15)
market_health = mean(trend_line > 20, axis=1)
```

### 4.5 类 Qlib 公式层：QuantX Formula DSL

为了避免后续每个策略都手写一套 Python 指标函数，FactorRuntime 上层需要提供一个类似 Qlib expression 的公式层。这个公式层的目标不是实现完整 Python，而是提供一套可解析、可依赖分析、可矩阵化执行的策略因子语言。

核心目标：

1. 策略因子易改：公式和参数可以放在 YAML/Python dict 中。
2. 计算和策略解耦：策略声明需要哪些因子，runtime 负责高效计算。
3. 自动依赖分析：`buy_signal` 依赖 `trend_line`，`trend_line` 依赖 `rsv`，按 DAG 自动排序。
4. 中间结果复用：同一个 `Max(high, 55)` 在多个公式中只计算一次。
5. 支持市场面因子：提供 `CS*` 横截面算子，例如 `CSMean`、`CSRank`、`CSCount`。
6. 支持广播：`market_health` 是 `[T]`，可以和 `buy_signal` 的 `[T, N]` 自动广播组合。

不做的事情：

- 不实现完整 Python 语法。
- 不允许任意 Python `eval`。
- 不把公式逐元素解释执行。
- 不把公式层做成长期落盘预计算系统。
- 第一版不追求完全兼容 Qlib expression，只借鉴其表达风格。

#### 4.5.1 公式语言边界

第一版只支持以下语法：

```text
变量名: close, high, low, trend_line, buy_signal
数字: 55, 0.22, 1e-6
四则运算: + - * / %
比较运算: > >= < <= == !=
布尔运算: & | ~
括号: (...)
函数调用: Ref(close, 1), EMA(v11, 3), CSMean(trend_line > 20)
命名参数: ShuijiaoTrend(close, high, low, period=55)
参数占位符: ${trend_period}, ${health_threshold}
```

第一版不支持：

```text
for/while/if 语句
lambda
列表推导
任意属性访问
任意模块导入
用户随意调用未注册函数
```

#### 4.5.2 算子分类

公式算子分为四类。

时间序列算子，沿 `axis=0` 计算，输出通常为 `[T, N]`：

```text
Ref(x, n)
Mean(x, n)
Min(x, n)
Max(x, n)
Std(x, n)
EMA(x, span)
SMA_TDX(x, n, m)
Cross(a, b)
Filter(cond, n)
```

横截面/市场面算子，沿 `axis=1` 计算，输出可能是 `[T]` 或 `[T, N]`：

```text
CSMean(x)              # 每天横截面均值；条件 bool 时等价于比例
CSCount(cond)          # 每天满足条件的股票数
CSRatio(cond)          # 每天满足条件的股票占比，等价于 CSMean(cond)
CSRank(x)              # 每天横截面排名，输出 [T, N]
CSPctRank(x)           # 每天横截面百分位排名，输出 [T, N]
CSQuantile(x, q)       # 每天横截面 q 分位数，输出 [T]
TopK(x, k)             # 每天 top-k mask，输出 bool [T, N]
BottomK(x, k)          # 每天 bottom-k mask，输出 bool [T, N]
```

元素级算子，对 `[T, N]` 或 `[T]` 做广播计算：

```text
Abs(x)
Maximum(a, b)
Minimum(a, b)
Where(cond, a, b)
IsFinite(x)
IsNan(x)
```

复合领域算子，用于封装很长、很容易写错的策略公式：

```text
ShuijiaoTrend(close, high, low, period=55)
ShuijiaoMidline(close, high, low)
ShuijiaoBuy(trend_line, close, midline)
ShuijiaoReadySell(trend_line, close, midline)
```

实现原则：复杂指标可以封装为高性能 operator，外层组合用公式语言。不要强迫所有复杂逻辑都写成超长字符串。

#### 4.5.3 公式配置示例

完整展开写法：

```yaml
factors:
  H1: "Maximum(high, Ref(close, 1))"
  L1: "Minimum(low, Ref(close, 1))"
  P1: "H1 - L1"
  resistance: "L1 + P1 * 7 / 8"
  support: "L1 + P1 * 0.5 / 8"
  midline: "(support + resistance) / 2"
  rsv: "(close - Min(low, ${trend_period})) / (Max(high, ${trend_period}) - Min(low, ${trend_period}) + 1e-10) * 100"
  v11: "3 * SMA_TDX(rsv, 5, 1) - 2 * SMA_TDX(SMA_TDX(rsv, 5, 1), 3, 1)"
  trend_line: "EMA(v11, 3)"
  bb1: "Ref(trend_line, 1) < 11 & Ref(trend_line, 1) > 6 & Cross(trend_line, 11)"
  bb2: "Ref(trend_line, 1) < 6 & Ref(trend_line, 1) > 3 & Cross(trend_line, 6)"
  bb3: "Ref(trend_line, 1) < 3 & Ref(trend_line, 1) > 1 & Cross(trend_line, 3)"
  bb4: "Ref(trend_line, 1) < 1 & Ref(trend_line, 1) > 0 & Cross(trend_line, 1)"
  bb5: "Ref(trend_line, 1) < 0 & Cross(trend_line, 0)"
  buy_signal: "(bb1 | bb2 | bb3 | bb4 | bb5) & (close < midline)"
  ready_sell: "trend_line > 89 & Filter(trend_line > 89, 15) & close > midline"
  market_health: "CSMean(trend_line > 20)"
```

推荐生产写法：

```yaml
params:
  trend_period: 55
  health_threshold: 20

factors:
  trend_line: "ShuijiaoTrend(close, high, low, period=${trend_period})"
  midline: "ShuijiaoMidline(close, high, low)"
  buy_signal: "ShuijiaoBuy(trend_line, close, midline)"
  ready_sell: "ShuijiaoReadySell(trend_line, close, midline)"
  market_health: "CSMean(trend_line > ${health_threshold})"
  trend_rank: "CSPctRank(trend_line)"
```

推荐生产写法的原因：

- 复杂通达信公式不在 YAML 中写成一大坨，降低误写概率。
- 复合算子内部可以用 numba/bottleneck 深度优化。
- 外层仍然可以灵活组合市场面过滤、打分和选股条件。

#### 4.5.4 执行流程

公式计算不应使用 Python `eval`。推荐流程：

```text
formula strings
  -> tokenize
  -> parse to AST
  -> resolve variables and registered operators
  -> build dependency DAG
  -> topological sort
  -> execute each node with matrix operators
  -> store outputs in FactorRuntime memory cache
```

数据形状约定：

```text
基础行情字段: [T, N]
时间序列因子: [T, N]
横截面 rank/top mask: [T, N]
市场标量序列: [T]
常数: scalar
```

广播规则：

```text
scalar 与任意 shape 运算 -> 广播
[T] 与 [T, N] 运算 -> [T, 1] 广播到 [T, N]
[T, N] 与 [T, N] 运算 -> 原形状
其他形状组合默认报错，不做隐式猜测
```

布尔规则：

- 比较运算输出 bool matrix 或 bool vector。
- `& | ~` 只用于 bool 类型。
- bool 可被 `CSMean` 当作 0/1 统计比例。
- bool 可作为 `Where(cond, a, b)` 的条件。

缺失值规则：

- 基础行情缺失使用 `NaN`。
- rolling 算子默认 `min_periods=1`，以对齐旧版 shuijiao。
- 横截面统计默认忽略 `NaN`。
- 对停牌/不可交易股票是否参与 `CSMean` 需要显式 mask，不做隐式排除。

#### 4.5.5 OperatorRegistry

所有公式函数必须注册后才能调用：

```python
@operator(
    name="CSMean",
    kind="cross_section",
    input_shapes=("matrix_or_bool",),
    output_shape="vector",
)
def op_csmean(x, mask=None):
    return np.nanmean(apply_mask(x, mask), axis=1)
```

注册信息至少包含：

```text
name
kind: time_series | cross_section | elementwise | composite
input dtype/shape 约束
output shape
是否支持 bool
是否需要 lookback
底层实现函数
文档说明
```

OperatorRegistry 的作用：

1. 禁止公式调用未注册函数。
2. 做形状和 dtype 校验。
3. 为 planner 提供 lookback 需求。
4. 为性能统计记录每个 operator 耗时。
5. 允许后续把同名 operator 从 numpy 实现替换为 numba 实现，而不改变公式。

#### 4.5.6 DAG 与缓存 key

每个公式节点生成稳定 cache key：

```text
operator_name
normalized_args
normalized_kwargs
input_node_hashes
params_hash
runtime_version
```

示例：

```text
Max(high, 55)
SMA_TDX(rsv, 5, 1)
CSMean(trend_line > 20)
```

同一次回测中，相同 cache key 只计算一次。参数变化后 key 变化，自动重算。

注意：这是进程内 run-level cache，不是长期落盘预计算。

#### 4.5.7 与 Qlib expression 的边界

Qlib expression 继续用于：

```text
基础数据读取
简单标准表达式
Qlib 已经很好支持的单股票时间序列表达式
```

QuantX Formula DSL 用于：

```text
通达信风格策略指标
复杂组合信号
市场面横截面因子
横截面 rank/分位数/top-k
需要和 QuantX 回测配置、股票池、停牌 mask 深度结合的因子
```

不要试图第一版完全替代 Qlib expression。正确边界是：Qlib 负责快读和一部分标准表达式；QuantX Formula 负责策略研究中需要快速迭代的高层因子语言。

#### 4.5.8 第一版最小可行范围

第一版只为 shuijiao 和常见市场面因子服务，必须实现：

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
CSMean
CSCount
CSRank
CSPctRank
TopK
ShuijiaoTrend
ShuijiaoMidline
ShuijiaoBuy
ShuijiaoReadySell
```

第一版必须配套的 debug 能力：

```python
runtime.explain("market_health")
runtime.profile()
runtime.inspect(symbol="SZ000008", date="2021-01-04", factors=["trend_line", "buy_signal"])
runtime.to_frame(["trend_line", "buy_signal"], symbols=["SZ000008"])
```

这些 debug API 是为了防止矩阵化以后难以对账。

---

## 5. Shuijiao 动态矩阵化实现方案

旧版逻辑按单股票 pandas Series 计算。新版应改为矩阵计算：

```text
close, high, low: [T, N]

H1 = maximum(high, shift(close, 1))
L1 = minimum(low, shift(close, 1))
P1 = H1 - L1
resistance = L1 + P1 * 7 / 8
support = L1 + P1 * 0.5 / 8
midline = (support + resistance) / 2

rsv = (close - rolling_min(low, 55)) / (rolling_max(high, 55) - rolling_min(low, 55)) * 100
v11 = 3 * tdx_sma(rsv, 5, 1) - 2 * tdx_sma(tdx_sma(rsv, 5, 1), 3, 1)
trend_line = ema(v11, 3)

BB1 = prev(trend_line) < 11 and prev(trend_line) > 6 and cross(trend_line, 11)
BB2 = prev(trend_line) < 6  and prev(trend_line) > 3 and cross(trend_line, 6)
BB3 = prev(trend_line) < 3  and prev(trend_line) > 1 and cross(trend_line, 3)
BB4 = prev(trend_line) < 1  and prev(trend_line) > 0 and cross(trend_line, 1)
BB5 = prev(trend_line) < 0  and cross(trend_line, 0)

buy_signal = (BB1 | BB2 | BB3 | BB4 | BB5) & (close < midline)
ready_sell = (trend_line > 89) & filter_signal(trend_line > 89, 15) & (close > midline)
market_health = mean(trend_line > 20, axis=1)
```

策略执行时：

```text
selector.act(date):
  row = factor_runtime.get_cross_section("shuijiao", last_trade_date)
  select row[buy_signal]
  optional filters/scores

execution.act(date):
  row = factor_runtime.get_cross_section("shuijiao", date)
  use ready_sell / momentum fields
```

这样不是长期预计算，而是本次回测启动阶段基于当前参数即时计算一次完整矩阵。

---

## 6. 回测引擎改造方案

### 6.1 数据加载阶段

当前：

```text
context.load_data(symbols, start, end)
  -> exchange.load_quote_data(symbols, start, end)
```

改为：

```text
context.load_data(symbols, start, end, look_back_days)
  -> resolve load_start by trading calendar
  -> exchange.load_quote_data(symbols, load_start, end)
  -> normalize index to (datetime, instrument)
  -> build MarketPanel
  -> trade_dates = start ~ end
```

### 6.2 策略准备阶段

新增 strategy prepare hook：

```python
class CompositeStrategy:
    def prepare(self, context):
        for policy in [selector, rebalance, execution]:
            if hasattr(policy, "prepare"):
                policy.prepare(context)
```

Shuijiao prepare：

```text
ShuijiaoSelector.prepare(context)
  -> register/compute ShuijiaoFactorSpec
  -> factor_runtime stores matrices in memory
```

### 6.3 Phase 1 的职责调整

当前 Phase 1 叫“预计算信号”，容易和长期因子预计算混淆。建议调整为：

```text
prepare phase: 动态计算本次回测所需因子矩阵
signal phase: 逐日从因子矩阵中抽取信号
execution phase: 顺序执行交易
```

关键点：

- 因子计算不在每日 `act()` 中重复发生。
- 因子结果不要求落盘。
- 参数变化时重新运行 prepare phase。

### 6.4 成交价配置化

为对齐旧版和支持不同假设，成交价应可配置：

```text
deal_price: open | close | vwap | next_open
valuation_price: close
```

旧版兼容模式用 `close`。
研究实盘近似模式可用 `open` 或 `next_open`。

### 6.5 成本模型配置化

将成本参数从默认 dataclass 转为 BacktestConfig 显式配置，支持 legacy preset：

```text
cost_preset: myquant_legacy
commission_rate: 0.0005
min_commission: 5.0
stamp_tax_rate: 0.0001
slippage: 0.0001
transfer_fee_rate: 0.0
```

---

## 7. 分阶段实施计划

### Phase 0: 建立对账基准

目标：先明确“对齐旧版”需要比较什么。

任务：

1. 固定样本股票池，例如旧版 `stock_pool=all` 过滤后的非科创、非创业板股票。
2. 固定日期范围，例如 `2021-01-01 ~ 2021-12-31`。
3. 从旧版 pkl 中导出以下基准：
   - raw close/open/high/low。
   - `trend_line`。
   - `buy_signal`。
   - `ready_sell`。
   - 每日选股列表。
   - 交易记录。
   - 每日 total_value。
4. 形成 `tests/fixtures/shuijiao_legacy_*` 或临时 verification script。

验收：

- 能够一键输出新旧差异报告。

### Phase 1: 修正 Qlib 数据层正确性

任务：

1. 修正 converter bin 写入格式。
2. 重建 qlib data。
3. 修正 index 顺序为 `(datetime, instrument)`。
4. 增加数据健康检查脚本。
5. 增加单元测试覆盖 `get_current_data/get_deal_price/get_close/get_change`。

验收：

- 原始 CSV 与 Qlib 读取值抽样一致。
- `SZ000008 2021-01-04 close` 从 Qlib 读取为 `2.52`。
- 不再出现日历 index 被读成价格。

### Phase 2: 实现 MarketPanel 与动态因子运行层最小版本

任务：

1. 新增 `MarketPanel`。
2. 实现 MultiIndex DataFrame 到矩阵的转换。
3. 实现基础算子：shift、rolling_min、rolling_max、rolling_mean、ema、tdx_sma、cross、filter_signal。
4. 实现 `FactorRuntime` 和 run-level memory cache。
5. 实现 `OperatorRegistry`，所有公式函数必须注册后才能使用。
6. 实现 QuantX Formula DSL 最小版本：lexer/parser、AST、DAG planner、执行器。
7. 支持公式配置中的 `${param}` 参数替换和 cache key 生成。
8. 对一小批股票验证矩阵算子与旧 pandas Series 算法一致。

验收：

- 100 只股票、1 年数据，矩阵版 `trend_line/buy_signal/ready_sell` 与旧版 pandas 结果高度一致。
- 因子参数变化后重新计算，不依赖落盘缓存。
- 能用公式字符串计算 `market_health = CSMean(trend_line > 20)`。
- `runtime.explain("market_health")` 能输出依赖 DAG，不允许使用 Python `eval`。

### Phase 3: 接入 Shuijiao 策略

任务：

1. 将 shuijiao 指标计算迁移到 runtime spec 和公式配置。
2. `ShuijiaoSelector` 从 runtime 获取 `last_trade_date` 横截面。
3. `ShuijiaoExecution` 从 runtime 获取当天 `ready_sell` 与动量字段。
4. 移除不可达 fallback 指标代码。
5. 补齐默认参数。
6. 增加 `legacy_myquant_compatible` 模式。
7. 为复杂公式提供 `ShuijiaoTrend/ShuijiaoMidline/ShuijiaoBuy/ShuijiaoReadySell` 复合算子，避免在策略代码中手写重复指标逻辑。

验收：

- 在 legacy 模式下，每日选股列表与旧版接近或完全一致。
- 差异能被数据源口径、停牌/涨跌停、成本或成交价配置解释。
- Shuijiao 策略需要的市场面因子通过 DSL 表达，例如 `market_health: "CSMean(trend_line > 20)"`。

### Phase 4: 回测交易逻辑对齐旧版

任务：

1. 成交价支持 `close`。
2. max positions 支持 `100`。
3. 成本 preset 对齐旧版。
4. 关闭新版额外风控和动量卖出。
5. 交易记录字段增加兼容输出，便于和旧版 `trades.json` 对账。

验收：

- legacy 模式下，收益率曲线与旧版接近。
- 每笔交易差异可追溯到明确原因。

### Phase 5: 性能优化

任务：

1. 基准测试：数据加载、panel 构建、shuijiao 计算、信号抽取、交易执行分别计时。
2. rolling min/max 优先接入 bottleneck。
3. tdx_sma/ema/filter_signal 接入 numba。
4. 控制 dtype 为 `float32`，降低内存压力。
5. 支持分块计算：当股票数量过多时按 instrument chunk 计算，但仍保持矩阵批量，不回退到逐股票 pandas。

验收建议：

- 约 4000 只股票、5 年日线数据，基础 shuijiao 因子动态计算控制在可接受时间内。
- 因子公式调整后重新运行，不需要清理落盘缓存。

---

## 8. 性能设计原则

### 8.1 禁止主路径逐日逐股循环

不应出现：

```python
for date in dates:
    for symbol in symbols:
        history = get_history(symbol, date)
        compute_factor(history)
```

也不应把几千只股票拆成几千个 pandas DataFrame 作为核心计算路径。

### 8.2 优先矩阵化

推荐：

```python
close = panel.fields["close"]  # [T, N]
trend = compute_trend(close, high, low)
market_health = np.nanmean(trend > 20, axis=1)
```

### 8.3 因子动态但中间结果可在内存复用

同一次回测内，如果多个因子都依赖 `rolling_max(high, 55)`，可以在内存中复用：

```text
memory_cache[("rolling_max", "high", 55)]
```

但该缓存随进程生命周期结束，不作为长期预计算资产。

### 8.4 参数变化触发重新计算

因子 runtime cache key 应包含：

```text
factor_name
params
input_fields
date_range
instrument_universe
runtime_version
```

这样同一次实验内可以避免重复计算，参数一变会自动重算。

### 8.5 公式层不得退化为 Python 函数集合

策略因子应优先声明为公式配置：

```yaml
factors:
  trend_line: "ShuijiaoTrend(close, high, low, period=${trend_period})"
  market_health: "CSMean(trend_line > ${health_threshold})"
```

不应在策略 `act()` 或 `prepare()` 中直接拼接大量 Python 指标计算代码。允许新增 Python 代码的地方只有：

- 底层矩阵算子实现，例如 `op_ema`、`op_csmean`。
- 复合算子实现，例如 `ShuijiaoTrend`。
- 公式 parser/planner/runtime 本身。

这样可以保证策略层保持声明式，后续策略迁移时不会重新走回“每个策略一堆 pandas 函数”的老路。

### 8.6 市场面因子必须显式使用 CS 算子

市场面因子统一使用 `CS*` 系列算子表达，不应在策略层手写每日横截面循环。

推荐：

```text
market_health = CSMean(trend_line > 20)
buy_count = CSCount(buy_signal)
trend_rank = CSPctRank(trend_line)
top_mask = TopK(score, 30)
```

不推荐：

```python
for date in dates:
    row = df.loc[date]
    market_health[date] = (row["trend_line"] > 20).mean()
```

CS 算子是公式层适配市场面因子的核心，不要把它绕开。

---

## 9. 测试与验收矩阵

### 9.1 数据层测试

| 测试 | 目标 |
|------|------|
| raw_vs_qlib_sample | raw CSV 与 D.features 抽样一致 |
| qlib_index_order | quote index 为 `(datetime, instrument)` |
| get_current_data | 能取到指定日期横截面 |
| get_deal_price | open/close/vwap 成交价配置正确 |
| limit_flags | 涨跌停判断与 `$change` 一致 |

### 9.2 因子 runtime 测试

| 测试 | 目标 |
|------|------|
| panel_shape | `[T, N]` 维度和 dates/instruments 对齐 |
| rolling_ops_compare | rolling 算子与 pandas 单股票结果一致 |
| tdx_sma_compare | 通达信 SMA 与旧版实现一致 |
| shuijiao_factor_compare | trend/buy/ready_sell 与旧版 pkl 对齐 |
| params_change_recompute | 参数变化后重新计算 |

### 9.3 公式 DSL 测试

| 测试 | 目标 |
|------|------|
| formula_parse_basic | 四则运算、比较、布尔、函数调用可解析 |
| formula_reject_eval | 不允许任意 Python eval、属性访问、未注册函数 |
| formula_dag_order | 依赖图拓扑排序正确 |
| formula_cache_reuse | 相同子表达式同一次运行只计算一次 |
| formula_broadcast | `[T]` 市场因子能广播到 `[T, N]` |
| cs_ops_compare | `CSMean/CSCount/CSRank/TopK` 与 pandas 横截面对账一致 |
| formula_explain | `runtime.explain()` 能输出可读 DAG |

### 9.4 回测对账测试

| 测试 | 目标 |
|------|------|
| daily_selection_compare | 每日选股列表对齐旧版 |
| trades_compare | 买卖日期、股票、价格、数量对齐旧版 |
| equity_curve_compare | 每日总资产曲线接近旧版 |
| legacy_mode_snapshot | legacy preset 不被后续改动破坏 |

---

## 10. 优先级排序

必须先做：

1. 修 Qlib bin 写入格式。
2. 修 Qlib index 顺序。
3. 建立 raw vs qlib 数据对账。

然后做：

4. 实现 MarketPanel。
5. 实现动态因子 runtime 最小版本。
6. 实现 QuantX Formula DSL 最小版本和 `CS*` 市场面算子。
7. Shuijiao 指标矩阵化并接入公式配置。

再做：

8. legacy myquant 兼容模式。
9. 新旧选股和交易对账。
10. 性能优化和 benchmark。

原因：如果数据层不正确，任何因子加速和回测对齐都没有意义。

---

## 11. 最终验收标准

本轮改造完成后，应满足：

1. Qlib 数据读取正确，字段值与 raw CSV 抽样一致。
2. QuantX 内部统一使用 `(datetime, instrument)` 数据索引。
3. Shuijiao 不再依赖损坏的 fallback 计算。
4. Shuijiao 因子可以根据当前参数在本次回测启动时动态高速计算。
5. 不需要长期预计算落盘即可完成几千只股票的横截面/市场面因子计算。
6. 市场面因子可以用 QuantX Formula DSL 表达，例如 `CSMean(trend_line > 20)`、`CSPctRank(score)`、`TopK(score, 30)`。
7. 公式层通过 parser/AST/DAG/runtime 执行，不使用任意 Python `eval`。
8. legacy 模式下，新版回测的选股列表、交易记录和收益率能与旧版 myquant 接近或可解释地一致。
9. 新版增强模式可以在 legacy 模式基础上逐步打开 RSI/KDJ/BBI/market health/dynamic positions 等新逻辑。

---

## 12. 备注

Qlib 的价值在这里主要是“快速、统一地读取全市场基础数据”。策略因子的性能问题不能只靠 Qlib 自动解决，必须避免逐日逐股计算，并建立 QuantX 自己的动态矩阵化因子运行层。

本方案中的 run-level 动态计算不是长期预计算。它允许因子随时调整，每次回测按当前公式和参数重新计算，同时通过矩阵化、numba/bottleneck 和进程内中间结果复用保证速度。
