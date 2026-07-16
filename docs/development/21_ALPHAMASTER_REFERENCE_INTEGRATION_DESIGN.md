# AlphaMaster 可借鉴能力的 QuantX 最小集成设计

> **版本**: v0.1.0
> **日期**: 2026-07-15
> **状态**: 设计提案，尚未实现
> **参考项目**: `git@github.com:rosemarycox5334-debug/AlphaMaster.git`
> **适用范围**: 因子质量诊断、策略质量评分、信号口径审计、数据源健康检查、属性测试、后续自动因子搜索
> **核心原则**: 不整体迁移 AlphaMaster，不复制 AGPL 代码，只吸收可低耦合集成的工程思想，并在 QuantX 现有架构内重新实现。

---

## 1. 文档目的

本文把 AlphaMaster 项目中可借鉴的能力转化为 QuantX-QMT 的最小集成需求文档。目标不是做项目调研摘要，而是给后续开发提供可拆分、可验收、可回滚的工业落地方案。

AlphaMaster 的核心特点是：

1. 使用深度模型在特征和算子空间中自动搜索公式。
2. 使用 StackVM 解释 token 公式，训练、回测和实时信号共用一套因子逻辑。
3. 对公式结构、因子方向偏置、换手率、成本压力、分段一致性等做了显式约束。
4. 提供 Web 控制台串联训练、回测和实时分析。
5. 有数据源可用性抽象和若干属性测试。

QuantX 当前的优势是：

1. A 股日频数据、QMT/xqshare、BaoStock、Qlib provider 和 MetaStore 已经形成稳定数据层。
2. `FactorRuntime`、配置化策略、回测引擎、交易成本、A 股交易规则和生产日报已经成体系。
3. `decision` 和 `research` 模块已经开始把训练、预测、组合和执行边界拆开。
4. 当前策略 YAML 与 run artifacts 数量较多，最需要的是筛选、诊断、审计和可复用评分，而不是立刻引入重型训练系统。

因此本文采用以下总路线：

```text
先诊断和评分
  -> 再把诊断接入 artifact 和 Web
  -> 再补属性测试保护关键不变量
  -> 最后才考虑自动因子搜索
```

---

## 2. 总体结论

### 2.1 不推荐整体迁移 AlphaMaster

AlphaMaster 不适合作为 QuantX 的替代架构，原因如下：

1. 项目边界更偏实验型，根目录存在大量一次性训练、检查、回测脚本。
2. 依赖较重，包括 PyTorch、MetaTrader5、TradingView、Playwright 等，不适合进入 QuantX 核心依赖。
3. 交易域是 MT5/外汇/指数/加密和连续仓位，QuantX 当前主域是 A 股日频选股、调仓、执行。
4. QuantX 已经具备更严格的信号日、执行日、A 股交易规则和生产日报体系。
5. AlphaMaster 采用 AGPL-3.0 协议，直接复制代码会带来开源义务和代码污染风险。

### 2.2 推荐吸收的能力

推荐吸收的是以下局部能力：

| 能力 | AlphaMaster 思想 | QuantX 落点 | 优先级 | 推荐程度 |
|---|---|---|---|---|
| 因子/公式 Lint | 检查公式结构退化、恒正链、常数因子、NaN/Inf | `quantx/core/analysis/quality` | P0 | 强烈推荐 |
| 策略质量评分 | 多目标评价：收益、回撤、成本、换手、稳定性 | run artifacts 二次评分 | P0 | 强烈推荐 |
| 信号口径合约 | 训练、回测、实时共用一套信号解释 | `decision` / artifacts / production report | P1 | 推荐 |
| 数据源健康检查 | `available()`、`coverage()`、latest session | 数据/生产诊断层 | P2 | 推荐但不急 |
| 属性测试 | 用 Hypothesis 验证不变量 | `tests/property` | P1 | 推荐 |
| Web 展示增强 | 展示质量分、warning、证据 | `quantx/server` | P1 | 推荐 |
| 自动因子搜索 | 在算子空间中自动生成候选公式 | `research/factor_search` | P3 | 后续可选 |

### 2.3 最小集成边界

第一阶段只新增只读分析能力，不改变任何交易行为：

```text
Strategy config / Backtest artifact / Factor output
  -> quality lint / quality score
  -> strategy_quality.json
  -> Web/report 展示
```

第一阶段不做以下事情：

1. 不引入 PyTorch。
2. 不引入 AlphaMaster 的 StackVM。
3. 不重写 `FactorRuntime`。
4. 不修改 `BacktestEngine` 撮合逻辑。
5. 不改变策略买卖信号和订单生成结果。
6. 不复制 AlphaMaster 代码。

---

## 3. 集成原则

### 3.1 架构原则

1. **高内聚**：质量诊断相关代码集中在 `quantx/core/analysis/quality`，不要散落在回测、策略、生产和 Web 层。
2. **低耦合**：诊断模块只消费配置、因子输出和 artifacts，不反向调用 Web、生产日报或具体策略类。
3. **如无必要，勿增实体**：第一版不新增数据库表、不新增后台任务、不新增训练服务。
4. **只读优先**：质量评分默认不影响回测、生产、下单，只作为提示和排序字段。
5. **现有接口优先**：优先复用 `explain_strategy`、`run_backtest` artifacts、`load_run_artifacts`、`ConfigService`。
6. **可选接入**：所有新功能都允许缺失 artifact 时降级为空结果。
7. **不污染核心依赖**：Hypothesis 只进 `dev` extra；PyTorch 等自动搜索依赖只进入后续 `dl` extra。
8. **A 股语义优先**：所有评分和诊断要适配日频 A 股，不照搬连续多空/外汇口径。

### 3.2 兼容原则

1. 现有 YAML 策略不需要修改即可被诊断。
2. 现有 run artifacts 缺字段时，评分器应返回 `insufficient_data` warning，而不是抛异常终止 Web 或日报。
3. 新增 `strategy_quality.json` 不影响旧报告读取。
4. Web 层读取质量信息失败时只隐藏质量卡片，不影响策略列表和回测详情。
5. 生产日报默认不因为质量分低而阻止发送。

### 3.3 合规原则

AlphaMaster 是 AGPL-3.0。后续实现必须遵守：

1. 不复制其源代码、注释、测试或前端文件。
2. 只使用公开可见的工程思想重新实现。
3. 新文件不得保留 AlphaMaster 的函数名、类名或大段结构。
4. 如确需引用，应在文档中标注为“参考思想”，代码中不保留来源实现。

---

## 4. 当前 QuantX 对应落点

### 4.1 相关现有模块

| QuantX 模块 | 当前职责 | 与本文关系 |
|---|---|---|
| `quantx/core/factor_runtime` | 动态因子算子和表达式运行时 | 因子输出健康检查、公式依赖诊断 |
| `quantx/core/strategy/config_strategy.py` | YAML 策略编译、公式 DAG、选股调仓执行配置 | 静态 lint、策略解释增强 |
| `quantx/core/analysis` | 指标、报告、形态分析 | 新增质量评分模块的自然归属 |
| `quantx/tools/run_backtest.py` | 标准回测入口和 artifacts 写出 | 可选生成 `strategy_quality.json` |
| `quantx/production/pipeline.py` | 日常生产流水线 | 后续读取质量信息展示，不作为第一阶段改动点 |
| `quantx/production/report_renderer.py` | 日报渲染 | 后续展示质量 warning 和信号健康 |
| `quantx/server/services.py` | Web workspace 服务层 | 后续读取质量 artifact |
| `quantx/core/decision` | 决策协议、预测、执行边界 | 信号口径合约的优先落点 |
| `quantx/core/research` | 训练、数据集、模型、预测 | 后续自动因子搜索落点 |

### 4.2 第一阶段不建议修改的模块

以下模块在第一阶段不建议修改：

```text
quantx/core/engine/*
quantx/core/strategy/base.py
quantx/core/engine/executor.py
quantx/core/engine/exchange.py
qlib/*
```

原因：第一阶段目标是诊断和评分，不应该触碰撮合、账户、交易执行和 Qlib 内核。

---

## 5. 推荐能力总览

| 编号 | 能力 | 是否合并 | 最小新增文件 | 最小修改文件 | 新依赖 | 风险 |
|---|---|---|---|---|---|---|
| QX-AM-001 | QualityIssue 类型和结果 schema | 是 | `quality/types.py` | 无 | 无 | 低 |
| QX-AM-002 | 因子输出健康检查 | 是 | `quality/factor_lint.py` | 无 | 无 | 低 |
| QX-AM-003 | 策略配置静态 Lint | 是 | `quality/strategy_lint.py` | 可选 `factory.py` | 无 | 中 |
| QX-AM-004 | 策略质量评分 | 是 | `quality/scoring.py` | 可选 `run_backtest.py` | 无 | 中 |
| QX-AM-005 | 质量 artifact 写出 | 是 | 无 | `run_backtest.py` | 无 | 中 |
| QX-AM-006 | Web 质量展示 | 是 | 无 | `server/services.py`, `workspace.html` | 无 | 中 |
| QX-AM-007 | 信号口径合约 | 是 | `decision/contracts.py` 或扩展现有类型 | production/report 可选 | 无 | 中 |
| QX-AM-008 | 属性测试 | 是 | `tests/property/*` | `pyproject.toml` dev extra | `hypothesis` | 低 |
| QX-AM-009 | 数据源健康检查 | 可选 | `data/sources/health.py` | production 可选 | 无 | 中 |
| QX-AM-010 | 自动因子搜索 PoC | 后续 | `research/factor_search/*` | 无 | 无 | 高 |

---

## 6. 能力一：因子与策略 Lint

### 6.1 目标

新增一个只读质量诊断模块，用于识别以下问题：

1. 因子输出为常数或近似常数。
2. 因子输出含 NaN、Inf 或覆盖率过低。
3. 买入信号过稀疏或过密。
4. 信号长期单边偏置，疑似 beta 或市场方向暴露。
5. 公式依赖过长、重复或难以维护。
6. 策略使用未知字段或公式依赖存在循环。
7. 选股结果高度集中于少数股票或少数日期。
8. 信号日不是最新可用交易日。

该模块不负责判断策略是否盈利，也不改变策略输出。

### 6.2 推荐目录结构

```text
quantx/core/analysis/quality/
  __init__.py
  types.py
  factor_lint.py
  strategy_lint.py
```

### 6.3 数据结构需求

`types.py` 定义统一问题结构。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    issues: tuple[QualityIssue, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
```

设计要求：

1. `code` 必须稳定，供测试、Web 和报告识别。
2. `message` 给人读，允许中文。
3. `evidence` 放机器可读证据，如比例、阈值、字段名、日期范围。
4. `suggestion` 是修复建议，可为空。
5. `ok=False` 仅表示存在 `error` 级别问题；`warning` 不阻止回测。

### 6.4 因子输出健康检查

`factor_lint.py` 建议提供函数：

```python
def lint_factor_matrix(
    values: Any,
    *,
    name: str = "factor",
    finite_min_ratio: float = 0.98,
    non_constant_std_threshold: float = 1e-8,
    sparse_signal_threshold: float = 0.005,
    dense_signal_threshold: float = 0.80,
    direction_bias_threshold: float = 0.90,
) -> QualityReport:
    ...
```

输入可接受：

1. `numpy.ndarray`
2. `pandas.Series`
3. `pandas.DataFrame`

输出检查项：

| code | severity | 触发条件 | evidence |
|---|---|---|---|
| `non_finite_factor_values` | warning/error | NaN/Inf 比例超过阈值 | `finite_ratio`, `nan_count`, `inf_count` |
| `constant_factor` | warning | 标准差过低 | `std`, `unique_count` |
| `low_coverage_factor` | warning | 有效值覆盖率过低 | `coverage_ratio` |
| `sparse_signal` | info/warning | 非零或 True 比例过低 | `active_ratio` |
| `dense_signal` | warning | 非零或 True 比例过高 | `active_ratio` |
| `direction_bias` | warning | 正/负方向占比极端 | `positive_ratio`, `negative_ratio` |
| `extreme_factor_values` | info | 分布有极端值 | `p01`, `p99`, `max_abs` |

最小实现注意事项：

1. 不依赖 FactorRuntime 内部对象，只处理最终矩阵。
2. 对空输入返回 `empty_factor_values` warning。
3. 布尔信号按 `True/False` 计算 active ratio。
4. 浮点信号默认以 `abs(value) > 1e-12` 视为 active。
5. 诊断函数不应抛出非预期异常，无法分析时返回 `lint_failed` error。

### 6.5 策略配置静态 Lint

`strategy_lint.py` 建议提供函数：

```python
def lint_strategy_config(config: dict[str, Any]) -> QualityReport:
    ...
```

第一版只做静态检查，优先复用现有 `compile_strategy_config(config)` 的结果，不重新写解析器。

检查项：

| code | severity | 说明 |
|---|---|---|
| `strategy_compile_failed` | error | `compile_strategy_config` 抛错 |
| `large_formula_count` | info | 公式数量超过阈值，提示维护成本 |
| `long_dependency_chain` | warning | 公式依赖链过长，提示可读性和调试风险 |
| `unused_formula` | info | 定义但未被 selector/rebalance/execution 使用 |
| `missing_description` | info | 策略缺少说明字段 |
| `unsafe_signal_lag` | warning/error | selector lag 配置可能导致信号日/执行日混淆 |
| `wide_position_rules` | info | 仓位规则过多，提示复杂度 |

最小修改建议：

1. 第一版不改 `compile_strategy_config`。
2. `lint_strategy_config` 调用 `compile_strategy_config`，把编译错误转换为 `QualityIssue`。
3. 需要依赖图时使用 `CompiledStrategySpec.dependencies`。
4. 不直接读市场数据。

### 6.6 与 `explain_strategy` 的接入

可选接入点：

```text
quantx/core/strategy/factory.py
```

最小改法：

1. 保持 `explain_strategy` 原返回结构兼容。
2. 在返回 dict 中新增可选字段：

```json
{
  "quality": {
    "ok": true,
    "issues": [],
    "metrics": {}
  }
}
```

3. 如果 lint 失败，不影响原 explain 输出，只增加 warning：

```json
{
  "code": "strategy_lint_failed",
  "severity": "warning",
  "message": "策略质量诊断失败，但策略解释仍可用",
  "evidence": {"error_type": "..."}
}
```

### 6.7 测试需求

新增：

```text
tests/analysis/test_quality_lint.py
```

测试用例：

1. `test_lint_factor_matrix_accepts_clean_values`
2. `test_lint_factor_matrix_flags_nan_inf`
3. `test_lint_factor_matrix_flags_constant_values`
4. `test_lint_factor_matrix_flags_sparse_signal`
5. `test_lint_factor_matrix_flags_direction_bias`
6. `test_lint_strategy_config_wraps_compile_error`
7. `test_lint_strategy_config_reports_dependency_metrics`
8. `test_lint_does_not_mutate_input_config`

验收标准：

1. 所有测试通过。
2. Lint 函数对空输入、全 NaN、全常数、布尔矩阵都稳定返回。
3. 不修改任何现有策略行为。

---

## 7. 能力二：策略质量评分

### 7.1 目标

基于现有回测 artifacts 生成统一的策略质量评分，用于批量比较 generated 策略、Web 展示和后续生产筛选。

该评分不是收益承诺，也不作为第一阶段交易硬门槛。它是“风险、稳定性、可维护性、信号健康”的综合诊断。

### 7.2 推荐文件

```text
quantx/core/analysis/quality/scoring.py
```

### 7.3 最小输入

评分器优先消费 run artifacts，而不是重新跑回测。推荐输入：

```text
runs/<run_id>/
  summary.json
  daily_nav.csv 或 daily_nav.json
  trades.csv 或 trades.json
  positions.csv 或 positions.json
  selection_candidates.json
  config.yaml 或 config.json
```

缺失字段处理原则：

1. 缺少 `daily_nav`：无法计算风险收益，返回 `insufficient_nav_data` error。
2. 缺少 `trades`：换手率和交易次数组件给 0 分，并给 warning。
3. 缺少 `selection_candidates`：信号健康组件降级。
4. 缺少 config：复杂度组件降级。

### 7.4 输出文件

```text
runs/<run_id>/strategy_quality.json
```

推荐 schema：

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-15T12:00:00",
  "run_id": "...",
  "ok": true,
  "overall_score": 72.4,
  "grade": "B",
  "components": {
    "risk_adjusted_return": 18.2,
    "drawdown_control": 14.1,
    "cost_robustness": 10.0,
    "stability": 13.5,
    "turnover_quality": 8.0,
    "signal_health": 8.6
  },
  "metrics": {
    "annual_return": 0.32,
    "max_drawdown": -0.18,
    "calmar": 1.77,
    "sortino": 2.13,
    "trade_count": 186,
    "turnover": 2.4
  },
  "issues": [
    {
      "code": "weak_cost_robustness",
      "severity": "warning",
      "message": "成本压力下策略质量下降明显",
      "evidence": {"stress_cost_multiplier": 2.0},
      "suggestion": "检查换手率或提高入场阈值"
    }
  ]
}
```

### 7.5 评分组件

总分 100。第一版建议：

| 组件 | 权重 | 输入 | 说明 |
|---|---:|---|---|
| `risk_adjusted_return` | 25 | daily nav | 年化收益、Sortino、Calmar 综合 |
| `drawdown_control` | 20 | daily nav | 最大回撤、回撤恢复、尾部风险 |
| `stability` | 20 | daily nav/trades | 分年度收益一致性、前后半段一致性 |
| `turnover_quality` | 15 | trades/positions | 交易频率过低或过高都扣分 |
| `cost_robustness` | 10 | trades/summary | 成本压力估计 |
| `signal_health` | 10 | candidates/config | 信号覆盖率、过期信号、集中度 |

### 7.6 关键公式建议

#### 7.6.1 Sortino 下行波动率地板

AlphaMaster 的有价值思想是避免稀疏亏损导致 Sortino 虚高。QuantX 可以重新实现为：

```text
downside_std = std(negative_returns)
full_std = std(all_returns)
effective_downside_std = max(downside_std, full_std * 0.2, eps)
sortino = mean_return / effective_downside_std * sqrt(periods_per_year)
```

#### 7.6.2 分段一致性

将日收益分成两段或按年度分组：

```text
first_half_score = score(nav[:split])
second_half_score = score(nav[split:])
consistency_penalty = abs(first_half_score - second_half_score)
```

评分要避免只在某一年爆发、其他年份失效的策略拿高分。

#### 7.6.3 换手率质量

对 A 股日频策略，过高换手带来费用、滑点、涨跌停不可成交风险；过低换手可能是样本太少。第一版只做经验评分：

```text
trade_count < min_required        -> warning + 低分
turnover 极低且收益主要来自少数交易 -> warning
turnover 在合理区间              -> 正常
turnover 过高                    -> 扣分
```

具体阈值应允许配置，默认只用于诊断。

#### 7.6.4 成本压力测试

第一版不重跑撮合，只基于 artifacts 做近似估计：

```text
estimated_extra_cost = turnover * base_cost * (stress_multiplier - 1)
stress_return = original_return - estimated_extra_cost
```

后续可以增加 `--quality-stress-cost` 模式重跑回测。

### 7.7 API 设计

```python
def score_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Read existing run artifacts and return a JSON-serializable quality report."""


def write_strategy_quality(run_dir: Path) -> Path:
    """Score artifacts and write strategy_quality.json."""
```

要求：

1. 返回值必须 JSON 可序列化。
2. 不直接 import Web 或 production 模块。
3. 文件不存在时返回结构化 issue，不抛到上层。
4. 对 NaN/Inf 指标写 `null` 或省略，不能写非法 JSON。

### 7.8 与 `run_backtest` 的接入

可选修改：

```text
quantx/tools/run_backtest.py
```

最小接入方式：

1. 增加 CLI 参数：

```text
--quality-report / --no-quality-report
```

2. 默认建议先不开启，或在配置中开启：

```yaml
analysis:
  quality_report: true
```

3. 回测完成并写出 artifacts 后调用 `write_strategy_quality(run_dir)`。
4. 质量评分失败不应让回测失败，只写 warning 到 summary 或 logs。

### 7.9 测试需求

新增：

```text
tests/analysis/test_strategy_quality_scoring.py
```

测试用例：

1. `test_score_run_artifacts_handles_missing_nav`
2. `test_score_run_artifacts_scores_clean_run`
3. `test_score_run_artifacts_penalizes_large_drawdown`
4. `test_score_run_artifacts_penalizes_too_few_trades`
5. `test_score_run_artifacts_handles_empty_trades`
6. `test_write_strategy_quality_outputs_valid_json`
7. `test_quality_score_contains_stable_component_keys`

验收标准：

1. 对最小 run_dir fixture 能生成 `strategy_quality.json`。
2. JSON schema 稳定。
3. 缺失 optional artifact 不导致异常。
4. 所有分数在 `[0, 100]` 或组件定义范围内。

---

## 8. 能力三：信号口径合约

### 8.1 目标

AlphaMaster 的一个核心优点是训练、回测和实时使用同一套信号转换逻辑。QuantX 不应照搬其连续仓位 `tanh(factor)`，但应借鉴“信号口径显式化”的思想。

QuantX 需要明确记录每条信号的：

1. 信号是哪一天产生的。
2. 使用的数据最晚到哪一天。
3. 最早允许在哪一天执行。
4. 来源策略或模型产物是什么。
5. 预测 horizon 是多少。
6. 是否为样本外预测。
7. 是否为最新交易日信号。

### 8.2 推荐数据结构

优先落点：

```text
quantx/core/decision/contracts.py
```

如果后续发现 `decision/predictions.py` 已有类型足够接近，可以只扩展现有类型，避免新增文件。

建议结构：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalContract:
    source_id: str
    source_type: str
    signal_date: str
    data_available_through: str
    execute_not_before: str
    horizon: int | None = None
    artifact_id: str = ""
    fold_id: str = ""
    schema_id: str = ""
    is_oos: bool | None = None
    extra: dict[str, Any] | None = None
```

### 8.3 最小接入方式

第一版不修改 `Signal` dataclass，只在 artifacts 中增加字段：

```json
{
  "symbol": "SZ000001",
  "score": 0.92,
  "signal_date": "2026-07-14",
  "signal_contract": {
    "source_id": "strategy:bbi_short_long",
    "source_type": "config_strategy",
    "signal_date": "2026-07-14",
    "data_available_through": "2026-07-14",
    "execute_not_before": "2026-07-15",
    "horizon": 1,
    "artifact_id": "run:...",
    "is_oos": null
  }
}
```

### 8.4 与现有模块关系

现有 `quantx/core/decision` 已经有严格时序类型，包括 `MarketTime`、`DecisionClock`、`PredictionRecord` 等。新增信号合约时必须避免重复造大框架：

1. 如果信号来自预测，优先从 `PredictionRecord` 派生合约。
2. 如果信号来自配置策略，使用回测上下文中的 `signal_date` 和执行日生成合约。
3. 如果是生产日报信号，使用 `latest_trade_date` 和 `strategy_runtime` 的输出生成合约。

### 8.5 报告与 Web 展示

最小展示字段：

```text
信号日: 2026-07-14
数据截止: 2026-07-14
最早执行: 2026-07-15
来源: config_strategy / prediction / research
状态: 最新 / 过期 / 不可判定
```

过期判断：

```text
is_latest_signal_date = signal_contract.signal_date == latest_trade_date
```

如果信号不是最新交易日，不阻止展示，但报告应提示。

### 8.6 测试需求

新增或扩展：

```text
tests/decision/test_signal_contracts.py
tests/production/test_daily_pipeline.py
tests/server/test_services.py
```

测试用例：

1. 合约可 JSON 序列化。
2. `execute_not_before` 必须晚于 `signal_date`。
3. 过期信号能被标记但不导致报告失败。
4. 缺少合约的旧 artifact 仍能读取。

---

## 9. 能力四：数据源健康检查

### 9.1 目标

AlphaMaster 的数据源抽象中有 `available()`、`supported_timeframes()`、`preset_symbols()`，对 Web 展示和依赖降级很有用。QuantX 第一版不应重写数据读取接口，只做健康检查接口。

### 9.2 推荐落点

```text
quantx/core/data/sources/
  __init__.py
  health.py
```

如果不想新增 `sources` 目录，也可放在：

```text
quantx/core/data/health.py
```

从最小开发量看，建议使用单文件 `quantx/core/data/health.py`。

### 9.3 接口设计

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DataSourceStatus:
    name: str
    available: bool
    message: str
    latest_session: str | None = None
    coverage: dict[str, Any] | None = None


class DataSourceHealthCheck(Protocol):
    def check(self) -> DataSourceStatus: ...
```

第一版实现：

1. `check_qlib_provider(provider_uri: Path) -> DataSourceStatus`
2. `check_meta_store(meta_uri: Path) -> DataSourceStatus`
3. `check_daily_run_data(run_dir: Path) -> DataSourceStatus`

### 9.4 接入点

优先接入：

1. `DailyPipeline._run_data_stage` 中已有 `verify_provider`，可后续把结果转为统一 `DataSourceStatus`。
2. Web workspace 可展示 provider 是否可用、最新交易日、股票数量。

第一阶段不接入实际行情读取，不改变数据同步逻辑。

### 9.5 测试需求

新增：

```text
tests/core_data/test_data_health.py
```

测试用例：

1. provider 路径不存在时返回 unavailable。
2. meta sqlite 不存在时返回 unavailable。
3. 空 provider 返回 coverage warning。
4. 正常 fixture 返回 available。

---

## 10. 能力五：属性测试

### 10.1 目标

用属性测试覆盖 QuantX 的关键不变量，借鉴 AlphaMaster 的测试思路，但适配 A 股日频和当前代码结构。

### 10.2 依赖修改

修改：

```text
pyproject.toml
```

在 `dev` extra 中增加：

```toml
"hypothesis>=6.0",
```

该依赖不进入核心 dependencies。

### 10.3 推荐测试目录

```text
tests/property/
  __init__.py
  test_factor_runtime_properties.py
  test_backtest_cost_properties.py
  test_signal_timing_properties.py
  test_strategy_config_properties.py
```

### 10.4 属性一：因子运行时不扩散非有限值

目标：任意合法 OHLCV 输入，基础因子和常用公式不应产生无法控制的 NaN/Inf。

断言：

1. 输出 shape 与输入日期和股票维度一致。
2. 非有限值比例低于阈值，或被明确记录为 warmup 缺失。
3. 不抛未捕获异常。

边界输入：

1. 全等价格。
2. 零成交量。
3. 极小价格。
4. 极大价格。
5. 单股票、多股票。
6. 短窗口、长窗口。

### 10.5 属性二：成本上升不应让同一交易序列净收益变好

目标：对同一订单/成交路径，提高手续费或滑点后，净值不应更高。

断言：

```text
final_value(high_cost) <= final_value(low_cost) + eps
```

适配方式：

1. 使用简化交易序列 fixture。
2. 或调用现有 `TransactionCost` / `Account` 单元逻辑。
3. 不需要跑完整大回测。

### 10.6 属性三：信号日必须早于执行日

目标：任何 selection 中的 `signal_date` 都必须是执行日之前的有效交易日。

断言：

1. 缺失 `signal_date` 必须报错。
2. 等于执行日必须报错。
3. 晚于执行日必须报错。
4. 非交易日必须报错。
5. 前一交易日通过。

该属性可以复用现有 `BacktestEngine._validate_signal_timing` 测试思想。

### 10.7 属性四：配置策略编译稳定

目标：配置表达式依赖排序稳定，未知字段必须报错。

断言：

1. 同一个 config 多次编译 `formula_order` 相同。
2. 依赖关系不受 dict 插入顺序影响。
3. 未知字段或未知函数返回清晰错误。
4. 循环依赖报错。

### 10.8 运行性能要求

属性测试必须分层：

1. 默认 `pytest` 可以运行核心属性测试，总耗时控制在 10 秒内。
2. 重型属性测试标记 `@pytest.mark.slow`。
3. 每个 property 默认 `max_examples` 控制在 50 到 100。
4. 失败案例应能最小化并给出清晰错误。

---

## 11. 能力六：Web 工作台展示增强

### 11.1 目标

在不重写 Web 前端的前提下，展示策略质量评分和诊断 warning。

### 11.2 现有落点

```text
quantx/server/services.py
quantx/server/app.py
quantx/server/static/workspace.html
```

### 11.3 最小后端改动

在服务层读取 run artifacts 时，附带读取：

```text
strategy_quality.json
```

推荐 helper：

```python
def _read_quality_if_exists(run_dir: Path) -> dict[str, Any]:
    return _read_json_if_exists(run_dir / "strategy_quality.json", default={})
```

要求：

1. 文件不存在返回 `{}`。
2. JSON 解析失败返回 warning，不让 API 失败。
3. API 返回中新增字段 `quality`。

### 11.4 最小前端展示

展示位置：

1. 策略/回测列表：显示 `grade` 和 `overall_score`。
2. 回测详情：显示组件分和 issues。
3. issue 展示字段：`severity`、`code`、`message`、关键 evidence。

不做：

1. 不新增训练页面。
2. 不新增复杂图表。
3. 不引入前端依赖。
4. 不照搬 AlphaMaster UI。

### 11.5 测试需求

扩展：

```text
tests/server/test_services.py
```

测试用例：

1. run_dir 有 `strategy_quality.json` 时 API 返回 quality。
2. 缺少文件时 API 仍正常。
3. quality JSON 损坏时 API 不崩溃。

---

## 12. 能力七：自动因子搜索后续方案

### 12.1 结论

自动因子搜索是可借鉴方向，但不应作为第一阶段合并内容。

原因：

1. 容易引入重依赖和训练复杂度。
2. 容易制造大量过拟合 YAML。
3. 当前更缺策略筛选和质量诊断。
4. QuantX 已有 generated 策略很多，先要解决如何比较和审计。

### 12.2 如果后续实现，先做非 DL PoC

推荐目录：

```text
quantx/core/research/factor_search/
  __init__.py
  search_space.py
  sampler.py
  evaluator.py
  exporter.py
```

第一版只做随机/规则搜索：

1. 从现有 `DEFAULT_REGISTRY` 或 factor runtime operators 构建搜索空间。
2. 随机生成短公式或组合已有公式。
3. 用现有回测入口评价。
4. 用 `strategy_quality` 过滤明显不健康结果。
5. 输出标准 YAML 到 `configs/strategies/generated/`。

第一版不做：

1. 不用 Transformer。
2. 不用 RL。
3. 不用 PyTorch。
4. 不绕过现有 YAML 策略体系。

### 12.3 后续 DL 搜索边界

如果未来需要类似 AlphaMaster 的深度搜索，应放在可选依赖：

```toml
[project.optional-dependencies]
dl = [
  "torch>=2.2",
]
```

训练输出必须仍然是 QuantX 标准产物：

1. YAML strategy config。
2. ModelArtifact / PredictionRecord。
3. `strategy_quality.json`。
4. 可复现 manifest。

不得让深度训练模型直接调用交易执行层。

---

## 13. 分阶段实施计划

### 13.1 Phase 0：文档确认

目标：确认本文方案和范围。

新增文件：

```text
docs/development/21_ALPHAMASTER_REFERENCE_INTEGRATION_DESIGN.md
```

修改文件：无。

验收标准：

1. 文档覆盖推荐能力、最小改动、测试和风险。
2. 明确第一阶段只做只读诊断。
3. 明确 AGPL 风险和不复制代码原则。

### 13.2 Phase 1：Quality 类型和因子/策略 Lint

目标：实现只读诊断基础设施。

新增文件：

```text
quantx/core/analysis/quality/__init__.py
quantx/core/analysis/quality/types.py
quantx/core/analysis/quality/factor_lint.py
quantx/core/analysis/quality/strategy_lint.py
tests/analysis/test_quality_lint.py
```

修改文件：无，或可选修改 `quantx/core/strategy/factory.py` 增加 explain quality 字段。

不修改：

```text
quantx/core/engine/*
quantx/core/strategy/base.py
quantx/production/*
quantx/server/*
```

验收标准：

1. `lint_factor_matrix` 可独立使用。
2. `lint_strategy_config` 可独立使用。
3. 缺失或异常输入返回结构化 issue。
4. 现有测试不受影响。

回滚方式：删除新增 `quality` 包和测试即可。

### 13.3 Phase 2：策略质量评分和 artifact

目标：基于现有 run artifacts 生成 `strategy_quality.json`。

新增文件：

```text
quantx/core/analysis/quality/scoring.py
tests/analysis/test_strategy_quality_scoring.py
```

修改文件：

```text
quantx/tools/run_backtest.py
```

最小修改：

1. 增加可选参数或配置开关。
2. 回测完成后调用 `write_strategy_quality`。
3. 失败时写 warning，不使回测失败。

验收标准：

1. 示例 run_dir 能生成合法 JSON。
2. 空 trades、缺 candidates、缺 config 均能稳定降级。
3. 不改变原有 summary、trades、nav 文件。

回滚方式：关闭参数或移除调用，保留评分模块不影响回测。

### 13.4 Phase 3：Web 展示

目标：在 Web 工作台展示质量分和 issues。

修改文件：

```text
quantx/server/services.py
quantx/server/static/workspace.html
tests/server/test_services.py
```

验收标准：

1. 没有 `strategy_quality.json` 时 Web 行为不变。
2. 有质量文件时 API 返回质量字段。
3. 前端能展示分数和 warning。
4. 质量文件损坏时 Web 不崩溃。

回滚方式：移除 Web 读取和展示逻辑，不影响 artifact。

### 13.5 Phase 4：属性测试

目标：引入 Hypothesis 验证关键不变量。

修改文件：

```text
pyproject.toml
```

新增文件：

```text
tests/property/__init__.py
tests/property/test_factor_runtime_properties.py
tests/property/test_backtest_cost_properties.py
tests/property/test_signal_timing_properties.py
tests/property/test_strategy_config_properties.py
```

验收标准：

1. `conda activate test && pip install -e '.[dev]'` 后可运行。
2. 默认属性测试总耗时可控。
3. 新测试不依赖真实行情数据。

回滚方式：移除 `hypothesis` dev 依赖和 `tests/property`。

### 13.6 Phase 5：信号口径合约

目标：让回测、生产、报告都能解释信号来源和时序边界。

新增或修改文件：

```text
quantx/core/decision/contracts.py
tests/decision/test_signal_contracts.py
```

可选修改：

```text
quantx/production/strategy_runtime.py
quantx/production/report_renderer.py
quantx/server/services.py
```

验收标准：

1. 合约可序列化。
2. 可标记过期信号。
3. 不破坏旧 artifact。
4. 不改变订单生成。

### 13.7 Phase 6：数据源健康检查

目标：提供统一数据可用性诊断。

新增文件：

```text
quantx/core/data/health.py
tests/core_data/test_data_health.py
```

可选修改：

```text
quantx/production/data_update.py
quantx/server/services.py
```

验收标准：

1. provider 不存在时返回 unavailable。
2. provider 正常时返回 latest session 和 coverage。
3. 生产 data stage 可复用结果。

### 13.8 Phase 7：自动因子搜索 PoC

目标：在不引入 DL 的前提下验证自动公式生成价值。

新增文件：

```text
quantx/core/research/factor_search/__init__.py
quantx/core/research/factor_search/search_space.py
quantx/core/research/factor_search/sampler.py
quantx/core/research/factor_search/evaluator.py
quantx/core/research/factor_search/exporter.py
tests/research/test_factor_search.py
```

验收标准：

1. 能生成合法 YAML。
2. 能调用现有回测或 evaluator。
3. 能用 quality score 过滤明显异常策略。
4. 不引入 PyTorch。

---

## 14. 最小修改文件清单

### 14.1 第一批建议新增

```text
quantx/core/analysis/quality/__init__.py
quantx/core/analysis/quality/types.py
quantx/core/analysis/quality/factor_lint.py
quantx/core/analysis/quality/strategy_lint.py
quantx/core/analysis/quality/scoring.py
tests/analysis/test_quality_lint.py
tests/analysis/test_strategy_quality_scoring.py
```

### 14.2 第一批可选修改

```text
quantx/tools/run_backtest.py
quantx/core/strategy/factory.py
```

### 14.3 第二批建议新增

```text
tests/property/__init__.py
tests/property/test_factor_runtime_properties.py
tests/property/test_backtest_cost_properties.py
tests/property/test_signal_timing_properties.py
tests/property/test_strategy_config_properties.py
```

### 14.4 第二批可选修改

```text
pyproject.toml
quantx/server/services.py
quantx/server/static/workspace.html
```

### 14.5 不建议第一阶段修改

```text
quantx/core/engine/*
quantx/core/strategy/base.py
quantx/core/engine/executor.py
quantx/core/engine/exchange.py
qlib/*
```

---

## 15. 验收标准总表

| 阶段 | 验收项 | 必须满足 |
|---|---|---|
| Phase 1 | Lint 独立可用 | 是 |
| Phase 1 | 不改变回测结果 | 是 |
| Phase 1 | 异常输入结构化返回 | 是 |
| Phase 2 | 写出 `strategy_quality.json` | 是 |
| Phase 2 | 缺失 artifacts 降级 | 是 |
| Phase 2 | 分数范围稳定 | 是 |
| Phase 3 | Web 缺质量文件不崩溃 | 是 |
| Phase 4 | 属性测试不依赖真实数据 | 是 |
| Phase 5 | 信号合约可序列化 | 是 |
| Phase 6 | 数据源状态可诊断 | 是 |
| Phase 7 | 自动搜索不引入 DL | 是 |

全局验收：

1. `pytest` 现有测试通过。
2. 新增测试覆盖质量诊断和评分。
3. 核心依赖不增加。
4. 文档和代码均不复制 AlphaMaster 源码。
5. 所有新 artifact 都可 JSON 序列化。
6. 新功能默认不影响生产下单。

---

## 16. 风险与规避

### 16.1 AGPL 代码污染风险

风险：直接复制 AlphaMaster 源码可能触发 AGPL 义务。

规避：只借鉴思想，重新实现；代码评审时检查是否存在大段相似实现。

### 16.2 质量分被误用为收益保证

风险：使用者把 `overall_score` 当作未来盈利概率。

规避：报告中明确其是工程质量和稳健性诊断，不是收益承诺；第一阶段不作为生产阻断条件。

### 16.3 过度复杂导致维护成本上升

风险：一次性引入过多模块，破坏现有简洁性。

规避：按 Phase 实施；第一阶段只做 `quality` 包；自动搜索放到最后。

### 16.4 属性测试运行过慢

风险：Hypothesis 生成过多案例拖慢 CI。

规避：默认 `max_examples` 较小；慢测试加 `@pytest.mark.slow`。

### 16.5 策略评分与现有指标口径冲突

风险：不同报告中 Sortino、Calmar 计算口径不一致。

规避：评分模块优先复用现有指标函数；若必须重算，在 `metrics` 中记录口径版本。

### 16.6 Web 展示读取损坏 artifact

风险：质量 JSON 损坏导致页面接口失败。

规避：Web 读取必须 try/except，损坏时返回 warning。

### 16.7 自动搜索制造过拟合策略

风险：自动生成大量过拟合 YAML 污染 generated 目录。

规避：自动搜索必须默认输出到单独子目录；必须带 quality score、训练/验证区间和 manifest。

---

## 17. 任务拆分

### QX-AM-001：Quality 类型定义

目标：新增统一质量诊断类型。

输入：无。

输出：

```text
quantx/core/analysis/quality/types.py
```

验收：

1. `QualityIssue` 和 `QualityReport` 可构造。
2. 可转换为 JSON 字典。
3. 测试覆盖 severity 和默认字段。

### QX-AM-002：因子输出健康检查

目标：实现 `lint_factor_matrix`。

输入：numpy/pandas 矩阵。

输出：`QualityReport`。

验收：

1. 能识别 NaN/Inf。
2. 能识别常数因子。
3. 能识别稀疏/过密/单边信号。
4. 不抛未捕获异常。

### QX-AM-003：策略静态 Lint

目标：实现 `lint_strategy_config`。

输入：策略 YAML 解析后的 dict。

输出：`QualityReport`。

验收：

1. 编译错误转换为 issue。
2. 能输出公式数量、依赖深度等 metrics。
3. 不修改输入 config。

### QX-AM-004：策略质量评分

目标：实现 `score_run_artifacts`。

输入：run_dir。

输出：质量 report dict。

验收：

1. 标准 run_dir 输出总分和组件分。
2. 缺少 optional 文件稳定降级。
3. 输出 JSON 合法。

### QX-AM-005：回测写出质量 artifact

目标：在 `run_backtest` 中可选写出 `strategy_quality.json`。

输入：回测配置和 run_dir。

输出：artifact 文件。

验收：

1. 开启开关时写文件。
2. 关闭开关时行为不变。
3. 写文件失败不导致回测失败。

### QX-AM-006：Web 展示质量信息

目标：Web 服务返回并展示 quality。

输入：run_dir 中的 `strategy_quality.json`。

输出：API 字段和前端展示。

验收：

1. 缺文件不影响 API。
2. 有文件时展示 grade、score、issues。
3. 损坏 JSON 不导致 500。

### QX-AM-007：属性测试

目标：引入 Hypothesis 覆盖关键不变量。

输入：随机生成的合法数据。

输出：property tests。

验收：

1. 默认测试耗时可控。
2. 不依赖真实数据。
3. 失败信息可定位。

### QX-AM-008：信号口径合约

目标：新增 `SignalContract` 并接入 artifacts。

输入：策略信号或预测记录。

输出：可序列化合约。

验收：

1. 合约时序字段完整。
2. 可标记过期信号。
3. 旧 artifact 兼容。

### QX-AM-009：数据源健康检查

目标：统一 provider/meta 状态检查。

输入：provider path、meta uri。

输出：`DataSourceStatus`。

验收：

1. 不存在路径返回 unavailable。
2. 正常路径返回 available 和 latest session。
3. 可被生产和 Web 复用。

### QX-AM-010：自动因子搜索 PoC

目标：非 DL 随机公式搜索。

输入：搜索空间、回测区间、股票池。

输出：候选 YAML 和质量评分。

验收：

1. 不引入 PyTorch。
2. 输出标准策略 YAML。
3. 通过 quality score 过滤异常策略。

---

## 18. 推荐首批落地顺序

建议首批只做四个任务：

```text
1. QX-AM-001 Quality 类型定义
2. QX-AM-002 因子输出健康检查
3. QX-AM-003 策略静态 Lint
4. QX-AM-004 策略质量评分
```

这四个任务满足：

1. 不改撮合和生产执行。
2. 不引入新核心依赖。
3. 能直接服务当前大量 generated 策略筛选。
4. 后续 Web、日报、自动搜索都可复用。

首批完成后，再决定是否接入 `run_backtest` 自动写 artifact 和 Web 展示。

---

## 19. 与现有文档关系

本文与以下文档互补：

| 文档 | 关系 |
|---|---|
| `08_TESTING_PLAN.md` | 本文补充属性测试和质量诊断测试 |
| `10_CONFIG_DRIVEN_STRATEGY_DESIGN.md` | 本文的策略静态 Lint 依赖配置化策略编译结果 |
| `11_VISUALIZATION_AGENT_SYSTEM.md` | 本文的 Web 展示增强可接入现有工作台 |
| `15_SCHEDULED_DAILY_PRODUCTION_DESIGN.md` | 本文的信号合约和数据健康可补充生产日报 |
| `19_UNIFIED_METHOD_RESEARCH_BACKTEST_DESIGN.md` | 本文的自动搜索和信号合约应服从统一研究/决策边界 |
| `20_MARKET_WORLD_MODEL_DESIGN.md` | 本文不触碰重型世界模型，质量评分可作为其后续评估输入 |

---

## 20. 最终建议

从当前 QuantX 的成熟度和未提交改动规模看，最优路径不是引入 AlphaMaster 的训练系统，而是先补齐“策略质量基础设施”：

```text
quality types
  -> factor lint
  -> strategy lint
  -> artifact scoring
  -> Web/report 展示
  -> 属性测试
  -> 自动搜索
```

这条路径开发量小、耦合低、风险可控，并且能立刻提升当前 generated 策略的筛选效率和生产可解释性。

