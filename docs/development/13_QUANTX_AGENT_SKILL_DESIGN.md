# QuantX Agent Skill 设计方案

> 版本: v0.2.0
> 日期: 2026-07-05
> 状态: `quantx-backtest` Skill 源目录和安装版已初步落地；`agent_context` JSON CLI 已初步落地；Metrics Registry、实验循环、run 对比和更强信号诊断接口仍需继续演进。
> 目标: 把 QuantX 包装成 Agent 可以安装、理解、调用、验证和扩展的量化研究 Skill，使 Agent 能通过 YAML config 自主执行回测、读取收益和交割单、分析差异、扩展指标，并把结论稳定反馈给用户。

---

## 1. 目标

QuantX 当前已经具备 Qlib 数据层、YAML/config 策略入口、`run_backtest`、`FactorRuntime`、结构化 run artifacts、股票名称/行业 MetaStore、本地 Web workspace 和 `agent_context` 初版。Skill 的目标不是复制这些逻辑，而是给 Agent 一个稳定的“操作手册 + 工具入口 + 参考资料索引”。

安装 Skill 后，Agent 应该能完成：

- 编写、修改、校验策略 YAML config。
- 通过 config 执行 dry-run 和完整回测。
- 获取总收益率、年化收益、最大回撤、夏普、交易次数、拒单、交割单和个股交易明细。
- 比较多个 run，分析策略差异和回测口径差异。
- 判断收益差异是 QuantX bug、数据口径差异、策略逻辑差异，还是旧系统 baseline 问题。
- 增加分析指标，例如 Sortino、Calmar、换手率、行业暴露、最大连续亏损。
- 查询股票名称、行业和行业相关结果。
- 启停可视化服务，并通过 API 验证展示数据。

最终体验应该接近 CodeGraph 这类专用 Skill：Agent 不需要每次从零扫描项目，而是先加载 Skill，再按固定路由读取少量 reference，通过稳定 JSON 接口拿到事实，然后基于事实行动。

---

## 2. 边界原则

### 2.1 Skill 只做 Agent 编排

- QuantX 负责数据访问、公式计算、选股、调仓、撮合、报告和指标计算。
- Skill 负责告诉 Agent 如何找到项目、读哪些 reference、运行哪些命令、如何解释输出、如何安全修改文件。
- Skill scripts 只做轻量探针或 fallback，不重新实现回测、指标或数据读取主逻辑。

### 2.2 Agent 优先调用结构化接口

Skill 内所有关键工作流都应该优先调用：

```bash
conda run -n test python -m quantx.tools.agent_context <subcommand> ...
```

只有当 `agent_context` 尚未覆盖某个能力时，才 fallback 到：

```bash
conda run -n test python -m quantx.tools.run_backtest ... --json
python tools/codex_skills/quantx-backtest/scripts/<probe>.py ...
```

### 2.3 Config 是策略资产

Agent 生成或修改策略时，核心资产是 YAML config，而不是专门 Python 脚本。Python 只提供通用原子能力，例如公式算子、选择器、调仓器、执行器、指标函数。

Skill 必须约束 Agent：不要把策略逻辑藏进 `ShuijiaoBuy(...)` 这类策略专属黑盒 operator。config 需要显式描述 buy/sell signal、selector、rebalance、execution。

### 2.4 所有长任务都要可诊断

每次完整回测至少要留下：

- config 路径和 run id。
- 输入时间范围、股票池、数据 provider。
- summary、metrics、trades、daily_nav、closed_positions。
- warnings、signal_errors、reject_summary。
- Agent 汇报时用到的关键数值来源。

---

## 3. 整体架构

```text
用户请求
  -> Codex 根据 skill metadata 触发 quantx-backtest
  -> 读取 SKILL.md
  -> 根据任务类型读取 references 中的一小部分文件
  -> 调用 agent_context / run_backtest / scripts
  -> 读取 JSON 输出和 run artifacts
  -> 必要时修改 YAML / Python / docs / tests
  -> 重新 dry-run / full run / metrics-compute
  -> 向用户汇报命令、run id、收益、风险、差异和下一步
```

推荐 Skill 源目录放在 repo 内，安装目录放在 Codex home：

```text
QuantX repo source:
  tools/codex_skills/quantx-backtest/

Installed skill:
  ~/.codex/skills/quantx-backtest/
```

这样 Skill 可以随项目一起 code review、测试和提交，同时安装版可以被 Agent 自动发现。

---

## 4. Skill 目录设计

推荐目录：

```text
tools/codex_skills/quantx-backtest/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── project-layout.md
│   ├── commands.md
│   ├── config-schema.md
│   ├── factor-language.md
│   ├── industry-meta.md
│   ├── report-artifacts.md
│   ├── diagnostics.md
│   ├── metrics-extension.md
│   ├── server-workspace.md
│   └── safety-rules.md
└── scripts/
    ├── quantx_skill_probe.py
    ├── latest_run.py
    ├── inspect_report.py
    ├── check_meta.py
    ├── validate_config.py
    └── restart_server.sh
```

目录约束：

- `SKILL.md` 保持短，只放项目默认值、任务路由和硬性规则。
- `references/` 放按需加载的详细说明。
- `scripts/` 放稳定、可执行、低自由度的小工具。
- 不在 Skill 里放 README、变更日志、长篇过程记录，避免污染 Agent 上下文。
- 源目录和安装目录要能通过验证脚本检查一致性。

---

## 5. SKILL.md 设计

### 5.1 Frontmatter

Skill frontmatter 只保留标准字段：

```yaml
---
name: quantx-backtest
description: Use when working with the QuantX project to write, validate, or modify YAML strategy configs; run dry-run or full backtests; inspect returns, metrics, trades, reports, and per-symbol K-line trade details; diagnose QuantX strategy/result differences such as Shuijiao alignment; update local metadata; operate the QuantX web workspace; or add agent-facing analysis metrics.
---
```

`description` 必须包含所有触发场景，因为 Codex 只有 metadata 常驻上下文。不在 frontmatter 里放非标准字段，避免验证器或未来平台不兼容。

### 5.2 Body

`SKILL.md` 正文建议结构：

```markdown
# QuantX Backtest

## Defaults

- Repo root: `/home/users/mingxiao.li/git/quantization/quantx`
- Python env: `conda run -n test python`
- Direct Python: `/home/users/mingxiao.li/anaconda3/envs/test/bin/python`
- Company pip mirror: `https://pypi.hobot.cc/simple`
- Default Qlib provider: `data/qlib_data_fixed`
- Default runs dir: `runs/`
- Metadata snapshots: `data/meta/snapshots/security_master.csv` and `data/meta/snapshots/industry_membership.csv`
- Web workspace URL: `http://127.0.0.1:8000`

## Task Routing

- Strategy YAML work: read `references/config-schema.md` and `references/factor-language.md`.
- Backtest commands: read `references/commands.md`.
- Report/trade analysis: read `references/report-artifacts.md`.
- Result mismatch or suspected bug: read `references/diagnostics.md`.
- Stock names, industries, or industry factors: read `references/industry-meta.md`.
- New metrics: read `references/metrics-extension.md`.
- Web workspace: read `references/server-workspace.md`.
- Data-changing, destructive, or long-running work: read `references/safety-rules.md`.

## Standard Workflow

1. `cd /home/users/mingxiao.li/git/quantization/quantx`.
2. Read only the matching reference files.
3. Prefer `quantx.tools.agent_context` JSON commands.
4. Dry-run generated or materially edited configs before full backtest.
5. Report exact config path, run id, total return, max drawdown, trade count, and material warnings.
6. Do not delete data or runs, reset git state, or overwrite user configs unless explicitly requested.
```

---

## 6. References 设计

### 6.1 `project-layout.md`

用途：让 Agent 快速知道项目结构和关键入口。

必须记录：

```text
configs/strategies/                 # YAML 策略配置
configs/strategies/generated/       # Agent 生成策略的默认目录
data/qlib_data_fixed/               # 推荐 Qlib provider
data/meta/snapshots/                # 股票名称、行业快照 CSV
data/meta/quantx_meta.sqlite        # 查询用 MetaStore
runs/                               # 回测输出
quantx/tools/agent_context.py       # Agent JSON 统一入口
quantx/tools/run_backtest.py        # 回测 CLI
quantx/tools/update_meta.py         # 元数据导入 CLI
quantx/core/strategy/config_strategy.py
quantx/core/factor_runtime/
quantx/core/analysis/reporting.py
quantx/server/app.py                # Web workspace
```

### 6.2 `commands.md`

用途：给 Agent 稳定命令，不让它临场拼错。

必须包含：

```bash
conda run -n test python -m quantx.tools.agent_context status
conda run -n test python -m quantx.tools.agent_context data-status
conda run -n test python -m quantx.tools.agent_context validate-config --config <path> --symbol-limit 3
conda run -n test python -m quantx.tools.agent_context run --config <path> --dry-run
conda run -n test python -m quantx.tools.agent_context run --config <path> --output-dir runs
conda run -n test python -m quantx.tools.agent_context report --run-id latest
conda run -n test python -m quantx.tools.agent_context symbol --run-id latest --symbol SH600000
conda run -n test python -m quantx.tools.agent_context metrics-list
conda run -n test python -m quantx.tools.agent_context metrics-compute --run-id latest --include sharpe sortino calmar
```

也要保留 fallback：

```bash
conda run -n test python -m quantx.tools.run_backtest --config <path> --dry-run --json
conda run -n test python -m quantx.tools.run_backtest --config <path> --output-dir runs --json
```

### 6.3 `config-schema.md`

用途：Agent 生成、修改、审查 YAML 策略。

必须说明标准结构：

```yaml
name: example_strategy
version: 1

data:
  provider_uri: data/qlib_data_fixed
  universe: all_a
  start: 2021-01-04
  end: 2025-10-17
  look_back_days: 120

fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume
  vwap: $vwap

signals:
  ma20: Mean(close, 20)
  momentum20: close / Ref(close, 20) - 1
  buy_signal: close > ma20
  sell_signal: close < ma20

selector:
  where: buy_signal
  rank_by: momentum20
  ascending: false
  topk: null
  lag: 1
  wrap_first_signal: false

rebalance:
  type: equal_weight
  max_positions: 10
  cash_use_ratio: 0.99

execution:
  deal_price: close
  sell_rules:
    - sell_signal
  buy:
    skip_limit_up: true
```

必须强调：config 必须说明 `buy_signal` 和 `sell_signal` 如何生成；不允许长期依赖策略专属黑盒 operator；新策略默认写到 `configs/strategies/generated/`；任何实质修改后必须 dry-run；不使用负 `Ref` 或其他未来函数。

### 6.4 `factor-language.md`

用途：记录公式语言、可用算子和未来计划算子。

必须包含：

- 时间序列算子：`Ref`、`Mean`、`EMA`、`SMA_TDX`、`Max`、`Min`、`Sum`、`Std`。
- 横截面算子：`CSMean`、`CSCount`、`CSRank`、`CSPctRank`。
- 逻辑算子：`and/or/not`、`And/Or/Not`、`Cross`、`Filter`。
- 数值算子：`Maximum`、`Minimum`、`Abs`、`Where`。
- 缺失算子标注 planned，不要让 Agent 假装可运行。

Agent 写公式的规则：优先使用已有通用算子；复杂公式拆成中间变量；如果需要新算子，先查 runtime，再提出实现任务；公式失败时先定位未定义 alias、类型错误、lookback 不足、未来数据问题。

### 6.5 `industry-meta.md`

用途：股票名称、行业、行业因子和行业交叉选股。

必须记录当前事实：

- 在线行业/名称接口不稳定，不作为默认路径。
- 默认使用 repo 内 committed metadata snapshots。
- 行业采用 latest static snapshot，不做历史 as-of。
- 部分股票可能没有行业或名称，Agent 需要汇报缺失比例。

未来 config 目标写法：

```yaml
meta:
  industry:
    mode: latest_static

groups:
  industry:
    by: meta.industry_name

group_factors:
  industry_momentum20:
    group: industry
    expr: GroupMean(close / Ref(close, 20) - 1)

selector:
  pipeline:
    - where: close > Mean(close, 20)
    - group_rank:
        group: industry
        factor: industry_momentum20
        top: 5
    - rank:
        factor: close / Ref(close, 20) - 1
        top: 10
```

如果当前 schema 尚未支持 `groups/group_factors/selector.pipeline`，Skill 必须提醒 Agent：这只能作为设计方案或待实现 config，不可声称已经能运行。

### 6.6 `report-artifacts.md`

用途：Agent 读回测结果。

必须说明 artifacts：

```text
summary.json
  name, run_id, start_date, end_date, initial_cash, final_value, total_return, symbols

metrics.json
  total_return, annual_return, max_drawdown, sharpe, sortino, calmar, trade_count, total_cost

trades.json
  date, symbol, action, price, quantity, trade_value, total_cost, reject_reason

daily_nav.json
  date, cash, market_value, total_value, daily_return, drawdown

closed_positions.json
  symbol, entry_date, exit_date, pnl, return, holding_days
```

Agent 汇报必须至少包含：config path、run id、total return、max drawdown、sharpe 或说明缺失、trade count、reject count、主要 reject_reason、top traded symbols，并尽量补名称和行业。

### 6.7 `diagnostics.md`

用途：收益差异、baseline 对齐和回测漏洞排查。

固定诊断顺序：

1. 数据覆盖：calendar、instrument、feature。
2. 数据口径：复权、价格字段、成交量、停牌、涨跌停。
3. 股票池：全市场、主板、科创、创业、北交、退市样本。
4. 公式和信号：每日 buy/sell 数量、lookback、首日处理。
5. 选股排序：score、topk、稳定排序、tie-break。
6. 调仓：最大持仓、等权、现金比例、卖出后是否当日买入。
7. 执行：成交价格、涨停不买、跌停不卖、手续费、滑点、100 股限制。
8. 报告：净值、收益率、回撤、年化口径。
9. baseline：确认旧系统是否也存在 bug 或口径不一致。

Shuijiao 已知基线：当前 YAML + qlib + 严格执行口径，2021-2025 全市场结果约 `total_return=0.6898`；旧的 `83%` 结果不是严格同口径，涉及第一日 wrap 和执行口径差异；不应把 `83%` 当作 QuantX 必须强行对齐目标，除非用户明确要求复现旧口径。

### 6.8 `metrics-extension.md`

用途：新增分析指标，暴露给 Agent/Web。

目标接口：

```text
quantx/core/analysis/metrics/
├── base.py
├── registry.py
├── builtin.py
└── custom.py
```

指标函数必须声明：

```python
MetricSpec(
    name="sortino",
    display_name="Sortino",
    category="risk",
    required_artifacts=("daily_nav",),
    description="Annual return divided by downside volatility.",
)
```

Agent 新增指标流程：判断公式和输入 artifact；如果 artifact 不够，先扩展 reporting；实现 metric；注册 metric；增加单元测试；对已有 run 执行 `metrics-compute`；如需展示，再更新 Web/API。

### 6.9 `server-workspace.md`

用途：启停和检查可视化服务。

必须包含：

```bash
ps -ef | rg 'quantx\.server\.app'
ss -ltnp 'sport = :8000'
bash tools/codex_skills/quantx-backtest/scripts/restart_server.sh /home/users/mingxiao.li/git/quantization/quantx
curl -s http://127.0.0.1:8000/api/reports
curl -s http://127.0.0.1:8000/api/meta/symbols/SH600000
```

规则：只 kill 明确匹配 `quantx.server.app` 的进程，不误杀其他服务。

### 6.10 `safety-rules.md`

用途：防止 Agent 做破坏性操作。

必须包含：不删除 `runs/`、`data/`、metadata snapshots，除非用户明确要求；不执行 `git reset --hard`、`git checkout --`；不覆盖用户已有 YAML；新策略默认放 `configs/strategies/generated/`；长回测前先 dry-run；探索阶段可用 `--symbol-limit`，最终结论必须跑全市场；不依赖在线元数据接口；不承诺高收益，只汇报实测结果和风险。

---

## 7. Agent Context CLI 设计

统一入口：

```bash
conda run -n test python -m quantx.tools.agent_context <subcommand> [options]
```

输出必须是 JSON。失败也要结构化：

```json
{
  "ok": false,
  "error_type": "UnknownFieldError",
  "message": "Unknown formula reference: foo_signal",
  "suggestions": ["Check signals.foo_signal or selector.where"]
}
```

已具备或应保持的子命令：

```text
status
data-status
meta
validate-config
run
latest-run
report
symbol
diagnose
metrics-list
metrics-compute
```

建议补强的子命令：

```text
compare-runs       # 比较两个 run 的收益、回撤、交易、拒单、每日净值差异
signals-summary    # 输出每日 buy/sell 数量、候选池数量、topK 变化
config-explain     # 编译 config 并输出公式 DAG、依赖、lookback、潜在未来函数风险
experiment-log     # 记录一次策略实验的 config、参数、run、结果和备注
industry-summary   # 输出行业覆盖、行业交易分布、行业收益贡献
```

完整回测结束后，`run` 应直接输出核心摘要，避免 Agent 再解析多个文件：

```json
{
  "ok": true,
  "run_id": "20260705_example_strategy",
  "run_dir": "runs/20260705_example_strategy",
  "config": "configs/strategies/generated/example_strategy.yaml",
  "elapsed_seconds": 38.2,
  "summary": {
    "initial_cash": 1000000,
    "final_value": 1689800,
    "total_return": 0.6898
  },
  "metrics": {
    "max_drawdown": -0.303,
    "sharpe": 0.92,
    "trade_count": 486
  },
  "warnings": []
}
```

---

## 8. Agent 标准工作流

### 8.1 写策略并回测

```text
1. 触发 quantx-backtest。
2. 读取 config-schema.md 和 factor-language.md。
3. 生成 YAML 到 configs/strategies/generated/<name>.yaml。
4. validate-config。
5. 修复所有校验错误。
6. run --dry-run。
7. 完整 run。
8. report。
9. metrics-compute 计算用户关心的额外指标。
10. 汇报 config、run id、收益、回撤、交易次数、拒单和主要风险。
```

### 8.2 自主策略实验循环

用于用户要求“找更好策略”“比较几个想法”“优化参数”。

```text
1. 明确样本区间、股票池、执行口径和目标指标。
2. 生成候选 config，每个 config 名称带参数摘要。
3. 先 symbol-limit smoke test，剔除明显无效配置。
4. 对候选策略跑全市场完整回测。
5. 生成 leaderboard: total_return, max_drawdown, sharpe, calmar, trade_count, reject_count。
6. 检查最优策略是否过度集中于少数股票/行业/交易日。
7. 对 top configs 做差异分析。
8. 汇报实测结果，不承诺未验证收益。
```

输出建议：

```text
rank | config | run_id | total_return | max_drawdown | sharpe | trade_count | notes
```

### 8.3 收益差异诊断

```text
1. 读取 diagnostics.md 和 report-artifacts.md。
2. 获取目标 run 和 baseline run。
3. 比较 summary、metrics、daily_nav、trades。
4. 检查 data-status。
5. 检查 reject_reason。
6. 检查信号数量、首日交易、涨停买入、跌停卖出、排序和调仓。
7. 如果是 QuantX bug，修复并加测试。
8. 如果是口径差异，写清楚差异来源。
9. 如果旧系统可能有 bug，记录证据并抛给用户确认。
```

### 8.4 新增指标

```text
1. 读取 metrics-extension.md。
2. 确认指标公式和 required_artifacts。
3. 如果 artifact 不够，先扩展 reporting。
4. 实现 metric 并注册。
5. 增加单元测试。
6. 对最新 run 或指定 run 执行 metrics-compute。
7. 如需要，更新 Web/API 展示。
8. 汇报公式、输入、测试和计算结果。
```

### 8.5 查看单只股票交割和 K 线

```text
1. 调用 report 找 run artifacts。
2. 调用 symbol --run-id ... --symbol ...。
3. 调用 meta 补股票名称和行业。
4. 如果用户要看图，启动或重启 Web workspace。
5. 验证 K 线页面有股票代码、名称、行业、价格坐标和交易点。
```

---

## 9. 指标扩展设计

当前 `agent_context` 已经有 `metrics-list` 和 `metrics-compute` 初版，但长期不应该把指标硬编码在一个数组和一个 reporting 函数里。建议演进为 Registry：

```text
quantx/core/analysis/metrics/
├── __init__.py
├── base.py
├── registry.py
├── builtin.py
└── custom.py
```

核心类型：

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    category: str
    required_artifacts: tuple[str, ...]
    description: str = ""

@dataclass
class MetricContext:
    summary: Dict[str, Any]
    daily_nav: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]

class MetricFn(Protocol):
    spec: MetricSpec

    def __call__(self, context: MetricContext) -> Any:
        ...
```

注册规则：metric name 唯一；输出 JSON serializable；不修改 artifact 和 backtest state；缺失 artifact 时返回结构化错误；每个新增 metric 至少一个单元测试。

优先内置指标：

```text
total_return, annual_return, max_drawdown, sharpe, sortino, calmar,
annual_volatility, downside_volatility, win_rate, profit_factor,
avg_holding_days, turnover, exposure, industry_exposure,
max_consecutive_loss, largest_win, largest_loss
```

Web workspace 后续应读取 `MetricSpec` 动态展示，而不是在前端硬编码指标列表。

---

## 10. 安装、同步和验证

### 10.1 安装命令

建议保留项目内安装入口：

```bash
conda run -n test python -m quantx.tools.install_skill \
  --skill tools/codex_skills/quantx-backtest \
  --target ~/.codex/skills/quantx-backtest
```

如果安装器支持默认参数，可简化为：

```bash
conda run -n test python -m quantx.tools.install_skill
```

### 10.2 Skill 验证

源目录和安装目录都要验证：

```bash
python /home/users/mingxiao.li/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  tools/codex_skills/quantx-backtest

python /home/users/mingxiao.li/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  ~/.codex/skills/quantx-backtest
```

如果系统路径不同，应以实际 `skill-creator` 路径为准。

### 10.3 Agent 接口验证

```bash
conda run -n test python -m quantx.tools.agent_context status
conda run -n test python -m quantx.tools.agent_context data-status
conda run -n test python -m quantx.tools.agent_context meta --symbols SH600000 SH600137
conda run -n test python -m quantx.tools.agent_context validate-config --config configs/strategies/shuijiao_legacy.yaml --symbol-limit 3
conda run -n test python -m quantx.tools.agent_context run --config configs/strategies/shuijiao_legacy.yaml --dry-run
conda run -n test python -m quantx.tools.agent_context report --run-id latest
conda run -n test python -m quantx.tools.agent_context metrics-compute --run-id latest --include sharpe sortino calmar
```

### 10.4 端到端验收任务

Skill 可用性至少要通过这些任务：

1. Agent 生成一个均线突破策略 YAML，dry-run 通过，再跑 `symbol-limit=100` 小回测。
2. Agent 跑 `shuijiao_legacy.yaml`，汇报 run id、收益、回撤、交易次数和拒单。
3. Agent 对最新 run 查询交易次数最多的 5 只股票，补名称和行业。
4. Agent 新增一个简单指标，写测试，对已有 run 计算结果。
5. Agent 重启 Web workspace，并验证 `/api/reports` 和 `/api/meta/symbols/SH600000`。

---

## 11. 分阶段落地路线

### Phase 0: Skill 源目录和说明型 Skill

状态：已初步完成。

产物：`tools/codex_skills/quantx-backtest/SKILL.md`、`references/*.md`、`scripts/*.py`、`agents/openai.yaml`。

### Phase 1: Agent Context CLI

状态：已初步完成。

已有能力：

```text
status, data-status, meta, validate-config, run, latest-run,
report, symbol, diagnose, metrics-list, metrics-compute
```

下一步：统一所有错误输出格式；`run` 输出 elapsed seconds、核心 metrics 和 warnings；`validate-config` 输出公式 DAG、依赖和 lookback。

### Phase 2: Metrics Registry

状态：待实现。

目标：把 `compute_metrics` 拆成 registry；`metrics-list` 从 registry 动态生成；`metrics-compute` 可选择 include/exclude；Web/API 从 `MetricSpec` 渲染。

### Phase 3: 实验和对比接口

状态：待实现。

目标：`compare-runs`、`experiment-log`、自动 leaderboard、config 与 run 的 diff 绑定。

### Phase 4: 信号和行业诊断接口

状态：待实现。

目标：`signals-summary`、`config-explain`、`industry-summary`、支持行业 group factor 的 config schema。

### Phase 5: Agent 自主研究闭环

状态：待实现。

目标：Agent 可以基于目标区间和目标指标批量生成 config，自动 smoke test、完整回测、leaderboard、差异分析，并对最佳策略输出证据。

---

## 12. 风险和防偏规则

### 12.1 不要让 Skill 变成第二套 QuantX

Skill scripts 只能是 thin wrapper。任何需要长期维护的业务逻辑都应该进入 `quantx/` 包，并由测试覆盖。

### 12.2 不要让 Agent 过拟合目标收益

用户可能要求很高收益，例如 5 年 10 倍。Skill 应要求 Agent 只汇报实测结果，标注样本期、股票池、执行规则，检查交易集中度和行业集中度，不把 symbol-limit smoke test 当最终结果。

### 12.3 不要把旧 baseline 当绝对真相

旧系统可能存在 bug，也可能只是口径不同。诊断时必须把 QuantX bug、旧系统 bug、数据差异、执行差异分开记录。

### 12.4 不要依赖在线数据源

公司网络慢且外部源不稳定。股票名称和行业默认走 committed snapshots；更新源只作为明确任务处理。

### 12.5 不要破坏用户工作区

Agent 不能重置 git、删除数据、覆盖用户策略。所有生成策略默认放 `configs/strategies/generated/`。

---

## 13. 最终目标形态

用户说：

```text
帮我写一个 20 日动量 + 行业强度过滤策略，跑 2021-2025，给我收益和主要交易。
```

Agent 应该能自动：触发 `quantx-backtest`；读取 config/factor/industry references；判断行业 group factor 是否已实现；生成可运行 YAML 或明确说明待实现能力；validate-config；dry-run；full run；report + metrics-compute；补股票名称、行业、交割单和风险分析；汇报 config path、run id、收益、回撤、交易次数和主要风险。

用户说：

```text
给 QuantX 加一个 Calmar 和 Sortino 指标，然后用最新 run 算一下。
```

Agent 应该能自动：触发 Skill；读取 metrics-extension；检查指标是否已存在；缺失则实现 metric、注册、测试；对 latest run 执行 metrics-compute；汇报公式、文件、测试和结果。

用户说：

```text
这个策略为什么没有对齐 myquant baseline？
```

Agent 应该能自动：读取 diagnostics；找到 QuantX run 和 baseline 证据；比较数据、信号、排序、调仓、执行、报告口径；如果是 QuantX bug，修复并重跑；如果是口径差异或旧系统问题，给出证据并记录。

这就是 QuantX Agent Skill 的核心价值：把 QuantX 从“人手动操作的一套研究代码”升级为“Agent 可以可靠调用的一套量化研究环境”。
