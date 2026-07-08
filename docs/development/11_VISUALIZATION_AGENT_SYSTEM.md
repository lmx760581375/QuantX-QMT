# QuantX Visualization And Agent System

本文档定义 QuantX 可视化层、在线回测工作台、数据任务状态、报告产物、以及 agent 托管策略优化的落地方案。本文档同时作为当前实现的工程契约。

## 1. 目标

QuantX 需要从命令行回测扩展为一个可交互研究系统：

```text
YAML strategy config
  -> backtest task
  -> standard run artifacts
  -> metrics/report service
  -> web workspace
  -> agent strategy loop
```

第一版目标不是做复杂多用户平台，而是本机研究工作台：

1. 可以选择 `configs/strategies/*.yaml` 并在线回测。
2. 可以编辑、校验、保存 config。
3. 可以打开历史 run 的收益曲线、回撤、交易、持仓和指标。
4. 可以触发数据状态检查，后续扩展为补齐/更新任务。
5. 可以让 agent 通过稳定 API 编写 config、跑回测、读取结果、迭代优化。

## 2. 标准 Run Artifacts

所有 CLI、Web、Agent 都必须共享同一套 run 目录格式。回测完成后生成：

```text
runs/
  20260705_165616_shuijiao_legacy_formula/
    config.yaml
    summary.json
    metrics.json
    daily_nav.json
    trades.json
    positions.json
    closed_positions.json
    explain.json
    logs.txt
```

字段说明：

```text
summary.json
  run_id、config_name、日期、股票数、收益、交易数、耗时、artifact 路径。

daily_nav.json
  date、cash、total_value、daily_return、cumulative_return、drawdown、position_count。

trades.json
  date、symbol、action、price、quantity、trade_value、total_cost、reject_reason。

positions.json
  每日持仓快照：date、symbol、quantity、avg_cost、market_value、weight、holding_days。

closed_positions.json
  FIFO 撮合买卖后的持仓周期：entry_date、exit_date、holding_days、return、pnl。

metrics.json
  total_return、annual_return、annual_volatility、sharpe、sortino、calmar、max_drawdown、win_rate、profit_factor、avg_holding_days 等。

explain.json
  config snapshot、公式列表、selector/rebalance/execution 参数、engine 参数、signal_errors。
```

落盘代码位置：

```text
quantx/core/analysis/reporting.py
  build_run_report(result, config, symbols, run_id)
  write_run_artifacts(report, output_dir, config_path)
```

`run_backtest.py` 增加：

```bash
python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --output-dir runs \
  --json
```

## 3. 指标计算设计

指标计算只依赖 `BacktestResult` 或标准 artifacts，不反向依赖策略类。

收益序列：

```text
daily_return = total_value.pct_change()
drawdown = total_value / total_value.cummax() - 1
annual_return = (1 + total_return) ** (365 / calendar_days) - 1
annual_volatility = std(daily_return) * sqrt(252)
sharpe = annual_return / annual_volatility
sortino = annual_return / downside_volatility
calmar = annual_return / abs(max_drawdown)
```

交易分析：

```text
TradeMatcher
  - 输入 trades.json
  - 按 symbol FIFO 匹配 buy/sell
  - 输出 closed_positions.json
```

第一版只做 FIFO，满足多数等权全卖策略。后续可扩展为多批次成本法、税费归因、交易原因归因。

## 4. 服务端设计

第一版使用 FastAPI，代码结构：

```text
quantx/server/
  app.py
  services.py
  static/workspace.html
```

API：

```text
GET  /api/configs
GET  /api/configs/{name}
PUT  /api/configs/{name}
POST /api/configs/validate

POST /api/backtests
GET  /api/backtests/{run_id}
GET  /api/backtests/{run_id}/logs

GET  /api/reports
GET  /api/reports/{run_id}
GET  /api/reports/{run_id}/daily-nav
GET  /api/reports/{run_id}/trades
GET  /api/reports/{run_id}/positions

GET  /api/data/status
POST /api/data/health-check
```

任务执行：

```text
TaskManager
  memory dict 保存 running/success/failed 状态
  ThreadPoolExecutor 后台跑 backtest
  每个任务写 logs.txt
  后续可替换为 Redis/Celery
```

## 5. Web 工作台

第一版不引入前端构建，直接提供静态 HTML：

```text
GET /
  返回 quantx/server/static/workspace.html
```

页面模块：

1. Strategy Configs：选择已有 YAML。
2. Config Editor：在线编辑、保存、校验。
3. Backtest Runner：启动回测、查看状态和日志。
4. Report Viewer：指标卡、收益曲线、回撤曲线、交易表、持仓表。
5. Data Status：查看 provider_uri、calendar 范围、instrument 数量、健康检查状态。

图表第一版使用浏览器 Canvas，无需 npm build。后续可替换为 React + ECharts。

## 6. Agent Skill 设计

Skill 名称：`quantx-strategy-lab`。

Agent 只通过 API 和 config 文件托管系统，不默认修改核心引擎代码。

允许操作：

```text
list_configs
read_config
write_config
validate_config
run_backtest
read_report
compare_runs
create_experiment
update_leaderboard
```

闭环流程：

```text
1. 读取 baseline config 和历史 report。
2. 生成候选 config，保存在 configs/strategies/generated/。
3. validate config。
4. 先 symbol_limit smoke run。
5. 通过后全市场 run。
6. 读取 metrics.json，按目标函数打分。
7. 保存 experiments/<name>/leaderboard.json。
8. 生成下一轮候选。
```

目标函数示例：

```text
score = sharpe * 1.0 + calmar * 0.5 + annual_return * 0.5 - max(0, abs(max_drawdown) - 0.35) * 3.0
```

Agent 必须遵守：

1. 不允许使用未来函数。
2. 不允许比较不同数据区间的结果，除非显式声明。
3. 每个 run 必须保存 config snapshot。
4. 优化报告必须说明收益提升来自收益、回撤、交易次数还是持仓变化。

## 7. 当前实现范围

本轮实现第一版闭环：

1. 标准 run artifacts。
2. 指标和 closed positions。
3. `run_backtest --output-dir`。
4. FastAPI configs/backtests/reports/data status API。
5. 内置静态 Web 工作台。
6. 测试：reporting 单测、API smoke test、shuijiao YAML 产物生成。

### 7.1 已落地代码路径

```text
quantx/tools/run_backtest.py
  load_config()
  dry_run_config()
  run_config()
  run_config_with_artifacts()

quantx/core/strategy/config_strategy.py
  compile_strategy_config()
  explain_strategy_config()
  FormulaSelector
  EqualWeightRebalance
  RuleExecution

quantx/core/analysis/reporting.py
  build_run_report()
  write_run_artifacts()
  load_run_artifacts()
  compute_metrics()
  match_closed_positions()

quantx/server/app.py
quantx/server/services.py
quantx/server/static/workspace.html
```

`run_backtest.py` 是统一入口，Web 和 agent 也应调用这个入口或其服务封装，不再新增每个策略专属的 runner。

### 7.2 YAML 编译和 explain

`compile_strategy_config(config)` 是策略配置的启动前校验层，必须满足：

1. `fields` 中的 alias 必须是合法 identifier，source 统一成 Qlib 字段格式，例如 `$close`。
2. 内置行情字段 `open/high/low/close/volume/vwap/change` 和 `fields` alias 同时可见。
3. `factors` 和 `signals` 必须是完整公式 DAG，未定义变量和循环依赖启动前报错。
4. 公式 AST 拒绝 `&`、`|`、`~`，使用 `and/or/not` 或 `And/Or/Not()`。
5. `selector.mode=precomputed` 不允许引用 `cash/pnl_pct/holding_days` 等账户或持仓状态变量。
6. `execution.sell_rules` 可以引用公式、行情字段和状态变量，但未知变量启动前报错。

`explain_strategy_config(config)` 输出给 CLI、Web validate 和 artifact `explain.json`，包括：

```json
{
  "name": "shuijiao_legacy_formula",
  "fields": {"close": "$close"},
  "formula_count": 22,
  "formula_order": ["h1", "l1", "buy_signal", "sell_signal"],
  "dependencies": {"buy_signal": ["bb1", "close", "midline"]},
  "selector": {"where": "buy_signal", "lag": 1},
  "rebalance": {"type": "equal_weight", "max_positions": 10},
  "execution": {"deal_price": "close", "sell_rules": []}
}
```

### 7.3 CLI 使用契约

编译检查，不跑交易：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --dry-run \
  --json
```

运行回测，只输出 summary：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --json
```

运行回测并写入标准产物：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --output-dir runs \
  --run-id shuijiao_legacy_latest \
  --json
```

### 7.4 Web/API 使用契约

启动：

```bash
conda run -n test python -m quantx.server.app
```

浏览器打开 `http://127.0.0.1:8000`。第一版静态工作台提供：

1. 读取 `configs/strategies/*.yaml`。
2. 在线编辑、保存和校验 YAML。
3. 发起后台回测任务。
4. 读取 `runs/<run_id>` 下的报告。
5. 展示 summary/metrics、净值、回撤、交易表和数据状态。

Agent 或脚本使用 HTTP API：

```text
GET  /api/configs
GET  /api/configs/{config_path}
PUT  /api/configs/{config_path}
POST /api/configs/validate
POST /api/backtests
GET  /api/backtests/{task_id}
GET  /api/backtests/{task_id}/logs
GET  /api/reports
GET  /api/reports/{run_id}
GET  /api/reports/{run_id}/{artifact}
GET  /api/data/status
POST /api/data/health-check
```

### 7.5 闭环验证命令

本轮闭环必须至少跑：

```bash
conda run -n test python -m pytest \
  tests/factor_runtime/test_runtime.py \
  tests/factor_runtime/test_panel_aliases.py \
  tests/core_data/test_converter_qlib_format.py \
  tests/strategy/test_config_strategy.py \
  tests/tools/test_run_backtest_config.py \
  tests/analysis/test_reporting.py \
  tests/server/test_app.py \
  -q

conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --dry-run \
  --json

conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --output-dir /tmp/quantx_runs_verify \
  --run-id shuijiao_yaml_verify \
  --json
```

最后一条应复现当前 qlib 严格旧源码口径的 `shuijiao` YAML 回测收益，目标值约 `total_return=0.6898`。如果该收益偏离，应优先检查公式 DAG、lag、涨停过滤、成本和成交价配置，而不是直接修改策略收益。

后续增强：

1. React/ECharts 前端。
2. SSE/WebSocket 实时日志。
3. 数据补齐任务真正执行 `sync_full.py` 和 converter rebuild。
4. Agent skill 插件化安装。
5. Walk-forward/多窗口防过拟合评估。
