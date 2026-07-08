# QuantX 定时任务与每日策略通知设计方案

> 版本: v0.1.0  
> 日期: 2026-07-05  
> 状态: 设计方案，尚未实现代码  
> 目标: 为 QuantX 增加一个稳定的日度生产层，定时完成数据拉取、策略最新信号/持仓/交易更新，并把每个策略当天的选股、仓位、买卖信号整理成中文文本，通过 QQ 邮箱发给固定用户。

## 1. 背景与目标

QuantX 当前已经具备：

1. Qlib 数据层和本地 `data/qlib_data_fixed` provider。
2. YAML/config 驱动策略。
3. 回测 runner 和标准 run artifacts。
4. 可视化 workspace。
5. 结构化 report、trades、positions、closed_positions。

下一步需要一个“每日生产系统”：

```text
定时触发
  -> 更新行情数据
  -> 对一批启用的策略计算今日信号
  -> 更新策略的最新持仓状态和建议交易
  -> 生成中文日报
  -> 通过 QQ 邮箱发送给固定收件人
  -> 留下可追溯的 daily artifacts 和日志
```

这里的重点不是再跑一遍多年历史回测，而是回答每日交易前后最关心的问题：

```text
今天有哪些策略选中了哪些股票？
当前策略持仓是什么？
哪些股票需要买入？哪些股票需要卖出？为什么？
哪些信号只是候选但没有成交建议？为什么？
数据是否更新成功？策略是否有报错？邮件是否发送成功？
```

## 2. 核心原则

### 2.1 日度生产层与历史回测层分离

历史回测的职责是验证策略长期收益，输出 `runs/<run_id>/`。日度生产层的职责是每日生成“当前策略状态和交易建议”，输出 `daily_runs/<trade_date>/`。

不能把日度生产层简单做成“每天重新跑一遍多年回测然后看最后一天”，原因是：

1. 多年回测慢，不适合每天定时稳定执行。
2. 生产日报需要更强的幂等、重试、邮件状态和人工可读解释。
3. 生产系统需要明确区分“已有真实持仓”和“策略模拟持仓”。
4. 回测 run 是研究产物，日度 run 是运营产物，二者生命周期不同。

### 2.2 策略仍然由 YAML config 描述

日度任务不应该要求为每个策略写专门 Python 脚本。定时任务只接收策略清单，例如：

```yaml
enabled_strategies:
  - configs/strategies/generated/stocktradebyz_mainboard/bbi_short_long_hlc3_pos3_t24_tp220_sl90_strict_2020_2026_mainboard.yaml
  - configs/strategies/generated/myquant_shuijiao_best_auto_mainboard.yaml
```

每个策略的选股、调仓、买卖逻辑继续来自 config 的：

```text
fields / factors / signals / selector / rebalance / execution
```

定时层只负责编排、状态保存、报告生成和通知。

### 2.3 生产信号必须可解释

邮件里不能只发一个股票列表。每条建议至少应该说明：

```text
策略名称
策略 description
股票代码、名称、行业
动作: BUY / SELL / HOLD / WATCH / REJECTED
触发原因: buy_signal、sell_rule、止盈、止损、时间止损、涨停不可买等
关键因子值: score、收盘价、成交价代理、pnl_pct、holding_days
目标仓位和预计数量
风险提示: 涨停、停牌、数据缺失、资金不足、非交易日等
```

### 2.4 幂等优先

同一个 `trade_date + strategy_name + stage` 重复执行时，不能重复发多封邮件或覆盖重要日志。建议使用 `daily_run_id` 做幂等键：

```text
daily_run_id = YYYYMMDD_<profile_name>
```

邮件发送也要有幂等记录：

```text
daily_runs/20260705/default/mail_status.json
  sent: true
  sent_at: ...
  recipients: [...]
  subject: ...
  content_hash: ...
```

除非显式传 `--force-send`，否则同一内容不重复发送。

## 3. 总体架构

建议新增四层：

```text
Scheduler
  负责定时触发、锁、重试、日志。

DailyPipeline
  负责按阶段执行: 数据更新 -> 策略日更 -> 报告生成 -> 邮件发送。

DailyStrategyRuntime
  负责基于最新数据和策略 config 计算今日选股、当前模拟持仓、建议订单。

NotificationService
  负责渲染中文文本/HTML，调用 QQ SMTP 发邮件，记录发送状态。
```

推荐目录结构：

```text
quantx/production/
  scheduler.py              # 调度入口和 profile 加载
  pipeline.py               # DailyPipeline
  strategy_runtime.py       # 日度策略运行逻辑
  state_store.py            # 持仓状态、任务状态、幂等锁
  report_renderer.py        # 中文日报渲染
  mailer.py                 # SMTP/QQ 邮箱发送

quantx/tools/
  run_daily_pipeline.py     # CLI 入口

configs/production/
  daily_default.yaml        # 定时任务 profile
  mail.example.yaml         # 邮件配置示例，不放真实密码

daily_runs/
  20260705/default/
    pipeline_status.json
    data_update.json
    strategy_signals.json
    strategy_positions.json
    suggested_orders.json
    rejected_orders.json
    report.md
    report.html
    mail_status.json
    logs.txt
```

## 4. 定时任务配置设计

建议新增 `configs/production/daily_default.yaml`：

```yaml
name: default
timezone: Asia/Shanghai

schedule:
  enabled: true
  data_update_time: "17:30"
  strategy_update_time: "18:10"
  mail_send_time: "18:30"
  trading_days_only: true
  allow_manual_run: true

data:
  provider_uri: data/qlib_data_fixed
  update_mode: incremental
  verify_after_update: true
  min_calendar_end: today_or_previous_trade_day

strategies:
  include:
    - configs/strategies/generated/stocktradebyz_mainboard/bbi_short_long_hlc3_pos3_t24_tp220_sl90_strict_2020_2026_mainboard.yaml
    - configs/strategies/generated/myquant_shuijiao_best_auto_mainboard.yaml
  exclude: []
  only_positive_description: true

portfolio_state:
  mode: simulated
  state_dir: daily_state/default
  init_cash: 1000000
  cash_use_ratio_from_config: true
  carry_positions: true

notification:
  enabled: true
  channel: qq_email
  recipients:
    - example@qq.com
  subject_template: "QuantX 每日策略信号 {{ trade_date }}"
  attach_json: false
  attach_html: true
  send_empty_signal_report: true
```

说明：

1. `schedule` 只定义意图，具体由 cron/systemd/APScheduler 读取。
2. `portfolio_state.mode` 第一版建议用 `simulated`，即 QuantX 自己维护策略模拟持仓；后续可以接入真实券商持仓。
3. `only_positive_description` 表示邮件里优先展示已有正收益策略的中文说明；没有 description 的策略仍可运行，但日报中标注“未配置策略说明”。
4. 真实 QQ 邮箱授权码不能写进仓库，应通过环境变量或本机私有文件读取。

## 5. 数据拉取阶段

### 5.1 目标

每天收盘后更新 Qlib 数据，让策略日更使用最新行情。

流程：

```text
检查今天是否交易日
  -> 拉取最新行情原始数据
  -> 转换/增量写入 Qlib provider
  -> 校验 calendar、instruments、features
  -> 生成 data_update.json
```

### 5.2 当前项目约束

当前机器网络慢，在线数据源不一定稳定。因此实现时要支持两种模式：

```text
online_incremental:
  调用已有数据同步工具拉取新增数据。

local_verify_only:
  不拉取，只检查 data/qlib_data_fixed 是否已经更新到目标交易日。
```

第一版可以先把“数据拉取命令”做成可配置 command，避免定时层绑定某个不稳定数据源：

```yaml
data:
  update_command:
    - conda
    - run
    - -n
    - test
    - python
    - -m
    - quantx.tools.sync_daily_data
```

如果未来实现稳定数据同步工具，定时层只负责调用它并解析 JSON 输出。

### 5.3 校验项

数据更新后必须校验：

1. `data/qlib_data_fixed/calendars/day.txt` 最新日期是否达到目标交易日。
2. `instruments/all.txt` 是否存在且股票数大于阈值。
3. `features/*/*.bin` 是否存在。
4. 随机抽样几只股票检查 `open/high/low/close/volume/change` 是否可读。
5. 如果元数据层启用，股票名称和行业 SQLite/CSV 是否可读。

失败策略：

```text
数据更新失败但旧数据可用:
  邮件仍发送，但标题和正文标记“数据未更新到目标日”。

数据不可读:
  不运行策略，只发送错误邮件或写 error artifact。
```

## 6. 策略日更阶段

### 6.1 与回测的关系

日更阶段需要复用现有 config strategy runtime，但不应该每次全量多年回测。建议新增一个“截至某日的单日/窗口运行”能力：

```text
输入:
  strategy_config
  trade_date
  lookback window
  previous simulated positions
  previous simulated cash

输出:
  candidates
  selected_stocks
  buy_signals
  sell_signals
  suggested_orders
  updated_positions
```

第一版可以接受较保守实现：对每个策略只跑 `look_back_days + 最近持仓所需天数` 的窗口，而不是从 2020 年重新跑到今天。

如果某些策略需要完整历史状态，必须在 config 或 runtime explain 里标记：

```yaml
production:
  requires_full_replay: true
```

### 6.2 状态模型

每个策略维护独立模拟账户状态：

```text
daily_state/default/<strategy_name>/
  account.json
  positions.json
  trade_ledger.json
  last_update.json
```

`positions.json` 示例：

```json
[
  {
    "symbol": "SH600000",
    "name": "浦发银行",
    "industry_name": "银行",
    "quantity": 1000,
    "avg_cost": 8.12,
    "last_price": 8.35,
    "market_value": 8350,
    "holding_days": 12,
    "unrealized_pnl": 230,
    "unrealized_return": 0.0283
  }
]
```

注意：这里是策略模拟持仓，不等于真实券商账户。邮件必须明确写：

```text
以下为 QuantX 策略模拟持仓和交易建议，不代表真实账户持仓。
```

### 6.3 日更计算步骤

每个策略执行：

```text
1. 读取 YAML config。
2. 读取策略 description。
3. 根据 data.provider_uri、universe、look_back_days 加载截至 trade_date 的行情窗口。
4. 使用 FactorRuntime 计算 factors/signals。
5. 在 trade_date 或最近交易日生成 selector candidates。
6. 根据 rebalance 规则和现有 simulated positions 计算目标仓位。
7. 根据 execution.sell_rules 计算卖出建议。
8. 根据 buy 规则计算买入建议，过滤涨停、已持仓、资金不足、非整手等。
9. 生成 suggested_orders 和 rejected_orders。
10. 如果 profile 设置 apply_simulated_orders=true，则按策略成交价代理更新模拟持仓。
11. 输出 strategy_signals.json、strategy_positions.json、suggested_orders.json。
```

### 6.4 是否自动更新持仓

这里有一个关键选择：

```text
只生成建议，不更新持仓:
  更安全，但第二天无法知道策略模拟持仓变化。

生成建议并按 config execution 规则模拟成交:
  能维护策略连续状态，但可能和真实成交不一致。
```

建议第一版默认：

```yaml
portfolio_state:
  apply_simulated_orders: true
  execution_price_policy: config_deal_price
```

邮件里明确写“按策略配置成交价代理模拟更新”。未来如果用户真实下单后想校正，可以增加：

```bash
python -m quantx.tools.production_state adjust-position --strategy ... --symbol ...
```

## 7. 每日邮件内容设计

### 7.1 邮件结构

建议同时生成 Markdown 和 HTML。邮件正文结构：

```text
标题: QuantX 每日策略信号 2026-07-05

一、运行概览
  - 数据日期
  - 策略数量
  - 有买入建议的策略数
  - 有卖出建议的策略数
  - 当前总模拟持仓数
  - 失败策略数

二、重要提醒
  - 数据是否更新成功
  - 是否有策略报错
  - 是否有涨停不可买/停牌/价格缺失
  - 当前内容是策略模拟，不是实盘账户

三、策略逐项摘要
  1. 策略 A
     - 策略思想 description
     - 今日选股
     - 当前持仓
     - 买入建议
     - 卖出建议
     - 拒绝/过滤原因

四、全部建议订单汇总
  - BUY 表
  - SELL 表

五、附件或链接
  - 本地 run artifact 路径
  - 可视化服务 URL
```

### 7.2 单策略文本示例

```text
策略: BBIShortLong HLC3 强势版
说明: 先要求 BBI 多均线趋势向上，再寻找长周期 RSV 持续强、短周期 RSV 从高位回落后重新转强的股票...

今日信号:
  - 候选股票: 8 只
  - 入选股票: 3 只
  - 当前持仓: 2 只
  - 建议买入: 1 只
  - 建议卖出: 1 只

建议买入:
  1. SH600000 浦发银行 / 银行
     原因: buy_signal=true，score=0.83，排名 1/8，未涨停，当前未持仓
     参考价格: HLC3=8.31，目标权重=32.67%，建议数量=39000 股

建议卖出:
  1. SZ000001 平安银行 / 银行
     原因: time_stop_24d，holding_days=25，当前浮盈=8.4%
     参考价格: HLC3=12.42，建议卖出数量=12000 股

继续持仓:
  1. SH600519 贵州茅台 / 白酒
     持仓天数=6，浮盈=5.2%，未触发卖出规则

过滤/拒绝:
  1. SH600137 浪莎股份: buy_signal=true，但 skip_limit_up=true，今日涨停不可买
```

### 7.3 文本解释字段来源

| 邮件字段 | 来源 |
|---|---|
| 策略说明 | config.description |
| 股票名称 | MetaStore security_master |
| 行业 | MetaStore industry_membership |
| buy_signal/sell_signal | FactorRuntime 当日信号 |
| score/rank | selector.score |
| 持仓数量/成本/天数 | production state positions |
| 止盈止损/时间止损 | execution.sell_rules |
| 拒绝原因 | execution buy filters / exchange validation |
| 参考成交价 | execution.deal_price |

## 8. QQ 邮箱发送设计

### 8.1 QQ 邮箱 SMTP

QQ 邮箱通常使用 SMTP：

```text
host: smtp.qq.com
port: 465
ssl: true
username: <your@qq.com>
password: <QQ 邮箱 SMTP 授权码，不是登录密码>
```

真实授权码不能提交到 git。建议读取环境变量：

```bash
export QUANTX_MAIL_USERNAME="your@qq.com"
export QUANTX_MAIL_PASSWORD="授权码"
export QUANTX_MAIL_TO="user1@qq.com,user2@qq.com"
```

也可以支持本机私有文件：

```text
~/.config/quantx/mail.yaml
```

该文件必须被 `.gitignore` 忽略，文档只提交 `mail.example.yaml`。

### 8.2 邮件发送幂等

发送前计算内容 hash：

```text
content_hash = sha256(subject + html + recipients)
```

若 `mail_status.json` 已存在且 hash 相同，则跳过发送：

```json
{
  "sent": true,
  "sent_at": "2026-07-05T18:30:12+08:00",
  "recipients": ["example@qq.com"],
  "subject": "QuantX 每日策略信号 2026-07-05",
  "content_hash": "...",
  "smtp_host": "smtp.qq.com"
}
```

手动强制重发：

```bash
conda run -n test python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage mail \
  --force-send
```

### 8.3 失败处理

邮件失败不能吞掉，要写入：

```text
mail_status.json
logs.txt
pipeline_status.json
```

错误类型：

```text
auth_failed      # 授权码错误
network_timeout  # 网络问题
recipient_error  # 收件人配置错误
smtp_rejected    # SMTP 拒绝
content_error    # 渲染内容异常
```

## 9. CLI 设计

建议新增统一入口：

```bash
conda run -n test python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml
```

常用参数：

```text
--trade-date YYYY-MM-DD      指定交易日，不传则自动取最近交易日
--stage all|data|signals|report|mail
--strategy <name-or-path>    只跑某个策略
--dry-run                    不更新状态、不发邮件，只生成预览
--force-data-update          强制更新数据
--force-send                 忽略邮件幂等重发
--json                       输出结构化 JSON
```

示例：

```bash
# 完整日更
conda run -n test python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --json

# 只生成信号和报告，不发邮件
conda run -n test python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage report \
  --dry-run

# 只重发邮件
conda run -n test python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage mail \
  --force-send
```

## 10. 调度方式选择

### 10.1 推荐第一版: systemd timer 或 cron

公司开发机上第一版建议使用 systemd user timer 或 cron，原因是简单、稳定、不需要引入额外服务。

cron 示例：

```cron
30 17 * * * cd /home/users/mingxiao.li/git/quantization/quantx && /home/users/mingxiao.li/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline --profile configs/production/daily_default.yaml --stage data >> /tmp/quantx_daily_data.log 2>&1
10 18 * * 1-5 cd /home/users/mingxiao.li/git/quantization/quantx && /home/users/mingxiao.li/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline --profile configs/production/daily_default.yaml --stage signals >> /tmp/quantx_daily_signals.log 2>&1
30 18 * * 1-5 cd /home/users/mingxiao.li/git/quantization/quantx && /home/users/mingxiao.li/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline --profile configs/production/daily_default.yaml --stage mail >> /tmp/quantx_daily_mail.log 2>&1
```

注意：数据更新 cron 默认每天 17:30 触发，保证节假日前后也不会漏掉补数据机会；如果明确只想工作日运行，可以用 `quantx.tools.install_daily_data_cron --weekdays-only --install`。数据更新阶段必须流式透传 `sync_daily_data` 的 stderr，因此 `/tmp/quantx_daily_data.log` 里应该能看到每只股票一行的进度，例如 `progress 120/5461 ... symbol=SH600000 status=unchanged rows_added=0`。生产 profile 的 `data.timeout_seconds` 建议保持 `21600`，避免全市场更新在网络慢或复权刷新较多时被 1 小时默认超时杀掉。

### 10.2 后续可选: APScheduler

如果想把调度也放进 QuantX Web workspace，可以后续加入 APScheduler：

```text
quantx.server.app
  -> scheduler service
  -> 页面查看任务状态、手动触发、查看历史邮件
```

但第一版不建议直接做复杂 Web 调度，先把 CLI 和 artifacts 做稳定。

## 11. 与可视化层结合

可视化层后续新增“每日信号”页面：

```text
GET /api/daily-runs
GET /api/daily-runs/{date}/{profile}
GET /api/daily-runs/{date}/{profile}/strategies/{strategy}
POST /api/daily-runs/run
POST /api/daily-runs/send-mail
```

页面展示：

1. 每日任务状态。
2. 数据更新状态。
3. 策略信号汇总。
4. 建议买入/卖出订单表。
5. 当前模拟持仓。
6. 邮件发送状态。
7. 每个策略的中文解释和详细候选股。

但第一版可以只生成 `daily_runs/.../report.html`，先不接入 Web。

## 12. 与 Agent Skill 结合

`quantx-backtest` Skill 后续应增加 reference：

```text
references/daily-production.md
```

Agent 可执行：

```bash
conda run -n test python -m quantx.tools.agent_context daily-status
conda run -n test python -m quantx.tools.agent_context daily-run --profile configs/production/daily_default.yaml --dry-run
conda run -n test python -m quantx.tools.agent_context daily-report --date latest
```

Agent 能力：

1. 检查昨日定时任务是否成功。
2. 汇总哪些策略今天有买卖。
3. 解释某个策略为什么买/卖某只股票。
4. 检查邮件是否发送成功。
5. 当任务失败时定位是数据、策略、渲染还是 SMTP 问题。

## 13. 风险与关键决策

### 13.1 数据源稳定性

目前在线数据源不一定稳定，因此定时层要先支持 `verify_only` 和可配置数据更新命令。不要把某个不稳定下载接口硬编码进 pipeline。

### 13.2 真实持仓与模拟持仓

第一版只做策略模拟持仓。邮件标题和正文必须明确“模拟”。如果要用于真实交易，需要新增真实账户导入或手动校正功能。

### 13.3 邮件安全

QQ 邮箱授权码不能进 git。日志里不能打印授权码。异常信息也要脱敏。

### 13.4 交易日判断

不能只按周一到周五判断。应以 Qlib calendar 或交易所日历为准。

### 13.5 涨停、停牌和成交可行性

生产日报必须展示被过滤的原因，尤其是：

```text
skip_limit_up
价格缺失
成交量为 0
停牌疑似
资金不足
非整手调整
```

### 13.6 策略状态漂移

如果每日模拟成交和真实人工操作不一致，后续信号会漂移。因此第一版要支持人工重置/校正 simulated positions。

## 14. 实施阶段

### Phase 1: 文档和配置骨架

1. 新增 `configs/production/daily_default.yaml` 示例。
2. 新增 `configs/production/mail.example.yaml`。
3. 新增 `.gitignore` 规则，忽略真实 mail 配置和 `daily_state/`。
4. 明确 daily artifacts schema。

### Phase 2: CLI 和 Pipeline 骨架

1. 实现 `quantx.tools.run_daily_pipeline`。
2. 支持 `--stage data|signals|report|mail`。
3. 支持 `--dry-run` 和 `--json`。
4. 输出 `pipeline_status.json` 和 `logs.txt`。

### Phase 3: 数据校验与策略日更

1. 复用 DataService/Exchange 检查数据状态。
2. 复用 ConfigStrategy/FactorRuntime 计算 trade_date 信号。
3. 实现 simulated portfolio state。
4. 输出 candidates、selected、suggested_orders、rejected_orders。

### Phase 4: 中文日报和 QQ 邮件

1. 实现 Markdown/HTML report renderer。
2. 实现 QQ SMTP mailer。
3. 支持环境变量读取授权码。
4. 支持发送幂等和 `--force-send`。

### Phase 5: 可视化与 Agent 接口

1. Web workspace 增加“每日信号”页面。
2. `agent_context` 增加 daily-* 命令。
3. Skill reference 增加 `daily-production.md`。

## 15. 第一版验收标准

第一版完成后，应满足：

1. 可以用一条 CLI 手动跑完整日更流程。
2. 非交易日能正确跳过。
3. 数据未更新时会清楚报错或警告。
4. 至少两个 YAML 策略能生成当天候选、持仓和建议订单。
5. 每个策略都有中文解释，包括买卖原因和拒绝原因。
6. QQ 邮箱能收到 HTML/文本日报。
7. 重复执行不会重复发相同邮件。
8. 所有产物落在 `daily_runs/<date>/<profile>/`，可被 agent 和可视化读取。
9. 真实邮箱授权码不进入 git，也不出现在日志中。

## 16. 建议优先实现的最小闭环

为了快速跑通，建议第一版只做这个闭环：

```text
manual CLI run
  -> verify data/qlib_data_fixed
  -> load enabled YAML strategies
  -> compute latest trade_date signals
  -> maintain simulated positions
  -> render report.md/report.html
  -> send QQ email
```

先不做 Web 调度，不做真实账户接入，不做复杂权限。等 CLI 稳定后，再把 daily artifacts 接到可视化和 Agent Skill。
