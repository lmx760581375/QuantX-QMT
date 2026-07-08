# Shuijiao Backtest Runbook

这份文档说明如何在 QuantX 中使用 Qlib 数据层运行 `shuijiao` 策略回测，以及如何重建、更新、验证数据层。

当前状态：

- QuantX 已经支持通过 Qlib 读取行情数据。
- `shuijiao` 的因子计算走 `MarketPanel + FactorRuntime`，在内存矩阵上计算，不依赖预计算因子文件。
- 当前已经有最小可用 YAML/config 策略 runner，推荐使用 `quantx.tools.run_backtest --config configs/strategies/shuijiao_legacy.yaml`。
- `quantx.tools.run_shuijiao_qlib_diagnostic` 保留为对齐诊断入口。
- 旧的 `data/qlib_data` 曾经由错误 converter 生成过，可能仍是坏格式。第一次正式使用前必须全量重建，或者把已经验证过的 `data/qlib_data_fixed` 替换成正式目录。

## 1. 环境准备

建议使用公司开发机上的 conda `test` 环境。

```bash
cd /home/users/mingxiao.li/git/quantization/quantx
conda activate test
```

确认 Python 环境：

```bash
which python
python -V
```

注意：仓库内有一份本地 `qlib/` 源码目录，但该目录的编译扩展可能和当前 Python 版本不一致。QuantX 的 `AStockExchange` 会优先使用 conda 环境里已安装的 qlib，避免从 repo 根目录启动时误导入未编译源码目录。

## 2. 数据目录

主要目录：

```text
data/raw/baostock/stocks/     # BaoStock 原始 CSV，本地已有 5000+ 股票
data/qlib_data/               # 正式 Qlib 数据目录，回测默认读取这里
data/qlib_data_fixed/         # 本次为了验证新 converter 生成的临时正确数据目录
```

Qlib 正确 bin 格式：

```text
[start_idx, value0, value1, value2, ...]
```

错误旧格式：

```text
[date_idx, value, date_idx, value, ...]
```

如果旧格式被 Qlib 读取，会把 `1.0, 2.0, 3.0...` 这类日历 index 当成价格，导致明显不可能的行情值。

## 3. 首次修复正式 Qlib 数据

如果 `data/qlib_data` 还没有重建，建议先备份，再使用已验证的 fixed 目录替换。

```bash
cd /home/users/mingxiao.li/git/quantization/quantx

mv data/qlib_data data/qlib_data_bad_backup
mv data/qlib_data_fixed data/qlib_data
```

如果你希望重新从本地 CSV 生成正式目录，而不是移动 fixed 目录，可以运行：

```bash
cd /home/users/mingxiao.li/git/quantization/quantx

python - <<'PY'
from quantx.core.data.converter import BaostockToQlibConverter

BaostockToQlibConverter(
    qlib_dir="data/qlib_data",
    csv_dir="data/raw/baostock/stocks",
).convert_all("2010-01-01", "2026-06-25")
PY
```

这个命令不联网，只从本地 CSV 转换为 Qlib bin。全市场 5000 多只股票，通常几十秒到数分钟，取决于机器状态。

## 4. 验证 Qlib 数据

### 4.1 检查 bin 文件头

```bash
python - <<'PY'
from pathlib import Path
import struct

base = Path("data/qlib_data")
for sym in ["sz000008", "sh600016", "sz000001"]:
    p = base / "features" / sym / "close.day.bin"
    raw = p.read_bytes()[:4 * 12]
    vals = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    print(sym, vals)
PY
```

正确结果应该类似：

```text
sz000008 (0.0, 1.3876, 1.3876, 1.3876, ...)
```

如果看到下面这种交错模式，说明目录还是旧坏数据：

```text
0.0, price, 1.0, price, 2.0, price, ...
```

### 4.2 通过 Qlib 读取行情

```bash
PYTHONPATH=qlib python - <<'PY'
import qlib
from qlib.data import D

qlib.init(provider_uri="data/qlib_data", region="cn")
df = D.features(
    ["SZ000008", "SH600016"],
    ["$close", "$open", "$high", "$low", "$volume", "$change"],
    "2021-01-04",
    "2021-01-08",
    freq="day",
)
print(df)
PY
```

关键抽样值应是合理价格，例如：

```text
SZ000008 2021-01-04 close ~= 2.52
SH600016 2021-01-04 close ~= 3.88
```

## 5. 更新数据

### 5.1 全量同步 BaoStock + 转 Qlib

全量同步脚本：

```bash
python tools/sync_full.py
```

这个脚本会：

1. 从 BaoStock 拉股票列表。
2. 拉全量日线 CSV。
3. 转换为 Qlib 格式。

全量拉取会比较慢，脚本注释里预估可能需要 2 到 4 小时，取决于网络和 BaoStock 状态。

### 5.2 只从已有 CSV 重建 Qlib 数据

如果只是 converter 修复后要重建 Qlib 数据，不需要重新拉 BaoStock，直接跑：

```bash
python - <<'PY'
from quantx.core.data.converter import BaostockToQlibConverter

BaostockToQlibConverter(
    qlib_dir="data/qlib_data",
    csv_dir="data/raw/baostock/stocks",
).convert_all("2010-01-01", "2026-06-25")
PY
```

### 5.3 增量更新

当前 converter 的增量路径已经修复，不会再追加旧的交错格式。`convert_incremental()` 会对更新股票重写 bin 文件，避免日历和复权错位。

但如果正式 `data/qlib_data` 里还混有旧坏 bin，不建议只增量更新少数股票，因为没更新到的旧文件仍然坏。正确顺序是：

1. 先全量重建一次 `data/qlib_data`。
2. 之后再使用增量更新。

## 6. 运行 Shuijiao 回测

推荐 YAML 入口：

```bash
python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --json
```

本次在 `data/qlib_data_fixed` 上实测：

```text
total_return: 68.98%
final_value: 1,689,807.15
trades: 486
buys: 248
sells: 238
max_drawdown: -30.31%
time_stats.total: 16.31s
```

可用少量股票做 smoke test：

```bash
python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --symbol-limit 300 \
  --json
```

诊断入口：

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic --help
```

主要参数：

```text
--provider-uri              Qlib 数据目录，正式使用 data/qlib_data
--start                     回测开始日期
--end                       回测结束日期
--init-cash                 初始资金，默认 1000000
--max-positions             最大持仓数
--look-back-days            指标 warmup 历史窗口，shuijiao 对齐旧配置建议 100
--max-workers               Phase 1 信号并行线程数
--symbol-limit              只跑前 N 只股票，用于 smoke test
--validate-trading-rules    开启 QuantX 交易所规则校验，默认关闭以对齐旧 myquant
--legacy-wrap-first-signal  复现旧 myquant 第一日取 timelist[-1] 信号的行为
--allow-limit-up-buy        允许涨停买入；默认不允许，以匹配旧 myquant 源码
--no-legacy-reuse-sell-cash 不按旧 myquant 方式复用同日卖出资金做买入 sizing
--max-profit-pct            止盈阈值，默认 0.20
--max-loss-pct              止损阈值，默认 -0.10
--json                      输出 JSON
```

### 6.1 Smoke test

先用少量股票和短时间确认链路可用：

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic \
  --provider-uri data/qlib_data \
  --start 2021-01-04 \
  --end 2021-03-31 \
  --symbol-limit 200 \
  --max-positions 100 \
  --look-back-days 100 \
  --json
```

### 6.2 按旧 myquant 源码参数跑

旧源码里 `max_positions=100`，可用这个命令：

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic \
  --provider-uri data/qlib_data \
  --start 2021-01-04 \
  --end 2025-10-17 \
  --max-positions 100 \
  --look-back-days 100 \
  --max-workers 1 \
  --json
```

本次诊断中，使用修复后的 Qlib 数据，全市场约 5435 个标的，该配置耗时约 38 秒，交易次数约 4905。

### 6.3 严格旧 myquant 源码兼容口径

旧输出 `trades.json` 的实际表现更像 10 仓位，不像源码里写的 100 仓位。严格旧源码兼容口径会：

- 禁用新版 RSI/KDJ/BBI/market health 等过滤。
- 使用 close 成交。
- 跳过涨停买入。
- sizing 时复用同日卖出后可用现金。
- 使用旧源码的 20% 止盈、10% 止损。

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic \
  --provider-uri data/qlib_data \
  --start 2021-01-04 \
  --end 2025-10-17 \
  --max-positions 10 \
  --look-back-days 100 \
  --max-workers 1 \
  --json
```

本次在 `data/qlib_data_fixed` 上实测：

```text
total_return: 68.98%
trades: 486
sells: 238
time_stats.total: 15.80s
```

如果加上 `--legacy-wrap-first-signal` 复现旧 myquant 第一日取最后一天信号的行为，本次 qlib 数据结果为：

```text
total_return: 0.52%
trades: 476
sells: 233
time_stats.total: 16.58s
```

这个结果说明：严格补齐旧源码执行语义后，qlib 数据/信号口径与旧 outputs artifact 仍有明显差异，不能只用收益率接近来判定框架正确。

### 6.4 放松 artifact 对齐口径

如果为了复现之前“接近旧 `trades.json` 收益”的诊断结果，可以显式放松两个旧源码行为：

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic \
  --provider-uri data/qlib_data \
  --start 2021-01-04 \
  --end 2025-10-17 \
  --max-positions 10 \
  --look-back-days 100 \
  --max-workers 1 \
  --legacy-wrap-first-signal \
  --allow-limit-up-buy \
  --no-legacy-reuse-sell-cash \
  --json
```

本次在 `data/qlib_data_fixed` 上实测：

```text
total_return: 83.70%
trades: 482
sells: 236
time_stats.total: 16.17s
```

这个口径接近旧 baseline 的 87.38%，但它依赖“允许涨停买入”和“买入 sizing 不复用同日卖出现金”两个与旧源码不一致的行为，所以只能作为 artifact 解释，不能作为严格正确性验收。

### 6.5 开启 QuantX 真实交易规则

默认命令为了对齐旧 myquant，会关闭交易所规则校验。要跑更接近真实 A 股规则的 QuantX 回测，加上：

```bash
--validate-trading-rules
```

例如：

```bash
python -m quantx.tools.run_shuijiao_qlib_diagnostic \
  --provider-uri data/qlib_data \
  --start 2021-01-04 \
  --end 2025-10-17 \
  --max-positions 10 \
  --look-back-days 100 \
  --validate-trading-rules \
  --json
```

开启后会启用停牌、涨跌停、一字板、T+1、价格跳变等校验。收益和交易次数可能明显不同，这属于更真实规则与旧系统的差异，不一定是 bug。

## 7. 输出字段说明

JSON 输出中常用字段：

```text
final_value          期末总资产
total_return         总收益率，小数形式，0.873 表示 87.3%
annual_return        年化收益率，小数形式
max_drawdown         最大回撤，小数形式
trades               成交记录数
buys                 买入次数
sells                卖出次数
rejects              被交易规则拒绝的订单数
final_cash           期末现金
final_positions      期末持仓数
time_stats.total     总耗时，秒
```

`baseline_*` 和 `diff_*` 是为了和 old myquant 的历史输出做诊断对比。它们不是 QuantX 的硬编码目标，只是当前审计用参考。

## 8. 本次已知诊断结论

本次排查过程中确认了几个重要点：

1. 原 `data/qlib_data` 曾经是坏 bin 格式，必须全量重建。
2. 修复后的 Qlib 数据可以正常读，抽样价格合理。
3. QuantX 引擎曾有一个 Phase 1/Phase 2 类型衔接 bug：信号缓存为 list，执行阶段当成 `StockSelection`。已修复。
4. QuantX 引擎此前没有真正使用 `look_back_days` warmup 行情。已修复。
5. `shuijiao` 因子计算已经是矩阵化运行，全市场 2021-2025 回测大约十几秒到几十秒。
6. `shuijiao` 当前运行路径已经把核心指标展开为 FactorRuntime 公式 DAG，不再依赖 `ShuijiaoBuy(...)` 这种黑盒公式作为主路径。
7. 旧 myquant outputs artifact 与当前旧源码不完全一致：例如 `000975.SZ` 在 2021-02-22 已超过 20% 止盈阈值，但旧 `trades.json` 一直持有到 2021-05-10。
8. 当前推荐使用统一 YAML runner；`run_shuijiao_qlib_diagnostic.py` 只保留为专项诊断入口。

### 8.1 2021-2025 myquant 对齐审计

这轮审计目标不是强行追某个收益数字，而是确认 QuantX 回测框架是否有会扭曲结果的漏洞。当前 myquant 参考结果来自：

```text
/home/users/mingxiao.li/git/quantization/myquant-strategy/outputs/shuijiao_best_auto_all5y_metrics_recheck.json
```

myquant 结果口径：

```text
period: 2021-01-04 ~ 2025-10-17
total_return: 186.47%
final_value: 2,864,715.22
max_drawdown: -25.10%
trades_count: 2044
sell_count: 1004
sharpe: 1.278
```

QuantX 已复现并检查过这些关键口径：

```text
all_a + look_back_days=120, execution-loop 修复后:          total_return=81.94%, trades=2124
all_mainboard + look_back_days=120:                         total_return=241.21%, trades=2056
myquant exact universe + look_back_days=120:                 total_return=234.09%, trades=2046
myquant exact universe + look_back_days=0:                   total_return=205.06%, trades=2016
myquant exact universe + look_back_days=0 + account fixes:   total_return=205.06%, trades=2016
```

最新复跑 run id：

```text
runs/full_myquant_shuijiao_best_auto_exact_universe_no_lookback_2021_2025_accountfix
```

最新复跑结果：

```text
symbols: 3183
final_value: 3,050,593.80
total_return: 205.06%
max_drawdown: -25.95%
sharpe: 1.3315
trades: 2016
buys: 1026
sells: 990
time_stats.total: 22.11s
```

这说明，QuantX 目前在最接近 myquant 的口径下，与 myquant 的收益差距约为 18.6 个百分点，而不是最初看起来的 `68%/81%` 与 `186%` 的巨大差距。

### 8.2 已确认并修复的框架问题

1. **执行循环 bug**：旧逻辑只在当日 selection 非空时调用交易逻辑，导致没有买入候选的日期不会执行止盈、止损、信号卖出。已修复为每天都调用 `strategy.get_trade_signal(...)`，空 selection 也进入卖出规则。
2. **股票池口径 bug/缺口**：QuantX `all_a` 包含科创板、创业板等，而 myquant 的 `stock_pool="all"` 会排除 `688`、`300`、`301`。已新增 `data.universe: all_mainboard`，并用 exact universe 做过进一步诊断。
3. **预计算安全边界**：两阶段并行预计算只适合纯行情/因子 selector。若 Python 策略的 `get_stock_signal` 依赖账户、持仓、累计收益或动态风控，预计算会读到错误状态。现在 `BacktestConfig.precompute_signals` 默认由策略显式声明；YAML 公式策略声明为可预计算，普通 Python 策略默认逐日计算。
4. **报告字段 bug**：交易成本原来被整体写进 `commission`，`stamp_tax` 字段为 0，导致 artifact 和 myquant 不方便逐项对比。已修复为 commission/stamp_tax/transfer_fee 分字段记录，总成本不变。
5. **日收益字段 bug**：`Account.update_daily_balance()` 原来用交易后的未重估资产做当日收益分母，可能污染 `daily_return` 派生分析。已改为以前一日收盘总资产为分母。`compute_metrics()` 仍使用 `daily_nav.total_value.pct_change()` 重新计算指标，因此这次修复没有改变最新 run 的 total_return、max_drawdown、sharpe。

相关回归测试：

```bash
conda run -n test python -m pytest \
  tests/engine/test_account_costs.py \
  tests/engine/test_engine_execution_loop.py \
  tests/tools/test_run_backtest_config.py \
  tests/factor_runtime/test_runtime.py \
  tests/strategy/test_config_strategy.py \
  tests/analysis/test_reporting.py \
  -q
```

当前结果：

```text
26 passed
```

### 8.3 当前差异更像数据/口径差异，不应先判为 QuantX 框架 bug

目前最接近口径的 QuantX 为 `205.06%`，myquant 为 `186.47%`。已知差异来源包括：

1. **复权/价格数据不同**：myquant pkl 中 `600276.SH` 在 2021-01-06 收盘价约 `93.04467`，QuantX Qlib 数据中 `SH600276` 同日收盘价约 `92.66861`；`000631.SZ` 也存在同类差异。这会影响买入数量、止盈止损和最终净值。
2. **冷启动/warmup 不同**：myquant pkl 从 2021-01-04 才开始，指标冷启动；QuantX 如果使用 `look_back_days=120`，首日信号已经有更长历史，早期买入会完全不同。exact universe + `look_back_days=0` 后已经明显靠近 myquant。
3. **股票池构造不同**：`all_a` 与 myquant `all` 不同，会改变 `market_health`、排名和持仓。exact universe 诊断说明这是核心差异之一。
4. **旧系统 artifact 本身可能不是严格源码口径**：旧 `trades.json` 中存在超过当前源码止盈阈值却长期未卖的样例，说明不能把历史 artifact 当成绝对真值。

结论：当前已修复发现的 QuantX 框架 bug。剩余 `205.06%` vs `186.47%` 的差异，优先按数据、复权、冷启动、旧 artifact 口径继续定位；除非后续能构造同数据同价格下的最小差异，否则不应直接判定为 QuantX 回测漏洞。

## 9. YAML 配置化入口

当前推荐入口已经切到统一 YAML runner，不再需要为 `shuijiao` 写专门启动脚本：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --json
```

启动前先做配置编译和公式 DAG 检查：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --dry-run \
  --json
```

`--dry-run` 只做 YAML 解析、字段 alias 检查、公式依赖拓扑排序、selector/rebalance/execution 校验和股票池覆盖检查，不执行交易。输出中会包含：

- `strategy.formula_order`：公式执行顺序。
- `strategy.dependencies`：每个公式依赖的字段或中间变量。
- `strategy.selector`：选股条件、排序、lag、topk。
- `strategy.rebalance`：等权调仓和最大持仓数。
- `strategy.execution`：卖出规则、买入 sizing、涨停过滤、是否复用卖出现金。

生成标准回测产物用于 Web 工作台和 agent 读取：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --output-dir runs \
  --run-id shuijiao_legacy_latest \
  --json
```

产物目录包含：

- `summary.json`：收益、交易次数、耗时等摘要。
- `metrics.json`：年化、回撤、夏普、胜率、交易成本等指标。
- `daily_nav.json`：每日净值、收益、回撤、持仓数。
- `trades.json`：成交明细。
- `positions.json`：每日持仓快照。
- `closed_positions.json`：FIFO 配对后的平仓统计。
- `explain.json`：config snapshot、engine 参数、公式 DAG、signal errors。

当前 `configs/strategies/shuijiao_legacy.yaml` 已经把买入、卖出和中间因子展开成通用公式 DAG，不依赖 `ShuijiaoBuy(...)`、`ShuijiaoTrend(...)` 这种策略专属黑盒 operator 作为主路径。关键结构如下：

```yaml
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

signals:
  trend_prev: Ref(trend_line, 1)
  bb1: (trend_prev < 11) and (trend_prev > 6) and Cross(trend_line, 11)
  bb2: (trend_prev < 6) and (trend_prev > 3) and Cross(trend_line, 6)
  bb3: (trend_prev < 3) and (trend_prev > 1) and Cross(trend_line, 3)
  bb4: (trend_prev < 1) and (trend_prev > 0) and Cross(trend_line, 1)
  bb5: (trend_prev < 0) and Cross(trend_line, 0)
  buy_signal: (bb1 or bb2 or bb3 or bb4 or bb5) and (close < midline)
  sell_signal: (trend_line > 89) and Filter(trend_line > 89, 15) and (close > midline)
```

本地 Web 工作台：

```bash
conda run -n test python -m quantx.server.app
```

打开 `http://127.0.0.1:8000` 后可以选择 YAML、在线校验、启动回测、读取报告和查看数据状态。
