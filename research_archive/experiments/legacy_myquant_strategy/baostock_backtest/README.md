# baostock_backtest

一个独立于旧 `backtrade` 的新回测项目，数据源改为 Baostock，并支持本地落盘与增量更新。

## 功能覆盖

- 回测引擎：预选股 + 逐日交易执行 + 账户净值更新。
- 账户与统计：交易记录、收益率、年化、最大回撤、Sharpe、Sortino、Calmar。
- 结果输出：`trades.json`、`trades_and_stock_dict.json`、`metrics.json`、净值曲线图。
- 指标处理链：`limit`、`shuijiao`、`tech`。
- 指标处理链：`limit`、`shuijiao`、`shuijiao_base`、`tech`。
- 配置驱动：支持 Python 配置文件。
- 兼容旧策略：可直接加载 `backtrade.strategy.*` 模块。
- 新增实时选股：根据 `策略 + 日期` 输出当日选股清单。
- 新增 `shuijiao_best_v2`：把买卖信号逻辑从 Processor 迁移到 Strategy（Processor 只保留指标计算）。
- 数据链路：
  - 股票池下载（`query_all_stock`）
  - 本地 CSV 落盘
  - 增量更新（按单票最后日期续拉）
  - 覆盖补齐（自动补前段和后段缺口）

## 安装依赖

```bash
pip install baostock pandas numpy tqdm matplotlib mplfinance
```

> 如果不安装 `mplfinance`，系统仍可回测，但不会输出成交标记 K 线图。

## 目录结构

```text
baostock_backtest/
  cli.py                     # 命令行入口
  codecs.py                  # 股票代码格式转换
  core/                      # Context / Account / Engine / 实时选股
  data/                      # Baostock 客户端 + 本地仓库 + 同步服务
  processors/                # 指标处理器
  runtime/                   # 配置加载、策略加载、股票池解析
  configs/                   # 示例配置
```

## 常用命令

### 1) 同步数据（支持增量）

```bash
python -m baostock_backtest sync-data \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --refresh-universe \
  --workers 1 \
  --pause-seconds 0.0
```

全量同步：

```bash
python -m baostock_backtest sync-data \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --full \
  --workers 1
```

> `sync-data` 不需要策略配置；通常只传 `--data-config` 即可。
> `--runtime-config` 仅在你希望复用其中的 `stock_pool/start_date/end_date/look_back_days` 默认值时才需要。

### 2) 运行回测

```bash
python -m baostock_backtest backtest \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --strategy-config baostock_backtest/configs/layers/strategy_shuijiao_best_v2.py \
  --runtime-config baostock_backtest/configs/layers/runtime_shuijiao_best_v2.py
```

### 3) 实时选股（新增功能）

```bash
python -m baostock_backtest pick \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --strategy-config baostock_backtest/configs/layers/strategy_shuijiao_best_v2.py \
  --runtime-config baostock_backtest/configs/layers/runtime_shuijiao_best_v2.py \
  --date 2025-03-03
```

只看前 20 只：

```bash
python -m baostock_backtest pick \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --strategy-config baostock_backtest/configs/layers/strategy_shuijiao_best_v2.py \
  --runtime-config baostock_backtest/configs/layers/runtime_shuijiao_best_v2.py \
  --date 2025-03-03 \
  --top-k 20
```

输出“截止某日”的全历史选股结果（JSON 文件）：

```bash
python -m baostock_backtest pick \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --strategy-config baostock_backtest/configs/layers/strategy_shuijiao_best_v2.py \
  --runtime-config baostock_backtest/configs/layers/runtime_shuijiao_best_v2_legacy_mainboard.py \
  --date 2026-03-03 \
  --all-dates-until \
  --top-k 50 \
  --no-auto-update \
  --output-file outputs_baostock/picks/picks_until_2026-03-03.json
```

> 兼容说明：`sync-data/backtest/pick` 仍支持旧版 `--config` 单文件写法。

## 稳健更新任务（推荐）

全量更新（首次建库）：

```bash
./scripts/run_stock_update.sh \
  --job-config baostock_backtest/configs/data_update_a_share.py \
  --mode full \
  --start-date 2020-01-01 \
  --workers 1 \
  --pause-seconds 0.0 \
  --max-retries 3 \
  --refresh-universe
```

日常增量更新（每个交易日收盘后）：

```bash
./scripts/run_stock_update.sh \
  --job-config baostock_backtest/configs/data_update_a_share.py \
  --mode incremental \
  --start-date 2020-01-01 \
  --workers 1 \
  --pause-seconds 0.0 \
  --max-retries 3 \
  --refresh-universe
```

> 任务内置文件锁（防重入）和失败重试，运行结果会落盘到 `outputs_baostock/job_reports/`。

### 定时任务（cron）

参考文件：`scripts/cron_stock_update.example`
示例配置为每周一到周五 15:20 自动执行增量更新。

## 配置说明

推荐使用三层配置解耦：

- `data-config`：数据源与落盘参数（`data_root/adjust_flag/pause_seconds`）。
- `strategy-config`：策略模块与指标链（`strategy_config/processor_config`）。
- `runtime-config`：账户、股票池、回测区间、缓存路径。

`stock_pool` 支持：
- `all`：全A股（含主板/创业板/科创板/北交所，以本地 universe 为准）
- `all_mainboard`：沪深主板口径（排除 `300/301/688`）
- `test`：内置小样本
- 显式列表：如 `["sh.600519", "sz.000001"]`

示例（`strategy_config` 支持 `str` 或 `dict`）：

```python
strategy_config = {
    "module": "backtrade.strategy.shuijiao_strategy",
    "params": {}
}
```

- `account_config`：资金与交易成本参数。
- `context_config`：股票池、回测区间、回看天数、处理器链。
- `data_config`：数据根目录、复权方式、自动补齐开关。
- `engine_config`：选股并发度。

### Shuijiao Best V2（推荐迁移）

- 配置文件：`baostock_backtest/configs/shuijiao_best_v2.py`
- 核心策略：`baostock_backtest.strategy.shuijiao_best_v2`
- 处理器：`shuijiao_base`（只计算 `trend_line/midline/rsi/kdj/bbi/ema...` 指标，不直接输出 `buy/sell` 信号）

旧版口径对齐（用于和历史 `shuijiao_best_auto` 对比）：

```bash
python -m baostock_backtest backtest \
  --data-config baostock_backtest/configs/layers/data_a_share.py \
  --strategy-config baostock_backtest/configs/layers/strategy_shuijiao_best_v2.py \
  --runtime-config baostock_backtest/configs/layers/runtime_shuijiao_best_v2_legacy_mainboard.py \
  --no-auto-update
```

## 代码风格

- 模块按职责拆分，避免 Context/Engine/Data 强耦合。
- 关键业务路径均附中文注释，便于后续扩展策略与风控逻辑。
