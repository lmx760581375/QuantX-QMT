# 工具链层开发计划

> **版本**: v0.1.0 | **日期**: 2026-06-25 | **预计工期**: 3-5 天 | **依赖**: 所有核心模块

---

## 1. 目标

工具链层提供 CLI 命令行接口，让用户通过命令行完成数据同步、回测运行、因子计算等操作。同时提供定时数据同步的 Job 调度。

---

## 2. 文件清单

```
quantx/tools/
├── __init__.py
├── cli.py                   # CLI 主入口（argparse）
├── sync_data.py             # 数据同步命令
├── cron_sync.py             # 定时数据同步 Job（cron 调度）
├── run_backtest.py          # 回测运行命令
├── run_factor.py            # 因子计算/校验命令
├── validate_data.py         # 数据校验命令
└── generate_report.py       # 报告生成命令
```

```
quantx/configs/
├── data/
│   └── baostock.yaml        # 数据源配置
├── backtest/
│   └── default.yaml         # 回测默认配置
├── factors/
│   └── alpha101.yaml        # 因子配置
└── strategies/
    └── ma_cross.yaml        # 策略配置
```

---

## 2.2 各文件详细说明

### 2.2.1 `cli.py` — CLI 主入口

**命令结构**：

```
quantx
├── sync-data       # 数据同步
│   ├── --mode full|incremental
│   ├── --start 2020-01-01
│   ├── --end 2025-12-31
│   ├── --workers 8
│   └── --config configs/data/baostock.yaml
│
├── backtest        # 运行回测
│   ├── --strategy strategies/ma_cross_strategy.py
│   ├── --start 2020-01-01
│   ├── --end 2025-12-31
│   ├── --cash 1000000
│   ├── --config configs/backtest/default.yaml
│   └── --output outputs/backtest_result/
│
├── factor          # 因子计算/校验
│   ├── --compute alpha005,alpha006
│   ├── --validate alpha005
│   ├── --list
│   └── --start 2020-01-01 --end 2025-12-31
│
├── validate-data   # 数据校验
│   └── --start 2020-01-01 --end 2025-12-31
│
└── report          # 生成报告
    └── --input outputs/backtest_result/
```

**实现方式**：使用 `argparse` 的 `add_subparsers()` 实现子命令。

**开发要点**：直接复用 `myquant-strategy/baostock_backtest/cli.py` 的 CLI 设计（约 100 行）

### 2.2.2 `sync_data.py` — 数据同步命令

**功能**：
1. 读取配置
2. 创建 BaoStockClient + DataSyncService + Converter
3. 执行全量/增量同步
4. 执行 Qlib 格式转换
5. 执行数据校验
6. 输出同步报告

**开发要点**：直接复用 `myquant-strategy/baostock_backtest/jobs/update_all_stocks.py` 的 Job 框架（约 80 行）

### 2.2.3 `run_backtest.py` — 回测运行命令

**功能**：
1. 读取配置
2. 加载策略模块
3. 初始化 Qlib
4. 创建 BacktestEngine + Account + Exchange
5. 运行回测
6. 计算绩效指标
7. 生成报告和图表

### 2.2.4 `run_factor.py` — 因子计算命令

**功能**：
- `--compute`: 计算指定因子并缓存
- `--validate`: 校验因子值（覆盖率、IC、异常值）
- `--list`: 列出所有已注册因子

### 2.2.5 `cron_sync.py` — 定时数据同步 Job

**功能**：定时自动同步数据，每个交易日收盘后自动触发增量更新。

```python
#!/usr/bin/env python3
"""定时数据同步 Job，通过 Linux cron 或 Python schedule 库调度"""

import fcntl
import json
from datetime import datetime
from pathlib import Path

class DataSyncJob:
    def __init__(self, config_path: str = "configs/data/baostock.yaml"):
        self.config = load_config(config_path)
        self.lock_file = Path(self.config.data.root) / ".sync.lock"

    def run(self):
        # 1. 文件锁防止并发执行
        with open(self.lock_file, 'w') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another sync job is running, exiting.")
                return

            # 2. 增量同步
            source = BaoStockDataSource(self.config.data)
            report = source.sync_service.sync_incremental(
                source.repository.load_universe()
            )

            # 3. Qlib 格式转换
            source.converter.convert_incremental(report.updated_stocks)

            # 4. 数据校验
            validation = source.validator.validate_coverage(
                report.all_stocks, self.config.data.start, self.config.data.end
            )

            # 5. 输出报告
            report_path = Path("outputs") / f"sync_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            report_path.parent.mkdir(exist_ok=True)
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "synced": report.synced_count,
                "failed": report.failed_count,
                "validation": validation,
            }, open(report_path, 'w'), indent=2)
```

**Cron 配置**（每个交易日 15:30 执行）：
```cron
# /etc/cron.d/quantx-sync
30 15 * * 1-5 mingxiao /usr/bin/python3 /path/to/quantx/tools/cron_sync.py
```

**开发要点**：直接复用 `myquant-strategy/baostock_backtest/jobs/update_all_stocks.py`（约 80 行）

---

## 3. CLI 入口点配置

在 `pyproject.toml` 中配置 CLI 入口点：
```toml
[project.scripts]
quantx = "quantx.tools.cli:main"
```

安装后用户可直接使用 `quantx` 命令：
```bash
pip install -e .
quantx sync-data --mode incremental
quantx backtest --strategy strategies/ma_cross_strategy.py --start 2020-01-01 --end 2025-12-31
quantx factor --list
quantx validate-data
quantx report --input outputs/backtest_result/
```

---

## 4. 配置管理

### 3.1 三层配置合并

借鉴 `myquant-strategy` 的三层配置：

```python
@dataclass
class DataConfig:
    source: str = "baostock"
    root: str = "./data"
    adjust: str = "forward"
    provider_uri: str = "./data/qlib_data"

@dataclass
class BacktestConfig:
    init_cash: float = 1_000_000
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "SH000300"
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.001

@dataclass
class StrategyConfig:
    module: str = "strategies.ma_cross_strategy"
    params: dict = field(default_factory=dict)

@dataclass
class RuntimeConfig:
    data: DataConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
```

---

## 4. 验收标准

- [ ] `quantx sync-data` 可完成全量数据同步 + Qlib 转换
- [ ] `quantx backtest` 可运行完整回测并输出结果
- [ ] `quantx factor --list` 可列出所有已注册因子
- [ ] 所有命令支持 `--config` 参数指定配置文件
