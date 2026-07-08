# QuantX 量化研究与回测系统 --- 总体架构设计方案

> **版本**: v0.1.0
> **日期**: 2026-06-25
> **状态**: 总体大纲方案（第一阶段）
> **语言**: 中文

---

## 目录

1. [现有项目阅读总结](#1-现有项目阅读总结)
2. [设计目标与原则](#2-设计目标与原则)
3. [总体架构设计](#3-总体架构设计)
4. [数据层设计](#4-数据层设计)
5. [因子系统设计](#5-因子系统设计)
6. [回测系统设计](#6-回测系统设计)
7. [性能优化设计](#7-性能优化设计)
8. [测试与验证工作流设计](#8-测试与验证工作流设计)
9. [分阶段落地路线图](#9-分阶段落地路线图)
10. [风险点与设计取舍](#10-风险点与设计取舍)
11. [假设前提](#11-假设前提)


---

## 1. 现有项目阅读总结

### 1.1 项目总览

`quantization` 目录下共有 9 个项目，按性质分为三类：

| 类别 | 项目 | 定位 |
|------|------|------|
| 自研项目 | `myquant-clean` | 最完整的回测框架，含三代引擎、因子系统、自动因子挖掘、RL |
| 自研项目 | `myquant-strategy` | 策略研究主项目，含三代回测、参数消融、CLI、数据同步 |
| 自研项目 | `myquant-strategy-baseline` | 基线快照版，冻结在消融/融合之前 |
| 自研项目 | `myquant-rl` | 强化学习扩展，含 Transformer 模型 |
| 自研项目 | `myyquant-1116` | 因子缓存优化分支，Alpha101 集成 |
| 自研项目 | `new-quant` | 下一代清洁架构，含数据源抽象、两阶段引擎、独立测试 |
| 开源参考 | `OSkhQuant` | PyQt5 桌面量化平台，MiniQMT 深度集成，MyTT 指标库 |
| 开源参考 | `alpha-gpt` | LangGraph LLM 流水线，自动因子生成，PostgreSQL 持久化 |

### 1.2 各项目详细分析

#### 1.2.1 myquant-clean --- 核心参考项目

**定位**: 我自研的最完整回测框架，是本次设计的核心参考。

**可直接复用的部分**:
- `backtrade_v2/` 的预计算信号矩阵 + 稀疏存储模式：`PrecomputedSelector` 在回测循环前批量计算所有日期的选股信号，信号矩阵只存 `selected=True` 的行，大幅降低内存占用
- `backtrade_v2/fast_engine.py` 的两阶段分离设计：Phase 1 并行计算信号，Phase 2 顺序执行交易
- `config.yaml` 的配置驱动设计：实验参数、数据参数、回测参数、PPO 参数全部 YAML 化
- `straregys/configs.json` 的选股器注册机制：通过 JSON 配置激活/停用选股器，`importlib` 动态加载类
- `dataset/alpha101.py` 和 `alphas.py` 的因子实现：WorldQuant 101 公式因子的完整 Python 实现
- `auto_factor_mining/` 的 LLM 驱动因子挖掘流水线：FactorGenerator -> StrategyGenerator -> AutoTester -> ResultAnalyzer 四阶段 pipeline
- `dataset/xtdata.py` 的复权处理：`origin` 模式（仿射变换 a*price+b）和 `equiv` 模式（纯乘法等价调整），两种模式可切换

**需要重构后复用的部分**:
- V1 引擎的架构耦合太紧（数据、信号、交易混在一起），需要按新架构解耦
- 数据路径硬编码问题（`/home/users/mingxiao.li/git_projects/myquant-main/data`），需要改为可配置
- V3 引擎的 API 兼容性问题（`buy/sell` 参数顺序不一致），需要统一接口

**仅作参考的部分**:
- V1 引擎的整体架构不适合直接复用，但其 T+1 逻辑实现可作为参考
- `context_optimized.py` 的多进程并行方案，multiprocessing.Pool 对 picklable 对象有要求，可改用 ThreadPoolExecutor

**关键设计模式**:
- Bar-Delayed Execution（T+1 逻辑）：用 T-1 日的收盘价做决策，在 T 日的开盘价执行
- MultiIndex DataFrame 作为核心数据结构：`(date, stock)` 索引，O(log n) 切片查询
- 调整因子预计算：在数据加载阶段一次性完成复权，不在回测循环中重复计算
- 信号矩阵稀疏存储：对于 900 天 x 50 股 x 5% 选中率，仅 2250 条记录 vs 45000 条全矩阵



#### 1.2.2 myquant-strategy --- 策略研究主项目

**定位**: 策略研究主项目，数据管理与回测能力最成熟，是本次设计数据层的主要参考。

**可直接复用的部分**:
- `baostock_backtest/data/baostock_client.py`：BaoStock API 封装，含登录/登出上下文管理器、自动日期回溯、字段重命名、重试逻辑
- `baostock_backtest/data/repository.py`：本地 CSV 数据仓库，按 `stocks/{code}.csv` 布局，支持 upsert 去重
- `baostock_backtest/data/sync_service.py`：全量/增量数据同步服务，多线程分片拉取，3 次重试线性退避，`ensure_coverage` 检查间隙
- `baostock_backtest/jobs/update_all_stocks.py`：生产级数据更新 Job，含文件锁（fcntl.flock）、失败重试、JSON 报告
- `baostock_backtest/cli.py`：CLI 命令行工具（sync-data / backtest / pick 三个子命令）
- `baostock_backtest/runtime/config_loader.py`：三层配置合并（Data + Strategy + Runtime），LoadedConfig dataclass
- `baostock_clean/formulas.py`：FormulaEngine，支持 REF/EMA/CROSS/FILTER/RANK_PCT/ZSCORE 等表达式求值
- `baostock_clean/indicator_library.py`：内置指标集（IndicatorSpec），支持 scope=symbol/cross_section
- `baostock_clean/strategy/templates.py`：声明式策略模板（SHUIJIAO_BEST_V2_PRESET）
- `backtrade/auto_ablate_shuijiao.py` 和 `auto_ablate_fusion.py`：两轮进化式参数消融框架

**需要重构后复用的部分**:
- 参数消融框架需要从策略代码中解耦，改为通用的超参搜索模块
- 三层配置合并需要与 new-quant 的配置模型统一
- FormulaEngine 需要与因子系统整合，作为因子表达式引擎

**仅作参考的部分**:
- 水饺策略的具体逻辑（这是特定策略，不适合作为系统级组件）
- Bazi 八字策略（实验性策略，不适合纳入核心系统）

**关键设计模式**:
- 策略参数注入：`set_runtime_params(params)` 使消融框架无需修改策略源码即可注入参数
- 声明式策略：`TemplateStrategy` 通过配置 dict 定义买卖信号、筛选、评分、风控规则
- 数据层分离：DataConfig / StrategyConfig / RuntimeConfig 三层配置，职责清晰

#### 1.2.3 myquant-strategy-baseline --- 基线快照

**定位**: `myquant-strategy` 的基线快照版，冻结在消融/融合之前。

**可复用价值**: 代码结构清晰简单，适合作为理解回测引擎基本流程的参考。无额外复用价值（后续版本已全面超越）。

#### 1.2.4 myquant-rl --- 强化学习扩展

**定位**: 基于强化学习的因子权重优化实验。

**可复用价值**:
- `train/dataset.py`：将 MultiIndex DataFrame 转换为 RL 训练样本的 Dataset 实现
- `train/validation_backtest.py`：训练过程中的回测验证
- `configs/train/quant_rl.yaml`：RL 训练配置模板

**仅作参考**: RL 方案整体属于实验性质，短期内不建议作为核心系统组件。

#### 1.2.5 myyquant-1116 --- 因子缓存优化分支

**定位**: 在 Context 中集成 Alpha101 因子缓存的优化版本。

**可复用价值**:
- `Context._add_quant_factors()` 的因子 CSV 缓存策略：按 `{start_date}-{end_date}` 作为缓存键，首次计算后持久化，后续直接读取
- 使用 `ThreadPoolExecutor` 并行计算多个因子
- 宽表格式（date x stock）存储因子值，便于批量读取

**关键设计模式**:
- 因子懒计算 + 持久化缓存：首次计算后落盘，回测时直接读取
- 缓存键与日期范围绑定：日期范围变更自动触发重算

#### 1.2.6 new-quant --- 下一代清洁架构

**定位**: 清洁架构的下一代回测框架，是本次设计架构层的主要参考。

**可直接复用的部分**:
- `core/data/base.py`：`DataSource` 抽象基类，定义 `get_daily_bars()` / `get_dividend_factors()` / `get_stock_list()` / `get_benchmark()` 接口
- `core/data/csv_source.py`：`CSVDataSource` 实现
- `core/data/adjuster.py`：独立的复权模块，支持 affine 和 multiplicative 两种模式
- `core/data/cache.py`：MD5 哈希键的 pickle 缓存层，含 `.meta.json` 元数据旁路文件
- `core/engine/engine.py`：两阶段回测引擎（Phase 1 并行信号计算，Phase 2 顺序交易执行）
- `core/engine/context.py`：含板别感知涨跌停阈值（创业板 20%、科创板 30%、主板 10%）
- `core/engine/account.py`：含完整的 Performance Metrics 计算（Sharpe、Sortino、Calmar、Win Rate）
- `core/engine/executor.py`：独立的订单执行器，支持停牌检查、涨跌停检查、价格跳变保护
- `core/engine/cost.py`：`TransactionCost` dataclass（佣金、印花税、过户费、滑点）
- `core/strategy/loader.py`：动态策略加载器，支持任意路径
- `tests/test_integration.py`：6 个使用合成数据的集成测试
- `pyproject.toml`：已声明依赖、可选 extras（baostock、rl、miniqmt）

**需要重构后复用的部分**:
- `Context` 和 `BacktestEngine` 需要与因子系统、数据源抽象层对接
- 缓存层需要从 `~/.quant_cache/` 全局路径改为项目级可配置路径
- 需要增加交易日历模块（目前依赖 pandas 的交易日判断）

**关键设计模式**:
- DataSource 抽象：通过 ABC 定义接口，支持多种数据源切换
- 两阶段分离：信号计算（可并行）与交易执行（必须顺序）解耦
- 成本模型注入：`TransactionCost` 同时注入 Account 和 Executor，保持一致
- 策略作为函数而非类继承：`get_stock_signal` / `get_trade_signal` / `on_init` / `on_finish`



#### 1.2.7 OSkhQuant --- 开源桌面量化平台

**定位**: PyQt5 桌面应用，与 MiniQMT 深度集成，提供完整的回测 + 数据管理 + 可视化。

**可借鉴的设计思想**:
- **策略作为插件**：策略通过 `importlib.util.spec_from_file_location()` 运行时加载，框架通过 duck typing 调用 `init` / `khHandlebar` / `khPreMarket` / `khPostMarket`
- **Context dict 模式**：所有状态通过 rich dict 传递给策略回调，而非 OOP 继承
- **信号制交易**：策略返回 signal dict 列表，由 `KhTradeManager.process_signals()` 统一执行，分离策略逻辑与执行机制
- **Trigger 体系**：TickTrigger / KLineTrigger / CustomTimeTrigger 的工厂模式，支持不同频率回测
- **MyTT.py 指标库**：624 行纯 numpy 向量化技术指标实现，可作为因子系统的底层计算库参考
- **交易成本建模**：佣金、印花税（卖向）、过户费（沪市）、滑点（tick 和 ratio 两种模式）
- **数据补充调度器**：`GUIScheduler.py` 使用 `schedule` 库 + `multiprocessing.Process` 实现定时自动数据更新

**不适合直接复用的部分**:
- 整个 GUI 层（PyQt5）不适合服务端/命令行场景
- 强依赖 MiniQMT / xtquant 的 Windows 本地 API
- 回测引擎不支持 T+1 逻辑（`can_use_volume` 在回测中当日即可用）
- 没有因子系统、组合优化等高级功能

#### 1.2.8 alpha-gpt --- LLM 驱动因子生成

**定位**: 基于 LangGraph + GPT-4o 的 Human-AI 协作因子挖掘框架。

**可借鉴的设计思想**:
- **四阶段流水线**: user_input -> hypothesis_generator -> alpha_generator -> alpha_coder，将交易想法逐步转化为可执行代码
- **迭代反馈机制**: hypothesis_agent 在下一次迭代时读取前次回测指标（IR、年化收益、最大回撤、IC），引导 LLM 改进假设
- **因子数据库**: PostgreSQL 持久化 hypotheses / alphas / backtest_results 三张表，支持 thread_id 维度的迭代追踪
- **因子编码规范**: 输出格式为 Qlib 兼容的 pandas 代码（MultiIndex input -> MultiIndex output）
- **Checkpointer 设计**: LangGraph PostgresSaver + 自定义 AlphaGPTCheckpointer 双写，既支持状态恢复，也支持领域查询

**P0 缺失功能**:
- 无回测代码（BacktestResult 表存在但无计算逻辑）
- 无 PyGAD / 遗传规划（README 提到但未实现）
- `coded_alphas` 持久化有 bug（key 不匹配 `expr` vs `expression`）
- 无自动反馈闭环（需要外部调用者手动触发下一次 graph.invoke）

### 1.3 复用价值总结

| 模块 | 主要来源 | 复用方式 |
|------|----------|----------|
| 数据源抽象 | new-quant `core/data/base.py` | 直接复用 |
| BaoStock 接入 | myquant-strategy `baostock_backtest/data/` | 直接复用 |
| 数据同步服务 | myquant-strategy `sync_service.py` | 直接复用 |
| 复权处理 | new-quant `adjuster.py` + myquant-clean `xtdata.py` | 整合复用 |
| 数据缓存 | new-quant `cache.py` + myyquant-1116 因子缓存 | 整合复用 |
| 回测引擎 | new-quant `core/engine/` + myquant-clean `backtrade_v2/` | 整合重构 |
| 因子计算 | myquant-clean `dataset/alpha101.py` + OSkhQuant `MyTT.py` | 直接复用 |
| 因子挖掘 | alpha-gpt 流水线架构 + myquant-clean `auto_factor_mining/` | 重构复用 |
| 参数消融 | myquant-strategy `auto_ablate_*.py` | 重构复用 |
| 公式引擎 | myquant-strategy `baostock_clean/formulas.py` | 直接复用 |
| 策略模板 | myquant-strategy `templates.py` + `template_strategy.py` | 重构复用 |
| 交易日历 | 需新建 | 新建 |
| 组合优化 | 需新建 | 新建 |
| 绩效评估 | new-quant `account.py` 指标计算 | 直接复用 |



---

## 2. 设计目标与原则

### 2.1 核心目标

| 目标 | 定义 | 优先级 |
|------|------|--------|
| **真实性** | 回测结果能真实反映策略在实盘中的表现，避免未来函数、幸存者偏差等问题 | P0 |
| **准确性** | 撮合逻辑、成本计算、复权处理、收益率计算等数值精确 | P0 |
| **可维护性** | 模块职责清晰，新增功能不破坏已有代码，依赖关系可控 | P0 |
| **解耦性** | 数据层、因子层、策略层、组合层、执行层、回测层、评估层相互独立 | P0 |
| **高效性** | 单机回测在合理时间内完成，支持增量计算和缓存 | P1 |

### 2.2 设计原则

1. **面向接口编程**：每个模块定义 ABC 抽象基类，具体实现可替换。例如 DataSource 定义接口，BaoStockDataSource / CSVDataSource / XTDataSource 分别实现
2. **配置驱动**：所有可变参数（数据源参数、回测参数、策略参数、因子参数）通过 YAML/JSON 配置文件注入，不硬编码
3. **不可变数据流**：原始数据一经加载和复权后即不可变，因子计算、回测执行均基于不可变快照，避免副作用
4. **预计算优先**：因子计算、信号生成等可在回测循环前完成的工作，统一在预处理阶段完成
5. **分层缓存**：从原始数据 -> 复权后数据 -> 因子值 -> 信号 -> 回测结果，每一层都可独立缓存
6. **独立可验证**：每个模块必须能独立测试，不依赖其他模块的完整实现
7. **渐进式复杂度**：Phase 1 只做最简单的可用版本，Phase 2 和 Phase 3 逐步增加高级特性

### 2.3 真实性保障措施

| 问题 | 如何保障 |
|------|----------|
| 未来函数 | T+1 延迟执行：T 日决策只用 <= T-1 日数据；因子计算使用 `shift(1)` 延迟 |
| 幸存者偏差 | 股票池包含已退市股票；数据仓库保留完整历史（含退市后数据） |
| 复权错误 | 使用前复权（forward adjustment），在数据加载阶段一次性完成；支持 origin 和 equiv 双模式校验 |
| 交易成本遗漏 | 建模佣金（万三，最低 5 元）、印花税（卖向千一）、过户费（沪市十万分之二）、滑点（可配置） |
| 撮合失真 | 处理涨跌停不可交易、停牌不可交易、价格跳变保护、T+1 卖出限制 |
| 基准偏差 | 使用全收益指数（含分红再投资）作为基准，而非价格指数 |

---

## 3. 总体架构设计

### 3.1 模块划分

```
quantx/
├── core/                    # 核心引擎
│   ├── data/                # 数据层
│   │   ├── base.py          # DataSource ABC
│   │   ├── baostock_source.py  # BaoStock 数据源（Phase 1）
│   │   ├── csv_source.py    # 本地 CSV 数据源
│   │   ├── adjuster.py      # 复权处理
│   │   ├── repository.py    # 本地数据仓库
│   │   ├── sync_service.py  # 数据同步服务
│   │   ├── calendar.py      # 交易日历
│   │   └── cache.py         # 缓存层
│   ├── factor/              # 因子层
│   │   ├── base.py          # Factor ABC + FactorRegistry
│   │   ├── registry.py      # 因子注册中心
│   │   ├── engine.py        # 因子计算引擎
│   │   ├── cache.py         # 因子缓存
│   │   ├── operators.py     # 算子库（MA/EMA/STD/RANK/...）
│   │   ├── alpha101.py      # WorldQuant 101 因子
│   │   ├── technical.py     # 技术指标因子
│   │   └── fundamental.py   # 基本面因子
│   ├── engine/              # 回测引擎
│   │   ├── context.py       # 回测上下文（数据快照 + 因子访问）
│   │   ├── engine.py        # 两阶段回测引擎
│   │   ├── account.py       # 账户管理
│   │   ├── executor.py      # 订单执行器
│   │   ├── cost.py          # 交易成本模型
│   │   └── signals.py       # 信号矩阵
│   ├── strategy/            # 策略层
│   │   ├── base.py          # Strategy ABC
│   │   ├── loader.py        # 策略动态加载器
│   │   └── templates.py     # 声明式策略模板
│   ├── portfolio/           # 组合层
│   │   ├── base.py          # Portfolio ABC
│   │   ├── equal_weight.py  # 等权组合
│   │   ├── risk_parity.py   # 风险平价
│   │   └── optimizer.py     # 组合优化器
│   └── analysis/            # 评估层
│       ├── metrics.py       # 绩效指标（Sharpe/Sortino/Calmar/...）
│       ├── attribution.py   # 收益归因（Brinson/因子归因）
│       ├── report.py        # 回测报告生成
│       └── plot.py          # 可视化（净值曲线/热力图/...）
├── strategies/              # 用户策略目录
├── configs/                 # 配置文件
├── data/                    # 本地数据存储
│   ├── raw/                 # 原始数据
│   ├── stocks/              # 按股票存储的 CSV
│   ├── universe/            # 股票池
│   └── cache/               # 缓存数据
├── tests/                   # 测试目录
├── tools/                   # CLI 工具
│   ├── sync_data.py         # 数据同步命令
│   ├── run_backtest.py      # 回测命令
│   └── run_factor.py        # 因子计算命令
├── docs/                    # 文档
├── pyproject.toml           # 项目元数据
└── README.md
```

### 3.2 模块职责与依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                        configs/                              │
│              (YAML 配置驱动所有模块)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  data/   │───>│ factor/  │───>│ engine/  │───>│analysis/ │
│ 数据层    │    │ 因子层    │    │ 回测层    │    │ 评估层    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
      │               │               │               │
      │               │        ┌──────┴──────┐        │
      │               │        │  strategy/  │        │
      │               │        │  portfolio/ │        │
      │               │        └─────────────┘        │
      │               │               │               │
      ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                       tests/                                 │
│           (每个模块独立测试，集成测试跨模块验证)                  │
└─────────────────────────────────────────────────────────────┘
```

**数据流**:
1. `data/` 从外部数据源拉取原始数据，执行复权，存入本地仓库
2. `factor/` 从 `data/` 读取复权后的 OHLCV 数据，计算因子值，缓存结果
3. `engine/` 从 `data/` 读取行情数据，从 `factor/` 读取因子值，构建回测上下文
4. `strategy/` 使用 `engine/` 提供的上下文生成买卖信号
5. `portfolio/` 根据信号和因子值计算持仓权重
6. `engine/executor.py` 执行订单，`engine/account.py` 更新账户状态
7. `analysis/` 读取回测结果，计算绩效指标，生成报告

**依赖规则**:
- 数据层不依赖任何其他模块
- 因子层只依赖数据层
- 回测层依赖数据层和因子层
- 策略层、组合层依赖回测层提供的接口
- 评估层依赖回测层的输出
- 所有模块不循环依赖

---

## 4. 数据层设计

### 4.1 数据源抽象

DataSource 抽象基类定义了统一的数据访问接口，所有数据源实现必须遵循此接口：

- `get_daily_bars(symbols, start_date, end_date, fields, adjust)` -> MultiIndex DataFrame
- `get_dividend_factors(symbols)` -> DataFrame
- `get_stock_list(date)` -> List[str]
- `get_benchmark(benchmark, start_date, end_date)` -> DataFrame
- `get_trading_dates(start_date, end_date)` -> List[str]

接口设计思路：
- 使用 ABC 抽象基类，确保所有数据源实现一致的方法签名
- 返回统一的数据结构：`(date, symbol)` MultiIndex DataFrame
- 支持 `adjust` 参数控制复权方式（forward / none / backward）
- 数据源工厂函数 `create_data_source(config)` 根据配置创建对应实例

### 4.2 BaoStock 优先接入方案

Phase 1 实现 `BaoStockDataSource(DataSource)`，直接复用以下现有代码：

| 复用模块 | 来源 | 用途 |
|----------|------|------|
| `baostock_client.py` | myquant-strategy | API 封装、登录管理、日期回溯、重试 |
| `repository.py` | myquant-strategy | 本地 CSV 存储、按 stocks/{code}.csv 布局 |
| `sync_service.py` | myquant-strategy | 多线程增量同步、ensure_coverage |
| `update_all_stocks.py` | myquant-strategy | 定时 Job、文件锁、失败重试、报告 |

**关键配置**:

```yaml
# configs/data/baostock.yaml
data:
  source: "baostock"
  root: "./data"
  adjust: "forward"
  update:
    mode: "incremental"
    auto: true
    schedule: "0 15 * * 1-5"
  sync:
    workers: 8
    pause_seconds: 0.5
    max_retries: 3
```

### 4.3 后续数据源扩展

扩展方式：实现 `DataSource` 接口即可，已有模块无需修改。

未来可扩展的数据源：
- `XTDataSource`：MiniQMT / xtquant
- `TushareDataSource`：Tushare Pro
- `AKShareDataSource`：AKShare
- `CSVDataSource`：纯本地 CSV（已有 new-quant 实现）

数据源工厂通过配置中的 `source` 字段动态选择实现：
```python
def create_data_source(config: dict) -> DataSource:
    source_type = config['data']['source']
    if source_type == 'baostock':
        return BaoStockDataSource(config)
    elif source_type == 'xtquant':
        return XTDataSource(config)
    # ...
```

### 4.4 数据落地、更新、校验、版本管理

**数据落地布局**:
```
data/
  raw/baostock/stocks/     # 原始股票数据
  raw/baostock/universe/   # 股票池
  processed/cache/         # 处理后数据缓存（MultiIndex pickle）
  factors/                 # 因子缓存（按日期范围分子目录）
```

**数据校验规则**:
1. 完整性检查：每个交易日期望有多少只股票有数据
2. 连续性检查：OHLC 价格关系（low <= min(open, close), high >= max(open, close)）
3. 复权一致性检查：前复权价格序列不应出现价格跳跃（除权除息日除外）
4. 停牌检查：停牌日应有标记
5. 涨跌停检查：当日涨跌幅在允许范围内

**数据版本管理**:
- 缓存文件元数据旁路（.meta.json）：记录数据源、日期范围、股票列表哈希、复权方式、生成时间
- 数据更新触发缓存失效：当数据源有新数据或股票池变更时，缓存自动失效
- 支持多版本数据并存：不同日期范围、不同股票池的缓存独立存储

---

## 5. 因子系统设计

### 5.1 因子定义方式

因子采用统一的 ABC 抽象基类 + 元信息 dataclass 定义：

- `FactorMeta`：因子元信息（名称、分类、频率、依赖字段、是否可缓存、是否横截面、是否市场级、描述、版本）
- `Factor.compute(data)` -> pd.Series：核心计算逻辑，输入 MultiIndex DataFrame，输出因子值
- `Factor.validate(result)` -> bool：因子值校验（非空、无异常值、覆盖率检查）

### 5.2 因子注册机制

设计统一的 `FactorRegistry` 注册中心（单例模式），支持三种注册方式：

1. **装饰器注册**（推荐，代码即注册）：
   `@register_factor(name='alpha005', category='momentum', ...)`
2. **YAML 配置注册**（批量管理）：
   在 `configs/factors/` 下用 YAML 描述因子，程序启动时自动注册
3. **自动发现注册**（扫描目录）：
   `FactorRegistry.auto_discover('quantx.core.factor')`
   扫描指定包下所有 Factor 子类，自动注册

注册时自动校验：
- 因子名唯一性
- 元信息完整性（必填字段检查）
- 依赖字段是否在数据源可用字段中
- 因子分类是否在预定义分类中

### 5.3 因子依赖管理

`FactorDependencyGraph` 管理因子间的依赖关系：

- 构建依赖图：扫描所有已注册因子的 `depends_on` 字段
- 拓扑排序：确定因子计算顺序，确保依赖先计算
- 变更影响分析：当某个因子变更时，找出所有受影响的下游因子
- 依赖解析规则：
  - 依赖原始字段：`depends_on = ['close', 'volume']`
  - 依赖其他因子：`depends_on = ['factor:alpha005']`
  - 依赖市场级因子：`depends_on = ['factor:market_breadth']`

### 5.4 因子自动化新增方案

**设计目标**：新增一个因子只需 3 步，无需修改任何已有代码。

**流程**：
1. 编写因子类（1 个文件，放在 `core/factor/` 或 `strategies/my_factors/`）
2. 注册因子（通过装饰器自动注册，或 YAML 配置批量注册）
3. 运行校验（自动检查覆盖率、IC 显著性、无异常值）
4. 投入使用（可在回测的因子配置中引用）

**借鉴 alpha-gpt 的自动化思路**：
- 因子描述 + 表达式 -> LLM 自动生成代码
- 自动校验流水线：因子值覆盖率 > 80%、无 inf/nan、IC 显著性检验
- 自动回测：新因子生成后自动运行 baseline 策略回测，评估因子效果

### 5.5 慢因子的优化策略

#### 5.5.1 因子分层架构

| Level | 类型 | 计算量 | 并行策略 | 示例 |
|-------|------|--------|----------|------|
| 0 | 原始字段 | 无 | 无 | open, high, low, close, volume |
| 1 | 单股票时序 | O(N*T) | 按股票并行 | MA(20), RSI(14), MACD |
| 2 | 横截面因子 | O(N*T) | 按日期并行 | RANK(close), ZSCORE(return) |
| 3 | 行业级因子 | O(I*N*T) | 按日期+行业并行 | 行业平均PE, 行业内排名 |
| 4 | 市场级因子 | O(N*T) | 按日期并行 | 市场宽度, 中位数收益率 |

#### 5.5.2 预计算机制

`FactorPrecomputer` 在回测前批量计算所有因子：
1. 解析依赖图，拓扑排序
2. 按 Level 分层计算，同层因子并行
3. 结果写入缓存，下游因子可直接复用
4. 使用 ThreadPoolExecutor 实现并行

#### 5.5.3 缓存机制

`FactorCache` 按 `(factor_name, start_date, end_date, stock_list_hash)` 作为缓存键：
- 缓存格式：pickle 文件 + 旁路 .meta.json
- 缓存粒度：按因子独立缓存，支持按需加载
- 缓存失效：日期范围变更或股票列表变更时自动重算

#### 5.5.4 增量更新机制

`IncrementalFactorUpdater` 支持增量更新：
1. 从缓存读取已有因子值
2. 只计算新增日期范围的因子值（头部扩展 + 尾部扩展）
3. 拼接已有结果 + 新计算结果
4. 更新缓存

#### 5.5.5 并行化方案

- Level 1 因子：按股票并行，使用 ThreadPoolExecutor
- Level 2 因子：按日期并行
- Level 3/4 因子：先按日期分组，组内并行聚合
- 跨因子并行：同 Level 的因子可以并行计算
- 使用 pandas 向量化操作，避免 Python 循环

#### 5.5.6 回测时的复用

回测引擎在初始化时：
1. 检查所需因子是否已缓存 -> 直接读取
2. 未缓存但可缓存 -> 触发预计算 + 缓存
3. 已缓存但日期范围不完全匹配 -> 增量更新
4. 不可缓存因子 -> 在回测循环中实时计算

---

## 6. 回测系统设计

### 6.1 回测流程概述

回测系统采用**两阶段分离**架构（借鉴 new-quant 和 myquant-clean backtrade_v2）：

**Phase 1: 信号预计算阶段**（可并行）
1. 加载数据快照（MultiIndex DataFrame，含复权后 OHLCV + 因子值）
2. 遍历每个交易日，调用 `strategy.get_stock_signal(context, date)`
3. 使用 ThreadPoolExecutor 并行计算所有日期的信号
4. 结果存入稀疏信号矩阵（只存有信号的日期+股票，降低内存）

**Phase 2: 交易执行阶段**（必须顺序）
1. 按交易日顺序遍历
2. 对每个交易日：
   a. 获取当日信号（从信号矩阵读取）
   b. 调用 `strategy.get_trade_signal(context, account)` 生成订单
   c. Executor 校验订单（停牌检查、涨跌停检查、现金检查）
   d. Executor 执行订单（应用滑点、计算成本）
   e. Account 更新持仓和资金
   f. Account 记录每日净值
3. 回测结束后，计算绩效指标

### 6.2 交易日历

`TradingCalendar` 模块：
- 从数据源获取交易日列表
- 支持检验某日是否为交易日
- 支持获取 N 个交易日前/后的日期
- 支持获取两个日期之间的交易日数量
- 初期基于 BaoStock 的交易日数据，后续可接入 exchange_calendars 库

### 6.3 调仓逻辑

调仓信号由策略层产生，回测引擎只负责执行：

**调仓频率支持**：
- 每日调仓
- 固定间隔调仓（如每 5 个交易日）
- 固定日期调仓（如每月第一个交易日）
- 自定义调仓日历

**调仓流程**：
1. 检查是否为调仓日
2. 获取卖出信号，生成卖出订单
3. 释放资金后，获取买入信号，生成买入订单
4. 根据组合层权重分配资金
5. 按优先级排序执行订单（先卖后买）

### 6.4 撮合逻辑

`Executor` 负责订单校验和执行（借鉴 new-quant 的 executor.py）：

**订单校验规则**：
- 停牌检查：停牌股票不可交易
- 涨跌停检查：涨停不可买入，跌停不可卖出
- 价格跳变保护：当日开盘价与前收盘价偏差 > 9.5% 时跳过买入
- 现金检查：买入所需资金 > 可用现金时，自动调整买入数量
- T+1 检查：当日买入的股票不可当日卖出

**订单执行**：
- 成交价格 = 信号价格 * (1 + slippage)（买入）或 * (1 - slippage)（卖出）
- 按整手（100 股）取整
- 记录成交时间、成交价格、成交数量、交易成本

### 6.5 手续费、滑点、停牌、涨跌停、复权等真实性处理

**交易成本 `TransactionCost`**（借鉴 new-quant cost.py + OSkhQuant khTrade.py）：

| 费用项 | 费率 | 适用方向 | 说明 |
|--------|------|----------|------|
| 佣金 | 0.03% | 买卖双向 | 最低 5 元 |
| 印花税 | 0.05% | 仅卖出 | 2024 年 A 股降税后费率 |
| 过户费 | 0.002% | 买卖双向 | 仅沪市（sh. 前缀） |
| 滑点 | 0.1% | 买卖双向 | 可配置 tick 或 ratio 模式 |

**停牌处理**：
- 数据加载时标记停牌日（volume=0 或价格连续不变）
- 停牌股票在当日不可交易
- 持仓中的停牌股票按停牌前最后价格估值
- 复牌首日处理：检查价格跳变，必要时跳过交易

**涨跌停处理**：
- 板别感知阈值（借鉴 new-quant context.py）：
  - 主板：10%
  - 创业板（300/301）：20%
  - 科创板（688）：20%
  - 北交所：30%
- 涨停股票不可买入，跌停股票不可卖出
- 一字板（开盘即涨停/跌停）：全天无成交机会

**复权处理**：
- 使用前复权（forward adjustment），在数据加载阶段一次性完成
- 支持 `origin` 模式（仿射变换 a*price+b）和 `equiv` 模式（纯乘法等价调整）
- 两种模式可交叉校验，确保复权正确性

### 6.6 绩效评估

`Analysis` 模块（借鉴 new-quant account.py 的指标计算）：

- 总收益率、年化收益率
- 最大回撤、回撤持续期
- Sharpe 比率（无风险利率 2%）
- Sortino 比率
- Calmar 比率
- 胜率（FIFO 匹配法）
- 日均换手率
- 总交易次数、总交易成本
- 与基准对比：超额收益、Beta、Alpha、信息比率
- 月度收益热力图
- 收益归因（Brinson 归因 / 因子归因）

---

## 7. 性能优化设计

### 7.1 缓存体系

分层缓存策略，从底层到上层：

| 缓存层 | 内容 | 格式 | 失效条件 | 来源 |
|--------|------|------|----------|------|
| L0 原始数据缓存 | 本地 CSV 文件 | CSV | 数据源有新数据 | myquant-strategy repository |
| L1 复权后数据缓存 | MultiIndex DataFrame | pickle + meta.json | 股票池变更/日期变更 | new-quant cache.py |
| L2 因子值缓存 | 按因子独立存储 | pickle + CSV | 因子代码变更/日期扩展 | myyquant-1116 因子缓存 |
| L3 信号矩阵缓存 | 稀疏信号矩阵 | pickle | 因子变更/策略参数变更 | myquant-clean backtrade_v2 |
| L4 回测结果缓存 | 账户净值序列 | JSON | 策略/参数变更 | 新增 |

缓存键设计：MD5 哈希，包含所有影响结果的参数

### 7.2 增量计算

增量计算策略：
1. **数据增量同步**：只拉取新增日期或新增股票的数据（myquant-strategy sync_service）
2. **因子增量更新**：只计算新增日期范围的因子值，拼接已有缓存（见 5.5.4）
3. **信号增量更新**：数据/因子变更时，只重新计算受影响日期的信号
4. **回测增量运行**：参数变更时，只重跑受影响的部分（如仅变更卖出规则，可复用买入信号）

### 7.3 预计算

预计算策略：
1. **因子预计算**：在回测前批量计算所有需要的因子，结果持久化（见 5.5.2）
2. **信号预计算**：Phase 1 阶段并行计算所有日期的信号，存入稀疏矩阵
3. **涨跌停/停牌预计算**：在 Context 加载时一次性构建 `_limit_up_map`、`_limit_down_map`、`_suspended_map`
4. **行业分类预计算**：在数据加载时一次性构建股票到行业的映射

### 7.4 分层存储

数据存储分层：
- 热数据（回测中频繁访问）：内存中的 MultiIndex DataFrame，使用 pd.IndexSlice 快速查询
- 温数据（可能复用）：pickle 缓存文件，快速加载
- 冷数据（原始归档）：本地 CSV 文件，按需读取

### 7.5 并行化建议

| 场景 | 并行策略 | 注意事项 |
|------|----------|----------|
| 数据加载 | ThreadPoolExecutor 按股票并行 | 注意 API 限流 |
| 因子计算 | ThreadPoolExecutor 按因子+股票并行 | 依赖因子先计算 |
| 信号计算 | ThreadPoolExecutor 按日期并行 | 每日期独立，天然并行 |
| 回测执行 | 顺序执行（不可并行） | 每日期依赖前一日状态 |
| 参数搜索 | ProcessPoolExecutor 按参数组合并行 | 每组合独立回测 |
| 多策略对比 | ProcessPoolExecutor 按策略并行 | 每策略独立回测 |

### 7.6 向量化与内存优化

- 使用 pandas 向量化操作替代 Python 循环（如 `df.groupby('date').apply(...)`）
- 使用 numpy 数组操作替代逐元素计算
- 信号矩阵稀疏存储（只存 selected=True 的行）
- 使用 `category` dtype 压缩股票代码和行业分类列
- 大数据集使用 `float32` 替代 `float64`（精度损失可控）
- 及时释放中间计算结果（`del df; gc.collect()`）

---

## 8. 测试与验证工作流设计

### 8.1 核心设计理念

将系统拆分为多个子功能模块，每个模块都能独立验证。
测试策略：**单元测试 -> 模块集成测试 -> 端到端回归测试 -> 业务逻辑验证**

### 8.2 子模块拆分与测试组织

| 模块 | 子功能 | 测试脚本 | 验证方法 |
|------|--------|----------|----------|
| data/ | 数据源接口 | tests/data/test_baostock_source.py | Mock API 响应，验证数据格式 |
| data/ | 复权处理 | tests/data/test_adjuster.py | 已知除权案例，验证前后价格 |
| data/ | 数据同步 | tests/data/test_sync_service.py | 增量更新后数据一致性 |
| data/ | 交易日历 | tests/data/test_calendar.py | 验证已知交易日/非交易日 |
| data/ | 缓存层 | tests/data/test_cache.py | 缓存读写、失效、MD5 键正确性 |
| factor/ | 因子注册 | tests/factor/test_registry.py | 注册、去重、自动发现 |
| factor/ | 因子计算 | tests/factor/test_factors.py | 与已知值对比（alpha005 等） |
| factor/ | 因子依赖 | tests/factor/test_dependency.py | 拓扑排序、变更影响分析 |
| factor/ | 因子缓存 | tests/factor/test_cache.py | 增量更新、缓存失效 |
| engine/ | 回测引擎 | tests/engine/test_engine.py | 合成数据，验证两阶段流程 |
| engine/ | 账户管理 | tests/engine/test_account.py | 买卖、成本计算、持仓估值 |
| engine/ | 订单执行 | tests/engine/test_executor.py | 停牌/涨跌停/现金检查 |
| engine/ | 交易成本 | tests/engine/test_cost.py | 佣金、印花税、过户费计算 |
| strategy/ | 策略加载 | tests/strategy/test_loader.py | 动态加载、参数注入 |
| portfolio/ | 组合优化 | tests/portfolio/test_optimizer.py | 等权、风险平价权重计算 |
| analysis/ | 绩效指标 | tests/analysis/test_metrics.py | 与手工计算对比 |
| integration/ | 端到端 | tests/integration/test_e2e.py | 合成数据完整回测流程 |

### 8.3 每类功能的验证方法

#### 8.3.1 数据更新正确性验证

验证要点：
- OHLC 价格关系一致性：low <= min(open, close), high >= max(open, close)
- 复权一致性：origin 和 equiv 两种模式的价格比值应接近
- 无幸存者偏差：股票池应包含已退市股票
- 数据完整性：每个交易日应有足够数量的股票有数据
- 复权无跳跃：除权除息日外的价格变化应连续

#### 8.3.2 因子值正确性验证

验证要点：
- 与已知值对比：alpha005 等标准因子与手工计算的参考值对比
- 因子覆盖率：因子值覆盖率 > 80%
- 无未来函数：T 日因子值不应包含 T+1 日信息（截断测试）
- 无异常值：因子值无 inf/nan（或 nan 可解释为停牌等）
- IC 显著性：因子值与未来收益的 IC 应在合理范围内

#### 8.3.3 回测撮合结果正确性验证

验证要点：
- T+1 结算：当日买入的股票不可当日卖出
- 涨停不可买入：买入价等于涨停价时订单应被拒绝
- 跌停不可卖出：卖出价等于跌停价时订单应被拒绝
- 交易成本计算准确性：佣金、印花税、过户费的手工验证
- 停牌不可交易：持仓中的停牌股票按停牌前最后价格估值
- 价格跳变保护：开盘价与前收盘价偏差过大时跳过交易

#### 8.3.4 收益归因与绩效统计一致性验证

验证要点：
- 总收益率 = 累计净值 - 1
- 现金 + 持仓市值 = 总资产（每个交易日）
- 交易日志中的买卖变化与持仓变化一致
- 每日收益率 = (当日总资产 - 前日总资产 - 净入金) / 前日总资产
- 累计收益率 = 每日收益率的复利累积
- Sharpe 比率用日收益率手工计算的结果与系统计算结果一致

### 8.4 项目级开发工作流

```
1. 功能开发
   ├── 编写模块代码
   ├── 编写单元测试
   ├── 运行单元测试: pytest tests/module/
   └── 代码审查

2. 集成验证
   ├── 运行集成测试: pytest tests/integration/
   ├── 运行回归测试: pytest tests/ --cov=quantx
   └── 检查覆盖率: >= 80%

3. 业务验证
   ├── 数据验证: python tools/validate_data.py
   ├── 因子验证: python tools/validate_factors.py
   ├── 回测验证: python tools/validate_backtest.py
   └── 一致性检查: python tools/check_consistency.py

4. 发布前检查
   ├── 全量测试: pytest tests/
   ├── 类型检查: mypy quantx/
   ├── 代码风格: ruff check quantx/
   └── 文档更新
```

---

## 9. 分阶段落地路线图

### 9.1 Phase 1: 最小可用系统（MVP）

**目标**: 搭建核心框架，能跑通完整的回测流程

**范围**:
- 数据层：BaoStock 数据源接入 + 本地 CSV 存储 + 复权处理
- 因子层：Alpha101 因子（移植 myquant-clean + OSkhQuant MyTT）+ 因子注册 + 因子缓存
- 回测层：两阶段引擎（移植 new-quant）+ 账户管理 + 执行器 + 成本模型
- 策略层：策略加载器 + 声明式策略模板
- 评估层：基础绩效指标（Sharpe/Sortino/Calmar/MaxDD/收益率）+ 净值曲线可视化
- 工具链：CLI 数据同步命令 + CLI 回测命令
- 测试：核心模块的单元测试 + 合成数据端到端集成测试

**验收标准**:
- 使用 BaoStock 真实数据，跑通一次完整的回测（2020-2025，全 A 股）
- 回测结果与 myquant-strategy 相同策略的结果偏差 < 1%
- 单次回测（5 年日线，全 A 股，5 个因子）耗时 < 5 分钟
- 单元测试覆盖率 >= 80%

**风险点**:
- BaoStock API 稳定性（有重试机制兜底）
- 复权处理正确性（origin 和 equiv 两种模式交叉校验）
- 新架构与旧代码的兼容性（通过一致性测试验证）

**预计工期**: 4-6 周

### 9.2 Phase 2: 功能完善与性能优化

**目标**: 完善因子系统和回测功能，提升性能

**范围**:
- 因子层：市场级/行业级因子支持 + 因子依赖图 + 增量更新 + 因子预计算引擎
- 回测层：更完善的撮合逻辑（一字板处理、T+1 卖出限制、更精细的滑点模型）
- 组合层：等权组合 + 风险平价 + 组合优化器
- 评估层：收益归因（Brinson）+ 因子归因 + 月度热力图 + 回测报告生成
- 性能优化：分层缓存体系 + 增量计算 + 并行化 + 向量化优化
- 工具链：参数搜索工具（消融框架）+ 因子挖掘工具（LLM 驱动）
- 数据源扩展：Tushare / AKShare 数据源接入

**验收标准**:
- 支持 100+ 因子同时回测（含市场级/行业级因子），耗时 < 10 分钟
- 因子增量更新时间 < 全量计算的 20%
- 参数搜索（10 个参数组合）可并行完成
- 归因分析结果与手工拆解一致

**风险点**:
- 慢因子计算性能可能不达标（需持续优化并行策略）
- 缓存失效策略可能过于激进或保守（需在实际使用中调优）
- 因子依赖图可能变得复杂（需限制依赖深度）

**预计工期**: 6-8 周

### 9.3 Phase 3: 高级特性与生态建设

**目标**: 增加高级量化研究能力，完善工具链和生态

**范围**:
- 因子挖掘：完整的 LLM 驱动因子挖掘流水线（借鉴 alpha-gpt LangGraph 架构）
- 强化学习：RL 因子权重优化（移植 myquant-rl）
- 多频率回测：支持分钟级回测 + 日级回测切换
- 实盘对接：模拟交易信号输出 + 实盘交易接口（MiniQMT 等）
- 可视化：Web 可视化面板（回测报告、因子分析、组合监控）
- 文档与教程：完整的 API 文档 + 策略开发教程 + 因子开发教程

**验收标准**:
- LLM 因子挖掘流水线可自动生成、编码、回测、评估因子
- 分钟级回测与日级回测结果在相同策略下逻辑一致
- 实盘信号输出与回测信号无偏差

**风险点**:
- LLM 生成因子质量不稳定（需人工审核机制）
- 分钟级回测的撮合逻辑更复杂（需重新设计部分执行器）
- 实盘对接的可靠性要求高（需充分测试）

**预计工期**: 8-12 周

### 9.4 后续持续优化方向

- 多资产支持（期货、ETF、可转债）
- 分布式回测（多机并行参数搜索）
- 实时因子监控与预警
- 策略风险管理系统（VaR、压力测试、情景分析）
- 社区贡献机制（因子市场、策略市场）

---

## 10. 风险点与设计取舍

### 10.1 容易做错的地方

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 复权处理错误 | 回测结果完全失真 | origin 和 equiv 双模式交叉校验；已知除权案例自动化测试 |
| 未来函数 | 回测虚高，实盘大幅回撤 | T+1 延迟执行；截断测试验证；因子层 shift(1) 延迟 |
| 幸存者偏差 | 高估策略收益 | 数据仓库保留退市股票全量历史数据；股票池按历史时点快照 |
| 撮合逻辑失真 | 策略容量高估 | 涨停/跌停/停牌 不可交易；价格跳变保护；滑点建模 |
| 交易成本遗漏 | 收益率虚高 | 佣金+印花税+过户费+滑点 完整建模；自动化测试验证 |
| 因子计算性能 | 回测耗时过长 | 分层缓存+增量更新+预计算+并行化；性能基准测试 |
| 缓存失效逻辑错误 | 使用过期数据 | MD5 哈希键包含所有影响因子；meta.json 记录参数；缓存版本管理 |
| 过度抽象 | 开发效率降低 | Phase 1 只做必要抽象；ABC 接口保持最小化；渐进式增加复杂度 |

### 10.2 需要优先保守设计的地方

1. **数据层接口**：DataSource ABC 的接口定义需要仔细设计，因为后续所有模块都依赖它。接口应保持稳定，避免频繁变更。
2. **复权处理**：这是回测准确性的根基。优先实现 origin 和 equiv 双模式并交叉校验。
3. **交易日历**：看似简单但影响广泛（回测循环、因子计算、绩效统计）。需要独立模块，充分测试。
4. **因子注册机制**：需要在 Phase 1 就设计好，因为后续因子会快速增加。注册中心应支持热加载和版本管理。
5. **回测引擎的 T+1 逻辑**：这是防止未来函数的核心机制。需要在引擎层面强制执行，不允许策略层绕过。

### 10.3 可以后续再优化的地方

1. **多频率回测**：Phase 1 只做日线，分钟级回测放到 Phase 3
2. **多资产支持**：Phase 1 只做 A 股，期货/ETF/可转债后续扩展
3. **分布式回测**：Phase 1 只做单机，参数搜索并行化 Phase 2 开始
4. **Web 可视化**：Phase 1 用 matplotlib 图表 + 命令行输出，Web 面板 Phase 3
5. **实盘对接**：Phase 1-2 仅回测，实盘交易接口 Phase 3
6. **高级因子挖掘**：Phase 1 手动编写因子，LLM 驱动挖掘 Phase 3
7. **组合优化器**：Phase 1 用等权组合，风险平价/均值方差 Phase 2
8. **收益归因**：Phase 1 只做基础绩效指标，Brinson 归因 Phase 2

### 10.4 关键设计取舍

**取舍 1: 策略接口用函数还是类？**
- 决策：用函数（借鉴 new-quant 的 loader.py 和 OSkhQuant 的 duck typing）
- 理由：策略开发者只需实现 `get_stock_signal` 和 `get_trade_signal` 两个函数，无需理解类继承体系。动态加载器支持任意路径。
- 代价：无法强制接口约束（可通过加载时校验函数签名来弥补）

**取舍 2: 因子用 Python 类还是 YAML 配置？**
- 决策：简单因子用 YAML 配置 + 公式引擎，复杂因子用 Python 类
- 理由：借鉴 myquant-strategy 的 FormulaEngine + indicator_library 设计，简单因子（如 MA(close, 20)）用表达式声明，复杂因子（如 Alpha101）用 Python 实现
- 代价：需要维护两套因子定义方式（可通过统一的 FactorMeta 和注册机制统一管理）

**取舍 3: 数据存储用 CSV 还是数据库？**
- 决策：Phase 1-2 用 CSV 文件 + pickle 缓存，Phase 3 考虑引入 SQLite/Parquet
- 理由：CSV 简单透明，便于调试和数据校验；pickle 缓存提供快速加载。已有 myquant-strategy 的 CSV 仓库实现可直接复用。
- 代价：CSV 的查询和过滤效率不如数据库（可通过 MultiIndex DataFrame 缓存弥补）

**取舍 4: 并行化用多线程还是多进程？**
- 决策：数据加载和因子计算用 ThreadPoolExecutor，参数搜索用 ProcessPoolExecutor
- 理由：数据加载和因子计算是 IO 密集型 + pandas 向量化（GIL 可释放），线程开销小。参数搜索需要独立 Python 进程，避免 GIL 竞争。
- 代价：多进程需要注意 picklable 和内存开销（每进程独立加载数据）

---

## 11. 假设前提

以下假设是本方案设计的基础，如果假设不成立，需要调整方案：

1. **数据源可用性**：BaoStock 在 2025-2026 年期间持续可用，API 接口不发生重大变更
2. **单机运行环境**：Phase 1-2 在单机 Linux 环境下运行，内存 >= 16GB，CPU >= 8 核
3. **A 股市场规则**：T+1 结算、涨跌停板（10%/20%/30%）、印花税（卖向 0.05%）、过户费（沪市 0.002%）等规则在 Phase 1-3 期间不发生重大变更
4. **数据规模**：全 A 股（约 5000 只股票）x 10 年日线数据（约 2500 个交易日），数据总量约 500MB-1GB
5. **因子数量**：Phase 1 支持 50-100 个因子，Phase 2 支持 200-500 个因子
6. **回测频率**：Phase 1 仅支持日线回测，不涉及分钟级或 tick 级回测
7. **Python 版本**：Python >= 3.10，依赖 pandas >= 2.0, numpy >= 1.24
8. **用户场景**：策略研究和个人量化分析，非生产级交易系统（Phase 1-2）
9. **BaoStock 数据覆盖**：日线数据覆盖沪深两市全部 A 股，含已退市股票的历史数据
10. **复权方式**：默认使用前复权（forward adjustment），在数据加载阶段一次性完成

---

## 附录 A: 后续阶段文档结构预留

本次总体方案确立后，后续将按以下结构展开详细设计文档：

```
docs/
├── QUANTX_DESIGN.md              # 本文件：总体架构设计方案
├── phase1/
│   ├── 01_data_layer.md          # 数据层详细设计
│   ├── 02_factor_system.md       # 因子系统详细设计
│   ├── 03_backtest_engine.md     # 回测引擎详细设计
│   ├── 04_strategy_layer.md      # 策略层详细设计
│   ├── 05_api_reference.md       # API 接口文档
│   └── 06_implementation_plan.md # 实施计划与任务分解
├── phase2/
│   ├── 01_performance_optimization.md  # 性能优化方案
│   ├── 02_advanced_factors.md          # 高级因子系统
│   ├── 03_portfolio_optimization.md    # 组合优化
│   ├── 04_attribution.md               # 收益归因
│   └── 05_parameter_search.md          # 参数搜索框架
└── phase3/
    ├── 01_factor_mining.md       # LLM 驱动因子挖掘
    ├── 02_reinforcement_learning.md  # 强化学习集成
    ├── 03_live_trading.md        # 实盘交易对接
    └── 04_visualization.md       # Web 可视化面板
```

---

## 附录 B: 参考项目对照表

| 设计要点 | 主要参考项目 | 具体文件/模块 |
|----------|-------------|---------------|
| 数据源抽象 | new-quant | core/data/base.py |
| BaoStock 接入 | myquant-strategy | baostock_backtest/data/ |
| 复权处理 | new-quant + myquant-clean | adjuster.py + xtdata.py |
| 数据缓存 | new-quant | core/data/cache.py |
| 两阶段引擎 | new-quant + myquant-clean | engine.py + fast_engine.py |
| 交易成本模型 | new-quant + OSkhQuant | cost.py + khTrade.py |
| 因子计算 | myquant-clean + OSkhQuant | alpha101.py + MyTT.py |
| 因子缓存 | myyquant-1116 | context.py _add_quant_factors |
| 因子注册 | myquant-clean + alpha-gpt | configs.json + 元信息管理 |
| 因子挖局 | alpha-gpt + myquant-clean | LangGraph pipeline + auto_factor_mining |
| 公式引擎 | myquant-strategy | baostock_clean/formulas.py |
| 策略模板 | myquant-strategy | templates.py + template_strategy.py |
| 参数消融 | myquant-strategy | auto_ablate_*.py |
| 绩效指标 | new-quant | account.py get_performance_metrics |
| 策略加载 | new-quant + OSkhQuant | loader.py + importlib pattern |
| CLI 工具 | myquant-strategy | cli.py |
| 集成测试 | new-quant | tests/test_integration.py |
| 交易日历 | 新建 | 借鉴 exchange_calendars 库设计 |

---

## 12. Qlib 深度评估与集成方案（修订）

> 本章基于 Qlib main 分支最新代码（2025 年版本）的完整阅读分析。

### 12.1 Qlib 源码阅读结论

#### 12.1.1 数据层（可直接复用）

Qlib 的数据层是最成熟、最值得复用的部分：

**存储格式**：按字段 × 股票拆分的二进制文件（`.day.bin`），列式存储，读取极快。

**数据访问接口**：`D.features(instruments, fields, start, end)` 统一入口，自动处理缓存。

**表达式引擎**：`ExpressionEngine` 支持 30+ 算子（Ref/Mean/Std/Rank/Scale/Corr/Delta/...），因子用字符串表达式定义，自动缓存计算结果。

**Alpha158**：内置 158 个技术因子，可通过 `custom_fields` 参数扩展自定义因子。

**数据转换工具**：`dump_bin` 提供 CSV -> Qlib 二进制格式的批量转换，`dump_fix` 支持增量更新。

#### 12.1.2 回测引擎 Exchange（A 股兼容性分析）

Qlib 的 `Exchange` 类（`qlib/backtest/exchange.py`）是撮合引擎的核心，经过仔细阅读，其 A 股支持情况如下：

**已支持的 A 股特性**：

| 特性 | 实现方式 | 代码位置 |
|------|----------|----------|
| 交易单位 100 股 | `trade_unit=100` (REG_CN) | config.py:298 |
| 涨跌停限制 | `limit_threshold=0.095` (9.5%) | config.py:299 |
| 停牌检测 | `$close` 为 NaN 视为停牌 | exchange.py:275 |
| 成交价选择 | `deal_price='$close'` | config.py:300 |
| 佣金 | `open_cost=0.0015`, `close_cost=0.0025` | exchange.py:48-49 |
| 最低佣金 | `min_cost=5.0` | exchange.py:50 |
| 冲击成本/滑点 | `impact_cost` 参数 | exchange.py:51 |
| 涨跌停表达式 | 支持自定义表达式 `(buy_limit_expr, sell_limit_expr)` | exchange.py:283-285 |
| 订单量限制 | `volume_threshold` 支持成交量百分比限制 | exchange.py:786-830 |

**缺失的 A 股特性**（需要自研或修改）：

| 特性 | 说明 | 影响 |
|------|------|------|
| T+1 卖出限制 | 当日买入的股票不可当日卖出，Qlib 无此检查 | 高估策略收益 |
| 板别感知涨跌停 | 主板 10%、创业板 20%、科创板 20%、北交所 30%，Qlib 只有一个全局阈值 | 错误允许/禁止交易 |
| 印花税（卖向） | 中国印花税仅卖出时收取（0.05%），Qlib 的 open_cost/close_cost 模型不区分 | 成本计算不精确 |
| 过户费（沪市） | 仅上海股票收取（0.002%），Qlib 无此模型 | 成本计算不精确 |
| 一字板判断 | 开盘即涨停/跌停，全天无成交机会，Qlib 无此判断 | 高估成交概率 |
| 价格跳变保护 | 复牌首日价格大幅跳变，Qlib 无此检查 | 可能在异常价格成交 |

#### 12.1.3 策略层（不适合直接复用）

Qlib 的策略层（`TopkDropoutStrategy`、`WeightStrategyBase`、`EnhancedIndexingStrategy`）是为 AI 预测模型设计的：

- **输入**：模型预测分数（`Signal`）
- **逻辑**：按分数排序 -> 选 Top K -> 卖出不在 Top K 的 -> 等权或优化器分配权重
- **不适用场景**：传统规则型策略（如 水饺 / 均线交叉 / RSI 等基于技术指标的策略）

Qlib 的策略接口要求实现 `generate_trade_decision(execute_result)` 方法，返回 `TradeDecisionWO` 对象。
这个接口本身是通用的，可以作为我们策略层的基类接口参考。

#### 12.1.4 账户与回测循环（可参考）

Qlib 的 `Account` 类（`qlib/backtest/account.py`）：
- 负责持仓管理、现金管理、成本累积
- 每根 bar 结束时更新持仓市值、计算组合指标
- `PortfolioMetrics` 生成净值曲线、收益率序列
- `Indicator` 生成交易指标（换手率、佣金等）

Qlib 的回测循环（`backtest_loop` / `collect_data_loop`）：
- `while not executor.finished():` 循环
- `strategy.generate_trade_decision()` -> `executor.collect_data()`
- 简洁清晰，可作为参考设计

### 12.2 最终架构决策：复用 Qlib 数据层 + 自研 A 股回测引擎

基于以上分析，最终架构方案如下：

```
┌──────────────────────────────────────────────────────────────────┐
│                        QuantX 最终架构                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              复用 Qlib 层（不修改 Qlib 源码）                  │ │
│  │                                                              │ │
│  │  1. 数据存储：Qlib 二进制格式（.day.bin）                      │ │
│  │     · 按字段 × 股票拆分，列式存储，读取极快                    │ │
│  │     · dump_bin / dump_fix 工具链                             │ │
│  │                                                              │ │
│  │  2. 数据访问：D.features() 统一接口                           │ │
│  │     · 按股票+日期+字段表达式查询                               │ │
│  │     · 自动缓存，按需加载                                       │ │
│  │                                                              │ │
│  │  3. 因子系统：ExpressionEngine + Alpha158                     │ │
│  │     · 字符串表达式定义因子，一行一个                            │ │
│  │     · 内置 158 个因子，自定义因子无限扩展                       │ │
│  │     · 自动缓存，自动拓扑排序，按需计算                          │ │
│  │                                                              │ │
│  │  4. 交易日历：Qlib 内置日历                                   │ │
│  │     · calendars/day.txt 自动管理                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                  │
│                                ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              自研层（QuantX 核心）                             │ │
│  │                                                              │ │
│  │  1. BaoStock → Qlib 数据管道                                 │ │
│  │     · BaostockToQlibConverter                                │ │
│  │     · 前复权处理（在转换前完成）                                │ │
│  │     · 增量更新（dump_fix）                                    │ │
│  │     · 每日定时同步 Job                                        │ │
│  │                                                              │ │
│  │  2. A 股增强回测引擎（自研）                                    │ │
│  │     · 两阶段引擎（信号预计算 + 交易执行）                        │ │
│  │     · 完整 A 股规则：                                          │ │
│  │       - T+1 卖出限制                                          │ │
│  │       - 板别感知涨跌停（主板10%/创业板20%/科创板20%/北交所30%）  │ │
│  │       - 印花税（卖向 0.05%）+ 过户费（沪市 0.002%）            │ │
│  │       - 一字板判断 + 价格跳变保护                              │ │
│  │     · 撮合逻辑：T-1 信号 + T 日开盘价执行                      │ │
│  │     · 交易成本：佣金+印花税+过户费+滑点，完整建模               │ │
│  │                                                              │ │
│  │  3. 策略层（自研，兼容 Qlib 接口）                              │ │
│  │     · 函数式策略接口：get_stock_signal / get_trade_signal      │ │
│  │     · 声明式策略模板（FormulaEngine 驱动）                      │ │
│  │     · 动态策略加载器（任意路径）                                │ │
│  │     · 兼容 Qlib 的 BaseStrategy 接口（generate_trade_decision） │ │
│  │                                                              │ │
│  │  4. 评估层（自研）                                             │ │
│  │     · 绩效指标：Sharpe/Sortino/Calmar/MaxDD/收益率             │ │
│  │     · 收益归因：Brinson 归因 / 因子归因                         │ │
│  │     · 可视化：净值曲线/月度热力图/回测报告                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 12.3 关键设计决策：为什么不直接修改 Qlib 的 Exchange

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 修改 Qlib Exchange 源码 | 复用 Qlib 的撮合逻辑 | Qlib 升级时需重新合并；修改点分散在 exchange.py 多个方法中；Qlib 的 backtest 模块已被官方标记为 deprecated | 不推荐 |
| 继承 Exchange 并覆盖 | 保持 Qlib 源码不变 | Exchange 的关键方法（_calc_trade_info_by_order、deal_order 等）设计为内部方法，不适合继承覆盖 | 不推荐 |
| 自研 A 股增强引擎 | 完全控制；A 股规则精确实现；不依赖 Qlib 版本 | 需要写约 500 行代码 | **推荐** |

自研引擎实际上只需要实现：
1. `AStockExecutor`：订单校验（T+1、板别涨跌停、一字板、价格跳变）+ 订单执行（滑点、手续费）
2. `AStockAccount`：持仓管理 + 现金管理 + 成本计算（印花税卖向、过户费沪市）
3. `AStockExchange`：从 Qlib 的 D.features() 读取行情数据，提供成交价查询
4. `BacktestEngine`：两阶段回测循环（信号预计算 + 交易执行）

这些组件可以复用 new-quant 和 myquant-strategy 中已验证的设计，开发量约 500 行。

### 12.4 开发量估算（修订版）

| 组件 | 代码量 | 复用来源 |
|------|--------|----------|
| BaostockToQlibConverter | ~200 行 | myquant-strategy baostock_client + sync_service + Qlib dump_bin |
| AStockExecutor | ~150 行 | new-quant executor.py + A 股规则增强 |
| AStockAccount | ~100 行 | new-quant account.py + A 股成本模型增强 |
| AStockExchange | ~80 行 | 封装 Qlib D.features() |
| BacktestEngine | ~120 行 | new-quant engine.py + myquant-clean fast_engine.py |
| 策略层 | ~150 行 | new-quant loader.py + myquant-strategy templates.py |
| 评估层 | ~100 行 | new-quant analysis/ |
| DuckDB 复杂因子存储 | ~200 行 | 仅用于表达式无法表达的复杂因子 |
| 数据同步 Job | ~100 行 | myquant-strategy update_all_stocks.py |
| **总计** | **~1200 行** | |

**核心结论**：约 1200 行新代码 + Qlib 数据层，就能构建一个完整的、A 股规则精确的量化回测系统。

---

## 13. 架构决策总结

### 13.1 核心决策：复用 Qlib 数据层，自研回测引擎

经过对 Qlib 最新源码的完整阅读，核心结论如下：

| 层 | 决策 | 理由 |
|------|------|------|
| 数据存储 | 复用 Qlib 二进制格式 | 性能最优，工具链成熟，按字段拆分存储支持按需加载 |
| 数据访问 | 复用 Qlib D.features() | 统一接口，表达式查询，自动缓存 |
| 因子系统 | 复用 Qlib ExpressionEngine + Alpha158 | 字符串表达式定义因子，自动缓存，一行一个因子 |
| 交易日历 | 复用 Qlib 内置日历 | calendars/day.txt 自动管理 |
| 数据源接入 | 自研 BaoStock → Qlib 转换器 | 约 200 行代码，复用 myquant-strategy 的数据同步模块 |
| 回测撮合引擎 | 自研 A 股增强引擎 | Qlib Exchange 缺失 T+1、板别涨跌停、印花税卖向、过户费沪市等关键规则 |
| 策略层 | 自研，兼容 Qlib 接口 | Qlib 策略层为 AI 预测模型设计，不适合传统规则策略 |
| 评估层 | 自研 | 需要 A 股特有的绩效归因和报告 |
| 复杂因子存储 | DuckDB 兜底 | 表达式引擎无法表达的因子用 Python 实现 + DuckDB 长表存储 |

### 13.2 为什么这个方案最优

1. **最大化复用**：Qlib 的数据层和表达式引擎是微软研究院多年打磨的成果，不需要重新发明轮子
2. **最小化修改**：不修改 Qlib 源码，只通过其公开 API（D.features()）读取数据
3. **精确的 A 股规则**：回测引擎完全自研，T+1、板别涨跌停、印花税、过户费、一字板等规则精确实现
4. **因子爆炸无忧**：Qlib 表达式引擎让新增因子零成本（一行字符串），复杂因子走 DuckDB
5. **开发量可控**：总计约 1200 行新代码，一个月内可完成 Phase 1 MVP

---

> **文档版本**: v0.2.0
> **更新日期**: 2026-06-25
> **更新内容**: 新增第 12-13 节，基于 Qlib main 分支最新源码的深度分析，修订架构方案为「复用 Qlib 数据层 + 自研 A 股回测引擎」
