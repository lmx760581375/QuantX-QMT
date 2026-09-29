# QuantX 开发计划总览

> **版本**: v0.1.0
> **日期**: 2026-06-25
> **依赖**: 总体架构设计方案 (QUANTX_DESIGN.md) v0.2.0

---

## 文档索引

| 编号 | 文档 | 内容 | 预计工期 |
|------|------|------|----------|
| 00 | OVERVIEW.md (本文件) | 开发计划总览、依赖关系、开发顺序 | - |
| 01 | 01_DATA_LAYER.md | 数据层：BaoStock → Qlib 转换、同步、复权、日历 | 5-7 天 |
| 02 | 02_FACTOR_LAYER.md | 因子层：表达式引擎、DuckDB 存储、因子注册、依赖管理 | 3-5 天 |
| 03 | 03_BACKTEST_ENGINE.md | 回测引擎：A 股增强撮合、账户、执行器、Policy 驱动循环 | 7-10 天 |
| 04 | 04_STRATEGY_LAYER.md | 策略层（核心）：三个 Policy 接口（选股/调仓/执行）、规则+模型双模式 | 5-7 天 |
| 05 | 05_PORTFOLIO_LAYER.md | ~~组合层~~（已合并到策略层，调仓策略 = RebalanceStrategy Policy） | - |
| 06 | 06_ANALYSIS_LAYER.md | 评估层：绩效指标、归因分析、可视化 | 3-5 天 |
| 07 | 07_TOOLS_LAYER.md | 工具链：CLI 命令、数据同步、回测运行、定时 Job | 3-5 天 |
| 08 | 08_TESTING_PLAN.md | 测试计划：单元测试、集成测试、验证工作流 | 持续 |
| 09 | 09_DYNAMIC_FACTOR_RUNTIME_FIX_PLAN.md | 动态因子运行时、MarketPanel、Qlib 对齐修复 | 已落地 |
| 10 | 10_CONFIG_DRIVEN_STRATEGY_DESIGN.md | Config 驱动策略语言、公式 DAG、选股/调仓/执行配置化 | 持续 |
| 11 | 11_VISUALIZATION_AGENT_SYSTEM.md | 本地 Web 工作台、报告产物、Agent API | 持续 |
| 12 | 12_META_INDUSTRY_CONFIG_DESIGN.md | 股票名称、行业/板块 MetaStore，以及行业因子 config 语言扩展 | 待实现 |
| 13 | 13_QUANTX_AGENT_SKILL_DESIGN.md | QuantX Codex Skill、Agent 工具接口、指标扩展与安装方案 | 待实现 |
| 14 | 14_STRATEGY_SEARCH_2020_2026_RESULTS.md | 2020-2026 主板策略搜索结果、严格口径冠军与风险备选 | 持续 |
| 16 | 16_TRADE_PATTERN_ML_ANALYSIS_DESIGN.md | 交易形态样本库、K 线柱子特征、监督学习/聚类分析与形态报告 | 待实现 |

## 项目目录结构

```
quantx/
├── pyproject.toml              # 项目元数据与依赖
├── README.md
├── qlib/                       # Microsoft Qlib 源码（内置，无需 pip install）
│   ├── qlib/                   # Qlib 核心包
│   │   ├── data/               # D.features() 数据访问
│   │   ├── contrib/data/       # Alpha158, dump_bin
│   │   ├── backtest/           # Exchange/Account（参考）
│   │   └── ...
│   └── setup.py
├── docs/
│   ├── QUANTX_DESIGN.md        # 总体架构设计
│   └── development/            # 开发计划文档
│       ├── 00_OVERVIEW.md
│       ├── 01_DATA_LAYER.md
│       ├── 02_FACTOR_LAYER.md
│       ├── 03_BACKTEST_ENGINE.md
│       ├── 04_STRATEGY_LAYER.md
│       ├── 05_PORTFOLIO_LAYER.md
│       ├── 06_ANALYSIS_LAYER.md
│       ├── 07_TOOLS_LAYER.md
│       ├── 08_TESTING_PLAN.md
│       ├── 09_DYNAMIC_FACTOR_RUNTIME_FIX_PLAN.md
│       ├── 10_CONFIG_DRIVEN_STRATEGY_DESIGN.md
│       ├── 11_VISUALIZATION_AGENT_SYSTEM.md
│       ├── 12_META_INDUSTRY_CONFIG_DESIGN.md
│       ├── 13_QUANTX_AGENT_SKILL_DESIGN.md
│       ├── 16_TRADE_PATTERN_ML_ANALYSIS_DESIGN.md
│       └── REVIEW_AND_IMPROVEMENTS.md
├── quantx/                     # 主包
│   ├── __init__.py
│   ├── core/                   # 核心引擎
│   │   ├── __init__.py
│   │   ├── data/               # 数据层
│   │   ├── factor/             # 因子层
│   │   ├── engine/             # 回测引擎
│   │   ├── strategy/           # 策略层（三个 Policy 接口）
│   │   │   ├── base.py         # PolicyState, Policy ABC, ModelBasedPolicy
│   │   │   ├── selector.py     # 选股策略（规则+模型）
│   │   │   ├── rebalance.py    # 调仓策略（规则+模型）
│   │   │   ├── execution.py    # 执行策略（规则+模型）
│   │   │   ├── composite.py    # 组合策略
│   │   │   ├── loader.py       # 策略加载器
│   │   │   └── templates.py    # 声明式策略模板
│   │   └── analysis/           # 评估层
│   ├── strategies/             # 用户策略目录
│   ├── configs/                # 配置文件
│   └── tools/                  # CLI 工具
├── data/                       # 本地数据存储
│   ├── raw/                    # 原始 CSV 数据
│   ├── qlib_data/              # Qlib 二进制数据
│   ├── meta/                   # 股票名称、行业、板块、概念等元数据
│   └── warehouse/              # DuckDB 因子仓库
└── tests/                      # 测试目录
    ├── data/                   # 数据层测试
    ├── factor/                 # 因子层测试
    ├── engine/                 # 引擎层测试
    ├── strategy/               # 策略层测试（含 selector/rebalance/execution）
    ├── analysis/               # 评估层测试
    └── integration/            # 集成测试
```

## 开发顺序与依赖关系

```
Phase 1: 基础架构 (Day 1-2)
  ├── pyproject.toml + 项目骨架
  └── Qlib 集成到项目目录

Phase 2: 并行开发 (Day 3-12)  ← 可并行
  ├── 数据层 (01_DATA_LAYER.md)       [开发者 A]
  │   ├── 数据源抽象 + 复权 + 日历
  │   ├── BaoStock API + 本地仓库 + 同步服务
  │   └── BaoStock → Qlib 转换器
  │
  └── 因子层 (02_FACTOR_LAYER.md)     [开发者 B，可同时开发]
      ├── 因子接口 + 注册中心（Mock DataSource 开发）
      ├── Qlib 表达式引擎封装
      ├── DuckDB 存储 + 依赖图管理
      └── Alpha101 移植 + 技术指标
             │
             ▼
Phase 3: 回测引擎 (Day 13-22)  ← 需数据层+因子层完成
  ├── 基础类型 (types.py, cost.py, board.py, config.py)
  ├── A 股增强撮合引擎 (exchange.py) ← 核心
  ├── 账户 + 执行器 (account.py, executor.py)
  ├── 回测上下文 + 信号矩阵 (context.py, signals.py)
  └── 两阶段回测引擎 (engine.py)
             │
             ├──────────────────────────────┐
             ▼                              ▼
Phase 4: 并行开发 (Day 23-32)  ← 可并行
  ├── 策略层 (04_STRATEGY_LAYER.md)   [开发者 A]
  │   ├── Policy 接口定义 (base.py)
  │   ├── 规则策略实现 (selector/rebalance/execution)
  │   ├── 组合策略 (composite.py)
  │   └── 加载器 + 校验
  │
  └── 评估层 (06_ANALYSIS_LAYER.md)   [开发者 B]
             │
             ▼
Phase 5: 工具链 + 集成测试 (Day 33-40)
  ├── CLI 命令 (sync-data, backtest, factor)
  ├── 定时同步 Job
  ├── 端到端集成测试
  └── 文档完善
```

**并行开发说明**：
- 数据层和因子层可以并行开发：因子层只需 `DataSource` 接口定义完成，使用 Mock 数据源独立开发
- 策略层和评估层可以并行开发：它们依赖回测引擎的接口，可以用 Mock 引擎独立开发
- 策略层不再需要独立的 portfolio 层——选股/调仓/执行三个 Policy 在策略层统一管理

## 每个模块的开发流程

每个模块遵循统一的开发流程：

```
1. 阅读对应的开发计划文档
2. 创建模块目录和 __init__.py
3. 编写接口定义（ABC 抽象基类 / Protocol）
4. 编写具体实现
5. 编写单元测试（每完成一个类就写测试）
6. 编写模块级集成测试
7. 运行测试: pytest tests/<module>/
8. 代码审查
```

## 技术栈与依赖

| 组件 | 技术选型 | 版本要求 |
|------|----------|----------|
| 语言 | Python | >= 3.10 |
| 数据处理 | pandas | >= 2.0 |
| 数值计算 | numpy | >= 1.24 |
| 数据源 | baostock | >= 0.9.2 |
| 数据存储 | Qlib (内置于 `qlib/` 目录) | 0.9.7+ |
| 因子存储 | DuckDB | >= 0.9 |
| 配置管理 | PyYAML + pydantic | - |
| 并行计算 | concurrent.futures | 内置 |
| 测试框架 | pytest | >= 7.0 |
| 类型检查 | mypy | >= 1.0 |
| 代码风格 | ruff | >= 0.1 |
| 可视化 | matplotlib | >= 3.5 |
| CLI 框架 | argparse | 内置 |

## pyproject.toml 内容

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "quantx"
version = "0.1.0"
description = "A股量化研究与回测系统"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "baostock>=0.9.2",
    "duckdb>=0.9",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "matplotlib>=3.5",
    "pydantic-settings>=2.0",  # Qlib 依赖
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

[project.scripts]
quantx = "quantx.tools.cli:main"

[tool.setuptools.packages.find]
include = ["quantx*"]

[tool.ruff]
line-length = 120

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "real_data: tests that require real market data",
    "slow: slow running tests",
]
```

## 开发约定

1. **类型注解**：所有公共函数/方法必须有完整的类型注解
2. **文档字符串**：所有公共类/函数必须有 docstring（Google 风格）
3. **错误处理**：使用自定义异常类，不使用裸 except
4. **日志**：使用 logging 模块，不使用 print
5. **配置**：所有可变参数通过 YAML 配置文件注入，不硬编码
6. **测试**：每个模块的测试覆盖率 >= 80%
7. **命名**：遵循 PEP 8 命名规范
8. **路径**：使用 pathlib.Path，不使用字符串拼接路径
