# 开发计划 Review 与改进方案

> **日期**: 2026-06-25
> **Review 范围**: 00_OVERVIEW.md ~ 08_TESTING_PLAN.md 全部 9 个文档

---

## 一、全局性问题

### 1.1 Qlib 依赖方式变更

**问题**：当前所有文档写的是 `pip install qlib`，但 Qlib 已放入项目目录 `quantx/qlib/`。
**改进**：
- 在 `pyproject.toml` 中添加本地依赖路径
- 在 `00_OVERVIEW.md` 的目录结构中增加 `qlib/` 目录
- 所有文档中 `pip install qlib` 改为「Qlib 已内置在项目中，无需额外安装」
- 注意：Qlib 自身有依赖（pandas/numpy/pydantic_settings/...），需要在 `pyproject.toml` 中声明

### 1.2 缺少 `pyproject.toml` 详细内容

**问题**：所有文档都没有说明 `pyproject.toml` 里应该写什么。
**改进**：在 `00_OVERVIEW.md` 中增加 `pyproject.toml` 的详细内容，包括：
- 项目元数据（name, version, python_requires）
- 依赖声明（pandas, numpy, baostock, duckdb, pyyaml, pydantic, matplotlib）
- Qlib 本地路径依赖
- CLI 入口点（console_scripts）
- 开发依赖（pytest, mypy, ruff）

### 1.3 缺少 `configs/` 目录的详细说明

**问题**：`07_TOOLS_LAYER.md` 提到了配置，但没有一个文档明确列出所有需要的配置文件及其内容。
**改进**：在 `00_OVERVIEW.md` 中增加配置文件清单：
- `configs/data/baostock.yaml` — 数据源配置
- `configs/backtest/default.yaml` — 回测默认配置
- `configs/factors/alpha101.yaml` — 因子配置
- `configs/strategies/ma_cross.yaml` — 策略配置示例

---

## 二、各文档详细 Review

### 2.1 00_OVERVIEW.md — 开发计划总览

**问题 1**：目录结构缺少 `qlib/` 和 `qlib-main.zip`
**改进**：增加 `qlib/` 到项目目录树中

**问题 2**：技术栈表中 Qlib 写的是 `latest`，应改为具体版本
**改进**：改为 `qlib (内置)`，注明版本号

**问题 3**：开发顺序是线性的，但实际有些模块可以并行
**改进**：标注哪些模块可以并行开发（详见下文 「并行开发优化」）

### 2.2 01_DATA_LAYER.md — 数据层

**问题 1**：`baostock_source.py` 在文件列表中但没有详细说明
**改进**：增加 `baostock_source.py` 的详细说明——它是 `BaoStockDataSource(DataSource)` 实现类，
  封装了 `baostock_client.py` 和 `repository.py`，对外提供 `DataSource` 接口。
  这是策略层和回测引擎访问数据的唯一入口。

**问题 2**：`$change` 字段计算公式有边界情况未处理
**改进**：`$change = (close - preclose) / preclose`，当 `preclose == 0` 时需要处理。
  BaoStock 数据中 `preclose` 不会为 0，但如果有新股上市首日，`preclose` 可能为 NaN。
  建议：`$change = np.where(preclose > 0, (close - preclose) / preclose, 0)`

**问题 3**：没有说明 `$preclose` 字段是否需要存入 Qlib 格式
**改进**：`$preclose` 不需要单独存为 Qlib字段，但 `$change` 的计算依赖 `preclose`。
  在 `converter.py` 中，计算 `$change` 后即可丢弃 `preclose`。

**问题 4**：缺少 `baostock_source.py` 的 `DataSource` 接口实现说明
**改进**：增加一节说明 `BaoStockDataSource` 如何实现 `DataSource` ABC 的 5 个方法：
  - `get_daily_bars()` → 从 Qlib D.features() 读取
  - `get_dividend_factors()` → 从 Baostock 获取除权除息数据
  - `get_stock_list()` → 从 repository 或 Qlib instruments 读取
  - `get_benchmark()` → 从 Qlib 读取指数数据
  - `get_trading_dates()` → 从 Qlib calendars 读取

**问题 5**：`adjuster.py` 的定位模糊——BaoStock 已返回前复权数据，是否还需要独立复权？
**改进**：明确说明：
  - Phase 1：直接使用 BaoStock 的 `adjustflag='2'`（前复权），`adjuster.py` 作为备用模块保留
  - Phase 2：当接入其他数据源（如 Tushare 不复权数据）时，启用 `adjuster.py`
  - 在 `converter.py` 中，复权是可选的——如果数据已复权，跳过 `adjuster.py`

### 2.3 02_FACTOR_LAYER.md — 因子层

**问题 1**：缺少因子依赖图管理模块
**改进**：增加 `dependency.py` 文件，管理因子间的依赖关系：
  - 拓扑排序：确定计算顺序
  - 变更影响分析：X 因子变更后，哪些下游因子需要重算
  - 循环依赖检测：防止 A 依赖 B 同时 B 依赖 A

**问题 2**：`ExpressionFactor` 和 `Factor` ABC 的关系不清晰
**改进**：明确设计：
  - `ExpressionFactor` 继承 `Factor` ABC
  - 其 `compute()` 方法内部调用 `D.features()`
  - 这样所有因子（表达式和 Python）都通过 `FactorRegistry` 统一管理

**问题 3**：`engine.py` 的 `ensure_factors` 太简略，缺少关键逻辑
**改进**：补充详细伪代码：
  1. 解析依赖图，获取所有需要计算的因子（含间接依赖）
  2. 将因子按类型分为两组：表达式因子 + Python 因子
  3. 表达式因子：直接通过 Qlib D.features() 批量计算（Qlib 自动缓存）
  4. Python 因子：按拓扑顺序逐个计算
     a. 检查 DuckDB 缓存存在性
     b. 缺失则计算 → 存入 DuckDB
     c. 已有但日期范围不完整 → 增量计算 → 更新 DuckDB
  5. 合并所有因子值为宽表 DataFrame

**问题 4**：缺少因子版本管理
**改进**：在 `FactorMeta` 中增加 `version` 字段，DuckDB 存储时记录因子版本。
  当因子代码变更（version 变化）时，自动失效缓存并重算。

### 2.4 03_BACKTEST_ENGINE.md — 回测引擎（最重要）

**问题 1**：`executor.py` 和 `exchange.py` 的职责边界模糊
**改进**：重新划分职责：
  - `exchange.py`：只负责行情数据查询——成交价、是否停牌、是否涨跌停、是否一字板。
    不负责订单校验。
  - `executor.py`：负责订单校验 + 订单执行。校验时调用 exchange 查询行情状态。
  - 这样职责更清晰：exchange = 行情数据层，executor = 订单处理层

**问题 2**：`context.py` 和 `exchange.py` 都从 Qlib 加载数据，存在重复
**改进**：统一数据加载入口：
  - `exchange.py` 负责从 Qlib 加载行情数据（$open/$high/$low/$close/$volume/$change/$factor）
  - `context.py` 通过 exchange 访问行情数据，不直接调用 Qlib API
  - 因子数据由 `LazyFactorEngine` 加载，注入到 context 中
  - 职责：exchange = 行情数据提供者，context = 策略数据视图（行情 + 因子）

**问题 3**：缺少 `DailySnapshot` 数据类定义
**改进**：在 `types.py` 中增加：
```python
@dataclass
class DailySnapshot:
    date: str
    cash: float
    total_value: float
    positions: Dict[str, Position]
    daily_return: float
    cumulative_return: float
```

**问题 4**：`engine.py` 主循环中缺少组合层调用
**改进**：在 Phase 2 循环中插入组合层：
```python
# Phase 2: 按交易日顺序执行
while not context.is_finished():
    date = context.next()
    # 1. 获取当日信号
    signals = context.get_stock_signals(date)
    # 2. 组合层：根据信号计算目标权重
    target_weights = portfolio.allocate(signals, account, context)
    # 3. 策略层：根据目标权重生成订单
    orders = strategy.get_trade_signal(context, account, target_weights)
    # 4. 执行订单
    for order in orders:
        executor.execute(order, account, exchange)
    # 5. 更新每日净值
    account.update_daily_balance(date, exchange)
```

**问题 5**：一字板判断逻辑有缺陷
**改进**：`open == high == low == close` 的判断不够精确，因为浮点数比较。
  改为：`abs(high - low) < 1e-6 and abs(open - close) < 1e-6`
  且需要同时满足 `change` 接近涨跌停阈值。
  此外，一字板还需要区分：
  - 一字涨停：`close == high == limit_up_price` 且 `volume == 0`（或极小）
  - 一字跌停：`close == low == limit_down_price` 且 `volume == 0`（或极小）
  - 非一字板涨停：`close == limit_up_price` 但 `volume > 0`（盘中涨停，可以卖出）

**问题 6**：缺少 `config.py` 模块用于管理回测配置
**改进**：增加 `config.py` 文件，定义回测配置的数据类：
```python
@dataclass
class BacktestConfig:
    init_cash: float = 1_000_000
    start_date: str = '2020-01-01'
    end_date: str = '2025-12-31'
    benchmark: str = 'SH000300'
    cost: TransactionCost = field(default_factory=TransactionCost)
    factor_names: List[str] = field(default_factory=list)
    max_workers: int = 8
```

### 2.5 04_STRATEGY_LAYER.md — 策略层

**问题 1**：策略接口函数签名的参数不够清晰
**改进**：明确函数签名和参数：
```python
def get_stock_signal(
    context: BacktestContext,  # 回测上下文（数据+因子）
    date: str                  # 当前决策日期（T-1 日）
) -> List[Signal]:             # 返回选股信号列表

def get_trade_signal(
    context: BacktestContext,   # 回测上下文
    account: Account,           # 当前账户状态
    target_weights: Dict[str, float]  # 组合层输出的目标权重
) -> List[Order]:               # 返回订单列表
```

**问题 2**：缺少策略与组合层的交互说明
**改进**：增加一节说明数据流：
  `get_stock_signal` → 选股 → `portfolio.allocate` → 权重 → `get_trade_signal` → 订单
  策略层决定「买什么」，组合层决定「买多少」，引擎层负责「执行」

### 2.6 05_PORTFOLIO_LAYER.md — 组合层

**问题 1**：文档太薄，缺少关键细节
**改进**：增加以下内容：
  - 等权组合的「再平衡」逻辑：当持仓偏离目标权重超过阈值时触发调仓
  - 分数加权组合的分数归一化逻辑
  - 组合层需要考虑交易成本（换手率过高导致成本吃掉收益）
  - 组合层需要处理现金不足的情况（按权重比例缩减）

**问题 2**：缺少 `rebalance.py` 文件
**改进**：增加 `rebalance.py`，实现调仓逻辑：
  - 计算当前持仓权重 vs 目标权重
  - 只对偏差超过阈值的股票下单
  - 尽量减少换手（卖出后再买入，避免双向交易成本）

### 2.7 06_ANALYSIS_LAYER.md — 评估层

**问题 1**：缺少基准数据加载说明
**改进**：增加说明：
  - 基准数据（如 CSI 300）从 Qlib 的 `D.features()` 加载
  - 基准收益率计算需要包含分红再投资
  - 如果基准数据不可用，Beta/Alpha/信息比率等指标会跳过

**问题 2**：缺少 `report.py` 输出格式说明
**改进**：明确报告输出格式：
  - JSON 格式：供程序化分析
  - Markdown 格式：供人类阅读
  - CSV 格式：daily_equity.csv, trades.csv

### 2.8 07_TOOLS_LAYER.md — 工具链

**问题 1**：缺少定时数据同步的 Job 调度
**改进**：增加 `cron_sync.py` 或 `scheduler.py`：
  - 使用 Linux cron 或 Python schedule 库
  - 每个交易日 15:30 自动触发增量同步
  - 文件锁防止并发执行
  - 直接复用 `myquant-strategy/baostock_backtest/jobs/update_all_stocks.py`

**问题 2**：缺少 `pyproject.toml` 的 CLI 入口点配置
**改进**：增加示例：
```toml
[project.scripts]
quantx = "quantx.tools.cli:main"
```

### 2.9 08_TESTING_PLAN.md — 测试计划

**问题 1**：缺少 Mock Qlib 数据的 fixture
**改进**：增加 `mock_qlib` fixture，在测试中替代真实的 Qlib 数据加载：
```python
@pytest.fixture
def mock_qlib_data():
    """Mock Qlib D.features() 返回"""
    with patch('qlib.data.D.features') as mock:
        mock.return_value = synthetic_market_data()
        yield mock
```

**问题 2**：缺少测试数据隔离策略
**改进**：增加说明：
  - 所有测试使用 `tmp_path` fixture 创建临时目录
  - 不依赖外部数据源（BaoStock 网络不可用时测试也能通过）
  - 集成测试使用小规模合成数据（10 只股票 x 100 天）

---

## 三、并行开发优化

原计划的开发顺序是严格线性的，但实际上以下模块可以并行开发：

```
Phase 1: 基础架构 (Day 1-2)
  ├── pyproject.toml + 项目骨架
  └── Qlib 集成到项目目录

Phase 2: 并行开发 (Day 3-12)  ← 可并行
  ├── 数据层 (01_DATA_LAYER.md)       [开发者 A]
  └── 因子层 (02_FACTOR_LAYER.md)     [开发者 B]
      因子层只需数据层接口定义完成即可开始
      使用 Mock DataSource 进行开发

Phase 3: 回测引擎 (Day 13-22)  ← 需数据层+因子层完成
  └── 03_BACKTEST_ENGINE.md

Phase 4: 并行开发 (Day 23-32)  ← 可并行
  ├── 策略层 (04_STRATEGY_LAYER.md)   [开发者 A]
  ├── 组合层 (05_PORTFOLIO_LAYER.md)  [开发者 B]
  └── 评估层 (06_ANALYSIS_LAYER.md)   [开发者 C]

Phase 5: 工具链 + 测试 (Day 33-40)
  ├── 工具链 (07_TOOLS_LAYER.md)
  └── 端到端集成测试
```

这样可以缩短总工期约 10 天（从 40 天降到 30 天）。

---

## 四、Review 总结

### 主要问题分类

| 类别 | 数量 | 严重程度 | 涉及文档 |
|------|------|----------|----------|
| Qlib 依赖方式 | 1 | 中 | 00, 01 |
| 缺少关键文件说明 | 3 | 高 | 01(baostock_source), 02(dependency), 03(config) |
| 模块职责边界模糊 | 2 | 高 | 03(exchange vs executor vs context) |
| 缺少交互流程说明 | 3 | 中 | 03(engine+portfolio), 04(strategy+portfolio), 05(rebalance) |
| 边界情况未处理 | 2 | 中 | 01($change), 03(一字板判断) |
| 文档内容过薄 | 1 | 中 | 05(portfolio) |
| 缺少关键内容 | 3 | 中 | 00(pyproject.toml), 07(scheduler), 08(mock fixtures) |

### 高优先级修改项

1. **03_BACKTEST_ENGINE.md**：重新划分 exchange/executor/context 职责边界
2. **03_BACKTEST_ENGINE.md**：补充 engine.py 主循环中组合层的调用
3. **01_DATA_LAYER.md**：增加 `baostock_source.py` 的详细说明
4. **02_FACTOR_LAYER.md**：增加 `dependency.py` 依赖图管理模块
5. **00_OVERVIEW.md**：增加 `pyproject.toml` 详细内容 + `qlib/` 目录

### 中优先级修改项

6. **03_BACKTEST_ENGINE.md**：补充 `DailySnapshot` 和 `config.py`
7. **03_BACKTEST_ENGINE.md**：修正一字板判断逻辑（浮点数比较 + 区分一字/非一字）
8. **01_DATA_LAYER.md**：修正 `$change` 边界情况 + `adjuster.py` 定位
9. **05_PORTFOLIO_LAYER.md**：补充再平衡逻辑 + `rebalance.py` ✅ 已完成
10. **08_TESTING_PLAN.md**：增加 Mock Qlib fixture + 数据隔离策略 ✅ 已完成

### 低优先级修改项

11. **04_STRATEGY_LAYER.md**：明确函数签名 + 与组合层交互 ✅ 已完成
12. **06_ANALYSIS_LAYER.md**：增加基准数据加载说明 + 输出格式 ✅ 已完成
13. **07_TOOLS_LAYER.md**：增加定时 Job 调度 + CLI 入口点配置 ✅ 已完成

---

## 五、第二轮优化完成状态

| 优先级 | 问题编号 | 状态 |
|--------|----------|------|
| 高 | 1. 03 职责边界 | ✅ 已完成 |
| 高 | 2. 03 engine loop | ✅ 已完成 |
| 高 | 3. 01 baostock_source | ✅ 已完成 |
| 高 | 4. 02 dependency | ✅ 已完成 |
| 高 | 5. 00 pyproject.toml | ✅ 已完成 |
| 中 | 6. 03 DailySnapshot+config | ✅ 已完成 |
| 中 | 7. 03 一字板逻辑 | ✅ 已完成 |
| 中 | 8. 01 $change+adjuster | ✅ 已完成 |
| 中 | 9. 05 rebalance | ✅ 已完成 |
| 中 | 10. 08 Mock fixtures | ✅ 已完成 |
| 低 | 11. 04 函数签名 | ✅ 已完成 |
| 低 | 12. 06 基准+输出 | ✅ 已完成 |
| 低 | 13. 07 cron Job | ✅ 已完成 |

**所有 13 个 Review 问题已全部修复。**
