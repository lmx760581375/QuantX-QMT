# 数据层开发计划

> **版本**: v0.1.0
> **日期**: 2026-06-25
> **预计工期**: 5-7 天
> **依赖**: 无（数据层是系统最底层，不依赖其他模块）

---

## 1. 目标

数据层负责：
1. 从 BaoStock 拉取 A 股日线数据
2. 将原始数据转换为 Qlib 二进制格式
3. 处理前复权
4. 支持增量数据同步
5. 提供交易日历
6. 管理本地数据仓库

---

## 2. 文件清单与功能

### 2.1 文件列表

```
quantx/core/data/
├── __init__.py              # 模块入口，导出公共接口
├── base.py                  # 数据源抽象基类 (DataSource ABC)
├── baostock_client.py       # BaoStock API 封装
├── baostock_source.py       # BaoStock 数据源实现 (BaoStockDataSource)
├── repository.py            # 本地 CSV 数据仓库
├── sync_service.py          # 数据同步服务（全量/增量）
├── converter.py             # BaoStock CSV → Qlib 二进制格式转换器
├── adjuster.py              # 复权处理（前复权）
├── calendar.py              # 交易日历
├── validator.py             # 数据校验
└── cache.py                 # MultiIndex 数据缓存（可选）
```

### 2.2 各文件详细说明

---

#### 2.2.1 `__init__.py` — 模块入口

**功能**：
- 导出核心数据类：`DataSource`, `BaoStockDataSource`
- 导出工厂函数：`create_data_source(config)`
- 导出工具类：`TradingCalendar`, `Adjuster`

**开发要点**：
- 只导出公共 API，内部实现细节不暴露

---

#### 2.2.2 `base.py` — 数据源抽象基类

**功能**：定义 `DataSource` 抽象基类，规定所有数据源必须实现的接口。

**接口定义**：
```python
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

class DataSource(ABC):
    @abstractmethod
    def get_daily_bars(
        self, symbols: List[str], start_date: str, end_date: str,
        fields: Optional[List[str]] = None, adjust: str = "forward"
    ) -> pd.DataFrame:
        """获取日线行情，返回 (date, symbol) MultiIndex DataFrame"""
        ...

    @abstractmethod
    def get_dividend_factors(self, symbols: List[str]) -> pd.DataFrame:
        """获取复权因子"""
        ...

    @abstractmethod
    def get_stock_list(self, date: Optional[str] = None) -> List[str]:
        """获取指定日期的可交易股票列表"""
        ...

    @abstractmethod
    def get_benchmark(self, benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取基准指数数据"""
        ...

    @abstractmethod
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        ...
```

**开发要点**：
- 接口设计要稳定，后续所有模块依赖此接口
- 返回格式统一为 `(date, symbol)` MultiIndex DataFrame
- 字段名统一：`open/high/low/close/volume/amount/vwap`
- 复权方式 `forward` / `none` / `backward`
- 直接复用 new-quant 的 `core/data/base.py` 设计

**参考代码**：`new-quant/core/data/base.py`

---

#### 2.2.3 `baostock_client.py` — BaoStock API 封装

**功能**：封装 BaoStock 的 HTTP API，提供可靠的数据获取接口。

**核心类**：`BaoStockClient`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `login()` / `logout()` | 登录/登出（上下文管理器） |
| `query_all_stocks(date)` | 获取全量股票列表，自动回溯到最近交易日 |
| `query_history_k_data(symbol, start, end, fields, adjust)` | 获取日线 K 线数据 |
| `query_dividend_data(symbol, start, end)` | 获取除权除息数据 |
| `_rename_fields(df)` | 字段标准化重命名 |
| `_retry(func, max_retries, backoff)` | 重试逻辑 |

**开发要点**：
- 直接复用 `myquant-strategy/baostock_backtest/data/baostock_client.py`（约 150 行）
- 需要适配的地方：
  - 字段重命名映射：`turn` -> `turnover`, `pctChg` -> `pct_chg`
  - 股票代码格式：BaoStock 用 `sh.600519`，需统一为 `SH600519`（Qlib 格式）
  - 日期格式：`%Y-%m-%d`
- 注意：BaoStock 需要登录才能查询，使用 `with BaoStockClient() as client:` 模式
- 注意：BaoStock 的 `adjustflag="2"` 是前复权，在 API 层面直接获取前复权数据

**参考代码**：`myquant-strategy/baostock_backtest/data/baostock_client.py`

---

#### 2.2.3b `baostock_source.py` — BaoStock 数据源实现

**功能**：实现 `DataSource` ABC，封装 `BaoStockClient` + `LocalDataRepository` + `Converter`，对外提供统一的数据访问接口。**这是策略层和回测引擎访问数据的唯一入口。**

**核心类**：`BaoStockDataSource(DataSource)`

**实现 DataSource ABC 的 5 个方法**：

| 方法 | 实现方式 |
|------|----------|
| `get_daily_bars(symbols, start, end, fields, adjust)` | 从 Qlib `D.features()` 读取已转换的二进制数据 |
| `get_dividend_factors(symbols)` | 从 BaoStock API 获取除权除息数据 |
| `get_stock_list(date)` | 从 Qlib `D.instruments()` 或 repository 读取 |
| `get_benchmark(benchmark, start, end)` | 从 Qlib 读取指数数据（如 SH000300） |
| `get_trading_dates(start, end)` | 从 Qlib calendars 或 TradingCalendar 读取 |

**数据流**：
```
策略层调用 BaoStockDataSource.get_daily_bars()
  → 检查 Qlib 数据目录是否已有数据
    → 有数据：直接调用 D.features() 读取 Qlib 二进制数据
    → 无数据：触发同步流程
      → sync_service.sync_incremental() → repository → converter → D.features()
```

**Qlib 初始化**：
```python
class BaoStockDataSource(DataSource):
    def __init__(self, config: DataConfig):
        self.config = config
        self.client = BaoStockClient()
        self.repository = LocalDataRepository(config.root)
        self.sync_service = DataSyncService(self.client, self.repository, config)
        self.converter = BaostockToQlibConverter(config)
        self.calendar = TradingCalendar(config.provider_uri)

        # 初始化 Qlib（全局只初始化一次）
        if not QlibInitializer.is_initialized():
            QlibInitializer.init(config.provider_uri, region='cn')

    def get_daily_bars(self, symbols, start, end, fields=None, adjust="forward"):
        # 确保数据已转换到 Qlib 格式
        self._ensure_data_ready(symbols, start, end)
        # 从 Qlib 读取
        qlib_fields = [f"${f}" for f in (fields or ['open','high','low','close','volume','vwap','change'])]
        return D.features(symbols, qlib_fields, start, end, freq='day')

    def _ensure_data_ready(self, symbols, start, end):
        """检查 Qlib 数据是否覆盖所需范围，不足则触发同步+转换"""
        if not self.converter.check_coverage(start, end):
            self.sync_service.ensure_coverage(symbols, start, end)
            self.converter.convert_all(start, end, symbols)
```

**开发要点**：
- 约 120 行新代码
- 这是数据层对外的唯一入口，其他模块通过 `DataSource` 接口访问，不直接调用 `BaoStockClient` 或 `repository`
- `QlibInitializer` 是单例，确保 `qlib.init()` 全局只调用一次（Qlib 不支持多次 init）
- 数据就绪检查是惰性的——只在首次访问数据时触发同步
- 直接复用 `new-quant/core/data/base.py` 的 `DataSource` ABC 接口设计

---

#### 2.2.4 `repository.py` — 本地 CSV 数据仓库

**功能**：管理本地 CSV 文件存储，按股票拆分存储日线数据。

**核心类**：`LocalDataRepository`

**存储布局**：
```
data/raw/baostock/
├── stocks/
│   ├── SH600519.csv
│   ├── SZ000001.csv
│   └── ...
└── universe/
    └── all_stocks.csv
```

**主要方法**：
| 方法 | 功能 |
|------|------|
| `save_symbol(symbol, df)` | 保存单只股票数据（upsert 去重合并） |
| `load_symbol(symbol)` | 加载单只股票数据 |
| `load_all(start, end, symbols)` | 批量加载，返回 MultiIndex DataFrame |
| `get_last_date(symbol)` | 获取某股票最新数据日期 |
| `get_first_date(symbol)` | 获取某股票最早数据日期 |
| `save_universe(stocks_df)` | 保存股票列表 |
| `load_universe()` | 加载股票列表 |
| `get_coverage_report(start, end)` | 生成数据覆盖报告 |

**开发要点**：
- 直接复用 `myquant-strategy/baostock_backtest/data/repository.py`（约 120 行）
- CSV 列：`date,code,open,high,low,close,preclose,volume,amount,turnover,pct_chg,is_st`
- upsert 逻辑：按 `date` 去重，新数据覆盖旧数据
- 使用 `pd.read_csv` 批量加载，`pd.concat` 合并

**参考代码**：`myquant-strategy/baostock_backtest/data/repository.py`

---

#### 2.2.5 `sync_service.py` — 数据同步服务

**功能**：从 BaoStock 拉取数据并同步到本地仓库，支持全量/增量模式。

**核心类**：`DataSyncService`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `sync_full(symbols, start, end)` | 全量同步 |
| `sync_incremental(symbols)` | 增量同步（只拉取每个股票的新数据） |
| `ensure_coverage(symbols, start, end)` | 检查数据覆盖范围，填补首尾间隙 |
| `_sync_batch(symbols, start, end)` | 批量同步（多线程分片） |
| `_retry_failed(failed_symbols)` | 失败重试（3 次，线性退避） |

**开发要点**：
- 直接复用 `myquant-strategy/baostock_backtest/data/sync_service.py`（约 200 行）
- 多线程同步：使用 `ThreadPoolExecutor`，Round-Robin 分片
- 注意 BaoStock API 限流：`pause_seconds=0.5` 控制请求间隔
- 增量同步逻辑：
  1. 从 repository 获取每个股票的最新日期
  2. 只拉取该日期之后的数据
  3. 如果某股票没有本地数据，全量拉取
- 失败重试：收集失败股票列表，降低并发数重新拉取

**参考代码**：`myquant-strategy/baostock_backtest/data/sync_service.py`
---

#### 2.2.6 `converter.py` — BaoStock CSV → Qlib 二进制格式转换器

**功能**：将本地 CSV 格式的股票数据转换为 Qlib 的二进制格式。

**核心类**：`BaostockToQlibConverter`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `convert_all(start, end, symbols)` | 全量转换所有股票数据 |
| `convert_incremental(symbols)` | 增量转换（只转换新数据） |
| `_to_qlib_csv_format(df)` | 将 DataFrame 转为 Qlib CSV 格式 |
| `_run_dump_bin(csv_dir, qlib_dir, freq)` | 调用 Qlib 的 dump_bin 工具 |
| `_run_dump_fix(csv_path, qlib_dir, freq)` | 调用 Qlib 的 dump_fix 增量更新 |

**转换流程**：
```
1. 从 repository 加载 CSV 数据
2. 前复权处理（调用 adjuster.py）
3. 转换为 Qlib CSV 格式：
   - 列名映射：open -> $open, close -> $close, ...
   - 股票代码格式：SH600519 (Qlib 格式)
   - 日期格式：%Y-%m-%d
4. 写入临时 CSV 目录
5. 调用 qlib.contrib.data.dump_bin.DumpDataAll / DumpDataFix
6. 验证转换结果
```

**Qlib CSV 格式要求**：
```csv
date,symbol,$open,$high,$low,$close,$volume,$vwap,$factor,$change
2020-01-02,SH600519,12.5,12.8,12.3,12.6,1000000,12.5,1.0,0.008
```

**Qlib dump_bin 调用方式**：
```python
from qlib.contrib.data.dump_bin import DumpDataAll, DumpDataFix

# 全量转换
dumper = DumpDataAll(
    csv_path=csv_dir,
    qlib_dir=qlib_dir,
    freq="day",
    max_workers=16,
    date_field_name="date",
    symbol_field_name="symbol",
    include_fields=["open", "high", "low", "close", "volume", "vwap", "factor", "change"],
)
dumper.dump()

# 增量更新
dumper = DumpDataFix(
    csv_path=new_csv_path,
    qlib_dir=qlib_dir,
    freq="day"
)
dumper.dump()
```

**开发要点**：
- **这是本模块最重要的新代码**（约 200 行），其他文件基本是复用
- 必需字段：`$open`, `$high`, `$low`, `$close`, `$volume`, `$vwap`, `$factor`, `$change`
- `$change` 字段计算公式：`(close - preclose) / preclose`，用于涨跌停判断
  - **边界情况处理**：`preclose` 不会为 0（A 股价格最小 0.01 元），但新股上市首日 `preclose` 可能为 NaN
  - 实现：`np.where(preclose > 0, (close - preclose) / preclose, 0)` 
- `$factor` 字段：复权因子，用于交易单位取整。如果使用前复权价格，`$factor` 通常为 1.0
- `$vwap` 字段：`amount / (volume * 100)`，成交量加权均价
- **注意**：`$preclose` 不需要单独存为 Qlib 字段，但 `$change` 的计算依赖 `preclose`。在 `converter.py` 中计算 `$change` 后即可丢弃 `preclose` 列
- 转换完成后必须验证：随机抽查几只股票，对比 Qlib 读取的值与原始 CSV 值
- 注意：Qlib 的 `dump_bin` 默认写入 `~/.qlib/qlib_data/cn_data`，可通过 `qlib_dir` 参数修改

**参考代码**：Qlib 的 `qlib/contrib/data/dump_bin.py`、`qlib/data/storage/file_storage.py`

---

#### 2.2.7 `adjuster.py` — 复权处理

**功能**：对股票价格进行前复权（forward adjustment）处理。

**核心类**：`Adjuster`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `forward_adjust(df, dividend_df)` | 前复权，返回复权后 DataFrame |
| `compute_affine_factors(dividend_df)` | 计算仿射变换因子 (a, b) |
| `apply_affine_adjust(df, factors)` | 应用仿射变换 `v' = a * v + b` |
| `compute_multiplicative_factors(dividend_df)` | 计算纯乘法因子 |
| `apply_multiplicative_adjust(df, factors)` | 应用乘法调整 `v' = v * factor` |

**两种复权模式**：
| 模式 | 公式 | 适用场景 |
|------|------|----------|
| `origin` | `price' = a * price + b` | 精确复权，处理现金分红 |
| `equiv` | `price' = price * factor` | 简化复权，纯乘法，便于跨股票比较 |

**开发要点**：
- 直接复用 `new-quant/core/data/adjuster.py`（约 80 行）+ `myquant-clean/dataset/xtdata.py` 的复权逻辑
- **Phase 1 定位**：BaoStock 的 `adjustflag="2"` 已返回前复权数据，`adjuster.py` 作为备用模块保留，不参与主流程
- **Phase 2 启用**：当接入其他数据源（如 Tushare 不复权数据）时，启用 `adjuster.py` 在 converter 中进行复权
- 在 `converter.py` 中，复权是可选的——通过配置 `adjust: "forward" | "none"` 控制
- 两种模式可交叉校验：`origin` 和 `equiv` 的调整后价格比值应接近 1
- 注意除权除息日的处理：除权日当天价格会跳变，这是正常的

**参考代码**：`new-quant/core/data/adjuster.py`、`myquant-clean/dataset/xtdata.py`

---

#### 2.2.8 `calendar.py` — 交易日历

**功能**：提供 A 股交易日历查询和管理。

**核心类**：`TradingCalendar`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `is_trading_day(date)` | 判断是否为交易日 |
| `get_trading_days(start, end)` | 获取日期范围内的所有交易日 |
| `get_next_trading_day(date, n=1)` | 获取 N 个交易日后的日期 |
| `get_prev_trading_day(date, n=1)` | 获取 N 个交易日前的日期 |
| `count_trading_days(start, end)` | 计算两个日期之间的交易日数量 |
| `get_nth_trading_day_of_month(year, month, n)` | 获取某月第 N 个交易日 |

**开发要点**：
- 初期从 Qlib 的 `calendars/day.txt` 读取交易日列表，或从 BaoStock 的 `query_trade_dates` 获取
- Qlib 数据目录中自动包含交易日历文件：`~/.qlib/qlib_data/cn_data/calendars/day.txt`
- 格式：每行一个日期 `2020-01-02`
- 使用 `pd.DatetimeIndex` 存储，支持高效查询
- 注意：A 股有调休交易日（如周六补班），需要验证日历准确性

---

#### 2.2.9 `validator.py` — 数据校验

**功能**：校验本地数据的完整性和正确性。

**核心类**：`DataValidator`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `validate_ohlc_consistency(df)` | 验证 OHLC 价格关系：low <= min(open,close), high >= max(open,close) |
| `validate_no_negative_prices(df)` | 验证价格非负 |
| `validate_volume_positive(df)` | 验证成交量非负 |
| `validate_adjustment_consistency(df_origin, df_equiv)` | 验证两种复权模式的一致性 |
| `validate_coverage(symbols, start, end)` | 验证数据覆盖范围 |
| `validate_no_sudden_jumps(df)` | 验证价格无异常跳变（除权除息日除外） |
| `generate_report()` | 生成校验报告 |

**开发要点**：
- 每个校验方法返回 `(passed: bool, message: str)` 元组
- 校验报告应包含：通过/失败项数量、失败详情、建议修复措施
- 校验在数据同步后自动运行

---

#### 2.2.10 `cache.py` — 数据缓存（可选）

**功能**：提供 MultiIndex DataFrame 的 pickle 缓存，加速数据加载。

**核心类**：`DataCache`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `save(df, key)` | 缓存 DataFrame，生成 MD5 键 + .meta.json 旁路文件 |
| `load(key)` | 读取缓存 |
| `exists(key)` | 检查缓存是否存在 |
| `invalidate(key)` | 失效缓存 |
| `clear()` | 清理所有缓存 |

**开发要点**：
- 直接复用 `new-quant/core/data/cache.py`（约 80 行）
- 缓存键：MD5 哈希 `{start}_{end}_{dividend_type}_{sorted_stocks}`
- 缓存路径：`data/cache/`（可配置），而非 `~/.quant_cache/`
- 注意：使用 Qlib 数据格式后，数据加载已很快，此缓存层可能不需要。作为可选模块保留。

**参考代码**：`new-quant/core/data/cache.py`
---

## 3. 开发顺序

```
Day 1-2: 基础组件
  ├── base.py (DataSource ABC)
  ├── calendar.py (交易日历)
  └── adjuster.py (复权处理)

Day 3-4: 数据源接入
  ├── baostock_client.py (API 封装)
  ├── repository.py (本地仓库)
  └── sync_service.py (同步服务)

Day 5-6: Qlib 转换
  ├── converter.py (BaoStock → Qlib 转换器) ← 核心新代码
  └── validator.py (数据校验)

Day 7: 集成测试
  ├── 全量同步测试
  ├── 增量同步测试
  ├── Qlib 转换验证
  └── 数据校验测试
```

## 4. 依赖安装

```bash
# 安装 Qlib
pip install qlib

# 安装 BaoStock
pip install baostock

# 初始化 Qlib 数据目录
python -c "import qlib; qlib.init(provider_uri='./data/qlib_data', region='cn')"
```

## 5. 关键注意事项

### 5.1 股票代码格式统一

整个系统中，股票代码格式统一为 Qlib 格式：`SH600519` / `SZ000001`

| 来源 | 原始格式 | 转换后 |
|------|----------|--------|
| BaoStock | `sh.600519` | `SH600519` |
| Qlib | `SH600519` | `SH600519` |
| 用户输入 | `600519` | `SH600519` |

**转换函数**：
```python
def normalize_symbol(symbol: str) -> str:
    """统一股票代码格式"""
    symbol = symbol.replace('.', '').upper()
    if symbol.startswith('SH') or symbol.startswith('SZ'):
        return symbol
    if symbol.startswith('6'):
        return f'SH{symbol}'
    return f'SZ{symbol}'
```

### 5.2 BaoStock 数据已知问题

- BaoStock 的 `adjustflag="2"` 返回前复权数据，但某些旧股票的复权可能不准确
- 建议：使用 `adjustflag="1"`（后复权）+ 自行计算前复权，以获得更精确的结果
- 或者：使用 BaoStock 原始数据（`adjustflag="3"` 不复权）+ `adjuster.py` 自行复权
- BaoStock 的 API 有频率限制，需要控制请求间隔（`pause_seconds=0.5`）
- BaoStock 的 session 可能过期，需要重试机制（`baostock_client.py` 中已实现）

### 5.3 Qlib 数据格式注意事项

- Qlib 要求 `$factor` 字段存在，否则无法进行交易单位取整
- 如果使用前复权价格，`$factor` 可以为 1.0
- `$change` 字段用于涨跌停判断，计算公式：`(close - preclose) / preclose`
- Qlib 的 `dump_bin` 会创建 `calendars/day.txt` 和 `instruments/all.txt`
- 转换后需要验证：`qlib.init(provider_uri=...)` 后调用 `D.features()` 测试

### 5.4 增量更新策略

```python
def sync_and_convert():
    # 1. 从 Baostock 同步新数据到本地 CSV
    sync_service.sync_incremental(all_stocks)

    # 2. 找出有更新的股票
    updated_stocks = sync_service.get_updated_stocks()

    # 3. 只转换有更新的股票
    converter.convert_incremental(updated_stocks)

    # 4. 校验
    validator.validate_coverage(all_stocks, start, end)
```

## 6. 测试计划

| 测试文件 | 测试内容 | 测试方法 |
|----------|----------|----------|
| `tests/data/test_calendar.py` | 交易日历正确性 | 验证已知交易日/非交易日 |
| `tests/data/test_adjuster.py` | 复权正确性 | 已知除权案例，验证两种模式一致性 |
| `tests/data/test_baostock_client.py` | API 调用 | Mock 响应，验证字段重命名 |
| `tests/data/test_repository.py` | 本地存储 | 读写、upsert 去重、覆盖范围 |
| `tests/data/test_sync_service.py` | 数据同步 | 增量同步后数据一致性 |
| `tests/data/test_converter.py` | Qlib 转换 | 小数据集转换后验证 D.features() 可读取 |
| `tests/data/test_validator.py` | 数据校验 | 构造错误数据，验证校验逻辑 |
| `tests/data/test_integration.py` | 端到端 | 同步 -> 转换 -> 校验 完整流程 |

## 7. 验收标准

- [ ] 能成功同步全量 A 股日线数据（含已退市股票）
- [ ] 增量同步只拉取新数据，耗时 < 全量同步的 10%
- [ ] 转换后的 Qlib 数据可通过 `D.features()` 正常读取
- [ ] 复权后价格与 BaoStock 前复权价格偏差 < 0.1%
- [ ] 数据校验通过率 > 99%
- [ ] 所有单元测试通过
