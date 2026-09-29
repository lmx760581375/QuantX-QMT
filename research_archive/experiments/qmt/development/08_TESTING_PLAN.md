# 测试计划

> **版本**: v0.1.0 | **日期**: 2026-06-25 | **类型**: 持续伴随开发

---

## 1. 测试策略

```
单元测试 → 模块集成测试 → 端到端回归测试 → 业务逻辑验证
```

每个模块的测试覆盖率目标 >= 80%。

---

## 2. 测试目录结构

```
tests/
├── conftest.py                  # 全局 fixtures（合成数据生成器）
├── data/
│   ├── test_calendar.py         # 交易日历
│   ├── test_adjuster.py         # 复权处理
│   ├── test_baostock_client.py  # BaoStock API（Mock）
│   ├── test_repository.py       # 本地仓库
│   ├── test_sync_service.py     # 同步服务
│   ├── test_converter.py        # Qlib 转换器
│   └── test_validator.py        # 数据校验
├── factor/
│   ├── test_registry.py         # 因子注册
│   ├── test_expression.py       # 表达式因子
│   ├── test_duckdb_store.py     # DuckDB 存储
│   ├── test_engine.py           # 惰性计算引擎
│   └── test_alpha101.py         # Alpha101 因子
├── engine/
│   ├── test_cost.py             # 交易成本
│   ├── test_board.py            # 板别管理
│   ├── test_exchange.py         # 撮合引擎（重点）
│   ├── test_account.py          # 账户管理
│   ├── test_executor.py         # 订单执行
│   ├── test_engine.py           # 回测引擎
│   ├── test_t1.py               # T+1 规则
│   └── test_limit.py            # 涨跌停规则
├── strategy/
│   ├── test_loader.py           # 策略加载
│   └── test_templates.py        # 声明式模板
├── portfolio/
│   └── test_equal_weight.py     # 等权组合
├── analysis/
│   ├── test_metrics.py          # 绩效指标
│   └── test_plot.py             # 可视化
└── integration/
    ├── test_synthetic_e2e.py    # 合成数据端到端测试
    ├── test_real_data_e2e.py    # 真实数据端到端测试
    └── test_consistency.py      # 与 myquant-strategy 一致性对比
```

---

## 3. 合成数据生成器 (`conftest.py`)

提供共享 fixtures，生成可控的合成行情数据：

```python
@pytest.fixture
def synthetic_market_data():
    """生成 10 只股票 x 100 个交易日的合成行情数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-02', periods=100, freq='B')
    symbols = [f'SH600{i:03d}' for i in range(10)]
    # 生成随机游走价格
    data = []
    for sym in symbols:
        close = 10 + np.cumsum(np.random.randn(100) * 0.1)
        open_p = close + np.random.randn(100) * 0.05
        high = np.maximum(open_p, close) + np.abs(np.random.randn(100) * 0.05)
        low = np.minimum(open_p, close) - np.abs(np.random.randn(100) * 0.05)
        volume = np.random.randint(1000000, 10000000, 100)
        for i, d in enumerate(dates):
            data.append({
                'date': d, 'symbol': sym,
                'open': open_p[i], 'high': high[i], 'low': low[i],
                'close': close[i], 'volume': volume[i],
                'change': (close[i] - close[i-1]) / close[i-1] if i > 0 else 0
            })
    return pd.DataFrame(data).set_index(['date', 'symbol'])

@pytest.fixture
def mock_qlib(synthetic_market_data):
    """Mock Qlib 的 D.features() 调用，避免依赖真实 Qlib 数据"""
    with patch('qlib.data.D.features') as mock_features:
        mock_features.return_value = synthetic_market_data
        yield mock_features

@pytest.fixture
def mock_qlib_with_limits(synthetic_market_data):
    """生成包含涨停/跌停/停牌场景的合成数据"""
    df = synthetic_market_data.copy()
    # 构造涨停日：SH600001 在第 50 天涨停
    preclose = df.loc[(pd.Timestamp('2020-03-13'), 'SH600001'), 'close']
    limit_up_price = preclose * 1.10
    df.loc[(pd.Timestamp('2020-03-16'), 'SH600001'), :] = [
        limit_up_price, limit_up_price, limit_up_price, limit_up_price, 0, 0.10
    ]
    # 构造停牌日：SH600002 在第 60 天停牌
    df.loc[(pd.Timestamp('2020-03-30'), 'SH600002'), :] = [np.nan] * 6
    return df
```

## 3b. 测试数据隔离策略

**原则**：所有测试不依赖外部数据源，BaoStock 网络不可用时测试也能通过。

```python
@pytest.fixture
def tmp_data_dir(tmp_path):
    """创建临时数据目录，测试结束后自动清理"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "qlib_data").mkdir()
    (data_dir / "warehouse").mkdir()
    return data_dir

@pytest.fixture
def tmp_duckdb(tmp_path):
    """创建临时 DuckDB 数据库"""
    db_path = tmp_path / "test_factors.db"
    store = DuckDBFactorStore(str(db_path))
    yield store
    store.conn.close()
```

**测试标记**：
```python
# 需要真实数据的测试使用 @pytest.mark.real_data 标记
@pytest.mark.real_data
def test_baostock_real_connection():
    ...

# 运行测试时跳过真实数据测试
# pytest tests/ -v -m "not real_data"
```

---

## 4. 关键测试场景

### 4.1 撮合引擎测试（最重要）

| 测试用例 | 输入 | 预期输出 |
|----------|------|----------|
| 涨停买入 | 买入价 = 涨停价 | 订单被拒绝 |
| 跌停卖出 | 卖出价 = 跌停价 | 订单被拒绝 |
| 一字板买卖 | open=high=low=close=涨停价 | 买卖都被拒绝 |
| T+1 卖出 | 当日买入后卖出 | 卖出被拒绝 |
| 停牌交易 | $close 为 NaN | 不可交易 |
| 现金不足 | 买入金额 > 现金 | 自动调整数量 |
| 正常交易 | 价格在涨跌停之间 | 正常成交 |

### 4.2 一致性测试

```python
def test_consistency_with_myquant_strategy():
    """使用相同数据和策略参数，对比 QuantX 和 myquant-strategy 的回测结果"""
    # 1. 加载相同数据
    # 2. 使用相同策略参数
    # 3. 运行双方回测
    # 4. 对比总收益率、最大回撤、Sharpe 比率
    # 5. 偏差 < 1%
```

---

## 5. 运行测试

```bash
# 所有测试
pytest tests/ -v

# 单个模块
pytest tests/engine/ -v

# 带覆盖率
pytest tests/ --cov=quantx --cov-report=html

# 只运行集成测试
pytest tests/integration/ -v

# 跳过需要真实数据的测试
pytest tests/ -v -m "not real_data"
```
