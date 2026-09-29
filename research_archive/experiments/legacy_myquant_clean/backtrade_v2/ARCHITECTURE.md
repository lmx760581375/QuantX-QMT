# 架构设计文档

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│  test_fast_backtest.py  │  example_simple.py  │  Custom Scripts │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      策略层 (Strategy Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│                    SignalBasedStrategy                           │
│  - 交易信号生成                                                    │
│  - 止盈止损逻辑                                                    │
│  - 仓位管理                                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      引擎层 (Engine Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│                    FastBacktestEngine                            │
│  - 回测流程控制                                                    │
│  - 性能统计                                                        │
│  - 结果计算                                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      上下文层 (Context Layer)                     │
├─────────────────────────────────────────────────────────────────┤
│                      SignalContext                               │
│  - 数据管理 (继承自Context)                                        │
│  - 选股信号管理                                                    │
│  - 高效查询接口                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      选股层 (Selector Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│                   PrecomputedSelector                            │
│  - 选股器加载                                                      │
│  - 信号预计算 (向量化)                                             │
│  - 信号查询 (O(log n))                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      数据层 (Data Layer)                          │
├─────────────────────────────────────────────────────────────────┤
│  MultiIndex DataFrame  │  Signal Matrix  │  Account Data        │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. PrecomputedSelector（预计算选股器）

**职责**：
- 加载选股器配置
- 预计算所有日期的选股信号
- 提供高效的信号查询接口

**关键方法**：
```python
class PrecomputedSelector:
    def __init__(config_path, selector_alias)
        # 加载选股器实例

    def precompute_signals(multiindex_df, date_list, stock_pool)
        # 向量化计算所有选股信号
        # 返回: MultiIndex DataFrame (date, stock) -> bool

    def get_selected_stocks(date)
        # O(log n) 查询指定日期的选股结果
        # 返回: List[str] 股票代码列表

    def save_signals(filepath)
        # 保存信号矩阵到文件（缓存）

    def load_signals(filepath)
        # 从文件加载信号矩阵
```

**数据结构**：
```python
# 信号矩阵 (稀疏存储)
signal_matrix = pd.DataFrame({
    'date': [2024-01-01, 2024-01-01, 2024-01-02, ...],
    'stock': ['000001.SZ', '600000.SH', '000001.SZ', ...],
    'selected': [True, True, True, ...]
}).set_index(['date', 'stock'])
```

### 2. SignalContext（信号上下文）

**职责**：
- 继承原有 Context 的所有功能
- 集成 PrecomputedSelector
- 在数据加载时自动预计算信号

**关键方法**：
```python
class SignalContext(Context):
    def __init__(start_date, end_date, stock_pool,
                 selector_config_path, selector_alias)
        # 初始化父类和选股器

    def load_data()
        # 1. 调用父类加载市场数据
        # 2. 预计算选股信号

    def get_selected_stocks(date=None)
        # 查询选股结果（默认当前日期）

    def get_signal_statistics()
        # 获取信号统计信息
```

**继承关系**：
```
Context (原有)
  ↓ 继承
SignalContext (新增)
  - 保持所有原有功能
  - 新增选股信号管理
```

### 3. FastBacktestEngine（快速回测引擎）

**职责**：
- 控制回测流程
- 统计性能指标
- 计算回测结果

**关键流程**：
```python
class FastBacktestEngine:
    def run():
        # 1. 加载数据并预计算信号（一次性）
        context.load_data()

        # 2. 初始化策略
        strategy.initialize(context, account)

        # 3. 主回测循环（只执行交易逻辑）
        while context.next():
            strategy.before_trading()
            strategy.handle_data()      # 查询信号，生成交易
            strategy.after_trading()    # 执行交易
            account.update_balance()

        # 4. 计算结果
        calculate_results()
```

**性能统计**：
```python
timing = {
    'data_loading': 10.5,        # 数据加载耗时
    'signal_precompute': 8.2,    # 信号预计算耗时
    'backtest_loop': 5.0,        # 回测循环耗时
    'total': 23.7                # 总耗时
}
```

### 4. SignalBasedStrategy（信号策略基类）

**职责**：
- 简化的策略接口
- 基于预计算信号的交易逻辑
- 止盈止损管理

**关键方法**：
```python
class SignalBasedStrategy(Strategy):
    def handle_data(account, yesterday_data):
        # 查询预计算的信号（无计算开销）
        selected_stocks = context.get_selected_stocks()

        # 生成交易信号
        _generate_trade_signals(selected_stocks, ...)

    def _generate_trade_signals(selected_stocks, ...):
        # 卖出逻辑：不在选股结果中 或 止盈止损
        # 买入逻辑：在选股结果中 且 有可用槽位

    def after_trading(account, current_data):
        # 执行交易信号
        # 等权重分仓
```

## 数据流

### 初始化阶段

```
1. 用户创建 FastBacktestEngine
   ↓
2. Engine 创建 SignalContext
   ↓
3. SignalContext 创建 PrecomputedSelector
   ↓
4. PrecomputedSelector 加载选股器配置
```

### 数据加载阶段

```
1. Engine.run() 调用 context.load_data()
   ↓
2. Context 加载市场数据 (MultiIndex DataFrame)
   ↓
3. PrecomputedSelector.precompute_signals()
   ↓
4. 向量化计算所有日期的选股信号
   ↓
5. 生成信号矩阵 (稀疏存储)
```

### 回测循环阶段

```
For each trading day:
  1. context.next() → 移动到下一个交易日
     ↓
  2. strategy.handle_data()
     ↓
  3. context.get_selected_stocks() → O(log n) 查表
     ↓
  4. strategy._generate_trade_signals() → 生成买卖信号
     ↓
  5. strategy.after_trading() → 执行交易
     ↓
  6. account.update_balance() → 更新账户
```

## 性能优化

### 1. 向量化计算

```python
# 原框架：逐日循环计算
for date in dates:
    for stock in stocks:
        # 计算指标
        # 判断条件
        # 记录结果

# 新框架：向量化计算
df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4  # 一次性计算所有
df['condition'] = (df['BBI'] > df['BBI'].shift(1)) & ...  # 向量化判断
selected = df[df['condition']]  # 一次性筛选
```

### 2. 稀疏存储

```python
# 只存储选中的股票（selected=True）
# 900天 × 50股票 × 5%选中率 = 2250条记录
# 相比全量存储（900×50=45000），节省 95% 内存
```

### 3. 高效查询

```python
# 使用 pandas MultiIndex 的二分查找
# 时间复杂度: O(log n)
selected = signal_matrix.loc[date]  # 毫秒级查询
```

### 4. 缓存机制

```python
# 首次运行：预计算 + 保存
context.save_signals('cache.csv')

# 后续运行：直接加载
if context.load_signals_from_file('cache.csv'):
    # 跳过预计算，直接进入回测循环
```

## 扩展性

### 自定义选股器

```python
# 1. 在 Selector_gemini.py 中实现新选股器
class MySelector:
    def select(self, date, data):
        # 向量化选股逻辑
        return selected_stocks

# 2. 在 configs.json 中配置
{
    "class": "MySelector",
    "alias": "我的选股器",
    "params": {...}
}

# 3. 直接使用
engine = FastBacktestEngine(
    selector_alias="我的选股器",
    ...
)
```

### 自定义策略

```python
class MyStrategy(SignalBasedStrategy):
    def _generate_trade_signals(self, selected_stocks, ...):
        # 自定义交易逻辑
        # 可以在 selected_stocks 基础上添加额外过滤

        # 例如：按价格排序
        sorted_stocks = sorted(
            selected_stocks,
            key=lambda s: close_prices[s]
        )

        # 调用父类方法或完全自定义
        super()._generate_trade_signals(sorted_stocks, ...)
```

### 批量计算接口（未来优化）

```python
# 当前：逐日调用 select()
for date in dates:
    selected = selector.select(date, data)

# 未来：批量接口
class MySelector:
    def select_batch_dates(self, dates, data):
        # 一次性计算所有日期
        # 进一步提升性能
        return signal_matrix
```

## 测试与验证

### 单元测试

```python
# 测试预计算选股器
def test_precomputed_selector():
    selector = PrecomputedSelector(...)
    signals = selector.precompute_signals(...)
    assert len(signals) > 0

# 测试信号查询
def test_signal_query():
    selected = context.get_selected_stocks(date)
    assert isinstance(selected, list)
```

### 集成测试

```python
# 测试完整回测流程
def test_full_backtest():
    engine = FastBacktestEngine(...)
    results = engine.run()
    assert results['performance']['total_return'] is not None
```

### 性能测试

```python
# 对比新旧框架
python compare_frameworks.py
```

### 结果验证

```python
# 验证结果一致性
old_results = run_old_framework(...)
new_results = run_new_framework(...)
assert abs(old_results['total_return'] - new_results['total_return']) < 0.01
```

## 最佳实践

1. **首次运行保存缓存**
   ```python
   context.save_signals('signals_cache.csv')
   ```

2. **后续运行加载缓存**
   ```python
   context.load_signals_from_file('signals_cache.csv')
   ```

3. **合理设置股票池大小**
   - 开发调试：10-30只
   - 正式回测：50-200只
   - 大规模测试：分批处理

4. **监控内存使用**
   ```python
   import psutil
   process = psutil.Process()
   print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

5. **日志级别控制**
   ```python
   # 开发调试：INFO
   logging.basicConfig(level=logging.INFO)

   # 生产运行：WARNING
   logging.basicConfig(level=logging.WARNING)
   ```

## 常见问题

### Q: 为什么还要逐日调用 select()？

A: 当前 Selector_gemini 的接口是单日查询。未来可以扩展批量接口 `select_batch_dates()` 进一步优化。

### Q: 信号矩阵占用多少内存？

A: 稀疏存储，只记录选中的股票。例如 900天×50股票×5%选中率 = 2250条记录，约几MB。

### Q: 如何处理大规模股票池？

A:
1. 分批处理
2. 使用缓存
3. 考虑分布式计算（Dask/Ray）

### Q: 结果为什么有微小差异？

A: 可能原因：
1. 浮点数精度
2. 随机数种子
3. 数据版本差异

通常差异 < 0.01% 可以接受。

## 未来优化方向

1. **批量计算接口**
   - 扩展 Selector 支持 `select_batch_dates()`
   - 一次性计算所有日期，进一步提速

2. **并行计算**
   - 多进程并行计算不同选股器
   - 使用 joblib 或 multiprocessing

3. **增量更新**
   - 支持增量更新信号矩阵
   - 只计算新增日期

4. **分布式计算**
   - 使用 Dask 处理超大规模数据
   - 使用 Ray 分布式计算

5. **GPU 加速**
   - 使用 cuDF 加速 pandas 操作
   - 使用 RAPIDS 加速向量化计算

## 参考资料

- [README.md](README.md) - 功能介绍
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [../REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) - 重构总结
- [../example_simple.py](../example_simple.py) - 简单示例
- [../test_fast_backtest.py](../test_fast_backtest.py) - 完整测试
- [../compare_frameworks.py](../compare_frameworks.py) - 性能对比
