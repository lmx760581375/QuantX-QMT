# 快速开始指南

## 安装依赖

确保已安装所有必要的依赖：

```bash
pip install pandas numpy tqdm matplotlib
```

## 基本使用

### 1. 最简单的例子

```python
from backtrade_v2 import FastBacktestEngine, SignalBasedStrategy

# 准备股票池
stock_pool = ['000001.SZ', '600000.SH', '601318.SH']

# 创建策略
strategy = SignalBasedStrategy(
    max_positions=3,      # 最多持有3只股票
    stop_profit=0.05,     # 止盈5%
    stop_loss=0.02        # 止损2%
)

# 创建回测引擎
engine = FastBacktestEngine(
    strategy=strategy,
    start_date='2021-02-05',
    end_date='2024-10-16',
    stock_pool=stock_pool,
    initial_cash=1000000,
    selector_config_path='./straregys/configs.json',
    selector_alias='少妇战法'
)

# 运行回测
results = engine.run()

# 打印结果
engine.print_results()
```

### 2. 使用测试脚本

直接运行提供的测试脚本：

```bash
# 测试新框架
python test_fast_backtest.py

# 对比新旧框架性能
python compare_frameworks.py
```

### 3. 自定义策略

```python
from backtrade_v2 import SignalBasedStrategy

class MyStrategy(SignalBasedStrategy):
    def __init__(self):
        super().__init__(
            max_positions=5,      # 自定义最大持仓
            stop_profit=0.10,     # 自定义止盈
            stop_loss=0.03        # 自定义止损
        )

    def _generate_trade_signals(self, selected_stocks, position_list,
                                close_prices, current_time):
        """自定义交易信号生成逻辑"""
        # selected_stocks 是预计算的选股结果
        # 你可以在此基础上添加额外的过滤或排序逻辑

        # 例如：按价格排序，优先买入低价股
        sorted_stocks = sorted(
            [s for s in selected_stocks if s in close_prices.index],
            key=lambda s: close_prices[s]
        )

        # 调用父类方法生成基础信号
        super()._generate_trade_signals(
            selected_stocks=sorted_stocks,
            position_list=position_list,
            close_prices=close_prices,
            current_time=current_time
        )
```

## 核心概念

### 预计算选股信号

新框架的核心优化是**预计算选股信号**：

```
传统流程：
数据加载 → 回测循环 {
    每天: 计算选股 → 生成交易信号 → 执行交易
}

新框架流程：
数据加载 → 一次性计算所有选股 → 回测循环 {
    每天: 查询选股结果 → 生成交易信号 → 执行交易
}
```

### 信号矩阵

选股信号以 MultiIndex DataFrame 的形式存储：

```python
# 查看信号统计
stats = context.get_signal_statistics()
print(stats)
# {
#     'total_signals': 2250,
#     'dates_with_signals': 900,
#     'avg_stocks_per_day': 2.5,
#     'max_stocks_per_day': 5,
#     'min_stocks_per_day': 0
# }

# 查询特定日期的选股结果
selected = context.get_selected_stocks(date='2024-01-15')
print(selected)  # ['000001.SZ', '600000.SH']
```

## 常见问题

### Q1: 如何加速首次运行？

使用信号缓存：

```python
# 首次运行后保存
context.save_signals('signals_cache.csv')

# 后续运行时加载
if context.load_signals_from_file('signals_cache.csv'):
    print("从缓存加载成功")
```

### Q2: 如何处理大规模股票池？

分批处理：

```python
# 将股票池分成多批
batch_size = 100
for i in range(0, len(all_stocks), batch_size):
    batch = all_stocks[i:i+batch_size]
    run_backtest(batch)
```

### Q3: 如何验证结果正确性？

运行对比脚本：

```bash
python compare_frameworks.py
```

这会同时运行新旧框架并对比结果。

### Q4: 内存不足怎么办？

1. 减少股票池大小
2. 缩短回测周期
3. 使用信号缓存，避免重复计算

## 性能优化建议

1. **首次运行**：保存信号缓存
2. **多次测试**：复用缓存文件
3. **大规模测试**：分批处理股票池
4. **调试阶段**：使用小股票池快速验证

## 下一步

- 阅读 [README.md](README.md) 了解架构设计
- 查看 [test_fast_backtest.py](../test_fast_backtest.py) 学习完整示例
- 运行 [compare_frameworks.py](../compare_frameworks.py) 验证性能提升

## 技术支持

如有问题，请检查：
1. 日志输出（详细的错误信息）
2. 数据完整性（股票数据是否正常）
3. 配置文件（configs.json 格式是否正确）
