# 选股器策略回测系统

基于现有的Selector类的量化策略回测框架。

## 文件说明

### 核心文件

1. **`straregys/strategy_selector.py`** - 选股器策略类
   - 继承自`Strategy`类
   - 使用现有的Selector类进行选股决策
   - 支持单个选股器或多个选股器组合使用

2. **`test_selector_strategy.py`** - 选股器策略回测测试脚本
   - 类似`test_factor_adjustment.py`的批量测试框架
   - 支持测试单个选股器或所有选股器组合

3. **`straregys/configs.json`** - 选股器配置文件
   - 定义各个选股器的参数
   - 支持启用/禁用特定选股器

## 使用方法

### 1. 运行单个选股器测试

```python
from test_selector_strategy import run_selector_test

# 测试"少妇战法"
results, context = run_selector_test(
    selector_alias="少妇战法",
    start_date='20210205',
    end_date='20241016',
    initial_cash=1000000
)
```

### 2. 运行所有选股器组合测试

```python
# 测试所有激活的选股器组合
results, context = run_selector_test(
    selector_alias=None,  # None表示使用所有选股器
    start_date='20210205',
    end_date='20241016',
    initial_cash=1000000
)
```

### 3. 批量测试所有选股器

直接运行测试脚本：

```bash
python test_selector_strategy.py
```

## 策略逻辑

### 选股逻辑
- 每天根据前一天的数据运行选股器
- 对股票池中的每只股票应用选股器过滤条件
- 合并所有激活选股器的选股结果

### 交易逻辑
1. **卖出条件**：
   - 股票不在当前选股结果中
   - 或达到止盈止损条件（盈利5%或亏损2%）

2. **买入条件**：
   - 股票在选股结果中
   - 当前没有持仓该股票
   - 限制买入前3只股票

### 数据需求
- 需要至少300天的历史数据用于选股器计算
- 支持的字段：close, open, high, low, volume, amount, turnover, date, vwap

## 输出结果

每个选股器测试会在独立的目录中生成：
- **日志文件** (`{selector_name}_test.log`) - 详细的运行日志
- **性能图表** (`{selector_name}_results.png`) - 净值曲线、收益率、回撤图表
- **统计信息** - 总收益率、最大回撤等关键指标

## 配置说明

### configs.json格式

```json
{
  "selectors": [
    {
      "class": "BBIKDJSelector",
      "alias": "少妇战法",
      "activate": true,
      "params": {
        "j_threshold": 15,
        "bbi_min_window": 20,
        "max_window": 120,
        "price_range_pct": 1,
        "bbi_q_threshold": 0.2,
        "j_q_threshold": 0.10
      }
    }
  ]
}
```

### 参数说明
- `class`: Selector类名（必须与Selector.py中的类名匹配）
- `alias`: 选股器别名（用于标识和日志）
- `activate`: 是否启用该选股器
- `params`: 传递给Selector构造函数的参数

## 支持的选股器

1. **BBIKDJSelector** (少妇战法)
2. **SuperB1Selector** (SuperB1战法)
3. **BBIShortLongSelector** (补票战法)
4. **PeakKDJSelector** (填坑战法)
5. **MA60CrossVolumeWaveSelector** (上穿60放量战法)
6. **BigBullishVolumeSelector** (暴力K战法)

## 注意事项

1. **数据要求**：确保有足够的历史数据（至少300天）
2. **内存使用**：对于大型股票池，建议适当限制股票数量
3. **运行时间**：选股器计算较为耗时，特别是多个选股器组合时
4. **参数调优**：可以根据需要调整configs.json中的参数

## 扩展开发

如需添加新的选股器：

1. 在`Selector.py`中实现新的Selector类
2. 在`configs.json`中添加配置
3. 重启测试脚本即可使用新的选股器
