# 🤖 自动因子挖掘系统

基于大语言模型的量化因子自动挖掘和回测评估系统。

## 📋 功能特性

- 🔍 **智能因子生成**: 使用LLM自动设计创新量化因子
- 🏗️ **策略自动构建**: 基于模板自动生成交易策略
- 📈 **自动化回测**: 集成完整的回测流程
- 📊 **智能评估**: 多维度因子表现分析和评分
- 💾 **结果持久化**: JSON格式保存完整评估报告

## 🏗️ 系统架构

```
auto_factor_mining/
├── main.py                 # 主程序入口
├── config.py              # 系统配置
├── factor_generator.py    # 因子生成器 (LLM集成)
├── strategy_generator.py  # 策略生成器
├── auto_tester.py         # 自动化测试器
├── result_analyzer.py     # 结果分析器
├── prompts/               # 提示词模板
│   └── factor_generation.txt
├── generated_factors/     # 生成的因子代码
├── generated_strategies/  # 生成的策略代码
├── results/              # 评估结果 (JSON)
└── README.md
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 设置OpenAI API Key
export OPENAI_API_KEY="<REDACTED:CREDENTIAL>"

# 进入项目目录
cd /path/to/myquant-clean/auto_factor_mining
```

### 2. 运行完整流水线

生成5个复杂度为3的因子，并自动完成策略生成、回测和评估：

```bash
python main.py --mode full --num-factors 5 --complexity 3
```

### 3. 测试单个因子

测试指定的因子（需要先有因子文件）：

```bash
python main.py --mode single --factor-name alpha_191
```

## 📊 评估指标

系统使用以下指标评估因子表现：

- **总收益率 (40%)**: 策略的总收益率
- **年化收益率 (25%)**: 年化收益率表现
- **夏普比率 (20%)**: 风险调整后的收益
- **胜率 (10%)**: 盈利交易占比
- **最大回撤 (5%)**: 最大亏损幅度

最终生成**综合评分**（0-100分）用于因子排序。

## 🎯 使用示例

### 生成创新因子

```python
from factor_generator import FactorGenerator

generator = FactorGenerator()
factor_name = generator.generate_factor(complexity=3)
print(f"生成因子: {factor_name}")
```

### 运行批量测试

```python
from auto_tester import AutoTester

tester = AutoTester()
results = tester.run_batch_backtests([
    ('alpha_191', 'Strategy_alpha_191'),
    ('alpha_192', 'Strategy_alpha_192'),
])
```

### 分析结果

```python
from result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()
analysis = analyzer.analyze_batch_results(results)
analyzer.print_summary_report(analysis)
```

## 📁 输出文件

### 因子文件 (`generated_factors/`)
```python
# alpha_191.py
"""
alpha_191 - 基于成交量的动量因子
生成时间: 2024-01-15 10:30:00
"""

def alpha_191(data):
    """
    结合成交量和价格动量的创新因子
    """
    volume_ma = data['volume'].rolling(20).mean()
    price_momentum = data['close'] / data['close'].shift(10) - 1
    return (price_momentum * volume_ma).rank()
```

### 策略文件 (`generated_strategies/`)
```python
# Strategy_alpha_191.py
"""
自动生成的策略 - 基于alpha_191
生成时间: 2024-01-15 10:30:00
基于策略模板: strategy_01.py
"""

from backtrade.strategy import Strategy
# ... 自动生成的策略代码
```

### 结果文件 (`results/`)
```json
{
  "total_factors": 5,
  "successful_factors": 4,
  "success_rate": 0.8,
  "best_factor": {
    "factor_name": "alpha_191",
    "overall_score": 78.5,
    "total_return_pct": 45.2
  },
  "avg_overall_score": 65.3,
  "detailed_results": [...]
}
```

## ⚙️ 配置说明

### LLM配置 (`config.py`)
```python
LLM_CONFIG = {
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 1000,
    'api_key': os.getenv('OPENAI_API_KEY'),
}
```

### 回测配置
```python
BACKTEST_CONFIG = {
    'start_date': '20210205',
    'end_date': '20251016',
    'initial_cash': 1000000,
    'stock_pool': ['000001.SZ', '000002.SZ', ...],
}
```

## 🔧 自定义开发

### 添加新的提示词模板

在 `prompts/` 目录下创建新的提示词文件：

```txt
# prompts/custom_factor.txt
你是一个高级量化分析师，设计[特定类型]因子...

因子要求：
1. 使用[特定字段]
2. 实现[特定逻辑]
...
```

### 扩展评估指标

在 `ResultAnalyzer` 类中添加新的评估方法：

```python
def calculate_custom_metric(self, result: Dict) -> float:
    """计算自定义指标"""
    # 实现自定义评估逻辑
    return custom_score
```

## 📈 性能优化

- **并行处理**: 可在 `AutoTester` 中添加多进程支持
- **缓存机制**: 对相同因子避免重复计算
- **增量学习**: 基于历史表现优化LLM提示词

## 🐛 故障排除

### 常见问题

1. **LLM调用失败**
   - 检查 `OPENAI_API_KEY` 是否正确设置
   - 确认网络连接正常

2. **因子验证失败**
   - 检查因子代码语法
   - 确认使用了允许的数据字段

3. **回测失败**
   - 检查数据文件是否存在
   - 确认策略类定义正确

### 日志调试

系统会生成详细的日志文件 `auto_factor_mining.log`：

```bash
tail -f auto_factor_mining.log  # 实时查看日志
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。
