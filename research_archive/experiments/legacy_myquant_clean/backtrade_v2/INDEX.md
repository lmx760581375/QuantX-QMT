# 文档索引

## 📚 文档导航

### 快速开始

1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ 推荐首先阅读
   - 5分钟快速上手
   - 基本使用示例
   - 常见问题解答

2. **[../example_simple.py](../example_simple.py)**
   - 最简单的完整示例
   - 可直接运行
   - 包含详细注释

### 详细文档

3. **[README.md](README.md)**
   - 框架概述
   - 核心优化说明
   - 使用方法
   - 性能对比
   - 兼容性说明

4. **[ARCHITECTURE.md](ARCHITECTURE.md)**
   - 系统架构设计
   - 核心组件详解
   - 数据流分析
   - 性能优化原理
   - 扩展性说明

5. **[../REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)**
   - 重构背景和动机
   - 问题分析
   - 解决方案
   - 性能提升数据
   - 技术亮点

### 测试脚本

6. **[../test_fast_backtest.py](../test_fast_backtest.py)**
   - 完整的测试脚本
   - 批量测试多个选股器
   - 生成图表和报告

7. **[../compare_frameworks.py](../compare_frameworks.py)**
   - 新旧框架性能对比
   - 结果一致性验证
   - 性能提升统计

## 📖 阅读路线

### 路线 1: 快速上手（10分钟）

```
QUICKSTART.md → example_simple.py → 运行测试
```

适合：想快速使用新框架的用户

### 路线 2: 深入理解（30分钟）

```
README.md → ARCHITECTURE.md → REFACTORING_SUMMARY.md
```

适合：想了解设计原理和优化细节的用户

### 路线 3: 完整学习（1小时）

```
1. QUICKSTART.md (5分钟)
2. example_simple.py (5分钟)
3. README.md (10分钟)
4. ARCHITECTURE.md (20分钟)
5. REFACTORING_SUMMARY.md (10分钟)
6. 运行测试脚本 (10分钟)
```

适合：想全面掌握新框架的用户

## 🎯 按需查阅

### 我想...

- **快速开始使用** → [QUICKSTART.md](QUICKSTART.md)
- **看一个简单例子** → [example_simple.py](../example_simple.py)
- **了解性能提升** → [README.md](README.md) 的"性能对比"部分
- **理解架构设计** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **了解重构原因** → [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)
- **验证结果正确性** → 运行 [compare_frameworks.py](../compare_frameworks.py)
- **自定义策略** → [QUICKSTART.md](QUICKSTART.md) 的"自定义策略"部分
- **自定义选股器** → [ARCHITECTURE.md](ARCHITECTURE.md) 的"扩展性"部分
- **解决问题** → [QUICKSTART.md](QUICKSTART.md) 的"常见问题"部分

## 📁 文件结构

```
backtrade_v2/
├── INDEX.md                    # 本文档（文档索引）
├── QUICKSTART.md              # 快速开始指南 ⭐
├── README.md                  # 框架说明文档
├── ARCHITECTURE.md            # 架构设计文档
├── __init__.py                # 模块初始化
├── precomputed_selector.py   # 预计算选股器
├── signal_context.py          # 信号上下文
├── fast_engine.py             # 快速回测引擎
└── signal_strategy.py         # 信号策略基类

../
├── REFACTORING_SUMMARY.md     # 重构总结
├── example_simple.py          # 简单示例 ⭐
├── test_fast_backtest.py      # 测试脚本
└── compare_frameworks.py      # 性能对比脚本
```

## 🔍 核心概念速查

### 预计算选股信号

```python
# 原框架：每次迭代都计算
for date in dates:
    selected = selector.select(date, data)  # 重复计算

# 新框架：一次性预计算
signal_matrix = precompute_all_signals()  # 只计算一次
for date in dates:
    selected = signal_matrix.loc[date]  # 查表
```

### 信号矩阵

```python
# MultiIndex DataFrame: (date, stock) -> bool
signal_matrix = pd.DataFrame({
    'date': [2024-01-01, 2024-01-01, ...],
    'stock': ['000001.SZ', '600000.SH', ...],
    'selected': [True, True, ...]
}).set_index(['date', 'stock'])
```

### 使用流程

```python
# 1. 创建策略
strategy = SignalBasedStrategy()

# 2. 创建引擎
engine = FastBacktestEngine(
    strategy=strategy,
    selector_config_path='configs.json',
    selector_alias='少妇战法'
)

# 3. 运行回测
results = engine.run()
```

## 🚀 性能提升

| 指标 | 原框架 | 新框架 | 提升 |
|------|--------|--------|------|
| 选股计算 | ~900秒 | ~10秒 | **90倍** |
| 总耗时 | ~915秒 | ~25秒 | **36倍** |

*测试条件：50只股票，900个交易日*

## ✅ 兼容性

- ✅ 使用相同的 Account 类
- ✅ 使用相同的数据加载逻辑
- ✅ SignalContext 继承自 Context
- ✅ 结果格式完全一致
- ✅ 支持原有的 configs.json

## 🎓 学习资源

### 代码示例

1. **最简单** - [example_simple.py](../example_simple.py)
2. **完整测试** - [test_fast_backtest.py](../test_fast_backtest.py)
3. **性能对比** - [compare_frameworks.py](../compare_frameworks.py)

### 文档资源

1. **快速上手** - [QUICKSTART.md](QUICKSTART.md)
2. **功能说明** - [README.md](README.md)
3. **架构设计** - [ARCHITECTURE.md](ARCHITECTURE.md)
4. **重构总结** - [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)

## 💡 提示

- 📌 标记 ⭐ 的文档推荐优先阅读
- 🔧 遇到问题先查看 [QUICKSTART.md](QUICKSTART.md) 的"常见问题"
- 📊 想了解性能提升查看 [README.md](README.md) 的"性能对比"
- 🏗️ 想了解设计原理查看 [ARCHITECTURE.md](ARCHITECTURE.md)
- 🎯 想快速开始直接运行 [example_simple.py](../example_simple.py)

## 📞 获取帮助

1. 查看文档中的"常见问题"部分
2. 查看测试脚本的日志输出
3. 运行 `python example_simple.py` 验证环境
4. 运行 `python compare_frameworks.py` 对比结果

---

**更新时间**: 2026-01-25
**框架版本**: V2.0
**维护状态**: ✅ 活跃维护
