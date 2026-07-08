# QuantX

A股量化研究与回测系统。

## 架构

- **数据层**: BaoStock → Qlib 二进制格式，零外部依赖
- **因子层**: 表达式引擎 + DuckDB 复杂因子存储
- **回测引擎**: Policy 架构（选股/调仓/执行），规则+模型双模式
- **策略层**: 三个 Policy 接口（StockSelector / RebalanceStrategy / ExecutionStrategy）

## 快速开始

```bash
# 安装依赖
pip install "baostock>=0.9.2" pandas numpy pyyaml pydantic matplotlib duckdb

# 同步数据
python -m quantx.tools.sync_data --start 2020-01-01

# 运行回测
python -m quantx.tools.run_backtest --strategy strategies/ma_cross.py --start 2020-01-01 --end 2025-12-31
```

## 开发状态

- [x] 数据层
- [ ] 因子层
- [ ] 回测引擎
- [ ] 策略层
- [ ] 评估层
- [ ] 工具链

## 许可

MIT
