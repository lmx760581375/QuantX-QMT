# 评估层开发计划

> **版本**: v0.1.0 | **日期**: 2026-06-25 | **预计工期**: 3-5 天 | **依赖**: 回测引擎 (03_BACKTEST_ENGINE.md)

---

## 1. 目标

评估层负责回测结果的绩效分析、收益归因和可视化。

---

## 2. 文件清单

```
quantx/core/analysis/
├── __init__.py              # 模块入口
├── metrics.py               # 绩效指标计算
├── attribution.py           # 收益归因（Phase 2）
├── report.py                # 回测报告生成
└── plot.py                  # 可视化
```

---

## 2.2 各文件详细说明

### 2.2.1 `metrics.py` — 绩效指标

**核心类**：`PerformanceMetrics`

**指标列表**：

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| 总收益率 | `(final_value - init_cash) / init_cash` | 整个回测期的总收益 |
| 年化收益率 | `(1 + total_return)^(252/days) - 1` | 按年化折算 |
| 最大回撤 | `max(1 - current_value / running_max)` | 最大净值回撤幅度 |
| 回撤持续期 | 回撤区间的交易日数 | 最长回撤修复时间 |
| Sharpe 比率 | `(daily_return.mean() - rf) / daily_return.std() * sqrt(252)` | 风险调整后收益，rf=2% |
| Sortino 比率 | `(daily_return.mean() - rf) / downside_std * sqrt(252)` | 只惩罚下行波动 |
| Calmar 比率 | `annual_return / max_drawdown` | 收益回撤比 |
| 胜率 | 盈利交易数 / 总交易数 | FIFO 匹配法 |
| 日均换手率 | `daily_turnover.mean()` | 平均每日买卖金额比例 |
| 总交易成本 | 佣金 + 印花税 + 过户费 | 交易成本合计 |
| 超额收益 | `total_return - benchmark_return` | 相对基准的超额收益 |
| Beta | `Cov(strategy_return, benchmark_return) / Var(benchmark_return)` | 系统性风险暴露 |
| Alpha | `strategy_return - rf - Beta * (benchmark_return - rf)` | 超额收益（CAPM） |
| 信息比率 | `(strategy_return - benchmark_return).mean() / tracking_error.std() * sqrt(252)` | 主动管理能力 |

**开发要点**：
- 直接复用 `new-quant/core/engine/account.py` 的 `get_performance_metrics()` 方法（约 100 行）
- 所有指标需提供手工计算对比验证
- 日收益率计算使用 `(today_value - yesterday_value) / yesterday_value`
- 基准数据加载：从 Qlib 的 `D.features(['SH000300'], ['$close'], start, end)` 加载 CSI 300 指数
- 基准收益率需包含分红再投资；如果 Qlib 数据中只有价格指数，则使用价格收益率
- 如果基准数据不可用，Beta/Alpha/信息比率等指标会跳过并记录警告

### 2.2.2 `plot.py` — 可视化

**核心函数**：

| 函数 | 功能 |
|------|------|
| `plot_equity_curve(daily_values, benchmark=None)` | 净值曲线 + 回撤曲线（双面板） |
| `plot_monthly_heatmap(daily_returns)` | 月度收益热力图 |
| `plot_trade_distribution(trades)` | 交易分布图（买卖点标注） |
| `plot_drawdown_periods(daily_values)` | 回撤期间标注 |

**开发要点**：直接复用 `new-quant/core/analysis/plot.py`（约 80 行）

### 2.2.3 `report.py` — 回测报告

**功能**：生成多种格式的回测报告。

**输出格式**：

| 格式 | 文件 | 用途 |
|------|------|------|
| JSON | `metrics.json` | 程序化分析、参数搜索对比 |
| CSV | `daily_equity.csv` | 每日净值序列，供 Excel 分析 |
| CSV | `trades.csv` | 逐笔成交记录 |
| PNG | `equity_curve.png` | 净值曲线图 |
| PNG | `monthly_heatmap.png` | 月度收益热力图 |
| Markdown | `report.md` | 人类可读的回测报告 |

**回测报告内容**：
- 绩效指标摘要表
- 净值曲线图
- 月度收益热力图
- 交易统计（总交易次数、日均换手率、总交易成本）
- 最大回撤区间（起止日期、持续时间、回撤幅度）
- 与基准对比（超额收益、Beta、Alpha、信息比率）

---

## 3. 验收标准

- [ ] Sharpe 比率与手工计算一致（误差 < 0.01）
- [ ] 净值曲线 + 回撤曲线正确绘制
- [ ] 回测报告包含所有关键指标
