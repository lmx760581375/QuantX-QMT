# baostock_clean

一版新的、CLI 优先的 clean baostock 回测系统。

## 设计目标

- 数据同步、指标构建、公式预览、回测运行全部走命令行
- 指标分为两类：
  - `symbol`：单支股票纵向时间序列指标
  - `cross_section`：同一交易日横截面指标
- 内置 `indicator_sets`，用于替代旧版 processor 链
- 增加指标时只需要改配置里的公式，不需要改 Python 代码
- 保留本地 CSV 落盘，方便审计和重建

## 命令

```bash
python -m baostock_clean sync --config baostock_clean/configs/demo_formula.py --refresh-universe
python -m baostock_clean build-indicators --config baostock_clean/configs/demo_formula.py
python -m baostock_clean formula-preview --config baostock_clean/configs/demo_formula.py --name ema_diff --expr "EMA(close,12)-EMA(close,26)" --scope symbol --stock sh.600519
python -m baostock_clean backtest --config baostock_clean/configs/demo_formula.py
python -m baostock_clean backtest --config baostock_clean/configs/shuijiao_best_v2.py
```

## 配置

配置文件支持：

- `data_config`
- `runtime_config`
- `account_config`
- `strategy_config`
- `indicator_sets`
- `indicator_configs`

其中 `indicator_configs` 示例：

```python
indicator_configs = [
    {"name": "ema20", "expr": "EMA(close, 20)", "scope": "symbol"},
    {"name": "cs_rank_close", "expr": "RANK_PCT(close, ascending=False)", "scope": "cross_section"},
]
```

策略支持：

- `formula`
- `template`

推荐优先使用 `template`。策略主要由模板字段描述，例如：

```python
strategy_config = {
    "type": "template",
    "trade_price": "close",
    "lot_size": 100,
    "params": {
        "template_spec": {
            "signal": {
                "buy_expr": "CROSS(trend_line, 20)",
                "sell_expr": "trend_line > 89",
            },
            "selection": {
                "filters": ["(rsi14 >= 35) & (rsi14 <= 80)"],
                "score_expr": "trend_line + ema_diff * 10",
                "top_k": 10,
            },
            "risk": {
                "cash_use_ratio": 0.95,
                "max_positions": 10,
                "exit_rules": [
                    {"name": "take_profit", "when": "pnl_pct > 0.15"},
                    {"name": "stop_loss", "when": "pnl_pct < -0.08"},
                ],
            },
        },
    },
}
```
