# Shuijiao 最优策略记录（Ablation Run 1）

## 1. 最优版本信息
- 生成时间：2026-03-01
- 来源实验：`outputs/ablation_run1`
- 最优变体：`round2 / r2_seed01_02`
- 自动策略模块：`backtrade.strategy.auto.shuijiao_auto_0016`
- 固化策略文件：`backtrade/strategy/shuijiao_best_auto.py`
- 固化配置文件：`outputs/ablation_run1/test_shuijiao_best_auto.py`

## 2. 回测结果（1159 天）
- 总收益率：`186.47%`
- 年化收益率：`24.59%`
- 最大回撤：`-25.10%`
- Sharpe（日频年化）：`1.278`
- Sortino（日频年化）：`1.836`
- Calmar：`0.980`
- 胜率：`44.02%`
- 交易次数：`2044`
- 交易成本：`56541.08`

## 3. 最优参数
```python
PARAMS = {
    "bbi_floor_ratio": 0.99,
    "cash_use_ratio": 0.97,
    "enable_bbi_filter": False,
    "enable_dynamic_positions": True,
    "enable_ema_score": True,
    "enable_hard_risk_off": False,
    "enable_kdj_filter": False,
    "enable_kdj_score": True,
    "enable_market_health_entry": False,
    "enable_momentum_sell": False,
    "enable_rsi_filter": True,
    "enable_rsi_score": False,
    "hard_risk_off_dd": -0.28,
    "kdj_bias": 2.0,
    "kdj_j_max": 98.0,
    "log_trades": False,
    "market_health_entry": 0.24,
    "market_health_very_weak": 0.14,
    "market_health_weak": 0.22,
    "max_loss_pct": -0.09,
    "max_positions_base": 36,
    "max_positions_very_weak": 10,
    "max_positions_weak": 12,
    "max_profit_pct": 0.18,
    "rsi_max": 82.0,
    "rsi_min": 32.0,
    "score_ema_weight": 10.0,
    "score_kdj_center": 60.0,
    "score_kdj_penalty": 0.12,
    "score_rsi_center": 58.0,
    "score_rsi_penalty": 0.0,
    "score_trend_weight": 1.0,
}
```

## 4. 复现命令
```bash
source ~/anaconda3/bin/activate test
PYTHONPATH=${HOME}/git_projects/myquant-strategy \
python backtrade/engine.py --config outputs/ablation_run1/test_shuijiao_best_auto.py
```

## 5. 关键实验产物
- 全量结果：`outputs/ablation_run1/ablation_all_results.csv`
- 第一轮：`outputs/ablation_run1/ablation_round1.csv`
- 第二轮：`outputs/ablation_run1/ablation_round2.csv`
- 汇总：`outputs/ablation_run1/ablation_summary.json`

## 6. 策略行为特征（简记）
- 保留 `RSI` 过滤（`rsi_min=32`, `rsi_max=82`），关闭 `KDJ/BBI` 硬过滤。
- 保留 `EMA + KDJ` 排序增强，关闭 `RSI` 排序惩罚。
- 关闭动量弱卖出，减少不必要震荡止损。
- 仓位上限 36，弱市仓位 12/10，止盈/止损为 0.18 / -0.09。
