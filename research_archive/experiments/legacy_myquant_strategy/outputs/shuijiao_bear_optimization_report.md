# Shuijiao Bear 优化结果

区间: 2023-01-01 ~ 2025-03-03，股票池: top10

| 策略 | 总收益 | 年化 | 最大回撤 | Sharpe |
|---|---:|---:|---:|---:|
| MA Cross | 55.51% | 22.63% | -17.33% | 1.025 |
| 原 Shuijiao Best | -7.51% | -3.54% | -28.27% | -0.188 |
| 新 Shuijiao Bear Best | 22.77% | 9.94% | -7.64% | 1.253 |

- best variant: r2_seed01_03 (round2)
- best module: backtrade.strategy.shuijiao_best_bear_auto

## 关键参数

```json
{
  "bbi_floor_ratio": 0.99,
  "cash_use_ratio": 0.95,
  "enable_bbi_filter": true,
  "enable_dynamic_positions": true,
  "enable_ema_score": true,
  "enable_hard_risk_off": false,
  "enable_kdj_filter": true,
  "enable_kdj_score": false,
  "enable_market_health_entry": false,
  "enable_momentum_sell": true,
  "enable_relax_entry_filters": false,
  "enable_rsi_filter": true,
  "enable_rsi_score": true,
  "enable_trend_guard": false,
  "hard_risk_off_dd": -0.24,
  "kdj_bias": 8.0,
  "kdj_j_max": 90.0,
  "log_trades": false,
  "market_health_entry": 0.18,
  "market_health_very_weak": 0.16,
  "market_health_weak": 0.16,
  "max_loss_days": 45,
  "max_loss_pct": -0.07,
  "max_positions_base": 30,
  "max_positions_very_weak": 6,
  "max_positions_weak": 12,
  "max_profit_pct": 0.18,
  "min_positions_floor": 4,
  "relax_rsi_max": 94.0,
  "relax_rsi_min": 24.0,
  "rsi_max": 78.0,
  "rsi_min": 35.0,
  "score_ema_weight": 30.0,
  "score_kdj_center": 60.0,
  "score_kdj_penalty": 0.18,
  "score_rsi_center": 58.0,
  "score_rsi_penalty": 0.3,
  "score_trend_weight": 1.0,
  "trailing_activate_profit": 0.02,
  "trailing_stop_pct": 0.08,
  "trend_guard_min_ema_spread": 0.0,
  "trend_guard_mode": "trend_line"
}
```
