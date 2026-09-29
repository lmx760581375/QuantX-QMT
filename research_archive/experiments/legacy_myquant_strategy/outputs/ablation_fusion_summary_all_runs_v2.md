# Fusion 全量实验汇总（run1~run5）

## 每轮最优结果

| run          |   variants | best_variant   |   score |   annual_return |   total_return |   max_drawdown |   sharpe |   win_rate |   trades_count |
|:-------------|-----------:|:---------------|--------:|----------------:|---------------:|---------------:|---------:|-----------:|---------------:|
| run1         |         18 | random_010     | 1.29053 |         24.5942 |        186.472 |       -25.1035 |  1.27842 |    44.0239 |           2044 |
| run2         |         28 | trend_strict   | 1.29053 |         24.5942 |        186.472 |       -25.1035 |  1.27842 |    44.0239 |           2044 |
| run3         |         36 | random_019     | 1.29053 |         24.5942 |        186.472 |       -25.1035 |  1.27842 |    44.0239 |           2044 |
| run4_shadow  |         28 | random_015     | 1.29053 |         24.5942 |        186.472 |       -25.1035 |  1.27842 |    44.0239 |           2044 |
| run5_replace |         28 | r2_seed01_02   | 1.29053 |         24.5942 |        186.472 |       -25.1035 |  1.27842 |    44.0239 |           2044 |

## 全局最优

- run: run1
- variant: random_010
- score: 1.290532
- annual_return: 24.594205
- total_return: 186.471522
- max_drawdown: -25.103494
- sharpe: 1.278422
- win_rate: 44.023904
- trades_count: 2044

- params_json:
```json
{"enter_confirm_days": 8, "log_switch": false, "ma_cash_use_ratio": 0.95, "ma_enter_breadth": 0.7, "ma_enter_breadth_avg": 0.55, "ma_enter_mom20": 0.08, "ma_enter_score": 0.16, "ma_enter_trend60": 0.12, "ma_long": 20, "ma_short": 10, "ma_top_n": 10, "regime_lookback_days": 180, "vol_punish_threshold": 0.03}
```
