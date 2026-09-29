# 静态双 sleeve：弱转强 3 仓 + Reward Top5（历史基线）

> 本组合已被 WTS65/RM35 季度现金再平衡双 sleeve 替代。

## 组合定义

该组合保留 Reward H15 的每日 Top5 排名和最多 5 个持仓，只将弱转强 sleeve
的最大持仓从 5 只缩减为 3 只：

```text
初始资金
 ├─ 50% -> Reward H15 Top5 / 最多5仓
 └─ 50% -> 弱转强 Top4 / 最多3仓
```

两个子账户独立运行，不再平衡、不共享现金、不互相替换持仓。

## 冻结结果

2026-09-27 使用当时冻结数据得到：

| 指标 | 5+5 baseline | 弱转强3 + Reward5 |
| --- | ---: | ---: |
| 累计收益 | +514.97% | **+679.36%** |
| 最终净值倍数 | 6.15x | **7.79x** |
| 年化收益 | 34.33% | **39.60%** |
| 最大回撤 | -23.46% | **-21.89%** |
| Sharpe | 1.351 | **1.498** |
| 月度收红率 | 62.34% | **63.64%** |

收益改善主要来自弱转强头部仓位集中，而不是改变 Reward 模型输出。Reward
selector 仍然是 Top5。

## 数据修复后的复跑

2026-09-28 修复增量同步截断 115 只股票历史数据的问题，并重建 Qlib 后：

| 指标 | 当前复跑 |
| --- | ---: |
| 累计收益 | **+696.90%** |
| 最终净值倍数 | **7.97x** |
| 年化收益 | 38.17% |
| 最大回撤 | **-21.25%** |
| Sharpe | **1.405** |
| 月度收红率 | 62.82% |

修复前的 `+562.65% / -35.83%` 已作废。根因是增量刷新把完整原始 CSV
覆盖为短窗口，并将 `all.txt` 的历史上市日期同步截断。上游修复提交为
`afa410c`。修复后的结果与冻结 `+679.36% / -21.89%` 处于同一量级。

## 复现

```bash
quantx-reward reproduce \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --wts-strategy weak-to-strong-pos3 \
  --reward-run-id reward_h15_top5 \
  --wts-run-id weak_to_strong_pos3 \
  --figure-dir artifacts/sleeve/wts3_reward5_50_50
```

配置：

- `configs/strategies/reward_h15.yaml`
- `configs/strategies/weak_to_strong_pos3.yaml`
- `configs/sleeve/static_wts3_reward5_50_50.yaml`
