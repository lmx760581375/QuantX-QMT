# WTS3 / RM5 季度双 Sleeve

## 结构

```text
组合账户 1亿元
├─ WTS Top3：目标 65%
└─ RM v1 Top5 runner：目标 35%
```

两个 sleeve 拥有独立账户、持仓、选股、退出和现金。调度器按同一交易日同步推进。

## 再平衡

- 季度首个交易日前，以前一交易日收盘资产计算目标权重；
- 只转移来源账户的可用现金；
- 不强制卖出持仓；
- 现金不足时记录待划拨金额，后续有现金时继续；
- 划拨不计交易费用；
- 子账户收益基线同步调整，避免把外部划拨误计为策略收益。

配置：`configs/sleeve/quarterly_wts3_reward5_65_35.yaml`

## 运行

```bash
quantx-reward dual-sleeve \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --run-id quarterly_wts3_reward5_65_35 \
  --dry-run

quantx-reward dual-sleeve \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --run-id quarterly_wts3_reward5_65_35
```

## 冻结结果

区间为 2020-01-02 至 2026-09-28：

| 指标 | 数值 |
| --- | ---: |
| 初始资金 | 1亿元 |
| 最终资产 | 11.131亿元 |
| 总收益 | +1013.12% |
| 年化收益 | 42.96% |
| 最大回撤 | -23.32% |
| Sharpe | 1.528 |
| 成交数 | 3430 |

该结果属于真实双账户执行，不是两条已有净值曲线的线性组合。
