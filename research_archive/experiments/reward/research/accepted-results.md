# 已接受的结果与证据边界

## H7 absolute full-pair Reward

已接受：

- H7 收益不低于 10% 的样本作为 absolute good；
- 每个 epoch 使用同日全部 good × bad；
- epoch9 为正式 checkpoint；
- 冻结 AE 和 FeatureScaler；
- 输出单列 `reward_score_7d`。

未接受：

- good-good 排序增强；
- 仅凭训练 loss 或离线 RankIC 选择 checkpoint；
- 将 score artifact 导出等同于可交易收益。

## RM runner

RM Top5 在 10% 盈利时卖出 85%，保留 15% runner；runner 从峰值回撤 4% 或 H7
退出，并且不占核心 Top5 槽位。

截至 2026-06-02：

- 总收益 `+633.91%`
- 最大回撤 `-46.11%`
- Sharpe `0.930`

## WTS 与 RM 的互补

- RM runner 与 WTS Top3 日收益相关性约 `0.205`；
- 两者主要回撤区间错位；
- 独立账户优于共享有限仓位。

## 正式双 sleeve

WTS 65% / RM 35%，季度现金再平衡，截至 2026-09-28：

- 总收益 `+1013.12%`
- 年化收益 `42.96%`
- 最大回撤 `-23.32%`
- Sharpe `1.528`
- 成交 `3430`

该结果来自真实订单、费用、滑点、交易限制和子账户现金划拨，不是净值线性合成。

## Score 延长

旧 score 实际只覆盖到 2026-06-02。最新 score 使用 label-free inference index
补到 2026-09-24，并从 2026-06-03 起拼接。重叠日期平均 Spearman 为
`0.999873`，每日 Top5 重叠均为 `5/5`。
