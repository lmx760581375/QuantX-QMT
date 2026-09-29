# 实验索引与历史归类

完整原始记录位于 [RESEARCH_LOG_2026-07.md](../archive/RESEARCH_LOG_2026-07.md)。本索引帮助定位，不替代原文。

| 时间 / 原始日志段 | 主题 | 当前归类 | 当前结论 |
| --- | --- | --- | --- |
| 2026-07-17，实验循环与 OOS 接入 | Diffusion、AE 重建、弱转强 overlay | 历史基础 | 建立数据、特征和正式回测纪律；不直接进入 v1。 |
| 2026-07-18，Close-token decoder | 5d/7d/10d decoder | 研究对照 | 宽容量 10d 有参考价值；窄头部和跨 regime 稳定性不足。 |
| 2026-07-19，Pairwise / independent Reward | 同日 pairwise 训练 | 历史正向证据 | 证明排序头可形成容量结果，但不是最终多任务合同。 |
| 2026-07-19，5d Reward / checkpoint 40 epoch | epoch 对照 | 方法证据 | 训练更久不保证头部策略更好；禁止后验 checkpoint 选择。 |
| 2026-07-20，3d/4d 对齐 | label-holding 对齐 | 方法修正 | 旧 H5 解释作废；严格对齐成为硬约束。 |
| 2026-07-20，退出、breadth、AMV | 风险控制 | 拒绝 | 个股止损和全清仓不能稳定改善组合。 |
| 2026-07-20 至 07-21，PPO | 动态持仓 | 拒绝部署 | 局部正例不迁移；严格 OOS 未通过。 |
| 2026-07-21，Listwise / raw-feature | 架构替代 | 拒绝 | 可成交 listwise、去 AE raw transformer 均未超过基线。 |
| 2026-07-23，Loop preference | 中间 BT / recurrent score | 诊断 | 有机制信息，但未替代一遍式 Reward。 |
| 2026-07-26，7D multi-task | return Top50 + 风险重排 | 历史正式模型 | epoch10 三头合同曾作为 Reward 来源，现保留为历史对照。 |
| 2026-09，H15 与 sleeve | 旧策略组合 | 历史正式 | H15、弱转强、静态 50/50 sleeve 曾进入 v1。 |
| 2026-09-28，H2 Event Reward | D+2 事件标签、epoch18、入场/退出矩阵 | 候选 | 短周期 Top5 存在正向信息；不追高开超过2%显著改善实现收益，但阈值仍需新增数据复核。 |
| 2026-09-27，弱转强3 + Reward5 | 固定50/50独立 sleeve，仅压缩弱转强持仓 | 正式候选 | 冻结结果 +679.36%；修复增量数据截断后的复跑 +696.90%、回撤 -21.25%。 |
| 2026-09-29，H7 absolute full-pair | 同日全部 good × bad，epoch9 | 正式模型 | 替代多任务 epoch10，输出 `reward_score_7d`。 |
| 2026-09-29，RM runner + WTS3 | 真实季度现金再平衡 | 正式组合 | WTS65/RM35，总收益 +1013.12%，最大回撤 -23.32%，Sharpe 1.528。 |

## 与当前代码的映射

| 当前资产 | 历史来源 |
| --- | --- |
| `models/reward/feature_store/` | 2026-07-17 建立的同源 memmap 特征协议。 |
| `models/reward/pipeline/h7_absolute.py` | H7 absolute 标签、同日 full-pair 数据集和精确 pair loss。 |
| `models/reward/pipeline/h7_absolute_train.py` | 当前正式 H7 full-pair 训练循环。 |
| `weights/reward_v1/` | 2026-09-28 epoch9 精简推理权重、冻结 AE 与 scaler。 |
| `configs/strategies/reward_h7_runner.yaml` | 当前 RM Top5 runner 策略。 |
| `configs/strategies/weak_to_strong.yaml` | 经过容量和退出研究后冻结的规则策略。 |
| `configs/sleeve/static_50_50.yaml` | 2026-09 独立 sleeve 组合决策。 |
| `weights/reward_h2_event_v1/` | H2 event-good/bad epoch18 推理权重及独立 scaler。 |
| `configs/strategies/reward_h2_event.yaml` | 主板加创业板、排除科创板、不追高开超过2%的 H2 候选策略。 |
| `configs/strategies/weak_to_strong_pos3.yaml` | 弱转强信号不变、最大持仓缩减为3只。 |
| `configs/sleeve/quarterly_wts3_reward5_65_35.yaml` | 当前季度现金再平衡组合。 |
