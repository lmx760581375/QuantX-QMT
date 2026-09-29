# 研究文档索引

当前正式版本冻结于 2026-09-29：

- H7 absolute good/bad full-pair Reward Model epoch9；
- RM Top5 TP10 keep15 runner；
- WTS Top3；
- WTS 65% / RM 35% 季度现金再平衡双 sleeve。

建议阅读顺序：

1. [最终策略决策](decisions/2026-09-final-strategy.md)
2. [Reward 模型合同](system/reward-model-contract.md)
3. [数据合同](system/data-contract.md)
4. [数据准备与复现](data_and_reproduction.md)
5. [RM Top5 runner](strategies/reward-h7-runner.md)
6. [季度双 sleeve](strategies/quarterly-dual-sleeve.md)
7. [接受结果](research/accepted-results.md)
8. [拒绝尝试](research/rejected-attempts.md)

`Reward H15 + 静态 50/50 sleeve` 文档保留为历史基线，不再代表当前正式策略。

所有收益结论必须来自完整 QuantX 执行回测；离线 score 指标、净值线性合成和容量估算不能替代真实订单回测。
