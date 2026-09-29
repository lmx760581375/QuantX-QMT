# 系统架构

```text
同源行情
  └─ quantx.core.data
       ├─ BaoStock 下载与增量更新
       ├─ 前复权 CSV
       └─ Qlib provider
            └─ models.reward.feature_store
       ├─ 历史特征与未来路径
       └─ 最新 label-free 推理特征

H7 absolute 标签
  └─ models.reward.pipeline.h7_absolute

冻结 AE + H7 full-pair Reward
  └─ models.reward.train_h7_absolute

标准 score artifact
  └─ models.reward.infer
       └─ signal_date + instrument + reward_score_7d

QuantX
  ├─ RM Top5 runner
  ├─ WTS Top3
  └─ 真实季度 65/35 双 sleeve
```

边界原则：

- 模型只生成分数，不包含交易规则；
- 策略只消费标准 score artifact，不加载神经网络；
- 双 sleeve 只调度独立回测账户，不复制撮合和账户实现；
- 大型数据、score、run 和训练中间产物不进入 Git。

更详细的模型与 score 约束见 [Reward Model 正式合同](system/reward-model-contract.md)。
