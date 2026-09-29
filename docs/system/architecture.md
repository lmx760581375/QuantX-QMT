# 系统边界

| 层 | 职责 | 不负责 |
| --- | --- | --- |
| `models/reward/feature_store` | 同源行情、特征和时间可用性 | 选股与交易 |
| `models/reward/pipeline/h7_absolute.py` | H7 absolute 标签和 full-pair loss | 模型结构 |
| `models/reward/pipeline/h7_absolute_train.py` | 冻结 AE 的 epoch 训练与验证 | 策略调参 |
| `models/reward/pipeline/infer_impl.py` | 加载 checkpoint 并导出 score artifact | 回测 |
| `quantx/core` | 账户、订单、撮合、策略执行 | 模型推理 |
| `quantx/tools/run_dual_sleeve_backtest.py` | 同步两个独立账户并划拨现金 | 重写子策略 |

正式调用链：

```text
bootstrap-data
  -> data-manifest verify
  -> prepare-data
  -> prepare-h7-labels
  -> train
  -> infer historical
  -> prepare-inference-index
  -> infer incremental
  -> merge-scores
  -> backtest / dual-sleeve
```

仅复现已发布模型的交易效果时，可以跳过训练标签和训练步骤，使用
`reproduce-full` 直接构建无标签特征、导出全区间 score 并回测。

`selector.lag: 1` 表示 T 日 score 在下一交易日执行。最新推理索引不读取未来标签，
但仍保留完整特征历史窗口和 T+1 可交易性约束。
