# Reward Model 正式合同

## 模型资产

正式 bundle 为 `weights/reward_v1/`：

- Reward：`reward_epoch009_inference.pt`
- 冻结 AE：`ae_epoch020.pt`
- scaler：`preprocessing/pre2020_eval2020_2026_raw_relative_scaler.json`

权重、分块数量和 SHA-256 由 `MANIFEST.json` 声明。新增行情推理必须复用冻结 scaler。

## 标签

固定合同：

```text
horizon = 7
good = H7 absolute return >= 10%
bad = 同日其他有效样本
```

每个存在 good 和 bad 的交易日形成完整笛卡尔积：

```text
{(good_i, bad_j) | good_i 与 bad_j 属于同一 signal_date}
```

每个 pair 每个 epoch 恰好使用一次。正式 v1 不加入 good-good 排序，避免头部内部排序过拟合。

## 模型与损失

- 冻结 AE 产生 27×128 latent；
- 单头 Reward Transformer：`d_model=768`、7 层、12 heads；
- DPO/Bradley-Terry logistic pair loss；
- `temperature=1.0`；
- batch logit center penalty：`0.01`；
- BF16；
- checkpoint 选择：epoch9。

输出：

```text
reward_score_7d = sigmoid(reward_logit)
```

## 时间边界

- 参数训练：2010–2018；
- validation：2019；
- 2020 年及以后只用于 score 导出与 QuantX 回测；
- 最新 label-free score 信号日截至 2026-09-24，对应 2026-09-28 的 T+1 交易。

## Score artifact

每个 parquet 必须包含：

- `signal_date`
- `instrument`
- `reward_score_7d`

并配套 `quantx_market_score_artifact_v1` JSON manifest。历史与增量 score 以显式截止日合并：

- 截止日及以前保留历史 score；
- 截止日之后采用 label-free 增量 score；
- 禁止重复 `(signal_date, instrument)`。

Score 导出成功不等于策略有效；正式结论还必须经过 YAML dry-run 和完整 QuantX 执行回测。
