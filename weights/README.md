# 权重目录

这里只保留正式 inference 所需的冻结资产。远端 Git 服务未启用 LFS，因此两份权重以不超过 48MiB 的普通
Git 分块保存于 `reward_v1/chunks/`：

- `reward_v1/reward_epoch009_inference.pt`：H7 absolute full-pair epoch 9 Reward Transformer，仅含模型状态；
- `reward_v1/ae_epoch020.pt`：冻结 AE。
- `reward_v1/preprocessing/pre2020_eval2020_2026_raw_relative_scaler.json`：冻结 FeatureScaler。

首次使用时执行：

```bash
quantx-reward weights restore
```

`quantx-reward train` 和 `quantx-reward infer` 也会在权重缺失时自动恢复，并验证 scaler。

新数据 inference 必须继续使用这份 scaler；不要因为新增数据而重拟合、覆盖或更新它。重拟合 scaler 会改变输入坐标系，
使冻结 AE/Reward checkpoint 的分数不再是严格可复现的输出。

`reward_h2_event_v1/` 保存 D+2 event-good/bad 模型：

- `reward_epoch018_inference.pt`：2019 validation 选出的 epoch18；
- 复用 `reward_v1/ae_epoch020.pt`，不重复保存 AE；
- 使用独立的 H2 scaler，不能与 Reward H7 scaler 混用。

```bash
quantx-reward weights restore --bundle reward_h2_event_v1
```

训练过程中的其他 epoch、optimizer、scheduler、DDP shard 和 TensorBoard 文件不进入仓库。
文件哈希见各权重目录的 `MANIFEST.json`。
