# 数据准备与复现

## 环境

```bash
conda activate test
pip install -e '.[dev,repro]' -i https://<INTERNAL_HOST>/simple
quantx-reward weights restore
quantx-reward environment
```

## 从空目录准备日线

```bash
quantx-reward bootstrap-data \
  --data-root /path/to/data \
  --start 2010-01-01 \
  --end 2026-09-28 \
  --dry-run

quantx-reward bootstrap-data \
  --data-root /path/to/data \
  --start 2010-01-01 \
  --end 2026-09-28

quantx-reward data-manifest verify \
  --data-root /path/to/data
```

该流程从 BaoStock 拉取前复权日线，同时生成原始 CSV、Qlib provider、security master、
历史 ST 快照和数据 manifest。完整格式见 [日线数据合同](system/data-contract.md)。

## 训练数据

```bash
quantx-reward prepare-data --data-root /path/to/data
quantx-reward prepare-h7-labels
```

第一条命令生成特征、候选索引和未来路径；第二条命令固定生成：

```text
H7 absolute return >= 10% -> good
其余同日有效样本          -> bad
```

训练：

```bash
quantx-reward train --dry-run
quantx-reward train
```

正式配置为 `configs/model/reward_v1_train.yaml`。训练使用同日完整 good × bad 配对，
冻结 AE 和 scaler。

原训练资源为 4 节点 × 8 GPU。多节点参数通过 `--nnodes`、`--node-rank`、
`--rdzv-endpoint` 和 `--rdzv-id` 提供。

## 历史 score

```bash
quantx-reward infer --config configs/model/reward_v1_infer.yaml
```

输出为 `artifacts/scores/reward_v1_historical.parquet`，并生成同名 JSON manifest。

## 新增行情的 label-free score

先更新特征存储，然后构建不要求未来 H30 标签的索引：

```bash
quantx-reward prepare-inference-index \
  --kronos-root workdirs/feature_store \
  --output-root workdirs/reward/latest_inference \
  --start 2026-05-25 \
  --end 2026-09-24

quantx-reward infer --config configs/model/reward_v1_latest_infer.yaml
```

纯推理索引只使用信号日及之前的特征；其中 2026-09-24 的信号用于 2026-09-28 的 T+1 交易。

## 合并 score

```bash
quantx-reward merge-scores \
  --historical artifacts/scores/reward_v1_historical.parquet \
  --incremental artifacts/scores/reward_v1_incremental.parquet \
  --output artifacts/scores/reward_v1.parquet \
  --cutoff-date 2026-06-02
```

截止日及以前保留历史 score，之后采用增量 score。合并结果会重新校验 schema、主键和覆盖范围。

## 回测

```bash
quantx-reward backtest \
  --strategy reward-h7-runner \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --dry-run

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

正式回测配置为：

- `configs/strategies/reward_h7_runner.yaml`
- `configs/strategies/weak_to_strong_pos3.yaml`
- `configs/sleeve/quarterly_wts3_reward5_65_35.yaml`

## 不进入 Git 的内容

- 原始行情和大型特征 memmap；
- 训练过程 checkpoint、optimizer、scheduler、TensorBoard 和 DDP shard；
- score parquet；
- `runs/` 完整回测目录。

正式权重分块、配置、代码、测试、摘要指标和必要图表可以进入 Git。

## 一键复现正式效果

使用仓库发布的 RM/AE/scaler 权重，不重新训练：

```bash
quantx-reward reproduce-full \
  --data-root /path/to/data \
  --as-of 2026-09-28 \
  --nproc-per-node 8 \
  --run-id quarterly_wts3_reward5_65_35 \
  --dry-run

quantx-reward reproduce-full \
  --data-root /path/to/data \
  --as-of 2026-09-28 \
  --nproc-per-node 8 \
  --run-id quarterly_wts3_reward5_65_35
```

命令依次完成：

1. 数据下载与 Qlib 转换；
2. 数据 manifest 校验；
3. 权重与环境校验；
4. 历史可结算区间的数据集和 score；
5. 末端 label-free 推理索引和增量 score；
6. 以 2026-06-02 为边界合并两段 score；
7. 真实双 sleeve 回测；
8. 冻结指标对比。

使用两段式 score 是必要的：全历史都改用 label-free 候选会改变历史候选集合，
不能视为原正式实验的严格复现。

默认会同时检查数据、最终 score 和回测指标。若数据供应商已经修订历史数据，
命令会在相应阶段停止。研究新数据版本时可以显式关闭严格检查，但不得沿用冻结结果标签。

长流程支持通过以下参数从已有工件恢复：

```text
--resume-data
--skip-features
--skip-dataset
--skip-historical-inference
--skip-inference-index
--skip-incremental-inference
--skip-merge
```
