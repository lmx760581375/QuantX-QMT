# QuantX Reward

这是 H7 Reward Model、弱转强策略和真实双 sleeve 回测的正式可复现仓库。

当前正式基线（截至 2026-09-28）：

- RM v1：H7 绝对收益 good/bad full-pair，epoch 9；
- RM 策略：Top5，盈利 10% 时卖出 85%，15% runner 回撤 4% 或 H7 退出；
- WTS 策略：主板 Top3；
- 组合：WTS 65% / RM 35%，季度现金再平衡；
- 真实双账户回测：总收益 `+1013.12%`、最大回撤 `-23.32%`、Sharpe `1.528`。

Reward H15、多任务 epoch10、静态 50/50 sleeve 和 H2 Event 仍保留为历史基线或独立研究分支，
但不再代表当前 H7 正式组合。

## 环境

公司开发机建议使用：

```bash
conda env create -f environment.yml
conda activate quantx-reward
quantx-reward weights restore
quantx-reward environment
```

如果机器上已有公司的 `test` 环境，也可以直接执行：

```bash
pip install -e '.[dev,repro]' -i https://pypi.hobot.cc/simple
```

大型数据、训练中间 checkpoint、DDP shard、run 和 score parquet 不进入 Git。正式推理权重使用普通
Git 分块保存，由 manifest 和 SHA-256 校验。

## 数据目录

从空目录构建 2026-09-28 冻结数据：

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
```

默认使用仓库内冻结的股票列表，通过 BaoStock 拉取前复权日线并同时生成原始 CSV、
Qlib provider、security master、历史 ST 快照和数据 manifest。

生成的数据结构：

```text
<data-root>/
├── qlib_data_fixed/
├── raw/baostock/stocks/
└── meta/snapshots/
    ├── security_master.csv
    └── historical_st_daily.parquet
```

严格核对是否与正式回测数据一致：

```bash
quantx-reward data-manifest verify --data-root /path/to/data
```

增量更新：

```bash
quantx-reward update-data \
  --data-root /path/to/data \
  --end 2026-09-28
```

## 正式权重

```bash
quantx-reward weights restore
quantx-reward weights verify
```

`weights/reward_v1/` 包含：

- H7 absolute full-pair epoch9 Reward checkpoint；
- 冻结 AE epoch20；
- 冻结 FeatureScaler。

## 训练

```bash
# 构建基础特征和未来路径数据
quantx-reward prepare-data --data-root /path/to/data

# 构建 H7 return >= 10% 的 good/bad 标签
quantx-reward prepare-h7-labels

# 查看并执行正式训练命令
quantx-reward train --dry-run
quantx-reward train
```

训练目标只使用同一 `signal_date` 内所有 good × bad 配对，不使用 good-good 排序。
原正式训练为 4 节点 × 8 GPU；每个节点可使用：

```bash
quantx-reward train \
  --nnodes 4 \
  --node-rank <0..3> \
  --rdzv-endpoint <master-host:port> \
  --rdzv-id reward-h7 \
  --nproc-per-node 8
```

## Score 导出

历史区间：

```bash
quantx-reward infer --config configs/model/reward_v1_infer.yaml
```

新增行情的 label-free 推理：

```bash
quantx-reward prepare-inference-index \
  --start 2026-05-25 \
  --end 2026-09-24

quantx-reward infer --config configs/model/reward_v1_latest_infer.yaml
```

合并历史与增量 score：

```bash
quantx-reward merge-scores \
  --historical artifacts/scores/reward_v1_historical.parquet \
  --incremental artifacts/scores/reward_v1_incremental.parquet \
  --output artifacts/scores/reward_v1.parquet \
  --cutoff-date 2026-06-02
```

合并时保留截止日及以前的历史 score，仅采用截止日之后的增量 score，并重新生成
`quantx_market_score_artifact_v1` manifest。

## 回测

单独验证 RM runner：

```bash
quantx-reward backtest \
  --strategy reward-h7-runner \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --dry-run
```

正式季度双 sleeve：

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

两个 sleeve 使用独立账户和原始交易规则。季度再平衡仅划拨可用现金，不强制卖出现有持仓；
现金不足的划拨会延后执行，并禁止负现金。

## 测试

```bash
pytest -q
quantx-reward verify
quantx-reward verify \
  --runs-root runs \
  --dual-run quarterly_wts3_reward5_65_35
```

## 从零复现

以下命令串联数据下载、Qlib 构建、数据摘要校验、历史训练口径 score、
末端 label-free score、合并和正式双 sleeve 回测：

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

默认启用严格数据和指标校验。如果 BaoStock 后续修订了历史前复权行情，数据 manifest
会明确失败；此时结果不能标记为对冻结基线的精确复现。

中断后可使用 `--skip-features`、`--skip-dataset`、
`--skip-historical-inference`、`--skip-inference-index`、
`--skip-incremental-inference` 或 `--skip-merge` 从已有阶段继续。

若只是研究新日期而不是复现冻结结果，可显式使用
`--no-strict-data --no-strict-score --no-strict-metrics`，但输出会被视为新的数据版本。
全量下载中断后可加 `--resume-data`，避免重新下载已经完整覆盖目标区间的股票。

详细说明见：

- [研究文档索引](docs/README.md)
- [Reward 模型合同](docs/system/reward-model-contract.md)
- [数据合同](docs/system/data-contract.md)
- [数据准备与复现](docs/data_and_reproduction.md)
- [最终策略决策](docs/decisions/2026-09-final-strategy.md)
- [季度双 sleeve](docs/strategies/quarterly-dual-sleeve.md)

## 研究归档

[`research_archive/`](research_archive/) 集中保存：

- QuantX 相关 Codex 用户/Codex 对话的脱敏 Markdown 副本；
- QuantX、QMT、Reward Model 与早期 myquant 系列实验文档；
- 原始来源路径、SHA-256、重复文件映射和脱敏统计。

归档脚本只读 Codex 原始会话，不会移动、修改或删除 `~/.codex` 中的任何文件：

```bash
python tools/archive_quant_research.py --output /path/to/new/archive
```
