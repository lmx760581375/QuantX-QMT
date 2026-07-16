# 实验证据与复现说明

> 建档日期：2026-07-13
>
> 范围：全市场 LightGBM、牛熊/风险模型、组合稳健性和 2010 长历史实验
>
> 原则：文档中的每个关键数字应能追溯到冻结预测、manifest 或标准回测 artifact。

## 1. 当前复现边界

实验配置、临时脚本和大型预测文件按用户要求保留在 `/tmp`，确认有价值后才进入项目。这样避免把大量失败配置和数 GB 预测写入仓库，但有两个后果：

1. `/tmp` 可能被系统清理，所以这里必须记录路径、实验 ID 和 SHA256。
2. 当前研究框架在 `qmt-mac` 工作树中仍有未提交文件；仅凭当前 HEAD `cfc1127` 不能逐位重建全部实验。

因此当前复现等级为：

```text
同一电脑、当前工作树、当前 QMT provider：可复查和复跑
仅克隆 Git 当前提交：尚不能完整复跑
更换时间重新同步 QMT：历史前复权数据可能修订，结果不保证逐位一致
```

文档落盘解决的是“研究过程不丢失”，不是替代数据快照、代码提交和 artifact 仓库。

## 2. 共同数据身份

| 字段 | 值 |
| --- | --- |
| Provider | `/Users/mingxiaoli/Documents/QuantX-QMT/data/qlib_data_fixed` |
| 数据版本 ID | `qmt:4b1c6d77e335b98056f7cf74` |
| adjustment mode | `front_ratio` |
| revision policy | `revised_adjusted_history` |
| calendar hash | `sha256:5c2b9c714b944cbabdee5b6f2e7997e5d9d5d34efa4ed2ae6d7dc6a98fd83900` |
| instruments hash | `sha256:fba31aad966f1b7a72fa6d848c8c4332d42fc6ca6c2dd9382979f9bb249512ce` |
| universe version | `universe:fba31aad966f1b7a` |
| feature schema | `sha256:612e57b49e10f760b1b419d86c58b3647f1e5295020dbe4867389da972c84e63` |
| 股票数 | 3,194 |

`physical_snapshot_id` 当前为空。数据版本 ID 是逻辑身份，不等于完整物理数据已冻结。

退市审计结果随研究起点不同：2016 基线报告识别为 0，2010-2026 扩展数据识别到 5 个带退市边界的代码。这不代表退市历史已经完整，只说明更长区间能识别少量退市记录；ST、停牌和历史成分仍未声明。

## 3. 代码入口

| 责任 | 代码 |
| --- | --- |
| 构建研究数据集 | `quantx/core/research/dataset.py` |
| 年度 anchored walk-forward | `quantx/core/research/split.py` |
| LightGBM/Ridge/Torch trainer | `quantx/core/research/trainer.py` |
| fold 训练和 OOS 预测 | `quantx/core/research/runner.py` |
| 模型和预测 artifact | `quantx/core/research/artifacts.py` |
| 模型指标 | `quantx/core/research/evaluation.py` |
| 研究 CLI | `quantx/tools/run_research.py` |
| 冻结预测契约 | `quantx/core/decision/predictions.py` |
| 决策时钟 | `quantx/core/decision/clock.py` |
| 标准订单规划 | `quantx/core/decision/order_planner.py` |
| 标准回测引擎 | `quantx/core/engine` |

重要实现事实：

1. `AnchoredWalkForwardSplitter` 按 OOS 年度构建独立 fold。
2. `label_end_session` 必须不跨越训练/验证/OOS 边界。
3. prediction contract 记录 `prediction_start`、`prediction_end` 和 `training_information_end`。
4. 冻结 OOS 预测通过 `PredictionStore` 加载，checksum 不匹配时拒绝运行。
5. 组合在 T 日收盘后形成目标，最早执行时间是 T+1 开盘。

## 4. 原 2016 起点全市场基线

### 4.1 三 seed 完整预测

| Seed | Artifact 目录 | 完整预测 SHA256 |
| ---: | --- | --- |
| 7 | `/tmp/quantx-research/mainboard-lgbm-full-market/mainboard_lgbm_relative_10d_full_market-8f97da578f5d9364` | `sha256:394975a91d3a72dc49b4df6ee1c04f7ed2ecf0dd1b74497b26667b7afe3ba0a0` |
| 11 | `/tmp/quantx-research/mainboard-lgbm-full-market/mainboard_lgbm_relative_10d_full_market-5700942ecf0f690c` | `sha256:bf80dc46a193f9165034bb23bdfcb8ccc273884e504332e96152f0634635b2f6` |
| 19 | `/tmp/quantx-research/mainboard-lgbm-full-market/mainboard_lgbm_relative_10d_full_market-df75d591c586cb4a` | `sha256:844f6fe1b134d0506dc033ae3f5ef375a16d6dfeb551cdfe1dba0550b7c1a60a` |

每份完整预测有 `3,716,399` 条记录。

### 4.2 三 seed Top20 集成

| 项目 | 值 |
| --- | --- |
| Artifact ID | `mainboard_lgbm_full_market_ensemble_seeds7_11_19` |
| Manifest | `/tmp/quantx-research/mainboard-lgbm-full-market/ensemble-seeds7-11-19-manifest.json` |
| Top20 文件 | `/tmp/quantx-research/mainboard-lgbm-full-market/ensemble-seeds7-11-19-top20.json` |
| Top20 SHA256 | `sha256:dbd66f2c2c79f0f0c8743c0a26031a9d4ff14b5dcebe9fdb8f8ab502a11ed2a6` |
| 交易日数 | 1,212 |
| 记录数 | 24,240 |

集成过程是先对三份完整分数按 `(signal_time, instrument, fold, schema, horizon)` 对齐，再平均并按日取 Top20。不能合并三份已截断 Top20。

### 4.3 标准回测证据

原主报告使用的运行产物位于：

```text
/tmp/quantx-combined/full-market-ensemble/
/tmp/quantx-research/portfolio-robustness-study/
```

每个标准 run 至少包含：

```text
metrics.json
summary.json
daily_nav.json
trades.json
positions.json
selection_candidates.json
daily_selection_candidates.json
closed_positions.json
explain.json
logs.txt
```

报告数字以 `metrics.json` 和 `daily_nav.json` 为准，不从控制台日志手抄。

## 5. 2010 长历史实验

### 5.1 开发期三 seed 模型

| Seed | Experiment ID | 完整预测 SHA256 | 日均 RankIC |
| ---: | --- | --- | ---: |
| 7 | `mainboard_lgbm_relative_10d_extended_2010_2021_2025_seed7-68c3a7f8ad015614` | `sha256:579ce8788dfb126cc9114dfe3b03578834cb8092e7058baf5e74e4055744cf40` | 0.12575568 |
| 11 | `mainboard_lgbm_relative_10d_extended_2010_2021_2025_seeds_11_19-90dc59340c013b02` | `sha256:0bf735db6818a16f9b5c6d6e555ff89ad205925dee718d414504d437c68e421e` | 0.12626236 |
| 19 | `mainboard_lgbm_relative_10d_extended_2010_2021_2025_seeds_11_19-2603298f6dc92ea9` | `sha256:7334374171ebcb95c15ea2dfc83b0be6a901772c7763b7f4aade7fa7e57fe550` | 0.12569064 |

目录：

```text
/tmp/quantx-research/extended-history-v1/development-seed7/
/tmp/quantx-research/extended-history-v1/development-seeds11-19/
```

seed 7 Top20：

| 项目 | 值 |
| --- | --- |
| 文件 | `/tmp/quantx-research/extended-history-v1/development-seed7/oos_predictions_top20.json` |
| SHA256 | `sha256:59392ed142671d6983856674ab4c4acde529abdf545c26d22cd63fc22ba05f5d` |

三 seed 集成：

| 项目 | 值 |
| --- | --- |
| Artifact ID | `mainboard_lgbm_extended_2010_ensemble_seeds7_11_19` |
| Manifest | `/tmp/quantx-research/extended-history-v1/ensemble-seeds7-11-19-manifest.json` |
| Manifest SHA256 | `sha256:a63898a22b6561584490c63d7dffe7b0639c38aec1a2da2b35a3c4f5ac444d76` |
| Top20 | `/tmp/quantx-research/extended-history-v1/ensemble-seeds7-11-19-top20.json` |
| Top20 SHA256 | `sha256:0ebbfd1a0c8a1c83e0e886ab8a075a51ca19f8dbfadfcc8fd2bb4a158e2ec0d3` |
| 交易日/记录数 | 1,212 / 24,240 |

### 5.2 seed 7 回测

| 口径 | 目录 |
| --- | --- |
| 标准成本 | `/tmp/quantx-research/extended-history-v1/backtests/seed7-top20/extended-2010-seed7-top20-standard` |
| 双倍成本 | `/tmp/quantx-research/extended-history-v1/backtests/seed7-top20-cost2x/extended-2010-seed7-top20-cost2x` |
| 2016 对照 | `/tmp/quantx-research/extended-history-v1/backtests/baseline-seed7-top20/baseline-2016-seed7-top20-standard` |
| 2010/2016 双窗口 | `/tmp/quantx-research/extended-history-v1/backtests/dual-window-seed7-top20/dual-window-seed7-top20-standard` |

三 seed 集成回测：

| 口径 | 目录 |
| --- | --- |
| 标准成本 | `/tmp/quantx-research/extended-history-v1/backtests/ensemble-top20/extended-2010-ensemble-seeds7-11-19-top20-standard` |
| 双倍成本 | `/tmp/quantx-research/extended-history-v1/backtests/ensemble-top20-cost2x/extended-2010-ensemble-seeds7-11-19-top20-cost2x` |

### 5.3 2026 三 seed

| 项目 | 值 |
| --- | --- |
| Experiment ID | `mainboard_lgbm_relative_10d_extended_2010_2026_2026_seed7-b4aba557aec7c0da` |
| 完整预测 | `/tmp/quantx-research/extended-history-v1/forward-2026-seed7/mainboard_lgbm_relative_10d_extended_2010_2026_2026_seed7-b4aba557aec7c0da/oos_predictions.json` |
| 完整预测 SHA256 | `sha256:286a759632d62f77493956fa44e57b70af4293871cfa0d778fde220923dc3932` |
| Top20 | `/tmp/quantx-research/extended-history-v1/forward-2026-seed7/oos_predictions_top20.json` |
| Top20 SHA256 | `sha256:609e794a344cf251468f50f57cb905edc81eea878070536d23b49a6c4554703f` |
| 回测 | `/tmp/quantx-research/extended-history-v1/backtests/forward-2026-seed7-top20/extended-2010-seed7-forward-2026-top20` |

seed 11/19：

| Seed | Experiment ID | 完整预测 SHA256 | 日均 RankIC |
| ---: | --- | --- | ---: |
| 11 | `mainboard_lgbm_relative_10d_extended_2010_2026_2026_seeds_11_19-bb7d224a0cc16e5a` | `sha256:3d11f3fb2a5d3b1c6b41325e6096b73409fcab9735b599491c93cfdb76e318d7` | 0.04303667 |
| 19 | `mainboard_lgbm_relative_10d_extended_2010_2026_2026_seeds_11_19-3352490c2ca64311` | `sha256:67f28730bf0099cdf88755dc7b12d51ae367fca7a706075c87a370ab5b999fd8` | 0.04496766 |

2026 三 seed 集成：

| 项目 | 值 |
| --- | --- |
| Artifact ID | `mainboard_lgbm_extended_2010_forward_2026_ensemble_seeds7_11_19` |
| Manifest | `/tmp/quantx-research/extended-history-v1/forward-2026-ensemble-seeds7-11-19-manifest.json` |
| Manifest SHA256 | `sha256:28cddf9c060ba22cef6fba4d7c82d785e579db9d0c9e0b48a4ace4149668aee1` |
| Top20 | `/tmp/quantx-research/extended-history-v1/forward-2026-ensemble-seeds7-11-19-top20.json` |
| Top20 SHA256 | `sha256:a7bacf3abe6bc947aa11f77c31f2bb9f47ef920e4846d7949fbc3dac18e38f46` |
| 交易日/记录数 | 124 / 2,480 |

2026 回测：

| 口径 | 目录 |
| --- | --- |
| 标准成本组合 | `/tmp/quantx-research/extended-history-v1/backtests/forward-2026-ensemble-top20/extended-2010-forward-2026-ensemble-seeds7-11-19-top20-standard` |
| 双倍成本组合 | `/tmp/quantx-research/extended-history-v1/backtests/forward-2026-ensemble-top20-cost2x/extended-2010-forward-2026-ensemble-seeds7-11-19-top20-cost2x` |
| ML 80% 单独 | `/tmp/quantx-research/extended-history-v1/backtests/forward-2026-ensemble-top20-ml80-only/extended-2010-forward-2026-ensemble-seeds7-11-19-top20-ml80-only` |

## 6. 多周期实验

### 6.1 模型 artifact

| Horizon | Experiment ID | 完整预测 SHA256 |
| ---: | --- | --- |
| 5 | `mainboard_lgbm_relative_5d_full_market-54d06ec8064d08a7` | `sha256:9da3fc2f52c21f96cab4794fe76cc0057385b45921be7c55820d1101e6d70b20` |
| 10 | `mainboard_lgbm_relative_10d_full_market-8f97da578f5d9364` | `sha256:394975a91d3a72dc49b4df6ee1c04f7ed2ecf0dd1b74497b26667b7afe3ba0a0` |
| 20 | `mainboard_lgbm_relative_20d_full_market-cc5c4f6fe59efb6c` | `sha256:8b67f1743b5b80a6c87141be8e1db2f97a82e4fc0a96ee12b63d2496de185e53` |
| 60 | `mainboard_lgbm_relative_60d_full_market-41de9703d60ea4b9` | `sha256:4d8bcd499e4af5cdadc760a54175a624bfedc72cac7876ce017110fd3813c39d` |

### 6.2 集成 manifest

| 集成 | Manifest | Top20 SHA256 |
| --- | --- | --- |
| 10+20 | `/tmp/quantx-research/multihorizon-alpha-v1/candidates/rank-ensemble-10-20-seed7-manifest.json` | `sha256:d18608b33dc0c52ecfd63ee56ef14c3456c31c7f3efb609afe708f7b086c53b6` |
| 5+10+20+60 | `/tmp/quantx-research/multihorizon-alpha-v1/candidates/rank-ensemble-5-10-20-60-seed7-manifest.json` | `sha256:61dd341ca8311edde39d7b089a7acff21684e32aa13340516dcb6d7acadf8a05` |

回测目录：

```text
/tmp/quantx-research/multihorizon-alpha-v1/backtests/
```

## 7. 已否决实验的证据路径

| 实验 | 路径 |
| --- | --- |
| 三状态专家和软融合 | `/tmp/quantx-research/regime-experts-v1/` |
| 全局市场上下文 Alpha | `/tmp/quantx-research/conditional-context-alpha-v1/seed7/` |
| 学习型市场风险 gate | `/tmp/quantx-research/market-risk-gate-v1/seed7/` |
| 五福静态权重 | `/tmp/quantx-research/static-allocation-study/` |
| Top20 和 2026 归因 | `/tmp/quantx-research/portfolio-robustness-study/` |
| 严格趋势 gate | `/tmp/quantx-research/risk-filter-study/` |
| 五福 MA20 | `/tmp/quantx-research/wufu-filter-study/` |
| 多周期 Alpha | `/tmp/quantx-research/multihorizon-alpha-v1/` |
| 2010/2016 双窗口 | `/tmp/quantx-research/extended-history-v1/dual-window-seed7-manifest.json` |

否决表示当前假设和实现没有通过，不表示目录可以立即删除。至少在结论进入 Git 并完成必要 artifact 归档前，应保留这些路径。

## 8. 临时研究脚本

| 脚本 | 用途 |
| --- | --- |
| `/tmp/run_extended_history_seed7.py` | 在不把实验配置写入项目的前提下，把训练起点改为 2010 |
| `/tmp/ensemble_full_market_predictions.py` | 校验多个完整预测 SHA256、逐条对齐平均、按日取 TopK |
| `/tmp/compact_full_market_predictions.py` | 从完整预测中按日保留 TopK |
| `/tmp/ensemble_multihorizon_ranks.py` | 在完整横截面上计算不同 horizon 百分位并固定权重融合 |
| `/tmp/run_quantx_combined_frozen.py` | 用统一账户运行冻结 ML + 五福组合，并强制 T+1 OPEN |

这些脚本是研究工具，不是正式产品入口。被接受的能力应迁移到 `quantx.tools` 并增加测试，失败实验脚本不需要进入项目。

## 9. 复现顺序

### 9.1 训练单个或多个 seed

配置通过正式研究入口运行：

```bash
python -m quantx.tools.run_research --config /tmp/experiment-config.yaml
```

长历史临时 launcher 的命令形状：

```bash
python /tmp/run_extended_history_seed7.py \
  --base-config /tmp/quantx-alpha-full-mainboard-v1.yaml \
  --output /tmp/quantx-research/extended-history-v1/development-seed7 \
  --end 2025-12-31 \
  --first-year 2021 \
  --last-year 2025 \
  --seeds 7
```

运行结束后必须核对：

1. `data_version.version_id`。
2. `feature_schema_hash`。
3. `fold_count`。
4. 每个 fold 的 `training_information_cutoff`。
5. `prediction_count` 和 `prediction_checksum`。
6. 按年度 RankIC 和 coverage。

### 9.2 完整分数集成

```bash
python /tmp/ensemble_full_market_predictions.py \
  --source /path/seed7/oos_predictions.json \
  --checksum sha256:... \
  --source /path/seed11/oos_predictions.json \
  --checksum sha256:... \
  --source /path/seed19/oos_predictions.json \
  --checksum sha256:... \
  --output /tmp/ensemble-top20.json \
  --manifest /tmp/ensemble-manifest.json \
  --artifact-id frozen-ensemble-id \
  --top-k 20
```

脚本使用 `zip(..., strict=True)` 并比较每条记录身份。任一 seed 缺行、顺序不同、fold 不同或 checksum 不匹配都会失败。

### 9.3 统一回测

```bash
python /tmp/run_quantx_combined_frozen.py \
  --prediction /tmp/ensemble-top20.json \
  --checksum sha256:... \
  --artifact-id frozen-ensemble-id \
  --universe-file data/qlib_data_fixed/instruments/all.txt \
  --start 2021-01-04 \
  --end 2025-12-31 \
  --run-id experiment-standard \
  --output /tmp/quantx-research/experiment/backtests \
  --max-stock-positions 20 \
  --cost-multiplier 1
```

成本压力只改变 `--cost-multiplier 2`，不能同时改其他参数。

## 10. 结果计算口径

### 10.1 年度收益

年度收益从 `daily_nav.json` 的 `daily_return` 在自然年内复利：

```text
year_return = product(1 + daily_return) - 1
```

不能简单用该年最后一天累计收益减第一天累计收益，也不能把不完整 2026 区间直接外推成年收益。

### 10.2 月度正收益比例

先在每个自然月内复利日收益，再统计月收益大于 0 的月份比例。2021-2025 有 60 个自然月；2026 前向只有 7 个部分或完整月份，解释时要注明样本很少。

### 10.3 股票持仓数量

`metrics.avg_position_count` 包含股票、五福 ETF 和 `SH511880`，不能直接证明平均股票数大于 5。股票持仓必须按主板代码前缀单独统计。

### 10.4 ML 收益归因

组合收益不能直接按 80% 和 20% 线性拆分，因为交易时点、现金、成本和净值路径会相互作用。正确方法是在同一引擎中分别运行：

1. `ml_allocation=0.8, wufu_allocation=0`。
2. `ml_allocation=0, wufu_allocation=0.2`。
3. `ml_allocation=0.8, wufu_allocation=0.2`。

三者用于解释贡献，不要求收益简单相加完全相等。

## 11. T+1 审计

每个正式研究回测至少检查：

1. selection 的 `signal_date < execution_date`。
2. 每笔订单的 decision time 为 T 日 `AFTER_CLOSE`。
3. earliest execution time 为 T+1 `OPEN`。
4. 实际交易日不早于 earliest execution time。
5. 不能出现信号日和成交日相同的订单。

已有 ML 统一回测报告的 T+1 违规为 0。后续如果改变为季度/月度模型重训，模型训练完成时点也必须写入 contract，不能只审计交易订单。

## 12. 晋级门槛

长历史候选进入正式策略库前，至少需要同时满足：

| 维度 | 门槛方向 |
| --- | --- |
| 模型随机性 | seed 7/11/19 指标和组合结果一致 |
| 跨年度 | 不能只由一两个年度贡献大部分收益 |
| 最差年度 | 正常成本和双倍成本都可接受 |
| 滚动窗口 | 最差滚动 12 个月可接受 |
| 回撤 | 不因收益提高而显著放大 |
| 成本 | 双倍成本后 Alpha 不消失 |
| 分散 | 平均股票持仓大于 5 |
| 前向 | 2026 ML 袖套本身保持正贡献 |
| 时序 | T+1 和训练信息边界 0 违规 |
| 数据 | 清楚披露退市/ST/停牌/历史成分缺口 |

当前 2010 长历史已经完成三 seed、标准/双倍成本、逐年、滚动 252 日、持仓数、T+1 和 2026 前向审计。它通过开发期相关门槛，但 2026 ML 80% 袖套为 `-0.25%`，明确未通过前向 Alpha 门槛，因此不能标记为 `promoted`。

## 13. 文档更新协议

每次实验完成后：

1. 先核对 `metrics.json`、`daily_nav.json`、manifest 和 SHA256。
2. 在 [02_ROBUSTNESS_AND_REGIME_STUDY.md](02_ROBUSTNESS_AND_REGIME_STUDY.md) 更新方法、结果和判定。
3. 在本文件补充 artifact ID、路径和 checksum。
4. 在 [README.md](README.md) 更新实验状态总表。
5. 只有全部晋级门槛通过后，才把配置和策略实现移入正式目录。
