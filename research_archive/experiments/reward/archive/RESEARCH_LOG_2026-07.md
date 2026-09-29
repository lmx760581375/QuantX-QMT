# 弱转强 TopN 路径 Diffusion 研究日志

> 建档日期：2026-07-17
>
> 状态：`implementation_ready`
>
> 目标：每天先用现有弱转强规则生成宽候选 TopN，再用 DDP 训练的条件 diffusion/flow 模型预测候选股票未来 30 个交易日收益路径分布，用于候选池内排序、右尾识别和左尾过滤。

## 1. 研究边界

本研究只在 `tmp/weak-to-strong-diffusion-v1/` 内推进，不修改 QuantX 核心框架，不修改已有 Kronos memmap 数据和既有 diffusion 实验脚本。

固定口径：

1. 弱转强候选池来自现有 YAML 的 `selector.where` 和 `selector.score`，不手写第二套策略逻辑。
2. 输入窗口复用 Kronos memmap，动态读取过去 60 session，不物化重复窗口。
3. 信号日为 `T`，执行假设为 `T+1 open`。
4. `y_abs[h] = close(T+h) / open(T+1) - 1`，`h=1..30`。
5. `y_relative = y_abs - y_baseline`，训练默认用 relative target，评估同时报告 absolute。
6. 开发期不得根据单次 `prediction` 或 2026 forward 结果调 score。

## 2. 新增文件

| 文件 | 用途 |
| --- | --- |
| `wts_diffusion_v1.py` | 共享路径、Dataset、scaler、模型、采样和指标工具 |
| `build_wts_diffusion_dataset_v1.py` | 构建弱转强 TopN 候选 index 和 30 日路径 target memmap |
| `train_wts_diffusion_ddp_v1.py` | AE + conditional diffusion/flow DDP 训练 |
| `evaluate_wts_diffusion_v1.py` | 采样预测并输出候选池内 IC / spread 诊断 |

## 3. 预注册 score

首轮只比较以下固定 score：

```text
score_mean_20d
score_mean_minus_0.5std_20d
score_mean_plus_q10_20d
score_prob_pos_20d
score_tail_utility_30d
```

## 2026-07-23 日频数据能否学习短期 alpha：H2/H3 直接标签与净收益验证

### 问题与预注册判定

本节检验的不是“日频 K 线是否能把次日涨跌预测到很高”，而是更可交易的命题：仅使用当前日频
特征，在 `close(T+1)` 执行后，是否能对随后很短的价格区间产生足以覆盖 QuantX 成本的 Top5 横截面
alpha，并把净回测收益提高。

目标路径仍严格为：

```text
target(H) = close(T+H) / close(T+1) - 1
```

所以 H2 有一个实际 close-to-close 区间，H3 有两个。实验前固定以下 gate：

1. 在未参与训练的 `validation_select`（2019-01-02 至 2019-09-10，170 个完整截面）选择 checkpoint；
   要求 H 对应的 Top5 超额和平均 RankIC 都为正。
2. 只有通过该 gate 才能导出 2020-2026 分数并运行候选 QuantX；训练 pair accuracy 不作为晋级指标。
3. 对已验证的 5D Reward Transformer，单独运行 H2/H3 精确时间止损控制回测。这一控制保持股票池、lag、
   close 成交、仓位和成本不变，只移除 trailing/loss exit，隔离短持有期本身。

### 标签本身可构造，但这不是可学习性的证据

绝对 `5%` trend threshold 不适用于 H2/H3，因此使用已有的 `tail_net` pair contract，不修改模型：同一
signal date 的 realized-return Top5 为 chosen，rank 6..200 为 rejected，且 `r_bad < r_good - 0.5%`。
每个 chosen 在一个 epoch 内抽 120 个不同 tail bad；同一个 frozen AE、50M Reward Transformer、优化器和
batch size 与强 5D 基线保持一致。

数据上该 pair contract 非常充足：

| split | H2：Top5均值 - rank200 均值 | H3：Top5均值 - rank200 均值 |
| --- | ---: | ---: |
| train（2010-2018） | `7.75%` | `13.51%` |
| validation_select（2019） | `7.14%` | `15.57%` |
| prediction diagnostic（2020-2026） | `6.83%` | `15.94%` |

H2 训练集有 2,068 个可配对日期、每 epoch 1,240,800 对；H3 为 2,076 个日期、每 epoch 1,245,600 对。
因此失败不能归因于“没有同日可区分标签”或 pair 太少；这些是事后 realized top tail 的巨大截面离散度，
并不保证日频特征能在未来日期识别它。

### 新的主板 Top5-tail 直接短标签实验：选择集失败

工件：

```text
H2 run:
  market-mainboard-pre2020-ae-reward-transformer50m-tailnet-top5-h2-bs1952-7gpu-20ep-20260723-142239/

H3 run:
  market-mainboard-pre2020-ae-reward-transformer50m-tailnet-top5-h3-bs1952-7gpu-20ep-20260723-143044/
```

两次训练均在 GPU 1-7 上执行，GPU 0 未使用。为避免把明显反向的模型训练到完整 20 epoch，H2 在保留
epoch1-5 后、H3 在保留 epoch1-6 后停止；这是由预先固定的 2019 gate 触发，不使用 2020-2026 选择。

| 模型 / checkpoint | 2019 RankIC | 2019 Top5 超额 | 验证 pair accuracy |
| --- | ---: | ---: | ---: |
| H2 epoch1 | `-0.0199` | `-0.431%` | `79.2%` |
| H2 epoch5 | `-0.0216` | `-0.520%` | `77.2%` |
| H3 epoch1 | `-0.0308` | `-0.854%` | `73.2%` |
| H3 epoch5 | `-0.0441` | `-1.602%` | `71.4%` |
| H3 epoch6 | `-0.0415` | `-1.172%` | `71.2%` |

这是核心反证：训练与验证 pair accuracy 可达 71%-79%，但全截面 RankIC 和实际 Top5 超额在独立日期上
均为负，且随 H3 训练加深变差。短期 pairwise preference objective 在同日 Top5-vs-tail 合约上可拟合，
但没有学到跨日期可泛化的短期 alpha；不能将 pair loss、pair accuracy 或巨大 realized rank gap 解释为可交易能力。

### 已验证 5D score 的短期信息与成本控制回测

对 frozen-AE Reward Transformer epoch020 的 2020-2026 score artifact，仍采用固定 cohort、稳定 Top5
tie-break、无摩擦口径：

| 真正目标 | Top5 超额 | RankIC | 毛 alpha density / 实际区间 |
| --- | ---: | ---: | ---: |
| H2（1 区间） | `+0.090%` | `0.0152` | `+0.090%` |
| H3（2 区间） | `+0.237%` | `0.0223` | `+0.119%` |
| H5（4 区间） | `+0.391%` | `0.0307` | `+0.098%` |

这说明日频表征并非完全没有短期相关性，H3 的**毛**密度甚至高于 H5；但当前 QuantX 成本合同约为完整
买卖 `0.31%`（买卖各 `0.05%` commission 和 `0.10%` slippage，卖出另有 `0.01%` stamp tax），已超过
H2/H3 的平均毛超额。下面的正式控制回测验证了这一点。

| 精确控制（Top5，主板，2020-01-02..2026-06-16） | 总收益 | 最大回撤 | Sharpe | trade count | 平均持有天数 | 总成本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2：`holding_days >= 1` | `-96.07%` | `-96.10%` | `-1.166` | 13,628 | `1.51` | `38,548,474` |
| H3：`holding_days >= 2` | `-31.74%` | `-73.07%` | `-0.169` | 7,610 | `3.03` | `63,125,070` |

控制配置与正式 run：

```text
quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_short_horizon_controls_v1/
  reward_transformer_mainboard_epoch020_5d_top5_h2_exact_2020_2026.yaml
  reward_transformer_mainboard_epoch020_5d_top5_h3_exact_2020_2026.yaml

QuantX runs:
  20260723_143019_reward_transformer_mainboard_epoch020_5d_top5_h2_exact_2020_2026
  20260723_143228_reward_transformer_mainboard_epoch020_5d_top5_h3_exact_2020_2026
```

两份 YAML 均已先通过 `--dry-run --json`。它们不是候选模型回测，而是强 5D score 在短持有期是否仍能
净赚钱的成本对照；没有 path-dependent exit，因此不把失败归咎于 trailing 或 loss exit。

### 稀疏置信度筛选：2019 有效、后续市场阶段不迁移

为检查短期弱相关性是否集中在少数高置信度日期，唯一一次预注册式筛选使用现有 5D Reward Transformer
epoch020 score 的日内分离度 `gap_5_20 = score(rank5) - score(rank20)`；不重训模型、不搜索股票数或
持有期。阈值仅由 2019 上半年 validation 的 70 分位固定为 `0.022331535816192627`，随后原样应用于
2019 下半年与 2020-2026。

| 固定阈值下的 H3 固定 cohort | 触发日期 | H3 Top5 毛超额 | 扣 31bp 往返成本后的粗略 alpha |
| --- | ---: | ---: | ---: |
| 2019 上半年（阈值拟合期） | 26 | `+0.713%` | `+0.403%` |
| 2019 下半年（时间留出） | 32 | `+0.557%` | `+0.247%` |

下半年样本的置信区间仍穿过零，因此它只满足进行一次诊断性 QuantX 回放的最低条件，不构成部署证据。筛选后
artifact 有 476 个 signal date、1,354,656 行；manifest 已验证，YAML 亦先通过 dry-run：

```text
score artifact:
  market-mainboard-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-20ep-20260720-143108/
  score_artifacts/reward_epoch020_h3_gap5_20_ge_0p0223315_20200102_20260715.parquet

config:
  quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_short_horizon_controls_v1/
  reward_transformer_mainboard_epoch020_5d_top5_h3_gap5_20_ge_0p0223315_diag_2020_2026.yaml

formal diagnostic run:
  20260723_144738_reward_transformer_mainboard_epoch020_5d_top5_h3_gap5_20_ge_0p0223315_diag_2020_2026
```

该诊断回测总收益为 `-62.57%`、最大回撤 `-67.52%`、Sharpe `-0.602`、3,330 笔交易、总成本
`24,718,000` 左右、平均持有 `2.91` 天；`signal_errors=[]`，另有 38 个正常成交约束导致的拒单。
与未筛选 H3 对照相比，筛选确实降低了换手和总成本，却没有转化为正净收益。

因此，`gap_5_20` 在 2019 的表观预测力是非平稳的，不能迁移到后续市场阶段。这个阈值和同类日频
confidence gate 到此停止，不再在已反复观察的 2020-2026 窗口上做阈值、覆盖率或持有期搜索。

### 既有 all-A H3 试验的边界

此前另一个 `market_all_close2close_balanced_v2`、`all_a` universe 的 `rank_horizon=3` global all-up 模型
曾在 H3 adjacent 配置出现 epoch10 `+28.55%`、epoch15 `+29.72%`，但 Sharpe 仅 `0.129/0.150`、最大回撤
约 `-50.7%/-50.5%`，epoch20 又变为 `-11.10%`。对应 H2-aligned 的 epoch5/10/15/20 全部亏损
(`-59.67%/-43.79%/-52.56%/-54.85%`)。

这些 all-A 历史结果不能证明稳定的短期 alpha：它们使用不同数据根和股票池，checkpoint 对同一 2020-2026
诊断窗口高度不稳定，且该窗口已经被重复观察。它们最多表明偶尔可以得到接近零或弱正的短周期实现，而不是一个
可部署、可复现且优于长期 5D/H30 基线的收益来源。

### 结论与后续约束

1. **当前结论：否。** 在现有日频 OHLCV/派生特征、当前股票池、执行成本与已做过的 H2/H3 监督中，没有证据表明
   日频数据已经学到能提高净 QuantX 回测收益的短期 alpha。直接 H2/H3 Top5-tail reward model 在独立 2019
   截面上反向；强 5D score 虽保留弱短期相关性，但 H2/H3 高换手控制回测显著亏损。
2. **不是“短期标签没有信息”。** realized future returns 的同日 rank gap 很大，且长周期模型 H2/H3 RankIC
   为正；失败位于预测条件信息不足/关系不稳定与成本覆盖不足，而不在 pair 构造或计算量。
3. **最小可交易门槛。** 对 H2/H3，新的候选至少须在完全未使用的 validation 上给出正的 Top5 excess，且
   `Top5 excess > 0.31%`（更严格地还应留 slippage/停牌/涨跌停缓冲），再有资格运行 QuantX。当前 H2/H3
   的 `+0.090%/+0.237%` 毛均值均未达到此条件。
4. **下一条高价值路径不是继续调 reward head。** 若仍要追求更高复利，需要新增能解释 1-2 日微观结构的条件信息，
   例如分钟级成交量/价格路径、盘口与开收盘特征、隔夜/日内拆分、事件与资金流；否则应把当前日频 reward score
   用于它已验证较稳定的较长持有期，并避免把 checkpoint 或持有期继续在 2020-2026 上调优后称为 OOS。

其中：

```text
score_tail_utility_30d = mean_20d + 0.5*q90_30d + q10_30d + 0.5*drawdown_30d
```

`drawdown_30d` 为负数，因此最后一项惩罚左尾路径。

## 4. 最小运行命令

小样本 smoke：

```bash
${HOME}/anaconda3/envs/test/bin/python build_wts_diffusion_dataset_v1.py \
  --start 2022-01-01 \
  --end 2022-06-30 \
  --max-instruments 120 \
  --limit-days 60 \
  --candidate-topn 20 \
  --force

${HOME}/anaconda3/envs/test/bin/python train_wts_diffusion_ddp_v1.py \
  --fold-id fold_2023 \
  --validation-split validation_select \
  --max-train-rows 2000 \
  --max-val-rows 500 \
  --ae-epochs 1 \
  --diffusion-epochs 1 \
  --batch-size 128 \
  --device auto \
  --ddp off
```

正式 DDP 示例：

```bash
torchrun --nproc_per_node=2 train_wts_diffusion_ddp_v1.py \
  --fold-id fold_2023 \
  --validation-split validation_select \
  --model-size base \
  --generative-objective flow \
  --ae-epochs 3 \
  --diffusion-epochs 8 \
  --batch-size 512
```

## 5. 2026-07-17 实验循环记录

### 5.1 宽候选与 rank 入模

假设：严格 `buy_signal` 候选容量太小，不利于 deep learning 学到弱转强附近的连续排序结构。应改成每天按弱转强 `score` 做宽候选排名，并把 rank/score 条件显式输入模型。

落地：

1. `build_wts_diffusion_dataset_v1.py` 新增 `--candidate-mode score_rank`。
2. 宽候选模式不严格过滤 `selector.where`，而是在 `history_ok & finite(score)` 的全主板池内按 `score` 排名取 TopN。
3. 样本保留 `wts_rank`、`wts_rank_pct`、`wts_score_z`、`wts_score_pct`、`wts_selector_where`、`wts_pool_size`。
4. `wts_diffusion_v1.py` 将候选静态特征作为 AE 条件输入，`50m` 模型使用 6 维 candidate feature。

全量数据：

```text
root: tmp/weak-to-strong-diffusion-v1/score_rank_top100_full_v1
candidate_mode: score_rank
candidate_topn: 100
samples: 249170
mainboard_instruments: 3189
fold_2025 train: 188988
fold_2025 validation_select: 16779
fold_2025 prediction: 20995
```

### 5.2 50M 模型与 checkpoint 策略

假设：首轮模型需要约 50M 参数，AE 约 10M，diffusion head 约 40M，且每个 epoch 必须保存 checkpoint，便于回溯。

落地参数量：

```text
model_size: 50m
AE params: 10648744
diffusion params: 42316830
total params: 52965574
```

训练脚本已支持：

```text
--init-ae-checkpoint     从已有 AE checkpoint 初始化并继续训练 AE
--resume-ae-checkpoint   从已有 AE checkpoint 初始化并跳过 AE，直接训练 diffusion
--ae-start-epoch         手动指定 AE 起始 epoch；默认可从 checkpoint epoch 推断
```

checkpoint 已按 epoch 保存，例如：

```text
wts-diffusion-20260717-095759/checkpoints/ae_epoch_001.pt..ae_epoch_005.pt
wts-diffusion-ae20-flow24-20260717-101042/checkpoints/ae_epoch_006.pt..ae_epoch_020.pt
wts-ae40-only-20260717-101800/checkpoints/ae_epoch_021.pt..ae_epoch_040.pt
wts-ae100-lr3e5-only-20260717-102527/checkpoints/ae_epoch_041.pt..ae_epoch_066.pt
```

### 5.3 AE 重建实验结论

目标：AE 的 normalized reconstruction error 应尽量进入 `1e-2` 量级，否则 latent 可能无法支撑后续路径 diffusion。

已完成实验：

```text
AE epoch 20 validation combined smooth_l1: 0.109928
AE epoch 40 validation combined smooth_l1: 0.099128
AE epoch 66 validation combined smooth_l1: 0.098375
```

AE40 抽样 4096 条 `fold_2025 validation_select` 的 reconstruction 诊断：

```text
combined smooth_l1 mean: 0.099375
ts smooth_l1 mean: 0.045644
market smooth_l1 mean: 0.076758
ts MAE mean: 0.202590
market MAE mean: 0.268513
ts RMSE mean: 0.303243
market RMSE mean: 0.439447
```

分析：

1. AE 从 20 训到 40 有明显收益，validation loss 从约 `0.1099` 降到约 `0.0991`。
2. AE40 之后低学习率继续到 66，validation 基本平台在 `0.0983` 附近，离 `1e-2` 目标差距很大。
3. 继续堆 epoch 的边际收益很低，当前瓶颈更可能来自 AE 结构/压缩率/重建目标，而不是训练轮数不足。
4. 当前 `50m` 配置的 AE latent 为 `4 * 64 = 256` 维，压缩 60 日多特征序列过强；同时同时重建 raw/derived/market 的目标较宽。

结论：不应基于当前 AE 继续投入正式 diffusion 训练。下一轮应先提高 AE 表征质量，再训练 diffusion。

### 5.4 下一轮假设

候选方案 A：提高 AE latent 容量，优先保持工程改动小。

```text
model_size: ae_wide_80m 或 70m
latent_tokens: 16
latent_dim: 128
d_model: 256 或 320
n_layers: 6
AE 参数目标: 30M-50M
diffusion head: 可暂用当前 40M 或先不训练
目标: validation ts_smooth_l1 <= 0.02，combined smooth_l1 显著低于 0.05
```

候选方案 B：改 reconstruction 目标，减少无效重建负担。

```text
不重建所有 derived/market 字段；只重建对未来收益路径更关键的 price/volume/relative factor 子集。
或者增加多头 loss：price fields 权重大，market/derived 权重小。
```

候选方案 C：跳过重建型 AE，改监督式 encoder 预训练。

```text
直接用 encoder + MLP 预测 y_relative 的 3/5/10/20/30 日点位或路径 PCA。
待 encoder 对路径有监督信号后，再接 diffusion/flow head。
```

下一步建议：先实施方案 A 的最小版本，仅新增一个更宽 AE model size，不改数据与评估口径；AE 达标后再恢复 diffusion 训练。

### 5.5 轻压缩 AE + flow head 首轮正式结果

假设：前一版 AE latent `256` 维压缩过强，导致重建误差平台在 `~0.098`。如果把 AE 改成轻压缩结构，使 latent 总维度接近输入容量的 `0.8`，encoder 能保留足够多的行情/市场状态信息，后续 flow head 才可能学到未来 30 日相对收益路径的排序信号。

落地：新增 `model_size=ae_wide_08`，只扩展模型配置，不改数据、split、评估口径。

```text
input_dim_reference: (52 stock + 20 market) * 60 + 6 candidate = 4326
latent_dim: 27 * 128 = 3456
latent_ratio: 0.7989
AE params: 40427736
flow/diffusion head params: 52556830
total params: 92984566
```

AE 训练：

```text
run: score_rank_top100_full_v1/runs/wts-ae-wide08-only-20260717-104101
data: fold_2025 train, validation_select
checkpoint policy: every epoch saved
stopped after complete epoch 44 to enter flow training
epoch 20 validation combined smooth_l1: 0.011434
epoch 44 validation combined smooth_l1: 0.004827
checkpoint: checkpoints/ae_epoch_044.pt
```

AE20 抽样 4096 条 `fold_2025 validation_select` reconstruction 诊断：

```text
combined smooth_l1 mean: 0.011488, median: 0.008247, p95: 0.022724
ts smooth_l1 mean: 0.004960, median: 0.004144, p95: 0.010148
market smooth_l1 mean: 0.009326, median: 0.005805, p95: 0.019989
ts MAE mean: 0.070064
market MAE mean: 0.080896
ts RMSE mean: 0.096923
market RMSE mean: 0.131599
```

结论：轻压缩 AE 解决了重建瓶颈。与 50M AE40 的 `combined smooth_l1 mean ~0.099` 相比，AE20 已进入 `1e-2`，AE44 进一步到 `~5e-3`。这版 encoder 可以作为后续预测头底座。

Flow head 训练：

```text
run: score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330
init AE: wts-ae-wide08-only-20260717-104101/checkpoints/ae_epoch_044.pt
model_size: ae_wide_08
objective: flow
diffusion/flow epochs: 24
best validation epoch: 22, validation loss: 0.046423
last epoch 24 validation loss: 0.048397
main eval checkpoint: checkpoints/diffusion_epoch_022.pt
```

Prediction split OOS 评估：

```text
split: fold_2025 prediction
rows: 20995
days: 213
k_samples: 16
inference_steps: 20
predictions: evaluation/predictions_prediction_epoch022_k16.parquet
metrics: evaluation/metrics_prediction_epoch022_k16.json
```

路径误差：

```text
horizon  ADE_abs_mean  FDE_abs_mean  ADE_abs_median  FDE_abs_median
3d       0.044500      0.056170      0.033038        0.038761
5d       0.053457      0.069904      0.039537        0.048578
7d       0.060589      0.080607      0.044630        0.055924
10d      0.069249      0.093651      0.050353        0.064301
15d      0.080490      0.110254      0.058198        0.075281
20d      0.090112      0.125946      0.065115        0.085859
30d      0.105159      0.142479      0.075223        0.097199
```

排序信号：模型分数显著优于原始弱转强分数。核心 score `score_mean_minus_0.5std_20d` 的相对收益 RankIC：

```text
label  RankIC mean  RankIC IR  positive days
y_3d   0.056656     0.325814   0.6432
y_5d   0.092717     0.552453   0.7183
y_7d   0.113617     0.657782   0.7183
y_10d  0.124506     0.765755   0.7371
y_15d  0.143233     0.927049   0.8122
y_20d  0.156705     1.141022   0.8826
y_30d  0.161332     1.296137   0.9108
```

对比原始弱转强 score：

```text
label  model RankIC        original WTS RankIC
y_10d  0.124506            -0.046086
y_20d  0.156705            -0.063769
y_30d  0.161332            -0.064009
```

TopK 策略含义：每天 Top100 内按模型分数重排，TopK 的真实收益明显优于原始弱转强 rank。下表为按天等权 TopK 的 OOS 均值。

```text
horizon score                         top5 rel  top5 abs  top10 rel  top10 abs  top20 rel  top20 abs
10d     score_tail_utility_30d        0.003755  0.017343  0.000358   0.013946   -0.002073  0.011516
10d     score_mean_minus_0.5std_20d   0.002489  0.016078  0.001138   0.014726   -0.002747  0.010841
10d     original WTS                  -0.021289 -0.007701 -0.019592  -0.006003  -0.014888 -0.001300

20d     score_mean_minus_0.5std_20d   0.001740  0.028037  0.001430   0.027726   -0.004736  0.021560
20d     score_tail_utility_30d        -0.002507 0.023789  -0.003096  0.023201   -0.004434  0.021863
20d     original WTS                  -0.044511 -0.018215 -0.033601  -0.007305  -0.026113  0.000183

30d     score_mean_minus_0.5std_20d   0.000138  0.038063  0.000225   0.038150   -0.005568  0.032357
30d     score_tail_utility_30d        -0.007421 0.030504  -0.003144  0.034781   -0.005288  0.032637
30d     original WTS                  -0.051502 -0.013577 -0.036938  0.000986   -0.030837  0.007088
```

分析：

1. AE 轻压缩是必要修正。重建误差达标后，flow head 在 OOS prediction split 上产生了清晰的排序信号。
2. 模型预测路径的绝对误差不算小，30 日 `ADE_abs_mean ~10.5%`、`FDE_abs_mean ~14.2%`，因此这版模型更像排序器/筛选器，而不是可直接用于精确路径交易的 point forecast。
3. 模型分数对 10-30 日相对收益的 RankIC 从 `0.12` 提升到 `0.16`，且 positive days 最高超过 `90%`；这说明 deep learning/flow 对弱转强宽候选内重排有真实帮助。
4. 原始 WTS score 在该宽候选 Top100 内对未来 10-30 日相对收益为负 RankIC，说明原始 rank 更像候选池生成器，不适合作为最终排序器。
5. TopK 结果显示模型能把原始 WTS TopK 的负相对收益明显拉回，尤其 10 日和 20 日 Top5/Top10；但 Top20 及 30 日相对收益仍偏弱，说明容量扩大后边际样本质量下降，需要结合执行与持仓规则进一步验证。

下一轮假设：

1. 先把模型 score 接入一个轻量回测评估，验证 Top5/Top10/Top20 在真实买入规则、持仓 5/10/20/30 日、止盈止损/换仓规则下的收益、回撤、换手，而不是只看标签均值。
2. 训练一个直接监督的 deterministic head 或多任务 head，预测 3/5/7/10/15/20/30 日相对收益点位，与 flow head 做 RankIC/TopK 对比；如果 deterministic head 相近，则 flow 的采样成本可能不必要。
3. 保留 AE44 encoder，尝试 fine-tune encoder + head，而不是完全冻结 AE，观察是否进一步提升 RankIC，但需要加早停防止过拟合。
4. 对候选池做容量曲线实验：Top50/Top100/Top200，判断模型排序收益随候选宽度的衰减情况。

## 2026-07-17 QuantX OOS 回测接入

目的：把 `score_mean_minus_0.5std_20d` 接入 QuantX 的真实弱转强策略，验证模型是否能改善弱转强宽候选池的收益、回撤和容量，而不是只停留在标签 RankIC/TopK replay。

实现方式：

```text
QuantX repo: ${HOME}/git/quantization/quantx
selector extension: FormulaSelector 支持 selector.external_score
external score behavior: 先按原始 buy_signal 生成候选，再用外部模型分数覆盖 __selector_score 排序
missing policy in this run: drop
prediction score file: score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330/evaluation/predictions_prediction_epoch022_k16.parquet
score column: score_mean_minus_0.5std_20d
```

验证：

```text
pytest: conda run -n test python -m pytest tests/strategy/test_config_strategy.py -q
result: 41 passed
dry-run original: ok
dry-run ML external score: ok
universe: all_mainboard, 3183 symbols
window: 2025-01-02 .. 2025-11-20
signal score coverage: 2025-01-02 .. 2025-11-19, 20995 rows, 213 days, 1595 instruments, score_nulls=0
```

回测配置：

```text
original config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_replay/wts_original_pos8_topk12_2025_prediction.yaml
ML config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_replay/wts_flow_score_pos8_topk12_2025_prediction.yaml
original run: ${HOME}/git/quantization/quantx/runs/20260717_114134_wts_original_pos8_topk12_2025_prediction
ML run: ${HOME}/git/quantization/quantx/runs/20260717_114134_wts_flow_score_pos8_topk12_2025_prediction
```

正式 QuantX 回测结果：

```text
metric                 original WTS      ML external score      delta
total_return           -13.2217%         -3.9551%               +9.2666 pct
annual_return          -14.8496%         -4.4713%               +10.3784 pct
max_drawdown           -26.0226%         -23.7027%              +2.3190 pct
sharpe                 -0.5380           -0.1614                +0.3766
sortino                -0.6990           -0.2343                +0.4648
calmar                 -0.5706           -0.1886                +0.3820
trades                 243               180                    -63
buys                   109               86                     -23
sells                  134               94                     -40
rejects                8                 1                      -7
win_rate               52.9851%          47.8723%               -5.1127 pct
profit_factor          0.8093            0.9530                 +0.1437
avg_closed_return      2.9781%           2.2755%                -0.7026 pct
avg_holding_days       21.84             27.90                  +6.06
avg_position_count     7.916             7.589                  -0.327
total_cost             14045.18          13437.00               -608.19
```

候选覆盖诊断：

```text
daily rows: 214

original WTS:
  raw_candidate_count mean: 133.19, median: 98.5, p90: 293, max: 511
  selected_count mean: 11.82, median: 12, nonzero selected days: 214
  sum raw: 28502, sum selected: 2530

ML external score with missing=drop:
  raw_candidate_count mean after drop: 9.72, median: 8, p90: 19.7, max: 30
  selected_count mean: 7.97, median: 8, nonzero selected days: 208
  sum raw after drop: 2080, sum selected: 1706
```

分析：

1. 正式 QuantX OOS 回测支持前面离线 replay 的方向：模型分数作为弱转强候选排序器明显改善了亏损，`total_return` 提升约 `9.27pct`，Sharpe 从 `-0.54` 改到 `-0.16`，profit factor 从 `0.81` 改到 `0.95`。
2. 但这轮还不能证明“扩大容量”。原因是模型预测文件只覆盖离线 score-rank Top100 候选，和 QuantX 当前 `weak_to_strong_pool` 的每日候选交集很小。`missing=drop` 后每日可评分候选均值只有 `9.72`，原始弱转强池均值是 `133.19`。
3. 因此当前 ML 回测本质是一个强过滤器实验：它过滤掉大量没有模型分数的弱转强候选，降低交易数和拒单数，同时避开了不少亏损交易。收益改善有效，但容量结论偏保守。
4. ML 版胜率低于原始版，但 profit factor 更高、回撤更小，说明模型更可能改善了亏损尾部和持仓质量，而不是单纯提高命中率。

下一轮假设：

1. 做 `missing=keep_original` 对照：模型覆盖到的候选用 ML 分数，未覆盖候选保留原始弱转强 score。若收益仍明显优于 original 且 selected_count 回到接近 12，则说明模型能在不牺牲容量的情况下改善排序。
2. 生成与 QuantX `weak_to_strong_pool` 完全同口径的模型推理候选文件，而不是只用离线 score-rank Top100。这样才能真正评估弱转强容量扩张。
3. 做 topk/max_positions 容量曲线：`topk=12/20/30`、`max_positions=8/12/16`，观察 ML 排序在更大容量下的收益和回撤衰减。
4. 如果 `keep_original` 结果显著差于 `drop`，说明模型更适合作为准入过滤器；下一步应把外部分数改成 selector gate，例如只买 `ml_score > threshold` 或每日 ML TopK，而不是直接尝试扩容。

### keep_original 保容量对照审计

随后补跑 `selector.external_score.missing=keep_original`，目的是保留 QuantX 原始弱转强池容量：模型覆盖到的候选使用 ML score，未覆盖候选保留原始 WTS score。

```text
config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_replay/wts_flow_score_keep_original_pos8_topk12_2025_prediction.yaml
run: ${HOME}/git/quantization/quantx/runs/20260717_114438_wts_flow_score_keep_original_pos8_topk12_2025_prediction
total_return: +51.8428%
annual_return: +60.5527%
max_drawdown: -7.1312%
sharpe: 2.6744
trades: 222
buys: 98
sells: 124
win_rate: 66.1290%
profit_factor: 2.1828
```

但这条不能作为 ML 有效结论，原因是分数尺度不可比：

```text
original WTS selected score median: 2.3497
ML score selected score median in drop run: -0.0997
keep_original selected rows captured: 2530
keep_original selected rows covered by model: 31
keep_original selected model coverage: 1.2253%
keep_original raw preview model coverage: 11.2962%
```

解释：`keep_original` 直接混排了两个不同尺度的分数。ML score 大多在 `-0.3 ~ 0.1`，原始 WTS score 多在 `2.0+`，所以未覆盖候选天然排在模型覆盖候选前面。最终 Top12 只有 `1.2%` 来自模型覆盖票，这条高收益更像是“排除了少量模型覆盖票后的原始分数策略”，不是 ML 排序策略。

修正后的下一轮方向：

1. 不能直接用 `keep_original` 混排不同尺度 score。若要保容量，需要新增一个同尺度融合方式，例如 `external_score.rank_overlay`：每天只在模型覆盖候选内部按 ML score 重排，未覆盖候选作为 fallback 排在覆盖候选之后；或者先把 ML score 转成当日分位 rank，再和原始 WTS rank 做可控融合。
2. 更干净的实验是导出 QuantX `weak_to_strong_pool` 的完整每日候选作为模型推理 universe，让每个候选都有 ML score，再做 `missing=drop` 或纯 ML 排序。这样才能回答容量问题。
3. 当前最小下一步：生成 `candidate_limit=1000` 的 QuantX 完整候选快照，确认每日候选全集和离线 score-rank Top100 的真实交集，再决定是补推理还是重建数据集。

### 基准口径修正

重要修正：上面的 `original WTS` 不是长期最优的基础弱转强规则，而是为了 diffusion 排序实验构造的宽候选扩容池：`weak_to_strong_score_pool_pos8_topk12_2016_2026_mainboard`。它放宽了原始强形态条件，目标是增加候选量，不代表基础弱转强收益。

QuantX 历史 run 中，基础/最优弱转强规则收益确实远高于这条宽池：

```text
example best long-run: stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_d18_p2_t8
window: 2016-01-04 .. 2026-07-15
total_return: 63.1826
annual_return: 48.4483%
max_drawdown: -30.6617%
sharpe: 1.5592
trades: 1134
```

同一 diffusion prediction OOS 窗口补跑基础最优近似对照：

```text
config source: stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85_recommended.yaml
temp config: /tmp/wts_best_scale12_keep85_2025_prediction.yaml
run: ${HOME}/git/quantization/quantx/runs/20260717_114744_wts_best_scale12_keep85_2025_prediction
window: 2025-01-02 .. 2025-11-20
total_return: +0.0616%
annual_return: +0.0698%
max_drawdown: -26.1965%
sharpe: 0.0022
trades: 88
buys: 44
sells: 44
```

解释：2025 这个 OOS 窗口对基础弱转强也不友好，但基础规则不是 `-13.22%`。因此 diffusion 当前结论必须改写为：

1. 现有模型在“过宽的弱转强扩容池”上有过滤/排序价值，把扩容池亏损从 `-13.22%` 改到 `-3.96%`。
2. 现有实验尚未证明模型能超越或扩容长期最优基础弱转强规则。
3. 下一步要围绕基础规则构造“近邻扩容池”，例如保留 `long_ok / upper_then_lower / short_upper_0 / zx_ok` 等核心形态，只轻微放宽安全日、流动性或 TopK/仓位，然后让模型在这个近邻池内排序。

### 基础弱转强 + diffusion 交集过滤

为了回应基准口径问题，补跑基础最优规则在同一 OOS 窗口的 ML 交集过滤实验。实验只改 selector 外部分数，基础规则的买点、卖出、仓位和交易成本保持不变。

```text
base config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_replay/wts_best_scale12_keep85_2025_prediction.yaml
ML config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_replay/wts_best_scale12_keep85_flow_drop_2025_prediction.yaml
base run: ${HOME}/git/quantization/quantx/runs/20260717_115051_wts_best_scale12_keep85_2025_prediction
ML run: ${HOME}/git/quantization/quantx/runs/20260717_115051_wts_best_scale12_keep85_flow_drop_2025_prediction
window: 2025-01-02 .. 2025-11-20
universe: all_mainboard, 3183 symbols
external score: score_mean_minus_0.5std_20d
missing policy: drop
```

正式 QuantX 结果：

```text
metric                 base best WTS      base + ML intersection     delta
total_return           +0.0616%           +41.9523%                  +41.8907 pct
annual_return          +0.0698%           +48.7509%                  +48.6811 pct
max_drawdown           -26.1965%          -17.0418%                  +9.1546 pct
sharpe                 0.0022             1.8909                     +1.8887
sortino                0.0032             2.4076                     +2.4043
calmar                 0.0027             2.8607                     +2.8580
trades                 88                 56                         -32
buys                   44                 25                         -19
sells                  44                 31                         -13
rejects                3                  1                          -2
win_rate               47.7273%           70.9677%                   +23.2405 pct
profit_factor          1.0661             3.2580                     +2.1920
avg_closed_return      1.4742%            7.6226%                    +6.1484 pct
avg_holding_days       29.64              25.65                      -3.99
avg_position_count     4.150              2.220                      -1.930
total_cost             9658.72            8731.25                    -927.47
```

候选覆盖审计：

```text
base best WTS:
  raw_candidate_count mean: 2.57, median: 1, p90: 6, sum: 549
  selected_count mean: 1.33, median: 1, p90: 4, sum: 284
  nonzero selected days: 121
  unique buy symbols: 43

base + ML intersection:
  raw_candidate_count mean after drop: 0.21, median: 0, p90: 1, sum: 45
  selected_count mean: 0.21, median: 0, p90: 1, sum: 45
  nonzero selected days: 29
  unique buy symbols: 25

base selected candidates covered by current model file:
  selected captured: 284
  model-covered selected: 45
  selected coverage: 15.85%

base raw candidates covered by current model file:
  raw captured from artifacts: 481
  model-covered raw: 41
  raw coverage: 8.52%

ML score distribution among covered base candidates:
  count: 45
  mean: -0.0864
  median: -0.0825
  p10: -0.1435
  p90: -0.0136
  max: 0.0122
```

分析：

1. 这是目前最强的正向证据：当前 diffusion/flow 分数对基础弱转强强形态也有价值。只买模型覆盖交集后，2025 OOS 从基本持平变成 `+41.95%`，回撤降低，胜率和 profit factor 大幅提升。
2. 但这仍不是容量扩张实验。模型文件只覆盖基础规则 selected 的 `15.85%`，覆盖 raw 候选约 `8.52%`；ML 版只有 25 次买入、平均持仓数 `2.22`，明显低于基础策略的持仓容量。
3. 当前模型更像“高置信过滤器”：当基础强形态候选也落入模型训练/推理的 score-rank Top100 universe 时，样本质量显著更高；但覆盖太低，不能直接用来扩仓。
4. 这也解释了前面宽池实验的现象：`missing=drop` 有效，是因为模型覆盖子集质量更高；`keep_original` 直接混排不同尺度分数无效，因为它没有真正把 ML 分数推到排序主导位置。

下一轮假设：

1. 以基础弱转强强形态为核心，构造“近邻扩容池”并重新生成模型推理文件。目标不是无限放宽，而是让模型覆盖基础规则附近的候选，例如保持 `long_ok / upper_then_lower / short_upper_0 / zx_ok`，只逐步放宽 `day_ok`、`bbi_trend_ok`、`dif > 0` 或 `topk/max_positions`。
2. 在这个近邻池上重跑 AE44 + flow head 推理，保证每个候选都有同尺度 ML score，再做 TopK/容量曲线：`topk=4/6/8/12`、`max_positions=5/6/8/10`。
3. 增加一个直接监督 rank head 或 quantile head，对基础近邻池预测 10/20/30 日相对收益，与 flow score 比较。如果直接监督 head 接近 flow，则优先用低成本 head 做大规模推理。
4. 下一轮评估标准应同时看三件事：收益是否超过基础强规则、平均持仓数/交易数是否提升、ADE/FDE/RankIC 是否保持稳定。只提高收益但容量下降，定义为过滤器；收益和容量同时提升，才算弱转强扩容成功。

### 全市场 diffusion score 预计算接入

用户澄清后，第一阶段改成更接近最终生产形态的流程：模型提前给全市场主板股票按 `(date, instrument)` 生成 score，QuantX 弱转强策略仍按原规则筛候选，然后候选查模型 score 做排序/TopK。这避免了旧 `predictions_prediction_epoch022_k16.parquet` 只覆盖训练 universe 小交集的问题。

推理脚本：

```text
script: ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/infer_market_scores_v1.py
run_dir: ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330
checkpoint: checkpoints/diffusion_epoch_022.pt
window: 2025-01-02 .. 2025-11-19 signal dates
universe: mainboard eligible stocks
k_samples: 4
inference_steps: 10
batch_size: 512
device: cuda
output: evaluation/market_scores_2025_mainboard_epoch022_k4_s10.parquet
```

全量输出：

```text
rows: 644194
days: 213
instruments: 3128
null score_mean_minus_0.5std_20d: 0
runtime: about 2m26s

score_mean_minus_0.5std_20d:
  mean: -0.100105
  std: 0.071562
  p01: -0.289854
  p50: -0.097292
  p95: 0.011443
  p99: 0.060642
  max: 0.311948
```

初始覆盖审计：

```text
base best WTS previous selected: 284
covered by full-market score: 275
coverage: 96.8310%

base best WTS previous raw preview: 481
covered by full-market score: 468
coverage: 97.2973%

wide WTS previous selected: 2530
covered by full-market score: 2478
coverage: 97.9447%

wide WTS previous raw preview: 4143
covered by full-market score: 4050
coverage: 97.7552%
```

随后生成三份 QuantX 配置，基础买卖规则、成本和执行逻辑保持和 `stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85_recommended.yaml` 一致，只改 OOS 窗口、外部分数、TopK 和最大持仓：

```text
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_best_scale12_keep85_market_score_top4_pos5_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_best_scale12_keep85_market_score_top8_pos8_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_best_scale12_keep85_market_score_top12_pos10_2025_prediction.yaml

external_score:
  path: ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330/evaluation/market_scores_2025_mainboard_epoch022_k4_s10.parquet
  date_col: signal_date
  instrument_col: instrument
  score_col: score_mean_minus_0.5std_20d
  missing: drop
```

三份配置 dry-run 均通过，正式回测结果：

```text
metric                    base top4/pos5    old intersection    market top4/pos5    market top8/pos8    market top12/pos10
total_return              +0.0616%          +41.9523%           +8.9466%            +2.9351%            +4.0131%
annual_return             +0.0698%          +48.7509%           +10.2004%           +3.3335%            +4.5611%
max_drawdown              -26.1965%         -17.0418%           -26.3468%           -26.5592%           -26.5014%
sharpe                    0.0022            1.8909              0.3422              0.1115              0.1547
sortino                   0.0032            2.4076              0.4856              0.1600              0.2176
calmar                    0.0027            2.8607              0.3872              0.1255              0.1721
trades                    88                56                  85                  122                 144
buys                      44                25                  41                  61                  71
sells                     44                31                  44                  61                  73
win_rate                  47.7273%          70.9677%            59.0909%            54.0984%            56.1644%
profit_factor             1.0661            3.2580              1.3530              1.1379              1.1695
avg_holding_days          29.64             25.65               30.52               29.98               29.74
total_cost                9658.72           8731.25             9740.15             9911.89             9704.15
```

候选池统计：

```text
label                     raw_total    selected_total    raw_cov    selected_cov    selected_score_mean    selected_score_p50
base_top4_pos5            481          284               97.2973%   96.8310%        -0.106715              -0.105917
old_intersection_top4     45           45                100.0000%  100.0000%       -0.126829              -0.124533
market_score_top4_pos5    471          276               100.0000%  100.0000%       -0.075155              -0.069727
market_score_top8_pos8    471          368               100.0000%  100.0000%       -0.078802              -0.074809
market_score_top12_pos10  471          419               100.0000%  100.0000%       -0.082916              -0.079378
```

与基础规则 selected 的重合：

```text
market_score_top4_pos5:
  selected: 276
  overlap with base: 198
  overlap_rate_of_variant: 71.7391%
  base_covered_by_variant: 69.7183%

market_score_top8_pos8:
  selected: 368
  overlap with base: 239
  overlap_rate_of_variant: 64.9457%
  base_covered_by_variant: 84.1549%

market_score_top12_pos10:
  selected: 419
  overlap with base: 255
  overlap_rate_of_variant: 60.8592%
  base_covered_by_variant: 89.7887%
```

阶段性结论：

1. 第一阶段已经证明“全市场预生成 score，再由弱转强候选查分排序”的工程路径可跑通，且候选覆盖从旧模型文件的 `8%~16%` 提升到回测内 `100%`。
2. 全市场 score 对基础弱转强有正收益贡献：同窗口基础规则 `+0.06%`，全市场 score top4/pos5 到 `+8.95%`，胜率从 `47.73%` 到 `59.09%`。
3. 但扩容证据还不强：top8/pos8 和 top12/pos10 虽然交易数、selected 数量增加，但收益只到 `+2.94%` 和 `+4.01%`，回撤仍接近 `-26.5%`。这说明当前 score 可以改善排序，但还不能证明能把弱转强容量明显放大并保持高收益。
4. 旧交集过滤的 `+41.95%` 仍明显更强，说明模型原训练/推理 universe 内的高置信样本质量更好；而全市场推理使用了中性候选特征，存在训练分布外推，分数被稀释。
5. 下一步最值得做的不是直接扩大 topk，而是训练/推理 universe 对齐：用基础弱转强近邻池或全市场候选重建训练样本，让模型学习“所有弱转强候选里的相对 rank”，再做容量曲线。若只做快速验证，可以先用全市场文件重跑 `k=16, steps=20`，看低采样推理噪声是否影响排序。

验证：

```text
conda run -n test python -m pytest tests/strategy/test_config_strategy.py -q
41 passed in 0.39s
```

### 主力可视化弱转强口径修正：fast_exit_d18_p2_t8

用户指出 QuantX 可视化层中纯规则弱转强收益很高。复查后确认，生产/可视化主力弱转强不是前一段用于快速对比的 `scale_12_keep85` 近似基线，而是：

```text
config: ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t8.yaml
strategy: stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_d18_p2_t8
title: 弱转强深亏提前
model/external_score: none
```

可视化长期 run：

```text
run: ${HOME}/git/quantization/quantx/runs/20260716_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_d18_p2_t8
window: 2016-01-04 .. 2026-07-16
total_return: +6025.6454%
annual_return: +47.7772%
max_drawdown: -30.5502%
sharpe: 1.5414
trades: 1139
win_rate: 62.3256%
profit_factor: 2.2098
```

该长期 run 的净值截片：

```text
2025-01-02 .. 2025-11-20: +0.5760%
2025-01-02 .. 2025-12-31: -11.5707%
2026-01-05 .. 2026-07-16: +115.2053%
2025-07-16 .. 2026-07-16: +119.5743%
```

解释：长期高收益是真实存在的纯规则结果，但 2025 年 ML OOS 窗口本身不是这条策略的强势区间；真正大幅贡献来自 2026 年初至今。前一轮 `+0.0616%` 是从 2025-01-02 重新初始化资金跑短窗口，不是可视化长期 run 的完整展示口径。

为了严格同口径比较，重新从 `fast_exit_d18_p2_t8.yaml` 派生 OOS 配置，并保持买点、卖点、成本、执行参数一致，只改：

1. OOS window: `2025-01-02 .. 2025-11-20`
2. ML 版本增加全市场 `external_score`
3. 容量曲线改 `topk/max_positions`

生成配置：

```text
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top4_pos5_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top6_pos6_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top8_pos8_2025_prediction.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top12_pos10_2025_prediction.yaml
```

所有配置 dry-run 通过。正式回测：

```text
metric              fast_exit rule   ML top4/pos5   ML top6/pos6   ML top8/pos8   ML top12/pos10
total_return        +0.0616%         +10.9134%      +10.6393%      +2.9351%       +4.0131%
annual_return       +0.0698%         +12.4583%      +12.1432%      +3.3335%       +4.5611%
max_drawdown        -26.1965%        -26.3393%      -26.5829%      -26.5592%      -26.5014%
sharpe              0.0022           0.4177         0.4059         0.1115         0.1547
sortino             0.0032           0.5955         0.5777         0.1600         0.2176
calmar              0.0027           0.4730         0.4568         0.1255         0.1721
trades              88               86             96             122            144
buys                44               42             48             61             71
sells               44               44             48             61             73
win_rate            47.7273%         54.5455%       54.1667%       54.0984%       56.1644%
profit_factor       1.0661           1.4200         1.4081         1.1379         1.1695
avg_closed_return   1.4742%          2.0407%        2.7549%        3.1918%        3.7501%
avg_holding_days    29.64            31.07          31.00          29.98          29.74
avg_position_count  4.150            4.070          4.626          5.743          6.678
reject_count        3                4              3              3              2
```

候选统计：

```text
label                    raw_total   selected_total   selected_avg_day   selected_nonzero_days   raw_cov     selected_cov   selected_score_mean
fast_exit_rule           481         284              1.327              121                     97.2973%    96.8310%       -0.106715
ML top4/pos5             471         276              1.290              118                     100.0000%   100.0000%      -0.075155
ML top6/pos6             471         334              1.561              118                     100.0000%   100.0000%      -0.078329
ML top8/pos8             471         368              1.720              118                     100.0000%   100.0000%      -0.078802
ML top12/pos10           471         419              1.958              118                     100.0000%   100.0000%      -0.082916
```

与纯规则 selected 的重合：

```text
ML top4/pos5:
  selected: 276
  overlap with rule: 198
  variant overlap rate: 71.7391%
  rule covered by variant: 69.7183%

ML top6/pos6:
  selected: 334
  overlap with rule: 227
  variant overlap rate: 67.9641%
  rule covered by variant: 79.9296%

ML top8/pos8:
  selected: 368
  overlap with rule: 239
  variant overlap rate: 64.9457%
  rule covered by variant: 84.1549%

ML top12/pos10:
  selected: 419
  overlap with rule: 255
  variant overlap rate: 60.8592%
  rule covered by variant: 89.7887%
```

修正后结论：

1. 以可视化主力纯规则 `fast_exit_d18_p2_t8` 为基线，当前全市场 diffusion score 在 2025 OOS 同初始资金口径下有明确增益：`+0.06% -> +10.91%`。
2. `top6/pos6` 是目前更有意义的容量点：selected 从 `284` 增到 `334`，平均持仓从 `4.15` 增到 `4.63`，收益仍有 `+10.64%`。这比前一段 `top8/top12` 更接近“收益和容量同时改善”。
3. `top8/top12` 继续扩容后收益明显下降，说明当前 score 的边际排序能力不够支撑更大容量；模型能帮忙重排和小幅扩容，但还不是无限扩仓信号。
4. 可视化长期收益很高主要是长期复利和 2026 强势段贡献。ML 当前只有 2025 score 文件，无法直接和 2016-2026 全周期可视化收益等价比较；下一步若要完全生产口径，需要生成更长时间段全市场 score，或至少覆盖 2025-07-16 .. 2026-07-16 这段强势窗口。
5. 下一步优先级：先把全市场 score 延展到 2026 强势窗口，验证模型在弱转强真正赚钱阶段是否还能提升；然后再考虑用 `top6/pos6` 作为第一版扩容参数，而不是直接上 `top8/top12`。

验证：

```text
conda run -n test python -m pytest tests/strategy/test_config_strategy.py -q
41 passed in 0.37s
```

### 训练外长窗口压力测试：2024-01-02 至 2026-07-16

新的假设：如果当前 diffusion score 真的具备弱转强排序能力，那么在训练标签之外的 2024-2026 长窗口中，直接接入主力纯规则 `fast_exit_d18_p2_t8` 后，应当至少不显著弱于纯规则 baseline，并在部分容量点上提升收益或容量。

模型状态：

```text
checkpoint: wts-flow-wide08-ae44-20260717-105330/checkpoints/diffusion_epoch_022.pt
fold: fold_2025
train signal_date: 2016-01-04 .. 2023-11-17
train label_end_date: 2016-02-22 .. 2023-12-29
validation signal_date: 2024-01-02 .. 2024-11-19
prediction signal_date in original fold: 2025-01-02 .. 2025-11-19
```

注意：2024 属于训练集之外，但曾用于 validation/epoch 选择，因此不能称为严格 OOS；2025-2026 是更干净的前向外推窗口。此实验的目的不是证明最终生产可用，而是压力测试“当前模型 + 全市场 neutral candidate features + 直接替换 selector score”的稳定性。

生成全市场 score：

```text
script: ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/infer_market_scores_v1.py
output: ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330/evaluation/market_scores_2024_2026_mainboard_epoch022_k4_s10.parquet
start: 2024-01-02
end: 2026-07-15 signal date
rows: 1853305
days: 612
instruments: 3164
k_samples: 4
inference_steps: 10
candidate_feature_mode: neutral
runtime: 7m03s
```

score 分布：

```text
score_mean_minus_0.5std_20d:
  count: 1853305
  mean: -0.103698
  std: 0.077492
  p01: -0.325214
  p50: -0.098543
  p95: 0.012184
  p99: 0.062394
  max: 0.388902

by year:
  2024 rows 735905, days 242, instruments 3108, mean -0.088433, p95 0.018155
  2025 rows 735110, days 243, instruments 3131, mean -0.101632, p95 0.010270
  2026 rows 382290, days 127, instruments 3068, mean -0.137055, p95 0.000992
```

生成配置：

```text
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_2024_2026_latest_baseline.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top4_pos5_2024_2026_latest.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top6_pos6_2024_2026_latest.yaml
configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_market_score_top8_pos8_2024_2026_latest.yaml
```

四份配置 dry-run 均通过。正式 QuantX 结果：

```text
metric             pure rule     ML top4/pos5   ML top6/pos6   ML top8/pos8
total_return       +127.5249%    -15.4364%      -12.0159%      -1.3730%
annual_return      +38.2705%     -6.3952%       -4.9207%       -0.5435%
max_drawdown       -25.1662%     -46.0542%      -45.7793%      -43.7809%
sharpe             1.1889        -0.2004        -0.1576        -0.0181
trades             271           255            300            383
win_rate           55.6291%      53.9568%       53.9877%       54.5455%
profit_factor      1.8509        0.8348         0.8683         0.9683
avg_position_count 3.853         3.838          4.382          5.602
```

分段收益：

```text
period              pure rule     ML top4/pos5   ML top6/pos6   ML top8/pos8
2024                +25.3677%     -28.5705%      -28.1060%      -25.3653%
2025 calendar       -7.8151%      -4.3001%       -4.5561%       -3.9012%
2025 ML OOS slice   +4.8796%      +8.7166%       +8.0562%       +7.9750%
2026 YTD            +89.8566%     +26.0772%      +30.3923%      +39.3695%
```

候选覆盖与容量：

```text
label       raw_total  selected_total  selected_avg_day  nonzero_days  raw_cov    selected_cov  selected_score_mean
pure rule   920        630             1.028             298           96.9565%   96.5079%      -0.110432
ML top4     895        610             0.995             289           100.0000%  100.0000%     -0.091808
ML top6     895        699             1.140             289           100.0000%  100.0000%     -0.092701
ML top8     895        754             1.230             289           100.0000%  100.0000%     -0.092859
```

与纯规则 selected 的重合：

```text
ML top4:
  selected: 610
  overlap with pure rule: 497
  variant overlap rate: 81.4754%
  pure rule covered by variant: 78.8889%

ML top6:
  selected: 699
  overlap with pure rule: 542
  variant overlap rate: 77.5393%
  pure rule covered by variant: 86.0317%

ML top8:
  selected: 754
  overlap with pure rule: 566
  variant overlap rate: 75.0663%
  pure rule covered by variant: 89.8413%
```

分析：

1. 该长窗口压力测试否定了“当前全市场 neutral diffusion score 可以直接替换弱转强 selector score，并在 2024-2026 稳定提升”的假设。纯规则从 2024-01-02 空仓起跑到 2026-07-16 是 `+127.52%`，ML 直接重排全部显著落后。
2. 模型在 2025 ML OOS slice 仍有正向贡献：`+4.88% -> +8%左右`。这说明 2025 短窗提升不是完全偶然，但泛化到 2024/2026 不稳定。
3. 最大失败来自 2024 和 2026。2024 中 ML 把纯规则 `+25.37%` 改成约 `-25%~-29%`；2026 中纯规则 `+89.86%`，ML top8 只有 `+39.37%`。这说明模型替换排序会错过强势行情里的关键弱转强票。
4. 2026 全市场 score 分布显著下移，p95 只有 `0.000992`，模型整体偏悲观；但 QuantX 纯规则在 2026 很强。这是典型 regime shift 或训练/推理特征错配信号。
5. 覆盖率不是问题。ML selected coverage 是 `100%`，说明失败不是缺分数，而是分数排序本身在长窗口不稳。
6. 当前推理使用 `candidate_feature_mode=neutral`，而训练样本来自 `score_rank_top100` 候选并带 WTS rank/score 特征。全市场 neutral 外推会削弱模型对弱转强候选相对质量的识别，尤其在强势 regime 中可能把高收益候选压低。

下一轮方向：

1. 不应继续用当前 score 直接替换 selector score 做生产候选排序。它可以作为 2025 短窗 alpha 线索，但不是稳定 selector。
2. 更合理的下一步是做“基线保底 + 模型高置信覆盖”：只在模型分数显著高于当日候选分位时上调候选，而不是完全替换纯规则排序。需要在 QuantX selector 支持 rank overlay / score blend，或者离线生成候选级融合 score。
3. 训练/推理 universe 必须对齐。下一版数据集应直接使用主力弱转强候选或近邻扩容池，而不是 score-rank Top100 + 全市场 neutral 推理。模型要学习“弱转强候选内部排序”，不是全市场股票绝对排序。
4. 2026 是必须纳入的压力测试窗口。可以先构建 `forward_2026` 风格的全市场/弱转强候选推理评估，不训练 2026，用它检验 score 是否会错过强势段关键票。
5. 如果继续用 diffusion，建议增加一个轻量监督 rank head 或 pairwise ranking head，与 diffusion path score 做集成；路径预测 ADE/FDE 只证明走势拟合，不必然等价于策略收益排序。

### 长窗口失败归因：直接替换排序的问题

目的：解释为什么 `2024-01-02 .. 2026-07-16` 中全市场 diffusion score 覆盖率已经接近/达到 `100%`，但直接替换 selector score 仍显著弱于主力纯规则。

诊断方法：

```text
baseline run: 20260717_125751_wts_fast_exit_d18_p2_t8_2024_2026_latest_baseline
ML top4 run: 20260717_125753_wts_fast_exit_d18_p2_t8_market_score_top4_pos5_2024_2026_latest
ML top6 run: 20260717_125754_wts_fast_exit_d18_p2_t8_market_score_top6_pos6_2024_2026_latest
ML top8 run: 20260717_125754_wts_fast_exit_d18_p2_t8_market_score_top8_pos8_2024_2026_latest

compare:
  1. baseline closed positions kept/missed by ML selection
  2. ML closed positions overlap/extras vs baseline selection
  3. model score on baseline winners/losers
  4. key NAV divergence days and active position weights
```

基准 closed position 被 ML top8 保留/错过：

```text
year  baseline all                 kept by ML top8              missed by ML top8
2024  n=47 pnl=+319504 ret=+2.1%   n=44 pnl=+78330 ret=+1.3%   n=3 pnl=+241175 ret=+13.7%
2025  n=47 pnl=+89804  ret=+0.4%   n=42 pnl=+168295 ret=+0.7%  n=5 pnl=-78491 ret=-2.1%
2026  n=25 pnl=+820683 ret=+1.9%   n=25 pnl=+820683 ret=+1.9%  n=0
```

ML top8 自身 closed position：

```text
year  ML top8 all                  overlap with baseline        ML extras
2024  n=67 pnl=-297738 ret=+0.7%   n=60 pnl=-237079 ret=+0.9%  n=7 pnl=-60658 ret=-1.0%
2025  n=70 pnl=+36130  ret=+1.3%   n=56 pnl=-17292 ret=+1.2%  n=14 pnl=+53422 ret=+1.7%
2026  n=36 pnl=+228319 ret=+2.5%   n=35 pnl=+227060 ret=+2.4% n=1 pnl=+1259 ret=+5.9%
```

关键发现：

1. 2024 的失败主要不是覆盖问题，而是少数高收益 baseline 票被 diffusion 降级或低配。ML top8 错过的 3 笔 baseline closed position 净贡献约 `+24.1万`，平均收益约 `+13.7%`。
2. 2024 的 ML 额外票并没有提供足够正贡献，反而净亏约 `-6.1万`。这使账户路径在进入 2025/2026 前已经明显落后。
3. 2026 中 ML top8 已经保留了全部 baseline closed positions，但仍显著落后，核心原因是账户净值基数更低、仓位更分散、强票实际资金占比更小。
4. diffusion score 对 baseline winner/loss 的区分在 2024/2026 不稳定。2024 baseline winners 的平均模型分数反而低于 losers；典型高收益票被模型打低分。

典型交易对：

```text
2024-05-15 signal:
  baseline: SZ002970 score=0.7122, selected rank=1, 2024-05-16 买入约 71.3万，后续盈利退出
  ML:       SZ002029 ml_score=0.0691, selected rank=1, 2024-05-16 买入约 71.0万，后续止损
  SZ002970 在 ML 中仅 rank=4，因资金/持仓路径没有实际买入

2024-12-23 signal:
  baseline: SH600673 rule score=0.8366, rank=1, 2024-12-24 买入约 92.1万，后续止盈
  ML top4:  未持有 SH600673
  ML top8:  仅小仓位买入，后续虽然止盈但贡献很小

2026-02-04 signal:
  SZ000539 被 baseline 与 ML 都选中
  baseline 买入约 115.7万，ML top8 买入约 17.0万
  2026-03-19 止盈时，baseline 对 NAV 贡献远大于 ML
```

结论修正：

1. 当前 diffusion score 不能作为主力弱转强的直接 selector 替换项。它在 2025 短窗有效，但在 2024/2026 会破坏原始规则的少数大盈利票和仓位集中度。
2. 下一轮不应继续跑更大的 `topk/max_positions` 直接替换。扩大 topk 只会进一步分散资金，除非模型能稳定识别强票。
3. 更合理的下一轮实验是“baseline 保底 + ML 高置信补位/覆盖”：保留纯规则优先级，至少不降低原始 top1/top4；ML 只用于额外容量、同分/近邻候选重排，或在分数显著高于当日候选分位时上调。
4. 如果要训练层面继续迭代，应改成主力弱转强候选/近邻池内 rank 任务，而不是 `score_rank_top100` 训练后用全市场 neutral feature 外推。

下一轮最小实验假设：

```text
H1: baseline top4/pos5 不被 diffusion 降级时，可以保住 2024/2026 大盈利票。
H2: 在保留 baseline 的基础上，只用 diffusion 给 top6/pos6 或 top8/pos8 的新增候选排序，可能提升容量而不破坏收益。
H3: 如果该融合仍不能提升，说明当前 score 只能做研究信号，必须重训主力弱转强候选池 rank/head。
```

### 保底融合 overlay v1 回测

假设：直接用 diffusion 替换 selector score 会破坏弱转强纯规则 top4 中的少数大盈利票；如果先保留 baseline 排名，再只让 ML 影响低优先级候选或额外容量，应能显著修复 2024/2026 长窗口失效。

实现：不改 QuantX 核心，通过离线生成 `external_score` parquet 完成 fused score。

```text
score dir:
  score_rank_top100_full_v1/runs/wts-flow-wide08-ae44-20260717-105330/evaluation/fusion_overlay_v1

generated configs:
  configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_fusion_lock1_ml_rest_top4_pos5_2024_2026_latest.yaml
  configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_fusion_lock4_ml_extra_top6_pos6_2024_2026_latest.yaml
  configs/strategies/generated/weak_to_strong_diffusion_replay/wts_fast_exit_d18_p2_t8_fusion_lock4_ml_p75_extra_top6_pos6_2024_2026_latest.yaml

dry-run: all ok
window: 2024-01-02 .. 2026-07-16
```

融合规则：

```text
lock1_ml_rest_top4_pos5:
  纯规则 rank1 的 fused_score 设为最高，剩余候选用 ML score 排序，topk=4, max_positions=5。

lock4_ml_extra_top6_pos6:
  纯规则 rank1..4 的 fused_score 设为最高，rank>4 的候选用 ML score 排序，topk=6, max_positions=6。

lock4_ml_p75_extra_top6_pos6:
  纯规则 rank1..4 保底，rank>4 只有 ML score >= 当日 raw candidate p75 才进入外部分数文件，topk=6, max_positions=6。
```

分数文件覆盖：

```text
variant                  rows  days  locked  extra  missing_ml_extra
lock1_ml_rest_top4_pos5  902   298   298     604    18
lock4_ml_extra_top6_pos6 914   298   630     284    6
lock4_ml_p75_extra       726   298   630     96     6
```

正式回测结果：

```text
metric          pure rule   direct ML top4  direct ML top8  lock1 top4  lock4 top6  lock4 p75 top6
total_return    +127.5249%  -15.4364%      -1.3730%        +79.1000%   +119.1005% +119.1005%
annual_return   +38.2705%   -6.3952%       -0.5435%        +25.8237%   +36.2294%  +36.2294%
max_drawdown    -25.1662%   -46.0542%      -43.7809%       -29.8122%   -25.2945%  -25.2945%
sharpe          1.1889      -0.2004        -0.0181         0.8746      1.1151     1.1151
trades          271         255            383             269         310        310
win_rate        55.6291%    53.9568%       54.5455%        58.2781%    53.8012%   53.8012%
profit_factor   1.8509      0.8348         0.9683          1.6067      1.7897     1.7897
```

分段收益：

```text
period          pure rule   lock1 top4   lock4 top6
2024            +25.3677%   +3.1311%    +23.8222%
2025 calendar   -7.8151%    -12.0475%   -7.7907%
2025 OOS slice  +4.8796%    -0.1374%    +4.9107%
2026 YTD        +89.8566%   +96.9745%   +84.4710%
```

候选重合：

```text
variant          selected_total  overlap_with_base  variant_overlap  base_covered
pure rule        630             630                100.00%          100.00%
lock1 top4       619             526                84.98%           83.49%
lock4 top6       719             618                85.95%           98.10%
lock4 p75 top6   702             618                88.03%           98.10%
```

归因：

1. `lock1` 不够。只保留纯规则第 1 名会丢掉太多 top2-top4 弱转强票，2024 收益从纯规则 `+25.37%` 降到 `+3.13%`。
2. `lock4` 明显修复直接替换排序的崩坏：直接 ML top8 是 `-1.37%`，`lock4 top6` 回到 `+119.10%`，回撤也回到接近纯规则。
3. 但 `lock4 top6` 仍没有超过纯规则 `+127.52%`。它保住了所有 baseline closed positions，但额外容量和不同账户路径带来的收益不足以覆盖仓位稀释和路径扰动。
4. `lock4_p75` 与 `lock4_all` 结果完全一致，说明真正进入交易路径的补位候选本来就落在高 ML 分位集合内；简单 p75 gate 没有产生额外区分。

当前判断：

```text
直接替换 selector score: 否定。
baseline top1 保底 + ML 排其余: 否定。
baseline top4 保底 + ML 补位: 可作为工程安全形态，但当前 diffusion score 没有产生可证明的扩容 alpha。
```

下一轮方向：

1. 推理分布对齐：当前全市场推理只能用 neutral candidate features，而训练时有 `wts_rank/wts_score_z/wts_score_pct`。应新增候选级推理，让主力弱转强 raw candidates 带真实 rule rank/score 条件入模，再比较当前 neutral score。
2. 训练目标对齐：如果候选级真实 feature 推理仍无增益，应重建数据集为 `fast_exit_d18_p2_t8` 主力候选/近邻池，训练 rank/head，而不是依赖 `score_rank_top100` 对全市场外推。
3. 策略形态上，短期只考虑 `lock4` 保底型接入；任何不保 baseline top4 的 ML 排序都不应再作为生产候选。

### Event-grid 短线兑现实验 v1

问题修正：

event-grid 里 `q90_3d` 的高 lift 主要对应“未来短期路径冲高/右尾触达”，之前直接接入原弱转强 18-25 日卖出规则是 horizon 错配。正确验证方式应当让交易规则跟预测事件一致：冲高止盈、低点止损、短期未兑现则退出。

当前 QuantX YAML 能直接表达：

```text
take_profit: high / avg_cost - 1 >= threshold
stop_loss:   low / avg_cost - 1 <= -threshold
time_stop:   holding_days >= N
deal_price:  close
```

限制：这不是严格的“盘中触达阈值价成交”。当前引擎的成交价仍是 `close`，所以本轮是 `high/low` 触发、`close` 成交的近似组合回测。真正阈值价成交需要给执行器增加 rule-level fill price 或 order price override。

生成配置与结果：

```text
grid config dir:
  quantx/configs/strategies/generated/weak_to_strong_diffusion_short_exit_grid
  36 configs = {baseline_rule, q90_direct, q90_lock4} x TP{8%,10%} x SL{5%,8%} x H{3,5,7}
  results: short_exit_backtest_results.csv

focus config dir:
  quantx/configs/strategies/generated/weak_to_strong_diffusion_short_exit_focus_grid
  24 configs = {baseline_rule, q90_lock4} x TP{8%,10%,12%} x SL{8%} x H{7,9,11,15}
  results: short_exit_focus_backtest_results.csv

dry-run:
  all ok
window:
  2024-01-02 .. 2026-07-16
```

第一轮短线网格最佳：

```text
family         TP   SL   H   total_return  max_drawdown  sharpe  trade_count  avg_hold
q90_lock4      10%  8%   7   +122.1711%    -17.9498%     1.1466  659          8.70
baseline_rule  8%   8%   7   +116.2517%    -18.9311%     1.1931  609          8.23
q90_direct     10%  8%   3   +81.6351%     -22.8962%     0.8904  793          4.42
```

对比旧结果：

```text
original pure rule baseline: +127.5249%, MDD -25.1662%, Sharpe 1.1889
old q90 lock4 overlay:       +119.1005%, MDD -25.2945%, Sharpe 1.1151
short q90 lock4 TP10/SL8/H7: +122.1711%, MDD -17.9498%, Sharpe 1.1466
```

解释：短线退出确实让 q90 lock4 比旧 lock4 有小幅改善，并明显降低回撤，但还没有超过原始纯规则 baseline。

焦点扩展后的整体最佳：

```text
family         TP   SL   H   total_return  annual_return  max_drawdown  sharpe  trade_count  yearly
baseline_rule  12%  8%   7   +162.8521%    +46.3655%      -16.5420%     1.4111  573          2024 +27.85%, 2025 +48.45%, 2026 +33.71%
q90_lock4      12%  8%   7   +154.0233%    +44.4113%      -18.3717%     1.3646  629          2024 +32.35%, 2025 +44.26%, 2026 +28.45%
q90_lock4      12%  8%   9   +137.1372%    +40.5381%      -22.6513%     1.3094  563          2024 +28.64%, 2025 +23.52%, 2026 +50.73%
```

同参数增益：

```text
q90_lock4 TP12/SL8/H15 vs baseline same params: +37.90pp, MDD +4.50pp
q90_lock4 TP10/SL8/H7  vs baseline same params: +17.56pp, MDD +8.08pp
q90_lock4 TP12/SL8/H9  vs baseline same params: +16.18pp, MDD +3.27pp
```

但是绝对最优仍是纯规则短线退出 `TP12/SL8/H7`，不是 diffusion overlay。

当前判断：

```text
1. “高准确率事件要配套短线兑现”这个方向是成立的。
   纯规则从原始 +127.52% 提升到 TP12/SL8/H7 的 +162.85%，说明弱转强本身存在很强的冲高兑现 alpha。

2. 当前 q90_3d diffusion score 在 lock4 保底形态下有局部帮助。
   在相同短线退出参数下，q90_lock4 多数组合优于 baseline_same_param，且常降低回撤。

3. 但当前 diffusion score 还没有证明能提升最终最优策略。
   最优 q90_lock4 +154.02% 低于最优纯规则短线退出 +162.85%，说明主要收益来自 exit rule，而不是模型排序 alpha。

4. direct 替换仍然不应继续。
   q90_direct 最优只有 +81.64%，仍远弱于纯规则和 lock4。
```

下一步建议：

```text
A. 先把纯规则短线退出 TP12/SL8/H7 作为新的强 baseline，回看 2016-2026 全窗口是否稳健。
B. 如果继续验证 diffusion，需要做真正阈值价成交版本；否则 high 触发 close 成交会混淆“冲高可兑现”与“收盘回吐”。
C. 模型方向应从 direct/neutral market score 转为候选级真实 feature 推理，至少用主力弱转强 raw candidates 的真实 rank/score 条件，而不是全市场 neutral 条件外推。
D. 如果候选级推理仍不能超过 TP12/SL8/H7，说明 diffusion 当前更适合作为事件研究工具，而不是生产 selector。
```

### TP12/SL8/H7 纯规则全窗口复核

为了确认短线兑现在 2024-2026 的强表现是否可以作为新 baseline，补跑 2016-2026 全窗口：

```text
config:
  quantx/configs/strategies/generated/weak_to_strong_diffusion_short_exit_focus_grid/wts_short_exit_focus_baseline_rule_tp12_sl08_h7_2016_2026.yaml
run_id:
  20260717_141803_wts_short_exit_focus_baseline_rule_tp12_sl08_h7_2016_2026
window:
  2016-01-04 .. 2026-07-16
result:
  total_return  +65.1458%
  annual_return +4.8761%
  max_drawdown  -56.9692%
  sharpe        0.1663
  trades        2211
```

对比已有强 baseline：

```text
stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_d18_p2_t8
total_return  +6025.65%
annual_return +47.78%
max_drawdown  -30.55%
```

结论再修正：

```text
TP12/SL8/H7 是 2024-2026 的局部强 exit，不是全周期稳健 baseline。
不能因为 2024-2026 提升就替换当前长窗口弱转强主策略。
短线冲高兑现仍值得研究，但必须按年份/市场 regime 分段评估。
```

### Week7 close-to-close balanced diffusion 全量评测

根据新的数据分布假设，目标改为未来 7 个 close-to-close 点：

```text
target:
  close(T+1..T+7) / close(T+1) - 1
trend threshold:
  t+7 相对 t+1 的最终收益 <= -5% 为 down，[-5%, +5%) 为震荡区，>= +5% 为 up
train split:
  pre-2020 all-market train_balanced
  up/down/range 各 801,377，共 2,404,131
prediction split:
  2020-01-02 .. 2026-07-06，共 6,899,925
model:
  reused AE ae_wide_08 checkpoint epoch20
  diffusion flow objective epoch17, validation loss 0.026769
```

为了加速全量评测，`evaluate_wts_diffusion_v1.py` 做了最小改动：

```text
1. 增加 deterministic slicing:
   --shard-index / --shard-count
   --row-offset / --row-limit

2. --predictions 支持 glob/多 parquet 聚合，只读 direct metrics 所需列。

3. direct metrics 增加:
   return_bucket: <-5%, -5~0%, 0~5%, >=5%
   score_top_quantile: top 10%, 5%, 2%, 1%, 0.5%, 0.1%
   yearly group metrics
   score_cutoff_total_count / score_cutoff_total_over_selected 用于识别同分 tie 假象
```

全量推理采用 6 卡并行，每卡一个连续 shard：

```text
shard files:
  predictions_prediction_epoch017_k16_full_shard0of6.parquet 1,149,987 rows
  predictions_prediction_epoch017_k16_full_shard1of6.parquet 1,149,988 rows
  predictions_prediction_epoch017_k16_full_shard2of6.parquet 1,149,987 rows
  predictions_prediction_epoch017_k16_full_shard3of6.parquet 1,149,988 rows
  predictions_prediction_epoch017_k16_full_shard4of6.parquet 1,149,987 rows
  predictions_prediction_epoch017_k16_full_shard5of6.parquet 1,149,988 rows
total:
  6,899,925 rows
metrics:
  metrics_prediction_epoch017_k16_full_merged_direct_v2.csv/json
```

整体 direct metrics：

```text
abs_7d_direction_accuracy  52.0195%
abs_trend_accuracy         51.8285%
gt_up_rate                 18.7516%
gt_down_rate               18.6610%
gt_range_rate              62.5874%
```

7 日四桶结果：

```text
bucket        base_rate  pred_rate  precision  recall   lift  selected_avg_gt
<-5%          18.66%     23.38%     23.22%     29.09%   1.24  -0.11%
-5%~0         32.33%     36.66%     33.33%     37.79%   1.03  +0.15%
0~5%          30.25%     31.31%     31.18%     32.27%   1.03  +0.62%
>=5%          18.75%      8.66%     26.36%     12.17%   1.41  +1.56%
```

结论：直接按模型预测桶交易仍然弱。`>=5%` precision 只有 26.36%，recall 只有 12.17%，相当于错误率仍然超过 70%，不能算强预测信号。

连续 score 的极端分位比四桶预测更有信息量：

```text
score                    q       selected  avg_gt  median_gt  win_rate  up5_rate  down5_rate
pred_abs_q10_3d          0.1%      6,900   +3.24%    +2.61%    66.58%    35.90%    10.52%
pred_abs_q10_5d          0.5%     34,500   +3.34%    +2.09%    65.20%    32.05%    11.28%
pred_abs_q10_7d          0.5%     34,500   +3.60%    +2.06%    62.31%    32.79%    13.47%
score_mean_plus_q10_7d   0.5%     34,500   +3.20%    +1.62%    58.49%    32.69%    17.42%
pred_abs_7d              1.0%     69,000   +2.46%    +1.06%    55.19%    31.29%    20.47%
```

`score_prob_pos_7d` top1% 表面上更强：

```text
score_prob_pos_7d top1%:
  avg_gt    +4.86%
  up5_rate   35.64%
  down5_rate  7.78%
```

但这个结果不能直接采信，因为 `score_prob_pos_7d` 来自 `k_samples=16`，只有 17 个离散值。全量中 `score_prob_pos_7d == 1.0` 的样本有 658,970 行，而 top1% 只选 69,000 行：

```text
score_cutoff_total_count / selected_count = 9.55
```

也就是说 top1% 是从大量同分样本中任意切出来的。更合理的阈值口径是 `score_prob_pos_7d >= 1.0`：

```text
n          658,970
avg_gt      +1.59%
median_gt   +0.65%
win_rate    54.29%
up5_rate    24.22%
down5_rate  14.26%
```

因此 `score_prob_pos` 暂时只能作为辅助，不作为强结论。

按年份看，连续 score 的高分位也不够稳定：

```text
pred_abs_q10_5d top0.5% within year:
2020 +2.34%, 2021 +1.26%, 2022 +3.38%, 2023 +0.23%, 2024 +5.11%, 2025 +2.63%, 2026 -0.01%

pred_abs_q10_7d top0.5% within year:
2020 +2.76%, 2021 +0.89%, 2022 +4.18%, 2023 +0.22%, 2024 +4.59%, 2025 +1.74%, 2026 +0.78%
```

当前判断：

```text
1. 平衡 up/down/range 后，模型确实能在极端高分位上找到一些右尾样本。
2. 但直接预测四桶的 precision/recall 不够强，不能作为最终选股规则。
3. 连续 score 的 top 0.5%~1% 有 alpha，但年稳定性不足，2023 和 2026 明显弱。
4. `score_prob_pos` 因为采样数只有 16，tie 太严重，不能直接做 top quantile 结论。
5. 当前模型形态更像“弱右尾排序器”，还不是强预测模型。
```

下一轮假设：

```text
A. 训练目标要从 path MSE/flow 回归，转向 tail/topK 直接优化：
   - 4 分类 cross entropy: <-5%, -5~0%, 0~5%, >=5%
   - right-tail binary: y_5d/y_7d >= 5%
   - pairwise/listwise ranking loss，直接优化 top 分位排序

B. 评测要固定为:
   - 四桶 precision/recall/lift
   - score top quantile 表
   - 年份/市场 regime 稳定性
   - tie diagnostics

C. 如果继续 diffusion，应增加 k_samples 或避免用 prob_pos 的离散概率做主 score。

D. 交易层不应使用全局阈值，应该转为每日 cross-sectional topK / weak-to-strong candidates 内排序，再接 QuantX 回测。
```

## 2026-07-17 pure all-market ML score backtest

目的：验证 `pred_abs_q10_5d` 是否可以不经过弱转强候选池，直接对全市场股票打分后做 daily topK / fixed threshold 交易。

Score 文件：

```text
${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/market_all_week7_c2c_balanced_v1/runs/market-all-week7-c2c-balanced-ae08-abs-20260717-163535/evaluation/strategy_scores/week7_c2c_pred_abs_q10_5d_full_scores.parquet

rows: 6,899,925
dates: 2020-01-02 ~ 2026-07-06
instruments: 5,173
score_col: fused_score = pred_abs_q10_5d
```

回测设置：

```text
QuantX universe: all_mainboard
backtest: 2024-01-02 ~ 2026-07-16
tradability filters only:
  close > 2
  turnover20 > 5,000,000
  abs(1d return) < 9.5%
  amplitude < 20%
selector:
  external_score.missing = drop
  sort = score_desc
execution:
  buy/sell price = close
  sell rule = holding_days >= 5
  no take-profit / stop-loss overlay
```

生成配置：

```text
configs/strategies/generated/weak_to_strong_diffusion_q10_5d_backtest/pure_ml_q10_5d_full_top1_pos1_h5_2024_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_diffusion_q10_5d_backtest/pure_ml_q10_5d_full_top5_pos5_h5_2024_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_diffusion_q10_5d_backtest/pure_ml_q10_5d_full_top20_pos20_h5_2024_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_diffusion_q10_5d_backtest/pure_ml_q10_5d_gate_top0050bp_top20_pos20_h5_2024_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_diffusion_q10_5d_backtest/pure_ml_q10_5d_gate_top0100bp_top20_pos20_h5_2024_2026_mainboard.yaml
```

Dry-run 结果：5 个配置全部通过，主板 universe 解析为 3,189 只股票。

完整回测结果：

```text
case                   run_id                                                            total_return  annual_return  max_drawdown  sharpe   closed_trades  win_rate  profit_factor
full daily top1/pos1   20260717_182223_pure_ml_q10_5d_full_top1_pos1_h5_2024_2026_mainboard   +200.28%       +54.25%       -46.20%    0.881      119          52.10%      1.452
full daily top5/pos5   20260717_182239_pure_ml_q10_5d_full_top5_pos5_h5_2024_2026_mainboard    +22.51%        +8.33%       -34.17%    0.230      598          43.31%      1.063
full daily top20/pos20 20260717_182301_pure_ml_q10_5d_full_top20_pos20_h5_2024_2026_mainboard   +2.55%        +1.00%       -31.63%    0.033    2,392          45.94%      1.008
gate top0.5% top20     20260717_182312_pure_ml_q10_5d_gate_top0050bp_top20_pos20_h5_2024_2026_mainboard -32.68%       -14.44%       -53.10%   -0.359      875          47.09%      0.852
gate top1.0% top20     20260717_182325_pure_ml_q10_5d_gate_top0100bp_top20_pos20_h5_2024_2026_mainboard  -1.39%        -0.55%       -43.62%   -0.014    1,149          45.87%      0.995
```

按年份拆分：

```text
case                   2024       2025       2026 YTD
full daily top1/pos1    -9.62%    +41.44%    +134.92%
full daily top5/pos5    -4.24%     +7.92%     +18.54%
full daily top20/pos20  -8.29%    +10.39%      +1.29%
gate top0.5% top20     -29.84%    -10.32%      +6.99%
gate top1.0% top20      -6.29%     -2.94%      +8.42%
```

单笔分布：

```text
case                   mean_return  median_return  q10       q90       min       max
full daily top1/pos1     +1.38%        +0.41%      -8.59%    +13.20%   -22.93%   +29.96%
full daily top5/pos5     +0.28%        -0.97%      -8.44%    +10.77%   -22.93%   +49.05%
full daily top20/pos20   +0.19%        -0.52%      -8.42%     +9.50%   -31.84%   +46.55%
gate top0.5% top20       +0.27%        -0.49%      -9.71%    +12.31%   -22.93%   +39.04%
gate top1.0% top20       +0.24%        -0.62%      -9.12%    +11.01%   -24.75%   +50.07%
```

Sharpe / 风险画像：

```text
focus run:
  20260717_182223_pure_ml_q10_5d_full_top1_pos1_h5_2024_2026_mainboard

total_return:       +200.28%
annual_return:       +54.25%
max_drawdown:        -46.20%
QuantX sharpe:         0.881

daily_return_mean:   +0.254%
daily_return_std:     3.880%
annualized_vol:      61.59%
daily_return_median: -0.059%
daily_q05:           -5.20%
daily_q95:           +9.72%
worst_daily:         -9.81%
best_daily:          +9.95%

trade_win_rate:      52.10%
trade_mean:          +1.38%
trade_median:        +0.41%
trade_q10:           -8.59%
trade_q90:          +13.20%
trade_min:          -22.93%
trade_max:          +29.96%
```

为什么 Sharpe 不高：

```text
1. 这是单票满仓 top1 策略，组合波动接近个股波动；日波动约 3.88%，年化波动约 61.59%，会显著压低 Sharpe。
2. 收益路径不平滑，日收益中位数为 -0.059%，总收益主要靠少数右尾交易贡献，而不是每天稳定小幅赚钱。
3. 左尾没有被控制，单笔 q10 为 -8.59%，最差单笔 -22.93%，最大回撤 -46.20%。
4. regime 依赖明显：2024 年亏损，2025 年正常盈利，2026 年大幅贡献收益。Sharpe 看完整收益路径，不只看最后总收益。
```

提高 Sharpe 的方向：

```text
1. 不再只追求 total_return，优先降低左尾和波动。
2. 测 top1/top3 rank-weight，例如 70/20/10，尝试保留极端头部 alpha，同时降低单票满仓波动。
3. 加短线止损、冲高止盈、3/5/7 日持有期网格，重点减少 -8% 到 -20% 的左尾交易。
4. 加 regime / confidence filter，只在模型分数足够极端、rank margin 足够大、市场状态更匹配时交易。
5. 评估行业、市值、流动性、波动率切片，找到模型 preference 最稳定的子空间。
```

观察：

```text
1. 纯全市场 daily top1 能跑出很高收益，但高度集中在 2026 年；2024 年为负，最大回撤达到 -46.20%，暂时不能当成稳定策略。
2. 扩容到 top5/top20 后收益迅速衰减：top5 只剩 +22.51%，top20 近似打平。这说明可交易 alpha 很可能只在极端头部，容量有限。
3. 2020-2023 校准出来的固定阈值 gate 在 2024-2026 不稳：top0.5% gate 亏 -32.68%，top1.0% gate 亏 -1.39%。全局阈值受年份/regime 漂移影响很大。
4. top1 的买入并非由单一股票反复贡献；买入次数最多的单票也只有 2 次。但收益主要来自少数右尾交易和 2026 年强 regime。
5. 相比弱转强 baseline 2024-2026 的 +118.43%，纯 ML top1 的收益更高但回撤更大、容量更低；candidate 内替换排序只有 +14.58%，说明模型直接覆盖弱转强候选排序不是正确用法。
```

当前结论：

```text
pred_abs_q10_5d 不是没有 alpha；它在全市场 cross-sectional 极端 top1 附近有明显右尾能力。
但这个信号的容量和稳定性不足：topK 一放大就衰减，固定阈值 gate 失败。
下一步更合理的交易研究方向不是“全局阈值买所有高分股”，而是：
  A. 用 daily rank/topK，而不是跨年份 fixed threshold；
  B. 在 top1/top3/top5 上加短线止盈/止损/冲高退出；
  C. 按年份、市场状态、行业/市值/流动性切片，找 top1 有效的 regime；
  D. 训练目标转向 right-tail ranking/listwise，使 top5/top20 的容量能保住。
```

## 2026-07-17 DiT 扩大模型中间评估

正式训练配置：

```text
run:
  market_all_week7_c2c_balanced_v1/runs/market-all-week7-c2c-balanced-ae08-dit1024-l12-abs-bs768-20260717-192315

model:
  AE 复用 ae_wide_08 checkpoint，约 40.4M 参数
  diffusion head 改为 cross_dit，d_model=1024, layers=12, heads=16，约 382.5M 参数
  总参数约 422.9M

training:
  6 卡 DDP，batch_size=768/GPU
  diffusion 计划 40 epoch
  每个 epoch 保存 checkpoint
```

当前训练状态：

```text
训练已进入 epoch20，epoch19 checkpoint 已落盘。
epoch19 train_loss = 0.01698
epoch19 validation_loss = 0.02784
吞吐约 8100 samples/s
```

基于已完成的 epoch011 5w 快评，先筛出的可回测 score 结构：

```text
candidate             score                    selection      hold  selected_avg_abs  selected_avg_rel  lift_event             lift
pure_ml_return_7d     score_prob_rel_ge_050_7d daily_top5pct  7d       +0.658%          +0.272%          final_abs_ge_100_7d   1.13
event_q90_5d          score_q90_5d             daily_top5pct  5d       +0.448%          +0.176%          final_rel_ge_200_5d   1.29
event_prob150_5d      score_prob_rel_ge_150_5d daily_top5pct  5d       +0.419%          +0.147%          final_rel_ge_080_5d   1.23
event_prob150_7d      score_prob_rel_ge_150_7d daily_top5pct  7d       +0.357%          -0.030%          final_abs_ge_200_7d   1.21
```

解读：

```text
1. 7d 的 score_prob_rel_ge_050_7d 是当前 5w 样本里平均收益最好的纯 ML daily rank 信号，适合先做 all_mainboard top5/top10/top20、hold 7d。
2. 5d 的 score_q90_5d / diff_q90_5d 是 lift 最高的事件型信号，但 final_rel>=20% 的 precision 只有约 1.12%，更适合加 TP/SL 或冲高卖出，而不是只看命中率。
3. score_prob_rel_ge_150_5d 的收益和 lift 都低于 q90_5d，但事件更宽，适合作为第二优先级。
4. score_prob_rel_ge_150_7d 虽有 lift，但相对收益为负，暂不作为第一批回测重点。
```

已生成的中间产物：

```text
strategy_candidates_formal_epoch011_k4s8_max50k.csv
  ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/market_all_week7_c2c_balanced_v1/runs/ddp-smoke-dit1024-bs768-20260717-191558/evaluation/strategy_candidates_formal_epoch011_k4s8_max50k.csv

QuantX external_score smoke config:
  ${HOME}/git/quantization/quantx/configs/strategies/generated/weak_to_strong_diffusion_dit_smoke/pure_ml_dit_smoke_score_prob050_7d_top2_h3_202401.yaml

smoke dry-run:
  passed
```

后台任务：

```text
wait_eval_epoch020_k4s8_max50k_20260717.log
  等 diffusion_epoch_020.pt 落盘
  等 GPU 6 或 7 无 compute app
  自动跑 prediction split 5w eval: k_samples=4, inference_steps=8
  自动生成 event_grid_formal_epoch020_k4s8_max50k
```

下一步：

```text
1. 对比 epoch011 和 epoch020/latest 的 5w event grid，确认上述 score 结构是否增强。
2. 若增强，生成 2024-01-02 到 2026-07-16 的 all_mainboard market score parquet。
3. 第一批 QuantX 回测：
   A. score_prob_rel_ge_050_7d，top5/top10/top20，hold 7d；
   B. score_q90_5d，top5/top10/top20，hold 5d；
   C. score_q90_5d + TP8/10、SL5/8、hold 5d/7d；
   D. score_prob_rel_ge_150_5d，top5/top10/top20，hold 5d。
```

## 2026-07-17 事件指标到交易收益的映射关系

背景：本轮 DiT 裸模型回测不使用弱转强候选池，而是在 `all_mainboard` 上做宽可交易过滤后，直接按 diffusion external score 排序买入。回测窗口为 `2021-01-04` 到 `2026-07-16`。为快速验证，score parquet 使用 date stride：7d score 每 7 个交易日打分一次，5d score 每 5 个交易日打分一次。

已验证回测摘要：

```text
score_mean_minus_0.5std_7d, epoch024:
  top1  total_return +194.1%, max_drawdown -57.0%, sharpe 0.49, win_rate 51.9%, profit_factor 1.22, avg_trade_return +0.86%
  top10 total_return  +63.3%, max_drawdown -38.0%, sharpe 0.37, win_rate 48.5%, profit_factor 1.09, avg_trade_return +0.38%
  top20 total_return  +57.2%, max_drawdown -30.8%, sharpe 0.36, win_rate 48.7%, profit_factor 1.08, avg_trade_return +0.34%

score_prob_rel_ge_080_5d, epoch022:
  top1  total_return -66.3%, max_drawdown -75.3%, sharpe -0.43, win_rate 43.0%, profit_factor 0.86, avg_trade_return -0.23%
  top10 total_return -15.5%, max_drawdown -42.4%, sharpe -0.13, win_rate 47.0%, profit_factor 0.96, avg_trade_return -0.01%
  top20 total_return  +1.6%, max_drawdown -35.5%, sharpe  0.01, win_rate 47.1%, profit_factor 1.00, avg_trade_return +0.05%
```

关键解释：event-grid 里的 `precision` 不是交易胜率，而是某个严格事件的命中率。例如：

```text
epoch024 / score_mean_minus_0.5std_7d
event: max_rel_ge_150_7d
含义: 未来 7 天内相对基准曾冲到 +15% 以上
base_rate: 4.16%
precision: 5.81%
recall: 8.99%
lift: 1.40
selected_avg_rel: +0.293%
```

`precision = 5.81%` 的含义不是“只有 5.81% 的交易赚钱”，而是“模型选中样本中有 5.81% 命中了未来 7 天内冲高 +15% 的稀有事件”。剩余 `94.19%` 样本并不全是亏损，里面还包含 `+1%/+3%/+8%` 的小中幅上涨、震荡和小亏，只是没有达到 `+15%` 这个事件阈值。

因此是否赚钱主要看期望收益，而不是单独看 event precision。粗略 EV 拆解：

```text
selected_count = 3216
random_hit_count ~= 3216 * 4.16% = 134
model_hit_count  = 187
extra_hit_count  ~= 53

若额外命中的 +15% 事件平均至少贡献 15%:
extra_EV ~= 53 * 15% / 3216 = +0.25%

这与 event-grid 的 selected_avg_rel +0.293% 量级一致。
```

结论：低 precision 仍然可能赚钱的条件是：

```text
1. 事件本身足够稀有，base_rate 很低。
2. 模型对事件有稳定 lift，即命中率相对随机显著提升。
3. 事件 payoff 足够厚，例如 +15% 级别右尾。
4. 未命中事件的样本不是系统性大亏，而是多数为小涨、小亏或震荡。
5. selected_avg_rel / avg_trade_return 为正，说明右尾收益足以覆盖未命中样本和交易成本。
```

反例：`score_prob_rel_ge_080_5d` 也曾在 event-grid 中显示较高 lift：

```text
epoch022 / score_prob_rel_ge_080_5d
event: final_rel_ge_150_5d
precision: 2.58%
recall: 9.30%
lift: 1.45
selected_avg_rel: +0.215%
```

但 5 年固定持有回测亏损，说明高 lift 不能单独作为交易依据。失败原因：

```text
1. precision 绝对值太低，97% 以上样本没有命中 +15% 终点事件。
2. 该事件是 final_rel_ge_150_5d，要求最终收盘仍保持强涨；它与实际交易退出方式不完全一致。
3. top1/top2/top3 的真实成交胜率只有 43%-45%，profit_factor < 1。
4. 尾部亏损没有被控制，5d top1 最大回撤 -75.3%。
5. 扩容到 top20 才接近打平，说明 score 的头部排序方向对交易不稳定。
```

本轮形成的指标到收益映射规则：

```text
不充分指标:
  1. 全局 direction accuracy: 约 49%，接近随机，不能解释交易收益。
  2. 单个 event precision: 只说明某个事件的命中率，不等同交易胜率。
  3. 单个 event recall: 扩大覆盖会提高 recall，但可能稀释收益。
  4. 单个 lift: 稀有事件 base_rate 很低时 lift 容易好看，但不保证 EV 为正。

更有交易解释力的组合指标:
  1. selected_avg_rel 是否显著大于 all_avg_rel。
  2. selected_avg_abs / selected_avg_rel 是否能覆盖成本、滑点和执行滞后。
  3. lift 是否出现在 payoff 足够厚的右尾事件上。
  4. selected_down_5p_rate / selected_avg_min_abs 是否同步恶化。
  5. top rank bucket 的真实成交均值是否单调或至少存在稳定有效 bucket。
  6. 回测 avg_trade_return、profit_factor、max_drawdown 是否与 event-grid 的 EV 方向一致。
```

rank bucket 反事实：

```text
score_mean_minus_0.5std_7d 的 top10 回测中:
  rank1   avg_trade_return +0.867%, win_rate 51.6%
  rank2-3 avg_trade_return -0.071%, win_rate 45.7%
  rank4-5 avg_trade_return +0.389%, win_rate 48.0%
  rank6-10 avg_trade_return +0.461%, win_rate 49.2%

这说明 score 并非严格单调排序。top1 有强 alpha，但 rank2-3 会拉低；top10 重新变好，是因为 rank5、rank8-10 又有正贡献。
```

交易研究含义：

```text
1. 对 7d mean-0.5std score，不应只追求 top1 高收益；top1 回撤过大，top10/top20 更适合作为容量化基底。
2. `max_rel_ge_150_7d` 是冲高事件，固定持有到期 close 会错过部分盘中/区间右尾；后续应测试 TP/SL 或冲高退出。
3. 对 5d prob080，不建议继续作为正向买入排序 score；若继续研究，应先检查是否需要反向、换退出规则或改用其他 5d score。
4. 后续模型评估不能只报 precision/recall，应同时报 base_rate、lift、selected_avg_rel、selected_down_rate、rank bucket return 和简单回测 EV。
```

## 2026-07-18 Close-token decoder epoch020: 5d/7d/10d 全市场横截面策略测算

背景：本轮从 diffusion/flow 路线切换到 Kronos-style close path tokenizer + causal decoder。训练集为 `2010-2018`，validation 为 `2019`，prediction split 为 `2020-01-02` 到 `2026-06-02`。本节只评估已经训练完成的 `decoder_epoch_020.pt` 在 5d / 7d / 10d 短线持有口径上的策略价值。

模型与产物：

```text
run:
  market_all_pre2020_v1/runs/market-all-pre2020-close-dctbpe-v1536-decoder50m-bs768-8gpu-20ep-20260718-015930

checkpoint:
  checkpoints/decoder_epoch_020.pt

model:
  AE params:      40,427,736
  decoder params: 49,751,232
  total params:   90,178,968
  tokenizer vocab final: 1536
  decoder: d_model=576, layers=12, heads=8, max_token_len=64

evaluation outputs:
  evaluation/decoder_epoch_020_prediction_50k_stratified_summary.json
  evaluation/decoder_epoch_020_prediction_50k_stratified.parquet
  evaluation/strategy_5d7d10d_epoch020/period_returns.csv
  evaluation/strategy_5d7d10d_epoch020/strategy_summary.csv
```

重要口径修正：

```text
1. 首次 50k eval 使用 prediction.parquet head(50000)，只覆盖 2020-01-02 到 2020-01-23，样本有明显时间偏差；该结果不作为策略结论。
2. 后续重新做了跨 prediction split row group 的 50k 分层抽样，覆盖 2020-01-02 到 2026-06-02，用于判断模型泛化。
3. 真正策略测算没有用 50k 随机样本，而是按 5d/7d/10d 换仓节奏抽取完整 signal_date 横截面。
```

50k 分层 OOS 评估结论：

```text
sample:
  rows: 50,000
  start: 2020-01-02
  end:   2026-06-02
  year counts: 2020=5649, 2021=6319, 2022=7224, 2023=7725, 2024=8040, 2025=8168, 2026=6875

global path metrics:
  ADE: 0.1269
  FDE: 0.2335
  direction accuracy:
    3d 49.67%, 5d 49.92%, 7d 50.05%, 10d 50.02%, 15d 50.27%, 20d 49.73%, 30d 48.89%
```

解释：epoch020 不是整体路径预测很准的模型；方向准确率接近随机，相关性也接近 0。它的可交易信息主要集中在极端高分桶，而不是全样本轨迹拟合。

50k 分层样本 top bucket 观察：

```text
best positive lift:
  7d  top0.1%: selected +5.74%, base +0.49%, lift +5.24%, win 58.0% vs 48.3%
  10d top0.5%: selected +3.76%, base +0.63%, lift +3.13%, win 56.8% vs 48.4%
  10d top0.1%: selected +3.11%, base +0.63%, lift +2.48%, win 52.0% vs 48.4%
  7d  top0.5%: selected +2.87%, base +0.49%, lift +2.38%, win 49.6% vs 48.3%
  5d  top0.1%: selected +2.74%, base +0.36%, lift +2.37%, win 56.0% vs 48.1%

negative / weak long horizon:
  15d top0.5%: selected -1.07%, base +0.90%
  20d top0.5%: selected -0.39%, base +1.17%
  30d top0.5%: selected +0.54%, base +1.75%
```

因此短线策略只重点看 5d / 7d / 10d，15d+ 暂不继续作为买入 score。

### 策略测算口径

本节是 label 级快速策略测算，不是 QuantX 撮合回测：

```text
universe:
  prediction split 全市场样本，每个 signal_date 保留完整横截面

schedule:
  5d:  从 2020-01-02 开始，每 5 个交易日换仓一次，311 个 period
  7d:  从 2020-01-02 开始，每 7 个交易日换仓一次，222 个 period
  10d: 从 2020-01-02 开始，每 10 个交易日换仓一次，156 个 period

selection:
  每个 signal_date 按 pred_close_ret_{h}d 降序
  测 topK: 1/3/5/10/20/50
  测 top pct: 0.1%/0.5%/1%/2%/5%/10%

return:
  period_return = selected stocks 的 true_close_ret_{h}d 等权均值
  base_return = 当天全横截面的 true_close_ret_{h}d 等权均值
  total_return = period_return 复利

execution/cost:
  不含手续费、滑点、涨跌停可成交约束、资金容量冲击
  标签口径为 T+1 open 到未来 close path，适合作为策略前筛，不等价于最终实盘回测

compute:
  8 张 RTX 5090 并行，按 signal_date 切分
  union dates: 488
  inference rows: 2,132,631
  batch_size: 4096/GPU
  observed memory: ~10GB/GPU
  下次应进一步放大 batch size，尽量吃满 32GB 显存
```

### 总体结果

5d 结果：

```text
best: top10.0pct
  periods: 311
  avg_selected_n: 436.7
  avg_period_return: +0.339%
  base_avg_period_return: +0.344%
  total_return: +114.63%
  base_total_return: +138.08%
  annual_return: +13.18%
  max_drawdown: -44.83%
  sharpe: 0.55
  profit_factor: 1.24
  period_win_rate: 51.77%
```

结论：5d 不是当前最优交易周期。top10% 虽然总收益为正，但没有跑赢全市场 base，topK 越窄越差，说明 5d score 的极端头部排序不稳定。

7d 结果：

```text
top0.1pct:
  periods: 222
  avg_selected_n: 3.7
  total_return: +218.01%
  base_total_return: +155.65%
  annual_return: +20.64%
  max_drawdown: -49.87%
  sharpe: 0.62
  profit_factor: 1.30

top5:
  periods: 222
  avg_selected_n: 5.0
  total_return: +200.31%
  base_total_return: +155.65%
  annual_return: +19.52%
  max_drawdown: -57.35%
  sharpe: 0.61
  profit_factor: 1.31

top10:
  periods: 222
  avg_selected_n: 10.0
  total_return: +197.70%
  base_total_return: +155.65%
  annual_return: +19.35%
  max_drawdown: -60.81%
  sharpe: 0.64
  profit_factor: 1.32

top10.0pct:
  periods: 222
  avg_selected_n: 436.5
  total_return: +144.92%
  base_total_return: +155.65%
  annual_return: +15.63%
  max_drawdown: -43.87%
  sharpe: 0.64
  profit_factor: 1.35
```

结论：7d 是高锐度小容量方向。`top0.1%/top5/top10` 能超过 base total return，但回撤偏大；`top10%` 容量大、回撤低一些，但总收益略低于 base。7d 适合作为右尾补充信号或与止盈止损结合，不宜单独生产化。

10d 结果：

```text
top10.0pct:
  periods: 156
  avg_selected_n: 435.9
  avg_period_return: +0.781%
  base_avg_period_return: +0.603%
  total_return: +167.38%
  base_total_return: +116.48%
  annual_return: +17.22%
  max_drawdown: -31.83%
  sharpe: 0.71
  profit_factor: 1.46
  period_win_rate: 55.77%

top5.0pct:
  periods: 156
  avg_selected_n: 217.7
  total_return: +150.98%
  base_total_return: +116.48%
  annual_return: +16.03%
  max_drawdown: -36.74%
  sharpe: 0.65
  profit_factor: 1.41

top2.0pct:
  periods: 156
  avg_selected_n: 86.8
  total_return: +117.96%
  base_total_return: +116.48%
  annual_return: +13.41%
  max_drawdown: -45.27%
  sharpe: 0.55
  profit_factor: 1.33
```

结论：10d 是当前最可落地的周期。`top10%` 在收益、回撤、Sharpe、PF、容量之间最平衡；`top5%` 也有正增益，但回撤更高。过窄 topK 反而大幅失败，说明当前 decoder 分数适合宽分位组合，不适合单票/少票极端下注。

### 年度拆分

说明：年度按 `signal_date` 所属年份统计，2026 仅覆盖到 `2026-06-02`，不是完整自然年。

5d representative:

| strategy | year | periods | selected_n | return | base_return | excess | period_win | stock_win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top10.0pct | 2020 | 49 | 341.3 | +40.89% | +31.69% | +9.20% | 46.9% | 48.1% |
| top10.0pct | 2021 | 49 | 385.2 | +17.31% | +28.89% | -11.58% | 61.2% | 48.1% |
| top10.0pct | 2022 | 48 | 432.8 | -4.78% | -1.10% | -3.68% | 47.9% | 44.9% |
| top10.0pct | 2023 | 48 | 469.2 | -9.06% | +1.70% | -10.76% | 43.8% | 42.6% |
| top10.0pct | 2024 | 49 | 485.5 | -12.59% | -9.12% | -3.47% | 42.9% | 44.6% |
| top10.0pct | 2025 | 48 | 485.8 | +54.61% | +55.75% | -1.14% | 68.8% | 48.6% |
| top10.0pct | 2026 | 20 | 490.1 | +10.96% | -1.47% | +12.44% | 50.0% | 48.2% |

7d small-capacity representative:

| strategy | year | periods | selected_n | return | base_return | excess | period_win | stock_win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top0.1pct | 2020 | 35 | 3.0 | +37.78% | +20.08% | +17.70% | 57.1% | 55.2% |
| top0.1pct | 2021 | 35 | 3.2 | +11.57% | +29.78% | -18.20% | 51.4% | 45.5% |
| top0.1pct | 2022 | 34 | 4.0 | +13.91% | -8.88% | +22.79% | 47.1% | 40.4% |
| top0.1pct | 2023 | 35 | 4.0 | -16.29% | +7.42% | -23.72% | 34.3% | 37.1% |
| top0.1pct | 2024 | 35 | 4.0 | -7.19% | +7.48% | -14.67% | 51.4% | 47.1% |
| top0.1pct | 2025 | 34 | 4.0 | +142.25% | +56.71% | +85.54% | 58.8% | 50.7% |
| top0.1pct | 2026 | 14 | 4.0 | -3.49% | -0.50% | -2.99% | 35.7% | 37.5% |

7d topK representative:

| strategy | year | periods | selected_n | return | base_return | excess | period_win | stock_win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 | 2020 | 35 | 10.0 | +21.37% | +20.08% | +1.29% | 51.4% | 48.3% |
| top10 | 2021 | 35 | 10.0 | +49.20% | +29.78% | +19.42% | 51.4% | 48.0% |
| top10 | 2022 | 34 | 10.0 | -4.11% | -8.88% | +4.76% | 50.0% | 43.5% |
| top10 | 2023 | 35 | 10.0 | -23.56% | +7.42% | -30.99% | 31.4% | 39.1% |
| top10 | 2024 | 35 | 10.0 | +10.32% | +7.48% | +2.85% | 54.3% | 49.1% |
| top10 | 2025 | 34 | 10.0 | +58.28% | +56.71% | +1.57% | 67.6% | 49.7% |
| top10 | 2026 | 14 | 10.0 | +28.45% | -0.50% | +28.95% | 57.1% | 50.7% |

10d representative:

| strategy | year | periods | selected_n | return | base_return | excess | period_win | stock_win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top10.0pct | 2020 | 25 | 341.6 | +23.59% | +13.49% | +10.10% | 44.0% | 48.5% |
| top10.0pct | 2021 | 24 | 385.0 | +43.12% | +34.33% | +8.79% | 62.5% | 52.0% |
| top10.0pct | 2022 | 24 | 432.5 | -5.71% | -3.91% | -1.81% | 58.3% | 46.9% |
| top10.0pct | 2023 | 24 | 468.9 | +3.61% | +6.21% | -2.60% | 54.2% | 45.4% |
| top10.0pct | 2024 | 25 | 484.0 | -8.64% | -6.79% | -1.85% | 40.0% | 42.6% |
| top10.0pct | 2025 | 24 | 482.9 | +64.93% | +55.78% | +9.15% | 79.2% | 57.3% |
| top10.0pct | 2026 | 10 | 489.8 | +2.70% | -4.17% | +6.87% | 50.0% | 45.1% |

10d narrower capacity check:

| strategy | year | periods | selected_n | return | base_return | excess | period_win | stock_win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top5.0pct | 2020 | 25 | 170.5 | +23.76% | +13.49% | +10.27% | 40.0% | 48.1% |
| top5.0pct | 2021 | 24 | 192.3 | +39.74% | +34.33% | +5.42% | 62.5% | 51.1% |
| top5.0pct | 2022 | 24 | 216.0 | -8.71% | -3.91% | -4.80% | 54.2% | 46.8% |
| top5.0pct | 2023 | 24 | 234.2 | -0.67% | +6.21% | -6.88% | 50.0% | 43.9% |
| top5.0pct | 2024 | 25 | 241.7 | -12.43% | -6.79% | -5.63% | 40.0% | 41.8% |
| top5.0pct | 2025 | 24 | 241.2 | +69.24% | +55.78% | +13.47% | 79.2% | 56.7% |
| top5.0pct | 2026 | 10 | 244.7 | +7.98% | -4.17% | +12.15% | 60.0% | 46.2% |

年度结论：

```text
1. 10d top10% 是年度稳定性最好的当前候选：2020、2021、2025、2026 跑赢 base，2022-2024 跑输或小幅跑输；整体回撤最低。
2. 7d top0.1% / top10 的总收益主要受 2025 和 2026 贡献，2023 是明显失败年份；这说明 7d 极端头部 alpha 有 regime 依赖。
3. 5d top10% 只有 2020 和 2026 明显跑赢，2021-2025 多数不如 base；不建议作为第一优先级。
4. 2024 对所有代表策略都偏弱，说明 close-token decoder epoch020 没有完全解决 regime 漂移问题。
5. 2025 是最强年份，几乎所有代表策略都有显著正收益；后续需要反事实拆解 2025 的行业、市值、波动率、市场状态偏好，避免只学到单一年份风格。
```

### 当前交易研究结论

```text
推荐优先级:
  1. 10d pred_close_ret_10d top10%: 当前最可落地，容量约 400-500 只/期，收益/回撤/PF 最平衡。
  2. 10d top5%: 收益略低但仍好，容量约 200+ 只/期，适合做容量压缩版本。
  3. 7d top10 或 top0.1%: 可作为高锐度小容量实验，必须搭配风控和 regime filter。
  4. 5d 暂不作为主线，只保留为短线事件研究。

不推荐:
  1. top1/top3/top5 的 5d/10d 窄 topK 裸持有，回撤和失效风险太高。
  2. 15d/20d/30d 的正向买入 score，50k 分层样本中 top bucket 已经失效。

下一步:
  1. 将 10d top10% / top5% 转成 QuantX external score 配置，做含执行约束、手续费、滑点的正式回测。
  2. 对 7d top10/top0.1% 做 TP/SL/冲高退出网格，重点控制 2023/2024 的左尾。
  3. 做 yearly/regime/industry/size/volatility 切片，解释 2025 强、2023/2024 弱的原因。
  4. 下次推理评测把 batch size 拉到显存上限；本次 4096/GPU 只吃约 10GB/32GB，吞吐还有优化空间。
```

## 2026-07-18 Formal QuantX closure: decoder epoch020

### Terminology correction

The preceding `5d/7d/10d` section was an offline score replay: it generated
cross-sectional scores and aggregated forward returns, but did not create
QuantX order, cost, position, rejection, or NAV artifacts. It must not be
treated as a formal backtest. The formal results below supersede its trading
conclusions.

### Reproducible full-market score artifact

```text
checkpoint:
  decoder_epoch_020.pt

artifact:
  evaluation/strategy_scores/decoder_epoch020_market_2020_2026_cadence5_7_10.parquet
  rows: 2,132,631
  union signal dates: 488
  5d score dates / rows: 311 / 1,359,384
  7d score dates / rows: 222 / 969,984
  10d score dates / rows: 156 / 680,622

QuantX bucket artifact:
  evaluation/strategy_scores/decoder_epoch020_market_2020_2026_cadence5_7_10_quantx_buckets.parquet
  dynamic top10% 10d rows: 68,000
  dynamic bottom10% 10d rows: 68,000

inference:
  8 GPUs, DDP by complete signal-date cross-section
  auto batch peak: about 28.1GB / 32GB per GPU, GPU utilization about 100%
  artifact and manifest validation: passed
```

Full-path offline metrics are retained in
`evaluation/strategy_scores/decoder_epoch020_market_2020_2026_cadence5_7_10_full_metrics.json`
with `status=offline_score_evaluation_only`; they are predictive diagnostics,
not trading returns.

### QuantX execution definition

```text
OOS signal range: 2020-01-02 to 2026-06-01
execution end:    2026-06-16 (allows final 10d holdings to exit)
universe:         all_a, 5,179 QuantX symbols; score artifact covers 5,173
selection:        dynamic daily top10% from the score artifact, not fixed topK
execution:        T+1 selector lag, close deal price, 10-trading-day time stop
constraints:      100-share lots, limit-up buy rejection, cash equal weighting
cost:             5bp commission, seller stamp tax 1bp, slippage 10bp
capital:          100M, sufficient for the declared 400+ stock capacity
```

The two full-market configurations passed QuantX dry-run, created complete
`summary.json`, `metrics.json`, `daily_nav.json`, `trades.json`, positions, and
formal bundle artifacts. Their run ids are listed below.

### Primary full-market formal result

| strategy | QuantX run id | total return | annual | maxDD | Sharpe | PF | trades | rejected | avg positions | total cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10d score top10% | `20260718_235538_decoder_epoch020_close_10d_top10pct_2020_2026_all_a` | +80.58% | +9.58% | -45.29% | 0.37 | 1.11 | 111,504 | 1,251 | 357.1 | 22.27M |
| 10d score bottom10% control | `20260718_235530_decoder_epoch020_close_10d_bottom10pct_2020_2026_all_a` | -52.54% | -10.90% | -76.21% | -0.43 | 0.87 | 104,739 | 2,499 | 335.6 | 10.13M |

The top10% result is a formal, positive all-market result, but it is not yet a
production recommendation: the Sharpe ratio is only 0.37 and drawdown remains
large. The bottom10% counterfactual is strongly negative under exactly the same
execution settings. This is the strongest evidence in this experiment that the
decoder score has genuine cross-sectional ordering value rather than merely
capturing market beta.

Annual all-market returns:

| strategy | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 partial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10d top10% | +23.43% | +25.76% | -19.04% | -3.72% | +3.33% | +44.92% | -2.68% |
| 10d bottom10% | -2.30% | -12.66% | -35.27% | -12.26% | -18.65% | +6.67% | +9.22% |

### Supporting mainboard slices

These are formal QuantX runs, but are only mainboard slices (3,189 symbols), so
they are supporting evidence rather than the all-market primary conclusion.

| strategy | total return | annual | maxDD | Sharpe | PF | trades | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5d top10% | -43.95% | -8.57% | -63.02% | -0.36 | 0.93 | 114,306 | failed after execution costs |
| 7d top10% | +1.68% | +0.26% | -50.25% | 0.01 | 1.00 | 86,510 | effectively neutral |
| 10d top5% | +23.46% | +3.32% | -47.96% | 0.13 | 1.04 | 31,050 | weaker than wider top10% |
| 10d top10% | +51.65% | +6.66% | -39.61% | 0.28 | 1.09 | 63,614 | positive but lower capacity slice |
| 10d bottom10% | -54.29% | -11.42% | -69.47% | -0.50 | 0.87 | 60,329 | negative control passed |

### Offline-to-formal reconciliation

```text
offline 10d top10% replay:
  total return +167.38%, maxDD -31.83%, Sharpe 0.71, PF 1.46

formal all-A QuantX 10d top10%:
  total return +80.58%, maxDD -45.29%, Sharpe 0.37, PF 1.11
```

The gap is expected and material. The replay used direct forward-return
aggregation without T+1 delay, lot sizing, rejected limit-up orders, trading
costs, slippage, cash constraints, or a live position book. QuantX records
22.27M total cost, 1,251 rejected orders, and an average of 357 open positions.
This is why only the QuantX result is used as the formal trading conclusion.

### Current decision and next research step

```text
confirmed:
  The decoder has tradable 10d cross-sectional ranking information on all A.

not confirmed:
  Production readiness. Current Sharpe 0.37 and -45.29% drawdown are inadequate.

reject:
  5d as a long-only direction, and 7d as a standalone long-only direction.

next:
  Keep the all-A 10d top10% score as the baseline. Test regime filters,
  transaction-cost-aware turnover reduction, and TP/SL/trailing exits against
  the same QuantX formal bundle. Every candidate must retain the score artifact,
  dry-run, full QuantX run, yearly table, and counterfactual control.
```

### Concentrated top10 counterfactual

The request to replace the broad dynamic top10% portfolio with the raw decoder
score top10 was tested as a separate all-A QuantX configuration. It used the
same score artifact, T+1 lag, costs, 10d exit, and 100M capital as the formal
top10% result; only `topk/max_positions` changed to 10.

```text
QuantX run id:
  20260719_000703_decoder_epoch020_close_10d_top10_2020_2026_all_a

result:
  total return:       -57.45%
  annual return:      -12.40%
  max drawdown:       -77.58%
  Sharpe:             -0.35
  profit factor:       0.88
  trades:              2,776
  rejected orders:        89
  average positions:     8.89
  max positions:          11
  total cost:         10.13M
```

Annual returns: 2020 -4.28%, 2021 -3.32%, 2022 -35.73%, 2023 -23.68%,
2024 +1.53%, 2025 +4.38%, 2026 partial -11.66%.

Conclusion: current decoder alpha is a broad cross-sectional distributional
signal, not a reliable extreme-head ranking signal. Concentrating it into 10
names destroys diversification and selects an unstable tail. Keep dynamic 10d
top10% as the formal baseline; do not use raw top10 concentration as a trading
strategy without a new model objective or a separate tail-calibration result.

### Concentrated top5 counterfactual

The otherwise identical all-A raw top5 test was also completed. It is worse
than raw top10, confirming that the decoder's raw 10d score does not have a
tradable extreme-head ordering.

```text
QuantX run id:
  20260719_001030_decoder_epoch020_close_10d_top5_2020_2026_all_a

result:
  total return:       -69.42%
  annual return:      -16.76%
  max drawdown:       -84.48%
  Sharpe:             -0.41
  profit factor:       0.86
  trades:              1,368
  rejected orders:        54
  average positions:      4.38
  max positions:          6
```

### Decoder versus diffusion concentrated-head control

The earlier positive diffusion top1 result was not evidence that every
diffusion top1 is profitable: the `score_prob_rel_ge_080_5d` top1 result was
negative. The positive comparator was the specific `pred_abs_q10_5d` score on
all_mainboard from 2024 onward, with liquidity/price/daily-volatility filters,
zero configured slippage, and a 5-trading-day time stop.

To isolate the material universe, date-range, filter, capital, and cost
differences, decoder raw `pred_close_ret_10d` was rerun with the same
all_mainboard filters, 2024-01-02 start, 1M capital, and cost configuration.
The decoder's own 10-day target, score cadence, and 10-trading-day time stop
were retained. Score coverage ends on 2026-06-01, so execution ends on
2026-06-16; this is not a fully equal holding-period/cadence comparison.

```text
case                 QuantX run id                                                        total     annual    maxDD     Sharpe   PF    trades
decoder raw top1     20260719_001646_decoder_epoch020_close_10d_top1_diffusion_controls   -39.96%   -18.76%   -55.69%   -0.31   0.82      114
decoder raw top5     20260719_001646_decoder_epoch020_close_10d_top5_diffusion_controls   -10.67%    -4.49%   -41.91%   -0.12   0.95      550
```

Annual returns:

```text
case                 2024       2025       2026 partial
decoder raw top1     -18.09%    +12.30%    -41.25%
decoder raw top5     -17.46%    +19.43%    -11.44%
```

Conclusion: neither all-A universe breadth, the 2020-2023 history, nor the
10bp decoder slippage explains the concentrated-head failure. This is a score
property: the decoder's raw 10d expected-return ranking contains broad
cross-sectional separation, as shown by the positive dynamic top10% result,
but is not calibrated or monotonic at ranks 1-5. The diffusion `pred_abs_q10`
signal and decoder `pred_close_ret_10d` are distinct learned objectives;
current evidence supports a signal/objective difference, not a general claim
that diffusion architecture is superior.

## 2026-07-19 Registered Plan: Matched Flow/DiT Versus Close-Token Decoder

Status: designed and recorded only. No new diffusion training, full inference,
or QuantX run is authorized by this section.

### Why a new matched control is required

The historical profitable diffusion control is
`market_all_week7_c2c_balanced_v1`: it uses the same pre-2020 temporal split,
close-to-close convention, and a balanced trend sample, but it predicts a
7-trading-day path. The current decoder experiment is
`market_all_close2close_balanced_v2`, which predicts a 30-trading-day path and
uses a separately seeded balanced sample. Therefore, comparing their existing
scores cannot isolate sampling, generative objective, or architecture.

The decoder's 8-path, 50,000-row time-stratified diagnostic did not establish
tail ranking. This is evidence against the claim that decoder sampling alone
explains the gap, but it is not a matched decoder-versus-diffusion result.

### Registered Experimental Design

```text
common dataset root:
  market_all_close2close_balanced_v2

common training specification:
  train split:     train_balanced, 2010-01-01 .. 2018-12-31
  selection split: validation_select, 2019
  target:          30d close-to-close absolute return path
  features:        same raw_relative inputs, scaler, neutral candidate features,
                   and frozen AE used by the decoder
  evaluation:      2020-01-02 .. 2026-06-01, sampled on 7-trading-day cadence

new control model:
  flow objective with a cross-DiT denoiser
  flow-head parameter count calibrated to approximately the decoder's 50M
  8-GPU DDP, 20 epochs, checkpoint every epoch
  TensorBoard and JSONL logs are mandatory

paired evaluation rows:
  exactly the same 50,000 (signal_date, instrument) pairs as the decoder
  time-stratified artifact
  222 signal dates, 225 or 226 instruments per date
  same 7d ground-truth close return and deterministic tie rule

distribution inference:
  32 stochastic paths for both models
  common score family: mean, q10, mean - 0.5 * std, P(return > 0)
  decoder additionally retains its greedy terminal-path score
```

### Required Evaluation and Decision Rule

1. Compare path MAE, direction accuracy, and `>=3%`, `>=5%`, `>=10%` event
   precision, recall, and lift.
2. Compare daily Top1/3/5/10 excess return, t-statistic, positive-excess-day
   rate, and annual slices under the paired sample.
3. The 50k result is diagnostic only: it cannot prove full-universe tail rank
   or trading performance.
4. Generate a full 2020-2026 score artifact and run QuantX only for a score
   that has a clearly positive, cross-year-stable paired advantage.
5. The historical 7d diffusion is retained only as context and must not be
   presented as a causal architecture comparison.

Research-integrity note: the existing 2020-2026 decoder outcomes have been
observed. They may diagnose the next model, but cannot be reused for post-hoc
score selection and then described as a fresh OOS conclusion.

## 2026-07-19 Implemented: Pairwise Reward Head for Close-Token Decoder

Status: implementation and 8-GPU DDP smoke test passed. No OOS score artifact,
offline alpha result, or QuantX result exists yet for this model.

### Objective

The prior decoder's 30d close-path imitation objective produced useful broad
10d separation but failed to preserve a monotone Top1/Top5 ordering. This run
keeps path imitation and adds an explicitly cross-sectional ranking objective:
the decoder must assign a larger scalar score to a stock with a larger realized
same-date 7d close-to-close return.

```text
frozen AE condition
  -> causal close-token decoder -> 30d token cross entropy
  -> decoder condition-only trunk + attention-pooling reward head -> s_rank

r_7d = close(T+7) / close(T+1) - 1
L_total = L_CE + 0.50 * L_pair
L_pair = weighted softplus(-sign(r_i - r_j) * (s_i - s_j) / temperature)
```

Only pairs from the same `signal_date` are compared. Thus a common market move
is removed from the label difference without introducing an unverified scalar
alpha regression target. Pairs with an absolute realized-return gap below
0.5% are excluded.

### Training Sampling and Controls

```text
path stream:       existing train_balanced decoder data, 30d token CE
rank stream:       raw all-market train split, one same-date group per step
rank group:        256 names, sampled as low/middle/high = 128/77/51
pair types:        high-middle, high-low, middle-low
tail emphasis:     pairs involving the high stratum receive 1.5x weight
optimizer:         pretrained decoder 5e-5, new reward head 3e-4
checkpointing:     every epoch, optimizer and scheduler state included
logging:           TensorBoard plus JSONL CE loss, pair loss, pair accuracy,
                   effective pair count, and validation ranking metrics
```

The underlying target metadata is asserted to be close-to-close with start
offset one, so the ranking endpoint is exactly the requested `T+1` to `T+7`
close return. The reward head is initialized fresh from the path decoder's
epoch-020 checkpoint; the AE and all existing decoder weights are reused.

### Implementation Verification

1. Real 2010-05-04 all-market data produced a 256-name group with the expected
   `128/77/51` strata and ordered realized 7d return ranges.
2. A real GPU forward/backward with the frozen AE, pretrained 49.75M decoder,
   and new 149,121-parameter reward head produced both CE and pairwise
   gradients.
3. An 8-GPU DDP one-epoch smoke run completed. It synchronized both losses,
   wrote TensorBoard/JSONL, validation ranking metrics, and an epoch checkpoint.
4. On 32GB RTX 5090 hardware, a single-process path batch of 928 plus one
   256-name rank group used 29.26GB peak allocated memory. DDP gradient
   buckets made 928 exceed the remaining memory during backward, so formal
   training uses batch 896 with `expandable_segments:True`; its observed DDP
   peak is 31.88GB allocated and 32.41GB reserved. The rank data loader uses
   two workers per GPU to avoid double-stream I/O contention with the
   four-worker path loader.

### Required Next Evidence

After the formal checkpoint is selected, export a complete persisted
`reward_rank_7d` score artifact for 2020-2026, inspect all-market daily
Top1/3/5/10 and fraction buckets, and run the chosen score through QuantX.
Neither training pair accuracy nor an offline bucket statistic is a trading
claim without that formal QuantX bundle.

### Fast Reward-Only Inference

The score exporter supports `--reward-only` for the ranking evaluation. It
computes frozen-AE conditions and `reward_rank_7d` only; it deliberately skips
the 30-step autoregressive close-path decoder. This is the correct fast path
for a 7d reward-head ranking study because decoded paths are not used to form
the selection score. It requires the 7d cadence, rejects stochastic path
sampling, preserves the normal score-artifact manifest, and produces 7d
cross-sectional TopK/fraction bucket metrics. The default exporter behavior
still generates paths and remains the required mode for trajectory evaluation.

### Registered q10 Scoring Control

The prior positive diffusion comparator used a sampled lower-tail score
(`pred_abs_q10_5d`). In contrast, the formal decoder QuantX runs used the
single greedy path endpoint (`pred_close_ret_10d`). This is a material scoring
confound: it cannot distinguish a diffusion architecture advantage from an
advantage of using a lower conditional-return quantile rather than a point
forecast.

The earlier decoder q10 result was only an epoch-010, 50k, eight-rollout
diagnostic. Eight rollouts make a 10th percentile nearly a worst-rollout
statistic and do not establish a stable tail estimate or a formal trading
result. It must not be used to reject the q10 hypothesis.

After epoch 20, the registered control is a 50,000-row time-stratified 2020-
2026 diagnostic with 32 decoder rollouts per row. One matched artifact will
retain `pred_close_ret_7d`, sampled mean, sampled q10,
mean-minus-0.5-standard-deviation, and `reward_rank_7d`. The decision sequence
is fixed: compare daily Top1/3/5/10 monotonicity and RankIC first; only a
stable candidate proceeds to a full-market artifact and QuantX. This control
tests the scoring hypothesis without relabeling a score-selection result as an
architecture result.

## 2026-07-19 综合结论：扩散模型、Close-Token 解码器与成对比较奖励头

### 研究边界与证据标准

本节汇总历史 diffusion、原始 close-token decoder，以及加入 pairwise
训练后的 decoder 与 reward head。它更新上文“pairwise 模型尚无 OOS score
或 QuantX 证据”的状态，但不改写任何历史实验；每个结果均按实际 universe、
日期范围、score 与执行条件标注。

下文的交易结论必须同时具备已持久化的 score artifact、QuantX dry-run 与已完成的
QuantX run，且 run 内存在 `metrics.json`、`daily_nav.json`、orders、trades、
positions 与拒单记录。离线 `RankIC`、TopK 收益和 `P@10` 仅是诊断证据。

### 实验谱系与可比性

| 实验家族 | 训练 / score artifact | 选股 score | 正式 universe 与执行 | 对比状态 |
| --- | --- | --- | --- | --- |
| 历史 diffusion | `market_all_week7_c2c_balanced_v1`；7d close-to-close diffusion score | `pred_abs_q10_5d` / `fused_score` | `all_mainboard`，自 2024-01-02，含价格/流动性/日内过滤，固定 Top1、5d 止时、配置滑点为 0 | 仅作历史背景，不是与 decoder 匹配的对比 |
| 原始 close-token decoder，宽组合 | `market_all_pre2020_v1`，epoch020 | greedy `pred_close_ret_10d` | `all_a`，2020-2026，动态 top10%、10d 止时 | 正式全市场宽排序基线 |
| 原始 close-token decoder，集中组合 | 同一 epoch020；32 条随机路径 | `pred_sample_q10_7d` | `all_a`，2020-2026，固定 Top10、7d 止时 | 正式集中尾部 decoder 对照 |
| Pairwise close-token decoder，路径对照 | `market_all_close2close_balanced_v2`，pairwise epoch020；32 条随机路径 | `pred_sample_q10_7d` | `all_a`，2020-2026，固定 Top10、7d 止时 | 与原始 decoder 的正式 score 对比，但不是纯 pairwise loss 消融 |
| Pairwise reward head | 同一 pairwise epoch020；reward-only 全量 score | `reward_rank_7d` | `all_a`，2020-2026，固定 Top10、7d 止时 | reward head 的正式直接检验 |

关键不可比性限制：

1. 历史 diffusion Top1 使用不同的起始日期、universe、5d 持有期、可交易过滤、
   初始资金与零配置滑点。因此不能用其结果证明 diffusion 架构优于 decoder。
2. 原始 decoder 与 pairwise decoder 属于相同的大模型家族，集中对照也使用同一
   7d QuantX 执行定义；但数据根、tokenizer 构造、平衡采样和训练路径并非严格的
   CE-only 对 CE-plus-pairwise 消融。两者 q10 的差异是模型组合的实证证据，
   不能归因于 pairwise loss 单独造成。
3. 原始 decoder 的 10d 动态 top10% 有意保持宽容量（平均约 357 个持仓），它回答
   的容量问题不同于固定 Top10 尾部实验。

### 正式 QuantX 结果总表

| 策略 | QuantX run id | 区间 / universe | 总收益 | 年化 | 最大回撤 | Sharpe | PF | 交易数 | 平均持仓 | 成本 | 正式结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 历史 diffusion `pred_abs_q10_5d` Top1 | `20260717_182223_pure_ml_q10_5d_full_top1_pos1_h5_2024_2026_mainboard` | 2024-2026，主板加过滤 | +200.28% | +54.25% | -46.20% | 0.881 | 1.452 | 238 | 0.97 | 0.18M | 强历史集中信号，但不可与后续 all-A 实验直接比较 |
| 原始 decoder `pred_close_ret_10d` 动态 top10% | `20260718_235538_decoder_epoch020_close_10d_top10pct_2020_2026_all_a` | 2020-2026，全 A | +80.58% | +9.58% | -45.29% | 0.370 | 1.110 | 111,504 | 357.1 | 22.27M | 最强的分散化 decoder 基线；正式结果已归档于本日志 |
| 原始 decoder 随机 q10 Top10 | `20260719_150442_decoder_epoch020_q10_7d_top10_2020_2026_all_a` | 2020-2026，全 A | +15.22% | +2.22% | -59.01% | 0.067 | 1.029 | 4,086 | 9.16 | 17.15M | 弱但为正的集中 decoder 信号 |
| Pairwise decoder 随机 q10 Top10 | `20260719_160905_pairwise_decoder_epoch020_q10_7d_top10_2020_2026_all_a` | 2020-2026，全 A | -14.89% | -2.47% | -64.42% | -0.079 | 0.981 | 4,142 | 9.29 | 27.28M | 拒绝：q10 尾部没有可交易 alpha |
| Pairwise reward head Top10 | `20260719_153533_pairwise_reward_epoch020_7d_top10_2020_2026_all_a` | 2020-2026，全 A | +41.01% | +5.47% | -70.65% | 0.149 | 1.046 | 3,964 | 8.88 | 26.17M | 已确认正向尾部 alpha，但风险质量不足以进入生产 |

原始 decoder 宽组合的 run 目录已不在当前 `quantx/runs/` 目录树内。其正式指标、
run id、执行定义和年度表保留在上文 `Formal QuantX closure: decoder epoch020`。
上表其余各行均已于 2026-07-19 重新读取当前 `metrics.json` 工件。

### 集中 7d 策略的年度结果

下表三项都使用相同的全 A 2020-2026 信号安排、T+1 收盘成交、7 个交易日止时、
5bp 佣金、1bp 卖出印花税、10bp 滑点与 1 亿初始资金。这是原始 decoder q10、
pairwise decoder q10 和 reward head 最有价值的年度对照。

| 策略 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 decoder q10 Top10 | -25.02% | -3.60% | -26.74% | +11.67% | +28.69% | +61.77% | -6.40% |
| Pairwise decoder q10 Top10 | +33.89% | +36.18% | -31.28% | -15.63% | -21.16% | +14.36% | -10.69% |
| Pairwise reward Top10 | +43.41% | +19.66% | -27.08% | -27.24% | -27.96% | +76.69% | +21.66% |

reward head 不是跨 regime 稳定的组合。最大回撤在 2024-02-05 达到 -70.65%。
其正总收益主要来自 2020-2021 与 2025-2026，而 2022-2024 是连续三年的不利 regime。

### 全市场离线诊断及其含义

pairwise 运行持久化了两份完整 2020-2026 score artifact，每份均覆盖 222 个
signal date、969,984 行：

```text
reward-only artifact:
  pairwise_reward_epoch020_market_2020_2026_7d.parquet

32-path trajectory artifact:
  pairwise_decoder_epoch020_market_2020_2026_stochastic32_cached_bs160_chunk32_7d.parquet

推理：
  8 GPUs，按完整 signal-date 横截面做 DDP
  batch 160，sampling chunk 32，top-k 32，temperature 0.8，固定 seed 20260719
  32-path 解码期间每卡峰值约 28.1GB
```

| Score | 全市场日度 RankIC | Top10 平均 7d 收益 | Top10 相对全市场超额 | P@10：真实收益位于当日 top decile | 含义 |
| --- | ---: | ---: | ---: | ---: | --- |
| Pairwise decoder sampled q10 | 0.0553 | +0.387% | -0.007% | 13.78% | 存在宽泛单调关系，但没有可用的极端头部收益 |
| Pairwise decoder sampled mean | 0.0309 | +0.405% | +0.011% | 17.93% | 选股后接近中性，不是决定性的尾部 score |
| Pairwise decoder mean - 0.5 std | 0.0442 | +0.400% | +0.006% | 16.04% | 选股后接近中性 |
| Pairwise decoder probability positive | 0.0553 | +0.229% | -0.165% | 5.36% | 拒绝作为尾部 score |
| Pairwise reward head | 0.0200 | +0.624% | +0.230% | 17.30% | 全局排序弱，但极端候选压缩能力强 |

`P@10` 的定义是：每天取 score Top10，计算其中真实收益位于当天真实收益前 10%
的比例。随机选择的期望为 10%；reward head 的 17.30% 对应约 1.73 的 lift，
但它不是高精度分类器，大部分被选中的股票仍不在真实 top decile。

从指标到 PnL 的关键结论：

1. `RankIC` 平均全横截面的排序质量，不能保证 ranks 1-10 的校准、稳定性，
   也不能保证扣除成本后可交易。
2. pairwise decoder q10 的 `RankIC` 相对较高，但 Top10 无超额且 QuantX 为负；
   宽泛相关性不能穿透极端尾部选股规则。
3. reward head 的 `RankIC` 较低，但能选择统计上富集的极端子集。它更像稀有候选
   检索 score，而不是平滑的收益回归因子。
4. 50,000 行时间均衡诊断会低估 reward head：它每天只保留约 225 个标的，
   225 个样本中的 Top10 不等于约 4,000 个全市场标的中的 Top10。

### Pairwise 训练已经证明和未能证明的内容

**已被正式 QuantX 证实**

```text
标量 reward head 可以作为独立的全市场 Top10 score。
在正式执行约束后，它取得 +41.01% 总收益；同一 pairwise 模型的 sampled-q10
路径 score 则为 -14.89%。
```

**尚未证实**

```text
pairwise 训练改善了 decoder 对未来 close 路径的排序。
当前 pairwise decoder q10 为负，而此前 decoder q10 弱正。这是当前模型组合不支持
预期目标的证据，但两次 decoder 的数据构造和训练历史不完全相同，不能视为 pairwise
loss 的因果消融。
```

**当前拒绝使用**

```text
把 pairwise decoder sampled q10 用作集中 Top10 交易 score。
即使 RankIC 为正，其正式结果为负且 PF 小于 1。
```

### 与历史 Diffusion 结果的关系

历史 diffusion `pred_abs_q10_5d` Top1 在其自身设置下是有效的 QuantX 证据：
总收益 +200.28%、Sharpe 0.881、PF 1.452。它表明条件路径的下分位统计量可以具有
经济价值，但不能证明以下命题：

```text
1. diffusion 架构普遍优于 close-token decoder；
2. q10 对每一种生成模型都是普适正确的 score；
3. 历史 diffusion 的收益可以与后续全 A、2020-2026 的 decoder 和 reward head
   收益做数值级别的直接比较。
```

历史 diffusion run 存在实质差异：它从 2024 年开始，只交易 `all_mainboard`，
使用价格/流动性/日内过滤、5d 止时和 Top1 容量，以 100 万起始资金运行，配置滑点为
零且买入收取印花税。在对架构作出判断前，仍需要执行上文注册的匹配 Flow/DiT 对 decoder
实验。

### 当前研究决策

| 决策 | 状态 | 原因 |
| --- | --- | --- |
| 保留原始 decoder 10d 动态 top10% 作为分散化基线 | 保留 | 本研究线中正式 decoder 风险调整结果最好：+80.58%、Sharpe 0.37、最大回撤 -45.29% |
| 保留原始 decoder 7d q10 Top10 作为研究对照 | 作为弱对照保留 | 虽为正但质量较低：+15.22%、Sharpe 0.067、最大回撤 -59.01% |
| 交易 pairwise decoder q10 Top10 | 拒绝 | 同口径下 -14.89%、Sharpe -0.079、PF 0.981 |
| 将 reward head 作为尾部 alpha 分支继续研究 | 继续，但先做风控研究 | 正式结果 +41.01%，但最大回撤 -70.65% 且存在 regime 依赖 |
| 声称 diffusion 优于 decoder | 拒绝 | 现有实验的数据、过滤、期限、score 和成本并不匹配 |

### 按信息价值排序的下一步实验

1. **先做 reward 风控，而不是继续挖 score**：固定 reward Top10 score，在同一 QuantX
   bundle 上测试预注册的 regime、波动率、换手和仓位风险控制。每个变体都需要基线对照
   和年度表，不得只针对 2025 年收益调参。
2. **严格 pairwise 消融**：固定一个数据根、tokenizer、AE checkpoint、seed、batch
   schedule 与 20-epoch 预算，训练 CE-only 和 CE-plus-pairwise 两个版本；随后用完全相同
   的 50k 行、32 条采样路径和全市场 score 协议评测。这是验证 pairwise loss 是否改变
   路径质量的唯一有效实验。
3. **匹配的 Flow/DiT 对照**：执行已注册的 Flow/DiT 实验后，才能将历史 diffusion
   解读为架构优势。
4. **工件纪律**：每次全量推理必须持久化 parquet、JSON manifest、score 覆盖、离线指标、
   QuantX config、dry-run 输出、正式 run id、年度收益和拒单统计。离线 replay 永远不能
   替代 QuantX。

### 复现指针

```text
pairwise reward score：
  market_all_close2close_balanced_v2/runs/
  market-all-close2close-pairwise-reward50m-bs896-8gpu-20ep-20260719-115110/
  oos/pairwise_reward_epoch020_market_2020_2026_7d.parquet

pairwise 轨迹 score：
  market_all_close2close_balanced_v2/runs/
  market-all-close2close-pairwise-reward50m-bs896-8gpu-20ep-20260719-115110/
  oos/pairwise_decoder_epoch020_market_2020_2026_stochastic32_cached_bs160_chunk32_7d.parquet

reward QuantX config / run：
  quantx/configs/strategies/generated/pairwise_reward_formal/
  pairwise_reward_epoch020_7d_top10_2020_2026_all_a.yaml
  quantx/runs/20260719_153533_pairwise_reward_epoch020_7d_top10_2020_2026_all_a

pairwise decoder q10 QuantX config / run：
  quantx/configs/strategies/generated/pairwise_reward_formal/
  pairwise_decoder_epoch020_q10_7d_top10_2020_2026_all_a.yaml
  quantx/runs/20260719_160905_pairwise_decoder_epoch020_q10_7d_top10_2020_2026_all_a

原始 decoder q10 QuantX run：
  quantx/runs/20260719_150442_decoder_epoch020_q10_7d_top10_2020_2026_all_a

历史 diffusion q10 QuantX run：
  quantx/runs/20260717_182223_pure_ml_q10_5d_full_top1_pos1_h5_2024_2026_mainboard
```

## 2026-07-19 独立 Reward Transformer：全上涨覆盖的同日 Pairwise 实验

### 研究问题与训练约束

本节是对上一节 pairwise reward 实验的独立续实验。目标不是训练 close-token
decoder 或 diffusion 路径头，而是仅训练一个标量 reward Transformer，检验冻结 AE
condition 能否形成同时具备较高容量和可交易收益的同日横截面排序。旧实验的
`pairwise_reward_epoch020` 使用的是另一份样本构造和 score 工件，本节结果不可回填
为旧实验结果。

训练目标和数据构造严格固定如下：

```text
feature condition:   冻结 AE encoder 输出 [B, 27, 128]
reward model:        6 层 Transformer, d_model=512, 8 heads, MLP ratio=4.0
public score:        reward_score_7d = sigmoid(reward_logit), FP32 输出 [0, 1]
pair objective:      softplus(-(logit_good - logit_bad))
centering penalty:   0.01 * mean(all pair logits)^2，防止仅排序损失的共同 logit 漂移
target:              同日 T+1 至 T+7 close-to-close 7d return
train source:        2020 年以前全市场 raw train split
epochs:              20，8 GPU DDP，每 epoch checkpoint + TensorBoard + JSONL
```

每个 `signal_date` 的 pair 不是按随机日组采样，而是遵守用户指定的覆盖契约：

1. `up` 定义为 7d return `>= +5%`，所有能找到更差同日 bad 的上涨股票都作为 good，
   不做上采样截断。
2. 若当天 eligible `up` 数为 `U`，从同日 `range` 与 `down` 各抽取恰好 `U` 条 good；
   `range/down` 不足才在该类别内确定性循环，绝不跨日补样。
3. 每条 good 仅配一条同日 bad，且满足
   `return_7d_bad < return_7d_good - 0.005`。
4. 每轮将三类 good 展平为独立 pair，由 DataLoader 按 pair batch 训练；range/down 与
   bad 的随机性随 epoch 改变，但同日约束不变。

数据统计：原始 train 为 `4,155,867` 行、`2,078` 个日期；严格可构造日期为
`2,049`，eligible up 为 `764,499`。有 `29` 个日期缺少满足严格 gap 的 down
counter-class，因而剔除 `29,936` 条 up。每 epoch 固定训练 `2,293,497` 个 pair，
即所有 eligible up 加上等量 range 与 down。

### 训练、验证与 score 工件

正式 run：

```text
market_all_close2close_balanced_v2/runs/
market-all-pre2020-ae-reward-transformer20m-allup-centered-bs3072-8gpu-20ep-20260719-175714

final checkpoint:
  checkpoints/reward_epoch_020.pt (218MB)
training manifest:
  train_manifest.json
logs:
  logs/train_metrics.jsonl + TensorBoard event files
```

训练使用每 rank `2,312` pair（每 epoch 124 个完整 batch），峰值约 `19.88GB/GPU`。
epoch 20 的 train pair accuracy 为 `69.91%`、loss `0.5261`；chosen/rejected score
均值为 `0.6715 / 0.5553`。下表是从未参与训练的横截面验证（170 日期、545,891 行）：

| epoch | RankIC | RankIC 正比例 | Top1 7d 超额 | Top5 7d 超额 | Top10 7d 超额 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.0504 | 64.12% | +0.657% | +0.513% | +0.444% |
| 10 | 0.0742 | 78.82% | +0.264% | +0.363% | +0.424% |
| 15 | 0.0692 | 73.53% | +0.252% | +0.547% | +0.620% |
| 20 | 0.0786 | 79.41% | +0.703% | +0.606% | +0.497% |

最终全市场推理以 8 GPU DDP 执行，rank 按完整 `signal_date` 分片，自动 batch 最终为
`32,768`。推理模型峰值约 `20.06GB/GPU`，达到当前 exporter 的 batch 上限；每 rank
约 30-37 秒完成自己的日期集合。持久化工件：

```text
score parquet:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer20m-allup-centered-bs3072-8gpu-20ep-20260719-175714/
  score_artifacts/reward_epoch020_20200102_20260715.parquet

manifest:
  score_artifacts/reward_epoch020_20200102_20260715.json
```

manifest 已校验：`6,788,182` 行、`1,552` 个 signal date、`5,173` 个 instrument，
score 列为 `reward_score_7d`。请求终止日为 `2026-07-15`，但原始数据可形成完整
未来 7 日 label 的末日为 `2026-06-02`；QuantX 运行到 `2026-06-16` 仅用于结清最后
一批持仓，不在 score 覆盖外新建仓。

全样本离线诊断仅作为进入 QuantX 前的辅助证据，不能替代交易结论：

```text
RankIC mean:                    0.0676
Top1 / Top5 / Top10 7d 超额:    +0.111% / +0.412% / +0.483%
Top20 7d 超额:                  +0.513%
Top0.5% / 1% / 2% 7d 超额:      +0.490% / +0.480% / +0.431%
```

### QuantX 正式容量回测

所有正式配置均通过 `quantx.tools.run_backtest --dry-run --json`，使用相同的
`all_a` 股票池、`2020-01-02` 至 `2026-06-16` 区间、`T+1 close` 成交、7 个交易日
time stop、`5bp` 佣金、`1bp` 卖出印花税、`10bp` 滑点、1 亿初始资金和外部 score
manifest 强校验。不同配置只改变 `topk=max_positions`，不对 2025 或任一年做参数调优。

| 容量 | QuantX run id | 总收益 | 年化 | 最大回撤 | Sharpe | PF | 交易数 | 平均持仓 | 平均已平仓收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top1 | `20260719_185729_reward_transformer_allup_epoch020_7d_top1_2020_2026_all_a` | +28.02% | +3.90% | -72.85% | 0.069 | 1.049 | 442 | 0.99 | +0.563% |
| Top5 | `20260719_185847_reward_transformer_allup_epoch020_7d_top5_2020_2026_all_a` | -29.34% | -5.24% | -69.18% | -0.134 | 0.949 | 2,216 | 4.97 | +0.145% |
| Top10 | `20260719_185945_reward_transformer_allup_epoch020_7d_top10_2020_2026_all_a` | +59.73% | +7.52% | -59.02% | 0.210 | 1.071 | 4,428 | 9.93 | +0.361% |
| Top20 | `20260719_190124_reward_transformer_allup_epoch020_7d_top20_2020_2026_all_a` | +219.22% | +19.69% | -35.02% | 0.576 | 1.173 | 8,852 | 19.85 | +0.528% |

Top20 的年度收益：

| 容量 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top1 | -30.74% | +53.38% | -25.96% | -41.13% | +47.29% | +91.78% | -2.11% |
| Top5 | -0.90% | -1.11% | -28.17% | -20.23% | +4.38% | +5.77% | +13.97% |
| Top10 | +18.20% | +28.99% | -38.84% | +17.62% | +16.05% | +15.14% | +9.01% |
| Top20 | +17.83% | +17.86% | -7.14% | +72.95% | +38.21% | +6.70% | -2.94% |

在本次预注册的固定容量格点 `{1, 5, 10, 20}` 内，Top20 同时拥有最大容量和最好的
收益、Sharpe、PF、回撤，因而是当前 reward-only 分支的首个“容量与收益未发生明显
互相牺牲”的正式证据。该结论仅限这四个容量，不可扩展为所有更大 TopK 的全局最优。

容量曲线没有单调性：Top5 为负、Top10 转正、Top20 显著更强。这说明 score 不是
逐 rank 平滑递减的收益预测；它更像需要一定横截面宽度来平均单名路径风险的候选检索
score。不能据此把 Top1 或 Top5 当作可生产部署的集中 alpha。

### 同日方向性反事实

为排除“任何全 A、20 持仓、7d 持有策略都能赚钱”的解释，补充了一个预注册式
directionality control。它和 Top20 使用相同日期、同一 score 工件、同一 all-A pool、
同一成交/成本/持有规则和同一 `max_positions=20`，唯一差异是每一天从
`score_desc` 改为 `score_asc`，即选择同日 score Bottom20：

| 策略 | QuantX run id | 总收益 | 年化 | 最大回撤 | Sharpe | PF | 2020-2026 各年表现 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Top20 | `20260719_190124_reward_transformer_allup_epoch020_7d_top20_2020_2026_all_a` | +219.22% | +19.69% | -35.02% | 0.576 | 1.173 | 2020-2025 六个完整年度中仅 2022 为负 |
| Bottom20 | `20260719_190536_reward_transformer_allup_epoch020_7d_bottom20_2020_2026_all_a` | -98.50% | -47.81% | -98.88% | -1.285 | 0.784 | 2020 至 2026 部分全部为负 |

Bottom20 分年收益为 `-18.01% / -23.81% / -65.37% / -63.84% / -63.46% / -36.06% /
-17.91%`。这个控制不是不同日期的好坏样本比较，而是每个交易日同一横截面内，保留
完全相同的 execution path，只反转 reward score 的选择方向。因此它直接支持如下命题：
当前正收益主要来自 reward head 的同日横截面排序方向，而不是静态 market beta、交易
频率或 QuantX 配置偶然产生的收益。

### 当前结论、风险与下一步

**已证实**：在 pre-2020 训练、2020-2026 全市场 OOS 的本次独立实验中，冻结 AE
condition 上的纯 reward Transformer 可以形成可交易的 7d 同日排序。Top20 的正式
收益、风险调整指标和 Bottom20 方向性反事实共同构成比离线 RankIC 更强的证据。

**未证实**：

1. Reward Transformer 架构普遍优于 diffusion 或 decoder。当前实验的目标、数据
   平衡方式、score 与执行条件不同，不能跨实验家族做架构归因。
2. Top20 是更大容量下的最优点，或 score 在 Top20 之后仍保持单调。
3. 该策略可直接生产。全 A 含 ST、小盘和高波动标的；即使 Top20 的回撤已显著下降，
   `-35.02%` 仍需要独立、预注册的流动性、涨停/停牌、行业集中和 regime 风控验证。

下一步应固定本节的 Top20 score、日期、成本和容量，把风控作为唯一变化项；每个变体
均需保留 Top20 基线、Bottom20 反事实和分年表。不要再根据 2023/2024 的高收益倒推
修改 reward score 或训练标签。若要继续探索容量，应新增独立的 `{30, 50, 100}` 预注册
格点，并对每个格点运行同口径 QuantX，而不能以离线 fraction bucket 代替。

### 复现指针

```text
training / checkpoint:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer20m-allup-centered-bs3072-8gpu-20ep-20260719-175714/
  checkpoints/reward_epoch_020.pt

full-market score / manifest:
  .../score_artifacts/reward_epoch020_20200102_20260715.parquet
  .../score_artifacts/reward_epoch020_20200102_20260715.json

QuantX configs:
  quantx/configs/strategies/generated/reward_transformer_allup_epoch020_formal/

QuantX runs:
  20260719_185729_reward_transformer_allup_epoch020_7d_top1_2020_2026_all_a
  20260719_185847_reward_transformer_allup_epoch020_7d_top5_2020_2026_all_a
  20260719_185945_reward_transformer_allup_epoch020_7d_top10_2020_2026_all_a
  20260719_190124_reward_transformer_allup_epoch020_7d_top20_2020_2026_all_a
  20260719_190536_reward_transformer_allup_epoch020_7d_bottom20_2020_2026_all_a
```

## 2026-07-19 独立 Reward Transformer：5d 目标、50M、40 Epoch 全量正式闭环

### 实验定位与训练契约

本节是上一节 7d / epoch020 Reward Transformer 的独立续实验，不回填或覆盖旧结果。目标
改为预测和排序 `T+1` 至 `T+5` 的 close-to-close 收益；冻结的 AE encoder 不训练，reward
Transformer 扩大为 `49,727,233` 个可训练参数：`d_model=768`、7 层、12 heads、MLP
ratio 4.0。训练仍只使用 pre-2020 全市场样本，8 GPU DDP 训练 40 epoch；每一个 epoch
持久化 checkpoint，且 JSONL 与 TensorBoard 为强制工件，没有 early stopping 或只保留最佳
checkpoint 的逻辑。

5d pair 构造遵循同日约束，不能把不同交易日的好/坏样本相互比较：

1. `up` 为未来 5d 收益 `>= +5%`。当天所有存在严格同日 bad 的 up 股票均作为 good，
   不做上采样截断。
2. 当天 eligible up 数为 `U` 时，从同日 `range=(-5%, +5%)` 与
   `down<=-5%` 各确定性采样或循环至 `U` 条 good。因此每个可用日期的 good 分布为
   up/range/down 各约三分之一。
3. 每条 good 配一个同日 bad，严格满足
   `return_5d_bad < return_5d_good - 0.005`；没有跨日期补样。

数据根共有 2,078 个原始日期，其中 2,016 个日期可严格构造 pair。每 epoch 使用
`1,702,144` 对；最终 epoch040 train loss 为 `0.5567`、pair accuracy 为 `66.88%`，
chosen/rejected score 均值为 `0.6531 / 0.5534`。这是训练集判别指标，不能直接视为 alpha
或交易收益。

内部横截面验证（170 日期、545,891 行）在 epoch040 的 5d RankIC 为 `0.0608`，正 RankIC
日期占比 `75.29%`；Top1/5/10 相对全市场平均收益为 `+0.099% / +0.532% / +0.479%`。
该诊断说明总体排序为正，但 Top1 的边际很弱，因而正式 QuantX 必须分别检验各个容量。

### 全量 Score Artifact

使用 epoch040 在 8 GPU DDP 上对完整 signal-date 横截面推理。每个 rank 负责完整日期集合，
自动 batch 探测最终取 `32,768`，单卡峰值 `27.54GB`，各 rank 的计算时间约 39-42 秒。输出
工件由 exporter 写入 parquet 和强制 manifest，不能以临时内存 score 代替：

```text
score parquet:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-40ep-20260719-193127/
  score_artifacts/reward_epoch040_20200102_20260715.parquet

manifest:
  .../score_artifacts/reward_epoch040_20200102_20260715.json

offline metrics:
  .../score_artifacts/reward_epoch040_20200102_20260715_full_metrics.json
```

manifest 校验结果：`6,788,182` 行、`1,552` 个 signal date、`5,173` 个 instrument，唯一 score
列为 `reward_score_5d`，范围为 sigmoid 后的 `[1e-6, 1-1e-6]`。虽然请求推理终止日为
`2026-07-15`，prediction split 能形成完整未来 5d target 的最后 signal date 为
`2026-06-02`。QuantX 仍运行至 `2026-06-16`，这段区间不再在 score 覆盖外开新仓，只用于
结清最后一批 5 个交易日持仓。

### QuantX 正式容量回测

5 个配置均已先通过 `quantx.tools.run_backtest --dry-run --json`。正式回测固定为 `all_a`、
`2020-01-02` 至 `2026-06-16`、外部 manifest 强校验、同日横截面排序、`lag: 1`、T+1
close 成交、5 个交易日 time stop、5bp 佣金、1bp 卖出印花税、10bp 滑点和 1 亿初始资金。
不同配置只改变 `topk=max_positions` 或从 `score_desc` 改为 `score_asc`；没有针对 2025
或任一年度调参。所有正式 run 的 `signal_errors` 均为空。

| 容量 | QuantX run id | 总收益 | 年化 | 最大回撤 | Sharpe | PF | 交易数 | 平均持仓数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top1 | `20260719_202805_reward_transformer_allup_epoch040_5d_top1_2020_2026_all_a` | -50.17% | -10.23% | -82.50% | -0.178 | 0.951 | 618 | 0.99 |
| Top5 | `20260719_202812_reward_transformer_allup_epoch040_5d_top5_2020_2026_all_a` | -59.21% | -12.97% | -79.94% | -0.341 | 0.914 | 3,090 | 4.95 |
| Top10 | `20260719_202820_reward_transformer_allup_epoch040_5d_top10_2020_2026_all_a` | +46.21% | +6.06% | -59.03% | 0.171 | 1.035 | 6,182 | 9.91 |
| Top20 | `20260719_202836_reward_transformer_allup_epoch040_5d_top20_2020_2026_all_a` | +130.03% | +13.77% | -47.99% | 0.407 | 1.088 | 12,356 | 19.82 |
| Bottom20 | `20260719_202838_reward_transformer_allup_epoch040_5d_bottom20_2020_2026_all_a` | -99.48% | -55.74% | -99.53% | -1.603 | 0.706 | 11,900 | 19.14 |

平均已平仓持有期约 `7.57` 个自然日，与 5 个交易日的 time stop 跨越周末一致，并非使用了
超过 5 个交易日的持有期。Bottom20 较 Top20 有更多拒单（`1,009` 对 `193`），这是两个方向
选到的股票可交易性不同所致；两者的日期、股票池、score 工件、成本和执行规则保持相同。

五个容量的年度复利收益如下，2026 为截至 `2026-06-16` 的部分年度：

| 容量 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top1 | +20.00% | +26.70% | +2.63% | -44.01% | -12.68% | -2.95% | -32.71% |
| Top5 | +12.93% | -20.11% | +7.32% | -43.56% | -11.19% | +3.98% | -19.18% |
| Top10 | +31.68% | +26.33% | +17.68% | -21.02% | -19.89% | +48.83% | -20.68% |
| Top20 | +23.79% | +42.15% | +2.82% | -4.33% | +18.67% | +29.20% | -13.33% |
| Bottom20 | -49.54% | -35.44% | -73.83% | -66.36% | -67.87% | -37.87% | -9.54% |

### 同日方向性反事实与边界结论

Top20 与 Bottom20 使用同一日的同一全 A 横截面、同一 score artifact、同一容量、同一成本和
同一持有规则，唯一模型选择方向为当日 `reward_score_5d` 的高分或低分。Top20 全期
`+130.03%`，而 Bottom20 为 `-99.48%`，且 Bottom20 在 2020 至 2026 部分每一年均为负。
这是强的方向性反事实：该收益不能合理归因于“任意全 A、20 持仓、5d 策略都可赚钱”。但它不
等同于已完成所有生产风险证明，尤其不能替代流动性、涨停停牌、ST、小盘暴露、行业集中和
regime 风控测试。

**本次已证实**：这个 5d reward score 在 pre-2020 训练、2020-2026 全 A OOS 的固定 QuantX
口径下有可交易的同日横截面方向性，且有效容量位于本次预注册格点中的 Top10 至 Top20。Top20
是四个正向容量格点中收益、风险调整指标和回撤最好的选择；Top10 为更集中的但质量较低的备选。

**本次未证实**：

1. Top1 或 Top5 是更优的尾部 alpha。两者正式结果均为负，说明该模型的最极端单名排序
   尚不稳定。
2. Top20 是任意更大容量下的全局最优，或 score 的期望收益随 rank 单调递减。现有格点只覆盖
   `{1, 5, 10, 20}`。
3. 5d / epoch040 模型优于上一节 7d / epoch020 模型、diffusion 或 decoder。目标期限、reward
   容量、训练 epoch 与模型状态不同，跨实验收益不能做架构归因。

后续应把本节 Top20 的 score、日期、成本和容量固定为基线，只测试预注册的风险控制变量；每个
变体必须同时保留 Top20 基线、Bottom20 同日方向性反事实和逐年收益表。不要为修复 2023 或
2026 部分年度而回溯修改训练标签、样本或 score。

### 复现指针

```text
training / checkpoint:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-40ep-20260719-193127/
  checkpoints/reward_epoch_040.pt

full-market score / manifest:
  .../score_artifacts/reward_epoch040_20200102_20260715.parquet
  .../score_artifacts/reward_epoch040_20200102_20260715.json

QuantX configs:
  quantx/configs/strategies/generated/reward_transformer_allup_epoch040_5d_formal/

QuantX runs:
  20260719_202805_reward_transformer_allup_epoch040_5d_top1_2020_2026_all_a
  20260719_202812_reward_transformer_allup_epoch040_5d_top5_2020_2026_all_a
  20260719_202820_reward_transformer_allup_epoch040_5d_top10_2020_2026_all_a
  20260719_202836_reward_transformer_allup_epoch040_5d_top20_2020_2026_all_a
  20260719_202838_reward_transformer_allup_epoch040_5d_bottom20_2020_2026_all_a
```

## 2026-07-19 后验 Checkpoint 对照：epoch010 对 epoch040 的尾部退化证据

### 动机与可比性

epoch040 的正式结果出现了一个不能被训练 loss 忽略的矛盾：Top10/20 为正，但最集中的
Top1/5 为负。基于“继续收敛可能损伤高分尾部”的假设，补做 epoch010 的完整 score export
与正式 QuantX。该比较不是训练前注册的 checkpoint sweep，因而属于**后验诊断**；它能识别
当前 loss 的风险，不应被直接解读为无偏的模型选择或生产收益估计。

为了使差异只来自 checkpoint，epoch010 和 epoch040 严格固定以下内容：

```text
AE checkpoint / reward architecture / input scaler / feature rows: 相同
score range / transform:       reward_score_5d = sigmoid(reward_logit)，相同
full score coverage:           6,788,182 行、1,552 日期、5,173 标的，相同
QuantX universe / dates:       all_a，2020-01-02 至 2026-06-16，相同
execution:                     同日 score，lag 1，T+1 close，5 交易日止时，相同
cost / initial cash:           5bp 佣金、1bp 卖出印花税、10bp 滑点、1 亿，相同
capacity / control:            Top1/5/10/20 与同日 Bottom20，相同
```

两个 checkpoint 的训练内表现反而朝相反方向变化：epoch010 的 train pair accuracy 为
`59.59%`、交叉截面 RankIC 为 `0.0528`；epoch040 分别升至 `66.88%`、`0.0608`。内部验证
的 Top1 5d 超额则从 epoch010 的 `+0.972%` 下降至 epoch040 的 `+0.099%`，已经提示整体
RankIC 改善未必等同于头部候选质量改善。

### 正式 QuantX 对照

| 容量 | epoch010 run id | epoch010 总收益 | epoch040 总收益 | epoch010 Sharpe | epoch040 Sharpe | epoch010 PF | epoch040 PF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top1 | `20260719_204330_reward_transformer_allup_epoch010_5d_top1_2020_2026_all_a` | -4.12% | -50.17% | -0.012 | -0.178 | 0.994 | 0.951 |
| Top5 | `20260719_204340_reward_transformer_allup_epoch010_5d_top5_2020_2026_all_a` | +266.97% | -59.21% | 0.573 | -0.341 | 1.192 | 0.914 |
| Top10 | `20260719_204344_reward_transformer_allup_epoch010_5d_top10_2020_2026_all_a` | +116.25% | +46.21% | 0.361 | 0.171 | 1.101 | 1.035 |
| Top20 | `20260719_204402_reward_transformer_allup_epoch010_5d_top20_2020_2026_all_a` | +141.22% | +130.03% | 0.432 | 0.407 | 1.116 | 1.088 |
| Bottom20 | `20260719_204402_reward_transformer_allup_epoch010_5d_bottom20_2020_2026_all_a` | -99.78% | -99.48% | -1.697 | -1.603 | 0.678 | 0.706 |

五个 epoch010 run 均通过 dry-run，正式 run 的 `signal_errors` 均为空。epoch010 Top5 的
最大回撤为 `-50.97%`、Top20 为 `-49.41%`；因此 Top5 更高收益并不代表已经具备低风险生产
质量。Top1 从 epoch040 的深度亏损恢复至接近持平，但仍没有形成独立可交易的集中 alpha。

### 修正后的结论

1. **强支持尾部/中高分排序随继续训练退化的假设**：epoch010 的 train accuracy 和 RankIC
   都低于 epoch040，但 Top5、Top10、Top20 的正式收益均更高，Top1 的损失也显著收敛。
   这不是单纯的 Top1 噪声，Top5 从 `+266.97%` 变为 `-59.21%` 是跨容量的一致变化。
2. 当前 pairwise loss 更擅长持续提升整体同日排序拟合，却不能保证高分候选之间的 OOS
   排序稳定。score 的整体方向没有消失，因为两个 checkpoint 的 Bottom20 都接近归零；退化
   主要发生在高分候选的相对次序和容量结构。
3. 上一节将 epoch040 Top20 作为 5d 分支基线的说法需要修正：在当前已经运行的两个 checkpoint
   中，epoch010 的 Top5/10/20 都更强，epoch040 只能作为“完整 40 epoch 的训练终点”，不能
   作为默认的交易 checkpoint。
4. 不能因为本次后验比较立即声称 epoch010 是无偏的最终最优 checkpoint。使用同一个
   2020-2026 OOS 区间发现并选择 epoch010 会产生 checkpoint selection bias。

后续正确的实验是预先固定候选 checkpoint 格点，例如 `{5, 10, 15, 20, 25, 30, 35, 40}`，
在独立的模型选择时间段选择 checkpoint，再只在未参与选择的后续时间段运行 QuantX。若继续
沿用当前训练目标，还应记录每个 checkpoint 的 Top1/5/10/20 和 Bottom20，而不能以 train
loss、pair accuracy 或全截面 RankIC 代替 checkpoint selection。

### 复现指针

```text
epoch010 score / manifest:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-40ep-20260719-193127/
  score_artifacts/reward_epoch010_20200102_20260715.parquet
  score_artifacts/reward_epoch010_20200102_20260715.json

epoch010 QuantX configs:
  quantx/configs/strategies/generated/reward_transformer_allup_epoch010_5d_formal/
```

## 2026-07-19 Top30 Hard Negative + Top100 无最小 Gap：5d Reward Transformer 10 Epoch 正式闭环

### 实验动机与训练契约

上一轮 head fan-out 把高分 good 与同日大量普通股票重复比较，epoch5/10 的未加权验证没有
形成稳定的 Top5/10 超额。该续实验保持冻结 AE、Reward Transformer 架构、pre-2020 训练集、
5d target、全 A OOS 股票池和成本参数不变，只把 high-rank good 的 bad pool 改为同日的局部
hard negative，以直接学习头部约 30 只股票的相对次序。

模型为冻结 AE `40,427,736` 参数加 reward Transformer `49,727,233` 可训练参数
(`d_model=768`、7 层、12 heads)。8 GPU DDP 训练 10 epoch，每 epoch checkpoint，JSONL 和
TensorBoard 均已保存。

训练 pair 的严格日内契约如下：

1. 每日按真实未来 5d close-to-close 收益在**同一横截面**排序。全局 rank=1 的 eligible up
   good 生成 100 个 bad draw；全局 rank=2..10 的 eligible up good 各生成 50 个 bad draw。
2. 这些 rank=1..10 的 bad 只能来自同日、收益排名更靠后且不超过 rank=30 的股票。取消
   `0.5%` 最小收益差，但仍严格要求 `bad_return < good_return`，即 gap 必须大于零。
3. 所有全局 rank<=100 的 pair 同样取消 `0.5%` 最小 gap，但仍要求 bad 的收益严格更低。
   全局 rank>100 的基础 pair 仍要求
   `bad_return < good_return - 0.005`。
4. 原有的基础分布不变：所有 eligible up 都保留一次；同日 range/down 各循环采样至同样数目，
   因而 unique good 层面仍约为 up/range/down 各三分之一。hard fan-out 只改变 pair-level
   loss 权重，不把不同日期样本互相配对。

全训练集有 2,016 个可用日期，`2,789,898` pair/epoch；其中基础 good 的类别平衡仍存在，
额外 `1,087,758` 个 pair 来自头部 fan-out。没有 rank<=10 样本因 Top30 内不存在严格更差
bad 而被跳过。epoch010 train loss 为 `0.6642`，pair accuracy 为 `58.60%`。这只是加权
训练目标的拟合指标，不能视为 alpha。

### 未加权横截面验证

验证集保持原有、不加 head weighting 的 170 日期 / 545,891 行全市场集，市场平均未来 5d
收益为 `+0.626%`。因此下表不是训练采样本身的重复统计。

| Checkpoint | RankIC | 正 RankIC 日期 | Top1 平均 / 超额 | Top5 平均 / 超额 | Top10 平均 / 超额 |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch005 | -0.0046 | 50.59% | +1.483% / +0.857% | +1.007% / +0.382% | +0.893% / +0.267% |
| epoch010 | +0.0024 | 52.94% | +1.240% / +0.614% | +1.165% / +0.539% | +0.847% / +0.221% |

与此前强 fan-out 版本相比，本实验第一次同时得到 Top1/5/10 正超额；但 RankIC 仅接近零，
说明收益信号仍集中于高分尾部而不是可靠的全截面线性排序。epoch005 的 Top1 验证更高、
epoch010 的 Top5 验证更高，不能据此在同一 OOS 验证集上无偏地选择 checkpoint。

### 全量 Score 工件

epoch005 与 epoch010 都使用 8 GPU DDP 对完整日期横截面导出 `reward_score_5d`。每卡自动
batch 为 `22,784`，峰值约 `27.54GB`，每卡推理约 36-49 秒。两份工件均为 6,788,182 行、
1,552 个日期、5,173 个标的；实际可形成完整未来 5d target 的最后 signal date 是
`2026-06-02`。

```text
run root:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-global-top30-hard-top100-nogap-bs1952-8gpu-10ep-20260719-221823/

checkpoints:
  checkpoints/reward_epoch_005.pt
  checkpoints/reward_epoch_010.pt

score artifacts / mandatory manifests:
  score_artifacts/reward_epoch005_20200102_20260715.parquet
  score_artifacts/reward_epoch005_20200102_20260715.json
  score_artifacts/reward_epoch010_20200102_20260715.parquet
  score_artifacts/reward_epoch010_20200102_20260715.json

formal QuantX summary:
  market_all_close2close_balanced_v2/reports/
  reward_transformer_top30_hard_5d_quantx_summary_20260719.json
  reward_transformer_top30_hard_5d_quantx_summary_20260719.csv
```

### QuantX 正式回测

8 个生成的 YAML 均先通过 dry-run。正式回测固定为全 A、同日 score、`lag: 1`、T+1 close、
5 个交易日 time stop、5bp 佣金、1bp 卖出印花税、10bp 滑点、1 亿初始资金。每个 checkpoint
都运行 Top1、Top5、Top10 与 Bottom20；Bottom20 是同日低分方向控制，不是与 Top5/10
严格同容量的反向组合。

| Checkpoint / 容量 | QuantX run id | 总收益 | 年化 | 最大回撤 | Sharpe | PF | 交易数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch005 Top1 | `20260719_224602_reward_transformer_top30_hard_epoch005_5d_top1_2020_2026_all_a` | -37.59% | -7.08% | -76.06% | -0.125 | 0.967 | 603 |
| epoch005 Top5 | `20260719_224737_reward_transformer_top30_hard_epoch005_5d_top5_2020_2026_all_a` | -27.15% | -4.82% | -60.85% | -0.118 | 0.967 | 3,067 |
| epoch005 Top10 | `20260719_224855_reward_transformer_top30_hard_epoch005_5d_top10_2020_2026_all_a` | -12.82% | -2.11% | -54.66% | -0.057 | 0.987 | 6,136 |
| epoch005 Bottom20 | `20260719_225119_reward_transformer_top30_hard_epoch005_5d_bottom20_2020_2026_all_a` | -55.32% | -11.79% | -75.28% | -0.479 | 0.912 | 12,286 |
| epoch010 Top1 | `20260719_225224_reward_transformer_top30_hard_epoch010_5d_top1_2020_2026_all_a` | +4.21% | +0.64% | -93.39% | 0.010 | 1.007 | 609 |
| epoch010 Top5 | `20260719_225404_reward_transformer_top30_hard_epoch010_5d_top5_2020_2026_all_a` | **+26.65%** | **+3.75%** | -58.10% | **0.092** | 1.032 | 3,061 |
| epoch010 Top10 | `20260719_225601_reward_transformer_top30_hard_epoch010_5d_top10_2020_2026_all_a` | +21.34% | +3.06% | -52.34% | 0.085 | 1.020 | 6,128 |
| epoch010 Bottom20 | `20260719_225801_reward_transformer_top30_hard_epoch010_5d_bottom20_2020_2026_all_a` | -87.84% | -27.98% | -91.93% | -0.881 | 0.832 | 12,133 |

平均已平仓持有期约 `7.55-7.61` 个自然日，符合 5 个交易日 stop 跨越周末的预期。全部 run 的
`signal_errors` 均为空。epoch010 Top5/10 的正收益不能掩盖风险问题：PF 仅略高于 1，且
最大回撤分别为 `-58.10% / -52.34%`，不具备直接生产条件。

分年复利收益如下，2026 为截至 2026-06-02 的部分年度：

| 策略 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch005 Top1 | -23.83% | +169.36% | -43.79% | -0.94% | +20.59% | -41.64% | -26.66% |
| epoch005 Top5 | -25.78% | +28.88% | -36.51% | +36.17% | +12.53% | +13.66% | -34.82% |
| epoch005 Top10 | -23.91% | +33.02% | -14.61% | +35.93% | -17.27% | -1.05% | -15.85% |
| epoch005 Bottom20 | +30.85% | -17.68% | -44.87% | -20.19% | -9.19% | -1.27% | +5.15% |
| epoch010 Top1 | -47.42% | -1.44% | -37.69% | -34.10% | +117.87% | +116.42% | -2.54% |
| epoch010 Top5 | -34.11% | +23.37% | -10.19% | +15.78% | +9.31% | +65.06% | -20.70% |
| epoch010 Top10 | -7.91% | +29.51% | -16.84% | +4.65% | +8.15% | +27.48% | -18.93% |
| epoch010 Bottom20 | -14.38% | -16.98% | -50.24% | -37.27% | -36.74% | -21.33% | +7.92% |

### 结论、反事实与限制

1. **epoch010 的 Top5/10 存在正式执行后的正向 alpha 证据**：Top5 `+26.65%`、Top10
   `+21.34%`，同时同日 Bottom20 为 `-87.84%`。这不是任意高换手全 A 组合都赚钱的现象，
   score 方向明显重要。
2. **更高 validation Top1 不是交易质量保证**：epoch005 验证 Top1 超额 `+0.857%`，但其
   QuantX Top1/5/10 全部亏损；epoch010 的验证 Top5 更高，且正式 Top5/10 才转正。训练
   checkpoint 和尾部容量必须由独立选择区间决定，不能由同一个 2020-2026 验证/回测区间的
   最高数字后验挑选。
3. **Top1 不能交易**：epoch010 Top1 虽小幅正收益，但 `-93.39%` 回撤、Sharpe `0.010`，
   这不是可接受的 concentrated alpha。当前可进一步研究的容量仅是 Top5/10。
4. **本轮尚不能证明 Top30 hard-negative 训练优于旧的全局 bad pool**：二者训练采样、
   checkpoint、回测终点都不同，且这里只有一个 seed。新的 pair 规则成功改善了 Top5/10
   的本轮结果，但没有完成架构或采样优越性的因果证明。

**重要可比性限制**：本轮 QuantX config 的 `end` 设为 score 的最后可用 signal date
`2026-06-02`，没有像此前部分实验一样额外保留 5 个交易日的结算窗口。因此最后一批持仓仍按
终点市值计入，且本轮绝对收益不能与 `end=2026-06-16` 的旧 run 做严格数值比较。后续正式
基线应把 data end 延至至少 5 个交易日之后、保持 score 缺失日不新开仓，以结清最后一批
持仓；重跑后再用该一致协议进行跨实验比较。

下一步不应继续挖更多 score。应固定 epoch010 Top5 与 Top10 的 parquet、日期、成本和股票池，
预注册并只改变风险变量：市场 regime、波动率、单日换手、持仓上限/止损和流动性/ST 约束。
每个变体都需要同一基线、同日低分反事实和分年表；不得只针对 2025 的收益调参。

## 2026-07-20 全量上涨覆盖 + 全局 1:1:1 + 严格日内 Gap：3d Reward Transformer 20 Epoch 正式闭环

### 数据合同修正

本轮的关键修正不是“每天各抽三分之一”，而是以下两个独立约束：

1. **上涨 good 全量覆盖**：训练集 2,078 个交易日中，所有满足未来 3d 相对收益大于 `+5%` 且可构造
   同日更差 bad 的上涨股票都必须进入每一个 epoch。实际为 `350,427` 条，
   `up_rows_skipped_missing_counter_class=0`，不再因某日缺少下跌/震荡反类而丢弃上涨样本。
2. **全局类别平衡，不是逐日平衡**：range 与 down good 从全训练区间的候选中按日期顺序循环抽取，
   各取 `350,427` 条，故 pair label 全局精确为 `up:range:down = 1:1:1`。这不要求任何单日天然
   三类各占三分之一；range/down 可在跨日期的候选池中重复，up 不重复也不遗漏。
3. 每一条 pair 的 good 与 bad 仍严格来自**同一个 signal_date**。所有类别都要求
   `bad_return < good_return - 0.005`；没有取消 gap，也没有把不同日期股票拿来配对。

冻结 AE 为 `40,427,736` 参数，Reward Transformer 为 `49,727,233` 参数
(`d_model=768`、7 层、12 heads)。6 GPU DDP、每卡 batch `1,952`、有效 batch `11,712`，
训练 20 epoch，每 epoch 保存 checkpoint，TensorBoard 与 JSONL 均已落盘。每 epoch
`1,051,281` pair，吞吐约 `29.9k pair/s`。

### 同口径横截面验证

验证集固定为 170 个日期、545,891 行的后期全市场截面，市场平均未来 3d 收益为 `+0.342%`。
下表的超额是每日 TopK 相对当天全市场均值，不能替代交易回测。

| Checkpoint | RankIC | 正 RankIC 日期 | Top1 超额 | Top5 超额 | Top10 超额 |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch005 | 0.0449 | 61.18% | +0.025% | +0.020% | +0.005% |
| epoch010 | 0.0625 | 69.41% | +0.077% | +0.254% | +0.268% |
| epoch015 | 0.0755 | 77.65% | -0.152% | -0.049% | +0.007% |
| epoch020 | 0.0769 | 77.06% | +0.218% | +0.261% | +0.207% |

epoch015 是本轮必须保留的反例：它有更高的 RankIC 和更多正 RankIC 日期，但 Top1/Top5
横截面超额为负，后续正式 QuantX Top1 也为负。因此模型选择不能只看 pair loss、RankIC 或
全截面平均拟合，必须同时检查容量结构。

### 必需的全量 Score 工件

四个预注册 checkpoint 都用 6 GPU DDP 对完整交易日横截面导出 `reward_score_3d`，每份均为
`6,788,182` 行、1,552 个 signal date、5,173 个标的，并写入 parquet、manifest 和离线指标。
自动 batch 最终为每卡 `22,784`，峰值 `29.58GB`（87.8%），每份导出约 2 分钟。

```text
run root:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-uniform-global-strictgap-3d-bs1952-6gpu-20ep-20260719-232608/

artifacts:
  score_artifacts/reward_epoch005_20200102_20260715.parquet + .json
  score_artifacts/reward_epoch010_20200102_20260715.parquet + .json
  score_artifacts/reward_epoch015_20200102_20260715.parquet + .json
  score_artifacts/reward_epoch020_20200102_20260715.parquet + .json
```

虽然请求范围至 2026-07-15，能够形成完整未来 target 的实际最后 signal date 是 `2026-06-02`。
正式 QuantX 将 data end 固定在 `2026-06-16`，并以 `missing: drop` 阻止 score 缺失日新开仓，
给最后一批 5 个交易日持仓留下结算窗口。这修正了此前部分 run 在 signal 末日直接截断的问题。

### QuantX 正式回测

20 个 YAML 均先通过 dry-run。正式条件完全固定：全 A、持久化外部 score、同日排序、`lag: 1`、
T+1 close、5 个交易日 time stop、5bp 佣金、1bp 卖出印花税、10bp 滑点、1 亿初始资金。
每个 checkpoint 同时运行 Top1/5/10/20 与同日 Bottom20；Bottom20 是方向性反事实，不是
风险中性的空头组合。

表内为“总收益 / 最大回撤 / Sharpe / PF”。

| Checkpoint | Top1 | Top5 | Top10 | Top20 | Bottom20 |
| --- | --- | --- | --- | --- | --- |
| epoch005 | +116.08% / -44.59% / 0.36 / 1.22 | +61.83% / -44.52% / 0.30 / 1.10 | +47.75% / -41.89% / 0.27 / 1.08 | +89.30% / -35.50% / 0.45 / 1.13 | -98.36% / -99.14% / -1.01 / 0.79 |
| epoch010 | +124.57% / -40.96% / 0.34 / 1.12 | +85.72% / -50.54% / 0.34 / 1.09 | +113.69% / -47.56% / 0.45 / 1.14 | +112.13% / -42.47% / 0.47 / 1.13 | -99.18% / -99.47% / -1.21 / 0.74 |
| epoch015 | -61.81% / -80.97% / -0.40 / 0.89 | +24.76% / -48.16% / 0.13 / 1.04 | +53.84% / -42.80% / 0.27 / 1.07 | +83.55% / -42.65% / 0.39 / 1.11 | -99.76% / -99.84% / -1.37 / 0.71 |
| epoch020 | +176.20% / -54.83% / 0.40 / 1.13 | +77.46% / -50.81% / 0.29 / 1.08 | **+187.42% / -44.09% / 0.60 / 1.18** | +100.19% / -47.45% / 0.41 / 1.11 | -99.84% / -99.88% / -1.44 / 0.69 |

平均已平仓持有期为 `7.53-7.58` 个自然日，符合 5 个交易日 stop 跨越周末的预期。Top1/5/10/20
的平均持仓数分别约为 `0.98/4.96/9.92/19.85`，因此 Top1 与 Top10 的巨大收益差异不是由
“买入超市”造成的。所有 20 个 run 的正式细项与 run id 在：

```text
quantx_formal/reward_3d_quantx_summary_20200102_20260616.json
quantx_formal/reward_3d_quantx_summary_20200102_20260616.csv
quantx/configs/strategies/generated/reward_transformer_uniform_global_strictgap_3d_formal/
```

分年复利收益如下，2026 为截至 2026-06-16 的部分年度：

| 策略 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch005 Top1 | -3.63% | +1.26% | +9.76% | +1.17% | -5.35% | +70.79% | +8.73% |
| epoch005 Top5 | -9.00% | +29.03% | +15.27% | -4.14% | -9.96% | +56.11% | -15.62% |
| epoch005 Top10 | -0.28% | +34.60% | -2.70% | -9.48% | +3.53% | +32.57% | -13.31% |
| epoch005 Top20 | +7.40% | +31.23% | +7.68% | -2.42% | +13.59% | +17.02% | -7.34% |
| epoch005 Bottom20 | -28.04% | -39.45% | -71.42% | -61.84% | -62.03% | -18.09% | -2.74% |
| epoch010 Top1 | +6.76% | +18.98% | +26.61% | +45.41% | +46.27% | -17.75% | -17.61% |
| epoch010 Top5 | +9.27% | +47.05% | +11.71% | -10.88% | +14.87% | +14.41% | -16.21% |
| epoch010 Top10 | -5.63% | +46.24% | +14.56% | -6.55% | +6.14% | +50.74% | -14.95% |
| epoch010 Top20 | +6.80% | +44.55% | +4.96% | -4.96% | +5.94% | +34.24% | -7.62% |
| epoch010 Bottom20 | -41.15% | -39.96% | -74.64% | -63.84% | -71.46% | -23.98% | +10.52% |
| epoch015 Top1 | -22.43% | +143.44% | -29.80% | -10.50% | -45.16% | -26.68% | -22.15% |
| epoch015 Top5 | -11.99% | +60.70% | -6.81% | -4.85% | -17.16% | +28.64% | -11.24% |
| epoch015 Top10 | -9.56% | +65.13% | -2.53% | -6.25% | -9.24% | +30.45% | -10.22% |
| epoch015 Top20 | +2.17% | +46.90% | -5.18% | -7.14% | +9.74% | +28.50% | -6.17% |
| epoch015 Bottom20 | -52.41% | -45.48% | -80.72% | -70.34% | -71.28% | -48.09% | -0.02% |
| epoch020 Top1 | +11.51% | +111.84% | +16.13% | -12.82% | +47.25% | -23.03% | -9.80% |
| epoch020 Top5 | +8.96% | +45.97% | +6.40% | -1.84% | +2.71% | +20.53% | -19.11% |
| epoch020 Top10 | +7.09% | +54.50% | -1.22% | +4.19% | +21.07% | +40.18% | -5.75% |
| epoch020 Top20 | +1.27% | +56.05% | +1.14% | -4.67% | +4.73% | +29.66% | -8.14% |
| epoch020 Bottom20 | -58.55% | -41.28% | -80.75% | -74.38% | -75.21% | -46.13% | -8.90% |

### 结论与反事实

1. **“上涨全取”是实质性数据修正，不是样本量小幅变化**：此前逐日平衡会在缺少反类的日期跳过
   上涨 good；本轮把全部 350,427 个 eligible up 固定进入 epoch，再只让 range/down 在全局补齐。
   该构造在全部四个 checkpoint 都给出 Top 组合显著优于同日 Bottom20 的方向分离。
2. **score 不是只在个别年份偶然赚钱**：epoch020 Top10 在 2020、2021、2023、2024、2025 都为正，
   总收益 +187.42%、PF 1.18；epoch010 Top1 则在 2020-2024 大多为正。这与同 score 的 Bottom20
   在所有 checkpoint 接近归零形成强反事实，说明分数方向确实携带信息。
3. **但 score 的尾部结构随 epoch 和容量变化**：epoch015 的 RankIC 最高之一，却 Top1 -61.81%；
   epoch020 Top10 最强而 Top5 仅 +77.46%。因此“更好地拟合整体同日排序”与“更准确地区分头部
   1/5/10 名”是不同目标，不能用单个训练/验证指标替代容量回测。
4. **本轮尚不是无偏的生产 alpha 宣称**：checkpoint `{5,10,15,20}`、容量 `{1,5,10,20}` 都在同一
   2020-2026 OOS 区间上被观察，选出 epoch020 Top10 会引入 checkpoint/容量选择偏差。下一轮必须
预先使用一个模型选择子区间，或使用 walk-forward 选择 checkpoint，再在未参与选择的后段做一次
最终 QuantX；同时至少保留同日 Bottom20 与年度表作为反事实。

## 2026-07-20 旧式日期内 1:1:1 严格 Gap 3d Reward Transformer：20 Epoch 正式反证

### 实验协议

本实验按“旧方式”重新运行，**不使用**全局 class balance、上涨全取保留、head fan-out、Top-N
hard negative 或 gap-free 规则。训练 pair 的唯一合同是：

1. 对每个可训练 signal date，使用未来 3d endpoint return 划分 `up/range/down`。
2. 当日所有三个类别都有合格样本时，以当日 eligible up 的数量为基准，range/down 在同日循环抽样到
   相同数量。因此每个可训练日期的 pair 标签严格为 `up:range:down = 1:1:1`。
3. good 与 bad 必须来自同一日期，并且没有例外地满足
   `bad_return < good_return - 0.005`。没有任何 top-rank 样本取消 gap。

完整 preflight 结果为 `same_date=true`、`gap_rule_respected=true`、
`pair_rows_exact_1_over_3=true`。共 1,918 个可训练日、每 epoch `847,749` 个 pair；6 GPU DDP、
冻结 AE `40,427,736` 参数、Reward Transformer `49,727,233` 参数，训练 20 epoch，每轮 checkpoint、
TensorBoard 与 JSONL 均已保存。这个协议不追求跨日期覆盖率，只复现日期内均匀采样的旧口径。

### 验证与全量 Score

170 个日期 / 545,891 行的固定验证集上，所有 checkpoint 的 RankIC 都低，但 TopK 3d 超额为正：

| Checkpoint | RankIC | Top1 超额 | Top5 超额 | Top10 超额 |
| --- | ---: | ---: | ---: | ---: |
| epoch005 | 0.0145 | +0.647% | +0.317% | +0.398% |
| epoch010 | 0.0138 | +0.134% | +0.298% | +0.254% |
| epoch015 | 0.0140 | +0.289% | +0.173% | +0.177% |
| epoch020 | 0.0271 | +0.419% | +0.145% | +0.259% |

四个 checkpoint 都以 6 GPU DDP 导出完整全市场 `reward_score_3d` parquet 与 manifest。每份均为
6,788,182 行、1,552 个 signal date、5,173 个标的，实际最后 signal date 为 `2026-06-02`；
QuantX 的 data end 固定为 `2026-06-16`，从而结算最后一批 5 个交易日持仓。

### QuantX 正式回测

12 个 YAML 均通过 dry-run，且 12 个正式 run 的 `signal_errors` 均为空。所有条件固定为全 A、同日
持久化 score、`lag:1`、T+1 close、5 个交易日 time stop、5bp 佣金、1bp 卖出印花税、10bp 滑点；
只改变 checkpoint 和 Top1/5/10 容量。

| Checkpoint | Top1 总收益 / 回撤 / Sharpe / PF | Top5 总收益 / 回撤 / Sharpe / PF | Top10 总收益 / 回撤 / Sharpe / PF |
| --- | --- | --- | --- |
| epoch005 | +15.46% / -88.72% / 0.035 / 1.019 | -46.89% / -79.60% / -0.209 / 0.944 | -33.60% / -75.23% / -0.153 / 0.962 |
| epoch010 | -84.18% / -87.32% / -0.426 / 0.815 | -74.61% / -86.18% / -0.455 / 0.856 | -47.22% / -74.46% / -0.247 / 0.932 |
| epoch015 | -70.71% / -89.56% / -0.278 / 0.824 | -38.83% / -68.36% / -0.170 / 0.942 | -49.42% / -72.39% / -0.261 / 0.925 |
| epoch020 | -66.20% / -71.06% / -0.263 / 0.897 | -63.45% / -79.41% / -0.361 / 0.891 | -48.26% / -79.11% / -0.260 / 0.925 |

唯一正总收益为 epoch005 Top1，但年化仅 `+2.25%`、最大回撤 `-88.72%`、PF `1.019`，不能视为
可交易 alpha。四个 checkpoint 的 Top5 和 Top10 均为负，故本协议下不存在可选的正向容量。

分年收益如下，2026 为截至 2026-06-16 的部分年度：

| 策略 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 部分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch005 Top1 | -30.44% | +49.65% | -39.63% | +50.57% | -51.18% | +127.04% | +18.71% |
| epoch005 Top5 | -13.59% | +36.74% | -17.35% | -25.78% | -32.44% | -2.02% | +11.28% |
| epoch005 Top10 | -1.91% | +30.89% | -24.80% | -29.71% | -23.86% | +19.16% | +7.90% |
| epoch010 Top1 | -19.77% | -3.41% | -57.41% | -17.70% | -30.57% | -21.41% | -13.95% |
| epoch010 Top5 | -16.49% | -13.21% | -49.89% | -27.99% | -3.33% | +27.56% | -25.57% |
| epoch010 Top10 | -9.41% | +10.10% | -38.83% | -9.60% | -24.05% | +41.54% | -13.21% |
| epoch015 Top1 | -58.21% | -39.31% | -45.96% | +63.27% | +17.57% | +87.70% | -41.00% |
| epoch015 Top5 | -28.28% | -32.38% | +20.01% | +3.93% | -19.38% | +72.84% | -33.46% |
| epoch015 Top10 | -26.21% | -14.45% | -8.18% | -11.07% | -2.89% | +19.94% | -18.62% |
| epoch020 Top1 | -6.68% | -18.70% | -24.63% | -15.40% | -12.35% | +8.22% | -35.57% |
| epoch020 Top5 | -25.07% | -11.24% | -29.90% | -23.95% | -1.02% | +67.85% | -41.57% |
| epoch020 Top10 | -20.64% | -14.54% | -26.22% | -19.52% | +21.38% | +48.21% | -34.08% |

### 记录结论

这个严格日期内 1:1:1 + 0.5% gap 的 3d pairwise 训练可以给出正的短期离线 TopK 超额，但在相同的
全市场 QuantX 执行协议下没有转化为稳定的 Top1/5/10 收益。本次只记录这一可复现实验结果，
不基于它进一步归因或调整数据分布。

```text
run root:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-date-balanced-strictgap-3d-bs1952-6gpu-20ep-20260720-001815/

formal summary:
  quantx_formal/reward_date_3d_quantx_summary_20200102_20260616.json
  quantx_formal/reward_date_3d_quantx_summary_20200102_20260616.csv

QuantX configs:
  quantx/configs/strategies/generated/reward_transformer_date_balanced_strictgap_3d_formal/
```

## 2026-07-20 3d 标签持有期对齐重跑：此前 H5 结论作废

### 必须覆盖的时间轴修正

此前两组 3d Reward Transformer 的正式 QuantX 配置都错误使用了
`holding_days >= 5`。这不是 3d 交易标签的执行协议：target contract 明确为
`close(T+1..T+30) / close(T+1) - 1`，所以 `rank_horizon=3` 的 endpoint 是
`close(T+3) / close(T+1) - 1`。在 QuantX 中 score 于 `T` 产生、`lag:1` 后于
`T+1` close 买入；当日收盘记为 holding day 1，`T+3` 开始时为 2。因此严格对齐的卖出规则
必须是 `holding_days >= 2`，在 `T+3` close 卖出。

原先的 H5 配置实际在 `T+6` close 卖出，属于“3d score + 5 个交易日持有”，不能再作为
3d 标签预测能力或 3d alpha 的证据。本文档此前两个 3d 正收益表中的 H5 结果保留作历史记录，
但后续引用必须以本节 H2 对齐结果为准。

### 执行与逐笔校验

没有重训或重推理，直接复用两个实验的完整 score parquet 与 manifest，只创建独立 H2 配置，
其余条件完全固定：全 A、同日横截面排序、`lag:1`、close 成交、相同成本、`2020-01-02` 至
`2026-06-16`、`missing:drop`、相同 TopK 和同日 Bottom20 反事实。

1. 全局上涨全取 + 全局 1:1:1 训练：4 checkpoint x `Top1/5/10/20/Bottom20`，20 个 run。
2. 日期内 1:1:1 训练：4 checkpoint x `Top1/5/10`，12 个 run。
3. 32 个 YAML 全部通过 dry-run，32 个正式 run 的 `signal_errors` 均为空，且都有
   `summary.json`、`metrics.json`、`trades.json`、`daily_nav.json`。
4. 对全局 epoch020 Top10 的 7,648 笔实际平仓逐笔按交易日审计：7,638 笔（99.87%）恰为
   `T+1 -> T+3` 的 2 个交易日间隔；其余 10 笔因跌停等卖出受限延后至 3/4/6/12 个交易日。
   因而 H2 规则的实际执行与 target endpoint 对齐。指标中的约 3.02 是自然日平均值，包含周末，
   不是 3 个交易日持有。

新配置位于：

```text
quantx/configs/strategies/generated/reward_transformer_3d_h2_aligned_formal/
```

### 全局上涨全取 + 全局平衡：H2 正式结果

表内为“总收益 / 最大回撤 / Sharpe / PF”。所有 Top 组合均为负；最佳为 epoch010 Top5，
总收益 `-43.79%`、年化 `-8.54%`、最大回撤 `-65.40%`、Sharpe `-0.28`、PF `0.95`。

| Checkpoint | Top1 | Top5 | Top10 | Top20 | Bottom20 |
| --- | --- | --- | --- | --- | --- |
| epoch005 | -71.52% / -78.50% / -0.58 / 0.82 | -59.67% / -68.62% / -0.53 / 0.89 | -52.22% / -64.17% / -0.47 / 0.90 | -47.59% / -59.37% / -0.43 / 0.92 | -99.59% / -99.72% / -1.31 / 0.77 |
| epoch010 | -79.27% / -80.56% / -0.59 / 0.80 | -43.79% / -65.40% / -0.28 / 0.95 | -56.81% / -68.48% / -0.45 / 0.91 | -60.48% / -68.17% / -0.51 / 0.90 | -99.76% / -99.81% / -1.50 / 0.80 |
| epoch015 | -67.14% / -83.81% / -0.46 / 0.93 | -52.56% / -63.79% / -0.41 / 0.93 | -58.22% / -66.93% / -0.49 / 0.90 | -63.39% / -69.56% / -0.58 / 0.89 | -99.94% / -99.95% / -1.73 / 0.74 |
| epoch020 | -67.63% / -75.42% / -0.41 / 0.90 | -54.85% / -66.50% / -0.37 / 0.92 | -60.04% / -72.84% / -0.45 / 0.91 | -57.33% / -69.33% / -0.45 / 0.91 | -99.95% / -99.95% / -1.78 / 0.73 |

此前最强的 epoch020 Top10 从 H5 的 `+187.42%` 变为 H2 的 `-60.04%`；其每笔平均已平仓收益为
`-0.0765%`，共 7,648 笔平仓、总成本约 5,491 万元。此前 H5 平均持有约 7.55 个自然日，
H2 约 3.02 个自然日，收益变化不能解释成小幅参数扰动，而是截然不同的持有期。

H2 的分年收益中，最佳 epoch010 Top5 为：2020 `-9.69%`、2021 `+30.07%`、2022 `-23.06%`、
2023 `-17.45%`、2024 `-16.34%`、2025 `+5.98%`、2026 部分 `-17.43%`。此前 H5 最强
epoch020 Top10 的 H2 分年为：2020 `-20.49%`、2021 `+4.69%`、2022 `-13.07%`、2023 `-27.18%`、
2024 `-26.35%`、2025 `+13.33%`、2026 部分 `-13.94%`。

完整汇总与年度表：

```text
market_all_close2close_balanced_v2/runs/
market-all-pre2020-ae-reward-transformer50m-uniform-global-strictgap-3d-bs1952-6gpu-20ep-20260719-232608/
  quantx_formal_h2_aligned/reward_3d_h2_aligned_quantx_summary_20200102_20260616.json
  quantx_formal_h2_aligned/reward_3d_h2_aligned_quantx_summary_20200102_20260616.csv
```

### 日期内 1:1:1：H2 正式结果

表内同样为“总收益 / 最大回撤 / Sharpe / PF”。12 组均无正收益；最佳为 epoch015 Top5，
总收益 `-77.16%`、年化 `-20.44%`、最大回撤 `-81.70%`、Sharpe `-0.48`、PF `0.91`。

| Checkpoint | Top1 | Top5 | Top10 |
| --- | --- | --- | --- |
| epoch005 | -91.50% / -98.14% / -0.51 / 0.79 | -86.04% / -90.29% / -0.62 / 0.88 | -82.97% / -89.81% / -0.62 / 0.89 |
| epoch010 | -94.09% / -96.57% / -0.59 / 0.84 | -87.99% / -89.18% / -0.68 / 0.87 | -78.02% / -82.66% / -0.55 / 0.89 |
| epoch015 | -83.43% / -87.59% / -0.40 / 0.89 | -77.16% / -81.70% / -0.48 / 0.91 | -81.70% / -85.55% / -0.61 / 0.89 |
| epoch020 | -94.76% / -95.07% / -0.62 / 0.82 | -81.69% / -83.81% / -0.56 / 0.89 | -84.32% / -87.27% / -0.69 / 0.87 |

此前该协议唯一的 H5 正收益 epoch005 Top1 从 `+15.46%` 变为 `-91.50%`。最佳 H2
epoch015 Top5 的分年收益为：2020 `-15.59%`、2021 `-16.95%`、2022 `-28.99%`、2023 `-20.98%`、
2024 `-31.63%`、2025 `+0.39%`、2026 部分 `-18.42%`。

完整汇总与年度表：

```text
market_all_close2close_balanced_v2/runs/
market-all-pre2020-ae-reward-transformer50m-date-balanced-strictgap-3d-bs1952-6gpu-20ep-20260720-001815/
  quantx_formal_h2_aligned/reward_date_3d_h2_aligned_quantx_summary_20200102_20260616.json
  quantx_formal_h2_aligned/reward_date_3d_h2_aligned_quantx_summary_20200102_20260616.csv
```

### 对齐后的结论

1. 在当前成本和 close-to-close 执行下，这两组模型都没有可交易的 3d alpha。此前的正收益来自
   与监督 endpoint 不一致的 5 个交易日持有期，不能作为模型 3d 预测成功的结论。
2. Bottom20 仍在所有 checkpoint 接近归零，说明 score 对非常差的横截面尾部仍有方向区分；但这不等于
   头部存在足以覆盖 2 个交易日高换手成本的正收益。每轮完整买卖的显式成本约为 31bp（不含最小佣金），
   而当前 Top 头部净 edge 不足。
3. 因此下一步如继续研究应预注册“标签 endpoint、成交时点、持有期、成本”四者的一致合同，先用
   同一 H2 协议验证离线 TopK 的净收益能否超过成本，再讨论采样、pair hard-negative 或模型结构的改进。

## 2026-07-20 3d Score 的 H3 邻接持有期敏感性：收益从标签外第一天出现

### 实验定位与执行合同

H2 的严格对齐结果全为负后，补充预注册 H3 敏感性实验。这里的 H3 不是新的监督标签：仍使用完全相同的
`reward_score_3d` parquet，而是把持有期从 `T+1 -> T+3` 延长一日到 `T+1 close -> T+4 close`。
QuantX 中的实现为 `holding_days >= 3`。因此它衡量“3d score 是否在标签外第一日仍有可交易的
持续性”，不能替代 H2 作为 3d endpoint 的严格验证。

两组训练协议均完整运行，且只改这一项：

1. 全局上涨全取 + 全局 1:1:1：4 checkpoint x `Top1/5/10/20/Bottom20`，20 组。
2. 日期内 1:1:1：4 checkpoint x `Top1/5/10`，12 组。
3. 所有 32 个 YAML 先通过 dry-run，32 个正式 QuantX run 的 `signal_errors` 都为空；score、
   `lag:1`、close 成交、全 A、成本、日期和仓位规则均与 H2/H5 相同。
4. 对全局 H3 最强的 epoch015 Top5 逐笔按交易日审计：2,573 笔完整平仓中 2,571 笔（99.92%）
   为 3 个交易日间隔，另 2 笔因交易约束延后到 4 个交易日；指标中约 4.53 是跨周末后的自然日均值。

配置与正式汇总：

```text
quantx/configs/strategies/generated/reward_transformer_3d_h3_adjacent_formal/

market-all-pre2020-ae-reward-transformer50m-uniform-global-strictgap-3d-bs1952-6gpu-20ep-20260719-232608/
  quantx_formal_h3_adjacent/reward_3d_h3_adjacent_quantx_summary_20200102_20260616.json
  quantx_formal_h3_adjacent/reward_3d_h3_adjacent_quantx_summary_20200102_20260616.csv

market-all-pre2020-ae-reward-transformer50m-date-balanced-strictgap-3d-bs1952-6gpu-20ep-20260720-001815/
  quantx_formal_h3_adjacent/reward_date_3d_h3_adjacent_quantx_summary_20200102_20260616.json
  quantx_formal_h3_adjacent/reward_date_3d_h3_adjacent_quantx_summary_20200102_20260616.csv
```

### 全局上涨全取 + 全局平衡：H3 正式结果

表内为“总收益 / 最大回撤 / Sharpe / PF”。H3 下已有多个 Top 组合转正，最佳为 epoch015 Top5：
`+29.72% / -50.45% / 0.15 / 1.03`。但 Sharpe 和 PF 仍弱，不能直接视为生产 alpha。

| Checkpoint | Top1 | Top5 | Top10 | Top20 | Bottom20 |
| --- | --- | --- | --- | --- | --- |
| epoch005 | -9.64% / -45.11% / -0.05 / 0.98 | +3.89% / -47.13% / 0.02 / 1.01 | -3.42% / -46.45% / -0.02 / 0.99 | +30.77% / -41.11% / 0.19 / 1.04 | -99.30% / -99.56% / -1.19 / 0.80 |
| epoch010 | -25.88% / -66.87% / -0.11 / 0.96 | +28.55% / -50.70% / 0.13 / 1.03 | -7.30% / -56.80% / -0.04 / 0.99 | -13.05% / -53.85% / -0.08 / 0.98 | -99.44% / -99.56% / -1.30 / 0.78 |
| epoch015 | -22.81% / -68.63% / -0.12 / 0.97 | +29.72% / -50.45% / 0.15 / 1.03 | +0.62% / -53.96% / 0.00 / 1.00 | +8.30% / -52.13% / 0.05 / 1.01 | -99.92% / -99.93% / -1.62 / 0.70 |
| epoch020 | -6.07% / -68.85% / -0.02 / 0.99 | -11.10% / -59.49% / -0.06 / 0.98 | -16.44% / -57.04% / -0.09 / 0.98 | -1.13% / -53.53% / -0.01 / 1.00 | -99.91% / -99.93% / -1.60 / 0.71 |

epoch015 Top5 的 H3 分年收益为：2020 `+9.26%`、2021 `+56.46%`、2022 `-10.25%`、
2023 `-16.55%`、2024 `-9.69%`、2025 `+16.53%`、2026 部分 `-9.29%`。正收益并非只来自 2025，
但 2022-2024 连续亏损和 `-50.45%` 回撤说明它尚不稳健。

### 日期内 1:1:1：H3 正式结果

除 epoch015 Top1 的弱正外其余组合仍亏损。该唯一正例为 `+9.54% / -73.99% / 0.02 / 1.01`，
风险调整后不具备可用性。

| Checkpoint | Top1 | Top5 | Top10 |
| --- | --- | --- | --- |
| epoch005 | -97.29% / -98.14% / -0.74 / 0.75 | -62.00% / -81.33% / -0.32 / 0.91 | -65.12% / -79.77% / -0.38 / 0.91 |
| epoch010 | -90.81% / -91.35% / -0.51 / 0.82 | -87.19% / -87.61% / -0.66 / 0.82 | -78.23% / -83.85% / -0.56 / 0.86 |
| epoch015 | +9.54% / -73.99% / 0.02 / 1.01 | -51.29% / -76.32% / -0.25 / 0.94 | -63.23% / -80.18% / -0.38 / 0.92 |
| epoch020 | -94.82% / -96.26% / -0.64 / 0.72 | -79.44% / -85.72% / -0.54 / 0.87 | -65.28% / -80.95% / -0.41 / 0.90 |

### H2/H3/H5 的关键对照与解释

H3 不是把 H2 的负收益随机拉高，而是在全局上涨全取协议中系统性改善了中等容量；但 H5 仍普遍更高。

| 同一 score / 容量 | H2 总收益 | H3 总收益 | H5 总收益 |
| --- | ---: | ---: | ---: |
| epoch005 Top20 | -47.59% | +30.77% | +89.30% |
| epoch010 Top5 | -43.79% | +28.55% | +85.72% |
| epoch015 Top5 | -52.56% | +29.72% | +24.76% |
| epoch020 Top10 | -60.04% | -16.44% | +187.42% |
| 日期内 epoch015 Top1 | -83.43% | +9.54% | -70.71% |

可复现结论是：在相同 score、成本和交易逻辑下，3d score 的可交易收益对持有期高度敏感，H2 不足以
覆盖成本，而部分 score 在 `T+4` 和更远仍保留正向排序信息。它支持“模型可能学到具有 3-5 日持续性
的横截面信号”，但尚不能说 reward head 准确预测了其训练的 `T+3` endpoint，更不能把 H3 的少数
正收益表述为稳健 alpha。

若继续这一支研究，下一项应是与 H3 严格匹配的 `rank_horizon=4` reward 模型，即监督
`close(T+4) / close(T+1) - 1`、同日 pair 也以 4d endpoint 构造、执行固定 H3。这样可区分
“3d score 的标签外 persistence”与“4d endpoint 本身可预测”两种解释。

## 2026-07-20 H4 Reward Transformer：endpoint 对齐的固定容量正式回测

### 模型与预注册执行合同

为验证上一节提出的 H4 假设，使用冻结 AE（40,427,736 参数）和 49,727,233 参数 Reward
Transformer 重训同日 pairwise reward score。训练标签为
`close(T+4) / close(T+1) - 1`，score 为 `[0, 1]` 内的 `reward_score_4d`。训练数据为
2020 年前全市场；上涨样本全部保留，震荡和下跌循环采样至全局 `1:1:1`，bad 样本同日且严格满足
`bad_return < good_return - 0.005`。20 个 checkpoint 均已保存，本次固定评估
epoch005/010/015/020。

四份全量 score parquet 均有 manifest，覆盖 2020-01-02 至 2026-06-02 的 1,552 个信号日、
6,788,182 行和 5,173 只股票。离线仅作诊断，不能替代 QuantX：epoch020 的全市场 RankIC 为
`0.07837`，其离线 Top1/5/10/20 的 4d 横截面超额分别为
`+0.0069%/+0.0049%/+0.0222%/+0.0379%`，幅度已很小。

正式 QuantX 合同固定为：全 A、2020-01-02 至 2026-06-16、`lag:1`、close 成交、
`holding_days >= 3`，即 score 在 T 产生、T+1 close 买入、T+4 close 卖出。成本为佣金 5bp、
卖出印花税 1bp、价格滑点 10bp。每个 checkpoint 均预注册 Top1/5/10/20 与同日 Bottom20，
共 20 组；YAML 均 dry-run 通过，20/20 `signal_errors=[]`，且每组都输出
`summary.json`、`metrics.json`、`trades.json` 和 `daily_nav.json`。

本系列与此前 Reward H2/H3/H5 一致地使用固定容量单批执行定义：`selector.topk=K` 且
`max_positions=K`。因而持有期内满仓时不建立重叠的新 cohort；这是与历史结果可直接比较的
“固定 TopK 容量”策略，而不是“每日都新建一批 TopK、总仓位为 H*K”的容量定义。后者是另一个
有效但不同的策略协议，未在本节替换或混入。

### 正式结果

表内为总收益；所有 20 组均亏损。最佳为 epoch020 Top1：总收益 `-41.32%`、年化
`-7.92%`、最大回撤 `-61.17%`、Sharpe `-0.25`、PF `0.90`。因此在与既有 Reward 实验
一致的固定容量协议及当前成本下，H4 reward score 没有可交易的净 alpha。

| Checkpoint | Top1 | Top5 | Top10 | Top20 | Bottom20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch005 | -68.74% | -75.06% | -71.36% | -69.96% | -96.32% |
| epoch010 | -77.55% | -75.76% | -74.91% | -71.06% | -99.81% |
| epoch015 | -84.93% | -75.70% | -71.78% | -67.17% | -99.91% |
| epoch020 | -41.32% | -60.35% | -44.58% | -44.16% | -99.91% |

epoch020 Top1 的分年收益为：2020 `-16.37%`、2021 `-8.98%`、2022 `-11.14%`、
2023 `-18.42%`、2024 `-12.51%`、2025 `+25.75%`、2026 部分 `-3.75%`。唯一强正年份是
2025，不能抵消其他年份的持续损失。epoch020 Bottom20 近乎归零，仍说明 score 对极差尾部具有
方向性；但 Top 端净 edge 不足以覆盖成本并形成正收益。

对 epoch020 Top1/5/10/20 的逐笔交易日审计显示，恰为 3 个交易日间隔的比例分别为
`99.60%/99.22%/98.68%/97.64%`。少数超期成交来自限价和停牌等约束；指标中的约 4.5 天为自然日
均值，包含周末，不能误读为持有 4-5 个交易日。因此本节负收益不是标签 endpoint 与实际卖出日
错位造成的。

结果目录与完整年度表：

```text
quantx/configs/strategies/generated/reward_transformer_4d_h3_endpoint_aligned_formal/

market_all_close2close_balanced_v2/runs/
market-all-pre2020-ae-reward-transformer50m-uniform-global-strictgap-4d-bs1952-6gpu-20ep-20260720-112333/
  quantx_formal_h3_endpoint_aligned/
    reward_4d_h3_endpoint_aligned_quantx_summary_20200102_20260616.json
    reward_4d_h3_endpoint_aligned_quantx_summary_20200102_20260616.csv
```

## 2026-07-20 5d Reward Top50 -> 历史 Diffusion q10 二阶段重排

### 目的与严格可比性

本实验不重训任何模型。它检验已有的 5d Reward Transformer 是否可作为全市场粗排器，
再由已有 7d close-path flow diffusion 的路径下分位 score 做候选内精排。该 diffusion 使用
epoch017、16 条采样路径，`fused_score` 经工件和历史配置确认等于 `pred_abs_q10_5d`。

两个 score artifact 的 `(signal_date, instrument)` 键完整对齐：Reward epoch010 的
6,788,182 行、1,552 个日期、5,173 只股票，在 diffusion full score 中缺失 `0` 行。
因此本次无需重新推理 diffusion；融合工件持久化了 source path、source manifest、join 审计和
每日候选规则。

二阶段规则在运行前固定：每日先按 `reward_score_5d` 取 Top50，随后仅在这 50 只内按
`pred_abs_q10_5d` 排序，分别取最终 Top5、Top10、Top20。候选外写入确定性排除分数；每个
日期经审计都恰有 50 只候选。另运行 matched all-A 的 diffusion q10-only Top5/10/20/Bottom20，
以及 Reward Top50 内 q10 Bottom20 方向控制。

所有新 run 固定为全 A、2020-01-02 至 2026-06-16、`lag:1`、close 成交、
`holding_days >= 5`、与 Reward epoch010 相同成本和固定 TopK 容量。8 个 YAML 全部通过
dry-run，8/8 `signal_errors=[]`，且都有完整 QuantX 工件。历史 diffusion Top1 的高收益使用
的是 2024-2026 主板过滤、不同成本和零滑点，不能与本节 matched all-A 结果混用。

### 正式结果

表内为“总收益 / 最大回撤 / Sharpe / PF”。Reward-only 为已存在的同设置 epoch010 正式基线，
本次没有重跑。Diffusion q10-only 在 matched all-A 设置下全部亏损，说明其历史主板 Top1
结果不能直接迁移为全 A 2020-2026 的 standalone selector。

| 选择方式 | Top5 | Top10 | Top20 | Bottom20 |
| --- | --- | --- | --- | --- |
| Reward epoch010 基线 | +266.97% / -50.97% / 0.573 / 1.192 | +116.25% / -55.98% / 0.361 / 1.101 | +141.22% / -49.41% / 0.432 / 1.116 | 不适用 |
| Diffusion q10-only | -32.27% / -67.73% / -0.140 / 0.965 | -29.06% / -61.84% / -0.142 / 0.971 | -33.44% / -64.52% / -0.187 / 0.965 | -99.94% / -99.94% / -1.862 / 0.680 |
| Reward Top50 -> q10 | -10.54% / -68.94% / -0.047 / 0.987 | +54.63% / -55.44% / 0.203 / 1.055 | **+159.07% / -49.53% / 0.489 / 1.142** | +7.82% / -59.33% / 0.034 / 1.009 |

Reward Top50 -> q10 的 Top20 相比 Reward-only Top20 增加 `+17.85` 个百分点总收益，Sharpe
从 `0.432` 升至 `0.489`，PF 从 `1.116` 升至 `1.142`，最大回撤几乎不变
（`-49.41% -> -49.53%`）。其年度收益为：2020 `+0.81%`、2021 `+39.59%`、
2022 `-14.27%`、2023 `+3.59%`、2024 `+32.31%`、2025 `+46.32%`、2026 部分 `+2.68%`。

但 q10 重排显著损伤集中容量：Top5 从 `+266.97%` 变为 `-10.54%`，Top10 从 `+116.25%`
降为 `+54.63%`。而 Reward Top50 内的 q10 Bottom20 仍有 `+7.82%`，因此虽然 Top20 相对
Bottom20 的差异很大，但该候选内 q10 的方向性控制不够强，不能据此声称 q10 已形成稳定、独立的
精排 alpha。

### 结论

1. 当前 diffusion q10 不能替换 Reward score 做全 A standalone 选股；matched all-A 控制全亏。
2. 在 Reward 已压缩到 Top50 的条件下，diffusion q10 对 Top20 具有小幅正向增量，但不适用于
   Top5/Top10，说明它更像中等容量的条件性多样化/风险分布信号，而不是通用尾部精排器。
3. 本次 Top50 阈值和输出列是在已有模型和 OOS 区间下做的探索性融合验证，不能直接升级为生产
   score 或作为训练目标的无偏证据。后续若继续，应固定 Reward Top50 -> q10 Top20 为候选基线，
   只预注册测试更稳定的 diffusion checkpoint、更多路径样本或风险控制，而不是在当前同一 OOS
   区间继续搜索候选池宽度和融合权重。

### 工件

```text
fusion score / manifest:
  market_all_close2close_balanced_v2/runs/
  market-all-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-40ep-20260719-193127/
  score_artifacts/reward_epoch010_diffusion_epoch017_q10_reward_top50_rerank_20200102_20260715.parquet

QuantX configs:
  quantx/configs/strategies/generated/reward_diffusion_q10_rerank_5d_formal/

formal summary and annual table:
  .../quantx_formal_reward_diffusion_q10_top50/
  reward010_diffusion017_q10_top50_rerank_quantx_summary_20200102_20260616.json
  reward010_diffusion017_q10_top50_rerank_quantx_summary_20200102_20260616.csv
```

## 2026-07-20 主板 Reward epoch020：退出策略、市场宽度和回撤归因

### 范围与可比性

本节只记录主板 Reward Transformer epoch020 的正式 QuantX 回测。score 工件为：

```text
market_mainboard_close2close_balanced_v1/runs/
market-mainboard-pre2020-ae-reward-transformer50m-allup-5d-bs1952-8gpu-20ep-20260720-143108/
  score_artifacts/reward_epoch020_20200102_20260715.parquet
```

所有下述策略均使用同一主板股票池、同一 score、`Top5` 固定容量、`lag:1`、close 成交和同一成本合同。
它们可以相互比较，但不能与前文全 A、不同持有期或不同成本的 Reward/Diffusion 结果混用。

### 已验证的退出策略

表内为“总收益 / 年化收益 / 最大回撤 / Sharpe / PF”。`loss_exit_d5` 的含义是：第 5 个持有日仍
亏损的仓位退出；盈利仓位继续持有到目标持有期。`trail_p20_dd12` 的含义是：仓位峰值收益超过 20% 后，
若相对自身峰值回撤超过 12% 则退出。

| 策略 | 总收益 | 年化收益 | 最大回撤 | Sharpe | PF | 定位 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H15 + loss_exit_d5 | +352.58% | 26.34% | -39.79% | 0.735 | 1.315 | 原始稳定基线 |
| H30 + loss_exit_d5 | **+519.49%** | **32.63%** | -46.98% | **0.892** | **1.544** | 收益最强，但回撤最大 |
| H30 + trail_p20_dd12 | +509.36% | 32.30% | **-41.86%** | 0.877 | 1.434 | 当前风险收益最平衡 |

当前用于风险控制研究的基线为 `H30 + trail_p20_dd12`，其配置和正式 run 为：

```text
quantx/configs/strategies/generated/
reward_transformer_mainboard_epoch020_top5_exit_policy_combo_v1/
  reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026.yaml

quantx/runs/
  20260720_185520_reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026
```

此前已测试 H10/H20/H25/H30、d3/d7 亏损退出、固定止盈、硬止损、移动止盈和部分减仓等变体。固定止盈、
d3/d7、仅硬止损及较短持有期均未优于上述两类 H30 策略。H15 的 `scale12_keep85` 曾得到
`+461.94% / -40.81% / 0.868`，但仍不优于 H30 + trail 的收益风险组合。

### 最大回撤期的市场因子特征

`H30 + trail_p20_dd12` 的最大回撤为 `-41.8601%`：净值在 `2023-08-30` 见顶
`273,530,970`，于 `2024-02-07` 见底 `159,030,621`。市场统计覆盖 3,189 只主板股票和
2020-01-02 至 2026-06-16 的 1,562 个交易日。`breadth20` 使用与 QuantX 一致的定义：
`CSRatio(close > Mean(close, 20))`。

| 指标 | 全期中位数 | 最大回撤窗口中位数 | 峰值日 2023-08-30 | 谷底日 2024-02-07 |
| --- | ---: | ---: | ---: | ---: |
| 上涨家数占比 | 45.94% | 38.85% | 50.92% | 47.70% |
| breadth20 | 45.56% | 34.81% | 38.29% | 20.57% |
| 横截面 20 日动量中位数 | -0.33% | -2.38% | -4.67% | -21.93% |
| 横截面 20 日波动率 | 2.69% | 2.26% | - | 3.89% |
| 日收益均值 | +0.14% | -0.16% | - | - |
| 日收益中位数 | 0.00% | -0.26% | - | - |

结论如下：

1. 净值峰值当天，`breadth20=38.29%` 且横截面 20 日动量中位数为 `-4.67%`；市场内部趋势已明显转弱，
   早于组合净值的最大回撤。因此“低 breadth20 + 负横截面中期动量”是可用于早期风险识别的状态组合。
2. 从峰值后，`breadth20 < 35%` 在下一个交易日即出现，但 `<30%` 到 `2023-09-21` 才出现。
   较高阈值预警早却会频繁触发，较低阈值可信度更高却滞后约三周，单阈值无法兼顾及时性和机会成本。
3. `2024-01-22` 是极端踩踏日：`breadth20=5.77%`、上涨家数占比 `2.62%`、日收益均值 `-5.49%`、
   日收益中位数 `-5.87%`。它更适合危机中的损失约束，而不是单独作为提前规避回撤的信号。
4. 回撤期前段横截面波动率并不高，说明风险主要来自持续普遍走弱，而非一开始就出现高波动冲击；
   临近谷底波动率才抬升至 `3.89%`，进入踩踏阶段。

### 市场宽度仓位控制的反事实结果

首先测试了用 `max_positions_rules` 根据 breadth20 降低持股数的做法，例如 `<40% -> 3` 或
`<35% -> 3, <45% -> 4`。该做法不是有效风控：固定 `cash_use_ratio=0.98` 会将近乎相同的资金
集中于更少股票，平均持仓数约从 `4.97` 降至 `4.27`，反而提高集中风险。以 H30 + d5 为例，
`breadth20 < 40% -> 3` 得到 `+697.56%`，但最大回撤恶化至 `-56.04%`。

随后实现并测试“仅影响新买入、不强平既有仓位”的动态现金使用率：

```yaml
rebalance:
  cash_use_ratio: 0.98
  cash_use_ratio_rules:
    - when: market_breadth20 < 0.35
      value: 0.35
    - when: market_breadth20 < 0.45
      value: 0.70
```

规则按首个命中项生效，并保持 `max_positions=5`，避免降低持股数导致的集中。正式结果如下：

| 基础退出策略 | 现金规则 | 总收益 | 最大回撤 | Sharpe | 平均现金 |
| --- | --- | ---: | ---: | ---: | ---: |
| H30 + d5 | 固定 98% | +519.49% | -46.98% | 0.892 | 1.22% |
| H30 + d5 | 30% / 70% / 98% | +282.11% | -43.07% | 0.703 | 13.95% |
| H30 + d5 | 35% / 70% / 98% | +208.20% | -43.98% | 0.602 | 15.93% |
| H30 + d5 | 40% / 98% | +398.90% | -44.10% | 0.832 | 8.33% |
| H30 + trail | 固定 98% | +509.36% | -41.86% | 0.877 | 1.58% |
| H30 + trail | 30% / 70% / 98% | +247.18% | -37.18% | 0.664 | 14.32% |
| H30 + trail | 35% / 70% / 98% | +195.28% | -34.55% | 0.592 | 16.93% |
| H30 + trail | 40% / 98% | +352.82% | -39.77% | 0.785 | 8.95% |

现有证据表明：市场宽度对弱市场状态有解释力，但单独通过全局减仓无法满足“收益不下降、回撤更小、
Sharpe 不下降”的严格目标。后续任何风控变体均以 H30 + trail 固定仓位基线为门槛：
总收益必须 `>= +509.36%`，最大回撤必须优于 `-41.86%`，Sharpe 必须 `>= 0.877`。未同时满足三项者，
只能作为风险偏好不同的替代配置，不得称为策略增强。

### 后续假设：离线 RL 持仓控制（尚未实施）

下一条可验证方向不是让 RL 重新学习选股，而是固定 Reward Top5 alpha，只让策略条件化控制器管理既有
持仓。个股层的离散动作可为 `hold / reduce / exit` 或目标仓位 `0 / 25% / 50% / 100%`；组合层统一处理
现金、最大持仓数和集中度约束，避免每只股票独立决策导致的资金冲突。

候选状态包括：当前盈亏、持有天数、峰值回撤、短期收益/波动、reward score 与 rank 的变化、
模型置信度、breadth20、横截面动量、市场波动率及基础策略上下文。奖励需使用净值增量并扣除换手成本、
增量回撤和集中风险，不能只优化次日收益。

该方向必须采用离线验证并严格时间切分，例如 2020-2023 训练、2024 冻结调参、2025-2026 留出正式检验。
可将其他 checkpoint、TopK 和持有期作为训练时的策略上下文以扩大轨迹覆盖，但正式结果必须按每个基础策略
单独报告。不得在完整 2020-2026 回测路径训练后，再以同一区间的提升作为 RL 有效性的证据。

## 2026-07-20 设计冻结：Reward Top5 约束下的 RL 动态仓位管理

### 状态

本节是后续实验的设计合同，不是已经完成的训练或 QuantX 回测结果。目标不是让 RL 重新做全市场选股，
而是在不改变 Reward 模型每日候选产生机制的前提下，学习动态卖出、减仓、买入接受和现金部署。

当前目标策略是主板 Reward epoch020 Top5。其现有配置为 `selector.topk=5`、
`rebalance.max_positions=5`、`cash_use_ratio=0.98`、`buy_only_new_positions=true`、
`buy.sizing=cash_equal` 和 `reuse_sell_cash=true`。这意味着：

1. 当前策略只在完整卖出后释放一个持仓槽位；部分卖出不会释放槽位。
2. 卖出规则先生成卖单；已计划完整卖出的股票会在分配阶段视为不占槽位。
3. 若有新槽位，策略仅从当日 Reward Top5 中选择未持仓股票；同一批新买入股票按可用现金等权分配。
4. `reuse_sell_cash=true` 将预计卖出所得并入当日买入资金。回测订单语义是同一交易日 close 先卖后买，
   而非模拟日内先后价格路径。
5. 已持仓股票不会每日被重新调成等权，实际权重会随价格涨跌漂移。配置 `description` 中的“15 日上限”
   是旧文本，实际规则是 `holding_days >= 30` 的 H30 time stop。

RL 版本必须保留如下 alpha 边界：当天 Reward Top5 是唯一可买候选；RL 不得购买 rank6 以后股票、
不得提前使用未来日期的候选，也不得改变 Reward score 的生成日期、排序规则或 `lag:1` 成交合同。RL 会改变
候选是否被接受及投入多少资金，这是仓位管理的必要权限；但它不改变基础 alpha 给出候选的时点和范围。

### 决策时点

```text
T 日收盘：
  读取 T 时点可见的账户、持仓、市场因子和 Reward 当日 Top5；RL 生成 T+1 的交易计划。

T+1 日 close：
  先执行 T 日计划中的卖出/减仓；基于实际成交后的现金余额执行买入；
  按 QuantX 的成本、滑点、整手、涨跌停、停牌和自动买入数量调整处理。

T+1 日收盘后：
  更新账户与持仓状态，读取新的 Reward Top5，生成 T+2 计划。
```

RL 在 T 时点不输出绝对买入股数，因为它未知 T+1 的实际成交价、卖单是否受限和实际到账现金。它输出卖出比例、
买入资格、现金部署比例及候选间分配比例；QuantX 在 T+1 使用实际成交后的可用现金和成交价换算整手数量。
T+1 的成交价只参与执行数量计算，不作为 T 时点的模型输入。

### 输入合同

模型输入为固定大小的集合，不输入股票代码、模型 epoch、策略名称、固定持有期或“距 H30 到期的剩余天数”。
所有 score 使用同日横截面分位数和 rank 变化等标准化表示，避免不同 checkpoint 的原始分数尺度泄漏策略身份。

```text
市场向量 G_t：
  breadth5/10/20、上涨家数占比、横截面20日动量、横截面波动率、
  指数趋势、指数回撤、市场流动性。

账户向量 A_t：
  最近30个交易日的组合净收益率序列 r_(t-29), ..., r_t，
  其中 r_t = NAV_t / NAV_(t-1) - 1，且已扣除实际交易成本；
  当前已投资仓位比例 invested_ratio = 持仓市值 / 总资产；
  当前持仓数归一化 position_count / max_positions；
  当前相对历史净值峰值的回撤 portfolio_drawdown_from_peak。

持仓矩阵 H_t（最多 5 行）：
  当前仓位权重、持有天数、浮盈亏、峰值收益、相对峰值回撤、
  近1/3/5/10日相对收益、个股波动、流动性、
  入场 score 分位、当前 score 分位、score/rank 变化、是否仍在当天 Top5。

候选矩阵 C_t（严格为当天 Reward Top5）：
  candidate_rank=1..5、score/rank 标准化特征、近期收益、波动、
  流动性、相对市场收益、是否已经持仓。
```

账户向量不输入绝对 NAV、现金金额、近期成本金额或“距最大回撤峰值距离”。在无杠杆合同下，
`cash_ratio` 与 `invested_ratio` 互补，只保留后者；“当前组合回撤”和“距净值峰值距离”也合并为
`portfolio_drawdown_from_peak`。30 日净收益路径提供组合近期的连续亏损、反弹与波动状态，单一历史
峰值回撤则补足 30 日窗口之外的路径风险。该账户状态共 33 个标量维度，且不依赖初始资金规模。

当持仓与当日 Top5 重叠时，股票同时带有 `is_held=true` 和 `is_candidate=true`。第一版保持
`skip_if_holding=true` 语义：该股票只由持仓卖出/持有头管理，不允许同日加仓、卖出后回补或反手交易。

### 动作输出与硬约束

RL 使用四个动作头：

```text
1. sell_ratio_i：对每只现有持仓 i 输出 {0%, 25%, 50%, 100%}。
2. enter_gate_j：对每只“当日 Top5 且当前未持仓”候选 j 输出 {不买, 买入}。
3. post_sell_cash_deploy_ratio：输出 {0%, 25%, 50%, 75%, 100%}。
4. allocation_logits_j：对 enter_gate_j=买入 的候选输出相对资金分配权重。
```

约束在执行器中强制，而不是依靠模型自行学习：

```text
full_exit_count = sell_ratio_i == 100% 的持仓数量
free_slots      = 当前空槽位 + full_exit_count
新买入候选数  <= free_slots

部分卖出只增加现金，不增加 free_slots；若已有 5 个持仓且无完整卖出，
即使部分卖出也不能买入第 6 只股票。
```

对于允许买入的候选，T+1 的实际订单金额定义为：

```text
sell_qty_i       = round_lot(current_qty_i * sell_ratio_i)
cash_after_sell  = 原现金 + 实际卖出净额
buy_budget       = cash_after_sell * post_sell_cash_deploy_ratio
buy_cash_j       = buy_budget * softmax(allocation_logits)_j
```

实际买入数量由 `buy_cash_j / T+1_close` 向下取整到整手，并扣除成本。卖出因交易约束失败、候选涨停、
停牌或买入金额不足一手时，不以 rank6 以后股票替代，未用资金保留为现金。因而所有持仓变化都能追溯到
T 日可见的 Reward Top5 与 RL 动作。

### 训练来源与双重留出

当前主板 Reward epoch020 Top5 H30+trail 策略及其全部轨迹不进入 RL 训练池。训练只使用可审计的历史
score artifact：旧 Reward checkpoint、不同 TopN 与历史持有/退出规则，且第一期只使用语义一致的 5d
Reward score，不混入 decoder 或 diffusion 的异构 score。每个 source strategy 在 catalog 中固定以下信息：

```text
score parquet 与 manifest、score 列及方向、主板过滤规则、候选定义、
执行时点、成本合同、可用日期、基础持有和退出规则。
```

仅排除 epoch020 策略仍不足以消除时间过拟合，因为不同策略可能经历同一市场阶段。后续采用策略留出和时间
留出的 walk-forward 验证：

| Fold | RL 训练 | RL 验证，仅旧策略 | 冻结后 epoch020 Top5 正式检验 |
| --- | --- | --- | --- |
| A | 旧策略 2020-2021 | 旧策略 2022 | 2023 |
| B | 旧策略 2020-2022 | 旧策略 2023 | 2024 |
| C | 旧策略 2020-2023 | 旧策略 2024 | 2025-2026 |

epoch020 目标期只用于冻结后迁移检验。任何根据其结果修改特征、奖励权重、动作集合或 checkpoint 的行为，
都必须进入下一轮 fold，不能继续将同一目标期表述为留出结果。

### 训练和正式评估

训练轨迹不能只取旧 QuantX 基线的一条实际成交路径；需要在训练期的历史行情上构造多种受约束的行为轨迹，
包括固定持有、loss-exit、trailing、部分减仓、已固定的市场状态规则和受约束的随机动作，以覆盖
`sell/reduce/hold/buy` 的反事实状态。价格对策略近似外生，但所有状态特征和动作时点必须严格因果。

第一版模型为共享个股编码器加集合聚合：持仓和候选经共享 MLP 编码，使用 attention/DeepSets 聚合，
再与市场和账户向量融合，输出上述四个动作头。模型参数控制在 1M-5M。训练优先采用带行为约束的离散
offline RL（IQL 加行为克隆/KL 约束），避免在训练轨迹未覆盖的极端动作上外推。

正式回测必须在 QuantX 内执行，而不是只在外部快速 simulator 宣称结果。训练侧可使用快速环境生成轨迹，
但需要先对固定规则逐日核对其 NAV、成交、成本与 QuantX 一致。QuantX 正式接入需保存：

```text
冻结的 policy artifact、feature schema、normalizer、training manifest、
每次 run 的 rl_decisions.parquet（日期、股票、状态摘要、动作概率、最终动作、
目标部署比例、实际订单、模型 hash）。
```

目标策略每个 fold 至少与四类对照比较：原始 Reward Top5、冻结的规则风控、仅市场现金部署 RL、
市场加个股联合 RL。除收益、年化、最大回撤、Sharpe、PF 外，还必须报告年度收益、换手、成本、平均现金、
平均持仓数、CVaR 和市场 regime 分组结果，并审计 `saved_loss`、`missed_winner` 与 `kept_winner`。

正式通过条件按同一目标 fold 的基础策略比较：净收益不低于基线、最大回撤小于基线、Sharpe 不低于基线。
未同时满足三项者只能作为不同风险偏好的候选配置，不得表述为对基础策略的增强。

## 2026-07-20 实现记录：PPO QuantX 动态仓位系统

上述设计稿中“第一版 IQL”和“共享个股编码器/attention”的描述已被实现合同取代：当前实现使用 PPO，
并采用固定维度的 masked factorized actor-critic。四个动作头分别为持仓卖出比例 `{0,25,50,100%}`、
候选买入开关、现金部署 `{0,25,50,75,100%}` 和候选分配 Dirichlet 权重。无效持仓和已持有候选的买入动作
在概率分布中被 mask，不以无效动作训练。

训练环境 `rl_position_manager_v1/quantx_replay_env.py` 已改为直接调用 QuantX
`BacktestSession -> RuleExecution -> Executor -> Account.update_daily_balance`。它没有自定义成交、费用、
持仓或 NAV 计算。2020-01 小窗口 smoke 验证得到 `state_dim=482`，首步产生 1 个订单和 1 笔 QuantX 成交；
成交成本和 reward 均由 QuantX 账户状态返回。`PositionPlan` 的全卖释放槽位、卖出现金复用和部分卖出不回补
已加入 QuantX 回归测试。

正式接入采用 YAML 显式 `execution.position_manager` 插件。冻结 checkpoint、normalizer、训练 manifest 与
配置均会被保存，正式 QuantX run 另写 `rl_decisions.parquet`，可将 PPO 意图与 QuantX 实际 `trades.json`
逐日核对。当前基线不声明该配置时保持原执行路径不变。

工程验证截至此处为止：`test_config_strategy.py` 与 PPO plugin 测试共 `45 passed`，代码静态检查和
`git diff --check` 通过。**尚未启动 PPO 训练，尚无 PPO replay 或 2020-2026 正式 QuantX 收益结论。**
后续必须先按 2020-2023 训练、2024 旧 source 验证，再冻结策略并只对 epoch020 Top5 holdout 做正式回测；
不得用目标回测结果反向调 PPO 超参数。

## 2026-07-21 PPO 动态调仓：完整 source 验证与反事实 swap 结论

### 结论状态

截至本轮，**尚无** PPO 动态调仓策略同时满足“相对主板 Reward epoch020 Top5 H30+trail 基线：
净收益不低、最大回撤更低、Sharpe 不低”的正式 QuantX 条件。当前最佳正式基线仍为
2025-01-02 至 2026-06-16 的 `+114.67%` 总收益、`-22.37%` 最大回撤、`2.16` Sharpe。

此前的 residual v8 在目标期把回撤由 `-22.37%` 降至 `-20.43%`，但总收益从 `+114.67%`
降至 `+85.55%`，只能作为风险偏好不同的候选，不能称为增强。v7 的混合异构 source 更出现
`-37.91%` 总收益、`-50.09%` 回撤，已保留为负对照。

### v9：仅额外卖出、赢家保护

v9 固定原策略买入准入、等权和 98% 部署，仅允许 PPO 额外卖出；原 H30/trailing/loss 退出保持硬约束。
同时加入“盈利且未明显回撤的持仓禁止 PPO 提前卖出”、现金拖累和卖出后 5 日错过赢家的训练惩罚。

- 训练：旧 Reward epoch005/010/015，2020-2022；2023 随机 episode 验证。
- update010 在 2023 随机验证仅有极小正超额；2024 全年 source 审计也只对 epoch010 有
  `-11.74% -> -11.50%` 的轻微改善。
- 目标期 QuantX 逐笔决策审计显示 `sell_ratios` 全为空，结果与基线逐项相同。
- 对 update010 的 2023 全年 margin 网格：`0.25` 过度换仓，`0.50` 只在部分 source 改善，
  `0.75` 完全退化为基线。无一个门槛在三 source 等权下同时提升收益与回撤。

这说明保守残差合同本身可靠，但单靠通用 PPO actor 没有学到稳定可迁移的卖出信息。

### v10-v12：full-swap 与训练/部署特征合同修正

v10 将动作收缩为 `hold / full_exit`，禁用 25%/50% 部分卖出；残差完整退出后只能由原 Top5
未持仓候选补位。其 5 日 replacement relative return 作为局部训练奖励。但随机 episode 验证曾出现
假阳性：update020 的短验证为正，2023 全年三 source 审计显示 epoch005、015 同时损失收益与回撤，
epoch010 虽改善也未迁移到 2024（`-11.74% -> -14.61%`）。v10 淘汰。

审计进一步发现确定性实现缺陷：正式插件能够得到全截面持仓 score/rank，而 replay 为节省内存只缓存
每日 Top5，持仓跌出 Top5 后 score 特征错误置零。v11/v12 修复为每日 Top200 有界 score context：
Top5 仍是唯一交易信号，Top200 只用于判断持仓是否刚跌出候选还是已显著变差；正式插件对 rank>200
也同样置零。PPO policy contract 已升级，旧 checkpoint 不再能静默加载到新特征语义。

此外，checkpoint 选择验证改为每个旧 source 从年初到年末的完整 QuantX 路径，而非随机抽取 8 段 episode。
它在 JSONL 中保存每个 source 的累计收益、最大回撤、最差 source 和等权汇总，消除了 v10 的选择偏差。
v12 在该完整验证下所有 checkpoint 仍为负，淘汰。

### v13：counterfactual swap-advantage 辅助学习

v13 在不把未来信息放入状态的前提下，为每个 T 日可卖持仓构造训练后果标签：首个未持仓 Top5
候选从 T+1 到 T+6 的收益，减去该持仓同期收益和约 `0.22%` 双边成本。该标签监督
`swap_advantage_head`，同时以二元对齐损失监督 full-exit logit；PPO 相对基线回报、行为克隆和原卖出规则
仍保留。正式推理要求 actor margin 与预测净优势至少 `+0.5%` 同时通过。

强制 QuantX replay 已验证标签实现：卖出 `SZ002052`、买入 `SZ000717` 后，5 日原持仓
`-8.10%`、替补 `-3.74%`，在 21.2% 实际 exposure 上产生 `+0.00926` swap signal；同一时点可卖
持仓得到非零 dense advantage 标签，说明不是空奖励。

v13 update020 的 2023 三 source 等权完整验证首次同时改善聚合目标：

| 指标 | PPO | 基线 | 差异 |
| --- | ---: | ---: | ---: |
| 平均累计收益 | +25.21% | +24.43% | +0.78pct |
| 平均最大回撤 | -17.04% | -17.96% | +0.92pct |

但 source 异质性仍明显：epoch005 收益增、回撤恶化；epoch010 收益和回撤均改善；epoch015 收益降、
回撤改善。按“等权累计收益不低且平均回撤不差”预注册规则，update020 是唯一合格 checkpoint。

对完全未参与选择的 2024 三 source 全年检验，结果未通过双目标：

| source | PPO 累计 | 基线累计 | PPO 最大回撤 | 基线最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| epoch005 | -16.49% | -11.83% | -42.04% | -31.69% |
| epoch010 | +42.96% | -11.74% | -31.78% | -40.45% |
| epoch015 | +82.97% | +74.16% | -39.23% | -34.08% |

epoch010 的收益与风险同时改善，说明 counterfactual swap 学习确实能在部分 source 中形成可交易信息；
但 epoch005/015 的风险不稳定，三 source 等权 2024 回撤仍恶化。因此它不能迁移到 epoch020 target
做正式增强宣称。

### 市场宽度约束反事实

在冻结 v13 update020 上，只允许 `breadth20 >= 0.40` 或 `>=0.50` 时发生残差 full-swap；原策略弱市场
执行不受影响。两个阈值在 2023 三 source 上均使等权收益转负，未进入 2024/target。市场宽度作为单一硬门槛
不能修复 source 间的 swap 排序不稳定性。

### 工程与后续要求

1. PPO replay 仍只使用 QuantX `BacktestSession -> RuleExecution -> Executor -> Account`；没有自定义成交、
   成本、持仓或 NAV。训练 collector 使用 32 个 CPU QuantX worker 和单卡批量策略推理，实测约
   700-800 agent step/s。
2. 每轮都保留 checkpoint、TensorBoard、JSONL、source audit parquet/summary；正式 QuantX 另保存
   `rl_decisions.parquet`。当前测试为 `51 passed`。
3. 下一轮不能继续仅调 margin、breadth 或同一 target 期。需要增加训练 source 的策略/市场覆盖，或先单独
   验证 counterfactual advantage 在不同 score checkpoint 上的 RankIC、AUC、calibration 和时间稳定性；
   只有 advantage 自身在 2024 source 上可迁移，才值得再进入 PPO 或正式 target 回测。

### 修正：relative-baseline PPO 合同

最初 absolute-return PPO smoke 的 reward 仅为当日 agent 净 log-return，且 `turnover_penalty=0`、
`drawdown_penalty=0`。该版本只能证明 PPO 与 QuantX 回放路径可运行，不能用于回答“是否相对原策略提高收益
且降低回撤”，也不能作为 Top5 正式策略。它在 update045 被停止并保留为负对照。

正式训练源现固定为 11 个 score artifact、35 个 `source x TopK` 合同：主板 5d checkpoint 使用各自
Top5 H15/loss-exit QuantX YAML；全市场 3d/4d/5d/7d source 使用其各自的 all-A Top1/5/10/20 YAML。
每个 contract 显式声明 score manifest、endpoint、universe、TopK 和 baseline YAML。全市场 source 不再
被主板过滤。

relative PPO 每步同时推进两个独立 `BacktestSession`：agent 由 PPO 输出 `PositionPlan`，baseline 运行
source YAML 的原始 rebalance 与 sell rules。二者共享同日只读行情，均通过 QuantX 的 `RuleExecution`、
`Executor`、`Account` 执行。奖励冻结为：

```text
u_t = log(1+r_agent,t) - log(1+r_baseline,t)
      - 0.001 * max(0, turnover_agent,t - turnover_baseline,t)
      - 0.01 * max(0, excess_drawdown_t - excess_drawdown_(t-1))
```

其中 `excess_drawdown = max(0, abs(DD_agent) - abs(DD_baseline))`。这不是全期 max drawdown 的替代指标，
而是对新增的超基线回撤施加因果、稠密的日频惩罚；最终是否通过仍以正式 QuantX 全期总收益、最大回撤和
Sharpe 比较决定。双账户 smoke 已分别在 mainboard 和 all-A source 通过，且 PPO first minibatch
`clip_fraction=0`。

### 2026-07-21 负对照与 residual-baseline 修正

第一轮并行 PPO `ppo_relative_baseline_parallel16_mixed_2020_2023_v3_20260721` 使用 16 个 QuantX
worker 和 GPU batch policy，实测 `156-191` agent decisions/s。其 update010 的旧 source 2024 validation
看似为正：`validation_mean_excess_log_return=+0.000714`、`validation_mean_utility=+0.000603`，且平均最大
回撤优于对应 baseline。但预注册的一次 epoch010 目标 OOS QuantX 对照（2025-01-02 至 2026-06-16）明确失败：

| 策略 | 总收益 | 最大回撤 | Sharpe | 交易数 |
| --- | ---: | ---: | ---: | ---: |
| 原 Reward epoch020 mainboard Top5 H30+trail | +114.67% | -22.37% | 2.16 | 300 |
| v3 PPO | -2.12% | -25.62% | -0.06 | 1,412 |

该结果不能解释为 target 策略没有 RL 空间，工程审计找到了两个确定性错误：训练 agent session 使用了空
`sell_rules=[]`，且 formal renderer 生成 PPO YAML 时清空了原 H30/trailing/loss exits；同时随机 actor
必须从头学会 baseline 的准入、持有和退出，导致 4.7 倍交易膨胀。v3 被停止并保留为负对照，不参与后续
checkpoint 选择。

修正后的 **residual-baseline v2 合同** 为：

1. agent 和 baseline 都从同一 source YAML 构造独立 QuantX session；只将外部 score selector 缓存替换。
2. `PositionPlan.replace_sell_rules=False`；PPO 的额外卖出叠加在原卖出规则之上。原规则预览出的全卖股票
   不允许被 PPO 的部分卖出覆盖，且为其预留换仓槽位。
3. PPO 确定性初始动作是“不卖、准入所有未持仓 TopK、0.98 现金部署、等权”；模型 checkpoint 必须声明
   `policy_contract=residual_baseline_v2`，旧 replace-rule checkpoint 不能被 formal plugin 加载。
4. 渲染器保留原 `execution.sell_rules`，并以同一原 YAML 同时生成 OOS baseline 与 PPO config。

该合同已用未训练 checkpoint 在 mainboard epoch020 Top5 H30+trail 的 2025-01-02 至 2026-06-16 正式
QuantX 回测验证：baseline 与 PPO 的总收益 `+114.6653%`、年化 `69.2307%`、最大回撤 `-22.3702%`、Sharpe
`2.1614`、300 笔交易、350 条 daily NAV、307 条订单/成交记录、1,714 条持仓快照和 150 条 closed position
均逐项相同。这证明 residual 接入本身不会降低原策略。

过度保守的 residual v4（action logit gap 8）在 update002 后停止：actor head 的更新量远小于翻转
deterministic argmax 所需的 gap，无法学习有效偏离。当前正式运行的是
`ppo_residual_exploration_parallel16_mixed_2020_2023_v5_20260721`：保持上述 formal baseline 合同，但将
训练期 baseline action margin 降为 2、Dirichlet concentration 降为 10，以提供可学习的受约束探索。
截至 update003 仅有训练期探索数据，尚无 v5 validation 或目标 OOS 收益结论；完成后仍须按固定 2024
validation 选择一个 checkpoint，再运行 paired QuantX 正式回测。

## 2026-07-21 PPO 动态调仓续测：动作合同、critic overlay 与失败结论

### 工程修正与效率

本轮先完成了两个会影响结论有效性的合同修正：

1. 原 v6 PPO rollout 对 baseline action 加了 `baseline_logit_bias`，但确定性验证和正式插件没有加同一
   bias。训练的是保守行为分布，部署的却是更激进的 raw logits。该合同已升级为 v7，训练、replay audit
   和 QuantX plugin 使用同一分布；所有旧 checkpoint 保留为不可正式部署的负对照。
2. 原独立持仓卖出头会在一天内对多个持仓独立采样。v8 将 `exits_only + full_exit_only` 重构为一个组合级
   categorical：`no_exit` 或最多一只完整卖出。v9 在此基础上增加显式 `swap_advantage` critic overlay，
   并把选择器、门槛、动作模式和 prior 写入 checkpoint/正式决策审计。

采集仍使用 QuantX 双账户在线 rollout。32 个 CPU worker 加一张 GPU 的稳定吞吐约为 `700-780 agent step/s`；
64 worker 因 QuantX 账户步进争用会跌至约 `600 step/s`，不采用。GPU 前向约占每个 rollout 的 1.3-1.5 秒，
瓶颈是 QuantX 逐日成交和账户结算，不是 PPO minibatch 或显存。

`evaluate_ppo_position_manager.py` 现导出每个持仓的 raw exit logit、有效 exit score、critic 预测、实际
counterfactual target/mask，支持对 critic tail 做 OOS calibration，而非只看最终净值。

### v19：非目标同构 source 上的正例，但不迁移

训练使用 Reward epoch025/030/035/040 的 2020-2022 数据，2023 做完整四 source 验证；目标 epoch020
没有参与训练。v19 update010 的 source 聚合结果为：

| 指标 | PPO critic overlay | 基线 |
| --- | ---: | ---: |
| 平均累计收益 | +34.65% | +19.96% |
| 平均最大回撤 | -18.13% | -20.54% |
| source 正超额数 | 3 / 4 | - |

这是组合级 single-exit critic 的有效正例，但迁移到完全未训练的 epoch020 2023 路径后失败：
`+9.45%` 对基线 `+37.00%`，最大回撤 `-19.08%` 对 `-11.80%`。因此 v19 不具备目标策略的可部署性，未进入
2024 或 2025-2026 正式 QuantX。

### v20：目标专属时间切分与门槛网格

v20 只使用 epoch020 的 2020-2022 replay 状态和标签训练，2023 为验证，未使用后续标签。固定 update005
后，按照 predicted 5 日净替换优势测试 `{2.5%, 3.5%, 5.0%, 7.5%}` gate：

| gate | 2023 累计收益 | 基线 | 最大回撤 | 基线回撤 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2.5% | +57.36% | +37.00% | -17.72% | -11.80% | 收益增、回撤恶化 |
| 3.5% | +14.54% | +37.00% | -27.55% | -11.80% | 双失败 |
| 5.0% | +37.74% | +37.00% | -17.65% | -11.80% | 收益微增、回撤恶化 |
| 7.5% | +36.97% | +37.00% | -11.82% | -11.80% | 近似退化为基线 |

5% gate 只触发 6 个额外卖出事件，但其真实 counterfactual advantage 均值为 `-1.04%`、胜率 50%，尽管
critic 平均预测为 `+6.87%`。这证明当前 5 日简化 price-return 标签的 tail calibration 不足，不能用阈值
网格修复。update010 的收益和回撤均更差，训练在 update005 后已出现尾部退化。

### v21：基线路径 critic 仍失败

为消除 rollout 状态分布偏移，v21 把训练随机 full-exit 降到约 `0.2%-0.3%/eligible holding`，使训练状态
几乎等于原 Top5 基线路径，并提高 critic 辅助损失。其 2023 critic overlay 却得到 `-23.07%` 累计收益、
`-28.93%` 最大回撤，对应基线 `+37.00%` 与 `-11.80%`。故问题不仅是 PPO 探索强度，而是现有
counterfactual target 与长期 QuantX 策略实际增量价值不一致。

### 当前结论与后续门槛

截至本轮，**没有** RL 动态调仓 checkpoint 通过目标策略的双目标验证，因而没有启动 2025-2026 正式 QuantX
回测，也不应宣称它增强了当前最优 Reward epoch020 Top5 H30+trail 策略。

下一步不能继续调 actor margin、critic gate 或单一 breadth 阈值。需要先用 QuantX 本身离线构造精确动作标签：
对每个基线持仓，分别执行“保持”和“卖出后按原策略实际补位”的受约束账户路径，并用相同的持有/退出规则、
成本和 horizon 比较 realized incremental NAV、drawdown 与 turnover。只有该 exact action-value label 在 2023/2024
有稳定 calibration、tail precision 和正净收益后，才重新接入 PPO actor 或 critic overlay。

## 2026-07-21 RL 动态调仓：精确标签、H30 对齐与多源 PPO 续测

### 工程与效率修正

1. `evaluate_ppo_position_manager.py` 现可加载 exact-label parquet，并导出每个 critic slot 对应的
   `holding_symbols`、prediction、target 与 mask。审计可逐行 join `(execution_date, holding_symbol)`，不再把
   槽位顺序误当证券身份。
2. 新增 `build_exact_quantx_swap_labels.py` 的 2023 OOS 标签：每条均从相同 QuantX 账户快照比较
   keep 与一次 force-swap，继续运行原 H30/trailing/loss 规则。它不是 close-to-close 近似。
3. PPO collector 实测最优并发为 96 个 QuantX CPU account-pair worker：`1,087 PPO step/s`、
   `2,937 QuantX account step/s`；32 worker 为约 `763 PPO step/s`，128 worker 反降至约 `978`。
   训练默认 `--num-envs` 改为 `min(96, CPU count)`。GPU 前向不足 1.5 秒/update，显存低占用是低维决策
   batch 的正常结果，瓶颈仍是真实 QuantX 回放。
4. 新增 `residual_exit_cooldown_days`，并同时接入 replay env、PPO checkpoint、audit 与正式 QuantX plugin。
   以计划提交日开始冷却，冷却期把 residual holding mask 置零；old checkpoint 默认 0，避免语义静默变化。
   plugin 回归测试 `6 passed`，冷却边界断言通过。

### v22 exact 5d critic：标签正确但 OOS 不可交易

训练标签 `2020-2022` 为 3,214 条 exact 5d swap，v22 update005 使用同一目标 strategy 的 2020-2022 训练、
2023 验证。2023 精确标签另建 1,051 条；扣除赢家保护和不可执行槽位后有 641 条可行动样本。

| 检验 | 训练期 2020-2022 | OOS 2023 |
| --- | ---: | ---: |
| 全局 RankIC | 0.689 | 0.025 |
| 日横截面平均 RankIC | 0.297 | 0.045 |
| 每日 Top1 exact delta | +0.282% | -0.038% |
| 可执行全部 exact delta | -0.115% | -0.086% |

v22 已明显记忆训练期 label，不能迁移。gate `{1.5%, 2.0%, 2.5%, 3.0%}` 的 2023 replay 也均未超过
Reward epoch020 基线 `+37.00% / -11.80%`；例如 2.0% 为 `+23.11% / -22.17%`。因此不使用该 checkpoint
进入目标正式回测。

### 窄 critic 预训练与 5d 非重叠部署

新增 `train_exact_swap_critic.py`：从 QuantX 基线状态构建静态数据，2020-2021 train、2022 validation，
actor 冻结为基线；64 hidden critic 使用 Huber exact-value + 同日 pairwise ranking，所有 epoch checkpoint、
JSONL 和 TensorBoard 均已保存。best validation checkpoint 的 2023 日 RankIC 为 0.075，Top1 5d exact delta
为 +0.029%，但不足以证明长期组合收益。

自然零阈值（prediction >= 0）在 2023 触发 121 次连续换仓，正式 replay 得到 `-15.64% / -22.95%`，基线仍为
`+37.00% / -11.80%`。这证明一次 5d counterfactual 不能直接滚动部署。按标签 horizon 预注册 5 日冷却后，
损失收敛为 `-3.44% / -22.13%`，仍不通过。该分支停止，禁止继续基于 2023 搜索 gate。

### H30 exact 标签对齐仍失败

为匹配目标策略的 H30/trailing/loss 合同，构建 30 日 exact label：训练期 3,102 条（65 条未成交剔除），
2023 OOS 937 条（16 条剔除）。训练仅用 2020-2021，2022 以日 RankIC 选择 epoch032（0.0654），部署固定为
30 日冷却。其 2023 可执行样本结果为：日 RankIC `-0.028`、Top1 30d exact delta `-0.594%`、全体 delta
`-0.686%`。所有自然正阈值子集仍为负，故不进入正式策略回测。

**结论**：在当前 feature/state、目标 Reward Top5 与原 H30 出场规则下，exact single-swap value 不是可迁移的
长期操作价值。问题不再是 fill/cost 对齐，也不是 gate；该 critic 路线停止。

### 跨 checkpoint PPO actor 负对照

按策略泛化合同，`source_catalog_mainboard_5d_h30_v1.yaml` 的 epoch005/010/015 三套 score 为训练 source，
目标 epoch020 不参与训练或 checkpoint 选择；均使用同一 H30/trailing/loss 成交合同和 30 日冷却。

1. 保守 prior=6、clone=0.3 训练 20 updates：训练动作率收缩到接近 0，2023 三 source 的 PPO 与基线逐项相同，
   平均累计均为 +24.43%、平均最大回撤均为 -17.96%。
2. 受控探索 prior=2.5、clone=0.15 训练至 update020 后仍然相同；训练期虽然采样了约 1%-3% full-exit，
   deterministic policy 在三条完整 2023 source path 上始终未跨越 inference margin。update020 后停止，未把
训练期采样收益当作策略收益。

补充审计：对受控探索 run 的 update005/010/015/020，将部署 `inference_margin` 从 0.1 预设降为 0.0，
仍在三条 2023 source path 上逐项等于基线（平均累计 +24.43%、平均最大回撤 -17.96%）。因此并非 margin
截断了已学会的残差动作；当前 PPO actor 在该跨 checkpoint、同构 H30 合同下没有形成可部署的状态依赖决策。

截至本节，仍没有 RL 动态调仓 checkpoint 能在目标 Reward epoch020 Top5 H30+trail 上显示经过独立 source/OOS
验证的收益或回撤改善，因此没有启动 2025-2026 目标正式 QuantX PPO 对照。后续应转向多策略离线轨迹训练的
市场级风险/仓位控制，或扩展 action-value 到整段策略生命周期；不得继续搜索当前 exact-swap critic gate、
H30 checkpoint 或单源 actor margin。

## 2026-07-21 PPO 动态调仓：相对 score 状态、时间切分与 QuantX 正例

### 工程修正

本轮完成以下可审计修正，所有 PPO rollout 继续由 QuantX
`BacktestSession -> RuleExecution -> Executor -> Account` 执行，未改写成交、费用、T+1、涨跌停或 NAV：

1. collector 固定为 96 个 CPU QuantX account-pair worker，加单卡 batched policy；稳定吞吐约
   `1,100-1,500 PPO step/s`，GPU 前向通常约 1 秒/update，环境回放仍是主瓶颈。
2. 新增组合级 `single_slot` 动作：每天最多对一只持仓输出 `hold / reduce25 / reduce50 / exit100`；全卖才释放买入槽位。
   任一 residual sell 后进入冷却，原 H30/loss/trailing 卖出规则始终保留。
3. PPO value loss 修正为拟合 lambda-return (`GAE advantage + V_old`)，而非错误地拟合当日 agent return；
   daily return 仅保留作日志指标。对应单测和 plugin 测试共 `10 passed`。
4. state 增加同日 `score_zscore`。此前 Top5 仅保留接近 1 的 percentile，丢失了 Top1 与 Top5 常达
   `0.5-0.7` 个日内标准差的 Reward 置信度差。训练 replay 与正式 plugin 使用同一 z-score 定义；state dim
   从 `482` 变为 `522`，旧 checkpoint 因 state schema 不匹配会显式拒绝加载。
5. 训练期 baseline prior 和 behavior clone 支持线性退火，且 collection 与 PPO update 使用同一时点的行为分布；
   完整 validation JSONL 新增 residual sell、候选拒绝和现金部署偏离计数。

### 跨 checkpoint 负对照

`v14` 使用非 epoch020 的 7 个历史 checkpoint 在 2020-2022 训练、2023 完整 source 选择。
update015 首次出现正例：平均累计 `+25.26%` 对 `+21.88%`、平均最大回撤 `-17.96%` 对 `-19.43%`，5/7 source
为正超额。但其 2024 七 source 平均累计 `+6.59%` 低于 `+9.74%`，虽平均回撤改善，仍未通过收益门槛。
因此该跨 checkpoint controller 不能作为 epoch020 的通用生产策略。

### epoch020 纯时间切分

为避免把 target 结果用于训练，另建独立 catalog：epoch020 score 仅使用 2020-2022 训练、2023 validation，
2024 与 2025-2026 仅作随后评估。`v16` 将超基线回撤的训练惩罚从 `0.05` 提高到 `0.20`，其他模型、动作、
feature、成本和执行合同不变。选用 checkpoint update010、margin `0.10` 的原因是其 2023 收益最强；它在 2023 的收益为
`+58.35%` 对 `+37.00%`，但回撤 `-12.49%` 劣于 `-11.80%`，故**未通过当时的严格双目标选择门槛**。

随后时序结果为：

| 区间 | PPO | 基线 | 结论 |
| --- | ---: | ---: | --- |
| 2024 replay | `+50.52% / -37.15%` | `+31.89% / -37.15%` | 收益提升，回撤持平 |
| 2025-01-02 至 2026-06-16 QuantX | `+122.42% / -22.03% / Sharpe 2.207` | `+114.67% / -22.37% / Sharpe 2.161` | 三项同步改善 |

2025 年收益为 PPO `+78.66%`、基线 `+71.38%`；2026 年截至 6 月 16 日为 PPO `+19.39%`、基线 `+20.31%`。
PPO 共 297 笔成交，基线 300 笔；350 个决策日中有 7 次 residual partial sell，均遵守 Top5、原卖出规则和
冷却约束。

正式 QuantX 工件：

```text
checkpoint:
  rl_position_manager_v1/runs/ppo_epoch020_temporal_scorez_dd20_h30_v16_20260721/
  checkpoints/checkpoint_update0010.pt

formal PPO config:
  quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_ppo_scorez_v14/
  reward_epoch020_top5_h30_trail_ppo_v16_diag_margin01_2025_2026.yaml

formal PPO run:
  quantx/runs/20260721_074036_reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_ppo_position_manager
```

### 结论与限制

本轮首次证明：在 QuantX 正式执行下，PPO dynamic position manager 可以在不改变 Reward Top5 alpha、
不替换原 H30/loss/trailing exits 的前提下，同时提高 target 时间留出期的收益、回撤和 Sharpe。正向增益主要来自
少量 partial sell 与更克制的候选接受，而非高频换仓。

但 2023 checkpoint 选择时没有通过回撤门槛，且 margin 在该年不存在同时改善收益与回撤的候选。因此该结果目前是
**强时序诊断正例，不是无条件生产结论**。下一轮必须冻结 v16 的训练与执行合同，在独立的后续市场窗口复核，或使用
更早的训练/选择 fold 复现同一收益-回撤改善，再升级为生产风险控制器。

## 2026-07-21 PPO 动态调仓：pre-2020 native selector v18 的严格 OOS 正式验证（失败）

### 时间与执行合同

本实验回答一个更严格的问题：仅使用目标 Reward epoch020 Top5 H30+trail 策略在 `2016-2019` 的历史，训练
PPO 后是否能提升从 `2020-01-02` 开始的未来表现。时间合同为：

| 阶段 | 时间 | 用途 |
| --- | --- | --- |
| fit | 2016-01-04 至 2018-11-16 | PPO 参数训练 |
| validation | 2019-01-02 至 2019-11-19 | checkpoint 选择 |
| refit | 2016-01-04 至 2019-11-19 | 固定 update025 后的最终拟合 |
| formal OOS | 2020-01-02 至 2026-06-16 | 一次性 QuantX 正式检验 |

pre-2020 score 工件为 `reward_epoch020_20160104_20191119_pre2020.parquet`，共 1,929,683 行、915 个交易日。
训练没有读取 2020+ 的 score、价格、标签或回测结果；formal OOS 使用原先冻结的
`reward_epoch020_20200102_20260715.parquet`。PPO 固定为 `single_slot`、`drawdown_penalty=0.20`、
`cash_drag_penalty=0.25`、`turnover_penalty=0.003`、`inference_margin=0.10`，96 个 QuantX account-pair
worker，25 updates。每个 rollout 仍由 QuantX 的
`BacktestSession -> RuleExecution -> Executor -> Account` 执行。

### v17 作废与 native selector 修正

旧 v17 replay 曾用手工 `score DESC, instrument ASC` 排列候选；正式 QuantX 的 `FormulaSelector` 对相同分数按
自身横截面顺序处理。两者会在同分时选到不同 Top5，例如 2020-01-02 的 `SH601108` 与 `SH601166`。因此 v17
训练态与正式候选空间不一致，旧 v17 的正式结果只保留为失配负对照，不能作为 PPO 的有效结论。

v18 replay 已移除手工 `SourceCandidateSelector`，改为原生 `FormulaSelector`，仅动态替换同一份 external-score
parquet。2020-01-02 smoke 的 Top5 已与正式候选逐项一致：`SH600048, SZ002146, SH600728, SZ002625, SH601166`。
plugin 测试 `10 passed`，`py_compile` 通过。

### 2019 checkpoint 选择与 refit

原生 selector 下，2019 validation 的 update025 为：PPO 累计 `+68.70%`、最大回撤 `-24.44%`；基线为
`+53.84%`、`-25.90%`。6 个 checkpoint 中 005、015、025 同时满足累计收益不低于基线且回撤不劣于基线；严格
双目标选择器按累计收益、回撤、utility、较早 update 依次排序，选出 update025。随后只在 2016-2019 做了
25-update refit，正式部署 checkpoint 为
`ppo_epoch020_pre2020_native_refit25_scorez_dd20_h30_v18_20260721/checkpoints/checkpoint_update0025.pt`。

### QuantX formal 配对结果

两份 YAML 均通过 dry-run，且严格保持 all_mainboard（3,189 只）、同一 epoch020 score、Top5、lag=1、close
成交、成本、`trail_p20_dd12 + loss_exit_d5 + time_stop_30d` 与原买入规则。唯一差异是 PPO YAML 注入 position
manager plugin。两条正式 run 均 `signal_errors=[]`。

| 指标 | 原策略基线 | pre-2020 PPO v18 | PPO - 基线 |
| --- | ---: | ---: | ---: |
| 累计收益 | +509.36% | +320.13% | -189.23 pct |
| 年化收益 | 32.30% | 24.89% | -7.41 pct |
| 最大回撤 | -41.86% | -47.11% | -5.25 pct |
| Sharpe | 0.877 | 0.668 | -0.210 |
| PF | 1.434 | 1.474 | +0.040 |
| 成交笔数 | 1,442 | 1,508 | +66 |
| 总成本 | 39.78M | 24.25M | -15.53M |

| 年份 | 基线收益 | PPO 收益 | 基线年度最大回撤 | PPO 年度最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| 2020 | +3.93% | +3.43% | -22.56% | -25.52% |
| 2021 | +125.20% | +14.50% | -20.98% | -26.74% |
| 2022 | -12.04% | -3.39% | -40.74% | -46.30% |
| 2023 | +20.41% | +8.96% | -27.03% | -24.67% |
| 2024 | +32.41% | +37.24% | -41.86% | -47.11% |
| 2025 | +35.95% | +66.22% | -32.24% | -19.74% |
| 2026-06-16 | +19.82% | +38.66% | -22.26% | -18.94% |

PPO 在 1,562 个决策日中触发 47 个 residual sell：28 次减仓 50%、6 次减仓 25%、13 次全卖；2020-2026 各年的
次数为 `7/8/4/6/6/13/3`。这不是成本失控导致的失败，因为 PPO 成本反而更低；核心是早期 OOS 的状态-动作泛化
错误，尤其 2021 把基线的强上涨路径压低到 `+14.50%`。2024-2026 的局部改善不能抵消前段复利损失，也不能被用于
事后选择策略。

**严格判定：拒绝，不部署。** PPO 没有同时改善 formal OOS 的收益与最大回撤，且 2020+ 已被使用一次性检验。
后续不得在该 OOS 窗口上继续搜索 margin、惩罚项、update 或 action gate；若继续研究，必须另行预注册新的训练/
验证/最终测试时间切分，或采用多策略历史轨迹训练并保留独立尾部测试期。

正式工件：

```text
formal summary:
  rl_position_manager_v1/reports/epoch020_pre2020_native_v18_quantx_formal_summary.json

baseline config/run:
  quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_ppo_pre2020_native_v18/
  reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_oos_baseline.yaml
  quantx/runs/20260721_095110_reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_oos_baseline

PPO config/run/decisions:
  quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_ppo_pre2020_native_v18/
  reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_ppo_position_manager.yaml
  quantx/runs/20260721_095213_reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_ppo_position_manager
  rl_decisions.parquet
```

## 2026-07-21 Reward Transformer：可成交 Listwise + Pairwise 排序实验（拒绝）

本节只研究 reward score 的排序，不修改现有 RL 买入、卖出或替换模型。目标是验证：在原有同日
pairwise `up / range / down` 损失外，加入按日横截面的 listwise loss，并将 QuantX 的 T+1 可成交约束
写入训练标签后，能否提高 Top5 的真实可执行收益。

### 训练与执行合同

新工程位于 `tmp/reward-model-listwise-v1/`。模型为单一 score head（`d_model=768`、7 层、12 heads，
约 50M head），使用 8 张 GPU DDP 完成 20 epochs。每个逻辑 batch 是一个完整 signal-date 横截面：

1. `L_list`：fillable 股票的真实 5 日收益 rank 转为 `rank^-1` 目标分布，对当天 model score 的 softmax
   做交叉熵；不可成交股票 target mass 为 0。
2. `L_pair`：保留同日 fillable 样本的 broad `up / range / down` Bradley-Terry pairwise loss。
3. 标签复用活动 QuantX 的原始执行合同：`deal_price=close`、T+1 买入、
   `price_jump_limit=0.095`、停牌/一字板/涨停/跳价拒绝。全量 11,828,037 行中 fillable 为
   11,593,796（98.0196%）；4,096 条 Executor 抽样逐项审计与向量标签一致。

候选 checkpoint 为 epoch017。它不是因为 2020+ 表现被选择，而是 2019 的诊断折中点；本节的
2020-2026 QuantX 仅作一次正式诊断，不能再被用于调 epoch、loss 权重或阈值。

8 卡 inference 使用固定每卡 `batch_size=16384`，关闭自动 batch 探测。自动探测曾在 31,488 时使
8 张 32GB GPU 同时 OOM；固定后每卡峰值 13.94GB。导出的 score artifact：

```text
reward_epoch017_2019.parquet:       545,891 rows, 170 dates
reward_epoch017_2020_2026.parquet: 6,788,182 rows, 1,552 dates
```

### 2019 选择折诊断

与 epoch010 broad reward baseline 使用相同的 executable TopK 评估口径：

| Top5 指标 | epoch010 baseline | listwise+pairwise epoch017 |
| --- | ---: | ---: |
| 精确可执行 P@5 | 0.3529% | 0.7059% |
| Fill@5 | 96.7059% | 95.5294% |
| 有效净 5 日收益 | +0.9491% | +0.4412% |

epoch017 的 P@5 更高，但同时降低 Fill 和平均可实现收益；因此它没有通过原有“precision、fill、
effective return 同时改善”的晋级门槛。

### 2020-2026 离线可执行检验

两个 artifact 都覆盖相同的 1,552 个信号日。Top5 的真实 5 日结果为：

| 指标 | epoch010 baseline | listwise+pairwise epoch017 |
| --- | ---: | ---: |
| 精确可执行 P@5 | 0.1675% | 0.3479% |
| Fill@5 | 97.6675% | 95.5541% |
| 有效 gross 5 日收益 | +0.4305% | +0.3649% |
| 有效净 5 日收益 | +0.1267% | +0.0678% |

P@5 翻倍只对应 7,760 个 Top5 选择中的约 14 个额外 exact hit，不能代表整个 Top5 的收益幅度。
当前 loss 让模型追逐极端 rank hit，却没有把平均可实现净收益作为足够强的训练/选择目标。

### 固定规则 QuantX 正式回测

两条策略严格使用同一 all-A、Top5、lag=1、`close` 成交、H4 (`holding_days >= 4`) 卖出、仓位、
成本和 `price_jump_limit=0.095`。两条 artifact 的最后一个可用 signal-date 都是 2026-06-02；
原基线在随后日期也没有新候选，因此两条正式 run 的尾部合同一致。

| 指标 | epoch010 baseline | listwise+pairwise epoch017 |
| --- | ---: | ---: |
| 累计收益 | +94.18% | +5.60% |
| 年化收益 | +10.82% | +0.85% |
| 最大回撤 | -52.24% | -68.04% |
| Sharpe | 0.286 | 0.022 |
| 成交笔数 | 3,866 | 3,834 |
| 拒单数 | 39 | 68 |
| `price_jump` 拒单 | 24 | 63 |
| 胜率 | 45.73% | 44.44% |
| 平均闭仓收益 | +0.2789% | +0.1047% |

epoch017 run：

```text
config:
  quantx/configs/strategies/generated/reward_listwise_pairwise_5d_h4_epoch017_diagnostic_v1/
  reward_epoch017_top5_2020_2026_all_a.yaml
run:
  quantx/runs/20260721_132929_reward_listwise_pairwise_5d_h4_epoch017_diagnostic_2020_2026_all_a
```

排序变化不是基线的轻微扰动：1,552 个共同 signal-date 中 1,488 个（95.88%）Top5 完全无重合，
平均交集仅 0.043 只，Top1 只在 3 天相同。新 loss 实际重写了候选偏好，却没有得到更高的平均
可实现收益；2020、2023、2024、2025 均弱于基线，2026 的局部反弹不足以弥补复利损失。

**严格判定：拒绝，不晋级，也不部署。** 当前 listwise+pairwise 形式不能证明“precision 上升”会变成
交易收益上升。后续若继续 reward 排序研究，应在新的预注册时间切分上比较固定 epoch/seed 的
pairwise-only 对照与 utility-aware listwise：目标分布需同时使用 T+1 可成交 mask 和扣成本后的 5 日
收益幅度，而不是只使用名次；checkpoint gate 必须同时要求 Fill@5、有效净收益与 QuantX 风险指标改善。

## 2026-07-21 Raw-feature Reward Transformer 50M：移除冻结 AE 的正式主板检验（拒绝）

### 实验合同

本实验直接以标准化的 `60 x (52 stock + 20 market + 6 candidate)` 原始特征输入 Reward
Transformer，不使用 AE encoder、latent bottleneck 或 AE checkpoint。模型为 `d_model=768`、
7 层、12 heads、MLP ratio 4.0，共 `49,714,177` 个可训练参数，与原冻结-AE reward head 的
`49,727,233` 参数量匹配。训练标签、同日 `up/range/down` pair 构造、5d endpoint、scaler、
pre-2020 train/2019 validation 和 pairwise Bradley-Terry loss 均保持不变。

学习率从 `1.5e-4` 经 20 epoch cosine 衰减至 `1.5e-5`，之后固定。训练在 epoch023 后没有
写出 `run_end` 而中断；同一时间段驱动日志出现 GPU OOM 记录，原因未被 runner 单独持久化。每个
epoch checkpoint 已保存。只使用 2019 Top5 excess 选择 epoch015（`+0.856%`，高于 epoch020 的
`+0.816%`）；没有用 2020-2026 再做 checkpoint sweep。

### 全量 score

epoch015 使用 7 GPU 全量推理，输出 `4,448,335` 行、`1,552` 个 signal date、`3,181` 个主板
标的的 score artifact，覆盖 `2020-01-02` 至 `2026-06-02`。其 2020-2026 离线排序已弱于 AE
epoch020：

| 指标 | Raw epoch015 | 冻结 AE epoch020 |
| --- | ---: | ---: |
| 日均 RankIC | `0.0151` | `0.0307` |
| 正 RankIC 日期占比 | `56.77%` | `61.08%` |
| Top5 平均 5d 收益 | `+0.650%` | 不足以替代正式回测 |
| Top5 相对日均超额 | `+0.408%` | 不足以替代正式回测 |

两份 score 均覆盖相同的 1,552 日和 4,448,335 行，但 raw 与 AE 每日 Top5 平均交集仅 `0.155`
只；`1,335` 日完全无交集。因此 raw 不是 AE score 的轻微扰动，而是重写了候选空间。

### QuantX 正式配对结果

raw YAML 与 `+509.36%` AE 基线严格保持 all_mainboard（3,189 只）、Top5、lag=1、close 成交、
5bp 佣金、1bp 卖出印花税、10bp 滑点、`trail_p20_dd12 + loss_exit_d5 + time_stop_30d`、初始
资金 1 亿和 `2020-01-02` 至 `2026-06-16` 窗口。唯一差异为 external score parquet。YAML dry-run
通过，正式 run `signal_errors=[]`。

| 指标 | 冻结 AE epoch020 基线 | Raw epoch015 | Raw - AE |
| --- | ---: | ---: | ---: |
| 累计收益 | `+509.36%` | `-10.01%` | `-519.37 pct` |
| 年化收益 | `+32.30%` | `-1.62%` | `-33.92 pct` |
| 最大回撤 | `-41.86%` | `-54.88%` | `-13.02 pct` |
| Sharpe | `0.877` | `-0.043` | `-0.920` |
| PF | `1.434` | `0.980` | `-0.454` |
| 成交笔数 | `1,442` | `1,434` | `-8` |
| 总成本 | `39.78M` | `16.14M` | `-23.64M` |

raw run 的拒单仅 35 笔（`price_jump=27`、`limit_down=8`），不足以解释收益失败。其核心问题是
raw score 在 future full-market score 上的 RankIC 减半且 Top5 候选完全迁移，2019 validation 的
正向头部指标没有稳定外推到 2020-2026。

**严格判定：拒绝，不部署。** 去 AE 的 raw epoch015 没有保留 500% AE 基线的长期可交易排序。由于
epoch015 已按 2019 选择并完成一次 formal OOS，禁止再在该 2020-2026 窗口上测试 raw epoch020 或搜索
其阈值、容量和退出规则。若继续研究 raw 输入，需要新的预注册时间切分，并将 AE 与 raw 的 checkpoint、
优化 schedule 和有效 batch 完全匹配后再进行独立尾部检验。

工件：

```text
raw checkpoint:
  market_mainboard_close2close_balanced_v1/runs/
  market-mainboard-pre2020-raw-reward-transformer50m-allup-5d-bs488x4-7gpu-40ep-lrfloor-20260721-154920/
  checkpoints/reward_epoch_015.pt

raw score / manifest:
  .../score_artifacts/reward_raw_epoch015_20200102_20260715.parquet
  .../score_artifacts/reward_raw_epoch015_20200102_20260715.json

QuantX config / run:
  quantx/configs/strategies/generated/reward_transformer_mainboard_raw_epoch015_formal/
  reward_raw_epoch015_5d_top5_h30_trail_p20_dd12_2020_2026.yaml
  quantx/runs/20260721_205733_reward_raw_epoch015_5d_top5_h30_trail_p20_dd12_2020_2026
```

## 2026-07-23 AE 重建表征 + 独立 DINO 语义表征：后续实验设计（未实现，未训练）

### 动机与当前判断

直接移除 AE 的 raw-feature Reward Transformer 在固定的 formal 区间得到 `-10.01%`，而冻结 AE
epoch020 基线为 `+509.36%`。这不能证明 AE 对所有市场阶段都最优，但已说明：当前 AE latent
包含 raw reward head 未能稳定重建的 OOS 泛化信息。因此下一步不是把 AE 替换掉，而是保留已验证的
重建表征，同时训练一条独立的、标签无关的语义表征分支。

该设计要检验的假设是：AE reconstruction objective 提供了价格/市场状态的稳定性，而 DINO-style
self-distillation 可以补充对扰动不变、横截面可迁移的语义信息；二者在 token channel 上并列输入
reward model，而不是先相互压缩。

### 固定的表示与下游接口

冻结的现有 AE 使用已验证 checkpoint，其 encode 输出保持：

```text
Z_ae   : [B, 27, 128]
Z_dino : [B, 27, 256]
concat : [B, 27, 384]   # dim=-1，逐 token channel concat
```

`Z_dino` 是独立 DINO encoder 的正式对外输出，维度固定为 `[27, 256]`。DINO encoder 从现有 AE
的有效 encode 路径初始化，即 `stock_proj / market_proj / candidate_proj / fusion / pos / encoder /
compress`；不复用或训练 AE decoder。其 `128 -> 256` semantic projection 属于 DINO encoder，DINO
pretraining 使用的 prototype head 在下游阶段丢弃。

**明确取消**此前提出的 `[B, 27, 384] -> [B, 27, 128]` 融合压缩层。下游 Reward Transformer
直接接收 384-channel token，并以自身的 input projection 映射到 `d_model`。Reward Transformer
不得为了回到 128 channel 而缩小；参数量至少保持当前约 50M 的 baseline，是否增加 `d_model` 或层数
只在新的历史验证协议中决定，并要与保持原 50M 宽度/层数的 direct-384 对照一起记录。

### 两阶段训练顺序

1. **AE 保持冻结。** 不重新训练或改写已验证 AE 的 reconstruction encoder/decoder，也不改变其
   checkpoint、scaler、原始输入或 latent contract。
2. **独立 DINO pretraining。** Student 初始化自上述 AE encode 路径；teacher 是 student encoder
   加 DINO projection/prototype head 的 EMA copy。输入仅取 pre-2020 的训练数据，不读取收益标签、
   pair 标签、2020+ 特征、价格或回测结果。DINO loss 使用 teacher centering、teacher/student
   temperature 与 EMA momentum；prototype 预测使用 token pooled representation，但下游保留完整
   的 `[27, 256]` token sequence。
3. **冻结双 encoder 后训练 reward。** 冻结 `Z_ae` 和 `Z_dino` 的所有参数，只训练 direct-384
   Reward Transformer；保持原有同一 signal-date 的 `up/range/down` Bradley-Terry pairwise
   数据构造、5d endpoint、scaler、score range 与 QuantX 执行合同。DINO 没有直接接触任何 reward
   或交易标签。

### DINO 数据增强边界

只允许不会泄漏未来、也不改变股票身份/日期语义的保守视图增强：

1. 60 日 lookback 内的连续时间块 mask。
2. stock 或 market 的特征组 mask。
3. 标准化特征上的小幅随机 jitter。

禁止时间顺序打乱、未来时间步拼接、跨股票 mixup、跨日期拼接、使用未来收益作为 augmentation 或
teacher target。训练日志应至少记录 DINO loss、student/teacher entropy、center norm、EMA momentum、
learning rate、样本吞吐和 GPU peak memory。

### 时间切分与判定纪律

该分支必须先预注册新的历史选择过程：representation/reward train 仅使用原 pre-2020 train，
checkpoint/模型宽度/层数只允许参考 2019 的历史验证 split。2020-2026 已被现有多轮 reward、RL 和
raw 实验使用，因而不能再作为 DINO epoch、prototype 数、增强强度、Reward Transformer 宽度/层数、
TopK 或退出规则的选择集；若仍做该区间比较，只能标记为诊断性复核，不能作为新的生产 OOS 声明。

下游 checkpoint 晋级仍需同时看 2019 横截面 RankIC、Top5 可成交率、有效净 5d 收益和 pairwise
validation，不接受只改善 pair accuracy 或单一 precision 的模型。冻结所有选择后，才可按完全相同的
all_mainboard、Top5、lag=1、close 成交、成本与 `trail_p20_dd12 + loss_exit_d5 + time_stop_30d`
合同做一次配对 QuantX 诊断。

**状态：仅完成设计记录；未新增 DINO/reward 代码，未启动训练，未产生 checkpoint 或 score artifact。**

## 2026-07-23 Loop Preference / Loop Diffusion：中间 BT 监督、持有期与多时间尺度诊断

### 目的与实验边界

本轮实验不改变已经验证的冻结 AE 表征、同日 pairwise 数据构造、5D close-to-close endpoint 或 QuantX
成交合同；只将 Reward Transformer reward head 替换为三步 loop preference head。该结构把前一步的
condition state 回灌到下一步 transformer-like block，最终输出一个坍缩的 `reward_score_5d`。它不是
diffusion 训练：没有真实中间 denoising target 或 flow-matching loss，训练信号始终是 Bradley-Terry
pairwise loss。

两种只差在中间步监督的实验已经完成：

| 代号 | checkpoint | loop steps | step loss weights | 说明 |
| --- | --- | ---: | --- | --- |
| C | deep epoch015 | 3 | `[1/6, 1/3, 1/2]` | 每一步均施加 BT deep supervision。 |
| D | final-only epoch020 | 3 | `[0, 0, 1]` | 中间 state 只用于内部计算，只有最终 score 受 BT 监督。 |

二者均使用冻结 AE、约 50.3M 可训练 reward 参数、相同的 2010-2018 train / 2019 validation split，以及
`rank_horizon=5`。C/D 的 checkpoint 分别按 2019 validation 的头部横截面指标选择；不使用 2020+ 选择
checkpoint。

### 2019 validation 与 loop 内部状态

选择的 C epoch015 在 2019 validation 的 5D RankIC 为 `0.0600`、Top5 5D 超额为 `+0.868%`；D epoch020
分别为 `0.0549` 和 `+0.846%`。因此仅看该选择集，C 略优于 D。

但中间循环的验证 pair accuracy 没有表现为 C 的稳定 refinement：

| 模型 | step1 | step2 | step3/final |
| --- | ---: | ---: | ---: |
| C deep epoch015 | `54.15%` | `53.19%` | `53.69%` |
| D final-only epoch020 | `52.84%` | `54.03%` | `53.70%` |

C 把 BT 目标同时施加给每一步后，第 2 步反而弱于第 1 步，最终只部分恢复；D 虽没有中间 BT loss，但其内部
state 能在第 2 步改善验证排序。这支持如下工作假设：对没有真实中间 label 的 loop state，强迫每一步直接
满足最终 BT 目标会限制其作为内部计算状态的自由度。该结论目前来自单个 seed / checkpoint 对照，不能单独
视为严格的结构因果证明。

### 2020-2026 QuantX 诊断回放

所有回放均采用 all_mainboard（3,189 只）、Top5、lag=1、close 成交、5bp commission、1bp 卖出印花税、
10bp slippage、初始资金 1 亿，以及同一 2020-01-02 至 2026-06-16 窗口。C/D 的 score artifact 各有
4,448,335 行、1,552 个 signal date、3,181 个可评分主板标的；所有正式回放均 `signal_errors=[]` 且日志
无 warning/error。

H30 使用 `trailing_peak20_dd12 + loss_exit_d5 + time_stop_30d`。与冻结 AE Reward Transformer epoch020
基线（`+509.36%`、最大回撤 `-41.86%`、Sharpe `0.877`）的配对结果如下：

| 模型 | Top5 总收益 | 最大回撤 | Sharpe | 成交笔数 | Bottom5 总收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C deep H30 | `-12.99%` | `-74.12%` | `-0.060` | `1,530` | `-99.44%` |
| D final-only H30 | `+191.84%` | `-63.96%` | `0.520` | `1,509` | `-99.60%` |

H5 保留同一 trailing/loss-exit/cost/选股合同，仅把 `time_stop_30d` 改为 `time_stop_5d`：

| 模型 | Top5 总收益 | 最大回撤 | Sharpe | 成交笔数 | Bottom5 总收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C deep H5 | `+24.05%` | `-74.94%` | `0.098` | `3,082` | `-99.62%` |
| D final-only H5 | `+15.91%` | `-59.63%` | `0.068` | `3,086` | `-99.97%` |

H5 的平均报告持仓约 7.55 个自然日，是五个交易日跨周末后的正常表现。H5/H30 回放均维持强烈的 Top5/Bottom5
方向分离；但它们不是纯持有期实验，因为仍包含 `loss_exit_d5`、trailing、持仓不替换、可成交限制与成本。
尤其 H30 的实际平均持仓约 15.5 个自然日，并非每只股票持满 30 天。

### 固定 cohort 多时间尺度分析

为剥离上述 QuantX 路径依赖，直接使用 score artifact 与保存的目标路径做无摩擦、无退出规则的固定 cohort
分析。目标路径定义严格为：

```text
target(H) = close(T+H) / close(T+1) - 1
```

因此训练的 `rank_horizon=5` 与固定 cohort 的 H5 严格一致；score artifact 的 `true_return_5d` 与
`target_abs_path_float32.mmap[:, 4]` 的最大绝对误差为 `0`。H1 在该定义下恒为零，故从 H2 开始报告。
“Top5 超额”指每个 signal date 的 score Top5 平均收益减去当日全部评分标的平均收益，再跨日期平均；未扣成本。

| H | C RankIC | C Top5 超额 | D RankIC | D Top5 超额 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `0.0206` | `+0.022%` | `0.0251` | `+0.053%` |
| 3 | `0.0249` | `+0.055%` | `0.0312` | `+0.124%` |
| 5 | `0.0307` | `+0.101%` | `0.0410` | `+0.202%` |
| 10 | `0.0380` | `+0.214%` | `0.0475` | `+0.308%` |
| 15 | `0.0436` | `+0.164%` | `0.0543` | `+0.547%` |
| 20 | `0.0468` | `+0.307%` | `0.0582` | `+0.854%` |
| 30 | `0.0492` | `+0.253%` | `0.0618` | `+1.147%` |

核心观察：5D BT 标签没有把 score 限死在五日。C/D 的横截面 RankIC 都随 horizon 延长而提高，D 的 Top5
超额也单调增强至 H30。这更像 score 在 5D endpoint 上学到了持续趋势的早期排序，而非只能存活五日的一次性
模式。D 同时在所有 horizon 都优于 C，故其 H30 QuantX 优势不能仅解释为退出规则偶然有利。

C 的单年最大无摩擦 Top5 超额发生在 2024 H30，约 `+1.912%`/signal date；其 2024 H5 超额约
`+0.546%`。这是事后诊断峰值，不能据此认定 C 的可部署最优持有期为 H30。D 的 2021/2022/2024/2025
分别在 H30/H30/H30/H20 附近出现较强头部超额，长期曲线更一致。2026 只有 97 个可完整结算的 signal date，
不用于强结论。

2023 是二者共同的 regime failure：C/D 的 H5 RankIC 仅 `0.0083/0.0094`，Top5 H5 超额均为负，正式 H5/H30
策略也均亏损。此时问题是横截面排序关系接近消失，而非选择 H5、H20 或 H30 的问题。

### 候选迁移与 loss_exit_d5 检查

C/D 不是同一 score 的轻微重标定。它们的每日 Top5 平均交集从 2020 的 `0.55` 只降至 2025/2026 的
`0.23/0.22` 只；后两年约 `79%` 日期完全无交集。deep supervision 改变了候选空间，而不仅是 score 的尺度。

检查 score Top5 的路径可知，`loss_exit_d5` 不是两者差异的主要解释：

| 指标 | C | D |
| --- | ---: | ---: |
| H5 为负、H30 转正的慢启动赢家比例 | `18.18%` | `19.60%` |
| H5 为正、H30 转负的早期赢家反转比例 | `19.20%` | `18.62%` |
| H5 为负候选的平均 H30 收益 | `-2.83%` | `-1.80%` |

慢启动赢家并未显著多于早期赢家反转，且 H5 为负候选在 H30 平均仍亏损；第五日止损在平均意义上没有系统性砍掉
长期赢家。H5/H30 QuantX 表现的剩余差异应理解为 score、价格路径、trailing、持仓替换约束、成交可行性和成本的
共同结果。

### Frozen-AE Reward Transformer epoch020 的多 horizon 基线与回测前筛选 gate

已验证的 `+509.36%` frozen-AE Reward Transformer epoch020 也在完全相同的无摩擦固定 cohort 口径下
计算。对每个 signal date，以稳定降序排序取 Top5，并与当日全部评分标的均值比较：

| H | Top5 平均收益 | 全体平均收益 | Top5 超额 | RankIC |
| ---: | ---: | ---: | ---: | ---: |
| 5 | `+0.633%` | `+0.242%` | `+0.391%` | `0.0307` |
| 10 | `+1.202%` | `+0.502%` | `+0.699%` | `0.0394` |
| 20 | `+2.344%` | `+1.029%` | `+1.315%` | `0.0515` |
| 30 | `+2.776%` | `+1.522%` | `+1.254%` | `0.0538` |

H5 的 `+0.391%` 与该 artifact 已有 full-metrics 的正式离线值一致。基线的头部超额在 H20 最高，
而全横截面 RankIC 在 H30 最高；这说明“最强 head excess”与“最广泛排序相关性”可以对应不同的 horizon。
D final-only 在四个 horizon 都低于这个基线，H30 最接近但仍为 `+1.147% < +1.254%`。

后续实验采用如下晋级纪律，以在不消耗 QuantX 回测资源前筛掉明显弱于基线的 score：

1. **预先固定比较合同。** 同一 frozen representation、股票池、信号日期、`close(T+1)` entry、目标路径、
   H 集合 `{5, 10, 20, 30}`、Top5、稳定 tie-break 和全体横截面基准；不得在计算后改 universe、TopK 或 H。
2. **预先指定主 horizon。** H5 用于检查是否保留标签一致性的头部 alpha；H20 用于检查长期 head excess，
   因为现有强基线在该点达到峰值。H10/H30 为必要的 guardrail，不允许只报告候选最有利的一个 H。
3. **晋级条件。** 若候选目标是替代该基线，必须在独立 validation 或滚动历史切分上，事先指定的主 horizon 的
   Top5 超额严格高于对应基线，同时报告四个 H 的 RankIC、正 RankIC 日期占比与 Top5 超额。只改善 pair accuracy、
   单个日期或事后挑出的单一 H 不晋级 QuantX。
4. **QuantX 仍是必要终检。** 离线 gate 不包含涨跌停/停牌、lag、成交成本、持仓重叠、trailing、loss exit 与
   容量。因此通过 gate 后，仍须以预先固定的持有期和相同执行合同运行 Top5/Bottom5 配对 QuantX；离线优势不是
   可交易优势的充分条件。
5. **禁止用已观察 OOS 选择阈值。** 上表的 2020-2026 数值只能作为诊断基线，不得据其为新实验选择 H、
   checkpoint 或晋级阈值后，再把同一窗口称为新的 OOS。新的正式筛选必须用未使用的历史 validation 或滚动切分。

### 当前判定与时间切分纪律

1. 在本次对照中，D final-only 是优于 C deep supervision 的 loop 训练形式；C 的中间 BT 约束更可能强化
   2019 的局部形态而削弱跨阶段泛化。
2. D 有可复核的多 horizon 持续性，但仍显著落后于已验证的 Reward Transformer epoch020 基线，不能替换基线。
3. 持有期 `H` 可以作为交易策略超参数，在独立历史 validation 上与 checkpoint 一起选择；这不会改变 5D 标签的
   训练目标，也不构成标签泄漏。
4. 但不得在同一 2020-2026 诊断区间比较 H5/H10/H20/H30 后，选择最优 H 并再次将该区间报告为新的 OOS。当前
   所有 H30/H5 与固定 cohort 结果均为诊断性复核，不能用于后续结构、checkpoint、持有期或退出规则的继续调参。
5. 若要正式选择 H，必须预先固定候选集合，在未使用的历史 validation 或滚动历史切分上选择，然后对完全未参与
   选择的尾部窗口只评估一次。

工件：

```text
C train / score:
  market-mainboard-pre2020-ae-loop-preference50m-k3-deep-bs488x4-8gpu-20ep-20260723-104311/
  checkpoints/reward_epoch_015.pt
  score_artifacts/reward_loop_c_deep_epoch015_20200102_20260715.parquet

D train / score:
  market-mainboard-pre2020-ae-loop-preference50m-k3-final-only-bs488x4-8gpu-20ep-20260723-115054/
  checkpoints/reward_epoch_020.pt
  score_artifacts/reward_loop_d_final_epoch020_20200102_20260715.parquet

QuantX formal runs:
  20260723_125905_loop_c_deep_epoch015_5d_top5_h30_trail_p20_dd12_2020_2026
  20260723_130057_loop_c_deep_epoch015_5d_bottom5_h30_trail_p20_dd12_2020_2026
  20260723_130133_loop_d_final_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026
  20260723_130216_loop_d_final_epoch020_5d_bottom5_h30_trail_p20_dd12_2020_2026
  20260723_131120_loop_c_deep_epoch015_5d_top5_h5_trail_p20_dd12_2020_2026
  20260723_131201_loop_c_deep_epoch015_5d_bottom5_h5_trail_p20_dd12_2020_2026
  20260723_131237_loop_d_final_epoch020_5d_top5_h5_trail_p20_dd12_2020_2026
  20260723_131312_loop_d_final_epoch020_5d_bottom5_h5_trail_p20_dd12_2020_2026
```

## 2026-07-23 K=5 final-only Loop Preference：40 epoch、持有期分布与 QuantX 诊断

### 训练合同与 checkpoint 选择

在原 K=3 D final-only 的 frozen-AE、同日 Bradley-Terry pair、5D 标签、2010-2018 train / 2019
validation_select 合同上，仅将 loop rollout 从 K=3 提升到 K=5；中间四步不施加 loss，最终第五步为唯一的
BT target。训练 40 epoch，7 卡 DDP，effective batch `15,680` pairs/update，cosine learning rate 从
`1.5e-4` 降至 `1.5e-5`。

checkpoint 选择规则在读取 2020+ 分数前固定：只使用 2019 validation_select，先最大化 Top5 5D 超额，
要求 RankIC 为正，若并列再比较 RankIC。epoch040 训练结束时原训练日志未写入最终横截面事件，但 checkpoint
已经完整写入；因此单独对相同的 428,100 行 / 170 日期 validation_select 复算，避免把它不当地排除。

| checkpoint | 2019 Top5 5D 超额 | 2019 RankIC | 正 RankIC 日期 |
| --- | ---: | ---: | ---: |
| K=5 epoch020 | `+1.091%` | `0.0687` | `76.47%` |
| K=5 epoch040（独立复算） | `+0.546%` | `0.0565` | `75.88%` |
| K=3 D final-only epoch020 | `+0.846%` | `0.0549` | 未单列 |

故正式冻结 K=5 epoch020。K=5 在 2019 H5 选择集上优于 K=3 D，但这个局部优势不足以推出其在后续窗口能替代
原 Reward Transformer。

### 固定 cohort 多持有期诊断

对 K=5 epoch020（正式选择）和 epoch040（仅训练时长诊断）均导出完整 2020-01-02 至 2026-06-02 的分数：
4,448,335 行、1,552 个 signal date。文件名中的 `20260715` 是最后一个 30D target 的结算边界，而不是
最后一个 signal date。每个日期按稳定 score 降序取 Top5，目标严格为
`close(T+H) / close(T+1) - 1`，Top5 超额为该 Top5 平均收益减全横截面平均收益；不含成本、换手或退出规则。

| checkpoint | H | RankIC | 正 RankIC 日期 | Top5 超额均值 | 日超额中位数 | 日超额为正 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch020 | 5 | `0.0246` | `59.60%` | `+0.059%` | `-0.154%` | `47.16%` |
| epoch020 | 10 | `0.0287` | `59.34%` | `+0.301%` | `-0.188%` | `47.94%` |
| epoch020 | 20 | `0.0341` | `61.98%` | `+0.589%` | `-0.333%` | `47.87%` |
| epoch020 | 30 | `0.0329` | `61.21%` | `+0.644%` | `-0.281%` | `48.26%` |
| epoch040 diagnostic | 5 | `0.0420` | `66.82%` | `+0.285%` | `+0.047%` | `50.58%` |
| epoch040 diagnostic | 10 | `0.0522` | `71.59%` | `+0.429%` | `-0.189%` | `48.45%` |
| epoch040 diagnostic | 20 | `0.0605` | `75.06%` | `+0.849%` | `-0.181%` | `48.58%` |
| epoch040 diagnostic | 30 | `0.0638` | `76.87%` | `+1.077%` | `-0.159%` | `48.52%` |

epoch040 的长持有期统计确实系统性高于 epoch020，且 RankIC 随 H 延伸提高；它说明后期 loop state 学到的更接近
中长期横截面排序。但这不是稳健 Top5 alpha：在 H10/H20/H30，epoch040 的每日 Top5 超额中位数仍为负，只有约
48.5% 日期为正，均值由少数大幅正日拉高。epoch020 的这个问题更明显。也就是说，K=5 的长周期信息是存在的，
但没有稳定地集中到每日 Top5。

这组 2020-2026 结果只可作为诊断：epoch040 不能因其 H30 表现较好而替换按 2019 H5 规则冻结的 epoch020，
更不能据此反向选择 H30 或训练 epoch 后重新将同一窗口报告为 OOS。

与同口径 frozen-AE Reward Transformer epoch020 基线相比，K=5 epoch040 即使在最有利的 H30 也仍为
`+1.077% < +1.254%`；H20 `+0.849% < +1.315%`，H5 `+0.285% < +0.391%`。因此该 loop 变体尚未通过替代
基线的离线 gate。

### K=5 epoch020 QuantX H30 正式诊断

策略使用 all_mainboard（3,189 只）、Top5 / Bottom5、lag=1、close 成交、5bp commission、1bp 卖出印花税、
10bp slippage、初始资金 1 亿、`trailing_peak20_dd12 + loss_exit_d5 + time_stop_30d`，窗口为
2020-01-02 至 2026-06-16。两份 YAML 均先 dry-run 通过，artifact manifest 完整，正式 run 均
`signal_errors=[]`。

| 方向 | 总收益 | 最大回撤 | Sharpe | 成交笔数 | 平均持有日 | 拒单 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top5 | `-4.39%` | `-76.40%` | `-0.020` | `1,501` | `15.63` | `37` |
| Bottom5 | `-99.84%` | `-99.91%` | `-1.315` | `1,866` | `12.13` | `263` |

Bottom5 的灾难性表现保留了 score 的方向性：K=5 score 能远离最差股票；但 Top5 仍亏损，说明它没有形成可覆盖
交易摩擦、路径退出与持仓约束的头部选股强度。K=5 epoch020 明显弱于 K=3 D H30 的 `+191.84%`，更远弱于原
Reward Transformer epoch020 的 `+509.36%`。因此结论不是“loop 无任何信息”，而是**当前 K=5 final-only
loop preference 对长期广义排序有诊断性增益，但不能作为可部署 Reward Transformer 的替代结构。**

工件：

```text
K=5 train:
  market-mainboard-pre2020-ae-loop-preference50m-k5-final-only-bs280x8-7gpu-40ep-20260723-204707/
  checkpoints/reward_epoch_020.pt
  checkpoints/reward_epoch_040.pt

K=5 score artifacts:
  score_artifacts/reward_loop_k5_final_epoch020_20200102_20260715.parquet
  score_artifacts/reward_loop_k5_final_epoch040_20200102_20260715.parquet
  score_artifacts/reward_loop_k5_epoch020_vs_epoch040_multihorizon_fixed_cohort.json
  score_artifacts/diagnostic_epoch040_validation_select_20190102_20190910_full_metrics.json

K=5 QuantX configs / runs:
  quantx/configs/strategies/generated/loop_preference_k5_final_epoch020_h30_formal/
  20260723_235939_loop_k5_final_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026
  20260724_000024_loop_k5_final_epoch020_5d_bottom5_h30_trail_p20_dd12_2020_2026
```

## 2026-07-26 7D Multi-task Reward Transformer：收益优先、路径风险二阶段排序与 TopK/Holding 网格

### 本节结论与适用边界

本轮不再训练一个把收益、Sharpe 和回撤直接混为单一总分的 head，而是采用**共享冻结 AE 编码器、三个独立
pairwise 排序 head**：`return`、`sharpe`、`drawdown`。推理严格分两阶段：先用收益 head 保证候选在当日
return Top50，再只在这 50 只内按 Sharpe/drawdown 排名重排。这样风险偏好不能把收益候选外的股票重新提入
最终组合。

截至本节，当前冻结的研究候选为 **Top10 + H20**：在下述同一 QuantX 合同与 2020-01-02 至 2026-06-02
窗口中，累计收益 `+563.33%`、年化 `34.28%`、最大回撤 `-42.58%`、Sharpe `1.107`、Calmar `0.805`。
它是 `Top1/3/5/10 x H5/7/10/20/30` 网格中 Sharpe 最高的组合；但不是原始收益或 Calmar 的全局最高值：
Top1-H20 分别为 `+999.61%` 和 `0.837`，代价是单股票集中、`-54.07%` 回撤且对持有期敏感。若首要约束是
回撤，Top3-H7 仍是更保守的选择（`+338.82%`、`-31.93%`、Sharpe `0.746`）。

重要时间纪律：模型 checkpoint 仅用 pre-2020 训练、2019 选择；因此 2020+ 是相对训练的未来窗口。**但是，
TopK、H 和本节的“当前候选”已在同一 2020-2026 窗口上比较过 20 个组合。** 所以 Top10-H20 只能称为该窗口
上的诊断性最佳风险收益组合，不能再称为未参与选择的独立 OOS 或直接视为生产承诺。后续若要冻结部署参数，必须
在未使用的历史 validation 或新的尾部测试窗口预先选择 `K/H`，然后只做一次正式检验。

### 时间、数据与冻结工件合同

训练 fold 为 `pre2020_eval2020_2026`，每个样本使用 60 个交易日 lookback 的 `raw_relative` 特征，目标路径
定义为：

```text
target[h] = close(T + 1 + h - 1) / close(T + 1) - 1
```

也就是目标数组保存从 `close(T+1)` 起的未来累计 close-to-close 路径；路径 Sharpe 的相邻对数收益是
`T+2 ... T+H`，不虚构不可观测的入场日收益。时间和样本量如下：

| 阶段 | 日期 | 样本/日期 | 用途 |
| --- | --- | ---: | --- |
| train | 2010-01-01 至 2018-12-31 | 4,155,867 rows，2,078 dates | 仅用于模型参数拟合。 |
| validation_select | 2019-01-02 至 2019-09-10 | 545,891 rows，170 dates | epoch-10 gate，不读取 2020+。 |
| prediction / QuantX | 2020-01-02 至 2026-06-02 | 6,788,182 rows，1,552 dates | 导出分数并做实际回测。 |
| target 结算边界 | 至 2026-07-15 | - | 保证最后一个 signal date 的长期路径可结算；不是 QuantX 最后交易日。 |

冻结 AE 为：

```text
market_all_pre2020_v1/runs/
market-all-pre2020-ae08-abs-20260717-144320/checkpoints/ae_epoch_020.pt
```

它有 `40,427,736` 个冻结参数，输出 `27 x 128` latent tokens。训练出的 Reward Transformer 是 7 层、
`d_model=768`、12 heads、MLP ratio 4、dropout 0.05 的共享编码器后排序网络；其三个输出 head 为
`return/sharpe/drawdown`，可训练参数 `49,728,771`，总注册参数 `90,156,507`。三个 raw logit 分别经 sigmoid
变换为 `(1e-6, 1-1e-6)` 内的独立 score，不能把它们的数值幅度跨 head 直接相加比较。

训练 run、checkpoint 和完整 resolved manifest：

```text
root:
  market_all_close2close_balanced_v2/

run:
  runs/reward-multitask-7d-return005-risk005-top50-10ep-gate-20260726-172019/

checkpoint:
  checkpoints/reward_epoch_010.pt

complete resolved settings / history:
  train_manifest.json
  logs/train_metrics.jsonl
```

为避免后续同名文件或未提交代码变化造成混淆，本次冻结文件的 SHA-256 为：

| 文件 | SHA-256 |
| --- | --- |
| `checkpoints/reward_epoch_010.pt` | `dd05b2c9dc94f4053793f5276f9c2d8b6b36cd638ea6ced125f5b581f29ecf02` |
| `score_artifacts/reward_multitask_epoch010_20200102_20260715.parquet` | `a8dba145a216268a93fd6e8130f4a1cf8c3e3696a24ab9cc234b3e6b5437b798` |
| `train_manifest.json` | `fbd852d4d2481a965fab4dee1170337782a599c7513949f6e44bef689858b814` |
| Top10-H20 YAML | `cedf0a6a77bc197d5b55ed4c63b0b6ddc0fc614067d9d5fdf0d4d496e8cb0885` |

### 三个 head 的 pair 合同与优化

三个 task 都只在同一 `signal_date` 内构造 good/bad pair；不同 task 的 pair 独立采样，loss 只读取对应
head 的 logit。每个 task 使用 Bradley-Terry loss：

```text
L_pair = softplus(-(logit_good - logit_bad) / temperature)
L = weighted_mean(L_pair) + 0.01 * weighted_mean(logit_good, logit_bad)^2
temperature = 1.0
```

`0.01` 的中心项只消除 BT 的公共 offset 不可辨识性。实际 task 权重为 return `1.00`、Sharpe `0.25`、
drawdown `0.25`；它们不是一个总分标签的预设权重。

| Head | good/bad 的严格条件 | 训练意图 |
| --- | --- | --- |
| `return_score` | 同日且 `return(good) - return(bad) > 0.005` | 保留收益 alpha；`0.005` 即至少 0.5 个百分点 endpoint gap。 |
| `sharpe_score` | 同日、`abs(return(good)-return(bad)) < 0.005`，且 future-path Sharpe 的同日百分位 `good - bad > 0.05` | 在收益几乎相同的 pair 中学习更平滑的上涨路径。 |
| `drawdown_score` | 同日、同样的收益近似条件，且 signed maximum drawdown 的同日百分位 `good - bad > 0.05` | 在收益几乎相同的 pair 中学习更小回撤；常规 signed MDD 越接近 0 越好。 |

风险 pair 在各自 metric 的日内百分位上保持精确 `1:1:1` 的 down/range/up class balance，每个 good 固定保留
4 个同日 bad alternatives 并按 epoch 轮换。return pair 则保留所有符合条件的 up 样本，并在每个 signal date
均衡 range/down；动态 up/range/down endpoint 阈值为 `+/-5%`。epoch 中的可用 pair 为：return `2,293,497`、
Sharpe `2,859,384`、drawdown `3,094,803`，合计 `9,284,409`；训练日志因 DDP 聚合记录为 `9,284,412`。

future-path 指标的精确定义为：

```text
nav_h              = 1 + target[h]
daily_log_return   = diff(log(nav_h))
path_sharpe        = sqrt(252) * mean(daily_log_return) / std(daily_log_return, ddof=1)
max_drawdown       = min(nav_h / cummax(nav_h) - 1)
```

训练使用 6 卡 DDP、BF16、每卡 `1,952` pairs/update，故每个 optimizer update 的有效 pairs 为 `11,712`；
10 epochs、每 epoch 793 updates，`lr=1.5e-4`、warmup ratio `0.05`、末端学习率 0、weight decay `0.1`、
grad clip `1.0`。训练 pair accuracy 从 epoch1 的 `58.11%` 提升到 epoch10 的 `62.22%`，loss 从 `0.6642`
降至 `0.6308`。本轮按要求没有再训练 20 epochs：epoch10 gate 使用的 2019 validation_select 指标为：

| 2019 validation_select，H7 | 全横截面均值 | Top5 selected 均值 | selection advantage |
| --- | ---: | ---: | ---: |
| return head endpoint return | `+0.918%` | `+2.094%` | `+1.176 pct` |
| two-stage endpoint return | `+0.918%` | `+1.391%` | `+0.473 pct` |
| drawdown head signed MDD | `-4.621%` | `-1.595%` | `+3.026 pct` |
| Sharpe head path Sharpe | `1.555` | `1.260` | `-0.295` |

gate 只要求 return head 和 two-stage Top5 的收益 advantage 均严格大于 0，故 epoch010 通过并冻结；它**没有**
证明 Sharpe head 在该 2019 Top5 切片上独立优于基准。这个负 Sharpe-head 诊断必须保留，不能因后续整体回测
较好而事后删除。

训练可复现命令的核心参数如下；完整的默认值和所有 resolved 参数以同一 run 下的 `train_manifest.json` 为准：

```bash
cd ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 \
  train_wts_ae_reward_transformer_ddp_v1.py train \
  --root market_all_close2close_balanced_v2 \
  --kronos-root ../kronos-classification-signal-v1 \
  --fold-id pre2020_eval2020_2026 \
  --ae-checkpoint market_all_pre2020_v1/runs/market-all-pre2020-ae08-abs-20260717-144320/checkpoints/ae_epoch_020.pt \
  --run-name reward-multitask-7d-return005-risk005-top50-10ep-gate \
  --epochs 10 --rank-horizon 7 --label-mode multitask \
  --reward-backbone frozen_ae --reward-architecture transformer \
  --reward-d-model 768 --reward-layers 7 --reward-heads 12 --reward-mlp-ratio 4 --reward-dropout 0.05 \
  --pairs-per-batch 1952 --groups-per-batch 8 --amp-dtype bf16 \
  --lr 1.5e-4 --weight-decay 0.1 --warmup-ratio 0.05 --grad-clip 1.0 --logit-center-weight 0.01 \
  --min-return-gap 0.005 --multitask-return-loss-weight 1.0 \
  --multitask-sharpe-loss-weight 0.25 --multitask-drawdown-loss-weight 0.25 \
  --multitask-risk-return-gap-max 0.005 --multitask-risk-min-percentile-gap 0.05 \
  --multitask-risk-bad-options 4 --multitask-return-top-n 50 \
  --multitask-risk-sharpe-weight 0.5 --multitask-risk-drawdown-weight 0.5 \
  --screen-epoch 10 --screen-min-top5-excess 0.0
```

### 推理与二阶段选股合同

epoch010 由 6 卡 inference 导出。每卡 batch `1,952`，共 `6,788,182` rows、1,552 个 signal dates、
5,173 个可评分 instrument。主 score parquet 为：

```text
score_artifacts/reward_multitask_epoch010_20200102_20260715.parquet
score_artifacts/reward_multitask_epoch010_20200102_20260715.json
score_artifacts/reward_multitask_epoch010_20200102_20260715_full_metrics.json
score_artifacts/reward_multitask_epoch010_h5_h7_h10_h20_h30_20200102_20260715_path_quality_diagnostics.json
```

输出列固定为：

```text
reward_return_score_7d
reward_sharpe_score_7d
reward_drawdown_score_7d
reward_return_topn_risk_score_7d       # 唯一供 QuantX selector 使用的最终列
```

最终列不是线性加权 raw logits，而是 checkpoint 保存的不可变两阶段合同。对每个日期：

1. 按 `reward_return_score_7d` 稳定降序取 `C = Top50`；稳定排序使用 `mergesort`，tie 保持输入行顺序。
2. 只在 `C` 内分别对 `reward_sharpe_score_7d` 和 `reward_drawdown_score_7d` 做稳定降序排名。排名从 0 开始，
   0 是最好。
3. 对候选 `i` 计算

```text
risk(i) = 0.5 * (1 - sharpe_rank(i)/(len(C)-1))
        + 0.5 * (1 - drawdown_rank(i)/(len(C)-1))
```

4. 按 `risk(i)` 稳定降序赋最终候选分数 `1 - rank/len(C)`；候选外股票赋小于 `-1` 的分数。因此任何
   `TopK <= 50` 的 selector 都不可能提取 return Top50 以外的股票。

用于多持有期的无摩擦离线诊断另定义了 path quality：

```text
quality = 0.50 * daily_percentile(terminal_return)
        + 0.30 * daily_percentile(path_sharpe)
        + 0.20 * daily_percentile(signed_max_drawdown)
```

这里的 `0.50/0.30/0.20` 仅是**诊断标签**，与推理阶段的 `0.5 Sharpe rank + 0.5 drawdown rank` 不同，也没有
进入 QuantX 的仓位计算。Top10 最终 two-stage score 的 quality advantage 随路径长度增加：

| 路径 horizon | quality RankIC | Top10 quality advantage vs 当日全市场 |
| ---: | ---: | ---: |
| H5 | `0.0742` | `+2.811 pct` |
| H7 | `0.0804` | `+3.024 pct` |
| H10 | `0.0864` | `+3.256 pct` |
| H20 | `0.0995` | `+3.787 pct` |
| H30 | `0.1034` | `+3.955 pct` |

这些离线数值未包含 `lag`、停牌/涨跌停、持仓重叠、成交成本或退出路径，且 H 的比较已经参与了后续策略选择；
它们只能解释“这个 score 在固定 cohort 上的路径质量排序”，不能替代下文 QuantX 结果。

可复现 score 导出的核心命令为：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 \
  infer_wts_ae_reward_transformer_scores_v1.py \
  --root market_all_close2close_balanced_v2 \
  --kronos-root ../kronos-classification-signal-v1 \
  --checkpoint market_all_close2close_balanced_v2/runs/reward-multitask-7d-return005-risk005-top50-10ep-gate-20260726-172019/checkpoints/reward_epoch_010.pt \
  --fold-id pre2020_eval2020_2026 --prediction-split prediction \
  --start 2020-01-02 --end 2026-07-15 --rank-horizon 7 \
  --diagnostic-horizons 5,7,10,20,30 --batch-size 1952 --amp-dtype bf16 \
  --output market_all_close2close_balanced_v2/runs/reward-multitask-7d-return005-risk005-top50-10ep-gate-20260726-172019/score_artifacts/reward_multitask_epoch010_20200102_20260715.parquet
```

### QuantX 正式执行合同

所有 grid run 共享如下策略合同，只有最终 `TopK` 与 `holding_days >= H` 改变：

| 项目 | 固定值 |
| --- | --- |
| 数据与股票池 | `data/qlib_data_fixed`、`all_a`，配置 universe 5,179 只；score artifact 覆盖 5,173 只，缺失分数 `drop`。 |
| 回测窗口 | 2020-01-02 至 2026-06-02；初始资金 100,000,000。 |
| 选股 | `precomputed` external-score，`reward_return_topn_risk_score_7d`，`lag: 1`，`wrap_first_signal: false`。 |
| 仓位 | `selector.topk = rebalance.max_positions = K`，等权，`cash_use_ratio=0.98`，`buy_only_new_positions=true`。 |
| 买卖 | close 成交、lot size 100、跳价限制 `0.095`、跳过涨停、可复用卖出现金；唯一卖出规则为 `holding_days >= H` 后全卖。没有 trailing 或 loss exit。 |
| 成本 | commission `5 bp`（最低 5）、仅卖出 stamp tax `1 bp`、slippage `10 bp`、transfer fee 0。 |
| 引擎 | `validate_trading_rules=true`、`error_policy=fail_fast`、`legacy_cost_price=true`、`auto_adjust_buy_quantity=true`。 |

Top1/Top3、Top5、Top10 YAML 分别在：

```text
quantx/configs/strategies/generated/reward_multitask_7d_epoch010_topk_hold_grid_v1/
quantx/configs/strategies/generated/reward_multitask_7d_epoch010_hold_grid_v1/
quantx/configs/strategies/generated/reward_multitask_7d_epoch010_top10_hold_grid_v1/
```

Top10 的五份 YAML 全部先通过 `--dry-run --json`。20 个正式 run 均为同一执行合同的真实 QuantX run，
`signal_errors=[]`，且 `logs.txt` 未匹配 warning/error/traceback。H20 Top10 的可复现执行命令为：

```bash
cd ${HOME}/git/quantization/quantx
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/reward_multitask_7d_epoch010_top10_hold_grid_v1/reward_multitask_7d_epoch010_top10_two_stage_h20_2020_2026_all_a.yaml \
  --output-dir runs --json
```

### 全部 TopK x Holding 结果

下表每个单元格依次为 `累计收益 / 最大回撤 / Sharpe`；都是扣除上述交易成本后的 QuantX 实际净值结果，而非
固定 cohort 离线收益。

| 最终容量 | H5 | H7 | H10 | H20 | H30 |
| --- | --- | --- | --- | --- | --- |
| Top1 | `-64.49% / -72.48% / -0.383` | `-5.91% / -60.33% / -0.024` | `+31.00% / -66.23% / 0.105` | `+999.61% / -54.07% / 0.915` | `+251.37% / -66.92% / 0.469` |
| Top3 | `+113.87% / -46.23% / 0.370` | `+338.82% / -31.93% / 0.746` | `+231.99% / -47.21% / 0.567` | `+279.90% / -44.92% / 0.656` | `+352.05% / -38.26% / 0.722` |
| Top5 | `+46.52% / -43.21% / 0.199` | `+263.26% / -37.36% / 0.691` | `+249.90% / -45.16% / 0.676` | `+492.71% / -44.88% / 0.954` | `+355.91% / -38.93% / 0.815` |
| Top10 | `+24.76% / -44.26% / 0.120` | `+197.04% / -39.68% / 0.612` | `+102.08% / -44.24% / 0.386` | **`+563.33% / -42.58% / 1.107`** | `+365.76% / -41.47% / 0.876` |

Top10 的完整风险、成本和持有期统计如下：

| H | 累计收益 | 年化 | 最大回撤 | Sharpe | Calmar | 成交笔数 | 总成本 | 平均持有日 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `+24.76%` | `3.51%` | `-44.26%` | `0.120` | `0.079` | `6,184` | `44.72M` | `7.56` |
| 7 | `+197.04%` | `18.48%` | `-39.68%` | `0.612` | `0.466` | `4,420` | `48.09M` | `10.58` |
| 10 | `+102.08%` | `11.58%` | `-44.24%` | `0.386` | `0.262` | `3,094` | `24.47M` | `15.11` |
| 20 | **`+563.33%`** | **`34.28%`** | `-42.58%` | **`1.107`** | `0.805` | `1,550` | `27.08M` | `30.25` |
| 30 | `+365.76%` | `27.08%` | `-41.47%` | `0.876` | `0.653` | `1,030` | `15.27M` | `45.34` |

Top10-H20 相比同一 score/执行合同的 Top5-H20，不是单纯用更多股票稀释 alpha：

| 指标 | Top5-H20 | Top10-H20 | Top10 - Top5 |
| --- | ---: | ---: | ---: |
| 累计收益 | `+492.71%` | `+563.33%` | `+70.62 pct` |
| 年化收益 | `31.95%` | `34.28%` | `+2.33 pct` |
| 最大回撤 | `-44.88%` | `-42.58%` | `+2.30 pct`（更小） |
| Sharpe | `0.954` | `1.107` | `+0.153` |
| Calmar | `0.712` | `0.805` | `+0.093` |
| 成交笔数 | `775` | `1,550` | `+775` |
| 总成本 | `23.41M` | `27.08M` | `+3.67M`（`+15.7%`） |

这项容量收益并非每个 holding 都存在。尤其 Top10-H7 低于 Top3-H7（`+197.04%/-39.68%/0.612` 对
`+338.82%/-31.93%/0.746`），说明“更多持仓”不是通用改进；模型的可交易容量和持有期存在强交互。

### 当前判定、不可作出的结论与后续纪律

1. 该实验支持“收益先筛选、路径风险后重排”的结构：最终策略没有以风险总分取代收益 head，且 Top10-H20 在
   当前诊断窗口给出最高 Sharpe。
2. 不能由此证明单独 Sharpe head 已充分泛化：2019 validation 的 Sharpe-head Top5 advantage 为负；更合理的
   表述是三 head 的联合两阶段合同在现有回测路径上有效，而不是每个子任务都已独立通过。
3. `H20`、`Top10`、Top50 阈值、等权风险 rank 和成本/退出合同均已在 2020-2026 上被查看。禁止继续在同一窗口
   微调 TopN、risk weight、K、H 或 checkpoint 后，再将改善称为新的 OOS。
4. 当前研究冻结候选可写为：`reward_multitask epoch010 -> return Top50 -> equal Sharpe/drawdown rerank -> Top10 -> time stop H20`。
   若目标是更低回撤，则使用已记录的 Top3-H7 作为独立的保守候选，而不是把 Top10-H20 的结果外推成任何风险偏好下
   的唯一最优策略。
5. 下一次正式选择必须预注册新的时间切分：只在旧历史 validation 固定 checkpoint、TopN、K、H 与风险权重；
   冻结后才对新的、不参与选择的尾部区间运行一次同合同 QuantX 回测。

### 结果工件索引

所有下列 run 位于 `quantx/runs/<run_id>/`，每个目录包含 `summary.json`、`metrics.json`、
`daily_nav.json`、`trades.json`、`positions.json` 和 `logs.txt`。20 个本节正式 run 为：

```text
Top1:
  H5  20260726_191611_reward_multitask_7d_epoch010_top1_two_stage_h5_2020_2026_all_a
  H7  20260726_191611_reward_multitask_7d_epoch010_top1_two_stage_h7_2020_2026_all_a
  H10 20260726_191611_reward_multitask_7d_epoch010_top1_two_stage_h10_2020_2026_all_a
  H20 20260726_191610_reward_multitask_7d_epoch010_top1_two_stage_h20_2020_2026_all_a
  H30 20260726_191611_reward_multitask_7d_epoch010_top1_two_stage_h30_2020_2026_all_a

Top3:
  H5  20260726_191706_reward_multitask_7d_epoch010_top3_two_stage_h5_2020_2026_all_a
  H7  20260726_191706_reward_multitask_7d_epoch010_top3_two_stage_h7_2020_2026_all_a
  H10 20260726_191704_reward_multitask_7d_epoch010_top3_two_stage_h10_2020_2026_all_a
  H20 20260726_191703_reward_multitask_7d_epoch010_top3_two_stage_h20_2020_2026_all_a
  H30 20260726_191704_reward_multitask_7d_epoch010_top3_two_stage_h30_2020_2026_all_a

Top5:
  H5  20260726_190752_reward_multitask_7d_epoch010_top50_risk_top5_h5_2020_2026_all_a
  H7  20260726_190746_reward_multitask_7d_epoch010_top50_risk_top5_h7_2020_2026_all_a
  H10 20260726_190914_reward_multitask_7d_epoch010_top50_risk_top5_h10_2020_2026_all_a
  H20 20260726_190910_reward_multitask_7d_epoch010_top50_risk_top5_h20_2020_2026_all_a
  H30 20260726_190910_reward_multitask_7d_epoch010_top50_risk_top5_h30_2020_2026_all_a

Top10:
  H5  20260726_192901_reward_multitask_7d_epoch010_top10_two_stage_h5_2020_2026_all_a
  H7  20260726_192857_reward_multitask_7d_epoch010_top10_two_stage_h7_2020_2026_all_a
  H10 20260726_192854_reward_multitask_7d_epoch010_top10_two_stage_h10_2020_2026_all_a
  H20 20260726_192853_reward_multitask_7d_epoch010_top10_two_stage_h20_2020_2026_all_a
  H30 20260726_192850_reward_multitask_7d_epoch010_top10_two_stage_h30_2020_2026_all_a
```

实现和回归测试入口：

```text
train_wts_ae_reward_transformer_ddp_v1.py
infer_wts_ae_reward_transformer_scores_v1.py
tests/test_multitask_reward_v1.py
```

测试覆盖三 head 输出 shape、task-specific loss（每个 pair 只监督其 head）、以及二阶段最终 score 不会把 return
TopN 外股票提升进候选。复核实现时应先运行：

```bash
cd ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1
conda run -n test pytest -q tests/test_multitask_reward_v1.py
```

## 2026-07-26 Multi-task Top10-H20：仓位语义修复、风控对照与滞后市场状态诊断

### 先修正基线合同

上一节 Top10-H20 网格中的
`20260726_192853_reward_multitask_7d_epoch010_top10_two_stage_h20_2020_2026_all_a` 不能继续作为正确的
Top10 等权基线。旧执行语义会在替换时把当期可用现金视为新股票的可用额度，而不是以组合总值计算目标权重；在
`buy_only_new_positions=true` 的 H20 滚动持仓下，这使部分保留仓位无法按目标权重回落，实际形成明显集中。

对旧 run 的逐日最大单票权重统计为：中位数 `33.36%`、P95 `43.24%`、全样本最大 `48.23%`。这与“Top10
等权”不一致，所以旧表中的 `+563.33% / -42.58% / 1.107` 只保留为错误仓位语义的历史诊断，不能再与后续策略
比较或作为当前候选的收益声明。

引擎新增的 opt-in 合同为：

```yaml
rebalance:
  weight_scope: portfolio_target
execution:
  max_position_weight: 0.12
  buy:
    sizing: target_weight
```

即新仓按组合总值的等权目标下单，已有仓仅在超过 cap 时被裁剪。默认配置仍保留旧行为，只有明确写入这三个字段
的 YAML 才启用新语义。修复后 run 的逐日最大单票权重中位数为 `10.69%`、P95 `11.85%`、最大 `13.25%`；最后
一个数可以因成交后的价格变动略高于下单时的 `12%` cap。

正确基线及完整执行合同：

```text
config:
  quantx/configs/strategies/generated/reward_multitask_7d_epoch010_risk_controls_v1/
  reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a.yaml

run:
  quantx/runs/
  20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a
```

它固定使用冻结的 `reward_multitask_epoch010_20200102_20260715.parquet` 中
`reward_return_topn_risk_score_7d`、return Top50 内 Sharpe/drawdown 二阶段排序、all-A（5,179 只）、Top10、
selector `lag=1`、close 成交、H20 time stop、佣金 5bp、卖出印花税 1bp、slippage 10bp、初始
资金 1 亿。窗口为 2020-01-02 至 2026-06-02，`signal_errors=[]`。

| 指标 | 修复后 Top10-H20 基线 |
| --- | ---: |
| 累计收益 | `+323.22%` |
| 年化收益 | `25.20%` |
| 最大回撤 | `-40.98%` |
| Sharpe | `0.865` |
| Calmar | `0.615` |
| 成交笔数 | `1,663` |
| 总成本 | `20.08M` |
| 平均持有日 | `28.88` |
| 胜率 / Profit factor | `59.39% / 1.561` |

### 组合级仓位和个股止损对照

在上述正确基线之外，只改动一个风险控制变量。四份 YAML 均通过 dry-run，正式 run 的日志未匹配
warning/error/traceback。

| 变体 | 唯一改动 | 累计收益 | 年化 | 最大回撤 | Sharpe | Calmar | 成交笔数 | 总成本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| portfolio-target 基线 | 无额外风控 | `+323.22%` | `25.20%` | `-40.98%` | `0.865` | `0.615` | 1,663 | 20.08M |
| fixed stop | `pnl_pct <= -10%` 单票全卖，再保留 H20 | `+295.39%` | `23.88%` | `-44.39%` | `0.820` | `0.538` | 1,823 | 19.10M |
| trailing stop | 单票先盈利 `10%`，再从自身峰值回撤 `8%` 全卖 | `+189.45%` | `18.01%` | `-42.94%` | `0.617` | `0.419` | 1,683 | 17.78M |
| breadth20 cash40 | `B20 < 35%` 时新建目标仓仅用普通目标的 `40%` | `+179.24%` | `17.35%` | `-22.35%` | `0.820` | `0.776` | 1,603 | 12.09M |

两个单票止损都被拒绝。弱市场中它们卖掉一批弱股后仍会替换进另一批按同一 score 排名的弱股，不能降低组合的
共同市场 beta，反而增加换手或错过反弹，因此回撤均劣于无止损基线。

`breadth20_cash40` 不是清仓规则：它不卖已有仓，只在低宽度下把后续 replacement 的部署强度降到 `40%`。在
2024 年的 242 个交易日中，其现金占组合总值的均值为 `35.09%`、中位数 `48.62%`，正确基线分别仅为
`2.21%`、`1.91%`。这解释了回撤从 `-40.98%` 降到 `-22.35%`，也解释了复利收益被显著牺牲。它是风险偏好
取舍的反事实，不证明“低宽度时策略没有未来收益”。

相关配置和 run：

```text
fixed stop:
  reward_multitask_7d_epoch010_top10_h20_fixed_stop10_2020_2026_all_a.yaml
  20260726_201319_reward_multitask_7d_epoch010_top10_h20_fixed_stop10_2020_2026_all_a

trailing stop:
  reward_multitask_7d_epoch010_top10_h20_trailing_peak10_dd08_2020_2026_all_a.yaml
  20260726_201318_reward_multitask_7d_epoch010_top10_h20_trailing_peak10_dd08_2020_2026_all_a

breadth deployment:
  reward_multitask_7d_epoch010_top10_h20_breadth20_cash40_2020_2026_all_a.yaml
  20260726_201435_reward_multitask_7d_epoch010_top10_h20_breadth20_cash40_2020_2026_all_a
```

所有 YAML 位于：
`quantx/configs/strategies/generated/reward_multitask_7d_epoch010_risk_controls_v1/`；run 均位于
`quantx/runs/`。

### 预先固定的 B20/B60 市场状态事件研究

为避免直接以回测收益反复搜索阈值，先只检查一个透明的、预先写定的全 A 宽度趋势候选，不把它接入清仓执行：

```text
B20 = 全 A 中 close > 20 交易日 close MA 的比例
B60 = 全 A 中 close > 60 交易日 close MA 的比例

risk-off = B20 < 35% 且 B60 < 45%
risk-on  = B20 > 50% 且 B60 > 50%
```

实现位于 `quantx/tools/analyze_market_regime_alignment.py`。它读取正确基线的 `daily_nav.json` 和同一 Qlib
provider 的 all-A close 矩阵，不读 score、个股未来标签或指数替代标签。为 MA60 额外读取 180 个日历日历史，
实际分析范围仍是 2020-01-02 至 2026-06-02、1,552 个策略交易日、5,179 只股票。

**无前视时间合同：** `t` 日收盘的 B20/B60 只写入下一个策略交易日 `t+1` 的状态；所有 H 日标签从可执行日
`NAV(t+1)` 到 `NAV(t+1+H)` 计算。H 日路径最大回撤是该路径内的
`min(NAV / running_path_high - 1)`。因此没有“同日收盘知道状态、同日收盘清仓”的不可能成交假设。

产物完整保留逐日状态、三段样本、连续 episode、未来收益/路径回撤、最大回撤覆盖及错误空仓事件：

```text
quantx/artifacts/market_regime_alignment/
20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a_breadth20_60_alignment_v1.json
```

H5/H10/H20 的主要对照如下。`material loss` 固定定义为未来策略收益 `<= -5%`，`material path drawdown`
固定定义为未来路径 MDD `<= -10%`；它们只用于报告 coverage/precision，不参与状态阈值选择。

| 可执行状态 | H | 样本数 | 平均未来策略收益 | 未来路径平均 MDD | 未来收益为负 | 路径 MDD `<= -10%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| risk-off | 5 | 403 | `+0.892%` | `-3.395%` | `41.69%` | `4.71%` |
| 非 risk-off | 5 | 1,143 | `+0.448%` | `-2.549%` | `46.28%` | `1.14%` |
| risk-off | 10 | 402 | `+1.753%` | `-5.473%` | `39.05%` | `11.69%` |
| 非 risk-off | 10 | 1,139 | `+0.906%` | `-4.306%` | `42.41%` | `5.09%` |
| risk-off | 20 | 402 | `+3.413%` | `-8.352%` | `36.07%` | `24.63%` |
| 非 risk-off | 20 | 1,129 | `+1.980%` | `-6.533%` | `41.81%` | `15.15%` |

这不是仅由某一段行情造成。固定三段的 H20 方向完全一致：

| 阶段 | risk-off 平均收益 | 非 risk-off 平均收益 | risk-off 路径平均 MDD | 非 risk-off 路径平均 MDD |
| --- | ---: | ---: | ---: | ---: |
| 2020-2021 | `+4.543%` | `+2.771%` | `-7.567%` | `-6.157%` |
| 2022-2023 | `+2.666%` | `+0.384%` | `-7.530%` | `-6.116%` |
| 2024-2026 | `+3.184%` | `+2.658%` | `-9.619%` | `-7.237%` |

所以该状态**稳定地识别了更高波动/更深路径风险，却没有识别“该策略未来无法赚钱”的区间**。它更接近恐慌后可
反弹的高波动状态；把它直接变成全仓空仓会系统性放弃策略的正期望收益。

H20 的风险覆盖也不足以支持清仓：241 个未来 `<= -5%` 的策略样本中，risk-off 只覆盖 `29.05%`；402 个
risk-off 样本中，只有 `17.41%` 最终落入该损失集合。270 个未来路径 MDD `<= -10%` 的样本中，覆盖率为
`36.67%`，risk-off 的 precision 为 `24.63%`。54 个连续 risk-off episode 的起点中，只有 9 个 H20
收益 `<= -5%`，却有 20 个 H20 收益 `>= +5%`；按每日样本计，402 个 risk-off 日中有 170 个（`42.29%`）
后续 H20 收益至少 `+5%`。

### 最大回撤为何发生，以及本规则的边界

正确基线的全样本最大回撤从 2023-11-20 的组合峰值到 2024-02-07 的谷值，为 `-40.98%`、57 个交易日。
这是一段明确的市场内部走弱：可执行 risk-off 最早在 2023-12-11 出现，当时组合已回撤 `-6.15%`、
`B20=19.98%`、`B60=41.77%`；谷值时 `B20=9.39%`、`B60=6.31%`。在 57 日峰谷区间内，规则覆盖 31 日
（`54.39%`）。因此市场环境确实是主要成因之一，且 H20 time stop 没有提供组合级去 beta 的动作；但它也说明
仅靠慢速宽度确认无法在第一段损失前就全部撤出。

更重要的是，最差 20 个单日策略收益中规则只处于 risk-off 的 10 日。其余例子包括 2020-02-03 的 `-9.34%`
（B20 `25.23%`、B60 `48.48%`，刚好未满足 B60 条件）、2024-10-09 的 `-9.30%`（前一日
B20 `99.58%`、B60 `99.50%`，risk-on）、2025-04-07 的 `-9.00%`（B20 `27.36%`、B60 `53.57%`）。
这表明回撤不全是“宽度已持续低迷”的可预测大盘下跌，也包含趋势很强时的突发反转和组合自身的选股暴露。单票
固定/移动止损无法解决前一种共同 beta，也不能预测后一种跳变。

**判定：拒绝把 `B20<35% && B60<45%` 作为该策略的全仓空仓规则，也不在本 2020-2026 已查看窗口继续搜索
阈值。** 它可保留为透明的路径风险特征，并解释为何 breadth20 降新仓能降低回撤；但尚不满足“稳定匹配策略
无法赚钱时段、同时保住收益 alpha”的要求。

若继续研究，下一步必须先在不重叠的开发期预注册少量规则族和选择准则，再只在尾部保留期验证。一个可检验但尚
未执行的假设是把慢速宽度与独立的市场趋势确认组合，并要求它在策略未来收益负向、而非仅路径波动高时才触发；
在得到独立样本的 coverage、precision 和实际 QuantX 反事实前，不应把任何新组合条件部署为清仓。

复现与测试：

```bash
cd ${HOME}/git/quantization/quantx

conda run -n test python -m pytest \
  tests/tools/test_analyze_market_regime_alignment.py -q

conda run -n test python -m quantx.tools.analyze_market_regime_alignment \
  --run-id 20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a \
  --json
```

## 2026-07-26 自建 0AMV 活跃成交额：固定状态、事件研究与真实全清仓反事实

### 问题、边界与结论先行

本轮的问题不是“能否复刻指南针终端的专有 0AMV 数值”，而是：是否可以按公开的
`成交量 * 当日均价 ~= amount` 定义，在本地自建一个全市场活跃成交额序列，并用它识别
Multi-task Top10-H20 策略应当全仓空仓的时段。

**最终结论：拒绝把本轮两个固定的自建 AMV 规则部署为该策略的全清仓规则。**

它们可以描述较低收益/较高风险的市场环境，但不稳定地匹配“策略未来无法赚钱”的区间。更关键的是，真实
QuantX 全清仓会打断 Top10-H20 的持仓路径和后续再入场，结果显著劣于正确基线。不得把下面的零成本日收益
遮罩结果当成可交易收益，也不得以该结果继续调阈值。

### 数据口径：为何可以自己算，以及为何不用 Qlib `$vwap`

当前 QuantX 的 Qlib 数据中有 `$volume`、`$close` 和 `$vwap`，但 `$vwap` 与原始成交量的单位不一致：例如
2020-01-02 的 `SH600000`，原始 `amount=647,446,166`，而 `Qlib $volume * $vwap` 只有约 `6,474,462`。
因此不能用 Qlib 的调整价 `$vwap` 直接相乘。

本轮直接读取同一回测数据来源的原始 BaoStock 个股 CSV：

```text
quantx/data/raw/baostock/stocks/<SYMBOL>.csv
```

每个交易日的自建活跃值为：

```text
AMV[t] = sum_i amount_i[t]

纳入条件：
  tradestatus_i[t] == 1
  volume_i[t] > 0
  amount_i[t] > 0
  i 属于正确基线的同一 all-A universe
```

`amount` 是 BaoStock 原始日成交额（CNY）。这使其成为可复算的“全 A 活跃成交额 / 0AMV proxy”；它**不是**
声称逐点等于指南针终端可能使用了专有股票池、盘中 OHLC 与处理规则的 0AMV 指数。

基线 all-A 为 5,179 只股票；AMV warm-up 从 2019-07-06 开始，正式对齐范围为 2020-01-02 至 2026-06-02。
所有 1,552 个基线 NAV 交易日都有 AMV 输入。有效成交股票数最小/中位/最大为 `3,545 / 4,811 / 5,174`，原始行数
最小/中位为 `3,556 / 4,812`；变化来自上市、停牌和无成交，而非缺失的基线日期。

实现与测试：

```text
quantx/tools/analyze_active_value_regime_alignment.py
quantx/tests/tools/test_analyze_active_value_regime_alignment.py
```

工具按批次、最多 4 个本地 I/O worker 读取 CSV；不访问网络，不写原始数据，也不会读取 future score/label。

### 预注册状态与无前视时间合同

在查看本轮结果前固定四个状态，不进行参数网格搜索。移动均线全部以交易日计算：

```text
AMV_downtrend:
  AMV[t] <= MA20(AMV)[t]
  and MA5(AMV)[t] < MA20(AMV)[t] < MA60(AMV)[t]

Breadth_only:
  B20[t] < 35%
  and B60[t] < 45%

Breadth_plus_AMV:
  Breadth_only and AMV_downtrend

Capitulation_guard (只作反例控制，不触发空仓):
  Breadth_only and AMV[t] > MA20(AMV)[t]
```

其中 B20/B60 复用上节已经冻结的 all-A close-above-MA 宽度。`AMV_downtrend` 没有从策略收益拟合数值阈值；
它表达的是“成交额已低于中期均线，且 5/20/60 日趋势均向下”的持续资金撤离假设。

**时间合同：** 原始状态仅在 `t` 日收盘后可见；产物中 `state_*[t+1] = raw_*_close[t]`，因此策略在可执行日
`t+1` 才能按该状态卖出/空仓。H5/H10/H20 标签均从 `NAV(t+1)` 起算；路径 MDD 为该前瞻路径中
`min(NAV / running_path_high - 1)`。没有同日收盘看见 AMV 后同日收盘成交的前视假设。

### 事件研究：AMV 有环境信息，但不足以当作“策略失效”分类器

固定阈值下的 H20 结果如下。`material loss` 为未来策略收益 `<= -5%`，路径深回撤为未来 H20 path MDD
`<= -10%`，它们只用于评价，未参与状态定义。

| 状态 | 可执行日 / episode | H20 平均收益 | 非状态 H20 平均收益 | 状态 H20 亏损率 | 深亏覆盖 | 深路径 MDD 覆盖 | 状态日后 H20 `>= +5%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Breadth_only | 408 / 54 | `+3.413%` | `+1.980%` | `36.07%` | `29.05%` | `36.67%` | `42.29%` |
| AMV_downtrend | 364 / 70 | `+1.323%` | `+2.678%` | `42.58%` | `23.65%` | `31.11%` | `30.22%` |
| Breadth_plus_AMV | 202 / 45 | `+1.712%` | `+2.454%` | `38.61%` | `14.11%` | `22.22%` | `36.14%` |
| Capitulation_guard | 115 / 54 | `+5.134%` | `+2.139%` | `33.33%` | `8.30%` | `7.78%` | `51.35%` |

解释：AMV_downtrend 的确使 H5/H10/H20 均值变低、亏损率变高，说明“资金撤离”包含市场状态信息；但 H20
仍是正 `+1.323%`，且只覆盖 `23.65%` 的深亏样本。把宽度叠加进去并没有改善识别，反而把深亏覆盖降至
`14.11%`。高于 MA20 的恐慌/活跃成交控制组有最高的后续收益，支持“下跌放量不应机械空仓”的判断。

三段 H20 方向也不稳定，尤其组合规则在弱市段反向错杀：

| 状态 | 阶段 | 状态 H20 平均收益 | 非状态 H20 平均收益 | 深亏覆盖 | 错杀 `>=+5%` |
| --- | --- | ---: | ---: | ---: | ---: |
| AMV_downtrend | 2020-2021 | `-0.080%` | `+3.885%` | `21.05%` | `13.10%` |
| AMV_downtrend | 2022-2023 | `+1.811%` | `+0.710%` | `20.55%` | `31.45%` |
| AMV_downtrend | 2024-2026 | `+1.690%` | `+3.231%` | `27.03%` | `38.46%` |
| Breadth_plus_AMV | 2020-2021 | `+0.124%` | `+3.438%` | `7.02%` | `14.29%` |
| Breadth_plus_AMV | 2022-2023 | `+3.596%` | `+0.537%` | `10.96%` | `50.00%` |
| Breadth_plus_AMV | 2024-2026 | `+0.870%` | `+3.197%` | `19.82%` | `33.68%` |

所以 AMV_downtrend 可以作为将来市场状态模型的候选输入，但不能宣称其稳定覆盖策略的亏损期；
Breadth_plus_AMV 更不能作为部署信号。

### 最大回撤的时序检验

正确基线最大回撤仍是 2023-11-20 至 2024-02-07、57 个交易日、`-40.98%`。在该峰谷区间，两个 AMV 风险状态
恰好相同：共覆盖 15 日、4 个 episode，第一次可执行状态是 2023-12-20。此时正确基线已从峰值回撤 `-7.85%`；
也就是说它比此前 breadth 风险状态还慢，不能阻止回撤的第一段。

AMV-only 真实回测在 2023-12-20 确实已经清至 0 持仓（自身从峰值回撤 `-4.66%`），但后续状态解除时重新买入，
到 2024-02-07 仍回撤 `-40.01%`。因此该信号不是能长期覆盖这一失效区间的“空仓开关”。

### 零成本日收益遮罩：仅作上界诊断，不能当作回测结果

为检验 AMV 是否有潜在日级 timing 信息，额外计算了下面的**非可交易**遮罩：

```text
r_overlay[t] = 0                         if lagged_state[t] is true
               baseline_daily_return[t]  otherwise
```

它没有交易成本、没有 T+1、并且在状态解除后假设可以无代价回到“原基线当日正持有的同一篮子”。结果看上去很好：

| 遮罩（非实际策略） | 累计收益 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: |
| 正确基线日收益 | `+323.22%` | `-40.98%` | `0.951` |
| AMV_downtrend 遮罩 | `+338.89%` | `-33.22%` | `1.089` |
| Breadth_plus_AMV 遮罩 | `+338.10%` | `-33.22%` | `1.044` |

这只是“若能暂停基线收益流并随后无损恢复”的乐观上界。它没有重建实际仓位，也没有反映该策略 H20 持仓路径被打断后
会失去哪些后续 alpha；不能用于选择规则、不能写入策略绩效表。下面的真实回测正是对这一假设的反证。

### 真实 QuantX 全清仓回测：结果否决

为避免把“只降低新买入仓位”误称为空仓，新增显式 PositionManager 插件：

```text
quantx/strategies/active_value_regime.py
quantx/tests/strategy/test_active_value_regime_plugin.py
```

插件仅读取已滞后一日的日状态 CSV；状态日对所有当前持仓输出 `sell_ratio=1.0`、`cash_deploy_ratio=0`、空
`entry_weights`。非状态日严格复现原基线的 `max_positions=10`、`portfolio_target`、`buy_only_new_positions=true`
空槽分配和每只 `10%` 目标权重。QuantX 引擎仍负责 close 成交、T+1、涨跌停、100 股整手、佣金、印花税、滑点
与 H20 sell rule；没有改动核心 engine。

两份生成配置：

```text
quantx/configs/strategies/generated/reward_multitask_7d_epoch010_active_value_regime_v1/
  reward_multitask_7d_epoch010_top10_h20_amv_downtrend_cash_2020_2026_all_a.yaml
  reward_multitask_7d_epoch010_top10_h20_breadth_amv_downtrend_cash_2020_2026_all_a.yaml
```

真实回测和正确基线的同口径比较：

| 策略 | 累计收益 | 年化 | 最大回撤 | Sharpe | 成交笔数 | 总成本 | 平均持有日 | 胜率 | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 正确 Top10-H20 基线 | `+323.22%` | `25.20%` | `-40.98%` | `0.865` | 1,663 | 20.08M | 28.88 | `59.39%` | 1.561 |
| AMV_downtrend 全清仓 | `+152.52%` | `15.52%` | `-40.01%` | `0.573` | 2,279 | 18.34M | 16.19 | `52.80%` | 1.545 |
| Breadth_plus_AMV 全清仓 | `+61.06%` | `7.71%` | `-41.66%` | `0.274` | 2,004 | 14.52M | 20.45 | `52.00%` | 1.228 |

对应 run：

```text
AMV_downtrend:
  quantx/runs/20260726_223942_reward_multitask_7d_epoch010_top10_h20_amv_downtrend_cash_2020_2026_all_a

Breadth_plus_AMV:
  quantx/runs/20260726_224031_reward_multitask_7d_epoch010_top10_h20_breadth_amv_downtrend_cash_2020_2026_all_a
```

审计确认它们真的执行了全清仓，而不是配置失效：

| 变体 | 风险状态日 | 有持仓的清仓计划日 | 计划全卖仓位数 | 实际 `position_manager_sell` | 平均投入比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMV_downtrend | 364 | 78 | 704 | 704 | `75.97%` |
| Breadth_plus_AMV | 202 | 45 | 447 | 447 | `86.27%` |

成本反而低于基线，是因为长期现金和更低的组合规模，不是因为换手不存在；真正损失来自 78/45 次卖出后无法无成本恢复
原 H20 持仓路径。AMV-only 将平均持有期从 28.88 天压到 16.19 天，收益少 `170.70` 个百分点，MDD 却只改善
`0.97` 个百分点。组合规则收益少 `262.16` 个百分点且 MDD 更差 `0.68` 个百分点。

**最终操作结论：**

```text
不部署 AMV_downtrend 全清仓。
不部署 Breadth_plus_AMV 全清仓。
不把零成本遮罩当作任何收益证据。
不在已查看的 2020-2026 窗口继续扫描 AMV 阈值、均线长度或组合条件。
```

下一步若继续，应另起一个预注册实验：只在独立开发期定义少量“降新仓/减仓而非全清仓”的 AMV 规则，冻结后在尾部
保留期验证，并把真实仓位切换成本和基线持仓连续性作为主指标。当前结果不支持直接把 AMV 用作策略的全空仓开关。

### 可复现命令与产物

```bash
cd ${HOME}/git/quantization/quantx

# 单元测试、插件测试、既有配置策略回归
conda run -n test env PYTHONPATH=. pytest -q \
  tests/strategy/test_active_value_regime_plugin.py \
  tests/tools/test_analyze_active_value_regime_alignment.py \
  tests/tools/test_analyze_market_regime_alignment.py \
  tests/strategy/test_config_strategy.py

# 重建自建 AMV 日状态和事件研究
conda run -n test env PYTHONPATH=. python -m quantx.tools.analyze_active_value_regime_alignment \
  --run-id 20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a \
  --analysis-id 20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a_active_value_alignment_v1 \
  --workers 4 --overwrite --json

# 先验证 YAML / 插件，再跑真实全清仓反事实
conda run -n test env PYTHONPATH=. python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/reward_multitask_7d_epoch010_active_value_regime_v1/reward_multitask_7d_epoch010_top10_h20_amv_downtrend_cash_2020_2026_all_a.yaml \
  --dry-run --json

conda run -n test env PYTHONPATH=. python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/reward_multitask_7d_epoch010_active_value_regime_v1/reward_multitask_7d_epoch010_top10_h20_amv_downtrend_cash_2020_2026_all_a.yaml \
  --output-dir runs --json

conda run -n test env PYTHONPATH=. python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/reward_multitask_7d_epoch010_active_value_regime_v1/reward_multitask_7d_epoch010_top10_h20_breadth_amv_downtrend_cash_2020_2026_all_a.yaml \
  --output-dir runs --json
```

主要可审计产物：

```text
quantx/artifacts/market_regime_alignment/
  20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a_active_value_alignment_v1.json
  20260726_201157_reward_multitask_7d_epoch010_top10_h20_portfolio_target_2020_2026_all_a_active_value_alignment_v1_daily.csv

每个真实 run 下：
  config.yaml, summary.json, metrics.json, daily_nav.json, trades.json,
  positions.json, closed_positions.json, rl_decisions.parquet, logs.txt
```
