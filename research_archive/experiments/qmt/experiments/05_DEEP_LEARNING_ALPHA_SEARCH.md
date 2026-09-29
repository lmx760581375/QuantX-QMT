# Deep Learning Alpha Search

> 研究目标：在严格 walk-forward、2026 forward 和 formal 成交约束下，寻找能显著提升一周持仓尺度横截面预测能力的本地可训练 Deep Learning alpha。

> 硬门槛：五年收益接近几十倍至 100 倍，平均股票持仓大于 5，五个自然年收益均为正，平均持仓约一周，2026 forward 为正，且不能依赖降低尾部仓位来卡口径。

> 当前状态：启动第 05 轮。所有探索脚本、配置、模型和 runs 均保留在项目 `.tmp/quantx-research`；未达到门槛前不融入项目策略库。

## 1. 背景

第 04 轮已经把传统 ML 和一系列规则修补方向推到 formal 裁判下验证。当前结论不是“市场不可预测”，而是：在现有 path/base 排序土壤内，继续做尾部补位、组层承接、行业热点独立化、静态曲线组合和超跌修复，都只能带来小增量，无法打开几十倍收益上限。

关键证据：

| 方向 | 开发期 | 2026 forward | 判读 |
| --- | ---: | ---: | --- |
| formal `path pool20 buy5` baseline | 5.48x | `+19.47%` | 稳定但不够厚 |
| `same_group_quality` | 5.93x | `+21.24%` | 组层补位小正增量，不是主收益层 |
| `path_sequence_path_topq_pool200_top20` rebalance5 | `+777.49%` | `+19.18%` | 平均持仓和持有周期合规，但收益远低于目标 |
| `robust_weekly_base_edge_blend_top20` | `+405.13%` from 2023 | `+29.71%` | 2026 更强，但开发期不完整且仍不够厚 |
| simplified daily sleeve | 90x-120x | 正收益 | 被 formal 涨停、跳变、停牌、手数和成本约束打掉，不能作为策略证据 |

因此第 05 轮不把 Deep Learning 当作简单替换 LightGBM 的工具，而是重点验证：小型序列模型是否能从过去一段时间的横截面状态中学习到 formal 可交易右尾，而不是只拟合单日特征或同一个 Top20 排序空间。

## 2. 研究纪律

1. 所有训练、诊断、临时配置和 runs 放在项目 `.tmp/quantx-research/deep-learning-alpha-search-v1/`。
2. 不随机切分时间序列；开发期使用 expanding/walk-forward，2026 单独 forward。
3. 特征只使用 T 日及以前可见信息；标签使用 T+1 open 入场、未来 open 出场或 formal-aware proxy。
4. 预测层不达标不进入账户层；账户层只承认标准 formal 成交口径。
5. 每个模型先过多尺度预测指标，再决定是否进入 formal account；不能只看单个 TopK label。
6. 不通过降低 Rank6-20 权重满足平均持仓要求；优先寻找自然等权 Top20 或一篮子持仓收益厚度。
7. 每轮都记录反事实分析：如果假设为真，应观察到什么；实际观察是否支持；失败时下一步为何改变方向。

## 3. 第 05 轮优先方向

### 3.1 formal_tradeable_sequence_soil_v1

假设：已有 LightGBM/path 模型的主要瓶颈不是完全没有预测力，而是 formal 可交易右尾被单日特征和简化标签错配。若从更宽候选池中直接学习“未来一周 formal 可交易且收益路径不脆弱”的样本，Top20 等权收益厚度应在开发期和 2026 同时超过现有 path baseline。

第一轮先不训练大模型，只复用已有 path/robust weekly 资产做土壤诊断：比较 base/path/robust 候选在 label、路径脆弱性、年度稳定性和 formal account 之间的断点。

最低晋级条件：

1. Top20 label 在 2022-2025 和 2026 同时明显高于 path baseline。
2. 年度 label 不靠单一年份贡献。
3. 进入 formal account 后不能被成交约束完全打掉。

### 3.2 compact_temporal_cross_section_model_v1

若土壤成立，训练本地可跑的小型 PyTorch 模型：TCN/GRU/Transformer Encoder 三者择一，参数规模控制在几十万到几百万。输入为过去 20-60 日横截面标准化序列，输出为 5 日收益、右尾概率、早期失败概率和路径稳健性多任务目标。

首版不引入图像和 diffusion，先验证结构化 market world tokens 是否优于已有 LGBM 单日特征；若不能优于，说明问题不在模型表达力，而在候选土壤或标签定义。

### 3.3 lightweight_market_world_encoder_v1

若序列模型有边际增益，再向 world model 靠近：用市场宽度、行业/概念状态、个股路径 token 做多任务预训练，预测未来 5/10/20/30 日路径分位、最大回撤、最大上涨和相对行业收益。该阶段仍以横截面排序和 formal account 为裁判，不以生成效果替代交易证据。

## 4. formal_tradeable_sequence_soil_v1

### 4.1 假设

已有 path/base 模型在 formal account 下仍有稳定正收益，但收益厚度停在 5x-9x 区间。若问题只是“单日排序模型没有捕捉过去一段时间的横截面路径”，那么复用已有 path sequence、tradable big winner、robust weekly 等资产时，应能看到某些序列/路径土壤在开发期和 2026 同时显著厚于当前 formal baseline。

本轮先做证据聚合，不新增训练模型，目的是给 Deep Learning 第一轮确定必须超越的基线。

### 4.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_sequence_soil_v1.py
sha256:813b719eaa4df42c733d45b5a4836ac6f4ef4f184f0191ed47e1023d00061fe3

.tmp/quantx-research/deep-learning-alpha-search-v1/formal_tradeable_sequence_soil_v1_summary.json
sha256:f30923c7c970d6e6d651443a5012fede7ebf1ba2993b7101b4e43dcf1eba13b4
```

聚合的既有证据包括：

1. `path-sequence-ranker-v1` 的 label 诊断和 rebalance5 formal account。
2. `tradable-big-winner-path-v1` 的可交易右尾 label 诊断。
3. `robust-weekly-candidate-v1` 的 label 诊断和 rebalance5 formal account。
4. `group-basket-substitute-v1` 的 same group quality formal account。

裁判优先级：formal account 高于 label 诊断；2026 forward 高于开发期局部提升；完整年度覆盖高于只覆盖 2023-2025 的候选。

### 4.3 结果

当前最强完整 formal baseline 仍是 `path_sequence_path_topq_pool200_top20` rebalance5：

| 口径 | 开发期/前向 | 总收益 | 最大回撤 | 平均持仓 | 平均持有 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path sequence rebalance5 | 2022-2025 | `+777.49%` | `-32.62%` | 19.40 | 9.36 | 完整且稳定，但远低于目标 |
| path sequence rebalance5 | 2026 | `+19.18%` | `-16.70%` | 19.48 | 9.94 | 前向正，但收益厚度不足 |
| robust weekly edge blend | 2023-2025 | `+405.13%` | `-34.88%` | 19.56 | 12.73 | 开发期不完整，持有略偏长 |
| robust weekly edge blend | 2026 | `+29.71%` | `-17.38%` | 19.50 | 11.02 | 2026 强一些，但不构成突破 |
| same group quality | 2022-2025 | 5.93x | `-39.23%` | 18.18 | 7.72 | 小正增量，不是主收益层 |
| same group quality | 2026 | `+21.24%` | `-17.42%` | 14.89 | 8.74 | 前向小幅优于 baseline |

label 层的序列/路径改进存在，但量级不足。以 path sequence label 诊断为例，2026 `all::pool200::path_topq::top20` 的 5 日 label 为 `+0.015905`，高于 base 的 `+0.012018`；但进入 formal account 后只得到 `+19.18%` 半年前向收益，仍无法接近五年几十倍目标。

### 4.4 反事实分析

第一反事实：如果现有序列/路径土壤已经足以支撑 Deep Learning 放大收益，那么 label 层优势应能在 formal account 中自然转成接近几十倍的收益斜率。实际 path sequence 虽然稳定，但 formal 开发期只有 8.77x final value，2026 半年只有 `+19.18%`。

第二反事实：如果问题主要是 2026 市场风格切换，robust weekly 应在开发期完整年度也明显更强。实际 robust weekly 2026 到 `+29.71%`，但 formal 开发期只有 2023-2025，且 2023-2025 总收益不超过 path sequence 的完整 2022-2025 基线。

第三反事实：如果组层承接是主收益引擎，same group quality 应显著超过 path sequence。实际它只是从 5.48x 级别提升到 5.93x，仍是局部修补。

第四反事实：如果继续在同一批单日聚合特征上调模型就能突破，上述 LightGBM/规则系列应该已经出现更大的 label 到 account 传导。实际所有改进都停在小数点级 label 增量和 5x-9x formal 账户区间。

### 4.5 判定

`baseline_established_with_signal`。

本轮没有找到足以直接进入 formal account 的新土壤，但明确了 Deep Learning 第一轮必须挑战的对象：`path_sequence_path_topq_pool200_top20` rebalance5。后续 DL 结果若只在 label 层小幅提升，或者不能超过 2026 path sequence，就不进入账户层。

下一步不是把 LightGBM 换成普通 MLP，而是测试真正使用过去一段时间 token 的小型序列模型，先看它能否超越同口径 base/path label。

## 5. compact_sequence_gru_v1

### 5.1 假设

若过去 20 日的横截面路径、行业/概念状态和市场宽度中存在 LightGBM 单日聚合特征没有吸收的短期可预测结构，那么小型 GRU 应该能在 Top20 label 上稳定超过 base，并在 2026 forward 中不反向。

本轮只做 label 层验证，不进入 formal account。原因是 Exp04 已经说明 label 小增量经常无法传导到正式账户；若预测层不能穿越 2026，就没有必要消耗 formal 回放。

### 5.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_compact_sequence_model_v1.py
sha256:66d89ebe4ca253d79cc1708a38cb3ecc9fcad47265cf5b4ced6ee5a90177b603

.tmp/quantx-research/deep-learning-alpha-search-v1/compact_sequence_gru_dev_2021_2025.json
sha256:32ac47a30d3b934750efefeba553909b471a7d19b06011e66096e07f9b404596

.tmp/quantx-research/deep-learning-alpha-search-v1/compact_sequence_gru_val63_2026.json
sha256:2a1b5a8685993a1a4b5b9a44063abb0c1e2352eb67621a970010abfba255e53f
```

口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | 基础 5 日 LightGBM 预测 Top200 |
| 样本频率 | due5 sessions |
| 输入 | 过去 20 日序列，每日 33 个结构化特征 |
| 模型 | PyTorch GRU，hidden size 48，4 epochs |
| 标签 | 同日候选池内未来 5 日超额收益五分位 |
| 开发期 | expanding walk-forward，2022-2025 OOS |
| 前向 | 2021-2025 train，2026 val63 forward |
| 对照 | `base_ml_rank_pct`、纯 GRU top quintile、GRU spread、25% 弱融合 |

特征只来自 T 日及以前；未来 5 日 label 为 T+1 open 至 T+6 open，并扣除当日全市场均值。2026 使用独立 forward prediction store，不把 2026 label 用于训练。

### 5.3 结果

Top20 label 结果：

| 口径 | 2022-2025 mean label5 | 相对 base | 正 label session | 2026 mean label5 | 相对 base | 2026 正 label session | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.009717` | `0` | 64.25% | `+0.005062` | `0` | 70.83% | 基线 |
| GRU topq | `+0.010125` | `+0.000409` | 61.14% | `+0.002659` | `-0.002404` | 58.33% | 前向反向 |
| GRU spread | `+0.008584` | `-0.001133` | 60.62% | `-0.000780` | `-0.005842` | 50.00% | 明确失败 |
| base 75% + GRU topq 25% | `+0.010660` | `+0.000944` | 65.80% | `+0.003485` | `-0.001577` | 66.67% | 开发期小增，前向失败 |
| base 75% + GRU spread 25% | `+0.010510` | `+0.000793` | 64.25% | `+0.002333` | `-0.002729` | 58.33% | 前向失败 |

年度拆分显示开发期增量并不稳。`base 75% + GRU topq 25%` 在 2022、2023、2024 高于 base，但 2025 基本持平；2026 则从 base 的 `+0.005062` 降到 `+0.003485`。

更关键的是，它明显低于已有 `path_sequence` LightGBM 特征模型的 2026 due5 结果：`due5::pool200::path_topq::top20` 为 `+0.012520`，而 compact GRU 最好只有 `+0.003485`。因此本轮不是“DL 小幅不够”，而是没有超过已经存在的序列特征工程。

### 5.4 反事实分析

第一反事实：如果过去 20 日序列中存在 GRU 容易捕捉、且 LightGBM 聚合特征没有吸收的稳健结构，纯 GRU 至少应在开发期和 2026 都超过 base。实际纯 GRU 在 2026 明显低于 base，spread 版本转负。

第二反事实：如果问题只是 GRU 分数尺度不适合直接排序，25% 弱融合应能保护 base 并保留增量。实际弱融合开发期有小增量，但 2026 仍反向，说明 GRU 学到的东西不穿越。

第三反事实：如果 Deep Learning 的表达力本身是瓶颈，简单 GRU 应至少超过已有 path sequence LightGBM。实际 2026 due5 对照中，path sequence `+0.012520` 远高于 GRU 弱融合 `+0.003485`，说明当前模型/输入方式没有形成更强表征。

第四反事实：如果开发期小增量有交易价值，2025 和 2026 不应最弱。实际 2025 基本无增量，2026 反向，最像是早期年份风格被学习后在近期失效。

### 5.5 判定

`rejected_with_diagnostic_signal`。

小型 GRU 证明“序列模型可以拟合一点开发期结构”，但没有证明“DL 能提升 alpha”。它不进入 formal account，不生成 PredictionStore，不融入策略库。

下一步不继续调 GRU hidden size、epoch、融合比例或 TopK。原因是这会变成在弱信号上调参，容易过拟合。更合理的方向是转向 `multitask_path_world_encoder_v1`：让模型先学习未来 5/10/20 日路径分布、最大回撤、最大上涨、相对行业收益和右尾/左尾概率，再用共享表征做一周横截面排序。若多任务路径表征仍不能超过 path sequence，则说明当前结构化 DL 路线暂时没有收益上限优势，需要寻找新的候选土壤而不是继续换网络。

## 6. multiscale_model_metrics_v1

### 6.1 假设

单一 Top20 label 容易误导研究方向：开发期小增量可能来自单一年份、单一市场状态或对 base 的微弱扰动，并不能证明模型有可交易预测力。第 05 轮需要一个模型层多尺度 metric，在进入 formal account 之前同时约束横截面能力、时间稳定性、2026 forward 穿越、强基线比较和后续账户门槛。

### 6.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/evaluate_multiscale_metrics_v1.py
sha256:f37b3ec9061debaf209c1c80c469aefa053e26c86a66a0cb3559dab5cc230282

.tmp/quantx-research/deep-learning-alpha-search-v1/multiscale_metrics_compact_sequence_gru_v1.json
sha256:e516e6516b5a3c9ad2081abe19107612c0025cfafa19dbf29010402a36b6d0d9
```

第一版 metric 分为五层：

| 尺度 | 目的 | 当前字段 |
| --- | --- | --- |
| 横截面 | 评估 Top20 是否真的更会选股 | mean label5、top/bottom quintile rate、positive session ratio |
| 时间稳定 | 防止单一年份贡献 | by-year label、positive years、worst-year label、worst-year delta |
| 前向穿越 | 防止开发期过拟合 | 2026 label、2026 delta、forward/dev retention |
| 强基线 | 防止只超过弱 base | gap vs `path_sequence` due5 Top20 |
| 账户门槛 | 决定是否进入 formal replay | 预测层通过后才生成 PredictionStore 和 formal account |

预测层晋级门槛暂定为：

1. 开发期 Top20 label 相对 base 至少 `+0.0015`。
2. 2026 forward 相对 base 不为负。
3. 开发期四个自然年均为正。
4. 2026/dev retention 至少 50%。
5. 必须超过已有 `path_sequence` 2026 due5 Top20 强基线。

综合分只用于排序和诊断，不替代硬门槛。当前定义：`0.30*dev_delta_score + 0.30*forward_delta_score + 0.20*year_stability + 0.20*strong_baseline_score`，各项裁剪到 `[0, 1]`。

### 6.3 结果

对 `compact_sequence_gru_v1` 的评分：

| 变体 | 综合分 | 判定 | 失败原因 |
| --- | ---: | --- | --- |
| base 75% + GRU topq 25% | 0.2944 | reject_before_formal_account | 开发期增量不足、2026 低于 base、forward retention 不足、低于 path sequence |
| base 75% + GRU spread 25% | 0.2793 | reject_before_formal_account | 同上 |
| GRU topq | 0.2409 | reject_before_formal_account | 同上 |
| base | 0.2000 | reject_before_formal_account | 低于 path sequence，不是新模型 |
| GRU spread | 0.2000 | reject_before_formal_account | 开发期增量不足、2026 低于 base、forward retention 不足、低于 path sequence |

最佳弱融合虽然在开发期 Top20 label 上有 `+0.000944`，但没有达到 `+0.0015` 的开发期门槛；更重要的是 2026 forward 相对 base 为 `-0.001577`，并且远低于 path sequence 2026 due5 Top20 的 `+0.012520`。

### 6.4 反事实分析

第一反事实：如果 GRU 的开发期增量是真实可迁移预测能力，多尺度指标中至少应通过 forward delta 和 retention。实际最佳 GRU 弱融合两项都失败。

第二反事实：如果只看开发期 label 就足够，综合评分应允许最佳弱融合晋级。新的 metric 明确拒绝它，说明该评估体系能拦住“开发期微增、前向反向”的典型过拟合。

第三反事实：如果 base 本身已经足够好，强基线比较不应改变结论。实际 base 在 2026 due5 上也低于 path sequence，说明后续模型必须挑战已有强基线，而不是挑战最弱 base。

### 6.5 判定

`metric_gate_established`。

多尺度 metric 成为第 05 轮之后所有模型的预测层裁判。模型只有通过该 gate，才允许进入 PredictionStore 和 formal account。下一轮 `multitask_path_world_encoder_v1` 必须同时报告这些指标，并额外加入路径分布任务自身的校准指标，例如 5/10/20 日路径方向、最大回撤分位覆盖、最大上涨分位覆盖和右尾/左尾召回。

## 7. multitask_path_world_encoder_v1

### 7.1 假设

`compact_sequence_gru_v1` 失败后，不能继续调 GRU 参数。更合理的反事实是：单一 5 日五分位标签太窄，模型没有被迫学习“市场路径世界”。若共享 encoder 同时学习未来 5/10/20 日收益分位、20 日右尾、5 日左尾和 5 日路径稳健性，再用多任务表征做一周横截面排序，应该比单任务 GRU 更稳，并且在 2026 forward 中至少不低于 base。

### 7.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_multitask_path_world_encoder_v1.py
sha256:2b244bc6168755b2893d194f6769dfe69c5a4ee767fcefb1699840b21f2748e0

.tmp/quantx-research/deep-learning-alpha-search-v1/multitask_path_world_dev_2021_2025.json
sha256:b1ccba2bd571b7c0d82eed45dddb21604dac08564693e8303adf02bb49f956c5

.tmp/quantx-research/deep-learning-alpha-search-v1/multitask_path_world_val63_2026.json
sha256:043a0882d26731ec4963f234f10ed53e859625681b7417acb706bcf0f5d1d53a

.tmp/quantx-research/deep-learning-alpha-search-v1/multiscale_metrics_multitask_path_world_v1.json
sha256:d281178701e2035531db33fd039bcb7f2f4b64bf20ca4e8a2c076f35cc3d9eb0
```

口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | 基础 5 日 LightGBM 预测 Top200 |
| 样本频率 | due5 sessions |
| 输入 | 过去 20 日序列，每日 48 个结构化特征 |
| 模型 | PyTorch GRU encoder，hidden size 64，6 个任务头 |
| 任务 | 5/10/20 日收益五分位、20 日右尾、5 日左尾、5 日路径稳健 |
| 开发期 | expanding walk-forward，2022-2025 OOS |
| 前向 | 2021-2025 train，2026 val63 forward |
| 变体 | 纯多任务分数、world composite、base 75% + 多任务 25% 弱融合 |

### 7.3 结果

Top20 主要结果：

| 口径 | 2022-2025 label5 | 相对 base | 2026 label5 | 相对 base | 2026 label20 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.009543` | `0` | `+0.008575` | `0` | `+0.008444` | 基线 |
| mt world | `+0.010775` | `+0.001232` | `+0.006432` | `-0.002143` | `+0.018860` | 20 日任务有信息，但 5 日前向掉线 |
| base 75% + q5 25% | `+0.011372` | `+0.001829` | `+0.007952` | `-0.000623` | `+0.019324` | 开发期过门槛，2026 低于 base |
| base 75% + world 25% | `+0.010871` | `+0.001328` | `+0.009000` | `+0.000425` | `+0.020660` | 2026 高于 base，但开发期略低于门槛 |

Top15 出现更清晰的局部信号：

| 口径 | 2022-2025 label5 | 相对 base | 2026 label5 | 相对 base | retention | 多尺度判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base 75% + q5 25% | `+0.012074` | `+0.001565` | `+0.008908` | `+0.002472` | 73.78% | 只因低于 path sequence 强基线被拒绝 |
| base 75% + world 25% | `+0.011805` | `+0.001296` | `+0.010009` | `+0.003573` | 84.79% | 开发期略低于门槛，且低于 path sequence |

多尺度评分中，最佳是 `blend_world_25::top15`，综合分 `0.6296`，失败原因是 `dev_delta_below_15bp` 和 `below_path_sequence_forward`；其次是 `blend_q5_25::top15`，综合分 `0.6037`，唯一失败原因是 `below_path_sequence_forward`。

强基线对照仍然很硬：2026 due5 `path_sequence` 的 `top15` label 为 `+0.018316`、`top20` label 为 `+0.012520`。多任务模型最好的 `top15` 只有 `+0.010009`，最好的 `top20` 只有 `+0.009000`，因此按既定 metric 不能进入 formal account。

### 7.4 反事实分析

第一反事实：如果多任务路径表征只是噪声，它不应比 compact GRU 更好。实际它在开发期和 2026 都明显强于 compact GRU，尤其 `top15` 弱融合能同时开发期和前向超过 base，说明多尺度路径任务确实比单一 5 日标签更接近有效表征。

第二反事实：如果多任务表征已经足够成为主 alpha，它应超过 path sequence 强基线。实际 2026 `top15/top20` 都明显低于 path sequence，说明当前 DL 仍没有打败已有 LightGBM 序列特征工程。

第三反事实：如果 20 日右尾任务能直接改善一周持仓，`mt_right20` 或 `mt_world` 应在 5 日 label 上稳定领先。实际它们经常提高 20 日 label，但 5 日 label 不稳定，说明长路径信息和一周交易目标之间存在错配。

第四反事实：如果弱融合只是防守，不应在 Top15 前向产生明显增量。实际 `blend_world_25::top15` 2026 相对 base 为 `+0.003573`，说明 DL 表征有局部互补价值，但还不能独立承担收益引擎。

### 7.5 判定

`rejected_with_stronger_signal`。

多任务路径 world encoder 是第 05 轮目前最有信息量的 DL 结果：它证明多任务路径表征优于普通 GRU，并在 Top15 上给出开发期与 2026 同向的局部增量。但它仍低于 path sequence 强基线，不进入 PredictionStore，不进入 formal account。

下一步不继续裸调多任务权重或 hidden size。更合理的方向是 `path_teacher_world_encoder_v1`：把已有 path sequence LightGBM 作为强教师/先验，让 DL 先学习并保持 path sequence 的排序能力，再用多任务路径表征只预测 teacher 失效或 formal 可交易性改善。核心问题从“DL 能否从零超过 base”切换为“DL 能否在不破坏 path sequence 强排序的前提下，识别它的失效区间和可交易增量”。

## 8. path_teacher_world_encoder_v1

### 8.1 假设

前两轮说明，裸 DL 表征能产生局部增量，但打不过已有 `path_sequence` 强基线。因此本轮不再让 DL 从零重排 Top200，而是把 `path_sequence_path_topq_pool200_top20` 作为 teacher：先固定 teacher Top20 候选池，只在这个强池内部测试 DL world encoder 是否能改善 Top10/Top15 的排序。

如果 DL 的价值是“识别 path sequence 失效或右尾质量差异”，那么 teacher Top20 内的弱融合应该在开发期和 2026 同时稳定超过 teacher 原排序。

### 8.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_path_teacher_world_encoder_v1.py
sha256:6e83c31a57c371c6e6faa5d87969850917ea49f7b948d22f631271a340c87104

.tmp/quantx-research/deep-learning-alpha-search-v1/path_teacher_world_dev_2021_2025.json
sha256:7549a91c924d69540abc39732ac55fb429097606e3761bbda2bfe5484d23fba5

.tmp/quantx-research/deep-learning-alpha-search-v1/path_teacher_world_val63_2026.json
sha256:0276a867343579050a45ca525d0ba886b46e1bc7d375844d05411b98f9c2d3c4
```

口径：

| 项 | 设置 |
| --- | --- |
| teacher | `path_sequence_path_topq_pool200_top20` PredictionStore |
| 候选池 | 每个 signal day 的 teacher Top20 |
| DL 模型 | 复用 `multitask_path_world_encoder_v1` |
| 评估 | teacher 原排序、teacher 90/80% + world/q5 10/20%、纯 world/q5/right20 |
| 开发期 | 2022-2025 OOS |
| 前向 | 2026 val63 forward |

### 8.3 结果

Teacher Top15 结果：

| 口径 | 2022-2025 label5 | 相对 teacher | 2026 label5 | 相对 teacher | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| teacher | `+0.011911` | `0` | `+0.018801` | `0` | 强基线 |
| teacher 90% + q5 10% | `+0.011938` | `+0.000027` | `+0.019193` | `+0.000393` | 前向小增，开发期几乎无增量 |
| teacher 90% + world 10% | `+0.011910` | `-0.000001` | `+0.019048` | `+0.000248` | 近似持平 |
| teacher 80% + world 20% | `+0.011783` | `-0.000127` | `+0.018702` | `-0.000099` | 变弱 |
| right20 only | `+0.012469` | `+0.000558` | `+0.015580` | `-0.003221` | 开发期看似有效，前向失败 |

Teacher Top20 内，所有变体与 teacher 基本相同，因为候选池本身只有 20 只；重排不能改变 Top20 集合，只能改变 Top10/Top15。因此本轮没有账户层意义上的新持仓集合，最多只能影响买入优先级或 Top15 子集。

### 8.4 反事实分析

第一反事实：如果 DL 能稳定识别 teacher 内部失效样本，弱融合应该在开发期和 2026 都有明确正增量。实际 `teacher 90% + q5 10%` 开发期只增 `+0.000027`，几乎为零；2026 仅增 `+0.000393`，属于噪声级别。

第二反事实：如果 20 日右尾任务能抓住 teacher 内部潜在赢家，`right20 only` 不应在 2026 大幅低于 teacher。实际它开发期增 `+0.000558`，但 2026 反向 `-0.003221`，说明右尾任务对一周交易目标仍然错配。

第三反事实：如果 teacher 内重排是突破口，Top15 增量至少应接近前面多任务裸模型相对 base 的增量。实际 teacher 已经很强，DL 在强池里几乎没有可提取空间。

第四反事实：如果本轮值得进入 formal account，必须生成不同且更好的 Top20/持仓集合。实际 Top20 集合不变，Top15 增量极小，不足以支付额外模型复杂度和过拟合风险。

### 8.5 判定

`rejected_as_marginal_teacher_rerank`。

Teacher 内重排没有形成可交易级别增量，不进入 PredictionStore，不进入 formal account。至此，第 05 轮已经验证了三条结构化 DL 路线：普通序列 GRU、多任务路径 world encoder、path teacher 弱融合。结论一致：DL 表征有局部信息，但在现有 Top200/Top20 土壤里，仍无法超过 `path_sequence` 强基线到足以改变收益量级。

下一步不继续在同一个 teacher Top20 内做网络、权重或 TopK 微调。方向应转为寻找新的候选土壤或外部方法：例如从更早期的事件传播、盘口/涨停前夜代理、低相关周期或更贴近 Kronos 的 masked pretraining 中寻找新的可交易右尾来源，而不是在已知强池里挤小数点增量。

## 9. event_group_continuation_soil_scan_v1

### 9.1 假设

第 04 轮后段和第 05 轮前三个 DL 实验都指向同一个瓶颈：继续在基础 ML/path Top200 或 teacher Top20 里重排，增量接近枯竭。若 A 股短线的主要金子来自概念/行业事件传播、启动结构、涨停扩散后的可交易承接，那么旧的事件类候选诊断中应该能看到一批开发期和 2026 同时为正、且不只是复制 base/path 排序的候选土壤。

本轮先不训练新模型，只扫描既有 `/tmp` 事件/启动候选诊断，统一不同实验的 `mean_daily_label`、`mean_daily_label5`、年度稳定性、相对本地基线增量和平均入选数量。目的不是找一个漂亮单点，而是判断是否有足够厚的新候选土壤值得进入下一轮 DL。

### 9.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/scan_event_soils_v1.py
sha256:817d4e964f3de9de384a8183c37794b9fb65c8cbd85d4f18d1900adb6454206d

.tmp/quantx-research/deep-learning-alpha-search-v1/event_soil_scan_v1.json
sha256:d0e27ff15a805ff0778fdeb45b546639ed1d2f199fd46e991a613a85cfb6132f
```

扫描输入包括：

1. `event-propagation-candidates-v1`。
2. `post-event-setup-v1`。
3. `pre-breakout-quintile-path-v1`。
4. `event-continuation-learner-v1`。
5. `right-tail-event-candidate-v1`。
6. `startup-structure-memory-v1`。
7. `limitup-concept-propagation-v1`。

诊断 gate：开发期 label 为正、2026 label 为正、开发期自然年为正、平均入选数量不低于 5、若有本地 `delta_vs_base/path/ml` 则开发期和 2026 不应同时为负。该 gate 只用于决定下一步研究方向，不等同于 formal account 晋级。

### 9.3 结果

最靠前的候选大多仍是旧 `path_topq` 土壤，而不是新的事件承接土壤：

| 候选 | TopK | 2022-2025 label | 2026 label | 年度 | 本地增量 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pre_breakout all pool300 path_topq` | 10 | `+0.017188` | `+0.023005` | 4/4 | dev `+0.000231`, 2026 `+0.004769` | 强，但本质仍是 path_topq |
| `pre_breakout due5 pool200 path_topq` | 10 | `+0.018490` | `+0.020716` | 4/4 | dev `+0.000894`, 2026 `+0.015147` | 强，但仍是旧路径排序 |
| `right_tail_event due5 pool100 p_winner` | 10 | `+0.017709` | `+0.017534` | 4/4 | dev `+0.000113`, 2026 `+0.011965` | 可作为诊断，不够宽 |
| `limitup_concept all pool100 concept_limit_burst` | 15 | `+0.018394` | `+0.014556` | 5/5 | dev `+0.000763`, 2026 `+0.001082` | 最像新事件土壤，但增量薄 |
| `limitup_concept all pool200 concept_limit_burst` | 20 | `+0.015683` | `+0.012860` | 5/5 | dev `+0.000025`, 2026 `+0.000842` | 可交易宽度更好，增量接近噪声 |

明确失败的方向也很重要：

| 候选族 | 观察 | 判读 |
| --- | --- | --- |
| 直接事件强度 `event_propagation` | 多数开发期 label 为负，2026 局部转正 | 追事件强度不是鲁棒承接 |
| `post_event_setup` | 开发期多为负，2026 局部转正 | 2026 风格线索，不可硬推 |
| 旧 `event_continuation_learner` | 开发期为负，2026 明显转正 | 典型风格翻转，不能据此训练新策略 |
| `startup_structure_memory path_topq` | label 强，但相对 path 本地增量为 0 或负 | 主要是 path_topq 的重命名，不是新土壤 |

### 9.4 反事实分析

第一反事实：如果组层事件承接本身是足够肥的新收益土壤，那么直接事件传播、post-event setup、事件 continuation learner 应至少在开发期整体为正。实际它们多数开发期为负，只在 2026 局部转强，说明更像风格阶段，而不是稳健主引擎。

第二反事实：如果 `limitup_concept` 是独立强 alpha，Top15/Top20 应明显超过 `ml_score`。实际 `concept_limit_burst` Top15 的开发期增量只有 `+0.000763`，2026 只有 `+0.001082`；Top20 增量更薄，接近噪声。

第三反事实：如果 `right_tail_event p_winner` 是新的候选生成器，Top15/Top20 宽度不应快速衰减。实际它的 Top10 还可以，但 Top15/Top20 开发期相对 base 多数为负，说明它更像窄头部诊断，而不是满足平均持仓和组合厚度的自然土壤。

第四反事实：如果扫描榜首就是答案，应该不是旧 path_topq 变体占据前排。实际榜首仍然主要来自 `path_topq`，说明当前可用结构化事件特征没有自然打开比 path sequence 更高的收益上限。

### 9.5 判定

`soil_scan_completed_without_fat_new_soil`。

本轮扫描没有找到足以直接进入 formal account 的新事件土壤，但给出两个有价值观察：

1. `limitup_concept` 的概念涨停爆发 Top15 是最接近“组层事件承接”的正向线索，但增量太薄，不足以单独成为几十倍策略来源。
2. `right_tail_event p_winner` 和 `pre_breakout path_topq` 提示“右尾分类/路径分位”有信息，但仍主要受已有 base/path 候选池约束。

下一步只允许做一次很窄的模型结构验证：参考 RankGLU 的 gated residual score formation，在已有 pre-breakout/path 特征上训练小型 residual gate，测试它是否能自然提高横截面预测力。若该模型仍只给出薄增量或 2026 单点亮点，则不继续调参，转向更接近账户边际 sleeve 或 masked world pretraining 的方向。

## 10. rankglu_residual_gate_v1

### 10.1 假设

外部线索 `RankGLU: Residual Gated Score Formation for Cross-Sectional Stock Prediction` 的核心启发是：不要让神经网络从零预测 alpha，而是让它在已有强排序上学习 gated residual 修正。若当前问题是 base/path 分数在某些事件/路径状态下需要非线性修正，小型 RankGLU 风格 head 应该能在开发期和 2026 同时提高 Top15/Top20 label，并且比纯 q4 概率更稳。

本轮复用 `pre_breakout_quintile_path_v1` 的 panel 构造和特征，新增一个小型 PyTorch gated residual head。模型输入为同日候选池内 T 日可见的路径、组层、市场宽度和 base rank 特征；标签为同日候选池内未来 5 日超额收益五分位。2026 forward 使用 2021-2025 train panel 训练，单独在 2026 val63 prediction store 上评估，不把 2026 label 用于训练。

### 10.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_rankglu_residual_gate_v1.py
sha256:2d886f1719246bdace2f0070bd22d1ed373e7123f6a9332ba933be5c20a81515

.tmp/quantx-research/deep-learning-alpha-search-v1/rankglu_residual_gate_dev_2021_2025.json
sha256:0f6ae71088413f51ff43f1b6c3cc05f6d67f21a2234987dd2327dbf2b5946e8c

.tmp/quantx-research/deep-learning-alpha-search-v1/rankglu_residual_gate_val63_2026.json
sha256:52161afc264366387bfb260815fdcb306676c2f4059e01baa38a656a9e4fff19
```

口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | base 5d prediction Top100/200/300/500 |
| 特征 | `pre_breakout_quintile_path_v1` 的 64 个结构化特征 |
| 模型 | RankGLU-style gated residual head，hidden 64，8 epochs |
| 开发期 | 2022-2025 expanding walk-forward OOS |
| 前向 | 2021-2025 train，2026 val63 forward |
| 变体 | base、pre_breakout、纯 `rankglu_q4`、纯 gate、base+gate 10/20%、base+q4 20%、base+pre+gate |

该实验只做 label 层验证，不进入 formal account。原因是预测层若不能稳定超过 base/path 强基线，账户层大概率只会重复第 04 轮“label 小增量无法传导”的问题。

### 10.3 结果

主要结果如下：

| 口径 | 2022-2025 label5 | 相对 base | 2026 label5 | 相对 base | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all pool100 base top10` | `+0.016957` | `0` | `+0.018237` | `0` | 强 base |
| `all pool100 rankglu_q4 top10` | `+0.013289` | `-0.003668` | `+0.020707` | `+0.002470` | 2026 亮，但开发期明显失败 |
| `all pool100 base_gate10 top15` | `+0.014030` | `+0.000290` | `+0.014104` | `+0.000630` | 同向薄增量 |
| `all pool100 base_gate20 top15` | `+0.013985` | `+0.000245` | `+0.013988` | `+0.000514` | 同向薄增量 |
| `all pool100 rankglu_q4 top15` | `+0.011604` | `-0.002136` | `+0.014417` | `+0.000943` | 2026 小正，开发期失败 |
| `all pool100 base_gate10 top20` | `+0.012406` | `-0.000156` | `+0.011695` | `-0.000323` | Top20 不成立 |
| `due5 pool100 base_gate20 top15` | `+0.015063` | `+0.000720` | `+0.004756` | `+0.003027` | due5 base 很弱，绝对值不够 |
| `due5 pool300 rankglu_q4 top15` | `+0.011773` | `-0.002570` | `+0.011410` | `+0.009681` | 2026 亮，开发期失败 |

年度拆分暴露了纯 `rankglu_q4` 的关键问题：

| 口径 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all pool100 base top10` | `+0.027000` | `+0.011345` | `+0.015950` | `+0.013460` | `+0.018237` |
| `all pool100 rankglu_q4 top10` | `+0.022505` | `+0.003811` | `+0.013511` | `+0.013331` | `+0.020707` |
| `all pool100 base_gate10 top15` | `+0.020561` | `+0.009409` | `+0.013518` | `+0.012602` | `+0.014104` |
| `all pool100 base top15` | `+0.020670` | `+0.009489` | `+0.012545` | `+0.012225` | `+0.013474` |

纯 `rankglu_q4` 在 2026 抓到一些右尾，但 2023 严重掉线，开发期总体低于 base。弱融合 gate 在 Top15 上有同向增量，但开发期只有 `+0.000245` 到 `+0.000290`，2026 只有 `+0.000514` 到 `+0.000630`；这个量级不足以解释五年几十倍收益缺口。

### 10.4 反事实分析

第一反事实：如果 RankGLU gated residual 是当前缺失的核心模型结构，纯 `rankglu_q4` 或 gate 分数应在开发期和 2026 都超过 base。实际纯 `rankglu_q4` 2026 Top10 很亮，但 2022-2025 相对 base 为负，尤其 2023 明显失效。

第二反事实：如果弱融合能把 2026 右尾优势稳定转化为策略增量，Top20 应至少不被破坏。实际 `base_gate10/20` Top15 有薄增量，但 Top20 为负，说明模型更像头部微调，不是平均持仓大于 5 的自然宽篮子 alpha。

第三反事实：如果本轮值得进入 formal account，预测层增量应超过第 05 轮 metric 的开发期 `+0.0015` 门槛，并接近或超过 path sequence 强基线。实际最佳同向弱融合开发期只有约 `+0.00029`，远低于门槛；纯 q4 虽有 2026 亮点，但开发期为负。

第四反事实：如果 2026 的纯 q4 亮点是稳定规律，那么开发期至少应有相似年份支持。实际开发期 2023 掉到 `+0.003811`，说明该信号可能是 2026 风格特化，不能在看到 2026 后硬卡成策略。

第五反事实：如果问题只是 LightGBM 表达力不足，小型神经 residual head 应自然超过基于相同特征的 LightGBM path_topq。实际它没有超过，说明瓶颈更可能仍在候选土壤和账户目标，而不是 score formation 的非线性表达。

### 10.5 判定

`rejected_as_forward_bright_but_dev_unsupported`。

RankGLU-style residual gate 证明了一个有用但不足的事实：神经网络能在 2026 抓到部分右尾形态，且 Top15 弱融合有同向薄增量。但它没有通过多尺度 gate，也没有改变当前收益上限判断。该实验不生成 PredictionStore，不进入 formal account，不融入策略库。

下一步不继续调 hidden size、epoch、gate 权重、Top10/Top15 或 2026 亮点组合。更合理的研究转向是：

1. 从“预测单只股票未来 5 日分位”改为“预测账户日新增 sleeve 的边际贡献”，直接把资金占用、重叠持仓、买入失败和卖出受阻纳入训练标签。
2. 或进入更接近 unified market world model/Kronos 的 masked pretraining：先在股票-行业/概念-市场多尺度 token 上预训练局部路径表征，再用少量监督头预测可交易右尾，而不是在同一批手工特征上换 head。
3. 若继续事件方向，必须寻找新的 point-in-time 信息或更早期的组层主线生命周期标签；现有静态行业/概念和涨停代理只能提供局部解释，不能单独打开收益厚度。

第 05 轮当前结论更新为：结构化 DL 在已有候选土壤内存在局部预测信息，但尚未找到能超过 path sequence 强基线并支撑五年几十倍收益目标的模型方向。下一轮应优先做账户边际 sleeve 标签或 masked world pretraining，而不是继续重排 base/path TopK。

## 11. account_marginal_sleeve_target_prior_v1

### 11.1 为什么先做历史复核

第 10 节后自然想到的方向是：既然单股 5 日 open-to-open label 与 formal 账户之间传导很差，是否应该直接训练“账户日新增 sleeve 的边际贡献”。但第 04 轮 Exp81 已经做过非常接近的实验，若不先吸收这条负证据，很容易把第 05 轮带回同一个局部最优。

本节不新增训练，只复核既有 `/tmp` 资产，判断“账户边际 sleeve 标签”是否仍值得作为第 05 轮下一主方向。

### 11.2 既有资产

相关脚本和结果来自第 04 轮：

```text
.tmp/quantx-research/account-marginal-sleeve-v1/write_account_marginal_sleeve_predictions.py
sha256:c72159204d9fbcd51e4b3ad05b9adb9fe752095c23b8d6cd1f66ebf9407cb53f

.tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_path_exec20_pool500_dev_2021_2025_top20_predictions.json
sha256:cc29a33894647f0f0cec236c6906d5d1bda4517926109d759751113137df66a0

.tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_base_exec20_pool500_dev_2021_2025_top20_predictions.json
sha256:8143b8b913aa2e6b33ba83aeb585ad04936a30e3b9d3c96ba55fdc34e497099e

.tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_base_exec20_pool500_val63_2026_top20_predictions.json
sha256:1a8deaeea87bcc6c8091a543bb66b5c465132d0fdf984b6026e6e0b50a88766d
```

formal overlap 账户裁判：

```text
.tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_path_exec20_pool500_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:a75c9df49f4a4f5d45de2db344288d5a84828165c55daf36548fcae4d5ef7275

.tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_base_exec20_pool500_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:5ff5ed16adef7bf700fa177eeded079caab450f3a3442980176e58f9de87b9f1

.tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_base_exec20_pool500_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:3b2e11e56624d79f97f186aca11326ca15096ad143a9099740c1e46bfc25f052
```

口径要点：基础 5 日 ML Top500 宽池，目标为近似账户边际 sleeve 可执行净收益；惩罚 open gap、历史不足、类涨停开盘和退出失败；formal 裁判为 Top20 pool、补位买5、hold5、`gap25+hist120`。

### 11.3 复核结果

代理层看起来有效：`base_exec_20` 把 2022-2025 的 entry ok 从 `96.87%` 提到 `99.72%`，平均日 exec target 从 `+0.569%` 提到 `+0.814%`；2026 也从 `+0.145%` 提到 `+0.189%`。

但 formal 账户裁判否决了它：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均持有 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 7.69 | 原强基线 |
| `path_exec_20` | 2022-2025 | 2.85x | `-41.27%` | 17.02 | 7.63 | 破坏 path 右尾 |
| `base_exec_20` | 2022-2025 | 3.92x | `-27.02%` | 17.59 | 7.64 | 更防守但收益变薄 |
| path baseline | 2026 | `+19.47%` | `-17.42%` | 14.52 | 8.79 | 前向正 |
| `base_exec_20` | 2026 | `-4.60%` | `-19.85%` | 14.79 | 7.82 | 前向失败 |

特征重要性也解释了失败原因：模型高度依赖 `market_ret60_median`、`market_ret5_median`、`market_breadth60`、`market_ret20_disp`、`market_ret20_median`、`market_breadth20`。它确实学到了市场风险、宽度和横截面离散度，但这更像“可执行/防守过滤器”，不是能保留高弹性右尾的 alpha 生成器。

### 11.4 反事实分析

第一反事实：如果 formal 执行损耗是主矛盾，账户边际标签应在正式账户中超过 path baseline。实际开发期低于 path，2026 从正收益变成负收益。

第二反事实：如果只需要在账户标签上加 path 锚，`path_exec_20` 应至少接近 path baseline。实际它开发期只有 2.85x，且 2023 为负，说明该标签会削掉 path 右尾。

第三反事实：如果可执行性提升就是收益提升，entry ok 从 `96.87%` 提到 `99.72%` 后 formal 收益不应下降。实际收益显著下降，说明右尾收益并不只是“买不到污染”；过度优化可执行性会把高收益弹性一起切掉。

第四反事实：如果市场状态因子足以因果识别账户边际 sleeve，它不应在 2026 转负。实际 2026 代理层小幅变好，账户层转负，说明代理目标仍然和真实资金路径错配。

### 11.5 判定

`prior_rejected_do_not_repeat_as_main_direction`。

账户边际 sleeve 标签可以作为未来诊断工具，但不作为第 05 轮下一主实验。继续调 `entry_fail_target`、`cost_proxy`、融合权重或 TopK，会回到第 04 轮已经证明的防守过滤路线，与用户要求的“鲁棒、容易出收益、不是降低尾部权重/防守卡口径”相冲突。

下一方向因此从“账户边际单股标签”改为更结构性的 representation learning：参考 Kronos / unified market world model，做小型 masked world pretraining。核心不是在同一组手工特征上换 head，而是让模型先学习股票-行业/概念-市场多尺度 token 的时间结构，再用少量监督头检验是否能产生新的可交易右尾表征。若 masked pretraining 仍不能在预测层超过 path sequence 强基线，就应继续寻找新的数据/候选源，而不是扩大模型。

## 12. masked_world_pretraining_gru_v1

### 12.1 假设

前面的 GRU、多任务 world encoder、teacher rerank 和 RankGLU residual gate 都是在监督标签上直接训练。若问题不是模型头，而是模型没有先学习“市场世界”的局部状态转移，那么参考 Kronos / unified market world model 的更小可证伪版本应当先做自监督：随机 mask 过去 20 日结构化 token，再要求 encoder 重建被 mask 的路径、组层和市场状态，之后用同一个 encoder 微调 5/10/20 日路径分位、右尾、左尾和路径稳健任务。

本轮暂时不用图像 tokenizer 和 diffusion。原因是正式 `20_MARKET_WORLD_MODEL_DESIGN.md` 的完整版本过大，第一步应先判断“masked pretraining 这个训练目标”是否有预测层增量，而不是把图像、diffusion、模型容量和数据管线同时引入。

### 12.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_masked_world_pretrain_v1.py
sha256:70df3b03592bb62b1cac92aa6310136b30aaee59f89b8a4fe4819693b5d338fb

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_smoke_2021_2023q1.json
sha256:c7c61c74892c97877ca9f9c39caeb05a889f6060020f44969f746fc920aa3cb3

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_dev_2021_2025.json
sha256:c95d21f97bf53a32899427a5ebe127dd54e905c0a7c110be98978230ab869797

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_val63_2026.json
sha256:70c8cee3211c4c6a11cb11521eac3b4ffa5242908df9decf1799746d42bf7e74
```

口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | base 5d prediction Top200 |
| 样本频率 | due5 sessions |
| 输入 | 过去 20 日、48 个结构化 token，复用 `multitask_path_world_encoder_v1` panel |
| 预训练 | mask 20% 时间-特征 token，重建标准化输入 |
| encoder | 小型 GRU，hidden 64 |
| 微调任务 | 5/10/20 日收益五分位、20 日右尾、5 日左尾、5 日路径稳健 |
| 开发期 | 2022-2025 expanding walk-forward OOS |
| 前向 | 2021-2025 train，2026 val63 forward |

### 12.3 结果

完整开发期结果比普通多任务 GRU 更有信息量：

| 口径 | 2022-2025 label5 | 相对 base | 2026 label5 | 相对 base | 2026 label20 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base Top15 | `+0.010509` | `0` | `+0.006436` | `0` | `+0.002212` | 基线 |
| `mw_world` Top15 | `+0.012183` | `+0.001674` | `+0.007165` | `+0.000728` | `+0.011936` | 开发期过门槛，2026 小正 |
| `blend_world_25` Top15 | `+0.011274` | `+0.000765` | `+0.008617` | `+0.002181` | `+0.014551` | 2026 最稳，但开发期不足 |
| `blend_q5_25` Top15 | `+0.012132` | `+0.001623` | `+0.007854` | `+0.001417` | `+0.013072` | 开发期过门槛，2026 正 |
| base Top20 | `+0.009543` | `0` | `+0.008575` | `0` | `+0.008444` | 基线 |
| `mw_world` Top20 | `+0.011851` | `+0.002308` | `+0.004691` | `-0.003883` | `+0.008065` | 开发期强，2026 失败 |
| `blend_world_25` Top20 | `+0.011402` | `+0.001859` | `+0.007718` | `-0.000857` | `+0.013825` | Top20 前向失败 |
| `blend_q5_25` Top20 | `+0.011558` | `+0.002016` | `+0.007356` | `-0.001219` | `+0.017060` | Top20 前向失败 |

年度稳定性方面，`mw_world::top15` 开发期四年均为正：2022 `+0.019336`、2023 `+0.009759`、2024 `+0.013509`、2025 `+0.005779`。但 2025 明显变薄，说明模型可能在近期风格上已经衰减。

与第 05 轮强基线比较仍不过关：path sequence 2026 due5 Top15 为 `+0.018316`，Top20 为 `+0.012520`；masked GRU 最好的 2026 Top15 只有 `+0.008617`，Top20 还低于 base。

### 12.4 反事实分析

第一反事实：如果 masked pretraining 完全无效，它不应比直接监督多任务 GRU 更好。实际 `mw_world` Top15 开发期相对 base `+0.001674`，Top20 `+0.002308`，明显强于早前普通多任务 GRU 的多数口径，说明自监督重建确实学到了一些有用状态表征。

第二反事实：如果 masked GRU 已经足以成为新 alpha，2026 Top15/Top20 应超过 path sequence 强基线。实际 2026 Top15 仅 `+0.008617`，离 path sequence `+0.018316` 很远，Top20 甚至低于 base。

第三反事实：如果收益目标是稳定的一周宽篮子，Top20 不应在 2026 反向。实际 Top20 开发期增量很强，但 2026 全部低于 base，说明该表征更像 Top15 局部信号，还没有自然满足平均持仓和宽篮子收益厚度。

第四反事实：如果 20 日路径任务能直接转化为一周持仓，2026 label20 的改善应同步改善 label5。实际 `blend_q5_25` Top20 的 2026 label20 达 `+0.017060`，但 label5 为 `+0.007356`、低于 base，说明长路径世界表征和一周交易兑现仍有错配。

### 12.5 判定

`rejected_but_pretraining_signal_confirmed`。

masked structured world pretraining 是第 05 轮目前最有方向感的 DL 线索：它证明自监督市场状态重建比单纯监督 GRU 更接近有效表征，开发期 Top15/Top20 均能给出明确正增量。但它仍未超过 path sequence 2026 强基线，也没有在 Top20 forward 保持正增量，因此不进入 PredictionStore，不进入 formal account。

下一步不是继续调 GRU epoch、mask ratio 或融合比例，而是回答用户提出的问题：为什么不用 Transformer。现在已经有足够证据说明 pretraining 目标本身有信号，值得做同口径轻量 Transformer 版。Transformer 版必须小模型、本地可训，只替换时间轴 encoder，复用相同 panel、相同 masked pretrain、相同多任务监督和相同强基线 gate；若仍不过关，则说明问题不是 GRU 表达力，而是候选土壤/标签目标仍不足。

## 13. masked_world_pretraining_transformer_v1

### 13.1 假设

用户指出一个合理问题：既然方向是 Kronos / unified market world model，为什么先用 GRU 而不是 Transformer。第 12 节已经证明 masked structured pretraining 有一定预测层增量，但 GRU 版 2026 仍打不过 path sequence 强基线。因此本轮只替换 encoder：把 GRU 改成轻量时间轴 Transformer，复用完全相同的 sequence panel、mask 重建目标、多任务监督头和 TopK label gate。

如果瓶颈主要是 GRU 表达力不足，Transformer 应在开发期和 2026 同时超过 GRU masked 版本，并缩小与 path sequence 强基线的差距。

### 13.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_masked_world_pretrain_transformer_v1.py
sha256:b88aa5434f36ea5f9e1c99c1268bab0ed2807307b17d6b57bf8e6ca8131ef288

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_transformer_smoke_2021_2023q1.json
sha256:41b713b5fc3aedae57cb4a75e2f230a61d89bb8f821211f5216773a823357d5a

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_transformer_dev_2021_2025.json
sha256:5004da479badc63ceb6aa3955e414e109457a3a0a588d803084fab6c2b2c60c1

.tmp/quantx-research/deep-learning-alpha-search-v1/masked_world_pretrain_transformer_val63_2026.json
sha256:b3270fb20f8bf25b6025e82ed4a82f1313d06fe3d77c8c59b8abad3ec846a888
```

口径：

| 项 | 设置 |
| --- | --- |
| 数据 | 与 `masked_world_pretraining_gru_v1` 完全相同 |
| encoder | input projection + 2 层 `TransformerEncoderLayer` |
| hidden | 64 |
| heads | 4 |
| pooling | last token + mean token |
| 预训练 | mask 20% 时间-特征 token，重建标准化输入 |
| 微调 | 5/10/20 日分位、右尾、左尾、路径稳健 |

### 13.3 结果

主要结果：

| 口径 | 2022-2025 label5 | 相对 base | 2026 label5 | 相对 base | 2026 label20 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base Top15 | `+0.010509` | `0` | `+0.006436` | `0` | `+0.002212` | 基线 |
| `mw_world` Top15 | `+0.011957` | `+0.001448` | `+0.006226` | `-0.000210` | `+0.009595` | 开发期略低于门槛，2026 失败 |
| `blend_world_25` Top15 | `+0.009908` | `-0.000601` | `+0.009769` | `+0.003333` | `+0.011915` | 2026 亮，开发期失败 |
| `blend_q5_25` Top15 | `+0.009383` | `-0.001126` | `+0.010166` | `+0.003730` | `+0.020288` | 2026 亮，开发期失败 |
| base Top20 | `+0.009543` | `0` | `+0.008575` | `0` | `+0.008444` | 基线 |
| `mw_world` Top20 | `+0.011059` | `+0.001517` | `+0.004932` | `-0.003643` | `+0.008887` | 开发期过门槛边缘，2026 失败 |
| `blend_q5_25` Top20 | `+0.010678` | `+0.001135` | `+0.005883` | `-0.002691` | `+0.015878` | 2026 label20 强，label5 失败 |

年度拆分显示 Transformer 也有近期衰减：`mw_world::top20` 为 2022 `+0.016076`、2023 `+0.010276`、2024 `+0.011278`、2025 `+0.006328`。`blend_q5_25::top15` 到 2025 只剩 `+0.000807`，说明 2026 Top15 的亮点没有足够开发期支持。

与 GRU masked 对比：Transformer 的开发期 Top20 增量 `+0.001517` 低于 GRU 的 `+0.002308`；2026 Top15 融合最高 `+0.010166`，略高于 GRU 的 `+0.008617`，但仍远低于 path sequence 2026 Top15 `+0.018316`。Top20 在 2026 仍全面低于 base。

### 13.4 反事实分析

第一反事实：如果问题只是 GRU 表达力不足，Transformer 应在开发期明显超过 GRU。实际 Transformer 开发期不如 GRU，只有 `mw_world::top20` 勉强过 `+0.0015` 门槛。

第二反事实：如果 Transformer 捕捉到更强的一周可交易右尾，2026 Top20 不应失败。实际 Top20 2026 全部低于 base，说明时间轴 self-attention 没有解决宽篮子收益厚度问题。

第三反事实：如果 2026 Top15 融合亮点是可迁移规律，开发期同口径应为正增量。实际 `blend_q5_25::top15` 和 `blend_world_25::top15` 开发期均低于 base，不能据此硬推。

第四反事实：如果 masked world pretraining 已经足够接近 unified market world model，强基线差距应明显缩小。实际 Transformer 2026 Top15 最高 `+0.010166`，仍远低于 path sequence `+0.018316`。

### 13.5 判定

`rejected_as_encoder_capacity_not_main_bottleneck`。

轻量 Transformer 没有自然突破 GRU masked pretraining，也没有通过强基线 gate。它说明：当前瓶颈不是简单把 GRU 换成 Transformer，而是当前 token 只按“单股票时间轴”建模，缺少同一交易日横截面内的股票-行业/概念-市场联合注意力；同时候选仍被 base Top200 约束，收益土壤没有变厚。

下一轮不继续堆 Transformer 层数、head 数或 hidden size。更合理的方向是改建模对象：从单股票时间序列 encoder 切到**横截面-时间联合 world model**。候选方案是每个 signal day 把 Top200 股票作为一组 tokens，token 内含过去 20 日路径摘要、行业/概念和市场状态，用小型 Set/Perceiver/Factorized Transformer 学习同日横截面相对关系，再输出 Top15/Top20 排序。这个方向更贴近用户强调的“特别重视横截面因子，不单单看一天，还要看过去一段时间”。

## 14. cross_sectional_world_model_v1

### 14.1 假设

第 13 节说明单股票时间轴 Transformer 不是主要瓶颈。真正可能缺失的是同一 signal day 内 Top200 股票之间的相对位置、扩散、拥挤度和行业/概念共振。因此本轮把每个 signal day 视作一个横截面样本：Top200 股票是 tokens，每个 token 由过去 20 日路径摘要构成，模型在同日股票集合上做小型 Transformer attention，再输出每只股票的多尺度分位、右尾、左尾和稳健路径任务。

核心反事实：如果横截面关系确实是缺失信息，那么开发期不应只在 Top10 变好，而应在 Top15/Top20 这类可执行宽度上同时增厚；2026 前向也应至少保持边际增量。若纯模型排序失败但与 base 融合有效，说明它还不是主排序器，而是提供了 base 之外的弱校正信号。

### 14.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_world_model_v1.py
sha256:f67773c4e626dd81d09a59d38c477aebe51651ecfe0fbf811bc7b1094123374d

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_world_smoke_2021_2023q1.json
sha256:d3b5535bf61abd5f059d4f168ecb32e95acc95480191c4ba60a8c7ecb0b957f9

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_world_dev_2021_2025.json
sha256:d555343fe24e7fc9babc99c273aaa68acdd2379c0f238cad43dab9fa0a3976a7

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_world_val63_2026.json
sha256:67ee60e43be82ad1ead016775e054f502f0a5ec512dc3789aa7aeb5c1c530df5
```

口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | 每个 signal day 取 base Top200 |
| lookback | 20 日 |
| token 特征 | `last / mean / std / last-first` 的路径摘要 |
| 特征族 | base rank/score、RPS、量价路径、概念/行业、相对概念/行业、市场状态 |
| encoder | 2 层小型 `TransformerEncoder`，hidden 64 |
| 预训练 | 随机 mask 股票 token，重建 token 摘要特征 |
| 监督任务 | 5/10/20 日分位、右尾、左尾、路径稳健 |
| 评估 | `base`、纯模型头 `cs_q5/cs_q20/cs_world`、25% 融合头 `blend_q5_25/blend_world_25` |

### 14.3 结果

smoke 先验证闭环：2022-2023Q1 小样本中 `cs_q20::top15` mean label5 为 `+0.023036`，相对 base `+0.002696`；`cs_q20::top20` 为 `+0.019254`，相对 base `+0.002424`。因此进入完整开发期。

开发期 2022-2025 主要结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.010509` | `0` | `+0.022211` | `+0.032180` | `59.47%` | 基线 |
| `cs_world` | 15 | `+0.013736` | `+0.003227` | `+0.023016` | `+0.032992` | `65.79%` | 开发期强 |
| `cs_q20` | 15 | `+0.013334` | `+0.002825` | `+0.023606` | `+0.033528` | `63.68%` | 开发期强 |
| `blend_q5_25` | 15 | `+0.011489` | `+0.000980` | `+0.022417` | `+0.029850` | `63.16%` | 弱正 |
| base | 20 | `+0.009543` | `0` | `+0.019722` | `+0.027603` | `63.68%` | 基线 |
| `cs_world` | 20 | `+0.013098` | `+0.003556` | `+0.021123` | `+0.029483` | `68.95%` | 开发期最强 |
| `cs_q20` | 20 | `+0.012027` | `+0.002484` | `+0.020476` | `+0.029471` | `67.37%` | 开发期强 |
| `blend_q5_25` | 20 | `+0.010267` | `+0.000725` | `+0.021107` | `+0.026851` | `63.68%` | 弱正 |

2026 val63 前向结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.006436` | `0` | `+0.008859` | `+0.002212` | `66.67%` | 基线 |
| `cs_world` | 15 | `+0.006777` | `+0.000341` | `+0.009359` | `+0.007695` | `57.14%` | 纯模型不稳 |
| `blend_q5_25` | 15 | `+0.011099` | `+0.004663` | `+0.012342` | `+0.019997` | `71.43%` | 前向显著增厚 |
| `blend_world_25` | 15 | `+0.008823` | `+0.002387` | `+0.013138` | `+0.016410` | `76.19%` | 前向正增量 |
| base | 20 | `+0.008575` | `0` | `+0.015123` | `+0.008444` | `76.19%` | 基线 |
| `cs_world` | 20 | `+0.006805` | `-0.001770` | `+0.008344` | `+0.009802` | `61.90%` | 纯模型失败 |
| `blend_q5_25` | 20 | `+0.011831` | `+0.003256` | `+0.018029` | `+0.029584` | `76.19%` | 前向显著增厚 |
| `blend_world_25` | 20 | `+0.009608` | `+0.001034` | `+0.014332` | `+0.017629` | `76.19%` | 前向正增量 |

宽度检查：开发期纯模型在 Top15/Top20 最强，但 Top30 只剩小增量，说明模型主要改善可执行中等宽度，而不是全池平移。2026 前向则相反：纯模型 Top10/Top20/Top30 多数弱于 base，只有 `blend_q5_25` 在 Top15/Top20 同时显著增厚，Top30 又转负。这说明模型提供的是“base 高分候选中的局部再排序校正”，不是可独立替代 base 的横截面主引擎。

### 14.4 反事实分析

第一反事实：如果横截面 world model 学到了稳定主排序规律，纯 `cs_world` 在 2026 Top20 不应低于 base。实际 `cs_world::top20` 前向 label5 为 `+0.006805`，低于 base `+0.008575`，正 session 占比也从 base `76.19%` 降到 `61.90%`。因此纯模型还不具备独立排序能力。

第二反事实：如果 2026 融合头只是偶然，开发期融合头应没有任何支撑。实际 `blend_q5_25::top15/top20` 开发期分别为 `+0.000980` 和 `+0.000725`，虽然不厚，但不是负；前向则分别扩大到 `+0.004663` 和 `+0.003256`。这更像 regime 下边际校正在 2026 被放大，而不是完全无根亮点。

第三反事实：如果模型只是学到左尾/防守过滤，label20 或路径上行不应同步改善。实际 2026 `blend_q5_25::top20` 的 label20 达到 `+0.029584`，显著高于 base `+0.008444`，mean path max20 也从 base `+0.116167` 提高到 `+0.127671`，不是单纯减少亏损尾部。

第四反事实：如果方向足够接近用户要求的收益厚度，它应自然逼近 path sequence 强基线。实际 2026 `blend_q5_25::top20` label5 `+0.011831` 低于 path sequence 2026 Top20 `+0.012520`，Top15 `+0.011099` 也低于 path sequence Top15 `+0.018316`。所以本轮是重要增量线索，但仍不足以进入 formal account。

第五反事实：如果问题只是模型容量不足，纯模型开发期强、前向弱可以通过堆层解决。但前向有效的是 25% 融合而非纯模型，说明更可能的问题是横截面模型校准和标签目标不稳定，而不是 attention 层数太少。继续扩大模型可能放大过拟合。

### 14.5 判定

`promising_but_not_integrated_as_primary_strategy`。

这是第 05 轮目前最有价值的 DL 方向之一：它首次证明“同日横截面股票 token + 过去路径摘要”的 world model 可以在开发期显著提高 Top15/Top20 预测层收益，并在 2026 通过 25% 融合产生更厚的前向多尺度收益。这个结果支持用户强调的判断：短周期 alpha 不能只看单股票时间轴，必须把过去一段时间的横截面相对关系纳入模型。

但它仍不能融入策略：纯模型 2026 不稳，融合权重是固定 25% 的研究口径，且预测层收益还没有转化为 formal backtest 的五年几十倍收益、平均持仓大于 5、五年自然年全正。它只能作为下一轮研究方向，而不是成品策略。

下一步不继续堆大模型，而是做两个更贴近问题本质的迭代：

1. `cross_sectional_residual_teacher_v2`：让横截面模型只学习 base Top200 内的 residual/teacher delta，而不是直接学习绝对分位，解决纯模型前向校准不稳。
2. `cross_sectional_state_gated_blend_v1`：只在市场扩散、概念共振、base 分数拥挤度适合时启用横截面校正，检验 2026 融合增量是否来自特定市场状态，而不是固定权重巧合。

两条线都必须继续保持 `/tmp` 实验、开发期/2026 前向分离、多尺度 TopK gate，只有预测层和 formal account 同时过硬才允许进入正式策略。

## 15. cross_sectional_residual_teacher_v2

### 15.1 假设

第 14 节的核心矛盾是：横截面 world model 开发期纯模型很强，但 2026 纯模型不稳；反而 `blend_q5_25` 在前向 Top15/Top20 同时增厚。一个自然解释是模型不适合替代 base 排序，只适合学习 base Top200 内的 residual 校正。因此本轮把监督目标从“绝对未来分位”扩展为“未来收益横截面 rank 减 base rank”的 residual 回归，检验显式 residual teacher 是否能保住第 14 节的融合增量，并改善纯模型校准不稳。

如果这个假设正确，`res5/res_world` 融合应该在开发期和 2026 的 Top15/Top20 同时超过普通 `blend_q5_25`，并且不需要靠更窄 TopK 才成立。

### 15.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_residual_teacher_v2.py
sha256:f169c7a498455b09066575370a9a0ef565716f8123d679f733deb7163431c4d6

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_residual_teacher_v2_smoke_2021_2023q1.json
sha256:d142b80bfae024edbc388c334a715b56047de06d4000c79ff86c9c289f40f90e

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_residual_teacher_v2_dev_2021_2025.json
sha256:af0ac7eb2c5cfc1dd4d51f72c2878b7ff229567b261c9697c02db90989715999

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_residual_teacher_v2_val63_2026.json
sha256:2eaefc9d758033dbf59cf82836edf3c294e90957a56dc60b30166eb24c696fba
```

相对第 14 节，模型主体保持一致：Top200 stock-token Transformer、20 日路径摘要、masked token pretraining、多任务分位/尾部/路径稳健任务。新增三个 residual 回归头：

```text
res5  = rank_pct(label5)  - base_ml_rank_pct
res10 = rank_pct(label10) - base_ml_rank_pct
res20 = rank_pct(label20) - base_ml_rank_pct
```

评估分数包括 `base + w * res5` 和 `base + w * res_world`，其中 `w` 为 `0.10/0.20/0.25/0.35`。保留 `blend_q5_25` 作为普通 q5 概率融合参照。

### 15.3 结果

smoke 结果先显示方向不算完全断裂：2022-2023Q1 小样本中 `res_world_20::top15` label5 为 `+0.021278`，相对 base `+0.000938`；但 Top20 多数不如 base，因此只进入完整开发期验证，不作为正证据。

开发期 2022-2025 主要结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.010509` | `0` | `+0.022211` | `+0.032180` | `59.47%` | 基线 |
| `cs_world` | 15 | `+0.013513` | `+0.003003` | `+0.023451` | `+0.032778` | `64.74%` | 仍是模型内最强 |
| `blend_q5_25` | 15 | `+0.011282` | `+0.000773` | `+0.022120` | `+0.030599` | `61.58%` | 弱正 |
| `res5_35` | 15 | `+0.011274` | `+0.000765` | `+0.022461` | `+0.031814` | `61.05%` | 不优于普通融合 |
| `res_world_35` | 15 | `+0.011094` | `+0.000585` | `+0.021776` | `+0.031422` | `61.05%` | 薄增量 |
| base | 20 | `+0.009543` | `0` | `+0.019722` | `+0.027603` | `63.68%` | 基线 |
| `cs_world` | 20 | `+0.012209` | `+0.002666` | `+0.021663` | `+0.029040` | `66.32%` | 仍强于 residual |
| `blend_q5_25` | 20 | `+0.010342` | `+0.000799` | `+0.020184` | `+0.026724` | `62.63%` | 弱正 |
| `res_world_25` | 20 | `+0.009881` | `+0.000339` | `+0.019780` | `+0.027402` | `62.63%` | 薄增量 |
| `res_world_35` | 20 | `+0.009689` | `+0.000146` | `+0.020041` | `+0.027754` | `62.63%` | 薄增量 |

2026 val63 前向结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.006436` | `0` | `+0.008859` | `+0.002212` | `66.67%` | 基线 |
| `blend_q5_25` | 15 | `+0.010708` | `+0.004272` | `+0.012625` | `+0.019553` | `76.19%` | 仍然最强 |
| `res5_35` | 15 | `+0.008817` | `+0.002380` | `+0.012886` | `+0.006273` | `76.19%` | label20 不厚 |
| `res_world_35` | 15 | `+0.008721` | `+0.002285` | `+0.012679` | `+0.006500` | `71.43%` | 不及普通融合 |
| `cs_world` | 15 | `+0.007191` | `+0.000755` | `+0.008044` | `+0.007028` | `61.90%` | 纯模型仍不稳 |
| base | 20 | `+0.008575` | `0` | `+0.015123` | `+0.008444` | `76.19%` | 基线 |
| `blend_q5_25` | 20 | `+0.009390` | `+0.000815` | `+0.013523` | `+0.020711` | `76.19%` | 仅小幅正增量 |
| `res_world_35` | 20 | `+0.008840` | `+0.000265` | `+0.015055` | `+0.010126` | `76.19%` | 薄增量 |
| `res5_35` | 20 | `+0.008726` | `+0.000151` | `+0.014565` | `+0.008996` | `76.19%` | 近似持平 |
| `cs_world` | 20 | `+0.005582` | `-0.002992` | `+0.006896` | `+0.008099` | `61.90%` | 失败 |

与第 14 节相比，`cross_sectional_residual_teacher_v2` 没有复现更强的 2026 Top20 增量。第 14 节 `blend_q5_25::top20` 前向 label5 为 `+0.011831`、label20 为 `+0.029584`；本轮降为 label5 `+0.009390`、label20 `+0.020711`。显式 residual 目标不但没有解释融合亮点，反而削弱了右尾厚度。

### 15.4 反事实分析

第一反事实：如果第 14 节的问题只是“模型应该学 residual 而非绝对分位”，那么 residual 融合应在开发期超过普通 q5 融合。实际开发期 `res_world_25::top20` 只增 `+0.000339`，低于 `blend_q5_25::top20` 的 `+0.000799`；Top15 也不优于普通融合。

第二反事实：如果 residual 目标解决了前向校准，2026 Top20 应明显优于 base。实际最好的 residual 类 Top20 只有 `res_world_35`，相对 base `+0.000265`，远低于第 14 节普通融合 `+0.003256` 的前向增量。

第三反事实：如果 residual 目标保留了右尾厚度，label20 应同步提升。实际 2026 `res5_35::top15` label5 有 `+0.002380` 增量，但 label20 只有 `+0.006273`，明显低于 `blend_q5_25::top15` 的 `+0.019553`。这说明 residual 回归更像短期 rank 微调，没有抓住真正厚右尾。

第四反事实：如果这是模型容量问题，新增 residual heads 至少不应破坏纯 `cs_world`。实际纯 `cs_world::top20` 2026 进一步降到 `+0.005582`，相对 base `-0.002992`。多头监督没有稳定模型，反而让主排序更弱。

### 15.5 判定

`rejected_as_residual_objective_not_root_cause`。

显式 residual teacher 没有解决第 14 节暴露的核心问题。它证明：横截面 world model 的有效部分并不只是“预测 base residual rank delta”；第 14 节前向融合增量更可能来自某些市场状态下 q5 概率与 base 分数的条件互补，而不是一个全年稳定的 residual 函数。

本轮不进入 PredictionStore，不进入 formal account，不继续调 residual loss 权重、融合权重或 epoch。下一步应转向第 14 节提出的第二条线：`cross_sectional_state_gated_blend_v1`。重点不是让模型全年校正 base，而是识别什么时候横截面校正有效：市场扩散、概念/行业共振、base 分数拥挤度、候选池内部右尾集中度，以及模型自身置信度是否共同指向“可启用校正”。若状态门控仍不能让开发期和 2026 同时增厚，则横截面 world model 应降级为观察线索，研究重心转向新候选土壤或更早期事件传播数据源。

## 16. cross_sectional_state_gated_blend_v1

### 16.1 假设

第 14 节显示固定 `blend_q5_25` 在 2026 前向显著增厚，但开发期只是弱正；第 15 节又否定了“显式 residual 目标就是根因”。因此本轮检验另一个解释：横截面 world model 的校正只在特定市场状态下有效，例如主题共振强、市场扩散较好、候选池拥挤、或波动分化较高。

为了避免未来函数，本轮不使用测试期标签或测试期分布调阈值。每个 fold 只用训练 session 的当日可见状态特征计算阈值，再把这些阈值应用到测试 session。gate 开启时使用 25% 融合，gate 关闭时回到 base 排序。

### 16.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_state_gated_blend_v1.py
sha256:3e91ad8dbf6138e48e5c96c49e5ef8f281aec63b1f9b49aa0b091c8118978fdd

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_state_gated_blend_v1_smoke_2021_2023q1.json
sha256:a92a72a34bdf8f86349d3747093eaecc55643774c927cd76ea2b584b183188ac

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_state_gated_blend_v1_dev_2021_2025.json
sha256:96035202a0b4d545a6e3a36acbbb8f566312ad98a9eb4927704cb8a4b98bcfcc

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_state_gated_blend_v1_val63_2026.json
sha256:58ba466e58669e276ff6b07b89582d3723406c7b7a84b002907109db5484bd12
```

状态指标使用训练标准化后的当日可见 feature 分布：

| 状态 | 构造 |
| --- | --- |
| `theme_strength` | 概念 5 日、近涨停、强势比例、相对概念强度与行业强度的组合 |
| `market_breadth20` | 市场 20 日宽度 |
| `market_disp` | 市场 20 日收益分化 |
| `base_score_z_std` | Top200 内 base score z 的横截面标准差 |

gate variants：

| gate | 含义 |
| --- | --- |
| `gate_theme_*` | 训练期 `theme_strength` 60% 分位以上 |
| `gate_theme_breadth_*` | 主题强且宽度不低于训练期 40% 分位 |
| `gate_crowded_theme_*` | 主题更强且 base score 更拥挤 |
| `gate_disp_theme_*` | 主题强且市场分化高 |

### 16.3 结果

smoke 先验证实现闭环：gate variants 正常生成，但 `base_score_z_std` 在训练标准化后几乎为常数，说明拥挤度定义的信息量不足。完整开发期仍保留该 gate，以观察它是否在年度数据中有区分度。

开发期 2022-2025 主要结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | gate 启用率 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.010509` | `0` | `+0.022211` | `+0.032180` | `59.47%` | `0%` | 基线 |
| `blend_q5_25` | 15 | `+0.011489` | `+0.000980` | `+0.022417` | `+0.029850` | `63.16%` | `0%` | 固定融合弱正 |
| `blend_world_25` | 15 | `+0.011207` | `+0.000698` | `+0.021489` | `+0.030115` | `60.00%` | `0%` | 固定融合弱正 |
| `gate_theme_world_25` | 15 | `+0.011733` | `+0.001224` | `+0.022859` | `+0.032154` | `61.05%` | `50.00%` | 开发期最稳 gate |
| `gate_theme_q5_25` | 15 | `+0.011498` | `+0.000989` | `+0.023011` | `+0.031465` | `61.05%` | `50.00%` | 接近固定 q5 |
| base | 20 | `+0.009543` | `0` | `+0.019722` | `+0.027603` | `63.68%` | `0%` | 基线 |
| `blend_q5_25` | 20 | `+0.010267` | `+0.000725` | `+0.021107` | `+0.026851` | `63.68%` | `0%` | 固定融合弱正 |
| `blend_world_25` | 20 | `+0.010564` | `+0.001022` | `+0.020502` | `+0.027804` | `63.68%` | `0%` | 固定 world 较好 |
| `gate_theme_world_25` | 20 | `+0.010791` | `+0.001248` | `+0.020493` | `+0.028271` | `65.79%` | `50.00%` | 开发期最佳 gate |
| `gate_theme_q5_25` | 20 | `+0.010670` | `+0.001127` | `+0.020506` | `+0.026838` | `66.84%` | `50.00%` | 开发期正 |

2026 val63 前向结果：

| 口径 | TopK | label5 | 相对 base | label10 | label20 | 正 session 占比 | gate 启用率 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 15 | `+0.006436` | `0` | `+0.008859` | `+0.002212` | `66.67%` | `0%` | 基线 |
| `blend_q5_25` | 15 | `+0.011099` | `+0.004663` | `+0.012342` | `+0.019997` | `71.43%` | `0%` | 固定融合最强 |
| `blend_world_25` | 15 | `+0.008823` | `+0.002387` | `+0.013138` | `+0.016410` | `76.19%` | `0%` | 固定融合正 |
| `gate_theme_world_25` | 15 | `+0.007113` | `+0.000677` | `+0.010271` | `+0.005848` | `66.67%` | `23.81%` | 远低于固定融合 |
| `gate_theme_q5_25` | 15 | `+0.006480` | `+0.000044` | `+0.007892` | `+0.003718` | `66.67%` | `23.81%` | 近似 base |
| base | 20 | `+0.008575` | `0` | `+0.015123` | `+0.008444` | `76.19%` | `0%` | 基线 |
| `blend_q5_25` | 20 | `+0.011831` | `+0.003256` | `+0.018029` | `+0.029584` | `76.19%` | `0%` | 固定融合最强 |
| `blend_world_25` | 20 | `+0.009608` | `+0.001034` | `+0.014332` | `+0.017629` | `76.19%` | `0%` | 固定融合正 |
| `gate_theme_world_25` | 20 | `+0.008683` | `+0.000108` | `+0.015112` | `+0.010351` | `76.19%` | `23.81%` | 几乎无增量 |
| `gate_theme_q5_25` | 20 | `+0.008960` | `+0.000385` | `+0.015388` | `+0.012066` | `76.19%` | `23.81%` | 远低于固定融合 |
| `gate_disp_theme_q5_25` | 20 | `+0.009043` | `+0.000468` | `+0.015198` | `+0.010364` | `76.19%` | `19.05%` | 仍然很薄 |

### 16.4 反事实分析

第一反事实：如果第 14 节的 2026 固定融合增量来自主题状态，`gate_theme_*` 应在 2026 保留大部分增量。实际 `gate_theme_q5_25::top20` 只有 `+0.000385`，远低于固定 `blend_q5_25::top20` 的 `+0.003256`。状态门控没有解释前向亮点。

第二反事实：如果 gate 是有效筛选器，开发期最佳 gate 应在 2026 至少不劣于 base。实际虽然没有明显低于 base，但增量几乎被削平，Top15 `gate_theme_q5_25` 只剩 `+0.000044`。它更像是把有效校正 session 错误关闭了。

第三反事实：如果横截面校正只在高主题强度下有效，2026 gate 启用率下降到 `23.81%` 后，被启用的 session 应贡献更厚 label20。实际 `gate_theme_world_25::top20` label20 只有 `+0.010351`，远低于固定 `blend_q5_25` 的 `+0.029584`。

第四反事实：如果拥挤度 gate 能识别 base 排序失效，`gate_crowded_theme_*` 应显著优于主题 gate。实际 smoke 和 dev 中 `base_score_z_std` 几乎常数，前向也没有胜出。这个拥挤度定义本身失败。

第五反事实：如果固定融合只是过度启用导致开发期弱，gate 应在开发期和前向同时改善固定融合。实际开发期 `gate_theme_world_25` 确实略优于固定 world 融合，但前向大幅落后固定 q5 融合，说明开发期 gate 学到的是历史 regime 的状态切片，不是稳定泛化规律。

### 16.5 判定

`rejected_as_state_gate_not_explaining_forward_alpha`。

状态门控没有解释第 14 节的前向增量。它给了一个有价值的负证据：2026 里固定 `blend_q5_25` 的强表现不是简单由“高主题强度/宽度/分化”这些粗状态触发；当前状态变量反而关闭了很多有效 session。继续调 gate 阈值、增加几个手写状态组合，很容易变成开发期筛选，不符合用户“不硬卡要求”的约束。

横截面 world model 线索仍有价值，但已经连续三轮暴露同一问题：它能产生边际 alpha，却无法稳定成为主收益引擎，也无法通过 residual 或粗状态门控解释第 14 节的前向亮点。下一步不继续围绕同一 Top200 base 池做门控微调，而应转向更可能提高收益厚度的方向：寻找新的候选土壤或更早的事件传播信号。

下一轮方向：`early_event_propagation_world_model_v1`。核心假设是，当前 base Top200 已经太接近成熟强势股池，DL 只能做薄校正；真正有几十倍收益潜力的短周期 alpha 可能来自概念/行业扩散早期、涨停/近涨停前后的二阶传播、以及“第一批强股之后的可交易承接”。验证方式应先做无训练的数据诊断：构建 signal day 前可见的事件传播候选池，检查 2022-2025 和 2026 是否存在同时为正、平均持仓宽度足够、且不只是复制 base/path 排序的肥土壤。若土壤存在，再训练小型 cross-sectional temporal model；若土壤不存在，继续扩大 DL 模型没有意义。

## 17. early_event_propagation_soil_v1

### 17.1 假设

第 14-16 节说明，横截面 world model 在 base Top200 内能提供边际 alpha，但无法自然成为主收益引擎。一个更高层的反事实是：模型没有错，土壤太晚、太成熟。A 股短线的厚右尾可能出现在概念/行业扩散早期、第一批强股之后的二阶传播、以及“组热但个股未过度追涨”的可交易承接里。

第 04 轮已经否定过多种事件方向，包括 `limitup_concept`、`group_event_basket` 和 `right_tail_event` formal replay。本轮不是重跑旧公式，而是把候选池从 base Top200 中移出，直接在全主板上做无训练土壤诊断：用 T 日收盘前可见的概念/行业事件、个股路径、不过热约束和温和放量构造早期传播候选，再检查 Top15/Top20 的 5/10/20 日超额收益、正 session 占比、候选宽度，以及与 base/path 强基线的重合度。

如果这个方向值得训练 DL，至少应看到一个开发期和 2026 同时为正、Top20 厚度接近 base/path 强基线、且与 base/path 重合度较低的候选土壤。

### 17.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_early_event_propagation_soil_v1.py
sha256:0c9cef99bd9c83ec1d2b581ff6b0e5cd5b4607f47744354d66283fedcd03f0d4

.tmp/quantx-research/deep-learning-alpha-search-v1/early_event_propagation_soil_v1_dev_2021_2025.json
sha256:80cec7e5b1524e429cb4afeb3a2860a85a27b12c94c1be68c72bb32357cf6c26

.tmp/quantx-research/deep-learning-alpha-search-v1/early_event_propagation_soil_v1_val63_2026.json
sha256:c3d4d9c2f1eefb4e235acc3ddfc284032a9fa3992783cb348f788cf983ff495e
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选域 | 全主板，不限 base Top200 |
| 信号 | T 日收盘前可见特征 |
| 入场/退出 | T+1 open 入场；5/10/20 日 open 退出标签 |
| 标签 | 个股 open-to-open 收益减同日全市场均值 |
| 样本 | `all` 和 `due5` 双口径 |
| TopK | 10/15/20/30 |
| 重合度 | 与 base Top200、path Top20 预测池比较 |

变体：

| 变体 | 含义 |
| --- | --- |
| `theme_early_follower` | 主题强、个股 5 日不过热、20 日趋势不弱 |
| `theme_second_wave_pullback` | 主题强、趋势仍在、接近高点但有回落承接 |
| `leader_spillover_calm` | 组层涨停/近涨停扩散强，但个股自身不过热 |
| `theme_lagging_catchup` | 主题强、个股相对组内落后、等待补涨 |
| `early_acceleration_prebreak` | 主题不弱、20 日相对强度加速、量能斜率改善 |
| `broad_theme_quality` | 主题强、中期质量和稳定路径较好 |

### 17.3 结果

开发期 2021-2025 的 Top20 重点结果：

| 口径 | label5 | label10 | label20 | 正 session 占比 | base 重合 | path 重合 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `due5::early_acceleration_prebreak` | `+0.003503` | `+0.004474` | `+0.004089` | `50.21%` | `2.28%` | `0.46%` | 开发期最强，但厚度不足 |
| `due5::theme_lagging_catchup` | `-0.000114` | `+0.001473` | `-0.001398` | `47.09%` | `6.59%` | `1.59%` | 开发期不成立 |
| `all::broad_theme_quality` | `-0.000749` | `-0.001320` | `-0.005752` | `47.71%` | `8.41%` | `1.29%` | 开发期负 |
| `all::theme_second_wave_pullback` | `-0.003162` | `-0.005361` | `-0.010337` | `43.37%` | `4.97%` | `1.28%` | 开发期明显负 |

`due5::early_acceleration_prebreak::top20` 年度拆分：2021 `+0.009010`、2022 `+0.002092`、2023 `+0.000800`、2024 `+0.004122`、2025 `+0.001296`。五年均为正，但绝对厚度只有 `+0.350%`，显著低于 path sequence 2026 Top20 `+0.012520` 和开发期强基线量级。

2026 val63 Top20 重点结果：

| 口径 | label5 | label10 | label20 | 正 session 占比 | base 重合 | path 重合 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `due5::early_acceleration_prebreak` | `-0.004077` | `-0.006790` | `-0.005103` | `28.57%` | `3.10%` | `0.95%` | 开发期强项前向反向 |
| `due5::theme_lagging_catchup` | `+0.013042` | `+0.021962` | `+0.053324` | `70.00%` | `2.50%` | `0.25%` | 2026 很强，但开发期不成立 |
| `all::broad_theme_quality` | `+0.013764` | `+0.024354` | `+0.051749` | `64.71%` | `5.39%` | `1.18%` | 2026 很强，但开发期负 |
| `all::theme_second_wave_pullback` | `+0.011835` | `+0.035921` | `+0.072367` | `63.10%` | `1.73%` | `0.54%` | 2026 很强，但开发期明显负 |

最有信息量的不是某个 2026 数字，而是两端反向：开发期唯一较稳的 `early_acceleration_prebreak` 在 2026 变成 `-0.4077%`；2026 最亮的 `theme_lagging_catchup / broad_theme_quality / theme_second_wave_pullback` 在开发期整体为负或接近零。

### 17.4 反事实分析

第一反事实：如果“更早期事件传播”是可穿越主土壤，开发期和 2026 应出现同一个变体同时强。实际开发期强项是 `early_acceleration_prebreak`，2026 强项是 `theme_lagging_catchup / broad_theme_quality / second_wave_pullback`，方向发生翻转。

第二反事实：如果该土壤足以支撑几十倍目标，开发期 Top20 label5 应接近或超过 path/base 强基线。实际开发期最强 Top20 只有 `+0.003503`，不足强基线的一半，更谈不上账户层收益跃迁。

第三反事实：如果低重合度代表独立 alpha，低重合度候选至少应有足够收益厚度。实际开发期确实低重合：`early_acceleration_prebreak` 与 base Top200 重合仅 `2.28%`，与 path Top20 重合仅 `0.46%`，但收益太薄。低相关本身不能替代收益厚度。

第四反事实：如果 2026 的 `broad_theme_quality` 是可训练规律，它在开发期至少不应整体为负。实际开发期 `all::broad_theme_quality::top20` label5 为 `-0.000749`、label20 为 `-0.005752`，说明 2026 更像风格切换，而不是可直接训练的稳定规律。

第五反事实：如果 DL 能通过复杂交互救回该土壤，裸规则至少应给出一个肥的训练目标。实际所有开发期候选都明显低于强基线，且正 session 占比约 `50%` 或更低。此时训练 DL 更可能拟合风格噪声，而不是放大稳定 alpha。

### 17.5 判定

`rejected_as_low_overlap_but_not_fat_soil`。

早期事件传播方向证明了一件有价值的事：从全主板构造的事件传播候选确实与 base/path 高度低重合，说明它是不同土壤；但它没有给出足够厚、可穿越的训练目标。开发期最好的候选太薄，2026 最亮的候选开发期不成立。这不满足用户要求的“鲁棒性强，容易出收益”，也不满足进入 DL 训练的前置条件。

本轮不进入模型训练，不进入 PredictionStore，不进入 formal account。下一步不继续手工调事件传播公式。更合理的方向是寻找**真正改变标签定义或执行对象**的路径，而不是继续在股票 TopK 土壤里换特征：

1. `formal_tradeable_right_tail_lifecycle_v1`：从 formal 成交失败、涨停/跳变/停牌损耗出发，学习可交易右尾生命周期，而不是 open-to-open 标签。
2. `multi_horizon_exit_world_model_v1`：把固定 5 日持有改成可见路径下的退出/延持决策，检验是否收益上限被退出策略截断。
3. 若继续候选土壤，则必须引入新的 point-in-time 数据源或更接近资金流/盘口的代理；仅靠日频 OHLCV + 静态概念/行业快照，已经多轮显示厚度不足。

## 18. exit_lifecycle_soil_v1

### 18.1 假设

第 17 节否定了继续在股票候选土壤里手写事件传播公式。另一个更可能改变收益上限的方向是退出生命周期：当前很多实验默认一周左右固定持有，但如果强势股的右尾生命周期更长，固定 5 日退出可能截断趋势；如果弱路径应更早退出，固定 5 日又会扩大损耗。

本轮先不训练模型，只做反事实诊断：对同一批 base/path 候选，比较固定 3/5/10/15/20 日 open-to-open 退出、事后最佳 horizon 的 oracle 上限，以及一个只用 T 日可见路径/主题状态的简单生命周期规则。目的不是把 oracle 当策略，而是判断“退出选择”是否有足够收益空间值得训练小模型。

如果固定 10/15/20 日在开发期和 2026 都明显优于固定 5 日，说明收益上限可能被退出周期截断；如果 oracle 上限很高但可见规则失败，则下一步应训练可验证的退出模型，而不是继续手写规则。

### 18.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_exit_lifecycle_soil_v1.py
sha256:c6d1380229ea70a0124628dfcf695dc5291cbb767e5072fc0efdd9d04293f322

.tmp/quantx-research/deep-learning-alpha-search-v1/exit_lifecycle_soil_v1_dev_2021_2025.json
sha256:46bd966eacc55b0ae8497c82a0ebe1a7c1a3eab900fabde3bbdc9d1b46c37db4

.tmp/quantx-research/deep-learning-alpha-search-v1/exit_lifecycle_soil_v1_val63_2026.json
sha256:b5b631e82d21a05a7a4de8be08ca908e639dec82a88830594ac6858fe0f0d7c8
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选源 | `base` Top20、`path_sequence` Top20 |
| 样本 | due5 |
| 入场 | T+1 open |
| 退出 | 固定 3/5/10/15/20 日 open、oracle best horizon、T 可见手写 lifecycle rule |
| 标签 | 对应 horizon 的个股收益减同日全市场均值 |
| TopK | 10/15/20，本节重点 Top20 |

`visible_lifecycle_rule` 只使用 T 日可见状态：20/60 日路径质量、主题强度、过热/衰竭度，决定 3/5/10/15/20 日退出。它不是模型，只是一个低成本可证伪规则。

### 18.3 结果

开发期 2021-2025 Top20 结果：

| 候选源 | 退出 | label | raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path | fixed 5 | `+0.011011` | `+0.013462` | `62.11%` | 5.00 | 当前一周基准 |
| path | fixed 10 | `+0.021976` | `+0.026607` | `67.89%` | 10.00 | 明显增厚 |
| path | fixed 15 | `+0.025080` | `+0.032044` | `67.37%` | 15.00 | 明显增厚 |
| path | fixed 20 | `+0.027956` | `+0.037685` | `64.74%` | 20.00 | 开发期最强固定退出 |
| path | visible rule | `+0.008467` | `+0.009336` | `64.74%` | 4.83 | 手写规则失败 |
| path | oracle best 3-20 | `+0.095515` | `+0.102938` | `99.47%` | 10.78 | 上限极高，不可作策略 |
| base | fixed 5 | `+0.013008` | `+0.016062` | `67.36%` | 5.00 | 当前一周基准 |
| base | fixed 10 | `+0.024561` | `+0.030586` | `75.73%` | 10.00 | 明显增厚 |
| base | fixed 15 | `+0.029016` | `+0.037909` | `72.38%` | 15.00 | 明显增厚 |
| base | fixed 20 | `+0.033931` | `+0.045815` | `69.46%` | 20.00 | 开发期最强固定退出 |
| base | visible rule | `+0.009922` | `+0.011344` | `66.95%` | 4.31 | 手写规则失败 |
| base | oracle best 3-20 | `+0.094000` | `+0.102039` | `100.00%` | 10.93 | 上限极高，不可作策略 |

开发期年度拆分显示，长 horizon 增厚不是单年幻觉。以 path Top20 为例，fixed 20 在 2022/2023/2024/2025 分别为 `+0.026962`、`+0.019454`、`+0.043888`、`+0.021280`，均高于 fixed 5 的对应年份 `+0.012835`、`+0.008028`、`+0.016145`、`+0.006838`。base Top20 fixed 20 在 2021-2025 五个自然年也均为正。

2026 val63 Top20 结果：

| 候选源 | 退出 | label | raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path | fixed 5 | `+0.013068` | `+0.012852` | `61.90%` | 5.00 | 一周基准 |
| path | fixed 10 | `+0.019270` | `+0.015678` | `66.67%` | 10.00 | 2026 最强固定退出 |
| path | fixed 15 | `+0.017339` | `+0.008793` | `66.67%` | 15.00 | 仍强于 5 日 |
| path | fixed 20 | `+0.011836` | `-0.002548` | `71.43%` | 20.00 | 2026 不宜太长 |
| path | visible rule | `+0.010124` | `+0.005446` | `66.67%` | 4.18 | 手写规则失败 |
| path | oracle best 3-20 | `+0.103625` | `+0.097008` | `100.00%` | 10.43 | 上限极高 |
| base | fixed 5 | `+0.008575` | `+0.008359` | `76.19%` | 5.00 | 一周基准 |
| base | fixed 10 | `+0.015123` | `+0.011531` | `80.95%` | 10.00 | 2026 最强固定退出 |
| base | fixed 15 | `+0.009355` | `+0.000808` | `71.43%` | 15.00 | 略强于 5 日 |
| base | fixed 20 | `+0.008444` | `-0.005940` | `71.43%` | 20.00 | 接近 5 日但 raw 转负 |
| base | visible rule | `+0.007958` | `+0.004100` | `71.43%` | 4.02 | 手写规则失败 |
| base | oracle best 3-20 | `+0.088194` | `+0.081075` | `100.00%` | 10.88 | 上限极高 |

最重要的观察：开发期最优固定退出偏 20 日，2026 最优固定退出偏 10 日；但两端都显示 fixed 10 明显优于 fixed 5，且 oracle 最佳 horizon 平均约 10-11 日。这说明右尾生命周期确实存在，但不是“越长越好”，而是需要预测退出周期。

### 18.4 反事实分析

第一反事实：如果固定 5 日不是瓶颈，固定 10/15/20 日不应系统性优于 5 日。实际开发期 base/path Top20 的 10/15/20 日都显著高于 5 日，2026 的 10 日也高于 5 日。因此固定 5 日确实截断了一部分右尾生命周期。

第二反事实：如果只是开发期长期趋势导致长持看起来好，2026 不应继续支持 10 日。实际 2026 path Top20 fixed 10 为 `+0.019270`，高于 fixed 5 的 `+0.013068`；base Top20 fixed 10 为 `+0.015123`，高于 fixed 5 的 `+0.008575`。

第三反事实：如果简单 T 可见规则已经足够，`visible_lifecycle_rule` 应至少接近 fixed 10。实际它开发期和 2026 都低于 fixed 5 或接近 fixed 5，说明手写路径/主题阈值没有抓住生命周期。这个方向需要学习模型，而不是继续调规则。

第四反事实：如果 oracle 上限只是少数极端样本，正 session 占比不应接近 100%。实际 oracle 在开发期和 2026 的正 session 占比接近或等于 100%，说明每个 rebalance session 内普遍存在更优退出周期。但 oracle 使用未来收益，不能作为策略，只能证明空间存在。

第五反事实：如果退出周期选择与选股无关，base/path 的 oracle 上限应差异很大或不可解释。实际 base/path 的 oracle 上限都约 `+9%` 到 `+10%`，平均最佳 horizon 都约 10-11 日，说明它是候选生命周期问题，而不是单个模型偶然。

### 18.5 判定

`promising_lifecycle_alpha_requires_learned_exit_model`。

这是第 05 轮后半段目前最强的新方向：它不是继续在同一 Top20 内挤排序小数点，而是直接改变收益实现方式。固定 5 日持有确实低估了 path/base 候选的右尾生命周期，固定 10 日在开发期和 2026 都显著优于 5 日；oracle 上限则说明理论空间非常大。

但本轮仍不能融入策略：oracle 有未来函数，fixed 20 在 2026 raw 转负，手写可见规则失败，说明生命周期不能靠简单阈值硬卡。下一步应该训练一个小型、本地可跑、严格 walk-forward 的 `multi_horizon_exit_world_model_v1`：输入 T 日可见的个股路径、base/path 排名、组层状态和市场状态，输出 3/5/10/15/20 日收益分布、最佳 horizon 分类、以及早退风险。评估不只看 label 层，还要进入 formal account，检查平均持仓、五年自然年收益和 2026 前向。

下一轮 gate：

1. 开发期 2022-2025 中，模型选择的 horizon 必须在 Top20 上超过 fixed 5，并接近 fixed 10，不能只接近 oracle。
2. 2026 val63 中，模型不能劣于 fixed 10 太多，并且必须高于 fixed 5。
3. 若预测层通过，必须生成 `/tmp` PredictionStore 或 exit schedule，进入 formal rebalance/overlap account 裁判。
4. 若模型只学成“全部持有 10 日”，则方向降级为固定退出参数实验；若能在 10/15/20/3 日之间形成可解释切换，才继续进入正式化。

## 19. multi_horizon_exit_world_model_v1

### 19.1 假设

第 18 节证明了一个很强的事实：同一批 path/base 候选在 5 日固定退出下明显被截断，10/15/20 日退出能显著增厚，oracle best horizon 的空间约 `+10%`。但 oracle 有未来函数，手写 lifecycle rule 又失败。因此本轮不再手写退出阈值，而是训练一个小型、本地可跑的退出世界模型，专门回答一个窄问题：**在 T 日只看可见路径时，能否预测 3/5/10/15/20 日哪个退出周期更合适。**

本轮故意先用 GRU 而不是 Transformer：当前任务不是重新选股，而是验证“退出周期是否可预测”。20 日 lookback 的短序列、小样本、强噪声场景下，GRU 是低方差探针；如果 GRU 都学不到稳定生命周期，直接上更大 Transformer 更可能拟合开发期风格。如果 GRU 前向通过，再做同标签、同切分、同候选池的 tiny Temporal Transformer 对照。

### 19.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_multi_horizon_exit_world_model_v1.py
sha256:4ac21538de127817b2b260961ca4fbdc7df3dfb0b444d717346b1728c091e584

.tmp/quantx-research/deep-learning-alpha-search-v1/multi_horizon_exit_world_model_v1_dev_2021_2025.json
sha256:539151b2ef304a673f5b48ba52e76a4167cca1ef3047153b07b6bdfcd8d131a9

.tmp/quantx-research/deep-learning-alpha-search-v1/multi_horizon_exit_world_model_v1_val63_2026.json
sha256:e5b2d6cf87a1f83a749b06096495d110b125aa359201ad4472f7d767c2cca0c0
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选源 | `path_sequence` Top20 |
| 输入 | T 日可见的过去 20 日个股路径、base/path 分数、局部相对强弱和路径状态 |
| 标签 | 3/5/10/15/20 日 open-to-open 超额收益、best horizon 分类 |
| 模型 | 小型 GRU encoder，hidden=64，多任务输出：best horizon 分类 + 各 horizon 收益回归 |
| 切分 | dev 为 walk-forward 年度训练/测试；forward 为 2021-2025 训练、2026 val63 测试 |
| 对照 | fixed 3/5/10/15/20、oracle best 3-20、`model_cls`、`model_reg` |
| TopK | 10/15/20，本节重点 Top15/Top20 |

训练日志中的 best horizon 类别分布并非单峰：例如 2021-2025 累计训练样本在 `[3,5,10,15,20]` 上约为 `[3645,2375,2290,2495,3715]`，说明标签本身不是“全部 10 日”问题，确实存在周期分歧。

### 19.3 结果

开发期 2021-2025，Top15/Top20 重点结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.012436` | `+0.015761` | `61.90%` | 5.00 | 当前一周退出基准 |
| fixed 10 | 15 | `+0.022564` | `+0.028852` | `63.17%` | 10.00 | 强基准 |
| fixed 15 | 15 | `+0.028998` | `+0.038080` | `63.31%` | 15.00 | 开发期很强 |
| fixed 20 | 15 | `+0.031244` | `+0.043090` | `61.90%` | 20.00 | 开发期最强固定退出 |
| model cls | 15 | `+0.027166` | `+0.034922` | `65.72%` | 11.91 | 赢 fixed10，输 fixed15/20 |
| model reg | 15 | `+0.028701` | `+0.037279` | `65.30%` | 11.95 | 接近 fixed15，明显赢 fixed10 |
| oracle best 3-20 | 15 | `+0.102742` | `+0.111058` | `98.44%` | 10.66 | 上限，不可作策略 |
| fixed 5 | 20 | `+0.011594` | `+0.014919` | `63.46%` | 5.00 | 当前一周退出基准 |
| fixed 10 | 20 | `+0.020725` | `+0.027013` | `65.01%` | 10.00 | 强基准 |
| fixed 15 | 20 | `+0.027006` | `+0.036087` | `66.86%` | 15.00 | 开发期很强 |
| fixed 20 | 20 | `+0.028999` | `+0.040845` | `65.86%` | 20.00 | 开发期最强固定退出 |
| model cls | 20 | `+0.024610` | `+0.032365` | `66.57%` | 11.95 | 赢 fixed10，输 fixed15/20 |
| model reg | 20 | `+0.026230` | `+0.034929` | `66.86%` | 12.06 | 赢 fixed10，略输 fixed15 |
| oracle best 3-20 | 20 | `+0.097432` | `+0.105876` | `98.87%` | 10.71 | 上限，不可作策略 |

开发期年度拆分显示，`model_reg::top15` 的动态 horizon 在 2023/2024/2025 分别为 `+0.016908`、`+0.046633`、`+0.022010`，平均 horizon 分别约 10.47、15.39、9.82。它在 2024 选择更长 horizon 并接近 fixed20，但 2025 又退回约 10 日，说明模型不是简单全选 20 日。

2026 val63 前向结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.020049` | `+0.018560` | `76.70%` | 5.00 | 前向一周基准变强 |
| fixed 10 | 15 | `+0.026975` | `+0.022315` | `77.67%` | 10.00 | 2026 最强 Top15 固定退出 |
| fixed 15 | 15 | `+0.025365` | `+0.015485` | `70.87%` | 15.00 | 接近 fixed10 |
| fixed 20 | 15 | `+0.020236` | `+0.005013` | `66.99%` | 20.00 | label 接近 fixed5，raw 明显变薄 |
| model cls | 15 | `+0.013427` | `+0.002258` | `68.93%` | 12.74 | 前向失败 |
| model reg | 15 | `+0.017554` | `+0.006723` | `66.02%` | 14.75 | 低于 fixed5/10 |
| oracle best 3-20 | 15 | `+0.113026` | `+0.107416` | `99.03%` | 10.34 | 上限仍极高 |
| fixed 5 | 20 | `+0.016282` | `+0.014793` | `69.90%` | 5.00 | 前向一周基准 |
| fixed 10 | 20 | `+0.020692` | `+0.016033` | `74.76%` | 10.00 | 强固定退出 |
| fixed 15 | 20 | `+0.021410` | `+0.011530` | `71.84%` | 15.00 | 2026 Top20 最强固定退出 |
| fixed 20 | 20 | `+0.016314` | `+0.001091` | `69.90%` | 20.00 | raw 已很弱 |
| model cls | 20 | `+0.009375` | `-0.001692` | `68.93%` | 13.06 | 前向失败，raw 转负 |
| model reg | 20 | `+0.013539` | `+0.002523` | `66.02%` | 14.89 | 低于 fixed5/10/15 |
| oracle best 3-20 | 20 | `+0.106465` | `+0.100792` | `100.00%` | 10.37 | 上限仍极高 |

本轮通过了 smoke 和 dev gate，但没有通过 forward gate：2026 中 `model_reg::top15` 为 `+0.017554`，低于 fixed10 的 `+0.026975`，也低于 fixed5 的 `+0.020049`；`model_reg::top20` 为 `+0.013539`，低于 fixed10 的 `+0.020692` 和 fixed15 的 `+0.021410`。

### 19.4 反事实分析

第一反事实：如果生命周期不可预测，模型在开发期应接近随机或低于 fixed10。实际 dev 中 `model_reg` 明显赢 fixed10，并接近 fixed15，说明可见路径里有一部分生命周期信息。不能把本轮简单归为“完全没信号”。

第二反事实：如果模型学到的是稳定生命周期规律，2026 前向应至少不低于 fixed5，并接近 fixed10。实际 2026 中 `model_reg` 同时低于 fixed5 和 fixed10，说明它在开发期学到的退出映射没有穿越市场状态切换。

第三反事实：如果开发期优势只是“学成更长持有”，模型平均 horizon 应接近 20 日。实际 dev 的平均 horizon 约 12 日，2023/2025 接近 10 日、2024 拉长到 15 日；它不是简单全选 20。但 2026 平均 horizon 升到约 15 日，而固定 10/15 才是前向最优，且 fixed20 raw 很薄，说明模型对 2026 的长持风险识别不足。

第四反事实：如果问题在候选本身失效，oracle best horizon 在 2026 应明显下降。实际 2026 oracle 仍有 `+10%` 以上、正 session 接近 100%，候选生命周期空间仍在；失败点更可能是模型没有学到“市场状态条件化的退出策略”，而不是退出方向不存在。

第五反事实：如果只需要单股路径，GRU 的 T 日局部序列应足够。实际前向失败说明退出周期可能强依赖横截面状态：主题拥挤度、市场风险偏好、同组扩散阶段、近期强股尾部是否开始衰竭。单股 path encoder 缺少“这一批股票在当下市场中处于什么生命周期阶段”的参照系。

### 19.5 判定

`rejected_as_single_stock_exit_model_not_forward_robust`。

本轮最有价值的结论不是模型失败，而是失败形态：退出生命周期空间仍然很大，dev 中小 GRU 能学到一部分，但 2026 前向没有超过固定 10/15 日。也就是说，“退出周期”是对的，但“只看单股路径预测退出周期”不够。继续扩大单股 GRU 或直接换单股 Transformer，都可能只是把开发期拟合得更好，不能解决状态条件化问题。

下一轮方向应转为 `cross_sectional_state_conditioned_exit_policy_v1`：在不重排股票的前提下，把每个 rebalance session 的横截面状态加入退出模型，让模型同时看到候选池整体的强弱、拥挤、分化、主题扩散和市场风险偏好。模型仍要小，本地可跑，优先选择两层结构：

1. session encoder：对当天 Top20/Top50 候选的横截面统计、行业/概念扩散、强弱分布、尾部风险做代理状态编码。
2. stock path encoder：复用本轮 20 日单股路径 GRU/TCN。
3. exit head：输出 3/5/10/15/20 日收益分布和 best horizon，但评估必须同时看 fixed10/fixed15 对照、2026 forward、以及 raw 是否避免 20 日长持变薄。

下一轮 gate：

1. dev 中必须超过 fixed10，且不能只靠拉长到 20 日。
2. 2026 Top15/Top20 必须至少超过 fixed5，接近或超过 fixed10/fixed15。
3. 若仍失败，则退出模型方向暂时降级，转去“横截面 world model 重新选股 + 固定 10 日退出”的组合，而不是继续调单股退出模型。

## 20. cross_sectional_state_conditioned_exit_policy_v1

### 20.1 假设

第 19 节的失败形态很明确：单股 GRU 退出模型在开发期能学到一部分生命周期，但 2026 前向偏向过长 horizon，导致 raw 被拖薄。一个自然解释是，退出周期不是单股路径的函数，而是单股路径和当日横截面状态的交互函数：同样一只强势股，在候选池普遍强、主题扩散顺畅、市场风险偏好高时可以持久；在候选池分化、拥挤、尾部开始衰竭时应缩短退出。

本轮不重排股票，只在第 19 轮基础上加入 T 日可见的 session state。session state 来自当天 path_sequence Top20 候选池的 20 日序列横截面统计：last/mean/std/quantile/slope，以及 base rank/score 的分布。它不使用未来的 `path_min5/path_max20` 或任意未来收益。

### 20.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_state_conditioned_exit_policy_v1.py
sha256:312389ccf3a74d225ab742efbd29194491b4a2fcc0ce194c0c4eeb38c4f48f7e

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_state_conditioned_exit_policy_v1_dev_2021_2025.json
sha256:b00c2709ddbc7f6657597f8d6f8c3486f4065d42fd8203c82abfd3cef12b83e9

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_state_conditioned_exit_policy_v1_val63_2026.json
sha256:0b8f6987e7bff3f09d1fb831cc5a868586d42254e0f0cc7c21f47e65f17db521
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选源 | `path_sequence` Top20 |
| 输入 | 单股 20 日路径 GRU embedding + 同 session Top20 横截面状态向量 |
| session state | 391 维，来自 T 日可见特征的横截面均值、标准差、分位数、斜率统计和 base rank/score 分布 |
| 标签 | 3/5/10/15/20 日 open-to-open 超额收益、best horizon 分类 |
| 模型 | 小型 GRU path encoder + session state MLP + 多任务 exit head |
| 切分 | dev 为 walk-forward 年度训练/测试；forward 为 2021-2025 训练、2026 val63 测试 |
| 对照 | fixed 3/5/10/15/20、oracle best 3-20、`model_cls`、`model_reg` |
| TopK | 10/15/20，本节重点 Top15/Top20 |

### 20.3 结果

开发期 2021-2025，Top15/Top20 重点结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.012436` | `+0.015761` | `61.90%` | 5.00 | 一周退出基准 |
| fixed 10 | 15 | `+0.022564` | `+0.028852` | `63.17%` | 10.00 | 强基准 |
| fixed 15 | 15 | `+0.028998` | `+0.038080` | `63.31%` | 15.00 | 开发期强 |
| fixed 20 | 15 | `+0.031244` | `+0.043090` | `61.90%` | 20.00 | 开发期最强固定退出 |
| model cls | 15 | `+0.023952` | `+0.027843` | `63.60%` | 12.29 | 赢 fixed10，弱于第 19 轮 |
| model reg | 15 | `+0.025511` | `+0.032057` | `64.31%` | 12.59 | 赢 fixed10，输 fixed15/20 |
| oracle best 3-20 | 15 | `+0.102742` | `+0.111058` | `98.44%` | 10.66 | 上限，不可作策略 |
| fixed 5 | 20 | `+0.011594` | `+0.014919` | `63.46%` | 5.00 | 一周退出基准 |
| fixed 10 | 20 | `+0.020725` | `+0.027013` | `65.01%` | 10.00 | 强基准 |
| fixed 15 | 20 | `+0.027006` | `+0.036087` | `66.86%` | 15.00 | 开发期强 |
| fixed 20 | 20 | `+0.028999` | `+0.040845` | `65.86%` | 20.00 | 开发期最强固定退出 |
| model cls | 20 | `+0.022239` | `+0.026214` | `65.16%` | 12.24 | 略赢 fixed10 |
| model reg | 20 | `+0.023763` | `+0.030260` | `65.16%` | 12.55 | 赢 fixed10，输 fixed15/20 |
| oracle best 3-20 | 20 | `+0.097432` | `+0.105876` | `98.87%` | 10.71 | 上限，不可作策略 |

与第 19 轮相比，开发期并没有提升：第 19 轮 `model_reg::top15` 为 `+0.028701`，本轮降至 `+0.025511`；Top20 从 `+0.026230` 降至 `+0.023763`。这说明简单 session state 并没有增强开发期拟合能力，甚至带来了一点噪声。

2026 val63 前向结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.020049` | `+0.018560` | `76.70%` | 5.00 | 前向一周基准 |
| fixed 10 | 15 | `+0.026975` | `+0.022315` | `77.67%` | 10.00 | 2026 最强 Top15 固定退出 |
| fixed 15 | 15 | `+0.025365` | `+0.015485` | `70.87%` | 15.00 | 接近 fixed10 |
| fixed 20 | 15 | `+0.020236` | `+0.005013` | `66.99%` | 20.00 | raw 明显变薄 |
| model cls | 15 | `+0.014271` | `+0.006729` | `68.93%` | 11.41 | 前向失败 |
| model reg | 15 | `+0.021047` | `+0.012051` | `65.05%` | 12.90 | 刚过 fixed5，低于 fixed10/15 |
| oracle best 3-20 | 15 | `+0.113026` | `+0.107416` | `99.03%` | 10.34 | 上限仍极高 |
| fixed 5 | 20 | `+0.016282` | `+0.014793` | `69.90%` | 5.00 | 前向一周基准 |
| fixed 10 | 20 | `+0.020692` | `+0.016033` | `74.76%` | 10.00 | 强固定退出 |
| fixed 15 | 20 | `+0.021410` | `+0.011530` | `71.84%` | 15.00 | 2026 Top20 最强固定退出 |
| fixed 20 | 20 | `+0.016314` | `+0.001091` | `69.90%` | 20.00 | raw 很弱 |
| model cls | 20 | `+0.009897` | `+0.002082` | `64.08%` | 11.69 | 前向失败 |
| model reg | 20 | `+0.016905` | `+0.008540` | `65.05%` | 13.02 | 略高于 fixed5，低于 fixed10/15 |
| oracle best 3-20 | 20 | `+0.106465` | `+0.100792` | `100.00%` | 10.37 | 上限仍极高 |

与第 19 轮前向相比，本轮有所修复：`model_reg::top15` 从 `+0.017554` 提到 `+0.021047`，Top20 从 `+0.013539` 提到 `+0.016905`。但它仍没有通过强 gate：Top15 低于 fixed10 的 `+0.026975` 和 fixed15 的 `+0.025365`；Top20 低于 fixed10 的 `+0.020692` 和 fixed15 的 `+0.021410`。

### 20.4 反事实分析

第一反事实：如果第 19 轮失败完全因为缺少横截面状态，那么加入 session state 后 2026 应接近或超过 fixed10。实际只从 `+0.017554` 修复到 `+0.021047`，仍低于 fixed10，说明横截面状态是必要信息之一，但简单统计向量不足以表达真正的横截面交互。

第二反事实：如果 session state 只是噪声，前向不应改善。实际 Top15/Top20 都比第 19 轮改善，且 Top15 刚超过 fixed5，说明“市场/候选池状态条件化退出”方向不是错的。

第三反事实：如果模型学到了 2026 的短持风险，平均 horizon 应明显向 10 日收敛，并且 raw 不应被拖薄。实际 `model_reg` 平均 horizon 仍约 13 日，raw 只有 `+0.012051/+0.008540`，低于 fixed10 的 `+0.022315/+0.016033`。它仍然没有充分识别 2026 中长持 raw 变薄的问题。

第四反事实：如果开发期提升是判断方向的充分条件，本轮 dev 应强于第 19 轮。实际 dev 变弱但 forward 变强，说明开发期最优不等于前向最优；这一点反过来支持必须用 2026 forward 和未来 rolling forward 做硬裁判，不能只按开发期挑模型。

第五反事实：如果退出策略已经接近上限，oracle 与模型差距应缩小。实际 2026 oracle 仍约 `+10.6%` 到 `+11.3%`，而模型只有 `+1.7%` 到 `+2.1%`。退出生命周期空间仍大，模型表达仍不够。

### 20.5 判定

`partial_forward_repair_but_not_enough`。

本轮证明了一个细节：加入 T 日横截面状态后，2026 前向确实比单股退出模型更好，说明第 19 轮的失败不是退出方向本身无效。但简单 session 统计 + MLP 仍没有穿过 fixed10/fixed15，也没有彻底解决 2026 长 horizon raw 变薄的问题。因此不能融入策略，也不进入 formal account。

下一轮不应继续堆更多手写 session 统计。更有价值的方向是两个：

1. `cross_sectional_token_exit_transformer_v1`：对当天 Top20/Top50 候选作为 stock tokens 做轻量 attention，让模型直接学习候选之间的相对生命周期、拥挤和分化，而不是把横截面压成一个均值/分位数向量。目标仍是不重排股票，只选择退出 horizon。
2. `world_model_rank_plus_fixed10_v1`：承认退出 horizon 暂时学不稳，回到第 14 节较有前向增量的 cross-sectional world model，用固定 10 日退出重新评估选股 alpha。第 18-20 节已经说明固定 10 日比 5 日更合理。

下一轮 gate：

1. token exit Transformer 参数必须小，本地可训；输入只能使用 T 日可见序列和候选横截面。
2. dev 不要求超过 fixed20，但必须稳定超过 fixed10，并且 2026 必须超过 fixed10 或至少同时超过 fixed5、fixed15 中较强者。
3. 若 token exit Transformer 仍不能过 fixed10，则退出模型路线暂时降级，转向“横截面选股 world model + fixed10/15 formal account”。

## 21. cross_sectional_token_exit_transformer_v1

### 21.1 假设

第 20 节说明，把横截面压缩成 session 统计向量有帮助，但不足以穿过 fixed10/fixed15。一个更接近 unified market world model 的做法，是让当天候选股票作为 tokens 互相 attention：模型能直接看到候选之间的相对强弱、拥挤、分化和生命周期位置，而不是只看均值/分位数。

本轮仍然不重排股票，只预测每个 token 的退出 horizon。这样可以把问题锁定为“退出周期是否可学”，避免用重新排序混淆结论。

### 21.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_token_exit_transformer_v1.py
sha256:b778dabb1a7a3ae7fadf760b66775725ba297980ca63d9256e976a7b58f16345

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_token_exit_transformer_v1_dev_2021_2025.json
sha256:1e0e8399d5ac3353e5ba74fcdf98a9360a05ae88db8101d4055a987a618bf2a4

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_token_exit_transformer_v1_val63_2026.json
sha256:6a348a9ad33a958d3098c58d333c15de21e36b3a9e1f76c6cdf1487038583112
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选源 | `path_sequence` Top20 |
| 输入 | 每个 session 的 Top20 stock tokens；token 特征为 T 日可见 20 日序列的 last/mean/std/slope summary |
| 标签 | 每个 token 的 3/5/10/15/20 日 open-to-open 超额收益、best horizon 分类 |
| 模型 | 2 层小型 TransformerEncoder，hidden=64，nhead=4；多任务输出 best horizon 分类 + horizon 收益回归 |
| 切分 | dev 为年度 walk-forward；forward 为 2021-2025 训练、2026 val63 测试 |
| 约束 | 模型只选退出 horizon；评估排序仍按原 path_sequence Top20 |
| 对照 | fixed 3/5/10/15/20、oracle best 3-20、`model_cls`、`model_reg` |

### 21.3 结果

开发期 2021-2025，Top15/Top20 重点结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.012436` | `+0.015761` | `61.90%` | 5.00 | 一周退出基准 |
| fixed 10 | 15 | `+0.022564` | `+0.028852` | `63.17%` | 10.00 | 强基准 |
| fixed 15 | 15 | `+0.028998` | `+0.038080` | `63.31%` | 15.00 | 开发期强 |
| fixed 20 | 15 | `+0.031244` | `+0.043090` | `61.90%` | 20.00 | 开发期最强固定退出 |
| model cls | 15 | `+0.019711` | `+0.026207` | `64.45%` | 9.65 | 低于 fixed10 |
| model reg | 15 | `+0.027682` | `+0.038250` | `63.74%` | 12.58 | 明显赢 fixed10，接近 fixed15 |
| oracle best 3-20 | 15 | `+0.102742` | `+0.111058` | `98.44%` | 10.66 | 上限，不可作策略 |
| fixed 5 | 20 | `+0.011594` | `+0.014919` | `63.46%` | 5.00 | 一周退出基准 |
| fixed 10 | 20 | `+0.020725` | `+0.027013` | `65.01%` | 10.00 | 强基准 |
| fixed 15 | 20 | `+0.027006` | `+0.036087` | `66.86%` | 15.00 | 开发期强 |
| fixed 20 | 20 | `+0.028999` | `+0.040845` | `65.86%` | 20.00 | 开发期最强固定退出 |
| model cls | 20 | `+0.018708` | `+0.025177` | `66.43%` | 9.82 | 低于 fixed10 |
| model reg | 20 | `+0.025389` | `+0.035860` | `65.72%` | 12.81 | 明显赢 fixed10，略低于 fixed15 |
| oracle best 3-20 | 20 | `+0.097432` | `+0.105876` | `98.87%` | 10.71 | 上限，不可作策略 |

开发期里 `model_reg` 有生命力，尤其 Top15 达到 `+0.027682`，强于第 20 节的 `+0.025511`，也接近第 19 节的 `+0.028701`。年度拆分显示它在 2023/2024/2025 的 Top15 分别为 `+0.015092`、`+0.041635`、`+0.026197`，平均 horizon 分别为 8.63、14.89、14.37。模型在开发期会随年份改变 horizon，不是固定 10/15/20 的硬规则。

2026 val63 前向结果：

| 口径 | TopK | mean label | mean raw | 正 session 占比 | 平均 horizon | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 15 | `+0.020049` | `+0.018560` | `76.70%` | 5.00 | 前向一周基准 |
| fixed 10 | 15 | `+0.026975` | `+0.022315` | `77.67%` | 10.00 | 2026 最强 Top15 固定退出 |
| fixed 15 | 15 | `+0.025365` | `+0.015485` | `70.87%` | 15.00 | 接近 fixed10 |
| fixed 20 | 15 | `+0.020236` | `+0.005013` | `66.99%` | 20.00 | raw 明显变薄 |
| model cls | 15 | `+0.016889` | `+0.001838` | `66.99%` | 17.46 | 过度长持，失败 |
| model reg | 15 | `+0.019469` | `+0.005971` | `66.99%` | 17.38 | 过度长持，低于 fixed5/10/15 |
| oracle best 3-20 | 15 | `+0.113026` | `+0.107416` | `99.03%` | 10.34 | 上限仍极高 |
| fixed 5 | 20 | `+0.016282` | `+0.014793` | `69.90%` | 5.00 | 前向一周基准 |
| fixed 10 | 20 | `+0.020692` | `+0.016033` | `74.76%` | 10.00 | 强固定退出 |
| fixed 15 | 20 | `+0.021410` | `+0.011530` | `71.84%` | 15.00 | 2026 Top20 最强固定退出 |
| fixed 20 | 20 | `+0.016314` | `+0.001091` | `69.90%` | 20.00 | raw 很弱 |
| model cls | 20 | `+0.013567` | `-0.001719` | `67.96%` | 17.45 | raw 转负，失败 |
| model reg | 20 | `+0.015696` | `+0.002138` | `68.93%` | 17.34 | 低于 fixed5/10/15 |
| oracle best 3-20 | 20 | `+0.106465` | `+0.100792` | `100.00%` | 10.37 | 上限仍极高 |

本轮没有通过 forward gate。虽然 dev 中 token attention 强于 fixed10，但 2026 中模型平均 horizon 被推到 17 日左右，正好落入 fixed20 raw 变薄的区域。它甚至弱于第 20 节的 session state MLP：第 20 节 `model_reg::top15` 为 `+0.021047`，本轮降至 `+0.019469`。

### 21.4 反事实分析

第一反事实：如果 token attention 能直接学到跨股票生命周期，2026 应至少不输 fixed5，并接近 fixed10。实际 Top15/Top20 都低于 fixed5/fixed10，说明 attention 结构本身不能自动解决状态切换问题。

第二反事实：如果第 20 节失败只是因为横截面被压缩成统计向量，本轮 token 化后应显著前向改善。实际本轮比第 20 节更差，说明问题不只是“有没有 token attention”，还包括训练目标会被开发期长持收益吸引。

第三反事实：如果模型能识别 2026 的长持风险，平均 horizon 应向 10/15 日靠拢。实际 `model_reg` 平均 horizon 约 17.3-17.4 日，接近开发期长持偏好，说明它没有学到 2026 的风险偏好切换。

第四反事实：如果更大模型容量一定更好，本轮应比第 19/20 节更稳。实际开发期看起来不错，前向更差，说明在当前样本规模下，token Transformer 更容易把开发期生命周期风格固化下来。

第五反事实：如果退出方向已经无空间，oracle 2026 不应仍有 `+10%` 以上。实际 oracle 仍很高，失败仍在模型可学性/状态建模，而不是生命周期空间消失。

### 21.5 判定

`rejected_as_token_exit_transformer_overextends_forward_horizon`。

第 19-21 节连在一起给出清晰结论：退出生命周期方向有空间，但直接学习动态 horizon 不够稳。单股 GRU、session state MLP、token Transformer 都会在开发期学到一部分，但 2026 前向无法稳定穿过 fixed10/fixed15，尤其容易过度长持。

因此退出模型路线暂时降级。下一轮不再继续调动态退出模型，而是回到更稳的执行假设：**固定 10/15 日退出明显优于 5 日**。接下来应该评估 `world_model_rank_plus_fixed10_v1`：用横截面 world model 改善选股排序，但退出周期固定为 10/15 日，避免动态 horizon 学偏。

下一轮 gate：

1. 不允许靠降低尾部权重卡指标，仍以 Top15/Top20 高收益厚度为主。
2. 以 fixed10/fixed15 作为统一退出，不再让模型预测 horizon。
3. 必须比较 base/path/world rank 在 label10/label15/label20、raw、正 session 占比和年度拆分上的表现。
4. 如果预测层通过，再生成 `/tmp` formal account 输入，检查平均持仓、五年自然年收益和 2026 前向。

## 22. world_model_rank_plus_fixed10_v1

### 22.1 假设

第 19-21 节说明动态退出模型容易在 2026 过度长持，但第 18 节又明确显示 fixed10/fixed15 比 fixed5 更合理。因此本轮不再训练新模型，先复评第 14 节已有的 cross-sectional world model：把注意力从 label5 转到 fixed10/fixed20，观察横截面选股 world model 是否在更长持仓周期下放大 alpha。

关键问题不是“2026 某个组合是否特别亮”，而是：开发期和 2026 是否同时支持同一 ranking 变体在 10/20 日相对 base 有增量。如果只有 2026 亮、开发期不支持，就更像风格切换，不应融入。

### 22.2 产物和口径

本轮不重新训练，只复评已有第 14 节产物。脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_world_model_v1.py
sha256:f67773c4e626dd81d09a59d38c477aebe51651ecfe0fbf811bc7b1094123374d

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_world_dev_2021_2025.json
sha256:d555343fe24e7fc9babc99c273aaa68acdd2379c0f238cad43dab9fa0a3976a7

.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_world_val63_2026.json
sha256:67ee60e43be82ad1ead016775e054f502f0a5ec512dc3789aa7aeb5c1c530df5

.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_world_model_rank_plus_fixed10_v1.py
sha256:b18e7b7a29f1e86a6b946a722840a77e67184260bfa90553dc86a22051c239a1

.tmp/quantx-research/deep-learning-alpha-search-v1/world_model_rank_plus_fixed10_v1_recheck.json
sha256:99f2adc2b1b978c846626831c551ddc9723460e2a52f42805e9768235b25363d
```

复评口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | 第 14 节 cross-sectional world model Top200 stock-token Transformer |
| 排序变体 | `base`、`cs_q5`、`cs_q20`、`cs_world`、`blend_q5_25`、`blend_world_25` |
| TopK | 15、20 |
| 指标 | label5/10/20、raw5、path_max20、正 session 占比、相对 base 的 delta |
| strict gate | dev 的 delta10/delta20 均为正，且 2026 的 delta10/delta20 均为正 |

这里的 strict gate 故意严一点：如果固定 10/20 日路线要替代动态退出路线，它必须不是单点前向亮点，而是开发期和 2026 都能解释。

### 22.3 结果

开发期 2021-2025 Top15/Top20 重点结果：

| 变体 | TopK | label5 | label10 | label20 | raw5 | 正 session 占比 | path max20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 15 | `+0.010509` | `+0.022211` | `+0.032180` | `+0.012960` | `59.47%` | `+0.142675` |
| cs_q5 | 15 | `+0.011445` | `+0.020544` | `+0.027304` | `+0.013897` | `65.79%` | `+0.144413` |
| cs_q20 | 15 | `+0.013334` | `+0.023606` | `+0.033528` | `+0.015785` | `63.68%` | `+0.148630` |
| cs_world | 15 | `+0.013736` | `+0.023016` | `+0.032992` | `+0.016187` | `65.79%` | `+0.146887` |
| blend_q5_25 | 15 | `+0.011489` | `+0.022417` | `+0.029850` | `+0.013940` | `63.16%` | `+0.141949` |
| blend_world_25 | 15 | `+0.011207` | `+0.021489` | `+0.030115` | `+0.013658` | `60.00%` | `+0.141033` |
| base | 20 | `+0.009543` | `+0.019722` | `+0.027603` | `+0.011994` | `63.68%` | `+0.134173` |
| cs_q20 | 20 | `+0.012027` | `+0.020476` | `+0.029471` | `+0.014478` | `67.37%` | `+0.141826` |
| cs_world | 20 | `+0.013098` | `+0.021123` | `+0.029483` | `+0.015550` | `68.95%` | `+0.141530` |
| blend_q5_25 | 20 | `+0.010267` | `+0.021107` | `+0.026851` | `+0.012719` | `63.68%` | `+0.137351` |
| blend_world_25 | 20 | `+0.010564` | `+0.020502` | `+0.027804` | `+0.013016` | `63.68%` | `+0.136918` |

2026 val63 Top15/Top20 重点结果：

| 变体 | TopK | label5 | label10 | label20 | raw5 | 正 session 占比 | path max20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 15 | `+0.006436` | `+0.008859` | `+0.002212` | `+0.006220` | `66.67%` | `+0.117543` |
| cs_q5 | 15 | `+0.004312` | `+0.007328` | `+0.012214` | `+0.004097` | `61.90%` | `+0.128659` |
| cs_q20 | 15 | `+0.008780` | `+0.010439` | `+0.014001` | `+0.008565` | `71.43%` | `+0.132012` |
| cs_world | 15 | `+0.006777` | `+0.009359` | `+0.007695` | `+0.006561` | `57.14%` | `+0.128754` |
| blend_q5_25 | 15 | `+0.011099` | `+0.012342` | `+0.019997` | `+0.010883` | `71.43%` | `+0.126916` |
| blend_world_25 | 15 | `+0.008823` | `+0.013138` | `+0.016410` | `+0.008607` | `76.19%` | `+0.128289` |
| base | 20 | `+0.008575` | `+0.015123` | `+0.008444` | `+0.008359` | `76.19%` | `+0.116167` |
| cs_q20 | 20 | `+0.005019` | `+0.009961` | `+0.016854` | `+0.004803` | `61.90%` | `+0.126465` |
| cs_world | 20 | `+0.006805` | `+0.008344` | `+0.009802` | `+0.006590` | `61.90%` | `+0.122057` |
| blend_q5_25 | 20 | `+0.011831` | `+0.018029` | `+0.029584` | `+0.011615` | `76.19%` | `+0.127671` |
| blend_world_25 | 20 | `+0.009608` | `+0.014332` | `+0.017629` | `+0.009393` | `76.19%` | `+0.122105` |

复评排序的前几名：

| 变体 | TopK | dev delta10 | dev delta20 | 2026 delta10 | 2026 delta20 | strict gate | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| blend_q5_25 | 20 | `+0.001386` | `-0.000752` | `+0.002906` | `+0.021139` | false | 2026 很亮，开发期 20 日不支持 |
| blend_q5_25 | 15 | `+0.000206` | `-0.002330` | `+0.003483` | `+0.017785` | false | 同样是前向亮点，开发期不支持 |
| cs_q20 | 15 | `+0.001395` | `+0.001348` | `+0.001580` | `+0.011789` | true | 唯一较有解释力的稳定苗头 |
| blend_world_25 | 15 | `-0.000722` | `-0.002065` | `+0.004279` | `+0.014197` | false | 开发期不支持 |
| cs_world | 15 | `+0.000805` | `+0.000812` | `+0.000500` | `+0.005483` | true | 稳但薄 |

### 22.4 反事实分析

第一反事实：如果 `blend_q5_25` 是稳定的固定长持选股 alpha，它在开发期 label20 相对 base 不应为负。实际 Top20 dev delta20 为 `-0.000752`，Top15 dev delta20 为 `-0.002330`。因此 2026 的 Top20 label20 `+0.029584` 更像风格前向亮点，不能直接融入。

第二反事实：如果 cross-sectional world model 完全无用，任何变体都不应同时在 dev 和 2026 的 10/20 日增量为正。实际 `cs_q20::top15` 与 `cs_world::top15` 通过 strict gate，说明横截面 token 模型有一条稳定但很薄的信号。

第三反事实：如果固定 10/20 的执行调整足以带来质变，稳定通过 gate 的变体应有明显厚度。实际 `cs_q20::top15` 的 dev delta10/20 只有约 `+0.14%/+0.13%`，2026 delta10/20 为 `+0.16%/+1.18%`。这不足以支撑用户要求的五年几十倍目标。

第四反事实：如果只看 2026 最强数值就推进，`blend_q5_25::top20` 会被选中；但它开发期 20 日不支持。这个反事实说明当前最危险的过拟合路径不是模型训练本身，而是研究者在后验挑 2026 亮点。

第五反事实：如果动态退出模型失败意味着应放弃所有 DL，固定退出复评也不会出现任何稳定正 delta。实际仍有 `cs_q20::top15` 这种小而稳定的苗头，说明 DL 横截面排序不是完全死路，只是当前标签和候选池的收益厚度不够。

### 22.5 判定

`stable_but_too_thin_world_rank_signal`。

本轮没有找到可融入策略的方向。`blend_q5_25` 在 2026 的 20 日表现很诱人，但开发期不支持，不能作为鲁棒策略；`cs_q20::top15` 和 `cs_world::top15` 能同时穿过 dev/forward 的 10/20 日增量 gate，但增量太薄，离“鲁棒性强，容易出收益”差很远。

下一轮不应在第 14 节模型上继续调 blend 权重，也不应因为 2026 的 `blend_q5_25` 亮点去硬卡。更可能的方向是重新定义训练目标，让模型直接学习**右尾延续的横截面排序**，而不是 q5/q20 分类的混合分数：

1. `right_tail_continuation_ranker_v1`：固定 10/20 日退出，目标改为 session 内 label20 或 raw20 的 pairwise/listwise rank loss，同时加 label10 稳定项。
2. 候选池仍用 Top200，但评估只看 Top15/Top20，不降低尾部权重来卡指标。
3. 必须输出多尺度 metric：dev/forward 的 label10、label20、raw10/raw20、正 session 占比、年度拆分、相对 base delta、以及 2026 是否只是单点风格。
4. 如果该方向仍只给出小 delta，则说明日频 OHLCV+静态行业概念的横截面 DL 上限仍太薄，需要引入新的 point-in-time 数据源或资金流/盘口代理。

## 23. right_tail_continuation_ranker_v1

### 23.1 假设

第 22 节复评说明，旧 cross-sectional world model 在 10/20 日上只有很薄的稳定增量。一个可能原因是训练目标不够直接：第 14 节模型通过 q5/q20 分类、多任务右尾/左尾 proxy 组合出分数，但并没有直接优化 TopK 在固定 10/20 日退出下的右尾延续排序。

本轮直接把训练目标改成 listwise rank：每个 rebalance session 是一个 Top200 股票 token 集合，模型输出 `score10/score20/score_mix`，用 label10/label20 的 session 内 softmax 分布作为目标，训练模型把 10/20 日超额收益高的股票排前。退出周期固定，不再预测 horizon。

### 23.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_right_tail_continuation_ranker_v1.py
sha256:10a63d75a7dd45231723968abf105d011c416703bba019312bd07b1510dfe2a5

.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_continuation_ranker_v1_smoke_2021_2023q1.json
sha256:2d42b5ee4280e371cb3fcc61df1e5d0d1e1ee0f21b32e74145bfbab5a06d41b5

.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_continuation_ranker_v1_dev_2021_2025.json
sha256:e26808b40992ccc507120793141dab0d2524679285e5ca1676aa9588796d898b

.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_continuation_ranker_v1_val63_2026.json
sha256:eed5e58d54a9a662e17e6cfb4e393a720d980d0fdf9eca2fa2ef698dcc301857
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | base Top200 |
| 输入 | T 日可见 20 日序列 summary，Top200 stock tokens |
| 模型 | 2 层小型 TransformerEncoder，hidden=64，nhead=4 |
| 目标 | label10/label20/listwise mix，固定 10/20 日右尾延续排序 |
| 排序变体 | `base`、`rt10`、`rt20`、`rt_mix`、`blend_rt20_25`、`blend_mix_25` |
| TopK | 10/15/20/30，本节重点 Top15/20 |
| 切分 | dev 年度 walk-forward；forward 为 2021-2025 训练、2026 val63 测试 |

### 23.3 结果

开发期 2021-2025 Top15/Top20 重点结果：

| 变体 | TopK | label5 | label10 | label20 | raw10 | raw20 | label10 正比 | label20 正比 | delta10 | delta20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 15 | `+0.010509` | `+0.022211` | `+0.032180` | `+0.026842` | `+0.041909` | `67.89%` | `65.79%` | `0.000000` | `0.000000` |
| rt10 | 15 | `+0.012622` | `+0.022434` | `+0.032106` | `+0.027065` | `+0.041836` | `66.32%` | `67.89%` | `+0.000223` | `-0.000074` |
| rt20 | 15 | `+0.011072` | `+0.021755` | `+0.029827` | `+0.026387` | `+0.039556` | `67.37%` | `66.32%` | `-0.000455` | `-0.002353` |
| rt_mix | 15 | `+0.011720` | `+0.023303` | `+0.030025` | `+0.027934` | `+0.039754` | `70.53%` | `65.79%` | `+0.001092` | `-0.002155` |
| blend_rt20_25 | 15 | `+0.012143` | `+0.024240` | `+0.031599` | `+0.028871` | `+0.041328` | `69.47%` | `68.42%` | `+0.002029` | `-0.000582` |
| blend_mix_25 | 15 | `+0.013264` | `+0.024136` | `+0.032092` | `+0.028768` | `+0.041821` | `70.53%` | `68.42%` | `+0.001926` | `-0.000088` |
| base | 20 | `+0.009543` | `+0.019722` | `+0.027603` | `+0.024353` | `+0.037333` | `71.58%` | `64.74%` | `0.000000` | `0.000000` |
| rt10 | 20 | `+0.010831` | `+0.021367` | `+0.028744` | `+0.025998` | `+0.038473` | `68.42%` | `68.42%` | `+0.001645` | `+0.001141` |
| rt20 | 20 | `+0.009557` | `+0.019204` | `+0.026031` | `+0.023836` | `+0.035760` | `68.42%` | `67.89%` | `-0.000517` | `-0.001573` |
| rt_mix | 20 | `+0.009717` | `+0.019361` | `+0.025842` | `+0.023992` | `+0.035571` | `66.84%` | `67.37%` | `-0.000361` | `-0.001762` |
| blend_rt20_25 | 20 | `+0.010756` | `+0.020659` | `+0.026430` | `+0.025290` | `+0.036159` | `68.95%` | `67.89%` | `+0.000937` | `-0.001173` |
| blend_mix_25 | 20 | `+0.011353` | `+0.021064` | `+0.027777` | `+0.025696` | `+0.037506` | `68.95%` | `66.84%` | `+0.001343` | `+0.000173` |

2026 val63 Top15/Top20 重点结果：

| 变体 | TopK | label5 | label10 | label20 | raw10 | raw20 | label10 正比 | label20 正比 | delta10 | delta20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 15 | `+0.006436` | `+0.008859` | `+0.002212` | `+0.005266` | `-0.012172` | `71.43%` | `71.43%` | `0.000000` | `0.000000` |
| rt10 | 15 | `+0.005309` | `+0.006997` | `+0.001509` | `+0.003404` | `-0.012875` | `57.14%` | `52.38%` | `-0.001862` | `-0.000703` |
| rt20 | 15 | `+0.005102` | `+0.012596` | `+0.017093` | `+0.009004` | `+0.002709` | `66.67%` | `52.38%` | `+0.003737` | `+0.014881` |
| rt_mix | 15 | `+0.007104` | `+0.013937` | `+0.012962` | `+0.010344` | `-0.001422` | `71.43%` | `57.14%` | `+0.005078` | `+0.010750` |
| blend_rt20_25 | 15 | `+0.009676` | `+0.016499` | `+0.017236` | `+0.012907` | `+0.002852` | `76.19%` | `71.43%` | `+0.007640` | `+0.015024` |
| blend_mix_25 | 15 | `+0.010224` | `+0.015681` | `+0.014059` | `+0.012088` | `-0.000325` | `76.19%` | `66.67%` | `+0.006822` | `+0.011847` |
| base | 20 | `+0.008575` | `+0.015123` | `+0.008444` | `+0.011531` | `-0.005940` | `80.95%` | `71.43%` | `0.000000` | `0.000000` |
| rt10 | 20 | `+0.004757` | `+0.005935` | `+0.003648` | `+0.002342` | `-0.010736` | `57.14%` | `52.38%` | `-0.009188` | `-0.004796` |
| rt20 | 20 | `+0.002191` | `+0.006469` | `+0.014677` | `+0.002876` | `+0.000293` | `66.67%` | `52.38%` | `-0.008654` | `+0.006233` |
| rt_mix | 20 | `+0.006280` | `+0.011195` | `+0.008307` | `+0.007603` | `-0.006077` | `61.90%` | `47.62%` | `-0.003928` | `-0.000137` |
| blend_rt20_25 | 20 | `+0.008173` | `+0.015087` | `+0.015812` | `+0.011494` | `+0.001428` | `76.19%` | `71.43%` | `-0.000037` | `+0.007368` |
| blend_mix_25 | 20 | `+0.010255` | `+0.016847` | `+0.012664` | `+0.013254` | `-0.001720` | `80.95%` | `66.67%` | `+0.001724` | `+0.004219` |

### 23.4 反事实分析

第一反事实：如果 right-tail listwise 目标真正解决了第 22 节“稳定但太薄”的问题，dev 中 Top15/Top20 的 label20 应明显超过 base。实际 dev Top15 的所有模型变体 label20 都低于 base；Top20 只有 `rt10` 和 `blend_mix_25` 小幅高于 base，delta20 仅 `+0.001141` 和 `+0.000173`。训练目标没有稳定吃到开发期右尾。

第二反事实：如果 smoke 的强表现是有效信号，完整 dev 应继续显著领先。实际 smoke 中 `rt_mix::top15` label20 很强，但完整 dev 回落到低于 base，说明 smoke 小样本不可采信。

第三反事实：如果 dev 中唯一小苗头 `rt10::top20` 是稳定规律，2026 应至少不低于 base。实际 `rt10::top20` 2026 delta10 为 `-0.009188`、delta20 为 `-0.004796`，前向崩掉。

第四反事实：如果 2026 的 `blend_rt20_25::top15` 是可融入方向，它在 dev label20 不应低于 base。实际 dev Top15 delta20 为 `-0.000582`。这与第 22 节 `blend_q5_25` 的问题类似：2026 亮点很强，但开发期不支持，不能因为前向漂亮就硬卡。

第五反事实：如果直接优化 label20 能自然避开 raw20 问题，2026 raw20 应显著改善且正 session 比例提升。实际 `blend_rt20_25::top15` raw20 仅 `+0.002852`，虽然好于 base 的 `-0.012172`，但 label20 正比仍只有 `71.43%`，并没有形成压倒性稳定。

### 23.5 判定

`rejected_as_forward_bright_but_dev_right_tail_not_supported`。

本轮提供了两个信息：

1. 直接 listwise rank label10/20 并没有在开发期稳定超过 base 的右尾延续；说明当前 Top200 日频序列特征对 20 日右尾排序的可学性仍然薄。
2. 2026 上 `blend_rt20_25::top15` 很亮，但与 dev 支持方向不一致，不能融入策略。它更像 2026 风格下的后验亮点。

第 19-23 节合并后的判断：继续在同一日频 OHLCV + 静态行业/概念特征上换小模型结构，收益上限正在收敛。真正可能改变收益厚度的方向不应再是“同数据、同候选池、换 loss/模型”，而应引入新的可见状态或新的执行对象。

下一轮方向切换为 `tradeability_and_intraday_proxy_soil_v1`：不先训练 DL，而是诊断 formal/account 层损耗、涨停/一字/高开/换手/成交可得性等可交易性代理，寻找是否存在“预测层 label 很厚但账户收益被交易实现吞掉”的金子。如果能找到肥土壤，再训练小模型；如果没有，就说明当前日频预测层本身厚度不足，需要外部 point-in-time 数据源。

## 24. tradeability_and_intraday_proxy_soil_v1

### 24.1 假设

第 19-23 节连续说明：在同一套日频 OHLCV、行业概念和 Top200 候选池上继续换 GRU/Transformer/loss，增量开始变薄。但 formal account 层的收益和预测层 label 之间仍可能存在一类损耗：涨停买不到、高开追入、日内回落、换手拥挤、成交可得性不足，导致预测层看起来厚，账户层吃不到。

本轮先不训练新 DL，只收拢已有的可交易性、日内/隔夜、VWAP/流动性代理诊断，检验一个反事实：如果问题主要在交易实现而不是预测层，简单的可见代理应该能在不牺牲 label15/label20 的情况下，降低早期失败率，并且在 dev 与 2026 同时成立。

### 24.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_tradeability_and_intraday_proxy_soil_v1.py
sha256:d4c6ae7e45b78f507b6273f3987f4b4921f37e871fbdc2fab89500f6fb311783

.tmp/quantx-research/deep-learning-alpha-search-v1/tradeability_and_intraday_proxy_soil_v1_summary.json
sha256:5c20030219dd4e84228578465a3e415469ad6e06f95938a25846ffb255e8f33f
```

输入诊断文件：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/formal_tradeable_sequence_soil_v1_summary.json
.tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_dev_2021_2025_diagnostic.json
.tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_val63_2026_diagnostic.json
.tmp/quantx-research/intraday-overnight-path-v1/intraday_overnight_path_dev_2021_2025_diagnostic.json
.tmp/quantx-research/intraday-overnight-path-v1/intraday_overnight_path_val63_2026_diagnostic.json
.tmp/quantx-research/vwap-liquidity-path-v1/vwap_liquidity_path_dev_2021_2025_diagnostic.json
.tmp/quantx-research/vwap-liquidity-path-v1/vwap_liquidity_path_val63_2026_diagnostic.json
```

本轮拆成两个 gate：

| gate | 要求 | 用途 |
| --- | --- | --- |
| label gate | dev 与 2026 的 label15/label20 均高于 base，且绝对收益不太薄 | 判断是否有更激进的右尾路径土壤 |
| tradeability gate | 通过 label gate，同时 dev 与 2026 的 early failure 不高于 base | 判断是否真的解决交易实现损耗 |

这个拆分是为了避免一个常见误判：某个代理让收益更厚，但早期失败也更高。它不是“可交易性改善”，而是“更激进地拥抱右尾”。这类方向可以继续研究，但不能被包装成交易过滤。

### 24.3 formal account 参照

已有 formal account 基准并不差，但远不到目标：

| 账户基准 | 区间 | total return | final multiple | avg holding | avg positions | 年度正收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path_sequence | 2022-2025 dev | `+7.774911` | `8.774911x` | `9.36` | `19.40` | 2022/2023/2024/2025 |
| path_sequence | 2026 val63 | `+0.191783` | `1.191783x` | `9.94` | `19.48` | 2026 |
| robust_weekly | 2023-2025 dev | `+4.051303` | `5.051303x` | `12.73` | `19.56` | 2023/2024/2025 |
| robust_weekly | 2026 val63 | `+0.297109` | `1.297109x` | `11.02` | `19.50` | 2026 |

这说明已有路线满足“平均持仓 >5、前向正收益”的雏形，但五年 5-9 倍级别离几十倍到 100 倍仍差一个数量级。第 24 轮要回答的是：这个数量级缺口能不能靠交易可实现性代理补回来。

### 24.4 结果

`tradable_big_winner` 的 due5/pool500 中，有 4 个组合通过 label gate，但没有任何组合通过 tradeability gate：

| 组合 | TopK | dev label15 | dev label20 | 2026 label15 | 2026 label20 | dev early failure delta | 2026 early failure delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| path_topq | 10 | `+0.036658` | `+0.039914` | `+0.034820` | `+0.030431` | `+0.097396` | `+0.066667` |
| tw20 | 15 | `+0.031960` | `+0.034942` | `+0.018333` | `+0.016847` | `+0.047222` | `+0.060317` |
| path_tw15_25 | 10 | `+0.035973` | `+0.040171` | `+0.033249` | `+0.028887` | `+0.085417` | `+0.038095` |
| path_tw20_25 | 10 | `+0.035745` | `+0.040357` | `+0.028885` | `+0.026323` | `+0.088021` | `+0.033333` |

这 4 个组合的重要性不在于“可以融入”，而在于它们共同指向一个事实：更强的右尾路径代理确实能让 label15/20 更厚，但代价是 early failure 也更高。也就是说，当前可见代理没有解决交易实现损耗，而是把组合推向更高波动、更强趋势、更高失败率的右尾暴露。

`intraday_overnight` 没有任何组合通过 strict gate。最亮的是 `overnight_crowding_fade`，但它属于典型 2026-only：

| 组合 | TopK | dev label5 | 2026 label5 | dev 正年份 | 判读 |
| --- | ---: | ---: | ---: | --- | --- |
| due5::overnight_crowding_fade | 10 | `-0.002247` | `+0.023630` | 2021/2024/2025 | 2026 很亮，dev 不支持 |
| due5::overnight_crowding_fade | 15 | `-0.001773` | `+0.017407` | 2021 | 2026 风格点 |
| due5::overnight_crowding_fade | 20 | `-0.001634` | `+0.016290` | 2021 | 2026 风格点 |

`vwap_liquidity` 也没有任何组合通过 strict gate。相对靠前的 `quiet_trend_liquidity` 在 2026 有正数，但 dev 仍为负：

| 组合 | TopK | dev label5 | 2026 label5 | dev 正年份 | 判读 |
| --- | ---: | ---: | ---: | --- | --- |
| due5::quiet_trend_liquidity | 20 | `-0.003785` | `+0.007364` | 2022 | dev 不支持 |
| all::quiet_trend_liquidity | 10 | `-0.005632` | `+0.009586` | 无 | dev 不支持 |
| all::quiet_trend_liquidity | 20 | `-0.003888` | `+0.006502` | 无 | dev 不支持 |

`tradable_big_winner` 的特征重要性前五也值得记录：

| 排名 | 特征 | mean importance |
| ---: | --- | ---: |
| 1 | `path::market_ret20_disp` | `1602.00` |
| 2 | `path::market_breadth60` | `1430.25` |
| 3 | `path::vol20_rank_path` | `1405.75` |
| 4 | `path::rps120` | `1379.50` |
| 5 | `path::market_ret60_median` | `1319.75` |

这组重要性说明，真正驱动右尾厚度的不是单票微观执行代理，而是市场状态、市场宽度、相对强弱路径和波动路径。

### 24.5 反事实分析

第一反事实：如果 formal/account 层主要损耗来自“可交易性差”，那么可见交易代理应该在 dev 和 2026 同时降低 early failure，同时保持或提高 label15/20。实际通过 label gate 的 4 个组合，early failure delta 全部为正，dev 增加约 `+4.72%` 到 `+9.74%`，2026 增加约 `+3.33%` 到 `+6.67%`。所以它们不是交易损耗修复。

第二反事实：如果降低 early failure 是正确主线，edge 类组合应该在降低失败率的同时保留收益厚度。实际 edge 类更常见的形态是 2026 局部很亮，dev 的 label20 不支持，或者收益厚度被明显削弱。这与用户“不认可通过降低尾部选股权重来卡要求”的直觉一致：防守过滤很容易把金子也过滤掉。

第三反事实：如果日内/隔夜代理是稳定 alpha，`overnight_crowding_fade` 不应该只在 2026 亮。实际它 dev label5 为负，正年份不完整，说明它更像 2026 的特殊风格，而不是跨 regime 的可迁移信号。

第四反事实：如果 VWAP/流动性代理能解释账户收益损耗，dev 至少应该接近正收益。实际 `quiet_trend_liquidity`、`vwap_breakout_not_chase` 等 dev 端多数为负，不能作为下一轮 DL 训练主土壤。

第五反事实：如果当前缺口只是执行层问题，formal baseline 的 5-9 倍到几十倍之间的差距应该能从代理诊断里看到大幅 label 改善。实际最有价值的改善来自更激进的右尾路径，而不是可交易性修复。这说明下一轮应该学“什么时候接受更高 early failure 去换更厚右尾”，而不是继续找静态交易过滤。

### 24.6 判定

`partial_label_thickness_but_tradeability_not_solved`。

本轮没有找到可以融入策略的交易可实现性方向。日内/隔夜和 VWAP/流动性代理在 2026 有局部亮点，但开发期不支持；交易 edge 过滤也没有形成“降低失败率且保持厚收益”的鲁棒形态。

但本轮不是纯失败。更重要的观察是：`path_topq::top10`、`tw20::top15`、`path_tw15_25::top10`、`path_tw20_25::top10` 同时在 dev 和 2026 提升 label15/20，只是 early failure 同步升高。这指向一个新的、更贴近市场动态的方向：右尾 alpha 可能天然伴随更高早期失败率，关键不是把失败率压低，而是识别**市场状态允许承担失败率时的右尾延续**。

下一轮方向切换为 `state_gated_right_tail_world_model_v1`：

1. 不是继续做交易过滤，也不是降低尾部选股权重。
2. 用市场宽度、20/60 日市场收益分布、vol path、RPS path、候选池右尾/左尾分布作为显式 regime state。
3. 训练轻量模型同时预测 `right_tail_continuation` 与 `early_failure`，但决策层不简单惩罚 early failure；而是学习在什么 regime 下 early failure 可被右尾收益补偿。
4. 评估必须保留 Top10/Top15/Top20，重点看 label15/20、raw15/20、early failure、右尾命中率、年度拆分、2026 前向，以及 formal account 是否能从 8-9x 级别明显上跳。
5. 如果该方向仍只有薄增量，则需要跳出当前数据源，引入 point-in-time 资金流、龙虎榜/大单、分钟级形态或更接近 kronos/unified market world model 的预训练数据。

## 25. state_gated_right_tail_world_model_v1

### 25.1 假设

第 24 节发现一个重要矛盾：更激进的右尾路径代理能够提高 label15/20，但 early failure 也会升高。这个现象不能简单理解成“失败率要压低”，因为右尾 alpha 本身可能天然伴随高失败率。真正的问题应该是：模型能否识别哪些市场状态允许承担更高 early failure，并用右尾收益补偿失败率。

本轮因此训练一个轻量 state-gated right-tail world model：复用第 23 轮 Top200 stock-token Transformer 的数据管线，引入第 24 节提示的重要市场状态特征，包括市场宽度、20/60 日市场收益分布、波动路径、RPS 路径、主题强度和候选池分布。模型同时输出 10/20 日右尾排序、early failure 风险和 session-level risk budget。决策层不把 early failure 简单作为扣分项，而是让 risk budget 决定什么时候允许更激进地追右尾。

### 25.2 产物和口径

训练、分析、formal writer 和账户复核均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_state_gated_right_tail_world_model_v1.py
sha256:8eee84a0dbee1bd861dc93146483a98365e818eb75805ddd6c6ad6513101887c

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_right_tail_world_model_v1_smoke_2021_2023q1.json
sha256:e3f271c8745c69942878bdce97a21d94b778e329ff0601ab278b8f1cb6f4c2c8

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_right_tail_world_model_v1_dev_2021_2025.json
sha256:cc74afdbf26310eb7d8ebd1977ccb90745773fb1882eef811be39e2d7c7b9e1d

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_right_tail_world_model_v1_val63_2026.json
sha256:56e7ec708b2df2195d8545d3ffe7188c94a5d0edd0308209bdfb6a6bf9490dff

.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_state_gated_right_tail_world_model_v1.py
sha256:d0eb7b3ce171c77c7475c27541a74b4057d16e245cb4a6cb7d265590046a70bc

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_right_tail_world_model_v1_analysis.json
sha256:69b7c8de7d67ff8e6d8e5610909e0bf5e0b0af6a55c0b50ad652b4f6ae38e4e2

.tmp/quantx-research/deep-learning-alpha-search-v1/write_state_gated_right_tail_predictions_v1.py
sha256:7c686e4d5e8466f85a4e223c2884435088fa6d73daceea7043b812cdf1054ea5

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_rt_blend_rt20_25_top15_dev_predictions.json
sha256:574c8c6e0986eed0035c9582a29bcdfa7fbe2e37b089f6a0e4e30312aae522cf

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_rt_blend_rt20_25_top15_dev_hold10_account.json
sha256:74c7d097aeaf70a17b44930c1304bd62b4530968a0c64dbffb12c8be9d0ef096

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_rt_blend_rt20_25_top15_dev_hold10_buy15_account.json
sha256:d43bb5216b09a2e44a3814de2d4a64c870aa3d6eacbe1f48c4daaf1899b55461

.tmp/quantx-research/deep-learning-alpha-search-v1/summarize_state_gated_right_tail_world_model_v1.py
sha256:87e6cf2fabce6d43309f8c91177b055c5c86565e1298901996ddc3a9cafb172b

.tmp/quantx-research/deep-learning-alpha-search-v1/state_gated_right_tail_world_model_v1_final_summary.json
sha256:8e9809c5bdd3ce7702dc88531868b28bdff7219b679af815b608f522c9cf4288
```

评估口径：

| 项 | 设置 |
| --- | --- |
| 候选池 | base Top200 |
| 输入 | 20 日 path/world sequence summary + session state |
| 模型 | 2 层小型 TransformerEncoder，hidden=64，session state projection |
| 目标 | label10/label20 listwise rank + early failure joint head + session risk budget |
| TopK | 10/15/20/30，重点 Top15/Top20 |
| 切分 | smoke、2021-2025 年度 walk-forward dev、2021-2025 训练/2026 val63 forward |
| formal 复核 | `blend_rt20_25::top15` 生成 PredictionStore，复用 formal overlap account 回放器 |

### 25.3 预测层结果

Smoke 小样本出现了理想形态：`srt_mix`、`srt_budget` 在 Top15/Top20 上提高 label20，同时 early failure 下降。但完整 dev 后，增量明显回落。

dev/2026 严格配对后有 3 个 strict gate 通过，其中 2 个在 dev 端 early failure 下降，因此触发 formal replay：

| 组合 | dev delta10 | dev delta20 | 2026 delta10 | 2026 delta20 | dev early failure delta | 2026 early failure delta | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| blend_rt20_25::top15 | `+0.001456` | `+0.001354` | `+0.003399` | `+0.005716` | `-0.002105` | `+0.015873` | 最强可复核苗头，但 forward 失败率上升 |
| blend_budget_35::top15 | `+0.001033` | `+0.000263` | `+0.001067` | `+0.003466` | `-0.007368` | `+0.009524` | 稳但极薄 |
| blend_rt20_25::top20 | `+0.001415` | `+0.000841` | `+0.000332` | `+0.002943` | `+0.006842` | `+0.023810` | 收益小增，失败率更高 |

这里最危险的误判是只看 2026 的 `blend_rt20_25::top15`：它 forward label20 增量 `+0.57%` 看起来不错，但 dev label20 只有 `+0.14%`。这不足以支撑用户要求的数量级跃迁。

### 25.4 formal account 复核

因为配对分析给出 `needs_formal_replay`，本轮为 `blend_rt20_25::top15` 生成 PredictionStore，并复用已有 formal overlap account 回放器。PredictionStore 覆盖 190 个 dev session、2850 条记录。

dev 账户复核结果：

| 账户口径 | final multiple | total return | max drawdown | avg positions | avg holding days | avg cash ratio | 年度收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| top15/buy5/hold10 | `1.386358x` | `+0.386358` | `-0.089279` | `9.03` | `15.23` | `80.01%` | 2022 `+6.26%`，2023 `+9.14%`，2024 `+4.07%`，2025 `+15.56%` |
| top15/buy15/hold10 | `1.294344x` | `+0.294344` | `-0.066448` | `23.46` | `15.28` | `80.00%` | 2022 `+7.02%`，2023 `+6.41%`，2024 `+2.37%`，2025 `+11.25%` |

这两个账户复核都不能通过策略要求。`buy5` 版本平均持仓数大于 5，年度收益也为正，但 2022-2025 只有 `1.39x`；`buy15` 提高持仓数量后反而下降到 `1.29x`。这说明预测层小增量没有被账户层放大。

需要注意一个口径问题：第 25 轮 PredictionStore 是 due5 信号，而 formal 回放器按 `hold_days` 分配资金，`hold10` 下会留下约 80% 主现金。因此这个账户复核不能拿来和第 24 节 `path_sequence` 的 8.77x 做完全等价比较。但即便考虑这个资金使用不匹配，当前账户结果也过弱，没有必要继续为它做 2026 formal replay 或融入正式策略。

### 25.5 反事实分析

第一反事实：如果 state-gated risk budget 真正学会了“何时接受失败率换右尾”，完整 dev 不应只留下 `+0.1%` 量级的 label20 增量。实际最强 Top15 dev delta20 只有 `+0.001354`，说明模型更多是在 base rank 上做很薄的微调。

第二反事实：如果 2026 的 forward 改善是稳定信号，dev 应该有足够厚度支撑。实际 2026 `blend_rt20_25::top15` 的 delta20 `+0.005716` 明显大于 dev 的 `+0.001354`，并且 2026 early failure delta 为正。这更像前向阶段风格适配，而不是跨 regime 的强预测能力。

第三反事实：如果账户层只是因为选股太集中而吃不到收益，那么 `buy15` 应该改善或至少不显著变差。实际 `buy15` 从 `1.386x` 降到 `1.294x`，说明扩大持仓不是解法，模型排序厚度本身不足。

第四反事实：如果问题是“过度防守导致金子被过滤”，那么不惩罚 early failure 的 `blend_rt20_25` 应该在账户层保留右尾优势。实际账户层仍只有低双位数年化，说明在当前日频特征里，右尾信号的可迁移厚度不足。

第五反事实：如果同数据源、同候选池、同日频序列还有明显模型红利，state branch 至少应该强于第 22-23 节旧 Transformer 的薄增量。实际第 25 轮虽然更可解释，但收益厚度没有质变。因此继续在这套日频特征上微调模型结构，边际收益越来越低。

### 25.6 判定

`rejected_as_prediction_delta_too_thin_and_account_not_amplified`。

本轮没有找到可融入策略。它给出的正确信息是：市场状态 gating 确实能把某些 2026 右尾改善解释出来，但当前数据源和模型结构只能产生很薄的排序增量。formal dev 账户没有放大这个增量，距离“几年几十倍到 100 倍”的要求非常远。

下一轮不应继续在 Top200 日频 OHLCV + 行业概念静态快照上改 Transformer 头部。更可能出现金子的方向是 `external_pit_flow_or_intraday_world_model_pretraining_v1`：

1. 引入 point-in-time 资金流、龙虎榜/大单、分钟级量价形态、涨停封单/炸板/回封等更接近交易行为的数据，至少先做土壤诊断。
2. 如果无法直接获得外部 PIT 数据，则从已有分钟或更细 OHLCV 数据中构造 intraday token，训练小型 masked world model，目标预测次日/未来一周的横截面右尾分布。
3. 下一轮先不追求复杂大模型，优先验证“新数据源/新状态是否让 label15/20 增量从 0.1% 量级跃迁到 1% 以上”，否则账户层不可能出现数量级提升。
4. 严格保留 train/dev/2026 forward，禁止使用未来函数；所有探索仍只放 `/tmp`。

## 26. external_pit_event_flow_soil_v1

### 26.1 假设

第 25 节之后，继续在 Top200 日频 OHLCV/行业概念 token 上换 Transformer 结构，边际收益已经非常低。真正可能带来数量级跃迁的，应该是更接近交易行为和市场事件传播的数据：涨停/炸板/回封、概念/行业领导股传播、龙虎榜、ETF 资金流、事件后的跟随扩散等。

本轮先不训练新 DL，而是统一收拢 `/tmp` 中已有的事件/资金流代理实验，做一个“新数据源土壤扫描”。目标是回答：这些更接近 PIT 行为的数据，是否已经出现 dev 与 2026 同时厚、并且不是 2026-only 的信号。

### 26.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_external_pit_event_flow_soil_v1.py
sha256:8cb3b176a7828c1943b9f0f4016619b6fb3e4bcc4b766fc6f9b8d08024d4f8f7

.tmp/quantx-research/deep-learning-alpha-search-v1/external_pit_event_flow_soil_v1_summary.json
sha256:f002f757276955e4455ead917f2b0876ed9ef79721484ceb27b3a4a514d92bad
```

重点输入包括：

```text
.tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_dev_2021_2025_label_diagnostic.json
sha256:031f98d76892dc5b56419b047a6abc624b17db11862b0d8f8ca82eba5189aaea

.tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_2026_val63_label_diagnostic.json
sha256:deb5abaa97d689d663314ed84478b64fc2cfc10c8a58c2752def6d0be870a419

.tmp/quantx-research/right-tail-event-formal-v1/formal_overlap_right_tail_event_base35_pool100_top10_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:a245a0c9c29300263fbfc77b354de252b7958643ef23e9e094e66f74fc9993cb

.tmp/quantx-research/right-tail-event-formal-v1/formal_overlap_right_tail_event_base35_pool100_top10_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:d81dbba78e80f3fa53367bc29070406e6403693c47ca0e3c040224ba3feec08c
```

扫描 family：

| family | 内容 |
| --- | --- |
| `limitup_concept_propagation` | 涨停/概念/行业领导股传播 |
| `right_tail_event_candidate` | 右尾事件候选，已有 formal replay |
| `event_propagation_candidates` | 事件传播候选 |
| `post_event_setup` | 事件后回调/冷却形态 |
| `event_continuation_learner` | 事件延续 learner |
| `etf_latent_flow_lead` | ETF latent flow lead |
| `etf_flow_regime` | ETF flow regime selector |
| `lhb_event_alpha` | 龙虎榜事件，但目前只有 2026 forward 诊断，不能做 dev 支持判断 |

gate 口径：

| gate | 要求 |
| --- | --- |
| strict | dev metric > `1.2%`，2026 metric > `0.6%`，dev 至少 4 个正年份 |
| thick | dev metric > `1.8%`，2026 metric > `1.0%`，dev 至少 4 个正年份 |
| incremental | strict 基础上，相对 base/ML 的 dev 与 2026 delta 均为正 |

### 26.3 土壤扫描结果

本轮发现 30 个 thick gate、158 个 strict gate、35 个 incremental gate。最靠前的可解释候选如下：

| family | key | dev metric | 2026 metric | dev delta | 2026 delta | dev 正年份 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| limitup concept | due5::pool100::industry_leader_confirmed::top10 | `+0.020977` | `+0.012090` | `+0.000883` | `+0.006521` | 2021-2025 | 最强 incremental 苗头 |
| limitup concept | all::pool100::concept_limit_burst::top15 | `+0.018392` | `+0.014556` | `+0.000763` | `+0.001082` | 2021-2025 | 同时厚且增量正 |
| right-tail event | due5::pool200::event_base_35::top10 | `+0.018124` | `+0.012414` | `+0.000529` | `+0.006845` | 2022-2025 | label 层厚，但 formal 已知 2024 负 |
| right-tail event | due5::pool100::event_base_35::top10 | `+0.018004` | `+0.012414` | `+0.000409` | `+0.006845` | 2022-2025 | 同上 |
| limitup concept | all::pool100::ml_score::top10 | `+0.022298` | `+0.018237` | `0.000000` | `0.000000` | 2021-2025 | ML 本身已很厚，说明土壤有效 |
| limitup concept | all::pool100::concept_limit_burst::top10 | `+0.022676` | `+0.017399` | `+0.000378` | `-0.000838` | 2021-2025 | 厚但前向增量不稳 |

与第 19-25 节相比，这一轮最大的不同是绝对 label 厚度上了一个台阶。之前 DL 结构改动通常只有 `+0.1%` 到 `+0.5%` 的薄 delta；而 limit-up concept 土壤本身的 dev/forward label 可以稳定在 `1.8%` 到 `2.2%` 级别。

### 26.4 账户层已有证据

`right_tail_event_candidate` 已有 formal replay：

| 账户 | 区间 | final multiple | total return | max drawdown | avg positions | avg holding | 年度收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| right-tail event base35 | 2022-2025 | `3.230466x` | `+2.230466` | `-0.400315` | `17.86` | `7.65` | 2022 `+35.18%`，2023 `+36.18%`，2024 `-6.38%`，2025 `+85.54%` |
| right-tail event base35 | 2026 val63 | `1.126261x` | `+0.126261` | `-0.193071` | `14.98` | `8.22` | 2026 `+12.63%` |

这个方向有收益，但 2024 负收益、回撤太深，不满足“每年正收益”。

`limitup_concept_propagation` 已有 backtest：

| 账户 | 区间 | final multiple | total return | max drawdown | avg holding | 年度收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| concept leader pool100 top15 | 2021-2025 | `5.048766x` | `+4.048766` | `-0.088383` | `9.14` | 2021 `+26.37%`，2022 `+39.60%`，2023 `+22.72%`，2024 `+66.92%`，2025 `+37.15%` |
| concept leader pool100 top15 | 2026 val63 | `1.058602x` | `+0.058602` | `-0.071135` | `8.61` | 2026 `+5.86%` |
| concept leader riskoff | 2021-2025 | `12.595370x` | `+11.595370` | `-0.267906` | `9.19` | 2021 `+76.56%`，2022 `+89.11%`，2023 `+44.79%`，2024 `+62.48%`，2025 `+58.72%` |
| concept leader riskoff | 2026 val63 | `0.984179x` | `-0.015821` | `-0.226914` | `9.84` | 2026 `-1.58%` |
| industry leader pool200 top15 | 2021-2025 | `3.389344x` | `+2.389344` | `-0.099346` | `8.95` | 2021 `+30.72%`，2022 `+21.09%`，2023 `+18.88%`，2024 `+37.74%`，2025 `+28.49%` |
| industry leader riskoff | 2021-2025 | `8.102317x` | `+7.102317` | `-0.253767` | `8.99` | 2021 `+76.71%`，2022 `+54.82%`，2023 `+36.58%`，2024 `+36.35%`，2025 `+57.51%` |
| industry leader riskoff | 2026 val63 | `0.996745x` | `-0.003255` | `-0.215368` | `9.76` | 2026 `-0.33%` |

这里出现了很有价值的结构：

1. 非 riskoff 版本 2021-2025 五年自然年全正，2026 也正，回撤浅，但收益只有约 `5.05x`。
2. riskoff 版本 2021-2025 大幅放大到 `8.10x` 到 `12.60x`，但 2026 转负且回撤很深。
3. 平均持仓约 9 天，和用户要求的持仓周期贴近。

### 26.5 反事实分析

第一反事实：如果第 19-25 节失败只是模型结构问题，那么第 26 节换到事件/资金流代理后不应出现数量级更厚的 label。实际 limit-up concept 的 label 土壤明显更厚，说明真正瓶颈可能是数据和市场状态定义，而不是 GRU/Transformer 架构本身。

第二反事实：如果 riskoff 放大版是鲁棒策略，2026 不应该转负。实际 concept riskoff 2026 `-1.58%`，industry riskoff 2026 `-0.33%`，并伴随 `-21%` 以上回撤。这个反事实说明 dev 高倍数可能来自对 2021-2025 风格的过度暴露。

第三反事实：如果非 riskoff 版本收益太薄是因为持仓周期不合适，那么它的 2026 应该不稳。实际非 riskoff 2026 仍为正且回撤较浅，说明它有稳定性，只是收益厚度不够。

第四反事实：如果 right-tail event 已经解决了问题，formal replay 不应出现 2024 负收益。实际 2024 `-6.38%` 且总回撤 `-40.03%`，说明该方向虽厚，但年度鲁棒性不足。

第五反事实：如果要满足几十倍目标，单纯在非 riskoff 稳定版本上加仓或调仓可能会直接放大收益。riskoff 的 2026 崩坏反驳了这个简单路径：收益放大必须由模型识别“何时进入高 beta 事件传播模式”，不能静态开关。

### 26.6 判定

`candidate_soil_found_but_not_strategy_yet`。

本轮没有找到可融入策略，但找到了比前 25 轮更接近目标的土壤：**涨停/概念/行业事件传播**。它满足几个关键条件：label 层足够厚、持仓周期贴近一周、非 riskoff 版本年度全正、2026 前向为正。缺口在于收益还只有 `5x` 左右；静态 riskoff 能放大到 `8-12x`，但前向不稳。

下一轮方向切换为 `limitup_event_world_model_router_v1`：

1. 不再训练普通 Top200 日频全市场 ranker，而是围绕 limit-up concept/industry event propagation 建模。
2. 目标不是简单提高 label5，而是训练轻量 world model/router：在非 riskoff 稳定模式和 riskoff 高收益模式之间做动态切换。
3. 输入应包含事件强度、领导股持续性、概念/行业扩散宽度、近期失败率、市场宽度、涨停数量、候选池拥挤度、前一阶段回撤/收益状态。
4. gate：dev 要保持每年正收益，并让 2021-2025 从 `5.05x` 明显提高；2026 val63 必须保持正收益，不能复现 riskoff 静态开关的前向转负。
5. 仍只在 `/tmp` 做实验，除非达到用户要求，否则不融入正式策略。

## 27. limitup_event_world_model_router_v1

### 27.1 假设

第 26 节最重要的观察不是“riskoff 版本收益更高”，而是同一个涨停/概念传播土壤里同时存在两种模式：

1. 非 riskoff 稳定模式：2021-2025 每年正收益，2026 forward 也正，但五年只有 `5x` 左右。
2. risk-filter-off 高 beta 模式：2021-2025 可以放大到 `8x-12x`，但 2026 forward 转负并出现 `-21%` 以上回撤。

这里的 `riskoff` 命名容易误导。复核 `explain.json` 后确认，它不是防守，而是把原来的市场风险过滤基本关闭：`risk_ret20_min=-999`、`risk_ret60_min=-999`、`risk_breadth20_min=0`。因此本轮假设是：如果高 beta 事件传播不是纯过拟合，那么用因果状态 router 只在少数合适阶段打开 aggressive 模式，应当能在 dev 提高收益，同时不破坏 2026 前向。

这轮不是最终策略回放，而是曲线级 soil test。它只用已有 stable/aggressive 两条账户曲线做状态切换，用来判断“动态 router 是否值得下沉到 PIT 特征级模型”。

### 27.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_limitup_event_router_v1.py
sha256:00194e0e4784c6e2256715fd4347a4761cd31aa0c8cca91dc9bcca8a6677582e

.tmp/quantx-research/deep-learning-alpha-search-v1/limitup_event_world_model_router_v1_summary.json
sha256:bb07611192607fa8de58576725458b22e8fcdf10550eb2f0b429759e432f6cf6
```

输入曲线：

| pair | stable dev | aggressive dev | stable 2026 | aggressive 2026 |
| --- | --- | --- | --- | --- |
| concept leader | `concept_leader_pool100_top15_mlonly_dev_2021_2025` | `concept_leader_pool100_top15_mlonly_dev_2021_2025_riskoff` | `concept_leader_pool100_top15_mlonly_val63_2026` | `concept_leader_pool100_top15_mlonly_val63_2026_riskoff` |
| industry leader | `industry_leader_pool200_top15_mlonly_dev_2021_2025` | `industry_leader_pool200_top15_mlonly_dev_2021_2025_riskoff` | `industry_leader_pool200_top15_mlonly_val63_2026` | `industry_leader_pool200_top15_mlonly_val63_2026_riskoff` |

防未来函数口径：

| 项 | 约束 |
| --- | --- |
| 日内选择 | 第 `t` 日模式只使用截至 `t-1` 的两条曲线收益、回撤、波动状态 |
| 参数选择 | 2021-2023 train 搜索，2024-2025 validation 选择；2026 val63 仅前向验证 |
| 可融入性 | 曲线级 router 不能直接融入；只有后续用 PIT 特征重放确认后才可能进入正式策略 |

router 规则很简单：当 aggressive 相对 stable 的过去窗口超额、stable 自身动量、aggressive 回撤、aggressive 波动、近期连亏状态同时满足阈值时，下一交易日切到 aggressive，否则使用 stable。它不是为了“硬卡条件”，而是验证高 beta 事件传播是否存在可观测状态边界。

### 27.3 结果

`concept_leader_pool100_top15`：

| 账户 | 2021-2025 multiple | max DD | aggressive mode ratio | avg position proxy | 年度收益 | 2026 return | 2026 max DD |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| stable | `5.048766x` | `-8.84%` | `0.00%` | `7.66` | 2021 `+26.37%`，2022 `+42.25%`，2023 `+22.67%`，2024 `+66.96%`，2025 `+37.13%` | `+5.86%` | `-7.11%` |
| aggressive | `12.595370x` | `-26.79%` | `100.00%` | `14.30` | 2021 `+76.56%`，2022 `+92.70%`，2023 `+45.50%`，2024 `+62.41%`，2025 `+56.66%` | `-1.58%` | `-22.69%` |
| router | `6.884312x` | `-10.61%` | `12.95%` | `8.53` | 2021 `+40.27%`，2022 `+68.29%`，2023 `+22.71%`，2024 `+66.02%`，2025 `+43.16%` | `+8.95%` | `-5.89%` |

选中参数：`lookback=20`、`rel_threshold=0.05`、`stable_mom_min=0.0`、`aggressive_dd_floor=-0.25`、`aggressive_vol_max=0.60`、`loss_streak_cut=3`、`cooldown_days=0`。

`industry_leader_pool200_top15`：

| 账户 | 2021-2025 multiple | max DD | aggressive mode ratio | avg position proxy | 年度收益 | 2026 return | 2026 max DD |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| stable | `3.389344x` | `-9.93%` | `0.00%` | `7.64` | 2021 `+30.72%`，2022 `+23.26%`，2023 `+18.84%`，2024 `+37.78%`，2025 `+28.47%` | `+5.55%` | `-7.59%` |
| aggressive | `8.102317x` | `-25.38%` | `100.00%` | `14.26` | 2021 `+76.71%`，2022 `+57.59%`，2023 `+36.18%`，2024 `+37.00%`，2025 `+55.96%` | `-0.33%` | `-21.54%` |
| router | `4.553039x` | `-10.43%` | `55.94%` | `10.30` | 2021 `+68.65%`，2022 `+26.94%`，2023 `+14.79%`，2024 `+36.83%`，2025 `+35.41%` | `+11.63%` | `-7.88%` |

选中参数：`lookback=40`、`rel_threshold=0.0`、`stable_mom_min=0.0`、`aggressive_dd_floor=-0.25`、`aggressive_vol_max=0.35`、`loss_streak_cut=999`、`cooldown_days=0`。

### 27.4 观察

第一，曲线级 router 并没有逼近用户要求的 `几十倍-100倍`。concept 从 `5.05x` 提到 `6.88x`，industry 从 `3.39x` 提到 `4.55x`，仍然差一个数量级以上。因此本轮不能作为策略候选。

第二，它确实修复了第 26 节最刺眼的问题：静态 aggressive 2026 转负，而 router 2026 为正。concept 2026 从 stable `+5.86%` 提到 `+8.95%`，且 max DD 从 `-7.11%` 改善到 `-5.89%`；industry 2026 从 `+5.55%` 提到 `+11.63%`，max DD 基本持平。

第三，concept router 只在 `12.95%` 的 dev 交易日打开 aggressive，却吃到一部分高 beta 收益。这说明 aggressive 模式的收益不是完全均匀分布，存在可被过去状态粗略捕捉的窗口。

第四，industry router 打开 aggressive 的比例高达 `55.94%`，但收益提升仍然有限。这说明行业传播的高 beta 窗口更宽，却没有 concept 传播那样强的边际收益密度。后续更应该优先建模 concept/event token，而不是只用行业层状态。

第五，曲线级 router 的输入包含“已有策略曲线表现”，这在研究上可以用于土壤判断，但不是 PIT 个股特征。真实可交易版本必须把它还原成当日可见的事件强度、扩散宽度、龙头承接、涨停/炸板/回封、市场宽度和候选池拥挤度，不能直接用两条回测曲线做生产开关。

### 27.5 反事实分析

第一反事实：如果第 26 节 aggressive 的高收益只是 2021-2025 过拟合，那么用 2021-2023/2024-2025 选择出来的简单状态规则，在 2026 应该仍然复现 aggressive 的负收益。实际两个 pair 的 2026 都转正，并且 concept 的回撤还低于 stable。这反驳了“完全过拟合”的解释，但不能证明可融入。

第二反事实：如果收益改善只是靠降低尾部权重或纯防守，router 的 aggressive mode ratio 应该很低且 dev 收益接近 stable。实际 concept 只开 `12.95%` aggressive 就把五年收益从 `5.05x` 提到 `6.88x`，industry 开 `55.94%` 也提高到 `4.55x`。这更像是对高 beta 事件窗口的选择，而不是单纯降风险。

第三反事实：如果市场状态边界可以被资金曲线代理捕捉，那么 feature-level 的事件状态也应当能近似替代曲线状态。这个反事实尚未被验证。当前最关键的下一步，就是训练一个只使用 PIT 事件/横截面特征的轻量 router，预测 aggressive 模式是否值得打开。

第四反事实：如果 concept/industry 两个传播土壤等价，router 应该给出相似开关比例和收益提升。实际 concept 的少量 aggressive 开关更有效，industry 需要更高开关比例却收益较薄。说明“概念传播”比“行业传播”更像右尾 alpha 的载体。

第五反事实：如果下一轮只把这个 router 变成一个更复杂的 Transformer，却没有补充 PIT 事件 token，那么收益不太可能跃迁到几十倍。当前曲线级上限只有 `6.88x`，模型复杂度本身不是主要缺口；缺口在于需要更直接地刻画事件早期传播和承接失败。

### 27.6 判定

`router_candidate_needs_feature_level_replay`。

本轮没有找到可融入策略，但显著提高了对下一轮方向的置信度：**涨停/概念事件传播 + 状态 router** 值得继续。它不是最终答案，因为收益还远低于目标，且当前 router 仍是曲线级代理；但它证明了 aggressive 模式并非只能静态全开或全关，存在可由过去状态粗筛的高收益窗口。

下一轮切换到 `pit_event_feature_router_transformer_v1`：

1. 不再使用资金曲线状态作为输入，而是构造 PIT 事件 token：概念涨停数量、近涨停数量、炸板/回封代理、龙头相对强度、概念扩散宽度、行业同步度、候选池拥挤度、过去 5/10/20 日事件延续。
2. 模型优先用轻量 Set Transformer / Cross-sectional Transformer / TCN-router，而不是普通全市场大 Transformer。核心是学习“是否打开 high-beta event propagation mode”。
3. label 不只预测个股收益，还要预测模式收益差：`aggressive_return - stable_return` 的下一段是否为正，以及该状态下的左尾风险。
4. 切分保持 train 2021-2023、validation 2024-2025、forward 2026 val63；严禁用 2026 调参。
5. gate：至少要超过 concept router 的 `6.88x`，2026 forward 为正，平均持仓大于 5，并保持 2021-2025 每年正收益；否则继续更换方向。

## 28. pit_event_feature_router_transformer_v1

### 28.1 假设

第 27 节证明了一个很有价值但不可直接融入的事实：用 stable/aggressive 两条资金曲线本身做状态切换，可以把静态 aggressive 的 2026 转负问题部分修复。但这个 router 使用了回测曲线状态，本质上仍是研究代理，不是可交易的 PIT 特征。

本轮把第 27 节下沉一步：不再使用资金曲线作为输入，而是只使用 signal date 当日可见的事件/横截面特征，训练轻量时序 router 去判断下一交易日是否应该打开 high-beta event propagation mode。

本轮名称保留 `transformer`，但实际先用 `TinyTemporalRouter` 的轻量 temporal MLP 作为最小基线。原因是有效日样本只有一千多条，如果 MLP 都无法在 2024-2025/2026 稳定迁移，直接上 Set Transformer 更可能是在制造自由度，而不是发现结构。

### 28.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_pit_event_feature_router_transformer_v1.py
sha256:bf298f77e19c605c33188fade61ea1e1d45baa552d6215ed5e762d1bc48971ec

.tmp/quantx-research/deep-learning-alpha-search-v1/pit_event_feature_router_transformer_v1_summary.json
sha256:b988e2ca71c0e9daf114fd4c2f9785032aae85cad4fd494de47413f5955812e1
```

输入沿用第 26-27 节的涨停/概念传播实验：

| pair | variant | pool | topk | 目标 |
| --- | --- | ---: | ---: | --- |
| concept leader | `concept_leader_confirmed` | 100 | 15 | 判断概念龙头 high-beta 模式是否值得打开 |
| industry leader | `industry_leader_confirmed` | 200 | 15 | 判断行业龙头 high-beta 模式是否值得打开 |

PIT 特征来自 signal date 当日候选池，主要包括：

| feature family | 内容 |
| --- | --- |
| base rank | 原 ML score 在候选池内的 rank/分布 |
| stock state | ret1/ret3/ret5/ret20 rank、volume ratio20 rank、距 20/60 日高点 rank |
| concept event | concept ret1/ret3/ret5 rank、limit ratio rank、near limit ratio rank、strong ratio5 rank、相对概念 ret5 rank |
| industry event | industry ret1/ret3/ret5 rank、limit ratio rank、near limit ratio rank、strong ratio5 rank、相对行业 ret5 rank |
| aggregation | selected top15 与 pool 全体分别取 mean/std/q20/q80/max，再拼接 20 日序列 |

防未来函数口径：

| 项 | 约束 |
| --- | --- |
| 特征 | 只使用 signal date 当日 close/volume 派生事件状态与 base prediction 候选池 |
| label | 使用下一交易日 aligned daily_nav 的 `aggressive_return - stable_return` |
| 切分 | 按收益发生日切分：2021-2023 train，2024-2025 validation，2026 val63 forward |
| 标准化 | mean/std 只在 2021-2023 train 拟合 |
| 阈值 | threshold 只在 2024-2025 validation 选择，2026 不参与调参 |
| warmup | 20 日序列不足的前段回放时使用 stable fallback，避免样本截断导致 2026 口径失真 |

### 28.3 结果

`concept_leader_pool100_top15`：

| 口径 | 2021-2025 multiple | max DD | aggressive mode ratio | avg position proxy | 年度收益 | 2026 return | 2026 max DD |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| full stable | `5.048766x` | `-8.84%` | `0.00%` | `7.66` | 2021 `+26.37%`，2022 `+42.25%`，2023 `+22.67%`，2024 `+66.96%`，2025 `+37.13%` | `+5.86%` | `-7.11%` |
| feature router full fallback | `23.133140x` | `-11.98%` | `30.28%` | `10.62` | 2021 `+60.42%`，2022 `+162.71%`，2023 `+88.50%`，2024 `+109.00%`，2025 `+39.33%` | `+0.06%` | `-14.73%` |

分类/预测指标：

| split | AUC | corr(diff) | mean diff | positive rate | prob mean | prob std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.823195` | `+0.484630` | `+0.001112` | `57.14%` | `0.544705` | `0.053670` |
| valid | `0.501434` | `+0.049389` | `+0.000320` | `58.35%` | `0.549100` | `0.046962` |
| 2026 | `0.474330` | `-0.160983` | `-0.000605` | `53.85%` | `0.515613` | `0.037779` |

threshold 为 `0.56`。dev 表现非常亮，但 2026 只剩 `+0.06%`，明显弱于 stable 的 `+5.86%`，并且 2026 AUC 低于 0.5、收益差相关性为负。因此 concept feature router 判定为 `weak_feature_router_candidate`，不能进入正式 replay。

`industry_leader_pool200_top15`：

| 口径 | 2021-2025 multiple | max DD | aggressive mode ratio | avg position proxy | 年度收益 | 2026 return | 2026 max DD |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| full stable | `3.389344x` | `-9.93%` | `0.00%` | `7.64` | 2021 `+30.72%`，2022 `+23.26%`，2023 `+18.84%`，2024 `+37.78%`，2025 `+28.47%` | `+5.55%` | `-7.59%` |
| feature router full fallback | `6.599362x` | `-9.93%` | `8.50%` | `8.76` | 2021 `+51.81%`，2022 `+65.74%`，2023 `+18.84%`，2024 `+71.79%`，2025 `+28.47%` | `+7.87%` | `-8.93%` |

分类/预测指标：

| split | AUC | corr(diff) | mean diff | positive rate | prob mean | prob std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.729914` | `+0.342375` | `+0.000946` | `54.74%` | `0.528408` | `0.061042` |
| valid | `0.486268` | `+0.042028` | `+0.000482` | `56.70%` | `0.557834` | `0.060073` |
| 2026 | `0.470015` | `-0.256671` | `-0.000459` | `55.77%` | `0.543483` | `0.049963` |

threshold 为 `0.65`。industry 版本在完整口径上从 `3.39x` 提高到 `6.60x`，2026 从 `+5.55%` 提高到 `+7.87%`，平均持仓 proxy 也从 `7.64` 提高到 `8.76`。但 forward AUC 只有 `0.470015`，收益差相关性为 `-0.256671`，说明它不是一个稳定分类器。更可能的解释是：高阈值只打开了少量历史上相似的强事件窗口，账户曲线碰巧改善，但模型对 2026 的逐日 aggressive/stable 优劣排序并不可靠。

### 28.4 观察

第一，PIT 日级事件特征能够在 dev 上产生很强放大。concept feature router 的 2021-2025 达到 `23.13x`，industry 达到 `6.60x`。这说明“事件状态 + 高 beta 模式选择”的方向不是空的，至少在训练/验证年代存在明显可学习结构。

第二，concept 的 dev 高倍数不能信。它的 2026 完整口径只有 `+0.06%`，低于 stable `+5.86%`，且 2026 AUC/相关性均为负。这是典型的 dev 学到强 regime，但 forward 失效。

第三，industry 的账户层结果比 concept 更稳，但预测层指标更尴尬。账户上 2026 从 `+5.55%` 提到 `+7.87%`，但 AUC 低于 0.5。换句话说，它可能不是在逐日预测 aggressive 是否更好，而是在少数阈值区域做了近似 regime filter。

第四，本轮没有使用 2026 调阈值；threshold 只来自 2024-2025 validation。industry 的 2026 改善因此有一定前向价值，但证据仍不足以融入，因为 forward 样本只有 104 个可打分日，且分类方向反向。

第五，本轮再次说明“换更复杂模型”不是最主要矛盾。若日频 PIT 特征中真的有稳定可迁移的 aggressive/stable 边界，valid/2026 AUC 不应持续落在 0.5 附近或以下。因此下一步不能直接上大 Transformer，而要先做消融和正式 replay，确认账户改善到底来自哪里。

### 28.5 反事实分析

第一反事实：如果 feature router 真正学到了可迁移状态边界，2026 AUC 应该显著高于 0.5，至少不应为负相关。实际 concept AUC `0.474`、industry AUC `0.470`，收益差相关性也为负。这强烈反驳“模型已学会开关”的解释。

第二反事实：如果 dev 高倍数来自稳定右尾窗口，concept 2026 不应该从 stable `+5.86%` 降到 `+0.06%`。实际 concept 失效，说明 concept 的日频事件状态更容易过拟合 2021-2025 的风格。

第三反事实：如果 industry 的 2026 改善只是因为更防守，它的 aggressive mode ratio 应该接近 0。实际 industry full fallback aggressive mode ratio 为 `8.50%`，并非完全防守；它确实在少数窗口打开了高 beta 模式。但这仍不足以证明模型可迁移，因为逐日排序指标不好。

第四反事实：如果问题只在模型容量不足，train AUC 不应远高于 valid/forward。实际 train AUC 很高，valid/forward 接近随机或更差，说明主要问题是泛化，而不是容量不足。

第五反事实：如果当前日频 PIT 特征已经足够表达涨停/封单/炸板/回封质量，那么 2026 不应出现负相关。实际负相关提示缺失了更接近交易行为的数据，例如分钟级封单强度、炸板后回封速度、竞价承接、盘口拥挤、开板次数和龙虎榜/大单资金。

### 28.6 判定

`not_integrated_feature_router_evidence_mixed`。

本轮没有找到可融入策略。industry feature router 是一个值得继续复核的候选苗头，但不是通过项：它 dev 从 `3.39x` 到 `6.60x`、2026 从 `+5.55%` 到 `+7.87%`，看起来有收益改善；然而 2026 AUC 和收益差相关性为负，说明预测能力证据不足。

下一轮切换到 `industry_router_replay_and_ablation_v1`：

1. 对 industry feature router 做消融：只用简单阈值/只用 event breadth/只用 volatility/随机同频率开关，判断 `6.60x` 是否来自模型真实预测，还是来自开关频率/阈值偶然性。
2. 如果消融仍支持模型，将 industry router 写成临时 PredictionStore 或模式信号，在 `/tmp` 里做正式 overlap account replay，检查交易成本、持仓、调仓和资金占用后是否还能保留 `6.60x` 量级。
3. 如果消融否定模型，则转向 `intraday_limit_order_event_world_model_v1`，寻找更接近涨停封单、炸板、回封、竞价承接的分钟/盘口代理数据。
4. 下一轮 gate：dev 必须高于 stable 且 2026 高于 stable；同时必须证明不是随机同频率开关或单一防守阈值导致，否则继续换方向。

## 29. industry_router_replay_and_ablation_v1

### 29.1 假设

第 28 节 industry feature router 看起来有一点前向价值：完整口径下 2021-2025 从 `3.39x` 提高到 `6.60x`，2026 从 `+5.55%` 提高到 `+7.87%`。但它同时有一个危险信号：2026 AUC 只有 `0.470015`，收益差相关性为 `-0.256671`。

因此本轮不推进正式策略，而是先做反证消融：如果模型只是因为“少量打开 aggressive”而碰巧改善，那么同频率随机开关或概率打乱也应该能接近它；只有当模型在 dev 和 2026 都显著超过同频率随机，才值得继续做 PredictionStore/formal replay。

### 29.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_industry_router_ablation_v1.py
sha256:20137f5bd113a56bbe4f20ba2fa206e34039046fbf28b94f996aeb166e2b4536

.tmp/quantx-research/deep-learning-alpha-search-v1/industry_router_replay_and_ablation_v1_summary.json
sha256:b58c68412380d1ed98fbac7c8923072d86d1d38f3fe9797c6dbd942725668110
```

输入为第 28 节额外落出的逐日决策 artifact：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/pit_event_feature_router_transformer_v1_decisions/industry_leader_pool200_top15_train_decisions.json
sha256:70cd1931676f657288156160edcea6ef0892fbad43f05b797f546545a58bd400

.tmp/quantx-research/deep-learning-alpha-search-v1/pit_event_feature_router_transformer_v1_decisions/industry_leader_pool200_top15_valid_decisions.json
sha256:c78646eebdfd0040de65b24f7fad07f8503996d2fc2885c1c9c45f639a99f1da

.tmp/quantx-research/deep-learning-alpha-search-v1/pit_event_feature_router_transformer_v1_decisions/industry_leader_pool200_top15_forward_decisions.json
sha256:0709884d2e47dd7e9b4704cbfb02f4bc567240c3709fa0e68f382e35f59aa0e6
```

消融口径：

| 对照 | 含义 |
| --- | --- |
| model | 第 28 节模型概率超过 threshold 的日期打开 aggressive |
| random same-k | 在可打分日期里随机选择相同数量日期打开 aggressive，跑 2000 次 |
| probability shuffle | 保留概率分布和 threshold，但打乱概率与日期的对应关系，跑 2000 次 |
| oracle same-k | 同样开关数量下，事后选择收益差最高日期，作为上界 |
| anti-oracle same-k | 同样开关数量下，事后选择收益差最低日期，作为下界 |

### 29.3 结果

dev 2021-2025：

| 口径 | multiple | total return | max DD | aggressive ratio | avg position proxy | 年度收益 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable | `3.389344x` | `+2.389344` | `-9.93%` | `0.00%` | `7.64` | 2021 `+30.72%`，2022 `+23.26%`，2023 `+18.84%`，2024 `+37.78%`，2025 `+28.47%` |
| aggressive | `8.102317x` | `+7.102317` | `-25.38%` | `100.00%` | `14.26` | 2021 `+76.71%`，2022 `+57.59%`，2023 `+36.18%`，2024 `+37.00%`，2025 `+55.96%` |
| model | `6.599362x` | `+5.599362` | `-9.93%` | `8.50%` | `8.76` | 2021 `+51.81%`，2022 `+65.74%`，2023 `+18.84%`，2024 `+71.79%`，2025 `+28.47%` |
| oracle same-k | `37.755144x` | `+36.755144` | `-9.93%` | `8.50%` | `8.79` | 2021 `+70.92%`，2022 `+155.03%`，2023 `+37.43%`，2024 `+247.35%`，2025 `+81.43%` |
| anti-oracle same-k | `0.317124x` | `-68.29%` | `-80.69%` | `8.50%` | `8.79` | 2021 `+13.93%`，2022 `-35.12%`，2023 `-4.46%`，2024 `-55.26%`，2025 `+0.36%` |

dev 上 model 的 final multiple `6.60x` 高于 2000 次同频率随机的最大值 `5.55x`，也高于概率打乱的最大值 `5.32x`。这说明第 28 节 dev 改善不是纯随机开关。

validation 2024-2025：

| 口径 | multiple | total return | max DD | aggressive ratio | 年度收益 |
| --- | ---: | ---: | ---: | ---: | --- |
| stable | `1.770026x` | `+77.00%` | `-9.93%` | `0.00%` | 2024 `+37.78%`，2025 `+28.47%` |
| aggressive | `2.136606x` | `+113.66%` | `-25.38%` | `100.00%` | 2024 `+37.00%`，2025 `+55.96%` |
| model | `2.207006x` | `+120.70%` | `-9.93%` | `12.16%` | 2024 `+71.79%`，2025 `+28.47%` |
| random same-k p95 | `2.124462x` | - | - | `12.16%` | - |
| probability shuffle p95 | `2.116136x` | - | - | - | - |

validation 上 model 只超过随机 p95，没有超过随机最大值。它有一定结构，但不够厚。

forward 2026：

| 口径 | multiple | total return | max DD | aggressive ratio | avg position proxy | 年度收益 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable | `1.055524x` | `+5.55%` | `-7.59%` | `0.00%` | `5.83` | 2026 `+5.55%` |
| aggressive | `0.996745x` | `-0.33%` | `-21.54%` | `100.00%` | `14.66` | 2026 `-0.33%` |
| model | `1.078679x` | `+7.87%` | `-8.93%` | `5.65%` | `6.64` | 2026 `+7.87%` |
| oracle same-k | `1.240100x` | `+24.01%` | `-2.92%` | `5.65%` | `6.62` | 2026 `+24.01%` |
| anti-oracle same-k | `0.833024x` | `-16.70%` | `-26.87%` | `5.65%` | `6.62` | 2026 `-16.70%` |

2026 上 model 的 `1.078679x` 没有超过同频率随机 p95 `1.109422x`，也没有超过概率打乱 p95 `1.111020x`。这意味着第 28 节看到的 `+7.87%` 前向改善，不足以证明模型有可迁移预测能力。

逐日选择质量：

| split | selected days | selected ratio | selected mean diff | unselected mean diff | selected positive ratio | unselected positive ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 103 / 1192 | `8.64%` | `+0.006660` | `+0.000199` | `73.79%` | `53.81%` |
| valid | 59 / 485 | `12.16%` | `+0.003946` | `+0.000003` | `67.80%` | `55.16%` |
| 2026 | 7 / 104 | `6.73%` | `+0.003150` | `-0.000719` | `85.71%` | `53.61%` |

这个表面上支持模型选择质量，但随机同频率对照推翻了强结论：2026 只有 7 个打开日期，样本太小，同频率随机的右尾足以覆盖模型结果。

### 29.4 观察

第一，dev 上模型确实学到了一些结构。`6.60x` 超过随机最大值，selected mean diff 明显高于 unselected。这说明第 28 节不是完全幻觉。

第二，结构没有足够前向厚度。2026 model 只选 7 天 aggressive，收益虽然高于 stable，但没超过随机 p95。也就是说，如果随机选 7 个可打分日打开 aggressive，前 5% 的随机样本会比模型更好。

第三，oracle same-k 上界很高。dev 同样只开 `8.50%` aggressive，oracle 可以到 `37.76x`；2026 只开 `5.65%` aggressive，oracle 可以到 `+24.01%`。这说明市场里确实存在高价值窗口，但当前日频 PIT 模型没有稳定抓住。

第四，anti-oracle 下界极差。dev 同频率反向选择会跌到 `0.317x`，2026 会跌到 `-16.70%`。这说明 aggressive 开关不是无害的，错误窗口会很伤。因此不能因为平均持仓和收益看起来改善，就贸然融入。

第五，本轮对第 28 节的正确解释应当收敛为：模型在 2021-2025 找到了部分事件窗口，但迁移到 2026 后证据不足；当前日频事件特征不足以支持 high-beta 开关策略。

### 29.5 反事实分析

第一反事实：如果 industry feature router 有稳定前向预测能力，2026 应该超过同频率随机 p95。实际 model `1.078679x` 低于随机 p95 `1.109422x` 和概率打乱 p95 `1.111020x`，因此不能认为它已通过。

第二反事实：如果第 28 节只是纯随机，dev 不应超过随机最大值。实际 dev model `6.60x` 高于 random max `5.55x`，说明训练/验证年代确有结构，但它没有跨 regime 稳定迁移。

第三反事实：如果问题只是 threshold 太保守，2026 oracle same-k 不会显著高于 model。实际 oracle `+24.01%` 远高于 model `+7.87%`，说明不是开关频率不足，而是日期选择不够准。

第四反事实：如果只需要在日频事件特征上换 Transformer，valid/forward 消融应该已经显示出模型显著胜过随机。实际 2026 不过随机 p95，提示应该优先寻找更接近交易行为的输入，而不是先加模型复杂度。

第五反事实：如果高 beta 模式本身无价值，oracle same-k 不会给出高上界。实际上界很高，说明金子可能在“日内/盘口/封单/回封/竞价承接”的更细状态里，而不是当前日频聚合特征里。

### 29.6 判定

`router_rejected_as_not_better_than_random_same_frequency`。

本轮没有找到可融入策略，也不继续推进 industry router formal replay。虽然 dev 有结构，2026 账户也略优于 stable，但它没有超过同频率随机 p95，证据不足。

下一轮方向切换为 `intraday_limit_order_event_world_model_v1`：

1. 优先检查本地是否有分钟级数据、涨停封单/炸板/回封代理、开盘竞价到收盘的路径特征。
2. 如果没有真实盘口数据，则从已有 OHLCV 构造 intraday proxy：开盘跳空、最高触板、收盘封板、长上影炸板、回封强度、成交放量、次日竞价承接。
3. 目标不再是用日频概念强度判断是否打开 aggressive，而是直接建模“事件质量”：封得住、炸而能回、回封后承接强、次日不被核按钮。
4. 训练仍保持 2021-2023 train、2024-2025 validation、2026 forward；gate 必须超过 stable 且超过同频率随机 p95，否则继续换方向。

## 30. intraday_limit_order_event_world_model_v1

### 30.1 假设

第 29 节说明日频 industry router 不能稳定抓住 high-beta 窗口，但 oracle same-k 上界很高，说明真正的金子可能在更细的交易行为状态里：涨停触板、封板质量、炸板后回封、尾盘承接和次日是否可买。

本轮先不训练新模型，而是做 QMT 可得数据范围内的土壤扫描：若本地没有分钟或盘口数据，则用日线 OHLCV/vwap 构造 intraday path proxy。核心问题是：在原 ML pool200 内，加入当日开盘跳空、最高触板、收盘封板、长上影、vwap 支撑、成交放量等 proxy 后，是否能显著提高未来一周 Top10 的收益厚度。

### 30.2 产物和口径

脚本和输出均在 `/tmp`：

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_intraday_limit_order_event_world_model_v1.py
sha256:794e1a1466827b8ea39fcd1cc6a44f9bfbd79508f31ca372002853f85aee994d

.tmp/quantx-research/deep-learning-alpha-search-v1/intraday_limit_order_event_world_model_v1_summary.json
sha256:0f35ac49968b55ef3552e5762949f90bf74efa76715552c51dbd6efdb50e9053

.tmp/quantx-research/deep-learning-alpha-search-v1/replay_intraday_proxy_light_account_v1.py
sha256:e337d1603f2531c9d149882b28aab50f042afae1e382c36b35e8ce353a13aa24

.tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_light_account_v1_summary.json
sha256:d706ff95e14ac916ae7aedc02c25517802c61ec148a04ebafe18bf3de6217f7f
```

数据可用性：本地 Qlib provider 主要有日频 `open/high/low/close/volume/vwap/change/factor`，少量 ETF 有 `amount`；本轮开始时未发现本地分钟或盘口目录。因此本轮使用日线 high/low/vwap proxy，不使用公告、新闻或非 QMT 来源数据。

构造的主要 score 包括：

| score | 含义 |
| --- | --- |
| `ml_score` | 原 ML pool 内 rank |
| `seal_confirmed` | 收盘接近涨停、收盘靠近日高、vwap 支撑、成交放量 |
| `failed_touch_avoid` | 避免触板后大幅回落、长上影和收盘弱势 |
| `low_chase_reseal` | 触板后接近回封，同时不过度追一字高开 |
| `event_quality_blend` | 原 ML rank 与封板/触板/回封/成交质量混合 |

因果边界：特征只使用 T 日收盘后可见 OHLCV/vwap 和 T 日 ML pool；账户轻量复核使用 T 日信号、T+1 open 调仓、5 日 rebalance、Top10 等权、粗略交易成本。该轻量复核只作为研究筛选，不等同 formal 引擎。

### 30.3 Label 土壤结果

label 层出现明显增量土壤，判定为 `incremental_intraday_proxy_soil_found_needs_replay`。

最强候选：

| key | dev label | 2026 label | dev delta vs ML | 2026 delta vs ML | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all::pool200::seal_confirmed::top10` | `+0.024483` | `+0.018886` | `+0.002185` | `+0.000649` | dev 五年均正，绝对收益厚 |
| `all::pool200::failed_touch_avoid::top10` | `+0.024054` | `+0.020302` | `+0.001756` | `+0.002065` | 2026 label 增量更明显 |
| `all::pool200::low_chase_reseal::top10` | `+0.023576` | `+0.018850` | `+0.001278` | `+0.000613` | 也通过增量 gate |

strict gate 数量为 `32`，incremental gate 数量为 `20`。这说明日线 proxy 不是完全幻觉，至少在 label 层能从原 ML pool 中找出更偏事件质量的右尾。

### 30.4 轻量账户结果

将 `seal_confirmed` 和 `failed_touch_avoid` 写成临时 PredictionStore 后，用 T+1 open、rebalance5、Top10 等权轻量账户复核，结果非常强：

| case | final multiple | total return | max DD | avg position count | 年度收益 |
| --- | ---: | ---: | ---: | ---: | --- |
| `seal_confirmed_dev` | `170.980667x` | `+16998.07%` | `-34.54%` | `9.99` | 2021 `+794.94%`，2022 `+230.72%`，2023 `+77.57%`，2024 `+127.94%`，2025 `+42.73%` |
| `seal_confirmed_2026` | `1.190233x` | `+19.02%` | `-21.80%` | `9.92` | 2026 `+19.02%` |
| `failed_touch_avoid_dev` | `119.330108x` | `+11833.01%` | `-31.94%` | `9.99` | 2021 `+694.90%`，2022 `+184.38%`，2023 `+76.78%`，2024 `+147.44%`，2025 `+20.68%` |
| `failed_touch_avoid_2026` | `1.183060x` | `+18.31%` | `-26.48%` | `9.92` | 2026 `+18.31%` |

表面看，`seal_confirmed_dev` 已达到五年百倍以上、五年全正、平均持仓接近 10；但该结果还不能作为策略证据，因为轻量账户没有拒绝 T+1 涨停/极端高开买入，也没有做同池随机反证。

### 30.5 反事实分析

第一反事实：如果日线封板 proxy 真正可交易，加入正式执行器近似规则后不应大幅坍缩。后续第 31 节显示，拒绝 `abs(T+1 open / T close - 1) > 9.5%` 后，`seal_confirmed` dev 从百倍级降到十倍级，2026 转负。这强烈说明第 30 节轻量账户吃进了大量难以成交收益。

第二反事实：如果 `seal_confirmed` 的强收益来自稳健横截面排序，而不是买到最热不可买标的，那么 `failed_touch_avoid` 和 `event_quality_blend` 应在严格口径下同样保持前向优势。实际只有 `event_quality_blend` 在 2026 strict 后为正，其它多数为负，说明日线 proxy 表达力不足。

第三反事实：如果日线 high/low/vwap 已足够刻画封板/炸板/回封质量，前向不应依赖少数高开或涨停日。实际第 31 节显示去掉最强若干调仓后，2026 的 event_quality_blend 很快跌破 1，说明收益来源脆弱。

第四反事实：如果本轮结果只是原 ML pool 本身很强，那么 `ml_score` strict 后应接近或超过事件 proxy。dev 中 `event_quality_blend` 和 `seal_confirmed` 略高于 `ml_score`，说明事件 proxy 有增量；但增量不够大，不足以达到目标。

### 30.6 判定

`strong_lightweight_candidate_needs_tradeability_ablation`。

本轮发现了重要土壤：日线事件质量 proxy 能在 ML pool 内提高右尾，轻量账户甚至出现百倍级曲线。但该结果不能融入，必须先通过交易可实现性、同池随机、收益贡献拆解和 formal 近似压力。下一轮执行 `intraday_proxy_tradeability_and_random_ablation_v1`。

## 31. intraday_proxy_tradeability_and_random_ablation_v1

### 31.1 假设

第 30 节最危险的问题不是收益不够，而是收益可能来自 T+1 极端高开、涨停或停牌缺失导致的不可交易区间。正式执行器会拒绝涨停、极端 price jump、停牌等情况，而轻量账户第一版没有这些约束。

本轮做反证消融：如果日线 proxy 是真实可交易 alpha，那么在加入成交量、price jump 和高开过滤后，dev 仍应接近目标量级，2026 仍应为正，并且至少超过同池随机 p95。

### 31.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_intraday_proxy_tradeability_and_random_ablation_v1.py
sha256:6562eb5732d1fcea8e7e3b755c15e2e6d51a6a24e576acc5c4fceeec92b2a3aa

.tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_tradeability_and_random_ablation_v1_summary.json
sha256:23e694b53a552872f3ddfa42d9d1ecca4bf7c805b77e5915becd10980df4f5c1
```

评估 case：`ml_score`、`seal_confirmed`、`failed_touch_avoid`、`low_chase_reseal`、`event_quality_blend`。交易口径仍为 T 信号、T+1 open 入场、rebalance5、Top10 等权、粗略成本；但新增交易约束：

| constraint | 含义 |
| --- | --- |
| `loose_with_volume` | 有 open 且 volume > 0 才能买 |
| `strict_price_jump` | 额外拒绝 `abs(T+1 open / T close - 1) > 9.5%` |
| `no_high_gap_7` | 额外拒绝 T+1 高开超过 7% |
| `no_high_gap_5` | 额外拒绝 T+1 高开超过 5% |

随机同池对照：从同一 `pool200` 中随机选 Top10，裁判版跑 `300` 次，随机只用于关键 `strict_price_jump` 口径。该随机次数足以判断本轮是否远离随机右尾；若候选接近通过线，再补跑 1000 次。

### 31.3 结果

严格 `strict_price_jump` 口径下，dev 仍有结构，但远低于目标量级：

| case | dev multiple | max DD | avg pos | 年度收益 | 是否超过随机 p95 |
| --- | ---: | ---: | ---: | --- | --- |
| `event_quality_blend` | `12.270222x` | `-34.38%` | `9.12` | 2021 `+101.57%`，2022 `+54.60%`，2023 `+71.29%`，2024 `+34.13%`，2025 `+71.39%` | 是 |
| `seal_confirmed` | `11.716214x` | `-35.74%` | `8.77` | 2021 `+110.73%`，2022 `+51.65%`，2023 `+66.55%`，2024 `+36.45%`，2025 `+61.33%` | 是 |
| `ml_score` | `10.032676x` | `-38.47%` | `9.06` | 2021 `+119.57%`，2022 `+77.72%`，2023 `+37.39%`，2024 `+19.68%`，2025 `+56.36%` | 是 |
| `low_chase_reseal` | `7.436204x` | `-41.06%` | `8.77` | 2021 `+67.47%`，2022 `+41.40%`，2023 `+77.68%`，2024 `+16.38%`，2025 `+51.86%` | 是 |
| `failed_touch_avoid` | `6.829024x` | `-32.89%` | `8.70` | 2021 `+73.28%`，2022 `+40.08%`，2023 `+69.33%`，2024 `+24.03%`，2025 `+33.96%` | 是 |

dev 同池随机 `strict_price_jump`：p50 `4.073333x`，p95 `6.078881x`，p99 `7.116220x`，max `8.512121x`。所以 dev 上事件 proxy 和原 ML 都超过随机右尾，说明不是纯随机；但最高也只有 `12.27x`，离五年几十倍到百倍目标较远。

2026 forward 严格口径：

| case | 2026 multiple | total return | max DD | avg pos | 是否超过随机 p95/p99 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `event_quality_blend` | `1.116956x` | `+11.70%` | `-18.19%` | `9.72` | 超过 p95 和 p99，但低于随机 max | 唯一前向为正 |
| `seal_confirmed` | `0.971272x` | `-2.87%` | `-24.40%` | `8.72` | 否 | 轻量强结果被交易约束打掉 |
| `ml_score` | `0.847420x` | `-15.26%` | `-27.13%` | `8.90` | 否 | 原 ML strict forward 失效 |
| `failed_touch_avoid` | `0.855047x` | `-14.50%` | `-31.63%` | `8.27` | 否 | 失效 |
| `low_chase_reseal` | `0.816267x` | `-18.37%` | `-29.73%` | `8.51` | 否 | 失效 |

2026 同池随机 `strict_price_jump`：p50 `0.926933x`，p95 `1.053356x`，p99 `1.108661x`，max `1.167985x`。`event_quality_blend` 的 `1.116956x` 超过 p99，但仍低于随机 max，且绝对收益只有 `+11.70%`，不满足目标。

收益贡献拆解显示前向收益很脆：`event_quality_blend` 去掉最强 1 次调仓后 multiple 降到 `1.078967x`，去掉最强 3 次降到 `0.958075x`，去掉最强 5 次降到 `0.865384x`，去掉最强 10 次降到 `0.758864x`。这说明 2026 正收益主要来自少数调仓，而不是稳定厚右尾。

### 31.4 观察

第一，日线 proxy 有 dev 结构。`event_quality_blend` strict dev `12.27x` 超过同池随机 max `8.51x`，五年全部为正，平均持仓 `9.12`。这说明第 30 节并非完全假信号。

第二，目标量级没有达到。用户要求五年几十倍到百倍，strict 后最佳仅 `12.27x`，且最大回撤仍约 `-34%`。这不是可以融入的收益厚度。

第三，2026 前向不稳。`seal_confirmed` 从轻量 `+19.02%` 变成 strict `-2.87%`，`failed_touch_avoid` 从 `+18.31%` 变成 `-14.50%`。这证明第 30 节最亮的两条曲线主要依赖高开/涨停附近买入。

第四，`event_quality_blend` 是有价值线索，但不是通过项。它 2026 strict `+11.70%` 且超过随机 p99，说明“事件质量混合”方向可能有迁移信息；但收益集中在少数调仓，去掉最强 3 次就低于 1，不能视为鲁棒 alpha。

第五，QMT 分钟线环境已经打通。第 31 轮期间给 `test` 环境安装并验证了 `xqshare`，QMT 远程连接正常；对 `SH600215` 下载 2026-07-01 到 2026-07-03 后，`1m` 返回 723 行，`5m` 返回 144 行，字段包括 `open/high/low/close/volume/amount/preClose/suspendFlag`。这意味着下一轮可以从真实分钟路径构造触板时间、炸板次数、回封次数、尾盘承接等特征，而不是继续用日线 high/low/vwap 猜。

### 31.5 反事实分析

第一反事实：如果第 30 节 `seal_confirmed` 是可交易核心 alpha，strict 后 dev 不应从 `170.98x` 级别跌到 `11.72x`，2026 不应从 `+19.02%` 变为 `-2.87%`。实际跌幅极大，说明其核心收益来自不可交易或高摩擦买入。

第二反事实：如果日线 OHLCV proxy 已足够表达封单和回封质量，`failed_touch_avoid` 应在 2026 strict 仍为正。实际为 `-14.50%`，说明日线 high/low/vwap 无法判断真实开板次数、封板持续性和尾盘封单承接。

第三反事实：如果 `event_quality_blend` 是稳定前向 alpha，去掉最强 3 次调仓后仍应为正。实际从 `1.116956x` 降到 `0.958075x`，说明它目前更像少数事件窗口，而不是可反复复利的稳定结构。

第四反事实：如果只是随机同池右尾，dev 不应超过随机 max，2026 不应超过随机 p99。实际 `event_quality_blend` 两者都超过，说明方向有金矿味；但它的厚度和稳定性不足，需要更精细的输入，而不是直接融入。

第五反事实：如果下一步只是在日线 proxy 上调权重或阈值，应该已经能在本轮看到稳定通过 gate。实际没有。因此继续硬调日线 proxy 是在卡口径，下一轮必须转向 QMT 分钟 K 线事件路径。

### 31.6 判定

`rejected_tradeability_and_random_ablation`。

本轮没有找到可融入策略。日线 intraday proxy 有真实结构，但严格可交易后远低于目标，且前向收益不稳。`event_quality_blend` 是下一步的重要线索，但不是策略。

下一轮切换到 `qmt_minute_limit_order_event_world_model_v1`：

1. 只用 QMT 可拉取的 `1m/5m` K 线，不用公告、新闻、龙虎榜或外部事件流。
2. 先在 2026 forward 的 ML pool200 候选上做小样本分钟线缓存和土壤扫描，控制下载范围，验证数据速度和字段质量。
3. 构造真实分钟路径特征：首次触板时间、触板后回落深度、炸板次数、回封次数、收盘前封板状态、尾盘承接、分钟 VWAP 支撑、成交量集中度、T+1 开盘可买性。
4. 若分钟路径特征在 strict 可交易账户中超过日线 `event_quality_blend` 且不依赖少数调仓，再扩展到 2021-2025 train/validation，并训练小型 cross-sectional temporal Transformer/attention world model。

## 32. qmt_minute_limit_order_event_world_model_v1

### 32.1 假设

第 31 节说明日线 intraday proxy 有一些结构，但交易约束一上来，2026 前向收益非常脆。最直接的反事实是：如果问题只是日线 proxy 太粗，那么真实 QMT 分钟 K 线应能恢复事件质量排序，至少在 2026 forward 上超过日线事件分数和同池随机 p95。

本轮只做 2026 forward 土壤扫描，不做多年分钟线大规模下载。原因是分钟线数据量明显更大，必须先证明真实分钟路径有增量，才值得扩展到 2021-2025 train/validation。

### 32.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_minute_limit_order_event_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_minute_limit_order_event_world_model_v1_summary.json
sha256:f379176ae00baf905ed05753ea57daa472464ce309ba4b1fd035b5c12ef43320
```

数据口径：

| item | value |
| --- | --- |
| daily provider | `data/qlib_data_fixed` |
| universe | `all_mainboard` |
| QMT period | `5m` |
| date range | 2026-01-01 到 2026-07-10 |
| signal sessions | `25` 个 5 日 rebalance session |
| fetch pool | 每个 session 取日线事件候选前 `60` |
| minute samples | `1500/1500` 可用，`0` empty，`0` error |
| replay | T 日信号，T+1 open 调仓，rebalance5，Top10 等权 |
| strict tradeability | 有 open/volume，拒绝 `abs(T+1 open / T close - 1) > 9.5%` |
| random judge | 同池随机 Top10，`300` 次 |

真实分钟特征包括：首次触板时间、触板次数、近涨停 bar 比例、收盘封板比例、炸板次数、回封次数、午后回封、触板后最大回撤、尾盘 30 分钟强度、尾盘成交占比、分钟 VWAP 支撑、高点回落、冲高速度、成交集中度等。

候选 score：`daily_event_quality`、`minute_seal_quality`、`minute_reseal_quality`、`minute_tail_support`、`minute_break_avoid`、`minute_event_quality_blend`、`minute_tradable_continuation`。

### 32.3 结果

本轮结论很明确：真实分钟路径没有修复事件方向，最佳分钟 score 仍为负，并且低于同池随机 p95。

| case | 2026 multiple | total return | max DD | avg pos | price jump rejects | 去掉最佳 3 次后 multiple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `minute_seal_quality` | `0.872391x` | `-12.76%` | `-29.20%` | `9.30` | `17` | `0.668953x` |
| `daily_event_quality` | `0.807274x` | `-19.27%` | `-38.30%` | `9.55` | `11` | `0.605104x` |
| `minute_tradable_continuation` | `0.795770x` | `-20.42%` | `-25.88%` | `9.55` | `11` | `0.687796x` |
| `minute_tail_support` | `0.697935x` | `-30.21%` | `-40.47%` | `9.79` | `5` | `0.607500x` |
| `minute_break_avoid` | `0.692605x` | `-30.74%` | `-37.68%` | `9.75` | `6` | `0.578349x` |
| `minute_reseal_quality` | `0.660326x` | `-33.97%` | `-38.22%` | `9.92` | `2` | `0.591612x` |
| `minute_event_quality_blend` | `0.597983x` | `-40.20%` | `-44.60%` | `9.83` | `4` | `0.539734x` |

同池随机 Top10：p50 `0.798090x`，p90 `0.969503x`，p95 `1.021956x`，p99 `1.116278x`，max `1.250549x`。最佳 `minute_seal_quality` 只略高于随机中位数，显著低于 p95；除了它超过日线事件质量外，其它 gate 全部失败。

### 32.4 观察

第一，QMT 数据链路没有问题。1500 个股票-日期 5m 样本全部可用，没有 empty 和 error，运行约 `264.11` 秒。这意味着失败不是数据缺失导致，而是方向本身没有前向增量。

第二，真实封板质量在 2026 是负贡献。`minute_seal_quality` 虽然是本轮最好，但仍亏 `-12.76%`；更综合的 `minute_event_quality_blend` 亏到 `-40.20%`。这说明“越像强事件、越像封板/回封/尾盘强承接”的路径，在 2026 这段样本内反而更容易成为拥挤交易。

第三，日线 proxy 的偶然正收益没有被真实分钟线确认。第 31 节 `event_quality_blend` 曾有 `+11.70%`，但本轮用真实分钟线重建事件质量后全部为负。最可能的解释不是分钟线没用，而是第 31 节少数调仓贡献太强，日线 proxy 碰巧抓到几个事件窗口。

第四，随机裁判给了更强的反证：随机 p95 已经是 `1.021956x`，随机 max 是 `1.250549x`，而最佳分钟 score 只有 `0.872391x`。如果强封板路径是稳定 alpha，它不应该输给同池随机这么多。

第五，平均持仓仍大于 9，问题不是持仓数量不足；问题是选股方向错了。继续在“追最强事件质量”上调权重，就是硬卡口径，不符合本轮证据。

### 32.5 反事实分析

第一反事实：如果第 31 节失败只是日线 high/low/vwap 不能表达真实触板路径，那么真实 5m 特征应明显改善 2026。实际最佳仍为 `0.872391x`，说明问题不是 proxy 粗糙，而是事件拥挤本身在前向样本中被惩罚。

第二反事实：如果封板持续性是强 alpha，`minute_seal_quality` 不应只有随机中位数水平。实际它低于随机 p95 约 `14.64%`，说明强一致性封板更像短线资金拥挤，而不是可稳健复利的次周 alpha。

第三反事实：如果炸板/回封包含更高的信息密度，`minute_reseal_quality` 或 `minute_event_quality_blend` 应至少不弱于日线 score。实际它们分别为 `0.660326x` 和 `0.597983x`，比日线更差，说明“更精细地追强事件”会强化错误方向。

第四反事实：如果收益主要受交易拒绝影响，那么 price jump reject 少的 score 应更好。实际 `minute_reseal_quality` 只有 2 个 reject 但收益很差，说明交易过滤不是主因，信号方向本身不对。

第五反事实：如果只需要扩大样本到多年即可恢复，至少 2026 forward 不应给出如此一致的负向排序。现在所有分钟事件 score 都为负，先扩展多年训练有很高概率是在拟合历史事件牛市，而不是解决前向拥挤。

### 32.6 判定

`rejected_minute_limit_order_event_world_model_v1`。

本轮没有找到可融入策略。真实 QMT 5m 事件路径没有增强日线事件方向，反而表明 2026 对强封板、强回封、强尾盘承接有拥挤惩罚。下一轮不再继续追“最强事件质量”，改做 `qmt_minute_anti_crowding_event_filter_v1`：在同一事件候选池里验证反拥挤路径，即强日线事件但分钟上不过早触板、不过度近涨停、不过度一致封板、T+1 不极端跳空的标的，是否才是可交易的次周延续 alpha。若反拥挤只在 2026 有效而 dev 不成立，也直接否决。

## 33. qmt_minute_anti_crowding_event_filter_v1

### 33.1 假设

第 32 节显示“越强封板、越强回封、越强尾盘承接”在 2026 反而亏损。一个合理反事实是：可能不是事件池错了，而是事件池内部应该避开已经完成一致预期的拥挤路径，选择强日线事件但分钟上不过早触板、不过度近涨停、不过度封死、成交不过度集中的标的。

本轮复用第 32 节已缓存的 1500 个 QMT 5m 样本，不重新下载数据，只改排序方向，验证反拥挤是否有土壤。

### 33.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_minute_anti_crowding_event_filter_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_minute_anti_crowding_event_filter_v1_summary.json
sha256:f4e2591a79200159bc4b0a0b2d4bb68f77a7bf879e69382e8b8d4d2570a78e5b
```

数据和账户口径与第 32 节一致：2026-01-01 到 2026-07-10，25 个 signal session，每期日线事件候选前 60，T+1 open 调仓，rebalance5，Top10 等权，拒绝 `abs(T+1 open / T close - 1) > 9.5%`，同池随机 Top10 跑 300 次。

新增 score：

| score | 含义 |
| --- | --- |
| `anti_late_touch` | 强日线事件中，偏晚触板、少触板、近涨停 bar 较少，同时保留 VWAP 和尾盘支撑 |
| `anti_soft_limit` | 避免收盘封死和全天近涨停一致性，寻找更软的事件形态 |
| `anti_smooth_breakout` | 避免开盘快速冲高和成交极端集中，保留平滑突破与承接 |
| `anti_low_concentration` | 低成交集中、低冲高速度、但保留 VWAP/尾盘强度 |
| `anti_gap_safe_proxy` | 偏中等收盘涨幅、低近涨停比例、低冲高速度，尝试降低 T+1 gap 风险 |
| `anti_crowding_composite` | 上述反拥挤维度组合 |
| `reverse_minute_event_quality` | 第 32 节分钟事件质量的反向版本，用于直接反证 |

### 33.3 结果

反拥挤方向没有通过。最佳 `anti_gap_safe_proxy` 仍然亏损，且没有超过日线事件质量，也没有超过随机 p95。

| case | 2026 multiple | total return | max DD | avg pos | price jump rejects | 去掉最佳 3 次后 multiple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anti_gap_safe_proxy` | `0.860396x` | `-13.96%` | `-24.59%` | `9.92` | `2` | `0.733870x` |
| `daily_event_quality` | `0.807274x` | `-19.27%` | `-38.30%` | `9.55` | `11` | `0.605104x` |
| `reverse_minute_event_quality` | `0.781292x` | `-21.87%` | `-38.07%` | `9.50` | `12` | `0.628687x` |
| `anti_soft_limit` | `0.729984x` | `-27.00%` | `-37.70%` | `9.88` | `3` | `0.644991x` |
| `anti_smooth_breakout` | `0.675805x` | `-32.42%` | `-41.39%` | `9.88` | `3` | `0.572143x` |
| `anti_low_concentration` | `0.635403x` | `-36.46%` | `-42.92%` | `9.96` | `1` | `0.569330x` |
| `anti_crowding_composite` | `0.594121x` | `-40.59%` | `-44.57%` | `9.92` | `2` | `0.541636x` |
| `anti_late_touch` | `0.549517x` | `-45.05%` | `-46.91%` | `9.88` | `3` | `0.496811x` |

同池随机与第 32 节一致：p50 `0.798090x`，p90 `0.969503x`，p95 `1.021956x`，p99 `1.116278x`，max `1.250549x`。最佳反拥挤只略高于随机中位数，离 p95 明显不足。

### 33.4 观察

第一，降低拥挤形态确实减少了交易拒绝。`anti_gap_safe_proxy` 只有 2 个 price jump reject，远低于日线事件质量的 11 个。但收益仍为 `-13.96%`，说明前向失败不是简单的 T+1 极端跳空买不进去问题。

第二，反向分钟事件质量没有成立。`reverse_minute_event_quality` 为 `0.781292x`，弱于 `anti_gap_safe_proxy`，也弱于随机中位数附近。这说明第 32 节不是简单“强事件反着买”就可以解决。

第三，越强行做反拥挤组合，结果越差。`anti_crowding_composite` 亏 `-40.59%`，`anti_late_touch` 亏 `-45.05%`。这说明在涨停/强事件候选池里，分钟路径的拥挤度不是稳定可交易的主轴。

第四，平均持仓稳定大于 9，本轮仍不是“持仓数太少导致曲线不稳定”。失败来自候选池和排序方向，而不是持仓约束。

### 33.5 反事实分析

第一反事实：如果第 32 节失败是因为追强封板太拥挤，那么反拥挤 score 应至少转正或超过随机 p95。实际最佳只有 `0.860396x`，说明拥挤不是唯一解释。

第二反事实：如果可交易性是主要矛盾，reject 从 11 降到 2 后收益应显著改善到正。实际仍亏 `-13.96%`，说明候选池本身在 2026 的次周延续很弱。

第三反事实：如果分钟形态包含强预测信息，正向和反向至少应有一边显著跑赢随机。实际两边都没有过 p95，说明在当前强日线事件池内，5m 路径更多是状态描述，不是足够强的次周 alpha。

第四反事实：如果继续在涨停事件池里调 score 能找到金矿，应该已经能在第 32/33 节看到某个方向接近通过线。实际最佳都在 `0.86x-0.87x`，离随机 p95 仍远。继续微调属于卡口径。

### 33.6 判定

`rejected_anti_crowding_event_filter_v1`。

本轮没有找到可融入策略。第 32/33 节合在一起说明：2026 的问题不只是“追强事件太拥挤”，而是涨停/强事件候选池本身不适合作为下一阶段主矿脉。下一轮切换出事件池，回到全市场多周期横截面：先做 `cross_sectional_multihorizon_soil_scan_v1`，用 QMT/Qlib 日线在全主板上扫描 1/3/5/10/20 日 horizon、过去一段时间横截面状态、趋势延续/反转/波动压缩/量价扩散等土壤，先找稳定 label/账户结构，再决定是否训练小型 Transformer/attention world model。

## 34. cross_sectional_multihorizon_soil_scan_v1

### 34.1 假设

第 32/33 节否决了涨停/强事件池。下一步回到更高维的全市场横截面：如果市场短周期可预测，应该先能在 1/3/5/10/20 日历史状态里看到某些稳定土壤，再考虑小型 Transformer 或 world model。

本轮不是训练模型，而是用手工多周期状态做土壤扫描，目的是先判断哪些状态天然有收益厚度。若手工状态都没有基本结构，直接深度学习很可能只是拟合噪声。

### 34.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_multihorizon_soil_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_multihorizon_soil_scan_v1_summary.json
sha256:2367a180cb98c17814577eb337e6940b7d2019d28b1c3e34c727ed95f69f7400
```

口径：`all_mainboard`，每期按 20 日平均成交额取可交易 `pool500`，T 日信号，T+1 open 调仓，rebalance5，Top10 等权，粗略成本，拒绝 `abs(T+1 open / T close - 1) > 9.5%`。dev 为 2021-2025，forward 为 2026-01-01 到 2026-07-10。同池随机 Top10 各跑 300 次。

扫描 score：`smooth_trend`、`trend_acceleration`、`vol_compression_breakout`、`anti_crowded_momentum`、`low_vol_uptrend`、`liquidity_momentum`、`mean_reversion_liquid`、`hybrid_multihorizon_state`。

### 34.3 结果

dev 最佳为 `low_vol_uptrend`，但收益厚度极弱，且 2022 为负，不满足要求。

| case | dev multiple | total return | max DD | avg pos | 年度收益 | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `low_vol_uptrend` | `1.214515x` | `+21.45%` | `-43.37%` | `9.99` | 2021 `+13.39%`，2022 `-34.19%`，2023 `+12.07%`，2024 `+34.20%`，2025 `+8.20%` | `1.136655x` |
| `vol_compression_breakout` | `1.174359x` | `+17.44%` | `-32.78%` | `10.00` | 2021 `-7.34%`，2022 `-23.87%`，2023 `+22.64%`，2024 `+33.67%`，2025 `+1.55%` | `1.111202x` |
| `hybrid_multihorizon_state` | `0.686649x` | `-31.34%` | `-59.99%` | `9.96` | 2022 大亏 `-47.18%` | `0.645068x` |
| `smooth_trend` | `0.045828x` | `-95.42%` | `-96.15%` | `9.69` | 五年全负 | `0.036732x` |
| `trend_acceleration` | `0.022478x` | `-97.75%` | `-98.32%` | `9.73` | 五年全负 | `0.018700x` |
| `liquidity_momentum` | `0.016317x` | `-98.37%` | `-98.79%` | `9.74` | 五年全负 | `0.013483x` |

dev 同池随机：p50 `0.484749x`，p95 `0.877713x`，p99 `1.051386x`，max `1.390429x`。`low_vol_uptrend` 和 `vol_compression_breakout` 超过随机 p99，但绝对收益太薄，且年度稳定性不足。

forward 最佳仍为 `low_vol_uptrend`：

| case | 2026 multiple | total return | max DD | avg pos | 是否超过随机 p95 | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `low_vol_uptrend` | `1.149720x` | `+14.97%` | `-14.50%` | `10.00` | 否 | `0.996576x` |
| `hybrid_multihorizon_state` | `1.104957x` | `+10.50%` | `-20.45%` | `9.88` | 否 | `0.883679x` |
| `smooth_trend` | `1.047982x` | `+4.80%` | `-31.55%` | `9.67` | 否 | `0.772229x` |
| `anti_crowded_momentum` | `0.931059x` | `-6.89%` | `-20.25%` | `9.96` | 否 | `0.782394x` |
| `mean_reversion_liquid` | `0.727948x` | `-27.21%` | `-27.38%` | `10.00` | 否 | `0.686913x` |

forward 同池随机：p50 `0.988664x`，p95 `1.239627x`，p99 `1.333813x`，max `1.401396x`。forward 最佳 `1.149720x` 没过随机 p95，且去掉最佳 3 次后接近 `1` 以下，不能作为稳定 alpha。

### 34.4 观察

第一，`low_vol_uptrend` 有一点真实结构，但太薄。dev 只 `1.21x`，五年不全正，2022 回撤严重；forward 虽正，但没过随机 p95。

第二，强动量/强流动性动量是强负向。`liquidity_momentum` dev 只剩 `0.016317x`，`trend_acceleration` 只剩 `0.022478x`，且五年全负。这不是普通噪声，更像 A 股主板里“热门大票追涨”的系统性负 alpha。

第三，dev 随机本身很差，说明 pool500 在 2021-2025 的随机 Top10 对一周持有非常不友好。手工状态超过随机不难，但要达到几十倍到百倍，需要远强于本轮的排序能力。

第四，2026 与 dev 的结构不完全一致。dev 中 `vol_compression_breakout` label 较好，但 forward 为负；`smooth_trend` dev 极差，forward 却小正。这说明市场状态迁移很强，模型必须显式处理 regime，而不是只学静态 rank。

### 34.5 反事实分析

第一反事实：如果全市场多周期趋势状态已经是主矿脉，`low_vol_uptrend` 应在 dev 接近几十倍并五年全正。实际只有 `1.21x` 且 2022 大亏，说明正向趋势土壤太薄。

第二反事实：如果 forward `+14.97%` 是稳定 alpha，去掉最佳 3 次调仓后仍应显著大于 1，并超过随机 p95。实际为 `0.996576x`，说明 2026 正收益仍依赖少数窗口。

第三反事实：如果强动量只是随机坏样本，不应在 2021-2025 五年全负且接近归零。实际 `liquidity_momentum` 和 `trend_acceleration` 都接近归零，反而提示“反热门/反拥挤横截面”可能是下一轮更值得验证的方向。

第四反事实：如果只需要把这些手工状态丢给 Transformer，手工土壤至少应有稳定正向 label。实际正向土壤弱、负向土壤强，下一轮应先扫描负向/反向结构，而不是马上训练模型。

### 34.6 判定

`rejected_multihorizon_positive_state_scan`。

本轮没有找到可融入策略，也不建议直接基于这些正向状态训练模型。新的线索是：热门大票动量/加速趋势在 dev 中是非常强的负 alpha。下一轮做 `cross_sectional_inverse_crowding_reversal_scan_v1`，系统验证强负向状态的反面：低拥挤、低短期追涨、非热门流动性动量、温和承接的组合，是否在 dev 和 2026 forward 都有稳定收益厚度。

## 35. cross_sectional_inverse_crowding_reversal_scan_v1

### 35.1 假设

第 34 节发现热门大票动量、趋势加速和流动性动量在 dev 中是强负向。一个自然反事实是：如果 A 股主板一周周期真正惩罚热门追涨，那么其反面，也就是低拥挤、低短期追涨、低注意力、温和回撤或恐慌衰竭，应该能提供更厚的 alpha。

本轮沿用第 34 节完全相同的数据、pool、账户和随机裁判，只替换 score。这样可以直接回答“反热门是不是主矿脉”，避免引入新数据口径造成混淆。

### 35.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_inverse_crowding_reversal_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_inverse_crowding_reversal_scan_v1_summary.json
sha256:df2255b0ed0721e28d60f4ee57dd8f867aa99a6822695a25715ffb00c5200da2
```

口径同第 34 节：`all_mainboard`，20 日平均成交额 `pool500`，T 日信号，T+1 open 调仓，rebalance5，Top10 等权，拒绝 `abs(T+1 open / T close - 1) > 9.5%`，dev 为 2021-2025，forward 为 2026-01-01 到 2026-07-10，同池随机 Top10 各 300 次。

扫描 score：`inverse_liquidity_momentum`、`cooldown_reversal_liquid`、`quiet_pullback`、`low_attention_reversal`、`panic_exhaustion`、`anti_hot_low_vol`、`hybrid_inverse_crowding`、`balanced_inverse_state`。

### 35.3 结果

反热门方向没有通过。dev 最好的 `panic_exhaustion` 只有 `1.352128x`，且 2021-2023 连续为负；forward 最好的 `quiet_pullback` 为 `0.890512x`，直接亏损。

dev 结果：

| case | dev multiple | total return | max DD | avg pos | 年度收益 | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `panic_exhaustion` | `1.352128x` | `+35.21%` | `-54.22%` | `9.98` | 2021 `-12.47%`，2022 `-14.70%`，2023 `-5.96%`，2024 `+21.29%`，2025 `+58.76%` | `1.172049x` |
| `inverse_liquidity_momentum` | `1.083901x` | `+8.39%` | `-50.96%` | `9.98` | 2021 `-9.60%`，2022 `-6.04%`，2023 `-9.28%`，2024 `-2.13%`，2025 `+43.71%` | `0.992485x` |
| `quiet_pullback` | `0.654779x` | `-34.52%` | `-42.76%` | `10.00` | 三年为负 | `0.662158x` |
| `cooldown_reversal_liquid` | `0.535084x` | `-46.49%` | `-66.52%` | `9.99` | 仅 2024 为正 | `0.521836x` |
| `anti_hot_low_vol` | `0.361995x` | `-63.80%` | `-71.78%` | `9.99` | 五年几乎全负 | `0.362060x` |

dev 同池随机：p50 `0.484749x`，p95 `0.877713x`，p99 `1.051386x`，max `1.390429x`。`panic_exhaustion` 超过随机 p99，但绝对收益太薄，年度结构完全不稳定，且最大回撤超过 `-54%`。

forward 结果：

| case | 2026 multiple | total return | max DD | avg pos | 是否超过随机 p95 | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `quiet_pullback` | `0.890512x` | `-10.95%` | `-13.35%` | `10.00` | 否 | `0.855563x` |
| `panic_exhaustion` | `0.872182x` | `-12.78%` | `-21.10%` | `9.96` | 否 | `0.767079x` |
| `inverse_liquidity_momentum` | `0.849326x` | `-15.07%` | `-18.53%` | `10.00` | 否 | `0.764011x` |
| `cooldown_reversal_liquid` | `0.831440x` | `-16.86%` | `-21.93%` | `10.00` | 否 | `0.749359x` |
| `hybrid_inverse_crowding` | `0.775192x` | `-22.48%` | `-24.52%` | `10.00` | 否 | `0.729388x` |

forward 同池随机：p50 `0.988664x`，p95 `1.239627x`，p99 `1.333813x`，max `1.401396x`。所有反热门 score 都低于随机中位数，说明 2026 明确不支持该方向。

### 35.4 观察

第一，反热门不是稳定 alpha。`panic_exhaustion` 的 dev 收益几乎全部来自 2024-2025，2021-2023 连续亏损；forward 2026 又转负。它更像 regime 条件策略，不是跨年度稳定主矿。

第二，第 34 节的“热门动量强负向”不能简单反过来买。`inverse_liquidity_momentum` dev 只有 `1.08x`，forward `0.849x`；这说明热门追涨是避坑线索，但其反面并不自然产出足够厚的收益。

第三，2026 的随机池本身并不差，随机均值约 `1.001x`，p95 `1.2396x`。反热门全线低于随机，不能解释成市场整体太差，而是排序方向错了。

第四，状态迁移非常强。2024-2025 有效的恐慌衰竭，在 2026 变成亏损；这提示下一步应建 regime router，而不是继续找单一静态 score。

### 35.5 反事实分析

第一反事实：如果第 34 节负向动量意味着反热门是主矿，`inverse_liquidity_momentum` 应在 dev/forward 都稳定为正。实际 dev 仅 `1.08x`，forward 亏 `-15.07%`，说明该推理不成立。

第二反事实：如果 `panic_exhaustion` 是真正的可复利 alpha，它不应只在 2024-2025 有效。实际 2021-2023 连续负，说明它依赖特定市场阶段。

第三反事实：如果 2026 亏损只是去掉最强事件导致，随机同池也应同样弱。实际随机中位数接近 1，p95 很高，说明反热门排序在 2026 主动选错了方向。

第四反事实：如果下一步继续手工调反热门权重，应至少看到某个 score 接近随机 p95。实际 forward 最佳 `0.890512x`，离 p95 很远。继续微调属于卡口径。

### 35.6 判定

`rejected_inverse_crowding_reversal_scan`。

本轮没有找到可融入策略。第 34/35 节共同指向一个更重要的结论：静态横截面 score 在不同 regime 下方向会翻转或失效。下一轮做 `cross_sectional_regime_router_oracle_scan_v1`，不是直接训练复杂模型，而是先做低自由度 regime router 的 oracle/可学性扫描：按市场宽度、指数趋势、波动、成交扩散、过去窗口策略表现等只使用 T 日以前信息，验证是否存在“某些 regime 用 low_vol_uptrend，某些 regime 避开或切换 panic_exhaustion/现金”的可学习结构。若 oracle 都不厚，就不训练 router；若 oracle 厚但简单可学习 router 不厚，则说明有过拟合风险。

## 36. cross_sectional_regime_router_oracle_scan_v1

### 36.1 假设

第 34/35 节显示静态横截面 score 的方向会随 regime 翻转。一个危险但必须验证的反事实是：如果每个 5 日窗口都能在 `low_vol_uptrend`、`vol_compression_breakout`、`panic_exhaustion`、`inverse_liquidity_momentum`、`quiet_pullback` 之间切换，收益上限可能很高；但这个切换如果只能事后知道，就是未来函数。

本轮把 oracle 上限和只用过去信息的 router 分开验证。若 oracle 很强但 rolling/grid router 不强，则不能训练复杂模型硬学，因为那很可能是在拟合不可学的 regime 标签。

### 36.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_regime_router_oracle_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_regime_router_oracle_scan_v1_summary.json
sha256:83b76f4e51fec1388e7359affe3d82ca01498001b6dbe5e6185219565e1dfd59
```

基础候选来自第 34/35 节：`low_vol_uptrend`、`vol_compression_breakout`、`panic_exhaustion`、`inverse_liquidity_momentum`、`quiet_pullback`。账户仍为 T 日信号、T+1 open、rebalance5、Top10、严格 price jump 过滤。切分：train 2021-2024，valid 2025，dev 2021-2025，forward 2026-01-01 到 2026-07-10。

router 类型：

| router | 含义 |
| --- | --- |
| `oracle_best_no_cash` | 每期事后选择收益最高基础策略，不允许空仓 |
| `oracle_best_with_cash` | 每期事后选择收益最高基础策略，若全部为负则空仓 |
| `rolling_best_12_no_cash` | 只用过去 12 个 rebalance 窗口均值选择策略 |
| `rolling_best_12_with_cash` | 同上，若过去均值非正则空仓 |
| `rolling_best_24_no_cash` | 只用过去 24 个窗口均值选择策略 |
| `grid_regime_router` | 在 train 上用低自由度市场状态规则选择策略或现金 |

grid 选择出来的规则：`breadth20 >= 0.48 且 median_ret20 >= 0.02` 用 `low_vol_uptrend`；`breadth20 <= 0.36 且 median_vol20 >= 0.028` 用 `panic_exhaustion`；`median_vol_ratio20 <= 0.75` 用 `quiet_pullback`；其它为空仓。该规则 train period multiple 只有 `1.026933x`，本身不是强规则。

### 36.3 结果

oracle 上限极高，但可学习 router 不厚。

| router | dev multiple | dev all years positive | forward multiple | forward max DD | forward avg pos | forward 去掉最佳 3 次后 | 判读 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `oracle_best_with_cash` | `1371.725686x` | 是 | `2.098280x` | `-4.84%` | `8.28` | `1.671441x` | 事后上限极强，不可作为策略 |
| `oracle_best_no_cash` | `853.699438x` | 是 | `2.028212x` | `-8.78%` | `10.00` | `1.624116x` | 事后上限极强，不可作为策略 |
| `rolling_best_12_with_cash` | `2.566544x` | 否 | `0.874994x` | `-21.13%` | `9.96` | `0.762947x` | 只用过去表现，forward 失败 |
| `rolling_best_12_no_cash` | `2.544169x` | 是 | `0.874994x` | `-21.13%` | `9.96` | `0.762947x` | valid 强但 2026 失效 |
| `rolling_best_24_no_cash` | `0.968855x` | 否 | `0.873071x` | `-19.16%` | `9.96` | `0.759841x` | dev/forward 均弱 |
| `grid_regime_router` | `1.517967x` | 否 | `1.093049x` | `-15.40%` | `4.40` | `0.965300x` | 正但太薄，且 avg pos 不达标 |

基础策略对照：`low_vol_uptrend` forward `1.123738x`，`vol_compression_breakout` forward `0.998764x`，`panic_exhaustion` forward `0.812105x`，`inverse_liquidity_momentum` forward `0.773207x`，`quiet_pullback` forward `0.845475x`。

### 36.4 观察

第一，oracle 证明“每个窗口该用哪个横截面状态”这件事有非常高的收益上限。dev oracle-with-cash 达 `1371x`，forward 也有 `2.10x`，并且回撤极低。这说明市场短周期横截面状态确实在动态切换，不是完全无结构。

第二，rolling router 证明“用过去哪个策略好”无法直接迁移。`rolling_best_12` 在 valid 2025 有 `1.69x`，但 forward 2026 亏 `-12.50%`；`rolling_best_24` 也亏。这说明 regime 切换不是简单的过去绩效延续。

第三，低自由度 market-state grid 只得到很弱的 forward 正收益，且平均持仓 `4.40`，不满足平均持仓大于 5 的要求。它像一个避险/择时规则，不是用户要求的厚 alpha。

第四，oracle 的选择分布并不极端。forward oracle-with-cash 中，`low_vol_uptrend` 9 次，`panic_exhaustion` 4 次，`inverse_liquidity_momentum` 3 次，`quiet_pullback` 3 次，`vol_compression_breakout` 1 次，cash 4 次。真正的难点不是找到单一策略，而是预测下一段窗口的适配策略。

### 36.5 反事实分析

第一反事实：如果 regime router 可以简单从过去 12/24 个窗口收益学到，rolling router 不应在 2026 明显亏损。实际两个 rolling router 都亏约 `-12.5%`，说明过去绩效延续不是有效路由信号。

第二反事实：如果市场宽度、趋势、波动和成交扩散已经足够解释 regime，低自由度 grid router 应在 dev 和 forward 都明显变厚。实际 dev 只有 `1.52x`，forward `1.09x` 且平均持仓不足，说明这些粗状态不够。

第三反事实：如果 oracle 强只是个别窗口贡献，去掉最佳 3 次后 forward 应迅速塌掉。实际 oracle-with-cash 去掉最佳 3 次仍 `1.67x`，说明动态切换上限是真实存在的；问题在可学习性，而不是收益只来自孤立极值。

第四反事实：如果直接训练复杂 Transformer router 是合理的，低自由度或 rolling router 至少应给出稳定正迁移迹象。实际没有。现在直接训练复杂 router，极可能把 oracle 标签拟合成未来函数。

### 36.6 判定

`rejected_regime_router_low_dof_scan`。

本轮没有找到可融入策略。重要结论是：动态横截面策略切换存在很高 oracle 上限，但现有可学习信号太弱，不能直接训练复杂路由器。下一轮做 `cross_sectional_supervised_router_probe_v1`：用 train 2021-2024 的 T 日以前 regime features、基础策略 score 统计、横截面分布特征去预测下一 5 日哪个基础策略收益最高，在 valid 2025 和 forward 2026 上验证。若线性/浅层模型仍不能迁移，则说明需要回到更底层的 token-level world model，而不是在策略层路由。

## 37. cross_sectional_supervised_router_probe_v1

### 37.1 假设

第 36 节显示 oracle router 很强，但 rolling/grid router 不强。为了判断“策略层路由”是否还有可学性，本轮用非常浅的 supervised probe：每个基础策略一条 ridge 回归，只使用 T 日以前的市场 regime 特征和过去窗口策略表现，预测下一 5 日各基础策略收益，再按预测收益选择策略或现金。

如果这个浅模型在 train/dev/valid 很强但 2026 forward 失败，则说明策略层 router 已经有明显过拟合风险；继续加 Transformer 或深模型大概率会把 oracle 标签拟合成未来函数。

### 37.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_supervised_router_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_supervised_router_probe_v1_summary.json
sha256:46d6cebc3ef8db756036ea97273a1b22deada33694b7b462e711f18f23a051bc
```

数据和基础策略沿用第 36 节。训练切分：train 2021-2024，valid 2025，dev 2021-2025，forward 2026-01-01 到 2026-07-10。模型为 ridge regression，alpha 扫描 `0.1/1/10/100`，空仓阈值扫描 `None/0/0.001/0.002`。

输入特征共 53 个，包括市场中位 5/20/60 日收益、5/20 日宽度、20 日波动、20 日离散度、成交量比，以及每个基础策略过去 4/12/24 个 rebalance 窗口的均值、波动、胜率、最近 1 次收益、最近 3 次均值。所有特征只使用当前 signal day 以前的信息。

### 37.3 结果

按 valid 2025 选择的最佳配置为 `alpha=0.1, threshold=0.0`。它在历史上非常漂亮，但 forward 失败：

| split | multiple | total return | max DD | avg pos | all years positive | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| train 2021-2024 | `23.173121x` | `+2217.31%` | `-11.28%` | `8.29` | 是 | `18.352825x` |
| valid 2025 | `1.793130x` | `+79.31%` | `-13.37%` | `7.88` | 是 | `1.377695x` |
| dev 2021-2025 | `40.765047x` | `+3976.50%` | `-13.37%` | `8.21` | 是 | `32.341265x` |
| forward 2026 | `0.942132x` | `-5.79%` | `-19.73%` | `9.53` | 否 | `0.753146x` |

按 train 最优选择的配置 `alpha=1.0, threshold=0.0` 更过拟合：train `27.379199x`，valid `1.690023x`，dev `45.352716x`，forward `0.915525x`。

valid leaderboard 前 8 名全部 forward 低于 1：

| rank | alpha | threshold | valid multiple | dev multiple | forward multiple |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | `0.1` | `0.0` | `1.793130x` | `40.765047x` | `0.942132x` |
| 2 | `0.1` | `0.001` | `1.793130x` | `41.474095x` | `0.986180x` |
| 3 | `0.1` | `0.002` | `1.793130x` | `39.062142x` | `0.986180x` |
| 4 | `0.1` | `None` | `1.789706x` | `31.420463x` | `0.915525x` |
| 5 | `1.0` | `0.001` | `1.703429x` | `44.975401x` | `0.915525x` |
| 6 | `1.0` | `0.002` | `1.703429x` | `44.595548x` | `0.915525x` |
| 7 | `1.0` | `0.0` | `1.690023x` | `45.352716x` | `0.915525x` |
| 8 | `10.0` | `None` | `1.674505x` | `33.542611x` | `0.973184x` |

### 37.4 预测质量

train 上各基础策略收益预测相关性约 `0.59-0.66`，valid 降到 `-0.07-0.35`，forward 约 `0.15-0.40`。但即使部分 forward 相关性为正，最终路由仍亏损，说明模型的排序/阈值和收益幅度校准没有迁移。

| variant | train corr | valid corr | forward corr | forward sign acc |
| --- | ---: | ---: | ---: | ---: |
| `inverse_liquidity_momentum` | `0.6284` | `0.3461` | `0.1856` | `45.83%` |
| `low_vol_uptrend` | `0.6223` | `-0.0695` | `0.1455` | `45.83%` |
| `panic_exhaustion` | `0.6613` | `0.2792` | `0.2651` | `54.17%` |
| `quiet_pullback` | `0.6287` | `0.1865` | `0.3978` | `37.50%` |
| `vol_compression_breakout` | `0.5947` | `-0.0703` | `0.2775` | `54.17%` |

### 37.5 观察

第一，历史结果太漂亮反而是风险信号。dev `40.77x`、五年全正、平均持仓 `8.21`，表面上已经达到用户要求；但 forward 直接亏损，说明这类策略层 router 很容易在历史 regime 上拟合出漂亮曲线。

第二，valid 2025 并不能代表 2026。所有 valid leaderboard 前列配置在 2026 都低于 1，说明单一年份 validation 不足以约束 regime 迁移。

第三，模型在 2026 的选择分布明显偏向 `quiet_pullback` 和 `inverse_liquidity_momentum`，而第 36 节 oracle forward 更偏 `low_vol_uptrend`。这说明模型没有学到 2026 的 regime 切换方向。

第四，空仓阈值没有解决问题。`threshold=0.001/0.002` 把 forward 从 `0.9421x` 提到 `0.9862x`，但仍低于 1，更低于用户要求，也没有形成厚收益。

### 37.6 反事实分析

第一反事实：如果策略层 supervised router 是可学的，valid 最优配置不应在 forward 亏损。实际 valid 第一名 forward `0.942132x`，说明可学性不足。

第二反事实：如果只是阈值导致亏损，加入现金阈值应明显转正。实际最高也只有 `0.986180x`，说明问题不是仓位控制，而是路由判断错。

第三反事实：如果浅模型容量不够，至少应看到 forward 接近正收益边缘且决策方向接近 oracle。实际决策分布偏离 oracle，且收益为负；直接加深模型很可能扩大过拟合。

第四反事实：如果 2025 valid 足以代表未来，leaderboard 应与 2026 同向。实际前 8 名全部 forward 低于 1，说明 regime 迁移比单年 valid 更复杂。

### 37.7 判定

`rejected_supervised_strategy_layer_router`。

本轮没有找到可融入策略。虽然 dev 已达到五年几十倍、五年全正、平均持仓大于 5，但 forward 失败，因此必须否决。第 36/37 节共同说明：策略层 oracle 上限存在，但用粗 regime 特征和过去策略表现学习路由会过拟合。下一轮转向更底层的 `cross_sectional_token_world_model_probe_v1`：不再预测“选哪个手工策略”，而是直接用每个股票过去 20-60 日 token、横截面 rank、市场状态 token 预测下一 5 日收益/胜率/尾部收益，先做小模型和严格 walk-forward，避免把策略层标签当真值。

## 38. cross_sectional_token_world_model_probe_v1

### 38.1 假设

第 36/37 节说明策略层路由很容易过拟合，必须回到底层股票 token。新假设是：如果市场短周期确实可预测，那么直接用每只股票过去 60 日的日线 token、横截面 rank、成交状态去预测未来 5 日收益，应该能在 forward 上保留一些 rank IC 和 Top10 label 增量。

本轮使用很小的 temporal Transformer，只做 probe，不追求最终策略。重点不是模型名字，而是验证底层 token 直接预测是否比手工策略层标签更稳。

### 38.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_token_world_model_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_token_world_model_probe_v1_summary.json
sha256:49762c9c975747ed6a33d9e82a4043736870d710f7e9572f204b7e1118217be9
```

模型：小型 temporal Transformer，`seq_len=60`，日线 token 维度 11，`d_model=32`，`2` 层 encoder，训练 `8` epoch，MSE 拟合每个 rebalance session 内的未来 5 日 open-to-open rank label。训练集 2021-2024，valid 2025，dev 2021-2025，forward 2026-01-01 到 2026-07-10。

输入 token 包括：1 日收益、开盘 gap、日内收益、日内 range、close/vwap、成交量比、5/20 日收益、1 日收益横截面 rank、20 日收益横截面 rank、成交额横截面 rank。所有特征只使用 T 日及以前数据。每期按 20 日成交额取 pool500，Top10 回放，T+1 open 入场，5 日持有，严格 price jump 过滤。

样本量：train `96677`，valid `24362`，dev `121065`，forward `11914`；对应 session：train `194`，valid `49`，dev `243`，forward `24`。forward 同池随机 Top10 跑 `200` 次。

### 38.3 训练和结果

valid 最佳 epoch 为第 3 轮。训练过程没有出现历史暴涨式过拟合，但收益厚度不足。

| split | multiple | total return | max DD | avg pos | all years positive | mean rank IC | Top10 raw label | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| train 2021-2024 | `1.192214x` | `+19.22%` | `-41.51%` | `10.00` | 否 | `0.0748` | `+0.002885` | `1.122996x` |
| valid 2025 | `1.429934x` | `+42.99%` | `-17.55%` | `10.00` | 是 | `0.0834` | `+0.010354` | `1.244916x` |
| dev 2021-2025 | `1.342577x` | `+34.26%` | `-41.51%` | `10.00` | 否 | `0.0698` | `+0.003758` | `1.344546x` |
| forward 2026 | `1.049797x` | `+4.98%` | `-22.99%` | `10.00` | 是 | `0.0476` | `+0.004675` | `0.836767x` |

forward 随机同池 Top10：p50 `1.069526x`，p90 `1.247929x`，p95 `1.308078x`，p99 `1.427856x`，max `1.523960x`。模型 forward `1.049797x` 低于随机中位数和 p95，不能作为策略证据。

训练历史摘要：

| epoch | train loss | valid multiple | valid mean rank IC | valid Top10 label |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.084412` | `1.353826x` | `0.0777` | `+0.009454` |
| 2 | `0.082204` | `1.195948x` | `0.0815` | `+0.005576` |
| 3 | `0.082048` | `1.429934x` | `0.0834` | `+0.010354` |
| 4 | `0.081976` | `1.266861x` | `0.1009` | `+0.007945` |
| 5 | `0.081933` | `1.141148x` | `0.0996` | `+0.005527` |
| 8 | `0.081549` | `0.950738x` | `0.0871` | `+0.002589` |

### 38.4 观察

第一，底层 token 模型有弱预测力。forward mean rank IC 为 `0.0476`，Top10 raw label 为 `+0.4675%`，不是完全无信号。这比第 37 节策略层 router 的“历史极强、forward 失败”更健康。

第二，弱 IC 没有转成可交易厚收益。forward only `+4.98%`，且低于随机 p50；去掉最佳 3 次调仓后只剩 `0.836767x`。这说明当前目标函数和选股尾部质量不足。

第三，valid 2025 明显好于 forward 2026。valid `1.429934x`，forward `1.049797x`，说明 token 模型仍有 regime 迁移问题，但没有第 37 节那种历史曲线极端虚高。

第四，MSE 拟合 session 内 rank label 可能太平均。用户目标需要右尾厚收益，当前 loss 更偏整体排序，未显式优化 Top10、胜率、右尾收益和坏尾规避。

### 38.5 反事实分析

第一反事实：如果底层 token 已经足以直接产生策略，forward 应超过随机 p95 并保持去掉最佳 3 次后为正。实际低于随机 p50，去掉最佳 3 次后低于 1，说明第一版不能融入。

第二反事实：如果 token 方向完全无效，forward rank IC 应接近 0 或为负。实际为 `0.0476`，说明有一点预测土壤，值得继续改目标和采样方式。

第三反事实：如果问题只是模型不够大，train/dev 应明显强于 valid/forward。实际 train/dev 也不强，说明瓶颈更可能在目标函数、样本权重、标签定义和右尾建模，而不是模型容量。

第四反事实：如果继续使用 MSE rank label 足够，valid 后续 epoch 的 IC 提升应同步提升账户收益。实际 epoch 4/5 的 IC 高但账户变差，说明账户目标和平均 rank IC 不完全一致。

### 38.6 判定

`rejected_token_world_model_probe_v1_not_enough`。

本轮没有找到可融入策略。结论是：底层 token 方向有弱土壤，但第一版 MSE rank-label Transformer 远达不到收益厚度。下一轮做 `cross_sectional_right_tail_token_ranker_v1`：仍用底层 token，但改成更贴近交易目标的 right-tail/pairwise ranking objective，增加 Top10 右尾收益、坏尾规避、session-balanced sampling 和多尺度 metric，验证能否把弱 IC 转化为可交易收益。

## 39. cross_sectional_right_tail_token_ranker_v1

### 39.1 假设

第 38 节的 MSE rank-label Transformer 有弱 IC，但账户收益不厚。一个直接反事实是：用户目标要的是 Top10 右尾收益，不是全体股票平均排序。因此本轮保留同一套 60 日 token 和回放口径，只把训练目标改成 session 内 right-tail 分类：未来 5 日收益位于当期 pool 前 20% 记为正样本，后 20% 给坏尾惩罚，中间样本降权。

如果问题只是 MSE 目标太平均，那么 right-tail 目标应提升 Top10 label，并在 forward 上至少超过第 38 节和随机 p95。

### 39.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_right_tail_token_ranker_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_right_tail_token_ranker_v1_summary.json
sha256:9f6b8d524c6ce20ee573b64e4399a3f14ef9be808643c344f6b449af99c55bde
```

模型结构、token、切分、pool500、Top10 回放、交易约束与第 38 节一致。目标函数改为 `BCEWithLogitsLoss`，正样本为 session 内未来 5 日收益 top 20%，坏尾为 bottom 20%。权重：top `3.0`，bottom `2.0`，middle `0.35`，并在 session 内归一化，避免某些 session 权重过大。

样本量仍为：train `96677`，valid `24362`，dev `121065`，forward `11914`；session：train `194`，valid `49`，dev `243`，forward `24`。forward 同池随机 Top10 跑 `200` 次。

### 39.3 结果

right-tail 目标在 valid 上增强了右尾，但 forward 明显恶化。

| split | multiple | total return | max DD | avg pos | all years positive | mean rank IC | Top10 raw label | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| train 2021-2024 | `1.456608x` | `+45.66%` | `-46.23%` | `10.00` | 否 | `0.0954` | `+0.003994` | `1.131593x` |
| valid 2025 | `1.582793x` | `+58.28%` | `-17.01%` | `10.00` | 是 | `0.1062` | `+0.012003` | `1.353243x` |
| dev 2021-2025 | `2.080457x` | `+108.05%` | `-46.23%` | `10.00` | 否 | `0.0873` | `+0.005940` | `1.654852x` |
| forward 2026 | `0.815930x` | `-18.41%` | `-22.28%` | `10.00` | 否 | `0.0295` | `-0.009462` | `0.697386x` |

forward 随机同池 Top10 与第 38 节一致：p50 `1.069526x`，p90 `1.247929x`，p95 `1.308078x`，p99 `1.427856x`，max `1.523960x`。本轮 forward `0.815930x` 不仅没有超过随机，甚至显著低于随机中位数。

训练历史显示 valid 最佳 epoch 仍为第 3 轮：

| epoch | train loss | valid multiple | valid mean rank IC | valid Top10 label |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.692263` | `1.183425x` | `0.1008` | `+0.006209` |
| 2 | `0.691114` | `1.252515x` | `0.1083` | `+0.007863` |
| 3 | `0.690492` | `1.582793x` | `0.1062` | `+0.012003` |
| 4 | `0.690001` | `1.198051x` | `0.0723` | `+0.005523` |
| 5 | `0.689266` | `1.431564x` | `0.0970` | `+0.010541` |
| 8 | `0.686702` | `0.958666x` | `0.0679` | `+0.003018` |

### 39.4 观察

第一，right-tail 目标确实增强了 valid。相比第 38 节，valid 从 `1.429934x` 提升到 `1.582793x`，Top10 label 从 `+1.0354%` 提升到 `+1.2003%`。这说明目标函数对账户尾部有影响。

第二，forward 方向翻转。第 38 节 forward 仍有 `+4.98%` 和正 Top10 label；本轮 forward 变成 `-18.41%`，Top10 raw label 为 `-0.9462%`。这不是收益不够厚，而是右尾定义在 2025 和 2026 之间发生了迁移失败。

第三，dev 仍不满足年度稳定性。虽然 dev multiple 到 `2.08x`，但 2023 为 `-14.82%`，最大回撤 `-46.23%`，离五年几十倍和五年全正很远。

第四，随机裁判非常强。2026 随机 p50 已经 `1.0695x`，说明 pool500 在该 forward 期有很多随机可赚窗口；模型却主动选到了负尾，进一步证明排序方向错了。

### 39.5 反事实分析

第一反事实：如果第 38 节瓶颈只是 MSE 目标太平均，right-tail BCE 应在 forward 上改善。实际显著恶化，说明问题不只是目标平均化。

第二反事实：如果 valid 2025 的右尾形态能代表 2026，forward Top10 raw label 应继续为正。实际为 `-0.9462%`，说明 2026 的右尾条件与 2025 不同，甚至可能相反。

第三反事实：如果模型容量不足是主因，改目标不应造成 forward 反向选择。实际目标改变后 forward 选到坏尾，说明标签定义/状态迁移比容量更关键。

第四反事实：如果继续调 top/bottom 权重能解决，至少本轮应接近随机中位数。实际低于随机 p50 约 `25%`，继续调权重很可能是在卡口径。

### 39.6 判定

`rejected_right_tail_token_ranker_v1`。

本轮没有找到可融入策略。底层 token 有弱信号，但简单 right-tail 目标会放大 regime 偏差，不能直接继续加权。下一轮应做 `token_label_regime_flip_diagnostic_v1`：诊断 2025 与 2026 的右尾样本在 token 空间、收益分布、市场宽度、短期动量/反转、成交状态上的差异，确认是标签 regime 翻转、pool 随机结构变化，还是模型只学到 2025 特有模式。只有理解右尾为什么翻转，才决定下一步是加 regime token、改 label horizon、还是引入 QMT 日内 token。

## 40. token_label_regime_flip_diagnostic_v1

### 40.1 假设

第 39 节 right-tail 目标在 valid 2025 明显增强，但 forward 2026 反向选股。一个关键问题是：2026 是否本身没有可预测右尾，还是右尾仍存在、只是右尾对应的 token 条件变了。本轮不训练模型，只比较 valid 2025 与 forward 2026 的真实 top20/bottom20 样本分布，判断第 39 节失败的主因是否为右尾条件 regime shift。

如果 2026 真实 top20 收益仍明显为正，而第 39 模型 Top10 raw label 为负，则不能继续简单加大 right-tail 权重；应把市场状态显式并入模型或标签。

### 40.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_token_label_regime_flip_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/token_label_regime_flip_diagnostic_v1_summary.json
summary sha256:a8f20767f907e3dafe400a32db8ffa4256d1a94bfb2a9f908bbd2a213966673d
script sha256:9d2096694559a53c45e24a3e1fda735959431d740d12110cdad4b73346ebe7a3
```

本轮复用第 39 节的 pool500、60 日 token、5 日持仓标签和样本切分。对每个调仓 session 内的真实未来 5 日收益按 top20、bottom20、middle、all 分组，比较 `last`、`mean5`、`mean20` 三个视角下的 token 均值、top-bottom contrast、forward-minus-valid shift 和 contrast sign flip。

### 40.3 结果

真实右尾在 2026 并未消失，反而更厚。

| split | samples | sessions | true top20 label mean | true bottom20 label mean | all label mean | 第 39 模型 Top10 raw label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid 2025 | `24362` | `49` | `+0.102398` | `-0.069716` | `+0.005363` | `+0.012003` |
| forward 2026 | `11914` | `24` | `+0.131760` | `-0.098466` | `+0.001897` | `-0.009462` |

第 39 模型在 2026 的账户结果仍为 `0.815930x`，最大回撤 `-22.28%`，mean rank IC `0.0295`。也就是说，模型不是遇到“没有右尾”的市场，而是在右尾存在的市场里把 Top10 选到了负尾。

最明显的 feature shift 出现在中期动量、价格相对 vwap、振幅和短期强度上。

| view | largest top feature shift forward-valid | largest contrast shift forward-valid | contrast sign flips |
| --- | --- | --- | --- |
| last | `close_vwap +0.129951`, `day_range +0.096934`, `rank_ret20 +0.071254`, `ret20 +0.060840` | `rank_ret20 +0.084663`, `ret20 +0.073524`, `ret5 +0.054852`, `day_range +0.045944` | `rank_ret20`, `rank_amount` |
| mean5 | `close_vwap +0.129249`, `day_range +0.096110`, `rank_ret20 +0.071086`, `ret20 +0.062583` | `rank_ret20 +0.074450`, `ret20 +0.072916`, `ret5 +0.053129`, `day_range +0.042192` | `log_vol_ratio`, `ret5`, `ret20`, `rank_ret20`, `rank_amount` |
| mean20 | `close_vwap +0.129790`, `day_range +0.081041`, `rank_ret20 +0.067508`, `ret20 +0.057949` | `rank_ret20 +0.054188`, `ret20 +0.053355`, `ret5 +0.040157`, `day_range +0.030598` | `open_gap`, `log_vol_ratio`, `ret5`, `ret20`, `rank_ret1`, `rank_ret20` |

### 40.4 观察

第一，2026 的真实 top20 label 均值为 `+13.1760%`，高于 2025 的 `+10.2398%`。这直接否定“forward 差是因为市场没有右尾”的解释。

第二，模型 forward Top10 raw label 为 `-0.9462%`，与真实 top20 的 `+13.1760%` 同时存在。问题不是右尾不存在，而是模型学到的 2025 右尾映射在 2026 失效。

第三，2025 的 top-bottom contrast 在 `ret20`、`ret5`、`rank_ret20` 上多为负，而 2026 多个视角翻正。也就是说，2025 更像“强势/拥挤之后反而不是右尾”，2026 更像“中期强势继续贡献右尾”。这正解释了 right-tail BCE 为什么会放大 2025 模式后在 2026 反向。

第四，`close_vwap`、`day_range`、`rank_ret20` 的 top feature shift 很大，说明右尾条件不只是收益标签漂移，而是横截面状态本身在迁移。仅靠股票自身 60 日序列，模型可能把 regime 当成稳定规律。

### 40.5 反事实分析

第一反事实：如果第 39 节只是训练不足或模型容量不足，2026 真实 top20 条件不应出现系统性翻转。实际多个 view 的 `ret20`、`ret5`、`rank_ret20` contrast 翻转，说明状态条件缺失比容量更关键。

第二反事实：如果继续调高 top 权重能解决，模型应该已经至少接近真实 top20 的方向。实际模型 Top10 raw label 为负，调权重很可能进一步强化错误 regime。

第三反事实：如果 2026 forward 随机强只是偶然，真实 top20 label 不应显著高于 2025。实际 2026 top20 更厚，说明市场提供了机会，但模型没有对齐状态。

第四反事实：如果只做单股票 token 就足够，2025 到 2026 的横截面 contrast 不应影响方向。实际右尾条件与横截面中期动量/振幅/成交状态一起迁移，后续必须把市场状态作为一等输入。

### 40.6 判定

`tail_feature_regime_shift_confirmed_needs_regime_conditioned_label`。

本轮没有产生可融入策略，但确认了一个重要转向：不要继续做简单 right-tail 权重扫描。下一轮做 `cross_sectional_regime_token_transformer_v1`：在第 39 节的 stock token 上追加调仓日横截面 regime token，让模型显式知道当前市场更接近“中期动量继续”还是“拥挤反转”。若 regime token 仍不能改善 forward，下一步再考虑分 horizon label 或引入 QMT 日内 token。

## 41. cross_sectional_regime_token_transformer_v1

### 41.1 假设

第 40 节确认 2025 与 2026 的右尾条件发生了 regime shift，尤其是 `ret20`、`ret5`、`rank_ret20` 的 top-bottom contrast 从负向切到正向。一个自然反事实是：第 39 节模型失败不是因为 Transformer 不行，而是因为它只看单股票 60 日序列，缺少当期横截面市场状态。

本轮不改变数据源、不改正式框架，只在第 39 节右尾 Transformer 上追加一个调仓日横截面 regime token，验证“给模型市场状态后，能否修复 2026 右尾方向”。

### 41.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_regime_token_transformer_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_regime_token_transformer_v1_summary.json
script sha256:d812bb97edfecb237919de64c199b67d035927b063c939277a0c1572954c0243
summary sha256:4e11bc202c99ffc1bf8d7f74f36375fe3a72aff6ea7220ebd7a19680e3f34ac0
```

模型主体、pool500、Top10、5 日持仓、right-tail BCE 权重、训练/验证/forward 切分与第 39 节一致。变化只有一个：每个样本从 `60` 个 stock token 变为 `61` 个 token，最后一个 token 是调仓日 pool500 横截面 regime token。

regime token 包含：`mkt_ret1_mean`、`mkt_ret5_mean`、`mkt_ret20_mean`、`mkt_rank_ret20_spread`、`mkt_day_range_mean`、`mkt_close_vwap_mean`、`mkt_log_vol_mean`、`mkt_up1_ratio`、`mkt_up5_ratio`、`mkt_high_mom_high_range`、`mkt_mom_vol_corr`。

### 41.3 结果

valid 仍强，forward 有所修复但仍为负，rank IC 接近 0。

| split | multiple | total return | max DD | avg pos | all years positive | mean rank IC | Top10 raw label | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| train 2021-2024 | `1.387835x` | `+38.78%` | `-35.59%` | `9.99` | 否 | `0.0920` | `+0.003073` | `0.972530x` |
| valid 2025 | `1.500076x` | `+50.01%` | `-10.39%` | `10.00` | 是 | `0.0856` | `+0.010503` | `1.250839x` |
| dev 2021-2025 | `2.029728x` | `+102.97%` | `-35.59%` | `10.00` | 否 | `0.0875` | `+0.004535` | `1.450886x` |
| forward 2026 | `0.900427x` | `-9.96%` | `-18.36%` | `9.96` | 否 | `-0.0016` | `-0.005735` | `0.772028x` |

forward 随机同池 Top10：p50 `1.069526x`，p90 `1.247929x`，p95 `1.308078x`，p99 `1.427856x`，max `1.523960x`。本轮 forward `0.900427x` 仍明显低于随机中位数。

训练历史：

| epoch | train loss | valid multiple | valid mean rank IC | valid Top10 label |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.693058` | `1.410513x` | `0.0747` | `+0.008825` |
| 2 | `0.691985` | `1.473494x` | `0.0849` | `+0.009742` |
| 3 | `0.691697` | `1.500076x` | `0.0856` | `+0.010503` |
| 4 | `0.691138` | `1.258071x` | `0.0882` | `+0.006933` |
| 5 | `0.690150` | `1.230599x` | `0.0890` | `+0.006749` |
| 6 | `0.688818` | `1.399314x` | `0.1047` | `+0.009028` |
| 7 | `0.685563` | `1.077690x` | `0.0432` | `+0.004819` |
| 8 | `0.682998` | `1.325283x` | `0.1073` | `+0.007634` |

### 41.4 观察

第一，regime token 确实缓解了一部分 forward 反向。第 39 节 forward 为 `0.815930x`，本轮提高到 `0.900427x`；Top10 raw label 从 `-0.9462%` 改善到 `-0.5735%`。这说明第 40 节的判断方向不是完全错的。

第二，改善幅度远不够。forward rank IC 为 `-0.0016`，基本没有横截面预测能力；forward 仍低于随机 p50，并且去掉最佳 3 次调仓后只有 `0.772028x`。

第三，valid 2025 仍然很好，但低于第 39 节的 `1.582793x`。regime token 带来了一点正则化效果，却没有形成可迁移的状态条件函数。

第四，dev 仍有 2023 负收益，且最大回撤 `-35.59%`。即便忽略 forward，也离“五年每年正收益、几十倍”很远。

### 41.5 反事实分析

第一反事实：如果失败主因只是缺少横截面状态，一个简单 regime token 应明显修复 2026。实际只从 `0.815930x` 修到 `0.900427x`，说明状态信息需要更强的表达或不同训练目标。

第二反事实：如果 2025 valid 是可靠的模型选择标准，valid 最佳 epoch 应在 forward 上至少不为负。实际 best epoch 3 forward 仍为负，说明单年 valid 选 epoch 本身也在过拟合 2025。

第三反事实：如果继续扩模型容量能解决，train/dev 应快速增强且 forward 至少改善到随机附近。实际 dev 只有 `2.03x`，forward rank IC 归零，说明瓶颈不在容量。

第四反事实：如果 right-tail BCE 是正确目标，加入 regime 后不应损失 valid 右尾并继续 forward 负。实际 valid 降低、forward 仍负，说明 BCE top20 标签可能仍过于离散，难以学习不同年份中“什么形态能继续走”的连续边界。

### 41.6 判定

`regime_token_transformer_not_enough`。

本轮没有可融入策略。下一轮不再沿同一个 right-tail BCE 加 token 或加权重，而是验证“单年 valid 选择 + 离散 top20 标签”是否是主瓶颈：做 `cross_sectional_year_invariant_rank_transformer_v1`。目标改为 session 内连续 rank/收益混合的 listwise 近似，训练和选择时按年份均衡评估，优先找跨 2021-2025 多年都不坏、2026 不反向的底层预测器，而不是追 2025 单年最高收益。

## 42. cross_sectional_year_invariant_rank_transformer_v1

### 42.1 假设

第 41 节说明，简单追加 regime token 只能轻微缓解 forward 反向，不能形成可迁移 alpha。一个更关键的反事实是：第 39-41 节一直用 2025 valid 收益最高点选 epoch，且 right-tail BCE 把标签离散化，可能放大了 2025 单年右尾形态。

本轮保留第 41 节的 `60 + 1` token 输入，不扩大模型、不换数据源，把目标改成 session 内连续 rank 的 `SmoothL1Loss`，并用 2021-2025 dev 的年度均衡分数选 epoch。目的不是追 2025 单年收益，而是看是否能得到多年更平滑、2026 不反向的底层预测器。

### 42.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_year_invariant_rank_transformer_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_year_invariant_rank_transformer_v1_summary.json
script sha256:a82bb47e873a4a9b8d358cfa0c20c4478202b7bd4f7e8b5b175f79f49629e754
summary sha256:0a9de4e15251b683d465360ecd84912a0af8dc4886cea4a32a3420a548fc81b7
```

模型结构、pool500、Top10、5 日持仓、交易回放与第 41 节一致。训练目标从 `BCEWithLogitsLoss(top20)` 改为连续 session rank 的 `SmoothL1Loss(beta=0.10)`；rank 两端样本权重 `1.5`，中部权重 `0.7`，每个 session 内重新归一化。

epoch 选择不再使用 2025 valid multiple，而是最大化 2021-2025 dev 上的年度均衡分数：`mean_year - 0.75 * std_year + 0.25 * min_year + 0.15 * log_multiple + 0.50 * rank_ic - 0.10 * drawdown`。

### 42.3 结果

年度均衡选择压低了 2025 过拟合收益，但 2026 仍为负。

| split | multiple | total return | max DD | avg pos | all years positive | mean rank IC | Top10 raw label | 去掉最佳 3 次后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| train 2021-2024 | `1.657185x` | `+65.72%` | `-37.21%` | `10.00` | 否 | `0.0831` | `+0.004368` | `1.122226x` |
| valid 2025 | `1.246738x` | `+24.67%` | `-8.30%` | `10.00` | 是 | `0.0881` | `+0.005891` | `1.102840x` |
| dev 2021-2025 | `1.947365x` | `+94.74%` | `-37.21%` | `10.00` | 否 | `0.0809` | `+0.004688` | `1.367983x` |
| forward 2026 | `0.879652x` | `-12.03%` | `-16.83%` | `10.00` | 否 | `0.0070` | `-0.004647` | `0.794372x` |

forward 随机同池 Top10：p50 `1.069526x`，p90 `1.247929x`，p95 `1.308078x`，p99 `1.427856x`，max `1.523960x`。本轮 forward `0.879652x` 仍显著低于随机中位数。

best epoch 为第 6 轮，年度均衡选择分数 `0.137373`。训练历史摘要：

| epoch | train loss | dev score | dev multiple | 2023 return | dev mean rank IC | valid multiple | valid Top10 label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.248024` | `0.051555` | `1.656578x` | `-16.71%` | `0.0872` | `1.227475x` | `+0.004871` |
| 2 | `0.244639` | `-0.064535` | `1.280186x` | `-21.20%` | `0.0895` | `1.156303x` | `+0.003760` |
| 3 | `0.244093` | `-0.093191` | `1.290548x` | `-22.77%` | `0.0803` | `1.272268x` | `+0.005985` |
| 4 | `0.243056` | `-0.115575` | `1.100232x` | `-13.90%` | `0.0979` | `1.208260x` | `+0.004823` |
| 5 | `0.240878` | `0.045701` | `1.604500x` | `-11.01%` | `0.1043` | `1.132134x` | `+0.003723` |
| 6 | `0.239321` | `0.137373` | `1.947365x` | `-6.63%` | `0.0809` | `1.246738x` | `+0.005891` |
| 7 | `0.237724` | `0.076611` | `1.823561x` | `-12.51%` | `0.0682` | `1.263808x` | `+0.006101` |
| 8 | `0.236644` | `0.082834` | `1.453049x` | `-5.27%` | `0.1138` | `1.162316x` | `+0.004264` |

### 42.4 观察

第一，年度均衡选择确实削弱了 2025 单年过拟合。valid 从第 41 节的 `1.500076x` 降到 `1.246738x`，回撤也从 `-10.39%` 收到 `-8.30%`。

第二，2023 仍然是硬伤。best epoch 的 2023 年收益仍为 `-6.63%`，说明当前日线 token 结构没有学出跨市场状态稳定赚钱的规律。

第三，forward 没有修复。forward rank IC 只有 `0.0070`，Top10 raw label 仍为负，账户 `0.879652x`，低于第 41 节 `0.900427x`。这说明问题不只是 valid 单年选点。

第四，连续 rank 保住了一点平均 IC，但没有右尾厚度。dev mean rank IC `0.0809` 不低，但 dev multiple 只有 `1.947x`，forward Top10 label 仍负。平均排序能力与用户目标所需的强右尾捕获之间存在明显断层。

### 42.5 反事实分析

第一反事实：如果第 41 节失败主要来自 2025 valid 选 epoch 过拟合，本轮年度均衡选择应显著改善 forward。实际 forward 更差，说明选 epoch 不是主因。

第二反事实：如果离散 top20 BCE 是主因，连续 rank loss 应恢复 2026 正收益。实际只恢复了一点 rank IC，账户仍负，说明标签形式不是核心瓶颈。

第三反事实：如果日线 60 日 token 已经包含足够的右尾信息，dev rank IC 转成 forward Top10 应至少为正。实际 forward Top10 raw label 为 `-0.4647%`，说明右尾条件没有迁移。

第四反事实：如果继续调权重或容量能解决，应至少看到 dev 年度稳定性明显改善。实际 2023 仍负、最大回撤 `-37.21%`，说明需要换信息结构，而不是在同一日线 token 上继续打磨。

### 42.6 判定

`year_invariant_rank_transformer_not_enough`。

本轮没有可融入策略。结论是：基于日线 OHLCV 的单股票 token + 简单横截面 regime token，已有弱 IC，但无法稳定捕获用户目标所需的右尾厚收益。下一轮转向 `cross_sectional_multi_horizon_label_surface_diagnostic_v1`：先不训练大模型，诊断 2021-2026 在 3/5/10 日持仓 horizon 上的右尾土壤和 regime 翻转位置，确认问题是 5 日标签错位、右尾出现在更短/更长 horizon，还是需要引入 QMT 日内 token。

## 43. cross_sectional_multi_horizon_label_surface_diagnostic_v1

### 43.1 假设

第 42 节说明，继续在 5 日标签上训练日线 token 模型，很可能是在错误 horizon 上消耗容量。一个更高维的反事实是：右尾并没有消失，但它可能集中在更短或更长持仓窗口；如果标签 horizon 错位，模型会把可预测的 3 日或 10 日短期结构压成 5 日噪声。

本轮不训练模型，只诊断 3/5/10 日 open-to-open 标签面，并按年份比较候选横截面方向的 Top10 label 稳定性。目标是找到“土壤”，不是直接给策略过关。

### 43.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_multi_horizon_label_surface_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_multi_horizon_label_surface_diagnostic_v1_summary.json
script sha256:cef7e164522e0cf2f8a5126d2311d75bb72046341affe15d0fef7b8413c3da21
summary sha256:1c7627aa3f47d9c5983ab0b92e3c9b405acf1603d8f9c06b442fbec2b306e848
```

本轮使用同一套 Qlib/QMT 日线 OHLCV token 特征、pool500、Top10、每 5 日抽样 session。horizon 扫描 `3/5/10` 日；视角扫描 `last/mean5/mean20`；候选方向包括单特征正/负方向，以及 `low_vol_uptrend`、`panic_exhaustion`、`quiet_pullback`、`liquid_momentum` 等组合方向。

稳定性评分：对 2021-2025 年度 Top10 label 做均值、波动、最差年份惩罚，并加入 2026 forward Top10 label。这个评分只用于定位土壤，不作为策略收益裁判。

### 43.3 标签面结果

整体右尾在 3/5/10 日都存在，而且 2026 的 top20 更厚；问题仍然是如何选中它。

| horizon | year | all mean | top20 mean | bottom20 mean | std |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 2021 | `-0.000795` | `+0.089953` | `-0.080440` | `0.063194` |
| 3 | 2022 | `-0.002243` | `+0.081863` | `-0.078751` | `0.059988` |
| 3 | 2023 | `-0.001466` | `+0.068652` | `-0.062423` | `0.050907` |
| 3 | 2024 | `-0.002245` | `+0.084914` | `-0.078176` | `0.062827` |
| 3 | 2025 | `+0.005669` | `+0.093713` | `-0.068510` | `0.062067` |
| 3 | 2026 | `+0.006423` | `+0.108106` | `-0.078600` | `0.069004` |
| 5 | 2026 | `+0.002362` | `+0.142882` | `-0.111158` | `0.094298` |
| 10 | 2026 | `+0.003480` | `+0.208402` | `-0.147475` | `0.134511` |

### 43.4 候选方向 leaderboard

最稳定的方向不是第 39-42 节尝试强化的动量右尾，而是 3 日 horizon 下的 `close_vwap_neg`：过去 20 日均值里收盘相对 VWAP 偏弱的股票，在 3 日维度更容易有短反弹。

| rank | horizon | view | candidate | dev 年度 Top10 label | 2026 Top10 label | 2026 rank IC | stability score |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 3 | mean20 | `close_vwap_neg` | 2021 `+0.001494`, 2022 `+0.001736`, 2023 `+0.000060`, 2024 `+0.007021`, 2025 `+0.006990` | `+0.008298` | `0.0094` | `0.004028` |
| 2 | 3 | mean20 | `rank_amount_pos` | 2021 `+0.000079`, 2022 `-0.001199`, 2023 `+0.000051`, 2024 `-0.003721`, 2025 `+0.004700` | `+0.018803` | `0.0377` | `0.002887` |
| 3 | 3 | mean5 | `close_vwap_neg` | 2021 `+0.000118`, 2022 `-0.000509`, 2023 `+0.000427`, 2024 `+0.006122`, 2025 `+0.007270` | `+0.007571` | `0.0302` | `0.002485` |
| 4 | 3 | last | `rank_amount_pos` | 2021 `+0.002077`, 2022 `-0.005469`, 2023 `+0.002397`, 2024 `-0.003559`, 2025 `+0.004590` | `+0.021681` | `0.0144` | `0.002337` |
| 5 | 3 | last | `close_vwap_neg` | 2021 `-0.000636`, 2022 `-0.000476`, 2023 `-0.002441`, 2024 `+0.004879`, 2025 `+0.007637` | `+0.009260` | `0.0187` | `0.001008` |

### 43.5 观察

第一，horizon 诊断否定了“市场没有右尾”的解释。3/5/10 日的真实 top20 都很厚，2026 的 top20 甚至高于多数历史年份。

第二，稳定土壤偏向 3 日，而非 5 日。最好的稳定方向全部集中在 3 日 horizon；5 日和 10 日右尾更厚，但候选方向年度稳定性更差。

第三，动量不是最稳右尾入口。`low_vol_uptrend` 在 10 日 forward 为正，但 2022/2023 dev 年份为负；第 39-42 节沿 right-tail/动量式 token 训练失败，与这个诊断一致。

第四，`close_vwap_neg` 的收益很薄。最优候选 2026 Top10 label 只有 `+0.8298%`，历史最低年份 2023 只有 `+0.0060%`。这不是可直接融入的策略，而是提示模型应转向短周期均值回归/吸筹结构，可能需要 QMT 日内 token 才能放大。

### 43.6 反事实分析

第一反事实：如果 5 日 horizon 是天然最优，5 日候选应在 leaderboard 领先。实际前 6 名几乎全是 3 日，说明之前一周持仓口径可能过长，吃掉了短反弹信号。

第二反事实：如果动量右尾是主线，`rank_ret20_pos` 或 `liquid_momentum` 应稳定靠前。实际靠前的是 `close_vwap_neg` 和 `rank_amount_pos`，说明短期可预测性更像“弱收盘/VWAP 折价后的微结构回归”。

第三反事实：如果日线特征已经足够厚，最优候选的 Top10 label 应明显超过交易成本和随机裁判。实际信号很薄，说明只用日线可能只能看到影子，真正可交易信息可能在日内路径。

第四反事实：如果第 42 节 forward 失败是模型训练问题，本轮手工候选不应稳定为正。实际有薄正信号，说明方向不是全错，但训练目标需要贴近 3 日短反弹，而不是 5 日右尾分类。

### 43.7 判定

`multi_horizon_label_surface_has_stable_candidate_needs_model_probe`。

本轮没有可融入策略，但找到新的研究方向：3 日 horizon 的 `close_vwap_neg` 短反弹土壤最稳。下一轮做 `cross_sectional_close_vwap_reversion_replay_v1`：先对这个信号做真实交易回放、随机基准、去掉最佳调仓和年度稳定性验证。如果回放仍薄或过不了随机，再考虑引入 QMT 1m/5m 日内 token 去建模“收盘弱但承接强”的 intraday world model。

## 44. cross_sectional_close_vwap_reversion_replay_v1

### 44.1 假设

第 43 节只看标签面，发现 3 日 horizon 的 `close_vwap_neg` 方向最稳，但标签均值很薄。本轮必须做真实交易回放：加入交易成本、涨跌停/不可交易过滤、随机同池裁判、去掉最佳调仓，并严格区分 dev 2021-2025 与 forward 2026。

本轮还做一个防未来函数修正：初版回放曾把 dev 最后一笔持仓延伸到 2026，导致 dev 年度收益里出现 `2026`。该版本已废弃。修正版回放强制 `end_idx <= end`，本节只采用修正版结果。

### 44.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_close_vwap_reversion_replay_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_close_vwap_reversion_replay_v1_summary.json
script sha256:7498cef4aa8b39729dbf6a0502debd67f1a4663c74aa2588743db9225397563a
summary sha256:ac94db4e88df71c3055cabee65b1f68af207bdc83956e6f67865699e5dbc357c
```

扫描 interval `3/5`，view `last/mean5/mean20`，variant：`close_vwap_neg`、`close_vwap_neg_rank_amount_pos`、`close_vwap_neg_low_range`、`close_vwap_neg_low_range_liquid`。每个组合 Top10 等权，使用可变 interval 真实回放。随机同池 Top10 跑 `300` 次。

随机基准：

| interval | split | sessions | random p50 | random p95 | random max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | dev 2021-2025 | `404` | `0.347300x` | `0.563450x` | `0.907496x` |
| 3 | forward 2026 | `40` | `0.943043x` | `1.174344x` | `1.376153x` |
| 5 | dev 2021-2025 | `243` | `0.434607x` | `0.729503x` | `1.072680x` |
| 5 | forward 2026 | `24` | `0.967197x` | `1.200366x` | `1.344797x` |

### 44.3 结果

最稳历史组合是 `interval=5, view=mean20, close_vwap_neg_low_range`，五年全正，但 forward 太薄且低于随机 p95。

| interval | view | variant | dev multiple | dev all years positive | dev max DD | dev remove best 3 | forward multiple | forward remove best 3 | forward beats random p95 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 5 | mean20 | `close_vwap_neg_low_range` | `2.502093x` | 是 | `-11.04%` | `2.040652x` | `1.034648x` | `0.860097x` | 否 |
| 3 | mean20 | `close_vwap_neg_low_range` | `2.340959x` | 否 | `-14.14%` | `2.096094x` | `1.041502x` | `0.897489x` | 否 |
| 5 | last | `close_vwap_neg` | `2.276243x` | 否 | `-27.65%` | `1.761059x` | `1.126369x` | `0.923896x` | 否 |
| 5 | mean20 | `close_vwap_neg` | `2.128314x` | 否 | `-27.77%` | `1.631467x` | `1.147719x` | `0.938801x` | 否 |
| 3 | mean20 | `close_vwap_neg` | `1.989259x` | 否 | `-29.48%` | `1.684620x` | `1.193106x` | `1.039958x` | 是 |

最稳组合年度收益：2021 `+11.50%`，2022 `+3.35%`，2023 `+14.02%`，2024 `+37.09%`，2025 `+38.91%`。这条线很干净，但 forward 只有 `+3.46%`，并且去掉最佳 3 次调仓后低于 1。

forward 最强组合是 `interval=3, view=mean20, close_vwap_neg`：forward `1.193106x`，超过 3 日随机 p95 `1.174344x`，remove best 3 后 `1.039958x`；但 dev 的 2022 `-8.58%`、2023 `-2.68%`，五年不全正，且 dev 最大回撤 `-29.48%`。

### 44.4 观察

第一，`close_vwap_neg` 确实不是纯噪声。多个组合 dev 都明显超过随机同池，说明第 43 节的标签面土壤能转成交易回放上的正收益。

第二，收益厚度仍远不足。最稳组合五年只有 `2.50x`，远低于用户要求的几十倍到 100 倍；forward 也低于随机 p95。

第三，稳定性和 forward 厚度互相冲突。历史五年全正的低振幅版本 forward 太薄；forward 最强的纯 `close_vwap_neg` 版本历史 2022/2023 为负。这不是简单调参能解决的形态。

第四，日线信号更像“承接影子”。`close_vwap_neg_low_range` 历史更稳，说明弱收盘但低振幅更安全；纯 `close_vwap_neg` forward 更强，说明 2026 的短反弹可能来自更剧烈的尾盘折价/次日修复。仅靠日线无法判断这是恐慌下跌还是有资金承接。

### 44.5 反事实分析

第一反事实：如果第 43 节只是标签面幻觉，真实回放应接近随机或亏损。实际 dev 多个组合超过随机，说明方向有真实土壤。

第二反事实：如果该方向已经足够交易化，最稳组合应在 forward 超过随机 p95 且 remove best 3 后为正。实际不满足，不能融入。

第三反事实：如果只靠日线能够区分“弱收盘后的反弹”和“弱收盘后的继续下跌”，历史全正组合不应 forward 变薄，forward 最强组合不应历史两年为负。实际出现分裂，说明缺少日内路径信息。

第四反事实：如果下一步继续在日线 close_vwap 上调权重，会在“历史稳”和“forward 强”之间来回摆动。更合理的方向是引入 QMT 5m/1m，识别收盘弱但盘中承接强、尾盘被动压价、次日开盘修复概率更高的状态。

### 44.6 判定

`close_vwap_reversion_replay_not_enough`。

本轮没有可融入策略，但确认了一个新的可交易土壤：日线 `close_vwap_neg` 短反弹可用但很薄。下一轮转向 `qmt_intraday_close_vwap_reversion_microstructure_probe_v1`：在 QMT 5m/1m 数据上诊断这些候选股票的日内承接结构，例如尾盘弱收、全天 VWAP 折价、午后回升、收盘前放量/缩量、次日开盘跳空等，验证是否能把薄日线信号放大成更强的短周期 alpha。

## 45. qmt_intraday_close_vwap_reversion_microstructure_probe_v1

### 45.1 假设

第 44 节显示，日线 `close_vwap_neg` 短反弹方向是真土壤，但收益很薄，并且“历史稳”和“forward 强”之间出现分裂。一个自然反事实是：日线只能看到收盘相对 VWAP 偏弱，却看不到盘中承接。如果 QMT 5m 能识别“弱收盘但有承接”的形态，应该能在 2026 forward 上打败纯日线 baseline。

本轮只做 forward 微结构 probe，不作为策略验收；目的是判断 QMT 5m 日内 token 是否值得扩展到 dev 2021-2025 大规模验证。

### 45.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_intraday_close_vwap_reversion_microstructure_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_intraday_close_vwap_reversion_microstructure_probe_v1_summary.json
script sha256:f6eb2870d2f37ac4180cf0a4acf3a04f6a7c280c9e7384cf32739d2c8ba1e1ec
summary sha256:690bfcc0f42fe184d31c00ff2269f4d71bcd923ebc095b6efa66ced7b362a226
```

本轮沿用第 44 节的 forward 候选池和可变 interval 回放，使用 QMT 5m 数据，缓存目录为 `.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_minute_cache_v1/5m`。候选池为每个 session 的日线 `mean20 close_vwap_neg` 前 60，只在这个池内比较日线 baseline 与日内微结构重排。

日内特征包括：`intraday_vwap_support`、`tail_strength_30m`、`tail_volume_share_30m`、`low_to_close_recovery`、`high_to_close_fade`、`morning_sell_pressure`、`pm_low_to_close_recovery`、`open_to_close_intraday`。变体包括 `qmt_tail_recovery`、`qmt_vwap_absorption`、`qmt_low_to_close_support`、`qmt_late_volume_support`、`qmt_intraday_composite`。

数据覆盖率很好，不是本轮失败原因：

| interval | sessions | requested | available | missing |
| ---: | ---: | ---: | ---: | ---: |
| 3 | `40` | `2400` | `2400` | `0` |
| 5 | `24` | `1440` | `1440` | `0` |

### 45.3 结果

QMT 5m 日内重排没有打败纯日线 baseline。两个 interval 最优都是 `daily_close_vwap_neg` 本身。

| interval | variant | multiple | total return | max DD | remove best 3 | beats random p95 | beats daily baseline |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 3 | `daily_close_vwap_neg` | `1.193106x` | `+19.31%` | `-14.96%` | `1.039958x` | 是 | 否 |
| 5 | `daily_close_vwap_neg` | `1.147719x` | `+14.77%` | `-17.32%` | `0.938801x` | 是 | 否 |
| 3 | `qmt_late_volume_support` | `1.078332x` | `+7.83%` | `-16.70%` | `0.928798x` | 是 | 否 |
| 3 | `qmt_tail_recovery` | `1.064310x` | `+6.43%` | `-22.09%` | `0.899592x` | 否 | 否 |
| 3 | `qmt_vwap_absorption` | `0.970401x` | `-2.96%` | `-20.88%` | `0.854037x` | 否 | 否 |
| 5 | `qmt_intraday_composite` | `0.827338x` | `-17.27%` | `-22.12%` | `0.722446x` | 否 | 否 |

forward 随机同池基准：

| interval | random p50 | random p95 | random max |
| ---: | ---: | ---: | ---: |
| 3 | `0.908778x` | `1.078071x` | `1.159947x` |
| 5 | `0.950058x` | `1.109389x` | `1.244658x` |

### 45.4 观察

第一，覆盖率排除了数据缺失解释。3 日与 5 日候选池 QMT 5m 数据均为 `100%` 可用，失败不是因为分钟数据没拉到。

第二，纯日线 baseline 已经很强。3 日 baseline `1.193106x` 超过随机 p95 和随机 max；5 日 baseline `1.147719x` 也超过随机 p95。但这是第 44 节已知的 forward 现象，不代表 dev 稳定。

第三，日内微结构特征反而削弱收益。`qmt_late_volume_support` 还能保留部分收益，但仍低于 baseline；`qmt_vwap_absorption`、`qmt_low_to_close_support`、`qmt_intraday_composite` 多数变成负收益或 remove best 后明显低于 1。

第四，当前 5m 承接定义可能没有刻画真正有用的信息。简单的尾盘回升、VWAP 支撑、低点到收盘修复、尾盘量占比，并没有解释这条日线短反弹的 forward 收益来源。

### 45.5 反事实分析

第一反事实：如果日内承接是关键缺失变量，至少一个 QMT 5m 变体应超过 `daily_close_vwap_neg`。实际没有任何一个变体 beat baseline，说明这个假设在当前定义下失败。

第二反事实：如果 QMT 5m 特征只是轻微噪声，结果应接近 baseline。实际多个变体明显变差，说明这些手工微结构特征可能在选错方向，或者把日线有效信号过滤掉了。

第三反事实：如果 2026 forward 的 `close_vwap_neg` 收益来自“收盘弱但尾盘承接强”，`tail_recovery` 和 `vwap_absorption` 应更强。实际它们都弱于 baseline，说明收益更可能来自横截面均值回归状态本身，而不是单日 5m 承接。

第四反事实：如果继续扩 QMT 5m 手工特征就能解决，应先看到某个简单特征有边际增益。实际没有边际增益，不应继续在手工日内规则上加复杂度。

### 45.6 判定

`qmt_intraday_close_vwap_microstructure_not_enough`。

本轮没有可融入策略。结论是：QMT 5m 手工微结构没有放大第 44 节的日线短反弹土壤。下一轮不继续手写日内承接规则，而是回到横截面模型本身：做 `cross_sectional_close_vwap_regime_model_probe_v1`，围绕 `close_vwap_neg` 方向构造更明确的 regime/context 特征，例如市场宽度、过去 20 日折价分布、低波动状态、成交排名、行业/概念拥挤代理，验证能否解释“历史稳版本 forward 太薄、forward 强版本历史不稳”的分裂。

## 46. cross_sectional_close_vwap_regime_model_probe_v1

### 46.1 假设

第 44/45 节确认 `close_vwap_neg` 是真实但很薄的短反弹土壤：纯日线 3 日版本在 2026 forward 有 `1.193106x`，但历史 2022/2023 不稳；QMT 5m 手工微结构没有放大这个土壤。一个自然反事实是：收益并不来自单日承接规则，而来自更高层横截面 context，例如市场宽度、过去 20 日折价分布、低波动状态、成交排名和候选池内部位置。

本轮训练一个很小的 MLP，只在 `mean20 close_vwap_neg` 前 120 候选池内重排，验证模型能否区分“折价后反弹”和“折价后继续弱”。这是 `.tmp` probe，不进入正式代码，也不生成可融入策略。

### 46.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_close_vwap_regime_model_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_close_vwap_regime_model_probe_v1_summary.json
script sha256:adc1bbfe6b0d25c5a37b240cb8439dc9c4e959cfa3a17a86a9c72ae3b53fba09
summary sha256:f49aaa3cc02a317f7f6175e3c5c6600fe6225635d7dcefe551ff586cc3d16c5c
```

口径：interval `3`，Top10 等权，候选池为每个 session 的 `mean20 close_vwap_neg` 前 120。训练集 `2021-2024`，验证集 `2025`，dev 汇总 `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。输入包括 last/mean5/mean20 个股 token、横截面 context、候选内 rank 和日线 `close_vwap_neg` 分数。目标为 session 内连续 rank，模型为两层小 MLP，best epoch 为 `5`。

### 46.3 结果

模型没有打败纯日线 baseline，且明显破坏 forward。

| 口径 | final multiple | total return | 年度表现 | max DD | remove best 3 | rank IC | Top10 label |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| train 2021-2024 | `1.595924x` | `+59.59%` | 2022/2023 为负 | `-18.78%` | `1.531469x` | `0.0898` | `+0.002075` |
| valid 2025 | `1.319350x` | `+31.94%` | 2025 为正 | `-8.56%` | `1.263804x` | `0.0608` | `+0.004225` |
| dev 2021-2025 | `2.052754x` | `+105.28%` | 2022 `-4.03%`，2023 `-1.52%` | `-18.78%` | `2.070666x` | `0.0840` | `+0.002506` |
| forward 2026 | `0.921588x` | `-7.84%` | 2026 为负 | `-9.98%` | `0.870440x` | `0.0383` | `-0.001442` |

对照很关键：同一 forward 区间，纯日线 `mean20 close_vwap_neg` baseline 为 `1.193106x`，total return `+19.31%`，max DD `-14.96%`，remove best 3 为 `1.039958x`。forward 候选池随机 Top10 的 p95 为 `1.056447x`，max 为 `1.223382x`。小 MLP 的 `0.921588x` 不仅低于 baseline，也低于随机 p50 `0.875909x` 附近后的可接受区间上沿。

### 46.4 观察

第一，`close_vwap_neg` 土壤仍然存在，但模型重排会破坏它。baseline 在 forward 是正的，小 MLP 重排后变成负收益，说明模型没有学到可迁移的反弹条件。

第二，valid 2025 强并不可靠。模型在 2025 有 `1.319350x`，但 2026 立刻转负；这比“训练集好、验证集差”的普通过拟合更危险，因为它会误导 early stopping。

第三，rank IC 与 Top10 交易价值分裂。forward rank IC 仍有 `0.0383`，但 Top10 label 为 `-0.001442`，说明模型可能在候选池中部有一点排序相关性，却把最需要交易的头部选错。

第四，context 特征学到的是历史状态而不是稳健机制。2022/2023 dev 年份仍为负，说明模型没有修复第 44 节已知的年度分裂；2026 又把 baseline 的正收益打坏，说明继续加 MLP/Transformer 容量很可能只会更精细地拟合历史 context。

### 46.5 反事实分析

第一反事实：如果横截面 context 是 `close_vwap_neg` 的关键缺失变量，小 MLP 应至少在 forward 超过纯日线 baseline。实际 forward 从 `1.193106x` 降到 `0.921588x`，直接否定该假设。

第二反事实：如果模型只是轻微噪声，forward 应接近 baseline 或随机 p95 附近。实际 remove best 3 为 `0.870440x`，说明模型头部选择有系统性错误。

第三反事实：如果 2025 valid 能代表近期市场，2026 不应立刻转负。实际 valid 到 forward 的翻转说明 validation 单年不足以作为这个方向的晋级依据。

第四反事实：如果继续在 close-vwap 上堆网络能解决问题，至少应先看到小模型有边际增益。实际小模型没有边际增益，因此更合理的动作是回到结构诊断，而不是换更大模型。

### 46.6 判定

`close_vwap_regime_model_not_enough`。

本轮没有可融入策略。结论是：`close_vwap_neg` 是薄土壤，但小 MLP context 重排不可迁移。下一轮不继续在同一候选池上加复杂模型，而是做 `cross_sectional_residual_reversal_structure_scan_v1`：把 `close_vwap_neg` 放到流动性、波动、中期动量、短期拥挤等横截面桶内做 residual/rank 诊断，检查这条土壤是否来自某种稳定的横截面残差结构。如果 residual 版本能修复 2022/2023 且 forward 不被打坏，再考虑 DeepSets/SetTransformer；如果不能，就离开 close-vwap 土壤。

## 47. cross_sectional_residual_reversal_structure_scan_v1

### 47.1 假设

第 46 节说明，小 MLP 在 `close_vwap_neg` 候选池内做 context 重排会破坏 forward。继续加模型容量之前，必须先回答一个更基础的问题：`close_vwap_neg` 的有效性是否来自某种稳定的横截面残差结构。

如果这条短反弹土壤只是全市场均值回归影子，那么在流动性、波动、中期动量、短期反转、拥挤状态桶内做 residual/rank 后不会变厚；如果它真正来自某类结构，桶内残差化应修复 2022/2023 的负年份，并且 forward 不低于原始 baseline。

### 47.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_residual_reversal_structure_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_residual_reversal_structure_scan_v1_summary.json
script sha256:551237cb283bca34cc1b8ec20141df54d6a17872f4f2d28932d601403dd4ae38
summary sha256:196ab65e48a2c351529f423f21a8f5e9105060eb5db1e1d7c10828832d8ef86b
```

本轮复用第 44 节真实回放、交易成本、涨跌停/不可交易过滤、remove best 和随机同池裁判。扫描 interval `3/5`，view `last/mean5/mean20`，Top10 等权，pool500。dev 为 `2021-2025`，forward 为 `2026-01-01` 到 `2026-07-10`，并沿用已修复的边界约束，dev 持仓不跨入 forward。

残差桶包括：流动性 rank、波动 range、中期动量 ret20、短期反转 ret5、拥挤 proxy。变体包括原始 `global_close_vwap_neg`、各单桶 residual、multi-bucket residual mean、residual 加 low range 和 liquid low range。

随机同池基准与第 44 节一致：

| interval | split | sessions | random p50 | random p95 | random max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | dev 2021-2025 | `404` | `0.347300x` | `0.563450x` | `0.907496x` |
| 3 | forward 2026 | `40` | `0.943043x` | `1.174344x` | `1.376153x` |
| 5 | dev 2021-2025 | `243` | `0.434607x` | `0.729503x` | `1.072680x` |
| 5 | forward 2026 | `24` | `0.967197x` | `1.200366x` | `1.344797x` |

### 47.3 结果

leaderboard 前排仍然是原始 `global_close_vwap_neg`，残差化没有改善主矛盾。

| rank | interval | view | variant | dev multiple | dev min year | dev all years positive | forward multiple | forward remove best 3 | forward beats random p95 |
| ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 5 | last | `global_close_vwap_neg` | `2.276243x` | `-8.65%` | 否 | `1.126369x` | `0.923896x` | 否 |
| 2 | 5 | mean20 | `global_close_vwap_neg` | `2.128314x` | `-7.50%` | 否 | `1.147719x` | `0.938801x` | 否 |
| 3 | 5 | mean5 | `global_close_vwap_neg` | `2.151835x` | `-12.76%` | 否 | `1.149779x` | `0.941501x` | 否 |
| 4 | 3 | mean20 | `global_close_vwap_neg` | `1.989259x` | `-8.58%` | 否 | `1.193106x` | `1.039958x` | 是 |
| 7 | 5 | last | `liquidity_bucket_close_vwap_resid` | `1.421176x` | `-13.76%` | 否 | `1.088654x` | `0.877005x` | 否 |
| 8 | 5 | mean20 | `close_vwap_resid_low_range` | `1.219495x` | `-11.21%` | 否 | `1.080057x` | `0.909144x` | 否 |
| 9 | 3 | mean20 | `close_vwap_resid_liquid_low_range` | `1.304641x` | `-13.39%` | 否 | `1.166978x` | `0.997133x` | 否 |

各 residual 家族的最好结果也没有达到候选要求：

| variant | best interval/view | dev multiple | dev min year | forward multiple | forward remove best 3 | forward beats random p95 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `momentum20_bucket_close_vwap_resid` | 3 / mean20 | `1.033561x` | `-18.68%` | `1.279859x` | `1.085057x` | 是 |
| `crowding_bucket_close_vwap_resid` | 5 / mean5 | `1.246160x` | `-37.73%` | `1.240197x` | `0.971244x` | 是 |
| `reversal5_bucket_close_vwap_resid` | 5 / mean20 | `1.348914x` | `-20.22%` | `1.229915x` | `0.932835x` | 是 |
| `multi_bucket_close_vwap_resid_mean` | 5 / mean20 | `0.630310x` | `-46.60%` | `1.167172x` | `0.905362x` | 否 |
| `volatility_bucket_close_vwap_resid` | 5 / last | `0.481773x` | `-45.15%` | `1.124713x` | `0.854099x` | 否 |

有几个 residual 变体在 2026 forward 超过随机 p95，但开发期明显塌陷，尤其 `crowding`、`reversal5`、`momentum20` 的年度最差分别为 `-37.73%`、`-20.22%`、`-18.68%`。这不是可穿越的结构，只是 2026 局部状态对桶内残差的一次奖励。

### 47.4 观察

第一，原始 `close_vwap_neg` 仍是这组扫描里最强的开发期结构。残差化后 dev multiple 普遍从 2x 附近掉到 0.48x-1.42x，说明桶内中心化削掉了主要边，而不是去除了噪声。

第二，residual 版本没有修复年度稳定性。所有候选都不是五年全正，且 residual 家族的最差年份通常比原始 baseline 更差。

第三，2026 forward 局部漂亮但不能晋级。`momentum20_bucket_close_vwap_resid` forward `1.279859x` 且 remove best 3 `1.085057x`，但 dev 只有 `1.033561x`、最差年份 `-18.68%`；这类形态很像 regime 后验，不是鲁棒 alpha。

第四，`multi_bucket` 失败尤其重要。如果多个桶内 residual 的平均能表达稳定结构，它不应在 dev 只有 `0.630310x`。实际 multi-bucket 明显亏损，说明当前桶维度并没有捕捉到通用可迁移机制。

### 47.5 反事实分析

第一反事实：如果 `close_vwap_neg` 的核心来自流动性/波动/动量桶内错定价，单桶 residual 应在 dev 和 forward 同时优于原始 baseline。实际没有任何 residual 同时满足，原始分数仍占 leaderboard 前排。

第二反事实：如果 2022/2023 的亏损只是因为混入了不合适的横截面状态，桶内 residual 应明显修复这些年份。实际 residual 家族仍有明显负年份，部分更差。

第三反事实：如果 2026 的 residual 强势是规律，开发期不应接近 1x 或低于 1x。实际 `momentum20`、`crowding`、`reversal5` 等 forward 强项的 dev 很薄或很差，说明它们不能作为下一轮训练标签。

第四反事实：如果继续围绕 close-vwap 做 DeepSets/SetTransformer 会有效，至少应先看到 residual 或手工结构有稳定边际增益。实际结构诊断否定了这个前提，因此不应继续在这个土壤上堆模型。

### 47.6 判定

`residual_reversal_structure_not_enough`。

本轮没有可融入策略，也不建议进入 `close_vwap_neg` 的 SetTransformer/DeepSets 训练。第 43-47 节形成一致证据链：日线 `close_vwap_neg` 是薄的短反弹影子，但不能被 QMT 5m 手工微结构、MLP context 或横截面 residual 稳定放大。

下一轮应离开 close-vwap 土壤，转向更高层、低相关的候选来源。优先方向是 `cross_sectional_market_state_path_tail_diagnostic_v1`：不再从单个反转因子出发，而是按市场宽度、离散度、趋势强度、短期拥挤惩罚、真实右尾厚度分层，寻找“哪些市场状态天然产生可交易 Top10/Top20 右尾”。如果能找到稳定状态，再训练小型 market-state conditioned ranker；如果找不到，说明模型训练不是瓶颈，候选生成土壤仍需继续换。

## 48. cross_sectional_market_state_path_tail_diagnostic_v1

### 48.1 假设

第 43-47 节说明，`close_vwap_neg` 是薄土壤，但不能被日内微结构、MLP context 或横截面 residual 稳定放大。下一步不应继续从单一反转因子出发，而应先问一个更高层问题：A 股一周/三日真实右尾在什么市场状态下自然变厚，T 日可见的候选土壤能否捕获这种厚尾。

如果市场状态确实是关键，按宽度、趋势、离散度、拥挤、安静趋势等 T 日可见状态分桶后，某些候选因子应该在 dev 和 forward 同时变厚，并且年度最差不为负。如果只有 oracle 右尾厚、候选土壤抓不住，说明问题仍在候选生成，而不是模型路由。

### 48.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_market_state_path_tail_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_market_state_path_tail_diagnostic_v1_summary.json
script sha256:52a1435e5b67be0afa3be1639143899d0f0b79e2f03744a295cb452ed8bcb11f
summary sha256:89db0db26366ed9d1beffd418908bbcb0e34dc259c1583fac93751a3ba14bec4
```

本轮只做诊断，不训练模型，不进入正式账户。数据只使用本地 Qlib/QMT 日线可得字段及其派生特征：`open/high/low/close/volume/vwap`、`ret5/ret20/ret60`、横截面宽度、离散度、close-vwap 分布和拥挤代理。

重要审计修正：初版 forward 曾按 2026 自身分位切状态桶。虽然不使用未来收益，但会用完整 forward 状态分布。修正版改为只用 dev `2021-2025` 分位阈值切 forward 桶，本节只采用修正版结果。

扫描 interval `3/5`，pool500，Top10/Top20 标签口径为 T 日信号、T+1 open 入场、T+interval+1 open 出场。状态特征包括 `market_ret20_median`、`market_ret60_median`、`breadth_ret5_pos`、`breadth_ret20_pos`、`near_high20_rate`、`disp_ret5`、`disp_ret20`、`tail_spread20`、`crowding_hot_vol_rate`、`close_vwap_neg_rate`、`quiet_trend_rate`。候选土壤包括 `rank_amount_pos`、`ret20_momentum`、`pullback_from_high20`、`low_vol_uptrend`、`close_vwap_neg`、`anti_crowding_quiet_trend`、`elastic_low_liquidity_proxy`。

### 48.3 整体结果

真实右尾非常厚，但 T 日可见候选抓不住。

| interval | split | sessions | oracle Top20 | oracle p95 | oracle Top20-bottom20 spread | best candidate Top20 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3 | dev 2021-2025 | `404` | `+14.9809%` | `+8.6389%` | `+25.7202%` | `close_vwap_neg +0.1782%` |
| 3 | forward 2026 | `40` | `+18.8419%` | `+11.7836%` | `+31.3164%` | `rank_amount_pos +0.5394%` |
| 5 | dev 2021-2025 | `243` | `+19.1150%` | `+11.0755%` | `+32.2434%` | `close_vwap_neg +0.3254%` |
| 5 | forward 2026 | `24` | `+25.6831%` | `+15.7668%` | `+41.3294%` | `low_vol_uptrend +1.0940%` |

这组结果很关键：市场并不是没有右尾。3/5 日未来真实 top20 都很厚，forward 甚至比 dev 更厚。但当前这些 T 日可见候选土壤的 Top20 只有千分位到 1% 左右，和 oracle 差两个数量级。

### 48.4 状态桶 leaderboard

按 dev 阈值切 forward 桶后，leaderboard 前排如下：

| interval | state | bucket | candidate | dev sessions | dev Top20 | dev worst year | forward sessions | forward Top20 | 判读 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | `breadth_ret5_pos` | 0 | `pullback_from_high20` | `60` | `+0.8253%` | `-0.8381%` | `5` | `+4.5835%` | forward 极强但样本少、dev 年度不稳 |
| 3 | `breadth_ret5_pos` | 0 | `pullback_from_high20` | `101` | `+0.6844%` | `-0.2555%` | `7` | `+3.1712%` | 同方向但 dev 不够厚 |
| 5 | `near_high20_rate` | 0 | `close_vwap_neg` | `34` | `+0.6022%` | `-0.4468%` | `5` | `+2.3342%` | 样本偏少 |
| 5 | `breadth_ret5_pos` | 3 | `close_vwap_neg` | `61` | `+0.6485%` | `-0.5102%` | `9` | `+1.3999%` | dev 不稳 |
| 5 | `market_ret60_median` | 3 | `low_vol_uptrend` | `61` | `+1.0238%` | `-0.7817%` | `14` | `+1.9850%` | dev 达 1%，但年度最差为负 |

只有两个状态桶的 dev 最差年份为正：

| interval | state | bucket | candidate | dev Top20 | dev worst year | forward sessions | forward Top20 | 判读 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3 | `market_ret60_median` | 3 | `low_vol_uptrend` | `+0.5858%` | `+0.2785%` | `22` | `+1.0009%` | 稳但太薄，低于强基线 |
| 5 | `close_vwap_neg_rate` | 0 | `close_vwap_neg` | `+0.9781%` | `+0.3032%` | `24` | `+0.5033%` | 年度稳定但 forward 衰减，仍太薄 |

### 48.5 观察

第一，真实右尾不是瓶颈。5 日 dev oracle Top20 `+19.1150%`，forward `+25.6831%`，说明市场短期存在非常厚的可选右尾。

第二，当前 T 日候选土壤捕获能力太弱。整体最好的候选 Top20 只有 `close_vwap_neg` dev `+0.3254%` 或 forward `low_vol_uptrend +1.0940%`，远低于 path sequence 强基线，也远低于用户要求需要的收益斜率。

第三，状态分桶能解释 2026，但不能形成穿越规律。低短期宽度里的 `pullback_from_high20` forward 很强，但 dev 最差年份为负，且 forward 样本只有 5-7 个 session，不可晋级。

第四，少数年度稳定状态太薄。`market_ret60_median` 高桶的 `low_vol_uptrend` 三日 Top20 dev `+0.5858%`、forward `+1.0009%`，方向一致但厚度不够；`close_vwap_neg_rate` 低桶的 close-vwap 五日 dev `+0.9781%`、forward `+0.5033%`，forward 衰减。

第五，模型训练不是当前主要瓶颈。若 oracle 与候选之间差距只有一点，可以训练模型挖掘边际；但这里差距是 `+19%` 对 `+0.3%`，说明当前候选生成还没有贴近右尾事件前兆。

### 48.6 反事实分析

第一反事实：如果市场状态本身足以生成可交易 alpha，某些状态桶内的候选 Top20 应在 dev/forward 都明显变厚且年度最差为正。实际最厚的状态桶年度不稳，年度稳定的状态桶太薄。

第二反事实：如果 2026 的强表现是可迁移规律，dev 同状态应至少接近 2026 的一半。实际 `breadth_ret5_pos` 低桶的 `pullback_from_high20` forward 到 `+4.58%`，dev 只有 `+0.83%` 且最差年为负，更像 2026 局部状态。

第三反事实：如果继续训练 state-conditioned ranker 值得做，输入候选至少应包含能稳定接近强基线的土壤。实际候选 Top20 整体不达标，因此训练 ranker 会学到弱筛子，而不是右尾机制。

第四反事实：如果市场不可预测，oracle 右尾厚度和 top-bottom spread 不应这么大。实际 oracle 右尾极厚，说明“有金子”成立；失败发生在“当前可见特征和候选土壤没有把金子筛出来”。

### 48.7 判定

`market_state_tail_diagnostic_not_enough`。

本轮没有可融入策略，也不进入 market-state conditioned ranker。结论是：市场短期右尾非常厚，但当前日线手工候选和状态桶无法稳定捕获。下一轮不应继续路由这些弱候选，而应转向 `right_tail_event_precursor_mining_v1`：从真实右尾样本倒查 T 日以前 20-60 日的共同前兆，重点看连续压缩后放量、相对强弱拐点、回踩不破、近高吸收、低波动上行、量价分歧等事件型路径，再用严格 walk-forward 验证这些前兆是否能在 T 日可见时生成 Top10/Top20 候选。

## 49. right_tail_event_precursor_mining_v1

### 49.1 假设

第 48 节说明，市场真实右尾非常厚，但状态桶和几个手工候选抓不住。下一步必须从真实右尾样本反查 T 日以前的共同前兆，而不是继续路由弱候选。

本轮做两件事：第一，按未来真实 Top20 右尾样本和 Bottom20 样本，对比 T 日以前 5/20/60 日路径、量能、近高、压缩、VWAP 和低点修复特征；第二，把这些观察写成几组事件型候选分数，直接验证 Top10/Top20 标签能否变厚。

### 49.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_right_tail_event_precursor_mining_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_event_precursor_mining_v1_summary.json
script sha256:93972165ccb2e0adad31030629b87f3e02961709a2c3ec06c9a5c7398a4aa5ec
summary sha256:c2d1b1861add133c13d7693499a4ae7ac0f04364cba199856b507797b50b62fc
```

扫描 interval `3/5`，pool500，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。每个 session 只使用 T 日及以前的日线 OHLCV/VWAP。未来收益只用于标记右尾/左尾和评估候选，不进入特征。

反查前兆包括：`ret5/20/60`、`ret20_minus_ret60`、`pullback5_after_ret20`、`near_high20/60`、`range5/20`、`compression_5_vs_20`、`vol5_vs_20`、`vol20_vs_60`、`close_vwap5/20`、`amount_rank`、`low_to_close5`、`high_to_close5`。

候选公式包括：`compression_uptrend`、`pullback_near_high_absorption`、`volume_dry_uptrend`、`volume_expand_reclaim`、`quiet_ret20_leader`、`low_to_close_recovery_trend`、`anti_chase_near_high`、`right_tail_precursor_composite`。

### 49.3 结果

候选公式没有形成可用 Top20，最好的也很薄且年度不稳。

| interval | candidate | dev Top10 | dev Top20 | dev worst year Top20 | forward Top10 | forward Top20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | `anti_chase_near_high` | `+0.1117%` | `+0.1472%` | `-0.4906%` | `-0.3525%` | `+0.4167%` | 最优但太薄，Top10 forward 反向 |
| 3 | `right_tail_precursor_composite` | `-0.0580%` | `+0.0116%` | `-0.2494%` | `+0.3688%` | `+0.2642%` | dev 近零 |
| 3 | `volume_dry_uptrend` | `+0.1691%` | `+0.0778%` | `-0.1271%` | `+0.0597%` | `+0.0960%` | 太薄 |
| 3 | `anti_chase_near_high` | `+0.2030%` | `+0.1825%` | `-0.1133%` | `-0.3563%` | `-0.1105%` | forward 失败 |
| 5 | `quiet_ret20_leader` | `+0.3326%` | `+0.1936%` | `-0.5130%` | `-0.1089%` | `+0.0265%` | 太薄且不稳 |

所有候选的 dev worst year Top20 都为负，没有一个满足年度稳定。

### 49.4 右尾前兆反查

虽然候选公式失败，但右尾样本的共同前兆非常清楚：未来右尾不是“刚启动低位股”，更像已经有中期强趋势和量能抬升的延续状态。

| interval | precursor | dev tail-pool | dev tail-bottom | forward tail-pool | forward tail-bottom | 解释 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3 | `ret60` | `+0.1700` | `-0.1476` | `+0.2893` | `-0.1502` | 右尾显著来自 60 日强势 |
| 5 | `ret60` | `+0.1571` | `-0.1834` | `+0.2742` | `-0.1448` | 同上，forward 更强 |
| 3 | `ret20` | `+0.0841` | `-0.0937` | `+0.1303` | `-0.0765` | 20 日强势也稳定 |
| 5 | `ret20` | `+0.0770` | `-0.1107` | `+0.1161` | `-0.0842` | 同上 |
| 3 | `vol20_vs_60` | `+0.0981` | `-0.0519` | `+0.0925` | `-0.0206` | 中期量能抬升明显 |
| 5 | `vol20_vs_60` | `+0.0812` | `-0.0706` | `+0.0821` | `-0.0370` | 同上 |
| 3 | `vol5_vs_20` | `+0.0747` | `-0.0804` | `+0.0726` | `-0.0444` | 短期量能也偏强 |
| 3 | `amount_rank` | `+0.0554` | `-0.0783` | `+0.0825` | `-0.0544` | 右尾不是低流动性孤立弹性 |
| 3 | `pullback5_after_ret20` | `+0.0646` | `-0.0554` | `+0.0874` | `-0.0481` | 强趋势中有短回踩/休整成分 |

注意 `ret20_minus_ret60` 为负，说明右尾更像 60 日强趋势里的 20 日阶段，而不是刚刚短期相对长期加速。这个结果反过来解释了候选公式失败：我们把“反追高、压缩、回踩、低量”加得太重，抵消了最清晰的强趋势/放量信号。

### 49.5 观察

第一，真实右尾前兆的主轴是中期强趋势 + 量能确认，不是低位弹性或单纯均值回归。这与第 48 节 oracle 厚尾一致：市场有右尾，但右尾藏在趋势延续/资金抬升中。

第二，手写事件公式失败，不代表前兆不存在。反查对比很清楚，但线性组合时加入了太多防守项，例如 anti-chase、compression、dry volume，导致 Top20 被削薄。

第三，forward 对强趋势前兆更敏感。`ret60` 的 forward tail-pool 差达到 `+0.2893/+0.2742`，比 dev 更大，说明 2026 不是没有趋势，而是需要更直接地拥抱强趋势，而不是回避它。

第四，所有候选年度最差为负，不能进入 formal，也不能直接训练模型。下一轮应先验证“纯强趋势/量能延续”而非“强趋势叠加防追高”的简化反事实。

### 49.6 反事实分析

第一反事实：如果右尾来自低位反转，`ret60/ret20` 在右尾样本中不应显著高于池均值。实际它们稳定大幅为正，否定低位反转作为主轴。

第二反事实：如果压缩/防追高是关键，`compression_uptrend`、`anti_chase_near_high` 应显著变厚。实际它们 Top20 很薄，说明防守项把收益弹性削掉了。

第三反事实：如果前兆只是 2026 后验，dev 反查不应同向。实际 dev 和 forward 在 `ret60/ret20/vol20_vs_60/amount_rank` 上同向，说明这里有真实结构。

第四反事实：如果手写公式足以捕获结构，候选 Top20 应至少接近第 48 节状态桶里 1% 级别。实际多数只有千分位，说明需要重新定义候选，不是继续调当前公式权重。

### 49.7 判定

`right_tail_event_precursor_not_enough`。

本轮没有可融入策略，但给出了重要方向修正：右尾前兆不是“低位、低量、防追高”，而是“中期强趋势、量能抬升、一定短回踩后的延续”。下一轮做 `trend_volume_continuation_soil_v1`：去掉过多防守项，直接扫描 20/60 日强趋势、20/60 日量能抬升、流动性排名、短回踩不破、近高位置的少量组合，先看 Top10/20 标签是否能接近强基线；若仍不行，再考虑用小型 sequence/set 模型学习这种非线性交互。

## 50. trend_volume_continuation_soil_v1

### 50.1 假设

第 49 节反查显示，未来右尾样本在 T 日以前已经具备明显的 60 日/20 日强势、20/60 日量能抬升、短期量能增强和较高流动性。一个直接反事实是：第 49 节候选失败可能不是因为强趋势无效，而是因为我们把 anti-chase、compression、低量和回踩防守项加得太重，抵消了真正的收益弹性。

本轮因此去掉大部分防守惩罚，直接扫描强趋势 + 量能确认：如果右尾前兆能直接转成可交易 alpha，纯强趋势/量能延续分数应该在 dev 和 forward 同时变厚。

### 50.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_trend_volume_continuation_soil_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/trend_volume_continuation_soil_v1_summary.json
script sha256:55a2fc16ffabfdb1251d9d62fdbf6d054fd9ecc912d2eb84b21725a4f12ae111
summary sha256:1e1ecdef801ebb4cf5e502b0de78ecfa980728991848c79dadf7d06945a42c28
```

扫描 interval `3/5`，pool500，Top10/Top20，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。候选包括：`ret60_leader`、`ret20_ret60_leader`、`trend_volume20_confirm`、`trend_volume5_confirm`、`liquid_trend_leader`、`near_high_trend_volume`、`trend_pullback_light`、`trend_vwap_confirm`、`trend_volume_composite`。

所有特征只使用 T 日及以前日线 OHLCV/VWAP；未来收益只用于标签评估。

### 50.3 结果

直接追强趋势/量能延续在 dev 全部为负，forward 虽有局部转正但厚度很弱。

| interval | candidate | dev Top10 | dev Top20 | dev worst year Top20 | forward Top10 | forward Top20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | `trend_pullback_light` | `-0.3606%` | `-0.4429%` | `-0.9986%` | `+0.2388%` | `+0.1512%` | leaderboard 最优但 dev 为负 |
| 3 | `ret60_leader` | `-0.6947%` | `-0.6676%` | `-1.0899%` | `+0.0520%` | `+0.4197%` | 强趋势裸买 dev 负 |
| 3 | `ret20_ret60_leader` | `-0.9629%` | `-0.7557%` | `-1.3167%` | `+0.1818%` | `+0.1869%` | dev 负 |
| 3 | `liquid_trend_leader` | `-0.8674%` | `-0.6761%` | `-1.3545%` | `+0.3591%` | `+0.0493%` | dev 负 |
| 5 | `ret20_ret60_leader` | `-1.5048%` | `-1.1256%` | `-2.4959%` | `+0.0278%` | `+0.6032%` | forward 局部正，dev 很差 |
| 5 | `ret60_leader` | `-0.9587%` | `-0.9290%` | `-2.3568%` | `-0.1685%` | `+0.2909%` | dev 很差 |

整体对照仍显示 oracle 很厚：3 日 dev oracle Top20 `+14.9809%`，forward `+18.8419%`；5 日 dev oracle Top20 `+19.1150%`，forward `+25.6831%`。但强趋势候选不但没有接近 oracle，dev 还系统性为负。

年度拆分也很明确。3 日 `ret60_leader` 的 2021-2025 年度 Top20 分别为 `-0.4584%/-1.0899%/-0.5880%/-0.9709%/-0.2345%`，五年全负；5 日 `ret60_leader` 为 `-0.3971%/-2.3568%/-0.8929%/-1.2365%/+0.2031%`，仅 2025 转正。

### 50.4 观察

第一，第 49 节“右尾有强趋势前兆”不能等价为“买最强趋势”。裸强趋势在 dev 是负 alpha，这说明强趋势区域同时包含未来右尾和未来左尾，而且左尾在简单排序里占主导。

第二，量能确认没有解决分叉。加入 `vol20_vs_60`、`vol5_vs_20`、`amount_rank` 后，dev 仍为负，说明“强趋势 + 放量”也同时可能意味着加速延续或高位崩坏。

第三，2026 forward 局部正不能晋级。2026 对强趋势更友好，但 dev 全负或大幅负，不能把 2026 后验当成可迁移规律。

第四，真正问题变成“强趋势中的右尾/左尾分叉识别”。这已经不是单个线性手写因子能解决的形态，更像需要看过去 20-60 日路径的形状：强趋势是平滑爬升、放量突破、缩量回踩，还是急拉拥挤、长上影、放量滞涨。

### 50.5 反事实分析

第一反事实：如果第 49 节失败只是因为防守项过重，去掉防守后强趋势候选应变厚。实际所有 dev Top20 为负，否定这个解释。

第二反事实：如果 60 日强势是可直接交易的主因子，`ret60_leader` 应至少在 dev 为正。实际 3 日五年全负，5 日多数年份负，说明强趋势本身不是 alpha，必须识别状态分叉。

第三反事实：如果量能抬升能确认趋势延续，`trend_volume20_confirm` 或 `trend_volume_composite` 应优于裸趋势。实际它们仍然更弱，说明量能既确认右尾，也确认拥挤左尾。

第四反事实：如果右尾反查只是伪信号，强趋势候选 dev 负就足以放弃。实际不能简单放弃，因为第 49 节中右尾和池均值的差异在 dev/forward 同向很强；更合理的解释是强趋势区域有高方差分叉，平均排序被左尾吞掉。

### 50.6 判定

`trend_volume_continuation_soil_not_enough`。

本轮没有可融入策略，也不进入 formal。结论是：右尾前兆中的强趋势/量能不是可直接排序的 alpha，而是一个高方差候选域。下一轮应做 `trend_continuation_bifurcation_diagnostic_v1`：只在强趋势/量能候选域内，对未来 Top20 和 Bottom20 做路径形态差异诊断，重点比较急涨 vs 平滑爬升、放量突破 vs 放量滞涨、缩量回踩 vs 破位回落、近高位置、长上影/高收低收、VWAP 承接等分叉特征。如果能找到稳定分叉，再训练小型序列/DeepSets ranker；如果找不到，说明日线 OHLCV 对右尾分叉仍不够，需要回到 QMT 分钟路径或换候选源。

## 51. trend_continuation_bifurcation_diagnostic_v1

### 51.1 假设

第 50 节证明，裸强趋势/量能延续在 dev 是负 alpha，但第 49 节又证明真实右尾样本确实有强趋势和量能前兆。一个必要反事实是：强趋势域内部同时存在未来右尾和未来左尾，简单买最强趋势会被左尾吞掉；必须识别“继续冲”和“高位崩坏”的路径分叉。

本轮只在强趋势/量能候选域内做诊断。先用 `ret60 + ret20 + vol20_vs_60 + amount_rank` 取每个 session 的前 120 个强趋势/量能股票，再在这个域内比较未来 Top20 与 Bottom20 的 T 日可见路径形态，并测试少量分叉候选分数。

### 51.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_trend_continuation_bifurcation_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/trend_continuation_bifurcation_diagnostic_v1_summary.json
script sha256:0a420e7646a49cd5ed8eef1190a177610bcea6ec8f595cdff4d5561026f1759a
summary sha256:3793d078899c86d1afef1269f1aab57d4acb4b08a7a503df013c4cebb2973b4f
```

扫描 interval `3/5`，外层 pool500，强趋势/量能域 `DOMAIN_SIZE=120`，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。所有特征只使用 T 日及以前的日线 OHLCV/VWAP。

分叉特征包括：`smooth_ret20/60`、`ret5/20/60`、裁剪后的 `ret20_over_ret60`、`vol5_vs_20`、`vol20_vs_60`、`range5/20`、`upper_shadow5`、`lower_recovery5`、`close_pos5/20`、`near_high20/60`、`close_vwap5/20`、`amount_rank`。

候选包括：`smooth_trend_continue`、`smooth_trend_volume_continue`、`trend_no_upper_shadow`、`trend_close_near_high`、`trend_vwap_absorption`、`trend_pullback_recover`、`trend_low_range_continue`、`bifurcation_composite`。

### 51.3 结果

强趋势/量能域内部 oracle 仍然很厚，但所有分叉候选在 dev 都为负。

| interval | split | domain oracle Top20 | domain oracle Bottom20 | best candidate Top20 | 判读 |
| ---: | --- | ---: | ---: | ---: | --- |
| 3 | dev 2021-2025 | `+10.7095%` | `-9.1571%` | `trend_low_range_continue -0.3928%` | 域内有右尾，但候选抓不到 |
| 3 | forward 2026 | `+14.9213%` | `-10.5465%` | `trend_low_range_continue +0.5383%` | forward 局部转正，dev 不支持 |
| 5 | dev 2021-2025 | `+13.5229%` | `-11.3096%` | `trend_low_range_continue -0.5341%` | 仍为负 |
| 5 | forward 2026 | `+20.0809%` | `-13.2193%` | `trend_low_range_continue +0.7338%` | forward 正但不穿越 |

leaderboard 前排：

| interval | candidate | dev Top10 | dev Top20 | dev worst year Top20 | forward Top10 | forward Top20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | `trend_low_range_continue` | `-0.5090%` | `-0.3928%` | `-1.0010%` | `-0.2671%` | `+0.5383%` | 最优但 dev 负 |
| 5 | `trend_low_range_continue` | `-0.5423%` | `-0.5341%` | `-2.0707%` | `-0.2056%` | `+0.7338%` | dev 更差 |
| 3 | `smooth_trend_continue` | `-0.9570%` | `-0.8213%` | `-1.2114%` | `-0.0687%` | `+0.2227%` | dev 负 |
| 3 | `trend_pullback_recover` | `-1.0973%` | `-0.8595%` | `-1.4413%` | `+0.7273%` | `+0.4006%` | forward Top10 亮但 dev 负 |

### 51.4 分叉观察

最反直觉、也最有价值的结果是：在强趋势/量能域内部，未来左尾比未来右尾更“强、更平滑、更放量”。Top20 - Bottom20 的差多数为负：

| interval | feature | dev tail-bottom | forward tail-bottom | 含义 |
| ---: | --- | ---: | ---: | --- |
| 5 | `smooth_ret60` | `-0.2284` | `-0.1385` | 左尾更平滑强势 |
| 3 | `smooth_ret60` | `-0.2020` | `-0.1553` | 同上 |
| 5 | `smooth_ret20` | `-0.2027` | `-0.1501` | 左尾短中期更强 |
| 3 | `smooth_ret20` | `-0.1845` | `-0.0787` | 同上 |
| 5 | `ret60` | `-0.1111` | `-0.1055` | 左尾 60 日更强 |
| 3 | `ret60` | `-0.0880` | `-0.0957` | 同上 |
| 3 | `vol5_vs_20` | `-0.0509` | `-0.0521` | 左尾短期量能更强 |
| 5 | `vol20_vs_60` | `-0.0355` | `-0.0348` | 左尾中期量能更强 |
| 5 | `close_pos5` | `-0.0453` | `-0.0526` | 左尾收盘位置更高 |
| 5 | `close_vwap20` | `-0.0402` | `-0.0347` | 左尾更贴近/高于 VWAP |

这说明第 49 的右尾前兆不能理解为“越强越好”。强趋势/放量是进入高方差域的门票，但域内越强越可能是拥挤和耗尽，未来左尾反而更集中在最平滑、最强、最放量的位置。

### 51.5 反事实分析

第一反事实：如果强趋势内的右尾/左尾可以靠平滑趋势、低波动、无上影、近高收盘区分，候选分数应在 dev 为正。实际所有候选 dev 为负，说明这些简单形态没有形成可交易筛子。

第二反事实：如果左尾只是随机噪声，Top20 - Bottom20 的趋势强度差不应系统性为负。实际 `smooth_ret60/20`、`ret60/20`、`vol5/20`、`vol20/60` 都在 dev/forward 同向为负，说明强趋势域内存在稳定的拥挤耗尽结构。

第三反事实：如果 2026 forward 的强趋势收益是可迁移主线，dev 中不应全为负。实际 2026 多个候选局部转正，但 dev 长期为负，因此不能用 2026 后验晋级。

第四反事实：如果继续用日线手工分叉特征就能解决，至少应看到一个候选接近第 48/49 的强基线。实际没有，说明日线手写形态仍不够，或者方向应该反过来：把极强趋势/放量当左尾风险，而不是多头入口。

### 51.6 判定

`trend_continuation_bifurcation_not_enough`。

本轮没有可融入策略，也不进入 formal。结论是：强趋势/量能域是高方差区域，但域内最强、最平滑、最放量的样本更像拥挤耗尽左尾。下一轮不应继续做“强趋势继续”多头筛子，而应转向 `crowded_trend_exhaustion_avoidance_v1`：把极强趋势/放量/高收盘位置作为风险排除项，重新回到更宽候选池，寻找“不过度拥挤但仍有右尾潜力”的次强趋势、回踩修复或横截面相对强度结构。如果这个方向只能降低左尾但不能增厚右尾，则说明日线 OHLCV 已不足以识别右尾，需要回到 QMT 分钟路径或换更低相关候选源。

## 52. crowded_trend_exhaustion_avoidance_v1

### 52.1 假设

第 51 节最重要的反直觉结果是：强趋势/量能域内部，未来左尾反而更强、更平滑、更放量、更贴近高收盘位置。也就是说，强趋势/放量不是“越强越好”的多头入口，而更像进入高方差域的门票；域内极端强势可能代表拥挤耗尽。

本轮把这个结论反过来用：回到 pool500，不再限定强趋势域，测试“次强趋势 + 不极端放量 + 不极端拥挤 + 低波动/不过热”是否能保留一部分右尾，同时避开第 50/51 节裸追强趋势的左尾。

### 52.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_crowded_trend_exhaustion_avoidance_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/crowded_trend_exhaustion_avoidance_v1_summary.json
script sha256:0e15a9329392243d551898b2fdd656c82eb0668b3c969a93c8d7d99656a8f598
summary sha256:47694dc9786a3d6e1f67cbcf05a3c04d77f737c5bd51b7fb59bf701d9e93252a
```

扫描 interval `3/5`，pool500，Top10/Top20，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。所有特征只使用 T 日及以前日线 OHLCV/VWAP；未来收益只用于评估。

候选包括：`mid_trend_not_extreme`、`mid_trend_volume_not_extreme`、`mid_trend_pullback_recover`、`mid_trend_low_crowding`、`mid_trend_vwap_support`、`avoid_extreme_trend_low_range`、`avoid_extreme_trend_composite`。

### 52.3 结果

避开极端拥挤趋势后，候选从第 50/51 节的系统负 alpha 改善为接近零到小正，但收益厚度仍然很薄，且所有前排候选的 dev worst year 仍为负。

| interval | candidate | dev Top10 | dev Top20 | dev worst year Top20 | forward Top10 | forward Top20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | `avoid_extreme_trend_low_range` | `+0.0499%` | `+0.1540%` | `-0.1621%` | `+0.6383%` | `+1.0255%` | 最优，但 dev 太薄且年度不稳 |
| 5 | `mid_trend_volume_not_extreme` | `+0.1060%` | `+0.0591%` | `-0.4524%` | `+2.1353%` | `+1.2577%` | forward 漂亮，dev 支撑不足 |
| 5 | `mid_trend_low_crowding` | `+0.1685%` | `+0.0974%` | `-0.2921%` | `+0.6595%` | `+0.7967%` | 太薄 |
| 5 | `mid_trend_not_extreme` | `+0.0576%` | `+0.1452%` | `-0.3810%` | `+1.0034%` | `+0.7831%` | 太薄 |
| 3 | `mid_trend_not_extreme` | `+0.1428%` | `+0.1148%` | `-0.1132%` | `+0.5844%` | `+0.4250%` | 3 日最佳之一，仍不稳 |
| 3 | `mid_trend_volume_not_extreme` | `+0.1629%` | `+0.0349%` | `-0.1976%` | `+0.8164%` | `+0.6831%` | dev 几乎为零 |

oracle 对照仍然极厚：3 日 dev oracle Top20 `+14.9809%`、forward `+18.8419%`；5 日 dev oracle Top20 `+19.1150%`、forward `+25.6831%`。本轮前排候选离 oracle 仍差两个数量级。

年度拆分显示，方向的主要问题不是 2026 无效，而是 2022/2023 系统性拖累。例如 5 日 `avoid_extreme_trend_low_range` 的年度 Top20 为 2021 `-0.0749%`、2022 `-0.1621%`、2023 `-0.0801%`、2024 `+0.3835%`、2025 `+0.7018%`；5 日 `mid_trend_volume_not_extreme` 为 2021 `+0.1924%`、2022 `-0.3529%`、2023 `-0.4524%`、2024 `+0.1496%`、2025 `+0.7523%`。

### 52.4 观察

第一，避开极端强趋势/极端放量确实改善了方向。第 50/51 节裸追强趋势在 dev 系统负，本轮多数前排候选至少回到小正，说明“拥挤耗尽风险”这个解释有因果价值。

第二，这个改善主要是避坑，不是出 alpha。最优 dev Top20 只有 `+0.1540%`，还不到一周交易成本和滑点扰动下足够安全的厚度；即便 forward Top20 到 `+1.0255%`，也缺少 2021-2025 全年一致性支撑。

第三，2025/2026 转强、2022/2023 转弱的结构非常明显。这不像单个候选权重的问题，更像市场状态边界问题：在某些 regime 下，次强不拥挤趋势有延续；在另一些 regime 下，它仍会被宽基下行、流动性收缩或风格反转吞掉。

第四，继续调 `center/width` 或加更重的拥挤惩罚，容易变成用户明确不认可的“硬卡要求”。本轮已证明静态日线手写公式只能把负 alpha 拉回薄正，不能自然产生几十倍收益所需的厚右尾。

### 52.5 反事实分析

第一反事实：如果第 51 节“极强趋势是拥挤左尾”这个判断是错的，避开极端强势后不应改善。实际本轮 dev 从强趋势候选的负值改善到小正，说明判断部分成立。

第二反事实：如果“次强不拥挤趋势”本身就是主矿，dev worst year 应转正，且 dev Top20 应达到 1% 量级。实际最优 dev Top20 只有 `+0.1540%`，年度最差仍为负，否定它作为独立策略方向。

第三反事实：如果 2026 forward 亮点可直接迁移，2024/2025 之前不应长期弱。实际 2022/2023 多数候选为负，说明这是明显 regime 条件信号，不能用 2026 局部漂亮晋级。

第四反事实：如果继续在线性手工公式上加惩罚能解决，前排候选应显示“越防守越厚”。实际 `avoid_extreme_trend_composite` 并不优，说明过度防守会再次削弱收益弹性。

### 52.6 判定

`crowded_trend_exhaustion_avoidance_not_enough`。

本轮没有可融入策略，也不进入 formal。结论是：把极强趋势/放量视为拥挤风险是对的，但“次强不拥挤”只是薄状态，不是金矿。下一轮做 `trend_exhaustion_regime_boundary_scan_v1`：解释 2022/2023 负、2025/2026 正的市场状态边界，用 T 日以前的市场宽度、指数趋势、横截面动量扩散、成交扩散、候选池拥挤度和候选自身分布来切分本轮前排分数。如果因果状态边界能让 dev 多年转正且 forward 保留，再考虑轻量 regime-conditioned ranker；如果边界不稳，则日线静态趋势/拥挤线暂时降级，转向新的候选源或更底层的 cross-sectional world model 预训练。

## 53. trend_exhaustion_regime_boundary_scan_v1

### 53.1 假设

第 52 节证明“避开极端拥挤趋势”能把强趋势方向从负 alpha 拉回薄正，但年度结构很不稳：2022/2023 拖累明显，2025/2026 转强。一个必要反事实是：如果这不是偶然，而是可学习的市场状态边界，那么用 T 日以前的市场宽度、市场趋势、成交扩散、候选池动量分布和候选分数拥挤度做低自由度切片，应能找到 dev 多年为正且 forward 保留的子状态。

本轮不训练模型，也不做复杂 router，只做诊断：所有 bucket 阈值只用 dev `2021-2025` 的 `q30/q70` 拟合，再原样应用到 2026 forward，避免在 forward 上重切分位。

### 53.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_trend_exhaustion_regime_boundary_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/trend_exhaustion_regime_boundary_scan_v1_summary.json
script sha256:91d6c9bcd052954d659ecb4e7df098139408fea236ca2f3c82aa13bedcf7a79e
summary sha256:0d45ad9fdfee5e62aeccd87aad34a78630814ffd572e813338b5061cf9558d09
```

复用第 52 节四个前排候选：`avoid_extreme_trend_low_range`、`mid_trend_not_extreme`、`mid_trend_volume_not_extreme`、`mid_trend_low_crowding`。扫描 interval `3/5`，pool500，Top20，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。

状态特征包括：`mkt_breadth5/20`、`mkt_median_ret20/60`、`mkt_dispersion20`、`mkt_vol20`、`mkt_vol20_vs_60`、`pool_median_ret20/60`、`pool_ret20_p80`、`pool_ret60_p80`、`pool_vol20_vs_60_median`、`pool_score_best_mean/p90/dispersion`。所有状态只使用 T 日及以前数据。

### 53.3 结果

低自由度状态切片没有找到可迁移边界。leaderboard 按综合分靠前的格子主要依赖 2026 的少数 session 抬升，dev 仍非常薄，且年度最差大幅为负。

| interval | feature bucket | candidate | dev sessions | forward sessions | dev Top20 | dev worst year Top20 | forward Top20 | 判读 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | `mkt_vol20_vs_60=low` | `mid_trend_low_crowding` | 73 | 6 | `+0.0415%` | `-2.2578%` | `+5.2341%` | forward 小样本暴涨，dev 不支持 |
| 5 | `mkt_vol20_vs_60=low` | `mid_trend_not_extreme` | 73 | 6 | `+0.0908%` | `-1.8907%` | `+4.6433%` | 同上 |
| 5 | `mkt_median_ret60=mid` | `avoid_extreme_trend_low_range` | 97 | 7 | `+0.3225%` | `-0.5879%` | `+3.3201%` | 仍年度不稳 |
| 5 | `pool_score_best_mean=mid` | `mid_trend_volume_not_extreme` | 97 | 14 | `-0.0676%` | `-1.1968%` | `+3.7724%` | dev 为负 |
| 3 | `pool_ret20_p80=low` | `mid_trend_low_crowding` | 121 | 5 | `+0.0035%` | `-0.3963%` | `+2.5412%` | dev 近零 |

如果按 dev Top20 排序，也没有真正稳的格子：

| interval | feature bucket | candidate | dev Top20 | dev worst year Top20 | forward Top20 | 判读 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 5 | `pool_median_ret20=high` | `mid_trend_not_extreme` | `+0.9489%` | `+0.1677%` | `+0.5520%` | 少数相对健康，但 dev sessions 73、forward 9，厚度仍低 |
| 5 | `mkt_median_ret20=low` | `avoid_extreme_trend_low_range` | `+0.8726%` | `+0.3010%` | `+0.5685%` | 看似稳，但不是综合最优，forward 不放大 |
| 5 | `mkt_breadth20=low` | `avoid_extreme_trend_low_range` | `+0.8225%` | `+0.2121%` | `+0.5930%` | 同上，仍不足以支持策略化 |
| 5 | `mkt_vol20=high` | `mid_trend_volume_not_extreme` | `+0.7385%` | `+0.0859%` | `+1.2706%` | 有一点边界，但样本太少且仍远离目标 |
| 3 | `pool_ret20_p80=high` | `mid_trend_not_extreme` | `+0.3837%` | `+0.0383%` | `-0.1433%` | 3 日 forward 翻转 |

第一个看似较健康的 5 日 `pool_median_ret20=high / mid_trend_not_extreme` 也只有 dev Top20 `+0.9489%`、forward `+0.5520%`，离用户要求的五年几十倍到 100 倍策略所需的厚度很远；更重要的是它只是一个状态切片，不是完整可交易回放，不能晋级。

### 53.4 状态对比

状态对比进一步说明边界不稳定。以第 52 最优候选 `avoid_extreme_trend_low_range` 的 5 日结果为例，dev 中好 session 相比坏 session 的 `mkt_vol20_vs_60` 更高，`good_minus_bad=+0.0167`；forward 中方向反过来，`good_minus_bad=-0.0276`。dev 中好 session 的 `mkt_breadth5` 更低，`good_minus_bad=-0.0153`；forward 中好 session 的 `mkt_breadth5` 更高，`good_minus_bad=+0.1130`。

也就是说，2026 的收益亮点不是简单复现 dev 的好状态，而是状态含义发生了翻转或强烈漂移。这个观察与第 36/37 节策略层 router 失败、第 40/41 节 token label regime shift 的结论一致：粗状态边界容易在历史中看起来合理，但 forward 的条件含义会变。

### 53.5 反事实分析

第一反事实：如果第 52 的 2025/2026 转强来自可因果观测的简单 regime，q30/q70 dev 阈值切片应让某些格子 dev 多年转正并在 2026 保留。实际综合前排全部 dev worst year 为负，否定这个解释。

第二反事实：如果 forward 最强格子是真实矿脉，dev 中不应只有 `+0.0415%/+0.0908%` 这种近零收益。实际 `mkt_vol20_vs_60=low` 的 forward 高达 4%-5%，但 dev 极弱且 worst year 深负，更像 2026 少数窗口的偶然共振。

第三反事实：如果按 dev 最强格子就能解决，forward 不应翻转。实际 3 日 `pool_ret20_p80=high / mid_trend_not_extreme` dev 为正且 worst year 小正，但 forward 为负；说明状态边界在不同 horizon 和不同年份不稳定。

第四反事实：如果继续训练 regime-conditioned ranker 是低风险路径，低自由度状态至少应给出明确可迁移边界。实际没有，直接上深模型很可能复刻第 37 节 supervised router 的问题：历史曲线漂亮，forward 失败。

### 53.6 判定

`trend_exhaustion_regime_boundary_not_enough`。

本轮没有可融入策略，也不进入 formal。静态日线“趋势/量能/拥挤”线索已经形成较完整闭环：强趋势/量能能定位高方差域，极强趋势更像拥挤左尾，避开极端能止血，但低自由度 regime 边界无法把薄正变成厚 alpha。

下一轮不继续在这条线上调分数或训练 router。方向切换为 `cross_sectional_sequence_pretraining_feature_ablation_v1`：回到第 38-47 节底层 token/world model 线索，但不直接扩大模型，而是做特征/标签可学性消融，比较单股序列、横截面 rank、市场状态 token、候选池状态 token、行业/概念代理 token 对真实 Top20 右尾预测的边际贡献。目标是找出“预测力到底来自哪类 token”，再决定是否训练小型 factorized Transformer/Perceiver；若所有 token 贡献仍薄，则说明仅靠本地日线 OHLCV/QMT 现有字段不足以达到收益目标，需要转向新的 PIT 数据源或更细粒度事件传播数据。

## 54. cross_sectional_sequence_pretraining_feature_ablation_v1

### 54.1 假设

第 38-47 节已经多次证明，底层 token 模型在开发期能学出弱 IC，但 forward 和账户层不稳。第 52-53 节又证明，静态日线趋势/拥挤线索只能止血，不能自然产生厚 alpha。因此本轮不再扩大模型，也不再继续调单一分数，而是做同口径输入消融：在完全相同的 tiny temporal Transformer、pool500、5 日 open-to-open 标签、year-invariant epoch selection 和账户回放下，只改变输入 token 保留哪些信息。

核心问题是：预测力到底来自单股原始路径、横截面 rank、regime token，还是这些信息本身都不足以支撑 forward alpha。

### 54.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_sequence_pretraining_feature_ablation_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_sequence_pretraining_feature_ablation_v1_summary.json
script sha256:5024e3ad6c2df21f5bae128763fc92e422bc8efc01871743b22d5f0d2c00f3e8
summary sha256:1105e1bf65519fe74200abf1a793db162e224c6a1e2b6846e8bf392cf78cf6b2
```

训练/验证/前向切分：train `2021-2024`，valid `2025`，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。样本数：train `96677`，valid `24362`，dev `121065`，forward `11914`；session 数：train `194`，valid `49`，dev `243`，forward `24`。

模型复用第 47 节 tiny temporal Transformer，`seq_len=60`，额外 1 个 regime token，Top10 回放，训练 `4` epoch。forward 随机 pool500 Top10 对照 `80` 次：p50 `1.0940x`，p90 `1.2645x`，p95 `1.3964x`，max `1.4276x`。

消融组：

| variant | 输入 |
| --- | --- |
| `full_stock_plus_regime` | 11 维单股 token + regime token |
| `stock_raw_path_no_regime` | ret/open/intraday/range/vwap/volume/ret5/ret20，不含横截面 rank，不含 regime |
| `stock_cross_rank_no_regime` | `rank_ret1/rank_ret20/rank_amount`，不含 regime |
| `stock_raw_plus_rank_no_regime` | 11 维单股 token，不含 regime |
| `regime_token_only` | 只保留调仓日候选池 regime token |
| `rank_plus_regime` | 横截面 rank + regime token |

### 54.3 结果

所有消融组都没有通过 forward 随机 p95，也没有满足 dev 五年全正。最优组是 `stock_raw_plus_rank_no_regime`：dev 有弱 IC 和弱收益，但 forward 亏损，且 remove best 3 后更差。

| variant | dev multiple | dev all years positive | dev rank IC | dev Top10 label | forward multiple | forward rank IC | forward Top10 label | beats random p95 | 判读 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `stock_raw_plus_rank_no_regime` | `1.7954x` | 否 | `+0.0960` | `+0.3181%` | `0.9402x` | `+0.0108` | `-0.0197%` | 否 | dev 弱可学，forward 账户反向 |
| `stock_cross_rank_no_regime` | `1.4361x` | 否 | `+0.0892` | `+0.2837%` | `0.9275x` | `+0.0138` | `-0.3891%` | 否 | rank 有 dev IC，forward Top10 很差 |
| `full_stock_plus_regime` | `1.6306x` | 否 | `+0.0878` | `+0.3072%` | `0.9085x` | `+0.0016` | `-0.2446%` | 否 | regime token 没改善迁移 |
| `rank_plus_regime` | `1.4090x` | 否 | `+0.0815` | `+0.2939%` | `0.8541x` | `+0.0081` | `-0.8386%` | 否 | 加 regime 后账户更差 |
| `regime_token_only` | `1.1588x` | 否 | `+0.0031` | `-0.1801%` | `0.8352x` | `0.0000` | `+1.5488%` | 否 | session token 单独不可排序 |
| `stock_raw_path_no_regime` | `1.0183x` | 否 | `+0.0859` | `+0.2055%` | `0.9100x` | `+0.0063` | `-0.4303%` | 否 | 原始路径单独几乎无账户收益 |

年度拆分也不支持晋级。最优组 `stock_raw_plus_rank_no_regime` 的 dev 年度收益为 2021 `+14.17%`、2022 `-2.84%`、2023 `-1.43%`、2024 `+39.37%`、2025 `+17.82%`。这不是用户要求的五年全正，更远没有几十倍收益。

### 54.4 观察

第一，横截面 rank 是主要信息源。`stock_raw_path_no_regime` dev multiple 只有 `1.0183x`，而 `stock_cross_rank_no_regime` 有 `1.4361x`，`stock_raw_plus_rank_no_regime` 有 `1.7954x`。这说明“绝对单股路径”本身很弱，短周期可学性更多来自横截面相对位置。

第二，regime token 没有带来稳定迁移。`full_stock_plus_regime` 弱于 `stock_raw_plus_rank_no_regime`，`rank_plus_regime` 弱于 `stock_cross_rank_no_regime`。这与第 53 节状态边界不稳一致：粗 regime 信息会帮助模型拟合历史，但不能自然解决 2026 条件漂移。

第三，forward 的随机池反而很强。pool500 随机 Top10 的 forward p50 是 `1.0940x`，p95 是 `1.3964x`，而所有模型都低于 `0.95x`。这不是“市场没有收益”，而是模型在 2026 主动选到了比随机更差的股票。

第四，dev rank IC 和账户收益并不等价。多个组的 dev rank IC 在 `0.08-0.096`，看起来不低，但 2026 账户层全部亏损。说明当前 rank label 学到的更像历史横截面风格，而不是可迁移的右尾捕捉器。

### 54.5 反事实分析

第一反事实：如果单股日线序列模型只是缺少合适 token，某个消融组应该在 forward 至少打过随机 p95。实际没有一个打过，说明问题不只是 token 子集选择。

第二反事实：如果 regime token 是关键缺失，带 regime 的 full/rank 版本应超过不带 regime 的版本。实际相反，说明当前粗 regime token 更像历史风格编码，不能承担 world model 的状态转移表达。

第三反事实：如果 dev IC 足以代表可交易预测力，dev IC 最高组 forward 不应亏损。实际 dev IC 最高的 `stock_raw_plus_rank_no_regime` forward `0.9402x`，说明评估必须继续坚持账户层、多年度、forward 和随机池对照，而不能只看 IC。

第四反事实：如果继续增加 epoch 或模型层数就能解决，至少应看到 full 模型在 2026 比 raw/rank 单独更稳。实际 full 模型更差，说明扩大模型大概率放大历史风格拟合。

### 54.6 判定

`sequence_feature_ablation_not_enough`。

本轮没有可融入策略，也不进入 formal。结论很明确：在当前 pool500、5 日个股 rank-label、60 日单股 token 的范式下，模型可以学习到开发期横截面弱 IC，但不能转成 forward 账户 alpha。继续做同范式的 GRU/Transformer/feature ablation 已经进入低收益区。

下一轮允许切换到一个更大胆、截然不同的假设：`cross_sectional_opportunity_first_world_model_v1`。不再先预测“哪只股票未来 5 日最好”，而是先预测“哪个调仓日/哪类市场状态存在可开采的右尾机会”，也就是 session-level opportunity first。第 48 节已经证明真实 oracle 右尾极厚，第 54 节又证明个股 ranker 会在 2026 主动选错。新的反事实是：也许可迁移信号主要在 session 层和 group diffusion 层，而不是单股票最终排序层。先用 T 日可见的横截面分布、候选池尾部厚度、行业/概念代理扩散、市场宽度和过去若干 session 的可得收益状态，训练小型 session-level/world-state 模型预测下一期 pool 内 Top20 oracle 厚度、随机池收益分布和模型应不应该启用；若 session 级机会都不可预测，再考虑换数据源。若 session 机会可预测，再在机会窗口内训练更小的 cross-sectional ranker，而不是全年硬选股。

## 55. cross_sectional_opportunity_first_world_model_v1

### 55.1 假设

第 54 节说明，全年强制做个股 ranker 会在 forward 主动选错；但第 48/51/53 节又反复显示 pool 内真实右尾非常厚。一个更大胆的解释是：可迁移信号可能主要不在“每个 session 都精确挑中哪只股票”，而在“哪些 session 本身存在可开采机会”。如果机会窗口可预测，那么后续可以只在高机会窗口内训练或启用更小的 ranker；如果机会窗口都不可预测，全年选股模型就更不可能鲁棒。

本轮因此训练一个 session-level tiny MLP world-state 模型。输入是 T 日可见的市场宽度、pool500 横截面收益/波动/量能分布、候选分数分布；输出是未来 5 日的 pool mean、oracle Top50、candidate-best Top20、oracle Top20 和 tail spread。它不直接融入策略，只验证“机会窗口优先”是否有可学性。

### 55.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_opportunity_first_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_opportunity_first_world_model_v1_summary.json
script sha256:bee6e06c954d3cb9b3a4610a64d97dd8f3e5d85708c4536f4b681527cfb3e914
summary sha256:45dbf04f9cfa3606060ae0401fee5a183cad7d683bb1ac3c9a4b135e6b583f3e
```

切分：train `2021-2024`，valid `2025`，dev `2021-2025`，forward `2026-01-01` 到 `2026-07-10`。样本是 5 日 rebalance session，数量为 train `194`、valid `49`、dev `243`、forward `24`。

模型为 hidden `32` 的 tiny MLP，多任务回归，valid 上按 `pool_mean/candidate_best_top20/oracle_top50` 的 rank IC 加权选 epoch。候选收益测试复用第 52 节四个静态候选：`avoid_extreme_trend_low_range`、`mid_trend_not_extreme`、`mid_trend_volume_not_extreme`、`mid_trend_low_crowding`。

### 55.3 结果

模型没有通过晋级 gate，verdict 为 `opportunity_first_world_model_not_enough`。但它给出一个有价值的半正结果：session 级机会预测比第 54 节个股 ranker 更像一条可继续下沉的线索。

目标层 rank IC：

| split | pool mean | candidate-best Top20 | oracle Top20 | oracle Top50 | tail spread20 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | `+0.6948` | `+0.6192` | `+0.8061` | `+0.8062` | `+0.8290` | 训练期很强，需防过拟合 |
| valid 2025 | `+0.3666` | `+0.2568` | `+0.2421` | `+0.2888` | `+0.3279` | 有正迁移 |
| dev 2021-2025 | `+0.4290` | `+0.3603` | `+0.6193` | `+0.5350` | `+0.7169` | 开发期机会可学 |
| forward 2026 | `+0.0843` | `+0.2096` | `+0.1591` | `+0.0991` | `+0.2357` | forward 弱但非零 |

按综合 opportunity score 切 top/mid/bottom session：

| split | bucket | sessions | pool mean | oracle Top20 | oracle Top50 | candidate-best Top20 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | top30% | 73 | `+1.9753%` | `+23.3961%` | `+17.2151%` | `+3.1681%` | 显著好于全样本 |
| dev | all | 243 | `-0.0756%` | `+19.1150%` | `+13.3643%` | `+1.1529%` | 全样本弱很多 |
| dev | bottom30% | 73 | `-1.0036%` | `+16.9746%` | `+11.4442%` | `+0.2047%` | 明显差 |
| forward | top30% | 7 | `+0.3259%` | `+26.1491%` | `+18.7615%` | `+3.2066%` | 好于全样本，但样本很少 |
| forward | all | 24 | `+0.1917%` | `+25.6831%` | `+18.4091%` | `+2.4575%` | 已经较肥沃 |
| forward | bottom30% | 7 | `+0.8197%` | `+27.5375%` | `+19.6718%` | `+2.8036%` | bottom 也不差，综合 score 单调性不足 |

对第 52 的具体候选来说，forward top30% 窗口里 `mid_trend_volume_not_extreme` 从全样本 `+1.2577%` 提高到 `+2.5368%`；dev top30% 里同候选从全样本 `+0.0591%` 提高到 `+2.0880%`。这是本轮最值得继续追的细节。

### 55.4 观察

第一，session 层 opportunity 比个股 ranker 更可学。第 54 节所有个股 ranker forward 账户亏损，而本轮 forward `candidate_best_top20` rank IC 为 `+0.2096`，说明模型对“何时候选公式更可能有收益”有一些预测力。

第二，综合 opportunity score 设计不够精确。forward top30% 的 candidate-best 确实较高，但 bottom30% 的 oracle/pool mean 也不差；这说明“预测 oracle 厚度”和“预测手上候选能否吃到厚度”不是同一个任务。真正应优化的是可交易候选收益，而不是 pool oracle。

第三，forward 样本太少，不能晋级。top30% 只有 7 个 session，且综合 rank IC 对 candidate-best 只有 `+0.0791`，没有达到稳健 gate。不能把 7 个窗口的漂亮数字当策略。

第四，dev top30% 年度结构相对有价值。top30% 的 candidate-best 在 2021-2024 分别为 `+2.6592%/+3.0450%/+2.8795%/+4.6476%`，说明至少在 train 期间不是单一年份偶然。但 2025 被放入 valid，且 forward 很短，仍需要更严格的 walk-forward/gated replay。

### 55.5 反事实分析

第一反事实：如果 session 机会完全不可预测，valid/forward 的 candidate-best rank IC 应接近 0 或反向。实际 valid `+0.2568`、forward `+0.2096`，否定“完全不可预测”。

第二反事实：如果只要预测 oracle 厚度就足够，综合 opportunity score 的 top bucket 应在 forward 对 pool/oracle/candidate 都明显单调。实际 bottom30% 的 oracle 和 pool mean 也很高，说明 oracle 厚度不是可交易候选收益的充分条件。

第三反事实：如果第 55 已经是可用策略，forward top30% 的优势应大且稳定，且 dev/forward 同一候选都明显穿越。实际只有 `mid_trend_volume_not_extreme` 较亮，forward 样本 7 个，不足以进入 formal。

第四反事实：如果继续训练个股 ranker才是正路，第 54 应至少给出某个 forward 正收益模型。实际没有；第 55 的 session 层反而出现正 IC，因此下一轮应围绕 candidate-return gate 下沉，而不是回到全年个股 ranker。

### 55.6 判定

`opportunity_first_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。但方向不应放弃：session-level opportunity first 是第 54 之后第一个在 forward 出现可解释正 IC 的 DL/world-state 线索。下一轮做 `candidate_return_opportunity_gate_replay_v1`：不要再用综合 opportunity score，而是直接预测第 52 四个候选的未来 Top20 收益，采用 walk-forward/valid 选择 gate 阈值，并做真实 gated replay。重点验证三件事：第一，gate 后 dev 是否五年全正；第二，forward 是否仍超过全启用和随机窗口；第三，平均持仓是否仍大于 5。如果单目标 candidate-return gate 也失败，再说明 session 机会方向只适合作为诊断，不适合策略化。

## 56. candidate_return_opportunity_gate_replay_v1

### 56.1 假设

第 55 节的综合 opportunity score 没有过 gate，但 `candidate_best_top20` 在 forward 仍有 `+0.2096` 的单目标 rank IC。一个直接反事实是：也许失败不是机会窗口不可预测，而是目标错了；应直接预测“手上候选未来能不能赚钱”，而不是预测 oracle 厚度、pool mean 或 tail spread。

本轮因此训练 session-level tiny MLP，输出第 52 节四个候选的未来 5 日 Top20 标签收益。训练仍用 `2021-2024`，valid `2025` 选择 gate 阈值，再固定应用到 dev `2021-2025` 和 forward `2026`。最后做真实 Top10 账户回放，验证 gate 是否能把第 52 的弱候选变成可交易开关。

### 56.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_candidate_return_opportunity_gate_replay_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/candidate_return_opportunity_gate_replay_v1_summary.json
script sha256:6deb09aee4832fbf45839d39b3a9021cf842d72ce20d90dc98ff95c516b1d68c
summary sha256:0d0f9afea0702660c1ca1b87e988d67f4bdda3e9bbad8aeb472f0d123666c829
```

切分和第 55 节一致：train `194` 个 session，valid `49`，dev `243`，forward `24`。输入复用第 55 节的 T 日可见 session/world-state 特征。目标为四个候选的未来 Top20 标签收益，账户回放用 Top10、5 日换仓、open-to-open、交易约束和成本口径。

gate 选择规则：只在 valid `2025` 上扫描每个候选预测值的 `q50/q60/q70/q80`，以及动态选择预测最高候选的 `q50/q60/q70/q80`；按 valid 回放排序选前列配置，再固定评估 dev/forward。没有使用 2026 调阈值。

### 56.3 结果

结论是否定。模型在 dev 上看起来极强，但 forward 的候选收益预测 IC 几乎消失；valid 选出来的 gate 不能修复 2022/2023 的历史亏损，也会错过 2026 的主要赚钱窗口。

目标预测 IC：

| split | candidate | Top20 rank IC | Top10 rank IC | Top20 mean | 判读 |
| --- | --- | ---: | ---: | ---: | --- |
| dev | `avoid_extreme_trend_low_range` | `+0.6007` | `+0.5283` | `+0.1540%` | 历史拟合很强 |
| dev | `mid_trend_low_crowding` | `+0.6312` | `+0.5886` | `+0.0974%` | 历史拟合很强 |
| dev | `mid_trend_not_extreme` | `+0.5990` | `+0.5763` | `+0.1452%` | 历史拟合很强 |
| dev | `mid_trend_volume_not_extreme` | `+0.5863` | `+0.5777` | `+0.0591%` | 历史拟合很强 |
| forward | `avoid_extreme_trend_low_range` | `-0.0026` | `-0.1887` | `+1.0255%` | 迁移失败 |
| forward | `mid_trend_low_crowding` | `+0.0591` | `+0.0930` | `+0.7967%` | 很弱 |
| forward | `mid_trend_not_extreme` | `-0.0104` | `+0.0635` | `+0.7831%` | 很弱 |
| forward | `mid_trend_volume_not_extreme` | `0.0000` | `-0.0200` | `+1.2577%` | 迁移失败 |

base 全启用回放也很有信息量：这些候选在 dev 全部亏损，但 2026 全部正，尤其 `mid_trend_volume_not_extreme` forward `1.6694x`。

| candidate | dev multiple | dev annual returns | forward multiple | 判读 |
| --- | ---: | --- | ---: | --- |
| `avoid_extreme_trend_low_range` | `0.6713x` | 2021 `-23.33%`，2022 `-21.41%`，2023 `-10.92%`，2024 `+24.91%`，2025 `+0.13%` | `1.1725x` | 2026 正，历史差 |
| `mid_trend_low_crowding` | `0.7761x` | 2021 `-4.02%`，2022 `-20.47%`，2023 `-10.56%`，2024 `-0.74%`，2025 `+14.53%` | `1.2420x` | 2026 正，历史差 |
| `mid_trend_not_extreme` | `0.6142x` | 2021 `-9.57%`，2022 `-25.02%`，2023 `-11.66%`，2024 `-8.75%`，2025 `+12.38%` | `1.3180x` | 2026 正，历史差 |
| `mid_trend_volume_not_extreme` | `0.7200x` | 2021 `-2.81%`，2022 `-12.10%`，2023 `-17.81%`，2024 `-2.47%`，2025 `+5.13%` | `1.6694x` | 2026 最强，但历史不可用 |

valid 选择出的最佳 gate 为 `dynamic_best_predicted_candidate q80`。它在 valid 2025 为 `1.1302x`，但 dev 仍非五年全正，forward 只启用 2 个 session，样本太少。

| split | enabled sessions | multiple | annual returns | max DD | avg pos | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| valid 2025 | 10 | `1.1302x` | 2025 `+13.02%` | `-14.18%` | 10.00 | `0.9892x` | valid 看起来可以，但脆 |
| dev 2021-2025 | 183 | `2.0111x` | 2021 `+35.35%`，2022 `-9.11%`，2023 `-8.40%`，2024 `+44.60%`，2025 `+23.43%` | `-30.64%` | 10.00 | `1.7211x` | 仍不满足五年全正 |
| forward 2026 | 2 | `1.1124x` | 2026 `+11.24%` | `-7.95%` | 10.00 | `1.0000x` | 仅 2 次交易，不能证明 |

其它看似有效的 valid gate 也不稳。例如 `mid_trend_not_extreme q80` 在 valid 为 `1.3939x`，但 dev 2022/2023 仍负，forward 只启用 1 次且几乎持平；`avoid_extreme_trend_low_range q60/q70` 在 forward 反而亏损。

### 56.4 观察

第一，第 56 把第 55 的半正线索推到真实 gate 后基本否定。第 55 的 forward `candidate_best` rank IC 看起来有希望，但当目标改成具体候选收益并用 valid 选门槛后，forward IC 近零，说明机会模型的可迁移部分很弱。

第二，2026 的风格与 2021-2025 强烈不同。base 候选在 dev 全部亏损、2026 全部正；模型在 dev 上的 IC 很高，正说明它学到了 2021-2025 的“什么窗口不要开”，结果 forward 恰好处在这些历史模型不愿意开的收益窗口里。

第三，gate 在 2026 的最大问题不是亏损，而是错过收益。`mid_trend_volume_not_extreme` 全启用 forward `1.6694x`，而 valid 最佳动态 gate 只做 2 次，`1.1124x`；它把最肥的 2026 beta/趋势窗口过滤掉了。

第四，valid 2025 不是可靠未来代理。多个 valid 高收益 gate 在 dev 或 forward 都不稳，这与第 37 节 supervised router 的经验一致：单年 validation 容易选择到历史 regime 的局部门槛。

### 56.5 反事实分析

第一反事实：如果第 55 的机会窗口信号可策略化，直接预测候选收益并用 valid gate 后，应改善 dev 年度稳定并保留 forward。实际 dev 2022/2023 仍负，forward 交易次数极少，否定。

第二反事实：如果候选收益模型真正泛化，forward Top20/Top10 rank IC 不应接近 0。实际四个候选 forward Top20 IC 分别为 `-0.0026/+0.0591/-0.0104/0.0000`，说明泛化失败。

第三反事实：如果 gate 是正确方向，它至少不应显著弱于全启用。实际 2026 全启用的 `mid_trend_volume_not_extreme` 是 `1.6694x`，gate 后最多只有少数交易的 `1.1124x`，说明 gate 削掉了收益弹性。

第四反事实：如果继续堆 session MLP/Transformer 就能解决，现有低自由度模型应给出方向一致的迁移迹象。实际历史 IC 越高，越像学到了不可迁移的 regime 边界；继续加模型容量只会放大过拟合风险。

### 56.6 判定

`candidate_return_opportunity_gate_not_enough`。

本轮没有可融入策略，也不进入 formal。第 55-56 合在一起说明：session-level opportunity first 有诊断价值，但不能直接形成稳定 gate。更关键的观察是，2026 中“次强趋势/不极端拥挤”候选明显赚钱，而 2021-2025 尤其 2022/2023 明显亏损；模型试图用历史 gate 避开这些窗口，反而错过 forward 收益。

下一轮不继续做 gate。方向切换到更激进的 `market_phase_contrastive_world_model_v1`：承认 2021-2023 与 2025-2026 可能处在不同 market phase，目标不是预测候选收益本身，而是学习“当前 phase 更像历史上的哪类市场生成机制”。先做非策略化诊断：用 T 日以前的全市场横截面分布、趋势扩散、波动/成交状态，把每个 session 嵌入到 phase space，检验 2026 是否更接近 2024/2025 的高 beta 趋势 phase，而不是 2021-2023。若 phase embedding 能解释候选收益方向翻转，再考虑 phase-conditioned ranker；如果 phase 也解释不了，就说明当前日线/QMT K 线输入对这个收益目标的信息不够，需要寻找新的 QMT 可拉取维度或更细 intraday path。

## 57. market_phase_contrastive_world_model_v1

### 57.1 假设

第 56 节暴露了一个关键矛盾：第 52 的次强趋势/不拥挤候选在 2021-2025 多数亏损，却在 2026 明显赚钱；session gate 用历史学习到的“不要开”规则，反而过滤掉了 2026 的主要收益窗口。一个更高层的反事实是：2026 可能属于不同 market phase，不能用全历史统一映射预测候选收益，而应先识别当前 phase 更像历史上的哪类生成机制。

本轮不训练策略、不做 gate，只做非策略化 phase 诊断：用 T 日以前的 session/world-state 特征构建 phase embedding，在 train `2021-2024` 上拟合标准化、PCA 和 k-means，再把 valid `2025` 与 forward `2026` 投影进去；同时用历史 kNN 邻居预测候选收益方向，检验 phase 是否能解释 2026 的方向翻转。

### 57.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_market_phase_contrastive_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/market_phase_contrastive_world_model_v1_summary.json
script sha256:5660a2b0bf3c93149af8acbdeec0dc075369644e077dae3f524fd5700f9dc312
summary sha256:8a37773ffde49f070b003444157648125a87e38f4b70d487e0ec1e5df2deaebb
```

样本数：train `194`，valid `49`，dev `243`，forward `24`。特征复用第 55 节 T 日可见 session/world-state 分布。PCA 维度 `6`，k-means `K=5`，kNN `K=12`。训练 phase 只用 `2021-2024`；forward 的标签不参与 phase 拟合。

### 57.3 结果

结论为 `market_phase_contrastive_not_enough`。phase embedding 的确显示 2026 更接近 2025，但这种接近不能充分解释候选收益方向，也不足以支持 phase-conditioned ranker。

最近邻年份占比：

| query | library | 2021 | 2022 | 2023 | 2024 | 2025 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| valid 2025 | train 2021-2024 | `16.33%` | `26.70%` | `33.16%` | `23.81%` | - | 2025 没有简单贴近 2024 |
| forward 2026 | train 2021-2024 | `63.19%` | `18.40%` | `3.47%` | `14.93%` | - | 只看 train 时，2026 反而像 2021 |
| forward 2026 | train+valid 2021-2025 | `27.78%` | `7.29%` | `1.39%` | `5.21%` | `58.33%` | 加入 2025 后，2026 明显贴近 2025 |

phase 均值距离也支持“2026 是独立 phase”：forward 到 valid 距离 `8.6173`，到 train 距离 `11.3810`，到 dev 距离 `10.7175`；虽然更靠近 2025，但仍相当远。

用 train+valid 历史邻居预测 forward：

| target | actual mean | pred mean | rank IC | 判读 |
| --- | ---: | ---: | ---: | --- |
| `pool_mean` | `+0.1917%` | `+0.0935%` | `+0.3087` | pool 肥沃度有一点可解释 |
| `oracle_top50` | `+18.4091%` | `+16.0283%` | `+0.4417` | oracle 厚度较可解释 |
| `oracle_top20` | `+25.6831%` | `+22.4203%` | `+0.1487` | Top20 较弱 |
| `candidate_best_top20` | `+2.4575%` | `+1.1908%` | `+0.1783` | 候选可吃到的收益解释不足 |
| `candidate_mid_trend_volume_not_extreme` | `+1.2577%` | `-0.1285%` | `+0.1774` | 方向均值仍错 |
| `candidate_mid_trend_low_crowding` | `+0.7967%` | `+0.2818%` | `+0.2687` | 相对最好，但仍薄 |

如果只用 train `2021-2024` 做邻居，forward 的 `mid_trend_volume_not_extreme` 预测均值为 `-0.4505%`，实际为 `+1.2577%`；加入 2025 后预测改善到 `-0.1285%`，但仍没有翻正。这说明 2025 对解释 2026 有帮助，但不够。

### 57.4 cluster 观察

k-means cluster 对 `mid_trend_volume_not_extreme` 的方向解释不稳定：

| cluster | forward sessions | forward mid-trend-volume | train sessions | train mid-trend-volume | 方向一致 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | `+5.1334%` | 22 | `+0.3282%` | 是，但 forward 样本 1 |
| 1 | 7 | `+1.4768%` | 2 | `+4.0092%` | 是，但 train 样本 2 |
| 2 | 11 | `+0.9089%` | 83 | `-0.0584%` | 否，最大 forward cluster 方向翻转 |
| 4 | 5 | `+0.9432%` | 6 | `-0.1791%` | 否 |

最大的 forward cluster 2 在 train 中有 83 个样本，但 train 的 `mid_trend_volume_not_extreme` 略负，forward 却为正。这是本轮最重要的否定证据：同一个粗 phase cluster 内，候选收益方向仍会跨年份翻转。

### 57.5 观察

第一，phase embedding 对“市场是否肥沃”比对“哪个候选赚钱”更有用。`oracle_top50` forward rank IC 达 `+0.4417`，但候选收益只有 `+0.17-0.27`，且 `mid_trend_volume_not_extreme` 均值方向仍错。

第二，2026 确实不是简单的 2021-2024 历史延续。只用 train 近邻时，2026 最近邻主要来自 2021；加入 2025 后，2026 最近邻 58.33% 来自 2025。这说明 2025 是理解 2026 的关键，但 2025 单年不足以训练稳定规则。

第三，粗 phase cluster 不能直接下沉成策略。cluster 2/4 的 train 与 forward 候选方向相反，说明目前的 state embedding 捕捉的是肥沃/波动/宽度等粗状态，而不是候选收益生成机制。

第四，这与第 55-56 的结论一致：当前日线 session/world-state 特征可以解释一部分“有没有右尾”，但解释不了“我手上的横截面候选能不能吃到右尾”。

### 57.6 反事实分析

第一反事实：如果 2026 只是 2024/2025 高 beta phase 的重复，phase 近邻应主要来自 2024/2025，且候选收益方向同向。实际加入 2025 后确实近邻多来自 2025，但候选均值预测仍不翻正，说明重复不充分。

第二反事实：如果 phase embedding 已足以解释候选方向，`mid_trend_volume_not_extreme` 的 pred mean 应为正，cluster 方向应一致。实际最大 cluster 方向相反，否定。

第三反事实：如果继续在日线 session state 上训练更大 phase model 是正确方向，低维 PCA/kNN 至少应给出明确可迁移边界。实际只对 oracle/pool 有边界，对候选收益没有，继续加容量容易重演第 56 的历史拟合。

第四反事实：如果当前失败源于市场无右尾，oracle 预测也应弱。实际 oracle_top50 可解释性强，说明市场有右尾，问题在“候选/输入不能捕捉到可交易右尾”。

### 57.7 判定

`market_phase_contrastive_not_enough`。

本轮没有可融入策略，也不进入 formal。第 55-57 的共同结论是：session/world-state 模型能够感知一部分市场肥沃度，但无法稳定转成候选收益或交易 gate。继续在日线 session state 上加深模型，边际收益很低。

下一轮切换到更低层、更接近交易行为的数据：`qmt_intraday_path_opportunity_probe_v1`。思路不是重回涨停事件，而是在第 52 的次强趋势/不拥挤候选中，用 QMT 5m/1m 日内路径特征解释 2026 为什么可交易、2021-2023 为什么容易亏：例如 T 日和 T+1 的开盘承接、盘中回撤修复、尾盘资金、日内趋势一致性、跳空后是否回补、量能分布是否平滑。若 QMT 日内路径能区分同一候选在不同 phase 下的可交易性，再训练小型 intraday-path encoder；若仍不能，则说明当前 QMT K 线维度也不足以满足收益目标，需要寻找新的 QMT 可拉取维度而不是继续日线 world model。

## 58. qmt_intraday_path_opportunity_probe_v1

### 58.1 假设

第 57 节说明，日线 session/world-state 可以解释一部分市场肥沃度，但解释不了候选能不能吃到右尾。一个更底层的反事实是：同一批 `mid_trend_volume_not_extreme` 候选在 2026 可交易，可能不是因为日线 phase，而是因为信号日 T 的日内路径暴露了资金行为，例如尾盘承接、VWAP 支撑、低点修复、日内趋势一致性、量能是否过度集中。

本轮只做 2026 小样本探针，不训练模型，不用 T+1 信息，也不融入策略。对每个 5 日 rebalance session，在第 52 的 `mid_trend_volume_not_extreme` 候选里取前 60，只使用信号日 T 的 QMT 5m K 线构造日内路径特征，再看这些特征能否在候选内部重新排序未来 5 日收益。

### 58.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_intraday_path_opportunity_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_intraday_path_opportunity_probe_v1_summary.json
script sha256:909e14caf8cb0440e763b0b0689e48809c4bef7027508c8288315cc0351855eb
summary sha256:476170e23650e84ca75c606b8061e6e7f83274cd3c9b3566f61ce8b0e082b8f3
```

数据：QMT `5m`，区间 `2026-01-01` 到 `2026-07-10`，每个 session 取 `mid_trend_volume_not_extreme` 前 `60`。共请求 `1440` 个股票-日期样本，`1440` 个可用，`0` empty，`0` error，其中 `174` 个命中已有缓存，其余通过 QMT 拉取后落在项目 `.tmp` 下。

特征只使用信号日 T 的 5m 路径，包括：`intraday_return`、`close_pos_range`、`vwap_support`、`low_to_close_recovery`、`high_to_close_fade`、`tail_return`、`tail_volume_share`、`second_half_return`、`drawdown_from_high`、`range_amp`、`volume_concentration_top20pct`、`smooth_path_score` 等。评估口径是在每个 session 内按特征排序，取 Top20 的未来 5 日 open-to-open 标签均值，并计算 session 内 rank IC。

### 58.3 结果

结论为 `qmt_intraday_path_opportunity_probe_not_enough`。T 日 5m 路径没有增强第 52 的 2026 收益，最好的日内特征和组合都低于原始日线候选 baseline。

原始 `mid_trend_volume_not_extreme` 的 2026 Top20 label baseline 为 `+1.2577%`。日内路径排序最好结果：

| ranker | Top20 mean | Bottom20 mean | spread | mean rank IC | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `smooth_path_score` | `+0.8993%` | `+0.7061%` | `+0.1932%` | `+0.0062` | 最佳单特征，但低于 baseline |
| `volume_concentration_top20pct` 反向 | `+0.7509%` | `+0.4336%` | `+0.3174%` | `+0.0326` | 有一点方向，但太薄 |
| `second_half_return` | `+0.7168%` | `+1.2889%` | `-0.5721%` | `+0.0156` | Top20 反而更差 |
| `vwap_support` | `+0.2505%` | `+1.3362%` | `-1.0856%` | `-0.0086` | 典型“强承接”反向 |
| `tail_return` | `+0.5887%` | `+1.4027%` | `-0.8140%` | `-0.0392` | 尾盘强反向 |
| `intraday_return` | `+0.1448%` | `+1.1004%` | `-0.9556%` | `-0.0415` | T 日日内强势反向 |

组合分数也没有改善：

| composite | Top20 mean | mean rank IC | 判读 |
| --- | ---: | ---: | --- |
| `smooth_late_accumulation` | `+0.6955%` | `+0.0159` | 最佳组合，仍低于 baseline |
| `anti_intraday_exhaustion` | `+0.2777%` | `+0.0168` | 不够 |
| `intraday_path_composite` | `+0.1597%` | `-0.0049` | 失败 |
| `tail_vwap_recovery` | `+0.1290%` | `-0.0195` | 强承接组合失败 |

### 58.4 观察

第一，QMT 数据链路本身可靠。本轮 `1440/1440` 可用，没有 empty/error，这说明失败不是数据缺失导致。

第二，最反直觉也最有价值的结果是：T 日 intraday 强收、VWAP 支撑、尾盘强、日内上涨这些常见“承接好”的路径，在这个候选池里反而选得更差。`vwap_support`、`tail_return`、`intraday_return` 的 Top20 都显著低于 bottom 侧。

第三，原始日线候选已经比日内路径重排更好。baseline `+1.2577%`，而最佳日内单特征只有 `+0.8993%`，最佳组合只有 `+0.6955%`。这说明对 `mid_trend_volume_not_extreme` 来说，T 日 5m 路径不是正向入口，至少不是简单的“越强承接越好”。

第四，这与第 51 节强趋势域观察一致：越强、越平滑、越放量、收得越高，可能更接近拥挤耗尽，而不是延续入口。但本轮的反向信息也没有强到可用，说明 T 日路径最多告诉我们“不要追强承接”，不能提供新的厚 alpha。

### 58.5 反事实分析

第一反事实：如果 2026 的可交易性来自 T 日内资金承接，`vwap_support`、`tail_return`、`second_half_return`、`close_pos_range` 应该提高 Top20 label。实际多数低于 baseline，否定这个解释。

第二反事实：如果日内路径只是需要非线性组合，composite 应至少超过单一日线 baseline。实际最佳 composite `+0.6955%`，仍远低于 `+1.2577%`，说明不是简单组合问题。

第三反事实：如果 QMT 分钟维度本身足以解决第 57 的候选收益解释问题，本轮应看到明显 rank IC。实际 mean rank IC 多数接近 0，只有少量弱正，说明 T 日 5m K 线信息不够。

第四反事实：如果失败源于缓存覆盖不足，本轮应有大量 empty/error。实际全量可用，否定数据质量解释。

### 58.6 判定

`qmt_intraday_path_opportunity_probe_not_enough`。

本轮没有可融入策略，也不进入 formal。它否定了“用信号日 T 的简单 5m 承接路径修复日线候选”的方向。第 55-58 连在一起说明：市场右尾和 2026 趋势收益确实存在，但当前日线 world-state、session gate、T 日 5m 路径都不能稳定解释“候选能否吃到右尾”。

下一轮需要进一步大胆切换：不再预测单股票或单 session，而是做 `cross_sectional_pairwise_spread_world_model_v1`。思路是把问题从“绝对收益预测”改成“候选内部两两相对胜负/扩散路径预测”：在同一 session 内，模型只判断 A 是否会跑赢 B，输入为二者过去 20/60 日横截面相对路径差、行业/市值/流动性代理差、T 日日内路径差。pairwise 目标可能比绝对收益更抗 regime 漂移；若 pairwise 在 2026 仍不能超过 baseline，则说明当前 K 线输入对这条收益目标的信息已经接近耗尽，应转向 QMT 可拉取的其它维度或更高层 group/industry diffusion。

## 59. cross_sectional_pairwise_spread_world_model_v1

### 59.1 假设

第 54-58 轮连续说明，单股序列、session opportunity gate、phase 近邻和 T 日 5m 承接路径都不能稳定解释 2026 的右尾。一个自然反事实是：绝对收益标签太受 market phase 漂移影响，但同一 session、同一候选池内部的相对胜负可能更稳定。

本轮把问题改写成 pairwise：在每个 5 日调仓 session 内，先按第 52 轮 `mid_trend_volume_not_extreme` 取候选前 60，再训练一个小 MLP 判断股票 A 未来 5 日 open-to-open 是否跑赢股票 B。输入只用信号日及之前的日线横截面路径特征差 `x_A - x_B`，训练 `2021-2024`，验证 `2025`，dev 统计 `2021-2025`，forward 为 `2026-01-01` 到 `2026-07-10`。

### 59.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_cross_sectional_pairwise_spread_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_pairwise_spread_world_model_v1_summary.json
script sha256:0cc9a81da77f85580c85cebb5d91d9862805a27c43f2c443f054da9cc5e4f99d
summary sha256:906b4acb93f67e8e8ba85b8a07b2368d20d24d45386f7a3ca3f9587b2d634c6b
```

样本：train 194 个 session，valid 49 个 session，forward 24 个 session。pairwise 样本数 train `686674`、valid `173448`。模型为两层小 MLP，BCEWithLogits，最多 8 epoch，按 2025 pair accuracy 选最优 epoch。

### 59.3 结果

结论为 `pairwise_spread_world_model_not_enough`。模型在历史 dev 上确实学到相对胜负，但 forward 明显输给原始候选排序。

| 区间 | base Top20 | model Top20 | mean rank IC | 判读 |
| --- | ---: | ---: | ---: | --- |
| train `2021-2024` | `-0.1160%` | `+0.2767%` | `+0.1101` | 训练期改善明显 |
| valid `2025` | `+0.6546%` | `+0.6665%` | `+0.0672` | 仅小幅持平略优 |
| dev `2021-2025` | `+0.0591%` | `+0.3737%` | `+0.1009` | 历史汇总很漂亮 |
| forward `2026` | `+1.2577%` | `+0.3766%` | `+0.0083` | 前向显著打薄 |

训练最优 epoch 为 1，best valid pair acc `0.5232`，高于随机但并不厚。dev 分年 Top20 为：2021 `+0.3452%`，2022 `-0.2019%`，2023 `+0.2869%`，2024 `+0.6752%`，2025 `+0.7577%`。年度最差仍为负，已经不满足五年全正。

### 59.4 观察

第一，pairwise 目标不是完全无效。dev rank IC `+0.1009`，train/valid 都为正，说明同 session 候选内部确实存在可学习的历史相对结构。

第二，失败点也非常清楚：2026 的原始 `mid_trend_volume_not_extreme` baseline Top20 已经有 `+1.2577%`，而 pairwise model 只剩 `+0.3766%`。模型不是没预测，而是把 2026 最有弹性的股票排掉了。

第三，valid 2025 没有暴露这个风险。2025 上 model Top20 `+0.6665%` 与 base `+0.6546%` 几乎持平，pair acc `0.5232` 看似可接受；但 2026 的收益生成机制发生了对模型不利的迁移。

第四，best epoch 为 1，说明模型容量即使很小也会迅速吃到历史统计结构；继续加深 GRU/Transformer/MLP 大概率只是增强历史拟合，不会自然解决 forward 削弹性问题。

### 59.5 反事实分析

第一反事实：如果绝对收益预测失败只是因为 market beta 漂移，pairwise 相对胜负应更抗 regime，并在 2026 至少不低于 base。实际 forward Top20 下降约 `0.88%`，否定。

第二反事实：如果 2026 的右尾来自同一套日线横截面结构，dev rank IC 应迁移到 forward。实际 forward rank IC 只有 `+0.0083`，接近随机。

第三反事实：如果验证集 2025 足以代表 2026，2025 持平略优应对应 2026 不差。实际 2026 大幅变薄，说明 2025 validation 对这个目标的保护不足。

第四反事实：如果问题只是 topK 选择太窄，Top10 应至少有改善。实际 forward Top10 从 base `+2.1353%` 降到 model `+0.1949%`，说明削掉的是头部弹性，不是尾部噪声。

### 59.6 判定

`pairwise_spread_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。它给出的强约束是：在第 52 候选域里，继续用同一批单股日线横截面特征训练更复杂的相对排序模型，容易得到 dev 好看、forward 削弹性的结果。下一轮不再沿着“单股 K 线特征 + 候选内部 ranker”加容量，而是切到更高层但仍可本地训练的 `group_diffusion_world_model_v1`：把行业/概念内扩散、组内相对位置、组热度分位和候选在组内的角色作为输入，验证 2026 的右尾是否来自板块扩散结构，而不是个股孤立形态。

## 60. group_diffusion_world_model_v1

### 60.1 假设

第 59 轮说明，同一候选域内继续用单股日线特征训练相对排序模型，会在 dev 上变好但在 2026 前向削掉头部弹性。一个更高层的反事实是：2026 的右尾不是来自孤立个股形态，而是来自行业/概念内部扩散、组内相对位置、组热度和候选在组内的角色。

本轮不做硬过滤，不把行业/概念当独立收益引擎，而是训练两个同结构小 MLP 做对照：`stock_only` 只用第 52 轮日线横截面特征，`stock_plus_group` 额外加入静态行业/概念内扩散特征。目标为同 session 候选内未来 5 日收益 rank，训练 `2021-2024`，验证 `2025`，forward 为 `2026-01-01` 到 `2026-07-10`。

### 60.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_group_diffusion_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/group_diffusion_world_model_v1_summary.json
script sha256:befdd0abf8adb98c14120f3613c1a1fddfe1f2c85c73cc7b21e41023daaa5e53
summary sha256:6d7b34223fd28d9ff938adb25b09bf9d7d99c2c738399e27e55fb4febd105674
```

样本：train 194 个 session，valid 49 个 session，dev 243 个 session，forward 24 个 session。候选域仍为每个 session 的 `mid_trend_volume_not_extreme` 前 60。组特征包括行业/概念内 ret5/20/60、宽度、离散度、量能扩散、个股相对组内 ret、组内 rank 等。

重要 caveat：行业/概念成员关系来自 2026 静态快照，不是 PIT 元数据。因此本轮即使有效，也只能作为探索性证据，不能直接进入策略。覆盖率记录为：行业数 `86`，概念数 `439`，行业覆盖 `52.79%`，概念覆盖 `52.63%`，平均每股概念数 `5.14`。

### 60.3 结果

结论为 `group_diffusion_world_model_not_enough`。组扩散特征确实增强了小模型，但仍然低于第 52 原始候选在 2026 的弹性。

| 模型 | 区间 | base Top20 | model Top20 | mean rank IC | 判读 |
| --- | --- | ---: | ---: | ---: | --- |
| `stock_only` | dev | `+0.0591%` | `+0.3814%` | `+0.0854` | 复现第 59 的历史重排优势 |
| `stock_only` | forward | `+1.2577%` | `+0.3591%` | `-0.0587` | 前向削弹性更明显 |
| `stock_plus_group` | dev | `+0.0591%` | `+0.4873%` | `+0.1196` | 组特征增强历史 IC 和收益 |
| `stock_plus_group` | forward | `+1.2577%` | `+0.6134%` | `-0.0261` | 好于纯股票，但仍输 baseline |

`stock_plus_group` 的 dev 分年 Top20 为：2021 `+0.3961%`，2022 `+0.0696%`，2023 `+0.5894%`，2024 `+0.6467%`，2025 `+0.7295%`。这是近期少见的 dev 五年全正，但绝对收益厚度远不足，且 2026 仍只有 baseline 的约一半。

验证集上，`stock_plus_group` 的 row rank IC `+0.0415`，高于 `stock_only` 的 `+0.0294`；说明组特征不是纯噪声。但 `stock_plus_group` forward Top10 只有 `+0.5581%`，远低于 baseline Top10 的 `+2.1353%`，仍在削掉头部赢家。

### 60.4 观察

第一，组扩散方向有真实边际信息。相同模型结构下，加入组特征后 dev Top20 从 `+0.3814%` 升到 `+0.4873%`，forward Top20 从 `+0.3591%` 升到 `+0.6134%`，valid row IC 也提升。

第二，这个边际信息不足以承担重排权。2026 原始候选 baseline 已经是 `+1.2577%`，`stock_plus_group` 虽然比纯股票模型好，但完整重排后仍损失约一半收益。

第三，dev 五年全正是积极信号，但要非常克制地解读。它来自静态 2026 组映射，存在历史成分回填风险；并且五年全正对应的 Top20 均值仍只有 `+0.4873%`，离账户层几十倍收益要求很远。

第四，forward rank IC 仍为负。组特征把 `stock_only` 的 forward rank IC 从 `-0.0587` 修到 `-0.0261`，但没有翻正。这说明组扩散在 2026 是一个弱修复变量，不是主预测变量。

### 60.5 反事实分析

第一反事实：如果 2026 右尾主要由行业/概念扩散解释，`stock_plus_group` 应在 forward 超过原始 baseline。实际只到 `+0.6134%`，否定“组扩散可单独重排”的假设。

第二反事实：如果第 59 失败只是缺少组结构，那么加入组特征后 forward rank IC 应明显转正。实际仍为 `-0.0261`，说明组结构只缓解了错误排序，没有解决方向迁移。

第三反事实：如果组特征是纯噪声，`stock_plus_group` 不应系统性优于 `stock_only`。实际 dev、valid、forward 都有边际改善，否定“完全无信息”。

第四反事实：如果 dev 五年全正足以证明方向可用，forward Top10 不应被压到 `+0.5581%`。实际 Top10 弹性被大幅削弱，说明完整模型重排仍会错过最值钱的头部。

### 60.6 判定

`group_diffusion_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。它保留了一个有价值线索：行业/概念扩散可以作为 residual 信息改善候选内部排序，但不能允许模型完整重排。下一轮改成 `group_residual_anchor_world_model_v1`：以第 52 原始排序为锚，只在验证集上选择很小的 group residual 权重，验证能否保住 2026 头部弹性的同时，让 dev 年度稳定性变好。如果连轻微 residual 都不能改善，则组扩散方向应降级为解释/风险特征，而不是 alpha 主干。

## 61. group_residual_anchor_world_model_v1

### 61.1 假设

第 60 轮说明，行业/概念扩散特征有边际信息，但完整重排会削弱 2026 头部弹性。一个更保守的反事实是：问题不是组扩散无效，而是模型自由度太高；如果以第 52 轮 `mid_trend_volume_not_extreme` 原始排序为锚，只允许小 residual 调整，可能既保住 2026 右尾，又改善 dev 年度稳定性。

本轮重用第 60 轮同结构小 MLP，但不再让模型直接排序。每个 session 的最终分数为：

```text
score = base_rank_anchor + weight * model_residual_rank
```

其中 `base_rank_anchor` 来自第 52 候选排序，`model_residual_rank` 来自 `stock_only` 或 `stock_plus_group` 小模型。权重只在 2025 valid 上从 `[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]` 选择，forward 不参与选权重。

### 61.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/train_group_residual_anchor_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/group_residual_anchor_world_model_v1_summary.json
script sha256:29bf3dc5b25ac31eac9cdfd44ac1272426e682cd21e28196eb375419ec9fe196
summary sha256:6f5a77228319a0d40991f451bb6f54603d58d1a704e3d19847aa4c36782c0586
```

样本和候选域与第 60 轮一致：train 194 个 session，valid 49 个 session，dev 243 个 session，forward 24 个 session；每个 session 为 `mid_trend_volume_not_extreme` 前 60。组映射仍是 2026 静态快照，因此本轮只做探索性标签层诊断。

### 61.3 结果

结论为 `group_residual_anchor_world_model_not_enough`。base 锚定明显减少了第 60 轮完整重排的 forward 损伤，但仍没有超过原始 baseline，dev 分年也没有全正。

valid 选择的 `stock_plus_group_residual` 权重为 `0.3`：

| 口径 | dev Top20 | forward Top20 | forward Top10 | forward Top20 换手 vs base | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| base anchor `w=0.0` | `+0.0591%` | `+1.2577%` | `+2.1353%` | `0.00%` | 原始基线 |
| `stock_only_residual` valid 选 `w=0.3` | `+0.1503%` | `+0.9880%` | `+1.2560%` | `12.92%` | 保住一部分，但削弱明显 |
| `stock_plus_group_residual` valid 选 `w=0.3` | `+0.2149%` | `+1.1149%` | `+1.2894%` | `9.79%` | 好于 stock-only，但仍输 baseline |

`stock_plus_group_residual w=0.3` 的 dev 分年 Top20 为：2021 `+0.3511%`，2022 `-0.2437%`，2023 `-0.0743%`，2024 `+0.2358%`，2025 `+0.7966%`。2022/2023 仍为负，不能满足五年全正。

`stock_plus_group_residual` 的权重网格显示，任何权重都没有在 forward Top20 上超过 baseline：

| weight | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward Top20 换手 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `+0.6546%` | `+0.0591%` | `+1.2577%` | `+2.1353%` | `0.00%` |
| `0.05` | `+0.6454%` | `+0.0682%` | `+1.2114%` | `+2.1650%` | `0.21%` |
| `0.10` | `+0.7356%` | `+0.0925%` | `+1.0948%` | `+1.8371%` | `3.13%` |
| `0.15` | `+0.7074%` | `+0.1153%` | `+1.0205%` | `+1.7073%` | `3.96%` |
| `0.20` | `+0.6859%` | `+0.1343%` | `+1.2311%` | `+1.4327%` | `6.46%` |
| `0.30` | `+0.7572%` | `+0.2149%` | `+1.1149%` | `+1.2894%` | `9.79%` |
| `0.40` | `+0.7281%` | `+0.2836%` | `+1.2057%` | `+1.3259%` | `16.04%` |

### 61.4 观察

第一，base anchor 是必要的。第 60 轮 `stock_plus_group` 完整重排 forward Top20 只有 `+0.6134%`；第 61 轮锚定后，同一组特征 residual 提升到 `+1.1149%`。这说明之前损失主要来自排序自由度过高。

第二，组 residual 仍然不能创造超额。权重 `0.05` 几乎不换仓，只把 forward Top20 从 `+1.2577%` 轻微降到 `+1.2114%`；权重变大后 dev 改善，但 forward Top10 明显受损。

第三，valid 2025 会偏向较大的 residual 权重。`w=0.3` 在 valid Top20 和 Top10 上最好，但 forward 并不最好，说明 2025 对 residual 强度的选择仍然不能代表 2026。

第四，dev 分年没有修复。`stock_plus_group_residual w=0.3` 虽然整体 dev Top20 从 `+0.0591%` 提到 `+0.2149%`，但 2022/2023 仍为负，无法满足五年全正约束。

### 61.5 反事实分析

第一反事实：如果第 60 的失败只是因为模型完整重排太激进，那么轻 residual 应能在保住 baseline 的同时改善 forward。实际所有非零权重 forward Top20 都低于 baseline，否定。

第二反事实：如果组 residual 是可迁移 alpha，valid 选中的 `w=0.3` 应在 forward 也表现最佳。实际 `w=0.3` 低于 `w=0.2/0.4`，且都低于 `w=0.0`，说明 valid 选权仍不稳。

第三反事实：如果 residual 主要修复尾部噪声，它不应伤害 Top10。实际 `w>=0.1` 后 forward Top10 从 `+2.1353%` 快速降到 `+1.84%` 以下，说明 residual 动到了最有价值的头部。

第四反事实：如果组扩散是主收益土壤，dev 年度最差应转正。实际 2022/2023 仍为负，说明它最多是弱解释变量。

### 61.6 判定

`group_residual_anchor_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。第 60-61 轮合并结论是：行业/概念扩散对候选内部排序有弱信息，但不足以成为 alpha 主干；锚定可以减少损伤，却不能产生新的 forward 超额，也不能修复 dev 年度稳定性。

下一轮需要更大胆地换问题定义：不再在第 52 候选前 60 内做细排序，而是回到“什么样的股票会成为 5 日右尾”的生成机制。方向为 `right_tail_prototype_world_model_v1`：使用历史每个 session 的 top winners 构造多尺度右尾原型，训练一个小型 contrastive/prototype 模型，让候选接近历史右尾原型、远离左尾原型，并用严格 train/valid/forward 检验。若 prototype 仍只提高 dev、削弱 forward，则说明当前 K 线/静态组维度对该收益目标的信息已接近耗尽，后续应转向新的 QMT 可拉取维度或账户结构，而不是继续局部重排。

## 62. right_tail_prototype_world_model_v1

### 62.1 假设

第 59-61 轮反复出现同一个问题：普通 ranker 或 residual ranker 一旦在第 52 候选前 60 内动排序，就会提高 dev、削掉 2026 头部弹性。一个更底层的反事实是：我们不应该学习连续收益 rank，而应该学习“右尾生成原型”。

本轮只用 `2021-2024` 训练集内每个 session 的 winner/loser 构造原型：取同一候选域内未来 5 日收益最高的若干只股票作为 right-tail prototype，收益最低的若干只作为 left-tail prototype，使用标准化后的 stock 特征或 stock+group 特征做 cosine k-means。评估时，候选得分为“接近 winner prototype 减去接近 loser prototype”。2025 valid 只负责选择原型配置和是否锚定，2026 forward 不参与选择。

### 62.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_right_tail_prototype_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_prototype_world_model_v1_summary.json
script sha256:70b946702d1cf5e6464d77b0409f0facd50f878b326adf1b7bd24eace41cf8bb
summary sha256:a93842b9bb61b6c4114aa5deadb52fa66942d21bb200e41304c7ea9ba8ca3014
```

候选域仍为每个 session 的 `mid_trend_volume_not_extreme` 前 60。扫描配置：`winner_n`/`loser_n` 为 `(5,10)`、`(10,10)`、`(10,20)`，prototype 数 `k` 为 `4/8/16`，特征为 `stock_only` 或 `stock_plus_group`，评估模式为 pure prototype 或 base anchor 加 residual。样本数：train 194、valid 49、dev 243、forward 24 个 session。

### 62.3 结果

结论为 `right_tail_prototype_world_model_not_enough`。valid 选择了 `stock_plus_group`、`winner_n=10`、`loser_n=10`、`k=16`、pure prototype。该配置在 forward Top20 几乎贴住原始 baseline，但 Top10 被明显削弱，dev 分年仍有负值。

| 区间 | base Top20 | prototype Top20 | base Top10 | prototype Top10 | rank IC | 换手 vs base Top20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train `2021-2024` | `-0.1160%` | `+0.3042%` | `-0.0579%` | `+0.6331%` | `+0.0844` | `65.82%` |
| valid `2025` | `+0.6546%` | `+1.0644%` | `+0.7505%` | `+1.2055%` | `+0.0232` | `65.00%` |
| dev `2021-2025` | `+0.0591%` | `+0.4093%` | `+0.1060%` | `+0.6221%` | `+0.0671` | `65.99%` |
| forward `2026` | `+1.2577%` | `+1.2543%` | `+2.1353%` | `+1.4277%` | `+0.0139` | `68.13%` |

dev 分年 Top20 为：2021 `+0.8575%`，2022 `-0.1130%`，2023 `-0.0162%`，2024 `+0.4838%`，2025 `+0.8252%`。2022/2023 仍为负，不能满足五年全正。

leaderboard 里有两个重要对照：

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | 换手 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pure `stock+group` `10/10 k16` | `+1.0644%` | `+0.4093%` | `+1.2543%` | `+1.4277%` | `68.13%` | 贴住 Top20，但削 Top10 |
| pure `stock_only` `10/10 k16` | `+0.9778%` | `+0.5337%` | `+0.7447%` | `+0.2471%` | `71.04%` | 无 group 时 forward 失效 |
| anchor `stock+group` `10/10 k16 w0.4` | `+0.9491%` | `+0.1728%` | `+1.2023%` | `+1.3591%` | `13.54%` | 低换手保住一部分 |
| anchor `stock+group` `10/20 k16 w0.3` | `+0.7902%` | `+0.0817%` | `+1.0998%` | `+2.1456%` | `10.83%` | Top10 很好但 Top20 薄 |

### 62.4 观察

第一，prototype 方向比普通 ranker 健康。第 59-61 轮的模型一旦提高 dev，forward Top20 通常明显低于 baseline；本轮 pure prototype 虽然高换手，但 forward Top20 为 `+1.2543%`，几乎等于 baseline `+1.2577%`。

第二，组特征在 prototype 里比在 MLP ranker 里更有价值。`stock_only` pure prototype 的 forward Top20 只有 `+0.7447%`，而 `stock+group` pure prototype 达 `+1.2543%`。这说明行业/概念扩散并非主排序器，但可能帮助定义“右尾形态族”。

第三，最大问题是 Top10 弹性。pure prototype 把 forward Top10 从 baseline `+2.1353%` 降到 `+1.4277%`。也就是说它能找一批平均不差的右尾相似样本，但没有保住最尖锐的头部 winner。

第四，换手很高。pure prototype 与 base Top20 的平均换手 `68.13%`，但收益只是打平 baseline。这不是策略级进步，因为同样收益下更高换手会在账户层增加成本和路径不确定性。

### 62.5 反事实分析

第一反事实：如果历史右尾原型完全不可迁移，forward Top20 应像第 59-60 的 ranker 一样明显变薄。实际 pure `stock+group` forward Top20 贴住 baseline，否定“完全不可迁移”。

第二反事实：如果原型已经抓住主收益机制，Top10 不应明显低于 baseline。实际 Top10 损失约 `0.71%`，说明 prototype 抓到的是宽右尾，不是最强右尾。

第三反事实：如果组特征只是静态映射噪声，`stock+group` 不应大幅优于 `stock_only`。实际 forward Top20 从 `+0.7447%` 提到 `+1.2543%`，说明组信息在定义原型时有边际价值。

第四反事实：如果 high-turnover pure prototype 可以直接替代 base，dev 分年应该更平滑。实际 2022/2023 仍为负，说明它没有解决跨年份稳定性。

### 62.6 判定

`right_tail_prototype_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。但它改变了后续方向：右尾 prototype 是近期少数没有明显破坏 2026 Top20 的学习方法，值得继续沿“候选生成”而不是“top60 内重排”探索。

下一轮改成 `right_tail_prototype_pool_expansion_v1`：不再限制在第 52 候选前 60，而是在 pool500 内测试 prototype 是否能找到 base top60 之外的右尾。如果 prototype 只是在 top60 内替换而不能扩池增厚，则它只能作为解释工具；如果 pool expansion 能同时提高 dev、forward 和年度稳定性，再考虑训练轻量 embedding/world model。

## 63. right_tail_prototype_pool_expansion_v1

### 63.1 假设

第 62 轮显示，right-tail prototype 在第 52 候选前 60 内能贴住 2026 Top20 baseline，但没有提高 Top10，也没有修复 dev 年度负值。一个自然反事实是：prototype 的价值可能不是重排 top60，而是从更宽的 pool 中找到 base top60 外的右尾股票。

本轮把候选域从 base top60 扩到 base 排序前 `120/200/500`，继续使用 train-only winner/loser prototype。评估时额外记录 Top20 中有多少股票来自 base top60 外，判断 prototype 是否真正具备候选生成能力。

### 63.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_right_tail_prototype_pool_expansion_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_prototype_pool_expansion_v1_summary.json
script sha256:6041f38753c49f62b3a392bfed49583345abea9cabe5f34a591faf646d062bef
summary sha256:46392e29f41daf43e2f9ebb637b0064b2fc6b28d1f1eb50ab5d7909c0120b4f0
```

本轮有一个重要实验设计问题：脚本要求每个 session 必须有满 `expand_size` 个有效未来标签，因此 expand500 样本骤减为 train 56、valid 8、forward 3 个 session。expand500 的 valid 选择不能作为真实结论，只能作为缺陷暴露。

正常样本的 expand120/200 均为 train 194、valid 49、dev 243、forward 24 个 session。

### 63.3 结果

结论为 `right_tail_prototype_pool_expansion_not_enough`，但需要分层解读：expand500 因小样本污染不能采信；expand120/200 的正常样本显示，真正选出 base top60 外股票的 pure prototype 在 forward 明显变薄，而 forward 较好的 anchor 版本几乎没有扩池。

| 扩池 | 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | outside20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 120 | anchor stock-only `10/10 k16 w0.4` | `+1.0673%` | `+0.1919%` | `+0.5539%` | `+0.7848%` | `0.00%` | valid 好，forward 薄，未扩池 |
| 120 | pure stock-only `10/20 k16` | `+0.8892%` | `+0.4689%` | `+0.5689%` | `+0.6402%` | `59.58%` | 真扩池但 forward 变薄 |
| 120 | anchor stock-only `10/20 k16 w0.2` | `+0.8412%` | `+0.0939%` | `+1.1812%` | `+0.9180%` | `0.00%` | 接近 baseline，但未扩池 |
| 200 | pure stock+group `10/10 k16` | `+0.9925%` | `+0.3651%` | `+0.3577%` | `-0.9443%` | `75.00%` | 真扩池后 forward 失败 |
| 200 | anchor stock-only `10/10 k16 w0.4` | `+0.9260%` | `+0.3787%` | `+1.3488%` | `+1.0948%` | `0.00%` | forward Top20 高，但本质未扩池 |
| 200 | pure stock+group `10/20 k16` | `+0.8190%` | `+0.5423%` | `+0.6053%` | `+0.3130%` | `74.58%` | dev 可学，forward 仍薄 |

expand500 的全局 selected 看似 valid Top20 `+1.7899%`，但它只有 valid 8、forward 3 个 session，forward Top20 `-1.7276%`，不能作为方向证据。它主要说明本轮脚本的满 500 标签约束不合理。

### 63.4 观察

第一，prototype 一旦真正扩出 top60，forward 大幅变薄。expand120 pure outside20 约 `59.6%`，forward Top20 只有 `+0.5689%`；expand200 pure outside20 约 `75%`，forward Top20 只有 `+0.3577%` 或 `+0.6053%`。

第二，forward 较好的版本几乎都没有扩池。expand200 anchor stock-only `w0.4` forward Top20 `+1.3488%`，但 outside20 为 `0%`，说明它只是微调 base top60 的内部顺序，不是候选生成器。

第三，valid 会偏好看起来更宽的原型，但 forward 不支持。expand200 pure stock+group valid Top20 接近 `+1%`，dev 也为正，但 forward Top10 甚至为 `-0.9443%`，这是典型的扩池候选质量不可迁移。

第四，expand500 暴露了评估设计缺陷。要求有效标签满 500 会让样本集中到少数特殊 session，导致 valid/forward 完全不可比。第 64 轮必须修正为“最多取 500，至少保留 120”。

### 63.5 反事实分析

第一反事实：如果 right-tail prototype 是候选生成器，那么 outside20 提高时 forward Top20 应不低于 base。实际 outside20 达 60%-75% 的 pure 配置 forward 明显低于 base，否定。

第二反事实：如果扩池失败只是因为 topK 太窄，Top30 应该改善。实际 pure 扩池 Top30 也没有显示足够收益厚度，不能修复 Top20 失败。

第三反事实：如果 anchor 版本的 forward 高收益说明扩池有效，它应有非零 outside20。实际 outside20 基本为 0，说明收益来自保持 base，而不是扩池。

第四反事实：如果 expand500 的 valid 强结果可信，forward 不应只剩 3 个 session且转为大负。实际样本太少且 forward Top20 `-1.7276%`，说明这是样本筛选污染。

### 63.6 判定

`right_tail_prototype_pool_expansion_not_enough`。

本轮没有可融入策略，也不进入 formal。它否定了朴素 prototype 从 base top60 外扩池找右尾的能力，同时暴露了 expand500 的样本约束缺陷。

下一轮做 `right_tail_prototype_pool_expansion_v2`：修复样本约束为每个 session 最多取 500、至少保留 120 个有效标签，重新评估 pool expansion。如果修正版仍显示 outside20 增加时 forward 变薄，则 right-tail prototype 方向应降级为 top60 内解释/辅助，而不是候选生成器。

## 64. right_tail_prototype_pool_expansion_v2_stock_only

### 64.1 假设

第 63 轮否定了朴素 prototype 扩池，但 expand500 因“必须有满 500 个有效未来标签”的约束只剩 forward 3 个 session，存在严重样本污染。第 64 轮先修正这个约束：每个 session 最多取 500 个候选，但只要求至少 120 个有效标签，避免把样本筛成特殊市场状态。

完整 stock+group flexible 扩池网格计算过重，因此本轮先做 stock-only 最小诊断，回答一个更基础的问题：只用日线 K 线右尾原型，从 base top60 外扩池是否有独立收益。如果 stock-only 扩池已经失败，再继续做 group 版缓存优化的价值会下降。

### 64.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_right_tail_prototype_pool_expansion_v2_stock_only.py
.tmp/quantx-research/deep-learning-alpha-search-v1/right_tail_prototype_pool_expansion_v2_stock_only_summary.json
script sha256:9b6406d2befe767406557fd561fb083dd6933fb3a29badeb74a8fe1f08d0f8c5
summary sha256:eb160a1cae2ddfba3d4f4c9796f98a703852d62ae9b2fdf5b4b68d5bf2561b11
```

扫描 `expand200/expand500`，每个 split 样本均恢复为 train 194、valid 49、dev 243、forward 24 个 session。expand500 的 forward 平均有效候选数为 `496.42`，最小 `485`，说明第 63 轮的样本塌缩已修复。

配置只保留 stock-only right-tail prototype：pure `10/10`、pure `10/20`、anchor `w=0.2`、anchor `w=0.4`。

### 64.3 结果

结论为 `right_tail_prototype_pool_expansion_v2_stock_only_not_enough`。修正样本后，真正扩出 top60 的 pure prototype 在 forward 明显失败；anchor 版本能略好，但 outside20 接近 0 或很低，本质不是候选生成。

| 扩池 | 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | outside20 | 判读 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 200 | anchor `10/10 k16 w0.4` | `+0.9260%` | `+0.3787%` | `+1.3488%` | `+1.0948%` | `0.00%` | forward Top20 略高，但未扩池 |
| 200 | anchor `10/10 k16 w0.2` | `+0.7310%` | `+0.2588%` | `+1.2597%` | `+1.3097%` | `0.00%` | 基本贴 baseline，未扩池 |
| 200 | pure `10/20 k16` | `+0.6243%` | `+0.2102%` | `+0.3560%` | `+0.1897%` | `76.04%` | 真扩池后显著变薄 |
| 200 | pure `10/10 k16` | `+0.5114%` | `+0.3457%` | `+0.2967%` | `+0.4425%` | `75.63%` | 真扩池后显著变薄 |
| 500 | anchor `10/10 k16 w0.2` | `+0.7267%` | `+0.2971%` | `+0.7155%` | `+0.3712%` | `0.63%` | 扩池极少，收益变薄 |
| 500 | pure `10/20 k16` | `+0.7110%` | `+0.1906%` | `-0.8134%` | `-0.8921%` | `96.46%` | 大幅扩池后 forward 转负 |
| 500 | pure `10/10 k16` | `+0.7151%` | `+0.2466%` | `-0.5214%` | `-0.3329%` | `87.29%` | 大幅扩池后 forward 转负 |
| 500 | anchor `10/10 k16 w0.4` | `+0.6610%` | `+0.2949%` | `+0.7580%` | `+0.1181%` | `13.75%` | 有少量扩池，但仍远低于 baseline |

全局 valid 选择为 expand200 anchor `w=0.4`，forward Top20 `+1.3488%`，略高于 base `+1.2577%`，但 outside20 为 `0%`，并且 dev 分年仍有 2022 `-0.1504%`、2023 `-0.1246%`。它不是扩池策略，只是 base top60 内的轻微重排。

### 64.4 观察

第一，样本修复成功。第 63 轮 expand500 forward 只有 3 个 session，本轮恢复为 24 个 session，平均有效池接近 500。因此本轮可以更可信地判断扩池质量。

第二，结论非常一致：outside20 越高，forward 越差。expand200 pure outside20 约 `75%-76%`，forward Top20 只有 `+0.30%-0.36%`；expand500 pure outside20 `87%-96%`，forward Top20 直接为负。

第三，anchor 能保住一部分收益，是因为几乎不扩池。expand200 anchor outside20 为 `0%`，expand500 anchor `w=0.2` outside20 只有 `0.63%`。它们不是候选生成器。

第四，valid 不能识别扩池风险。expand500 pure 在 valid Top20 约 `+0.71%`，看起来不差，但 forward 转负，说明从更宽池里找“历史右尾相似”会吸入大量不可迁移形态。

### 64.5 反事实分析

第一反事实：如果第 63 的扩池失败只是样本塌缩造成，修复 expand500 后 pure prototype 应恢复 forward。实际修复样本后 forward 更明确为负，否定。

第二反事实：如果 prototype 是候选生成器，outside20 高的 pure 版本应至少接近 base。实际 outside20 越高，收益越差，否定。

第三反事实：如果 anchor 版本的 forward 高收益证明扩池有效，它应有显著 outside20。实际 outside20 为 0 或接近 0，说明收益来自保持原 top60。

第四反事实：如果 top60 外有大量可由日线原型识别的右尾，expand500 pure Top30 应改善。实际 Top30 也为负或很薄，说明不是 Top20 太窄的问题。

### 64.6 判定

`right_tail_prototype_pool_expansion_v2_stock_only_not_enough`。

本轮没有可融入策略，也不进入 formal。第 62-64 轮合并结论是：right-tail prototype 在 top60 内有解释价值，能比普通 ranker 更不容易破坏 forward Top20；但它不能作为候选生成器。只要从 base top60 外大量扩池，forward 就快速变薄甚至转负。

下一轮切换方向，不再围绕第 52 候选域或 prototype 扩池继续拧。新的方向应该回到更接近市场微观结构的“资金状态/成交额路径”而不是价格形态：例如构造 `liquidity_cycle_world_model_v1`，按行业/主题/股票多尺度成交额占比、放量宽度、流动性回落、价格-流动性背离来预测未来 5 日横截面右尾。煤炭旁路诊断也提示，流动性脉冲会先出现再衰减，能否捕捉“脉冲正在扩散而未耗尽”的阶段，比单纯价格原型更可能接近周期性 alpha。

## 65. liquidity_cycle_world_model_v1

### 65.1 假设

第 62-64 轮基本否定了 right-tail prototype 的扩池能力：价格形态相似可以解释 top60 内部的一部分右尾，但一旦从更宽池里大量拉入 base top60 外股票，2026 forward 会快速变薄甚至转负。新的问题定义改为：右尾是否更多来自“资金状态/成交额路径”而不是单纯价格形态。

本轮在第 52 的 `mid_trend_volume_not_extreme` top60 域内，构造股票自身、行业、概念的多尺度成交额和价格-流动性状态特征，包括自身 `amount20/amount120`、`amount5/amount20`、成交额份额、20/60 日收益，组内成交额份额、组内 `amount20/amount120`、组内放量宽度、组内上涨宽度、组内收益离散度等。用 `2021-2024` 训练 ridge 线性模型，`2025` 只用于选择 pure/anchor 权重，`2026` 严格前向。

### 65.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_liquidity_cycle_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/liquidity_cycle_world_model_v1_summary.json
script sha256:dba5a1d6b2158ae8cca4fc419a543630f689155a1581de2b22d912fb0fed968e
summary sha256:8f1c014e2099a94a52484ed75e9d8aebbd70c659ad260ea23292a0292d968ed9
```

候选域为每个 session 的 `mid_trend_volume_not_extreme` 前 60。样本数：train 194、valid 49、dev 243、forward 24 个 session。模型只使用训练集拟合均值、方差和 ridge 系数；评估模式包含 pure liquidity residual，以及 base anchor 加 residual，权重为 `0/0.15/0.30/0.45/0.60/0.80`。

### 65.3 结果

结论为 `liquidity_cycle_world_model_not_enough`。valid 选择了 pure liquidity 模型，但该模型在 forward 明显失效；若人为看 anchor 权重，`w=0.45` 和 `w=0.80` 的 forward Top20 略高于 base，但 valid 不会稳定选中，而且 Top10 被削弱、dev 年度仍不稳。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | valid rank IC | forward rank IC | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pure | `+1.1718%` | `+0.4567%` | `+0.2022%` | `+0.6299%` | `+0.0532` | `-0.0129` | valid 最强，forward 反转 |
| anchor `w=0.0` base | `+0.6546%` | `+0.0591%` | `+1.2577%` | `+2.1353%` | `+0.0019` | `+0.0116` | 原始基准 |
| anchor `w=0.45` | `+0.6735%` | `+0.1452%` | `+1.4023%` | `+1.4007%` | `+0.0197` | `+0.0063` | Top20 略高，但削 Top10 |
| anchor `w=0.30` | `+0.6376%` | `+0.1040%` | `+1.2305%` | `+1.7165%` | `+0.0128` | `+0.0080` | 低于 base Top20 |
| anchor `w=0.80` | `+0.6157%` | `+0.2585%` | `+1.3661%` | `+1.3099%` | `+0.0326` | `+0.0078` | dev 较好但 valid 不选 |
| anchor `w=0.60` | `+0.5875%` | `+0.1677%` | `+1.2827%` | `+1.2203%` | `+0.0233` | `+0.0082` | 仅小幅贴近 base |

pure 模型的 dev 分年 Top20 为：2021 `+0.8697%`，2022 `-0.1844%`，2023 `-0.0640%`，2024 `+0.4908%`，2025 `+1.1591%`。anchor `w=0.45` 的 dev 分年 Top20 为：2021 `+0.2654%`，2022 `-0.2874%`，2023 `-0.2139%`，2024 `+0.1918%`，2025 `+0.7625%`。均未满足五年全正。

### 65.4 观察

第一，流动性状态有历史信号，但不是可直接外推的排序器。pure 模型在 valid Top20 达到 `+1.1718%`，valid rank IC 为 `+0.0532`，说明 2025 的横截面里资金状态确实解释了未来 5 日收益；但 forward rank IC 变成 `-0.0129`，Top20 只有 `+0.2022%`，说明它把 2026 的头部弹性大幅打薄。

第二，anchor 可以把损伤压低甚至小幅提高 Top20，但代价是削 Top10。`w=0.45` forward Top20 为 `+1.4023%`，略高于 base `+1.2577%`；但 forward Top10 从 base `+2.1353%` 降到 `+1.4007%`。这不符合当前目标，因为目标需要找到更强 alpha，而不是牺牲尖锐头部来换平均层的微小改善。

第三，2025 valid 容易追逐 pure。按 valid score，pure 排第一；但 pure 正是 forward 最差的版本。说明“用 2025 选择模型强度”会高估资金状态 residual 的可迁移性。

第四，dev 年度没有修复。无论 pure 还是 anchor，2022/2023 都仍为负。资金状态解释变量能增加平均 IC，却没有解决目标里最关键的年度稳定性。

### 65.5 反事实分析

第一反事实：如果第 64 后价格 prototype 失败只是缺少资金状态，那么 liquidity residual 应在 forward 明显超过 base。实际 pure forward Top20 仅 `+0.2022%`，否定。

第二反事实：如果流动性周期是稳定 alpha，valid 选择的 pure 模型应在 forward 继续有效。实际 valid rank IC 为正、forward rank IC 转负，说明 2025 学到的是阶段性资金偏好，不是稳定排序规则。

第三反事实：如果 anchor `w=0.45` 的 forward Top20 提升代表真实方向，它不应严重削弱 Top10。实际 Top10 损失约 `0.73%`，说明 residual 仍在移动最有价值的头部候选。

第四反事实：如果资金状态能修复年份约束，dev 2022/2023 应转正。实际仍为负，说明它最多是条件变量，不能直接作为主排序器。

### 65.6 判定

`liquidity_cycle_world_model_not_enough`。

本轮没有可融入策略，也不进入 formal。关键结论不是“流动性无效”，而是“流动性 residual 不能无条件重排”。更合理的下一步不是继续让 2025 valid 选 pure，而是把流动性状态离散成因果状态：只在“脉冲扩散未耗尽”这类状态里轻量启用 residual 或 gate，在“脉冲衰减/耗尽/无脉冲”状态里回到 base。

下一轮做 `liquidity_cycle_causal_state_bucket_v1`：用 train-only 或滚动过去窗口阈值，把 session/股票组状态分成扩散、衰减、无脉冲等桶，不做完整重排，只测试因果 gate 和轻 anchor residual 是否能保住 2026 Top10/Top20，同时改善 dev 2022/2023。

## 66. liquidity_cycle_causal_state_bucket_v1

### 66.1 假设

第 65 轮说明 liquidity residual 不能无条件重排：2025 valid 很强，但 2026 forward 反转。一个更谨慎的假设是，流动性不是排序器，而是条件变量。只有在某些因果可见的状态桶里，流动性 residual 才应该轻量生效；在衰减、耗尽或无脉冲状态里，应回到 base 或惩罚。

本轮不引入新数据，不看 forward 调阈值。所有状态阈值只从 `2021-2024` train 的 liquidity 特征分位数得到，然后把每个候选划入 `pulse_diffusing_fresh`、`pulse_accelerating`、`pulse_fading`、`pulse_exhausted`、`no_pulse`、`neutral`。评估只做三类轻量动作：状态 gate only、只在 active bucket 内启用 residual、只在 active bucket 内启用正 residual，并可对 avoid bucket 加惩罚。

### 66.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_liquidity_cycle_causal_state_bucket_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/liquidity_cycle_causal_state_bucket_v1_summary.json
script sha256:f43be3d14383dc210aec1a6c2524acb5ee4e535be12773f7c489199fc792b04d
summary sha256:ace274c9f375cb9eacd7a4fb4d1c98601d6147fde186d257f5c1b8fe6b30eeb0
```

样本数仍为 train 194、valid 49、dev 243、forward 24 个 session。候选域为 `mid_trend_volume_not_extreme` top60。共扫描 577 个 gate/residual 配置，选择标准只使用 valid Top20、valid Top10、valid rank IC，以及 dev 最差年份的小惩罚，不使用 forward。

### 66.3 结果

结论为 `liquidity_cycle_causal_state_bucket_not_enough`。valid 选择的配置为：active bucket = `pulse_diffusing_fresh + neutral`，avoid bucket = `pulse_fading`，模式为 `active_residual`，`weight=0.15`，`avoid_penalty=0.20`。它提高了 valid Top20 和 dev 平均值，但 forward Top20、Top10 均低于 base，dev 2022/2023 仍为负。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward Top20 overlap | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.6546%` | `+0.0591%` | `+1.2577%` | `+2.1353%` | `100.00%` | 原始基准 |
| selected active residual `w=0.15` | `+0.7610%` | `+0.0802%` | `+1.2249%` | `+1.8884%` | `97.71%` | valid 改善，forward 低于 base |
| active residual `w=0.30` | `+0.7439%` | `+0.0880%` | `+1.3266%` | `+1.7374%` | `96.04%` | forward Top20 略高，但 Top10 损伤明显 |
| gate only `fresh` bonus `0.06` | `+0.7316%` | `+0.0800%` | `+1.2416%` | `+2.0841%` | `--` | 接近 base，但仍未超过 |
| gate only `fresh+neutral` bonus `0.03` | `+0.7317%` | `+0.0935%` | `+1.2908%` | `+2.0377%` | `98.54%` | 小幅 Top20，但不解决年度 |

selected 的 dev 分年 Top20 为：2021 `+0.1925%`，2022 `-0.3177%`，2023 `-0.4146%`，2024 `+0.1164%`，2025 `+0.8169%`。相比 base 的 2022 `-0.3529%`、2023 `-0.4524%` 略有改善，但离五年全正很远。

### 66.4 状态桶观察

本轮最重要的结果不是 selected 配置，而是状态桶的含义发生了翻转。

| 状态桶 | dev label mean | valid label mean | forward label mean | 观察 |
| --- | ---: | ---: | ---: | --- |
| `neutral` | `+0.2273%` | `+1.1348%` | `+0.9609%` | 稳定较好，但不是强右尾 |
| `no_pulse` | `+0.1012%` | `+0.9395%` | `+0.3818%` | 2025 偏好，2026 变弱 |
| `pulse_accelerating` | `+0.0418%` | `+0.6382%` | `+1.7470%` | 2026 变成强桶 |
| `pulse_diffusing_fresh` | `-0.2374%` | `+0.2699%` | `+0.0234%` | “新鲜扩散”并不强 |
| `pulse_exhausted` | `-0.1343%` | `+0.5312%` | `+2.3986%` | 2026 最强，但历史均值不支持 |
| `pulse_fading` | `-0.6156%` | `+0.1646%` | `+0.2745%` | dev 明显差，avoid 有道理 |

这张表否定了本轮最初的命名先验：所谓 `pulse_diffusing_fresh` 并没有成为主要右尾土壤；2026 真正高收益的反而是历史上偏负的 `pulse_exhausted`，以及 `pulse_accelerating`。也就是说，2026 的强势资金状态可能不是“低位扩散未耗尽”，而是“高热度仍继续挤压上行”的趋势抱团阶段。

### 66.5 反事实分析

第一反事实：如果 liquidity state bucket 是稳健条件变量，那么 train-only 阈值下的 active bucket 应在 dev、valid、forward 都有一致优势。实际 selected valid Top20 提升，但 forward Top20 低于 base，否定。

第二反事实：如果 `pulse_diffusing_fresh` 是正确的右尾土壤，它的 label mean 应至少在 dev/forward 为正且领先。实际 dev 为 `-0.2374%`，forward 仅 `+0.0234%`，说明命名先验错误。

第三反事实：如果惩罚 `pulse_fading` 足以修复第 65 的问题，Top10 不应被削弱。实际 selected forward Top10 从 `+2.1353%` 降到 `+1.8884%`，仍然动到了头部 winner。

第四反事实：如果 2026 的 `pulse_exhausted` 强只是噪声，那么 gate 不应在 saved leaderboard 中反复出现“避免 exhausted 反而不占优”。实际 2026 bucket mean 最高的是 `pulse_exhausted`，说明至少当前市场阶段里，高热度状态不能简单当作衰竭。

第五反事实：如果状态分桶能解决年度稳定性，dev 2022/2023 应转正。实际 selected 仍为 2022 `-0.3177%`、2023 `-0.4146%`，否定。

### 66.6 判定

`liquidity_cycle_causal_state_bucket_not_enough`。

本轮没有可融入策略，也不进入 formal。流动性状态分桶确实提供了一个更清晰的观察框架：`pulse_fading` 在 dev 中明显差，适合作为风险状态；但“fresh diffusion”不是金矿，2026 的高收益更像是“热度延续/拥挤趋势继续挤压”，这与过去两年历史均值相冲突。

下一轮不应继续在当前手工桶上调参。第 67 轮应做 `liquidity_state_regime_flip_diagnostic_v1`：专门分析状态桶在不同年份、市场宽度、指数趋势、候选排名层中的收益翻转，判断 2026 的 `pulse_exhausted` 强势是特殊市场阶段、样本偶然，还是 base 策略本来就在捕捉高热度延续。若确认是市场阶段依赖，再考虑做一个“市场阶段 router”而不是股票级固定状态阈值。

## 67. liquidity_state_regime_flip_diagnostic_v1

### 67.1 假设

第 66 轮最刺眼的结果是状态桶含义翻转：`pulse_exhausted` 在 dev 里偏弱，在 2026 forward 却是最强桶。此时不能继续调 gate，因为那会变成用 forward 事实反向改命名。第 67 轮只做诊断，问题是：2026 的高热度桶强势到底是状态本身的可迁移 alpha、市场阶段依赖，还是 base rank 层里的局部现象。

本轮复用第 66 的 train-only 状态桶阈值，不重新训练模型，不做策略选择。新增全市场可因果观测的市场状态特征：全市场 20/60 日平均收益、上涨宽度、20/120 日成交额比、5/20 日成交额比、20 日收益离散度。再把每个候选按年份、市场状态、base rank 层和流动性桶聚合，观察收益翻转来自哪里。

### 67.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_liquidity_state_regime_flip_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/liquidity_state_regime_flip_diagnostic_v1_summary.json
script sha256:dd7d37e161dabb62c1f1b01f81a0495c505b89ef7167fedd776ee2f5ca2a1863
summary sha256:f12947e55f92fd2f8921f752d67dbc2a4492b37dfd2f4a9a760af8ce168808ee
```

样本数：train 194、valid 49、dev 243、forward 24 个 session。候选行数：train 11640、valid 2940、dev 14580、forward 1440。base rank 层分为 `rank01_10`、`rank11_20`、`rank21_40`、`rank41_60`。

### 67.3 结果

结论为 `liquidity_state_regime_flip_confirms_market_phase_dependency`。2026 的 hot bucket 强势非常明显，但它不是 dev 中稳定存在的收益结构。

| 对象 | dev label mean | forward label mean | dev top20 label mean | forward top20 label mean | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `pulse_accelerating` | `+0.0418%` | `+1.7470%` | `-0.5621%` | `+0.5444%` | 2026 从弱桶变强桶 |
| `pulse_exhausted` | `-0.1343%` | `+2.3986%` | `+0.0916%` | `+1.3831%` | 2026 最强，dev 不支持 |
| 两个 hot bucket 平均 | `-0.0462%` | `+2.0728%` | `--` | `--` | 典型 regime flip |

分年看，`pulse_exhausted` 并非历史稳定强：2021 `+0.5753%`、2022 `+0.0827%`、2023 `-1.8618%`、2024 `+0.2962%`、2025 `-0.1064%`，到 2026 才跳到 `+2.3986%`。`pulse_accelerating` 也类似：2021 `+0.0691%`、2022 `-0.6952%`、2023 `-0.0843%`、2024 `-0.0989%`、2025 `+0.7026%`，2026 为 `+1.7470%`。

### 67.4 市场阶段和 rank 层观察

第一，forward hot bucket 的强势与市场阶段强相关：

| market regime + bucket | count | label mean | top20 count | top20 label mean | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `market_down_liq_down + pulse_exhausted` | 11 | `+4.1739%` | 1 | `+5.0571%` | 小样本但极强 |
| `market_down_price_only + pulse_exhausted` | 14 | `+3.8372%` | 2 | `+6.0418%` | 小样本极强 |
| `market_up_liq_up + pulse_accelerating` | 166 | `+2.4440%` | 51 | `+2.5332%` | 较大样本，最可信 |
| `market_up_liq_up + pulse_exhausted` | 51 | `+2.3317%` | 22 | `+0.7688%` | 整体强，Top20 不尖 |
| `market_down_liq_down + pulse_accelerating` | 77 | `+2.3298%` | 22 | `-1.4879%` | 全域强但 base Top20 反而弱 |

第二，forward 的最强收益有明显 rank 层依赖，尤其集中在 base 排名靠后的 `rank41_60`：

| bucket + rank layer | count | forward label mean | dev label mean | 观察 |
| --- | ---: | ---: | ---: | --- |
| `pulse_exhausted + rank41_60` | 30 | `+3.7885%` | `--` | 2026 最强，不在 base Top20 |
| `pulse_accelerating + rank41_60` | 116 | `+3.4407%` | `+0.2732%` | forward 明显放大 |
| `pulse_exhausted + rank21_40` | 26 | `+1.8105%` | `-0.2024%` | 2026 翻正 |
| `pulse_accelerating + rank21_40` | 121 | `+1.1668%` | `+0.3382%` | 有一定延续 |

这说明第 66 轮固定 gate 失败的一个原因是：hot bucket 在 2026 最强的位置不在 base Top20，而在 rank41_60。若只在 top20 轻微排序，抓不到；若大幅把 rank41_60 拉上来，又会重演第 63-64 轮扩池失败的风险。

第三，dev 里并没有同样强的 rank41_60 hot bucket。dev 中 `pulse_accelerating + rank41_60` 只有 `+0.2732%`，`pulse_exhausted + rank21_40` 为 `-0.2024%`；2026 的强度不是历史平均能自然推出的。

### 67.5 反事实分析

第一反事实：如果 `pulse_exhausted` 是稳定 alpha，它应该在 dev 多数年份和 rank 层里都强。实际 2023 大负、2025 也负，否定。

第二反事实：如果 2026 hot bucket 强只是 base top20 本来就捕捉到了，那么 hot bucket 的 top20 label mean 应接近整体 hot bucket。实际 `pulse_accelerating` forward 整体 `+1.7470%`，top20 仅 `+0.5444%`；最强收益反而在 rank41_60，说明 base top20 没吃满这块。

第三反事实：如果只要惩罚衰竭/追新鲜扩散即可，`pulse_diffusing_fresh` 应在 forward 领先。实际 `pulse_diffusing_fresh` forward 只有 `+0.0234%`，再次否定。

第四反事实：如果市场状态无关，那么 `market_up_liq_up + pulse_accelerating` 不应显著强于普通 hot bucket。实际该组合样本 166、均值 `+2.4440%`，明显高于 hot bucket 平均，说明需要市场阶段 router。

第五反事实：如果可以直接做 rank41_60 拉升，dev 中对应层应已稳定正且年度不差。实际 dev 只弱正甚至为负，不能直接规则化。

### 67.6 判定

`liquidity_state_regime_flip_confirms_market_phase_dependency`。

本轮没有可融入策略，也不进入 formal。它给出一个重要方向：当前市场下，右尾可能来自“市场阶段允许的热度延续”，而不是普适的流动性新鲜扩散；但这个方向必须用市场阶段 router 约束，否则会在 2022/2023 被反噬。

下一轮做 `market_phase_hot_liquidity_router_v1`。约束如下：只用 train/valid 选择市场阶段 router，不用 forward；只允许在被市场阶段确认的 session 内把 `pulse_accelerating/pulse_exhausted` 的 rank21_60 候选小幅上提；同时必须检查 dev 2022/2023 是否恶化、forward Top10 是否被削弱。若仍然只提升 forward rank41_60 但 valid/dev 不支持，则说明 2026 hot continuation 是前向阶段机会，不能用当前历史窗训练成稳健策略。

## 68. market_phase_hot_liquidity_router_v1

### 68.1 假设

第 67 轮显示，2026 的 hot bucket 强势具有明显市场阶段和 base rank 层依赖，尤其 `pulse_accelerating/pulse_exhausted` 在 `rank21_60` 中被放大。一个自然反事实是：如果这种热度延续不是纯 forward 偶然，那么只用 `2021-2024` train 和 `2025` valid 选择市场阶段 router，应该能在 2026 保住 Top10 的同时提高 Top20，并且不能恶化 dev 2022/2023。

本轮只做轻量 router，不训练新深度模型，不看 2026 调参。动作限定为：当市场阶段落入 selected active regimes 时，把指定 hot buckets 和 rank layers 的候选加一个固定 boost；当不在 active regimes 且配置启用 cool penalty 时，对这些 hot 候选小幅降权。核心测试是：这种“市场阶段允许的热度延续”能否成为可迁移规则，而不是事后解释。

### 68.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_market_phase_hot_liquidity_router_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/market_phase_hot_liquidity_router_v1_summary.json
script sha256:a7365109fd6ae494798e67ba13fddcedb1d9457078e6b9edd3dca079d4f38b5f
summary sha256:c85cecbdf5af3915342885f00305caa8355c1a9ecdcf781fd51a3c1400c43429
```

样本数：train 194、valid 49、dev 243、forward 24 个 session。候选域仍为 `mid_trend_volume_not_extreme` top60。扫描 2017 个配置，包含 active market regime 集合、hot bucket 集合、rank layer 集合、boost 和 cool penalty。选择标准只使用 valid Top20、valid Top10、valid rank IC、dev 最差年份惩罚和 valid 换手惩罚，不使用 forward。

### 68.3 结果

结论为 `market_phase_hot_liquidity_router_not_enough`。valid 选择的配置为：active regimes = `market_up_liq_up + liq_up_high_dispersion`，hot buckets = `pulse_accelerating + pulse_exhausted`，rank layers = `rank11_20 + rank21_40 + rank41_60`，boost `0.25`，cool penalty `0.10`。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | dev 最差年 | forward overlap | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.6546%` | `+0.0591%` | `+1.2577%` | `+2.1353%` | 2023 `-0.4524%` | `100.00%` | 原始基准 |
| selected router | `+0.7958%` | `+0.0457%` | `+1.2732%` | `+1.7671%` | 2023 `-0.4525%` | `90.00%` | valid 好，forward Top10 受伤 |
| valid rank 2 router | `+0.7673%` | `+0.0663%` | `+1.3880%` | `+1.8879%` | 2023 `-0.4072%` | `84.58%` | Top20 小幅提升，换手和 Top10 损伤较大 |
| forward saved best | `+0.7584%` | `+0.0665%` | `+1.7187%` | `+2.1588%` | 2023 `-0.4258%` | `87.50%` | forward 很好，但不是 valid 最优且年度不稳 |
| forward saved rank 2 | `+0.7720%` | `+0.0627%` | `+1.7070%` | `+2.1588%` | 2023 `-0.4162%` | `87.29%` | 同样依赖 forward 观察 |

selected router 在 valid 的 active sessions 为 22，active session 中 promoted pool share `30.45%`，promoted top20 share `32.73%`；在 forward 的 active sessions 为 13，promoted pool share `27.82%`，promoted top20 share `32.69%`。它确实把一批 hot rank11_60 候选上提了，但收益没有稳定转化为策略级改善。

selected 的 dev 分年 Top20 为：2021 `+0.2874%`，2022 `-0.3205%`，2023 `-0.4525%`，2024 `+0.0154%`，2025 `+0.6908%`。相比 base，2022 略改善，但 2023 没改善，2024/2025 变弱，dev 平均也从 `+0.0591%` 降到 `+0.0457%`。

### 68.4 观察

第一，router 能解释一部分 forward 机会，但 valid 选择不够稳。saved leaderboard 中 forward 最好的配置 Top20 达 `+1.7187%`，Top10 `+2.1588%`，看起来超过 base；但它不是 valid 第一，而且 dev 最差年份仍为负。这是典型的“方向有金子味，但选择器还不能无未来函数地拿到”。

第二，selected router 明显伤害 Top10。forward Top10 从 base `+2.1353%` 降到 `+1.7671%`，即使 Top20 微增到 `+1.2732%`，也不是目标形态。我们需要的是更强右尾，而不是把更宽的候选层平均化。

第三，提升 `rank11_60` 的动作带来不小换手。selected forward Top20 overlap 只有 `90.00%`，valid 为 `91.12%`；valid rank 2 的 forward overlap 低到 `84.58%`。这说明 router 已经不是温和解释器，而是在改变核心持仓结构。

第四，年度约束仍未修复。即便 forward saved best 的 Top20 很强，dev 2023 仍约 `-0.4258%`，与目标“五年全正”相距很远。

第五，市场阶段 router 的信号更像“机会状态”，不是独立 alpha。它需要另一个更强的 session-level 判断器来决定何时把 hot continuation 当作可交易状态，否则容易在 2022/2023 被反噬。

### 68.5 反事实分析

第一反事实：如果第 67 发现的 market-phase hot continuation 是稳健规则，valid 选中的 router 应在 forward 明显超过 base 且不伤 Top10。实际 selected forward Top20 只从 `+1.2577%` 到 `+1.2732%`，Top10 大幅下降，否定。

第二反事实：如果 `rank21_60` hot 候选可以直接上提，dev 平均应该提高。实际 selected dev Top20 低于 base，说明直接上提会污染历史大多数状态。

第三反事实：如果 forward saved best 是可迁移配置，它应被 valid 排在前面且 dev 年度更健康。实际它不是 valid 第一，dev 2023 仍负，说明不能作为策略晋级。

第四反事实：如果 cool penalty 能解决非 active 状态的热度反噬，2022/2023 应明显改善。实际 2023 几乎不改善，说明问题不是简单“非 active 状态惩罚”能解决。

第五反事实：如果 market phase router 已足够，它不应需要大幅降低 overlap。实际强 forward 配置 overlap 约 `87%`，说明它在用较大持仓改动换收益，鲁棒性仍不足。

### 68.6 判定

`market_phase_hot_liquidity_router_not_enough`。

本轮没有可融入策略，也不进入 formal。第 65-68 轮合并结论是：流动性/市场阶段确实解释 2026 的一部分右尾，尤其热度延续在特定市场阶段和 rank21_60 层中很强；但用 train/valid 选择的固定 router 还不能稳定拿到收益，且容易削弱 Top10、无法修复 2022/2023。

下一步不应继续手调 router。优先做两件事：第一，建立 `.tmp` 下的 cached session tensor，把第 65-68 反复构造的 top60、label、liquidity bucket、market regime、rank layer、基础特征缓存下来，提高迭代速度；第二，换成更像 world model 的 session-level 动态判断：`session_phase_hot_continuation_meta_model_v1`，用过去若干 session 的市场宽度、成交额、bucket payoff、base rank 层收益来预测“下一期是否允许 hot continuation 上提”。如果这个 session-level router 仍不能在 valid/forward 同时成立，就应暂时离开流动性状态分支，转向新的横截面时间序列模型或 QMT 分钟级微结构方向。

## 69. session_phase_hot_continuation_meta_model_v1

### 69.1 假设

第 68 轮固定 market router 无法稳定拿到 2026 的 hot continuation 机会，核心问题是“何时允许上提 hot rank11_60”没有被 train/valid 稳定识别。第 69 轮换成 session-level meta model：不再固定某几个 market regime，而是用当前市场宽度、成交额状态、hot bucket 占比，以及过去若干 session 中 base/hot continuation 的已实现 payoff，预测下一期是否允许 hot continuation 上提。

本轮还承担一个工程目标：把第 65-68 反复构造的 session 表缓存到 `.tmp`，后续同一分支可以直接读取 tensor，减少每轮 3-4 分钟的重复构造。

### 69.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_session_phase_hot_continuation_meta_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/session_phase_hot_continuation_meta_model_v1_summary.json
.tmp/quantx-research/deep-learning-alpha-search-v1/session_tensor_cache_v1.npz
.tmp/quantx-research/deep-learning-alpha-search-v1/session_tensor_cache_v1_meta.json
script sha256:297e06c5cea71e34e3373666e53db7aabc8e133584e757198bfe326f36a0a7ff
summary sha256:164995b4ef6c1c2427fd3045e0ecbdec575d7554711c94d9d0f6a35b36643d00
cache sha256:ae088eb1739cb962613b81d1c440d570642c6360e7f3b5b39bcaaf7f25b79a93
cache meta sha256:40863efa3496fa6ccb6fe86a075a565146cfc52d0bf14d5151b7ebcbcff0d8c6
```

缓存样本数：train 194、valid 49、forward 24 个 session。每个 session 保存 top60 未来 5 日 label、liquidity bucket、market regime、rank layer、全市场状态特征。meta model 为 ridge 线性分类器，训练集 `2021-2024`，valid `2025` 选择 alpha、阈值和 boost，forward `2026` 只评估。

标签定义为：在该 session 启用 hot continuation 后，Top20 相比 base 提高超过 `0.10%`，且 Top10 损伤不超过 `0.30%`。hot continuation 只作用于 `pulse_accelerating/pulse_exhausted` 且 rank layer 为 `rank11_20/rank21_40/rank41_60` 的候选。

### 69.3 结果

结论为 `session_phase_hot_continuation_meta_model_not_enough`。缓存成功，但 meta router 没有学出可迁移开关。valid 选中的配置为 boost `0.10`、alpha `3.0`、threshold `0.40`；它在 2026 的 allow rate 为 `0%`，等于完全退回 base。

| 配置 | train allow | valid allow | forward allow | valid Top20 | forward Top20 | forward Top10 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `--` | `--` | `--` | `+0.6546%` | `+1.2577%` | `+2.1353%` | 原始基准 |
| selected meta router `boost=0.10` | `5.15%` | `16.33%` | `0.00%` | `+0.7006%` | `+1.2577%` | `+2.1353%` | valid 小增，forward 不触发 |
| `boost=0.40` alt | `28.87%` | `22.45%` | `16.67%` | `+0.6703%` | `+1.2548%` | `+2.2920%` | Top10 增，Top20 不增 |
| `boost=0.15` alt | `4.64%` | `4.08%` | `4.17%` | `+0.6687%` | `+1.2610%` | `+2.0708%` | 几乎贴 base |
| `boost=0.25` alt | `29.38%` | `14.29%` | `12.50%` | `+0.6780%` | `+1.2443%` | `+2.0128%` | forward 低于 base |

always-hot 对照显示，forward 中 hot continuation 本身仍有机会：selected boost `0.10` 的 always-hot forward Top20 为 `+1.4353%`，高于 base `+1.2577%`；但 meta router 没有在 2026 识别出这些 session。label rate 也说明 2026 并非没有机会：train `16.49%`、valid `24.49%`、forward `33.33%`。

### 69.4 观察

第一，缓存工程成功。第 69 轮首次构建缓存并完成 meta model 总耗时约 39 秒，明显短于第 68 的 230 秒；后续同一分支可直接读取 `.npz`。

第二，meta router 的核心失败是“不敢开”。selected 在 valid 开 `16.33%` session、Top20 小幅改善，但在 forward 一个 session 都不开。这不是收益模型成功，而是分布漂移下的过度保守。

第三，forward 的机会仍然存在，但 current meta features 抓不到。selected boost `0.10` 的 always-hot forward Top20 `+1.4353%`，而 model Top20 退回 `+1.2577%`。说明问题不是 hot continuation 完全消失，而是 session-level 判断器没有识别 2026 的机会状态。

第四，boost 更大的替代模型也不够。`boost=0.40` 在 forward Top10 到 `+2.2920%`，但 Top20 只有 `+1.2548%`，没有超过 base；`boost=0.25` 甚至 Top20 低于 base。这说明简单 ridge meta router 不足以把机会转成稳定收益。

第五，年度目标仍然远。selected train 分年 Top20 仍有 2022 `-0.3529%`、2023 `-0.4524%`，valid 2025 虽为正，但五年全正完全没有解决。

### 69.5 反事实分析

第一反事实：如果第 68 的失败只是固定 market regime 太粗，那么 session-level meta model 应该在 forward 打开一部分 hot continuation。实际 selected forward allow rate 为 `0%`，否定。

第二反事实：如果 hot continuation 在 2026 已经消失，always-hot forward 不应高于 base。实际 always-hot Top20 高于 base，说明机会仍在，失败来自识别器。

第三反事实：如果过去 session 的 payoff 能预测下一期 hot continuation，ridge meta router 应在 valid 和 forward 都有相近 allow rate。实际 valid `16.33%`、forward `0%`，说明 temporal payoff 特征不可迁移。

第四反事实：如果更激进 boost 可以解决，`boost=0.40` 应在 Top20 明显超过 base。实际 Top20 略低于 base，只提高 Top10，不能作为策略。

第五反事实：如果流动性状态分支已经接近可融入策略，至少 dev 年度最差项应改善。实际 2022/2023 仍负，否定。

### 69.6 判定

`session_phase_hot_continuation_meta_model_not_enough`。

本轮没有可融入策略，也不进入 formal。第 65-69 轮流动性/市场阶段分支可以阶段性收束：它解释了 2026 一部分机会，但不能用当前 train/valid 特征稳定识别，也无法修复 2022/2023。继续在这条线上调 gate，大概率是在围绕 forward 现象过拟合。

下一轮切换方向。保留缓存作为后续工具，但研究主线转向更贴近用户原始要求的 deep learning/world model：不再手工定义 hot bucket，而是构造横截面时间序列 token，让模型学习过去一段时间的横截面路径和 rank 层迁移。第 70 轮建议做 `cross_sectional_temporal_rank_transition_transformer_v1`：用缓存/现有候选域构造每个 session 的 top60 多期 rank/收益/状态序列，训练一个小 Transformer 或 gated MLP 预测未来 5 日右尾概率；如果仍然 valid 强、forward 弱，则进一步转向 QMT 分钟级微结构方向。

## 70. cross_sectional_temporal_rank_transition_transformer_v1

### 70.1 假设

第 65-69 轮说明，手工定义流动性 bucket 和 market router 能解释 2026 的一部分右尾，但无法稳定训练成可迁移开关。第 70 轮切换为更接近 world model 的问题定义：不再手工决定哪个状态是 hot，而是让模型学习横截面 rank slot 的时间迁移。

本轮用第 69 的 cached session tensor 构造 rank-slot token。每个 session 有 top60 候选，token 包含当前 rank layer、liquidity bucket、market regime、market features，以及历史 session 的同 rank slot 已实现 payoff/rank。为避免 5 日持仓重叠造成未来函数，故意排除 lag1，只使用 lag2、3、4、6、8、12。模型为小 Transformer，训练目标为未来 5 日横截面 rank label。

### 70.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_temporal_rank_transition_transformer_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_temporal_rank_transition_transformer_v1_summary.json
script sha256:e29892dbe7b8743805495203c8d8c7b0577ae72b6e07d9369da868a796a4b835
summary sha256:bf35ee27a1aae3d52d539ba0fb015e3deaecdebd749c6c46f843cd6bff73f94c
```

样本数因为需要最大 lag=12，train 从 194 个 session 降为 182，valid 49，forward 24。模型特征维度 39，训练 80 epoch，valid 选择 pure 或 anchor residual 权重，forward 严格只评估。

### 70.3 结果

结论暂定为 `cross_sectional_temporal_rank_transition_transformer_not_enough`，但这是近期最值得继续验伪的一轮。valid 选择 anchor `w=0.75`，dev 五年全正，forward Top20 高于 base，但收益提升没有达到策略晋级阈值，且 dev 强度远高于 forward，必须先做安慰剂诊断。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | dev 最差年 | forward overlap | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.6546%` | `+0.1115%` | `+1.2577%` | `+2.1353%` | 2023 `-0.4524%` | `100.00%` | 原始基准 |
| selected anchor `w=0.75` | `+0.9392%` | `+1.3691%` | `+1.4663%` | `+2.1020%` | 2025 `+0.9392%` | `72.29%` | 五年全正，forward 小幅增厚 |
| anchor `w=1.0` | `+0.8932%` | `+1.6343%` | `+1.1713%` | `+2.0967%` | 2025 `+0.8932%` | `65.63%` | dev 更强，forward 变弱 |
| pure model | `+0.9689%` | `+2.4281%` | `+1.5903%` | `+2.1642%` | 2025 `+0.9689%` | `36.04%` | forward 最厚但换手极高 |
| anchor `w=0.35` | `+0.8578%` | `+0.6935%` | `+1.5439%` | `+1.8715%` | 2023 `+0.2486%` | `88.54%` | Top20 好，Top10 受伤 |
| anchor `w=0.50` | `+0.8346%` | `+0.9904%` | `+1.5877%` | `+2.0418%` | 2023 `+0.6480%` | `81.88%` | forward 接近 pure，valid 不选 |

selected 的 dev 分年 Top20：2021 `+3.0280%`，2022 `+1.1057%`，2023 `+0.9595%`，2024 `+1.2108%`，2025 `+0.9392%`。这是到目前少见的五年全正结构。

### 70.4 观察

第一，rank transition 方向明显强于前面多数手工 bucket/ranker。selected forward Top20 从 base `+1.2577%` 到 `+1.4663%`，pure forward Top20 到 `+1.5903%`；同时 selected forward Top10 基本保住，`+2.1020%` 接近 base `+2.1353%`。

第二，最值得警惕的是 dev 过强。train/dev rank IC 分别约 `0.25/0.20`，forward IC 只有 `0.0435`。这种衰减可能是正常泛化折损，也可能是 rank-slot 历史收益本身存在样本机制偏差。

第三，pure 模型换手很高。pure forward overlap 只有 `36.04%`，但 Top20 `+1.5903%`；这说明模型确实在大幅重排 rank slot。selected anchor overlap `72.29%`，仍然不是很保守。

第四，dev 五年全正是亮点，但不能直接晋级。用户目标是五年几十倍到 100 倍级别的策略方向，本轮只是在 5 日横截面 Top20 均值上有改善，还没有组合级五年复利、平均持仓、成本和路径验证。

### 70.5 反事实分析

第一反事实：如果所有日线 ML 已经到上限，rank transition 不应明显优于 base。实际 dev 五年全正且 forward Top20 增厚，否定“完全无增量”。

第二反事实：如果这是纯过拟合，forward 应像前面很多模型一样明显低于 base。实际 selected 和 pure forward Top20 均高于 base，暂时不能判为纯过拟合。

第三反事实：如果模型主要靠未来函数，安慰剂/lag shift 应仍然给出异常强结果。第 71 必须验证这一点。

第四反事实：如果 rank-slot 记忆是真 alpha，打乱历史 lag 或移除 payoff lag 后，valid/forward 应明显下降。第 71 也必须验证。

第五反事实：如果该方向可策略化，后续组合回放应在成本后保持年度全正，并且平均持仓 >5。本轮尚未验证。

### 70.6 判定

`cross_sectional_temporal_rank_transition_transformer_not_enough`。

本轮没有可融入策略，也不进入 formal。但它是近期最值得继续的方向之一：相比手工流动性 router，rank-slot temporal world model 同时给出 dev 年度稳定性和 forward Top20 增厚。

下一轮必须先做 `rank_transition_leakage_placebo_diagnostic_v1`：包括移除 lag payoff、打乱 lag payoff、label 时间错位、只用 rank position/bucket 的弱模型，以及 walk-forward 式重训。若安慰剂也强，说明这轮结果可能来自 slot/样本偏差；若只有真实 lag2+ payoff 强，再进入组合级回放和更严谨的 transformer 复现实验。

## 71. rank_transition_leakage_placebo_diagnostic_v1

### 71.1 假设

第 70 轮的 rank-slot temporal transformer 给出了少见的 dev 五年全正和 forward Top20 增厚，但这种结果太强，必须先验伪。尤其样本只有 182 个 train session，模型又使用 rank-slot 历史 payoff，存在两类风险：第一，slot/market/bucket 结构本身已经能形成强 null；第二，训练选择可能在小样本上放大了与真实 lag 无关的样本偏差。

本轮复用第 70 的模型与评估框架，训练 4 个变体：真实 lag payoff、移除 lag payoff、打乱 lag payoff、训练标签错位到下一 session。若安慰剂也能跑出接近甚至超过真实 lag 的 valid/forward，则第 70 不能被当成有效 alpha 证据。

### 71.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_rank_transition_leakage_placebo_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/rank_transition_leakage_placebo_diagnostic_v1_summary.json
script sha256:901656677104c4093d99f9691aefb447e0604ad9fdfa3ee2d99ef2a6796f0ebf
summary sha256:cd40217d8558f11e3db8191e846d5ba14e17bf04b71cb02425e676bcbcf6d3c4
```

样本数同第 70：train 182、valid 49、forward 24。每个变体训练 35 epoch。评估时为每个变体选择 pure 或 anchor 权重，并记录 train/valid/dev/forward。

### 71.3 结果

结论为 `rank_transition_placebo_too_strong_possible_slot_bias`。安慰剂太强，尤其 `permuted_lag_payoff` 在 valid 上强于真实 lag，`train_label_next_session_shift` 在 forward 上甚至显著高于真实 lag。这说明第 70 的绝对收益不能作为模型有效性的证明。

| 变体 | selected mode | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward overlap | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| real lag payoff | pure | `+0.9262%` | `+0.3513%` | `+1.3287%` | `+1.0662%` | `36.25%` | 真实 lag 不强，Top10 大伤 |
| no lag payoff | anchor `w=0.5` | `+0.8497%` | `+0.4430%` | `+1.3647%` | `+1.5292%` | `78.33%` | 移除 lag 仍接近 |
| permuted lag payoff | pure | `+0.9380%` | `+1.1448%` | `+1.0371%` | `+1.1406%` | `36.88%` | valid/dev 很强，说明 null 很强 |
| train label next-session shift | pure | `+0.9260%` | `+0.2995%` | `+1.7345%` | `+2.1436%` | `42.92%` | 错位标签 forward 反而最强 |

真实 lag 的 dev 分年 Top20 为：2021 `+0.5754%`，2022 `+0.1308%`，2023 `-0.1344%`，2024 `+0.3081%`，2025 `+0.9262%`。这已经不再保持第 70 的五年全正。说明第 70 的强结果与训练轮数/随机种子/选择路径高度相关。

### 71.4 观察

第一，真实 lag 没有压倒安慰剂。真实 lag forward Top20 `+1.3287%`，只略高于 base `+1.2577%`，且 forward Top10 从 base `+2.1353%` 降到 `+1.0662%`。这不是可接受的右尾增强。

第二，no-lag 模型仍然有效。移除 lag payoff 后，valid Top20 `+0.8497%`，forward Top20 `+1.3647%`，说明当前 rank/bucket/market/slot 特征本身就有很强选择能力，不能把收益归因于时间迁移。

第三，permuted lag 在 valid/dev 上过强。打乱 lag payoff 后 dev Top20 `+1.1448%`，valid Top20 `+0.9380%`，接近第 70 的强结果。这说明模型可能在利用 slot/bucket/market 结构和训练选择器，而不是真实时间顺序。

第四，label shift 安慰剂最危险。训练标签错位后 forward Top20 `+1.7345%`、Top10 `+2.1436%`，看起来甚至优于真实模型。若错位标签都能如此强，说明当前 24 个 forward session 的检验样本太小，且 rank-slot 结构存在强样本噪声，不能凭一次 forward 好看晋级。

第五，第 70 的“dev 五年全正”不稳定。第 71 同一方向、较短训练后，real lag dev 2023 转负。这说明需要更严格的复现实验和 null 扣除，而不是继续加模型复杂度。

### 71.5 反事实分析

第一反事实：如果真实 lag payoff 是核心 alpha，real lag 应显著优于 no-lag 和 permuted lag。实际 no-lag forward 更高，permuted valid/dev 更强，否定。

第二反事实：如果第 70 没有 slot/sample bias，label shift 安慰剂不应在 forward 最强。实际错位标签 forward Top20 `+1.7345%`，说明当前框架的 forward 单期样本不能充分证明模型有效。

第三反事实：如果 rank-slot temporal 模型稳定，第 71 的 real lag 应保持 dev 五年全正。实际 2023 为负，否定稳定性。

第四反事实：如果收益来自可迁移时间顺序，打乱 lag 后应该明显失效。实际打乱 lag 在 valid/dev 仍强，说明时间顺序贡献未被证实。

第五反事实：如果这个方向可以直接进入组合级回放，Top10 不应被真实 lag 大幅削弱。实际 Top10 被腰斩，不能晋级。

### 71.6 判定

`rank_transition_placebo_too_strong_possible_slot_bias`。

本轮没有可融入策略，也不进入 formal。第 70 的结果被强安慰剂显著削弱，不能当作有效 deep learning alpha 证据。真正有价值的发现是：rank-slot/bucket/market 结构本身形成了很强 null，后续任何模型都必须先扣除这个 null，再看残差信号。

下一轮做 `rank_transition_slot_null_residual_alpha_v1`：先用 train-only 的 slot/bucket/market null 模型估计每个 token 的基准预期收益或 rank，再训练/评估 residual 模型是否能在扣除 null 后仍提供 valid/forward 增量。若 residual 不成立，则暂时放弃 rank-slot temporal 分支，转向 QMT 分钟级微结构或更宽的横截面因子库。

## 72. rank_transition_slot_null_residual_alpha_v1

### 72.1 假设

第 71 轮证明 rank transition 的安慰剂过强，尤其 no-lag、permuted lag、label shift 都能跑出相近甚至更强的结果。第 72 轮不再问“Transformer 能不能重排 rank slot”，而是先问一个更基础的问题：扣除 train-only 的 slot/bucket/market null 后，是否仍然存在可迁移 residual alpha。

本轮构造分层 null：只用 train 集，按 `slot_layer_bucket_regime -> slot_layer_bucket -> slot_layer -> slot` 做 shrinkage 估计每个 token 的基准预期 rank。随后训练一个小 residual MLP 预测 `label - null`，并比较 `null_only`、`residual_only`、`null_plus_residual`。如果 residual 不能在 valid 和 forward 同时超过 base/null，则第 70 的强结果应被判定为 slot/null/sample 结构，而不是可交易 deep learning alpha。

### 72.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_rank_transition_slot_null_residual_alpha_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/rank_transition_slot_null_residual_alpha_v1_summary.json
script sha256:b23af2bad4d9d872dbc7517b53f2d417b25d80d345dcf183eea781d4d82bdd40
summary sha256:46977377a8ab2cb921699f566dba37ce6c5465b83dfee0fd1be60792d38ff03b
```

样本数同第 70/71：train 182、valid 49、forward 24。null 的全局均值为 `0.008333`，分层桶数量为：`slot=6`、`slot_layer=6`、`slot_layer_bucket=36`、`slot_layer_bucket_regime=249`，shrinkage 参数为 `25.0`。residual MLP 训练 50 epoch，valid 只用于选择 `null_only / residual_only / null_plus_residual` 以及 residual 权重，forward 只评估。

### 72.3 结果

结论为 `rank_transition_slot_null_residual_alpha_not_enough`。valid 选中的是 `residual_only`，但它在 forward Top20 低于 base，并且 forward Top10 被严重削弱；`null_only` 在 valid/dev 很强，但 forward 崩溃，说明 slot null 自身就是不稳定样本结构。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward IC | forward overlap | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.6546%` | `+0.1115%` | `+1.2577%` | `+2.1353%` | `--` | `100.00%` | 原始基准 |
| selected `residual_only` | `+1.0145%` | `+0.2198%` | `+1.0841%` | `+0.9112%` | `+0.0264` | `33.75%` | valid 最强，forward 低于 base，Top10 大伤 |
| `null_only` | `+0.9531%` | `+0.6594%` | `+0.5270%` | `+1.0339%` | `-0.0307` | `--` | valid/dev 强但 forward 崩，典型 null 不迁移 |
| `null_plus_residual w=1.0` | `+1.0087%` | `+0.3076%` | `+0.8825%` | `+0.8425%` | `+0.0261` | `--` | 加 residual 后仍不如 base |
| `null_plus_residual w=0.50` | `+0.9683%` | `+0.3863%` | `+1.2064%` | `+0.6260%` | `+0.0220` | `--` | Top20 接近 base，但 Top10 明显受伤 |
| `null_plus_residual w=0.25` | `+0.8799%` | `+0.4571%` | `+1.1984%` | `+0.9810%` | `+0.0203` | `--` | 没有稳定增量 |

selected `residual_only` 的 dev 分年 Top20：2021 `+0.6836%`，2022 `-0.2376%`，2023 `-0.3551%`，2024 `+0.0953%`，2025 `+1.0145%`。这说明它没有修复年度稳定性，反而把 2022/2023 的问题暴露出来。

### 72.4 观察

第一，null 本身解释了第 70/71 的相当一部分“好看”。`null_only` valid Top20 `+0.9531%`，dev Top20 `+0.6594%`，但 forward Top20 只有 `+0.5270%`，forward IC 为 `-0.0307`。这意味着 rank slot、bucket、market regime 的历史均值可以在回测内制造强排序，但不是稳定 alpha。

第二，扣除 null 后 residual 没有成为新 alpha。selected `residual_only` valid Top20 到 `+1.0145%`，但 forward Top20 `+1.0841%`，低于 base `+1.2577%`；forward Top10 只有 `+0.9112%`，远低于 base `+2.1353%`。它不是右尾增强器，而是在大幅重排后损伤头部。

第三，`null_plus_residual` 没有形成互补。权重从 `0.25` 到 `1.5` 都不能同时改善 forward Top20 和 Top10；`w=0.50` 的 Top20 最接近 base，但 Top10 只有 `+0.6260%`。这说明 null 与 residual 的组合更像噪声叠加，不是可迁移结构。

第四，valid 选择仍然被小样本牵引。valid 越强的配置在 forward 越容易伤 Top10；这与第 71 的 label-shift 安慰剂一致，说明当前 49 个 valid session 不足以从 slot-heavy 特征中选出稳定模型。

第五，第 70 的“dev 五年全正”不能再作为方向证据。第 72 在更严格的 null/residual 框架下，selected dev 2022/2023 重新转负，且 forward 低于 base。rank transition 分支需要阶段性停止。

### 72.5 反事实分析

第一反事实：如果第 70 的收益来自真实时间迁移，扣除 slot/bucket/market null 后 residual 应仍在 forward 超过 base。实际 selected forward Top20 低于 base，否定。

第二反事实：如果 slot null 是稳定 alpha，`null_only` 应在 forward 保持 valid/dev 的强度。实际 forward Top20 从 valid `+0.9531%` 掉到 `+0.5270%`，且 IC 转负，否定。

第三反事实：如果 residual 与 null 互补，`null_plus_residual` 至少应出现一个权重同时保住 Top10 并提升 Top20。实际所有权重都不能满足，否定。

第四反事实：如果 valid 能可靠挑出配置，valid 第一的 `residual_only` 不应在 forward 损伤 base。实际 Top20 和 Top10 都低于 base，说明选择过程仍被样本噪声牵引。

第五反事实：如果 rank-slot temporal world model 是近期最可能出金子的方向，经过安慰剂和 null 扣除后不应连续失败。第 71/72 连续否定后，继续加深 Transformer 或调权重，大概率是在围绕样本结构过拟合。

### 72.6 判定

`rank_transition_slot_null_residual_alpha_not_enough`。

本轮没有可融入策略，也不进入 formal。第 70-72 的合并结论是：rank-slot temporal 模型能在小样本上制造漂亮的 valid/dev 和 forward 片段，但安慰剂、label shift、slot null 都过强；扣除 null 后 residual alpha 不成立。因此 rank transition 分支阶段性收束。

下一轮切换到更贴近真实交易路径的新方向：QMT 分钟级微结构。原因是当前日线 top60/rank-slot 缓存缺少个股连续路径，容易把横截面 slot 结构当成 alpha；而用户要求的持仓周期约一周，恰好适合用日内路径识别“当天被资金强推但未充分兑现”或“被挤兑后仍有承接”的事件，再预测未来 5 日横截面收益。第 73 轮优先做 `qmt_intraday_path_event_world_model_v1`：只用 QMT 可得分钟 K，构造开盘压力、VWAP 偏离、尾盘承接、日内回撤恢复、量能分布等事件 token，并严格做 train/valid/forward。若分钟缓存覆盖不足，则先做可用性诊断和样本覆盖报告，不把数据缺口伪装成模型结论。

## 73. qmt_intraday_path_event_world_model_v1 / cache_only_v1

### 73.1 假设

第 70-72 轮说明，日线 rank-slot temporal world model 容易把 slot/null/sample 结构误当成 alpha。第 73 轮切换到 QMT 分钟级路径事件：用当日 5m 路径识别“资金推动但未完全兑现”“尾盘仍有承接”“VWAP 上方吸收”“日内高位不过度衰竭”等状态，预测未来约一周收益。

本轮先按严格 train/valid/forward 口径尝试 `qmt_intraday_path_event_world_model_v1`：train `2021-2024`、valid `2025`、forward `2026`。由于直接拉取历史 5m 数据耗时过长，长跑被中断；随后新增 `cache_only_v1`，只读 `.tmp` 现有分钟缓存，不触发 QMT 下载，先回答一个基础问题：现有缓存是否足以训练和验证分钟级 deep learning 模型。

### 73.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_intraday_path_event_world_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_qmt_intraday_path_event_world_model_cache_only_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/qmt_intraday_path_event_world_model_cache_only_v1_summary.json
world model script sha256:24c72012869f75767fb978c988d19f0036b9178d52b2b9a0ee84d3a8067c70ca
cache-only script sha256:845bed768ccc17878fba47c58beac72b50da7e70661f52ca6cd2d9229120bf5c
summary sha256:912b6d26f961db5aec25cab3f8aa423c0abfe834250a177447a84b86b5d7d7b9
```

候选池复用日线 `mid_trend_volume_not_extreme`，每个 session 取前 50 个候选，提取 5m 路径特征：`intraday_return`、`first_half_return`、`second_half_return`、`tail_return`、`vwap_support`、`low_to_close_recovery`、`high_to_close_fade`、`drawdown_from_high`、`tail_volume_share`、`late_volume_share`、`volume_concentration_top20pct`、`smooth_path_score` 等。

事件规则包括：`daily_base`、`tail_vwap_recovery`、`late_absorption`、`anti_exhaustion_path`、`path_quality_blend`。若 train/valid/forward 覆盖满足最低门槛，脚本会训练一个小 MLP；若不满足，则只输出覆盖诊断和 forward 事件规则扫描，不进入训练。

### 73.3 覆盖诊断

结论为 `qmt_intraday_path_event_world_model_cache_only_insufficient_valid_coverage`。现有非空 QMT 5m 缓存只有 2026，不能支撑 train/valid/forward 训练闭环。

| split | planned sessions | sessions with rows | available rows | requested | available | empty | missing | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train `2021-2024` | 194 | 0 | 0 | 9700 | 0 | 3713 | 5987 | 2021/2022 为 1 字节占位，2023/2024 缺失 |
| valid `2025` | 10 | 0 | 0 | 500 | 0 | 0 | 500 | 完全无 valid 缓存 |
| forward `2026` | 24 | 24 | 1200 | 1200 | 1200 | 0 | 0 | forward 覆盖完整 |

缓存总体：`.tmp` 下共有 9831 个 5m csv 文件，但 3713 个为 1 字节占位文件；非空文件 6118 个，全部来自 2026。按年份统计：原始文件 `2021=2450`、`2022=1263`、`2026=6118`；非空文件只有 `2026=6118`。

因此本轮不能训练 deep learning 模型，也不能用 2026 forward 事件规则作为策略证据。严格意义上，第 73 是一次数据覆盖验证和方向预筛，而不是完整模型实验。

### 73.4 forward 事件扫描

虽然不能训练，cache-only 仍在 2026 forward 的 24 个 session、1200 个候选上扫描了事件规则。结果显示，分钟路径事件没有超过 daily base；反而 daily base 本身仍是 forward 最强。

| 规则 | forward Top20 | forward Top10 | forward rank IC | 判读 |
| --- | ---: | ---: | ---: | --- |
| `daily_base` | `+1.2577%` | `+2.1353%` | `+0.0147` | forward 最强，实际是日线基准 |
| `late_absorption` | `+0.8437%` | `+1.3472%` | `+0.0142` | 有一点方向，但低于 daily base |
| `anti_exhaustion_path` | `+0.5647%` | `+0.3488%` | `+0.0127` | 避免衰竭不够强 |
| `tail_vwap_recovery` | `+0.3902%` | `+0.4218%` | `-0.0001` | 尾盘/VWAP 修复无效 |
| `path_quality_blend` | `+0.3706%` | `+0.2633%` | `+0.0089` | 综合路径质量低于基准 |

forward random Top20 label 对照：p50 `+0.8594%`、p90 `+1.2516%`、p95 `+1.3997%`、p99 `+1.5586%`、max `+1.7137%`。`daily_base` 的 Top20 `+1.2577%` 接近 random p90，但低于 p95；分钟事件规则全部弱于 daily base。

### 73.5 观察

第一，分钟方向不是被模型否定，而是被有效训练数据覆盖卡住。没有 2025 valid，就无法做用户要求的 train/valid/forward 防过拟合闭环；没有 2023/2024 train，也无法训练一个像样的 regime-aware 模型。

第二，2021/2022 的缓存文件具有迷惑性。文件存在但多为 1 字节换行占位，不能算有效分钟 K 线。若只看文件数，会误判为 train 有覆盖；因此后续所有 QMT 分钟实验必须记录 `nonempty_files`、`available_rows`、`missing`、`empty`。

第三，2026 forward 的日线基准仍强于分钟路径规则。若分钟事件是真正强 alpha，在只有 2026 的 forward 扫描中至少应能显著超过 daily base；实际 `late_absorption`、`tail_vwap_recovery`、`path_quality_blend` 都更弱，说明当前手写路径事件并没有明显出金子。

第四，长跑中断是数据层问题，不是模型计算问题。直接拉取历史 5m 数据在逐日逐股上耗时过长，不能作为高频迭代默认路径；后续若继续分钟方向，应先设计批量缓存构建任务，而不是每次实验临时下载。

第五，这轮反过来支持一个判断：在缺少完整分钟历史之前，主线不应依赖 QMT 5m deep learning。否则会在“只用 2026 forward 看起来还可以”的陷阱里过拟合。

### 73.6 反事实分析

第一反事实：如果现有分钟缓存足以训练模型，train/valid 应有足够 `available_rows`。实际 train/valid 都是 0，否定。

第二反事实：如果分钟路径事件天然强，哪怕只看 2026 forward，也应明显超过 `daily_base`。实际所有分钟事件弱于 daily base，否定当前手工事件定义。

第三反事实：如果 2021/2022 文件数可以作为覆盖证据，读取后应能提取有效 5m bar。实际是 1 字节占位，否定。

第四反事实：如果继续在该脚本中等待 QMT 下载就能快速得到训练集，长跑不应超过快速验证窗口。实际长时间无输出并卡在历史分钟读取，说明需要单独的数据缓存任务，而不是混在模型实验里。

第五反事实：如果第 73 可以成为策略候选，至少应有 valid 选择和 forward 检验。实际 valid 为 0，不能晋级。

### 73.7 判定

`qmt_intraday_path_event_world_model_cache_only_insufficient_valid_coverage`。

本轮没有可融入策略，也不进入 formal。QMT 分钟级方向暂时不作为主线继续训练，除非先完成独立的历史分钟缓存构建，并保证 2021-2025 有可用 train/valid 覆盖。

下一轮需要换回可立即严格验证的数据域，但不能回到 rank-slot 过拟合。建议第 74 轮做 `cross_sectional_daily_path_memory_residual_v1`：只用现有日线 Qlib/QMT 日线可得数据，构造个股级过去 20/60 日路径 token，而不是 rank slot token；先用 train-only 的行业/市值/流动性/动量 null 扣除，再训练小型 TCN/GRU/MLP 预测未来 5 日 residual。核心问题从“rank 位置迁移”改成“个股路径在同类股票中的残差可预测性”，并继续保留安慰剂、label shift、year split 和 forward 检验。

## 74. cross_sectional_daily_path_memory_residual_v1

### 74.1 假设

第 73 轮说明 QMT 分钟方向暂时被历史覆盖卡住；第 70-72 轮又说明 rank-slot temporal 分支容易产生 null/sample 偏差。第 74 轮换成可立即严格验证的数据域：只用现有日线数据，构造个股级过去 60 日路径压缩特征，不再使用 rank-slot 记忆。

核心假设是：短期 alpha 不一定来自“当前一天横截面排名”，而可能来自个股路径在同类股票中的残差。先用 train-only 的流动性、动量、波动、VWAP 偏离分层 null 扣掉基础结构，再训练小 MLP 预测未来 5 日横截面 residual rank。若 residual 在 valid/forward 都有效，说明个股路径方向比 rank-slot 更值得继续；若 null 或 residual 安慰剂也强，则继续收束。

### 74.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_daily_path_memory_residual_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_daily_path_memory_residual_v1_summary.json
script sha256:9c6c3f56bcf60efb8e1fb975b80b36dee17528f259324c66e9f0eb82fe44ddf3
summary sha256:46c682041f00765046622b059d3ef03cbcb9de07aed725cebe5d5dc05810b373
```

样本：train 194 个 session、96677 行；valid 49 个 session、24362 行；dev 243 个 session、121039 行；forward 24 个 session、11914 行。候选池为全市场 pool500，持仓周期 5 日，组合回放 Top10，成本沿用既有 `POS.replay` 口径。

特征为个股过去 60 日日线路径压缩：last、mean5、mean20、mean60、std20、mean5-mean20 drift，再拼接日线 base score 与流动性层。null 只用 train：按 `liq_ret_vol_vwap -> liq_ret_vol -> liq_ret -> liq` 分层 shrinkage，null shrink `30.0`。模型为小 MLP，训练 35 epoch，valid 选 `base_daily / null_only / residual_only / null_plus_residual`。

### 74.3 结果

结论为 `cross_sectional_daily_path_memory_residual_not_enough`，但这是近期更值得继续验伪的方向。selected 为 `residual_only`，说明真正起作用的是路径 residual，而不是 train-only null。横截面指标明显好于本轮 base；组合 dev 五年全正，forward 为正，但尚未达到用户要求的几十倍，也没有通过 forward random p95 和 remove-best 稳健性。

| 配置 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward IC | forward overlap | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| selected `residual_only` | `+0.9431%` | `+0.9760%` | `+0.8453%` | `+0.4782%` | `+0.0793` | `4.38%` | valid/dev/forward 都有预测力，但 Top10 不够强 |
| `null_plus_residual w=1.0` | `+0.8247%` | `+0.9402%` | `+0.8744%` | `+0.6485%` | `+0.0616` | `2.29%` | forward Top20 稍高，valid 不选 |
| `null_plus_residual w=1.5` | `+0.8611%` | `+0.9453%` | `+0.7346%` | `+0.3581%` | `+0.0682` | `2.08%` | valid 次强，forward 弱 |
| `null_only` | `+0.4427%` | `+0.4160%` | `-0.6882%` | `-0.7222%` | `+0.0008` | `0.21%` | null 不迁移，不能解释 residual |
| `base_daily` | `-0.7855%` | `-1.1671%` | `+0.3617%` | `-0.6635%` | `+0.0114` | `100.00%` | 本轮 base 方向本身很差 |

selected dev 分年 Top20：2021 `+1.3358%`，2022 `+0.6145%`，2023 `+0.5937%`，2024 `+1.3938%`，2025 `+0.9431%`。forward 2026 Top20 为 `+0.8453%`。

### 74.4 组合回放

selected `residual_only` 组合回放：

| 区间 | final multiple | total return | annual returns | max drawdown | avg position | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| dev 2021-2025 | `9.2024x` | `+820.24%` | 2021 `+124.32%`, 2022 `+31.86%`, 2023 `+45.55%`, 2024 `+70.57%`, 2025 `+25.31%` | `-20.30%` | `10.00` | `8.3642x` | 五年全正，路径不错但不足几十倍 |
| forward 2026 | `1.1461x` | `+14.61%` | 2026 `+14.61%` | `-19.00%` | `9.96` | `0.8860x` | 为正，但去掉最佳 3 笔后不稳 |

本轮 base replay 很差：dev final multiple `0.0247x`，forward `0.8792x`。这说明 residual 模型确实在相同候选池内做了巨大重排，但也意味着它离 base overlap 极低，forward overlap 只有 `4.38%`，需要特别警惕训练选择偏差和隐性 label 结构。

forward random pool500 Top10 final multiple：p50 `1.0554x`，p90 `1.2517x`，p95 `1.3126x`，p99 `1.4241x`，max `1.5245x`。selected forward `1.1461x` 只高于 p50，低于 p90/p95，因此不能晋级。

### 74.5 观察

第一，个股路径 residual 方向明显比 rank-slot residual 更有生命力。第 72 扣除 slot null 后 residual forward Top20 低于 base，而第 74 residual 在 train/valid/dev/forward 都保持正 IC 和正 Top20，forward IC 到 `+0.0793`。

第二，null 被否定得比较干净。`null_only` valid 有一点正收益，但 forward Top20 为 `-0.6882%`，IC 接近 0；selected 是 `residual_only`，不是 null_plus。说明本轮不是简单复刻流动性/动量/VWAP 分层均值。

第三，组合 dev 质量不错但收益强度不够。五年 9.2 倍、全正、最大回撤 20.3%、平均持仓 10，已经比很多前序模型更像策略雏形；但距离用户要求的五年几十倍到 100 倍仍有大差距。

第四，forward 有正收益但不稳。2026 forward `+14.61%`，但 `remove_best_3=0.8860x`，说明少数交易对结果贡献过大；同时没有超过 random p90/p95，不能作为鲁棒 alpha。

第五，Top20 比 Top10 更强，提示模型更像“中上层广谱预测器”，不是极端右尾抓手。forward Top20 `+0.8453%`，Top30 `+1.0807%`，Top10 只有 `+0.4782%`。用户不认可降低尾部权重硬卡，后续不能简单扩大持仓来美化结果，而要提高右尾排序能力。

### 74.6 反事实分析

第一反事实：如果日线个股路径没有新增信息，residual-only 在扣除 null 后不应在 valid/forward 保持正 IC。实际 valid IC `+0.0915`、forward IC `+0.0793`，说明有真实预测力的可能。

第二反事实：如果收益只是流动性/动量/VWAP null，`null_only` 应该接近 selected。实际 `null_only` forward 为负，否定。

第三反事实：如果模型已经接近策略候选，forward 应至少超过 random p95，且 remove-best 后仍为正。实际 forward multiple 低于 random p90，`remove_best_3<1`，否定。

第四反事实：如果模型是强右尾 alpha，Top10 应强于 Top20/Top30。实际 forward Top10 最弱，说明还不是用户要的右尾抓取模型。

第五反事实：如果本轮没有训练选择偏差，label shift、permutation、no-path 特征安慰剂应明显弱于真实模型。第 75 必须验证。

### 74.7 判定

`cross_sectional_daily_path_memory_residual_not_enough`。

本轮没有可融入策略，也不进入 formal。但与第 70-73 相比，第 74 给出了更清晰的下一步：个股级日线路径 residual 可能有真实横截面预测力，值得继续做严谨验伪，而不是立即放弃。

下一轮做 `daily_path_memory_residual_placebo_v1`：保留同样数据、split、null 和评估，训练 real path、no-path compression、permuted labels、next-session label shift、base-only MLP 几个安慰剂。如果真实 residual 显著强于这些安慰剂，再继续做右尾强化、pairwise/listwise loss、TopK consistency 和组合 walk-forward；如果安慰剂同样强，则收束本方向。

## 75. daily_path_memory_residual_placebo_v1

### 75.1 假设

第 74 轮的个股级日线路径 residual 给出少见的正向证据：valid/forward IC 均为正，dev 五年全正。但它还没有达到策略要求，并且 forward remove-best 不稳。第 75 轮不急着加复杂模型，而是做安慰剂诊断：同样 split、同样 null、同样评估，只替换输入特征或训练标签，判断第 74 的收益到底来自真实 60 日路径，还是来自基础结构、小样本选择或标签泄漏。

本轮训练 6 个小 MLP 变体：`real_path`、`no_path_stats`、`last_day_only`、`base_liq_only`、`permuted_train_residual_label`、`next_session_shift_label`。如果真实路径不能显著压过安慰剂，不能继续右尾强化；如果标签安慰剂弱、无路径弱，但基础结构安慰剂在 valid 上过强，则说明方向有价值但 valid 选择必须更严格。

### 75.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_daily_path_memory_residual_placebo_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/daily_path_memory_residual_placebo_v1_summary.json
script sha256:a5477d40bb5ec310768913239b6fa3aa13a7b30ea76fb840ed399da7caaddba2
summary sha256:19a4398d7b3011edf3c4762b87961628eec2fedabf91d1b4242abd4404eb2d4e
```

样本同第 74：train 96677 行、valid 24362 行、dev 121039 行、forward 11914 行；session 数 train 194、valid 49、dev 243、forward 24。每个变体训练 22 epoch，评估 Top10/Top20、rank IC、dev/forward 组合回放、remove-best 和 random 对照。

### 75.3 结果

脚本 verdict 为 `daily_path_memory_residual_placebo_too_strong`。这个结论需要细分理解：标签错位和标签打乱明显失败，说明不是纯标签泄漏；无路径和只看 last-day 在 forward 也失败，说明 60 日路径有增量；但 `base_liq_only` 在 valid 上排第一且 forward multiple 接近 real，说明基础结构仍能牵引 valid 选择，不能说 real path 已经干净通过。

| 变体 | valid Top20 | dev Top20 | forward Top20 | forward Top10 | forward IC | dev multiple | forward multiple | remove best 3 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base_liq_only` | `+0.9772%` | `-0.0640%` | `+0.3313%` | `+0.6441%` | `-0.0167` | `0.7896x` | `1.1151x` | `0.9069x` | valid 过强但 dev/IC 不支持 |
| `real_path` | `+0.9200%` | `+0.8576%` | `+0.5836%` | `+0.6205%` | `+0.0722` | `6.0383x` | `1.1544x` | `0.8734x` | 唯一 dev 五年全正，forward IC 强 |
| `no_path_stats` | `+0.8923%` | `+0.6352%` | `-0.3628%` | `-0.5653%` | `+0.0236` | `3.4627x` | `0.8961x` | `0.8142x` | 去路径后 forward 失败 |
| `last_day_only` | `+0.9039%` | `+0.5980%` | `-0.5241%` | `-0.9062%` | `+0.0259` | `2.0494x` | `0.8308x` | `0.7194x` | 只看当天不够 |
| `next_session_shift_label` | `+0.4005%` | `-0.3441%` | `-0.0088%` | `-0.1317%` | `+0.0099` | `0.2015x` | `0.8874x` | `0.7845x` | 错位标签失败 |
| `permuted_train_residual_label` | `+0.3603%` | `-0.4386%` | `-0.4677%` | `-0.2741%` | `+0.0129` | `0.1171x` | `0.9110x` | `0.7871x` | 打乱标签失败 |

forward random pool500 Top10 final multiple：p50 `1.0554x`、p90 `1.2517x`、p95 `1.3126x`、p99 `1.4241x`。`real_path` forward `1.1544x` 仍低于 p90/p95，不能晋级。

### 75.4 观察

第一，标签安慰剂没有通过。`permuted_train_residual_label` 和 `next_session_shift_label` 的 dev multiple 分别只有 `0.1171x`、`0.2015x`，forward Top20 也不成立。这与第 71 的 label-shift 安慰剂过强不同，说明第 74/75 这条个股路径 residual 不是明显的标签错位幻觉。

第二，路径信息有增量。`no_path_stats` 和 `last_day_only` 在 valid 看起来不差，但 forward Top20 分别为 `-0.3628%`、`-0.5241%`，组合 forward 也亏损。real path 的 forward IC `+0.0722` 和 forward multiple `1.1544x` 明显更好。

第三，`base_liq_only` 是主要警报。它 valid Top20 `+0.9772%` 排第一，说明 valid 年对基础结构很友好；但它 dev Top20 为负、dev multiple `0.7896x`、forward IC 为负。这意味着如果只按 valid Top20 选模型，会被基础结构误导。

第四，real path 的质量是“方向有效但不够强”。它是唯一 dev 五年全正的变体，dev multiple `6.0383x`，forward multiple `1.1544x`；但 forward remove-best-3 为 `0.8734x`，仍不稳。

第五，第 75 的结果不是彻底否定第 74，而是改变下一步方向：不能再用均值 Top20/valid score 做选择，必须直接优化右尾 Top10 和组合稳健性，并把 `base_liq_only` 作为必须压过的 hard baseline。

### 75.5 反事实分析

第一反事实：如果第 74 只是标签泄漏或样本记忆，permuted/shift 标签应接近 real。实际两者 dev 和 forward 都很弱，否定。

第二反事实：如果 60 日路径没有价值，`no_path_stats` 和 `last_day_only` 应接近 real。实际两者 forward 亏损，说明 60 日路径贡献存在。

第三反事实：如果 real path 已经足够干净，valid 上应压过 `base_liq_only`，forward 也应过 random p90/p95。实际 valid 第一是 `base_liq_only`，real forward 低于 random p90，否定。

第四反事实：如果基础结构是主要 alpha，`base_liq_only` 应该 dev/forward IC 也强。实际 dev multiple 不到 1、forward IC 为负，说明它只是 valid 年偏差或局部结构。

第五反事实：如果下一轮继续做均值回归 MLP 就能解决，Top10/forward remove-best 应已显著改善。实际 Top10 和 remove-best 仍弱，说明下一轮必须换目标函数和选择指标。

### 75.6 判定

`daily_path_memory_residual_placebo_too_strong`，但人工解释为：标签和无路径安慰剂被 real path 压过，60 日个股路径 residual 有继续研究价值；不过 `base_liq_only` valid 太强，当前选择口径不干净，不能晋级。

本轮没有可融入策略，也不进入 formal。下一轮不收束方向，但要改变优化目标。第 76 轮做 `daily_path_right_tail_pairwise_v1`：继续使用个股 60 日路径，但训练目标从均值 residual rank 改为 Top10/right-tail pairwise 或 listwise loss，评估时必须同时压过 `base_liq_only`、`real_path_mse`、random p90/p95，并要求 forward remove-best-3 为正。若右尾强化后仍无法改善 Top10 和 remove-best，则该方向阶段性收束。

## 76. daily_path_right_tail_pairwise_v1

### 76.1 假设

第 74-75 轮说明，个股 60 日路径 residual 有横截面预测力，但它更像 Top20/广谱预测器，Top10 和组合 remove-best 不够强。第 76 轮直接改变训练目标：不再只做均值 residual rank，而是比较 `pointwise_residual`、`binary_top_tail`、`top_minus_bottom` 三种右尾目标；同时用 `base_liq_only` 跑同样目标，作为 hard baseline。

如果问题只是 loss 不对，右尾目标应提升 forward Top10、组合 forward multiple 和 remove-best；如果右尾目标反而破坏 dev 或 forward，说明问题不是简单调 loss，而是模型需要识别“何时使用路径 alpha”的市场机会状态。

### 76.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_daily_path_right_tail_pairwise_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/daily_path_right_tail_pairwise_v1_summary.json
script sha256:6024acbb25144d2d383b8a7180bc786d63faf8342bb5f8c26e015a95d5c00cdb
summary sha256:9cfdb3d30c4edf2a2bf59a96d4caf0d53d50d6498bdd252c6b5fef65a957d1ad
```

样本同第 74/75：train 96677 行、valid 24362 行、dev 121039 行、forward 11914 行；session 数 train 194、valid 49、dev 243、forward 24。每个配置训练 24 epoch。输入分两类：`real_path` 和 `base_liq_only`。目标分三类：`pointwise_residual`、`binary_top_tail`、`top_minus_bottom`。

选择指标从第 74 的 Top20 均值改为右尾导向：`0.55 * valid Top10 + 0.35 * valid Top20 + 0.10 * valid IC`。forward 仍用组合 replay、random p90/p95、remove-best-3 做硬检验。

### 76.3 结果

结论为 `daily_path_right_tail_pairwise_not_enough`。右尾目标没有解决策略级问题。`real_path_binary_top_tail` 是 real path 中 forward multiple 最好的配置，但 dev 不全正、forward 低于 random p90/p95、remove-best-3 仍小于 1；`real_path_pointwise_residual` dev 仍好，但 forward 亏损。

| 配置 | valid Top10 | valid Top20 | dev Top20 | forward Top10 | forward Top20 | forward IC | dev multiple | forward multiple | remove best 3 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `real_path_pointwise_residual` | `+0.8837%` | `+0.9901%` | `+0.8656%` | `-0.1506%` | `+0.2314%` | `+0.0549` | `8.2761x` | `0.9526x` | `0.7751x` | dev 好，forward 亏损 |
| `base_liq_only_pointwise_residual` | `+1.1665%` | `+0.8075%` | `-0.0772%` | `+0.3594%` | `+0.1758%` | `-0.0171` | `0.4706x` | `1.0155x` | `0.8443x` | valid 仍被基础结构牵引 |
| `base_liq_only_binary_top_tail` | `+0.8075%` | `+0.8273%` | `-0.1020%` | `+0.2089%` | `+0.1138%` | `-0.0178` | `0.4167x` | `0.9721x` | `0.7879x` | 不成立 |
| `real_path_top_minus_bottom` | `+0.5273%` | `+0.4342%` | `+0.4177%` | `-0.8587%` | `-0.6477%` | `+0.0164` | `2.5707x` | `0.8380x` | `0.7994x` | 右尾目标伤害 forward |
| `base_liq_only_top_minus_bottom` | `+0.6787%` | `+0.5920%` | `+0.2194%` | `-0.0554%` | `-0.3288%` | `-0.0137` | `1.0865x` | `0.9283x` | `0.7922x` | 不成立 |
| `real_path_binary_top_tail` | `+0.8290%` | `+1.0767%` | `+0.0644%` | `+0.4117%` | `+0.4672%` | `+0.0350` | `0.7601x` | `1.0738x` | `0.8428x` | forward 略正但 dev 崩 |

forward random pool500 Top10 final multiple：p50 `1.0554x`、p90 `1.2517x`、p95 `1.3126x`、p99 `1.4241x`。第 76 最好的 real forward multiple `1.0738x` 只略高于 p50，明显低于 p90/p95。

### 76.4 观察

第一，右尾目标没有带来策略级提升。`binary_top_tail` 是唯一让 real path forward multiple 高于 1 的右尾目标，但 dev multiple 只有 `0.7601x`，且 remove-best-3 `0.8428x`，不能接受。

第二，`pointwise_residual` 再次暴露出 dev/forward 断裂。它 dev multiple `8.2761x`、dev 五年全正，但 forward multiple `0.9526x`，Top10 为负。这说明第 74 的 MSE/pointwise 方向在历史内能拟合路径残差，但不能稳定外推到 2026 的右尾组合。

第三，`base_liq_only` 继续是 valid 误导源。`base_liq_only_pointwise` valid Top10 `+1.1665%` 是全场最高，但 dev multiple 只有 `0.4706x`、forward IC 为负。valid 年对基础结构极友好，不能作为唯一选择器。

第四，Top10 没有被救起来。第 75 real path forward Top10 `+0.6205%`，第 76 最好的 real Top10 是 binary top-tail `+0.4117%`，反而下降。直接加右尾 loss 没解决“右尾抓手”问题。

第五，本方向的核心问题从“模型有没有预测力”转为“何时预测力能转成组合收益”。real path 在多轮里都有正 IC，但组合 forward 低于 random p90，说明模型在某些市场状态有效、某些状态失效，需要 session/market opportunity filter，而不是继续调 loss。

### 76.5 反事实分析

第一反事实：如果第 74-75 的不足只是 loss 不够右尾，`binary_top_tail` 或 `top_minus_bottom` 应提升 forward Top10 和 remove-best。实际没有，否定。

第二反事实：如果 `base_liq_only` 是真正强 alpha，它应该在 dev/forward 都强。实际 dev 崩、forward IC 为负，说明它只是 valid 选择噪声。

第三反事实：如果 pointwise residual 可直接策略化，forward 不应从第 74 的 `1.1461x` 掉到第 76 的 `0.9526x`。实际波动很大，说明模型训练/选择仍不稳。

第四反事实：如果右尾目标有效，Top10 应显著强于 Top20。实际 best real 的 Top10 `+0.4117%`、Top20 `+0.4672%`，没有右尾优势。

第五反事实：如果继续沿这个方向调 threshold/weight 就能达标，至少应出现一个配置同时满足 dev 全正、forward > random p90、remove-best-3 > 1。实际没有。

### 76.6 判定

`daily_path_right_tail_pairwise_not_enough`。

本轮没有可融入策略，也不进入 formal。第 74-76 合并判断：个股 60 日路径 residual 有可重复的横截面预测力，标签和无路径安慰剂没有完全解释它；但它还不能稳定转化为用户要求的强右尾组合收益。继续调 MLP loss 大概率收益有限。

下一轮不再调 loss，而是做 `daily_path_opportunity_filter_v1`：把第 74/75 的 real path 分数当作候选 alpha，按 session-level 市场状态、宽度、波动、成交、路径模型分散度、TopK score gap、过去 1-3 个 session 的路径模型 payoff，训练/扫描一个“何时使用路径模型”的机会过滤器。目标不是降低尾部权重，而是避开模型失效状态；若 opportunity filter 仍不能让 forward remove-best 转正并超过 random p90/p95，则日线个股路径 residual 分支阶段性收束。

## 77. daily_path_opportunity_filter_v1

### 77.1 假设

第 74-76 轮说明，个股 60 日路径 residual 有横截面预测力，但它不能稳定转化为强右尾组合收益。第 77 轮不再调 loss，而是把第 74 的 real path 分数当作候选 alpha，尝试寻找 session-level opportunity filter：哪些市场状态下应该交易路径模型，哪些状态下应该清仓到现金。

本轮特别加入两个约束：第一，active rate 不得低于 `50%`，避免靠极少交易美化结果；第二，inactive session 必须清仓到现金，不能继续持有上一期股票，避免隐性延长持仓造成偏差。

### 77.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_daily_path_opportunity_filter_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/daily_path_opportunity_filter_v1_summary.json
script sha256:99215b7bfc0fc9a897ca58c1fe2653f4b2f77ebc2e14ae28971cd7f6cee5a80d
summary sha256:38ab51e4034f0b4642fa024871edaf5293334c4718087abe96d5ed844184e78b
```

样本同第 74-76：train 96677 行、valid 24362 行、dev 121039 行、forward 11914 行；session 数 train 194、valid 49、dev 243、forward 24。先重训第 74 的 real path MLP，再构造 session 特征：`score_gap_10_30`、`score_dispersion`、`score_mean_top10`、`mkt_ret20_mean`、`mkt_range_mean`、`mkt_vwap_mean` 等。扫描 train 分位阈值 `0.20/0.35/0.50`，valid 选择，forward 只评估。

### 77.3 结果

结论为 `daily_path_opportunity_filter_not_enough`。valid 选择出来的是 `all_active`，也就是不过滤。所有分位过滤器都没有在 valid 选择和 forward 稳健性之间形成可迁移改善。

| gate | valid active | valid multiple | dev multiple | forward active | forward multiple | forward remove best 3 | avg position | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `all_active` | `100%` | `1.3107x` | `9.2820x` | `100%` | `1.1461x` | `0.8860x` | `9.96` | valid 选择的基准，无过滤 |
| `score_gap_10_30_high_q0.50` | `82%` | `1.3806x` | `4.1546x` | `92%` | `1.0554x` | `0.8137x` | `9.09` | valid 更好，forward 更弱 |
| `score_gap_10_30_high_q0.35` | `88%` | `1.3553x` | `4.1302x` | `96%` | `1.1091x` | `0.8561x` | `9.53` | 没改善 remove-best |
| `score_dispersion_high_q0.35` | `98%` | `1.3414x` | `3.7811x` | `96%` | `1.2438x` | `0.9604x` | `9.53` | forward 接近改善，但仍不达标 |
| `mkt_ret20_mean_high_q0.50` | `69%` | `1.2787x` | `4.0923x` | `71%` | `1.0212x` | `0.7926x` | `6.94` | 降低交易频率后更弱 |
| `mkt_ret20_mean_high_q0.20` | `86%` | `1.2535x` | `5.6274x` | `83%` | `1.0793x` | `0.8296x` | `8.23` | 不成立 |

baseline forward `all_active` 为 `1.1461x`，remove-best-3 为 `0.8860x`。最接近的过滤器是 `score_dispersion_high_q0.35`，forward `1.2438x`、remove-best-3 `0.9604x`，但仍低于 1，且 dev multiple 从 `9.2820x` 降到 `3.7811x`，没有形成鲁棒改善。

### 77.4 观察

第一，简单机会过滤没有学到稳定开关。valid 最高评分是 `all_active`，说明在现有特征下，过滤器没有稳定地识别“路径模型失效状态”。

第二，过滤通常损伤 dev。`score_gap_10_30` 和 `score_dispersion` 在 valid 上看起来更强，但 dev multiple 明显下降，说明它们更像 valid 年局部结构，不是可迁移市场状态。

第三，forward remove-best 没有转正。所有过滤器的 remove-best-3 都小于 1，核心风险没有解决。即使 `score_dispersion_high_q0.35` 的 forward multiple 到 `1.2438x`，也仍然依赖少数交易贡献。

第四，清仓到现金的严格口径很重要。若 inactive 时继续持有旧仓，过滤器可能通过隐性延长持仓美化结果；本轮没有这样做，因此结论更可信。

第五，第 74-77 的共同结论是：日线个股路径 residual 有预测力，但无法用简单 loss 或简单 session filter 变成用户要求的强策略。继续在同一 pool500 高流动性候选域里调参，收益上限大概率有限。

### 77.5 反事实分析

第一反事实：如果路径模型只在高置信度 session 有效，`score_gap_10_30` 或 `score_dispersion` 过滤应提高 forward remove-best。实际没有，否定。

第二反事实：如果市场宽度/趋势状态是关键，`mkt_ret20_mean` 或 `mkt_range_mean` gate 应明显改善 forward。实际不改善，否定当前状态刻画。

第三反事实：如果第 74 的 forward 不稳只是少数坏 session 造成，过滤应能在 active rate >50% 下去掉坏 session。实际 remove-best 仍小于 1，说明坏状态不是简单阈值可分。

第四反事实：如果继续调机会过滤就能接近目标，至少应出现一个 gate 同时保持 dev 全正、forward > baseline、remove-best-3 > 1。实际没有。

第五反事实：如果 pool500 高流动性候选域足够承载用户的几十倍目标，第 74-77 至少应能看到明显的 forward 组合强度。实际多轮最高仍远低于目标，提示应改变候选域或收益来源。

### 77.6 判定

`daily_path_opportunity_filter_not_enough`。

本轮没有可融入策略，也不进入 formal。第 74-77 的日线个股路径 residual 分支阶段性收束：它提供了可解释的预测力，但不是足够强的右尾策略方向。

下一轮切换更本质的问题：候选域。此前大量实验集中在 pool500 高流动性股票，这可能天然限制五年几十倍目标。第 78 轮做 `liquidity_tier_daily_path_residual_v1`：同样使用日线个股路径 residual，但按成交额/流动性层级扫描 pool500、pool1000、pool2000、mid-liquidity、lower-liquidity 候选域，验证 alpha 是否在较低流动性但仍可交易的股票层中显著增强。若收益只来自极低流动性或不可交易样本，则否定；若中低流动性层出现更强且年度稳定的收益，再回到模型优化。

## 78. liquidity_tier_daily_path_residual_v1

### 78.1 假设

第 74-77 轮都在高流动性 pool500 内打磨个股路径 residual，但收益强度始终达不到用户要求。第 78 轮检查一个更基础的可能：目标级别的五年几十倍收益也许不在最流动 500 只股票里，而在中低流动性但仍可交易的候选域中。

本轮保持第 74 的日线个股路径 residual 框架不变，只改变候选域：按 20 日成交额排序切成 `pool_top500`、`pool_501_1000`、`pool_1001_2000`、`pool_2001_3500`。每层独立训练小 MLP、独立 null、独立 replay，并检查 forward、remove-best、随机对照和可交易性。

### 78.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_liquidity_tier_daily_path_residual_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/liquidity_tier_daily_path_residual_v1_summary.json
script sha256:80026582e1fbe4e01a180a9b8509007e85d94f10ecbee4c5b7beae91beca8a4e
summary sha256:54275beeb2d6f53aae01b7a40da0c1a9b2aea8b05419da6bbb1443c944c27f3d
```

每层使用同样 split：train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`。每层训练 24 epoch。组合 Top10，5 日换仓，仍使用涨跌停/停牌开盘过滤和成本口径。

### 78.3 结果

结论为 `liquidity_tier_daily_path_residual_not_enough`。离开 pool500 没有改善 forward；中低流动性层虽然在 dev 里可能更高，但 forward 全部亏损或低于随机。尤其 `pool_2001_3500` dev 到 `18.3342x`、五年全正，但 forward `0.9127x`，是典型历史阶段偏差。

| tier | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | forward IC | remove best 3 | random p90/p95 | avg position | 判读 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `pool_top500` | `6.0383x` | 是 | `1.1544x` | `+0.6205%` | `+0.5836%` | `+0.0722` | `0.8734x` | `1.2685x / 1.3144x` | `9.96` | 仍是最稳，但不达标 |
| `pool_501_1000` | `1.6826x` | 否 | `0.7846x` | `-0.9312%` | `-0.4458%` | `+0.0352` | `0.7162x` | `1.0633x / 1.1012x` | `10.00` | forward 明显失败 |
| `pool_1001_2000` | `2.7199x` | 否 | `0.9261x` | `-0.1306%` | `-0.1698%` | `+0.0166` | `0.8261x` | `0.9712x / 1.0330x` | `10.00` | forward 亏损 |
| `pool_2001_3500` | `18.3342x` | 是 | `0.9127x` | `-0.2934%` | `-0.1912%` | `+0.0057` | `0.8145x` | `0.9466x / 0.9751x` | `9.87` | dev 很强但 forward 反转 |

各层 forward 缺失/可交易情况可接受：`pool_top500` forward candidate rows 11914、future return missing 86；`pool_501_1000` rows 11949、missing 51；`pool_1001_2000` rows 23917、missing 83；`pool_2001_3500` rows 28209、missing 61。因此失败不能简单归因于缺失数据或持仓数不足。

### 78.4 观察

第一，pool500 仍是本框架下最稳的层。它 dev 五年全正、forward 为正、IC 最高，但 forward 仍低于 random p90/p95，remove-best-3 小于 1。

第二，中低流动性层没有带来 forward 金矿。`pool_501_1000`、`pool_1001_2000`、`pool_2001_3500` 的 forward multiple 分别为 `0.7846x`、`0.9261x`、`0.9127x`，均为负收益。

第三，`pool_2001_3500` 是最危险的过拟合诱惑。dev `18.3342x`、五年全正、remove-best-3 `15.2724x`，看起来接近用户的收益强度；但 forward 直接亏损，Top10/Top20 均为负，说明中低流动性历史收益结构在 2026 不迁移。

第四，base replay 在各层都很差，说明模型确实在做重排，但重排没有跨 regime 稳定。低流动性层的历史收益可能来自特定年份的小票风格，而不是可迁移 deep learning alpha。

第五，候选域扩展没有解决核心问题。第 74-78 证明日线个股路径 residual 是一个可预测但收益强度有限的因子，而不是用户要求的高倍策略方向。

### 78.5 反事实分析

第一反事实：如果 pool500 限制了高倍收益，中低流动性层应在 forward 明显优于 pool500。实际全部更弱，否定。

第二反事实：如果 `pool_2001_3500` 的 dev 高收益是真 alpha，forward 不应反转为亏损。实际 forward `0.9127x`，否定。

第三反事实：如果低流动性层失败只是可交易性不足，forward 应出现大量缺失或持仓不足。实际 avg position 仍接近 10，缺失很少，否定。

第四反事实：如果继续在路径 residual 框架内换层级能找到金矿，至少应有一个层同时满足 dev 全正、forward > random p90、remove-best-3 > 1。实际没有。

第五反事实：如果五年几十倍目标可以靠日线 ML 的细调达到，第 74-78 的连续改动应逐步接近目标。实际 improvement 在 forward 上没有积累，说明需要换收益来源而非继续调参。

### 78.6 判定

`liquidity_tier_daily_path_residual_not_enough`。

本轮没有可融入策略，也不进入 formal。第 74-78 日线个股路径 residual 分支阶段性收束：它能产生稳定一些的预测力，但无法达到用户要求的高倍收益/右尾稳健性，也无法通过候选域扩展解决。

下一轮需要大胆换方向：从“连续路径预测”切到“事件/状态驱动的右尾机会”。第 79 轮建议做 `daily_extreme_event_rebound_continuation_v1`：只用日线/QMT 可得数据，挖掘可重复的强事件状态，比如放量突破后缩量承接、长上影失败后的二次反包、连续涨停/近涨停后的断板承接、跌停/大跌后的流动性恢复、行业/概念同步扩散的事件。先做事件定义和 forward-safe 统计，不急着上深度模型；如果事件本身没有右尾土壤，再换方向。

## 79. daily_extreme_event_rebound_continuation_v1

### 79.1 假设

第 74-78 轮说明日线连续路径 residual 有预测力，但无法转化为用户要求的高倍、稳健、平均持仓大于 5 的策略。第 79 轮转向“事件/状态驱动的右尾土壤”：如果市场短期可预测性主要来自强状态，而不是全市场连续排序，那么应先看到某些日线事件本身有稳定右尾，再考虑 deep learning 对事件内部做排序。

本轮不先训练模型，而是定义可由日线/QMT 数据直接得到的事件状态：放量突破后缩量承接、长上影失败后二次反包、大跌后放量修复、连续上涨后缩量休整、强趋势近高位回撤吸收，以及事件 composite。每个候选先用 boolean 事件筛选，再在事件内部按可解释分数排序。严格使用 t 日及之前特征，收益为 t+1 open 到 t+interval+1 open。

### 79.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_daily_extreme_event_rebound_continuation_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/daily_extreme_event_rebound_continuation_v1_summary.json
script sha256:4d7679b7016a609cdba7f82db414a4fe30577dcb6f798d8e79c18c8430e90d8d
summary sha256:c1b91aa6ad4f2491b8d624ca2b26cc531209145fa707664419c93152f9622600
```

split 为 train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`。扫描 `pool500/pool1000` 与 `3/5` 日持仓。组合 replay 使用 Top10、开盘成交、涨跌停/停牌过滤和成本口径，并对 forward 做 matched active-count random 对照。valid 只用于选择，不用 forward 选策略。

### 79.3 结果

结论为 `daily_extreme_event_rebound_continuation_not_enough`。valid 选择的是 `pool500_interval3 / volume_dry_after_breakout`，但 dev 只有 `1.0188x` 且年度不全正，forward 亏损到 `0.6815x`，平均持仓也只有 `3.80`，低于用户约束。

| candidate | key | valid multiple | dev multiple | dev all positive | forward multiple | forward remove best 3 | random p95 | avg position | 判读 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `volume_dry_after_breakout` | `pool500_interval3` | `1.6443x` | `1.0188x` | 否 | `0.6815x` | `0.4956x` | `1.4570x` | `3.80` | valid 诱惑，forward 失败 |
| `absorption_near_high_pullback` | `pool1000_interval5` | `0.8144x` | `0.3180x` | 否 | `1.4811x` | `1.0642x` | `1.2415x` | `8.18` | forward 看似好，但 dev 极差 |
| `absorption_near_high_pullback` | `pool500_interval5` | `1.2744x` | `0.3653x` | 否 | `1.4176x` | `0.8506x` | `1.4453x` | `6.44` | 没打过随机 p95，dev 极差 |
| `failed_upper_shadow_reclaim` | `pool500_interval5` | `0.7250x` | `0.1347x` | 否 | `1.3190x` | `0.8569x` | `1.5437x` | `5.45` | forward 低于随机 |
| `failed_upper_shadow_reclaim` | `pool1000_interval3` | `1.1333x` | `0.1510x` | 否 | `1.1408x` | `0.8662x` | `1.2011x` | `5.54` | 不稳 |
| `multi_day_acceleration_rest` | `pool1000_interval3` | `0.8031x` | `0.0444x` | 否 | `1.0418x` | `0.8223x` | `1.0983x` | `9.98` | dev 崩溃 |

从 dev 角度看，最高也只有 `volume_dry_after_breakout/pool500_interval3` 的 `1.0188x`，且最差年度 `-40.17%`。从 forward 角度看，最高是 `absorption_near_high_pullback/pool1000_interval5` 的 `1.4811x`，remove-best-3 `1.0642x`、平均持仓 `8.18`，但 valid `0.8144x`、dev `0.3180x`，因此不能作为候选，只能作为“2026 局部状态碰巧有效”的线索。

样本覆盖不是主要问题：`pool500_interval5` forward session 24、missing future 86；`pool500_interval3` session 40、missing 170；`pool1000_interval5` session 24、missing 137；`pool1000_interval3` session 40、missing 263。失败更多来自事件定义本身缺乏跨年稳定收益，而不是数据缺口。

### 79.4 观察

第一，泛化事件模板没有右尾土壤。多数候选在 train/dev 的 Top20 均值接近 0 或为负，说明这些“看起来像交易员语言”的状态，并没有天然稳定的下一周收益优势。

第二，valid 选择不可靠。`volume_dry_after_breakout` 在 valid replay `1.6443x`，但 train Top20 `-0.005bp`、dev `1.0188x`、forward `0.6815x`。这说明 2025 的缩量突破承接更像局部年份特征，不是可迁移事件。

第三，forward 的亮点不能倒推为 alpha。`absorption_near_high_pullback/pool1000_interval5` forward `1.4811x` 且 remove-best-3 `1.0642x`，但 dev `0.3180x`、valid `0.8144x`，如果拿它继续训练，就是用 forward 选方向。

第四，事件稀疏度不满足用户约束。valid 选中的 `volume_dry_after_breakout` forward 平均持仓只有 `3.80`，即使收益为正也不满足平均持仓大于 5。很多 panic/reclaim 类事件 active rate 太低，天然难以承担稳定组合。

第五，第 79 和第 74-78 的差异很重要：连续路径 residual 是“有预测力但不够强”，泛化事件模板是“连土壤都不稳定”。下一步不应继续堆泛化事件，而要更贴近 A 股制度结构。

### 79.5 反事实分析

第一反事实：如果强事件本身是右尾来源，则至少应有一个候选在 train/valid/dev/forward 的 Top20 或 replay 上同向为正。实际没有，否定泛化事件模板。

第二反事实：如果 `volume_dry_after_breakout` 是缩量承接 alpha，train 和 dev 不应接近 0 或年度大亏。实际 train Top20 约 0、dev 最差年度 `-40.17%`，否定。

第三反事实：如果 `absorption_near_high_pullback` 是可迁移 alpha，dev 不应只有 `0.3180x`。实际 forward 好、历史差，说明它更可能是 2026 局部小样本机会，不能作为策略基础。

第四反事实：如果失败来自 pool500 太窄，pool1000 应显著改善。实际 pool1000 只产生 forward 局部亮点，但 dev 更差，否定。

第五反事实：如果事件稀疏能靠降低持仓数解决，那会违背用户平均持仓大于 5 的约束。本轮 valid 最优 forward avg position `3.80`，说明不能沿这个方向硬卡。

### 79.6 判定

`daily_extreme_event_rebound_continuation_not_enough`。

本轮没有可融入策略，也不进入 formal。泛化日线事件模板暂时收束。下一轮转向更 A 股制度化的状态：`daily_limit_up_break_board_state_scan_v1`。核心不是“泛化突破/回撤”，而是用日线近似识别涨停、连板、断板、炸板后修复、涨停后缩量承接等制度事件，检验这些状态是否提供更厚的右尾土壤；仍保持 train/valid/forward 纪律，不用 2026 反选方向。

## 80. daily_limit_up_break_board_state_scan_v1

### 80.1 假设

第 79 轮的泛化事件模板没有稳定右尾，但 A 股短周期强收益经常围绕涨停、连板、断板、炸板和涨停后承接展开。第 80 轮把事件定义从“通用突破/回撤”进一步收窄到交易制度相关状态，用日线近似识别涨停/炸板/连板路径，检验制度事件是否比泛化事件有更厚的右尾土壤。

本轮仍不训练深度模型。原因是：如果制度事件本身历史不可迁移，直接上模型只会把稀疏状态过拟合得更漂亮。先验证土壤，再决定是否做小模型。

### 80.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_daily_limit_up_break_board_state_scan_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/daily_limit_up_break_board_state_scan_v1_summary.json
script sha256:5eabef5fe5ddd11e5abb6850f074e15296030d7ccadcc70b64a6ed41cd8ca8ba
summary sha256:60a731f87b8a12e525369d93faacfa3c4859d5ba0c532334cfd4b25e613db0f5
```

复用第 79 的严格 replay 框架：train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`；扫描 `pool500/pool1000` 和 `3/5` 日持仓；Top10 replay、开盘成交、涨跌停/停牌过滤、成本和 matched active-count random 对照。涨停近似阈值为日涨幅或日内高点相对昨收 `>= 9.2%`。

候选包括：`first_limit_follow_through`、`two_board_continuation`、`break_board_reclaim`、`failed_board_low_absorb`、`post_limit_volume_dry_hold`、`limit_pullback_relaunch` 和 `limit_state_composite`。

### 80.3 结果

结论为 `daily_limit_up_break_board_state_not_enough`。制度事件确实出现了比第 79 更强的 2026 forward 亮点，但历史 dev 是灾难级反证，因此不能作为候选策略。

| candidate | key | valid multiple | dev multiple | dev worst year | forward multiple | remove best 3 | random p95 | avg position | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `post_limit_volume_dry_hold` | `pool1000_interval5` | `1.4120x` | `0.1738x` | `-59.27%` | `2.3346x` | `1.2440x` | `1.3506x` | `5.15` | forward 很强，但 dev 极差 |
| `post_limit_volume_dry_hold` | `pool500_interval5` | `1.9870x` | `0.0844x` | `-66.36%` | `2.3170x` | `0.9994x` | `1.5581x` | `3.46` | valid/forward 亮，但持仓不足且 dev 崩 |
| `limit_pullback_relaunch` | `pool500_interval3` | `0.7809x` | `0.0187x` | `-78.39%` | `1.6234x` | `0.9594x` | `1.4738x` | `3.94` | forward 亮点，不满足约束 |
| `limit_pullback_relaunch` | `pool500_interval5` | `1.8325x` | `0.0543x` | `-68.32%` | `1.1838x` | `0.6948x` | `1.4572x` | `4.90` | 历史不可用 |
| `failed_board_low_absorb` | `pool1000_interval3` | `0.4885x` | `0.0487x` | `-73.09%` | `1.1801x` | `0.9523x` | `1.1503x` | `9.83` | 持仓足但历史崩 |
| `limit_state_composite` | `pool500_interval3` | `0.6677x` | `0.0161x` | `-77.72%` | `0.9568x` | `0.7669x` | `1.1532x` | `9.98` | composite 无法稳定化 |

valid 选择项是 `post_limit_volume_dry_hold/pool500_interval5`：valid `1.9870x`、forward `2.3170x`，但 dev 仅 `0.0844x`，最差年度 `-66.36%`，forward 平均持仓 `3.46`，remove-best-3 也只有 `0.9994x`。按用户约束和防过拟合纪律必须否定。

最接近“看起来诱人”的是 `post_limit_volume_dry_hold/pool1000_interval5`：forward `2.3346x`，random p95 `1.3506x`，remove-best-3 `1.2440x`，平均持仓 `5.15`。但 valid 只是 `1.4120x`，dev `0.1738x`，最差年度 `-59.27%`，train Top20 还是负的 `-0.1769%`。这不是可迁移 alpha，而是 2026 局部 regime 的强烈反应。

### 80.4 观察

第一，制度事件比泛化事件更有“右尾味道”。第 79 forward 最高约 `1.48x`，第 80 的 `post_limit_volume_dry_hold` forward 到 `2.33x`，且 pool1000/5 日版本打过 random p95、remove-best-3 大于 1、平均持仓略高于 5。

第二，历史表现极差，不能忽视。所有 forward 亮点的 dev multiple 都小于 `0.25x`，这不是小幅不稳，而是长期持有会归零式损伤。任何用它进入模型训练的做法，都会是在用 2026 反向选题。

第三，涨停后缩量承接可能是 2026 的短期 regime，而不是长期 alpha。`post_limit_volume_dry_hold` 的 train 平均事件数很低：pool500/5 日只有 `1.56`，pool1000/5 日 `2.54`；稀疏状态放大了年份差异，也让 replay 更容易被少数阶段主导。

第四，composite 没有修复稀疏事件的不稳。`limit_state_composite` 持仓数接近 10，但 dev `0.0161x`、forward `0.9568x`，说明把多种涨停状态混在一起只会引入更多噪声。

第五，第 80 轮提供的线索不是“直接做涨停策略”，而是“2026 年涨停后承接状态可能被市场结构强化”。下一步要诊断这种强化来自市场阶段、事件拥挤度、流动性层、还是日线涨停近似误差。

### 80.5 反事实分析

第一反事实：如果涨停后缩量承接是可迁移 alpha，dev 不应只有 `0.1738x/0.0844x`。实际历史极差，否定直接策略化。

第二反事实：如果第 80 的 forward 强只是随机运气，remove-best-3 应该很差。pool1000/5 日 remove-best-3 为 `1.2440x`，说明 2026 的局部状态确实有一定分散性，不能简单当成单笔偶然。

第三反事实：如果扩大到 pool1000 能解决持仓不足并保持历史收益，dev 应明显改善。实际 pool1000/5 日 avg position 到 `5.15`，但 dev 仍 `0.1738x`，否定。

第四反事实：如果把涨停事件 composite 化能提高鲁棒性，composite 应优于单事件。实际 composite 历史和 forward 都弱，否定简单混合。

第五反事实：如果五年几十倍目标来自制度事件土壤，那么不应在 2021-2025 出现近乎全灭的 replay。实际 dev 近乎全灭，说明当前日线近似口径仍不是答案。

### 80.6 判定

`daily_limit_up_break_board_state_not_enough`。

本轮没有可融入策略，也不进入 formal。但它暴露了一个比第 79 更有价值的线索：`post_limit_volume_dry_hold` 在 2026 有明显右尾，且 pool1000/5 日版本满足 forward random/remove-best/持仓三个局部条件，只是历史完全不迁移。第 81 轮应做 `limit_state_regime_migration_diagnostic_v1`：不训练模型，专门拆解这个状态在不同年份、市场强弱、事件密度、涨停拥挤度和流动性层上的表现，判断它是可被 regime router 捕捉，还是纯 2026 偶然/定义误差。

## 81. limit_state_regime_migration_diagnostic_v1

### 81.1 假设

第 80 轮最有信息量的不是“涨停后缩量承接可以直接交易”，而是它在 2026 forward 明显变强，同时在 2021-2025 历史中接近全灭。第 81 轮专门诊断这个迁移失败：如果 2026 的强表现来自可识别的 market regime，那么用 train 分位阈值构造的简单 regime gate 应该能改善 valid，并且至少把 dev 从灾难状态里显著救回来。

本轮固定 `post_limit_volume_dry_hold`，只看第 80 中最有信息量的 `pool500_interval5` 和 `pool1000_interval5` 两个 cell。不再扫描事件本身，避免继续用 forward 反选方向。gate 特征包括市场 5/20/60 日平均收益、上涨比例、涨停密度、炸板密度、近 5 日涨停密度、事件数量、事件 score 均值和分散度；阈值全部来自 train 分位数，valid 选择，forward 只验证。

### 81.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_limit_state_regime_migration_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/limit_state_regime_migration_diagnostic_v1_summary.json
script sha256:42623604049ddda5d0748a9b3f2efcc3d73dad5f2fc9f94da5d95f6897dfceea
summary sha256:08eca4d5463442d013fff037252270d4f69fdaae1f634e2d2003ddac79c250d7
```

重要说明：本轮 `selected_forward_random` 的采样池在稀疏事件下会退化为事件已选列表，导致 random p95 可能等于原始 replay，因此 random 对照只作为弱证据。本轮主要证据是 train/valid/dev/forward 的迁移结构、年度 top10 均值、active rate 和事件数量。

### 81.3 结果

结论为 `limit_state_regime_migration_no_stable_router`。regime gate 能把 valid 做得更漂亮，但 train 和 dev 仍无法修复。也就是说，2026 的强表现不能被一个从 train/valid 学到的简单 regime router 稳定解释。

| cell | selected gate | train multiple | valid multiple | dev multiple | forward multiple | forward remove best 3 | forward avg position | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pool500_interval5` | `mkt_ret60_mean_high_q0.65` | `0.6976x` | `2.9215x` | `0.6308x` | `2.3170x` | `0.9994x` | `3.46` | gate 改善 dev 但仍不合格，持仓不足 |
| `pool1000_interval5` | `event_score_mean_high_q0.65` | `0.1436x` | `2.6686x` | `0.1157x` | `1.8249x` | `0.9470x` | `4.90` | valid 更好但历史更差 |

baseline 也显示同样问题：`pool1000_interval5` baseline train `0.2717x`、valid `1.4120x`、dev `0.1738x`、forward `2.3346x`；`pool500_interval5` baseline train `0.1855x`、valid `1.9870x`、dev `0.0844x`、forward `2.3170x`。这是非常强的 regime split，而不是小幅参数不稳。

年度层面更清楚：`pool1000_interval5` 的 dev 年度 top10 均值为 2021 `+0.741%`、2022 `-1.170%`、2023 `-0.371%`、2024 `+0.081%`、2025 `+0.063%`；`pool500_interval5` 为 2021 `+0.817%`、2022 `-1.257%`、2023 `-0.697%`、2024 `+0.105%`、2025 `-0.393%`。2026 forward top10 均值约 `+4%`，强度与历史年份不在一个量级。

### 81.4 观察

第一，2026 的强度不是简单市场强弱能解释。`pool500_interval5` 选择了 `mkt_ret60_mean_high_q0.65`，valid 到 `2.9215x`，但 train 仍低于 1、dev 只有 `0.6308x`。如果 60 日市场强度是真 router，历史强势区间不应仍大幅亏损。

第二，事件 score 本身不是稳定置信度。`pool1000_interval5` 选择 `event_score_mean_high_q0.65` 后，valid 到 `2.6686x`，但 train `0.1436x`、dev `0.1157x`，说明这个 score 均值在 2025/2026 可能对应更好的承接结构，在 2021-2024 反而可能是陷阱。

第三，事件数量明显 regime 化。`pool1000_interval5` baseline forward 平均事件数 `5.42`，历史 dev 平均 `2.61`；`pool500_interval5` forward `3.54`，历史 dev `1.66`。2026 不只是事件收益变强，事件出现密度也更高。

第四，2022 是关键反证年份。两个 cell 在 2022 的 top10 均值都约 `-1.2%`，说明涨停后承接状态在弱/下行市场里可能是亏钱状态，而不是无效状态。

第五，简单 gate 不足以做模型。哪怕 gate 把 pool500 dev 从 `0.0844x` 提到 `0.6308x`，离用户要求仍差几个数量级。继续在同一日线涨停近似事件上调阈值，只会更接近过拟合。

### 81.5 反事实分析

第一反事实：如果 2026 强表现是可识别 regime，train 分位 gate 至少应在 dev 上把策略恢复到正收益。实际最好的 dev 仍只有 `0.6308x`，否定。

第二反事实：如果事件 score 是有效置信度，`event_score_mean_high` 应在 train/dev 都提升。实际 train/dev 更差，否定。

第三反事实：如果问题只是事件太少，扩大到 pool1000 应该显著修复。实际 pool1000 持仓略改善，但 dev 仍 `0.1157x`，否定。

第四反事实：如果 2022 是单一年份拖累，remove 或 gate 应明显恢复 dev。实际 2023 也弱，且 2025 并不强，说明不是单一年份异常。

第五反事实：如果这一方向值得进入 deep learning，应该先看到一个可学习的状态分界。当前 gate 没有稳定分界，因此暂不训练模型。

### 81.6 判定

`limit_state_regime_migration_no_stable_router`。

本轮没有可融入策略，也不进入 formal。涨停/断板/承接方向阶段性收束：它在 2026 有明显右尾，但历史迁移性太差，简单 regime router 无法解释。下一轮需要大胆换收益来源，不再围绕日线个股事件模板调阈值。第 82 轮转向“横截面关系/扩散”而非单股票状态：做 `cross_sectional_lead_lag_diffusion_probe_v1`，用过去一段时间的股票间相似度/共同运动构造轻量 lead-lag diffusion 特征，检验强势簇内的滞后补涨或领涨扩散是否比单股路径、单股事件更稳定。这更贴近“市场是动态、趋势性、横截面结构可预测”的假设，也更接近后续 unified market world model 可学习的对象。

## 82. cross_sectional_lead_lag_diffusion_probe_v1

### 82.1 假设

第 60/61 轮已经验证过静态行业/概念扩散：有边际信息，但完整重排会削弱 forward 右尾。第 82 轮改成不依赖外部行业映射的动态关系：每个 session 内用过去 20/60 日收益相关性寻找相似邻居，测试“邻居已经上涨、自己短期滞后”的 lead-lag diffusion 是否能提供更稳定的横截面 alpha。

本轮仍先做土壤探针，不训练模型。原因是如果动态邻居的手工扩散分数都没有稳定正向，再直接上 deep learning 很容易只是把相关矩阵噪声拟合进去。

### 82.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_cross_sectional_lead_lag_diffusion_probe_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/cross_sectional_lead_lag_diffusion_probe_v1_summary.json
script sha256:223b3ef8d33a19547e0d090c7f7d700d2b1f90752eed94fa356f5e73025eb2ef
summary sha256:86fc4b828e4befd06db8c4f576f8b5f5c71babbd5ade489cb7cdc079c2ff5d25
```

split 为 train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`。使用 pool500、5 日持仓、Top10 replay；邻居数扫描 `10/20/40`。每个 session 用历史收益相关性构造动态邻居，候选包括邻居 20 日领先而自身 5 日滞后、邻居 5 日加速自身休整、簇确认领涨、扩散压力 composite、反拥挤簇滞后。

### 82.3 结果

结论为 `cross_sectional_lead_lag_diffusion_not_enough`。动态邻居扩散有一点 forward 弹性，但没有通过 dev 稳定性、remove-best 和随机对照。

| candidate | neighbor k | valid multiple | dev multiple | dev worst year | forward multiple | remove best 3 | random p95 | avg position | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `neighbor20_lead_self5_lag` | 10 | `1.2831x` | `0.5115x` | `-41.90%` | `1.2400x` | `0.9750x` | `1.3111x` | `10.00` | valid 第一，forward 未过随机 |
| `neighbor20_lead_self5_lag` | 20 | `1.1775x` | `0.6698x` | `-38.72%` | `1.2377x` | `0.9804x` | `1.1959x` | `10.00` | 唯一过 random p95，但 dev/robust 不够 |
| `neighbor20_lead_self5_lag` | 40 | `1.2295x` | `0.6204x` | `-44.07%` | `1.1541x` | `0.9096x` | `1.2427x` | `10.00` | 不成立 |
| `diffusion_pressure_composite` | 40 | `1.2334x` | `1.0292x` | `-47.91%` | `1.1380x` | `0.7896x` | `1.2427x` | `10.00` | dev 总体接近 1，但年度极差 |
| `anti_crowded_cluster_lag` | 20 | `1.2090x` | `0.6256x` | `-33.83%` | `0.9158x` | `0.8058x` | `1.1959x` | `10.00` | forward 失败 |

valid 选择项为 `neighbor20_lead_self5_lag/neighbor10`：valid `1.2831x`，forward `1.2400x`，但 dev `0.5115x`、remove-best-3 `0.9750x`、random p95 `1.3111x`。不满足策略条件。

### 82.4 观察

第一，动态邻居确实比完全随机更接近右尾，但强度太薄。最好的 forward 在 `1.24x` 左右，和用户要求的五年几十倍不在一个数量级。

第二，lead-lag 的最佳形态很一致：三个邻居规模里 forward 最好的都是 `neighbor20_lead_self5_lag`，说明“相似邻居中期强、自己短期滞后”不是纯噪声。但它没有变成稳定策略。

第三，dev 年度稳定性仍是硬伤。最好的 dev multiple 也只有 `1.0292x`，且最差年度 `-47.91%`；valid 好并不能迁移到五年历史。

第四，remove-best 没过。即使 `neighbor20_lead_self5_lag/neighbor20` forward 打过 random p95，remove-best-3 仍只有 `0.9804x`，说明收益仍依赖少数交易段。

第五，手工扩散分数的表达力有限。动态关系方向有一点信号，但线性分数无法在低信噪比场景里稳定分离正负样本；这直接支持下一轮改用正负样本对比学习。

### 82.5 反事实分析

第一反事实：如果动态邻居扩散是强 alpha，valid 选择项应在 dev 和 forward 同时稳定。实际 dev `0.5115x`，否定。

第二反事实：如果扩大邻居数能降低噪声，neighbor40 应显著更稳。实际 forward 下降，否定简单平滑。

第三反事实：如果收益来自分散的簇扩散，remove-best-3 应大于 1。实际所有主要候选都小于 1，否定。

第四反事实：如果静态行业/概念失败只是因为分组过粗，动态相关邻居应明显超越随机。实际只有一格略过 random p95，证据很弱。

第五反事实：如果继续手工打分能接近目标，至少应出现 dev 五年全正或 forward 显著厚尾。实际都没有，应该切换 loss 形式而不是继续调分数。

### 82.6 判定

`cross_sectional_lead_lag_diffusion_not_enough`。

本轮没有可融入策略，也不进入 formal。第 82 轮给出一个重要方向修正：横截面关系/扩散有微弱信号，但手工线性分数和回归式目标不够适合金融低信噪比。结合用户补充的判断，第 83 轮改为 `contrastive_pairwise_diffusion_ranker_v1`：用同 session 内未来 5 日右尾股票作为正样本、左尾股票作为负样本，训练小模型学习正负样本相对胜负，loss 优先采用 pairwise logistic / InfoNCE 式对比学习，而不是收益回归。目标是验证对比学习能否在低信噪比下更稳地提取 alpha 边界。

## 83. contrastive_pairwise_diffusion_ranker_v1

### 83.1 假设

用户指出金融领域信噪比很低，loss 应尽可能采用对比学习/正负样本。第 83 轮接受这个方向修正：不再拟合未来收益值，也不只做 top-tail 加权分类，而是在同一 session 内把未来 5 日收益 top quantile 股票作为正样本、bottom quantile 股票作为负样本，训练单股 scorer，使 `score(pos) > score(neg)`。

本轮使用第 74 的日线 60 日路径压缩特征和 replay 口径，只替换训练目标为 pairwise logistic contrastive loss：`softplus(-(s_pos - s_neg) / temperature)`。对照 `real_path` 与 `base_liq_only` 两种输入，并扫描 temperature `0.07/0.15/0.30`。评估加入多尺度指标：Top10/Top20、rank IC、同 session 正负 pair AUC、pair margin、组合 replay、remove-best 和随机对照。

### 83.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_contrastive_pairwise_diffusion_ranker_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/contrastive_pairwise_diffusion_ranker_v1_summary.json
script sha256:b6f7837e2b884938632a7327b43cf960f3337ec3371b53c5daa8b388750046ea
summary sha256:50fd232ffd68db022f0bbc9fc3f808787bcf2e4636d14625d3dcdd4999c6e5e3
```

split：train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`。样本数 train `96677`、valid `24362`、dev `121039`、forward `11914`；session 数 train `194`、valid `49`、dev `243`、forward `24`。pair 只在同 session 内构造，避免跨日标签混合。

### 83.3 结果

结论为 `contrastive_pairwise_diffusion_ranker_not_enough`。对比学习明显改善了预测指标，但未改善到可交易策略。尤其 `real_path/t=0.15` 在 dev Top20 五年全正、dev multiple `4.8957x`，说明正负样本 loss 确实比普通回归更能提取历史边界；但 forward replay 只有 `0.9485x`，仍失败。

| config | valid pair AUC | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | forward pair AUC | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `real_path_pairwise_logistic_t0.30` | `0.5926` | `1.9059x` | 否 | `1.0550x` | `+0.2891%` | `-0.1336%` | `0.5377` | `0.9718x` | valid 选择，forward 仅小正 |
| `real_path_pairwise_logistic_t0.15` | `0.5874` | `4.8957x` | 是 | `0.9485x` | `-0.1208%` | `-0.0688%` | `0.5611` | `0.8935x` | 历史最强，forward 交易失败 |
| `real_path_pairwise_logistic_t0.07` | `0.5856` | `2.1748x` | 否 | `1.0334x` | `+0.2546%` | `-0.3097%` | `0.5365` | `0.9666x` | Top10 有点信号，Top20 负 |
| `base_liq_only_pairwise_logistic_t0.15` | `0.5709` | `1.2643x` | 否 | `0.8687x` | `-0.4561%` | `-0.1396%` | `0.4961` | `0.7593x` | 证明路径输入有增量 |
| `base_liq_only_pairwise_logistic_t0.30` | `0.5707` | `0.9344x` | 否 | `0.9391x` | `-0.2500%` | `-0.5782%` | `0.4949` | `0.8095x` | 不成立 |
| `base_liq_only_pairwise_logistic_t0.07` | `0.5719` | `0.9340x` | 否 | `0.9753x` | `-0.0343%` | `-0.1983%` | `0.4942` | `0.8351x` | 不成立 |

forward random pool500 Top10 p95 为 `1.3126x`，所有对比模型均未超过。valid 选择项 `real_path/t=0.30` forward `1.0550x`，remove-best-3 `0.9718x`，不满足约束。

### 83.4 观察

第一，对比学习确实更适合低信噪比预测指标。`real_path` 的 valid pair AUC 到 `0.5856-0.5926`，forward pair AUC 到 `0.5365-0.5611`，明显高于 base-liq 的 forward `0.494-0.496`。这说明模型不是完全没学到东西。

第二，预测指标与交易收益再次脱节。`real_path/t=0.15` 的 forward pair AUC 最高 `0.5611`，但 forward multiple `0.9485x`，Top10/Top20 都为负。模型在全体 pair 上能分辨一些相对强弱，但 Top10 交易选择没有收益。

第三，temperature 有明确影响。`t=0.15` 让 dev Top20 五年全正、dev `4.8957x`；`t=0.30` valid 综合分最高但 dev 2022 转负；`t=0.07` 更尖锐但 forward Top20 更差。对比 loss 的温度是有效超参，但仍没解决迁移。

第四，路径输入是必要的。base-liq-only 在 valid 也能有 AUC，但 forward AUC 接近随机，forward replay 全部亏损；real_path 至少在 forward AUC 上保持正值。第 75 的路径安慰剂结论再次得到支持。

第五，完全重排可能损伤右尾。对比模型学到的是全局相对边界，不一定保留原始日线候选中的头部弹性；这与第 60/61 的 group diffusion 经验一致：模型有信息，但 pure rerank 容易削掉真正右尾。

### 83.5 反事实分析

第一反事实：如果对比学习足以解决低信噪比，valid 选择项应在 forward replay 超过 random p95。实际 `1.0550x < 1.3126x`，否定。

第二反事实：如果 pair AUC 与交易收益直接一致，`t=0.15` forward AUC 最高应对应更好 replay。实际 replay 亏损，否定。

第三反事实：如果路径特征只是噪声，real_path 不应系统性优于 base_liq_only。实际 real_path 的 dev/forward AUC 和 replay 都更好，否定“路径无效”。

第四反事实：如果问题只是 loss 形式，pure contrastive rerank 应明显超过第 74/76。实际没有，说明还需要处理 anchor、候选域或目标口径。

第五反事实：如果继续加模型容量就能解决，当前小模型至少应在 forward Top20 为正。实际 Top20 全负，说明下一步应先做 anchor/residual 反事实，而不是直接加大模型。

### 83.6 判定

`contrastive_pairwise_diffusion_ranker_not_enough`。

本轮没有可融入策略，也不进入 formal。但它保留了一个重要结论：对比学习能提升低信噪比下的预测指标，尤其 pair AUC 和 dev 稳定性；失败点在于 pure rerank 的交易收益迁移。第 84 轮做 `contrastive_pairwise_anchor_residual_v1`：将对比模型作为 residual，叠加到第 74 的 base/null 或原始 base_daily anchor 上，只允许小权重由 valid 选择，检验能否保留原始右尾弹性，同时利用对比学习改善历史稳定性。

## 84. contrastive_pairwise_anchor_residual_v1

### 84.1 假设

第 83 轮证明对比学习能提升 pair AUC 和 dev 稳定性，但 pure rerank 的交易收益不迁移。第 84 轮检查一个关键反事实：失败是否只是因为完全重排破坏了原始右尾弹性。如果是，那么把 contrastive scorer 作为小 residual，叠加到 `base_daily` 或 train-only `null_only` anchor 上，应能保住 forward 头部收益，同时改善 dev 稳定性。

本轮不扩大模型容量，只复现第 83 最有信息的 `real_path/t=0.15` 和 `real_path/t=0.30` 两个对比模型；将模型分数转为 session 内 rank residual，再与 `base_daily`、`null_only` 或 `model_only` 融合。权重由 valid 选择，forward 只评估。

### 84.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_contrastive_pairwise_anchor_residual_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/contrastive_pairwise_anchor_residual_v1_summary.json
script sha256:16916b52a21207c3988b4ccc1e92a64b336833d6fda3216e6b1c2525faae6949
summary sha256:87b262495e02d617a83098038636b0e702933f3b18661b93e69a69bfcc014cee
```

权重网格为 `0/0.05/0.10/0.15/0.20/0.30/0.40/0.60/0.80/1.00`。评估仍使用 Top10/Top20、rank IC、pair AUC、dev replay、forward replay、remove-best 和 random p95。forward random pool500 Top10 p95 为 `1.3126x`。

### 84.3 结果

结论为 `contrastive_pairwise_anchor_residual_not_enough`。按 valid 选择，第一名仍是 `model_only/t=0.30`，也就是 anchor 没被 valid 选中；它 forward 只有 `1.0550x`，未过 random p95，remove-best-3 `0.9718x`。

| config | valid score | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `t0.30 model_only` | `0.1385` | `1.9059x` | 否 | `1.0550x` | `+0.2891%` | `-0.1336%` | `0.9718x` | valid 第一，但不达标 |
| `t0.15 model_only` | `0.1368` | `4.8957x` | 是 | `0.9485x` | `-0.1208%` | `-0.0688%` | `0.8935x` | 历史强，forward 失败 |
| `t0.30 null_only w1.0` | `0.1366` | `2.3047x` | 是 | `0.8872x` | `-0.3682%` | `-0.3970%` | `0.8392x` | dev 稳但 forward 更差 |
| `t0.15 null_only w1.0` | `0.1357` | `5.4392x` | 是 | `0.8985x` | `-0.3695%` | `-0.5707%` | `0.8420x` | anchor 未改善 forward |
| `t0.15 base_daily w0.20` | `0.0804` | `1.6628x` | 否 | `1.7810x` | `+2.481%` | `+1.737%` | `1.1550x` | forward 很强，但 valid/dev 不支持 |
| `t0.15 base_daily w0.30` | `0.0838` | `1.7478x` | 否 | `1.7788x` | `+2.537%` | `+2.056%` | `1.1459x` | 有金子味，选择器不可用 |
| `t0.15 base_daily w0.60` | `0.0952` | `3.4217x` | 是 | `1.3797x` | `+1.521%` | `+1.409%` | `1.0614x` | dev 全正但仍低于 random p95 |
| `t0.15 base_daily w1.00` | `0.1083` | `3.0708x` | 是 | `1.2040x` | `+1.028%` | `+0.962%` | `1.0086x` | 稳但不够强 |

第 84 最重要的现象是：按 forward 排名，`base_daily + t0.15 residual w=0.2/0.3` 很强，forward multiple 约 `1.78x`、Top10 超 `+2.4%`、Top20 超 `+1.7%`、remove-best-3 大于 1，且打过 random p95。但这些权重的 valid score 很低，dev 也不全正，因此不能作为无未来函数候选。

### 84.4 观察

第一，anchor 确实能保住 forward 头部弹性。第 83 pure `t0.15` forward `0.9485x`，但 `base_daily + w0.2/0.3` 到 `1.78x`，Top10 超 `+2.4%`。这说明对比模型不是完全没用，问题在于如何选择残差权重和适用状态。

第二，valid 选择器完全拿不到 forward 最优权重。valid 排名前几乎都偏向 model-only 或 null anchor，高 forward 的 base_daily 小权重残差 valid score 只有 `0.080-0.084`，不是 valid 选择会自然拿到的配置。

第三，null anchor 让 dev 好看但伤 forward。`null_only + residual` 多个权重 dev 五年全正，dev multiple 约 `2-5x`，但 forward 全部亏损。这说明 train-only null 可以稳定历史统计，却可能压掉 2026 的右尾结构。

第四，base_daily anchor 暴露了“金子味但不可选”的结构。小权重 residual 在 forward 明显强，但 dev 不全正；大权重 residual dev 更稳，但 forward 变薄。这是典型 regime/权重选择问题，而不是简单模型容量问题。

第五，当前收益仍远低于用户目标。即使 forward 最好权重很强，也只是半年前向 `1.78x`，dev 不全正，更不是五年几十倍候选。

### 84.5 反事实分析

第一反事实：如果第 83 失败只是 pure rerank 太自由，valid 应选出某个 base anchor 小 residual 且 forward 改善。实际 valid 没选，否定“简单 anchor 即可解决”。

第二反事实：如果 forward 最优小权重是真稳定 alpha，dev 应至少五年全正。实际 `w0.2/0.3` dev 不全正，否定直接策略化。

第三反事实：如果 null anchor 是正确稳态，forward 不应全线变差。实际 null anchor forward 亏损，否定。

第四反事实：如果对比模型只有噪声，base_daily 加 residual 不应在 forward Top10/Top20 大幅增强。实际增强明显，否定“完全无用”。

第五反事实：如果继续只靠 valid 静态选权重，无法选中 forward 强权重。实际正是如此，下一步必须研究状态/权重选择器，而不是固定权重。

### 84.6 判定

`contrastive_pairwise_anchor_residual_not_enough`。

本轮没有可融入策略，也不进入 formal。但它给出一个值得继续追的方向：`base_daily + contrastive residual` 有 forward 右尾增强潜力，失败在“无未来函数地选择何时/多大权重”。第 85 轮做 `contrastive_residual_weight_router_diagnostic_v1`：固定 `t0.15 base_daily residual`，用 train/valid 可见的 session 状态特征（market trend、score dispersion、base/contrastive agreement、pair margin、候选拥挤度）训练或扫描一个轻量权重 router，检验是否能在不看 forward 的情况下选择接近 `w0.2/0.3` 的状态，同时保持 dev 年度稳定。

## 85. contrastive_residual_weight_router_diagnostic_v1

### 85.1 假设

第 84 轮给出一个矛盾：`base_daily + contrastive residual` 的小权重 `w=0.2/0.3` 在 forward 有明显右尾增强，但 valid 和 dev 不支持；较大权重 `w=1.0` dev 更稳但 forward 不够强。第 85 轮专门检查这个矛盾能否被 session 状态解释：如果某些当日可见状态能告诉我们何时使用 residual，小型 router 应能在 valid 上选出接近 forward 强权重的规则，并保持 dev 稳定。

本轮固定 `real_path/t=0.15` 对比模型和 `base_daily` anchor，不再扫模型。规则包括静态权重，以及基于 train 分位数的状态 gate。状态特征包括 model/base top20 overlap、model-base rank corr、model rank dispersion、base top10 在 model 中的 rank、市场 20/60 日均收益、上涨比例和成交额 rank 等。所有阈值只来自 train；valid 选择，forward 只评估。

### 85.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_contrastive_residual_weight_router_diagnostic_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/contrastive_residual_weight_router_diagnostic_v1_summary.json
script sha256:edafba903d56b1e0a27d3b54f70c1892fe367f1dafbf51a9f02ab4c962e2d499
summary sha256:972e6c39fec8d56b4b8b607f54c4199f0b31e0cf3628d1adbec99d7f784af117
```

权重候选：静态 `0/0.1/0.2/0.3/0.4/0.6/0.8/1.0`，gate 高权重 `0.2/0.3/0.4/0.6/1.0`。forward random pool500 Top10 p95 为 `1.3126x`。

### 85.3 结果

结论为 `contrastive_residual_weight_router_no_stable_selector`。valid 选择的是 `static_w1.0`，也就是没有学出有效状态 gate。该配置 dev 五年全正，forward remove-best-3 大于 1，但 forward multiple `1.2040x` 低于 random p95 `1.3126x`，收益强度不达标。

| rule | valid score | train Top20 | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | remove best 3 | 判读 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `static_w1.0` | `0.0263` | `+0.7564%` | `3.0708x` | 是 | `1.2040x` | `+1.0284%` | `+0.9619%` | `1.0086x` | valid 选择，稳但弱 |
| `mkt_ret60_high_q0.35_w0.2` | `0.0021` | `-0.437%` | `0.4630x` | 否 | `1.7810x` | `+2.481%` | `+1.737%` | `1.1550x` | forward 强，历史失败 |
| `static_w0.2` | `0.0018` | `+0.143%` | `1.6628x` | 否 | `1.7810x` | `+2.481%` | `+1.737%` | `1.1550x` | forward 强，valid 不选 |
| `mkt_ret60_high_q0.50_w0.3` | `0.0050` | `-0.490%` | `0.2974x` | 否 | `1.7788x` | `+2.537%` | `+2.056%` | `1.1459x` | 更强 forward，历史更差 |
| `model_rank_dispersion_high_q0.35_w1.0` | `0.0257` | `+0.324%` | `1.4769x` | 否 | `1.1644x` | `+0.952%` | `+1.000%` | `0.9760x` | gate 变差 |

很多 top valid gate 实际退化成全场 active，例如 `model_base_top20_overlap_high_q0.20_w1.0` 的 active rate 为 100%，本质等同静态 `w=1.0`。这说明当前状态特征没有形成有用分界。

### 85.4 观察

第一，valid 选择偏向稳定而非右尾。`static_w1.0` dev 五年全正，forward 也为正，但强度低于 random p95，不能满足用户目标。

第二，forward 最强仍是低权重 residual。`w=0.2/0.3` 在 forward Top10/Top20 很强，且 remove-best-3 大于 1，但 train/dev 明显不稳，valid 分数也低，无法无未来函数选择。

第三，简单状态特征没有解释权重差异。市场 60 日强弱、model/base overlap、rank dispersion 等都不能在 train/valid 中稳定识别低权重 residual 的适用状态。

第四，`base_daily + contrastive residual` 方向有信号，但不是策略主干。它更像一个在特定 2026 regime 下增强右尾的 residual，而不是跨年稳定收益来源。

第五，继续在同一权重 router 上调阈值会变成用 forward 反选。当前证据要求换收益来源或换标签空间，而不是继续调 gate。

### 85.5 反事实分析

第一反事实：如果 session 状态能解释 residual 权重，valid 应选出某个非退化 gate。实际选到静态 `w=1.0`，否定。

第二反事实：如果低权重 residual 是稳定 alpha，dev 应全正且 train Top20 为正。实际 `w=0.2/0.3` dev 不全正，否定直接策略化。

第三反事实：如果高权重 residual 是答案，forward 应超过 random p95。实际 `w=1.0` forward `1.2040x < 1.3126x`，否定。

第四反事实：如果简单 market trend 能捕捉适用状态，`mkt_ret60` gate 应改善 dev/forward 的一致性。实际 forward 强的 `mkt_ret60` gate 历史更差，否定。

第五反事实：如果继续在第 83-85 框架内调参能接近用户目标，应已出现 dev 五年全正且 forward 打过 p95 的配置。实际没有，方向阶段性收束。

### 85.6 判定

`contrastive_residual_weight_router_no_stable_selector`。

本轮没有可融入策略，也不进入 formal。第 83-85 的对比学习分支结论是：对比 loss 能提升预测指标和部分历史稳定性，但无法稳定转成交易收益；anchor/residual 能在 2026 增强右尾，但无法无未来函数选择权重。下一轮需要换更激进的标签空间：从“预测下周收益”切到“预测可交易爆发概率/路径成形”，即 `explosive_path_contrastive_event_model_v1`。思路是把正样本定义为未来 5-10 日内出现大幅上行且路径平滑/可持有的样本，负样本定义为同 session 内大跌或冲高回落样本，用对比学习训练小模型；先做标签土壤和模型可迁移性，不再继续围绕普通 5 日收益 rank 打磨。

## 86. explosive_path_contrastive_event_model_v1

### 86.1 假设

用户强调金融信噪比很低，loss 应尽可能采用对比学习。第 83-85 已经验证：同 session 正负样本 loss 能提升 pair AUC，但普通未来 5 日收益 rank 目标和交易收益不完全一致。第 86 轮因此更换标签空间：不再问“未来 5 日收益谁更高”，而是问“未来 5-10 日是否形成可交易、可持有的爆发路径”。

本轮只使用 t 日及以前的 60 日日线路径特征。标签只用于训练和评估，来自 t+1 之后 open/high/low/close 路径，不进入特征。正样本要求未来 10 日内有足够高点、10 日 open-to-open 留存收益为正且中途回撤/高点回吐受控；负样本包括未来 10 日明显下跌、open 路径下探，或冲高后回落失败。训练仍使用同 session 正负样本 pairwise contrastive loss。

### 86.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_explosive_path_contrastive_event_model_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/explosive_path_contrastive_event_model_v1_summary.json
script sha256:f2cd6e2379a6846e121c45253d20b5146a230188ddcfd1c406988e26bab59c2f
summary sha256:8710eede3aa591319f9f5ee11b4889ec90f15c6de4ec4eaaac923997ecb67191
```

split 仍为 train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 至 `2026-07-10`。模型为小型 MLP scorer，temperature 扫描 `0.07/0.15/0.30`。标签组为三类固定定义：`loose_explosive_holdable`、`strict_explosive_holdable`、`smooth_explosive_holdable`。forward random pool500 Top10 p95 为 `1.3054x`。

标签土壤如下：

| label | train pos | train neg | valid pos | valid neg | forward pos | forward neg | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `loose_explosive_holdable` | `15.22%` | `37.88%` | `16.73%` | `29.64%` | `16.86%` | `44.04%` | 正样本频率稳定，2026 负样本增多 |
| `smooth_explosive_holdable` | `10.12%` | `42.05%` | `11.11%` | `33.40%` | `11.53%` | `48.47%` | 更稀疏、更偏路径质量 |
| `strict_explosive_holdable` | `8.84%` | `35.08%` | `9.83%` | `27.19%` | `11.20%` | `41.98%` | 正样本少但 forward 并未枯竭 |

### 86.3 结果

结论为 `explosive_path_contrastive_event_model_not_enough`。valid 选择项是 `loose_explosive_holdable/t=0.30`，但 forward 只有 `1.0400x`，remove-best-3 为 `0.8508x`，dev replay 2022 为负，因此不能作为候选。

更重要的是，forward 最强项 `loose_explosive_holdable/t=0.07` 同时满足几个此前少见的性质：dev 五年 replay 全正、dev multiple `6.7535x`、forward `1.3938x`、forward remove-best-3 `1.1922x`、avg position `10`，并超过 random p95 `1.3054x`。但它不是 valid score 第一，不能直接策略化。

| config | valid score | dev multiple | dev all positive | forward multiple | forward Top10 ret5 | forward Top20 ret5 | forward hit | forward fail | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `loose/t0.30` | `0.1128` | `4.8384x` | 否 | `1.0400x` | `+0.374%` | `+0.663%` | `21.30%` | `43.48%` | `0.8508x` | valid 第一但 forward 失败 |
| `loose/t0.07` | `0.1090` | `6.7535x` | 是 | `1.3938x` | `+1.528%` | `+1.388%` | `18.26%` | `35.65%` | `1.1922x` | 金子味最强，但 valid 非第一 |
| `loose/t0.15` | `0.1014` | `9.9945x` | 是 | `1.0414x` | `+0.255%` | `-0.172%` | `16.09%` | `33.04%` | `0.9256x` | dev 极强但 forward 薄 |
| `smooth/t0.30` | `0.0918` | `2.8724x` | 否 | `1.1398x` | `+0.638%` | `+0.372%` | `13.04%` | `42.61%` | `0.9429x` | forward 小正但不稳 |
| `strict/t0.30` | `0.0910` | `2.1475x` | 否 | `1.1149x` | `+0.714%` | `+0.828%` | `16.96%` | `42.17%` | `0.9219x` | 不够强 |
| `smooth/t0.07` | `0.0906` | `16.9494x` | 是 | `1.2333x` | `+1.264%` | `+0.715%` | `11.30%` | `45.22%` | `0.9247x` | dev 很强，forward 未过 p95 |
| `strict/t0.07` | `0.0837` | `12.9635x` | 是 | `1.0830x` | `+0.256%` | `+0.554%` | `10.87%` | `40.87%` | `0.9566x` | 过窄，迁移弱 |
| `smooth/t0.15` | `0.0816` | `6.3078x` | 是 | `1.1277x` | `+0.774%` | `+0.502%` | `11.74%` | `37.39%` | `0.9082x` | 不够强 |
| `strict/t0.15` | `0.0812` | `2.3690x` | 否 | `0.9794x` | `+0.207%` | `+0.573%` | `10.00%` | `43.48%` | `0.8052x` | 失败 |

`loose/t0.07` 的 replay 年度收益为 dev：2021 `+111.76%`、2022 `+14.25%`、2023 `+25.15%`、2024 `+50.38%`、2025 `+48.32%`；forward 2026 `+39.38%`。这一项的 forward max drawdown 为 `-8.82%`，sharpe `4.20`。这些指标说明爆发路径标签比普通收益 rank 更贴近交易收益，但收益数量级仍远低于用户五年几十倍到 100 倍的目标。

### 86.4 观察

第一，换标签空间是有效的。第 83 pure contrastive 只在 pair AUC 上明显改善，forward replay 没有过 random p95；第 86 的 `loose/t0.07` 已经出现 dev 五年全正、forward 过 p95、remove-best-3 大于 1 的组合。这说明“可持有爆发路径”比普通收益 rank 更贴近策略目标。

第二，valid 选择纪律仍然不够。valid 第一 `loose/t0.30` 的 valid event pair AUC `0.6528`、valid hit `20.61%`，但 forward fail 高达 `43.48%`，replay 只有 `1.0400x`。forward 最强 `loose/t0.07` 的 valid score 只低约 `0.0037`，却在 forward 上显著更好。这里不是 forward 反选可以直接使用，而是提示 valid 评分函数对“hit rate”和“失败率/路径收益”的权衡还不对。

第三，temperature 不是普通超参，而是在控制右尾保留方式。`t=0.07` 更尖锐，能把爆发路径收益转成 Top10 收益；`t=0.30` valid hit 更高，但 forward fail 也更高，像是学到更宽泛的事件外壳，没有隔离冲高回落和假爆发。

第四，strict/smooth 标签未必更好。更严格或更平滑的标签能让 dev multiple 很高，尤其 `smooth/t0.07` dev `16.9494x`，但 forward 没有超过 random p95，remove-best 也低于 1。这说明正样本过窄时模型容易学到历史样式而不是稳定机制。

第五，2026 的负样本率显著更高。三种标签 forward 负样本率均高于 train/valid，说明 2026 的市场路径更容易出现失败/冲高回落。下一轮若继续这个方向，应重点降低 forward fail，而不是只提高 event hit。

### 86.5 反事实分析

第一反事实：如果爆发路径标签只是换名的收益 rank，结果应和第 83 类似，forward replay 不应过 random p95。实际 `loose/t0.07` forward `1.3938x > 1.3054x`，且 remove-best-3 `1.1922x`，否定。

第二反事实：如果 valid event pair AUC 足以选择配置，`loose/t0.30` 应在 forward 最好。实际它 forward 只有 `1.0400x`，否定。

第三反事实：如果更严格的正样本定义必然更鲁棒，`strict` 或 `smooth` 应明显胜出。实际 best forward 仍是 `loose/t0.07`，否定“越严越好”。

第四反事实：如果收益来自少数极端交易段，remove-best-3 应跌破 1。`loose/t0.07` remove-best-3 为 `1.1922x`，说明它比前面多数方向更分散；但收益仍不够大。

第五反事实：如果这个方向已达到候选策略纪律，valid 第一应同时 dev 全正、forward 过 p95、remove-best 大于 1。实际 valid 第一不满足，因此不能融入 formal。

### 86.6 判定

`explosive_path_contrastive_event_model_not_enough`。

本轮没有可融入策略，也不进入 formal。但这是第 83-86 对比学习分支里最有价值的一轮：标签空间从普通收益 rank 换成可持有爆发路径后，出现了少数同时具备 dev 五年全正、forward 过 random p95、remove-best 大于 1 的结构。下一轮应继续沿这个方向，但不要直接拿 `loose/t0.07` 做 forward 反选。第 87 轮建议做 `explosive_path_failure_aware_contrastive_selector_v1`：固定不看 forward 的标签定义，改进 valid 选择函数和 loss，重点惩罚冲高回落/失败负样本，比较 pairwise logistic 与 focal/weighted contrastive，并做 train 内部滚动年份选择，检验是否能无未来函数地选出低 fail、高收益、dev/forward 一致的形态。

## 87. explosive_path_failure_aware_contrastive_selector_v1

### 87.1 假设

第 86 轮显示“可持有爆发路径”标签比普通收益 rank 更接近交易收益，但 valid 选择函数选中了 `loose/t0.30`，forward 失败；forward 最强的 `loose/t0.07` 不能直接使用，因为它不是 valid 第一。第 87 轮专门检查一个反事实：如果第 86 的失配主要来自冲高回落/失败样本没有被充分惩罚，那么 failure-aware loss 和更严格的选择纪律应能无未来函数地选出低 fail、高收益的配置。

本轮固定第 86 的 `loose_explosive_holdable` 标签，不再看 forward 调标签。训练仍为同 session 正负 pairwise contrastive，小模型不扩大。配置包括 base logistic、失败样本加权、失败+下跌样本加权、focal contrastive。选择纪律从只看 2025 valid 改为 `2021-2023 -> 2024` 内部验证和 `2021-2024 -> 2025` 正式 valid 的组合分数，forward 2026 仍只评估。

### 87.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_explosive_path_failure_aware_contrastive_selector_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/explosive_path_failure_aware_contrastive_selector_v1_summary.json
script sha256:6dc7fbd55fdb3468db4a31bac667981654ace53885781a6cd43877236599aca0
summary sha256:88a93dcede070f1b00821a9c8d102d89a0763aab58976116756824261ed9751b
```

选择分数同时惩罚 Top10 fail、回撤和 remove-best 脆弱性，并加入 replay multiple 的对数项。forward random pool500 Top10 p95 仍为 `1.3054x`。本轮第一次运行在最终 JSON 写入时被 ndarray 序列化阻塞，修正为不保存完整预测数组并加入 JSON default 后重跑；实验口径未改。

### 87.3 结果

结论为 `explosive_path_failure_aware_selector_not_enough`。combined inner2024+valid2025 选择项为 `failed_w2_logistic_t0.07`，dev 五年全正且 dev multiple 达 `18.7855x`，非常接近用户“五年几十倍”的下沿，但 forward 只有 `1.1722x`，低于 random p95 `1.3054x`。best forward 为 `focal_g1_t0.07`，也只有 `1.2308x`，仍未过 p95。

| config | combined score | inner2024 score | valid score | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | forward hit | forward fail | remove best 3 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `failed_w2_logistic_t0.07` | `0.0303` | `-0.0210` | `0.0618` | `18.7855x` | 是 | `1.1722x` | `+0.953%` | `+0.970%` | `21.30%` | `42.61%` | `1.0057x` | 选择项，历史强但 forward 不够 |
| `failed_w2_focal_g1_t0.07` | `0.0214` | `-0.0351` | `0.0561` | `2.0380x` | 否 | `0.9381x` | `-0.154%` | `-0.172%` | `11.74%` | `27.39%` | `0.8662x` | fail 下降但收益被打掉 |
| `failed_w2_crash_w1_logistic_t0.07` | `0.0156` | `-0.0785` | `0.0732` | `15.8914x` | 是 | `1.0655x` | `+0.464%` | `+0.730%` | `15.65%` | `38.26%` | `0.9594x` | 历史强，forward 薄 |
| `failed_w3_logistic_t0.07` | `0.0131` | `-0.0307` | `0.0399` | `4.5260x` | 是 | `0.9601x` | `-0.118%` | `+0.676%` | `9.57%` | `38.26%` | `0.8713x` | 过度惩罚失败，右尾消失 |
| `base_logistic_t0.07` | `0.0033` | `-0.0287` | `0.0229` | `9.3234x` | 是 | `1.1636x` | `+0.901%` | `+0.662%` | `15.65%` | `37.83%` | `0.9311x` | 稳但不够 |
| `focal_g1_t0.07` | `-0.0029` | `-0.0688` | `0.0375` | `11.8873x` | 是 | `1.2308x` | `+1.237%` | `+0.787%` | `23.91%` | `37.83%` | `1.0266x` | forward 最好但未过 p95 |
| `base_logistic_t0.15` | `-0.0147` | `-0.0624` | `0.0145` | `2.7117x` | 否 | `1.0719x` | `+0.488%` | `+0.447%` | `19.13%` | `44.35%` | `0.8819x` | 不成立 |

选择项 `failed_w2_logistic_t0.07` 的 dev 年度收益为 2021 `+159.73%`、2022 `+39.39%`、2023 `+74.47%`、2024 `+81.22%`、2025 `+64.10%`；forward 2026 只有 `+17.22%`，max drawdown `-17.54%`。这一点很关键：内部验证和 valid 选择出一个历史非常强的模型，但 2026 的 forward 右尾并没有跟上。

### 87.4 观察

第一，failure-aware loss 改善了历史稳定性，却没有改善 forward 右尾。`failed_w2_logistic_t0.07` 的 dev multiple 从第 86 `loose/t0.07` 的 `6.7535x` 提升到 `18.7855x`，五年全正，但 forward 从 `1.3938x` 降到 `1.1722x`。

第二，过度惩罚失败样本会压掉收益来源。`failed_w2_focal_g1` 的 forward fail 降到 `27.39%`，但 Top10 ret5 变成 `-0.154%`，forward multiple `0.9381x`。这说明失败样本里可能混有“高弹性结构”的必要邻域，简单剔除会把右尾一起剔掉。

第三，focal loss 保留了一点 forward 右尾，但选择纪律不支持。`focal_g1_t0.07` forward `1.2308x` 是本轮最高，remove-best-3 `1.0266x`，但低于 random p95，combined score 也靠后。它更像是损失函数形态有一点帮助，而不是可直接策略化。

第四，内部 2024 验证没有解决选择问题。所有配置的 inner2024 selection score 都为负，说明 2024 的内部验证并没有给出清晰稳定排序；combined 选择反而更偏向 2025 valid 上的历史强配置，未能捕捉 2026 的右尾 regime。

第五，第 86 的强 forward 不是“只要更好惩罚失败”就能复制。第 87 反而把第 86 的 forward 强点压薄，说明下一步如果继续爆发路径方向，应研究候选域/市场状态/右尾保护，而不是继续加大 failure 权重。

### 87.5 反事实分析

第一反事实：如果第 86 的问题只是失败样本权重不足，failure-aware 选择项应在 forward 同时降低 fail 并提升 replay。实际选择项 forward fail `42.61%`，forward `1.1722x`，否定。

第二反事实：如果 focal loss 更适合低信噪比，`focal_g1` 应显著超过 random p95。实际 `1.2308x < 1.3054x`，否定充分性，但保留微弱方向价值。

第三反事实：如果内部 2024 验证能解决 valid 单年误选，combined 第一应更接近 forward 最优。实际 combined 第一不是 best forward，且 forward 更弱，否定。

第四反事实：如果降低 fail rate 本身就是收益来源，`failed_w2_focal_g1` 应表现更好。实际 fail 最低但收益亏损，说明 fail rate 和收益不是单调关系。

第五反事实：如果本轮已经达到候选策略纪律，选择项应 dev 接近几十倍且 forward 过 p95。实际 dev 接近但 forward 未过，不能融入 formal。

### 87.6 判定

`explosive_path_failure_aware_selector_not_enough`。

本轮没有可融入策略，也不进入 formal。第 86-87 的综合结论是：可持有爆发路径标签是目前对比学习分支里最有价值的标签空间，但单纯 failure-aware weighting 会把历史变漂亮、把 forward 右尾压薄。第 88 轮不应继续加大失败权重，而应换到“右尾保护/候选域路由”问题：在不看 forward 的条件下，识别何时使用尖锐 `t0.07` 爆发模型，以及哪些候选域更容易把 event score 转化成真实收益。建议方向为 `explosive_path_right_tail_protection_router_v1`：固定第 86 的 `loose/t0.07` 族，不再惩罚失败，改用 train/valid 可见的 session 状态和候选横截面结构（正负样本密度、模型分数分散度、近期市场趋势、候选池拥挤度、base_daily overlap）决定是否启用/降权爆发模型，并用 train 年度稳定性约束，检验能否保留 forward 右尾而不破坏 dev 全正。

## 88. explosive_path_right_tail_protection_router_v1

### 88.1 假设

第 87 轮证明：继续加大 failure-aware 权重会把历史指标做漂亮，但会压薄 2026 forward 右尾。第 88 轮因此不再惩罚失败样本，而是回到第 86 最有信息的 `loose_explosive_holdable/t=0.07` 爆发路径模型，检查能否用当日可见的 session 状态和候选横截面结构决定何时使用 model、base 或 blend，从而保留 forward 右尾，同时不破坏 dev 五年全正。

本轮训练一个固定 `loose/t=0.07` 同 session pairwise contrastive 模型，然后把模型分数和 `base_daily` 分数都转换成 session 内 rank。规则包括静态 `base_only/model_only/base_plus_w`，以及基于 train 分位阈值的 gate。gate 特征包括 model/base rank corr、Top20 overlap、model 分散度、model Top10 与中段差距、base/model 交叉排名、市场 20/60 日趋势、上涨比例、候选池成交额集中度。阈值只由 train 得到，选择用 train+valid 分数，forward 只评估。

### 88.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_explosive_path_right_tail_protection_router_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/explosive_path_right_tail_protection_router_v1_summary.json
script sha256:bf408f6139f8b41d66dbc1f757f425ff5dd21e2dc58d7442d534eda49b11d03f
summary sha256:804c9e36c41abf9874290e6992e94c86d0d88ef4314042c632a8ce2805bbde3d
```

forward random pool500 Top10 p95 为 `1.3054x`。本轮共扫描静态规则和 train 分位 gate，目标不是降低尾部权重，而是检验是否存在可由历史状态识别的右尾保护条件。

### 88.3 结果

结论为 `explosive_path_right_tail_protection_router_not_enough`。train+valid 选择项仍是 `static_model_only`，即没有学出有效 router。它 dev 五年全正、dev multiple `18.1097x`，但 forward 只有 `1.0666x`，remove-best-3 `0.8531x`，明显不达标。

forward 最强规则为 `base_top10_model_rank_mean_low_q0.65_base_plus_w1.00`，forward `1.5587x`，Top10 ret5 `+2.523%`，Top20 ret5 `+1.740%`，remove-best-3 `1.0882x`，超过 random p95。但它 dev multiple 只有 `1.9162x`，dev 年度不全正，valid score 很差，不能作为候选。

| rule | combined score | train score | valid score | dev multiple | dev all positive | forward multiple | forward Top10 | forward Top20 | forward hit | forward fail | remove best 3 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `static_model_only` | `0.1601` | `0.4634` | `-0.0031` | `18.1097x` | 是 | `1.0666x` | `+0.439%` | `+0.130%` | `13.48%` | `41.74%` | `0.8531x` | valid 选择，历史强但 forward 失败 |
| `base_top10_model_rank_mean_low_q0.80_model_only` | `0.1098` | `0.3835` | `-0.0376` | `11.4965x` | 否 | `1.3369x` | `+1.477%` | `+1.152%` | `17.83%` | `44.78%` | `0.9865x` | forward 过 p95 但 dev/稳健性不足 |
| `amount_top20_share_low_q0.80_model_only` | `0.1044` | `0.2856` | `0.0068` | `13.8510x` | 是 | `1.0978x` | `+0.894%` | `+0.687%` | `14.78%` | `42.17%` | `0.8783x` | dev 稳，forward 弱 |
| `model_rank_dispersion_high_q0.35_model_only` | `0.0698` | `0.1777` | `0.0117` | `8.7608x` | 否 | `1.0767x` | `+0.828%` | `+0.589%` | `13.91%` | `42.61%` | `0.8609x` | 不成立 |
| `static_base_plus_w1.00` | `0.0510` | `0.2369` | `-0.0490` | `6.8167x` | 否 | `1.3814x` | `+1.965%` | `+1.292%` | `23.91%` | `43.04%` | `1.0403x` | forward 有右尾，历史不稳 |
| `base_top10_model_rank_mean_low_q0.65_base_plus_w1.00` | `-0.0462` | `0.0551` | `-0.1007` | `1.9162x` | 否 | `1.5587x` | `+2.523%` | `+1.740%` | `24.35%` | `43.48%` | `1.0882x` | forward 最强，但不可选 |

`static_model_only` 的 dev 年度收益为 2021 `+129.24%`、2022 `+92.40%`、2023 `+36.01%`、2024 `+142.31%`、2025 `+24.59%`；但 forward 2026 只有 `+6.66%`。forward 最强 gate 的 dev 年度收益含 2022 `-19.52%` 和 2025 `-33.78%`，这是典型 forward 反选。

### 88.4 观察

第一，右尾保护 router 没有解决选择失配。train+valid 仍选择 `model_only`，说明历史评分更偏向模型纯排序的 dev 稳定性，而不是 forward 右尾收益。

第二，forward 右尾确实存在，但仍不可选。`base_top10_model_rank_mean_low_q0.65_base_plus_w1.00` forward `1.5587x`、remove-best 大于 1，说明 model/base 关系里有能放大 2026 收益的结构；但 dev 和 valid 明确不支持，不能融入。

第三，base+model blend 更容易保留 forward 右尾，但历史年度稳定性差。`static_base_plus_w1.00` forward `1.3814x`，但 dev 不全正；这和第 84 的 base anchor residual 现象一致：anchor 能保右尾，但无未来函数选择很难。

第四，简单 session 状态 gate 多数退化或失真。部分 overlap/dispersion gate 与静态 `model_only` 等价，说明这些状态特征没有足够解释力；能提升 forward 的 gate 在历史上反而不稳。

第五，第 86-88 的共同矛盾已经清楚：爆发路径模型能产生右尾，但历史稳定项和 forward 右尾项不是同一个方向。继续在同一 pool500 session router 上细调，很容易变成 forward 反选。

### 88.5 反事实分析

第一反事实：如果右尾保护只需要一个 session gate，valid 应选出非退化 gate 且 forward 过 p95。实际 valid 选到 `model_only`，forward `1.0666x`，否定。

第二反事实：如果 forward 最强 gate 是稳定 alpha，dev 应五年全正。实际 2022 和 2025 为负，否定。

第三反事实：如果 model/base overlap 或 dispersion 是有效状态变量，它们应在 valid 和 forward 同时改善。实际多数退化或只在 forward 好，否定。

第四反事实：如果 anchor blend 是答案，`static_base_plus_w1.00` 应 dev 全正。实际 dev 不全正，否定直接策略化。

第五反事实：如果继续在 pool500 内做同类 router 能接近用户目标，至少应出现 valid 选择项 forward 过 p95。实际没有，阶段性收束。

### 88.6 判定

`explosive_path_right_tail_protection_router_not_enough`。

本轮没有可融入策略，也不进入 formal。第 86-88 的爆发路径分支给出一个比前面更有希望但仍不够的结论：对比学习 + 可持有爆发标签能学到右尾结构，但 pool500 内的静态/简单 session gate 无法无未来函数地把右尾转成稳定策略。下一轮需要做更大胆的候选域假设，而不是继续调同一模型：`explosive_path_liquidity_tier_domain_shift_v1`。核心问题是 pool500 可能过于拥挤/高流动性，爆发路径右尾更可能在可交易但不那么拥挤的 liquidity tier 中出现。第 89 轮应扫描 pool500、pool1000、pool1500 以及 top500-1000/top1000-1500 liquidity slice，用同样的爆发路径标签和轻量对比模型/或先做标签土壤，检查是否存在更厚右尾、更高 event-to-return 转化率，仍必须保持 train/valid/forward 纪律。

## 89. explosive_path_liquidity_tier_domain_shift_v1

### 89.1 假设

第 86-88 轮已经说明：`loose_explosive_holdable/t=0.07` 的同 session 对比学习模型能在历史上形成很强的爆发路径排序，但 pool500 内的右尾在 2026 forward 中不稳定。第 89 轮不再继续调 loss 或 router，而是检查一个更底层的候选域假设：pool500 可能过于拥挤、信息定价更充分，爆发路径的右尾收益可能更容易出现在可交易但不那么拥挤的流动性层级中。

本轮扫描五个候选域：`top500`、`top1000`、`top1500`、`tier500_1000`、`tier1000_1500`。每个域使用同样的可持有爆发路径标签和轻量 pairwise contrastive 模型，固定 train `2021-2024`、valid `2025`、dev `2021-2025`、forward `2026-01-01` 到 `2026-07-10`。forward 仍只评估，不参与选择。

### 89.2 产物和口径

```text
.tmp/quantx-research/deep-learning-alpha-search-v1/analyze_explosive_path_liquidity_tier_domain_shift_v1.py
.tmp/quantx-research/deep-learning-alpha-search-v1/explosive_path_liquidity_tier_domain_shift_v1_summary.json
script sha256:9cb2f153050ca73e1e3d9f4c0c28fd73cbc979c47f88f5bdec5fd67e1df76d05
summary sha256:efb995c989e39b8d5c755a0809316b917e99ca8d5e95995c282b11ad8ebf4b3c
elapsed_seconds:385.08
```

选择分数只使用 valid 域内表现和 dev 稳定性；每个域单独计算 forward random p95，因此不能用不同域的原始 multiple 直接替代随机基准比较。

### 89.3 结果

结论为 `explosive_path_liquidity_tier_domain_shift_not_enough`。valid domain score 选择 `top1500`，dev 五年全正但只有 `3.2506x`，forward 为 `0.8722x`，低于本域 random p95 `1.1505x`，remove-best-3 为 `0.7639x`。forward 最好仍是 `top500`，dev `18.1097x` 且五年全正，但 forward 只有 `1.0666x`，低于 random p95 `1.3754x`，remove-best-3 `0.8531x`。

| domain | selection score | dev multiple | dev all positive | valid Top10 | valid hit | valid fail | forward multiple | forward random p95 | forward Top10 | forward hit | forward fail | remove best 3 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `top1500` | `0.0184` | `3.2506x` | 是 | `+0.710%` | `19.80%` | `17.35%` | `0.8722x` | `1.1505x` | `-0.450%` | `14.78%` | `39.13%` | `0.7639x` | valid 选择，forward 失效 |
| `tier1000_1500` | `0.0170` | `7.8281x` | 是 | `+1.171%` | - | - | `0.9817x` | `1.0371x` | `-0.103%` | - | - | `0.8715x` | valid 很强但 forward 不过随机 |
| `tier500_1000` | `0.0133` | `2.4774x` | 否 | `+0.720%` | - | - | `0.8543x` | `1.1391x` | `-0.713%` | - | - | `0.7609x` | 年度不稳且 forward 亏损 |
| `top500` | `0.0109` | `18.1097x` | 是 | `+0.700%` | `17.55%` | `21.22%` | `1.0666x` | `1.3754x` | `+0.439%` | `13.48%` | `41.74%` | `0.8531x` | forward 最好但仍不达标 |
| `top1000` | `0.0091` | `2.7330x` | 是 | `+0.698%` | - | - | `0.8755x` | `1.1833x` | `-0.405%` | - | - | `0.7710x` | 扩池后右尾稀释 |

`top1500` 的 forward 土壤显示：未来 10 日最大高点均值 `8.65%`，但 5 日收益均值 `-0.044%`，正样本率 `14.72%`，负样本率 `41.78%`。这说明更宽候选域里仍有冲高空间，但“可持有收益”没有跟上，模型更容易选到冲高回落或噪声弹性。

`top500` 的 forward 土壤更强：未来 10 日最大高点均值 `10.55%`，5 日收益均值 `+0.461%`，正样本率 `16.86%`，但负样本率仍高达 `44.04%`。这解释了为什么 forward 最好仍是 top500，也解释了为什么它没能过随机 p95：右尾土壤存在，但失败/回落密度过高。

### 89.4 观察

第一，扩大流动性域没有带来更稳的 event-to-return 转化。`top1500` 在 valid 上最强，forward 却直接亏损；`top1000` 和两个 tier slice 也都没有过各自 random p95。这否定了“只要去更低流动性就会有更厚右尾”的简单假设。

第二，真正的右尾仍集中在 top500，但 2026 的 failure regime 更强。`top500` forward Top10 仍为正，是所有域里最好；但 forward fail `41.74%`、remove-best-3 `0.8531x`，说明收益被少数路径和大量失败样本共同牵制。

第三，中低流动性层更像“冲高可见、持有不可见”。`top1500`/`tier1000_1500` 的 valid Top10 看起来不弱，forward 土壤也有最大高点，但 5 日可持有收益转负。对于用户要求的一周左右持仓，这类域并不自然适配，除非改成日内/分钟级卖点模型；在日线持有口径下不够。

第四，候选域本身不是主因，标签的时间结构可能才是问题。第 86-89 的共同现象是：模型能识别“会动”的股票，但对“动完还能不能持有到 5 日”判断不足。正负样本如果只围绕单一未来窗口，可能把短促脉冲和可持续趋势混在一起。

第五，本轮支持收束一个方向：不要继续在更宽 liquidity tier 里微调同一爆发标签。它没有形成无未来函数可选的稳定域。

### 89.5 反事实分析

第一反事实：如果 pool500 只是过度拥挤导致收益弱，那么 top1000/top1500 或 tier slice 应在 forward 超过 top500。实际 best forward 仍是 `top500`，否定。

第二反事实：如果 valid 域内收益能选择正确候选域，`top1500` 应在 forward 至少超过 random p95。实际 `0.8722x < 1.1505x`，否定。

第三反事实：如果中低流动性域的爆发更适合一周持有，`tier1000_1500` 应表现突出。实际 forward `0.9817x` 且 remove-best-3 `0.8715x`，否定。

第四反事实：如果第 86 的 top500 右尾只是偶然，换域后应出现完全不同的优势域。实际 top500 仍是 forward 最好，说明高流动性核心域仍有信息，但当前模型无法稳定提取。

第五反事实：如果当前爆发路径模型已接近候选策略，至少一个 valid 支持域应满足 dev 五年全正、forward 过 p95、remove-best 大于 1、平均持仓大于 5。实际没有任何域满足，不能融入 formal。

### 89.6 判定

`explosive_path_liquidity_tier_domain_shift_not_enough`。

本轮没有可融入策略，也不进入 formal。第 86-89 的结论进一步清楚：对比学习的方向是对的，爆发路径标签也比普通收益 rank 更有信息，但当前标签把“短促冲高”和“可持有趋势”混在一起，导致 valid/forward 选择不稳定。下一轮不再扩 liquidity domain，而应换到多尺度时间结构的对比目标：`multi_horizon_trend_persistence_contrastive_v1`。核心假设是，一周持仓需要同时预测短期启动和后续持续性，正样本应要求未来 2/5/10 日路径分层一致，负样本应显式包含“1-2 日冲高但 5-10 日回落”的 hard negatives。模型仍保持小型、本地可训练，优先用横截面和过去一段时间的路径特征，不引入新闻公告或非 QMT 数据。
