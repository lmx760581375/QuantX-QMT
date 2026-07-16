# 08 相对趋势 Flow Matching Diffusion 验证

> 建档日期：2026-07-16
>
> 当前状态：`diagnostic_only`
>
> 目标：验证“Transformer AutoEncoder encoder + conditional generative path model”是否能从过去 60 日个股路径和横截面市场指标中学习未来股票相对大盘趋势路径，并评估 flow matching 替代 DDPM loss 后的工程价值、预测价值和交易边界。

## 1. 研究问题

本实验检验一个生成式相对趋势建模假设：

```text
过去 60 日个股量价路径 + 同期市场横截面指标
  -> Transformer AutoEncoder 压缩去噪
  -> conditional flow matching 生成未来相对大盘路径分布
  -> 从 K 个采样路径中构造 score
  -> 用 RankIC、分组收益、成本后收益和 regime 表现判断是否有 alpha 价值
```

这里的核心不是预测绝对涨跌，而是预测股票相对 baseline 指数的未来路径：

```text
y_relative = y_abs - y_baseline
```

因此横截面排序指标应主要看 relative return。若评估 long-only 绝对收益，需要加回 baseline；但同一信号日内 baseline 对所有股票相同，所以横截面 RankIC 和 top-bottom spread 在 relative/absolute 之间不会改变，long-only top return 会改变。

## 2. 实验实现

实验脚本位于：

```text
tmp/relative_trend_diffusion_mvp/run_relative_trend_diffusion.py
```

实验产物位于：

```text
tmp/relative_trend_flow_base_s4/
```

本轮只在 `tmp/` 下实现和运行，不进入核心工程代码。

### 2.1 输入设计

输入是过去 60 日序列，包含两类信息：

1. 个股路径特征：价格、成交量、动量、波动等日频特征，统一转换成相对百分比或近似同量纲特征。
2. 市场横截面指标：弱转强策略、砖形图策略等常用的市场横截面状态指标，同样按过去 60 日拼接，不做单日广播。

用户确认后的 token 设计是：个股和市场特征在同一时间 token 上按 channel 维度拼接或分别 embedding 后对齐融合，而不是扩展 token 数量。这样每个 token 对应同一天，避免把市场状态广播成额外伪时间点。

### 2.2 Target 设计

训练目标是未来 30 日相对路径：

```text
stock_path[t] = future_stock_close[t] / condition_close - 1
baseline_path[t] = future_baseline_close[t] / condition_baseline_close - 1
y[t] = stock_path[t] - baseline_path[t]
```

数据集中同时保存：

```text
y            = y_relative
y_relative   = y_relative
y_abs        = stock_path
y_baseline   = baseline_path
```

本轮数据构建后验证过：

```text
max(abs((y_abs - y_baseline) - y)) = 0.0
```

## 3. 模型结构

### 3.1 Transformer AutoEncoder

AE 用于先压缩、再重建输入序列，目的是让 encoder 学到去噪后的 condition 表示。

本轮 base 配置：

```text
d_model = 96
d_stock = 64
d_market = 48
n_layers = 3
latent_tokens = 8
latent_dim = 96
latent_dim_total = 768
raw_dim = 60 * (19 + 17) = 2160
compression_ratio = 768 / 2160 = 0.3556
```

压缩比小于 1，满足“hidden state 一开始先压缩，再放大重建”的约束。

### 3.2 Flow Matching Generative Head

上一版 DDPM loss 被替换为 flow matching。训练方式：

```text
x0 ~ N(0, I)
x1 = y
t ~ U(0, 1)
xt = (1 - t) * x0 + t * x1
target_velocity = x1 - x0
loss = SmoothL1(model(xt, t, condition), target_velocity)
```

推理时用 Euler integration 从噪声积分到预测路径分布。评估中对每个样本生成 K 条路径，并统计 mean、std、prob_pos、q10、q90 等 score variant。

## 4. 数据和训练配置

本轮更严肃版本配置：

```text
output_root = tmp/relative_trend_flow_base_s4
start = 2022-01-01
end = 2024-12-31
max_instruments = 220
sample_every = 2
samples = 76560
train = 53680
val = 11440
test = 11440
test_signal_days = 52
model_size = base
generative_objective = flow
ae_epochs = 3
flow_epochs = 12
batch_size = 512
k_samples = 32
main_inference_steps = 10
inference_steps_sweep = 5,10,20
```

最初尝试 `ae_epochs=10`、`flow_epochs=40`，但 CPU 上 base AE 首个 epoch 约 2 分钟以上，完整训练时间过长。本轮改为同数据集、同 base 参数量、较少 epoch 的中等强度闭环实验，优先完成严谨评估闭环。

## 5. 训练结果

AE 训练 3 轮，验证损失持续下降：

```text
epoch 1: train_loss=0.322030, val_loss=0.313351
epoch 2: train_loss=0.119320, val_loss=0.232795
epoch 3: train_loss=0.071261, val_loss=0.192721
```

Flow matching 训练 12 轮，验证损失下降较快：

```text
epoch 1:  train_loss=0.298695, val_loss=0.116608
epoch 2:  train_loss=0.078278, val_loss=0.080467
epoch 4:  train_loss=0.053909, val_loss=0.063359
epoch 6:  train_loss=0.046852, val_loss=0.056079
epoch 8:  train_loss=0.043743, val_loss=0.053581
epoch 10: train_loss=0.041331, val_loss=0.055226
epoch 12: train_loss=0.039891, val_loss=0.050244
```

结论：在这个低维路径 target 上，flow matching 的训练效率明显好于传统 DDPM 风格 denoise loss，12 轮即可得到稳定下降的 validation loss。

## 6. AE 重建评估

AE 重建指标文件：

```text
tmp/relative_trend_flow_base_s4/runs/ae_reconstruction_metrics.csv
tmp/relative_trend_flow_base_s4/runs/ae_reconstruction_by_feature.csv
tmp/relative_trend_flow_base_s4/runs/ae_reconstruction_metrics.json
```

核心结果：

| split | masked_input | stock_mse | stock_corr_mean | market_mse | market_corr_mean | combined_mse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | false | 0.076898 | 0.960623 | 0.034818 | 0.976943 | 0.055858 |
| train | true | 0.094767 | 0.952051 | 0.040837 | 0.969999 | 0.067802 |
| val | false | 0.147051 | 0.939447 | 0.514979 | 0.750359 | 0.331015 |
| val | true | 0.163029 | 0.933297 | 0.544502 | 0.744163 | 0.353765 |
| test | false | 0.124690 | 0.943308 | 0.405762 | 0.803393 | 0.265226 |
| test | true | 0.140801 | 0.936033 | 0.428470 | 0.796802 | 0.284635 |

解读：

1. 个股路径重建质量较好，test clean stock correlation 达到 `0.9433`。
2. masked input 下性能只小幅下降，说明 AE 有一定去噪能力。
3. market/cross-section 部分 train 与 val/test 差距较大，可能反映市场状态漂移，也可能是 AE 只训 3 轮尚未充分收敛。

## 7. 主评估结果

主评估文件：

```text
tmp/relative_trend_flow_base_s4/runs/metrics.json
tmp/relative_trend_flow_base_s4/runs/predictions.parquet
```

summary：

```text
rows = 11440
days = 52
diff_score_rankic_mean = 0.021746
diff_mean_rankic_mean = 0.001737
momentum_rankic_mean = -0.029257
ridge_rankic_mean = 0.036574
diff_top_bottom_spread = -0.004637
ridge_top_bottom_spread = 0.002627
```

默认 `diff_score_10d = diff_mean_10d / diff_std_10d` 的 10 日 RankIC 为正，但 top-bottom spread 为负。Ridge baseline 的 RankIC 和 spread 当前更稳。

## 8. Horizon IC 和 Spread

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/horizon_ic.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/horizon_spread.csv
```

默认 10d score 在不同 horizon 的表现：

| label horizon | score | RankIC mean | RankIC pos ratio | spread mean | spread pos ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 5d | score_mean_over_std_10d | 0.026563 | 0.538462 | -0.001836 | 0.519231 |
| 10d | score_mean_over_std_10d | 0.021746 | 0.596154 | -0.004637 | 0.557692 |
| 20d | score_mean_over_std_10d | -0.016692 | 0.519231 | -0.015456 | 0.442308 |
| 30d | score_mean_over_std_10d | -0.035900 | 0.346154 | -0.024341 | 0.307692 |

默认 10d score 只在 5d/10d 方向有弱正排序，20d/30d 明显转弱。

## 9. Bootstrap 显著性

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/bootstrap_significance.csv
```

核心结果：

| horizon | score | IC mean | IC CI 5% | IC CI 95% | signflip p | spread mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10d | score_mean_over_std_10d | 0.021746 | -0.015864 | 0.060444 | 0.3500 | -0.004637 |
| 10d | score_prob_pos_10d | 0.028117 | -0.008397 | 0.065549 | 0.2270 | -0.004175 |
| 10d | baseline_ridge_10d | 0.036574 | 0.006148 | 0.065680 | 0.0510 | 0.002627 |

结论：flow score 有弱正 IC，但置信区间跨 0，不能判为显著。Ridge baseline 在这轮测试中统计证据更强。

## 10. 成本后收益

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/cost_stress.csv
```

默认 10d score 的 relative market-neutral 表现：

| score | cost_bps_per_side | turnover | gross_spread | net_spread |
| --- | ---: | ---: | ---: | ---: |
| score_mean_over_std_10d | 0 | 0.595365 | -0.004637 | -0.004637 |
| score_mean_over_std_10d | 10 | 0.595365 | -0.004637 | -0.005827 |
| score_mean_over_std_10d | 20 | 0.595365 | -0.004637 | -0.007018 |

Ridge baseline 的 relative market-neutral 表现：

| score | cost_bps_per_side | turnover | gross_spread | net_spread |
| --- | ---: | ---: | ---: | ---: |
| baseline_ridge_10d | 0 | 0.370098 | 0.002627 | 0.002627 |
| baseline_ridge_10d | 10 | 0.370098 | 0.002627 | 0.001887 |
| baseline_ridge_10d | 20 | 0.370098 | 0.002627 | 0.001147 |

解读：默认 flow score 还没有形成可交易 market-neutral spread。Ridge baseline 当前更健康：正 spread、低换手、成本后仍为正。

long-only 绝对收益需要注意口径。默认 flow score 的 10d top_long_only：

```text
relative gross_top_return = 0.010101
absolute gross_top_return = 0.021514
```

绝对收益更高是因为加回了大盘 baseline，这部分不能直接视为 alpha。

## 11. Inference Step Sweep

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/inference_steps_sweep.csv
```

10d `score_mean_over_std_10d`：

| inference_steps | RankIC mean | spread mean | RankIC pos ratio | spread pos ratio |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.004249 | -0.005430 | 0.500000 | 0.442308 |
| 10 | 0.011719 | -0.004697 | 0.538462 | 0.519231 |
| 20 | 0.025990 | -0.002143 | 0.519231 | 0.500000 |

结论：低维路径确实不需要上百步，但本轮不是 5 步最好。`20 steps` 明显优于 `5/10 steps`，说明后续应固定评估 `10/20` 两档，而不是默认认为 5 步足够。

## 12. Score Variants 的关键发现

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/score_variants_ic.csv
```

默认“用 10d 输出排 10d label”不是最优。更长路径统计量用于排序 5d/10d，反而出现更强信号。

10d label 上 RankIC 最强的 score：

| score | RankIC mean | RankIC IR | RankIC pos ratio |
| --- | ---: | ---: | ---: |
| score_mean_plus_q10_30d | 0.077987 | 0.444056 | 0.634615 |
| score_mean_minus_0.5std_30d | 0.076720 | 0.442708 | 0.653846 |
| score_mean_plus_q10_20d | 0.075911 | 0.604140 | 0.826923 |
| score_mean_30d | 0.074933 | 0.526772 | 0.692308 |
| score_mean_minus_0.5std_20d | 0.074033 | 0.587702 | 0.846154 |

10d label 上 spread 最强的 score：

| score | spread mean | top mean | bottom mean | spread pos ratio |
| --- | ---: | ---: | ---: | ---: |
| score_mean_minus_0.5std_20d | 0.011765 | 0.019325 | 0.007560 | 0.788462 |
| score_mean_over_std_20d | 0.010837 | 0.018726 | 0.007888 | 0.692308 |
| score_mean_20d | 0.010468 | 0.019184 | 0.008715 | 0.692308 |
| score_qmid_20d | 0.010294 | 0.019231 | 0.008937 | 0.634615 |
| score_mean_over_sqrtstd_20d | 0.009854 | 0.018635 | 0.008781 | 0.692308 |

这是本轮最有价值的线索：flow 模型生成的长路径分布形状里可能包含比直接 10d mean/std 更有用的排序信息。

但这已经有多 score variant 搜索的风险。下一轮必须先固定候选 score，再做不同 seed、时间段和股票池复验，不能把本轮最佳变体直接当作结论。

## 13. Regime 表现

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/regime_metrics.csv
```

`score_mean_over_std_10d` 在不同相对市场环境下差异明显：

| regime | RankIC mean | days | day_y10_mean |
| --- | ---: | ---: | ---: |
| weak_rel | 0.114663 | 17 | -0.016638 |
| mid_rel | 0.083452 | 17 | 0.011225 |
| strong_rel | -0.124287 | 18 | 0.043970 |

解读：默认 flow score 在弱/中等相对环境中有效，在强相对趋势环境中反向。它更像学到了“相对修复/风险调整”结构，而不是稳定追随强趋势结构。

## 14. 不确定性分层

诊断文件：

```text
tmp/relative_trend_flow_base_s4/runs/diagnostics/uncertainty_buckets.csv
```

按 `diff_std_10d` 分三层：

| bucket | std_mean | abs_y10_mean | RankIC mean |
| --- | ---: | ---: | ---: |
| low | 0.039842 | 0.039694 | 0.019194 |
| mid | 0.059143 | 0.053124 | -0.012720 |
| high | 0.087372 | 0.072559 | 0.001486 |

不确定性暂时不能直接解释为“置信度”。高 std 更像未来行情幅度或波动强度，而不是简单的低质量预测标记。

## 15. 当前结论

本轮实验结论分三层。

### 15.1 工程路线成立

1. AE encoder 能在 0.3556 压缩比下保留相当多的个股路径信息。
2. masked reconstruction 结果说明 AE 具备一定去噪能力。
3. Flow matching 比 DDPM 风格 denoise loss 更适合当前低维路径 target，收敛更快。
4. K 个采样路径能自然产出 mean、std、quantile、probability 等分布统计量，适合做 score variant 和 uncertainty diagnostics。
5. Inference steps 必须评估；本轮 `20 steps` 明显好于 `5/10 steps`。

### 15.2 预测信号存在，但默认口径不够

默认 10d `mean/std` score 有弱正 RankIC，但 spread 为负，bootstrap 不显著，不能判断为可交易 alpha。

更有价值的是：20d/30d path statistics 在 10d label 上出现更强 RankIC 和正 spread。这说明生成式路径分布可能捕捉到了“未来路径形状”而不仅是单点预测。

### 15.3 交易价值尚未证明

默认 flow score 当前弱于 ridge baseline。ridge 具备更好的 IC、正 spread、更低换手和成本后正 spread。

因此当前状态应定为 `diagnostic_only`：路线有研究价值，但不能进入策略候选或实盘逻辑。

## 16. 下一步建议

不建议立即盲目加 epoch 或扩大模型。更合理的下一步是固定本轮发现的候选规则并复验：

```text
score_mean_minus_0.5std_20d
score_mean_plus_q10_20d
score_mean_over_std_20d
score_mean_plus_q10_30d
```

复验维度：

1. 不同随机 seed 重训，确认 score variant 是否稳定。
2. 不同股票池规模，例如 180、240、320。
3. 不同时间切分或追加 forward window。
4. 固定 10/20 inference steps，不再临时挑 step。
5. 对最佳候选 score 做成本后 market-neutral 和 long-only absolute 两套口径。
6. 对 weak/mid/strong regime 分别评估，确认强相对环境反向是否稳定。

只有当固定规则在复验中同时满足正 RankIC、正 spread、成本后不崩、regime 不极端反向，才值得进入正式账户回放或更大模型训练。

## 17. 复现证据

本轮已确认的产物：

```text
tmp/relative_trend_flow_base_s4/runs/ae_checkpoint.pt
tmp/relative_trend_flow_base_s4/runs/diffusion_checkpoint.pt
tmp/relative_trend_flow_base_s4/runs/predictions.parquet
tmp/relative_trend_flow_base_s4/runs/metrics.json
tmp/relative_trend_flow_base_s4/runs/metrics.csv
tmp/relative_trend_flow_base_s4/runs/ae_reconstruction_metrics.csv
tmp/relative_trend_flow_base_s4/runs/ae_reconstruction_by_feature.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/horizon_ic.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/horizon_spread.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/bootstrap_significance.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/cost_stress.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/regime_metrics.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/uncertainty_buckets.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/inference_steps_sweep.csv
tmp/relative_trend_flow_base_s4/runs/diagnostics/score_variants_ic.csv
```

脚本语法检查通过：

```text
conda run -n test python -m py_compile tmp/relative_trend_diffusion_mvp/run_relative_trend_diffusion.py
```
