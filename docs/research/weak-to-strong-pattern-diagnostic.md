# 弱转强历史交易形态诊断

## 数据来源与定位

归档目录 `docs/archive/weak_to_strong_pattern_diagnostics/` 保存了对 2016-03-15 至 2026-06-10 的 482 笔弱转强历史交易所做的形态分析：

- 40 日入场前窗口；
- 15 日入场后窗口；
- 8 个无监督形态 cluster；
- 每个 cluster 的代表图和反例图；
- 一份静态 HTML 报告与 JSON 总结。

这是**交易管理诊断**，不是当前模型训练数据，也不是新的正式回测。它不考虑组合现金、重叠持仓、交易成本和替换逻辑，因此不能据此直接改变策略。

## 关键统计

| 项目 | 值 |
| --- | ---: |
| 样本交易数 | 482 |
| 平均已实现收益 | 2.34% |
| 平均最大有利波动 MFE | 11.66% |
| 平均最大不利波动 MAE | -6.95% |
| 曾出现至少 8% 机会的比例 | 55.60% |
| 峰值后出现回落的比例 | 60.79% |
| 8% 机会后最终仍为负的比例 | 19.78% |

这说明弱转强交易并非单一类型：

- 一部分交易在前几日就失败；
- 一部分有显著机会但没有有效兑现；
- 只有 MFE 超过 18% 的组具有显著更高的平均收益和更低的最终转负率。

## 为什么没有据此再加 ML 过滤器

在该诊断的时间切分中，预测“最终 success”的逻辑回归 AUC 约 0.486，未超过可用基线；预测峰值后 fade 的 AUC 也接近随机。只有“是否存在机会”有弱信息，不能支撑新的买入过滤器或复杂退出模型。

这与最终策略选择一致：

- 保留经过正式 QuantX 验证的规则退出；
- 不把静态形态分类器当作新的生产模型；
- 将这些图作为理解失败交易、审查将来退出规则的档案。

## 图像索引

每个 cluster 含一张代表图和一张反例图：

| Cluster | 代表图 | 反例图 |
| --- | --- | --- |
| 1 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_001_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_001_counter_examples.png) |
| 2 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_002_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_002_counter_examples.png) |
| 3 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_003_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_003_counter_examples.png) |
| 4 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_004_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_004_counter_examples.png) |
| 5 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_005_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_005_counter_examples.png) |
| 6 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_006_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_006_counter_examples.png) |
| 7 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_007_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_007_counter_examples.png) |
| 8 | [代表](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_008_representatives.png) | [反例](../archive/weak_to_strong_pattern_diagnostics/figures/pattern_cluster_008_counter_examples.png) |

完整原始 HTML 在 `docs/archive/weak_to_strong_pattern_diagnostics/report.html`。
