# Trade Pattern ML Analysis Design

本文档定义 QuantX 的交易形态机器学习分析模块。目标是把弱转强等策略产生的大量模拟交易记录，转换成可复盘、可统计、可训练的交易形态样本库，并用 LightGBM、XGBoost、RandomForest、Logistic Regression 等方法分析成功/失败交易的 K 线与量价共性。

第一版重点不是直接训练一个自动交易模型，而是建立稳定的数据集、标签、特征、分析报告和相似形态检索能力，为后续策略过滤器、形态命名和模型化决策提供依据。

## 1. 目标

核心问题：

```text
给定一笔历史交易及其买点前后更宽的日线视野，系统能否回答：
1. 这笔交易属于哪类历史形态？
2. 成功交易和失败交易在买点前后的量价结构上有什么差异？
3. 哪些 K 线柱子、窗口结构、量能和相对强弱特征最能解释成败？
4. 新交易最像历史上哪些交易，它们后续表现如何？
5. 某个策略的收益主要来自哪些形态族，亏损主要来自哪些形态族？
```

第一版产物：

1. 交易形态样本库，按交易窗口落盘。
2. 单根 K 线柱子特征表。
3. 窗口级形态特征表。
4. 成功/失败标签和多目标结果标签。
5. 监督学习分析结果，包含特征重要性和分箱表现。
6. 聚类/相似样本结果，包含每个形态族的表现统计。
7. HTML/Markdown 报告，包含代表 K 线图和形态族摘要。

非目标：

1. 第一版不把模型预测结果直接接入实盘或生产交易。
2. 第一版不要求自动生成完美中文形态名。
3. 第一版不使用卖点后数据作为训练输入，避免未来数据泄露。
4. 第一版不依赖深度学习图像模型；K 线图片 embedding 可作为后续扩展。

## 2. 数据来源

模块从标准 run artifacts 和本地日线数据构建样本。

输入目录：

```text
runs/<run_id>/
  summary.json
  metrics.json
  trades.json
  closed_positions.json
  positions.json
  daily_nav.json
  config.yaml
```

优先使用 `closed_positions.json` 作为一笔完整交易的来源。如果某些 run 没有 `closed_positions.json`，可从 `trades.json` 按 symbol FIFO 匹配买卖生成闭合交易。

日线数据来源使用 QuantX 现有数据层和默认 Qlib provider：

```text
data/qlib_data_fixed
```

每笔交易需要扩展更宽的 K 线窗口：

```text
window_start = entry_date 前 pre_n 个交易日
window_end   = exit_date 后 post_n 个交易日
```

推荐默认值：

```text
pre_n = 40
post_n = 15
```

研究弱转强、高位反包、回踩再转强等形态时，可以把 `pre_n` 提高到 `60`，用于观察更长的蓄势、加速、衰竭和压力位结构。

## 3. 数据集分层

模块应落盘三类核心数据，避免每次分析都重新抽取 K 线。

```text
artifacts/pattern_analysis/<analysis_id>/
  config.json
  trade_windows.parquet
  bar_features.parquet
  window_features.parquet
  labels.parquet
  model_results.json
  cluster_results.parquet
  pattern_summary.json
  report.html
  figures/
```

### 3.1 `trade_windows.parquet`

一笔交易一行，记录窗口边界和交易结果。

字段：

| 字段 | 说明 |
|------|------|
| `sample_id` | 稳定样本 ID，建议 `run_id:symbol:entry_date:exit_date` |
| `run_id` | 来源 run |
| `strategy_name` | 来源 config 或策略名 |
| `symbol` | 股票代码 |
| `entry_date` | 买入日期 |
| `exit_date` | 卖出日期 |
| `entry_price` | 买入价格 |
| `exit_price` | 卖出价格 |
| `holding_days` | 持仓交易日数 |
| `window_start` | K 线窗口起点 |
| `window_end` | K 线窗口终点 |
| `pre_n` | 买点前扩展交易日数 |
| `post_n` | 卖点后扩展交易日数 |
| `return` | 交易最终收益 |
| `pnl` | 交易盈亏金额，可选 |
| `max_favorable_excursion` | 买入后最大浮盈 |
| `max_adverse_excursion` | 买入后最大浮亏/最大不利波动 |
| `source_artifact` | 来源 artifact 路径 |

### 3.2 `bar_features.parquet`

一笔交易窗口内每天一行，保留每天 K 线柱子的形态。该表是后续形态聚类、相似样本检索和可视化复盘的基础。

主键：

```text
sample_id, symbol, date
```

对齐字段：

| 字段 | 说明 |
|------|------|
| `offset_from_entry` | 相对买点的交易日偏移，买点为 0 |
| `offset_from_exit` | 相对卖点的交易日偏移，卖点为 0 |
| `is_pre_entry` | 是否买点前 |
| `is_holding` | 是否持仓区间 |
| `is_post_exit` | 是否卖点后观察区间 |

原始行情字段：

```text
open, high, low, close, volume, amount, turnover
```

单日 K 线柱子形态字段：

| 字段 | 计算/含义 |
|------|-----------|
| `return_pct` | 当日涨跌幅 |
| `gap_pct` | `open / prev_close - 1` |
| `range_pct` | `(high - low) / prev_close` |
| `body_pct` | `abs(close - open) / prev_close` |
| `body_to_range` | `abs(close - open) / (high - low)` |
| `upper_shadow_ratio` | `upper_shadow / (high - low)` |
| `lower_shadow_ratio` | `lower_shadow / (high - low)` |
| `close_position` | `(close - low) / (high - low)` |
| `is_up_day` | `close > open` |
| `is_down_day` | `close < open` |
| `is_doji` | 实体占振幅较小 |
| `is_long_body` | 实体显著大于近期均值 |
| `is_long_upper_shadow` | 上影线占比高 |
| `is_long_lower_shadow` | 下影线占比高 |
| `is_wide_range` | 振幅显著大于近期均值 |

量能字段：

| 字段 | 说明 |
|------|------|
| `volume_ratio_5` | 成交量 / 5 日均量 |
| `volume_ratio_20` | 成交量 / 20 日均量 |
| `amount_ratio_20` | 成交额 / 20 日均额 |
| `volume_trend_5` | 近 5 日量能趋势斜率 |
| `is_high_volume` | 是否显著放量 |
| `is_volume_contracting` | 是否相对缩量 |

位置和趋势字段：

| 字段 | 说明 |
|------|------|
| `ma5_distance` | 收盘价相对 MA5 偏离 |
| `ma10_distance` | 收盘价相对 MA10 偏离 |
| `ma20_distance` | 收盘价相对 MA20 偏离 |
| `ma60_distance` | 收盘价相对 MA60 偏离 |
| `ma5_slope` | MA5 斜率 |
| `ma20_slope` | MA20 斜率 |
| `breakout_20d_high` | 是否突破 20 日新高 |
| `distance_to_20d_high` | 距 20 日高点距离 |
| `distance_to_60d_high` | 距 60 日高点距离 |

### 3.3 `window_features.parquet`

一笔交易一行，聚合买点前、买点当天、持仓期间和卖点后观察区间的特征。训练模型时只允许使用买点前和买点当天可见字段。

分组字段建议使用统一前缀：

```text
pre_5_*
pre_10_*
pre_20_*
pre_40_*
entry_*
hold_*
post_*
```

买点前窗口特征：

| 字段 | 说明 |
|------|------|
| `pre_5_return` | 买点前 5 日涨幅 |
| `pre_10_return` | 买点前 10 日涨幅 |
| `pre_20_return` | 买点前 20 日涨幅 |
| `pre_20_max_drawdown` | 买点前 20 日最大回撤 |
| `pre_20_volatility` | 买点前 20 日收益波动 |
| `pre_20_range_mean` | 买点前 20 日平均振幅 |
| `pre_20_up_day_ratio` | 买点前 20 日阳线比例 |
| `pre_20_long_upper_count` | 买点前 20 日长上影数量 |
| `pre_20_long_lower_count` | 买点前 20 日长下影数量 |
| `pre_20_high_volume_count` | 买点前 20 日放量日数量 |
| `pre_20_volume_trend` | 买点前 20 日量能趋势 |
| `pre_20_volume_price_corr` | 买点前 20 日量价相关性 |
| `pre_20_close_position_mean` | 买点前 20 日平均收盘位置 |
| `pre_20_above_ma20_ratio` | 买点前 20 日站上 MA20 比例 |

买点当天特征：

| 字段 | 说明 |
|------|------|
| `entry_return_pct` | 买点当天涨跌幅 |
| `entry_gap_pct` | 买点当天跳空幅度 |
| `entry_body_pct` | 买点当天实体大小 |
| `entry_upper_shadow_ratio` | 买点当天上影线比例 |
| `entry_lower_shadow_ratio` | 买点当天下影线比例 |
| `entry_close_position` | 买点当天收盘位置 |
| `entry_volume_ratio_20` | 买点当天量比 |
| `entry_breakout_20d_high` | 买点当天是否突破 20 日新高 |
| `entry_distance_to_ma20` | 买点当天距 MA20 偏离 |
| `entry_distance_to_60d_high` | 买点当天距 60 日高点距离 |

持仓后和卖点后字段默认只用于复盘、标签生成和报告，不作为买点预测训练输入：

| 字段 | 说明 |
|------|------|
| `hold_max_return` | 持仓期最大浮盈 |
| `hold_min_return` | 持仓期最大浮亏 |
| `hold_first_3d_return` | 买入后 3 日收益 |
| `hold_first_5d_return` | 买入后 5 日收益 |
| `hold_drawdown_after_peak` | 冲高后回撤 |
| `post_5_return` | 卖点后 5 日表现 |
| `post_10_return` | 卖点后 10 日表现 |

若分析目标是“卖出时是否应该等待反弹”，可以使用 `feature_scope=trade_management`。该模式允许使用卖出时已经可见的字段，例如 `hold_*`、`holding_days`、`return`、`max_favorable_excursion`、`max_adverse_excursion`、`exit_reason_*`，但仍然禁止使用 `post_*`。`post_*` 只能用于定义事后标签，例如是否能在卖出后 20 日内缩小亏损或转正。

## 4. 标签设计

标签应支持多口径，避免只用最终收益定义成功/失败。

### 4.1 分类标签

默认三分类：

```text
success: return >= success_return_threshold 且 max_adverse_excursion >= -max_adverse_threshold
failure: return <= failure_return_threshold 或 max_adverse_excursion <= -hard_loss_threshold
neutral: 其他样本
```

推荐默认阈值：

```text
success_return_threshold = 0.05
failure_return_threshold = 0.00
hard_loss_threshold = 0.05
max_adverse_threshold = 0.08
```

还应支持更贴近弱转强策略的行为标签：

| 标签 | 含义 |
|------|------|
| `quick_confirm_success` | 买入后 3 或 5 日内快速浮盈确认 |
| `slow_success` | 最终盈利但确认较慢 |
| `buy_and_drop_failure` | 买入后很快下跌 |
| `spike_and_fade_failure` | 买入后冲高回落 |
| `sideways_failure` | 持仓期横盘消耗，最终收益差 |

### 4.2 回归目标

监督模型不仅可以做成功/失败分类，也可以预测连续目标：

```text
return
max_favorable_excursion
max_adverse_excursion
hold_first_3d_return
hold_first_5d_return
holding_days
```

第一版优先做分类，回归目标作为报告补充。

## 5. 训练输入边界和数据泄露规则

训练输入必须严格区分 `train_features` 和 `review_context`。

可作为训练输入：

```text
pre_*       # 买点前窗口特征
entry_*     # 买点当天收盘后可见特征；若用于盘中预测，需要另设 intraday 口径
market_*    # 买点当天及之前的大盘/行业环境
meta_*      # 股票静态或买点前已知元数据，如行业、市值分组
```

不可作为训练输入：

```text
hold_*
post_*
exit_*
return
max_favorable_excursion
max_adverse_excursion
任何由买点后行情计算出的字段
```

补充规则：当 `feature_scope=trade_management` 用于退出管理分析时，训练时点从“买点”变成“当前卖出决策点”。此时 `hold_*`、当前 `return`、MFE/MAE 和 `exit_reason_*` 是已知信息，可以用于判断是否值得延迟退出；但 `post_*` 仍然是未来数据，必须排除。

这些字段可以用于：

1. 生成标签。
2. 生成复盘报告。
3. 解释交易结果。
4. 检查卖点后是否卖飞或止损有效。

## 6. 模型路线

第一版支持四类基础模型：

| 模型 | 用途 | 优点 |
|------|------|------|
| Logistic Regression | 线性基线 | 可解释、稳定、容易发现方向性 |
| RandomForest | 非线性基线 | 对特征尺度不敏感，能发现组合关系 |
| LightGBM | 主力树模型 | 训练快、效果好、特征重要性强 |
| XGBoost | 对照树模型 | 稳健，适合和 LightGBM 互相验证 |

推荐第一版训练流程：

```text
1. 读取 window_features + labels。
2. 过滤 neutral 或采用三分类。
3. 按时间切分训练/验证/测试集，避免随机切分造成未来泄露。
4. 训练 Logistic Regression 作为基线。
5. 训练 RandomForest / LightGBM / XGBoost。
6. 输出 AUC、Precision、Recall、F1、分位数组合收益。
7. 输出 feature importance 和可选 SHAP 解释。
8. 按策略、年份、市场环境、行业分层复核。
```

时间切分建议：

```text
train:      2020-01-01 到 2023-12-31
validation: 2024-01-01 到 2024-12-31
test:       2025-01-01 到 2026-12-31
```

实际切分应由分析配置控制。

评估指标不仅看分类准确率，还要看交易价值：

| 指标 | 说明 |
|------|------|
| `auc` | 成功/失败区分能力 |
| `precision_at_top_k` | 模型最看好的样本中成功比例 |
| `return_by_score_quantile` | 按模型分数分组后的平均收益 |
| `drawdown_by_score_quantile` | 高分组是否降低回撤 |
| `coverage` | 模型筛选后保留样本比例 |
| `lift` | 高分组相对全样本胜率提升 |

## 7. 聚类和相似样本检索

监督学习用于回答“哪些特征区分成功失败”，聚类用于回答“市场里自然存在什么形态族”。

第一版可以使用：

```text
StandardScaler + PCA + KMeans
```

后续可扩展：

```text
UMAP + HDBSCAN
sequence embedding
K-line image embedding
```

聚类输入建议使用买点前序列特征和买点当天特征：

```text
offset -40 到 0 的 return_pct 序列
offset -40 到 0 的 volume_ratio_20 序列
offset -40 到 0 的 close_position 序列
offset -40 到 0 的 upper_shadow_ratio / lower_shadow_ratio 序列
entry_* 特征
```

每个 cluster 输出：

| 字段 | 说明 |
|------|------|
| `cluster_id` | 形态族 ID |
| `sample_count` | 样本数 |
| `success_rate` | 成功率 |
| `avg_return` | 平均收益 |
| `median_return` | 中位收益 |
| `avg_mfe` | 平均最大浮盈 |
| `avg_mae` | 平均最大浮亏 |
| `avg_holding_days` | 平均持仓天数 |
| `top_features` | 相比全样本最突出的特征 |
| `representative_samples` | 离簇中心最近的代表交易 |
| `counter_examples` | 同簇中的失败样本或反例 |

相似样本检索：

```text
输入一笔交易或一个 sample_id
-> 取其训练输入特征向量
-> 在历史样本中找 top_k 最近邻
-> 输出相似样本的收益、回撤、K 线图、cluster 分布
```

## 8. 形态命名策略

第一版不要求模型直接给形态取名。形态命名分两步：

```text
阶段一：自动编号和客观描述
pattern_cluster_001
pattern_cluster_002
pattern_cluster_003

阶段二：研究者查看代表图后手动命名
pattern_cluster_003 -> 缩量横盘后放量突破
pattern_cluster_007 -> 高位放量长上影失败
```

自动描述由 cluster 的突出特征生成，不强行取文学化名称。

描述维度：

| 维度 | 示例 |
|------|------|
| 位置 | 低位、中位、高位、远离均线、接近前高 |
| 节奏 | 蓄势、回踩、加速、衰竭、反包 |
| 量能 | 缩量、温和放量、巨量、量能递增 |
| K 线 | 强收盘、长上影、长下影、大实体、小实体 |
| 趋势 | 均线上行、均线纠缠、突破、跌破 |

临时描述示例：

```text
低波动蓄势_温和放量_强收盘
高位加速_巨量_长上影
回踩均线_缩量_再转强
远离均线_连续加速_冲高回落
```

手动命名应保存在 `pattern_summary.json` 中，后续分析复用同一套名称。

## 9. 报告设计

报告需要同时服务机器和人眼复盘。

### 9.1 总览报告

内容：

1. 输入 run 列表、样本数量、时间范围。
2. 成功/失败/中性样本数量。
3. 收益、最大浮盈、最大浮亏、持仓天数分布。
4. 按年份、策略、行业、市场环境的样本分布。
5. 数据缺失和过滤样本说明。

### 9.2 成功/失败对比

内容：

1. 每个核心特征在成功组和失败组的均值、中位数、分位数差异。
2. 特征分箱后的胜率、平均收益和样本数。
3. 关键条件组合表现，例如：

```text
entry_close_position > 0.8
entry_volume_ratio_20 between 1.5 and 4.0
pre_20_max_drawdown > -0.12
entry_distance_to_ma20 < 0.15
```

### 9.3 模型报告

内容：

1. 各模型训练/验证/测试指标。
2. 特征重要性排名。
3. 模型分数分位数组合表现。
4. 高分样本代表 K 线。
5. 高分但失败、低分但成功的反例。

### 9.4 Cluster 报告

每个 cluster 一段摘要：

```text
Cluster: pattern_cluster_003
样本数: 218
胜率: 63.8%
平均收益: 7.2%
平均最大浮亏: -3.1%
临时描述: 低波动蓄势_温和放量_强收盘
突出特征:
  - pre_20_volatility 低于全样本 P30
  - entry_volume_ratio_20 高于全样本 P70
  - entry_close_position 高于全样本 P75
代表交易:
  - SH600000 2024-03-12
  - SZ000001 2024-05-20
反例:
  - SH600123 2024-08-16
```

配图：

1. cluster 代表 K 线网格图。
2. cluster 中成功样本和失败样本对照图。
3. UMAP/PCA 二维散点图，颜色表示成功/失败或收益。

## 10. 模块结构建议

第一版代码结构建议如下，后续实现时再根据现有包结构微调：

```text
quantx/core/analysis/patterns/
  __init__.py
  config.py              # PatternAnalysisConfig
  artifacts.py           # 读取 runs/<run_id> artifacts
  window_builder.py      # 构建扩展 K 线窗口
  bar_features.py        # 单根 K 线柱子特征
  window_features.py     # 窗口级聚合特征
  labels.py              # 成功/失败/行为标签
  dataset.py             # 样本库落盘和读取
  supervised.py          # LR/RF/LGB/XGB 分析
  clustering.py          # PCA/KMeans/相似样本检索
  naming.py              # cluster 自动描述和手动名称管理
  plotting.py            # K 线图、cluster 网格图
  report.py              # HTML/Markdown 报告

quantx/tools/
  analyze_trade_patterns.py
```

CLI 草案：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/run_a runs/run_b runs/run_c \
  --pre-n 40 \
  --post-n 15 \
  --label-preset weak_to_strong \
  --models logistic random_forest lightgbm \
  --output-dir artifacts/pattern_analysis
```

支持只构建数据集：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/latest \
  --build-dataset-only
```

支持查询相似样本：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --analysis-id 20260707_weak_to_strong \
  --similar-to sample_id \
  --top-k 50
```

## 11. 配置草案

分析配置建议用 JSON/YAML 保存到每个 analysis 目录，保证结果可复现。

```yaml
analysis:
  name: weak_to_strong_pattern_lab
  input_runs:
    - runs/20260705_165616_shuijiao_legacy_formula
  output_dir: artifacts/pattern_analysis

window:
  pre_n: 40
  post_n: 15
  min_pre_bars: 30
  min_post_bars: 5

labels:
  preset: weak_to_strong
  success_return_threshold: 0.05
  failure_return_threshold: 0.00
  hard_loss_threshold: 0.05
  max_adverse_threshold: 0.08
  quick_confirm_days: 5
  quick_confirm_return: 0.04

features:
  train_prefixes:
    - pre_
    - entry_
    - market_
    - meta_
  review_prefixes:
    - hold_
    - post_
  include_market_context: true
  include_industry_context: true

split:
  method: time
  train_end: "2023-12-31"
  validation_end: "2024-12-31"
  test_end: "2026-12-31"

models:
  enabled:
    - logistic_regression
    - random_forest
    - lightgbm
  random_state: 42
  class_weight: balanced

clustering:
  enabled: true
  method: pca_kmeans
  n_clusters: 12
  sequence_offsets: [-40, 0]
  sequence_fields:
    - return_pct
    - volume_ratio_20
    - close_position
    - upper_shadow_ratio
    - lower_shadow_ratio

report:
  top_feature_count: 30
  representative_sample_count: 20
  similar_sample_count: 50
  render_kline: true
```

## 12. 依赖和安装策略

第一版尽量使用现有依赖：

```text
pandas
numpy
scikit-learn
matplotlib
```

可选依赖：

```text
lightgbm
xgboost
shap
mplfinance
umap-learn
hdbscan
```

由于公司网络和 pip 源限制，新增依赖时应使用公司源：

```bash
pip install -i https://pypi.hobot.cc/simple lightgbm xgboost
```

第一版实现应允许可选依赖缺失：

1. 没有 LightGBM 时跳过 LightGBM。
2. 没有 XGBoost 时跳过 XGBoost。
3. 没有 SHAP 时只输出内置 feature importance。
4. 没有 mplfinance 时使用 matplotlib 绘制简化 K 线。

## 13. 验收标准

数据集：

- [ ] 给定一个或多个 run，可以生成 `trade_windows.parquet`、`bar_features.parquet`、`window_features.parquet`、`labels.parquet`。
- [ ] 每个样本包含买点前 `pre_n` 和卖点后 `post_n` 的窗口边界。
- [ ] `bar_features.parquet` 包含 `offset_from_entry`，可按买点对齐分析。
- [ ] 训练输入字段不包含 `hold_*`、`post_*`、`return`、`max_favorable_excursion`、`max_adverse_excursion`。

标签：

- [ ] 成功/失败/中性标签可由配置阈值复现。
- [ ] 行为标签能识别快速确认、买入即跌、冲高回落等典型结果。

模型：

- [ ] Logistic Regression 和 RandomForest 在无额外依赖下可运行。
- [ ] LightGBM/XGBoost 在依赖存在时自动启用。
- [ ] 输出模型指标、特征重要性、模型分数分位数组合表现。
- [ ] 使用时间切分，默认不使用随机切分。

聚类和报告：

- [ ] 可以生成 cluster ID、cluster 表现统计和代表样本列表。
- [ ] 每个 cluster 至少输出样本数、胜率、平均收益、平均最大浮亏、临时描述。
- [ ] 报告中能查看成功样本、失败样本、反例样本的 K 线图。
- [ ] 相似样本检索能返回 top-k 历史交易及其后续表现。

## 14. 后续扩展

1. 把高失败率 cluster 转成策略过滤器。
2. 把高胜率 cluster 转成弱转强策略的加分项。
3. 引入行业/板块相对强弱和大盘环境标签。
4. 引入 K 线图片 embedding，用视觉相似度辅助检索。
5. 增加 active learning：研究者手动标注形态名后，训练命名/分类器。
6. 把分析结果接入 Web workspace，支持按 cluster、策略、年份筛选。
7. 做跨策略验证，检查同一形态族是否在不同弱转强策略中稳定有效。

## 15. 2026-07-07 弱转强最佳 run 退出管理诊断

本节记录一次真实 run 的诊断结果，用于回答“失败单是否可以通过等反弹来减少亏损或提高胜率”。

输入 run：

```text
runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85
```

分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_recoverable_failure_trade_mgmt_targets_v2_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

输出目录：

```text
artifacts/pattern_analysis/weak_best_recoverable_failure_trade_mgmt_targets_v2_20260707
```

### 15.1 总体结论

样本数 `632`，失败交易 `243`。失败交易平均收益 `-8.03%`。

卖点后 20 日复盘显示：

| 指标 | 数值 |
|------|------|
| 可恢复失败单比例 | `58.44%` |
| 收盘价可缩小亏损 2pct 以上比例 | `44.03%` |
| 卖后 20 日内最高收盘价可转正比例 | `39.09%` |
| 卖后 20 日内最高价可转正比例 | `45.27%` |
| 平均收盘改善 | `+1.74%` |
| 平均最高价改善 | `+9.20%` |

这说明“很多失败单割在反弹前”是存在的，但它不是一个简单的统一等待问题。不同卖出原因的后续特征不同：

| 卖出原因 | 失败数 | 平均亏损 | 可恢复 | 可缩亏 | 收盘可转正 | 最高价可转正 |
|----------|--------|----------|--------|--------|------------|--------------|
| `stop_loss_89permil` | `146` | `-10.93%` | `54.79%` | `50.00%` | `26.71%` | `32.88%` |
| `time_stop_25d` | `97` | `-3.67%` | `63.92%` | `35.05%` | `57.73%` | `63.92%` |

直观解释：

1. `time_stop_25d` 更像“没涨出来但还没坏透”，更容易等到收盘转正。
2. `stop_loss_89permil` 更像“已经跌深后有技术反抽”，更常见的是缩小亏损，不一定能转正。
3. 统一等待规则会把这两类混在一起，因此信号不准。

### 15.2 模型诊断

`feature_scope=entry` 只看买点前和买点当天时，恢复类目标基本学不出来。`feature_scope=trade_management` 加入卖出时已知的持仓状态后，信号有所增强，但仍属于弱信号。

RandomForest 的关键目标表现：

| 目标 | 样本数 | 正例率 | AUC | 准确率相对多数类 |
|------|--------|--------|-----|------------------|
| `recoverable_failure` | `243` | `58.44%` | `0.536` | `-8.20pct` |
| `loss_reducible` | `243` | `44.03%` | `0.583` | `+3.28pct` |
| `close_turn_profitable` | `243` | `39.09%` | `0.637` | `+6.56pct` |

结论：

1. 混合标签 `recoverable_failure` 太宽，包含“缩亏”和“转正”两种不同问题，模型很难学。
2. `loss_reducible` 有弱信号，适合用于“是否可以延迟止损以减少亏损”的候选规则。
3. `close_turn_profitable` 信号最清楚，适合优先做“哪些 time-stop 失败单值得等转正”的规则验证。

### 15.3 可缩小亏损单子的特征

目标：`loss_reducible_after_exit`，定义为失败单卖出后 20 日内固定窗口收盘收益相比原卖点改善至少 `2pct`。

差异最大的特征方向：

| 特征 | 可缩亏均值 | 不可缩亏均值 | 解释 |
|------|------------|--------------|------|
| `exit_reason_stop_loss` | `68.22%` | `53.68%` | stop-loss 单更容易出现技术性反抽缩亏 |
| `exit_reason_time_stop` | `31.78%` | `46.32%` | time-stop 对“缩亏”不是最强目标 |
| `return` | `-8.67%` | `-7.53%` | 跌得更深的 stop-loss 单，后续反抽空间更大 |
| `max_adverse_excursion` / `hold_min_return` | `-10.35%` | `-9.45%` | 最大浮亏更深，反抽缩亏概率更高 |
| `pre_40_above_ma20_ratio` | `71.73%` | `66.79%` | 前期仍有较多时间在 MA20 上方，结构未完全破坏 |
| `pre_10_long_lower_count` | `1.96` | `1.65` | 近期下影线更多，可能有承接或反抽惯性 |

候选假设：

```text
如果是 stop_loss 触发，且前 40 日大部分仍在 MA20 上方、近期有下影承接，
则不要简单固定等 N 天，而是尝试“反抽到成本附近/短均线附近就退出”的缩亏规则。
```

注意：这类目标不一定提高胜率，因为很多单只是少亏，不会转正。它更适合降低平均亏损和回撤，而不是直接追求胜率。

### 15.4 可等到收盘转正单子的特征

目标：`close_turn_profitable_after_exit`，定义为失败单卖出后 20 日内最高收盘收益可超过 `0`。

差异最大的特征方向：

| 特征 | 可转正均值 | 不可转正均值 | 解释 |
|------|------------|--------------|------|
| `return` | `-6.21%` | `-9.20%` | 当前亏损越浅，越可能等到收盘转正 |
| `exit_reason_time_stop` | `58.95%` | `27.70%` | time-stop 明显更适合等待转正 |
| `exit_reason_stop_loss` | `41.05%` | `72.30%` | stop-loss 单转正概率较低 |
| `max_adverse_excursion` / `hold_min_return` | `-8.91%` | `-10.45%` | 持仓期间跌得没那么深，更容易修复 |
| `pre_40_above_ma20_ratio` | `72.39%` | `66.77%` | 中期结构更强 |
| `hold_first_10d_return` | `-2.21%` | `-4.02%` | 买入后前 10 日没明显崩，后续修复概率更高 |
| `max_favorable_excursion` / `hold_max_return` | `6.17%` | `4.86%` | 持仓中曾经给过更高浮盈，说明弹性仍在 |

候选假设：

```text
如果是 time_stop 退出、当前亏损不深、MAE 未明显恶化、前 10 日没有大幅走坏，
则可以试验延迟退出或转为观察仓，目标是等收盘转正或接近成本退出。
```

这类目标才更可能提高胜率，但覆盖率应控制。若把 stop-loss 深亏单也纳入，容易把回撤扩大。

### 15.5 下一步验证优先级

建议先验证三个规则族，而不是继续写统一等待规则：

候选规则的事后诊断结果如下。该表只使用卖点后标签复盘筛选条件的质量，不代表真实组合回测收益。

| 候选规则 | 用途 | 样本数 | 覆盖失败单 | 原始均亏 | 平均 MAE | 可恢复 | 可缩亏 | 收盘可转正 | 最高价可转正 | 平均收盘改善 | 平均最高价改善 |
|----------|------|--------|------------|----------|----------|--------|--------|------------|--------------|--------------|----------------|
| `time_stop_shallow_wait_profit` | 等收盘转正 | `67` | `27.57%` | `-3.09%` | `-6.00%` | `68.66%` | `32.84%` | `61.19%` | `68.66%` | `+0.82%` | `+7.92%` |
| `time_stop_conservative_wait_profit` | 等收盘转正 | `58` | `23.87%` | `-2.14%` | `-5.89%` | `77.59%` | `32.76%` | `67.24%` | `77.59%` | `+0.37%` | `+8.22%` |
| `stop_loss_rebound_loss_reduction` | 反抽缩亏 | `74` | `30.45%` | `-11.11%` | `-11.85%` | `59.46%` | `54.05%` | `36.49%` | `37.84%` | `+4.25%` | `+12.23%` |
| `deep_stop_loss_do_not_wait` | 避免等待 | `97` | `39.92%` | `-11.59%` | `-12.60%` | `55.67%` | `49.48%` | `27.84%` | `35.05%` | `+2.45%` | `+10.41%` |

解读：

1. `time_stop_conservative_wait_profit` 的收盘转正率最高，达到 `67.24%`，但平均收盘改善只有 `+0.37%`。这说明它适合做“浅亏等回本/接近回本”的胜率修复，而不是指望大幅增厚收益。
2. `stop_loss_rebound_loss_reduction` 的可缩亏率最高，达到 `54.05%`，平均收盘改善 `+4.25%`、平均最高价改善 `+12.23%`。这说明止损单确实常有反抽，但平均最高收盘仍为负，目标应是减少亏损，不应直接当作提高胜率规则。
3. `deep_stop_loss_do_not_wait` 虽然也有一定反抽，但收盘可转正率只有 `27.84%`，且原始均亏和 MAE 都最深。它应作为“不等待或只做极短反抽”的风险过滤组。

1. `time_stop` 浅亏等待转正规则：
   - 条件：`exit_reason_time_stop == true`，`return > -0.07`，`max_adverse_excursion > -0.10`，`hold_first_10d_return > -0.04`。
   - 退出：最多等待 5/10/15 日，若收盘价回到成本或接近成本则退出；若跌破新的防守阈值立即退出。
   - 目标：提高胜率，避免明显增加回撤。

2. `stop_loss` 技术反抽缩亏规则：
   - 条件：`exit_reason_stop_loss == true`，`pre_40_above_ma20_ratio` 较高，近期有下影承接。
   - 退出：等短反抽，不要求转正；改善 2pct 或触及短均线即退出。
   - 目标：降低平均亏损和 max drawdown，不以提高胜率为第一目标。

3. 不等待过滤规则：
   - 条件：深亏 stop-loss、`max_adverse_excursion <= -0.11`、前 10 日已经明显走坏、前 40 日 MA20 上方比例低。
   - 行为：保持原止损，不参与等待。
   - 目标：防止等待规则吞掉原策略的风险控制。

验证方式应使用完整回测，而不是只看 post-exit 复盘。复盘证明“事后存在机会”，回测才能证明“规则在当时能捕捉机会且不会扩大亏损”。

### 15.6 两个手写等待规则的真实回测验证

基于 15.5 的事后诊断，先做了两个保守版本的真实回测，用于验证“简单等待/放宽止损”能否直接改善组合表现。

验证配置：

```text
configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_time_stop_wait_profit.yaml
configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_rebound_reduce_loss.yaml
```

回测 run：

```text
runs/20260707_191032_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_time_stop_wait_profit
runs/20260707_191002_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_rebound_reduce_loss
```

组合级结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 平均收益 | 平均持仓天数 |
|------|--------|----------|--------|------|---------------|----------|--------------|
| 基线 `scale_12_keep85` | `45.05` | `-30.67%` | `1.408` | `61.55%` | `2.271` | `4.98%` | `27.95` |
| `stop_loss_rebound_reduce_loss` | `42.82` | `-30.65%` | `1.385` | `62.66%` | `2.170` | `5.27%` | `27.97` |
| `time_stop_wait_profit` | `21.56` | `-30.31%` | `1.159` | `66.30%` | `1.682` | `5.02%` | `28.72` |

结论：两个手写规则都没有通过真实组合验证。

1. `stop_loss_rebound_reduce_loss` 略微提高胜率，但总收益、Sharpe、Profit Factor 都低于基线，最大回撤几乎不变。它没有兑现“缩亏降低回撤”的目标。
2. `time_stop_wait_profit` 把胜率提高到 `66.30%`，但总收益从 `45.05` 降到 `21.56`，Profit Factor 从 `2.271` 降到 `1.682`。这说明它提高的是小赢/回本次数，同时明显伤害了大盈利捕捉和资金效率。
3. 因此，事后看到“可反弹”不等于实盘规则能赚钱。等待规则必须同时约束触发范围、等待期限、退出目标和占仓成本。

卖出原因拆解：

| 策略 | 卖出原因 | 数量 | 胜率 | 平均收益 | 平均持仓 | 总净收益 |
|------|----------|------|------|----------|----------|----------|
| 基线 | `time_stop_25d` | `248` | `60.89%` | `2.42%` | `39.8` | `5,150,325` |
| 基线 | `stop_loss_89permil` | `146` | `0.00%` | `-10.93%` | `20.7` | `-26,564,965` |
| 基线 | `take_profit_210permil` | `77` | `100.00%` | `25.04%` | `22.4` | `56,568,766` |
| `stop_loss_rebound_reduce_loss` | `stop_loss_89permil_weak_structure` | `144` | `0.00%` | `-11.03%` | `21.1` | `-28,278,252` |
| `stop_loss_rebound_reduce_loss` | `stop_loss_rebound_final_125permil` | `1` | `0.00%` | `-14.25%` | `2.0` | `-3,575` |
| `stop_loss_rebound_reduce_loss` | `take_profit_210permil` | `79` | `100.00%` | `24.95%` | `22.7` | `54,994,386` |
| `time_stop_wait_profit` | `time_stop_25d_profit_or_deep_loss` | `241` | `74.69%` | `2.72%` | `41.2` | `6,945,608` |
| `time_stop_wait_profit` | `time_stop_35d_final` | `9` | `0.00%` | `-2.85%` | `51.4` | `-416,595` |
| `time_stop_wait_profit` | `take_profit_210permil` | `76` | `100.00%` | `24.56%` | `22.4` | `32,776,372` |

关键解释：

1. `time_stop_wait_profit` 表面上把 time-stop 相关胜率提高了，但 `take_profit_210permil` 总净收益从基线的约 `56.57M` 降到 `32.78M`。这不是单笔 time-stop 改善能补回来的，说明延迟退出带来了明显的资金占用和路径改变。
2. `stop_loss_rebound_reduce_loss` 几乎没有真正触发最终放宽止损，弱结构止损的平均亏损反而略差，说明当前结构过滤条件没有抓住“值得等反抽”的子集。
3. 目前不能继续沿用“统一等到盈利/深亏再走”的规则。下一轮应让分析模块输出更窄的候选子集，并加入组合级验证指标，例如被等待交易的机会成本、等待期间是否错过新买点、等待后是否减少单笔亏损但降低大盈利交易资金占用。

后续规则方向调整：

1. 对 `time_stop`：不要直接延后到固定 35 日。只允许浅亏、MAE 浅、且当前仍接近成本的单进入极短等待；退出目标应是回本或接近回本，不追求继续持有。
2. 对 `stop_loss`：不要用宽泛的 `zxdq/zxdkx/rsv` 结构过滤。应继续从 `loss_reducible_after_exit` 的特征差异里找更窄子集，优先优化平均亏损，而不是胜率。
3. 对分析模块：需要增加“策略级反事实归因”，把事后 post-exit 机会转成规则时，同时估计占仓成本和错过的后续交易。这是本次失败验证暴露出的主要缺口。

### 15.7 失败单等待诊断：为什么“看起来有反弹”仍然不够

为了避免继续用过粗的等待信号，分析模块新增了 `failure_wait_diagnostics`。它仍然只使用卖点后的 `post_*` 复盘标签，不进入训练特征；用途是拆清楚：

1. 反弹是收盘可执行，还是只有盘中最高价可见。
2. 缩亏和转正分别需要等几天。
3. 5/10 日内能否完成缩亏或转正，避免把 20 日内才出现的慢反弹误当成可用退出规则。

新增分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_failure_wait_diagnostics_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

输出目录：

```text
artifacts/pattern_analysis/weak_best_failure_wait_diagnostics_20260707
```

总体失败单等待诊断：

| 指标 | 数值 | 解读 |
|------|------|------|
| 失败单数量 | `243` | 基线失败交易样本 |
| 平均失败收益 | `-8.03%` | 原始卖点亏损较深 |
| 20 日内收盘可缩亏 2pct | `44.03%` | 存在缩亏空间 |
| 20 日内最高收盘可转正 | `39.09%` | 转正机会明显少于缩亏机会 |
| 20 日内最高价可转正 | `45.27%` | 盘中机会略高于收盘机会 |
| 只有盘中可转正、收盘不可转正 | `6.17%` | 视觉上看到的部分反弹不可稳定执行 |
| 5 日内收盘转正 | `17.28%` | 短等转正覆盖很低 |
| 10 日内收盘转正 | `26.34%` | 等更久仍不到三成 |
| 5 日内收盘缩亏 2pct | `52.26%` | 短等更适合“缩亏”，不是“转正” |
| 10 日内收盘缩亏 2pct | `61.73%` | 缩亏机会比转正更稳定 |
| 平均等到收盘转正天数 | `7.85` | 转正需要较长等待，容易引入占仓成本 |
| 平均等到缩亏 2pct 天数 | `5.06` | 缩亏更快，更适合作为退出管理目标 |
| 平均最佳收盘改善 | `+7.56%` | 最优点有空间，但需要捕捉时机 |
| 平均 20 日末收盘改善 | `+1.74%` | 固定等到窗口末效果很弱 |
| 平均最高价改善 | `+9.20%` | 盘中高点比收盘多约 `1.65pct` |

按退出原因拆解：

| 卖出原因 | 失败数 | 原始均亏 | 可缩亏 | 收盘可转正 | 5 日内转正 | 5 日内缩亏 | 平均转正天数 | 平均缩亏天数 | 平均最佳收盘改善 |
|----------|--------|----------|--------|------------|------------|------------|--------------|--------------|--------------------|
| `stop_loss_89permil` | `146` | `-10.93%` | `50.00%` | `26.71%` | `6.16%` | `56.85%` | `9.56` | `4.96` | `+8.32%` |
| `time_stop_25d` | `97` | `-3.67%` | `35.05%` | `57.73%` | `34.02%` | `45.36%` | `6.66` | `5.22` | `+6.40%` |

这组数字解释了为什么手写等待信号不准：

1. `stop_loss` 的主要价值是快速缩亏，不是等转正。它 5 日内缩亏率 `56.85%`，但 5 日内转正率只有 `6.16%`。如果把目标写成“等盈利再走”，大部分止损单会等太久。
2. `time_stop` 的确更适合等转正，但 5 日内转正也只有 `34.02%`，平均转正要 `6.66` 天。简单延后到 35 日会显著增加占仓和路径改变，所以真实回测里胜率提高但收益坍缩。
3. 20 日内最佳收盘改善远高于 20 日末收盘改善，说明“等固定 N 天”不是好规则。更合理的是等待后用触发式退出，例如回本、接近成本、缩亏 2pct、触及短均线就走。
4. 盘中最高价和最高收盘之间平均差 `1.65pct`，盘中视觉反弹不能直接等价为可执行规则。后续规则应优先用收盘触发或明确盘中成交假设。

候选规则的新诊断：

| 候选规则 | 样本数 | 原始均亏 | 可缩亏 | 收盘可转正 | 最高价可转正 | 20 日末改善 | 平均转正天数 | 平均缩亏天数 |
|----------|--------|----------|--------|------------|--------------|------------|--------------|--------------|
| `time_stop_shallow_wait_profit` | `67` | `-3.09%` | `32.84%` | `61.19%` | `68.66%` | `+0.82%` | `6.24` | `5.00` |
| `time_stop_conservative_wait_profit` | `58` | `-2.14%` | `32.76%` | `67.24%` | `77.59%` | `+0.37%` | `4.62` | `4.68` |
| `stop_loss_rebound_loss_reduction` | `74` | `-11.11%` | `54.05%` | `36.49%` | `37.84%` | `+4.25%` | `9.19` | `4.78` |
| `deep_stop_loss_do_not_wait` | `97` | `-11.59%` | `49.48%` | `27.84%` | `35.05%` | `+2.45%` | `10.41` | `4.76` |

下一步策略研究应改成两个不同目标：

1. `stop_loss` 缩亏规则：最多短等 3-5 日，目标是收盘缩亏 2pct 或触及短均线后退出；如果继续创新低或无法快速缩亏，则保留原止损。这类规则不应要求转正。
2. `time_stop` 胜率修复规则：只筛浅亏且 MAE 浅的单，等待目标是接近成本或回本，不应固定延后到 35 日。需要额外估算占仓成本，因为真实回测已证明胜率提高可能牺牲大盈利交易。
3. 分析模块下一步应补“等待候选的组合级机会成本”：等待期间账户是否满仓、是否错过新的高分买点、等待单最终改善是否足以覆盖被错过交易的收益。这比继续加复杂形态名字更重要。

### 15.8 等待机会成本 proxy：为什么胜率修复会伤资金效率

分析模块新增 `wait_opportunity_cost`，用于近似回答：失败单卖出后如果多等 10 个交易日，原策略在这段时间是否出现了新的买入候选，以及账户是否已经接近满仓。

注意：这是 proxy，不是完整组合反事实回放。它统计原始 run 中卖出后窗口里的 `daily_selection_candidates` 和 `daily_nav.position_count`，不会重放现金复用、排序变化、成交、费用和滑点。

新增分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_failure_wait_opportunity_cost_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

输出目录：

```text
artifacts/pattern_analysis/weak_best_failure_wait_opportunity_cost_20260707
```

总体机会成本 proxy：

| 指标 | 数值 | 解读 |
|------|------|------|
| 失败单数量 | `243` | 与失败单等待诊断一致 |
| 任一等待窗口出现新候选比例 | `87.65%` | 大多数失败单如果继续持有，会遇到新买入机会 |
| 平均候选日数 | `3.82 / 10` | 等待窗口里约 38.62% 交易日有候选 |
| 平均选中候选数 | `6.95` | 不是偶发信号，候选密度较高 |
| 平均 raw 候选数 | `10.37` | 备选机会更多 |
| 平均持仓数量 | `3.54` | 等待期组合已有较高仓位 |
| 账户接近满仓窗口比例 | `60.91%` | 等待常常发生在容量紧张阶段 |
| 平均接近满仓天数 | `4.08 / 10` | 约 41.11% 等待交易日接近满仓 |

按退出原因拆解：

| 卖出原因 | 失败数 | 任一新候选 | 平均候选日数 | 平均选中候选数 | 平均持仓数 | 接近满仓窗口 | 平均接近满仓天数 |
|----------|--------|------------|--------------|----------------|------------|--------------|--------------------|
| `stop_loss_89permil` | `146` | `83.56%` | `3.19` | `5.61` | `3.16` | `50.68%` | `3.19` |
| `time_stop_25d` | `97` | `93.81%` | `4.76` | `8.98` | `4.11` | `76.29%` | `5.41` |

这说明 `time_stop` 等待规则的机会成本尤其高：

1. `time_stop_25d` 失败单之后 10 个交易日内，`93.81%` 会遇到新候选，平均有 `8.98` 个选中候选。
2. 同一窗口里，`76.29%` 接近满仓，平均接近满仓 `5.41` 天。也就是说，延迟卖出很容易占住本来可以给新信号的仓位。
3. 这解释了 `time_stop_wait_profit` 的真实回测现象：胜率从 `61.55%` 提到 `66.30%`，但总收益从 `45.05` 降到 `21.56`。它修复了一些小亏/浅亏交易，却破坏了资金周转和大盈利捕捉。
4. `stop_loss` 的机会成本也存在，但相对低于 `time_stop`。它更适合做短窗口缩亏：最多 3-5 日，等到缩亏触发就走，不能等到长期占仓。

下一轮规则验证必须同时满足两个条件：

1. 单笔层面：等待后能快速缩亏或回本，不能只看 20 日内最高点。
2. 组合层面：等待窗口内新候选密度和接近满仓程度不能太高，或者等待规则必须释放部分仓位，避免吞掉更强的新机会。

因此，更合理的方向不是“失败单统一等待”，而是：

```text
stop_loss：短等缩亏，不追求转正，不超过 3-5 日。
time_stop：只对极浅亏、低 MAE、低机会成本窗口做回本等待；否则按原规则释放仓位。
```

### 15.9 部分减仓等待规则的真实回测验证

在 15.7/15.8 之后，又验证了两个更保守的规则：触发等待时先释放大部分仓位，只保留残仓等缩亏或回本，目的是降低 full waiting 的机会成本。

验证配置：

```text
configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_partial_rebound.yaml
configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_time_stop_partial_wait.yaml
```

回测 run：

```text
runs/20260707_193833_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_partial_rebound
runs/20260707_193815_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_time_stop_partial_wait
```

组合级结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 交易数 | 平均持仓天数 |
|------|--------|----------|--------|------|---------------|--------|--------------|
| 基线 `scale_12_keep85` | `45.05` | `-30.67%` | `1.408` | `61.55%` | `2.271` | `1120` | `27.95` |
| `stop_loss_partial_rebound` | `9.53` | `-38.18%` | `0.841` | `48.93%` | `1.457` | `1145` | `29.42` |
| `time_stop_partial_wait` | `37.05` | `-30.43%` | `1.370` | `60.61%` | `2.042` | `1172` | `29.45` |

卖出原因拆解：

| 策略 | 卖出原因 | 数量 | 胜率 | 平均收益 | 平均持仓 | 总净收益 |
|------|----------|------|------|----------|----------|----------|
| 基线 | `stop_loss_89permil` | `146` | `0.00%` | `-10.93%` | `20.7` | `-26,564,965` |
| 基线 | `time_stop_25d` | `248` | `60.89%` | `2.42%` | `39.8` | `5,150,325` |
| 基线 | `take_profit_210permil` | `77` | `100.00%` | `25.04%` | `22.4` | `56,568,766` |
| `stop_loss_partial_rebound` | `stop_loss_89permil_reduce_to_30pct` | `137` | `0.00%` | `-10.88%` | `19.5` | `-11,641,447` |
| `stop_loss_partial_rebound` | `stop_loss_rebound_reduce_loss_exit` | `6` | `0.00%` | `-4.54%` | `19.0` | `-21,379` |
| `stop_loss_partial_rebound` | `stop_loss_residual_deep_loss_exit` | `12` | `0.00%` | `-13.39%` | `24.8` | `-606,692` |
| `stop_loss_partial_rebound` | `time_stop_25d` | `349` | `41.83%` | `-2.08%` | `39.5` | `-273,449` |
| `time_stop_partial_wait` | `stop_loss_89permil` | `146` | `0.00%` | `-10.89%` | `20.8` | `-26,699,004` |
| `time_stop_partial_wait` | `time_stop_25d_profit_or_bad_loss` | `240` | `74.58%` | `2.70%` | `41.6` | `6,785,124` |
| `time_stop_partial_wait` | `time_stop_25d_shallow_loss_reduce_to_35pct` | `64` | `0.00%` | `-2.65%` | `39.2` | `-1,425,218` |

结论：部分减仓等待也没有通过组合级验证。

1. `stop_loss_partial_rebound` 明显失败，总收益从 `45.05` 降到 `9.53`，最大回撤扩大到 `-38.18%`。虽然止损主仓释放后单项亏损金额变小，但它改变了后续路径，`time_stop_25d` 数量从 `248` 增到 `349` 且转为负贡献，说明等待信号没有筛准，反而引入了更多低质量持仓路径。
2. `time_stop_partial_wait` 接近但仍低于基线，总收益 `37.05`、Profit Factor `2.042`。它提高了部分 time-stop 胜率，但新增的浅亏残仓退出贡献为负，无法覆盖资金效率和路径变化损失。
3. 这轮验证强化了一个判断：等待规则的问题不是“仓位留多少”这么简单，而是等待触发信号本身不准。下一轮必须先用分析模块找到更窄的可缩亏失败单画像，再做规则搜索。

### 15.10 可缩亏失败单画像：等待信号应从这里重新构建

为避免继续手写粗糙等待信号，分析模块新增 `failure_recovery_profiles`。它面向失败交易，按不同 recovery target 输出正/负样本画像、按卖出原因拆解，以及 top 特征差异。

新增分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_failure_recovery_profiles_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

输出目录：

```text
artifacts/pattern_analysis/weak_best_failure_recovery_profiles_20260707
```

总体画像：

| 目标 | 正样本数 | 正样本率 | 正样本原始均亏 | 负样本原始均亏 | 正样本 MAE | 负样本 MAE | 正样本最佳收盘改善 |
|------|----------|----------|----------------|----------------|------------|------------|--------------------|
| `loss_reducible_after_exit` 缩亏型 | `107 / 243` | `44.03%` | `-8.67%` | `-7.53%` | `-10.35%` | `-9.45%` | `+13.64%` |
| `close_turn_profitable_after_exit` 收盘可回本型 | `95 / 243` | `39.09%` | `-6.21%` | `-9.20%` | `-8.91%` | `-10.45%` | `+14.65%` |
| `high_turn_profitable_after_exit` 盘中可回本型 | `110 / 243` | `45.27%` | `-6.24%` | `-9.51%` | `-8.85%` | `-10.67%` | `+13.40%` |

按卖出原因拆解：

| 目标 | 卖出原因 | 样本数 | 正样本数 | 正样本率 | 正样本原始均亏 | 负样本原始均亏 | 正样本最佳收盘改善 |
|------|----------|--------|----------|----------|----------------|----------------|--------------------|
| 缩亏型 | `stop_loss_89permil` | `146` | `73` | `50.00%` | `-10.89%` | `-10.97%` | `+13.47%` |
| 缩亏型 | `time_stop_25d` | `97` | `34` | `35.05%` | `-3.91%` | `-3.54%` | `+14.00%` |
| 收盘可回本型 | `time_stop_25d` | `97` | `56` | `57.73%` | `-2.92%` | `-4.69%` | `+10.23%` |
| 收盘可回本型 | `stop_loss_89permil` | `146` | `39` | `26.71%` | `-10.93%` | `-10.93%` | `+20.99%` |
| 盘中可回本型 | `time_stop_25d` | `97` | `62` | `63.92%` | `-2.79%` | `-5.21%` | `+9.35%` |
| 盘中可回本型 | `stop_loss_89permil` | `146` | `48` | `32.88%` | `-10.69%` | `-11.05%` | `+18.62%` |

缩亏型失败单的 top 特征差异：

| 特征 | 缩亏型均值 | 不可缩亏均值 | 标准化差异 | 解读 |
|------|------------|--------------|------------|------|
| `pre_40_above_ma20_ratio` | `0.7173` | `0.6679` | `+0.302` | 更长期仍保持在 MA20 上方，结构没有完全坏掉 |
| `pre_5_return` | `0.59%` | `1.27%` | `-0.301` | 买点前短线冲得没那么急，可能少一些加速透支 |
| `exit_reason_stop_loss` | `0.6822` | `0.5368` | `+0.297` | 缩亏机会更多来自 stop-loss，而不是 time-stop |
| `return` | `-8.67%` | `-7.53%` | `-0.266` | 可缩亏单原始亏损不一定更浅，不能简单用亏损小筛选 |
| `max_adverse_excursion` | `-10.35%` | `-9.45%` | `-0.263` | 同上，单纯 MAE 浅不是缩亏型的核心条件 |
| `pre_10_long_lower_count` | `1.9626` | `1.6471` | `+0.262` | 近期下影线更多，可能有承接/反抽基础 |
| `pre_10_return` | `4.73%` | `5.81%` | `-0.247` | 买点前 10 日涨幅略低，避免过度追高 |
| `pre_20_max_drawdown` | `-4.73%` | `-5.40%` | `+0.237` | 中短期回撤略浅，结构更稳 |

收盘可回本型的 top 特征差异更清楚：

| 特征 | 可回本均值 | 不可回本均值 | 标准化差异 | 解读 |
|------|------------|--------------|------------|------|
| `return` | `-6.21%` | `-9.20%` | `+0.697` | 回本等待首先要求卖点亏损不能太深 |
| `exit_reason_time_stop` | `0.5895` | `0.2770` | `+0.638` | time-stop 更适合等回本 |
| `exit_reason_stop_loss` | `0.4105` | `0.7230` | `-0.638` | stop-loss 更不适合追求转正 |
| `max_adverse_excursion` | `-8.91%` | `-10.45%` | `+0.448` | 回本型 MAE 更浅 |
| `pre_40_above_ma20_ratio` | `0.7239` | `0.6677` | `+0.344` | 长期结构更好 |
| `hold_first_10d_return` | `-2.21%` | `-4.02%` | `+0.333` | 持仓前 10 日没有快速走坏 |
| `max_favorable_excursion` | `6.17%` | `4.86%` | `+0.319` | 持仓期曾经有更好的上冲能力 |
| `hold_first_3d_return` | `-0.13%` | `-1.27%` | `+0.318` | 入场后早期表现不能太弱 |

策略含义：

1. `stop_loss` 的目标应是“短等缩亏”，不是“等回本”。缩亏型 stop-loss 覆盖 `50.00%`，但收盘可回本只有 `26.71%`。
2. `time_stop` 可以研究“浅亏回本等待”，但必须叠加低机会成本过滤。它可回本率 `57.73%`，同时 10 日机会成本 proxy 最高。
3. 可缩亏失败单不是简单的“亏得浅”。缩亏型的原始亏损和 MAE 反而略深，真正更像是：长期结构仍好、买点前没有过度加速、近期有下影承接、持仓早期没有明显崩坏。
4. 下一轮规则搜索应先做窄条件组合，而不是继续手写单一等待信号。候选方向可以从这些条件组合开始：

```text
stop_loss 缩亏候选：
- exit_reason_stop_loss
- pre_40_above_ma20_ratio 较高
- pre_10_long_lower_count 较高
- pre_5/pre_10_return 不过热
- 等待目标只设为缩亏 2pct 或 3-5 日内触发退出

time_stop 回本候选：
- exit_reason_time_stop
- return / max_adverse_excursion 较浅
- hold_first_3d_return、hold_first_10d_return 不弱
- pre_40_above_ma20_ratio 较高
- 机会成本 proxy 低时才允许等待
```

注意：本次 `--feature-scope trade_management` 会使用卖点时已知信息，因此模型里 `success/opportunity/fade/efficient_capture` 的高 AUC 只能解释为“交易管理诊断有效”，不能反推为“买点前可预测”。真正用于买点过滤时必须回到 `--feature-scope entry` 或单独限制特征集合。

### 15.11 可解释规则搜索：用真实失败单寻找更窄等待信号

在 15.10 的画像之后，分析模块继续新增 `failure_recovery_rule_search`。它不是训练模型，也不是直接生成策略，而是在失败单里枚举少量卖点时已知的可解释条件组合，然后用 post-exit 标签评分。

测试口径必须分开：

1. 单元测试只使用合成数据验证规则搜索模块能跑通、能写入 summary/HTML，不用于证明策略有效。
2. 策略研究结论只来自真实 run artifact。本节数字来自真实弱转强基准 run。

真实分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_failure_rule_search_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

输出目录：

```text
artifacts/pattern_analysis/weak_best_failure_rule_search_20260707
```

真实样本规模：

| 项目 | 数值 |
|------|------|
| 总交易样本 | `632` |
| 失败单样本 | `243` |
| `loss_reducible_after_exit` 基准率 | `44.03%` |
| `close_turn_profitable_after_exit` 基准率 | `39.09%` |

`loss_reducible_after_exit` 的 top 候选规则：

| 条件组合 | 样本数 | 命中数 | 命中率 | Lift | 原始均亏 | 最佳收盘改善 | 平均缩亏天数 | 卖出原因 |
|----------|--------|--------|--------|------|----------|--------------|--------------|----------|
| `exit_reason_stop_loss` + `pre_5_return <= 0` + `hold_first_10d_return >= -4%` | `18` | `14` | `77.78%` | `+33.74pct` | `-10.90%` | `+12.13%` | `4.11` | `stop_loss_89permil` |
| `exit_reason_stop_loss` + `pre_20_above_ma20_ratio >= 1.0` + `hold_first_10d_return >= -4%` | `20` | `15` | `75.00%` | `+30.97pct` | `-11.83%` | `+7.83%` | `4.17` | `stop_loss_89permil` |
| `exit_reason_stop_loss` + `pre_20_above_ma20_ratio >= 1.0` + `hold_first_3d_return >= -1%` | `26` | `19` | `73.08%` | `+29.04pct` | `-11.12%` | `+9.91%` | `4.13` | `stop_loss_89permil` |
| `exit_reason_stop_loss` + `pre_20_above_ma20_ratio >= 1.0` + `hold_first_3d_return >= 0` | `22` | `16` | `72.73%` | `+28.69pct` | `-11.08%` | `+10.39%` | `3.80` | `stop_loss_89permil` |
| `exit_reason_stop_loss` + `pre_10_return <= 4%` + `hold_first_3d_return >= 0` | `18` | `13` | `72.22%` | `+28.19pct` | `-10.91%` | `+16.51%` | `3.65` | `stop_loss_89permil` |

这组候选强化了 15.10 的结论：`stop_loss` 不应该等回本，而应该研究“结构尚好时的短等缩亏”。真实规则搜索里高命中子集共同特征是：

1. 退出原因必须是 `stop_loss_89permil`。
2. 买点前没有明显过热，例如 `pre_5_return <= 0` 或 `pre_10_return <= 4%`。
3. 持仓早期没有快速崩坏，例如 `hold_first_3d_return >= -1%/0` 或 `hold_first_10d_return >= -4%`。
4. 中短期结构较强，例如 `pre_20_above_ma20_ratio >= 1.0`。
5. 这些组合的平均缩亏触发在 `3.65-4.17` 天附近，支持“最多短等 3-5 日”的方向。

`close_turn_profitable_after_exit` 的 top 候选规则：

| 条件组合 | 样本数 | 命中数 | 命中率 | Lift | 原始均亏 | 最佳收盘改善 | 平均缩亏天数 | 卖出原因 |
|----------|--------|--------|--------|------|----------|--------------|--------------|----------|
| `exit_reason_time_stop` + `pre_20_above_ma20_ratio >= 1.0` + `hold_first_3d_return >= -1%` | `18` | `16` | `88.89%` | `+49.79pct` | `-3.77%` | `+7.63%` | `5.25` | `time_stop_25d` |
| `exit_reason_time_stop` + `pre_20_above_ma20_ratio >= 1.0` + `hold_first_3d_return >= 0` | `12` | `10` | `83.33%` | `+44.24pct` | `-4.22%` | `+8.12%` | `6.09` | `time_stop_25d` |
| `exit_reason_time_stop` + `pre_20_above_ma20_ratio >= 1.0` + `return >= -5%` | `17` | `14` | `82.35%` | `+43.26pct` | `-1.79%` | `+5.78%` | `4.23` | `time_stop_25d` |
| `exit_reason_time_stop` + `pre_20_above_ma20_ratio >= 1.0` + `max_adverse_excursion >= -9%` | `19` | `15` | `78.95%` | `+39.85pct` | `-2.82%` | `+6.18%` | `5.73` | `time_stop_25d` |
| `exit_reason_time_stop` + `pre_10_long_lower_count >= 3` + `return >= -5%` | `18` | `14` | `77.78%` | `+38.68pct` | `-1.87%` | `+5.44%` | `5.00` | `time_stop_25d` |

这组候选说明：`time_stop` 的回本等待确实有更清楚的可解释子集，但它必须和机会成本 proxy 一起使用。尤其是 `return >= -5%`、`max_adverse_excursion >= -9%`、`pre_20_above_ma20_ratio >= 1.0` 这类条件，只能说明单笔有回本机会，不能说明组合级值得等。

下一轮真实回测候选应从两条线开始：

1. `stop_loss` 短等缩亏：只对高结构强度、不过热、持仓早期不弱的止损单，等待 3-5 日，触发缩亏 2pct 或到期就走，不追求转正。
2. `time_stop` 浅亏回本：只对浅亏、MAE 浅、结构强的 time-stop 单等待回本，同时加入机会成本过滤或强制释放大部分仓位。

仍需注意：本节规则搜索只是在真实失败单上做 post-exit 标签诊断，下一步必须转成 YAML 条件并跑完整组合回测。只有真实组合回测改善总收益、回撤和 Profit Factor，才能算策略改进成立。

### 15.12 入场/早期持仓画像特征接入真实回测

15.11 的规则搜索暴露了一个工程缺口：分析模块里的 `pre_*` 和 `hold_first_*` 是交易窗口特征，不能直接用当前日 runtime 因子近似。为避免继续用不准确 proxy，本轮在回测引擎里新增了持仓上下文特征，使分析画像可以被 YAML 卖出规则精确引用。

新增能力：

1. `entry_<factor>`：买入订单成交后，将买入日可见的同名行情/公式因子固化到持仓里。例如公式里定义 `pre_20_above_ma20_ratio` 后，卖出规则可写 `entry_pre_20_above_ma20_ratio`。
2. `hold_first_3d_return` / `hold_first_5d_return` / `hold_first_10d_return`：账户在日终更新时，按第 3/5/10 个持仓收盘价相对成本价固化早期持仓收益。
3. 卖出规则 evaluator 会自动把这些 position context 注入规则上下文。配置校验要求 `entry_<factor>` 的基础 `<factor>` 必须是已有字段或公式，避免拼错字段静默变成无效规则。

相关实现点：

| 文件 | 作用 |
|------|------|
| `quantx/core/engine/types.py` | `Order` / `Position` 增加 `context`，`Position` 保留 `initial_quantity` |
| `quantx/core/engine/account.py` | 买入时写入 entry context，日终固化 `hold_first_*_return` |
| `quantx/core/engine/executor.py` | 将 buy order context 传入账户 |
| `quantx/core/engine/engine.py` | 将 position context 传入 `PolicyState` |
| `quantx/core/strategy/config_strategy.py` | 校验/采集/注入 `entry_*` 和 `hold_first_*` 上下文字段 |

单元测试覆盖：

```bash
conda run -n test python -m pytest tests/strategy/test_config_strategy.py tests/engine/test_account_costs.py
```

验证结果：`43 passed`。

覆盖点包括：

1. `compile_strategy_config` 接受合法的 `entry_pre_5_return` / `hold_first_3d_return` 卖出规则。
2. 未定义基础因子的 `entry_missing_factor` 会被配置校验拒绝。
3. `RuleExecution` 生成买单时会写入 `entry_pre_5_return` 快照。
4. 卖出规则能用 `entry_pre_*` 与 `hold_first_*` 触发退出。
5. `Account` 会在第 3/5/10 个持仓日固化早期收益。

#### 15.12.1 精确画像 stop-loss 候选回测

新配置：

```text
configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_exact.yaml
```

核心新增公式：

```yaml
ma20: Mean(close, 20)
pre_5_return: close / (Ref(close, 5) + 1e-10) - 1
pre_20_above_ma20_ratio: Sum(close > ma20, 20) / 20
```

核心卖出规则：

```yaml
- name: stop_loss_context_reduce_to_30pct
  when: pnl_pct < -0.089 and remaining_position_pct > 0.30 and entry_pre_20_above_ma20_ratio >= 1.0 and hold_first_3d_return >= -0.01
  action: sell_to_position_pct
  position_pct: 0.30
- name: stop_loss_89permil_other_full_exit
  when: pnl_pct < -0.089 and remaining_position_pct > 0.50
  action: sell_all
- name: stop_loss_context_rebound_reduce_loss_exit
  when: remaining_position_pct < 0.50 and pnl_pct > -0.065
  action: sell_all
- name: stop_loss_context_residual_deep_loss_exit
  when: remaining_position_pct < 0.50 and pnl_pct < -0.120
  action: sell_all
- name: stop_loss_context_residual_time_exit
  when: remaining_position_pct < 0.50 and holding_days > 30
  action: sell_all
```

第一次回测曾使用 `remaining_position_pct <= 0.30` 判断残仓，结果受 100 股整手取整影响，残仓比例经常略高于 `0.30`，随后被普通止损规则立即扫出。这一版已改为 `remaining_position_pct < 0.50` 识别残仓。

dry-run：

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_exact.yaml \
  --dry-run \
  --json
```

结果：`ok=true`。

完整回测 run：

```text
runs/20260707_202153_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_exact
```

组合级结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 交易数 |
|------|--------|----------|--------|------|---------------|--------|
| 基线 `scale_12_keep85` | `45.05` | `-30.67%` | `1.408` | `61.55%` | `2.271` | `1120` |
| runtime proxy stop-loss | `44.66` | `-30.67%` | `1.405` | `61.73%` | `2.265` | `1123` |
| context exact stop-loss | `43.65` | `-31.67%` | `1.387` | `58.97%` | `2.218` | `1150` |

卖出原因拆解：

| 卖出原因 | 数量 | 胜率 | 平均收益 | 总净收益 | 说明 |
|----------|------|------|----------|----------|------|
| `stop_loss_89permil_other_full_exit` | `117` | `0.00%` | `-10.74%` | `-23,892,744` | 未命中画像的普通止损 |
| `stop_loss_context_reduce_to_30pct` | `29` | `0.00%` | `-11.27%` | `-2,091,188` | 命中画像后主仓减到约 30% |
| `stop_loss_context_rebound_reduce_loss_exit` | `10` | `10.00%` | `-4.14%` | `-182,828` | 残仓缩亏退出，确实比原止损浅 |
| `stop_loss_context_residual_deep_loss_exit` | `13` | `0.00%` | `-14.36%` | `-409,431` | 残仓继续恶化 |
| `time_stop_25d` | `253` | `58.89%` | `2.24%` | `5,600,211` | 比基线略弱 |

画像止损残仓后续去向：

| 残仓退出 | 数量 | 从部分减仓价算的平均变化 | 中位数 |
|----------|------|--------------------------|--------|
| `stop_loss_context_rebound_reduce_loss_exit` | `10` | `+7.50%` | `+6.22%` |
| `stop_loss_context_residual_deep_loss_exit` | `13` | `-2.96%` | `-3.10%` |
| `time_stop_25d` | `6` | `+1.29%` | `+1.21%` |

#### 15.12.2 当前结论

这轮验证很关键：它证明了分析画像特征已经可以被真实回测精确执行，但当前 stop-loss 等待规则仍没有形成组合级改进。

结论如下：

1. `entry_pre_20_above_ma20_ratio >= 1.0` + `hold_first_3d_return >= -1%` 能筛出一部分确实会反弹的残仓，`10/29` 笔触发缩亏退出，残仓从部分减仓价平均反弹 `+7.50%`。
2. 但同一画像也筛出了更多继续恶化的残仓，`13/29` 笔触发深亏退出，残仓从部分减仓价平均再跌 `-2.96%`。
3. 主仓先减到 30% 会降低部分单笔暴露，但也改变组合路径；最终总收益、胜率、Profit Factor 均低于基线，最大回撤还略扩大。
4. 因此，“画像字段精确落地”这条工程方向成立；但 `pre_20_above_ma20_ratio + hold_first_3d_return` 这组等待信号仍不够准，还不能作为策略改进。

下一步不应继续扩大等待，而应继续收窄画像：

1. 加入 `entry_pre_5_return <= 0` 或 `entry_pre_10_return <= 4%`，优先验证 15.11 中 lift 更高的组合。
2. 加入 `hold_first_10d_return >= -4%`，但要注意它只有持仓第 10 天后才可用。
3. 研究残仓退出阈值：当前 `pnl_pct > -6.5%` 缩亏退出偏宽，可能需要比较 `-7.5%/-7.0%/-6.5%` 与最大等待天数 `3/5/8`。
4. 把 opportunity cost 放入规则搜索或回测配置，例如等待期间候选密度高时直接释放仓位。

阶段判断：弱转强的失败单里确有可缩亏子集，但目前最直接的 stop-loss 画像规则仍无法提升总收益/回撤/Profit Factor。后续优化需要更窄规则搜索，而不是再用粗等待。

### 15.13 更窄 stop-loss 画像规则回测

15.12 证明了 `entry_*` 与 `hold_first_*` 可以真实落地，但第一组 `entry_pre_20_above_ma20_ratio >= 1.0 + hold_first_3d_return >= -1%` 没有赢基线。本轮继续把 15.11 里 lift 更高的 stop-loss 组合翻译成 YAML 精确回测。

新增候选配置：

| 候选 | 配置 | 画像触发条件 |
|------|------|--------------|
| `pre5_hold10` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre5_hold10.yaml` | `entry_pre_5_return <= 0` + `hold_first_10d_return >= -4%` |
| `pre20_hold10` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre20_hold10.yaml` | `entry_pre_20_above_ma20_ratio >= 1.0` + `hold_first_10d_return >= -4%` |
| `pre10_hold3` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre10_hold3.yaml` | `entry_pre_10_return <= 4%` + `hold_first_3d_return >= 0` |

三条候选均通过 dry-run，公式数 `43`，新增 `pre_10_return`：

```yaml
pre_10_return: close / (Ref(close, 10) + 1e-10) - 1
```

完整回测 run：

```text
runs/20260707_202718_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre5_hold10
runs/20260707_202831_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre20_hold10
runs/20260707_202935_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_context_pre10_hold3
```

组合级结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 交易数 |
|------|--------|----------|--------|------|---------------|--------|
| 基线 `scale_12_keep85` | `45.05` | `-30.67%` | `1.408` | `61.55%` | `2.271` | `1120` |
| `pre20_hold3` | `43.65` | `-31.67%` | `1.387` | `58.97%` | `2.218` | `1150` |
| `pre5_hold10` | `31.50` | `-30.62%` | `1.259` | `59.04%` | `2.008` | `1157` |
| `pre20_hold10` | `39.44` | `-30.30%` | `1.348` | `59.06%` | `2.126` | `1146` |
| `pre10_hold3` | `43.85` | `-30.65%` | `1.393` | `60.12%` | `2.268` | `1138` |

`pre10_hold3` 最接近基线，Profit Factor 几乎持平，但总收益、Sharpe、胜率仍低于基线。`pre5_hold10` 明显失败，说明虽然离线规则搜索里 `pre_5_return <= 0 + hold_first_10d_return >= -4%` 命中率高，但真实组合路径下等待残仓的收益损耗很大。

残仓后续表现：

| 候选 | 画像减仓数 | 缩亏退出数 | 深亏退出数 | time-stop 残仓数 | 缩亏残仓从减仓价平均变化 | 深亏残仓从减仓价平均变化 |
|------|------------|------------|------------|------------------|----------------------------|----------------------------|
| `pre20_hold3` | `29` | `10` | `13` | `6` | `+7.50%` | `-2.96%` |
| `pre5_hold10` | `25` | `9` | `8` | `8` | `+5.60%` | `-0.94%` |
| `pre20_hold10` | `23` | `7` | `11` | `5` | `+8.19%` | `-2.66%` |
| `pre10_hold3` | `18` | `8` | `8` | `2` | `+5.91%` | `-2.52%` |

这张表说明：残仓等待确实能抓到一些缩亏反弹，但同时也保留了一批继续恶化的尾部。即使主仓已经先卖到 30%，组合层面的总收益和胜率仍没有改善。

本轮结论：

1. 更窄画像并没有解决核心问题。`pre10_hold3` 已经相当接近基线，但没有超过基线，说明继续在 stop-loss 后“留残仓等反弹”的空间有限。
2. 离线 `loss_reducible_after_exit` 标签只说明未来 20 日内出现过缩亏机会，不等于组合里值得保留仓位。真实回测会受到仓位占用、路径变化、整手取整和后续新机会影响。
3. 当前方向不应继续扩大等待，而应转向两类验证：
   - 把画像用于“不同止损处理”，例如画像差的 stop-loss 提前全出或更快全出，而不是画像好的单留残仓。
   - 在规则搜索里加入机会成本标签，例如等待窗口内是否出现高分新候选、是否接近满仓、等待期间组合是否需要资金。
4. 如果继续做残仓等待，应只做参数化小网格：残仓比例 `10%/20%/30%`，缩亏退出阈值 `-7.5%/-7.0%/-6.5%`，最大等待 `3/5/8` 天，并以组合级 `total_return + max_drawdown + profit_factor` 排序，而不是只看单笔缩亏命中率。

阶段判断更新：分析模块已经能找到“看起来会缩亏”的失败单画像，引擎也能精确回测这些画像；但目前 stop-loss 残仓等待不是提升弱转强策略的有效突破口。下一步更应该分析“哪些亏损单应该更快卖/更少留仓”，以及把机会成本直接纳入画像标签。

### 15.14 机会成本画像：哪些亏损单不该等

15.13 之后，分析模块继续补了一层 `failure_wait_cost_profiles`。它不是新的交易规则，而是把三类信息合并起来做画像：

1. 卖出时已经知道的交易管理特征，例如 `return`、`max_adverse_excursion`、`hold_first_3d_return`、`hold_first_10d_return`、`pre_*` 结构特征、`exit_reason_*`。
2. 未来 20 日恢复标签，例如 `loss_reducible_after_exit`、`close_turn_profitable_after_exit`、`recoverable_failure`。
3. 原始组合在卖出后等待窗口内的机会成本代理，例如是否有新候选、候选天数、候选数量、是否接近满仓。

机会成本代理的定义仍是诊断用途：它统计原始 run 在失败单卖出后的 10 个交易日内，是否出现新的 selection candidate，以及账户是否接近满仓。它不重放现金复用、排序、成交和组合路径，所以不能直接当成收益结论。

新增真实分析产物：

```text
artifacts/pattern_analysis/weak_best_wait_cost_profiles_20260707
```

分析命令：

```bash
conda run -n test python -m quantx.tools.analyze_trade_patterns \
  --runs runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_wait_cost_profiles_20260707 \
  --pre-n 40 \
  --post-n 20 \
  --models logistic_regression random_forest \
  --feature-scope trade_management \
  --no-clusters \
  --json
```

#### 15.14.1 失败单恢复与机会成本概览

基线 run 共分析 `632` 笔闭合交易，其中失败单 `243` 笔。

失败单恢复标签：

| 指标 | 数值 |
|------|------|
| 失败单平均收益 | `-8.03%` |
| 可恢复失败单比例 | `58.44%` |
| 卖出后可缩亏比例 | `44.03%` |
| 卖出后收盘可回本比例 | `39.09%` |
| 卖出后盘中可回本比例 | `45.27%` |

按卖出原因拆解：

| 卖出原因 | 失败单数 | 平均收益 | 可缩亏 | 收盘可回本 | 盘中可回本 | 平均收盘回本天数 | 平均缩亏天数 |
|----------|----------|----------|--------|------------|------------|------------------|--------------|
| `stop_loss_89permil` | `146` | `-10.93%` | `50.00%` | `26.71%` | `32.88%` | `9.56` | `4.96` |
| `time_stop_25d` | `97` | `-3.67%` | `35.05%` | `57.73%` | `63.92%` | `6.66` | `5.22` |

等待窗口机会成本代理：

| 分组 | 任意候选出现率 | 平均候选天数 | 平均入选候选数 | 接近满仓窗口比例 | 平均接近满仓天数 |
|------|----------------|--------------|----------------|------------------|------------------|
| 全部失败单 | `87.65%` | `3.82` | `6.95` | `60.91%` | `4.08` |
| `stop_loss_89permil` | `83.56%` | `3.19` | `5.61` | `50.68%` | `3.19` |
| `time_stop_25d` | `93.81%` | `4.76` | `8.98` | `76.29%` | `5.41` |

这个结果解释了为什么“多等一等”在组合层面容易失效：很多失败单后面确实有反弹，但同时组合也经常有新机会，尤其 `time_stop_25d` 后等待的机会成本更高。

#### 15.14.2 高机会成本且不值得等

本轮先定义一个粗粒度的高机会成本标记：

```text
candidate_days >= 3
or selected_candidate_count >= 5
or near_full_days >= 3
```

在 `243` 笔失败单中，高机会成本失败单 `168` 笔，占 `69.14%`。

关键目标 `high_cost_not_worth_waiting` 定义为：高机会成本，且卖出后既不能缩亏，也不能收盘回本。真实结果：

| 分组 | 样本数 | 命中数 | 命中率 | 命中组平均收益 | 非命中组平均收益 | 命中组平均候选天数 | 命中组平均入选候选数 | 命中组平均满仓天数 |
|------|--------|--------|--------|----------------|------------------|--------------------|----------------------|--------------------|
| 全部失败单 | `243` | `80` | `32.92%` | `-8.30%` | `-7.90%` | `5.06` | `9.69` | `5.78` |
| `stop_loss_89permil` | `146` | `47` | `32.19%` | `-10.97%` | `-10.91%` | `4.72` | `9.43` | `5.70` |
| `time_stop_25d` | `97` | `33` | `34.02%` | `-4.50%` | `-3.24%` | `5.55` | `10.06` | `5.88` |

这类单的含义很明确：卖出后占用等待窗口时，新机会和满仓压力都高，但未来并没有提供缩亏或收盘回本。它们是后续“更快卖 / 不留残仓 / 不等待”的第一优先候选，而不是继续找反弹。

`high_cost_not_worth_waiting` 的前几项特征差异：

| 特征 | 命中组均值 | 非命中组均值 | 差异 | 标准化差异 | 解读 |
|------|------------|--------------|------|------------|------|
| `pre_10_return` | `6.41%` | `4.80%` | `+1.60%` | `0.366` | 买前 10 日涨幅更高，可能更偏短线冲高后衰竭 |
| `hold_first_3d_return` | `-1.67%` | `-0.41%` | `-1.26%` | `-0.353` | 入场后前三天更弱 |
| `entry_volume_ratio_20` | `0.904` | `1.011` | `-0.107` | `-0.272` | 入场量能相对不足 |
| `pre_20_long_upper_count` | `4.51` | `5.01` | `-0.49` | `-0.256` | 该差异较弱，暂不单独使用 |
| `entry_body_pct` | `1.46%` | `1.87%` | `-0.41%` | `-0.249` | 入场实体更小，确认力度不足 |

另一个更干净的目标是 `high_cost_not_close_profitable`：高机会成本，且卖出后收盘不能回本。命中 `103/243`，命中率 `42.39%`。它的特征差异更清晰：

| 特征 | 命中组均值 | 非命中组均值 | 差异 | 标准化差异 |
|------|------------|--------------|------|------------|
| `hold_first_3d_return` | `-1.89%` | `-0.04%` | `-1.85%` | `-0.518` |
| `max_favorable_excursion` | `4.37%` | `6.11%` | `-1.74%` | `-0.424` |
| `hold_first_10d_return` | `-4.53%` | `-2.41%` | `-2.12%` | `-0.389` |
| `entry_volume_ratio_20` | `0.887` | `1.041` | `-0.154` | `-0.392` |

这组画像更适合翻译成真实卖出上下文规则：入场后前三天弱、前十天弱、持仓期间最高浮盈不足、入场量能弱，同时等待窗口机会成本高。由于真实策略在交易当天无法知道未来机会成本，所以最终规则不能直接使用 `candidate_days`，但可以用这些卖出时已知的代理特征来设计“不要等”的候选。

#### 15.14.3 低机会成本且可恢复

反方向目标 `low_cost_recoverable` 也有意义：低机会成本，且未来可恢复/可缩亏/可回本。命中 `46/243`，命中率 `18.93%`。

| 分组 | 命中数 | 命中率 | 命中组平均收益 | 平均候选天数 | 平均入选候选数 | 可缩亏率 | 收盘可回本率 |
|------|--------|--------|----------------|--------------|----------------|----------|--------------|
| 全部失败单 | `46` | `18.93%` | `-9.34%` | `0.67` | `0.80` | `89.13%` | `65.22%` |
| `stop_loss_89permil` | `37` | `25.34%` | `-11.00%` | `0.76` | `0.92` | `97.30%` | `56.76%` |
| `time_stop_25d` | `9` | `9.28%` | `-2.54%` | `0.33` | `0.33` | `55.56%` | `100.00%` |

它的特征更像“可以考虑等待”的样子：

| 特征 | 命中组均值 | 非命中组均值 | 标准化差异 | 解读 |
|------|------------|--------------|------------|------|
| `pre_40_max_drawdown` | `-7.14%` | `-10.13%` | `0.569` | 买前结构回撤更浅 |
| `exit_reason_stop_loss` | `80.43%` | `55.33%` | `0.513` | 主要是 stop-loss 后的低成本恢复，而不是 time-stop |
| `hold_first_3d_return` | `0.64%` | `-1.16%` | `0.504` | 入场前三天不弱 |
| `pre_40_volatility` | `2.02%` | `2.40%` | `-0.448` | 前期波动更低 |

但这不是马上扩大等待的理由。因为 `low_cost_recoverable` 只有 46 笔，且机会成本是事后代理，必须先翻译成卖出时已知特征再回测。

#### 15.14.4 下一步候选规则方向

基于这次画像，不建议继续做“亏损单统一等待反弹”。更合理的下一步是做两组真实回测候选：

1. `do_not_wait_weak_early_hold`：如果 `hold_first_3d_return < -1%` 且 `hold_first_10d_return < -4%`，则 time-stop/stop-loss 后不留残仓，甚至可测试更快全出。
2. `do_not_wait_low_volume_entry`：如果 `entry_volume_ratio_20 < 0.95` 且 `max_favorable_excursion < 5%`，则不做反弹等待。
3. `stop_loss_wait_only_low_cost_proxy`：只在 `pre_40_max_drawdown > -8%`、`hold_first_3d_return > 0`、`pre_40_volatility < 2.2%` 的 stop-loss 子集里测试残仓等待。
4. `time_stop_fast_exit_when_high_cost_proxy`：time-stop 后如果 `hold_first_3d_return < -1%`、`hold_first_10d_return < -4%`、`max_favorable_excursion < 5%`，直接全出，不再等回本。

阶段判断：这次新增画像把问题从“失败单后面会不会反弹”推进到了“哪些反弹不值得占仓等”。真实数据上，失败单中约三分之一属于高机会成本且不值得等，且早期持仓弱、入场量能弱、持仓最大浮盈不足是较稳定的可回测上下文特征。下一步应优先把这些画像翻译成“更快卖/不等待”的卖出规则，而不是继续增加等待天数。

### 15.15 从等待转向提前退出：fast-exit 真实回测

15.14 的结论是：等待信号不够准，尤其 stop-loss 后留残仓等反弹会带来明显组合路径损耗。因此本轮不再扩大等待，而是把画像反向翻译成“画像差的单更快卖”。

新增候选配置：

| 候选 | 配置 | 规则意图 |
|------|------|----------|
| `fast_exit_weak_early_hold` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_weak_early_hold.yaml` | 早期持仓弱、10 日仍弱、最高浮盈不足时，在 18 日后提前全出 |
| `fast_exit_weak_early_hold_narrow` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_weak_early_hold_narrow.yaml` | 在第一版基础上进一步要求最高浮盈不足 2%、最低浮亏超过 7%，减少误杀 |
| `stop_loss_low_cost_proxy_wait` | `configs/strategies/generated/stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_low_cost_proxy_wait.yaml` | 只在低机会成本代理画像里保留 stop-loss 残仓，其它止损全出 |

关键规则：

```yaml
# fast_exit_weak_early_hold
holding_days > 18
and pnl_pct < 0
and hold_first_3d_return < -0.01
and hold_first_10d_return < -0.04
and peak_pnl_pct < 0.05

# fast_exit_weak_early_hold_narrow
holding_days > 18
and pnl_pct < 0
and hold_first_3d_return < -0.01
and hold_first_10d_return < -0.04
and peak_pnl_pct < 0.02
and trough_pnl_pct < -0.07
```

`stop_loss_low_cost_proxy_wait` 原计划使用 `pre_40_volatility`，但当前公式运行时没有 `Std`，因此先用可运行的 `pre_40_range_mean` 作为波动代理：

```yaml
pre_40_drawdown_from_high: close / (Max(close, 40) + 1e-10) - 1
pre_40_range_mean: Mean((high - low) / (Ref(close, 1) + 1e-10), 40)
```

三条候选均已通过 dry-run。

完整回测 run：

```text
runs/20260707_204713_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_weak_early_hold
runs/20260707_205340_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_weak_early_hold_narrow
runs/20260707_204715_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_stop_loss_low_cost_proxy_wait
```

组合级结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | Sortino | Calmar | 胜率 | Profit Factor | 交易数 | 平均持仓天数 |
|------|--------|----------|--------|---------|--------|------|---------------|--------|--------------|
| 基线 `scale_12_keep85` | `45.05` | `-30.67%` | `1.408` | `2.082` | `1.434` | `61.55%` | `2.271` | `1120` | `27.95` |
| `fast_exit_weak_early_hold` | `46.62` | `-30.57%` | `1.438` | `2.108` | `1.453` | `60.91%` | `2.076` | `1130` | `27.88` |
| `fast_exit_weak_early_hold_narrow` | `50.20` | `-30.55%` | `1.472` | `2.171` | `1.487` | `61.73%` | `2.121` | `1124` | `28.01` |
| `stop_loss_low_cost_proxy_wait` | `38.47` | `-31.56%` | `1.350` | `2.012` | `1.327` | `58.58%` | `2.117` | `1159` | `28.20` |

这轮结果很关键：

1. `stop_loss_low_cost_proxy_wait` 继续失败，说明“筛出看似低机会成本的 stop-loss 后留残仓等待”仍然不是当前有效突破口。
2. `fast_exit_weak_early_hold` 第一次证明“画像差的亏损单提前退出”能在组合级跑赢基线，但第一版胜率和 Profit Factor 下滑，说明触发仍偏粗。
3. `fast_exit_weak_early_hold_narrow` 把触发从 `13` 次压到 `5` 次后，总收益、Sharpe、Sortino、Calmar、胜率、回撤均优于基线，是目前最干净的一条画像转规则证据。

卖出原因触发次数：

| 策略 | time-stop | stop-loss | fast-exit | scale | take-profit | trailing |
|------|-----------|-----------|-----------|-------|-------------|----------|
| 基线 | `508` | `156` | `0` | `144` | `77` | `17` |
| `fast_exit_weak_early_hold` | `500` | `159` | `13` | `144` | `76` | `16` |
| `fast_exit_weak_early_hold_narrow` | `506` | `156` | `5` | `146` | `76` | `16` |

#### 15.15.1 fast-exit 逐笔对照

第一版 `fast_exit_weak_early_hold` 触发 `13` 笔。把这些单按相同 `symbol + entry_date` 映射回基线后：

| 指标 | 数值 |
|------|------|
| 匹配 round trip | `13` |
| 单笔收益改善数 | `4` |
| 单笔收益变差数 | `9` |
| 平均收益差 | `-0.42pct` |
| 总净收益差 | `+1,727,776` |
| 平均节省持仓天数 | `9.54` |

这个现象说明：第一版并不是“每笔卖得更准”，而是通过释放仓位、改变后续组合路径改善了总收益。它对了大方向，但触发太宽，误杀了一些后续能修复的 time-stop 单。

因此继续收窄成 `fast_exit_weak_early_hold_narrow`：额外要求 `peak_pnl_pct < 2%` 且 `trough_pnl_pct < -7%`。这两个条件来自逐笔对照：真正值得提前卖的单，通常不是单纯早期弱，而是持仓期间最高浮盈也很弱，同时已经出现较深浮亏。

收窄版触发 `5` 笔，逐笔映射基线结果：

| 指标 | 数值 |
|------|------|
| 匹配 round trip | `5` |
| 单笔收益改善数 | `3` |
| 单笔收益变差数 | `2` |
| 平均收益差 | `+1.13pct` |
| 总净收益差 | `+243,137` |
| 平均节省持仓天数 | `10.80` |

收窄版仍不是每笔都更好，但它保留了第一版中最有价值的触发子集，组合层面的改善也更干净。尤其胜率从基线 `61.55%` 小幅提升到 `61.73%`，亏损率从 `38.45%` 降到 `38.27%`，说明它没有靠牺牲胜率换收益。

#### 15.15.2 当前结论

这轮验证支持一个更明确的方向：不要再优先研究“亏损单等待反弹”，而要研究“哪些亏损单不值得继续占仓，应该提前释放”。

第一版收窄后较有效的画像规则是：

```text
holding_days > 18
and pnl_pct < 0
and hold_first_3d_return < -1%
and hold_first_10d_return < -4%
and peak_pnl_pct < 2%
and trough_pnl_pct < -7%
```

它背后的形态语言是：入场后快速转弱，10 日仍未修复，期间几乎没有像样浮盈，但已经出现较深浮亏。这样的单继续等到 25 日 time-stop 的机会成本较高，提前退出更容易改善组合路径。

#### 15.15.3 fast-exit 小网格验证

随后围绕 `holding_days`、`peak_pnl_pct`、`trough_pnl_pct` 做了 6 个真实组合回测小网格。所有配置均位于：

```text
configs/strategies/generated/weak_to_strong_fast_exit_grid/
```

真实回测结果：

| 策略 | 触发数 | 总收益 | 最大回撤 | Sharpe | Sortino | Calmar | 胜率 | Profit Factor | 交易数 |
|------|--------|--------|----------|--------|---------|--------|------|---------------|--------|
| 基线 `scale_12_keep85` | `0` | `45.05` | `-30.67%` | `1.408` | `2.082` | `1.434` | `61.55%` | `2.271` | `1120` |
| `fast_exit_weak_early_hold_narrow` (`d18_p2_t7`) | `5` | `50.20` | `-30.55%` | `1.472` | `2.171` | `1.487` | `61.73%` | `2.121` | `1124` |
| `fast_exit_d18_p2_t8` | `2` | `58.47` | `-30.55%` | `1.527` | `2.243` | `1.555` | `62.05%` | `2.335` | `1123` |
| `fast_exit_d18_p1_t7` | `4` | `49.19` | `-30.55%` | `1.463` | `2.158` | `1.478` | `61.79%` | `2.102` | `1125` |
| `fast_exit_d18_p2_t6` | `9` | `49.13` | `-30.55%` | `1.462` | `2.143` | `1.477` | `60.79%` | `2.170` | `1127` |
| `fast_exit_d18_p3_t7` | `6` | `48.73` | `-30.56%` | `1.468` | `2.161` | `1.473` | `62.01%` | `2.096` | `1127` |
| `fast_exit_d16_p2_t7` | `6` | `44.42` | `-30.55%` | `1.437` | `2.166` | `1.433` | `60.82%` | `2.123` | `1131` |
| `fast_exit_d20_p2_t7` | `4` | `36.65` | `-30.67%` | `1.345` | `2.006` | `1.345` | `61.32%` | `2.086` | `1126` |

最优真实回测配置：

```text
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t8.yaml
runs/20260707_210917_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_fast_exit_d18_p2_t8
```

对应规则：

```text
holding_days > 18
and pnl_pct < 0
and hold_first_3d_return < -1%
and hold_first_10d_return < -4%
and peak_pnl_pct < 2%
and trough_pnl_pct < -8%
```

逐笔看，`d18_p2_t8` 只触发 2 笔：

| 标的 | 买入日 | 提前卖出日 | 基线卖出日 | 变体收益 | 基线收益 | 收益差 | 净利润差 | 节省天数 |
|------|--------|------------|------------|----------|----------|--------|----------|----------|
| `SH603180` | `2019-11-20` | `2019-12-17` | `2019-12-26` | `-1.22%` | `-0.38%` | `-0.84pct` | `-19,566` | `9` |
| `SZ002025` | `2024-08-08` | `2024-09-06` | `2024-09-13` | `-0.23%` | `-5.57%` | `+5.34pct` | `+304,676` | `7` |

这说明 `d18_p2_t8` 不是“每笔都卖得更好”，而是精准砍掉了组合路径里非常贵的一类深亏滞留单。它把总收益从 `45.05` 提高到 `58.47`，同时胜率从 `61.55%` 提高到 `62.05%`，Profit Factor 从 `2.271` 提高到 `2.335`。

该策略已加入可视化候选列表：

```text
configs/production/daily_default.yaml
```

展示标题为 `弱转强深亏提前`。`ReportService().list_reports()` 当前可见 6 个策略，其中 `弱转强深亏提前` 排第 1。

#### 15.15.4 分析模块更新：直接输出 fast-exit 候选

等待信号不准确的问题，不能只靠继续调“等反弹”规则解决。因此分析模块新增了 `fast_exit_rule_candidates`，专门搜索运行时可知的卖出上下文：

```text
holding_days
pnl_pct / return
hold_first_3d_return
hold_first_10d_return
peak_pnl_pct / max_favorable_excursion
trough_pnl_pct / max_adverse_excursion
```

真实分析报告：

```text
artifacts/pattern_analysis/weak_best_fast_exit_candidates_20260707
```

这次分析基于基线 run 的 `632` 笔闭合交易、`243` 笔失败交易。`fast_exit_rule_candidates` 会输出候选规则的 `yaml_when`，可以直接翻译到策略 YAML 后做完整组合回测。

真实分析里排名靠前的候选包括：

| 排名 | 候选 | 样本数 | 失败覆盖 | 平均收益 | 平均 MFE | 平均 MAE | 可恢复率 | 可减亏率 | 收盘转盈利率 | YAML 条件 |
|------|------|--------|----------|----------|----------|----------|----------|----------|--------------|-----------|
| 1 | `fast_exit_d16_p2_t6` | `7` | `2.88%` | `-7.02%` | `1.14%` | `-9.40%` | `28.57%` | `28.57%` | `28.57%` | `holding_days > 16 ... peak_pnl_pct < 0.02 and trough_pnl_pct < -0.06` |
| 4 | `fast_exit_d18_p2_t7` | `7` | `2.88%` | `-7.02%` | `1.14%` | `-9.40%` | `28.57%` | `28.57%` | `28.57%` | `holding_days > 18 ... peak_pnl_pct < 0.02 and trough_pnl_pct < -0.07` |
| 10 | `fast_exit_d18_p2_t8` | `6` | `2.47%` | `-7.60%` | `1.14%` | `-9.78%` | `33.33%` | `33.33%` | `33.33%` | `holding_days > 18 ... peak_pnl_pct < 0.02 and trough_pnl_pct < -0.08` |

注意：分析模块排名最高的是 `d16/d18 + peak<2% + trough<-6/-7`，但真实组合回测最佳是更保守的 `d18_p2_t8`。这点很重要：分析模块只负责把“像是该提前卖”的画像候选捞出来，最终收益、回撤、胜率、资金复用路径必须靠完整回测确认。

#### 15.15.5 分析前排候选补回测

为了验证 `fast_exit_rule_candidates` 的前排候选是否真能跑赢组合，继续把报告中高分但尚未回测的候选补成 YAML 并跑完整回测。

新增候选配置：

```text
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d16_p2_t6.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t7.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d16_p3_t6.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d16_p3_t7.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p3_t6.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d16_p2_t8.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d16_p3_t8.yaml
configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d20_p3_t6.yaml
```

这些配置 dry-run 全部通过。完整回测结果如下：

| 策略 | 触发数 | 总收益 | 最大回撤 | Sharpe | Sortino | Calmar | 胜率 | Profit Factor | 交易数 |
|------|--------|--------|----------|--------|---------|--------|------|---------------|--------|
| `fast_exit_d18_p2_t8` | `2` | `58.47` | `-30.55%` | `1.527` | `2.243` | `1.555` | `62.05%` | `2.335` | `1123` |
| `fast_exit_d16_p2_t8` | `2` | `52.92` | `-30.67%` | `1.480` | `2.195` | `1.504` | `61.91%` | `2.206` | `1128` |
| `fast_exit_d16_p3_t8` | `2` | `52.92` | `-30.67%` | `1.480` | `2.195` | `1.504` | `61.91%` | `2.206` | `1128` |
| `fast_exit_d20_p3_t6` | `9` | `51.84` | `-30.67%` | `1.496` | `2.226` | `1.495` | `61.16%` | `2.198` | `1126` |
| `fast_exit_d18_p2_t7` | `5` | `50.20` | `-30.55%` | `1.472` | `2.171` | `1.487` | `61.73%` | `2.121` | `1124` |
| `fast_exit_d18_p3_t6` | `10` | `49.54` | `-30.57%` | `1.465` | `2.147` | `1.480` | `61.07%` | `2.170` | `1129` |
| `fast_exit_d16_p3_t7` | `7` | `44.58` | `-30.66%` | `1.438` | `2.167` | `1.429` | `60.97%` | `2.122` | `1130` |
| `fast_exit_d16_p3_t6` | `11` | `42.60` | `-30.55%` | `1.432` | `2.150` | `1.415` | `60.75%` | `2.127` | `1130` |
| `fast_exit_d16_p2_t6` | `10` | `42.17` | `-30.66%` | `1.429` | `2.144` | `1.405` | `60.75%` | `2.126` | `1130` |

这轮结果把分析候选的边界讲清楚了：

1. 分析模块排名最高的 `d16_p2_t6` 真实回测只有 `42.17`，低于基线 `45.05`，说明“高 do-not-wait 分数”不等于组合收益更好。
2. `d16` 系列普遍比 `d18` 系列差，说明提前到 16 日会误杀一部分本来还能修复或不该释放的仓位路径。
3. `p3` 放宽最高浮盈阈值没有带来增益，`d16_p3_t8` 和 `d16_p2_t8` 结果完全一致，说明新增条件没有触发额外有效样本。
4. `d20_p3_t6` 的 Sharpe 不差，但总收益、胜率、Profit Factor 都不如 `d18_p2_t8`，说明等到 20 日会错过关键释放窗口。
5. 当前仍保留 `fast_exit_d18_p2_t8` 作为可视化和后续研究的主线版本。

阶段判断更新：等待信号不准这个判断成立；真正有效的第一条突破不是“等反弹”，而是“早期弱、无浮盈、已深浮亏的亏损单提前退出”。当前最优真实回测版本是 `fast_exit_d18_p2_t8`，它也是第一条由分析画像推动、并在完整组合回测中同时改善收益、胜率、Sharpe 和 Profit Factor 的弱转强卖出上下文规则。

#### 15.15.6 候选池级别扩大样本验证

`fast_exit_d18_p2_t8` 在真实组合回测中只触发 2 笔，收益提升主要来自 `SZ002025` 这一笔大幅少亏，样本量不足以单独证明画像稳健。因此新增候选池级别诊断工具，把每日记录的候选股当作“虚拟入场样本”模拟基线退出和 fast-exit 退出：

```text
quantx/tools/validate_candidate_pool_fast_exit.py
```

注意：这是诊断工具，不是组合回测。它忽略资金竞争、真实成交、仓位大小、同日排序互斥和组合净值路径，只回答一个更基础的问题：如果把这个弱转强选股池扩大到更多候选样本，`早期弱 + 峰值低 + 回撤深` 的 fast-exit 画像是否普遍能比继续按基线退出更好？

工具使用 Qlib 日线数据，基于基线 run 的候选池：

```text
runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85
```

验证命令：

```bash
conda run -n test python -m quantx.tools.validate_candidate_pool_fast_exit \
  --run runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_candidate_pool_fast_exit_selected_20260707 \
  --source selected \
  --json

conda run -n test python -m quantx.tools.validate_candidate_pool_fast_exit \
  --run runs/20260707_125517_stocktradebyz_bbi_short_long_base_pos5_topk4_close_strict_2016_2026_mainboard_scale_12_keep85 \
  --analysis-id weak_best_candidate_pool_fast_exit_raw_20260707 \
  --source raw \
  --json
```

输出位置：

```text
artifacts/candidate_pool_validation/weak_best_candidate_pool_fast_exit_selected_20260707
artifacts/candidate_pool_validation/weak_best_candidate_pool_fast_exit_raw_20260707
```

候选池级别结果：

| 样本池 | 候选样本 | 模拟样本 | 股票数 | 触发数 | 触发率 | 触发平均基线收益 | 触发平均 fast 收益 | 触发平均差值 | 改善率 |
|--------|----------|----------|--------|--------|--------|------------------|--------------------|--------------|--------|
| selected 入选候选 | `2461` | `2461` | `1378` | `23` | `0.93%` | `-4.05%` | `-4.96%` | `-0.91pct` | `34.78%` |
| raw 原始候选 | `3397` | `3397` | `1703` | `37` | `1.09%` | `-5.46%` | `-5.08%` | `+0.38pct` | `51.35%` |

解读：

1. 扩大样本后，规则确实不再只是 2 笔成交样本；在 selected 候选池触发 23 笔、raw 候选池触发 37 笔，覆盖股票数分别是 22 和 33。
2. 但 selected 候选池的结果是负的：触发后平均少赚/多亏 `0.91pct`，改善率只有 `34.78%`。
3. raw 候选池略正，但强度很弱：平均只改善 `0.38pct`，改善率 `51.35%`，接近随机边界。
4. 因此 `d18_p2_t8` 的完整组合回测收益提升，暂时不能解释为一个高覆盖、高胜率的普遍卖出规律；更像是它在组合路径中避开了少数高影响亏损单。
5. 后续如果继续研究，应把方向从“固定阈值直接提前卖”转向“候选池级别二阶段判别”：先识别这类低浮盈深回撤单，再用更丰富的上下文判断它属于 `应该砍` 还是 `可能反弹`。

这轮扩大样本验证降低了对 `fast_exit_d18_p2_t8` 的置信度：它仍可以保留在可视化层作为当前最佳组合回测版本，但不能当作已经被大样本证明的稳定规律。下一步更合理的是继续扩展分析模块，把这 23/37 个触发样本拆开看，寻找“改善组”和“误杀组”的差异。

#### 15.15.7 弱转强打分扩容与容量归因

由于 `fast_exit_d18_p2_t8` 虽然收益最好，但交易次数、股票数和候选覆盖仍偏少，继续按“宽候选池 + 打分排序”的思路做容量实验。核心问题不是再提高历史收益，而是判断弱转强能否从高选择性策略扩成更适合实盘的主策略。

新增配置：

```text
configs/strategies/generated/weak_to_strong_score_capacity/weak_to_strong_score_pool_pos8_topk12_2016_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_score_capacity/weak_to_strong_score_pool_mid_pos6_topk8_2016_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_score_capacity/weak_to_strong_score_pool_shape_pos6_topk6_2016_2026_mainboard.yaml
configs/strategies/generated/weak_to_strong_score_capacity/weak_to_strong_original_shape_pos6_topk6_2016_2026_mainboard.yaml
```

四个版本分别验证：

1. `wide`：大幅放宽入口，弱转强要素全部变成软打分。
2. `mid`：中等放宽入口，保留更强的趋势、长 RSV 热度和短 RSV 修复。
3. `shape`：保留短 RSV 上轨后下洗再重新站上的形态骨架，只轻度放宽长 RSV 条件。
4. `orig_pos6`：完全不放松原始买点，只把 `topk` 从 4 提到 6、`max_positions` 从 5 提到 6，用于做容量归因。

完整回测结果：

| 策略 | 总收益 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 交易数 | 闭合单 | raw 候选 | selected 候选 | 唯一入选股票 | 入选日占比 | 持仓日占比 | 最长空仓 | 平均持仓 |
|------|--------|----------|--------|------|---------------|--------|--------|----------|----------------|--------------|------------|------------|----------|----------|
| `fast_exit_d18_p2_t8` | `58.47` | `-30.55%` | `1.527` | `62.05%` | `2.335` | `1123` | `635` | `3557` | `2461` | `1378` | `46.78%` | `95.29%` | `46` | `3.94` |
| `wide` | `-67.11%` | `-81.99%` | `-0.361` | `52.26%` | `0.883` | `2838` | `1594` | `208763` | `29112` | `2727` | `99.22%` | `99.80%` | `3` | `7.84` |
| `mid` | `-13.18%` | `-82.66%` | `-0.046` | `51.22%` | `0.982` | `1984` | `1109` | `59143` | `16830` | `2636` | `95.92%` | `98.90%` | `28` | `5.84` |
| `shape` | `184.44%` | `-64.25%` | `0.350` | `58.45%` | `1.181` | `1413` | `787` | `4489` | `3510` | `1620` | `52.96%` | `96.90%` | `28` | `4.78` |
| `orig_pos6` | `44.14` | `-30.37%` | `1.431` | `59.81%` | `2.065` | `1322` | `744` | 原形态 | 原形态 | 原形态 | 原形态 | 原形态 | 原形态 | `4.54` |

盈亏结构解释：

| 策略 | 胜率 | 平均盈利 | 平均亏损 | 百分比盈亏比 | 真实总盈利 | 真实总亏损 | Profit Factor |
|------|------|----------|----------|--------------|------------|------------|---------------|
| `fast_exit_d18_p2_t8` | `62.05%` | `+13.11%` | `-8.01%` | `1.64` | `10226万` | `4379万` | `2.335` |
| `wide` | `52.26%` | `+13.95%` | `-9.06%` | `1.54` | `506万` | `572万` | `0.883` |
| `mid` | `51.22%` | `+13.86%` | `-8.77%` | `1.58` | `699万` | `712万` | `0.982` |
| `shape` | `58.45%` | `+12.79%` | `-8.25%` | `1.55` | `1204万` | `1019万` | `1.181` |
| `orig_pos6` | `59.81%` | `+13.18%` | `-8.15%` | `1.62` | `8559万` | `4145万` | `2.065` |

这里有一个重要发现：`wide/mid` 的笔数胜率不低，百分比口径的平均盈利也大于平均亏损，但资金加权后真实总亏损不小，Profit Factor 低于或接近 1。也就是说，问题不是“胜率太低”，而是扩容后低质量边缘信号占用更多真实仓位，亏损发生时资金暴露更大，导致组合层面的盈亏比恶化。

阶段结论：

1. 不能靠放宽买点把弱转强变成高覆盖主策略。`wide/mid` 明显失败，`shape` 虽然正收益，但回撤达到 `-64.25%`，不适合实盘。
2. 保留原始形态、只扩 `topk/max_positions` 的 `orig_pos6` 是健康的容量对照：交易数从 `1123` 提到 `1322`，平均持仓从 `3.94` 到 `4.54`，最大回撤不恶化，但总收益从 `58.47` 降到 `44.14`，Profit Factor 从 `2.335` 降到 `2.065`。
3. 这说明弱转强策略的有效边界很窄，多买低排名候选会稀释收益，但不至于像放宽形态那样崩掉。
4. 当前最优实盘展示仍应保留 `fast_exit_d18_p2_t8`；`orig_pos6` 可以作为可视化层的“容量对照版”，用于观察多买候选带来的收益稀释。
5. 后续如果要实盘化，建议把弱转强定位为高选择性子策略，而不是强行扩成主策略。主策略应该来自更高覆盖的趋势/质量/低回撤模型，弱转强只做卫星增强。

#### 15.15.8 股票级低波趋势动量 fallback 验证

基于“弱转强空仓时用更稳的核心策略承接资金”的思路，先做股票级低波趋势动量 fallback，不引入行业聚合。所有版本都使用 Qlib 数据源 `data/qlib_data_fixed`、主板股票池 `all_mainboard`、区间 `2016-01-04` 到 `2026-07-07`，低波代理使用 `Mean(Abs(ret1), n)`，没有使用行业字段、行业分组或行业 rank。

新增配置：

```text
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_pos30_top50_2016_2026_mainboard.yaml
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_defensive_pos20_top40_2016_2026_mainboard.yaml
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_risk_switch_pos30_top80_2016_2026_mainboard.yaml
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_balanced_risk_pos30_top80_2016_2026_mainboard.yaml
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_guarded_pos20_top80_2016_2026_mainboard.yaml
```

五个版本分别验证：

1. `v1_base`：宽持仓低波趋势动量，目标是先验证股票级低波趋势是否天然稳健。
2. `v2_defensive`：更严格的市场广度、长趋势、低波和回撤过滤。
3. `v3_risk_switch`：放宽个股候选，加入市场广度仓位开关和较低资金使用。
4. `v4_balanced`：恢复更严格个股趋势/动量，叠加 v3 的风险开关。
5. `v5_guarded`：在 v4 基础上进一步提高市场保护、降低 `cash_use_ratio` 和持仓上限。

完整回测结果：

| 版本 | run id | 总收益 | 年化 | 最大回撤 | Sharpe | 胜率 | Profit Factor | 交易数 | 平均持仓 | 持仓日占比 |
|------|--------|--------|------|----------|--------|------|---------------|--------|----------|------------|
| `v1_base` | `20260707_223008_low_vol_momentum_pos30_top50_2016_2026_mainboard` | `-9.80%` | `-0.98%` | `-42.75%` | `-0.063` | `30.89%` | `0.954` | `2348` | `19.11` | `90.67%` |
| `v2_defensive` | `20260707_223306_low_vol_momentum_defensive_pos20_top40_2016_2026_mainboard` | `-35.99%` | `-4.16%` | `-45.09%` | `-0.425` | `34.72%` | `0.665` | `1106` | `3.59` | `19.72%` |
| `v3_risk_switch` | `20260707_223651_low_vol_momentum_risk_switch_pos30_top80_2016_2026_mainboard` | `-10.51%` | `-1.05%` | `-25.21%` | `-0.103` | `36.47%` | `0.899` | `1788` | `8.20` | `30.18%` |
| `v4_balanced` | `20260707_223914_low_vol_momentum_balanced_risk_pos30_top80_2016_2026_mainboard` | `-4.81%` | `-0.47%` | `-29.21%` | `-0.044` | `38.03%` | `0.960` | `1988` | `8.63` | `32.93%` |
| `v5_guarded` | `20260707_224146_low_vol_momentum_guarded_pos20_top80_2016_2026_mainboard` | `+1.34%` | `+0.13%` | `-27.51%` | `0.014` | `38.65%` | `1.013` | `1242` | `4.25` | `24.58%` |

年度拆解显示，`v1` 的主要问题是长期高暴露，`2021` 和 `2022` 分别亏 `-13.09%`、`-23.42%`；`v3/v4/v5` 的市场风险开关能明显降低整体回撤，但 `2022` 仍是核心伤口：`v3` 年度亏 `-11.57%`，`v4` 年度亏 `-19.14%`，`v5` 年度亏 `-19.38%`。这说明当前股票级低波趋势的风险开关仍然滞后，无法稳定避开系统性下跌阶段。

重要观察：

1. 单纯加严过滤不可行。`v2_defensive` 持仓日占比降到 `19.72%`，但总收益和回撤都变差，说明它不是防守版，而是低频追高后止损的形态。
2. 市场广度仓位开关有效降低回撤。`v3` 把最大回撤从 `v1` 的 `-42.75%` 压到 `-25.21%`，但收益仍为负。
3. 严格个股趋势能改善收益端。`v4` 比 `v3` 的总收益更好，说明候选质量仍然重要；但回撤回升到 `-29.21%`。
4. `v5_guarded` 是相对最好的股票级版本，但年化只有 `+0.13%`，最大回撤仍有 `-27.51%`，不满足“一年 10 个点且特别稳”的 fallback 目标。
5. top traded symbols 主要集中在银行、电力等低波大票，这符合低波筛选直觉；但由于本轮不引入行业聚合，行业集中度没有被显式控制。

阶段结论：股票级低波趋势动量可以作为“风险控制参考版”保留，当前相对最好配置是：

```text
configs/strategies/generated/fallback_low_vol_momentum/low_vol_momentum_guarded_pos20_top80_2016_2026_mainboard.yaml
```

但它暂不适合作为弱转强空仓时自动接管资金的主 fallback，也不建议直接加入生产可视化主策略列表。它解决了部分回撤问题，却没有产生足够收益；更关键的是，`2022` 系统性下跌阶段仍然会造成接近 `-20%` 的年度亏损。

下一步如果继续做稳健核心仓，应从“股票级个股打分”上移到组合层风险预算：例如指数/市场状态过滤、现金/债券/ETF 替代资产、或行业/风格分散约束。行业动量可以作为下一阶段，但这轮已经证明：不引入行业聚合时，单靠股票级低波趋势很难满足稳健 fallback 的目标。

#### 15.15.9 ETF 动量轮动 fallback 验证

基于“弱转强空仓时，不在 A 股股票里硬找低质量候选，而是切到更高层资产池”的思路，新增 ETF 轮动验证。这个方向受 JoinQuant ETF 轮动策略启发，但没有直接照搬分钟级交易逻辑，而是先在 QuantX 的日线回测框架里验证 ETF fallback 本身是否有效。

数据与工具：

```text
quantx/tools/sync_etf_data.py
data/qlib_etf_data
data/raw/eastmoney/etfs
```

数据源结论：

1. BaoStock 对 ETF 日线返回空，不能直接复用股票同步路径。
2. 妙想 `mx-data` skill 已安装到 `~/.codex/skills/mx-data`，`MX_APIKEY` 已配置到 `~/.bashrc`，可查询 ETF 历史行情；验证查询 `510300` 在 `2024-01-02` 到 `2024-01-10` 的开高低收量成功，输出 7 行数据。
3. 妙想更适合自然语言查数/验证，返回表格带中文单位；批量回测落库仍使用结构化接口更合适。
4. EastMoney 历史 K 线接口在当前网络出口上不稳定，`sync_etf_data.py` 支持优先 EastMoney、失败后用 AkShare/Sina ETF 日线 fallback。
5. Sina ETF 日线为未复权口径，存在份额拆分/折算断点。已在 `sync_etf_data.py` 中加入连续化修正：当相邻收盘价跳变超过 `35%` 时，按跳变比例缩放断点之前的 OHLC，消除非交易性假涨跌。修正后已知断点如 `SH515050`、`SZ159995`、`SH512480`、`SH512010`、`SH513100` 的异常 `50%~80%` 跳变被消除；`SZ159915` 在 2024 年的 `20%` 正常波动保留。

ETF 池为 25 只，覆盖宽基、行业、海外、黄金、转债、货币类 ETF：

```text
SH510300 SH510500 SH512100 SZ159915 SH588080 SH510880 SH512890 SH512800 SH512880 SH512480 SZ159995 SH515050 SH512660 SH512010 SH512170 SH515790 SZ159928 SH512400 SH515220 SH518880 SH511380 SH511880 SH513100 SH513500 SH513030
```

新增配置：

```text
configs/strategies/generated/etf_rotation/etf_momentum_quality_top2_2016_2026.yaml
configs/strategies/generated/etf_rotation/etf_momentum_quality_top1_2016_2026.yaml
configs/strategies/generated/etf_rotation/etf_momentum_quality_defensive_top2_2016_2026.yaml
```

三版设计：

1. `top2`：最多持有 2 只 ETF，使用 `ret20/ret60/ret120` 动量、低波、回撤和趋势距离打分；近 3 日单日跌幅过滤、成交量过热过滤、趋势/回撤止损。
2. `top1`：更接近 JoinQuant 单 ETF 轮动，验证集中持有是否提升收益。
3. `defensive_top2`：提高市场、趋势、回撤和波动约束，降低资金使用，验证是否能降低回撤。

关键数据质量教训：未连续化之前，`top2` 回测出现 `2026` 单年 `-79.13%`、总收益 `-72.30%`、最大回撤 `-83.22%`，主要由 ETF 份额折算断点导致；连续化后同一策略恢复为正常结果。因此 ETF 回测必须先处理拆分/折算口径，否则策略结论完全不可信。

连续化后完整回测结果：

| 版本 | run id | 总收益 | 年化 | 最大回撤 | 年化波动 | Sharpe | Sortino | Calmar | 胜率 | Profit Factor | 平均单笔 | 平均持仓天数 | 平均持仓 | 交易数 |
|------|--------|--------|------|----------|----------|--------|---------|--------|------|---------------|----------|--------------|----------|--------|
| `top2` | `20260707_232541_etf_momentum_quality_top2_2016_2026` | `164.74%` | `9.70%` | `-23.80%` | `22.38%` | `0.434` | `0.500` | `0.408` | `38.06%` | `1.453` | `+0.61%` | `18.01` | `1.48` | `622` |
| `top1` | `20260707_232744_etf_momentum_quality_top1_2016_2026` | `114.72%` | `7.54%` | `-39.34%` | `24.71%` | `0.305` | `0.351` | `0.192` | `42.42%` | `1.349` | `+0.71%` | `18.02` | `0.78` | `331` |
| `defensive_top2` | `20260707_232744_etf_momentum_quality_defensive_top2_2016_2026` | `122.68%` | `7.91%` | `-48.62%` | `18.46%` | `0.429` | `0.414` | `0.163` | `44.74%` | `1.474` | `+0.62%` | `13.86` | `0.97` | `532` |

年度拆解：

| 年份 | `top2` 收益 / 回撤 | `top1` 收益 / 回撤 | `defensive_top2` 收益 / 回撤 |
|------|--------------------|--------------------|-------------------------------|
| 2016 | `+3.29% / -14.90%` | `+11.27% / -4.02%` | `+1.35% / -3.99%` |
| 2017 | `+6.65% / -8.01%` | `+8.51% / -7.90%` | `+8.91% / -5.94%` |
| 2018 | `-0.27% / -13.60%` | `+8.71% / -9.99%` | `+14.92% / -3.67%` |
| 2019 | `+7.21% / -9.20%` | `+3.71% / -14.49%` | `+30.48% / -10.10%` |
| 2020 | `+9.80% / -22.63%` | `-8.34% / -26.38%` | `+7.79% / -19.82%` |
| 2021 | `+8.90% / -17.89%` | `+7.52% / -33.45%` | `-19.96% / -27.11%` |
| 2022 | `-5.51% / -12.61%` | `-20.00% / -25.65%` | `-10.15% / -15.38%` |
| 2023 | `+0.24% / -16.44%` | `-1.44% / -18.78%` | `+6.82% / -7.86%` |
| 2024 | `+5.22% / -13.34%` | `+1.49% / -17.01%` | `-18.12% / -20.38%` |
| 2025 | `+25.88% / -18.56%` | `+31.39% / -27.60%` | `+24.90% / -12.53%` |
| 2026 | `+36.52% / -16.54%` | `+37.88% / -18.61%` | `+49.60% / -14.72%` |

阶段结论：

1. 当前最适合继续作为弱转强 fallback 底座的是 `top2`。它最接近目标：年化 `9.70%`，最大回撤 `-23.80%`，收益/回撤比明显优于股票级低波趋势 fallback。
2. `top1` 集中轮动并没有更好，收益更低、回撤更大，说明单 ETF 暴露对当前 25 只 ETF 池不够稳。
3. `defensive_top2` 的名字虽然防守，但实际回撤更大，主要是约束更严后暴露时机变窄，部分年份如 `2021/2024` 表现很差；它不适合作为当前 fallback。
4. `top2` 的问题是胜率只有 `38.06%`，依赖少数趋势行情贡献，且 `2020` 年内回撤仍超过 `-22%`。它可以作为资金停靠层候选，但还不能直接称为“特别稳”。
5. 下一步应该做组合层验证：当弱转强没有信号时才启用 ETF `top2`，弱转强出现时 ETF 出清，观察组合总收益、最大回撤、空仓率和资金利用率，而不是继续孤立微调 ETF 参数。

#### 15.15.10 五福 ETF/相关性策略日线复刻验证

基于 JoinQuant “ETF池子相关性研究 / 五福闹新春” 贴文，新增一个专用回测工具，而不是硬塞进现有 YAML 因子表达式：

```text
quantx/tools/run_etf_wufu_backtest.py
```

原因是该策略不只是简单动量排序，还包含：

1. 大 ETF 池，覆盖商品、海外、港股、宽基、行业、债券/货币 ETF。
2. 三态市场：正常期、震荡期、走弱期。
3. 加权 log-price 回归动量：`annualized_return * R²`。
4. 短动量、R²、量比、近 3 日跌幅、均线、Laplace、Gaussian 等状态差异化过滤。
5. 修正相关性 `P_adj = corr * exp(-MAE(累计收益曲线差)) * min(vol)/max(vol)`。
6. 高相关换仓守卫：当前持仓与目标高相关且持仓动量不强时，优先继续持有。

数据补齐结果：

```text
data/raw/eastmoney/etfs/*.csv
```

从贴文 ETF 池抽取后，实际可用 ETF 数为 `122` 只。本轮已用 Sina/AkShare 路径补齐缺失 ETF，正式回测 `missing_symbols=[]`。市场三态源使用本地指数与 ETF 代理：

| 状态源 | 实际使用 |
|--------|----------|
| 沪深300 | `SH510300` |
| 深证综指 | `SZ399101` |
| 创业板 | `SZ399006` |
| 中证A500 | `SH510300` |
| 中证1000 | `SH512100` |
| 国证2000 | `SZ399303` |

重要实现修正：

1. `sync_etf_data.py` 已处理 ETF 份额折算/拆分导致的非交易性跳变。
2. 本轮复刻工具额外修复了买入资金计算：买入数量必须把佣金/最低手续费一起算进去，否则很多低价 ETF 会因 `value + cost > cash` 被静默跳过。
3. 本轮复刻工具额外修复了缺价日估值：若持仓 ETF 当日无行情，用最近收盘价估值；若当日无可交易价格，不执行卖出，也不会把仓位凭空清掉。这个问题曾导致 `SH513500` 在 `2022-03-29` 无行情时净值被打穿，修复后该日净值保持正常。

最新全池日线复刻回测：

```text
runs/20260707_235419_etf_wufu_corr_daily_2016_2026
```

结果：

| 指标 | 数值 |
|------|------|
| ETF 数 | `122` |
| 缺失 ETF | `0` |
| 总收益 | `+1456.75%` |
| 年化收益 | `+29.84%` |
| 最大回撤 | `-26.36%` |
| Sharpe | `1.075` |
| Sortino | `1.635` |
| Calmar | `1.132` |
| 交易数 | `2145` |
| 平均持仓 | `0.98` |
| 胜率 | `45.52%` |
| Profit Factor | `1.362` |
| 平均持仓天数 | `3.50` |

候选/状态统计：

| 项目 | 数值 |
|------|------|
| 动量模式天数 | `2318` |
| 防御模式天数 | `174` |
| 正常期 | `1128` |
| 震荡期 | `269` |
| 走弱期 | `1095` |
| 有过筛候选天数 | `2318` |
| 平均过筛候选数 | `7.64` |

交易最频繁 ETF：

```text
SH511880  SH501018  SH518880  SH511220  SH513100
SH513500  SZ159928  SH513030  SZ159920  SZ159985
```

这解释了为什么五福类 ETF 策略可以明显强于前一版 25 只 ETF 的 `top2`：它不是单纯从 A 股行业 ETF 里找动量，而是在商品、海外、债券/货币、港股、A 股行业之间做资产级轮动。尤其在 A 股走弱期，商品/海外/防御资产提供了收益来源和回撤缓冲。

仍需注意的口径差异：

1. 当前是日线收盘复刻，原聚宽是 `13:10` 卖、`13:11` 买，分钟内价格和量比会有差异。
2. 当前分钟累计量用 `daily_volume * 0.67` 近似，不能完全等价于 13:10 的真实分钟累计量。
3. 当前没有可靠 ETF NAV 历史，暂未实现溢价率和溢价偏离过滤。QDII/LOF 极端溢价阶段可能被高估。
4. 中证A500 和部分指数历史用 ETF 代理，早期状态判定不完全等价于聚宽指数行情。
5. 当前复刻没有完整实现原策略中全部止损、回撤缩放、买回冷却和状态切换日细节；这版目标是先把“大池 + 状态 + 动量 + 相关性守卫”的主干跑通。

测试覆盖：

```text
tests/tools/test_run_etf_wufu_backtest.py
```

覆盖内容包括：手续费内嵌的可买数量、缺价日最近收盘估值、修正相关性矩阵、加权动量得分。执行结果：

```text
PYTHONPATH=. conda run -n test pytest tests/tools/test_run_etf_wufu_backtest.py -q
4 passed
```

阶段结论：相比前一版 25 ETF `top2`，五福全池复刻版的收益能力明显更接近原贴描述，但回撤仍有 `-26.36%`，并且当前日线口径还缺分钟成交和 ETF 溢价过滤。因此它可以作为“弱转强空仓 fallback”的强候选继续验证，但不应直接拿这版收益当实盘口径。下一步更有价值的是做组合层切换验证：弱转强有信号时优先弱转强，无信号时由五福 ETF 接管资金，并单独统计弱转强段、ETF fallback 段和切换成本。
