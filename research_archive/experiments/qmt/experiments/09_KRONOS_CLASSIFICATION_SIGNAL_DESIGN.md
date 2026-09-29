# Kronos Classification Signal Design

> 版本: v0.1.0
>
> 日期: 2026-07-16
>
> 状态: `design_proposal`
>
> 目标: 基于外部 Kronos 魔改思路，设计一套可在 QuantX-QMT 中审计、复现和 formal replay 的 A 股日频量价三分类信号生成器。
>
> 边界: 本文是实验方案，不是已验证策略，不进入正式策略库，不承诺收益。

## 1. 背景和外部线索

外部文章展示了一个轻量级 Kronos 魔改信号生成器。公开可见信息有限，但正文和评论提供了足够的方向线索。

### 1.1 明确信息

正文中可见的信息包括：

1. 模型 checkpoint 文件约 11 MB，属于轻量级精简版本。
2. 优化目标是分类信号，而不是直接优化回测收益。
3. 模型在训练和验证阶段不知道回测结果。
4. 输出是三分类：`Buy`、`Hold`、`Sell`。
5. 测试集为未见过的 A 股股池和未见过的时间段。
6. 给出了三分类 precision、recall、f1-score，其中 Buy/Sell precision 接近 0.96。

评论中可见的信息包括：

| 主题 | 外部作者回复 | 对本方案的含义 |
| --- | --- | --- |
| 是否微调 | 改网络结构，加分类器，从头训练 | 不按原版 Kronos fine-tune 复刻，优先设计轻量三分类模型 |
| Kronos 保留程度 | 只留了一部分 kronos-base 的结构 | Kronos 只作为时序 encoder 启发，不作为完整依赖 |
| 输入 | 就是量价 | 首版只使用 QMT/qlib 日线量价，不引入消息面 |
| 原版效果 | 原版确实不太行 | 必须有强基线对照，不能只因为用了 Kronos 命名就推进 |
| 类别不平衡 | 训练的时候多删掉点持有片段，用 focal loss | Hold 下采样和 focal loss 是首版核心训练策略 |
| tokenizer | 原版 Kronos tokenizer 效果不如不用 | 首版不使用原版离散 tokenizer，先连续特征输入 |
| 执行时点 | 当日 3 点后跑模型生成信号，后一日执行 | QuantX 方案严格使用 T 日收盘后信号、T+1 open 执行 |

### 1.2 不能从外部文章确定的内容

以下内容没有公开证据，本文不会把它们写成事实：

1. 精确 label 公式。
2. Kronos-base 保留了哪些层、多少层、hidden size 和 attention 结构。
3. 是否使用行业、指数、市场宽度或其他外部特征。
4. 训练集年份、股票池、复权方式和停牌处理。
5. 买卖信号最终如何转成账户组合。
6. 测试指标是否来自真实自然分布，还是来自下采样后的测试集。

本文后续方案是 QuantX-QMT 的保守可审计落地方案，而不是对外部私有实现的断言。

## 2. 项目定义

本实验要构建一个日频三分类信号模型：

```text
QMT / qlib 日线量价序列
  -> 轻量时序 encoder
  -> 三分类 head
  -> P(Buy), P(Hold), P(Sell)
  -> QuantX PredictionStore / formal account replay
```

模型输出不是直接订单，而是信号概率。账户层根据概率生成 long-only 组合，并用 QuantX formal account 统一裁判。

### 2.1 核心目标

1. 将未来路径信息压缩成 `Buy/Hold/Sell` 三分类监督标签。
2. 检验量价序列模型是否能学习到比现有 path/base 强基线更厚的可交易右尾。
3. 在严格 `T` 日收盘后信号、`T+1 open` 执行的边界下验证信号。
4. 在 label 层、prediction 层和 formal account 层逐级淘汰，避免直接用漂亮分类指标推进。
5. 保持首版实现轻量、高内聚、低耦合，不引入完整 Kronos 依赖。

### 2.2 非目标

第一阶段不做：

1. 完整复现原版 Kronos 预训练流程。
2. 使用原版 Kronos tokenizer 作为默认方案。
3. 引入公告、新闻、龙虎榜、研报等消息面特征。
4. 使用分钟线、订单簿或真实日内盘口。
5. 直接接入实盘。
6. 训练阶段优化回测收益、最大回撤或夏普。
7. 用 2026 后验挑 label 参数。

## 3. QuantX 集成边界

### 3.1 文档和实验位置

文档保存在：

```text
docs/experiments/09_KRONOS_CLASSIFICATION_SIGNAL_DESIGN.md
```

后续实验产物建议放在：

```text
.tmp/quantx-research/kronos-classification-signal-v1/
```

原因：当前只是研究设计，尚未进入正式策略库；大型训练数据、模型 checkpoint 和预测文件不应直接放入仓库。

### 3.2 后续脚本建议

若进入实现阶段，建议按职责拆成以下脚本：

| 脚本 | 单一职责 |
| --- | --- |
| `build_kronos_labels_v1.py` | 根据 QMT 日线数据生成三分类 label 和路径诊断字段 |
| `diagnose_kronos_label_distribution_v1.py` | 检查 label 分布、年份分布、行业分布和阈值敏感性 |
| `train_kronos_signal_classifier_v1.py` | 训练轻量三分类模型，输出 fold 级预测 |
| `evaluate_kronos_signal_predictions_v1.py` | 评价分类指标、TopK label、calibration 和多 seed 稳定性 |
| `replay_kronos_signal_formal_account_v1.py` | 将预测转为 PredictionStore 并进入 formal account replay |

首版实现时不需要新增复杂框架。优先复用现有数据层、预测层和 formal 回测工具。

## 4. 信息时点和因果边界

本实验固定采用收盘后信号模型：

```text
观测截止: T 日收盘后
可见信息: T 日及以前的日线量价、当时有效的股票状态和元数据
信号时间: T 日收盘后
执行时间: T+1 open
标签窗口: T+1 至 T+H
```

### 4.1 样本定义

对股票 `s` 和信号日 `t`，单个样本定义为：

```text
X(s, t) = features(s, t-L+1 : t)
Y(s, t) = label generated from tradable path after t
```

其中：

| 符号 | 含义 |
| --- | --- |
| `s` | 股票 |
| `t` | 信号日 |
| `L` | 历史输入序列长度，例如 20/60/120 日 |
| `H` | 未来观察窗口，例如 5/10/20 日 |
| `X` | 截至 `t` 日可见的量价序列 |
| `Y` | 使用 `t+1` 之后路径生成的三分类标签 |

### 4.2 允许使用的信息

第一版允许使用：

1. `t` 日及以前的 open/high/low/close。
2. `t` 日及以前的 volume、amount、turnover proxy、vwap。
3. `t` 日可确认的停牌、ST、涨跌停状态。
4. `t` 日及以前滚动计算出的收益、波动、回撤、量能状态。
5. 若项目已有严格历史有效的行业/概念元数据，可作为后续消融，不作为首版必需输入。

### 4.3 禁止使用的信息

以下行为视为未来函数或污染：

1. 使用 `t+1` 及之后行情构造特征。
2. 使用全样本均值、方差、分位数做 scaler。
3. 使用当前股票池回填历史样本，导致幸存者偏差。
4. 使用未来复权因子错误回写历史特征。
5. 使用未来最高价/最低价调整输入图像或输入标准化范围。
6. `T` 日 close 生成信号后假设在同一 close 成交。
7. 随机打散时间序列切分训练和测试。
8. 在验证/测试阶段按训练下采样分布报告指标。
9. 根据 2026 forward 后验选择 label 阈值或模型结构。

## 5. Label 设计原则

这个实验最关键的不是模型名字，而是 label 是否和真实交易动作对齐。

普通下一日涨跌 label 太薄，也和一周持仓策略错位。外部文章中的 Buy/Sell precision 极高，且 Hold 明显较多，说明其 label 大概率不是简单下一日涨跌，而是未来窗口内较明确的机会/风险分类。

本方案把 label 分成三类候选：

| Label 版本 | 定义方式 | 定位 |
| --- | --- | --- |
| v1 Triple Barrier | 看未来路径先触发止盈还是止损 | 首选主方案 |
| v2 Forward Return Band | 看未来窗口终点收益落在哪个区间 | 简单 baseline |
| v3 Path Quality | 同时考虑终点收益、最大上涨、最大回撤和路径稳定 | 后续多任务或消融 |

### 5.1 标签不是交易收益优化

训练阶段只使用三分类标签，不把 formal 回测结果输入模型。formal account 只在模型训练完成后做裁判。

这样可以避免模型把账户规则、调仓约束和收益曲线直接过拟合进训练目标。

## 6. Label v1: Triple Barrier 三分类

Triple Barrier 是首版推荐 label。它比终点收益更贴近交易动作，因为它关心未来路径先触发上涨机会还是下跌风险。

### 6.1 入场价

推荐使用：

```text
entry_price(s, t) = open(s, t+1)
```

原因：模型在 `T` 日收盘后生成信号，下一可执行时点是 `T+1 open`。

### 6.2 观察窗口

候选窗口：

```text
H in {5, 10, 20}
```

首版建议：

```text
H = 5
```

原因：QuantX 当前大量强基线和 formal account 以一周尺度为核心，先对齐现有裁判。

### 6.3 阈值

候选阈值：

```text
take_profit in {0.05, 0.06, 0.08, 0.10}
stop_loss   in {0.03, 0.04, 0.05, 0.08}
```

首版建议：

```text
take_profit = 0.06
stop_loss = 0.04
```

这组阈值会自然产生较多 Hold 样本，符合外部文章中 Hold 片段多、需要下采样的现象。

### 6.4 标签规则

对未来 `1..H` 个交易日逐日检查：

```text
ret_high(k) = high(t+k) / entry_price - 1
ret_low(k)  = low(t+k)  / entry_price - 1
```

规则：

```text
如果未来路径先触发 ret_high >= take_profit: label = BUY
如果未来路径先触发 ret_low <= -stop_loss:    label = SELL
如果 H 日内都没有触发:                         label = HOLD
```

当同一天同时触发止盈和止损时，日线无法知道日内先后。首版必须保守处理，推荐三种口径同时诊断：

| 口径 | 同日同时触发时 | 用途 |
| --- | --- | --- |
| conservative | 记为 SELL | 压力测试，防止乐观偏差 |
| neutral | 记为 HOLD | 主诊断候选 |
| optimistic | 记为 BUY | 只作为上界，不作为主结果 |

正式推进时不得只引用 optimistic 结果。

### 6.5 伪代码

```python
BUY = 2
HOLD = 1
SELL = 0

def make_triple_barrier_label(open_, high, low, t, horizon, take_profit, stop_loss):
    entry = open_[t + 1]
    if not is_valid_price(entry):
        return None

    for k in range(1, horizon + 1):
        up = high[t + k] / entry - 1.0
        down = low[t + k] / entry - 1.0

        hit_up = up >= take_profit
        hit_down = down <= -stop_loss

        if hit_up and hit_down:
            return HOLD  # neutral same-day tie policy
        if hit_down:
            return SELL
        if hit_up:
            return BUY

    return HOLD
```

### 6.6 A 股交易限制下的解释

Triple Barrier 的 label 描述未来路径事实，不等价于一定可成交收益。

例如：

1. 未来触发上涨阈值那天可能一字涨停，账户层未必买得到。
2. 未来触发下跌阈值那天可能跌停，账户层未必卖得出。
3. 股票可能停牌、ST、退市或存在价格跳变。

因此 label 层只能证明模型是否能识别路径机会，最终仍要进入 formal account 验证。

## 7. Label v2: Forward Return Band

Forward Return Band 是最简单的三分类 baseline。

### 7.1 定义

```text
entry = open(t+1)
exit = close(t+H) 或 open(t+H+1)
r = exit / entry - 1
```

标签：

```text
if r >= upper_threshold:
    label = BUY
elif r <= lower_threshold:
    label = SELL
else:
    label = HOLD
```

首版候选：

```text
H = 5
upper_threshold = 0.05
lower_threshold = -0.04
```

### 7.2 优点

1. 实现简单。
2. 可解释性强。
3. 便于和现有 5 日收益标签对齐。
4. 不依赖日内先后判断。

### 7.3 缺点

1. 忽略路径中最大回撤。
2. 不能区分先涨后跌和先跌后涨。
3. 容易把中途不可承受波动的样本标成 Buy。
4. 和实际止盈止损交易动作不完全一致。

因此 v2 只作为 baseline，不作为首选主标签。

## 8. Label v3: Path Quality 三分类

Path Quality label 用于后续增强，不建议第一版直接复杂化。

### 8.1 路径统计

对 `t+1..t+H` 计算：

```text
max_up = max(high[t+1:t+H]) / entry - 1
max_down = min(low[t+1:t+H]) / entry - 1
end_return = close[t+H] / entry - 1
drawdown_from_entry = min(close[t+1:t+H] / entry - 1)
positive_days = count(close[t+k] > close[t+k-1])
```

### 8.2 示例规则

```text
BUY:
  max_up >= 0.06
  end_return >= 0.02
  max_down > -0.04

SELL:
  max_down <= -0.04
  end_return <= 0.00

HOLD:
  otherwise
```

### 8.3 使用边界

Path Quality label 更贴近“强但不脆弱”的候选学习，但也更容易把人工偏好写进标签。只有在 v1/v2 label 土壤诊断完成后，才值得引入。

## 9. 类别不平衡处理

三分类交易信号天然不平衡。大多数交易日、股票和路径都应是 Hold，真正 Buy/Sell 信号较少。

外部作者明确提到：

```text
训练的时候多删掉点持有片段，用 focal loss
```

QuantX 首版建议按以下方式处理。

### 9.1 训练集处理

1. 保留全部 Buy 样本。
2. 保留全部 Sell 样本。
3. 对 Hold 样本做分层下采样。
4. 分层维度至少包括年份；若行业元数据可靠，可加入行业。
5. 下采样比例必须作为实验参数记录。

示例：

```text
target_train_ratio:
  BUY  : 1.0
  SELL : 1.0
  HOLD : min(original_hold, 2.0 * (BUY + SELL))
```

### 9.2 验证和测试集处理

验证集、测试集和 2026 forward 不做下采样，必须保持真实自然分布。

如果只在下采样后的测试集上报 precision/recall/f1，会显著夸大模型可用性。

### 9.3 Loss 函数

首版使用 focal loss：

```text
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

推荐初始参数：

```text
gamma = 2.0
alpha = class-balanced weights from training fold
```

后续消融：

| Loss | 用途 |
| --- | --- |
| CrossEntropy | 最朴素 baseline |
| Weighted CrossEntropy | 只处理类别权重 |
| Focal Loss | 降低易分类 Hold 的权重 |
| Class-Balanced Focal Loss | 同时处理类别频率和难样本 |

## 10. 特征设计

首版遵循外部线索，只使用量价。特征应尽量从已有 QMT/qlib 数据中生成，避免额外数据依赖。

### 10.1 输入形状

```text
X shape = [num_samples, seq_len, feature_dim]
```

候选 `seq_len`：

```text
20, 60, 120
```

首版建议：

```text
seq_len = 60
```

### 10.2 基础 OHLCV 特征

每个交易日可构造：

| 特征 | 定义 |
| --- | --- |
| `ret_close_1d` | `log(close_t / close_{t-1})` |
| `ret_open_1d` | `log(open_t / close_{t-1})` |
| `ret_intraday` | `log(close_t / open_t)` |
| `high_open` | `high_t / open_t - 1` |
| `low_open` | `low_t / open_t - 1` |
| `range_hl` | `high_t / low_t - 1` |
| `close_position` | `(close_t - low_t) / (high_t - low_t)` |
| `volume_log` | `log1p(volume_t)` |
| `amount_log` | `log1p(amount_t)` |
| `vwap_gap` | `close_t / vwap_t - 1` |

### 10.3 滚动路径特征

滚动窗口建议：

```text
window in {5, 10, 20, 60}
```

可构造：

| 特征 | 用途 |
| --- | --- |
| rolling return | 多周期趋势 |
| rolling volatility | 波动率状态 |
| rolling max drawdown | 脆弱性 |
| rolling high distance | 距离阶段高点 |
| rolling low distance | 距离阶段低点 |
| volume z-score | 放量/缩量 |
| amount z-score | 资金活跃度 |
| range z-score | 波动扩张 |

### 10.4 同日横截面特征

如果已有全市场面板，可以在每个交易日做横截面 rank：

```text
rank_pct(feature, date=t)
```

首版可加入：

1. 当日收益 rank。
2. 5/20/60 日收益 rank。
3. 波动率 rank。
4. 成交额 rank。
5. 距高点/低点 rank。

这些 rank 只能使用 `t` 日已知信息，且每个 fold 独立计算，不使用未来交易日。

### 10.5 暂不使用的特征

首版不使用：

1. 公告标题。
2. 新闻和研报。
3. 龙虎榜。
4. 盘后才披露且发布时间不稳定的数据。
5. 分钟线特征。
6. 订单簿特征。

原因是本文目标是复现“量价三分类信号”路线，而不是重新打开消息面工程。

## 11. 归一化方案

归一化是最容易产生未来函数的环节之一。

### 11.1 推荐规则

1. 价格字段优先转成收益或相对位置，避免绝对价格尺度。
2. volume/amount 使用滚动 z-score 或 log 后再横截面 rank。
3. 所有 scaler 只在训练 fold fit。
4. validation/test/forward 只 transform，不 refit。
5. 不使用全区间均值、方差、最大值、最小值。

### 11.2 两类归一化

| 类型 | 说明 | 风险 |
| --- | --- | --- |
| 时序内归一化 | 使用股票自身过去窗口统计 | 需要确保窗口不含未来 |
| 同日横截面 rank | 使用同一天所有股票的相对位置 | 需要确保股票池是当日可见股票池 |

### 11.3 首版建议

首版采用混合方案：

1. OHLC 主要转为 log return 和相对比值。
2. volume/amount 用 `log1p` 加滚动 z-score。
3. 部分稳定因子增加同日 rank_pct。
4. 最终特征用训练 fold 的 robust scaler 或标准 scaler。

## 12. 模型结构

外部线索显示作者没有直接使用完整原版 Kronos，而是保留部分 `kronos-base` 结构、加分类器、从头训练。

QuantX 首版不应依赖完整 Kronos。更保守的做法是实现一个小型时序分类器，并把 Kronos 作为结构启发。

### 12.1 输入和输出

```text
Input:  [batch, seq_len, feature_dim]
Output: [batch, 3]
```

类别顺序建议固定为：

```text
0 = SELL
1 = HOLD
2 = BUY
```

### 12.2 Encoder 候选

| Encoder | 优点 | 缺点 | 首版优先级 |
| --- | --- | --- | --- |
| TCN | 快、稳定、适合局部形态 | 长程依赖有限 | 高 |
| GRU | 实现简单，参数少 | 表达力有限 | 中 |
| Small Transformer Encoder | 接近 Kronos 类结构 | 更容易过拟合 | 高 |
| Kronos-inspired blocks | 贴近外部思路 | 需要研究原结构 | 后续 |

### 12.3 推荐首版结构

首版建议用小型 Transformer Encoder 或 TCN 二选一。

```text
Linear feature projection
  -> positional encoding
  -> 2-4 layer Transformer Encoder or TCN blocks
  -> sequence pooling
  -> LayerNorm
  -> Linear classification head
```

参数规模目标：

```text
1M - 5M parameters
checkpoint 5MB - 30MB
```

这与外部 11MB checkpoint 的轻量化方向一致。

### 12.4 Pooling 方案

候选：

| Pooling | 说明 |
| --- | --- |
| last token | 使用最新交易日状态 |
| mean pooling | 汇总整个窗口 |
| attention pooling | 学习不同日期权重 |

首版建议：

```text
last token + mean pooling concat
```

原因是它开发量低，并同时保留最新状态和窗口整体状态。

## 13. Tokenizer 策略

外部评论提到原版 Kronos tokenizer 效果不如不用。因此首版不引入原版 tokenizer。

### 13.1 首版策略

```text
连续量价特征 -> Linear projection -> Encoder
```

### 13.2 后续消融

如果连续输入模型有正信号，再考虑：

| Tokenizer | 说明 |
| --- | --- |
| return bin token | 将收益离散成 bins |
| OHLCV patch token | 每几天作为一个 patch |
| learned vector quantization | 学习离散 codebook |
| hybrid continuous + discrete | 连续特征和离散 token 拼接 |

### 13.3 保留原则

任何 tokenizer 方案必须同时满足：

1. prediction 层超过连续输入 baseline。
2. 2026 forward 不退化。
3. formal account 不退化。
4. 不显著增加工程复杂度。

否则不保留。

## 14. 训练方案

### 14.1 训练方式

首版从头训练，不加载原版 Kronos 权重。

原因：

1. 外部作者明确提到从头训练。
2. 原版 tokenizer 和任务目标不一定适合 A 股 Buy/Hold/Sell。
3. 从头训练更容易审计数据边界和模型行为。

### 14.2 优化器和调度

建议：

```text
optimizer = AdamW
lr = 1e-3 or 3e-4
weight_decay = 1e-4
scheduler = cosine decay with warmup
batch_size = constrained by memory
epochs = 10-30 with early stopping
```

### 14.3 Early stopping

不以训练 loss 作为唯一停止指标。优先使用 validation 上的：

1. macro F1。
2. Buy/Sell average F1。
3. Top20 validation label。
4. Calibration error。

如果分类指标提升但 TopK label 下降，应视为交易目标错位。

### 14.4 多随机种子

至少使用：

```text
seed in {7, 11, 19}
```

每个 seed 独立训练、独立输出预测。最终报告均值和最差 seed，不只报告最好 seed。

## 15. Walk-forward 切分

不允许随机切分。推荐使用 expanding walk-forward。

### 15.1 时间切分

示例：

| Fold | Train | Validation/Test |
| --- | --- | --- |
| 1 | 2016-2020 | 2021 |
| 2 | 2016-2021 | 2022 |
| 3 | 2016-2022 | 2023 |
| 4 | 2016-2023 | 2024 |
| 5 | 2016-2024 | 2025 |
| Forward | 2016-2025 | 2026 |

如果数据起点不同，应保持同样原则：训练期必须早于测试期。

### 15.2 股票池隔离

外部文章提到未见过 A 股股池和未见过时间段。QuantX 可以设计两类验证：

| 验证方式 | 目的 |
| --- | --- |
| time OOS | 验证未见时间段 |
| asset OOS | 验证未见股票 |
| time + asset OOS | 同时验证未见股票和未见时间 |

asset OOS 示例：

```text
按股票代码 hash 分成 train_assets / heldout_assets
训练只使用 train_assets
测试在 heldout_assets 的未来时间段上评价
```

注意：asset OOS 不能替代 time OOS。金融时间序列最重要的仍是未来时间穿越。

### 15.3 Fold 内 scaler

每个 fold 必须独立：

```text
fit scaler on train fold only
transform validation/test with frozen scaler
```

不能在全数据上提前计算标准化参数。

## 16. 评价指标

评价必须分三层：分类层、预测排序层、formal account 层。

### 16.1 分类层指标

报告：

1. Confusion matrix。
2. Per-class precision。
3. Per-class recall。
4. Per-class F1。
5. Macro F1。
6. Buy/Sell average F1。
7. Class distribution。

必须同时报告自然分布测试集上的指标。下采样测试集指标只能作为训练诊断，不作为策略证据。

### 16.2 概率质量指标

报告：

1. Calibration curve。
2. Expected calibration error。
3. Buy probability decile future return。
4. Sell probability decile future drawdown。
5. `P_buy - P_sell` 分层收益。

### 16.3 TopK label 指标

候选 score：

```text
score_buy = P_buy
score_edge = P_buy - alpha * P_sell
score_logit = logit_buy - alpha * logit_sell
```

报告：

| 指标 | 说明 |
| --- | --- |
| Top10/15/20 mean future return | 和现有 QMT 实验对齐 |
| TopK hit rate | TopK 中 BUY label 占比 |
| TopK SELL contamination | TopK 中 SELL label 占比 |
| yearly TopK label | 防止单年贡献 |
| 2026 TopK label | 前向验证 |

### 16.4 Formal account 指标

只有 prediction 层通过门槛后，才进入 formal account。报告：

1. 2022-2025 final value / total return。
2. 2026 forward return。
3. 年度收益。
4. 最大回撤。
5. 平均持仓数。
6. 平均持有期。
7. 换手。
8. 拒单原因分布。
9. 双倍成本压力测试。
10. 与 Exp40/path baseline 的差异。

## 17. 信号转组合规则

首版只做 long-only，不做做空。

### 17.1 选股分数

推荐主分数：

```text
score = P_buy - 0.5 * P_sell
```

消融：

```text
score = P_buy
score = P_buy - P_sell
score = logit_buy - 0.5 * logit_sell
```

### 17.2 TopK

报告：

```text
TopK in {10, 15, 20}
```

首版账户主口径：

```text
Top20
```

原因：QuantX 现有强基线大量使用 Top20，可比性更强，且能避免极少数股票撑收益。

### 17.3 调仓和持有

首版：

```text
rebalance = 5 trading days
entry = next open
exit = next rebalance or formal account standard exit
```

暂不引入动态止盈止损。否则 label 中的 barrier 和账户中的 barrier 会混在一起，难以判断模型是否真的有选股能力。

## 18. Formal Account 验证要求

QuantX 的经验显示，很多标签层和轻量账户高收益会被正式成交约束打掉。因此本实验必须把 formal account 作为最终裁判。

### 18.1 必须纳入的约束

1. 涨停不可买。
2. 跌停或停牌导致退出受限。
3. ST/退市处理。
4. 手数约束。
5. 成交成本。
6. 资金分配。
7. overlap/rebalance 规则。
8. 价格跳变和异常数据过滤。

### 18.2 强基线对照

至少对照：

| 基线 | 用途 |
| --- | --- |
| Exp40 path sequence | 当前强 path/base 土壤 |
| QMT 成交额/资金流 ranker | 量价特征同源对照 |
| Deep Learning Alpha compact GRU | 现有 DL 路线反例 |
| naive return-band classifier | 简单标签/模型对照 |

### 18.3 晋级门槛

进入下一阶段至少需要：

1. 2022-2025 prediction 层 Top20 label 超过强基线或形成明显互补。
2. 2026 forward 不低于强基线太多，最好正增量。
3. formal account 开发期和 2026 均不退化。
4. 多 seed 不依赖单个幸运 seed。
5. 年度收益不能只靠单一年份。
6. 平均持仓数不能过低。
7. 双倍成本后不完全失效。

如果只在分类 F1 上好看，但 TopK label 和 formal account 没有改善，应判定为 `rejected_before_formal_account` 或 `rejected_after_formal_replay`。

## 19. Lookahead 审计清单

Buy/Sell precision 接近 0.96 时，第一反应应是审计泄漏，而不是庆祝模型有效。

### 19.1 Label 审计

1. `entry` 是否确认为 `t+1 open`。
2. label 是否只用于训练目标，不进入特征。
3. 同日止盈止损同时触发是否使用保守或中性处理。
4. label 参数是否在 2026 结果后才选择。
5. 不同 horizon/threshold 是否完整记录，而不是只保留最优。

### 19.2 特征审计

1. 每个特征是否只使用 `<= t` 数据。
2. rolling window 是否包含未来行。
3. 横截面 rank 是否使用当日真实可见股票池。
4. 行业/概念是否使用历史有效版本，而不是当前标签回填。
5. 复权处理是否引入未来价格调整。

### 19.3 Scaler 审计

1. scaler 是否只 fit 在训练 fold。
2. validation/test 是否没有 refit。
3. rank/zscore 是否按日期或训练期正确隔离。
4. 是否保存每个 fold 的 scaler artifact。

### 19.4 股票池审计

1. 是否包含历史退市股票。
2. 是否错误剔除当时可交易但现在不存在的股票。
3. ST、停牌、上市不足窗口样本是否处理一致。
4. 新股样本是否有足够历史窗口。

### 19.5 回测审计

1. signal date 和 trade date 是否错开。
2. 是否用 `t` 日 close 执行了 `t` 日信号。
3. 涨停不可买是否生效。
4. 停牌是否不能交易。
5. 成本是否和基线一致。
6. 调仓日历是否和 prediction date 对齐。

## 20. 最小落地路径

为了控制开发量，建议分阶段推进。每阶段都可以独立失败，不需要一开始写完整模型系统。

### 20.1 阶段 0: Label 分布诊断

只生成 label，不训练模型。

产出：

1. 各年份 Buy/Hold/Sell 分布。
2. 各行业/市值/成交额分组分布。
3. 不同 `H/take_profit/stop_loss` 的敏感性。
4. Buy/Sell 样本未来真实收益路径。
5. Hold 样本数量是否过大。

通过条件：

1. Buy/Sell 样本不是极端稀少。
2. label 分布跨年份不崩坏。
3. Buy/Sell 样本在未来路径上确实有可解释差异。

### 20.2 阶段 1: 非深度 baseline

先用 Logistic Regression、LightGBM 或 ExtraTrees 检查 label 是否可学。

原因：如果简单模型完全学不到，直接上深度模型大概率只是过拟合。

产出：

1. 自然分布测试集 macro F1。
2. Buy/Sell F1。
3. TopK by `P_buy - P_sell` label。
4. 2026 forward。

### 20.3 阶段 2: 轻量序列模型

在 label 和非深度 baseline 通过后，训练 TCN/Small Transformer。

产出：

1. fold 级预测文件。
2. seed 级预测文件。
3. classification report。
4. TopK label report。
5. calibration report。

### 20.4 阶段 3: PredictionStore 接入

将模型输出转为 QuantX 可识别的 prediction records：

```text
date, instrument, score, p_buy, p_hold, p_sell, fold, seed, model_id
```

### 20.5 阶段 4: Formal replay

只在 prediction 层通过后做 formal account。

对照：

```text
base/path baseline
kronos_cls score=P_buy
kronos_cls score=P_buy-0.5*P_sell
kronos_cls weak blend with path score
```

### 20.6 阶段 5: 融合或二阶段重排

若独立模型不够强但有互补性，再考虑：

```text
score_final = 0.8 * path_score + 0.2 * kronos_cls_score
```

或者只在 path Top100/200 候选池内重排，降低候选土壤风险。

## 21. 推荐首版参数

以下是初始建议，不是实验结论。

| 参数 | 建议值 |
| --- | --- |
| label | Triple Barrier |
| horizon | 5 |
| take_profit | 0.06 |
| stop_loss | 0.04 |
| same-day tie | HOLD |
| seq_len | 60 |
| feature source | QMT/qlib 日线 OHLCV/amount/vwap |
| model | Small Transformer Encoder or TCN |
| hidden size | 64-128 |
| layers | 2-4 |
| loss | Class-balanced focal loss |
| gamma | 2.0 |
| hold sampling | 分层下采样 |
| score | `P_buy - 0.5 * P_sell` |
| topk | 20 |
| rebalance | 5 |
| execution | T+1 open |
| seeds | 7, 11, 19 |

## 22. 失败判定

以下情况应停止或降级，不继续调参追结果。

### 22.1 Label 失败

1. Buy/Sell 样本极少，无法稳定训练。
2. Buy/Sell 分布高度集中在单一年份。
3. 不同阈值下 label 方向不稳定。
4. label 层机会被 formal 成交约束系统性打掉。

### 22.2 模型失败

1. 训练集指标高、自然验证集指标低。
2. Buy/Sell precision 高但 recall 极低，TopK label 没改善。
3. 2022-2025 小增，2026 明显反向。
4. 多 seed 方差过大。
5. 只超过 Logistic baseline，不超过强 path baseline。

### 22.3 账户失败

1. formal account 低于 Exp40/path baseline。
2. 收益来自极少数股票或极少数年份。
3. 双倍成本后明显失效。
4. 拒单集中在涨停/停牌，说明模型主要预测不可买右尾。
5. 2026 forward 退化且无法用事前规则解释。

## 23. 预期风险

### 23.1 高 precision 可能来自标签设计

如果 Buy/Sell 阈值设置很极端，模型可能只在非常明显的样本上预测 Buy/Sell，从而得到高 precision。但这不代表账户收益高，因为：

1. Buy/Sell 样本可能太少。
2. 模型可能大部分时间预测 Hold。
3. TopK 可能依然混入不可交易强股。
4. precision 不考虑持仓容量和资金效率。

### 23.2 Triple Barrier 可能和账户规则不一致

label 中的止盈止损只是路径定义。如果账户层固定持有 5 日，不执行止盈止损，则 label 和账户行为存在错位。

首版接受这种错位，因为 label 只是机会识别目标。但若 formal account 不传导，后续需要改成更贴近账户的 label。

### 23.3 Hold 下采样可能破坏概率校准

Hold 下采样有助于训练，但会改变类别先验。模型输出概率可能不能直接解释为自然分布概率。

因此需要 calibration 和自然验证集评价，必要时做温度缩放或 prior correction。

### 23.4 只用量价可能不够

如果只用 OHLCV 无法超过现有强基线，不应立刻引入更多网络层。更可能的问题是 label、候选土壤或可交易约束，而不是模型表达力不足。

## 24. 与现有 QuantX 实验的关系

这条路线和现有实验不是替代关系，而是一个新的 label/model 组合。

### 24.1 和 Deep Learning Alpha Search 的关系

`05_DEEP_LEARNING_ALPHA_SEARCH.md` 已经证明普通 compact GRU 没有超过 path sequence 强基线。本方案区别在于：

1. 不直接做收益回归或五分位排序。
2. 明确做 Buy/Hold/Sell 三分类。
3. 使用路径型 label。
4. 明确处理 Hold 类别不平衡。
5. 重点检查分类信号是否更贴近交易动作。

### 24.2 和 QMT 成交额/资金流 Ranker 的关系

成交额/资金流 Ranker 已经说明 QMT 内生量价特征有解释力但 formal 不够厚。本方案可复用相同数据源，但目标函数不同。

如果本方案只得到类似薄信号，应判定为没有打开新收益土壤。

### 24.3 和 path sequence 强基线的关系

path sequence 是必须挑战的强基线。本方案有两种可能用途：

1. 独立候选生成器：全市场直接选 TopK。
2. 二阶段重排器：只在 path/base Top100/200 内重排。

首版建议两个口径都评估，但先看独立 TopK label，再看 path pool 内重排。

## 25. 结论

外部 Kronos 魔改思路的关键不在于“使用 Kronos 名字”，而在于以下组合：

1. 将金融时序任务改成交易动作三分类。
2. 使用未来路径构造 Buy/Hold/Sell label。
3. 用 Hold 下采样和 focal loss 处理类别不平衡。
4. 只使用 T 日及以前量价信息。
5. T 日收盘后生成信号，T+1 执行。
6. 用轻量模型从头训练，而不是盲目微调原版 Kronos。
7. 最终用 formal account 裁判，而不是只看分类指标。

QuantX-QMT 的第一步不应该直接写复杂 Kronos 复刻，而应先做 label-first 验证：确认 triple-barrier 三分类标签在 A 股日频、QMT 成交约束和 2026 forward 中是否真的形成厚土壤。

若 label 土壤不厚，继续堆模型没有意义。若 label 土壤成立，再训练轻量 TCN/Transformer 三分类器，并严格对照 path sequence、QMT 量价 ranker 和 formal account 结果。
