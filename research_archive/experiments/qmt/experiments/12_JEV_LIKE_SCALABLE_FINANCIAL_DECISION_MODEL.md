# 12 Jev-like 可扩展金融决策模型

> 版本：v0.3.3
>
> 建档日期：2026-09-20
>
> 状态：`implementation_plan`
>
> 文档定位：本文件是长期维护的主规格文档。后续关于 query、choice、label、loss、模型结构、训练结果、失败证据和正式 QuantX 回测的变更，都应持续更新到这里。
>
> 当前阶段：V1 股票状态理解模型的接口、数据、Query 插件、文本 Processor、模型、训练和回测方案已进入实现前评审；本轮只修改文档，不修改训练代码，不启动新实验。

## 0. v0.3 当前有效设计裁决

本节优先级高于本文后续历史设计段落。若旧章节与本节冲突，以本节和第 37 节之后的 v0.3 实现方案为准。

### 0.1 V1 只理解股票未来状态

V1 范围：

```text
股票历史状态
+ 市场上下文
+ 文本 Query
+ 文本 Candidates
-> Candidate Logits / Probabilities
```

V1 不包含：

```text
账户状态
持仓状态
加仓
减仓
止盈
止损
组合现金比例
Portfolio action
```

这些任务保留为远期扩展，不能进入 V1 的模型、数据集或验收标准。

### 0.2 模型只看到文本语义，不额外看到 horizon 字段

模型输入：

```text
AE latent
Query text embedding
N 个 Candidate text embedding
Candidate mask
```

模型不额外接收：

```text
horizon id
threshold scalar
query family id
label builder id
query id embedding
```

例如：

```text
Query:
  从下一个交易日收盘开始计算，
  这只股票未来七个交易日的收益是否为正？

Candidates:
  是，未来七个交易日收益为正
  否，未来七个交易日收益不为正
```

`horizon=7` 仍会存在于 Query 插件配置中，用于构建真实 label、审计时间语义和生成规范文本，但不会作为额外数值特征喂给模型。

这样才能真实验证文本 Query embedding 是否承载任务语义，而不是让模型绕过文本、只记住 task ID。

### 0.3 每个 Query 必须同时定义分类标签和比较值

每个具体 QuerySpec 必须提供：

```text
query_text
candidate_texts
target_candidate_builder
preference_value_builder
valid_mask_builder
```

其中：

```text
target_candidate
```

用于训练当前股票应该选择哪个 candidate；

```text
preference_value
```

用于在同一交易日随机寻找更差股票，构造 Pairwise。

### 0.4 Pairwise 是核心，不是可选补丁

V1 同时训练：

```text
单股票 Candidate Choice
+ 同 Query、同日期、跨股票 Pairwise
```

概率必须既能解释单只股票的真实 outcome，又能保持横截面 A/B 排序。

### 0.5 一个 Python 文件代表一个 Query Family

推荐：

```text
queries/
  return_direction.py
  return_range.py
  relative_return.py
```

每个文件负责一种 query 类型，并可以批量生成几十或几百个具体 QuerySpec。

不推荐：

```text
每个具体 query 一个重复 Python 文件
```

这样既保留每类 query 独立的 label 逻辑，又能规模化扩展。

## 1. 核心目标

当前 Reward Model 已经证明，深度学习可以从高噪声日频量价和市场状态中学习到具有交易价值的弱横截面因子。但当前模型仍然是固定任务结构：

```text
股票状态
  -> 固定 Reward Transformer
  -> 固定 return / Sharpe / drawdown logits
```

这种结构存在三个扩展瓶颈：

1. 新增 3 日、10 日、20 日收益任务时，通常需要增加新 head 或新模型。
2. 固定 scalar score 很难表达概率、不确定性、收益幅度、路径风险和可成交性之间的区别。
3. 选股、风险预测、买入、持有、止盈和止损被拆成相互独立的实验，不能共享统一的决策表示。

本研究希望把 Reward Model 升级为一个统一的、query-conditioned 金融决策模型：

```text
State + Query + Choices
  -> Choice Logits
  -> Calibrated Probabilities
  -> 显式决策组合
```

长期目标不是构建一个不可解释的万能交易 score，而是构建一个能够回答多个原子金融问题、输出可信概率，并由策略代码显式组合答案的通用金融决策模型。

## 2. Jev 启发与证据边界

### 2.1 可确认的公开范式

TypeSafe AI 对 Jev/System One 的公开说明中，核心接口可以概括为：

```text
共享 state
+ 一个或多个独立 question
+ 每个 question 的 typed choices
-> 每个 choice 的 logits / probabilities / confidence
```

公开资料还强调：

1. 将复杂任务拆成多个原子问题。
2. 多个问题可以共享同一个 state。
3. 最终决策可以在代码中组合多个问题的输出。
4. Choice、Score、二值判断属于不同的 typed question。

参考：

- `https://docs.typesafe.ai/introduction`
- `https://docs.typesafe.ai/models`

### 2.2 不能声称的内容

Jev 的完整内部网络、训练数据构造、RLCD 细节和优化过程没有公开到可直接复刻的程度。因此本文只采用其公开接口启发，定义一个适配 QuantX 数据与实验纪律的 **Jev-like** 模型。

本文不得使用以下表述：

```text
复刻 Jev
实现 Jev 原始架构
复现 Jev 的 RLCD
达到 Jev 官方效果
```

正确表述是：

```text
受 Jev 的 state + query + choices + logits 接口启发，
构建一个 query-conditioned 金融决策模型。
```

## 3. 当前实验证据

### 3.1 现有 Multi-task Reward baseline

当前 all-A Multi-task Reward baseline 使用：

```text
过去 60 日 raw_relative 特征
-> 冻结 AE
-> 27 x 128 latent tokens
-> 49.73M Reward Transformer
-> return / Sharpe / drawdown 三个固定 head
```

训练使用固定 H7 标签：

```text
return_7d = close(T+7) / close(T+1) - 1
```

三个任务均为同一 signal date 内的 A/B Bradley-Terry pairwise 排序：

```text
return head:
  return(A) > return(B) + 0.5%

Sharpe head:
  abs(return(A) - return(B)) < 0.5%
  SharpePercentile(A) > SharpePercentile(B) + 5%

drawdown head:
  abs(return(A) - return(B)) < 0.5%
  DrawdownPercentile(A) > DrawdownPercentile(B) + 5%
```

推理时：

```text
全市场按 return score 取 Top50
-> 只在 Top50 内按 Sharpe / drawdown rank 等权重排
-> 取最终 Top10
```

### 3.2 Residual Reward v1 的证据

Residual Reward v1 固定 H7，使用 T+1 close 买入、T+7 close 卖出、双端可成交并扣成本后的净收益构造 pair。

正式 QuantX 结果：

| 策略 | 累计收益 | 年化收益 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Multi-task baseline Top10-H7 | `+203.91%` | `18.91%` | `-38.03%` | `0.644` |
| Residual-only Top10-H7 | `+235.39%` | `20.75%` | `-50.58%` | `0.577` |
| Baseline Top50 -> Residual Top10-H7 | `+204.64%` | `18.95%` | `-40.74%` | `0.548` |

该结果说明：

1. Residual Reward 学到了与 baseline 低相关的高收益因子。
2. 该因子同时带来更高左尾风险和更多 T+1 `price_jump` / 涨停拒单。
3. 单一 scalar Reward 无法同时清晰表达收益概率、收益幅度、路径风险和可成交性。
4. 简单用 Residual 全量替换 baseline 风险重排，没有提高组合风险调整收益。

这些证据构成 Jev-like 方向的直接动机：不再要求一个 scalar score 隐式编码所有决策语义。

## 4. 模型定义

### 4.1 总体接口

模型输入定义为：

```text
State:
  股票历史状态
  市场状态
  候选上下文
  可选持仓/账户状态

Query:
  要回答的金融问题

Choices:
  该问题允许的答案集合
```

模型输出：

```text
logits = [logit(choice_1), ..., logit(choice_K)]
probabilities = softmax(logits)
```

形式化表示：

```text
Z_state = StateEncoder(state)
Z_query = QueryEncoder(query)
Z_choice_j = ChoiceEncoder(choice_j)

logit_j = DecisionScorer(Z_state, Z_query, Z_choice_j)
```

### 4.2 State Encoder

第一版不重新训练底层 AE：

```text
过去 60 日股票/市场/候选特征
-> 已验证的 frozen AE
-> Z_ae: [B, 27, 128]
```

原因：

1. raw-feature Reward 正式结果显著弱于 frozen-AE baseline。
2. 第一轮应只验证 query-choice 训练范式，不同时改变 representation、label 和决策接口。
3. 若 query-conditioned 模型不能在 frozen representation 上形成增量，不应先归因于 AE。

注意：冻结 AE 时，改善的是 query-conditioned decision representation，AE 本身不会变强。

只有 query-choice 范式通过独立验证后，才允许在后续阶段：

```text
冻结 AE 大部分层
仅以极低学习率解冻顶部 encoder block
或增加独立 semantic adapter
```

### 4.3 Query Encoder

第一版 query 正式使用自然语言文本 embedding，同时保留结构化 typed fields：

```yaml
canonical_text_zh: >
  从信号日后的第一个交易日收盘价作为起点，
  该股票未来七个交易日的收盘收益是否为正？
query_family: return_direction
horizon: 7
entry_offset: 1
price_mode: close_to_close
threshold: 0.0
execution_contract: t1_close
state_scope: flat
```

Query Encoder 将冻结文本编码器的 embedding 与这些离散、连续字段共同编码为 query tokens。

文本 embedding 是正式模型输入，但不是唯一任务身份。结构化字段负责 horizon、threshold、成交时点和区间边界的精确语义，避免文本编码器对数值合同产生歧义。

第一版不声称支持任意自然语言零样本 query。只有 Query Registry 中注册、拥有 label builder 和 choice schema 的文本 query 才能参与训练和正式推理。

### 4.4 Choice Encoder

Choice 是问题允许的离散答案。

二值示例：

```text
Query:
  future_return_positive, H=7

Choices:
  no
  yes
```

有序区间示例：

```text
Query:
  future_return_bucket, H=7

Choices:
  return < -5%
  -5% <= return < 0
  0 <= return < 5%
  return >= 5%
```

动作示例：

```text
Query:
  position_action

Choices:
  hold
  reduce_25
  reduce_50
  exit
```

第一版 choice 使用结构化字段和可学习 embedding，不声称理解任意自然语言 choice。

## 5. Query Taxonomy

### 5.1 收益方向

```text
return_direction(H, threshold)
```

示例：

```text
未来 3 日收益是否大于 0？
未来 7 日收益是否大于 5%？
未来 20 日收益是否低于 -5%？
```

### 5.2 收益区间

```text
return_bucket(H, boundaries)
```

推荐首版：

```text
[-inf, -5%)
[-5%, 0)
[0, 5%)
[5%, +inf)
```

后续可比较：

1. 固定经济阈值。
2. 训练期同日分位区间。
3. 波动率标准化收益区间。

不得在 OOS 回测后修改 bucket 再将同一区间称为新 OOS。

### 5.3 路径风险

```text
max_drawdown_bucket(H)
downside_event(H, threshold)
path_volatility_bucket(H)
```

示例：

```text
未来 7 日最大回撤是否低于 -10%？
未来 20 日是否出现单日大跌？
未来路径属于低波、正常、高波哪个区间？
```

### 5.4 路径形态

```text
path_shape(H)
```

Choices 示例：

```text
steady_up
spike_then_fade
sideways
steady_down
drawdown_then_recover
```

路径形态必须由可复算规则或聚类协议生成，不允许人工看图打标签。

### 5.5 可成交性

```text
entry_fillability(offset=1)
exit_fillability(H)
```

首个优先任务：

```text
T+1 是否能按 close 买入？

Choices:
  fillable
  suspended
  limit_up
  price_jump
```

该 query 只能使用 T 日及以前的特征。真实 T+1 状态只作为训练标签。

### 5.6 买入决策

空仓状态：

```text
Query:
  should_enter

Choices:
  no_trade
  enter_small
  enter_normal
```

该任务不能只由未来收益定义，还应明确成本、可成交性和风险阈值。

### 5.7 持仓动作

持仓状态必须额外输入：

```text
entry_price
current_pnl
holding_days
peak_pnl
drawdown_from_peak
current_reward_rank
score_change
position_weight
portfolio_concentration
available_cash
```

Query：

```text
position_action
```

Choices：

```text
hold
reduce_25
reduce_50
exit
```

动作标签必须来自固定执行规则下的反事实 utility，不能简单使用“未来下跌就是卖出”。

### 5.8 组合级决策

远期 query：

```text
portfolio_deployment
portfolio_rebalance
risk_budget
```

Choices 示例：

```text
cash_20
cash_40
cash_60
cash_80
cash_100
```

组合级任务不能只输入单票 latent，必须读取组合、市场和持仓集合状态。

## 6. 第一版模型结构

### 6.1 推荐结构

```text
Frozen AE state tokens: [B, 27, 128]

Query tokens:
  family
  horizon
  threshold
  target type
  execution contract
  state scope

Choice tokens:
  one token per choice

Shared Decision Transformer:
  [STATE] + [QUERY] + [CHOICE_1 ... CHOICE_K]

Output:
  one logit per choice
```

推荐第一版维持当前模型规模：

```text
d_model = 768
layers = 7
heads = 12
MLP ratio = 4
dropout = 0.05
```

第一轮不以扩大参数量作为变量。

### 6.2 Query 计算复用

同一个 state 可以并行回答多个 query：

```text
一次 AE encode
-> 多个 query/choice batch
-> 多个概率分布
```

必须避免为每个 query 重复运行 AE。

### 6.3 不采用固定多 head 作为长期接口

固定多 head 可以作为对照：

```text
return_head
risk_head
fillability_head
```

但长期主接口应当由 query 指定任务。固定 head 数量不应随 query 数量线性增长。

## 7. 训练目标

### 7.1 Choice 分类损失

基本损失：

```text
L_choice = CrossEntropy(choice_logits, target_choice)
```

它负责学习同一股票、同一 query 下的结果概率。

### 7.2 概率校准损失

需要同时记录并可选加入：

```text
Brier score
NLL
ECE
reliability diagram
```

可选训练项：

```text
L_calibration = Brier(choice_probabilities, target_one_hot)
```

不允许仅凭 accuracy 声称概率可信。

### 7.3 保留 Pairwise 排序

当前 pairwise 已有正式交易证据，因此第一版不能直接移除。

从 query-choice 概率导出一个任务 score：

```text
expected_return_score =
  sum_j P(choice_j) * representative_return(choice_j)
```

同日 A/B 排序损失：

```text
L_pair =
  softplus(-(score(A) - score(B)))
```

首版总损失：

```text
L_total =
  w_choice * L_choice
  + w_pair * L_pair
  + w_calibration * L_calibration
```

首轮必须包含消融：

```text
choice only
pairwise only
choice + pairwise
```

### 7.4 有序 Choice Loss

收益 bucket、回撤 bucket 等 choice 天然有顺序。后续可比较：

```text
普通 CE
ordinal cumulative loss
Earth Mover / Wasserstein-style ordinal loss
```

首版先用普通 CE，避免同时改变接口和 loss。

### 7.5 多 Query 采样

一只股票可以生成多个 query，但这不会增加独立市场日期数量。

训练采样需要控制：

1. 每个 state 每轮只抽取有限 query，避免同一状态被重复几十次。
2. 稀有 query 不得因 oversampling 主导 shared representation。
3. 每个 query family 独立记录 loss、accuracy、NLL 和 Brier。
4. 不用简单的样本行数掩盖有效日期数量不足。

## 8. 标签与信息时点

### 8.1 统一时间定义

若 signal date 为 T：

```text
特征截止：T close
最早可执行买入：T+1
未来收益：从 T+1 可执行价格起算
```

每个 query 必须携带明确 horizon 和 entry/exit 口径。

### 8.2 Query 与标签一一对应

禁止以下混用：

```text
query 问 H7，标签使用 H20
query 问 close-to-close，标签使用 open-to-close
query 问固定 H7，回测再事后选择 H20
```

### 8.3 可成交性

第一版至少显式记录：

```text
entry_fillable
entry_reject_reason
exit_fillable
exit_reject_reason
```

收益 query 可以选择：

1. 只在可成交样本上监督收益，同时单独训练 fillability query。
2. 将不可成交视为独立 choice，而不是收益为 0。

不得把未来不可成交股票静默删除后，再把模型概率解释成全市场无条件概率。

## 9. 从概率到决策

### 9.1 原子概率不直接等于交易 score

模型输出示例：

```text
P(H7 return >= 5%) = 0.31
P(H7 return < 0) = 0.22
P(H7 MDD < -10%) = 0.14
P(T+1 unfillable) = 0.08
```

策略代码显式组合：

```text
score =
  expected_return
  - lambda_downside * downside_probability
  - lambda_fill * unfillable_probability
```

组合公式和权重属于策略参数，不属于隐藏在模型内部的语义。

### 9.2 初期不训练一个统一 opaque utility

首版不直接让模型输出：

```text
universal_financial_score
```

原因是这种 scalar 会再次混合收益、风险、成交和持仓动作，回到当前 Reward Model 的瓶颈。

### 9.3 Abstain

当概率不确定时，可引入：

```text
abstain
insufficient_edge
```

但 abstain 不能成为回避亏损标签的捷径，必须有明确成本和覆盖率指标。

## 10. 评价体系

### 10.1 Query 层

每个 query family 独立报告：

```text
accuracy
macro F1
ROC-AUC / PR-AUC
NLL
Brier
ECE
reliability curve
coverage
```

### 10.2 横截面排序层

从 choice probabilities 派生 score 后报告：

```text
RankIC mean / median
positive RankIC ratio
Top1 / Top5 / Top10 / Top20 excess
BottomK direction control
score 与现有 Reward baseline 的相关性
```

### 10.3 可成交性

```text
T+1 fill rate
price_jump rejects
limit_up rejects
suspended rejects
T+H exit delay / rejects
```

### 10.4 正式账户层

必须通过 QuantX 完整回测：

```text
total return
annual return
max drawdown
Sharpe
profit factor
win rate
trade count
total cost
average holding days
rejection counts
annual returns
rolling drawdown
```

## 11. 时间切分纪律

当前 2020-2026 区间已经被 Reward、Residual、Diffusion、Loop、PPO 和多组持有期实验反复查看。

因此：

1. 该区间只能用于历史诊断和兼容性复核。
2. 不得在该区间选择 query family、bucket、horizon、loss 权重、TopK 或组合公式后，再称其为新的独立 OOS。
3. Query-choice 第一轮应先在 pre-2020 的滚动切分内完成模型选择。
4. 真正正式结论需要新的尾部数据，或预先注册并严格执行的 walk-forward 评估。

建议开发切分：

```text
Fold A:
  train <= 2016
  validation = 2017
  test = 2018

Fold B:
  train <= 2017
  validation = 2018
  test = 2019
```

在两个历史 fold 上冻结：

```text
query schema
choice buckets
loss weights
checkpoint rule
TopK
组合公式
```

之后才能进入新的尾部验证。

## 12. 最小实验路线

### 12.1 E0：Query/Choice 数据合同

状态：`planned`

目标：

1. 定义 typed query schema。
2. 定义 choice schema 和版本号。
3. 从现有 target memmap 生成 query labels。
4. 审计 query、choice、label 和信息时点一致性。

不训练模型。

### 12.2 E1：H7 收益区间 Query

状态：`planned`

唯一 query family：

```text
return_bucket(H=7)
```

Choices：

```text
< -5%
-5% ~ 0
0 ~ 5%
>= 5%
```

模型：

```text
同一 frozen AE
同一 49.73M Transformer 规模
冻结文本 Query/Choice Encoder embedding
query + choices 替换固定 scalar head
```

对照：

```text
当前 H7 pairwise baseline
choice-only
choice + pairwise
```

E1 是第一项应真正实施的实验。

### 12.3 E2：多 Horizon 收益 Query

状态：`blocked_by_E1`

Query：

```text
return_direction H3/H7/H10/H20
return_bucket H3/H7/H10/H20
```

目标：

1. 验证 query embedding 是否真正区分 horizon。
2. 检查多任务是否提升 H7，而不是负迁移。
3. 检查未单独训练或低频训练的 horizon 是否有组合泛化。

### 12.4 E3：风险和可成交性 Query

状态：`blocked_by_E1`

新增：

```text
max_drawdown_bucket
downside_event
entry_fillability
path_shape
```

重点验证 Residual Reward 暴露出的高收益、高跳空、高回撤问题能否被概率分解。

### 12.5 E4：持仓动作 Query

状态：`blocked_by_E3`

输入持仓状态，输出：

```text
hold
reduce_25
reduce_50
exit
```

先用固定执行反事实 utility 做 supervised action-value learning，不直接进入 PPO。

### 12.6 E5：组合级 Query

状态：`blocked_by_E4`

研究组合部署、现金比例和风险预算。该阶段必须使用集合状态，不得将单票决策简单独立相加。

## 13. E1 预注册比较合同

### 13.1 唯一允许变化

```text
固定 scalar pairwise head
->
query + return-bucket choices
```

以下保持一致：

```text
数据 root
frozen AE checkpoint
特征
时间切分
模型参数量级
训练 epoch
optimizer
有效 batch
随机种子
QuantX universe
执行成本
持有期
TopK
```

### 13.2 必须报告

```text
choice accuracy / macro F1
NLL / Brier / ECE
H7 expected-return RankIC
Top10 H7 可执行净收益
T+1 fill rate
正式 QuantX return / MDD / Sharpe
与 baseline score 的相关性
```

### 13.3 晋级条件

E1 只有在以下至少一项成立且其他关键指标不发生不可接受退化时才进入 E2：

1. 概率校准明显优于固定 scalar score 的后验映射。
2. H7 TopK 可执行净收益高于 pairwise baseline。
3. 正式 QuantX Sharpe 提升且最大回撤不恶化。
4. Choice probabilities 对收益幅度和左尾风险提供 baseline scalar 无法表达的稳定分层。

不能仅凭分类 accuracy 提升晋级。

## 14. 失败模式

### 14.1 Query 只是 task id

如果模型只记住不同 query 的固定偏置，而没有利用 state-query 交互，则它等价于多个固定 head。

需要检查：

```text
交换 horizon query 后输出是否合理变化
同一 state 对不同 query 的概率是否有结构
query embedding 是否被模型实际使用
```

### 14.2 多 Query 制造伪样本量

同一股票日期生成 20 个 query，不代表新增 20 个独立样本。必须以 signal date 为统计单位评估稳定性。

### 14.3 Choice 不平衡

固定收益区间可能高度不平衡。禁止只通过全局 oversampling 得到虚假概率；校准集必须保留自然分布。

### 14.4 概率不校准

Softmax 数值不天然等于真实概率。若 reliability curve 和 Brier 不通过，不得在策略中把 0.8 解释为 80%。

### 14.5 Query/执行错位

训练问 H7、回测选择 H20，或问 T+1 close、实际按 open 执行，都属于任务合同失效。

### 14.6 多任务负迁移

风险和可成交性 query 可能压制收益表示。必须同时保留：

```text
single-query H7
multi-query shared model
```

不能只报告 multi-query 最终结果。

### 14.7 Action label 使用未来最优动作

止盈止损任务容易退化为 hindsight oracle。必须用固定反事实执行合同，并明确其只能作为 supervised value target。

## 15. Artifact 规范

建议未来实验根目录：

```text
tmp/jev-like-financial-decision-v1/
```

建议结构：

```text
query_schema/
  query_schema_v1.json
  choice_schema_v1.json

data/
  query_label_manifest.json
  query_labels_*.parquet
  folds/
  scalers/

runs/
  <run_id>/
    checkpoints/
    logs/train_metrics.jsonl
    tensorboard/
    train_manifest.json
    query_metrics/
    score_artifacts/

reports/
  experiment_registry.json
  formal_quantx_summary.json
```

每个 checkpoint 必须持久化：

```text
query schema version
choice schema version
AE checkpoint
model config
task weights
label contract
time split
random seed
code/version identity
```

## 16. 文档更新协议

本文件是 living spec。每次实验必须追加一个独立小节，至少包含：

```text
日期
实验 ID
状态
假设
唯一变量
训练合同
query/choice schema
checkpoint 选择规则
离线结果
QuantX 结果
反事实
结论
下一假设
工件路径
```

实验状态只允许：

```text
planned
running
partial
diagnostic_only
retained_candidate
rejected
promoted
```

失败实验不得删除或改写成成功叙事。

## 17. 当前决策

截至 2026-09-20：

1. 保留现有 frozen-AE Multi-task Reward baseline 作为固定对照。
2. 保留 Residual Reward v1 作为高收益、高左尾风险的诊断因子。
3. 不继续在已查看的 2020-2026 区间搜索 Residual TopN、融合权重或持有期。
4. Jev-like 第一轮只做带文本 embedding 的 H7 return-bucket query，并保留 pairwise 辅助目标。
5. 第一轮冻结文本编码器和 AE，不开放任意未注册文本 query，不加入持仓动作，不扩大 State Backbone。
6. E1 通过后，再扩展多 horizon、风险、可成交性和 action choices。

## 18. v0.2 设计冻结点

本轮设计冻结以下方向：

1. Query 和 Choice 文本 embedding 是正式模型输入。
2. 精确金融语义由 typed structured fields 兜底。
3. State Backbone 从现有 Multi-task Reward checkpoint 迁移。
4. Query/Choice Decoder 独立且轻量，支持上千 query 分块执行。
5. Query Registry 和 Label Builder Registry 是模型能力的唯一正式入口。
6. E1 只实现 H7 return-bucket，不同步扩展其他 query family。

## 19. v0.2 实现级总览

### 19.1 最终目标接口

模型正式接口定义为：

```text
FinancialDecisionModel(
    state,
    queries,
    choices,
    masks,
) -> choice_logits
```

张量合同：

```text
state_tokens:
  [B, 27, 128]

query_text_embedding:
  [B, Q, D_text]

query_structured_features:
  [B, Q, D_query_struct]

choice_text_embedding:
  [B, Q, C, D_text]

choice_structured_features:
  [B, Q, C, D_choice_struct]

choice_mask:
  [B, Q, C]

choice_logits:
  [B, Q, C]
```

其中：

```text
B = unique financial states in the batch
Q = queries sampled for each state
C = padded maximum choices in this batch
```

不同 query 可以有不同数量的 choices，通过 `choice_mask` 屏蔽 padding。

### 19.2 核心约束

1. 一个 state 的 AE encode 只能执行一次。
2. 一个 state 的 49.73M state backbone 只能执行一次。
3. Query/Choice text encoder 不得在主训练循环内重复前向。
4. Query 和 Choice embedding 必须预计算并通过版本化 artifact 加载。
5. 新增 query 不允许新增独立输出 head。
6. Query 的 label builder 必须由 registry 注册，禁止从 YAML 动态 import 任意 Python 路径。
7. 所有 query 必须有稳定的 semantic ID、版本、choice schema 和 label contract。

## 20. 为什么需要拆分 State Backbone 与 Decision Decoder

### 20.1 不可采用的朴素结构

最直接的写法是：

```text
[STATE TOKENS] + [QUERY TOKEN] + [CHOICE TOKENS]
-> 完整 7 层 50M Transformer
-> logits
```

如果每个 state 有 1000 个 query，该结构会把相同的 27 个 state token 重复计算 1000 次，无法扩展。

### 20.2 推荐结构

正式模型拆成两部分：

```text
Frozen AE
  -> State Backbone
  -> State Memory

Text Query / Choices
  -> Query/Choice Decoder
  -> Choice Logits
```

计算图：

```text
raw state
  |
  v
Frozen AE encode
  |
  v
27 x 128 latent
  |
  v
State Backbone, 7 layers
  |
  +----------------------+
  | cached state memory  |
  +----------------------+
        |       |       |
        v       v       v
      Query1  Query2  QueryN
        |       |       |
      choices choices choices
        |       |       |
        +--- lightweight decoder --->
                  logits
```

### 20.3 State Backbone

State Backbone 直接迁移当前 `FrozenAERewardTransformer`：

```text
input_proj
cls_token
position
7 x RewardTransformerBlock
norm
```

移除：

```text
固定 Linear head
固定 score_heads = return/sharpe/drawdown
```

输出：

```text
state_memory: [B, 28, 768]
state_summary: [B, 768]
```

其中 28 个 token 为：

```text
1 个 state CLS
27 个 AE latent token
```

### 20.4 Query/Choice Decoder

每个 query 构造：

```text
query_token =
  QueryTextProjection(text_embedding)
  + QueryStructuredEncoder(structured_fields)
  + query_type_embedding
```

每个 choice 构造：

```text
choice_token =
  ChoiceTextProjection(text_embedding)
  + ChoiceStructuredEncoder(structured_fields)
  + choice_type_embedding
```

轻量 Decoder 首版建议：

```text
d_model = 768
layers = 2
heads = 12
MLP ratio = 2
dropout = 0.05
```

每层包括：

```text
1. 同一 query 内 query/choices self-attention
2. query/choices 对 state_memory 的 cross-attention
3. MLP
```

每个 choice token 的最终输出经同一个共享标量 head：

```text
logit_j = SharedChoiceLogitHead(choice_hidden_j)
```

禁止为每个 query 创建独立 classifier。

### 20.5 Query 分块

推理上千 query 时：

```text
state_memory = encode_state_once(state)

for query_chunk in chunks(query_registry, size=32 or 64):
    logits = decode(state_memory, query_chunk)
```

State Backbone 只运行一次，Query Decoder 按 chunk 运行。

首版必须记录：

```text
state encoding latency
per-query decoder latency
queries / second
choices / second
GPU peak memory
```

## 21. 文本 Query Encoder

### 21.1 第一版模型

首版默认候选：

```text
model_id: BAAI/bge-small-zh-v1.5
framework: transformers.AutoTokenizer + AutoModel
pooling: CLS
embedding_dim: 512
normalize_embeddings: true
trainable: false
max_length: 128
```

官方模型卡：

```text
https://huggingface.co/BAAI/bge-small-zh-v1.5
```

当前 `test` 环境已有：

```text
transformers 4.57.3
```

当前没有：

```text
sentence_transformers
```

因此实现不增加 `sentence-transformers` 依赖，直接使用 `transformers`。

当前未确认本机已有 BGE snapshot。正式实现前必须显式准备模型权重并记录：

```text
model_id
resolved revision / commit
tokenizer files checksum
model files checksum
pooling contract
normalization contract
```

训练入口禁止隐式联网下载文本模型。若缓存不存在，必须在训练前明确失败。

### 21.2 为什么文本 embedding 不能是唯一语义

以下语义不能只依赖文本模型自行理解：

```text
H=7
threshold=0.05
entry_offset=1
左闭右开区间
T+1 close 执行
```

因此 Query embedding 使用双通道：

```text
query_embedding =
  Project(text_embedding)
  + StructuredQueryEncoder(numeric_and_enum_fields)
```

Choice embedding 同理：

```text
choice_embedding =
  Project(choice_text_embedding)
  + StructuredChoiceEncoder(boundaries_and_flags)
```

文本负责通用语义，结构化字段负责精确金融合同。

### 21.3 文本规范

每个 query 必须包含：

```text
canonical_text_zh
canonical_text_en，可选
description
```

E1 只使用 `canonical_text_zh`。

示例：

```text
从信号日后的第一个交易日收盘价作为起点，
该股票未来七个交易日的收盘收益属于哪个区间？
```

Choice 示例：

```text
收益低于负百分之五
收益介于负百分之五和零之间
收益介于零和正百分之五之间
收益不低于正百分之五
```

文本发生语义修改时必须提升 query version，不能静默覆盖。

### 21.4 Embedding Cache

文本编码在训练前一次性完成：

```text
query_text_embeddings.npy
choice_text_embeddings.npy
text_embedding_manifest.json
```

体量估算：

```text
1000 query
x 平均 5 choices
x 512 dims
x float32
< 12 MB
```

文本 embedding 不是训练吞吐瓶颈。

## 22. Query Registry

### 22.1 Registry 是模型能力边界

所有可训练 query 必须来自一个版本化 registry：

```text
query_registry_v1.yaml
```

训练代码不接受任意临时字符串作为带监督 query。

### 22.2 QuerySpec Schema

```yaml
query_id: return.bucket.h7.t1_close.v1
version: 1
enabled: true

family: return_bucket
state_scope: security
label_builder: return_bucket

canonical_text_zh: >
  从信号日后的第一个交易日收盘价作为起点，
  该股票未来七个交易日的收盘收益属于哪个区间？

parameters:
  horizon: 7
  entry_offset: 1
  entry_price: close
  exit_price: close
  target_kind: gross_return

choice_set:
  - choice_id: loss_large
    text_zh: 收益低于负百分之五
    lower: null
    upper: -0.05
    lower_inclusive: false
    upper_inclusive: false

  - choice_id: loss_small
    text_zh: 收益介于负百分之五和零之间
    lower: -0.05
    upper: 0.0
    lower_inclusive: true
    upper_inclusive: false

  - choice_id: gain_small
    text_zh: 收益介于零和正百分之五之间
    lower: 0.0
    upper: 0.05
    lower_inclusive: true
    upper_inclusive: false

  - choice_id: gain_large
    text_zh: 收益不低于正百分之五
    lower: 0.05
    upper: null
    lower_inclusive: true
    upper_inclusive: false

training:
  family_weight: 1.0
  query_sampling_weight: 1.0
  loss: categorical_ce
  pairwise_auxiliary: expected_return

metrics:
  - accuracy
  - macro_f1
  - nll
  - brier
  - ece
  - expected_return_rankic
```

### 22.3 Query ID 稳定性

`query_id` 一旦用于正式 checkpoint 就不可修改语义。

以下任何变化都必须生成新 version：

```text
horizon
threshold
entry/exit price
choice boundaries
label builder
valid mask
execution semantics
canonical text 的实质语义
```

### 22.4 Query Factory

成百上千 query 不应手写。

Registry 支持 factory：

```yaml
factory_id: return_direction_grid_v1
family: return_direction

grid:
  horizon: [3, 5, 7, 10, 20, 30]
  threshold: [-0.10, -0.05, 0.0, 0.05, 0.10]

choices:
  - false
  - true
```

编译后生成 30 个不可变 QuerySpec。

另一个示例：

```yaml
factory_id: max_drawdown_grid_v1

grid:
  horizon: [5, 7, 10, 20, 30]
  threshold: [-0.05, -0.10, -0.15, -0.20]
```

Query Factory 负责规模化，Label Builder 负责语义正确性。

### 22.5 Registry 编译产物

```text
compiled_query_registry_v1.json
query_id_to_index.json
query_registry_manifest.json
```

Manifest 必须记录：

```text
source YAML checksum
compiled JSON checksum
query count
choice count distribution
families
horizons
label builder versions
text embedding manifest
```

## 23. Label Builder 协议

### 23.1 设计原则

用户提出“每个 query 有自己的 choices 和 label 构建方法”，实现上采用：

```text
一个 QuerySpec
-> 一个受控 Label Builder 类型
-> 一组 builder parameters
```

不为每个 query 写一个 Python 类。

同一 family 的大量 query 共享 builder：

```text
return_direction builder
return_bucket builder
barrier_event builder
max_drawdown builder
fillability builder
position_action builder
```

### 23.2 Builder 接口

```python
class QueryLabelBuilder(Protocol):
    family: str
    version: int

    def validate(self, query: QuerySpec, outcome_schema: OutcomeSchema) -> None:
        ...

    def build(
        self,
        outcome_batch: OutcomeBatch,
        query: QuerySpec,
    ) -> QueryLabelBatch:
        ...
```

输出：

```text
target_choice_index: [B]
valid_mask: [B]
sample_weight: [B]
continuous_value: [B]，可选，仅用于指标和 pairwise
audit_flags: dict
```

### 23.3 Builder Registry

```text
LABEL_BUILDERS = {
  "return_direction": ReturnDirectionLabelBuilder,
  "return_bucket": ReturnBucketLabelBuilder,
  "barrier_event": BarrierEventLabelBuilder,
  "max_drawdown_bucket": MaxDrawdownLabelBuilder,
  "entry_fillability": EntryFillabilityLabelBuilder,
}
```

YAML 只能引用上述注册名称。

### 23.4 区间完整性

Choice 区间 builder 必须验证：

```text
无重叠
无空洞
边界单调
每个 valid 样本恰好属于一个 choice
NaN / 无法结算样本被 valid_mask 排除
```

### 23.5 不可成交标签

收益与可成交性分开建模：

```text
return_bucket:
  描述可观察价格路径结果

entry_fillability:
  描述 T+1 执行状态
```

不能把所有不可成交样本都隐式映射为收益 0。

若需要交易 utility query，应单独定义：

```text
tradable_return_bucket
```

并明确：

```text
不可成交是独立 choice
或样本无效
```

### 23.6 Action Label Builder

持仓动作不能使用单一 hindsight 标签。

建议 builder：

```text
CounterfactualActionUtilityBuilder
```

它对每个 choice 执行固定模拟：

```text
hold
reduce_25
reduce_50
exit
```

输出每个动作的未来净 utility，再生成：

1. 最优动作 choice。
2. 动作 utility bucket。
3. 动作间 pairwise preference。

所有动作使用同一交易成本、T+1、价格限制和持仓合同。

## 24. Outcome Store：支持上千 Query 的关键

### 24.1 禁止物化 state x query 全矩阵

假设：

```text
10,000,000 states
x 1000 queries
```

直接保存 target choice 将产生 100 亿标签，不可接受。

### 24.2 保存充分统计量

大量 query 都来自相同未来路径。应保存可复用 outcome primitives：

```text
future_close_return_path: [N, 30]
future_high_return_path:  [N, 30]
future_low_return_path:   [N, 30]
future_path_mdd:          [N, H_set]
future_path_volatility:   [N, H_set]
future_path_sharpe:       [N, H_set]
entry_execution_state:    [N]
exit_execution_state:     [N, H_set]
```

Query label 在 batch 中由 builder 从 primitives 编译。

### 24.3 首版复用现有数据

E1 只需要：

```text
现有 target path memmap
candidate index
fold
```

不新增全量未来高低价 store。

E2/E3 再增加 high/low、drawdown 和 fillability primitives。

### 24.4 Outcome Manifest

```text
outcome_store_manifest.json
```

必须记录：

```text
row identity
shape
dtype
target start offset
calendar
price mode
horizon coverage
feature/label checksum
```

## 25. Dataset 与 Batch 组织

### 25.1 两条训练数据流

概率校准和 pairwise 排序不能使用同一个重采样分布。

正式训练包含两条流：

```text
Natural Query Stream
  保持 validation/test 的自然类别分布
  训练 choice probability

Same-date Pair Stream
  构造 A/B 排序样本
  训练横截面 ranking
```

### 25.2 Natural Query Stream

样本单位：

```text
state_row_id
query_id
target_choice
valid_mask
sample_weight
```

训练集允许 class-balanced sampler，但必须保留一个不重采样的 calibration stream。

### 25.3 Pair Stream

样本单位：

```text
signal_date
query_id
row_a
row_b
target_choice_a
target_choice_b
preference_direction
continuous_value_a
continuous_value_b
```

约束：

```text
A/B 必须同日
A/B 必须使用同一个 query_id
pair score 必须由同一个 choice probability 派生规则生成
```

### 25.4 Query Sampler

支持上千 query 时，每个 state 不在每个 epoch 回答所有问题。

建议：

```text
queries_per_state = 4，E1
queries_per_state = 8，E2
```

采样概率：

```text
P(query) =
  family_weight
  / active_query_count_in_family
  * query_sampling_weight
```

这样新增 500 个同 family query 不会自动把该 family 梯度放大 500 倍。

### 25.5 Batch Collation

Batch 先对 state 去重：

```text
unique_state_rows: [B]
query_assignments: [num_assignments]
```

执行：

```text
state_memory = encode(unique_state_rows)
logits = decode(
    state_memory[assignment_state_index],
    query_embedding,
    choice_embeddings,
)
```

同一 state 的多个 query 共享 state memory。

### 25.6 Choice Padding

不同 query 的 choice 数不同：

```text
binary: 2
return bucket: 4
path shape: 5
action: 4
```

Collate 到：

```text
max_choices_in_batch
```

Padding logits 必须在 softmax 前设为 `-inf`。

## 26. Loss 设计

本节的几个 loss 不应理解为彼此无关的技巧。它们共同约束同一个对象：

```text
node = (state, query, choice, logit)
```

Choice CE 校准同一 state/query 内不同 choices 的相对关系；Pairwise 校准同一 query/choice 在不同股票状态之间的相对关系；Brier 校准概率数值与真实发生频率；Consistency 校准不同 query 之间的逻辑关系。

### 26.1 基本总损失

```text
L_total =
  1.0 * L_choice_ce
  + 0.5 * L_pairwise
  + 0.1 * L_brier
  + w_consistency * L_cross_query_consistency
```

E1：

```text
w_consistency = 0
```

先验证基础结构。

### 26.2 Choice CE

按 family 归一：

```text
L_choice_ce =
  sum_family family_weight * mean(loss_in_family)
```

不能按全 batch 样本数直接平均，否则 query 数量最多的 family 主导训练。

### 26.3 Pairwise Score

Return bucket 的 expected return：

```text
score_return =
  sum_choice P(choice) * train_fold_bucket_mean(choice)
```

Bucket mean 只能由训练 fold 计算并写入 query manifest。

同日 pair：

```text
L_pairwise =
  softplus(-(score_A - score_B) / temperature)
```

### 26.4 Brier

```text
L_brier =
  mean(sum_choice (P(choice) - one_hot(choice))^2)
```

Brier 用于改善概率质量，但权重保持较小，避免牺牲 TopK 排序。

### 26.5 跨 Query 一致性

E2 后启用。

示例：

```text
P(return > 5%)
<=
P(return > 0)
```

Return bucket 与 direction query：

```text
P(return > 0)
≈
sum(P(positive buckets))
```

一致性损失：

```text
L_consistency =
  threshold_monotonicity
  + bucket_marginal_consistency
```

Horizon 之间不强制单调，因为短期和长期收益事件没有天然包含关系。

### 26.6 Query Paraphrase Consistency

同一 semantic query 可有多个等价文本模板：

```text
未来七日收益是否为正？
七个交易日后该股票是否上涨？
```

同一 query_id 的 paraphrase 应输出一致。

该功能属于 E2，不进入 E1。

## 27. Checkpoint 迁移

### 27.1 Warm Start 来源

E1 从当前 Multi-task baseline：

```text
reward-multitask-7d-return005-risk005-top50-10ep-gate-20260726-172019
checkpoints/reward_epoch_010.pt
```

加载：

```text
input_proj
cls_token
position
blocks
norm
```

不加载：

```text
固定 return/sharpe/drawdown head
```

### 27.2 学习率组

建议：

```text
Frozen AE:
  lr = 0

State Backbone:
  lr = 3e-5

Query/Choice Decoder:
  lr = 1.5e-4

Text Encoder:
  lr = 0
```

### 27.3 两阶段训练

```text
epoch 1:
  freeze State Backbone
  只训练 Query/Choice Decoder

epoch 2-10:
  解冻 State Backbone
  使用较低学习率联合训练
```

这样可以避免随机初始化的 query decoder 在首轮破坏已有 state representation。

### 27.4 架构归因边界

Warm-start E1 只能回答：

```text
在已有 Reward representation 上，
query-choice 接口是否能产生更好的概率与选股结果？
```

它不能证明：

```text
query-choice 从头训练一定优于 fixed head。
```

若 E1 成功，后续需补 matched-from-scratch 对照。

## 28. Forward API

### 28.1 模型接口

```python
class JevLikeFinancialDecisionModel(nn.Module):
    def encode_state(
        self,
        ae_tokens: Tensor,  # [B, 27, 128]
    ) -> StateMemory:
        ...

    def decode_queries(
        self,
        state_memory: StateMemory,
        query_text_embedding: Tensor,       # [B, Q, 512]
        query_structured: Tensor,           # [B, Q, Dq]
        choice_text_embedding: Tensor,      # [B, Q, C, 512]
        choice_structured: Tensor,           # [B, Q, C, Dc]
        choice_mask: BoolTensor,             # [B, Q, C]
    ) -> Tensor:                             # [B, Q, C]
        ...

    def forward(...) -> Tensor:
        state_memory = self.encode_state(...)
        return self.decode_queries(state_memory, ...)
```

### 28.2 输出语义

模型只输出 logits：

```text
choice_logits
```

Softmax、temperature calibration、expected return 和策略 score 在模型外执行。

### 28.3 不输出统一 scalar

模型 API 不提供：

```text
model.universal_score()
```

每个 scalar 必须声明来源 query 和组合公式。

## 29. Inference Artifact

### 29.1 Query Prediction Store

上千 query 不适合单个超宽 parquet。

采用长表或按 query 分区：

```text
predictions/
  query_id=return.bucket.h7.t1_close.v1/
    part-*.parquet
  query_id=entry.fillability.t1_close.v1/
    part-*.parquet
```

字段：

```text
signal_date
instrument
query_id
choice_id
logit
probability
```

### 29.2 Derived QuantX Score

QuantX 只读取最终需要的少量派生 score：

```text
jev_expected_return_h7
jev_probability_gain_ge_5pct_h7
jev_downside_adjusted_h7
jev_entry_adjusted_h7
```

派生过程独立写入：

```text
decision_formula.json
score parquet
score manifest
```

不得把 1000 个原始 query 列直接塞入 QuantX selector。

### 29.3 Manifest

预测 manifest 必须包含：

```text
checkpoint
AE checkpoint
text encoder model/revision/checksum
query registry checksum
choice registry checksum
calibration parameters
query IDs
date coverage
row coverage
DDP shards
```

## 30. 文件级实现方案

未来代码落在独立研究目录：

```text
tmp/jev-like-financial-decision-v1/
```

建议文件：

```text
query_registry_v1.yaml
compile_query_registry_v1.py
jev_query_schema_v1.py
jev_text_encoder_v1.py
jev_outcome_store_v1.py
jev_label_builders_v1.py
jev_query_dataset_v1.py
jev_model_v1.py
jev_losses_v1.py
train_jev_financial_decision_ddp_v1.py
infer_jev_financial_decision_v1.py
derive_jev_quantx_scores_v1.py
evaluate_jev_queries_v1.py
tests/
```

职责：

| 文件 | 单一职责 |
| --- | --- |
| `jev_query_schema_v1.py` | QuerySpec、ChoiceSpec、schema validation |
| `compile_query_registry_v1.py` | 展开 factories，生成稳定 query IDs 和 manifest |
| `jev_text_encoder_v1.py` | 冻结文本编码器、embedding cache、checksum |
| `jev_outcome_store_v1.py` | 复用未来路径充分统计量 |
| `jev_label_builders_v1.py` | 受控 label builder registry |
| `jev_query_dataset_v1.py` | natural stream、pair stream、query sampler、collate |
| `jev_model_v1.py` | State Backbone + Query/Choice Decoder |
| `jev_losses_v1.py` | CE、Brier、pairwise、consistency |
| `train_jev_financial_decision_ddp_v1.py` | DDP 训练、验证、checkpoint |
| `infer_jev_financial_decision_v1.py` | State cache、query chunk、prediction store |
| `derive_jev_quantx_scores_v1.py` | 概率到 QuantX score 的显式公式 |
| `evaluate_jev_queries_v1.py` | 分类、校准、排序与覆盖率报告 |

### 30.1 对现有代码的最小依赖

复用：

```text
WTSPaths
WeakToStrongPathDataset
frozen AE loader
RewardTransformerBlock
RMSNorm
DDP helpers
market_score_artifact_v1
```

不修改：

```text
现有 Reward checkpoint
现有 Multi-task trainer 默认行为
QuantX engine
QuantX selector
```

### 30.2 不在旧训练脚本内继续堆分支

当前 `train_wts_ae_reward_transformer_ddp_v1.py` 已包含：

```text
endpoint
path_quality
gap_quality
multitask
residual_utility
loop preference
```

Jev-like 改变了 sample schema、模型 forward 和 artifact，因此新建独立入口比继续增加 `label_mode` 更高内聚。

## 31. E1 完整实现合同

### 31.1 Query

```yaml
query_id: return.bucket.h7.t1_close.v1
family: return_bucket
horizon: 7
entry_offset: 1
```

### 31.2 Choices

```text
0: return < -5%
1: -5% <= return < 0
2: 0 <= return < 5%
3: return >= 5%
```

### 31.3 State

```text
与当前 Multi-task baseline 完全相同：
过去 60 日 raw_relative
现有 scaler
现有 frozen AE
27 x 128 latent
```

### 31.4 模型

```text
State Backbone:
  从现有 epoch010 warm start
  7 layers, d_model 768, 12 heads

Query/Choice Decoder:
  2 layers, d_model 768, 12 heads

Text Encoder:
  frozen BAAI/bge-small-zh-v1.5
```

### 31.5 Loss

主 run：

```text
L =
  1.0 * categorical CE
  + 0.5 * same-date pairwise
  + 0.1 * Brier
```

对照：

```text
A: 当前固定 head pairwise baseline
B: Jev-like choice-only
C: Jev-like choice + pairwise + Brier
```

### 31.6 Pairwise 派生 score

每个 choice 的代表收益由 train fold 自然样本均值计算：

```text
mu_choice_j = mean(return | choice_j, train fold)
```

```text
expected_return =
  sum_j P(choice_j) * mu_choice_j
```

Pairwise 使用该 expected return。

### 31.7 训练切分

不直接使用已反复查看的 2020-2026 选择模型。

至少执行：

```text
Fold A:
  train <= 2016
  validation = 2017
  test = 2018

Fold B:
  train <= 2017
  validation = 2018
  test = 2019
```

只有两个 fold 方向一致，才导出 2020-2026 诊断 score。

### 31.8 训练资源

初始建议：

```text
6 GPU DDP
BF16
10 epochs
每 epoch checkpoint
TensorBoard
JSONL
固定 seed
```

### 31.9 选择规则

Checkpoint 只使用 validation：

```text
主指标:
  expected_return Top10 excess

必要 gate:
  NLL 不恶化
  Brier 不恶化
  ECE 可接受
  Top10 fill rate 报告完整
```

不根据 2020-2026 选择 checkpoint。

### 31.10 成功条件

E1 不是只看分类准确率。

成功至少需要：

1. Choice probabilities 有可解释的 reliability 分层。
2. `P(return >= 5%)` 随真实命中率单调上升。
3. Expected return score 的 Top10 excess 不低于固定 pairwise baseline。
4. Choice + pairwise 不弱于 choice-only 和固定 pairwise。
5. 两个历史 test fold 方向一致。

## 32. 上千 Query 的扩展计划

### 32.1 Core Query Set

E1 通过后，先扩展到约 32 个核心 query：

```text
return_direction:
  H = 3, 5, 7, 10, 20, 30
  threshold = 0, 5%

return_bucket:
  H = 3, 5, 7, 10, 20, 30

max_drawdown_event:
  H = 7, 10, 20, 30
  threshold = -5%, -10%, -20%

entry_fillability:
  T+1 close
```

### 32.2 Extended Query Set

核心 query 通过后，再扩展：

```text
不同收益阈值
不同 path event
barrier touch
冲高回落
低点修复
相对市场/行业收益
收益风险联合事件
```

规模约 100-300。

### 32.3 Large Query Set

达到上千 query 前必须满足：

```text
Query Registry 编译稳定
Label Builder 向量化
family-balanced sampler
query chunk inference
state cache
跨 query consistency
per-family calibration
```

Query 数量本身不是目标。只有新增 query 对 shared representation 或决策组合产生增量，才保留。

## 33. 测试计划

### 33.1 Schema

```text
query_id 唯一
choice_id 在 query 内唯一
区间无重叠无空洞
factory 展开确定性
registry checksum 稳定
```

### 33.2 Label Builder

```text
边界值归类
NaN/不可结算 mask
每个 valid 样本只有一个 target choice
T+1/H 时点正确
无未来特征进入 state
```

### 33.3 Text Embedding

```text
同一 registry 重建 embedding 完全一致
model revision/checksum 一致
embedding normalization 正确
训练入口无网络访问
```

### 33.4 Model

```text
variable choice count
choice mask
state encode only once
query chunk 与一次性推理结果一致
padding 不影响有效 logits
所有 choice 共享 logit head
```

### 33.5 Loss

```text
CE 梯度方向
Brier 梯度方向
pairwise expected-return 梯度方向
family normalization
masked query 不产生梯度
```

### 33.6 Checkpoint

```text
registry mismatch 拒绝加载
text encoder mismatch 拒绝加载
AE mismatch 拒绝加载
choice schema mismatch 拒绝加载
warm-start key coverage 审计
```

### 33.7 Inference

```text
DDP 按完整日期分片
query partition 无重复无遗漏
概率和为 1
artifact coverage
derived score 可复现
```

## 34. 开发顺序

后续实际实现按以下顺序，不并行扩大范围：

```text
Milestone 0:
  QuerySpec / ChoiceSpec
  registry compiler
  label builder protocol
  单元测试

Milestone 1:
  BGE embedding cache
  text encoder manifest
  structured query/choice encoder

Milestone 2:
  State Backbone checkpoint migration
  Query/Choice Decoder
  forward shape tests

Milestone 3:
  H7 return-bucket natural stream
  H7 same-date pair stream
  CE + pairwise + Brier

Milestone 4:
  2017/2018/2019 历史 fold
  calibration / ranking 报告

Milestone 5:
  full prediction export
  QuantX score derivation
  paired formal backtest
```

每个 milestone 完成后，先在本文追加：

```text
实际文件
测试
结果
偏差
是否进入下一阶段
```

再开始下一 milestone。

## 35. RLCD-like Calibration Graph

### 35.1 公开事实与工作假设

TypeSafe 的公开资料说明 RLCD 目标是让模型输出经过校准的决策概率，使相近的预测概率对应相近的真实正确频率。但公开资料没有给出足以复刻的 loss、采样器、reward、constraint graph 或优化器细节。

因此本文使用：

```text
RLCD-like Calibration Graph
```

作为本项目自己的工作假设，不声称它等同于 Jev 内部实现。

### 35.2 Pairwise 是相对校准

当前 Reward Model 的 Pairwise：

```text
L_pair(A, B) =
  -log sigmoid(score(A) - score(B))
```

它约束：

```text
在同一 query 语义下，
如果 A 的真实结果优于 B，
则 A 对目标 choice 的支持度应高于 B。
```

例如：

```text
Query:
  未来 H7 是否上涨？

A:
  H7 return = +8%

B:
  H7 return = -3%

Constraint:
  logit_yes(A) > logit_yes(B)
```

这确实是一种 calibration，但它是 **ordinal / relative calibration**。

### 35.3 Pairwise 不能单独产生绝对概率

只要排序保持：

```text
score'(x) = a * score(x) + b
a > 0
```

Pairwise 的方向可以不变，但：

```text
sigmoid(score'(x))
```

会得到完全不同的概率。

因此以下模型可能 Pairwise 同样正确：

```text
Model A:
  P(up | A) = 0.55
  P(up | B) = 0.45

Model B:
  P(up | A) = 0.99
  P(up | B) = 0.98
```

两者都满足：

```text
A > B
```

但 Model B 明显不能解释成可靠的上涨概率。

当前 baseline 的 logit center penalty 只消除公共 offset 漂移，仍不能证明 sigmoid score 是校准概率。

### 35.4 Calibration Graph 的节点

统一节点定义：

```text
CalibrationNode:
  state_row_id
  query_id
  choice_id
  logit
  probability
```

一个节点表达：

```text
模型在给定金融状态下，
对某个 query 的某个 choice 的支持程度。
```

### 35.5 Calibration Graph 的边

#### A. Outcome Choice Edge

同一 state、同一 query 内：

```text
correct choice > incorrect choices
```

示例：

```text
实际 H7 return = +7%

gain_large
  >
gain_small
loss_small
loss_large
```

Categorical CE 可以理解为同时构造正确 choice 对所有错误 choices 的偏好约束，并进行归一化。

#### B. Cross-state Pairwise Edge

同一 query、同一目标 choice，跨股票状态比较：

```text
如果 A 的真实连续值优于 B，
则支持目标 choice 的 score(A) > score(B)。
```

示例：

```text
return(A) = +8%
return(B) = +2%

expected_return_score(A)
  >
expected_return_score(B)
```

这保留当前 Reward Model 已验证的横截面排序能力。

#### C. Frequency Calibration Edge

概率必须匹配经验频率：

```text
预测 P(up) 约为 0.7 的样本集合，
真实上涨比例也应接近 0.7。
```

训练代理：

```text
Cross Entropy
Brier
```

评估：

```text
NLL
Brier
ECE
reliability curve
```

#### D. Threshold Monotonicity Edge

同一 state、同一 horizon：

```text
P(return >= 10%)
<=
P(return >= 5%)
<=
P(return >= 0%)
```

若违反该关系，说明不同 query 的概率语义不一致。

#### E. Bucket Aggregation Edge

方向 query 与 bucket query 必须一致：

```text
P(return > 0)
≈
P(0% <= return < 5%)
+ P(return >= 5%)
```

#### F. Equivalent Query Edge

同一个 semantic ID 的不同文本：

```text
未来七日是否上涨？
未来七个交易日收益是否为正？
```

其输出应一致。

#### G. Action Utility Edge

持仓动作 query：

```text
utility(hold) > utility(exit)
```

则：

```text
logit(hold) > logit(exit)
```

动作 utility 必须来自同一执行合同下的反事实模拟。

### 35.6 Relation Schema

训练数据不应只存 target class，还应允许编译关系：

```yaml
relation_type: cross_state_preference
query_id: return.bucket.h7.t1_close.v1

left:
  state_row_id: 123
  score_projection: expected_return

right:
  state_row_id: 456
  score_projection: expected_return

direction: left_greater
margin: 0.005
weight: 1.0
```

统一结构：

```text
CalibrationRelation:
  relation_type
  left_node_or_projection
  right_node_or_projection
  direction
  margin
  weight
  valid_mask
```

### 35.7 Relation Compiler

Label Builder 除了输出 target choice，还可以输出关系：

```python
QueryLabelBatch:
    target_choice_index
    valid_mask
    sample_weight
    continuous_value
    relations
```

Relation Compiler 将其转换为：

```text
choice preference
cross-state pair
threshold monotonicity
bucket consistency
action utility preference
```

模型训练器只处理统一 relation，不需要为每个 query family 手写训练循环。

### 35.8 E1 的 Calibration Graph

E1 只启用三种关系：

```text
1. Outcome Choice Edge
2. Cross-state Pairwise Edge
3. Frequency Calibration Edge
```

对应：

```text
L_E1 =
  1.0 * L_choice_ce
  + 0.5 * L_cross_state_pairwise
  + 0.1 * L_brier
```

E1 不启用：

```text
threshold monotonicity
bucket aggregation
paraphrase consistency
action utility
```

原因是 E1 只有一个 H7 bucket query，应先验证最小 Calibration Graph。

### 35.9 E2 之后的统一目标

```text
L_total =
  w_outcome * L_outcome_choice
  + w_rank * L_cross_state_rank
  + w_frequency * L_probability_calibration
  + w_logic * L_cross_query_logic
  + w_equivalence * L_query_equivalence
  + w_action * L_action_utility
```

所有权重必须按 relation family 归一，不能因为新增数百个 threshold query 而让某一类关系淹没其他任务。

### 35.10 这套设计对当前问题的解释

当前 Residual Reward 只拥有：

```text
Cross-state Pairwise Edge
```

它知道：

```text
A 的可执行 H7 utility 应高于 B
```

但不知道：

```text
P(A 会上涨) 是多少
P(A 会大幅回撤) 是多少
P(A 明天无法买入) 是多少
不同阈值 query 是否互相一致
```

因此它可以学到高收益排序，却同时暴露高回撤和高 `price_jump`。

Jev-like 模型的提升点不是简单增加更多分类 head，而是让这些相对关系、绝对概率和跨任务逻辑在同一 Calibration Graph 中共同约束 representation。

### 35.11 实现文件补充

文件级方案增加：

```text
jev_calibration_graph_v1.py
```

职责：

```text
CalibrationNode / CalibrationRelation schema
relation compiler
relation-family normalization
pairwise projection registry
cross-query consistency loss
relation audit metrics
```

测试至少覆盖：

```text
Pairwise 方向
Choice preference
概率频率分桶
阈值单调性
bucket 聚合一致性
同 family 新增 query 不改变总梯度权重
```

## 36. v0.1-v0.2 历史变更记录

### v0.2.1 - 2026-09-20

1. 将 Pairwise 明确定义为相对/序数校准，而非完整概率校准。
2. 增加 Calibration Node 和 Calibration Relation 统一抽象。
3. 定义 Outcome、Cross-state、Frequency、Threshold、Bucket、Equivalent Query 和 Action Utility 七类关系。
4. 定义 Relation Compiler 与 `jev_calibration_graph_v1.py` 实现边界。
5. 将 E1 的 CE + Pairwise + Brier 重新解释为最小 Calibration Graph。
6. 明确 RLCD-like 是本项目工作假设，不声称复刻 Jev 未公开训练算法。

### v0.2.0 - 2026-09-20

1. 将文档从研究路线图升级为实现级规格。
2. 将冻结文本编码器生成的 Query/Choice embedding 确定为正式模型输入。
3. 增加结构化 query/choice fields，保证 horizon、threshold、执行时点和区间边界精确可审计。
4. 定义可扩展到成百上千 query 的 Query Registry、Query Factory 和 Label Builder 协议。
5. 将模型拆成只计算一次的 State Backbone 与可分块执行的 Query/Choice Decoder。
6. 定义从现有 Multi-task Reward checkpoint 迁移 State Backbone 的精确规则。
7. 定义 Outcome Store、稀疏 Query Sampling、自然分布 Calibration Stream 和同日 Pair Stream。
8. 定义 batch、forward API、loss、跨 query 一致性、checkpoint 和 inference artifact schema。
9. 将 E1 细化为 H7 return-bucket 文本 query 的文件级实现方案。

### v0.1.0 - 2026-09-20

1. 建立 Jev-like scalable financial decision model 的长期设计文档。
2. 记录当前 Multi-task Reward 和 Residual Reward 正式证据。
3. 定义 State / Query / Choices / Logits 通用接口。
4. 定义收益、风险、可成交性、买入、持仓和组合级 query taxonomy。
5. 定义 Choice + Pairwise + Calibration 混合训练目标。
6. 冻结 H7 return-bucket query 作为第一项候选实验。

## 37. V1 股票状态理解模型

### 37.1 研究问题

V1 只回答：

```text
根据当前股票与市场状态，
未来不同时间尺度的价格结果更可能属于哪个 Candidate？
```

V1 不回答：

```text
当前账户是否应该买入
当前持仓是否应该卖出
应该配置多少仓位
```

选股策略可以在模型外部读取概率并构造 score，但模型本身先专注于理解股票未来状态。

### 37.2 模型能力

V1 至少支持：

```text
未来 N 日是否上涨
未来 N 日是否上涨超过 X%
未来 N 日是否下跌超过 X%
未来 N 日收益所在区间
未来 N 日相对市场收益所在区间
```

首版 horizon 集：

```text
3
5
7
10
20
30
```

### 37.3 成功定义

V1 成功不等于：

```text
分类 accuracy 很高
```

需要同时成立：

1. Candidate probability 有校准意义。
2. 同一 Query 下，高概率股票真实命中率更高。
3. Pairwise A/B 排序优于或不弱于当前 Reward baseline。
4. 从概率派生的选股 score 在固定 QuantX 合同下有增量。
5. 多 Query 训练不损害核心 H7 能力。

## 38. Query Plugin 系统

### 38.1 目录结构

建议代码目录：

```text
tmp/jev-like-financial-decision-v1/
  queries/
    __init__.py
    base.py
    return_direction.py
    return_range.py
    relative_return.py
```

V1 不扫描任意用户路径，也不从 YAML 动态加载任意模块。

`queries/__init__.py` 显式注册允许使用的 Query Family：

```python
QUERY_FAMILIES = {
    "return_direction": ReturnDirectionQueryFamily(),
    "return_range": ReturnRangeQueryFamily(),
    "relative_return": RelativeReturnQueryFamily(),
}
```

### 38.2 Query Family 接口

```python
class QueryFamily(Protocol):
    family_id: str
    family_version: int

    def build_specs(self) -> list[QuerySpec]:
        ...

    def build_labels(
        self,
        outcomes: OutcomeBatch,
        spec: QuerySpec,
    ) -> QueryTargets:
        ...

    def preference_values(
        self,
        outcomes: OutcomeBatch,
        spec: QuerySpec,
    ) -> Tensor:
        ...
```

一个 Python 文件代表一种 Query Family。

例如 `return_direction.py` 可以生成：

```text
H3 return > 0
H5 return > 0
H7 return > 0
H10 return > 0
H20 return > 0
H30 return > 0

H3 return > 3%
H5 return > 3%
...

H3 return > 5%
H5 return > 5%
...
```

### 38.3 QuerySpec

```python
@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    version: int
    family_id: str

    query_text: str
    candidate_specs: tuple[CandidateSpec, ...]

    label_config: dict[str, Any]
    sampling_weight: float
    loss_weight: float
    metrics: tuple[str, ...]
```

`label_config` 不进入模型，只供真实标签构建和审计使用。

### 38.4 CandidateSpec

```python
@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_text: str
    ordinal_rank: int | None
    utility_value: float | None
```

示例：

```yaml
query_text: >
  从下一个交易日收盘开始计算，
  这只股票未来七个交易日的收益属于哪个区间？

candidates:
  - candidate_id: large_loss
    candidate_text: 收益低于负百分之五
    ordinal_rank: 0

  - candidate_id: small_loss
    candidate_text: 收益介于负百分之五和零之间
    ordinal_rank: 1

  - candidate_id: small_gain
    candidate_text: 收益介于零和正百分之五之间
    ordinal_rank: 2

  - candidate_id: large_gain
    candidate_text: 收益不低于正百分之五
    ordinal_rank: 3
```

### 38.5 Query ID

Query ID 必须同时表达：

```text
family
语义版本
文本版本
label 版本
```

示例：

```text
return.direction.h7.gt0.textv1.labelv1
return.range.h7.fixed4.textv1.labelv1
```

Query 文本、Candidates、边界或 label 规则变化时，必须生成新 ID。

## 39. Query 与 Candidate 文本

### 39.1 文本是模型唯一任务语义

模型不会收到：

```text
horizon=7
threshold=0.05
family=return_range
```

它只收到对应文本 embedding。

因此 Query 文本必须完整、自包含：

```text
从信号日后的第一个交易日收盘价开始计算，
这只股票未来七个交易日的收益是否大于正百分之五？
```

不能写成：

```text
未来是否上涨？
```

因为它缺少 entry、horizon 和 threshold 语义。

### 39.2 Candidate 文本必须重复关键上下文

仅写：

```text
是
否
```

可能导致 Candidate embedding 语义过弱。

推荐：

```text
是，未来七个交易日收益大于正百分之五
否，未来七个交易日收益不大于正百分之五
```

### 39.3 Prompt Template

每个 Query Family 定义规范模板：

```python
QUERY_TEMPLATE_ZH = (
    "从信号日后的第一个交易日收盘价开始计算，"
    "这只股票未来{horizon_text}个交易日的收益"
    "是否{comparison_text}{threshold_text}？"
)
```

Query Factory 负责填充文本并生成稳定 QuerySpec。

### 39.4 数字文本

V1 固定使用中文规范数字：

```text
三
五
七
十
二十
三十
```

阈值：

```text
零
正百分之三
正百分之五
负百分之五
```

不同文本格式不能在同一 query_id 下混用。

### 39.5 Paraphrase

V1 每个 query 只有一个 canonical text。

后续可增加：

```text
paraphrase_texts
```

但它们共享同一 query_id、label 和 choices，并接受等价性校准。

## 40. FinancialDecisionProcessor

### 40.1 Processor 职责

生产或训练数据不会被完整转换成自然语言。

正确的数据路径：

```text
市场数值数据
-> Feature Processor
-> AE latent

Query/Candidate 文本
-> Text Processor
-> text embeddings

AE latent + text embeddings
-> FinancialDecisionProcessor
-> 模型 tensors
```

股票 OHLCV 不进行字符串序列化。

### 40.2 Processor 输入

```python
processor(
    state_rows,
    query_ids,
) -> FinancialDecisionBatch
```

### 40.3 Processor 输出

```python
@dataclass
class FinancialDecisionBatch:
    state_inputs: StateInputs

    query_indices: LongTensor       # [A]
    query_embeddings: FloatTensor   # [A, D_text]

    choice_embeddings: FloatTensor  # [A, C, D_text]
    choice_mask: BoolTensor         # [A, C]

    target_choice: LongTensor       # [A]
    valid_mask: BoolTensor          # [A]
    sample_weights: FloatTensor     # [A]

    state_assignment: LongTensor    # [A]
```

`A` 是当前 batch 的 query assignments 数量。

### 40.4 文本 Embedding 预计算

启动正式训练前执行：

```text
compile query registry
-> encode all query texts
-> encode all candidate texts
-> write embedding cache
```

训练热路径只做 embedding lookup。

### 40.5 Text Encoder 合同

V1：

```text
BAAI/bge-small-zh-v1.5
transformers.AutoTokenizer
transformers.AutoModel
CLS pooling
L2 normalization
frozen
```

模型 checkpoint 必须记录 resolved revision 与文件 checksum。

训练时若 embedding cache 不存在，直接报错，不联网下载。

## 41. V1 模型结构

### 41.1 输入

```text
AE latent:
  [B, 27, 128]

Query assignments:
  A 个

Query text embeddings:
  [A, 512]

Candidate text embeddings:
  [A, C, 512]
```

### 41.2 State Backbone

从当前 baseline 迁移：

```text
input_proj
cls_token
position
7 个 RewardTransformerBlock
norm
```

输出：

```text
state_memory: [B, 28, 768]
```

每个 unique state 只运行一次。

### 41.3 Query/Choice 投影

```text
query_token =
  Linear(512, 768)(query_text_embedding)

choice_token =
  Linear(512, 768)(choice_text_embedding)
```

V1 不加入：

```text
query ID embedding
horizon embedding
threshold embedding
family embedding
```

### 41.4 Decision Decoder

对于 assignment `a`：

```text
tokens_a =
  [QUERY]
  + [CANDIDATE_1 ... CANDIDATE_C]
```

Decoder：

```text
2 层
d_model=768
12 heads
```

每层：

```text
query/candidate self-attention
query/candidate -> state_memory cross-attention
MLP
```

### 41.5 Candidate Logits

每个 candidate token 共享同一个 logit head：

```text
logit_candidate =
  Linear(768, 1)(candidate_hidden)
```

输出：

```text
logits: [A, C]
```

不同 query 的 candidate 数量通过 mask 控制。

### 41.6 为什么不用固定 Query Head

禁止：

```text
if query_id == X:
    use head_X
```

否则即使接口看起来像 query-choice，本质仍是上千个固定 head。

## 42. Label 构建

### 42.1 真实标签

每个 query 的 target candidate 必须由真实未来数据确定。

二分类：

```text
future_return > threshold
  -> positive candidate
else
  -> negative candidate
```

区间分类：

```text
future_return 落入哪个 range
  -> 对应 candidate
```

### 42.2 Preference Value

每个 Query Family 还必须输出连续比较值。

`return_direction`：

```text
preference_value = future_return
```

`return_range`：

```text
preference_value = future_return
```

后续风险 query：

```text
preference_value = signed_max_drawdown
```

### 42.3 同日随机 Bad Case

对 state A：

```text
date(B) = date(A)
query(B) = query(A)
preference_value(B)
  < preference_value(A) - min_gap(query)
```

从候选 bad pool 中按 epoch seed 随机抽取。

这保留当前 Reward Model 的训练优势：

```text
每个 epoch 看见不同的比较关系
不跨日期比较
不把市场整体涨跌误当个股 alpha
```

### 42.4 Query-specific Gap

每类 query 独立定义 pair gap：

```yaml
return_direction:
  pair_gap: 0.005

return_range:
  pair_gap: 0.005
```

Pair gap 属于 label builder 配置，不进入模型。

### 42.5 Label Builder 输出

```python
@dataclass
class QueryTargets:
    target_choice: ndarray
    valid_mask: ndarray
    preference_value: ndarray
    sample_weight: ndarray
    audit_columns: dict[str, ndarray]
```

## 43. 概率与 Pairwise 的联合训练

### 43.1 Candidate Choice Loss

对单只股票：

```text
L_choice =
  CrossEntropy(candidate_logits, true_candidate)
```

该 loss 让模型知道：

```text
这只股票真实属于哪个 Candidate。
```

### 43.2 Candidate Probability

```text
P(candidate_j | state, query)
  = softmax(logits)_j
```

概率必须通过自然分布 validation 做 Brier/NLL/ECE 校准。

### 43.3 Query Score Projection

Pairwise 需要把 Candidate probabilities 投影为该 query 的可比较 scalar。

Binary Direction：

```text
score =
  logit(positive)
  - logit(negative)
```

Return Range：

```text
score =
  sum_j P(candidate_j) * train_fold_mean_return(candidate_j)
```

每个 Query Family 定义自己的：

```python
score_projection(logits, probabilities, spec)
```

### 43.4 Cross-state Pairwise

```text
L_pair =
  softplus(
    -(score(A, query) - score(B, query))
    / temperature
  )
```

这里 A/B 必须：

```text
同一天
同一个 query
真实 preference value 有足够差距
```

### 43.5 为什么需要两种 Loss

只用 Choice CE：

```text
可能分类准确，但 TopK 排序不够尖锐。
```

只用 Pairwise：

```text
知道 A > B，但概率没有绝对语义。
```

联合训练：

```text
Choice CE:
  校准单股票 outcome

Pairwise:
  校准横截面相对强弱
```

### 43.6 V1 总 Loss

```text
L =
  1.0 * L_choice
  + 0.5 * L_pairwise
  + 0.1 * L_brier
```

该权重先作为预注册默认值，不在 2020-2026 搜索。

## 44. V1 Query Set

### 44.1 Return Direction Family

文件：

```text
queries/return_direction.py
```

生成：

```text
horizons = [3, 5, 7, 10, 20, 30]
thresholds = [0.0, 0.03, 0.05]
```

共：

```text
18 个 positive-direction queries
```

另生成 downside：

```text
return < -3%
return < -5%
```

共：

```text
12 个 downside queries
```

### 44.2 Return Range Family

文件：

```text
queries/return_range.py
```

每个 horizon 一个 query：

```text
H3
H5
H7
H10
H20
H30
```

Choices 默认：

```text
< -5%
-5% ~ 0
0 ~ 5%
>= 5%
```

共 6 个 query。

### 44.3 Relative Return Family

文件：

```text
queries/relative_return.py
```

V1.1 再启用：

```text
相对全市场收益是否为正
相对行业收益是否为正
相对全市场收益区间
```

V1 首轮不启用，避免同时改变 label 语义。

### 44.4 V1 Active Set

基础设施支持全部 36 个绝对收益 query。

训练 curriculum：

```text
阶段 A:
  只启用 H3/H7 direction + range
  共 6 个 query

阶段 B:
  启用全部 36 个绝对收益 query
```

阶段 A 包含：

```text
H3 return > 0
H3 return > 5%
H3 return range
H7 return > 0
H7 return > 5%
H7 return range
```

先证明模型真正使用文本中的“三天/七天”和不同 Candidate 语义。

## 45. 数据集与采样

### 45.1 State 不重复物化

现有样本仍由：

```text
row_id
instrument_idx
date_idx
target_row
```

定位。

Query assignment 单独生成，不复制 60 日窗口。

### 45.2 Natural Stream

用于 Choice probability：

```text
按真实日期和类别分布采样
```

允许训练阶段对 query family 做均衡，但 validation/calibration 不改变自然分布。

### 45.3 Pair Stream

用于 A/B：

```text
先抽 state A
再抽 query
再从同日、同 query 的较差样本池随机抽 state B
```

### 45.4 一个 Batch

推荐每步：

```text
Natural assignments:
  B states x Q sampled queries

Pair assignments:
  P same-date A/B pairs
```

两条流共享同一次 state encoding。

### 45.5 Query Family 平衡

```text
family_loss =
  mean(losses inside family)

total_loss =
  sum(family_weight * family_loss)
```

增加 100 个同 family query 不会把该 family 权重放大 100 倍。

## 46. 训练阶段

### 46.1 Warm Start

从当前 Multi-task Reward epoch010 加载 State Backbone。

丢弃固定三 head。

### 46.2 Epoch 1

```text
Frozen AE: freeze
State Backbone: freeze
Text Encoder: freeze
Query/Choice Decoder: train
```

目标：

```text
先让随机 decoder 学会使用已有股票表示。
```

### 46.3 Epoch 2-10

```text
Frozen AE: freeze
State Backbone: train, lr=3e-5
Query/Choice Decoder: train, lr=1.5e-4
Text Encoder: freeze
```

### 46.4 每 Epoch 保存

```text
checkpoint
JSONL
TensorBoard
per-query metrics
per-family metrics
calibration bins
pairwise metrics
query sampling counts
```

## 47. 推理与生产 Processor

### 47.1 启动

生产进程启动时：

```text
加载 Query Registry
校验 registry checksum
加载 Query/Candidate embedding cache
加载 frozen AE
加载 Jev-like checkpoint
```

### 47.2 接收股票数据

```text
股票/市场原始数据
-> 现有特征 Processor
-> scaler
-> AE latent
```

不是：

```text
把 K 线数字转换成文本 prompt
```

### 47.3 构建模型 Prompt

模型内部的 prompt 是 tensor prompt：

```text
[STATE MEMORY]
[QUERY TEXT EMBEDDING]
[CANDIDATE TEXT EMBEDDINGS]
```

### 47.4 请求接口

```python
predict(
    state_batch,
    query_ids=[
        "return.direction.h3.gt0.textv1.labelv1",
        "return.direction.h7.gt0.textv1.labelv1",
        "return.range.h7.fixed4.textv1.labelv1",
    ],
) -> QueryDecisionBatch
```

输出：

```python
{
  "query_id": ...,
  "candidates": [
    {
      "candidate_id": ...,
      "logit": ...,
      "probability": ...,
    }
  ],
  "entropy": ...,
  "confidence_margin": ...,
}
```

### 47.5 选股 Score

例如 H7 range query：

```text
expected_return =
  sum(P(candidate) * train_fold_bucket_mean(candidate))
```

该 score 才导出给 QuantX。

## 48. 文件级实现清单

V1 目录：

```text
tmp/jev-like-financial-decision-v1/
```

新增：

```text
queries/
  __init__.py
  base.py
  return_direction.py
  return_range.py

query_registry_v1.yaml
compile_query_registry_v1.py

jev_query_schema_v1.py
jev_text_processor_v1.py
jev_label_builders_v1.py
jev_query_dataset_v1.py
jev_model_v1.py
jev_losses_v1.py
jev_calibration_graph_v1.py

train_jev_financial_decision_ddp_v1.py
infer_jev_financial_decision_v1.py
evaluate_jev_financial_decision_v1.py
derive_jev_quantx_score_v1.py

tests/
  test_query_registry_v1.py
  test_query_labels_v1.py
  test_text_processor_v1.py
  test_jev_model_v1.py
  test_jev_losses_v1.py
  test_jev_checkpoint_v1.py
  test_jev_inference_v1.py
```

复用现有：

```text
WTSPaths
WeakToStrongPathDataset
frozen AE checkpoint
RewardTransformerBlock
RMSNorm
DDP helpers
market_score_artifact_v1
```

不修改现有 baseline 默认行为。

## 49. 实现验收

### 49.1 Query Registry

```text
factory 编译确定性
Query ID 无重复
文本和 label config checksum 固定
Candidate 至少 2 个
```

### 49.2 Label

```text
每个 valid state/query 恰好一个真实 Candidate
H3/H7 目标索引正确
Query 文本与 label config 一致
Pair A/B 同日同 Query
```

### 49.3 Processor

```text
Query/Candidate embeddings 可复现
无隐式网络请求
同一 state 多 query 只 encode 一次
```

### 49.4 Model

```text
输出 [assignments, max_choices]
mask 后概率和为 1
交换 Candidate 顺序后 logits 同步交换
不使用 query ID/horizon 旁路特征
```

### 49.5 Loss

```text
正确 Candidate logit 梯度向上
错误 Candidate logit 梯度向下
Pairwise A/B 梯度方向正确
Brier 对过度自信产生惩罚
```

### 49.6 语义测试

同一股票：

```text
H3 query
H7 query
```

必须能够输出不同分布。

将 H3/H7 文本 embedding 交换时，输出必须随文本交换，证明模型真正读取 Query，而不是只读取股票状态。

## 50. 第一轮实验矩阵

### 50.1 A：Fixed-head Baseline

当前 H7 Multi-task Reward baseline。

### 50.2 B：Jev-like Choice Only

```text
6 个阶段 A Query
CE + Brier
无 Pairwise
```

### 50.3 C：Jev-like Choice + Pairwise

```text
相同 6 个 Query
CE + Brier + Pairwise
```

### 50.4 比较目的

```text
B vs A:
  query-choice probability 是否优于固定 head

C vs B:
  Pairwise 是否提高横截面 TopK 能力

C vs A:
  新建模范式是否真正提高收益/回撤/Sharpe
```

### 50.5 训练后顺序

```text
1. 分类和校准指标
2. H3/H7 Query 语义交换测试
3. 同日 Pairwise 指标
4. Expected-return TopK
5. 固定 QuantX 回测
```

在 1-4 未通过前不运行正式 QuantX。

## 51. v0.3 实现顺序

```text
Step 1:
  Query Family 接口
  return_direction.py
  return_range.py
  Registry compiler

Step 2:
  Label builders
  Query/Candidate label 审计
  Same-date bad sampler

Step 3:
  BGE text embedding cache
  FinancialDecisionProcessor

Step 4:
  State Backbone 迁移
  Query/Choice Decoder

Step 5:
  Choice-only trainer
  Choice + Pairwise trainer

Step 6:
  历史 fold 训练与校准评估

Step 7:
  score export
  QuantX 配对回测
```

任何一步出现语义、时间或概率校准问题，先在本文记录后停止扩展 Query 数量。

## 52. 当前变更日志

### v0.3.3 - 2026-09-20

1. Query 时间语义扩展为 `start_offset x window`，模型侧仍只读取完整文本 embedding。
2. 新增 `max_drawdown_event` 和 `max_drawdown_range` Query Family。
3. 新增 `stock_understanding_scale_v1` profile，共 480 个 Query。
4. Candidate 数量覆盖 2、4、5、7，并保持 sequence + mask 统一接口。
5. 新增静态 HTML Query Catalog，可按 Family、起点、Window、Candidate N 和文本筛选。
6. E1 registry、真实标签和 BGE embedding 已按新 Query ID 重建。
7. 更新后的 E1 工件已完成真实单卡 1-step smoke。

### v0.3.2 - 2026-09-20

1. Registry Compiler 增加统一 `inspect` 命令，可在训练前查看全部或单个 Query。
2. Preview 固定输出 Query 文本、动态 Candidate 数量、Candidate 文本、Label Builder、Label Config、Score Projection 和 Pair Gap。
3. 明确 Candidate 是可变长度 sequence；二分类和四分类 Query 已在同一模型中通过 mask 测试。
4. 将百分比文本从阿拉伯数字统一为中文规范语义，并同步重建 registry、label manifest 和 BGE embedding cache。

### v0.3.1 - 2026-09-20

1. 开始实现 V1，不修改既有 Reward trainer 的默认行为。
2. 实现 QuerySpec、CandidateSpec、Query Family 和 Registry Compiler。
3. 实现 `return_direction` 与 `return_range` 两类 Query Family。
4. 编译 E1 Core 6 Query 和 Full Return 36 Query registry。
5. 实现真实未来路径 Query Label Builder 与 row-addressable memmap label store。
6. 实现冻结 BGE Query/Choice embedding cache 和 FinancialDecisionProcessor。
7. 实现 State Backbone + Query/Choice Decoder + shared Candidate logit head。
8. 实现 Choice CE、Brier 和 Cross-state Pairwise loss。
9. 实现 Natural Query Dataset 与 Same-date Query Pair Dataset。
10. 实现单卡和 2 卡 DDP 训练入口 smoke；正式训练尚未启动。

### v0.3.0 - 2026-09-20

1. 将 V1 范围收缩为股票未来状态理解，不包含账户、持仓和加减仓。
2. 明确模型侧只接收 AE latent、Query text embedding、Candidate text embeddings 和 mask。
3. horizon、threshold、family、label builder 等字段只用于生成文本和真实标签，不作为模型旁路输入。
4. 定义每个 Python 文件代表一种 Query Family，并由工厂批量生成具体 QuerySpec。
5. 每个 QuerySpec 同时定义真实 Candidate label、连续 preference value 和同日 Bad Case 采样合同。
6. 明确 Pairwise 是 V1 核心训练关系，与 Choice CE 和 Brier 联合训练。
7. 定义 FinancialDecisionProcessor：数值金融数据进入 AE，文本 Query/Candidates 进入冻结文本编码器，二者在模型 token 空间融合。
8. 定义 V1 的 36 个绝对收益 query 能力边界，以及先启用 H3/H7 六个核心 query 的 curriculum。
9. 定义 State Backbone checkpoint 迁移、Query/Choice Decoder、训练数据流、生产推理接口和文件级实现清单。
10. 定义 Fixed-head、Choice-only、Choice+Pairwise 三组严格对照。

## 53. 2026-09-20 V1 实现记录

### 53.1 当前状态

```text
status: partial
```

当前已完成：

```text
Milestone 0:
  Query Schema
  Query Family
  Registry Compiler
  Label Builder
  单元测试

Milestone 1:
  BGE 模型工件
  Query/Choice embedding cache
  FinancialDecisionProcessor

Milestone 2:
  State Backbone checkpoint migration
  Query/Choice Decoder
  Choice logits

Milestone 3:
  Natural Query Stream
  Same-date Pair Stream
  CE + Brier + Pairwise
  单卡 smoke
  2 卡 DDP smoke
```

尚未完成：

```text
正式 10 epoch 训练
完整 validation/calibration evaluator
正式 inference exporter
QuantX derived score
正式配对回测
```

### 53.2 实际文件

```text
tmp/jev-like-financial-decision-v1/
  query_registry_v1.yaml
  compile_query_registry_v1.py
  compile_query_labels_v1.py
  build_text_embedding_cache_v1.py

  jev_query_schema_v1.py
  jev_label_builders_v1.py
  jev_query_dataset_v1.py
  jev_text_processor_v1.py
  jev_model_v1.py
  jev_losses_v1.py
  train_jev_financial_decision_ddp_v1.py

  queries/
    __init__.py
    base.py
    return_direction.py
    return_range.py

  tests/
    test_query_registry_v1.py
    test_query_labels_v1.py
    test_query_dataset_v1.py
    test_text_processor_v1.py
    test_jev_model_v1.py
    test_jev_losses_v1.py
```

### 53.3 Query Registry

E1 Core：

```text
query count: 6
binary direction queries: 4
four-choice range queries: 2
```

Query：

```text
H3 return > 0
H3 return > 5%
H3 return range
H7 return > 0
H7 return > 5%
H7 return range
```

Full Return Registry：

```text
query count: 36
binary direction queries: 30
four-choice range queries: 6
```

工件：

```text
artifacts/query_registry/e1_core_v1.json
artifacts/query_registry/e1_core_v1.manifest.json
artifacts/query_registry/absolute_return_full_v1.json
artifacts/query_registry/absolute_return_full_v1.manifest.json
```

### 53.4 Query Label Store

E1 Core 真实标签已从现有 future path target 编译：

```text
states: 11,828,037
queries: 6
shape: [6, 11,828,037]
max horizon: 7
```

文件：

```text
artifacts/query_labels/e1_core_v1/
  target_choice_uint8.mmap
  valid_uint8.mmap
  preference_float32.mmap
  manifest.json
```

标签分布：

```text
H3 return > 0:
  no  = 6,112,211
  yes = 5,715,826

H3 return > 5%:
  no  = 10,790,619
  yes = 1,037,418

H7 return > 0:
  no  = 6,001,980
  yes = 5,826,057

H7 return > 5%:
  no  = 9,594,384
  yes = 2,233,653
```

### 53.5 Text Encoder

模型：

```text
BAAI/bge-small-zh-v1.5
model.safetensors: 95,827,648 bytes
embedding dim: 512
CLS pooling
L2 normalize
frozen
```

由于直连 `huggingface.co` 超时，使用：

```text
HF_ENDPOINT=https://hf-mirror.com
hf download
```

完成显式模型下载。正式训练入口只读取本地模型/embedding，不进行网络访问。

Embedding 工件：

```text
artifacts/text_embeddings/
  e1_core_bge_small_zh_v1_5.npz
  e1_core_bge_small_zh_v1_5.manifest.json
```

### 53.6 State Backbone 迁移

使用当前 Multi-task Reward epoch010 checkpoint。

迁移结果：

```text
loaded state-backbone keys: 75
ignored source keys:
  head.weight
  head.bias

missing target keys: 0
unexpected target keys: 0
```

说明固定三 head 被正确丢弃，已有 state representation 完整迁移。

### 53.7 测试

```text
12 passed
```

覆盖：

```text
E1 Core 6 Query 编译
Full Return 36 Query 编译
真实 H3/H7 label
range 边界
Natural Query 虚拟采样
同日随机较差 Pair
Choice CE/Brier 梯度
Pairwise 梯度
Candidate mask
Candidate permutation
embedding cache / registry 对齐
State Backbone warm-start
```

### 53.8 单卡 Smoke

```text
run:
  jev-like-e1-core-smoke-fixed-20260920-133434

train dates: 2
validation dates: 1
steps: 2
```

结果：

```text
train choice accuracy: 43.75%
train pairwise accuracy: 50.00%
validation choice accuracy: 43.75%
validation pairwise accuracy: 43.75%
checkpoint: written
```

该结果只证明闭环，不表示模型效果。

### 53.9 两卡 DDP Smoke

```text
run:
  jev-like-e1-core-ddp2-smoke-20260920-133511

world size: 2
train dates: 2
validation dates: 1
steps: 2
```

结果：

```text
train choice accuracy: 53.13%
train pairwise accuracy: 59.38%
validation choice accuracy: 43.75%
validation pairwise accuracy: 56.25%
checkpoint: written
run_end: trained
```

该结果证明：

```text
DDP sampler 正常
梯度同步正常
rank0 完整 validation 正常
TensorBoard 正常
JSONL 正常
checkpoint 正常
```

### 53.10 Smoke 中发现并修复的问题

二分类 query pad 到四个 Candidate 后，早期 Pairwise projection 错误地按 tensor 宽度判断二分类。

已修复为：

```text
显式读取 positive_choice_index
显式读取 negative_choice_index
通过 choice_mask 验证二者有效
```

修复后全部测试和两卡 smoke 通过。

### 53.11 下一步

下一步在启动正式训练前完成：

```text
完整 validation/calibration evaluator
按 query/family 输出 NLL、Brier、ECE、accuracy
H3/H7 expected-return RankIC / TopK
Query 文本交换语义测试
正式 inference exporter
```

这些评估接口完成后，才启动 6 卡 10 epoch 正式训练。

## 54. 训练前 Query/Candidate 预览入口

### 54.1 查看 E1 Core 全部 Query

```bash
cd ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/jev-like-financial-decision-v1

${HOME}/anaconda3/envs/test/bin/python \
  compile_query_registry_v1.py inspect \
  --profile e1_core
```

### 54.2 查看完整 36 Query Profile

```bash
${HOME}/anaconda3/envs/test/bin/python \
  compile_query_registry_v1.py inspect \
  --profile absolute_return_full
```

### 54.3 查看单个 Query 的 JSON

```bash
${HOME}/anaconda3/envs/test/bin/python \
  compile_query_registry_v1.py inspect \
  --profile e1_core \
  --query-id return.range.h7.fixed4.textv1.labelv1 \
  --format json
```

输出固定包含：

```text
query_id
family_id
query_text
candidate_count
candidates:
  index
  candidate_id
  candidate_text
  ordinal_rank
  representative_value
label_builder
label_config
score_projection
pair_gap
```

### 54.4 动态 Candidate Sequence

模型不假设 Candidate 数量固定：

```text
Query A:
  2 candidates

Query B:
  4 candidates

Query C:
  5 candidates
```

同一 batch 中：

```text
choice_embeddings: [assignments, max_choices, text_dim]
choice_mask:       [assignments, max_choices]
choice_logits:     [assignments, max_choices]
```

不足 `max_choices` 的部分只做 padding，并在 softmax 前 mask 为无效。

Candidate 以 sequence token 进入 Query/Choice Decoder，并通过 self-attention 相互比较；每个有效 Candidate 使用同一个共享 logit head，不存在固定二分类或固定四分类输出层。

当前测试已覆盖：

```text
2-choice 与 4-choice Query 混合
padding Candidate 概率为 0
有效 Candidate 概率和为 1
Candidate sequence 置换时 logits 同步置换
```

## 55. 2026-09-20 Query 扩展与 Catalog 实现

### 55.1 时间变量

统一未来窗口：

```text
start_offset = s
window = w

start price = close(T+s)
end price   = close(T+s+w-1)
```

当前 Scale Profile：

```text
start_offset = [1, 2, 3, 4]
window       = [3, 5, 7, 10, 15, 20]
```

所有组合满足：

```text
start_offset + window - 1 <= 30
```

这些变量只用于 Query 文本生成和 Label Builder，不作为模型旁路输入。

### 55.2 Query Family

```text
return_direction:    216
return_range:         72
max_drawdown_event:  144
max_drawdown_range:   48
total:               480
```

### 55.3 动态 Candidate

```text
2 Candidates: 360 Queries
4 Candidates:  48 Queries
5 Candidates:  48 Queries
7 Candidates:  24 Queries
```

### 55.4 最大回撤标签

对指定窗口价格：

```text
running_peak[t] = max(price[0:t])
drawdown[t] = price[t] / running_peak[t] - 1
max_drawdown_magnitude = -min(drawdown)
```

Event Query：

```text
最大回撤是否大于 3% / 5% / 8% / 10% / 15% / 20%
```

Range Query：

```text
coarse4:
  <3%
  3%-8%
  8%-15%
  >=15%

fine5:
  <3%
  3%-5%
  5%-10%
  10%-20%
  >=20%
```

### 55.5 Query Catalog

HTML：

```text
tmp/jev-like-financial-decision-v1/
artifacts/query_catalog/stock_understanding_scale_v1.html
```

JSON：

```text
artifacts/query_catalog/stock_understanding_scale_v1.json
```

页面展示：

```text
Query 总数
Family 数量
Candidate N 分布
start_offset 覆盖
window 覆盖
Family x start/window 覆盖矩阵
Query 文本
Candidate 文本
Label Builder
Label Config
Score Projection
Pair Gap
```

页面支持：

```text
文本搜索
Family 筛选
起点筛选
Window 筛选
Candidate N 筛选
Query 详情展开
```

### 55.6 统一 CLI

查看 E1：

```bash
python compile_query_registry_v1.py inspect --profile e1_core
```

查看完整收益 Profile：

```bash
python compile_query_registry_v1.py inspect --profile absolute_return_full
```

查看 480 Query：

```bash
python compile_query_registry_v1.py inspect --profile stock_understanding_scale_v1
```

查看单个 Query：

```bash
python compile_query_registry_v1.py inspect \
  --profile stock_understanding_scale_v1 \
  --query-id drawdown.range.s2.w10.fine5.textv1.labelv1 \
  --format json
```

### 55.7 工件

```text
artifacts/query_registry/
  e1_core_v1.json
  absolute_return_full_v1.json
  stock_understanding_scale_v1.json

artifacts/query_labels/e1_core_v1/
  target_choice_uint8.mmap
  valid_uint8.mmap
  preference_float32.mmap
  manifest.json

artifacts/text_embeddings/
  e1_core_bge_small_zh_v1_5.npz
  e1_core_bge_small_zh_v1_5.manifest.json
```

该阶段 Scale Profile 暂时只生成 registry 和 Catalog；后续已经生成
480 Query x 1182 万 state 的全量标签矩阵，见第 56 节。

### 55.8 验证

```text
17 passed
```

新增覆盖：

```text
start_offset 对真实收益窗口的影响
最大回撤 event label
浮点 range 边界
480 Query 数量
Family 覆盖
2/4/5/7 Candidate 长度
HTML Catalog 过滤入口
```

更新后 E1 工件真实 smoke：

```text
run:
  jev-like-e1-core-start-window-smoke-20260920-142024

train choice accuracy: 50.00%
train pairwise accuracy: 62.50%
validation choice accuracy: 50.00%
validation pairwise accuracy: 50.00%
```

该 smoke 只证明新 Query ID、文本 embedding、真实 label store、模型和训练入口闭环。

## 56. 2026-09-20 E2 Scale 实验记录

### 56.1 实验目标

E2 用来回答两个问题：

1. 同一个 Query/Choice Decoder 和同一个 Candidate Head，能否同时学习收益与风险语义。
2. 风险 Query 是否能在正式 Top10 回测中降低回撤并提升风险调整后收益。

本实验不改变 AE、State Backbone、Query/Choice Decoder 和 Candidate Head 的网络结构。

### 56.2 Query 与标签

E2 使用完整 Scale Registry：

```text
Queries: 480

return_direction:     216
return_range:          72
max_drawdown_event:   144
max_drawdown_range:    48
```

全量标签已经生成，不再是 55.7 节记录的“仅 Registry/Catalog”状态：

```text
states:  11,828,037
queries: 480
shape:   [480, 11,828,037]
size:    约 32 GiB
```

工件：

```text
tmp/jev-like-financial-decision-v1/
  artifacts/query_labels/stock_understanding_scale_v1/
  artifacts/text_embeddings/stock_understanding_scale_v1_bge_small_zh_v1_5.npz
```

标签编译按基础结果缓存：

```text
24 个 return start/window
24 个 drawdown start/window
= 48 个基础 outcome
```

每个 chunk 只计算 48 个基础 outcome，再映射到 480 个 Query，避免对相同未来路径重复计算。

### 56.3 Query Sampling

每个训练 step 只回答 8 个 Query：

```text
return_range:       3
max_drawdown_range: 3
return_direction:   1
max_drawdown_event: 1
```

每步固定包含：

```text
return.range.s1.w7.coarse4.textv1.labelv1
drawdown.range.s1.w7.coarse4.textv1.labelv1
```

其他 Query 在 family 内确定性随机采样。

这意味着：

```text
480 个 Query 都能在 epoch 内轮转出现
单步 Decoder 计算量只从 E1 的 6 Query 增长到 8 Query
区间任务占 75%
二分类任务占 25%
```

### 56.4 二分类不平衡策略

二分类 Query 根据训练集正样本比例决定 Choice CE/Brier 权重：

```text
正负较平衡:
  weight = 0.5

少数类比例 5%-15%:
  weight = 0.2

少数类比例 <5%:
  weight = 0.0
  只保留基于连续 outcome 的 Pairwise
```

实际分布：

```text
range full weight:       120
binary weight 0.5:       228
binary weight 0.2:        80
binary choice disabled:   52
```

该策略不通过 oversampling 修改真实概率分布。

### 56.5 Warm Start 与训练

Warm Start：

```text
E1 checkpoint:
  jev-packed-e1-core-1024-5gpu-10ep-v2
  epoch 7
```

E2 Run：

```text
jev-e2-scale-480q-q8-ft-epoch7-5gpu-4ep-20260920-172122
```

训练设置：

```text
GPU: 3,4,5,6,7
stocks/rank/step: 1024
global stock batch: 5120
epochs: 4

epoch 1:
  State Backbone frozen

epoch 2-4:
  State Backbone trainable

state lr:   1e-5
decoder lr: 5e-5
```

### 56.6 固定验证结果

固定验证 Query：

```text
H3 return range coarse4
H7 return range coarse4
H3 drawdown range coarse4
H7 drawdown range coarse4
```

H7 按日 RankIC：

| E2 Epoch | H7 Return RankIC | H7 Drawdown RankIC |
| ---: | ---: | ---: |
| 1 | 0.07821 | **0.42290** |
| 2 | 0.07476 | 0.41465 |
| 3 | **0.07829** | 0.41624 |
| 4 | 0.07746 | 0.41080 |

风险标签显著比固定终点收益更容易学习：

```text
drawdown RankIC ≈ 0.41-0.42
return RankIC   ≈ 0.075-0.078
```

### 56.7 正式 QuantX 结果

统一执行合同：

```text
universe: all_a
window: 2020-01-02 through 2026-06-02
signal lag: 1
buy: T+1 close
capacity: Top10
exit: holding_days >= 7
costs: commission + stamp tax + 10 bp slippage
```

| Checkpoint / Score | Total Return | Annual Return | MDD | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| E2 E1 return only | +100.85% | 11.48% | -58.00% | 0.338 |
| E2 E1 70/30 return-risk | +33.58% | 4.61% | -41.14% | 0.184 |
| E2 E1 return Top50 -> low-DD Top10 | +26.07% | 3.68% | -54.17% | 0.137 |
| E2 E3 return only | **+154.60%** | **15.67%** | -48.04% | 0.480 |
| E2 E3 70/30 return-risk | +112.99% | 12.50% | **-29.91%** | **0.524** |
| E2 E3 return Top50 -> low-DD Top10 | +103.51% | 11.70% | -38.74% | 0.464 |

对照：

```text
E1 epoch 7:
  total return +121.77%
  MDD -44.06%
  Sharpe 0.398

old multitask reward baseline:
  total return +203.91%
  MDD -38.03%
  Sharpe 0.644
```

### 56.8 E2 结论

E2 证明：

1. 统一 Query/Choice Decoder 可以同时学习收益与回撤。
2. 回撤 Query 提供了独立于收益 Query 的有效信息。
3. E2 E3 纯收益超过 E1。
4. 70/30 风险组合显著降低最大回撤并提高 Sharpe。
5. 完全使用风险对 Return Top50 进行重排会损失过多收益。
6. 当前瓶颈已经从“风险是否可学”转向“收益监督是否足够干净并与 Top10 对齐”。

## 57. E3 设计目标

### 57.1 核心目标

E3 不继续把“更多阈值”当成更多知识。

E3 的目标是：

```text
使用更独立、更稳定、更贴近 Top10 决策的连续 outcome
构建少量高质量核心 Query
同时保留 Scale Query 作为低频辅助监督
```

最终目标：

```text
Sharpe > 0.644
MDD 不差于 -38.03%
纯收益版本 Total Return > +154.60%
风险组合尽量保持 MDD 约 -30%
```

### 57.2 非目标

E3 V1 不做：

```text
账户状态
动态加减仓
PPO / RL
盘中价格
未来最优止盈止损 oracle
通过完整 prediction period 搜索大量超参数
```

模型结构原则：

```text
AE 不变
State Backbone 不变
Query/Choice Decoder 不变
Candidate Head 不变
优先修改 outcome、query、sampling 和 loss
```

## 58. 为什么当前 480 Query 仍然不够好

### 58.1 Query 数量不等于独立监督数量

当前 480 Query 主要是 48 个基础连续 outcome 的不同离散化：

```text
return:
  4 start offsets x 6 windows = 24

drawdown:
  4 start offsets x 6 windows = 24
```

例如以下 Query 高度相关：

```text
H7 return > 0
H7 return > 1%
H7 return > 3%
H7 return > 5%
H7 coarse4 range
H7 fine7 range
```

它们并没有提供 6 份独立市场事实，只是对同一个 H7 return 重复切分。

### 58.2 固定终点收益噪声

固定 T+7 收益：

```text
R_7 = P(T+7) / P(T+1) - 1
```

会被 T+7 单日冲击显著影响。

但真实策略可能在：

```text
T+5
T+6
T+7
T+8
T+9
T+10
```

任一可成交日期退出。

因此单终点标签可能把“稳定上涨”与“最后一天偶然拉升”视为相同收益。

### 58.3 绝对收益与 Top10 目标错位

Top10 是同日横截面问题：

```text
给定当天所有股票，哪些股票相对更好？
```

而绝对收益 Query 回答：

```text
该股票未来是否上涨？
```

牛市中大量股票都上涨，熊市中大量股票都下跌。

绝对方向 Query 容易学习市场 regime，却不一定能区分同日 Top10。

## 59. OutcomeStore V2

### 59.1 目标

E3 不再为每个 Query 存一份 dense label。

建议新增：

```text
artifacts/outcome_store/e3_v1/
```

每个 state 只存少量连续 outcome，Query label 在训练时动态映射。

新增 Query 时：

```text
不重新生成 Query x State dense matrix
只新增 label mapper / candidate contract
```

### 59.2 Core Keys

```text
row_id
signal_date
instrument
target_row
valid_mask
```

### 59.3 Return Outcomes

对 start offset `s` 和 window `H`：

```text
terminal_return[s,H]
```

定义：

```text
P0 = close(T+s)
PH = close(T+s+H-1)
terminal_return = PH / P0 - 1
```

### 59.4 Robust Multi-exit Return

H7 主定义：

```text
R5  = close(T+5)  / close(T+1) - 1
R7  = close(T+7)  / close(T+1) - 1
R10 = close(T+10) / close(T+1) - 1

robust_return_5_7_10 = median(R5, R7, R10)
```

另一个局部退出定义：

```text
robust_return_5_to_9 =
  mean(close(T+5 ... T+9)) / close(T+1) - 1
```

V1 同时保留 median 和 mean 版本，实验前固定主版本。

### 59.5 Cross-sectional Return Rank

在同一个 signal date 的有效股票池中：

```text
cross_sectional_return_pct[H]
  = percentile_rank(terminal_return[H])
```

该标签直接对应：

```text
Bottom / Middle / Top 股票
```

而不是绝对市场方向。

### 59.6 Market Excess Return

基础定义：

```text
market_excess_return[H]
  = terminal_return_i[H]
  - median_j(terminal_return_j[H])
```

median 基线比 mean 更不易被极端股票污染。

后续可增加：

```text
benchmark index excess return
beta-neutral residual return
```

但 V1 先使用同日股票池 median。

### 59.7 Industry Excess Return

```text
industry_excess_return[H]
  = terminal_return_i[H]
  - median_{j in same industry}(terminal_return_j[H])
```

硬约束：

```text
必须使用 signal_date 当时可获得的 point-in-time industry membership
禁止使用未来行业分类回填历史
行业内有效股票数过少时 label invalid
```

如果 point-in-time 行业数据无法确认，E3 V1 暂不启用该 Family。

### 59.8 Path Return Series

相邻收盘收益：

```text
r_t = close_t / close_{t-1} - 1
```

用于构建风险与路径质量标签。

### 59.9 Maximum Drawdown

```text
running_peak_t = max(P_0 ... P_t)
drawdown_t = P_t / running_peak_t - 1
max_drawdown = -min(drawdown_t)
```

值越小越安全。

### 59.10 Downside Volatility

```text
downside_return_t = min(r_t, 0)
downside_volatility =
  sqrt(mean(downside_return_t^2))
```

只衡量负向波动，不把上涨波动当成风险。

### 59.11 MFE / MAE

```text
MFE = max(P_t / P_0 - 1)
MAE = -min(P_t / P_0 - 1)
```

稳定比率：

```text
mfe_mae_ratio = MFE / max(MAE, epsilon)
```

同时存原始 MFE、MAE 与截断后的 ratio。

### 59.12 Path Efficiency

```text
path_length = sum(abs(r_t))
path_efficiency = terminal_return / max(path_length, epsilon)
```

解释：

```text
接近 +1: 稳定向上
接近 0: 大幅震荡但终点变化小
接近 -1: 稳定向下
```

### 59.13 Positive-day Ratio

```text
positive_day_ratio = count(r_t > 0) / valid_days
```

用于区分：

```text
持续上涨
单日跳涨
震荡上涨
```

### 59.14 Drawdown Recovery

记录：

```text
maximum drawdown trough position
trough 后是否恢复到此前 peak
recovery_days
recovery_ratio
```

若窗口结束仍未恢复：

```text
recovery_days = invalid
recovered = false
```

### 59.15 Barrier Outcome

对固定 TP / SL：

```text
take_profit = +5%
stop_loss   = -3%
```

Candidates：

```text
take_profit_first
stop_loss_first
neither
```

如果同一天同时触发且仅有日线数据：

```text
label invalid
```

禁止用未来日内顺序进行猜测。

### 59.16 Net Utility

基础效用：

```text
net_utility =
  robust_return
  - lambda_drawdown * max_drawdown
  - fixed_round_trip_cost
```

V1 固定：

```text
lambda_drawdown candidates:
  0.25
  0.50
```

这些是不同 Query，不允许在 prediction backtest 上搜索后再声明最终结果。

## 60. E3 Core Query Set

### 60.1 Core v1 总量

E3 Core v1 建议 24 个 Query。

它们不是 24 个阈值变体，而是来自互补 outcome。

### 60.2 Return / Alpha Queries

| Family | Windows / Variants | Count |
| --- | --- | ---: |
| cross-sectional return rank | H3, H7, H15 | 3 |
| market excess return rank | H3, H7, H15 | 3 |
| industry excess return rank | H7, H15 | 2 |
| robust multi-exit return | median 5/7/10, mean 5-9 | 2 |

小计：

```text
10
```

### 60.3 Risk / Path Queries

| Family | Windows / Variants | Count |
| --- | --- | ---: |
| maximum drawdown rank | H3, H7, H15 | 3 |
| downside volatility rank | H7, H15 | 2 |
| MFE/MAE quality | H7, H15 | 2 |
| path efficiency | H7, H15 | 2 |
| drawdown recovery | H7 | 1 |
| barrier outcome | TP5/SL3, TP8/SL5 | 2 |

小计：

```text
12
```

### 60.4 Utility Queries

| Family | Variant | Count |
| --- | --- | ---: |
| return-drawdown utility rank | lambda=0.25 | 1 |
| return-drawdown utility rank | lambda=0.50 | 1 |

小计：

```text
2
```

总计：

```text
10 + 12 + 2 = 24
```

## 61. Candidate 设计

### 61.1 等频 Candidate

用于普通概率训练的 rank Query：

```text
Q0: Bottom 20%
Q1: 20%-40%
Q2: 40%-60%
Q3: 60%-80%
Q4: Top 20%
```

优点：

```text
训练分布接近平衡
Candidate 有序
Expected percentile 可直接计算
避免 rare-event CE collapse
```

### 61.2 Top-tail 不由极端 Candidate 承担

Top10 细分不使用：

```text
是否 Top1%
是否未来收益 > 10%
```

作为主 CE 任务。

Top-tail 由：

```text
Listwise
Hard Pairwise
```

负责。

### 61.3 Absolute Range Anchor

保留少量绝对区间 Query：

```text
absolute return range
absolute drawdown range
```

用途：

```text
概率解释
风险阈值
策略规则
跨日期绝对比较
```

但它们不再占主要训练预算。

### 61.4 Representative Value

Candidate 代表值优先使用：

```text
train split 内该 Candidate 的中位数
```

而不是均值。

尾部区间允许使用截断中位数，避免极端事件使 expected value 失真。

### 61.5 Binary Query

Binary probability 尽量从分布推导：

```text
P(return > x)
  = sum(P(return_bin_i)) for bins above x

P(drawdown > x)
  = sum(P(drawdown_bin_i)) for bins above x
```

独立 Binary Query 仅作为低权重语义一致性任务。

## 62. Query Quality Gate

每个 Query 加入 E3 Core 前必须通过以下检查。

### 62.1 Label Coverage

```text
valid ratio >= 95%
```

Barrier / Recovery 等特殊 Query 可单独声明更低 coverage，但不得静默填充。

### 62.2 Candidate Entropy

归一化熵：

```text
normalized_entropy =
  entropy(candidate_distribution) / log(candidate_count)
```

普通分布 Query 要求：

```text
normalized_entropy >= 0.80
```

### 62.3 Temporal Stability

按年份统计 Candidate 分布：

```text
train year distribution
validation year distribution
```

禁止某 Candidate 只在少数年份出现。

### 62.4 Redundancy

同一 Family 内连续 outcome 相关性：

```text
abs(Spearman correlation) >= 0.95
```

时视为高度冗余。

高度冗余 Query：

```text
保留一个主 Query
其他降为低频 auxiliary
```

### 62.5 Top-tail Relevance

至少满足一项：

```text
对未来净收益有稳定 RankIC
对 Top10/Top50 有稳定 lift
能降低同收益水平下的 realized drawdown
能解释执行失败或收益不稳定
```

## 63. E3 Sampling

### 63.1 每 Step Query Slots

建议：

```text
queries_per_batch = 12
```

固定 Anchor：

```text
H7 cross-sectional return rank
H7 market excess return rank
H7 robust multi-exit return
H7 maximum drawdown rank
H7 return-drawdown utility rank
```

其他 7 个 slots：

```text
5 个 E3 Core 随机 Query
2 个 Scale auxiliary Query
```

### 63.2 Sampling Budget

长期采样占比：

```text
E3 Core:        80%
Scale auxiliary: 20%
```

Binary auxiliary 不得超过总 Query slots 的 10%。

### 63.3 State Sampling

继续使用 fixed stock budget packed batch：

```text
同 rank 每 step 股票数基本一致
Pairwise / Listwise 只在同日 section 内计算
```

### 63.4 Regime Balance

训练日期按：

```text
年份
市场涨跌 regime
市场波动 regime
```

做日期级平衡。

不能因为近年股票数量更多，使近年日期在训练中占据不成比例的权重。

## 64. E3 Loss

### 64.1 总 Loss

建议初始权重：

```text
L_total =
  0.30 * L_query_distribution
  + 0.35 * L_h7_listwise
  + 0.20 * L_h7_hard_pairwise
  + 0.10 * L_risk_ranking
  + 0.05 * L_consistency
```

这些权重必须在 validation/calibration split 固定。

### 64.2 Query Distribution Loss

包括：

```text
Choice CE
Brier
Ordinal CDF loss
```

Ordinal CDF：

```text
predicted_cdf[k] = sum_{i <= k} P(choice_i)
target_cdf[k] = 1(i >= target_choice)

L_ordinal = mean(abs(predicted_cdf - target_cdf))
```

### 64.3 H7 Listwise

对同日股票 section：

```text
teacher_distribution =
  softmax(realized_net_utility / tau_teacher)

model_distribution =
  softmax(model_score / tau_model)

L_listwise =
  cross_entropy(teacher_distribution, model_distribution)
```

真实 target 使用：

```text
cost-adjusted H7 robust return
```

不是普通分类 label。

### 64.4 Top-tail Hard Pairwise

重点采样：

```text
真实 rank 1-5 vs 6-20
真实 rank 6-10 vs 11-30
真实 rank 11-20 vs 21-50
真实 Top50 内相近收益、不同风险 pair
```

减少：

```text
Top1 vs Bottom50%
```

这类过于容易、对 Top10 边界帮助有限的 pair。

### 64.5 Risk Ranking

风险 score 方向固定：

```text
expected drawdown 越小越好
downside volatility 越小越好
MAE 越小越好
recovery 越快越好
```

禁止不同风险 Query 使用相反 score 方向。

### 64.6 Consistency

示例：

```text
P(return > 0)
≈ sum(P(return_range positive bins))

P(drawdown > 5%)
≈ sum(P(drawdown bins above 5%))

expected return rank
应与 return percentile query 单调一致
```

一致性 Loss 只作为小权重约束，不能压制真实标签。

## 65. 数据与防泄漏合同

### 65.1 Split

继续使用：

```text
train: pre-2020
validation_select: checkpoint / hyperparameter selection
prediction: formal QuantX
```

### 65.2 Quantile Boundary

所有分位边界、代表值、winsorize 参数：

```text
只允许从 train split 计算
```

validation 和 prediction 只能使用冻结后的映射。

### 65.3 Cross-sectional Label

同日 percentile 使用未来 outcome 是监督标签的一部分，不是输入特征。

允许：

```text
使用同日未来收益定义真实横截面排名
```

禁止：

```text
把同日未来横截面统计作为模型输入
```

### 65.4 Industry Metadata

行业超额收益必须使用 point-in-time 行业信息。

如果只能获得当前行业分类：

```text
不进入正式 E3 Core
```

### 65.5 Execution Contract

所有核心收益标签优先使用：

```text
T+1 close entry
真实可执行 exit
交易成本
```

如果标签不处理不可成交：

```text
必须另外报告 fill rate 和 rejection breakdown
```

### 65.6 Multiple Testing

2020-2026 prediction period 已被多次观察。

E3 必须：

```text
先在 validation_select 固定 Query、loss、score 权重
再只执行一轮预注册正式回测
同时报告逐年和 rolling-window 结果
```

任何基于 prediction 结果继续调参的实验必须标记：

```text
development / data-mined
```

不能再称为完全 untouched OOS。

## 66. E3 训练阶段

### 66.1 Warm Start

首选：

```text
E2 epoch 3
```

原因：

```text
H7 return RankIC 最好
风险表示仍然稳定
正式纯收益回测最好
```

### 66.2 Stage A

```text
1 epoch
AE frozen
State Backbone frozen
训练 Query/Choice Decoder
```

目的：

```text
让新 Query 先适配共享语义空间
避免新 label 立即破坏 State Representation
```

### 66.3 Stage B

```text
3-5 epochs
State Backbone trainable
state lr <= 1e-5
decoder lr <= 5e-5
```

### 66.4 Checkpoint 选择

不能只按 aggregate accuracy。

优先级：

```text
1. H7 net-utility RankIC
2. H7 top-tail Listwise metric
3. H7 return RankIC
4. H7 drawdown RankIC
5. calibration / Brier
```

## 67. E3 实验矩阵

### 67.1 A：E2 Control

```text
E2 epoch 3
return-only score
```

### 67.2 B：E3 Core Distribution

```text
OutcomeStore V2
24 Core Query
CE + Brier + Ordinal
无 Listwise
```

用于判断标签质量本身是否提升 return RankIC。

### 67.3 C：E3 Core + Hard Pair

```text
B
+ top-tail hard pair
```

### 67.4 D：E3 Full

```text
C
+ H7 Listwise
+ risk ranking
+ consistency
```

D 是正式候选。

### 67.5 E：Old Reward Return + Jev Risk

诊断组合：

```text
old reward return score
+ E3 risk score
```

用于判断：

```text
风险 Query 是否能提升旧 baseline
收益差距是否主要来自 Jev return representation
```

该组合是诊断，不是统一模型最终目标。

## 68. E3 验收指标

### 68.1 离线

必须报告：

```text
每 Query candidate distribution
normalized entropy
每年 label drift
RankIC mean / median / positive ratio
Top1/5/10/20 lift
Listwise NDCG
Pairwise accuracy by rank boundary
Brier / ECE
```

### 68.2 正式 QuantX

固定报告：

```text
Total Return
Annual Return
MDD
Sharpe
Sortino
Calmar
Win Rate
Profit Factor
Trade Count
Reject Count
Total Cost
Average Holding Days
yearly returns
rolling 12-month Sharpe
```

### 68.3 晋级条件

E3 Full 至少满足：

```text
Sharpe > 0.644
MDD >= -38.03%
```

并满足以下至少一项：

```text
Total Return > +203.91%
每年收益稳定性明显改善
rolling Sharpe 显著优于旧 baseline
```

如果：

```text
Sharpe 提升仅来自极低仓位
交易次数显著减少
成本合同变化
不可成交股票被静默删除
```

则不得晋级。

## 69. E3 Artifact 设计

建议：

```text
tmp/jev-like-financial-decision-v1/
  artifacts/
    outcome_store/e3_v1/
      manifest.json
      return_outcomes.mmap
      risk_outcomes.mmap
      path_outcomes.mmap
      rank_outcomes.mmap

    query_registry/
      e3_core_v1.json
      e3_core_v1.manifest.json

    query_catalog/
      e3_core_v1.html
      e3_core_v1.json

  runs/
    <e3_run_id>/
      checkpoints/
      logs/
      tensorboard/
      query_metrics/
      score_artifacts/
      formal_quantx/
```

OutcomeStore manifest 必须包含：

```text
source data checksum
split contract
outcome formulas
train-only quantile boundaries
industry metadata version
cost assumptions
execution assumptions
validity rules
```

## 70. E3 实现顺序

严格顺序：

```text
1. OutcomeStore V2 schema + unit tests
2. OutcomeStore builder + small-data parity
3. E3 Query Registry / Catalog
4. Label mapper + candidate distribution audit
5. Query quality report
6. Listwise / hard-pair dataset
7. Trainer integration
8. CPU unit tests
9. single-GPU smoke
10. multi-GPU preflight
11. formal training
12. fixed score export
13. pre-registered QuantX
```

在第 5 步 Query quality report 通过前：

```text
禁止启动正式训练
```

## 71. E3 待确认项

实现前仍需最终冻结：

1. robust return 主定义使用 `median(R5,R7,R10)` 还是 `mean(exit T+5..T+9)`。
2. market excess 使用同日 median 还是指定指数。
3. point-in-time industry metadata 是否可用。
4. net utility 的固定 cost 与 `lambda_drawdown`。
5. barrier 的 TP/SL 参数。
6. E3 Core 是否保留 H3/H15，还是第一版只做 H7。

默认推荐：

```text
robust return:
  median(R5,R7,R10)

market excess:
  same-date median

industry excess:
  有 point-in-time 数据才启用

lambda_drawdown:
  0.25 and 0.50

barrier:
  TP5/SL3
  TP8/SL5

windows:
  H3, H7, H15
```

## 72. 变更日志

### v0.4.0 - 2026-09-20

```text
记录 E2 Scale 训练与正式 QuantX 结果
确认统一 Query/Head 可学习收益与回撤
新增 OutcomeStore V2 设计
新增 E3 Core 24 Query 合同
新增横截面、超额收益、稳健退出、路径质量与效用 Query
新增 Top-tail Listwise / Hard Pair 训练合同
新增二分类派生与一致性规则
新增数据防泄漏与 multiple-testing 约束
新增 E3 实验矩阵和晋级标准
```

## 73. E5：H1～H8 多 Horizon 语义扩展

### 73.1 实验目标

E5 用同一个模型同时学习 H1～H8，验证：

```text
增加 Horizon、Query 语义和 Candidate 粒度后，
模型是否真正理解不同持有周期，
而不是继续复用一条通用股票质量排序。
```

本轮只扩展数据与训练合同，不改变主体结构：

```text
Frozen AE
+ State Backbone
+ Query/Choice Decoder
+ Candidate Head
```

保留 Adaptive E3 中已经验证的：

```text
Query learning-progress scheduler
Date-section replay
Persistent noise filter
Candidate permutation consistency
```

明确不加入：

```text
Listwise
Top-tail hard pair
特异化 numeric encoder
额外 horizon id / task id embedding
```

### 73.2 唯一 Horizon 合同

E5 将 `H` 定义为完整的 close-to-close 持有间隔数：

```text
entry = close(T+1)
exit  = close(T+H+1)
holding_intervals = H
terminal_return_H = close(T+H+1) / close(T+1) - 1
```

对应关系：

| Horizon | 训练价格区间 | 完整持有间隔 | QuantX 最小持有日 |
| --- | --- | ---: | ---: |
| H1 | T+1 → T+2 | 1 | 1 |
| H2 | T+1 → T+3 | 2 | 2 |
| H3 | T+1 → T+4 | 3 | 3 |
| H4 | T+1 → T+5 | 4 | 4 |
| H5 | T+1 → T+6 | 5 | 5 |
| H6 | T+1 → T+7 | 6 | 6 |
| H7 | T+1 → T+8 | 7 | 7 |
| H8 | T+1 → T+9 | 8 | 8 |

禁止再使用以下含混表达：

```text
window=7，但终点为 T+7
H7 回测使用 holding_days=6
```

旧 E2/E3/E4 结果仍按其历史口径保留，不追溯改写；E5 新工件必须在
manifest 中记录 `horizon_contract=close_interval_v1`。

### 73.3 64 个 Canonical Tasks

每个 H1～H8 固定构建 8 个互补任务：

| Task family | 连续 outcome | 学习目标 |
| --- | --- | --- |
| terminal return | `terminal_return_H` | 绝对终点收益 |
| cross-sectional return | `cross_return_pct_H` | 同日横截面收益排名 |
| market excess return | `market_excess_return_H` | 相对同日市场中位数的超额收益 |
| drawdown safety | `drawdown_safety_pct_H` | 最大回撤安全性排名 |
| downside-vol safety | `downside_vol_safety_pct_H` | 下行波动安全性排名 |
| MFE/MAE quality | `mfe_mae_quality_pct_H` | 上行空间相对下行风险 |
| path efficiency | `path_efficiency_pct_H` | 收益路径效率 |
| net utility | `net_utility_pct_H` | 收益与回撤共同作用的净效用 |

总量：

```text
8 horizons × 8 families = 64 canonical tasks
```

任务优先使用平衡的 ordinal range 分类，不额外复制大量阈值二分类。
`terminal return` 作为绝对锚点，其余任务优先使用同日 percentile。

### 73.4 OutcomeStore E5

每个 `row_id`、每个 `H` 保存以下连续量：

```text
terminal_return_H
cross_return_pct_H
market_excess_return_H
max_drawdown_H
drawdown_safety_pct_H
downside_volatility_H
downside_vol_safety_pct_H
mfe_H
mae_H
mfe_mae_ratio_H
mfe_mae_quality_pct_H
path_efficiency_H
path_efficiency_pct_H
net_utility_H
net_utility_pct_H
```

路径定义：

```text
prices = close(T+1), ..., close(T+H+1)
returns = prices[1:] / prices[:-1] - 1
```

风险定义沿用第 59 节，但必须基于上述统一路径。净效用固定为：

```text
net_utility_H =
  terminal_return_H
  - 0.50 * max_drawdown_H
  - fixed_round_trip_cost
```

E5 V1 的 `fixed_round_trip_cost` 必须来自当前正式回测合同，并写入
manifest；训练后不得根据 prediction 回测结果搜索该参数。

### 73.5 Query 模板合同

每个 canonical task 提供 8 个真正不同的受控模板。模板必须改变句法，
不能只在原句前增加“请回答”“请选择”等前缀。

以 H3 terminal return 为例：

```text
1. 从 T+1 收盘买入并持有三个完整交易间隔，最终收益属于哪个区间？
2. 以 P(T+1) 为基准，P(T+4)/P(T+1)-1 最符合哪个选项？
3. 若持仓跨越三个 close-to-close 区间，最终盈亏属于什么水平？
4. 信号出现后的下一次收盘建仓，到第四次收盘退出，回报落在哪一档？
5. 比较 T+4 与 T+1 的收盘价，这段持有期的价格变化应归入哪一区间？
6. 完成三个交易日间隔后卖出，该笔交易的终点收益属于哪种描述？
7. 从次日收盘开始计时，经过三次相邻收盘变化后，累计收益最接近哪个范围？
8. 如果买入价是 T+1 收盘、卖出价是 T+4 收盘，盈亏幅度对应哪个候选项？
```

其他 family 必须使用与自身 outcome 相符的独立模板词汇，例如：

```text
横截面：同日股票池、相对位置、收益分位
市场超额：市场中位数、相对市场、超额部分
回撤：路径峰值、峰谷损失、最大回落
下行波动：负收益波动、下行不稳定性
MFE/MAE：最大有利变化、最大不利变化、收益风险空间
路径效率：累计净变化、路径总波动、趋势效率
净效用：收益扣除回撤惩罚与交易成本
```

所有模板表达同一个 label，不改变 outcome 或 candidate 边界。

### 73.6 Candidate Partition 与动态 N

每个连续 ordinal outcome 支持三个可共同细分的分区：

```text
coarse3
fine5
fine7
```

边界只从 train split 计算并冻结。推荐使用训练分位：

```text
coarse3: 1/3, 2/3
fine5:   1/5, 2/5, 3/5, 4/5
fine7:   1/7, 2/7, 3/7, 4/7, 5/7, 6/7
```

三种分区统一投影到 `LCM(3,5,7)=105` 个等宽分位微区间，再计算概率
一致性；不能假设 candidate 数相同，也不能依赖固定位置。Candidate sequence
继续支持：

```text
identity
rotate
reverse
deterministic random
```

每个 candidate 至少提供以下表达面：

```text
自然语言区间
百分比
小数
基点（适用于收益与风险量）
```

这些表达只改变文本表面，不改变 canonical interval。

### 73.7 Consistency Loss

E5 的 consistency 分为两类：

```text
L_surface:
  同一 task、同一 partition 在 Query 改写和 Candidate 置换前后，
  对齐到 canonical candidate 后的概率应一致。

L_partition:
  同一 task 的 fine7/fine5/coarse3 概率，
  投影到共同区间后应一致。
```

初始权重固定为：

```text
surface/permutation consistency = 0.03
partition consistency           = 0.05
```

总 loss：

```text
L_total =
  L_supervised
  + L_pairwise
  + adaptive replay/noise-filter terms
  + 0.03 * L_surface
  + 0.05 * L_partition
```

Consistency 只约束同一金融事实的不同表达，不能跨 Horizon 强制概率相等。

### 73.8 Sampling

每个训练 step 使用：

```text
query_slots = 12
```

长期目标：

```text
每个 Horizon 获得近似相同的有效监督量
每个 family 获得近似相同的基础曝光
partition 采样优先级：fine5 > coarse3 > fine7
```

建议初始比例：

```text
fine5:   50%
coarse3: 25%
fine7:   25%
```

Adaptive scheduler 可以根据 learning progress 调整 task 权重，但：

```text
任何 Horizon 的长期采样占比不得低于均匀占比的 50%
任何单一 task 不得长期占据超过 4 个 query slots
```

### 73.9 Warm-start 与训练阶段

Warm-start：

```text
Adaptive E3 epoch 3:
tmp/weak-to-strong-diffusion-v1/market_all_close2close_balanced_v2/
runs/jev-e3c-adaptive-core24-ft-e2e3-5gpu-4ep-20260920-201253/
checkpoints/jev_epoch_003.pt
```

训练计划：

```text
GPU: 0,1,3,4,5
Epoch 1:
  AE frozen
  State Backbone frozen
  只训练 Query/Choice Decoder 与 Candidate Head

Epoch 2～5:
  AE frozen
  State Backbone 解冻
  state lr = 1e-5
  decoder/head lr = 5e-5
```

Batch size 先通过单卡 smoke 和五卡 preflight 决定，以不 OOM 且保持
日期 section 完整为准。不能仅为吃满显存破坏同日 pairwise 语义。

### 73.10 数据与防泄漏

必须满足：

```text
1. 所有 quantile boundary、candidate representative value 只由 train 计算。
2. validation/prediction 只加载冻结后的 registry 与映射。
3. 同日横截面 percentile 只作为 label，不进入 state。
4. Query template 选择由 row_id/query_id/epoch 的确定性种子产生。
5. prediction score 导出固定使用 canonical template 和 canonical partition。
6. H8 要求至少存在 T+9 收盘价；不足时 valid=false，禁止向前填充。
7. 回测买卖价格与训练 Horizon 合同严格一致。
```

### 73.11 文档 Review 结论

E5 实现前 review 检查：

| 检查项 | 结论 |
| --- | --- |
| H1～H8 label 与回测 holding days 对齐 | 通过 |
| 64 tasks 数量和 family 定义一致 | 通过 |
| Query 改写不是简单前缀 | 通过 |
| Candidate N 支持 3/5/7 动态变化 | 通过 |
| Partition 使用 105 个共同微区间对齐 | 通过 |
| Consistency 权重低于主要监督 | 通过 |
| 不引入 Listwise/Top-tail hard pair | 通过 |
| Warm-start checkpoint 明确 | 通过 |
| Train-only 边界和 prediction 冻结映射明确 | 通过 |

Review 结论：

```text
E5 设计内部一致，可以进入实现。
```

## 74. E5 验证与回测矩阵

### 74.1 离线验证

每个 Horizon、每个 family 至少报告：

```text
accuracy
macro F1
Brier
RankIC mean / median / positive ratio
candidate distribution
surface consistency JS
partition consistency JS
```

另做 Horizon 可辨识性诊断：

```text
固定 State 与 Candidate，只替换 Query Horizon：
H1 ... H8 score 不应继续接近完全相同。
```

### 74.2 Score 导出

固定 checkpoint 一次性导出：

```text
H1 return / risk / utility scores
...
H8 return / risk / utility scores
```

每个 Horizon 至少生成：

```text
absolute return score
cross-sectional return score
market excess score
risk composite
net utility score
```

第一轮正式 Horizon Frontier 使用 `cross-sectional return score` 排 Top10；
风险与净效用组合作为预注册的补充对照，不能看完 H1～H8 后逐周期挑权重。

### 74.3 QuantX 回测

统一设置：

```text
TopK = 10
entry = T+1 close
exit = T+H+1 close
minimum holding_days = H
same universe
same transaction cost
same price-jump rules
same prediction date range
```

回测矩阵：

```text
H1 hold>=1
H2 hold>=2
H3 hold>=3
H4 hold>=4
H5 hold>=5
H6 hold>=6
H7 hold>=7
H8 hold>=8
```

每个配置必须先 dry-run，再执行正式 QuantX。

### 74.4 Horizon Frontier 报告

每个 Horizon 固定报告：

```text
RankIC
Top10 gross return
Top10 net return
Annual return
Sharpe
MDD
Trade count
Reject count
Total cost
Average return per trade
Break-even edge
```

E5 的主要研究问题不是只挑出收益最高的 H，而是回答：

```text
1. 从哪个 Horizon 开始出现稳定正 RankIC？
2. 短 Horizon 的换手优势能否覆盖噪声和交易成本？
3. 多 Horizon 联合训练是否改善 H3/H4/H5？
4. 不同 Horizon 的 score 是否仍然高度同质？
5. 增加 Query/Candidate scale 是否带来可测量的能力扩展？
```

## 75. E5 实现顺序与工件

实现顺序：

```text
1. E5 OutcomeStore 与 Horizon 对齐单元测试
2. 64-task canonical registry
3. 8-template renderer
4. coarse3/fine5/fine7 partition 与映射
5. Semantic View Cache V2
6. surface/partition consistency
7. Trainer 集成
8. 全量单元测试
9. 单卡 smoke
10. 五卡 preflight
11. 五卡正式训练
12. 固定 checkpoint 离线验证
13. H1～H8 score 导出
14. 8 个 QuantX dry-run
15. 8 个正式回测
16. Horizon Frontier 汇总
```

新增工件使用独立版本，不能覆盖 E3/E4：

```text
artifacts/outcome_store/e5_h1_h8_v1/
artifacts/query_registry/e5_h1_h8_v1.json
artifacts/query_catalog/e5_h1_h8_v1.*
artifacts/text_embeddings/e5_h1_h8_semantic_views_v2.*
runs/jev-e5-h1-h8-*/
configs/strategies/generated/jev_e5_horizon_frontier_v1/
```

### v0.5.0 - 2026-09-20

```text
冻结 E5 H1～H8 close-interval Horizon 合同
定义 8 horizons × 8 families = 64 canonical tasks
定义每个 task 的 8 套真实 Query 改写
定义 coarse3/fine5/fine7 动态 Candidate partition
新增 surface/permutation 与 partition consistency
固定 Adaptive E3 epoch 3 warm-start
固定 H1～H8 QuantX 回测矩阵与 Horizon Frontier 指标
完成实现前一致性 review
```

## 76. E5 H1～H8 训练与正式回测结果

### 76.1 训练工件

正式 H20 任务：

```text
acloud-9d77a287bb33
1 x 8 H20
BF16
stocks_per_rank = 2048
global stock batch = 16384
SDPA backend = math
gradient checkpointing = false
```

训练输出：

```text
/horizon-bucket/saturn_v_dev/mingxiao.li/checkpoints/quantization/
jev_e5_h1_h8_v1/runs/
jev-e5-h1-h8-resume-e1-unfreeze-h20-8gpu-4ep-20260921-114109
```

任务状态：

```text
Succeeded
```

固定回测 checkpoint：

```text
checkpoints/jev_epoch_004.pt
```

最终验证：

```text
choice accuracy:       0.28685
pairwise accuracy:     0.57824
partition consistency: 0.00479
```

### 76.2 Horizon RankIC

| Horizon | Mean RankIC | Median RankIC | Positive Ratio |
| --- | ---: | ---: | ---: |
| H1 | 0.0659 | 0.0650 | 63.98% |
| H2 | 0.0754 | 0.0736 | 67.46% |
| H3 | 0.0818 | 0.0801 | 68.81% |
| H4 | 0.0872 | 0.0903 | 69.72% |
| H5 | 0.0899 | 0.0927 | 68.30% |
| H6 | 0.0922 | 0.0919 | 69.07% |
| H7 | 0.0945 | 0.0987 | 69.46% |
| H8 | 0.0971 | 0.1018 | 70.30% |

RankIC 随 Horizon 基本单调提高，说明短周期目标仍然更难预测。

但 H1～H8 score 的 Spearman 相关性为：

```text
0.975 ～ 0.999
```

因此模型虽然能从 Query 中获得一部分 Horizon 信息，但主要排序仍是一条
高度共享的股票质量因子，尚未形成充分独立的期限条件化策略。

### 76.3 正式 QuantX Horizon Frontier

统一口径：

```text
Top10
T+1 close 执行
H1 hold>=1
...
H8 hold>=8
price_jump_limit=9.5%
commission=0.05%
stamp tax=0.01%
slippage=0.10%
```

| H | Total Return | Annual Return | Sharpe | MDD | Trades | Total Cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | -95.42% | -38.15% | -1.554 | -95.42% | 27,500 | 56.93M |
| H2 | -52.51% | -10.95% | -0.372 | -68.89% | 15,355 | 59.04M |
| H3 | +25.28% | +3.57% | 0.118 | -58.08% | 10,263 | 70.10M |
| H4 | +84.74% | +10.03% | 0.336 | -51.67% | 7,717 | 70.62M |
| H5 | +6.08% | +0.92% | 0.033 | -57.72% | 6,183 | 34.08M |
| H6 | +79.89% | +9.58% | 0.366 | -40.78% | 5,159 | 37.12M |
| H7 | +37.45% | +5.08% | 0.241 | -44.77% | 4,426 | 23.60M |
| H8 | +121.80% | +13.21% | 0.576 | -35.15% | 3,868 | 32.86M |

### 76.4 结论

1. H1/H2 的高换手没有产生复利优势，预测噪声与交易成本占主导。
2. H3 首次转正，但 Sharpe 仅 0.118，短周期因子仍不稳定。
3. H4、H6、H8 有明显收益，其中 H8 的收益、Sharpe 和回撤均为 E5 最优。
4. Horizon Frontier 不是平滑单调的，H5/H7 明显弱于相邻期限。
5. 高度相关的 score 却产生显著不同的回测结果，说明 Top10 边界排序和持有期执行合同会放大小幅排名差异。
6. E5 证明多 Horizon 统一训练可以同时得到 H1～H8 的正 RankIC，但没有证明能力随 Query/Candidate 数量持续 scale up。
7. 下一步不应继续机械增加 Horizon 数量；应提高 Horizon 条件可辨识度，并使用更贴近 Top10 边界的高质量监督。

## 77. E5 与旧 Reward Model 公平对照

### 77.1 对照合同

为排除仓位管理差异，以下对照统一使用：

```text
2020-01-02 ～ 2026-06-02
all-A
Top10
portfolio_target
target_weight
max_position_weight = 12%
cash_use_ratio = 98%
相同交易成本、涨跌停和 T+1 close 执行规则
```

### 77.2 H7 同持有期对照

| Score | Total Return | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| Old Reward composite H7 | +203.91% | 0.644 | -38.03% |
| E5 terminal-return H7 | +109.28% | 0.366 | -50.99% |
| E5 cross-return H7 | +50.40% | 0.333 | -42.44% |

H7 上旧 Reward Model 明显优于 E5。该差距不能由仓位配置或回测日期解释。

### 77.3 H8 同持有期对照

| Score | Total Return | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| Old Reward return-only | +77.10% | 0.264 | -43.41% |
| Old Reward composite | +182.66% | 0.602 | -38.77% |
| E5 terminal-return | +120.12% | 0.413 | -46.34% |
| E5 cross-return | **+126.92%** | **0.609** | **-30.60%** |
| E5 net-utility | +118.20% | 0.531 | -35.45% |
| E5 70/30 return-risk blend | +46.37% | 0.302 | -25.98% |

### 77.4 公平对照结论

1. E5 terminal-return 明显优于旧 Reward 的 return-only score，说明统一 Query 模型并非完全损失收益预测能力。
2. 旧 Reward composite 仍保留最高绝对收益，说明旧模型的 return + risk 两阶段排序更擅长捕捉高收益 Top10。
3. E5 cross-return 的 Sharpe 略高于旧 Reward composite，且最大回撤改善约 8.2 个百分点，但累计收益低约 55.7 个百分点。
4. E5 net-utility 相比 E5 terminal-return，将最大回撤从 -46.34% 改善到 -35.45%，同时仅损失约 1.9 个百分点累计收益；风险 Query 确实学到了有效信息。
5. 70/30 risk blend 将回撤进一步压到 -25.98%，但收益下降到 +46.37%，说明 30% 风险权重明显过强。
6. 当前最合理的 E5 输出是 cross-return H8；若强调收益与风险平衡，net-utility H8 次之。
7. E5 仍未超过旧 Reward H7 的 `+203.91% / Sharpe 0.644`，核心瓶颈仍是 Top10 收益尾部的区分能力，而不是模型完全不会学习风险。

## 78. 年度与月度收益分布可视化

回测区间：

```text
2020-01-02 ～ 2026-06-02
```

旧 Reward H7 的：

```text
累计收益：+203.91%
复合年化：18.91%
```

即初始资金在约 6.4 年内增长到约 3.04 倍。它不是每年稳定获得
约 18.91%，而是多个高收益年份和回撤年份共同复合后的 CAGR。

年度收益：

| 策略 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reward Composite H7 | +45.3% | +16.2% | +6.7% | +32.1% | -0.1% | +41.1% | -9.4% |
| Reward Composite H8 | +25.0% | +14.1% | +7.6% | +45.5% | +16.2% | +20.0% | -9.2% |
| E5 Cross H8 | +10.6% | +25.0% | -3.9% | +1.7% | +36.3% | +28.8% | -4.2% |
| E5 Net Utility H8 | +7.3% | +21.5% | +6.1% | -3.0% | +25.9% | +26.4% | +2.2% |
| E5 Terminal H8 | -7.6% | +4.4% | +12.9% | +1.5% | +15.6% | +39.9% | +23.0% |
| E5 Risk Blend H8 | +8.9% | +14.3% | -3.4% | +4.8% | +6.7% | +15.4% | -5.8% |

可视化约定：

```text
红色：正收益
绿色：负收益
2026：仅截至 6 月 2 日
```

![策略总览](assets/jev_e5_strategy_diagnostics_v1/strategy_overview.png)

![月度收益热力图](assets/jev_e5_strategy_diagnostics_v1/monthly_return_heatmaps.png)

详细说明与 CSV：

```text
docs/experiments/assets/jev_e5_strategy_diagnostics_v1/
```

## 79. Market-only JEV v1：策略无关市场情绪与仓位预测器

### 79.1 目标与边界

第一版单独训练一个市场预测器：

```text
过去 60 个交易日的全市场横截面特征
+ 当前月份 / 月内交易周 / 季度 / 月内阶段
+ Market Query
+ Dynamic Candidates
→ Candidate probabilities
→ 策略无关的目标市场暴露
```

它不输入：

```text
个股 OHLCV
个股 AE latent
Reward Top10 股票名单
账户状态
未来 Reward 策略收益
```

旧 Reward Composite 继续负责：

```text
买哪些股票
```

Market-only JEV 只负责：

```text
承担多少市场风险
```

### 79.2 输入状态

复用现有：

```text
tmp/kronos-classification-signal-v1/data/
market_cross_section_v1_float32.mmap
```

输入形状：

```text
[B, 60, 20]
```

20 个字段包括：

```text
up/down/flat ratio
全市场平均与中位日收益
不同涨跌幅区间占比
成交额与成交量
成交额/成交量 20 日 Z-score
头部成交集中度
横截面收益离散度
```

V1 不等待沪深 300、中证 500、中证 1000 历史数据补齐；指数和风格差进入 V2。

### 79.3 日历语义

日历上下文必须进入完整 Query 文本：

```text
月份
月内第几个交易周
季度
月初 / 月中 / 月末
```

不输入具体年份。

例如：

```text
当前处于九月第三个交易周，属于第三季度月中阶段。
从下一交易日收盘开始，未来七个完整交易间隔内，
全 A 市场等权收益处于哪个历史区间？
```

训练时使用 calendar dropout，防止模型只学习月份季节性。

### 79.4 时间与 Split 合同

```text
State 截止：T close
执行：T+1 close
H7 outcome：T+1 close → T+8 close
```

市场日收益 Outcome 使用：

```text
T+2 ... T+H+1 的 close-to-close 横截面日收益
```

Split：

```text
train:       2010-01-01 ～ 2017-12-31
validation:  2018-01-01 ～ 2018-12-31
calibration: 2019-01-01 ～ 2019-12-31
prediction:  2020-01-01 ～ 数据末日
```

每个 split 的最后 H10 信号必须保证未来窗口完全落在本 split 内，形成自然 purge。

### 79.5 OutcomeStore

Horizon：

```text
H3 / H5 / H7 / H10
```

每个 Horizon 保存六类策略无关 Outcome：

1. `equal_weight_return`
2. `median_stock_return`
3. `average_up_ratio`
4. `max_drawdown`
5. `downside_volatility`
6. `market_opportunity`

市场机会指数使用冻结的 train CDF：

```text
opportunity =
  mean(
    CDF(equal_weight_return),
    CDF(median_stock_return),
    CDF(average_up_ratio),
    1 - CDF(max_drawdown),
    1 - CDF(downside_volatility)
  )
```

它不依赖任何选股策略。

### 79.6 Query 与 Candidate

```text
4 horizons × 6 families = 24 canonical tasks
```

每个任务使用：

```text
coarse3
medium4
fine5
```

总计：

```text
24 × 3 = 72 Query variants
```

候选分箱只由 2010～2017 train split 计算。

3/4/5 类概率统一投影到：

```text
LCM(3,4,5) = 60 个分位微区间
```

训练保留：

```text
8 套真实 Query 改写
Candidate identity / rotate / reverse / random
surface consistency
partition consistency
跨日期 pairwise
hard-date replay
persistent noise filter
calendar dropout
market feature masking
```

### 79.7 模型结构

新增轻量 Market Encoder：

```text
[B,60,20]
→ Linear(20,128)
→ 2-layer Temporal Transformer
→ Adaptive Pool 60→27
→ [B,27,128]
```

后续直接复用 E5：

```text
Jev State Backbone
Query/Choice Decoder
Candidate Head
```

Warm-start：

```text
E5 epoch 4
```

训练阶段：

```text
Stage A:
  冻结 E5 主体
  只训练 Market Encoder

Stage B:
  解冻最后 2 个 State Blocks
  解冻 Decoder 与 Candidate Head
  小学习率微调
```

### 79.8 部署 Query 与仓位

部署主 Query：

```text
H7 market_opportunity fine5
```

Candidate 基础仓位：

```text
历史最弱 20% → 0%
偏弱          → 25%
中性          → 50%
偏强          → 75%
历史最强 20% → 100%
```

使用完整概率：

```text
raw_exposure = sum(P(candidate_i) * exposure_i)
```

H7 drawdown safety fine5 给出风险上限：

```text
risk_cap = sum(P(safety_i) * exposure_i)
target_exposure = min(raw_exposure, risk_cap)
```

在 2019 calibration split 固定平滑参数：

```text
EMA
单日最大仓位变化
最低状态持续天数
```

2020～2026 不再调参。

### 79.9 第一轮实验矩阵

1. 无市场控制的 Reward Composite H7。
2. 未来标签 Oracle exposure，仅用于上限诊断。
3. Logistic / LightGBM market baseline。
4. Market-only JEV raw exposure。
5. Market-only JEV risk-capped exposure。
6. Market-only JEV smoothed exposure。

固定报告：

```text
Total Return
Sharpe
MDD
月度胜率
最差月
连续亏损月
共同亏损月份召回率
错误空仓率
错过上涨收益
满仓 / 半仓 / 空仓占比
```

### 79.10 验收

```text
月度胜率 >= 72%
Sharpe > 0.70
MDD 至少改善 5 个百分点
总收益 >= Reward baseline 的 85%
```

必须额外通过：

```text
完整模型 > 仅 Calendar 模型
完整模型 > 无 Calendar 模型
H3/H5/H7/H10 Query 对自身 Horizon 的指标最佳
Candidate permutation Spearman > 0.99
Query surface JS < 0.01
```

## 80. Market JEV v2：Canonical CDF 与密集 Pairwise

### 80.1 为什么 72 个 Query 不等于 72 份独立任务

Market JEV v1 的：

```text
24 canonical outcomes × coarse3/medium4/fine5 = 72 Query
```

实际仍然只有 24 条不同的连续 outcome 序列。3/4/5 分类只是对同一个未来结果进行不同粒度的切箱，并没有产生新的行情事实。

因此 v2 将训练语义改成：

```text
24 个 canonical task
+ 动态 Candidate partition view
+ 动态 Query/Candidate surface view
```

`coarse3/medium4/fine5` 不再被视为三个互相独立的任务，而是同一 canonical outcome 的三种观察方式。

### 80.2 Canonical label

每个 canonical task 使用训练集经验 CDF，将原始 outcome 转换为统一的历史分位：

```text
z = TrainCDF(outcome)
z ∈ [0, 1]
```

例如某日未来 H7 全市场等权收益处于训练历史第 14% 分位：

```text
z = 0.14
```

则它在不同 Candidate partition 下分别对应：

```text
coarse3 → 历史偏弱
medium4 → 历史显著偏弱
fine5   → 历史最弱
```

这些 label 都来自同一个 `z`，不能作为三份独立市场样本。

### 80.3 动态 Candidate partition

每次训练先抽取 canonical task，再为每个 task 抽取两个不同 Candidate 粒度：

```text
canonical task
├── partition view A: N=3/4/5
└── partition view B: N=3/4/5，且不同于 A
```

两个 partition 都保留：

```text
Query 模板变化
Candidate 文本变化
Candidate 顺序变化
```

模型仍然接收可变数量 Candidate，并输出对应 logits；模型结构和 E5 Query/Choice Decoder 接口不变。

### 80.4 CDF ordinal supervision

Candidate 概率会转换成候选边界上的累计概率。例如五分类概率为：

```text
[p0, p1, p2, p3, p4]
```

则模型同时预测：

```text
P(z <= 20%) = p0
P(z <= 40%) = p0 + p1
P(z <= 60%) = p0 + p1 + p2
P(z <= 80%) = p0 + p1 + p2 + p3
```

CDF loss 对所有有效边界进行监督。它能够区分“错一档”和“错多档”，并让不同 N 的 Candidate 表达同一个潜在历史分位分布。

### 80.5 Pairwise 主监督

v1 使用原始 outcome representative value 作为排序 score。收益率通常只有几个百分点，而 pairwise temperature 为 1，导致不同日期的 score 差异过小，pairwise loss 长期接近随机基线：

```text
log(2) ≈ 0.693
```

v2 统一把任意 N 个 Candidate 投影到 0～1 历史分位：

```text
score = sum(P(candidate_i) × quantile_midpoint_i)
```

每个 batch 内枚举所有满足以下条件的有向日期对：

```text
z_better >= z_worse + pair_gap_quantile
```

默认：

```text
pair_gap_quantile = 0.15
pairwise_temperature = 0.15
max_pairs_per_task = 4096
```

因此一个 64 日期 batch、6 个 canonical task 可以产生约八千个可靠 pair，而不是每个 anchor 只随机选择一个 bad case。

总损失默认权重：

```text
0.20 × Candidate CE
+ 1.00 × CDF loss
+ 2.00 × Pairwise loss
+ 0.05 × Brier
+ 0.03 × Surface consistency
+ 0.10 × Partition consistency
```

Pairwise 是权重最大的主监督，CDF 负责概率与候选边界校准。

### 80.6 容量与 Replay

v2 不再解冻约 2942 万参数的 E5 Decoder：

```text
冻结完整 E5 decision backbone
训练 Market Encoder
+ 32 维 bottleneck residual adapter
```

实际可训练参数：

```text
284,192
```

v1 的 batch replay 会用 replay batch 替换 fresh batch。v2 第一轮关闭 replay，确保每个 epoch 的全部新日期都参与训练。后续若恢复 replay，只允许作为 fresh loss 之外的附加监督。

### 80.7 第一轮实现验证

实现：

```text
tmp/jev-like-financial-decision-v1/train_market_jev_cdf_v2.py
```

正式验证 Run：

```text
tmp/weak-to-strong-diffusion-v1/
market_all_close2close_balanced_v2/runs/
market-jev-cdf-v2-16ep-r2-20260921
```

从 Market JEV v1 最佳 epoch 6 warm-start，运行 16 epoch 上限并设置 patience=3。

训练规模：

```text
train dates: 1874
validation dates: 232
canonical tasks: 24
每步抽取 canonical tasks: 6
每个 task 同时训练两个 Candidate partition
```

第一轮每个 epoch 实际生成约：

```text
训练 pair: 245,000
验证 pair: 98,304
```

结果：

| Epoch | Train Pair Acc | Validation Pair Acc | Validation CDF | Selection Objective |
|---:|---:|---:|---:|---:|
| 1 | 58.86% | 58.87% | 0.6305 | 2.3012 |
| 2 | 60.31% | 60.47% | 0.6816 | 2.3754 |
| 3 | 62.18% | 58.64% | 0.6835 | 2.4036 |
| 4 | 62.98% | 52.81% | 0.6682 | 2.3712 |

epoch 4 触发 early stopping，综合目标最佳 checkpoint 为 epoch 1。

同时单独保存：

```text
best_checkpoint.pt           → 综合 CDF/Pairwise 校准最佳，epoch 1
best_pairwise_checkpoint.pt  → Pairwise accuracy 最佳，epoch 2
```

初步结论：

1. 密集 pairwise 已经不再处于随机状态，验证排序准确率一度达到 60.47%。
2. CDF 概率校准在第一轮后恶化，说明 1874 个日期仍不足以支持持续拟合。
3. 约 24.5 万个 pair 是大量排序约束，不是 24.5 万个独立市场样本；这些 pair 共享同一批日期，不能等价替代真实行情数据。
4. 当前应分别保存“综合校准最佳”和“pairwise accuracy 最佳”checkpoint，随后通过 H7 仓位回测判断哪一种更适合实际策略。

## 81. Market JEV Pairwise R1：先学习明显好坏，再做弱势减仓

### 81.1 训练目标

R1 从 Market JEV v1 的 epoch 6 重新 warm-start，不继承 CDF v2 后续已经出现的校准偏移。

训练阶段关闭：

```text
Candidate CE
CDF loss
Brier loss
```

仅保留：

```text
Pairwise ranking
Query surface consistency
Candidate partition consistency
```

Pair gap 使用 curriculum：

```text
epoch 1～2: 0.40
epoch 3～4: 0.30
epoch 5～6: 0.20
epoch 7～8: 0.15
```

固定：

```text
batch size: 128
canonical tasks per batch: 6
pairwise temperature: 0.15
max pairs per task: 8192
trainable parameters: 284,192
```

Run：

```text
tmp/weak-to-strong-diffusion-v1/
market_all_close2close_balanced_v2/runs/
market-jev-pairwise-r1-8ep-20260921
```

使用四个 H7 核心任务的 macro pairwise accuracy 选择 checkpoint：

```text
market.average_up_ratio.h7
market.equal_weight_return.h7
market.market_opportunity.h7
market.median_stock_return.h7
```

最佳 checkpoint 为 epoch 6：

```text
H7 core macro pairwise accuracy: 62.06%
all-task macro pairwise accuracy: 61.72%
H7 opportunity:                63.60%
H7 equal-weight return:        63.21%
H7 median-stock return:        62.78%
H7 breadth:                    58.64%
```

相对于 CDF/Pairwise 混合训练的 H7 核心排序，纯 Pairwise curriculum 更稳定，但提升仍主要存在于 2018 validation。

### 81.2 策略无关 Gate

推理对每个 H7 canonical task 同时计算 3/4/5 Candidate partition 的统一分位 score，再取平均：

```text
score_q =
mean(
  score_coarse3,
  score_medium4,
  score_fine5
)
```

导出四个核心 score：

```text
H7 breadth
H7 equal-weight return
H7 market opportunity
H7 median-stock return
```

以及：

```text
H7 consensus = 四个 score 的平均
```

只使用 2019 calibration 的模型 score 经验分布确定门槛，没有使用 Reward 策略收益。

门控规则：

```text
conservative:
  bottom 10% → 0%
  10%～20%   → 50%
  other      → 100%

gentle:
  bottom 10% → 25%
  10%～25%   → 50%
  other      → 100%
```

工件：

```text
market-jev-pairwise-r1-8ep-20260921/market_pairwise_gate_r2.parquet
```

### 81.3 Position Manager 公平性修复

第一轮回测发现，只要启用旧 `market_exposure` 插件，即使 exposure 始终为 100%，也无法复现原始 Reward baseline。

根因：

1. 启用 position manager 后，框架跳过原 `EqualWeightRebalance`。
2. 插件自行重建 entry weights。
3. 插件把 baseline 的 `cash_use_ratio=0.98` 改写成 exposure。
4. exposure=100% 时仍会把自然波动后的持仓强制修剪到 98%，产生大量额外卖单。

修复后：

```text
CompositeStrategy 先计算 baseline allocation
MarketExposureManager 只对 baseline allocation 做门控
exposure=100% 时完全旁路主动减仓
保留 baseline cash_use_ratio=0.98
```

100% exposure parity control：

| 指标 | 当前 Baseline | Full-exposure Plugin |
|---|---:|---:|
| Total Return | +202.3349% | +202.3349% |
| Sharpe | 0.640739 | 0.640739 |
| MDD | -38.0305% | -38.0305% |
| Trades | 4,459 | 4,459 |

每日 NAV 最大绝对差为 0，公平性门通过。修复前的所有 MarketExposure 回测结果均不能用于归因。

### 81.4 公平回测结果

固定 baseline：

```text
20260921_173337_reward_multitask_baseline_top10_h7_2020_2026_all_a
```

结果：

| 方案 | Total Return | Annual Return | Sharpe | MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 4,459 |
| Opportunity conservative | +211.41% | +19.36% | 0.658 | -38.03% | 4,556 |
| Opportunity gentle | +183.11% | +17.60% | 0.601 | -38.03% | 4,502 |
| Consensus conservative | +206.51% | +19.06% | 0.649 | -38.03% | 4,398 |
| Consensus gentle | +220.58% | +19.90% | 0.682 | -39.00% | 4,495 |

最好总收益来自 `consensus gentle`：

```text
相对 baseline 总收益增加 18.25 个百分点
Sharpe 从 0.641 提升到 0.682
MDD 反而恶化约 0.97 个百分点
月度盈利数均为 51 / 78，没有提高月度胜率
```

`opportunity conservative` 的结果更保守：

```text
总收益增加 9.07 个百分点
Sharpe 提升到 0.658
MDD 基本不变
月度盈利数从 51 增加到 52
```

### 81.5 关键诊断

门控在 2020～2026 的触发非常稀疏：

```text
Opportunity conservative: 19 天，全部位于 2025 年
Opportunity gentle:       20 天，全部位于 2025 年
Consensus conservative:   30 天，全部位于 2025 年
Consensus gentle:         34 天，其中 33 天位于 2025 年
```

原因是模型 score 相对于 2019 calibration 整体上移。2019 固定分位阈值在后续年份没有保持稳定。

更重要的是，真实跨期排序能力明显衰减：

| Query | 2018 Validation | 2019 Calibration | 2020～2026 Prediction |
|---|---:|---:|---:|
| H7 market opportunity | 63.60% | 42.95% | 53.09% |
| H7 equal-weight return | 63.21% | 46.59% | 52.13% |
| H7 median-stock return | 62.78% | 46.29% | 52.31% |
| H7 breadth | 58.64% | 47.29% | 50.85% |

因此：

1. Pairwise curriculum 确实提高了 2018 validation 排序。
2. 这种排序没有稳定迁移到 2019 和 2020～2026。
3. `+220.58%` 是少量门控日期产生的路径收益，不能证明模型已经学会稳定识别弱行情。
4. 当前瓶颈从“损失函数不会排序”进一步收敛为“市场状态跨年份分布漂移与有效样本不足”。

## 82. Pairwise 相对弱势：因果滚动 252 日 Gate

### 82.1 动机

固定 2019 score 阈值只在 2025 年触发，原因不是其他年份没有弱行情，而是 Pairwise score 的绝对尺度发生跨年漂移。

Pairwise 更可靠的语义是：

```text
当前状态相对近期历史更好还是更差
```

而不是：

```text
任意年份的 score=0.47 都代表同样的绝对市场状态
```

因此改成严格因果滚动分位：

```text
rolling_rank(T) =
rank(
  score(T),
  scores[T-252:T]
)
```

约束：

```text
只使用当前日期之前的 252 个交易日
不包含当前日期
最少历史长度 120
不使用未来市场 label
不使用 Reward 策略收益
```

### 82.2 两种门控

连续 Consensus：

```text
consensus_score =
mean(
  H7 breadth,
  H7 equal-weight return,
  H7 market opportunity,
  H7 median-stock return
)

rolling rank <= 10% → 25% 仓位
rolling rank <= 25% → 50% 仓位
其他                → 100% 仓位
```

离散 3/4 Vote：

```text
至少 3 个 Query rolling rank <= 10% → 25% 仓位
至少 3 个 Query rolling rank <= 25% → 50% 仓位
其他                                 → 100% 仓位
```

### 82.3 信号覆盖

在 2020～2026-06-02 回测日期中：

| Gate | 25% 仓位日 | 50% 仓位日 | 100% 仓位日 | 平均目标仓位 |
|---|---:|---:|---:|---:|
| Rolling consensus | 241 | 204 | 1,107 | 81.78% |
| Rolling 3/4 vote | 232 | 199 | 1,121 | 82.38% |

Rolling consensus 对真实 H7 弱势状态的离线诊断：

```text
weak recall:         29.13%
reduction precision: 15.06%
false reduction:     28.19%
```

它不再只集中于 2025 年，而是在 2020～2025 多个年份触发。2026 截至 6 月 2 日没有触发。

### 82.4 公平回测

所有策略均通过：

```text
100% exposure position-manager parity
```

与无插件 baseline 的每日 NAV 最大绝对差为 0。

结果：

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 | 4,459 |
| Rolling consensus gentle | **+277.25%** | **+22.98%** | **0.921** | **-24.56%** | 52 / 78 | 5,251 |
| Rolling 3/4 vote gentle | +163.89% | +16.32% | 0.642 | -25.14% | 53 / 78 | 5,045 |

Rolling consensus 相对 baseline：

```text
总收益增加 74.91 个百分点
年化收益增加 4.17 个百分点
Sharpe 增加 0.281
最大回撤改善 13.48 个百分点
月度盈利数增加 1
```

逐年收益：

| 年份 | Baseline | Rolling consensus |
|---:|---:|---:|
| 2020 | +45.25% | +50.20% |
| 2021 | +16.23% | +16.16% |
| 2022 | +7.66% | -0.75% |
| 2023 | +30.31% | +62.37% |
| 2024 | -0.12% | +48.01% |
| 2025 | +41.10% | +6.20% |
| 2026-06-02 | -9.42% | -14.65% |

### 82.5 结论

1. Pairwise score 必须按近期历史做相对解释，固定 2019 绝对阈值不适用。
2. 连续四维 consensus 明显优于 3/4 离散投票；硬投票丢失了 Query 强弱程度信息。
3. Rolling consensus 同时提高收益、Sharpe 并显著降低最大回撤，证明“相对弱势门控”在当前 Reward H7 策略上具有实际价值。
4. 改善并非每年一致：主要收益来自 2023 和 2024，2022、2025、2026 表现弱于 baseline。
5. 当前结果支持继续研究，但还不能证明跨策略或跨参数稳定，需要对 rolling window、阈值和不同 Reward baseline 做少量稳健性验证。

## 83. Pairwise R2：移除绝对成交额与成交量

### 83.1 假设

2026 输入相对于 2010～2017 train 出现明显漂移：

```text
amount_sum: +4.655 train std
volume_sum: +3.756 train std
```

这两个全市场求和字段还会受到上市公司数量和市场总规模增长影响，并不只表达行情强弱。

R2 仅固定屏蔽：

```text
amount_sum
volume_sum
```

保留：

```text
amount_zscore_20
volume_zscore_20
amount_top10_share
所有收益、宽度和横截面离散度特征
```

模型结构、warm-start、Pairwise curriculum、随机种子和训练切分均与 R1 一致。

Run：

```text
tmp/weak-to-strong-diffusion-v1/
market_all_close2close_balanced_v2/runs/
market-jev-pairwise-r2-no-absolute-liquidity-8ep-20260921
```

### 83.2 训练结果

R2 最佳 checkpoint 为 epoch 5：

```text
2018 H7 core macro pairwise accuracy:
R1: 62.06%
R2: 64.14%
```

但逐年排序并没有整体改善。

H7 market opportunity：

| 年份 | R1 | R2 |
|---:|---:|---:|
| 2020 | 56.2% | 45.4% |
| 2021 | 57.6% | 50.4% |
| 2022 | 57.1% | 47.8% |
| 2023 | 64.8% | 65.3% |
| 2024 | 62.3% | 55.2% |
| 2025 | 45.8% | 43.0% |
| 2026 | 34.8% | 69.4% |

截至 2026-06-02：

| Query | R1 | R2 |
|---|---:|---:|
| H7 market opportunity | 34.45% | 67.07% |
| H7 equal-weight return | 36.66% | 62.90% |
| H7 median-stock return | 34.60% | 53.20% |
| H7 breadth | 33.06% | 61.25% |

结论：

```text
绝对成交额/成交量确实是 2026 排序反转的重要污染源，
但它们在 2020～2024 也包含过有效信息。
直接删除只修复了 2026，没有提高整体跨年稳定性。
```

### 83.3 Rolling Gate

R2 使用与 R1 相同的：

```text
252 日严格因果 rolling rank
continuous H7 consensus
bottom 10% → 25% exposure
bottom 10%～25% → 50% exposure
```

R2 的 2020～2026 离线弱势诊断：

```text
weak recall:         21.30%
reduction precision: 13.76%
false reduction:     22.89%
```

虽然 2026 的同年排序恢复，但 2026 score 的绝对层级仍高于此前 252 日，因此截至 6 月 2 日仍没有触发减仓。

### 83.4 公平回测

Run：

```text
20260921_182049_reward_h7_pairwise_r2_rolling_consensus_gentle
```

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 |
| R1 rolling consensus | +277.25% | +22.98% | 0.921 | -24.56% | 52 / 78 |
| R2 no absolute liquidity | +168.68% | +16.65% | 0.635 | -39.18% | 52 / 78 |

R2 逐年收益：

```text
2020: +54.12%
2021: +17.35%
2022:  -0.44%
2023: +30.46%
2024:  -3.99%
2025: +36.24%
2026-06-02: -12.56%
```

### 83.5 结论

1. R2 是明确负结果，不能替代 R1。
2. 删除绝对成交额/成交量能修复 2026 排序方向，但损害了其他年份。
3. 当前正确方向不是简单删除流动性信息，而是把绝对总量替换为因果、规模无关的表达：
   - 每只活跃股票平均成交额/成交量；
   - 相对过去 252 日的滚动分位或 robust Z-score；
   - 对数变化率；
   - 保留短期 amount/volume Z-score。
4. 仅优化 2018 validation 会误选跨年泛化更差的模型；后续模型选择必须增加多年份 walk-forward 或 regime validation。

## 84. Pairwise R3：规模无关流动性与全局 PairPool

### 84.1 R3 状态特征

R3 不再直接使用：

```text
amount_sum
volume_sum
```

而是根据每日 universe mask 计算有效股票数：

```text
amount_per_active =
  amount_sum / active_stock_count

volume_per_active =
  volume_sum / active_stock_count
```

随后转换为：

```text
amount_per_active_rank_252
volume_per_active_rank_252
```

滚动分位严格只使用当前日期以前最多 252 个交易日，最少需要 60 日历史。输出位于 `[0,1]`，不会随上市公司数量和全市场总规模机械增长。

独立数据工件：

```text
artifacts/market_jev_r3_relative_liquidity/
```

它保持：

```text
3910 个日期
24 canonical tasks
72 Candidate partition views
相同 Outcome、Query、Candidate 与 split
相同 registry hash
```

### 84.2 R3-A 与 R3-B

R3-A：

```text
规模无关流动性特征
+ 原有随机日期 batch
```

R3-B：

```text
规模无关流动性特征
+ Global PairPool 日期采样
```

Global PairPool 每个 step：

1. 从完整 2010～2017 train 日期中按 Query 抽取满足当前 pair gap 的 better/worse 日期。
2. 合并多个 Query 的 pair 端点。
3. 对日期 ID 去重。
4. 每个日期只执行一次 Market Encoder。
5. 在统一 score 上计算全部 batch pair loss。

R3-A/R3-B 均保持：

```text
batch size: 128
15 steps / epoch
8 epochs
120 optimizer updates
相同 Pairwise curriculum
```

训练 Run：

```text
market-jev-pairwise-r3a-relative-liquidity-8ep-20260921
market-jev-pairwise-r3b-relative-liquidity-global-pairs-8ep-20260921
```

### 84.3 排序结果

2018 H7 core macro：

| 模型 | Accuracy |
|---|---:|
| R1 | 62.06% |
| R2 | 64.14% |
| R3-A | 65.82% |
| R3-B | 62.74% |

单看 2018，R3-A 最好；但逐年结果再次说明单年 validation 不可靠。

H7 market opportunity：

| 年份 | R3-A | R3-B |
|---:|---:|---:|
| 2020 | 44.6% | 49.4% |
| 2021 | 40.6% | 59.8% |
| 2022 | 49.8% | 60.6% |
| 2023 | 64.1% | 63.4% |
| 2024 | 56.9% | 54.3% |
| 2025 | 46.9% | 44.8% |
| 2026 | 42.2% | 66.2% |

Global PairPool 明显改善了 R3-A 在 2021、2022、2026 的跨年排序，证明增加跨日期比较确实有价值；但 2025 仍然反向，且 2019 calibration 仍低于 50%。

### 84.4 公平回测

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 |
| R1 rolling consensus | **+277.25%** | **+22.98%** | **0.921** | **-24.56%** | 52 / 78 |
| R2 no absolute liquidity | +168.68% | +16.65% | 0.635 | -39.18% | 52 / 78 |
| R3-A relative liquidity | +147.07% | +15.13% | 0.570 | -42.70% | 52 / 78 |
| R3-B global PairPool | +168.36% | +16.62% | 0.639 | -28.69% | 50 / 78 |

R3-B 相比 R3-A：

```text
总收益增加 21.29 个百分点
Sharpe 从 0.570 提升到 0.639
MDD 从 -42.70% 改善到 -28.69%
```

说明全局 PairPool 对风险排序和跨期稳定性有局部价值，但不足以抵消新流动性特征带来的收益损失。

### 84.5 结论

1. 组合数量扩充是有效机制：R3-B 明显优于相同特征的 R3-A。
2. Pair 数量并不能替代缺失的市场 regime；2010～2017 的更多组合仍然无法覆盖 2025 的新关系。
3. `amount_sum/volume_sum` 直接删除失败，替换为 per-active rolling rank 也没有超过原 R1。
4. 原始流动性总量包含真实有效信息，只是其语义随年份变化；下一步更可能需要模型同时接收：
   - 短期相对流动性；
   - 长期规模变化；
   - 当前活跃股票数量；
   - 明确的 regime 或时间归一化。
5. 当前最佳仍是 R1 rolling consensus，R3-A/R3-B 均不应替代它。

## 85. Pairwise R4：显式 Pair Dataset 与高训练量

### 85.1 与此前 Pairwise 的区别

R1/R2/R3 的训练单位仍然是日期 batch：

```text
128 个日期前向一次
→ 在 batch 内展开大量 Pair loss
→ 每个 epoch 只有 15 次 optimizer update
```

R4 将训练单位改成显式 Pair：

```text
一条训练样本 =
Query
+ better date
+ worse date
+ Candidate partition
```

同一日期允许：

```text
与多个不同 opponent 比较
在不同 step 中重复前向
在不同 Query 下重复比较
```

每个 step：

```text
6 个 canonical Query
每个 Query 64 个显式 Pair
= 384 Pair examples
= 768 次带重复的状态前向
```

训练规模：

```text
250 steps / epoch
8 epochs
2000 optimizer steps
768,000 explicit Pair examples
```

对照 R1：

```text
120 optimizer steps
```

因此 R4 的参数更新次数提高约 16.7 倍。

实现：

```text
tmp/jev-like-financial-decision-v1/train_market_jev_pair_dataset_v1.py
```

Run：

```text
market-jev-explicit-pairs-r4-2000step-20260921
```

### 85.2 训练曲线

大量显式 Pair 确实提高了训练排序：

```text
训练 Pair accuracy:
epoch 1: 66.31%
epoch 2: 69.66%
后续约 63%～66%
```

但 2018 validation 很快恶化：

| Epoch | Optimizer Steps | H7 Core Validation |
|---:|---:|---:|
| 1 | 250 | 56.19% |
| 2 | 500 | 53.38% |
| 3 | 750 | 58.43% |
| 4 | 1000 | 53.59% |
| 6 | 1500 | 40.11% |
| 8 | 2000 | 43.40% |

最佳 checkpoint 为 epoch 3，即：

```text
750 optimizer steps
288,000 explicit Pair examples
```

继续增加到 2000 step 后发生明显过拟合。

### 85.3 跨年排序

R4 最佳 checkpoint 的 H7 Pairwise Accuracy：

| Query | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Market opportunity | 53.5% | 50.9% | 63.1% | 50.4% | 60.6% | 60.1% | 59.0% |
| Equal-weight return | 51.8% | 48.4% | 58.8% | 52.8% | 61.8% | 57.8% | 56.3% |
| Median-stock return | 49.0% | 47.4% | 60.3% | 52.9% | 66.5% | 58.4% | 44.9% |
| Breadth | 51.6% | 46.6% | 56.9% | 48.7% | 54.8% | 56.8% | 57.7% |

与 R1 相比，R4 明显修复：

```text
2025 market opportunity: 45.8% → 60.1%
2026 market opportunity: 34.8% → 59.0%
```

这证明显式重复 Pair 和更高 optimizer step 确实增加了可迁移排序监督。

### 85.4 公平回测

Run：

```text
20260921_191333_reward_h7_pairwise_r4_explicit_pairs_rolling_consensus
```

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 |
| R1 rolling consensus | +277.25% | +22.98% | **0.921** | **-24.56%** | 52 / 78 |
| R4 explicit Pair | **+283.07%** | **+23.27%** | 0.919 | -38.84% | 50 / 78 |

R4 逐年收益：

```text
2020: +28.68%
2021: +26.01%
2022: +20.08%
2023: +27.76%
2024: +14.13%
2025: +42.93%
2026-06-02: -5.60%
```

R4 比 R1 的年度收益更均衡，并明显改善 R1 的 2025、2026 表现；但没有保住 R1 的最大回撤优势。

### 85.5 结论

1. 用户提出的判断成立：同一日期与不同 opponent 反复前向训练，能够提供有效的额外 Pairwise 优化信号。
2. Pair 数量增加不能无限扩展；约 750 step 后开始过拟合，2000 step 的最后模型明显崩溃。
3. R4 提升了跨年排序与总收益，但回撤退回 baseline 水平。
4. 当前出现了明确取舍：
   - R1 更偏风险控制；
   - R4 更偏跨年收益与排序稳定性。
5. 下一步更合理的是融合 R1 与 R4 的 market score 或增加早停后的轻量风险校准，而不是继续无上限增加 Pair 训练量。

## 86. R1/R4 50:50 融合与月度亏损归因

### 86.1 融合方式

不直接平均两个模型的原始 logits，而是分别使用严格因果的 252 日 rolling rank：

```text
fused_rank =
  0.5 × R1_consensus_rolling_rank
  + 0.5 × R4_consensus_rolling_rank
```

仓位：

```text
fused_rank <= 10% → 25%
fused_rank <= 25% → 50%
其他              → 100%
```

工件：

```text
market-jev-r1-r4-average-fusion-20260921/
market_pairwise_fusion.parquet
```

预测期信号诊断：

```text
平均目标仓位:       88.83%
减仓覆盖率:         19.41%
真实弱势召回率:     25.22%
减仓精度:           19.02%
非弱势误减仓率:     18.42%
```

### 86.2 公平回测

Run：

```text
20260921_192639_reward_h7_pairwise_r1_r4_average_fusion
```

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 |
| R1 rolling consensus | +277.25% | +22.98% | **0.921** | -24.56% | 52 / 78 |
| R4 explicit Pair | **+283.07%** | **+23.27%** | 0.919 | -38.84% | 50 / 78 |
| R1/R4 average fusion | +267.21% | +22.46% | 0.861 | **-24.22%** | **55 / 78** |

融合没有超过 R1/R4 的总收益，但：

```text
保留 R1 的回撤优势
月度盈利数提高到 55 / 78
月胜率从 65.38% 提高到 70.51%
```

它将 7 个 baseline 亏损月转为盈利，同时把 3 个 baseline 盈利月转为亏损。

### 86.3 Baseline 月度亏损归因

使用全 A 横截面每日等权平均收益构造策略无关的市场月收益。

Baseline 共 27 个亏损月：

```text
市场同时下跌:       25 个月
市场非负但策略亏损:  2 个月
```

因此 baseline 月度亏损的主要关联是整体市场环境，而不是单纯选股轮动失效。

市场下跌月共 31 个：

| 指标 | Baseline | Fusion |
|---|---:|---:|
| 盈利月数 | 6 / 31 | 12 / 31 |
| 月胜率 | 19.35% | 38.71% |
| 平均月收益 | -4.31% | -2.78% |

市场非负月共 47 个：

| 指标 | Baseline | Fusion |
|---|---:|---:|
| 盈利月数 | 45 / 47 | 43 / 47 |
| 月胜率 | 95.74% | 91.49% |
| 平均月收益 | +5.63% | +4.97% |

月收益与全 A 市场的相关性：

```text
Baseline: 0.875
Fusion:   0.782
```

说明融合仓位控制降低了策略对整体市场的暴露，但也会在部分正常行情中误减仓。

### 86.4 两类亏损月

市场弱势导致的亏损月：

```text
25 / 27
Baseline 平均: -6.03%
Fusion 平均:   -3.71%
Fusion 改善:   15 / 25
转为盈利:       7 / 25
```

市场非负但选股亏损的月份：

```text
2020-10
2023-05
```

这两个月 Fusion 都没有改善，平均仓位仍约 98.75%。这符合市场门控的职责边界：

```text
Market JEV 可以处理市场 Beta 风险，
不能修复 Reward model 的行业轮动或个股选择错误。
```

### 86.5 结论

1. 当前月度亏损的第一主因是整体弱行情：27 个亏损月中 25 个市场也为负。
2. R1/R4 并非完全没有学到弱势信息；融合后市场下跌月胜率从 19.35% 提高到 38.71%。
3. 但弱势识别仍不充分，市场下跌月仍有 19 个亏损。
4. Reward 选股模型的独立月度失效只明确出现在少数市场非负月份，当前不是主要矛盾。
5. 后续应分开处理：
   - Market JEV 继续优化弱行情召回与误减仓；
   - Reward model 另行分析行业轮动、风格偏好与选股失效月份。

## 87. R1/R4 最优综合方案可视化

当前以综合风险收益表现定义的最佳方案为：

```text
R1/R4 50:50 rolling-rank fusion
```

对照：

| 策略 | 累计收益 | 年化收益 | Sharpe | 最大回撤 | 盈利月份 |
|---|---:|---:|---:|---:|---:|
| Reward H7 Baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 |
| R1/R4 50:50 Fusion | +267.21% | +22.46% | 0.861 | -24.22% | 55 / 78 |

2026 年数据截至 2026-06-02，属于 YTD；月度热力图中的 2026 年 6 月仅包含 6 月 1～2 日。

![月度收益热力图](assets/jev_market_fusion_visuals_v1/monthly_return_heatmap.png)

![年度收益与回撤图](assets/jev_market_fusion_visuals_v1/yearly_return_and_drawdown.png)

详细 CSV 与 manifest：

```text
docs/experiments/assets/jev_market_fusion_visuals_v1/
```

## 88. 三月与十二月固定空仓实验

### 88.1 执行合同

测试两组：

```text
Reward H7 baseline + 三月/十二月空仓
R1/R4 Fusion + 三月/十二月空仓
```

为了保证目标月份首日已经处于空仓状态：

```text
目标月份前一个交易日收盘清仓
目标月份全月 exposure = 0
下一非目标月首个交易日收盘重新建仓
```

2020-01-02 至 2026-06-02 共包含：

```text
7 个三月
6 个十二月
13 个固定空仓月
```

这 13 个月的实际策略月收益均精确为 0。

### 88.2 回测结果

| 方案 | Total Return | Annual Return | Sharpe | MDD | 正收益月 | 不亏月份 |
|---|---:|---:|---:|---:|---:|---:|
| Reward H7 baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 | 51 / 78 |
| Baseline + Mar/Dec Flat | **+280.91%** | **+23.16%** | 0.846 | -45.20% | 42 / 78 | 55 / 78 |
| R1/R4 Fusion | +267.21% | +22.46% | 0.861 | **-24.22%** | **55 / 78** | 55 / 78 |
| Fusion + Mar/Dec Flat | +256.23% | +21.88% | **0.868** | -38.89% | 42 / 78 | 55 / 78 |

胜率口径：

```text
正收益月胜率:
42 / 78 = 53.85%

不亏月份比例（空仓持平计入）:
55 / 78 = 70.51%

只计算实际交易月份:
42 / 65 = 64.62%
```

### 88.3 解释

原策略的 13 个三月/十二月复合收益：

```text
Baseline: -20.86%
Fusion:   -14.89%
```

因此从静态月份贡献看，避开这些月份有一定依据。但完整账户回测具有路径依赖：

```text
清仓
→ 下一月份重新选择当时 Top10
→ 后续持仓、资金和卖出日期全部变化
```

所以不能把结果理解为简单删除原净值中的三月和十二月。

实际结果：

1. Baseline 季节性版本总收益提高，但最大回撤恶化到 `-45.20%`，最大回撤发生在 `2024-07-24`。
2. Fusion 季节性版本总收益下降，最大回撤从 `-24.22%` 恶化到 `-38.89%`。
3. 正收益月数量没有提高；`70.51%` 只是把 13 个目标月份机械变成了持平月。
4. 固定月份空仓不是比动态 Fusion 更稳健的方案。

## 89. Reward H7 Top5 与 R1/R4 Fusion

### 89.1 公平口径

新增严格 `portfolio_target` Top5 对照：

```text
相同 Reward score
topk = 5
max_positions = 5
cash_use_ratio = 0.98
H7 固定退出
close 成交
相同成本与交易限制
```

同时运行 Top5 100% exposure position-manager parity。结果与无插件 Top5 Baseline：

```text
每日 NAV 最大绝对差 = 0
每日收益最大绝对差 = 0
交易数完全一致
```

因此 Top5 + Fusion 的差异可以归因于仓位门控。

### 89.2 回测结果

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Top10 Baseline | +202.33% | +18.81% | 0.641 | -38.03% | 51 / 78 | 4,459 |
| Top10 R1/R4 Fusion | +267.21% | +22.46% | 0.861 | -24.22% | 55 / 78 | 4,849 |
| Top5 Baseline | **+250.23%** | **+21.56%** | **0.689** | -37.82% | 51 / 78 | 2,278 |
| Top5 R1/R4 Fusion | +164.86% | +16.39% | 0.603 | **-30.17%** | 51 / 78 | 2,525 |

Top5 Baseline 明显强于 Top10 Baseline：

```text
总收益增加 47.90 个百分点
Sharpe 从 0.641 提升到 0.689
MDD 基本相同
```

但 Top5 加入现有 Fusion 后：

```text
总收益下降 85.37 个百分点
Sharpe 从 0.689 降到 0.603
MDD 改善 7.65 个百分点
月度盈利数没有增加
```

### 89.3 市场环境分层

市场下跌月：

| 策略 | 盈利月数 | 平均月收益 |
|---|---:|---:|
| Top5 Baseline | 9 / 31 | -4.00% |
| Top5 Fusion | 12 / 31 | -3.29% |

市场非负月：

| 策略 | 盈利月数 | 平均月收益 |
|---|---:|---:|
| Top5 Baseline | 42 / 47 | +5.83% |
| Top5 Fusion | 39 / 47 | +4.64% |

Fusion 对 Top5 的作用是：

```text
略微改善弱市
明显削弱正常和强势市场的收益
```

这和 Top10 不同。Top5 已经是高度集中的高置信度组合，降低仓位的机会成本更高；同一套由全市场状态学习出的阈值不能直接假设对 Top5 和 Top10 都最优。

### 89.4 结论

1. Reward Top5 的确比 Top10 更有选股质量，公平口径总收益为 `+250.23%`。
2. 当前 R1/R4 Fusion 最适合 Top10，不适合直接迁移到 Top5。
3. Top5 Fusion 虽把 MDD 改善到 `-30.17%`，但总收益降到 `+164.86%`，风险收益比没有提升。
4. Top5 若做仓位控制，应使用更温和的仓位映射或单独校准阈值，不能复用 Top10 的 `25% / 50% / 100%` 配置。

## 90. Reward H7 TopK 集中度：Top10 / Top5 / Top3 / Top2

固定：

```text
同一 Reward score
portfolio_target
cash_use_ratio = 0.98
H7 固定退出
相同成交与成本规则
不使用市场仓位控制
```

结果：

| TopK | Total Return | Annual Return | Sharpe | MDD | Calmar | 月度盈利数 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | +202.33% | +18.81% | 0.641 | -38.03% | 0.495 | 51 / 78 |
| 5 | **+250.23%** | **+21.56%** | **0.689** | -37.82% | 0.570 | 51 / 78 |
| 3 | +235.97% | +20.78% | 0.633 | **-32.14%** | **0.647** | 51 / 78 |
| 2 | +151.91% | +15.48% | 0.440 | -34.74% | 0.446 | 44 / 78 |

逐年收益：

| TopK | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | +45.25% | +16.23% | +7.66% | +30.31% | -0.12% | +41.10% | -9.42% |
| 5 | +58.30% | +23.90% | +3.66% | +47.71% | +12.41% | +22.43% | -15.26% |
| 3 | +61.37% | +19.40% | -10.02% | +42.25% | +22.96% | +23.57% | -10.34% |
| 2 | +19.73% | +18.06% | -17.67% | +20.96% | +49.60% | +36.74% | -12.52% |

结论：

1. 总收益和 Sharpe 的最佳点为 Top5。
2. Top3 总收益略低于 Top5，但 MDD 和 Calmar 最优，是更均衡的集中组合。
3. Top2 集中过度，日波动上升、2022 年明显亏损、月度盈利数降到 44，风险调整收益显著恶化。
4. Reward score 的有效信息主要集中在前 3～5 名，但 Top1～2 不足以分散个股和入场日风险。

## 91. E5 风险收益分数的 Top5 / Top1

E5 中两种风险收益分数：

```text
Net Utility =
  terminal return
  - 0.5 × max drawdown
  - round-trip cost

70/30 Blend =
  0.7 × terminal-return same-date rank
  + 0.3 × mean(drawdown-safety rank, downside-safety rank)
```

固定公平口径：

```text
2020-01-02 ～ 2026-06-02
portfolio_target
cash_use_ratio = 0.98
H8 固定退出
相同成交与成本规则
```

结果：

| Score | TopK | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|---:|
| Net Utility | 10 | +118.20% | +12.92% | 0.531 | -35.45% | 43 / 78 |
| Net Utility | 5 | **+156.83%** | **+15.83%** | **0.596** | -38.11% | 38 / 78 |
| Net Utility | 1 | +126.93% | +13.62% | 0.373 | -43.36% | 40 / 78 |
| 70/30 Blend | 10 | +46.37% | +6.11% | **0.302** | **-25.98%** | 44 / 78 |
| 70/30 Blend | 5 | +44.98% | +5.96% | 0.286 | -27.69% | 38 / 78 |
| 70/30 Blend | 1 | **+58.62%** | **+7.45%** | 0.260 | -46.94% | 39 / 78 |

Net Utility Top5 相比 Top10：

```text
总收益增加 38.63 个百分点
Sharpe 从 0.531 提升到 0.596
MDD 从 -35.45% 恶化到 -38.11%
月度盈利数从 43 降到 38
```

70/30 Blend 从 Top10 收缩到 Top5 没有改善收益或风险；30% 风险排序仍然过强。

Top1 结论：

```text
两种分数的单票集中都明显扩大回撤
Net Utility Top1 MDD: -43.36%
70/30 Blend Top1 MDD: -46.94%
```

E5 风险 Query 能降低 Top10 组合波动，但不足以支撑 98% 资金集中在单票。当前 E5 中更合理的集中版本是 Net Utility Top5，但它仍明显弱于旧 Reward Top5 的 `+250.23% / Sharpe 0.689`。

## 92. Reward H7 Top5 固定止盈止损

### 92.1 历史路径分析

基线：

```text
20260921_204357_reward_h7_top5_portfolio_target
```

共重建 1,101 个完整入场 episode 的逐日收盘路径。

| 指标 | 数值 |
|---|---:|
| 单笔平均收益 | +0.77% |
| 单笔中位数 | +0.11% |
| 胜率 | 51.59% |
| 收益 P10 | -7.09% |
| 收益 P05 | -9.42% |
| 最大浮盈中位数 | +2.65% |
| 最大浮亏中位数 | -2.46% |

赢家路径：

```text
最终赢家中曾跌破 -5%：3.70%
最终赢家中曾跌破 -8%：0.35%
```

赢家通常不会先经历深度亏损。

右尾路径：

```text
达到过 +6% 后最终转亏：2.44%
达到过 +10% 后最终转亏：3.48%
达到 +10% 后的中位回吐：约 0.63%
```

因此低位固定止盈会明显截断右尾收益。

### 92.2 正式参数矩阵

固定：

```text
Reward H7 Top5
portfolio_target
H7 为最长持仓期限
close 触发、close 成交
相同交易成本与交易限制
```

实验：

```text
SL -5%
SL -8%
TP +18% / SL -5%
TP +25% / SL -5%
```

规则按配置顺序首个命中即全卖：

```text
take profit
stop loss
H7 time stop
```

### 92.3 正式账户结果

| 方案 | Total Return | Annual Return | Sharpe | MDD | 月度盈利数 |
|---|---:|---:|---:|---:|---:|
| Top5 H7 Baseline | **+250.23%** | **+21.56%** | **0.689** | **-37.82%** | **51 / 78** |
| SL -5% | +100.20% | +11.42% | 0.362 | -43.47% | 47 / 78 |
| SL -8% | +105.02% | +11.83% | 0.377 | -49.57% | 47 / 78 |
| TP +18% / SL -5% | +102.16% | +11.59% | 0.383 | -50.35% | 48 / 78 |
| TP +25% / SL -5% | +139.74% | +14.59% | 0.474 | -47.96% | 48 / 78 |

四个版本全部低于基线，且没有改善最大回撤。

### 92.4 退出原因

Baseline：

| 原因 | 次数 | 平均收益 | 胜率 |
|---|---:|---:|---:|
| H7 time stop | 1,101 | +0.77% | 51.59% |
| 单票超22%权重减仓 | 71 | +21.51% | 97.18% |

Top5 基线已经通过 `max_position_weight=22%` 对上涨过快的赢家进行部分止盈。这一机制保留剩余仓位继续运行，而固定 TP 会把整只股票全部卖出。

SL -5%：

| 原因 | 次数 | 平均收益 | 胜率 |
|---|---:|---:|---:|
| Stop Loss | 345 | -7.13% | 0% |
| H7 time stop | 908 | +3.37% | 66.96% |
| 超权重减仓 | 90 | +19.87% | 100% |

止损阈值为 -5%，实际成交平均约 -7.13%，原因包括离散日收盘跳变、卖出滑点和交易成本。

### 92.5 为什么交易级代理误判

交易级静态代理假定：

```text
某笔交易提前止损
其他股票和未来入场保持原 Baseline 不变
```

真实账户则会：

```text
提前止损
→ 当天产生空位
→ 买入新的当日 Top5
→ 新股票改变下一次退出日期
→ 后续空位和补票日期持续变化
```

SL -5% 增加：

```text
345 次止损
152 次额外买入
```

只有约 324 个入场 episode 仍能与原基线按 `股票 + 入场日期` 匹配，后续大部分账户路径已经改变。

可匹配的 SL -5% 交易：

```text
止损成交平均: -7.20%
原 H7 最终平均: -6.74%
原 H7 最终转正: 6 / 97
```

多数止损票最终仍会亏损，但提前卖出没有改善到足以覆盖持续的账户路径重排。

### 92.6 结论

1. 当前 Top5 H7 不适合加入简单固定止盈止损。
2. 固定止盈与已有的22%单票权重上限重复，并破坏大赢家右尾。
3. -5%/-8%止损在交易级可以收窄左尾，但账户层的持续补票和持仓时序变化使收益与回撤同时恶化。
4. 若继续研究止损，必须同时约束止损后的再入场，例如：
   - 止损当日不补票；
   - 单票或账户设置冷静期；
   - 仅在市场弱势时启用止损；
   - 不做全额止损，先减半仓。
5. 当前正式最优仍为无固定止盈止损的 Reward H7 Top5。
