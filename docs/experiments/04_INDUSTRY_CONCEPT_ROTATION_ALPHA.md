# 行业/概念轮动 Alpha 第一轮

> 实验日期：2026-07-13
>
> 评估层级：`development_oos`
>
> 研究目标：验证行业、概念和市场风险状态特征能否改善一周持仓尺度的全市场 ML 横截面排序。
>
> 结论：第一轮全局 LightGBM 特征增强没有改善预测层；尾部资本效率实验只作为诊断；强行业/强概念硬候选池会削弱 5 日 ML TopK，不进入回测。

## 1. 背景

用户目标是五年近几十倍至 100 倍收益、平均股票持仓大于 5、五个自然年全部正收益、平均持仓约一周，并且通过 2026 前向。现有 10 日 LightGBM 长历史候选虽然改善了 2021-2025 稳定性，但五年累计只有 `+360.26%`，2026 ML 80% 袖套为 `-0.25%`，不满足目标。

规则策略和概念/行业实验给出三个线索：

1. `pos3/topK2` 弱转强 close 版本能达到几十倍级别，但持仓集中且对成交时点高度敏感。
2. 扩到 `pos8/pos10` 后平均持仓可以超过 5，但尾部候选质量下降，收益和胜率明显变差。
3. 市场宽度、行业强度和概念强度有解释力，但硬过滤会误伤大赢家；更合理的方向是作为横截面排序和弱市尾部质量判断。

因此本轮不是测试“概念强就买”，而是测试行业/概念/市场状态是否能提高 5 日标签下的全市场排序质量。

## 2. 最小代码能力

本轮为研究框架补充了可选静态 group 输入：

1. 新增 `quantx.core.data.meta.groups`，用于把行业 CSV 加载成 `[instrument]` group code，把概念 CSV 加载成 `[instrument, group]` 多成员矩阵。
2. `FeatureSpec` 支持 `groups` 配置。
3. `DatasetBuilder` 在 `FactorRuntime` 中注入 group，并允许一维市场状态特征广播成二维特征列。
4. group 配置和 CSV checksum 纳入 feature schema identity，避免换元数据来源后复用旧 artifact 身份。

验证情况：

```text
conda run -n test python -m compileall -q quantx/core/data/meta/groups.py quantx/core/research/dataset.py quantx/core/research/specs.py quantx/core/research/causality.py quantx/tools/run_research.py
```

通过。

`pytest` 在当前机器的 `_pytest/capture.py` 初始化阶段触发 `Segmentation fault: 11`，即使运行单个测试也在收集前崩溃；这不是业务断言失败。已用 `conda run -n test python -c` 和全市场 dry-run 验证新增路径可用。

## 3. 数据和配置

共同设置：

| 项目 | 值 |
| --- | --- |
| Provider | `/Users/mingxiaoli/Documents/QuantX-QMT/data/qlib_data_fixed` |
| 数据版本 | `qmt:4b1c6d77e335b98056f7cf74` |
| 股票池 | 当前 QMT 全主板 3,194 只 |
| 训练起点 | 2010-01-04 |
| OOS | 2021-2025 年度 walk-forward |
| 标签 | T+1 open 入场、未来 5 个交易日 open 退出的横截面超额收益 |
| 模型 | LightGBM seed 7 |
| 输出位置 | `/tmp/quantx-research/industry-concept-rotation-v1/` |

基础 5 日模型配置：

```text
/tmp/quantx-research/industry-concept-rotation-v1/configs/base_5d_seed7_dev.yaml
```

行业/概念/风险增强配置：

```text
/tmp/quantx-research/industry-concept-rotation-v1/configs/industry_concept_regime_5d_seed7_dev.yaml
```

增强特征包括：

1. `industry_ret5/10/20/60`、`industry_rank20/60`、`rel_industry_ret5/10/20`。
2. `concept_ret5/10/20/60`、`concept_rank20/60`、`rel_concept_ret5/10/20`。
3. `market_breadth20/60`、`market_ret20/60`。
4. `weak_market_industry_rank20`、`weak_market_concept_rank20`、`strong_group_stock_ret5`。

重要边界：行业和概念来自 2026-06-25 静态快照，不是 point-in-time 历史成分。因此即使结果变好，也只能作为探索性证据，不能直接晋级。

## 4. 预测层结果

dry-run 两组均通过，数据集为 `9,515,014` 行，OOS 预测为 `3,716,399` 条。

| 模型 | 日均 RankIC | 全期 RankIC | 五分位 Top-Bottom |
| --- | ---: | ---: | ---: |
| base 5d | `0.102586` | `0.101884` | `+0.010966` |
| industry/concept/regime 5d | `0.099279` | `0.099390` | `+0.010878` |

逐年日均 RankIC：

| 模型 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base 5d | `0.104285` | `0.118045` | `0.071529` | `0.102390` | `0.116970` |
| industry/concept/regime 5d | `0.103469` | `0.111793` | `0.065786` | `0.103831` | `0.111754` |

增强版只在 2024 略高，其余年份都低于基础版；全期日均 RankIC、全期 RankIC 和五分位 Top-Bottom 均未改善。

## 5. 判定

`rejected`。

本轮证据说明，把静态行业/概念强度和弱市交互直接塞进同一个全局 LightGBM，并不能改善一周尺度横截面排序。考虑到预测层已经小幅变差，本轮不继续运行三 seed、组合回测或 2026 前向；否则会继续消耗算力验证一个已经没有预测层优势的候选。

不能从本轮推出“行业/概念没有价值”。更准确的结论是：

1. 静态快照 group 特征直接加入全局模型，没有解决 TopK 尾部候选质量问题。
2. 行业/概念可能更适合用于候选池分层、两阶段排序、弱市尾部仓位启用条件，或作为残差模型，而不是普通全局特征。
3. 后续必须避免围绕 2021-2025 对行业/概念阈值做硬调参。

## 6. Top20 分层诊断

为避免漏掉“全横截面略差但 TopK 更好”的情况，本轮又对两份完整预测做了流式 Top20 标签诊断。方法是每天只保留预测分数最高的 20 只股票，再用同一 5 日 next-open 横截面超额标签计算 Top1-5、Rank6-10、Rank11-20 的平均理论收益。诊断脚本和输出均在 `/tmp/quantx-research/industry-concept-rotation-v1/`。

| 模型 | Top1-5 | Rank6-10 | Rank11-20 | Top1-20 |
| --- | ---: | ---: | ---: | ---: |
| base 5d | `+0.035581` | `+0.009015` | `+0.009019` | `+0.015659` |
| industry/concept/regime 5d | `+0.034229` | `+0.009677` | `+0.008187` | `+0.015070` |

基础模型的 Rank6-20 并不是负收益，平均 5 日横截面超额仍约 `+0.9%`；问题更可能在资金效率、交易成本、风险开关、执行路径和组合持仓管理，而不是“尾部候选完全没有 Alpha”。增强版 Rank6-10 略高，但 Top1-5 和 Rank11-20 变差，Top1-20 总体仍低于基础模型。

## 7. 下一方向

下一轮切换到 `tail_capital_efficiency_v1`：

```text
第一阶段：保留基础 5 日 ML 排序，因为它的 Top20 理论标签更强
第二阶段：比较 Top5、Top10、Top20 的纯 ML 组合路径、换手、成本和年度稳定性
第三阶段：若尾部理论收益无法转化为账户收益，再研究动态仓位或两阶段资本分配
```

理由：用户要求平均持仓大于 5，且本轮诊断显示基础 5 日模型的 Rank6-20 理论 Alpha 仍为正。下一步应先证明这些尾部预测能否在真实交易约束、调仓频率、成本和风险规则下转化为净值，而不是马上训练新模型。

下一轮最低验证要求：

1. 单独报告 Top5、Top10、Top20 的 ML-only 账户收益、逐年收益、平均股票数、换手和成本。
2. 先使用完整预测压缩出的 TopK 文件，避免反复加载 1.6GB 完整预测。
3. 若 Top20 ML-only 开发期仍不能接近目标收益，不进入 2026 前向。
4. 任何后续候选都必须在 2026 ML 袖套本身保持正贡献，否则不晋级。

## 8. tail_capital_efficiency_v1

### 8.1 假设

基础 5 日模型的 Top1-5 理论标签明显强于 Rank6-20，但用户目标要求平均股票持仓大于 5。若把 Top5 作为收益核心，同时保留 Top6-10 的小权重持仓，可能比等权 Top10 更接近收益目标，并仍满足持仓数量约束。

本轮不重新训练模型，只使用基础 5 日模型的冻结 TopK 预测文件：

| 文件 | SHA256 | 记录数 |
| --- | --- | ---: |
| `/tmp/quantx-research/industry-concept-rotation-v1/base_5d_top5_predictions.json` | `sha256:39c341b8ea633276236b6a7c096d178c283d94f982d448d54147aba84dc77bfa` | 6,060 |
| `/tmp/quantx-research/industry-concept-rotation-v1/base_5d_top10_predictions.json` | `sha256:c696d9183dc4dfe084206d0c8b244ae7c794b9a2b47f8801871e25caab385816` | 12,120 |
| `/tmp/quantx-research/industry-concept-rotation-v1/base_5d_top20_predictions.json` | `sha256:010a735d7ca4a90df15ede12b035df8893acf88b82ba2cc4df8cca0153b673c8` | 24,240 |

共同回测口径：

| 项目 | 值 |
| --- | --- |
| 区间 | 2021-01-04 至 2025-12-31 |
| 组合 | ML-only，`ml_allocation=1.0`，`wufu_allocation=0` |
| 实际股票总仓 | 临时 runner 内部为 ML 袖套的约 90% |
| 执行 | T 日信号，T+1 开盘，5 个交易日调仓 |
| 成本 | 标准成本，`deal_price=open` |
| 输出 | `/tmp/quantx-research/industry-concept-rotation-v1/backtests/` 和 `backtests-weighted/` |

### 8.2 等权 TopK 结果

| 方案 | 最终倍数 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Top5 等权 | 11.38x | `+1037.52%` | `-14.59%` | 1.66 | 2.65 | 10.04 | `+50.20%、+85.88%、+25.19%、+91.52%、+69.93%` |
| Top10 等权 | 6.28x | `+527.71%` | `-8.65%` | 2.05 | 5.16 | 9.37 | `+47.88%、+47.92%、+24.19%、+59.49%、+44.88%` |
| Top20 等权 | 3.76x | `+275.83%` | `-8.98%` | 2.04 | 10.17 | 9.08 | `+34.72%、+26.12%、+18.04%、+40.29%、+33.58%` |

Top5 的收益效率最高，但平均持仓只有 2.65，不满足目标。Top10 和 Top20 都五年全正且平均持仓大于 5，但收益被尾部仓位明显稀释，距离五年近几十倍至 100 倍的要求仍远。

### 8.3 Top10 内部加权

加权方案使用同一份 Top10 冻结预测。`head5=80%` 表示 Top1-5 平分 80% 股票仓位，Top6-10 平分 20% 股票仓位；股票总仓仍保持 runner 内部约 90% 的口径。

| 方案 | 最终倍数 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Top10 head5 70% | 8.05x | `+705.26%` | `-10.81%` | 1.87 | 5.16 | 9.32 | `+48.80%、+63.32%、+24.69%、+71.84%、+54.65%` |
| Top10 head5 80% | 9.08x | `+808.00%` | `-12.05%` | 1.79 | 5.16 | 9.34 | `+49.36%、+70.98%、+24.92%、+78.23%、+59.69%` |
| Top10 head5 90% | 10.18x | `+917.74%` | `-13.33%` | 1.72 | 5.16 | 9.46 | `+49.72%、+78.56%、+25.03%、+84.78%、+64.78%` |

加权方向有效：随着 Top5 权重提高，累计收益单调改善，并且五年全部为正、平均股票持仓仍大于 5。代价是最大回撤和 Sharpe 逐步向 Top5 集中版本靠近。

### 8.4 判定

`diagnostic_only`，不进入 2026 前向，不作为候选策略。

Top10 head5 90% 是当前同一 5 日模型下收益最高的持仓合规版本：它保留了 Top5 的大部分收益效率，又用 Top6-10 维持平均股票数大于 5。但这不是鲁棒 Alpha 的改进，而是通过降低 Rank6-10 资本占用来规避尾部质量问题。它仍只有约 10.18 倍最终倍数，低于用户要求的五年近几十倍至 100 倍收益，因此不能作为最终方向。

下一步不继续调 head 权重，也不把加权 Top10 前推到 2026。研究方向转向更符合 A 股结构的鲁棒 Alpha：市场风险、横截面因子和行业/概念候选池。优先验证强行业/强概念候选池能否让 Top10/Top20 等权持仓自然具备更高收益，而不是依赖权重技巧。

## 9. group_rotation_candidate_pool_v1

### 9.1 假设

A 股的高收益机会可能集中在强行业、强概念和主题扩散状态中。与其把行业/概念作为普通特征直接塞进全局 LightGBM，不如先用 T 日可见的行业/概念强度构造候选池，再在候选池内使用基础 5 日 ML 分数等权选 Top10/Top15/Top20。

本轮只做标签层诊断，不做回测。若强行业/强概念池不能提高理论 5 日横截面超额 label，则不进入组合层，避免在失败筛选规则上继续调参。

### 9.2 数据和边界

| 项目 | 值 |
| --- | --- |
| 基础预测 | `/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json` |
| 诊断脚本 | `/tmp/quantx-research/group-rotation-candidate-pool-v1/analyze_group_candidate_pool.py` |
| 诊断输出 | `/tmp/quantx-research/group-rotation-candidate-pool-v1/group_pool_label_diagnostic.json` |
| 输出 SHA256 | `sha256:599d68a4e99b64adcd0af7e65b8a3cd7b4a8b2073c5157ed0e2f779279b9abbe` |
| 行业元数据 | 东财一级行业，86 个行业，静态快照 2026-06-25 |
| 概念元数据 | 东财概念/板块，过滤非主题和过大/过小板块后保留 342 个概念 |
| 标签 | T+1 open 入场、未来 5 个交易日 open 退出的横截面超额收益 |

概念池过滤掉了明显非主题标签，例如融资融券、沪股通、创业板综、昨日涨停、昨日连板、ST、退市、机构重仓等。行业和概念仍然是 2026 静态快照，不是 point-in-time 历史成员，因此本轮即使结果变好也只能作为探索性证据。

候选池强度使用 T 日可见信息：组内 5/20/60 日收益、MA20 宽度、20 日成交量放大、组内强股比例和组内离散度。每天构造以下候选池：

1. 全市场基线。
2. 强行业 Top3 / Top5。
3. 强概念 Top10 / Top20。
4. 强行业 Top3 与强概念 Top10 的并集。
5. 强行业 Top3 与强概念 Top10 的交集。

### 9.3 标签诊断结果

以下数字为每天候选池内部按基础 5 日 ML 分数取 TopK 后的平均 5 日横截面超额 label。

| 候选池 | Top10 | Top15 | Top20 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 全市场 | `+0.022298` | `+0.017629` | `+0.015659` | 基线最强 |
| 强行业 Top3 | `+0.002799` | `+0.001370` | `+0.000356` | 大幅削弱，2022/2025 转负 |
| 强行业 Top5 | `+0.005108` | `+0.003612` | `+0.002641` | 大幅弱于全市场 |
| 强概念 Top10 | `+0.006019` | `+0.005147` | `+0.004485` | 弱于全市场 |
| 强概念 Top20 | `+0.008766` | `+0.007204` | `+0.006182` | 概念池内最好，但仍低于全市场约 60% |
| 强行业 Top3 或强概念 Top10 | `+0.007384` | `+0.006016` | `+0.005107` | 弱于全市场 |
| 强行业 Top3 且强概念 Top10 | `-0.003136` | `-0.004344` | `-0.004923` | 明确失败 |

Top20 分层也显示硬候选池削弱了最强股票：

| 候选池 | Top1-5 | Rank6-10 | Rank11-15 | Rank16-20 |
| --- | ---: | ---: | ---: | ---: |
| 全市场 | `+0.035581` | `+0.009015` | `+0.008290` | `+0.009749` |
| 强概念 Top20 | `+0.009681` | `+0.007851` | `+0.004081` | `+0.003116` |
| 强行业 Top5 | `+0.006084` | `+0.004132` | `+0.000621` | `-0.000275` |
| 强行业 Top3 且强概念 Top10 | `-0.000609` | `-0.004845` | `-0.007640` | `-0.008515` |

### 9.4 判定

`rejected`。

强行业/强概念硬候选池没有形成更强、更鲁棒的 Alpha。它更像是在日内把已有 5 日 ML 分数中最强的全市场机会过滤掉，尤其交集规则会明显转负。因此本轮不进入等权 Top10/Top15/Top20 回测，也不做 2026 前向。

不能从本轮推出“行业/概念没有价值”。更准确的结论是：当前这种先按静态行业/概念强度硬筛候选池，再在池内取 ML TopK 的方法失败。下一轮不应继续调强行业 TopN 或强概念 TopN 阈值，而应测试更细的方向：

1. **分组残差模型**：先保留全市场 TopK，再学习哪些高分股票的行业/概念状态会导致预测失效，而不是先删掉全市场机会。
2. **策略级 trust gate**：按交易日判断 5 日 ML ranker 是否可信，只决定是否交易，不改变 TopK 尾部权重。
3. **行业/概念内相对强弱**：不要买强行业里的所有高分股，而是寻找“行业内最强且相对行业仍超额”的股票。
4. **近期自适应训练协议**：2026 衰减可能来自年度冻结模型对近期结构利用不足，应优先检验季度/月度 fold 或时间衰减样本权重。

## 10. strategy_trust_gate_v1

### 10.1 假设

硬行业/概念候选池失败后，下一步转向策略级 trust gate：不提前删股票，而是判断当前 5 日 ML ranker 是否处在更可信的横截面环境。外部资料中关于 sector rotation 和 ranker uncertainty 的结论都指向同一点：非平稳市场里，部署决策应拆成两层：先判断策略是否值得运行，再判断买哪些股票。

本轮只测试一个简单、可解释的 gate：当全市场 20 日收益横截面离散度高于过去 252 个交易日自身中位数时，允许 ML TopK 交易；否则 ML 袖套转入 `SH511880`。这个 gate 的直觉是：横截面分化高时，排序模型更容易从强弱差异中赚钱。

### 10.2 标签层诊断

诊断脚本：

```text
/tmp/quantx-research/strategy-trust-gate-v1/analyze_strategy_trust_gate.py
```

输出：

```text
/tmp/quantx-research/strategy-trust-gate-v1/trust_gate_diagnostic.json
sha256:0a8962b3e3edbd874aaf781944e4fdad2cec097fe766027e1a882dcb39382c0e
```

标签层看起来有价值：固定覆盖约 50% 交易日的 `market_ret20_disp` gate 能把 Top10/15/20 的平均 5 日超额 label 全部抬高。

| TopK | 全部日期 label | gate 日期 label | gate 正 label 日期比例 |
| ---: | ---: | ---: | ---: |
| 10 | `+0.022298` | `+0.034302` | 72.14% |
| 15 | `+0.017629` | `+0.026778` | 76.78% |
| 20 | `+0.015657` | `+0.023125` | 77.11% |

逐年看，gate 对 2022、2024、2025 提升明显，但 2021 因为高分化日期占比过高，提升不稳定；2023 选择日期极少，提示这个 gate 可能更像“机会浓度环境识别”，不是完整交易策略。

### 10.3 组合层回测

临时 runner：

```text
/tmp/run_quantx_combined_frozen_trust_gate.py
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本。gate 使用过去 252 日 rolling median，不使用全样本中位数，避免阈值前视。

| 方案 | 最终倍数 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Top10 等权基线 | 6.28x | `+527.71%` | `-8.65%` | 2.05 | 5.16 | 9.37 | `+47.88%、+47.92%、+24.19%、+59.49%、+44.88%` |
| Top10 + trust gate | 2.81x | `+180.78%` | `-7.15%` | 2.13 | 3.25 | 10.31 | `+27.17%、+4.33%、+7.87%、+50.39%、+30.45%` |
| Top15 + trust gate | 2.38x | `+138.19%` | `-8.82%` | 1.88 | 4.62 | 9.62 | `+20.06%、+6.49%、+8.21%、+34.63%、+27.88%` |
| Top20 + trust gate | 2.19x | `+119.30%` | `-9.06%` | 1.74 | 6.00 | 9.32 | `+20.03%、+6.94%、+7.83%、+30.91%、+21.05%` |

### 10.4 反事实分析

如果只看标签层，会误以为 trust gate 是好方向；但组合层证明它把高收益日期和普通正收益日期一起删掉了。更关键的是，Top20 gate 虽然满足平均持仓大于 5、五年全正和约一周持仓，但收益只有 2.19x，低于原始 Top20 的 3.76x，也远低于目标。

可能原因：

1. **标签收益不是资金路径收益**：5 日平均超额 label 提升，不等于每 5 日换仓的账户路径提升。
2. **防守资产稀释收益**：gate 关闭时转 `SH511880`，降低回撤的同时删除大量正收益暴露。
3. **机会不是日频二元开关**：A 股短期趋势更像连续的强弱路径，简单开/关 gate 太粗。
4. **横截面分化是必要非充分条件**：高分化有助于排序，但仍需要挑对处在持续上升路径里的个股。

### 10.5 判定

`rejected`。

策略级 trust gate 不是当前要找的“鲁棒且容易出收益”的方向。下一步应回到选股层，但不再只看单日横截面：要研究横截面强弱在过去一段时间里的持续性、加速度和路径质量。新的方向是 `temporal_cross_sectional_momentum_v1`：在已有 5 日 ML TopK/Top100 候选中，用 RPS 路径、相对行业路径、强势持续天数和量价扩散重新排序，验证是否能自然提高等权 Top10/15/20 收益。

## 11. temporal_cross_sectional_momentum_v1

### 11.1 假设

市场不是静态横截面，A 股短期强弱通常沿行业、概念和个股趋势路径扩散。若只看 T 日 ML 分数，可能无法区分“持续强势中的高分股”和“一日噪声中的高分股”。本轮因此不改训练模型，先在基础 5 日 ML 高分候选内部，用过去 5/20/60/120 日 RPS、RPS 持续性、加速度、相对行业强弱、量能确认和均线位置做再排序，验证能否自然提高等权 Top10/15/20 的 5 日标签。

### 11.2 数据和方法

诊断脚本：

```text
/tmp/quantx-research/temporal-cross-sectional-momentum-v1/analyze_temporal_cs_momentum.py
```

输出：

```text
/tmp/quantx-research/temporal-cross-sectional-momentum-v1/temporal_cs_momentum_label_diagnostic_v2.json
```

基础预测仍使用 2021-2025 OOS 的 5 日基础模型完整预测。每天先取 ML Top50/100/200 候选，再测试以下固定 selector：

1. `rps_persistence`：20/60 日 RPS 均值、最小值和斜率。
2. `rps_acceleration`：5/20/60 日 RPS 加速度和放量确认。
3. `ml_plus_persistence`、`ml_plus_acceleration`：ML 分数和路径分数混合。
4. `ml_weak_persistence`、`ml_weak_acceleration`：85% ML + 15% 路径的弱校正。
5. `ml_quality_guard`、`ml_anti_chase`：只惩罚明显弱路径或过热追高。
6. `relative_industry_strength`、`volume_confirmed_path`、`medium_trend_pullback`：相对行业、量价确认和中期趋势回撤。

### 11.3 标签诊断结果

最好的结果来自非常弱的 `ml_weak_acceleration`，但提升幅度很小：

| 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始 ML TopK | `+0.022298` | `+0.017629` | `+0.015657` | 基线 |
| pool100 `ml_weak_acceleration` | `+0.022404` | `+0.017848` | 未显著优于基线 | 微弱改善 |
| pool50 `ml_plus_acceleration` | `+0.021553` | `+0.018109` | `+0.015440` | Top15 略高，Top10/20 变差 |
| pool50 `relative_industry_strength` | `+0.013378` | `+0.013096` | `+0.012513` | 明显削弱头部收益 |
| pool50 `rps_persistence` | 未进入前列 | 未进入前列 | `+0.010352` | 变差 |

桶拆分显示路径分数经常把收益从 Top1-5 摊平到尾部，但没有提高总收益。例如 pool50 `ml_plus_acceleration` 的 Top20 桶为：Top1-5 `+0.029971`、Rank6-10 `+0.013135`、Rank11-15 `+0.011221`、Rank16-20 `+0.007431`；原始 ML 则是 Top1-5 `+0.035581`、Rank6-10 `+0.009015`、Rank11-20 约 `+0.009019`。尾部更均匀，但头部被削弱，总体没有形成数量级改善。

### 11.4 判定

`rejected`。

时序横截面路径不是完全无效，但在“已有 ML 高分候选内部再排序”的实现下，只有极弱的边际提升，无法解释五年收益从 6x 提升到几十倍。更重要的是，它没有自然改善等权 Top10/15/20 的整体收益效率，只是在头部收益和尾部均匀性之间做交换。这和用户明确拒绝的“通过降低尾部权重满足持仓”属于同一类弱修补，不应继续调 selector 权重。

可能原因：

1. 基础 5 日 ML 已经吸收了大量动量/位置/量能信息，简单 RPS 再排序只是重复特征。
2. A 股强势路径有用，但强势股真正爆发常集中在更小头部；等权扩到 Top15/20 后仍会被普通正收益样本稀释。
3. 静态行业快照无法表达真实历史概念扩散，`relative_industry_strength` 可能把非 PIT 信息和错误分组一起引入。
4. 再排序不是重训，无法让模型学习“哪些路径特征在不同市场环境下有效”。

下一步转向训练协议，而不是继续调路径分数：验证年度冻结模型是否因为把最近一年留作 validation 却没有使用 early stopping，导致 2026 近期结构没有进入训练。

## 12. adaptive_training_protocol_v1

### 12.1 假设

现有 `AnchoredWalkForwardSplitter` 会把预测年前 252 个交易日留作 validation；但当前 `LightGBMTrainer` 没有 early stopping，validation 实际只传入 `eval_set`，没有改变训练树数或样本权重。因此 2026 forward 的年度模型基本没有学习 2025 年标签。若市场结构动态变化明显，缩短 validation 窗口、让更多 2025 标签进入训练，应改善 2026 预测和 ML-only 账户路径。

本轮不修改正式 splitter 和 trainer，只在 `/tmp` 复制 5 日配置，比较 `validation_sessions=252/63/20` 三个版本。

### 12.2 dry-run 信息边界

配置位置：

```text
/tmp/quantx-research/adaptive-training-protocol-v1/configs/
```

dry-run 证实训练信息边界按预期变化：

| 方案 | 训练行数 | validation 行数 | 预测行数 | training_information_cutoff |
| --- | ---: | ---: | ---: | --- |
| val252 | 8,703,686 | 779,754 | 395,051 | `2024-12-11` |
| val63 | 9,299,392 | 183,915 | 395,051 | `2025-09-18` |
| val20 | 9,435,647 | 47,617 | 395,051 | `2025-11-26` |

### 12.3 2026 预测层结果

| 方案 | IC | RankIC | 日均 IC | 日均 RankIC | 五分位 Top-Bottom | 预测 checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| val252 | `0.019753` | `0.050595` | `0.022462` | `0.048050` | `+0.000761` | `sha256:69597a2e0e9c5122ba70701c968b1ae4dd65ab001592872a043043a6172a8501` |
| val63 | `0.020319` | `0.052895` | `0.022719` | `0.050476` | `+0.001456` | `sha256:c7048977fbcffc274dd7ef8bfa9dc3c8744a75fb7ad8148aa22f2f73b43c3647` |
| val20 | `0.018002` | `0.050578` | `0.020512` | `0.047392` | `+0.000866` | `sha256:348be77b82475065d5d29ffce19568baed35aeb359d5c9da9643c4ee94245eb4` |

`val63` 是预测层最好版本，但改善幅度不大。`val20` 说明越近越好并不成立，过度压缩 validation 可能引入近期噪声。

### 12.4 2026 TopK 标签诊断

诊断输出：

```text
/tmp/quantx-research/adaptive-training-protocol-v1/forward_2026_5d_topk_label_diagnostic.json
sha256:c1959e6c4d6015892ebcb80f80bf9d90ba005aa8c8fb1ca0875c1ab4efd5e998
```

| 方案 | Top10 label | Top15 label | Top20 label | Top1-5 | Rank6-10 | Rank11-15 | Rank16-20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val252 | `+0.011689` | `+0.010287` | `+0.008634` | `+0.019548` | `+0.003830` | `+0.007485` | `+0.003674` |
| val63 | `+0.018237` | `+0.013474` | `+0.012018` | `+0.023242` | `+0.013232` | `+0.003948` | `+0.007652` |
| val20 | `+0.017937` | `+0.013917` | `+0.010406` | `+0.022856` | `+0.013018` | `+0.005876` | `-0.000125` |

TopK 标签层比五分位更支持缩短 validation：`val63` 的 Top10/Top20 明显高于 `val252`，`val20` 的 Top15 略高但 Top20 尾部变差。

### 12.5 2026 ML-only 账户结果

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本、现有 frozen runner 风险口径。输出汇总：

```text
/tmp/quantx-research/adaptive-training-protocol-v1/adaptive_training_protocol_summary.json
sha256:21b66b674261f5b440c9c0bdb2237dc14f2fd342b29c9a4053962003d83b3baf
```

| 方案 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 胜率 | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val252 Top15 | `-2.62%` | `-9.25%` | -0.48 | 5.83 | 8.60 | 40.46% | 0.87 |
| val63 Top15 | `+2.97%` | `-7.22%` | 0.64 | 5.83 | 8.40 | 40.30% | 1.13 |
| val20 Top15 | `+5.15%` | `-7.54%` | 1.07 | 5.87 | 8.58 | 40.91% | 1.25 |
| val252 Top20 | `-2.84%` | `-9.91%` | -0.59 | 7.65 | 8.83 | 40.80% | 0.86 |
| val63 Top20 | `+4.07%` | `-5.91%` | 0.90 | 7.65 | 8.54 | 46.86% | 1.21 |
| val20 Top20 | `+4.09%` | `-6.76%` | 0.93 | 7.69 | 8.53 | 43.75% | 1.22 |
| val63 Top10 | `+10.09%` | `-5.83%` | 2.13 | 4.02 | 8.60 | 40.91% | 1.49 |

Top10 半年收益最高，但平均股票数只有 4.02，不满足用户硬约束。Top15/Top20 满足平均持仓数和持仓周期，并且缩短 validation 后 2026 ML-only 从负转正，但收益量级只有半年约 `+3%` 至 `+5%`，远低于五年几十倍目标。

### 12.6 判定

`partial`，不晋级策略库。

本轮证明一个重要问题：年度冻结模型确实过度丢弃近期标签；把 validation 从 252 缩到 63 或 20 能改善 2026 账户路径，尤其合规 Top15/Top20 从负收益转为正收益。这说明“近期适应性”方向是对的。

但它不是用户要找的高收益主线：

1. 2026 合规 Top15/Top20 只从负收益改善到低个位数正收益，没有形成收益量级跃迁。
2. 最强 Top10 不满足平均持仓大于 5，不能作为候选。
3. `val20` 在预测层不如 `val63`，说明简单纳入越多近期数据并不单调有效。
4. 当前 trainer 没有真正使用 validation 做 early stopping 或超参选择，继续只调 validation 长度会变成弱调参。

下一轮不继续微调 validation_sessions。更值得验证的是结构性目标改变：让模型直接学习 TopK 账户关心的尾部排序，例如 LambdaRank/Pairwise ranking、TopK 加权样本、近期时间衰减样本权重，或按市场状态/行业概念残差做二阶段模型。核心问题应从“训练到哪天”升级为“模型到底在优化什么”。

## 13. topk_objective_v1

### 13.1 假设

普通 LightGBM 回归优化的是全体股票的标签均方误差，不直接优化 Top10/15/20 组合收益。一个自然想法是给训练期每天未来 5 日超额收益靠前的样本更高权重，让模型更重视真正可能进入账户 TopK 的赢家。本轮先做最小诊断，不改正式 trainer，也不做账户回测：只比较同一数据、同一 split、同一参数下，训练样本权重改变后 OOS TopK label 是否稳定改善。

### 13.2 方法

诊断脚本：

```text
/tmp/quantx-research/topk-objective-v1/analyze_weighted_topk_objective.py
```

输出：

```text
/tmp/quantx-research/topk-objective-v1/weighted_topk_objective_smoke_summary.json
sha256:9f9b1bfcf2d5f03eb90e5d72015b1f4b8607fe1d804c7460f5c106f40fa54cf9
```

测试两个单年 fold：

1. 2024：基础 5 日模型收益较强的一年。
2. 2025：基础 5 日模型开发期最后一年，也更接近 2026。

训练目标：

1. `uniform`：原始等权回归基线。
2. `top_decile_x4`：每天标签前 10% 样本权重 4 倍。
3. `top5_x8`：每天标签前 5% 权重 8 倍、5%-10% 权重 3 倍。
4. `recent_top_decile_x4`：时间衰减 + 前 10% 权重 4 倍。

### 13.3 结果

2024 单年，近期 + Top10% 加权看起来有效：

| 2024 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| `uniform` | `+0.015910` | `+0.012461` | `+0.011743` | 基线 |
| `recent_top_decile_x4` | `+0.024102` | `+0.016514` | `+0.013201` | Top10/15/20 全部改善 |
| `top_decile_x4` | `+0.021533` | `+0.015059` | `+0.011629` | Top10/15 改善，Top20 基本持平 |
| `top5_x8` | `+0.015466` | `+0.010390` | `+0.005589` | 过度强调头部后变差 |

但 2025 单年完全反转：

| 2025 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| `uniform` | `+0.013750` | `+0.012305` | `+0.011169` | 基线最强 |
| `recent_top_decile_x4` | `+0.004356` | `+0.000193` | `-0.001630` | 显著变差 |
| `top_decile_x4` | `+0.003679` | `+0.000348` | `-0.001435` | 显著变差 |
| `top5_x8` | `-0.004203` | `-0.006642` | `-0.009226` | 明确失败 |

### 13.4 反事实分析

如果只看 2024，会误判为“TopK 加权目标有效”。但 2025 的反杀说明这个方向在风格上非常不稳定：它更像是在追逐上一阶段的赢家形态，而不是学习稳定的横截面结构。

可能原因：

1. **标签前 5%-10% 噪声很大**：单日未来 5 日赢家里有大量偶然跳涨、消息和流动性噪声，简单加权会放大噪声。
2. **2024 和 2025 的赢家形态不同**：2024 的强趋势/高弹性形态可能更适合加权，2025 则可能更偏轮动、反转或结构分散。
3. **回归样本权重不是排序损失**：它并没有显式优化同一天内 TopK 的相对顺序，只是改变了误差权重，容易牺牲中高分段的稳定排序。
4. **越头部越不鲁棒**：`top5_x8` 在两年都明显差，说明把目标过度聚焦到极端赢家会损害泛化。

### 13.5 判定

`rejected`。

简单 TopK 样本加权不进入完整五年训练和账户回测。它在 2024 有效果、2025 明确失败，不满足鲁棒性要求。下一步不能继续调权重倍数，而应转向更有结构的方式：

1. 真正的 pairwise/listwise ranking，按同日横截面排序学习，而不是加权回归。
2. 状态条件下的目标函数：先判断当前市场是趋势延续、轮动扩散还是反转环境，再选择/融合模型。
3. 分层残差模型：先保留基础 ML 的全市场排序，再学习高分股在行业/概念/市场状态下何时失效。

## 14. state_conditional_residual_v1

### 14.1 假设

`topk_objective_v1` 暴露出一个关键矛盾：TopK 样本加权在 2024 有效、2025 失效。若这不是随机波动，而是市场状态切换，则下一步不应该继续调权重，而应先理解 2024/2025 的高分股形态差异，以及哪些状态下基础 ML Top15/20 的尾部自然更可靠。

### 14.2 市场状态和候选形态诊断

诊断脚本：

```text
/tmp/quantx-research/state-conditional-residual-v1/analyze_regime_residual.py
```

输出：

```text
/tmp/quantx-research/state-conditional-residual-v1/regime_residual_diagnostic.json
sha256:4fba60626a56c0003e9a5e35b489238199a94447b63957eddf8c7cc31079967d
```

核心观察：

1. 2025 的高分候选并不是“不强”。相对 2024，Top20 候选的 `ret20_mean` 高 `+0.0644`、`ret60_mean` 高 `+0.1766`、`rps20_mean` 高 `+0.1340`、`rps60_mean` 高 `+0.1839`、`ma60_dist_mean` 高 `+0.0942`。
2. 2025 的市场也更宽：`breadth20` 高 `+0.0406`、`breadth60` 高 `+0.0839`，60 日市场收益更强。
3. 但 2025 的 20 日横截面离散度略低，候选更像广谱上涨里的强势拥挤，而不是 2024 那种更清晰的强弱分化。
4. 全样本 Top20 label 与横截面离散度关系更明确：20 日收益离散度最高分位的 Top20 label 为 `+0.024732`，正收益日比例 `79.14%`；最低两个分位只有 `+0.009617` 和 `+0.006758`。
5. 高 RPS 本身并非越高越好。Top20 候选 `rps20_mean` 最高分位的 label 只有 `+0.012304`，低于最低分位 `+0.017939`，说明“强势路径”可能已经被市场定价，过强时反而更像拥挤。

这解释了为什么简单 TopK 加权会在 2025 失效：它可能把模型推向更极端的历史赢家形态，而 2025 的高分股已经足够强、足够靠上，再追强只会增加拥挤暴露。

### 14.3 状态条件 anti-chase 反事实

为了避免停留在解释层，本轮又测试了一个小反事实：在已有 ML Top100 内，只在“20 日横截面离散度低于过去 504 日 rolling median”时惩罚过热候选；高分化状态保持原始 ML 排序。

诊断脚本：

```text
/tmp/quantx-research/state-conditional-residual-v1/test_state_conditional_antichase.py
```

输出：

```text
/tmp/quantx-research/state-conditional-residual-v1/state_conditional_antichase_label_diagnostic.json
sha256:00a4967f2d4830c42f067d524428f37b68b2cb2c6b146c7de33f2b2e881193ce
```

结果：

| 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始 ML | `+0.022298` | `+0.017629` | `+0.015657` | 基线 |
| 全时段 anti-chase | `+0.019641` | `+0.016929` | `+0.015123` | 变差 |
| 低分化 anti-chase | `+0.020565` | `+0.017324` | `+0.015253` | 小幅变差 |
| 低分化 pullback | `+0.018457` | `+0.016358` | `+0.014946` | 变差 |

分状态看，低分化日原始 ML Top20 label 为 `+0.013368`，低分化 anti-chase 后降到 `+0.012588`；高分化日不调整时原始 Top20 label 为 `+0.018129`。因此“低分化下惩罚过热”这个手工规则没有修复问题。

### 14.4 判定

`partial`。

本轮最重要的结论不是某个规则有效，而是明确了市场状态的作用方式：

1. 横截面离散度高时，基础 ML Top15/20 更可靠。
2. 但把它做成二元 trust gate 会删除大量正收益日期，之前组合层已失败。
3. 高 RPS/强趋势不是单调好，2025 更像高强度但低分化的拥挤环境。
4. 简单 anti-chase 也会破坏 ML 排序，说明不能靠一个手写惩罚项解决。

下一步应转向模型层的交互学习，而不是继续加手工 gate：

1. **状态交互残差模型**：保留基础 ML 分数，训练一个二阶段模型预测“高分股在当前市场状态下是否会失效”，输入包括市场离散度、宽度、候选 RPS、行业集中度和 score gap。
2. **真正 listwise/pairwise ranker**：按日构造排序任务，让模型学习同一市场状态下 TopK 内部相对顺序。
3. **状态分桶模型融合**：高分化状态使用趋势/赢家形态，低分化状态使用更稳健的基础排序或残差过滤，但必须用 walk-forward 学出来，不能再手写阈值。

## 15. state_interaction_residual_v1

### 15.1 假设

前一节说明手写状态规则不够，但市场状态确实影响 TopK 质量。本轮测试一个最小二阶段模型：保留基础 5 日 ML 的全市场排序，每天只取 Top100 候选，用候选自身路径、市场离散度/宽度、score gap、行业集中度等状态特征训练 residual meta-model，预测高分候选的未来 5 日超额 label 或正收益概率，再在 Top100 内重排。

训练方式避免当前年标签泄漏：2022 用 2021 的 OOS 候选训练，2023 用 2021-2022 训练，依次类推。2026 forward 则用 2021-2025 候选训练，再应用到 2026 的 `val63`/`val20` 预测。

### 15.2 开发期结果

诊断脚本：

```text
/tmp/quantx-research/state-interaction-residual-v1/analyze_residual_meta_model.py
```

输出：

```text
/tmp/quantx-research/state-interaction-residual-v1/residual_meta_model_diagnostic.json
sha256:8a616758db8d1a012260829f5361b47c16456ed0553ce618acc32d3c04d464fd
```

2022-2025 汇总：

| 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始 ML | `+0.016957` | `+0.013740` | `+0.012562` | 基线 |
| residual reg only | `+0.018724` | `+0.014723` | `+0.012395` | Top10/15 改善，Top20 变差 |
| ML + residual 30% | `+0.017260` | `+0.013750` | `+0.012506` | 近似持平 |
| ML + residual 50% | `+0.017744` | `+0.014126` | `+0.012339` | Top10/15 改善，Top20 变差 |
| ML + positive 20% | `+0.016463` | `+0.013532` | `+0.012142` | 变差 |

逐年看，residual reg 的 Top15 并不稳定：

| 年份 | 原始 ML Top15 | residual reg Top15 | 变化 |
| --- | ---: | ---: | ---: |
| 2022 | `+0.020670` | `+0.022928` | 改善 |
| 2023 | `+0.009489` | `+0.008904` | 变差 |
| 2024 | `+0.012545` | `+0.016762` | 改善 |
| 2025 | `+0.012225` | `+0.010206` | 变差 |

这说明二阶段模型学到了一部分 2022/2024 的有效状态，但没有穿越到 2023/2025。

### 15.3 2026 forward 标签验证

应用脚本：

```text
/tmp/quantx-research/state-interaction-residual-v1/apply_residual_meta_forward.py
```

输出：

```text
/tmp/quantx-research/state-interaction-residual-v1/residual_meta_forward_2026_val63_label_diagnostic.json
sha256:16665f3e1adaec5d954d47c89d73be3e16d316755a98d1ad85ec3d60b2a6f214

/tmp/quantx-research/state-interaction-residual-v1/residual_meta_forward_2026_val20_label_diagnostic.json
sha256:a726306ba8524eb0c96efa0a0e7b544b2db504545f20373932359fc42f9b0103
```

2026 `val63`：

| 方案 | Top10 label | Top15 label | Top20 label |
| --- | ---: | ---: | ---: |
| 原始 ML | `+0.018237` | `+0.013474` | `+0.012018` |
| ML + positive 20% | `+0.018641` | `+0.014511` | `+0.011774` |
| ML + residual 30% | `+0.018196` | `+0.014028` | `+0.011298` |
| residual reg only | `+0.014773` | `+0.011662` | `+0.009902` |

2026 `val20`：

| 方案 | Top10 label | Top15 label | Top20 label |
| --- | ---: | ---: | ---: |
| 原始 ML | `+0.017937` | `+0.013917` | `+0.010406` |
| ML + positive 20% | `+0.017432` | `+0.012730` | `+0.011346` |
| ML + residual 30% | `+0.015608` | `+0.013741` | `+0.010541` |
| residual reg only | `+0.012460` | `+0.008987` | `+0.008157` |

forward 结果同样不稳定：`val63` 的 Top15 略有改善但 Top20 下降；`val20` 的 Top20 略有改善但 Top10/15 下降。

### 15.4 判定

`rejected`，不进入账户回测。

二阶段状态 residual 有弱信号，但没有满足“鲁棒、容易出收益”的要求：

1. 开发期 Top15 改善来自 2022/2024，2023/2025 变差。
2. Top20 是用户更容易满足平均持仓约束的方向，但 residual 在开发期和 2026 `val63` 都削弱 Top20。
3. 2026 forward 的增益依赖 `val63`/`val20` 训练边界和 TopK，不稳定。
4. 这个 residual 仍然是点预测/分类，不是真正按同日 TopK 排序优化。

下一步不再继续调 residual 混合权重。更合理的方向是直接测试真正的排序学习：用 LightGBM ranker/LambdaRank 或 pairwise objective 按每日候选组训练，明确优化同日横截面内的相对顺序，而不是用回归或二阶段分类间接逼近。

## 16. daily_ranker_v1

### 16.1 假设

前几轮说明，回归、样本加权和二阶段 residual 都只是间接逼近 TopK 排序。真正的问题是：每天 Top100 高分候选内部，哪些股票应该排进 Top15/20。`daily_ranker_v1` 因此把每天的候选集作为 query group，用 LightGBM LambdaRank 直接学习同日横截面内的相对顺序。

本轮仍不改正式框架，只在 `/tmp` 中诊断。训练样本是基础 5 日 ML OOS Top100 候选；特征包括基础 ML score/rank、市场离散度/宽度、个股 RPS/量价路径、score gap、行业集中度等。测试时比较纯 ranker 排序和 ML/ranker 混合排序。

### 16.2 开发期结果

诊断脚本：

```text
/tmp/quantx-research/daily-ranker-v1/analyze_daily_ranker.py
```

输出：

```text
/tmp/quantx-research/daily-ranker-v1/daily_ranker_diagnostic.json
sha256:b43b278631a0230bd540c10e0d334c8b586add23f0d2435fb3c2747fd37c2ea4
```

2022-2025 walk-forward 汇总：

| 方案 | Top10 label | Top15 label | Top20 label | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原始 ML | `+0.016957` | `+0.013740` | `+0.012562` | 基线 |
| ranker quintile | `+0.017749` | `+0.014062` | `+0.012362` | Top10/15 改善，Top20 变差 |
| ranker topheavy | `+0.016761` | `+0.013271` | `+0.011948` | 变差 |
| ML + quintile 30% | `+0.017400` | `+0.014266` | `+0.012812` | Top10/15/20 均小幅改善 |
| ML + quintile 50% | `+0.016962` | `+0.013718` | `+0.012745` | Top20 小幅改善 |
| ML + topheavy 30% | `+0.016188` | `+0.013639` | `+0.012219` | 变差 |

逐年看，`ML + quintile 30%` 的 Top20 改善较温和但比前几轮稳定：

| 年份 | 原始 ML Top20 | ML + quintile 30% Top20 | 变化 |
| --- | ---: | ---: | ---: |
| 2022 | `+0.018209` | `+0.018510` | 改善 |
| 2023 | `+0.009229` | `+0.009513` | 改善 |
| 2024 | `+0.011697` | `+0.012132` | 改善 |
| 2025 | `+0.011084` | `+0.011056` | 基本持平 |

开发期信号仍然很弱，Top20 总增益只有 `+0.000249`，不能直接进入策略库；但这是目前少数没有明显伤害 2023/2025 的结构性方向。

### 16.3 2026 forward 标签层

应用脚本：

```text
/tmp/quantx-research/daily-ranker-v1/apply_daily_ranker_forward.py
```

输出：

```text
/tmp/quantx-research/daily-ranker-v1/daily_ranker_forward_2026_val63_label_diagnostic.json
sha256:2b9d632d1618d51c84c7ada5ec37d98a1a8387a520396e03c2bd0c3dc3e268f3

/tmp/quantx-research/daily-ranker-v1/daily_ranker_forward_2026_val20_label_diagnostic.json
sha256:e88374f558b816fa7c523b1c05e75ce82e07d0bb1b177bd4c26d2be1b8803ff2
```

2026 `val63`：

| 方案 | Top10 label | Top15 label | Top20 label |
| --- | ---: | ---: | ---: |
| 原始 ML | `+0.018237` | `+0.013474` | `+0.012018` |
| ranker topheavy | `+0.024142` | `+0.017072` | `+0.013371` |
| ranker quintile | `+0.021286` | `+0.015338` | `+0.012265` |

2026 `val20`：

| 方案 | Top10 label | Top15 label | Top20 label |
| --- | ---: | ---: | ---: |
| 原始 ML | `+0.017937` | `+0.013917` | `+0.010406` |
| ranker topheavy | `+0.020606` | `+0.014962` | `+0.011959` |
| ranker quintile | `+0.017074` | `+0.013448` | `+0.010187` |

2026 标签层比开发期更强：`ranker_topheavy` 在 `val63` 和 `val20` 两个训练边界下都同时提高 Top10/15/20。

### 16.4 2026 ML-only 账户层

为 `val63` 生成 ranker topheavy Top15/Top20 PredictionStore：

```text
/tmp/quantx-research/daily-ranker-v1/ranker_topheavy_val63_2026_top15_predictions.json
sha256:af01cf04ff036ec232f56cb72394fbdd6f251d8ce5dabfa65e9d263a22cc906d

/tmp/quantx-research/daily-ranker-v1/ranker_topheavy_val63_2026_top20_predictions.json
sha256:8e1ca2cb311b04fd709d67f973d3f421aea5863b3ba29675d8bdf3887805878d
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本。

| 方案 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 胜率 | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val63 原始 ML Top15 | `+2.97%` | `-7.22%` | 0.64 | 5.83 | 8.40 | 40.30% | 1.13 |
| ranker topheavy Top15 | `+6.13%` | `-7.67%` | 1.27 | 5.83 | 8.73 | 43.85% | 1.27 |
| val63 原始 ML Top20 | `+4.07%` | `-5.91%` | 0.90 | 7.65 | 8.54 | 46.86% | 1.21 |
| ranker topheavy Top20 | `+2.05%` | `-8.33%` | 0.42 | 7.65 | 8.55 | 43.10% | 1.08 |

账户层确认了 ranker 的收益更集中在 Top15，而不是 Top20。Top15 满足平均持仓大于 5 和约一周持仓，2026 半年 ML-only 从 `+2.97%` 提到 `+6.13%`；但 Top20 反而变差。

### 16.5 开发期账户层补证

补跑 2022-2025 ML-only Top15 账户层后，`ranker_topheavy` 的开发期复利路径明显弱于原始 ML：

```text
/tmp/quantx-research/daily-ranker-v1/daily_ranker_account_summary.json
sha256:bc3fd0e8f93b33bd78e5a08cac0156cbad73486d68a80b109622dfb8eac2f59d
```

| 方案 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 胜率 | Profit factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 ML Top15 | 2022-2025 | `+242.91%` | `-17.08%` | 1.66 | 7.48 | 8.87 | 51.08% | 2.06 |
| ranker topheavy Top15 | 2022-2025 | `+92.85%` | `-23.61%` | 1.30 | 7.46 | 9.02 | 47.79% | 1.52 |
| ranker topheavy Top15 | 2026 val63 | `+6.13%` | `-7.67%` | 1.27 | 5.83 | 8.73 | 43.85% | 1.27 |
| ranker topheavy Top15 | 2026 val20 | `+7.13%` | `-5.79%` | 1.46 | 5.83 | 8.65 | 45.80% | 1.30 |

2026 `val63/val20` 的 Top15 改善是真实的，但开发期账户层代价过大。它不是在原始 ML 的收益路径上做稳健增强，而是把组合换成了另一条更贴近 2026 风格、但显著伤害 2022-2025 复利的路径。

### 16.6 判定

`rejected`，不进入正式策略库。

`daily_ranker_v1` 的反事实结论如下：

1. 真正同日排序目标比回归残差更贴近问题，但 `ranker_topheavy` 直接接管排序会破坏开发期账户收益。
2. 2026 半年 Top15 从 `+2.97%` 提到 `+6.13%/+7.13%`，仍远低于五年几十倍级别的目标。
3. Top20 账户层在 2026 变差，说明收益更集中于头部，不满足“自然提高 Top15/20 等权质量”的要求。
4. 行业/概念仍是 2026 静态快照辅助特征，不是 PIT-safe。

下一步不继续推进纯 `ranker_topheavy`。若继续 ranker 家族，只验证一个更弱的反事实：保留原始 ML 为主，只用 `ML + quintile 30%` 做小幅排序修正，观察它能否在账户层保住开发期复利并改善 2026；若仍不能同时成立，则切换到账户路径感知目标或新的市场状态建模方向。

## 17. daily_ranker_blend_account_check

### 17.1 假设

`ranker_topheavy` 失败后，唯一还值得补的反事实是弱融合：不让 ranker 接管排序，只用开发期标签层相对稳定的 `ML + quintile 30%` 修正原始 ML 分数：

```text
blend_score = 0.70 * base_pct + 0.30 * ranker_quintile_pct
```

这样可以检验 ranker 是否只是“力度太大”，而不是方向完全无效。

### 17.2 账户层结果

预测导出脚本：

```text
/tmp/quantx-research/daily-ranker-v1/write_ranker_blend_predictions.py
```

账户汇总：

```text
/tmp/quantx-research/daily-ranker-v1/ranker_blend_account_summary.json
sha256:1d4f5b34450bfa9d0da452f6d8a3d9385709420781f11f91c69bfe0e20159d7f
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本。

开发期 2022-2025：

| 方案 | TopK | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 ML | Top15 | `+242.91%` | `-17.08%` | 1.66 | 7.48 | 8.87 | `+14.46%、+4.32%、+70.16%、+68.78%` |
| ML + quintile 30% | Top15 | `+247.15%` | `-16.58%` | 1.64 | 7.48 | 8.86 | `+15.91%、+7.55%、+68.87%、+64.90%` |
| ranker topheavy | Top15 | `+92.85%` | `-23.61%` | 1.30 | 7.46 | 9.02 | 弱于原始 ML |
| 原始 ML | Top20 | `+189.31%` | `-12.71%` | 1.64 | 9.90 | 8.96 | 未展开 |
| ML + quintile 30% | Top20 | `+189.13%` | `-13.91%` | 1.61 | 9.89 | 8.96 | `+9.67%、+3.51%、+61.91%、+57.32%` |

2026 forward：

| 方案 | TopK | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val63 原始 ML | Top15 | `+2.97%` | `-7.22%` | 0.64 | 5.83 | 8.40 |
| val63 ML + quintile 30% | Top15 | `+6.83%` | `-6.41%` | 1.40 | 5.83 | 8.53 |
| val63 ranker topheavy | Top15 | `+6.13%` | `-7.67%` | 1.27 | 5.83 | 8.73 |
| val63 原始 ML | Top20 | `+4.07%` | `-5.91%` | 0.90 | 7.65 | 8.54 |
| val63 ML + quintile 30% | Top20 | `+2.48%` | `-6.47%` | 0.55 | 7.65 | 8.47 |
| val20 原始 ML | Top15 | `+5.15%` | `-7.54%` | 1.07 | 5.87 | 8.58 |
| val20 ML + quintile 30% | Top15 | `+5.61%` | `-7.33%` | 1.18 | 5.87 | 8.49 |
| val20 ranker topheavy | Top15 | `+7.13%` | `-5.79%` | 1.46 | 5.83 | 8.65 |
| val20 原始 ML | Top20 | `+4.09%` | `-6.76%` | 0.93 | 7.69 | 8.53 |
| val20 ML + quintile 30% | Top20 | `+4.26%` | `-7.06%` | 0.94 | 7.69 | 8.49 |

### 17.3 判定

`rejected`，不进入正式策略库。

弱融合证明了 ranker 不是完全无效：Top15 开发期没有被破坏，`val63` 2026 Top15 也明显改善。但这仍不是目标级别的鲁棒 Alpha：

1. 开发期 Top15 只从 `+242.91%` 到 `+247.15%`，增益太小，且 Sharpe 略降。
2. 开发期 Top20 几乎没有改善，回撤和 Sharpe 还略差。
3. 2026 forward 对 Top15 有帮助，但 Top20 在 `val63` 变差，`val20` 只小幅改善。
4. 这个方向仍表现为头部增强，不是自然提高 Top15/20 等权质量。

下一轮不继续调 ranker 混合权重。研究重点切换到**账户路径感知/周频一致性**：检查原始 ML 分数在 5 日调仓节奏下的持续性、换手、持仓复用和交易日状态，而不是只优化每天独立的 TopK 标签。

## 18. weekly_prediction_persistence_v1

### 18.1 假设

前几轮反复出现“每日 TopK 标签改善，但账户净值不改善”的问题。原因可能是账户只按 5 个交易日调仓，实际吃到的是一串离散的调仓日，而不是所有交易日的平均标签。

本轮不训练新模型，只在原始 ML Top100 内做一个因果可见的周频一致性重排：

```text
persist_top20_30 = 0.70 * current_base_pct + 0.30 * previous_4_sessions_top20_frequency
```

直觉：如果一只股票不是当天突然冲到高分，而是在最近 4 个信号日里反复进入 Top20，那么它更可能是可持有的一周级别机会。这个方向不改变尾部权重，也不使用行业/概念静态快照。

### 18.2 标签层诊断

脚本：

```text
/tmp/quantx-research/weekly-persistence-v1/analyze_prediction_persistence.py
```

输出：

```text
/tmp/quantx-research/weekly-persistence-v1/prediction_persistence_dev_2021_2025_label_diagnostic.json
sha256:5df404a9315c6c8b7cd234562fa031ff8b2f4937b3028da06dd0398c6179cea7

/tmp/quantx-research/weekly-persistence-v1/prediction_persistence_2026_val63_label_diagnostic.json
sha256:0f38660bb98f7395a649bcbbc2b4b31a49e5e65445f9726a0c72209542f85921

/tmp/quantx-research/weekly-persistence-v1/prediction_persistence_2026_val20_label_diagnostic.json
sha256:a9559a15012024fe9ea328422a44d3f44a8bd6a16d06bedde3864f4c53f796ea
```

在所有交易日上，持久性信号只是弱改善；但在账户实际 5 日调仓会用到的 `due5` 子样本上，开发期 Top20 的改善更稳定：

| 样本 | 方案 | Top15 label | Top20 label | 逐年稳定性 |
| --- | --- | ---: | ---: | --- |
| all | 原始 ML | `+0.017629` | `+0.015657` | 基线 |
| all | persist_top20_30 | `+0.017927` | `+0.015687` | Top15 小幅改善，Top20 基本持平 |
| due5 | 原始 ML | `+0.015040` | `+0.013104` | 基线 |
| due5 | persist_top20_30 | `+0.015763` | `+0.014189` | Top20 五年全部改善 |

2026 标签层并不完美：`val63` 明显支持持久性，`val20` 在 all-sessions Top15/20 上偏弱。但由于账户只吃 due5 路径，仍进入账户层验证。

### 18.3 账户层结果

预测文件：

```text
/tmp/quantx-research/weekly-persistence-v1/persist_top20_30_dev_2021_2025_top20_predictions.json
sha256:5461eef4ee5d0587224891a5ce3fca0a2ec20e4efc2be57666081030d36cb970

/tmp/quantx-research/weekly-persistence-v1/persist_top20_30_val63_2026_top20_predictions.json
sha256:66294a7547c653edd8780a88ba1926e6828ba0fada884a262c8fe643d7918cd4

/tmp/quantx-research/weekly-persistence-v1/persist_top20_30_val20_2026_top20_predictions.json
sha256:33dd62d89e3c20683f51dbc45cba4efa44d42df56f0bde738239bfa8c55c4f4e
```

账户汇总：

```text
/tmp/quantx-research/weekly-persistence-v1/weekly_persistence_account_summary.json
sha256:f7b1dd135d27e34eacf0dadf39642bbfa1a1934134579aa5037df02afa2ab416
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本。

开发期 2021-2025：

| 方案 | TopK | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 ML | Top13 | `+386.30%` | `-8.46%` | 2.06 | 6.66 | 9.33 | 未展开 |
| persist_top20_30 | Top13 | `+394.00%` | `-9.84%` | 2.04 | 6.66 | 10.02 | `+47.10%、+27.50%、+23.40%、+48.50%、+43.80%` |
| persist_top20_30 双倍成本 | Top13 | `+320.89%` | `-10.16%` | 1.80 | 6.66 | 10.02 | 五年全正 |
| 原始 ML | Top20 | `+275.83%` | `-8.98%` | 2.04 | 10.17 | 9.08 | `+34.72%、+26.12%、+18.04%、+40.29%、+33.58%` |
| persist_top20_30 | Top20 | `+364.59%` | `-8.68%` | 2.37 | 10.16 | 9.66 | `+41.45%、+27.94%、+24.27%、+47.43%、+40.13%` |
| persist_top20_30 双倍成本 | Top20 | `+295.62%` | `-8.78%` | 2.09 | 10.16 | 9.67 | 五年全正 |

2026 forward：

| 方案 | TopK | val63 | val20 | 双倍成本 val63 | 双倍成本 val20 | 平均股票数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 ML | Top13 | `+5.41%` | `+8.12%` | 未跑 | 未跑 | 约 5.1 |
| persist_top20_30 | Top13 | `+8.61%` | `+10.74%` | `+6.79%` | `+8.90%` | 5.15 / 5.10 |
| 原始 ML | Top20 | `+4.07%` | `+4.09%` | 未跑 | 未跑 | 约 7.7 |
| persist_top20_30 | Top20 | `+6.24%` | `+5.02%` | `+4.49%` | `+3.27%` | 7.65 / 7.69 |

补充消融后，Top13 最佳变体不是连续频率版，而是更保守的 `prev_top20_bonus`：

```text
prev_top20_bonus = current_base_pct + 0.10 * has_previous_top20 + 0.05 * appeared_at_least_twice_in_previous_4_sessions
```

| 方案 | TopK | 开发期收益 | 开发期回撤 | 开发期 Sharpe | 2026 val63 | 2026 val20 | 双倍成本开发期 | 双倍成本 val63/val20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 ML | Top13 | `+386.30%` | `-8.46%` | 2.06 | `+5.41%` | `+8.12%` | 未跑 | 未跑 |
| persist_top20_30 | Top13 | `+394.00%` | `-9.84%` | 2.04 | `+8.61%` | `+10.74%` | `+320.89%` | `+6.79% / +8.90%` |
| prev_top20_bonus | Top13 | `+434.53%` | `-8.03%` | 2.18 | `+8.58%` | `+9.37%` | `+353.00%` | `+6.78% / +7.51%` |
| persist_strong50 | Top13 | `+416.35%` | `-9.55%` | 2.05 | `+6.14%` | `+10.84%` | 未跑 | 未跑 |

### 18.4 判定

`retained_candidate`，但不进入正式策略库。

这是本轮里第一个同时满足以下条件的方向：

1. 开发期账户层改善，Top20 从 `+275.83%` 提高到 `+364.59%`，并且五年全部正收益。
2. Top13 的 `prev_top20_bonus` 变体收益效率更高，开发期 `+434.53%`、2026 `val63/val20` 为 `+8.58%/+9.37%`。
3. 2026 `val63` 和 `val20` 两个训练边界均改善，ML-only 袖套自身为正。
4. Top13/Top20 的平均股票数均满足大于 5；Top13 在 2026 也刚好合规。
5. 双倍成本仍为正，说明不是完全靠成本边缘。
6. 规则只使用过去 4 个信号日的预测持久性，不依赖静态行业/概念快照，也不是降低尾部权重。

但它仍未达到用户最终目标：

1. 合规 Top13/Top20 五年约 4-5 倍，仍低于“几十倍至 100 倍”的收益要求。
2. Top13 最好变体相对原始 Top13 有增量，但仍不是量级突破。
3. 当前只有 seed7 和单一基础模型边界，缺多 seed、多训练边界和更长 sealed forward。

下一步保留该方向作为新的稳健基座，不急于合代码。后续研究应围绕它继续找“更强但不破坏持仓合规”的 Alpha，例如把周频持久性作为二阶段框架的硬约束，再寻找能提高 Top13/Top20 收益效率的因子，而不是继续调权重或 ranker 混合比例。

## 19. liquidity_pth_momentum_v1

### 19.1 假设

`weekly_prediction_persistence_v1` 给出了当前最干净的稳健基座，但收益量级仍不足。本轮测试一个更贴近 A 股短周期结构的叠加方向：在原始 ML Top100 内，用可因果观测的量价路径信号修正排序，包括：

1. 股价接近 120 日高点的程度。
2. 20 日 RPS 和短期动量。
3. 成交量相对 20 日均量的放大。
4. 20 日波动率和回撤强度。

直觉是：如果一只股票既反复被 ML 选中，又处于接近阶段高点、波动较低、没有过度放量追涨的状态，那么它可能比单纯的预测持久性更容易转化为一周持仓收益。

本轮仍不训练新模型，只做因果可见的二阶段重排。所有脚本和产物均在 `/tmp/quantx-research/liquidity-pth-momentum-v1/`。

### 19.2 标签层诊断

脚本：

```text
/tmp/quantx-research/liquidity-pth-momentum-v1/analyze_liquidity_pth_overlay.py
```

输出：

```text
/tmp/quantx-research/liquidity-pth-momentum-v1/liquidity_pth_overlay_dev_2021_2025_label_diagnostic.json
sha256:6ce864a2f06f0519a0a5f3508140a398956345c1cc854266c4be20fb3d8af7a1

/tmp/quantx-research/liquidity-pth-momentum-v1/liquidity_pth_overlay_2026_val63_label_diagnostic.json
sha256:65db9e426a543c40e9a472d3c843c14f770c22ea2e00304f0f7988dba81c0860

/tmp/quantx-research/liquidity-pth-momentum-v1/liquidity_pth_overlay_2026_val20_label_diagnostic.json
sha256:b4495cdf3a4e6370d6911f0abba58a0b2b9e7671a216ea5b73a2e9b5ad05da1d
```

标签层最有希望的变体是 `near_high_low_vol`。在账户实际会用到的 `due5` 调仓路径上，它相对 `prev_top20_bonus` 有一定改善：

| 样本 | 方案 | Top13 due5 label | Top20 due5 label |
| --- | --- | ---: | ---: |
| 2021-2025 | `prev_top20_bonus` | `+0.017152` | `+0.013440` |
| 2021-2025 | `near_high_low_vol` | `+0.017727` | `+0.013698` |
| 2026 val63 | `prev_top20_bonus` | `+0.006330` | `+0.006097` |
| 2026 val63 | `near_high_low_vol` | `+0.008735` | `+0.006054` |
| 2026 val20 | `prev_top20_bonus` | `+0.007907` | `+0.002058` |
| 2026 val20 | `near_high_low_vol` | `+0.008989` | `+0.003534` |

从标签层看，Top13 的改善比较一致，Top20 在 `val63` 基本持平，在 `val20` 明显改善。因此进入账户层验证。

### 19.3 账户层结果

预测文件：

```text
/tmp/quantx-research/liquidity-pth-momentum-v1/near_high_low_vol_dev_2021_2025_top20_predictions.json
sha256:70381e0dbd5ac107db7d1d69abe3d09173d50117c091c90a018c4ecfda4b2e84

/tmp/quantx-research/liquidity-pth-momentum-v1/near_high_low_vol_val63_2026_top20_predictions.json
sha256:f0342ba9bb269852262d70382bb482ec799b51283418759726f00725f7379373

/tmp/quantx-research/liquidity-pth-momentum-v1/near_high_low_vol_val20_2026_top20_predictions.json
sha256:d0975f7126ce2866c1ad1745e984ce05f02569a61a8ed0af34cabffd835918b9
```

账户汇总：

```text
/tmp/quantx-research/liquidity-pth-momentum-v1/liquidity_pth_account_summary.json
sha256:ab60b618e3b45f639941d18887c6acac40e79ff9f08a06cdb0fdee98a5fabb2e
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open、标准成本。

| 方案 | TopK | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `near_high_low_vol` | Top13 | 2021-2025 | `+413.79%` | `-8.81%` | 2.13 | 6.67 | 9.69 |
| `near_high_low_vol` | Top20 | 2021-2025 | `+336.42%` | `-8.64%` | 2.29 | 10.17 | 9.63 |
| `near_high_low_vol` | Top13 | 2026 val63 | `+8.49%` | `-7.23%` | 1.75 | 5.10 | 8.43 |
| `near_high_low_vol` | Top20 | 2026 val63 | `+4.98%` | `-6.66%` | 1.10 | 7.65 | 8.56 |
| `near_high_low_vol` | Top13 | 2026 val20 | `+7.38%` | `-8.11%` | 1.61 | 5.15 | 8.48 |
| `near_high_low_vol` | Top20 | 2026 val20 | `+5.60%` | `-6.28%` | 1.25 | 7.69 | 8.56 |

对比当前保留候选：

| 方案 | TopK | 开发期收益 | 2026 val63 | 2026 val20 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `prev_top20_bonus` | Top13 | `+434.53%` | `+8.58%` | `+9.37%` | 当前最优 Top13 基座 |
| `near_high_low_vol` | Top13 | `+413.79%` | `+8.49%` | `+7.38%` | 标签层改善没有转化为账户升级 |
| `persist_top20_30` | Top20 | `+364.59%` | `+6.24%` | `+5.02%` | 当前较分散基座 |
| `near_high_low_vol` | Top20 | `+336.42%` | `+4.98%` | `+5.60%` | 只有 `val20` 略好，不够稳定 |

### 19.4 判定

`rejected`，不进入正式策略库。

这轮反事实很有价值，但结论是否定的：`near_high_low_vol` 在 `due5` 标签层看起来能改善 Top13/Top20，但账户层没有超过 `weekly_prediction_persistence_v1`。这说明简单的接近高点、低波动和成交量约束，可能与原始 ML 或周频持久性已经高度重合，也可能在真实换手、成本和路径依赖下变成弱信号。

后续不继续手工调 `pth120`、低波动或成交量阈值。更合理的方向是保留周频持久性作为基座，改做账户路径感知的状态/因子交互：只接受能同时改善开发期、2026 两个训练边界、Top13/Top20 和双倍成本的候选。

## 20. rebalance_memory_v1

### 20.1 假设

`weekly_prediction_persistence_v1` 使用最近 4 个普通信号日的预测持久性，但账户实际每 5 个交易日才重建一次 ML 股票仓位。当前 frozen runner 已经会优先保留“仍在本次 TopK 内”的旧持仓，但如果旧持仓从 Top15 边缘滑到 Top20/30 附近，仍会被换掉。

本轮测试更贴近账户路径的记忆项：在 `persist_top20_30` 基座上，如果一只股票在上一轮调仓已经入选，且本轮仍在 Top100 内、排名没有明显掉队，则给它一个很小的优先级修正：

```text
persist_rebalance_hold_not_fading =
    persist_top20_30 + 0.05 * selected_in_previous_rebalance * current_rank_not_worse_than_previous_rank_plus_15
```

这不是降低尾部权重，也不是调资金分配，而是测试“减少调仓日附近的随机替换”是否能提升真实账户路径。

### 20.2 标签层诊断

脚本：

```text
/tmp/quantx-research/rebalance-memory-v1/analyze_rebalance_memory.py
sha256:c690638795bcc9c414dbbd003be9575cb8ff30cf3f6e7b4d5a9c3e94bc29a9c4
```

输出：

```text
/tmp/quantx-research/rebalance-memory-v1/rebalance_memory_dev_2021_2025_label_diagnostic.json
sha256:97e06dbfb92de8d238e9b8a393b54b4fddf9637b83fd273fbcab5c3b59339ffb

/tmp/quantx-research/rebalance-memory-v1/rebalance_memory_2026_val63_label_diagnostic.json
sha256:9752c3867dc412b22241386ffa1903164a38b9878d3e9ed9144fce08724a445b

/tmp/quantx-research/rebalance-memory-v1/rebalance_memory_2026_val20_label_diagnostic.json
sha256:a222a82ba840666c6352ad47782ccfa95d1fe20d512c81d800b78c13cdc729ed
```

`persist_rebalance_hold_not_fading` 的 Top15 在 `due5` 调仓路径上相对 `persist_top20_30` 三段均为正：

| 样本 | `persist_top20_30` Top15 label | `persist_rebalance_hold_not_fading` Top15 label | 增量 | 换手变化 |
| --- | ---: | ---: | ---: | ---: |
| 2021-2025 | `+0.015763` | `+0.015962` | `+0.000199` | 下降约 4.8pct |
| 2026 val63 | `+0.008020` | `+0.008756` | `+0.000736` | 下降约 6.7pct |
| 2026 val20 | `+0.004440` | `+0.005288` | `+0.000849` | 下降约 4.4pct |

标签增量不大，但方向符合账户路径直觉，并且换手下降，因此进入账户层验证。

### 20.3 账户层结果

预测文件：

```text
/tmp/quantx-research/rebalance-memory-v1/persist_rebalance_hold_not_fading_dev_2021_2025_top15_predictions.json
sha256:09d8de883910286687c24f7c09d2da50bd4d2f2b1753f02719a0341ec6646b87

/tmp/quantx-research/rebalance-memory-v1/persist_rebalance_hold_not_fading_val63_2026_top15_predictions.json
sha256:3c6ac161252e4578e3d39c967d16ec92c0a0f0190bdc9ae3bb3983214b3d14de

/tmp/quantx-research/rebalance-memory-v1/persist_rebalance_hold_not_fading_val20_2026_top15_predictions.json
sha256:8686e4f7add4770cb5ce3199458b51f8e50c9462fa55569734752ed34747c3af
```

账户汇总：

```text
/tmp/quantx-research/rebalance-memory-v1/rebalance_memory_account_summary.json
sha256:7189b618c991181148ed794b5497be81d888372528149ecf91c6cba15d93fef4
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open。

| 方案 | TopK | 成本 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `persist_top20_30` | Top15 | 标准 | 2021-2025 | `+390.16%` | `-8.34%` | 2.20 | 7.66 | 未展开 |
| `persist_rebalance_hold_not_fading` | Top15 | 标准 | 2021-2025 | `+416.54%` | `-8.34%` | 2.28 | 7.66 | 10.01 |
| `persist_rebalance_hold_not_fading` | Top15 | 双倍 | 2021-2025 | `+339.51%` | `-8.54%` | 2.02 | 7.66 | 9.96 |
| `persist_top20_30` | Top15 | 标准 | 2026 val63 | `+5.45%` | `-6.15%` | 1.10 | 5.83 | 未展开 |
| `persist_rebalance_hold_not_fading` | Top15 | 标准 | 2026 val63 | `+6.89%` | `-5.84%` | 1.37 | 5.83 | 8.52 |
| `persist_rebalance_hold_not_fading` | Top15 | 双倍 | 2026 val63 | `+5.07%` | `-6.99%` | 0.99 | 5.83 | 8.52 |
| `persist_top20_30` | Top15 | 标准 | 2026 val20 | `+7.68%` | `-6.72%` | 1.64 | 5.83 | 未展开 |
| `persist_rebalance_hold_not_fading` | Top15 | 标准 | 2026 val20 | `+8.26%` | `-6.89%` | 1.73 | 5.83 | 8.89 |
| `persist_rebalance_hold_not_fading` | Top15 | 双倍 | 2026 val20 | `+6.49%` | `-7.40%` | 1.33 | 5.83 | 8.89 |

对比当前最强合规 Top13：

| 方案 | TopK | 开发期收益 | 2026 val63 | 2026 val20 | 双倍成本开发期 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prev_top20_bonus` | Top13 | `+434.53%` | `+8.58%` | `+9.37%` | `+353.00%` |
| `persist_rebalance_hold_not_fading` | Top15 | `+416.54%` | `+6.89%` | `+8.26%` | `+339.51%` |

### 20.4 判定

`diagnostic_only`，不进入正式策略库。

这轮证明账户路径记忆是有效的稳定器：Top15 开发期和 2026 两个训练边界都超过 `persist_top20_30` Top15，且双倍成本仍为正。但它没有超过当前最强的 `prev_top20_bonus` Top13，也没有把五年收益推到目标量级。因此它只能保留为“路径稳定性组件”的证据，不能作为主策略方向。

后续不继续调 hold bonus 的 0.05/0.10 系数。下一步应寻找更强的自然 alpha 来源；账户路径记忆可以作为候选增强项，但不能替代核心收益来源。

## 21. head_union_ensemble_v1

### 21.1 假设

此前单模型 Top13/Top15 的难点是：为了满足平均持仓大于 5，不得不买同一个模型头部之后的尾部候选，收益效率被稀释。更自然的替代方案是：不买单模型尾部，而是买多个相对独立模型/训练窗口/seed 的 Top5/Top6 头部并集。

本轮使用 7 个已有预测源，不重新训练模型：

1. 5 日长历史 seed7。
2. 10 日长历史 seed7/11/19。
3. 10 日 2016 起训全市场三 seed。

核心思路：

```text
每天每个模型只贡献 Top5 或 Top6；
把这些头部股票做并集；
用 base5d_anchor / sum_pct / best_pct 等简单分数排序；
账户层仍等权，不做头尾权重技巧。
```

### 21.2 标签层诊断

脚本：

```text
/tmp/quantx-research/head-union-ensemble-v1/analyze_head_union.py
sha256:a34dd870870d37024bddf9b1276f6bc43add6c175d3ad4fa35365435ed4ef4a9

/tmp/quantx-research/head-union-ensemble-v1/analyze_head_union_fill.py
sha256:fbda6355851441a2662315e6ccbe69d91768ea01e2f46964c26827c80dacc3ad
```

重要反事实：

| 方案 | 开发期 due5 label | 2026 val63 due5 label | 2026 val20 due5 label | 平均选择数 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 模型 head5 自然并集 Top13/15 | `+0.0212` | `+0.0101` | `+0.0105` | 约 9-10 | Alpha 强，但 2026 平均持仓大概率不足 |
| 4 模型 head5 + `prev_top20_bonus` 补齐 Top13 | `+0.0182` | `+0.0046` | `+0.0082` | 13 | filler 在 2026 val63 明显拖累 |
| 7 模型 head5 自然并集 Top15 | `+0.0185` | `+0.0090` | `+0.0093` | 约 12-15 | 2026 持仓仍略不足 |
| 7 模型 head6 自然并集 Top15 | `+0.0181` | `+0.0091` | `+0.0114` | 约 14-15 | 进入账户层 |

结论：多模型头部并集确实比“单模型尾部补齐”更像自然 Alpha。强行用 weekly persistence filler 补足持仓数会在 2026 `val63` 破坏标签，因此不采用 filler 方案。head6 是当前折中点：候选数足够，标签没有严重塌陷。

### 21.3 账户层结果

写出脚本：

```text
/tmp/quantx-research/head-union-ensemble-v1/write_head_union_predictions.py
sha256:8c518d149e903884192203fad5612d96bea6bd586ebc1c4d1feb0b8dc3da4694
```

代表方案：

```text
head_union_7model_head6_base5d_anchor_top15
```

预测文件：

```text
/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_dev_2021_2025_top15_predictions.json
sha256:d85342d74363fb2fd49be5e64e5540d54355658fe20666dbe84aa9a286ced023

/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_val63_2026_top15_predictions.json
sha256:549311ba4aa8fc5a120d432dd5af5ccc6b3426c4550b8dc05a6e60a73d8586e9

/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_val20_2026_top15_predictions.json
sha256:cb26c7e7c7eb549ad860e22ba19878e4308302ac421f324052f837591a498b9c
```

账户汇总：

```text
/tmp/quantx-research/head-union-ensemble-v1/head_union_ensemble_summary.json
sha256:43efd347e6b2acdea024be2ca316ff59b999f33b6a7928aba994bca4b8e81b40
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open。

| 方案 | 成本 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7model head6 base5d_anchor Top15 | 标准 | 2021-2025 | `+598.72%` | `-10.86%` | 2.74 | 7.58 | 10.75 |
| 7model head6 base5d_anchor Top15 | 双倍 | 2021-2025 | `+497.97%` | `-11.47%` | 2.49 | 7.58 | 10.73 |
| 7model head6 base5d_anchor Top15 | 标准 | 2026 val63 | `+5.11%` | `-6.93%` | 0.93 | 5.47 | 9.09 |
| 7model head6 base5d_anchor Top15 | 双倍 | 2026 val63 | `+3.37%` | `-7.67%` | 0.60 | 5.47 | 9.09 |
| 7model head6 base5d_anchor Top15 | 标准 | 2026 val20 | `+8.26%` | `-6.11%` | 1.43 | 5.47 | 9.11 |
| 7model head6 base5d_anchor Top15 | 双倍 | 2026 val20 | `+6.48%` | `-6.90%` | 1.11 | 5.47 | 9.11 |

`sum_pct` 排序反事实只小幅改善 2026，但明显伤害开发期：

| 方案 | 开发期 | 2026 val63 | 2026 val20 |
| --- | ---: | ---: | ---: |
| head6 base5d_anchor Top15 | `+598.72%` | `+5.11%` | `+8.26%` |
| head6 sum_pct Top15 | `+508.16%` | `+5.19%` | `+8.34%` |

### 21.4 判定

`retained_candidate`，但不进入正式策略库。

这是目前最符合“不要降低尾部权重，要找自然 Alpha”的方向：它不是在一个模型里硬买尾部，而是用多个模型的头部并集自然形成 7-8 只开发期平均持仓、5.47 只 2026 平均持仓。开发期收益也从 weekly persistence Top13 的 `+434.53%` 提高到 `+598.72%`，双倍成本仍有 `+497.97%`。

但它还没达最终门槛：

1. 五年最终倍数约 6.99x，仍低于几十倍至 100x 的目标。
2. 2026 `val63` 只有 `+5.11%`，低于 `prev_top20_bonus` Top13 的 `+8.58%`。
3. 当前 7 个模型不是同一套严格设计的 fresh multi-seed 5 日模型，而是复用了既有 5d/10d、长短历史混合产物，需要进一步审计模型家族和训练边界。
4. 2026 平均持仓刚过线，安全边际不大。

下一步保留这个方向，不急于合代码。更值得做的是训练一组专门用于 head-union 的 5 日多 seed/多训练边界模型，验证“多头部并集”是否能在 2026 保住收益，而不是继续调 head5/head6 或排序公式。

## 22. 5d_head_union_v1

### 22.1 假设

`head_union_ensemble_v1` 的 7 模型版本混合了 5 日长历史、10 日长历史和 10 日短历史模型。它在开发期明显强于周频持久性，但 2026 `val63` 偏弱。一个合理怀疑是：弱点来自 10 日标签或短历史模型污染，而不是“多模型头部并集”这个思路本身。

本轮只做一个更干净的反事实：使用同一套 2010 起训、5 日标签 LightGBM 的 3 个 seed，分别取每天 Top5/Top6 头部做并集。这样候选来自同一标签定义和同一训练协议，不引入 10 日或短历史混合。

### 22.2 数据和产物

新增训练配置均在 `/tmp/quantx-research/5d-head-union-v1/configs/`，只补 seed 11/19；seed 7 复用既有 5 日长历史预测。

共同设置：

| 项目 | 值 |
| --- | --- |
| Provider | `/Users/mingxiaoli/Documents/QuantX-QMT/data/qlib_data_fixed` |
| 股票池 | 当前 QMT 全主板 |
| 训练起点 | 2010-01-04 |
| 标签 | T+1 open 入场、未来 5 个交易日 open 退出的横截面超额收益 |
| 模型 | LightGBM seed 7/11/19 |
| 输出根目录 | `/tmp/quantx-research/5d-head-union-v1/` |

seed 11/19 的 2021-2025 开发期模型指标正常：

| Seed | 日均 RankIC | 全期 RankIC | 五分位 Top-Bottom | 预测数 | SHA256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11 | `0.103097` | `0.103015` | `+0.010945` | 3,716,399 | `sha256:024a1a5a011f59d5eaf1c693d133842a607958cd43849cd0f8419b38236ff590` |
| 19 | `0.102701` | `0.102779` | `+0.010952` | 3,716,399 | `sha256:4d37f41f14e51e8eca35e44f34009cc799690cf724f1e81ac8896914526560bc` |

2026 forward 模型本身也没有崩：

| 窗口 | Seed | 日均 RankIC | 全期 RankIC | 五分位 Top-Bottom | SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| val63 | 11 | `0.049534` | `0.052380` | `+0.001305` | `sha256:cdd73aa5ae5f592d378d671ede6056e9be9e3d3662d5e567f5ea6287dfe91461` |
| val63 | 19 | `0.049576` | `0.051539` | `+0.001106` | `sha256:e9503c6b616af2b062c1726c8f136f5508f6ef77e43c870ef8670bed664d4588` |
| val20 | 11 | `0.048611` | `0.050819` | `+0.000985` | `sha256:c3d98435bf94dd88995fd203333c6000b38b9bc00bdf89dc9372404db5fb1e76` |
| val20 | 19 | `0.048963` | `0.051126` | `+0.000929` | `sha256:02fb4e0b2720f91b8f31634901b6cd86ec49d7b6a8cfec981a239407e44cfedb` |

汇总文件：

```text
/tmp/quantx-research/5d-head-union-v1/5d_head_union_summary.json
sha256:a73fe846797c94eee91affcc625474c253f7a89d5f8eb1b4a87f6261da4b8d7d
```

### 22.3 标签层诊断

3 个纯 5 日 seed 的头部并集在开发期非常强，尤其 head5：

| 方案 | 开发期 due5 Top15 label | 开发期平均选择数 | 2026 val63 due5 Top15 label | val63 平均选择数 | 2026 val20 due5 Top15 label | val20 平均选择数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3x5d head5 | `+0.027331` | 7.92 | `+0.004597` | 7.33 | `+0.004029` | 7.33 |
| 3x5d head6 | `+0.023609` | 9.42 | `+0.005071` | 8.79 | `+0.006060` | 8.54 |

开发期 head5 的标签甚至强于 7 模型混合 head6，但 2026 `due5` 调仓路径明显衰减。head6 增加了候选数，却稀释了开发期标签，不能自然解决问题。

### 22.4 账户层结果

写出的标准 PredictionStore：

```text
/tmp/quantx-research/5d-head-union-v1/head_union_3x5d_head5_base5d_anchor_dev_2021_2025_top15_predictions.json
sha256:383538e38a523de221affa8166ec06e2a6e02e4f0c6699ff8c5146000bb32c87

/tmp/quantx-research/5d-head-union-v1/head_union_3x5d_head5_base5d_anchor_val63_2026_top15_predictions.json
sha256:31230706ab699dcba603de91d3fb6bba4f26c974be9caeb7bce4065c1fe18380

/tmp/quantx-research/5d-head-union-v1/head_union_3x5d_head5_base5d_anchor_val20_2026_top15_predictions.json
sha256:e95aeedc61760595dac5a394b333e58b0daccbc450d1e3f31fef8184d06fe973
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open。

| 风控 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 标准风险闸门 | 2021-2025 | `+822.36%` | `-11.88%` | 2.21 | 4.04 | 9.97 |
| 标准风险闸门 | 2026 val63 | `+4.85%` | `-10.11%` | 0.77 | 2.81 | 8.90 |
| 标准风险闸门 | 2026 val20 | `+2.31%` | `-11.35%` | 0.37 | 2.93 | 8.77 |
| 关闭风险闸门 | 2021-2025 | `+1299.84%` | `-33.40%` | 2.34 | 7.18 | 9.59 |
| 关闭风险闸门 | 2026 val63 | `-4.93%` | `-25.81%` | -0.40 | 7.05 | 10.42 |
| 关闭风险闸门 | 2026 val20 | `-1.55%` | `-27.91%` | -0.13 | 7.00 | 11.03 |

这组对照很重要：标准风险闸门下 2026 小幅正收益主要来自降低股票暴露；关闭风险闸门后平均股票持仓满足要求，但 2026 变成负收益且回撤扩大到约 26%-28%。因此问题不是 10 日模型污染，也不是风险闸门过于保守，而是 2026 纯股票 alpha 本身不够鲁棒。

### 22.5 判定

`rejected`，不进入正式策略库。

纯 5 日 head-union 是非常有价值的反事实，但结论是否定的：它证明“多模型头部并集”在 2021-2025 能自然制造很高的开发期收益，却不能穿越 2026。标准风险闸门把 2026 救成小正，但也把平均股票数压到 3 只以下，不满足用户要求；关闭风控后持仓数量满足，但收益转负。

后续不继续沿着 `head5/head6`、排序公式或 5 日 seed 数量微调。更符合目标的下一方向应寻找能在 2026 纯股票暴露下自然为正的收益来源，例如：

1. 只在市场横截面机会本身足够强时交易的可学习 trust regime，但必须避免手工阈值。
2. 在同一日候选中学习“真实可成交 TopK 组合收益”的路径目标，而不是只优化个股点预测 RankIC。
3. 把 2026 弱点作为主要反事实，优先寻找 forward 先变强、开发期不崩的信号，而不是继续提高开发期收益。

## 23. absolute_return_label_v1

### 23.1 假设

前面多轮失败都指向同一个问题：相对全市场的 5 日横截面超额标签，在弱市里可能选到“相对没那么差、但绝对仍亏钱”的股票。一个直接反事实是把标签从横截面超额收益改成未来 5 日绝对收益，让模型显式学习弱市里的绝对抗跌/上涨能力。

本轮只改标签，不改特征、不加权、不改持仓规则：同一套 2010 起训 5 日 LightGBM seed7，同一套 T+1 open、5 日调仓账户口径。

### 23.2 数据和产物

配置和产物均在：

```text
/tmp/quantx-research/absolute-return-label-v1/
```

共同设置：

| 项目 | 值 |
| --- | --- |
| Provider | `/Users/mingxiaoli/Documents/QuantX-QMT/data/qlib_data_fixed` |
| 股票池 | 当前 QMT 全主板 |
| 训练起点 | 2010-01-04 |
| 标签 | T+1 open 入场、未来 5 个交易日 open 退出的绝对收益 |
| 模型 | LightGBM seed 7 |
| 特征 | 与基础 5 日模型完全一致，feature schema `sha256:1c706289a5485d904852858488f8c19ae6a8a9fa2b96aacf1d9c212d618adc26` |
| 标签 hash | `sha256:91128bb15958dd2688e4d1b43a66fadfc587e560f90db1b5a83a377aaaf31f97` |

汇总文件：

```text
/tmp/quantx-research/absolute-return-label-v1/absolute_return_label_summary.json
sha256:039f3be89db9179833972348ccd024ea9565defa122b6064869a149e3bacff80
```

### 23.3 预测层结果

绝对收益标签在开发期看起来能学到一点排序，但弱于相对收益模型；2026 forward 则出现五分位 Top-Bottom 反向。

| 区间 | 日均 RankIC | 五分位 Top-Bottom | 结论 |
| --- | ---: | ---: | --- |
| 2021-2025 dev | `0.074583` | `+0.007535` | 预测层为正，但低于原相对收益 5 日模型 |
| 2026 val63 | `0.021413` | `-0.001563` | 高分桶绝对收益更差 |
| 2026 val20 | `0.019308` | `-0.002008` | 反向更明显 |

2026 的平均 5 日绝对标签约为 `-0.31%`。如果绝对收益目标有效，Top 桶至少应该比 Bottom 桶抗跌；实际 Top-Bottom 为负，说明模型没有学到弱市里的可交易绝对赢家。

### 23.4 账户层结果

Top15 PredictionStore：

```text
/tmp/quantx-research/absolute-return-label-v1/absolute_5d_seed7_dev_2021_2025_top15_predictions.json
sha256:40b752b0ae8c558622184af34c955b890f3a4842b56445c7ddc8175c1ff47cc9

/tmp/quantx-research/absolute-return-label-v1/absolute_5d_seed7_val63_2026_top15_predictions.json
sha256:804453013030fecc90ad082493326fba85762c1e45f6d4dc3e89628bd0ba778c

/tmp/quantx-research/absolute-return-label-v1/absolute_5d_seed7_val20_2026_top15_predictions.json
sha256:e20e59acbdcb80f24e9593e921496fe58c72b4271e0bb40cba6ba4af0d26c2de
```

共同口径：ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 个交易日调仓、T+1 open。

| 风控 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 标准风险闸门 | 2021-2025 | `+187.58%` | `-17.04%` | 1.64 | 8.01 | 8.16 |
| 标准风险闸门 | 2026 val63 | `-9.19%` | `-15.94%` | -1.51 | 5.95 | 8.45 |
| 标准风险闸门 | 2026 val20 | `-7.04%` | `-13.75%` | -1.22 | 5.95 | 8.45 |
| 关闭风险闸门 | 2021-2025 | `+386.25%` | `-27.80%` | 1.67 | 14.89 | 8.20 |
| 关闭风险闸门 | 2026 val63 | `-18.56%` | `-25.46%` | -1.44 | 14.76 | 10.14 |
| 关闭风险闸门 | 2026 val20 | `-17.85%` | `-25.28%` | -1.35 | 14.72 | 10.00 |

### 23.5 判定

`rejected`，不进入正式策略库。

这轮反事实明确否定了一个朴素解释：2026 纯股票 Alpha 弱，不是简单因为标签用了横截面超额收益。换成绝对收益后，开发期收益更低，2026 反而明显亏损。标准风险闸门也没能救回收益，关闭风控后亏损扩大，说明不是风控过严导致错过机会，而是绝对收益模型选股方向本身在 2026 失效。

后续不继续在“相对标签 vs 绝对标签”之间调参。更可能出现金子的方向应转到 A 股更结构化的短期机制：近期强势扩散、涨停/接近涨停、成交量放大、行业/概念内领涨股，以及这些信号在过去一段时间的横截面持续性。也就是说，不是让基础特征自己从全市场噪声里学出概念轮动，而是把 A 股短周期交易机制显式编码进候选特征，再做严格 2026 forward 检验。

## 24. limitup_concept_propagation_v1

### 24.1 假设

A 股短周期机会经常以行业/概念扩散、涨停或接近涨停、成交量放大和板块内领涨股的形式出现。前面的行业/概念实验失败，主要是把静态行业/概念强度直接塞进全局模型，或用强行业/强概念硬筛候选池。本轮改为更贴近交易机制的反事实：保留基础 5 日 ML 的 Top100/Top200 候选，只在候选内部用显式行业/概念强势扩散因子重排。

本轮不训练新模型，不降低尾部权重，不做资金分配技巧，只验证这些机制信号能否改善真实账户路径。

### 24.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/
```

诊断脚本：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/analyze_limitup_concept_propagation.py
```

写出脚本：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/write_limitup_concept_predictions.py
```

汇总文件：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_propagation_summary.json
sha256:18545d8bc70e3f4745ae374cd9bc28a65889355937efb90ed9df81ae39f5b966
```

共同设置：

| 项目 | 值 |
| --- | --- |
| 基础预测 | 2010 起训基础 5 日 LightGBM seed7 |
| 股票池 | 当前 QMT 全主板 |
| 元数据 | 2026-06-25 静态行业/概念快照 |
| 机制因子 | 行业/概念 1/3/5 日收益、涨停比例、接近涨停比例、强股比例、个股相对行业/概念强弱、成交量放大、接近高点 |
| 组合口径 | ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 日调仓、T+1 open |

重要边界：行业/概念成员仍是当前静态快照，不是 point-in-time 历史成分。因此本轮只能作为机制方向的探索证据，不能直接晋级。

### 24.3 标签层诊断

诊断文件：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_dev_2021_2025_label_diagnostic.json
sha256:031f98d76892dc5b56419b047a6abc624b17db11862b0d8f8ca82eba5189aaea

/tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_2026_val63_label_diagnostic.json
sha256:deb5abaa97d689d663314ed84478b64fc2cfc10c8a58c2752def6d0be870a419

/tmp/quantx-research/limitup-concept-propagation-v1/limitup_concept_2026_val20_label_diagnostic.json
sha256:4ccd949f712b2a88c710d7f1e3bcc514fcfeb161d896d65d4c5320b1464afd14
```

重点看 `due5`，因为账户每 5 个交易日调仓一次。

| 方案 | TopK | 开发期 label 增量 | 2026 val63 增量 | 2026 val20 增量 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `concept_leader_confirmed` pool100 | Top15 | `+0.002347` | `+0.003783` | `+0.001312` | 三段均为正，进入账户层 |
| `industry_leader_confirmed` pool200 | Top15 | `+0.000747` | `+0.005639` | `+0.003523` | 2026 更强，进入账户层 |

标签层说明行业/概念强势扩散不是纯噪声。尤其 2026 `due5` 上，行业/概念内领涨和扩散因子能改善基础 ML 的 Top15 排序。但 `all` 日频口径下多数重排不如基础 ML，说明信号更贴近账户调仓节奏，或者存在样本少导致的路径偶然性，必须进入账户层验证。

### 24.4 账户层结果

PredictionStore：

```text
/tmp/quantx-research/limitup-concept-propagation-v1/concept_leader_pool100_dev_2021_2025_top15_predictions.json
sha256:868f44950a45be2b6095291740055c9675736b8a441d4cfd850ab5abef6fe02b

/tmp/quantx-research/limitup-concept-propagation-v1/concept_leader_pool100_val63_2026_top15_predictions.json
sha256:bfbebde7c727b3b587de5f4f049c1f13a081f0959ede026942e50461c900d09c

/tmp/quantx-research/limitup-concept-propagation-v1/concept_leader_pool100_val20_2026_top15_predictions.json
sha256:e44236a1010d537de5428ec0d21b9024c93d3c715296fdd2d43eb985933d04d5

/tmp/quantx-research/limitup-concept-propagation-v1/industry_leader_pool200_dev_2021_2025_top15_predictions.json
sha256:8887f89daeb47d605c82490346bb0e18f896590e57e02245e9866f1aec1919b2

/tmp/quantx-research/limitup-concept-propagation-v1/industry_leader_pool200_val63_2026_top15_predictions.json
sha256:ccb1794447f922464fa63809cedbf0cfacb798a955a25a2aec7e7ff8a07588dd

/tmp/quantx-research/limitup-concept-propagation-v1/industry_leader_pool200_val20_2026_top15_predictions.json
sha256:27699c8b68b68b03b22ac1d2f2856996102c8e646868d2266404b5dbe1889893
```

账户结果：

| 方案 | 风控 | 区间 | 累计收益 | 最大回撤 | Sharpe | 观察 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| concept leader pool100 Top15 | 标准 | 2021-2025 | `+404.88%` | `-8.84%` | 2.23 | 开发期中等，低于 head-union |
| concept leader pool100 Top15 | 标准 | 2026 val63 | `+5.86%` | `-7.11%` | 1.21 | 正收益但期末只剩 1 只 |
| concept leader pool100 Top15 | 标准 | 2026 val20 | `+5.17%` | `-7.29%` | 1.09 | 正收益但期末只剩 1 只 |
| concept leader pool100 Top15 | 关闭风控 | 2021-2025 | `+1159.54%` | `-26.79%` | 2.59 | 收益高但回撤大 |
| concept leader pool100 Top15 | 关闭风控 | 2026 val63 | `-1.58%` | `-22.69%` | -0.14 | 纯股票暴露仍不稳 |
| concept leader pool100 Top15 | 关闭风控 | 2026 val20 | `+1.19%` | `-20.77%` | 0.12 | 勉强为正但风险收益弱 |
| industry leader pool200 Top15 | 标准 | 2021-2025 | `+238.93%` | `-9.93%` | 1.65 | 开发期收益太低 |
| industry leader pool200 Top15 | 标准 | 2026 val63 | `+5.55%` | `-7.59%` | 1.17 | 正收益但期末只剩 1 只 |
| industry leader pool200 Top15 | 标准 | 2026 val20 | `+4.69%` | `-8.32%` | 0.95 | 正收益但不强 |
| industry leader pool200 Top15 | 关闭风控 | 2021-2025 | `+710.23%` | `-25.38%` | 2.10 | 低于 concept leader |
| industry leader pool200 Top15 | 关闭风控 | 2026 val63 | `-0.33%` | `-21.54%` | -0.03 | 接近打平但未过关 |
| industry leader pool200 Top15 | 关闭风控 | 2026 val20 | `+9.90%` | `-16.08%` | 0.97 | 两个 forward 不一致 |

### 24.5 判定

`diagnostic_only`，不进入正式策略库。

这轮比 absolute-return 标签更接近正确方向：行业/概念强势扩散、板块内领涨和量能确认确实包含短周期信息，特别是能把 2026 关闭风控的结果从明显亏损拉到接近打平或小正。但它仍不满足最终目标：

1. concept leader 开发期关闭风控达到 `+1159.54%`，但 2026 `val63` 关闭风控仍为负。
2. industry leader 2026 更稳，但开发期收益只有 `+710.23%`，离几十倍目标很远。
3. 标准风控下 2026 正收益主要伴随股票暴露下降，期末持仓只剩 1 只，不能证明纯股票 Alpha 已经修复。
4. 手工重排公式对 Top10/Top15/Top20、pool100/pool200 的敏感性较高，不适合继续调公式。

下一步不继续手工调 `concept_burst`、`industry_burst` 或 chase penalty。更合理的方向是把这些机制因子纳入可学习模型：在基础 ML Top100/Top200 候选内，使用行业/概念扩散、涨停接近度、成交量确认、相对行业/概念强弱和市场状态作为二阶段学习特征，训练一个专门优化调仓日 TopK 的 meta-ranker。也就是说，保留本轮机制特征，但从手写公式切换为可学习、可交叉验证的路径模型。

## 25. mechanism_meta_ranker_v1

### 25.1 假设

上一轮 `limitup_concept_propagation_v1` 证明行业/概念强势扩散、板块内领涨、成交量确认和相对行业/概念强弱有短周期信息，但手工公式不够鲁棒。本轮测试更自然的版本：在基础 5 日 ML Top200 候选内，把这些机制因子和市场状态一起交给二阶段模型学习，同时保留一个不训练的 `mech_static` 反事实。

本轮目标不是降低尾部权重，也不是靠资金分配卡结果，而是验证“可学习的机制特征二阶段排序”能否同时提升开发期和 2026 forward。

### 25.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/mechanism-meta-ranker-v1/
```

脚本：

```text
/tmp/quantx-research/mechanism-meta-ranker-v1/analyze_mechanism_meta_ranker.py
/tmp/quantx-research/mechanism-meta-ranker-v1/write_mechanism_static_predictions.py
```

汇总文件：

```text
/tmp/quantx-research/mechanism-meta-ranker-v1/mechanism_meta_ranker_summary.json
sha256:e53c6582216aaf3ad555fa5320a11969b7e09924db7ae66ccb8990b3193710bb
```

共同设置：

| 项目 | 值 |
| --- | --- |
| 基础预测 | 2010 起训基础 5 日 LightGBM seed7 |
| 候选池 | 每日 Top200 |
| 二阶段特征 | base rank/score、市场 5/20/60 日中位收益和离散度、breadth20/60、个股 1/3/5/20 日强度、成交量放大、接近高点、行业/概念收益、涨停/接近涨停比例、强股比例、个股相对行业/概念强弱 |
| 模型 | LightGBM regressor/classifier，按年份 walk-forward；2026 用 2021-2025 训练后 forward |
| 账户口径 | ML-only、`ml_allocation=1.0`、`wufu_allocation=0`、5 日调仓、T+1 open |
| 元数据边界 | 行业/概念仍为 2026-06-25 静态快照，不是 point-in-time 历史成分 |

### 25.3 标签层诊断

诊断文件：

```text
/tmp/quantx-research/mechanism-meta-ranker-v1/mechanism_meta_dev_2021_2025_label_diagnostic.json
sha256:63273fd17cfcfc10e4b45069fd80264ea211a8d74837f8598d282b402025f723

/tmp/quantx-research/mechanism-meta-ranker-v1/mechanism_meta_val63_2026_label_diagnostic.json
sha256:31ab08ae823b6fbac92d8ff63ef700749d349d59ae969dbaaec3d3ac2e5bbe2b

/tmp/quantx-research/mechanism-meta-ranker-v1/mechanism_meta_val20_2026_label_diagnostic.json
sha256:fe57e4fbd2f2f11b33a00f30bceb37421f14b3a3e09ce577de15a54263db114b
```

重点看 `due5`，因为账户每 5 个交易日才重建股票仓位。

| 方案 | TopK | 开发期 label 增量 | 2026 val63 增量 | 2026 val20 增量 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `meta_reg` | Top15 | `+0.003823` | `+0.000531` | `-0.005591` | 开发期好，但 val20 明显反向 |
| `meta_pos` | Top15 | `-0.001085` | `+0.006536` | `-0.000973` | 2026 val63 好，开发期和 val20 不稳 |
| `mech_static` | Top15 | `+0.003447` | `+0.006823` | `-0.000142` | 更稳，但 val20 Top15 未改善 |
| `mech_static` | Top20 | `+0.003001` | `+0.000552` | `+0.001085` | 三段同向为正，但幅度不大 |

标签层结论：学习型 `meta_reg` 不是鲁棒增强。它把 2021-2025 的二阶段关系学得很强，但 2026 val20 调仓日反向，说明二阶段模型在弱样本下仍会追训练期噪声。相比之下，无训练的 `mech_static` 更稳，尤其 Top20 due5 三段均为正，因此进入账户层做真实路径反事实。

### 25.4 账户层结果

PredictionStore：

```text
/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_dev_2021_2025_top15_predictions.json
sha256:021cede78703b6d700d9905cbb9114f0395cd6cfb86a27491935ca8a7551d1de

/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_val63_2026_top15_predictions.json
sha256:339ee4e86723d4bdda9da5052e31121b528eff08bb13ceb94170e772fc86c23a

/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_val20_2026_top15_predictions.json
sha256:f8a0a99a972b6b23d504dc2fb803cc96e8f4fb2fc95e919bcc8aea3085446d7c

/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_dev_2021_2025_top20_predictions.json
sha256:8bf3716f2eb6cb51afab5ae468d46f9410ac59905530a6300eb42c619cc9ed09

/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_val63_2026_top20_predictions.json
sha256:6aaeea4e17191a077bfdbdd43f7b8429dc97b871b701ef8b80e4c58bc0573a1b

/tmp/quantx-research/mechanism-meta-ranker-v1/mech_static_pool200_val20_2026_top20_predictions.json
sha256:6ef5d5751e7afa3cc2f82a703bf34036b20261eca30886949ce62882abf46b7e
```

账户结果：

| 方案 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均股票数 | 平均持仓天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mech_static` Top15 | 2021-2025 | `+391.29%` | `-8.37%` | 2.18 | 7.65 | 9.10 |
| `mech_static` Top15 | 2026 val63 | `+6.01%` | `-6.50%` | 1.30 | 5.87 | 8.53 |
| `mech_static` Top15 | 2026 val20 | `+1.57%` | `-10.12%` | 0.31 | 5.83 | 8.36 |
| `mech_static` Top20 | 2021-2025 | `+314.27%` | `-8.66%` | 2.20 | 10.16 | 9.25 |
| `mech_static` Top20 | 2026 val63 | `+3.36%` | `-6.55%` | 0.75 | 7.65 | 8.41 |
| `mech_static` Top20 | 2026 val20 | `+4.18%` | `-6.26%` | 0.97 | 7.65 | 8.40 |

### 25.5 判定

`rejected`，不进入正式策略库。

这轮给出了一个清晰结论：机制特征本身有用，但二阶段学习和静态机制重排都还不是收益主引擎。

1. `meta_reg` 在开发期 due5 Top15 label 增量为 `+0.003823`，但 2026 val20 变成 `-0.005591`，符合二阶段模型过拟合训练期候选关系的特征。
2. `mech_static` Top20 在 dev、val63、val20 的 due5 label 增量都为正，说明行业/概念扩散和相对强弱确实是稳健边际因子。
3. 账户层 Top15/Top20 都通过平均持仓大于 5、平均持仓约一周、2026 正收益这些局部约束，但开发期累计仅 `+391.29%`/`+314.27%`，低于已有 head-union，也远低于几十倍到 100 倍目标。
4. 2026 正收益不强，val20 Top15 只有 `+1.57%`，不能证明弱市纯股票 Alpha 已经恢复。

下一步不继续调二阶段模型参数，也不继续把机制特征做手工线性组合。更可能的方向是市场状态条件下的动态择因子/择模型：先识别何时应该相信趋势持久性、何时相信概念扩散、何时需要防守或降低追涨暴露，而不是让一个全局二阶段模型在所有状态下学习同一套排序关系。

## 26. state_router_v1

### 26.1 假设

前面的实验说明几个候选族都不是纯噪声：基础 5 日、周频持久性、多模型 head-union、概念/行业强势扩散、机制静态重排在不同年份和不同 2026 forward 中各有强弱。本轮验证一个更高层的问题：市场状态能否告诉我们下一个调仓日该信哪个候选族。

本轮不拼账户、不训练股票级模型，只在调仓日层面做标签诊断。如果状态路由在 due5 标签层都不能超过固定最强候选，就没有必要进入账户层。

### 26.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/state-router-v1/
```

脚本：

```text
/tmp/quantx-research/state-router-v1/analyze_state_router.py
```

汇总文件：

```text
/tmp/quantx-research/state-router-v1/state_router_summary.json
sha256:9392c84df0461c146719c68f329249c9e351ff7df3bbebe6f260ed3d5fd1024c
```

诊断文件：

```text
/tmp/quantx-research/state-router-v1/state_router_dev_2021_2025_label_diagnostic.json
sha256:820d770e06c58c180972d1d49b103d73f6b76233dc1d7d1c7e66ca1ba7426644

/tmp/quantx-research/state-router-v1/state_router_val63_2026_label_diagnostic.json
sha256:b8b08e4ce1eb09648815260cdd8e3bcf8495d4d6d1e2da2a6bca469f785925d1

/tmp/quantx-research/state-router-v1/state_router_val20_2026_label_diagnostic.json
sha256:732e4d647e6f874ef55f98223f179968b3b78991a2c7d7e9cbebaecc3212f03e
```

候选族：`base5d_top15`、`weekly_prev_top20`、`head_union`、`concept_leader`、`industry_leader`、`mech_static`。

状态特征：全市场 5/20/60 日中位收益、离散度、80 分位收益、breadth20/60。路由器用训练期三分位分箱，分别测试完整状态 key 和简化的 `market_ret20_median | breadth20` key。

### 26.3 标签层结果

| 路由/候选 | 2021-2025 due5 label | 2026 val63 due5 label | 2026 val20 due5 label | 观察 |
| --- | ---: | ---: | ---: | --- |
| oracle | `+0.02853` | `+0.02265` | `+0.02204` | 事后空间很大，候选族确实有互补性 |
| 固定 `head_union` | `+0.01578` | `+0.00909` | `+0.01138` | 实际最强固定候选 |
| compact regime router | `+0.01277` | `+0.00508` | `+0.00556` | 不如固定 head_union |
| exact regime router | `+0.01178` | `+0.00581` | `+0.00794` | 状态维度更细反而更不稳 |
| 固定 `concept_leader` | `+0.01388` | `+0.00551` | `+0.00732` | 开发期第二，但 2026 弱于 head_union |
| 固定 `mech_static` | `+0.01272` | `+0.00561` | `+0.00443` | 稳但不强 |

### 26.4 判定

`rejected`，不进入账户层。

这轮说明了一个重要反事实：不是“候选族之间没有切换空间”，而是“用普通市场状态特征无法稳定识别该切到谁”。oracle 很强，说明候选族错误是分散的；但真实可用的状态路由器开发期和 2026 都低于固定 head_union。如果继续把状态分箱调得更细，很容易变成训练期记忆。

下一步不继续调市场状态 bin。更合理的检查是：候选族自身的近期已实现表现是否有动量，也就是在线择模。

## 27. online_alpha_router_v1

### 27.1 假设

如果 A 股短周期是动态趋势市场，那么不仅个股/行业可能有动量，Alpha 家族也可能有短期动量：最近几轮调仓表现好的候选族，下一轮也可能继续有效。本轮测试这个更贴近“市场自己告诉我们信谁”的路由方式。

为避免偷看，在线路由至少滞后一轮更新：某次调仓的 5 日结果只有在后续调仓日之后才进入已知历史。开发期第一次错误地把同区间全期结果作为初始 history 的版本已作废；以下只记录修正后的因果版本。

### 27.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/online-alpha-router-v1/
```

脚本：

```text
/tmp/quantx-research/online-alpha-router-v1/analyze_online_alpha_router.py
```

汇总文件：

```text
/tmp/quantx-research/online-alpha-router-v1/online_alpha_router_summary.json
sha256:ff5f7ce37396bac7ee1ef8dd2c0d3eee6cfc4baa428035a2be5e82cba4de1b85
```

诊断文件：

```text
/tmp/quantx-research/online-alpha-router-v1/online_alpha_router_dev_2021_2025_label_diagnostic.json
sha256:41c26e708fccc24c6002c6ceb6939cc7d4aa91b7c27dc5a0afbbaa43e5b8f0b7

/tmp/quantx-research/online-alpha-router-v1/online_alpha_router_val63_2026_label_diagnostic.json
sha256:b572debff84d3b302454cacf9a3d1ba31d480c99f7a160a7f7729ff20eada451

/tmp/quantx-research/online-alpha-router-v1/online_alpha_router_val20_2026_label_diagnostic.json
sha256:b0db80ba579d6e7cb67894c1d2eb25975d46329704fffde74cf133213b9093ad
```

路由规则：看过去已完成调仓的候选族 `mean_raw_return` 或 `mean_label`，窗口为 2/3/5 轮，另测 EWMA 和一个稳定性打分。

### 27.3 标签层结果

| 路由/候选 | 2021-2025 due5 label | 2026 val63 due5 label | 2026 val20 due5 label | 观察 |
| --- | ---: | ---: | ---: | --- |
| oracle | `+0.03178` | `+0.02265` | `+0.02204` | 事后空间仍很大 |
| 固定 `head_union` | `+0.01808` | `+0.00909` | `+0.01138` | 仍是最强可用固定候选 |
| online raw/label w3 | `+0.01746` | `+0.00856` | `+0.01004` | 三段都低于 head_union |
| online raw/label w2 | `+0.01644` | `+0.00904` | `+0.00536` | val63 接近但不稳定 |
| online raw/label ewma | `+0.01638` | `+0.00680` | `+0.00703` | 平滑后更弱 |
| online stability w5 | `+0.01561` | `+0.00496` | `+0.00899` | 防守式择模也没有优势 |

### 27.4 判定

`rejected`，不进入账户层。

在线择模没有证明候选族表现有足够强的短期延续性。它经常在 head_union、concept leader、industry leader、mech_static 之间切换，但切换后的平均标签没有超过固定 head_union。也就是说，已有候选族之间的“谁强”更多是事后噪声或不可由低阶历史表现捕捉的状态，而不是稳定的 Alpha 家族动量。

后续不继续在已有候选族之间做路由。下一轮应回到候选生成和标签目标本身，寻找更高信噪比的机制：例如事件/涨停后续扩散、行业/概念时间序列传播标签、或更接近 A 股短线资金行为的候选池生成，而不是继续重排已有基础 ML 候选。

## 28. event_propagation_candidates_v1

### 28.1 假设

前面的机制因子在基础 ML TopK 内有边际增益，但不能形成高收益主引擎。本轮反过来测试更激进的候选生成：不依赖基础 ML TopK，直接从全市场按涨停/近涨停、概念/行业扩散、量能突破、领涨和跟随确认生成候选。如果 A 股短周期事件本身有足够高的收益密度，这一步应该在标签层明显为正。

### 28.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/event-propagation-candidates-v1/
```

脚本：

```text
/tmp/quantx-research/event-propagation-candidates-v1/analyze_event_propagation_candidates.py
```

汇总文件：

```text
/tmp/quantx-research/event-propagation-candidates-v1/event_propagation_candidates_summary.json
sha256:82186080e3aa11c3384522ee08836e6c089b8de5f61615f577250f4215afa159
```

诊断文件：

```text
/tmp/quantx-research/event-propagation-candidates-v1/event_propagation_dev_2021_2025_label_diagnostic.json
sha256:e42cf2c87a938d03012440c89b8812d5c60531778cf40742b9f3f9bea3e2ab85

/tmp/quantx-research/event-propagation-candidates-v1/event_propagation_val63_2026_label_diagnostic.json
sha256:1e690e47a5d6b5d68d00f9366e3170bdfde67abd014f63236fab1cb45beb2141

/tmp/quantx-research/event-propagation-candidates-v1/event_propagation_val20_2026_label_diagnostic.json
sha256:1e690e47a5d6b5d68d00f9366e3170bdfde67abd014f63236fab1cb45beb2141
```

2026 val63/val20 文件相同是预期行为：本轮不训练模型、不使用训练边界，只依赖同一段 2026 行情和静态行业/概念快照。

### 28.3 标签层结果

| 方案 | TopK | 2021-2025 due5 label | 2026 due5 label | 观察 |
| --- | ---: | ---: | ---: | --- |
| `concept_low_chase` | Top30 | `-0.00924` | `-0.00413` | 开发期最好但仍明显负 |
| `concept_follow_confirmed` | Top30 | `-0.00967` | `-0.00375` | 直接买概念跟随仍追高 |
| `industry_follow_confirmed` | Top20 | `-0.01470` | `+0.00548` | 2026 有效，但开发期长期负 |
| `event_leader` | Top30 | `-0.01075` | `+0.00085` | 事件领涨不是稳健买点 |
| `volume_breakout` | Top30 | `-0.01151` | `+0.00149` | 量能突破同样不稳 |
| `limit_continuation` | Top20 | `-0.01720` | `+0.00073` | 涨停延续直接追买最弱 |

### 28.4 判定

`rejected`，不进入账户层。

这轮结论非常明确：全市场直接买强事件不是鲁棒高收益方向。2026 的 `industry_follow_confirmed` 有小正收益，但 2021-2025 开发期所有 tested TopK 都是负超额，说明它更像特定弱市阶段的局部现象，不是可穿越样本的收益主引擎。

这个负结果也解释了为什么前面的行业/概念机制只适合作为基础 ML 候选内的边际增强：基础 ML 可能已经过滤掉了大量纯追高陷阱；一旦脱离 ML 候选池，事件强度本身暴露的是追涨后的均值回撤。

下一步不继续买“当天强”。更合理的事件方向是 post-event setup：首板/近涨停后的 1-3 日回踩、二次放量确认、概念仍强但个股追涨压力下降，或者把事件作为条件标签训练模型去区分延续和失败，而不是直接买事件强度。

## 29. post_event_setup_v1

### 29.1 假设

第 28 轮证明“当天强事件”大多是追高陷阱。本轮测试更贴近短线资金节奏的 post-event setup：近期出现涨停/近涨停事件后，不在最极端追涨当天买，而是在 1-5 日内寻找回踩、二次放量确认、热概念但个股降温、二波量能等形态。

如果事件后的失败和延续可以被简单形态区分，这一轮应该至少在 2021-2025 开发期转为正超额，并且候选数量不能太少。

### 29.2 数据和产物

输出根目录：

```text
/tmp/quantx-research/post-event-setup-v1/
```

脚本：

```text
/tmp/quantx-research/post-event-setup-v1/analyze_post_event_setup.py
```

汇总文件：

```text
/tmp/quantx-research/post-event-setup-v1/post_event_setup_summary.json
sha256:f999a89ecaee61f2320230e579b16ff041e2052c2f85b26c457f6bf93ecabfcf
```

诊断文件：

```text
/tmp/quantx-research/post-event-setup-v1/post_event_setup_dev_2021_2025_label_diagnostic.json
sha256:4ebc7cd716b0a1da9a821dee4d4f13dc785d595059249ca612c40d42ec2e3330

/tmp/quantx-research/post-event-setup-v1/post_event_setup_val63_2026_label_diagnostic.json
sha256:86cdd2b89411879e415d57c85c51c84a012c22cf6f0bdbf032db78d01afd56c2

/tmp/quantx-research/post-event-setup-v1/post_event_setup_val20_2026_label_diagnostic.json
sha256:86cdd2b89411879e415d57c85c51c84a012c22cf6f0bdbf032db78d01afd56c2
```

2026 val63/val20 文件相同是预期行为：本轮没有训练模型，两个 forward 边界不影响同一段行情上的事件形态诊断。

### 29.3 标签层结果

重点看 `due5`，并同时检查平均候选数。

| 方案 | TopK | 2021-2025 due5 label | 2026 due5 label | 平均候选数 dev / 2026 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `post_near_limit_pullback` | Top10 | `-0.00211` | `+0.00605` | 125.34 / 139.42 | 候选充足，但开发期仍负 |
| `hot_concept_cool_stock` | Top15 | `-0.00231` | `+0.00454` | 17.96 / 19.38 | 2025 和 2026 有效，2022/2024 反向 |
| `second_confirm` | Top10 | `-0.00467` | `+0.01109` | 29.59 / 35.83 | 2026 最强，但开发期不穿越 |
| `volume_second_wave` | Top10 | `-0.00969` | `+0.00790` | 48.00 / 54.46 | 二波量能在 dev 明显反向 |
| `post_limit_pullback` | Top10 | `-0.00480` | `+0.00219` | 40.61 / 42.67 | 比直接涨停延续好，但仍不够 |

分年看，`post_near_limit_pullback` Top10 在 2021 为 `+0.00392`，2025 原始收益为正但超额仍弱；2022 为 `-0.00413`，2024 为 `-0.00842`。`second_confirm` 在 2026 很强，但 2022 为 `-0.01055`，2024 为 `-0.00572`。这说明 post-event setup 的有效性高度依赖市场阶段，不是稳定穿越样本的主 Alpha。

### 29.4 判定

`rejected`，不进入账户层。

这轮比第 28 轮明显好：事件后冷却/二次确认能减少一部分追高损失，并且在 2026 有正向信号。但它仍不满足鲁棒方向的要求：

1. 2021-2025 开发期最好的 due5 方案仍是负超额，正日比例也不到 50%。
2. 2022 和 2024 两个年份明显反向，说明手工 setup 会在弱趋势或高波动阶段持续买入失败事件。
3. 2026 的 `second_confirm` 很强，但这更像近期市场结构下的局部机会，不能覆盖五年几十倍收益目标。

下一步不继续手工调 post-event 阈值。更合理的方向是事件条件学习器：先生成“近期事件候选池”，再用模型学习哪些事件会延续、哪些会失败，把 2022/2024 的失败样本作为训练目标的一部分，而不是继续靠固定形态规则。

## 30. big_winner_attribution_v1

### 30.1 假设

前面几轮手工事件、概念扩散和 post-event setup 都说明一个问题：我们知道 A 股大收益常常来自主题、强趋势、波动和扩散，但固定规则很难区分“会延续的大涨股”和“事件后失败样本”。本轮因此换一个角度：不先定义买点，而是直接标记历史上未来 10/15/20 个交易日横截面涨幅前 5% 的大赢家，用 ML 做归因，观察哪些可见特征最能解释大赢家，并检验它能否自然转成周频 TopK。

本轮是诊断实验，不写入正式策略库，也不生成账户层预测文件。

### 30.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/big-winner-attribution-v1/
```

脚本：

```text
/tmp/quantx-research/big-winner-attribution-v1/analyze_big_winner_attribution.py
```

共同口径：全主板、2021-2025 年度 walk-forward 开发期、2026-01-05 至 2026-07-10 forward，模型为 `LightGBMClassifier`，正样本为每天未来目标周期收益前 5% 股票，负样本做平衡抽样。特征包括市场收益/离散度/宽度、个股 1/3/5/10/20/60 日强弱、波动率、成交量放大、前期涨停/近涨停、行业/概念强度和相对强弱。

重要边界：行业和概念成员仍来自 2026-06-25 静态快照，不是 point-in-time 成分，因此这些特征只能作为归因线索，不能直接作为可实盘证据。

汇总文件：

```text
/tmp/quantx-research/big-winner-attribution-v1/big_winner_attribution_summary.json
```

诊断文件和校验和：

| 目标 | 文件 | SHA256 |
| --- | --- | --- |
| 20 日 winner，2021-2025 | `big_winner_dev_2021_2025_label_diagnostic.json` | `sha256:6c3998d25c6bb4d04af77a19793ee11fad50e4df9a58edc0ed4b36d3e2ca8ae2` |
| 20 日 winner，2026 val63/val20 | `big_winner_val63_2026_label_diagnostic.json` / `big_winner_val20_2026_label_diagnostic.json` | `sha256:b9c89341764050b89c8439321d7294dd47bb1d47d9abd01b6be386c6e4f0c51b` |
| 15 日 winner，2021-2025 | `big_winner_h15_dev_2021_2025_label_diagnostic.json` | `sha256:6407f76641ba53e13ed887583924b860b71665a64eabb9c69d5dbce0e56b85df` |
| 15 日 winner，2026 | `big_winner_h15_val63_2026_label_diagnostic.json` | `sha256:30f3f65dc3fa2b9a1bba63dbd6cb366d2aab7bfacd0b689fe4639c250c4cfa99` |
| 10 日 winner，2021-2025 | `big_winner_h10_dev_2021_2025_label_diagnostic.json` | `sha256:cea363ff0b58975e4ee9195082ffe483c37ea2a737b4c790bbb362d240094679` |
| 10 日 winner，2026 | `big_winner_h10_val63_2026_label_diagnostic.json` | `sha256:aef4756828255ba9ab2784314719f81242f94b976fea4e8c47187683d1b8b8d9` |

### 30.3 标签层结果

重点看 `due5` 周频采样，因为用户目标是平均持仓约一周，并且 Top15/20 更接近平均持仓大于 5 的约束。

| 目标 | 区间 | due5 Top10 label5 | due5 Top15 label5 | due5 Top20 label5 | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| 20 日 winner | 2021-2025 | `+0.005321` | `+0.001494` | `-0.002241` | 头部有用，但扩到 Top20 转负 |
| 20 日 winner | 2026 | `-0.010257` | `+0.000086` | `+0.004035` | 2026 Top20 正，但 Top10 和开发期不一致 |
| 15 日 winner | 2021-2025 | `+0.002287` | `-0.004317` | `-0.005716` | 缩短目标后开发期更差 |
| 15 日 winner | 2026 | `+0.005011` | `+0.002852` | `+0.001387` | 2026 看起来有效，但开发期不过 |
| 10 日 winner | 2021-2025 | `-0.003916` | `-0.008752` | `-0.012011` | 一周附近目标在开发期明确失败 |
| 10 日 winner | 2026 | `+0.005241` | `+0.002146` | `+0.005151` | 2026 有正信号，但不能覆盖历史失败 |

日频 `all` 口径比周频更好。例如 20 日 winner 在开发期 `all::top10` 的 5 日超额 label 为 `+0.009365`，2026 `all::top20` 为 `+0.005587`。但只要按 5 日调仓节奏看，Top15/20 的开发期质量就明显不足，说明它还不是自然的一周持仓组合引擎。

### 30.4 归因结果

三个目标和两个区间的特征重要性高度一致，最靠前的是：

1. `vol60_rank`。
2. `market_ret60_disp`、`market_ret60_median`、`market_ret60_p80`。
3. `ml_ret60_rank`。
4. `high60_dist_rank`、`low60_dist_rank`。
5. `breadth20`、`breadth60`。
6. `vol20_rank`、`ml_vol_ratio20_rank`。

这说明历史大涨股确实可以被学习，但它们更像由 60 日左右的市场结构、横截面离散度、宽度、波动率和中期强弱共同决定，而不是单纯由当天涨停、概念热度或一周内事件形态决定。行业/概念事件特征有一定重要性，但不是主导项。

### 30.5 判定

`diagnostic_only`，不进入账户层，不合代码。

对用户的问题“能不能通过分析历史上大幅上涨股票，用 ML 归因”来说，答案是：可以，而且归因结果很清楚。大涨股的主结构集中在中期波动、市场离散度、60 日强弱/位置和市场宽度上。

但它还不是可晋级策略：

1. 20 日 winner 目标在开发期只对 Top10 头部有效，Top15 很弱，Top20 转负，不满足“自然扩到持仓大于 5 后仍高收益”的要求。
2. 10/15 日 winner 目标虽然更贴近一周持仓，但 2021-2025 开发期 due5 Top15/20 明确为负；不能因为 2026 半年转正就采用。
3. 当前结果只有标签层，没有账户层收益、换手、成本、平均持仓数和年度路径证明。
4. 静态行业/概念快照带来信息边界问题，不能作为可实盘特征证据。

下一步不应该把这个模型直接做成策略，而应把它作为方向约束：后续候选生成器应围绕“中期结构 + 横截面离散 + 波动/宽度 + 事件延续失败分类”设计，并且必须先在 2021-2025 的周频 Top15/20 标签层转正，再进入账户回测。

## 31. event_continuation_learner_v1

### 31.1 假设

第 28-29 轮说明直接买事件强度和手工 post-event setup 都会在 2021-2025 买入大量失败事件；第 30 轮又说明历史大涨股的可学习结构更偏中期市场状态、波动和宽度。本轮做一个直接反事实：保留“近期有涨停/近涨停”的事件候选池，但不再手写回踩/确认规则，而是用 ML 学习候选内哪些事件会延续、哪些会失败。

若事件候选池本身仍有足够金矿，那么学习器应该至少把 2021-2025 的周频 `due5` Top15/20 拉到正超额，并且不能只在 2026 单段有效。

### 31.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/event-continuation-learner-v1/
```

脚本：

```text
/tmp/quantx-research/event-continuation-learner-v1/analyze_event_continuation_learner.py
```

候选池：过去 5 个交易日至少一次近涨停，信号日没有极端涨跌，20 日趋势没有严重破位，并过滤 5 日内涨停次数过多的过热样本。平均候选数约 218 只/日，2026 约 281 只/日。

模型：

1. `reg`：LightGBMRegressor，预测未来 5 日 open-to-open 横截面超额收益。
2. `cls`：LightGBMClassifier，预测候选池内部未来 5 日超额收益前 20%。
3. `blend`：`reg` 和 `cls` 的同日 rank blend。

特征包括市场收益/离散度/宽度、个股 1/3/5/10/20/60 日强弱、波动率、成交量、价格位置、近期事件计数、事件后失败计数、行业/概念事件强度和相对强弱。行业和概念成员仍为静态快照，因此只作诊断。

诊断文件：

```text
/tmp/quantx-research/event-continuation-learner-v1/event_continuation_dev_2021_2025_label_diagnostic.json
sha256:28c67fc97b8d28d960f3f3aef51013618af7eb3292d21f10eea2398d84505bf2

/tmp/quantx-research/event-continuation-learner-v1/event_continuation_val63_2026_label_diagnostic.json
sha256:0a00519250683218c759ffd616200053ef3b66d63bcf8b1aab577588730df3b2

/tmp/quantx-research/event-continuation-learner-v1/event_continuation_summary.json
```

### 31.3 标签层结果

重点看周频 `due5`：

| 模型 | TopK | 2021-2025 due5 label5 | 2026 due5 label5 | 候选内 Top20% 命中率 dev / 2026 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `reg` | Top10 | `-0.004943` | `+0.003086` | 20.00% / 19.58% | 开发期每年为负 |
| `reg` | Top15 | `-0.005814` | `+0.001750` | 20.03% / 18.33% | 没有学出正超额 |
| `reg` | Top20 | `-0.005379` | `+0.005669` | 19.67% / 20.42% | 2026 转正但开发期失败 |
| `cls` | Top10 | `-0.009595` | `+0.026473` | 24.10% / 31.67% | 2026 极强，历史极差 |
| `cls` | Top15 | `-0.009712` | `+0.021205` | 24.07% / 30.56% | 近期现象不能外推 |
| `cls` | Top20 | `-0.009253` | `+0.021382` | 24.13% / 29.58% | 候选内命中高但全市场超额负 |
| `blend` | Top20 | `-0.006875` | `+0.005058` | 21.26% / 22.92% | 融合不能修复 |

`reg` 的 Top20 分年：2022 `-0.009398`、2023 `-0.001758`、2024 `-0.007742`、2025 `-0.002561`。`cls` 的 Top20 分年：2022 `-0.011577`、2023 `-0.008870`、2024 `-0.006662`、2025 `-0.009915`。这不是某一年拖累，而是开发期每年都不成立。

### 31.4 反事实分析

这轮最值得注意的是：`cls` 确实提高了候选池内部 top20% 命中率，开发期 due5 Top20 命中约 24.13%，高于随机 20%；2026 更高到 29.58%。但它的 2021-2025 全市场超额收益仍显著为负。这说明问题不是“模型完全不会学”，而是候选池本身处在负收益土壤：事件后候选即便在池内相对靠前，拿到全市场横截面里仍然偏弱。

可能原因：

1. **事件候选池带有追高后的均值回撤暴露**：即使过滤信号日极端涨幅，过去 5 日近涨停本身已经把股票推到拥挤位置。
2. **2026 的事件风格不能外推**：2026 cls Top20 很强，但 2022/2023/2024/2025 同口径全部负，说明近期可能是事件延续更强的局部市场。
3. **候选内分类目标和全市场收益目标错位**：候选内 top20% 不等于全市场 top20%；池内相对赢家仍可能是全市场弱者。
4. **行业/概念静态快照只能解释，不能补足 PIT 轮动**：即使模型重要性仍集中于市场离散度、宽度、60 日强弱和波动，静态概念不能稳定定义历史真实主题扩散。

### 31.5 判定

`rejected`，不进入账户层，不合代码。

这轮把“事件条件学习器”也否掉了。它证明 2026 的强事件延续不能覆盖 2021-2025 的系统性失败；继续在涨停/近涨停事件候选池里调模型，大概率是在负收益候选土壤上找局部相对强者。

下一步应脱离事件候选池，转向更底层的横截面因子土壤筛选：先用标签层比较经典 A 股短周期因子族在 `due5` Top15/20 上的开发期和 2026 表现，特别是市场风险状态、横截面离散度、行业/概念轮动、过去一段时间的趋势/反转/波动/流动性组合，而不是继续围绕涨停事件做候选生成。

## 32. factor_soil_scan_v1

### 32.1 假设

连续多轮事件方向失败后，本轮先不训练复杂模型，也不预设“涨停/概念事件”是收益主线，而是做一个低假设的横截面因子土壤扫描：在同一个未来 5 日超额标签和 `due5` 周频采样下，比较一批经典 A 股短周期因子族，看哪些方向本身更容易在 Top15/20 里出正超额。

本轮目标不是直接找到账户策略，而是回答：下一轮 ML 应该在哪类因子土壤里学习，避免在天然负收益候选池里继续调模型。

### 32.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/factor-soil-scan-v1/
```

脚本：

```text
/tmp/quantx-research/factor-soil-scan-v1/analyze_factor_soil_scan.py
```

共同口径：全主板、T 日因子排序、T+1 open 入场、5 个交易日后 open 退出，标签为个股收益减同日全市场均值。每个因子每天全市场排序取 Top10/15/20/30，并按 `all` 和 `due5` 采样评价。

扫描因子族：短/中期动量、短期反转、低波/高波、成交量放大、接近高点、从高点回踩、中期趋势回踩、行业/概念事件强度、行业/概念相对强弱、若干中期结构混合因子。

市场状态拆分：20 日横截面离散度高/低、宽度强/弱、20 日市场收益正/负，以及市场收益与离散度交互。行业和概念仍是静态快照，相关结论只能作诊断。

诊断文件：

```text
/tmp/quantx-research/factor-soil-scan-v1/factor_soil_dev_2021_2025_label_diagnostic.json
sha256:a3bed95fa6ec033a7d78472749570a56a37b7b276771a94783a5eac9f7a27ff8

/tmp/quantx-research/factor-soil-scan-v1/factor_soil_val63_2026_label_diagnostic.json
sha256:a1f7fd6a29268c75422a820d01af9d1a422d16f7888474d53105f182eced823f

/tmp/quantx-research/factor-soil-scan-v1/factor_soil_summary.json
```

### 32.3 开发期结果

只看全时段 `due5::all::Top20`，最好的裸因子也只有弱正超额：

| 因子 | 2021-2025 due5 Top20 label5 | raw5 | 正超额比例 | 分年观察 |
| --- | ---: | ---: | ---: | --- |
| `low_vol20` | `+0.001703` | `+0.004791` | 48.35% | 2021/2022/2024 正，2025 `-0.005388` |
| `low_vol_trend` | `+0.001427` | `+0.004514` | 50.83% | 2022 小负，2025 近零 |
| `near_high_quality` | `+0.001427` | `+0.004514` | 50.83% | 与 `low_vol_trend` 同构 |
| `low_vol60` | `+0.000746` | `+0.003834` | 45.04% | 弱正但正日比例不足 |
| `concept_burst` | `-0.001926` | `+0.001162` | 46.28% | 2022/2024/2025 负 |
| `mom60` | `-0.011884` | `-0.008797` | 37.19% | 2021-2025 每年均负 |

条件拆分里，最强的几个点是：

| 条件 + 因子 | due5 Top20 label5 | 样本数 | 观察 |
| --- | ---: | ---: | --- |
| `strong_market_high_disp + concept_burst` | `+0.003545` | 50 | 2025 很强，但 2022/2024 仍负 |
| `low_disp20 + low_vol60` | `+0.003460` | 97 | 除 2021 外较稳，但收益仍小 |
| `positive_market20 + low_vol20` | `+0.002980` | 114 | 2025 转负 |
| `strong_breadth20 + concept_burst` | `+0.001972` | 100 | 2025 强、2022/2024 弱 |

### 32.4 2026 前向结果

2026 的排序结构和开发期明显不同：

| 因子 | 2026 due5 all Top20 label5 | raw5 | 正超额比例 | 观察 |
| --- | ---: | ---: | ---: | --- |
| `pullback_from_high20` | `+0.017992` | `+0.015422` | 75.00% | 2026 最强 |
| `trend_pullback` | `+0.013119` | `+0.010549` | 58.33% | 趋势回踩有效 |
| `mom60` | `+0.011561` | `+0.008992` | 58.33% | 与开发期强烈反号 |
| `high_vol_leader` | `+0.009313` | `+0.006744` | 66.67% | 高波领涨有效 |
| `low_vol20` | `-0.004749` | `-0.007318` | 45.83% | 开发期最好因子在 2026 失效 |
| `low_vol_trend` | `-0.001332` | `-0.003901` | 45.83% | 质量低波底座不适应 2026 |

注意 2026 只有 24 个 due5 调仓点，部分条件样本更少。因此不能把 2026 的趋势/高波强势直接当作新规则，但它已经足够说明单一静态因子不是答案。

### 32.5 反事实分析

这轮给出三个关键反事实：

1. 如果只看 2021-2025，会以为低波/近高质量是最稳土壤；但 2026 它转负。
2. 如果只看 2026，会以为趋势/回踩/高波领涨是核心金矿；但 2021-2025 的 `mom60` 每年都负，`pullback_from_high20` 也开发期全时段为负。
3. 行业/概念 burst 不是单调收益来源：它在强市场高离散或强宽度状态下有局部正收益，但 2022/2024 反向，仍不能作为静态候选规则。

因此，裸因子本身不是五年几十倍收益引擎；真正的问题变成：能否用因果可见的信息识别“现在该用低波质量、趋势回踩、高波领涨，还是概念强度”。这比继续发明单因子更接近用户强调的动态市场和横截面因子要求。

### 32.6 判定

`diagnostic_only`，不进入账户层，不合代码。

本轮没有找到可直接晋级的高收益策略，但明确了下一步方向：训练或验证一个因子风格路由器。该路由器必须只用调仓日前可见的市场状态和近期因子表现，不能事后选择 2026 强的趋势因子；并且必须先在 2021-2025 的 `due5` Top15/20 标签层证明比固定因子和固定基础 ML 更稳，再进入账户回测。

## 33. factor_style_router_v1

### 33.1 假设

第 32 轮显示单一裸因子不是答案：2021-2025 更偏低波/近高质量，2026 则切换到趋势回踩、60 日动量和高波领涨。一个自然反事实是：如果把这些因子族当作“专家”，能不能用调仓日前可见的市场状态或近期已实现表现，因果地选出当前应使用的专家。

本轮只做 `due5` 周频标签层路由，不进入账户层。若路由器连固定低波专家都打不过，就说明“离散地择因子”太粗，不能作为主方向。

### 33.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/factor-style-router-v1/
```

脚本：

```text
/tmp/quantx-research/factor-style-router-v1/analyze_factor_style_router.py
```

候选专家来自第 32 轮：`low_vol20`、`low_vol_trend`、`near_high_quality`、`pullback_from_high20`、`trend_pullback`、`mom60`、`high_vol_leader`、`medium_structure`、`concept_burst`、`group_relative_leader`。

路由器包括：

1. 固定专家。
2. 训练期最佳固定专家。
3. 最近 3/6/12 个已完成调仓周期的专家实现收益动量。
4. 最近 3/6/12 个周期只计正收益的专家表现。
5. 按市场 20 日收益、横截面离散度、宽度及其组合分桶选择训练期最佳专家。
6. `oracle` 只作事后上限，不可交易。

walk-forward 开发期按年份滚动：预测某年时只用此前年份的专家实现收益和状态统计。2026 forward 使用 2021-2025 历史，并在每个已完成调仓周期后在线更新近期表现。

诊断文件：

```text
/tmp/quantx-research/factor-style-router-v1/factor_style_router_dev_2021_2025_label_diagnostic.json
sha256:8e20f0c2e5212ee49d5b3b9a693a33f43ff5736604a6eb5b27479c2b54bdb18e

/tmp/quantx-research/factor-style-router-v1/factor_style_router_val63_2026_label_diagnostic.json
sha256:acff844d8ad42c6b35bdaf3e2c2608902850aef5391cfc21c778614decdb8ae8

/tmp/quantx-research/factor-style-router-v1/factor_style_router_summary.json
```

### 33.3 结果

重点看 Top20，因为它更接近平均持仓大于 5 的目标。

| 路由器 | 2021-2025 label5 | 2026 label5 | 观察 |
| --- | ---: | ---: | --- |
| `oracle` | `+0.036613` | `+0.063513` | 事后切换空间巨大，但不可交易 |
| `fixed_low_vol20` | `+0.001177` | `-0.004749` | 开发期最佳因果基线，2026 失效 |
| `fixed_low_vol_trend` | `+0.000823` | `-0.001332` | 与低波质量同类，收益很弱 |
| `state_state_trend` | `+0.000768` | `+0.000002` | 接近固定低波，但没有真正识别 2026 趋势风格 |
| `recent_w6` | `-0.002852` | `+0.013559` | 2026 能转向动量/回踩，但开发期失败 |
| `recent_w3` | `-0.005090` | `+0.013254` | 更短近期表现追随更不稳 |
| `state_state_combo` | `-0.003309` | `-0.002218` | 状态组合过拟合或样本不足 |
| `fixed_pullback_from_high20` | `-0.005923` | `+0.017992` | 2026 强风格，开发期明确失败 |
| `fixed_mom60` | `-0.013576` | `+0.011561` | 强烈风格反转，不可静态采用 |

开发期 `oracle` 分年均很高：2022 `+0.036058`、2023 `+0.030710`、2024 `+0.039682`、2025 `+0.040125`。这说明因子专家之间确实存在巨大事后互补；但所有可因果执行的路由都没有把这个上限转化出来。

### 33.4 反事实分析

这轮给出一个很有价值的负结论：收益空间不是没有，问题在于“怎么识别当前该用什么风格”。简单市场状态分桶太粗，近期表现路由又太容易追噪声。

可能原因：

1. **风格切换发生在股票级，而不是日级**：同一天内可能既有低波质量股有效，也有趋势回踩股有效；硬选一个专家会丢失横截面内部结构。
2. **近期专家赢家延续性弱**：2026 近期动量路由有效，但 2021-2025 的 w3/w6/w12 都为负，说明专家级收益动量不稳定。
3. **状态分桶样本太少**：组合状态把 193 个开发期周频样本切得过碎，容易把 2022/2024 的反例学错。
4. **静态因子专家过于粗糙**：专家本身是手写组合分数，可能无法表达“低波 + 趋势 + 概念强度”在不同股票上的非线性交互。

### 33.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“离散专家路由”而不是“状态-因子交互”。下一步应把因子风格作为股票级特征交给模型学习，让模型在同一天内对不同股票学习不同交互，而不是每 5 天只选择一个全局专家。最低要求仍然是先在 2021-2025 `due5` Top15/20 标签层超过固定低波和基础 ML，再进入 2026 和账户层。

## 34. stock_state_factor_interaction_v1

### 34.1 假设

第 33 轮显示因子专家之间存在很高的事后切换空间，但按调仓日硬选一个专家无法因果识别。一个更合理的反事实是：风格切换可能发生在股票级，同一天既可能有低波质量股有效，也可能有趋势回踩股有效。因此本轮不再做全局专家路由，而是把第 32 轮的因子土壤、市场状态和显式交互项放入股票级 LightGBM，让模型直接学习未来一周超额收益。

### 34.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/stock-state-factor-interaction-v1/
```

脚本：

```text
/tmp/quantx-research/stock-state-factor-interaction-v1/analyze_stock_state_factor_interaction.py
```

共同口径：只取 `due5` 周频调仓日，全主板股票级面板，T 日特征，T+1 open 入场，5 个交易日后 open 退出，标签为个股收益减同日全市场均值。

模型：

1. `reg`：LightGBMRegressor，预测 5 日横截面超额收益。
2. `cls`：LightGBMClassifier，预测同日未来 5 日超额收益前 5% 股票。
3. `blend`：`reg` 和 `cls` 分数同日 rank blend。

特征包括低波、趋势回踩、60 日动量、高波领涨、概念/行业强度、组内相对强弱、中期结构、市场收益/离散度/宽度，以及 `mom60_x_high_disp`、`pullback_x_pos_mkt`、`lowvol_x_low_disp`、`concept_x_strong_breadth` 等显式交互。行业和概念仍为静态快照，只作诊断。

诊断文件：

```text
/tmp/quantx-research/stock-state-factor-interaction-v1/stock_state_factor_interaction_dev_2021_2025_label_diagnostic.json
sha256:89cb498666b45de3fd2cb427f4ccba2dd55e33e5c4d18364b3859cd05270b9a2

/tmp/quantx-research/stock-state-factor-interaction-v1/stock_state_factor_interaction_val63_2026_label_diagnostic.json
sha256:e856909cec6083c3e0104bd9604fd98fbc67379b1771c859ecedbb02ef12764f

/tmp/quantx-research/stock-state-factor-interaction-v1/stock_state_factor_interaction_summary.json
```

### 34.3 结果

重点看 Top15/20：

| 模型 | 2021-2025 Top15 label5 | 2021-2025 Top20 label5 | 2026 Top20 label5 | 观察 |
| --- | ---: | ---: | ---: | --- |
| `reg` | `-0.002992` | `-0.001958` | `-0.008165` | 开发期和 2026 都不如固定低波 |
| `blend` | `-0.003888` | `-0.003391` | `-0.000977` | 融合无法修复 |
| `cls` | `-0.014224` | `-0.016915` | `+0.002929` | 2026 小正，但开发期显著失败 |

开发期分年看，`reg` Top20 在 2022 为 `+0.003094`，但 2023 `-0.000457`、2024 `-0.004002`、2025 `-0.006501`。`cls` Top20 在 2022-2025 全部为负，分别为 `-0.018435`、`-0.022955`、`-0.016511`、`-0.009634`。

特征重要性显示模型确实在使用市场离散度、市场 60 日收益、宽度、60 日动量、高波领涨、低波等变量；但这些变量没有转化为可用的 Top15/20 排序。

### 34.4 反事实分析

本轮最重要的负结论是：把状态和因子交给普通点预测模型并不能自动解决风格切换。更值得警惕的是 `cls`：它把 Top20 内同日 top5% winner 命中率从随机 5% 提到开发期约 16.89%，2026 约 20.63%，但开发期 Top20 超额收益反而严重为负。这和第 31 轮事件延续学习器、第 30 轮大涨股归因出现了同一种错位：极端赢家分类会偏向高噪声、高弹性、不可稳定等权持有的股票。

可能原因：

1. **点预测/极端分类和 TopK 等权收益错位**：抓极少数大赢家会同时引入更多失败样本，Top15/20 平均收益被拖累。
2. **训练样本粒度太粗**：周频 due5 面板减少了日样本数，树模型容易学习年度风格而非稳健结构。
3. **因子特征仍是手写二阶摘要**：它们解释市场状态，但没有足够表达真实概念轮动、资金拥挤和交易约束。
4. **静态行业/概念快照限制了主题表达**：真正的 A 股概念轮动需要 point-in-time 主题成员和热度，不是静态板块标签。

### 34.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决普通股票级状态-因子点预测模型。下一步不继续调 LightGBM 参数或 top5 分类阈值，而应改变优化目标：直接学习同日排序分位或 TopK 排序，例如按日 quintile/multiclass、pairwise/listwise ranking，且评价必须以 `due5` Top15/20 标签为主。另一个并行方向是寻找更强的外部候选源或 point-in-time 概念热度数据，因为当前静态因子土壤本身太薄。

## 35. quintile_objective_v1

### 35.1 假设

第 34 轮证明 top5 winner 分类虽然能提高极端赢家命中率，但会伤害 Top15/20 等权收益。一个更贴近目标的训练方式，是让模型学习同日未来 5 日超额收益的整体分位，而不是只追最极端的 5% 股票。本轮因此复用第 34 轮的股票级状态-因子特征，但把标签改成同日五分位分类，再用预测期望分位选择 TopK。

### 35.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/quintile-objective-v1/
```

脚本：

```text
/tmp/quantx-research/quintile-objective-v1/analyze_quintile_objective.py
```

模型：LightGBM 五分类，标签为每个 `due5` 调仓日未来 5 日超额收益同日五分位 `0..4`。评分方式：

1. `ev`：预测分位期望值。
2. `topq`：最高分位概率。
3. `blend`：`ev` 和 `topq` 的同日 rank blend。

诊断文件：

```text
/tmp/quantx-research/quintile-objective-v1/quintile_objective_dev_2021_2025_label_diagnostic.json
sha256:797b9c4886bbf0538ed0400ff3802c888508faf4dfbedbb4da7230be8c5967c6

/tmp/quantx-research/quintile-objective-v1/quintile_objective_val63_2026_label_diagnostic.json
sha256:1c3cd21d470bc502f03c76659ee4927551fdf50c0b04573b74955c55dce4f94d

/tmp/quantx-research/quintile-objective-v1/quintile_objective_summary.json
```

### 35.3 结果

| 评分 | 2021-2025 Top15 label5 | 2021-2025 Top20 label5 | 2026 Top15 label5 | 2026 Top20 label5 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `ev` | `+0.001048` | `+0.002187` | `-0.000663` | `+0.000995` | Top20 开发期逐年正，2026 小正 |
| `blend` | `+0.001310` | `+0.002021` | `-0.003891` | `-0.004966` | 开发期可用但 2026 失效 |
| `topq` | `-0.002146` | `-0.001879` | `-0.004438` | `-0.003432` | 最高分位概率仍追噪声 |

`ev::top20` 分年：2022 `+0.000441`、2023 `+0.002063`、2024 `+0.005293`、2025 `+0.000956`，是最近几轮里少见的开发期每年正。2026 `ev::top20` 也保持小正 `+0.000995`，但 raw5 为 `-0.001574`，且 Top15 为负。

### 35.4 反事实分析

本轮说明目标函数确实重要：从 top5 winner 分类切换到同日分位期望后，开发期 Top20 从第 34 轮的负超额修复为逐年小正，并且 2026 没有完全失效。与此同时，`topq` 仍然为负，进一步证明“只追最高分位概率”会回到高噪声极端赢家问题。

但这仍然不是高收益方向：

1. 收益厚度太薄，开发期 Top20 只有每 5 日约 `+0.22%` 超额。
2. 2026 Top20 虽为小正，但原始收益为负，说明它更像弱防守排序，不是强 Alpha。
3. Top15 在 2026 转负，说明头部稳定性不足。
4. 这批特征仍来自静态行业/概念和手工因子土壤，信息含量有限。

### 35.5 判定

`diagnostic_only`，暂不进入账户层，不合代码。

本轮保留一个重要方法论结论：若后续有更强候选池或更真实的概念/行业热度数据，应优先使用同日分位/排序目标，而不是 top5 winner 分类或普通点预测。下一步把这个目标函数放到更肥的基础 ML Top100/200 候选池中做二阶段排序，检验它是否能在已有 ML 候选质量上形成更厚的 Top15/20 边际收益。

## 36. candidate_quintile_meta_ranker_v1

### 36.1 假设

第 35 轮说明同日五分位目标比 top5 winner 分类更贴近 Top15/20 等权收益，但全市场手写因子土壤太薄。一个自然反事实是：不让五分位模型从全市场几千只股票里直接找股票，而是只在基础 5 日 ML 已经筛出的 Top100/Top200 候选池里做二阶段排序。若基础 ML 负责生成较肥候选池，五分位模型负责校准候选池内的短周期收益分布，可能比普通点预测和 LambdaRank 更稳。

### 36.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/candidate-quintile-meta-ranker-v1/
```

脚本：

```text
/tmp/quantx-research/candidate-quintile-meta-ranker-v1/analyze_candidate_quintile_meta_ranker.py
```

输入预测：

```text
/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json
/tmp/quantx-research/adaptive-training-protocol-v1/forward-2026-5d-val63/mainboard_lgbm_base_5d_2010_2026_forward_val63_seed7-c52c47f834608a3c/oos_predictions.json
```

预测文件较大，实验脚本使用 `ijson` 对 `records.item` 做流式读取，只保留每个调仓日基础 ML Top200。特征为 `base_ml_score/base_ml_rank_pct` 加第 34-35 轮的低波、动量、近高、趋势回踩、行业/概念 burst、市场宽度和市场离散度等状态-因子交互特征。标签是候选池内同日未来 5 日超额收益五分位 `0..4`。

评分方式：

1. `base`：保留基础 5 日 ML 原始排序。
2. `ev`：五分位预测期望。
3. `topq`：最高五分位概率。
4. `blend`：`0.65 * ev_rank + 0.35 * base_rank`。

诊断文件：

```text
/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_dev_2021_2025_label_diagnostic.json
sha256:40bea782ab5b905a58af7164eb45260a3e683816fe2c73b883df40625982006c

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_dev_2021_2025_min8000_label_diagnostic.json
sha256:a11b6b4d82b580e69f78515fcdafa490c1dce7bfdaea18020d0265aea15aef51

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_val63_2026_label_diagnostic.json
sha256:f59ef8f93a54e643d869f4cacae1b3457108fcb60fea426aaa000ddd8366cfb7

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_summary.json
sha256:ad3e16bbc03e4a553c36ce6c9b68ee165200b41c7dd8fc6a7f8e4d96715b9333
```

### 36.3 结果

严格补测把 walk-forward 最小训练行数从 `10000` 降到 `8000`，使 2022 也参与 OOS。这个改动不是调收益参数，而是避免 2021 单年训练样本 `9800` 行被防御性门槛跳过。

| 评分 | 2022-2025 Top15 label5 | 2022-2025 Top20 label5 | 2026 Top15 label5 | 2026 Top20 label5 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `pool100::blend` | `+0.011641` | `+0.010361` | `+0.005218` | `+0.003549` | Top15/20 开发期全正，2026 保持正 |
| `pool200::blend` | `+0.011631` | `+0.010344` | `+0.005218` | `+0.003507` | 与 pool100 接近，说明不是池大小偶然 |
| `pool200::topq` | `+0.010139` | `+0.010062` | `+0.012521` | `+0.009706` | 2026 最强，但开发期只小幅高于 base Top20 |
| `pool200::base` | `+0.010679` | `+0.009717` | `+0.001729` | `+0.005062` | 基础 ML 候选池本身仍有较强土壤 |

含 2022 补测的分年结果：

| 评分 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: |
| `pool100::blend::top15` | `+0.019423` | `+0.007243` | `+0.012546` | `+0.007444` |
| `pool200::blend::top20` | `+0.015970` | `+0.007550` | `+0.010590` | `+0.007326` |
| `pool200::topq::top20` | `+0.012710` | `+0.005984` | `+0.014824` | `+0.006815` |
| `pool200::base::top20` | `+0.017456` | `+0.008658` | `+0.008316` | `+0.004459` |

第一个 dev 版本在默认 `min_train_rows=10000` 下从 2023 开始评估，`pool200::topq::top20` 为 `+0.009185`，基础 `pool200::base::top20` 为 `+0.007155`。补入 2022 后，`pool200::blend::top20` 为 `+0.010344`，基础 Top20 为 `+0.009717`，仍有边际改善，但改善幅度变小。

特征重要性前列稳定包含 `low_vol20`、`high_vol_leader`、`base_ml_rank_pct`、`base_ml_score`、`group_relative_leader`、`medium_structure`、`mom60`、`anti_chase_quality`、`pullback_from_high20`、`industry_burst`。这说明模型并非只复刻基础 ML 分数，而是在基础候选质量、横截面风格、行业/概念扩散和市场状态之间做二阶校准。

### 36.4 反事实分析

这轮和第 35 轮形成了清晰对照：同样的五分位目标，在全市场手写因子上只有每 5 日约 `+0.22%` 的薄边际；放进基础 ML Top100/200 候选池后，Top15/20 标签提升到每 5 日约 `+1.0%` 附近，且 2022-2025 和 2026 都为正。主要原因更可能是候选生成器贡献了“收益土壤”，五分位模型只负责在较好的土壤里减少目标错配。

账户层补证进一步验证了这个判断。用当前仓库支持的标准 `decision_pipeline`，T 日收盘信号、T+1 开盘成交、标准成本、Top20 等权满仓，补证如下：

```text
/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_blend_pool200_top20_dev_2022_2025_account_diagnostic.json
sha256:932df3c18eeb5427dd94efbcfa1d6ac9167213bc094ca797021d4d84d93dfaeb

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_base_pool200_top20_dev_2022_2025_account_diagnostic.json
sha256:80727d4932a2bc213c80f478d89635a87a8413e3ed5b1fb385e25b4aa9386ffa
```

| 账户层口径 | 区间 | 累计收益 | 年化 | 最大回撤 | 平均持仓 | 平均持仓天数 | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pool200::blend::top20` | 2022-2025 | `+406.39%` | `+50.26%` | `-32.97%` | `19.57` | `8.79` | 收益略高，但 2024 年初回撤更深 |
| `pool200::base::top20` | 2022-2025 | `+374.42%` | `+47.82%` | `-31.39%` | `19.48` | `9.62` | 基础排序已提供主要收益土壤 |

`blend` 逐年账户收益：2022 `+51.12%`、2023 `+33.63%`、2024 `+40.90%`、2025 `+76.13%`。但最大回撤发生在 2024-01-04 至 2024-02-07，从 `207.19万` 跌到 `138.87万`，回撤 `-32.97%`；最差 20 日滚动收益为 `-31.60%`。`base` 的最大回撤也发生在 2023-11-23 至 2024-02-05，回撤 `-31.39%`。因此二阶段模型没有解决真正的主风险。

账户层反事实说明：

1. T+1 开盘、标准成本、换手和持仓数进入账户层后，二阶段排序确有小幅增益，但不是数量级变化。
2. 开发期相对基础排序的边际并不巨大，尤其补入 2022 后 `topq` Top20 只略高于 `base`。
3. 2026 标签层 `topq` 很强，但 `blend` 只中等，说明 2026 风格可能更偏最高分位概率，仍有状态依赖。
4. 它更像二阶段排序增强器，不是新的高收益候选生成器；距离五年几十倍至 100 倍目标仍远。

### 36.5 判定

`diagnostic_only`，不进入正式策略库，不合代码。

本轮保留的方法论结论是：基础 ML 候选池内的同日分位目标可以做小幅排序校准，但它不能解决短周期强势股在市场风险段集体失效的问题。下一步不继续调 `blend/topq` 或账户权重，而应专门研究 2024 年初这种系统性失效是否可由市场宽度、横截面离散度、下跌扩散、行业/概念拥挤和候选池内部脆弱性提前识别。

## 37. strong_stock_failure_risk_v1

### 37.1 假设

第 36 轮的账户层结果说明，基础 ML 候选池和五分位二阶段排序都能在正常年份产生收益，但 2024 年初和 2026 中期会出现短周期强势股集体失效。若这种失效来自可观测的市场风险和候选池脆弱性，那么在信号日前应该能看到市场宽度、候选池下跌扩散、选中组合近期回撤、量能衰减或横截面波动等风险状态。

本轮不直接做风险 gate，也不调整仓位或权重，只做因果诊断：用调仓日前已经可见的市场/候选/选中组合状态解释未来 5 日账户收益，判断风险是否足够清晰到值得进入下一轮模型化。

### 37.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/strong-stock-failure-risk-v1/
```

脚本：

```text
/tmp/quantx-research/strong-stock-failure-risk-v1/analyze_strong_stock_failure_risk.py
sha256:b3d28bffc05e3f6afbfa4763fe547b40a738b7d4ea31f3799a08fe65d222c1d0
```

输入为第 36 轮 `pool200::blend::top20` 选择 store 和账户层净值；2026 补证另导出 `base/blend` 两个 `val63` 预测 store，并用同一 `decision_pipeline` 账户口径回测。特征只使用信号日前可见数据：全市场、基础 ML Top200 候选池和最终选中 Top20 的 `ret1/5/20/60`、MA20 宽度、近 20 日高点位置、20 日波动、量能比、下跌扩散率，以及候选池相对市场、选中组合相对候选池的状态差。

关键产物：

```text
/tmp/quantx-research/strong-stock-failure-risk-v1/strong_stock_failure_risk_blend_top20_dev_2022_2025.json
sha256:fc3f12df886fb00757a88a26519fdfa527b7b359f0b539b5660f734756b479b6

/tmp/quantx-research/strong-stock-failure-risk-v1/strong_stock_failure_risk_base_top20_val63_2026.json
sha256:0379344e022e5b909235db2cc72fae22fb3f98a5d13b96f8137a205df5946a6f

/tmp/quantx-research/strong-stock-failure-risk-v1/strong_stock_failure_risk_blend_top20_val63_2026.json
sha256:89447758682fb22711d54e13c8d5dc9d98882d68030a5f63ddf7e4ed3e7df8e1

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_base_pool200_top20_val63_2026_predictions.json
sha256:026ad54da2c567e8177df5677fd7be4c3beaf070e2470a4994934d4c7bf70dcb

/tmp/quantx-research/candidate-quintile-meta-ranker-v1/candidate_quintile_blend_pool200_top20_val63_2026_predictions.json
sha256:07ad4c69459b3c6ae41b9bd2a7bbdfdc965e25321216dea13d250c625c8a8e1a
```

### 37.3 结果

2022-2025 开发期 `blend` 共 170 个周频信号，未来 5 日账户平均收益 `+1.016%`，底部 20% 阈值 `-1.2899%`，其中 `nav_ret5 <= -5%` 的 crash week 有 8 个。简单压力规则能识别一部分已经扩散的风险：

| 压力规则 | 覆盖率 | bad rate | crash rate | crash capture | 平均未来 5 日账户收益 | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pool_drawdown_broad` | `12.35%` | `33.33%` | `19.05%` | `50.00%` | `+0.80%` | 能抓到后半段系统性下跌，但均值仍为正 |
| `selected_drawdown_broad` | `17.06%` | `34.48%` | `13.79%` | `50.00%` | `+0.90%` | 选中组合已普跌时风险升高，但不是充分条件 |
| `weak_breadth20` | `21.18%` | `13.89%` | `11.11%` | `50.00%` | `+0.64%` | 弱宽度能覆盖 crash，但误杀较多且 bad rate 不高 |

最差周里，2024-01-26 信号、2024-01-29 执行后的未来 5 日账户收益为 `-20.91%`，当时市场 20 日中位收益 `-5.69%`，市场 MA20 宽度 `0.296`，候选池 20 日中位收益 `-15.17%`，候选池 20 日跌超 10% 比例 `0.71`，选中组合跌超 10% 比例 `0.90`。这类风险在信号日前已经非常明显。

但第一刀失效并不稳定可见。2024-01-12 信号、2024-01-15 执行后未来 5 日账户收益 `-9.12%`，当时市场和候选池虽偏弱，但候选池跌超 10% 比例只有 `0.025`，选中组合跌超 10% 比例为 `0.0`。2024-04-10 信号、2024-04-11 执行后未来 5 日账户收益 `-9.47%`，市场和候选池事前状态甚至不差，属于简单宽度/回撤规则很难提前识别的失败。

2026 `val63` 账户层补证更直接否定了第 36 轮作为策略方向的可能：

| 账户层口径 | 区间 | 累计收益 | 年化 | 最大回撤 | 平均持仓 | 平均持仓天数 | 最差 20 日滚动 | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pool200::base::top20` | 2026-01-06 至 2026-07-10 | `-1.81%` | `-3.54%` | `-21.62%` | `19.02` | `10.64` | `-18.70%` | 持仓结构合规但收益/回撤不合格 |
| `pool200::blend::top20` | 同上 | `-1.82%` | `-3.56%` | `-23.72%` | `19.11` | `9.42` | `-23.62%` | 二阶段排序没有修复 2026，回撤更深 |

2026 风险画像共 24 个周频信号。`blend` 最差周为 2026-06-15 信号、2026-06-16 执行，未来 5 日账户收益 `-15.65%`，市场 MA20 宽度 `0.265`，市场 20 日中位收益 `-9.06%`，候选池 20 日跌超 10% 比例 `0.83`，选中组合跌超 10% 比例 `0.80`。这和 2024-01-26 类似，是“风险已扩散后继续接刀”。但 2026-02-24、2026-03-10 等失败周在市场和候选池状态上并不极端，仍说明简单风险 gate 不能覆盖所有第一刀。

2026 的相关性也提示更像脆弱性诊断，而不是稳定预测器：`base` 中最强非泄露 Spearman 是 `sel_volume_ratio20_median` `0.496`、`pool_volume_ratio20_p90` `0.464`、`pool_volume_ratio20_p10` `0.463`；`blend` 中最强为 `pool_volume_ratio20_p90` `0.411`、`sel_volume_ratio20_p10` `0.382`。量能衰减、60 日弱势、候选池波动上升与未来亏损同向，但样本只有 24 个信号，不能据此直接训练或手写阈值。

### 37.4 反事实分析

本轮最重要的反事实是：如果第 36 轮失败只是因为 `blend` 选择不佳，那么 2026 的 `base` 账户层应该显著好于 `blend`；实际二者都为约 `-1.8%`，且回撤都超过 `-20%`。因此主问题不是二阶段排序参数，而是基础强势候选在特定市场状态下整体变成负收益土壤。

第二个反事实是：如果市场风险宽度能直接变成一个可用 gate，那么被规则命中的周应该显著负收益，且能覆盖大多数 bad/crash week。实际开发期 `pool_drawdown_broad` 命中后平均未来 5 日仍为 `+0.80%`，2026 `pool_drawdown_broad` 能抓住 6 月大跌，但覆盖/误杀都重。风险状态解释力存在，但还没有达到可以硬切仓位的程度。

第三个反事实是：如果失败都来自市场系统性风险，那么 2024-04-10、2026-02-24 这类市场和候选池表面不差的失败不应该出现。它们的存在说明还缺股票级/行业级的“脆弱强势”刻画，例如近期拥挤、行业热度衰减、强势股内部量能断层、相对市场 beta、前期涨幅路径和候选池同质化。

### 37.5 判定

`diagnostic_only`，不进入正式策略库，不合代码。

本轮保留三个结论：

1. 2024 和 2026 的最坏段有共性，都是市场宽度转弱、候选池大面积回撤、选中组合近期也已明显受损时，强势股策略继续暴露导致账户层急跌。
2. 简单市场/候选池风险规则只能识别“已经扩散”的风险，不能可靠捕捉第一刀，也会误杀一部分正收益周。
3. 下一轮不应调仓位或调尾部权重，而应改候选生成目标：学习“强但不脆弱”的股票级条件，重点加入行业/概念热度衰减、候选拥挤、量能断层、近期路径同质性、市场 beta 和弱市抗跌后的再启动特征。

## 38. strong_but_not_fragile_v1

### 38.1 假设

第 37 轮把强势股失效拆成两类：一类是市场宽度和候选池已经明显转弱后的系统性风险，另一类是市场和候选池表面不差、但选中组合突然失败的第一刀。第一类不适合直接做硬 gate，因为误杀多；第二类更像股票级或行业/概念级脆弱性。因此本轮把问题前移到候选生成：在基础 ML Top200 候选池内，学习哪些强势股既有上行，又不属于高脆弱暴露。

本轮不是降权，也不是账户层风险 gate。它仍在候选池内做等权 Top15/20 标签层验证，只是训练时同时学习未来 5 日同日五分位、top quintile 和 bottom quintile 概率，试图减少脆弱强势股。

### 38.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/strong-but-not-fragile-v1/
```

脚本：

```text
/tmp/quantx-research/strong-but-not-fragile-v1/analyze_strong_but_not_fragile.py
sha256:e0ce115b04f639447fc82a86c0613b6c9b007d90a4f41861bf30fba34714e75d
```

本轮复用第 36 轮的基础 ML Top100/200 候选池、同日五分位目标和 walk-forward/forward 结构；复用第 34/35 轮的低波、近高、趋势回踩、动量、行业/概念 burst、市场宽度等股票级状态-因子特征。新增特征只在候选池面板中计算，不改正式框架：

1. 候选池内相对排名：`pool_mom60_rank`、`pool_low_vol20_rank`、`pool_pullback_from_high20_rank`、`pool_high_vol_leader_rank`、`pool_group_relative_leader_rank` 等。
2. 脆弱热度代理：`relative_heat_gap`、`concept_minus_industry_heat`、`fragile_heat_without_volume_proxy`。
3. 市场风险交互：`fragile_highvol_weakbreadth`、`fragile_mom60_weakbreadth`、`fragile_group_heat_weakbreadth`。
4. 候选拥挤：`industry_crowding`、`concept_crowding`、`crowded_group_heat`、`crowded_high_vol`。

评分方式：

1. `base`：基础 ML 原始排序。
2. `blend`：第 36 轮五分位期望与基础分数弱融合。
3. `resilient_ev`：五分位期望减 bottom-quintile 概率排名。
4. `strong_resilient`：top-quintile 概率、五分位期望、基础分数和 bottom-quintile 概率的候选级融合。
5. `fragile_penalty_blend`：保留基础分数和五分位期望，同时惩罚 bottom-quintile 概率。

关键产物：

```text
/tmp/quantx-research/strong-but-not-fragile-v1/strong_but_not_fragile_dev_2021_2025_label_diagnostic.json
sha256:2d8a8f1fc37485ced326ccb6411266dce3970af66e49952cd73a6a6a37749aa0

/tmp/quantx-research/strong-but-not-fragile-v1/strong_but_not_fragile_val63_2026_label_diagnostic.json
sha256:a713c94d1f673cf29ab10e863de33b92049ebc14821dd61d02ce8479f3337671

/tmp/quantx-research/strong-but-not-fragile-v1/smoke_strong_but_not_fragile_2021_2023.json
sha256:deccc5b93076847bea6471c5222f612482b5384401bba9c784ae8cb34ce14e83
```

### 38.3 结果

标签层结果没有通过晋级门槛：

| 评分 | TopK | 2022-2025 label5 | 2026 label5 | 2026 raw5 | top quintile rate dev / 2026 | bottom quintile rate dev / 2026 | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pool200::base` | Top20 | `+0.009717` | `+0.005062` | `+0.002493` | `24.79% / 28.12%` | `23.01% / 26.67%` | 基础候选仍是最强开发期基线 |
| `pool200::blend` | Top20 | `+0.008016` | `+0.005505` | `+0.002936` | `22.10% / 23.75%` | `20.03% / 22.08%` | 2026 小幅高于 base，但开发期降厚度 |
| `pool200::strong_resilient` | Top20 | `+0.006379` | `+0.001761` | `-0.000808` | `20.44% / 21.67%` | `18.37% / 19.58%` | 降低 bottom，但 top 和收益一起下降 |
| `pool200::fragile_penalty_blend` | Top20 | `+0.006664` | `-0.000657` | `-0.003226` | `18.68% / 17.29%` | `16.30% / 15.21%` | 最像避险惩罚，但 2026 转负 |
| `pool200::resilient_ev` | Top20 | `+0.005392` | `-0.005871` | `-0.008440` | `16.99% / 12.50%` | `15.39% / 17.50%` | 明显过度防守，失去 Alpha |

分年看，新增 resilient 评分虽然仍保持开发期各年正，但全都低于 `base`：

| 评分 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: |
| `pool200::base::top20` | `+0.017456` | `+0.008658` | `+0.008316` | `+0.004459` |
| `pool200::blend::top20` | `+0.011609` | `+0.006172` | `+0.008292` | `+0.006028` |
| `pool200::strong_resilient::top20` | `+0.010944` | `+0.006047` | `+0.002659` | `+0.005874` |
| `pool200::fragile_penalty_blend::top20` | `+0.009717` | `+0.007108` | `+0.005898` | `+0.003923` |

Top15 同样没有晋级：`pool200::base::top15` 开发期 `+0.010679`，2026 只有 `+0.001729`；`pool200::blend::top15` 开发期 `+0.009580`，2026 `+0.006098`；`strong_resilient` 和 `fragile_penalty_blend` 在 2026 Top15 分别为 `-0.000795`、`-0.002152`。

特征重要性显示，模型确实使用了新增脆弱性代理：`pool_mom60_rank`、`pool_low_vol20_rank`、`pool_pullback_from_high20_rank`、`pool_industry_burst_rank`、`fragile_highvol_weakbreadth`、`fragile_mom60_weakbreadth`、`relative_heat_gap` 等进入前列。但这些特征没有把“强但不脆弱”分离出来，只是把高收益和高风险同时降掉。

### 38.4 反事实分析

本轮最重要的反事实是：如果第 37 轮隐藏失败主要来自可学习的候选级脆弱性，那么惩罚 bottom-quintile 概率后，Top15/20 应该至少在开发期不明显低于 `base/blend`，并在 2026 或 bad-week 上更稳。实际结果相反：bottom quintile rate 的确下降了，但 top quintile rate 和平均收益也同步下降，说明 bottom 风险和上行弹性在这些特征里高度纠缠。

第二个反事实是：如果新增特征只是没被模型利用，特征重要性应该集中在 `base_ml_score/base_ml_rank_pct`。实际 `pool_mom60_rank`、`pool_low_vol20_rank`、`fragile_*`、行业/概念候选内 rank 都进入前列，说明模型看到了这些信息，但它学到的是“更保守的候选”，不是“保留上行、剔除脆弱”的候选。

第三个反事实是：如果这条路只是 2026 不适应，但开发期显著更稳，那么仍可进入账户层看回撤。实际开发期收益厚度已经明显低于 `base`，2026 的 `fragile_penalty_blend` 和 `resilient_ev` 又转负，因此没有账户层补证价值。

### 38.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“在基础 ML Top200 内用 bottom-quintile 概率惩罚脆弱候选”的第一版实现。它降低了失败分位暴露，但本质上是把组合变保守，不能自然产生更高收益。下一步不应继续调惩罚权重，也不应把它做成仓位 gate；更可能的方向是跳出单股票脆弱性惩罚，寻找新的候选生成土壤，例如：

1. 以行业/概念为一级对象先预测轮动强度，再在强组内选股，而不是先全市场选股再补行业/概念特征。
2. 对短周期路径做序列化建模，直接学习多日横截面路径形态，而不是把路径压成少数手工 rank。
3. 研究 2026 有效但开发期失败的事件延续现象是否只在特定市场微结构下成立，寻找可因果识别的 regime，而不是直接买事件候选。

## 39. group_forecast_stock_rerank_v1

### 39.1 假设

第 38 轮否定了在基础 ML Top200 内做“脆弱性惩罚”的方向，但它没有否定行业/概念轮动本身。A 股的短周期机会经常沿行业、概念和主题传播；旧的第 7 轮 `group-rotation-candidate-pool-v1` 只是用 T 日组强度做硬过滤，已经证明会删掉很多全市场 ML 强股。因此本轮改成更温和的结构：先学习行业/概念未来 5 日机会，再把 group forecast 作为股票排序增量，而不是把非强组股票直接过滤掉。

本轮要回答的反事实问题是：如果全市场 ML Top200 里已经包含强股，但缺少板块轮动方向感，那么 group forecast 的弱融合应该在 Top10/15/20 的未来 5 日 label 上稳定高于原始 ML 排序，并且至少不能损害开发期收益厚度。

### 39.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/group-forecast-stock-rerank-v1/
```

脚本：

```text
/tmp/quantx-research/group-forecast-stock-rerank-v1/analyze_group_forecast_stock_rerank.py
sha256:3c067a6b17ddf1f8894092ab2fbeddfb55d3c5d92d13da834978de7f3cf2c03a
```

输入预测使用 full-market 基础 ML OOS 结果，而不是第 36 轮已经截断到 Top20 的二阶段预测：

```text
/tmp/quantx-research/mainboard-lgbm-full-market/mainboard_lgbm_relative_10d_full_market-5700942ecf0f690c/oos_predictions.json
/tmp/quantx-research/mainboard-lgbm-forward-2026-full-market/mainboard_lgbm_relative_10d_forward_2026_full_market-4ba4ed8d937507ee/oos_predictions.json
```

核心流程：

1. 每日从 full-market ML 预测中取 Top100/200 作为候选池。
2. 用静态行业/概念成员关系构造 group 样本，特征包括 group ret1/3/5/10/20/60、宽度、量能、近涨停比例、强股比例、离散度、热度加速/衰减，以及市场宽度和市场离散度。
3. group 目标为未来 5 日 group 超额收益，并加入 Top200 候选在该 group 内的未来机会聚合，避免只学习平均板块收益。
4. 训练 `LGBMRegressor` 预测 group future opportunity，按当日行业/概念内 rank 得到 `group_forecast_rank`。
5. 股票排序不做硬过滤，只测试 `ml_score`、`blend_25/40/55`、`group_forecast`、`group_only_soft`。

关键产物：

```text
/tmp/quantx-research/group-forecast-stock-rerank-v1/group_forecast_stock_rerank_dev_2021_2025_label_diagnostic.json
sha256:7f8eeb4b3a9fe07c47c827b3c63596c1c0138f79188c1d19aab656bb25d16bfe

/tmp/quantx-research/group-forecast-stock-rerank-v1/group_forecast_stock_rerank_val63_2026_label_diagnostic.json
sha256:ebb0bbe02d788e5720df7a8763658153d5d54b57bde18b805f3b3e5b873286b6

/tmp/quantx-research/group-forecast-stock-rerank-v1/smoke_group_forecast_stock_rerank_2021_2022q1_v2.json
sha256:080b0dbebbb357976bb3310df5e78d893483922d0acc994eb7f0c5ff3b0e1b9f
```

### 39.3 结果

开发期结果没有支持假设。`all` 每日样本中，所有 group forecast 融合都低于原始 ML 排序：

| 口径 | TopK | `ml_score` label5 | `blend_25` label5 / delta | `group_forecast` label5 / delta | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all::pool200` | Top10 | `+0.013555` | `+0.009009 / -0.004546` | `+0.007955 / -0.005600` | 板块增量明显稀释头部强股 |
| `all::pool200` | Top15 | `+0.011526` | `+0.009608 / -0.001918` | `+0.007719 / -0.003807` | 正收益但低于 base |
| `all::pool200` | Top20 | `+0.010067` | `+0.009607 / -0.000460` | `+0.007286 / -0.002781` | 最弱融合接近，但仍负增量 |
| `all::pool200` | Top30 | `+0.008859` | `+0.008841 / -0.000019` | `+0.006949 / -0.001911` | 只有 Top30 几乎打平 |

开发期分年看，`blend_25` 在个别年份和更宽 TopK 接近 `ml_score`，但没有稳定胜出；`group_forecast` 在 2023 接近 0，说明模型容易把板块预测变成防守/均值回归代理：

| 口径 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: |
| `all::pool200::ml_score::top20` | `+0.0034` | `+0.0034` | `+0.0190` | `+0.0146` |
| `all::pool200::blend_25::top20` | `+0.0034` | `+0.0034` | `+0.0189` | `+0.0128` |
| `all::pool200::group_forecast::top20` | `+0.0025` | `+0.0013` | `+0.0156` | `+0.0097` |
| `due5::pool200::ml_score::top20` | `+0.0066` | `+0.0026` | `+0.0188` | `+0.0146` |
| `due5::pool200::blend_25::top20` | `+0.0059` | `+0.0010` | `+0.0220` | `+0.0119` |
| `due5::pool200::group_forecast::top20` | `+0.0041` | `+0.0002` | `+0.0168` | `+0.0091` |

2026 forward 也没有全局修复。每日 `all` 样本中，group forecast 对 Top10/15/20/30 都是负增量：

| 口径 | TopK | `ml_score` label5 | `blend_25` label5 / delta | `group_forecast` label5 / delta | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all::pool200` | Top10 | `+0.016403` | `+0.008225 / -0.008178` | `+0.007434 / -0.008969` | 2026 每日头部被显著稀释 |
| `all::pool200` | Top15 | `+0.015074` | `+0.010369 / -0.004705` | `+0.005879 / -0.009194` | 负增量更大 |
| `all::pool200` | Top20 | `+0.012545` | `+0.009860 / -0.002685` | `+0.005490 / -0.007055` | 不解决 2026 每日排序 |
| `all::pool200` | Top30 | `+0.009514` | `+0.009050 / -0.000464` | `+0.005142 / -0.004372` | 仅最弱融合接近 |

唯一值得记录的亮点在 `due5` 周频切片：2026 `due5::pool200::blend_55::top10` label5 `+0.013205`，高于 `ml_score` Top10 的 `+0.009359`；`group_forecast` Top10 为 `+0.012248`，也高于 base。但这个亮点没有扩展到 Top15/20/30，且开发期 `due5` 对应口径均为负增量：

| 口径 | Top10 | Top15 | Top20 | Top30 |
| --- | ---: | ---: | ---: | ---: |
| 2026 `due5::ml_score` | `+0.009359` | `+0.010454` | `+0.007382` | `+0.006540` |
| 2026 `due5::blend_55` | `+0.013205` | `+0.004658` | `+0.003609` | `+0.001134` |
| 2026 `due5::group_forecast` | `+0.012248` | `+0.003088` | `+0.003362` | `+0.002508` |

特征重要性显示，group 模型主要依赖市场状态而非可迁移的组内轮动结构：开发期前列为 `market_disp20`、`market_ret20_median`、`market_ret5_median`、`market_breadth20`、`ret60_rank`、`group_size_rank`、`heat_decay`；2026 也类似。这说明模型学到的首先是“什么市场环境下板块平均未来收益更好”，而不是足以重排强股的行业/概念细粒度传播信号。

### 39.4 反事实分析

第一反事实：如果旧的 hard filter 失败只是因为过滤太粗，那么不硬过滤、只做弱融合应当保留 ML 头部收益，并在行业/概念同步时增加收益厚度。实际 `blend_25` 虽然比 `blend_40/55` 更接近 base，但 Top10/15/20 仍普遍负增量，说明问题不是过滤力度，而是 group forecast 分数没有提供足够精细的股票级边际信息。

第二反事实：如果 group forecast 真能解释 2026 强势股失效，那么 2026 每日 `all` 样本应该改善。实际 2026 `all::pool200::group_forecast::top20` 只有 `+0.005490`，远低于 `ml_score` 的 `+0.012545`。`due5` Top10 的正增量更像调仓日切片和小样本共振，不足以作为策略方向。

第三反事实：如果本轮方向只是 TopK 选择错了，那么 Top30 应该更明显胜出。实际 Top30 只是 `blend_25` 几乎打平，`group_forecast` 仍明显低于 base。把 TopK 放宽不能把它变成收益土壤。

### 39.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“学习行业/概念未来机会后，直接作为 full-market ML Top200 股票排序增量”的第一版。它保留了行业/概念的重要性假设，但说明静态 group forecast 还不够细，容易变成市场状态代理或板块均值代理，不能稳定改善强势股头部排序。

下一步不继续调 `blend_25/40/55` 权重，也不把 2026 `due5` Top10 亮点硬卡成策略。更可能的方向是回到用户强调的“过去一段时间的横截面路径”：用多日 rank 序列直接刻画股票和候选池动态，例如最近 5/10/20 个交易日的 ML rank 轨迹、收益 rank 轨迹、量能 rank 轨迹、行业/概念热度 rank 轨迹，以及 rank 加速/稳定性/拥挤度。目标仍应先在 Top200 候选土壤里验证 label 层是否自然变厚，再考虑账户层。

## 40. path_sequence_ranker_v1

> 2026-07-15 审计更新：Exp40 未发现类似 Exp93 的未来函数问题，但已经不能继续作为 retained candidate 或硬基线。后续 Top7 bridge 接入主线时暴露出旧实验递归没有前置处理 ST/退市、停牌占位、一字板买不到、开盘大幅高开追入和卖出后反复回补等可交易性污染。补齐不可交易、一字板、-10% 止损、当前 ST/退市名称过滤、历史 5% 限幅 ST-like proxy、3% 高开不追和卖出冷却后，主线 Top7 bridge 2022-01-04 至 2026-07-15 仅为 `+164.39%`，年化 `+23.95%`，最大回撤 `-39.69%`，Sharpe `0.822`。因此本节旧 `+777.49%` / `+19.18%` 只能作为“路径特征含有信息”的历史诊断，状态降为 `quarantined_pending_tradability_replay`；必须用统一可交易性审计和完整 QuantX 回测重跑后才能重新定级。

> 审计备注：`write_path_sequence_predictions.py` 输出的 PredictionStore contract 中 `training_information_end` 为 store 级粗略字段，dev store 写为 `2021-01-03`，2026 store 写为 `2026-01-04`，不能精确表达每个年度 fold 的真实训练截止。实际训练边界以脚本代码和复算结果为准：开发期年度 fold 使用 `panel[year < test_year]`，2026 使用 2021-2025 train panel。后续若进入正式候选，应把 contract 元数据改成 fold-level 或显式记录 walk-forward 年度边界。

### 40.1 假设

第 39 轮说明，单独学习行业/概念未来机会再作为股票排序增量，会把强股头部收益稀释成市场或板块均值代理。但它没有否定用户强调的核心方向：A 股短周期机会常常不是单日截面，而是过去一段时间的横截面路径，包括个股强弱轨迹、量能路径、近高/回踩位置、行业/概念热度路径和市场宽度状态。

本轮要回答的反事实问题是：如果基础 5 日 ML Top200 已经提供了较好的收益土壤，但缺少对多日路径形态的刻画，那么在 Top200 内训练一个同日五分位路径 ranker，应当在 Top15/20 label 和账户层同时优于原始 ML 排序，并且在 2026 前向保持正收益。

### 40.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/path-sequence-ranker-v1/
```

脚本：

```text
/tmp/quantx-research/path-sequence-ranker-v1/analyze_path_sequence_ranker.py
sha256:685a95f6282f914e2b5e46937d576581b9bc1b75b7a029912ee48ad2ea0db80b

/tmp/quantx-research/path-sequence-ranker-v1/write_path_sequence_predictions.py
sha256:6a696cc38dda603a400833adc5aabcd86414a531b4b2fb7be5a8aac8ecf06da8
```

输入预测使用基础 5 日 ML：

```text
/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json
/tmp/quantx-research/adaptive-training-protocol-v1/forward-2026-5d-val63/mainboard_lgbm_base_5d_2010_2026_forward_val63_seed7-c52c47f834608a3c/oos_predictions.json
```

核心流程：

1. 每日从基础 5 日 ML 预测中取 Top200 作为候选池。
2. 构造 54 个路径和状态特征，包括基础 ML score/rank、RPS 1/3/5/10/20/60/120、RPS 20 日路径均值/最小/最大/斜率、量能路径 rank、MA 距离、近高/回撤、波动 rank、行业/概念收益和近涨停比例、市场收益/宽度/离散度。
3. 目标为同日未来 5 日超额收益五分位，训练 `LGBMClassifier` multiclass，评分包括 `path_ev`、`path_topq`、`path_spread` 和弱融合版本。
4. 标签层先看 `all/due5` 的 Top15/20，再导出 `path_topq::pool200::top20` prediction store，并用标准 `decision_pipeline` 做账户层补证。
5. 账户层固定等权 Top20，另测 `rebalance_interval_sessions: 5`，避免用每日高换手硬卡收益。

关键产物：

```text
/tmp/quantx-research/path-sequence-ranker-v1/path_sequence_ranker_dev_2021_2025_label_diagnostic.json
sha256:54845371cba70cc02db1049872ca83fa45c281c1eaec6e84325cfde83e0e5ec3

/tmp/quantx-research/path-sequence-ranker-v1/path_sequence_ranker_val63_2026_label_diagnostic.json
sha256:83a6a0cd82fef5ff0169157c844541c5fc0247b46a71cf94ab488b219c528309

/tmp/quantx-research/path-sequence-ranker-v1/path_sequence_path_topq_pool200_top20_dev_2021_2025_predictions.json
sha256:cbaae42bbc5ff8783884ffa03baf8f6de715577da0d44d8d44129e3f60df735e

/tmp/quantx-research/path-sequence-ranker-v1/path_sequence_path_topq_pool200_top20_dev_2022_2025_reb5_account_diagnostic.json
sha256:58a7af9a88e950a0a9143887d92cc9069e764a342b6d6448f1f46a382f798782

/tmp/quantx-research/path-sequence-ranker-v1/path_sequence_path_topq_pool200_top20_val63_2026_reb5_account_diagnostic.json
sha256:2177b296be39adf1bc8d2601cb848fd38642a8d483177c4f6b9a54cf36a03888
```

复核产物：

```text
/tmp/quantx-research/path-sequence-audit-v1/path_sequence_ranker_dev_2021_2025_label_audit.json
sha256:54845371cba70cc02db1049872ca83fa45c281c1eaec6e84325cfde83e0e5ec3

/tmp/quantx-research/path-sequence-audit-v1/path_sequence_ranker_val63_2026_label_audit.json
sha256:83a6a0cd82fef5ff0169157c844541c5fc0247b46a71cf94ab488b219c528309

/tmp/quantx-research/path-sequence-audit-v1/backtests/path-sequence-dev-2022-2025-reb5-audit/summary.json
sha256:17f8803e0bd426c6772d86641dd851afbc7b4c1402f220d41142f9c9fb6eecbc

/tmp/quantx-research/path-sequence-audit-v1/backtests/path-sequence-val63-2026-reb5-audit/summary.json
sha256:bb3a46153dee4adcf21af98205eb280027fbd9b793523b4a7c3b290fb3da809f
```

### 40.3 结果

标签层明显好于第 39 轮。开发期 `due5::pool200::path_topq::top15` label5 为 `+0.016013`，高于基础 ML `+0.014343`；`blend_topq_25::top15` 为 `+0.016123`。Top20 的增量较薄：`path_topq` 为 `+0.014170`，高于基础 ML `+0.013524`。

2026 前向更强，说明多日路径对 2026 的风格有解释力：每日 `all::pool200::path_topq::top20` label5 为 `+0.015905`、raw5 `+0.012836`，高于基础 ML label5 `+0.012018`、raw5 `+0.008950`；周频 `due5::pool200::path_topq::top20` label5 为 `+0.012520`，高于基础 ML `+0.005062`。

账户层采用 `path_topq::pool200::top20`：

| 口径 | 区间 | 累计收益 | 年化 | 最大回撤 | 平均持仓 | 平均持仓天数 | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 每日调仓 | 2026-01-06 至 2026-07-10 | `+25.60%` | `+56.78%` | `-17.96%` | `19.53` | `3.66` | 收益更高但持仓周期短于目标 |
| 每 5 个交易日调仓 | 2026-01-06 至 2026-07-10 | `+19.18%` | `+41.36%` | `-16.70%` | `19.48` | `9.94` | 通过 2026 前向和持仓约束，但回撤仍深 |
| 每 5 个交易日调仓 | 2022-01-07 至 2025-12-31 | `+777.49%` | `+72.50%` | `-32.62%` | `19.40` | `9.36` | 四年全正且收益显著变厚，但仍不到几十倍 |

开发期逐年账户收益为：2022 `+45.87%`、2023 `+62.93%`、2024 `+58.80%`、2025 `+127.50%`。交易层平均闭合收益 `+2.15%`，平均持仓 `9.36` 天，胜率 `51.47%`。2026 周频账户平均闭合收益 `+1.66%`，平均持仓 `9.94` 天，胜率 `48.01%`。

最大问题仍是风险和收益量级。开发期最大回撤发生在 2023-12-15 至 2024-02-05，回撤 `-32.62%`；最差 20 日滚动收益 `-27.51%`。2026 最大回撤发生在 2026-04-14 至 2026-06-29，回撤 `-16.70%`；最差 20 日滚动收益 `-12.09%`。

特征重要性前列包括 `vol20_rank_path`、`rps120`、`market_ret20_disp`、`market_breadth60`、`market_ret60_median`、`rps20_min20`、`rps1`、`market_ret20_median`、`vol_ratio20_mean5_rank_path`、`rps60_minus120`。这比第 39 轮更符合假设：模型没有只学板块均值，而是在使用个股中长期相对强弱、量能路径和市场风险状态的组合。

### 40.4 反事实分析

第一反事实：如果第 39 轮失败只是因为行业/概念预测太粗，那么加入更多路径特征后应当只在行业/概念相关特征上体现增量。实际重要性前列更多是个股 RPS、量能路径和市场宽度/离散度，说明真正有效的是“股票路径 + 市场状态”的联合结构，而不是单纯板块预测。

第二反事实：如果路径 ranker 只是开发期过拟合，那么 2026 前向不应显著改善基础 ML。实际 2026 `all` 和 `due5` Top20 label 都明显高于基础 ML，账户层也正收益，说明这个方向有真实增量。

第三反事实：如果这个方向已经满足用户目标，那么 2022-2025 加 2026 应该接近几十倍，并且回撤应可接受。实际 2022-2025 终值约 `8.77x`，叠加 2026 半年约 `1.19x`，仍只有约 `10.45x` 量级，离几十倍至 100 倍差距明显；开发期最大回撤超过 `-30%`，说明它不能作为最终策略直接晋级。

第四反事实：如果下一步继续调 TopK 或弱融合权重能解决目标，那么当前 Top20 周频账户应该已经接近收益量级，只差小修小补。实际缺口是数量级问题，不是尾部权重问题。因此下一步不应继续降低 rank6-10 或调混合权重，而应寻找新的收益弹性来源。

### 40.5 判定

`quarantined_pending_tradability_replay`，只作为历史诊断保留，不再作为候选底座或硬基线，不合代码。

本轮的正面结论需要降级表述：多日横截面路径学习确实能在旧标签和旧账户口径下增厚收益，说明路径、市场风险和行业/概念状态有信息；但旧结果没有通过统一可交易性污染审计，特别是未把 ST/退市、停牌占位、一字板、开盘追高和反复回补作为实验准入门槛。因此它不能证明存在可融入代码的强策略方向。

下一步不再围绕旧 Exp40 继续调尾部权重或轻微融合。新的递归实验必须从一开始纳入可交易性审计，并在每轮闭环里执行完整 QuantX 回测和交易异常审计：有效买入中 ST/退市、ST-like、缺失行情、零成交、零金额、一字涨停、高开追入、拒单原因、单票重复买入集中度都必须落档。只有通过该 gate 的结果，才允许和历史基线比较。

## 41. pre_breakout_quintile_path_v1

### 41.1 假设

第 30 轮 `big_winner_attribution_v1` 证明历史大涨股可以被 ML 归因，但直接预测未来 10/15/20 日 top5% winner 会把组合推向高噪声、高弹性头部，开发期周频 Top15/20 不稳。第 40 轮又证明，同日五分位路径目标比极端 winner 分类更贴近一周 Top15/20 等权收益。

本轮因此做一个组合反事实：不再用 winner 概率直接排序，而是把大涨股归因特征作为“爆发前兆候选土壤”约束，再用五分位路径目标决定可持有性。若这个方向成立，基础 5 日 ML Top500 中由前兆土壤筛出的候选，应当在开发期 due5 Top20 label 明显超过第 40 轮 `path_topq` 的 `+0.014170`，并且 2026 不低于第 40 轮 `+0.012520`。

### 41.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/pre-breakout-quintile-path-v1/
```

脚本：

```text
/tmp/quantx-research/pre-breakout-quintile-path-v1/analyze_pre_breakout_quintile_path.py
sha256:90abef2b50effc1f9943f0a18644cec513ae5ef261ed127daebbab43877b67bd
```

输入预测复用第 40 轮基础 5 日 ML：

```text
/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json
/tmp/quantx-research/adaptive-training-protocol-v1/forward-2026-5d-val63/mainboard_lgbm_base_5d_2010_2026_forward_val63_seed7-c52c47f834608a3c/oos_predictions.json
```

核心流程：

1. 每日从基础 5 日 ML 预测中取 Top500，而不是只在 Top200 内重排。
2. 新增大涨股归因启发的前兆特征：`vol60_rank_path`、`high120_dist_rank`、`low60_dist_rank`、`rps60_improve10_rank`、`warm_volume_rank`、`overheat_volume_rank`、`limit_count20_rank`、`near_limit_count20_rank`。
3. 构造 `pre_breakout_score`，偏好中长期强弱改善、接近中期高位但不过热、温和放量、行业/概念热度改善、市场宽度可接受，并惩罚近期过多涨停/近涨停。
4. 目标仍为未来 5 日同日超额收益五分位，训练 `LGBMClassifier` multiclass。
5. 评分包括 `base`、`pre_breakout`、`path_ev`、`path_topq`、`blend_20`、`blend_topq_25`、`pre_path_topq_35`、`base_pre_topq` 等；额外记录 bottom quintile rate，避免只看 winner 命中。

关键产物：

```text
/tmp/quantx-research/pre-breakout-quintile-path-v1/pre_breakout_quintile_path_dev_2021_2025_label_diagnostic.json
sha256:d3377a6ee6d9758e7244fdfd9a4cee7a4b97c6256e07fe944be826d1a970d22b

/tmp/quantx-research/pre-breakout-quintile-path-v1/pre_breakout_quintile_path_val63_2026_label_diagnostic.json
sha256:5fd73ac2848ffe06a7e7d3cf7128df341647fe89591835f743fe8449290550a7

/tmp/quantx-research/pre-breakout-quintile-path-v1/smoke_pre_breakout_quintile_path_2021_2022q1.json
sha256:d755fe99e04e5f8198779da8a48bfae6374a35a7bb3259db9b3a7aa2d5cc0ec1
```

### 41.3 结果

开发期没有通过晋级门槛。Top15 有局部接近第 40 轮，但 Top20 主目标没有超过路径底座：

| 口径 | Top15 label5 | Top20 label5 | Top20 bottom rate | 观察 |
| --- | ---: | ---: | ---: | --- |
| 第 40 轮 `path_topq::pool200` | `+0.016013` | `+0.014170` | 未记录 | 路径底座基准 |
| 第 41 轮 `blend_20::pool200/300/500` | `+0.016150` | `+0.013610` | `20.31%` | Top15 极小提高，但 Top20 低于第 40 轮 |
| 第 41 轮 `path_topq::pool200` | `+0.015455` | `+0.012813` | `30.08%` | top-quintile 命中提高但 bottom 暴露也升高 |
| 第 41 轮 `base_pre_topq::pool200` | `+0.011214` | `+0.010914` | `26.59%` | 前兆融合明显削弱开发期收益厚度 |

开发期 `blend_20` Top20 逐年仍为正，但低于第 40 轮主口径：2022 `+0.021082`、2023 `+0.009875`、2024 `+0.013410`、2025 `+0.009999`。第 40 轮 `path_topq::pool200::top20` 分年为 2022 `+0.015079`、2023 `+0.008028`、2024 `+0.017679`、2025 `+0.015929`，尤其 2024/2025 更厚。

2026 forward 显示前兆结构有局部有效性，但不足以覆盖开发期失败：

| 口径 | 2026 due5 Top15 label5 | 2026 due5 Top20 label5 | Top20 bottom rate | 观察 |
| --- | ---: | ---: | ---: | --- |
| 第 40 轮 `path_topq::pool200` | `+0.018316` | `+0.012520` | 未记录 | 2026 基准 |
| 第 41 轮 `pre_path_topq_35::pool100` | `+0.008004` | `+0.012559` | `26.46%` | Top20 与第 40 轮几乎打平，但 Top15 明显弱 |
| 第 41 轮 `path_topq::pool200` | `+0.010539` | `+0.009981` | `33.13%` | 低于第 40 轮且 bottom 暴露高 |
| 第 41 轮 `blend_20::pool200` | `+0.010866` | `+0.007590` | `22.71%` | 2026 正增量但不够厚 |

### 41.4 反事实分析

第一反事实：如果大涨股前兆特征能产生更肥候选土壤，那么开发期 due5 Top20 应该明显高于第 40 轮 `path_topq::pool200::top20` 的 `+0.014170`。实际最好 Top20 只有 `+0.013610`，说明前兆土壤没有自然扩成合规 Top20 收益厚度。

第二反事实：如果问题只是原 Top200 太窄，那么 Top300/Top500 应该带来明显增量。实际 `blend_20` 在 pool200/300/500 的 Top15/20 基本相同，`path_topq` 放到 pool500 后 Top20 还降到 `+0.011990`。这说明更宽候选池并没有给出新的金矿，只是引入更多噪声。

第三反事实：如果前兆分数有效但五分位模型没有利用，那么 `pre_breakout` 静态排序至少应当接近 base。实际开发期 `pre_breakout` 和 `base_pre_topq` 都显著低于 base/path，说明手工前兆分数本身并不是稳健 alpha 土壤。

第四反事实：如果 2026 的强势风格能够证明方向可保留，那么 2026 Top20 应该在保持开发期不弱的前提下超过第 40 轮。实际 `pre_path_topq_35::pool100::top20` 仅 `+0.012559`，只比第 40 轮 `+0.012520` 多 `0.000039`，而开发期不过关；这更像 2026 局部共振，不足以进入账户层。

### 41.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“用大涨股归因启发的爆发前兆土壤 + 五分位路径目标”第一版。它保留了两个有价值的观察：第一，2026 确实偏好部分前兆/路径组合；第二，bottom quintile rate 是必要诊断，因为 top-quintile 命中提高时 bottom 暴露也可能同步上升。但它没有解决核心问题：合规 Top20 的开发期收益厚度没有超过第 40 轮，更谈不上把账户收益从约 10x 推向几十倍。

下一步不继续微调 `pre_breakout_score` 或 Top500 池宽度。更可能的方向应转到“因果可识别的市场状态下切换收益土壤”，例如把第 40 轮路径底座按市场宽度/离散度/行业扩散状态拆成状态内专家，先证明某些状态下 Top20 label 大幅高于均值，再研究是否能因果识别这些状态，而不是继续扩大候选池或手写前兆分数。

## 42. state_conditioned_path_edge_v1

### 42.1 假设

第 40 轮 `path_sequence_ranker_v1` 已经是目前最接近目标结构的 ML 底座：Top20 周频账户 2022-2025 为 `+777.49%`，逐年全正，平均持仓 `19.40`，平均持仓 `9.36` 天；2026 前向为 `+19.18%`。但它离五年几十倍至 100 倍仍差一个收益弹性层。

第 41 轮说明，用大涨股归因手工扩 Top500 候选土壤没有自然变厚。因此本轮不再扩池、不调 TopK、不做权重。只做一个因果诊断：第 40 轮 `path_topq` 的收益边际，是否集中在信号日前可观测的市场/候选状态中。如果存在稳定的高收益状态和低收益状态，下一步才值得做状态条件模型；如果状态分桶在开发期和 2026 反向，就不应继续走硬 gate 或全局路由。

### 42.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/state-conditioned-path-edge-v1/
```

脚本：

```text
/tmp/quantx-research/state-conditioned-path-edge-v1/analyze_state_conditioned_path_edge.py
sha256:629740400a2b57d9997052838961bbe23a7828c43ea8f6979e9b0681968e053b
```

本脚本复用第 40 轮 `analyze_path_sequence_ranker.py` 的面板、特征和 `LGBMClassifier` 五分位目标，再额外输出 session-level 状态分桶。为避免口径漂移，本轮修正了 `due5` 定义：walk-forward 开发期必须在每个年度 fold 内各自取 `sessions[::5]`，而不是拼接 2022-2025 后全局取每 5 天。修正后与第 40 轮归档基线逐项一致：

| 口径 | 第 40 轮归档 | 本轮复跑 | 差异 |
| --- | ---: | ---: | ---: |
| `due5::pool200::base::top20` | `+0.013524138` | `+0.013524138` | `0.000000000` |
| `due5::pool200::path_topq::top20` | `+0.014169645` | `+0.014169645` | `0.000000000` |
| `due5::pool200::path_topq::top15` | `+0.016012961` | `+0.016012961` | `0.000000000` |
| `due5::pool200::path_topq::top10` | `+0.017398445` | `+0.017398445` | `0.000000000` |

状态变量全部是信号日前可见信息，包括：

1. 全市场 `market_breadth20/60`、`market_ret20/60_median`、`market_ret20_disp`。
2. 基础 ML Top200 候选池的 `rps20/rps60/rps20_mean20/rps20_slope10/rps5_minus20`、量能、近高回撤、波动、概念强度和概念拥挤。
3. `path_topq` 选中 Top20 自身的 RPS、概念强度和波动状态。

每个状态按样本内三分位切成 `low/mid/high`，报告 `path_topq::pool200::top20` 的 label、base 对照和 high-low spread。本轮只做标签诊断，不进入账户层。

关键产物：

```text
/tmp/quantx-research/state-conditioned-path-edge-v1/state_conditioned_path_edge_dev_2021_2025_diagnostic.json
sha256:9bdda757b79334a2939a43a5c3581413977580c9d945bd3c40561b0dbb02d5b5

/tmp/quantx-research/state-conditioned-path-edge-v1/state_conditioned_path_edge_val63_2026_diagnostic.json
sha256:2299dbf0053550f41aaee6d8b933affe432eb1262223abd7bf49076124a13698

/tmp/quantx-research/state-conditioned-path-edge-v1/smoke_state_conditioned_path_edge_2021_2022q1.json
sha256:9d79760bcbbc3c1df279c84a829b26aa76fe046d471892487284c1309219a118
```

### 42.3 结果

先看主基线：本轮没有创造新策略，只复现第 40 轮路径底座并做状态切片。

| 口径 | 开发期 due5 Top20 label5 | 2026 due5 Top20 label5 | 观察 |
| --- | ---: | ---: | --- |
| `base::pool200` | `+0.013524` | `+0.005062` | 2026 基础 ML 明显变薄 |
| `path_topq::pool200` | `+0.014170` | `+0.012520` | 2026 增益明显，但开发期增益只有 `+0.000646` |
| `blend_topq_25::pool200` | `+0.014335` | `+0.005326` | 开发期略强，2026 基本退回 base |
| `path_topq::pool200::top15` | `+0.016013` | `+0.018316` | Top15 2026 很强，但 Top20 才是更符合持仓约束主目标 |

状态切片中，唯一较稳定的正向解释变量是横截面离散度：

| 状态 high-low | 开发期 `path_topq` Top20 spread | 2026 spread | 开发期 delta spread vs base | 2026 delta spread vs base | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `market_ret20_disp` | `+0.014441` | `+0.020219` | `+0.002065` | `+0.003636` | 高分化市场中一周 Top20 更厚，且 path 边际略强于 base |
| `market_breadth20` | `+0.001349` | `-0.001676` | `-0.005814` | `-0.015059` | 宽度不是稳定正向 gate |
| `market_breadth60` | `+0.002450` | `+0.020519` | `-0.001451` | `-0.014204` | 2026 高宽度收益高，但主要来自 base 环境，不是 path 特有边际 |
| `pool200_rps20_median` | `-0.009992` | `+0.000310` | `-0.006011` | `-0.013638` | 候选池越强不等于越好，容易追高或拥挤 |
| `pool200_rps5_minus20_median` | `+0.010361` | `-0.017258` | `+0.009722` | `+0.008334` | 短期加速在开发期和 2026 label 反向 |
| `pool200_top_concept_share` | `+0.007322` | `-0.011909` | `+0.012270` | `+0.021773` | 概念拥挤的边际排序有信号，但总体收益方向不稳 |
| `selected20_path_vol20_median` | `-0.003676` | `-0.016990` | `+0.008136` | `-0.001218` | 选中组合高波动不是稳定收益来源 |
| `selected20_path_concept_ret5_median` | `+0.006574` | `+0.002704` | `+0.009255` | `+0.020883` | 选中组合概念强度有局部边际，但不足以形成硬规则 |

### 42.4 反事实分析

第一反事实：如果第 40 轮的收益缺口只是因为市场状态没有 gate，那么宽度、趋势或候选池强度的 high-low 分桶应该在开发期和 2026 同向且显著。实际只有 `market_ret20_disp` 稳定正向；宽度、候选 RPS、短期加速度、概念拥挤多项在开发期和 2026 反向，不能作为硬 gate。

第二反事实：如果高概念拥挤就是 A 股轮动的金矿，那么 `pool200_top_concept_share` 的 high 桶应该稳定更厚。实际开发期 high-low 为 `+0.007322`，但 2026 为 `-0.011909`。这说明概念拥挤既可能代表主题共振，也可能代表过热交易，不能只靠 session-level 拥挤程度判断。

第三反事实：如果高横截面离散度足以成为下一轮策略开关，那么它应该不仅提高 `path_topq` label，也应显著贡献相对 base 的边际收益。实际 `market_ret20_disp` 的 path label spread 很大，但 delta spread 只有开发期 `+0.002065`、2026 `+0.003636`。它更像“市场里有机会”的环境变量，而不是单独创造几十倍收益弹性的引擎。

第四反事实：如果把状态专家/路由再细分就能解决问题，那么本轮应当看到多个可解释状态在开发期和 2026 同向。实际状态信号稀疏且容易反向，和第 33 轮因子风格路由、第 37 轮强势股失效风险诊断、第 41 轮前兆土壤失败互相印证：硬状态切换容易把少量调仓样本切碎，学习到年份风格而不是可迁移规律。

### 42.5 判定

`diagnostic_only`，不进入账户层，不合代码。

本轮保留一个重要观察：高 20 日横截面离散度是路径底座最清晰的机会环境，A 股短周期收益确实依赖“市场是否给强势股足够横截面空间”。但它不能直接升级成仓位 gate 或候选族路由，因为多数状态变量不稳定，而且高离散度本身不能把 `path_topq` 从约 10x 量级推到几十倍。

下一步不继续做全局状态分箱、硬 gate 或离散专家路由。更合理的新方向是从股票级多日路径本身找收益弹性：不是问“今天该信哪个全局状态”，而是问“历史上哪些 20-60 日横截面路径片段，在类似市场离散度环境下，后续一周最容易继续赚钱”。这可以转成**路径相似性/shapelet/kNN 原型**方向：用过去多年每只股票的 RPS、量能、近高回撤、行业/概念相对强弱序列，寻找和当前候选相似的历史片段，按这些片段的未来 5 日分布给当前股票打分；市场离散度只作为条件变量，而不是硬开关。

## 43. path_prototype_memory_v1

### 43.1 假设

第 42 轮说明，全局市场状态和候选池状态能解释一部分机会环境，但不能稳定变成 gate 或路由。一个更贴近用户要求的方向，是不再问“今天是什么状态”，而是问“当前股票的多日横截面路径，历史上像不像那些随后一周继续上涨的股票”。

本轮因此测试一个最小记忆模型：在基础 5 日 ML Top200 候选池内，把每只股票的 RPS、量能、近高回撤、行业/概念相对强弱和市场状态压成路径/状态向量；只用训练期历史候选聚成原型，每个原型记录未来 5 日 label、top-quintile rate、bottom-quintile rate 和 raw return；预测期候选继承其所属原型的历史未来分布，再和第 40 轮 `path_topq` 对照。

若这个方向成立，原型分数或原型 + path 弱融合应在开发期 due5 Top20 明显超过第 40 轮 `path_topq::pool200::top20` 的 `+0.014170`，并在 2026 不低于 `+0.012520`。本轮不以 Top10 为主目标，因为平均持仓必须大于 5，且用户明确反对通过尾部权重技巧硬卡要求。

### 43.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/path-prototype-memory-v1/
```

脚本：

```text
/tmp/quantx-research/path-prototype-memory-v1/analyze_path_prototype_memory.py
sha256:9abb1a2a0a0f8bfa5a966071824e33ee82cebe095735cf8b83d83a02365a92ca
```

核心流程：

1. 复用第 40 轮的基础 ML Top200 候选、路径特征、同日五分位标签和 walk-forward/forward 训练边界。
2. 第一步仍训练第 40 轮的 `LGBMClassifier` 五分位 path ranker，保留 `path_ev/path_topq/path_spread` 作为基线。
3. 第二步在每个训练 fold 中用 37 个路径/状态列做标准化，并用 `MiniBatchKMeans` 聚成历史原型。原型数为 `min(384, max(48, train_rows // 500))`。
4. 每个原型记录训练期 `mean_label`、`mean_raw`、`topq_rate`、`bottom_rate` 和 `topq-bottom` spread。预测期股票根据所属原型得到 `proto_label/proto_topq/proto_spread/proto_raw` 分数。
5. 评分口径包括独立原型排序，以及 `path_topq` 与原型 rank 的 25% 弱融合：`path_proto_label_25`、`path_proto_topq_25`、`path_proto_spread_25`。这些融合只用于诊断，不作为调权重方向继续优化。

关键产物：

```text
/tmp/quantx-research/path-prototype-memory-v1/smoke_path_prototype_memory_2021_2022q1.json
sha256:a33064b8ddd6e050c1b4211c7cb80bb52fdb304c6c7c2d4fd3989fbaeba501fe

/tmp/quantx-research/path-prototype-memory-v1/path_prototype_memory_dev_2021_2025_label_diagnostic.json
sha256:0a6b907a5e825322c3d3f66a225caf3027f5b7c53911e09e15941d69cbbcaca0

/tmp/quantx-research/path-prototype-memory-v1/path_prototype_memory_val63_2026_label_diagnostic.json
sha256:6638777af42409bd5287ca96630d98eea242a63d3f9b5518961d01e3402e3530
```

### 43.3 结果

开发期主目标没有通过。所有原型相关 Top20 口径都低于第 40 轮 `path_topq`，包括看起来最贴近 winner 概率的 `proto_topq`：

| 口径 | 开发期 due5 Top20 label5 | 2026 due5 Top20 label5 | 开发期 Top20 top-quintile rate | 2026 Top20 top-quintile rate | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `base::pool200` | `+0.013524` | `+0.005062` | `25.82%` | `28.12%` | 基础 ML 对照 |
| 第 40 轮 `path_topq::pool200` | `+0.014170` | `+0.012520` | `30.21%` | `36.04%` | 当前路径底座 |
| `blend_topq_25::pool200` | `+0.014335` | `+0.005326` | `27.72%` | `29.38%` | 开发期略强但 2026 退化 |
| `proto_label::pool200` | `+0.012275` | `+0.009020` | `23.72%` | `24.37%` | 历史原型平均 label 不够厚 |
| `proto_topq::pool200` | `+0.010919` | `+0.011964` | `26.23%` | `31.25%` | 2026 接近 path，但开发期明显失效 |
| `proto_spread::pool200` | `+0.009050` | `+0.008421` | `20.82%` | `23.12%` | top-bottom 原型 spread 反而最弱 |
| `path_proto_label_25::pool200` | `+0.012562` | `+0.011729` | `28.38%` | `34.17%` | 融合后仍低于 `path_topq` |
| `path_proto_topq_25::pool200` | `+0.012392` | `+0.008647` | `28.95%` | `32.92%` | 开发期和 2026 都不优于 path |
| `path_proto_spread_25::pool200` | `+0.011595` | `+0.008932` | `27.74%` | `32.29%` | 融合削弱 path |

逐年看，`proto_topq` 开发期四年都是正，但每年都不够厚：2022 `+0.013785`、2023 `+0.006672`、2024 `+0.008663`、2025 `+0.014633`。第 40 轮 `path_topq` 对应为 2022 `+0.015079`、2023 `+0.008028`、2024 `+0.017679`、2025 `+0.015929`，尤其 2024 的收益厚度被原型记忆明显削弱。

2026 有局部强信号，但集中在 Top10 而非 Top20：

| 口径 | 开发期 due5 Top10 label5 | 2026 due5 Top10 label5 | 观察 |
| --- | ---: | ---: | --- |
| `path_topq::pool200` | `+0.017398` | `+0.023061` | 2026 已经很强 |
| `proto_topq::pool200` | `+0.013816` | `+0.023264` | 2026 略高于 path，但开发期明显弱 |
| `path_proto_spread_25::pool200` | `+0.017239` | `+0.023640` | 2026 Top10 最好，但开发期不优于 path |

### 43.4 反事实分析

第一反事实：如果“历史相似路径”能自然找出更肥收益土壤，那么开发期 Top20 应该超过第 40 轮路径底座。实际所有原型口径都低于 `path_topq`，最好原型融合 `path_proto_label_25` 也只有 `+0.012562`，比第 40 轮少约 `0.001608`。这不是小幅调参能解释的差距。

第二反事实：如果问题只是原型目标选错，那么 `proto_label`、`proto_topq`、`proto_spread` 至少应有一个在开发期 Top20 接近 path。实际三者分别为 `+0.012275`、`+0.010919`、`+0.009050`，说明历史原型把路径压成簇后，丢失了第 40 轮 LightGBM 在连续特征交互中学到的细粒度排序信息。

第三反事实：如果 2026 Top10 的强表现代表真实晋级方向，那么同一口径在开发期 Top10/Top20 不应显著落后。实际 `proto_topq` 2026 Top10 为 `+0.023264`，但开发期 Top10 只有 `+0.013816`，低于 `path_topq` 的 `+0.017398`；Top20 开发期更弱。这更像 2026 风格局部共振，而不是可穿越的收益引擎。

第四反事实：如果继续调聚类数、距离度量或 25% 融合权重能解决问题，那么 prototype 至少应该在某个主目标维度显示稳定正边际。实际它只在 2026 Top10 局部漂亮，开发期 Top20 全面负增量。继续调聚类数很容易变成围绕 2026/Top10 的事后拟合，和用户要求的“鲁棒、容易出收益”相反。

### 43.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“历史路径原型记忆作为收益弹性层”的第一版。它保留了一个观察：2026 的短周期强势股确实存在可被历史路径原型捕捉的头部 winner 概率。但这个概率没有稳定转化为开发期合规 Top20 等权收益，和此前大涨股 winner 分类、事件延续学习器的问题相同：能识别更尖锐的头部，却不能自然扩成可一周持有、平均持仓大于 5 的鲁棒收益池。

下一步不继续调 `MiniBatchKMeans` 聚类数、距离度量或原型融合权重。更值得换方向的是：寻找一个**更强的候选源或更贴近账户路径的训练目标**。当前基础 ML Top200 + 二阶段重排多次证明有上限，真正缺的是能把 Top20 整体收益分布抬厚的候选生成机制，而不是再对同一批候选做相似度记忆。

## 44. cross_candidate_oracle_v1

### 44.1 假设

第 40 轮 `path_sequence_ranker_v1` 是当前最好的 ML 底座，但五年加 2026 仍只是约 10x 量级，离“几十倍到 100 倍”明显不够。第 42/43 轮说明，继续在同一基础 ML Top200 内做状态分桶、历史路径原型或弱融合，无法自然把 Top20 整体收益抬厚。

本轮换一个更高层的诊断问题：收益弹性是否存在于**候选族之间的动态切换**。也就是说，不再问“同一批候选怎么微调排序”，而是问“当期应该用 path、head-union、多模型头部并集、周频持久性或基础 ML 中的哪一种候选来源”。

本轮只测上限，不训练可交易路由器。如果连未来信息 oracle 都不够厚，则该方向立即否决；如果 oracle 很厚，才说明下一步值得研究“能否用信号日前特征因果识别赢家候选族”。

### 44.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/cross-candidate-oracle-v1/
```

脚本：

```text
/tmp/quantx-research/cross-candidate-oracle-v1/analyze_cross_candidate_oracle.py
sha256:102760c8139e86f37e2c5ce6d522ef4a36c1a2ce5d9c07ab276ab4df67977dbf
```

候选族包括：

1. 第 40 轮路径底座内部族：`base_top20`、`path_ev_top20`、`path_topq_top20`、`path_spread_top20`、`blend_topq_25_top20`、`blend_spread_25_top20`。
2. 外部自然候选源：`head_union_7m_h6_top15`、`head_union_7m_h6_sum_top15`、`weekly_prev_top20_bonus`。

统一标签口径为信号日后 T+1 open 到 T+6 open 的 5 日收益，减同日主板均值 raw return。所有候选族只在第 40 轮 path 模型可评分的同一批 due5 session 上比较；walk-forward 开发期的 due5 仍按年度 fold 分别取 `sessions[::5]`，避免跨年拼接带来的调仓相位漂移。

oracle 口径：每个 due5 session 事后选择当期 `mean_label5` 最高的候选族，且候选数必须不低于 5。它使用未来真实收益，因此只能代表上限，不能作为策略收益。

关键产物：

```text
/tmp/quantx-research/cross-candidate-oracle-v1/cross_candidate_oracle_dev_2021_2025_diagnostic.json
sha256:23079a0c159b7ebaa49df31ea3f2bddcd1705a2abc0f294cd7fa1ba1dfb2ca27

/tmp/quantx-research/cross-candidate-oracle-v1/cross_candidate_oracle_val63_2026_diagnostic.json
sha256:1781d5c22886bb925a88aaeae0776912bf4b686458177d3552e4633567319d5e
```

### 44.3 结果

oracle 上限显著高于所有固定候选族，并且 2026 没有消失：

| 口径 | 开发期 due5 label5 | 2026 due5 label5 | 正 label session 比例 | 平均候选数 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| cross-family oracle | `+0.034764` | `+0.025719` | 开发期 `91.28%`，2026 `83.33%` | 开发期 `18.46`，2026 `18.92` | 上限很厚，且 2026 仍明显高于固定族 |
| `head_union_7m_h6_top15` | `+0.018187` | `+0.009089` | 开发期 `68.72%`，2026 `62.50%` | 开发期 `14.72`，2026 `14.42` | 开发期最强固定候选族 |
| `head_union_7m_h6_sum_top15` | `+0.018072` | `+0.009203` | 开发期 `66.67%`，2026 `62.50%` | 开发期 `14.72`，2026 `14.42` | 与 base5d anchor 接近 |
| `path_topq_top20` | `+0.014170` | `+0.012520` | 开发期 `67.69%`，2026 `62.50%` | `20.00` | 2026 最强固定候选族 |
| `weekly_prev_top20_bonus` | `+0.013658` | `+0.006097` | 开发期 `68.72%`，2026 `70.83%` | `20.00` | 稳但厚度不足 |
| `base_top20` | `+0.013524` | `+0.005062` | 开发期 `67.69%`，2026 `70.83%` | `20.00` | 2026 基础 ML 明显变薄 |

逐年看，oracle 开发期四年都为正，且没有只靠某一年拉高：

| 年份 | oracle due5 label5 | 正 label session 比例 |
| --- | ---: | ---: |
| 2022 | `+0.038621` | `93.88%` |
| 2023 | `+0.025949` | `93.88%` |
| 2024 | `+0.042046` | `83.67%` |
| 2025 | `+0.032391` | `93.75%` |
| 2026 | `+0.025719` | `83.33%` |

oracle 的选择分布也不是单一候选族垄断。开发期 195 个 due5 session 中，选择最多的是 `head_union_7m_h6_sum_top15` 43 次、`path_topq_top20` 32 次、`path_ev_top20` 24 次、`path_spread_top20` 20 次、`weekly_prev_top20_bonus` 17 次，其余也都有贡献。2026 的 24 个 session 中，`path_ev_top20` 7 次、`path_topq_top20` 5 次、`blend_spread_25_top20` 3 次、`head_union_7m_h6_sum_top15` 3 次，说明前向期也存在候选族轮换空间。

### 44.4 反事实分析

第一反事实：如果当前收益瓶颈只是第 40 轮 path 模型还没调好，那么跨候选族 oracle 不应大幅高于固定 `path_topq`。实际开发期 oracle `+0.034764`，是 `path_topq_top20` `+0.014170` 的 2.45 倍；2026 oracle `+0.025719`，也是 `path_topq_top20` `+0.012520` 的 2.05 倍。收益空间确实存在于候选族切换，而不只是同一排序器尾部微调。

第二反事实：如果候选族切换空间只是开发期过拟合，2026 应该明显消失。实际 2026 oracle 仍有 `+0.025719`，高于全部固定族，且正 label session 比例为 `83.33%`。这说明“不同市场片段适合不同 Alpha”是前向期仍可观察的结构。

第三反事实：如果固定拿最强候选族就足够，则开发期最强 `head_union_7m_h6_top15` 或 2026 最强 `path_topq_top20` 应接近 oracle。实际二者都差一大截：开发期最强固定族只有 `+0.018187`，2026 最强固定族只有 `+0.012520`。固定族收益厚度不足，是当前五年整体倍率不够的核心限制之一。

第四反事实：如果下一步继续做全局市场状态路由就够了，那么本轮应该能直接继承第 31-33 轮路由思路。问题是此前市场状态路由和在线近期表现路由都失败：有 oracle、无因果识别。本轮因此不能直接推进账户拼接，必须先验证“赢家候选族能否由信号日前的 session-level、候选池结构、候选族交集/分歧、风险状态和近期可实现收益特征预测”。

### 44.5 判定

`diagnostic_only`，不进入账户层，不合代码。

本轮是一个重要转向：候选族之间的 oracle 上限足够厚，且 2026 前向仍成立。这说明继续围绕单一基础 ML Top200 做二阶段排序、聚类、弱融合，边际很可能不如研究“何时该用哪类 Alpha”。

但本轮不是策略，因为 oracle 使用未来收益。下一轮必须做可因果识别诊断：把每个 due5 session 的候选族表现作为标签，构造信号日前可见特征，包括市场收益/宽度/离散度、基础 Top200 的路径分布、候选族之间交集和分歧、各族过去已完成调仓的 realized label/raw、候选族自身的行业/概念集中度和波动暴露。先验证能否提前选中或加权赢家候选族；如果不能，就说明 oracle 只是不可交易的后验上限。

## 45. causal_candidate_router_v1

### 45.1 假设

第 44 轮证明候选族之间存在很厚的未来信息 oracle：开发期 due5 label5 为 `+0.034764`，2026 为 `+0.025719`。但这只是上限，不是策略。真正需要验证的是：能否用信号日前可见的信息，在当期调仓前识别应该使用哪个候选族。

本轮因此测试一个低自由度因果 router。它不使用当前调仓未来 5 日收益，只使用市场状态、基础 ML Top200 候选池状态、候选族自身暴露、候选族之间交集/分歧，以及各候选族至少滞后两次 due5 调仓后才可见的历史实现表现。若这个方向成立，因果 router 应在同一批 due5 session 上明显超过固定最强候选族，并在 2026 不低于第 40 轮 `path_topq_top20`。

### 45.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/causal-candidate-router-v1/
```

脚本：

```text
/tmp/quantx-research/causal-candidate-router-v1/analyze_causal_candidate_router.py
sha256:57c0e0d70721e079e7b56c58f8bc4e2cf4423be9cde3371c8438fbc5016893c9
```

候选族与第 44 轮一致：`base_top20`、`path_ev_top20`、`path_topq_top20`、`path_spread_top20`、`blend_topq_25_top20`、`blend_spread_25_top20`、`head_union_7m_h6_top15`、`head_union_7m_h6_sum_top15`、`weekly_prev_top20_bonus`。

可见特征包括：

1. 信号日前全市场 `market_ret5/20/60_median`、`market_ret20_disp`、`market_breadth20/60`。
2. 基础 ML Top200 候选池的 RPS、量能、近高回撤、波动、行业/概念相对强弱均值和离散度。
3. 当前候选族选中股票在上述特征上的均值。
4. 当前候选族与其他候选族的最大交集和平均交集。
5. 候选族历史实现表现：`lag2_label/raw/pos`、滞后两次 due5 后的 3/5 轮 rolling label/raw。这个设计避免把当前持仓未来 5 日收益泄露进路由器。

测试了三个低自由度 router：

| router | 含义 |
| --- | --- |
| `router_train_prior` | 只根据训练期候选族平均表现选族，是“固定族但允许按训练期重估”的基线 |
| `router_lag_roll3` | 每期选滞后可见 3 轮 realized label 最好的候选族 |
| `router_reg` | LightGBMRegressor 用上述可见特征预测当期候选族 `mean_label5`，每期选预测最高族 |

关键产物：

```text
/tmp/quantx-research/causal-candidate-router-v1/smoke_causal_candidate_router_2021_2023q1.json
sha256:c477d3004fb896423d484e5a0c1bd6f56714febfc2649b97cbc4ced6557e5fb4

/tmp/quantx-research/causal-candidate-router-v1/causal_candidate_router_dev_2021_2025_diagnostic.json
sha256:71836dd1f4cc8a9dcfd22c21a575d00dc4b5ad617866dff350ce98015f772b79

/tmp/quantx-research/causal-candidate-router-v1/causal_candidate_router_val63_2026_diagnostic.json
sha256:c388eca1d827877725bc1076a2471cc07d53109a5afdc7c6ece7ca026d191312
```

### 45.3 结果

开发期 router 评估从 2023 年开始，因为 2022 年之前没有同口径候选族历史表现可训练。第 45 轮所有对比都限定在同一批可评估 session 上。

| 口径 | 开发期 due5 label5 | 2026 due5 label5 | 正 label session 比例 | 平均候选数 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| oracle | `+0.033469` | `+0.025719` | 开发期 `90.41%`，2026 `83.33%` | 开发期 `18.39`，2026 `18.92` | 上限仍厚，但不可交易 |
| 固定最强族：开发期 `head_union_7m_h6_sum_top15` / 2026 `path_topq_top20` | `+0.017039` | `+0.012520` | 开发期 `65.07%`，2026 `62.50%` | 开发期 `14.68`，2026 `20.00` | 因果 router 必须超过的基线 |
| `router_train_prior` | `+0.016605` | `+0.009089` | 开发期 `66.44%`，2026 `62.50%` | 开发期 `14.68`，2026 `14.42` | 开发期接近固定 head-union，但 2026 错选 head-union |
| `router_lag_roll3` | `+0.013317` | `+0.009211` | 开发期 `67.12%`，2026 `79.17%` | 开发期 `17.80`，2026 `18.63` | 近期候选族赢家延续性不足 |
| `router_reg` | `+0.010573` | `+0.011108` | 开发期 `64.38%`，2026 `70.83%` | 开发期 `19.16`，2026 `18.75` | 2026 接近 path_ev，但仍低于固定 path_topq |

逐年看，`router_train_prior` 的开发期表现主要来自 2024：2023 `+0.009916`、2024 `+0.024326`、2025 `+0.015551`。`router_lag_roll3` 为 2023 `+0.009766`、2024 `+0.017850`、2025 `+0.012314`。两者逐年均为正，但没有超过对应固定族的厚度，也没有接近 oracle。

2026 更关键：oracle 选择分散在 `path_ev/path_topq/blend/head_union` 多个候选族之间，说明候选族轮换空间仍存在；但因果 router 不能提前识别。`router_reg` 选择了 8 次 `base_top20`，而 2026 固定 `base_top20` 只有 `+0.005062`，这直接拖低了前向结果。

### 45.4 反事实分析

第一反事实：如果第 44 轮 oracle 的主要信息能由市场状态和候选池结构解释，那么 `router_reg` 应该能显著超过固定候选族。实际开发期 `router_reg` 只有 `+0.010573`，低于固定 `head_union_7m_h6_sum_top15` 的 `+0.017039`；2026 `router_reg` 为 `+0.011108`，也低于固定 `path_topq_top20` 的 `+0.012520`。信号日前状态没有足够识别力。

第二反事实：如果候选族近期强弱存在可交易动量，那么 `router_lag_roll3` 应明显超过固定族。实际开发期 `+0.013317`，2026 `+0.009211`，都没有超过同口径固定最强族。这与第 32 轮在线候选族动量路由的失败互相印证：候选族短期赢家延续性不够稳定。

第三反事实：如果只要根据训练期平均选择长期强族即可，那么 `router_train_prior` 应在 2026 仍好。实际它在 2026 全部选择 `head_union_7m_h6_top15`，只有 `+0.009089`，明显低于 `path_topq_top20` 的 `+0.012520`。开发期最强族和 2026 最强族发生切换，固定训练期 prior 不能穿越前向风格变化。

第四反事实：如果继续堆更复杂 router 就能解决，本轮低自由度模型至少应显示清楚正边际。实际 oracle 与 router 差距过大：开发期 oracle `+0.033469`，最好 router `+0.016605`；2026 oracle `+0.025719`，最好 router `+0.011108`。复杂模型更可能追逐后验赢家标签，而不是稳定提高可交易收益。

### 45.5 判定

`rejected`，不进入账户层，不合代码。

第 45 轮否决的是“用可见 session/候选族特征提前选择候选族”这个方向的第一版。第 44 轮 oracle 仍然是真实上限，但本轮说明这个上限暂时不可交易：当前可见特征拿不回它，近期候选族表现也没有足够延续性。

下一步不继续堆复杂 router、不做更多候选族择时。更合理的方向是改训练目标和候选生成机制：当前标签层 Top20 平均 label 已经反复显示上限，账户层几十倍目标可能需要直接优化一周持仓组合的路径收益/回撤，或者寻找更肥的全市场候选土壤，而不是在已有候选族之间事后择时。

## 46. portfolio_path_target_v1

### 46.1 假设

第 40 轮路径 ranker 仍是当前最好的 ML 底座，但它训练目标仍然是未来第 5 日终点收益的同日五分位。账户层问题可能不只来自终点，而来自一周持仓过程中的路径脆弱性：有些股票终点收益尚可，但中途回撤深、组合体验差，也可能在真实账户中拖累复利。

本轮因此测试一个更贴近一周持仓路径的目标：在基础 ML Top200 内，计算 T+1 open 到 T+6 open 期间每个候选相对全市场的路径收益，把 `0.50*终点超额 + 0.25*路径最差超额 + 0.25*路径均值超额` 做成同日五分位标签，再训练与第 40 轮同复杂度的 LightGBM 五分位分类器。

如果这个方向成立，路径质量目标应在合规 Top15/20 上显著超过第 40 轮 `path_topq`，并且 2026 前向不能退化。本轮不以 Top10 为主要目标，因为用户目标要求平均持仓大于 5，且此前多轮实验已证明头部小样本改善容易误导。

### 46.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/portfolio-path-target-v1/
```

脚本：

```text
/tmp/quantx-research/portfolio-path-target-v1/analyze_portfolio_path_target.py
sha256:e82cb788a62d5596342f2bddc7708319a3636d7016a654d50c5e95c577b6775a
```

本轮复用第 40 轮 `analyze_path_sequence_ranker.py` 的基础 ML Top200 候选、路径/市场/行业概念特征、walk-forward 和 forward 训练边界。新增字段包括：

1. `path_end_excess`：T+1 open 到 T+6 open 的终点超额收益，等价于原 `label5`。
2. `path_min_excess`：持仓期间 1-5 日路径超额收益中的最差值。
3. `path_mean_excess`：持仓期间 1-5 日路径超额收益均值。
4. `path_quality`：`0.50*path_end_excess + 0.25*path_min_excess + 0.25*path_mean_excess`。
5. `path_quality_quintile`：同 session 内按 `path_quality` 切五分位。

训练两个模型并同场评估：第 40 轮终点目标 `path_topq/path_spread`，以及本轮路径质量目标 `quality_ev/quality_topq/quality_spread`。另外报告两个弱融合反事实：`blend_quality_topq_25` 和 `path_quality_topq_50`。这些融合只用于诊断，不作为继续调权重方向。

关键产物：

```text
/tmp/quantx-research/portfolio-path-target-v1/smoke_portfolio_path_target_2021_2022q1.json
sha256:a0cbfa7ad205192930d1e05c1034e62253b980cbd58537b775f703324fc0375f

/tmp/quantx-research/portfolio-path-target-v1/portfolio_path_target_dev_2021_2025_diagnostic.json
sha256:4a84ae6ae0a3d61a40cd096d66fc348457de5f0ebd15a26501a58c56ecf00d73

/tmp/quantx-research/portfolio-path-target-v1/portfolio_path_target_val63_2026_diagnostic.json
sha256:83bb42c951c53b49079964a75f90b03c2c87cab392640defb1bf6fa5c5a1bb65
```

### 46.3 结果

开发期有轻微正边际，但不够厚。主目标 Top20 中，最好的是 `blend_quality_topq_25`，仅比第 40 轮 `path_topq` 高 `+0.000647`：

| 口径 | 开发期 due5 Top20 label5 | 2026 due5 Top20 label5 | 开发期路径最差超额 | 2026 路径最差超额 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `path_topq` | `+0.014170` | `+0.012520` | `-0.024390` | `-0.032166` | 第 40 轮基线 |
| `quality_topq` | `+0.014599` | `+0.008077` | `-0.024345` | `-0.035506` | 开发期略高，2026 明显退化 |
| `path_quality_topq_50` | `+0.014715` | `+0.011611` | `-0.024551` | `-0.032084` | 开发期小增益，2026 仍低于 path |
| `blend_quality_topq_25` | `+0.014816` | `+0.005163` | `-0.021583` | `-0.033730` | 开发期路径更平滑，但 2026 失效 |

Top15 开发期也有小幅改善，但 2026 不过：

| 口径 | 开发期 due5 Top15 label5 | 2026 due5 Top15 label5 | 观察 |
| --- | ---: | ---: | --- |
| `path_topq` | `+0.016013` | `+0.018316` | 基线强，尤其 2026 |
| `quality_topq` | `+0.017191` | `+0.012101` | 开发期 +0.001178，但 2026 -0.006215 |
| `path_quality_topq_50` | `+0.017067` | `+0.015524` | 开发期 +0.001054，2026 仍低于 path |

Top10 是本轮最漂亮的地方，但它不是主目标：

| 口径 | 开发期 due5 Top10 label5 | 2026 due5 Top10 label5 | 观察 |
| --- | ---: | ---: | --- |
| `path_topq` | `+0.017398` | `+0.023061` | 第 40 轮 Top10 基线 |
| `quality_topq` | `+0.020741` | `+0.023221` | 开发期 +0.003343，2026 小幅 +0.000161 |
| `path_quality_topq_50` | `+0.020062` | `+0.023720` | 开发期 +0.002664，2026 小幅 +0.000660 |

开发期逐年看，`blend_quality_topq_25` Top20 的收益为 2022 `+0.020535`、2023 `+0.010587`、2024 `+0.015859`、2025 `+0.012232`，四年均正；但 2025 已经明显低于 `path_topq` 的 `+0.015929`，前向 2026 则大幅退化。

### 46.4 反事实分析

第一反事实：如果账户收益不足主要来自一周路径脆弱，那么路径质量目标应同时改善 `path_min_excess` 和最终 `label5`。实际 `blend_quality_topq_25` 开发期 Top20 的路径最差超额从 `-0.024390` 改善到 `-0.021583`，但最终 label 只提高 `+0.000647`；2026 最终 label 反而从 `+0.012520` 降到 `+0.005163`。路径平滑和终点收益并没有稳定同向。

第二反事实：如果目标函数更贴近账户，就应该在 Top15/20 这种合规持仓层稳定改善。实际改善主要集中在 Top10；Top20 开发期只有薄增益，2026 明显变差。这个结构和用户反对的“靠头部或权重卡要求”不一致。

第三反事实：如果问题只是终点标签太粗，那么纯 `quality_topq` 应显著强于 `path_topq`。实际开发期 Top20 只高 `+0.000429`，2026 却低 `-0.004444`。说明第 40 轮路径模型已经隐含学习到一部分路径质量；显式替换目标没有产生新的厚收益土壤。

第四反事实：如果继续调 `0.50/0.25/0.25` 权重能解决问题，那么当前权重至少应在主目标显示清楚方向性。实际只有 Top10 和开发期路径最差超额好看，Top20 前向不通过。继续调路径权重很容易变成围绕 Top10 或 2026 局部的事后拟合。

### 46.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“把终点 5 日收益标签替换为一周路径质量标签”这个方向。它确实能改善部分路径指标和 Top10 winner 捕捉，但没有把合规 Top20 收益变厚，也没有通过 2026 前向。

下一步不继续调路径质量权重，也不做更多弱融合。当前证据更支持一个判断：已有基础 ML Top200 候选池和二阶段目标的收益上限已经比较清楚，真正缺的是更肥的候选生成机制，或完全不同的数据/特征土壤，而不是继续在同一批候选上微调标签形状。

## 47. honest_subgroup_soil_v1

### 47.1 假设

第 46 轮以后，继续在基础 ML Top200 内微调目标函数的空间已经很薄。更可能的问题是候选池本身不够肥：Top200 排序器把很多可交易候选混在一起，但真正容易出收益的股票可能只出现在某些可观测的“土壤”里，例如过去 60/120 日强弱、近高位置、量能确认、市场宽度、行业/概念热度和候选池内部拥挤度的组合状态。

本轮因此不再调尾部权重，也不做后验候选族路由，而是做一个诚实子群发现：每个 OOS 年份只用该年前训练样本，在基础 ML Top500/Top1000 候选内训练浅层决策树，寻找历史上未来 5 日收益显著较高的叶子；然后把锁定叶子直接应用到测试年份和 2026 forward。若这个方向成立，应该同时满足两个条件：标签层能显著提高宽候选池 Top20，账户层能超过第 40 轮 `path_topq`，并且 2026 不退化。

### 47.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/honest-subgroup-soil-v1/
```

核心脚本：

```text
/tmp/quantx-research/honest-subgroup-soil-v1/analyze_honest_subgroup_soil.py
sha256:390bb74f28dc7cbc6ca61c9afad573ee0b28921af2b92a4bd466654b6942a00b

/tmp/quantx-research/honest-subgroup-soil-v1/write_honest_subgroup_predictions.py
sha256:50963d491a7cc7da146aef63e50304041b37db2c4cdc6c41bb4b0df3068c9a70
```

本轮复用第 40 轮路径 ranker 的市场、行业、概念和多日路径特征，但候选池扩大到基础 ML Top500/Top1000。训练规则为：

1. walk-forward 开发期：2022、2023、2024、2025 各年只使用此前年份样本训练浅树，再应用到当年。
2. 2026 forward：使用 2021-2025 全部开发期样本训练浅树，再锁定规则应用到 2026-01-05 至 2026-07-10。
3. 叶子筛选要求叶样本量至少 2500，叶均值相对训练集整体高出 `0.003`，最多保留 32 个叶子。
4. 评估 `soil_leaf`、`soil_path`、`soil_base` 三类排序，其中 `soil_leaf` 只用训练期叶均值作为土壤分数，不用测试期收益。

关键产物：

```text
/tmp/quantx-research/honest-subgroup-soil-v1/smoke_honest_subgroup_soil_2021_2022q1.json
sha256:5ab6c05f506679cfc33ed4409e400bbd8ad42ea8b4d8073bfd83ff9022c3aff8

/tmp/quantx-research/honest-subgroup-soil-v1/honest_subgroup_soil_dev_2021_2025_diagnostic.json
sha256:ceebfaf17c13daffe1b01bd901241e2541fe7789955de28ce855f578d8b791c3

/tmp/quantx-research/honest-subgroup-soil-v1/honest_subgroup_soil_val63_2026_diagnostic.json
sha256:1ce9e494edc6ae11dd27fd7515f7c2cf5457d5c4b884a8253f4308a0e06615e3
```

账户层导出的标准 PredictionStore：

```text
/tmp/quantx-research/honest-subgroup-soil-v1/honest_soil_pool500_soil_leaf_top20_dev_2021_2025_predictions.json
checksum: sha256:9b54cd159cb9ba8e8fed955ac438ee694d6a99e2104b31dc4bdfcab66ff05580

/tmp/quantx-research/honest-subgroup-soil-v1/honest_soil_pool500_soil_leaf_top20_val63_2026_predictions.json
checksum: sha256:ece2125165b49320b0c0d2197b90bc1e134a096c9da3af5658cfe0a3110713cd
```

账户层仍使用标准 `decision_pipeline`：5 日调仓、T 日信号、T+1 open 成交、Top20 等权、标准费用、真实交易规则校验。

### 47.3 标签层结果

开发期 `pool500::soil_leaf::top20` 的 due5 label 为 `+0.013597`，高于同池 `path_topq::top20` 的 `+0.010637`，但仍低于第 40 轮 `path_topq::pool200::top20` 的 `+0.014170`。`pool500::soil_path::top20` 为 `+0.014942`，略高于第 40 轮标签层，但它混入路径排序，账户层优先验证更朴素的 `soil_leaf`。

2026 是本轮最有价值的观察：`pool500::soil_leaf::top20` 的 due5 label 为 `+0.012569`，明显高于同池 `path_topq::top20` 的 `+0.003593`，正 label session 比例为 `73.91%`。这说明训练期发现的土壤规则在 2026 前向并没有消失，且能修复基础 Top500 在 2026 的宽候选池稀释问题。

| 口径 | 开发期 due5 Top20 label5 | 2026 due5 Top20 label5 | 开发期平均土壤候选数 | 2026 平均土壤候选数 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `pool500::path_topq` | `+0.010637` | `+0.003593` | - | - | 宽池 path 基线变薄 |
| `pool500::soil_leaf` | `+0.013597` | `+0.012569` | `133.85` | `50.13` | 2026 改善最明显 |
| `pool500::soil_path` | `+0.014942` | `+0.009128` | `133.85` | `50.13` | 开发期更高，2026 低于 soil_leaf |
| `pool1000::soil_leaf` | - | `+0.008984` | - | `51.57` | 更宽候选池反而稀释 |

逐年看，`pool500::soil_leaf::top20` 开发期四年均为正：2022 `+0.019260`、2023 `+0.007478`、2024 `+0.010885`、2025 `+0.014942`。但 2023 和 2024 的厚度并不够，提示它更像稳定筛选器，不像高弹性收益源。

### 47.4 账户层结果

账户层补证如下：

```text
/tmp/quantx-research/honest-subgroup-soil-v1/honest_soil_pool500_soil_leaf_top20_dev_2022_2025_reb5_account_diagnostic.json
sha256:ac7fbb53d78f1a7862db0530c43028aa2e235ce72fd2bc18f2c1a04ff03c68e1

/tmp/quantx-research/honest-subgroup-soil-v1/honest_soil_pool500_soil_leaf_top20_val63_2026_reb5_account_diagnostic.json
sha256:20862dc78342c58dcb6762a70f359325f7d56ac59cdfd40c347b03a9743507a2
```

| 口径 | 总收益 | 年化 | 最大回撤 | 平均持仓数 | 平均持仓天数 | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Exp40 `path_topq` 2022-2025 | `+777.49%` | `+72.50%` | `-32.62%` | `19.40` | `9.36` | 当前最强账户层基线 |
| Exp47 `soil_leaf` 2022-2025 | `+361.14%` | `+46.77%` | `-36.08%` | `15.42` | `11.55` | 四年全正，但倍率显著下降 |
| Exp40 `path_topq` 2026 | `+19.18%` | - | `-16.70%` | `19.48` | `9.94` | 2026 基线 |
| Exp47 `soil_leaf` 2026 | `+33.50%` | `+76.84%` | `-16.11%` | `17.96` | `10.40` | 2026 前向明显更好 |

开发期逐年账户结果：

| 年份 | 收益 | 最大回撤 | 平均持仓数 | 平均持仓天数 |
| --- | ---: | ---: | ---: | ---: |
| 2022 | `+31.36%` | `-19.97%` | `15.41` | `9.25` |
| 2023 | `+46.03%` | `-11.28%` | `12.07` | `11.77` |
| 2024 | `+10.49%` | `-35.38%` | `16.77` | `13.92` |
| 2025 | `+115.75%` | `-15.74%` | `17.40` | `11.78` |
| 2026 | `+33.50%` | `-16.11%` | `17.96` | `10.40` |

这些数字满足“平均持仓大于 5、持仓约一周、逐年为正、2026 forward 为正”，但不满足最关键的收益量级：2022-2025 只有 `4.61x`，而第 40 轮已有 `8.77x`，离几十倍到百倍目标仍很远。

### 47.5 反事实分析

第一反事实：如果诚实子群土壤是真正的新高弹性 Alpha，那么账户层开发期应至少超过第 40 轮路径 ranker。实际 Exp47 开发期总收益 `+361.14%`，低于第 40 轮 `+777.49%`，且最大回撤更深。这说明土壤筛选提升了候选质量的稳定性，但削掉了一部分高弹性赢家。

第二反事实：如果 2026 改善只是标签层偶然，账户层应出现明显转换损耗。实际 2026 账户收益从第 40 轮 `+19.18%` 提高到 `+33.50%`，回撤略低，说明 forward 线索是真的。问题不在 2026，而在长期收益弹性不足。

第三反事实：如果继续扩大到 Top1000 能找到更多赢家，那么更宽池应该更强。实际 2026 `pool1000::soil_leaf::top20` 标签只有 `+0.008984`，低于 `pool500::soil_leaf` 的 `+0.012569`。盲目扩大候选池会引入噪声，不是自然出收益的路。

第四反事实：如果这只是需要调参数，比如叶样本数、叶边际、TopK，那么当前默认规则至少应在开发期显示超过强基线的方向。实际强在 2026、弱在 2022-2025，继续围绕浅树阈值调参很容易变成按前向局部特征拟合，而不是找到普适机制。

### 47.6 判定

`rejected_with_signal`，不合代码。

本轮不能作为合格策略，因为长期收益倍率不足，且没有超过第 40 轮账户层基线。但它留下一个有价值信号：在 2026，训练期发现的高收益土壤能显著改善宽候选池，说明“候选生成层”比“尾部权重和目标函数微调”更值得继续。

下一步应转向更直接的“大涨股/爆发前归因”方向：不是从普通 Top500 里找浅树叶子，而是先定义历史上 5-20 日大幅上涨的事件，反向学习它们在爆发前 5-60 日的横截面、行业/概念扩散、量价路径、近高位置、市场风险状态和同组领涨扩散特征，再把这个爆发概率作为候选生成器，与第 40 轮路径 ranker 做同口径 forward 和账户层验证。

## 48. right_tail_event_candidate_v1

### 48.1 假设

第 30 轮证明历史大涨股可以被 ML 归因，但直接全市场预测 top5% winner 会把组合推向高噪声头部；第 41 轮把归因手工写成前兆分数，又在开发期 Top20 变薄；第 47 轮说明候选生成层在 2026 有价值，但长期弹性不足。

本轮因此做一个更可交易的事件学习版本：不在全市场追极端 winner，而是在基础 ML Top500 候选池内分别学习未来 5 日同池右尾 top quintile 和左尾 bottom quintile。排序不只看 `p_winner`，还看 `p_winner - p_loser`，并测试与基础 ML rank 的弱融合。若历史大涨股归因能自然转成合规一周组合，`event_edge` 或 `event_base_35` 应该在开发期 due5 Top20 明显超过基础 Top20 和第 40 轮 `path_topq`，且 2026 不退化。

### 48.2 方法和边界

输出根目录：

```text
/tmp/quantx-research/right-tail-event-candidate-v1/
```

脚本：

```text
/tmp/quantx-research/right-tail-event-candidate-v1/analyze_right_tail_event_candidate.py
sha256:155145651a94b90cafb038f4a6e88884479815b868a082fbab935318682bb39a
```

脚本直接复用第 40 轮 `path_sequence_ranker_v1` 的基础 ML Top500 候选、市场/路径/行业/概念特征和同日未来 5 日 label quintile。新增两个二分类目标：

1. `winner_event = label_quintile == 4`。
2. `loser_event = label_quintile == 0`。

每个 walk-forward 年份分别训练 winner/loser 两个 LightGBM 二分类器，并在测试年计算：

| 变体 | 含义 |
| --- | --- |
| `base` | 基础 ML rank，不加事件模型 |
| `p_winner` | 只按右尾概率排序 |
| `event_edge` | 按 `p_winner - p_loser` 排序 |
| `event_base_20` | `0.80*base_rank + 0.20*event_edge_rank` |
| `event_base_35` | `0.65*base_rank + 0.35*event_edge_rank` |
| `win_base_20` | `0.80*base_rank + 0.20*p_winner_rank` |

关键产物：

```text
/tmp/quantx-research/right-tail-event-candidate-v1/smoke_right_tail_event_candidate_2021_2022q1.json
sha256:a7063949479d03a69e67701d481e0d0a4f3c0ce5afb3fbab0bd0f7e1af5a24a3

/tmp/quantx-research/right-tail-event-candidate-v1/right_tail_event_candidate_dev_2021_2025_diagnostic.json
sha256:cdc65ee61d6623c5894fd71de7744d9ceaaf613d066d703ed06033f6042f4c3b

/tmp/quantx-research/right-tail-event-candidate-v1/right_tail_event_candidate_val63_2026_diagnostic.json
sha256:2a62236e8c2e365a1eec3699e829cd9138ca64ab38738f00bb2acd244cd11a91
```

### 48.3 结果

开发期 due5 Top20 没有通过。右尾模型确实提高了 winner 命中率，但同时把 bottom 暴露抬高；左尾惩罚降低了 bottom 暴露，但也削掉了收益厚度。

| 口径 | 开发期 due5 Top20 label5 | delta vs base | top quintile rate | bottom quintile rate | 2026 due5 Top20 label5 | 2026 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `+0.013524` | `0.000000` | `27.23%` | `23.36%` | `+0.005062` | `0.000000` |
| `p_winner` | `+0.012208` | `-0.001316` | `30.54%` | `31.44%` | `+0.007404` | `+0.002341` |
| `event_edge` | `+0.009539` | `-0.003985` | `21.18%` | `18.41%` | `+0.007901` | `+0.002839` |
| `event_base_35` | `+0.012350` | `-0.001174` | `23.33%` | `19.49%` | `+0.008387` | `+0.003325` |

与第 40 轮 `path_topq::pool200::top20` 对比更直接：开发期第 40 轮为 `+0.014170`，2026 为 `+0.012520`。本轮最好的合规 Top20 在开发期低于第 40 轮，2026 也没有超过第 40 轮。

逐年看，`p_winner` 的右尾捕捉并非没有信息：2024 为 `+0.015409`、2025 为 `+0.015208`，高于 base 的 2025 `+0.011108`。但 2022 和 2023 被 bottom 风险拖累，尤其 2023 positive label session ratio 只有 `46.94%`，说明右尾事件模型在弱环境里会把高弹性失败样本也选出来。

`event_edge` 的反方向也很清楚：开发期 bottom quintile rate 从 base 的 `23.36%` 降到 `18.41%`，但 top quintile rate 也从 `27.23%` 降到 `21.18%`，最终 Top20 label 从 `+0.013524` 降到 `+0.009539`。这不是收益增强器，而是同步降风险和降弹性的过滤器。

特征重要性也支持这个解释。开发期最重要的变量集中在市场离散度、宽度、60 日市场收益、长期 RPS 和波动：

| 排名 | 特征 | 平均重要性 |
| ---: | --- | ---: |
| 1 | `loser::market_ret20_disp` | `391.25` |
| 2 | `loser::market_breadth60` | `350.75` |
| 3 | `winner::market_breadth60` | `289.50` |
| 4 | `loser::market_ret60_median` | `287.75` |
| 5 | `winner::rps120` | `286.25` |
| 6 | `winner::market_ret20_disp` | `285.25` |
| 7 | `winner::market_ret60_median` | `276.50` |
| 8 | `loser::market_ret20_median` | `264.75` |

这说明 winner 和 loser 不是两套完全分离机制，二者共享市场状态、波动和强弱结构。A 股的短期右尾和左尾都集中在高弹性状态里，只学“大涨概率”很容易同时买到大跌概率。

### 48.4 反事实分析

第一反事实：如果历史右尾 winner 概率就是新 alpha，那么 `p_winner` 应该在开发期 Top20 明显超过 base。实际 `p_winner` 为 `+0.012208`，低于 base `+0.013524`。虽然 top quintile rate 从 `27.23%` 升到 `30.54%`，bottom quintile rate 也从 `23.36%` 升到 `31.44%`，收益被左尾共振吃掉。

第二反事实：如果问题只是右尾模型没有避开失败样本，那么 `event_edge = p_winner - p_loser` 应该修复。实际 `event_edge` 把 bottom quintile rate 降到 `18.41%`，但 top quintile rate 同时降到 `21.18%`，label 只有 `+0.009539`。左尾惩罚没有保留右尾收益，只是把高弹性样本整体拿掉。

第三反事实：如果弱融合可以保留 base 的稳健性同时增加事件弹性，那么 `event_base_35` 应该至少超过 base。实际开发期 `event_base_35` 为 `+0.012350`，仍低于 base；2026 虽从 `+0.005062` 提升到 `+0.008387`，但低于第 40 轮 path baseline `+0.012520`。这和第 47 轮类似：2026 有局部线索，但长期不够厚。

第四反事实：如果这个方向值得进账户层，标签层 Top20 应先过强基线。实际开发期和 2026 都没有超过第 40 轮 `path_topq`，因此账户层验证预计只能消耗时间，不进入。

### 48.5 判定

`rejected`，不进入账户层，不合代码。

本轮直接回答了“能不能通过分析历史大涨股，用 ML 归因来找策略”：可以归因，也能在 2026 局部改善基础 ML，但它不能自然扩成合规 Top20 的一周持仓 alpha。右尾事件和左尾风险共享高弹性市场/个股状态，单独预测大涨会同步带来大跌；减去左尾后又把弹性一起削掉。

下一步不继续在 winner/loser 分类器上调阈值或融合权重。更可能的方向是把“高弹性状态”拆成可交易的二阶段问题：先识别什么时候市场/行业概念扩散足以支持高弹性右尾，再在这些状态内用路径 ranker 选股；或者反过来，在高弹性但扩散不足时降低参与。也就是说，下一轮应研究“行业/概念扩散确认下的高弹性专家”，而不是继续全样本右尾事件分类。

## 49. elastic_dispersion_concept_state_v1

### 49.1 假设

第 43 轮状态条件路径边际诊断发现，第 40 轮 `path_topq` 在高 20 日横截面离散度状态里明显更厚：开发期 due5 Top20 high-low spread 为 `+0.014441`，2026 为 `+0.020219`。第 48 轮右尾/左尾事件学习器又显示，大涨和大跌都集中在高弹性状态里；只学右尾会同步买到左尾风险。

本轮假设是：高离散状态本身不够，还需要行业/概念扩散确认。如果用训练期分位数因果定义“高离散 + 概念不弱/高概念”状态，并只在这些状态里启用第 40 轮 `path_topq` 专家，普通状态退回基础 ML 排序，可能保留弹性同时降低错误参与。

### 49.2 方法和产物

先做非因果交互诊断，再做因果阈值诊断，最后只对最自然的 `high_disp_not_low_concept` 版本进入账户层。所有脚本、预测、YAML 和回测均在 `/tmp`，未合入策略库。

关键产物：

```text
/tmp/quantx-research/elastic-dispersion-concept-state-v1/analyze_elastic_dispersion_concept_state.py

/tmp/quantx-research/elastic-dispersion-concept-state-v1/analyze_causal_elastic_state_gate.py

/tmp/quantx-research/elastic-dispersion-concept-state-v1/write_elastic_state_switch_predictions.py
sha256:45c5ea999e5474660b0dfc24a0f39a45390d2b167db014126a6c459d805eeec7

/tmp/quantx-research/elastic-dispersion-concept-state-v1/causal_elastic_state_gate_dev_2021_2025_diagnostic.json
sha256:7acc24202cb7a9d3a11fa4d1b4a934289c42f4caf552dab4d97bc4eb1df950d8

/tmp/quantx-research/elastic-dispersion-concept-state-v1/causal_elastic_state_gate_val63_2026_diagnostic.json
sha256:112fe84ed50d6de1639bef1c0c81c3f359a37d1339774e1fdc306461fa0f51e4

/tmp/quantx-research/elastic-dispersion-concept-state-v1/elastic_state_switch_hdnlc_base_dev_2021_2025_predictions.json
sha256:9708079fc116a1a5befeb1d72edb09c9015cc75a02b2dfa753e0cf87a28c8339

/tmp/quantx-research/elastic-dispersion-concept-state-v1/elastic_state_switch_hdnlc_base_val63_2026_predictions.json
sha256:9b76ba8f332cbeccdecb3b11c06a5319881f30c77bc592c466f4cc846d9cd313

/tmp/quantx-research/elastic-dispersion-concept-state-v1/elastic_state_switch_hdnlc_base_dev_2022_2025_reb5_account_diagnostic.json
sha256:77061162ea772d22420a8f0b10f85fec8f1f871f17f0a0951deb285a1ca2bb21

/tmp/quantx-research/elastic-dispersion-concept-state-v1/elastic_state_switch_hdnlc_base_val63_2026_reb5_account_diagnostic.json
sha256:d946e9bbec0e2e96e3d316b6fdb0069859d248e2de21e13895f15d1f59e11bc0
```

因果阈值来自训练期 due5 session，而不是测试期分桶：

| 阈值 | 定义 |
| --- | --- |
| `disp_high` | 训练期 due5 `market_ret20_disp` 的 2/3 分位 |
| `concept_low` | 训练期 due5 `selected20_path_concept_ret5_median` 的 1/3 分位 |
| `concept_high` | 同一概念确认特征的 2/3 分位 |
| `breadth_high/low` | 训练期 due5 `market_breadth60` 的 2/3 和 1/3 分位 |
| `vol_high` | 训练期 due5 `pool200_vol20_median` 的 2/3 分位 |

账户层版本只在 `high_disp_not_low_concept` 且 due5 调仓 session 上把排序分数切到 `rank_path_topq`；其他 session 使用 `base_ml_rank_pct`。组合仍为标准 `rebalance_interval_sessions=5`、Top20 等权、标准成本、T+1 open。

### 49.3 标签层结果

非因果交互诊断看起来很强，但会使用评估样本自身分位数，不能直接交易。它的价值只是提示方向：高离散叠加概念确认的 session 更厚。

| 口径 | 开发期 due5 sessions | 开发期 path Top20 label | 2026 sessions | 2026 path Top20 label |
| --- | ---: | ---: | ---: | ---: |
| `high_disp` 非因果 | 65 | `+0.024008` | 8 | `+0.031085` |
| `high_disp_not_low_concept` 非因果 | 33 | `+0.030635` | 6 | `+0.036965` |
| `high_disp_high_concept` 非因果 | 20 | `+0.025703` | 5 | `+0.040611` |
| `high_disp_high_breadth` 非因果 | 29 | `+0.032163` | 3 | `+0.015208` |

因果阈值后，信号显著变薄。开发期 `high_disp_not_low_concept` 仍然高于全 due5，但 2026 只小幅高于全 due5。

| 口径 | 开发期 sessions | 开发期 path Top20 label | 开发期 delta vs base | 2026 sessions | 2026 path Top20 label | 2026 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全 due5 | 195 | `+0.014170` | `+0.000646` | 24 | `+0.012520` | `+0.007458` |
| `high_disp` | 21 | `+0.028113` | `-0.000039` | 17 | `+0.011752` | `+0.005509` |
| `high_disp_not_low_concept` | 21 | `+0.028113` | `-0.000039` | 16 | `+0.012945` | `+0.006535` |
| `high_disp_high_concept` | 21 | `+0.028113` | `-0.000039` | 16 | `+0.012945` | `+0.006535` |
| `high_disp_high_breadth` | 13 | `+0.033345` | `+0.006154` | 5 | `+0.014294` | `-0.005704` |
| `high_disp_high_vol` | 8 | `+0.015312` | `-0.008771` | 7 | `+0.014149` | `+0.015345` |

开发期因果阈值里 `concept_low` 和 `concept_high` 都等于 `0.5`，所以 `high_disp_not_low_concept`、`high_disp_high_concept` 和 `high_disp` 在开发期命中完全重合。这说明当前概念确认特征离散度不足，非因果诊断里的“概念确认”有一部分来自样本分桶，而不是稳定可部署阈值。

### 49.4 账户层结果

账户层 `high_disp_not_low_concept + base fallback` 开发期基本贴近第 40 轮，但没有超越；2026 明显弱于第 40 轮。

| 方案 | 区间 | 累计收益 | 年化 | 最大回撤 | 平均持仓数 | 平均持仓天数 | 逐年收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 第 40 轮 `path_topq` | 2022-2025 | `+777.49%` | `+72.50%` | `-32.62%` | 19.40 | 9.36 | 全正 |
| 本轮状态切换 | 2022-2025 | `+769.17%` | `+72.09%` | `-32.37%` | 19.52 | 9.84 | `+52.91%、+59.95%、+53.55%、+129.06%` |
| 第 40 轮 `path_topq` | 2026-01-06 至 2026-07-10 | `+19.18%` | 未列 | `-16.70%` | 19.48 | 9.94 | `+19.18%` |
| 本轮状态切换 | 2026-01-06 至 2026-07-10 | `+11.04%` | `+22.96%` | `-17.99%` | 19.54 | 10.56 | `+11.04%` |

覆盖率也解释了为什么账户层没有明显增益：开发期 963 个 session 中只有 21 个 state-on，2026 118 个 session 中有 16 个 state-on。开发期状态太稀疏，无法改变长期账户；2026 状态更频繁，但切到 path expert 后反而弱于第 40 轮固定 path。换句话说，这个状态不是稳定择模器。

### 49.5 反事实分析

第一反事实：如果“高离散 + 概念确认”是可部署收益状态，那么因果阈值下应该保留非因果分桶的大部分优势。实际从非因果开发期 `+0.030635` 降到因果 `+0.028113`，2026 从 `+0.036965` 降到 `+0.012945`，2026 优势几乎被压平。

第二反事实：如果这个状态适合做账户切换器，2026 应该至少不弱于固定第 40 轮 path。实际状态切换 2026 为 `+11.04%`，低于固定 path `+19.18%`，最大回撤也更深。这说明状态识别未能避开 2026 的脆弱段。

第三反事实：如果概念确认是关键变量，那么 `not_low_concept` 与 `high_concept` 应该有不同覆盖和不同收益。实际因果阈值里两者在开发期完全重合，2026 也高度接近，说明当前概念确认度量不够连续或不够稳定，不能承载一个策略级开关。

第四反事实：如果高离散状态只是“机会更厚”，而不是“该切换模型”，那么标签层会更好但账户层不一定增强。实际正是如此：它解释了什么时候 path label 更厚，却没有把 fixed path 的账户表现推高。

### 49.6 判定

`rejected_with_signal`，不合代码，不继续调 gate。

本轮保留的有效观察是：20 日横截面离散度确实是机会环境变量，高离散时一周路径 ranker 的右尾更厚。但当前“高离散 + 概念确认”不能直接做硬 gate 或状态切换，原因是因果阈值后 2026 优势变薄，账户层低于固定第 40 轮。

下一轮不再继续调整 `disp_high`、`concept_low/high`、`breadth_high` 或状态开关。更有希望的方向是回到候选生成本身：寻找能在普通状态也自然变厚的股票级启动前结构，尤其是多日横截面路径、行业/概念内相对位置、成交量温和扩散和历史大涨前相似结构，而不是用市场状态去选择已有弱专家。

## 50. startup_structure_memory_v1

### 50.1 假设

第 30、41、48、50 前的多轮实验反复出现同一个问题：历史大涨股和高弹性状态可以被识别，但只要直接追 winner 或用风险惩罚过滤，就会在 Top15/20 等权层面同步削掉上行弹性。本轮换一个更结构化的反事实：不训练 winner 分类器，也不手写前兆分数，而是在基础 ML Top200 候选池内保存历史启动前结构。

具体假设是：真正可交易的启动前样本，应当在多日 RPS、量能、行业/概念相对位置和市场状态上更接近历史 top quintile 样本，同时远离历史 bottom quintile 样本。如果这个假设成立，`path_topq` 加上“离赢家近且离输家远”的记忆分数，应该在 due5 Top15/20 上超过第 40 轮 `path_topq`，并且 bottom quintile rate 下降时 top quintile rate 不应明显下降。

### 50.2 方法和产物

本轮复用第 40 轮 `path_sequence_ranker_v1` 的基础 ML Top200 面板、同日五分位 path 模型和评估口径，只新增一个 startup memory 分数：

1. 用训练期候选池内 `label_quintile == 4` 的样本拟合 winner startup centers。
2. 用训练期 `label_quintile == 0` 的样本拟合 loser startup centers。
3. 测试候选的 `startup_edge = min_dist_to_loser - min_dist_to_winner`。
4. 比较纯 `startup_edge`、`path_topq` 和 `0.75*path_topq + 0.25*startup_edge_rank`。

关键产物：

```text
/tmp/quantx-research/startup-structure-memory-v1/analyze_startup_structure_memory.py
sha256:31c11f1d2e3df6dd1967b9985a3b9d9427da4eca197edb9e13baae2924993d79

/tmp/quantx-research/startup-structure-memory-v1/smoke_startup_structure_memory_2021_2022q1.json
sha256:1ddf6407de6613fc6ed6ae7ffc7f850dbbdf5b251be2d705e0c12e5a1a58d8db

/tmp/quantx-research/startup-structure-memory-v1/startup_structure_memory_dev_2021_2025_diagnostic.json
sha256:5bcb6a38c9c67c6b67da447406f749a37ef2b596b44d9492aaf0df69e03e18af

/tmp/quantx-research/startup-structure-memory-v1/startup_structure_memory_val63_2026_diagnostic.json
sha256:1e83ef9fb781c5e850d0cea6588366961b96382b009c07d1f4bc2b25ea2c3785
```

使用的 memory columns 包括基础 ML rank、1/3/5/10/20/60/120 日 RPS、RPS 滚动均值/斜率/相对强弱、20 日量能、MA 距离、20 日高点回撤、行业/概念 5 日强度和强股比例、相对行业/概念强弱，以及 20/60 日市场收益、离散度和宽度。

### 50.3 结果

主目标没有通过。开发期和 2026 的 Top15/20 都低于第 40 轮 `path_topq`。

| 口径 | 开发期 due5 Top20 label | 开发期 top quintile | 开发期 bottom quintile | 2026 due5 Top20 label | 2026 top quintile | 2026 bottom quintile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `path_topq` | `+0.014170` | `30.21%` | `28.00%` | `+0.012520` | `36.04%` | `30.62%` |
| `path_startup_edge_25` | `+0.010994` | `26.90%` | `26.26%` | `+0.011341` | `35.63%` | `32.50%` |
| `path_startup_win_25` | `+0.010496` | `26.64%` | `25.05%` | `+0.011895` | `32.71%` | `29.17%` |
| `startup_edge` | `+0.006988` | `21.67%` | `20.67%` | `+0.005210` | `23.54%` | `21.67%` |
| `base_startup_edge_25` | `+0.013110` | `24.74%` | `20.92%` | `+0.010938` | `28.96%` | `26.25%` |

Top15/Top10 也没有出现足以推进账户层的反转：

| 口径 | 开发期 due5 Top15 label | 2026 due5 Top15 label | 开发期 due5 Top10 label | 2026 due5 Top10 label |
| --- | ---: | ---: | ---: | ---: |
| `path_topq` | `+0.016013` | `+0.018316` | `+0.017398` | `+0.023061` |
| `path_startup_edge_25` | `+0.012665` | `+0.014348` | `+0.015075` | `+0.022185` |
| `base_startup_edge_25` | `+0.014385` | 未进主排名 | `+0.017746` | 未进主排名 |

`base_startup_edge_25` 在开发期 Top10 比 base 略高，但低于 `path_topq`，且不是 Top15/20 合规收益增强。它只说明 startup memory 对基础 ML 有一点头部校正，但仍不如第 40 轮 path 模型。

### 50.4 反事实分析

第一反事实：如果历史启动前结构相似性是缺失的收益弹性层，那么 `path_startup_edge_25` 应该在 Top15/20 超过 `path_topq`。实际开发期 Top20 从 `+0.014170` 降到 `+0.010994`，2026 从 `+0.012520` 降到 `+0.011341`。

第二反事实：如果它主要是避开失败高弹性样本，那么 bottom quintile rate 应下降且 top quintile rate 不大幅下降。实际开发期 Top20 bottom 从 `28.00%` 降到 `26.26%`，但 top 也从 `30.21%` 降到 `26.90%`，净 label 变薄；2026 bottom 反而从 `30.62%` 升到 `32.50%`。

第三反事实：如果只需要近赢家，而不需要远离输家，那么 `path_startup_win_25` 应该更好。实际开发期 Top20 `+0.010496`，仍低于 `path_startup_edge_25` 和 `path_topq`；2026 Top20 `+0.011895`，略高于 edge 但仍低于 `path_topq`。

第四反事实：如果 startup memory 本身就是独立 alpha，那么纯 `startup_edge` 应该有可观 Top20 label。实际开发期只有 `+0.006988`，2026 只有 `+0.005210`，低于基础 ML 和 path 模型。

### 50.5 判定

`rejected`，不进入账户层，不合代码。

本轮否决的是“历史启动前结构近邻记忆”作为第 40 轮路径 ranker 的收益弹性层。它复现了前几轮的共同机制：历史 winner/loser 结构确实含有风险信息，但直接用于过滤或弱融合时，会把右尾和左尾一起削掉；在 A 股短周期里，能产生大收益的结构和会失败的高弹性结构距离太近，简单相似性不能把它们切开。

下一轮不再做 winner 相似性、loser 距离或状态 gate。更值得测试的是账户目标本身：第 40 轮已经有较强 Top20 label 和合规持仓，但账户五年只有约 8.77x。下一步应研究“同一 Top20 预测下，组合路径和持仓管理如何损失收益”，例如重叠持仓、止损/持有期延长、成交失败、极端亏损路径、以及是否存在由 T+1 open 到 5 日退出之间的可学习路径管理 alpha。若收益损失主要来自持仓路径而不是选股排序，再继续改排序器会边际递减。

## 51. account_path_loss_v1

### 51.1 假设

第 40 轮 `path_topq` 是当前最接近目标结构的 ML 底座：2022-2025 账户 `+777.49%`、逐年全正、平均持仓约 19.40、平均持仓约 9.36 天，2026 前向 `+19.18%`。但用户目标是五年近几十倍至 100 倍，现有账户倍率仍不足。

前几轮继续改选股排序器都没有把 Top20 明显变厚，因此本轮先暂停排序器开发，诊断一个更底层的问题：第 40 轮的标签是 T+1 open 到未来 5 个交易日 open 的收益，但真实账户平均持仓接近 9-10 天。若账户收益损耗来自持仓路径管理，那么可能存在比继续重排 Top20 更高杠杆的方向。

### 51.2 方法和产物

本轮不改交易规则，只读取第 40 轮标准回测目录中的 `closed_positions.json` 和 `selection_candidates.json`，再从 qlib open price 还原每个已平仓持仓的路径：

1. 入场后第 3/5/10/15/20 个交易日 open 收益。
2. 实际退出 open 收益和回测 closed return。
3. 持仓期内 open 口径最大浮亏 `MAE`、最大浮盈 `MFE`、从最大浮盈回吐。
4. 持仓期间该股票被连续入选的次数。
5. 按持仓长度、连续入选次数和年度分组。

关键产物：

```text
/tmp/quantx-research/account-path-diagnostics-v1/analyze_account_path_loss.py
sha256:d0924fc90b7c81f0786e9dbc70ce36c7475847e336b164281fbf2f960bb84d53

/tmp/quantx-research/account-path-diagnostics-v1/path_topq_dev_2022_2025_account_path_loss.json
sha256:5b72b87f25522ef7e70c4dc092bec287122850e9dc392042a6fa5f3e0242d56b

/tmp/quantx-research/account-path-diagnostics-v1/path_topq_val63_2026_account_path_loss.json
sha256:5c26b3e63daab44c1a15a64f34ab6b6b6337cc7a714875d1bceaef6a1de1c390
```

### 51.3 结果

第一结论是反直觉的：账户把部分持仓延长到 5 日以后，并不是开发期收益不足的主因。开发期实际退出收益均值高于第 5 日收益，且越是连续入选、持有更久的股票，平均收益越高。

| 区间 | 已平仓数 | 实际退出均值 | 第 3 日 | 第 5 日 | 第 10 日 | 第 15 日 | 第 20 日 | 实际 - 第 5 日 | MAE | MFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-2025 | 3641 | `+2.33%` | `+0.64%` | `+1.51%` | `+2.54%` | `+3.81%` | `+4.82%` | `+0.81%` | `-3.99%` | `+6.36%` |
| 2026 | 427 | `+1.85%` | `+0.47%` | `+1.42%` | `+1.54%` | `+0.79%` | `+0.36%` | `+0.43%` | `-5.26%` | `+6.84%` |

按持仓长度看，开发期延长持仓平均收益更高：

| 持仓 sessions | 开发期数量 | 开发期实际 | 开发期第 5 日 | 开发期差值 | 2026 数量 | 2026 实际 | 2026 第 5 日 | 2026 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<=5` | 3050 | `+1.78%` | `+1.78%` | `0.00%` | 330 | `+2.02%` | `+2.02%` | `0.00%` |
| `6-10` | 404 | `+3.46%` | `+0.35%` | `+3.11%` | 73 | `+3.51%` | `+0.26%` | `+3.25%` |
| `11-15` | 114 | `+7.78%` | `-0.23%` | `+8.00%` | 18 | `-3.80%` | `-3.16%` | `-0.63%` |
| `16-25` | 66 | `+11.28%` | `-0.62%` | `+11.90%` | 5 | `-10.21%` | `-1.45%` | `-8.76%` |
| `>25` | 7 | `+1.57%` | `+0.06%` | `+1.52%` | 1 | `-12.09%` | `-13.57%` | `+1.48%` |

按连续入选次数看，开发期连续入选越多，平均收益越高；2026 的 `>6` 连续入选仍改善第 5 日，但整体收益显著弱于开发期。

| 连续入选桶 | 开发期数量 | 开发期实际 | 开发期第 5 日 | 开发期差值 | 2026 数量 | 2026 实际 | 2026 第 5 日 | 2026差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0-1` | 1758 | `+1.43%` | `+1.42%` | `0.00%` | 162 | `+0.20%` | `+0.20%` | `0.00%` |
| `2-3` | 1088 | `+1.79%` | `+1.84%` | `-0.05%` | 120 | `+2.34%` | `+2.25%` | `+0.10%` |
| `4-6` | 502 | `+2.79%` | `+1.81%` | `+0.98%` | 86 | `+4.63%` | `+4.99%` | `-0.35%` |
| `>6` | 293 | `+8.94%` | `+0.34%` | `+8.61%` | 59 | `+1.33%` | `-2.10%` | `+3.44%` |

### 51.4 反事实分析

第一反事实：如果账户收益不足主要因为持仓超过 5 日，那么强制第 5 日退出应该明显更好。诊断不支持。开发期实际退出均值 `+2.33%`，第 5 日只有 `+1.51%`；延长持仓样本平均 `actual - ret5 = +5.02%`。

第二反事实：如果连续入选只是重复买入噪声，那么连续入选越多应该越差。开发期正好相反：`>6` 连续入选桶实际均值 `+8.94%`，第 5 日仅 `+0.34%`，说明持续入选在开发期确实捕捉到趋势延续。

第三反事实：如果 2026 问题和开发期一样，那么第 15/20 日远期收益也应继续抬升。实际 2026 第 10 日 `+1.54%` 后开始衰减，第 15 日 `+0.79%`，第 20 日 `+0.36%`。2026 更像“短期有效但延续弱化”，不是简单排序失效。

第四反事实：如果路径管理没有价值，那么延长持仓中的损失样本不应集中。实际延长持仓里，开发期 40.95% 的样本被延长拖累，平均损失 `-6.36%`；2026 46.39% 被拖累，平均损失 `-7.91%`。这说明不能机械延长，也不能机械 5 日卖出，可能需要学习“哪些延续值得留”。

### 51.5 判定

`diagnostic_only`。

本轮没有产生策略，但改变了下一步研究方向：第 40 轮收益不足不应简单归因于持仓周期超过标签周期。开发期趋势延续是有价值的，连续入选越多的持仓反而收益更高；2026 的问题是延续质量变弱、长持有回吐更明显。

下一轮进入 `holding_path_continuation_v1`：只用入场后已经发生的 3-5 日路径、浮盈/浮亏、连续入选、市场/候选状态，因果预测“继续持有到 10/15 日是否优于第 5 日退出”。若这个标签可学，才考虑把它转成账户层持仓管理；若不可学，则说明当前路径损耗不适合作为 ML alpha，研究应回到选股或更高频数据。

## 52. holding_path_continuation_v1

### 52.1 假设

第 51 轮发现，开发期延长持有是正贡献，但 2026 的 15/20 日后续收益明显衰减。因此本轮测试一个更细的账户目标：对第 40 轮已经入场的持仓，能否只用入场后已经观察到的短路径，识别“应该继续持有”还是“应该在第 5 日附近退出”。

本轮分两层验证：

1. **非因果单笔诊断**：在 entry+5 open 已知时，使用 entry 到 day5 open 的收益、MAE/MFE、连续入选等特征，预测继续到 day10/day15 是否优于 day5 退出。该层只看标签可学性，不作为可交易策略。
2. **因果账户层近似**：严格使用下一次调仓信号日前收盘可见的信息，即入场后第 4 个 open 和第 4 日 close 以内的路径，改写 prediction store，把模型认为应续持的老持仓重新插入下一次调仓候选，再交给标准 `decision_pipeline` 回测。

### 52.2 产物

```text
/tmp/quantx-research/holding-path-continuation-v1/analyze_holding_path_continuation.py
sha256:e6e5be393c4d6047e90df0737f61b1cee0e39f325e2878008100616533686731

/tmp/quantx-research/holding-path-continuation-v1/apply_causal_continuation_store.py
sha256:d64010cfbceee8d8fe29051f001b46d612864e02e053bf51907aefb666bdcb25

/tmp/quantx-research/holding-path-continuation-v1/holding_path_continuation_dev_2022_2025_diagnostic.json
sha256:6f654df75842897e69370f8263662067f7be9fd547704bcab358c506840f03ab

/tmp/quantx-research/holding-path-continuation-v1/holding_path_continuation_val63_2026_diagnostic.json
sha256:0dce87d852b62f92cd02104b7aa38176e921437c76a13de1aa515a2de8dff5df

/tmp/quantx-research/holding-path-continuation-v1/causal_continuation_cont10_q30_dev_2022_2025_diagnostic.json
sha256:0bf4e6895c66e3035d5af28035dfe375bf072319b202481f19800aa7e5849e3d

/tmp/quantx-research/holding-path-continuation-v1/causal_continuation_cont10_q30_val63_2026_diagnostic.json
sha256:cd8f53a164a7e79a0ed8dcd04f192a0c7d4c4e8916025671bebfc203ef56091a

/tmp/quantx-research/holding-path-continuation-v1/causal_continuation_cont10_q30_val63_2026_reb5_account_diagnostic.json
sha256:53bc877f60699d1af550bd27243fbc27748d25afb8bc5fc4b8db27d1ac4da7ef
```

非因果模型使用 `LGBMRegressor`，目标为 `cont10 = ret10 - ret5` 和 `cont15 = ret15 - ret5`。因果 store 改写版使用 `cont10`、top30% continuation，并保持标准交易口径：5 日调仓、T 日信号、T+1 open 成交、Top20 等权、标准成本、真实交易规则校验。

### 52.3 非因果单笔诊断结果

非因果层说明“续持空间存在，但模型不够强”。开发期全持到 10/15 日本身就优于第 5 日退出；模型只改善 exit5，却打不过全持。2026 中，模型对 10 日续持有局部帮助，但收益改善太小。

| 区间 | 目标 | all_exit5 | all_hold | 最好模型变体 | 模型均值 | 相对 exit5 | 相对 all_hold |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 2022-2025 | cont10 | `+1.61%` | `+2.93%` | top50 | `+2.70%` | `+1.09%` | `-0.23%` |
| 2022-2025 | cont15 | `+1.61%` | `+4.31%` | top50 | `+3.62%` | `+2.01%` | `-0.69%` |
| 2026 | cont10 | `+1.27%` | `+1.50%` | top30 | `+1.75%` | `+0.48%` | `+0.25%` |
| 2026 | cont15 | `+1.27%` | `+0.79%` | top20 | `+1.40%` | `+0.13%` | `+0.61%` |

Oracle 空间很大，2026 `cont10` oracle 均值 `+4.97%`、`cont15` oracle `+6.20%`，说明事后确实能区分续持成败；但前 5 日路径特征只能捕捉其中很小一部分。

### 52.4 因果账户层结果

非因果诊断存在一个交易时点问题：entry+5 open 是当天开盘后才知道的，而标准 pipeline 的调仓信号在前一交易日收盘生成。因此本轮又做了因果 store 改写，只使用信号日前可见路径。

因果诊断中，开发期各年度 selected continuation 仍高于 not selected：2023 `+0.86%` vs `+0.07%`，2024 `+0.96%` vs `+0.72%`，2025 `+2.00%` vs `+1.20%`。2026 selected `cont10` 也略高：`+0.22%` vs `-0.17%`。但是这点差异进入 Top20 等权账户后几乎没有净值改善。

2026 标准账户层：

| 方案 | 收益 | 最大回撤 | Sharpe | 平均持仓 | 平均持仓天数 | 交易数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第 40 轮 `path_topq` 基线 | `+19.18%` | `-16.70%` | 约 `1.74` | `19.48` | `9.94` | 约 `804` |
| 因果 continuation cont10 top30 | `+19.18%` | `-16.70%` | `1.74` | `19.48` | `9.94` | `804` |

开发期因果标准回测曾启动，但因改写 store 扩大了 `prediction_store` universe，标准引擎长时间满 CPU 未完成；考虑到 2026 标准账户已无改善，且非因果开发期也打不过全持，本轮未继续消耗时间等待该慢任务。此处保留为运行限制，不作为正向证据。

### 52.5 判定

`rejected`，不合代码。

本轮回答了第 51 轮留下的问题：持仓路径里确实存在事后可见的续持/退出空间，但用当前日线 open/close 路径和连续入选特征只能学到很弱的一部分；一旦切到严格可交易信号时点并进入账户层，2026 没有带来可见净值改善。

因此，当前收益量级不足的主瓶颈不在“第 5 日以后怎么续持/退出”。继续围绕持仓天数、续持阈值或 Top20 内老持仓插入规则调参，容易变成交易规则微调，不会自然产生几十倍收益弹性。下一步应回到更高杠杆的问题：候选生成和组合结构，尤其是能否在不削掉右尾弹性的前提下，构造比第 40 轮更厚的高收益候选土壤。

## 53. store_consensus_v1

### 53.1 假设

第 40 轮 `path_topq` 是目前最好的 ML 底座，但仍未达到收益量级。此前多轮实验显示，单一排序器继续微调容易在 Top10 有效、Top20 变薄。因此本轮换成组合结构问题：若多个低相关 Alpha 家族在同一股票上形成共识，这种股票可能既有右尾弹性，又不依赖降低尾部权重。

本轮只使用已有 prediction store，不重新训练模型。四个家族为：

1. `path`：第 40 轮 `path_topq::pool200::top20`。
2. `head_sum`：7 模型 head6 多模型头部并集 Top15。
3. `weekly`：weekly persistence `prev_top20_bonus` Top20。
4. `soil`：honest subgroup soil Top20。

### 53.2 产物

```text
/tmp/quantx-research/store-consensus-v1/analyze_store_consensus.py
sha256:9f2c076e8313f73c985185f92f583c01117faad415c8544d5f5a697a75b78d7f

/tmp/quantx-research/store-consensus-v1/write_store_consensus_predictions.py
sha256:e605957c80d47d3d0b0f71479620d7b8d61e84f37a136db44ccfbf69494ae496

/tmp/quantx-research/store-consensus-v1/store_consensus_dev_2022_2025_diagnostic.json
sha256:60ba74784aef7d1819c3f57a60efc1416465d30fccfffaf4367b2888b6136e0b

/tmp/quantx-research/store-consensus-v1/store_consensus_val63_2026_diagnostic.json
sha256:79bfc09dd205af189019b9c1fd2691bf92f2aedd2e956838804548fc6b4f90e4
```

标准账户层使用 `decision_pipeline`、5 日调仓、T 日收盘信号、T+1 open 成交、Top20 等权、标准费用。

### 53.3 结果

标签层中，四族共识比第 40 轮 path 底座更厚，且 2026 改善明显：

| 方案 | 开发期 due5 Top20 label | 2026 due5 Top20 label |
| --- | ---: | ---: |
| 第 40 轮 `path_topq` | `+0.014170` | `+0.012520` |
| 四族 `vote_sum` | `+0.016407` | `+0.022500` |
| 四族 `consensus_only` | `+0.019758` | `+0.021512` |

账户层中，2026 明显强于第 40 轮，开发期也维持较高收益，但仍远不到几十倍级别：

| 方案 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均持仓 | 平均持仓天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第 40 轮 `path_topq` | 2022-2025 | `+777.49%` | `-32.62%` | 2.62 | 19.40 | 9.36 |
| 四族 `vote_sum` | 2022-2025 | `+731.22%` | `-34.76%` | 2.28 | 约 19.5 | 约 10 |
| 四族 `consensus_only` | 2022-2025 | `+885.60%` | `-40.31%` | 2.30 | 约 13.7 | 约 10 |
| 第 40 轮 `path_topq` | 2026 | `+19.18%` | `-16.70%` | 1.74 | 19.48 | 9.94 |
| 四族 `vote_sum` | 2026 | `+28.42%` | `-14.64%` | 2.46 | 19.48 | 9.98 |
| 四族 `consensus_only` | 2026 | `+24.09%` | `-15.24%` | 2.03 | 18.93 | 9.47 |

### 53.4 判定

`retained_candidate`，但不合代码。

四族共识是目前比较干净的方向：它不是尾部降权，也不是硬 gate，而是多个低相关家族在股票级形成一致性，2026 前向也明显改善。问题是收益量级仍不足：2022-2025 最好也只是约 8-10 倍，离五年几十倍到 100 倍还有很大距离；同时 `consensus_only` 虽然收益更高，但回撤扩大到约 `-40%`，不是“鲁棒容易出收益”。

下一步不继续调 consensus 权重，而是寻找真正新的低相关右尾专家，或能解释并避开 2024/2026 脆弱段的候选生成器。

## 54. expanded_store_consensus_v1

### 54.1 假设

若四族共识有效，那么加入更多已有候选族也许能进一步提高稳定性。本轮把第 53 轮四族扩展到九族，新增机制静态、行业 leader、概念 leader、candidate quintile blend 和 near-high low-vol 等旧候选。

### 54.2 结果

扩展后标签层变差：

| 方案 | 开发期 due5 Top20 label | 2026 due5 Top20 label |
| --- | ---: | ---: |
| 四族 `vote_sum` | `+0.016407` | `+0.022500` |
| 九族 `vote_sum` | `+0.015361` | `+0.018796` |
| 四族 `consensus_only` | `+0.019758` | `+0.021512` |
| 九族 `consensus_only` | `+0.014088` | `+0.015922` |

### 54.3 判定

`rejected`。

更多家族不是自动更鲁棒。弱专家和旧失败方向加入后，会稀释四族高质量共识。后续不再做“把所有历史候选都投票”的方向；如果要扩展共识，必须先找到新的、独立且通过开发期/2026 的右尾专家。

## 55. consensus_concept_elastic_smoke_v1

### 55.1 假设

第 53 轮说明四族共识有效，但 A 股又有强行业/概念轮动特征。本轮测试一个轻量反事实：在四族 `vote_sum` Top20 内，用旧行业/概念 leader 信号做弹性修正，看看是否能自然增强 Top20。

### 55.2 结果

结果只在头部局部有效，Top20 没有稳健改善：

| 口径 | base | industry leader | concept leader |
| --- | ---: | ---: | ---: |
| 开发期 due5 Top10 | `+0.023256` | `+0.023691` | `+0.021144` |
| 开发期 due5 Top15 | `+0.018073` | `+0.019707` | 未改善 |
| 2026 due5 Top10 | `+0.025563` | `+0.030352` | 未改善 |
| 2026 due5 Top20 | `+0.022500` | `+0.022500` | `+0.022500` |

### 55.3 判定

`rejected`。

旧行业/概念 leader 可以增强 2026 Top10，但不能自然扩成 Top20。它更像头部弹性筛选器，不是用户要求的平均持仓大于 5、约一周持仓的鲁棒主引擎。

## 56. concept_inflection_v1

### 56.1 假设

复查第 55 轮后发现，旧行业/概念脚本使用了过时 CSV 字段名。当前 `industry_membership.csv` 和 `sector_membership.csv` 使用 `symbol`、`industry_code`、`sector_code` 等列，且部分概念股票是 `603938.SH` 格式。因此旧行业/概念诊断可能存在空映射风险。

本轮修正映射后重新测试更细的概念/行业信号：不是追强概念，而是在四族 `vote_sum` Top60 宽池中测试概念/行业相对强度拐点、低位二波和相对组内转强。

### 56.2 产物

```text
/tmp/quantx-research/concept-inflection-v1/analyze_concept_inflection.py
sha256:49b51a2d93fb3fb88713b23c3cab7e71e7ab125e021bac0559fd1c639cc8682f

/tmp/quantx-research/store-consensus-v1/store_consensus_4fam_vote_sum_top60_dev_2022_2025_predictions.json
sha256:79de568cc381aec177839572c723e8f670b177dd2298d540dbff31ac9106f8f5

/tmp/quantx-research/store-consensus-v1/store_consensus_4fam_vote_sum_top60_val63_2026_predictions.json
sha256:5604278dda2c617de87de703187eb1060be41ab1990cff9e65354ea67c5f26ad

/tmp/quantx-research/concept-inflection-v1/concept_inflection_on_4fam_vote_sum_top60_dev_2022_2025_diagnostic.json
sha256:19f864c5b676f934054e0dac06d4c07c8a49524c3a0256f9d046714b55aac3ee

/tmp/quantx-research/concept-inflection-v1/concept_inflection_on_4fam_vote_sum_top60_val63_2026_diagnostic.json
sha256:dac1189fa395b79ceb61857c2cf571e6a4fbde6f485a49ab1a255c4ff2e30cf1
```

修正后分组计数为 83 个行业、247 个概念。

### 56.3 结果

宽池 Top60 使 Top20 有替换空间。结果显示，概念拐点对 Top10 很强，但 Top20 不成立：

| 方案 | 开发期 due5 Top10 | 开发期 due5 Top20 | 2026 due5 Top10 | 2026 due5 Top20 |
| --- | ---: | ---: | ---: | ---: |
| base | `+0.022406` | `+0.016535` | `+0.026441` | `+0.022049` |
| `concept_inflect` | `+0.023686` | `+0.016489` | `+0.030784` | `+0.018321` |
| `industry_inflect` | `+0.022445` | `+0.016923` | `+0.027709` | `+0.019073` |
| `relative_turn` | `+0.020884` | `+0.017415` | `+0.022071` | `+0.018760` |

### 56.4 判定

`rejected_with_signal`。

本轮最重要的正结论是：修正 CSV 后，行业/概念拐点确实能在 2026 Top10 抓到更强右尾，说明 A 股主题/概念轮动仍有可用信息。负结论同样明确：一旦扩到 Top20，概念拐点会稀释收益，尤其 2026 从 `+0.022049` 降到 `+0.018321`。它不是合规 Top20 主引擎，只能作为后续头部弹性或风险诊断的候选特征。

## 57. robust_weekly_candidate_v1

### 57.1 假设

第 30 轮大涨股归因、第 48 轮右尾/左尾事件学习和第 56 轮概念拐点都出现同一个结构：能识别更尖锐的头部 winner，但 Top20 等权容易变薄。于是本轮把问题限定在更优质的四族 Top60 候选池内，不再全市场追极端 winner，而是同时学习：

1. 未来 5 日候选池内右尾 top quintile。
2. 未来 5 日候选池内左尾 bottom quintile。
3. 右尾且一周 open 路径不脆弱的 robust winner。
4. 未来 5 日超额收益回归。

主测试分数为 `base_edge_blend = 0.5 * base consensus rank + 0.5 * rank(p_winner - p_loser)`。若它成立，应同时超过四族 base Top20，并在账户层改善 2023-2025 和 2026。

### 57.2 产物

```text
/tmp/quantx-research/robust-weekly-candidate-v1/analyze_robust_weekly_candidate.py
sha256:93e6ecd7f12104c1ff1deec3f19251f0d9ee6f8b9f17df23b6d3ff33d6afd0a5

/tmp/quantx-research/robust-weekly-candidate-v1/write_robust_weekly_predictions.py
sha256:ba132ab6a3c3371a70d17c818e4a4cb0dd08871c0db1d4738d8461f5e0c731da

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_candidate_on_4fam_vote_sum_top60_dev_2022_2025_diagnostic.json
sha256:8bf956890bafed98a53a76ac5f16f9c34ee8093b6186cf38dfe19b49d97f4edb

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_candidate_on_4fam_vote_sum_top60_val63_2026_diagnostic.json
sha256:8a011b5ca78bcc618d7f589b0ff1e6bfef038e53500a8dbf909a7f947a5c9131

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_base_edge_blend_top20_dev_2022_2025_predictions.json
sha256:315e235f3b3e7d834ef637aecfaa013980a7cdeb45fdb5c70ad3823bf24b7786

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_base_edge_blend_top20_val63_2026_predictions.json
sha256:1ac0801e0ae271332c3315b3e2e3a24077f5094e6fafe91f74b1a0a8bf8deeb7

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_base_edge_blend_top20_dev_2023_2025_reb5_account_diagnostic.json
sha256:91384789ea9e19221096d6bf667a89d690342e5fcd34eb132187e547915f2976

/tmp/quantx-research/robust-weekly-candidate-v1/robust_weekly_base_edge_blend_top20_val63_2026_reb5_account_diagnostic.json
sha256:bd25baa77fc52ac5bfc578e341f2008d817008b9980e2f22112080c8ec9412cf
```

开发期模型为逐年 walk-forward，但因为 Top60 四族共识从 2022 开始，账户 store 只覆盖 2023-2025；这不能作为完整五年证据。

### 57.3 标签层结果

`base_edge_blend` 是唯一值得保留的分数。纯 `p_winner`、纯回归、纯 `p_robust` 在 Top20 上都不稳定，说明单独追大涨或单独追稳健都会削弱收益厚度。

| 方案 | 开发期 due5 Top20 | 2026 due5 Top20 | 观察 |
| --- | ---: | ---: | --- |
| base | `+0.016073` | `+0.022049` | 四族 Top60 base |
| `base_edge_blend` | `+0.017422` | `+0.023219` | Top20 小幅增厚 |
| `event_edge` | `+0.014768` | `+0.021528` | 降左尾但也削右尾 |
| `p_winner` | `+0.016084` | `+0.018132` | winner 概率不够 |
| `p_robust` | `+0.014059` | `+0.018508` | 稳健目标削弹性 |

2026 Top10/15 的改善更明显：`base_edge_blend` Top10 从 `+0.026441` 到 `+0.037842`，Top15 从 `+0.023179` 到 `+0.028365`。但用户目标要求持仓大于 5 且不靠头部集中，因此主判定仍看 Top20 和账户层。

### 57.4 账户层结果

2026 标准账户层强于第 40 轮和四族 `vote_sum`，但回撤略深于四族 `vote_sum`：

| 方案 | 区间 | 累计收益 | 最大回撤 | Sharpe | 平均持仓 | 平均持仓天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第 40 轮 `path_topq` | 2026 | `+19.18%` | `-16.70%` | 1.74 | 19.48 | 9.94 |
| 四族 `vote_sum` | 2026 | `+28.42%` | `-14.64%` | 2.46 | 19.48 | 9.98 |
| 本轮 `base_edge_blend` | 2026 | `+29.71%` | `-17.38%` | 2.58 | 19.50 | 11.02 |

开发期只能公平比较 2023-2025。按同区间 daily nav 切片：

| 方案 | 2023-2025 累计收益 | 最大回撤 | 平均持仓 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第 40 轮 `path_topq` | `+485.18%` | `-32.62%` | 19.53 | `+64.47%` | `+58.80%` | `+127.50%` |
| 四族 `vote_sum` | `+398.57%` | `-34.76%` | 19.54 | `+58.10%` | `+59.26%` | `+101.38%` |
| 四族 `consensus_only` | `+463.15%` | `-40.31%` | 13.74 | `+50.35%` | `+59.27%` | `+136.30%` |
| 本轮 `base_edge_blend` | `+405.13%` | `-34.88%` | 19.56 | `+57.59%` | `+56.00%` | `+107.19%` |

本轮账户在 2023-2025 没有超过第 40 轮，且 2024 年初最大回撤 `-34.88%`。最差闭仓里出现多笔 `-30%` 到 `-48%` 的长持有亏损，说明右尾/左尾差仍未切开强势弹性和脆弱性。

### 57.5 反事实分析

第一反事实：如果“强且不脆弱”目标真正解决了右尾/左尾绑定，`p_robust` 应该在 Top20 明显改善。实际开发期 Top20 从 base `+0.016073` 降到 `+0.014059`，2026 从 `+0.022049` 降到 `+0.018508`。稳健目标会削掉收益弹性。

第二反事实：如果右尾/左尾差是独立主 Alpha，纯 `event_edge` 应该胜出。实际它在开发期和 2026 Top20 都低于 base，说明它只能作为原排序的修正项，不能替代底座。

第三反事实：如果 `base_edge_blend` 已经是可晋级策略，它应在账户层同时超过第 40 轮开发期和 2026。实际 2026 略好，但 2023-2025 同区间输给第 40 轮，收益量级也仍是几倍，不是几十倍。

第四反事实：如果它解决了脆弱段，2024 年初和 2026 5-6 月回撤应明显改善。实际开发期最大回撤 `-34.88%`，2026 最大回撤 `-17.38%`，均未优于最稳的四族 `vote_sum`。

### 57.6 判定

`rejected_with_signal`，不合代码。

本轮保留一个重要观察：在高质量 consensus 宽池内，`原排序锚 + 右尾概率 - 左尾概率` 比纯 winner、纯 loser penalty、纯 robust target 都更合理，2026 账户也有小幅增益。但它仍没有达到目标：开发期同区间弱于第 40 轮，最大回撤没有改善，且只有三年开发期账户证据，不能证明五年全正和几十倍收益。

下一轮不再继续调 `base_edge_blend` 权重或 TopK。更可能的金子不在“候选内再预测 winner/loser”，而在对 2024/2026 这种高弹性脆弱段做更结构化的识别：哪些强势股在系统性流动性/微盘拥挤/连板退潮/行业拥挤切换中会突然失效。下一步应优先研究强势股崩塌前的横截面风险传播和拥挤出清，而不是继续提高 winner 命中率。

## 58. crowding_unwind_risk_v1

### 58.1 假设

第 57 轮没有真正切开强势弹性和脆弱性，因此本轮不再训练新选股器，而是对第 40 轮 `path_topq` 标准账户做坏周归因：如果 2024/2026 的回撤来自行业/概念拥挤退潮，那么在调仓日前应能看到概念集中、概念 20 日强度高但 5 日边际转弱、组内成员普跌等可见信号。

### 58.2 产物

```text
/tmp/quantx-research/crowding-unwind-risk-v1/analyze_crowding_unwind_risk.py
sha256:ebfd03b4c0eb1b978fac2fcb09403098cf2808d80decb77595f123af26e6862f

/tmp/quantx-research/crowding-unwind-risk-v1/crowding_unwind_path_topq_dev_2022_2025_diagnostic.json
sha256:4644d9386eaaa578766cfc92cc89e1e2be92b2818fa3d4402ff87fdf7d954ff2

/tmp/quantx-research/crowding-unwind-risk-v1/crowding_unwind_path_topq_val63_2026_diagnostic.json
sha256:fce7b6fbbf2688d06096d2a2938b8605031388d31c55f4caacfde4a27d739b24
```

本轮使用修正后的行业/概念 CSV 字段映射，并过滤非主题概念。账户收益标签为 T+1 open 执行日至未来 5 个交易日后的账户净值变化。

### 58.3 结果

静态拥挤退潮组合分数不是稳定坏周信号：

| 规则 | 开发期选中收益 | 开发期其它收益 | 2026 选中收益 | 2026 其它收益 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `crowding_unwind_q80` | `+1.34%` | `+1.21%` | `+1.21%` | `+0.55%` | 不是风险信号 |
| `industry_unwind_q80` | `+1.08%` | `+1.28%` | `+1.38%` | `+0.51%` | 方向不支持过滤 |
| `concept_drawdown_q80` | `+1.51%` | `+1.17%` | `-2.32%` | `+1.45%` | 2026 很危险，开发期反而更强 |
| `concept_ret20_high_ret5chg_low` | `+1.64%` | `+1.20%` | `+0.59%` | `+0.69%` | 样本少且无增益 |

其中最值得注意的是 `concept_drawdown_q80`：2026 选中周坏周率 `45.8%`、崩盘周率 `16.7%`，明显高于其它周；但开发期同一规则收益反而更高，崩盘率只是 `7.8%`。这说明概念组内普跌确实解释了 2026 一部分风险，但不能直接做静态 gate。

### 58.4 判定

`diagnostic_only`。

本轮证明“概念退潮”是 2026 的真实风险载体之一，但没有证明它是跨年度稳定风险。若把它写成固定过滤器，会在开发期错杀收益，甚至可能把 2026 特定风格当成普遍规律。下一步需要判断风险载体的方向是否能动态识别，而不是继续加静态概念拥挤阈值。

## 59. liquidity_crowding_risk_v1

### 59.1 假设

除概念退潮外，强势股崩塌还可能来自流动性、低价、高波动、高位追逐和量能衰竭。本轮继续沿用第 40 轮 `path_topq` 账户，检查调仓日前持仓组合的成交额分位、价格分位、20 日波动、20/60 日动量、高位接近度和量能耗尽是否能解释未来 5 日坏周。

### 59.2 产物

```text
/tmp/quantx-research/liquidity-crowding-risk-v1/analyze_liquidity_crowding_risk.py
sha256:aaf4995cdd6658612d07c312c5b2436bcd09c7481ca521db14d05cbe4401649f

/tmp/quantx-research/liquidity-crowding-risk-v1/liquidity_crowding_path_topq_dev_2022_2025_diagnostic.json
sha256:ba46c8fe7505d3711771e494ee9c99b759e17a3324fa183e3a0a00128b896cbe

/tmp/quantx-research/liquidity-crowding-risk-v1/liquidity_crowding_path_topq_val63_2026_diagnostic.json
sha256:ebe7c42d2f6a70117ad67c918b7db3566d46cb7877aada3021aff277202ee552
```

### 59.3 结果

流动性风险呈现明显风格翻转：

| 规则 | 开发期选中收益 | 开发期坏周率 | 2026 选中收益 | 2026 坏周率 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fragile_liquidity_q80` | `+0.65%` | 26.4% | `+1.22%` | 12.5% | 开发期危险，2026 不危险 |
| `low_amount_q20` | `+1.21%` | 23.8% | `+2.48%` | 12.5% | 2026 低成交额反而更强 |
| `high_vol_q80` | `+1.62%` | 20.7% | `-0.84%` | 33.3% | 2026 高波动变成风险 |
| `low_price_q20` | `+0.63%` | 23.3% | `+1.07%` | 12.5% | 开发期偏弱，2026 不弱 |

开发期的坏周更像低价、低流动性和脆弱流动性暴露；2026 的坏周更像高波动暴露。单个静态风格过滤器无法同时解释两段样本。

### 59.4 判定

`diagnostic_only`。

本轮把“风险载体会翻转”这件事钉得更清楚：开发期惩罚低流动性/低价，2026 惩罚高波动；而 2026 低成交额和高位低流动性组合反而收益更高。下一步不能写固定流动性过滤器，也不应靠人工选择 2026 风格。更合理的方向是 `dynamic_style_payoff_state_v1`：只用历史已实现风格收益，判断当前市场正在奖励还是惩罚某类风险暴露，再看这种动态状态能否解释未来坏周。

## 60. dynamic_style_payoff_state_v1

### 60.1 假设

第 58/59 轮说明，静态风险过滤会遇到风格翻转：开发期低流动性/低价偏危险，2026 则高波动和概念回撤更危险。若这些不是随机噪声，那么可以只用过去已经完成的调仓结果，动态估计当前市场正在奖励还是惩罚某类风格暴露，再判断同类高暴露组合是否更容易出现未来坏周。

本轮仍只做诊断，不写策略 gate。核心防前视边界是：风格收益状态使用过去 `24` 个已完成调仓截面的 realized style spread，并整体 `shift(1)`；高暴露阈值使用过去 `120` 个截面的 rolling median，同样 `shift(1)`。

### 60.2 产物

```text
/tmp/quantx-research/dynamic-style-payoff-state-v1/analyze_dynamic_style_payoff_state.py
sha256:55639450d069dd4dcf586b0f4947d470564191e47509fd5196d0e70128625e5e

/tmp/quantx-research/dynamic-style-payoff-state-v1/dynamic_style_payoff_path_topq_dev_2022_2025_diagnostic.json
sha256:153440068625bebf335713c78f47c8830ef372b2cd1e390b86746722499794da

/tmp/quantx-research/dynamic-style-payoff-state-v1/dynamic_style_payoff_path_topq_val63_2026_diagnostic.json
sha256:db5d93591591e2763f7ac6a6476a28613ecd3db77b38fa8633ed90fdad07098d
```

风格包括低成交额、低价格、高波动、高 20 日动量、接近 20 日高点和概念成员普跌。每期先在第 40 轮 `path_topq` Top20 内计算该风格高/低暴露股票的未来 5 日 realized spread，再用滞后 rolling mean 形成风格 payoff state。

### 60.3 结果

动态状态在 2026 能解释坏周，但开发期没有形成同方向收益惩罚：

| 规则 | 开发期选中收益 | 开发期坏周率 | 开发期崩盘率 | 2026 选中收益 | 2026 坏周率 | 2026 崩盘率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `high_vol_harmful_state` | `+1.13%` | 24.4% | 9.0% | `-0.72%` | 30.8% | 15.4% |
| `concept_drawdown_harmful_state` | `+1.36%` | 18.7% | 5.6% | `-0.48%` | 40.0% | 15.0% |
| `dynamic_style_risk_count_ge2` | `+1.37%` | 18.7% | 7.7% | `-0.16%` | 34.7% | 18.4% |

2026 中，`dynamic_style_risk_count>=2` 的其它周收益为 `+1.29%`、坏周率 `10.1%`、崩盘率 `1.4%`，说明它确实抓到了 2026 的坏周簇。可是开发期同一规则选中周收益仍为 `+1.37%`，高于其它周 `+1.15%`，只是崩盘率从 `4.5%` 升到 `7.7%`。

风格状态本身也显示 2026 与开发期不同：开发期低成交额 payoff 均值 `+0.0068`、低价 payoff 均值 `+0.0118`；2026 二者变成 `-0.0135` 和 `-0.0134`。高波动 payoff 在开发期接近零，2026 为 `-0.0133`；概念回撤 payoff 在 2026 也更负。

### 60.4 判定

`diagnostic_only`。

本轮给出一个有价值但不能直接交易的结论：动态风格收益状态能解释 2026 的一部分风险翻转，尤其高波动和概念回撤同时不被奖励时，未来坏周概率显著上升。但开发期同一状态并不对应负收益，只对应更高尾部风险。因此它不是“鲁棒且容易出收益”的主方向，不能写成策略过滤器。

下一步不继续调 state window 或 risk count 阈值。更合理的方向是回到用户提出的大涨股归因，但把标签改成“可交易的大涨路径”：不再追未来 5 日或 20 日极端 winner，而是学习未来 15/20 日右尾、同时前 5 日路径不难持有的样本，看它能否在一周 Top20 标签上自然变厚。

## 61. tradable_big_winner_path_v1

### 61.1 假设

第 30/41/48/50 轮反复说明，历史大涨股可以被 ML 归因，但直接追未来 top5% winner 或 winner 近邻，会同步买入高弹性失败样本。第 60 轮又说明动态风险状态更像解释器，不是主收益引擎。本轮因此把大涨股标签改成更贴近交易的问题：未来 15/20 日位于候选池右尾，同时前 5 日路径不难持有。

如果这个方向成立，`tradable_winner15/20` 概率或它与第 40 轮 `path_topq` 的融合，应当在开发期 due5 Top20 明显超过第 40 轮路径底座，并且 2026 不退化。

### 61.2 产物

```text
/tmp/quantx-research/tradable-big-winner-path-v1/analyze_tradable_big_winner_path.py
sha256:73c2e0b1bfd24c441c5514dd72ce14baa3a0432cc9ab25e052008264e79f0282

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_dev_2021_2025_diagnostic.json
sha256:65d0a1ccea3c1e31ed96e7922b145655994616160145b5f49cf7b65f692278a9

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_val63_2026_diagnostic.json
sha256:daca3fd84fa4a757f6ad5a73ac47dd3b9e9837cd1d394efb30744dca3a086c88
```

本轮复用第 40 轮 `path_sequence_ranker_v1` 的基础 ML Top500 候选池、路径/行业/概念/市场特征和同日 5 日五分位 path ranker。新增目标：

1. `tradable_winner15`：未来 15 日候选池内超额收益五分位为 top quintile，且前 5 日最差 open 路径不低于 `-6%`，5 日超额不低于 `-2%`。
2. `tradable_winner20`：同上，但远端目标为未来 20 日。
3. `early_failure`：未来 5 日 bottom quintile 或前 5 日最差 open 路径低于 `-7.5%`。

评分包括纯 `tw15/tw20`、`tw15_edge/tw20_edge = p_tradable_winner - p_early_failure`，以及 `path_topq` 与这些分数的 25% 弱融合。

### 61.3 结果

开发期主目标没有通过。Top10 有小幅增强，但 Top20 只是几个 bp 的微弱增量，且不稳定：

| 口径 | 开发期 due5 Top20 label5 | delta vs `path_topq` | 2026 due5 Top20 label5 | delta vs `path_topq` | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `pool200::path_topq` | `+0.012948` | `0.000000` | `+0.009286` | `0.000000` | 本轮复用口径底座 |
| `pool200::path_tw15_25` | `+0.013360` | `+0.000412` | `+0.008046` | `-0.001240` | 开发期小增，2026 变差 |
| `pool200::path_tw20_25` | `+0.012557` | `-0.000391` | `+0.010171` | `+0.000884` | 2026 小增，开发期变差 |
| `pool200::path_tw15_edge_25` | `+0.013056` | `+0.000108` | `+0.011466` | `+0.002180` | 2026 有用，开发期几乎无增量 |
| `pool200::path_tw20_edge_25` | `+0.011965` | `-0.000983` | `+0.011702` | `+0.002415` | 2026 有用，开发期变差 |
| `pool200::tw20` | `+0.013190` | `+0.000242` | `+0.012028` | `+0.002742` | 纯 20 日可交易 winner 2026 最好，但开发期仍只是小增 |

逐年看，开发期最强的 `path_tw15_25::pool200::top20` 四年都为正，但增量太薄：2022 `+0.01536`、2023 `+0.00741`、2024 `+0.01711`、2025 `+0.01358`；对应 `path_topq` 为 2022 `+0.01484`、2023 `+0.00672`、2024 `+0.01697`、2025 `+0.01329`。这不是收益量级跃迁，而是同一底座上的微小校正。

风险拆分也没有打开新的结构。以开发期 `pool200::top20` 为例，`path_topq` 的 5 日 top quintile rate 为 `30.7%`、bottom quintile rate 为 `29.8%`；`path_tw15_25` 变成 `31.3%` 和 `29.7%`，右尾略增但左尾几乎没降。`tw15_edge/tw20_edge` 能把 bottom rate 降到约 `16%-17%`，但 Top20 label 同时降到 `+0.00949/+0.00855`，仍是“削风险也削收益弹性”的老问题。

### 61.4 反事实分析

第一反事实：如果“可交易大赢家”标签解决了极端 winner 噪声，纯 `tw15/tw20` 应显著超过 `path_topq`。实际开发期 Top20 只多 `+0.00024` 到 `+0.00056`，没有数量级意义。

第二反事实：如果一周路径可持有条件能切掉早期失败，`tw*_edge` 应保留右尾同时降低左尾。实际 edge 分数确实降低 bottom rate，但同步降低 top rate 和 label，说明失败风险与上涨弹性仍然强绑定。

第三反事实：如果 2026 的改善能证明方向可晋级，开发期也应至少稳定明显为正增量。实际 2026 `path_tw20_edge_25` 有 `+0.0024` 增量，但开发期为 `-0.0010`，典型的前向局部共振。

第四反事实：如果这是新的收益弹性层，Top20 应比 Top10 更受益，因为用户要求平均持仓大于 5。实际最明显改善主要在 Top10/2026，Top20 开发期仍很薄。

### 61.5 判定

`rejected_with_signal`，不进入账户层，不合代码。

本轮说明，把大涨股标签改成“未来 15/20 日右尾且前 5 日可持有”确实比原始极端 winner 更贴近交易，也能在 2026 修正一部分左尾。但它没有解决核心矛盾：A 股短周期右尾和左尾仍共享高弹性结构；一旦显式扣除 early failure，收益弹性也被削掉；不扣除时，Top20 增量太小。

下一轮不继续调 `early_min_return`、`early_label5_floor` 或融合权重。更值得切换的方向是组合时序结构：第 40 轮账户只每 5 日使用一次预测，可能丢掉了大量日频横截面机会。下一步应测试日频滚动 sleeve/重叠持仓是否能在保持单笔约一周持有和平均持仓大于 5 的前提下，把同一个 path 底座的日频预测转化为更高资本效率；若只是换手更高但收益不增，再回到候选生成。

## 62. daily_overlap_sleeve_v1

### 62.1 假设

第 40 轮 `path_topq` 的预测是日频生成的，但标准账户每 5 个交易日只使用一次信号。若 A 股短周期横截面机会每天都在滚动出现，那么只用每 5 天一次信号可能浪费资本效率。一个更自然的组合结构是每天开一个 5 日持有 sleeve，形成约 5 个重叠 cohort；单笔持仓仍约一周，但账户每天使用新的横截面预测。

本轮不改变选股模型、不改变 Top20、不做尾部降权，只测试同一日频预测在 `stride=5` 和 `stride=1` 下的简化组合路径差异。

### 62.2 产物和边界

```text
/tmp/quantx-research/daily-overlap-sleeve-v1/analyze_daily_overlap_sleeve.py
sha256:5d5551e29168f1caa96a8c0582103cb571deedf74abf7a74b5eef0a7fcde3d65

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_dev_2022_2025_diagnostic.json
sha256:c200a99d032b0b538c2f7da58c1f5f01d83dc37db096f35cad8f93bf0d561e04

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_val63_2026_diagnostic.json
sha256:c5abb1cf825efeeb4718c937cc9aa24f1796cc10212dc1d243820866bba5bbdf

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_4fam_vote_sum_top20_dev_2022_2025_diagnostic.json
sha256:bbf5b36bc662d417a7003c8ea97c0da66bc179fde3fd031cfc325c9551ef94d2

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_4fam_vote_sum_top20_val63_2026_diagnostic.json
sha256:ef87e877606f496dd48fc837cc52bba2c973da16a9b9bfb74ed11f30b4f7fe6a
```

这是组合结构诊断，不是正式回测替代。简化模拟口径为 T 日信号、T+1 open 入场、持有 5 个交易日 open 退出；买入成本约 `0.00052`，卖出成本约 `0.00102`，接近标准成本配置，但不完全复刻正式 runner 的最小佣金、撮合细节和风险口径。

重要边界：四族共识 Top20 store 的开发期覆盖不完整，只有 `705` 个 session，其中 2023/2024/2025 分别只有 `181/140/142` 个 session；第 40 轮 path store 是完整的 `963` 个 session。因此四族 overlap 结果不能与标准账户硬横比，只能说明覆盖不完整时的模拟行为。

### 62.3 path 底座结果

第 40 轮 path store 覆盖完整，`stride=5` 近似每 5 日调仓基线，`stride=1` 为每日重叠 5 日 sleeve：

| 方案 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | Sharpe | 平均 cohort | 平均股票数 | 逐年收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `path_topq stride=5` | 2022-2025 | 7.79x | `+679.19%` | `-37.04%` | 1.81 | 1.00 | 19.98 | `+63.28%、+40.30%、+93.11%、+73.44%` |
| `path_topq stride=1` | 2022-2025 | 11.18x | `+1017.52%` | `-35.27%` | 2.25 | 4.97 | 66.47 | `+82.32%、+46.86%、+74.46%、+135.47%` |
| `path_topq stride=5` | 2026 | 1.20x | `+19.92%` | `-16.61%` | 1.45 | 0.99 | 19.83 | `+19.92%` |
| `path_topq stride=1` | 2026 | 1.28x | `+27.91%` | `-17.36%` | 2.10 | 4.80 | 53.46 | `+27.91%` |

每日重叠持仓对 path 底座是有效的：开发期最终倍数从 7.79x 到 11.18x，2026 从 `+19.92%` 到 `+27.91%`，Sharpe 也提高。它说明第 40 轮确实有一部分日频机会被 5 日一次调仓节奏浪费。

但它仍未达到目标。开发期 4 年 11.18x 折算到 5 年也只是十几倍量级，不是几十倍至 100 倍；同时平均唯一持仓数变成约 `66`，这满足大于 5，但更像把同一弱 alpha 的机会铺开，而不是找到更强的股票级收益土壤。

### 62.4 四族共识补充诊断

四族共识简化模拟结果如下：

| 方案 | 区间 | 最终倍数/收益 | 最大回撤 | 平均股票数 | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| `4fam vote_sum stride=1` | 2022-2025 | 14.65x / `+1365.16%` | `-22.33%` | 44.23 | 看起来更平滑，但 store 覆盖不完整 |
| `4fam vote_sum stride=5` | 2022-2025 | 15.93x / `+1493.11%` | `-27.32%` | 14.63 | 因 session 缺失和现金分配，不可当正式基线 |
| `4fam vote_sum stride=1` | 2026 | `+23.56%` | `-17.97%` | 46.88 | 低于同 store stride=5 |
| `4fam vote_sum stride=5` | 2026 | `+51.17%` | `-18.43%` | 19.17 | 明显受采样日期影响 |

这组结果不能支持“四族 stride=5 已经是 15x/半年 51% 策略”的结论。原因是四族 store 并非完整日频覆盖，尤其开发期中后段缺失大量 session；简化模拟会把缺失日期当作不开新仓，从而改变现金使用和持仓结构。

### 62.5 判定

`diagnostic_only`。

本轮确认了一个真实但不足够的发现：日频重叠 5 日 sleeve 能更充分使用第 40 轮 path alpha，并在开发期和 2026 都改善资本效率。这不是尾部降权，也不改变平均单笔持仓周期，因此是可以保留的组合结构线索。

但它不是最终策略方向：收益量级仍不足，回撤仍约 `-35%`，平均持仓扩到 50-60 只更像分散化利用同一 alpha，而不是找到新的强收益来源。下一步不把 overlap 写入正式策略库，而是把它作为未来候选的账户结构备选。真正需要继续找的是能在日频完整覆盖下自然更厚的 alpha：行业/概念轮动中的日频领涨扩散、相对板块强弱、以及市场风险状态下的横截面风格切换。

## 63. daily_group_diffusion_v1

### 63.1 假设

第 62 轮说明日频信号有资本效率价值，但没有找到新的股票级收益土壤。本轮回到行业/概念轮动本身，在第 40 轮 path-sequence 面板和基础 ML Top500 候选内，测试更细的日频组内传播结构：如果 A 股短周期机会沿行业/概念扩散，那么 leader、laggard、扩散质量、leader-not-chase 或 laggard-turn 这类软分数，应能在不硬过滤股票的情况下改善 Top20。

本轮只做标签层诊断，不做账户回测；所有脚本和产物仍在 `/tmp`，未合入策略库。

### 63.2 产物和边界

```text
/tmp/quantx-research/daily-group-diffusion-v1/analyze_daily_group_diffusion.py
sha256:a19266bb1f92e20080c5a9790cf05aa5b673ca55b414fff9b627bfde240a3984

/tmp/quantx-research/daily-group-diffusion-v1/daily_group_diffusion_dev_2021_2025_diagnostic.json
sha256:8a3b4e2620050ea432af5ea0badcb86dda2574941ac7ed4acf38b66c6dd749d5

/tmp/quantx-research/daily-group-diffusion-v1/daily_group_diffusion_val63_2026_diagnostic.json
sha256:081c55132a93f1800b1661a1ef4ee131712c26b6e1a2a2f8604495a395df8452
```

组分数只使用信号日前可见的行业/概念内相对位置和短期扩散代理。行业和概念仍来自当前 CSV 静态映射，因此本轮即使有效也只能作为探索性证据，不能直接晋级。

### 63.3 标签诊断结果

核心结果如下，均为基础 ML Top500 内、pool200 Top20 的未来 5 日横截面超额 label：

| 口径 | 开发期 Top20 | delta vs `path_topq` | 2026 Top20 | delta vs `path_topq` | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `path_topq` | `+0.013294` | `0.000000` | `+0.008695` | `0.000000` | 本轮基线 |
| `base` | `+0.013524` | `+0.000230` | `+0.005062` | `-0.003633` | 开发期略高但 2026 明显变差 |
| `path_group_leader_25` | `+0.008911` | `-0.004383` | `-0.003133` | `-0.011828` | leader 分数明显破坏 2026 |
| `path_group_laggard_25` | `+0.008291` | `-0.005003` | `+0.000344` | `-0.008351` | laggard 也不能扩成 Top20 |
| `path_diffusion_25` | `+0.008505` | `-0.004789` | `-0.003828` | `-0.012523` | 扩散质量最差 |
| `path_leader_not_chase_25` | `+0.008323` | `-0.004972` | `+0.001956` | `-0.006739` | 防追高仍削弱收益 |
| `path_laggard_turn_25` | `+0.010067` | `-0.003227` | `+0.006883` | `-0.001812` | 最不差，但仍低于基线 |

所有手写组内传播分数弱融合后都低于 `path_topq`，并且 2026 的破坏更明显。`laggard_turn` 是最接近基线的版本，但开发期和 2026 都没有正增量。

### 63.4 反事实分析

第一反事实：如果行业/概念日频扩散是稳定 alpha，软融合 leader 或 diffusion 分数至少应在 Top20 上有正增量。实际两者开发期分别损失 `0.44%` 和 `0.48%` 的 5 日 label，2026 损失超过 `1%`。

第二反事实：如果问题只是追高，`leader_not_chase` 应显著好于 leader。实际它在开发期仍低于基线 `0.50%`，2026 也低于基线 `0.67%`，说明手写防追高没有恢复 path 的有效结构。

第三反事实：如果组内补涨是关键，`laggard_turn` 应改善 Top20。实际它只是最不差，仍然低于基线，说明“组强 + 个股落后 + 短期拐头”的公式不足以自然生成高收益 Top20。

第四反事实：如果组扩散能解释 2026 的风险和收益，它不应在 2026 把基线打成负 label。实际 `path_group_leader_25` 和 `path_diffusion_25` 均转负，说明当前静态 group 映射和手写传播公式更像噪声注入。

### 63.5 判定

`rejected`。

本轮否定的是“手写行业/概念扩散分数弱融合 path ranker”这个具体方向，不是否定行业/概念轮动本身。现有 path ranker 已经包含多日横截面路径、相对强弱和市场状态信息，粗糙的 leader/diffusion/laggard 公式反而破坏了它筛出的收益结构。

下一步不继续调 leader、diffusion、laggard-turn 的公式或融合权重。更合理的方向是把行业/概念影响拆成两个可学习问题：先学习某个交易日哪些行业/概念有未来 5 日组机会，再学习同一组内哪些股票有相对残差收益；也就是说，从手写扩散切到可学习的组机会 + 组内相对残差诊断。

## 64. group_residual_ranker_v1

### 64.1 假设

第 63 轮否定了手写行业/概念扩散公式，但仍留下一个更合理的问题：行业/概念影响可能不是简单 leader/laggard，而是由两部分组成：

1. 股票所属行业/概念在未来 5 日是否有组机会。
2. 股票相对自己所属组是否有额外残差收益。

如果 A 股短周期机会真的沿行业/概念轮动传播，那么把 `label5` 拆成组机会和组内残差后，两个可学习五分位分类器应能比手写扩散更自然地改善 Top20，而不是只在 Top10 或 2026 小样本里闪一下。

### 64.2 产物和边界

```text
/tmp/quantx-research/group-residual-ranker-v1/analyze_group_residual_ranker.py
sha256:2bd445c101d6bf266411b19377f83d9e8fa69e944caba25cb49f217f52b6abc3

/tmp/quantx-research/group-residual-ranker-v1/group_residual_ranker_dev_2021_2025_diagnostic.json
sha256:5862d15c45e393aaccc0113dc740b86fcd41a48f6e92c72029770eed32289c28

/tmp/quantx-research/group-residual-ranker-v1/group_residual_ranker_val63_2026_diagnostic.json
sha256:656782de9a87de983bbb74fb2c62d5eb0a160a8d546c9a73b42ad0ac74fa4bd3
```

本轮复用第 40 轮 path-sequence 面板，基础候选为 ML Top500。对每个 session，先用候选内所属行业/概念成员的平均 `label5` 构造股票的 `target_group_label5`，再用 `label5 - target_group_label5` 构造 `target_residual_label5`。两者分别转成同日五分位，用 LightGBM 分类器学习。排序口径包括纯 `group_topq`、纯 `residual_topq`、以及与 `path_topq` 的 20%/35% 弱融合。

行业和概念仍是当前静态 CSV 映射，不是 point-in-time 历史成员；本轮只做标签层诊断，不做账户回测，不合代码。

### 64.3 标签诊断结果

主口径为 due5、pool200、Top20：

| 口径 | 开发期 Top20 | delta vs `path_topq` | 2026 Top20 | delta vs `path_topq` | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `path_topq` | `+0.013294` | `0.000000` | `+0.008695` | `0.000000` | 基线 |
| `path_group_20` | `+0.008736` | `-0.004558` | `+0.002253` | `-0.006442` | 组机会单独很弱 |
| `path_residual_20` | `+0.011753` | `-0.001541` | `+0.005197` | `-0.003498` | 组内残差也低于 path |
| `path_group_residual_20` | `+0.012485` | `-0.000809` | `+0.009665` | `+0.000970` | 2026 小增，开发期变薄 |
| `path_group_residual_35` | `+0.011326` | `-0.001968` | `+0.009093` | `+0.000398` | 融合更重后开发期更差 |
| `path_group_residual_min_25` | `+0.010839` | `-0.002455` | `+0.003044` | `-0.005651` | 要求组机会和残差同时强会明显削弱 |
| `group_topq` | `+0.006322` | `-0.006972` | `+0.001537` | `-0.007158` | 独立组机会不是收益引擎 |
| `residual_topq` | `+0.010659` | `-0.002635` | `+0.005843` | `-0.002852` | 独立残差不足 |

全日频 pool200 Top20 更接近，但仍没有正增量：开发期 `path_group_residual_20` 为 `+0.011997`，低于 `path_topq` 的 `+0.012101`；2026 为 `+0.013059`，也低于 `path_topq` 的 `+0.013119`。

局部亮点在 Top15/2026：`due5::pool200::path_group_residual_20::top15` 开发期为 `+0.015224`，比 `path_topq` 高 `+0.001021`；2026 为 `+0.015301`，比 `path_topq` 高 `+0.005832`。但 Top15 的改善不能抵消 Top20 主口径失败，也没有账户层收益量级证据。

### 64.4 反事实分析

第一反事实：如果行业/概念组机会本身是强收益土壤，`group_topq` 或 `path_group_20` 应明显提高 Top20。实际开发期分别低于 path `0.70%` 和 `0.46%` 的 5 日 label，2026 也低于 path `0.72%` 和 `0.64%`。

第二反事实：如果关键在“强组内选强股”，`residual_topq` 或 `path_residual_20` 应稳定改善。实际开发期 `path_residual_20` 低于 path `0.15%`，2026 低于 `0.35%`，说明组内残差模型没有独立承载 Top20 收益。

第三反事实：如果组机会和组内残差互补，20% 融合应至少不损伤开发期。实际 `path_group_residual_20` 在开发期 due5 Top20 低于基线，且 top-quintile rate 持平、bottom-quintile rate 从 `29.8%` 升到 `30.3%`，不是更厚、更安全的结构。

第四反事实：如果 2026 的小幅改善能证明方向可保留，开发期全日频或 Top20 主口径应同步不差。实际全日频 Top20 开发期和 2026 都略低于 path，说明 2026 due5 Top20 的 `+0.00097` 更像局部修复，而不是稳定新 alpha。

### 64.5 判定

`rejected_with_signal`，不进入账户层，不合代码。

学习式组机会 + 组内残差比第 63 轮手写扩散更接近正确问题，也确实在 2026 和 Top15 上有局部信号。但它没有通过用户关心的合规 Top20/平均持仓约一周主口径：开发期 Top20 变薄，独立组机会和独立残差都打不过 path。

这轮之后不继续围绕静态行业/概念映射做组机会公式、组内残差权重或阈值微调。更高层的结论是：当前可用的行业/概念信息更像风险解释层和局部修复层，不是足够强的独立收益引擎。下一步应回到收益厚度本身：测试日频重叠结构下 Top10/Top15 等权是否已经满足平均持仓大于 5，并寻找能与 path 正交的右尾专家，而不是继续在组特征上挤几个 bp。

## 65. daily_overlap_topk_ladder_v1

### 65.1 假设

第 62 轮只测试了第 40 轮 path Top20 的日频重叠 sleeve，结果从周频约 7.79x 提升到 11.18x，但收益量级仍不足。第 64 轮又说明，继续在静态行业/概念组信息里挤增量很难自然增厚 Top20。

本轮回到一个更直接的反事实：第 40 轮 path ranker 的收益是否主要集中在头部 Top5/Top8，而日频重叠后即使只取 Top5/Top8 等权，也能通过 5 个 cohort 自然满足平均持仓大于 5。这个方向不是降低 Rank6-10 权重，而是选择更高置信的等权 TopK，并用日频重叠保持一周持仓和分散度。

### 65.2 产物和边界

本轮复用第 62 轮简化 sleeve 脚本，不改正式代码：

```text
/tmp/quantx-research/daily-overlap-sleeve-v1/analyze_daily_overlap_sleeve.py
sha256:5d5551e29168f1caa96a8c0582103cb571deedf74abf7a74b5eef0a7fcde3d65
```

关键输出：

```text
/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top5_dev_2021_2025_diagnostic.json
sha256:a5693ad4e9a6bc4d6b24db5ea0cd0855aca47cce434398ac17c3a7494cbc2e15

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top8_dev_2021_2025_diagnostic.json
sha256:d79bf633b493fc928515e25f34ae76642a448cbb501f6f1d8abf7ed8ef03eb33

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top10_dev_2021_2025_diagnostic.json
sha256:2584f808e262236437ce831773c5d3facc1dcd3f980dd4a925e32c09c299fd75

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top5_val63_2026_diagnostic.json
sha256:307ebaf6f0e6d3c909d73036f9b95ce1fc2a041606212ea8bba9376e6417a745

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top8_val63_2026_diagnostic.json
sha256:c7694af2546fc8ba296b4d6fd288e260e8e106b2cf7f36fb04b0ba1cddc34708

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top10_val63_2026_diagnostic.json
sha256:f6ae1478a3e05a03141c32316184628a69c1296e0e6183373548799ffd035029

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top5_dev_2022_2025_doublecost_diagnostic.json
sha256:274332e7ed46aff15bfd0acdca1f888c947a9e2d41496e3d873fe054d53b81b3

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top8_dev_2022_2025_doublecost_diagnostic.json
sha256:f4cd9fb8d91728f3c3640d1678c8dae4a39803838836542b2b87eb5badea20c8

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top5_val63_2026_doublecost_diagnostic.json
sha256:7d48b1856dd8df0834e99b1552d60e544b2ae038d371896ba381f33b4507c923

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_path_topq_top8_val63_2026_doublecost_diagnostic.json
sha256:5780417b34fceeba016ff0bb41f828c3f8c3a2016a749b2e9f5606b079814afb
```

重要边界：第 40 轮 path prediction store 虽然文件名含 `dev_2021_2025`，实际可用 session 只有 2022-2025：2022/2023/2024 各 242 个 session，2025 为 237 个 session。因此本轮不能声称已经通过“五年全正”，只能声称 2022-2025 四年开发期全正，并通过 2026 半年前向。

另一个边界是模拟口径：这是简化 sleeve 账户，T 日信号、T+1 open 入场、持有 5 个交易日、近似买卖成本；还没有进入正式账户引擎的撮合、最小佣金、风险约束和组合账本复核。

### 65.3 TopK 收益台阶

标准成本、stride1、持有 5 日的结果如下：

| TopK | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Top5 | 2022-2025 | 123.89x | `+12288.99%` | `-43.98%` | 17.65 | `+529.73%、+96.88%、+210.16%、+216.33%` |
| Top8 | 2022-2025 | 48.10x | `+4710.19%` | `-35.40%` | 27.82 | `+279.29%、+90.43%、+140.59%、+172.22%` |
| Top10 | 2022-2025 | 29.02x | `+2802.05%` | `-34.29%` | 34.33 | `+198.07%、+80.39%、+106.99%、+156.25%` |
| Top15 | 2022-2025 | 14.56x | `+1355.81%` | `-35.21%` | 50.73 | `+109.94%、+58.29%、+80.28%、+139.26%` |
| Top20 | 2022-2025 | 11.18x | `+1017.52%` | `-35.27%` | 66.47 | `+82.32%、+46.86%、+74.46%、+135.47%` |
| Top5 | 2026 val63 | 2.03x | `+102.78%` | `-15.71%` | 14.22 | `+102.78%` |
| Top8 | 2026 val63 | 1.72x | `+71.67%` | `-15.67%` | 22.44 | `+71.67%` |
| Top10 | 2026 val63 | 1.57x | `+56.51%` | `-16.52%` | 27.57 | `+56.51%` |
| Top15 | 2026 val63 | 1.38x | `+38.00%` | `-17.43%` | 40.30 | `+38.00%` |
| Top20 | 2026 val63 | 1.28x | `+27.91%` | `-17.36%` | 53.46 | `+27.91%` |

这个收益台阶非常清晰：越靠 path ranker 头部，收益越厚；但即使 Top5，因为每天开一个 5 日 sleeve，平均唯一持仓仍约 17.65，满足大于 5 的硬约束。持仓周期仍由 sleeve 固定为约 5 个交易日。

### 65.4 双倍成本压力

双倍成本下，Top5 和 Top8 仍然保持强收益：

| TopK | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Top5 | 2022-2025 | 92.25x | `+9125.32%` | `-45.09%` | 17.65 | `+485.58%、+82.85%、+188.09%、+193.91%` |
| Top8 | 2022-2025 | 35.80x | `+3479.79%` | `-36.68%` | 27.82 | `+252.66%、+76.85%、+123.44%、+152.86%` |
| Top5 | 2026 val63 | 1.96x | `+95.60%` | `-16.05%` | 14.22 | `+95.60%` |
| Top8 | 2026 val63 | 1.66x | `+65.58%` | `-16.01%` | 22.44 | `+65.58%` |

双倍成本没有打掉核心收益，说明这不是单纯交易成本幻觉。Top8 的分散度更高、回撤更浅，但收益量级低于 Top5；Top5 更接近用户 100x 目标，但最大回撤也更深。

### 65.5 反事实分析

第一反事实：如果第 62 轮 Top20 overlap 的收益不足只是因为日频结构不够，而不是选股头部足够强，那么收窄 TopK 不应带来单调收益台阶。实际 Top5/8/10/15/20 的收益几乎单调下降，说明 path ranker 的头部置信度有很强的经济含义。

第二反事实：如果 Top5 只是违反“平均持仓大于 5”的集中持仓技巧，那么日频重叠后的平均唯一持仓应该仍接近 5 以下。实际 Top5 平均唯一持仓 17.65，2026 为 14.22，符合约束；这不是降低 Rank6-10 权重，而是用多个日频 cohort 让高置信 Top5 自然分散。

第三反事实：如果收益来自交易成本或现金分配漏洞，双倍成本应大幅坍塌。实际 Top5 双倍成本仍为 92.25x，2026 仍为 1.96x；Top8 双倍成本也为 35.80x，2026 为 1.66x。成本敏感，但不致命。

第四反事实：如果它只是开发期过拟合，2026 val63 应明显衰减或转负。实际 2026 半年 Top5/8/10 全部强正，并且收益台阶仍保持，说明它至少通过了当前前向段。

第五反事实：如果它已经可以直接晋级，应该已经满足五年全正并能在正式账户引擎复现。实际当前 path store 缺 2021，且本轮还只是简化 sleeve 模拟，不能直接合代码。

### 65.6 判定

`retained_high_priority`，但不晋级、不合代码。

这是目前最接近用户目标的自然 ML 方向：第 40 轮 path ranker 的头部 Top5/Top8 在日频重叠一周 sleeve 下，形成了几十倍至百倍级别的四年开发期收益、逐年全正、平均持仓大于 5、2026 前向强正，并通过双倍成本压力。它不依赖尾部降权，也不是行业/概念手写公式，而是“多日横截面路径 ranker 的头部置信度 + 日频滚动机会使用率”。

但证据边界同样明确：

1. 缺 2021，尚未满足“五年全正”的完整证据。
2. 最大回撤仍约 `-35%` 至 `-45%`，不算轻松鲁棒。
3. 简化 sleeve 模拟需要正式账户引擎复核，包括撮合、最小佣金、停牌/涨跌停可交易性、真实换手和资金分配。
4. path store 本身来自前面多轮研究选择，2022-2025 已不是 sealed holdout。

下一步把它作为高优先级候选复核：先寻找或生成能覆盖 2021 的同类 path/head prediction store，验证五年；再把 Top5/Top8 日频重叠一周 sleeve 写成 `/tmp` 正式账户复核脚本，而不是直接合入策略库。

## 66. full_coverage_base5d_overlap_top5_v1

### 66.1 假设

第 65 轮证明第 40 轮 path ranker 头部 Top5/Top8 在日频重叠一周 sleeve 下非常强，但 path prediction store 缺 2021。本轮换成完整覆盖 2021-2025 的基础 5 日 ML Top5/Top10 store，做同一 overlap 口径的可复制性反事实。

核心问题：高收益是否来自 path 特有的后续 ranker，还是更基础的“5 日 ML 头部信号 + 日频滚动使用率”？如果基础 5d Top5 也能五年全正、几十倍以上、平均持仓大于 5，并通过 2026，那么它比 Exp65 更接近用户目标。

### 66.2 产物和口径

本轮继续只使用 `/tmp` 复核脚本，不合正式代码。脚本在第 62 轮基础上增加两个压力参数：

1. `--max-open-gap 0.25`：如果持仓路径中某只股票任一日 open 相对前一日 close 的跳变超过 25%，该股票按 1.0 倍处理，不贡献收益或亏损，用来压制复牌/复权/数据异常。
2. `--min-history-days 120`：入场日前至少有 120 个有效 open 历史，用来压制上市初期和复牌早期暴露。

脚本和关键产物：

```text
/tmp/quantx-research/daily-overlap-sleeve-v1/analyze_daily_overlap_sleeve.py
sha256:39244e0c3fe5de4f104e336d25eac4c89efabaf17f6a5014c60d17e51d891d25

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_diagnostic.json
sha256:ee87b639504559722f58e435c46c031cd72125efd9563ddb00313e39a42c44c0

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_doublecost_diagnostic.json
sha256:db9e12167952b3639b1397268fd966c969121ef0695a16b70f9afa7030f349ba

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_gap25_diagnostic.json
sha256:768473c15b254cb538ee346443f79922dda33ac8725d45838c6e8fd2af6c1c59

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_doublecost_gap25_diagnostic.json
sha256:7703c79f521c771b2d754ecfb2fb588dec915cb3e97eca4a10c4fc51c15327dc

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_gap25_hist120_diagnostic.json
sha256:85c85a15d13ba3ac388d60adcce20a28ddb4a25e1ff50d447df05585559c59c5

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_top5_dev_2021_2025_doublecost_gap25_hist120_diagnostic.json
sha256:715b72ebdfcb7f2e78ece4db921a838c25cec79cda011020ba98c436536543e7

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_val63_top5_2026_diagnostic.json
sha256:1ef5e46617b4afcf84097f8ba202c10760eb5db26399fbb24ce71bc77ed7f6b3

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_val63_top5_2026_doublecost_diagnostic.json
sha256:6a9c14a2ed1d7574f52394c9d7e30a8e7a4615d66bb60f2420279c86da88289b

/tmp/quantx-research/daily-overlap-sleeve-v1/daily_overlap_base5d_val63_top5_2026_gap25_hist120_diagnostic.json
sha256:6cbcc88ad9dc9dbd2b889d986f3ef3cbe006f2a1e095262e51920fe165226499
```

基础预测覆盖审计：`base_5d_top5_predictions.json` 覆盖 2021-2025 共 1212 个 session，逐年为 2021 `243`、2022 `242`、2023 `242`、2024 `242`、2025 `243`。2026 使用 `adaptive-training-protocol-v1/val63_2026_5d_top10_predictions.json`，覆盖 124 个 session。

### 66.3 原始结果和异常跳变审计

原始基础 5d Top5 日频重叠结果非常高：2021-2025 最终 3055.78x，五年全正，平均唯一持仓 16.46，最大回撤 `-42.53%`。双倍成本仍为 2112.28x。

但原始曲线存在需要压制的非标准跳变。最典型的是 2022-08-22，`SZ000670` 从 open `2.25` 跳到 `11.02`，在多个 active cohort 中贡献约 4.9 倍价格比，使组合单日收益达到 `+75.74%`。这类收益不能直接视为普通可交易 alpha。

启用 `gap25` 后，五年仍为 1608.11x，双倍成本为 1111.49x；2022 年收益从 `+549.05%` 降至 `+289.09%`，说明异常跳变确实夸大收益，但不是高收益的唯一来源。

### 66.4 保守压力结果

更可信的主证据采用 `gap25 + min_history_days=120`：

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 原始 | 2021-2025 | 3055.78x | `+305477.84%` | `-42.53%` | 16.46 | `+3028.93%、+549.05%、+88.22%、+153.91%、+205.36%` |
| 原始双倍成本 | 2021-2025 | 2112.28x | `+211127.60%` | `-43.59%` | 16.46 | `+2808.97%、+502.92%、+74.80%、+135.84%、+183.69%` |
| `gap25` | 2021-2025 | 1608.11x | `+160711.34%` | `-42.53%` | 16.46 | `+3028.93%、+289.09%、+88.22%、+122.91%、+205.35%` |
| `gap25` 双倍成本 | 2021-2025 | 1111.49x | `+111048.62%` | `-43.59%` | 16.46 | `+2808.97%、+261.41%、+74.80%、+107.04%、+183.67%` |
| `gap25+hist120` | 2021-2025 | 93.04x | `+9203.64%` | `-42.81%` | 15.22 | `+330.69%、+82.67%、+66.51%、+122.17%、+204.91%` |
| `gap25+hist120` 双倍成本 | 2021-2025 | 65.47x | `+6446.69%` | `-43.87%` | 15.22 | `+307.09%、+69.81%、+54.73%、+106.36%、+183.27%` |
| 2026 val63 原始 | 2026-01-06 至 2026-07-10 | 1.49x | `+49.27%` | `-20.52%` | 13.84 | `+49.27%` |
| 2026 val63 双倍成本 | 2026-01-06 至 2026-07-10 | 1.44x | `+43.96%` | `-20.96%` | 13.84 | `+43.96%` |
| 2026 val63 `gap25+hist120` | 2026-01-06 至 2026-07-10 | 1.49x | `+48.52%` | `-21.22%` | 13.72 | `+48.52%` |

保守口径已经满足用户提出的核心数值目标：五年几十倍至 100 倍、五年全正、平均持仓大于 5、平均持仓周期约一周、2026 前向为正。双倍成本后仍为 65.47x。

### 66.5 反事实分析

第一反事实：如果 Exp65 的高收益只是 path ranker 特有，基础 5d Top5 不应复现。实际基础 5d Top5 在完整 2021-2025 覆盖下更强，说明真正的结构是“5 日 ML 头部置信度 + 日频重叠使用率”，path ranker 不是必要条件。

第二反事实：如果 3055x 主要来自复牌/复权异常，`gap25` 应把收益打到普通水平。实际 `gap25` 后仍有 1608x，说明异常跳变会夸大，但不是核心来源。

第三反事实：如果高收益主要来自上市初期/次新暴露，`gap25+hist120` 应显著坍塌。实际收益从 1608x 降到 93x，但仍满足目标，说明次新/早期历史暴露贡献很大，却不是全部 alpha。

第四反事实：如果只是 2021 单年泡沫，2022-2025 在保守口径下应明显不稳。实际 2022-2025 分别为 `+82.67%、+66.51%、+122.17%、+204.91%`，全部为正。

第五反事实：如果成本是主要幻觉，双倍成本应明显跌出目标。实际 `gap25+hist120` 双倍成本仍为 65.47x，五年全部正。

第六反事实：如果 2026 风格已经失效，2026 val63 应转弱或转负。实际 2026 半年 `gap25+hist120` 为 `+48.52%`，虽然最大回撤 `-21.22%` 不浅，但方向通过。

### 66.6 判定

`retained_high_priority`，仍不合代码。

这是目前第一个完整触达用户硬目标的 ML 结构：基础 5 日 ML Top5，日频开仓、每笔持有 5 个交易日、多个 cohort 重叠，且加入异常开盘跳变和上市历史压力后仍然有 93x 五年收益、双倍成本 65x、五年全正、平均持仓约 15、2026 强正。

但它还不能直接晋级：

1. 当前仍是简化 sleeve 模拟，不是正式账户引擎。
2. 最大回撤约 `-43%`，风险不轻，需要进一步做账户级风险归因。
3. 当前 QMT 股票池不是 point-in-time 全历史股票池，仍有生存者偏差风险。
4. `min_history_days` 用有效 open 近似上市历史，不等于正式上市日期。
5. 需要确认正式回测中涨跌停、停牌、最小佣金、成交价和资金分配后是否仍能复现。

下一步不再发散新 alpha，先做正式账户口径复核：在 `/tmp` 写一个专门的日频重叠 Top5 账户复核脚本或配置，尽量复用正式引擎成本/交易规则；复核通过后，再准备合代码方案并向用户确认。

## 67. formal_overlap_account_replay_v1

### 67.1 假设

第 65/66 轮的简化 sleeve 仿真显示，日频重叠一周持有能把 5 日 ML 头部信号放大到几十倍乃至百倍量级。但简化仿真没有完整处理正式账户中的手数、最低佣金、滑点、停牌、涨跌停、价格跳变保护、T+1 和成交失败后的现金闲置。

本轮做正式账户口径反事实：如果 Exp66 的 93.04x 是真实可交易结构，那么在复用 QuantX `Account`、`Executor`、`AStockExchange` 和 `TransactionCost` 后，收益不应塌到个位数，更不应在 2026 转负。

### 67.2 产物和口径

本轮仍只在 `/tmp` 下实现，不合正式代码。脚本用“主现金 + 每个 cohort 一个正式 Account”的结构，避免同一股票跨 cohort 被单账户合并后无法精确卖出到期层。每个交易日：

1. 处理到期 cohort，使用正式 `Executor` 尝试卖出，遇跌停/停牌则延迟。
2. 用 T 日 prediction store 的 T+1 open 建立新 cohort。
3. 买入前使用 `gap25`、`min_history_days=120` 和正式可交易性预过滤。
4. 买入数量按 100 股整手，成本使用正式 `TransactionCost`：佣金、最低佣金、印花税、过户费、滑点。
5. 输出每日总权益、拒单原因、成交成本、平均持仓和逐年收益。

关键产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/analyze_formal_overlap_account.py
sha256:3f8208a78d5c9c25710f6d3787e2de866fdd75450e77710da6e41203c4b0f5cc

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_base5d_top5_dev_2021_2025_gap25_hist120_dynamic_diagnostic.json
sha256:efcaaecbf1e6b001b6a865fba5509fbd91234d8b660c35ec3ec65bc3edb7cd13

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_base5d_val63_top5_2026_gap25_hist120_dynamic_diagnostic.json
sha256:5b2b9362f8aaba50dc0fdd3f34462d5376a4865173b129f511003630a11b4f99

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:534e1f387b4c767dc0269a6eba8261a0cc582e5f198d589eca337e4c4d48f16a

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:ea3ab9c64b43627bb1526f2c4221152d8e83adea24d5bf2567b66ede53777d14
```

### 67.3 正式账户结果

核心结果如下：

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 平均持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base5d Top5 | 2021-2025 | 7.52x | `+652.34%` | `-33.11%` | 13.67 | 7.71 | `+91.24%、+31.57%、+26.01%、+12.21%、+109.18%` |
| base5d Top5 | 2026 val63 | 0.93x | `-6.65%` | `-20.03%` | 12.15 | 7.58 | `-6.65%` |
| path Top20 池补位买5 | 2022-2025 | 5.48x | `+447.55%` | `-37.35%` | 18.00 | 7.69 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| path Top20 池补位买5 | 2026 val63 | 1.19x | `+19.47%` | `-17.42%` | 14.52 | 8.79 | `+19.47%` |

正式账户版大幅低于简化 sleeve：Exp66 保守简化口径是 93.04x，正式账户只剩 7.52x；更关键的是 2026 从简化 `+48.52%` 变为 `-6.65%`。这已经推翻了“base5d Top5 overlap 可以直接合代码”的假设。

path Top20 池补位买5 不是几十倍结构，但比 base5d 更稳：开发期 2022-2025 为 5.48x，2026 为 `+19.47%`。不过它缺 2021，且收益量级仍远低于用户目标。

正式账户拒单解释了简化仿真的主要乐观来源。base5d 2021-2025 有 `757` 次拒单，其中 `limit_up 219`、`price_jump 167`、`suspended 350`、`limit_down 21`；2026 有 `99` 次拒单。也就是说，简化仿真把相当一部分“看得到但买不到/卖不出/价格跳变”的收益计进了组合。

### 67.4 反事实分析

第一反事实：如果简化 sleeve 的高收益主要来自可交易股票的真实开盘价延续，正式账户约束不应使 2026 从 `+48.52%` 变成 `-6.65%`。实际转负，说明 base5d Top5 很大一部分 2026 账面 alpha 与正式可交易约束冲突。

第二反事实：如果最小佣金和整手只是小摩擦，五年倍率不应从 93.04x 降到 7.52x。实际降幅巨大，说明资金分配、买不到高分票、现金闲置和拒单后的替代买入共同构成核心损耗。

第三反事实：如果 path ranker 只是简化仿真的幸运版本，正式账户应同样崩塌。实际 path Top20 补位买5 在 2026 仍为 `+19.47%`，说明 path 结构比 base5d 更抗正式成交约束，但收益弹性不足。

第四反事实：如果通过“买不到就补位”能完全修复问题，path 补位买5 应回到几十倍。实际只有 5.48x，说明可交易性补位只修复执行损耗，不会凭空创造厚收益。

### 67.5 判定

`base5d_overlap_top5` 从 `retained_high_priority` 下调为 `rejected_after_formal_replay`。

正式账户复核推翻了把 Exp66 直接合代码的可能。它仍然揭示了一个重要市场事实：5 日 ML 头部信号在简化开盘收益上很厚，但正式 A 股交易规则会显著压缩这类收益，尤其在 2026。

`path_top20_pool_buy5_overlap` 保留为 `diagnostic_baseline`，不是最终策略。它说明更稳的方向可能不是“更尖锐地买 Top5”，而是要在候选生成阶段显式学习可交易、可持有、不过度脆弱的短期右尾，而不是事后用执行补位修补。

## 68. big_winner_path_rerank_v1

### 68.1 假设

用户提出“通过分析历史上大幅上涨股票，用 ML 归因”。第 63-67 轮显示，手写行业/概念扩散、组残差和简化 overlap 都不足以形成可合并策略。本轮把已有历史大涨股归因模型接到 path-sequence Top20 候选池上：不直接全市场买“大涨概率”，而是在已较强的 path 候选池内做自然重排。

核心问题：大涨股归因能否成为低相关右尾专家，补上 path 模型在正式账户下收益不够厚的问题？

### 68.2 产物和口径

大涨股模型来自 `/tmp/quantx-research/big-winner-attribution-v1/analyze_big_winner_attribution.py`：用未来 20 日横截面前 5% 赢家训练 LightGBM classifier，特征包含市场 5/20/60 日收益、离散度、宽度，个股 1-60 日 RPS、波动、量能、近高/近低，以及行业/概念强度。

桥接脚本只在 `/tmp` 生成 prediction store：

```text
/tmp/quantx-research/big-winner-path-rerank-v1/write_big_winner_path_rerank_predictions.py
sha256:c296222fd9b325ca357e1dccf1801d2c5117711bd9758dbce2a67fb4484f2e03

/tmp/quantx-research/big-winner-path-rerank-v1/big_winner_path_h20_w075_pool20_dev_2022_2025_top20_predictions.json
sha256:79b6d3f8fb32fddc39f6bb372bb204eb1893c37e66a39313dada3187228e21e4

/tmp/quantx-research/big-winner-path-rerank-v1/big_winner_path_h20_w075_pool20_val63_2026_top20_predictions.json
sha256:feb1f7fe0439c2dfd17b49ef86c66dda0f0645aac2771ea90409f4a6518e60bc

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_bigwinner_path_h20_w075_pool20_buy5_dev_2023_2025_gap25_hist120_dynamic_diagnostic.json
sha256:2b56e7685e2697f17f74f9b02aee296d0e1e12cf38b7a7af176e537c00c7511e

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_bigwinner_path_h20_w075_pool20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:08275dd3a93b656f77fcc9bca6a694e8fd10392042e372adf5be32d7d93e9ddb
```

排序公式为：`path_pct + 0.75 * big_winner_pct`，在 path Top20 池内重排，正式账户仍使用 Top20 池补位买5、持有 5 个交易日、`gap25+hist120`、正式交易成本。

dev 只能覆盖 2023-2025，因为 2022 需要 2021 作为第一年训练；2026 使用 2021-2025 训练后 forward apply。

### 68.3 结果

正式账户结果：

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 平均持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| path Top20 补位买5 baseline | 2023-2025 | 3.36x | `+236.48%` | `-36.54%` | 17.72 | 7.70 | `+24.00%、+30.85%、+109.36%` |
| big-winner path 重排 | 2023-2025 | 3.35x | `+235.36%` | `-37.05%` | 16.98 | 7.70 | `+21.13%、+47.53%、+88.91%` |
| path Top20 补位买5 baseline | 2026 val63 | 1.19x | `+19.47%` | `-17.42%` | 14.52 | 8.79 | `+19.47%` |
| big-winner path 重排 | 2026 val63 | 1.20x | `+19.99%` | `-17.42%` | 14.37 | 8.77 | `+19.99%` |

大涨股重排对 2026 有极小增益，但开发期 2023-2025 略弱于原始 path baseline。它把 2024 从 `+30.85%` 提升到 `+47.53%`，但牺牲了 2023 和 2025，尤其 2025 从 `+109.36%` 降到 `+88.91%`。

归因特征依然稳定：不同 fold 的重要特征反复落在 `vol60_rank`、`market_ret60_disp`、`market_ret60_median/p80`、`ml_ret60_rank`、`breadth20/60`、`low60_dist_rank`、`high60_dist_rank`。这说明“大涨股”更多是一种中期市场离散度和个股波动/趋势位置状态，而不是能直接提升一周等权组合收益的独立 alpha。

### 68.4 反事实分析

第一反事实：如果历史大涨股概率是缺失的右尾专家，在 path Top20 池内重排应同时提升开发期和 2026。实际只微增 2026，开发期略降，说明它不是当前缺失主因。

第二反事实：如果归因模型只是噪声，2024 不应有显著改善。实际 2024 提升明显，说明它确实捕捉到某些市场阶段的大涨股状态；但这个状态不是跨年稳定收益源。

第三反事实：如果大涨股归因能解决正式账户损耗，拒单和回撤应明显下降。实际 2026 最大回撤基本不变，拒单同为 `84` 次，说明它没有改变交易可达性和脆弱段暴露。

第四反事实：如果 60 日波动/离散度/宽度是下一阶段主线，应该把它作为市场状态或候选土壤特征，而不是直接作为 path 重排分数。当前结果支持这个解释：归因稳定，但直接重排收益不厚。

### 68.5 判定

`rejected_with_signal`。

历史大涨股 ML 归因是有价值的诊断方向，但不是当前可以推进合代码的策略方向。它告诉我们：A 股短期右尾机会与 60 日波动、市场离散度、宽度和中期趋势位置高度相关；但把这个概率直接叠到 path Top20 上，不能把正式账户收益推向几十倍，也没有降低 2026 回撤。

下一步不再沿“大涨概率直接重排”调权重。更可能的方向是重新定义训练目标：从“未来 5 日终点收益”或“未来 20 日大赢家”改成“可交易、前 1-5 日不脆弱、且在高离散状态下能快速兑现的一周右尾路径”。也就是把大涨归因转成标签设计和候选土壤，而不是作为后处理分数。

## 69. tradable_big_winner_path_account_v1

### 69.1 假设

第 68 轮说明“历史大涨股概率”直接重排 path 候选池不是主收益层，但它稳定指向 60 日波动、市场离散度、宽度和趋势位置。于是本轮把大涨归因改成更贴近账户的标签：未来 15/20 日属于候选池右尾，同时前 5 日路径不能太脆弱。

核心问题：学习“可交易、前 1-5 日不脆弱、且后续有右尾弹性”的路径标签，能否同时保留 path 的开发期收益，并修复 2026 的正式账户回撤？

### 69.2 产物和口径

本轮复用 `/tmp/quantx-research/tradable-big-winner-path-v1/analyze_tradable_big_winner_path.py` 的标签诊断，再新增 prediction store 写出脚本：

```text
/tmp/quantx-research/tradable-big-winner-path-v1/write_tradable_big_winner_path_predictions.py
sha256:522ee78b5c09464e4b17ca8e00c261f03973010c8a6df4b10eff3f86d91b5d85

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_tw20_pool200_dev_2021_2025_top20_predictions.json
sha256:439b7bc056a8e65daa2094e98593b3024acd291701c88daaee4037066535c88d

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_tw15_edge_pool500_dev_2021_2025_top20_predictions.json
sha256:6ee364de19597b34f5f5eb375040beeda3672727715207458499722c31c7d4c5

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_tw20_pool200_val63_2026_top20_predictions.json
sha256:b0a9e7ee76fc1eb4b594933b7597646191859dc99cb52364ef594367ac09fbd6

/tmp/quantx-research/tradable-big-winner-path-v1/tradable_big_winner_path_tw15_edge_pool500_val63_2026_top20_predictions.json
sha256:8548d53d012811a4f75b429811c41cff2b9d1128046de91c8c04686c67f3ab5a
```

两个代表变体：

1. `tw20_pool200`：基础 ML Top200 内直接用未来 20 日“可交易右尾 winner”概率排序。
2. `path_tw15_edge_pool500`：基础 ML Top500 内，用 `75% path_topq + 25% (可交易 15 日 winner - early failure)`，测试 2026 防守信号是否能账户化。

正式账户回放继续使用 Exp67 的同一工具，Top20 池补位买5、日频重叠、持有 5 个交易日、`gap25+hist120`、正式成本/交易规则。

账户产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_tradable_tw20_pool200_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:34d3fa448d371055b0f286d8627dadc76bdac31032de2ebb7653e03d9988e5f6

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_tradable_path_tw15_edge_pool500_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:bca61c2ef71119b8cb5b0998ca490d26639b561430e91e5d4b4ebfa01f1c7006

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_tradable_tw20_pool200_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:3fed797a3f6457325fbd09ddc6a9331d7f34fa1f82d6e884d4cdba299b63323e

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_tradable_path_tw15_edge_pool500_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:fab7f97b91f85feae15416072ae6a2e80d30d4f30f695fd0868bb14a5196c43b
```

### 69.3 结果

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 平均持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tw20_pool200` | 2022-2025 | 5.41x | `+441.03%` | `-36.35%` | 14.80 | 7.67 | `+51.26%、+31.24%、+43.25%、+86.74%` |
| `tw20_pool200` | 2026 val63 | 0.95x | `-4.62%` | `-19.23%` | 11.26 | 8.75 | `-4.62%` |
| `path_tw15_edge_pool500` | 2022-2025 | 5.05x | `+404.61%` | `-33.59%` | 16.83 | 7.67 | `+55.65%、+35.82%、+57.91%、+48.71%` |
| `path_tw15_edge_pool500` | 2026 val63 | 1.10x | `+9.78%` | `-10.57%` | 13.08 | 8.76 | `+9.78%` |

对照 Exp67/68：path Top20 补位买5 在 2022-2025 为 5.48x、2026 为 `+19.47%`；big-winner path 重排 2026 为 `+19.99%`。因此本轮没有提升收益厚度。

`path_tw15_edge_pool500` 的确降低了 2026 回撤，从 path/big-winner 的约 `-17.4%` 降到 `-10.6%`，开发期回撤也略低于 `tw20_pool200`。但收益被明显削弱：2026 只有 `+9.78%`，2022-2025 也只有 5.05x，远低于几十倍目标。

### 69.4 反事实分析

第一反事实：如果“可交易右尾路径”是缺失主 alpha，纯 `tw20_pool200` 应在 2026 更强。实际它正式账户为 `-4.62%`，说明右尾概率一旦脱离 path 锚，容易买到账户上不可兑现的弹性。

第二反事实：如果 early failure 惩罚解决了正式账户损耗，`path_tw15_edge_pool500` 应至少不输 path baseline。实际它回撤降低但收益显著降低，说明它更像风险过滤器，而不是收益生成器。

第三反事实：如果 2026 风险只是偶然噪声，防守型标签不应稳定降低回撤。实际 2026 最大回撤从约 `-17%` 降到 `-10.6%`，说明它确实捕捉到一部分 2026 脆弱状态。

第四反事实：如果继续调 `tw15/tw20/edge` 权重能接近目标，开发期应该已经出现收益厚度扩张。实际四年最终倍数仍约 5x，说明这条线没有足够弹性，不值得继续围绕权重做搜索。

### 69.5 判定

`rejected_with_signal`。

可交易右尾路径标签证明了一个有用风险事实：市场宽度、20 日离散度、长期 RPS、近期波动和 early-failure 概率可以解释 2026 的账户回撤，并能做防守。但它不能成为用户要求的主策略，因为它牺牲收益厚度，正式账户仍只有个位数倍数。

下一步应放弃“在基础 ML/path 候选池里继续做防守型重排”。更可能的金子不在过滤器，而在新的候选生成土壤：例如更贴近 A 股短线资金行为的“板块主升期内的二阶跟随/补涨扩散”，但要用 ML 学习可兑现的候选，而不是手写涨停扩散公式。

## 70. sector_follower_ml_candidate_v1

### 70.1 假设

第 69 轮说明防守型右尾路径标签能解释风险，但不是主收益源。本轮按“行业/概念主升期内的二阶跟随/补涨扩散”重新做候选生成，而不是在 path Top20 里调尾部权重。

核心问题：在基础 5d 全市场 ML Top500 这个宽候选池内，用行业/概念热度、宽度、放量、加速、组内 lag/catch-up、个股短中期 RPS 等特征训练股票级一周收益，能否生成比 path baseline 更厚、且能通过 2026 forward 的正式账户组合？

### 70.2 产物和口径

临时脚本和产物均在 `/tmp`：

```text
/tmp/quantx-research/sector-follower-ml-v1/write_sector_follower_ml_predictions.py
sha256:5b38af899d144ddb5990dd4534519b4fd1b55c72c06a90be73764e05dca0f35d

/tmp/quantx-research/sector-follower-ml-v1/sector_follower_ml_pool500_blend40_dev_2021_2025_top20_predictions.json
sha256:68c7727018b752a34166a94f45625dc075ad9d6d1afbc0889109fee8e97d5ba1

/tmp/quantx-research/sector-follower-ml-v1/sector_follower_ml_pool500_blend40_val63_2026_top20_predictions.json
sha256:d392f54e3005d3e01bc5204ec54c64122b9fa2a99979cd09d16eac0f720d1fe2
```

训练/验证方式：

1. dev 使用 2021-2025 walk-forward，实际账户从 2022 起回放。
2. 2026 使用 2021-2025 训练后 forward apply 到 `2026-01-05` 至 `2026-07-10`。
3. 每日从基础 5d 全市场 ML store 中取 Top500，输出 Top20 PredictionStore。
4. 正式账户仍使用 Exp67 工具：Top20 池补位买5、日频重叠、持有 5 个交易日、`gap25+hist120`、正式成本/交易规则。

账户产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_ml_pool500_blend40_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:9c0ea622b04b1bcccbbe054418d018ba75a2c6c29c175d86596e4b7847763c3a

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_ml_pool500_blend40_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:426a4dffce38acf6ba483f002e4829d8aa36e3db3ca4b7e036403e17d5cbf17c
```

### 70.3 结果

标签层 `blend40` 相比 base 有轻微改善：dev Top20 平均日 label5 从 `+0.012564` 到 `+0.012863`；2026 从 `+0.012018` 到 `+0.012920`。但账户层没有放大成收益厚度。

正式账户结果：

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 平均持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| sector follower ML `blend40` | 2022-2025 | 4.44x | `+343.73%` | `-40.20%` | 16.81 | 7.69 | `+43.77%、+37.55%、+7.82%、+105.50%` |
| sector follower ML `blend40` | 2026 val63 | 1.09x | `+8.89%` | `-15.12%` | 15.35 | 8.24 | `+8.89%` |

对照 Exp67：path Top20 补位买5 为 2022-2025 5.48x、2026 `+19.47%`。因此本轮既没有提升 2026，也没有提升开发期收益厚度。

### 70.4 反事实分析

第一反事实：如果“行业/概念主线扩散 + 组内补涨”是缺失主 alpha，宽 Top500 内训练应明显超过 path baseline。实际 dev 4.44x 低于 5.48x，2026 `+8.89%` 低于 `+19.47%`，主 alpha 假设不成立。

第二反事实：如果问题只是原 path 池太窄，Top500 宽候选应在账户层释放更多可替代股票。实际平均持仓和拒单都正常，但收益变薄，说明宽候选引入了更多低质量可交易样本，而不是新增厚 alpha。

第三反事实：如果组内 lag/catch-up 特征能自然避开追高不可买，正式账户拒单应明显下降并提升收益。实际 2026 拒单 `45` 次，收益仍弱，说明它最多缓和部分执行问题，不能重构收益来源。

第四反事实：如果只需要调 blend 权重，标签层最好的 blend25/blend55 应在 formal 上修复。补测 `blend25` dev 4.61x、2026 `+2.65%`，反而更弱。因此继续围绕 blend 权重搜索不是好方向。

### 70.5 判定

`rejected_with_signal`。

行业/概念二阶跟随特征在标签层有轻微信号，但正式账户不够厚，且 2026 不如 path baseline。这个结果支持一个更高层判断：行业/概念目前更像解释层和风险层，而不是用简单组热度/组内 lag 直接生成主收益的独立引擎。

下一步不继续做“宽候选池内轻重排”。要么重新定义账户目标，要么改成动态路径决策，例如学习买入后 1-5 日是否续持/提前退出/延迟退出，而不是固定 5 日卖出。

## 71. executable_target_sector_follower_v1

### 71.1 假设

Exp70 可能失败的一个原因是标签仍是理论 open-to-open 收益，没有把正式账户中的涨停不可买、停牌、跳空、历史不足和卖出受阻写进训练目标。本轮把目标改成“可执行净收益”：不可买样本直接大幅惩罚，卖出受阻样本惩罚，正常样本扣粗略往返成本。

核心问题：如果正式账户损耗是主要矛盾，那么把可执行性写进标签后，应改善 2026 formal，并且不能显著牺牲开发期收益。

### 71.2 产物和口径

同一临时脚本增加 `--target-mode executable`：

```text
/tmp/quantx-research/sector-follower-ml-v1/sector_follower_executable_pool500_blend40_dev_2021_2025_top20_predictions.json
sha256:0f58eccd65303a22188f2b9ccbe774634a233c803d959bb9e5ea4dada7905ab7

/tmp/quantx-research/sector-follower-ml-v1/sector_follower_executable_pool500_blend40_val63_2026_top20_predictions.json
sha256:46ea74a471be8f8b6de8c3851b396c6112eb8172103c7857015f9179a0c3ab89
```

账户产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_executable_pool500_blend40_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:e053f3be3569d306d3de90b16616c192bac3e41c921c61eb69ec7867eed62eb1

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_executable_pool500_blend40_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:5cb214394db2109af4ad0aa2e5365d52b3fafea19d4634d0b94b69928f2b04e8
```

### 71.3 结果

标签层看起来更“账户友好”：2026 `blend40` 可执行标签均值从 base 的 `+0.011418` 提升到 `+0.014746`，正 label 日比例从 `66.10%` 到 `71.19%`。但注意 `target_raw5` 均值接近 0 或为负，说明可执行惩罚非常强，模型更多是在学习避险。

正式账户结果：

| 口径 | 区间 | 最终倍数 | 累计收益 | 最大回撤 | 平均唯一持仓 | 平均持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| executable target `blend40` | 2022-2025 | 3.29x | `+228.79%` | `-37.88%` | 17.60 | 7.61 | `+25.74%、+26.37%、+2.57%、+101.48%` |
| executable target `blend40` | 2026 val63 | 1.13x | `+12.98%` | `-13.60%` | 16.08 | 8.31 | `+12.98%` |

对照 Exp70，2026 从 `+8.89%` 改善到 `+12.98%`，回撤从 `-15.12%` 改善到 `-13.60%`；但开发期从 4.44x 降到 3.29x。对照 path baseline，仍弱于 2022-2025 5.48x、2026 `+19.47%`。

### 71.4 反事实分析

第一反事实：如果 formal 损耗是主因，可执行标签应在 2026 和 dev 同时改善。实际 2026 小幅改善，dev 大幅变薄，说明它主要学到风险规避，而不是主收益。

第二反事实：如果可执行标签可以替代 formal account replay，标签层提升应对应账户层提升。实际标签层提升很明显，但账户只改善 2026 一部分，证明最终仍必须以正式账户为裁判。

第三反事实：如果 A 股一周 alpha 主要来自“可买、可卖、低跳空”的普通样本，开发期不应下降这么多。实际 2024 只有 `+2.57%`，说明厚收益常和更高执行风险绑定，粗暴惩罚会把右尾也切掉。

第四反事实：如果继续加强惩罚能过关，开发期已经给出反证：收益厚度不足，越防守越远离几十倍目标。

### 71.5 判定

`rejected_with_signal`。

可执行标签是有用的风险诊断：能降低 2026 回撤并改善一部分 forward 收益。但它不是主策略方向，因为它明显牺牲开发期收益厚度，正式账户仍只有 3.29x/四年和 2026 `+12.98%`。下一步应转向动态持有路径或退出/续持模型，检验固定 5 日退出是否截断了可交易右尾，而不是继续在买入候选上做防守型训练。

## 72. dynamic_hold_state_counterfactual_v1

### 72.1 假设

Exp67-71 都固定持有 5 个交易日。但 A 股短线 alpha 的兑现速度可能是状态依赖的：强趋势/宽度充足时，右尾需要更久释放；弱势或拥挤状态下，收益半衰期更短，应该更快兑现。

核心问题：同一 path Top20 预测、同一正式账户引擎，只改变持有期，能否说明固定 5 日退出截断收益或暴露风险？进一步，能否用可见市场状态或影子组合近期表现做因果的持有期切换？

### 72.2 产物和口径

固定持有期扫描复用 Exp67 正式账户工具：Top20 池补位买5、日频重叠、`gap25+hist120`、正式交易成本和交易规则。代表产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold2_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:e721ae5c7354a3413dfa855899bb87675f426c46e13544de5440f54da92b077d

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold2_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:6417a8a1f71028c158a5aff12909fc4680313ba074c326ecf7db638eb236c78e

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold7_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:2d63c128d3c3546cfcc73d463fd1808a0095d2963bb36242ef08d81bb4f38e72

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold7_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:5791af18b395fcb08b445eb28860eccce8af4e6c727449c79617c1fcb8a91218

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold10_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:6aca1732916535470101383d8fdcd0d20e5304cfbfbcc04ca1515a476c66a0b6

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_pathseq_pool20_buy5_hold10_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:68e48c483491055abe6d75dc664c92bd7ed2d350b41ba178e23b949c667032fa
```

新增 `/tmp` 诊断脚本用于两类持有期控制器：

1. `state-rule`：用当日可见市场宽度、20 日中位收益、20 日离散度、行业/概念 top group heat，在 short/fixed/long hold 间切换。
2. `adaptive-performance`：读取固定持有期 shadow formal 曲线，只用过去 N 天哪个 shadow 曲线表现更好来决定下一笔 sleeve 的持有期。

```text
/tmp/quantx-research/dynamic-hold-state-v1/analyze_dynamic_hold_state.py
sha256:4fe30630bca10816b6443f813b4ec8580bf368ea7c18a4917871d9aa4e876313

/tmp/quantx-research/dynamic-hold-state-v1/dynamic_hold_state_rule_s3_f5_l10_dev_2022_2025.json
sha256:763546b3499002496a65e41ac881a2012b1626e2e3711b0d9bccf237cd03fdbd

/tmp/quantx-research/dynamic-hold-state-v1/dynamic_hold_state_rule_s3_f5_l10_val63_2026.json
sha256:78f4b23e9add1247452316063ed56b0d68fcfc9cd2734b570c59d0255edf010d

/tmp/quantx-research/dynamic-hold-state-v1/dynamic_hold_adaptive_perf_h2_5_10_w20_dev_2022_2025.json
sha256:b5ee54c691275dec479939f95fdcd654bb98435879cd507b0704cb7759f823b0

/tmp/quantx-research/dynamic-hold-state-v1/dynamic_hold_adaptive_perf_h2_5_10_w20_val63_2026.json
sha256:7519489107fecca255c0ea7d66e0a0236ced7d74020d67396728a9c7a44e44ca
```

### 72.3 固定持有期结果

| 持有交易日 | 2022-2025 最终倍数 | 2022-2025 最大回撤 | 2022-2025 平均唯一持仓 | 2026 收益 | 2026 最大回撤 | 2026 平均唯一持仓 | 2026 平均日历持有日 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.64x | `-47.95%` | 8.59 | `+47.05%` | `-17.22%` | 7.48 | 3.96 |
| 3 | 4.98x | `-38.10%` | 11.95 | `+41.09%` | `-15.54%` | 10.12 | 5.87 |
| 4 | 5.45x | `-39.71%` | 15.10 | `+23.37%` | `-17.63%` | 12.00 | 7.32 |
| 5 | 5.48x | `-37.35%` | 18.00 | `+19.47%` | `-17.42%` | 14.52 | 8.79 |
| 6 | 5.09x | `-37.48%` | 20.98 | `+17.73%` | `-16.83%` | 15.96 | 10.88 |
| 7 | 6.07x | `-34.93%` | 23.86 | `+12.50%` | `-16.79%` | 18.21 | 12.77 |
| 10 | 6.51x | `-32.25%` | 31.86 | `+11.62%` | `-17.45%` | 23.71 | 17.33 |

固定持有期呈现强烈状态差异：2022-2025 越长越强，hold10 为 6.51x；2026 越短越强，hold2 为 `+47.05%`，hold3 为 `+41.09%`。这说明 2026 的 path 信号半衰期显著缩短，固定 5 日不是 2026 最优。

### 72.4 动态控制器结果

手写状态规则第一版 `short=3/fixed=5/long=10`：

| 控制器 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | hold 分布 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| state-rule s3/f5/l10 | 2022-2025 | 4.62x | `-34.97%` | 17.19 | 7.28 | `3:604, 5:151, 10:214` |
| state-rule s3/f5/l10 | 2026 | `+35.88%` | `-17.20%` | 11.86 | 6.73 | `3:105, 5:12, 10:7` |

保守/激进阈值扫描：

| 控制器 | 2022-2025 | 2026 | 结论 |
| --- | ---: | ---: | --- |
| conservative s3/f5/l10 | 4.51x | `+34.36%` | dev 更弱，没有保住长持收益 |
| aggressive s2/f5/l10 | 4.29x | `+47.32%` | 基本退化成 2026 全程 hold2，dev 明显变薄 |

影子组合表现自适应：

| 控制器 | 2022-2025 | 2026 | 结论 |
| --- | ---: | ---: | --- |
| adaptive h2/5/10 w20 | 5.80x | `+23.73%` | 折中但不够好，2026 没抓住短持优势 |
| adaptive h2/5/10 w10 | 5.23x | `+10.31%` | 短窗口被噪声反噬 |
| adaptive h2/5/10 w5 | 3.78x | `+6.24%` | 切换过度且失效 |

### 72.5 反事实分析

第一反事实：如果固定 5 日已经接近最优，hold2/3/7/10 不应出现系统性差异。实际 dev 喜长持、2026 喜短持，说明持有期是一个真实的市场状态变量。

第二反事实：如果 2026 的低收益只是候选选择问题，缩短持有期不应显著改善同一候选池。实际同一 path Top20 下 2026 从 hold5 `+19.47%` 提升到 hold2 `+47.05%`，说明 2026 的关键矛盾之一是兑现速度和持仓半衰期，而非只在买入端。

第三反事实：如果手写市场状态规则足够，它应同时保住 dev 的长持收益和 2026 的短持收益。实际第一版 state-rule dev 只有 4.62x，保守/激进版本也没有同时成立，说明几个阈值不足以表达路径半衰期。

第四反事实：如果影子组合近期动量能代表状态，自适应 w5/w10/w20 应逼近两边最优。实际 w20 只是折中，w5/w10 失效，说明最近 shadow 曲线的噪声过大，不能直接作为控制器。

第五反事实：如果继续调持有期控制器就能达到几十倍，固定扫描里应该已有接近目标的 dev/forward 组合。实际固定最好 dev 6.51x、2026 最好 `+47.05%`，仍没有达到五年几十倍目标，说明持有期是重要改良层，但不是单独主 alpha。

### 72.6 判定

`rejected_with_signal`。

本轮给出一个强线索：path 信号的收益半衰期随市场状态变化，2026 明显需要更快兑现，而 2022-2025 的右尾更适合长持。但当前手写状态规则和 shadow 表现自适应都不能同时保住开发期收益和 2026 前向收益，因此不能合代码。

下一步不继续围绕固定持有期或简单控制器调参。更可能的方向是直接学习“路径半衰期/兑现速度”作为标签：用入选后第 1-10 日的路径形状、市场宽度、行业/概念热度、波动离散度，预测该笔 sleeve 应短持还是长持。并且这个模型必须 walk-forward 输出真实 hold map，再用 formal account 验证；否则容易退化成事后挑持有期。

## 73. ml_half_life_hold_map_v1

### 73.1 假设

Exp72 说明同一 path Top20 候选池的最优持有期随市场状态变化：开发期偏长持，2026 偏短持。于是本轮直接学习“短持还是长持”：用信号日前可见的市场状态、候选池短中期路径、行业/概念热度等特征，预测 hold2 相对 hold10 的未来收益差，再输出逐日 hold map 交给正式账户回放。

核心问题：如果半衰期可由这些可见状态识别，那么 ML hold map 应同时接近开发期长持优势和 2026 短持优势，而不是只做折中。

### 73.2 产物和口径

本轮仍只在 `/tmp` 下实现，不合代码。标签是 path Top20 当日入选股票从 T+1 开盘到 hold2/hold10 退出的平均收益差，特征复用市场宽度、20/60 日中位收益、20 日离散度、top group heat、候选池 1/5/20 日收益和涨停/弱势比例。

```text
/tmp/quantx-research/dynamic-hold-state-v1/write_ml_half_life_hold_map.py
sha256:6cae55801bd85a46328a39a377126ece684bcb65d73f497ae8d923a2c0e59938

/tmp/quantx-research/dynamic-hold-state-v1/ml_half_life_hold_map_h2_h10_dev_2022_2025.json
sha256:d3713d154b04278532d1608b6884f8b962b5394c84b9f16582f221918e72a7fb

/tmp/quantx-research/dynamic-hold-state-v1/ml_half_life_hold_map_h2_h10_val63_2026.json
sha256:c50edb3e820687e8c2168ec246c0c7a35f63e975c76003cbda08df5723beed6a

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_ml_half_life_h2_h10_dev_2022_2025.json
sha256:3c8920434369740b94cda94964aae57e18daa2fdbcf48232ebd890746116bb9e

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_ml_half_life_h2_h10_val63_2026.json
sha256:c30ebc63aad0f695e32f24b82b5c36b31926da8447ac88375ae69bd855f89a01
```

dev 使用 2022-2025 walk-forward，但 2022 因训练行数不足只作为训练期，实际 scored rows 为 2023-2025。2026 使用 2022-2025 训练后 forward apply。

### 73.3 结果

标签层模型显著偏向长持：

| 区间 | scored rows | 实际短持更优比例 | 模型选择 hold2 比例 | hold 分布 | mean short | mean long | mean chosen |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2023-2025 | 716 | 42.60% | 20.67% | `2:148, 10:568` | `+0.01293` | `+0.04476` | `+0.04245` |
| 2026 | 113 | 36.28% | 10.62% | `2:12, 10:101` | `+0.01571` | `+0.04125` | `+0.03953` |

正式账户结果：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | hold 分布 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ML half-life h2/h10 | 2022-2025 | 6.24x | `-31.89%` | 25.11 | 11.56 | `2:148, 10:568, fallback 5:253` |
| ML half-life h2/h10 | 2026 | `+18.52%` | `-16.93%` | 21.71 | 15.85 | `2:12, 10:101, fallback 5:11` |

开发期接近 hold10，但仍低于固定 hold10 的 6.51x；2026 基本退回 hold5 baseline，明显低于固定 hold2 的 `+47.05%`。

### 73.4 反事实分析

第一反事实：如果单笔 open-to-open 的 hold2-vs-hold10 差值就是账户半衰期，模型在 2026 应大量选择 hold2。实际 2026 只选 12 天 hold2，说明单笔收益标签和账户资本路径不一致。

第二反事实：如果长持在 2026 单笔收益上确实更优，正式账户 long-heavy map 应超过 fixed hold5。实际只有 `+18.52%`，低于 fixed hold5 `+19.47%`，远低于 fixed hold2 `+47.05%`。这说明 2026 的优势来自资本周转、重叠 sleeve 和减少 stale exposure，而不是单笔均值收益。

第三反事实：如果当前特征足以描述半衰期，dev 和 2026 都应接近各自 fixed oracle。实际两边都没接近，说明状态特征对持有期选择的可识别度不够。

### 73.5 判定

`rejected_with_signal`。

本轮证明：裸 open-to-open 收益不能作为持仓半衰期训练目标。正式账户里的资金占用、重叠开仓、卖出受阻、现金闲置和组合级 stale exposure 是一等变量。下一步如果继续做持有期，必须直接用账户曲线或账户级增量作为标签；否则会把 2026 学反。

## 74. account_aware_hold_map_v1

### 74.1 假设

Exp73 的失败可能来自标签不贴近账户。本轮把目标改成账户感知：读取固定 hold2/3/5/7/10 的正式账户 shadow 曲线，以信号日后 10 个交易日哪条账户曲线表现最好作为标签，训练多个 hold 的未来账户分数，再选择预测分数最高的持有期。

核心问题：如果持有期半衰期确实可以由信号日前市场/候选池状态识别，那么账户感知标签应比 Exp73 更能捕捉 2026 短持，同时不牺牲开发期收益厚度。

### 74.2 产物和口径

实现仍在 `/tmp`，不合代码。脚本复用 Exp73 的 feature panel，只替换标签生成和多目标回归逻辑。

```text
/tmp/quantx-research/dynamic-hold-state-v1/write_account_aware_hold_map.py
sha256:4dc8fa77253c4764900a2c1926654c5d2b4fcc5ce710a4c0a7bb2b6a899f1105

/tmp/quantx-research/dynamic-hold-state-v1/account_aware_hold_map_w10_r0_dev_2022_2025.json
sha256:b611c2ac06455b6751664cadcab03525e0e783753c5c0d710c64bfa8e636d96d

/tmp/quantx-research/dynamic-hold-state-v1/account_aware_hold_map_w10_r0_val63_2026.json
sha256:43a0cce5355b4916191d8253949dfdd1a0706ee105bd7fdc96628e68b304b192

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_account_aware_w10_r0_dev_2022_2025.json
sha256:b5e1743062c2d9ee62099642bf1bfb194c80ea04d824d2145a98b428b602e0ba

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_account_aware_w10_r0_val63_2026.json
sha256:16375397afbc0fe91eb1e01c9ecd352b5aaef9ce77eba43254b392234a3ffba4
```

标签窗口为 10 个交易日，风险惩罚为 0。本轮有一个重要边界：训练标签来自固定 hold shadow 曲线的未来 10 日表现，虽然预测是 walk-forward/forward，但标签本身是组合级曲线近似，不是精确的“新开 sleeve 边际贡献”。

### 74.3 结果

标签层选择能力较弱：

| 区间 | scored rows | 实际最佳 hold 分布 | 模型选择 hold 分布 | 选择准确率 | mean best score | mean chosen score |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 2023-2025 | 716 | `2:160, 3:106, 5:83, 7:91, 10:276` | `2:92, 3:148, 5:216, 7:117, 10:143` | 19.13% | `+0.03626` | `+0.01718` |
| 2026 | 113 | `2:47, 3:26, 5:7, 7:9, 10:24` | `2:42, 3:6, 5:19, 7:21, 10:25` | 23.01% | `+0.04202` | `+0.01597` |

正式账户结果：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | hold 分布 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| account-aware hold map | 2022-2025 | 4.30x | `-39.64%` | 18.85 | 8.24 | `2:92, 3:148, 5:469, 7:117, 10:143` |
| account-aware hold map | 2026 | `+28.40%` | `-14.02%` | 15.17 | 8.57 | `2:42, 3:6, 5:30, 7:21, 10:25` |

相比 Exp73，2026 确实更像短持：hold2 从 12 天增加到 42 天，正式账户从 `+18.52%` 提升到 `+28.40%`，回撤也降到 `-14.02%`。但开发期从 path baseline hold5 的 5.48x 和 hold10 的 6.51x 降到 4.30x，收益厚度明显不够。

### 74.4 反事实分析

第一反事实：如果账户感知标签解决了 Exp73 的核心错配，2026 应明显改善。实际确实从 `+18.52%` 到 `+28.40%`，说明账户标签方向比单笔收益标签更正确。

第二反事实：如果状态特征足以学到半衰期，模型应接近 fixed oracle：开发期接近 hold10、2026 接近 hold2。实际开发期只有 4.30x，2026 也低于 hold2 的 `+47.05%`，说明可见状态对“未来哪条账户曲线赢”的识别仍然很弱。

第三反事实：如果继续调 label window、风险惩罚或 hold 集合就能达到几十倍，当前账户层至少应超过固定 hold5/hold10 的一端。实际只是 2026 中间改善、开发期明显损失，表现为折中器，不是主收益生成器。

第四反事实：如果用户目标可以靠持仓控制完成，固定持有期扫描中应已有接近目标的收益量级。实际固定最好仍只有 6.51x/四年，说明持仓期不是缺失的核心 alpha。

### 74.5 判定

`rejected_with_signal`。

账户感知标签比裸收益标签方向更对，能把 2026 从 hold5 附近提升到 `+28.40%`，但仍不能同时保住开发期和 2026，更远低于五年几十倍目标。持仓半衰期是重要风险/效率层，不是当前主收益层。

下一轮不继续围绕 hold map 调窗口、惩罚或阈值。研究方向切回候选生成：寻找新的低相关右尾专家，尤其是“可交易、不过热、仍处于行业/概念二阶扩散早段”的候选土壤；正式验证仍必须直接跑账户层，而不是只看标签层。

## 75. four_family_consensus_formal_replay_v1

### 75.1 假设

此前四族共识把第 40 轮 path、head-union、weekly persistence、honest soil 四类候选做股票级 consensus，标签层和部分账户口径显示 2026 有改善。但这些结果并未完全复用 Exp67 的正式 overlap account 裁判。本轮补正式账户复核。

核心问题：如果多低相关候选族共识是真正更鲁棒的候选生成器，那么在正式成交、最小佣金、滑点、停牌、涨跌停、`gap25+hist120` 和日频重叠 sleeve 下，应至少保住 2026 相对 path baseline 的优势。

### 75.2 产物和口径

输入 prediction store：

```text
/tmp/quantx-research/store-consensus-v1/store_consensus_4fam_vote_sum_top20_dev_2022_2025_predictions.json
sha256:65dba72a2a71b58a224f989d432810e7698c67255bcdd48cd6057f949032d213

/tmp/quantx-research/store-consensus-v1/store_consensus_4fam_vote_sum_top20_val63_2026_predictions.json
sha256:016221da40493d9856eeb2bef91f3de341abef9173ea380a9bd02924ecc078f6
```

正式账户回放：Top20 池补位买5、持有 5 个交易日、日频重叠 sleeve、`max_open_gap=0.25`、`min_history_days=120`。

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_4fam_vote_sum_top20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:4ef5724b5c23b6343f1e940c086c77779c41aa8ec6a299c67cbdfbd46168d1d1

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_4fam_vote_sum_top20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:4d525169b301412ee3e19c5a1f02592c04f5e7bb3353fdbe719c0076a538bcac
```

重要边界：四族共识 dev store 只有 `705` 个 session，不是 path store 的完整 `963` 个 session；本轮作为正式成交约束反事实，不作为完整五年最终候选。

### 75.3 结果

正式账户结果：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4-family vote_sum formal | 2022-2025 | 6.50x | `-21.32%` | 11.89 | 7.72 | `+33.89%、+51.25%、+57.64%、+104.83%` |
| 4-family vote_sum formal | 2026 val63 | `+8.66%` | `-16.44%` | 13.19 | 7.93 | `+8.66%` |

开发期的风险形态比 path baseline 更稳：path baseline 为 5.48x、回撤 `-37.35%`，四族共识为 6.50x、回撤 `-21.32%`。但 2026 被正式账户打穿：path baseline `+19.47%`，四族共识只有 `+8.66%`。

### 75.4 反事实分析

第一反事实：如果共识捕捉的是低相关主 alpha，2026 应强于 path baseline。实际正式账户下只有 `+8.66%`，说明之前 2026 的共识优势无法穿透正式成交约束或不覆盖同一交易机会。

第二反事实：如果共识的价值主要是风险分散，开发期应体现更浅回撤。实际成立，回撤从 path baseline `-37.35%` 改善到 `-21.32%`。它是风险结构改善，不是收益量级突破。

第三反事实：如果继续堆候选族可以接近几十倍，四族中应已经能在开发期收益厚度上明显超越 path。实际只有 6.50x，且覆盖不完整，离目标仍远。

第四反事实：如果 2026 主要问题是 path 单模型过拟合，多模型共识应自然修复。实际没有修复，说明 2026 的主矛盾更像可交易收益土壤变化，而不是单模型噪声。

### 75.5 判定

`rejected_with_signal`。

四族共识在开发期降低回撤、改善风险形态，但正式 2026 不通过，且收益量级仍不到目标。它可以作为未来组合风险结构的参考，但不是当前主候选生成方向。下一步不继续盲目扩展候选族数量，而要找新的可交易右尾土壤。

## 76. early_group_diffusion_soil_v1

### 76.1 假设

前几轮说明行业/概念热度、组扩散和组内残差在标签层有弱信号，但正式账户不够厚。一个可能原因是旧方法买得太热或太晚。本轮直接构造“早段扩散但不过热”的组土壤：组 20 日趋势不弱、5 日加速、组内宽度和温和放量启动，但个股不追一日尖峰和 5 日极端涨幅。

核心问题：如果 A 股概念/行业轮动的金子在扩散早段，那么这个可解释分数应至少在 base5d Top500 宽池内提升 Top20 未来 5 日标签，且 2026 不能反向。

### 76.2 产物和口径

本轮只做标签层和 PredictionStore 生成；由于标签层已经失败，不进入正式账户回放。实现仍在 `/tmp`，复用 Exp70 的 base Top500 管线和行业/概念特征矩阵。

```text
/tmp/quantx-research/early-group-diffusion-v1/write_early_group_diffusion_predictions.py
sha256:392e46aef92fc7234246b145a5ea53a6e3c5c74a512d2fcfd4db91b869c18598

/tmp/quantx-research/early-group-diffusion-v1/early_group_diffusion_blend40_pool500_top20_dev_2021_2025_predictions.json
sha256:427e9edb8d81df8316050c080cec5abdea6f959328120a18c359488e6466ccaa

/tmp/quantx-research/early-group-diffusion-v1/early_group_diffusion_blend40_pool500_top20_dev_2021_2025_predictions.summary.json
sha256:e50d31bad688e78199ace9fa51db9cb18252cde675da5b02d260d0f5a6d72726

/tmp/quantx-research/early-group-diffusion-v1/early_group_diffusion_blend40_pool500_top20_val63_2026_predictions.json
sha256:fd94a541ccc7bc7f9024fcd9c4f1ce333b6f70a2798d49c5973879f76cd497fd

/tmp/quantx-research/early-group-diffusion-v1/early_group_diffusion_blend40_pool500_top20_val63_2026_predictions.summary.json
sha256:aa465f65961198466f22f48274ee16929786f294583b8a073445507457f05e67
```

基础输入：

```text
dev base prediction:
/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json

2026 base prediction:
/tmp/quantx-research/adaptive-training-protocol-v1/forward-2026-5d-val63/mainboard_lgbm_base_5d_2010_2026_forward_val63_seed7-c52c47f834608a3c/oos_predictions.json
```

### 76.3 结果

标签层结果：

| 变体 | 2021-2025 日均 label5 | 2021-2025 raw5 | 2026 日均 label5 | 2026 raw5 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| base Top500 内 Top20 | `+0.01566` | `+0.01887` | `+0.01202` | `+0.00895` | 基线 |
| pure early_group | `+0.00508` | `+0.00829` | `+0.00193` | `-0.00114` | 明显变薄 |
| early_group_antichase | `+0.00638` | `+0.00959` | `+0.00389` | `+0.00082` | 仍弱 |
| early_group_blend25 | `+0.00911` | `+0.01232` | `+0.00606` | `+0.00299` | 弱于 base |
| early_group_blend40 | `+0.00866` | `+0.01188` | `+0.00464` | `+0.00158` | 弱于 base |
| early_group_blend55 | `+0.00821` | `+0.01142` | `+0.00459` | `+0.00152` | 弱于 base |

所有早段组扩散变体在 dev 和 2026 都输给 base。特别是 2026，pure early group 的 raw5 已转负，说明这个可解释分数选择到的是更弱的可交易土壤。

### 76.4 反事实分析

第一反事实：如果旧组扩散失败只是因为买太热，那么加入不过热/反追高约束后应改善。实际 `early_group_antichase` 仍显著弱于 base，说明问题不只是追高。

第二反事实：如果组扩散早段是独立右尾土壤，pure early_group 至少应在某一端明显强。实际 dev/2026 都弱，说明静态行业/概念快照下的早段扩散特征无法直接生成 Top20 alpha。

第三反事实：如果只需要把 early group 作为弱增量，blend25/40 应保住 base 并小幅改善。实际 blend25/40 均降标签，说明这个增量方向和 base5d 的有效排序冲突。

第四反事实：如果行业/概念是主 alpha 层，提升组热度纯度应带来收益厚度。实际 avg group heat 明显升高，但收益下降，说明更强的组热度在当前数据中更像拥挤/后验解释，而不是可交易收益来源。

### 76.5 判定

`rejected`。

早段行业/概念扩散这个手工土壤不成立，且没有必要进入正式账户回放。当前证据进一步支持：行业/概念更适合做状态解释、风险归因或分层诊断；若要变成收益层，不能只靠组热度/扩散公式，而要找到更直接的可交易微观结构或新的标签目标。

下一步切换方向：不再围绕组热度公式找增量，转向“正式账户失败样本归因”。具体做法是从 path/base 正式账户的成交和拒单样本出发，学习哪些 T 日可见特征能区分真正成交后贡献收益的股票和买不到/卖不出/兑现差的股票，把标签定义在正式账户里的已成交持仓贡献上，而不是理论 future return。

## 77. formal_trade_contribution_attribution_v1

### 77.1 假设

Exp66 以后已经确认，简化 sleeve 的高收益会被正式交易约束大幅压缩；Exp70-76 又说明行业/概念、可执行标签和持仓控制更多是解释/风险层，不是自然主 alpha。本轮把目标进一步贴近正式账户：从 path Top20 正式补位买5的候选和实际买入样本出发，为每个可入场候选计算实际 open-to-open 退出路径、粗略成本后的 `net_return_proxy`，再学习哪些 T 日可见特征能提升正式账户里的成交贡献。

核心问题：如果“历史大幅上涨股票 ML 归因”有价值，它不应只解释理论大赢家，而应能在正式账户可交易 Top20 内，把前 5 个买入候选排得更好，并同时改善开发期和 2026 forward。

### 77.2 产物和口径

实现仍在 `/tmp`，不合代码。正式账户基线为 Exp67/72 使用的 path Top20 pool、补位买5、固定 hold5、`max_open_gap=0.25`、`min_history_days=120`。

归因脚本先生成两类样本：实际被正式账户选中的 `accepted`，以及同一天 Top20 内可交易但排在买入名额之后的 `rank_after_buy_count`。v2 版本为所有可交易候选补充 `exit_path` 和 `net_return_proxy`，避免只在已买入样本上训练导致标签太窄。

```text
/tmp/quantx-research/formal-trade-attribution-v1/analyze_formal_trade_attribution.py
sha256:30d0a97358059f0d39cfdf2dffe8cc07e84d9423fdde94fdbb40622a4cf095b1

/tmp/quantx-research/formal-trade-attribution-v1/pathseq_pool20_buy5_dev_2022_2025_trade_attribution_v2.json
sha256:063265d0e927733bcb00542af437f1581ee6886bddc94ace78c4bf7c3ad1e637

/tmp/quantx-research/formal-trade-attribution-v1/pathseq_pool20_buy5_val63_2026_trade_attribution_v2.json
sha256:2146a72e23e377c91705391d403f1e691ad17e9124988a1f0f7589e3c74ef1da
```

本轮测试了两类 reranker：

1. 绝对贡献回归：用 `net_return_proxy` 直接训练 LightGBM 回归器，再与 path rank 弱融合。
2. 同日相对贡献排序：把目标改成同一天可交易候选内的收益分位 `daily_rank`，并加入市场状态和路径强弱交互项，再与 path rank 弱融合。

```text
/tmp/quantx-research/formal-trade-attribution-v1/write_trade_contribution_predictions.py
sha256:32ecb1ef376ff531f184490b5000a13a71566a6abd84211c15410547c98bc11b

/tmp/quantx-research/formal-trade-attribution-v1/analyze_state_conditional_attribution.py
sha256:3768f8afe7b0889a20b2c79f4c8a2a5ae31134f9552f1f30f1fc7d23141037da

/tmp/quantx-research/formal-trade-attribution-v1/write_session_relative_contribution_predictions.py
sha256:cd74808f241cc848d0a15dfc2527e3fc23da105d0eff242cf1a34fb22b40f880
```

### 77.3 归因发现

v2 全可交易候选样本显示，开发期和 2026 的赢家结构明显不一致。

| 样本 | 可交易候选数 | mean net proxy | median | winner >3% | loser < -3% | 主要 winner-loser spread |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | 17,877 | `+0.574%` | `-0.235%` | 30.02% | 30.43% | `rps60 -0.0464`、`rps20 -0.0462`、`ret60 -0.0420`、`rps1 +0.0288` |
| 2026 | 2,078 | `+0.286%` | `-0.562%` | 31.76% | 36.62% | `rps60 +0.0491`、`ret60 +0.0357`、`rps5 -0.0269`、`ind_ret5_rank +0.0224` |

开发期更像短爆发/低中期 rank 的胜出；2026 则更偏中期趋势质量，同时低短期过热和低量能拥挤更重要。状态分桶进一步显示，`market_ret20_median` 和 `market_breadth20` 不同分位下，`rps1/rps5/rps20/rps60`、`ma20_dist_rank`、`vol_ratio20_rank` 的方向频繁翻转。因此，全局线性地奖励“趋势强”或“趋势弱”都不稳。

### 77.4 正式账户结果

正式账户基线：path Top20 pool、补位买5、hold5、`gap25+hist120`。

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 约 7.7 | 基线 |
| path baseline | 2026 | `+19.47%` | `-17.42%` | 14.52 | 约 8.9 | 基线 |
| absolute contrib `path_contrib_20` | 2022-2025 | 5.27x | `-40.19%` | 17.94 | 7.69 | 低于基线 |
| absolute contrib `path_contrib_20` | 2026 | `+17.63%` | `-17.01%` | 14.73 | 8.98 | 低于基线 |
| relative contrib `path_rel_10` | 2022-2025 | 5.12x | `-38.16%` | 17.96 | 7.68 | 低于基线 |
| relative contrib `path_rel_10` | 2026 | `+21.06%` | `-17.81%` | 14.34 | 8.82 | 小幅高于基线 |
| relative contrib `path_rel_20` | 2022-2025 | 4.89x | `-40.08%` | 18.04 | 7.70 | 明显低于基线 |
| relative contrib `path_rel_20` | 2026 | `+20.04%` | `-17.40%` | 14.61 | 8.89 | 小幅高于基线 |
| relative contrib `path_rel_35` | 2022-2025 | 5.60x | `-40.92%` | 18.40 | 7.68 | 倍数略高但回撤更深 |
| relative contrib `path_rel_35` | 2026 | `+18.12%` | `-18.34%` | 15.47 | 8.97 | 低于基线 |

关键正式回放产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_trade_contrib_path20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:165f572cafcd68ff25fe87da9569b713e27f53dc73fa558633fd433eb9a69855

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_trade_contrib_path20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:b558cf5f750a7d35fda8f46fcb5effec82f22d5d556059e5a3b5ee0cf36e269b

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path10_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:25b4c6b1738b1303b416cacc3be6669c42ca5656eebf357fb54c8333758ccab9

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path10_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:fca4db35a04103da49f3363e844de5990525ee1ef6acedfd90481f3911ae1188

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:d57c1a69377b285a2f175f3d4bf4f415d4515ab9955c7965e4887d43809c32aa

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:69125654fda6cd86ddb0f72597082fe5373bdc686fecf32152fc4916fa1e2766

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path35_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:8395d8e3e9fbfd3256d80881a88686282c2966d47bbc761acf43eb50f2b1522b

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_session_relative_rank_path35_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:2b18b0b9a04b27f744cdc8f7dc9af088bc9dc502eaab3c5b266b288eae2056f2
```

### 77.5 反事实分析

第一反事实：如果正式账户贡献标签比理论 future return 更贴近目标，那么绝对贡献回归应同时改善 dev 和 2026。实际 `path_contrib_20` 两端都低于基线，说明绝对收益回归仍被市场 beta、重叠 sleeve 和状态翻转噪声污染。

第二反事实：如果同日相对排序能解决市场 beta 噪声，`daily_rank` 目标应显著提高买入前 5 的账户表现。实际 2026 只小幅提升到 `+21.06%`，开发期则从 5.48x 降到 5.12x，说明这个标签有局部解释力但不是稳定主 alpha。

第三反事实：如果只是混合权重问题，`path_rel_10/20/35` 应存在一个同时强于基线的平滑区间。实际权重非常敏感：`path_rel_10` 2026 最好但 dev 不行，`path_rel_35` dev 略高但 2026 变差，说明贡献模型不是鲁棒增益层。

第四反事实：如果历史大涨股归因能直接转成一周收益策略，特征重要性中市场状态、60 日趋势、量能和近高位置应能穿透正式回放。实际它们只帮助解释赢家结构和状态翻转，不能稳定提高正式账户曲线。

### 77.6 判定

`rejected_with_signal`。

正式账户成交贡献归因是有用诊断：它清楚揭示了 2026 与开发期的 winner 结构翻转，也说明行业/概念、量能拥挤、短爆发和中期趋势质量都必须放在市场状态里解释。但把这些归因直接做成 Top20 内 reranker 后，收益只是在某些权重和年份局部改善，不能同时保住开发期和 2026，更远未达到几十倍目标。

下一步不继续在同一 path Top20 池内做贡献重排。研究应转向更高层的自然收益土壤：要么生成新的低相关候选池，要么重新定义账户级训练目标为“日频资金使用下的边际 sleeve 增益”，而不是在固定 path Top20 内微调前 5 个名字。

## 78. account_soil_gate_v1

### 78.1 假设

Exp77 说明固定 path Top20 内重排空间太窄，且开发期和 2026 的个股 winner 结构会翻转。一个更高层的可能性是：不是每天都适合开新 sleeve，真正的金子可能来自可因果识别的“账户土壤日”。本轮把粒度从股票降到 signal session，聚合 Top20 可交易候选池的多日横截面路径、量能、近高位置、行业/概念热度和市场宽度特征，训练 session 级模型预测当天 path 前 5 个买入候选的可执行净收益代理。

核心问题：如果市场/候选池土壤能被 T 日可见特征识别，那么跳过预测土壤最差的 25% 交易日，应至少提升 2026 forward 或在不明显牺牲开发期收益的情况下压低回撤。

### 78.2 产物和口径

实现仍在 `/tmp`，不合代码。输入为 Exp77 的 path Top20 attribution v2，每个 signal session 聚合为一行。为避免未来信息泄漏，策略化 gate 不使用 `pool_rank_target_spearman`、`tradable_count`、`selected_count`、winner/loser rate 或任何 T+1 可交易/未来收益派生字段，只用 T 日候选池聚合特征。

```text
/tmp/quantx-research/account-soil-v1/analyze_account_soil_panel.py
sha256:cdfb2366b2ab543aaba2e273d6dad5b7ba42c98018ffc17357314059053cb204

/tmp/quantx-research/account-soil-v1/write_account_soil_gate_predictions.py
sha256:f847612571b3856bd8fa40c58da7eda641218932db474ebb48e7a6dc2fa7299f

/tmp/quantx-research/account-soil-v1/pathseq_pool20_account_soil_panel_v1.json
sha256:3b6408e8e555a09c592641396ddf088f1453c582608f25f7241c036cc3a5880f
```

gate 口径：`keep_ratio=0.75`，即保留预测土壤分数最高的 75% session，直接透传原 path Top20 预测；被跳过的 session 不开新 sleeve。正式账户仍为 Top20 pool、补位买5、hold5、`gap25+hist120`。

```text
/tmp/quantx-research/account-soil-v1/account_soil_gate_keep75_path20_dev_2022_2025_predictions.json
sha256:155571de875bbeaa85fba5893df18718b32f9f589c5d9fa86a22dca7daa8d441

/tmp/quantx-research/account-soil-v1/account_soil_gate_keep75_path20_val63_2026_predictions.json
sha256:447b3c7e5a48da9946c7ea95459a9a8098901f660110a967b91e88eff56284c3
```

### 78.3 诊断发现

session 层面有一些解释性相关性，但它们很弱，且很多强相关项不能用于因果策略。

| 口径 | session 数 | path top5 净收益代理均值 | tradable 均值 | 主要可见相关信号 |
| --- | ---: | ---: | ---: | --- |
| 2022-2025 | 963 | `+0.950%` | `+0.634%` | 候选池 `rps60_mean` 偏低、`rps1_top5mean` 偏高、量能分散更高时略好 |
| 2026 | 118 | `+1.238%` | `+0.310%` | `ret1/rps1` 分散度、低 `rps20/ma20_dist`、低市场宽度略相关 |

最强的 `pool_rank_target_spearman` 在 dev/2026 都很强，但它用未来 `target` 计算，只能说明 path 原始排序在坏日子里更容易反向，不能作为策略特征。剔除泄漏项后，剩余信号强度不足，且 2026 与开发期仍存在风格翻转。

label 层 gate 已经显示方向不稳：

| 口径 | kept sessions | skipped sessions | kept top5 proxy | skipped top5 proxy | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| dev walk-forward | 904 | 59 | `+0.893%` | `+1.819%` | 主要只在 2025 跳过，且跳过日反而更厚 |
| 2026 forward | 88 | 30 | `+1.040%` | `+1.817%` | 明确学反 |

### 78.4 正式账户结果

正式账户结果：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 袖套数 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 约 7.69 | 940 | 基线 |
| soil gate keep75 | 2022-2025 | 4.51x | `-37.35%` | 17.19 | 7.67 | 886 | 收益明显变薄，回撤没有改善 |
| path baseline | 2026 | `+19.47%` | `-17.42%` | 14.52 | 8.79 | 107 | 基线 |
| soil gate keep75 | 2026 | `+15.45%` | `-16.09%` | 11.73 | 8.90 | 83 | 小幅降回撤但切掉收益 |

正式回放产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_soil_gate_keep75_path20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:d74bd81971cd4ff64be753129131a9442c9f986241ad9a552777d7c776990d39

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_soil_gate_keep75_path20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:ac2d14104ecbf6af51a05de065854449e24950207c0b4ee1956786d76a4dc000
```

### 78.5 反事实分析

第一反事实：如果 session 土壤可被当前候选池聚合特征识别，跳过低预测土壤日应至少在 2026 改善收益或明显改善回撤。实际 2026 从 `+19.47%` 降到 `+15.45%`，回撤只小幅从 `-17.42%` 到 `-16.09%`，收益弹性被切掉。

第二反事实：如果它是有效的风险 gate，开发期应以更浅回撤换取少量收益损失。实际开发期收益从 5.48x 降到 4.51x，而最大回撤仍为 `-37.35%`，说明没有切中主要风险段。

第三反事实：如果只是 keep ratio 不合适，label 层至少应显示被跳过日收益更差。实际 dev 和 2026 的 skipped session 代理收益都高于 kept session，说明模型方向本身不稳，不应继续调阈值。

第四反事实：如果候选池整体状态是主收益土壤，session 级特征应比个股重排更稳。实际仍受开发期/2026 风格翻转影响，说明“日子好坏”不是当前可见特征能稳定识别的主矛盾。

### 78.6 判定

`rejected`。

账户日级土壤 gate 不成立。它把低土壤日识别反了，正式账户也只表现为少开仓、少收益、略降回撤。下一步不继续做 session gate、风险开关或阈值扫描；方向切回“自然候选生成”，尤其是要寻找不依赖固定 path Top20 的新右尾来源，或直接学习日频边际 sleeve 的候选池生成，而不是决定某天是否交易。

## 79. fill_quality_rerank_v1

### 79.1 假设

正式账户里有一个容易误读的现象：`path Top20 pool + 补位买5` 明显强于直接买 `path Top5`。这可能意味着原始 path 排序前几名包含更多涨停/停牌/高拥挤/不可兑现样本，而经过正式交易过滤后，被补位买入的 rank6+ 候选反而更“冷静”、更可交易。

本轮先做补位归因，再用一个很窄的手写分数验证：偏向中期趋势仍在、短期不过热、不过度贴近高位、市场宽度不差的候选，并以 `75% path + 25% fill_quality` 在 path Top20 内重排。

核心问题：如果“补位型质量”是真正可主动排序的 alpha，而不是正式执行过滤后的幸存者偏差，那么弱融合后应同时改善开发期和 2026 正式账户。

### 79.2 产物和口径

实现仍在 `/tmp`，不合代码。归因输入复用 Exp77 的 path Top20 attribution v2。

```text
/tmp/quantx-research/formal-trade-attribution-v1/analyze_fill_candidate_edge.py
sha256:80a9ed53422174576994d33e55ecbb1982ae1a2abbe605bc18fe9a19bf16f7d8

/tmp/quantx-research/formal-trade-attribution-v1/fill_candidate_edge_pathseq_pool20_buy5_v1.json
sha256:7c340fbd9f707e44cc13ee6e1509070a844e153ca4fb9575f1027bf9ccbfe250

/tmp/quantx-research/fill-quality-v1/write_fill_quality_predictions.py
sha256:a5500e27da12dc959e21ad6655002bd8bfd3bf27f7721a8d98a9b425dd822a87
```

PredictionStore：

```text
/tmp/quantx-research/fill-quality-v1/fill_quality_path25_path20_dev_2022_2025_predictions.json
sha256:8e14dd0375957b30503de7d1f5d484272b0546381f890fc8e1d3470a8ceadd24

/tmp/quantx-research/fill-quality-v1/fill_quality_path25_path20_val63_2026_predictions.json
sha256:e9552d03496f21c7ddde2716e70400301ed3d4e5015e80688695745ddffb3caf
```

正式账户仍为 Top20 pool、补位买5、hold5、`gap25+hist120`。

### 79.3 补位归因

正式买入样本中，rank6+ 补位候选的均值确实略高于直接 rank1-5：

| 样本 | direct rank1-5 count | direct mean | fill rank6+ count | fill mean | 主要差异 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | 3,940 | `+0.876%` | 875 | `+1.283%` | 补位候选 `market_breadth20`、`ret60/rps60` 更高，`rps1` 和 `high20_pos_rank` 更低 |
| 2026 | 449 | `+1.196%` | 141 | `+1.369%` | 补位候选 `market_breadth20` 更高、`rps1/rps5/high20_pos_rank` 更低，`rps60` 略高 |

但全体可交易 rank6-20 并不更好。开发期全可交易 rank1-5 均值 `+0.876%`，rank6-10 只有 `+0.508%`；2026 rank1-5 为 `+1.196%`，rank6-10 只有 `+0.423%`，rank11-20 更弱。说明补位样本好，不等于中段候选整体好；它可能只是正式执行过滤后的条件样本。

### 79.4 结果

label/代理层结果：

| 口径 | 2022-2025 top5 net proxy | 2026 top5 net proxy | 判读 |
| --- | ---: | ---: | --- |
| path | `+1.034%` | `+1.322%` | 基线 |
| pure fill_quality | `+0.420%` | `+0.176%` | 明显变薄 |
| `path_fill_quality_25` | `+1.106%` | `+1.146%` | dev 小升，2026 下降 |

正式账户结果：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 7.69 | 基线 |
| `path_fill_quality_25` | 2022-2025 | 5.14x | `-37.10%` | 18.63 | 7.71 | 收益下降，回撤几乎不变 |
| path baseline | 2026 | `+19.47%` | `-17.42%` | 14.52 | 8.79 | 基线 |
| `path_fill_quality_25` | 2026 | `+18.96%` | `-16.61%` | 15.63 | 8.86 | 小幅降回撤但收益下降 |

正式回放产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_fill_quality_path25_path20_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:3a2126e9378fb1d6e6c63b04872db8d1b0db80ad108c5be6e24219a33d53ed3a

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_fill_quality_path25_path20_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:ade0936f9da35d77b48e0e6bdf86e91dd023307adbd0174b02b0b57f7244ee43
```

### 79.5 反事实分析

第一反事实：如果补位候选质量是可主动排序的 alpha，`fill_quality` 本身不应显著弱于 path。实际 pure fill_quality 在 dev/2026 都大幅变薄，说明公式捕捉的是补位样本的解释性外观，不是可独立交易的收益源。

第二反事实：如果弱融合能把补位优势账户化，应至少不伤 2026。实际 2026 从 `+19.47%` 降到 `+18.96%`，代理层也从 `+1.322%` 降到 `+1.146%`。

第三反事实：如果收益来自“rank6+ 本身更好”，全体 rank6-20 可交易候选应优于 rank1-5。实际两端都不是，只有被正式执行过滤后实际买入的少量 fill 样本均值更高。

第四反事实：如果这是执行约束主导的 alpha，主动排序应降低拒单并增厚收益。实际拒单没有实质改善，dev/2026 都表现为收益略降、回撤略降。

### 79.6 判定

`rejected_with_signal`。

补位现象是真实诊断信号，但不是可直接排序的主 alpha。它说明正式账户过滤会把部分高拥挤头部样本替换成更冷静的候选，因而 pool-buy5 优于直接 Top5；但主动按这种“冷静中期趋势”重排，会把 path 的有效强信号稀释掉。

下一步不继续在 path Top20 里做补位质量公式。要寻找几十倍量级，需要离开固定 path pool：要么生成新的日频 sleeve 候选池，要么把模型训练目标直接改成“在正式账户约束下，今天新增一个 sleeve 的边际贡献”，而不是事后解释哪些补位股票幸存。

## 80. head_union_formal_replay_v1

### 80.1 假设

Exp21/22 的多模型头部并集是少数比较“自然”的候选生成方向：它不是在同一个 path Top20 里调 6-10 名权重，而是让多个训练窗口、标签周期和随机种子的头部股票自然投票形成候选池。此前标签层和简化账户显示 7model head6 在开发期有较好收益弹性，但尚未用 Exp67 之后统一的正式 overlap account 复核。

本轮只做正式账户裁判：同样使用补位买5、hold5、`max_open_gap=0.25`、`min_history_days=120`，并复核两个自然头部并集 store：7model head6 Top15 与纯 5 日 3seed head5 Top15。

### 80.2 产物和覆盖

本轮没有新合代码，复用已有 PredictionStore 并进入正式账户回放。

7model head6：

```text
/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_dev_2021_2025_top15_predictions.json
sha256:d85342d74363fb2fd49be5e64e5540d54355658fe20666dbe84aa9a286ced023

/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_val63_2026_top15_predictions.json
sha256:549311ba4aa8fc5a120d432dd5af5ccc6b3426c4550b8dc05a6e60a73d8586e9
```

覆盖审计：开发期 1212 个 session，平均每期 14.80 只，134 期少于 15 只；2026 有 124 个 session，平均每期 14.27 只，49 期少于 15 只。

纯 5 日 3seed head5：

```text
/tmp/quantx-research/5d-head-union-v1/head_union_3x5d_head5_base5d_anchor_dev_2021_2025_top15_predictions.json
sha256:383538e38a523de221affa8166ec06e2a6e02e4f0c6699ff8c5146000bb32c87

/tmp/quantx-research/5d-head-union-v1/head_union_3x5d_head5_base5d_anchor_val63_2026_top15_predictions.json
sha256:31230706ab699dcba603de91d3fb6bba4f26c974be9caeb7bce4065c1fe18380
```

覆盖审计：开发期 1212 个 session，平均每期 7.91 只；2026 有 124 个 session，平均每期 7.59 只。该 store 天然更窄，所有 session 都少于 15 只。

### 80.3 正式账户结果

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 7.69 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| 7model head6 Top15 | 2022-2025 | 5.85x | `-31.35%` | 14.39 | 7.68 | `+31.95%、+30.82%、+74.44%、+95.87%` |
| 7model head6 Top15 | 2026 | `+12.77%` | `-17.37%` | 12.63 | 7.60 | `+12.77%` |
| 3x5d head5 Top15 | 2022-2025 | 5.21x | `-35.74%` | 15.86 | 7.75 | `+37.04%、+41.90%、+13.71%、+136.36%` |
| 3x5d head5 Top15 | 2026 | `-13.53%` | `-25.48%` | 13.22 | 7.59 | `-13.53%` |

正式回放产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_head_union_7model_h6_anchor_top15_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:04de470273847ab6f1cb16eac23e1420a5bcccd387b09d2f63b4d13c68c059e5

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_head_union_7model_h6_anchor_top15_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:ca99a2b3750739913b726618b660bf0cca84ea26b1594a3d163c5a6cf27f05f0

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_head_union_3x5d_h5_anchor_top15_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:442a31654a042f11fff4630ca0e733272b35468efa32e69f37e133611eb60a34

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_head_union_3x5d_h5_anchor_top15_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:41d00c40a7474ad4f81a87b612d26ab8240474159f9a7239a6fa6ea50b86e4e1
```

### 80.4 反事实分析

第一反事实：如果多模型头部并集是比 path 更鲁棒的自然候选源，正式账户下应同时改善开发期和 2026。实际 7model head6 开发期从 5.48x 到 5.85x，回撤从 `-37.35%` 到 `-31.35%`，但 2026 从 `+19.47%` 降到 `+12.77%`。

第二反事实：如果 7model 的问题来自 10 日标签或短历史模型污染，那么纯 5 日 3seed head-union 应修复 2026。实际 3x5d head5 在 2026 为 `-13.53%`，比 7model 更差，说明不是简单的模型族污染。

第三反事实：如果候选数不足是主因，7model head6 平均接近 15 只且平均唯一持仓 12.63，应至少明显缓解 2026。实际仍弱于 path baseline，说明问题不是单纯持仓宽度，而是 2026 头部共识选到的股票结构不对。

第四反事实：如果这个方向只是 path 尾部调权的替代，它不该改善开发期回撤。实际开发期回撤明显改善，说明头部并集确实有独立结构；但它还不是可穿越 2026 的主 alpha。

### 80.5 判定

`rejected_with_signal`。

自然头部并集在开发期是有价值的低相关候选源，尤其 7model head6 能把正式账户开发期从 5.48x 提到 5.85x，同时降低回撤。但它没有通过 2026 forward：7model 只有 `+12.77%`，纯 5 日版本甚至为 `-13.53%`。因此不合代码，也不继续调 head 大小、seed 数量或 5d/10d 配比。

下一步应离开“已有模型头部取并集”的范式，转向新的候选生成标签：从历史可交易大涨路径和正式账户边际 sleeve 贡献里反推候选土壤，重点学习“容易兑现的一周右尾”，而不是解释事后大涨概率或调现有候选族权重。

## 81. account_marginal_sleeve_target_v1

### 81.1 假设

Exp77-80 连续说明两个问题：第一，固定 path Top20 内的单股重排太窄；第二，自然头部并集虽然改善开发期结构，但没有穿越 2026。于是本轮把目标改到更账户化的宽池候选生成：在基础 5 日 ML Top500 内，为每个候选计算一个近似“账户边际 sleeve 可执行净收益”标签，惩罚明显不可入场的 open gap、次新历史不足、类涨停开盘，并用 5 日 open-to-open 粗成本收益作为训练目标。

核心问题：如果正式账户损耗和可执行性是当前主矛盾，那么这个账户边际标签应在正式 overlap account 下同时改善开发期和 2026，而不是只提高代理层 entry_ok 或降低回撤。

### 81.2 产物和口径

实现仍在 `/tmp`，不合代码。脚本复用第 40 轮 path sequence 的 Top500 宽池特征，包括 1-120 日 RPS 路径、量能、近高/低位置、行业/概念强度、市场收益、宽度和横截面离散度。

为控制实验成本，本轮训练标签采用向量化近似：

1. 入场日前 open 有效。
2. 上市历史不少于 120 个有效 open。
3. T+1 open 相对 T close 的绝对跳空不超过 25%。
4. 类涨停/跌停开盘，即绝对跳空超过 9.5%，给入场失败负标签。
5. 成功入场后，用 5 日 open-to-open 收益减 `0.314%` 粗成本作为目标。

真正能否通过停牌、涨跌停、手数、最低佣金和滑点，仍以后续正式 overlap account 为裁判。

脚本：

```text
/tmp/quantx-research/account-marginal-sleeve-v1/write_account_marginal_sleeve_predictions.py
sha256:c72159204d9fbcd51e4b3ad05b9adb9fe752095c23b8d6cd1f66ebf9407cb53f
```

PredictionStore：

```text
/tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_path_exec20_pool500_dev_2021_2025_top20_predictions.json
sha256:cc29a33894647f0f0cec236c6906d5d1bda4517926109d759751113137df66a0

/tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_base_exec20_pool500_dev_2021_2025_top20_predictions.json
sha256:8143b8b913aa2e6b33ba83aeb585ad04936a30e3b9d3c96ba55fdc34e497099e

/tmp/quantx-research/account-marginal-sleeve-v1/account_marginal_sleeve_base_exec20_pool500_val63_2026_top20_predictions.json
sha256:1a8deaeea87bcc6c8091a543bb66b5c465132d0fdf984b6026e6e0b50a88766d
```

正式账户仍为 Top20 pool、补位买5、hold5、`gap25+hist120`。

### 81.3 代理层结果

`base_exec_20` 在近似标签层确实改善了可执行性和均值，且 2022-2025 每年都比基础 Top20 更厚：

| 口径 | 区间 | entry ok | 平均日 exec target | 逐年 exec target |
| --- | --- | ---: | ---: | --- |
| base Top20 | 2022-2025 | `96.87%` | `+0.569%` | `+0.438%、+0.257%、+0.602%、+0.987%` |
| `base_exec_20` | 2022-2025 | `99.72%` | `+0.814%` | `+0.532%、+0.421%、+0.985%、+1.331%` |
| `path_exec_20` | 2022-2025 | `99.49%` | `+0.686%` | `+0.216%、+0.198%、+1.058%、+1.284%` |
| base Top20 | 2026 | `98.26%` | `+0.145%` | `+0.145%` |
| `base_exec_20` | 2026 | `99.15%` | `+0.189%` | `+0.189%` |

特征重要性高度集中在市场风险和横截面环境：开发期前六位为 `market_ret60_median`、`market_ret5_median`、`market_breadth60`、`market_ret20_disp`、`market_ret20_median`、`market_breadth20`；2026 forward 也主要是 `market_ret20_disp`、`market_ret60_median`、`market_breadth60`、`market_ret5_median`、`market_ret20_median`、`market_breadth20`。这说明模型确实在学“市场风险/宽度/离散度”，不是噪声特征。

### 81.4 正式账户结果

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 逐年/前向收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| path baseline | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 7.69 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| `path_exec_20` | 2022-2025 | 2.85x | `-41.27%` | 17.02 | 7.63 | `+5.00%、-4.63%、+64.83%、+67.86%` |
| `base_exec_20` | 2022-2025 | 3.92x | `-27.02%` | 17.59 | 7.64 | `+0.80%、+17.31%、+58.23%、+108.88%` |
| path baseline | 2026 | `+19.47%` | `-17.42%` | 14.52 | 8.79 | `+19.47%` |
| `base_exec_20` | 2026 | `-4.60%` | `-19.85%` | 14.79 | 7.82 | `-4.60%` |

正式回放产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_path_exec20_pool500_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:a75c9df49f4a4f5d45de2db344288d5a84828165c55daf36548fcae4d5ef7275

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_base_exec20_pool500_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:5ff5ed16adef7bf700fa177eeded079caab450f3a3442980176e58f9de87b9f1

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_account_marginal_base_exec20_pool500_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:3b2e11e56624d79f97f186aca11326ca15096ad143a9099740c1e46bfc25f052
```

### 81.5 反事实分析

第一反事实：如果正式执行损耗是主矛盾，`base_exec_20` 应该在正式账户中明显超过 path baseline。实际开发期只有 3.92x，低于 path baseline 5.48x；2026 更从 path 的 `+19.47%` 变成 `-4.60%`。

第二反事实：如果账户标签只是需要 path 锚来保留右尾，`path_exec_20` 应该至少接近 path baseline。实际开发期只有 2.85x，且 2023 为 `-4.63%`，说明把可执行净收益目标叠到 path 上会破坏 path 的右尾结构。

第三反事实：如果模型学到的是收益土壤而不是防守过滤，正式账户收益和回撤应同时改善。实际 `base_exec_20` 回撤从 `-37.35%` 降到 `-27.02%`，但收益从 5.48x 降到 3.92x；这是典型防守过滤器，而不是新 alpha。

第四反事实：如果市场风险、宽度和离散度是足够的因果特征，2026 应至少不转负。实际 2026 近似标签小幅增厚，但正式账户为 `-4.60%`。说明这些状态特征能解释可执行性和一部分风险，但不能稳定预测“可正式兑现的一周右尾”。

第五反事实：如果不可买/跳变样本是导致高收益无法合并的主要来源，惩罚入场失败后应保住收益厚度。实际入场成功率从约 `96.87%` 提到 `99.72%`，收益厚度却下降，说明高收益来源不只是执行污染；过度优化可执行性会把高弹性右尾一起切掉。

### 81.6 判定

`rejected_with_signal`。

账户边际 sleeve 标签是有效诊断工具，但不是当前策略方向。它清楚证明模型能学习市场风险、宽度和横截面离散度，也能提高近似可执行率；但正式账户下收益显著变薄，2026 甚至转负。这个方向本质是在学习“更容易买、路径更平滑、风险更低”的股票，而不是学习“更容易出几十倍收益的自然右尾”。

下一步不继续调 `entry_fail_target`、`cost_proxy`、融合权重或 TopK。更有希望的方向不是单股可执行标签，而是**组合结构层面的收益放大**：寻找能在正式账户中保留右尾、但不靠不可买污染的机制，例如日频多 sleeve 的资本分配方式、低相关 Alpha 家族的并行组合、或先验更强的行业/概念主线识别，而不是把所有候选都拉向平滑可执行。

## 82. half_life_oracle_and_curve_combo_v1

### 82.1 假设

Exp67 之后的正式账户结果暴露出一个更基础的问题：同一个 path Top20 候选源在开发期更喜欢较长持仓，在 2026 更喜欢较短持仓。固定 hold5 是折中，hold10 开发期强但 2026 弱，hold2 开发期弱但 2026 强。如果这个半衰期状态可以被因果识别，就可能在不调尾部权重的情况下提高资本效率。

本轮同时测试三个反事实：

1. 固定多持仓期并行组合能否自然穿越开发期和 2026。
2. 既有正式曲线之间的静态组合是否已经足够接近目标。
3. 用账户级 shadow 曲线的未来短窗口表现训练持仓期选择器，是否能学到半衰期状态。

### 82.2 产物和口径

本轮仍只在 `/tmp` 研究，不合代码。正式账户口径保持 Top20 pool、补位买5、`max_open_gap=0.25`、`min_history_days=120`，只改变持仓期或曲线组合。

脚本和诊断产物：

```text
/tmp/quantx-research/hold-family-combo-v1/analyze_hold_family_combo.py
sha256:a15169d449de944c5f1a87c6cc710e0e6b74b3b07dd62befbd15544ee62ab93c

/tmp/quantx-research/hold-family-combo-v1/hold_family_combo_v1_diagnostic.json
sha256:12b83d445736a92f371c7f2b88e0ec0d8f02e6987c6d0d358e226d8199f15fe3

/tmp/quantx-research/formal-curve-combo-v1/analyze_formal_curve_combo.py
sha256:5e0b5dc2ace7b380b9298d2349a066aac673da2e259e29f2f8ec75f6f07ef4dd

/tmp/quantx-research/formal-curve-combo-v1/formal_curve_combo_v1_diagnostic.json
sha256:b3010e853708690a544691b801850bd19c62d26f2f1812430796377c3369c1fe

/tmp/quantx-research/dynamic-hold-state-v1/write_account_aware_hold_map.py
sha256:12770 bytes, tmp research script
```

账户感知 hold map 产物：

```text
/tmp/quantx-research/dynamic-hold-state-v1/account_aware_hold_map_w1_r0_dev_2022_2025.json
sha256:0a62beaaf628b78bc9ca422e6e522241587b7c90cc2bc5793c729022de05984a

/tmp/quantx-research/dynamic-hold-state-v1/account_aware_hold_map_w1_r0_val63_2026.json
sha256:d10f2ece578672b01cad711a20e6f9ab15c3c46df29d6fb9fbd22d338e2142e0

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_account_aware_w1_r0_dev_2022_2025.json
sha256:f5c3af6690a9fe10b94a7d4f2e9c92a9013bc58a1fdd0fd75cb0b420afb819c7

/tmp/quantx-research/dynamic-hold-state-v1/formal_dynamic_hold_account_aware_w1_r0_val63_2026.json
sha256:0768f6cafe32748e088e5a79105eacc0b1b63ccc36e365c3d47d6a63f390e388
```

### 82.3 固定持仓期和静态组合结果

固定持仓期基线显示出清楚的开发期/2026 半衰期翻转：

| 口径 | 2022-2025 | 2026 | 平均唯一持仓/持有期特征 |
| --- | ---: | ---: | --- |
| path hold2 | 3.64x，DD `-47.95%` | `+47.05%`，DD `-17.22%` | 2026 平均唯一持仓 7.48，平均日历持有 3.96 天 |
| path hold3 | 未作为主开发基线 | `+41.09%`，DD `-15.54%` | 短半衰期在 2026 明显占优 |
| path hold5 | 5.48x，DD `-37.35%` | `+19.47%`，DD `-17.42%` | 当前正式基线 |
| path hold10 | 6.51x，DD `-32.25%` | `+11.62%`，DD `-17.45%` | 开发期收益最高，但 2026 变慢 |

多持仓期静态组合不能解决目标量级：

| 静态组合 | 2022-2025 | 2026 | 判读 |
| --- | ---: | ---: | --- |
| `long_5_7_10` | 6.10x | `+13.91%` | 接近长持开发期，但前向弱 |
| `mid_3_5_7` | 5.51x | `+22.79%` | 折中有效但收益不厚 |
| `equal_2_3_5_7_10` | 5.28x | `+25.79%` | 分散改善 2026，但开发期下降 |
| `short_2_3_5` | 4.53x | `+37.87%` | 更适合 2026，开发期明显不足 |

10% 网格搜索也没有找到接近目标的静态配比：若要求开发期至少 5x，2026 最好约 `+38.80%`，但开发期只有约 5.03x；若要求 2026 至少 `+20%`，开发期最好约 5.97x，仍远低于几十倍量级。

### 82.4 Oracle 上限和曲线组合

每日非因果 oracle 显示机会不是不存在，而是半衰期选择含有大量未来信息：开发期可达 821.88x、最大回撤 `-24.02%`；2026 可达 2.97x、最大回撤 `-6.09%`。开发期 oracle 在 hold2/3/5/7/10 间都有选择，2026 也不是单一 hold2 垄断。这说明问题不是“短持或长持谁永久正确”，而是“当期应该快进快出还是让利润奔跑”的状态识别。

把已有正式曲线做静态组合也无法补足收益量级。纳入曲线包括 `base5d_top5`、`path_h2`、`path_h5`、`path_h10`、`head7_h6` 和 `vote4`：

| 约束 | 最好结果 | 判读 |
| --- | --- | --- |
| 开发期最大 | 100% `base5d_top5`，开发期 7.52x，但 2026 `-6.65%` | 头部高弹性不合平均持仓要求且前向失败 |
| 2026 正、开发期尽量高 | 开发期约 6.58x，2026 约 `+10%~+11%` | 仍低于 path hold5 的 2026 |
| 开发期 >=6x 下最大 2026 | 开发期约 6.01x，2026 `+15.99%` | 不如短持路径的 2026 |
| 2026 >=20% 下最大开发期 | 开发期约 5.61x，2026 `+20.05%` | 静态并行仍然太薄 |

### 82.5 账户感知 hold map 结果

为了避免 Exp78 的裸 open-to-open 持仓标签错位，本轮改用固定 hold2/3/5/7/10 正式账户 shadow 曲线的短窗口表现做标签，训练信号日前可见的账户状态到持仓期的映射。特征仍包括市场收益/宽度/离散度、候选池路径状态、行业/概念热度、候选族近期表现等。

标签层结果显示模型没有学到 2026 的短半衰期：

| 区间 | 样本数 | 模型选择 | 真实最佳 | 准确率 | 判读 |
| --- | ---: | --- | --- | ---: | --- |
| 2022-2025 | 716 | hold5 484 次、hold10 232 次 | hold2/3/5/7/10 分散 | `18.30%` | 基本学成长持/中持 prior |
| 2026 | 113 | hold10 113 次 | hold2 40、hold3 22、hold10 27 | `23.89%` | 2026 明显选错半衰期 |

正式动态持仓回放：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有日 | 持仓状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dynamic hold w1 | 2022-2025 | 6.44x | `-37.35%` | 20.94 | 9.50 | hold5 737 次、hold10 232 次 |
| dynamic hold w1 | 2026 | `+14.55%` | `-17.45%` | 23.36 | 17.39 | hold10 113 次、默认 hold5 11 次 |

开发期 6.44x 接近固定 hold10 的 6.51x，但没有超过；2026 `+14.55%` 低于固定 hold5 的 `+19.47%`，更大幅低于固定 hold2 的 `+47.05%`。这说明账户标签比裸收益标签更贴近目标，但仍只是把训练期 prior 学成“偏长持”，无法因果识别 2026 的快节奏兑现环境。

### 82.6 反事实分析

第一反事实：如果静态多持仓期并行已经足够鲁棒，它应同时接近 hold10 开发期和 hold2 2026。实际没有任何静态组合同时做到；越短持越牺牲开发期，越长持越牺牲 2026。

第二反事实：如果已有 alpha 曲线组合可以解决目标，只要网格配权即可出现几十倍级别。实际最好的合规静态组合仍在 5x-6x 附近，收益量级没有突破。

第三反事实：如果半衰期 oracle 主要由市场状态、候选池状态和行业/概念热度解释，账户感知 hold map 应在 2026 选择短持。实际它 2026 全部选择 hold10，正好与真实最佳方向相反。

第四反事实：如果问题只是持仓管理，动态 hold 应显著超过固定 path hold5。实际开发期只是接近长持基线，2026 还低于 hold5。瓶颈仍在候选生成和可因果识别的新信息源，而不是单纯退出规则。

### 82.7 判定

`rejected_with_signal`。

半衰期 oracle 很强，说明一周账户中确实存在巨大的“何时快卖、何时长拿”上限；但当前可见状态特征无法稳定识别它。静态持仓期组合和已有正式曲线组合都只能做收益/前向的折中，不能把策略推到几十倍量级。账户感知 hold map 进一步证明：即使训练目标直接使用正式 shadow 账户曲线，模型仍会在 2026 学错成偏长持。

下一步不继续调 hold2/3/5/7/10 权重、状态窗口或 hold map 分类器。更合理的转向是寻找新的信息源和候选生成机制：例如主线生命周期、行业/概念领导者扩散到补涨再退潮的阶段识别，或从多日横截面路径中直接预测“未来几天右尾是否会被快速兑现”。这类方向必须先证明能生成更厚的 Top15/20 候选，而不是只改善持仓期选择。

## 83. mainline_lifecycle_handcrafted_v1

### 83.1 假设

Exp82 说明半衰期选择上限很高，但现有市场状态、候选池状态和行业/概念热度不能稳定识别。于是本轮先测试一个更贴近 A 股主线交易语言的方向：把行业/概念主线生命周期拆成启动、龙头延续、补涨扩散和退潮过热四类可见状态，在基础 5 日 ML 候选池内重排 TopK。

核心问题不是“概念强不强”，而是：主线是否处于容易出一周右尾的阶段。如果手写生命周期阶段连标签层都不能增厚 Top15/20，则不进入账户层，也不训练更复杂模型。

### 83.2 产物和口径

脚本只在 `/tmp`，不合代码。它复用第 40 轮 path sequence analyzer 的行情、行业/概念映射和 base ML TopK 面板构建方式，只新增生命周期阶段分数和标签层评估。

```text
/tmp/quantx-research/mainline-lifecycle-v1/analyze_mainline_lifecycle.py
sha256:aabf5e857cec3225ae3f1260d565f3c69da77c3eeb54fd44c8d816f07c10c4ac

/tmp/quantx-research/mainline-lifecycle-v1/mainline_lifecycle_dev_2021_2025_diagnostic.json
sha256:94c32e7e62f83a6a712ae2344e88cb7726ca495d5c86f7f62556bececb8db2f6

/tmp/quantx-research/mainline-lifecycle-v1/mainline_lifecycle_val63_2026_diagnostic.json
sha256:0d61ce09d46ca713e460c63811394be5e876f132d5eaeebb5d256a0ed74108a8
```

阶段定义均为 T 日可见信息：

1. `ignition`：组内 3/5 日启动、近涨停比例、强股比例、量能温和确认、个股短期不过热。
2. `leader_continuation`：组内强度和个股 20/60 日趋势质量共同较强。
3. `laggard_catchup`：组强但个股相对组内不极端，试图捕捉补涨。
4. `exhaustion`：组和个股都过热、涨停/近涨停和量能过高、贴近短期高位。

评估只看未来 5 日 open-to-open 横截面超额 label，在 `pool100/200/500` 内比较 Top10/15/20/30。

### 83.3 结果

开发期 Top20 全样本：

| 口径 | mean daily label5 | 相对 base | positive day ratio | 判读 |
| --- | ---: | ---: | ---: | --- |
| base Top20 | `+0.015657` | `0` | `69.40%` | 基线 |
| `base_lifecycle_20` | `+0.009585` | `-0.006073` | `64.59%` | 弱融合也明显变薄 |
| `base_rotation_20` | `+0.009484` | `-0.006173` | `65.01%` | 补涨/轮动不增厚 |
| `leader_continuation` | `+0.009160` | `-0.006497` | `65.09%` | 直接选龙头延续低于 base |
| `laggard_catchup` | `+0.008691` | `-0.006967` | `65.26%` | 补涨阶段更薄 |

2026 Top20 全样本也没有翻转：

| 口径 | mean daily label5 | 相对 base | positive day ratio | 判读 |
| --- | ---: | ---: | ---: | --- |
| base Top20 | `+0.012018` | `0` | `68.64%` | 基线 |
| `base_lifecycle_20` | `+0.010130` | `-0.001889` | `65.25%` | 2026 仍低于 base |
| `base_lifecycle_35` | `+0.008281` | `-0.003737` | `65.25%` | 融合越重越差 |
| `base_rotation_20` | `+0.008158` | `-0.003861` | `67.80%` | 局部稳定但收益变薄 |
| `lifecycle_max` | `+0.006911` | `-0.005107` | `62.71%` | pure lifecycle 失败 |

2026 due5 小样本中 `laggard_catchup` 在 `pool500::top20` 相对 base 有局部正增量，mean label5 为 `+0.009272`、相对 base `+0.004209`。但这是 24 个左右调仓点的局部现象，且绝对收益厚度仍低于 2026 全样本 base，也不能覆盖开发期显著负增量。

### 83.4 反事实分析

第一反事实：如果主线生命周期的手写阶段抓到了自然右尾，至少弱融合 `base_lifecycle_20` 应在开发期不伤 base。实际开发期 Top20 label 从 `+0.015657` 降到 `+0.009585`，损伤非常大。

第二反事实：如果开发期只是阶段定义偏保守，2026 应出现清晰前向改善。实际 2026 全样本 Top20 仍从 `+0.012018` 降到 `+0.010130`。

第三反事实：如果补涨扩散是当前 2026 的核心收益土壤，它应在 Top15/20 全样本上变厚。实际只有 due5 小样本局部改善，不能作为稳健证据。

第四反事实：如果行业/概念主线是独立收益引擎，pure lifecycle 阶段应不低于 base ML 排序。实际 pure `lifecycle_max`、`leader_continuation`、`laggard_catchup` 都明显低于 base，说明它们更像事后叙事或风险解释，而不是可直接交易的 alpha。

### 83.5 判定

`rejected`。

手写主线生命周期阶段没有证明能增厚一周 Top15/20 候选。它在开发期和 2026 全样本均低于 base ML，且越强调生命周期分数，收益越薄。本轮不进入正式账户，不继续调阶段权重、阈值或 TopN。

不能由此推出“主线生命周期无价值”。更准确的结论是：用静态行业/概念快照和低阶组内强度/涨停/量能公式，无法把主线阶段转成可交易的合规 Top20 alpha。若未来重启，应改成可学习的序列/事件状态，且必须先用标签层证明开发期和 2026 同时增厚。

## 84. daily_overlap_sleeve_formal_recheck_v1

### 84.1 假设

日频重叠 sleeve 是少数看起来可能满足用户目标的结构：每天开一组 Top5，持有约一周，天然叠出十几只持仓，不需要降低 Rank6-10 权重。简化回放曾显示 path Top5 日频 sleeve 在开发期达到百倍级别、2026 也翻倍。因此必须用正式成交口径复核，判断收益是否来自真实可成交 alpha，还是来自买入简化假设。

本轮不是新策略开发，而是对 Exp67 之后 formal overlap account 与早期 simplified daily sleeve 的差异做裁判。

### 84.2 产物和口径

简化日频 sleeve 脚本：

```text
/tmp/quantx-research/daily-overlap-sleeve-v1/analyze_daily_overlap_sleeve.py
sha256:afefbb90b6886b01307f91a4f9366e5cf524a9711dd5fa4a534e852a1c3656bf
```

正式 overlap account 脚本：

```text
/tmp/quantx-research/formal-overlap-account-v1/analyze_formal_overlap_account.py
sha256:3f8208a78d5c9c25710f6d3787e2de866fdd75450e77710da6e41203c4b0f5cc
```

对比重点：

1. 简化脚本按 open 成交、等额 cohort、成本简化，虽然有 `gap25+hist120` 版本，但不经过正式 `Executor` 的涨跌停、停牌、手数、最低佣金和拒单记录。
2. 正式脚本用 `AStockExchange + Executor`，执行 T 日信号、T+1 open 入场，检查涨跌停/停牌/price jump，按 100 股手数和真实成本成交。
3. 所有正式结论只以后者为裁判。

### 84.3 简化回放结果

简化日频 sleeve 的 `path_topq Top5` 看起来非常强：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均 active cohort | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| simplified `path_topq Top5` | 2022-2025 | 123.89x | `-43.98%` | 17.65 | 4.97 | 达到几十倍以上量级 |
| simplified `path_topq Top5` | 2026 | 2.03x | `-15.71%` | 14.22 | 4.80 | 前向也强 |
| simplified `base5d Top5 gap25+hist120` | 2021-2025 | 93.04x | `-42.81%` | 15.22 | 4.75 | 结构上同样强 |
| simplified `base5d Top5 gap25+hist120` | 2026 | 1.49x | `-21.22%` | 13.72 | 4.80 | 低于 path，但仍正 |

如果只看这张表，日频 sleeve 似乎已经满足“高收益、平均持仓大于 5、一周持有、2026 forward 正”的结构要求。

### 84.4 正式成交复核结果

正式 overlap account 明显打掉了简化收益：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均 active sleeve | 平均日历持有 | 拒单 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| formal `pathseq top5` | 2022-2025 | 3.88x | `-37.90%` | 14.95 | 4.92 | 7.72 | 783 |
| formal `pathseq top5` | 2026 | `+13.11%` | `-16.62%` | 11.78 | 5.07 | 8.26 | 181 |
| formal `base5d top5` | 2021-2025 | 7.52x | `-33.11%` | 13.67 | 4.70 | 7.71 | 757 |
| formal `base5d top5` | 2026 | `-6.65%` | `-20.03%` | 12.15 | 4.85 | 7.58 | 99 |

正式 `pathseq top5` 开发期逐年收益为 `+46.41%、+16.13%、+21.95%、+84.56%`，四年全正但只有 3.88x；2026 为 `+13.11%`。正式 `base5d top5` 开发期更厚，但 2026 转负。

拒单结构解释了简化收益为什么不可直接引用。以 formal `pathseq top5` 为例：开发期拒单包括 `limit_up=243`、`price_jump=185`、`suspended=339`、`limit_down=16`；2026 拒单包括 `limit_up=21`、`price_jump=16`、`suspended=142`、`limit_down=2`。简化回放把很多正式买不进或不能连续按 open 标价的强势股收益计入了 cohort 曲线，正式账户则会丢失这些右尾或延后退出。

### 84.5 正式账户候选排名

把 formal overlap account 目录下现有候选按正式成交收益排序，开发期最高仍是：

| 口径 | 区间 | 最终倍数/收益 | 2026 对应表现 | 判读 |
| --- | --- | ---: | ---: | --- |
| formal `base5d top5` | 2021-2025 | 7.52x | `-6.65%` | 开发期强，前向失败 |
| formal `path hold10` | 2022-2025 | 6.51x | `+11.62%` | 开发期最高合规持仓之一，2026 弱 |
| formal `4fam vote_sum` | 2022-2025 | 6.50x | `+8.66%` | 回撤好，但 2026 弱 |
| formal `path hold7` | 2022-2025 | 6.07x | `+12.50%` | 折中，不够厚 |
| formal `head_union 7model` | 2022-2025 | 5.85x | `+12.77%` | 自然候选源但前向弱 |
| formal `path hold2` | 2022-2025 | 3.64x | `+47.05%` | 2026 强，开发期不足 |

当前 formal 裁判下，仍没有候选同时达到几十倍量级、平均持仓大于 5、年度全正、2026 forward 强。

### 84.6 反事实分析

第一反事实：如果日频 sleeve 的高收益来自真实可成交 alpha，formal `pathseq top5` 应接近 simplified 123.89x。实际只有 3.88x，说明简化收益主要依赖正式口径无法拿到的成交路径。

第二反事实：如果问题只是持仓数，formal `pathseq top5` 已经平均持仓 14.95、active sleeve 4.92，仍只有 3.88x。持仓数量不是瓶颈。

第三反事实：如果 base5d 头部弹性可自然叠加成策略，formal `base5d top5` 应通过 2026。实际 2026 为 `-6.65%`，说明开发期右尾结构不穿越前向。

第四反事实：如果 formal 执行只是轻微摩擦，拒单数和收益差距不应这么大。实际开发期 formal `pathseq top5` 有 783 笔拒单，其中涨停、跳变和停牌占绝大多数；这些正是短线强势右尾最容易出现的不可买/不可卖位置。

### 84.7 判定

`rejected_with_signal`。

日频重叠 sleeve 是有价值的账户结构线索：它自然把 Top5 信号扩成十几只持仓，不靠 Rank6-10 降权，也保持一周左右持仓。但简化回放的百倍收益不能作为策略证据；正式成交口径把它打回 3.88x，2026 只有 `+13.11%`。

下一步不继续引用 simplified daily sleeve 的高收益数字，也不围绕 Top5/Top8/Top10 做参数调优。若继续利用这个线索，必须直接在 formal 执行约束下学习“可买入且仍保留右尾”的候选生成器，而不是先在简化 open-to-open 世界里找高收益，再事后希望成交约束不破坏它。

## 85. formal_style_router_v1

### 85.1 假设

Exp84 把问题收缩到正式成交层：简化右尾被涨停、跳变、停牌和手数成本打掉后，仍需要在 formal 可交易候选里找到更厚的右尾。对 path Top20 formal attribution 的赢家/输家拆解显示一个明显翻转：开发期赢家更偏短期强、但 20/60 日不过热；2026 赢家更偏 60 日强、20 日位置高。一个自然猜想是：当前市场在奖励的横截面风格会随时间切换，如果能因果识别短动量/长趋势哪个正在赚钱，就可能在 formal 约束下保留右尾。

本轮只做 attribution 层诊断，不生成新 PredictionStore，不进入账户回放。裁判目标是 formal attribution 中已考虑可交易和退出延迟的 `net_return_proxy`。

### 85.2 产物和口径

脚本和输出均在 `/tmp`：

```text
/tmp/quantx-research/formal-style-router-v1/analyze_formal_style_router.py
sha256:b838405d55d0b7b941ca6821c3e656950a80b1098221ce0a038aeb1445272543

/tmp/quantx-research/formal-style-router-v1/formal_style_router_v1_diagnostic.json
sha256:b0799567783d73edabd76be1e739d92722e6adec454f028ce27656f8a3c077d3
```

输入为 path Top20 buy5 formal attribution：

```text
/tmp/quantx-research/formal-trade-attribution-v1/pathseq_pool20_buy5_dev_2022_2025_trade_attribution_v2.json
/tmp/quantx-research/formal-trade-attribution-v1/pathseq_pool20_buy5_val63_2026_trade_attribution_v2.json
```

比较的固定风格：

1. `path`：原始 path rank，经 formal tradability filter 后买前 5。
2. `short_momentum`：高 `rps1/rps5/ret1/ret5`，但 20/60 日不过热。
3. `long_trend`：高 `rps20/rps60/ret20/ret60`，接近高位和 MA 强度高。
4. `cool_long_trend`：长趋势强，同时短期 spike、量能 rank 和入场 gap 更低。
5. `short_vs_long_barbell`：短动量和冷静长趋势取最大。

路由器：

1. `train_prior`：训练历史平均最强风格。
2. `recent_w5/w10/w20/w40`：只用已完成 session 的最近窗口 realized `net_return_proxy`，选择近期平均最强风格。
3. `oracle`：同日事后选择最强风格，仅作为上限。

### 85.3 结果

开发期 walk-forward attribution 层结果：

| 口径 | mean daily target | 相对 path | 选择分布/判读 |
| --- | ---: | ---: | --- |
| oracle | `+0.034935` | `+0.025434` | 上限很高，说明风格间错误不完全重合 |
| `recent_w5` | `+0.012488` | `+0.002987` | 开发期可用，但只有 attribution 层小增量 |
| `recent_w10` | `+0.010650` | `+0.001148` | 小幅超过 path |
| `recent_w20` | `+0.010392` | `+0.000891` | 小幅超过 path |
| `short_momentum` | `+0.009712` | `+0.000210` | 接近 path |
| path | `+0.009502` | `0` | 基线 |
| `recent_w40` | `+0.009332` | `-0.000170` | 窗口拉长后失效 |
| `long_trend` | `+0.006182` | `-0.003320` | 开发期明显不适合 |
| `cool_long_trend` | `+0.005210` | `-0.004292` | 更弱 |

2026 forward attribution 层结果：

| 口径 | mean daily target | 相对 path | 选择分布/判读 |
| --- | ---: | ---: | --- |
| oracle | `+0.038632` | `+0.026255` | 上限仍高 |
| path | `+0.012377` | `0` | 2026 最强因果固定口径 |
| `recent_w20` | `+0.010268` | `-0.002108` | 低于 path |
| `recent_w40` | `+0.009673` | `-0.002704` | 低于 path |
| `recent_w5` | `+0.008739` | `-0.003638` | 低于 path |
| `recent_w10` | `+0.008016` | `-0.004361` | 低于 path |
| `long_trend` | `+0.004202` | `-0.008175` | 固定长趋势也不够 |
| `short_momentum` / `train_prior` | `+0.001586` | `-0.010790` | 训练期 prior 选短动量，2026 失效 |
| `cool_long_trend` | `-0.001382` | `-0.013759` | 明确失败 |

2026 的 `oracle` 选择分布为 path 41 次、long trend 22 次、cool long trend 22 次、short momentum 20 次、barbell 13 次，说明风格轮换空间确实存在；但所有因果路由都没有超过固定 path。

### 85.4 反事实分析

第一反事实：如果开发期/2026 的风格翻转能由近期 realized 风格表现识别，`recent_w5/w10/w20/w40` 至少应在 2026 超过 path。实际 2026 全部低于 path，说明近期赢家延续性不足。

第二反事实：如果 2026 只是从短动量切到长趋势，固定 `long_trend` 应超过 path。实际 `long_trend` 只有 `+0.004202`，低于 path 的 `+0.012377`。翻转不是简单从短切长，而是同日风格选择更复杂。

第三反事实：如果训练期平均最强风格能作为稳健先验，`train_prior` 应在 2026 不差。实际它选择 `short_momentum`，2026 只有 `+0.001586`，说明训练期 prior 会错误延续旧风格。

第四反事实：如果 oracle 主要来自某个单一风格被遗漏，那么固定该风格应接近 oracle。实际 oracle 的选择分散在 path、长趋势、冷静长趋势和短动量之间；问题是同日择风格不可由低阶近期表现因果识别。

### 85.5 判定

`rejected_with_signal`。

formal 可交易候选里的风格切换上限真实存在，开发期和 2026 的赢家结构也确实发生翻转。但近期 realized 风格收益路由不能穿越 2026，固定短动量/长趋势也都不能替代 path。这个方向暂不进入 PredictionStore 或正式账户回放。

下一步不继续调 recent window 或短/长风格权重。更合理的方向是**直接学习 formal 约束下的可买右尾替代品**：不是问“现在短动量还是长趋势强”，而是从那些 formal 买不进的右尾中寻找可成交的同组/同形态替代候选，或者把训练目标改成“正式可买且未来一周仍有右尾”的候选生成器。

## 86. right_tail_substitute_oracle_v1

### 86.1 假设

Exp84 显示简化日频 sleeve 的百倍收益被正式成交约束打掉，主要损耗来自涨停、跳变、停牌等右尾不可买路径。一个自然反事实是：如果 Rank1-5 中有强势但买不进的股票，Rank6-20 里是否存在同组、同形态或同风格的可买替代品，可以保留部分右尾暴露。

这不是降低 Rank6-10 权重来卡口径，而是只在正式账户本来就买不到头部强股时，检查可交易尾部是否有真正的替代 alpha。第一轮先做 oracle/手写规则诊断，不进入账户回放。

### 86.2 产物和口径

```text
/tmp/quantx-research/formal-right-tail-substitute-v1/analyze_right_tail_substitute.py
sha256:62f1a222c0c761302fece28f864c23f8c52e8d3d258f8d0959f11d6104090b7c

/tmp/quantx-research/formal-right-tail-substitute-v1/right_tail_substitute_v1_diagnostic.json
sha256:60bd7e83b8fef8e73c5d74fad49de9f580a7d4a697caf6ca51b0f639c3c193e1
```

输入仍是 path Top20 buy5 formal attribution v2。只统计 Rank1-5 中存在不可买头部的 session，并在 formal 可交易的 Rank6-20 候选中选择补位。目标为已考虑买入过滤和退出延迟的 `net_return_proxy`。

候选规则包括：

1. `tail_lowest_rank` / `actual_accepted`：正式回放自然补位的最低 rank 候选。
2. `same_group_lowest_rank`：和不可买头部同概念或同行业的最低 rank 可买候选。
3. `similar_lowest_distance`：按路径/强弱/量能特征找相似候选。
4. `tail_oracle` / `similar_oracle`：同日事后选择 Rank6-20 中未来净收益最高者，只作为上限。

### 86.3 结果

开发期替代候选诊断：

| 口径 | mean daily target | coverage | 平均 rank | 判读 |
| --- | ---: | ---: | ---: | --- |
| `tail_oracle` / `similar_oracle` | `+0.125203` | `100.0%` | 12.63 | Rank6-20 内确实有巨大右尾替代上限 |
| `same_group_oracle` | `+0.037152` | `55.2%` | 13.07 | 同组替代也有上限，但覆盖只有一半 |
| `same_group_lowest_rank` | `+0.015674` | `55.2%` | 11.43 | 手写同组最低 rank 只小幅高于自然补位 |
| `same_concept_lowest_rank` | `+0.015414` | `54.6%` | 11.56 | 类似，同概念不够强 |
| `actual_accepted` / `tail_lowest_rank` | `+0.012302` | `100.0%` | 6.49 | 正式 baseline 补位 |
| `similar_lowest_distance` | `+0.009953` | `100.0%` | - | 相似度最近反而更弱 |
| `same_ind_lowest_rank` | `+0.002684` | `11.3%` | - | 行业覆盖太低 |

2026 forward 替代候选诊断：

| 口径 | mean daily target | coverage | 平均 rank | 判读 |
| --- | ---: | ---: | ---: | --- |
| `tail_oracle` / `similar_oracle` | `+0.141816` | `100.0%` | 12.67 | 上限在 2026 仍然很高 |
| `same_ind_lowest_rank` | `+0.021426` | `8.0%` | 16.00 | 只有极低覆盖的局部信号 |
| `same_group_oracle` | `+0.016901` | `56.0%` | - | 同组 oracle 仍有上限但变弱 |
| `actual_accepted` / `tail_lowest_rank` | `+0.011110` | `100.0%` | 6.81 | 正式 baseline 补位 |
| `similar_lowest_distance` | `-0.004174` | `100.0%` | - | 相似度规则前向失败 |
| `same_group_lowest_rank` | `-0.010672` | `56.0%` | - | 同组最低 rank 前向反向 |
| `same_concept_lowest_rank` | `-0.010726` | `56.0%` | - | 同概念最低 rank 前向反向 |

### 86.4 反事实分析

第一反事实：如果右尾不可买只是纯损耗，Rank6-20 中不应存在高收益替代 oracle。实际开发期和 2026 的 `tail_oracle` 都在 `+12%~14%` 单次补位目标附近，说明可交易替代空间真实存在。

第二反事实：如果同组扩散可以自然替代头部不可买强股，`same_group_lowest_rank` 应在 2026 至少不差于自然补位。实际它在 2026 为 `-1.07%`，明显反向。

第三反事实：如果相似形态就是替代品，`similar_lowest_distance` 应稳定优于最低 rank。实际开发期更弱，2026 也为负，说明“像不可买强股”本身可能意味着同样拥挤或过热。

第四反事实：如果替代品只是 rank 更深带来的偶然右尾，oracle 的平均 rank 应极端靠后。实际平均 rank 约 12-13，说明机会在中尾部广泛存在，但需要因果选择器。

### 86.5 判定

`rejected_with_signal`。

右尾替代 oracle 是强信号：formal 可交易 Rank6-20 中确实有大量未来赢家，且开发期和 2026 都存在。但低阶同组、同行业、相似度和最低 rank 规则不能因果识别这些赢家，尤其 2026 的同组/同概念最低 rank 明显反向。

因此本轮不把手写替代规则进入账户层。唯一值得继续的是窄口径 ML：只在头部不可买的 session 内，学习 Rank6-20 formal 可交易候选的补位选择。

## 87. right_tail_substitute_ml_v1

### 87.1 假设

Exp86 证明 Rank6-20 有非因果替代上限，但手写规则失败。下一步改成窄口径 ML：训练样本只来自“Rank1-5 存在不可买头部”的 session，候选只用 formal 可交易 Rank6-20，目标是 `net_return_proxy` 或同日目标五分位。

如果这是真正的可学习右尾替代机制，ML 选择器应在 attribution 层和 formal account 层都稳定超过自然补位，并且不能牺牲开发期来换 2026。

### 87.2 产物和口径

Attribution 层 ML 诊断：

```text
/tmp/quantx-research/formal-right-tail-substitute-v1/write_right_tail_substitute_predictions.py
sha256:ca8678b9c24439ea6f8c13202ef6c1e9cbb4176bdba4c660eb676959f442bffe

/tmp/quantx-research/formal-right-tail-substitute-v1/right_tail_substitute_ml_v1_diagnostic.json
sha256:595ac949373dfcac56fc49bc8cc5966eca17ba58f37497226bd03e82d741b318
```

正式账户 PredictionStore 生成器：

```text
/tmp/quantx-research/formal-right-tail-substitute-v1/write_right_tail_substitute_store.py
sha256:fb124f766931a723f053c27f6f6602fc8e3d47b9542dd42fdce9eb646b1c56da
```

特征包括：原始 rank/score、1/5/20/60 日 RPS 和收益、量能、MA/高点位置、市场宽度和 20 日中位收益、行业/概念 rank、entry gap，以及派生的短动量、长趋势、冷静趋势分数。

模型包括：

1. `LGBMRegressor` 预测 formal `net_return_proxy`。
2. `LGBMClassifier` 预测同日目标五分位，用 top quintile 概率和 top-bottom spread 排序。
3. `blend_reg_20` 等弱融合变体，只在 Rank6-20 补位顺序上改动，Rank1-5 保持优先。

正式账户口径沿用 baseline：Top20 pool、buy5、hold5、`gap25+hist120`、T 日信号 T+1 open、`AStockExchange + Executor` 成交约束。

### 87.3 Attribution 层结果

在只看头部不可买 session 的篮子目标上，开发期结果为：

| 口径 | basket mean target | 相对 actual | tail mean target | 判读 |
| --- | ---: | ---: | ---: | --- |
| `oracle_tail` | `+0.047487` | `+0.033177` | `+0.125203` | 上限巨大 |
| `topq` | `+0.014710` | `+0.000401` | `+0.013094` | 只有极小增量 |
| `actual` / `tail_lowest_rank` | `+0.014309` | `0` | `+0.012302` | baseline |
| `spread` | `+0.014236` | `-0.000074` | `+0.012527` | 基本持平 |
| `reg` | `+0.014008` | `-0.000302` | `+0.009220` | 低于 baseline |
| `blend_reg_20` | `+0.013971` | `-0.000338` | `+0.011786` | 低于 baseline |

2026 forward attribution 层结果为：

| 口径 | basket mean target | 相对 actual | tail mean target | 判读 |
| --- | ---: | ---: | ---: | --- |
| `oracle_tail` | `+0.059122` | `+0.044976` | `+0.141816` | 上限仍巨大 |
| `reg` | `+0.016186` | `+0.002040` | `+0.017482` | 2026 有小幅增量 |
| `spread` | `+0.015793` | `+0.001646` | `+0.018672` | 2026 有小幅增量 |
| `blend_reg_20` | `+0.015742` | `+0.001596` | `+0.017200` | 2026 有小幅增量 |
| `actual` / `tail_lowest_rank` | `+0.014146` | `0` | `+0.011110` | baseline |
| `topq` | `+0.011347` | `-0.002799` | `+0.002814` | 前向失败 |

Attribution 结论是：ML 能在 2026 局部补位上找到一点信号，但开发期不稳；非因果 oracle 与因果模型之间仍有数量级差距。

### 87.4 正式账户回放结果

为了检查 attribution 小增量能否转成账户收益，生成 `reg`、`spread`、`blend_reg_20` 三个 PredictionStore 并进入 formal overlap account。

开发期 2022-2025：

| 口径 | 最终倍数 | 总收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline `path pool20 buy5` | 5.48x | `+447.55%` | `-37.35%` | 18.00 | 7.69 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| `reg` substitute | 5.05x | `+404.58%` | `-38.80%` | 18.33 | 7.70 | `+43.34%、+26.01%、+34.29%、+105.60%` |
| `spread` substitute | 5.52x | `+451.94%` | `-38.07%` | 18.51 | 7.76 | `+43.34%、+25.46%、+36.88%、+120.65%` |
| `blend_reg_20` substitute | 4.97x | `+397.41%` | `-39.33%` | 17.93 | 7.69 | `+53.83%、+24.94%、+24.18%、+105.40%` |

2026 forward：

| 口径 | 收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `path pool20 buy5` | `+19.47%` | `-17.42%` | 14.52 | 8.79 | baseline |
| `reg` substitute | `+21.13%` | `-17.76%` | 15.10 | 8.80 | 小幅高于 baseline |
| `spread` substitute | `+23.54%` | `-17.56%` | 15.53 | 8.82 | 小幅高于 baseline |
| `blend_reg_20` substitute | `+24.46%` | `-17.82%` | 14.37 | 8.82 | 2026 最好，但开发期低于 baseline |

### 87.5 反事实分析

第一反事实：如果 ML 已经学到右尾替代机制，开发期正式账户不应低于 baseline。实际只有 `spread` 从 5.48x 微升到 5.52x，增量很小且回撤略差；`reg` 和 `blend_reg_20` 均下降。

第二反事实：如果 2026 小幅增量来自稳健机制，`spread` 或 `blend_reg_20` 应同时在开发期 attribution 和账户层稳定增厚。实际开发期 attribution 基本持平或变差，账户层也没有显著改善。

第三反事实：如果问题只是 PredictionStore 排序表达，正式账户应放大 attribution 层的 2026 小增量。实际 2026 从 `+19.47%` 提到最高 `+24.46%`，有帮助但不改变收益量级，也不降低回撤。

第四反事实：如果右尾替代是当前主金矿，formal 账户应向几十倍目标靠近。实际最佳开发期仍只有 5.52x，距离目标差一个数量级。

### 87.6 判定

`rejected_with_signal`。

右尾替代方向有真实上限，且窄口径 ML 在 2026 有小幅 forward 改善；但因果选择器远远够不到 oracle，上限不能稳定转成账户收益。开发期最佳只从 5.48x 到 5.52x，2026 最好到 `+24.46%`，平均持仓和持仓周期合规但收益量级不合格。

下一步不继续调 LGBM 参数、blend 比例或 tail rank 范围。更重要的结论是：**固定 path Top20 内的补位微调已经接近边际收益上限**。要接近几十倍目标，必须离开“同一个 Top20 pool 里排序”的局部问题，寻找新的低相关候选土壤或更高频/多日路径结构，并且从一开始就以 formal 可交易路径为裁判。

## 88. early_right_tail_entry_v1

### 88.1 假设

Exp86/87 说明在不可买头部出现当天，从 Rank6-20 找替代品的因果增量很有限。另一个更根本的反事实是：这些后来变成 Rank1-5 且正式买不到的强势股，是否在更早的 1-10 个信号日前已经出现在 path Top20，并且当时仍可买。如果答案成立，真正的收益层不是“当天补位”，而是“提前发现即将进入右尾加速段的股票”。

本轮只做诊断，不训练模型，不生成 PredictionStore。核心检验分两面：

1. 从未来不可买头部事件倒查：它们在过去 1-10 个信号日是否曾经可买，且当时一周收益是否高。
2. 从当时所有可买 Rank6-20 正向看：未来 1-10 日会变成不可买头部的样本，是否在当时已经有稳定可识别收益边际。

### 88.2 产物和口径

```text
/tmp/quantx-research/early-right-tail-entry-v1/analyze_early_right_tail_entry.py
sha256:fc2417620f0988fe0e3c30a30dbfc85491eb5edf9540b428d2b243fdcc043ac0

/tmp/quantx-research/early-right-tail-entry-v1/early_right_tail_entry_v1_diagnostic.json
sha256:719690db4244e857fb4a778dcb69ec63c82b9b7c462343fca8b7824f7e8bd54e
```

输入为 path Top20 buy5 formal attribution v2。不可买头部定义为 Rank1-5 且 `filter_reason` 不是 `accepted` / `rank_after_buy_count`；提前可买足迹定义为同一股票在过去 1-10 个 signal session 中出现在 path Top20，且当时 formal attribution 标记为可交易并有 `net_return_proxy`。

### 88.3 结果

倒查不可买头部事件：

| 区间 | 不可买头部事件 | 过去 1-10 日曾在 Top20 | 过去 1-10 日曾可买 | 最近可买足迹 mean target | 最近可买足迹胜率 >3% | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | 875 | 67.43% | 9.49% | `+0.077833` | 61.45% | 事后足迹很强，但覆盖很低 |
| 2026 | 141 | 80.85% | 22.70% | `+0.024012` | 62.50% | 覆盖提高，但收益厚度大幅下降 |

按事件原因拆分最近可买足迹：

| 区间 | 原因 | 最近可买事件数 | 平均 lag | 平均 rank | mean target | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | `buy_untradable` | 73 | 3.90 | 5.86 | `+0.067189` | 倒查强 |
| 2022-2025 | `limit_like_open` | 10 | 5.00 | 9.30 | `+0.155537` | 样本极少但很强 |
| 2022-2025 | `insufficient_history` | 0 | - | - | - | 无可买历史 |
| 2026 | `buy_untradable` | 32 | 4.16 | 9.16 | `+0.024012` | 正，但远弱于开发期 |
| 2026 | `limit_like_open` | 0 | - | - | - | 无可买历史 |
| 2026 | `insufficient_history` | 0 | - | - | - | 无可买历史 |

正向看当时所有可买 Rank6-20：

| 区间 | 可买尾部样本 | 未来 1-10 日变不可买头部比例 | 未来事件样本 mean target | 非未来事件样本 mean target | edge | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | 13,937 | 0.61% | `+0.033895` | `+0.004710` | `+0.029185` | 正向 edge 存在但命中极稀疏 |
| 2026 | 1,629 | 1.66% | `-0.020850` | `+0.000702` | `-0.021552` | 2026 正向 edge 反向 |

按未来事件 lag band 拆分，开发期未来 1-3 日变不可买头部的样本 mean target 为 `+0.071795`，2026 同 band 仍为 `+0.032534`；但 2026 的 4-6 日和 7-10 日分别为 `-0.026190`、`-0.088114`。这说明如果有信号，也只可能是极短窗口的加速前夜，而不是稳定的 1-10 日提前入场土壤。

### 88.4 反事实分析

第一反事实：如果不可买右尾普遍可以提前买到，过去 1-10 日曾可买覆盖应较高。实际开发期只有 `9.49%`，2026 虽到 `22.70%`，仍不足以支撑高容量主策略。

第二反事实：如果提前足迹是可正向交易的 alpha，所有可买 Rank6-20 中“未来变不可买头部”的样本应在 2026 也有正 edge。实际 2026 为 `-0.021552`，说明倒查的强收益有明显幸存者偏差。

第三反事实：如果更早发现越有价值，长 lag 应稳定正。实际开发期 7-10 日已接近 0，2026 7-10 日显著为负，说明提前太久会买到尚未确认或已经脆弱的噪声。

第四反事实：如果 `limit_like_open` 是最有价值的提前入口，2026 应至少有可买历史样本。实际 2026 的 `limit_like_open` 最近可买事件数为 0，无法形成可验证规则。

### 88.5 判定

`rejected_with_signal`。

提前入场线索解释了部分简化高收益：有一小批未来不可买右尾在几天前确实可买，且事后收益很厚。但正向交易难点太大：未来变不可买头部的发生率只有 `0.61%/1.66%`，开发期 edge 到 2026 反转，且可用信号主要集中在极短 1-3 日窗口。

下一步不直接训练“未来是否变不可买头部”的稀疏分类器，也不围绕倒查样本做 winner mining。更合理的方向是离开单股不可买右尾链条，寻找**组层收益弹性**：A 股强势行情往往从概念/行业扩散到一篮子可买股票，如果个股龙头买不到，组内可交易组合或组层指数化代理可能比单个替代股更稳。

## 89. group_basket_substitute_v1

### 89.1 假设

Exp86/87 的单股补位只是在 path Top20 内重新找一只可买替代股，收益增量很小。A 股的强势行情常常以概念/行业为传播单位：龙头不可买时，真正可交易的代理可能不是某一只“最像龙头”的股票，而是同概念/同行业中一篮子仍可买成员。

本轮先做 attribution 层组篮子诊断，再把最稳的组质量排序写成 PredictionStore 进入 formal overlap account。若组层承接是有效收益层，应在开发期和 2026 同时改善账户收益，而不是只在 oracle 或局部样本有效。

### 89.2 产物和口径

Attribution 诊断脚本和输出：

```text
/tmp/quantx-research/group-basket-substitute-v1/analyze_group_basket_substitute.py
sha256:d6baed9d05a500bf5b98d60369522c4b7a88c40728b477413de0adaf34d83117

/tmp/quantx-research/group-basket-substitute-v1/group_basket_substitute_v1_diagnostic.json
sha256:ffa9b5cd8faea476ff0de65261dbe4fc139cc5f20f6037847c0926827c819151
```

PredictionStore 生成器：

```text
/tmp/quantx-research/group-basket-substitute-v1/write_group_basket_substitute_store.py
sha256:a1ed92beaa98db54486085c207a8d49f4d4cbf3a88147619c75da5e160dadf89
```

输入仍是 path Top20 buy5 formal attribution v2，行业/概念映射使用：

```text
data/meta/snapshots/industry_membership.csv
data/meta/snapshots/sector_membership.csv
```

规则：

1. Rank1-5 保持优先，不主动替换可买头部。
2. 当 Rank1-5 有不可买头部时，找这些股票同一级行业或同概念的 Rank6-20 formal 可交易成员。
3. `same_group_rank` 只按原 path rank 提前同组候选。
4. `same_group_quality` 按概念 5 日强度、概念强势占比、行业 5 日强度、个股 RPS 和不过热量能组合成质量分。
5. 正式账户沿用 Top20 pool、buy5、hold5、`gap25+hist120`、T 日信号 T+1 open、`AStockExchange + Executor`。

### 89.3 Attribution 层结果

只看头部不可买 session 中“已有可买头部 + 组篮子补位”的 combined target：

| 口径 | 开发期 combined target | 相对 actual | coverage | 2026 combined target | 相对 actual | coverage | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tail_oracle_basket` | `+0.047487` | `+0.033177` | 100.0% | `+0.059122` | `+0.044976` | 100.0% | Rank6-20 oracle 上限仍高 |
| `same_group_oracle_basket` | `+0.022740` | `+0.008430` | 55.18% | `+0.022911` | `+0.008765` | 56.00% | 同组 basket oracle 两端稳定有上限 |
| `same_group_quality` | `+0.016972` | `+0.002663` | 55.18% | `+0.015419` | `+0.001273` | 56.00% | 低阶质量规则两端小幅有效 |
| `same_concept_basket_rank` | `+0.016660` | `+0.002351` | 54.61% | `+0.012445` | `-0.001701` | 54.67% | 2026 不稳 |
| `same_group_basket_rank` | `+0.016617` | `+0.002307` | 55.18% | `+0.013647` | `-0.000499` | 56.00% | 仅同组 rank 前向不够 |
| `actual_fill` / `tail_basket_rank` | `+0.014309` | `0` | 100.0% | `+0.014146` | `0` | 100.0% | baseline |
| `same_ind_basket_rank` | `+0.006533` | `-0.007776` | 11.30% | `+0.090205` | `+0.076059` | 8.00% | 覆盖太低，不能引用为稳健结论 |

Attribution 层说明：同组 basket oracle 的上限比单股同组规则更稳定，`same_group_quality` 两端均有小幅因果增量；但 coverage 只有 55% 左右，且增量仍是 `+0.1%~0.3%` 单次目标级别，不是大幅收益层。

### 89.4 正式账户回放结果

开发期正式账户：

| 口径 | 最终倍数 | 总收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 逐年收益 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline `path pool20 buy5` | 5.48x | `+447.55%` | `-37.35%` | 18.00 | 7.69 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| `same_group_quality` | 5.93x | `+492.91%` | `-39.23%` | 18.18 | 7.72 | `+54.22%、+24.65%、+41.96%、+114.55%` |
| `same_group_rank` | 5.76x | `+476.08%` | `-38.06%` | 18.18 | 7.72 | `+53.99%、+26.38%、+37.51%、+112.08%` |

2026 forward：

| 口径 | 收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `path pool20 buy5` | `+19.47%` | `-17.42%` | 14.52 | 8.79 | baseline |
| `same_group_quality` | `+21.24%` | `-17.42%` | 14.89 | 8.74 | 小幅高于 baseline |
| `same_group_rank` | `+20.12%` | `-17.40%` | 14.74 | 8.76 | 接近 baseline |

### 89.5 反事实分析

第一反事实：如果组层承接只是 attribution 假象，formal account 不应改善。实际 `same_group_quality` 在开发期从 5.48x 提到 5.93x，2026 从 `+19.47%` 提到 `+21.24%`，说明信号确实能进入账户。

第二反事实：如果“同组优先”本身足够，`same_group_rank` 应接近 `same_group_quality`。实际 `rank` 两端都弱于 `quality`，说明组内质量/不过热信息有边际价值。

第三反事实：如果组层篮子已经解决右尾损耗，开发期收益应明显接近几十倍。实际最佳只有 5.93x，仍处于 formal path baseline 附近，说明它只是补位修补，不是主收益弹性层。

第四反事实：如果行业/概念本身可作为独立 alpha，coverage 低的同业/同概念 basket 应有稳定强表现。实际行业覆盖只有 8%-11%，概念 rank 在 2026 变负，说明必须结合原 path 个股结构和组内质量，不能直接买组。

### 89.6 判定

`rejected_with_signal`。

组层可交易篮子替代是目前右尾补位系列里最干净的小正增量：它在 attribution 层、开发期 formal account、2026 forward 中方向一致，且平均持仓和一周持有口径合规。但它仍只是把正式账户从 5.48x 推到 5.93x，最大回撤还略恶化到 `-39.23%`，远低于几十倍目标。

下一步不继续调组质量权重、概念黑名单或同组覆盖。这个实验告诉我们：行业/概念确实能作为**收益传播和补位解释层**，但不能靠在 path Top20 内做承接修补来突破收益天花板。下一条应寻找更高弹性的结构，例如把“组层热点”作为独立交易对象或组合袖套，而不是只在个股不可买时做局部替代。

## 90. sector_follower_group_hotspot_recheck_v1

### 90.1 假设

Exp89 证明组层承接能带来小幅账户增量，但仍依赖 path Top20 个股结构。下一条自然反事实是：如果 A 股的收益弹性来自行业/概念热点本身，那么不应只在 path 不可买时补位，而应把组层热点作为独立交易对象，在强概念/强行业里选一篮子可交易成员。

本轮不重新写同类脚本，而是复核已有 `sector_follower_ml_v1` 与 formal overlap account 产物。该路线已经在基础 ML Top500 宽池内构造行业/概念主升扩散、组内 follower/catch-up、组热度/宽度/放量/加速等特征，并输出 Top20 进入正式账户。

### 90.2 产物和口径

核心生成脚本：

```text
/tmp/quantx-research/sector-follower-ml-v1/write_sector_follower_ml_predictions.py
sha256:5b38af899d144ddb5990dd4534519b4fd1b55c72c06a90be73764e05dca0f35d
```

主要标签层 summary：

```text
/tmp/quantx-research/sector-follower-ml-v1/sector_follower_ml_pool500_blend40_dev_2021_2025_top20_predictions.summary.json
sha256:61b8c98ad3b726c21d03534fb2635d3818d11381c9b42f4617a7139d2c5c9ce0

/tmp/quantx-research/sector-follower-ml-v1/sector_follower_ml_pool500_blend40_val63_2026_top20_predictions.summary.json
sha256:9339080fb14009b4ce2ec5901687d0ee6a7cdb8648d9572fccfa7cc8286ed7a9
```

正式账户产物：

```text
/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_ml_pool500_blend40_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:9c0ea622b04b1bcccbbe054418d018ba75a2c6c29c175d86596e4b7847763c3a

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_ml_pool500_blend40_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:426a4dffce38acf6ba483f002e4829d8aa36e3db3ca4b7e036403e17d5cbf17c

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_executable_pool500_blend40_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:e053f3be3569d306d3de90b16616c192bac3e41c921c61eb69ec7867eed62eb1

/tmp/quantx-research/formal-overlap-account-v1/formal_overlap_sector_follower_executable_pool500_blend40_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:5cb214394db2109af4ad0aa2e5365d52b3fafea19d4634d0b94b69928f2b04e8
```

正式账户口径与近期实验一致：Top20 pool、buy5、hold5、`gap25+hist120`、T 日信号 T+1 open、formal `AStockExchange + Executor`。

### 90.3 标签层复核

`sector_follower_ml_pool500_blend40` 标签层 Top20：

| 区间 | base mean label5 | blend40 mean label5 | base raw5 | blend40 raw5 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2022-2025 | `+0.012564` | `+0.012863` | `+0.015190` | `+0.015489` | 标签层小幅增厚 |
| 2026 | `+0.012018` | `+0.012920` | `+0.008950` | `+0.009852` | 2026 标签层也小幅增厚 |

`executable` 目标的标签层更明显：开发期 `score_blend_40` mean label5 为 `+0.016309`，高于 `score_base` 的 `+0.011381`；2026 为 `+0.014746`，也高于 `score_base` 的 `+0.011418`。这说明组层热点/跟随特征在标签层并非纯噪声。

### 90.4 正式账户复核

但 formal account 没有把标签层优势转成收益突破：

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 逐年/判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline `path pool20 buy5` | 2022-2025 | 5.48x | `-37.35%` | 18.00 | 7.69 | 基线 |
| sector follower ML `blend40` | 2022-2025 | 4.44x | `-40.20%` | 16.81 | 7.69 | 低于 baseline，2024 只有 `+7.82%` |
| sector follower ML `blend25` | 2022-2025 | 4.61x | `-38.85%` | 16.82 | 7.68 | 仍低于 baseline |
| executable follower `blend40` | 2022-2025 | 3.29x | `-37.88%` | 17.60 | 7.61 | 可执行目标更防守但更薄 |
| baseline `path pool20 buy5` | 2026 | `+19.47%` | `-17.42%` | 14.52 | 8.79 | 基线 |
| sector follower ML `blend40` | 2026 | `+8.89%` | `-15.12%` | 15.35 | 8.24 | 前向明显低于 baseline |
| sector follower ML `blend25` | 2026 | `+2.65%` | `-14.85%` | 14.94 | 8.23 | 几乎失效 |
| executable follower `blend40` | 2026 | `+12.98%` | `-13.60%` | 16.08 | 8.31 | 回撤略低但收益仍低于 baseline |

### 90.5 反事实分析

第一反事实：如果组层热点本身是独立交易对象，formal account 应至少超过 path baseline。实际开发期和 2026 都低于 baseline，尤其 2026 `blend40` 只有 `+8.89%`。

第二反事实：如果问题只是不可执行噪声，`executable` 目标应修复账户收益。实际 executable `blend40` 开发期降到 3.29x，2026 仅 `+12.98%`；它降低了一部分回撤，但削掉了收益弹性。

第三反事实：如果标签层小增量能自然放大到账户层，账户曲线应和 label5 同向。实际标签层增厚，账户层变薄，说明组层热点更容易买到形式上强、但真实成交后弹性不足或持有路径不佳的股票。

第四反事实：如果独立组层热点能突破 5-6x 天花板，2024/2026 应明显改善。实际 2024 账户收益被压到低个位数，2026 低于 path，说明它没有解决最核心的正式收益厚度问题。

### 90.6 判定

`rejected_with_signal`。

行业/概念热点作为独立交易对象，标签层确实有小幅信息，但正式账户层不成立。它比 Exp89 的“组层补位”更主动，却更容易破坏 path 已经筛出的可交易右尾结构；可执行目标版本更防守，但收益量级进一步下降。

下一步不继续围绕组热度、组宽度、follower/catch-up 做独立组层选股。当前证据链已经很一致：行业/概念更适合作为解释、风险、补位和局部状态特征，而不是单独承担主收益引擎。要找几十倍收益，需要换到**账户结构/资金使用效率/低相关 alpha 并行**，或者寻找完全不同的可交易右尾来源，而不是继续在同一个股票排序空间里挤边际。

## 91. account_curve_portfolio_v1

### 91.1 假设

Exp86-90 连续说明，在固定 path Top20 股票排序空间里做右尾补位、组层承接、行业/概念热点独立化，收益都停在 5-6x 附近，或者在 2026 forward 明显变薄。一个必要的反事实是：问题是否不是 alpha 本身，而是账户结构没有把已有低相关曲线组合起来。

本实验不做动态择时、不加杠杆、不看 2026 后调权，只测试一个保守问题：把已经存在的 formal account 曲线做静态凸组合，是否天然能显著超过单条曲线，同时保住 2026。

### 91.2 产物和口径

诊断脚本：

```text
/tmp/quantx-research/account-curve-portfolio-v1/analyze_account_curve_portfolio.py
sha256:041b9723ccc56b450e192c80fa501afc95d309d1b7aec8150175e23aede592cf
```

诊断结果：

```text
/tmp/quantx-research/account-curve-portfolio-v1/account_curve_portfolio_v1_diagnostic.json
sha256:590f4e819d91444595b2d7bee07f623b9d403df2a028a5fe37ca46d33d3b2beb
```

输入曲线全部来自已有 formal account 产物：`path_hold5`、`path_hold10`、`path_hold2`、`four_family_vote`、`group_quality`、`head_union`。开发期为 2022-2025，forward 为 2026；权重只扫 10% 粒度的静态凸组合，并额外记录等权组合。

### 91.3 结果

| 组合 | 2022-2025 最终倍数 | 开发期最大回撤 | 2026 收益 | 2026 最大回撤 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| 等权六曲线 | 5.69x | `-32.67%` | `+20.19%` | `-16.48%` | 比 baseline 略平滑，但收益没有突破 |
| dev 最优：`four_family_vote` 100% | 6.56x | `-21.32%` | `+8.69%` | `-16.44%` | 开发期最好，但 2026 明显弱 |
| dev 次优：`path_hold10` 10% + `four_family_vote` 90% | 6.56x | `-22.19%` | `+8.98%` | `-16.31%` | 与单曲线几乎相同 |
| forward 最优：`path_hold2` 100% | 3.70x | `-47.95%` | `+47.22%` | `-17.22%` | 2026 强但开发期弱，且持仓周期太短 |
| forward 次优：`path_hold2` 90% + `group_quality` 10% | 3.93x | `-46.68%` | `+44.63%` | `-17.11%` | 仍是短持有风格暴露 |

### 91.4 反事实分析

第一反事实：如果已有曲线之间有足够独立的收益来源，静态组合应明显超过最好单曲线。实际 dev 最优仍退化为 `four_family_vote` 单曲线，混入其它曲线只带来很小变化。

第二反事实：如果组合结构可以同时保住开发期和 2026，等权或均衡组合应接近开发期 6x 以上并提高 2026。实际等权只有 5.69x，2026 `+20.19%`，和 path baseline `+19.47%` 接近。

第三反事实：如果 2026 最强曲线可直接作为收益突破来源，它在开发期也应至少不太差。实际 `path_hold2` 2026 `+47.22%`，但开发期只有 3.70x、回撤 `-47.95%`，且平均持有周期偏离一周目标。

第四反事实：如果已有 formal 曲线能靠账户拼接达到几十倍量级，10% 粒度的凸组合至少应出现接近 10x 的候选。实际最高仍约 6.56x，说明这些曲线高度共享同一收益土壤。

### 91.5 判定

`rejected_with_signal`。

静态账户曲线组合能略微平滑回撤，但不能打开收益上限。已有 formal 曲线不是足够低相关的 alpha sleeves，而是同一 path/ML 右尾土壤的不同投影；直接拼组合只是在收益、回撤和 2026 适应性之间折中。

下一步不继续做已有曲线静态调权，也不做看 2026 后的动态择时。真正需要的是新的可交易右尾候选土壤，或者能产生低相关收益的独立 sleeve；否则账户层再组合也只是把 5-6x 的上限重新排列。

## 92. oversold_repair_soil_v1

### 92.1 假设

Exp91 说明已有强势/path 曲线高度共享同一收益土壤，账户层静态组合不能打开上限。一个更正交的候选方向是短期超跌修复：它理论上不依赖追逐强势右尾，可能在 2026 这种快兑现、强势股脆弱的环境中形成低相关收益 sleeve。

本实验先不训练模型、不进正式账户，只做周频 due5 标签层土壤验证。核心问题是：全市场可交易股票中，“中期趋势仍在 + 短期回踩 + 不脆弱”的候选，是否能在 2022-2025 和 2026 同时保持足够厚的 5 日收益。

### 92.2 产物和口径

诊断脚本：

```text
/tmp/quantx-research/oversold-repair-soil-v1/analyze_oversold_repair_soil.py
sha256:37cebf2d10827f0ccca3623eec778ebdc2c0e53d6547851880a176d36234a601
```

开发期诊断：

```text
/tmp/quantx-research/oversold-repair-soil-v1/oversold_repair_soil_dev_2022_2025_diagnostic.json
sha256:a71bff39630b45dcc9fba8dc233a08eb290f1a9937f0b7a534150b23c934a232
```

2026 forward 诊断：

```text
/tmp/quantx-research/oversold-repair-soil-v1/oversold_repair_soil_val63_2026_diagnostic.json
sha256:63aaf8696647b1d9bd6d2d1ce893c0a0bcf1bceb3ad7aff9fb4762a3ba7669c0
```

口径：`all_mainboard`、T 日信号、T+1 open 入场、T+6 open 出场、5 日 raw return 和相对全市场均值的 excess return；只取 due5 周频 session；加入粗略 `gap25` 过滤，避免明显跳变污染标签层。候选分数只用价格、成交量和波动率构造，包括纯短反、深跌修复、趋势回踩、低波趋势回踩、放量确认修复、缩量安静修复、恐慌反弹和受控回撤修复。

### 92.3 结果

Top20 主要结果：

| 变体 | 2022-2025 mean label5 | 2022-2025 raw5 | 开发期逐年 label5 | 2026 mean label5 | 2026 raw5 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `pure_deep_repair` | `+0.001745` | `+0.004331` | `+0.000195/+0.000692/+0.001102/+0.005108` | `-0.008290` | `-0.010849` | 开发期最强，但 2026 反向 |
| `controlled_drawdown_repair` | `+0.001387` | `+0.003974` | `+0.003768/+0.000700/+0.000989/+0.000022` | `-0.000496` | `-0.003056` | 开发期很薄，2026 不成立 |
| `low_vol_trend_pullback` | `+0.000472` | `+0.003058` | `+0.001698/-0.000953/+0.000556/+0.000561` | `+0.003533` | `+0.000973` | 两端同向但太薄，且 2023 负 |
| `trend_pullback_repair` | `-0.004577` | `-0.001990` | `-0.003735/-0.004991/-0.008108/-0.001350` | `+0.011299` | `+0.008740` | 2026 最强，但开发期系统性反向 |
| `near_high_pullback` | `-0.002194` | `+0.000393` | `-0.004907/-0.004837/-0.001683/+0.002803` | `+0.006797` | `+0.004238` | 2026 有效，开发期不穿越 |
| `rev3` | `-0.005121` | `-0.002535` | 全期偏负 | `-0.007231` | `-0.009790` | 纯短反明确失效 |
| `rev1` | `-0.008368` | `-0.005781` | 全期偏负 | `-0.007961` | `-0.010520` | 纯隔夜/短反明确失效 |

### 92.4 反事实分析

第一反事实：如果超跌修复是鲁棒低相关 sleeve，开发期和 2026 应有同向正收益。实际开发期最好的 `pure_deep_repair` 在 2026 转为 `-0.829%` 周频超额；2026 最强的 `trend_pullback_repair` 在开发期为 `-0.458%`。

第二反事实：如果纯反转能作为自然收益土壤，`rev1/rev3/rev5` 至少应不显著为负。实际 `rev1` 和 `rev3` 在开发期与 2026 都为负，说明 A 股当前数据中短期跌幅本身不是可买修复信号。

第三反事实：如果“趋势仍在 + 回踩”能解决纯反转问题，它应比深跌修复更稳。实际该类在 2026 明显有效，但开发期 2022-2025 系统性为负，说明它更像 2026 特定风格，而不是可长期复用的主 alpha。

第四反事实：如果该方向值得扩成 ML，裸分数至少应给出足够厚的训练土壤。实际两端都同向的 `low_vol_trend_pullback` 只有 `+0.047%` 开发期周频超额和 `+0.353%` 2026 超额，远低于进入正式账户前的收益厚度要求。

### 92.5 判定

`rejected`。

超跌修复不是当前要找的鲁棒高收益方向。它在 2026 的确有局部有效形态，但与开发期有效形态相互冲突，扩成学习器很容易变成风格择时幻觉。下一步不继续调反转窗口、缩量/放量权重或回踩深度；应转向能从正式成交约束出发生成新候选池的方向，而不是裸风格因子。

## 93. formal_exit_lifecycle_policy_v1

> 2026-07-14 审计更新：本节原始结果保留为发现过程记录，但不再作为 retained candidate。复核发现原脚本在 OPEN 退出判断前调用了当日收盘 `update_daily_balance(date)`，导致止损、盈利续持和 trailing stop 使用了当前交易日收盘后才知道的信息。严格改为“OPEN 退出只使用上一交易日收盘后状态，当前交易日收盘再更新净值”后，head-union h6 `winner_extend max7` 降至 2021-2025 `8.07x`、2026 `+10.85%`，不满足目标。因此本实验最终判定改为 `rejected_after_lookahead_audit`。

### 93.1 假设

Exp86-92 基本排除了“在同一 Top20 内微调排序”“裸因子土壤”“已有曲线静态拼接”几条路。一个新的反事实是：当前 formal account 不是缺少右尾候选，而是固定 5 日退出把右尾生命周期截断，同时在弱路径里又被动等到第 5 日才止损。

本实验不改变候选排序、不降低尾部权重、不加入 2026 后验择时，只在正式 overlap account 副本中加入 sleeve 级别的因果退出规则：先按原规则买入，持有至少 2 个交易日；若到第 5 日仍盈利则允许续持，直到最大持有天数或回撤触发；若第 5 日不盈利则退出；若 sleeve 级别亏损达到阈值则止损。核心验证对象是“自然 ML 候选源 + 持仓生命周期控制”是否能在正式成交约束下同时满足五年收益、逐年正、平均持仓数、一周附近持有和 2026 forward。

### 93.2 产物和口径

诊断脚本：

```text
/tmp/quantx-research/formal-exit-policy-v1/analyze_formal_exit_policy.py
sha256:07ad1de4c1fec409c0a1b2a2dbaff3e5c84e10963a705df851dd1be55b678b62
```

主候选输入 store：

```text
/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_dev_2021_2025_top15_predictions.json
/tmp/quantx-research/head-union-ensemble-v1/head_union_7model_head6_base5d_anchor_val63_2026_top15_predictions.json
```

正式账户口径：`all_mainboard`、T 日信号、T+1 open 入场、formal `AStockExchange + Executor`、正常交易成本、`gap25+hist120`、Top15 pool、buy5、初始 `hold_days=5`。所有中间脚本和运行产物均在 `/tmp`，未合入正式代码。

核心退出规则 `winner_extend max7`：

1. 最少持有 2 个交易日，避免 T+1 噪声立刻出场。
2. 第 5 日若 sleeve 总收益不为正，退出。
3. 第 5 日若 sleeve 总收益为正，允许续持到最多 7 个交易日。
4. 持有期内 sleeve 总收益低于 `-5%` 触发止损。
5. 续持后从 sleeve 净值高点回撤超过 `4%` 触发 trailing exit。

### 93.3 原始主结果，已被时序审计降级

| 候选源/退出 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 逐年收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| head-union h6 fixed hold5 | 2021-2025 | 9.12x | `-31.46%` | 14.33 | 7.66 | `+61.05%、+28.73%、+30.61%、+74.31%、+95.98%` |
| head-union h6 fixed hold5 | 2026 val63 | `+12.80%` | `-17.37%` | 12.65 | 7.66 | `+12.80%` |
| head-union h6 `winner_extend max7` | 2021-2025 | 25.88x | `-20.07%` | 14.55 | 8.78 | `+100.67%、+43.68%、+61.01%、+126.08%、+146.62%` |
| head-union h6 `winner_extend max7` | 2026 val63 | `+35.64%` | `-10.31%` | 12.86 | 9.01 | `+35.64%` |
| head-union h6 `winner_extend max7` 双倍成本 | 2021-2025 | 14.47x | `-20.70%` | 14.50 | 8.70 | `+76.85%、+29.85%、+39.61%、+106.78%、+118.55%` |
| head-union h6 `winner_extend max7` 双倍成本 | 2026 val63 | `+27.09%` | `-10.72%` | 12.63 | 8.76 | `+27.09%` |

主候选结果产物：

```text
/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_dev_2021_2025.json
sha256:95de65591d72014e55907920bc67d85faf6330acedfb455045416608d48667e1

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_val63_2026.json
sha256:676576aaf00cb8491a81dc24b7cf6c03c86d4b7b60732ee9e652e83861de501f

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_dev_2021_2025_doublecost.json
sha256:aea6cc5d029893f596925c2384d761dfa39cd7bb80ae9a1f75aca016ed043c15

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_val63_2026_doublecost.json
sha256:c2514bb49c237f719f60bac226e365aaba1a01c24ef80f0cc84096633e66238a
```

严格无未来函数复核产物：

```text
/tmp/quantx-research/formal-exit-policy-nolookahead-v1/analyze_formal_exit_policy_nolookahead.py
sha256:d084ee879865616b9aa9b3cc0ef3c81458dbd2fb0176f33940897dda9f841f03

/tmp/quantx-research/formal-exit-policy-nolookahead-v1/formal_exit_headunion_h6_winnerextend_max7_dev_2021_2025_nolookahead.json
sha256:1e8aaf102cbb519d754a6cbee714044fa55834ea4fa8cc16bfb94b02716b8554

/tmp/quantx-research/formal-exit-policy-nolookahead-v1/formal_exit_headunion_h6_winnerextend_max7_val63_2026_nolookahead.json
sha256:fe9808c1ab9e71646530905ae23b9253e45b4a05a94ab93487ce711435b32611
```

| 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 逐年收益 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 原始脚本 `winner_extend max7` | 2021-2025 | 25.88x | `-20.07%` | 14.55 | 8.78 | `+100.67%、+43.68%、+61.01%、+126.08%、+146.62%` |
| 原始脚本 `winner_extend max7` | 2026 val63 | `+35.64%` | `-10.31%` | 12.86 | 9.01 | `+35.64%` |
| 无未来函数复核 `winner_extend max7` | 2021-2025 | 8.07x | `-28.07%` | 14.62 | 8.93 | `+61.27%、+15.22%、+37.62%、+47.16%、+115.58%` |
| 无未来函数复核 `winner_extend max7` | 2026 val63 | `+10.85%` | `-17.42%` | 12.69 | 8.65 | `+10.85%` |

关键差异是退出决策的可见信息边界。原始脚本在当天开盘卖出前先把 sleeve 账户按当天收盘价更新，因此会在同一天 OPEN 之前知道当天 CLOSE 后的 sleeve 盈亏和高点回撤。无未来函数版本把卖出判断放在当日收盘更新之前，只允许使用上一交易日收盘后已知的 `sleeve.account.get_total_value()` 和 `sleeve.peak_value`。修正后收益量级回到 5-10x 区间，2026 也只剩约 `+10.85%`，说明原强结果不能用于合代码。

### 93.4 参数邻域和迁移复核

| 复核 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均日历持有 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| head-union h6 `max7 trail3` | 2021-2025 | 26.99x | `-20.06%` | 14.55 | 8.76 | 与 trail4 几乎一致 |
| head-union h6 `max7 trail3` | 2026 val63 | `+36.55%` | `-10.31%` | 12.85 | 9.00 | 2026 不敏感 |
| head-union h6 `max10 trail4` | 2021-2025 | 43.83x | `-20.73%` | 15.47 | 10.72 | 收益更高，但平均持有偏离“一周附近” |
| head-union h6 `max10 trail4` | 2026 val63 | `+31.48%` | `-9.90%` | 12.84 | 11.38 | 仍通过 2026，但持有偏长 |
| weekly persistence `prev_top20_bonus max7` | 2021-2025 | 27.30x | `-23.06%` | 15.50 | 8.87 | 生命周期规则可迁移到另一自然候选源 |
| weekly persistence `prev_top20_bonus max7` | 2026 val63 | `+15.21%` | `-15.70%` | 12.55 | 8.38 | 2026 未优于 path/head-union |

参数邻域和迁移产物：

```text
/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_trail3_dev_2021_2025.json
sha256:3626819df2ec64604593b8795a888cda8d0da9caee4ae0ed90c684f95eb96914

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max7_trail3_val63_2026.json
sha256:db044d240a8a77c068beb94d27bb886a4e8a67a8b04bb6d5fbca453e88ed3995

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max10_dev_2021_2025.json
sha256:32e2fe5340ea4f217602872c7b4d5285f23986ad707e99ec5e104d684241a7ca

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_headunion_h6_winnerextend_max10_val63_2026.json
sha256:e283777fcaecefa59c173b073ef02ec21179022356d7cafad96d3d90fa2b6bd7

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_prevtop20_winnerextend_max7_dev_2021_2025.json
sha256:af295b20dc644ca1a8c1f6d62987ba165733a1812ec711eb1e3b073c56e4152f

/tmp/quantx-research/formal-exit-policy-v1/formal_exit_prevtop20_winnerextend_max7_val63_2026.json
sha256:93d21a523052802bcb90a166d92a4750ada15f577f0d77e51374f04c58fe21f8
```

### 93.5 原始反事实分析的审计修正

第一反事实：如果收益只是 head-union 候选本身带来的，固定退出也应达到相近量级。原始脚本里 fixed 只有 9.12x、2026 `+12.80%`，动态生命周期到 25.88x、2026 `+35.64%`；但无未来函数复核后动态生命周期只有 8.07x、2026 `+10.85%`，并未稳定超过固定退出。原先“主要增量来自退出生命周期”的判断被审计推翻。

第二反事实：参数邻域稳定性仍有解释价值，但因为共同依赖同一个未来信息边界错误，不能再作为晋级证据。

第三反事实：无未来函数版本 2021-2025 仍逐年为正，说明 head-union 候选源本身有一定稳定性；但最终倍数仅 8.07x，低于目标收益量级。

第四反事实：如果该规则只适配某一个 store，迁移到 weekly persistence 应失败。实际 weekly persistence 五年也到 27.30x，但 2026 仅 `+15.21%`，说明生命周期控制本身有真实收益弹性，但候选源仍决定 2026 适应性；当前更优主线是 head-union h6。

第五反事实：无未来函数复核后，动态生命周期没有形成“收益和回撤同时显著改善”的强证据，不能作为当前主线继续合代码。

### 93.6 判定

`rejected_after_lookahead_audit`。

本实验最初看起来是第一条接近目标的正式成交候选，但时序审计显示 OPEN 退出判断前使用了当天 CLOSE 状态，属于不可用信息。修正后 2021-2025 为 8.07x、2026 为 `+10.85%`，只保留“head-union 候选源和持仓生命周期值得解释”的诊断价值，不再进入合代码。

后续研究必须把这条教训前置：凡是涉及退出、止损、trailing、持有期延长的实验，OPEN 交易只能使用上一交易日收盘后已知状态；如果要使用当日盘中/收盘信息，则成交时点必须相应后移，不能再按当天 OPEN 成交计入收益。

## 94. deep_learning_sequence_alpha_search_v1

### 94.1 假设

Exp40 之后最强的可复用底座仍是 `path_sequence`：它把过去多日横截面路径、量能、近高、行业/概念和市场宽度等特征做成五分位排序目标，账户层 2022-2025 约 8.77x、2026 `+19.18%`。一个自然反事实是：LightGBM 只看手工聚合特征，可能没有学到路径序列形态；若用紧凑 GRU 或多任务 path-world encoder 学习过去一段时间的路径表示，应该能在 Top20 标签层超过 path_sequence，并至少在 2026 不弱于它。

本轮只做预测层和强基线门控，不进入正式账户复核。原因是已有 formal account 表明，同一标签层的小增益很难把 5-9x 上限推到几十倍；DL 必须先在多尺度指标上显著超过 path_sequence，才值得消耗正式账户回放。

### 94.2 产物和口径

全部产物均在 `/tmp/quantx-research/deep-learning-alpha-search-v1/`：

```text
analyze_sequence_soil_v1.py
sha256:813b719eaa4df42c733d45b5a4836ac6f4ef4f184f0191ed47e1023d00061fe3

train_compact_sequence_model_v1.py
sha256:66d89ebe4ca253d79cc1708a38cb3ecc9fcad47265cf5b4ced6ee5a90177b603

train_multitask_path_world_encoder_v1.py
sha256:2b244bc6168755b2893d194f6769dfe69c5a4ee767fcefb1699840b21f2748e0

evaluate_multiscale_metrics_v1.py
sha256:f37b3ec9061debaf209c1c80c469aefa053e26c86a66a0cb3559dab5cc230282
```

主要输出：

```text
formal_tradeable_sequence_soil_v1_summary.json
sha256:f30923c7c970d6e6d651443a5012fede7ebf1ba2993b7101b4e43dcf1eba13b4

compact_sequence_gru_dev_2021_2025.json
sha256:32ac47a30d3b934750efefeba553909b471a7d19b06011e66096e07f9b404596

compact_sequence_gru_val63_2026.json
sha256:2a1b5a8685993a1a4b5b9a44063abb0c1e2352eb67621a970010abfba255e53f

multiscale_metrics_compact_sequence_gru_v1.json
sha256:e516e6516b5a3c9ad2081abe19107612c0025cfafa19dbf29010402a36b6d0d9

multitask_path_world_dev_2021_2025.json
sha256:b1ccba2bd571b7c0d82eed45dddb21604dac08564693e8303adf02bb49f956c5
```

强基线：

| 基线 | 区间 | 账户收益/标签 |
| --- | --- | ---: |
| path_sequence formal rebalance5 | 2022-2025 | 8.77x，回撤 `-32.62%`，平均持仓 19.40，平均持有 9.36 天 |
| path_sequence formal rebalance5 | 2026 val63 | `+19.18%`，回撤 `-16.70%`，平均持仓 19.48，平均持有 9.94 天 |
| path_sequence due5 Top20 label | 2021-2025 | `+0.014170` |
| path_sequence due5 Top20 label | 2026 val63 | `+0.012520` |

### 94.3 结果

compact GRU 的 Top20 标签：

| 变体 | 开发期 label5 | 开发期相对 base | 2026 label5 | 2026 相对 base | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| base | `+0.009717` | 0 | `+0.005062` | 0 | 当前 DL 面板的内部基线 |
| `gru_topq` | `+0.010125` | `+0.000409` | `+0.002659` | `-0.002404` | 开发期小增，2026 明显反向 |
| `blend_topq_25` | `+0.010660` | `+0.000944` | `+0.003485` | `-0.001577` | 开发期最佳，但仍未达门槛 |
| `blend_spread_25` | `+0.010510` | `+0.000793` | `+0.002333` | `-0.002729` | 两端不稳 |
| `gru_spread` | `+0.008584` | `-0.001133` | `-0.000780` | `-0.005842` | 明确失败 |

多任务 path-world encoder 只完成开发期诊断，最好 `blend_q5_25` 为 `+0.011372`，相对 base `+0.001829`；但 2025 只有 `+0.004383`，且缺 2026 前向。结合 compact GRU 的 2026 反向结果，不直接进入正式账户。

多尺度门控结论：所有 compact GRU 变体均为 `reject_before_formal_account`。失败原因集中在：开发期增量不足 15bp、2026 低于 base、2026/开发期保留率不足、低于 path_sequence 2026 强基线。

### 94.4 反事实分析

第一反事实：如果序列模型学到了缺失的路径形态，纯 `gru_topq` 应显著超过手工 path_sequence。实际开发期 `gru_topq` 只有 `+0.010125`，远低于 path_sequence 的 `+0.014170`，说明当前紧凑序列表示没有超过手工路径特征。

第二反事实：如果只是 DL 分数噪声大但方向有用，弱融合 `blend_topq_25` 应该在 2026 至少不低于 base。实际 2026 从 base `+0.005062` 降到 `+0.003485`，说明开发期增量不能迁移。

第三反事实：如果要解决收益上限，预测层必须给出远大于已有 5-9x 正式账户基线的厚度。实际开发期最好的 compact GRU 增量不足 `+0.001`，不可能自然把账户收益推到几十倍。

第四反事实：如果多任务 world encoder 是正确方向，它应该先完成 2026 forward 并过强基线门控。当前只见开发期小增量，而且 2025 较弱；在 compact GRU 前向失败后，不应该直接投入正式账户复核。

### 94.5 判定

`rejected_before_formal_account`。

深度序列模型当前没有打开收益上限。它更像对已有手工路径特征的弱拟合，而不是新的可交易右尾土壤。下一轮不继续堆 GRU/encoder 架构，也不围绕 2026 后验调网络；应转向更高层的市场结构假设：入场时可见的横截面风险、行业/概念轮动强度和强势股可交易性如何共同决定“哪类股票在一周内自然产生右尾”。

## 95. causal_factor_blend_v1

### 95.1 假设

Exp32/33 的裸因子扫描和因子路由显示一个很强的反事实：单个因子族很弱，但同日 oracle 选择因子专家的 Top20 label 很高。A 股确实有风格切换和概念/行业轮动，只是离散地在低波、动量、回踩、概念爆发之间 all-in 选择一个专家太粗，且近期因子表现会追错风格。

本轮测试一个更保守的因果版本：不再单选专家，而是用截至当前信号日前已经完成的因子表现，对多个横截面因子专家做非负连续权重组合，并保留长期先验、近期表现和市场状态收缩。若“风格上限”能被因果识别，连续组合应至少超过固定低波因子，并在 2026 适应到回踩/动量环境。

关键时序约束：每个 due5 信号的未来 5 日标签有 `exit_date`，在线权重只允许使用 `exit_date <= current signal session` 的历史因子表现。这样避免上一期 5 日收益尚未完成就被下一期使用。

### 95.2 产物和口径

```text
/tmp/quantx-research/causal-factor-blend-v1/analyze_causal_factor_blend.py
sha256:32365de659549399d9f5454f9436c664b8a91da4eb53b0a12e00bda199f85619

/tmp/quantx-research/causal-factor-blend-v1/causal_factor_blend_dev_2021_2025_diagnostic.json
sha256:95f9eddfe387fabc7fc4ba22922716bf67079266e2eeeb3d1f171a209521f9a6

/tmp/quantx-research/causal-factor-blend-v1/causal_factor_blend_val63_2026_diagnostic.json
sha256:8a92eee687643e4fb8c74180d64fa87db09117fac57d47ba88ccba41105081ad
```

因子专家包括：`low_vol20`、`low_vol_trend`、`near_high_quality`、`pullback_from_high20`、`trend_pullback`、`mom60`、`high_vol_leader`、`medium_structure`、`concept_burst`、`group_relative_leader`。评估只在 due5 周频信号上做标签层诊断，标签为 T+1 open 入场到 T+6 open 退出的 5 日收益减同日全市场均值。行业/概念 membership 仍使用 2026-06-25 静态快照，因此带行业/概念的解释仅作探索性证据。

测试的连续组合：

| 组合 | 说明 |
| --- | --- |
| `blend_uniform` | 所有专家等权 |
| `blend_long_relu` | 历史均值为正的专家加权，并给每个专家最低 3% 权重 |
| `blend_shrunk_recent6` | `70%` 长期均值 + `30%` 最近 6 个已完成信号表现 |
| `blend_shrunk_state` | `70%` 长期均值 + `30%` 同市场状态历史均值 |
| `blend_hedge_long` | 长期均值 softmax，单专家上限 35%，最低 3% |
| `blend_hedge_recent6` | 长期和近期均值各半的 softmax 权重 |

### 95.3 结果

Top20 结果：

| 方案 | 2021-2025 label5 | 开发期逐年 | 2026 label5 | 判读 |
| --- | ---: | --- | ---: | --- |
| `oracle_expert` | `+0.036613` | `+0.0361/+0.0307/+0.0397/+0.0401` | `+0.063513` | 事后上限很高，但不可交易/不可因果 |
| 固定 `low_vol20` | `+0.001177` | `+0.0034/+0.0004/+0.0063/-0.0054` | `-0.004749` | 开发期最好因果单因子，但 2026 反向 |
| 固定 `pullback_from_high20` | `-0.005923` | `-0.0083/-0.0048/-0.0073/-0.0033` | `+0.017992` | 2026 最强，但开发期系统性反向 |
| 固定 `mom60` | `-0.013576` | `-0.0239/-0.0118/-0.0156/-0.0030` | `+0.011561` | 2026 有效，开发期失败 |
| `blend_long_relu` | `+0.000704` | `-0.0003/+0.0021/+0.0055/-0.0045` | `-0.001351` | 低于固定低波，2026 仍负 |
| `blend_shrunk_recent6` | `+0.000019` | `-0.0004/+0.0008/+0.0044/-0.0048` | `+0.005281` | 2026 有适应但太弱，开发期几乎无 alpha |
| `blend_shrunk_state` | `-0.001233` | `-0.0054/+0.0012/+0.0036/-0.0043` | `-0.000514` | 状态收缩无效 |
| `blend_uniform` | `-0.008099` | 全部较弱 | `+0.001744` | 混合会稀释因子方向 |

### 95.4 反事实分析

第一反事实：如果风格 oracle 只是因为单因子太极端，连续权重应能保留部分上限。实际最好的开发期连续组合只有 `+0.000704`，低于固定 `low_vol20` 的 `+0.001177`，说明把风格混合在一起会互相抵消。

第二反事实：如果近期因子表现有因果动量，`blend_shrunk_recent6` 应在开发期改善。实际开发期仅 `+0.000019`，几乎为零；2026 虽然变为 `+0.005281`，但远低于固定 `pullback_from_high20` 的 `+0.017992` 和 `mom60` 的 `+0.011561`。近期表现能在 2026 捕到一点风格，但不是穿越样本的稳定规律。

第三反事实：如果市场状态能解释风格切换，`blend_shrunk_state` 应强于长期组合。实际开发期为负、2026 也接近零，说明当前趋势/宽度/离散度这些粗状态无法决定哪个因子该生效。

第四反事实：如果行业/概念轮动可以直接用裸因子承接，`concept_burst` 或 `group_relative_leader` 应有正厚度。开发期它们仍明显偏弱，说明行业/概念信息更可能要和股票级位置、可交易性和风险状态交互，而不能单独作为买入排序主轴。

### 95.5 判定

`rejected`。

本轮证明“因子风格切换上限很高”这个观察是真的，但连续因果加权也无法识别它。单专家路由、近期因子动量、长期先验、市场状态收缩都没有把 oracle 上限转成稳定收益。下一步不再做因子族择时，而应转成股票级共同条件学习：直接在每只股票层面学习“市场状态、横截面分化、行业/概念热度、个股中期位置、可交易性”如何共同决定未来一周可交易右尾，而不是先选一个风格专家。

## 96. online_factor_meta_v1

### 96.1 假设

Exp95 说明“选因子专家”或“给因子专家做连续权重”都不能稳定识别风格切换。但这不等于横截面因子没有价值。更细的反事实是：风格不是在因子族层面切换，而是在股票级发生交互，例如同样是 `mom60`，在高分化/弱宽度/概念热度不同的市场中含义不同。

本轮不训练年度冻结 LightGBM，而是做一个更因果、更快适应的在线股票级 meta-ranker：每个信号日单独用已经完成的历史样本拟合 ridge 回归，目标为候选池内未来 5 日超额收益，特征为 base ML 排名、多日 RPS 路径、量能、近高/回踩、行业/概念热度、相对组强弱和市场风险状态。若短期横截面因子系数可在线学习，它应该在开发期和 2026 同时超过 base ML Top20。

时序约束：每个训练样本都有 `exit_date`，信号日 `T` 的 ridge 只允许使用 `exit_date <= T` 的样本。这样不会使用尚未完成的未来 5 日结果。

### 96.2 产物和口径

```text
/tmp/quantx-research/online-factor-meta-v1/analyze_online_factor_meta.py
sha256:445bd1e473c3fb1250721c85d0b307be229a95ddcee125bfc1bedf109462ddc0

/tmp/quantx-research/online-factor-meta-v1/online_factor_meta_dev_2021_2025_diagnostic.json
sha256:7d3e43dc347be432338dfacaf23108cecfd605bb97273a868dc985827a0cdc4e

/tmp/quantx-research/online-factor-meta-v1/online_factor_meta_val63_2026_diagnostic.json
sha256:00adc9a58afd003dd1eb8f910d5b87f594d7ae319f146405ac010570efe65f01
```

复用第 40 轮 `path_sequence_ranker_v1` 的候选池和特征构造，但模型从年度 LightGBM 改成在线 ridge：

| 变体 | 说明 |
| --- | --- |
| `base` | base 5d ML 候选池原始排序 |
| `ridge_expand` | 使用全部已完成历史样本的 ridge 预测排序 |
| `ridge_roll` | 使用最近 80 个已完成信号日的 ridge 预测排序 |
| `ridge_blend_expand_25` | `75% base + 25% expanding ridge rank` |
| `ridge_blend_roll_25` | `75% base + 25% rolling ridge rank` |

本轮仍为标签层诊断，不进入 formal account。晋级门槛是：开发期 Top20 label 明显超过 base，并且 2026 也超过 base；否则视为同一候选池内的微调失败。

### 96.3 结果

Top20 结果：

| 方案 | 2021-2025 label5 | 相对 base | 2021-2025 逐年 | 2026 label5 | 相对 base | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `base` | `+0.015657` | 0 | `+0.0279/+0.0182/+0.0092/+0.0117/+0.0111` | `+0.012018` | 0 | 强基线 |
| `ridge_expand` | `+0.015676` | `+0.000019` | `+0.0267/+0.0195/+0.0091/+0.0142/+0.0087` | `+0.013070` | `+0.001052` | 2026 小增，但开发期几乎为零 |
| `ridge_blend_expand_25` | `+0.015569` | `-0.000088` | `+0.0277/+0.0185/+0.0091/+0.0120/+0.0104` | `+0.012472` | `+0.000454` | 融合后开发期略降 |
| `ridge_roll` | `+0.014046` | `-0.001612` | `+0.0267/+0.0191/+0.0073/+0.0084/+0.0085` | `+0.010182` | `-0.001836` | 近期滚动系数不稳 |
| `ridge_blend_roll_25` | `+0.015491` | `-0.000166` | 未优于 base | `+0.011643` | `-0.000375` | 弱于 base |

Top quintile 命中和 bottom 暴露也没有改善：开发期 `base` Top20 top-quintile rate 为 `26.17%`、bottom rate 为 `22.59%`；`ridge_expand` top rate 降到 `25.92%`，bottom rate 升到 `22.96%`。2026 `ridge_expand` 的 label 有小增，但 top/bottom 结构基本未改善。

### 96.4 反事实分析

第一反事实：如果短期横截面因子系数可在线学习，`ridge_expand` 应在开发期形成稳定正增量。实际开发期只提升 `+0.000019`，本质等于噪声；并且 2025 从 base `+0.011084` 降到 `+0.008692`。

第二反事实：如果市场风格有近期动量，`ridge_roll` 应更快适应 2026。实际它在开发期和 2026 都低于 base，说明最近 80 个信号日的线性系数追噪声。

第三反事实：如果 ridge 有局部信号但太激进，25% 弱融合应至少不伤开发期。实际 `ridge_blend_expand_25` 开发期略低于 base，说明这不是一个可稳定叠加的小 alpha。

第四反事实：如果 2026 小增代表方向正确，它应该伴随 top-quintile rate 上升或 bottom rate 下降。实际结构变化很小，且开发期没有同步改善，因此更像 2026 局部修补，不足以进入账户层。

### 96.5 判定

`rejected_with_signal`。

在线股票级线性适应没有打开收益上限。它说明 2026 的确有一点可从历史因子系数中迁移的结构，但增量太薄，开发期几乎为零。继续在同一个 base ML/path 候选池内做线性、弱融合或小模型微调，已经接近边际上限。下一步应换候选土壤，而不是继续重排同一批候选：优先研究更靠近 A 股强概念/行业轮动的“组层事件后的可交易承接”，但必须从一开始就以 T+1 可买、非涨停追高、平均持仓大于 5 的组合目标设计标签。

## 97. group_event_basket_v1

### 97.1 假设

Exp90-92 显示，行业/概念热度可以解释局部收益和补位，但独立组层热点 formal account 低于 path baseline。一个仍值得反证的方向是：旧实验可能过度追逐个股涨停/近涨停，导致买入已经拥挤的股票；如果只要求行业/概念组层出现事件，而个股本身不过热，买入组内 follower、质量股或回踩股，可能形成更可交易的一周承接篮子。

本轮因此不再要求个股自己涨停或近涨停，而是用 T 日收盘可见的组层近涨停比例、涨停比例、组 3 日收益、强势比例识别热点组，再用个股短期不过热、温和跟随、回踩和 T+1 开盘 gap 可交易性做篮子筛选。`group_event_low_chase_gap7` 使用 T+1 open gap 作为执行时过滤，只能视为诊断口径；其它变体只用 T 收盘信息。

### 97.2 产物和口径

```text
/tmp/quantx-research/group-event-basket-v1/analyze_group_event_basket.py
sha256:0ac3fff109a612a0ac28e97223dcf8d23871531f1e17208a9fec5e441e90127c

/tmp/quantx-research/group-event-basket-v1/group_event_basket_dev_2021_2025_diagnostic.json
sha256:80038b02615921ade8deea204c15de1c67c63097b1a175fd0b80abc7440c5ee9

/tmp/quantx-research/group-event-basket-v1/group_event_basket_val63_2026_diagnostic.json
sha256:72753cce0497aedbd940cf8d9454db7cd5e5199b7d0c94d29ad1d5576888c708
```

评估口径：全主板，T 日信号，T+1 open 入场，T+6 open 退出，5 日收益减同日全市场均值。样本同时看 `all` 和 `due5`，TopK 为 10/15/20/30。行业/概念 membership 仍使用当前静态快照，因此本轮只作为候选土壤诊断，不可直接晋级。

变体包括：`concept_event_follower`、`industry_event_follower`、`group_event_follower`、`group_event_quality`、`group_event_pullback`、`group_event_low_chase_gap7`。

### 97.3 结果

Top20 主结果：

| 方案 | 2021-2025 label5 | 2021-2025 raw5 | 2021-2025 逐年 label5 | 2026 label5 | 2026 raw5 | 判读 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `all::group_event_pullback` | `+0.001005` | `+0.004323` | `+0.0051/-0.0019/+0.0015/+0.0020/-0.0017` | `+0.002249` | `-0.000820` | 开发期最强 Top20，但厚度极薄 |
| `due5::group_event_low_chase_gap7` | `+0.000734` | `+0.003821` | `+0.0084/-0.0003/-0.0001/-0.0011/-0.0034` | `+0.000709` | `-0.001861` | gap 过滤未形成稳定收益 |
| `due5::group_event_quality` | `+0.000547` | `+0.003635` | 未逐年稳定 | `+0.002275` | `-0.000294` | 仍远低于强基线 |
| `all::group_event_follower` | `-0.000123` | `+0.003090` | 多年偏弱 | `+0.005282` | `+0.002214` | 2026 局部有效，开发期无 alpha |
| `due5::industry_event_follower` | `-0.000955` | `+0.002133` | 多年偏弱 | `+0.003119` | `+0.000550` | 2026 小样本局部强，不能穿越 |

对比强基线：Exp96 同口径 base Top20 开发期为 `+0.015657`、2026 为 `+0.012018`；Exp40 path_sequence due5 Top20 开发期约 `+0.014170`、2026 约 `+0.012520`。Exp97 最强开发期 Top20 只有 `+0.001005`，due5 主口径不足 `+0.001`，不是厚标签；2026 即使局部 all/top10 较强，也没有超过强基线。

### 97.4 反事实分析

第一反事实：如果“组热但个股不过热”是缺失的右尾土壤，开发期 Top20 应明显高于 1% 级别标签，并至少接近 path/base 强基线。实际最好的开发期 Top20 只有 `+0.001005`，不足强基线的十分之一。

第二反事实：如果旧实验失败只是因为追高，`low_chase` 和 T+1 gap 可交易过滤应显著改善。实际 `due5::group_event_low_chase_gap7::top20` 开发期只有 `+0.000734`，2026 只有 `+0.000709`，说明不追高并不能自然产生一周收益。

第三反事实：如果 2026 的行业/概念事件承接是真主线，2026 强项应在开发期至少不为零。实际 `group_event_follower` 和 `industry_event_follower` 在开发期 Top20 为负或接近零，更像 2026 局部风格，而不是可穿越规律。

第四反事实：如果组层事件能作为独立主引擎，它不应只在 raw5 上正、在 excess label 上薄。实际多数变体 raw5 尚可，但 excess label 很弱，说明它更多暴露市场/行业 beta，而不是稳定横截面 alpha。

### 97.5 判定

`rejected`。

组层事件后的 follower/quality/pullback 篮子没有形成厚标签。它进一步确认：当前静态行业/概念信息可以解释风险、做局部补位或帮助归因，但靠组热度、事件扩散和不过热条件独立生成 Top20，不能接近 path/base 强基线，更不可能把正式账户收益推向几十倍。下一步不继续手工调行业/概念事件公式，转向正式可成交约束下的低相关右尾 sleeve 或更直接的可交易右尾候选源。

## 98. right_tail_event_top10_formal_replay_v1

### 98.1 假设

Exp48 的历史右尾事件模型在 Top20 口径没有超过 path/base 强基线，因此当时没有进账户层。但 Exp48 和后续 event soil scan 留下一个未完全验证的反事实：Top10 标签层比 Top20 厚，日频 overlap 下即使每个 sleeve 只买 5 只，平均持仓仍可能大于 5。若右尾事件模型真正捕捉了可交易弹性，那么 `event_base_35 pool100 top10` 进入 formal overlap account 后，应至少显著超过 path Top20 formal baseline，并保持 2026 不塌。

本轮只补这个正式账户裁判，不重新调 winner/loser 分类器。这样可以避免把第 48 轮已经拒绝的 Top20 方向重复调参，也能检验“窄 Top10 + 日频重叠”是否是被漏掉的账户结构。

### 98.2 产物和口径

```text
/tmp/quantx-research/right-tail-event-formal-v1/write_right_tail_event_predictions.py
sha256:39969e2226eb9fca72cf96270a7761d9e9b19222535c56d2bfa16b9b89119223

/tmp/quantx-research/right-tail-event-formal-v1/right_tail_event_event_base35_pool100_top10_dev_2021_2025_predictions.json
sha256:3fa2194308b6453fecf6bc15248830a621af0b1a79aadcce7d7eef57331a0879

/tmp/quantx-research/right-tail-event-formal-v1/right_tail_event_event_base35_pool100_top10_val63_2026_predictions.json
sha256:55199e564193ef68d67d698b844c9e73808f2cf81fdc9e9c0dd7e2f71a19cd2c

/tmp/quantx-research/right-tail-event-formal-v1/formal_overlap_right_tail_event_base35_pool100_top10_buy5_dev_2022_2025_gap25_hist120_dynamic_diagnostic.json
sha256:a245a0c9c29300263fbfc77b354de252b7958643ef23e9e094e66f74fc9993cb

/tmp/quantx-research/right-tail-event-formal-v1/formal_overlap_right_tail_event_base35_pool100_top10_buy5_val63_2026_gap25_hist120_dynamic_diagnostic.json
sha256:d81dbba78e80f3fa53367bc29070406e6403693c47ca0e3c040224ba3feec08c
```

预测层复用 Exp48 的无未来函数设置：开发期 walk-forward 每年只用更早年份训练；2026 forward 使用 2021-2025 训练，不使用 2026 标签。账户层复用 Exp67/84 之后的 formal overlap account：T 日信号、T+1 open 入场、Top10 池补位买 5、hold5、`max_open_gap=0.25`、`min_history_days=120`、正式 `AStockExchange + Executor + TransactionCost`。

对照 baseline：同一 formal overlap 裁判下，path Top20 buy5 为 2022-2025 `5.48x`、最大回撤 `-37.35%`、2026 `+19.47%`、最大回撤 `-17.42%`。

### 98.3 结果

| 方案 | 区间 | 最终倍数/收益 | 最大回撤 | 平均持仓 | 平均日历持有 | 年度结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| right-tail event Top10 buy5 | 2022-2025 | `3.23x` / `+223.05%` | `-40.03%` | 17.86 | 7.65 天 | `+35.18%、+36.18%、-6.38%、+85.54%` |
| path Top20 buy5 baseline | 2022-2025 | `5.48x` / `+447.55%` | `-37.35%` | 18.00 | 7.69 天 | `+54.82%、+26.74%、+30.68%、+110.47%` |
| right-tail event Top10 buy5 | 2026 val63 | `1.13x` / `+12.63%` | `-19.31%` | 14.98 | 8.22 天 | `+12.63%` |
| path Top20 buy5 baseline | 2026 val63 | `1.19x` / `+19.47%` | `-17.42%` | 14.52 | 8.79 天 | `+19.47%` |

右尾事件 Top10 确实满足平均持仓大于 5 和一周左右持有期，但它没有满足收益量级，也没有逐年为正；2024 为负，2026 也低于 path baseline。

### 98.4 反事实分析

第一反事实：如果 Exp48 的问题只是 Top20 太分散，Top10 formal 应显著强于 path Top20。实际 Top10 formal 开发期只有 `3.23x`，低于 path 的 `5.48x`，说明标签层 Top10 厚度没有转成正式成交收益。

第二反事实：如果右尾事件模型捕捉的是稳定可交易弹性，2024 不应转负。实际 2024 为 `-6.38%`，同时最大回撤扩大到 `-40.03%`，说明它仍在高弹性环境中同步买到左尾失败样本。

第三反事实：如果 2026 局部标签增量是真前向优势，formal 2026 应超过 path baseline。实际 2026 为 `+12.63%`，低于 path `+19.47%`，且回撤更大。这说明 2026 标签层的事件增量被正式成交、拒单、持仓路径和高弹性左尾吞掉。

第四反事实：如果账户结构是核心缺口，日频 overlap Top10 应通过平均持仓约束并放大收益。实际平均持仓确实达到 17.86/14.98，但收益仍低于 baseline，说明“缩窄 TopK + overlap”不是缺失的鲁棒收益来源。

### 98.5 判定

`rejected_after_formal_replay`。

历史大涨股/右尾事件 ML 归因仍有解释价值，但不能作为当前主策略方向。它在标签层能捕捉一部分右尾概率，进入正式账户后却表现为高弹性但不稳定的 sleeve：2024 负收益、2026 不如 path、回撤更大。下一步不继续调 `p_winner`、`event_edge` 或 Top10/Top15 阈值；应换到更直接的结构性收益来源，例如真正低相关的事件类型、跨行业资金流扩散、或者能在 formal 可交易约束下先证明右尾厚度的候选源。

## 99. intraday_overnight_path_v1

### 99.1 假设

Exp93-98 连续说明，在同一 base/path 候选池内重排、右尾事件概率、组热度和因果风格权重都难以把 formal 收益从 3-6x 推到几十倍。一个更换信息源的反事实是：A 股短周期强弱可能不只体现在收盘价横截面，而体现在 T 日以前的日内资金推动和隔夜情绪拥挤之间的分解。例如持续日内买盘但隔夜不过热，或者隔夜拥挤后日内不再追涨，可能对应不同的未来一周路径。

本轮只做全市场标签层诊断，不进入 formal account。所有主分数只使用 T 日已经完成的 OHLCV；`intraday_accum_calm_gap7` 额外使用 T+1 open gap 作为执行时过滤，因此只能视为诊断口径，不作为纯信号。

### 99.2 产物和口径

```text
/tmp/quantx-research/intraday-overnight-path-v1/analyze_intraday_overnight_path.py
sha256:073b7ad334bb2765dd10dbef43cb39c1865f74b7090660c9a55e06d893284e45

/tmp/quantx-research/intraday-overnight-path-v1/intraday_overnight_path_dev_2021_2025_diagnostic.json
sha256:18513901b235b578cedb08447d3abe553673d195338b720b52aa979a86dec5d7

/tmp/quantx-research/intraday-overnight-path-v1/intraday_overnight_path_val63_2026_diagnostic.json
sha256:829aaac62a9739416fbb99112e0b8bbcfe5fbfeae35d59624b311a5d8764fd7c
```

评估口径：全主板，T 日信号，T+1 open 入场，T+6 open 退出，5 日收益减同日全市场均值。样本同时看 `all` 和 `due5`，TopK 为 10/15/20/30。变体包括：`intraday_accum_calm`、`intraday_accum_calm_gap7`、`smart_money_pullback`、`gap_down_bought_back`、`overnight_crowding_fade`、`close_strength_continuation`。

### 99.3 结果

关键 Top10/Top20 结果：

| 方案 | 开发期 label5 | 开发期逐年 label5 | 2026 label5 | 判读 |
| --- | ---: | --- | ---: | --- |
| `all::overnight_crowding_fade::top20` | `-0.001182` | `+0.0025/-0.0021/-0.0049/+0.0012/-0.0028` | `+0.013739` | 2026 很强，开发期为负 |
| `all::overnight_crowding_fade::top10` | `-0.002297` | `+0.0021/-0.0034/-0.0065/+0.0014/-0.0052` | `+0.014983` | 更集中但开发期更差 |
| `due5::overnight_crowding_fade::top20` | `-0.001634` | `+0.0044/-0.0057/-0.0032/-0.0015/-0.0022` | `+0.016290` | 周频主口径不穿越 |
| `due5::overnight_crowding_fade::top10` | `-0.002247` | `+0.0015/-0.0093/-0.0061/+0.0027/+0.0002` | `+0.023630` | 2026 局部最强，但开发期反向 |
| `due5::intraday_accum_calm_gap7::top20` | `-0.010772` | 五年均负 | `+0.003475` | 日内累积结构明显失败 |
| `due5::intraday_accum_calm_gap7::top10` | `-0.013128` | 五年均负 | `+0.005278` | 不是可用右尾土壤 |

`overnight_crowding_fade` 在 2026 的确形成了厚标签：due5 Top10 label5 为 `+0.023630`，all Top20 为 `+0.013739`。但开发期同口径均为负，且 2022、2023、2025 多数为负。其它日内买盘、回踩、gap down bought back 等变体在开发期更弱，不能作为候选。

### 99.4 反事实分析

第一反事实：如果日内资金推动是稳定主 alpha，`intraday_accum_calm` 或 `intraday_accum_calm_gap7` 应在开发期有正厚度。实际 due5 Top20 为 `-0.010757/-0.010772`，五年逐年均负，说明简单追逐日内累积买盘更像拥挤或反转暴露。

第二反事实：如果 2026 的 `overnight_crowding_fade` 是鲁棒规律，它在 2021-2025 至少不应系统性为负。实际开发期 all Top20 为 `-0.001182`，due5 Top20 为 `-0.001634`，且 2022、2023、2025 多数为负，不能用 2026 后验直接晋级。

第三反事实：如果这个方向值得进 formal account，标签层应先超过 path/base 强基线。实际开发期最好的主变体仍为负，远低于 path_sequence due5 Top20 `+0.014170` 和 base/path formal 前置标签，因此不进入正式账户。

第四反事实：如果 `overnight_crowding_fade` 只是噪声，它不该在 2026 all/due5、Top10/Top20 全部明显为正。实际 2026 一致正，说明这是有解释价值的状态信号：2026 市场在奖励“隔夜拥挤后不追日内”的防拥挤结构。但它是状态翻转证据，不是可穿越主策略。

### 99.5 判定

`rejected_with_regime_signal`。

日内/隔夜分解不是当前可直接推进的主候选。它给出一个重要观察：2026 的收益环境明显偏向防隔夜拥挤，而 2021-2025 该形态不稳定甚至反向。后续可以把“隔夜拥挤惩罚”作为 2026 风险解释或状态变量，但不能单独生成 TopK 策略，也不应继续调 intraday/overnight 手写权重。下一步需要寻找更直接的低相关候选源，或者用更强的状态识别机制解释何时防拥挤、何时追趋势。

## 100. vwap_liquidity_path_v1

### 100.1 假设

Exp99 说明单纯 OHLC 拆分的日内/隔夜路径不能穿越开发期，但它暴露出 2026 对“安静趋势”和“防拥挤”的偏好。进一步的反事实是：OHLC 只能看到最终价格路径，`vwap` 和 `volume` 可能更接近成交分布。如果收盘价相对 VWAP、VWAP 相对开盘价、成交量温和放大或缩量趋势能刻画资金吸收，那么它可能形成一个比纯价格路径更可交易的一周候选土壤。

本轮仍只做标签层诊断，不进入 formal account。全部主分数只使用 T 日已经完成的 `open/high/low/close/vwap/volume`；`vwap_accumulation_gap7` 额外使用 T+1 open gap 作为执行时过滤，只能视为诊断口径。

### 100.2 产物和口径

```text
/tmp/quantx-research/vwap-liquidity-path-v1/analyze_vwap_liquidity_path.py
sha256:ca99f3e7dfda0a8c8794f61bf20582d76df83ef7acdb04d8d4b2f1f004871892

/tmp/quantx-research/vwap-liquidity-path-v1/vwap_liquidity_path_dev_2021_2025_diagnostic.json
sha256:8a7324e57d380e9c053306aec6b729b64e563852ad40cee4a63ebf3cc2aeb7b3

/tmp/quantx-research/vwap-liquidity-path-v1/vwap_liquidity_path_val63_2026_diagnostic.json
sha256:a6e79cf2420de7fe7ece16bde76e07167dd2e4093c9e22945ca4b36d692ffc3a
```

评估口径：全主板，T 日信号，T+1 open 入场，T+6 open 退出，5 日收益减同日全市场均值。样本同时看 `all` 和 `due5`，TopK 为 10/15/20/30。变体包括：`vwap_accumulation`、`vwap_accumulation_gap7`、`vwap_reclaim_pullback`、`volume_absorption_trend`、`quiet_trend_liquidity`、`vwap_breakout_not_chase`。

数据字段审计：`qlib_data_fixed` 全市场稳定字段为 `open/high/low/close/volume/vwap/factor/change`，`amount` 只在少数标的目录出现，因此本轮不使用 `amount` 或 turnover 类字段。

### 100.3 结果

关键结果：

| 方案 | 开发期 label5 | 开发期逐年 label5 | 2026 label5 | 判读 |
| --- | ---: | --- | ---: | --- |
| `all::quiet_trend_liquidity::top20` | `-0.003888` | `-0.0033/-0.0034/-0.0033/-0.0073/-0.0024` | `+0.006502` | 2026 转正但开发期全负 |
| `all::quiet_trend_liquidity::top10` | `-0.005632` | `-0.0040/-0.0064/-0.0041/-0.0096/-0.0042` | `+0.009586` | 更集中但开发期更弱 |
| `due5::quiet_trend_liquidity::top20` | `-0.003785` | `-0.0052/+0.0029/-0.0030/-0.0096/-0.0040` | `+0.007364` | 2026 局部有效，不穿越 |
| `due5::quiet_trend_liquidity::top10` | `-0.005424` | `-0.0058/+0.0012/-0.0028/-0.0088/-0.0107` | `+0.006222` | 周频 Top10 仍不足 |
| `all::vwap_breakout_not_chase::top20` | `-0.005222` | `-0.0012/-0.0046/-0.0039/-0.0088/-0.0077` | `+0.003004` | 不是主土壤 |
| `due5::vwap_breakout_not_chase::top20` | `-0.006630` | 五年均负 | `+0.001475` | 明确失败 |

所有开发期主变体均为负，且多数年份负。2026 的 `quiet_trend_liquidity` 有局部正收益，但 label5 只有 `+0.0062` 到 `+0.0096`，低于 path/base 强基线，且开发期完全不支持。

### 100.4 反事实分析

第一反事实：如果 VWAP 相对收盘价能稳定识别资金吸收，`vwap_accumulation` 或 `volume_absorption_trend` 应在开发期变厚。实际这些变体开发期 Top10/20 均为负，说明收盘高于 VWAP、温和放量或成交吸收公式没有产生稳定一周 alpha。

第二反事实：如果 2026 的安静趋势偏好是可穿越规律，`quiet_trend_liquidity` 在 2021-2025 至少应不系统性为负。实际 all Top20 五年逐年均负，尤其 2024 为 `-0.007292`，开发期均值 `-0.003888`，不能用 2026 后验晋级。

第三反事实：如果 VWAP/volume 比 Exp99 的 OHLC 路径更接近可交易主线，开发期应明显改善。实际 Exp100 开发期甚至比 Exp99 的 `overnight_crowding_fade` 更弱，说明在当前日线数据下，VWAP/volume 代理仍主要解释状态和拥挤，而不是生成右尾。

第四反事实：如果这个方向值得进 formal，标签层应先过 path/base 强基线。实际 2026 最强也只有 `+0.009586`，开发期为负，因此不进入账户层。

### 100.5 判定

`rejected_with_regime_signal`。

VWAP/volume 成交路径没有打开收益上限。它进一步支持 Exp99 的状态观察：2026 偏向安静趋势和防拥挤，但这个结构在开发期多年份反向。后续可把安静趋势/低成交拥挤作为风险解释特征，但不应继续围绕日线手写价格/成交路径调权重。下一步转向外部资料和更不同的数据假设，寻找真正低相关的候选源。

## 101. etf_flow_regime_v1

### 101.1 假设

Exp99/100 共同说明：2026 明显奖励“安静趋势、防拥挤、资金不追高”的状态，但股票自身 OHLCV/VWAP 手写公式在 2021-2025 开发期多数反向。一个更外生的状态源是 ETF 市场：ETF 收益、成交额/成交量、宽度、离散度和集中度可能更接近板块资金流、风险偏好和主题轮动，而不是单只股票自己的价格路径。

本轮不生成新股票候选，也不做正式账户回放，只做状态诊断：用 T 日已完成的 ETF 日线状态解释 path/base 强基线在未来 5 日标签上的表现，并测试一个因果在线 selector 能否在 path 与 base 之间选择。若 ETF 状态能稳定识别何时 path 有效、何时 base 更优，它才值得进入下一轮 regime-aware 候选生成。

### 101.2 产物和口径

```text
/tmp/quantx-research/etf-flow-regime-v1/analyze_etf_flow_regime.py
sha256:46450d5da40088e0818fd127dc5b6c103d776b20c4e698e1cdc5a2aceb2bde55

/tmp/quantx-research/etf-flow-regime-v1/etf_flow_regime_dev_2021_2025_diagnostic.json
sha256:b9188034483725e9363e0eb1c5726d6f5cd625da2dbed1639c15222de03fa40a

/tmp/quantx-research/etf-flow-regime-v1/etf_flow_regime_val63_2026_diagnostic.json
sha256:efe2c4e2b066fce5cb200ab0aa6837dd51f7dc283cdbe3f297fedb77ebf63d95
```

ETF universe 使用本地 `qlib_data_fixed/features` 中 `sh51/sh56/sh58/sz15/sz16` 前缀标的，共 119 只。状态特征包括 ETF 5/20 日均收益、5/20 日宽度、20 日成交额 z-score 均值和 Top10、上涨 ETF 成交流相对下跌 ETF 的 flow momentum、成交集中度、5 日离散度、risk-on score。

评估口径：T 日 ETF 状态只使用 T 日收盘和成交额；股票标签使用 T+1 open 到 T+6 open；候选源为已审计过的 Exp40 path Top20 prediction store 和 base 5d prediction store。在线 selector 的阈值只用当前 session 前、且 T+6 标签已经完成的历史样本，不使用未来标签。

### 101.3 结果

整体标签层只是复现强基线，不产生新股票收益：

| 口径 | 开发期 label5 | 开发期 raw5 | 样本数 | 2026 label5 | 2026 raw5 | 样本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all::path_topq_top20::top20` | `+0.017611` | `+0.025541` | 303 | `+0.015905` | `+0.012836` | 118 |
| `all::base_top20::top20` | `+0.014914` | `+0.022845` | 303 | `+0.012018` | `+0.008950` | 118 |
| `due5::path_topq_top20::top20` | `+0.013342` | `+0.021775` | 61 | `+0.012520` | `+0.009951` | 24 |
| `due5::base_top20::top20` | `+0.007916` | `+0.016349` | 61 | `+0.005062` | `+0.002493` | 24 |

ETF 状态分层有解释力，但不稳定。开发期 all/path Top20 在 ETF 5 日离散度最高桶达到 `+0.03501`，低桶为 `+0.01218`；ETF 20 日收益最高桶为 `+0.03509`，但中间桶只有 `+0.00958`。2026 分层更明显：ETF 20 日收益从低桶 `+0.00160` 到高桶 `+0.02926`，ETF 成交额 z-score 均值从低桶 `-0.00470` 到高桶 `+0.02659`。这说明 ETF 状态能解释 2026 风格，但开发期结构并不单调、也不足以独立择时。

因果在线 selector 结果更关键：开发期最佳 `all::top20::etf_risk_on_score::low_path_else_base` label5 为 `+0.013740`，但它的 path chosen ratio 为 `1.0`，完全退化成总选 path；同一有效样本内 path 本身也是 `+0.013740`，base 为 `+0.010916`。其它 ETF selector 均低于总选 path。2026 最佳 selector 同样退化或近似退化成总选 path：`etf_risk_on_score::high_path_else_base` label5 `+0.015715`，path chosen ratio `1.0`，path 本身 `+0.015715`，base `+0.006418`。

### 101.4 反事实分析

第一反事实：如果 ETF 资金流是可用 regime selector，它应能在开发期用历史已完成标签学习出“何时 path、何时 base”的切换，并超过总选 path。实际最优 selector 直接退化成总选 path，其它 selector 低于 path，说明 ETF 状态没有提供稳定可执行的 path/base 路由增量。

第二反事实：如果 2026 的 ETF 状态分层是长期规律，开发期分桶应单调、跨年份稳定。实际开发期有高桶强信号，但中间桶和低桶并不稳定，且在线阈值只在 2025 有效样本上触发，不能证明可跨周期使用。

第三反事实：如果 ETF 状态是新的收益土壤，它不应只是解释 path/base 的既有收益，而应在不看股票未来标签的情况下显著改善候选选择。实际本轮没有生成新候选，也没有让 path/base 路由变厚，说明 ETF 状态更像风险解释变量，而不是主 alpha 来源。

第四反事实：如果继续调 ETF 阈值能解决问题，当前简单高/低阈值至少应出现一个超过 path 的方向。实际没有，继续围绕 ETF risk-on、成交额 z-score、宽度阈值打磨容易变成后验拟合。

### 101.5 判定

`rejected_with_regime_signal`。

ETF 资金流/风险状态解释了 2026 部分 regime，但没有形成可因果交易的收益增量。它可以保留为后续风险解释或样本分层维度，但不应继续做 ETF 阈值择时，也不应把它作为合代码候选。下一步应回到股票级候选生成本身：寻找独立于 Exp40 path/base 的右尾土壤，尤其是历史大幅上涨前的多日横截面路径、行业/概念扩散和市场风险共振，而不是在已有 path/base 曲线之间做状态路由。

## 102. elastic_smallcap_soil_v1

### 102.1 假设

Exp40 之后的很多实验都在已有 path/base 候选池内做重排、补位或状态路由，边际改善很薄。一个更独立的候选土壤是 A 股常见的弹性结构：低价格、低成交额代理、小市值倾向、高中期路径强度、短期不过热。若该结构是真正的右尾来源，它不应只解释 Exp40 账户暴露，而应能从全市场 due5 直接生成比 Exp40 path Top20 更厚的 5 日标签。

本轮只做全市场 label 层诊断，不进入 formal account。原因是门槛前置：只有 Top20 label 在 2021-2025 walk-forward 和 2026 forward 都稳定超过 Exp40，才值得承担正式成交、拒单和账户路径验证成本。

### 102.2 产物和口径

```text
/tmp/quantx-research/elastic-smallcap-soil-v1/analyze_elastic_smallcap_soil.py
sha256:4af03ba89c1308e5eda2d30c5e968ff5919d27aaf3591692575f0c7b0cb312bb

/tmp/quantx-research/elastic-smallcap-soil-v1/elastic_smallcap_soil_dev_2021_2025_diagnostic.json
sha256:58f9a4ad211e72e884f1ea4e57ab9191fbf2364ad1b31ea4eba8f7ed83255ee8

/tmp/quantx-research/elastic-smallcap-soil-v1/elastic_smallcap_soil_val63_2026_diagnostic.json
sha256:5c2ee28698dc90ee5ab67dbd252886470d51cfe7ea72c8d908d6aa75f2714782
```

评估口径：全主板，due5 周频信号，T 日 after-close 特征，T+1 open 入场，T+6 open 退出，5 日收益减同日全市场均值。2021-2025 使用年度 walk-forward，每年只用更早年份训练；2026 forward 只用 2021-2025 训练，不使用 2026 标签。

特征只使用 T 日及以前的 `open/high/low/close/volume/vwap` 和滚动历史。由于本地股票字段没有稳定 `amount`，成交额使用 `vwap * volume` 的 20 日均值横截面 rank 作为代理。模型包括一个 5 日超额收益 regressor、一个同日 top5 winner classifier，以及一个手写 `not_overheated_elastic` 静态反事实；`blend` 是 reg/cls 日内 rank 加少量静态分数。

### 102.3 结果

关键结果如下：

| 方案 | 开发期 label5 | 开发期逐年 label5 | 2026 label5 | 判读 |
| --- | ---: | --- | ---: | --- |
| `reg::top10` | `+0.020176` | `+0.0323/+0.0105/+0.0151/+0.0230` | `+0.007025` | 开发期头部有弹性，但 2026 衰减 |
| `reg::top15` | `+0.014718` | `+0.0226/+0.0084/+0.0122/+0.0157` | `+0.004335` | 开发期 Top15 接近 Exp40，但 2026 弱 |
| `reg::top20` | `+0.012121` | `+0.0176/+0.0081/+0.0108/+0.0121` | `+0.001070` | Top20 开发期低于 Exp40，2026 明显不足 |
| `blend::top20` | `+0.005292` | `+0.0115/+0.0025/+0.0025/+0.0048` | `+0.006031` | 2026 Top20 最好但仍只有 Exp40 一半左右 |
| `static::top20` | `+0.004842` | `+0.0073/+0.0032/+0.0055/+0.0033` | `+0.003156` | 手写弹性土壤很薄 |
| `cls::top20` | `-0.007455` | 四年均弱或负 | `-0.001044` | top5 winner 分类器无法转成等权 Top20 |

对照基线：Exp40 `path_sequence_ranker_v1` due5 Top20 label 开发期为 `+0.014170`，2026 为 `+0.012520`。因此本轮最佳 Top20 在开发期低于基线，2026 更明显落后。`reg::top15` 的开发期 `+0.014718` 只是 Top15 局部接近，并没有形成用户要求的 Top20 自然厚度；且 2026 只有 `+0.004335`。

特征重要性也说明模型主要还是在学习市场状态和中期强弱，而不是打开一个新的小盘弹性源。开发期 reg 重要性最高为 `market_disp20`、`market_ret60_median`、`market_ret20_median`、`market_disp20_q252`、`market_breadth20`；`low_price_rank` 有贡献但排在其后。2026 也类似，reg 最高仍是市场 20/60 日收益、离散度、宽度和波动。

### 102.4 反事实分析

第一反事实：如果低价格/低成交额弹性是真正独立右尾土壤，手写 `static` 至少应在 Top20 形成正且接近 path 的标签。实际开发期 `static::top20` 只有 `+0.004842`，2026 只有 `+0.003156`，说明裸弹性结构很薄。

第二反事实：如果 ML 能从弹性特征里学到稳定右尾，`reg/cls/blend` Top20 应超过 Exp40 path_sequence。实际开发期最好 Top20 `+0.012121`，低于 Exp40 `+0.014170`；2026 最好 Top20 `+0.006031`，低于 Exp40 `+0.012520`。这不是 formal 交易摩擦导致的失败，而是在预测层已经不够厚。

第三反事实：如果问题只是 Top20 太宽，Top10/15 可能提示一个可用的高收益 sleeve。实际 `reg::top10` 开发期为 `+0.020176`，但 2026 只有 `+0.007025`；`reg::top15` 开发期接近基线但 2026 降到 `+0.004335`。这更像开发期风格暴露，而不是可穿越的右尾候选源。

第四反事实：如果 top5 winner 分类器能解决右尾/左尾纠缠，cls Top20 应提升 winner rate 同时保持正均值。实际 `cls::top20` 开发期 label 为 `-0.007455`，2026 为 `-0.001044`，虽然 winner_top5_rate 不低，但等权收益被左尾吞掉，重复了大涨股归因和右尾事件实验中的老问题。

### 102.5 判定

`rejected`。

弹性小盘/低成交额结构有解释力，但不是当前要找的独立主 Alpha。它在开发期 Top10/15 局部有收益，Top20 自然厚度不足，2026 前向更弱；继续围绕低价、低成交额、不过热阈值或 cls/reg 融合权重调参，容易变成风格后验拟合。按前置门槛，本轮不进入 formal account，也不生成 PredictionStore。

下一步应从这轮失败里吸取两点：第一，真正的收益源仍可能是市场状态和路径强度的共振，但不能用普通全市场点预测直接捕捉；第二，低流动性/小盘弹性会同时携带左尾，必须换成更强的候选事件或更可信的外部信息，而不是继续在同一日线 OHLCV 特征里调权重。

## 103. beta_industry_residual_label_v1

### 103.1 假设

Exp102 失败后，一个更接近“市场风险 + 横截面因子”的反事实是：现有一周标签虽然减去了同日全市场均值，但仍可能被个股 beta 和行业 beta 污染。若模型学到的是“高 beta + 强市场/行业”的混合暴露，2026 风格切换时会失效；若把训练目标改成未来 5 日收益中的市场 beta 残差、行业残差或二者组合，模型可能学到更纯的 idiosyncratic alpha，同时真实 excess Top20 也会变厚。

这轮不是已有二阶段 residual 重排的重复。旧实验多在 base/path 候选池内用状态或行业/概念残差微调排序；本轮直接改变训练标签，并从全市场 due5 面板训练，再统一用真实 future excess label 裁判。只有真实 excess Top20 在开发期和 2026 都超过 Exp40，才考虑进入 formal account。

### 103.2 产物和口径

```text
/tmp/quantx-research/beta-industry-residual-label-v1/analyze_beta_industry_residual_label.py
sha256:92d77b9118fdfba42323885fdd54248461368e33a1511ab82502662f001ba4c8

/tmp/quantx-research/beta-industry-residual-label-v1/beta_industry_residual_label_dev_2021_2025_diagnostic.json
sha256:cc4df7f09a62de410cee5ef39531ca0b02f16aaf5762cd6da60addaa447582a3

/tmp/quantx-research/beta-industry-residual-label-v1/beta_industry_residual_label_val63_2026_diagnostic.json
sha256:151977628a1c6a9354687f6fc3e5004b9b7ab8183ba3a95f991c56809b2930ff
```

评估口径：全主板，due5 周频信号，T 日 after-close 特征，T+1 open 入场，T+6 open 退出。2021-2025 使用年度 walk-forward，每年只用更早年份训练；2026 forward 只用 2021-2025 训练，不使用 2026 标签。

特征只使用 T 日及以前的 `open/high/low/close/volume/vwap`、120 日历史 beta、路径 RPS、成交额代理、近高/均线位置、行业 20 日强弱和市场宽度/离散度。训练目标包括：

1. `excess`：未来 5 日 raw return 减同日全市场均值。
2. `beta_resid`：未来 raw return 减 `beta120 * market_future_return`。
3. `industry_resid`：未来 raw return 减所属行业未来均值。
4. `beta_industry_resid`：未来 raw return 减 beta 市场项，再减行业相对全市场项。

每个目标分别训练 regressor、top5 classifier 和 reg/cls blend。注意：残差标签当然包含未来市场/行业收益，但只作为历史训练标签使用；预测 2026 时模型只输入 T 日可见特征，不输入未来市场或行业收益。最终裁判始终回到真实 `excess`，防止只在残差目标上自洽。

行业映射仍是 2026-06-25 静态东财一级行业快照，因此本轮即使通过也还需要 point-in-time 行业成员补证。

### 103.3 结果

关键 Top15/20 结果：

| 方案 | 开发期真实 excess | 开发期逐年 excess | 2026 真实 excess | 判读 |
| --- | ---: | --- | ---: | --- |
| `beta_industry_resid_reg::top20` | `+0.016343` | `+0.0247/+0.0082/+0.0219/+0.0107` | `+0.003602` | 开发期过 Exp40，2026 失效 |
| `industry_resid_reg::top20` | `+0.015125` | 开发期四年均正 | 低于 2026 最优 | 开发期有增量但不穿越 |
| `beta_resid_reg::top20` | `+0.013707` | 开发期四年均正 | `+0.004453` | 低于 Exp40 开发期和 2026 |
| `beta_resid_cls::top20` | 开发期为负或弱 | 未作为开发期候选 | `+0.011176` | 2026 最好 Top20，但仍低于 Exp40 |
| `beta_industry_resid_blend::top20` | `+0.011334` | 弱于 reg | `+0.008802` | 两端都不足 |

对照基线：Exp40 `path_sequence_ranker_v1` due5 Top20 label 开发期为 `+0.014170`，2026 为 `+0.012520`。因此本轮的核心发现是：开发期 `beta_industry_resid_reg::top20` 真实 excess 达到 `+0.016343`，是少数能在 Top20 预测层超过 Exp40 的新方向；但 2026 最好的 Top20 是另一类 `beta_resid_cls::top20`，也只有 `+0.011176`，仍低于 Exp40。

逐年看，开发期最强 `beta_industry_resid_reg::top20` 四年都为正：2022 `+0.024666`、2023 `+0.008202`、2024 `+0.021937`、2025 `+0.010738`。它的平均 beta 约 `0.928`，说明确实降低了一部分市场 beta 暴露。但 2026 中该方案降到 `+0.003602`，不是轻微回撤，而是目标结构失效。

特征重要性显示，模型仍高度依赖市场环境：开发期和 2026 前列都是 `market_disp20`、`market_ret20_median`、`market_ret60_median`、`market_disp20_q252`、`market_breadth20`。这说明 residual label 没有让模型摆脱市场状态，而是改变了市场状态下选择哪类个股的方式。

### 103.4 反事实分析

第一反事实：如果普通 excess label 的问题只是 beta/行业污染，那么 beta+industry residual 训练应提高真实 excess Top20。开发期确实成立：`beta_industry_resid_reg::top20` 从 Exp40 due5 `+0.014170` 提到 `+0.016343`，且四年全正。这说明方向不是纯噪声，残差标签有研究价值。

第二反事实：如果残差标签学到的是稳定 idiosyncratic alpha，2026 应至少不弱于 Exp40。实际 2026 `beta_industry_resid_reg::top20` 只有 `+0.003602`，明显失效；2026 最好的 `beta_resid_cls::top20` 也只有 `+0.011176`，仍低于 Exp40 `+0.012520`。因此它不是可直接推进的穿越候选。

第三反事实：如果 2026 只需要把 reg 换成 cls 或换成 beta-only 目标，那开发期对应方案也应有合理表现。实际 classifier 方案在开发期 Top15/20 多为负或明显弱，不能因为 2026 后验较好就切换模型；那会变成年度风格事后选择。

第四反事实：如果行业中性化是核心，`industry_resid_reg` 应两端都稳定。实际它开发期 Top20 为 `+0.015125`，但 2026 不在最优队列；而 beta+industry 组合开发期最强、2026 失效。说明行业 beta 和市场 beta 的扣除方式本身有阶段性，不能直接当成长期稳定标签。

第五反事实：如果 residual label 值得进入 formal account，预测层必须先同时超过 Exp40。实际 2026 未过线，因此不生成 PredictionStore，不做 formal account，避免在一个前向弱于基线的方向上浪费账户层调参。

### 103.5 判定

`rejected_with_signal`。

Beta/行业残差标签是本轮少数有真实增量的方向：开发期 Top20 比 Exp40 更厚，且降低了平均 beta，说明“剥离市场/行业 beta 后训练”确实改变了收益结构。但它没有通过 2026 前向，且 2026 有效模型形态从 reg/beta+industry 转成 cls/beta-only，属于明显风格翻转。当前不能合代码，也不能进入 formal account。

后续不应做的事：不要根据 2026 后验在 `reg/cls/beta/industry` 之间硬切换，也不要继续微调残差权重。更有价值的下一步是把这个信号上升一层：研究什么样的市场状态下应学习 raw/excess 右尾，什么状态下应学习 beta/行业残差，而且这个状态识别必须是因果的、能在开发期和 2026 同时成立。若做不到，残差标签只能作为候选生成或特征工程线索，而不是独立策略方向。

## 104. residual_target_online_selector_v1

### 104.1 假设

Exp103 的核心矛盾是：开发期最强目标是 `beta_industry_resid_reg`，但 2026 最强 Top20 变成 `beta_resid_cls`，且二者都不能同时穿越。一个最小反事实是：如果这种目标/模型形态切换有短期延续性，那么只用已完成历史 session 的真实 excess 表现，应能在线选择 `excess/beta/industry/beta+industry` 与 `reg/cls/blend`，并在开发期和 2026 同时接近非因果 oracle。

本轮不重新设计特征，不生成 PredictionStore，只复用 Exp103 的全市场 due5 多目标分数，测试目标选择是否能被因果识别。

### 104.2 产物和口径

```text
/tmp/quantx-research/residual-target-online-selector-v1/analyze_residual_target_online_selector.py
sha256:45725efa324b83727db7ec332ad2397816a7225333927a6012ccb3f38a1f0026

/tmp/quantx-research/residual-target-online-selector-v1/residual_target_online_selector_dev_2021_2025_diagnostic.json
sha256:1ea385ce337600db0bd49bce1d35ea8dab4acb8638ff971fb45111fa323c7684

/tmp/quantx-research/residual-target-online-selector-v1/residual_target_online_selector_val63_2026_diagnostic.json
sha256:255db8bff6610a5ab1d8dc1a5210b330a7ff6a8a646844614d774053927d7076
```

口径：全主板 due5，T 日信号，T+1 open 入场，T+6 open 退出，最终评价仍为真实 excess label。候选模型为 Exp103 的 12 个分数：四类目标 `excess/beta_resid/industry_resid/beta_industry_resid` 乘以 `reg/cls/blend`。

因果约束：selector 在当前 session 只能使用至少滞后两个 due5 信号的历史表现，确保对应 T+6 label 已完成。2026 forward 的 selector 历史使用 2022-2025 walk-forward OOS 表现，不使用 2021-2025 final model 的 in-sample 表现；2026 股票打分仍由 2021-2025 训练出的 final model 生成。

选择规则包括 expanding 全候选、rolling 6/12/24 全候选、rolling 12 仅 reg、rolling 12 residual-only、shrunk rolling 12。非因果 `oracle_same_session` 只作为上限诊断，不作为策略证据。

实现注记：脚本最终 print 摘要主要列出 fixed/oracle；selector 结果在 JSON 顶层 `results` 中，本文以 JSON 直接读取为准。

### 104.3 结果

开发期 Top20：

| 方案 | 真实 excess | 正 excess session 比例 | 主要选择 | 判读 |
| --- | ---: | ---: | --- | --- |
| fixed `beta_industry_resid_reg::top20` | `+0.016343` | `66.32%` | 固定 | Exp103 固定最佳 |
| `expanding_all_min8::top20` | `+0.015558` | `63.73%` | 多数选 `beta_industry_resid_reg` | 过 Exp40，但低于固定最佳 |
| `rolling12_reg_only_min8::top20` | `+0.015201` | `65.28%` | 在 reg 目标间切换 | 接近 fixed，但仍弱 |
| `rolling12_residual_only_min8::top20` | `+0.013698` | `60.62%` | 多目标切换 | 低于 Exp40 |
| `rolling6_all_min6::top20` | `+0.008331` | `56.99%` | 高频切换 | 明显过度追噪声 |
| 非因果 oracle Top20 | `+0.043822` | - | 同日最优 | 上限巨大但不可交易 |

2026 Top20：

| 方案 | 真实 excess | 正 excess session 比例 | 主要选择 | 判读 |
| --- | ---: | ---: | --- | --- |
| Exp40 due5 Top20 基线 | `+0.012520` | - | 固定 path | 主基线 |
| fixed `beta_resid_cls::top20` | `+0.011176` | `62.50%` | 2026 后验固定最优 | 仍低于 Exp40 |
| `rolling6_all_min6::top20` | `+0.003906` | `58.33%` | 多目标快速切换 | 因果 selector 最好，但很弱 |
| `rolling24_all_min8::top20` | `+0.003766` | `58.33%` | 大多选 `beta_industry_resid_blend` | 未识别 2026 最优 |
| `expanding_all_min8::top20` | `+0.003602` | `58.33%` | 全部选 `beta_industry_resid_reg` | 延续开发期赢家，2026 失败 |
| `rolling12_all_min8::top20` | `-0.002334` | `54.17%` | 多目标切换 | 反向 |
| 非因果 oracle Top20 | `+0.042254` | - | 同日最优 | 上限仍大但不可因果识别 |

开发期最好的因果 selector 虽然高于 Exp40 due5 `+0.014170`，但低于固定 `beta_industry_resid_reg`；2026 所有因果 selector 都远低于 Exp40，甚至低于第 103 轮 2026 后验固定最佳 `beta_resid_cls::top20`。

### 104.4 反事实分析

第一反事实：如果残差目标的风格切换有稳定短期延续，rolling selector 应超过固定目标。实际开发期 rolling12/reg-only Top20 为 `+0.015201`，低于固定 `+0.016343`；rolling6 更差，说明近期表现追踪主要放大噪声。

第二反事实：如果 2026 的目标翻转可由历史 OOS 表现识别，selector 应逐步转向 `beta_resid_cls`。实际 expanding 全程选开发期赢家 `beta_industry_resid_reg`，rolling24 大多选 `beta_industry_resid_blend`，rolling6 虽偶尔选 `beta_resid_cls` 但整体只有 `+0.003906`。历史表现没有给出可靠切换信号。

第三反事实：如果目标选择本身是金矿，非因果 oracle 强不应只是同日噪声。实际 oracle 在开发期和 2026 都有 `+0.04` 以上 Top20 excess，但因果 selector 几乎拿不到，说明目标/模型之间确实存在状态空间，但当前可见历史表现无法识别。

第四反事实：如果这轮值得进 formal，至少预测层 2026 应超过 Exp40。实际 2026 因果 selector 最好仅 `+0.003906`，远低于 Exp40 `+0.012520`，因此不生成账户预测，也不做 formal 回放。

### 104.5 判定

`rejected`。

残差目标在线选择失败。它证明了两件事：一是目标/模型选择的非因果上限很高，市场确实在不同状态下奖励不同标签结构；二是“用近期已完成表现选择目标”不足以因果识别 2026 的风格翻转。后续不继续调 rolling 窗口、min history、shrunk 系数或 candidate subset。

下一步必须离开“在已有标签/模型之间路由”的框架，转向更有信息增量的候选生成：要么引入更外生的数据线索，要么把 A 股主线生命周期做成可检验的事件结构，而不是继续用历史表现猜哪个目标今天有效。

## 105. learned_group_lifecycle_v1

### 105.1 假设

Exp104 之后，继续在已有 path/base 或 residual 目标之间路由已经没有意义；它们有非因果上限，但因果识别失败。本轮转向更不同的候选生成：先用行业/概念组自身的多日路径、宽度、涨停/近涨停比例、热度衰减和市场状态学习未来 5 日组机会，再从全市场股票中按高机会组生成候选，而不是先依赖 Exp40 的股票候选池。

如果 A 股“主线生命周期”是真正可交易 alpha，它不应只在手写行业/概念公式里解释历史，也不应只作为 Exp40 的补位层，而应能从组层机会自然生成一组 Top20 股票候选，并在开发期和 2026 的 due5 标签层同时超过 Exp40 `path_sequence_ranker_v1`。

### 105.2 产物和口径

```text
/tmp/quantx-research/learned-group-lifecycle-v1/analyze_learned_group_lifecycle.py
sha256:95d793d5a4f41100b1d8871bd4ab667f7c66f67c5ce1b653b76ee9e422b7a7c4

/tmp/quantx-research/learned-group-lifecycle-v1/learned_group_lifecycle_dev_2021_2025_diagnostic.json
sha256:e14170e0a0245211040d8b4e53cfa26dc5afa33af4e45a1bad8a1b1311ea42a1

/tmp/quantx-research/learned-group-lifecycle-v1/learned_group_lifecycle_val63_2026_diagnostic.json
sha256:f3d69cfc11b25a810ab0c962407ba3e621279deb6888e44228b5129ea89246d0
```

口径：全主板，T 日 after-close 特征，T+1 open 入场，T+6 open 退出，评价为未来 5 日 raw return 减同日全市场均值。开发期为 2021-2025 年度 walk-forward；2026 forward 只用 2021-2025 训练，不使用 2026 标签。

组层训练目标为行业/概念组未来 T+1 open 到 T+6 open 的组 excess return。股票端测试五种候选生成方式：`group_only`、`group_leader`、`group_catchup`、`group_balanced`、`group_early_not_chase`。特征和打分都只使用 T 日及以前的收盘、成交量、组内宽度、组强度和市场状态。

边界：行业/概念成员仍使用 2026 静态快照，因此本轮即使通过也只能作为诊断；若要合代码，还需要 point-in-time 行业/概念成员补证。

### 105.3 结果

关键 Top20 结果如下：

| 方案 | 开发期 label5 | 开发期正样本期比例 | 2026 label5 | 2026 正样本期比例 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all::group_only::top20` | `-0.000996` | `46.83%` | `+0.005980` | `60.17%` | 开发期最好但仍为负 |
| `due5::group_early_not_chase::top20` | `-0.002335` | `40.41%` | `-0.007441` | `45.83%` | 不追高版本两端不足 |
| `all::group_early_not_chase::top20` | `-0.002648` | `41.33%` | `-0.005404` | `38.98%` | 早段/不过热没有形成收益厚度 |
| `all::group_leader::top20` | `-0.009338` | `39.36%` | `+0.007515` | `56.78%` | 2026 最好但开发期显著反向 |
| `due5::group_leader::top20` | `-0.009174` | `37.82%` | `+0.007449` | `50.00%` | 2026 局部有效，不能穿越开发期 |

对照基线：Exp40 `path_sequence_ranker_v1` due5 Top20 label 开发期为 `+0.014170`，2026 为 `+0.012520`。本轮开发期所有 Top20 方案均为负；2026 虽然 `group_leader` 局部转正到约 `+0.0075`，仍明显低于 Exp40。由于预测层开发期已经失败，本轮不进入 formal account，也不生成 PredictionStore。

特征重要性显示，组预测器仍主要依赖市场状态而不是稳定的组生命周期结构：开发期前列为 `market_disp20`、`market_ret20_median`、`market_breadth20`、`market_ret5_median`、`market_ret60_median`，组自身的 `heat_decay`、`ret60_rank`、`group_size_rank` 有贡献但不主导。2026 也类似，`market_ret20_median`、`market_disp20`、`market_ret5_median` 和 `market_breadth20` 排在前列。

### 105.4 反事实分析

第一反事实：如果组层机会预测能作为独立主引擎，`group_only` 至少应在开发期 Top20 形成稳定正 excess。实际开发期最好 `all::group_only::top20` 只有 `-0.000996`，说明只知道“哪个组好”不足以选出一周等权股票收益。

第二反事实：如果主线生命周期的关键是龙头延续，`group_leader` 应在开发期和 2026 同时有效。实际 `all::group_leader::top20` 开发期为 `-0.009338`，只有 2026 转正到 `+0.007515`，更像阶段性追强风格，而不是长期鲁棒 alpha。

第三反事实：如果问题只是追高，`group_early_not_chase` 应改善开发期和 2026。实际开发期仍为负，2026 也为负，说明“早段且不过热”的手写约束没有解决组热度和个股收益之间的错位。

第四反事实：如果 2026 正收益代表新方向，至少应超过 Exp40 2026 due5 `+0.012520`。实际本轮 2026 最好 Top20 约 `+0.007515`，只有基线六成左右；开发期又全线为负，因此没有推进账户层的必要。

### 105.5 判定

`rejected`。

学习型组生命周期候选生成失败。它验证了一个重要边界：行业/概念组层机会可以解释部分市场状态和 2026 局部风格，但从静态行业/概念映射出发，直接生成全市场 Top20 股票候选并不能形成厚的一周 alpha。主线生命周期目前仍更适合作为风险解释、补位线索或事后归因，而不是新候选主引擎。

后续不继续围绕组热度、leader/catchup、early-not-chase 公式调权重。若要重新研究主线生命周期，需要更外生或更 point-in-time 的信息，例如实时概念归属、题材新闻、龙虎榜/资金流、成交席位或更细的盘口数据；仅靠当前日线 OHLCV 加静态行业/概念快照，已经多轮验证不足以突破 Exp40。

## 106. brick_active_value_walkforward_v1

### 106.1 假设

前面多轮基于 Exp40 path/base、行业/概念静态快照、ETF 状态和 residual target 的实验都没有找到新的收益弹性层。仓库中另有一条外部启发线索：Brick V11 复现实验。它的结构与 Exp40 明显不同：先用全市场成交额代理识别活跃市值波段，再在红砖/强涨幅/成交活跃候选中训练动态退出 forward label，卖出由绿砖、7 红风险或最长 10 日触发。

已有 Brick 文档中，严格 2013-2024 训练、2025-2026 OOS 的尾部模型在 2025 较强，但执行层高收益依赖 2025-2026 上的 score floor / max positions 搜索，不能作为封闭 OOS 证据。因此本轮只做年度 walk-forward 复核：如果 Brick/活跃市值是真正独立右尾土壤，它应该在 2022-2025 多年和 2026 forward 同时形成正的候选质量，而不是只在 2025 单段有效。

### 106.2 产物和口径

```text
/tmp/quantx-research/brick-active-value-walkforward-v1/analyze_brick_active_value_walkforward.py
sha256:01b0ff5138ac39d0102f43c1897939f18b880c64064aee924495304bf7cb2095

/tmp/quantx-research/brick-active-value-walkforward-v1/brick_active_value_walkforward_2022_2026_diagnostic.json
sha256:90f6ae5c9b846b7355ca98c871659f6c002b3949a9c734e29a70e26a65048d77

/tmp/quantx-research/brick-active-value-walkforward-v1/brick_active_value_walkforward_2022_2026_diagnostic.scored.parquet
sha256:99464bbea1a83a61adda5147f48014acfe32dd709e1cfded97799e9cc199e2b0
```

本轮复用 `/tmp/quantx_brick_v11_forward_oos_v1/` 中已有年度 parquet，不重建因子系统。训练样本按年度拼接，并强制过滤 `exit_date <= train_end`，避免 2024 年底信号的 2025 退出收益进入 2025 训练，也避免任何 2026 标签进入 2026 训练。

信号和交易标签边界如下：T 日特征和活跃市值 gate 只用 T 日及以前信息；买入为 T+1 open，包含买入滑点；动态退出在 T+N open 前只使用上一交易日砖形图状态，即代码中 `decision = e - 1`；最终评价为按动态退出实际得到的 `forward_return`。本轮不使用公开 Top5 teacher label，不使用 2025-2026 上搜索出的 score floor / max positions，也不进入账户层。

年度 fold：

| 测试年 | 训练截止 | 训练行数 | 测试行数 | 信号日 |
| --- | --- | ---: | ---: | ---: |
| 2022 | 2021-12-31 | 313581 | 38040 | 102 |
| 2023 | 2022-12-31 | 351973 | 53120 | 111 |
| 2024 | 2023-12-31 | 404148 | 43633 | 107 |
| 2025 | 2024-12-31 | 448249 | 51661 | 116 |
| 2026 | 2025-12-31 | 448956 | 26859 | 57 |

模型只保留轻量验证：`hgb_rank_tail`、`hgb_blend_tail`、`hgb_top10_cls` 和三者日内 rank 融合。目标包括 `label_rank_blend`、`label_blend` 和 `label_top10`，并用训练期 top10/top20/正收益样本加权。注意这不是用户反对的“降低尾部权重卡要求”，而是候选质量验证；若 label 层不过，直接拒绝。

### 106.3 结果

关键 label 层结果：

| 方案 | 五年均值 | 2022 | 2023 | 2024 | 2025 | 2026 | 平均持有 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 候选池均值 | `-0.001416` | `-0.003265` | `-0.003928` | `+0.003179` | `+0.000846` | `-0.006447` | `2.61` | 活跃市值+红砖候选本身不是正收益土壤 |
| `hgb_rank_tail::top5` | `+0.000546` | `-0.006612` | `-0.005268` | `+0.002929` | `+0.015355` | `-0.009894` | `2.75` | 2025 强，但 2022/2023/2026 反向 |
| `hgb_rank_tail::top10` | `-0.000498` | `-0.006334` | `-0.005055` | `+0.004667` | `+0.009050` | `-0.010214` | `2.77` | Top10 不穿越 |
| `hgb_rank_tail::top20` | `-0.001298` | `-0.005843` | `-0.006226` | `+0.004017` | `+0.004718` | `-0.005606` | `2.77` | Top20 更薄 |
| `fusion_rank::top5` | `-0.000364` | `-0.007595` | `-0.010147` | `+0.002207` | `+0.014135` | `-0.002659` | `2.82` | 融合不能修复穿越性 |
| `hgb_top10_cls::top10` | `-0.001357` | `-0.010657` | `-0.007754` | `+0.001814` | `+0.010043` | `-0.001357` | `2.80` | 分类目标仍反向 |

最好的五年均值只是 `hgb_rank_tail::top5 = +0.000546`，且只靠 2025 拉高；2022、2023 和 2026 均为负。Top10/Top20 不但没有自然变厚，五年均值直接为负。平均持有期约 2.7-2.8 个交易日，明显短于用户希望的一周左右，也不满足平均持仓大于 5 的后续账户目标前提。

因此，本轮不进入 formal account。即使 2025 执行层通过 score floor / max positions 搜索能得到较高收益，也已经被年度 label 层证明为后验集中到单年风格，而不是自然鲁棒 alpha。

### 106.4 反事实分析

第一反事实：如果活跃市值波段本身是高质量市场状态，候选池均值应在多数年份为正。实际候选池均值五年为 `-0.001416`，2022、2023、2026 都为负，说明活跃成交环境也会放大失败红砖，并非天然收益土壤。

第二反事实：如果尾部加权模型学到稳定右尾，Top5 至少应穿越 2022-2026。实际 `hgb_rank_tail::top5` 只有 2024/2025 为正，2022/2023/2026 为负。它更像 2025 单年风格拟合，而不是可跨周期右尾识别。

第三反事实：如果问题只是 Top5 太窄，Top10/Top20 应更稳。实际 `hgb_rank_tail::top10/top20` 五年均值为负，说明扩大持仓后收益更薄，不符合“平均持仓 >5 且自然出收益”的要求。

第四反事实：如果 2026 弱只是某个目标失效，分类目标或融合应改善。实际 `hgb_top10_cls::top10` 和 `fusion_rank` 仍为负，2026 没有被修复。

第五反事实：如果这条线能作为低相关 sleeve，至少 label 层不能系统反向。实际 2022/2023/2026 都反向，且持有期短于目标，低相关不等于有正期望；不能为了组合平滑加入负 alpha。

### 106.5 判定

`rejected`。

Brick/活跃市值/动态退出 forward label 是一条有外部启发、结构上独立于 Exp40 的候选土壤，但严格年度 walk-forward 后没有通过。2025 的强表现不能代表穿越 alpha；2026 forward 失效，Top10/20 不厚，平均持有期也偏短。本轮不做账户层、不生成策略配置、不继续围绕 score floor、topk、max positions 或 top-heavy 权重搜索。

这轮的主要价值是把“公开高收益 Brick 线索”拆成两部分：蒸馏公开 Top5 可以复现页面，2025 上执行阈值搜索也能做出高收益，但一旦改成封闭年度 walk-forward，它没有自然穿越。后续若再研究外部启发，必须优先找能提供 point-in-time 新信息的数据源，而不是继续在当前 OHLCV 红砖候选里调模型或执行参数。

## 107. big_winner_attribution_v2

### 107.1 假设

用户提出可以先分析历史上大幅上涨的股票，再用 ML 做归因。第 30 轮曾做过大涨股归因，但当时信号偏头部、Top20 不厚。本轮把它改成更严格的因果候选验证：先在全市场日线面板上学习未来 20 日大涨结构，再把学到的分数带回 Exp40 `path_sequence_ranker_v1` 的 Top200 候选池，看它能否作为收益弹性层自然增厚一周 Top20。

核心反事实：如果历史大涨股归因是真正可复用的右尾土壤，它不应只提升 Top10，也不应只在开发期有效；它应在开发期和 2026 的 Exp40 Top200 候选池内同时改善 `due5 Top20`，且至少超过 Exp40 强基线。

### 107.2 产物和口径

```text
/tmp/quantx-research/big-winner-attribution-v2/analyze_big_winner_attribution.py
sha256:b2236061e9e8fe6e5fbe38382c830d893e6eb08747595cc96a333782b4cdbe4e

/tmp/quantx-research/big-winner-attribution-v2/big_winner_attribution_v2_dev_2021_2025_diagnostic.json
sha256:f46d8df8940ea422842caa79ecc60c4ec3d2d4b23cc05add0a1e57aa844bcbda

/tmp/quantx-research/big-winner-attribution-v2/big_winner_attribution_v2_val63_2026_diagnostic.json
sha256:596045ef3e96092bfc5ec49e44a6e1e4c9cdd50eeb5e18c61a86e9d5d4b600ed
```

本轮复用第 40 轮的 `analyze_path_sequence_ranker.py` 数据读取、行业/概念静态映射、路径特征矩阵和 Exp40 Top200 候选口径。开发期输入候选为：

```text
/tmp/quantx-research/industry-concept-rotation-v1/base-5d-dev/mainboard_lgbm_base_5d_2010_2021_2025_seed7-9a8e7ff285664cef/oos_predictions.json
```

2026 输入候选为：

```text
/tmp/quantx-research/adaptive-training-protocol-v1/forward-2026-5d-val63/mainboard_lgbm_base_5d_2010_2026_forward_val63_seed7-c52c47f834608a3c/oos_predictions.json
```

因果边界如下：

1. 全市场归因训练样本只使用信号日 T 及以前可见的路径、量能、行业/概念和市场宽度特征。
2. 大涨标签为 T+1 open 到 T+21 open 的未来 20 日收益，训练 fold 内作为历史标签使用；测试年只用 `< year` 的历史样本训练。
3. `target_big20` 为当日全市场未来 20 日 raw 和 excess 同时进入前 5%；`target_clean20` 在此基础上要求未来前 5 日 open 路径回撤不差；`reg20` 回归未来 20 日 excess。
4. 开发期按年度 walk-forward：2022/2023/2024/2025 的训练行数分别为 143719/288703/440148/590379，测试候选行数为 48400/48400/48400/47400。
5. 2026 forward 只用 2021-2025 全市场归因样本训练，训练行数 732428，测试候选行数 23599。
6. 评价仍是 Exp40 同口径一周标签：T 日 after-close 信号，T+1 open 到 T+6 open 的 5 日 excess return；主判定看 `due5::pool200::top20`。

### 107.3 结果

开发期主口径：

| 方案 | `due5 pool200 Top20` | 相对本轮 base | 2022 | 2023 | 2024 | 2025 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | `+0.013524` | `0.000000` | `+0.019758` | `+0.008658` | `+0.014523` | `+0.011108` | Exp40 Top200 输入基准 |
| `blend_attr35` | `+0.014699` | `+0.001174` | `+0.020182` | `+0.008403` | `+0.016548` | `+0.013640` | 开发期唯一有小幅 Top20 增量 |
| `blend_clean25` | `+0.013806` | `+0.000282` | `+0.018552` | `+0.007090` | `+0.016287` | `+0.013286` | 增量很薄 |
| `big20` | `+0.011585` | `-0.001940` | `+0.017525` | `+0.003151` | `+0.013705` | `+0.011966` | 纯大涨概率削弱 Top20 |
| `clean20` | `+0.011365` | `-0.002159` | `+0.016208` | `+0.005427` | `+0.010907` | `+0.012949` | 干净大涨也不厚 |
| `reg20` | `+0.007592` | `-0.005932` | `+0.005081` | `+0.005758` | `+0.012526` | `+0.006992` | 20 日超额回归明显错位 |

2026 forward 主口径：

| 方案 | `due5 pool200 Top20` | 相对本轮 base | 对照 Exp40 due5 Top20 | 判读 |
| --- | ---: | ---: | ---: | --- |
| base | `+0.005062` | `0.000000` | `+0.012520` | 输入 base 不等于 Exp40 path_topq 强基线 |
| `reg20` | `+0.009669` | `+0.004606` | `+0.012520` | 2026 本轮最好但仍低于 Exp40 |
| `clean20` | `+0.007937` | `+0.002875` | `+0.012520` | 局部改善不足 |
| `attr_edge` | `+0.007326` | `+0.002264` | `+0.012520` | 不足以晋级 |
| `blend_attr35` | `+0.005455` | `+0.000393` | `+0.012520` | 开发期最佳到 2026 失效 |
| `big20` | `+0.004569` | `-0.000493` | `+0.012520` | 纯大涨概率无效 |

对照 Exp40：Exp40 `path_sequence_ranker_v1` due5 Top20 label 开发期为 `+0.014170`，2026 为 `+0.012520`。本轮开发期最佳 `blend_attr35` 为 `+0.014699`，只是约 5bp 级别的小增量；但 2026 最好 `reg20` 只有 `+0.009669`，低于 Exp40。开发期最佳和 2026 最佳不是同一模型，也不能用 2026 后验切换。

因此，本轮不进入 formal account，不生成 PredictionStore，不做 top-heavy 或阈值搜索。

### 107.4 反事实分析

第一反事实：如果历史大涨股概率就是一周右尾 alpha，`big20` 应在 Top20 直接胜出。实际开发期 `big20` 从 base `+0.013524` 降到 `+0.011585`，2026 也只有 `+0.004569`。说明 20 日大涨概率更偏中期右尾，不等于下一周 Top20 等权收益。

第二反事实：如果问题只是大涨样本路径太脆弱，加入前 5 日路径不差的 `clean20` 应修复。实际 `clean20` 开发期仍低于 base，2026 虽有 `+0.002875` 的本轮增量，但绝对值 `+0.007937` 仍低于 Exp40。干净大涨目标会筛掉一部分波动失败样本，也同步筛掉收益弹性。

第三反事实：如果 20 日超额回归学到更连续的趋势收益，开发期和 2026 都应稳定。实际 `reg20` 开发期 Top20 只有 `+0.007592`，显著低于 base；2026 虽变成本轮最好，但仍低于 Exp40。这是典型风格翻转，不能按 2026 后验选择。

第四反事实：如果大涨归因可以作为 Exp40 的收益弹性层，开发期最佳 `blend_attr35` 应在 2026 保留。实际它在 2026 只有 `+0.005455`，几乎退回本轮 base。开发期的增量更像对 2024/2025 风格的局部拟合。

第五反事实：如果只需扩大到 pool200 就能自然增厚，`all` 和 `due5` Top20 应共同改善。实际开发期 `all::pool200::blend_attr35::top20` 为 `+0.012485`，低于 all base `+0.012562`；只有 `due5` 抽样口径略增。这个增量不够稳健。

### 107.5 判定

`rejected`。

历史大涨股可以被 ML 归因，重要变量仍集中在多日强弱、波动、市场宽度/离散度、行业概念热度这些已有路径世界内。但它不能稳定转化为一周 Top20 厚收益：纯 winner、clean winner 和 20 日回归目标分别对应头部化、削弹性或风格翻转。开发期局部小增量不足以覆盖 2026 失效。

后续不继续围绕未来 20 日大涨概率、干净大涨、Top5 winner 分类或 blend 权重调参。若继续研究“历史大涨股归因”，需要引入当前数据没有的新信息，例如事件公告、新闻题材、龙虎榜、资金流、实时概念归属或盘口结构；仅靠日线 OHLCV、静态行业/概念和已有路径特征，大涨归因仍只是 Exp40 路径土壤的弱变体。

## 108. etf_latent_flow_lead_v1

### 108.1 假设

Exp101 证明 ETF 资金流状态能解释部分 2026 regime，但没有让 path/base 路由变厚。一个更主动的反事实是：ETF 不只作为市场风险状态，也可能近似表达板块资金流。若股票对 ETF 的滚动收益暴露可作为隐含行业/主题映射，那么“强 ETF 资金/趋势暴露最高的一组股票”可能形成独立于 Exp40 的全市场候选土壤。

本轮不依赖 Exp40 prediction store，也不使用静态行业/概念映射；只用 T 日以前股票收益和 ETF 收益估计滚动相关暴露，再用 T 日 ETF 近 5/20 日强弱、成交额 z-score、risk-on 状态给股票打分。若该方向成立，应在开发期和 2026 同时形成厚的一周 Top20 标签，而不是只解释 2026 单段风格。

### 108.2 产物和口径

```text
/tmp/quantx-research/etf-latent-flow-lead-v1/analyze_etf_latent_flow_lead.py
sha256:33fdd18b1efd64fe86d0f6f91319265476e10ee2d1d0458fd9678341f86b7502

/tmp/quantx-research/etf-latent-flow-lead-v1/etf_latent_flow_lead_dev_2021_2025_diagnostic.json
sha256:460d7f5c9497b3767ff27c23e6b92f0ee2c1242f1c71f251949e4922d98453e1

/tmp/quantx-research/etf-latent-flow-lead-v1/etf_latent_flow_lead_val63_2026_diagnostic.json
sha256:2a21a035c5ccfaef84c9ce524085d6aab10954aad2b792a47593df8d0a0c3db3
```

数据和边界：

1. 股票池为 `all_mainboard`；ETF 池从 qlib features 中按 `sh51/sh56/sh58/sz15/sz16` 前缀取近 252 日成交额最高的 80 只 ETF。
2. T 日股票对 ETF 的暴露只用 T 日及以前的日收益，在 40/60/120 日窗口上计算滚动相关矩阵。
3. ETF 信号只用 T 日及以前：ETF 5/20 日收益横截面 rank、ETF 成交额 z-score、二者乘积和 risk-on composite。
4. 股票候选分数为 `exposure @ ETF_signal`，另加两个股票自身反事实：`stock_mom20` 和 `stock_lowcrowd_mom20`。
5. 标签为 T+1 open 到 T+6 open 的 5 日收益，减同日全市场均值；主口径仍看 `due5 Top20`。
6. 本轮只做标签层土壤验证，不进账户层，不生成 PredictionStore。

### 108.3 结果

2026 forward 的 ETF 20 日趋势暴露非常强：

| 方案 | 2026 `due5 Top20` | 2026 `due5 Top15` | 2026 `all Top20` | 观察 |
| --- | ---: | ---: | ---: | --- |
| `w60::etf_ret20_beta` | `+0.021557` | `+0.024477` | `+0.014831` | 2026 最强 Top20 |
| `w120::etf_ret20_beta` | `+0.021014` | `+0.024939` | `+0.019507` | all/due5 都强 |
| `w40::etf_ret20_beta` | `+0.018574` | `+0.023150` | `+0.015995` | 短窗口也有效 |
| `w120::etf_riskon_beta` | `+0.017379` | `+0.017690` | `+0.011775` | 弱于纯 20 日 ETF 趋势 |
| `stock_lowcrowd_mom20` | `+0.007154` | `+0.005855` | `+0.009866` | 股票自身低拥挤动量不足 |

但开发期完全不够厚：

| 方案 | 开发期 `due5 Top20` | 2021 | 2022 | 2023 | 2024 | 2025 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `w60::etf_ret20_beta` | `+0.002146` | `+0.001400` | `+0.003562` | `+0.003574` | `+0.002132` | `+0.000046` | 穿越为正但太薄 |
| `w120::etf_ret20_beta` | `+0.003021` | `+0.001044` | `+0.003806` | `+0.003709` | `+0.007497` | `-0.000925` | 2025 转负 |
| `w60::etf_riskon_beta` | `+0.003428` | `+0.004683` | `+0.002455` | `+0.001811` | `+0.009308` | `-0.001111` | 本轮开发期最好 Top20 仍很薄 |
| `w120::etf_riskon_beta` | `+0.002288` | `+0.000993` | `+0.005122` | `+0.000447` | `+0.009880` | `-0.004938` | 2025 明显反向 |
| `stock_lowcrowd_mom20` | `-0.001080` | `+0.001142` | `-0.002563` | `-0.000206` | `-0.002346` | `-0.001494` | 裸股票低拥挤动量失败 |

对照 Exp40：Exp40 `path_sequence_ranker_v1` due5 Top20 label 开发期约 `+0.014170`，2026 约 `+0.012520`。本轮 ETF 暴露在 2026 明显超过 Exp40，但开发期最好 Top20 只有 `+0.003428`，量级差得太远，不进入账户层。

### 108.4 反事实分析

第一反事实：如果 ETF 隐含资金流是稳定主 alpha，开发期 Top20 应接近或超过 Exp40。实际最佳开发期 Top20 只有 `+0.003428`，说明 ETF 暴露本身不是长期厚收益土壤。

第二反事实：如果 2026 强只是窗口选择问题，40/60/120 日都应在开发期稳定变厚。实际三个窗口的 `etf_ret20_beta` 开发期 Top20 都只有 `+0.002` 到 `+0.003`，并且 2025 接近 0 或转负，不能靠选窗口解决。

第三反事实：如果成交额资金流比 ETF 价格趋势更接近真实资金，`flow5/flow20` 应强于 `ret20_beta`。实际 top results 几乎都由 `etf_ret20_beta` 和 `etf_riskon_beta` 占据，成交额乘积信号没有打开上限，说明当前 ETF amount 代理没有提供足够新信息。

第四反事实：如果这只是股票自身中期动量，`stock_mom20` 或低拥挤动量应同样有效。实际 `stock_lowcrowd_mom20` 开发期为负，2026 也低于 ETF 暴露版本；ETF 映射确实捕捉了 2026 板块趋势，但历史不穿越。

第五反事实：如果 2026 的强信号可以作为合代码候选，不能依赖后验知道 2026 奖励 ETF 20 日趋势暴露。开发期 2025 已经转负，说明将其直接上线会是在做 2026 风格后验，不符合研究门槛。

### 108.5 判定

`rejected_with_regime_signal`。

ETF 隐含资金流/暴露是非常有价值的 2026 regime 解释：2026 明显奖励对强 ETF 20 日趋势暴露高的股票，`due5 Top20` 达到 `+0.021557`，超过 Exp40 2026 标签层。但开发期信号太薄，2025 已经转负，不能作为穿越候选土壤。

后续不继续调 ETF 数量、窗口长度或 flow 权重。若要使用 ETF 信息，方向应是更高层的 regime 诊断或与外部实时行业/主题数据结合，而不是单独用 ETF 暴露生成股票 TopK。下一步仍需寻找新的 point-in-time 信息源或更贴近正式可交易收益的候选生成目标。

## 109. lhb_event_alpha_v1

### 109.1 假设

Exp107 的历史大涨股归因提示：仅靠日线 OHLCV、静态行业/概念和已有路径世界，右尾结构可学习但不能稳定变成一周 Top20 厚收益。一个更不同的数据假设是：龙虎榜可能提供点时外部事件线索，反映短线资金、机构参与、换手和关注度变化。

若龙虎榜是独立右尾土壤，事件日 T 收盘后已知的净买额、买入额、成交占比、机构文本等字段，应能在 T+1 open 到 T+6 open 的一周窗口里形成厚的 Top15/20 超额收益，至少 2026 forward 不能弱于 Exp40 `path_sequence_ranker_v1` 的 `due5 Top20 +0.012520`。

### 109.2 产物和口径

```text
/tmp/quantx-research/lhb-event-alpha-v1/analyze_lhb_event_alpha.py
sha256:2766f5c18c20c0154528f575df58d216e2e0d91a892d17676fb75e11c921f9b8

/tmp/quantx-research/lhb-event-alpha-v1/lhb_event_alpha_val63_2026_diagnostic.json
sha256:7b78de47a8c0846f0f08092ed7f662f8b8005f3f52ca580d59307ec4958eb8d5
```

数据和因果边界：

1. 数据源为 AkShare `stock_lhb_detail_em`，历史区间先只取 2026-01-05 至 2026-07-10 做 forward 第一关。
2. AkShare 返回的 `上榜后1日/2日/5日/10日` 属于未来收益字段，全部剔除，不进入特征或排序。
3. 事件字段按 T 日收盘后已知处理；交易口径为 T+1 open 买入、T+6 open 退出，标签减同日全市场 5 日 open-to-open 均值。
4. 股票池为 `all_mainboard`，过滤 ST/退市/B 股相关名称或原因。
5. 测试排序包括 `net_buy_ratio`、`net_buy_to_float`、`buy_amount_to_float`、`active_net_buy`、`active_turnover`、`contrarian_sell`、`institution_text_buy`，TopK 为 10/15/20/30。
6. 抓取缓存位于 `/tmp/quantx-research/lhb-event-alpha-v1/cache`；2026-02-16 至 2026-02-22 为 1 个空缓存周，记录在 `fetch_audit.empty_cached_chunks`，其余 2026 样本完成抓取。

2026 结果覆盖：`event_rows_raw=11135`，`daily_row_count=3976`；主样本 `due5` 有 24 个信号日，平均每日约 46 个龙虎榜候选。

### 109.3 结果

2026 forward 第一关没有通过：

| 方案 | 2026 `mean_daily_label5` | 2026 `mean_daily_raw5` | 样本 | 命中率 | 判读 |
| --- | ---: | ---: | --- | ---: | --- |
| `due5::active_net_buy::top15` | `+0.000072` | `-0.002497` | 24 | `0.500` | 全部方案里最高，但收益几乎为 0，raw 为负 |
| `due5::net_buy_ratio::top15` | `+0.000002` | 未列入主摘要 | 24 | `0.500` | 近似 0 |
| `due5::active_net_buy::top20` | `-0.003961` | `-0.006530` | 24 | `0.500` | 主目标 Top20 为负 |
| `due5::net_buy_ratio::top20` | `-0.004055` | `-0.006624` | 24 | `0.500` | 主目标 Top20 为负 |
| `all::active_net_buy::top20` | `-0.005010` | `-0.008078` | 118 | `0.449` | 全样本 Top20 也为负 |

对照 Exp40：Exp40 `path_sequence_ranker_v1` 2026 `due5 Top20` label 约 `+0.012520`。本轮龙虎榜 Top20 不仅低于 Exp40，而且为负；Top15 虽略正，也只有 0.7bp 量级，且原始 5 日收益为负。

因此本轮没有继续拉 2021-2025 长数据，也不进入账户层、不生成 PredictionStore。

### 109.4 反事实分析

第一反事实：如果龙虎榜净买入代表可交易短线资金延续，`net_buy_ratio` 或 `active_net_buy` 应在 Top20 明显为正。实际 2026 `due5 Top20` 分别为 `-0.004055` 和 `-0.003961`，说明事件关注度和净买额更可能同步带来追高/拥挤风险。

第二反事实：如果机构文本能过滤游资噪声，`institution_text_buy` 应改善主口径。实际它在结果排序中低于净买额类信号，`due5::institution_text_buy::top20` 为负且命中率更低，不构成可继续扩展的证据。

第三反事实：如果问题只是 Top20 太分散，Top15 应显著变厚。实际 Top15 最好只有 `+0.000072`，并且 raw5 为负；这不是收益弹性，只是相对全市场小幅少亏。

第四反事实：如果 2026 春节附近数据缺口污染结果，全样本和 due5 的方向不应一致。实际 `all::active_net_buy::top20` 也为 `-0.005010`，说明结论不是单个空周造成。

第五反事实：如果外部事件源天然比路径特征更接近右尾，至少 forward 第一关应接近 Exp40。实际差距超过 160bp/5d label，远远不足以覆盖进入五年长拉取和 formal account 的成本。

### 109.5 判定

`rejected`。

龙虎榜作为外部事件源没有自然生成合规一周 Top20 厚 alpha。更像是高关注/高换手事件后的拥挤风险，或者只适合更短周期、盘口、席位明细和执行约束完全不同的交易方式；在当前日线 T+1 open 到 T+6 open 框架下，不能作为新的主候选土壤。

后续不继续调龙虎榜阈值、TopK 或净买额公式，也不拉 2021-2025 长历史做成本高的确认。若未来重新使用龙虎榜，必须是带点时席位明细、事件类型和更细执行模型的新数据假设，而不是复用本轮粗粒度事件字段。

## 110. margin_north_regime_v1

### 110.1 假设

Exp109 之后继续寻找 point-in-time 外部数据。股票级融资融券、北向持股和个股资金流接口在当前环境下要么 SSL 失败，要么返回 `result=None`，不能作为严格横截面回测输入；但市场级融资融券长历史和北向资金历史可以稳定获取。

本轮不把市场级资金数据当股票排序源，而是做一个更低自由度的 regime 反事实：真实资金/杠杆状态是否能解释 Exp40 `path_sequence_ranker_v1` 的收益半衰期，从而因果选择 hold2/5/10。若成立，它应在只使用已完成历史标签的在线选择中超过固定 hold10；若退化为总选 hold10，则说明市场级外部状态只有解释力，没有动态可执行增量。

### 110.2 产物和口径

```text
/tmp/quantx-research/margin-north-regime-v1/analyze_margin_north_regime.py
sha256:9cb70e426ae61f3263d6201ca297185856d6e83d4f16afa504770c262b75ed4d

/tmp/quantx-research/margin-north-regime-v1/margin_north_regime_2021_2026_diagnostic_v2.json
sha256:e613883e7cd589763b51e999efed0a663a65f351b390b435ebfe88029b351986

/tmp/quantx-research/margin-north-regime-v1/cache/macro_margin_sh.csv
/tmp/quantx-research/margin-north-regime-v1/cache/macro_margin_sz.csv
/tmp/quantx-research/margin-north-regime-v1/cache/northbound_hist.csv
```

因果边界：

1. 输入 prediction store 为 Exp40 `path_topq::pool200::top20` 的 2022-2025 walk-forward 和 2026 forward。
2. 市场级融资融券字段来自 T 日交易所/宏观历史数据；北向资金字段来自 AkShare `stock_hsgt_hist_em(symbol="北向资金")`。
3. 标签为同一批 Exp40 TopK 股票，从 T+1 open 到 T+1+horizon open，horizon 取 2/5/10，并减同 horizon 全市场均值。
4. 在线 selector 对每个信号日只使用 `exit_date <= session` 的历史样本，避免当前调仓未来收益泄露。
5. 第一版结果不可引用：它要求所有融资/北向特征同时非空，而北向净买额字段在 2025/2026 为 NaN，导致 2025/2026 被整体过滤。`v2` 改为 overall hold 标签不依赖 regime 非空；每个 feature 只在自身非空样本中参与切片和 selector。

### 110.3 结果

修正后的 2022-2026 due5 Top20 标签显示，Exp40 的 10 日收益半衰期明显更厚：

| 口径 | 2022-2026 `mean_daily_label` | 2022 | 2023 | 2024 | 2025 | 2026 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `due5::hold2::top20` | `+0.005279` | `+0.003463` | `+0.003250` | `+0.006784` | `+0.006323` | `+0.008071` | 太薄 |
| `due5::hold5::top20` | `+0.011443` | `+0.012835` | `+0.008028` | `+0.016145` | `+0.007124` | `+0.015018` | 原一周口径，逐年全正 |
| `due5::hold10::top20` | `+0.022537` | `+0.018515` | `+0.016582` | `+0.037741` | `+0.016958` | `+0.023571` | 标签层最厚，逐年全正 |
| `all::hold10::top20` | `+0.021294` | `+0.022394` | `+0.013098` | `+0.027802` | `+0.023156` | `+0.018644` | 全日频也稳 |

但在线持有期 selector 没有提供动态增量：

| selector | 在线 `mean_daily_label` | 同样样本固定 h2 | 同样样本固定 h5 | 同样样本固定 h10 | 选择分布 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `north_hs300_ret` | `+0.024918` | `+0.005779` | `+0.011163` | `+0.024981` | h10 139 次、h5 1 次 | 略低于固定 h10 |
| `margin_balance_chg20` | `+0.022508` | `+0.005594` | `+0.009686` | `+0.022508` | h10 102 次 | 完全退化为固定 h10 |
| `margin_buy_amt_z20` | `+0.020519` | `+0.005041` | `+0.008651` | `+0.020519` | h10 106 次 | 完全退化为固定 h10 |
| `north_net_buy_ma5` | `+0.014172` | `+0.004481` | `+0.006630` | `+0.016683` | h10 49 次、h5 7 次、h2 1 次 | 明显低于固定 h10 |

特征切片有解释力但不稳定。比如 `margin_balance_chg20` 在 lowest/highest 两端都高，中间低；`margin_buy_amt_z20` 的中间分桶最高；`north_hs300_ret` 两端也偏高。它们更像对市场环境的事后分层，而不是单调可交易规则。

### 110.4 反事实分析

第一反事实：如果融资/北向状态能真正识别半衰期，在线 selector 应在同样样本上超过固定 h10。实际最佳 `north_hs300_ret` 为 `+0.024918`，略低于同样样本固定 h10 的 `+0.024981`；多数 selector 直接全选 h10。这不是动态择时能力，而是历史均值告诉它“总是持久一点”。

第二反事实：如果市场级资金状态能作为前向开关，2025/2026 不应依赖缺失字段。实际北向净买额、买入额、卖出额在 2025/2026 缺失，只能保留沪深300涨跌幅等字段；因此北向资金本身不能作为完整前向 selector。

第三反事实：如果 h10 标签厚度可以直接视为策略成功，那么账户层应该无需担心资金效率。历史正式账户反事实已经显示固定 hold10 在 2022-2025 可提高到约 6.5x，但 2026 只有约 `+11.62%`，低于固定 hold5 的 `+19%` 量级和固定 hold2 的 2026 强表现。标签层 h10 厚并不自动转化为账户复利更优。

第四反事实：如果状态切片有单调规律，high/low 分桶应该给出明确方向。实际 `margin_balance_chg20`、`margin_buy_amt_z20`、`north_hs300_ret` 都呈非单调结构，说明它们是机会/风险共振的解释变量，不是可直接门控的线性信号。

第五反事实：如果外部市场级数据可以替代股票级资金流，应该能指导横截面 Top20。实际本轮只解释同一批 Exp40 Top20 的持有期，不能生成新股票候选；而股票级融资/北向/资金流接口目前不可用或不可靠。因此它不满足“找新的鲁棒收益土壤”的要求。

### 110.5 判定

`rejected_with_signal`。

市场级融资融券和北向状态有研究价值：它们支持一个重要观察，即 Exp40 path Top20 的标签层收益在 10 日窗口更厚，说明当前一周固定持有可能没有完全吃到中期趋势。但这些外部状态不能因果识别何时该 hold2/5/10，在线 selector 退化为固定 hold10；而固定 hold10 在正式账户和 2026 资金效率上已经被证明不稳。

后续不继续调市场级融资/北向阈值，也不把它们接入策略。若未来重新研究资金流，必须拿到股票级、point-in-time、历史完整且字段可审计的数据；仅市场级资金状态最多作为诊断特征，不是新的主 alpha。

## 111. earnings_announcement_alpha_v1

### 111.1 假设

Exp109/110 说明：粗粒度龙虎榜事件不出厚收益，市场级融资/北向只能解释 regime，不能生成股票级 Top20。下一条更不同的点时数据是假设业绩公告/业绩预告能提供股票级基本面事件线索，尤其是预增、扭亏、业绩快报等公告可能触发后续一周重估。

若该方向成立，仅用公告标题和公告时间就应在 2026 forward 中形成足够稠密的 Top15/20 事件篮子，并且一周 T+1 open 到 T+6 open 超额收益至少接近 Exp40 `path_sequence_ranker_v1` 的 2026 `due5 Top20 +0.012520`。本轮只做 2026 第一关；若第一关不过，不拉 2021-2025 长历史。

### 111.2 产物和口径

```text
/tmp/quantx-research/earnings-announcement-alpha-v1/analyze_earnings_announcement_alpha.py
sha256:1e8c8003a3696df9504a71b00c211f210343f7ec83ab51a535bf499323375583

/tmp/quantx-research/earnings-announcement-alpha-v1/earnings_announcement_val63_2026_diagnostic.json
sha256:698ff9e2d96cc6b6d233b52374b8df17006aa2732c715b9b56de232d69225808
```

数据和因果边界：

1. 数据源为巨潮 `stock_zh_a_disclosure_report_cninfo`，全市场关键词拉取：`预增`、`扭亏`、`业绩快报`、`预减`、`首亏`、`续亏`。
2. 只使用公告标题、公告时间、代码和简称，不解析公告正文，不使用财务数值或未来业绩字段。
3. 公告发布时间只作为事件发生时点；入场日是公告日之后第一个交易日 open，退出是入场后第 5 个交易日 open。
4. 若公告在周末或盘后发布，统一后移到下一交易日 open；若公告日为交易日，也保守地不在当日交易。
5. 过滤 ST/退市，股票池为 `all_mainboard`。
6. 排序分数只来自标题关键词：`positive_momentum`、`turnaround_first`、`forecast_or_express`、`bad_news_contrarian`、`bad_news_avoidance`。

2026 样本：原始公告 783 条，映射到主板可交易事件 181 条，`daily_row_count=860`。

### 111.3 结果

2026 第一关没有通过：

| 方案 | 样本 | 信号日数 | 平均事件数 | 平均选中数 | `mean_daily_label5` | `mean_daily_raw5` | 命中率 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `due5::positive_momentum::top20` | due5 | 7 | 3.57 | 3.57 | `+0.006514` | `+0.022485` | `0.429` | Top20 选不满，样本太少 |
| `due5::forecast_or_express::top20` | due5 | 7 | 3.57 | 3.57 | `+0.006514` | `+0.022485` | `0.429` | 与 positive 基本同一批事件 |
| `all::positive_momentum::top20` | all | 36 | 4.08 | 4.08 | `+0.001953` | `+0.010402` | `0.500` | 全样本很薄 |
| `all::bad_news_contrarian::top20` | all | 36 | 4.08 | 4.08 | `+0.001953` | `+0.010402` | `0.500` | 标题分数无法区分 |

对照 Exp40：Exp40 2026 due5 Top20 label 约 `+0.012520`，且平均持仓约 20。本轮 due5 最好只有 `+0.006514`，且平均选中只有 3.57 只，不满足平均持仓大于 5；all 样本更只有 `+0.001953`。因此不进入 2021-2025 长拉取，不做关键词或 TopK 调参。

### 111.4 反事实分析

第一反事实：如果业绩预告/快报是足够强的股票级事件源，2026 forward 至少应形成稠密候选。实际半年只有 181 条主板可交易事件，按交易日聚合后平均每天约 4 个，Top15/20/30 全部选不满。这不满足用户要求的持仓约束。

第二反事实：如果预增、扭亏、快报能提供自然右尾，`positive_momentum` 或 `turnaround_first` 应明显优于其它标题分数。实际各类排序在 due5 上几乎完全相同，说明事件太少导致排序没有发挥空间。

第三反事实：如果 due5 的 `+0.006514` 是真实 alpha，all 样本不应降到 `+0.001953`。实际 due5 只有 7 个信号日，样本过薄，更像公告季时间切片噪声，而不是稳定候选土壤。

第四反事实：如果公告正文里有强信息，标题弱并不完全否定公告方向。但本轮目标是先验证低成本点时标题事件是否自然出收益；若需要解析正文数值和业绩上下限，工程成本和未来函数审计成本都会上升，必须先有更强的第一关证据。当前没有。

第五反事实：如果用公告事件作为独立 sleeve，可以接受低频高收益，那也与当前目标冲突：用户要求平均持仓大于 5、一周左右持有、五年逐年正。当前事件密度远远不够，不能作为主策略方向。

### 111.5 判定

`rejected`。

巨潮公告标题/公告时间是可用的股票级 point-in-time 数据，但业绩预告/快报事件在当前框架下太稀疏，且标题级分类无法形成厚 Top20。后续不继续围绕公告关键词、TopK 或公告类别调参，也不拉 2021-2025 长历史。若未来重新研究公告，需要正文解析、业绩幅度结构化和公告后流动性/涨停可交易性审计，而不是只用标题事件。

## 112. corporate_action_announcement_alpha_v1

### 112.1 假设

本轮原假设是：相比业绩预告，回购、增持、股权激励、重大合同、中标、定增、资产重组等公司动作公告可能更贴近 A 股短期资金偏好，且事件密度更高，可能在公告后一周形成可交易横截面收益。

用户随后明确更新研究范围：消息面/公告/事件流方向先不做，因为消息面本身有滞后性；后续所有外部数据若不是 QMT 能拉取的就不考虑。因此本实验只记录已经完成的 2026 初筛，不再拉取 2025 或更长历史，不进入账户层。

### 112.2 产物和口径

```text
/tmp/quantx-research/corporate-action-announcement-alpha-v1/analyze_corporate_action_announcement_alpha.py
sha256:4b74e7e1768e7ef9faf60b2104559769e535b30cae9b69767d3fcda4ec2598d6

/tmp/quantx-research/corporate-action-announcement-alpha-v1/corporate_action_val63_2026_diagnostic.json
sha256:a31b4c2a66bf8fdcce41b2c166c355aa76839edabde4ec904178f23946ed67b5
```

2026 初筛数据源仍为巨潮公告标题/时间/代码/简称，不读取正文，不使用未来收益字段；入场为公告日之后第一个交易日 open，退出为入场后第 5 个交易日 open。关键词包括：`回购`、`增持`、`减持`、`股权激励`、`员工持股`、`重大合同`、`中标`、`订单`、`定增`、`资产重组`、`收购`。

2026 样本：原始公告 11,910 条，主板可交易事件 4,660 条，`daily_row_count=3408`。其中 `员工持股` 在 2026 拉取中返回异常并按空缓存处理。

### 112.3 结果

2026 初筛有局部信号，但不足以覆盖新的研究约束：

| 方案 | 样本 | 信号日数 | 平均事件数 | 平均选中数 | `mean_daily_label5` | `mean_daily_raw5` | 命中率 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `due5::incentive_alignment::top10` | due5 | 24 | 21.38 | 7.46 | `+0.018616` | `+0.016047` | 0.500 | 局部较强，但 Top10 不满足主持仓宽度 |
| `due5::incentive_alignment::top20` | due5 | 24 | 21.38 | 12.33 | `+0.014060` | `+0.011491` | 0.500 | 略高于 Exp40 2026 due5 `+0.012520`，但未选满 Top20 |
| `due5::all_positive::top20` | due5 | 24 | 21.38 | 12.33 | `+0.012756` | `+0.010187` | 0.542 | 接近 Exp40，但优势很薄 |
| `all::incentive_alignment::top20` | all | 118 | 37.31 | 15.81 | `+0.006320` | `+0.003251` | 0.551 | 全交易日收益明显变薄 |
| `all::all_positive::top20` | all | 118 | 37.31 | 15.81 | `+0.006103` | `+0.003034` | 0.585 | 不足以视为主 alpha |

### 112.4 反事实分析

第一反事实：如果公司动作公告是自然厚 alpha，`all` 口径不应明显弱于 due5。实际 due5 Top20 到 `+0.014060`，但 all Top20 只有 `+0.006320`，说明信号很可能依赖调仓日/公告季共振，而不是稳定的日频候选土壤。

第二反事实：如果它是可推进主策略方向，应该继续拉 2025-2021 做穿越检验。但用户明确指出消息面本身有滞后性，后续只考虑 QMT 能拉取的数据。因此无论 2026 局部结果如何，该方向不再符合研究范围。

第三反事实：如果事件流能作为辅助 sleeve，也仍需外部公告数据、网络拉取和公告发布时间审计。它与后续“只用 QMT 可拉取 K 线/日内数据”的约束冲突，工程和数据依赖都不应继续扩大。

### 112.5 判定

`stopped_by_scope`。

公司动作公告事件在 2026 due5 上有一点局部信号，但消息面方向被用户明确排除，且 all 口径不厚。后续不继续公告、新闻、龙虎榜、业绩预告、公司动作事件等信息类数据；新的实验只考虑 QMT 可拉取数据，优先日线/日内 K 线里更前置的市场风险、横截面因子、行业/概念轮动和多日路径结构。

## 113. qmt_amount_flow_ranker_v1

### 113.1 假设

用户明确排除消息面/公告/事件流之后，本轮回到 QMT 可直接拉取的数据：日线 OHLCV、成交额、涨跌幅、ST 标记，以及 qlib 已有 VWAP。假设是：相比手工路径特征，成交额离散度、量价吸收、上/下跌成交占比、市场 breadth/dispersion 等资金行为结构，可能更早反映 A 股短期概念/行业轮动和横截面风险偏好，从而形成一周 Top20 厚 alpha。

### 113.2 产物和因果口径

```text
/tmp/quantx-research/qmt-amount-flow-ranker-v1/analyze_qmt_amount_flow_ranker.py
sha256:dfaaabf33c7b5dadb8d3e1a3b89983868b0c20426a66d45a0a7c6c43b05227c1

/tmp/quantx-research/qmt-amount-flow-ranker-v1/write_qmt_amount_flow_predictions.py
sha256:2528cf97429f22a766455adec679526137e70e9f4daff6ccd83dac612e8847e7
```

特征只使用 T 日收盘后可见的 QMT 日线字段和滚动统计；标签为 T+1 open 入场、T+6 open 退出的 5 日收益，减去同日全市场主板均值。walk-forward 开发期只用 `< year` 训练；2026 forward 使用 2016-2025 训练。PredictionStore 为 T 日 `after_close` 信号，账户层用标准 `decision_pipeline`，T+1 open 成交、`rebalance_interval_sessions: 5`、Top20 等权、标准费用。

冻结的 PredictionStore：

| Store | records | sessions | checksum | schema |
| --- | ---: | ---: | --- | --- |
| `qmt_amount_flow_reg_top20_dev_2021_2025_predictions.json` | 3860 | 193 | `sha256:eb3a3417a353ffdc277608fc0cd8bf6dfd7a5e5a283d633531c0167ea28a1c4b` | `qmt_amount_flow_ranker_v1:9b3590830a7c4b82` |
| `qmt_amount_flow_blend_top20_dev_2021_2025_predictions.json` | 3860 | 193 | `sha256:9f14fa0ea17fa3e448398777f13d17da824846b5945e006ebba769c5ee9b17a9` | `qmt_amount_flow_ranker_v1:c5f3e0a51a5af074` |
| `qmt_amount_flow_reg_top20_val63_2026_predictions.json` | 480 | 24 | `sha256:70eec2d9f481edfad4fb8ddb8b0dd1cd0ecbd001eba27879910cecd0f9ba2ff0` | `qmt_amount_flow_ranker_v1:9b3590830a7c4b82` |
| `qmt_amount_flow_blend_top20_val63_2026_predictions.json` | 480 | 24 | `sha256:1971852ad3fba916477e10445d6348bb1d1be7c35a28b4370900edc43d11a6de` | `qmt_amount_flow_ranker_v1:c5f3e0a51a5af074` |

### 113.3 标签层结果

扩展 walk-forward 2016-2025 中，`reg::top20` 在 2021-2025 标签层逐年为正，说明 QMT amount/flow 特征确有一部分横截面解释力：

| 年份 | `reg::top20` label5 | raw5 | positive ratio |
| --- | ---: | ---: | ---: |
| 2021 | `+0.014037` | `+0.019481` | 0.653 |
| 2022 | `+0.011604` | `+0.011937` | 0.625 |
| 2023 | `+0.006814` | `+0.007745` | 0.667 |
| 2024 | `+0.016345` | `+0.018832` | 0.755 |
| 2025 | `+0.016920` | `+0.024354` | 0.660 |
| 2026 | `+0.008549` | `+0.005980` | 0.625 |

2026 forward 中，`blend::top20` 标签更好，为 `+0.012667`、raw `+0.010098`、positive ratio `0.750`，但这是 forward 后才看到的更优 variant；不能据此替代开发期预选的 `reg` 作为唯一可信选择。

### 113.4 正式账户层结果

| 方案 | 区间 | final | 总收益 | 最大回撤 | Sharpe | 平均持仓 | 平均持有天数 | 年度结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reg::top20` | 2022-2025 | 4.67x | `+367.09%` | `-35.35%` | 1.75 | 19.02 | 9.32 | 2022 `-2.02%`、2023 `+53.58%`、2024 `+59.69%`、2025 `+90.53%` |
| `blend::top20` | 2022-2025 | 2.57x | `+157.50%` | `-31.05%` | 1.06 | 19.23 | 8.48 | 2022 `-18.47%`、2023 `+43.03%`、2024 `+53.43%`、2025 `+42.60%` |
| `reg::top20` | 2026 val63 | 1.10x | `+9.85%` | `-20.22%` | 0.77 | 18.70 | 9.15 | 2026 `+9.85%` |
| `blend::top20` | 2026 val63 | 1.13x | `+12.66%` | `-11.23%` | 1.18 | 18.70 | 8.92 | 2026 `+12.66%` |

对照 Exp40 formal rebalance5：2022-2025 为 8.77x、最大回撤 `-32.62%`、平均持仓 19.40、平均持有 9.36；2026 为 `+19.18%`、最大回撤 `-16.70%`。本轮无论开发期还是 2026 都没有超过 Exp40，且开发期 2022 为负，不满足“五年都是正收益”的门槛。

### 113.5 反事实分析

第一反事实：如果 amount/flow 是独立主 alpha，标签层 `reg` 的逐年正应转成账户层逐年正。实际账户层 2022 为 `-2.02%`，blend 更是 `-18.47%`，说明标签均值对可交易、成本、涨跌停和重叠持仓的保护不够。

第二反事实：如果 2026 的 `blend` 更强代表稳健性，开发期不应明显弱于 `reg`。实际 blend 开发期只有 2.57x，远低于 reg 的 4.67x，也低于 Exp40 8.77x；把 blend 作为主方案会变成 forward 后验选择。

第三反事实：如果成交额特征能打开收益上限，2026 至少应超过 Exp40 前向。实际最好的 blend 只有 `+12.66%`，低于 Exp40 `+19.18%`；reg 更只有 `+9.85%` 且回撤更深。

第四反事实：如果只是 TopK 或 reg/blend 权重问题，标签层会出现远高于 Exp40 的厚度作为调参空间。实际 2026 最好标签 `+0.012667` 只是贴近 Exp40 due5 `+0.012520`，开发期账户又明显落后，因此继续调 TopK、ranker 或融合权重很容易变成后验卡参数。

第五反事实：QMT amount/flow 的有效特征重要性集中在市场 `amount_disp20`、breadth、dispersion、ret20 median 等市场状态，说明它更像风险状态解释器，而不是股票级强候选生成器。后续应把这类状态用作风控/状态分层参考，主方向继续寻找更前置、更厚的候选土壤。

### 113.6 判定

`rejected_after_formal_replay`。

本轮严格符合“只用 QMT 可取数据”的约束，也通过了 T 日特征、T+1 open 执行的因果口径；但正式账户层收益不够厚，开发期不逐年为正，且 2026 低于 Exp40。后续不继续围绕 `reg/rank/blend`、TopK 或成交额阈值调参；保留的启发是：成交额离散度和市场 breadth 对 2026 风格有解释力，但需要新的股票级候选生成机制。

## 114. endogenous_theme_pca_v1

### 114.1 假设

在不使用外部行业/概念标签、也不使用公告/消息面的约束下，本轮测试 QMT 日线自身是否能抽出“内生主题”。若 A 股短周期机会沿同涨同跌的主题扩散，那么每个信号日用过去 60 日全市场收益做 SVD，应能得到类似概念/行业轮动的隐含暴露；再结合主题动量、主题 breadth、主题规模和个股相对主题强弱，应该能改善一周 Top20 横截面收益。

### 114.2 产物和因果口径

```text
/tmp/quantx-research/endogenous-theme-pca-v1/analyze_endogenous_theme_pca.py
sha256:837c20e9e780ff773ae70e51d7d59850c6c8f588d1af55be267cae96a7c6cca9

/tmp/quantx-research/endogenous-theme-pca-v1/endogenous_theme_pca_dev_2021_2025.json
sha256:d00068e0c26145127d08199e2f91f5f8e96d427da66d2b8238ee22efc9f783e8

/tmp/quantx-research/endogenous-theme-pca-v1/endogenous_theme_pca_val63_2026.json
sha256:468ba463d3ad097d68f4ac49a21f52be8a2368ad5bc415724078fba5ffef5b98
```

每个 due5 信号日 T，只使用 `[T-59, T]` 的 close-to-close 收益做横截面 SVD，抽取 12 个内生主题。标签仍为 T+1 open 入场、T+6 open 退出的 5 日收益，减同日全主板均值。开发期按年度 walk-forward，只用 `< year` 样本训练；2026 forward 使用 2016-2025 训练。未进入 PredictionStore 或正式账户层。

### 114.3 结果

| 方案 | 区间 | `mean_daily_label5` | raw5 | positive ratio | 年度/前向 |
| --- | --- | ---: | ---: | ---: | --- |
| `reg::top20` | 2022-2025 | `+0.007832` | `+0.010334` | 0.596 | 2022 `+0.005801`、2023 `+0.002585`、2024 `+0.006439`、2025 `+0.016611` |
| `reg::top15` | 2022-2025 | `+0.010602` | `+0.013105` | 0.596 | 2022 `+0.010491`、2023 `+0.003461`、2024 `+0.008433`、2025 `+0.020173` |
| `blend::top20` | 2022-2025 | `+0.003235` | `+0.005737` | 0.565 | 2023 接近 0，2025 较强 |
| `rank::top20` | 2022-2025 | `-0.000845` | `+0.001657` | 0.456 | 排序目标失败 |
| `reg::top20` | 2026 val63 | `+0.003910` | `+0.001341` | 0.583 | 明显低于 Exp40 2026 due5 `+0.012520` |
| `blend::top20` | 2026 val63 | `-0.005304` | `-0.007873` | 0.500 | forward 为负 |
| `rank::top20` | 2026 val63 | `-0.004445` | `-0.007014` | 0.417 | forward 为负 |

对照 Exp40 标签层：开发期 due5 Top20 约 `+0.014170`，2026 due5 Top20 约 `+0.012520`。本轮开发期最稳 `reg::top20` 只有 `+0.007832`，2026 只有 `+0.003910`，没有达到进入账户层的门槛。

特征重要性显示，模型确实使用了内生主题变量，例如 `theme_size_rank`、`theme_disp20_rank`、`theme_mom20_rank`、`theme_ret60_rank`、`theme_breadth20_rank` 等；但回归模型最重要的仍是 `mkt_disp20`、`mkt_ret20_median`、`mkt_breadth20`、`mkt_near_high20` 这类市场状态。

### 114.4 反事实分析

第一反事实：如果价格相关性 SVD 能自然抽出可交易概念轮动，2026 forward 不应只有 `+0.003910`，更不应出现 `blend/rank` 为负。实际说明同涨同跌结构存在，但并不等于下一周股票级收益排序。

第二反事实：如果主题特征是主 alpha，模型重要性不应主要落在市场状态上。实际回归头部仍是市场离散度、市场 20 日收益中位数和 breadth，主题变量更多像状态解释和补充。

第三反事实：如果只是 Top15/Top20 宽度问题，Top15 应明显接近 Exp40。实际开发期 `reg::top15` 为 `+0.010602`，仍低于 Exp40 Top20；2026 Top15 只有 `+0.003574`，没有厚度。

第四反事实：如果该方向值得工程优化，长样本标签层应先给出明确优势。实际全市场逐信号日 SVD 计算很重，但标签不够；不应把时间投入到增量 SVD 或性能优化上。

### 114.5 判定

`rejected_before_formal_account`。

内生主题 PCA 是一个干净的 QMT-only 反事实：不用外部概念表，也不用消息数据，只从历史收益共同运动中提取主题。但它没有打开收益上限，且 2026 明显弱于 Exp40。后续不继续调 SVD 组件数、lookback 或 rank/blend 权重；主题结构可保留为解释变量，主方向应回到已经出现过高收益证据的日频重叠 Top5/TopK 结构和正式账户复核。

## 115. qmt_path_quality_ranker_v1

### 115.1 假设

第 30、48、61、69、107 轮都说明，历史大涨股和右尾路径可以被 ML 学到，但高弹性右尾与左尾失败高度纠缠；第 113 轮又说明 QMT 成交额/资金流特征更多解释市场状态，而不是直接打开收益上限。本轮做一个更窄的 QMT-only 反事实：不再追消息、公告或外部事件，也不依赖历史分钟线，而是把训练目标从“一周终点收益”改成“终点收益 + 未来 1-5 日 open 路径可持有质量”。

如果这个思路成立，路径质量目标应该在不明显牺牲真实一周收益的前提下降低 early-failure，并让 Top20 开发期和 2026 都超过 Exp40 标签基线。

### 115.2 产物和因果口径

```text
/tmp/quantx-research/qmt-path-quality-ranker-v1/analyze_qmt_path_quality_ranker.py
sha256:8f27d9ac804ff49ae09218a512a103e40e7fad6cee1ff608055d6b65cf2db270

/tmp/quantx-research/qmt-path-quality-ranker-v1/qmt_path_quality_dev_2021_2025.json
sha256:bfb337bf716393d72627f9d30c075a9093e52405c4cff6d1f69ccf6f3eadbf59

/tmp/quantx-research/qmt-path-quality-ranker-v1/qmt_path_quality_val63_2026.json
sha256:c24342ef88bb36cdce27655116e2b53cede4d83b4cc375d75d755b6825d9cb4e
```

特征完全复用第 113 轮 QMT 日线量价面板：T 日及以前的 OHLCV、成交额、pct_chg、ST、qlib VWAP、成交额离散度、市场 breadth/dispersion、个股 RPS、VWAP 位置和量价路径。没有使用新闻、公告、龙虎榜、外部资金流或非 QMT 数据。

训练标签使用未来 T+1 到 T+6 的 open 路径，仅作为 supervised label：

```text
path_quality = label5
             + 0.45 * 同期最差 open 路径相对均值
             + 0.20 * 同期最好 open 路径相对均值
             + 0.15 * 前 2 日 open 路径相对均值
```

最终评价不看该训练标签，而仍看真实 `label5 = T+1 open -> T+6 open` 减同日全主板均值。开发期按年度 walk-forward，只用 `< year` 样本训练；2026 forward 只用 2016-2025 训练。因为标签层已明显低于 Exp40，本轮未进入 PredictionStore 或正式账户层。

### 115.3 结果

开发期 Top20：

| 方案 | `mean_daily_label5` | raw5 | mean min path | early failure | positive ratio | 逐年 label5 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ret::top20` | `+0.011025` | `+0.013528` | `-0.030181` | 29.22% | 66.84% | 2022 `+0.004079`、2023 `+0.010677`、2024 `+0.016272`、2025 `+0.013081` |
| `ret_quality_50::top20` | `+0.009903` | `+0.012405` | `-0.028128` | 26.32% | 63.73% | 2022 `+0.005665`、2023 `+0.009673`、2024 `+0.015151`、2025 `+0.009127` |
| `quality::top20` | `+0.009730` | `+0.012233` | `-0.027770` | 26.66% | 63.73% | 2022 `+0.008174`、2023 `+0.007330`、2024 `+0.013070`、2025 `+0.010398` |
| `quality_rank_50::top20` | `+0.004785` | `+0.007287` | `-0.029684` | 28.11% | 57.51% | 2022 `-0.002694`、2023 `+0.009266`、2024 `+0.005256`、2025 `+0.007217` |
| `quality_rank_70::top20` | `+0.003268` | `+0.005771` | `-0.029216` | 28.55% | 53.89% | 2022 `-0.003265`、2023 `+0.006006`、2024 `+0.004688`、2025 `+0.005587` |
| `qrank::top20` | `-0.003081` | `-0.000579` | `-0.030626` | 35.93% | 41.97% | 四年均为负 |

2026 val63 Top20：

| 方案 | `mean_daily_label5` | raw5 | mean min path | early failure | positive ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quality_rank_50::top20` | `+0.009846` | `+0.007277` | `-0.029742` | 30.21% | 75.00% |
| `ret::top20` | `+0.009598` | `+0.007029` | `-0.033629` | 28.96% | 62.50% |
| `quality_rank_70::top20` | `+0.008685` | `+0.006116` | `-0.029574` | 30.63% | 70.83% |
| `ret_quality_50::top20` | `+0.004006` | `+0.001437` | `-0.035094` | 29.17% | 50.00% |
| `quality::top20` | `+0.001273` | `-0.001296` | `-0.035269` | 29.58% | 50.00% |

对照 Exp40 标签层：开发期 due5 Top20 约 `+0.014170`，2026 due5 Top20 约 `+0.012520`。本轮开发期最好 Top20 只有 `+0.011025`，2026 最好 Top20 只有 `+0.009846`，没有达到进入账户层的门槛。

特征重要性再次集中在市场状态：开发期前列包括 `mkt_amount_disp20`、`mkt_disp20`、`mkt_breadth20`、`mkt_ret20_median`、`ret60_rank`；2026 前列包括 `mkt_disp20_q252`、`mkt_breadth20`、`mkt_disp20`、`mkt_ret20_median`、`amount_rank`、`pos20_rank` 和 `close_vwap_rank`。

### 115.4 反事实分析

第一反事实：如果路径质量目标能切开右尾和左尾，`quality` 或 `quality_rank` 应在 Top20 超过直接 `ret`。实际开发期 `quality::top20` 为 `+0.009730`，低于 `ret::top20` 的 `+0.011025`；`quality_rank_50/70` 更低，说明路径质量学习不是收益增强器。

第二反事实：如果问题只是 early failure 太高，那么降低 early failure 应带来更高真实 label。实际 `ret_quality_50` 把开发期 early failure 从 29.22% 降到 26.32%，但 label 从 `+0.011025` 降到 `+0.009903`；这和前面 right-tail/early-failure 实验一致，降低脆弱性会同步削掉收益弹性。

第三反事实：如果 2026 的路径质量排序有独立价值，开发期不应明显失败。实际 2026 `quality_rank_50::top20` 略高于 `ret::top20`，但仍低于 Exp40，且开发期 `quality_rank_50` 在 2022 为负，不能作为 forward 后验选择。

第四反事实：如果该方向值得进入 formal account，标签层 Top20 至少应超过 Exp40 或给出低相关高厚度证据。实际两端都低于 Exp40，因此账户层预计只会消耗时间，不进入。

### 115.5 判定

`rejected_before_formal_account`。

本轮严格遵守 QMT-only 和 T 日特征、T+1 open 标签的因果口径，也直接测试了“可持有路径质量”这个看似更贴近账户的问题。但结果再次支持一个核心判断：A 股短周期强势样本的右尾收益和路径脆弱性纠缠很深，路径风险惩罚更像防守过滤器，不是收益主引擎。后续不继续调 path_quality 权重、rank/reg 融合或 TopK；下一步要离开“风险惩罚型标签”，寻找新的可交易右尾候选土壤或真正低相关 sleeve。

## 116. intraday_proxy_formal_replay_v1

### 116.1 假设

第 99、100 轮用日线 OHLCV/VWAP 拆过日内/隔夜和 VWAP/成交路径，标签层开发期不够厚；但 `/tmp/quantx-research/deep-learning-alpha-search-v1/` 里仍保留了一批“日内代理”PredictionStore，轻量账户曾显示 `seal_confirmed` 开发期 170x、2026 `+19%`，`failed_touch_avoid` 开发期 119x、2026 `+18%`。这个数字足够异常，必须做强复核。

本轮不是新挖一个 alpha，而是审判一个高收益线索：如果日线封板/炸板代理确实是新的可交易右尾土壤，它应该穿过 formal overlap account 的交易规则、停牌/涨跌停、手数、成本和资金分配约束；如果穿不过，后续不能再引用轻量账户百倍结果。

### 116.2 产物和因果口径

日线代理构造脚本和预测写出脚本：

```text
/tmp/quantx-research/deep-learning-alpha-search-v1/analyze_intraday_limit_order_event_world_model_v1.py
sha256:794e1a1466827b8ea39fcd1cc6a44f9bfbd79508f31ca372002853f85aee994d

/tmp/quantx-research/deep-learning-alpha-search-v1/write_intraday_limit_order_predictions_v1.py
sha256:802411a238e796482e7fe039997f17bf6541aa62b5f1be82af025a00409c1558
```

输入特征只用 QMT/qlib 日线字段：T 日及以前 `open/high/low/close/volume/vwap`，构造 `sealed_limit_rank`、`failed_touch_low_rank`、`close_from_high_rank`、`vwap_dist_rank`、`volume_ratio20_rank`、`upper_shadow_low_rank` 等代理。没有使用新闻、公告、龙虎榜、外部事件流，也没有依赖历史分钟线。信号在 T 日收盘后生成，formal account 在 T+1 open 买入。

预测文件：

```text
/tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_seal_confirmed_pool200_top10_dev_predictions.json
sha256:e462c267af14218e04ae3c4151ba227195b7f3e52817a25950d27335a0ad7ed0

/tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_seal_confirmed_pool200_top10_val63_2026_predictions.json
sha256:54b03af351c09c2ef73709140da37e5cbcea3bf16dd05c98680aa30055254399

/tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_failed_touch_avoid_pool200_top10_dev_predictions.json
sha256:c20b9c4f68277330f6a0e52767e542f982ee54ea90846f71c2128f809144b118

/tmp/quantx-research/deep-learning-alpha-search-v1/intraday_proxy_failed_touch_avoid_pool200_top10_val63_2026_predictions.json
sha256:bd2278c598a87e5a40e4c6d92b79738430b70a0ccb24b8f8c51310c8bb3a3e0c
```

formal overlap account 复核使用现有脚本：

```text
/tmp/quantx-research/formal-overlap-account-v1/analyze_formal_overlap_account.py
```

参数：`topk=10`、`buy-count=10`、`hold-days=5`、`max-open-gap=0.25`、`min-history-days=120`、`min-entry-count=3`、slippage `0.001`、正式交易规则校验。因为是日频重叠 sleeve，平均唯一持仓会大于 5，平均 calendar holding 约 7.8-8.0 天。

formal 产物：

```text
/tmp/quantx-research/deep-learning-alpha-search-v1/formal_overlap_intraday_proxy_seal_confirmed_top10_buy10_dev_2022_2025_gap25_hist120.json
sha256:1ece3e21fc4b56c59f48ffc22dae8afbe0434b6158ac3f16360159d043d9c0c3

/tmp/quantx-research/deep-learning-alpha-search-v1/formal_overlap_intraday_proxy_seal_confirmed_top10_buy10_val63_2026_gap25_hist120.json
sha256:a11f7d1f9da8a695099f810162e5e73278f624d2a6db14c2cad956fc1479df0b

/tmp/quantx-research/deep-learning-alpha-search-v1/formal_overlap_intraday_proxy_failed_touch_avoid_top10_buy10_dev_2022_2025_gap25_hist120.json
sha256:95a4f9fc2ea5a940b73e85d79d6abac5d2797f90758dabd205dc9dccc529c227

/tmp/quantx-research/deep-learning-alpha-search-v1/formal_overlap_intraday_proxy_failed_touch_avoid_top10_buy10_val63_2026_gap25_hist120.json
sha256:1af913591f41b34acb4fc3e1f86d3e75eeee50778edbb4410670253ac0f00665
```

### 116.3 结果

轻量账户和 formal account 的差异非常大：

| 方案 | 口径 | 区间 | 最终倍数/收益 | 最大回撤 | 平均唯一持仓 | 平均持有 | 年度/前向 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `seal_confirmed` | 轻量账户 | 2021-2025 | 170.98x | `-34.5%` | 9.99 | 约一周 | 2021 `+794.9%`、2022 `+230.7%`、2023 `+77.6%`、2024 `+127.9%`、2025 `+42.7%` |
| `seal_confirmed` | 轻量账户 | 2026 | `+19.0%` | `-21.8%` | 9.92 | 约一周 | 2026 `+19.0%` |
| `seal_confirmed` | formal overlap | 2022-2025 | 3.95x / `+294.92%` | `-31.44%` | 30.63 | 7.78 天 | `+34.44%、+34.31%、+17.31%、+87.04%` |
| `seal_confirmed` | formal overlap | 2026 | 1.03x / `+2.82%` | `-19.01%` | 27.29 | 8.00 天 | 2026 `+2.82%` |
| `failed_touch_avoid` | 轻量账户 | 2021-2025 | 119.33x | `-31.9%` | 9.99 | 约一周 | 2021 `+694.9%`、2022 `+184.4%`、2023 `+76.8%`、2024 `+147.4%`、2025 `+20.7%` |
| `failed_touch_avoid` | 轻量账户 | 2026 | `+18.3%` | `-26.5%` | 9.92 | 约一周 | 2026 `+18.3%` |
| `failed_touch_avoid` | formal overlap | 2022-2025 | 4.11x / `+311.08%` | `-28.86%` | 32.50 | 7.81 天 | `+27.32%、+30.07%、+28.46%、+91.19%` |
| `failed_touch_avoid` | formal overlap | 2026 | 1.04x / `+3.62%` | `-16.23%` | 27.25 | 7.77 天 | 2026 `+3.62%` |

formal 回放里交易成本很重：`seal_confirmed` 开发期 total cost 约 `372,214`，`failed_touch_avoid` 约 `367,938`；2026 分别约 `26,430` 和 `26,678`。拒单主要来自停牌和少量跌停：开发期 `seal_confirmed` 有 `177` 次 rejected trades，`failed_touch_avoid` 有 `192` 次；2026 分别为 `37` 和 `26` 次。

对照 Exp40 formal rebalance5：2022-2025 为 8.77x、2026 为 `+19.18%`，平均持仓约 19.4，平均持有约 9.4 天。本轮 formal 虽然持仓数和持有周期都合规，逐年也为正，但收益上限只有约 4x，2026 几乎没有 alpha，明显不满足用户要求。

### 116.4 反事实分析

第一反事实：如果轻量账户百倍收益来自真实可交易 alpha，formal account 不应把 2026 从约 `+19%` 打到 `+2.82%/+3.62%`。实际差异说明轻量账户没有充分表达正式交易中的资金分配、停牌/涨跌停、手数、成本和重叠 sleeve 现金占用。

第二反事实：如果只是 `seal_confirmed` 过度追高，`failed_touch_avoid` 应在 formal 下显著修复。实际 `failed_touch_avoid` 开发期从 3.95x 到 4.11x、2026 从 `+2.82%` 到 `+3.62%`，只是小幅改善，不是收益量级跃迁。

第三反事实：如果该方向能成为低相关主 alpha，formal 开发期应至少接近或超过 Exp40，同时 2026 不应退化。实际开发期不到 Exp40 一半，2026 更只有 Exp40 的约五分之一。

第四反事实：如果日线封板/炸板代理的价值主要在 Top10 高弹性，正式账户平均唯一持仓扩到 27-32 后仍应保留收益厚度。实际扩成自然重叠账户后收益变薄，说明头部右尾不可稳定转成账户级多持仓收益。

第五反事实：如果继续调封板、炸板、VWAP 支撑权重能接近目标，至少应看到 formal 结果仍有很高上限。实际 formal 已经只有 4x 且 2026 接近无收益，继续调代理权重更像后验搜索。

### 116.5 判定

`rejected_after_formal_replay`。

本轮的价值在于排除了一个很容易误导的高收益数字。日线封板/炸板代理确实有方向性，轻量账户能跑出百倍级曲线；但一旦进入正式成交和资金约束，收益上限快速坍缩，2026 前向也明显低于 Exp40。因此后续不再引用 `intraday_proxy_light_account_v1` 的 119x/171x 作为候选证据，也不继续围绕封板/炸板/VWAP 代理调权重。

下一轮必须把正式账户约束前置到研究目标里：不是先找标签层或轻量账户高收益再补 formal，而是直接寻找在 formal overlap account 或其足够保真的代理下仍能保留右尾收益的候选土壤；否则容易反复被不可交易右尾吸引。

## 117. formal_curve_soil_diagnostic_v1

### 117.1 问题

在 Exp112-116 后，最重要的问题不是继续微调某个已有模型，而是确认这些正式账户曲线是不是来自同一块土壤。如果它们低相关，组合可能自然提高收益/降低回撤；如果高度同源，则继续做路由、拼接、权重分配很可能只是把同一个弱 alpha 重新包装。

本轮只做诊断，不生成新交易信号，不使用新闻、公告、龙虎榜、事件流或任何信息面数据。输入全部来自已有 QMT-only 或日线派生实验的正式账户日收益曲线。

### 117.2 方法

脚本：

原系统 `/tmp` 输出在环境切换后不可见，关键结果已在本文保留，并迁移为项目内摘要：

```text
.tmp/quantx-research/formal-curve-soil-diagnostic-v1/exp117_summary.md
```

纳入曲线包括：Exp40 pathseq rebalance5 audit、formal pathseq hold2/5/10、base5d Top5、4fam vote sum、head union 7model、group quality、sector follower executable、日线 intraday proxy failed/seal，以及短覆盖的 robust weekly 作为兼容性检查。

组合规则保持克制：

- 统一读取 `daily_nav.json` 或 formal JSON 的 `daily_rows`，转为日收益。
- 开发期使用 2022-2025 的共同交易日，2026 使用固定前向曲线。
- 先算 pairwise correlation。
- 再算 formal-only 和 all-no-short 的等权日再平衡组合。
- 最后只用开发期做 greedy equal-weight 和 max4 最优子集选择，然后固定组合看 2026。

这不是可交易策略，只是判断已有正式曲线组合是否值得继续。

### 117.3 结果

单曲线账户指标：

| 曲线 | 开发期 | 最大回撤 | 2026 | 2026 最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| Exp40 `pathseq_reb5_audit` | 8.77x | `-32.62%` | `+19.18%` | `-16.70%` |
| formal pathseq hold5 | 5.48x | `-37.35%` | `+19.47%` | `-17.42%` |
| formal pathseq hold2 | 3.64x | `-47.95%` | `+47.05%` | `-17.22%` |
| formal pathseq hold10 | 6.51x | `-32.25%` | `+11.62%` | `-17.45%` |
| formal base5d Top5 | 7.52x | `-33.11%` | `-6.65%` | `-20.03%` |
| formal 4fam vote sum | 6.50x | `-21.32%` | `+8.66%` | `-16.44%` |
| formal head union 7model | 5.85x | `-31.35%` | `+12.77%` | `-17.37%` |
| formal group quality | 5.93x | `-39.23%` | `+21.24%` | `-17.42%` |
| formal sector follower executable | 3.29x | `-37.88%` | `+12.98%` | `-13.60%` |
| formal intraday proxy failed touch | 4.11x | `-28.86%` | `+3.62%` | `-16.23%` |
| formal intraday proxy seal | 3.95x | `-31.44%` | `+2.82%` | `-19.01%` |
| robust weekly rebalance5 | 5.05x | `-34.88%` | `+29.71%` | `-17.38%` |

相关性很高：formal-only 45 个 pair 的平均相关为 `0.8393`，中位数 `0.8369`，最低也有 `0.7027`，最高 `0.9908`。把 Exp40 加入后的 all-no-short 平均相关仍为 `0.8365`。最高相关包括 pathseq hold5 与 group quality `0.9908`、两个 intraday proxy formal 曲线 `0.9752`、pathseq hold5 与 hold10 `0.9362`。

组合检查：

| 组合 | 开发期 | 最大回撤 | 2026 | 2026 最大回撤 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| formal-only 等权 | 4.86x | `-32.06%` | `+12.96%` | `-16.09%` | 低于 Exp40，收益被摊薄 |
| all-no-short 等权 | 5.16x | `-31.94%` | `+13.56%` | `-15.85%` | 仍低于 Exp40 |
| formal-only greedy/max4 | 6.58x | `-26.27%` | `+10.42%` | `-16.22%` | 选中 hold10 + 4fam，但前向变弱 |
| all-no-short greedy/max4 | 8.77x | `-32.62%` | `+19.18%` | `-16.70%` | 直接退化为 Exp40 |

### 117.4 判定

`rejected_as_combination_path`。

已有正式账户曲线可以略微分散回撤，但不能自然打开收益上限。formal-only 等权从多个候选拼出来，开发期反而只有 4.86x，2026 也只有 `+12.96%`；开发期最优子集虽然回撤更低，但 2026 只有 `+10.42%`。一旦允许加入 Exp40，dev-only 选择直接退化为 Exp40 本身。

这说明当前候选不是缺一个组合器，而是缺新的低相关、高弹性收益来源。下一步不再围绕已有 path/base/group/intraday-proxy formal 曲线做拼接、路由或权重优化，而应回到候选生成层，寻找 QMT 行情本身可解释、可正式成交、且在 2026 不坍缩的右尾土壤。

## 118. qmt_big_winner_formal_aware_v1

### 118.1 问题

用户提出一个合理方向：能否分析历史大幅上涨股票，用 ML 做归因，再把这些归因转成候选股选择。此前 Exp82/Exp103/Exp105 等 winner/right-tail 相关实验多停留在候选池重排或标签层，本轮换成更严格的 QMT-only 与 formal-aware setting：只用日线行情本身，训练 20 日右尾、干净大涨、早期失败和一周收益目标，先检验是否自然形成一周 Top20 厚收益。

本轮不使用新闻、公告、龙虎榜、ETF、北向、融资融券或任何信息流数据。输入只包括 QMT/qlib 可得的日线 OHLCV、VWAP、raw QMT `amount/pct_chg/is_st`，以及由这些字段在 T 日及以前构造的横截面 rank、市场宽度、市场离散度、成交额离散度、突破/影线/震荡等特征。

### 118.2 方法

实验脚本和原始 JSON 产物曾位于系统 `/tmp/quantx-research/qmt-big-winner-formal-aware-v1/`。环境切换到受限 profile 后，原脚本/JSON 所在临时目录不可见，仅根据已完成 run 的终端输出恢复了项目内摘要文件。本章节依据该终端输出和补写摘要归档；后续实验目录统一改为项目内 `.tmp/quantx-research/`：

```text
.tmp/quantx-research/qmt-big-winner-formal-aware-v1/exp118_summary.md
sha256:1a89ad9b94e684ce11667f7036df159ce06c9fe138a540a1546f4d2e70d39978
```

无未来函数设定：

- 特征只使用 T 日收盘已完成的日线行情和 raw QMT 字段。
- 训练标签使用未来 T+1 open 到 T+6 open 的一周真实收益，以及 T+1 open 到 T+21 open 的右尾/大涨路径标签。
- walk-forward 每年只用 `< year` 的历史样本训练。
- 2026 forward 只用 2016-2025 训练，再预测 2026-01-01 到 2026-07-10。
- 信号含义仍是 T 日 after-close 可见，若进入账户层则只能 T+1 open 执行。

训练目标包括：

- `reg5`：直接预测 T+1 open 到 T+6 open 的一周 excess label。
- `regw/rankw`：预测/排序 20 日 winner horizon 的 excess label。
- `clean/right`：20 日干净大涨/右尾分类。
- `fail`：早期路径失败分类。
- `formal_edge/tail_formal`：把短期收益、右尾概率、winner rank 和早期失败概率组合成 formal-aware 分数。

### 118.3 结果

walk-forward 折切正常：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 141,278 | 45 |

开发期 Top20 最好结果不是右尾模型，而是短期 `reg5`：

| 变体 | Top20 一周 excess label | 20 日 winner label | positive ratio | early failure rate |
| --- | ---: | ---: | ---: | ---: |
| `reg5` | `+0.010593` | `+0.030627` | 0.6737 | 0.3008 |
| `clean` | `+0.009223` | `+0.023204` | 0.5947 | 0.3442 |
| `right` | `+0.008393` | `+0.019573` | 0.6263 | 0.4545 |
| `tail_formal` | `+0.007194` | `+0.023256` | 0.6842 | 0.1813 |
| `regw` | `+0.006960` | `+0.020543` | 0.6368 | 0.2171 |
| `formal_edge` | `+0.005957` | `+0.018942` | 0.6474 | 0.1608 |
| `rankw` | `+0.003921` | `+0.009674` | 0.5842 | 0.2224 |

2026 forward 同样没有突破：

| 变体 | Top20 一周 excess label | 备注 |
| --- | ---: | --- |
| `reg5` | `+0.008852` | 最好 Top20，但低于 Exp40 2026 due5 Top20 `+0.012520` |
| `regw` | `+0.007676` | 20 日回归目标不如短期目标 |
| `formal_edge` | `+0.002926` | 降失败率但收益变薄 |
| `tail_formal` | `-0.000873` | 右尾 formal-aware 组合前向失效 |
| `rankw` | `-0.003535` | 20 日 rank 目标前向为负 |

2026 `reg5::top15` 有 `+0.014853`，但不采用为方向证据：第一，Top20 已降到 `+0.008852`；第二，用户要求不是靠缩 TopK 或尾部调权来卡收益；第三，开发期 Top20 本身也低于 Exp40 开发期 due5 Top20 `+0.014170`。

特征重要性显示，模型主要依赖市场状态和流动性离散度：`mkt_amount_disp20`、`mkt_disp20`、`mkt_breadth20`、`mkt_ret20_median`、`limit_like_rank_bw`、`amount_rank` 等。这说明 winner 归因更像在解释市场状态/成交离散度，而不是发现新的独立右尾发动机。

### 118.4 反事实分析

第一反事实：如果历史大涨股归因能自然转成一周 alpha，那么右尾/干净大涨目标应在 Top20 上超过短期 `reg5`，尤其应提高 2026。实际最强始终是 `reg5`，右尾目标开发期和 2026 都更低。

第二反事实：如果 formal-aware 早期失败惩罚是关键缺口，那么 `formal_edge` 或 `tail_formal` 应在降低失败率同时维持收益厚度。实际它们把 early failure rate 从 `reg5` 的约 0.30 降到 0.16-0.18，但 Top20 一周 label 也同步降到 `+0.005957/+0.007194`，收益弹性被削掉。

第三反事实：如果只是 2026 风格特殊，开发期应至少明显超过 Exp40。实际开发期最佳 Top20 `+0.010593` 低于 Exp40 due5 Top20 `+0.014170`，不是前向偶然失败。

第四反事实：如果右尾归因是低相关新土壤，20 日 winner label 高的变体至少应该在一周 label 上不显著变薄。实际 `regw/rankw` 有一定 winner label，但一周 label 很弱，说明 20 日大涨结构与一周可执行账户收益错配。

### 118.5 判定

`rejected_before_formal_account`。

这轮实验回答了“大涨股归因能不能做 ML”的问题：可以学习到部分历史 winner 结构，但不能稳定转成一周 Top20 厚收益，也没有超过 Exp40 的标签层基线。因此不写 PredictionStore、不进入 formal overlap account。

下一轮不继续优化大涨股/winner 标签，而转向更直接的 QMT-only 正式账户 alpha 来源：市场状态条件下的横截面因子时序选择、行业/概念轮动的可执行代理、或 QMT 可拉取的近期 intraday 与日线状态交互。重点应从“预测哪些股票未来 20 日会大涨”改成“在当前市场状态下，哪类一周横截面因子能自然产生账户级厚收益”。

## 119. qmt_factor_expert_timing_v1

### 119.1 问题

Exp118 的特征重要性显示，winner 归因主要落在市场状态和流动性离散度上，而不是稳定的大涨股结构。因此本轮先回到更底层的问题：如果把 QMT 日线横截面因子直接当作专家，是否存在某个专家或因果专家选择器，在 2022-2026 一周 Top20 上天然有厚收益？

本轮只用 QMT/qlib 日线 OHLCV、VWAP、raw QMT `amount/pct_chg/is_st`，不使用任何信息面、事件流或非 QMT 数据。所有产物统一放在项目内 `.tmp/quantx-research/`。

### 119.2 方法

产物：

```text
.tmp/quantx-research/qmt-factor-expert-timing-v1/analyze_qmt_factor_expert_timing.py
sha256:81b1b270afb3c4958e93f72ab3c97f65fbfffe6ed8bea9ea8eb9ea43ddf9c2fa

.tmp/quantx-research/qmt-factor-expert-timing-v1/qmt_factor_expert_timing_2021_2026_diagnostic.json
sha256:ebeec32bae161fcf2e42dc5c0a098d983ad82be80c6983ecb68cb7f0eaeb895a

.tmp/quantx-research/qmt-factor-expert-timing-v1/exp119_summary.md
```

面板：2021-01-04 到 2026-07-10，周频 due5，`808,676` 行，`266` 个信号日。标签为 T+1 open 到 T+6 open 的全市场 excess return。因果 selector 对每个信号日只使用 `exit_date <= session` 的历史专家表现。

专家包括 `low_vol20`、`quiet_trend`、`trend60`、`trend20_accel`、`pullback20`、`near_high20`、`breakout20`、`reversal5`、`volume_surge`、`amount_leader`、`vwap_reclaim`、`close_strength`、`range_expansion`、`liquidity_dispersion_leader`，以及 expanding/rolling best、softmax、positive blend 等因果 selector。

### 119.3 结果

Top20 按 2022-2025 开发期均值排序：

| 方案 | 2022-2025 label5 | 2026 label5 | 全期 label5 | 判读 |
| --- | ---: | ---: | ---: | --- |
| `low_vol20` | `+0.000964` | `-0.004104` | `+0.000360` | 开发期最好但极薄，2026 反向 |
| `selector_expanding_best` | `+0.000574` | `-0.004104` | `-0.001762` | 退化/跟随低波，前向无效 |
| `trend20_accel` | `-0.002185` | `-0.002037` | `-0.002394` | 全期弱 |
| `reversal5` | `-0.002251` | `-0.007072` | `-0.003301` | 反转无效 |
| `amount_leader` | `-0.005383` | `+0.007298` | `-0.004784` | 2026 有效但开发期反向 |
| `trend60` | `-0.013169` | `+0.014226` | `-0.009258` | 2026 最强，但历史系统性反向 |
| `pullback20` | `-0.018748` | `+0.006850` | `-0.014319` | 2026 局部有效，开发期大幅反向 |

所有因果 selector 都没有把 2026 风格提前识别出来：`selector_expanding_best` 2026 为 `-0.004104`，`selector_roll_best` 2026 只有 `+0.000099`，`selector_positive_blend` 2026 为 `-0.005676`。

### 119.4 反事实分析

第一反事实：如果裸 QMT 因子专家本身就是金矿，至少应有一个专家在 2022-2025 和 2026 同时为正且接近 Exp40 标签厚度。实际开发期最好的 `low_vol20` 只有 `+0.000964`，2026 还为负，远低于 Exp40。

第二反事实：如果 2026 是可因果识别的趋势/成交额 regime，历史表现 selector 应能逐步切到 `trend60/amount_leader/pullback20`。实际这些专家在 2022-2025 多数年份显著为负，selector 不能合理选择它们；2026 强只是后验风格。

第三反事实：如果问题只是单因子太极端，softmax/positive blend 应优于单因子。实际 expanding/rolling softmax 和 positive blend 均为负，说明混合会把弱专家互相稀释。

### 119.5 判定

`rejected_before_formal_account`。

本轮否定的是“单因子专家 + 因果专家选择”作为新主引擎。裸专家没有穿越厚度，2026 的趋势/成交额风格又与 2022-2025 历史相反，不能靠历史收益选择器提前识别。因此不生成 PredictionStore，不进 formal account。

下一步改为股票级非线性交互：让模型直接学习市场风险、流动性离散、趋势状态、个股路径位置和可交易性之间的联合关系，而不是先人为选一个裸专家。

## 120. qmt_factor_interaction_ranker_v1

### 120.1 问题

Exp119 证明裸 QMT 因子专家很弱，但仍有一个反事实需要验证：裸专家弱，是否只是因为人工单因子太线性？如果低波、趋势、成交额、回踩、VWAP 回收、收盘强度之间存在股票级非线性交互，LightGBM 应该能从这些专家 rank 和交互项中学出稳定一周 Top20。

本轮仍然只用 QMT/qlib 日线 OHLCV、VWAP、raw QMT `amount/pct_chg/is_st`，不使用任何信息面、事件流或非 QMT 数据。所有产物位于项目内 `.tmp/quantx-research/`。

### 120.2 方法

产物：

```text
.tmp/quantx-research/qmt-factor-interaction-ranker-v1/analyze_qmt_factor_interaction_ranker.py
sha256:739c6a4afdf70f93d644358be723a9d5b05d697d412247949d7a7918ed190ed8

.tmp/quantx-research/qmt-factor-interaction-ranker-v1/qmt_factor_interaction_ranker_2021_2026_diagnostic.json
sha256:255b064a2bb3a9cf2d3e9d3b1e062da4f135b47de099ffabf2eebd53a86afafe

.tmp/quantx-research/qmt-factor-interaction-ranker-v1/exp120_summary.md
```

复用 Exp119 的 QMT 因子专家面板，并新增交互项，例如 `low_vol20_x_trend60`、`trend60_x_amount_leader`、`pullback20_x_amount_leader`、`vwap_reclaim_x_close_strength`、`range_expansion_x_volume_surge` 等。模型包括：

- `reg`：LightGBM regression 预测一周 excess label。
- `rank`：LightGBM lambdarank 排序一周 label quintile。
- `blend`：`reg` rank 与 `rank` rank 等权。

时序约束：每年 fold 只用 `< year` 的历史样本训练。2026 fold 训练 2021-2025，预测 2026-01-01 到 2026-07-10。

### 120.3 结果

fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

Top20 结果：

| 方案 | 全折 label5 | 2022 | 2023 | 2024 | 2025 | 2026 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reg` | `+0.011047` | `+0.011148` | `+0.004674` | `+0.016537` | `+0.015814` | `+0.002805` | 开发期逐年为正，但 2026 坍缩 |
| `blend` | `+0.005957` | `-0.003172` | `+0.005052` | `+0.005788` | `+0.014858` | `+0.008332` | 2026 稍好但 2022 为负，且低于 Exp40 |
| `rank` | `+0.003226` | `-0.004384` | `-0.000230` | `+0.005963` | `+0.011473` | `+0.003189` | 排序目标明显弱 |

Top15 `reg` 全折 `+0.013653`，但 2026 只有 `+0.003908`，同样不是前向可用方向。重要特征集中在 `vwap_reclaim`、`reversal5`、`close_strength`、`range_expansion`、`quiet_trend`、`trend20_accel`、`low_vol20_x_amount_leader` 等，说明模型确实学到了短期微结构/路径交互，但前向稳定性不足。

### 120.4 反事实分析

第一反事实：如果 Exp119 的失败只是因为单因子太线性，`reg` 或 `blend` 应该在 2026 仍保持厚度。实际 `reg` 开发期很漂亮，但 2026 只有 `+0.002805`，说明它主要拟合了 2022-2025 的状态关系。

第二反事实：如果 rank 目标能提供更稳的横截面排序，`rank` 应该比回归更鲁棒。实际 `rank` 在 2022/2023 为负，2026 也只有 `+0.003189`。

第三反事实：如果 2026 的成交额/趋势风格能通过交互学习自动适应，`blend` 应至少超过 Exp40 2026。实际 `blend` 只有 `+0.008332`，低于 Exp40 due5 `+0.012520`，且 2022 为负，不满足逐年鲁棒。

### 120.5 判定

`rejected_before_formal_account`。

本轮把“手工 QMT 因子专家 + 非线性 ML 交互”这条路基本排除了：它能在开发期提取一些一周信号，但前向 2026 不稳，离几十倍至百倍账户目标很远。因此不生成 PredictionStore，不进入 formal account。

下一轮需要换更结构性的 QMT-only 方向：行业/概念轮动的可执行日线代理，或者 QMT 可拉取分钟线覆盖期内的 intraday/daily 状态交互，而不是继续在手工裸因子及其交互上加模型复杂度。

## 121. qmt_endogenous_leader_follower_v1

### 121.1 问题

Exp120 否定了“裸 QMT 因子专家 + 非线性交互”作为新主引擎，但还没有充分回答一个更贴近 A 股的问题：如果真实收益沿主题共同运动扩散，而静态行业/概念表又不可依赖，能不能只用 QMT 日线自身内生识别 leader basket，再学习 leader/follower/catch-up 关系？

本轮不使用静态行业/概念成分、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。输入只包括 QMT/qlib 日线 open/close/volume/vwap、QMT raw amount/is_st，以及由 T 日及以前历史行情构造的 leader basket、相关、β、残差和市场状态特征。

### 121.2 方法

产物：

```text
.tmp/quantx-research/qmt-endogenous-leader-follower-v1/analyze_qmt_endogenous_leader_follower.py
sha256:99d0bbe39499b1224bbefc0acf52bd0c75f16cba62be9649f67642ad57f12483

.tmp/quantx-research/qmt-endogenous-leader-follower-v1/qmt_endogenous_leader_follower_2021_2026_diagnostic.json
sha256:b96a9054b70aa61f610d8a6bc25d58ec5186b30e89a2dda171854befec16b951

.tmp/quantx-research/qmt-endogenous-leader-follower-v1/exp121_summary.md
```

无未来函数设定：每个信号日 T 只使用 T 日及以前的 QMT 日线数据。先用过去收益和成交额识别 leader basket，再用过去 60 日股票收益与 leader basket 收益计算相关、β、残差强弱和相对 leader 的 catch-up 特征。标签仍为 T+1 open 到 T+6 open 的一周 excess return。年度 fold 只用 `< year` 样本训练，2026 forward 只用 2021-2025 训练。

模型和规则包括：

- `reg`：LightGBM regression 学习一周 excess label。
- `blend`：ML 分数与部分 leader/follower 规则弱融合。
- `follower_catchup`：偏向高相关、低近期跟涨、具备补涨空间的 follower。
- `leader_pullback`：偏向 leader 暴露高、近期回撤后的主题核心股。
- `leader_continue`、`theme_beta_amount`：固定 leader 延续和主题成交额暴露规则。

构建相关/β时出现的 `nan` 警告来自部分股票过去 60 日历史为空；这些值最终被填充，不改变时序因果性，但后续同类脚本应抑制该类日志。

### 121.3 结果

fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

Top20 结果：

| 方案 | 全折 label5 | 2022 | 2023 | 2024 | 2025 | 2026 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reg` | `+0.008680` | `+0.002334` | `+0.006271` | `+0.012415` | `+0.013120` | `+0.009806` | 逐年全正，是本轮最稳形态，但低于 Exp40 |
| `blend` | `+0.006764` | `+0.005152` | `+0.000249` | `+0.007590` | `+0.011813` | `+0.011523` | 2026 接近 Exp40，但开发期偏薄 |
| `follower_catchup` | `-0.000379` | 负 | 正 | 负 | 正 | `+0.012818` | 2026 有效，开发期不稳 |
| `leader_pullback` | `-0.002606` | 负 | 负 | 负 | 负 | `+0.016447` | 2026 很强，但 2022-2025 全负 |

`leader_continue::top20` 和 `theme_beta_amount::top20` 在开发期系统性为负，说明固定追 leader 或固定追主题成交额暴露并不能穿越。

### 121.4 反事实分析

第一反事实：如果内生 leader/follower 是新的主收益土壤，`reg` 至少应在 2026 接近或超过 Exp40 due5 Top20 `+0.012520`，并在开发期有更厚收益。实际 `reg` 逐年为正但 2026 只有 `+0.009806`，开发期也不够厚。

第二反事实：如果 2026 的 leader pullback/catch-up 结构能由历史识别，固定规则在 2022-2025 不应系统性为负。实际 `leader_pullback` 2026 达 `+0.016447`，但开发期四年全负，这是典型后验风格，不可作为因果主引擎。

第三反事实：如果共同运动主题只是缺一个 ML 非线性层，`blend` 应显著超过 `reg`。实际 `blend` 2026 有所改善，但开发期变薄，说明当前融合更多是在贴近 2026 风格，而不是发现稳定机制。

第四反事实：如果不需要静态行业/概念也能抽出足够强的轮动结构，那么本轮应明显优于 Exp119/120 的裸因子与交互。实际本轮确实更稳，证明“共同运动主题”有信息，但仍没有打开收益上限。

### 121.5 判定

`rejected_before_formal_account`。

本轮得到的正面信息是：只用 QMT 日线构造的内生 leader/follower 特征，比裸因子专家更稳定，`reg` 在 2022-2026 Top20 逐年为正，说明共同运动主题不是纯噪声。

但它仍不是可合代码候选：标签厚度低于 Exp40，2026 没有前向突破；固定 follower/leader 规则虽然在 2026 局部很强，却在开发期反向。下一步若继续沿内生主题，不应继续调固定 leader/follower 公式，而应验证“市场状态识别 + 主题模式切换”：每个信号日只能使用 `exit_date <= session` 的历史表现和 T 日已知市场状态，检验是否能因果选择 `reg/blend/follower_catchup/leader_pullback` 等模式，而不是事后挑 2026 最强规则。

## 122. qmt_theme_regime_router_v1

### 122.1 问题

Exp121 证明内生 leader/follower 的 `reg` 版本有稳定但不够厚的信号，同时 `leader_pullback` 和 `follower_catchup` 在 2026 局部很强但开发期反向。关键反事实是：这是否说明主题模式空间本身有高上限，只是需要一个因果市场状态识别器来决定当天用哪种模式？

本轮不再调固定公式，而是把 Exp121 的主题模式当作可选专家，严格测试路由器能否只用已完成历史表现和 T 日市场状态，在信号日 T 因果选择 `reg/blend/follower_catchup/leader_pullback/leader_continue/theme_beta_amount`。

### 122.2 方法

产物：

```text
.tmp/quantx-research/qmt-theme-regime-router-v1/analyze_qmt_theme_regime_router.py
sha256:dc0fb3e14d7497120ec2aaa832507c3484443a62bd51805f6b5d2787ec92e1c8

.tmp/quantx-research/qmt-theme-regime-router-v1/qmt_theme_regime_router_2021_2026_diagnostic.json
sha256:3f1289b864f2de6d35764a4fa28bae1041648beb49fac996ce8ef3256cbcd778

.tmp/quantx-research/qmt-theme-regime-router-v1/exp122_summary.md
```

无未来函数设定：基础股票分数复用 Exp121 年度 walk-forward，每年只用 `< year` 样本训练。路由器在信号日 T 只能使用 `exit_date <= T` 的历史主题模式表现，以及 T 日收盘已可见的市场状态，例如市场 5/20/60 日中位收益、宽度、离散度、成交额离散度、近 20 日新高比例、leader basket 相对市场强弱等。

路由器包括：

- `expanding_best`：只用所有已完成历史信号日，选择历史均值最高模式。
- `rolling20_best`：只用最近 20 个已完成历史信号日选择均值最高模式。
- `ewma20_best`：对已完成历史模式表现做半衰期 20 的 EWMA。
- `meta_lgbm`：用历史已完成信号日的市场状态和模式 one-hot 预测模式收益，在 T 日对各模式打分。
- `oracle_noncausal`：当天事后选择收益最高模式，只作为不可交易上限诊断。

### 122.3 结果

Top20 路由结果：

| 路由器 | 全折 label5 | 2022 | 2023 | 2024 | 2025 | 2026 | 模式使用摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `oracle_noncausal` | `+0.038447` | `+0.030527` | `+0.028924` | `+0.041320` | `+0.041776` | `+0.062177` | 六种模式均被使用，不可交易上限 |
| `ewma20_best` | `+0.009091` | `+0.004212` | `+0.005189` | `+0.014117` | `+0.012513` | `+0.009806` | 2026 全部选 `reg` |
| `expanding_best` | `+0.008925` | `+0.007184` | `+0.002604` | `+0.012415` | `+0.013120` | `+0.009806` | 2026 全部选 `reg` |
| `meta_lgbm` | `+0.008779` | `+0.007184` | `+0.002604` | `+0.011543` | `+0.012172` | `+0.012267` | 2026 选 `reg` 20 次、`blend` 3 次 |
| `rolling20_best` | `+0.008575` | `+0.006608` | `+0.003715` | `+0.013672` | `+0.012463` | `+0.004117` | 2026 切错，明显变弱 |

固定模式对照：

| 固定模式 | 全折 label5 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg` | `+0.008680` | `+0.002334` | `+0.006271` | `+0.012415` | `+0.013120` | `+0.009806` |
| `blend` | `+0.006764` | `+0.005152` | `+0.000249` | `+0.007590` | `+0.011813` | `+0.011523` |
| `follower_catchup` | `-0.000379` | `-0.007744` | `+0.003268` | `-0.003910` | `+0.000451` | `+0.012818` |
| `leader_pullback` | `-0.002606` | `-0.005242` | `-0.004261` | `-0.006524` | `-0.003472` | `+0.016447` |

### 122.4 反事实分析

第一反事实：如果主题模式空间没有价值，非因果 oracle 也不会明显高。实际 oracle 全折 `+0.038447`，2026 `+0.062177`，说明每天不同主题模式之间确实存在巨大收益差异。

第二反事实：如果历史表现能够承接风格，`expanding/rolling/EWMA` 应该在 2026 切到 `leader_pullback` 或 `follower_catchup`。实际 expanding 和 EWMA 都退化为 `reg`，rolling 在 2026 降到 `+0.004117`，说明历史收益排序不能识别 2026 后验强模式。

第三反事实：如果 T 日市场状态足以识别主题模式，`meta_lgbm` 应显著高于固定 `reg`，并超过 Exp40 2026 due5 Top20 `+0.012520`。实际 `meta_lgbm` 2026 为 `+0.012267`，略低于 Exp40，且全折 `+0.008779` 仅与固定 `reg` 相近。

第四反事实：如果继续调阈值就能出鲁棒收益，那么最强 2026 模式不应在开发期系统性为负。实际 `leader_pullback` 2022-2025 全负，`follower_catchup` 也有 2022/2024 为负，因此沿这个方向调阈值会变成追 2026 后验风格。

### 122.5 判定

`rejected_before_formal_account`。

这轮保留一个重要观察：内生主题模式的 oracle 上限很高，A 股短周期确实有强烈的动态主题机制；但当前用历史已完成表现和日线市场状态做因果路由，还不能稳定提前识别该机制。最好的 `meta_lgbm` 只是在 2026 接近 Exp40 标签层，开发期和全期都不够厚，因此不生成 PredictionStore，不进入 formal account。

下一步不继续在这批模式上调路由阈值，而应转向更底层的新候选土壤：直接在股票级刻画“拥挤反转、市场一致性瓦解、右尾可买化、多 horizon 路径状态”的结构，让模型生成候选，而不是先限定在 Exp121 的几个主题公式里做选择。

## 123. qmt_robust_router_selection_v1

### 123.1 问题

Exp122 说明内生主题模式有很高非因果上限，但因果路由没有稳定抓住。与此同时，项目 `.tmp` 中已有多 horizon、逆拥挤和 supervised router 快速扫描，其中部分训练/开发期结果高达 40x 以上，但 2026 前向转负。本轮不再新增候选公式，而是问一个反过拟合问题：如果把这些状态路由候选放到更严格的鲁棒筛选器里，并修正重叠持仓的历史可见性，它们是否还能留下能穿越 2026 的结构？

### 123.2 方法

产物：

```text
.tmp/quantx-research/qmt-robust-router-selection-v1/analyze_qmt_robust_router_selection.py
sha256:c9826aadab4ba7c859c1a703bf6ffdc8d6b1c7cf1aa7dd6a0887f07393574858

.tmp/quantx-research/qmt-robust-router-selection-v1/qmt_robust_router_selection_2021_2026_diagnostic.json
sha256:6c031ba9b444b081c998fed7f9ad46490a1461fa9098299b71a84933b75a4988

.tmp/quantx-research/qmt-robust-router-selection-v1/exp123_summary.md
```

输入候选复用已有 QMT-only 多 horizon 与逆拥挤扫描中的五类 base variant：`low_vol_uptrend`、`vol_compression_breakout`、`panic_exhaustion`、`inverse_liquidity_momentum`、`quiet_pullback`。所有特征只来自 QMT/qlib 日线 OHLCV/VWAP，不使用静态行业/概念、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

本轮最重要的审计修正是历史收益可见性：路由器在信号日 T 只允许使用 `exit_date <= T` 的历史模式收益。由于交易是 T+1 open 入场、T+6 open 退出，而信号每 5 个交易日一次，紧邻上一信号日的持仓在当前信号日时通常尚未完整退出，不能作为已知收益。此前一些 rolling/router 快扫用“上一信号日收益”做历史，容易轻微乐观；Exp123 统一改为完整退出后才可见。

候选包括：

- 固定五类 base variant。
- `exit_rolling`：只用完整退出历史的 rolling best，lookback 为 4/8/12/24/36/48。
- `exit_ewma`：只用完整退出历史的 EWMA best，半衰期为 4/8/12/24/36。
- `robust_grid`：从低自由度市场状态规则中，按 2021-2024 train 的最差年、平均年收益和空仓惩罚选前 20 个。

最终选择只看 2021-2024 train 和 2025 valid 的鲁棒分数，2026 forward 不参与选择。

### 123.3 结果

候选总数 47。最佳鲁棒候选为 `exit_ewma::8::cash`：

| 区间 | final | 收益 | 最大回撤 | 平均持仓 | 逐年 | 去掉最佳 3 期后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| train 2021-2024 | 1.23x | `+23.09%` | `-20.04%` | 8.29 | `+6.47%、+3.88%、-3.32%、+15.12%` | 1.14x |
| valid 2025 | 1.75x | `+75.16%` | `-23.06%` | 9.79 | `+75.16%` | 1.33x |
| dev 2021-2025 | 1.92x | `+91.63%` | `-23.06%` | 8.59 | 2023 为负 | 1.74x |
| forward 2026 | 0.72x | `-27.52%` | `-31.75%` | 9.96 | `-27.52%` | 0.67x |

2026 最强诊断候选是 `robust_grid::00`，但它不是可选择结果，只能说明 2026 后验局部状态：

| 区间 | final | 收益 | 最大回撤 | 平均持仓 | 逐年 | 去掉最佳 3 期后 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| train 2021-2024 | 1.11x | `+11.35%` | `-33.52%` | 5.80 | `-6.84%、-8.57%、-6.21%、+39.41%` | 0.85x |
| valid 2025 | 1.48x | `+47.57%` | `-6.98%` | 8.09 | `+47.57%` | 1.21x |
| dev 2021-2025 | 1.55x | `+54.92%` | `-33.52%` | 6.27 | 2021-2023 连续为负 | 1.19x |
| forward 2026 | 1.24x | `+23.99%` | `-13.85%` | 6.12 | `+23.99%` | 1.04x |

### 123.4 反事实分析

第一反事实：如果开发期 40x 级状态路由是真正可穿越机制，经过 train/valid 最差年和去极端收益筛选后，应仍有候选在 2026 为正且开发期逐年为正。实际最佳鲁棒候选开发期只有 1.92x，2023 为负，2026 大幅亏损。

第二反事实：如果 2026 的强状态能由早年训练识别，2026 最强 `robust_grid::00` 不应在 2021-2023 连续亏损。实际它明显是后验风格，不可作为主引擎。

第三反事实：如果此前 rolling selector 的失败只是窗口不合适，加入 4/8/12/24/36/48 多窗口和 EWMA 后应有稳健改善。实际 leaderboard 前列在 2026 多数为负，说明问题不是窗口，而是历史收益无法因果识别市场机制换挡。

第四反事实：如果重叠持仓下的上一期收益可安全使用，`exit_date <= T` 修正不应改变结论方向。实际修正后鲁棒筛选更弱，说明后续所有重叠持仓路由都必须把“完整退出后才可见”作为硬约束。

### 123.5 判定

`rejected_before_formal_account`。

本轮把“候选族择时/鲁棒筛选”这条路基本否定：在严格可见性和 train/valid 鲁棒裁判下，已有多 horizon、逆拥挤和状态路由候选不能穿越 2026。下一步不继续做候选族路由，而应回到正式可成交约束下的新候选生成，优先寻找能在买入时自然避开涨停/跳空拒单、同时保留右尾收益弹性的股票级结构。

## 124. qmt_close_execution_formal_proxy_v1

### 124.1 问题

此前多轮正式 open 回放显示，简化高收益经常被 T+1 open 的涨停、跳空、停牌和拒单打掉。一个关键反事实是：如果主要瓶颈只是开盘买不到强股，那么不改变信号生成时点，只把执行价预先改成 T+1 close，应该能显著恢复收益厚度。

本轮信号仍然在 T 日收盘后生成，特征只用 T 日及以前数据；执行时点预先固定为 T+1 close，持有到 T+6 close。T+1 当天行情只用于成交拒绝和收益计算，不用于决定买什么。

### 124.2 方法

产物：

```text
.tmp/quantx-research/qmt-close-execution-formal-proxy-v1/analyze_qmt_close_execution_formal_proxy.py
sha256:7624afe0b8099bbae8d7cbfb2ab0bb677ebbc5b9eaa864b32979e4ecc5c06875

.tmp/quantx-research/qmt-close-execution-formal-proxy-v1/qmt_close_execution_formal_proxy_2021_2026_diagnostic.json
sha256:a6bad0e9870da8d1d709439d6f97581e44347351ce5c9d6467f5f8ce5bae637c

.tmp/quantx-research/qmt-close-execution-formal-proxy-v1/exp124_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。特征包括 1/3/5/10/20/60 日横截面强弱、成交额 rank、量能放大、VWAP 位置、上下影线、近 20 日新高、回撤、涨停触碰/封板历史、低波趋势、可交易动量和市场状态。

标签为 T+1 close 到 T+6 close 的 5 日收益，并加入近似可执行惩罚：T+1 close 类涨停、停牌、历史不足、ST、T+6 close 类跌停/不可卖等样本被惩罚。年度 fold 只用 `< year` 训练。

### 124.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分与 Exp121/122 一致：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_quiet_trend::top10` | 1.82x | `-15.29%` | `+34.38%` | `+36.55%` | `+20.17%` | `-2.68%` | 9.91 | 5.00 |
| `static_quiet_trend::top20` | 1.33x | `-14.11%` | `+13.71%` | `+20.04%` | `+15.49%` | `-1.94%` | 19.75 | 5.00 |
| `blend::top20` | 1.12x | `-28.85%` | `-3.92%` | `-18.13%` | `+127.89%` | `-11.80%` | 19.56 | 5.02 |
| `reg::top20` | 1.06x | `-16.11%` | `+7.02%` | `-30.06%` | `+77.87%` | `-4.81%` | 19.68 | 5.01 |
| `cls::top20` | 0.16x | `-64.56%` | `-46.68%` | `-49.29%` | `+58.02%` | `+7.17%` | 18.46 | 5.04 |

标签层同样不厚：

| 方案 | Top20 close label | Top20 exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.001092` | `-0.002250` | 98.43% | `-0.000653` |
| `blend` | `+0.001697` | `-0.002772` | 97.83% | `-0.004792` |
| `static_quiet_trend` | `+0.001277` | `-0.001695` | 98.76% | `+0.000555` |
| `cls` | `-0.006154` | `-0.014687` | 92.40% | `+0.001094` |

特征重要性仍集中在市场状态：`mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`mkt_breadth20`，以及涨停触碰、成交额和 60 日强弱等。股票级可兑现右尾没有被当前 QMT 日线特征自然抓出。

### 124.4 反事实分析

第一反事实：如果 open 买不到是主要瓶颈，T+1 close 执行应显著恢复简化高收益。实际最好只有 1.82x，且 2022/2026 为负，说明瓶颈不是单纯开盘拒单。

第二反事实：如果 close 执行能保留右尾，ML 的 `reg/blend` 应明显优于静态低波趋势。实际 `blend::top20` 只有 1.12x，且 2025 以外多年份为负；模型更多是在学阶段性市场状态。

第三反事实：如果可执行惩罚标签能解决正式损耗，exec label 应在 Top20 上稳定为正。实际 `reg/blend/static_quiet_trend` 的全期 exec label 都为负，说明可执行化后收益已经变薄。

第四反事实：如果 2026 只是 open 撮合问题，close 版 2026 应明显转强。实际 `static_quiet_trend` 2026 仍负，`cls` 虽为正但全期崩坏，不可作为穿越候选。

### 124.5 判定

`rejected_before_formal_account`。

T+1 close 执行反事实否定了“只是 open 买不到”的单因解释。换执行价并没有打开几十倍收益上限，下一步不继续研究 open/close 执行切换，而应寻找新的股票级候选生成机制：能在 T 日收盘前识别未来一周右尾，同时避免 T+1 过度拥挤和不可成交的结构。

## 125. qmt_latent_right_tail_v1

### 125.1 问题

Exp124 说明换成 T+1 close 执行并不能恢复收益厚度，瓶颈不是单纯 open 买不到。因此本轮回到 T+1 open 执行，但改变候选生成假设：不买 T 日最强，也不继续换执行价，而是寻找“右尾但不拥挤”的股票级结构。

直觉是：A 股短周期右尾可能来自已经有资金痕迹的温和趋势股，而不是当天涨停/极端追高股。如果一个股票过去 5-20 日已有成交额持续放大、VWAP 支撑、趋势平滑、近高但上影和跳空风险不高，它可能在下一周继续释放收益，同时 T+1 open 可成交性更好。

### 125.2 方法

产物：

```text
.tmp/quantx-research/qmt-latent-right-tail-v1/analyze_qmt_latent_right_tail.py
sha256:86ce2dc1655b0ae80f3de98be89e6616059579d300ba637ebc61820feda7df93

.tmp/quantx-research/qmt-latent-right-tail-v1/qmt_latent_right_tail_2021_2026_diagnostic.json
sha256:90e30813ce20f7308c01e6d71a69bcb3578da477d1b13d214ca9ae967c9e1952

.tmp/quantx-research/qmt-latent-right-tail-v1/exp125_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：特征只使用 T 日收盘及以前数据；信号在 T 日收盘后生成；执行和标签为预先固定的 T+1 open 到 T+6 open。T+1/T+6 行情只用于标签实现和账户拒绝，不用于候选选择。年度 fold 只用 `< year` 样本训练。

特征围绕三类手工结构，并交给 LightGBM 学习：

- `quiet_accumulation`：平滑 20 日趋势、成交额持续、VWAP 支撑、低波、低上影、近期少封板。
- `latent_right_tail`：20 日趋势、温和 5 日收益、量能放大、VWAP 位置、近高、低涨停拥挤、低跳空风险。
- `noncrowded_trend`：中期趋势改善、低振幅、低上影、少封板、不过度回撤、成交额确认。

模型包括 `reg`、`cls`、`blend`，以及三类静态手工分数。账户代理仍为 Top10/Top20、T+1 open 入场、T+6 open 退出，遇到涨停/停牌拒买、跌停/停牌延迟卖出，扣粗略往返成本。

### 125.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top20` | 1.69x | `-1.19%` | `+4.75%` | `-4.55%` | `+81.24%` | `-5.70%` | 19.94 | 5.01 |
| `blend::top10` | 1.58x | `-42.93%` | `+44.53%` | `+8.08%` | `+63.97%` | `+8.33%` | 9.96 | 5.01 |
| `reg::top10` | 1.54x | `-20.74%` | `+10.51%` | `+7.47%` | `+84.63%` | `-11.65%` | 9.95 | 5.01 |
| `quiet_accumulation::top20` | 1.31x | `-1.48%` | `+17.77%` | `+1.10%` | `+23.90%` | `-9.54%` | 19.79 | 5.01 |
| `blend::top20` | 1.24x | `-28.08%` | `+19.59%` | `-12.97%` | `+66.26%` | `-0.14%` | 19.90 | 5.01 |
| `latent_static::top20` | 0.91x | `-25.81%` | `-7.06%` | `-7.24%` | `+33.85%` | `+5.88%` | 19.88 | 5.01 |

标签层：

| 方案 | Top20 open label | Top20 exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.002831` | `+0.000607` | 99.77% | `-0.001156` |
| `blend` | `+0.001678` | `-0.000468` | 99.59% | `+0.000958` |
| `quiet_accumulation` | `+0.001105` | `-0.001907` | 99.06% | `-0.002899` |
| `latent_static` | `-0.000440` | `-0.003084` | 99.54% | `+0.004807` |

`blend::top10` 的 2026 exec label 为 `+0.004629`，但 2022 账户亏损 `-42.93%`，不能作为缩 TopK 后验证据。

特征重要性仍集中在市场状态和通用流动性：`mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`amount_rank`、`limit_close5_low_rank`、`limit_touch20_low_rank`、`vwap_support5_rank` 等。模型没有稳定抽出一个股票级右尾发动机。

### 125.4 反事实分析

第一反事实：如果“右尾但不拥挤”是缺失的可交易主土壤，Top20 应在 2022-2026 至少逐年为正，并显著超过 Exp40 标签/账户基线。实际最好 `reg::top20` 只有 1.69x，2022、2024、2026 都为负。

第二反事实：如果此前失败主要来自入场不可买，本轮 entry ok 接近 99% 后应恢复收益。实际收益仍薄，说明可买不等于有厚右尾。

第三反事实：如果静态手工机制方向正确，`latent_static/quiet_accumulation/noncrowded_trend` 至少应有一个穿越。实际要么 2026 正但开发期长期负，要么开发期略正但 2026 负。

第四反事实：如果 `blend::top10` 的 2026 正收益代表真实机制，那么 Top20 不应坍缩、2022 不应大亏。实际它明显是窄 TopK 和年度后验现象，不符合用户要求。

### 125.5 判定

`rejected_before_formal_account`。

本轮否定的是 QMT 日线“温和趋势 + 资金痕迹 + 低拥挤”的手工机制特征作为新主引擎。它改善了可买性，但没有打开收益上限。后续不继续微调温和趋势、上影线、量能持续、跳空风险或 VWAP 权重。

下一轮应换到更不同的假设：要么寻找更强的动态 group/相关簇事件定义，而不是股票孤立形态；要么把 QMT 可用分钟线作为 2026 以后前向辅助，但不能把短覆盖分钟线当五年主证据。

## 126. qmt_dynamic_corr_cluster_v1

### 126.1 问题

Exp121/122 显示内生主题模式存在非因果 oracle 上限，但固定 leader/follower 与市场状态路由不能稳定因果识别。Exp125 又否定了孤立股票日线“温和右尾/不拥挤”形态。于是本轮回到 A 股强主题轮动假设，但不使用静态行业/概念表：每个信号日 T 只用过去 60 日共同运动，把股票临时归到强势 anchor leader 周围，构造动态相关簇。

核心反事实：如果真正的短周期收益沿临时主题簇扩散，那么簇内 leader 延续、follower catch-up 或 lag repair 应该比孤立股票形态更厚，并且能在 2026 前向中保住。

### 126.2 方法

产物：

```text
.tmp/quantx-research/qmt-dynamic-corr-cluster-v1/analyze_qmt_dynamic_corr_cluster.py
sha256:fc780c049006558d12ca1485d1165b1e651800a1836a456d8f12b5d50968cca9

.tmp/quantx-research/qmt-dynamic-corr-cluster-v1/qmt_dynamic_corr_cluster_2021_2026_diagnostic.json
sha256:11c2fbcae670242aa8314d0c90fcff3f858d6093fbefd63dd826ff227edc7b1a

.tmp/quantx-research/qmt-dynamic-corr-cluster-v1/exp126_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：每个信号日 T 只用 T 日及以前的日线收益和成交额选 anchor leader，并用过去 60 日收益相关/β构造动态簇；标签和账户代理为 T+1 open 到 T+6 open。年度 fold 只用 `< year` 样本训练。

具体做法：

- 每个 T 日选 40 个 anchor leader：基于 T 日已知的 20 日收益、5 日收益和成交额 rank。
- 对全市场股票计算过去 60 日收益相对 anchor 的相关与 β，把股票归到最高相关 anchor。
- 对每个动态簇计算簇大小、20 日正收益宽度、簇中位收益、簇成交额放大。
- 股票级特征包括 best corr/beta、anchor 强弱、簇质量、相对 anchor 滞后、残差强弱、leader/follower/lag repair 手工分。
- 模型包括 `reg`、`cls`、`blend`，以及 `cluster_follower`、`cluster_leader`、`lag_repair` 三类静态规则。

### 126.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top20` | 1.00x | `-35.54%` | `+12.98%` | `-11.34%` | `+63.50%` | `-5.42%` | 19.97 | 5.00 |
| `lag_repair::top20` | 0.97x | `-3.46%` | `-9.53%` | `-11.57%` | `+24.72%` | `+0.57%` | 19.87 | 5.01 |
| `reg::top10` | 0.82x | `-30.38%` | `+5.80%` | `-20.72%` | `+50.03%` | `-6.82%` | 9.99 | 5.00 |
| `lag_repair::top10` | 0.64x | `-28.40%` | `-16.80%` | `-18.84%` | `+18.08%` | `+11.48%` | 9.94 | 5.01 |
| `cluster_follower::top20` | 0.14x | `-60.46%` | `-29.43%` | `-51.11%` | `+8.99%` | `-5.86%` | 19.35 | 5.01 |
| `cluster_leader::top20` | 0.04x | `-60.64%` | `-40.46%` | `-61.55%` | `-45.60%` | `-18.07%` | 19.45 | 5.02 |

标签层：

| 方案 | Top20 open label | Top20 exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.000014` | `-0.001380` | 99.86% | `+0.000243` |
| `lag_repair` | `+0.000554` | `-0.001947` | 99.42% | `+0.002630` |
| `cluster_follower` | `-0.009161` | `-0.012384` | 96.84% | `+0.000056` |
| `cluster_leader` | `-0.012947` | `-0.018024` | 97.30% | `-0.006065` |

`lag_repair::top10` 2026 exec label `+0.007838`、账户 `+11.48%`，但全期账户只有 0.64x，2022-2024 全部大幅亏损。

特征重要性：模型仍主要依赖 `mkt_disp20`、`mkt_ret20_median`、`mkt_breadth20`、`amount_rank`、`ret60_rank` 等市场状态和通用强弱；簇特征如 `cluster_size_rank`、`cluster_ret20_rank`、`anchor_ret5_rank`、`cluster_quality` 有一定权重，但没有形成独立收益源。

### 126.4 反事实分析

第一反事实：如果动态相关簇是真主题代理，`cluster_follower/cluster_leader/lag_repair` 应至少有一个在 Top20 上逐年为正。实际 follower/leader 大幅亏损，lag repair 只在 2025/2026 局部有效。

第二反事实：如果相关簇能把 Exp122 的 oracle 主题上限转成因果信号，ML `reg/blend` 应明显超过孤立形态 Exp125。实际 `reg::top20` 约 1.00x，低于 Exp125 的 1.69x，也远低于 Exp40。

第三反事实：如果 2026 lag repair 是可穿越机制，2022-2024 不应连续亏损。实际 `lag_repair::top10/top20` 在 2022-2024 均为负，属于 2026 后验局部风格。

第四反事实：如果 anchor 数或相关窗口只是参数问题，手工簇规则不应全面为负。实际簇内直接买 leader/follower 都很差，说明问题不只是 anchor_count 或 60 日窗口，而是“收益相关聚簇”本身没有足够表达交易主题。

### 126.5 判定

`rejected_before_formal_account`。

用过去 60 日收益相关构造动态主题簇，没有把主题 oracle 上限转成可交易收益。后续不继续调 anchor 数、相关窗口或 follower/leader 权重；如果继续主题方向，需要更强的 point-in-time group 事件定义，例如横截面冲击/扩散事件，而不是简单共同运动聚簇。

## 127. qmt_cross_sectional_shock_diffusion_v1

### 127.1 问题

Exp126 否定了“过去 60 日收益相关簇”作为主题代理，但 Exp122 的非因果主题模式 oracle 仍然提示主题空间有高上限。本轮换一种更短、更事件化的 QMT-only 定义：不看静态行业/概念表，也不看新闻公告，只用 T 日已经完成的横截面量价冲击定义 shock basket，再看其它股票是否存在 co-move、滞后和修复机会。

核心反事实：如果 A 股一周收益沿当日横截面冲击扩散，那么 high co-move 但未过热的 follower、短窗 lag repair，应该比简单收益相关簇更厚，并且能在 2022-2026 穿越。

### 127.2 方法

产物：

```text
.tmp/quantx-research/qmt-cross-sectional-shock-diffusion-v1/analyze_qmt_cross_sectional_shock_diffusion.py
sha256:f61d0c6535fd4330bf46e6ac551663a28042a7b7a7e5f0bbce515ac5ad03c700

.tmp/quantx-research/qmt-cross-sectional-shock-diffusion-v1/qmt_cross_sectional_shock_diffusion_2021_2026_diagnostic.json
sha256:1a2d599340f7ef804b37f78771f3e50dc6713cb3970709aa7b1c4bd5e041ff8d

.tmp/quantx-research/qmt-cross-sectional-shock-diffusion-v1/exp127_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：信号日 T 收盘后生成信号；shock basket 和全部特征只用 T 日及以前的日线数据；标签和账户代理为预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 行情只用于实现、拒单和延迟卖出；年度 fold 只用 `< year` 样本训练。

具体做法：

- 每个 T 日用异常上涨、放量、近 20 日高位、收盘强度、突破等 rank 构造 shock score。
- 从正收益且有流动性的股票里选 20-80 只 shock stocks，形成 T 日 shock basket。
- 对全市场股票计算相对 shock basket 的 5/10/20 日相关、beta、同向性、相对滞后和残差强弱。
- 手工规则包括 `diffusion_follower`、`lag_repair`、`leader_continue`。
- 模型包括 `reg`、`cls`、`blend`，训练目标仍是一周可执行 open label。

### 127.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top10` | 1.58x | `+3.30%` | `-5.99%` | `+6.43%` | `+54.74%` | `-1.37%` | 9.97 | 5.00 |
| `reg::top20` | 1.33x | `-7.06%` | `-17.97%` | `+6.47%` | `+77.89%` | `-8.03%` | 19.93 | 5.01 |
| `blend::top20` | 1.21x | `-22.54%` | `-17.64%` | `-4.86%` | `+107.15%` | `-3.81%` | 19.88 | 5.01 |
| `diffusion_follower::top20` | 1.10x | `-27.43%` | `+3.01%` | `+22.40%` | `+13.84%` | `+5.25%` | 19.96 | 5.00 |
| `lag_repair::top20` | 1.00x | `-11.25%` | `-18.22%` | `+3.68%` | `+29.20%` | `+3.15%` | 19.95 | 5.00 |
| `leader_continue::top20` | 0.01x | `-74.69%` | `-59.27%` | `-65.07%` | `-67.64%` | `-26.69%` | 18.80 | 5.03 |

标签层 Top20：

| 方案 | open label | exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.001943` | `-0.000421` | 99.70% | `-0.001402` |
| `blend` | `+0.001535` | `-0.001158` | 99.42% | `+0.000320` |
| `diffusion_follower` | `+0.000775` | `-0.001106` | 99.84% | `+0.004309` |
| `lag_repair` | `+0.000220` | `-0.001619` | 99.79% | `+0.003506` |
| `leader_continue` | `-0.019243` | `-0.026301` | 94.08% | `-0.011536` |

`diffusion_follower` 和 `lag_repair` 在 2026 Top20 exec label 为正，但开发期被 2022/2023 拖垮；`leader_continue` 几乎全区间亏损，说明直接买冲击强度仍主要是追高风险。

### 127.4 反事实分析

第一反事实：如果当日横截面冲击是真主题扩散代理，`diffusion_follower` 或 `lag_repair` 应在 Top20 逐年为正。实际它们只在 2024-2026 局部有效，2022/2023 明显亏损。

第二反事实：如果 shock basket 比 Exp126 的收益相关簇更贴近真实主题，ML 账户应显著超过 Exp40。实际最好 `reg::top20` 只有 1.33x，2026 为负，远低于 Exp40 的 8.77x 开发期和 2026 `+19.18%`。

第三反事实：如果追 leader 是核心收益来源，`leader_continue` 不应几乎归零。实际它因为涨停/拥挤和后续回撤严重亏损。

第四反事实：如果 2026 follower/lag repair 正收益可穿越，开发期不应由 2022/2023 抵消。实际 2026 强项是后验年度风格，不能作为鲁棒方向。

### 127.5 判定

`rejected_before_formal_account`。

当日横截面冲击扩散没有把主题 oracle 上限转成可交易收益。它比简单收益相关簇更能解释 2026 局部 follower/lag repair，但开发期不穿越、Top20 不厚、账户层远低于 Exp40。后续不继续调 shock 分位、shock 数量、相关窗口或扩散权重。

## 128. qmt_risk_resilient_path_ranker_v1

### 128.1 问题

Exp124-127 连续说明：换执行价、孤立温和右尾、动态相关簇、当日冲击扩散都没有打开收益上限。另一个可能解释是：强候选并非没有 alpha，而是失败在市场风险冲击下过于脆弱。于是本轮不继续找主题事件，而是测试“组合生存质量”：过去一段时间里，在市场下跌日更抗跌、极端下跌日能活下来，同时市场上涨日还能跟涨的股票，是否能自然形成更厚的一周 Top20。

核心反事实：如果 2026 和近几年失败主要来自市场风险状态下强股脆弱，那么下跌日残差、极端日生存和上行捕获应在 Top20 上自然变厚；如果只是避险弱化版，标签会薄甚至为负。

### 128.2 方法

产物：

```text
.tmp/quantx-research/qmt-risk-resilient-path-ranker-v1/analyze_qmt_risk_resilient_path_ranker.py
sha256:c84262f32a9402b98d742bc03a79cd0b8bff0cdf4bbc8e3a155b451f3ff7ff42

.tmp/quantx-research/qmt-risk-resilient-path-ranker-v1/qmt_risk_resilient_path_ranker_2021_2026_diagnostic.json
sha256:aafd1a941eeea1050ea47e5182e11553ef9e052c56bf7675b6c4816e3dbb9208

.tmp/quantx-research/qmt-risk-resilient-path-ranker-v1/exp128_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：信号日 T 收盘后生成信号；所有市场风险路径特征只使用 T 日及以前的 20/60 日收益窗口；标签和账户代理为预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 行情只用于实现、拒单和延迟卖出；年度 fold 只用 `< year` 样本训练。

具体做法：

- 计算个股相对市场中位收益的 20/60 日 beta 和相关。
- 计算市场上涨日的个股 up capture。
- 计算市场下跌日的 beta 残差、下跌日正收益频率。
- 计算过去 60 日市场最差 5 天中的个股超额表现。
- 叠加 20/60 日趋势、近高、回撤、低波、低振幅、成交额和上影线。
- 模型包括 `reg`、`cls`、`blend`，以及 `trend_resilience`、`risk_convex_momentum`、`survival_quality` 三类静态规则。

### 128.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top10` | 0.85x | `-7.15%` | `-8.85%` | `-34.06%` | `+59.27%` | `-4.86%` | 9.95 | 5.00 |
| `reg::top20` | 0.83x | `-19.48%` | `-6.13%` | `-29.93%` | `+49.07%` | `+4.94%` | 19.94 | 5.00 |
| `blend::top20` | 0.51x | `-50.88%` | `-5.56%` | `-21.01%` | `+56.18%` | `-11.74%` | 19.90 | 5.01 |
| `survival_quality::top20` | 0.32x | `-46.03%` | `-29.67%` | `+37.06%` | `-9.94%` | `-32.70%` | 19.52 | 5.01 |
| `trend_resilience::top20` | 0.04x | `-68.01%` | `-49.32%` | `-55.74%` | `-30.27%` | `-17.14%` | 19.21 | 5.03 |
| `risk_convex_momentum::top20` | 0.01x | `-70.60%` | `-50.83%` | `-79.65%` | `-49.29%` | `-17.00%` | 18.35 | 5.04 |

标签层 Top20：

| 方案 | open label | exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `-0.000169` | `-0.002036` | 99.77% | `+0.004151` |
| `blend` | `-0.001985` | `-0.003995` | 99.61% | `-0.002820` |
| `survival_quality` | `-0.005276` | `-0.009558` | 97.76% | `-0.016371` |
| `trend_resilience` | `-0.013235` | `-0.018088` | 96.20% | `-0.005867` |
| `risk_convex_momentum` | `-0.017346` | `-0.025975` | 91.91% | `-0.006524` |

特征重要性主要落在市场状态特征：`mkt_disp20`、`mkt_worst5_60`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`。股票级风险韧性特征没有成为稳定收益来源。

### 128.4 反事实分析

第一反事实：如果强候选失败主要来自市场下跌阶段脆弱性，学习下跌日残差和极端日生存后，Top20 至少应在开发期变厚。实际 `reg::top20` 2022-2024 标签和账户均为负。

第二反事实：如果风险韧性是收益弹性层而非避险弱化层，静态 `survival_quality/trend_resilience/risk_convex_momentum` 不应大幅亏损。实际三者全部显著负收益，且 entry ok 下降，说明它们选到了拥挤、不可买或回撤后的失败强股。

第三反事实：如果 2026 `reg::top20` 小正代表可穿越机制，2022-2024 不应连续为负。实际它只是年度风格局部修复，不能作为鲁棒方向。

第四反事实：如果账户执行是主要问题，open label 应明显为正而 exec label 被打掉。实际多数方案 open label 本身为负，说明土壤本身不厚。

### 128.5 判定

`rejected_before_formal_account`。

风险韧性路径特征没有形成新的收益土壤。强股失败不只是少了抗跌/生存质量特征；这类特征直接用于 Top20 选择时，标签层已经不厚，还会引入拥挤、不可买和回撤后的失败强股。后续不继续调下跌日窗口、beta、up capture 或 survival 手工权重。

## 129. qmt_supply_demand_rebalance_v1

### 129.1 问题

Exp128 否定了“下跌抗性/上行捕获/极端日生存”作为新主引擎。另一个更贴近 A 股短线资金结构的假设是供需再平衡：一段时间内先有放量换手或压力释放，随后价格不破、缩量承接，最后再出现放量确认。这类结构不依赖消息，也不需要静态行业/概念表，只用 QMT 日线就能观察。

核心反事实：如果放量压力释放后的缩量承接是真正的可交易筹码结构，那么 `supply_absorption`、`dryup_reaccumulation` 或学习型 `reg/blend` 应在 Top20 上自然变厚，并能穿越 2022-2026。

### 129.2 方法

产物：

```text
.tmp/quantx-research/qmt-supply-demand-rebalance-v1/analyze_qmt_supply_demand_rebalance.py
sha256:42a735685f0f6f178be82aaf622a96b5ad4b233133e97d3f1a56dc0b49b4e62d

.tmp/quantx-research/qmt-supply-demand-rebalance-v1/qmt_supply_demand_rebalance_2021_2026_diagnostic.json
sha256:e82f3cde55fb4785382845cc22f87bc8385afcf96c3906c4e6c029504fd95c27

.tmp/quantx-research/qmt-supply-demand-rebalance-v1/exp129_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：信号日 T 收盘后生成信号；spike、post-spike、缩量、承接和再确认特征只使用 T 日及以前的 20 日窗口；标签和账户代理为预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 行情只用于实现、拒单和延迟卖出；年度 fold 只用 `< year` 样本训练。

具体做法：

- 在过去 20 日内找到最大成交额 spike，记录发生时间、成交额强度、spike 后收益和回撤。
- 计算 spike 后缩量、区间收敛、价格是否保持在 spike close 附近。
- 计算下跌/弱收盘日的成交额吸收和收盘强度。
- 计算上涨日成交额确认、VWAP 支撑、低上影、近高和突破。
- 手工规则包括 `supply_absorption`、`dryup_reaccumulation`、`reconfirm_breakout`、`rebalance_quality`。
- 模型包括 `reg`、`cls`、`blend`，训练目标仍是一周可执行 open label。

### 129.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend::top10` | 1.78x | `-46.22%` | `+26.42%` | `-3.17%` | `+141.55%` | `+12.17%` | 9.96 | 5.01 |
| `reg::top10` | 1.41x | `-32.28%` | `-14.97%` | `-6.68%` | `+118.87%` | `+19.97%` | 9.96 | 5.01 |
| `blend::top20` | 1.31x | `-44.10%` | `+18.20%` | `-3.82%` | `+118.74%` | `-5.43%` | 19.91 | 5.01 |
| `reg::top20` | 1.12x | `-24.26%` | `-11.69%` | `-19.83%` | `+88.17%` | `+11.19%` | 19.93 | 5.01 |
| `dryup_reaccumulation::top20` | 0.48x | `-45.59%` | `-7.57%` | `+20.82%` | `-10.15%` | `-12.95%` | 18.93 | 5.01 |
| `supply_absorption::top20` | 0.42x | `-18.38%` | `-16.34%` | `-33.37%` | `+4.81%` | `-11.47%` | 19.29 | 5.01 |
| `rebalance_quality::top20` | 0.04x | `-64.61%` | `-49.81%` | `-66.19%` | `-24.27%` | `-18.60%` | 18.34 | 5.03 |

标签层 Top20：

| 方案 | open label | exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.001076` | `-0.001152` | 99.75% | `+0.006222` |
| `blend` | `+0.002115` | `-0.000215` | 99.63% | `-0.000495` |
| `dryup_reaccumulation` | `-0.001250` | `-0.010086` | 94.75% | `-0.007983` |
| `supply_absorption` | `-0.003219` | `-0.008650` | 96.57% | `-0.005609` |
| `rebalance_quality` | `-0.011492` | `-0.023390` | 91.91% | `-0.012951` |

特征重要性仍主要落在市场状态、成交额和通用路径特征：`mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`amount_rank`、`range20_low_rank`、`ret60_rank`、`vwap_support_rank`。供需再平衡的专门特征没有成为稳定主信号。

### 129.4 反事实分析

第一反事实：如果“供需再平衡”是缺失的一周收益土壤，手工规则 `supply_absorption/dryup_reaccumulation/rebalance_quality` 至少应有一个 Top20 逐年为正。实际三者都亏损，且 entry ok 明显下降。

第二反事实：如果 ML 学到了可穿越的供需结构，`reg/blend` 不应只在 2025/2026 变强。实际 2022 大幅负，2024 也弱，说明它更像顺风市场下的成交额/趋势共振。

第三反事实：如果问题主要是账户执行，open label 应显著为正而 exec label 被打掉。实际 Top20 open label 也很薄，exec 后转负，不能支撑 formal 复核。

第四反事实：如果缩量承接是真正 alpha，`dryup_reaccumulation` 不应在 2026 负收益。实际 2026 账户 `-12.95%`，说明缩量更多是流动性不足或弱势延续，而不是可靠承接。

### 129.5 判定

`rejected_before_formal_account`。

QMT 日线供需再平衡没有形成可穿越的一周收益源。它给出一个弱线索：成交额/供需路径在 2025/2026 顺风时有一定解释力，但手工供需规则为负、Top20 不穿越、2022 大幅亏损。后续不继续调 spike 窗口、缩量承接、再确认或供需手工权重；若复用，只能作为市场状态/成交额路径条件，而不是主引擎。

## 130. qmt_style_flow_rotation_v1

### 130.1 问题

Exp129 否定了“供需再平衡”作为独立主引擎，但仍留下一个线索：市场状态、成交额离散度和中期强弱反复成为模型重要特征。与其继续调单股票形态，一个更贴近横截面资金迁移的问题是：每天市场资金偏好的风格桶是否在轮动，个股是否能从所在风格桶的强弱、宽度、成交额放大和桶内相对滞后中获得一周可交易收益。

核心反事实：如果横截面风格资金迁移是缺失主引擎，`reg/blend` Top20 应在 2022-2026 逐年为正，并明显超过 Exp40；如果 2026 的风格滞后修复是真穿越机制，`style_lag_catchup` 不应在 2022-2024 连续为负。

### 130.2 方法

产物：

```text
.tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py
sha256:2a83c5b438df6b9c8c109ee42668a7592ba792bfff2ef38639ecacd48ab8a89c

.tmp/quantx-research/qmt-style-flow-rotation-v1/qmt_style_flow_rotation_2021_2026_diagnostic.json
sha256:6ca59e45cc62a2bef4bf519445f5b245d1afe37010446051766d0020a8044b34

.tmp/quantx-research/qmt-style-flow-rotation-v1/exp130_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：信号日 T 收盘后生成信号；动态风格桶和所有桶内强弱/滞后特征只使用 T 日及以前的横截面 rank 和收益/成交额窗口；标签和账户代理为预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 行情只用于实现、拒单和延迟卖出；年度 fold 只用 `< year` 样本训练。

具体做法：每天用当前横截面 rank 动态定义四类风格桶：

- 动量桶：按 20 日收益 rank 分 5 桶。
- 流动性桶：按成交额 rank 分 5 桶。
- 波动桶：按 20 日波动 rank 分 5 桶。
- 价格桶：按价格 rank 分 5 桶。

对每个桶计算桶内 5/20 日收益强度、20 日正收益宽度和成交额放大。每只股票获得所在桶的风格强度、桶内相对 5 日强弱和相对滞后。手工规则包括 `style_lag_catchup`、`style_leader_confirm`、`style_rotation_quality`。模型包括 `reg`、`cls`、`blend`。

### 130.3 结果

面板 808,676 行，scored 668,932 行，2022-2026 共 217 个信号日。fold 切分：

| 测试年 | 训练行数 | 测试行数 | 信号日 |
| --- | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 |
| 2023 | 282,808 | 149,756 | 49 |
| 2024 | 432,564 | 149,402 | 48 |
| 2025 | 581,966 | 153,904 | 49 |
| 2026 | 735,870 | 72,806 | 23 |

账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top10` | 1.86x | `+15.14%` | `+22.84%` | `-23.24%` | `+81.69%` | `-5.67%` | 9.95 | 5.01 |
| `reg::top20` | 1.66x | `+2.28%` | `+13.70%` | `-15.47%` | `+78.92%` | `-5.75%` | 19.94 | 5.01 |
| `blend::top20` | 1.53x | `-26.92%` | `+11.74%` | `+6.70%` | `+62.91%` | `+7.52%` | 19.89 | 5.01 |
| `style_rotation_quality::top20` | 1.32x | `-13.70%` | `+9.90%` | `+31.44%` | `+5.81%` | `+0.23%` | 19.94 | 5.00 |
| `style_lag_catchup::top20` | 0.69x | `-25.16%` | `-35.43%` | `-1.62%` | `+14.66%` | `+26.68%` | 19.87 | 5.00 |
| `style_leader_confirm::top20` | 0.01x | `-74.25%` | `-56.68%` | `-68.04%` | `-66.59%` | `-23.81%` | 18.64 | 5.03 |

标签层 Top20：

| 方案 | open label | exec label | entry ok | 2026 exec label |
| --- | ---: | ---: | ---: | ---: |
| `reg` | `+0.002877` | `+0.000568` | 99.77% | `-0.000014` |
| `blend` | `+0.003042` | `+0.000388` | 99.54% | `+0.004946` |
| `style_rotation_quality` | `+0.000989` | `-0.000919` | 99.75% | `+0.002212` |
| `style_lag_catchup` | `-0.001226` | `-0.003786` | 99.47% | `+0.012263` |
| `style_leader_confirm` | `-0.018768` | `-0.026173` | 93.27% | `-0.009729` |

特征重要性主要仍落在 `mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`mkt_breadth20`、`price_rank`、`amount_rank`、`ret60_rank`、`range20_low_rank` 等市场状态和通用风格特征。动态桶特征有贡献，但没有成为独立厚收益引擎。

### 130.4 反事实分析

第一反事实：如果横截面风格资金迁移是缺失主引擎，`reg/blend` Top20 应在 2022-2026 逐年为正，并明显超过 Exp40。实际 `reg::top20` 2024/2026 为负，`blend::top20` 2022 大幅负，全期只有 1.5-1.7x。

第二反事实：如果 2026 的风格滞后修复是真穿越机制，`style_lag_catchup` 不应在 2022-2024 连续负。实际它只是 2026 后验局部风格，全期账户 0.69x。

第三反事实：如果强风格 leader 是收益来源，`style_leader_confirm` 不应几乎归零。实际 leader 规则大幅亏损，说明追逐风格强股仍主要暴露于拥挤和开盘不可买。

第四反事实：如果动态风格桶提供独立 alpha，特征重要性不应继续由市场状态主导。实际模型仍主要依赖市场状态、价格、成交额和中期强弱，桶特征只是弱解释层。

### 130.5 判定

`rejected_before_formal_account`。

QMT 日线横截面风格资金迁移比孤立单股票形态略接近市场结构，但没有形成可穿越的一周收益源。账户代理最好 `reg::top10` 为 1.86x，主口径 `reg::top20` 为 1.66x 且 2024/2026 为负；`blend::top20` 2026 转正但 2022 大幅负；`style_lag_catchup` 只是 2026 后验局部风格。后续不继续调风格桶数量、leader/catchup 权重或 TopK，只把“动态风格桶能解释部分状态”作为后续诊断线索。

## 131. qmt_label_to_account_leakage_v1

### 131.1 问题

Exp130 说明横截面风格资金迁移不能直接成为主引擎，但它也暴露了一个更一般的问题：部分方案在标签层有薄正收益，到账户代理后却很难穿越年份。需要先判断瓶颈到底来自标签本身、T+1 open 拒单、T+6 open 卖出延迟、重叠调仓，还是复利路径。

核心反事实：如果 Exp130 的收益天花板主要是账户工程问题，那么入场成功率、退出成功率、持仓重叠或标签到账户周期收益之间应出现显著异常；如果这些都正常，则应停止围绕风格桶做成交修补和 TopK 调参。

### 131.2 方法

产物：

```text
.tmp/quantx-research/qmt-label-to-account-leakage-v1/analyze_qmt_label_to_account_leakage.py
sha256:def2d628738d2ad5776c37c74891da0547681e9d28b736a9ccb723d95156279b

.tmp/quantx-research/qmt-label-to-account-leakage-v1/qmt_label_to_account_leakage_2021_2026_diagnostic.json
sha256:6a0c70237b27ab89c94c4697d8c332209f8a297a4f56f60f44f630fafdcba314

.tmp/quantx-research/qmt-label-to-account-leakage-v1/exp131_summary.md
```

本轮不新增因子、不调模型、不换 TopK。脚本直接 import Exp130，重新生成同一批 QMT 日线风格资金迁移 scored 面板，然后对每个 `variant::topk` 拆解：平均原始 open 收益、excess label、可执行 excess label、账户代理周期收益、年度收益、最大回撤、入场成功率、退出成功率、平均选中数、平均持有天数、相邻信号日持仓重叠率，以及标签代理收益与实际周期收益的差异。

无未来函数设定：直接复用 Exp130 的特征和 walk-forward，所有特征只使用 T 日及以前已完成日线数据；信号在 T 日收盘后生成；诊断只评价预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 之后的数据只用于度量成交、拒单和延迟卖出，不参与信号或路由；年度 fold 只用 `< year` 样本训练。

### 131.3 结果

Exp131 复现 Exp130 的账户代理结果，说明诊断口径对齐。核心结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | mean exec excess | entry ok | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top10` | 1.86x | `+15.14%` | `+22.84%` | `-23.24%` | `+81.69%` | `-5.67%` | `+0.000812` | 99.59% | 9.95 | 5.01 |
| `reg::top20` | 1.66x | `+2.28%` | `+13.70%` | `-15.47%` | `+78.92%` | `-5.75%` | `+0.000568` | 99.77% | 19.94 | 5.01 |
| `blend::top20` | 1.53x | `-26.92%` | `+11.74%` | `+6.70%` | `+62.91%` | `+7.52%` | `+0.000388` | 99.54% | 19.89 | 5.01 |
| `style_lag_catchup::top20` | 0.69x | `-25.16%` | `-35.43%` | `-1.62%` | `+14.66%` | `+26.68%` | `-0.003786` | 99.47% | 19.87 | 5.00 |

`reg::top20` 细项：

```text
mean_label_excess:        +0.002877
mean_exec_excess:         +0.000568
mean_raw_open:            +0.005041
mean_exec_raw_proxy:      +0.002732
mean_period_return:       +0.003350
median_period_return:     +0.001100
period_hit_rate:          51.15%
period_p10/p90:           -4.34% / +4.91%
max_drawdown_period:      -32.02%
avg_overlap_prev:         14.40%
label_to_period_gap:      -0.000618
entry rejects:            10 / 4340
exit rejects:             3 / 4340
```

年度层面，`reg::top20` 的 `mean_exec_excess` 为：

```text
2022 +0.000295
2023 +0.001776
2024 -0.002649
2025 +0.003050
2026 -0.000014
```

这说明 2024 和 2026 的账户亏损已经在可执行标签层出现，不是 replay 账户额外制造的亏损。

### 131.4 反事实分析

第一反事实：如果收益漏损主要来自入场拒单，`entry_ok` 应显著低、拒单数量应集中在亏损年份。实际 `reg::top20` entry ok 为 99.77%，总拒入 10/4340，不足以解释账户收益天花板。

第二反事实：如果收益漏损主要来自卖出不可成交，`exit_ok` 或平均持有天数应明显异常。实际 `reg::top20` exit ok 为 99.56%，平均持有 5.01 天，拒出 3/4340。

第三反事实：如果重叠调仓是主问题，相邻持仓重叠率应很高并导致收益重复暴露。实际 `reg::top20` 平均重叠率约 14.40%，`blend::top20` 约 18.50%，不是主要瓶颈。

第四反事实：如果 2026 风格滞后修复可以直接拿来做主引擎，它至少应在开发期不大亏。实际 `style_lag_catchup::top20` 2026 强，但 2022/2023 大幅负，全期 0.69x。

### 131.5 判定

`diagnostic_rejected_as_main_path`。

Exp130 这类 QMT 日线动态风格桶的账户漏损主要来自“可执行标签太薄且年度翻转”，不是简单账户工程问题。后续不继续围绕 Exp130 做成交修补、TopK、重叠率或桶权重调参；下一步应寻找更厚的可执行收益土壤，或者直接学习账户可实现的更高阶路径目标。

## 132. qmt_regime_conditioned_path_family_soil_v1

### 132.1 问题

Exp127-131 反复显示，QMT-only 新土壤的问题不是成交细节，而是可执行标签太薄和年度状态翻转。横向复盘又发现，市场离散度、成交额离散度、市场宽度和市场中位收益反复成为重要变量。因此本轮不再从零造单股形态，也不训练复杂模型，而是做低自由度土壤扫描：市场风险状态是否能条件化某类股票路径族，让 Top20 在一周 T+1 open 执行下自然变厚。

核心反事实：如果“市场风险状态 × 股票路径族”是缺失的主引擎，那么至少应出现一个高激活率、平均持仓大于 5、逐年正收益、五年收益接近 Exp40 或更高的组合；如果只得到低频平滑曲线，则状态 gate 不是收益主引擎。

### 132.2 方法

产物：

```text
.tmp/quantx-research/qmt-regime-conditioned-path-family-soil-v1/analyze_qmt_regime_conditioned_path_family_soil.py
sha256:84177679ab83c8cb106ea07b3b4d147cfde8a11bfc48bc814b51425732512b1c

.tmp/quantx-research/qmt-regime-conditioned-path-family-soil-v1/qmt_regime_conditioned_path_family_soil_2021_2026_diagnostic.json
sha256:c1c72695eed53939c3262388b4140f30967f5aa4cad8eb537021f930d303c454

.tmp/quantx-research/qmt-regime-conditioned-path-family-soil-v1/exp132_summary.md
```

输入只使用 QMT/qlib 日线 OHLCV/VWAP、raw QMT `amount/is_st`。不使用静态行业/概念表、新闻、公告、龙虎榜、ETF、北向、融资融券或其它信息流数据。

无未来函数设定：复用 Exp130 的 QMT 日线面板，股票特征只使用 T 日及以前已完成日线数据；信号在 T 日收盘后生成；年度测试时，市场状态阈值只用 `< year` 的历史 session 拟合；标签和账户代理为预定 T+1 open 入场、T+6 open 目标退出；T+1/T+6 行情只用于实现、拒单和延迟卖出。

市场状态包括：`high_disp/low_disp`、`high_amount_disp`、`high_breadth/low_breadth`、`positive_market/weak_market`、`near_high_market`、`risk_on_confirmed`、`risk_off_divergent` 等。股票路径族包括：`momentum_quality`、`quiet_trend`、`pullback_in_uptrend`、`liquid_right_tail`、`anti_crowded_trend`、`lag_catchup`、`leader_confirm`、`low_price_elastic`、`risk_resilient_trend`、`short_term_reversal`。

每个年度先用历史 session 拟合市场状态三分位阈值，再在测试年选择对应状态下的路径族 Top10/Top20。未激活状态持现金。

### 132.3 结果

面板 808,676 行，scored 668,932 行。状态激活 session 数：

| 年份 | all | high_disp | high_amount_disp | high_breadth | positive_market | near_high_market | risk_on_confirmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 48 | 4 | 1 | 17 | 17 | 12 | 17 |
| 2023 | 49 | 0 | 0 | 13 | 11 | 27 | 11 |
| 2024 | 48 | 12 | 0 | 18 | 17 | 10 | 17 |
| 2025 | 49 | 5 | 8 | 27 | 24 | 24 | 24 |
| 2026 | 23 | 15 | 15 | 7 | 5 | 7 | 5 |

核心账户代理结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 | 激活率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `high_breadth::quiet_trend::top20` | 1.57x | `+5.98%` | `+9.51%` | `+12.53%` | `+17.49%` | `+2.05%` | 7.11 | 5.01 | 37.79% |
| `near_high_market::quiet_trend::top20` | 1.54x | `+8.94%` | `+7.13%` | `+3.49%` | `+22.92%` | `+3.62%` | 7.08 | 5.01 | 36.87% |
| `positive_market::quiet_trend::top20` | 1.42x | `+4.37%` | `+8.09%` | `+5.09%` | `+17.75%` | `+1.58%` | 6.40 | 5.01 | 34.10% |
| `positive_market::lag_catchup::top20` | 1.41x | `+5.77%` | `+0.31%` | `+10.08%` | `+21.00%` | `+0.08%` | 6.74 | 5.00 | 34.10% |
| `high_breadth::lag_catchup::top20` | 1.55x | `+5.77%` | `-2.85%` | `+13.44%` | `+26.98%` | `+4.94%` | 7.48 | 5.00 | 37.79% |
| `all::quiet_trend::top20` | 1.67x | `+4.13%` | `+16.93%` | `+26.89%` | `+14.70%` | `-5.57%` | 19.05 | 5.01 | 100.00% |
| `high_amount_disp::lag_catchup::top20` | 1.37x | `+2.98%` | `0.00%` | `0.00%` | `+13.33%` | `+17.76%` | 2.18 | 5.00 | 11.06% |

标签层关键观察：

```text
high_breadth::quiet_trend::top20       mean exec label -0.004104, entry ok 94.21%
near_high_market::quiet_trend::top20   mean exec label -0.002001, entry ok 96.19%
positive_market::lag_catchup::top20    mean exec label +0.000335, entry ok 99.05%
high_breadth::lag_catchup::top20       mean exec label +0.000849, 2023 为负
high_amount_disp::lag_catchup::top20   mean exec label +0.004619, 2026 强但样本稀疏且持仓不足
```

### 132.4 反事实分析

第一反事实：如果“市场风险状态 × 路径族”是缺失的主引擎，那么逐年正收益组合应接近或超过 Exp40。实际最高只有 1.57x，远低于 Exp40 的 8.77x 开发期。

第二反事实：如果状态 gate 能把 2026 高离散环境转化成鲁棒 alpha，那么 2026 强的 `high_amount_disp::lag_catchup` 不应只激活 24 个信号日、平均选中 2.18 只。实际它更像 2026 局部风格诊断，不是可交易主策略。

第三反事实：如果安静趋势路径是真正收益土壤，标签层不应为负。实际几个逐年正收益的 quiet_trend 组合 mean exec label 为负，账户正收益主要来自少出手和市场顺风，不是厚 alpha。

第四反事实：如果 2026 亏损主要由错误市场状态导致，`all::quiet_trend::top20` 经状态 gate 后应显著保留收益厚度。实际它从全激活 1.67x 变成 1.4-1.6x 的低频曲线，收益上限被削掉。

### 132.5 判定

`rejected_as_state_gate_flattens_but_not_thick`。

QMT 日线市场状态确实能解释部分风险环境，尤其 2026 的高成交额离散和 lag/catchup 结构；但它不能自然打开收益上限。后续不继续做三分位 hard gate、少出手状态过滤或 TopK 缩窄；如果继续沿市场风险方向，应让状态用于“选择更高收益候选生成机制”，而不是作为独立入场开关。

## 133. qmt_cross_sectional_payoff_surface_diagnostic_v1

### 133.1 问题

Exp132 否定了“市场状态 hard gate + 固定路径族”作为主方向，但这还没有回答更底层的问题：收益上限到底是市场机会不够，还是我们现有候选生成器离真实右尾太远。本轮因此做横截面收益机会面诊断，直接比较真实 oracle、固定路径族 oracle、随机 TopK 和 T 日可见状态对高机会 session 的预测能力。

核心反事实：如果市场没有足够右尾机会，真实 oracle Top20 不应逐年极厚；如果当前固定路径族已经覆盖主要收益结构，路径族 oracle 应接近真实 oracle；如果只要预测高机会日就能解决，T 日可见状态模型应稳定高 AUC。

### 133.2 方法

产物：

```text
.tmp/quantx-research/qmt-cross-sectional-payoff-surface-diagnostic-v1/analyze_qmt_cross_sectional_payoff_surface.py
sha256:d84cf786b7e2776c12c25a4af1c007e1ec345768feedc8608a7e946670e6f5c3

.tmp/quantx-research/qmt-cross-sectional-payoff-surface-diagnostic-v1/qmt_cross_sectional_payoff_surface_2021_2026_diagnostic.json
sha256:33fad256c95ce27008bc65901a1d58e6ac8fd20d5a04a55237c0de43d9d1834a

.tmp/quantx-research/qmt-cross-sectional-payoff-surface-diagnostic-v1/exp133_summary.md
```

本轮复用 Exp130/Exp132 的 QMT 日线面板和路径族分数。股票特征只使用 T 日及以前已完成日线数据。真实 `oracle_topk`、`best_family_oracle`、随机 TopK 只作为诊断上限，使用未来收益但不作为可交易信号。opportunity model 只使用 T 日可见的市场状态、横截面统计和路径族 score mean；已剔除 `entry_ok`、`oracle`、`random`、`raw_proxy` 等未来或结果字段。opportunity model 年度 walk-forward 只用 `< year` 的 session 训练。

对每个信号日计算：

- `oracle_top10/top20`：事后真实可执行标签最好的 TopK，用作机会上限。
- `best_family_oracle_top10/top20`：只在 10 个固定路径族中事后选择当日 raw proxy 最好的 family，用作路径族空间上限。
- `random_mean/random_p95`：同池随机 TopK 的均值和每期 p95。
- 单个固定 family 的账户 proxy。
- T 日可见状态对 `oracle_topk_raw_proxy` 的年度 walk-forward 预测能力。

### 133.3 结果

机会面：

| 口径 | Top10 平均 raw | Top20 平均 raw | 2026 Top20 raw | 正 session 比例 |
| --- | ---: | ---: | ---: | ---: |
| 真实 oracle | `+39.08%` | `+32.21%` | `+34.02%` | 100.00% |
| 路径族 oracle | `+4.02%` | `+3.08%` | `+3.74%` | 83.46% |
| 随机均值 | `+0.06%` | `+0.07%` | `-0.40%` | 约 52% |

Top20 分年：

| 年份 | 真实 oracle raw | 路径族 oracle raw | 随机均值 raw | oracle - family |
| --- | ---: | ---: | ---: | ---: |
| 2021 | `+32.57%` | `+3.65%` | `+0.30%` | `+28.92%` |
| 2022 | `+32.29%` | `+2.83%` | `-0.20%` | `+29.45%` |
| 2023 | `+27.09%` | `+2.03%` | `-0.12%` | `+25.06%` |
| 2024 | `+33.15%` | `+2.96%` | `-0.06%` | `+30.19%` |
| 2025 | `+35.13%` | `+3.60%` | `+0.62%` | `+31.54%` |
| 2026 | `+34.02%` | `+3.74%` | `-0.40%` | `+30.27%` |

账户 proxy：

| 口径 | final | 2026 | 最差年 | 最大回撤 | 平均周期收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_top20` | 极高 | 极高 | 极高 | 0.00% | `+32.21%` |
| `best_family_oracle_top20` | 2679.63x | `+129.92%` | `+164.59%` | `-7.04%` | `+3.08%` |
| `random_p95_top20` | 336.99x | `+60.12%` | `+122.56%` | `-12.54%` | `+2.26%` |
| `random_mean_top20` | 1.05x | `-9.58%` | `-11.64%` | `-37.41%` | `+0.07%` |
| `family_risk_resilient_trend_top20` | 0.94x | `-11.79%` | `-19.44%` | `-31.71%` | `+0.00%` |
| `family_lag_catchup_top20` | 0.65x | `+14.95%` | `-37.31%` | `-66.26%` | `-0.09%` |

这里的 `best_family_oracle` 和 `random_p95` 都是诊断上限，不是可交易策略。它们说明“每期选对结构”有巨大空间，但当前固定 family 不够。

机会预测：

| 目标 | 全期 Spearman reg | 全期 Spearman prob | 全期 AUC | 2026 AUC | 2026 Spearman prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top10 oracle raw | `-0.0440` | `+0.0569` | `0.5414` | `0.6587` | `+0.2174` |
| Top20 oracle raw | `+0.0724` | `+0.1074` | `0.5678` | `0.5379` | `+0.1383` |

2024 的 Top20 相关较高，但 2025 转负，2026 又偏弱，说明当前状态变量仍不能稳定提前识别机会丰厚日。

### 133.4 反事实分析

第一反事实：如果市场没有足够右尾机会，真实 oracle Top20 不应逐年极厚。实际每年都很厚，2026 也有 `+34.02%` 平均 raw，说明问题不是市场没金子。

第二反事实：如果当前固定路径族已经覆盖主要收益结构，`best_family_oracle` 应接近真实 oracle。实际 Top20 平均只到 `+3.08%`，与真实 oracle 差 `+29.13%`，说明路径族空间缺失关键结构。

第三反事实：如果只要识别高机会 session 就能解决，T 日可见状态模型应有稳定高 AUC/相关。实际 AUC 只有 0.54-0.57，且年度不稳，说明状态预测本身不够。

第四反事实：如果随机右尾就足够，随机均值应能自然赚钱。实际随机均值五年约 1.05x，2026 为负；必须有强候选生成，而不是依赖市场平均机会。

第五反事实：如果继续调现有 `quiet_trend/lag_catchup/risk_resilient` 等路径族即可，单 family 账户不应大多亏损。实际所有固定 family 都不达标，许多显著亏损。

### 133.5 判定

`diagnostic_opportunity_exists_but_current_families_do_not_capture`。

本轮不产生可合代码策略。核心价值是确定下一步方向：市场右尾机会很厚，但现有 QMT 日线固定路径族和 T 日市场状态不能捕捉。后续不继续做 hard gate、固定路径族组合、机会日预测小模型；应重新设计候选生成机制，重点寻找能更接近真实 oracle 的横截面结构，例如自适应同涨网络、短期残差扩散、跨股票相似事件匹配或更强的点时主题/行业结构表示。

## 134. qmt_adaptive_neighbor_leaf_memory_v1

### 134.1 问题

Exp133 说明全市场真实右尾机会很厚，但固定路径族和 T 日市场状态无法捕捉。Exp134 因此测试一个更自适应的候选生成器：不用手写固定 family，而是让历史样本在高维状态空间中形成近邻/叶子记忆，再用每个测试年前已经完成的历史样本估计当前股票所处状态的可执行收益均值或右尾概率。

核心反事实：如果“相似历史状态”是缺失的收益土壤，那么叶子均值、叶子右尾率或原型记忆应能在一周 T+1 open 执行下明显超过 Exp40，并且 2026 不退化。

### 134.2 方法

产物：

```text
.tmp/quantx-research/qmt-adaptive-neighbor-leaf-memory-v1/analyze_qmt_adaptive_neighbor_leaf_memory.py
sha256:e070c81a9a2555241504ada1e434a96632afc53e6cfde763940339d0778777c7

.tmp/quantx-research/qmt-adaptive-neighbor-leaf-memory-v1/qmt_adaptive_neighbor_leaf_memory_2021_2026_diagnostic.json
sha256:195d1cc911d78b6bd14844285e88b49c000343cfe5cbf502729325319409d44d

.tmp/quantx-research/qmt-adaptive-neighbor-leaf-memory-v1/exp134_summary.md
```

输入只使用 QMT/qlib 可拉取的日线 OHLCV/VWAP 和 raw `amount/is_st`，不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念表或其它信息流数据。

无未来函数设定：股票和市场特征只使用信号日 T 及以前已完成日线；信号在 T 日收盘后生成；标签为 T+1 open 到 T+6 open 的 5 日可执行收益代理；每个测试年只用 `< year` 的历史样本训练 ExtraTrees/原型记忆；T+1/T+6 行情只用于当年测试评估和拒单/卖出延迟处理。

模型族：

- `leaf_mean`：ExtraTreesRegressor 叶子内历史执行收益均值。
- `leaf_toprate`：ExtraTreesClassifier 叶子内历史 top-exec 概率。
- `proto_mean/proto_tail/proto_toprate`：MiniBatchKMeans 原型记忆的均值、右尾均值和右尾率。
- `leaf_blend/proto_blend/memory_blend`：上述记忆分数的简单融合。

年度 walk-forward 样本：2022 到 2026，共 217 个信号期。训练样本从 139,744 行逐年扩展到 735,870 行，2026 只使用 2021-2025 历史训练。

### 134.3 结果

账户代理核心结果：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf_mean::top10` | 4.02x | `+40.37%` | `+21.89%` | `+0.45%` | `+118.80%` | `+6.92%` | 9.96 | 5.01 | `-45.00%` |
| `leaf_mean::top20` | 3.55x | `+21.79%` | `+18.11%` | `+6.63%` | `+111.11%` | `+9.62%` | 19.94 | 5.00 | `-36.68%` |
| `leaf_blend::top20` | 1.41x | `-23.08%` | `+20.27%` | `-7.17%` | `+61.02%` | `+1.89%` | 19.91 | 5.01 | `-42.39%` |
| `memory_blend::top20` | 1.52x | `-20.93%` | `+0.93%` | `+15.08%` | `+55.87%` | `+6.18%` | 19.94 | 5.01 | `-45.54%` |
| `proto_mean::top20` | 1.23x | `+4.49%` | `-10.53%` | `-0.00%` | `+55.32%` | `-15.39%` | 19.93 | 5.00 | `-33.30%` |
| `proto_tail::top20` | 0.11x | `-42.53%` | `-45.96%` | `-60.85%` | `-20.28%` | `+16.04%` | 19.63 | 5.01 | `-92.28%` |
| `leaf_toprate::top20` | 0.006x | `-81.77%` | `-71.83%` | `-74.84%` | `-49.37%` | `-12.15%` | 19.57 | 5.03 | `-99.51%` |

标签层：

| 方案 | mean exec label | entry ok | 正 session 比例 | 2026 mean exec label |
| --- | ---: | ---: | ---: | ---: |
| `leaf_mean::top10` | `+0.004265` | 99.77% | 58.53% | `+0.005826` |
| `leaf_mean::top20` | `+0.003966` | 99.82% | 61.29% | `+0.006523` |
| `leaf_blend::top20` | `+0.000175` | 99.61% | 49.77% | `+0.002575` |
| `memory_blend::top20` | `+0.000540` | 99.75% | 48.39% | `+0.004496` |
| `leaf_toprate::top20` | `-0.024598` | 97.95% | 30.41% | `-0.001233` |

`leaf_reg` 特征重要性前十：

| 特征 | importance |
| --- | ---: |
| `mkt_amount_disp20` | 0.0670 |
| `mkt_disp20` | 0.0626 |
| `mkt_ret20_median` | 0.0532 |
| `mkt_near_high20` | 0.0530 |
| `mkt_breadth20` | 0.0516 |
| `price_rank` | 0.0441 |
| `amount_rank` | 0.0404 |
| `price_bucket_breadth` | 0.0345 |
| `price_bucket_amt` | 0.0307 |
| `ret60_rank` | 0.0295 |

### 134.4 反事实分析

第一反事实：如果自适应近邻/叶子记忆是缺失的主收益引擎，它应超过 Exp40 的可复核基线。实际最好的 `leaf_mean::top20` 只有 3.55x，低于 Exp40 formal rebalance5 的 8.77x。

第二反事实：如果右尾概率记忆能抓住历史大涨结构，`leaf_toprate` 和 `proto_tail` 不应系统性亏损。实际二者几乎归零或大幅亏损，说明“历史右尾概率/尾部原型”容易选到拥挤失败股。

第三反事实：如果原型聚类能自然表示收益状态，`proto_mean` 应接近或超过树叶均值。实际 `proto_mean::top20` 只有 1.23x，且 2026 为 `-15.39%`，高维欧式原型不如监督树叶分段。

第四反事实：如果 Exp130 的动态风格桶只是缺少非线性模型，融合叶子/原型/右尾率应进一步增厚。实际融合分数比纯 `leaf_mean` 更差，说明稳定部分来自局部均值，不来自右尾追逐或简单融合。

### 134.5 判定

`rejected_with_stable_but_too_thin_signal`。

`leaf_mean` 是一个干净、因果、逐年全正且 2026 正收益的稳定信号；但它收益太薄，最大回撤仍深，且没有超过 Exp40，更远低于几十倍到百倍目标。它暂不合代码，也不作为主策略推进。

后续可把 `leaf_mean` 当作诊断线索：高维状态分段均值比固定路径族和右尾概率更稳，但真正需要的是新的可交易右尾候选土壤，或验证它是否能作为低相关 sleeve/候选条件为 Exp40 弱年补收益。

## 135. qmt_leaf_mean_complement_v1

### 135.1 问题

Exp134 发现 `leaf_mean` 是干净、因果、逐年全正的稳定薄信号，但收益只有 3.55x/4.02x，低于 Exp40。Exp135 不把它包装成主策略，而是审计它是否能作为低相关 sleeve 或候选条件，补 Exp130/QMT 风格资金流模型的弱段。

核心反事实：如果 `leaf_mean` 真能提供低相关互补收益，那么与风格模型 `style_reg/style_blend/style_lag_catchup` 的融合、共识或并集 Top20 应自然超过 `leaf_mean` 本身，至少不能把逐年正收益和 2026 前向削弱。

### 135.2 方法

产物：

```text
.tmp/quantx-research/qmt-leaf-mean-complement-v1/analyze_qmt_leaf_mean_complement.py
sha256:e9e499ae765633bdcd97ef61e2459efd93a838813e764ed5014a2c94c76a05f6

.tmp/quantx-research/qmt-leaf-mean-complement-v1/qmt_leaf_mean_complement_2021_2026_diagnostic.json
sha256:0dc6b81b17462b1409da3b53f43fd4b1aff3420d4e36726cef1733f85fa18222

.tmp/quantx-research/qmt-leaf-mean-complement-v1/exp135_summary.md
```

本轮只复用 QMT/qlib 日线 OHLCV/VWAP 和 raw `amount/is_st`。不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念表或其它信息流数据。

无未来函数设定：复用 Exp130 的 T 日已完成日线面板；`style` 和 `leaf` 两组模型都按年度 walk-forward 训练，每个测试年只使用 `< year` 的历史样本；融合分数只使用同日 OOS 模型分数；T+1/T+6 open 只用于评价与账户代理。

对比对象：

- 单独 `leaf_mean`、`style_reg`、`style_blend`、`style_lag_catchup`。
- `leaf_reg_25/50`、`leaf_blend_25/50`：叶均值与风格模型 rank 融合。
- `leaf_reg_consensus`：两者最低 rank，强调共识。
- `leaf_reg_max`：两者最高 rank，强调并集。
- Top20 交集、leaf-only、other-only、session 相关、弱段补偿比例。

### 135.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最差年 | 逐年正 | 平均选中 | 平均持有 | mean exec label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf_mean::top10` | 4.02x | `+6.92%` | `+0.45%` | 是 | 9.96 | 5.01 | `+0.004265` |
| `leaf_mean::top20` | 3.55x | `+9.62%` | `+6.63%` | 是 | 19.94 | 5.00 | `+0.003966` |
| `leaf_blend_50::top20` | 2.53x | `+7.43%` | `+4.29%` | 是 | 19.92 | 5.01 | `+0.002380` |
| `leaf_reg_50::top20` | 2.67x | `+4.83%` | `-8.41%` | 否 | 19.95 | 5.01 | `+0.002908` |
| `leaf_reg_consensus::top20` | 2.38x | `+2.37%` | `-16.67%` | 否 | 19.96 | 5.01 | `+0.002390` |
| `style_reg::top20` | 1.66x | `-5.75%` | `-15.47%` | 否 | 19.94 | 5.01 | `+0.000568` |
| `style_blend::top20` | 1.53x | `+7.52%` | `-26.92%` | 否 | 19.89 | 5.01 | `+0.000388` |
| `leaf_lag_50::top20` | 1.53x | `+6.49%` | `-13.64%` | 否 | 19.97 | 5.00 | `+0.000363` |

互补性诊断：

| 对比 | Top20 平均重合 | session 收益相关 | leaf 胜率 | other 负且 leaf 正比例 | leaf-only exec | other-only exec | union/fusion Top20 exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf` vs `style_reg` | 2.21 | `+0.5406` | 52.07% | 23.50% | `+0.003897` | `+0.000056` | `+0.002908` |
| `leaf` vs `style_blend` | 1.66 | `+0.4199` | 58.06% | 27.19% | `+0.004435` | `+0.000369` | `+0.002380` |
| `leaf` vs `style_lag_catchup` | 0.12 | `-0.1599` | 62.67% | 36.87% | `+0.003969` | `-0.003888` | `+0.000363` |

关键年度观察：

- `leaf` 与 `style_lag_catchup` 几乎零重合，2026 `style_lag_catchup` 单期 exec 更强，但 2022-2025 多年为负。
- `leaf` 与 `style_reg/style_blend` 有弱到中等正相关，leaf-only 明显强于 other-only。
- 所有融合、共识、并集口径都低于单独 `leaf_mean`；最好的逐年正融合 `leaf_blend_50::top20` 只有 2.53x。
- 相对 Exp40：`leaf_mean::top20` 3.55x 仍远低于 Exp40 formal 8.77x；2026 `+9.62%` 也低于 Exp40 `+19.18%`。

### 135.4 反事实分析

第一反事实：如果 `leaf_mean` 是低相关可叠加 sleeve，融合后应超过单独 `leaf_mean`。实际所有融合都变薄，最好的逐年正融合只有 2.53x，低于 `leaf_mean::top20` 的 3.55x。

第二反事实：如果 `style_lag_catchup` 是 2026 的互补答案，低重合和负相关应带来组合增厚。实际 `leaf_lag_50::top20` 只有 1.53x，2022 为负；2026 强项来自后验风格，不是穿越机制。

第三反事实：如果风格模型能补 leaf 弱年，leaf-only 与 other-only 应各有优势。实际 leaf-only exec 明显高于 other-only；风格模型更多是在稀释 leaf 的稳定均值。

第四反事实：如果当前问题只是缺组合方式，共识或并集应至少改善回撤并保持收益。实际 `leaf_reg_consensus/top20`、`leaf_reg_max/top20` 都有负年份，收益也低于 leaf 本体。

### 135.5 判定

`rejected_as_complement_overlay`。

`leaf_mean` 仍是一个稳定但太薄的诊断信号；它和部分风格模式确实有低重合甚至负相关，但低相关没有转化为可组合收益厚度。当前不推进 formal account，不合代码，也不继续调融合权重、共识阈值或并集 TopK。

下一步应离开“薄信号拼接/互补 overlay”，继续寻找新的可交易右尾候选土壤。优先方向应满足：候选本身在标签层 Top20 就比 Exp40 更厚，而不是靠组合已有薄信号增厚。

## 136. qmt_session_topk_lambdarank_v1

### 136.1 问题

Exp135 否定了“薄信号互补拼接”作为突破方向。下一步必须让候选本身在标签层 Top20 变厚，而不是靠已有弱专家融合。因此 Exp136 直接把目标改为同日 TopK 排序：每个信号日作为 query group，用 LambdaRank 学习一周可执行收益的同日分位，检查更贴近 Top20 的排序目标是否能超过 Exp40。

核心反事实：如果此前点预测/分类目标与 Top20 等权组合错位是主要瓶颈，那么同日 query 的 LambdaRank 应自然提高 Top20 标签厚度，并在 2026 不退化；如果越强调右尾越亏，则说明问题不是目标函数，而是可见特征无法区分可交易右尾和拥挤失败右尾。

### 136.2 方法

产物：

```text
.tmp/quantx-research/qmt-session-topk-lambdarank-v1/analyze_qmt_session_topk_lambdarank.py
sha256:8beea29f8d73761c74bbf6a75e11701ae9ee26398724a06ce4e0c3b3c64ca43e

.tmp/quantx-research/qmt-session-topk-lambdarank-v1/qmt_session_topk_lambdarank_2021_2026_diagnostic.json
sha256:ed4f8b5e4792c9a1afc9415abc87d7121d30ade64b3c93977c95ae13eba9dc68

.tmp/quantx-research/qmt-session-topk-lambdarank-v1/exp136_summary.md
```

本轮只使用 QMT/qlib 日线 OHLCV/VWAP 和 raw `amount/is_st`，复用 Exp130 的日线横截面风格面板。不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念表或其它信息流数据。

无未来函数设定：所有特征只使用信号日 T 及以前已完成日线；标签是 T+1 open 到 T+6 open 的可执行收益代理；每个测试年只用 `< year` 的历史 session 训练；LightGBM Ranker 的 group 是历史信号日；当年未来收益只用于评价和账户代理。

训练目标：

- `rank_q5`：同日可执行收益五分位，label gain `[0,1,3,7,15]`。
- `rank_q10`：同日十分位，label gain 更偏右尾。
- `rank_top5`：同日 top 5% 二分类式排序。
- `rank_top10_focus`：同日 top10%/top20% 多级排序。

### 136.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均选中 | 平均持有 | mean exec label | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank_q5::top20` | 2.59x | `+5.65%` | `-11.86%` | 19.95 | 5.00 | `+0.002514` | `-21.82%` |
| `rank_q5::top10` | 1.97x | `+6.30%` | `-11.45%` | 9.99 | 5.00 | `+0.001528` | `-30.59%` |
| `rank_q10::top10` | 0.96x | `-11.53%` | `-34.04%` | 9.98 | 5.00 | `-0.001741` | `-56.11%` |
| `rank_top10_focus::top20` | 0.59x | `-14.56%` | `-38.65%` | 19.91 | 5.01 | `-0.003888` | `-55.96%` |
| `rank_top5::top20` | 0.04x | `-20.68%` | `-65.35%` | 19.67 | 5.02 | `-0.015609` | `-96.43%` |

`rank_q5::top20` 年度标签：

| 年份 | mean exec label | mean raw | 正 session 比例 | entry ok |
| --- | ---: | ---: | ---: | ---: |
| 2022 | `-0.002537` | `-0.000230` | 41.67% | 100.00% |
| 2023 | `+0.003792` | `+0.006182` | 65.31% | 99.80% |
| 2024 | `+0.009077` | `+0.012423` | 66.67% | 99.90% |
| 2025 | `-0.001048` | `+0.009071` | 40.82% | 99.80% |
| 2026 | `+0.004223` | `+0.004781` | 56.52% | 99.13% |

`rank_top5::top20` 年度标签全负，且 entry ok 下降到 96%-99%，说明强调极端右尾会选到更拥挤、不可买或失败的股票。

特征重要性前列集中在旧结构：`range20_low_rank`、`ret60_rank`、`price_rank`、`amt_ratio20_rank`、`amount_rank`、`upper_wick_low_rank`。没有出现新的主题/行业/状态表示，模型仍主要在旧的低波、中期强弱、价格和成交额风格上排序。

相对 Exp40：Exp40 due5 Top20 开发期 `+0.014170`、2026 `+0.012520`；本轮最好 `rank_q5::top20` 全期 exec label 仅 `+0.002514`，2026 `+0.004223`，账户代理 2.59x 也远低于 Exp40 formal 8.77x/2026 `+19.18%`。

### 136.4 反事实分析

第一反事实：如果目标函数错位是主瓶颈，同日 LambdaRank 应显著提高 Top20 标签厚度。实际最好 `rank_q5::top20` 仍只有 2.59x，且 2022/2025 为负。

第二反事实：如果右尾排序目标能抓住可交易赢家，`rank_top5` 或 `rank_top10_focus` 应强于温和五分位。实际越强调右尾越亏，`rank_top5::top20` 接近归零。

第三反事实：如果当前 QMT 日线特征已经包含可交易右尾结构，只是模型没对齐，重要性应出现新的动态主题/风险结构。实际重要性仍集中在旧因子，说明特征空间没有新土壤。

第四反事实：如果失败来自入场拒单，温和 `rank_q5` 的 entry ok 接近 99%-100% 时应恢复收益。实际 2022/2025 标签仍负，说明不是单纯可交易性问题。

### 136.5 判定

`rejected_as_topk_objective_not_enough`。

同日 LambdaRank 不能打开收益上限。温和五分位排序保留薄信号，但远低于 Exp40；右尾化目标直接学到拥挤失败股。后续不继续调 label gain、NDCG cutoff、TopK 或 ranker 参数。

下一步应换更大胆的候选土壤假设：不是在旧日线风格面板上换目标函数，而是引入能描述“资金行为/筹码状态/主线位置”的新 QMT 可拉取特征，或者从更高频 K 线中寻找日线看不到的可交易右尾前置信号。

## 137. qmt_cost_pressure_soil_v1

### 137.1 问题

Exp136 说明，在旧 QMT 日线风格面板上换成同日 TopK/LambdaRank 目标，不能把 Top20 标签自然抬厚；越强调右尾越容易追到拥挤失败股。因此本轮换一个 QMT-only 的新特征土壤：用过去 20/60 日价格与成交额分布近似“筹码成本/套牢压力/支撑密集区”，检查它是否能识别更容易出一周收益的可交易右尾。

核心反事实：如果当前缺的是筹码结构而不是目标函数，那么成本压力特征本身或加入成本压力后的 ML，应在 Top20 账户代理上明显超过 Exp40，并且 2026 不退化；如果手写筹码质量分为负、ML 也只学成薄稳定信号，则说明日线成交额分布代理太粗，不能作为主收益土壤。

### 137.2 方法

产物：

```text
.tmp/quantx-research/qmt-cost-pressure-soil-v1/analyze_qmt_cost_pressure_soil.py
sha256:8b7b15f6655e1be4869ca8e814e721168135922abc40d2cba27c4401e197856d

.tmp/quantx-research/qmt-cost-pressure-soil-v1/qmt_cost_pressure_soil_2021_2026_diagnostic.json
sha256:78dacd816f89b825217311f99673cf56fa730fc1346fe466c440def6fa1d327c

.tmp/quantx-research/qmt-cost-pressure-soil-v1/exp137_summary.md
```

本轮只使用 QMT/qlib 日线 OHLCV/VWAP 和 raw `amount/is_st`，复用 Exp130 的日线横截面风格面板。不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念表或其它信息流数据。

无未来函数设定：特征只用信号日 T 及以前已完成日线；20/60 日成本分布代理只用 `idx-window+1:idx+1`；每个测试年只用 `< year` 的历史样本训练；标签和账户代理是 T+1 open 入场、T+6 open 目标退出。

fold 输出：

| 测试年 | train rows | test rows | sessions |
| --- | ---: | ---: | ---: |
| 2022 | 139744 | 143064 | 48 |
| 2023 | 282808 | 149756 | 49 |
| 2024 | 432564 | 149402 | 48 |
| 2025 | 581966 | 153904 | 49 |
| 2026 | 735870 | 72806 | 23 |

新增 20/60 日成交额加权成本代理：获利盘比例、当前价上方 0-12% 套牢压力、当前价附近 3.5% 成交密集度、当前价下方 0-8% 支撑密集度、当前价相对成交额加权成本均值的位置、最近上方压力/下方支撑距离。

模型和评分：

- `reg`：旧特征 + 筹码特征预测一周可执行超额。
- `cls`：旧特征 + 筹码特征预测同日 top5%。
- `chip_reg`：只用筹码特征 + 市场状态预测。
- `blend`：`reg/cls/chip_reg` 的 session rank 融合。
- 三个手写分：`cost_pressure_quality`、`cost_breakout_quality`、`cost_repair_quality`。

### 137.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均选中 | 平均持有 | mean exec label | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend::top10` | 4.97x | `+1.28%` | `+1.28%` | 9.95 | 5.01 | `+0.005458` | `-33.62%` |
| `reg::top10` | 2.83x | `+2.04%` | `-1.35%` | 9.96 | 5.01 | `+0.002738` | `-31.91%` |
| `blend::top20` | 2.78x | `+6.91%` | `+0.52%` | 19.91 | 5.01 | `+0.002935` | `-37.31%` |
| `reg::top20` | 2.69x | `+8.45%` | `-0.28%` | 19.91 | 5.01 | `+0.002477` | `-27.15%` |
| `chip_reg::top20` | 1.56x | `+2.20%` | `-13.05%` | 19.94 | 5.01 | `+0.000307` | `-35.31%` |
| `cost_repair_quality::top20` | 0.97x | `+7.57%` | `-37.72%` | 19.92 | 5.00 | `-0.002299` | `-44.95%` |
| `cost_pressure_quality::top20` | 0.84x | `-5.79%` | `-31.47%` | 19.83 | 5.00 | `-0.003376` | `-38.56%` |
| `cost_breakout_quality::top20` | 0.53x | `-7.57%` | `-39.54%` | 19.94 | 5.00 | `-0.004915` | n/a |

`blend::top20` 年度账户收益：

| 年份 | return |
| --- | ---: |
| 2022 | `+0.52%` |
| 2023 | `+12.63%` |
| 2024 | `+14.21%` |
| 2025 | `+101.09%` |
| 2026 | `+6.91%` |

`blend::top20` 年度标签：

| 年份 | mean exec label | mean raw | 正 session 比例 | entry ok |
| --- | ---: | ---: | ---: | ---: |
| 2022 | `+0.000255` | `+0.002861` | 54.17% | 99.69% |
| 2023 | `+0.001425` | `+0.004619` | 46.94% | 99.80% |
| 2024 | `+0.004214` | `+0.007198` | 56.25% | 100.00% |
| 2025 | `+0.004965` | `+0.017342` | 48.98% | 99.18% |
| 2026 | `+0.004751` | `+0.005698` | 60.87% | 99.78% |

重要性前列仍由市场状态和旧风格主导：`mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`range20_low_rank`、`amt_ratio20_rank`、`ret60_rank`、`price_rank`。筹码特征中较靠前的是 `cost60_distance_rank`、`cost20_turnover_rank`，但没有形成主导收益结构。

相对 Exp40：Exp40 formal rebalance5 开发期 8.77x、2026 `+19.18%`，due5 Top20 标签开发期 `+0.014170`、2026 `+0.012520`。本轮最合规 Top20 `blend::top20` 只有 2.78x、2026 `+6.91%`、mean exec label `+0.002935`，明显低于 Exp40。

### 137.4 反事实分析

第一反事实：如果筹码成本压力是缺失主引擎，手写质量分应至少在 Top20 上为正。实际 `cost_pressure_quality/top20`、`cost_breakout_quality/top20`、`cost_repair_quality/top20` 全期 mean exec label 均为负，账户层多为亏损。

第二反事实：如果筹码特征本身足够厚，`chip_reg` 不应依赖旧特征也能穿越。实际 `chip_reg::top20` 只有 1.56x，2022/2024 为负，说明它不是独立 alpha。

第三反事实：如果新筹码特征能为旧风格模型增厚，`blend::top20` 应超过 Exp40 或至少超过 Exp134 leaf_mean。实际 2.78x 低于 Exp40，也低于 Exp134 `leaf_mean::top20` 的 3.55x。

第四反事实：如果问题来自可交易性，entry ok 很低时收益应被拒单拖累。实际 `blend::top20` entry ok 接近 99%-100%，收益仍薄，说明核心不是买不进去，而是候选收益厚度不足。

第五反事实：如果 Top10 高倍数可以作为方向，Top20 也应自然增厚。实际 `blend::top10` 4.97x 但 2026 只有 `+1.28%`，且平均持仓虽大于 5 但低于主目标 Top20；不能通过缩窄 TopK 或尾部集中来满足用户要求。

### 137.5 判定

`rejected_with_stable_but_too_thin_signal`。

筹码/成本压力日线代理没有打开收益上限。它与市场状态一起可以形成逐年正的薄信号，但 Top20 收益远低于 Exp40，手写筹码质量分本身多数为负。后续不继续调成本带宽、20/60 窗口、手写权重或 chip/reg 融合比例。

下一步应离开“日线成交额成本分布代理”，转向更能描述真实资金行为的 QMT 可拉取粒度：优先验证分钟线日内承接/修复/吸筹形态是否能在 T 日收盘后为 T+1 open 提供可交易右尾前置信号；若分钟历史覆盖不足，再回到全市场动态主线图或账户边际 sleeve 的新标签设计。

## 138. qmt_minute_coverage_probe_v1

### 138.1 问题

Exp137 否定了日线成交额成本分布代理。下一步自然会想到 QMT 分钟线：日内承接、修复、吸筹、尾盘资金行为都可能比日线 OHLCV 更接近真实资金行为。但当前主目标要求 2021-2026 五年级别验证、逐年为正并通过 2026 前向，因此在做任何分钟线策略实验前，必须先确认本地/QMT 远程分钟历史是否覆盖 2021-2026。

本轮只做数据覆盖探针，不生成交易信号，不做策略结论。

### 138.2 方法

产物：

```text
.tmp/quantx-research/qmt-minute-coverage-probe-v1/probe_qmt_minute_coverage.py
sha256:0ac53643aeeacdf418548355a4417fec698b0cdbc922752005158e32f7e8e940

.tmp/quantx-research/qmt-minute-coverage-probe-v1/qmt_minute_coverage_probe.json
sha256:13b299e63a3f0c574ce517c067c1c230c5b303bbdf837deab6b9783a8e095019

.tmp/quantx-research/qmt-minute-coverage-probe-v1/exp138_summary.md
```

探测样本：`SH600000`、`SZ000001`、`SH600519`、`SZ300750`。年份点：2021、2022、2023、2024、2025、2026 年初和 2026 年中。周期：`5m`、`1m`。

### 138.3 结果

覆盖汇总：

| 周期 | 年份 | checks | non-empty | avg rows | errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5m` | 2021 | 4 | 0 | 0 | 0 |
| `5m` | 2022 | 4 | 0 | 0 | 0 |
| `5m` | 2023 | 4 | 0 | 0 | 0 |
| `5m` | 2024 | 4 | 0 | 0 | 0 |
| `5m` | 2025 | 4 | 0 | 0 | 0 |
| `5m` | 2026 | 8 | 8 | 48 | 0 |
| `1m` | 2021 | 4 | 0 | 0 | 0 |
| `1m` | 2022 | 4 | 0 | 0 | 0 |
| `1m` | 2023 | 4 | 0 | 0 | 0 |
| `1m` | 2024 | 4 | 0 | 0 | 0 |
| `1m` | 2025 | 4 | 0 | 0 | 0 |
| `1m` | 2026 | 8 | 8 | 241 | 0 |

2026 返回字段稳定：`time/open/high/low/close/volume/amount/preClose/suspendFlag` 等；`5m` 每日 48 根，`1m` 每日 241 根。2021-2025 全部为空，且没有远程错误，说明是历史分钟覆盖限制，不是连接失败。

### 138.4 判定

`coverage_limited_to_2026_only`。

QMT 远程分钟线可以用于 2026 辅助解释和小样本诊断，但不能作为 2021-2026 主策略验证数据源。任何依赖分钟线的结论都不能满足当前“五年逐年正、通过 2026 前向”的主目标。

下一轮主线不再推进分钟线策略实验；应回到 QMT 日线可完整覆盖的数据，但换更结构性的方向：例如动态横截面主线图、股票-市场风险暴露的时间变系数、或者从账户收益贡献反推低相关 sleeve 的训练标签。分钟线只保留为 2026 失败/成功样本的解释工具。

## 139. qmt_dynamic_residual_exposure_v1

### 139.1 问题

Exp138 确认 QMT 分钟线只有 2026 覆盖，不能作为五年主验证数据源。因此本轮回到 2021-2026 完整覆盖的 QMT 日线，但不继续做旧日线目标函数、薄信号融合或单一 leader/follower。

本轮假设：A 股强行业/概念轮动可以先表现为“动态横截面风险暴露”。如果只追强势 leader 会拥挤失败，那么也许应该买“跟随主线，但个股残差仍强、风险暴露不极端”的股票。核心反事实是：如果这种动态风险暴露/残差结构是缺失主引擎，它应自然提高 Top20 标签和账户代理，且 2026 不退化。

### 139.2 方法

产物：

```text
.tmp/quantx-research/qmt-dynamic-residual-exposure-v1/analyze_qmt_dynamic_residual_exposure.py
sha256:9f839938f43dbcf2967bc2a6109d33dc1d1c0a1f5ef111080d4f153f993d7b59

.tmp/quantx-research/qmt-dynamic-residual-exposure-v1/qmt_dynamic_residual_exposure_2021_2026_diagnostic.json
sha256:d059851af18bc4d00b3d62300eebc16b6f908a10158f30e11f8f91b4e682585e

.tmp/quantx-research/qmt-dynamic-residual-exposure-v1/exp139_summary.md
```

无未来函数设定：只使用 QMT/qlib 日线 OHLCV/VWAP 和 raw `amount/is_st`；信号日 T 的动态因子组合、滚动 beta 和残差只使用 T 及以前 60 日收益；每个测试年只用 `< year` 样本训练；标签和账户代理仍是 T+1 open 入场、T+6 open 目标退出。

fold 输出：

| 测试年 | train rows | test rows | sessions |
| --- | ---: | ---: | ---: |
| 2022 | 139744 | 143064 | 48 |
| 2023 | 282808 | 149756 | 49 |
| 2024 | 432564 | 149402 | 48 |
| 2025 | 581966 | 153904 | 49 |
| 2026 | 735870 | 72806 | 23 |

在 Exp130 QMT 风格资金流面板上新增动态多因子暴露：

- 每个 T 日用过去 60 日收益构造 5 个可见横截面因子组合：市场均值、20 日动量头部、低波头部、高成交额头部、低价格头部。
- 对每只股票用过去 60 日收益回归这些因子组合，得到 `mkt/mom/lowvol/liq/price beta` 和残差路径。
- 再用最近 20 日暴露相对 60 日暴露的变化，构造 beta instability、mom/liquidity beta change。
- 派生残差动量、残差修复、低拥挤残差质量、暴露突破质量等特征。

训练/评分：`reg`、`cls`、`exp_reg`、`blend`，以及手写分 `crowding_resid_quality`、`exposure_breakout_quality`、`factor_aligned_resid`。

### 139.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均选中 | 平均持有 | mean exec label | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exposure_breakout_quality::top10` | 1.16x | `-7.77%` | `-18.89%` | 9.93 | 5.00 | `-0.001321` | `-49.01%` |
| `exposure_breakout_quality::top20` | 1.08x | `-8.93%` | `-14.33%` | 19.88 | 5.00 | `-0.001881` | `-42.98%` |
| `reg::top20` | 1.06x | `-2.56%` | `-31.65%` | 19.93 | 5.00 | `-0.001549` | `-44.71%` |
| `crowding_resid_quality::top20` | 1.03x | `-17.75%` | `-17.75%` | 19.95 | 5.00 | `-0.001711` | `-27.23%` |
| `blend::top20` | 0.89x | `+11.20%` | `-32.91%` | 19.94 | 5.00 | `-0.001579` | `-57.24%` |
| `exp_reg::top20` | 0.64x | `-7.08%` | `-34.53%` | 19.95 | 5.00 | `-0.003441` | `-54.50%` |
| `factor_aligned_resid::top20` | 0.04x | `+15.49%` | `-67.88%` | 18.95 | 5.03 | `-0.018664` | `-96.48%` |
| `cls::top20` | 0.04x | `-32.82%` | `-72.67%` | 19.57 | 5.03 | `-0.016155` | `-96.36%` |

`reg::top20` 年度账户收益：2022 `-7.00%`、2023 `+12.48%`、2024 `-31.65%`、2025 `+51.90%`、2026 `-2.56%`。

`blend::top20` 年度账户收益：2022 `-32.91%`、2023 `+15.33%`、2024 `-28.66%`、2025 `+45.10%`、2026 `+11.20%`。

`factor_aligned_resid::top20` 年度账户收益：2022 `-54.83%`、2023 `-57.81%`、2024 `-67.88%`、2025 `-37.90%`、2026 `+15.49%`。这是典型 2026 后验风格，开发期完全不可用。

`reg::top20` 年度标签：

| 年份 | mean exec label | mean raw | 正 session 比例 | entry ok |
| --- | ---: | ---: | ---: | ---: |
| 2022 | `-0.001291` | `+0.001213` | 52.08% | 99.48% |
| 2023 | `+0.001582` | `+0.004151` | 46.94% | 99.90% |
| 2024 | `-0.007885` | `-0.004612` | 47.92% | 100.00% |
| 2025 | `+0.000193` | `+0.010488` | 36.73% | 99.69% |
| 2026 | `+0.000756` | `+0.001724` | 52.17% | 99.57% |

特征重要性前列仍主要是市场状态和旧风格：`mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`range20_low_rank`、`amt_ratio20_rank`、`ret60_rank`、`price_rank`。动态暴露特征中有 `resid_vol20_low_rank`、`price_beta_rank`、`lowvol_beta_rank`、`mom_beta_rank` 进入前列，但没有带来收益厚度。

相对 Exp40：Exp40 formal rebalance5 开发期 8.77x、2026 `+19.18%`，due5 Top20 标签开发期 `+0.014170`、2026 `+0.012520`。本轮最好 Top20 只有 1.08x，且没有逐年正，远低于 Exp40。

### 139.4 反事实分析

第一反事实：如果“跟随主线但残差强”是缺失主引擎，`reg/top20` 或 `exp_reg/top20` 应至少逐年为正。实际 `reg::top20` 2022/2024/2026 为负，`exp_reg::top20` 全期亏损。

第二反事实：如果暴露突破能替代固定 leader/follower，`exposure_breakout_quality` 应比 Exp121/127 稳定。实际 Top20 只有 1.08x，2023/2024/2026 为负，仍是薄弱局部状态。

第三反事实：如果 2026 的残差主线可穿越，开发期不应系统性反向。实际 `factor_aligned_resid::top20` 在 2026 `+15.49%`，但 2022-2025 全部大幅为负，不能作为策略方向。

第四反事实：如果问题来自可交易性，entry ok 很低才应拖累收益。实际 `reg/blend/top20` entry ok 接近 99%-100%，但收益仍薄；而 `factor_aligned_resid` 的 entry ok 下降且收益崩盘，说明它追到了更拥挤的错误侧。

第五反事实：如果新增暴露特征是新土壤，模型重要性应从旧市场状态迁移到暴露残差结构。实际前列仍被市场状态和旧风格占据，暴露特征只是被模型利用但没有收益弹性。

### 139.5 判定

`rejected_as_residual_exposure_not_enough`。

动态风险暴露/残差动量没有打开收益上限。它比单一 leader/follower 更结构化，但结果仍退化为市场状态和旧风格解释，且 2026 的正残差模式在开发期强烈反向。后续不继续调因子组合数量、回归窗口、暴露变化权重或手写残差分数。

下一步需要更大胆地跳出“解释当日强弱”的日线特征：优先审视已有高收益但 formal 打折的 Top5 日频重叠结构，反推为什么简化收益巨大而正式账户只剩 7.52x，寻找可交易替代结构；或者构造账户层边际贡献/拒单替代的标签，而不是继续扩充股票日线解释特征。

## 140. qmt_formal_loss_attribution_v1

### 140.1 问题

Exp66/67/84 反复说明：日频 Top5 sleeve 在简化 open 口径可以到几十倍甚至百倍，但 formal account 被涨停、跳变、停牌、手数和成本打到个位数，2026 也可能转负。本轮不再发明新候选，而是直接问一个更窄的问题：能否用 T 日可见的 QMT 日线横截面特征，提前区分“理论右尾”和“可执行右尾”，从而找到可交易替代候选？

### 140.2 方法

产物：

```text
.tmp/quantx-research/qmt-formal-loss-attribution-v1/analyze_qmt_formal_loss_attribution.py
sha256:7e616a51c224ec82985d7ed6bf2563705e3dd6a4dfabc4d032d89ab0b2d9d8ef

.tmp/quantx-research/qmt-formal-loss-attribution-v1/qmt_formal_loss_attribution_2021_2026_diagnostic.json
sha256:7c0cb3d2d2647f00957d409e16885b44e09c81c16f386469f3b2e5cb664c1e17

.tmp/quantx-research/qmt-formal-loss-attribution-v1/exp140_summary.md
```

无未来函数设定：复用 Exp130 的 QMT 日线面板，所有特征只使用信号日 T 及以前已完成日线数据；标签是预定 T+1 open 到 T+6 open 的 raw/exec 收益和 formal loss；每个测试年只用 `< year` 样本训练。T+1/T+6 行情只用于历史标签和测试评估，不参与当年信号。

模型：LightGBM 分别学习 `raw5_open`、`exec_label5_open`、`formal_loss = raw5_open - exec_label5_open`、`entry_ok`、`top_exec`、`bad_exec`，再组合出 `raw`、`exec`、`raw_minus_loss`、`exec_tradeable`、`substitute`、`formal_avoid`。

fold 输出：

| 测试年 | train rows | test rows | sessions | train entry ok | train bad exec |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139744 | 143064 | 48 | 99.53% | 20.01% |
| 2023 | 282808 | 149756 | 49 | 99.55% | 20.02% |
| 2024 | 432564 | 149402 | 48 | 99.62% | 20.02% |
| 2025 | 581966 | 153904 | 49 | 99.61% | 20.02% |
| 2026 | 735870 | 72806 | 23 | 99.59% | 20.02% |

### 140.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最大回撤 | 平均选中 | 平均持有 | mean exec label | entry ok | bad exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw::top10` | 2.70x | `+2.17%` | `-41.01%` | 7.86 | 5.01 | `-0.015051` | 78.7% | 40.4% |
| `exec::top20` | 1.92x | `+7.37%` | `-35.29%` | 19.94 | 5.01 | `+0.001057` | 99.8% | 21.3% |
| `exec_tradeable::top20` | 1.56x | `+9.35%` | `-34.82%` | 19.95 | 5.00 | `+0.000575` | 99.8% | 21.6% |
| `exec::top10` | 1.49x | `-6.32%` | `-46.60%` | 9.97 | 5.01 | `-0.000365` | 99.7% | 22.8% |
| `raw::top20` | 1.32x | `+7.43%` | `-46.72%` | 17.34 | 5.01 | `-0.010973` | 86.8% | 32.3% |
| `substitute::top20` | 1.16x | `+5.66%` | `-40.67%` | 19.99 | 5.00 | `-0.000984` | 100.0% | 14.2% |
| `raw_minus_loss::top20` | 1.01x | `+11.93%` | `-51.76%` | 19.95 | 5.00 | `-0.001965` | 99.7% | 21.5% |
| `top_exec::top20` | 0.03x | `-26.29%` | `-97.81%` | 19.56 | 5.03 | `-0.018549` | 97.9% | 50.4% |

损耗归因：

| 池 | mean raw | mean exec label | formal loss | entry ok | top raw rate | top exec rate | bad exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predicted raw tail | `+0.009919` | `-0.010973` | `+0.020892` | 86.8% | 11.1% | 7.2% | 32.3% |
| predicted exec tail | `+0.005466` | `+0.001057` | `+0.004408` | 99.8% | 7.0% | 7.0% | 21.3% |
| predicted substitute | `+0.002802` | `-0.000984` | `+0.003786` | 100.0% | 3.7% | 3.8% | 14.2% |

年度看，predicted raw tail 在 2025 mean raw `+0.02511`，但 mean exec `-0.01434`、formal loss `+0.03946`、entry ok 只有 79.3%；2026 mean raw `+0.01508`，mean exec `-0.00707`、formal loss `+0.02215`、entry ok 80.0%。也就是说越像 raw 右尾，越靠近 formal 拿不到或拿不稳的区域。

特征重要性前列仍是市场状态：`loss::mkt_ret20_median`、`raw::mkt_disp20`、`exec::mkt_disp20`、`loss::mkt_disp20`、`raw::mkt_ret20_median`、`raw/loss/exec::mkt_amount_disp20`、`mkt_near_high20` 等。损耗本身可被市场状态解释，但解释之后只剩薄的可执行收益。

相对 Exp40：Exp40 formal rebalance5 开发期 8.77x、2026 `+19.18%`，due5 Top20 标签开发期 `+0.014170`、2026 `+0.012520`。本轮最好 Top20 `exec::top20` 只有 1.92x、2026 `+7.37%`，远低于 Exp40。

### 140.4 反事实分析

第一反事实：如果 formal 打折只是执行工程噪声，那么 raw 右尾扣损耗后应保留厚收益。实际 `raw_minus_loss::top20` 只有 1.01x，mean exec label 为负，说明损耗不是小噪声，而是与右尾信号同源。

第二反事实：如果可交易替代候选存在，`substitute::top20` 应在 entry ok 接近 100% 时保持收益弹性。实际 final 1.16x，mean exec label 仍为负，说明可交易性改善主要拿掉右尾，而不是保留右尾。

第三反事实：如果直接学习可执行右尾可以解决问题，`exec::top20` 应显著超过 Exp40。实际 1.92x、2026 `+7.37%`，远低于 Exp40 formal 8.77x/2026 `+19.18%`。

第四反事实：如果坏成交规避本身是 alpha，`formal_avoid` 应至少逐年正。实际它只降低 bad exec，但收益更薄，说明避险不是收益来源。

第五反事实：如果 TopK/尾部权重能自然修复，Top10 应同步改善。实际 `raw::top10` final 2.70x 但 mean exec label 为 `-0.015051`，且平均选中低于主目标；不能作为满足要求的方向。

### 140.5 判定

`rejected_as_raw_tail_is_not_tradeable_edge`。

本轮证明了一个重要负结论：在当前 QMT 日线特征空间里，“理论右尾”与“formal 损耗”高度纠缠。raw 预测越强，越容易选到涨停/跳变/停牌或后续坏执行区域；把可交易性显式写进模型后，收益上限被压到 1-2x，远低于 Exp40，更远低于几十倍至 100 倍目标。

后续不继续围绕 raw/exec/loss 多头模型、entry_ok 分类器、bad_exec 惩罚或可交易性 rank blend 调参。下一步应换一个结构层级：不要再用股票级日线特征去“净化”不可交易右尾，而应寻找不依赖买入极端强势票的收益土壤，例如跨 session 的市场风险预算/行业概念轮动账户结构、低相关 sleeve 组合、或从可成交持仓的已实现贡献反推更长期的主线生命周期。

## 141. qmt_nonextreme_runway_v1

### 141.1 问题

Exp140 证明理论 raw 右尾与 formal 损耗高度纠缠：越像右尾，越容易落在涨停、跳变、停牌或后续坏执行区域。本轮换一个结构假设：不要追极端强势买入，而是寻找“非极端、可连续持有”的平滑 runway 候选。核心反事实是：如果可交易右尾并不在涨停票上，而是在强主线环境中温和延续的股票上，那么 2/3/5/8 日 open 路径质量应能自然增厚 Top20，并穿越 2026。

### 141.2 方法

产物：

```text
.tmp/quantx-research/qmt-nonextreme-runway-v1/analyze_qmt_nonextreme_runway.py
sha256:1eb0dcde47020ef2568bd08c92878e292223df6c16a8e5423c0aca5944338cf8

.tmp/quantx-research/qmt-nonextreme-runway-v1/qmt_nonextreme_runway_2021_2026_diagnostic.json
sha256:6847038340dad856721a4aa2837ba93af80850b050ad86aef8812b0e4cbeb56a

.tmp/quantx-research/qmt-nonextreme-runway-v1/exp141_summary.md
```

无未来函数设定：复用 Exp130/140 的 QMT 日线面板，所有特征只使用信号日 T 及以前已完成日线数据；runway 标签使用 T+1 open 入场后的 T+3/T+4/T+6/T+9 open 路径，仅作为历史训练标签和测试评价；每个测试年只用 `< year` 样本训练。

runway 标签：

```text
runway_raw = 0.35 * ret5 + 0.25 * ret8 + 0.25 * mean(ret2, ret3, ret5) + 0.15 * min(ret2, ret3, ret5)
runway_exec = runway_raw - cost if entry/exit ok else -0.08
smooth_winner = session top 10% runway_exec 且 min path > -3% 且 entry ok
```

模型：LightGBM 学习 `runway_exec`、`exec_label5_open`、`smooth_winner`、`runway_bad`，输出 `runway`、`exec`、`smooth`、`runway_smooth`、`nonextreme`，另保留手写 `mid_strength`。

fold 输出：

| 测试年 | train rows | test rows | sessions | train smooth rate | train runway exec mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139744 | 143064 | 48 | 9.56% | `-0.00049` |
| 2023 | 282808 | 149756 | 49 | 9.52% | `-0.00232` |
| 2024 | 432564 | 149402 | 48 | 9.58% | `-0.00268` |
| 2025 | 581966 | 153904 | 49 | 9.51% | `-0.00320` |
| 2026 | 735870 | 72806 | 23 | 9.54% | `-0.00167` |

### 141.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最大回撤 | 逐年正 | 平均选中 | 平均持有 | mean exec label | mean runway | entry ok | smooth rate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runway_smooth::top10` | 3.09x | `-11.68%` | `-32.03%` | 否 | 9.96 | 5.01 | `+0.002934` | `+0.002186` | 99.6% | 15.1% |
| `exec::top10` | 2.99x | `+0.46%` | `-33.13%` | 否 | 9.95 | 5.00 | `+0.003097` | `+0.001154` | 99.6% | 14.0% |
| `runway_smooth::top20` | 2.69x | `-1.89%` | `-31.80%` | 否 | 19.95 | 5.01 | `+0.002643` | `+0.002005` | 99.8% | 14.7% |
| `exec::top20` | 2.36x | `+7.42%` | `-26.72%` | 是 | 19.94 | 5.01 | `+0.002107` | `+0.000856` | 99.8% | 12.4% |
| `nonextreme::top20` | 2.28x | `+0.77%` | `-31.00%` | 否 | 19.97 | 5.00 | `+0.002028` | `+0.001506` | 99.9% | 13.6% |
| `runway::top20` | 1.75x | `+0.27%` | `-27.03%` | 否 | 19.95 | 5.00 | `+0.000852` | `+0.000300` | 99.8% | 11.2% |
| `smooth::top20` | 0.16x | `-5.95%` | `-87.32%` | 否 | 19.84 | 5.01 | `-0.009683` | `-0.011749` | 99.3% | 18.0% |

`exec::top20` 是唯一逐年正 Top20：2022 `+4.01%`、2023 `+9.39%`、2024 `+0.10%`、2025 `+93.18%`、2026 `+7.42%`，但 final 只有 2.36x，远低于 Exp40。

`runway_smooth::top20` 年度账户收益：2022 `+19.80%`、2023 `+36.62%`、2024 `+2.44%`、2025 `+63.58%`、2026 `-1.89%`。它在开发期比 `exec::top20` 稍厚，但 2026 转负。

年度标签看，`runway_smooth::top20` 在 2025 mean runway `+0.00878`，但 2026 降为 `-0.00149`；`exec::top20` 在 2026 mean exec label `+0.00535`，但 mean runway 只有 `+0.00068`。也就是说，平滑路径质量和一周可执行收益不是同一个稳定目标。

特征重要性仍以市场状态为主：`runway::mkt_disp20`、`exec::mkt_disp20`、`runway::mkt_ret20_median`、`runway::mkt_amount_disp20`、`exec::mkt_amount_disp20`、`runway::mkt_near_high20`。路径可持有性主要依赖市场状态，而不是独立股票级结构。

### 141.4 反事实分析

第一反事实：如果非极端平滑 runway 是缺失主引擎，`runway_smooth::top20` 应比 Exp40 更厚且通过 2026。实际只有 2.69x，2026 `-1.89%`，不满足。

第二反事实：如果 smooth_winner 分类器能识别可持有赢家，`smooth::top20` 不应大幅亏损。实际 final 0.16x、最大回撤 `-87.32%`，说明 top10% runway 事件概率本身会吸引到错误的失败路径。

第三反事实：如果可交易收益目标已足够，`exec::top20` 应至少接近 Exp40。实际逐年正但只有 2.36x，且 2025 单年贡献过重，收益厚度仍不够。

第四反事实：如果问题只是极端强势票不可买，非极端 rank blend 应保留右尾并提高 entry ok。实际 `nonextreme::top20` entry ok 达 99.9%，但 final 2.28x、2024 为负、2026 只有 `+0.77%`，说明可交易性不是收益来源。

第五反事实：如果手写中段强度能避开拥挤，`mid_strength` 应至少为正。实际 Top20 final 0.85x、2026 `-12.52%`，说明旧风格组合公式不能自然捕捉平滑右尾。

### 141.5 判定

`rejected_as_smooth_runway_not_thick_enough`。

平滑、非极端、可持有 runway 不是当前 QMT 日线空间里的独立厚 alpha。它能把入场成功率维持在 99% 附近，也能在 2025 顺风市场放大，但 2026 不穿越，整体收益只有 2-3x。后续不继续调 runway 权重、smooth winner 阈值、min path 阈值或 nonextreme blend。

Exp140/141 合起来说明：极端 raw 右尾不可交易，非极端平滑右尾又不够厚。下一步应离开单股票候选净化，转向更高层的账户结构或横截面机会分解：例如把 session 级市场风险/横截面离散度/行业概念轮动作为“何时需要哪种持仓形态”的条件，或者重新挖掘低相关 sleeve，而不是继续从同一套股票日线特征里榨 Top20。

## 142. qmt_robust_cell_soil_v1

### 142.1 问题

Exp140/141 说明：极端 raw 右尾不可交易，非极端平滑 runway 又不够厚。本轮跳出连续股票级打分，改做“诚实横截面 cell mining”：用 T 日可见的多维 rank 把股票切成风格单元，只用过去年份统计 cell 的 exec 均值、最差年、坏执行比例和 top_exec 命中，再把这些历史统计映射到测试年。

核心反事实：如果金子藏在稳定风格土壤里，而不是单票右尾排序里，那么历史稳定 cell 应该在 Top20 上自然形成厚收益，并通过 2026。

### 142.2 方法

产物：

```text
.tmp/quantx-research/qmt-robust-cell-soil-v1/analyze_qmt_robust_cell_soil.py
sha256:1b3d89d70a7fd8258810e0cee54225511e7ea7f8f7d46ff4719b421492669072

.tmp/quantx-research/qmt-robust-cell-soil-v1/qmt_robust_cell_soil_2021_2026_diagnostic.json
sha256:b578297e764332af62305b8c95a27307b5b1119a14860a45e1ee3da146b36e5b

.tmp/quantx-research/qmt-robust-cell-soil-v1/exp142_summary.md
```

无未来函数设定：复用 Exp130/140 的 QMT 日线面板，所有特征只使用信号日 T 及以前已完成日线；每个测试年只用 `< year` 样本拟合 cell 边界和 cell target encoding；T+1/T+6 open 仅用于历史标签和测试评价。

cell schema：

- `trend_liq_vol`: `ret20_rank/ret60_rank/amount_rank/range20_low_rank`
- `rotation_position`: `style_rotation_quality/ret20_rank/near_high20_rank/upper_wick_low_rank`
- `catchup_quality`: `style_lag_catchup/mom_lag5_rank/near_high20_rank/vol20_low_rank`
- `market_style`: `mkt_breadth20/mkt_near_high20/ret20_rank/amount_rank`
- `risk_position`: `mkt_breadth20/mkt_near_high20/range20_low_rank/near_high20_rank`

每维四分箱。每个 cell 计算：

```text
mean    = shrink(mean_exec)
robust  = shrink(0.55 * mean_exec + 0.35 * worst_year - 0.10 * year_std)
topedge = shrink(mean_exec + 0.015 * top_exec_edge - 0.010 * bad_exec_edge)
```

fold 输出：

| 测试年 | train rows | test rows | sessions | train exec mean | train entry ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139744 | 143064 | 48 | `-0.002301` | 99.53% |
| 2023 | 282808 | 149756 | 49 | `-0.002248` | 99.55% |
| 2024 | 432564 | 149402 | 48 | `-0.002091` | 99.62% |
| 2025 | 581966 | 153904 | 49 | `-0.002079` | 99.61% |
| 2026 | 735870 | 72806 | 23 | `-0.002103` | 99.59% |

### 142.3 结果

账户代理 leaderboard：

| 方案 | final | 2026 | 最大回撤 | 逐年正 | 平均选中 | 平均持有 | mean exec label | mean raw | entry ok | top exec | bad exec |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend_liq_vol_mean::top10` | 3.27x | `-1.91%` | `-24.17%` | 否 | 9.95 | 5.00 | `+0.003181` | `+0.007712` | 99.6% | 6.4% | 18.0% |
| `trend_liq_vol_topedge::top10` | 2.91x | `-1.91%` | `-24.17%` | 否 | 9.95 | 5.00 | `+0.002689` | `+0.007131` | 99.6% | 6.0% | 17.3% |
| `trend_liq_vol_topedge::top20` | 2.11x | `-6.50%` | `-30.58%` | 否 | 19.93 | 5.00 | `+0.001472` | `+0.005641` | 99.7% | 5.5% | 16.8% |
| `trend_liq_vol_mean::top20` | 2.09x | `-6.50%` | `-30.51%` | 否 | 19.93 | 5.00 | `+0.001423` | `+0.005608` | 99.7% | 5.5% | 17.3% |
| `ensemble_mean::top10` | 1.66x | `-8.46%` | `-30.26%` | 否 | 9.98 | 5.00 | `+0.000615` | `+0.004517` | 99.8% | 3.4% | 11.9% |
| `ensemble_robust::top20` | 1.31x | `-11.09%` | `-33.79%` | 否 | 19.97 | 5.00 | `-0.000567` | `+0.003348` | 99.9% | 3.1% | 11.8% |
| `market_style_mean::top20` | 1.23x | `-8.73%` | `-28.94%` | 否 | 19.91 | 5.00 | `-0.001053` | `+0.003272` | 99.6% | 4.2% | 15.8% |

最好 Top20 是 `trend_liq_vol_topedge::top20`/`trend_liq_vol_mean::top20`，仅约 2.1x，2026 均为 `-6.50%`。最好 Top10 也只有 3.27x 且 2026 为负，不满足主目标。

`trend_liq_vol_mean::top20` 年度账户收益：2022 `-23.69%`、2023 `+28.19%`、2024 `+7.35%`、2025 `+113.31%`、2026 `-6.50%`。2025 是主要贡献，2022/2026 都不能穿越。

`trend_liq_vol_mean::top20` 年度标签：2022 mean exec `-0.00552`、2023 `+0.00419`、2024 `+0.00087`、2025 `+0.00695`、2026 `-0.00061`。这不是稳定跨年土壤，而是明显状态依赖。

2026 fold 的 cell stats 显示，历史训练期确实存在看似高均值 cell，例如 `catchup_quality` best mean exec `+0.14405`、`rotation_position` best mean exec `+0.08661`；但这些 cell 稀疏或不稳，映射到测试年后并没有转成组合收益。更稳定、覆盖更大的 `trend_liq_vol` cell best mean exec 只有 `+0.00763`，robust score `+0.00271`，收益厚度天然有限。

### 142.4 反事实分析

第一反事实：如果稳定风格 cell 是缺失主引擎，`robust` 评分应优于 mean/topedge，并且 Top20 应逐年为正。实际 robust 版本普遍更弱，`trend_liq_vol_robust::top20` 2022/2024/2026 为负。

第二反事实：如果高均值小 cell 可穿越，`catchup_quality` 或 `rotation_position` 应能在测试年释放厚收益。实际它们没有进入 leaderboard 前列，说明训练期极高 cell 均值主要是稀疏/状态偶然，而不是稳定金矿。

第三反事实：如果横截面风格土壤比单票模型更鲁棒，2026 不应系统性为负。实际前列方案 2026 全部为负，说明 2026 风格环境与训练期 cell 奖励结构发生翻转。

第四反事实：如果问题来自入场可交易性，entry ok 提高应改善收益。实际 ensemble/cell 的 entry ok 接近 99.8%-99.9%，收益仍薄或为负，再次证明可交易性不是收益来源。

第五反事实：如果 Top10 强度可作为主线，Top20 应自然增厚。实际 Top10 最好 3.27x，Top20 只有 2.1x 且 2026 为负，不能靠缩 TopK 或尾部集中满足要求。

### 142.5 判定

`rejected_as_robust_cell_soil_not_enough`。

诚实 cell target encoding 没有找到稳定厚土壤。`trend_liq_vol` 说明中期趋势、流动性和波动分箱有局部解释力，但收益只有 2-3x，且 2022/2026 失败。后续不继续调分箱数、schema 组合、robust 权重或 topedge 系数。

Exp140-142 共同收敛出一个更高层结论：当前 QMT 日线股票级/风格 cell 空间，可以解释风险和局部状态，但无法自然生成几十倍级别的一周 Top20 alpha。下一步应真正换层级：从“选哪只股票”转为“组合如何利用已知弱 edge 放大资本效率且控制年份翻转”，例如账户级多 sleeve 风险预算、动态持仓形态、或重新审计历史上少数高收益正式曲线的非同源来源。

## 143. qmt_long_history_view_v1

### 143.1 问题

Exp140-142 说明当前 QMT 日线股票级/风格 cell 特征能解释局部状态，但不能自然生成几十倍级别一周 Top20 alpha。本轮回到一个更基础的问题：2021 起训是否太短，导致模型只学到近年局部结构？如果训练历史从 2021 扩到 2016 或 2010，在固定 2022-2026 评价窗口上是否能显著提高稳定性和 2026 前向？本轮不把 2010-2021 的收益计入目标，只把它们作为训练历史。

### 143.2 方法

产物：

```text
.tmp/quantx-research/qmt-long-history-view-v1/analyze_qmt_long_history_view.py
sha256:9786ec0d59397bed17e10e45abfae735e6a786ee2a807bd9757d58f4119ca6c5

.tmp/quantx-research/qmt-long-history-view-v1/qmt_long_history_view_2010_2026_diagnostic.json
sha256:27286ad52d9326ffd2f9cf134e6dee7a102301164e0d7bed12e91bb53f0eec12

.tmp/quantx-research/qmt-long-history-view-v1/exp143_summary.md
```

数据覆盖检查：`data/qlib_data_fixed/calendars/day.txt` 覆盖 2010-01-04 至 2026-07-10，共 4010 个交易日；raw QMT CSV 代表标的如 `SH600000`、`SH600519`、`SZ000001` 均覆盖 2010-01-04 至 2026-07-10，字段含 `open/high/low/close/preclose/volume/amount/turnover/pct_chg/is_st`。

无未来函数设定：复用 Exp130 QMT 日线风格/市场特征；所有特征只使用信号日 T 及以前已完成日线；标签为 T+1 open 到 T+6 open 的 `exec_label5_open` 和同日 top_exec；每个测试年只用 `train_start <= year < test_year` 的样本训练；评价窗口固定为 2022-2026。

对比训练起点：`2010`、`2016`、`2021`。模型：同参数 LightGBM reg/cls/blend，评价 Top10/Top20 账户代理。

fold 训练样本：

| train start | 2022 train rows | 2026 train rows | 2026 train exec mean | 2026 train entry ok |
| --- | ---: | ---: | ---: | ---: |
| 2010 | 1226701 | 1822860 | `-0.004928` | 96.84% |
| 2016 | 737605 | 1333764 | `-0.003662` | 98.06% |
| 2021 | 139785 | 735944 | `-0.002062` | 99.56% |

长历史训练标签更负，entry ok 更低，说明 2010-2020 的执行环境和近年不同，可能带来更保守的学习偏置。

### 143.3 结果

账户代理 leaderboard：

| train start | 方案 | final | 2026 | 最大回撤 | 逐年正 | 平均选中 | 平均持有 | mean exec label | mean raw | entry ok | top exec |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010 | `reg::top10` | 4.32x | `+2.67%` | `-23.94%` | 是 | 9.95 | 5.01 | `+0.005052` | `+0.009570` | 99.7% | 7.9% |
| 2010 | `reg::top20` | 3.75x | `-4.76%` | `-27.08%` | 否 | 19.94 | 5.01 | `+0.004099` | `+0.008443` | 99.8% | 7.0% |
| 2010 | `blend::top20` | 3.24x | `-2.00%` | `-30.60%` | 否 | 19.89 | 5.01 | `+0.003607` | `+0.008327` | 99.5% | 10.8% |
| 2016 | `blend::top20` | 2.52x | `+2.87%` | `-40.33%` | 否 | 19.86 | 5.01 | `+0.002138` | `+0.006969` | 99.4% | 10.5% |
| 2021 | `blend::top20` | 1.70x | `-3.03%` | `-47.58%` | 否 | 19.85 | 5.01 | `+0.000722` | `+0.005611` | 99.4% | 11.0% |
| 2021 | `reg::top20` | 1.39x | `-0.45%` | `-44.95%` | 否 | 19.89 | 5.01 | `+0.000056` | `+0.004210` | 99.6% | 7.6% |

2010 起训的 `reg::top10` 是少数逐年正结果：2022 `+34.99%`、2023 `+17.10%`、2024 `+61.48%`、2025 `+64.93%`、2026 `+2.67%`，但 Top10 不是主目标，且 final 只有 4.32x，远低于几十倍要求。

主口径 Top20：2010 `reg::top20` 为 3.75x，但 2026 `-4.76%`；2016 `blend::top20` 为 2.52x、2026 `+2.87%`，但 2022 `-12.01%`；2021 `blend::top20` 只有 1.70x，2022 和 2026 都为负。

2010 `reg::top20` 年度账户收益：2022 `+22.60%`、2023 `+28.14%`、2024 `+35.19%`、2025 `+85.28%`、2026 `-4.76%`。年度标签中 2026 mean exec 仍为正 `+0.00101`，但账户代理转负，说明 2026 的收益分布/路径和复利顺序仍脆弱。

特征重要性在三个训练起点都高度一致，仍由市场状态主导：`mkt_amount_disp20`、`mkt_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`mkt_breadth20`，再到 `range20_low_rank`、`amt_ratio20_rank`、`ret60_rank`、`amount_rank` 等。长历史没有发现新的独立结构，只是让 reg 对市场状态的估计更稳。

### 143.4 反事实分析

第一反事实：如果 2021 起训太短是主要瓶颈，2010/2016 起训应接近或超过 Exp40。实际 2010 起训确实显著优于 2021 起训，但最好 Top20 只有 3.75x，低于 Exp40 formal 8.77x/2026 `+19.18%`，更远低于目标。

第二反事实：如果越长历史越好，2010 应在 2026 明显更强。实际 2010 `reg::top20` 2026 为负，2016 `blend::top20` 2026 为正但开发期 2022 为负；历史窗口不是单调越长越好。

第三反事实：如果分类 top_exec 是关键，cls 应提升收益。实际所有 cls 账户几乎归零，虽然 top_exec rate 高，但 exec label 极负，说明右尾分类继续吸引坏执行/左尾风险。

第四反事实：如果 Top10 逐年正可以外推到主目标，Top20 应自然保持正收益。实际 2010 `reg::top10` 逐年正但 Top20 2026 转负，不能用缩 TopK 规避主目标。

第五反事实：如果长历史带来新结构，特征重要性应从旧市场状态迁移。实际重要性仍是市场离散度、成交额离散度和宽度，说明它只是更稳地学习旧结构，不是新 alpha 引擎。

### 143.5 判定

`rejected_with_useful_training_horizon_signal`。

长历史训练视野是有价值的工程/研究线索：2010 起训明显改善这套 QMT 日线风格模型，Top10 甚至逐年正。但它仍不满足用户目标：Top20 不逐年正，2026 前向弱，五年倍率只有 3-4x，低于 Exp40，也远低于几十倍到 100 倍要求。

后续不继续在同一特征/同一 LGBM 目标上微调训练起点、min_child_samples 或 reg/cls blend。更合理的使用方式是：把“长历史让 reg 更稳”作为底层风控/状态估计器，而不是主收益引擎。下一步应转向账户层资本效率和多 sleeve 结构，或者重新审计已有高收益正式曲线是否存在非同源 alpha，而不是继续从单一 QMT 日线股票排序里挤收益。

## 144. qmt_causal_sleeve_risk_budget_v1

### 144.1 问题

Exp140-143 连续说明，当前 QMT 日线单票排序、可执行右尾、稳健 cell 和长历史训练都无法自然生成几十倍级的一周 Top20 alpha。一个合理反事实是：也许问题不在某个单一候选器，而在账户层没有把多个弱 edge 做好风险预算。若 `leaf_mean`、风格资金流、style lag/catchup 等 sleeve 低相关且年度强弱可由历史已完成表现识别，那么因果动态预算应该至少接近最强单 sleeve，并改善 2026 或回撤。

本轮检验的是账户层结构，而不是降低 TopK 或尾部权重。每个 sleeve 仍取 Top20，formal-aware 入场/退出约束从一开始进入 ledger，预算只用已完成退出的历史 cohort，不允许看未来。

### 144.2 方法

产物：

```text
.tmp/quantx-research/qmt-causal-sleeve-risk-budget-v1/analyze_qmt_causal_sleeve_risk_budget.py
sha256:cda11b9f810c1ef9907d256bafa2ce6ecc38529ba632087473b63d36dfe714f4

.tmp/quantx-research/qmt-causal-sleeve-risk-budget-v1/qmt_causal_sleeve_risk_budget_2021_2026_diagnostic.json
sha256:2ef0dfdce3b6426990824d562ae4387a1e6c6f0cf38f7824a5bd867fd2856926

.tmp/quantx-research/qmt-causal-sleeve-risk-budget-v1/exp144_summary.md
```

复用模块：

- Exp130 `qmt_style_flow_rotation_v1`：QMT 日线风格资金流面板和 LGBM reg/cls/blend。
- Exp134 `qmt_adaptive_neighbor_leaf_memory_v1`：ExtraTrees leaf mean/toprate 和原型记忆。
- Formal-aware replay 口径：T 日 after-close 信号，T+1 open 买入，T+6 open 目标退出；入场涨停/停牌/缺失拒绝，退出跌停/缺失最多顺延 6 个交易日，成本 `0.00154`。

sleeve 集合全部为 Top20：

```text
style_reg, style_blend, style_lag_catchup, style_leader_confirm,
leaf_mean, leaf_blend, memory_blend,
leaf_reg_25, leaf_reg_50, leaf_blend_25,
lag_leaf_35, leader_leaf_35
```

账户对比：

1. `single::*`：每个 sleeve 单独账户。
2. `fixed_equal::all_sleeves`：所有 sleeve 等权。
3. `causal_online::ret60_ret120_drawdown_reject`：每个 T 日只用 `exit_session <= T` 的最近 60/120 期收益、下行波动、回撤、拒单率和平滑底仓分配权重。
4. `causal_year_start::prior_completed_years`：每年初只用 `exit_session < 当年 01-01` 的历史记录固定全年权重。

无未来函数设定：股票分数按年度 walk-forward，测试年只用 `< test_year` 训练；账户预算在信号日 T 只使用已完成退出的 cohort ledger；T+1 及之后价格只用于回放或已完成历史，不进入当日分数和权重形成。

### 144.3 结果

账户 leaderboard：

| 方案 | final | 2026 | 最差年 | 最大回撤 | 平均持仓 | 平均持有 | 逐年正 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `single::leaf_mean` | 3.55x | `+9.62%` | `+6.63%` | `-36.68%` | 19.94 | 5.00 | 是 |
| `single::leaf_reg_50` | 2.67x | `+4.83%` | `-8.41%` | `-31.59%` | 19.95 | 5.01 | 否 |
| `single::leaf_reg_25` | 2.39x | `+1.16%` | `-5.73%` | `-31.51%` | 19.96 | 5.01 | 否 |
| `single::leaf_blend_25` | 2.02x | `-1.13%` | `-3.82%` | `-32.96%` | 19.93 | 5.01 | 否 |
| `single::style_reg` | 1.66x | `-5.75%` | `-15.47%` | `-32.02%` | 19.94 | 5.01 | 否 |
| `causal_online::ret60_ret120_drawdown_reject` | 1.41x | `+2.65%` | `-13.60%` | `-34.40%` | 155.27 distinct / 19.88 weighted | 5.01 | 否 |
| `causal_year_start::prior_completed_years` | 1.36x | `+2.79%` | `-17.64%` | `-37.29%` | 155.27 distinct / 19.87 weighted | 5.01 | 否 |
| `fixed_equal::all_sleeves` | 1.17x | `+2.09%` | `-17.64%` | `-40.88%` | 155.27 distinct / 19.82 weighted | 5.01 | 否 |

`leaf_mean` 年度收益：2022 `+21.79%`、2023 `+18.11%`、2024 `+6.63%`、2025 `+111.11%`、2026 `+9.62%`。它仍是稳定薄信号，但远低于 Exp40 的 8.77x 和 2026 `+19.18%`。

`causal_online` 年度收益：2022 `-13.60%`、2023 `+7.26%`、2024 `-4.53%`、2025 `+54.78%`、2026 `+2.65%`。动态预算没有保住 `leaf_mean` 的逐年正收益，反而把不稳 sleeve 引入组合。

`causal_online` 的平均权重没有形成有效聚焦：`leaf_reg_50` 约 `10.29%`、`leaf_mean` 约 `10.00%`、`leaf_reg_25` 约 `9.78%`、`leader_leaf_35` 约 `9.76%`、`lag_leaf_35` 约 `9.27%`、`style_reg` 约 `8.92%`。历史 60/120 期表现无法提前识别哪些 sleeve 会在当年有效。

### 144.4 反事实分析

第一反事实：如果账户层预算是缺失收益弹性层，`causal_online` 应至少接近单一最强 sleeve，并改善 2026。实际它只有 1.41x，远低于 `leaf_mean` 3.55x，2026 也只有 `+2.65%`。

第二反事实：如果低相关 sleeve 能自然互补，等权组合应更平滑且不显著牺牲收益。实际等权只有 1.17x，2022/2024 为负，最大回撤 `-40.88%`，说明低重合不是可交易收益互补。

第三反事实：如果历史已完成 cohort 表现能因果识别年度风格，`causal_year_start` 不应在 2022/2024 亏损。实际 2022 `-17.64%`、2024 `-1.08%`，历史表现无法提前分辨薄 sleeve 年度翻转。

第四反事实：如果 2026 的 `style_lag_catchup` 是可学习 regime，历史预算应该能在 2026 前提高其权重。实际 `style_lag_catchup` 单 sleeve 2026 `+26.68%`，但 2022-2024 连续为负；`causal_online` 只给它约 `6.95%` 均权，这是后验风格，不是可稳定承接主线。

第五反事实：如果持仓数量或账户频率是瓶颈，多 sleeve distinct 持仓扩大到约 155 只应显著改善复利。实际资金加权持仓仍约 20，只是把弱信号摊薄，收益下降。

### 144.5 判定

`rejected_as_weak_sleeve_budget_not_enough`。

账户层多 sleeve 风险预算没有打开收益上限。已有 QMT 日线弱信号之间的低重合、2026 局部强项和历史均值差异，都不足以形成可因果利用的组合收益。后续不继续调 rolling 窗口、风险预算公式、底仓权重或 Exp130/134/142 sleeve 拼接。

这轮把“从单票排序切到账户层预算”的备选方向关掉了一大半。下一步应回到候选生成本身：Exp199 已说明真实 oracle Top20 机会极厚，但当前固定路径族和薄信号抓不住。后续更可能的方向是寻找能直接捕获横截面右尾机会的新 QMT-only 候选机制，而不是继续拼已有薄曲线。

## 145. qmt_local_relative_graph_ranker_v1

### 145.1 问题

Exp144 否定了把已有薄 sleeve 做账户层预算。Exp133/199 则说明市场真实 oracle Top20 机会极厚，但固定路径族和 T 日市场状态抓不住。Exp121/122/126/127 还说明内生主题、动态相关簇和冲击扩散都有局部上限，但固定 leader/follower、近期路由或单一 shock basket 都不能穿越。

本轮换一个候选生成角度：不预设谁是 leader/follower，而是对每只股票构造它在 T 日局部邻域中的相对位置，让模型学习“相对邻居强弱、滞后、冲击、共识”是否能预测 T+1 到 T+6 open 可执行收益。

### 145.2 方法

产物：

```text
.tmp/quantx-research/qmt-local-relative-graph-ranker-v1/analyze_qmt_local_relative_graph_ranker.py
sha256:515f151b40163f8584e2c2435e3313ab780b0aff5db0a3268ec1d895c1463399

.tmp/quantx-research/qmt-local-relative-graph-ranker-v1/qmt_local_relative_graph_ranker_2021_2026_diagnostic.json
sha256:4d41c5b72af470ece408262560b70c1c25f699b8524d9c2dd41c2b0eb003ced0

.tmp/quantx-research/qmt-local-relative-graph-ranker-v1/exp145_summary.md
```

复用 Exp130 QMT 日线面板。每个信号日 T 用 T 日可见横截面 rank 构造四类动态邻域：

- `trend`: `ret20_rank/ret60_rank/amount_rank`
- `flow`: `amt_ratio20_rank/close_strength_rank/upper_wick_low_rank`
- `position`: `near_high20_rank/ret5_rank/vol20_low_rank`
- `style`: `multi_style_strength/style_lag_catchup/style_leader_confirm`

每类邻域统计 peer 短中期强弱、成交额、近高、收盘强度、冲击比例，并生成相对收益残差、滞后修复、局部 follower setup、leader risk、quiet catchup 和 consensus strength。模型为年度 LGBM `reg/cls/blend`，并测试少量图特征手写分数和弱融合。

无未来函数设定：邻域 cell、peer 聚合和局部图特征只使用 T 日及以前的 QMT 日线数据；标签为 T+1 open 到 T+6 open 的 `exec_label5_open`；账户回放为 T+1 open 入场、T+6 open 目标退出，涨跌停/停牌约束沿用 Exp130；每个测试年只使用 `< test_year` 样本训练，2026 只用 2021-2025。

fold：

| 测试年 | train rows | test rows | sessions | train years | train exec mean | train entry ok |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 2022 | 139744 | 143064 | 48 | 2021 | `-0.002301` | 99.53% |
| 2023 | 282808 | 149756 | 49 | 2021-2022 | `-0.002248` | 99.55% |
| 2024 | 432564 | 149402 | 48 | 2021-2023 | `-0.002091` | 99.62% |
| 2025 | 581966 | 153904 | 49 | 2021-2024 | `-0.002079` | 99.61% |
| 2026 | 735870 | 72806 | 23 | 2021-2025 | `-0.002103` | 99.59% |

### 145.3 结果

账户代理 leaderboard：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均选中 | 平均持有 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top10` | 2.58x | `-4.86%` | `-6.01%` | `+0.73%` | `+96.42%` | `+45.57%` | 9.98 | 5.01 |
| `graph_blend_25::top10` | 2.36x | `-27.46%` | `-3.86%` | `+3.82%` | `+98.70%` | `+64.09%` | 9.94 | 5.00 |
| `reg::top20` | 2.24x | `-16.98%` | `-1.05%` | `+8.79%` | `+71.43%` | `+45.95%` | 19.98 | 5.01 |
| `graph_blend_25::top20` | 1.93x | `-28.17%` | `+1.34%` | `+0.09%` | `+61.08%` | `+64.85%` | 19.92 | 5.00 |
| `graph_quiet_catchup::top20` | 1.43x | `-16.26%` | `+12.06%` | `+40.45%` | `+13.39%` | `-4.34%` | 19.93 | 5.00 |
| `graph_follower::top20` | 0.09x | `-55.65%` | `-14.77%` | `-54.99%` | `-44.53%` | `-8.90%` | 19.72 | 5.01 |

主口径 `reg::top20` 虽然 2026 很强，但 2022/2023 为负，final 只有 2.24x，低于 Exp40，也低于 Exp134 `leaf_mean` 的 3.55x。

标签层同样显示强烈年度翻转：

| 方案 | mean exec | 2022 | 2023 | 2024 | 2025 | 2026 | mean raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top20` | `+0.002136` | `-0.003858` | `-0.000635` | `+0.003018` | `+0.002361` | `+0.018226` | `+0.006155` |
| `graph_blend_25::top20` | `+0.001368` | `-0.006416` | `-0.000293` | `+0.000155` | `+0.001535` | `+0.023327` | `+0.005416` |
| `graph_quiet_catchup::top20` | `-0.000599` | `-0.004363` | `+0.001492` | `+0.005916` | `-0.005802` | `+0.000287` | `+0.003508` |
| `graph_follower::top20` | `-0.013762` | `-0.017095` | `-0.005137` | `-0.018331` | `-0.020563` | `-0.001159` | `-0.008369` |

特征重要性前列仍由市场状态主导：`mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`mkt_breadth20`。图特征中靠前的是 `trend_peer_upper_wick_low`、`trend_peer_ret1`、`trend_peer_ret60`、`trend_peer_amount`、`trend_cell_size_rank`、`trend_peer_near_high`、`trend_local_consensus_strength`。模型确实使用了局部图特征，但主导收益结构仍是市场状态/风险环境。

### 145.4 反事实分析

第一反事实：如果局部邻域图是缺失主引擎，Top20 应在 2022-2026 逐年为正，并显著超过 Exp40 或至少 Exp134 `leaf_mean`。实际 `reg::top20` 只有 2.24x，2022/2023 为负，低于 Exp40 和 `leaf_mean`。

第二反事实：如果模型学到的是穿越主题结构，2026 强不应伴随开发期系统性翻车。实际 2026 强度非常明显，但 2022/2023 亏损，说明它更像 2026 风格暴露。

第三反事实：如果显式 follower/catchup 是答案，`graph_follower` 或 `graph_quiet_catchup` 应稳定。实际 `graph_follower` 几乎归零，`graph_quiet_catchup` 2026 转负，固定 follower/catchup 仍不可靠。

第四反事实：如果图特征替代了市场状态，特征重要性应由图特征主导。实际市场状态仍排在最前，图特征主要是辅助。

第五反事实：如果只是 TopK 宽度问题，Top10 强势应自然外推到 Top20。实际 Top10/Top20 都不逐年正，不能靠缩 TopK 通过要求。

### 145.5 判定

`rejected_with_2026_regime_signal`。

局部相对强弱图不是当前可合代码候选。它保留一个有价值的观察：2026 非常奖励“局部趋势/资金邻域 + 市场状态”的组合，但 2022/2023 明显反向。后续不继续调 cell 数、schema 权重或 graph blend 系数。

下一步应把 Exp145 当作 regime diagnostic：需要先解释“为什么 2026 局部图有效、而 2022/2023 反向”。如果不能因果识别这种 regime，继续在局部图上调模型只会变成后验追 2026。

## 146. qmt_local_graph_regime_diagnostic_v1

### 146.1 问题

Exp145 的局部相对强弱图出现了一个诱人的现象：`graph_blend_25::top20` 在 2026 达到 `+64.85%`，明显高于 Exp40 的 2026 `+19.18%`。但它在 2022 大亏 `-28.17%`，全期 final 只有 1.93x。因此不能直接把 2026 顺风当成策略方向。

本轮只做 regime 因果诊断：看 T 日可见的市场状态和入选组合画像，能否提前识别“什么时候应该相信局部图，什么时候应该回到 `reg` 或回避图信号”。这不是新选股器，也不把 hard gate 视为达标策略。

### 146.2 方法

产物：

```text
.tmp/quantx-research/qmt-local-graph-regime-diagnostic-v1/analyze_qmt_local_graph_regime_diagnostic.py
sha256:c2ee61104c85ba8d343e2e4a0d4169cb536952d99576095745b14369aa0f5ee7

.tmp/quantx-research/qmt-local-graph-regime-diagnostic-v1/qmt_local_graph_regime_diagnostic_2021_2026.json
sha256:5b2829fabc5f0db0c0b5ca9db59d58feb835fd8503519a85afbc1c0f11ee432c

.tmp/quantx-research/qmt-local-graph-regime-diagnostic-v1/exp146_summary.md
```

复用 Exp145 脚本，重新生成 QMT-only 日线 scored panel。每个调仓 session 对以下 Top20 方案做 formal-aware 单期账户回放：

```text
reg, graph_blend_25, graph_blend_50,
graph_quiet_catchup, graph_follower, graph_consensus
```

session 特征只用 T 日已知信息：市场离散度、成交额离散度、市场宽度、20 日市场中位收益、近高占比，以及各方案 Top20 入选组合的图冲击、图共识、leader risk、follower/catchup、相对趋势、短中期 rank、量能、近高、上影、收盘强度和与 `reg` 的重合度。

session 目标只用于历史训练和测试统计：

1. `graph_blend_25` 单期收益是否为正。
2. `graph_blend_25` 是否跑赢 `reg`。
3. `graph_blend_25 - reg` 的连续收益增量。

无未来函数设定：股票分数沿用 Exp145 年度 walk-forward；session 模型每个测试年只用 `< year` 的 session 训练，2026 只用 2021-2025；T+1 到 T+6 open 收益只作为历史标签或测试评估，不进入 T 日状态。

### 146.3 结果

账户和 session 复现：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 均期收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reg::top20` | 2.24x | `-16.98%` | `-1.05%` | `+8.79%` | `+71.43%` | `+45.95%` | `+0.004720` |
| `graph_blend_25::top20` | 1.93x | `-28.17%` | `+1.34%` | `+0.09%` | `+61.08%` | `+64.85%` | `+0.004010` |
| `graph_blend_50::top20` | 1.47x | `-39.29%` | `-2.60%` | `+1.10%` | `+58.23%` | `+55.07%` | `+0.002668` |
| `graph_quiet_catchup::top20` | 1.43x | `-16.26%` | `+12.06%` | `+40.45%` | `+13.39%` | `-4.34%` | `+0.001950` |
| `graph_follower::top20` | 0.09x | `-55.65%` | `-14.77%` | `-54.99%` | `-44.53%` | `-8.90%` | `-0.010212` |
| `graph_consensus::top20` | 0.01x | `-77.73%` | `-58.26%` | `-69.45%` | `-59.18%` | `-20.98%` | `-0.019717` |

`graph_blend_25` 相对 `reg` 的全期平均增量为 `-0.000710`，跑赢比例 `49.77%`。逐年看：

| 年份 | graph 年收益 | reg 年收益 | graph-reg 均期增量 | graph 跑赢比例 |
| --- | ---: | ---: | ---: | ---: |
| 2022 | `-28.17%` | `-16.98%` | `-0.002849` | 45.83% |
| 2023 | `+1.34%` | `-1.05%` | `+0.000534` | 57.14% |
| 2024 | `+0.09%` | `+8.79%` | `-0.002192` | 43.75% |
| 2025 | `+61.08%` | `+71.43%` | `-0.001562` | 51.02% |
| 2026 | `+64.85%` | `+45.95%` | `+0.006009` | 52.17% |

session 小模型结果：

| 诊断目标 | 均值 | 最差年 | 解释 |
| --- | ---: | ---: | --- |
| 图信号为正 AUC | `0.5439` | `0.4432` | 正收益 session 不能稳定识别 |
| 图信号跑赢 reg AUC | `0.5518` | `0.4321` | 图相对 reg 的优势不稳定 |
| 图信号跑赢 reg top-half target rate | `54.92%` | `41.67%` | 高预测 session 在坏年份反而更差 |
| delta regression top-half delta | `+0.001365` | `-0.003124` | 连续增量排序也年度翻转 |

2022/2023 弱年份与 2025/2026 强年份的特征对比显示：强年份中 `graph_blend_25_ret5_rank` 更高、市场宽度和成交额离散略高、近高占比略高，但 `graph_blend_25_graph_rel_trend_mean` 明显更低，组合与 `reg` 的重合度也略低。它不像一个稳定单调的 regime，更像不同年份由不同结构共同造成的后验结果。

session 模型特征重要性前列是 `mkt_near_high20`、`graph_blend_25_ret1_rank`、`reg_upper_wick_low_rank`、`mkt_breadth20`、`reg_style_lag_catchup`、`mkt_amount_disp20` 等。模型并非完全无信息，但信息不足以跨年度稳定使用。

### 146.4 反事实分析

第一反事实：如果 2026 图信号是能提前识别的市场状态，`graph_beats_reg` 分类器不应在某些年份 AUC 低于随机。实际最低 AUC 为 `0.4321`，说明训练期学到的图有效状态会在测试期反向。

第二反事实：如果局部图是缺失的主引擎，`graph_blend_25` 应长期跑赢 `reg`。实际全期平均增量为负，只在 2023 和 2026 局部占优。

第三反事实：如果 hard gate 可救这个方向，预测 top-half session 的增量应至少逐年不为负。实际 `graph_beats_reg` top-half delta 最差年 `-0.002784`，`delta_reg` top-half delta 最差年 `-0.003124`。

第四反事实：如果 follower/catchup 是 2026 强度的底层规律，固定 `graph_follower` 或 `graph_quiet_catchup` 应有穿越性。实际 `graph_follower` 几乎归零，`graph_quiet_catchup` 在 2026 为负。

第五反事实：如果市场状态足够解释，特征对比应该给出清晰单调条件。实际强弱年份差异混杂，既有宽度/成交额离散上升，也有相对趋势指标下降，不能形成稳健因果规则。

### 146.5 判定

`rejected_as_graph_regime_not_causally_identified`。

Exp145 的 2026 强表现不能外推为可合代码方向。局部图信号确实捕捉到 2026 某种顺风，但 T 日可见的市场状态和组合画像无法稳定提前识别它。后续不继续调图 cell、schema、graph blend 权重，也不把 session 小模型做成 hard gate。

下一轮应回到底层候选生成：Exp199 已证明真实 oracle Top20 机会极厚，但固定路径族、局部图、供需、韧性和风格迁移都没有捕获。更值得尝试的是直接学习历史右尾股票在 T 日之前的多日相对路径原型，用近邻/原型方式生成候选，而不是继续在已有薄因子上做状态路由。

## 147. qmt_causal_right_tail_path_prototype_v1

### 147.1 问题

Exp146 否定了“解释 2026 局部图顺风并因果路由”的方向，但 Exp133/199 仍然说明真实 oracle Top20 极厚。一个直接反事实是：也许我们不应再用固定风格/图特征解释右尾，而应从历史真实右尾股票本身学习 T 日以前的多日横截面路径原型。

此前 deep-learning 线做过 right-tail prototype/token ranker，但那些实验多为固定 train/valid/forward、pool500 或 token 模型，不是当前 QMT formal 主口径。本轮重做成年度因果、QMT-only、formal-aware 的路径原型实验：每个测试年只用过去年份真实 winner/loser 的路径拟合原型，再看当前股票是否更像历史 winner。

### 147.2 方法

产物：

```text
.tmp/quantx-research/qmt-causal-right-tail-path-prototype-v1/analyze_qmt_causal_right_tail_path_prototype.py
sha256:e4bb110e8ff7f9ddbba155697d5e4b0d6e1612135908418741842a9e1db020b8

.tmp/quantx-research/qmt-causal-right-tail-path-prototype-v1/qmt_causal_right_tail_path_prototype_2021_2026.json
sha256:36437722e0254fc77d3679800bfbe2cb4374a300e30f76e4d033eeb9f5d5a074

.tmp/quantx-research/qmt-causal-right-tail-path-prototype-v1/exp147_summary.md
```

复用 Exp130 QMT 日线风格资金流面板和 formal-aware open replay。新增路径特征：过去 `1/2/3/5/10` 个信号日前的收益 rank、成交额 rank、量能 rank、近高、收盘位置、上影、低波、区间压缩，以及若干 1-5 日和 3-10 日的路径变化量。

每个测试年：

1. 只取 `< year` 的历史样本。
2. 在每个历史 session 内按未来一周 `exec_label5_open` 取 top/bottom `10/20/40` 只股票。
3. 用历史 winner/loser 路径特征分别拟合 MiniBatchKMeans 原型，`k=16/32/64`。
4. 测试年 T 日股票分数为当前路径到 winner 原型最大相似度减去到 loser 原型最大相似度。
5. 另外训练一个 `path_reg`，输入基础特征、路径特征和原型分数；`proto_best_blend` 只在训练集内选择相关性最高的原型列，与 `path_reg` 弱融合。

无未来函数设定：原型、标准化参数、最佳原型列和 `path_reg` 都只用 `< test_year` 历史；2026 只用 2021-2025。T+1/T+6 open 收益只用于历史标签和测试评估，不进入 T 日特征。

### 147.3 结果

账户 leaderboard 前排：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均持仓 | 平均持有 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proto_best_blend::top10` | 3.15x | `+40.11%` | `+14.23%` | `+4.31%` | `+72.77%` | `+9.07%` | 9.99 | 5.00 | `+0.003669` |
| `proto_best_blend::top20` | 2.69x | `+17.30%` | `+16.67%` | `+11.13%` | `+72.04%` | `+2.98%` | 19.96 | 5.00 | `+0.002761` |
| `proto_w20_l20_k32::top10` | 1.89x | 混合 | 混合 | 混合 | 混合 | `-0.00%` | 9.93 | 5.00 | `+0.000618` |
| `proto_w40_l40_k16::top20` | 1.80x | 混合 | 混合 | 混合 | 混合 | `-0.76%` | 19.91 | 5.00 | `+0.000595` |
| `path_reg::top20` | 0.43x | `-53.21%` | `-13.75%` | `-10.53%` | `+98.02%` | `+25.90%` | 19.96 | 5.00 | `-0.002176` |

`proto_best_blend::top20` 是一个干净的稳定弱信号：五年逐年正，平均持仓和持有周期合规，最大回撤 `-21.35%`。但 final 只有 2.69x，2026 只有 `+2.98%`，显著低于 Exp40 的 8.77x 和 2026 `+19.18%`。

标签层：

| 方案 | mean exec | 2022 | 2023 | 2024 | 2025 | 2026 | mean raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proto_best_blend::top20` | `+0.002761` | `+0.003068` | `+0.002560` | `+0.002018` | `+0.002851` | `+0.003904` | `+0.006722` |
| `proto_best_blend::top10` | `+0.003669` | `+0.006941` | `+0.002082` | `+0.000775` | `+0.003578` | `+0.006452` | `+0.007603` |
| `path_reg::top20` | `-0.002176` | `-0.015023` | `-0.003899` | `-0.002136` | `+0.005369` | `+0.012150` | `+0.002152` |

纯 `path_reg` 是重要负证据：它在 2025/2026 很强，但 2022-2024 连续反向，账户 final 只有 0.43x。说明监督点预测仍会学到强烈年度风格漂移，不能直接把路径特征/原型分数塞进 LGBM 期待它自动学穿越规律。

2026 的 `path_reg` 特征重要性仍由市场状态主导：`mkt_ret20_median`、`mkt_amount_disp20`、`mkt_disp20`、`mkt_near_high20`、`mkt_breadth20` 排最前；原型列 `score_proto_w10_l10_k64`、`score_proto_w40_l40_k64`、`score_proto_w20_l20_k64` 被使用，但只是辅助。

### 147.4 反事实分析

第一反事实：如果历史右尾路径原型是缺失主引擎，纯原型或原型融合应接近 Exp40。实际最好的 Top20 只有 2.69x，2026 仅 `+2.98%`。

第二反事实：如果监督 `path_reg` 能从路径原型中学到厚右尾，2022-2024 不应连续大亏。实际 `path_reg::top20` 为 `-53.21%/-13.75%/-10.53%`，说明它更像 2025/2026 风格暴露。

第三反事实：如果纯右尾原型可迁移，单个 `proto_w*_l*_k*` 不应在 2026 普遍转负。实际多数纯原型 2026 为负，说明“像历史 winner”并不等价于“当前可交易右尾”。

第四反事实：如果 `proto_best_blend` 的逐年正来自厚 alpha，2026 也应保持明显收益斜率。实际 2026 只有 `+2.98%`，更像稳定弱分段均值。

第五反事实：如果只是 TopK 过宽，Top10 应足以接近目标。实际 Top10 final 也只有 3.15x，不能靠缩 TopK 或调融合权重满足要求。

### 147.5 判定

`rejected_with_stable_but_too_thin_signal`。

历史右尾路径原型能形成稳定弱信号，但不能打开收益上限。后续不继续调原型数量、winner/loser 样本数、路径滞后窗口或 `proto/path_reg` 融合权重。

这轮留下的核心观察是：全市场单票路径原型会被市场状态和年度风格漂移压住。下一轮应转向更贴近 A 股轮动机制的横截面群体结构：只用 QMT 日线动态构造可交易主题簇/扩散簇，在簇内学习 leader/follower 的相对位置和扩散阶段，而不是继续对全市场单票直接做右尾原型。

## 148. qmt_dynamic_theme_cluster_ranker_v1

### 148.1 问题

Exp121 的内生 leader/follower `reg` 有稳定标签信号但收益太薄；Exp126/127 的动态相关簇和 shock diffusion 说明手写 follower、lag repair、leader continue 都不穿越。Exp147 又说明全市场单票右尾路径原型会被市场状态和年度风格漂移压住。

本轮换一种方式使用“主题/概念轮动”思想：不用静态行业/概念表，也不手写谁是 leader/follower，而是在每个 T 日用 QMT 日线动态构造主题簇，把 `session#cluster` 当作 query group，让模型在簇内学习相对排序。核心反事实是：A 股短周期机会可能主要存在于局部主题簇内部的相对位置，而不是全市场单票统一排序。

### 148.2 方法

产物：

```text
.tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/analyze_qmt_dynamic_theme_cluster_ranker.py
sha256:d3199eea062fa596b3c34b71b442c67f7cce365e1a3f060ea485953ae226a9cc

.tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/qmt_dynamic_theme_cluster_ranker_2021_2026.json
sha256:916c127799c69b344e55b45f8c5891ad8150a34264b3cc213804c3dc2efc0a2b

.tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/exp148_summary.md
```

复用 Exp130 QMT 日线风格资金流面板和 formal-aware open replay。每个信号日 T：

1. 用 `ret20/ret60/amount/near_high/amt_ratio` 选 48 个强势 anchor。
2. 用过去 60 日日收益相关，把每只股票分配到最相关 anchor，形成动态主题簇。
3. 计算簇规模、簇均值、簇内 breadth/dispersion、股票相对簇均值、簇内 rank、与 anchor 相关性，以及少量 lag repair/mid follower/leader quality 诊断分数。
4. 年度 walk-forward 训练全局 `reg` 和簇内 `LGBMRanker`。ranker 的 query group 是 `session#cluster_id`，标签是同簇内未来 `exec_label5_open` 五分位。
5. 最终仍在全市场按分数取 Top10/Top20，并做 formal-aware T+1 open 回放。

无未来函数设定：anchor、收益相关、簇分配、簇内 rank 和所有簇统计只使用 T 日及以前数据；簇内五分位标签只用于历史训练；每个测试年只用 `< year` 训练，2026 只用 2021-2025；T+1/T+6 open 收益只用于训练标签和测试评估。

### 148.3 结果

账户 leaderboard：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均持仓 | 平均持有 | 最大回撤 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_rank::top10` | 8.05x | `+29.32%` | `+8.78%` | `+116.06%` | `+125.10%` | `+17.63%` | 9.94 | 5.01 | `-32.12%` | `+0.007566` |
| `cluster_rank::top20` | 3.98x | `+6.34%` | `+24.53%` | `+38.40%` | `+110.01%` | `+3.31%` | 19.91 | 5.01 | `-33.26%` | `+0.004725` |
| `blend::top10` | 3.26x | `+17.44%` | `+33.96%` | `-6.24%` | `+99.92%` | `+10.51%` | 9.96 | 5.01 | `-37.31%` | `+0.003761` |
| `blend::top20` | 2.17x | `-4.24%` | `+22.75%` | `-15.92%` | `+102.55%` | `+8.62%` | 19.94 | 5.01 | `-39.76%` | `+0.001950` |
| `reg::top20` | 1.49x | `-5.82%` | `-1.58%` | `-22.72%` | `+73.88%` | `+19.41%` | 19.92 | 5.00 | `-46.16%` | `+0.000245` |

固定诊断规则仍失败：`cluster_mid_follower::top20` final `0.01x`、2026 `-23.52%`；`cluster_leader_quality::top20` final `0.02x`、2026 `-20.97%`；`cluster_lag_repair::top20` final `0.81x`、2026 `-16.82%`。这说明收益不是来自手写 follower/leader，而是来自簇内学习排序。

标签层：

| 方案 | mean exec | 2022 | 2023 | 2024 | 2025 | 2026 | mean raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_rank::top10` | `+0.007566` | `+0.006229` | `+0.001236` | `+0.016381` | `+0.005893` | `+0.009011` | `+0.011811` |
| `cluster_rank::top20` | `+0.004725` | `+0.001656` | `+0.004005` | `+0.007311` | `+0.006245` | `+0.004033` | `+0.008760` |
| `reg::top20` | `+0.000245` | `-0.000717` | `-0.001195` | `-0.004231` | `+0.002651` | `+0.009540` | `+0.004552` |

特征重要性前列仍有市场状态：`mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`mkt_breadth20`。但簇特征也进入前列：`cluster_ret60_mean`、`cluster_amount_mean`、`cluster_amt_ratio_mean`、`cluster_near_high_mean`、`cluster_breadth20`、`cluster_corr_anchor`、`cluster_rel_amount`。相比 Exp126/127，这更像真正的横截面群体结构信号。

### 148.4 反事实分析

第一反事实：如果动态主题簇只是旧相关簇的重复，`cluster_rank` 不应显著强于 Exp126/127。实际 Top10 达到 8.05x 且逐年正，说明“簇内 query 排序学习”有真实增量。

第二反事实：如果固定 follower/leader 是核心，手写 `cluster_mid_follower/leader_quality/lag_repair` 应同步变强。实际它们几乎归零或亏损，说明不能回到固定规则调参。

第三反事实：如果该方向已经满足用户目标，Top20 应自然接近几十倍或至少超过 Exp40。实际 Top20 只有 3.98x，2026 只有 `+3.31%`，收益厚度不够。

第四反事实：如果靠 Top10 即可，`cluster_rank::top10` 仍只有 8.05x，低于 Exp40 8.77x，且远低于几十倍到 100 倍，不符合“不靠缩 TopK 卡要求”的约束。

第五反事实：如果全局 reg 与簇内 ranker 互补，blend 应超过 `cluster_rank`。实际 blend 变差，说明全局 reg 的年度风格漂移稀释簇内 ranker。

### 148.5 判定

`rejected_but_cluster_ranker_is_promising_diagnostic`。

Exp148 不是可合代码候选，但它是近期最有研究价值的 QMT-only 方向之一：簇内 query 排序明显优于手写簇规则，也明显优于全局 reg。当前失败点不是“没有信号”，而是 Top10 到 Top20 急剧变薄，且 Top10 仍未达到用户收益目标。

后续不继续调手写 follower/leader 权重，也不把 Top10 当成过关。下一轮应诊断 `cluster_rank` 的 Top10/Top20 差异：簇级机会厚度、簇内 top-decile/bottom-decile 分离度、Top20 后 10 名失败结构、是否需要簇级多样性/每簇限额，且这些改动必须保持因果和平均持仓 `>5`。

## 149. qmt_cluster_rank_thickness_diagnostic_v1

### 149.1 问题

Exp148 的 `cluster_rank::top10` final `8.05x`、逐年正、2026 `+17.63%`，但主口径 Top20 只有 `3.98x`、2026 `+3.31%`。本轮不尝试把 Top10 包装成策略，而是只做厚度诊断：Top20 变薄到底来自簇集中度、最强簇押注失败、还是 ranker 在 11-20 名学到了错误结构。

### 149.2 方法

产物：

```text
.tmp/quantx-research/qmt-cluster-rank-thickness-diagnostic-v1/analyze_qmt_cluster_rank_thickness_diagnostic.py
sha256:4b104065db1f4e03b63d2311696af097278600cc6045a222c870e532a9f7dfe9

.tmp/quantx-research/qmt-cluster-rank-thickness-diagnostic-v1/qmt_cluster_rank_thickness_diagnostic_2021_2026.json
sha256:17ab7fe327d62fc4f2ec34ca508857c279f985200c859150b0761bdc036f3d98

.tmp/quantx-research/qmt-cluster-rank-thickness-diagnostic-v1/exp149_summary.md
```

复用 Exp148 的 QMT 日线动态主题簇、簇内 query ranker 和 formal-aware open replay。重新生成 2022-2026 年度 walk-forward OOS scored panel，每个测试年只用 `< year` 历史训练，2026 只用 2021-2025。

诊断内容：

1. `1_5/6_10/11_15/16_20` 分数切片的未来 open exec/raw 表现。
2. Top10/Top20 的簇集中度和唯一簇数量。
3. `cap1/cap2/cap3` 每簇限额、`top10_plus_cap2_fill` 和 `best_cluster_first_top20` 反事实账户。
4. Top10 与第 11-20 名的 T 日可见特征均值对比。

无未来函数设定：分数来自年度 walk-forward OOS；反事实选股只使用 OOS score 与 T 日可见簇特征；未来 T+1/T+6 open 收益只用于评估和账户回放，不用于构造分数、筛选阈值或选择参数。

### 149.3 结果

账户反事实：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均持仓 | 平均持有 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_top10` | 8.05x | `+29.32%` | `+8.78%` | `+116.06%` | `+125.10%` | `+17.63%` | 9.94 | 5.01 | `-32.12%` |
| `plain_top20` | 3.98x | `+6.34%` | `+24.53%` | `+38.40%` | `+110.01%` | `+3.31%` | 19.91 | 5.01 | `-33.26%` |
| `cap1_top20` | 3.56x | `+7.36%` | `-1.92%` | `+67.53%` | `+96.95%` | `+13.66%` | 19.91 | 5.01 | `-34.78%` |
| `cap2_top20` | 3.77x | `+8.24%` | `+1.31%` | `+57.41%` | `+109.85%` | `+4.27%` | 19.91 | 5.01 | `-36.06%` |
| `cap3_top20` | 4.05x | `+9.95%` | `+8.02%` | `+51.95%` | `+102.51%` | `+3.67%` | 19.91 | 5.01 | `-33.65%` |
| `top10_plus_cap2_fill` | 3.88x | `+7.64%` | `+7.44%` | `+45.56%` | `+108.10%` | `+3.65%` | 19.91 | 5.01 | `-34.48%` |
| `best_cluster_first_top20` | 1.45x | `-0.66%` | `-8.86%` | `-17.09%` | `+106.28%` | `+2.49%` | 19.91 | 5.02 | `-42.15%` |

关键切片：

| 排名切片 | mean exec | 2022 | 2023 | 2024 | 2025 | 2026 | mean raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1_10` | `+0.007566` | `+0.006229` | `+0.001236` | `+0.016381` | `+0.005893` | `+0.009011` | `+0.011811` |
| `11_20` | `+0.001885` | `-0.002917` | `+0.006774` | `-0.001758` | `+0.006597` | `-0.000945` | `+0.005710` |
| `16_20` | `-0.000214` | `-0.004227` | `+0.005475` | `-0.006793` | `+0.005348` | `-0.002083` | `+0.003559` |

Top10 与 11-20 的画像对比：

- Top10 mean exec `+0.007566`，11-20 mean exec `+0.001885`。
- Top10 entry ok `99.59%`，11-20 entry ok `99.77%`，因此不是入场拒单导致变薄。
- 11-20 相比 Top10 更高：`ret60_rank` `+0.098204`、`ret20_rank` `+0.097304`、`cluster_rank_ret20` `+0.096424`、`cluster_rank_ret60` `+0.094691`、`near_high20_rank` `+0.078738`、`cluster_leader_quality` `+0.059023`、`cluster_overheat_risk` `+0.050545`。
- Top10 反而更“簇内滞后”：`cluster_rel_ret60` Top10 `-0.140180` vs 11-20 `-0.037756`，`cluster_rel_ret20` Top10 `-0.178338` vs 11-20 `-0.077691`，`cluster_rel_ret5` Top10 `-0.195031` vs 11-20 `-0.109012`，`cluster_rel_near_high` Top10 `-0.177382` vs 11-20 `-0.098324`。

### 149.4 反事实分析

第一反事实：如果 Top20 失败只是因为簇集中度太高，每簇限额应修复厚度。实际 `cap3_top20` final 仅 `4.05x`、2026 `+3.67%`，只比 plain Top20 略好，远低于 Top10 和目标。

第二反事实：如果最强主题簇本身就是 alpha，应优先从最强簇拿满。实际 `best_cluster_first_top20` final 仅 `1.45x`，2022/2023/2024 为负，说明“押最强簇”不是答案。

第三反事实：如果 11-20 只是随机噪声，画像不应有系统差异。实际 11-20 系统性更强、更近高、更 leader、更过热；Top10 反而是强簇内部相对滞后的修复结构。这说明 ranker 学到了有用结构，但排序目标没有把 Top20 厚度校准好。

第四反事实：如果入场可达性解释变薄，11-20 的 entry ok 应明显更差。实际 11-20 entry ok 还略高，因此瓶颈在标签/结构本身，不在 formal 拒单。

### 149.5 判定

`diagnostic_complete_top20_thickness_not_solved`。

本轮没有得到可部署策略，但明确了后续方向：不要做每簇限额、Top10 补位、单簇优先或手写 hard gate。下一轮应把“强簇内相对滞后、低过热、可修复”的结构转成训练目标，例如 residual target、样本权重或双头弱融合，让模型自然增厚 Top20，而不是账户层补丁。

## 150. qmt_cluster_lag_repair_ranker_v1

### 150.1 问题

Exp149 说明 Top10 的有效结构更像强主题簇内部的相对滞后/低过热修复，而不是追强 leader；第 11-20 名更强、更近高、更过热，导致 Top20 变薄。本轮把这个观察放进训练目标，而不是做后验过滤：如果修复结构是真的，应能自然增厚 Top20，并保持 2026 前向。

### 150.2 方法

产物：

```text
.tmp/quantx-research/qmt-cluster-lag-repair-ranker-v1/analyze_qmt_cluster_lag_repair_ranker.py
sha256:c6e5aebc672899eba01b6b4eb53e5125d6d76bf989b2a254fc05c2b419d6f505

.tmp/quantx-research/qmt-cluster-lag-repair-ranker-v1/qmt_cluster_lag_repair_ranker_2021_2026.json
sha256:5a59ade1558df4134344871e85de612843e9fc5972e58cd7c48d244bab3de1fe

.tmp/quantx-research/qmt-cluster-lag-repair-ranker-v1/exp150_summary.md
```

复用 Exp148 动态主题簇。新增 T 日可见修复特征：`repair_lag_depth`、`repair_confirmation`、`repair_overheat_penalty` 和 `repair_prior_rank`。训练四类头：

1. `cluster_rank`：簇内五分位 ranker。本轮特征集中加入修复特征，因此与 Exp148 原始 `cluster_rank` 不完全等价。
2. `repair_rank`：按 `repair_prior_rank` 加权的簇内 ranker。
3. `residual_reg`：学习 `exec_label5_open - cluster_query_mean` 的簇内残差收益。
4. `weighted_reg`：按 `repair_prior_rank` 加权学习真实 `exec_label5_open`。

固定融合包括 `rank_repair_35`、`rank_residual_35`、`rank_weighted_35` 和 `repair_stack`。所有权重实验前固定，不用 2026 后验选参。

无未来函数设定：动态簇、修复特征、簇内 rank 只用 T 日及以前；每个测试年只用 `< year` 训练，2026 只用 2021-2025；未来 open 收益只用于历史训练标签和 OOS 评估。

### 150.3 结果

账户 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank_repair_35::top10` | 9.45x | `+8.83%` | `+8.83%` | 9.92 | 5.01 | `-30.21%` | `+0.008312` |
| `rank_residual_35::top10` | 5.60x | `+2.35%` | `+2.35%` | 9.94 | 5.01 | `-36.80%` | `+0.006246` |
| `repair_stack::top10` | 5.13x | `+8.60%` | `+8.60%` | 9.97 | 5.00 | `-33.09%` | `+0.006578` |
| `cluster_rank::top20` | 4.43x | `+14.68%` | `+12.53%` | 19.88 | 5.01 | `-35.21%` | `+0.004943` |
| `rank_residual_35::top20` | 4.47x | `+0.49%` | `+0.49%` | 19.92 | 5.01 | `-32.71%` | `+0.005141` |
| `rank_repair_35::top20` | 5.40x | `-4.40%` | `-4.40%` | 19.89 | 5.00 | `-30.57%` | `+0.006201` |
| `repair_prior::top20` | 1.17x | `-13.62%` | `-13.62%` | 19.95 | 5.00 | `-38.13%` | `-0.000901` |

标签层：`rank_repair_35::top10` mean exec `+0.008312`，明显高于本轮 `cluster_rank::top10` 的 `+0.005129`，但 2026 标签均值只有 `+0.006561`，正式账户 2026 也只有 `+8.83%`。Top20 的 `rank_repair_35` mean exec 达 `+0.006201`，但 2026 标签均值降到 `+0.001148`，正式账户转负。

修复先验分桶从低到高的 mean exec 分别约 `-0.003804/-0.002643/-0.001851/-0.000999/-0.001022`。高修复分桶只是少亏，并没有形成正收益。

特征重要性仍被市场状态主导，前列包括 `mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`mkt_breadth20`，说明模型并没有摆脱市场风险状态对收益的支配。

### 150.4 反事实分析

第一反事实：如果修复画像本身是 alpha，`repair_prior` 应直接赚钱。实际 `repair_prior::top20` final 仅 `1.17x`、2026 `-13.62%`。

第二反事实：如果加权修复样本能自然增厚 Top20，`rank_repair_35::top20` 应逐年正。实际它 2026 `-4.40%`，说明修复加权强化了开发期结构，却破坏前向。

第三反事实：如果簇内残差目标能解决厚度，`rank_residual_35::top20` 应明显超过基线。实际 final `4.47x` 与 `cluster_rank::top20` 接近，2026 仅 `+0.49%`。

第四反事实：如果本轮是新的主 alpha，市场状态特征不应继续压倒簇/修复特征。实际重要性仍集中于市场宽度、离散度、近高和成交额离散，说明收益仍主要受状态约束。

### 150.5 判定

`rejected_before_formal_candidate`。

Exp150 有价值的结论是：簇内修复结构能改善 Top10 排序，但不是独立 alpha，且不能自然增厚 Top20。后续不继续调 `repair_prior` 权重、样本权重强度或融合比例。

下一步应跳出“在同一动态主题簇里继续重排股票”的框架，转向更上层的信号日机会面：先识别某个调仓日到底是哪类横截面结构有厚右尾，再用对应结构生成候选，而不是让一个统一 ranker 同时处理所有市场状态。

## 151. qmt_session_opportunity_expert_selector_v1

### 151.1 问题

Exp150 证明在同一个簇内 ranker 里修补滞后/过热结构不能自然增厚 Top20。本轮转向更高一层：把不同股票排序方式视为“结构专家”，先判断某个信号日哪类横截面机会更厚，再用对应专家出候选。

核心反事实是：如果 A 股短周期收益来自动态市场结构，那么统一 ranker 会把互相冲突的结构平均掉；信号日级别的专家选择可能比股票级别统一排序更接近真实机制。

### 151.2 方法

产物：

```text
.tmp/quantx-research/qmt-session-opportunity-expert-selector-v1/analyze_qmt_session_opportunity_expert_selector.py
sha256:9df0a2869f6a9386ddda21339db7a34865229e7ef9715da38f428058b29a628b

.tmp/quantx-research/qmt-session-opportunity-expert-selector-v1/qmt_session_opportunity_expert_selector_2021_2026.json
sha256:547a2f898e67303ba0eb9f805aad81a8dd6c9c364dbfc6216f694b1d4952d112

.tmp/quantx-research/qmt-session-opportunity-expert-selector-v1/exp151_summary.md
```

复用 Exp150 股票层 OOS 分数，构造 8 个专家：`cluster_rank`、`repair_rank`、`residual_reg`、`weighted_reg`、`rank_repair_35`、`rank_residual_35`、`rank_weighted_35`、`repair_stack`。

每个 `session#expert` 用 T 日可见字段构造机会特征：市场状态、专家分数分布、专家 Top20 组合画像、簇集中度、修复/过热/相对强弱画像。用历史 OOS expert 表现训练 `LGBMRegressor` 预测该 expert 当日 Top20 的未来 `exec_label5_open`。

无未来函数设定：股票层分数是年度 walk-forward OOS；selector 每个测试年只用之前年份已完成 OOS expert 表现训练。2022 冷启动固定 `cluster_rank`，2023 只用 2022，2026 只用 2022-2025。未来收益只用于历史训练目标和 OOS 评估。

### 151.3 结果

因果 selector：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selector_top3_merge::top10` | 6.60x | `+8.71%` | `+8.71%` | 9.95 | 5.00 | `-40.27%` | `+0.007558` |
| `selector_top1::top10` | 6.29x | `+8.19%` | `+8.19%` | 9.94 | 5.01 | `-41.13%` | `+0.006968` |
| `selector_top2_merge::top10` | 5.81x | `+7.99%` | `+7.99%` | 9.95 | 5.01 | `-40.16%` | `+0.006791` |
| `selector_all_weighted::top10` | 5.74x | `+13.29%` | `-2.50%` | 9.95 | 5.01 | `-36.55%` | `+0.006527` |
| `selector_all_weighted::top20` | 4.15x | `-2.00%` | `-2.00%` | 19.91 | 5.01 | `-29.33%` | `+0.004779` |
| `selector_top1::top20` | 4.12x | `-4.39%` | `-4.39%` | 19.93 | 5.00 | `-40.46%` | `+0.005011` |
| `selector_top3_merge::top20` | 3.70x | `-2.05%` | `-2.05%` | 19.94 | 5.01 | `-38.00%` | `+0.004716` |
| `selector_top2_merge::top20` | 3.56x | `-2.24%` | `-2.24%` | 19.94 | 5.01 | `-39.14%` | `+0.004505` |

selector 预测质量：2023 相关性 `0.0213`，2024 `0.0008`，2025 `0.0893`，2026 `0.1091`。2026 选择出来的 expert Top20 真实均值只有 `+0.000229`，低于全专家平均 `+0.001981`。

非因果 oracle：

| 方案 | final | 2022 | 2023 | 2024 | 2025 | 2026 | 平均持仓 | 平均持有 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_best_expert_top20` | 299.75x | `+249.09%` | `+147.43%` | `+361.70%` | `+370.49%` | `+59.75%` | 19.94 | 5.00 | `-15.52%` |

oracle 只是上限诊断，不可交易，但它说明专家空间不是没有收益，而是因果识别失败。

### 151.4 反事实分析

第一反事实：如果专家空间没有上限，oracle 不会高。实际 oracle `299.75x`，而且 2026 `+59.75%`，说明同一批专家在不同信号日之间确实有巨大切换价值。

第二反事实：如果当前 T 日特征足以识别机会，因果 selector 应接近 oracle，至少 Top20 不应 2026 转负。实际 Top20 全部 2026 为负，说明缺少关键可见状态变量。

第三反事实：如果多专家合并能规避选择误差，`selector_all_weighted::top20` 应穿越。实际 2026 `-2.00%`，说明简单合并会把错误专家一并带进组合。

第四反事实：如果只是 TopK 太宽，Top10 逐年正确实更平，但收益仍只有 `5.8x-6.6x`，低于 Exp40 和目标，不符合用户对厚收益的要求。

第五反事实：如果 2026 只是训练样本少，预测相关性应完全失效；实际 2026 相关性略正但极值选择失败，说明方向上有微弱信息，但排序校准远远不够。

### 151.5 判定

`rejected_before_formal_candidate`。

不合代码，不继续调 selector 模型、专家合并数量或融合权重。下一步应专门研究 oracle 胜出专家的 T 日可见差异：哪些市场/横截面路径状态能稳定区分 `cluster_rank`、`repair_stack`、`rank_repair`、`residual_reg` 等专家什么时候获胜。只有找到可解释且穿越的专家胜率状态，才值得重做因果选择器。

## 152. qmt_oracle_expert_state_diagnostic_v1

### 152.1 问题

Exp151 的 expert oracle 上限极高，但因果 selector 失败。本轮检验失败是否来自目标形式：与其预测 expert 绝对收益，不如预测 expert 相对当日均值的优势、是否为当日最佳、以及 session 内排序。

如果目标形式是主要问题，Exp152 应显著接近 oracle；如果仍失败，则说明当前 T 日可见状态变量本身不足以识别专家切换。

### 152.2 方法

产物：

```text
.tmp/quantx-research/qmt-oracle-expert-state-diagnostic-v1/analyze_qmt_oracle_expert_state_diagnostic.py
sha256:ea24f12763dcde8cc2aa7b199626e4920d07cb5c7a9980e542d40704e2c040e3

.tmp/quantx-research/qmt-oracle-expert-state-diagnostic-v1/qmt_oracle_expert_state_diagnostic_2021_2026.json
sha256:ad34ad4b0c6b4e9c933f775e40c6b651b21f91b4adc29a41017273150a5da474

.tmp/quantx-research/qmt-oracle-expert-state-diagnostic-v1/exp152_summary.md
```

复用 Exp151 的 expert ledger 和 Exp150 股票层 OOS 分数。每个测试年只使用之前年度训练，2026 只用 2022-2025。

新增三个 selector：

1. `relreg`：预测 `expert_top20_exec - session_mean_expert_exec`。
2. `wincls`：预测 expert 是否为当日最佳。
3. `ranker`：在每个 session 内学习 expert 表现五分位排序。

`consensus` 固定融合三个预测的 session 内 rank。全部选择逻辑只使用 T 日可见 market/score/portfolio profile 特征，未来收益只作为历史训练目标和 OOS 评估。

### 152.3 结果

账户 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `relreg_top2_merge::top10` | 7.56x | `+4.00%` | `+4.00%` | 9.93 | 5.01 | `-30.21%` | `+0.007372` |
| `relreg_top3_merge::top10` | 7.24x | `+10.47%` | `+10.47%` | 9.95 | 5.01 | `-36.80%` | `+0.007567` |
| `consensus_top2_merge::top10` | 5.50x | `+22.00%` | `+10.20%` | 9.96 | 5.01 | `-24.07%` | `+0.006372` |
| `consensus_top2_merge::top20` | 5.14x | `+4.79%` | `+4.79%` | 19.95 | 5.01 | `-26.83%` | `+0.006178` |
| `ranker_top1::top20` | 3.46x | `+0.11%` | `+0.11%` | 19.94 | 5.00 | `-28.20%` | `+0.004713` |
| `wincls_top1::top20` | 3.14x | `+0.76%` | `+0.76%` | 19.91 | 5.01 | `-29.56%` | `+0.003562` |
| `relreg_top2_merge::top20` | 5.22x | `-1.80%` | `-1.80%` | 19.89 | 5.01 | `-30.57%` | `+0.005997` |

非因果 oracle 仍为 `299.75x`，2026 `+59.75%`，平均持仓 `19.94`，平均持有 `5.00`，最大回撤 `-15.52%`。

oracle 胜出 expert 全期分布：`weighted_reg` 42 次、`repair_rank` 29 次、`cluster_rank` 28 次、`rank_repair_35` 28 次、`repair_stack` 28 次、`residual_reg` 25 次、`rank_weighted_35` 20 次、`rank_residual_35` 17 次。2026 也没有单一主导 expert。

best-vs-other 可见特征差异很弱：最大标准化差异是 `unique_clusters` `-0.185`、`score_all_std` `-0.177`、`score_top20_max` `-0.176`、`score_top20_min` `-0.174`、`score_top20_mean` `-0.172`、`max_cluster_share` `+0.150`。

### 152.4 反事实分析

第一反事实：如果 Exp151 只是目标形式不对，相对优势/分类/ranker 应明显接近 oracle。实际最佳 Top20 只有 `5.14x`，说明目标形式不是主瓶颈。

第二反事实：如果存在固定胜出专家，oracle 胜出次数应高度集中。实际 8 个 expert 均有胜出，无法用固定 expert 替代。

第三反事实：如果当前可见状态足以解释 oracle，best-vs-other 特征差异应大且稳定。实际最大标准化差异不足 `0.2`，说明状态信息不足。

第四反事实：如果共识能弥补模型误差，`consensus_top2_merge::top20` 应接近 Exp40 或更高。实际仅 `5.14x`，只是弱稳定化。

第五反事实：如果 Top10 可作为替代，`consensus_top2_merge::top10` 2026 有 `+22.00%`，但 final 仅 `5.50x`，仍远低于几十倍目标，不能作为达标口径。

### 152.5 判定

`rejected_before_formal_candidate`。

Exp152 不合代码，也不继续调 selector/ranker/分类器/consensus 权重。当前专家空间上限虽高，但缺少能因果识别专家切换的 QMT 日线状态变量。

下一轮应换底层 alpha 方向：不要继续围绕现有专家做选择器。更值得探索的是能直接产生厚 Top20 的股票级结构，例如更接近筹码/供需/涨停资金行为的日线代理，或者从更长窗口横截面路径中识别“即将成为市场主线”的早期主题。

## 153. qmt_early_mainline_cluster_opportunity_v1

### 153.1 问题

Exp151/152 说明 expert selector 难以因果识别，但 oracle 上限很高。Exp153 暂时离开 expert selector，转向“早期主线识别”：先在动态主题簇层预测未来一周是否有厚右尾，再把高机会簇与股票层 OOS ranker 结合生成候选。

核心反事实：如果 A 股短周期收益来自概念/行业主线轮动，那么先找未来有右尾的主题簇，应该比全市场统一股票排序更容易形成厚 Top20。

### 153.2 方法

产物：

```text
.tmp/quantx-research/qmt-early-mainline-cluster-opportunity-v1/analyze_qmt_early_mainline_cluster_opportunity.py
sha256:024ce891893d444e4e89be7d776f3d45a9c8f64e9d23467b77e2ee8b0caf0351

.tmp/quantx-research/qmt-early-mainline-cluster-opportunity-v1/qmt_early_mainline_cluster_opportunity_2021_2026.json
sha256:48c6a7ef8379dae9302f968a7018eab959f281cf3edfdb8e2e7fcc6a4f3a402c

.tmp/quantx-research/qmt-early-mainline-cluster-opportunity-v1/exp153_summary.md
```

复用 Exp148 动态主题簇和年度 walk-forward 股票层 `cluster_rank` 分数。每个 `session#cluster` 构造 T 日可见簇级特征：簇规模、anchor 相关、簇内收益/成交额/近高/低波/上影均值、early lag/leader/overheat/repair share 和市场状态。

簇机会目标包括：

1. `target_top5_exec`：簇内未来一周 `exec_label5_open` 前 5 只均值。
2. `target_rel_top5_exec`：相对当日所有簇均值的优势。
3. `target_is_opportunity`：当日簇机会前 20% 分类。

每个测试年只用 `< year` 的簇样本训练，2026 只用 2021-2025。测试选股只用 OOS 股票分数和 OOS 簇机会分数。

### 153.3 结果

账户 leaderboard：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 平均簇数 | 最大回撤 | mean exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_rank::top10` | 8.05x | `+17.63%` | `+8.78%` | 9.94 | 5.01 | 8.06 | `-32.12%` | `+0.007566` |
| `top_clusters_stock_rank::top10` | 4.02x | `+28.82%` | `+4.12%` | 9.97 | 5.00 | 3.00 | `-39.47%` | `+0.005152` |
| `cluster_rank::top20` | 3.98x | `+3.31%` | `+3.31%` | 19.91 | 5.01 | 13.36 | `-33.26%` | `+0.004725` |
| `top_clusters_stock_rank::top20` | 2.47x | `+14.91%` | `-1.04%` | 19.94 | 5.00 | 5.00 | `-37.37%` | `+0.002551` |
| `top_clusters_lag_repair::top10` | 2.46x | `-5.43%` | `-7.06%` | 9.97 | 5.00 | 3.00 | `-39.64%` | `+0.002908` |
| `early_mainline::top20` | 1.77x | `+4.70%` | `-8.59%` | 19.95 | 5.00 | 1.27 | `-36.94%` | `+0.000819` |

簇级机会预测非常强：2026 `score_cluster_abs/rel/cls/blend` 对 `target_top5_exec` 的相关性分别为 `0.6806/0.6818/0.6151/0.6407`。2022-2025 也基本在 `0.60-0.74` 区间。

非因果簇 oracle 显示热点簇确实存在：全市场簇 `target_top5_exec` 均值约 `+0.0963`，每个 session 的 top5 簇均值约 `+0.2356`；2026 top5 簇均值约 `+0.2600`。

### 153.4 反事实分析

第一反事实：如果之前失败是因为找不到热点主题簇，簇机会模型应明显改善账户。实际簇级预测强，但账户不改善，说明热点簇识别不是主瓶颈。

第二反事实：如果集中到预测热点簇能自然增厚收益，`early_mainline` 或 `top_clusters_stock_rank` 应超过 `cluster_rank`。实际 Top20 从 `3.98x` 降到 `2.47x/1.77x`。

第三反事实：如果“早期主线 + lag repair”是正确形态，`top_clusters_lag_repair` 应穿越。实际 Top20 final `1.87x` 且 2026 `-7.74%`。

第四反事实：如果 Top10 可以替代，`top_clusters_stock_rank::top10` 2026 `+28.82%`，但 final 只有 `4.02x`，低于原始 `cluster_rank::top10` 的 `8.05x`，不能作为达标策略。

第五反事实：如果只是簇机会模型需要微调，至少标签层 Top20 mean exec 应提高。实际 `top_clusters_stock_rank::top20` mean exec `+0.002551`，低于 `cluster_rank::top20` 的 `+0.004725`。

### 153.5 判定

`rejected_before_formal_candidate`。

Exp153 不合代码，也不继续调簇机会模型、top cluster 数、per-cluster 配额或 lag repair 权重。动态主题簇能解释市场右尾在哪里，但无法自然转成 formal Top20 厚收益。

下一轮应暂时离开动态主题簇主线，转向更底层的股票级可执行结构，尤其是 QMT 日线可构造的供需/筹码/涨停资金行为代理，目标是直接提高单票一周可执行 Top20 厚度。

## 154. qmt_executable_transition_ranker_v1

### 154.1 问题

Exp153 说明动态主题簇能找出热点在哪里，但热点簇内股票排序兑现不了 formal Top20 厚收益。Exp129/125 又已经否定了固定供需再平衡和温和右尾不拥挤。本轮换成股票级“可执行过渡态”：寻找那些有资金痕迹、压力修复和趋势过渡，但还没有极端追涨或明显买入失败风险的股票。

核心反事实：如果 A 股短周期收益来自“分歧后重新转强”而不是“已被一致预期定价的强封板”，那么可执行过渡态应该比简单供需/追强形态更能形成 Top20 一周收益厚度。

### 154.2 方法

产物：

```text
.tmp/quantx-research/qmt-executable-transition-ranker-v1/analyze_qmt_executable_transition_ranker.py
sha256:3e30fdb2579452c30c762cca603dd399faf7ece54011091febd7c5b43bc7a74e

.tmp/quantx-research/qmt-executable-transition-ranker-v1/qmt_executable_transition_ranker_2021_2026.json
sha256:42be08800105a7c43ce7e2422304d26e0bf10660666a754d7c0aa93a460e06fd

.tmp/quantx-research/qmt-executable-transition-ranker-v1/exp154_summary.md
```

复用 Exp129 的 QMT 日线 open 执行面板和回放函数。新增 T 日可见派生特征：压力释放后的修复、非极端强势、趋势连续、短回踩修复、供需修复平衡、左尾风险、拥挤风险、买入失败可见风险和市场风险调整后的过渡态。

模型包括 `return_reg`、`weighted_reg`、`right_tail`、`left_tail_risk`、`entry_fail_risk`，以及固定 rank blend：`risk_adjusted_reg`、`tail_transition`、`executable_blend`。每个测试年只用 `< year` 样本训练，2026 只用 2021-2025。

无未来函数边界：所有特征只使用 T 日及以前的 QMT/qlib 日线 OHLCV/VWAP、成交额、ST 和横截面统计；T+1/T+6 open 只用于训练标签、拒单/延迟卖出和 OOS 评估；不使用新闻、公告、龙虎榜、ETF、北向、融资融券或静态概念表。

### 154.3 结果

面板 808,676 行，scored 668,932 行。2022-2026 共 217 个信号日。

账户 leaderboard 关键项：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `quality_without_chase::top20` | 1.55x | `+2.61%` | `+2.61%` | 18.10 | 5.01 | `-20.80%` | `-0.007998` | 1.16x |
| `quality_without_chase::top10` | 1.44x | `+6.71%` | `+2.88%` | 8.78 | 5.01 | `-27.95%` | `-0.010346` | 0.94x |
| `risk_adjusted_reg::top10` | 1.73x | `-1.56%` | `-11.18%` | 10.00 | 5.00 | `-31.67%` | `+0.000873` | 1.26x |
| `risk_adjusted_reg::top20` | 1.71x | `-6.08%` | `-15.06%` | 19.99 | 5.00 | `-33.05%` | `+0.000777` | 1.26x |
| `weighted_reg::top20` | 1.66x | `+1.84%` | `-28.97%` | 19.90 | 5.00 | `-36.69%` | `+0.000383` | 1.04x |

`quality_without_chase::top20` 是唯一逐年正 Top20，但标签层很差：全期 mean raw `+0.008395`，mean exec `-0.007998`，entry ok 只有 90.60%。2026 mean exec 仍为 `-0.005926`。

`risk_adjusted_reg::top20` 更接近可执行账户：entry ok 99.95%，mean exec `+0.000777`。但 2024 mean exec `-0.003847`、2026 `-0.000183`，账户层 2026 `-6.08%`，不能穿越。

特征重要性仍集中在市场风险和通用横截面：`mkt_disp20`、`mkt_amount_disp20`、`mkt_near_high20`、`mkt_ret20_median`、`mkt_breadth20`、`range20_low_rank`、`amount_rank`、`ret60_rank`。新增过渡态特征没有成为厚主信号。

### 154.4 反事实分析

第一反事实：如果“可执行过渡态”是缺失的厚 alpha，`quality_without_chase` 应在标签层和账户层同时强。实际账户逐年正但 final 只有 1.55x，且 mean exec 为负，说明它更多是平滑效果，不是收益发动机。

第二反事实：如果主要瓶颈是 T+1 买不到，`risk_adjusted_reg` 在 entry ok 接近 100% 后应显著变厚。实际 Top20 final 1.71x，2026 为负，说明可执行性不是唯一瓶颈。

第三反事实：如果左尾/拥挤风险头能稳定过滤失败股，`transition_minus_risk` 和 `executable_blend` 应优于单收益头。实际 `transition_minus_risk::top20` 只有 1.17x、2026 `-14.64%`。

第四反事实：如果该方向学到的是鲁棒股票级机制，去掉最强 3 次调仓后仍应有显著复利。实际最好逐年正 Top20 去最强 3 次只剩 1.16x。

第五反事实：如果它能承接 Exp153 的主题机会，2026 不应弱。实际最稳定 Top20 2026 只有 `+2.61%`，模型化 Top20 2026 为负，说明股票级过渡态没有把热点主题转成可执行 Top20 收益。

### 154.5 判定

`rejected_before_formal_candidate`。

Exp154 不合代码，不继续调过渡态手工权重、风险 blend、可执行性惩罚或 TopK。可执行性和不过度追涨能略平滑曲线，但收益土壤不厚。

下一轮应换更底层的候选生成机制。当前更值得研究的是“当日全市场右尾机会的形态本身”：用历史 OOS 真实 winner/loser 反推信号日横截面形态和候选池生成，而不是继续在单票日线形态上小修补。

## 155. qmt_right_tail_cohort_subspace_generator_v1

### 155.1 问题

Exp133 说明真实 oracle Top20 每期都很厚，但固定路径族抓不住；Exp134 说明股票级近邻/叶子均值稳定但太薄；Exp155 因此换成更高一层的假设：A 股一周右尾不是孤立股票，而是某类历史 winner cohort 的共同横截面方向。

核心反事实：如果历史 winner cohort 的共同方向代表了隐含概念/主题/风格审美，那么当前股票落入这些 winner subspace 时，应自然形成厚 Top20，而不是只在 Top10 局部有效。

### 155.2 方法

产物：

```text
.tmp/quantx-research/qmt-right-tail-cohort-subspace-generator-v1/analyze_qmt_right_tail_cohort_subspace_generator.py
sha256:d4bd64b3315ee72074d3651a313a191c8f736d45e227c9d6b5a6f32644c251ed

.tmp/quantx-research/qmt-right-tail-cohort-subspace-generator-v1/qmt_right_tail_cohort_subspace_generator_2021_2026.json
sha256:d8033bf064da3f5d245f305cc2380571808b51ad91c58fa7fce15660249f9535

.tmp/quantx-research/qmt-right-tail-cohort-subspace-generator-v1/exp155_summary.md
```

复用 Exp129 的 QMT 日线 open 执行面板和回放函数。每个年度 fold 内，只用过去年份构造 winner cohort library；2026 只用 2021-2025。

每个训练 session：

1. 当日股票特征做横截面标准化。
2. `exec_label5_open` Top20/Top30 作为 winner cohort，Bottom20/Bottom30 作为 loser cohort。
3. 构造 `winner_mean`、`winner_mean - pool_mean`、`winner_mean - loser_mean`、`loser_mean` 和市场状态。
4. 对历史 cohort 聚类形成 subspace library。
5. 测试日股票计算 `subspace_max/top3/weighted/contrast/margin/stable/neg_distance/market_fit`。
6. 同时评估纯 subspace 分数和增强 LGBM：`aug_reg`、`aug_right_tail`、`aug_blend`。

无未来函数边界：测试年不使用当年 winner 信息；T 日特征只使用 T 日及以前 QMT/qlib 日线 OHLCV/VWAP、成交额、ST 和横截面统计；T+1/T+6 open 只用于训练标签和 OOS 回放；不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态概念表或消息数据。

### 155.3 结果

面板 808,676 行，scored 668,932 行。2022-2026 共 217 个信号日。

| 测试年 | 训练行数 | 测试行数 | 信号日 | library rows | clusters | library mean payoff | min years/cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 48 | 24 | 24 | `+30.35%` | 1 |
| 2023 | 282,808 | 149,756 | 49 | 36 | 36 | `+30.45%` | 1 |
| 2024 | 432,564 | 149,402 | 48 | 36 | 36 | `+28.71%` | 1 |
| 2025 | 581,966 | 153,904 | 49 | 36 | 36 | `+29.60%` | 1 |
| 2026 | 735,870 | 72,806 | 23 | 36 | 36 | `+29.95%` | 1 |

训练侧 winner cohort 很厚，但 `min_years_per_cluster=1` 说明很多 cohort 类型只在单一年份出现，更像历史碎片，而非稳定主题结构。

账户 leaderboard 关键项：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aug_reg::top10` | 1.90x | `+15.81%` | `-31.10%` | 9.95 | 5.01 | `-40.44%` | `+0.001434` | 1.12x |
| `aug_blend::top15` | 1.44x | `+0.42%` | `-15.02%` | 14.97 | 5.01 | `-37.51%` | `+0.000383` | 0.88x |
| `subspace_weighted::top10` | 1.31x | `-13.16%` | `-13.16%` | 9.99 | 5.00 | `-31.99%` | `-0.000153` | 0.91x |
| `subspace_weighted::top20` | 1.12x | `-12.07%` | `-13.59%` | 19.98 | 5.00 | `-35.95%` | `-0.000928` | 0.80x |
| `subspace_blend::top20` | 1.11x | `-9.54%` | `-19.94%` | 19.99 | 5.00 | `-35.76%` | `-0.001056` | 0.77x |
| `aug_reg::top20` | 1.25x | `+6.76%` | `-29.23%` | 19.93 | 5.00 | `-48.07%` | `-0.000528` | 0.77x |
| `base_reg::top20` | 1.26x | `+12.36%` | `-26.56%` | 19.94 | 5.01 | `-48.61%` | `-0.000747` | 0.81x |

`subspace_weighted::top20` 标签拆解：2022/2023 mean exec 分别为 `+0.001256`、`+0.000618`，但 2024 `-0.003141`、2025 `-0.001743`、2026 `-0.002427`。entry ok 接近 100%，说明失败不是买不到，而是 OOS 方向本身不厚。

特征重要性仍主要是市场风险和通用横截面：`mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`mkt_breadth20`、`range20_low_rank`、`ret60_rank`、`amount_rank`、`vwap_support_rank`。新增 subspace 特征没有进入前 30。

### 155.4 反事实分析

第一反事实：如果历史 winner cohort 的共同方向稳定，纯 subspace 分数应自然增厚 Top20。实际 `subspace_weighted::top20` mean exec 为负，2026 `-12.07%`，说明共同方向不能迁移。

第二反事实：如果失败只是纯 subspace 表达不够，增强 LGBM 应利用 subspace 特征显著超过 base。实际 `aug_reg::top20` 1.25x，几乎不超过 `base_reg::top20` 1.26x，且 subspace 特征不在重要性前列。

第三反事实：如果 cohort library 抓住稳定主题/概念隐空间，每个 cluster 应跨多年复现。实际 `min_years_per_cluster=1`，很多 cohort 是单年碎片，风格漂移严重。

第四反事实：如果 Top10 局部亮点可以代表方向，`aug_reg::top10` 至少应逐年正。实际 2022 大亏 `-31.10%`，最大回撤 `-40.44%`，不能用缩 TopK 替代目标。

第五反事实：如果问题主要在交易实现，subspace 标签层应显著为正但账户被打掉。实际 entry ok 很高，mean exec 仍为负，说明不是交易损耗主导。

### 155.5 判定

`rejected_before_formal_candidate`。

Exp155 不合代码，不继续调 winner cohort 的 TopN、KMeans K 值、market fit、subspace blend 或增强模型。它进一步支持一个重要结论：真实 oracle Top20 很厚，但 QMT 日线里可稳定迁移的“赢家共同横截面形态”很弱。

下一轮不应继续围绕历史 winner 形态做相似度/聚类/原型。更值得换到另一个维度：从“预测谁是赢家”转向“预测什么结构会被市场连续定价”。如果仍限定 QMT 日线，下一轮应考虑多日横截面转移矩阵/排序流：研究股票在 rank 分布中的迁移速度、拥挤扩散和资金审美从低位到高位的连续流，而不是静态 cohort 形状。

## 156. qmt_rank_flow_transition_v1

### 156.1 问题

Exp155 否定了静态 winner cohort/subspace：历史右尾很厚，但 winner 的共同横截面形态不能稳定迁移。本轮改问一个更动态的问题：A 股短线收益是否来自资金审美在多日横截面 rank 分布里的连续迁移，例如成交额 rank 先上移、价格 rank 稳定 climb、低位到中位再到高位扩散，或者强势 rank 持续但未过热。

核心反事实：如果多日 rank-flow 是真实机制，那么手工 flow prior 至少应比静态 subspace 更厚；如果只有模型强、手工 prior 弱，则可能只是旧 LGBM 继续学习市场状态；如果 Top20 或 2026 不穿越，就不能作为用户要求的鲁棒方向。

### 156.2 方法

产物：

```text
.tmp/quantx-research/qmt-rank-flow-transition-v1/analyze_qmt_rank_flow_transition.py
sha256:fec3f933793af896e125cfc5cd8c1ff4ff86d78c9c33ee38eab4d08e1eb0e1fa

.tmp/quantx-research/qmt-rank-flow-transition-v1/qmt_rank_flow_transition_2021_2026.json
sha256:f1d9e8fa59bdf8601d2c7a050b3c56f57dfd277995303e42d0dfe7e96ca9dd26

.tmp/quantx-research/qmt-rank-flow-transition-v1/exp156_summary.md
```

复用 Exp129 的 QMT 日线 open 执行面板和回放函数。新增特征只使用 T 日收盘后已完成的 QMT/qlib 日线 OHLCV/VWAP、成交额和 ST：

1. `ret5/ret20/ret60/amount/amt_ratio/near_high/range_low/vwap_support` 的 3/5/10/20 日横截面 rank 迁移。
2. `ret5_accel_5_10`、`ret20_accel_5_20`、`amount_accel_5_10` 等 rank acceleration。
3. 过去 5/10 日 rank 上移比例。
4. `amount_leads_price`、`steady_rank_climb`、`low_to_mid_transition`、`mid_to_high_transition`、`high_rank_persistence`、`pullback_after_climb_repair`、`flow_exhaustion_risk` 等股票级 flow prior。
5. 市场级 flow breadth：全市场 climb breadth、mid-to-high share、high-rank fail share、amount-leads share 和 flow dispersion。

模型包括 `return_reg`、`weighted_reg`、`right_tail`、`top_quintile`、`left_tail_risk`，以及固定 rank blend：`flow_blend`、`flow_risk_adjusted_reg`、`ml_flow_blend`。每个测试年只用 `< year` 样本训练，2026 只用 2021-2025。

无未来函数边界：所有 rank flow 只用 T 日及以前的历史 rank；T+1/T+6 open 只用于训练标签、拒单/延迟卖出和 OOS 回放；不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态概念表或其他消息数据。

### 156.3 结果

面板 808,676 行，scored 668,932 行。2022-2026 共 217 个信号日。

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flow_risk_adjusted_reg::top10` | 1.65x | `-5.77%` | `-10.01%` | 9.98 | 5.00 | `-32.49%` | `+0.000376` | 1.26x |
| `flow_risk_adjusted_reg::top15` | 1.58x | `-5.89%` | `-9.03%` | 14.98 | 5.00 | `-32.51%` | `+0.000289` | 1.22x |
| `flow_risk_adjusted_reg::top20` | 1.52x | `-7.37%` | `-8.93%` | 19.98 | 5.00 | `-31.68%` | `+0.000151` | 1.16x |
| `ml_flow_blend::top10` | 1.41x | `+5.61%` | `-28.90%` | 9.94 | 5.01 | `-57.58%` | `-0.001174` | 0.93x |
| `return_reg::top20` | 1.07x | `+0.29%` | `-38.34%` | 19.93 | 5.00 | `-52.02%` | `-0.001572` | 0.72x |
| `rank_flow_prior::top20` | 0.34x | `-12.09%` | `-45.68%` | 19.57 | 5.01 | `-70.34%` | `-0.008452` | 0.28x |
| `amount_leads_price::top20` | 0.41x | `-22.32%` | `-30.26%` | 19.74 | 5.01 | `-60.16%` | `-0.006846` | 0.27x |
| `steady_rank_climb::top20` | 0.20x | `-22.69%` | `-49.18%` | 19.16 | 5.01 | `-80.01%` | `-0.012187` | 0.16x |

最好的 `flow_risk_adjusted_reg::top20` 年度收益为：

```text
2022 +17.47%
2023 +10.30%
2024  -8.93%
2025 +39.12%
2026  -7.37%
```

标签层同样薄：全期 mean exec 只有 `+0.000151`，2024 mean exec `-0.002505`，2026 mean exec `-0.000691`。entry ok 为 `99.93%`，说明失败不是买不到，而是方向本身不厚。

特征重要性前列仍是 `mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`、`mkt_amount_leads_share`、`mkt_high_rank_fail_share`、`amount_rank`、`ret60_rank`、`range20_low_rank`。新增 flow 特征只有市场级 flow 被模型较多使用，股票级 flow prior 没有形成正向主信号。

### 156.4 反事实分析

第一反事实：如果 rank-flow 是缺失机制，手工 `rank_flow_prior/steady_rank_climb/amount_leads_price` 至少应在标签层为正。实际三者 Top20 mean exec 都显著为负，说明该方向更像追已完成迁移或过热结构。

第二反事实：如果成交可达性是主要问题，entry ok 接近 100% 的 `flow_risk_adjusted_reg` 应显著变厚。实际 Top20 只有 1.52x 且 2026 为负，说明不是执行口径打掉了 alpha。

第三反事实：如果模型能从 flow 中学到稳定结构，`ml_flow_blend` 应超过 `return_reg` 且逐年更稳。实际 `ml_flow_blend::top20` final 0.96x，2022-2024 连续为负，只在 2025/2026 局部有效。

第四反事实：如果 Top10 局部强能代表方向，`flow_risk_adjusted_reg::top10` 应接近 Exp40 或至少通过 2026。实际 Top10 final 1.65x、2026 `-5.77%`，不能用缩 TopK 或头部化替代目标。

第五反事实：如果市场级 flow 是主收益来源，重要性高的 `mkt_amount_leads_share/mkt_high_rank_fail_share` 应转成强账户。实际它们只是帮助模型解释风险状态，不能打开收益上限。

### 156.5 判定

`rejected_before_formal_candidate`。

Exp156 不合代码，不继续调 rank slope、climb count、flow blend、exhaustion 权重或 TopK。它否定的是“多日横截面排序流手工/弱学习”这个具体方向，不否定多日路径本身；Exp40 仍说明路径学习有价值。

下一轮应换到更直接对齐金融低信噪比的目标函数：使用同日正负样本的对比/排序损失，避免普通点预测在低信噪比环境里学习到市场状态解释器。一个可验证方向是在 QMT-only open 面板上构造 session 内 pairwise/contrastive ranker：每个信号日只比较未来可执行收益右尾与左尾股票，让模型学习“同一天为什么这只比那只更值得买”，并重点检查 Top20 是否自然变厚。

## 157. qmt_session_contrastive_ranker_v1

### 157.1 问题

Exp156 说明，多日 rank-flow 手工方向会追到已完成迁移或过热结构；普通 reg/cls 又容易退回市场状态解释器。用户也明确提醒金融低信噪比下 loss 应尽量采用正负样本对比。本轮因此不预测绝对收益，也不直接做全局 right-tail 分类，而是在每个历史信号日内学习“右尾股票相对左尾股票为什么更强”。

核心反事实：如果低信噪比下的主要问题是目标函数错位，同日正负样本 pairwise contrast 应该比普通 reg/cls 更稳定，并在 Top20 上自然变厚；如果它只是换皮，特征重要性仍会被市场状态主导，账户层仍会低于 Exp156/Exp40。

### 157.2 方法

产物：

```text
.tmp/quantx-research/qmt-session-contrastive-ranker-v1/analyze_qmt_session_contrastive_ranker.py
sha256:eca2a53f0b5f9cccf7de60185aad5ec1bd3cc46f1afa4e82cf30cc65e4a5ff09

.tmp/quantx-research/qmt-session-contrastive-ranker-v1/qmt_session_contrastive_ranker_2021_2026.json
sha256:aca080f9a18fde5cb16030f52a92c64ac1d856ae463db4fe959d5cd0a13a8bd8

.tmp/quantx-research/qmt-session-contrastive-ranker-v1/exp157_summary.md
```

复用 Exp129 的 QMT 日线 open 执行面板和回放函数。所有特征只使用 T 日及以前 QMT/qlib 日线 OHLCV/VWAP、成交额和 ST。

训练目标：

1. 每个训练 session 内按 `exec_label5_open` 排序。
2. 正样本为当日右尾约 top 8%，负样本为 bottom 25%。
3. 构造 `x_pos - x_neg`，标签为 1；反向 `x_neg - x_pos`，标签为 0。
4. 用 `LGBMClassifier` 学 pairwise preference。
5. 测试时不使用未来标签，使用两类因果分数：
   - `anchor_loser/mid/margin`：当前股票相对训练集历史 loser/mid/winner anchor 的胜率或边际。
   - `duel_pool`：测试日内只用 T 日特征进行候选互相 duel，不使用测试标签。

每个测试年只用 `< year` 样本训练，2026 只用 2021-2025。T+1/T+6 open 只用于训练标签、拒单/延迟卖出和 OOS 回放；不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态概念表或其他消息数据。

### 157.3 结果

面板 808,676 行，scored 668,932 行。2022-2026 共 217 个信号日。

| 测试年 | 训练行数 | 测试行数 | pair rows | pair sessions | mean pair gap | winner anchor | loser anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 254,016 | 49 | `+0.2113` | 7,013 | 27,928 |
| 2023 | 282,808 | 149,756 | 502,848 | 97 | `+0.2024` | 14,192 | 56,522 |
| 2024 | 432,564 | 149,402 | 756,864 | 146 | `+0.1836` | 21,702 | 86,456 |
| 2025 | 581,966 | 153,904 | 1,005,696 | 194 | `+0.1849` | 29,199 | 116,318 |
| 2026 | 735,870 | 72,806 | 1,259,712 | 243 | `+0.1843` | 36,915 | 147,079 |

账户 leaderboard 关键项：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor_margin::top10` | 3.17x | `+10.20%` | `+7.80%` | 9.94 | 5.01 | `-38.70%` | `+0.003594` | 2.08x |
| `anchor_margin::top15` | 3.06x | `+7.69%` | `+6.42%` | 14.91 | 5.01 | `-36.95%` | `+0.003372` | 2.06x |
| `anchor_loser::top20` | 2.78x | `+3.99%` | `+2.15%` | 19.90 | 5.01 | `-35.55%` | `+0.002640` | 1.87x |
| `anchor_margin::top20` | 2.66x | `+4.55%` | `+3.63%` | 19.89 | 5.01 | `-36.80%` | `+0.002647` | 1.81x |
| `duel_pool::top20` | 3.34x | `-2.66%` | `-2.66%` | 19.91 | 5.01 | `-33.98%` | `+0.003849` | 2.15x |
| `contrast_blend::top20` | 2.83x | `-4.08%` | `-4.08%` | 19.90 | 5.01 | `-32.88%` | `+0.002986` | 1.78x |
| `return_reg::top20` | 1.33x | `+9.33%` | `-26.59%` | 19.93 | 5.01 | `-43.63%` | `-0.000469` | 0.88x |
| `right_tail::top20` | 0.02x | `-16.10%` | `-75.88%` | 19.48 | 5.03 | `-98.11%` | `-0.018844` | 0.01x |

`anchor_loser::top20` 年度账户收益：

```text
2022  +2.15%
2023 +20.39%
2024  +9.92%
2025 +98.02%
2026  +3.99%
```

`anchor_margin::top20` 年度账户收益：

```text
2022  +3.63%
2023 +18.33%
2024  +6.10%
2025 +95.35%
2026  +4.55%
```

标签层也比 Exp156 厚：`anchor_margin::top20` 全期 mean exec `+0.002647`，2026 mean exec `+0.003075`；entry ok `99.59%`。但去最强 5 次后仍只有约 1.47x，说明收益仍依赖少数强调仓段，厚度不足。

特征重要性显示，对比模型不再只依赖市场状态，前列是同日特征差分：`d_range20_low_rank`、`d_amount_rank`、`d_amt_ratio20_rank`、`d_ret60_rank`、`d_spike_age_rank`、`d_ret20_rank`、`d_ret10_rank`、`d_upper_wick_low_rank`、`d_post_spike_return_rank` 等。这说明 pairwise objective 确实学到了“同一天股票之间的相对差异”，不是简单复读市场 regime。

### 157.4 反事实分析

第一反事实：如果对比学习只是换皮，`anchor_*` 不应明显超过普通 reg/cls。实际 `anchor_loser::top20` 2.78x 且逐年正，而 `return_reg::top20` 只有 1.33x、2022/2024 负，`right_tail::top20` 几乎归零。对比目标确实改善了低信噪比下的排序稳定性。

第二反事实：如果 session 内 duel 更贴近真实 TopK，`duel_pool` 应最稳。实际 `duel_pool::top20` final 3.34x，但 2026 `-2.66%`，说明当日互相 duel 更容易放大 2025 风格，前向不稳。

第三反事实：如果 anchor 方法是新主引擎，Top20 final 应接近或超过 Exp40。实际 Top20 只有 2.66-2.78x，低于 Exp40 的 8.77x，说明它只是稳定弱信号，不是收益弹性层。

第四反事实：如果问题是 Top20 尾部薄，Top10/15 应明显接近目标。实际 `anchor_margin::top10` 也只有 3.17x，仍远低于几十倍目标，不能靠缩 TopK 过关。

第五反事实：如果对比学习能解决 2026，2026 应至少接近 Exp40 的 `+19.18%`。实际 anchor Top20 只有 `+4%左右`，说明它通过前向但没有足够弹性。

### 157.5 判定

`rejected_before_formal_candidate`。

Exp157 不合代码，不继续在 Exp129 基础特征上调 pair 阈值、anchor 权重、duel pool、blend 权重或 TopK。

但它给出一个正面方向：同日正负样本对比目标能显著提高稳定性，并让模型关注股票间相对差异。下一轮应把这个目标函数迁移到当前最强的 Exp40 路径世界，而不是继续使用较弱的 Exp129 供需面板。更合理的 Exp158 是：在 Exp40 path-sequence 的 Top200/Top500 候选池和路径特征上做 session contrastive ranker，测试对比目标能否在强土壤上提供收益弹性，同时保住 2026。

## 158. qmt_path_world_contrastive_ranker_v1

### 158.1 问题

Exp157 证明同日正负样本对比目标比普通 reg/cls 更稳，但收益厚度只有 2.6-2.8x，远低于 Exp40。当前项目内没有找到可直接复用的 Exp40 prediction store 或 path-sequence 原始脚本，因此本轮先做一个最小可证伪版本：只用 QMT 日线复建 path-world 候选池和路径压缩特征，再迁移 Exp157 的 contrastive ranker。

核心反事实：如果 Exp157 的瓶颈主要是基础面板弱，那么迁移到路径世界后 Top20 应自然变厚，并至少接近 Exp40；如果自建 path-world 土壤不够，contrastive 只能平滑弱信号，不会产生几十倍收益。

### 158.2 方法

产物：

```text
.tmp/quantx-research/qmt-path-world-contrastive-ranker-v1/analyze_qmt_path_world_contrastive_ranker.py
sha256:059816c868ef33f52ecac7b8e8e564bca9b5c33c1b503e75bb0a5fb4459456f0

.tmp/quantx-research/qmt-path-world-contrastive-ranker-v1/qmt_path_world_contrastive_2021_2026_min30k.json
sha256:e94cad463cfb30583b20dad18cabbd0a6b97382a94f8653f0617abc8705dbc32

.tmp/quantx-research/qmt-path-world-contrastive-ranker-v1/exp158_summary.md
```

`min30k` 是完整五年口径；另一个 `min100k` 结果只覆盖 2024-2026，不能作为五年 formal 结论。

数据只使用 QMT/qlib 可取的日线 OHLCV/VWAP、成交额和 ST，不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念快照或其他消息数据。

特征：用过去 60 日路径压缩为 `last/mean5/mean20/mean60/std20/drift5_20`。底层变量包括 `ret1/open_gap/intraday/range_low/close_strength/close_vwap/ret5/ret20/ret60/amount/amt_ratio20/near_high20` 的横截面 rank，并加入市场宽度、离散度、近高占比和几个 T 日可见 path prior。

候选池：每天先用 T 日可见 `path_base_score` 取 Top800，不使用任何未来标签或测试年结果。

训练：年度 walk-forward，每个测试年只用 `< year` 样本；2026 只用 2021-2025。每个训练 session 内取 `exec_label5_open` 右尾约 top 8% 和左尾 bottom 25%，构造 `x_pos - x_neg` 和反向样本。测试时只用训练集 anchor 或测试日 T 特征互斗，不使用测试标签。

### 158.3 结果

完整口径面板 212,800 行，scored 173,600 行，覆盖 2022-2026 共 217 个信号日。

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor_mid::top30` | 2.12x | `+24.50%` | `-21.43%` | 29.81 | 5.01 | `-32.52%` | `+0.000887` | 1.55x |
| `anchor_mid::top20` | 1.99x | `+25.06%` | `-24.67%` | 19.85 | 5.01 | `-37.55%` | `+0.000487` | 1.38x |
| `duel_blend::top30` | 1.93x | `+19.51%` | `-14.52%` | 29.83 | 5.01 | `-32.19%` | `+0.000445` | 1.43x |
| `duel_pool::top30` | 1.93x | `+10.83%` | `-5.02%` | 29.83 | 5.01 | `-28.06%` | `+0.000560` | 1.40x |
| `anchor_margin::top20` | 1.90x | `-4.73%` | `-4.73%` | 19.94 | 5.01 | `-26.83%` | `+0.000665` | 1.40x |
| `path_blend::top20` | 1.77x | `+20.40%` | `-18.76%` | 19.91 | 5.01 | `-38.07%` | `+0.000201` | 1.25x |

`anchor_mid::top30` 年度收益：

```text
2022  +9.00%
2023 +15.86%
2024 -21.43%
2025 +71.48%
2026 +24.50%
```

`anchor_mid::top20` 年度收益：

```text
2022 +12.92%
2023  +9.18%
2024 -24.67%
2025 +71.28%
2026 +25.06%
```

标签层显示失败不是执行漏损：主要组合 entry ok 基本在 99% 以上，平均持有约 5 天，但 mean exec 只有 `+0.0002` 到 `+0.0009`。2024 多数组合标签层和账户层同步为负。

特征重要性中，contrastive 模型前列仍是 `d_amount_rank_last`、`d_range_low_rank_last`、`d_ret20_rank_last`、`d_ret60_rank_last`、`d_close_vwap_rank_drift5_20` 等股票路径差分；reg/top-quintile 则更依赖 `mkt_disp20`、`mkt_amount_disp20`、`mkt_ret20_median`、`mkt_near_high20`。说明模型确实学习了路径相对差异，但这些差异不足以构成厚收益土壤。

### 158.4 反事实分析

第一反事实：如果对比目标只缺一个更强 path 土壤，那么自建 path-world + contrastive 应显著超过 Exp157。实际最好只有 2.12x，低于 Exp157 的 `duel_pool::top20` 3.34x，也远低于 Exp40 8.77x，说明这个自建候选池不是强土壤。

第二反事实：如果失败来自执行漏损，entry ok 应明显差。实际主要组合 entry ok 基本在 99% 以上，说明失败在标签层，买不到不是主因。

第三反事实：如果 2026 正收益代表方向成功，2022-2025 应至少不被打穿。实际 2024 多数组合为 `-15%` 到 `-25%`，这是风格局部适配，不是鲁棒性。

第四反事实：如果 path_base 手工候选池能复建 Exp40，普通 reg/top_quintile/path_blend 应接近 Exp40。实际 `return_reg::top20` 只有 1.06x，`path_blend::top20` 1.77x，说明 Exp40 的收益来自更具体的 path-sequence store/候选机制，而不是泛化的 60 日压缩日线特征。

第五反事实：如果继续调 TopK 或 blend 权重有意义，Top10/Top15 应明显更厚。实际 Top10 的 2022 或 2024 更容易被打穿，不能用头部化替代用户要求的平均持仓大于 5 和逐年正收益。

### 158.5 判定

`rejected_before_formal_candidate`。

Exp158 不合代码，不继续调 `pool_size`、pair quantile、anchor 权重、duel pool、path_base 权重或 TopK。

这个实验保留一个负面结论：contrastive loss 是稳定器，不是收益发动机；如果候选土壤不厚，它只能把曲线略平滑，不能把 2x 变成几十倍。后续应回到“新的可交易右尾土壤”或找回/复建真正的 Exp40 path-sequence 候选机制，而不是在自建 path-world 上继续调参。

## 159. qmt_cluster_contrastive_ranker_v1

### 159.1 问题

Exp148/150 说明 QMT 动态主题簇是近期少数接近强基线的 QMT-only 线索：簇内排序 Top10 很强，但 Top20 变薄。Exp157/158 又说明 contrastive loss 是有效稳定器，但只有在足够好的候选土壤上才有意义。本轮把两条线索合并：在动态主题簇 `session#cluster` 内做正负样本对比，直接学习“同一主题里哪只股票更值得持有一周”。

核心反事实：如果 Top20 变薄主要来自簇内把过热 leader 和可延续 follower 混在一起，那么簇内 contrastive 应修复 Rank11-20；如果修复后仍低于 Exp40，说明簇内排序是稳定器而不是收益弹性层。

### 159.2 方法

产物：

```text
.tmp/quantx-research/qmt-cluster-contrastive-ranker-v1/analyze_qmt_cluster_contrastive_ranker.py
sha256:96ac18a80129daaaefdec117c2dbd056494a0349206fd0e1f3ff3225c805d031

.tmp/quantx-research/qmt-cluster-contrastive-ranker-v1/qmt_cluster_contrastive_2021_2026.json
sha256:ce65da78f6e050a7882db3a838308112c76ccfdacec6e67ecf56d86cd4cd4347

.tmp/quantx-research/qmt-cluster-contrastive-ranker-v1/exp159_summary.md
```

复用 Exp148 的 QMT-only 动态主题簇：每个 T 日用过去 60 日收益相关，把股票映射到强势 anchor 周围的临时簇，形成 `session#cluster` query。复用 Exp150 的滞后修复特征，但不再做手写修复加权。

训练目标：

1. 每个训练 `session#cluster` 内按 `exec_label5_open` 排序。
2. 正样本为簇内约 top 22%，负样本为 bottom 32%。
3. 构造 `x_pos - x_neg` 标签 1，反向标签 0。
4. 测试时只用训练集 winner/loser/mid anchor、T 日可见特征和簇内 duel，不使用测试标签。

年度 walk-forward：每个测试年只用 `< year` 样本；2026 只用 2021-2025。所有动态簇、特征、anchor 和 duel 都只使用 T 日及以前的 QMT/qlib 日线 OHLCV/VWAP、成交额和 ST。不使用新闻、公告、龙虎榜、ETF、北向、融资融券、静态行业/概念表或其他消息数据。

### 159.3 结果

面板 808,676 行，scored 668,932 行，覆盖 2022-2026 共 217 个信号日。

| 年份 | 训练行数 | 测试行数 | train queries | pair rows | mean pair gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 139,744 | 143,064 | 2,352 | 503,152 | `+0.13999` |
| 2023 | 282,808 | 149,756 | 4,656 | 917,176 | `+0.13600` |
| 2024 | 432,564 | 149,402 | 7,008 | 1,368,464 | `+0.12415` |
| 2025 | 581,966 | 153,904 | 9,312 | 1,691,704 | `+0.12437` |
| 2026 | 735,870 | 72,806 | 11,664 | 2,149,046 | `+0.12421` |

关键账户结果：

| 方案 | final | 2026 | 最差年 | 平均持仓 | 平均持有 | 最大回撤 | mean exec | 去最强3次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_base::top10` | 5.57x | `+13.89%` | `+13.89%` | 9.98 | 5.00 | `-33.59%` | `+0.006474` | 3.58x |
| `duel_blend::top10` | 5.21x | `+14.59%` | `+14.59%` | 9.99 | 5.00 | `-27.66%` | `+0.006193` | 3.35x |
| `cluster_base::top15` | 4.91x | `+4.58%` | `+4.58%` | 14.97 | 5.00 | `-32.02%` | `+0.005977` | 3.22x |
| `cluster_base::top20` | 4.21x | `+6.10%` | `+6.10%` | 19.97 | 5.00 | `-29.06%` | `+0.005239` | 2.70x |
| `duel_blend::top20` | 4.06x | `+7.21%` | `+7.21%` | 19.97 | 5.00 | `-25.11%` | `+0.004952` | 2.72x |
| `contrast_stack::top20` | 3.46x | `+5.34%` | `+5.34%` | 19.97 | 5.00 | `-33.32%` | `+0.004508` | 2.18x |

`cluster_base::top20` 年度收益：

```text
2022 +19.06%
2023 +25.53%
2024 +23.70%
2025 +114.75%
2026  +6.10%
```

`duel_blend::top20` 年度收益：

```text
2022 +14.16%
2023 +31.69%
2024 +27.43%
2025 +97.89%
2026  +7.21%
```

### 159.4 厚度诊断

以 `contrast_stack` 为切片诊断口径，Top20 的 11-20 名不再明显拖后腿：

| 切片 | mean exec | 2022 | 2023 | 2024 | 2025 | 2026 | 平均簇数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top10 | `+0.004987` | `+0.005183` | `+0.004871` | `+0.003196` | `+0.007334` | `+0.003560` | 7.56 |
| Rank11-20 | `+0.004029` | `+0.004103` | `+0.000391` | `+0.003734` | `+0.006533` | `+0.006907` | 7.59 |
| Top20 | `+0.004508` | `+0.004643` | `+0.002631` | `+0.003465` | `+0.006934` | `+0.005233` | 12.63 |

这比 Exp148/149 的“Top10 强、11-20 变薄”有所改善。但账户 final 仍只有 4.21x，说明厚度虽然变平，但收益弹性没有打开。

特征重要性显示，reg/residual/分类头仍大量使用市场离散度、成交额离散度、市场 20 日中位收益、近高比例；contrastive 前列是 `d_price_rank`、`d_mom_bucket_breadth`、`d_vol_bucket_amt`、`d_vol_bucket_strength`、`d_vol20_low_rank`、`d_range20_low_rank` 等横截面/簇内差分。模型确实在学相对结构，但仍没有摆脱市场状态主导收益弹性的瓶颈。

### 159.5 反事实分析

第一反事实：如果簇内 contrastive 是缺失主引擎，Top20 应超过 Exp40 或至少接近 8.77x。实际最佳 Top20 只有 4.21x，说明它是稳定器，不是几十倍收益发动机。

第二反事实：如果 Exp148 的 Top20 变薄只是因为 11-20 过热 leader 混入，那么簇内 contrastive 应显著修复切片。实际 Rank11-20 mean exec 达 `+0.004029`，2026 甚至高于 Top10，说明这个反事实部分成立。

第三反事实：如果这个修复足以进入候选，2026 应接近或超过 Exp40 的 `+19.18%`。实际 Top20 只有 `+6%-7%`，2026 是前向通过但弹性不足。

第四反事实：如果靠 Top10 就能代表方向，Top10 应接近几十倍。实际 Top10 最高 5.57x，仍低于 Exp40，也不能用缩 TopK 替代平均持仓要求。

第五反事实：如果收益来自可稳定迁移的簇内关系，去掉最强三次后仍应有较高倍数。实际 `cluster_base::top20` remove best 3 只有 2.70x，收益仍受少数强阶段影响。

### 159.6 判定

`rejected_before_formal_candidate`。

Exp159 不合代码，不继续调 pair quantile、anchor 权重、duel 对手数、blend 权重或 TopK。

保留的有效观察是：动态主题簇 + 簇内对比学习，是当前少数能同时做到 Top20 逐年正、平均持仓接近 20、持有约一周的 QMT-only 方向。但它没有收益弹性。下一步应利用这个稳定簇内排序作为“底座/标签生成器”，转向寻找低相关的收益弹性层，例如簇级主升早期识别、跨簇资金迁移起点、或更贴近 formal 可交易右尾的候选生成，而不是继续在簇内排序权重上微调。
