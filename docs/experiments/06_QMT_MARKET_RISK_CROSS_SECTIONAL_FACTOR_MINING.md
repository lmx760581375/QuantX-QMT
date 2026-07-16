# QMT 市场风险与横截面因子新一轮挖掘

> 建档日期: 2026-07-15  
> 状态: active research loop  
> 范围: QMT/qlib 可取得的日线和必要时日内 K 线；不使用新闻、公告、龙虎榜等消息类数据。

## 1. 为什么重开 06

Exp40 `path_sequence_ranker_v1` 的无未来函数审计没有发现问题，但后续 Top7 bridge 合入主线时暴露了更关键的缺口：旧递归实验没有把 ST/退市、停牌占位、一字板买不到、开盘追高和反复回补等可交易性污染作为前置 hard gate。旧 `+777.49%` / `2026 +19.18%` 只能说明多日路径特征有信息，不能再作为 retained candidate 或硬基线。

因此 06 文档从新的研究协议开始：每轮必须完成“寻找方向 -> 验证方向 -> 结果深度分析 -> 落档实验文档 -> 更改研究方向”的闭环，并且每次都使用 QuantX 跑完整回测，再独立分析交易是否异常。

## 2. 硬目标

候选方向进入策略库前，必须同时满足：

1. 五年近几十倍到 100 倍级别收益，不接受靠降低尾部权重硬卡结果。
2. 五年每年收益为正，且 2026 前向通过。
3. 平均持仓股票数大于 5。
4. 平均持仓周期约一周。
5. 完整 QuantX 回测通过交易异常审计。
6. 所有中间 config、脚本、模型、run 和缓存都留在 `.tmp/quantx-research/`；只有满足门槛的策略才允许进入正式 artifacts/configs。

## 3. 每轮实验硬流程

每轮实验必须记录以下内容：

1. **Setting 无未来函数审计**：特征只使用 T 日及以前信息；训练年度 walk-forward 只用 `< test_year` 样本；2026 只用 2021-2025 或更早历史。
2. **可交易性污染审计**：有效买入中的当前 ST/退市、历史 ST-like proxy、缺失行情、零成交量/零成交额、一字涨停、开盘高开超过阈值、拒单原因分布和单票重复买入集中度。
3. **完整 QuantX 回测**：导出 PredictionStore，使用标准 `run_backtest` 跑完整账户，不只看 label 或简化 sleeve。
4. **反事实分析**：每次至少列出 3 个可能原因，并用结果判断哪些被支持、哪些被否定。
5. **下一步方向更新**：实验失败时必须说明不继续调哪些参数，以及下一轮为何换方向。

若某轮没有完成完整 QuantX 回测和交易异常审计，只能标记为 `diagnostic_only`，不能标记为候选。

## 4. 方向约束

当前优先方向：

1. 市场风险状态：宽度、离散度、市场收益、候选池拥挤和弱宽度环境。
2. 横截面因子路径：不仅看单日 rank，还看过去 5/10/20/60 日的 rank 轨迹、斜率、稳定性和反转。
3. 行业/概念轮动：A 股短周期强依赖行业/概念扩散，但当前行业/概念快照不是 point-in-time 历史成员，必须作为诊断/弱特征披露。
4. 对比学习和排序学习：低信噪比环境优先用同日正负样本、pairwise/listwise 目标，而不是单点回归。
5. QMT-only：若不是 QMT/本地 qlib 可取得的数据，暂不考虑；消息类方向暂停。

## 5. Run Research 审计更新

2026-07-15 已更新 `run_research`：`DatasetBuilder` 在构建标签时同步输出执行诊断列，并在 dry-run 和正式结果 JSON 中增加 `execution_audit`。

当前审计列包括：

```text
entry_has_bar
entry_zero_volume_or_amount
entry_one_price_limit_up
entry_open_gap_pct
entry_open_gap_ge_3pct
st_like_limit_history
```

其中 `st_like_limit_history` 是基于历史 5% 涨跌停特征的无未来函数代理，不是 point-in-time 历史 ST 名称。当前 QMT 路径只能可靠得到当前 `InstrumentName`，不能直接得到历史 ST 名称。

验证命令：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m pytest -p no:capture \
  tests/tools/test_run_research.py tests/research -q
```

结果：`39 passed, 2 skipped, 4 warnings`。

## 6. Exp001 qmt_market_contrastive_ranker_v1


### 6.1 假设

旧 Exp40 被可交易性污染后，不能继续把“极端右尾强股”当作收益来源。新的第一轮假设是：真正可交易的一周收益更可能来自“市场风险状态允许 + 行业/概念扩散未退潮 + 个股处在横截面路径中的相对有利位置”。

如果这个假设成立，那么年度 walk-forward 的同日 pairwise/listwise 排序应在 Top20 周频完整 QuantX 回测中体现为：五年逐年正、平均持仓大于 5、平均持仓约一周，并且 2026 前向不失效。

### 6.2 Setting

输出目录：

```text
.tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/
```

数据边界：

1. provider: `data/qlib_data_fixed`
2. 股票池: 主板 A 股，排除创业板、科创板、北交所和 ETF。
3. 特征: QMT/qlib 日线 OHLCV、成交额、VWAP、市场宽度/离散度、行业/概念快照派生的组内相对强弱。
4. 标签: T+1 open 到 T+6 open 的 5 日收益；训练只用于历史年度，测试年度不参与训练。
5. 模型: LightGBM LambdaRank 作为第一版 pairwise/listwise 代理；后续若方向有信号，再上显式正负样本神经网络。
6. 回测: 导出 PredictionStore 后用 QuantX 标准 `decision_pipeline` Top20、5 日 rebalance、T+1 open 完整回测。

### 6.3 运行结果

运行脚本：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python \
  .tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/run_qmt_market_contrastive_ranker.py
```

产物：

```text
.tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/predictions.json
.tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/backtest_config.yaml
.tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/result.json
.tmp/quantx-research/06-qmt-market-contrastive-ranker-v1/runs/qmt_market_contrastive_ranker_v1_top20_reb5/
```

无未来函数 setting 审计：

1. 特征在完整日线面板上构造，只使用 T 日及以前的 OHLCV、成交额、VWAP、rolling rank、市场宽度/离散度和当前快照行业/概念弱特征。
2. 训练/预测信号日为每 5 个交易日一次；先在完整面板计算 rolling 和标签，再裁剪信号日，避免裁剪导致历史窗口断裂。
3. 标签为 T+1 open 到 T+6 open 的 5 日收益；标签只参与历史训练，不参与测试年预测特征。
4. 年度 walk-forward：2022、2023、2024、2025、2026 各测试年只使用 `< test_year` 样本训练。
5. 训练标签对 T+1 不可交易样本做惩罚：一字板、零成交、开盘高开超过 3%、ST-like 代理和当前 ST/退市名会被压成负标签；这只影响历史训练标签，不向预测时注入 T+1 信息。

完整 QuantX 回测结果：

| 指标 | 数值 |
| --- | ---: |
| 区间 | 2022-01-04 至 2026-07-15 |
| 初始资金 | 1,000,000 |
| final value | 1,311,778.72 |
| total return | +31.18% |
| annual return | +6.18% |
| max drawdown | -28.69% |
| Sharpe | 0.302 |
| buys / sells | 3,439 / 3,435 |
| 平均持仓数 | 19.44 |
| 持仓周期均值 / 中位数 | 11.77 / 7.00 天 |

年度收益：

| 年份 | 收益 |
| --- | ---: |
| 2022 | -17.03% |
| 2023 | +10.47% |
| 2024 | -2.78% |
| 2025 | +38.56% |
| 2026 YTD | +3.34% |

交易异常审计：

| 项目 | 数值 |
| --- | ---: |
| effective buys | 3,439 |
| 当前 ST/退市名买入 | 0 |
| 缺失行情买入 | 0 |
| 零成交量/成交额买入 | 0 |
| 一字涨停买入 | 0 |
| 开盘高开 >= 3% 买入 | 0 |
| unique buy symbols | 832 |
| reject rows | 6 |

Top20 label 诊断已经提前提示弱信号：2022、2023、2024、2025、2026 的 Top20 mean exec label 分别为 `-2.24%`、`-0.45%`、`-0.82%`、`-0.18%`、`-1.57%`。完整回测没有暴露交易异常，但收益和逐年稳定性远低于硬目标。

### 6.4 判定

状态：`failed_valid_replay`

这轮实验不是因为未来函数或交易污染失败，而是方向本身 alpha 弱。它满足平均持仓大于 5、持仓周期接近一周、交易异常审计为 0，但不满足“五年近几十倍到 100 倍收益”和“五年逐年正收益”。因此不能作为候选，也不继续通过轻微调参、降低持仓数或尾部权重来硬卡结果。

### 6.5 反事实分析

可能原因 1：全市场泛化 ranker 把“风险可控的大盘/低波/高流动性股票”学成了稳态偏好，而不是短周期爆发 alpha。证据是重复买入集中在银行、公用事业和大盘蓝筹较多，回撤可控但收益弹性不足。

可能原因 2：行业/概念只用当前快照做弱特征，没有真正刻画“板块扩散路径”。证据是行业/概念特征没有带来明显年度穿越能力，2022 和 2024 仍然为负。

可能原因 3：训练标签惩罚不可交易右尾后，模型避开了污染票，但也没有识别出可交易的高弹性票。证据是交易异常为 0，说明过滤有效；但 Top20 mean exec label 在各年均为负，说明排序目标没有抓到可交易 alpha。

可能原因 4：单阶段横截面模型把市场状态、板块状态和个股状态混在一起学，信噪比被稀释。A 股强轮动环境中，可能需要先判断“哪些日期/哪些板块值得冒险”，再在板块内做个股对比。

下一轮不再沿着 `qmt_market_contrastive_ranker_v1` 做小参数优化。方向切换到“市场状态门控 + 行业扩散路径 + 板块内个股二阶段排序”：先只在市场宽度和行业扩散允许的日期/板块内产生候选，再做同板块内正负样本对比，避免全市场 ranker 被低波稳态票主导。

## 7. Exp002 market_industry_diffusion_two_stage_v1

### 7.1 假设

A 股一周级别收益的主要来源不是稳定全市场排序，而是“市场风险状态允许 + 行业/概念扩散正在发生 + 板块内个股尚未完全透支”的共振。Exp002 将把问题拆成两层：

1. 日期/行业层：判断哪些市场状态和行业扩散路径值得承担风险。
2. 个股层：仅在候选行业内做同日、同板块正负样本对比，训练目标仍是可交易的 T+1 open 到 T+6 open 相对收益。

### 7.2 Setting 计划

1. 数据仍然只使用 `data/qlib_data_fixed` 日线和本地 MetaStore 行业成员；概念快照只作为诊断，不作为强 PIT 特征。
2. 先计算行业层 5/10/20 日收益、行业内上涨家数占比、行业内成交额扩散、行业强度斜率、行业拥挤度和市场宽度。
3. 只保留当日行业扩散排名靠前且市场宽度不过弱的行业作为候选池；这是风险暴露选择，不是降低最终持仓硬卡收益。
4. 个股层特征强调板块内相对强弱、成交额放大、回撤后修复、波动收缩后突破和开盘不可交易风险代理。
5. 模型先用 LightGBM LambdaRank 做二阶段基线；若 label 诊断有效，再改显式 pair sampling 的对比学习。
6. 回测仍使用 PredictionStore + QuantX 完整回测，Top20、5 日 rebalance、T+1 open、相同买入过滤和止损。

### 7.3 运行结果

运行脚本：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python \
  .tmp/quantx-research/06-market-industry-diffusion-two-stage-v1/run_market_industry_diffusion_two_stage.py
```

产物：

```text
.tmp/quantx-research/06-market-industry-diffusion-two-stage-v1/predictions.json
.tmp/quantx-research/06-market-industry-diffusion-two-stage-v1/backtest_config.yaml
.tmp/quantx-research/06-market-industry-diffusion-two-stage-v1/result.json
.tmp/quantx-research/06-market-industry-diffusion-two-stage-v1/runs/market_industry_diffusion_two_stage_v1_top20_reb5/
```

无未来函数 setting 审计：

1. 行业扩散特征只使用 T 日及以前的行业收益、宽度、成交额扩散、波动和市场宽度。
2. 先在完整日线面板上计算 rolling 特征和 T+1/T+6 连续标签，再裁剪每 5 个交易日信号日。
3. 行业门控只使用 T 日已知特征：行业扩散 rank >= 0.82，市场 20 日宽度 >= 0.28，行业样本数 >= 12。
4. 行业内 `label_q5` 后移到周频且门控后的样本上计算，避免在全量无关样本上慢速 qcut，同时不改变训练标签定义。
5. 年度 walk-forward：每个测试年只使用 `< test_year` 样本训练。

门控后样本诊断：总样本 `45,753` 行，信号日 `269` 个；每个信号日约 `235-250` 个候选、约 `8.8-9.6` 个行业。Top20 mean exec label 在 2022-2026 分别为 `-1.91%`、`-1.55%`、`-2.00%`、`-0.89%`、`-1.25%`。

完整 QuantX 回测结果：

| 指标 | 数值 |
| --- | ---: |
| 区间 | 2022-01-04 至 2026-07-15 |
| 初始资金 | 1,000,000 |
| final value | 763,151.77 |
| total return | -23.68% |
| annual return | -5.79% |
| max drawdown | -35.03% |
| Sharpe | -0.283 |
| buys / sells | 2,887 / 2,932 |
| 平均持仓数 | 17.74 |
| 持仓周期均值 / 中位数 | 10.31 / 7.00 天 |

年度收益：

| 年份 | 收益 |
| --- | ---: |
| 2022 | -25.52% |
| 2023 | +14.93% |
| 2024 | -14.71% |
| 2025 | +22.02% |
| 2026 YTD | -14.67% |

交易异常审计：

| 项目 | 数值 |
| --- | ---: |
| effective buys | 2,887 |
| 当前 ST/退市名买入 | 0 |
| 缺失行情买入 | 0 |
| 零成交量/成交额买入 | 0 |
| 一字涨停买入 | 0 |
| 开盘高开 >= 3% 买入 | 0 |
| unique buy symbols | 1,057 |
| reject rows | 6 |

### 7.4 判定

状态：`failed_valid_replay`

Exp002 的可交易性是干净的，但收益显著失败。它满足平均持仓大于 5、持仓周期接近一周、交易异常为 0，但总收益为负，且 2022、2024、2026 均为负，不满足任何收益硬目标。

### 7.5 反事实分析

可能原因 1：行业扩散 rank 选择到的是“已经热过的行业”，T+1 open 到 T+6 open 更像追高回撤，而不是扩散初期。Top20 label 全年为负支持这个解释。

可能原因 2：当前行业分类太粗，无法表达 A 股短周期真正交易的概念/题材链条。使用粗行业门控会把概念强票与同业弱票混在一起，板块内比较反而稀释信号。

可能原因 3：市场宽度 floor 太弱，无法真正隔离风险期；但这不是继续调阈值的理由，因为 label 诊断在所有年份都为负，说明排序目标自身不成立。

可能原因 4：A 股一周级 alpha 可能不在“行业扩散强者恒强”，而在“已证明弹性但尚可买入”的微观路径，例如涨停后非一字的断板、回撤修复、量能再扩张、弱转强失败后的再选择。

下一轮转向更激进的可交易高弹性路径，不再围绕行业扩散二阶段做阈值微调。

## 8. Exp003 tradable_elastic_path_ranker_v1

### 8.1 假设

Exp40 被 ST、一字板和停牌污染后，不能直接追极端右尾；但“股票先证明自己有弹性，再进入可交易窗口”这个结构仍可能是真 alpha。Exp003 不买停牌/零成交/一字板/当前 ST 或退市名；普通高开不作为硬过滤，只作为交易诊断项，因为高开可能正是弱转强和强势延续策略的收益来源。

核心假设：未来一周收益更可能来自以下状态的交集：

1. 过去 20 日出现过大阳线或接近涨停，但当前 T 日不是一字板，T+1 open 也必须可交易。
2. 个股经历 2-5 日回撤/横盘后重新放量，属于“弹性确认后的二次选择”，而不是直接追一字。
3. 市场宽度不能太弱，行业/横截面相对强度不能完全退潮。
4. 用同日正负样本排序，比较“有弹性路径中的好票/差票”，而不是全市场泛化 rank。

### 8.2 Setting 计划

1. 数据仍只使用 QMT/qlib 日线 OHLCV、成交额、VWAP 和本地 MetaStore 当前 ST 名称。
2. 候选池先用 T 日以前特征限定为可交易高弹性路径：近 20 日至少一次 `ret1 >= 7%` 或振幅/收盘位置显著强，且近 3-10 日有回撤修复/量能再扩张迹象。
3. 显式排除当前 T 日一字板、零成交、当前 ST/退市名和 ST-like 代理；普通开盘高开只记录诊断，不作为硬过滤。训练标签继续惩罚 T+1 真正不可交易样本。
4. 模型先用 LightGBM LambdaRank，分组为同日候选池；若 label 诊断转正，再升级为显式 pair sampling 的对比学习。
5. 回测仍用 PredictionStore + QuantX 完整账户，Top20、5 日 rebalance、T+1 open、相同买入过滤和止损。

下一步执行 Exp003。若该方向仍然不能让 label 诊断转正，说明仅靠日线形态/弹性路径可能不足，需要考虑日内 K 线确认或更强的市场状态空仓机制。

## 9. Brick 公开候选池复现审计：Exp40 失败经验补充

在继续新因子挖掘前，补做了投资科学 `strategy-renko` 公开候选池的公式审计。用户给出的公式是：

```text
砖型图的绿砖后面出现反转红砖
```

本轮结论对后续 run research 很重要：不能再把 QuantX 当前 `BrickChart(...)` 输出直接当成官方砖高度。QuantX 算子当前返回的是 `raw_brick` 的一阶差分，即 `raw[t] - raw[t-1]`；而公开 screener 在 2026-07-03 以后暴露的 `factor_values.brick / prev_brick` 是砖高度本体，`brick_delta_1 / reversal_h` 才是高度变化。

临时审计产物：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/analyze_raw_brick_alignment.py
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_alignment_report.json
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_candidate_rule_eval.json
```

公开完整字段的一致性：

| 检查项 | 结果 |
| --- | ---: |
| `official_brick_delta_1 == official_brick - official_prev_brick` | 100.00% |
| `official_brick_delta_1 == official_reversal_h` | 100.00% |
| `official_brick_delta_1 > 0` | 100.00% |
| `official_brick_decline_sum_5d > 0` | 100.00% |
| `official_brick_max_consec_down_5d >= 1` | 100.00% |

本地 raw brick 与官方字段相关性：

| 字段关系 | 相关性 |
| --- | ---: |
| `official_brick` vs local `raw_brick` | 0.8490 |
| `official_prev_brick` vs local `raw_prev` | 0.8780 |
| `official_brick_delta_1` vs local `raw_delta` | 0.5098 |
| `official_decline_sum_5d` vs local `raw_decline_sum_incl5` | 0.8651 |

候选池层面：仅使用“昨日下行、今日上行”的本地差分反转，可以覆盖官方候选 `80.94%`，但平均每天会产生约 `888` 只候选，precision 只有 `10.04%`。加入可解释强度条件后：

```text
raw_delta_prev < 0 and raw_delta > 0
and raw_brick > 61.14
and ret1 > 0.04
```

全区间 precision `39.86%`，recall `58.81%`，平均每日候选约 `163`。更复杂的浅层树规则可以达到 precision `36.29%`、recall `71.29%`、平均每日候选约 `216`。

### 9.1 对研究循环的修正

这说明 Exp40/Brick 过去的失败经验不能简单归因于“Brick 方向无效”，也不能简单归因于 ST 或一字板污染。更准确的归因是：候选池一阶公式存在口径错配，`BrickChart` 差分被误当成了官方 brick 高度；在这个错误候选土壤上训练 ML，后续收益判断会被系统性扭曲。

后续所有砖形图、强势反转、弱转强类实验必须遵守以下约束：

1. 明确区分 `brick_height` 和 `brick_delta`，不得用同一个字段名 `brick` 混淆高度与变化量。
2. 候选池必须先通过可交易性审计：停牌、零成交、一字涨停、一字跌停、当前 ST/退市名、缺失行情都要显式统计。
3. 如果使用公开策略做复现，只能用公开候选池校准公式，不能用公开 Top5 交易明细做训练标签，除非明确标记为蒸馏复现。
4. 在候选池没有达到足够保真度前，不做模型层结论；否则模型只是在错误候选池里学习噪声。
5. 后续若继续 Brick，应优先做参数扫描和公式反推，而不是继续调 `score_floor` 或降低尾部持仓权重。

### 9.2 当前判定

状态：`audit_in_progress`

Brick 公开候选池已经部分对齐，但还不足以作为正式策略 milestone。它为后续研究提供了一个更清晰的方向：先复原真实的横截面候选土壤，再用 forward-label / pairwise ranking 学习排序，而不是用错误的一阶候选池强行训练。

### 9.3 后续复核：三线翻红不是主公式，n=5 强度公式更接近

用户提供了另一个通达信三线翻红/翻绿公式：

```text
N:=3;
A3:=REF(C,N);
突破:=C>HHV(A3,N);
破位:=C<LLV(A3,N);
三线翻红:BARSLAST(突破)<BARSLAST(破位) AND C>O;
三线翻绿:BARSLAST(突破)>BARSLAST(破位) AND C<O;
```

本轮全 A 股覆盖验证后，判定它不像公开 `strategy-renko` 的主候选公式：

| 规则 | Precision | Recall | F1 | 平均每日候选 |
| --- | ---: | ---: | ---: | ---: |
| `N=3 三线翻红` | 6.65% | 62.87% | 12.03% | 1,042 |
| `N=3 前一日翻绿信号后翻红` | 15.10% | 17.70% | 16.29% | 133 |
| `N=2 前一日翻绿信号后翻红` | 15.22% | 33.65% | 20.96% | 251 |

也就是说，它可以解释“红绿状态”的视觉表达，但不能解释公开候选池。后续不要把该三线公式作为 Brick 主复现路径。

与此同时，强度公式参数从 `n=8` 改为 `n=5` 后，官方完整字段对齐显著改善：

| 字段关系 | n=8 | n=5 |
| --- | ---: | ---: |
| `official_reversal_h` vs local `raw_delta` | 0.5098 | 0.7659 |
| `official_decline_sum_5d` vs local `raw_decline_sum_incl5` | 0.8651 | 0.9181 |
| `official_range_from_5d_min` vs local range | 0.6320 | 0.7650 |

但候选池粗覆盖没有同步改善，说明官方还有排序前硬过滤。当前更可信的候选土壤是：

```text
brick_height = raw BrickChart height with n=5
brick_delta = brick_height - REF(brick_height, 1)
candidate_core = REF(brick_delta, 1) < 0 and brick_delta > 0
```

再叠加强度/流动性过滤后，较好规则为：

```text
candidate_core
and brick_height > 60
and ret1 > 0.04
and amount_rank >= 0.5
```

该规则 precision `39.80%`，recall `58.10%`，F1 `47.24%`，平均每日候选约 `161`。它仍不足以合入正式策略，但已经比旧候选池更接近公开候选土壤。

该判断已被后续 K 线页前端公式复核修正：`n=5 brick_height/brick_delta` 只是近似，后续 Brick 方向研究必须切换到公开 K 线页实际使用的 `frontend_brick` 公式。

### 9.4 K 线页前端公式修正：Brick 主线切换到 frontend brick

用户补充了 K 线页链接后，前端 chunk `touzikexue_useKLineBars-CyArTqP-.js` 暴露了砖形图绘图公式：

```text
HHV4 = 最近 4 根 K 线最高价
LLV4 = 最近 4 根 K 线最低价
var2 = SMA((HHV4 - close) / (HHV4 - LLV4) * 100 - 90, 4, 1)
var4 = SMA((close - LLV4) / (HHV4 - LLV4) * 100, 6, 1)
var5 = SMA(var4, 6, 1)
brick_height = max(0, var5 - var2 - 4)
brick_delta = brick_height - REF(brick_height, 1)
```

这和此前猜测的 QuantX `BrickChart(n=8)` 或扫参得到的 `n=5` 都不同。官方 `daily-bars` 接口本身也返回了 `brick` 序列，单票 `002485` 最近 200 根 K 线按前端公式复算，最大绝对误差约 `0.0013`，说明公式本体已经锁定。

全市场 36 个公开候选池日的字段对齐：

| 字段关系 | 相关性 |
| --- | ---: |
| `official_brick` vs `frontend_brick` | 0.9975 |
| `official_prev_brick` vs `frontend_prev` | 0.9998 |
| `official_delta/reversal_h` vs `frontend_delta` | 0.9504 |
| `official_decline_sum_5d` vs `frontend_decline_sum_incl5` | 0.9994 |
| `official_range_from_5d_min` vs `frontend_range` | 0.9718 |

因此后续所有砖形图实验必须使用以下语义：

```text
brick_height = frontend_brick
brick_delta = frontend_brick - REF(frontend_brick, 1)
candidate_core = REF(brick_delta, 1) < 0 and brick_delta > 0
```

候选池覆盖结果显示，`candidate_core` 几乎全召回官方候选，但过宽：precision `12.54%`，recall `99.77%`，平均每日约 `877` 只。硬过滤探针显示官方排序前过滤的主轴是：

1. `frontend_brick / brick_rank_pct`：砖高度和横截面高度分位。
2. `ret1 / ret1_rank_pct`：当日涨幅和涨幅横截面强度。
3. `$amount / amount_rank_pct`：成交额和流动性。
4. `frontend_delta`：反转增量。
5. `amplitude/body`：强实体、振幅状态。

代表性规则：

```text
candidate_core
and brick_rank_pct >= 0.60
and frontend_delta >= 3
and amount_rank_pct >= 0.60
```

全 36 日 precision `52.99%`，recall `64.58%`，F1 `58.21%`，平均每日候选约 `134`。更偏召回的 `amount_rank_pct >= 0.50` 版本 precision `49.60%`，recall `70.23%`，平均每日候选约 `156`。

但逐日看仍不稳定，2026-07-02、2026-07-06、2026-07-15 召回偏低。当前判定是：`frontend_brick_aligned_candidate_filter_partially_aligned_not_mergeable`。Brick 可以继续作为“强势反转候选土壤”研究，但在官方候选硬过滤/branch 未进一步对齐前，不能合入策略，也不能据此评价最终 alpha。

### 9.5 score 层复核：公开排序不能跨版本混训

继续审计公开 screener 内部排序后发现，7/3 之后候选项直接暴露 `factor_values.v11_score/v11_rank`，且页面 `score/rank` 与其完全一致：`score` vs `v11_score` 相关性约 `1.0`，最大绝对差仅 `0.00028`，`rank == v11_rank` 比例 `100%`。

但 7/3 前后 score 口径明显切换：早期 score 主要在 `-18` 到 `8`，6/18-7/03 又像 `0-100` rank-like 分数，7/06 后才是 `v11_score` 的正负模型分。用日期切分做 OOS 解释性验证：

| 区间 | 模型 | Mean Daily Spearman | Mean Daily Top5 Recall |
| --- | --- | ---: | ---: |
| 7/3 后 v11 暴露段 | Ridge | 0.8635 | 75.00% |
| 7/3 后 v11 暴露段 | ExtraTrees | 0.8540 | 70.00% |
| 7/3 前非 v11 段 | Ridge | 0.1001 | 10.00% |
| 7/3 前非 v11 段 | ExtraTrees | 0.1021 | 10.00% |
| 全 36 日混训 | ExtraTrees | 0.4048 | 23.08% |

结论：公开 score 可以用于理解后段 `v11` 排序偏好，但不能跨版本混训成一个统一模型，更不能作为收益结论。后续研究应把公开 score 只作为 sanity check；真正策略训练仍应基于本地 point-in-time 特征和未来可交易收益/风险标签，采用 pairwise/ranking 或对比学习目标。

### 9.6 严格 OOS 复核：frontend brick 不是当前可继续微调的 alpha 土壤

在公式本体对齐后，补跑了非蒸馏严格 OOS baseline：训练期 2016-2024，OOS 2025-2026；不使用公开 Top5、公开 score 或公开 rank 做训练。候选池使用 `frontend_delta` 绿后红反转，并加入砖高度、反转增量、成交额、涨幅强度的排序前硬过滤近似；执行层过滤 ST/退市名、ST-like、停牌/零成交、一字涨停和 T+1 高开 >= 3%。

核心结果：

| 指标 | 结果 |
| --- | ---: |
| 候选标签均值 train 2016-2024 | -1.24% |
| 候选标签均值 OOS 2025-2026 | -1.34% |
| 模型 OOS Top7 标签均值 | -0.59% |
| OOS 回测总收益 | 1.59% |
| OOS 最大回撤 | -22.03% |
| 2025 收益 | 0.01% |
| 2026 收益 | -2.29% |
| 平均持仓数 | 9.91 |
| 买入可交易违规 | 0 |

进一步分层显示，`frontend_brick` 候选更像追高后均值回撤，而不是一周级正 alpha：OOS 中高 `ret1/ret3/ret5`、高实体、高收盘位置、高振幅、高砖高度、高均线乖离的分层后续收益均显著更差。用训练期选择的反追高排序和候选拥挤度/候选日动量/科创创业占比/候选均线乖离等 regime 门控，也只能把亏损缩小，不能转成稳定正期望。

当前判定：`frontend_brick_formula_aligned_but_alpha_rejected_under_strict_tradable_oos`。

对后续研究循环的修正：

1. 不再围绕 `strategy-renko` 候选池做仓位、TopK 或阈值微调；该候选土壤本身在训练期和 OOS 都是负期望。
2. 砖线方向可以保留为风险反例和可视化校准对象，但不能作为 milestone 合入策略。
3. 下一轮转向全市场横截面候选土壤：先证明候选池 label 层有正期望，再上 ML/pairwise/contrastive 排序。
4. 继续坚持可交易审计：ST、停牌、零成交、一字板、缺失行情、高开不可成交，都必须在标签和回测两层统计。

## 10. 下一轮方向：全市场横截面高弹性但不追高土壤

Brick 失败给出的反事实很有价值：A 股短期强势信号如果直接买“已充分表现的强红砖”，会变成追高回撤。更可能的方向不是更复杂地拟合红砖，而是在全市场中寻找“已经证明弹性，但当日还没有过度透支，且处于市场/概念扩散窗口”的股票。

下一轮实验不从公开 screener 候选池出发，而从全市场日线构造候选：

1. 先要求过去 20 日出现过弹性证明，例如大阳线、接近涨停、成交额显著扩张或 20 日相对强度明显。
2. 当前 T 日避免追高：限制当日涨幅、实体、收盘位置、均线乖离和一字板；允许普通强势但不买不可交易一字。
3. 加入市场风险和横截面状态：全市场宽度、涨跌停扩散、候选拥挤度、行业/概念静态分组的同日扩散强度。
4. 标签先用 T+1 open 至 5-7 日动态退出，明确统计 stop loss、max hold、趋势转弱退出；只有 label 层转正后再进入完整回测。
5. 排序模型优先采用同日 pairwise/ranking；对比学习只有在候选土壤正期望且样本足够后再引入。

这轮目标不是硬卡收益，而是找到一个训练期逐年 label 层都不坏、OOS 2025-2026 不塌的候选土壤。
