# 07 活跃市值 + 砖形图复现线索整理

> 建档日期：2026-07-15
>
> 当前状态：`research_notes_before_full_replay`
>
> 目标：把当前关于“活跃市值启动后做砖形图交易”的所有线索、口径、已知反证和下一步实验规格统一沉淀，避免后续复现时混用不同定义。

## 1. 当前核心假设

上一轮单独复现 `strategy-renko / frontend_brick` 时，公式本体已经基本对齐，但在全样本、严格可交易约束下，绿砖后反转红砖候选池整体是负期望。这不必然否定砖形图方向，因为公开视频和用户补充信息反复强调：砖形图交易应该发生在“活跃市值进入多头/启动”之后。

因此当前主假设修正为：

```text
活跃市值启动/进入多头
  -> 市场有增量资金入场
  -> 启动当天或两天内涨幅更高、红砖增量更大的股票更可能是主线
  -> 用砖形图规则持有，绿砖或连续 7 块红砖卖出
```

这和此前失败 baseline 的区别非常关键：此前测的是“任意市场状态下的红砖反转”，现在要测的是“活跃资金入场窗口中的强红砖”。

## 2. 外部线索来源

### 2.1 用户转述的视频 1：活跃市值启动后做强红砖

用户转述的核心信息：

1. 活跃市值启动意味着大量资金入场。
2. 启动当天大涨的板块和股票更可能是这个波段主线。
3. 对很多初学者反直觉的是：活跃市值开始时应优先看涨幅高的股票，而不是低位、没涨的股票。
4. “T字”定义：活跃市值确认后，两天以内涨幅 4% 以上的当天。
5. 实验观察：
   - T字当天涨幅和未来收益正相关。
   - 涨停板未来收益特别高。
   - 砖形图反转的未来收益更高。
   - 红砖高度/增量和未来收益正相关。
6. 不应选 KDJ 很低、J 增量小、涨幅小的股票；这些在活跃市值当天出现时，未来收益不会特别高。
7. 卖出策略：绿砖卖出，或连续七块红砖卖出；中途止盈会减少收益，所以不采用中途止盈。
8. 排序模型有一定效果但不是主要收益来源；主要收益来自活跃市值、牛市环境和卖出策略，最后才是排序模型。
9. 作者提到实际回测年化约 95%。

### 2.2 用户转述的视频 2：活跃市值波段退出用 MA10

用户转述的核心信息：

1. 如果只等活跃市值跌破较慢条件，可能在跌破之前已经亏很多。
2. 作者比较了 2013-2026 的活跃市值数据和多种策略。
3. 入场条件候选：
   - 1 到 3 天内涨幅 4% 以上入场。
   - 1 到 2 天内涨幅 4% 以上入场，条件更严格，波段更少，胜率提升。
   - 2 天内涨幅 3% 以上入场会增加很多波段，但胜率下降，全 A / 上证收益没有提高，交易摩擦和心态压力更差。
4. 退出条件比较：
   - 当天跌幅超过约 -2% 离场。
   - -2% 或 MACD 死叉任一条件离场。
   - 跌破某条均线离场。
   - 去掉 -2% 后，跌破均线离场收益更好，说明 -2% 止损反而是拖累项。
5. 最终策略：1-2 天内涨幅超过 4% 入场，跌破 MA10 离场。
6. 用户补充问答：这里的 MA10 是普通移动均线。

### 2.3 用户提供的公开链接和定义

链接：

```text
https://www.compass.cn/shownews.php?nid=33751625
https://zhuanlan.zhihu.com/p/1926021631257605444
```

指南针活跃市值的核心计算法，按用户提供内容整理为：

```text
活跃市值 = 当日成交量 * 当日均价
```

应用解释：该指标通过剔除长期沉淀、不交易的筹码，仅计算短期内频繁交易的股票市值，用于度量市场资金存量和活跃资金规模。

指南针系统中可输入 `0AMV` 调出活跃市值指数。该指数每日包含开盘价、最高价、最低价、收盘价，可形成 K 线走势图。

重要说明：本地 QMT/qlib 当前最稳定可用的是个股日线 `$amount`。由于：

```text
成交额 ~= 成交量 * 当日均价
```

所以本地第一版可用全市场 `$amount` 汇总近似活跃市值的收盘值。但这只是 0AMV 的 close 近似，不等于真实 0AMV 指数的 OHLC 序列。若后续能通过 QMT 获取 `0AMV` 指数本体，应替换这一层。

## 3. 当前已确认事实

### 3.1 `frontend_brick` 公式已对齐

公开 K 线页前端暴露的砖形图公式已经复核：

```text
HHV4 = 最近 4 根 K 线最高价
LLV4 = 最近 4 根 K 线最低价
RANGE = HHV4 - LLV4

var2 = SMA((HHV4 - close) / RANGE * 100 - 90, 4, 1)
var4 = SMA((close - LLV4) / RANGE * 100, 6, 1)
var5 = SMA(var4, 6, 1)

brick_height = max(0, var5 - var2 - 4)
frontend_delta = brick_height - REF(brick_height, 1)
```

单票 `002485` 与官方 `daily-bars[].brick` 对齐到约 `0.0013` 误差；公开候选池字段层相关性也很高：

| 字段关系 | 相关性 |
| --- | ---: |
| `official_brick` vs `frontend_brick` | 0.9975 |
| `official_prev_brick` vs `frontend_prev` | 0.9998 |
| `official_delta/reversal_h` vs `frontend_delta` | 0.9504 |
| `official_decline_sum_5d` vs 本地 decline sum | 0.9994 |

因此后续砖形图复现必须使用 `frontend_brick`，不能再用 QuantX 当前 `BrickChart(...)` 的差分口径直接代替。

### 3.2 单独红砖反转在全样本里失败

已完成严格 OOS baseline：训练 2016-2024，OOS 2025-2026，不使用公开 Top5、公开 score/rank。

结果：

| 指标 | 结果 |
| --- | ---: |
| 候选标签均值 train | -1.24% |
| 候选标签均值 OOS | -1.34% |
| 模型 OOS Top7 标签均值 | -0.59% |
| OOS 回测总收益 | 1.59% |
| OOS 最大回撤 | -22.03% |
| 2025 收益 | 0.01% |
| 2026 收益 | -2.29% |

分层显示：高 `ret1/ret3/ret5`、高实体、高收盘位置、高振幅、高砖高度、高均线乖离的候选，在全样本 OOS 中后续收益更差。

当前解释：单独“红砖反转”容易变成追高回撤；必须加活跃市值 regime 才可能接近原策略逻辑。

### 3.3 官方公开 score 不能跨版本混训

公开 screener 在 2026-07-03 前后 score 口径切换明显。7/3 后 `score == factor_values.v11_score` 基本成立，但前段不是同一量纲。

结论：公开 score 只能作为 sanity check，不能当成统一训练标签；正式研究仍应使用本地历史 forward label。

## 4. 当前本地活跃市值近似口径

### 4.1 第一版近似

本地暂定：

```text
active_value[t] = sum_all_symbols($amount[t])
```

原因：日线 `$amount` 近似 `成交量 * 当日均价`，与用户提供的活跃市值核心公式一致。

### 4.2 入场 regime

根据用户转述，当前主口径应为：

```text
active_ret1 = active_value[t] / active_value[t-1] - 1
active_ret2 = active_value[t] / active_value[t-2] - 1

active_start = active_ret1 >= 4% or active_ret2 >= 4%
active_entry_window = active_start 当天到后续 2 个交易日
```

注意视频里同时出现过“1 到 3 天”和“1 到 2 天”，但用户后续转述明确说最终策略是“1-2 天内涨幅 4% 入场”。因此下一轮主实验用 `1-2 天 >=4%`，可以把 `1-3 天` 作为对照。

### 4.3 活跃市值波段退出

根据第二段视频：

```text
active_ma10 = SMA(active_value, 10)
active_wave_exit = active_value < active_ma10
```

MA10 是普通移动均线。

旧文档中曾使用 MA5 退出，那是旧近似；现在应改为 MA10。

### 4.4 个股买入候选

视频口径强调“高涨幅”和“红砖增量”，因此初始候选不应再低追。当前候选建议：

```text
in active_entry_window
and ret1 >= 4%
and frontend_delta > 0
and amount_rank_pct >= 0.35
and amplitude_pct < 22%
and not ST / 退市名
and not ST-like 5% limit history proxy
and not signal-day one-price limit-up
and volume/amount > 0
```

排序基准可以先做两类：

1. 规则排序：`ret1_rank + frontend_delta_rank + brick_rank + amount_rank`。
2. 非蒸馏模型排序：用训练期 forward label 学 `label_rank_blend` 或 `label_blend`。

视频里明确说模型不是主要收益来源，所以第一轮必须同时报告“规则排序”和“模型排序”，不能只看模型。

### 4.5 个股卖出策略

按视频口径：

```text
绿砖卖出
or 连续 7 块红砖卖出
```

本地 `frontend_brick` 语义下，建议先定义：

```text
red_brick = frontend_delta > 0
green_exit = frontend_delta < 0
red7_exit = Sum(red_brick, 7) >= 7
```

需要注意：公开页面/作者是否把红绿砖定义为 `delta > 0/<0`，还是用视觉状态 `brick_height` 变化、或另一个三线状态，仍需通过公开 K 线页继续核验。但目前 `frontend_delta` 是最接近“红砖增量”的本地可计算字段。

### 4.6 可交易性过滤

之前 exp40/Brick 失败说明必须严控：

1. 当前 ST/退市名过滤。
2. 历史 ST 代理：过去窗口内多次 5% 限幅且无 10% 限幅。
3. 停牌/缺失行情过滤。
4. 零成交/零成交额过滤。
5. 一字涨停不可买过滤。
6. 一字跌停/不可卖需在卖出层审计。
7. 普通高开不应一刀切过滤，因为视频强调强势高涨幅可能是收益来源；但高开应做诊断项。

## 5. 已有/新增实验脚本状态

### 5.1 已完成的 Brick 公式和负例实验

关键目录：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/
```

重要产物：

```text
analyze_frontend_brick_formula.py
frontend_brick_alignment_report_frontend.json
frontend-oos-baseline/frontend_oos_baseline_result.json
frontend-oos-baseline/frontend_filter_grid_v1.json
frontend-oos-baseline/frontend_regime_probe_v1.json
```

结论：公式对齐，但无活跃市值 gate 时 alpha 不成立。

### 5.2 新建但未跑完的 active-value 复现脚本

新增脚本：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/run_active_value_frontend_brick_oos.py
```

当前脚本意图：

1. 用全市场 `$amount` 汇总近似活跃市值。
2. 用 `1d/2d >= 4%` 开启 active wave。
3. 用 `active_value < MA10(active_value)` 退出 active wave。
4. 支持两种候选模式：
   - `active_start2`：只保留启动当天到后续 2 个交易日。
   - `active_wave`：保留完整活跃波段。
5. 使用已对齐 `frontend_brick`。
6. 用本地 forward label 训练 HGB 排序，不使用公开 Top5/score/rank。
7. 暂用自定义轻量账户回放，因为正式 factor engine 尚未内置 `frontend_brick` operator。

当前执行状态：已启动过 `active_start2` 全市场长跑，但用户要求先整理文档，所以已停止。进度停在数据读取/面板构造阶段，尚无结果结论。

已有 progress：

```text
start=2013-01-01
train_end=2024-12-31
oos_start=2025-01-02
end=2026-07-15
universe=all_a
rows=17,080,084
```

## 6. 关键未解问题

### 6.1 0AMV 指数本体是否可从 QMT 获取

用户提供定义表明 `0AMV` 是指南针系统中的活跃市值指数，包含 OHLC。当前本地仅用全市场 `$amount` 近似 close。

待验证：

```text
QMT 是否能直接拉取 0AMV / 活跃市值指数？
如果能，字段名、代码、交易日历、复权/单位是什么？
```

若能获取，应替换：

```text
active_value = sum($amount)
```

为真实：

```text
active_open/high/low/close = QMT 0AMV OHLC
active_ret = active_close pct_change
active_ma10 = SMA(active_close, 10)
```

### 6.2 活跃市值“涨幅 4%”用 close 还是 high

视频里说“1-2 天内涨幅 4%”，但 0AMV 有 OHLC。当前本地近似只用 close-to-close。

可能口径：

1. `active_close / REF(active_close, 1) - 1 >= 4%`
2. `active_high / REF(active_close, 1) - 1 >= 4%`
3. 2 日 close-to-close 涨幅。
4. 1-2 天累计涨幅。

第一版先用 close，因为本地近似只有 close；若获得 0AMV OHLC，需要做对照。

### 6.3 个股“涨幅 4%”和 T 字定义

用户转述中“T字”定义为活跃市值确认后两天以内涨幅 4% 以上的当天。这里的涨幅更像个股当日涨幅，而不是活跃市值自身涨幅。

需要区分：

```text
active_start: 活跃市值 1-2 天涨幅 >= 4%
stock_T_day: active_start 后两天内，个股 ret1 >= 4%
```

当前脚本按此理解实现。

### 6.4 红砖/绿砖状态的精确定义

`frontend_delta > 0` 作为红砖增量是目前最合理近似；但仍需继续对照公开 K 线页颜色/数值。

待验证：

1. K 线页是否用 `brick - prev_brick > 0` 着红色。
2. 连续 7 块红砖是否等价于连续 7 天 `frontend_delta > 0`。
3. 绿砖卖出是否 `frontend_delta < 0`，还是另有红绿状态公式。

### 6.5 公开策略中的 ST/涨停污染

公开 screener 里曾出现 ST 雪发等样本，且部分历史交易存在不可买一字板。本地实盘/回测必须过滤这些污染。

这意味着：即便本地收益低于公开页，也不能为了对齐公开收益而保留不可交易/ST 污染。

## 7. 下一步实验规格

### 7.1 第一阶段：不训练模型，先验证候选土壤

必须先回答：活跃市值启动窗口内，强涨幅 + 红砖增量候选是否 label 层转正？

需要输出：

1. `active_start2` 候选数量、训练/OOS 年度分布。
2. 全候选 forward return 均值/中位数/胜率。
3. 按 `ret1`、`frontend_delta`、`frontend_brick`、`amount_rank` 分层的未来收益。
4. 规则 Top5/Top10 标签层表现。
5. 2025、2026 分年表现。
6. entry 不可交易比例：一字涨停、零成交、缺失行情、高开 >=3% 诊断。

若候选池整体和规则 TopK 仍为负，说明本地 active-value 近似或红绿砖定义仍错，先不要训练模型。

### 7.2 第二阶段：非蒸馏 forward-label 排序

若第一阶段正期望成立，再训练模型：

```text
train <= 2024-12-31，且 exit_date <= 2024-12-31
OOS = 2025-01-02 到 2026-07-15
```

模型优先：

1. HGB `label_rank_blend` + top10/top20 尾部加权。
2. 规则分数作为 baseline。
3. 后续再考虑 pairwise / contrastive，不在第一版上来就深度学习。

### 7.3 第三阶段：账户回放

账户回放必须使用：

1. T 日信号，T+1 open 买入。
2. 买入过滤：停牌、零成交、一字涨停、ST/退市名。
3. 普通高开保留，但记录高开诊断。
4. 卖出：个股绿砖、连续 7 红砖、最大持有 10 日兜底。
5. 市场层：active_value 跌破 MA10 后，不再开新仓；是否强制清仓需要单独做对照。

需要报告：

| 指标 | 要求 |
| --- | --- |
| 总收益 | 先不硬卡，但需明显高于无 active gate 的 1.59% |
| 2025/2026 分年 | 必须分别报告 |
| 最大回撤 | 必须报告 |
| 平均持仓数 | 目标 > 5 |
| 平均持仓周期 | 约一周 |
| 买入违规 | ST、缺失行情、零成交、一字板必须为 0 |
| 重复买入集中度 | 必须审计 |

### 7.4 第四阶段：真实 0AMV 替换

如果 QMT 能获取 0AMV，则重跑：

1. `sum($amount)` proxy。
2. QMT 真实 0AMV close。
3. QMT 真实 0AMV high/close 混合触发。

比较 active wave 日期、波段数量、胜率和最终账户收益。

## 8. 当前判断

当前最重要的结论不是“砖形图失败”，而是：

```text
无活跃市值 gate 的红砖反转失败；
视频里的策略主因子很可能是活跃市值启动 regime，砖形图更多是该 regime 下的交易载体和退出机制。
```

因此后续复现优先级应为：

1. 先校准活跃市值指标和波段。
2. 再验证活跃启动窗口内的强涨幅/强红砖是否正期望。
3. 最后才训练排序模型。

不能继续在全样本红砖候选池里调阈值，也不能为了对齐公开收益引入 ST、一字板或 2025-2026 执行参数搜索污染。

## 9. 2026-07-15 实证复核

### 9.1 活跃市值增量资产

已新增并实际构建：

```text
data/derived/active_value/daily.parquet
data/derived/active_value/metadata.json
```

构建工具：

```text
quantx/tools/build_active_value_data.py
```

当前资产覆盖 `2013-01-04` 至 `2026-07-15`，共 3284 个交易日、5201 只股票。工具直接复用项目内 `QlibBinReader` 读取 Qlib 二进制行情；已有文件存在时默认只回看并重算最近 140 个交易日，再合并去重和重算状态指标。

主代理口径 `active_core_amount` 为：

```text
主板 + 非当前 ST/退市 + 非历史 ST-like + 上市满 120 个交易日
+ 当日可交易、非零成交
对应股票 $amount 求和
```

数据资产同时保存每日 bar/tradable 覆盖率和 `is_complete_day`。当前交易日 15:30 前不生成有效状态；历史比较日任一不完整时，1 日/2 日强启动状态均不成立。

### 9.2 代理口径边界

质量过滤后，`active_core_amount_strong_up_day` 仍有 1621/3284 天。`active_core_amount` 日收益标准差约 15.85%，与全 A 成交额总和相关性约 0.983。

这证明高触发频率不是股票覆盖变化造成，而是口径本身造成：

```text
sum($amount) 是全市场当日成交额代理，
不是可直接视为指南针真实 0AMV 指数收盘值的高质量复刻。
```

因此当前资产适合验证“全市场成交活跃度 regime”，但报告中不得称为真实 0AMV。用户提供的 Tushare `doc_id=397` 已确认是 `stock_st` 历史 ST 股票接口，可用于改善历史 ST 过滤，不能提供活跃市值指数。

### 9.3 同价口径标签与回放审计

研究脚本已统一为：

```text
T 日信号
T+1 open 买入
固定 5 日：T+6 open 卖出
固定 10 日：T+11 open 卖出
```

标签和账户回放使用相同滑点、佣金及印花税。准时退出交易逐笔比较结果：

| 配置 | 审计交易 | 延迟退出 | 标签/回放相关性 | MAE |
| --- | ---: | ---: | ---: | ---: |
| rule Top10 fixed5 | 608 | 1 | 1.0000 | 1.63e-16 |
| rule Top10 fixed10 | 328 | 1 | 1.0000 | 1.81e-16 |
| brick momentum Top10 fixed10 | 328 | 1 | 1.0000 | 1.79e-16 |

延迟退出来自一字跌停限制，不属于计算错位。由此确认此前“候选均值为正、组合回放为负”的主要原因不是 forward label 与回放价格不一致，而是 TopK 排序、资金占用和候选时序集中后的真实组合结果。

### 9.4 OOS 结果与砖型图边际贡献

2025-01-02 至 2026-07-15 的可交易候选：

| 样本 | fwd10 净收益均值 | 中位数 | 胜率 |
| --- | ---: | ---: | ---: |
| 全砖型反转候选 | +0.679% | -0.439% | 46.05% |
| 当日涨幅 >= 4% | +1.126% | -0.548% | 45.82% |
| 最高砖型反转值分位 | +0.930% | - | 48.59% |
| 涨幅 >= 4% 且最高分位 | +1.575% | - | 51.50% |

真实组合回放：

| 配置 | 总收益 | 最大回撤 |
| --- | ---: | ---: |
| 砖型候选 + momentum Top10 fixed10 | +2.80% | -36.38% |
| active-only + momentum Top10 fixed10 | -36.76% | -54.43% |
| 砖型规则 Top10 fixed10 | -36.09% | -43.19% |
| 砖型规则退出 | -56.31% | -62.17% |

同一 momentum 排序下，砖型门槛相对 active-only 的配对日收益增量约 `0.119%/日`，简单年化约 `29.99%`，但配对 t 值仅 `1.26`，不足以证明稳定独立贡献。

双因子 OLS 使用全 A 等权收益和横截面 5 日动量多空收益。表现最好的砖型 momentum 组合市场 beta 约 `1.10`、动量 beta 约 `0.24`，日 alpha 约 `-0.826%`；其绝对收益主要不能解释为稳定独立 alpha。

### 9.5 ML 结论与当前门禁

HGB OOS Top10 的 fwd10 净收益均值约 `+0.215%`，高于规则 Top10 的 `-0.184%`；但中位数约 `-2.99%`、胜率约 `41.39%`，且 2026 年均值约 `-0.799%`，出现明显衰减。

当前正式策略接入门禁未通过：

1. 本地 `sum($amount)` 仍不是可信的真实 0AMV 复刻。
2. 砖型边际贡献方向为正但统计不显著。
3. 规则组合和砖型退出组合显著亏损。
4. ML 在 2026 年失效，尚未完成 walk-forward 稳定性验证。

下一阶段应优先获取真实 0AMV OHLC，或依据原始公式进一步还原“活跃筹码市值”而非每日成交额；在此之前，不把当前研究模型接入正式 QuantX 策略配置。

## 10. 2026-07-15 第二轮：模拟 0AMV、市场宽度与严格 Walk-Forward

### 10.1 可复用数据资产

当前已落地三类增量数据：

1. `data/derived/0amv/daily.csv`
   - `0AMV_SH / SZ / KC / CY / ALL` 五个虚拟标的。
   - `close = 当日成分股 amount 聚合`，`open = 前日 close`，high/low 取二者极值。
   - 这是成交额活跃度模拟 K 线，不是真实指南针 0AMV。
2. `data/derived/active_value/daily.parquet`
   - 新增全市场右侧占比、核心股票池右侧占比、KDJ-J<13 右侧占比及成交额加权占比。
   - 右侧定义：`close > 黄线 and 白线 > 黄线`。
   - 白线：`EMA(EMA(close, 10), 10)`。
   - 黄线：`(MA14 + MA28 + MA57 + MA114) / 4`。
3. `data/reference/security_state/`
   - Tushare `stock_st` 覆盖 2016-08-31 至 2026-07-15，共 2394 个交易日。
   - 2370 个原始有效日，24 个空返回日全部沿用前一有效交易日，0 个失败日。
   - `st_daily.csv` 共 333507 行，`st_intervals.csv` 共 925 个连续区间。
   - 信号日和 T+1 均使用真实历史 ST 状态过滤；候选因此剔除 870 条记录。

2026-07-15 的核心股票池右侧占比约 `7.82%`，成交额加权右侧占比约 `38.57%`。这说明少数大成交股票能够维持指数或资金表面的强势，但多数股票已经不在右侧，是解释 2026 年砖型反转失效的重要状态变量。

### 10.2 横截面关系

在上证均线多头区间内，无条件砖型反转仍为负期望：

| 样本 | 5 日净收益 | 10 日净收益 |
| --- | ---: | ---: |
| 全候选 | -0.67% | -0.69% |
| 2025-2026 OOS | -0.34% | -0.24% |

反转值从最低到最高五分位，10 日均值约从 `-0.46%` 改善到接近 `0%`，但 Spearman 相关性仅约 `0.02`。每日 Top1 有相对候选池超额，但 2025/2026 方向相反，不能证明收益与反转值稳定成正比。

个股 KDJ 提供了有效但较弱的筛选信息：

- J 值最低 20% 的 OOS 10 日均值约 `-0.73%`。
- J 值中间区域约 `+0.06%`。
- J 日增量最高 20% 约 `+0.09%`。

因此视频线索“不要 J 很低、要 J 增量大”与数据方向一致，但 KDJ 不能单独把候选池转成稳定正期望。

### 10.3 模拟 0AMV 状态的边界

把 `0AMV_ALL` 的固定 `+4%` 直接当真实 0AMV 启动并不可靠。该模拟序列的 1/2 日变化波动过大，固定阈值触发频率约 40%，与专有活跃市值指数的经济含义不一致。

本轮改用无未来信息的高质量复刻状态：

```text
amv_shock = max(0AMV_ALL ret1, ret2)
threshold = prior 252 trading days q85(amv_shock)
start = amv_shock > 0 and amv_shock >= threshold
wave exit = 0AMV_ALL close < MA10
```

阈值在信号日前 `shift(1)`，只依赖历史数据。这个状态适合作为资金活跃度代理，但报告和策略命名必须保留“proxy/模拟”字样。

### 10.4 无泄漏 Walk-Forward

Walk-forward 规则：

- 每月重新训练扩展窗口 HGB。
- 预测月训练数据必须满足 `label_end < 当月首日`。
- 标签为 T+1 open 到 T+11 open 的 10 日净收益每日横截面排名。
- 股票池要求：模拟 0AMV 波段、上证多头、砖型反转、当日涨幅 >=4% 但未涨停、个股右侧、信号日/T+1 非历史 ST。

OOS 共 103 个信号日。标签层 TopK：

| 方法 | TopK | 10 日净收益均值 |
| --- | ---: | ---: |
| ML | 3 | +1.03% |
| 规则分 | 3 | -0.54% |
| 反转值 | 3 | -0.33% |
| 动量 | 5 | +1.10% |
| 单次固定随机 | 3 | +1.45% |

ML Top3 相对规则和反转值的日度配对超额为正，但 10 日块 bootstrap 的 95% 区间均穿过零；相对动量的正超额概率约 61.5%。因此不能把标签层结果解释为独立 ML alpha。

### 10.5 账户回放与盈利门禁

账户回放使用 100 万初始资金、100 股整手、最低 5 元佣金、双边滑点、印花税、涨跌停和停牌约束。固定 10 日逐笔收益与标签的相关性为 1，MAE 约 `1e-16`。

| 配置 | 总收益 | 最大回撤 |
| --- | ---: | ---: |
| ML Top3 fixed10 | -22.53% | -28.71% |
| 规则 Top3 fixed10 | -3.51% | -26.36% |
| 反转值 Top3 fixed10 | -4.69% | -16.28% |
| ML Top3 绿砖/红砖退出 | -22.82% | -47.85% |
| 规则 Top3 绿砖/红砖退出 | -32.68% | -44.12% |
| 反转值 Top3 绿砖/红砖退出 | -55.45% | -56.94% |

50 组确定性随机砖型退出账户的总收益中位数为 `-23.64%`，5%-95% 区间约 `[-53.11%, +19.08%]`。ML 砖型退出只位于随机分布约第 54 百分位，未展示稳定选股能力。

持有期/仓位敏感性中，表现最好的是动量 Top5 fixed10：总收益 `+14.13%`、最大回撤 `-27.17%`，但收益主要来自 2026，2025 仅约 `+1.98%`。这更像市场 beta/动量暴露，仍不满足成熟策略门禁。

当前盈利门禁保持关闭，原因：

1. 模拟 0AMV 不是专有活跃市值真值，固定 +4% 不能原样迁移。
2. 砖型反转值只有弱排序信息，不存在稳定强正相关。
3. ML 相对动量和随机基准没有显著、稳定的独立增量。
4. 账户 NAV、最大回撤和分年一致性均未达标。
5. 绿砖/连续红砖退出在当前口径下明显拖累收益。

在门禁通过前，不生成正式 QMT 策略配置。下一轮应围绕真实/更高质量活跃筹码口径、行业主线强度、资金集中度和可执行的账户级目标训练，而不是继续只优化候选标签均值。

## 11. 2026-07-15 第三轮：资金聚焦结构与组合感知 Walk-Forward

### 11.1 研究改动与无泄漏边界

本轮没有使用 `2026-06-25` 的静态行业/概念快照。该快照不带历史生效日期，且概念中包含“昨日涨停”“昨日连板”等动态板块，直接回填历史会产生穿越。

改为从既有 1708 万行全 A 面板按批次聚合 13 个信号日可观测的资金结构特征：

- 全市场成交额 Top10/20/50 占比；
- 上涨股票、涨幅 >=4% 股票、砖型反转股票的数量与成交额占比；
- 当日可交易候选数量及 5 日变化；
- 候选池成交额 Top3/Top5 集中度；
- 个股成交额占全市场比重；
- 全市场 Top20 成交集中度的 5 日变化。

首次扫描结果缓存为：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/daily_market_structure.parquet
```

缓存约 3.7 MB，并用源 Parquet 的大小和修改时间自动失效。月度扩展窗口仍要求 `label_end < 预测月首日`，15 个 OOS 折叠全部通过 purge。正式模型未使用静态行业或动态概念数据。

### 11.2 严格特征消融

同一月度切分、随机种子、HGB 参数和样本下同时训练：

1. `ml_base`：原有特征，直接预测 10 日净收益；
2. `ml`：原有特征 + 13 个资金结构特征，直接预测 10 日净收益；
3. `ml_base_rank`：原有特征，预测同日收益分位；
4. `ml_rank`：原有特征 + 13 个资金结构特征，预测同日收益分位。

直接收益模型按候选日倒数加权，避免候选特别多的日期支配训练；预测值不大于零时允许账户保留现金。排名模型保留旧目标，作为可复现消融基线。

Top3 标签层结果：

| 模型 | 10 日净收益均值 | 2025 | 2026 |
| --- | ---: | ---: | ---: |
| 完整直接收益 | +2.47% | +1.28% | +4.32% |
| 基础直接收益 | +1.08% | +0.53% | +2.01% |
| 完整排名 | +1.64% | +0.73% | +3.50% |
| 基础排名 | +1.03% | +0.45% | +2.20% |
| 规则分 | -0.54% | -1.83% | +2.07% |
| 反转值 | -0.33% | +0.65% | -2.31% |

完整直接收益模型相对基础模型的 10 日标签日度配对增量约 `+1.34%`，块 bootstrap 正向概率约 `93.8%`，但 95% 区间仍穿零。标签层方向明显改善，尚不能单独证明稳定 alpha。

### 11.3 账户结果与资金结构增量

账户仍使用 T 日信号、T+1 开盘、100 股整手、最低 5 元佣金、双边滑点、印花税、停牌及一字涨跌停约束。账户先卖出，再仅在存在空余仓位时选股。

固定 10 日 Top3：

| 配置 | 总收益 | 最大回撤 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: |
| 完整直接收益 | +38.49% | -27.97% | +10.07% | +25.82% |
| 基础直接收益 | -4.66% | -25.18% | -0.16% | -4.50% |
| 完整排名 | +57.02% | -13.30% | +22.66% | +28.01% |
| 基础排名 | -22.53% | -28.71% | -16.01% | -7.76% |
| 规则分 | -3.51% | -26.36% | -14.16% | +12.40% |
| 反转值 | -4.69% | -16.28% | +2.32% | -6.85% |
| 动量 | -36.55% | -38.25% | -36.75% | +0.31% |

基础排名模型精确复现上一轮 `-22.53%`，加入资金结构特征后变为 `+57.02%`。完整排名相对基础排名的账户日收益增量约 `0.196%/日`，10 日块 bootstrap 95% 区间约 `[+0.056%, +0.362%]`，正向概率 `99.8%`。这提供了本轮最强的证据：改善主要来自资金结构，而不是回放或切分变化。

50 个同口径随机 fixed10 Top3 账户的总收益均值约 `-13.37%`、95% 分位约 `+15.95%`、最高约 `+35.44%`。完整直接收益和完整排名账户均超过本组随机账户最高值。

### 11.4 共识选择与可解释画像

没有新增模型或 OOS 阈值搜索。共识选择仅组合当月已训练分数：

```text
ml_score > 0 决定是否参与
ml_rank_score 决定同日 Top3 排序
```

共识 fixed10 Top3 总收益 `+40.06%`、最大回撤 `-24.76%`，2025/2026 分别约 `+7.82%`、`+29.91%`，位于随机 fixed10 分布 100 百分位。基础共识 fixed10 为 `-15.30%`。

完整模型实际买入相对同日候选池，在 2025 和 2026 均呈现同方向画像：

- 候选数量更少，且候选数量近 5 日在收缩；
- 候选池成交额更集中于 Top3/Top5；
- 全市场涨幅 >=4% 和砖型反转扩散度更低；
- 资金尚处于少数领涨标的聚焦，而不是全面拥挤扩散；
- 个股仍偏强实体、一定砖反转强度；KDJ 仅提供次级信息。

因此更接近视频策略的可解释表述是：

```text
活跃度启动后，参与资金聚焦于少数强势、右侧、砖型反转标的的阶段；
不是无条件追全市场最强涨幅，也不是反转值越大收益必然越高。
```

### 11.5 审计边界与生产门禁

交易/Nav 产物现已带完整配置名；63 个账户配置内没有重复的同日同股票买单，7220 笔买入均有对应卖出。

`fwd5_net/fwd10_net` 标签按比例佣金估算；账户回放使用实际股数、整手和最低 5 元佣金。小仓位配置因此存在少量费用差，不是行情日期错位。共识 Top3 fixed10 的逐笔标签/回放相关性约 `0.999999998`，最大差约 `0.0071%`；其量级与最低佣金差一致。

生产门禁仍为 `CLOSED`：

1. 共识 fixed3 为 `-47.50%`，fixed5 为 `-2.78%`，砖型动态退出为 `-9.63%`，收益强依赖固定 10 日退出；
2. 共识 fixed10 的市场/动量双因子年化 alpha 仍约 `-13.43%`；完整直接收益模型约 `-51.58%`；
3. 共识相对基础共识的账户增量 95% 区间仍穿零；
4. 模拟 0AMV 仍是成交活跃度代理，不是真实活跃筹码市值；
5. 当前 OOS 仅 103 个信号日、53-54 笔 fixed10 买入，样本仍偏少。

本轮首次发现可复现且经济含义清晰的资金聚焦效应，但尚未达到成熟策略标准。下一阶段优先验证退出机制和真正的相对基准收益，扩展历史有效信号，并将全量账户回放的约 17 GB 峰值内存降下来；门禁通过前仍不生成正式 QMT 策略配置。

## 12. 2026-07-15 第四轮：Stop6 退出与正式回测复核

### 12.1 退出机制

上一轮最大问题是收益强依赖固定 10 日退出，砖型动态退出和短持有都失效。本轮只测试事先可解释的少量退出规则，且均按“决策日可见信息，下一交易日开盘执行”的无未来口径：

- `stop6`：持仓后收盘亏损达到 `-6%`，次日开盘卖出；
- `take12`：持仓后收盘盈利达到 `+12%`，次日开盘卖出；
- `trail8_6`：盈利达到 `+8%` 后，从最高收盘回撤 `6%`，次日开盘卖出；
- `breadth`：核心右侧占比跌破自身 EMA20，次日开盘卖出；
- `amv`：模拟 0AMV 波段结束，次日开盘卖出；
- `combined`：stop6 + trail8_6 + breadth + amv。

同时修正归因口径：除原始日历回归外，新增按实际股票敞口缩放市场/动量因子的 exposure-matched alpha，以及仅持仓日 alpha。此前负 alpha 有一部分来自账户大量空仓日与全日历市场因子的口径不匹配。

### 12.2 临时审计回放结果

共识 Top3 的核心退出对比：

| 退出 | 总收益 | 最大回撤 | 2025 | 2026 | 买入 | 平均持仓 | 主要卖出原因 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed10 | +40.06% | -24.76% | +7.82% | +29.91% | 53 | 10.00 | fixed10: 53 |
| brick exit | -9.63% | -32.72% | -16.14% | +7.77% | 139 | 2.60 | green: 129, 7red: 10 |
| stop6 | +141.80% | -18.17% | +92.43% | +25.66% | 68 | 7.41 | max_hold_10: 40, stop_loss: 28 |
| take12 | +16.18% | -22.76% | +0.82% | +15.23% | 57 | 9.18 | max_hold_10: 44, take_profit: 13 |
| trail8_6 | +36.95% | -27.05% | +9.13% | +25.49% | 53 | 9.83 | max_hold_10: 47, trailing_stop: 6 |
| breadth | +27.51% | -23.83% | +12.22% | +13.62% | 109 | 3.69 | breadth_break: 88, max_hold_10: 21 |
| amv | -8.66% | -35.13% | -25.03% | +21.83% | 63 | 4.92 | amv_wave_end: 57, max_hold_10: 6 |
| combined | -19.52% | -29.48% | -20.44% | +1.15% | 119 | 2.45 | breadth/amv/stop/trail mixed |

`stop6` 是唯一同时提升收益、降低回撤并保持两年为正的退出。50 个同口径随机 `stop6` 账户总收益均值约 `-22.73%`，95% 分位约 `+16.56%`，最高约 `+42.84%`；`ml_consensus + stop6` 位于同出口随机分布 100 百分位。

敞口匹配归因：

| 配置 | 原始日历 alpha | 敞口匹配 alpha | 平均股票敞口 | 市场 beta |
| --- | ---: | ---: | ---: | ---: |
| fixed10 | -13.43% | +4.25% | 47.7% | 1.31 |
| stop6 | -22.45% | +21.61% | 46.5% | 1.27 |

这说明原始日历 alpha 对大量空仓账户并不公平；按实际敞口缩放后，`stop6` 具备正 alpha，但 beta 仍偏高。

### 12.3 风险退出门禁

本轮新增的研究候选门禁 `ml_consensus_top3_risk_stop6` 已打开：

```text
total_return > 0: true
2025 > 0 and 2026 > 0: true
max_drawdown > -25%: true
exposure_matched_alpha > 0: true
same_exit_random_percentile >= 95%: true
buys >= 60: true
```

证据：总收益 `+141.80%`，最大回撤 `-18.17%`，2025 `+92.43%`，2026 `+25.66%`，敞口匹配年化 alpha `+21.61%`，买入 68 笔。

注意：这是“研究候选门禁打开”，不是实盘门禁打开。原因是 OOS 信号仍只有 92 个信号日、68 笔买入，且收益分布右尾很重。

### 12.4 正式回测入口复核

已生成正式研究配置：

```text
configs/strategies/generated/active_value_brick_ml_consensus_stop6_2025_2026.yaml
```

对应外部分数资产：

```text
data/derived/active_value_brick_ml/ml_consensus_scores_2025_2026.parquet
data/derived/active_value_brick_ml/metadata.json
```

分数资产来自已审计 walk-forward 预测，仅保留 `ml_consensus_score` 非空记录，共 2137 行、92 个信号日、1549 只股票。正式配置使用：

候选样本、训练与导出命令：

```bash
conda run -n test python -m quantx.tools.build_active_brick_candidates \
  --regime-path data/derived/market_regime/sh000001_daily.parquet \
  --st-daily-path data/reference/security_state/st_daily.csv \
  --candidates-output .tmp/quantx-research/active-value-frontend-brick-v1/sh_index_bull_brick_candidates.parquet \
  --json-output .tmp/quantx-research/active-value-frontend-brick-v1/sh_index_bull_brick_relationship.json \
  --markdown-output .tmp/quantx-research/active-value-frontend-brick-v1/sh_index_bull_brick_relationship.md \
  --oos-start 2025-01-02 \
  --end 2026-07-15 \
  --json
```

迁移后的候选生成工具与原 `.tmp` 研究脚本输出样本完全一致：`sh_index_bull_brick_candidates.parquet` 为 508,921 行、75 列，按 `datetime/instrument` 排序后 dataframe hash 相等。报告中的 `market.bull_days` 是从当前 `data/derived/market_regime/sh000001_daily.parquet` 即时读取的派生摘要；若 regime 资产后续重建，该摘要会随之变化，但候选样本对齐检查以 parquet 为准。

默认底层 replay/base 逻辑来自正式模块 `quantx.core.research.active_brick_replay`；`--base-script` 仅保留为兼容旧 `.tmp` 研究脚本的可选参数。

标准训练与导出命令：

```bash
conda run -n test python -m quantx.tools.train_active_brick_ml_scores \
  --candidate-path .tmp/quantx-research/active-value-frontend-brick-v1/sh_index_bull_brick_candidates.parquet \
  --active-path data/derived/active_value/daily.parquet \
  --0amv-path data/derived/0amv/daily.csv \
  --panel-path .tmp/quantx-research/active-value-frontend-brick-v1/panel_all_a_20130101_20260715.parquet \
  --market-structure-path .tmp/quantx-research/active-value-frontend-brick-v1/daily_market_structure.parquet \
  --market-structure-meta-path .tmp/quantx-research/active-value-frontend-brick-v1/daily_market_structure.meta.json \
  --scores-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_scores.parquet \
  --json-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_ml.json \
  --markdown-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_ml.md \
  --oos-start 2025-01-02 \
  --end 2026-07-15 \
  --export-score-output data/derived/active_value_brick_ml/ml_consensus_scores_2025_2026.parquet \
  --export-metadata-output data/derived/active_value_brick_ml/metadata.json \
  --export-min-rows 2000 \
  --export-min-signal-days 90 \
  --json
```

迁移后的正式训练工具与原 `.tmp` 研究脚本逐行对齐：8236 行、103 个候选交易日、15 个 walk-forward folds 一致，`ml_score`、`ml_rank_score`、`ml_base_score`、`ml_base_rank_score`、`ml_consensus_score`、`ml_base_consensus_score`、`random_score` 最大绝对差异均为 0。

仅重新导出已审计分数资产时，可使用：

```bash
conda run -n test python -m quantx.tools.export_active_brick_ml_scores \
  --source .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_scores.parquet \
  --output data/derived/active_value_brick_ml/ml_consensus_scores_2025_2026.parquet \
  --metadata-output data/derived/active_value_brick_ml/metadata.json \
  --start 2025-01-02 \
  --end 2026-07-15 \
  --min-rows 2000 \
  --min-signal-days 90 \
  --json
```

导出工具会记录源文件 SHA256、输出 SHA256、源列、行数、信号日数和分数范围。当前输出 SHA256 为 `sha256:46afd9a2a51bb4e159794db478e436a8cd1afa43d5f7d439d62c3fdfb7e1359a`。

账户级审计 replay 命令：

```bash
conda run -n test python -m quantx.tools.run_active_brick_portfolio_replay \
  --scores-path .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_scores.parquet \
  --json-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_portfolio_replay.json \
  --markdown-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_portfolio_replay.md \
  --trades-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_portfolio_replay_trades.parquet \
  --nav-output .tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_portfolio_replay_nav.parquet
```

迁移后的 replay 工具与原 `.tmp` replay 产物对齐：69 个 summary、`risk_exit_gate`、`production_gate`、3 组随机分布、trades parquet、nav parquet 均一致。`ml_consensus_top3_risk_stop6` 仍为研究候选门禁 `OPEN`：总收益 `+141.80%`，最大回撤 `-18.17%`，买卖各 68 笔，平均持仓 7.41 天；同出口随机 stop6 的 95% 分位为 `+16.56%`。

默认 replay 路径已验证使用正式 base 模块和正式 ML helper：单配置 `ml_consensus_top3_risk_stop6` 复现总收益 `+141.80%`、最大回撤 `-18.17%`、买卖各 68 笔。

账户回放公共模块已做第一轮低风险内存瘦身：`build_replay_context` 现在只把回放实际需要的 `REPLAY_CONTEXT_COLUMNS` 放入逐股票上下文，包括 `instrument`、`datetime`、`$open`、`$close`、`$volume`、`$amount`、`basic_tradable`、`one_price_limit_up`、`one_price_limit_down`、`frontend_delta`、`red_run7`；`active_by_date` 仍从原始 panel 中单独读取 `active_above_ma10`。随后进一步把 replay context 行范围裁剪到 `OOS_START` 前 20 个交易日到 `END`，本轮实测 `context_start=2024-12-04`。单配置 `ml_consensus_top3_risk_stop6` 已复验结果不变：总收益 `+141.80%`、最大回撤 `-18.17%`、买卖各 68 笔，卖出原因仍为 `max_hold_10: 40`、`stop_loss: 28`。full replay 复验仍为 69 个 summary、`risk_exit_gate=OPEN`、`production_gate=CLOSED`；`/usr/bin/time -l` 显示 RSS 从约 `16.99GB` 小降至约 `16.53GB`，但 `peak memory footprint` 仍约 `24.92GB`。这说明裁剪不改变语义，但主要内存压力仍在 panel 构建/前处理阶段，而不是 replay context 本身。

换手率基础字段已确定写入 Qlib feature 主链路，而不是只放在派生 parquet 中。`quantx.tools.build_turnover_features` 使用本地 Qlib 日线 `close/volume/amount` 和 QMT historical financial `Capital` 表写入每只股票的 `turnover_rate.day.bin`、`turnover_rate_circulating.day.bin`、`turnover_rate_free_float.day.bin`、`circ_mv.day.bin`、`free_float_mv.day.bin`。为避免股本公告未来函数，股本匹配日期使用 `effective_date = max(m_timetag, m_anntime)` 并向后匹配到交易日。默认 `$turnover_rate` 对齐东方财富常见换手率口径，使用自由流通股本：`volume * 100 / freeFloatCapital`；全流通股本口径保留为 `$turnover_rate_circulating = volume * 100 / circulating_capital`。`SH600000` 在 `2026-07-15` 的读回验证为 `$turnover_rate=0.296048%`、`$turnover_rate_circulating=0.158826%`，与东方财富显示的 `0.31%` 基本对齐。

日常增量维护也已接入同一条主链路：`quantx.tools.sync_daily_data --source qmt` 在非 dry-run 且日线更新成功后默认刷新换手率 feature，不需要再额外生成 `data/derived/active_capital/daily.parquet`。刷新起点默认读取 `data/qlib_data_fixed/metadata/turnover_features.json` 的上一轮 `end`，向前回补 90 个自然日；首次没有 metadata 时才从 `--full-refresh-start` 全量构建。`write_qlib_features` 写入 `.day.bin` 时会先合并已有文件，短窗口增量刷新只覆盖窗口内日期，窗口外历史值保留。需要临时跳过时使用 `--skip-turnover-features`；需要强制全量或指定窗口时使用 `--turnover-start YYYY-MM-DD`。

全量刷新命令：

```bash
conda run -n test python -m quantx.tools.build_turnover_features \
  --provider-uri data/qlib_data_fixed \
  --output-provider-uri data/qlib_data_fixed \
  --start 2010-01-01 \
  --end 2026-07-15 \
  --json
```

```text
selector.mode = external_score
lag = 1
topk = 3
max_positions = 3
deal_price = open
sell_decision_price = close
sell_rules = pnl_pct < -0.06 or holding_days >= 10
```

Dry-run 编译通过。

第一版正式 `quantx.tools.run_backtest` 曾按 `deal_price=open` 同时计算卖出规则和成交价，结果如下：

| 指标 | 数值 |
| --- | ---: |
| final_value | 2,115,027.62 |
| total_return | +111.50% |
| annual_return | +63.09% |
| max_drawdown | -19.01% |
| sharpe | 1.89 |
| buy/sell | 68 / 68 |
| closed positions | 68 |
| win rate | 47.06% |
| profit factor | 1.88 |
| avg capital utilization | 45.24% |
| total cost | 51,179.86 |

分年净值：2025 `+56.02%`，2026 `+35.56%`。该结果与临时审计回放方向一致，但止损触发点不完全一致。

随后正式策略执行层新增 `execution.sell_decision_price`，允许卖出规则按前一交易日收盘价判定，同时订单仍按当日开盘价执行。配置切换为：

```text
deal_price = open
sell_decision_price = close
```

前收判定版本正式回测输出目录：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/formal_backtest_close_decision/active_value_brick_ml_consensus_stop6_2025_2026_close_decision
```

当前主证据如下：

| 指标 | 数值 |
| --- | ---: |
| final_value | 2,198,352.75 |
| total_return | +119.84% |
| annual_return | +67.25% |
| max_drawdown | -18.25% |
| sharpe | 2.05 |
| buy/sell | 66 / 66 |
| closed positions | 66 |
| win rate | 53.03% |
| profit factor | 1.98 |
| avg capital utilization | 45.45% |
| total cost | 54,249.79 |

分年净值：2025 `+74.98%`，2026 `+25.64%`。相比第一版正式回测，前收判定口径更接近审计回放的退出语义；与审计回放仍有差异，主要来自正式引擎的撮合、现金复用、手续费滑点、持仓天数统计和候选执行细节。

交易集中度需要特别警惕：前收判定版本 66 笔交易胜率 `53.03%`，平均单笔收益 `+3.85%`，中位数 `+0.94%`，但右尾仍很厚。最大单笔收益约 `+48.25%`，Top3 单笔收益合计约 `+128.59%`，Top5 约 `+189.60%`，最差 Top5 合计约 `-55.05%`。这说明策略收益仍高度依赖少数强趋势票，后续必须做更长历史、更多市场阶段和尾部依赖验证。

### 12.5 真实换手率/市值口径后的流动性过滤

换手率与市值字段已进入 Qlib feature 主链路后，本轮把候选样本补齐 `turnover_rate`、`turnover_rate_rank_pct`、`free_float_mv`、`free_float_mv_rank_pct`、`active_free_mv_proxy`、`active_free_mv_proxy_rank_pct` 等字段。旧候选 parquet 从 75 列扩展到 93 列，508,921 行保持不变；新增字段覆盖率约 `99.89%`。备份文件为：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/sh_index_bull_brick_candidates.pre_turnover.parquet
```

直接把换手率/市值加入 walk-forward ML 后，增强模型并没有稳定优于旧基线：增强 `ml` 相对 `ml_base` 仅 Top1 为正增量，Top3/Top5/Top10 不稳。因此本轮转向把换手率/市值作为过滤条件。分层诊断输出：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/active_brick_liquidity_filter_diagnostics.json
.tmp/quantx-research/active-value-frontend-brick-v1/active_brick_liquidity_filter_diagnostics.md
```

核心发现：最高换手率、最高活跃自由流通市值并不是最优区域，过热档未来收益反而偏弱；更有效的是中低换手与中等自由流通市值组合。账户级轻量 replay 中，当前最优组合为：

```text
turnover_rate_rank_pct in [0.20, 0.60]
free_float_mv_rank_pct in [0.20, 0.80]
score = rule_score
topk = 3
exit = 6% stop loss or max holding 10 sessions
```

账户级过滤 replay 结果：

| 配置 | 总收益 | 最大回撤 | 买/卖 |
| --- | ---: | ---: | ---: |
| liquidity_turnover_020_060_mv_020_080_rule_top3_risk_stop6 | +86.49% | -13.76% | 66 / 66 |
| liquidity_turnover_060_100_ml_base_consensus_top3_risk_stop6 | +35.55% | -25.33% | 69 / 69 |
| liquidity_turnover_020_060_active_proxy_020_080_rule_top3_risk_stop6 | +23.07% | -30.27% | 62 / 62 |
| liquidity_all_ml_base_consensus_top3_risk_stop6 | +22.63% | -24.90% | 71 / 71 |
| liquidity_turnover_040_080_momentum_top3_risk_stop6 | -20.71% | -39.90% | 76 / 76 |

同过滤池随机排序对照也已完成：在 `turnover_rate_rank_pct in [0.20, 0.60]` 且 `free_float_mv_rank_pct in [0.20, 0.80]` 的同一候选池内，保留 `top3 + stop6 + max_hold10` 账户规则，仅把排序分数替换为 50 个固定随机种子。随机排序总收益分布为：均值 `+7.52%`、中位数 `+2.56%`、5% 分位 `-28.29%`、95% 分位 `+61.01%`、最大 `+69.81%`、正收益比例 `56.00%`。规则排序账户收益 `+86.49%`，超过同池随机 95% 分位，说明当前收益不是单纯来自过滤池行情，而是 `rule_score` 在该过滤池内有排序增量。

该组合的研究候选门禁已固化到：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/active_brick_liquidity_filter_replay.json
```

`liquidity_gate.status=OPEN`，检查项包括总收益为正、2025/2026 分年收益为正、最大回撤优于 `-20%`、超过同过滤池随机 95% 分位、买入样本数不少于 60。当前证据：总收益 `+86.49%`，2025 `+28.38%`，2026 `+45.27%`，最大回撤 `-13.76%`，同池随机 q95 `+61.01%`，买入 66 笔。

该候选已导出正式 external_score 资产：

```text
data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_scores_2025_2026.parquet
data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_metadata.json
```

资产规模为 906 行、100 个信号日、644 只股票；标签审计 `fwd10_net` 均值约 `+1.47%`、胜率约 `46.36%`。正式策略配置：

```text
configs/strategies/generated/active_value_brick_liquidity_rule_turnover20_60_mv20_80_stop6_2025_2026.yaml
```

正式 `quantx.tools.run_backtest` 结果如下：

| 指标 | 数值 |
| --- | ---: |
| final_value | 1,861,630.91 |
| total_return | +86.16% |
| annual_return | +50.05% |
| max_drawdown | -13.83% |
| buy/sell | 66 / 66 |
| rejects | 0 |
| final_positions | 0 |

交易集中度审计显示，该候选虽然比旧强右尾版本更稳，但仍依赖少数大赢家：66 笔卖出交易胜率 `51.52%`，平均单笔收益 `+3.78%`，中位数 `+0.65%`；Top1 单笔收益贡献约 `+40.69%`，Top3 合计约 `+113.88%`，Top5 合计约 `+174.77%`，最差 5 笔合计约 `-49.30%`。卖出原因为 `max_hold_10: 42`、`stop_loss: 24`。分年账户收益为 2025 `+28.38%`、2026 `+45.27%`，平均仓位暴露约 `46.91%`。

个股集中度审计输出为：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/liquidity_rule_symbol_concentration.json
```

66 笔卖出覆盖 62 只不同股票，单只股票最多交易 2 次，说明不是反复押注单一标的。但盈利仍有右尾集中：Top3 股票贡献正收益股票收益的 `32.04%`，Top5 贡献 `45.00%`。最大贡献标的为 `SH688652`，2 笔合计 `+71.15%`；其次为 `SH688257`，2 笔合计 `+42.92%`；`SZ002773`，1 笔 `+33.01%`。因此当前风险不是“个股复用过度”，而是“少数右尾强票贡献过高”。

轻量上证 beta 归因输出为：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/liquidity_rule_sh_index_attribution.json
```

用上证指数日收益解释策略日收益，样本 369 个交易日，`beta_sh000001=0.4485`，`R2=4.31%`，相关系数 `0.2075`，日 alpha 约 `+0.162%`，算术年化 alpha 约 `+40.86%`。这说明候选有市场 beta 暴露，但当前收益不能主要由上证 beta 解释。

进一步加入候选池横截面动量因子后，输出文件为：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/liquidity_rule_market_momentum_attribution.json
```

二元回归使用 `上证日收益 + 候选池高低 ret5_rank_pct 的 fwd10_net 价差` 解释策略日收益。有效样本 92 个交易日，`beta_sh000001=0.3953`，`beta_candidate_momentum_fwd10_spread=0.0199`，`R2=1.92%`；策略与上证日收益相关 `0.1143`，与候选池动量价差相关 `0.0783`，算术年化 alpha 约 `+91.88%`。样本仍短，但初步看该候选并非简单由市场 beta 或横截面动量暴露解释。

这说明真实换手率/市值口径在当前样本里更适合作为“避开过热、约束容量区间”的过滤器，而不是简单作为 ML 特征直接加权。该候选比旧 `ml_base_consensus` 过滤前组合在回撤上明显改善，但总收益低于旧 `ml_consensus` 强右尾版本；后续需要做更长 OOS、随机/因子归因复验和交易集中度审计。

生产前门禁汇总工具已新增：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.tools.assess_active_brick_production_readiness
```

当前输出：

```text
.tmp/quantx-research/active-value-frontend-brick-v1/active_brick_production_readiness.json
.tmp/quantx-research/active-value-frontend-brick-v1/active_brick_production_readiness.md
```

该工具把流动性 replay、正式回测、同池随机对照、个股集中度、行业/板块/概念集中度、walk-forward/OOS 标签层证据、active brick 日更分数桥接、上证 beta 归因、候选池动量归因、external score 资产和交易右尾压力审计汇总成两层门禁。当前结果为：`research_gate=OPEN`，`production_gate=CLOSED`。

关键证据：正式回测总收益 `+86.16%`，年化 `+50.05%`，最大回撤 `-13.83%`，买/卖 `66 / 66`，拒单 `0`；同池随机 50 次 q95 为 `+61.01%`，规则排序超过随机 95% 分位；external score 资产 `906` 行、`100` 个信号日、`644` 只股票；walk-forward 标签层审计为 `PASS`，共 `15` 个 fold、`8236` 行打分、`103` 个打分日，purge 规则为 `train label_end < prediction month first day`，覆盖 2025-02 到 2026-06；active brick 上游刷新链路审计为 `PASS`，`configs/production/daily_default.yaml` 的 `predictions` 阶段已调用 `quantx.tools.refresh_active_brick_liquidity_scores`，由一个 job 串起候选生成、walk-forward 打分、流动性 external_score 导出，并已把策略配置加入日更策略列表；`run_daily_pipeline --stage predictions --dry-run` 已实测该 job 返回 `ok=true`；交易覆盖 `62` 只股票，单票最多交易 `2` 次；行业审计覆盖率约 `89.39%`，最高行业为半导体，占可映射卖出交易约 `10.17%`，未见单一行业极端集中；板块分布为主板 `45.45%`、科创板 `43.94%`、创业板 `10.61%`；上证 beta `0.4485`，上证 R2 `4.31%`，候选动量二元归因 R2 `1.92%`。右尾压力审计已可读交易表的 `realized_return`：去掉 Top5 卖出收益后，剩余卖出收益代理仍为 `+74.40%`，说明不是纯靠 Top5 单笔维持为正。

行业/板块/概念集中度、标签层 walk-forward/OOS 证据、active brick 上游刷新链路接入 readiness 后，生产 blocker 从 3 个降为 2 个。生产门禁仍关闭，剩余 blocker：

1. `sample_long_enough`：当前正式样本为 2025-01-02 到 2026-07-15，不足 3 年；
2. `account_level_oos_long_enough`：已有标签层 walk-forward/OOS，但账户级 OOS 仍只覆盖 2025-2026，不足以称为生产级长样本。

### 12.6 当前结论

目前可以认为：

```text
活跃度代理启动 + 资金聚焦结构 + 砖型反转触发 + ML 共识选股 + 6% 止损/10日上限
```

已经从“研究失败”推进到“研究候选成立”。但成熟策略仍未完成，原因：

1. 0AMV 仍是成交额活跃度代理，不是真实活跃筹码市值；但 QMT 历史 financial 的 `Capital` 表已接入换手率/市值 Qlib feature，当前已可用 `$turnover_rate`、`$free_float_mv`、`$circ_mv` 做候选过滤和策略回测；
2. OOS 样本仍偏少；右尾压力审计已初步通过，但仍需更长样本验证；
3. 正式回测已支持“前一收盘决策、次日开盘执行”，但还需要更多成交细节和不可交易日边界审计；
4. beta 仍偏高，必须继续做市场阶段匹配和风险暴露控制；
5. 候选样本生成、walk-forward 训练、流动性分数导出、账户级 replay/随机对照入口、生产前 readiness 汇总、上游 active-brick scored panel 刷新链路已迁入正式工具，底层 panel/forward-return/replay 公共函数已抽入 `quantx.core.research.active_brick_replay`；后续重点转向长样本/OOS、真实 0AMV 口径和账户级生产门禁。

下一阶段可以进入项目回测层/研究策略候选管理，但仍不建议直接接实盘 QMT。
