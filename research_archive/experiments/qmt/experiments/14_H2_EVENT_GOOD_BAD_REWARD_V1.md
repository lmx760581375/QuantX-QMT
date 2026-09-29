# H2 事件化 Good/Bad Reward 实验方案

> 日期：2026-09-27
>
> 状态：方案落地，等待代码级 Review 结论；Review 通过后按本文直接实施。

## 1. 实验目标

本实验只回答一个问题：

> 保持现有日线 Frozen-AE Reward Transformer 模型和 Bradley-Terry
> pairwise loss 不变，仅将训练数据改成稀疏 H2 强事件 good 与全市场其余
> bad，模型能否在 2019 年 validation 上学出稳定的 Top-5 超额收益和排序能力。

不先做独立事件 Alpha 筛选，不新增模型，不改 AE，不改 Reward Transformer
结构，不增加额外预测 head。

固定时间合同：

```text
信号日：D，模型输入只到 D 收盘
买入日：D+1，以开盘价为策略买入价
退出日：D+2，以收盘价为固定退出价
训练：2010-01-01 至 2018-12-31
验证：2019 全年
正式回测：2020-01-01 至数据末日
训练轮数：20 epochs
```

现有 Kronos memmap 与 `market_all` builder 已经满足：

- 全 A 股、非指数；
- 输入窗口截止 D；
- D+1 开盘不可成交样本已由 `entry_feasible` 和
  `label_execution_invalid` 过滤；
- 与 Frozen AE checkpoint 使用同一 60 日输入 schema。

现有 `market_all_pre2020_v1` 的 fold 按 H30 标签 purge，直接复用会提前约
30 个交易日截断 2018 训练，也会提前截断数据末端 prediction。为严格使用
H2 可用数据，本实验复用同一基础特征与 AE，但通过现有
`build_wts_diffusion_dataset_v1.py` 新建轻量 H2 root：

```text
candidate_mode = all_market
universe = all_a
horizon = 2
target_start_offset = 1
target_price_mode = entry_open
```

该步骤只生成索引和 2 列未来目标，不复制 60 日输入特征。H2 fold 按
`label_end_date <= split_end` 切分：

```text
train：signal_date <= 2018-12-31 且 D+2 <= 2018-12-31
validation：2019 signal，且 D+2 <= 2019-12-31
prediction：2020 起，且 D+2 在数据范围内
```

因此只排除自然缺少 D+2 标签的年度边界样本，不沿用 H30 的额外截断。

## 2. Good/Bad 标签

### 2.1 基础价格

对每个 `(instrument, signal_date=D)`，从现有 raw OHLCV memmap 读取：

```text
O1 = open(D+1)
C1 = close(D+1)
O2 = open(D+2)
H2 = high(D+2)
C2 = close(D+2)
```

派生：

```text
d1_intraday_return = C1 / O1 - 1
d2_open_gap        = O2 / C1 - 1
d2_close_return    = C2 / C1 - 1
entry_to_exit      = C2 / O1 - 1
```

其中 `entry_to_exit` 是最终 validation Top-K 和回测方向对应的真实收益标签。

### 2.2 涨停触及

`D+2` 盘中触及涨停使用 `H2 / C1 - 1` 判断，并按当时制度设置阈值：

```text
ST：4.5%
沪深主板：9.5%
创业板：
  2020-08-24 以前 9.5%
  2020-08-24 起 19.5%
科创板：19.5%
```

使用略低于名义涨停幅度的阈值，是为了容纳价格取整和前复权浮点误差。
样本至少已有 60 日历史，因此科创板上市前五日无涨跌停的特殊阶段不会进入
本实验的有效事件判断。

### 2.3 Good 条件

满足以下任一条件即为 good：

```text
event_limit_touch:
  D+2 盘中触及涨停

event_close_surge:
  d2_close_return >= 5%

event_gap_surge:
  d2_open_gap >= 4%

event_continuation:
  d1_intraday_return >= 5%
  AND d2_open_gap >= 1%
```

即：

```text
is_good =
    event_limit_touch
    OR event_close_surge
    OR event_gap_surge
    OR event_continuation
```

其余所有有效全市场样本统一标记为：

```text
is_good = false
is_bad = true
```

阈值不在训练过程中调参。第一轮固定以上合同，避免用 2019 validation
反向搜索标签。

### 2.4 当前数据密度

基于现有 2010-2019 全市场数据的只读统计：

| split | 总行数 | good 比例 | good/日均值 | good/日中位数 |
|---|---:|---:|---:|---:|
| 2010-2018 train | 4,243,374 | 4.93% | 99.4 | 65 |
| 2019 完整 validation | 783,575 | 4.97% | 160.8 | 120.5 |

2019 股票覆盖更多，因此日均数量高于早期训练段。这不改变全市场 bad
定义，也不做正负类别下采样。

标签 artifact 使用按 `row_id` 对齐的紧凑 mmap，避免 1,000 万级样本反复
读取大 Parquet：

```text
event_flags_uint8.mmap:
  is_good
  event_limit_touch
  event_close_surge
  event_gap_surge
  event_continuation

event_values_float32.mmap:
  d1_intraday_return
  d2_open_gap
  d2_close_return
  entry_to_exit_return
  event_strength

manifest.json:
  row_id 对齐合同、shape、字段顺序、阈值、来源和 SHA-256

daily_counts.parquet:
  signal_date / rows / good
```

`event_strength` 只用于 validation 排序诊断，不改变二元 pair 方向：

```text
event_strength = max(
    limit_touch_strength,
    d2_close_return / 0.05,
    d2_open_gap / 0.04,
    min(d1_intraday_return / 0.05, d2_open_gap / 0.01)
)
```

## 3. Pair 构建

### 3.1 核心合同

每个 epoch、每个有 good 的 `signal_date`：

1. 保留当日全部 good；
2. 保留当日全部 bad；
3. 每个 bad 恰好出现一次；
4. 每个 bad 与当日一个 good 配成一对；
5. good 通过确定性的循环排列被复用；
6. 每个 epoch 改变 good 循环起点，扩展组合覆盖；
7. 不跨日期配 Pair；
8. 不按未来收益 gap 删除 bad；
9. 不做 up/range/down 重采样；
10. 不做类别均衡。

因此：

```text
pairs_per_epoch
= 所有含 good 日期上的 bad 总数
```

上述是逻辑 Dataset 的精确覆盖。PyTorch `DistributedSampler` 为保证各 rank
步数一致，最多会额外补齐 `world_size - 1` 个重复 Pair；训练日志必须单独记录
`distributed_sampler_padding_pairs`，不能把补齐行误报成新的 bad 样本。

训练数据定义上，所有不满足事件条件的有效股票都是 bad。若某日没有任何 good，
同日 pairwise loss 无法形成比较，该日所有 bad 仍保留在标签 artifact 和审计统计中，
但不进入该 epoch 的 pair。当前抽样统计中 train 与 2019 validation 均未发现
零-good 日期。

### 3.2 模型与损失保持不变

继续使用：

```text
Frozen daily AE
-> FrozenAERewardTransformer
-> scalar reward score
-> Bradley-Terry loss:
   softplus(-(score_good - score_bad) / temperature)
```

不增加 BCE head，不做回归，不改模型参数量。

Pair batch 返回的 `good_return` / `bad_return` 使用
`entry_to_exit_return = C2 / O1 - 1`，用于日志和审计；训练方向只由
`is_good > is_bad` 决定。

## 4. Validation

每个 epoch 在完整 2019 validation 上执行：

### 4.1 Pair 指标

- validation pair accuracy；
- good/bad score 均值；
- score margin；
- validation 全 bad 覆盖数量。

### 4.2 横截面 Top-K 指标

每日按模型 score 排序，至少记录 Top-1、Top-5、Top-10：

- `entry_to_exit_return` 均值；
- 相对当日全市场均值的超额收益；
- good precision；
- good lift：Top-K good rate / 当日全市场 good rate；
- 与真实 `entry_to_exit_return` Top-5 的 overlap；
- 模型 Top-5 在真实收益排序中的平均 percentile；
- score 与真实收益的日均 RankIC；
- score 与 `event_strength` 的日均 RankIC。

最佳 epoch 的主排序键固定为：

```text
1. 2019 Top-5 entry_to_exit excess return，越大越好
2. 2019 Top-5 good lift，越大越好
3. 2019 daily RankIC，越大越好
4. epoch 较小者优先
```

所有 20 个 epoch 均保存，不默认使用最后一轮。

## 5. 最小代码改动

### 5.1 新增事件标签模块

新增：

```text
tmp/weak-to-strong-diffusion-v1/event_good_bad_v1.py
```

职责：

- 从现有全市场 rows 与 Kronos raw memmap 计算标签；
- 写入 Parquet + JSON manifest；
- 提供标签加载与 schema/hash 校验；
- 提供全 bad 覆盖的 same-date Pair Dataset；
- 不包含模型定义或训练循环。

基础 H2 dataset 使用现有 builder 生成到独立目录：

```text
tmp/weak-to-strong-diffusion-v1/market_all_h2_event_v1/
```

不会覆盖 `market_all_pre2020_v1` 或其他历史数据。

### 5.2 复用现有训练入口

修改：

```text
tmp/weak-to-strong-diffusion-v1/train_wts_ae_reward_transformer_ddp_v1.py
```

仅增加：

- `label_mode=event_good_bad`；
- `--event-labels`；
- 事件 Pair Dataset 分派；
- 事件 validation 指标；
- checkpoint/manifest 中的事件标签合同。

模型类、forward、optimizer、scheduler、DDP、checkpoint 主流程均不改。

### 5.3 复用现有推理入口

检查并仅在必要时兼容：

```text
tmp/weak-to-strong-diffusion-v1/infer_wts_ae_reward_transformer_scores_v1.py
```

输出保持现有 market score artifact 协议：

```text
signal_date
instrument
reward_score_2d
```

### 5.4 集群入口

新增独立文件，避免影响现有任务：

```text
cluster/job_event_good_bad_h2_reward_multi.sh
cluster/job_config_event_good_bad_h2_reward_4x8.py
```

资源：

```text
4 machines × 8 H20
BF16
20 epochs
2010-2018 train
2019 validation every epoch
```

数据与 checkpoint 写入新的 bucket 子目录，不覆盖历史实验。

## 6. 测试与 Review

代码提交训练前必须通过：

1. 标签边界测试：5%、4%、1%、涨停阈值的等号与浮点容差；
2. 创业板 2020-08-24 前后涨停制度测试；
3. D/D+1/D+2 索引与无未来特征泄漏测试；
4. D+1 不可买样本不进入训练测试；
5. good OR 条件和 bad 补集测试；
6. 每个 bad 每 epoch 恰好一次测试；
7. Pair 必须同日且方向恒为 good > bad；
8. epoch 改变 good 配对但不改变 bad 全覆盖测试；
9. validation Top-5 excess、good lift、oracle overlap 与 RankIC 测试；
10. checkpoint 保存全部 20 epoch 测试；
11. 单进程 CPU smoke；
12. 2-process DDP smoke；
13. `compileall`、定向 `pytest`、shell syntax 与 diff 检查。

Review 重点：

- 模型结构是否零改动；
- 是否真正使用全部 bad，而不是随机负采样；
- 是否错误使用 D+1/D+2 字段作为输入；
- 2019 是否每个 epoch 完整评估；
- 是否按固定主指标选择 checkpoint；
- score artifact 是否能直接进入现有 QuantX 回测。

## 7. Review 结论

方案级 Review 已完成，结论为通过，包含以下修正和确认：

1. 放弃直接复用 H30 fold，改为复用基础特征并生成独立 H2 root，避免少用
   2018 年约 30 个交易日。
2. good 标签覆盖用户提出的涨停、大涨、高开和前一日上涨延续四类事件；
   普通 `gap > 0` 不单独成为 good，避免正样本膨胀成普通涨跌分类。
3. 全市场其余有效样本全部进入 bad 池；每个 epoch 每个 bad 恰好使用一次，
   不做负样本随机下采样。
4. Pair 仍限制在同一交易日，避免模型通过市场日期状态完成容易比较。
5. 训练 loss 与模型结构保持不变，事件强度仅用于 validation 诊断。
6. 最佳 epoch 的选择规则在训练前固定，不根据 2020-2026 回测反选。
7. 基础数据、标签、checkpoint、score 和回测均写入独立目录，不覆盖历史实验。
8. 2019 validation 使用完整可标注年度，而不是原 H30 实验为 calibration
   预留后两个月的 `validation_select` 子段。

## 8. AIDI 提交

提交前按顺序执行：

1. 本地生成并验证 event label artifact；
2. 将只读数据与 AE checkpoint 准备到独立 bucket 路径；
3. AIDI dry-run，核对任务名、4×8 H20、镜像、挂载、启动命令和输出目录；
4. 查询本人任务与队列资源；
5. 正常有资源则直接提交；
6. 仅当资源不足时，精确定位本人名下命令为 sleep、资源为 4×8 H20 的任务；
7. 展示其 Job ID、任务名、资源与状态；
8. 只停止该精确 Job ID，并确认进入终态；
9. 不操作任何其他任务；
10. 提交正式训练并跟踪到至少 epoch-1 validation 正常落盘。

## 9. 最佳 Epoch 推理与回测

训练成功后：

1. 汇总 20 个 epoch 的 2019 validation；
2. 按第 4.2 节固定规则选择最佳 epoch；
3. 使用该 checkpoint 对 2020-2026 prediction split 导出全市场 score；
4. 复用现有 Reward score QuantX 接口；
5. 每日 Top-5；
6. D+1 开盘买入；
7. 每层使用当时权益的 10%；
8. 最多 10 只；
9. D+2 收盘卖出；
10. 一字涨停无法卖出则顺延；
11. 最长持有 10 个交易日后按收盘强制退出；
12. 第一份主结果不含手续费，以对齐用户给出的参考口径；
13. 同时补充含现有 QuantX 默认成本的敏感性结果。

最终报告至少包含：

- 数据行数、good/bad 数量、逐日分布；
- 每 epoch train/validation 指标；
- 最佳 epoch 与选择依据；
- 2019 Top-5 超额、good lift、真实排序 gap；
- 2020-2026 总收益、最大回撤、胜率、交易数、月度收益；
- 无手续费与含成本对照；
- 可成交拒单、涨停顺延和超期退出统计；
- 明确区分已测结果、基础设施失败和未证实假设。

## 10. 完成标准

只有以下全部完成才给最终结论：

- 事件标签与全 bad Pair 合同通过测试；
- AIDI 4×8 H20 正式训练完成约 20 epochs；
- 20 个 checkpoint 和 2019 validation 指标完整；
- 最佳 epoch 按预先固定规则选择；
- 最佳 checkpoint 的 2020-2026 score artifact 导出完成；
- QuantX 回测完成且结果 artifact 可审计；
- 最终结论说明该数据构建是否让现有模型学出可用的 H2 Top-5 Alpha。

## 11. 实施与测试结果

### 11.1 数据

正式 H2 root：

```text
tmp/weak-to-strong-diffusion-v1/market_all_h2_event_v1
```

数据统计：

```text
全量：11,949,548
train：4,243,374
  good：209,355
  bad：4,034,019
2019 validation：783,575
  good：38,918
  bad：744,657
prediction：6,909,618
```

事件标签：

```text
全量 good：671,746
good rate：5.62%
零 good 日期：0
```

label mmap、split、H2 target 和 scaler 均完成校验。

### 11.2 本地与 AIDI testing

本地：

- 相关 Reward 定向测试：25 passed；
- QuantX 成交与策略定向测试：65 passed；
- 单 GPU 正式模型 smoke：通过；
- 2 GPU DDP smoke：通过；
- 8 GPU 全市场 score export smoke：通过；
- Ruff、compileall、shell syntax、diff check：通过。

AIDI testing：

```text
有效测试 Job：acloud-273d6137c723
资源：2×8 H20
状态：Succeeded
```

无效测试边界：

- `acloud-0c949e6dd986`：代码包错误地排除了 Reward 源码；
- `acloud-20009a36a03d`：代码包错误地排除了 Kronos memmap 源码；
- `acloud-0a04e9c92ac3`：AIDI PyArrow 无法读取本地新版 Parquet；
- `acloud-5e1e9c944ae8`：fold 已切到 pickle，但旧 loader 仍无条件调用
  `read_parquet()`。

上述任务均在正式训练前失败，不计入模型结果。修复后 testing 完整通过。

为释放资源，只停止了用户精确指定且核验命令为
`exec sleep 1209600` 的任务：

```text
acloud-6a66b2787d4e
(H20*8)*4
Stopped
```

## 12. 正式训练结果

正式 Job：

```text
job_id：acloud-53fd3059a298
资源：(H20*8)*4
状态：Succeeded
开始：2026-09-27 23:26:00
结束：2026-09-28 00:08:44
```

训练合同：

```text
20 epochs
每 epoch 逻辑 Pair：4,034,019
DDP 补齐 Pair：29
完整 2019 validation：每 epoch
20 个 checkpoint：全部落盘
```

日志不存在 non-finite、NaN、Traceback 或 DDP hang。

按预先固定的选择顺序，最佳 checkpoint 为 epoch 18：

```text
2019 Top-5 平均 entry-to-exit return：0.5594%
2019 全市场平均 entry-to-exit return：0.2736%
2019 Top-5 超额：+0.2858 pct / 笔
Top-5 good precision：22.31%
全市场 good rate：4.98%
good lift：4.48×
validation Pair accuracy：68.48%
收益 RankIC：-0.0305
真实收益 Top-5 overlap：2.15%
```

这表明模型学习到了 event membership，但没有学习成全市场单调收益排序。

## 13. 2020-2026 严格样本外分数

score artifact：

```text
event_good_bad_epoch018_20200102_20260708.parquet
rows：6,909,618
dates：1,577
instruments：5,173
score column：reward_score_2d
```

完整横截面 OOS 统计：

```text
Top-5 平均 D+1 open → D+2 close return：0.4943%
全市场平均：0.1781%
Top-5 超额：+0.3162 pct / 笔
Top-5 good precision：19.49%
全市场 good rate：6.15%
good lift：3.17×
收益 RankIC：-0.0357
```

逐年 Top-5 超额：

```text
2020：+0.3276 pct / 笔
2021：+0.5158 pct / 笔
2022：-0.0491 pct / 笔
2023：+0.1693 pct / 笔
2024：+0.6008 pct / 笔
2025：-0.0909 pct / 笔
2026：+1.1584 pct / 笔
```

7 个年度区间中 5 个为正，2022 与 2025 为负。

## 14. 正式 QuantX 回测

成交审计：

- score `D` 通过 `lag=1` 在 `D+1` 执行；
- BUY 使用 open；
- SELL 使用 close；
- 抽查 40 笔成交与 Qlib 对应字段误差为 0；
- 每只新仓目标为当时权益 10%；
- 最多 10 只；
- 正常在买入后的下一交易日收盘退出；
- 一字涨停延期；
- `holding_days >= 10` 提交强制退出；
- 停牌或跌停仍可能阻止成交，因此少量实际持有超过 10 日。

### 14.1 零成本主结果

```text
run_id：20260928_003641_event_good_bad_epoch018_top5_h2_no_cost
区间：2020-01-02 至 2026-07-10
总收益：+1095.44%
年化：46.28%
最大回撤：39.15%
Sharpe：1.199
Sortino：2.029
Calmar：1.182
胜率：47.02%
Profit factor：1.154
平均单笔收益：0.4352%
平均持有：1.53 交易状态日
完成交易：6,891
```

年度收益：

```text
2020：+52.97%
2021：+125.18%
2022：-19.05%
2023：+13.60%
2024：+81.53%
2025：+31.48%
2026：+58.11%
```

78 个月中 46 个月为正；最好月份为 2024-02（+42.73%），最弱月份为
2022-01（-20.22%）。

### 14.2 标准成本敏感性

成本合同：

```text
commission：0.05%
stamp tax：0.01%
slippage：0.10% / 边
```

结果：

```text
run_id：20260928_003930_event_good_bad_epoch018_top5_h2_with_cost
总收益：+40.73%
年化：5.38%
最大回撤：66.45%
Sharpe：0.139
胜率：45.48%
Profit factor：1.018
平均单笔收益：0.1243%
总成本：86,121,229
```

年度收益：

```text
2020：+10.90%
2021：+62.80%
2022：-41.95%
2023：-17.83%
2024：+29.80%
2025：-6.26%
2026：+34.30%
```

78 个月中 40 个月为正。

执行统计：

```text
BUY：6,892
SELL：6,891
拒单：124
  limit_down：79
  suspended：31
  price_jump：13
  limit_up：1
常规 H2 close exit：6,890
10 日强制退出成交：1
持有超过 10 日后才成交：19
最长实际持有：24
```

## 15. 最终结论

本实验支持以下结论：

1. 事件化数据构建有效。保持模型结构不变，仅把标签改成强 H2 event
   good 与全市场其余 bad，模型在 2019 和 2020-2026 均能显著富集 good。
2. 这是“事件分类 Alpha”，不是全横截面收益排序 Alpha。good lift 达到
   `3.17×` OOS，但收益 RankIC 为负，真实收益 Top-5 overlap 很低。
3. 零成本下有很强的可复利毛收益，说明信号不是完全随机。
4. 当前版本尚不可直接部署。持有期极短、换手极高，标准成本后年化只剩
   `5.38%`，最大回撤扩大到 `66.45%`，风险收益比不合格。
5. 因此，“这种数据能不能让模型筛出 D+2 强事件股票”的答案是可以；
   “当前 Top-5 H2 策略是否已形成稳健可交易强 Alpha”的答案是否定的。

若继续，优先级不是改模型，而是继续改数据目标或持有/换手合同：

1. 将 good 从 OR 事件改成“事件发生且 D+1 open → D+2 close 净收益超过成本”
   的可交易事件；
2. 给 gap-up 后回落样本单独标 bad，避免模型只预测高开/触板而不预测收盘收益；
3. 用同一 score 测 Top10/Top20 或延长持有，降低单位 Alpha 所需换手；
4. 任何后续版本仍应保持 2018 前训练、2019 选 epoch、2020 后一次性回测。

## 16. 退出策略优化

### 16.1 固定项

本轮冻结以下内容：

- epoch 18 Reward checkpoint；
- 每日 Top-5；
- D+1 open 买入；
- 每只新仓为当时权益 10%；
- 最多同时持有 10 只；
- 模型、数据和 score artifact 均不重训。

只比较退出规则。

### 16.2 新增状态

QuantX sell rule 增加板块、ST 和日期感知的：

```text
limit_up
one_side_limit_up
```

买入涨停校验也改为按实际 `buy_deal_price=open` 判断，避免“开盘可买但收盘
涨停”被错误拒单。

相关定向测试：

```text
66 passed
```

### 16.3 实验矩阵

固定 H2、仅涨停延续、普通强势延续，以及三种止盈止损/移动止盈组合，共
6 个版本；每个版本同时运行标准成本与零成本，共 12 次全市场回测。

标准成本全区间结果：

| 版本 | 累计收益 | 年化 | 最大回撤 | Sharpe | 交易数 |
|---|---:|---:|---:|---:|---:|
| fixed_h2 | +20.49% | 2.90% | -67.56% | 0.076 | 13,783 |
| **limit_hold** | **+128.68%** | **13.52%** | **-62.77%** | **0.345** | 13,697 |
| strong_hold | +54.64% | 6.91% | -66.66% | 0.162 | 13,327 |
| strong_hold_conservative | +57.72% | 7.24% | -68.89% | 0.173 | 13,447 |
| strong_hold_wide | +64.06% | 7.88% | -67.82% | 0.187 | 13,381 |
| strong_hold_trailing_only | +57.93% | 7.26% | -66.01% | 0.170 | 13,325 |

零成本结果：

| 版本 | 累计收益 | 年化 | 最大回撤 | Sharpe |
|---|---:|---:|---:|---:|
| fixed_h2 | +924.31% | 42.86% | -39.86% | 1.115 |
| **limit_hold** | **+1813.57%** | **57.22%** | **-38.29%** | **1.460** |
| strong_hold | +1091.02% | 46.20% | -39.39% | 1.084 |
| strong_hold_conservative | +1137.69% | 47.06% | -39.97% | 1.129 |
| strong_hold_wide | +1145.40% | 47.20% | -41.55% | 1.121 |
| strong_hold_trailing_only | +1115.93% | 46.66% | -39.39% | 1.096 |

### 16.4 时间切分选择

只使用标准成本：

```text
2020-2023：开发
2024：规则选择
2025-2026：冻结后评估
```

`limit_hold`：

```text
2020-2023：+12.68%，最大回撤 -54.34%
2024：+41.63%，最大回撤 -32.66%，Calmar 1.27
```

它是唯一在开发段保持正累计收益，同时在 2024 拥有最高 Calmar 的版本，因此
冻结为最终规则。

冻结后的 2025-2026：

```text
标准成本：+43.29%，最大回撤 -22.99%
零成本：+135.84%，最大回撤 -20.68%

2025 标准成本：+4.25%
2026 标准成本：+37.45%
```

### 16.5 机制结论

收益改善并不主要来自降低频率：

```text
fixed_h2 trades：13,783
limit_hold trades：13,697
下降：仅 0.62%

平均持有：
1.53 -> 1.59
```

真正改善来自保留右尾：

```text
收益 >= 20% 的交易：143 -> 171
收益 >= 30% 的交易：24 -> 54
最大单笔收益：46.70% -> 82.09%
```

更宽泛的“普通强势收盘继续持有”会同时延长大量错误持仓，在 2020-2023
开发段全部为负；固定止盈、止损和移动止盈没有修复该问题。

本轮推荐规则因此保持极简：

```text
D+2 收盘涨停：继续持有
之后仍收盘涨停：继续持有
首次不再涨停：收盘卖出
holding_days >= 10：提交强制卖出
```

这验证了用户的判断：网站策略中最有效的继续持有条件很可能就是“涨停状态
延续”，而不是宽泛的上涨或 K 线强势。

## 17. 主板股票池复跑

用户检查 Web 明细后确认正式股票池应使用：

```text
data.universe = all_mainboard
```

该口径同时排除：

```text
SH688 科创板
SZ300 / SZ301 创业板
```

主板股票数为 3,122；三个正式 run 的实际成交中，上述代码前缀违规数量均为 0。

`SH688072` 的 11 日持仓不是跨多轮交易累计：

```text
2026-06-24：成功清仓
2026-06-26：重新买入
2026-06-29 至 2026-07-10：卖单因 suspended 全部拒绝
```

本地 Qlib 在该期间价格固定为 832、volume 为 NaN，因此 position 没有真正清仓。
用户确认不修改该停牌持仓逻辑。已完成交易的 `closed_positions.holding_days`
仍使用自然日差；本轮同样不修改。

主板复跑结果：

| 口径 | 累计收益 | 年化 | 最大回撤 | Sharpe | 成交数 |
|---|---:|---:|---:|---:|---:|
| 零成本 | +620.51% | 35.35% | -34.04% | 1.008 | 13,566 |
| 万0.85佣金 + 卖出万5印花税，零滑点 | +357.26% | 26.24% | -41.86% | 0.748 | 13,566 |
| 同费税 + 双边10bp滑点 | +17.58% | 2.51% | -63.93% | 0.072 | 13,566 |

对应 run：

```text
20260928_122140_event_good_bad_epoch018_limit_hold_mainboard_no_cost
20260928_122036_event_good_bad_epoch018_limit_hold_mainboard_fee085_stamp5_zero_slippage
20260928_121931_event_good_bad_epoch018_limit_hold_mainboard_fee085_stamp5_slippage10bp
```

结论：此前的高收益明显依赖科创板/创业板右尾。切换到沪深主板后，零成本和
费税零滑点仍然有明显毛 Alpha；加入每边 10bp 滑点后，累计收益仅剩 17.58%，
说明主板版本仍然高度依赖实际成交冲击。
