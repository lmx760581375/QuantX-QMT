# Kronos 交易区间买点信号 Alpha 替代研究

> 实验日期：2026-07-16
>
> 状态：`screening_only`
>
> 研究目标：验证基于 memmap 动态窗口的 Kronos/Transformer 架构，在重新定义交易语义后，是否有潜力替代传统 alpha 因子挖掘流程。
>
> 当前结论：旧无状态三分类标签存在动作语义混淆，暂停继续扩大旧任务训练；新方向切换为“oracle 可赚钱交易区间 -> 买点二分类”。

## 1. 背景

旧 `SELL/HOLD/BUY` 三分类任务把 long-only 交易动作压成无状态分类：任意日期都要预测 `SELL`、`HOLD` 或 `BUY`。这会混淆两个不同问题：

1. 空仓时：今天是否值得买。
2. 持仓时：今天是否应该卖。

其中 `SELL` 的语义尤其不稳定：它既可能表示未来下跌风险，也可能表示一段上涨后应该止盈。这个标签空间不符合真实交易状态机，因此即使增加训练年份、扩大模型、提升 batch 或改 loss，也可能只是在拟合一个语义混合任务。

本轮把问题改成第一阶段买点检测：先通过每只股票的历史价格路径生成可赚钱、非重叠交易区间，再只训练“空仓状态下是否买入”。

## 2. 固定边界

| 项目 | 值 |
| --- | --- |
| 实验目录 | `tmp/kronos-classification-signal-v1/` |
| 输入数据 | 既有 raw/derived/market memmap，不重复落 60 日窗口 |
| 标签周期 | `max_hold_days=5` |
| 训练窗口 | 动态 60 session |
| 执行语义 | T 日信号，T+1 open 买入，之后最多 5 个交易日内 open 卖出 |
| 调参禁区 | 不使用 `forward_2026` |
| 第一阶段任务 | `FLAT_NO_TRADE` vs `BUY_ENTRY` 二分类 |

新增文件：

| 文件 | 用途 |
| --- | --- |
| `build_kronos_trade_interval_labels_v1.py` | 按 instrument 生成交易区间 oracle 标签，ParquetWriter 流式写索引 |
| `build_kronos_trade_interval_buy_folds_v1.py` | 从交易区间标签生成买点二分类 fold |
| `train_kronos_transformer_classifier_v1.py` | 扩展支持 `--num-classes 2`、`--class-names flat,buy`、`binary_balanced` sampler |
| `evaluate_kronos_classifier_v1.py` | 扩展二分类 precision/recall/F1、TopK buy hit 和 `net_return` 评测 |

标签定义：

| id | 名称 | 含义 |
| ---: | --- | --- |
| 0 | `FLAT_NO_TRADE` | 空仓且不应买入 |
| 1 | `BUY_ENTRY` | oracle 交易区间入口信号日 |
| 2 | `POSITION_HOLD` | oracle 持仓中间段 |
| 3 | `SELL_EXIT` | oracle 卖出信号日 |
| -1 | `INVALID` | 不可训练或不可执行 |

第一阶段训练只保留 `FLAT_NO_TRADE` 和 `BUY_ENTRY`，并映射为二分类 `flat=0, buy=1`。

## 3. R10：旧三分类大模型诊断

### 假设

如果旧三分类标签语义是合理的，那么扩大训练范围到 2012-2019、保留全部 BUY/SELL、采样 HOLD、使用 20M 参数模型，应该显著提升 BUY/SELL precision/recall/F1。

### 实验结果

旧任务已完成 6 epoch 8 卡训练：

```text
runs/K02-large20m-fixed2012_2019-balanced-with_market-ce-seed7-ddp8-b1536-progress-20260716-183917
```

关键设置：

| 项目 | 值 |
| --- | --- |
| 模型 | `large20m`，约 21.56M 参数 |
| train | 2012-2019，4,009,315 行 |
| validation_select | 2020-01-02 至 2020-09-11，572,816 行 |
| batch | 1536/GPU，8 卡 DDP |
| loss | CE |

验证集趋势：

| epoch | buy/sell avg F1 | macro F1 |
| ---: | ---: | ---: |
| 1 | `0.3450` | - |
| 2 | `0.3823` | - |
| 3 | `0.3779` | - |
| 4 | `0.3887` | - |
| 5 | `0.4005` | - |
| 6 | `0.4008` | `0.4346` |

续训到 20 epoch 曾启动，但在重新审视 label 语义后停止，避免继续消耗 GPU。

### 反事实分析

1. F1 随训练增加有改善，说明模型和数据链路能学习到部分模式。
2. 但旧 `SELL` 同时承载“风险下跌”和“上涨后止盈”，从交易状态机看不是同一个动作。
3. 如果博主类似方案能得到很高 precision，更可能是其正样本来自稀疏且高质量的可交易入口，而不是对每个交易日做无状态三分类。
4. 继续扩大旧任务模型或 epoch，无法修复动作语义混淆。

### 更新假设和结论

`rejected_for_label_semantics`。旧三分类可作为工程与模型容量验证，但不能作为替代 alpha 因子的主研究任务。下一轮改为交易区间买点二分类。

## 4. R11：交易区间买点标签 smoke

### 假设

如果 oracle 交易区间标签能表达“可赚钱买点”，那么在小样本真实数据上应满足：

1. 能生成非重叠交易区间。
2. `BUY_ENTRY` 明显稀疏。
3. 二分类 fold 能只保留 `FLAT_NO_TRADE/BUY_ENTRY`。
4. 训练、校准、评测链路无需重复物化窗口数据即可跑通。

### 实验结果

隔离 root：

```text
/tmp/kronos_trade_interval_smoke_WANeKS
```

输入通过软链指向正式 memmap，输出落在临时目录，不污染正式数据。

20 只股票标签 smoke：

| 指标 | 值 |
| --- | ---: |
| sample rows | `73,264` |
| trade intervals | `3,785` |
| FLAT_NO_TRADE | `56,526` |
| BUY_ENTRY | `3,785` |
| POSITION_HOLD | `9,168` |
| SELL_EXIT | `3,785` |
| BUY_ENTRY 平均 net return | `+0.068916` |

二分类 fold smoke：

| split | rows |
| --- | ---: |
| train | `3,966` |
| validation | `3,964` |
| validation_select | `2,832` |
| validation_calibration | `1,037` |
| prediction | `20,081` |

训练 smoke：

```text
/tmp/kronos_trade_interval_smoke_WANeKS/runs/smoke-trade-interval-buy-debug-20260716-201012
```

| 项目 | 值 |
| --- | --- |
| model | `debug` |
| classes | `flat,buy` |
| train rows | 512 capped |
| validation rows | 256 capped |
| device | CPU |
| epoch | 1 |

validation smoke：

| 指标 | 值 |
| --- | ---: |
| accuracy | `0.8164` |
| buy precision | `0.1220` |
| buy recall | `0.3125` |
| buy F1 | `0.1754` |
| flat F1 | `0.8967` |

评测 smoke：

| 指标 | 值 |
| --- | ---: |
| validation_select rows | `2,832` |
| buy precision | `0.1196` |
| buy recall | `0.3952` |
| buy F1 | `0.1836` |
| top5 buy hit rate | `0.1111` |
| top5 net return | `+0.062526` |
| rank IC mean | `-0.02434` |

验证命令：

```text
${HOME}/anaconda3/envs/test/bin/python -m pytest -q tests/test_kronos_memmap_v1.py
20 passed
```

### 反事实分析

1. smoke 指标不能说明 alpha 有效，因为只用了 20 只股票和极小训练预算。
2. 但 smoke 证明新标签/fold/训练/评测链路是闭环的，且不会重复落训练窗口。
3. 小样本 TopK `net_return` 为正但 RankIC 为负，提示买点任务应优先看 precision、daily TopK 命中率和真实可执行收益，不应只看 RankIC。
4. BUY_ENTRY 占比约 5%，说明这是稀疏事件检测问题；训练集需要采样平衡，但验证和预测必须保持自然分布。

### 更新假设和结论

`retained_for_full_label_build`。下一步构建全量 2010-2026 交易区间标签，生成 2012-2019 训练、2020 验证、2021-2025 预测的二分类 fold，然后启动 6 卡 `large20m` 训练。

## 5. R12：全量训练计划

### 假设

使用 2012-2019 的全量 `BUY_ENTRY`，并按年采样 `FLAT_NO_TRADE` 到 1:1，能让 20M 参数 Kronos 模型学习到买点入口特征。若该假设成立，2020 validation_select 和 2021-2025 prediction 上应出现稳定的 buy precision / TopK buy hit / TopK net return 改善。

### 预注册命令

全量标签：

```bash
${HOME}/anaconda3/envs/test/bin/python build_kronos_trade_interval_labels_v1.py \
  --root . \
  --max-hold-days 5 \
  --min-profit 0.04 \
  --round-trip-cost 0.003 \
  --max-drawdown -0.06
```

二分类 fold：

```bash
${HOME}/anaconda3/envs/test/bin/python build_kronos_trade_interval_buy_folds_v1.py \
  --root . \
  --fold-id trade_interval_buy_2012_2019_v1 \
  --train-start 2012-01-01 \
  --train-end 2019-12-31 \
  --validation-start 2020-01-01 \
  --validation-end 2020-12-31 \
  --prediction-start 2021-01-01 \
  --prediction-end 2025-12-31 \
  --negative-multiplier 1.0 \
  --seed 7
```

6 卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 train_kronos_transformer_classifier_v1.py \
  --root . \
  --run-name K03-large20m-trade-interval-buy-2012_2019-ddp6 \
  --fold-id trade_interval_buy_2012_2019_v1 \
  --model-size large20m \
  --num-classes 2 \
  --class-names flat,buy \
  --epochs 20 \
  --batch-size 1536 \
  --sampler-mode binary_balanced \
  --positive-label 1 \
  --negative-label 0 \
  --negative-multiplier 1.0 \
  --loss ce \
  --device cuda \
  --log-every-steps 25
```

首轮评测指标：

1. buy precision / recall / F1。
2. daily TopK buy hit rate。
3. TopK `net_return`。
4. 分年份、board、asset bucket 的 precision/recall/F1。
5. 信号覆盖率和日均候选数。
6. 与旧三分类 K02 的结论隔离：K03 只验证买点入口，不再解释为完整买卖状态机。

### 实验结果

全量标签构建完成：

```text
runs/K03_launch_logs/K03_trade_interval_label_build.log
```

| 项目 | 值 |
| --- | ---: |
| 全量原始 rows | `12,021,652` |
| sample index rows | `12,021,652` |
| trade intervals | `1,032,838` |
| FLAT_NO_TRADE | `7,550,974` |
| BUY_ENTRY | `1,032,838` |
| POSITION_HOLD | `2,405,002` |
| SELL_EXIT | `1,032,838` |
| INVALID | `8,769,701` |
| BUY_ENTRY 平均 net return | `+0.076488` |

正式 fold：

```text
data/folds/trade_interval_buy_2012_2019_v1.json
```

| split | rows |
| --- | ---: |
| train | `765,704` |
| validation | `563,728` |
| validation_select | `381,355` |
| validation_calibration | `170,084` |
| prediction | `3,963,120` |

训练集采样前后分布：

| 阶段 | FLAT_NO_TRADE | BUY_ENTRY |
| --- | ---: | ---: |
| before | `2,766,981` | `382,852` |
| after | `382,852` | `382,852` |

正式训练：

```text
runs/K03-large20m-trade-interval-buy-2012_2019-ddp6-b1536-20260716-201630
```

| 项目 | 值 |
| --- | --- |
| 模型 | `large20m`，二分类 `flat,buy` |
| 训练区间 | 2012-2019 |
| 验证区间 | 2020 |
| 预测区间 | 2021-2025 |
| epoch | `20` |
| DDP | 6 卡 |
| batch | `1536/GPU` |
| sampler | `binary_balanced` |
| loss | CE |
| checkpoint 选择 | 按用户要求使用 `final_model.pt`，不使用 best checkpoint |
| peak allocated | `29.05 GB/GPU` |
| peak reserved | `30.17 GB/GPU` |
| 吞吐 | 约 `16.3k samples/s` |
| 每 epoch steps | `84` |

训练过程显示明显过拟合：best validation buy F1 出现在 epoch 3，而 final epoch 20 的 train accuracy 已接近 99.5%。

| checkpoint | validation buy precision | validation buy recall | validation buy F1 | validation loss |
| --- | ---: | ---: | ---: | ---: |
| best epoch 3 | - | - | `0.3716` | - |
| final epoch 20 | `0.2161` | `0.5803` | `0.3149` | `3.1648` |

final checkpoint 评测：

| split | rows | buy precision | buy recall | buy F1 | flat F1 | macro F1 | Top20 buy hit | Top20 5d open-to-close | rank IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 validation_select | `381,355` | `0.2161` | `0.5803` | `0.3149` | `0.7236` | `0.5192` | `0.2844` | `+0.000154` | `0.0141` |
| 2021-2025 prediction | `3,963,120` | `0.1699` | `0.5311` | `0.2575` | `0.7677` | `0.5126` | `0.2146` | `-0.002836` | `0.0106` |

prediction 分年分类指标：

| 年份 | rows | 自然 buy 占比 | 预测 buy 占比 | buy precision | buy recall | buy F1 | macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | `662,216` | `0.1264` | `0.3666` | `0.1848` | `0.5362` | `0.2749` | `0.5188` |
| 2022 | `752,223` | `0.1218` | `0.4011` | `0.1669` | `0.5497` | `0.2561` | `0.4964` |
| 2023 | `888,794` | `0.0816` | `0.3239` | `0.1215` | `0.4824` | `0.1941` | `0.4946` |
| 2024 | `814,317` | `0.1400` | `0.3940` | `0.2084` | `0.5866` | `0.3076` | `0.5277` |
| 2025 | `845,570` | `0.1136` | `0.3286` | `0.1659` | `0.4797` | `0.2465` | `0.5163` |

prediction 分年 daily Top20：

| 年份 | Top20 buy hit | Top20 5d open-to-close | Top-bottom spread | 日均预测 buy 数 | 日均真实 buy 数 | 正收益 Top20 天数占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | `0.2360` | `-0.000428` | `0.01419` | `999` | `344` | `0.5185` |
| 2022 | `0.2140` | `-0.005331` | `0.01178` | `1,247` | `379` | `0.4587` |
| 2023 | `0.1758` | `-0.005914` | `0.00615` | `1,189` | `300` | `0.4298` |
| 2024 | `0.2415` | `-0.003389` | `0.01028` | `1,326` | `471` | `0.4174` |
| 2025 | `0.2053` | `+0.000935` | `0.00994` | `1,167` | `404` | `0.5462` |

prediction 分板块：

| board | rows | 自然 buy 占比 | 预测 buy 占比 | buy precision | buy recall | buy F1 | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chinext | `960,552` | `0.1376` | `0.4061` | `0.1910` | `0.5635` | `0.2853` | `0.5092` |
| mainboard_sh | `1,419,781` | `0.0986` | `0.3251` | `0.1520` | `0.5014` | `0.2333` | `0.5136` |
| mainboard_sz | `1,210,169` | `0.1086` | `0.3455` | `0.1617` | `0.5146` | `0.2461` | `0.5123` |
| star | `372,618` | `0.1458` | `0.4327` | `0.1916` | `0.5686` | `0.2867` | `0.4982` |

市值桶之间差异很小，buy F1 基本在 `0.255-0.259`，说明本轮主要问题不是单纯市值分层失效，而是整体阈值/标签分布/排序质量问题。

### 反事实分析

1. 2020 validation_select 到 2021-2025 prediction 的 buy F1 从 `0.3149` 降到 `0.2575`，Top20 5 日收益从接近 0 转为 `-0.002836`。这支持“跨年份分布差异会伤害泛化”的假设。
2. 但不是所有年份都同向失败：2025 Top20 5 日收益为正，2022/2023 明显为负。模型可能捕捉到一部分横截面排序信息，因为每年 Top-bottom spread 都为正，但 Top20 绝对收益不足以覆盖直接交易。
3. 最大的工程/建模问题是预测 buy 覆盖率过高：prediction 自然 buy 占比只有 `11.55%`，模型预测 buy 占比却是 `36.11%`。日均预测 buy 数约 `1,167`，真实 buy 数约 `379`。因此 recall 高但 precision 被大量 false positive 稀释。
4. 训练集 1:1 采样改善了学习稳定性，但改变了先验分布；temperature calibration 改善概率校准，却没有解决默认 argmax/0.5 阈值下过度触发 buy 的问题。
5. final epoch 20 明显过拟合，best epoch 3 的 validation buy F1 更高。按 final checkpoint 得到的结论更偏保守：当前训练预算和正则设置不能证明 20M 模型越训越好。
6. 如果只看 rank IC，会低估或误判该任务；本轮 rank IC 约 `0.0106` 很弱，但 Top-bottom spread 为正，说明排序尾部对比存在信号。然而 TopK 真实收益为负，表示排序信号尚未转化为可交易 alpha。

### 更新假设和结论

`diagnostic_only_not_tradable_yet`。交易区间买点二分类比旧三分类语义更合理，且模型能学到部分排序/召回信号；但当前 K03 final checkpoint 不能替代传统 alpha 因子：precision 不够、自然分布下预测过宽、2021-2025 Top20 绝对收益为负、20 epoch 过拟合明显。

下一轮不应继续只扩大 epoch。优先实验方向：

1. 阈值/TopK 策略从 argmax 改为按日定额或按分位筛选，直接优化 precision@K 和收益，而不是默认分类阈值。
2. 保留自然分布验证，训练端尝试 `negative_multiplier > 1` 或 focal/asymmetric loss，降低 buy 过度触发。
3. 使用 epoch 3 附近 checkpoint 做反事实评测，量化“早停是否显著优于 final”。该结果只作为诊断，不替代本轮 final checkpoint 结论。
4. 分年份训练/验证滚动实验，确认 2022/2023 的失败是行情状态差异、标签 oracle 参数不适配，还是模型过拟合 2012-2019。
5. 在不重复物化窗口的前提下，增加交易可执行评测：每日 TopK 组合、换手、费用、停牌/涨跌停过滤、与传统 alpha 因子组合后的增益。

## 6. R13：final prediction 的阈值与 TopK 反事实

### 假设

R12 显示 final checkpoint 在自然分布下预测 buy 过宽。若主要问题只是默认 `argmax/0.5` 阈值不适合稀疏买点任务，那么不重新训练，仅把同一份 prediction 改成每日固定 TopK 或高分位筛选，应显著提升 precision@K，并把 TopK 5 日收益推到正数。

### 实验结果

输入文件：

```text
runs/K03-large20m-trade-interval-buy-2012_2019-ddp6-b1536-20260716-201630/predictions_prediction.parquet
```

输出诊断：

```text
runs/K03-large20m-trade-interval-buy-2012_2019-ddp6-b1536-20260716-201630/analysis_prediction_threshold_topk.json
```

每日固定 TopK 结果：

| 策略 | 日均数量 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | hit lift | 正收益天数占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top5 | `5` | `0.2210` | `-0.003174` | `+0.002235` | `+0.0958` | `0.4308` |
| Top10 | `10` | `0.2183` | `-0.003246` | `+0.002163` | `+0.0931` | `0.4565` |
| Top20 | `20` | `0.2146` | `-0.002836` | `+0.002573` | `+0.0893` | `0.4739` |
| Top50 | `50` | `0.2084` | `-0.003334` | `+0.002075` | `+0.0832` | `0.4540` |
| Top100 | `100` | `0.2013` | `-0.003371` | `+0.002038` | `+0.0760` | `0.4640` |

每日高分位筛选：

| 策略 | 日均数量 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | hit lift | 正收益天数占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score >= P99 | `33.3` | `0.2129` | `-0.003099` | `+0.002310` | `+0.0876` | `0.4681` |
| score >= P98 | `66.2` | `0.2065` | `-0.003121` | `+0.002288` | `+0.0813` | `0.4640` |
| score >= P95 | `164.6` | `0.1976` | `-0.003191` | `+0.002218` | `+0.0724` | `0.4590` |
| score >= P90 | `328.8` | `0.1882` | `-0.003608` | `+0.001801` | `+0.0629` | `0.4590` |
| score >= P80 | `657.1` | `0.1754` | `-0.004063` | `+0.001346` | `+0.0501` | `0.4490` |

概率阈值筛选：

| prob_buy 阈值 | 日均数量 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | hit lift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.50` | `1185.6` | `0.1659` | `-0.004558` | `+0.000851` | `+0.0406` |
| `0.60` | `700.7` | `0.1807` | `-0.004115` | `+0.001294` | `+0.0555` |
| `0.70` | `310.4` | `0.1968` | `-0.003603` | `+0.001806` | `+0.0716` |
| `0.75` | `161.7` | `0.2049` | `-0.003023` | `+0.002306` | `+0.0795` |
| `0.80` | `57.9` | `0.2154` | `-0.003218` | `+0.001449` | `+0.0874` |

Top-bottom spread：

| K | Top-bottom 5d open-to-close | Top-bottom buy hit |
| ---: | ---: | ---: |
| 5 | `+0.01167` | `+0.1317` |
| 10 | `+0.01098` | `+0.1308` |
| 20 | `+0.01047` | `+0.1280` |
| 50 | `+0.00851` | `+0.1213` |
| 100 | `+0.00742` | `+0.1153` |

分年 Top10：

| 年份 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | 正收益天数占比 |
| ---: | ---: | ---: | ---: | ---: |
| 2021 | `0.2420` | `-0.000027` | `+0.005160` | `0.4979` |
| 2022 | `0.2190` | `-0.005444` | `+0.003930` | `0.4711` |
| 2023 | `0.1851` | `-0.006177` | `-0.000134` | `0.4256` |
| 2024 | `0.2413` | `-0.003744` | `+0.002315` | `0.4256` |
| 2025 | `0.2038` | `-0.000811` | `-0.000513` | `0.4622` |

### 反事实分析

1. TopK/高分位筛选确实提升了命中率：Top5 buy hit `0.2210`，显著高于 prediction 自然 buy 占比约 `0.1252`；Top-bottom buy hit spread 约 `0.12-0.13`。
2. 但绝对收益没有转正：Top5/Top10/Top20 的 5 日 open-to-close 均值仍在 `-0.0032` 附近。说明“阈值过宽”不是唯一问题，score 排序强度还不足以直接作为多头交易信号。
3. 相对全市场有稳定超额：TopK 相对同日全市场平均 5 日收益约 `+0.0020` 到 `+0.0026`。这说明模型并非完全无效，更像是弱排序信号，可能需要叠加市场状态过滤、传统 alpha 或风险中性化。
4. 2023 Top10 相对全市场超额也转负，说明跨年份失效不只是阈值问题；该年份可能是行情状态、标签 oracle 参数或训练分布不匹配的压力测试。
5. 概率阈值越高，buy hit 基本越高，但覆盖快速下降且绝对收益仍负，进一步说明 calibration/阈值调节不能单独解决可交易性。

### 更新假设和结论

`weak_ranker_not_standalone_alpha`。K03 final 模型已经具备弱横截面排序能力，但不能独立替代 alpha 因子。下一轮应把优化目标从“分类准确率/F1”切到“日频 TopK precision@K、相对收益、交易收益”，并优先验证三条路径：

1. 训练分布：提高负样本比例或使用 asymmetric/focal loss，降低 false positive，同时保留自然分布验证。
2. 目标函数：加入 pairwise/listwise ranking 或收益加权 BCE，让 score 直接服务每日排序，而不是只拟合二分类标签。
3. 交易过滤：对 TopK 叠加市场状态、趋势/波动/流动性过滤，验证弱排序信号能否转化为正绝对收益。

## 7. R14：best checkpoint 早停反事实

### 假设

R12 的 20 epoch final checkpoint 明显过拟合。如果主要问题是训练过久导致泛化下降，那么使用 validation buy F1 最优的 epoch 3 checkpoint，应在 2020 validation_select 和 2021-2025 prediction 上同时改善 precision、F1、TopK hit 和真实收益。

### 实验结果

为避免覆盖 final 评测产物，创建隔离诊断目录：

```text
runs/K03-best-epoch3-diagnostic-20260716
```

该目录仅复制 `train_manifest.json` 和 `scaler.json`，不复制 final 的 `calibrator.json`。原因是原 calibrator 是按 `final_model.pt` 拟合的，不应套用到 best checkpoint。二分类下 temperature 不改变 argmax 与 TopK 排序，因此未校准 softmax 足够用于本轮早停反事实。

输出文件：

```text
runs/K03-best-epoch3-diagnostic-20260716/evaluation_validation_select.json
runs/K03-best-epoch3-diagnostic-20260716/evaluation_prediction.json
runs/K03-best-epoch3-diagnostic-20260716/analysis_prediction_best_yearly_threshold_topk.json
```

best vs final 总指标：

| checkpoint | split | buy precision | buy recall | buy F1 | flat F1 | macro F1 | Top20 buy hit | Top20 5d open-to-close | rank IC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final epoch 20 | 2020 validation_select | `0.2161` | `0.5803` | `0.3149` | `0.7236` | `0.5192` | `0.2844` | `+0.000154` | `0.0141` |
| best epoch 3 | 2020 validation_select | `0.2737` | `0.5785` | `0.3716` | `0.7984` | `0.5850` | `0.3424` | `+0.003253` | `-0.0183` |
| final epoch 20 | 2021-2025 prediction | `0.1699` | `0.5311` | `0.2575` | `0.7677` | `0.5126` | `0.2146` | `-0.002836` | `0.0106` |
| best epoch 3 | 2021-2025 prediction | `0.2454` | `0.4378` | `0.3145` | `0.8686` | `0.5916` | `0.3073` | `-0.002441` | `-0.0397` |

best checkpoint prediction 分年分类指标：

| 年份 | 自然 buy 占比 | 预测 buy 占比 | buy precision | buy recall | buy F1 | macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | `0.1264` | `0.2489` | `0.2412` | `0.4752` | `0.3200` | `0.5815` |
| 2022 | `0.1218` | `0.2552` | `0.2371` | `0.4968` | `0.3210` | `0.5816` |
| 2023 | `0.0816` | `0.1591` | `0.1977` | `0.3857` | `0.2614` | `0.5802` |
| 2024 | `0.1400` | `0.2933` | `0.2649` | `0.5551` | `0.3587` | `0.5907` |
| 2025 | `0.1136` | `0.0942` | `0.3001` | `0.2488` | `0.2721` | `0.5938` |

best checkpoint 每日固定 TopK：

| 策略 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | hit lift | 正收益天数占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | `0.3243` | `-0.001035` | `+0.004374` | `+0.1990` | `0.4565` |
| Top10 | `0.3152` | `-0.002059` | `+0.003350` | `+0.1899` | `0.4557` |
| Top20 | `0.3073` | `-0.002441` | `+0.002968` | `+0.1821` | `0.4449` |
| Top50 | `0.2917` | `-0.002934` | `+0.002475` | `+0.1665` | `0.4441` |
| Top100 | `0.2750` | `-0.003355` | `+0.002054` | `+0.1498` | `0.4482` |

best checkpoint Top-bottom spread：

| K | Top-bottom 5d open-to-close | Top-bottom buy hit |
| ---: | ---: | ---: |
| 5 | `+0.00724` | `+0.3034` |
| 10 | `+0.00540` | `+0.2929` |
| 20 | `+0.00407` | `+0.2830` |
| 50 | `+0.00282` | `+0.2645` |
| 100 | `+0.00184` | `+0.2438` |

best checkpoint 分年 Top20：

| 年份 | buy hit | 5d open-to-close | 相对全市场 5d 超额 | 正收益天数占比 |
| ---: | ---: | ---: | ---: | ---: |
| 2021 | `0.3313` | `+0.002037` | `+0.007224` | `0.4897` |
| 2022 | `0.3054` | `-0.004654` | `+0.004720` | `0.4587` |
| 2023 | `0.2725` | `-0.007595` | `-0.001552` | `0.3884` |
| 2024 | `0.3048` | `-0.004871` | `+0.001187` | `0.4091` |
| 2025 | `0.3229` | `+0.002949` | `+0.003247` | `0.4790` |

best checkpoint 概率阈值：

| prob_buy 阈值 | 日均数量 | buy hit | 5d open-to-close | 相对全市场 5d 超额 |
| ---: | ---: | ---: | ---: | ---: |
| `0.50` | `676.7` | `0.2378` | `-0.005034` | `+0.000375` |
| `0.60` | `391.9` | `0.2673` | `-0.004218` | `+0.001191` |
| `0.70` | `191.8` | `0.2938` | `-0.003429` | `+0.001900` |
| `0.75` | `136.4` | `0.3131` | `-0.002452` | `+0.002660` |
| `0.80` | `115.7` | `0.3309` | `+0.000579` | `+0.005001` |

### 反事实分析

1. 早停显著缓解了 final 过拟合。prediction buy precision 从 `0.1699` 提升到 `0.2454`，buy F1 从 `0.2575` 提升到 `0.3145`，预测 buy 占比从 `36.11%` 降到 `20.61%`。
2. OOS TopK hit 大幅提升：Top20 buy hit 从 final 的 `0.2146` 提高到 best 的 `0.3073`；Top5 buy hit 达到 `0.3243`。这说明 epoch 3 checkpoint 的买点识别质量明显更高。
3. 但绝对收益仍未稳定转正。best Top20 2021 和 2025 为正，2022/2023/2024 仍为负，整体 Top20 5 日收益为 `-0.002441`。早停改善了分类和相对超额，但还没有证明可独立交易。
4. best 的 Top-bottom buy hit spread 很强，但 Top-bottom 5 日收益 spread 反而弱于 final。说明 label 命中率和真实收益排序并不完全一致，当前 `BUY_ENTRY` oracle 标签仍可能没有充分对齐交易收益目标。
5. `prob_buy >= 0.80` 在 best checkpoint 上首次给出正的整体 5 日 open-to-close 均值 `+0.000579`，但只覆盖 769 个交易日、日均约 116 只，且未扣真实交易费用、滑点、涨跌停和换手约束。这个结果只能视为候选信号，不足以判定可替代 alpha。
6. 2023 同时出现 Top20 绝对收益和相对全市场超额为负，是关键压力年份。下一轮必须把年份/行情状态作为评测维度，而不是只汇总 2021-2025。

### 更新假设和结论

`early_stopping_required_but_not_sufficient`。K03 的主要失败不只是标签语义或模型无效，训练动态本身很重要：epoch 3 比 epoch 20 更适合 OOS 买点检测。但即使用 best checkpoint，模型仍只是强一些的弱排序器，尚不能独立替代传统 alpha 因子。

下一轮优先级更新：

1. 训练流程必须默认启用早停或按 validation TopK/precision@K 选 checkpoint，不能无条件用最后一个 epoch 做研究结论。
2. 新训练实验应减少 epoch 或加强正则，而不是继续把 20M 模型训满 20 epoch。
3. 训练目标要从二分类 CE 转向收益/排序对齐：收益加权 BCE、focal/asymmetric loss、pairwise/listwise ranking 三类至少做一组对照。
4. 评测必须同时报告分类指标、TopK hit、TopK 绝对收益、相对全市场超额、分年结果，尤其单独看 2023。
5. `prob_buy >= 0.80` 和每日 Top5/Top20 是下一步交易回测候选，但必须加入费用、换手、涨跌停/停牌过滤后再判断能否替代 alpha。

## 8. R15：负样本比例与 focal loss 预注册计划

### 假设

R14 证明早停能显著提高 OOS precision 与 TopK hit，但 TopK 绝对收益仍不稳定。当前最小开发量的下一步不是先改模型结构，而是验证训练分布和 loss 是否能降低 false positive、提高高置信买点质量。

若问题主要来自 1:1 采样导致 buy 先验过高，那么提高 `FLAT_NO_TRADE` 采样比例应提升 precision 和 `prob_buy >= 0.80` 的交易收益。若问题主要来自易样本主导 CE，则 focal loss 应进一步提升 TopK hit 与高置信收益。

### 预注册实验

三组实验均复用现有代码能力，不新增训练逻辑：

| run | negative_multiplier | loss | epochs | early_stop_patience | 目的 |
| --- | ---: | --- | ---: | ---: | --- |
| K04 | `2.0` | `ce` | `10` | `2` | 只验证更高负样本先验是否降低 buy 过度触发 |
| K05 | `2.0` | `focal` | `10` | `2` | 在相同负样本比例下验证 focal 是否提高高质量买点 |
| K06 | `3.0` | `focal` | `10` | `2` | 验证更强负样本先验是否继续提升 precision，或导致 recall 崩塌 |

共同设置：

| 项目 | 值 |
| --- | --- |
| fold | `trade_interval_buy_2012_2019_v1` |
| train | 2012-2019 |
| validation_select | 2020 |
| prediction | 2021-2025 |
| model | `large20m` |
| classes | `flat,buy` |
| sampler | `binary_balanced` |
| positive_label | `1` |
| negative_label | `0` |
| batch | `1536/GPU` |
| DDP | 6 卡 |
| checkpoint 选择 | 同时评估 `best` 与 `final`；研究结论以 `best` 诊断为主，保留 final 对照 |
| 调参禁区 | 不使用 `forward_2026` |

K04 命令模板：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONUNBUFFERED=1 \
${HOME}/anaconda3/envs/test/bin/python -m torch.distributed.run --nproc_per_node=6 \
  train_kronos_transformer_classifier_v1.py \
  --root . \
  --run-name K04-large20m-trade-interval-buy-neg2-ce-ddp6 \
  --fold-id trade_interval_buy_2012_2019_v1 \
  --model-size large20m \
  --num-classes 2 \
  --class-names flat,buy \
  --epochs 10 \
  --batch-size 1536 \
  --num-workers 0 \
  --max-train-rows 0 \
  --max-val-rows 0 \
  --max-calibration-rows 0 \
  --max-scaler-samples 4096 \
  --sampler-mode binary_balanced \
  --positive-label 1 \
  --negative-label 0 \
  --negative-multiplier 2.0 \
  --loss ce \
  --device cuda \
  --early-stop-patience 2 \
  --log-every-steps 25
```

K05/K06 只改：

```text
K05: --run-name K05-large20m-trade-interval-buy-neg2-focal-ddp6 --negative-multiplier 2.0 --loss focal
K06: --run-name K06-large20m-trade-interval-buy-neg3-focal-ddp6 --negative-multiplier 3.0 --loss focal
```

### 评测协议

每个 run 都必须输出：

1. `validation_select` 和 `prediction` 的 `best/final` 分类指标。
2. 每日 Top5/Top10/Top20：buy hit、5 日 open-to-close、相对全市场超额、正收益天数占比。
3. `prob_buy >= 0.75/0.80`：日均数量、buy hit、5 日 open-to-close、相对全市场超额。
4. 分年份结果，必须单独报告 2023。
5. 与 K03 best checkpoint 的同口径对比。

成功判据：

| 层级 | 判据 |
| --- | --- |
| 弱通过 | OOS Top20 buy hit 高于 K03 best 的 `0.3073`，且 Top20 相对全市场超额为正 |
| 中等通过 | OOS Top20 5 日 open-to-close 从 `-0.002441` 提升到接近 0 或转正 |
| 强通过 | OOS Top5/Top20 或 `prob_buy >= 0.80` 在 2021-2025 整体转正，并且 2023 不显著拖累 |
| 失败 | precision 提升但 TopK 绝对收益仍明显为负，或 recall/覆盖崩塌导致候选不可用 |

### 反事实预期

1. 如果 K04 明显优于 K03 best，说明 1:1 训练先验过高是主因，后续可继续优化采样比例。
2. 如果 K05 优于 K04，说明 focal loss 有效，后续可考虑 focal gamma/alpha 或 asymmetric loss。
3. 如果 K06 precision 提升但 TopK 收益不升，说明继续压负样本不能解决收益排序，必须改 label 或目标函数。
4. 如果三组都不能改善 TopK 绝对收益，则当前 `BUY_ENTRY` 标签与真实收益目标仍不够对齐，下一步应优先做收益加权 BCE / pairwise ranking / 交易过滤，而不是继续调分类训练。

### K04 执行结果：negative_multiplier=2.0 + CE

K04 已执行完成：

```text
runs/K04-large20m-trade-interval-buy-neg2-ce-ddp6-20260717-000843
runs/K04-best-diagnostic-20260717
runs/K04-final-diagnostic-20260717
```

训练设置与预注册一致：`large20m`、6 卡 DDP、`negative_multiplier=2.0`、`loss=ce`、`epochs=10`、`early_stop_patience=2`。实际 early stop 于 epoch 5，best checkpoint 为 epoch 3，best validation buy F1 为 `0.3700`。

诊断目录说明：

1. `K04-final-diagnostic` 复制 `train_manifest.json`、`scaler.json`、`calibrator.json`，评估 final checkpoint。
2. `K04-best-diagnostic` 只复制 `train_manifest.json`、`scaler.json`，不复制 final calibrator，避免把 final checkpoint 的 calibration 套到 best checkpoint。

输出文件：

```text
runs/K04-best-diagnostic-20260717/evaluation_validation_select.json
runs/K04-best-diagnostic-20260717/evaluation_prediction.json
runs/K04-best-diagnostic-20260717/analysis_high_precision_slices_k04_best.json
runs/K04-best-diagnostic-20260717/analysis_prediction_best_yearly_threshold_topk.json
runs/K04-final-diagnostic-20260717/evaluation_validation_select.json
runs/K04-final-diagnostic-20260717/evaluation_prediction.json
runs/K04-final-diagnostic-20260717/analysis_high_precision_slices_k04_final.json
runs/K04-final-diagnostic-20260717/analysis_prediction_final_yearly_threshold_topk.json
```

K04 best/final 总指标：

| checkpoint | split | buy precision | buy recall | buy F1 | flat F1 | macro F1 | Top20 buy hit | Top20 5d open-to-close | rank IC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best epoch 3 | 2020 validation_select | `0.2825` | `0.5361` | `0.3700` | `0.8160` | `0.5930` | `0.3506` | `+0.003847` | `-0.0200` |
| final epoch 5 | 2020 validation_select | `0.2552` | `0.5681` | `0.3522` | `0.7821` | `0.5671` | `0.3165` | `+0.000058` | `0.0100` |
| best epoch 3 | 2021-2025 prediction | `0.2551` | `0.3961` | `0.3103` | `0.8807` | `0.5955` | `0.3077` | `-0.002513` | `-0.0388` |
| final epoch 5 | 2021-2025 prediction | `0.2262` | `0.4420` | `0.2992` | `0.8558` | `0.5775` | `0.2880` | `-0.002396` | `-0.0079` |

与 K03 best 对比：

| run | checkpoint | prediction buy precision | prediction buy F1 | Top20 buy hit | Top20 5d open-to-close |
| --- | --- | ---: | ---: | ---: | ---: |
| K03 | best epoch 3 | `0.2454` | `0.3145` | `0.3073` | `-0.002441` |
| K04 | best epoch 3 | `0.2551` | `0.3103` | `0.3077` | `-0.002513` |
| K04 | final epoch 5 | `0.2262` | `0.2992` | `0.2880` | `-0.002396` |

K04 best 每日固定 TopK：

| 策略 | buy hit | 5d open-to-close | 5d open-to-open | 正收益天数占比 |
| --- | ---: | ---: | ---: | ---: |
| Top10 | `0.3198` | `-0.001559` | `-0.003819` | `0.4714` |
| Top20 | `0.3077` | `-0.002513` | `-0.004679` | `0.4565` |
| Top50 | `0.2945` | `-0.002732` | `-0.004534` | `0.4648` |

K04 best 全市场高阈值切片：

| prob_buy 阈值 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | 5d open-to-open | 正收益天数占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.85` | `38273` | `277` | `0.4210` | `0.3976` | `+0.013004` | `+0.013062` | `0.5668` |
| `0.875` | `28621` | `165` | `0.4323` | `0.4056` | `+0.015122` | `+0.015963` | `0.5576` |
| `0.90` | `20562` | `93` | `0.4440` | `0.4027` | `+0.018097` | `+0.013188` | `0.5591` |

K04 best 代表性条件切片：

| 条件 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | hit lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `board=chinext,asset_bucket=2,prob>=0.85` | `2626` | `116` | `0.4543` | `0.4259` | `+0.011360` | `+0.1880` |
| `board=chinext,prob>=0.85` | `13449` | `191` | `0.4397` | `0.3942` | `+0.014055` | `+0.2003` |
| `board=chinext,asset_bucket=1,prob>=0.85` | `2619` | `108` | `0.4303` | `0.4435` | `+0.039483` | `+0.1920` |
| `board=mainboard_sz,asset_bucket=2,prob>=0.85` | `2048` | `107` | `0.4131` | `0.4659` | `+0.029886` | `+0.2152` |

K04 final 高阈值也有正收益，但明显弱于 best：

| prob_buy 阈值 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | 5d open-to-open | 正收益天数占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.85` | `67570` | `765` | `0.3532` | `0.2974` | `-0.000737` | `-0.002944` | `0.4719` |
| `0.875` | `49035` | `604` | `0.3609` | `0.3078` | `+0.000527` | `-0.000891` | `0.4901` |
| `0.90` | `33630` | `428` | `0.3661` | `0.3154` | `+0.004460` | `+0.001795` | `0.4977` |

切片挖掘的防泄漏约束：筛选条件只使用 `year`、`board`、`asset_bucket`、`prob_buy`、`score` 等非未来列；`future_open_to_close_5d`、`future_open_to_open_5d`、`label` 只用于评价。需要注意，按 `year` 观察是 regime 诊断，不是可直接上线的因果交易规则。

事后样本审计发现，R15 的“提高负样本比例”假设在当前 fold 上没有真正被检验。`trade_interval_buy_2012_2019_v1` 的 train split 已经是 1:1 平衡：

```text
train label counts: flat=382852, buy=382852
neg_over_pos = 1.0
negative_multiplier=2.0 target negatives = 382852
negative_multiplier=3.0 target negatives = 382852
```

`BalancedBlockSampler` 对 `binary_balanced` 的逻辑是：

```text
negative_target = min(negative_count, ceil(positive_count * negative_multiplier))
```

因此只要 `negative_multiplier >= 1.0`，当前 train split 都会采满全部 flat 负样本。K04 相对 K03 best 的差异不能归因于 `negative_multiplier=2.0`，更可能来自 10 epoch schedule、early stop、checkpoint selection 或随机路径差异。

### K04 反事实分析

1. 表面上 K04 best OOS precision `0.2551` 高于 K03 best 的 `0.2454`，Top20 buy hit `0.3077` 与 K03 best `0.3073` 基本持平。但事后样本审计说明这不能归因于 `negative_multiplier=2.0`。
2. K04 没有改善 broad TopK 交易收益：K04 best Top20 5 日 open-to-close `-0.002513`，略弱于 K03 best 的 `-0.002441`。在当前 fold 上，“调高 negative_multiplier”实际是 no-op，不能作为解决收益排序的证据。
3. 用户关于“特定情况下 precision 应该很高”的判断被验证。K04 best 在 `prob>=0.90` 全市场切片达到 buy precision `0.4440`，`board=chinext,asset_bucket=2,prob>=0.85` 达到 `0.4543`，且多个条件切片 5 日收益为正。
4. 问题不再是“模型完全没有高 precision”，而是“高 precision 发生在窄覆盖/特定 regime/特定板块资产桶里”。全市场每日 Top20 会混入大量弱条件样本，导致收益均值被稀释。
5. best checkpoint 明显优于 final checkpoint。K04 final 的 OOS precision 只有 `0.2262`，高阈值 `prob>=0.90` precision `0.3661`，弱于 K04 best 的 `0.4440`。早停仍是必要条件。
6. 高阈值切片的收益多为 open-to-close 口径，尚未扣真实手续费、滑点、涨跌停、停牌和换手成本。它证明存在候选信号，不等同于已经可交易。

### 更新结论

`conditional_high_precision_exists_but_broad_topk_unresolved`。K04 否定了“模型无法产生高 precision”的悲观结论，但也没有通过 broad TopK 替代 alpha 的标准。当前更合理的下一步不是继续只看全市场 accuracy/F1，而是把研究目标拆成两层：

1. 训练侧继续执行 K05 验证 focal loss；K06 需要先审计采样是否真能改变训练分布，否则不能作为更强负样本先验实验。
2. 评测侧新增“条件部署”口径：高置信阈值 + 非未来板块/资产桶/市场状态过滤 + 分年稳定性 + 真实交易约束。
3. 若 K05/K06 仍只提升切片 precision、不改善 broad TopK，则下一轮应转向收益加权 BCE、pairwise/listwise ranking 或直接以条件过滤后的候选集训练二阶段 ranker。

R15 当前状态：`k04_k05_completed_k06_aborted_noop`。

## 9. R16：K05 focal loss 对照

### 假设

R15/K04 显示提高负样本比例只能小幅提升 OOS precision，不能改善 broad TopK 收益。若问题来自 CE 被易样本主导、难负样本没有被充分惩罚，则在相同 `negative_multiplier=2.0` 下改用 focal loss，应该提升高质量买点识别，表现为：

1. validation_select buy F1 或 precision 提升。
2. 2021-2025 prediction Top20 buy hit 高于 K04 best 的 `0.3077`。
3. 高置信 `prob_buy` 切片覆盖不崩塌，并保持正的 5 日收益。

### 实验结果

K05 训练 run：

```text
runs/K05-large20m-trade-interval-buy-neg2-focal-ddp6-20260717-015423
runs/K05-best-diagnostic-20260717
runs/K05-final-diagnostic-20260717
```

训练设置：

| 项目 | 值 |
| --- | --- |
| model | `large20m` |
| DDP | 6 卡 |
| loss | `focal` |
| negative_multiplier | `2.0` |
| epochs | `10` |
| early_stop_patience | `2` |
| early stop | epoch 5 |
| best checkpoint | epoch 3 |
| best validation buy F1 | `0.3756` |

输出文件：

```text
runs/K05-best-diagnostic-20260717/evaluation_validation_select.json
runs/K05-best-diagnostic-20260717/evaluation_prediction.json
runs/K05-best-diagnostic-20260717/analysis_high_precision_slices_k05_best.json
runs/K05-best-diagnostic-20260717/analysis_prediction_best_yearly_threshold_topk.json
runs/K05-final-diagnostic-20260717/evaluation_validation_select.json
runs/K05-final-diagnostic-20260717/evaluation_prediction.json
runs/K05-final-diagnostic-20260717/analysis_high_precision_slices_k05_final.json
runs/K05-final-diagnostic-20260717/analysis_prediction_final_yearly_threshold_topk.json
```

K05 best/final 总指标：

| checkpoint | split | buy precision | buy recall | buy F1 | flat F1 | macro F1 | Top20 buy hit | Top20 5d open-to-close | rank IC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best epoch 3 | 2020 validation_select | `0.2866` | `0.5448` | `0.3756` | `0.8173` | `0.5965` | `0.3415` | `+0.003346` | `-0.0190` |
| final epoch 5 | 2020 validation_select | `0.2641` | `0.5829` | `0.3635` | `0.7876` | `0.5755` | `0.3309` | `+0.002162` | `0.0021` |
| best epoch 3 | 2021-2025 prediction | `0.2572` | `0.3461` | `0.2951` | `0.8895` | `0.5923` | `0.3031` | `-0.003293` | `-0.0355` |
| final epoch 5 | 2021-2025 prediction | `0.2329` | `0.4204` | `0.2998` | `0.8646` | `0.5822` | `0.2898` | `-0.001697` | `-0.0167` |

K04/K05 同口径对比：

| run | checkpoint | prediction buy precision | prediction buy recall | prediction buy F1 | Top20 buy hit | Top20 5d open-to-close |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| K04 CE | best epoch 3 | `0.2551` | `0.3961` | `0.3103` | `0.3077` | `-0.002513` |
| K05 focal | best epoch 3 | `0.2572` | `0.3461` | `0.2951` | `0.3031` | `-0.003293` |
| K04 CE | final epoch 5 | `0.2262` | `0.4420` | `0.2992` | `0.2880` | `-0.002396` |
| K05 focal | final epoch 5 | `0.2329` | `0.4204` | `0.2998` | `0.2898` | `-0.001697` |

K05 best 每日固定 TopK：

| 策略 | buy hit | 5d open-to-close | 5d open-to-open | 正收益天数占比 |
| --- | ---: | ---: | ---: | ---: |
| Top10 | `0.3123` | `-0.002764` | `-0.005447` | `0.4598` |
| Top20 | `0.3031` | `-0.003293` | `-0.005680` | `0.4623` |
| Top50 | `0.2875` | `-0.002799` | `-0.004752` | `0.4623` |

K05 best 分年分类指标：

| 年份 | 自然 buy 占比 | 预测 buy 占比 | buy precision | buy recall | buy F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | `0.1264` | `0.1891` | `0.2515` | `0.3764` | `0.3015` |
| 2022 | `0.1218` | `0.2111` | `0.2493` | `0.4319` | `0.3161` |
| 2023 | `0.0816` | `0.1022` | `0.2185` | `0.2738` | `0.2431` |
| 2024 | `0.1400` | `0.2533` | `0.2691` | `0.4868` | `0.3466` |
| 2025 | `0.1136` | `0.0414` | `0.3439` | `0.1254` | `0.1837` |

K05 best 概率分布出现明显收缩：

| checkpoint | max prob_buy | p99 | p99.9 | `prob>=0.75` rows | `prob>=0.80` rows | `prob>=0.85` rows | `prob>=0.90` rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K05 best | `0.8538` | `0.6469` | `0.7538` | `4593` | `469` | `3` | `0` |
| K05 final | `0.9979` | `0.8930` | `0.9743` | `178127` | `113104` | `67062` | `35924` |

K05 best 高阈值切片不具备稳定覆盖：

| prob_buy 阈值 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | 5d open-to-open |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.75` | `4593` | `14` | `0.6797` | `0.4905` | `+0.066635` | `+0.053730` |
| `0.80` | `469` | `6` | `0.8166` | `0.4768` | `+0.100282` | `+0.081062` |
| `0.85` | `3` | `1` | `0.3333` | `0.3333` | `+0.240615` | `+0.168309` |

这组高阈值结果不能视为通过：date_count 过低，主要集中在极少数行情日，不满足稳定部署要求。

K05 final 的高置信切片虽然存在，但它不是 validation buy F1 选中的 checkpoint：

| prob_buy 阈值 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | 5d open-to-open | 正收益天数占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.85` | `67062` | `700` | `0.3569` | `0.3283` | `+0.005724` | `+0.003700` | `0.4843` |
| `0.875` | `49999` | `537` | `0.3666` | `0.3231` | `+0.004111` | `+0.002011` | `0.4953` |
| `0.90` | `35924` | `367` | `0.3831` | `0.3387` | `+0.007661` | `+0.003486` | `0.5177` |
| `0.95` | `13057` | `95` | `0.4701` | `0.3637` | `+0.013310` | `+0.004132` | `0.6000` |

K05 final 代表性条件切片：

| 条件 | rows | date_count | buy precision | daily buy hit | 5d open-to-close | hit lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `asset_bucket=2,prob>=0.95` | `2501` | `56` | `0.4802` | `0.4021` | `+0.018900` | `+0.1307` |
| `board=mainboard_sz,prob>=0.95` | `3265` | `58` | `0.4784` | `0.3656` | `+0.018366` | `+0.1029` |
| `board=chinext,prob>=0.95` | `5137` | `71` | `0.4686` | `0.3785` | `+0.015492` | `+0.1292` |
| `year=2024,board=chinext,prob>=0.925` | `6340` | `34` | `0.4494` | `0.4814` | `+0.037798` | `+0.1615` |

### 反事实分析

1. focal loss 在 validation_select 上改善了 F1：K05 best `0.3756` 高于 K04 best `0.3700`，precision 也从 `0.2825` 提升到 `0.2866`。
2. 但 focal 没有改善 OOS broad TopK：K05 best Top20 buy hit `0.3031` 低于 K04 best `0.3077`，Top20 5 日 open-to-close 从 `-0.002513` 降到 `-0.003293`。
3. K05 best 的 OOS precision 只比 K04 best 多 `0.0021`，但 recall 从 `0.3961` 降到 `0.3461`，buy F1 下降到 `0.2951`。这更像是 focal 让模型更保守，而不是学到了更好的可交易排序。
4. K05 best 的概率分布明显压缩，高置信 `prob_buy>=0.85` 几乎不存在。因此不能用 K04 的高阈值部署逻辑直接迁移到 K05 best。
5. K05 final 虽然重新产生大量高置信切片，并且 `prob>=0.95` precision 达到 `0.4701`，但 final checkpoint 的 broad prediction 和 validation F1 都弱于 best。说明高置信概率与 checkpoint selection 目标不一致，单靠 validation buy F1 无法选出最适合高阈值部署的模型。
6. 2025 分年很关键：K05 best precision `0.3439` 最高，但预测 buy 占比只有 `0.0414`，recall 只有 `0.1254`。这说明 precision 可以通过极低覆盖获得，但不一定能形成稳定交易候选池。

### 更新结论

`focal_improves_validation_f1_but_not_oos_topk`。K05 支持 focal loss 对 validation 分类指标有帮助，但不支持它能改善 2021-2025 broad TopK 或替代 alpha。更重要的是，K05 暴露出 calibration/logit scale 与 checkpoint selection 的问题：best checkpoint 未必有可用高置信区域，final checkpoint 有高置信切片但不是 validation F1 最优。

K06 已不应继续按原定义执行：`negative_multiplier=3.0 + focal` 在当前 1:1 train fold 上与 K05 的训练样本完全相同，不能构成更强负样本先验实验。下一步应停止在当前 fold 上搜索 CE/focal/采样比例，转向：

1. 以 `precision@K`、高阈值覆盖、TopK 收益作为 checkpoint selection。
2. 做收益加权 BCE 或 pairwise/listwise ranking，让 score 直接对齐每日排序收益。
3. 把高置信条件切片作为二阶段候选集，再训练/回测交易过滤器。

## 10. R17：K06 no-op 审计

### 假设

K06 原计划为 `negative_multiplier=3.0 + focal`，用于验证更强负样本先验是否能进一步提升 precision，或是否会导致 recall/覆盖崩塌。

### 执行与中止

K06 启动 run：

```text
runs/K06-large20m-trade-interval-buy-neg3-focal-ddp6-20260717-024337
```

启动后 epoch 1 的训练进度、loss、accuracy 与 K05 完全一致，随后检查 train fold class count：

```text
train_path = data/folds/trade_interval_buy_2012_2019_v1/train.parquet
flat = 382852
buy = 382852
negative_multiplier=2.0 target negatives = 382852
negative_multiplier=3.0 target negatives = 382852
```

K06 epoch 1 validation 也与 K05 epoch 1 完全一致：

| run | epoch | buy precision | buy recall | buy F1 | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K05 | 1 | `0.2789` | `0.5559` | `0.3715` | `0.5900` |
| K06 | 1 | `0.2789` | `0.5559` | `0.3715` | `0.5900` |

因此 K06 被主动中止，避免重复训练和重复 evaluation。中止不是模型失败，而是实验定义无效。

### 更新结论

`k06_aborted_because_negative_multiplier_noop`。当前 fold 已经把 train split 做成 flat/buy 1:1 平衡，`negative_multiplier>=1` 都不会改变样本分布。后续如果要验证负样本先验，必须重新构建自然分布或更高 flat:buy 比例的 train index，而不是只改训练脚本参数。

这也修正 R15/K04 的解释：K04 的结果不能说明提高负样本比例有效；K05 说明 focal 对 validation F1 有小幅帮助，但没有改善 OOS TopK。下一版实验应优先做 checkpoint selection/收益排序目标/条件部署过滤，而不是继续在当前 balanced fold 上调 `negative_multiplier`。
