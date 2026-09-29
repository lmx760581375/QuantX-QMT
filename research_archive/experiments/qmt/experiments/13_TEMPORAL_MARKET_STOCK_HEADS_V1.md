# 日线 Temporal Market / Stock Heads v1 实验方案

> 日期：2026-09-27
> 状态：已确认，进入实现、测试、Review 与 AIDI 提交流程。

## 1. 实验目标

在不改变现有日线 Reward 基线的前提下，冻结：

- 日线 AE；
- Reward Transformer backbone；
- 原 return / Sharpe / drawdown heads；
- 原 Top50 后风险重排与 QuantX 执行合同。

只在冻结的 Reward CLS 表征后增加两个互不依赖的轻量 head：

```text
Frozen daily AE
    -> Frozen Reward backbone
    -> frozen state representation z(i,t), 768 dims
        ├─ Market Head
        └─ Stock Temporal Head
```

两个新 head 不反向更新 AE、Reward backbone 或原 head。

职责固定为：

```text
原 Reward heads       -> 决定买谁
Market Head           -> 决定组合总仓位
Stock Temporal Head   -> 决定已选股票之间的相对权重
```

第一阶段只验证两个新 head 是否含有稳定信息，不直接修改正式选股结果。

## 2. 时间切分

为了与历史 Market JEV 的跨期实验保持可比，并避免把原 Reward backbone 已经训练过的
2018 数据当作独立验证：

```text
训练：       2010-01-01 至 2017-12-31
2018：       不作为新 head 的独立验证
验证：       2019-01-02 至 2019-09-10
诊断回测：   2020-2026，仅在模型与规则冻结后使用
```

实际可用日期由当前 frozen Reward 数据合同决定。当前股票状态数据中：

```text
训练股票-日期行：3,544,976
训练交易日：1,865
训练股票：3,124
```

历史 Market JEV 的训练日期为 1,874；差异来自两套数据的 lookback、target
完整性和样本可用性。新实验不填造缺失状态，Market Head 使用当前缓存中全部
1,865 个可用训练日期。

## 3. 冻结表征缓存

训练时禁止重复运行大型 AE 和 Reward backbone。先在本机 `/home` 生成缓存：

```text
stock_state_float16.mmap   [N_stock_date, 768]
rows.parquet               row_id / instrument / signal_date / date_code
stock_labels.parquet       H7 return / cross-sectional percentile
market_state_float32.npy   [N_date, 1536]
market_labels.parquet      H7 market outcomes / market opportunity
manifest.json              shape / hash / checkpoint / scaler / split
```

`market_state` 定义为同一日期全部股票冻结状态表示的：

```text
concat(cross-sectional mean, cross-sectional std)
```

这样 Market Head 能同时观察市场中心状态和横截面分歧。

缓存生产流程固定为：

```text
本机 /home/... 临时目录
-> 完整生成
-> shape、行数、finite、SHA-256 和随机样本校验
-> 一次性复制到 bucket
-> AIDI 训练阶段只读 bucket cache
```

不在生产缓存过程中向 bucket 随机写 mmap。

预计训练缓存体积约：

```text
3,544,976 × 768 × float16 ≈ 5.1 GiB
```

加上验证、标签、索引和 manifest，整体应低于 8 GiB。

## 4. Market Head

### 4.1 标签

对每个 signal date `t`，用全市场股票未来 H7 路径构造策略无关结果：

- 全市场等权 H7 收益；
- 全市场股票 H7 收益中位数；
- H7 正收益股票比例；
- 全市场等权路径最大回撤安全性；
- 全市场等权路径下行波动安全性。

每个分量仅使用训练期经验 CDF 转为 `[0,1]` 历史分位，风险项转换为
“越安全越高”。最终：

```text
market_opportunity_h7 =
mean(
    equal_weight_return_percentile,
    median_stock_return_percentile,
    positive_ratio_percentile,
    drawdown_safety_percentile,
    downside_safety_percentile
)
```

该标签不依赖 Reward、弱转强或任何具体策略。

### 4.2 全量 Pair

训练日期为 1,865 日，因此一个 epoch 的无序 Pair 总数必须严格为：

```text
C(1865, 2) = 1,738,180
```

硬性要求：

- 每个 epoch 必须遍历全部 1,738,180 对；
- 不按 gap 丢弃 Pair；
- 不设置 `steps_per_epoch` 截断；
- 不用“batch 内组合数量”冒充已训练样本；
- DDP 各 rank 处理不重叠分片，汇总 pair count 必须等于 1,738,180；
- epoch 结束后才能运行完整 validation。

方向：

```text
market_opportunity_h7(t_good) > market_opportunity_h7(t_bad)
-> market_score(t_good) > market_score(t_bad)
```

### 4.3 模型

```text
LayerNorm(1536)
-> Linear(1536, 256)
-> GELU
-> Dropout(0.05)
-> Linear(256, 1)
```

只训练该 head。

## 5. Stock Temporal Head

### 5.1 标签

为避免重复学习市场牛熊，个股标签不是原始绝对收益，而是：

```text
stock_relative_h7(i,t)
= 股票 i 的未来 H7 净收益在 signal date=t 横截面中的 percentile
```

因此该 head 学习：

> 同一股票在什么状态下更可能跑赢当日其他股票。

### 5.2 PairPool 规模

2010-2017 当前数据的同股无序 Pair 总数：

```text
2,567,050,467
```

要求两个日期至少相隔 60 个训练交易日后，仍有：

```text
2,374,084,195
```

该规模不适合完整遍历。首轮每个 epoch 固定采样：

```text
2,000,000 个同股跨期 Pair
```

采样合同：

- 同一股票；
- 日期至少间隔 60 个训练交易日；
- 时间距离按 `60-250 / 251-750 / 750+` 三档平衡；
- 标签差按困难、中等、明显三档平衡；
- 股票按可用 Pair 数的平方根分配 quota，防止长历史股票垄断；
- 每个 epoch 使用不同确定性 seed；
- A/B 输入顺序随机翻转；
- 不把股票代码输入模型；
- 记录 unique stock、unique date、unique endpoint 与重复率。

### 5.3 模型

```text
LayerNorm(768)
-> Linear(768, 256)
-> GELU
-> Dropout(0.05)
-> Linear(256, 1)
```

方向：

```text
同一股票 i：
stock_relative_h7(i,t_good) > stock_relative_h7(i,t_bad)
-> stock_temporal_score(i,t_good) > stock_temporal_score(i,t_bad)
```

## 6. 训练与多机多卡

使用一个 AIDI `4 machines × 8 H20` 作业，并行运行两个独立实验：

```text
节点 0-1：Market Head，2×8 H20
节点 2-3：Stock Temporal Head，2×8 H20
```

两边使用独立 rendezvous、日志和 checkpoint 目录。

计划参数：

```text
epochs: 5
per-GPU pair batch: 8192
optimizer: AdamW
precision: BF16
validation: 每个 epoch 后完整执行
```

缓存后的 head 很小，目标是每个 head 在 1-2 小时内完成；若 preflight 估算超过
2 小时，先提高 batch 或优化 mmap 顺序读取，不缩减 Market Head 的全量 Pair。

日志必须包含：

- 缓存和 PairPool manifest；
- 每个 epoch 的 expected / processed pair count；
- step、吞吐、ETA、GPU peak memory；
- train pair accuracy/loss；
- validation pair accuracy/loss；
- 分年份 validation；
- checkpoint 与配置。

## 7. AIDI 提交与占卡处理

提交前：

1. 代码与测试 Review 通过；
2. 本地 cache 完整生成并复制到 bucket；
3. AIDI dry-run 核对 4×8 H20、镜像、挂载、命令和输出路径；
4. 查询本人当前任务。

只有在训练提交因资源不足不能运行时：

- 精确识别本人名下 `sleep` 命令且资源为 `4×8 H20` 的任务；
- 展示任务 ID、名称、资源和状态；
- 仅停止该精确任务；
- 不停止其他用户任务或本人其他训练任务；
- 停止后验证其进入终态，再提交训练。

用户已在本实验方案中明确授权上述精确 sleep-holder 停止操作。

## 8. 测试与 Review

提交前必须完成：

1. Pair 数学与索引测试；
2. Market 全 Pair 无遗漏、无重复测试；
3. Stock Pair 同股、最小间隔、方向和 quota 测试；
4. frozen backbone 参数无梯度测试；
5. cache manifest/shape/hash 测试；
6. 单卡 smoke：完成一个小 epoch 与 validation；
7. 2-process DDP smoke：全局 processed pair count 精确；
8. `compileall`、定向 pytest、`diff --check` 等价检查；
9. 独立代码 Review 与测试 Review，问题修正后复跑。

## 9. 本轮完成标准

本轮不以“提交成功”作为完成。

必须看到 AIDI 正式任务：

- 进入 `Running`；
- 两个 head 均开始训练；
- Market Head 完成第一个全量 `1,738,180` Pair epoch；
- Stock Head 完成第一个 `2,000,000` Pair epoch；
- 两个 head 的 epoch-1 validation 均成功落盘；
- 日志不存在 non-finite、DDP hang、输入缺失或 pair count 不一致。

达到以上条件后，才报告本轮训练已健康启动。模型最终收益效果需待完整训练和冻结后的独立评估。

## 10. 本地 preflight 结果

2026-09-27 已完成：

- 8×RTX 5090 导出正式缓存，4,090,867 个状态，约 6.5 GiB；
- Market Pair：1,865 个训练日的 1,738,180 个无序 Pair 全量落盘；
- Stock Pair：每 epoch 2,000,000 对，覆盖 3,023 只股票；
- Stock Pair 重复率约 0.04%，unique endpoint 约 238 万；
- 4 GPU Market 完整 epoch：1,738,180 对，约 3.9 秒；
- 4 GPU Stock 完整 epoch：2,000,000 对，约 3.2 秒；
- Market 2019 validation：14,365 对；
- Stock 2019 validation：500,000 对。

本地 preflight 证明训练主体远低于 1-2 小时时限；正式 AIDI 时间主要将来自
每节点顺序复制约 6.5 GiB cache、容器启动和多机 rendezvous，而不是 head 训练本身。

## 11. 提交前 Review 记录

Review 已发现并修复：

1. validation rows 的日期序号最初未按 validation split 重建，导致独立 verifier
   无法证明时间间隔；现已改为每个 split 内独立、确定性编号。
2. Stock Pair 最初总是把 better 样本放在左侧；现已随机交换左右端，并持久化
   `target=0/1`，排除位置捷径。
3. Pair 文件最初只有数量审计；现已增加每个 `.npy` 的 SHA-256、Market 全量
   Pair 唯一性检查和 Stock Pair 抽样语义检查。
4. AIDI 启动脚本在每个节点先顺序复制 bucket cache 到本机 `/home`，再执行完整
   verifier，避免训练阶段直接对 bucket mmap 做随机读取。

提交前验证：

- 新增定向测试：4 passed；
- 相关 JEV/Pair/Residual 测试：16 passed；
- Ruff：通过；
- compileall：通过；
- shell syntax：通过；
- bucket cache 完整 verifier：通过；
- AIDI submit dry-run：4×8 H20 配置通过。

全量 `tests/research` 收集被当前环境已有的 `gymnasium` 与新版 JAX
`DeviceArray` 不兼容阻断；该错误发生在未涉及本实验的
`tests/research/test_rl_environment.py` 导入阶段，不是本次代码回归。

## 12. AIDI 正式训练结果

有效正式作业：

```text
job_id: acloud-d9af05010933
resource: (H20*8)*4
phase: Succeeded
start: 2026-09-27 13:46:00
finish: 2026-09-27 13:47:37
```

4 个节点分成两个独立的 `2×8 H20` DDP group；每个节点先把 cache
顺序复制到按 `JOB_ID + HOSTNAME` 隔离的本地目录，并通过 verifier 后再训练。

### Market Head

| Epoch | Train pairs | Train accuracy | 2019 validation pairs | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,738,180 | 57.40% | 14,365 | 56.16% |
| 2 | 1,738,180 | 61.51% | 14,365 | **59.84%** |
| 3 | 1,738,180 | 63.42% | 14,365 | 59.52% |
| 4 | 1,738,180 | 64.21% | 14,365 | 59.40% |
| 5 | 1,738,180 | 64.49% | 14,365 | 59.42% |

Market Head 的最佳 validation 为 epoch 2。训练继续改善而 validation
略回落，说明即使全量 Pair 很快，也不应仅因训练 loss 下降而选择最后 checkpoint。

### Stock Temporal Head

| Epoch | Train pairs | Train accuracy | 2019 validation pairs | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,000,000 | 54.34% | 500,000 | 55.03% |
| 2 | 2,000,000 | 55.18% | 500,000 | 55.26% |
| 3 | 2,000,000 | 55.31% | 500,000 | 55.29% |
| 4 | 2,000,000 | **55.36%** | 500,000 | 55.28% |
| 5 | 2,000,000 | 55.33% | 500,000 | **55.29%** |

Stock Temporal Head 的信号较弱但稳定为正，训练与验证差距很小。当前只证明
同股跨期排序中存在可学习信息，尚未证明它能改善 Top10 权重或最终回测。

### 无效重试边界

- `acloud-75db5874e3ff`：启动脚本变量提前展开，`ExitCode1`，无有效训练。
- `acloud-87282930b103`：Review 后 Pair cache 未同步，主动停止，结果不使用。
- `acloud-1126122a66e0`：节点 verifier 的 Parquet 读取失败后仍继续训练；
  虽然作业显示 `Succeeded`，但违反“校验通过后才训练”的合同，结果作废。
- `acloud-1cbac2b4b2bc`：加入节点独立目录和 fail-fast 后，AIDI PyArrow
  仍无法读取本地生成的 Parquet，
  verifier 阻止训练，`ExitCode1`。

只有 `acloud-d9af05010933` 使用 NPZ 审计索引并通过了所有节点缓存 verifier，
作为正式结果。

## 13. Stock Temporal 20M × 10 扩展实验

用户确认继续扩大 Stock Temporal Head 的训练量，Market Head 不重训，保留
epoch 2 作为当前历史验证最优 checkpoint。

固定合同：

```text
Stock Temporal train pairs / epoch: 20,000,000
Epochs: 10
Total train pairs: 200,000,000
2019 validation pairs / epoch: 500,000
Minimum train date gap: 60 trading dates
DDP: 2 machines × 8 H20
Pair batch / GPU: 8192
Learning rate: 1e-4
Warmup: 5%
Schedule: cosine decay
```

继续保持：

- 同一股票的两个时期；
- 三档时间距离、三档标签差；
- 每个 epoch 独立确定性 seed；
- 随机左右端与显式 target；
- 不输入股票 ID；
- 每轮完整 validation；
- 保存全部 10 个 checkpoint，以 validation 选择，不默认使用最后一轮。

本地正式规模 preflight：

```text
4×RTX 5090
20,000,000 train pairs: 23.14 seconds
500,000 validation pairs: 0.54 seconds
epoch-1 validation accuracy: 55.22%
```

Pair cache：

```text
10 × 20,000,000 = 200,000,000 train pairs
size: approximately 1.7 GiB
epoch-1 unique directed pairs: 19,925,074
epoch-1 duplicate ratio: 0.37%
unique endpoints: 3,537,593
```

### AIDI 训练结果

```text
job_id: acloud-f73cabf4688a
resource: (H20*8)*2
phase: Succeeded
start: 2026-09-27 14:04:39
finish: 2026-09-27 14:08:23
```

节点缓存与 20M×10 Pair manifest 均通过 verifier；10 个 checkpoint 全部落盘，
没有 non-finite、DDP hang 或 Pair count 错误。

| Epoch | Train pairs | Train accuracy | Validation pairs | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20,000,000 | 54.45% | 500,000 | **55.26%** |
| 2 | 20,000,000 | 55.44% | 500,000 | 55.25% |
| 3 | 20,000,000 | 55.54% | 500,000 | 55.20% |
| 4 | 20,000,000 | 55.61% | 500,000 | 55.18% |
| 5 | 20,000,000 | 55.66% | 500,000 | 55.17% |
| 6 | 20,000,000 | 55.70% | 500,000 | 55.13% |
| 7 | 20,000,000 | 55.71% | 500,000 | 55.15% |
| 8 | 20,000,000 | 55.74% | 500,000 | 55.13% |
| 9 | 20,000,000 | **55.76%** | 500,000 | 55.15% |
| 10 | 20,000,000 | 55.75% | 500,000 | 55.14% |

总计训练 200,000,000 Pair。最佳 validation 是 epoch 1；随着训练量增加，
train accuracy 持续提高，而 validation 缓慢下降。相比上一轮 2M×5 的最佳
validation `55.29%`，本轮最佳 `55.26%` 没有提升。

当前证据说明：

- Stock Temporal 信息确实稳定高于随机；
- 训练量不再是主要瓶颈；
- frozen backbone 下的轻量 MLP head 已接近当前可提取信号上限；
- 后续不应继续单纯增加 Pair 或 epoch；
- 若继续，应检验更丰富的 temporal task/head、有限 adapter，或直接验证该弱信号
  能否在 Top10 权重调整中产生净收益。
