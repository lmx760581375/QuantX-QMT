# 分钟级独立 AE Reward Baseline 设计

> 状态（2026-09-22）：MinuteAE v4 紧凑标准化 stock cache 已构建并校验；新 64-token 可分离线性 MinuteAE 已完成 20 epochs。H47/H95 三 head Reward 已按日线 `independent_risk_pairs` 合同启动训练。
>
> 目标：只验证 **分钟状态本身** 能否形成独立 alpha；第一轮不输入日线 H7 Reward、WTS score、WTS rank 或候选池特征。

## 1. 范围与不变项

### 1.1 研究问题

在每个交易日收盘时，模型观察最近若干交易日的 5 分钟行情、分钟横截面状态和分钟市场状态，预测之后短期的可交易趋势质量：

```text
分钟状态
-> 新训练的 MinuteAE
-> 冻结的 MinuteAE latent
-> Minute Reward Transformer
-> return / Sharpe / drawdown 三个 reward score
-> 日线 QuantX T+1 回测
```

第一轮的判断标准是：

```text
不输入 H7 reward 或 WTS 候选特征时，
MinuteAE Reward 是否能在独立严格时间外样本中形成正向横截面 alpha。
```

### 1.2 不做的事情

第一轮不做以下扩展：

1. 不复用日线 AE checkpoint 作为 MinuteAE 权重。
2. 不把分钟数据直接塞进当前日线 `BacktestContext.quote`。
3. 不修改 QuantX 的日线成交、涨跌停、T+1 和成本规则。
4. 不把日线 H7 score、WTS score、WTS rank 或 Top50 身份输入 MinuteAE。
5. 不做日线/分钟融合，不做多模型加权。
6. 不对 BJ 单独施加特殊特征规则；若训练池第一版限定沪深 A 股，BJ 直接不进入训练与打分 universe。

### 1.3 5 分钟时间语义

目标 5 分钟 Qlib provider 使用与 Tushare 5 分钟线一致的 48 根完整交易日标签：

```text
上午：09:35, 09:40, ..., 11:30  共 24 根
下午：13:05, 13:10, ..., 15:00  共 24 根
```

因此：

| bar horizon | 完整交易日 |
|---:|---:|
| `H48` | 1 日 |
| `H96` | 2 日 |
| `H144` | 3 日 |
| `H192` | 4 日 |

第一轮正式比较：

```text
输入窗口 L = 240 bars = 5 个完整交易日
主任务一：H96  = 未来 2 个完整交易日
主任务二：H144 = 未来 3 个完整交易日
```

不使用 `H100`、`H200` 等不贴合交易日边界的主 horizon。

## 2. 数据合同

### 2.1 原始与派生数据

原始分钟行情：

```text
/horizon-bucket/saturn_v_dev/mingxiao.li/data/
└── china_a_share_1min_ohlcv_hf_ba589a115_20260921/
```

前复权 5 分钟 Qlib provider：

```text
/horizon-bucket/saturn_v_dev/mingxiao.li/data/
└── china_a_share_5min_qfq_qlib_v1/
```

MinuteAE 实现使用紧凑 lazy cache，不修改现有日线 Kronos 或 Reward 数据集：

```text
dataset_pre2020_stock_cache_v4/
└── data/
│   ├── instruments.parquet
│   ├── calendar_5min.parquet
│   ├── rank_index.parquet                 # 审计副本
│   ├── cross_section_rank_float16.mmap    # 紧凑的 10 维横截面 rank
│   ├── market_cross_section_5min_float32.mmap
│   ├── samples/*.parquet                  # 审计副本
│   ├── runtime_index/*.npy                # H20 训练运行时索引
│   ├── stock_features_normalized_float16.mmap
│   ├── stock_bar_state_uint8.mmap
│   └── scalers/minute_scaler_train_2010_2018_v1.json
```

运行时路径：

```text
/horizon-bucket/saturn_v_dev/mingxiao.li/data/
└── minute_ae_reward_v1/dataset_pre2020_stock_cache_v4/
```

### 2.2 训练 universe

MinuteAE 第一版训练 universe：

```text
沪深 A 股
不包含指数、ETF、BJ
```

不依赖 `is_st` 字段。每个 5 分钟时刻的有效横截面集合由真实分钟行情决定。

对时刻 `t`，定义：

```text
eligible_t:
  有该时刻原始 bar
  open/high/low/close 均有限且 > 0
  high >= low

active_t:
  eligible_t
  且 volume > 0 或 amount > 0
```

用途区分：

| 场景 | 使用集合 |
|---|---|
| 单股序列保留 | `eligible_t`，并通过 `bar_mask` 标记成交有效性 |
| 横截面 rank | `active_t` |
| 收益/宽度 market 统计 | `active_t` |
| 流动性缺失比例 | `eligible_t` 与 `active_t` 的差集 |

训练样本要求：

```text
signal bar 必须 active
过去 L=240 根中至少 220 根 active
目标路径 H 根 bar 均必须有有效 close
```

这样不需要 ST 标签，也不会把停牌或零成交填充 bar 当作正常趋势。

### 2.3 不完整交易日

训练与标签只使用完整交易日。

例如原始 HF 快照最后一天若只覆盖到 10:25，则：

```text
该日不能作为 signal date
该日不能进入 future H96 / H144 path
```

它可以保留在原始 Qlib provider 中用于审计，但不得进入训练样本。

## 3. MinuteAE 输入张量

```text
x_stock  : [B, 240, 52]
x_market : [B, 240, 24]
bar_mask : [B, 240]
```

第一轮：

```text
candidate_dim = 0
```

即没有日线 H7、WTS、候选身份或任何外部 score 输入。

## 4. 52 个个股特征

```text
52 = 10 个 raw-level / 量价字段
   + 32 个个股时序特征
   + 10 个分钟横截面 rank
```

所有收益、比例和滚动统计只能使用时刻 `t` 及此前数据。

### 4.1 10 个 raw-level / 量价字段

这些字段是逻辑输入字段。实现中不保存全市场 raw dense memmap：DataLoader 从前复权 Qlib provider 按股票的 cache block 读取原始窗口，再使用训练 split 拟合并冻结的 scaler 做逐字段 clip + 标准化。它们不是未处理的绝对值直接进入 AE。

| # | 字段 | 原始定义 | 保留原因 |
|---:|---|---|---|
| 1 | `open_qfq` | 当前 5 分钟前复权开盘价 | 保留价格层级和当前 K 线起点 |
| 2 | `high_qfq` | 当前 5 分钟前复权最高价 | 保留上方冲击和价格尺度 |
| 3 | `low_qfq` | 当前 5 分钟前复权最低价 | 保留下方压力和价格尺度 |
| 4 | `close_qfq` | 当前 5 分钟前复权收盘价 | 保留当前价格水平与序列状态 |
| 5 | `volume` | 当前 5 分钟成交量 | 保留单 bar 真实参与量 |
| 6 | `vwap_qfq` | `amount / volume` 后按同日复权比例调整 | 保留成交均价与 close 的关系 |
| 7 | `change_1b` | `close_t / close_(t-1) - 1` | 原始量价块内的最基础单 bar 变化 |
| 8 | `amount` | 当前 5 分钟成交额 | 保留资金规模与容量 |
| 9 | `cum_volume_tod` | 当日截至当前 bar 的累计成交量 | 保留当日参与进度 |
| 10 | `cum_amount_tod` | 当日截至当前 bar 的累计成交额 | 保留当日资金参与进度 |

不放入 raw block 的字段：

```text
factor：
  目标 provider 已经是前复权，factor 恒为 1，没有学习信息。

turnover：
  当前分钟原始数据没有与日线相同口径的分钟换手率，不能伪造。
```

### 4.2 raw block 标准化合同

MinuteAE 必须和现有日线 AE 一样，保存独立的 `minute_scaler.json`。统计量不在 AE 权重内，而是 checkpoint 的必要外部依赖。

对每个 raw 字段独立拟合：

```text
clip_low  = 训练样本 token 的 P0.5
clip_high = 训练样本 token 的 P99.5
mean      = clip 后训练样本均值
std       = clip 后训练样本标准差

x_norm = [clip(x, clip_low, clip_high) - mean] / std
```

其中：

```text
volume
amount
cum_volume_tod
cum_amount_tod
```

先做：

```text
log1p(x)
```

再拟合上述统计量。

约束：

1. scaler 仅从训练时间段样本拟合；
2. 验证、测试、2020-2026 回测与线上推理只读取冻结 scaler；
3. 推理的字段顺序、复权口径、`log1p` 处理和 scaler 必须与训练完全相同；
4. `minute_scaler.json` 必须记录训练日期范围、样本数、token 数和 feature schema hash。

这样保留 OHLC 与量价的 level 信息，同时把价格、成交额和成交量的数值尺度收敛到 AE 可稳定训练的范围。

### 4.3 8 个当前 bar / 当日状态特征

| # | 字段 | 计算 | 含义与目的 |
|---:|---|---|---|
| 1 | `ret_close_1b` | `close_t / close_(t-1) - 1` | 当前 5 分钟动量 |
| 2 | `gap_open_1b` | `open_t / close_(t-1) - 1` | 当前 bar 开盘跳变 |
| 3 | `ret_intrabar` | `close_t / open_t - 1` | 当前 bar 内买卖方向 |
| 4 | `high_open` | `high_t / open_t - 1` | 上方冲击强度 |
| 5 | `low_open` | `low_t / open_t - 1` | 下方抛压强度 |
| 6 | `range_hl` | `high_t / low_t - 1` | 当前 5 分钟振幅 |
| 7 | `close_position` | `(close_t-low_t)/(high_t-low_t)` | 收盘接近日内高还是低 |
| 8 | `cum_amount_tod_z20` | 截至当前 bar 的当日累计成交额，与过去 20 个交易日同一时刻的累计成交额比较 | 当前交易日异常活跃度 |

其中 `tod` 是一天中的 48 个固定 bar 槽位之一。例如 14:55 只与过去交易日的 14:55 比较，不与开盘 09:35 比较。

累计成交额特征具体定义：

```text
A_tod(d, k) = sum(amount(d, 1:k))

cum_amount_tod_z20(d, k) =
  [log(1 + A_tod(d, k)) - mean(log(1 + A_tod(d-20:d-1, k)))]
  / std(log(1 + A_tod(d-20:d-1, k)))
```

只使用过去 20 个交易日，不使用当日之后数据。

### 4.4 3 个滚动窗口，每窗口 8 个特征

窗口：

```text
W ∈ {12, 48, 240}
```

对应约：

```text
12 bars  = 1 小时
48 bars  = 1 个交易日
240 bars = 5 个交易日
```

每个窗口生成：

| # | 字段 | 计算 | 含义与目的 |
|---:|---|---|---|
| 1 | `return_W` | `close_t / close_(t-W) - 1` | 不同尺度趋势 |
| 2 | `volatility_W` | 过去 W 根 `ret_close_1b` 标准差 | 趋势稳定性/噪声 |
| 3 | `max_drawdown_W` | 过去 W 根 close 路径最大回撤 | 最近趋势的左尾风险 |
| 4 | `distance_to_high_W` | `close_t / max(high[t-W+1:t]) - 1` | 当前距近期高点距离 |
| 5 | `distance_to_low_W` | `close_t / min(low[t-W+1:t]) - 1` | 当前距近期低点距离 |
| 6 | `volume_residual_z_W` | 当前 bar 成交量时段残差在过去 W 根内的 z-score | 异常量能 |
| 7 | `amount_residual_z_W` | 当前 bar 成交额时段残差在过去 W 根内的 z-score | 异常资金流入/流出 |
| 8 | `range_z_W` | 当前 `range_hl` 在过去 W 根内的 z-score | 异常波动 |

量能时段残差先去除日内季节性：

```text
volume_residual_t =
  log(1 + volume_t)
  - mean(log(1 + volume) of same TOD slot in previous 20 trading days)

amount_residual_t =
  log(1 + amount_t)
  - mean(log(1 + amount) of same TOD slot in previous 20 trading days)
```

再在 `W` 根滚动窗口内计算 z-score。

这样不会把开盘天然放量、午后天然缩量误判为异常流动性。

合计：

```text
8 当前状态
+ 3 × 8 个窗口特征
= 32 个个股时序特征
```

### 4.5 10 个分钟横截面 rank

对每个时刻 `t`，仅在 `active_t` 股票集合内做稳定百分位排序：

```text
rank_pct(x_i,t) = rank(x_i,t) / active_count_t
```

结果范围：

```text
(0, 1]
```

| # | 字段 | 排序原始量 | 目的 |
|---:|---|---|---|
| 1 | `ret_1b_rank` | `ret_close_1b` | 当前相对强弱 |
| 2 | `return_12b_rank` | `return_12` | 近 1 小时相对趋势 |
| 3 | `return_48b_rank` | `return_48` | 日内/一日相对趋势 |
| 4 | `return_144b_rank` | `return_144` | 3 日相对趋势 |
| 5 | `return_240b_rank` | `return_240` | 5 日相对趋势 |
| 6 | `volatility_48b_rank` | `volatility_48` | 相对波动状态 |
| 7 | `bar_amount_residual_rank` | 当期 `amount_residual_t` | 相对异常资金活跃度 |
| 8 | `cum_amount_tod_ratio_rank` | 当日累计成交额 / 历史同刻累计成交额均值 | 当日相对放量程度 |
| 9 | `distance_to_high_48b_rank` | `distance_to_high_48` | 相对突破/接近高点程度 |
| 10 | `range_12b_rank` | `range_z_12` | 相对短期异常波动 |

最终：

```text
10 个 raw-level / 量价字段
+ 32 个单股时序特征
+ 10 个横截面 rank
= 52 个 x_stock 维度
```

`change_1b` 与派生块中的 `ret_close_1b` 在数值上同源。第一版保留两者不是为了增加信息维度，而是为了和现有日线 `raw_relative` 数据合同保持一致：

```text
raw block：
  保留基础行情字段的标准化值

derived block：
  保留用于趋势/横截面/窗口统计的相对表达
```

后续可通过去掉其中一个重复字段的消融，验证该冗余是否实际有帮助；但主 baseline 不应在未比较前改变已验证日线 schema 的基本思想。

## 5. 24 个共享市场特征

每个时刻只生成一行：

```text
market[t] ∈ R^24
```

该向量复制给所有同一时刻的股票样本，但在模型中单独作为 `x_market` 分支输入，而不是拼入 52 维个股特征。

### 5.1 4 个市场宽度字段

统计池：`eligible_t`；收益方向只对 `active_t` 股票计算。

| # | 字段 | 计算 | 目的 |
|---:|---|---|---|
| 1 | `up_ratio` | `mean(ret_1b > 0)` | 市场短线扩散上涨程度 |
| 2 | `down_ratio` | `mean(ret_1b < 0)` | 市场短线扩散下跌程度 |
| 3 | `flat_ratio` | `mean(ret_1b == 0)` | 当前不动/平衡状态 |
| 4 | `zero_volume_ratio` | `1 - active_count / eligible_count` | 停滞、流动性枯竭或大面积无成交状态 |

### 5.2 5 个收益分布字段

统计池：`active_t` 股票的 `ret_1b`。

| # | 字段 | 计算 | 目的 |
|---:|---|---|---|
| 5 | `ret_1b_mean` | 横截面均值 | 市场平均方向 |
| 6 | `ret_1b_median` | 横截面中位数 | 抗极端值的典型方向 |
| 7 | `ret_1b_std` | 横截面标准差 | 分化/波动状态 |
| 8 | `ret_1b_amount_weighted_mean` | 按当前 bar 成交额加权的收益均值 | 资金主导方向 |
| 9 | `ret_1b_p90_minus_p10` | P90 - P10 | 横截面离散度 |

### 5.3 8 个标准化宽度直方图字段

先定义：

```text
z_i,t = [ret_1b_i,t - median(ret_1b_t)] / max(MAD_scaled(ret_1b_t), ε)
```

其中 `MAD_scaled = 1.4826 × median(|x - median(x)|)`。

| # | 区间 |
|---:|---|
| 10 | `z < -2` |
| 11 | `-2 <= z < -1` |
| 12 | `-1 <= z < -0.25` |
| 13 | `-0.25 <= z < 0` |
| 14 | `0 <= z <= 0.25` |
| 15 | `0.25 < z <= 1` |
| 16 | `1 < z <= 2` |
| 17 | `z > 2` |

每个字段为该区间股票比例。

目的：用相对当前市场波动的标准化收益分布描述“普涨、普跌、强分化、极端冲击”，避免固定 `±1%/±5%` 阈值在分钟尺度失真。

### 5.4 5 个流动性与参与度字段

| # | 字段 | 计算 | 目的 |
|---:|---|---|---|
| 18 | `active_ratio` | `active_count / eligible_count` | 当前参与交易股票比例 |
| 19 | `bar_amount_sum_tod_z20` | 全市场当前 bar 成交额相对过去 20 日同一时刻的 z-score | 当前市场瞬时活跃度 |
| 20 | `bar_volume_sum_tod_z20` | 全市场当前 bar 成交量相对过去 20 日同一时刻的 z-score | 当前市场参与量 |
| 21 | `cum_amount_sum_tod_z20` | 当日累计全市场成交额相对过去 20 日同一时刻的 z-score | 当天资金活跃状态 |
| 22 | `amount_top10_share` | 当期成交额 Top10% 股票成交额 / 市场总成交额 | 资金集中度 |

全市场成交额/成交量的 TOD z-score 同样只使用过去交易日的同一 bar 槽位。

### 5.5 2 个日内时钟字段

令完整交易日 bar 位置为：

```text
k ∈ {0, ..., 47}
```

| # | 字段 | 定义 | 目的 |
|---:|---|---|---|
| 23 | `tod_sin` | `sin(2πk/48)` | 学习日内周期连续位置 |
| 24 | `tod_cos` | `cos(2πk/48)` | 与 `tod_sin` 共同消除周期边界不连续 |

最终：

```text
4 宽度
+ 5 收益分布
+ 8 标准化直方图
+ 5 流动性/参与度
+ 2 日内时钟
= 24 个 x_market 维度
```

## 6. 离线构建与存储

### 6.1 已实施的紧凑 stock cache（v4）

为避免把上市前/退市后 NaN dense 化，实际实现不采用全市场 dense panel。它保留 Qlib provider 作为唯一原始量价真源，并在离线阶段只对实际覆盖区间一次性生成标准化 52 维个股输入：

```text
5 分钟 Qlib provider
-> 每只股票计算单股来源特征
-> 全市场 10 维 cross-sectional rank（float16、仅实际覆盖区间）
-> 24 维 market vector（float32）
-> frozen scaler
-> stock_features_normalized_float16 [有效股票-bar, 52]
-> runtime_index/*.npy + compact bar state

训练时：
stock feature memmap
+ market 24（mmap + scaler）
-> bar_mask（compact state）
-> x_stock [240, 52] + x_market [240, 24] + bar_mask
```

`runtime_index` 为 H20 训练使用的 NumPy 二进制索引，不依赖 PyArrow；Parquet 索引仍保留为审计副本。已完成的 pre-2020 v4 数据集包含：

```text
5,466 个沪深股票
321,062,208 个有效 stock-bar
5,195,190 个 AE train 样本（2010-2018）
876,459 个 validation 样本（2019）
5.98 GiB cross-section rank cache
31.10 GiB normalized stock cache
```

因此，下述 dense memmap 方案只保留为早期存储对照，不是当前训练运行时合同。

分钟横截面不能在 DataLoader 内动态计算。应按时间顺序离线构建：

```text
5 分钟 Qlib bin
-> raw minute memmap
-> 单股 32 维时序特征
-> 每个 bar 的 10 维横截面 rank
-> 每个 bar 的 24 维 market vector
-> float16 个股 feature shard
-> float32 market memmap
-> 训练时只 mmap 窗口切片
```

推荐存储：

```text
raw_ohlcv_5min_float32:
  [instrument, bar, 8]

raw_trade_5min_float32:
  [instrument, bar, 4]

features_derived_5min_float16.part-*:
  [instrument_shard, bar, 42]

market_cross_section_5min_float32:
  [bar, 24]
```

`bar_mask` 可由 raw state 在线派生，或单独存为：

```text
raw_state_5min_uint8:
  [instrument, bar, has_raw_bar / is_active / reserved]
```

`raw_ohlcv_5min_float32` 保持：

```text
open_qfq, high_qfq, low_qfq, close_qfq,
volume, vwap_qfq, change_1b, factor
```

其中 `factor` 仅为 provider 兼容字段，不送入 MinuteAE。

`raw_trade_5min_float32` 记录：

```text
amount, cum_volume_tod, cum_amount_tod, reserved
```

训练 DataLoader 按固定 schema 拼接：

```text
raw 10
+ derived 42
→ raw_relative: 52

derived 42
→ relative_only: 42
```

不复制 `L=240` 窗口。训练样本只保存：

```text
instrument_idx
bar_idx
target_row
signal_timestamp
label_start_timestamp
label_end_timestamp
```

DataLoader 再按 `bar_idx-L+1 : bar_idx+1` 从 memmap 切片。

## 7. MinuteAE

MinuteAE 独立训练，不加载日线 AE checkpoint。

已实施结构：

```text
x_stock  [B, 240, 52]
x_market [B, 240, 24]
bar_mask [B, 240]

stock projection + market projection
-> 5 分钟 Transformer encoder
-> 每个 channel 共享的时间线性投影 240 → 64
-> channel projection 96 → 128
-> 64 × 128 latent tokens
-> channel projection 128 → 96
-> 时间线性投影 64 → 240
-> reconstruction decoder
```

约束：

1. 不使用 learned query；
2. 不使用 pooling；
3. 不使用会随 `lookback × d_model` 膨胀的 full flatten Linear；
4. Transformer 主体配置对齐日线 AE：`d_model=96`、4 层 encoder、4 heads、1 层 decoder；
5. 总参数量为 659,580。

AE 重建目标：

1. 重建全部有效 bar 的标准化 `x_stock` 与 `x_market`；
2. 对个股 raw、时序、横截面 rank、市场特征分别记录 reconstruction loss；
3. `bar_mask=0` 的位置不参与 reconstruction loss。

建议损失：

```text
L_AE =
  0.70 * Huber(stock_raw_and_time_series)
+ 0.15 * Huber(stock_cross_section_rank)
+ 0.15 * Huber(market)
```

权重是首版起点，不是固定结论。每类 loss 必须分别记录，避免 market branch 或 rank branch 被总 loss 掩盖。

## 8. Minute Reward 训练

### 8.1 因果时间合同

日线 QuantX 首版仍使用：

```text
D 日收盘生成 score
D+1 收盘成交
```

因此 Minute Reward 的状态与标签定义为：

```text
状态截止：D 日 15:00
entry：D+1 日 15:00 close
future path：从 D+2 日 09:35 开始
```

对于 H96：

```text
预测 D+2 至 D+3 的完整 96 根 5 分钟路径
```

对于 H144：

```text
预测 D+2 至 D+4 的完整 144 根 5 分钟路径
```

这样 score、成交与 label 不重叠，不会泄露 D+1 日盘中信息。

### 8.2 三个 reward head

第一版保持现有日线 Reward 的三头结构：

| head | 连续监督量 | 排序方向 |
|---|---|---|
| `return` | `close_end / entry_close - 1` | 越大越好 |
| `sharpe` | H 根 5 分钟 path return 的 Sharpe | 越大越好 |
| `drawdown` | H 根路径最大回撤的相反数或 percentile | 风险越小越好 |

训练仍使用同一 `signal_date` 内的 Bradley-Terry pairwise loss。

`up/range/down` 只用于样本均衡，不是 Reward 输出类别：

```text
同一 signal date 内按 terminal return 分成上 1/3、中 1/3、下 1/3。
```

主 reward pair：

```text
return(A) > return(B) + min_return_gap
```

风险 head：

```text
当 |return(A) - return(B)| 足够小，
Sharpe 或 drawdown percentile 的优势足够大时构造风险 pair。
```

## 9. 时间切分与 Purge

MinuteAE 和 Minute Reward 必须按时间切分，不能随机切样本。

建议：

```text
训练：pre-2020
验证：2019 或 rolling validation
测试/正式回测：2020-2026
```

相邻 split 边界 purge：

```text
purge trading days >= 1 + ceil(H / 48)
```

因此：

| horizon | 最小 purge |
|---:|---:|
| H96 | 3 个交易日 |
| H144 | 4 个交易日 |

若验证集也参与 scaler、AE checkpoint 或 early stopping，所有 scaler 只拟合训练区间。

## 10. Score Artifact 与日线回测接入

Minute Reward 推理仅在每日 15:00 输出一次全市场 score artifact：

```text
signal_date
instrument
minute_return_score_h96b
minute_sharpe_score_h96b
minute_drawdown_score_h96b
minute_return_topn_risk_score_h96b
```

格式沿用现有 QuantX artifact：

```text
Parquet:
  signal_date + instrument + score columns

JSON manifest:
  kind = quantx_market_score_artifact_v1
  rows / dates / instruments
  score_columns
  AE checkpoint
  feature schema hash
  calendar hash
  horizon / entry contract
  source provider
```

日线 QuantX 无需改造，使用外部 score：

```yaml
selector:
  mode: precomputed
  lag: 1
  external_score:
    path: minute_reward_h96b.parquet
    date_col: signal_date
    instrument_col: instrument
    score_col: minute_return_topn_risk_score_h96b
    missing: drop
    require_artifact_manifest: true
```

因果链：

```text
D 15:00 的 5 分钟状态
-> Daily minute reward score
-> QuantX lag=1
-> D+1 close 日线成交
```

## 11. 首轮实验矩阵

| 实验 | 输入 | Horizon | 目的 |
|---|---|---:|---|
| `M0` | 当前日线 H7 Reward | 7 日 | 固定对照 |
| `M1` | `raw_relative`，`x_stock=52`，`x_market=24` | H96 | 主 baseline：验证 2 日分钟趋势 alpha |
| `M2` | `raw_relative`，`x_stock=52`，`x_market=24` | H144 | 主 horizon 对照：验证 3 日分钟趋势 alpha |
| `M3` | `relative_only`，`x_stock=42`，`x_market=24` | H96 | 检验移除 OHLC/量价 level 后是否仍有效 |
| `M4` | `raw_relative`，`x_stock=52`，market 输入置零 | H96 | 检验分钟市场/横截面状态是否贡献 alpha |

第一轮不做 H7 / Minute fusion。

每个实验必须报告：

1. 同日全市场 Top5 / Top10 的 future terminal return；
2. future path Sharpe 与最大回撤；
3. same-date pair accuracy；
4. 日均 RankIC；
5. 日线 QuantX 回测：收益、年化、最大回撤、交易数、拒单；
6. 与 M0 H7 score 的 Spearman 相关，仅作为诊断，不作为输入；
7. 每个 feature family 的缺失率与 scaler 统计；
8. 完整的 score artifact manifest。

## 12. 通过与失败标准

MinuteAE Reward 不是因训练 loss 收敛就成立。

`M1` 或 `M2` 至少满足：

```text
独立时间外样本：
  日均 RankIC > 0
  Top5 或 Top10 selection advantage > 0
  日线 QuantX 回测不劣于明确基线
```

若分钟模型只在训练期有效、只改善 pair accuracy、但 TopK 和日线回测无增益，则结论是：

```text
分钟状态在当前表示/标签合同下未形成可交易独立 alpha。
```

不得以训练 loss、单日案例或未经 purge 的分钟样本替代正式结论。
