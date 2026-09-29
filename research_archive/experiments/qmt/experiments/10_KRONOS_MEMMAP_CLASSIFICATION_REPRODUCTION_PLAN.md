# 10 Kronos 三分类信号 Memmap 复现工程方案

> 版本：v0.3.0
>
> 日期：2026-07-16
>
> 状态：`implementation_plan`
>
> 目标：基于 QuantX BaoStock/Qlib 数据，使用 memmap 管线复现 Kronos 风格的 BUY/HOLD/SELL 日频三分类信号，避免物化重复的滑动窗口张量。
>
> 边界：本文是工程复现方案，不声明外部 Kronos 结果已经在本地验证；在预测层指标和正式账户校验通过之前，不进入生产策略逻辑。

## 1. 问题定义

`09_KRONOS_CLASSIFICATION_SIGNAL_DESIGN.md` 定义了研究方向：一个轻量级时间序列编码器读取日频价量序列，并输出三个类别：

```text
0 = SELL
1 = HOLD
2 = BUY
```

首要工程问题是数据体量。如果用朴素数据集把每个样本都保存成完整窗口：

```text
X_windows shape = [num_samples, 60, feature_dim]
```

会非常浪费，因为相邻信号日期共享 60 行里的 59 行。以 1000 万样本和 52 个特征为例：

```text
10,000,000 * 60 * 52 * 4 bytes ~= 124.8 GB
```

这还会带来写入慢、实验迭代慢和不必要的存储压力。第一版实现应当只存一次稠密的个股 instrument-date-feature 面板，以及一份按交易日对齐的市场横截面因子面板，并用索引表在训练时切片生成窗口。

## 2. 当前可用数据路径

解压出的 `QuantX-QMT-qmt-mac` 树不包含完整市场数据 provider。可用数据位于现有 QuantX 项目：

```text
本方案代码根目录：
  ${HOME}/git/quantization/QuantX-QMT-qmt-mac

快速 Qlib provider：
  ${HOME}/git/quantization/quantx/data/qlib_data_fixed

原始 BaoStock CSV：
  ${HOME}/git/quantization/quantx/data/raw/baostock

证券元数据：
  ${HOME}/git/quantization/quantx/data/meta/snapshots/security_master.csv

行业元数据，v1 仅用于诊断：
  ${HOME}/git/quantization/quantx/data/meta/snapshots/industry_membership.csv
```

已确认的 provider 数据规模：

```text
qlib_data_fixed 大小: 510M
raw BaoStock 大小: 1.6G
all.txt 中的 instruments 数量: 5461
交易日历行数: 4012
日历范围: 2010-01-04 至 2026-07-15
```

已确认的 Qlib 字段：

```text
open, high, low, close, volume, vwap, factor, change
```

已确认的原始 CSV 字段：

```text
date, code, open, high, low, close, preclose, volume, amount,
turnover, tradestatus, pct_chg, is_st
```

主规则：

```text
使用 Qlib bin 文件快速读取 OHLCV。
原始 BaoStock CSV 只在离线构建时读取一次，用于 tradestatus/is_st 和可选诊断。
不要在 PyTorch Dataset 训练循环中读取原始 CSV。
```

## 3. 存储架构

不要存储：

```text
[num_samples, seq_len, feature_dim]
```

改为存储：

```text
raw_ohlcv:  [num_instruments, num_dates, raw_field_dim]
raw_state:  [num_instruments, num_dates, state_dim]
raw_trade:  [num_instruments, num_dates, 2]，BaoStock 原始 amount/turnover
features:   [num_instruments, num_dates, 42]，仅派生个股特征，物理上可分多个 instrument shard
model_ts:   读取时拼接 raw_ohlcv 的 8 列、raw_trade 的 2 列和 features 的 42 列，shape = [num_instruments, num_dates, 52]
market:     [num_dates, F_market]，按交易日聚合的市场横截面因子
labels:     [num_instruments, num_dates]
valid_mask: [num_instruments, num_dates]
index:      shard_id + local_instrument_idx + date_idx + label + split 元数据的行表
```

训练时读取窗口：

```python
x_raw = concat(raw_ohlcv[instrument_idx, date_idx - 59 : date_idx + 1, :], raw_trade[instrument_idx, date_idx - 59 : date_idx + 1, :])
x_derived = features_shard[local_instrument_idx, date_idx - 59 : date_idx + 1, :]
x_ts = concat(x_raw, x_derived, axis=-1)
x_market = market_cross_section[date_idx - 59 : date_idx + 1, :]
y = labels[instrument_idx, date_idx]
```

`x_market` 是已经由当日股票横截面聚合得到的市场状态因子，不是 `[股票数, 特征数]` 的额外股票矩阵，也不需要 `cross_section_mask`、固定证券槽位或 padding。这样每个日期-特征行只存一次，并让 OS page cache 处理重复读取。

## 4. 产物布局

所有研究产物应放在：

```text
.tmp/quantx-research/kronos-classification-signal-v1/
```

推荐布局：

```text
.tmp/quantx-research/kronos-classification-signal-v1/
  data/
    manifest.json
    instruments.parquet
    calendar.parquet
    raw_ohlcv_float32.mmap
    raw_ohlcv_meta.json
    raw_state_uint8.mmap
    raw_state_meta.json
    raw_trade_float32.mmap
    raw_trade_meta.json
    features_derived_v1_float16.part-000.mmap
    features_derived_v1_float16.part-001.mmap
    features_derived_v1_manifest.json
    market_cross_section_v1_float32.mmap
    market_cross_section_v1_meta.json
    labels_tb_h5_tp006_sl004_int8.mmap
    labels_tb_h5_tp006_sl004_meta.json
    label_diag_tb_h5_tp006_sl004.parquet
    valid_mask_v1_uint8.mmap
    valid_mask_v1_meta.json
    sample_index_v1.parquet
    folds/
      fold_2021.json
      fold_2022.json
      fold_2023.json
      fold_2024.json
      fold_2025.json
      forward_2026.json
    scalers/
      fold_2021_scaler.json
      fold_2022_scaler.json
      fold_2023_scaler.json
      fold_2024_scaler.json
      fold_2025_scaler.json
      forward_2026_scaler.json
  runs/
  reports/
```

一个逻辑 memmap 数据集不限制为单个物理文件。仅派生个股 `features` 应按连续 `instrument_idx` 范围分片；不应按单条样本拆文件，也不应为了“只有一个文件”强制写成超大文件。`features_derived_v1_manifest.json` 是逻辑数据集入口，记录每个 shard 的路径、`instrument_idx` 范围、`local_instrument_idx` 映射、shape、dtype、checksum 与 schema 版本。原始 10 个字段只保留在 `raw_ohlcv` / `raw_trade`，Dataset 读取时与派生字段拼接，避免重复物化和 `float16` 溢出。市场因子数组较小且按日期连续，保持单个紧凑 memmap 即可。

每个 `.mmap` 文件和 shard 都必须有配套元数据，包含：

```text
shape
dtype
轴名
字段名
provider_uri
raw_data_dir
calendar checksum 或 size/mtime
instrument checksum 或 size/mtime
created_at
feature_schema_hash 或 label_schema_hash
```

## 5. Memmap Shape 与预估大小

按 5461 个 instrument 和 4012 个日期估算：

| 产物 | Shape | Dtype | 预估大小 | 用途 |
| --- | ---: | --- | ---: | --- |
| `raw_ohlcv_float32.mmap` | `[5461, 4012, 8]` | `float32` | ~701 MB | 用于特征、标签和审计的原始行情字段 |
| `raw_state_uint8.mmap` | `[5461, 4012, 4]` | `uint8` | ~88 MB | 交易状态、ST 标记、原始行存在性 |
| `raw_trade_float32.mmap` | `[5461, 4012, 2]` | `float32` | ~175 MB | BaoStock 原始 `amount`、`turnover` |
| `features_derived_v1_float16.part-*.mmap` | 逻辑 `[5461, 4012, 42]` | `float16` | ~1.71 GB | 按 instrument 分片的派生个股特征 |
| `market_cross_section_v1_float32.mmap` | `[4012, F_market]` | `float32` | `4012 * F_market * 4 bytes` | 按日期聚合的市场横截面因子 |
| `labels_tb_h5_tp006_sl004_int8.mmap` | `[5461, 4012]` | `int8` | ~22 MB | 三分类标签 |
| `valid_mask_v1_uint8.mmap` | `[5461, 4012]` | `uint8` | ~22 MB | 最终样本有效性 |

派生个股特征存为 `float16`，因为它们主要是收益率、相对位置、z-score 和 rank。原始 `volume`、`amount` 等大数值字段保留在 `float32` raw memmap 中，禁止写入 `float16` shard。市场横截面因子存为 `float32`，以保留计数、比例和聚合统计的精度。训练 Dataset 中会在缩放和模型输入前转换为 `float32`。

原始 OHLCV 保持 `float32`，因为标签生成和审计不应使用降精度价格。

## 6. Instrument 与 Calendar 索引

`calendar.parquet` 列：

```text
date_idx: int32
date: string YYYY-MM-DD
year: int16
```

`instruments.parquet` 列：

```text
instrument_idx: int32
instrument: string, QuantX 格式，例如 SH600000
start_date: string
end_date: string
board: category/int8
is_index: bool
asset_bucket: int8
```

v1 的板块分类：

```text
SH60* -> mainboard_sh
SZ00* -> mainboard_sz
SZ30* -> chinext
SH68* -> star
SZ399* -> index，从股票训练中排除
other -> v1 排除
```

`asset_bucket` 应保持确定性：

```text
asset_bucket = stable_hash(instrument) % 5
```

它用于 asset-OOS 诊断。

### 6.1 Point-in-Time 股票池

所有个股 rank、市场横截面因子、样本有效性和标签可交易性必须使用同一份逐日股票池，不能用最终 `all.txt`、当前证券快照或已知的未来退市日期回填历史。

对每个交易日 `t`，`point_in_time_universe[t]` 的最小准入规则是：

```text
非指数且属于 v1 支持板块
截至 t 已有至少 80 个实际交易日历史
t 日存在原始行情和原始状态行
t 日 tradestatus == 1
t 日 is_st == 0
t 日 OHLCV 有效且 volume > 0
```

`universe_daily_v1.parquet` 必须记录 `date_idx`、`instrument_idx` 和每条准入/排除原因；其 schema hash、行数和 checksum 写入所有 rank/market/label manifest。历史行业、概念或情绪分组仅在其归属具有生效日期时加入；没有 point-in-time 归属的分组不得进入 v1。

## 7. Raw OHLCV Memmap

`raw_ohlcv_float32.mmap` 字段顺序：

```text
0 open
1 high
2 low
3 close
4 volume
5 vwap
6 change
7 factor
```

数据来源：

```text
${HOME}/git/quantization/quantx/data/qlib_data_fixed/features/<instrument_lower>/*.day.bin
```

读取策略：

```text
1. 读取一次全局 calendar。
2. 读取一次 instruments/all.txt。
3. 对每个选中的 instrument，读取全部 8 个字段 bin。
4. 使用 Qlib bin start offset 对齐到全局 date_idx。
5. 缺失值保持 NaN。
6. 定期 flush memmap。
```

优先复用现有 QuantX reader 逻辑直接读取 bin，避免在 builder 中反复初始化 Qlib。

## 8. Raw State Memmap

`raw_state_uint8.mmap` 字段顺序：

```text
0 tradestatus
1 is_st
2 has_raw_row
3 reserved
```

数据来源：

```text
${HOME}/git/quantization/quantx/data/raw/baostock/stocks/<instrument>.csv
```

读取策略：

```text
1. 每个原始 CSV 只扫描一次。
2. 读取 date、amount、turnover、tradestatus、is_st。
3. 将 BaoStock code/date 转换为 instrument_idx/date_idx。
4. 填充 raw_state 与 raw_trade。
5. 缺失原始行时设置 has_raw_row = 0。
```

`raw_trade_float32.mmap` 字段顺序：`0 amount`、`1 turnover`。它们用于基础特征的原始字段输入；训练循环不读取原始 CSV。

## 9. 过滤规则

基础价格有效性：

```text
open > 0
high > 0
low > 0
close > 0
vwap > 0
high >= low
所有必需字段都是有限值
```

流动性有效性：

```text
volume > 0
```

状态有效性：

```text
tradestatus == 1
is_st == 0
has_raw_row == 1
```

信号日 `t` 的历史有效性：

```text
t >= 59
[t-59, t] 内至少有 55 行有效价格
上市/交易历史长度 >= 80 个交易日
```

标签有效性：

```text
open(t+1) 有效
t+1 ... t+5 的 high/low 有效
```

最终样本有效性：

```text
valid_mask = stock_universe_valid
           & signal_day_price_valid
           & signal_day_state_valid
           & history_valid
           & label_valid
           & entry_feasible
           & label != -1
```

其中 `stock_universe_valid` 必须等于 `instrument_idx in point_in_time_universe[t]`。`entry_feasible` 和 `exit_feasible` 用于将标签路径与账户可成交性对齐：

```text
entry_feasible:
  t+1 存在原始状态行，tradestatus == 1，is_st == 0
  open(t+1) > 0，volume(t+1) > 0
  t+1 不是一字涨停或其他按正式账户规则不可买状态

exit_feasible(k):
  t+k 存在原始状态行，tradestatus == 1，is_st == 0
  t+k 存在有效可成交价格，且不是按正式账户规则不可卖状态
```

`entry_feasible == false` 的样本不进入三分类训练。v1 采用保守口径：若 first-hit barrier 所在日 `exit_feasible == false`，将样本标为 `label_execution_invalid` 并从训练集中剔除，不把该日当作可实现的止盈/止损，也不尝试以之后价格替代 barrier 日价格。具体不可买/不可卖规则必须复用正式账户 replay 的同一配置，并在 label metadata 中记录版本 hash。

## 10. 个股时序特征 Schema

`features_derived_v1_float16.part-*.mmap` 的单 shard shape：

```text
[num_instruments_in_shard, num_dates, 42]
```

逻辑 shape 仍为：

```text
[num_instruments, num_dates, 42]
```

模型输入的基础日频原始字段，10 维：

```text
0  open
1  high
2  low
3  close
4  volume
5  vwap
6  change
7  factor
8  amount
9  turnover
```

这 10 列直接从 `raw_ohlcv_float32.mmap` 与 `raw_trade_float32.mmap` 读取，不做重复写入、对数、收益率、比值、相对位置或直接乘积等变换。

派生时序特征，32 维：

前 8 列是直接提供给模型的单日相对形态和量能状态：

```text
0 ret_close_1d      = close_t / close_{t-1} - 1
1 ret_open_1d       = open_t / close_{t-1} - 1
2 ret_intraday      = close_t / open_t - 1
3 high_open          = high_t / open_t - 1
4 low_open           = low_t / open_t - 1
5 range_hl           = high_t / low_t - 1
6 close_position     = (close_t - low_t) / max(high_t - low_t, eps)
7 amount_zscore_20   = (amount_t - mean(amount[t-19:t])) / max(std(amount[t-19:t]), eps)
```

后 24 列由 3 个较长窗口的路径统计构成。对 `w in {5, 20, 60}`，每个窗口依次写入 8 列：

```text
rolling_return_w       = close_t / close_{t-w} - 1
rolling_volatility_w   = std(ret_close_1d[t-w+1:t])
rolling_max_drawdown_w = min(close_u / max(close[t-w+1:u]) - 1), u in [t-w+1, t]
distance_to_high_w     = close_t / max(close[t-w+1:t]) - 1
distance_to_low_w      = close_t / min(close[t-w+1:t]) - 1
volume_zscore_w        = (volume_t - mean(volume[t-w+1:t])) / max(std(volume[t-w+1:t]), eps)
amount_zscore_w        = (amount_t - mean(amount[t-w+1:t])) / max(std(amount[t-w+1:t]), eps)
range_zscore_w         = (high_t / low_t - 1 - mean(range[t-w+1:t])) / max(std(range[t-w+1:t]), eps)
```

因此派生字段的固定索引为：

```text
0..7   单日相对形态/量能
8..15  w = 5 的路径特征
16..23 w = 20 的路径特征
24..31 w = 60 的路径特征
```

截面 rank 派生特征，10 维：

```text
32 ret_close_1d_rank        = rank_pct(ret_close_1d)
33 rolling_return_5_rank   = rank_pct(rolling_return_5)
34 rolling_return_20_rank  = rank_pct(rolling_return_20)
35 rolling_return_60_rank  = rank_pct(rolling_return_60)
36 volatility_20_rank      = rank_pct(rolling_volatility_20)
37 amount_rank             = rank_pct(amount_t)
38 distance_to_high_20_rank = rank_pct(distance_to_high_20)
39 distance_to_high_60_rank = rank_pct(distance_to_high_60)
40 distance_to_low_20_rank  = rank_pct(distance_to_low_20)
41 range_hl_rank            = rank_pct(range_hl)
```

rank 使用 `[0, 1]` 区间内的百分位排名，并严格按 `point_in_time_universe[t]` 计算。

## 11. 市场横截面因子 Schema

`market_cross_section_v1_float32.mmap` 是按交易日连续存储的聚合市场状态数组：

```text
shape = [num_dates, F_market]
axes = [date_idx, market_feature]
```

其中每行由 `date_idx` 唯一定位，并对应 `calendar.parquet` 的同一交易日；它不是按股票展开的矩阵。`F_market` 在实现前由可用、无泄漏的因子清单固定，并写入 `market_cross_section_v1_meta.json`。

v1 市场因子必须在实现前逐列固定到 `market_cross_section_v1_meta.json`，包括列名、公式、输入字段、股票池、缺失处理和 schema hash。首版只使用 `point_in_time_universe[t]` 上可由当日行情直接聚合的因子，不使用没有历史生效日期的行业/概念归属、连板/高位股标签或专有活跃市值真值。

推荐的 v1 固定 schema 为 `F_market = 20`：

```text
0  up_ratio                 = count(ret_close_1d > 0) / N
1  down_ratio               = count(ret_close_1d < 0) / N
2  flat_ratio               = count(ret_close_1d == 0) / N
3  ret_close_1d_mean
4  ret_close_1d_median
5  ret_close_1d_std

6  ret_lt_minus_10_ratio    = count(ret_close_1d < -0.10) / N
7  ret_minus_10_to_minus_5_ratio = count(-0.10 <= ret_close_1d < -0.05) / N
8  ret_minus_5_to_minus_1_ratio  = count(-0.05 <= ret_close_1d < -0.01) / N
9  ret_minus_1_to_0_ratio         = count(-0.01 <= ret_close_1d < 0) / N
10 ret_0_to_1_ratio               = count(0 < ret_close_1d <= 0.01) / N
11 ret_1_to_5_ratio               = count(0.01 < ret_close_1d <= 0.05) / N
12 ret_5_to_10_ratio              = count(0.05 < ret_close_1d <= 0.10) / N
13 ret_gt_10_ratio                = count(ret_close_1d > 0.10) / N

14 amount_sum                = sum(amount_t)
15 amount_zscore_20          = (amount_sum_t - mean(amount_sum[t-19:t])) / max(std(amount_sum[t-19:t]), eps)
16 volume_sum                = sum(volume_t)
17 volume_zscore_20          = (volume_sum_t - mean(volume_sum[t-19:t])) / max(std(volume_sum[t-19:t]), eps)
18 amount_top10_share        = sum(amount_t of top 10% by amount) / amount_sum
19 ret_p90_minus_p10         = percentile(ret_close_1d, 0.90) - percentile(ret_close_1d, 0.10)
```

其中收益区间比例是互斥直方图，而不是重叠的阈值计数；列 6..13 与 `flat_ratio` 之和应为 1。它同时保留每日涨幅超过 `1%`、`5%`、`10%` 与各个下跌区间的横截面信息，例如 `ret_1_to_5_ratio + ret_5_to_10_ratio + ret_gt_10_ratio` 表示涨幅超过 `1%` 的比例。每个交易日必须断言该恒等式在容差内成立，并把 `N`、各区间计数和缺失排除数写入诊断报告。

原始 `amount_sum` 和 `volume_sum` 表示市场绝对活跃规模；对应的 `zscore` 和 `amount_top10_share` 表示相对活跃状态与成交集中度。首版不把 `amount_sum` 称为专有活跃市值指标，也不使用涨跌停数量替代收益区间比例，因为不同板块的涨跌停规则和复权口径需要独立审计。

市场因子与个股特征一样只可使用 `<= t` 的数据。若在 `t` 日收盘后生成 T+1 交易信号，可使用完整的 `t` 日聚合；不得使用 `t+1` 或后续市场数据。

## 12. 特征构建算法

个股与市场特征分三遍构建，避免把整个面板放进内存。

第 1 遍，按 instrument：

```text
1. 读取 raw_ohlcv[instrument_idx, :, :]。
2. 计算基础特征。
3. 只在当前 instrument 上用 pandas/numpy 计算滚动特征。
4. 按固定索引写入 8 个单日相对特征，以及 `w in {5, 20, 60}` 的 24 个路径特征。
5. 校验派生值均有限且绝对值不超过 `float16` 最大有限值 `65504`；否则记录原因并拒绝写入。
6. 将 32 个派生时序特征写入该 instrument shard 的列 0..31。
```

第 2 遍，按日期：

```text
1. 顺序扫描每个 feature shard 的 `[:, date_idx, selected_source_columns]`。
2. 拼接或流式汇总当前日期的有效股票池值，计算 rank_pct。
3. 校验 rank 值均位于 `[0, 1]` 且有限。
4. 按 shard 中的 local row 将 10 个 rank 特征写入对应 shard 的列 32..41。
```

第 3 遍，按日期构建市场横截面因子：

```text
1. 读取当日有效股票池上的原始行情、个股因子和规则状态。
2. 对固定的市场因子 schema 做 count/ratio/mean/median/dispersion/group 聚合。
3. 将 market_cross_section[date_idx, :] 写入单个连续 memmap。
4. 使用仅依赖 <= t 的滚动统计生成需要历史窗口的市场因子。
```

builder 不能使用 `t` 之后的行来计算 `t` 时点的个股或市场特征。

## 13. 标签设计

主标签文件：

```text
labels_tb_h5_tp006_sl004_int8.mmap
```

Shape：

```text
[num_instruments, num_dates]
```

编码：

```text
-1 = 无效 / 无标签
 0 = SELL
 1 = HOLD
 2 = BUY
```

Triple Barrier 参数：

```text
horizon = 5
entry = open(t+1)
take_profit = 0.06
stop_loss = 0.04
same_day_tie = HOLD
label_end_date = t+5
```

规则：

```text
if not entry_feasible:
    label = -1
    stop

for k in 1..5:
    candidate_label = unset
    up = high(t+k) / open(t+1) - 1
    down = low(t+k) / open(t+1) - 1

    if up >= 0.06 and down <= -0.04:
        candidate_label = HOLD

    elif down <= -0.04:
        candidate_label = SELL

    elif up >= 0.06:
        candidate_label = BUY

    if candidate_label is set:
        if exit_feasible(k):
            label = candidate_label
            first_hit_day = k
            stop
        else:
            label = -1
            label_execution_invalid = true
            stop

if 全部 k 未触发 barrier 且 label_execution_invalid == false:
    label = HOLD
```

该标签只把未来路径作为监督信号使用，绝不能 join 到 feature memmap 中。

## 14. 标签诊断

`label_diag_tb_h5_tp006_sl004.parquet` 应为每个有效样本存一行：

```text
instrument_idx
date_idx
instrument
signal_date
entry_date
label_end_date
label
first_hit_day
entry_feasible
first_hit_exit_feasible
label_execution_invalid
label_execution_rule_hash
max_up_5d
max_down_5d
future_open_to_close_5d
future_open_to_open_5d
tie_hit
board
asset_bucket
```

训练前必须完成的诊断：

```text
按年份统计类别分布
按 board 统计类别分布
BUY/SELL/HOLD 路径统计
同日 tie 数量
无效样本原因
TopK oracle label 合理性检查
entry/exit 不可成交比例及按 label 的分布
label_execution_invalid 原因与数量
```

如果 BUY 或 SELL 极度稀疏、集中在某一年，或者在正式账户约束下大多不可交易，应停止模型训练。

## 15. 样本索引

`sample_index_v1.parquet` 只存窄表元数据，不存序列窗口。

列：

```text
row_id: int64
instrument_idx: int32
shard_id: int16
local_instrument_idx: int32
date_idx: int32
instrument: string
signal_date: string YYYY-MM-DD
label_end_date: string YYYY-MM-DD
year: int16
label: int8
entry_feasible: bool
label_execution_invalid: bool
board: category or int8
asset_bucket: int8
```

仅用于评估的可选列：

```text
future_open_to_close_5d
future_open_to_open_5d
max_up_5d
max_down_5d
```

PyTorch Dataset 必须忽略仅用于评估的列。

Dataset lookup：

```python
row = sample_index.iloc[i]
inst = int(row.instrument_idx)
shard = self.get_feature_shard(int(row.shard_id))
local_inst = int(row.local_instrument_idx)
t = int(row.date_idx)
x_raw = concat(raw_ohlcv[inst, t - 59 : t + 1, :], raw_trade[inst, t - 59 : t + 1, :])
x_derived = shard[local_inst, t - 59 : t + 1, :]
x_ts = concat(x_raw, x_derived, axis=-1)
x_market = market_cross_section[t - 59 : t + 1, :]
y = int(row.label)
```

## 16. Fold 设计

主时间 OOS fold：

| Fold | 训练 | 验证 | 预测 |
| --- | --- | --- | --- |
| fold-2021 | 2016-2019 | 2020 | 2021 |
| fold-2022 | 2016-2020 | 2021 | 2022 |
| fold-2023 | 2016-2021 | 2022 | 2023 |
| fold-2024 | 2016-2022 | 2023 | 2024 |
| fold-2025 | 2016-2023 | 2024 | 2025 |
| forward-2026 | 2016-2024 | 2025 | 2026-01-01 至 2026-07-15 |

split 以 `label_end_date` 而不是 `signal_date` 为唯一边界。设 `horizon = 5`，则每个 fold 都必须执行 purge：

```text
train:      signal_date 在训练区间，且 label_end_date < validation_start
validation: signal_date 在验证区间，且 label_end_date < prediction_start
prediction: signal_date 在预测区间，且 label_end_date <= prediction_end
```

因此每个相邻区间的末尾最多 5 个交易日不进入前一 split。`fold_*.json` 必须显式记录 `train_end_signal_date`、`validation_end_signal_date`、`prediction_end_signal_date`、`horizon`、purged row 数和每个 split 的 `label_end_date` 最大值；任何一个不等式不成立即拒绝训练或评估。

验证区间按时间再分为 `validation_select` 和 `validation_calibration`，中间同样 purge `horizon` 个交易日：前者只用于 early stopping、模型和分数选择；后者只用于最终概率校准与阈值审计，禁止参与 early stopping、超参数选择或 score 选择。预测集保持完全自然分布且不参与任何选择。只有训练集可以对 HOLD 做下采样。

Asset-OOS 诊断：

```text
heldout_assets = asset_bucket == 4
```

每个预测年份都应同时报告全部预测资产和仅 heldout assets 上的指标。

## 17. Fold 本地 Scaler

不要创建 fold 专属的 feature 或 market memmap 副本。

只保存全局因果的个股 feature shard 和市场因子 memmap，并在 Dataset 中分别应用 fold-local scaler。

Scaler 产物：

```text
scalers/fold_2021_scaler.json
```

字段：

```text
raw_feature_mean[10]
raw_feature_std[10]
raw_clip_low[10]
raw_clip_high[10]
derived_feature_mean[42]
derived_feature_std[42]
derived_clip_low[42]
derived_clip_high[42]
market_feature_mean[F_market]
market_feature_std[F_market]
market_clip_low[F_market]
market_clip_high[F_market]
fit_sample_count
fit_token_count
fit_start
fit_end
fit_seed
```

拟合规则：

```text
1. 只使用训练 split。
2. 从训练窗口中最多采样 2,000,000 个个股 time-token 行，并从训练日期采样市场行。
3. 对 raw、derived 和 market 三组数组分别计算 0.5% 和 99.5% clipping 阈值。
4. clipping 后分别计算 mean/std；市场因子不因每只股票重复计权。
5. 不要在 validation/prediction 行或其市场日期上拟合。
```

原始字段落盘保持原值；scaler 仅定义模型读取时的数值尺度。派生字段已显式提供单日相对收益、日内形态、相对量能和 5/20/60 日路径统计，模型不必从绝对价格或成交额量级中反推这些形态。除全局 fold scaler 外，必须实现并比较 `causal_instance_scaler`：每只股票仅用当前窗口内 `<= t` 的统计量缩放 `open/high/low/close/volume/vwap/amount/turnover`，`change` 和 `factor` 按各自训练 fold scaler 处理。最终输入口径只能根据验证期消融选择，选择后冻结至所有 prediction/forward 区间。

必须完成以下原始/相对输入消融，三组使用相同 fold、seed、sampler、模型容量和训练预算：

```text
A. raw + relative：10 个原始字段 + 42 个派生相对字段，使用选定 scaler
B. relative only：仅 42 个派生相对字段
C. raw + relative + causal_instance_scaler：A 的字段集合，原始 10 列使用个股因果窗口 scaler
```

比较 `validation_select` 的自然分布分类指标、TopK 收益、SELL 污染、校准指标与 asset-OOS 指标。若 B 与 A 相当或更好，默认移除原始绝对量级通道；若 A 或 C 在多个 fold/seed 稳定更好，才保留原始字段。不能只因训练集指标提升而保留绝对价格、成交额或流动性输入。

## 18. 训练 Dataset 与 Memmap 访问

Dataset 初始化：

```python
feature_manifest = load_feature_manifest(features_manifest_path)
raw_ohlcv = np.memmap(raw_ohlcv_path, dtype=np.float32, mode="r", shape=(num_instruments, num_dates, 8))
raw_trade = np.memmap(raw_trade_path, dtype=np.float32, mode="r", shape=(num_instruments, num_dates, 2))

market_cross_section = np.memmap(
    market_path,
    dtype=np.float32,
    mode="r",
    shape=(num_dates, F_market),
)

labels = np.memmap(
    labels_path,
    dtype=np.int8,
    mode="r",
    shape=(num_instruments, num_dates),
)
```

`__getitem__`：

```python
shard_id = self.shard_id[i]
local_inst = self.local_instrument_idx[i]
inst = self.instrument_idx[i]
t = self.date_idx[i]
x_raw = np.concatenate((
    self.raw_ohlcv[inst, t - 59 : t + 1, :],
    self.raw_trade[inst, t - 59 : t + 1, :],
), axis=-1)
x_derived = self.get_feature_shard(shard_id)[local_inst, t - 59 : t + 1, :]
x_market = self.market_cross_section[t - 59 : t + 1, :]
x_raw = np.asarray(x_raw, dtype=np.float32)
x_derived = np.asarray(x_derived, dtype=np.float32)
x_market = np.asarray(x_market, dtype=np.float32)
x_raw = (np.clip(x_raw, self.raw_clip_low, self.raw_clip_high) - self.raw_mean) / self.raw_std
x_derived = (np.clip(x_derived, self.derived_clip_low, self.derived_clip_high) - self.derived_mean) / self.derived_std
x_ts = np.concatenate((x_raw, x_derived), axis=-1)
x_market = (np.clip(x_market, self.market_clip_low, self.market_clip_high) - self.market_mean) / self.market_std
y = int(self.label[i])
return x_ts, x_market, y
```

`load_feature_manifest` 只能读取 manifest，不能打开任何 shard。每个 Dataset worker 的 `get_feature_shard(shard_id)` 使用按需 `np.memmap` 的 LRU 缓存，`max_open_shards` 作为配置项且初始值为 `2`；超过上限时关闭最久未使用的映射。raw 和 market 是单一紧凑 memmap，可在 worker 内打开一次。这样避免每条样本重复 `open`，也避免把全部分片映射进 RAM。Dataset worker 必须在 worker 初始化后各自打开映射，不得把父进程的打开句柄经 fork 继承为共享可写状态。

## 19. I/O 友好的 Sampler

纯随机采样会产生大量小随机读。应使用 block sampler，在平衡类别的同时保留部分局部性。

推荐 `BalancedBlockSampler`：

```text
1. 从 train sample_index 构建 BUY、SELL、HOLD 的 row_id 池。
2. BUY 和 SELL 保留全部或接近全部样本。
3. 按 year + board + asset_bucket 对 HOLD 下采样。
4. 合并当前 epoch 选中的 row_id。
5. 按 (shard_id, local_instrument_idx, date_idx) 对选中行排序。
6. 将排序后的行拆成 8192 大小的 blocks。
7. 打乱 block 顺序。
8. block 内保持顺序。
```

训练配置：

```text
batch_size = 512
block_size = 8192
num_workers = 0，或先从 2 开始
pin_memory = 仅 GPU 训练时为 true
```

这样能在 block 层面提供足够随机性，同时减少磁盘 seek、跨 shard 切换和 page cache miss。`block_size` 与 shard 的 instrument 范围必须通过实测吞吐确定，不预先假设单一文件必然更快。训练日志必须报告每 epoch 的 shard miss、LRU eviction、样本读取 P50/P95 和 batch 吞吐。

## 20. 训练类别平衡

训练集采样：

```text
BUY: keep all
SELL: keep all
HOLD: stratified downsample
```

初始目标：

```text
num_hold_sampled = min(num_hold_available, 2 * (num_buy + num_sell))
```

分层键：

```text
year
board
asset_bucket
```

验证和预测：

```text
不下采样。
报告自然类别分布。
`validation_calibration` 保持自然分布，不参与 early stopping 或 score 选择。
```

## 21. 模型输入输出

输入：

```text
x_ts shape = [batch, 60, 52]
x_market shape = [batch, 60, F_market]
```

市场因子按与个股窗口相同的 60 个交易日输入，而不是只取预测日单点或把市场状态广播成额外 token。每个 token 对应同一个交易日的个股路径和市场状态。

输出：

```text
logits shape = [batch, 3]
probs = softmax(logits)
```

类别顺序：

```text
0 = SELL
1 = HOLD
2 = BUY
```

主排序分数：

```text
score = P(BUY) - 0.5 * P(SELL)
```

分数消融：

```text
score_buy = P(BUY)
score_edge_1 = P(BUY) - P(SELL)
score_logit = logit_buy - 0.5 * logit_sell
```

分数消融必须在 forward evaluation 前固定。不能在看过 2026 后再挑选最优分数。

训练采样或 focal loss 改变了类别先验，`probs` 必须表示在 `validation_calibration` 上冻结校准后的概率：

```text
训练结束后：仅用 validation_calibration 拟合 multiclass temperature scaling 或 prior correction
预测阶段：原始 logits -> 已冻结校准器 -> calibrated_probs
评估与 score：只使用 calibrated_probs
```

校准器、拟合样本数、calibration split 边界、拟合前后 ECE/NLL/Brier score 和校准器 hash 必须随 checkpoint 导出。不得用 prediction/forward 数据再次拟合。

## 22. 基础模型结构

v1 主模型是连续输入的小型 Transformer 分类器。它采用第 08 号 diffusion 方案的“个股与市场同日 token 对齐融合”，并借鉴本地 `Kronos/model` 的 `RMSNorm + RoPE 多头注意力 + SwiGLU FFN + PreNorm 残差` block；不使用 Kronos 的 BSQ/DCT-BPE 离散 tokenizer、hierarchical token embedding、DualHead 或自回归 next-token 预测目标。

基础配置：

```text
seq_len = 60
ts_feature_dim = 52
market_feature_dim = F_market
d_model = 192
n_heads = 6
head_dim = 32
n_layers = 4
ffn_dim = 384
attn_dropout = 0.05
resid_dropout = 0.05
ffn_dropout = 0.05
head_hidden = 256
num_classes = 3
```

架构：

```text
Individual input x_ts [B, 60, 52]
  -> Linear(52, 128)
  -> RMSNorm(128)
Market input x_market [B, 60, F_market]
  -> Linear(F_market, 64)
  -> RMSNorm(64)
  -> concat by same date token [B, 60, 192]
  -> Linear(192, 192)
  -> RMSNorm(192)
  -> KronosStyleEncoderBlock x 4
       residual + RoPE MultiHeadSelfAttention(PreRMSNorm, d_model=192, heads=6, non-causal, attn/resid dropout=0.05)
       residual + SwiGLU FFN(PreRMSNorm, Linear(192, 384) x 2 -> SiLU gate -> Linear(384, 192), dropout=0.05)
  -> pooling
       last_token [B, 192]
       mean_pool  [B, 192]
       concat     [B, 384]
  -> RMSNorm(384)
  -> Linear(384, 256)
  -> SiLU
  -> Dropout(0.05)
  -> Linear(256, 3)
```

近似参数量：

```text
input and fusion projections: ~40k
each transformer layer: ~300k
4 encoder layers: ~1.2M
classification head: ~100k
total: ~1.3M-1.5M
float32 checkpoint: ~5-7 MB
```

RoPE 提供相对时序位置，不额外加入 learned positional embedding；三分类窗口全部是 `<= t` 的已知历史，因此 encoder self-attention 可使用 non-causal mask。若以后加入窗口内缺失 token，才传入 attention padding mask。该结构约为低百万级参数，仍适合 QuantX 研究脚本。

首版只保留上述拼接投影，不增加 cross-attention、离散 tokenizer 或全市场个股矩阵分支。Kronos 风格 block 以研究脚本内的最小实现维护，需和本地 `Kronos/model/module.py` 的 RMSNorm、RoPE attention、SwiGLU 公式做单元级数值对照；不直接依赖其带路径副作用的训练代码。融合方式、block 版本和市场因子 schema 必须在训练前固定，并在 run 配置中记录。

## 23. 小型调试模型

在基础运行前，先用更小的模型验证管线：

```text
d_model = 96
n_heads = 4
n_layers = 3
ffn_dim = 192
head_hidden = 128
params ~= 0.3M
```

小模型不是最终复现目标。它用于快速测试 memmap 读取吞吐、标签连线、loss 稳定性、验证报告生成和预测导出 shape。

## 24. 优化

训练目标：

```text
focal loss，仅由采样处理类别先验
gamma = 2.0
alpha = 1.0 for all classes
```

训练只保留一种显式类别频率校正：`BalancedBlockSampler` 的 HOLD 下采样。focal loss 只降低易分类样本的梯度，不再叠加 class-balanced `alpha` 或 weighted cross entropy。`CrossEntropy + 同一 sampler`、`focal loss + 同一 sampler` 是必须报告的损失消融；最终模型只按 `validation_select` 的预先定义指标选择一次。

优化器：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
```

调度：

```text
前 5% total steps 使用 linear warmup
之后使用 cosine decay
```

训练：

```text
batch_size = 512
epochs = 20
early_stop_patience = 5
seeds = 7, 11, 19
```

Early stopping 不应只看 loss。优先级：

```text
1. validation 上的 BUY/SELL 平均 F1
2. validation 上的 Top20 未来收益
3. validation Top20 中的 SELL 污染
4. macro F1
5. loss
```

## 25. 验证报告

每个 fold 和 seed 都必须产出分类、概率和排序报告。

分类层：

```text
类别分布
混淆矩阵
按类别统计 precision/recall/F1
macro F1
BUY/SELL 平均 F1
按年份报告
按 board 报告
asset-OOS 报告
按市场 regime 报告
```

概率层：

```text
expected calibration error
NLL
multiclass Brier score
校准分桶
按 P(BUY) 十分位统计未来收益
按 P(SELL) 十分位统计未来回撤
按 score 十分位统计未来收益
```

排序层：

```text
Top10/Top15/Top20 的 future_open_to_close_5d
Top10/Top15/Top20 的 BUY 命中率
Top10/Top15/Top20 的 SELL 污染
按日期统计 RankIC
按日期统计 top-bottom spread
按年份统计 TopK 指标
RankIC 和 spread 的 bootstrap CI
```

市场输入还必须单独诊断：

```text
每个 market feature 的缺失率、极值、训练期与 OOS 分布漂移
按 market regime 的类别分布、F1、TopK 收益和 SELL 污染
去除全部 x_market 的消融，与完整模型使用相同 split、seed 和采样预算
市场因子生成时间、股票池定义和 point-in-time 元数据审计
每日 universe 行数、准入/排除原因、rank 覆盖率与 market 聚合覆盖率
split 的 label_end_date 边界与 purge 行数审计
raw float32 与 derived float16 的 NaN/inf/overflow 检查
raw + relative、relative only、causal_instance_scaler 三组的统一预算消融
```

正式账户 replay 不属于第一版数据管线。只有排序层指标通过后才开始。

## 26. PredictionStore 对接

如果排序层指标通过，按现有 `PredictionStore` 契约导出 QuantX 预测。

记录字段：

```text
signal_time = signal_date 15:00 Asia/Shanghai AFTER_CLOSE
instrument
score
prediction_horizon = 5
artifact_id = kronos_cls_memmap_v1
fold_id
feature_schema_hash
evaluation_tier = development_oos or sealed_holdout
rank
uncertainty = entropy(probs) or 1 - max(probs)
```

不要默认导出全部 instrument 给账户 replay。每个日期导出足够稳定选股的行数，例如按 score 取 Top200；同时为诊断单独保留完整 prediction parquet。

## 27. 最小脚本计划

第一版实现保持为 research scripts。在管线证明有用之前，不要加入框架级抽象。

推荐脚本：

```text
.tmp/quantx-research/kronos-classification-signal-v1/
  build_kronos_memmap_dataset_v1.py
  diagnose_kronos_memmap_dataset_v1.py
  train_kronos_baseline_models_v1.py
  train_kronos_transformer_classifier_v1.py
  evaluate_kronos_classifier_v1.py
  export_kronos_prediction_store_v1.py
```

职责：

| 脚本 | 职责 |
| --- | --- |
| `build_kronos_memmap_dataset_v1.py` | 构建 point-in-time universe、raw memmaps、派生 feature shards、market 因子、可成交 labels/masks/index |
| `diagnose_kronos_memmap_dataset_v1.py` | 报告数据覆盖、分片完整性、数值溢出、split purge、市场因子可见性、可交易性、标签分布、样本数量 |
| `train_kronos_baseline_models_v1.py` | 基于非窗口摘要特征做 Logistic/LightGBM/ExtraTrees 检查 |
| `train_kronos_transformer_classifier_v1.py` | 使用 memmap Dataset 训练小型/基础 Transformer |
| `evaluate_kronos_classifier_v1.py` | 分类、校准、排序、seed/fold 聚合 |
| `export_kronos_prediction_store_v1.py` | 将固定 score 输出转换为 QuantX PredictionStore |

## 28. 实施顺序

不要从模型开始。先构建并验证数据管线。

顺序：

```text
1. 路径审计与 manifest 生成。
2. instrument/calendar 索引。
3. 构建 raw_ohlcv_float32.mmap。
4. 构建 raw_state_uint8.mmap。
5. 构建 `point_in_time_universe[t]` 与每日准入/排除审计；未通过审计不得继续。
6. 构建 `features_derived_v1_float16.part-*.mmap` 的 32 个时序派生列和 shard manifest，并执行 overflow 检查。
7. 截面 rank 特征遍历，生成派生列 32..41；构建 `market_cross_section_v1_float32.mmap`。
8. 市场因子的可见性、point-in-time、缺失率和分布诊断。
9. 基于正式账户执行规则构建 label memmap、entry/exit 可成交审计与标签诊断。
10. 按 `label_end_date` 执行 split purge，构建带 `shard_id` / `local_instrument_idx` 的 sample index。
11. 拟合 raw、derived、market scaler；切出 `validation_select` / `validation_calibration`。
12. Dataset 的 LRU shard 缓存和吞吐 smoke test。
13. 标签分布、市场 regime 与 oracle 合理性报告。
14. 小型 Transformer 单 fold 单 seed 运行，完成 CrossEntropy/focal、raw + relative / relative only / causal instance scaler 消融。
15. 基础 Transformer 全 fold 多 seed 运行，完成去除 `x_market` 的消融和冻结概率校准。
16. 排序层评估。
17. 只有排序层通过后，才进行 PredictionStore 导出与正式账户 replay。
```

## 29. 停止条件

以下情况应在模型训练前停止：

```text
有效样本数量过小
BUY 或 SELL 极度稀疏
BUY/SELL 分布集中在某一年
标签诊断显示大多数 BUY 路径不可交易
memmap Dataset 吞吐过慢，无法支撑实际训练
raw/derived memmap 出现 NaN、inf 或 float16 overflow
point-in-time universe、rank 或 market 聚合覆盖率审计不通过
市场因子使用了无法在信号日证明可见的数据，或 point-in-time 审计不通过
```

以下情况应在 baseline model 后停止：

```text
简单模型在自然分布 time-OOS validation 上不能超过随机水平
TopK label metrics 显示没有可用排序信号
```

以下情况应在 Transformer 后停止：

```text
classification metrics 提升，但 TopK metrics 没有提升
2022-2025 表现良好，但 2026 明显反转
multi-seed 方差过大
base Transformer 不能超过 simple baseline
完整模型不能稳定超过去除 x_market 的消融模型
校准器在 validation_calibration 上未改善或显著恶化 NLL/ECE
计入成本后，formal account replay 差于 path/Exp40 baseline
```

## 30. 关键工程原则

该实现应保持小而可审计：

```text
不在磁盘上保存重复的 sliding-window dataset。
个股特征允许按 shard 分片，市场因子保持按日期连续的小型数组。
原始字段只保留在 float32 raw memmap，派生字段才允许使用 float16 shard。
原始字段与相对形态字段同时进入候选输入；是否保留原始绝对量级必须由 OOS 消融决定。
训练循环中不读取原始 CSV。
不为每个 fold 复制 features。
不在 validation 或 prediction 周期上拟合 scaler。
模型输入文件中不包含 label 列。
不把按日期聚合的市场因子误建成全市场个股矩阵。
以 label_end_date 执行 split purge，禁止标签跨入后续 split。
训练标签的进出场可成交性与正式账户 replay 使用同一规则版本。
不在检查 2026 后选择 score。
不只凭 classification F1 宣称正式账户有效。
```

第一成功标准不是模型精度，而是一个正确、快速、可审计的 memmap 数据管线，并且标签诊断稳定。只有在这之后，Kronos 风格分类器才应被视为严肃的复现候选。
