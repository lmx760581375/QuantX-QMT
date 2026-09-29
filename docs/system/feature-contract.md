# 特征合同：每个输入从哪里来

## 样本单位与形状

一个 Reward 样本对应一只股票在一个信号日 $T$：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `x_ts` | `60 × 52` | 个股逐日输入。 |
| `x_market` | `60 × 20` | 同一 60 日窗口内的全市场逐日状态。 |
| `x_candidate` | `6` | 该股票在候选池中的相对位置与分数上下文。 |
| `y_abs` | `30` | 从 $T+1$ 收盘起的未来累计 close-to-close 路径。 |

`52 = 8` 个 Qlib 原始字段 `+ 2` 个原始交易字段 `+ 32` 个个股派生字段 `+ 10` 个日内横截面 rank 字段。

训练正式使用 `input_mode: raw_relative`，因此 52 个个股字段全部进入 AE。

## 一、10 个原始个股字段

| 组 | 字段 | 来源 | 说明 |
| --- | --- | --- | --- |
| Qlib 日线 | `open, high, low, close, volume, vwap, change, factor` | Qlib provider | 与信号日一致的已落地日频字段；`factor` 为 provider 中的复权因子字段。 |
| 原始交易 | `amount, turnover` | BaoStock 同源 CSV | 成交额和换手率；用于流动性、横截面排名和市场状态。 |

交易状态不直接作为连续模型特征，但 `tradestatus`、`is_st`、原始行存在标记会进入样本可用性过滤，避免把不可交易样本当作训练或交易候选。

## 二、32 个个股派生时序字段

以下字段逐股票、逐日计算。所有 rolling 窗口都只使用当日及其历史。

| 类别 | 字段 | 定义 |
| --- | --- | --- |
| 单日收益与形态 | `ret_close_1d` | $close_t / close_{t-1} - 1$。 |
|  | `ret_open_1d` | $open_t / close_{t-1} - 1$。 |
|  | `ret_intraday` | $close_t / open_t - 1$。 |
|  | `high_open` | $high_t / open_t - 1$。 |
|  | `low_open` | $low_t / open_t - 1$。 |
|  | `range_hl` | $high_t / low_t - 1$。 |
|  | `close_position` | $(close_t-low_t)/(high_t-low_t)$。 |
| 流动性异常 | `amount_zscore_20` | 当日成交额相对 20 日滚动均值/标准差的 z-score。 |
| 5 日窗口 | `rolling_return_5` | $close_t / close_{t-5} - 1$。 |
|  | `rolling_volatility_5` | 5 日 `ret_close_1d` 标准差。 |
|  | `rolling_max_drawdown_5` | 5 日 close 路径的最大回撤。 |
|  | `distance_to_high_5` | $close_t / max(high_{t-4:t}) - 1$。 |
|  | `distance_to_low_5` | $close_t / min(low_{t-4:t}) - 1$。 |
|  | `volume_zscore_5` | 成交量 5 日滚动 z-score。 |
|  | `amount_zscore_5` | 成交额 5 日滚动 z-score。 |
|  | `range_zscore_5` | 振幅 5 日滚动 z-score。 |
| 20 日窗口 | `rolling_return_20`、`rolling_volatility_20`、`rolling_max_drawdown_20`、`distance_to_high_20`、`distance_to_low_20`、`volume_zscore_20`、`amount_zscore_20_window`、`range_zscore_20` | 与 5 日定义相同，只把窗口替换为 20。 |
| 60 日窗口 | `rolling_return_60`、`rolling_volatility_60`、`rolling_max_drawdown_60`、`distance_to_high_60`、`distance_to_low_60`、`volume_zscore_60`、`amount_zscore_60`、`range_zscore_60` | 与 5 日定义相同，只把窗口替换为 60。 |

这里 `amount_zscore_20` 和 `amount_zscore_20_window` 名称相近但来源不同：前者是单独保留的流动性异常字段，后者属于 20 日窗口特征组。

## 三、10 个日内横截面 rank 字段

每个交易日只在当天有效股票池内计算稳定百分位 rank，数值越大表示该字段的原值越大。

| 字段 | 被排名的原始量 |
| --- | --- |
| `ret_close_1d_rank` | `ret_close_1d` |
| `rolling_return_5_rank` | `rolling_return_5` |
| `rolling_return_20_rank` | `rolling_return_20` |
| `rolling_return_60_rank` | `rolling_return_60` |
| `volatility_20_rank` | `rolling_volatility_20` |
| `amount_rank` | 当日 `amount` |
| `distance_to_high_20_rank` | `distance_to_high_20` |
| `distance_to_high_60_rank` | `distance_to_high_60` |
| `distance_to_low_20_rank` | `distance_to_low_20` |
| `range_hl_rank` | `range_hl` |

排名只在 `universe_daily` 标记为可用的股票中计算；索引、非支持板块、ST、停牌、无历史或无有效 OHLCV/成交量的股票不参与。

## 四、20 个市场状态字段

这些字段每天在同一有效股票池上聚合，因此所有股票在同一天共享一个市场向量：

| 类别 | 字段 |
| --- | --- |
| 涨跌广度 | `up_ratio, down_ratio, flat_ratio` |
| 单日收益分布 | `ret_close_1d_mean, ret_close_1d_median, ret_close_1d_std` |
| 收益桶占比 | `ret_lt_minus_10_ratio, ret_minus_10_to_minus_5_ratio, ret_minus_5_to_minus_1_ratio, ret_minus_1_to_0_ratio, ret_0_to_1_ratio, ret_1_to_5_ratio, ret_5_to_10_ratio, ret_gt_10_ratio` |
| 流动性总量 | `amount_sum, volume_sum` |
| 流动性异常 | `amount_zscore_20, volume_zscore_20`，分别是上述日总量的 20 日 z-score。 |
| 集中度/横截面离散度 | `amount_top10_share` 为成交额最高 10% 股票占总成交额比例；`ret_p90_minus_p10` 为收益 P90 减 P10。 |

## 五、6 个候选上下文字段

这六个字段不是价格序列，而是当前股票在 Reward 候选 construction 中的相对位置：

| 字段 | 定义 |
| --- | --- |
| `wts_rank_pct` | 候选名次 / 候选池大小。 |
| `wts_inv_rank` | $1 / max(rank,1)$。 |
| `wts_score_z` | 候选分数 z-score，经截断到 $[-8,8]$ 后除以 8。 |
| `wts_score_pct` | 候选分数的 rank 百分位。 |
| `wts_selector_where` | 原弱转强 selector 是否通过，取 0/1。 |
| `wts_rank_in_topn_pct` | 候选 rank / `topn`，截断至 $[0,1]$。 |

在 AE 中，该 6 维向量先投影为 16 维，再沿 60 个时间步复制；这表示候选身份在整个历史窗口内保持同一个条件，而不是伪造为每天变化的特征。

## 六、归一化与无前视

正式 scaler 是模型版本的一部分，不是每次 inference 的预处理步骤。它固定保存在：

```text
weights/reward_v1/preprocessing/pre2020_eval2020_2026_raw_relative_scaler.json
```

其 SHA-256 记录在 `weights/reward_v1/MANIFEST.json`。该 scaler 拟合时使用：

| 项目 | 值 |
| --- | ---: |
| 样本数 | 4,096 |
| token 数 | 245,760 |
| seed | 20260717 |
| 实际拟合样本日期跨度 | 2010-05-04 至 2010-05-31 |

拟合过程为：

1. 原始个股、派生个股和市场字段分别统计；
2. 每个维度先截断到训练 token 的 0.5% / 99.5% 分位；
3. 再用截断后的均值和标准差标准化；
4. 非有限值填为 0；标准差过小则用 1。

候选 6 维字段使用上述明确的裁剪/缩放规则，不共享时序 scaler。

新数据 inference 必须沿用这份 JSON：即使数据更新后均值、方差或分位数变化，也不得重拟合 scaler。重新拟合会改变 52+20 个输入维度的坐标系，因而不能与冻结 AE 和 Reward v1 checkpoint 逐分数复现。

训练数据会拒绝长度不足 60 日、无有效价格、无成交量或未来路径不能结算的行。
`prepare-inference-index` 则只要求信号日以前的特征和下一交易日可交易信息，不要求未来收益标签。
详情见 `models/reward/feature_store/prepare_features_impl.py`、
`models/reward/pipeline/prepare_dataset_impl.py` 和
`models/reward/pipeline/inference_index.py`。
