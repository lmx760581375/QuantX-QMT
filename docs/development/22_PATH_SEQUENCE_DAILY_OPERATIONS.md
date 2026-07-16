# Path Sequence 日度跟踪与运维说明

> 日期: 2026-07-15  
> 状态: milestone 已接入主线  
> 策略: `path_sequence_top7_bridge_v1`

## 1. 当前 milestone 是什么

当前合入主线的是 `path_sequence_top7_bridge_v1`，中文标题为“路径序列Top7”。它使用冻结的 `PredictionStore` 作为 alpha 输入，并通过标准 `decision_pipeline` 在下一交易日开盘执行目标组合调仓。

当前版本的关键边界：

1. 排序分数来自研究产物 `scored_panel_v2_with_aux.parquet` 中的 `grid0878` 组合，不是 live DL 模型实时推理。
2. 持仓规则是 Top7 等权，目标总仓位 `0.98`，每 5 个交易日允许一次 rebalance。
3. 日度生产不会隐式重训模型，也不会在缺少 DL 分数字段时自动退化成别的模型。
4. 当前 PredictionStore 最新可用信号日是 `2026-07-02`；`2026-07-15` 的模拟持仓是历史预测延续后的状态，不是 7/15 新预测。

相关文件：

```text
configs/strategies/generated/path_sequence_top7_bridge_v1.yaml
artifacts/predictions/path_sequence_top7_bridge/latest.json
artifacts/predictions/path_sequence_top7_bridge/latest.manifest.json
quantx/tools/generate_path_sequence_predictions.py
```

## 2. 每日手工流程

常用 Python 环境：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python
```

建议每日流程：

```bash
# 1. 拉取 QMT 增量数据，同时默认增量刷新 Qlib 换手率 feature
#    会写入 $turnover_rate / $circ_mv / $free_float_mv 等普通 Qlib 字段；临时跳过可加 --skip-turnover-features
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.tools.sync_daily_data \
  --source qmt \
  --provider-uri data/qlib_data_fixed \
  --raw-dir data/raw/qmt \
  --mode incremental \
  --max-requests 45000 \
  --socket-timeout 30 \
  --progress-every 1 \
  --json

# 2. 刷新 PredictionStore bridge
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage predictions \
  --trade-date YYYY-MM-DD \
  --json

# 3. 生成策略模拟持仓和交易建议
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage signals \
  --trade-date YYYY-MM-DD \
  --strategy path_sequence_top7_bridge_v1 \
  --json

# 4. 生成中文日报
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.tools.run_daily_pipeline \
  --profile configs/production/daily_default.yaml \
  --stage report \
  --trade-date YYYY-MM-DD \
  --strategy path_sequence_top7_bridge_v1 \
  --json
```

产物目录：

```text
daily_runs/YYYYMMDD/default/
  data_update.json
  prediction_jobs.json
  strategy_signals.json
  strategy_positions.json
  suggested_orders.json
  rejected_orders.json
  next_session_guides.json
  daily_report.json
  report.md
  report.html
```

## 3. 看板入口

启动本地 workspace：

```bash
/Users/mingxiaoli/anaconda3/envs/test/bin/python -m quantx.server.app
```

打开：

```text
http://127.0.0.1:8000
```

侧边栏进入“每日跟踪”。可直接验证 API：

```bash
curl http://127.0.0.1:8000/api/daily-runs
curl http://127.0.0.1:8000/api/daily-runs/YYYYMMDD/default
```

## 4. 2026-07-15 验证状态

数据已更新到 `2026-07-15`：

```text
calendar_end=2026-07-15
instrument_count=5324
feature_file_count=47913
```

`path_sequence_top7_bridge_v1` 在 2026-07-15 的日度状态：

```text
当前持仓: 1 只
今日买入: 0 笔
今日卖出: 0 笔
拒单/过滤: 0 笔
PredictionStore 最新可用信号日: 2026-07-02
2026-07-15 查询信号日 2026-07-14: missing_prediction
```

当前持仓：

```text
SH605255 天普股份 5500 股，持仓 10 天，浮盈约 29.48%
```

最终口径回测：

```text
run_id=20260715_path_sequence_top7_bridge_v1_stoploss10_qmt_st_buyfilters_no_untradable_exec
start=2022-01-04
end=2026-07-15
final_value=2,643,939.06
total_return=164.39%
annual_return=23.95%
max_drawdown=-39.69%
sharpe=0.822
effective_buys=1364
unique_buy_symbols=655
```

独立交易审计结果：

```text
SZ000609 2025-10-27 买入: 0
SZ000506 2024-10-16 买入: 0
SZ000506 2024-10-09 买入: 0
当前名称含 ST/退 的有效买入: 0
缺失行情有效买入: 0
零成交量/零成交额有效买入: 0
一字涨停有效买入: 0
开盘高开 >= 3% 有效买入: 0
```

注意：当前 QMT 路径只能可靠拿到当前 `InstrumentName`，不能拿到 point-in-time 历史 ST 名称。历史回测里的 ST 风险使用 5% 涨跌停特征代理过滤；实盘/日度跟踪则用当天刷新到 MetaStore 的 QMT 当前名称过滤 `ST` 和 `退`。

## 5. 当前买卖执行规则

当前主线 Top7 不再是单纯“预测 TopK 就买”。有效买入必须同时通过以下规则：

1. 使用 T 日冻结 `PredictionStore`，最早 T+1 开盘执行，不做日内回看。
2. 每 5 个交易日最多 rebalance 一次；止损触发时可先卖出并用当日可买候选补位。
3. 目标组合为 Top7 等权，总仓位 `0.98`，按 100 股整数手下单。
4. `$volume <= 0` 或 `$amount <= 0` 的停牌/占位 bar 不可交易。
5. OHLC 同价且相对前收接近涨停的一字板不可买入；一字跌停不可卖出。
6. 当前名称包含 `ST` 或 `退` 的股票不可新买入。
7. 历史回测中，过去 60 个交易日多次出现 5% 限幅且无 10% 正常涨跌幅特征的 ST-like 股票不可新买入。
8. 开盘价相对前收高开 `>= 3%` 不追买。
9. 持仓亏损达到 `-10%` 触发止损卖出。
10. 卖出后 5 个交易日内不重新买入同一只股票。

## 6. 为什么全量 signals 会慢

当前 `run_daily_pipeline --stage signals` 为了得到可追溯的模拟持仓，会对策略从起始日完整回放到目标交易日，再读取最后一天持仓和订单。因此它不是单日增量计算。全 profile 会跑 9 个策略；日常只看 Top7 时建议带 `--strategy path_sequence_top7_bridge_v1`。

这对 milestone 验证是可接受的，但不是长期日更的最优形态。后续优化方向：

1. 把“状态重建/验证回放”和“每日单日增量”分成两个 stage 或两个模式。
2. `daily_state/default` 保存每个策略的上一日持仓、现金、最高价、成本和 pending orders。
3. 日常默认只加载上一日状态，计算最新交易日信号与订单。
4. 只有数据修复、策略配置变更、模型切换或状态损坏时才运行完整回放。

## 7. DL 版本如何推进

当前 bridge 不需要日常训练。真正的 DL 版本不能每天临时训练后直接用于生产，因为这样会破坏可追溯性，也难以判断收益来自模型还是随机训练波动。

建议的 live DL 产物标准：

1. 固定 pairwise/contrastive 模型结构和 `state_dict`。
2. 固定 scaler、feature list、feature schema hash、训练窗口、随机种子和 checksum。
3. 保存训练 manifest，明确训练信息截止日和预测区间。
4. 建立无标签 live feature 生成链路，不能依赖未来 T+5 标签列。
5. 生成新的 PredictionStore，与 bridge 版同屏 shadow 跟踪。
6. 通过 promotion gate 后再替换主线策略，例如最近 N 次换仓收益、回撤、换手、ST 占比和缺预测率。

重训频率建议月度或季度，除非发现明显漂移或数据口径变化。日度生产只做推理和产物校验，不做隐式重训。

## 8. 当前限制

1. 研究面板最新信号日停在 `2026-07-02`，所以当前 bridge 不能提供 2026-07-15 最新预测。
2. 2026-07-15 的 path sequence 持仓是历史预测延续，不代表新数据已经产生新信号。
3. 当前全 profile 日度计算仍是完整回放，耗时明显高于单日实盘所需。
4. 主线版本不是 `grid0878 + pairwise MLP alpha 0.03` 的完整 DL live 版；DL live 需要补模型产物和无标签特征链路后再切换。
