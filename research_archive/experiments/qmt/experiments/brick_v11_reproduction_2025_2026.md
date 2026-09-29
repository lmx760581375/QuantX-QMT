# Brick V11 2025-2026 本地复现实验

本文记录对投资科学公开页面 Brick V11 策略的本地复现过程。这里的目标不是声称拿到了原作者的真实 AI 模型，而是在公开页面能看到的交易结果、排序结果和策略描述基础上，构造一个 QuantX 内可重复运行的代理策略，使 2025-2026 至今的买入集合、收益和回撤尽量贴近公开回测。

公开目标页面：

`https://touzikexue.com/strategy-backtest/20260703_171700_brick_v6_v11_stage1_ablation_ridge_v10_brick_top5_oos_2025_2026`

当前可运行配置：

`configs/strategies/generated/brick_v11_external_score_reproduction_2025_2026.yaml`

当前外部打分表：

`/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_scores.parquet`

大体结论：原策略最难复现的部分是每日 Top-5 的隐藏 AI 打分。卖出规则、T+1 open、买入滑点和仓位规则可以通过配置近似表达；真正决定交易集合的是选股排序。因此本实验的核心是用公开 Top-5 买入明细做弱监督标签，训练一个本地排序代理模型，再把代理模型输出的每日分数接入 QuantX 回测。

## 一、复现目标

公开页面给出的核心目标指标如下：

- 区间：2025 年至 2026-07-02 左右。
- 初始资金：`1,000,000`。
- 总收益：`1.606061`，即约 `+160.6061%`。
- 期末净值：`2,606,061.10`。
- 最大回撤：`-0.063714`。
- 买入次数：`385`。

本地当前最佳 QuantX 复现结果如下：

- 区间：`2025-01-02` 到 `2026-07-02`。
- 本地数据源：`data/qlib_data_fixed`。
- 加载股票数：`352`，已经按外部分数实际候选集裁剪。
- 总收益：`1.604498`，即约 `+160.4498%`。
- 期末净值：`2,604,498.24`。
- 年化收益：`0.896316`。
- 最大回撤：`-0.072254`。
- Sharpe：`5.854578`。
- 买入次数：`383`。
- 卖出次数：`374`。
- 期末持仓：`8`。
- 已成交卖出原因：`green 319`、`7red 42`、`max_hold_10d 13`。

买入集合重合度是这次复现最关键的检查：本地买入 `383` 笔，公开目标买入 `385` 笔，交集 `383` 笔。缺少的两笔是 `2025-08-27 SH600608` 和 `2026-01-15 SH600636`，这两个标的在当前本地 `data/qlib_data_fixed` 数据源中不存在，所以不能在本地回测中被交易。

## 二、公开策略拆解

根据用户提供的视频文字和公开页面信息，我把策略拆成四层：

1. 市场环境过滤：只在活跃市值多头波段内允许新买入；波段外只处理已有持仓退出，不新增仓位。
2. 每日选股排序：按砖形图相关模型打分，取每日 Top-5 作为候选。
3. 执行和仓位：买入使用 T+1 open；买入端滑点 `30bps`；单仓最高约 `50%`，单仓最低约 `10%`；剩余资金低于约 `5%` 后不再继续拆仓。
4. 卖出规则：持仓满 1 个交易日后绿砖退出；持仓过程中触发 7 块红砖风险退出；最长持有 10 个交易日退出。

公开内容还暗示了几个和排序模型有关的特征方向：

- 活跃市值启动后，启动当天或两天内涨幅越高，未来收益越好。
- 涨停或强涨幅标的未来收益更高。
- 砖形图反转，也就是从绿砖转红砖，未来收益更高。
- 红砖高度或红砖增量越强，未来收益越好。
- 筹码获利盘、趋势状态、成交活跃度等可能进入打分模型。
- 原作者提到使用 2013-2024 数据训练 AI，再用 2025 以后数据回测，但没有公开特征、标签和模型参数。

由于原始训练标签没有公开，本实验没有办法真实复刻原作者的训练过程。我的处理方式是：把公开页面 2025-2026 已经展示出来的 Top-5 交易结果视为教师输出，用这些输出训练一个本地学生排序器。这个方法更准确地说是公开结果蒸馏，不是原模型复刻。

## 三、数据和临时实验目录

所有早期实验文件、训练脚本、候选配置和回测输出都放在 `/tmp/quantx_brick_v11_proxy` 下，避免在没有找到有效方案前污染代码库。

关键临时文件：

- `/tmp/quantx_brick_v11_proxy/target_trades.json`：从公开页面/API 保存的目标交易明细。
- `/tmp/quantx_brick_v11_proxy/target_daily.json`：公开页面的每日净值或日度数据。
- `/tmp/quantx_brick_v11_proxy/target_meta.json`：公开页面元信息。
- `/tmp/quantx_brick_v11_proxy/target_feature_samples.parquet`：目标信号日上，本地 QuantX 因子运行器计算出的候选样本特征。
- `/tmp/quantx_brick_v11_proxy/train_rank_distill_proxy.py`：第一版公开排序蒸馏脚本。
- `/tmp/quantx_brick_v11_proxy/train_iterative_rank_distill.py`：最终使用的迭代 hard-negative 蒸馏脚本。
- `/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_scores.parquet`：最终外部打分表。
- `/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_summary.json`：迭代训练摘要。
- `/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_iter_et_r5.joblib`：最终一轮训练出的模型文件。
- `/tmp/quantx_brick_v11_proxy/native_runs/brick_v11_repo_config_reproduction_trimmed`：使用仓库配置跑出的验证结果。

仓库里只保留可运行配置和必要的 QuantX 扩展，不把大体积 parquet 打分表提交进仓库。这样做的原因是外部打分表是研究产物，而且路径仍在 `/tmp`，后续可以继续替换或重训。

## 四、目标交易如何变成训练标签

公开页面的买卖明细里有每笔买入对应的选股日、股票、买入时排名和打分。脚本把这些信息转成以 `(signal_date, instrument)` 为 key 的监督信号。

代码逻辑在 `/tmp/quantx_brick_v11_proxy/train_rank_distill_proxy.py` 的 `attach_public_labels` 中，核心构造如下：

- `score_map[(signal_date, symbol)] = score_on_buy`。
- `rank_map[(signal_date, symbol)] = rank_on_buy`。
- 对本地候选样本逐行匹配 `(date, instrument)`。
- 命中公开 Top-5 的样本，`public_rank` 为 1 到 5，`public_score` 为页面显示分数。
- 没命中的样本，`public_rank = 0`，`public_score = 0`。

训练标签不是简单的 0/1，而是把公开排名和公开分数混合：

- `label_rank = (6 - public_rank) / 5`，排名第 1 的标签为 `1.0`，第 5 的标签为 `0.2`，非目标为 `0`。
- `label_score = public_score / max(public_score)`，把公开分数归一化。
- `label_blend = 0.55 * label_rank + 0.45 * label_score`。

这样构造的原因是：公开排名决定了 Top-5 的相对次序，公开分数保留了同一排名附近的强弱差异。单独用 rank 会丢掉分数强度，单独用 score 又容易被页面分数尺度影响，所以最终采用混合标签。

这一步的本质是 teacher-student 蒸馏：公开页面的 Top-5 是 teacher 的输出，本地模型学习在同一候选池里把这些股票排到前面。

## 五、候选样本和特征工程

候选样本不是全市场任意股票，而是在目标信号日附近用本地 QuantX 因子系统计算得到的一批候选行。这样做可以让训练任务更接近原页面的选股环境，也能控制样本规模。

基础特征来自几类信息：

- 砖形图状态：`brick`、`brick_prev`、`brick_growth`、`prior_green_bars`、`red_risk_7`。
- 近期涨幅：`ret1`、`ret3`、`ret5`、`ret10` 以及对应横截面 rank。
- 成交和流动性：`vol_rank`、`vol_ratio`、`liquidity_rank`、`turnover43`、`turnover_amount`。
- 均线和趋势：`ma5`、`ma10`、`ma20`、`ma50`、`ma100`、`ma150`、`ma14`、`ma28`、`ma57`、`ma114`、`zxdq`、`zxdkx`。
- 形态派生：`green_to_red`、`brick_abs_prev`、`brick_strength`。
- 价格和流动性相对位置：`low_liq_rank`、`price_rank_low`。
- 趋势比值：`trend_zxdq_zxdkx`、`trend_ma50_100`、`trend_ma100_150`。
- 交互项：`ret1_x_brick`、`ret1_x_vol`、`low_liq_x_brick`、`price_x_brick`。
- 市场和板块类型：`is_sh`、`is_sz`、`is_star`、`is_chinext`、`is_mainboard`、`code_bucket`。

最终迭代脚本又增加了按日标准化和二次项：

- 日内 z-score：`day_brick_z`、`day_ret1_z`、`day_vol_z`、`day_liq_z`、`day_price_z`。
- 二次项：`brick_rank2`、`ret1_rank2`、`vol_rank2`、`low_liq_rank2`、`price_rank_low2`。
- 额外交互：`green_red_x_ret1`、`green_red_x_low_liq`、`mainboard_x_brick`。

这些特征的选择依据不是任意堆叠，而是围绕视频里反复强调的逻辑：活跃市值启动后的强涨幅、砖形图反转、红砖强度、成交活跃、趋势位置和主板过滤。

## 六、为什么使用 hard-negative 蒸馏

公开 Top-5 是非常稀疏的标签。每天几千只股票里只有最多 5 只正样本，如果随机采负样本，模型很容易只学到“普通股票不是目标”，但学不会把看起来也很强的股票排除掉。

所以训练时专门构造 hard negatives，也就是那些虽然不是公开 Top-5，但从手工因子上看也很像候选的股票。第一版 hard negative 分数大致由以下部分组成：

- `brick_rank` 权重 `0.28`。
- `ret1_rank` 权重 `0.22`。
- `vol_rank` 权重 `0.15`。
- `low_liq_rank` 权重 `0.15`。
- `brick_strength` 权重 `0.12`。
- `price_rank_low` 权重 `0.08`。

也就是说，负样本优先选择“砖形图强、短期涨幅强、成交强、流动性和价格位置也像目标”的股票。这类负样本最能逼模型学习公开 Top-5 的细微偏好。

最终脚本 `/tmp/quantx_brick_v11_proxy/train_iterative_rank_distill.py` 采用迭代 hard-negative：

1. 先用每天 hard-negative 分数最高的一批非目标股票做负样本。
2. 训练模型。
3. 用模型给全候选池打分。
4. 每天取模型排在前面的非目标股票，加入下一轮 hard negatives。
5. 重复 5 轮。

这样做的目的，是让模型逐轮修正自己的误报。第一轮模型认为很像目标、但公开页面没有选中的股票，会在下一轮变成更强的负样本。

## 七、模型选择

我试过多种轻量模型，包括：

- `HistGradientBoostingRegressor`。
- `RandomForestRegressor`。
- `ExtraTreesRegressor`。
- 简单公开分数直接 replay。
- 若干手工公式排序。

最终使用的是 sklearn 的 `ExtraTreesRegressor`，放在 `SimpleImputer(strategy='median')` 后面组成 pipeline。最终迭代脚本的模型参数是：

- 模型：`ExtraTreesRegressor`。
- 缺失值处理：中位数填充。
- 轮数：5 轮迭代。
- 每轮树数量：`520 + round_id * 80`。
- `max_depth=None`。
- `min_samples_leaf=1`。
- `max_features=0.72`。
- `bootstrap=False`。
- `n_jobs=-1`。
- `random_state=20260713 + round_id`。

训练权重也偏向正样本和更靠前的公开排名：

- 正样本权重：`420.0 + 90.0 * label_rank`。
- 负样本权重：`1.0`。

选择 ExtraTrees 的原因是它对非线性特征、rank 特征、交互项和离散市场类型特征比较稳，训练速度也适合在 `/tmp` 里快速迭代。这里没有用深度学习模型，因为公开监督信号只有 383 个本地可匹配正样本，复杂模型更容易只是记忆公开结果，而且开发量和维护成本不必要。

## 八、迭代训练结果

训练摘要来自 `/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_summary.json`。

每轮在公开目标日上检查：每天按模型分数排序取 Top-5，看是否命中公开 Top-5。结果如下：

| 轮次 | 训练行数 | 正样本数 | Top-5 命中 | 目标槽位 | Top-5 recall | false Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 156383 | 383 | 368 | 383 | 0.960836 | 277 |
| 2 | 166402 | 383 | 383 | 383 | 1.000000 | 262 |
| 3 | 176419 | 383 | 383 | 383 | 1.000000 | 262 |
| 4 | 186436 | 383 | 383 | 383 | 1.000000 | 262 |
| 5 | 196453 | 383 | 383 | 383 | 1.000000 | 262 |

第 2 轮开始已经能在本地可用数据中覆盖全部 383 个公开目标买入。最终配置使用 `iter_et_r4_score`，不是第 5 轮分数。原因是后续回测对收益、买入次数、仓位路径一起验证时，`iter_et_r4_score` 的交易路径更贴近公开结果。

## 九、QuantX 接入方式

原有配置策略主要支持在配置中写公式和 selector，但这个复现需要直接读取外部模型分数表。因此在 QuantX 中新增了一个很小的 selector 扩展：`selector.mode=external_score`。

相关代码：

- `quantx/core/strategy/config_strategy.py`：新增 `ExternalScoreSelector`。
- `quantx/tools/run_backtest.py`：支持 `data.universe: external_score`，并按 selector 的 `score_floor/topk/sort` 裁剪实际需要加载的股票池。

这个 selector 的行为：

- 从 parquet/csv 等外部表读取分数。
- 使用配置指定 `date_col`、`instrument_col`、`score_col`。
- 每个交易日按分数排序。
- 应用 `score_floor`。
- 取每日 `topk`。
- 生成标准 QuantX `Signal`，后续仍然走原有撮合、仓位、卖出规则和成本模型。
- 支持 `lag: 1`，用于表达选股日信号在 T+1 open 执行。

这样做的好处是把“模型打分”与“交易执行”解耦。未来如果重训出更好的分数表，只需要替换 `/tmp` parquet 或改 `score_col`，不需要改回测引擎。

## 十、成本模型调整

公开策略描述里明确提到买入端滑点 `30bps`，但没有说明卖出端也加同样滑点。为了贴近该执行假设，成本模型增加了非对称滑点：

- `buy_slippage`：买入滑点。
- `sell_slippage`：卖出滑点。
- 旧字段 `slippage` 保留，作为默认兼容值。

相关代码：

- `quantx/core/engine/cost.py`。
- `quantx/tools/run_backtest.py` 的 `build_cost`。

当前配置使用：

- `commission_rate: 0.0005`。
- `min_commission: 5.0`。
- `stamp_tax_rate: 0.0001`。
- `stamp_tax_on_buy: false`。
- `transfer_fee_rate: 0.0`。
- `slippage: 0.0`。
- `buy_slippage: 0.003`。
- `sell_slippage: 0.0`。

## 十一、最终配置细节

最终仓库配置在：

`configs/strategies/generated/brick_v11_external_score_reproduction_2025_2026.yaml`

核心参数如下。

数据：

- `provider_uri: data/qlib_data_fixed`。
- `universe: external_score`。
- `start: '2025-01-02'`。
- `end: '2026-07-02'`。
- `look_back_days: 720`。

砖形图和退出信号：

- `brick: BrickChart(high, low, close, 8, 3, 12, 12, 8, 92, 114, 1, 1, 1)`。
- `red_risk: brick > 0`。
- `red_risk_7: Sum(red_risk, 7)`。
- `green_exit: brick < 0`。
- `red7_exit: red_risk_7 >= 7`。

外部排序：

- `mode: external_score`。
- `path: /tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_scores.parquet`。
- `score_col: iter_et_r4_score`。
- `score_floor: 0.02`。
- `lag: 1`。
- `sort: score_desc`。
- `topk: 5`。
- `candidate_limit: 20`。

仓位：

- `type: equal_weight`。
- `max_positions: 11`。
- `buy_only_new_positions: false`。
- `rank_weights: [0.9, 0.85, 1.2, 1.05, 1.15]`。

执行：

- `deal_price: open`。
- `cash_use_ratio: 0.98`。
- `buy.sizing: cash_equal`。
- `buy.lot_size: 100`。
- `buy.skip_if_holding: false`。
- `buy.skip_limit_up: true`。
- `buy.reuse_sell_cash: true`。

卖出：

- `green`：`holding_days >= 1 and green_exit`。
- `7red`：`red7_exit`。
- `max_hold_10d`：`holding_days >= 10`。

这里的 `max_positions: 11` 看起来和“单仓最高 50%、最低 10%”不完全一样，但它是为了让 QuantX 的现金复用、已有持仓叠加和每日 Top-5 买入路径更贴近公开交易明细。公开策略页面展示的是最终交易行为，不是完整仓位引擎参数，因此仓位部分采用了回放拟合。

## 十二、如何重新运行

使用 conda 的 `test` 环境：

```bash
conda run --no-capture-output -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/brick_v11_external_score_reproduction_2025_2026.yaml \
  --output-dir /tmp/quantx_brick_v11_proxy/native_runs \
  --run-id brick_v11_repo_config_reproduction_trimmed \
  --json
```

已验证输出目录：

`/tmp/quantx_brick_v11_proxy/native_runs/brick_v11_repo_config_reproduction_trimmed`

其中重要文件：

- `summary.json`：汇总指标。
- `trades.json`：全部成交记录。
- `closed_positions.json`：已平仓持仓及退出原因。
- `daily_nav.json`：日度净值。
- `daily_selection_candidates.json`：每日选股候选。
- `selection_candidates.json`：候选明细。

## 十三、为什么不是直接写死公开交易

这次复现没有把公开交易明细直接写成买入表。原因是直接写死 trades 虽然能更贴近页面，但它不能回答“这个策略如何生成候选”的问题，也不能用于后续扩展到新日期。

当前方法保留了一个可替换的模型分数层：

1. 公开目标交易只用于训练代理打分。
2. 回测时 QuantX 每天仍按分数排序、过滤、下单。
3. 卖出仍由砖形图规则实时触发。
4. 仓位、涨停跳过、T+1 open、成本都由引擎执行。

所以它比交易回放更接近“策略复现”，但仍然要明确：这是从公开输出反推的代理模型，不是原作者内部 AI 模型。

## 十四、尝试过但没有作为最终方案的方向

1. 纯手工公式搜索。

   用砖形图强度、涨幅、成交量、流动性、主板过滤等手工组合做排序，收益和交易集合都不够稳定。它能抓到一部分“强势红砖”逻辑，但无法解释公开 Top-5 里的细微排序。

2. 直接使用公开页面分数。

   如果只拿页面已有 public score 做 replay，可以得到部分相似路径，但买入次数和目标集合不如迭代蒸馏稳定。公开分数只覆盖已经展示出来的候选，对完整候选池的排序能力不足。

3. 调卖出参数优先匹配回撤。

   `/tmp/quantx_brick_v11_proxy/artifacts/sell_param_search_public_20260713_234052.json` 中有卖出参数搜索。某些组合能让收益或回撤看起来更接近，但买入次数、持仓路径和退出原因会偏离公开页面。最终没有采用这些过拟合退出参数。

4. 更高复杂度模型。

   原作者可能用过 Ridge、树模型或其他模型，但公开正样本数量太少。更复杂模型对当前任务没有明显收益，反而会增加不可解释和不可维护程度。

## 十五、当前误差和限制

1. 缺少两个本地数据标的。

   公开目标的 `SH600608` 和 `SH600636` 在本地 `data/qlib_data_fixed` 中不存在。因此即使外部分数表包含它们，QuantX 也无法加载行情和执行交易。这解释了 `383 / 385` 的买入重合度上限。

2. 卖出原因分布仍有差异。

   当前本地已成交退出原因是 `green 319`、`7red 42`、`max_hold_10d 13`。公开页面存在类似 `green_t1_deferred` 的桶，本地配置目前把它归入更普通的绿砖退出逻辑。收益已经非常接近，但退出原因命名和部分延迟处理不完全一致。

3. 活跃市值波段不是单独显式建模。

   当前外部分数表已经隐含了公开目标交易日期，也就是它实际上只在公开 Top-5 所在环境里产生有效高分。后续如果要扩展到未来日期，需要把活跃市值多头波段识别单独做成可运行过滤器，而不是只依赖蒸馏分数。

4. 这不是严格样本外模型。

   原作者声称用 2013-2024 训练、2025 以后回测。我们没有原始训练标签，所以当前模型是用 2025-2026 公开输出训练代理分数，再复现同一区间。它适合用来研究公开策略的结构和交易路径，不适合作为真实样本外绩效证明。

5. 外部 parquet 仍在 `/tmp`。

   仓库配置依赖 `/tmp/quantx_brick_v11_proxy/ml_iter_rank_proxy_scores.parquet`。如果清理 `/tmp`，需要重新生成或恢复该文件。后续若要正式化，可以把分数生成脚本、特征样本生成脚本和模型 artifact 管理流程再整理进仓库。

## 十六、代码改动摘要

为了让这个策略以正常 QuantX 配置运行，仓库里加入了少量通用能力：

- `quantx/core/strategy/config_strategy.py`：新增 `ExternalScoreSelector`，读取外部打分表并生成标准信号。
- `quantx/tools/run_backtest.py`：支持 `data.universe: external_score`，并按外部 selector 裁剪实际股票池，避免加载全市场几千只股票。
- `quantx/core/engine/cost.py`：支持 `buy_slippage` 和 `sell_slippage`，保留 `slippage` 兼容旧配置。
- `configs/strategies/generated/brick_v11_external_score_reproduction_2025_2026.yaml`：最终可运行复现配置。

对应测试也补了覆盖：

- 外部打分 selector 的 T+1 行为和 score floor。
- `where` 为空时配置编译。
- 非对称滑点。
- `external_score` universe 裁剪。

注意：当前环境里 `pytest` 自身会在启动时发生 `Segmentation fault: 11`，甚至 `python -m pytest --version` 也会崩溃，所以没有拿到完整 pytest 结果。已做过 Python 文件 `py_compile` 验证，回测命令也已经成功跑通。

## 十七、后续改进方向

1. 把活跃市值多头波段识别显式配置化。
2. 用 2013-2024 的本地历史数据构造真正的 forward-return label，训练一个不依赖公开 2025-2026 输出的排序模型。
3. 将公开蒸馏模型作为 teacher，把历史 forward label 模型作为 student 的约束之一，减少对公开结果的记忆。
4. 补齐或更新本地行情数据源，确认 `SH600608`、`SH600636` 是否能恢复。
5. 单独复刻 `green_t1_deferred` 这类退出桶，进一步对齐卖出原因和日内资金曲线。

## 十八、当前判断

当前复现已经达到了研究阶段的目标：收益接近公开页面，买入集合在本地可交易标的上完全重合，卖出规则和执行模型也基本符合公开描述。

但它的性质必须写清楚：这是“基于公开 Top-5 输出的本地蒸馏复现”，不是原作者 AI 模型的真实复刻。它可以作为后续研究砖形图策略、活跃市值过滤、AI 打分标签构造和 QuantX 外部模型接入的基准 artifact。

## 十九、严格 OOS forward-label 版本

上面的蒸馏复现回答的是“如何把公开页面 2025-2026 的 Top-5 输出复现出来”。但它没有回答更接近博主真实做法的问题：如果不用公开 Top-5 当标签，只用历史行情和砖形图交易规则构造训练标签，能不能在 2025-2026 做样本外排序。

因此我又做了一版严格 OOS forward-label 管线。这个版本的核心约束是：训练标签只来自 2013-2024 的真实行情和模拟退出收益，2025-2026 只作为预测和回测区间，不参与训练标签构造。

临时脚本：

`/tmp/quantx_brick_v11_proxy/build_forward_label_oos_pipeline.py`

输出目录：

`/tmp/quantx_brick_v11_forward_oos_v1`

关键输出：

- `/tmp/quantx_brick_v11_forward_oos_v1/train_2013_2024.parquet`：2013-2024 训练样本。
- `/tmp/quantx_brick_v11_forward_oos_v1/predict_2025_2026.parquet`：2025-2026 预测期候选样本及真实事后 label，用于标签层评估。
- `/tmp/quantx_brick_v11_forward_oos_v1/forward_label_oos_scores.parquet`：2025-2026 OOS 分数表，可直接给 `selector.mode=external_score` 使用。
- `/tmp/quantx_brick_v11_forward_oos_v1/model_hgb.joblib`：当前表现最好的 HGB 模型。
- `/tmp/quantx_brick_v11_forward_oos_v1/forward_label_oos_backtest.yaml`：严格 OOS 回测配置。
- `/tmp/quantx_brick_v11_forward_oos_v1/native_runs/forward_label_oos_hgb_top5`：资金路径回测结果。

### 19.1 标签定义

每一行训练样本对应一个候选股票在某个 `signal_date` 的状态。标签不是公开 Top-5，而是按策略规则从未来行情模拟出来：

1. 选股日在 `signal_date`。
2. 买入价使用下一交易日 open，并加买入滑点 `30bps`：`buy_price = next_open * 1.003`。
3. 持仓后按砖形图退出：
   - 满 1 个交易日后出现绿砖，按下一交易日 open 卖出。
   - 触发 7 红风险条件，按下一交易日 open 卖出。
   - 最长持有 10 个交易日，仍未触发其他退出则按 open 卖出。
4. `forward_return = sell_open / buy_price - 1`。
5. 同时记录 `mfe_close`、`mae_close` 和 `risk_adjusted_return`。

最终训练用的主标签是混合标签：

`label_blend = 0.75 * forward_return + 0.25 * tanh(risk_adjusted_return / 2)`

这样做是为了不只奖励绝对收益，也让模型稍微关注回撤后的风险调整表现。脚本同时保留了 `label_return`、`label_positive`、`label_tail`，方便后续改用“收益 / 风险调整收益 / 是否进入前分位”的不同标签口径。

### 19.2 活跃市值代理过滤

因为没有博主内部的“活跃市值”真实指标，我先实现了一个可替换的本地代理：

1. 用全市场成交额近似 active value：`sum((open + close) / 2 * volume)`。
2. 若 1 日或 2 日 active value 涨幅达到 `4%`，进入活跃多头波段。
3. 进入波段后，当 active value 跌破自身 MA5，退出波段。
4. 只有处在这个代理波段内的日期才生成新买入候选。

这个口径不是博主内部指标的精确还原，只是为了让“只在活跃市值多头区间内训练和交易”的约束先闭环跑通。后续如果拿到更准确的活跃市值序列，可以只替换这层过滤，不需要重写标签和模型框架。

### 19.3 候选池和特征

候选池仍围绕砖形图交易逻辑构造，不使用公开页面 Top-5：

- `brick > 0`。
- `brick_prev < 1.5`。
- `ret1_rank >= 0.35`。
- `vol_rank >= 0.15`。
- `amplitude < 0.18`。
- `liquidity_rank <= 0.95`。

使用的特征包括：

- 砖形图状态：`brick`、`brick_prev`、`brick_growth`、`brick_strength`、`prior_green_bars`、`red_risk_7`。
- 近期涨幅：`ret1`、`ret2`、`ret3`、`ret5`、`ret10` 和横截面 rank。
- 成交和流动性：`vol_ratio`、`vol_rank`、`turnover43`、`liquidity_rank`、`low_liq_rank`。
- 趋势位置：`ma5`、`ma10`、`ma20`、`ma50`、`ma100`、`ma150`、`zxdq`、`zxdkx`。
- 日内标准化：`day_brick_z`、`day_ret1_z`、`day_vol_z`、`day_liq_z`、`day_price_z`。
- 交互项：`green_to_red`、`ret1_x_brick`、`ret1_x_vol`、`low_liq_x_brick`、`price_x_brick`。
- 趋势比值：`trend_zxdq_zxdkx`、`trend_ma50_100`、`trend_ma100_150`。
- 股票类型：`is_star`、`is_chinext`、`is_mainboard`、`code_bucket`。

### 19.4 训练和预测边界

训练集：

- 文件：`/tmp/quantx_brick_v11_forward_oos_v1/train_2013_2024.parquet`。
- 行数：`448,249`。
- 信号日数量：`1,353`。
- 股票数：`2,977`。
- 信号日期：`2013-01-04` 到 `2024-12-27`。
- 退出日期：`2013-01-08` 到 `2024-12-31`。
- 平均 `forward_return`：`0.000154`。
- 中位数 `forward_return`：`-0.009649`。
- 正收益比例：`35.6853%`。

这里特意检查了 `exit_date <= 2024-12-31`，避免 2024 年底信号使用 2025 年退出收益污染训练。

预测和评估集：

- 文件：`/tmp/quantx_brick_v11_forward_oos_v1/predict_2025_2026.parquet`。
- 行数：`78,520`。
- 信号日数量：`173`。
- 股票数：`3,082`。
- 信号日期：`2025-01-02` 到 `2026-07-01`。
- 退出日期：`2025-01-06` 到 `2026-07-10`。
- 平均 `forward_return`：`-0.001514`。
- 中位数 `forward_return`：`-0.009862`。
- 正收益比例：`33.6589%`。

### 19.5 模型

这版先训练轻量模型，确保严格 OOS 流程完整跑通：

- `RidgeCV + median imputer + StandardScaler`。
- `HistGradientBoostingRegressor + median imputer`。

全量 `RandomForestRegressor` 和 `ExtraTreesRegressor` 在 `448,249` 行训练集上训练时间较长，本轮先没有作为最终分数。它们可以后续用抽样、减树数或分年度训练方式单独补。

标签层 OOS 结果：

| 模型 | OOS Top-5 行数 | Top-5 平均 forward_return | Top-5 中位数 | Top-5 正收益比例 | Top-5 退出原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| HGB | 865 | `0.011399` | `-0.009089` | `39.8844%` | `green 712`, `7red 153` |
| Ridge | 865 | `0.003002` | `-0.011661` | `35.1445%` | `green 698`, `7red 167` |

全候选池平均 `forward_return` 是 `-0.001514`，HGB Top-5 平均 `forward_return` 是 `+0.011399`，说明这个非蒸馏模型在标签层确实学到了一些排序能力。

### 19.6 资金路径回测

回测配置：

`/tmp/quantx_brick_v11_forward_oos_v1/forward_label_oos_backtest.yaml`

回测命令：

```bash
conda run --no-capture-output -n test python -m quantx.tools.run_backtest \
  --config /tmp/quantx_brick_v11_forward_oos_v1/forward_label_oos_backtest.yaml \
  --output-dir /tmp/quantx_brick_v11_forward_oos_v1/native_runs \
  --run-id forward_label_oos_hgb_top5 \
  --json
```

资金路径结果：

- 初始资金：`1,000,000`。
- 期末净值：`1,720,706.03`。
- 总收益：`0.720706`，即约 `+72.0706%`。
- 年化收益：`0.429862`。
- 最大回撤：`-0.156507`。
- Sharpe：`1.767910`。
- 加载股票数：`556`。
- 交易数：`1,378`。
- 买入：`696`。
- 卖出：`682`。
- 期末持仓：`0`。
- 卖出原因：`green 571`、`7red 119`、`max_hold_10d 3`。

这个结果和蒸馏复现有明显区别：蒸馏版贴近公开页面，收益约 `+160%`；严格 OOS forward-label 版收益约 `+72%`，回撤也更大。但严格 OOS 版没有使用公开 Top-5 标签，因此更接近“2013-2024 训练、2025 以后回测”的真实研究流程。它的收益量级也更接近用户贴的视频里另一个版本提到的“2025 到现在年化约 68% / 回测收益约 68%”那类结果，而不是完全拟合公开页面的 `+160%` 曲线。

### 19.7 当前判断

这一版已经完成了真正 label 的第一条闭环：

1. 用 2013-2024 历史行情构造砖形图动态退出 label。
2. 训练排序模型。
3. 对 2025-2026 生成 OOS 分数。
4. 用同一套 QuantX 执行、仓位、成本和卖出规则跑资金路径。

它仍然有几个限制：

- 活跃市值只是本地成交额代理，不是博主真实活跃市值指标。
- 候选池规则仍是人工近似，可能和博主 B 策略候选池不同。
- 当前只跑了 Ridge/HGB，重树模型还没有完整调优。
- 资金路径买入次数 `696` 明显多于公开页 `385`，说明活跃市值波段、score floor 或仓位限制还需要继续收敛。
- 最大回撤 `-15.65%` 明显大于公开页 `-6.37%`，风险过滤和市场波段退出仍是主要差距。

但从研究方法上，这已经比公开 Top-5 蒸馏更接近博主声称的训练路径：标签来自历史交易结果，预测区间严格样本外，不依赖 2025-2026 公开选股输出。

### 19.8 前分位标签补充实验

目标里还提到“是否进入前分位”这类监督标签，所以我在同一条 `/tmp` 管线里补了横截面分位标签。做法是每天在候选池内部，按实际退出后的 `forward_return`、`risk_adjusted_return`、`label_blend` 分别做横截面百分位排名，然后生成：

- `label_return_rank_pct`：当天候选内按实际退出收益的百分位。
- `label_risk_rank_pct`：当天候选内按风险调整收益的百分位。
- `label_blend_rank_pct`：当天候选内按混合收益标签的百分位。
- `label_top10`：`label_blend_rank_pct >= 0.90`。
- `label_top20`：`label_blend_rank_pct >= 0.80`。
- `label_rank_blend = 0.50 * label_blend_rank_pct + 0.30 * label_return_rank_pct + 0.20 * label_risk_rank_pct`。

这组标签仍然只用 2013-2024 训练期内的真实退出结果训练，2025-2026 只是 OOS 预测和评估。

临时输出目录：

`/tmp/quantx_brick_v11_forward_oos_rank_v1`

关键文件：

- `/tmp/quantx_brick_v11_forward_oos_rank_v1/forward_label_oos_scores.parquet`。
- `/tmp/quantx_brick_v11_forward_oos_rank_v1/forward_label_oos_result.json`。
- `/tmp/quantx_brick_v11_forward_oos_rank_v1/forward_label_rank_oos_backtest.yaml`。
- `/tmp/quantx_brick_v11_forward_oos_rank_v1/native_runs/forward_label_rank_oos_hgb_top5`。

训练命令：

```bash
conda run --no-capture-output -n test python /tmp/quantx_brick_v11_proxy/build_forward_label_oos_pipeline.py \
  --train-start 2013-01-01 \
  --train-end 2024-12-31 \
  --predict-start 2025-01-02 \
  --predict-end 2026-07-10 \
  --universe all_mainboard \
  --output-root /tmp/quantx_brick_v11_forward_oos_rank_v1 \
  --skip-build \
  --models ridge,hgb \
  --label label_rank_blend
```

标签层结果：

| 训练标签 | 模型 | OOS Top-5 平均 forward_return | Top-5 中位数 | Top-5 正收益比例 | Top-5 top10 命中率 | Top-5 top20 命中率 | Top-5 退出原因 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `label_rank_blend` | HGB | `0.002485` | `-0.004749` | `36.8786%` | `6.3584%` | `17.6879%` | `green 609`, `7red 256` |
| `label_rank_blend` | Ridge | `0.000905` | `-0.005126` | `38.9595%` | `6.1272%` | `16.5318%` | `green 624`, `7red 241` |

作为对照，预测集全候选池本身的 `label_top10` 比例约 `10.1172%`，`label_top20` 比例约 `20.1286%`。分位标签模型选出的 Top-5 在 top10/top20 命中率上没有超过全候选基准，说明当前特征和候选池下，“学习相对前分位”并没有学好真正的赢家集合。

资金路径回测命令：

```bash
conda run --no-capture-output -n test python -m quantx.tools.run_backtest \
  --config /tmp/quantx_brick_v11_forward_oos_rank_v1/forward_label_rank_oos_backtest.yaml \
  --output-dir /tmp/quantx_brick_v11_forward_oos_rank_v1/native_runs \
  --run-id forward_label_rank_oos_hgb_top5 \
  --json
```

资金路径结果：

- 期末净值：`1,152,738.00`。
- 总收益：`0.152738`，即约 `+15.2738%`。
- 年化收益：`0.098173`。
- 最大回撤：`-0.155148`。
- Sharpe：`0.475361`。
- 加载股票数：`543`。
- 交易数：`1,622`。
- 买入：`821`。
- 卖出：`801`。
- 卖出原因：`green 566`、`7red 235`、`max_hold_10d 2`。

这个补充实验的结论很明确：在当前候选池和活跃市值代理下，`label_rank_blend` 分位标签弱于 `label_blend` 收益/风险混合标签。它满足了“是否进入前分位”标签构造和严格 OOS 验证，但不是当前最优训练目标。当前最强的非蒸馏 OOS 版本仍是 19.6 的 HGB `label_blend` 模型，资金路径收益约 `+72.07%`。

### 19.9 尾部收益加权训练和执行层再优化

用户指出：如果训练版收益只有 `+72.07%`，它没有对齐公开页面约 `+160%` 的策略收益，说明训练或执行还需要优化。因此我在不使用公开 Top-5 买入标签的前提下，继续做了一轮严格 forward-label 训练优化。

这轮实验仍然遵守一个核心边界：模型训练标签只来自 2013-2024 的历史 forward label，不把 2025-2026 公开页面买入名单作为 teacher 输出使用。因此它不是第 4 到第 8 节那种公开 Top-5 蒸馏，而是“用历史交易结果构建 label，再预测 2025-2026”的训练版。

临时目录：

`/tmp/quantx_brick_v11_forward_oos_tail_v1`

关键脚本：

- `/tmp/quantx_brick_v11_proxy/optimize_forward_label_tail_models.py`。
- `/tmp/quantx_brick_v11_proxy/search_tail_execution_params.py`。
- `/tmp/quantx_brick_v11_proxy/search_tail_execution_targeted.py`。

关键产物：

- `/tmp/quantx_brick_v11_forward_oos_tail_v1/tail_oos_scores.parquet`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/tail_model_summary.json`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/execution_search.json`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/interrupted_run_summary.json`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/interrupted_run_summary.csv`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/best_tail_forward_label_oos_2025_2026.yaml`。

#### 训练输入

训练数据直接复用 19.6 生成的严格 forward-label 数据集：

- 训练集：`/tmp/quantx_brick_v11_forward_oos_v1/train_2013_2024.parquet`。
- 预测集：`/tmp/quantx_brick_v11_forward_oos_v1/predict_2025_2026.parquet`。

训练集中的每一行是一只股票在某个信号日的候选样本。它已经包含砖形图候选特征，以及从信号日之后按策略卖出规则回放得到的真实未来结果。卖出回放规则仍是：

- 满 1 个交易日后，出现绿砖信号退出。
- 持仓过程中触发 7 块红砖风险条件退出。
- 最长持有 10 个交易日退出。

因此这里的监督标签不是公开页面的买入结果，而是本地历史行情里“如果当天买这个候选，之后按同一套砖形图卖出规则退出，会得到什么结果”。

#### 标签构建

训练前先调用 `add_cross_sectional_outcome_labels`，在每日候选横截面内部构造相对结果标签。核心标签如下：

- `label_return`：按砖形图退出后的实际未来收益。
- `label_blend`：收益和风险调整后的混合标签，用来兼顾收益幅度和路径风险。
- `label_return_rank_pct`：当天候选内部按未来收益排序后的百分位。
- `label_risk_rank_pct`：当天候选内部按风险调整收益排序后的百分位。
- `label_blend_rank_pct`：当天候选内部按混合标签排序后的百分位。
- `label_top10`：`label_blend_rank_pct >= 0.90`。
- `label_top20`：`label_blend_rank_pct >= 0.80`。
- `label_rank_blend = 0.50 * label_blend_rank_pct + 0.30 * label_return_rank_pct + 0.20 * label_risk_rank_pct`。

这套标签的思路是：公开视频里说 AI 学的是“打分高的未来收益更高”，所以训练目标不应该只预测绝对收益，也应该学习每天横截面里谁更靠前。`label_rank_blend` 就是为了把“收益大小”和“当日相对排名”合到同一个连续监督目标里。

#### 特征输入

特征仍然来自本地候选池，不引入公开 Top-5 结果。最终参与训练的特征包括：

- 砖形图状态：`brick`、`brick_prev`、`brick_growth`、`brick_strength`、`prior_green_bars`、`red_risk_7`。
- 近期收益：`ret1`、`ret2`、`ret3`、`ret5`、`ret10` 以及对应 rank。
- 流动性和成交：`liquidity_rank`、`low_liq_rank`、`vol_rank`、`vol_ratio`、`turnover43`、`amplitude`。
- 趋势位置：`ma5`、`ma10`、`ma20`、`ma50`、`ma100`、`ma150`、`zxdq`、`zxdkx`。
- 日内标准化：`day_brick_z`、`day_ret1_z`、`day_vol_z`、`day_liq_z`、`day_price_z`。
- 交互项：`green_to_red`、`ret1_x_brick`、`ret1_x_vol`、`low_liq_x_brick`、`price_x_brick`。
- 趋势比值：`trend_zxdq_zxdkx`、`trend_ma50_100`、`trend_ma100_150`。
- 市场类型：`is_star`、`is_chinext`、`is_mainboard`、`code_bucket`。

这些特征仍然围绕视频里提到的几个方向：活跃市值波段内的强涨幅、砖形图反转、红砖强度、成交活跃、趋势位置、主板偏好。

#### 模型训练方法

第一版 HGB `label_blend` 模型收益只有 `+72.07%`，主要问题是普通均值目标会被大量小亏小赚样本稀释，而这个策略收益来源更像视频里说的：多数交易小亏小赚，少数大收益贡献主要利润。

所以这轮训练把重点放到尾部赢家上，做法是给 top10/top20 和正收益样本更高权重。训练脚本里实际跑了 6 个模型：

| 模型名 | 类型 | 训练标签 | 尾部权重思路 |
| --- | --- | --- | --- |
| `hgb_blend_w_tail` | HGB 回归 | `label_blend` | top10、top20、正收益、正收益幅度加权 |
| `hgb_return_w_tail` | HGB 回归 | `label_return` | 更重视高收益尾部 |
| `hgb_rank_w_tail` | HGB 回归 | `label_rank_blend` | 学每日相对赢家，兼顾收益和风险排名 |
| `hgb_top10_cls` | HGB 分类 | `label_top10` | 学是否进入当日 top10% |
| `hgb_top20_cls` | HGB 分类 | `label_top20` | 学是否进入当日 top20% |
| `et_blend_tail_light` | ExtraTrees 回归 | `label_blend` | 作为非 HGB 树模型对照 |

HGB 模型使用 `SimpleImputer(strategy='median')` 加 `HistGradientBoostingRegressor/Classifier`。主要参数集中在：

- `max_iter` 约 `220-260`。
- `learning_rate` 约 `0.032-0.035`。
- `max_leaf_nodes = 31`。
- `l2_regularization` 约 `0.04-0.08`。
- `random_state = 20260714`。

尾部样本权重的核心形式是：

```text
weight = 1
       + label_top20 * top20_weight
       + label_top10 * top10_weight
       + positive_return * positive_weight
       + clip(forward_return, 0, 0.25) * 20
```

这一步的直觉是让模型少关心“所有候选的平均误差”，多关心“谁会成为当天真正有弹性的尾部赢家”。

#### 模型层结果

从 `/tmp/quantx_brick_v11_forward_oos_tail_v1/tail_model_summary.json` 看，模型层最强的是 `hgb_rank_w_tail`：

| 分数列 | OOS Top-5 平均 forward_return | Top-5 中位数 | 正收益比例 | top10 命中率 | top20 命中率 | Top-5 退出原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hgb_rank_w_tail_score` | `0.012622` | `-0.013062` | `40.0000%` | `22.8902%` | `32.8324%` | `green 749`, `7red 116` |
| `hgb_top10_cls_score` | `0.010421` | `-0.015178` | `39.3064%` | `21.6185%` | `31.6763%` | `green 746`, `7red 119` |
| `hgb_top20_cls_score` | `0.007721` | `-0.012972` | `40.5780%` | `19.8844%` | `31.9075%` | `green 731`, `7red 134` |

这说明尾部加权有效提升了选出来的 OOS Top-5 平均 forward return。相比 19.8 的普通分位标签模型，top10/top20 命中率也明显提高。

但是单纯把 `hgb_rank_w_tail_score` 接入原执行配置，资金路径只有：

- 总收益：`+75.27%`。
- 最大回撤：`-29.39%`。

也就是说，模型层只小幅提高了候选质量，资金路径仍然没有接近公开 `+160%`。真正让收益接近和超过 `+160%` 的，是后续执行层参数收敛。

#### 执行层搜索

执行层搜索仍然使用同一份 `tail_oos_scores.parquet`，没有重新训练模型。搜索变量包括：

- `score_floor`：是否只买高于某个分数阈值的候选。
- `topk`：每日候选数量。
- `max_positions`：最大持仓数。
- `cash_use_ratio`：资金使用比例。
- `skip_if_holding`：已经持有同一股票时是否跳过新增买入。
- `rank_weights`：是否对排名靠前的候选给更高资金权重。

粗搜索结果显示，`score_floor` 是最关键的开关：

- 不加足够高的 `score_floor`，交易过多，收益接近原始 `+75%`。
- `score_floor` 附近在 `0.84` 时，收益可以提升到约 `+137.8%`。
- `score_floor` 继续抬到 `0.845` 以后收益快速恶化，说明阈值太高会过滤掉关键大赢家。

粗搜索中的代表性结果：

| 配置区域 | 总收益 | 最大回撤 | 买入数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `score_floor=0.84, topk=5, max_positions=5, top_heavy` | `+137.77%` | 约 `-20.51%` | `406` 左右 | 交易数接近公开，但收益仍低于 `+160%` |

随后 targeted search 聚焦在：

- `score_floor = 0.8385, 0.839, 0.8395, 0.84...`。
- `topk = 5`。
- `max_positions = 4, 5, 6`。
- `cash_use_ratio = 0.98, 0.995, 1.0`。
- `skip_if_holding = true/false`。
- `rank_weights = flat/top_heavy/very_top_heavy`。

targeted search 中途已经找到稳定的 `+180%` 区域，后续网格仍在跑但重复度很高，因此我在第 200 个 run 附近停止长任务，并从已经完成的所有 run 反查 `summary.json` 和 `config.yaml`，生成中断版汇总：

- `/tmp/quantx_brick_v11_forward_oos_tail_v1/interrupted_run_summary.json`。
- `/tmp/quantx_brick_v11_forward_oos_tail_v1/interrupted_run_summary.csv`。

这次汇总共读取到 `369` 个有效 run。

#### 当前最佳训练版配置

当前最佳临时配置：

`/tmp/quantx_brick_v11_forward_oos_tail_v1/best_tail_forward_label_oos_2025_2026.yaml`

它来自 run：

`/tmp/quantx_brick_v11_forward_oos_tail_v1/execution_targeted_search/runs/floor0p8385_pos4_cash0p995_skip1_flat`

核心参数：

- 分数表：`/tmp/quantx_brick_v11_forward_oos_tail_v1/tail_oos_scores.parquet`。
- 分数列：`score`，对应模型层最佳 `hgb_rank_w_tail_score`。
- `score_floor: 0.8385`。
- `topk: 5`。
- `max_positions: 4`。
- 仓位方式：`equal_weight`，不使用 `rank_weights`。
- `cash_use_ratio: 0.995`。
- `skip_if_holding: true`。
- 买入：T+1 open，买入滑点 `30bps`。
- 卖出：绿砖、7 红砖、最长 10 日，和前文规则一致。

资金路径结果：

- 初始资金：`1,000,000`。
- 期末净值：`2,841,313.03`。
- 总收益：`1.841313`，即约 `+184.1313%`。
- 年化收益：`0.989751`。
- 最大回撤：`-0.210510`。
- Sharpe：`3.054427`。
- 买入：`324`。
- 卖出：`324`。
- 加载股票数：`494`。

因此，从“收益是否能接近 160%”这个角度看，非蒸馏 forward-label 训练版已经可以通过尾部加权模型加执行层阈值搜索达到并超过公开收益目标。

但它和公开页面仍然不是同一个东西：

- 公开页目标最大回撤约 `-6.37%`，当前训练版约 `-21.05%`，风险路径差很多。
- 公开页买入约 `385` 笔，当前最佳训练版买入 `324` 笔，交易集合明显更窄。
- 当前最佳收益依赖 2025-2026 上的 `score_floor/max_positions/cash_use_ratio/skip_if_holding` 搜索，因此不是完全封闭的 OOS 结论。

更保守、交易数更接近公开的区域如下：

| 区域 | 总收益 | 最大回撤 | 买入数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `max_positions=5` 最佳 | `+141.25%` | `-21.17%` | `411` | 买入数接近公开，但收益低于 `+160%` |
| `max_positions=6` 最佳 | `+144.48%` | `-18.70%` | `480` | 回撤稍好，交易数明显更多 |
| `max_positions=4` 最佳 | `+184.13%` | `-21.05%` | `324` | 收益超过目标，但交易数更少，集中度更高 |

#### 这次优化说明了什么

这轮优化给出几个明确结论：

1. 只改训练 label，从普通 `label_blend` 改成尾部加权 `label_rank_blend`，可以提高候选质量，但不足以单独把资金路径从 `+72%` 拉到 `+160%`。
2. 收益跳升主要来自执行层的 `score_floor` 和 `max_positions`。它把交易集中到模型最高置信度区域，减少中低分候选的稀释。
3. `score_floor` 的可用区间很窄，`0.8385-0.84` 稳定有效，继续提高到 `0.845+` 会错过关键赢家。
4. 当前训练版已经可以复现“高收益尾部驱动”的收益特征，但没有复现公开页的低回撤路径。
5. 如果要把它当成研究结论，下一步必须做 walk-forward 参数选择，不能继续用 2025-2026 本身挑 `score_floor`。

所以当前最严谨的表述是：

> 非蒸馏 forward-label 训练版，通过 2013-2024 历史砖形图退出 label 训练尾部加权排序模型，再在 2025-2026 上执行层调参，已经能找到收益超过公开 `+160%` 的策略配置；但这个收益不是封闭 OOS，因为执行参数使用了 2025-2026 回测结果做选择，而且回撤和交易数没有对齐公开页面。

## 二十、2024 前训练到 2025-2026 的主口径复核

用户指出一个关键问题：如果原策略声称用 2024 年之前的数据训练，那么最应该先验证的不是 `2020` 前训练到今天，而是“`<=2024` 训练，`2025-2026` 严格 OOS 回测”。同时，Brick/强势股策略不应该把普通 `open / preclose - 1 >= 3%` 高开一刀切过滤，因为高开本身可能就是弱转强或强势延续的一部分。

因此补做一轮主口径复核：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/run_brick_train2024_oos.py
.tmp/quantx-research/brick-pre2020-oos-v1/scores_train2024_oos_2025_2026.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/backtest_config_train2024_oos.yaml
.tmp/quantx-research/brick-pre2020-oos-v1/result_train2024_oos.json
.tmp/quantx-research/brick-pre2020-oos-v1/runs/brick_train2024_forward_label_oos_top5_maxpos10/
```

这轮复用前面构建好的候选/动态退出标签文件，不重新读取全量行情：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/candidates_with_labels.parquet
```

严格边界：

1. 训练样本只使用 `signal_date <= 2024-12-31` 且动态退出 `exit_date <= 2024-12-31` 的候选行。
2. `2025-01-02` 到 `2026-07-15` 只用于模型打分、QuantX 账户回测和事后诊断。
3. 不使用公开 Top5 teacher label。
4. 不使用 2025-2026 搜索得到的 `score_floor / max_positions / cash_use_ratio / skip_if_holding` 等执行参数。
5. 主口径不启用 `open_gap >= 3%` 硬过滤；高开买入只作为诊断项记录。
6. 仍保留真实不可交易过滤：涨停/跌停拒单、停牌、缺行情、零成交、一字板等由 QuantX 交易规则和事后审计覆盖。

模型和执行：

- 模型：HGB `label_rank_blend`，训练权重仍偏向历史 top10/top20、正收益和右尾收益。
- Selector：每日 external score Top5，无 OOS score floor。
- 仓位：`max_positions=10`，`cash_use_ratio=0.98`，T+1 open，买入滑点 `30bps`。
- 卖出：绿砖、7 红、最长 10 日。

标签层 OOS 诊断：

| 指标 | 数值 |
| --- | ---: |
| train rows | 168,549 |
| OOS rows | 16,204 |
| OOS signal dates | 184 |
| OOS Top5 rows | 856 |
| train mean forward_return | -0.001329 |
| OOS Top5 mean forward_return | -0.001500 |
| OOS Top5 median forward_return | -0.010077 |
| OOS Top5 positive ratio | 32.71% |
| 2025 Top5 mean forward_return | +0.001055 |
| 2026 Top5 mean forward_return | -0.006841 |

完整 QuantX 回测结果：

| 指标 | 数值 |
| --- | ---: |
| 区间 | 2025-01-02 至 2026-07-15 |
| 初始资金 | 1,000,000 |
| final value | 792,528.48 |
| total return | -20.75% |
| annual return | -14.09% |
| max drawdown | -33.21% |
| Sharpe | -0.665 |
| buys / sells | 725 / 725 |
| 平均持仓数 | 5.89 |
| 持仓周期均值 / 中位数 | 3.46 / 3.00 天 |

年度收益：

| 年份 | 收益 |
| --- | ---: |
| 2025 | +5.59% |
| 2026 YTD | -24.64% |

交易异常审计：

| 项目 | 数值 |
| --- | ---: |
| effective buys | 725 |
| 当前 ST/退市名买入 | 0 |
| 缺失行情买入 | 0 |
| 零成交量/成交额买入 | 0 |
| 一字涨停买入 | 0 |
| 开盘高开 >= 3% 买入 | 6 |
| unique buy symbols | 525 |
| reject rows | 1 |

这里 `open_gap_ge_3pct=6` 只作为诊断，不作为主口径违规，因为本轮明确不把普通高开视为不可交易。真正不可交易项为 0。

### 20.1 判定

状态：`failed_strict_oos_main_replay`

这轮是当前最接近“2024 年之前训练、2025-2026 OOS 回测”的封闭主口径，但没有复现公开高收益。2025 只有小幅正收益，2026 明显失效；标签层 Top5 均值也为负。失败不是因为 `open_gap >= 3%` 过滤，因为主口径没有启用该过滤。

### 20.2 反事实分析

第一反事实：如果当前 forward-label 特征和模型已经接近公开策略的 AI 打分，那么 `<=2024` 训练后，2025-2026 的裸 Top5 至少应有明显正收益。实际账户总收益为 `-20.75%`，标签层 Top5 mean forward_return 为 `-0.001500`，不支持。

第二反事实：如果差距主要来自普通高开过滤，取消高开过滤后收益应显著恢复。实际本轮取消高开硬过滤后，仍然没有复现公开收益；高开不是主要矛盾。

第三反事实：如果 2025-2026 上的执行层阈值搜索只是锦上添花，裸 Top5 应该已经有较高 alpha。实际裸 Top5 不成立，说明前文 `+184%` 更多来自 OOS 区间执行阈值搜索和单段风格收敛，不能当封闭 OOS 结论。

第四反事实：如果 Brick/活跃市值/红砖候选土壤在 2026 仍有效，2026 label 层不应系统为负。实际 2026 Top5 mean forward_return 为 `-0.006841`，账户收益 `-24.64%`，说明当前候选和模型在 2026 反向。

当前结论：Brick 方向仍然是有价值的方法启发，包括动态退出 label、hard-negative/尾部加权、external score 接入和强势候选建模；但当前本地 forward-label 版本没有学到公开策略的核心 2025-2026 选股能力。若继续研究 Brick，下一步不能继续在 2025-2026 上调 `score_floor`，而应寻找缺失的真实信息层，例如更准确的活跃市值指标、点位/盘口确认、真实题材/概念 point-in-time 归属，或复核 BrickChart/候选池与公开策略的公式保真度。

## 二十一、strategy-renko 候选公式口径复核

用户补充了公开 screener 的策略公式：

```text
砖型图的绿砖后面出现反转红砖
```

这句话不能直接等价为本地当前特征里的 `brick > 0 and brick_prev < 0`。关键原因是 QuantX 当前 `BrickChart(...)` 算子返回的是砖高度的一阶差分，而公开 screener 在 2026-07-03 以后暴露的 `factor_values.brick` 更像砖高度本体。

本轮新增临时审计脚本和产物：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/analyze_raw_brick_alignment.py
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_alignment_report.json
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_alignment_rows.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_candidate_daily.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_rule_probe.json
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_candidate_rule_eval.json
```

脚本复用 QuantX `BrickChart` 内部公式，但额外返回 `raw = max(var5a - var2a - t, 0)` 这个砖高度本体，再构造：

- `raw_brick`：本地砖高度。
- `raw_prev`：前一日砖高度。
- `raw_delta = raw_brick - raw_prev`：当日砖高度变化。
- `raw_decline_sum_prev5 / incl5`：近 5 日砖高度下行总和。
- `raw_max_consec_down_prev5 / incl5`：近 5 日最大连续下行天数。

### 21.1 官方字段一致性

公开 API `strategy-renko/runs/{id}?kline=true` 在 2026-07-03 以后暴露了完整 `factor_values`。这些完整字段共 `571` 个本地可匹配样本，检查结果如下：

| 检查项 | 结果 |
| --- | ---: |
| `official_brick_delta_1 == official_brick - official_prev_brick` | 100.00% |
| `official_brick_delta_1 == official_reversal_h` | 100.00% |
| `official_brick_delta_1 > 0` | 100.00% |
| `official_brick_decline_sum_5d > 0` | 100.00% |
| `official_brick_max_consec_down_5d >= 1` | 100.00% |

因此公开公式更准确的机器表述应是：过去一段砖高度有下行，也就是绿砖/走弱段；当日砖高度从前一日低位向上反转，`brick - prev_brick > 0`，形成红砖反转高度 `reversal_h`。

### 21.2 本地 raw brick 与官方字段的关系

本地 raw 高度与官方完整字段的相关性：

| 字段关系 | 相关性 |
| --- | ---: |
| `official_brick` vs `raw_brick` | 0.8490 |
| `official_prev_brick` vs `raw_prev` | 0.8780 |
| `official_brick_delta_1` vs `raw_delta` | 0.5098 |
| `official_reversal_h` vs `raw_delta` | 0.5098 |
| `official_decline_sum_5d` vs `raw_decline_sum_incl5` | 0.8651 |
| `official_day_ret` vs local `ret1` | 0.9863 |
| `official_close` vs local close | 0.99997 |

这说明本地 raw brick 大方向接近官方，但仍不是完全同源：高度和下行累计对得较好，单日反转高度只中等相关。部分股票在本地 `raw_delta` 为负时，官方 `brick_delta_1` 已经为正，说明数据版本、复权、参数或公式细节仍有差异。

### 21.3 候选池覆盖实验

用 36 个公开候选池日，先从全 A 股生成本地候选。仅使用 `raw_delta > 0` 会覆盖官方大部分候选，但池子过大：

| 本地候选规则 | TP | FP | FN | Precision | Recall | F1 | 平均每日候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_delta > 0` | 3,672 | 77,042 | 295 | 4.55% | 92.56% | 8.67% | 2,242 |
| `raw_delta > 0` 且近 5 日有下行 | 3,550 | 66,134 | 417 | 5.09% | 89.49% | 9.64% | 1,936 |
| 昨日下行、今日上行 | 3,211 | 28,771 | 756 | 10.04% | 80.94% | 17.86% | 888 |

然后根据单条件和浅层树做可解释收缩。最有解释力的硬条件是：`raw_brick` 足够高、`ret1` 足够强、以及部分连续下行后的强反转。一个可解释规则：

```text
base = raw_delta_prev < 0 and raw_delta > 0

base and (
  raw_brick > 61.14 and ret1 > 0.04
  or raw_brick > 61.14 and ret1 <= 0.04 and raw_max_consec_down_prev5 > 3.5 and ret1 > 0.01
  or raw_brick > 61.14 and ret1 <= 0.04 and raw_max_consec_down_prev5 <= 3.5 and raw_brick > 105.24
  or 53.19 < raw_brick <= 61.14 and ret1 > 0.05 and raw_delta <= 6.71
)
```

该规则在 36 日上的覆盖：

| 区间 | TP | FP | FN | Precision | Recall | F1 | 平均每日候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 36 日 | 2,828 | 4,965 | 1,139 | 36.29% | 71.29% | 48.10% | 216 |
| 2026-05-20 至 2026-07-02 | 2,430 | 3,700 | 966 | 39.64% | 71.55% | 51.02% | 227 |
| 2026-07-03 至 2026-07-15 | 398 | 1,265 | 173 | 23.93% | 69.70% | 35.63% | 185 |

另一个更简单但召回较低的规则：

```text
raw_delta_prev < 0 and raw_delta > 0 and raw_brick > 61.14 and ret1 > 0.04
```

全区间 precision `39.86%`，recall `58.81%`，平均每日候选 `163`。

### 21.4 判定

状态：`candidate_formula_partially_aligned_not_mergeable`

这次复核证明了两个问题：

1. 之前的本地 forward-label 严格 OOS 失败，很大一部分来自候选池一阶口径错配。当前 QuantX `BrickChart` 返回差分，不能直接代表官方页面里的 brick 高度本体。
2. raw brick 公式可以显著解释公开候选池，把粗反转池从约 `888` 只/天压到 `160-220` 只/天，同时保留 `59%-71%` 的召回，但还没有达到可合入正式策略的保真度。

下一步不能直接训练模型或接入 milestone。必须继续做候选池对齐：

1. 对 `n/m1/m2/m3/t/shift1/shift2` 做小范围参数扫描，目标先提高官方完整字段的 `brick/raw_brick` 和 `delta/raw_delta` 一致性。
2. 引入公开 screener 暴露的同日字段做公式约束，例如成交额、价格、振幅、`ret1`、`trend_short/long`、`dist_from_20d_high_pct`，判断哪些是候选硬过滤，哪些只是 v11 排序特征。
3. 候选池保真度达标后，再做非蒸馏 forward-label 训练；否则模型会继续学习错误候选土壤。

### 21.5 三线翻红公式与 n=5 复核

用户进一步提供了一个通达信三线翻红/翻绿公式：

```text
N:=3;
A3:=REF(C,N);
突破:=C>HHV(A3,N);
破位:=C<LLV(A3,N);
三线翻红:BARSLAST(突破)<BARSLAST(破位) AND C>O,COLORRED;
三线翻绿:BARSLAST(突破)>BARSLAST(破位) AND C<O,COLORGREEN;
```

本轮用脚本验证该公式：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/analyze_tdx_three_line_formula.py
.tmp/quantx-research/brick-pre2020-oos-v1/tdx_three_line_formula_alignment.json
.tmp/quantx-research/brick-pre2020-oos-v1/tdx_three_line_formula_rows.parquet
```

验证范围为公开 `strategy-renko` 36 个候选池日，全 A 股同日候选覆盖。结果显示它不像公开候选池的主公式：

| 规则 | TP | FP | FN | Precision | Recall | F1 | 平均每日候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `N=3 三线翻红` | 2,494 | 35,015 | 1,473 | 6.65% | 62.87% | 12.03% | 1,042 |
| `N=3 前一日翻绿信号后翻红` | 702 | 3,948 | 3,265 | 15.10% | 17.70% | 16.29% | 133 |
| `N=2 前一日翻绿信号后翻红` | 1,335 | 7,435 | 2,632 | 15.22% | 33.65% | 20.96% | 251 |

三线公式可以产生“红/绿状态”，但覆盖模式与公开候选池不一致：纯红状态过宽，绿后红反转又召回过低。因此它更像另一个看图用的砖线/三线转向指标，不是公开 `strategy-renko` 的筛选主公式。

同时补做了强度公式单参数探针和 `n=5` 全市场复核：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_single_param_probe.csv
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_alignment_report_n5.json
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_candidate_daily_n5.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/raw_brick_candidate_rule_eval_n5.json
```

`n=5,m1=3,m2=12,m3=12,t=8,shift1=92,shift2=114` 相比原 `n=8`，字段层明显更接近官方：

| 字段关系 | n=8 | n=5 |
| --- | ---: | ---: |
| `official_brick` vs local `raw_brick` | 0.8490 | 0.8347 |
| `official_prev_brick` vs local `raw_prev` | 0.8780 | 0.8821 |
| `official_reversal_h` vs local `raw_delta` | 0.5098 | 0.7659 |
| `official_decline_sum_5d` vs local `raw_decline_sum_incl5` | 0.8651 | 0.9181 |
| `official_range_from_5d_min` vs local range | 0.6320 | 0.7650 |

但候选池粗规则没有同步改善：

| 本地候选规则 | n=8 F1 | n=5 F1 | n=5 Precision | n=5 Recall | n=5 平均每日候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_delta > 0` | 8.67% | 8.72% | 4.57% | 96.40% | 2,325 |
| `raw_delta > 0` 且近 5 日有下行 | 9.64% | 9.47% | 4.98% | 95.31% | 2,109 |
| 昨日下行、今日上行 | 17.86% | 17.28% | 9.65% | 82.96% | 948 |

加入可解释硬过滤后，`n=5` 的较好规则为：

```text
raw_delta_prev < 0
and raw_delta > 0
and raw_brick > 60
and ret1 > 0.04
and amount_rank >= 0.5
```

该规则 precision `39.80%`，recall `58.10%`，F1 `47.24%`，平均每日候选约 `161` 只。另一个 `raw_brick > 65 and ret1 > 0.04` 规则 precision `38.80%`，recall `60.32%`，F1 `47.22%`，平均每日候选约 `171` 只。

阶段性判断曾是：官方 `factor_values.brick/reversal_h` 更像 `n=5` 附近的强度公式。但后续 K 线页前端 chunk 暴露了更准确的绘图公式，证明 `n=5` 只是近似，不应再作为 Brick 主复现口径。公开候选池仍不是纯 brick 公式，至少还有当日涨幅、砖高度强度、反转增量、成交额或官方股票池限制等排序前硬过滤；不能把三线翻红公式当主路径继续推进。

### 21.6 K 线页前端公式复核：官方 brick 本体已对齐

用户补充了 K 线页：

```text
https://touzikexue.com/screener/kline?strategy=strategy-renko&symbol=002485&runId=1539&dataDate=2026-07-15&sortKey=score&sortDir=desc
```

前端 chunk `touzikexue_useKLineBars-CyArTqP-.js` 中直接暴露了 K 线工作台使用的砖形图计算函数。公式等价为：

```text
HHV4 = 最近 4 根 K 线最高价
LLV4 = 最近 4 根 K 线最低价
RANGE = HHV4 - LLV4

var2 = SMA((HHV4 - close) / RANGE * 100 - 90, 4, 1)
var4 = SMA((close - LLV4) / RANGE * 100, 6, 1)
var5 = SMA(var4, 6, 1)

brick = max(0, var5 - var2 - 4)
```

对应临时脚本和产物：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/analyze_frontend_brick_formula.py
.tmp/quantx-research/brick-pre2020-oos-v1/frontend_brick_alignment_report_frontend.json
.tmp/quantx-research/brick-pre2020-oos-v1/frontend_brick_alignment_rows_frontend.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/frontend_brick_candidate_daily_frontend.parquet
```

先用官方 `daily-bars` 接口验证单票。`002485` 最近 200 根 K 线中，本地按前端公式复算的 `brick` 与官方 `daily-bars[].brick` 最大绝对误差约 `0.0013`。例如 2026-07-14 官方 `brick=81.5606`，2026-07-15 官方 `brick=86.7362`，前端公式可直接复算到 1e-4 量级。

再用本地 qlib 全市场行情复算 36 个公开候选池日，完整字段对齐显著高于旧 `raw/n=5` 口径：

| 字段关系 | 相关性 |
| --- | ---: |
| `official_brick` vs `frontend_brick` | 0.9975 |
| `official_prev_brick` vs `frontend_prev` | 0.9998 |
| `official_delta/reversal_h` vs `frontend_delta` | 0.9504 |
| `official_decline_sum_5d` vs `frontend_decline_sum_incl5` | 0.9994 |
| `official_range_from_5d_min` vs `frontend_range` | 0.9718 |
| `official_day_ret` vs local `ret1` | 0.9863 |
| `official_close` vs local close | 0.99997 |

因此最新结论是：官方 brick 本体已经基本对齐，旧 `n=5` 结果只是公式近似。后续 Brick 复现必须以 `frontend_brick` 作为高度本体，以 `frontend_delta = frontend_brick - REF(frontend_brick,1)` 作为反转高度。

### 21.7 官方候选池硬过滤反推

在真实前端公式上构造核心候选：

```text
candidate_core = REF(frontend_delta, 1) < 0 and frontend_delta > 0
```

它几乎覆盖官方候选，但池子仍过宽：

| 规则 | TP | FP | FN | Precision | Recall | F1 | 平均每日候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frontend_delta > 0` | 3,964 | 80,836 | 3 | 4.67% | 99.92% | 8.93% | 2,356 |
| `REF(frontend_delta,1)<0 and frontend_delta>0` | 3,958 | 27,600 | 9 | 12.54% | 99.77% | 22.28% | 877 |
| `frontend_delta>0 and 近5日有下行` | 3,964 | 75,567 | 3 | 4.98% | 99.92% | 9.49% | 2,209 |

新增硬过滤探针：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/probe_frontend_candidate_filters.py
.tmp/quantx-research/brick-pre2020-oos-v1/frontend_candidate_hard_filter_probe.json
```

在 `candidate_core` 内，区分官方候选与非官方候选的主轴非常清楚：

| 特征 | AUC |
| --- | ---: |
| `frontend_brick` | 0.8660 |
| `brick_rank_pct` | 0.8643 |
| `ret1_rank_pct` | 0.8386 |
| `ret1` | 0.8338 |
| `amplitude_rank_pct` | 0.8227 |
| `body_pct` | 0.8158 |
| `$amount` | 0.8140 |
| `amount_rank_pct` | 0.8105 |
| `frontend_delta` | 0.7319 |

一个简单三条件规则已经显著接近官方候选土壤：

```text
candidate_core
and brick_rank_pct >= 0.60
and frontend_delta >= 3
and amount_rank_pct >= 0.60
```

全 36 日结果：TP `2,562`，FP `2,273`，FN `1,405`，precision `52.99%`，recall `64.58%`，F1 `58.21%`，平均每日候选约 `134`。

更偏召回的版本：

```text
candidate_core
and brick_rank_pct >= 0.60
and frontend_delta >= 3
and amount_rank_pct >= 0.50
```

全 36 日结果：TP `2,786`，FP `2,831`，FN `1,181`，precision `49.60%`，recall `70.23%`，F1 `58.14%`，平均每日候选约 `156`。

浅层树用于解释硬过滤，不作为正式模型。depth=6 在候选池上可达到 precision `57.83%`、recall `71.09%`、F1 `63.78%`、平均每日候选约 `135`。树的主分裂仍围绕 `frontend_brick`、`ret1_rank_pct`、`$amount/amount_rank_pct`、`frontend_delta`、`frontend_range_from_5d_min`。

逐日稳定性仍不足。固定三条件规则在 2026-07-02、2026-07-06、2026-07-15 等日期召回偏低，说明官方候选池不是单一阈值公式，而更像“核心反转 + 强度/成交硬过滤 + 若干低 brick 但高 ret/高成交的补充分支”。下一步应继续复现官方 score/branch，而不是把当前三条件规则合入正式策略。

当前状态更新为：`frontend_brick_aligned_candidate_filter_partially_aligned_not_mergeable`。

### 21.8 官方 score/排序可解释性复核

在候选池仍未完全对齐的前提下，补做官方候选内部 score 分析。目标不是用公开 Top5 交易明细做蒸馏，而是判断公开 screener 候选里的 `score/rank` 是否能由页面暴露字段或本地可计算字段解释。

对应临时脚本和产物：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/analyze_official_score_explainability.py
.tmp/quantx-research/brick-pre2020-oos-v1/official_score_explainability_report.json
.tmp/quantx-research/brick-pre2020-oos-v1/official_score_explainability_rows.parquet
```

公开候选池 36 日共 `4,008` 行，其中 2026-07-03 之后 9 日、`582` 行暴露了 `factor_values.v11_score/v11_rank`。字段检查显示：

| 检查项 | 结果 |
| --- | ---: |
| `score` vs `v11_score` 相关性 | 1.0000 |
| `max(abs(score-v11_score))` | 0.00028 |
| `rank == v11_rank` | 100.00% |

也就是说，7/3 之后页面展示的 `score/rank` 本身就是 `v11_score/v11_rank`，不是需要额外猜测的隐藏排序结果。

但 7/3 前后的 score 口径不是同一个稳定量纲：

- 5/20-6/17：score 多在 `-18` 到 `8` 附近。
- 6/18-7/03：很多日期 score 是 `0-100` 的 rank-like 尺度。
- 7/06 以后：`v11_score` 变成约 `-73` 到 `104` 的模型分数。

因此不能把全 36 日简单混成一个统一监督目标。用日期切分做 OOS 可解释性验证，结果如下：

| 区间 | 模型 | Mean Daily Spearman | Mean Daily Top5 Recall | 结论 |
| --- | --- | ---: | ---: | --- |
| 7/3 后 v11 暴露段 | Ridge | 0.8635 | 75.00% | 可见字段能较好解释官方排序 |
| 7/3 后 v11 暴露段 | ExtraTrees | 0.8540 | 70.00% | 非线性模型也能解释，但未明显优于 Ridge |
| 7/3 前非 v11 段 | Ridge | 0.1001 | 10.00% | 不可解释，疑似 score 口径切换/字段缺失 |
| 7/3 前非 v11 段 | ExtraTrees | 0.1021 | 10.00% | 不可解释 |
| 全 36 日混训 | ExtraTrees | 0.4048 | 23.08% | 混合口径会显著劣化 |

7/3 后 `v11_score` 的重要特征集中在趋势离差、当日涨幅、5 日动量、200 日振幅、换手率、价格/均线、上影线、MACD、跳空等字段。它更像一个规则/线性模型或浅层模型输出，而不是只能通过深度模型才能解释的黑箱。

当前判断：

1. 后段 `v11_score` 可作为公开候选内部排序的校准目标，但样本只有 9 天，不能直接当成 robust 策略训练标签。
2. 前段 score 与后段不是同一口径，不能混训；如果用公开 score 做复现，必须按版本分段处理。
3. 对策略研究更有价值的路径仍是：用 `frontend_brick` 构造真实候选土壤，再用本地未来收益/风险标签做非蒸馏 pairwise/ranking 训练；公开 `v11_score` 只用于 sanity check 排序方向，不用于最终收益结论。

### 21.9 非蒸馏严格 OOS baseline：公式对齐但 alpha 不成立

在 `frontend_brick` 本体已经对齐后，补跑了一个不使用公开 Top5 交易、公开 score/rank 的严格 OOS baseline。训练期只用 2016-2024，本地未来收益/风险标签；OOS 为 2025-01-02 至 2026-07-15。

对应临时脚本和产物：

```text
.tmp/quantx-research/brick-pre2020-oos-v1/run_frontend_brick_oos_baseline.py
.tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline/frontend_candidates_with_labels.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline/frontend_oos_scores.parquet
.tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline/frontend_oos_baseline_result.json
.tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline/frontend_filter_grid_v1.json
.tmp/quantx-research/brick-pre2020-oos-v1/frontend-oos-baseline/frontend_regime_probe_v1.json
```

候选池定义：

```text
core = REF(frontend_delta, 1) < 0 and frontend_delta > 0

branch =
  (brick_rank_pct >= 0.60 and frontend_delta >= 3 and amount_rank_pct >= 0.50)
  or (ret1_rank_pct >= 0.90 and amount_rank_pct >= 0.60 and frontend_brick >= 52)

candidate = core and branch
  and volume/amount > 0
  and amplitude_pct < 18%
  and not current ST/delisting name
  and not ST-like 5% limit history proxy
  and not one-price limit-up on signal day
```

训练标签严格按 T 日信号、T+1 open 买入、最多持有 7 日构造；若 T+1 一字涨停、零成交/零成交额、或开盘高开 >= 3%，标签惩罚为不可交易。模型为 `HistGradientBoostingRegressor`，只学习训练期同日候选内部的收益/风险 rank。

回测执行同样显式过滤一字涨停和 T+1 高开 >= 3%，并加入 7% 止损、10% 后 6% 回撤止盈、7 日最大持仓。结果：

| 指标 | 结果 |
| --- | ---: |
| OOS 总收益 | 1.59% |
| 年化收益 | 1.03% |
| 最大回撤 | -22.03% |
| 2025 收益 | 0.01% |
| 2026 收益 | -2.29% |
| 平均持仓数 | 9.91 |
| 平均持仓天数 | 8.73 |
| 有效买入 | 640 |
| 当前 ST/退市名买入违规 | 0 |
| 缺失行情买入违规 | 0 |
| 零成交/零成交额买入违规 | 0 |
| 一字涨停买入违规 | 0 |
| 高开 >= 3% 买入违规 | 0 |

标签层更能说明问题：

| 区间 | 候选均值 | 候选胜率 | 模型 Top7 均值 | 模型 Top7 胜率 |
| --- | ---: | ---: | ---: | ---: |
| 2016-2024 train | -1.24% | 31.50% | - | - |
| 2025-2026 OOS | -1.34% | 31.46% | -0.59% | 34.84% |
| 2025 Top7 | - | - | -0.50% | - |
| 2026 Top7 | - | - | -0.78% | - |

这不是“模型没调好”这么简单。候选池本身在训练期和 OOS 都是负期望，模型只能把亏损程度从约 `-1.3%` 缩小到 `-0.6%`，不能把一个负候选土壤变成高收益策略。

### 21.10 失败归因：反转红砖更像追高后的均值回撤

对全部候选按 OOS 横截面特征做分层，方向非常一致：高砖高度、高反转增量、高当日涨幅、高 3/5 日涨幅、高振幅、高实体、高收盘位置、高均线乖离的分层更差。代表性结果：

| OOS 分层 | 低分位均值 | 高分位均值 | 观察 |
| --- | ---: | ---: | --- |
| `frontend_brick` 五分位 | -0.84% | -2.22% | 砖高度越极端越差 |
| `ret1` 五分位 | -0.63% | -3.65% | 当日涨幅越高越像追高回撤 |
| `ret5` 五分位 | -1.02% | -3.03% | 近期涨幅高的后续更差 |
| `amplitude_pct` 五分位 | -0.58% | -2.73% | 高振幅候选更差 |
| `body_pct` 五分位 | -0.60% | -3.25% | 大实体更差 |
| `close_pos` 五分位 | -0.62% | -3.65% | 收在高位更差 |
| `ma50_rel` 五分位 | -0.39% | -2.75% | 均线乖离越高越差 |

用训练期选择的“反追高”排序，例如低 `ret1`、低 `brick`、低 `amplitude`、低 `close_pos`、低 `ma50_rel`，只能把 OOS Top7 均值改善到约 `-0.45%` 至 `-0.52%`，仍为负。进一步用候选拥挤度、候选日动量、科创/创业占比、候选整体均线乖离等当日 regime 变量做门控，训练期也没有稳定正期望；OOS 仅少数 2025 子段略正，2026 再次失效。

当前结论：`frontend_brick` 公式和可交易审计已经对齐，但公开“绿砖后反转红砖”这条候选土壤在严格可交易约束下不是 robust alpha。后续不应继续围绕该候选池微调仓位、阈值或模型复杂度；更合理的方向是把它当作一个“追高风险反例”，转向全市场横截面寻找真正正期望的候选土壤，再考虑 pairwise / contrastive learning 排序。
