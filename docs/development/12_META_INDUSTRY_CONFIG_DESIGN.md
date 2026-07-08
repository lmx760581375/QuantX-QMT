# Meta And Industry Config Design

本文档定义 QuantX 的股票名称、行业、板块、概念等非 Qlib 标准数据的设计方案，以及这些元数据如何进入 config-driven 策略语言。

重点目标：

```text
股票名称、行业、板块等元数据独立存储。
策略 config 可以直接引用行业做过滤、分组、排序、行业因子计算和交叉选股。
第一版使用最新静态行业快照，不强做历史行业 as-of。
```

## 1. 背景和边界

当前 Qlib 数据层适合存储数值时序数据：

```text
open/high/low/close/volume/vwap/change/factor
```

但股票名称、行业、概念、板块、指数成分不是标准行情字段，直接塞进 Qlib bin 有几个问题：

1. 字符串和分类数据不适合 Qlib feature bin。
2. 行业、概念需要多源更新和覆盖率检查。
3. 策略需要按行业分组排序，不只是简单读取字段。
4. 可视化需要名称和行业，但不应该反向影响行情数据层。

因此新增一个 `MetaStore`，和 Qlib 数据并行存在：

```text
Qlib provider_uri
  -> 高速数值时序数据

MetaStore
  -> 股票名称、行业、板块、概念、指数成分、自定义 universe
```

## 2. 本轮原则

### 2.1 行业使用最新快照

第一版行业分类使用最新快照：

```text
industry metadata uses latest snapshot by default.
```

原因：

1. 行业分类变化频率低。
2. 历史行业分类很难稳定获取。
3. 强制 as-of 会显著增加系统复杂度。
4. 当前目标是研究迭代和策略表达能力，而不是先做最严格的历史行业回放。

需要在回测报告 `explain.json` 中记录：

```json
{
  "meta": {
    "snapshot": "latest",
    "snapshot_date": "2026-07-05",
    "industry_source": "akshare_eastmoney",
    "lookahead_note": "industry metadata uses latest snapshot"
  }
}
```

这样收益解释中明确知道行业使用的是最新分类，避免误以为是历史 as-of。

### 2.2 不把 ST 放进 MetaStore 主方案

不建议在 `MetaStore` 中加入：

```text
is_st
suspended
limit_up
limit_down
tradable
```

这些字段时效性很强，应该属于后续的 `DailySecurityStatus`，按交易日存储和查询。

当前 MetaStore 只处理相对稳定的主数据：

```text
name
exchange
board
industry
concept
index constituent
custom universe
```

## 3. 数据源选择

当前本机环境已经安装：

```text
baostock: True
akshare: True
tushare: True
duckdb: False
```

第一版推荐：

1. 股票名称：优先 BaoStock `query_all_stock`，缺失时用 AkShare/Tushare 补齐。
2. 行业分类：优先 AkShare 东方财富行业接口。
3. 概念/板块：优先 AkShare 东方财富概念接口。
4. Tushare 作为可选源，需要 token，不作为默认依赖。
5. 存储第一版用 SQLite 或 CSV，不强依赖 DuckDB。

AkShare 本机已确认存在的函数：

```text
stock_individual_info_em
stock_board_industry_name_em
stock_board_industry_cons_em
stock_board_concept_name_em
stock_board_concept_cons_em
```

这些接口可以支持：

```text
行业列表
行业成分股
概念列表
概念成分股
个股基础信息
```

### 3.1 当前接口稳定性验证结果

本机在 `conda test` 环境中已验证包可导入：

```text
baostock: True
akshare: True
tushare: True
duckdb: False
```

但当前网络下在线源不稳定：

```text
AkShare stock_board_industry_name_em:
  failed: Remote end closed connection without response

AkShare stock_board_concept_name_em:
  failed: Remote end closed connection without response

AkShare stock_individual_info_em:
  failed: Remote end closed connection without response

BaoStock login:
  failed: 网络接收错误
```

因此第一版实现必须支持离线 CSV 导入作为稳定主路径，在线接口只作为可选更新源：

```bash
conda run -n test python -m quantx.tools.update_meta --probe --json
```

如果 probe 失败，系统不得生成假行业数据，必须明确返回失败原因。

推荐当前落地顺序：

```text
1. 先用 CSV/公司内部数据导入 security_master 和 industry_membership。
2. AkShare/BaoStock 作为可选在线源，成功时更新，失败时只汇报。
3. 后续如果找到稳定内网源，再实现新的 MetadataSource adapter。
```

## 4. 存储结构

目录：

```text
data/meta/
  quantx_meta.sqlite
  snapshots/
    security_master_20260705.csv
    industry_membership_20260705.csv
    sector_membership_20260705.csv
  manual_overrides/
    security_master_overrides.csv
    industry_overrides.csv
```

### 4.1 security_master

```sql
CREATE TABLE security_master (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    board TEXT,
    list_date TEXT,
    delist_date TEXT,
    source TEXT,
    snapshot_date TEXT,
    updated_at TEXT
);
```

说明：

```text
symbol: Qlib/QuantX 统一格式，例如 SH600000。
name: 股票名称，例如 浦发银行。
board: mainboard / chi_next / star / beijing。
source: baostock / akshare / tushare / manual。
```

### 4.2 industry_membership

第一版只存最新行业：

```sql
CREATE TABLE industry_membership (
    symbol TEXT,
    industry_source TEXT,
    level INTEGER,
    industry_code TEXT,
    industry_name TEXT,
    source TEXT,
    snapshot_date TEXT,
    updated_at TEXT,
    PRIMARY KEY (symbol, industry_source, level)
);
```

示例：

```text
SH600000, eastmoney, 1, BK0475, 银行
SZ000001, eastmoney, 1, BK0475, 银行
```

后续如需历史 as-of，可以扩展：

```sql
valid_from TEXT
valid_to TEXT
```

但第一版不要求。

### 4.3 sector_membership

概念、主题、指数成分统一放这里：

```sql
CREATE TABLE sector_membership (
    symbol TEXT,
    sector_type TEXT,
    sector_code TEXT,
    sector_name TEXT,
    weight REAL,
    source TEXT,
    snapshot_date TEXT,
    updated_at TEXT,
    PRIMARY KEY (symbol, sector_type, sector_code)
);
```

示例：

```text
SH600000, index, SH000300, 沪深300
SZ300750, concept, BK0964, 锂电池
```

### 4.4 symbol_map

```sql
CREATE TABLE symbol_map (
    symbol TEXT PRIMARY KEY,
    baostock_code TEXT,
    akshare_code TEXT,
    tushare_code TEXT,
    qlib_code TEXT
);
```

用于多数据源代码转换：

```text
SH600000 <-> sh.600000 <-> 600000.SH
SZ000001 <-> sz.000001 <-> 000001.SZ
```

## 5. 新增模块设计

```text
quantx/core/data/meta/
  __init__.py
  store.py
  service.py
  sources/
    baostock_meta.py
    akshare_meta.py
    tushare_meta.py
    manual_meta.py
  resolver.py

quantx/tools/update_meta.py
```

### 5.1 MetaStore

```python
class MetaStore:
    def __init__(self, uri: str = "data/meta/quantx_meta.sqlite"):
        ...

    def upsert_security_master(self, frame: pd.DataFrame) -> None:
        ...

    def upsert_industry_membership(self, frame: pd.DataFrame) -> None:
        ...

    def upsert_sector_membership(self, frame: pd.DataFrame) -> None:
        ...

    def get_symbol_meta(self, symbols: list[str]) -> pd.DataFrame:
        ...

    def get_industry(self, symbols: list[str], source: str = "eastmoney", level: int = 1) -> pd.DataFrame:
        ...
```

返回格式：

```text
index: symbol
columns: name, exchange, board, industry_name, industry_code
```

### 5.2 MetaUpdateService

```python
class MetaUpdateService:
    def update_security_master(self, source: str = "baostock") -> pd.DataFrame:
        ...

    def update_industries(self, source: str = "akshare_eastmoney") -> pd.DataFrame:
        ...

    def update_concepts(self, source: str = "akshare_eastmoney") -> pd.DataFrame:
        ...
```

CLI：

```bash
conda run -n test python -m quantx.tools.update_meta \
  --source akshare \
  --security-master \
  --industries \
  --concepts
```

当前已实现的稳定离线导入命令：

```bash
conda run -n test python -m quantx.tools.update_meta \
  --meta-uri data/meta/quantx_meta.sqlite \
  --security-master-csv data/meta/snapshots/security_master.csv \
  --industry-csv data/meta/snapshots/industry_membership.csv \
  --json
```

CSV 字段要求：

```text
security_master.csv:
  symbol,name,exchange,board,list_date,delist_date

industry_membership.csv:
  symbol,industry_source,level,industry_code,industry_name

sector_membership.csv:
  symbol,sector_type,sector_code,sector_name,weight
```

输出：

```text
data/meta/quantx_meta.sqlite
data/meta/snapshots/security_master_YYYYMMDD.csv
data/meta/snapshots/industry_membership_YYYYMMDD.csv
data/meta/snapshots/sector_membership_YYYYMMDD.csv
```

## 6. 策略语言扩展

行业数据必须能像现在的 config 一样声明式使用。

新增顶层配置：

```yaml
meta:
  provider_uri: data/meta/quantx_meta.sqlite
  snapshot: latest
  fields:
    name:
      table: security_master
      column: name
    industry_l1:
      table: industry_membership
      column: industry_name
      source: eastmoney
      level: 1
    concept:
      table: sector_membership
      column: sector_name
      sector_type: concept
```

编译结果：

```text
MetaResolver 读取 symbols 对应的 meta。
MetaContext 挂到 BacktestContext。
Config explain 记录 meta snapshot、字段覆盖率、缺失 symbol。
```

## 7. 行业作为 selector 过滤条件

简单行业过滤：

```yaml
universe:
  base: all_a
  filters:
    - field: meta.industry_l1
      op: in
      value: ["电子", "计算机", "通信"]
```

编译为：

```text
load all_a symbols
join meta.industry_l1
filter symbols by industry list
pass filtered symbols to BacktestEngine
```

这是最简单的使用方式，但不能覆盖复杂行业因子策略。

## 8. 行业内排序

很多策略会先生成股票 alpha，再在行业内部排序。

Config：

```yaml
selector:
  where: buy_signal
  score: alpha_score
  group_by:
    field: meta.industry_l1
    missing: "其他"
  rank:
    method: topk_per_group
    topk: 2
  max_total: 20
  sort: score_desc
```

语义：

```text
1. 先按 buy_signal 得到候选股票。
2. 给候选股票 join industry_l1。
3. 每个行业内部按 alpha_score 降序。
4. 每个行业取前 2 只。
5. 全局最多取 20 只。
```

实现：

```python
selected = selected.join(meta["industry_l1"])
selected = (
    selected.sort_values("alpha_score", ascending=False)
    .groupby("industry_l1", group_keys=False)
    .head(topk)
)
selected = selected.head(max_total)
```

这一步不需要把行业塞进 FactorRuntime，属于 selector 的分组排序逻辑。

## 9. 行业因子和交叉选股

用户重点场景：

```text
先筛选股票
再根据行业计算全市场股票的行业因子
再把行业因子广播回股票
最后做交叉选股
```

这类策略很多，例如：

1. 股票自身出现买点。
2. 所属行业最近 20 日平均收益较强。
3. 所属行业内超过 60% 股票站上 MA20。
4. 只买强行业里的强股票。

为了适配这个逻辑，不能只支持 `industry in [...]`，必须支持 `group_factor`。

### 9.1 设计为 pipeline

新增 selector pipeline：

```yaml
selector:
  pipeline:
    - name: raw_candidates
      type: filter
      where: buy_signal

    - name: industry_strength
      type: group_factor
      group_by: meta.industry_l1
      scope: all_a
      factors:
        industry_ret20: GroupMean(ret20, group=industry_l1)
        industry_breadth20: GroupMean(close > ma20, group=industry_l1)
        industry_momentum_rank: GroupRank(industry_ret20)

    - name: industry_gate
      type: filter
      where: raw_candidates and (industry_breadth20 > 0.60) and (industry_momentum_rank <= 10)

    - name: final_rank
      type: rank
      score: alpha_score * 0.7 + industry_ret20 * 0.3
      group_by: meta.industry_l1
      method: topk_per_group
      topk: 2
      max_total: 20
```

### 9.2 scope 语义

`group_factor.scope` 很重要：

```yaml
scope: all_a
```

含义：行业因子用全市场股票计算，而不是只用候选股。

也可以支持：

```yaml
scope: raw_candidates
```

含义：行业因子只用第一步筛出来的候选股计算。

两者区别很大，必须在 config 中显式写。

推荐默认：

```text
scope: all_a
```

原因：行业强度通常应该基于全市场行业成分，而不是基于已筛选候选股，否则会引入二次筛选偏差。

### 9.3 GroupFactorRuntime

新增一层运行时：

```text
MarketPanel        -> 数值矩阵 [T, N]
MetaContext        -> 行业标签 [N]
GroupFactorRuntime -> 行业聚合矩阵 [T, N]
```

行业标签是最新快照，所以第一版形状是：

```text
industry_l1: [N]
```

数值因子仍是：

```text
ret20: [T, N]
ma20: [T, N]
```

GroupMean 计算：

```text
GroupMean(ret20, industry_l1)
  -> 先按 industry_l1 对每个日期横截面聚合
  -> 得到每个行业每天的均值 [T, G]
  -> 广播回每只股票 [T, N]
```

每只股票拿到的是“所属行业当天的行业因子值”。

### 9.4 group operator 列表

第一版推荐支持：

```text
GroupMean(x, group)
GroupMedian(x, group)
GroupSum(x, group)
GroupCount(mask, group)
GroupRatio(mask, group)       # 等价 GroupMean(mask)
GroupRank(x, ascending=False) # 行业之间排名
GroupZScore(x, group)         # 行业内标准化
GroupNeutralize(x, group)     # 行业中性化
```

最优先落地：

```text
GroupMean
GroupRatio
GroupRank
```

因为这三者已经能覆盖：

```text
行业平均收益
行业上涨家数比例
行业强度排名
```

### 9.5 Config 中 group 字段如何引用

不要在公式里写 `meta.industry_l1`，因为当前公式 AST 不支持点号属性。

编译时把 meta group 暴露为普通变量：

```yaml
groups:
  industry_l1:
    source: meta.industry_l1
    missing: "其他"
```

然后公式写：

```yaml
factors:
  ret20: close / Ref(close, 20) - 1
  ma20: Mean(close, 20)
  above_ma20: close > ma20

group_factors:
  industry_ret20: GroupMean(ret20, industry_l1)
  industry_breadth20: GroupRatio(above_ma20, industry_l1)
  industry_rank: GroupRank(industry_ret20)
```

`industry_l1` 编译后是一个 `[N]` 的整数 group code 向量，并携带 labels：

```python
industry_l1_codes: np.ndarray  # shape [N]
industry_l1_labels: dict[int, str]
```

这样公式语言仍然是安全 AST，不需要支持字符串矩阵。

## 10. 完整示例：强行业强个股

```yaml
name: strong_industry_strong_stock
version: 1

data:
  provider_uri: data/qlib_data_fixed
  universe: all_a
  start: 2021-01-04
  end: 2025-10-17
  look_back_days: 120

meta:
  provider_uri: data/meta/quantx_meta.sqlite
  snapshot: latest
  fields:
    name:
      table: security_master
      column: name
    industry_l1:
      table: industry_membership
      column: industry_name
      source: eastmoney
      level: 1

fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume
  change: $change

groups:
  industry_l1:
    source: meta.industry_l1
    missing: "其他"

factors:
  ret20: close / Ref(close, 20) - 1
  ma20: Mean(close, 20)
  above_ma20: close > ma20
  alpha_score: ret20 + CSRank(volume)

group_factors:
  industry_ret20: GroupMean(ret20, industry_l1)
  industry_breadth20: GroupRatio(above_ma20, industry_l1)
  industry_rank: GroupRank(industry_ret20)

signals:
  raw_buy: (ret20 > 0.05) and (close > ma20)
  strong_industry: (industry_breadth20 > 0.60) and (industry_rank <= 10)
  buy_signal: raw_buy and strong_industry

selector:
  where: buy_signal
  score: alpha_score * 0.7 + industry_ret20 * 0.3
  group_by:
    field: industry_l1
  rank:
    method: topk_per_group
    topk: 2
  max_total: 20
  sort: score_desc

rebalance:
  type: equal_weight
  max_positions: 20
  group_by:
    field: industry_l1
  group_weight:
    method: equal_group

execution:
  deal_price: close
  sell_rules:
    - name: stop_loss
      when: pnl_pct < -0.10
      action: sell_all
  buy:
    sizing: cash_equal
    skip_limit_up: true
```

这个例子表达了：

```text
股票自身强
行业整体强
行业内排名
行业间分散
```

并且不需要为这个策略写专门 Python 文件。

## 11. 编译流程

```text
1. 读取 YAML。
2. 加载 Qlib instruments。
3. 加载 MetaStore latest snapshot。
4. 对 symbols 做 meta join。
5. 编译 groups：
   industry_l1 -> codes [N], labels dict
6. 编译普通 factors：
   FactorRuntime.compute_formulas(factors)
7. 编译 group_factors：
   GroupFactorRuntime.compute(group_factors, groups)
8. 编译 signals：
   buy_signal 等可以引用普通 factors 和 group_factors
9. selector 执行：
   where -> score -> group_by/rank -> max_total
10. rebalance 执行：
   可选 group_weight 行业约束
11. explain 输出：
   meta snapshot、group coverage、industry labels、group_factors DAG
```

## 12. 和当前 FactorRuntime 的关系

当前 `FactorRuntime` 保持职责：

```text
数值矩阵公式 [T, N]
```

新增 `GroupFactorRuntime`：

```text
分类分组聚合 [T, N] + [N] group codes
```

不要在第一版把所有 meta 功能硬塞进 `FactorRuntime`。

推荐结构：

```text
FactorRuntime
  - Ref / Mean / Max / CSRank / Cross

GroupFactorRuntime
  - GroupMean / GroupRatio / GroupRank

ConfigSelector
  - where / score / group_by / topk_per_group

ConfigRebalance
  - equal_weight / equal_group / fixed_group_weight
```

## 13. 性能设计

行业数通常几十个，股票数几千只，日期几千天。

`GroupMean(x, industry)` 可以用矩阵化实现：

```python
for each date:
    sums = np.bincount(group_codes, weights=x[date], minlength=G)
    counts = np.bincount(group_codes, weights=valid_mask[date], minlength=G)
    group_mean = sums / counts
    out[date] = group_mean[group_codes]
```

复杂度：

```text
O(T * N)
```

和普通全市场因子同级别，不需要逐股票 pandas groupby。

后续可以优化：

```text
numba
bottleneck
precomputed group index
chunked matrix
```

## 14. 可视化接入

可视化只读 MetaStore，不参与策略计算。

新增 API：

```text
GET /api/meta/symbols/{symbol}
GET /api/meta/symbols?query=浦发
GET /api/meta/industries
GET /api/meta/sectors
```

`GET /api/reports/{run_id}/symbols/{symbol}` 返回增加：

```json
{
  "symbol": "SH600000",
  "name": "浦发银行",
  "industry": {
    "source": "eastmoney",
    "level": 1,
    "name": "银行"
  },
  "sectors": [
    {"type": "index", "name": "沪深300"},
    {"type": "concept", "name": "互联金融"}
  ],
  "bars": [],
  "trades": []
}
```

交易表支持：

```text
按 symbol 搜索
按 name 搜索
按 industry 搜索
按 concept 搜索
```

报告增加：

```text
行业持仓分布
行业交易次数
行业收益贡献
行业胜率
```

## 15. 测试计划

### 15.1 MetaStore

```text
写入 security_master 后能按 symbol 查 name。
写入 industry_membership 后能按 symbol 查 industry。
manual override 优先级高于 akshare。
缺失行业填充为 "其他"。
```

### 15.2 GroupFactorRuntime

```text
GroupMean(ret20, industry_l1) 结果正确广播回股票。
GroupRatio(close > ma20, industry_l1) 结果正确。
GroupRank(industry_ret20) 排名方向正确。
缺失行业不会导致崩溃。
```

### 15.3 Config 编译

```text
groups.industry_l1 引用未知 meta 字段时报错。
group_factors 引用未知 group 时报错。
selector.group_by 引用未知 group 时报错。
scope 只能是 all_a 或已有 pipeline stage。
```

### 15.4 回测语义

```text
不配置 meta 时，现有 shuijiao_legacy.yaml 收益不变。
配置 meta 但不引用行业时，收益不变。
行业内 topk_per_group 结果符合预期。
equal_group 分权结果符合预期。
```

## 16. 分阶段实施

### Phase 1: MetaStore 和更新工具

交付：

```text
quantx/core/data/meta/store.py
quantx/core/data/meta/sources/akshare_meta.py
quantx/core/data/meta/sources/baostock_meta.py
quantx/tools/update_meta.py
```

验收：

```bash
conda run -n test python -m quantx.tools.update_meta --security-master --industries
```

生成：

```text
data/meta/quantx_meta.sqlite
data/meta/snapshots/*.csv
```

### Phase 2: 可视化名称和行业

交付：

```text
GET /api/meta/symbols/{symbol}
GET /api/meta/industries
报告和个股页面展示 name/industry
交易表支持 name/industry 搜索
```

### Phase 3: Config 行业过滤和行业内排序

交付：

```text
meta:
groups:
selector.group_by
selector.rank.topk_per_group
rebalance.group_by
```

### Phase 4: GroupFactorRuntime

交付：

```text
group_factors:
GroupMean
GroupRatio
GroupRank
```

验收：

```text
强行业强个股 YAML 能跑通。
不写 Python 策略代码。
explain.json 能看到 meta snapshot 和 group factor DAG。
```

### Phase 5: 行业分析报告

交付：

```text
行业收益贡献
行业持仓分布
行业交易胜率
行业 B/S 分布
```

## 17. 结论

第一版不要做复杂历史行业 as-of，而是：

```text
使用最新行业快照。
把行业作为 config 可引用的 meta group。
用 GroupFactorRuntime 计算行业因子。
用 selector pipeline 表达先筛选、再行业因子、再交叉选股。
```

这条路径能满足：

1. 可视化展示股票名称和行业。
2. 策略按行业过滤。
3. 策略按行业内排序。
4. 策略计算行业强度、行业宽度、行业排名。
5. 策略做强行业强个股的交叉选股。

同时保持：

```text
Qlib 数值数据层不被污染。
MetaStore 可独立更新。
Config 仍然完整描述策略逻辑。
Python 只提供通用 operator 和 runtime。
```
