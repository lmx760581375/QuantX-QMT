# 回测引擎层开发计划

> **版本**: v0.1.0
> **日期**: 2026-06-25
> **预计工期**: 7-10 天
> **依赖**: 数据层 (01_DATA_LAYER.md)、因子层 (02_FACTOR_LAYER.md)
> **重要性**: 这是整个系统最核心、最复杂的模块，回测结果的真实性和准确性取决于此

---

## 1. 目标

回测引擎层负责模拟真实的 A 股交易环境，包括：
1. 完整的 A 股交易规则（T+1、板别涨跌停、停牌、一字板）
2. 精确的交易成本模型（佣金、印花税卖向、过户费沪市、滑点）
3. 两阶段回测流程（信号预计算 + 交易顺序执行）
4. 与 Qlib 数据层无缝对接

---

## 2. 文件清单与功能

### 2.1 文件列表

```
quantx/core/engine/
├── __init__.py              # 模块入口
├── types.py                 # 数据类型定义（Order, Trade, Signal, Position, DailySnapshot）
├── config.py                # 回测配置数据类（BacktestConfig）
├── cost.py                  # 交易成本模型
├── board.py                 # 板别管理（涨跌停阈值、板别判断）
├── exchange.py              # A 股行情数据提供者（从 Qlib 加载数据，提供查询接口）
├── account.py               # 账户管理
├── executor.py              # 订单执行器（校验 + 执行）
├── context.py               # 回测上下文（策略数据视图 = 行情 + 因子 + 信号）
├── engine.py                # 两阶段回测引擎（主循环）
└── signals.py               # 信号矩阵（稀疏存储）
```

**职责边界**：

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `exchange.py` | 行情数据查询：成交价、停牌、涨跌停、一字板 | 订单校验、账户操作 |
| `executor.py` | 订单校验 + 订单执行 | 行情数据查询（调用 exchange） |
| `context.py` | 策略数据视图：行情 + 因子 + 信号矩阵 | 数据加载（委托 exchange + factor_engine） |
| `account.py` | 现金、持仓、交易记录、每日净值 | 订单校验（委托 executor） |

### 2.2 各文件详细说明

---

#### 2.2.1 `types.py` — 数据类型定义

**功能**：定义回测引擎中使用的所有数据类型。

**数据类**：
```python
@dataclass
class Order:
    symbol: str           # 股票代码
    action: OrderAction   # BUY / SELL
    price: float          # 委托价格
    quantity: int         # 委托数量（股）
    date: str             # 委托日期
    reason: str = ""      # 下单原因

@dataclass
class Trade:
    symbol: str
    action: OrderAction
    price: float          # 成交价格
    quantity: int         # 成交数量
    date: str
    commission: float     # 佣金
    stamp_tax: float      # 印花税
    transfer_fee: float   # 过户费
    slippage: float       # 滑点

@dataclass
class Position:
    symbol: str
    quantity: int         # 持仓数量
    avg_cost: float       # 持仓均价（摊薄后）
    buy_date: str         # 最近买入日期（用于 T+1 判断）
    market_value: float   # 市值
    holding_days: int = 0 # 已持有天数（用于策略判断）

@dataclass
class Signal:
    symbol: str
    date: str
    score: float          # 信号分数，越高越优先
    selected: bool        # 是否选中
    reason: str = ""      # 选股原因（调试用）

@dataclass
class DailySnapshot:
    date: str
    cash: float
    total_value: float
    positions: Dict[str, Position]
    daily_return: float          # 当日收益率
    cumulative_return: float     # 累计收益率
    benchmark_return: float = 0.0
```

**开发要点**：
- 使用 `@dataclass` 确保类型安全和可序列化
- `OrderAction` 用 `Enum`：`BUY = 1`, `SELL = 2`
- 注意 `quantity` 必须是整数（A 股按股交易，但实际按手取整）
- 直接复用 `new-quant/core/engine/executor.py` 中的 Order 定义

---

#### 2.2.2 `cost.py` — 交易成本模型

**功能**：精确计算 A 股交易的各项成本。

**核心类**：`TransactionCost`

**参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `commission_rate` | 0.0003 (万三) | 佣金费率 |
| `min_commission` | 5.0 | 最低佣金（元） |
| `stamp_tax_rate` | 0.0005 (万五) | 印花税（仅卖出） |
| `transfer_fee_rate` | 0.00002 (十万分之二) | 过户费（仅沪市） |
| `slippage` | 0.001 (0.1%) | 滑点比例 |

**主要方法**：
| 方法 | 功能 |
|------|------|
| `calculate_buy_cost(symbol, price, quantity)` | 计算买入成本（佣金+过户费+滑点） |
| `calculate_sell_cost(symbol, price, quantity)` | 计算卖出成本（佣金+印花税+过户费+滑点） |
| `calculate_total_cost(order)` | 计算总成本 |
| `apply_slippage(price, direction)` | 应用滑点：买入加价，卖出降价 |

**成本计算逻辑**：
```python
def calculate_buy_cost(self, symbol, price, quantity):
    trade_amount = price * quantity
    # 佣金：max(万三, 5元)
    commission = max(trade_amount * 0.0003, 5.0)
    # 过户费：仅沪市，十万分之二
    transfer_fee = trade_amount * 0.00002 if symbol.startswith('SH') else 0
    # 滑点：买入价格上浮
    slippage = trade_amount * 0.001
    return commission + transfer_fee + slippage

def calculate_sell_cost(self, symbol, price, quantity):
    trade_amount = price * quantity
    commission = max(trade_amount * 0.0003, 5.0)
    # 印花税：仅卖出，千一（实际 2024 年降为万五）
    stamp_tax = trade_amount * 0.0005
    transfer_fee = trade_amount * 0.00002 if symbol.startswith('SH') else 0
    slippage = trade_amount * 0.001  # 卖出价格下浮
    return commission + stamp_tax + transfer_fee + slippage
```

**开发要点**：
- 直接复用 `new-quant/core/engine/cost.py` 的 TransactionCost dataclass（约 40 行）
- 增强：添加印花税（卖向）和过户费（沪市）
- 所有成本参数通过 YAML 配置注入，不硬编码
- 注意：2024 年 A 股印花税从 0.1% 降为 0.05%，使用最新税率

**参考代码**：`new-quant/core/engine/cost.py`

---

#### 2.2.3 `board.py` — 板别管理

**功能**：判断股票所属交易板别，返回对应的涨跌停阈值。

**核心类**：`BoardManager`

**板别规则**：
| 板别 | 代码前缀 | 涨跌停幅度 | 说明 |
|------|----------|------------|------|
| 主板 | SH60/SZ00 | ±10% | 沪深主板 |
| 创业板 | SZ30 | ±20% | ChiNext |
| 科创板 | SH68 | ±20% | STAR Market |
| 北交所 | BJ | ±30% | 北京证券交易所 |
| ST 股票 | 含 ST | ±5% | 风险警示板 |

**主要方法**：
| 方法 | 功能 |
|------|------|
| `get_board(symbol)` | 返回股票板别：mainboard / chi_next / star / beijing / st |
| `get_limit_up_rate(symbol)` | 返回涨停幅度 |
| `get_limit_down_rate(symbol)` | 返回跌停幅度 |
| `is_st_stock(symbol)` | 判断是否为 ST 股票 |
| `is_chi_next(symbol)` | 判断是否为创业板 |
| `is_star_market(symbol)` | 判断是否为科创板 |

**开发要点**：
- 这是 Qlib 缺失的关键功能，需要全新实现（约 80 行）
- 板别判断基于股票代码前缀，简单可靠
- ST 股票需要从数据中获取 `is_st` 字段（BaoStock 有提供）
- 参考 `new-quant/core/engine/context.py` 的 `_build_limit_maps` 方法
---

#### 2.2.4 `exchange.py` — A 股行情数据提供者

**功能**：从 Qlib 加载行情数据，提供查询接口。**不负责订单校验**（校验在 executor.py 中）。

**核心类**：`AStockExchange`

**主要方法**：
| 方法 | 功能 | 复杂度 |
|------|------|--------|
| `get_deal_price(symbol, date, direction)` | 获取成交价格（T 日开盘价） | 低 |
| `is_stock_suspended(symbol, date)` | 判断是否停牌（$close 为 NaN） | 低 |
| `is_limit_up(symbol, date)` | 判断是否涨停（$change >= limit_up_rate） | 中 |
| `is_limit_down(symbol, date)` | 判断是否跌停（$change <= -limit_down_rate） | 中 |
| `is_one_side_limit_up(symbol, date)` | 判断是否一字涨停（开盘即涨停，全日无成交） | 中 |
| `is_one_side_limit_down(symbol, date)` | 判断是否一字跌停（开盘即跌停，全日无成交） | 中 |
| `is_stock_tradable(symbol, date, direction)` | 综合判断是否可交易 | 高 |
| `get_preclose(symbol, date)` | 获取前收盘价（用于价格跳变保护） | 低 |
| `load_quote_data(symbols, start, end)` | 从 Qlib 加载行情数据 | 低 |

**一字板判断逻辑（修正版）**：
```python
def is_one_side_limit_up(self, symbol, date):
    \"\"\"判断是否一字涨停：开盘即涨停，全日无成交机会\"\"\"
    open_p = self.get_deal_price(symbol, date, OrderDir.BUY)
    high = self.quote.loc[(date, symbol), '$high']
    low = self.quote.loc[(date, symbol), '$low']
    close = self.quote.loc[(date, symbol), '$close']
    volume = self.quote.loc[(date, symbol), '$volume']
    limit_up_rate = self.board.get_limit_up_rate(symbol)
    limit_up_price = preclose * (1 + limit_up_rate)

    # 1) 收盘价等于涨停价
    if abs(close - limit_up_price) > 1e-6:
        return False
    # 2) 开盘价等于涨停价（一字板的关键特征）
    if abs(open_p - limit_up_price) > 1e-6:
        return False
    # 3) 全日无波动（high == low == open）
    if abs(high - low) > 1e-6:
        return False
    # 4) 成交量极小（一字板几乎没有成交）
    if volume > 0:
        return False  # 有成交量说明不是一字板，是盘中涨停
    return True

def is_limit_up(self, symbol, date):
    \"\"\"判断是否涨停（包括一字板和盘中涨停）\"\"\"
    change = self.quote.loc[(date, symbol), '$change']
    limit_up_rate = self.board.get_limit_up_rate(symbol)
    return change >= limit_up_rate - 1e-6  # 浮点数容差
```

**成交价格规则**：
- 使用 T 日开盘价（`$open`）作为成交价格（T-1 日收盘信号 + T 日开盘执行）
- 应用滑点：买入价 = open * (1 + slippage)，卖出价 = open * (1 - slippage)
- 按 100 股（1 手）取整

**从 Qlib 加载数据的方式**：
```python
import qlib
from qlib.data import D

class AStockExchange:
    def __init__(self, provider_uri):
        qlib.init(provider_uri=provider_uri, region='cn')

    def load_quote_data(self, symbols, start, end):
        # 从 Qlib 加载行情数据
        self.quote = D.features(
            symbols,
            ['$open', '$high', '$low', '$close', '$volume', '$change', '$factor'],
            start, end,
            freq='day'
        )
```

**开发要点**：
- 这是最核心的新代码，需要仔细实现（约 200 行）
- 撮合规则必须按优先级顺序检查，不可跳过
- 一字板判断逻辑：`open == high == low == close` 且 `change` 接近涨跌停阈值
- 需要从 Qlib 读取 `$change` 字段来判断涨跌停
- 需要读取 `$factor` 字段来进行交易单位取整
- 注意：Qlib 的 `$change` 字段是 `(close - preclose) / preclose`
- 涨跌停判断：`$change >= limit_up_rate` 为涨停，`$change <= -limit_down_rate` 为跌停

**参考代码**：Qlib 的 `qlib/backtest/exchange.py`（参考撮合逻辑，但需增强 A 股规则）

---

#### 2.2.5 `account.py` — 账户管理

**功能**：管理回测账户的现金、持仓、交易记录和每日净值。

**核心类**：`Account`

**主要属性**：
| 属性 | 类型 | 说明 |
|------|------|------|
| `cash` | float | 可用现金 |
| `total_value` | float | 总资产（现金 + 持仓市值） |
| `positions` | Dict[str, Position] | 当前持仓 |
| `trades` | List[Trade] | 成交记录 |
| `daily_snapshots` | List[DailySnapshot] | 每日账户快照 |
| `latest_daily_return` | float | 最近一个交易日的收益率（供 PolicyState 使用） |
| `cumulative_return` | float | 累计收益率（供 PolicyState 使用） |
| `init_cash` | float | 初始资金 |

**主要方法**：
| 方法 | 功能 |
|------|------|
| `buy(symbol, price, quantity, date)` | 买入股票，更新现金和持仓 |
| `sell(symbol, price, quantity, date)` | 卖出股票，更新现金和持仓 |
| `update_daily_balance(date, exchange)` | 每日收盘更新持仓市值 |
| `get_position(symbol)` | 获取持仓 |
| `can_sell(symbol, date)` | 检查是否可卖出（T+1 检查） |
| `get_available_cash()` | 获取可用现金 |
| `get_total_value()` | 获取总资产 |

**T+1 检查逻辑**：
```python
def can_sell(self, symbol, date):
    position = self.positions.get(symbol)
    if position is None or position.quantity == 0:
        return False
    # T+1: 当日买入的股票不可卖出
    if position.buy_date == date:
        return False
    return True
```

**持仓成本摊薄逻辑**：
```python
def buy(self, symbol, price, quantity, date):
    cost = price * quantity + self.cost.calculate_buy_cost(symbol, price, quantity)
    if cost > self.cash:
        # 自动调整买入数量
        max_quantity = int(self.cash / (price * (1 + self.cost.get_buy_cost_ratio())))
        quantity = (max_quantity // 100) * 100  # 按手取整
        if quantity == 0:
            return None
        cost = price * quantity + self.cost.calculate_buy_cost(symbol, price, quantity)

    self.cash -= cost
    if symbol in self.positions:
        # 加仓：摊薄成本
        old = self.positions[symbol]
        total_quantity = old.quantity + quantity
        old.avg_cost = (old.avg_cost * old.quantity + price * quantity) / total_quantity
        old.quantity = total_quantity
        old.buy_date = date
    else:
        self.positions[symbol] = Position(
            symbol=symbol, quantity=quantity, avg_cost=price, buy_date=date
        )
    return Trade(symbol=symbol, action=OrderAction.BUY, ...)
```

**开发要点**：
- 直接复用 `new-quant/core/engine/account.py`（约 200 行），增强 T+1 检查和成本模型
- 注意：加仓时需要摊薄持仓成本（加权平均）
- 注意：卖出时要区分减仓和清仓
- 注意：`buy_date` 字段是 T+1 检查的关键，每次买入都要更新
- 注意：`holding_days` 在 `update_daily_balance` 中每日 +1，供 PolicyState 使用
- 停牌股票按最后成交价估值（通过 `update_daily_balance` 中的 `exchange.is_stock_suspended` 判断）
- `latest_daily_return` 和 `cumulative_return` 在 `update_daily_balance` 中更新：
  ```python
  def update_daily_balance(self, date, exchange):
      prev_value = self.total_value
      for sym, pos in self.positions.items():
          if not exchange.is_stock_suspended(sym, date):
              close = exchange.get_close(sym, date)
              pos.market_value = close * pos.quantity
          pos.holding_days += 1  # 每日递增
      self.total_value = self.cash + sum(p.market_value for p in self.positions.values())
      self.latest_daily_return = (self.total_value - prev_value) / prev_value if prev_value > 0 else 0.0
      self.cumulative_return = self.total_value / self.init_cash - 1
  ```

**参考代码**：`new-quant/core/engine/account.py`
---

#### 2.2.6 `executor.py` — 订单执行器

**功能**：接收策略生成的订单，进行 A 股规则校验后执行，返回成交结果。

**核心类**：`Executor`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `execute(order, account, exchange)` | 执行订单，返回 Trade 或 None（被拒绝） |
| `validate(order, account, exchange)` | 校验订单是否可执行（调用 exchange 查询行情状态） |
| `_execute_buy(order, account, exchange)` | 执行买入 |
| `_execute_sell(order, account, exchange)` | 执行卖出 |

**校验规则（按优先级，在 executor 中执行）**：
```
1. 停牌检查（调用 exchange.is_stock_suspended）→ 不可交易
2. 涨跌停检查（调用 exchange.is_limit_up / is_limit_down）：
   - 涨停不买入，跌停不卖出
3. 一字板检查（调用 exchange.is_one_side_limit_up / down）→ 买卖都不可
4. T+1 检查（调用 account.can_sell）→ 当日买入不可卖出
5. 价格跳变保护（调用 exchange.get_preclose）→ 偏差 > 9.5% 跳过买入
6. 现金检查（调用 account.get_available_cash）→ 自动调整买入数量
7. 持仓检查（调用 account.get_position）→ 卖出数量不超过持仓
```

**执行流程**：
```
Order → validate() → 检查通过?
                       ├── No → 返回 None + 拒绝原因
                       └── Yes → _execute_buy() / _execute_sell()
                                    │
                                    ├── 获取成交价格 (exchange.get_deal_price)
                                    ├── 应用滑点 (cost.apply_slippage)
                                    ├── 按手取整 (// 100 * 100)
                                    ├── 调用 account.buy() / account.sell()
                                    └── 返回 Trade 对象
```

**开发要点**：
- 直接复用 `new-quant/core/engine/executor.py`（约 80 行），增强校验规则
- 先卖后买：先执行所有卖出订单，释放资金，再执行买入订单
- 买入订单按资金分配优先级排序
- 每笔成交记录到 account.trades 列表

**参考代码**：`new-quant/core/engine/executor.py`

---

#### 2.2.7 `context.py` — 回测上下文

**功能**：提供策略在回测中访问数据的统一接口。

**核心类**：`BacktestContext`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `load_data(symbols, start, end)` | 从 Qlib 加载行情数据 + 因子数据 |
| `get_current_data(date)` | 获取当日横截面数据 |
| `get_history(symbol, field, start, end)` | 获取单股票历史数据 |
| `get_history_by_range(symbols, field, start, end)` | 批量获取历史数据 |
| `get_factor_value(factor_name, date)` | 获取因子值 |
| `get_stock_list(date)` | 获取当日可交易股票列表 |
| `set_stock_signals(signals)` | 存储预计算信号 |
| `get_stock_signals(date)` | 获取当日预计算信号 |
| `next()` | 推进到下一个交易日 |
| `is_finished()` | 回测是否结束 |

**数据加载流程**：
```python
def load_data(self, symbols, start, end, factor_names):
    # 1. 从 Qlib 加载行情数据
    self.market_data = D.features(
        symbols,
        ['$open', '$high', '$low', '$close', '$volume', '$change'],
        start, end, freq='day'
    )
    # 2. 从 Qlib 表达式引擎加载因子
    if factor_names:
        self.factor_data = D.features(
            symbols, factor_names, start, end, freq='day'
        )
    # 3. 构建交易日历
    self.trade_dates = calendar.get_trading_days(start, end)
    # 4. 预计算涨跌停/停牌 map
    self._build_limit_maps()
```

**开发要点**：
- 直接复用 `new-quant/core/engine/context.py`（约 150 行），适配 Qlib 数据接口
- 使用 Qlib 的 `D.features()` 而不是手动加载 CSV
- 因子加载也通过 Qlib 的表达式引擎，统一接口
- 预计算涨跌停/停牌 Map，避免在回测循环中重复判断

**参考代码**：`new-quant/core/engine/context.py`

---

#### 2.2.8 `engine.py` — 两阶段回测引擎（主循环）

**功能**：编排整个回测流程，Phase 1 并行计算信号，Phase 2 顺序执行交易。

**核心类**：`BacktestEngine`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `run(strategy: CompositeStrategy, start, end, symbols, config)` | 运行完整回测 |
| `_phase1_precompute_signals(strategy, context, account)` | Phase 1: 并行计算所有日期的选股信号 |
| `_phase2_execute_trades(strategy, context, account, executor, exchange)` | Phase 2: 顺序执行交易 |
| `_build_state(date, context, account) -> PolicyState` | 构建 PolicyState，供 Policy.act() 使用 |
| `_collect_results(account)` | 收集回测结果 |

**回测主循环（Policy 架构版）**：
```python
def run(self, strategy: CompositeStrategy, start, end, symbols, config):
    # 初始化 Qlib
    qlib.init(provider_uri=config.provider_uri, region='cn')

    # 初始化各组件
    cost_model = config.cost
    board = BoardManager()
    exchange = AStockExchange(board, provider_uri=config.provider_uri)
    exchange.load_quote_data(symbols, start, end)

    context = BacktestContext(exchange)
    context.load_data(symbols, start, end, config.factor_names)

    account = Account(init_cash=config.init_cash, cost=cost_model)
    executor = Executor()

    strategy.on_init(context)

    # Phase 1: 并行预计算选股信号
    #   每个日期调用 strategy.selector.act(state) → StockSelection
    self._phase1_precompute_signals(strategy, context, account)

    # Phase 2: 按交易日顺序执行
    #   调仓 + 生成订单在 strategy 内部完成:
    #     strategy.rebalance.act(state, selection) → WeightAllocation
    #     strategy.execution.act(state, allocation) → OrderList
    while not context.is_finished():
        date = context.next()
        state = self._build_state(date, context, account)
        selection = context.get_stock_signals(date)
        order_list = strategy.get_trade_signal(state, selection)

        for order in order_list.orders:
            trade = executor.execute(order, account, exchange)
            if trade:
                account.trades.append(trade)

        account.update_daily_balance(date, exchange)

    strategy.on_finish(context)
    return self._collect_results(account)

def _build_state(self, date, context, account) -> PolicyState:
    """构建 PolicyState，供 Policy.act() 使用"""
    from quantx.core.strategy.base import PolicyState, AccountSnapshot, PositionSnapshot
    return PolicyState(
        date=date,
        market_data=context.get_current_data(date),
        factor_data=context.get_factor_data(date),
        context=context,  # 规则策略可通过 context 访问历史数据
        account=AccountSnapshot(
            cash=account.cash, total_value=account.get_total_value(),
            daily_return=account.latest_daily_return,
            cumulative_return=account.cumulative_return,
        ),
        positions={
            sym: PositionSnapshot(
                symbol=sym, quantity=pos.quantity,
                avg_cost=pos.avg_cost, market_value=pos.market_value,
                weight=pos.market_value / account.get_total_value(),
                holding_days=pos.holding_days,
                unrealized_pnl=pos.market_value - pos.avg_cost * pos.quantity,
            )
            for sym, pos in account.positions.items()
        },
    )
```
            trade = executor.execute(order, account, exchange)
            if trade:
                account.trades.append(trade)

        # 5. 每日收盘更新持仓市值
        account.update_daily_balance(date, exchange)

    # 收集结果
    strategy.on_finish(context)
    return self._collect_results(account)

def _phase1_precompute_signals(self, strategy, context, account):
    # 使用 ThreadPoolExecutor 并行计算所有日期的选股信号
    # 每个日期构建 PolicyState，调用 strategy.selector.act(state)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for date in context.trade_dates:
            state = self._build_state(date, context, account)
            futures[pool.submit(strategy.get_stock_signal, state)] = date
        for future in as_completed(futures):
            date = futures[future]
            context.set_stock_signals(date, future.result())
```

**开发要点**：
- 直接复用 `new-quant/core/engine/engine.py` 的两阶段设计（约 100 行）+ `myquant-clean/backtrade_v2/fast_engine.py` 的信号预计算
- Phase 1 并行化：每日期独立，天然并行，使用 ThreadPoolExecutor
- Phase 2 必须顺序：每日期依赖前一日持仓状态
- 引擎需要记录各阶段耗时，便于性能分析
- 策略函数签名：`get_stock_signal(context, date)` -> List[Signal]
- 策略函数签名：`get_trade_signal(context, account)` -> List[Order]

**参考代码**：`new-quant/core/engine/engine.py`、`myquant-clean/backtrade_v2/fast_engine.py`
---

#### 2.2.9 `signals.py` — 信号矩阵

**功能**：预计算信号的稀疏存储和管理。

**核心类**：`SignalMatrix`

**主要方法**：
| 方法 | 功能 |
|------|------|
| `set(date, signals)` | 存储某日信号 |
| `get(date)` | 获取某日信号 |
| `to_sparse()` | 转换为稀疏矩阵（只存 selected=True 的行） |
| `from_sparse()` | 从稀疏矩阵恢复 |

**稀疏存储格式**：
```python
# 全矩阵：900 天 × 50 股 = 45,000 条记录
# 稀疏矩阵：900 天 × 50 股 × 5% 选中率 = 2,250 条记录
# 内存节省：95%

signal_matrix = pd.DataFrame({
    'date': ['2020-01-02', '2020-01-02', ...],
    'symbol': ['SH600519', 'SZ000001', ...],
    'score': [0.85, 0.72, ...],
    'selected': [True, True, ...]
})
```

**开发要点**：
- 直接复用 `myquant-clean/backtrade_v2/precomputed_selector.py` 的稀疏存储模式（约 60 行）
- 只存 `selected=True` 的行，大幅降低内存
- 信号矩阵在 Phase 1 计算完成后不再修改

**参考代码**：`myquant-clean/backtrade_v2/precomputed_selector.py`

---

## 3. 开发顺序

```
Day 1-2: 基础数据类型
  ├── types.py (Order, Trade, Position, Signal, DailySnapshot)
  ├── config.py (BacktestConfig)
  ├── cost.py (TransactionCost)
  └── board.py (BoardManager)

Day 3-5: 撮合引擎（核心）
  ├── exchange.py (AStockExchange) ← 最关键，需要仔细实现和测试
  └── 编写 exchange 的单元测试

Day 6-7: 账户与执行器
  ├── account.py (Account)
  ├── executor.py (Executor)
  └── 编写 account + executor 的单元测试

Day 8-9: 回测引擎
  ├── context.py (BacktestContext)
  ├── signals.py (SignalMatrix)
  ├── engine.py (BacktestEngine)
  └── 使用合成数据编写端到端集成测试

Day 10: 集成测试与验证
  ├── 合成数据：手动构造已知结果的行情，验证回测输出
  ├── 真实数据：小范围真实数据回测，与 myquant-strategy 对比
  └── 修复问题
```

## 4. 关键注意事项

### 4.1 撮合引擎的正确性验证

撮合引擎是回测准确性的根基，需要重点验证：
1. 构造涨停日数据，验证买入订单被拒绝
2. 构造跌停日数据，验证卖出订单被拒绝
3. 构造一字板数据，验证买卖订单都被拒绝
4. 构造 T+1 场景，验证当日买入不可卖出
5. 构造停牌日数据，验证不可交易
6. 构造现金不足场景，验证自动调整买入数量

### 4.2 与 Qlib 数据层的对接

- 确保 `qlib.init(provider_uri=...)` 在回测前被调用
- 使用 `D.features()` 统一加载行情和因子数据
- 注意 Qlib 的 MultiIndex 格式：`(datetime, instrument)`
- `$change` 字段用于涨跌停判断，必须确保数据正确

### 4.3 性能考虑

- Phase 1 信号预计算使用 ThreadPoolExecutor，最大线程数 = min(8, cpu_count)
- Phase 2 交易执行是顺序的，不需要并行
- 涨跌停/停牌 Map 在 Context 初始化时预计算，避免循环中重复查询
- 信号矩阵使用稀疏存储，降低内存占用

### 4.4 成交价选择

- 使用 T 日开盘价（`$open`）作为成交价，符合 T-1 信号 + T 日开盘执行的逻辑
- 如果策略需要 T 日收盘价成交，可作为配置选项
- 注意：Qlib 的 `deal_price` 默认是 `$close`，我们需要改为 `$open`

## 5. 测试计划

| 测试文件 | 测试内容 | 测试方法 |
|----------|----------|----------|
| `tests/engine/test_cost.py` | 成本计算 | 手工计算对比 |
| `tests/engine/test_board.py` | 板别判断 | 不同代码前缀验证 |
| `tests/engine/test_exchange.py` | 撮合规则 | 合成数据模拟各种场景 |
| `tests/engine/test_account.py` | 账户管理 | 买卖操作后验证现金/持仓 |
| `tests/engine/test_executor.py` | 订单执行 | 订单校验 + 执行流程 |
| `tests/engine/test_engine.py` | 回测引擎 | 合成数据端到端回测 |
| `tests/engine/test_t1.py` | T+1 规则 | 同一天买卖应被拒绝 |
| `tests/engine/test_limit.py` | 涨跌停规则 | 涨停买/跌停卖应被拒绝 |

## 6. 验收标准

- [ ] 涨停股票买入被拒绝，跌停股票卖出被拒绝
- [ ] 一字板股票买卖都被拒绝
- [ ] 当日买入股票不可当日卖出（T+1）
- [ ] 停牌股票不可交易
- [ ] 交易成本计算与手工计算一致（误差 < 0.01 元）
- [ ] 板别涨跌停阈值正确（主板 10%、创业板 20%、科创板 20%、北交所 30%、ST 5%）
- [ ] 合成数据回测结果与预期一致
- [ ] 与 myquant-strategy 相同策略回测偏差 < 1%
