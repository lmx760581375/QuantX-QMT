"""两阶段回测引擎

Phase 1: 并行预计算所有日期的选股信号
Phase 2: 按交易日顺序执行交易（必须顺序，依赖前日状态）
"""

import logging
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .account import Account
from .board import BoardManager
from .context import BacktestContext
from .cost import TransactionCost
from .exchange import AStockExchange
from .executor import Executor
from .types import DailySnapshot, Trade

logger = logging.getLogger(__name__)


def _position_manager_decisions(strategy) -> List[Dict[str, Any]]:
    manager = getattr(strategy, "position_manager", None)
    getter = getattr(manager, "get_decisions", None)
    if not callable(getter):
        return []
    rows = getter()
    if not isinstance(rows, list):
        raise TypeError("PositionManager.get_decisions() must return a list")
    return [dict(row) for row in rows]


@dataclass
class BacktestConfig:
    """回测配置"""
    init_cash: float = 1_000_000
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "SH000300"
    provider_uri: str = "data/qlib_data"
    historical_st_path: Optional[str] = None
    cost: Optional[TransactionCost] = None
    factor_names: List[str] = field(default_factory=list)
    max_workers: int = 8
    deal_price: str = "open"
    buy_deal_price: Optional[str] = None
    sell_deal_price: Optional[str] = None
    look_back_days: int = 0
    validate_trading_rules: bool = True
    price_jump_limit: float | str = 0.095
    legacy_cost_price: bool = False
    auto_adjust_buy_quantity: bool = True
    error_policy: str = "fail_fast"
    market_field_aliases: Dict[str, str] = field(default_factory=dict)
    precompute_signals: Optional[bool] = None


@dataclass
class BacktestResult:
    """回测结果"""
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    selection_candidates: List[Dict[str, Any]] = field(default_factory=list)
    daily_selection_candidates: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[BacktestConfig] = None
    time_stats: Dict[str, float] = field(default_factory=dict)
    signal_errors: List[Dict[str, str]] = field(default_factory=list)
    position_manager_decisions: List[Dict[str, Any]] = field(default_factory=list)


class BacktestSession:
    """Stateful facade over QuantX's existing daily execution path.

    It intentionally owns no alternative accounting or fill logic. PPO may
    provide a PositionPlan one date at a time while final backtests continue
    to use BacktestEngine.run unchanged.
    """

    def __init__(
        self,
        engine: "BacktestEngine",
        strategy,
        symbols: List[str],
        config: Optional[BacktestConfig] = None,
        shared_context: Optional[BacktestContext] = None,
    ):
        self.engine = engine
        self.strategy = strategy
        self.symbols = list(symbols)
        self.config = config or engine.config
        self.cost_model = self.config.cost or TransactionCost()
        self.board = BoardManager()
        if shared_context is not None:
            self.context = shared_context
            self.exchange = shared_context.exchange
        else:
            self.exchange = AStockExchange(self.board, provider_uri=self.config.provider_uri)
            self._configure_historical_st(self.exchange, self.config.historical_st_path)
            self.exchange.deal_price = self.config.deal_price
            setter = getattr(self.exchange, "set_execution_prices", None)
            if callable(setter):
                setter(self.config.buy_deal_price, self.config.sell_deal_price)
            self.context = BacktestContext(self.exchange)
            self.context.load_data(
                self.symbols,
                self.config.start_date,
                self.config.end_date,
                self.config.factor_names,
                look_back_days=self.config.look_back_days,
                field_aliases=self.config.market_field_aliases,
            )
        self.executor = Executor(
            cost=self.cost_model,
            validate_trading_rules=self.config.validate_trading_rules,
            price_jump_limit=self.config.price_jump_limit,
        )
        self.strategy.on_init(self.context)
        precompute_signals = self.config.precompute_signals
        if precompute_signals is None:
            precompute_signals = bool(getattr(self.strategy, "precompute_stock_signals", False))
        self.precompute_signals = bool(precompute_signals)
        self.signal_errors: List[Dict[str, str]] = []
        if self.precompute_signals:
            self.signal_errors = self.engine._phase1_precompute_signals(
                self.strategy,
                self.context,
                self._new_account(),
                self.config.max_workers,
                self.config.error_policy,
            )
        self.reset()

    def reset(self, start_index: int = 0, end_index: Optional[int] = None) -> None:
        if not 0 <= int(start_index) < len(self.context.trade_dates):
            raise ValueError("start_index is outside the trade calendar")
        self.account = self._new_account()
        self.cursor = int(start_index)
        self.end_index = min(len(self.context.trade_dates), int(end_index)) if end_index is not None else len(self.context.trade_dates)
        if self.end_index <= self.cursor:
            raise ValueError("end_index must be after start_index")
        reset = getattr(getattr(self.strategy, "position_manager", None), "reset", None)
        if callable(reset):
            reset()

    @property
    def done(self) -> bool:
        return self.cursor >= self.end_index

    @property
    def date(self) -> Optional[str]:
        return None if self.done else self.context.trade_dates[self.cursor]

    def observe(self):
        if self.done:
            return None, None
        date = self.context.trade_dates[self.cursor]
        state = self.engine._build_policy_state(date, self.context, self.account)
        if self.precompute_signals:
            selection = self.context.get_stock_signals(date)
        else:
            selection = self.strategy.get_stock_signal(state)
        from quantx.core.strategy.base import StockSelection

        if isinstance(selection, list):
            selection = StockSelection(signals=selection)
        return state, selection

    def step_with_plan(self, plan):
        """Execute one date through the existing RuleExecution + Executor path."""
        if self.done:
            raise StopIteration("BacktestSession is finished")
        from quantx.core.strategy.base import WeightAllocation

        state, selection = self.observe()
        state.extra["position_plan"] = plan
        state.extra["planned_sell_symbols"] = set(plan.full_exit_symbols)
        orders = self.strategy.execution.act(state, WeightAllocation(weights=dict(plan.entry_weights)))
        return self._execute_observed(state, selection, orders)

    def step(self):
        """Execute one ordinary strategy day through the same formal path."""
        if self.done:
            raise StopIteration("BacktestSession is finished")
        state, selection = self.observe()
        orders = self.strategy.get_trade_signal(state, selection)
        return self._execute_observed(state, selection, orders)

    def _execute_observed(self, state, selection, orders):
        date = state.date
        before_trade_count = len(self.account.trades)
        if orders and hasattr(orders, "orders"):
            self.executor.execute_batch(orders.orders, self.account, self.exchange)
        self.account.update_daily_balance(date, self.exchange)
        self.cursor += 1
        return {
            "date": date,
            "state": state,
            "selection": selection,
            "orders": list(getattr(orders, "orders", []) or []),
            "trades": list(self.account.trades[before_trade_count:]),
            "snapshot": self.account.daily_snapshots[-1],
            "done": self.done,
        }

    def _new_account(self) -> Account:
        return Account(
            init_cash=self.config.init_cash,
            cost=self.cost_model,
            legacy_cost_price=self.config.legacy_cost_price,
            auto_adjust_buy_quantity=self.config.auto_adjust_buy_quantity,
        )

    @staticmethod
    def _configure_historical_st(exchange: AStockExchange, path: Optional[str]) -> None:
        if path:
            exchange.set_historical_st_path(path)


class BacktestEngine:
    """两阶段回测引擎

    Phase 1: 并行预计算选股信号
        - 每个交易日调用 strategy.get_stock_signal(state)
        - 使用 ThreadPoolExecutor 并行计算
        - 结果存入信号矩阵

    Phase 2: 顺序执行交易
        - 按交易日顺序遍历
        - 调仓 + 生成订单在 strategy 内部完成:
          strategy.rebalance.act(state, selection) → WeightAllocation
          strategy.execution.act(state, allocation) → OrderList
        - Executor 校验 + 执行订单
        - Account 更新持仓和净值
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    @staticmethod
    def _configure_historical_st(exchange: AStockExchange, path: Optional[str]) -> None:
        if path:
            exchange.set_historical_st_path(path)

    def create_session(
        self,
        strategy,
        symbols: List[str],
        config: Optional[BacktestConfig] = None,
        shared_context: Optional[BacktestContext] = None,
    ) -> BacktestSession:
        """Create a stateful replay session using this engine's exact components."""
        return BacktestSession(self, strategy, symbols, config=config or self.config, shared_context=shared_context)

    def run(
        self,
        strategy,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """运行完整回测

        Args:
            strategy: 策略对象，需实现 get_stock_signal(state) 和 get_trade_signal(state, selection)
            symbols: 股票列表
            start: 起始日期
            end: 结束日期
            config: 回测配置
        """
        cfg = config or self.config
        start = start or cfg.start_date
        end = end or cfg.end_date
        cost_model = cfg.cost or TransactionCost()

        time_stats = {}
        t0 = time.time()

        # 初始化各组件
        board = BoardManager()
        exchange = AStockExchange(board, provider_uri=cfg.provider_uri)
        self._configure_historical_st(exchange, cfg.historical_st_path)
        exchange.deal_price = cfg.deal_price
        setter = getattr(exchange, "set_execution_prices", None)
        if callable(setter):
            setter(cfg.buy_deal_price, cfg.sell_deal_price)
        context = BacktestContext(exchange)
        context.load_data(
            symbols,
            start,
            end,
            cfg.factor_names,
            look_back_days=cfg.look_back_days,
            field_aliases=cfg.market_field_aliases,
        )

        account = Account(
            init_cash=cfg.init_cash,
            cost=cost_model,
            legacy_cost_price=cfg.legacy_cost_price,
            auto_adjust_buy_quantity=cfg.auto_adjust_buy_quantity,
        )
        executor = Executor(
            cost=cost_model,
            validate_trading_rules=cfg.validate_trading_rules,
            price_jump_limit=cfg.price_jump_limit,
        )

        strategy.on_init(context)

        precompute_signals = cfg.precompute_signals
        if precompute_signals is None:
            precompute_signals = bool(getattr(strategy, "precompute_stock_signals", False))

        # Phase 1: 并行预计算选股信号。只有纯行情/纯因子选股器才能安全预计算。
        t1 = time.time()
        if precompute_signals:
            signal_errors = self._phase1_precompute_signals(
                strategy, context, account, cfg.max_workers, cfg.error_policy
            )
        else:
            signal_errors = []
        time_stats["phase1_signals"] = time.time() - t1

        # Phase 2: 顺序执行交易
        t2 = time.time()
        while not context.is_finished():
            date = context.next()
            state = self._build_policy_state(date, context, account)
            if precompute_signals:
                selection = context.get_stock_signals(date)
            else:
                selection = strategy.get_stock_signal(state)

            from quantx.core.strategy.base import StockSelection

            if isinstance(selection, list):
                selection = StockSelection(signals=selection)
            order_list = strategy.get_trade_signal(state, selection)
            if order_list and hasattr(order_list, "orders"):
                executor.execute_batch(order_list.orders, account, exchange)

            account.update_daily_balance(date, exchange)

        time_stats["phase2_execution"] = time.time() - t2
        time_stats["total"] = time.time() - t0

        strategy.on_finish(context)

        selection_getter = getattr(context, "get_selection_candidates", None)
        daily_selection_getter = getattr(context, "get_daily_selection_candidates", None)
        return BacktestResult(
            daily_snapshots=account.daily_snapshots,
            trades=account.trades,
            selection_candidates=list(selection_getter() if callable(selection_getter) else []),
            daily_selection_candidates=list(daily_selection_getter() if callable(daily_selection_getter) else []),
            config=cfg,
            time_stats=time_stats,
            signal_errors=signal_errors,
            position_manager_decisions=_position_manager_decisions(strategy),
        )

    def _phase1_precompute_signals(
        self,
        strategy,
        context: BacktestContext,
        account: Account,
        max_workers: int,
        error_policy: str = "fail_fast",
    ) -> List[Dict[str, str]]:
        """Phase 1: 并行预计算所有日期的选股信号"""
        logger.info(f"Phase 1: Precomputing signals for {len(context.trade_dates)} days")
        if error_policy not in {"fail_fast", "continue"}:
            raise ValueError(f"Unsupported error_policy: {error_policy}")

        errors: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for date in context.trade_dates:
                state = self._build_policy_state(date, context, account)
                futures[pool.submit(strategy.get_stock_signal, state)] = date

            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        if hasattr(result, "signals"):
                            context.set_stock_signals(date, result.signals)
                        elif isinstance(result, list):
                            context.set_stock_signals(date, result)
                except Exception as e:
                    logger.error(f"Phase 1 error on {date}: {e}")
                    if error_policy == "fail_fast":
                        raise
                    errors.append({
                        "date": date,
                        "error_type": type(e).__name__,
                        "message": str(e),
                    })

        logger.info(f"Phase 1 complete: {len(context._signal_matrix)} days with signals")
        return errors

    def _build_policy_state(self, date: str, context: BacktestContext, account: Account) -> Any:
        """构建 PolicyState，供策略 Policy.act() 使用"""
        from quantx.core.strategy.base import PolicyState, AccountSnapshot, PositionSnapshot

        return PolicyState(
            date=date,
            market_data=context.get_current_data(date),
            factor_data=context.get_factor_data(date),
            context=context,
            account=AccountSnapshot(
                cash=account.cash,
                total_value=account.get_total_value(),
                daily_return=account.latest_daily_return,
                cumulative_return=account.cumulative_return,
                drawdown=account.latest_drawdown,
            ),
            positions={
                sym: PositionSnapshot(
                    symbol=sym, quantity=pos.quantity,
                    avg_cost=pos.avg_cost, market_value=pos.market_value,
                    weight=pos.market_value / account.get_total_value() if account.get_total_value() > 0 else 0.0,
                    holding_days=pos.holding_days,
                    unrealized_pnl=pos.market_value - pos.avg_cost * pos.quantity,
                    highest_price=pos.highest_price,
                    lowest_price=pos.lowest_price,
                    initial_quantity=pos.initial_quantity or pos.quantity,
                    context=dict(pos.context),
                )
                for sym, pos in account.positions.items()
            },
            extra={
                "account_return_history": [
                    float(getattr(snapshot, "daily_return", 0.0))
                    for snapshot in account.daily_snapshots[-30:]
                ],
            },
        )
