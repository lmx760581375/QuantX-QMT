"""两阶段回测引擎

Phase 1: 并行预计算所有日期的选股信号
Phase 2: 按交易日顺序执行交易（必须顺序，依赖前日状态）
"""

import logging
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from .account import Account
from .board import BoardManager
from .context import BacktestContext
from .cost import TransactionCost
from .exchange import AStockExchange
from .executor import Executor
from .types import DailySnapshot, Order, Signal, Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    init_cash: float = 1_000_000
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "SH000300"
    provider_uri: str = "data/qlib_data"
    cost: Optional[TransactionCost] = None
    factor_names: List[str] = field(default_factory=list)
    max_workers: int = 8
    deal_price: str = "open"
    look_back_days: int = 0
    validate_trading_rules: bool = True
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
        exchange.deal_price = cfg.deal_price
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
        executor = Executor(cost=cost_model, validate_trading_rules=cfg.validate_trading_rules)

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

        return BacktestResult(
            daily_snapshots=account.daily_snapshots,
            trades=account.trades,
            selection_candidates=context.get_selection_candidates(),
            daily_selection_candidates=context.get_daily_selection_candidates(),
            config=cfg,
            time_stats=time_stats,
            signal_errors=signal_errors,
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
        )
