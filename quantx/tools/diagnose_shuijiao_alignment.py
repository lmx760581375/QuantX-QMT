"""Diagnose QuantX shuijiao alignment against the old myquant baseline.

This tool intentionally bypasses Qlib data loading and feeds the old myquant
pickle into QuantX. That isolates the backtest engine/account/execution layer
from data conversion and factor recomputation differences.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from quantx.core.engine import Account, AStockExchange, BacktestConfig, BoardManager, Executor, TransactionCost
from quantx.core.engine.context import BacktestContext
from quantx.core.engine.types import OrderAction
from quantx.core.factor_runtime import FactorRuntime, MarketPanel
from quantx.core.strategy.base import AccountSnapshot, PolicyState, PositionSnapshot
from quantx.strategies.shuijiao_strategy import create_shuijiao_strategy


BASELINE_TOTAL_RETURN = 0.8738455329265315
BASELINE_FINAL_VALUE = 1_873_845.5329265315
BASELINE_TRADES = 480
BASELINE_SELLS = 235


def load_myquant_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("myquant pkl must be a MultiIndex DataFrame")
    frame = frame.copy().sort_index()
    frame.index = frame.index.set_names(["datetime", "instrument"])
    return frame


def build_quantx_quote(frame: pd.DataFrame) -> pd.DataFrame:
    quote = pd.DataFrame(index=frame.index)
    quote["$open"] = frame["open"]
    quote["$high"] = frame["high"]
    quote["$low"] = frame["low"]
    quote["$close"] = frame["close"]
    quote["$volume"] = frame.get("volume", 1.0)
    quote["$change"] = frame.get("pct", 0.0)
    quote["$vwap"] = frame["vwap"] if "vwap" in frame.columns else frame["close"]
    quote["$factor"] = 1.0
    if "limit_up" in frame.columns:
        quote["limit_up"] = frame["limit_up"]
    if "limit_down" in frame.columns:
        quote["limit_down"] = frame["limit_down"]
    return quote


def _matrix_from_column(frame: pd.DataFrame, dates: pd.DatetimeIndex, instruments: pd.Index, column: str):
    wide = frame[column].unstack("instrument").reindex(index=dates, columns=instruments)
    if wide.dtypes.astype(str).str.contains("bool").any():
        return wide.fillna(False).to_numpy(dtype=bool)
    return wide.to_numpy(dtype=np.float32)


def build_runtime_from_myquant(frame: pd.DataFrame, quote: pd.DataFrame) -> Tuple[MarketPanel, FactorRuntime]:
    panel = MarketPanel.from_frame(quote)
    runtime = FactorRuntime(panel)
    for name in ["midline", "trend_line", "buy_signal", "ready_sell"]:
        runtime.values[name] = _matrix_from_column(frame, panel.dates, panel.instruments, name)
    trend_line = np.asarray(runtime.values["trend_line"], dtype=np.float32)
    runtime.values["market_health"] = np.nanmean((trend_line > 20).astype(np.float32), axis=1).astype(np.float32)
    return panel, runtime


def make_context(frame: pd.DataFrame) -> BacktestContext:
    quote = build_quantx_quote(frame)
    exchange = AStockExchange(BoardManager())
    exchange.load_quote_from_raw(quote)
    exchange.deal_price = "close"
    context = BacktestContext(exchange)
    context.market_data = quote
    context.trade_dates = pd.to_datetime(quote.index.get_level_values("datetime").unique()).strftime("%Y-%m-%d").tolist()
    context.market_panel, context.factor_runtime = build_runtime_from_myquant(frame, quote)
    return context


def build_state(date: str, context: BacktestContext, account: Account) -> PolicyState:
    total_value = account.get_total_value()
    return PolicyState(
        date=date,
        market_data=context.get_current_data(date),
        factor_data=None,
        context=context,
        account=AccountSnapshot(
            cash=account.cash,
            total_value=total_value,
            daily_return=account.latest_daily_return,
            cumulative_return=account.cumulative_return,
        ),
        positions={
            sym: PositionSnapshot(
                symbol=sym,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_value=pos.market_value,
                weight=pos.market_value / total_value if total_value > 0 else 0.0,
                holding_days=pos.holding_days,
                unrealized_pnl=pos.market_value - pos.avg_cost * pos.quantity,
                highest_price=pos.highest_price,
                lowest_price=pos.lowest_price,
                initial_quantity=pos.initial_quantity or pos.quantity,
            )
            for sym, pos in account.positions.items()
        },
    )


def myquant_legacy_params() -> Dict[str, Any]:
    return {
        "enable_market_health_entry": False,
        "enable_rsi_filter": False,
        "enable_kdj_filter": False,
        "enable_bbi_filter": False,
        "enable_dynamic_positions": False,
        "enable_hard_risk_off": False,
        "enable_momentum_sell": False,
        # The checked-in myquant source says 100, but outputs/trades.json behaves
        # like a 10-position portfolio. Keep the diagnostic tied to the artifact
        # we are aligning against.
        "max_positions_base": 10,
        "max_profit_pct": 0.20,
        "max_loss_pct": -0.10,
        "deal_price": "close",
        "legacy_full_cash_per_buy": False,
        "legacy_equal_cash_when_empty": False,
        "legacy_reuse_sell_cash_for_buys": True,
        "commission_rate": 0.0005,
        "min_commission": 5.0,
        "stamp_tax_rate": 0.0001,
        "transfer_fee_rate": 0.0,
        "legacy_wrap_first_signal": True,
        "legacy_preserve_signal_order": True,
        "skip_buy_limit_up": True,
        "log_trades": False,
    }


def run_quantx_legacy(frame: pd.DataFrame) -> Dict[str, Any]:
    context = make_context(frame)
    cost = TransactionCost(
        commission_rate=0.0005,
        min_commission=5.0,
        stamp_tax_rate=0.0001,
        transfer_fee_rate=0.0,
        slippage=0.0,
        stamp_tax_on_buy=True,
    )
    cfg = BacktestConfig(
        init_cash=1_000_000,
        deal_price="close",
        validate_trading_rules=False,
        legacy_cost_price=True,
        auto_adjust_buy_quantity=False,
        cost=cost,
    )
    account = Account(
        init_cash=cfg.init_cash,
        cost=cost,
        legacy_cost_price=cfg.legacy_cost_price,
        auto_adjust_buy_quantity=cfg.auto_adjust_buy_quantity,
    )
    executor = Executor(cost=cost, validate_trading_rules=cfg.validate_trading_rules)
    strategy = create_shuijiao_strategy(**myquant_legacy_params())
    strategy.on_init(context)

    for date in context.trade_dates:
        state = build_state(date, context, account)
        selection = strategy.get_stock_signal(state)
        context.set_stock_signals(date, selection.signals)

    while not context.is_finished():
        date = context.next()
        state = build_state(date, context, account)
        selection = context.get_stock_signals(date)
        order_list = strategy.get_trade_signal(state, type("Selection", (), {"signals": selection})())
        if order_list and hasattr(order_list, "orders"):
            executor.execute_batch(order_list.orders, account, context.exchange)
        account.update_daily_balance(date, context.exchange)

    return summarize(account, context.trade_dates)


def summarize(account: Account, trade_dates: Iterable[str]) -> Dict[str, Any]:
    snapshots = account.daily_snapshots
    final_value = snapshots[-1].total_value if snapshots else account.get_total_value()
    total_return = final_value / account.init_cash - 1
    dates = list(trade_dates)
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days if len(dates) > 1 else 0
    annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
    values = pd.Series([s.total_value for s in snapshots], index=pd.to_datetime([s.date for s in snapshots]))
    drawdown = values / values.cummax() - 1 if not values.empty else pd.Series(dtype=float)
    sells = [t for t in account.trades if t.action == OrderAction.SELL]
    buys = [t for t in account.trades if t.action == OrderAction.BUY]
    open_positions = snapshots[-1].positions if snapshots else account.positions
    return {
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "trades": len(account.trades),
        "buys": len(buys),
        "sells": len(sells),
        "final_cash": account.cash,
        "final_positions": len(open_positions),
        "baseline_final_value": BASELINE_FINAL_VALUE,
        "baseline_total_return": BASELINE_TOTAL_RETURN,
        "baseline_trades": BASELINE_TRADES,
        "baseline_sells": BASELINE_SELLS,
        "diff_final_value": final_value - BASELINE_FINAL_VALUE,
        "diff_total_return": total_return - BASELINE_TOTAL_RETURN,
        "diff_trades": len(account.trades) - BASELINE_TRADES,
        "diff_sells": len(sells) - BASELINE_SELLS,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        default="../myquant-strategy-baseline/data/multiindex_df_20210101_20251231.pkl",
        help="Path to old myquant multiindex dataframe pkl.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args(argv)

    frame = load_myquant_frame(args.pkl)
    result = run_quantx_legacy(frame)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
