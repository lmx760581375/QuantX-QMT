"""Run shuijiao with QuantX's Qlib data layer and report alignment metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def _ensure_local_qlib_importable() -> None:
    # AStockExchange prefers the installed qlib package. Keep this hook for
    # backwards compatibility with older run commands without forcing the
    # repo's uncompiled qlib source tree onto sys.path.
    return


_ensure_local_qlib_importable()

from quantx.core.engine import BacktestConfig, BacktestEngine, TransactionCost  # noqa: E402
from quantx.core.engine.types import OrderAction  # noqa: E402
from quantx.strategies.shuijiao_strategy import create_shuijiao_strategy  # noqa: E402


BASELINE_TOTAL_RETURN = 0.8738455329265315
BASELINE_FINAL_VALUE = 1_873_845.5329265315
BASELINE_TRADES = 480
BASELINE_SELLS = 235


def load_a_share_symbols(provider_uri: str | Path, start: str, end: str, limit: int | None = None) -> List[str]:
    instruments = Path(provider_uri) / "instruments" / "all.txt"
    symbols: List[str] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    with open(instruments) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            sym, listed, delisted = parts[:3]
            if not (sym.startswith("SH6") or sym.startswith("SZ0") or sym.startswith("SZ3")):
                continue
            if pd.Timestamp(delisted) < start_ts or pd.Timestamp(listed) > end_ts:
                continue
            symbols.append(sym)
            if limit is not None and len(symbols) >= limit:
                break
    return symbols


def baseline_like_params(max_positions: int) -> Dict[str, Any]:
    return {
        "enable_market_health_entry": False,
        "enable_rsi_filter": False,
        "enable_kdj_filter": False,
        "enable_bbi_filter": False,
        "enable_dynamic_positions": False,
        "enable_hard_risk_off": False,
        "enable_momentum_sell": False,
        "max_positions_base": max_positions,
        "max_profit_pct": 0.20,
        "max_loss_pct": -0.10,
        "deal_price": "close",
        "legacy_wrap_first_signal": False,
        "legacy_preserve_signal_order": True,
        "legacy_reuse_sell_cash_for_buys": True,
        "commission_rate": 0.0005,
        "min_commission": 5.0,
        "stamp_tax_rate": 0.0001,
        "transfer_fee_rate": 0.0,
        "skip_buy_limit_up": True,
        "log_trades": False,
    }


def summarize(result, symbols: Iterable[str], strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = result.daily_snapshots
    trades = result.trades
    final_value = snapshots[-1].total_value if snapshots else 0.0
    total_return = final_value / result.config.init_cash - 1 if result.config else 0.0
    dates = [s.date for s in snapshots]
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days if len(dates) > 1 else 0
    annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
    values = pd.Series([s.total_value for s in snapshots], index=pd.to_datetime(dates))
    drawdown = values / values.cummax() - 1 if not values.empty else pd.Series(dtype=float)
    buys = [t for t in trades if t.action == OrderAction.BUY]
    sells = [t for t in trades if t.action == OrderAction.SELL]
    rejects = [t for t in trades if t.reject_reason]
    return {
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "symbols": len(list(symbols)),
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "trades": len(trades),
        "buys": len(buys),
        "sells": len(sells),
        "rejects": len(rejects),
        "final_cash": snapshots[-1].cash if snapshots else 0.0,
        "final_positions": len(snapshots[-1].positions) if snapshots else 0,
        "baseline_final_value": BASELINE_FINAL_VALUE,
        "baseline_total_return": BASELINE_TOTAL_RETURN,
        "baseline_trades": BASELINE_TRADES,
        "baseline_sells": BASELINE_SELLS,
        "diff_final_value": final_value - BASELINE_FINAL_VALUE,
        "diff_total_return": total_return - BASELINE_TOTAL_RETURN,
        "diff_trades": len(trades) - BASELINE_TRADES,
        "diff_sells": len(sells) - BASELINE_SELLS,
        "time_stats": result.time_stats,
        "signal_errors": getattr(result, "signal_errors", []),
        "strategy_params": strategy_params,
    }


def run(args) -> Dict[str, Any]:
    symbols = load_a_share_symbols(args.provider_uri, args.start, args.end, args.symbol_limit)
    params = baseline_like_params(args.max_positions)
    params["legacy_wrap_first_signal"] = args.legacy_wrap_first_signal
    params["skip_buy_limit_up"] = not args.allow_limit_up_buy
    params["legacy_reuse_sell_cash_for_buys"] = not args.no_legacy_reuse_sell_cash
    params["max_profit_pct"] = args.max_profit_pct
    params["max_loss_pct"] = args.max_loss_pct
    strategy = create_shuijiao_strategy(**params)
    cfg = BacktestConfig(
        init_cash=args.init_cash,
        start_date=args.start,
        end_date=args.end,
        provider_uri=args.provider_uri,
        deal_price="close",
        max_workers=args.max_workers,
        look_back_days=args.look_back_days,
        cost=TransactionCost(
            commission_rate=0.0005,
            min_commission=5.0,
            stamp_tax_rate=0.0001,
            transfer_fee_rate=0.0,
            slippage=0.0,
            stamp_tax_on_buy=True,
        ),
        validate_trading_rules=args.validate_trading_rules,
        legacy_cost_price=True,
        auto_adjust_buy_quantity=False,
    )
    result = BacktestEngine(cfg).run(strategy, symbols)
    return summarize(result, symbols, params)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--start", default="2021-01-04")
    parser.add_argument("--end", default="2025-10-17")
    parser.add_argument("--init-cash", type=float, default=1_000_000)
    parser.add_argument("--max-positions", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--look-back-days", type=int, default=100)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument("--validate-trading-rules", action="store_true")
    parser.add_argument("--legacy-wrap-first-signal", action="store_true")
    parser.add_argument("--allow-limit-up-buy", action="store_true")
    parser.add_argument("--no-legacy-reuse-sell-cash", action="store_true")
    parser.add_argument("--max-profit-pct", type=float, default=0.20)
    parser.add_argument("--max-loss-pct", type=float, default=-0.10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
