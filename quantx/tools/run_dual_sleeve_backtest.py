"""运行两个独立 QuantX sleeve，并按月度或季度目标比例划拨可用现金。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from quantx.core.engine import BacktestEngine
from quantx.tools.run_backtest import (
    build_backtest_config,
    build_cost,
    load_config,
    load_symbols,
    resolve_config_dates,
)
from quantx.core.strategy.config_strategy import build_formula_strategy, explain_strategy_config


def is_rebalance_date(previous_date: str | None, date: str, schedule: str) -> bool:
    if previous_date is None or schedule == "none":
        return False
    previous = pd.Timestamp(previous_date)
    current = pd.Timestamp(date)
    if schedule == "monthly":
        return (previous.year, previous.month) != (current.year, current.month)
    if schedule == "quarterly":
        return (previous.year, previous.quarter) != (current.year, current.quarter)
    raise ValueError(f"Unsupported schedule: {schedule}")


def requested_transfer(
    wts_value: float,
    rm_value: float,
    *,
    wts_weight: float,
) -> tuple[str, str, float]:
    total = float(wts_value) + float(rm_value)
    target_wts = total * float(wts_weight)
    delta = target_wts - float(wts_value)
    if delta >= 0:
        return "rm", "wts", delta
    return "wts", "rm", -delta


def transfer_available_cash(
    accounts: dict[str, Any],
    *,
    source: str,
    target: str,
    requested: float,
) -> float:
    amount = min(max(0.0, float(requested)), max(0.0, float(accounts[source].cash)))
    if amount <= 0:
        return 0.0
    accounts[source].apply_external_cash_flow(-amount)
    accounts[target].apply_external_cash_flow(amount)
    return amount


def _metrics(nav: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(nav)
    values = pd.to_numeric(frame["total_value"], errors="coerce")
    returns = values.pct_change().fillna(0.0)
    days = (pd.Timestamp(frame["date"].iloc[-1]) - pd.Timestamp(frame["date"].iloc[0])).days
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (365.0 / days) - 1.0) if days > 0 else 0.0
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(252))
    downside = returns[returns < 0]
    downside_volatility = float(downside.std(ddof=0) * math.sqrt(252)) if not downside.empty else 0.0
    max_drawdown = float(pd.to_numeric(frame["drawdown"], errors="coerce").min())
    executed = [row for row in trades if not row.get("reject_reason")]
    return {
        "final_value": float(values.iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "downside_volatility": downside_volatility,
        "sharpe": annual_return / annual_volatility if annual_volatility > 0 else 0.0,
        "sortino": annual_return / downside_volatility if downside_volatility > 0 else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "order_count": len(trades),
        "trade_count": len(executed),
        "buy_count": sum(row.get("action") == "BUY" for row in executed),
        "sell_count": sum(row.get("action") == "SELL" for row in executed),
        "reject_count": len(trades) - len(executed),
        "total_cost": float(sum(float(row.get("total_cost", 0.0)) for row in executed)),
        "avg_position_count": float(pd.to_numeric(frame["position_count"], errors="coerce").mean()),
        "max_position_count": int(pd.to_numeric(frame["position_count"], errors="coerce").max()),
        "avg_cash_ratio": float(
            (pd.to_numeric(frame["cash"], errors="coerce") / values).mean()
        ),
    }


def _trade_row(trade, sleeve: str) -> dict[str, Any]:
    action = trade.action.name if hasattr(trade.action, "name") else str(trade.action)
    return {
        "date": trade.date,
        "sleeve": sleeve,
        "symbol": trade.symbol,
        "action": action,
        "price": float(trade.price),
        "quantity": int(trade.quantity),
        "trade_value": float(trade.trade_value),
        "total_cost": float(trade.total_cost),
        "commission": float(trade.commission),
        "stamp_tax": float(trade.stamp_tax),
        "transfer_fee": float(trade.transfer_fee),
        "slippage_cost": float(trade.slippage_cost),
        "reject_reason": trade.reject_reason,
        "reason": trade.reason,
    }


def _position_rows(snapshot, sleeve: str, combined_value: float) -> list[dict[str, Any]]:
    rows = []
    for symbol, position in snapshot.positions.items():
        rows.append({
            "date": snapshot.date,
            "sleeve": sleeve,
            "symbol": symbol,
            "quantity": int(position.quantity),
            "avg_cost": float(position.avg_cost),
            "market_value": float(position.market_value),
            "weight": float(position.market_value / combined_value) if combined_value > 0 else 0.0,
            "holding_days": int(position.holding_days),
            "highest_price": float(position.highest_price),
            "lowest_price": float(position.lowest_price),
            "initial_quantity": int(position.initial_quantity or position.quantity),
        })
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dual_sleeve(
    *,
    wts_config_path: str | Path,
    rm_config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    wts_weight: float,
    schedule: str,
    initial_cash: float | None = None,
    start: str | None = None,
    end: str | None = None,
    provider_uri: str | None = None,
    rm_score_path: str | None = None,
) -> dict[str, Any]:
    if not 0.0 < wts_weight < 1.0:
        raise ValueError("wts_weight must be within (0, 1)")
    if schedule not in {"none", "monthly", "quarterly"}:
        raise ValueError("schedule must be none, monthly, or quarterly")

    raw_configs = {
        "wts": load_config(wts_config_path),
        "rm": load_config(rm_config_path),
    }
    if start:
        for config in raw_configs.values():
            config["data"]["start"] = str(start)
    if end:
        for config in raw_configs.values():
            config["data"]["end"] = str(end)
    if provider_uri:
        for config in raw_configs.values():
            config["data"]["provider_uri"] = str(provider_uri)
    if rm_score_path:
        raw_configs["rm"]["selector"]["external_score"]["path"] = str(rm_score_path)
    raw_configs = {
        sleeve: resolve_config_dates(config)
        for sleeve, config in raw_configs.items()
    }
    for config in raw_configs.values():
        explain_strategy_config(config)
    starts = {str(config["data"]["start"]) for config in raw_configs.values()}
    ends = {str(config["data"]["end"]) for config in raw_configs.values()}
    providers = {str(config["data"]["provider_uri"]) for config in raw_configs.values()}
    if len(starts) != 1 or len(ends) != 1 or len(providers) != 1:
        raise ValueError("Both sleeve configs must use identical provider/start/end")

    initial_total = (
        float(initial_cash)
        if initial_cash is not None
        else sum(float(config["engine"]["init_cash"]) for config in raw_configs.values())
    )
    if initial_total <= 0:
        raise ValueError("initial_cash must be positive")
    initial_weights = {"wts": wts_weight, "rm": 1.0 - wts_weight}
    configs: dict[str, dict[str, Any]] = {}
    sessions = {}
    for sleeve in ("wts", "rm"):
        config = json.loads(json.dumps(raw_configs[sleeve]))
        config["engine"]["init_cash"] = initial_total * initial_weights[sleeve]
        configs[sleeve] = config
        cost = build_cost(config)
        backtest_config = build_backtest_config(config, cost)
        strategy = build_formula_strategy(config, cost)
        symbols = load_symbols(config)
        sessions[sleeve] = BacktestEngine(backtest_config).create_session(
            strategy, symbols, config=backtest_config
        )

    calendars = [session.context.trade_dates for session in sessions.values()]
    if calendars[0] != calendars[1]:
        raise ValueError("Sleeve trade calendars do not match")

    transfers: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    daily_nav: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    previous_date: str | None = None
    running_max = initial_total
    previous_total = initial_total

    while not sessions["wts"].done:
        date = sessions["wts"].date
        if date != sessions["rm"].date:
            raise RuntimeError("Sleeve sessions are not synchronized")

        if is_rebalance_date(previous_date, date, schedule):
            source, target, requested = requested_transfer(
                sessions["wts"].account.get_total_value(),
                sessions["rm"].account.get_total_value(),
                wts_weight=wts_weight,
            )
            pending = {
                "rebalance_date": date,
                "source": source,
                "target": target,
                "requested": float(requested),
                "remaining": float(requested),
            }

        if pending is not None and pending["remaining"] > 1e-8:
            amount = transfer_available_cash(
                {name: session.account for name, session in sessions.items()},
                source=pending["source"],
                target=pending["target"],
                requested=pending["remaining"],
            )
            pending["remaining"] -= amount
            transfers.append({
                "date": date,
                "rebalance_date": pending["rebalance_date"],
                "source": pending["source"],
                "target": pending["target"],
                "requested": pending["requested"],
                "transferred": amount,
                "remaining": max(0.0, pending["remaining"]),
            })
            if pending["remaining"] <= 1e-8:
                pending = None

        steps = {sleeve: session.step() for sleeve, session in sessions.items()}
        for sleeve, step in steps.items():
            trades.extend(_trade_row(trade, sleeve) for trade in step["trades"])

        snapshots = {sleeve: step["snapshot"] for sleeve, step in steps.items()}
        total_value = sum(float(snapshot.total_value) for snapshot in snapshots.values())
        total_cash = sum(float(snapshot.cash) for snapshot in snapshots.values())
        running_max = max(running_max, total_value)
        daily_nav.append({
            "date": date,
            "cash": total_cash,
            "total_value": total_value,
            "daily_return": total_value / previous_total - 1.0 if previous_total > 0 else 0.0,
            "cumulative_return": total_value / initial_total - 1.0,
            "drawdown": total_value / running_max - 1.0 if running_max > 0 else 0.0,
            "position_count": sum(len(snapshot.positions) for snapshot in snapshots.values()),
            "wts_value": float(snapshots["wts"].total_value),
            "rm_value": float(snapshots["rm"].total_value),
            "wts_weight": float(snapshots["wts"].total_value / total_value),
            "rm_weight": float(snapshots["rm"].total_value / total_value),
            "pending_transfer": float(pending["remaining"]) if pending else 0.0,
        })
        for sleeve, snapshot in snapshots.items():
            positions.extend(_position_rows(snapshot, sleeve, total_value))
        previous_total = total_value
        previous_date = date

    for session in sessions.values():
        session.strategy.on_finish(session.context)

    metrics = _metrics(daily_nav, trades)
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "name": f"dual_sleeve_wts_{wts_weight:.0%}_rm_{1-wts_weight:.0%}_{schedule}",
        "description": "WTS Top3 与 RM Top5 runner 独立账户，按计划仅划拨可用现金。",
        "start_date": daily_nav[0]["date"],
        "end_date": daily_nav[-1]["date"],
        "initial_cash": initial_total,
        "final_value": metrics["final_value"],
        "total_return": metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "trades": metrics["trade_count"],
        "buys": metrics["buy_count"],
        "sells": metrics["sell_count"],
        "final_cash": daily_nav[-1]["cash"],
        "final_positions": daily_nav[-1]["position_count"],
        "schedule": schedule,
        "target_weights": initial_weights,
        "transfer_event_count": len(transfers),
        "unfilled_transfer": float(pending["remaining"]) if pending else 0.0,
        "signal_errors": {
            sleeve: session.signal_errors for sleeve, session in sessions.items()
        },
    }
    explain = {
        "method": "independent accounts; cash-only transfers before trading; no forced liquidation",
        "schedule": schedule,
        "target_weights": initial_weights,
        "configs": configs,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "daily_nav.json", daily_nav)
    _write_json(run_dir / "trades.json", trades)
    _write_json(run_dir / "positions.json", positions)
    _write_json(run_dir / "transfers.json", transfers)
    _write_json(run_dir / "explain.json", explain)
    return {**summary, "run_dir": str(run_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行真实资金划拨的 QuantX 双 sleeve 回测")
    parser.add_argument("--wts-config", required=True)
    parser.add_argument("--rm-config", required=True)
    parser.add_argument("--wts-weight", type=float, required=True)
    parser.add_argument("--schedule", choices=["none", "monthly", "quarterly"], required=True)
    parser.add_argument("--initial-cash", type=float)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--provider-uri")
    parser.add_argument("--rm-score-path")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_dual_sleeve(
        wts_config_path=args.wts_config,
        rm_config_path=args.rm_config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        wts_weight=args.wts_weight,
        schedule=args.schedule,
        initial_cash=args.initial_cash,
        start=args.start,
        end=args.end,
        provider_uri=args.provider_uri,
        rm_score_path=args.rm_score_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
