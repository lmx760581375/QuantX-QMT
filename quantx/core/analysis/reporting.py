"""Build and persist standard QuantX backtest run artifacts."""

from __future__ import annotations

import json
import math
import shutil
from datetime import date, datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from quantx.core.engine.engine import BacktestResult
from quantx.core.engine.types import OrderAction


@dataclass
class RunReport:
    run_id: str
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    daily_nav: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]
    selection_candidates: List[Dict[str, Any]] = field(default_factory=list)
    daily_selection_candidates: List[Dict[str, Any]] = field(default_factory=list)
    explain: Dict[str, Any] = field(default_factory=dict)


def build_run_report(
    result: BacktestResult,
    config: Dict[str, Any],
    symbols: Iterable[str],
    run_id: str,
) -> RunReport:
    """Convert a BacktestResult into visualizable artifacts."""
    daily_nav = _build_daily_nav(result)
    trades = _build_trades(result)
    positions = _build_positions(result)
    closed_positions = match_closed_positions(trades)
    selection_candidates = list(getattr(result, "selection_candidates", []) or [])
    daily_selection_candidates = list(getattr(result, "daily_selection_candidates", []) or [])
    metrics = compute_metrics(daily_nav, trades, closed_positions, result.config.init_cash if result.config else 0.0)
    summary = _build_summary(result, config, symbols, run_id, metrics)
    artifact_refs = _artifact_refs(config)
    if artifact_refs:
        summary["artifact_refs"] = artifact_refs
    explain = {
        "config": config,
        "engine": asdict(result.config) if result.config is not None else None,
        "time_stats": result.time_stats,
        "signal_errors": result.signal_errors,
        "artifact_refs": artifact_refs,
    }
    return RunReport(
        run_id=run_id,
        summary=summary,
        metrics=metrics,
        daily_nav=daily_nav,
        trades=trades,
        positions=positions,
        closed_positions=closed_positions,
        selection_candidates=selection_candidates,
        daily_selection_candidates=daily_selection_candidates,
        explain=explain,
    )


def write_run_artifacts(
    report: RunReport,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    logs: Optional[List[str]] = None,
) -> Path:
    """Persist a RunReport to a run directory."""
    run_dir = Path(output_dir) / report.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        src = Path(config_path)
        if src.exists():
            shutil.copyfile(src, run_dir / "config.yaml")

    _write_json(run_dir / "summary.json", report.summary)
    _write_json(run_dir / "metrics.json", report.metrics)
    _write_json(run_dir / "daily_nav.json", report.daily_nav)
    _write_json(run_dir / "trades.json", report.trades)
    _write_json(run_dir / "positions.json", report.positions)
    _write_json(run_dir / "closed_positions.json", report.closed_positions)
    _write_json(run_dir / "selection_candidates.json", report.selection_candidates)
    _write_json(run_dir / "daily_selection_candidates.json", report.daily_selection_candidates)
    _write_json(run_dir / "explain.json", report.explain)
    (run_dir / "logs.txt").write_text("\n".join(logs or []), encoding="utf-8")
    return run_dir


def load_run_artifacts(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    return {
        "summary": _read_json(run_dir / "summary.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
        "daily_nav": _read_json(run_dir / "daily_nav.json"),
        "trades": _read_json(run_dir / "trades.json"),
        "positions": _read_json(run_dir / "positions.json"),
        "closed_positions": _read_json(run_dir / "closed_positions.json"),
        "selection_candidates": _read_json(run_dir / "selection_candidates.json", default=[]),
        "daily_selection_candidates": _read_json(run_dir / "daily_selection_candidates.json", default=[]),
        "explain": _read_json(run_dir / "explain.json"),
    }


def compute_metrics(
    daily_nav: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    closed_positions: List[Dict[str, Any]],
    init_cash: float,
) -> Dict[str, Any]:
    if not daily_nav or init_cash <= 0:
        return {}

    nav = pd.DataFrame(daily_nav)
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date")
    values = pd.to_numeric(nav["total_value"], errors="coerce")
    cash = pd.to_numeric(nav["cash"], errors="coerce") if "cash" in nav else pd.Series(float("nan"), index=nav.index)
    utilization = ((values - cash) / values.where(values > 0)).clip(lower=0.0, upper=1.0)
    returns = values.pct_change().fillna(0.0)
    final_value = float(values.iloc[-1])
    total_return = final_value / init_cash - 1
    days = max((nav["date"].iloc[-1] - nav["date"].iloc[0]).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(252))
    downside = returns[returns < 0]
    downside_volatility = float(downside.std(ddof=0) * math.sqrt(252)) if not downside.empty else 0.0
    max_drawdown = float(pd.to_numeric(nav["drawdown"], errors="coerce").min())

    executed_trades = [t for t in trades if not t.get("reject_reason")]
    rejected_trades = [t for t in trades if t.get("reject_reason")]
    sell_trades = [t for t in executed_trades if t.get("action") == "SELL"]
    buy_trades = [t for t in executed_trades if t.get("action") == "BUY"]
    closed = pd.DataFrame(closed_positions)
    wins = closed[closed["net_pnl"] > 0] if not closed.empty else pd.DataFrame()
    losses = closed[closed["net_pnl"] < 0] if not closed.empty else pd.DataFrame()
    gross_profit = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["net_pnl"].sum()) if not losses.empty else 0.0

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "downside_volatility": downside_volatility,
        "sharpe": _safe_div(annual_return, annual_volatility),
        "sortino": _safe_div(annual_return, downside_volatility),
        "calmar": _safe_div(annual_return, abs(max_drawdown)),
        "max_drawdown": max_drawdown,
        "order_count": len(trades),
        "trade_count": len(executed_trades),
        "reject_count": len(rejected_trades),
        "buy_count": len(buy_trades),
        "sell_count": len(sell_trades),
        "closed_position_count": len(closed_positions),
        "win_rate": _safe_div(len(wins), len(closed_positions)),
        "profit_factor": _safe_div(gross_profit, abs(gross_loss)),
        "avg_closed_return": float(closed["return"].mean()) if not closed.empty else 0.0,
        "avg_holding_days": float(closed["holding_days"].mean()) if not closed.empty else 0.0,
        "avg_position_count": float(pd.to_numeric(nav["position_count"], errors="coerce").mean()),
        "max_position_count": int(pd.to_numeric(nav["position_count"], errors="coerce").max()),
        "avg_capital_utilization": float(utilization.mean()),
        "recent_30_capital_utilization": float(utilization.tail(30).mean()),
        "recent_60_capital_utilization": float(utilization.tail(60).mean()),
        "high_utilization_day_ratio": float((utilization >= 0.7).mean()),
        "zero_utilization_day_ratio": float((utilization <= 1e-6).mean()),
        "total_cost": float(sum(t.get("total_cost", 0.0) for t in executed_trades)),
    }


def match_closed_positions(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FIFO match trades into closed positions."""
    lots: Dict[str, List[Dict[str, Any]]] = {}
    closed: List[Dict[str, Any]] = []
    for trade in trades:
        if trade.get("reject_reason"):
            continue
        symbol = trade["symbol"]
        action = trade["action"]
        quantity = int(trade["quantity"])
        if quantity <= 0:
            continue
        if action == "BUY":
            lots.setdefault(symbol, []).append(
                {
                    "date": trade["date"],
                    "quantity": quantity,
                    "price": float(trade["price"]),
                    "cost_per_share": (float(trade["trade_value"]) + float(trade.get("total_cost", 0.0))) / quantity,
                }
            )
        elif action == "SELL":
            remaining = quantity
            queue = lots.get(symbol, [])
            while remaining > 0 and queue:
                lot = queue[0]
                matched = min(remaining, int(lot["quantity"]))
                entry_cost = float(lot["cost_per_share"]) * matched
                exit_value = float(trade["price"]) * matched
                exit_cost = float(trade.get("total_cost", 0.0)) * matched / quantity
                net_pnl = exit_value - exit_cost - entry_cost
                entry_date = pd.Timestamp(lot["date"])
                exit_date = pd.Timestamp(trade["date"])
                closed.append(
                    {
                        "symbol": symbol,
                        "entry_date": entry_date.strftime("%Y-%m-%d"),
                        "exit_date": exit_date.strftime("%Y-%m-%d"),
                        "quantity": matched,
                        "entry_price": float(lot["price"]),
                        "exit_price": float(trade["price"]),
                        "holding_days": int((exit_date - entry_date).days),
                        "return": _safe_div(net_pnl, entry_cost),
                        "net_pnl": net_pnl,
                    }
                )
                lot["quantity"] -= matched
                remaining -= matched
                if lot["quantity"] <= 0:
                    queue.pop(0)
    return closed


def _artifact_refs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    strategy = dict(config.get("strategy") or {})
    alpha = dict(strategy.get("alpha") or {})
    artifact_id = alpha.get("artifact_id")
    if not artifact_id:
        return []
    return [
        {
            "artifact_id": str(artifact_id),
            "feature_schema_hash": alpha.get("feature_schema_hash"),
            "prediction_store": alpha.get("path"),
            "prediction_checksum": alpha.get("checksum"),
        }
    ]


def _build_summary(
    result: BacktestResult,
    config: Dict[str, Any],
    symbols: Iterable[str],
    run_id: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    snapshots = result.daily_snapshots
    return {
        "run_id": run_id,
        "name": config.get("name"),
        "description": config.get("description"),
        "version": config.get("version"),
        "start_date": snapshots[0].date if snapshots else None,
        "end_date": snapshots[-1].date if snapshots else None,
        "symbols": len(list(symbols)),
        "final_value": metrics.get("final_value", 0.0),
        "total_return": metrics.get("total_return", 0.0),
        "annual_return": metrics.get("annual_return", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "trades": metrics.get("trade_count", 0),
        "buys": metrics.get("buy_count", 0),
        "sells": metrics.get("sell_count", 0),
        "final_cash": snapshots[-1].cash if snapshots else 0.0,
        "final_positions": len(snapshots[-1].positions) if snapshots else 0,
        "time_stats": result.time_stats,
        "signal_errors": result.signal_errors,
    }


def _build_daily_nav(result: BacktestResult) -> List[Dict[str, Any]]:
    rows = []
    running_max = None
    for snapshot in result.daily_snapshots:
        total_value = float(snapshot.total_value)
        running_max = total_value if running_max is None else max(running_max, total_value)
        drawdown = total_value / running_max - 1 if running_max and running_max > 0 else 0.0
        rows.append(
            {
                "date": snapshot.date,
                "cash": float(snapshot.cash),
                "total_value": total_value,
                "daily_return": float(snapshot.daily_return),
                "cumulative_return": float(snapshot.cumulative_return),
                "drawdown": float(drawdown),
                "position_count": len(snapshot.positions),
            }
        )
    return rows


def _build_trades(result: BacktestResult) -> List[Dict[str, Any]]:
    rows = []
    for trade in result.trades:
        action = trade.action.name if isinstance(trade.action, OrderAction) else str(trade.action)
        rows.append(
            {
                "date": trade.date,
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
                "reason": getattr(trade, "reason", ""),
            }
        )
    return rows


def _build_positions(result: BacktestResult) -> List[Dict[str, Any]]:
    rows = []
    for snapshot in result.daily_snapshots:
        total_value = float(snapshot.total_value)
        for symbol, position in snapshot.positions.items():
            rows.append(
                {
                    "date": snapshot.date,
                    "symbol": symbol,
                    "quantity": int(position.quantity),
                    "avg_cost": float(position.avg_cost),
                    "market_value": float(position.market_value),
                    "weight": _safe_div(float(position.market_value), total_value),
                    "holding_days": int(position.holding_days),
                    "highest_price": float(position.highest_price),
                    "lowest_price": float(position.lowest_price),
                    "initial_quantity": int(getattr(position, "initial_quantity", 0) or position.quantity),
                }
            )
    return rows


def _safe_div(a: float, b: float) -> float:
    if b is None or abs(b) < 1e-12:
        return 0.0
    return float(a) / float(b)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
