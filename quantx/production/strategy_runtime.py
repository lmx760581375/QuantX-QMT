"""Daily strategy execution helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd
import yaml

from quantx.core.analysis import annual_returns, build_next_session_guide, load_run_artifacts
from quantx.core.data.meta import MetaStore
from quantx.tools.run_backtest import load_config, run_config_with_artifacts

from .artifacts import write_json
from .config import ProductionProfile


StrategyRunner = Callable[[Path, Path, str, int | None], Dict[str, Any]]


@dataclass(frozen=True)
class DailyStrategyRuntime:
    """Run configured strategies and summarize their latest production state."""

    profile: ProductionProfile
    trade_date: str
    output_dir: Path
    meta_store: MetaStore
    symbol_limit: int | None = None
    runner: StrategyRunner | None = None

    def run_all(self, strategy_filter: str | None = None) -> List[Dict[str, Any]]:
        rows = []
        for strategy_path in self.profile.strategies:
            if strategy_filter and strategy_filter not in strategy_path:
                continue
            rows.append(self.run_one(strategy_path))
        return rows

    def run_one(self, strategy_path: str) -> Dict[str, Any]:
        source_path = _resolve_path(self.profile.paths.root, strategy_path)
        config = load_config(source_path)
        strategy_name = str(config.get("name") or source_path.stem)
        strategy_title = _strategy_title(config, strategy_name)
        run_id = f"{self.trade_date.replace('-', '')}_{strategy_name}"
        strategy_dir = self.output_dir / "strategy_runs" / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        prepared_config_path = self._prepare_config(source_path, config, strategy_dir)
        runs_dir = self.profile.paths.root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Any]
        try:
            runner = self.runner or _default_strategy_runner
            result = runner(prepared_config_path, runs_dir, run_id, self.symbol_limit)
            artifacts = load_run_artifacts(runs_dir / run_id)
            summary = self._summarize_artifacts(
                strategy_path=strategy_path,
                strategy_name=strategy_name,
                strategy_title=strategy_title,
                description=str(config.get("description") or ""),
                run_id=run_id,
                run_dir=str(runs_dir / run_id),
                artifacts=artifacts,
            )
            summary["ok"] = True
            summary["runner_summary"] = result
            return summary
        except Exception as exc:
            error = {
                "ok": False,
                "strategy_path": strategy_path,
                "strategy_name": strategy_name,
                "strategy_title": strategy_title,
                "display_title": strategy_title,
                "title": strategy_title,
                "description": str(config.get("description") or ""),
                "run_id": run_id,
                "run_dir": str(strategy_dir / run_id),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            write_json(strategy_dir / "error.json", error)
            return error

    def _prepare_config(self, source_path: Path, config: Dict[str, Any], strategy_dir: Path) -> Path:
        prepared = dict(config)
        data_cfg = dict(prepared.get("data") or {})
        data_cfg["end"] = self.trade_date
        if self.profile.data.provider_uri:
            data_cfg["provider_uri"] = str(_resolve_path(self.profile.paths.root, self.profile.data.provider_uri))
        prepared["data"] = data_cfg
        prepared_path = strategy_dir / "daily_config.yaml"
        prepared_path.write_text(yaml.safe_dump(prepared, allow_unicode=True, sort_keys=False), encoding="utf-8")
        shutil.copyfile(source_path, strategy_dir / "source_config.yaml")
        return prepared_path

    def _summarize_artifacts(
        self,
        strategy_path: str,
        strategy_name: str,
        strategy_title: str,
        description: str,
        run_id: str,
        run_dir: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = artifacts.get("summary") or {}
        latest_date = str(summary.get("end_date") or self.trade_date)
        trades = [dict(row) for row in artifacts.get("trades") or []]
        positions = [dict(row) for row in artifacts.get("positions") or []]
        daily_nav = [dict(row) for row in artifacts.get("daily_nav") or []]
        selection_candidates = [dict(row) for row in artifacts.get("selection_candidates") or []]
        daily_selection_candidates = [dict(row) for row in artifacts.get("daily_selection_candidates") or []]
        today_trades = [row for row in trades if str(row.get("date")) == latest_date]
        accepted = [row for row in today_trades if not row.get("reject_reason")]
        rejected = [row for row in today_trades if row.get("reject_reason")]
        buys = [row for row in accepted if row.get("action") == "BUY"]
        sells = [row for row in accepted if row.get("action") == "SELL"]
        latest_positions = [
            row for row in positions
            if str(row.get("date")) == latest_date and int(row.get("quantity") or 0) > 0
        ]
        enriched_symbols = sorted({
            *(row.get("symbol") for row in today_trades if row.get("symbol")),
            *(row.get("symbol") for row in latest_positions if row.get("symbol")),
            *(row.get("symbol") for row in trades[-5:] if row.get("symbol")),
            *(row.get("symbol") for row in reversed(trades) if row.get("reject_reason") and row.get("symbol")),
            *(row.get("symbol") for item in selection_candidates[-3:] for row in item.get("selected_candidates", []) if row.get("symbol")),
            *(row.get("symbol") for item in selection_candidates[-3:] for row in item.get("raw_candidates", []) if row.get("symbol")),
            *(row.get("symbol") for item in daily_selection_candidates[-3:] for row in item.get("selected_candidates", []) if row.get("symbol")),
            *(row.get("symbol") for item in daily_selection_candidates[-3:] for row in item.get("raw_candidates", []) if row.get("symbol")),
        })
        meta = _meta_map(self.meta_store, enriched_symbols)
        latest_candidates = _latest_selection_candidates(selection_candidates, latest_date, meta)
        next_session_candidates = _latest_daily_selection_candidates(daily_selection_candidates, latest_date, meta)
        recent_trades = [
            _enrich_trade(row, meta)
            for row in reversed(trades)
            if not row.get("reject_reason")
        ][:5]
        recent_trades.reverse()
        recent_rejections = [
            _enrich_trade(row, meta)
            for row in reversed(trades)
            if row.get("reject_reason")
        ][:5]
        return {
            "strategy_path": strategy_path,
            "strategy_name": strategy_name,
            "strategy_title": strategy_title,
            "display_title": strategy_title,
            "title": strategy_title,
            "description": description,
            "run_id": run_id,
            "run_dir": run_dir,
            "latest_date": latest_date,
            "summary": summary,
            "metrics": artifacts.get("metrics") or {},
            "buys": [_enrich_trade(row, meta) for row in buys],
            "sells": [_enrich_trade(row, meta) for row in sells],
            "rejected_orders": [_enrich_trade(row, meta) for row in rejected],
            "positions": [_enrich_position(row, meta) for row in latest_positions],
            "selection_candidates": latest_candidates,
            "next_session_candidates": next_session_candidates,
            "next_session_guide": build_next_session_guide(
                next_session_candidates,
                [_enrich_position(row, meta) for row in latest_positions],
                config=(artifacts.get("explain") or {}).get("config") or {},
                latest_date=latest_date,
            ),
            "annual_returns": annual_returns(daily_nav),
            "recent_trades": recent_trades,
            "recent_rejections": recent_rejections,
            "equity_curve": _equity_curve(daily_nav),
            "trade_count_today": len(today_trades),
            "buy_count_today": len(buys),
            "sell_count_today": len(sells),
            "reject_count_today": len(rejected),
            "position_count": len(latest_positions),
            "notes": [
                "第一版日度生产使用完整回放生成模拟持仓和当日交易建议。",
                "邮件内容为 QuantX 策略模拟结果，不代表真实账户持仓。",
            ],
        }


def _default_strategy_runner(config_path: Path, output_dir: Path, run_id: str, symbol_limit: int | None) -> Dict[str, Any]:
    return run_config_with_artifacts(
        config_path,
        output_dir,
        symbol_limit=symbol_limit,
        run_id=run_id,
    )


def _strategy_title(config: Dict[str, Any], fallback: str) -> str:
    for key in ("display_title", "title"):
        value = str(config.get(key) or "").strip()
        if value:
            return value[:10]
    return fallback


def _resolve_path(root: Path, path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else root / item


def _meta_map(meta_store: MetaStore, symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    symbols = [sym for sym in symbols if sym]
    if not symbols:
        return {}
    frame = meta_store.get_symbol_meta(symbols)
    if frame.empty:
        return {}
    return {str(idx): row.dropna().to_dict() for idx, row in frame.iterrows()}


def _enrich_trade(row: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item = dict(row)
    item.update(_meta_fields(item.get("symbol"), meta))
    return item


def _enrich_position(row: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item = dict(row)
    qty = int(item.get("quantity") or 0)
    market_value = float(item.get("market_value") or 0.0)
    avg_cost = float(item.get("avg_cost") or 0.0)
    current_price = market_value / qty if qty > 0 else 0.0
    item["current_price"] = current_price
    item["unrealized_pnl"] = (current_price - avg_cost) * qty
    item["unrealized_return"] = current_price / avg_cost - 1 if avg_cost > 0 else 0.0
    item.update(_meta_fields(item.get("symbol"), meta))
    return item


def _latest_selection_candidates(
    rows: List[Dict[str, Any]],
    latest_date: str,
    meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {}
    exact = [row for row in rows if str(row.get("date")) == latest_date]
    item = exact[-1] if exact else rows[-1]
    result = dict(item)
    result["is_latest_trade_date"] = str(item.get("date")) == latest_date
    result["raw_candidates"] = [_enrich_trade(row, meta) for row in item.get("raw_candidates") or []]
    result["selected_candidates"] = [_enrich_trade(row, meta) for row in item.get("selected_candidates") or []]
    return result


def _latest_daily_selection_candidates(
    rows: List[Dict[str, Any]],
    latest_date: str,
    meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {}
    exact = [row for row in rows if str(row.get("date")) == latest_date]
    item = exact[-1] if exact else rows[-1]
    result = dict(item)
    result["is_latest_signal_date"] = str(item.get("date")) == latest_date
    result["raw_candidates"] = [_enrich_trade(row, meta) for row in item.get("raw_candidates") or []]
    result["selected_candidates"] = [_enrich_trade(row, meta) for row in item.get("selected_candidates") or []]
    return result


def _meta_fields(symbol: Any, meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = meta.get(str(symbol), {})
    return {
        "name": row.get("name") or symbol,
        "industry_name": row.get("industry_name") or "",
        "industry_code": row.get("industry_code") or "",
    }


def _equity_curve(daily_nav: List[Dict[str, Any]], max_points: int = 240) -> List[Dict[str, Any]]:
    if not daily_nav:
        return []
    base_value = _first_positive_total_value(daily_nav)
    if len(daily_nav) <= max_points:
        rows = daily_nav
    else:
        step = max(1, len(daily_nav) // max_points)
        rows = daily_nav[::step]
        if rows[-1] is not daily_nav[-1]:
            rows = [*rows, daily_nav[-1]]
    curve = []
    for row in rows:
        curve.append({
            "date": row.get("date"),
            "total_value": row.get("total_value"),
            "cumulative_return": _cumulative_return(row, base_value),
            "drawdown": row.get("drawdown"),
            "position_count": row.get("position_count"),
        })
    return curve


def _first_positive_total_value(daily_nav: List[Dict[str, Any]]) -> float | None:
    for row in daily_nav:
        try:
            value = float(row.get("total_value"))
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _cumulative_return(row: Dict[str, Any], base_value: float | None) -> Any:
    value = row.get("cumulative_return")
    if value is not None:
        return value
    if not base_value:
        return None
    try:
        return float(row.get("total_value")) / base_value - 1.0
    except Exception:
        return None
