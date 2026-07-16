"""Run a QuantX config-driven backtest from YAML."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from quantx.core.engine import BacktestConfig, BacktestEngine, TransactionCost
from quantx.core.engine.types import OrderAction
from quantx.core.engine.context import compute_load_start, validate_trading_calendar_coverage
from quantx.core.analysis import build_run_report, write_run_artifacts
from quantx.core.strategy.factory import build_strategy, explain_strategy


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_latest_date(provider_uri: str | Path) -> str:
    calendar_path = Path(provider_uri) / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing qlib calendar file: {calendar_path}")
    dates = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not dates:
        raise FileNotFoundError(f"Missing or empty qlib calendar: {calendar_path}")
    return dates[-1]


def resolve_config_dates(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    data_cfg = dict(resolved.get("data") or {})
    provider_uri = data_cfg.get("provider_uri")
    if provider_uri and str(data_cfg.get("end", "")).lower() in {"latest", "auto"}:
        data_cfg["end"] = resolve_latest_date(provider_uri)
    if provider_uri and str(data_cfg.get("start", "")).lower() in {"earliest", "first"}:
        calendar_path = Path(provider_uri) / "calendars" / "day.txt"
        dates = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not dates:
            raise FileNotFoundError(f"Missing or empty qlib calendar: {calendar_path}")
        data_cfg["start"] = dates[0]
    resolved["data"] = data_cfg
    return resolved


def load_a_share_symbols(
    provider_uri: str | Path,
    start: str,
    end: str,
    limit: int | None = None,
    universe: str = "all_a",
) -> List[str]:
    instruments = Path(provider_uri) / "instruments" / "all.txt"
    if not instruments.exists():
        raise FileNotFoundError(f"Missing qlib instruments file: {instruments}")
    symbols: List[str] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    with open(instruments, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            sym, listed, delisted = parts[:3]
            if not _is_a_share_symbol(sym):
                continue
            if universe == "all_mainboard" and (
                sym.startswith("SH688") or sym.startswith("SZ300") or sym.startswith("SZ301")
            ):
                continue
            if pd.Timestamp(delisted) < start_ts or pd.Timestamp(listed) > end_ts:
                continue
            symbols.append(sym)
            if limit is not None and len(symbols) >= limit:
                break
    return symbols


def _is_a_share_symbol(sym: str) -> bool:
    """Return true for stock instruments, excluding indices such as SZ399xxx."""
    return (
        sym.startswith("SH600")
        or sym.startswith("SH601")
        or sym.startswith("SH603")
        or sym.startswith("SH605")
        or sym.startswith("SH688")
        or sym.startswith("SZ000")
        or sym.startswith("SZ001")
        or sym.startswith("SZ002")
        or sym.startswith("SZ003")
        or sym.startswith("SZ300")
        or sym.startswith("SZ301")
    )


def build_cost(config: Dict[str, Any]) -> TransactionCost:
    cost_cfg = config.get("cost") or {}
    return TransactionCost(
        commission_rate=float(cost_cfg.get("commission_rate", 0.0003)),
        min_commission=float(cost_cfg.get("min_commission", 5.0)),
        stamp_tax_rate=float(cost_cfg.get("stamp_tax_rate", 0.0005)),
        transfer_fee_rate=float(cost_cfg.get("transfer_fee_rate", 0.00002)),
        slippage=float(cost_cfg.get("slippage", 0.001)),
        buy_slippage=(
            float(cost_cfg["buy_slippage"])
            if cost_cfg.get("buy_slippage") is not None else None
        ),
        sell_slippage=(
            float(cost_cfg["sell_slippage"])
            if cost_cfg.get("sell_slippage") is not None else None
        ),
        stamp_tax_on_buy=bool(cost_cfg.get("stamp_tax_on_buy", False)),
    )


def build_backtest_config(config: Dict[str, Any], cost: TransactionCost) -> BacktestConfig:
    config = resolve_config_dates(config)
    data_cfg = config.get("data") or {}
    engine_cfg = config.get("engine") or {}
    execution_cfg = config.get("execution") or {}
    if not data_cfg.get("provider_uri"):
        raise ValueError("data.provider_uri is required")
    if not data_cfg.get("start") or not data_cfg.get("end"):
        raise ValueError("data.start and data.end are required")
    return BacktestConfig(
        init_cash=float(engine_cfg.get("init_cash", 1_000_000)),
        start_date=str(data_cfg["start"]),
        end_date=str(data_cfg["end"]),
        provider_uri=str(data_cfg["provider_uri"]),
        cost=cost,
        max_workers=int(engine_cfg.get("max_workers", 1)),
        deal_price=str(execution_cfg.get("deal_price", engine_cfg.get("deal_price", "open"))),
        look_back_days=int(data_cfg.get("look_back_days", 0)),
        validate_trading_rules=bool(engine_cfg.get("validate_trading_rules", True)),
        legacy_cost_price=bool(engine_cfg.get("legacy_cost_price", False)),
        auto_adjust_buy_quantity=bool(engine_cfg.get("auto_adjust_buy_quantity", True)),
        error_policy=str(engine_cfg.get("error_policy", "fail_fast")),
        market_field_aliases=dict(config.get("fields") or {}),
        precompute_signals=engine_cfg.get("precompute_signals"),
    )


def validate_backtest_data_coverage(config: BacktestConfig) -> Dict[str, Any]:
    calendar_path = Path(config.provider_uri) / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing qlib calendar file: {calendar_path}")

    all_dates = pd.DatetimeIndex(
        pd.to_datetime(
            [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()],
            errors="coerce",
        )
    ).dropna().sort_values()
    load_start = compute_load_start(config.start_date, config.look_back_days)
    dates = all_dates[(all_dates >= pd.Timestamp(load_start)) & (all_dates <= pd.Timestamp(config.end_date))]
    validate_trading_calendar_coverage(
        dates,
        start=config.start_date,
        end=config.end_date,
        load_start=load_start,
        look_back_days=config.look_back_days,
        provider_uri=config.provider_uri,
    )
    return {
        "load_start": load_start,
        "first_available_date": dates[0].strftime("%Y-%m-%d") if len(dates) else None,
        "calendar_rows": int(len(dates)),
    }


def load_symbols(config: Dict[str, Any], limit: int | None = None) -> List[str]:
    config = resolve_config_dates(config)
    data_cfg = config.get("data") or {}
    universe = data_cfg.get("universe", "all_a")
    if isinstance(universe, list):
        return [str(sym) for sym in universe[:limit]]
    if universe == "external_score":
        selector = dict(config.get("selector") or {})
        path = selector.get("path")
        date_col = str(selector.get("date_col", "date"))
        instrument_col = str(selector.get("instrument_col", "instrument"))
        score_col = selector.get("score_col")
        score_floor = selector.get("score_floor")
        topk = selector.get("topk")
        sort = selector.get("sort", "score_desc")
        if not path:
            raise ValueError("external_score universe requires selector.path")
        score_path = Path(path)
        suffix = score_path.suffix.lower()
        columns = [instrument_col]
        if score_col:
            columns.extend([date_col, str(score_col)])
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(score_path, columns=list(dict.fromkeys(columns)))
        elif suffix == ".csv":
            frame = pd.read_csv(score_path, usecols=list(dict.fromkeys(columns)))
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(score_path, lines=suffix == ".jsonl")[list(dict.fromkeys(columns))]
        else:
            raise ValueError(f"Unsupported external_score universe file type: {score_path.suffix}")
        if score_col and date_col in frame.columns and score_col in frame.columns:
            frame = _external_score_selected_rows(
                frame,
                date_col=date_col,
                score_col=str(score_col),
                score_floor=score_floor,
                topk=topk,
                sort=str(sort),
            )
        symbols = tuple(sorted(frame[instrument_col].dropna().astype(str).unique()))
        extras = tuple(str(symbol) for symbol in data_cfg.get("extra_symbols", ()))
        combined = tuple(dict.fromkeys((*symbols, *extras)))
        return list(combined[:limit])
    if universe == "prediction_store":
        from quantx.core.decision.predictions import PredictionStore

        alpha = dict((config.get("strategy") or {}).get("alpha") or {})
        if not alpha.get("path"):
            raise ValueError("prediction_store universe requires strategy.alpha.path")
        store = PredictionStore.load(
            alpha["path"], expected_checksum=alpha.get("checksum")
        )
        pool_topk = data_cfg.get("prediction_pool_topk")
        if pool_topk is None:
            symbols = store.instruments
        else:
            symbols = store.top_instruments(artifact_id=alpha.get("artifact_id"), top_k=int(pool_topk))
        extras = tuple(str(symbol) for symbol in data_cfg.get("extra_symbols", ()))
        combined = tuple(dict.fromkeys((*symbols, *extras)))
        return list(combined[:limit])
    if universe == "wufu_etf":
        from quantx.strategies.universe_presets import wufu_etf_symbols

        symbols = wufu_etf_symbols()
        return symbols[:limit] if limit is not None else symbols
    if universe not in {"all_a", "all_mainboard"}:
        raise ValueError(f"Unsupported data.universe: {universe}")
    return load_a_share_symbols(
        data_cfg["provider_uri"],
        str(data_cfg["start"]),
        str(data_cfg["end"]),
        limit=limit if limit is not None else data_cfg.get("symbol_limit"),
        universe=universe,
    )


def _external_score_selected_rows(
    frame: pd.DataFrame,
    *,
    date_col: str,
    score_col: str,
    score_floor: Any,
    topk: Any,
    sort: str,
) -> pd.DataFrame:
    selected = frame.copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    selected[score_col] = pd.to_numeric(selected[score_col], errors="coerce")
    selected = selected.dropna(subset=[date_col, score_col])
    if score_floor is not None:
        selected = selected.loc[selected[score_col] >= float(score_floor)]
    pieces = []
    for _, day in selected.groupby(date_col, sort=True):
        if sort == "score_desc":
            day = day.sort_values(score_col, ascending=False)
        elif sort == "score_asc":
            day = day.sort_values(score_col, ascending=True)
        if topk is not None:
            day = day.head(int(topk))
        pieces.append(day)
    return pd.concat(pieces, ignore_index=True) if pieces else selected.iloc[0:0]


def summarize(result, symbols: Iterable[str], config: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = result.daily_snapshots
    trades = result.trades
    final_value = snapshots[-1].total_value if snapshots else 0.0
    init_cash = result.config.init_cash if result.config else 1.0
    total_return = final_value / init_cash - 1 if init_cash else 0.0
    dates = [s.date for s in snapshots]
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days if len(dates) > 1 else 0
    annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
    values = pd.Series([s.total_value for s in snapshots], index=pd.to_datetime(dates))
    drawdown = values / values.cummax() - 1 if not values.empty else pd.Series(dtype=float)
    buys = [t for t in trades if t.action == OrderAction.BUY]
    sells = [t for t in trades if t.action == OrderAction.SELL]
    rejects = [t for t in trades if t.reject_reason]
    return {
        "name": config.get("name"),
        "version": config.get("version"),
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
        "time_stats": result.time_stats,
        "signal_errors": getattr(result, "signal_errors", []),
    }


def run_config(config_path: str | Path, symbol_limit: int | None = None) -> Dict[str, Any]:
    config = resolve_config_dates(load_config(config_path))
    explain_strategy(config)
    cost = build_cost(config)
    cfg = build_backtest_config(config, cost)
    validate_backtest_data_coverage(cfg)
    symbols = load_symbols(config, limit=symbol_limit)
    strategy = build_strategy(config, cost)
    result = BacktestEngine(cfg).run(strategy, symbols)
    return summarize(result, symbols, config)


def run_config_with_artifacts(
    config_path: str | Path,
    output_dir: str | Path,
    symbol_limit: int | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    config = resolve_config_dates(load_config(config_path))
    explain = explain_strategy(config)
    cost = build_cost(config)
    cfg = build_backtest_config(config, cost)
    validate_backtest_data_coverage(cfg)
    symbols = load_symbols(config, limit=symbol_limit)
    strategy = build_strategy(config, cost)
    result = BacktestEngine(cfg).run(strategy, symbols)
    run_id = run_id or _make_run_id(config)
    report = build_run_report(result, config, symbols, run_id)
    report.explain["strategy"] = explain
    with TemporaryDirectory() as tmp_dir:
        resolved_config_path = Path(tmp_dir) / "config.yaml"
        resolved_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        run_dir = write_run_artifacts(report, output_dir, config_path=resolved_config_path)
    summary = dict(report.summary)
    summary["run_dir"] = str(run_dir)
    return summary


def dry_run_config(config_path: str | Path, symbol_limit: int | None = None) -> Dict[str, Any]:
    config = resolve_config_dates(load_config(config_path))
    cost = build_cost(config)
    backtest_config = build_backtest_config(config, cost)
    coverage = validate_backtest_data_coverage(backtest_config)
    symbols = load_symbols(config, limit=symbol_limit)
    strategy_explain = explain_strategy(config)
    build_strategy(config, cost)
    return {
        "ok": True,
        "config": str(config_path),
        "strategy": strategy_explain,
        "data": {
            "provider_uri": backtest_config.provider_uri,
            "start": backtest_config.start_date,
            "end": backtest_config.end_date,
            "look_back_days": backtest_config.look_back_days,
            "load_start": coverage["load_start"],
            "first_available_date": coverage["first_available_date"],
            "calendar_rows": coverage["calendar_rows"],
            "symbols": len(symbols),
            "symbol_limit": symbol_limit,
        },
        "engine": {
            "init_cash": backtest_config.init_cash,
            "deal_price": backtest_config.deal_price,
            "validate_trading_rules": backtest_config.validate_trading_rules,
            "legacy_cost_price": backtest_config.legacy_cost_price,
            "auto_adjust_buy_quantity": backtest_config.auto_adjust_buy_quantity,
            "error_policy": backtest_config.error_policy,
            "precompute_signals": backtest_config.precompute_signals,
        },
    }


def _make_run_id(config: Dict[str, Any]) -> str:
    name = str(config.get("name") or "strategy").replace("/", "_").replace(" ", "_")
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to strategy YAML config.")
    parser.add_argument("--symbol-limit", type=int, help="Override data.symbol_limit for smoke tests.")
    parser.add_argument("--output-dir", help="Write standard run artifacts under this directory.")
    parser.add_argument("--run-id", help="Optional run id when writing artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Compile and validate config without running trades.")
    parser.add_argument("--explain", action="store_true", help="Print strategy formula DAG and runner settings.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    if args.dry_run or args.explain:
        result = dry_run_config(args.config, symbol_limit=args.symbol_limit)
    elif args.output_dir:
        result = run_config_with_artifacts(
            args.config,
            args.output_dir,
            symbol_limit=args.symbol_limit,
            run_id=args.run_id,
        )
    else:
        result = run_config(args.config, symbol_limit=args.symbol_limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
