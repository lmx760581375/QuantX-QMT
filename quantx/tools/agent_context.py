"""Agent-facing JSON context commands for QuantX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from quantx.core.analysis.reporting import compute_metrics, load_run_artifacts
from quantx.core.data import AdjustmentAwareIncrementalUpdater, BaoStockClient, BaostockToQlibConverter, LocalDataRepository
from quantx.core.data.meta import MetaStore, normalize_symbol
from quantx.tools.run_backtest import dry_run_config, run_config_with_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER = "data/qlib_data_fixed"
DEFAULT_META_URI = PROJECT_ROOT / "data" / "meta" / "quantx_meta.sqlite"


METRIC_SPECS = [
    {"name": "final_value", "display_name": "Final Value", "category": "return", "required_artifacts": ["daily_nav"]},
    {"name": "total_return", "display_name": "Total Return", "category": "return", "required_artifacts": ["daily_nav"]},
    {"name": "annual_return", "display_name": "Annual Return", "category": "return", "required_artifacts": ["daily_nav"]},
    {"name": "annual_volatility", "display_name": "Annual Volatility", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "downside_volatility", "display_name": "Downside Volatility", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "sharpe", "display_name": "Sharpe", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "sortino", "display_name": "Sortino", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "calmar", "display_name": "Calmar", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "max_drawdown", "display_name": "Max Drawdown", "category": "risk", "required_artifacts": ["daily_nav"]},
    {"name": "trade_count", "display_name": "Trade Count", "category": "trading", "required_artifacts": ["trades"]},
    {"name": "win_rate", "display_name": "Win Rate", "category": "trading", "required_artifacts": ["closed_positions"]},
    {"name": "profit_factor", "display_name": "Profit Factor", "category": "trading", "required_artifacts": ["closed_positions"]},
    {"name": "avg_holding_days", "display_name": "Avg Holding Days", "category": "trading", "required_artifacts": ["closed_positions"]},
    {"name": "total_cost", "display_name": "Total Cost", "category": "cost", "required_artifacts": ["trades"]},
]


def _json_default(value: Any) -> Any:
    try:
        return value.item()
    except Exception:
        return str(value)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _runs_root(root: Path) -> Path:
    return root / "runs"


def _latest_run(root: Path) -> Path | None:
    runs = _runs_root(root)
    if not runs.exists():
        return None
    candidates = [path for path in runs.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_run_dir(root: Path, run_id: str) -> Path:
    if run_id == "latest":
        latest = _latest_run(root)
        if latest is None:
            raise FileNotFoundError("No run with summary.json found under runs/")
        return latest
    run_dir = _runs_root(root) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return run_dir


def _load_report(root: Path, run_id: str) -> tuple[Path, Dict[str, Any]]:
    run_dir = _resolve_run_dir(root, run_id)
    data = load_run_artifacts(run_dir)
    data["run_id"] = run_dir.name
    return run_dir, data


def _meta_rows(symbols: Iterable[str], meta_uri: Path = DEFAULT_META_URI) -> List[Dict[str, Any]]:
    store = MetaStore(meta_uri)
    frame = store.get_symbol_meta(symbols)
    rows = []
    for symbol in symbols:
        norm = normalize_symbol(symbol)
        if frame.empty or norm not in frame.index:
            rows.append({"symbol": norm, "found": False})
            continue
        row = frame.loc[norm].to_dict()
        row["found"] = bool(row.get("name") or row.get("industry_name"))
        rows.append(row)
    return rows


def cmd_status(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    configs = root / "configs" / "strategies"
    runs = _runs_root(root)
    return {
        "ok": True,
        "project_root": str(root),
        "has_pyproject": (root / "pyproject.toml").exists(),
        "has_run_backtest": (root / "quantx" / "tools" / "run_backtest.py").exists(),
        "has_agent_context": True,
        "configs": len(list(configs.rglob("*.yaml"))) + len(list(configs.rglob("*.yml"))) if configs.exists() else 0,
        "runs": len([path for path in runs.iterdir() if path.is_dir()]) if runs.exists() else 0,
        "qlib_provider_exists": (root / DEFAULT_PROVIDER).exists(),
        "meta_exists": DEFAULT_META_URI.exists(),
    }


def cmd_data_status(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    provider = Path(args.provider_uri)
    if not provider.is_absolute():
        provider = root / provider
    calendar = provider / "calendars" / "day.txt"
    instruments = provider / "instruments" / "all.txt"
    dates = [line.strip() for line in calendar.read_text(encoding="utf-8").splitlines() if line.strip()] if calendar.exists() else []
    inst = [line.strip() for line in instruments.read_text(encoding="utf-8").splitlines() if line.strip()] if instruments.exists() else []
    feature_count = len(list((provider / "features").glob("*/*.bin"))) if (provider / "features").exists() else 0
    return {
        "ok": provider.exists() and bool(dates) and bool(inst),
        "provider_uri": str(provider),
        "calendar_start": dates[0] if dates else None,
        "calendar_end": dates[-1] if dates else None,
        "calendar_days": len(dates),
        "instrument_count": len(inst),
        "feature_file_count": feature_count,
    }


def cmd_data_update(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    provider = Path(args.provider_uri)
    if not provider.is_absolute():
        provider = root / provider
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    repository = LocalDataRepository(str(raw_dir))
    converter = BaostockToQlibConverter(qlib_dir=str(provider), csv_dir=str(raw_dir / "stocks"))
    updater = AdjustmentAwareIncrementalUpdater(
        repository=repository,
        converter=converter,
        client_factory=lambda: BaoStockClient(pause_seconds=args.pause_seconds, socket_timeout=args.socket_timeout),
        overlap_days=args.overlap_days,
        tolerance=args.tolerance,
        full_refresh_start=args.full_refresh_start,
        max_requests=args.max_requests,
    )
    progress_every = int(args.progress_every or 0)

    def _progress(report):
        done = len(report.symbol_results)
        if progress_every > 0 and (done == 1 or done % progress_every == 0 or done == report.total_symbols):
            pct = (done / report.total_symbols * 100.0) if report.total_symbols else 100.0
            latest = report.symbol_results[-1] if report.symbol_results else None
            symbol_part = (
                f" symbol={latest.symbol} status={latest.status} rows_added={latest.rows_added}"
                if latest is not None
                else ""
            )
            print(
                f"progress {done}/{report.total_symbols} ({pct:.2f}%) "
                f"requests={report.request_count} updated={len(report.updated_symbols)} "
                f"full_refresh={len(report.full_refresh_symbols)} failed={len(report.failed_symbols)} "
                f"skipped={len(report.skipped_symbols)}{symbol_part}",
                file=sys.stderr,
                flush=True,
            )

    report = updater.update(
        symbols=_load_update_symbols(root, args),
        end=args.end_date,
        limit=args.limit,
        dry_run=args.dry_run,
        progress_callback=_progress if progress_every > 0 else None,
    ).to_dict()
    return {"ok": bool(report.get("ok")), "provider_uri": str(provider), "raw_dir": str(raw_dir), "report": report}


def _load_update_symbols(root: Path, args) -> List[str] | None:
    symbols = list(args.symbols or [])
    symbol_file = getattr(args, "symbol_file", None)
    if symbol_file:
        path = Path(symbol_file)
        if not path.is_absolute():
            path = root / path
        symbols.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return symbols or None


def cmd_meta(args) -> Dict[str, Any]:
    meta_uri = Path(args.meta_uri)
    if not meta_uri.is_absolute():
        meta_uri = Path(args.root).resolve() / meta_uri
    return {"ok": True, "meta_uri": str(meta_uri), "symbols": _meta_rows(args.symbols, meta_uri)}


def cmd_validate_config(args) -> Dict[str, Any]:
    result = dry_run_config(args.config, symbol_limit=args.symbol_limit)
    return {"ok": True, **result}


def cmd_run(args) -> Dict[str, Any]:
    if args.dry_run:
        return cmd_validate_config(args)
    summary = run_config_with_artifacts(args.config, args.output_dir, symbol_limit=args.symbol_limit, run_id=args.run_id)
    return {"ok": True, "summary": summary, "run_id": summary.get("run_id"), "run_dir": summary.get("run_dir")}


def cmd_latest_run(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    run = _latest_run(root)
    if run is None:
        return {"ok": False, "message": "No run found"}
    summary = _read_json(run / "summary.json", {})
    metrics = _read_json(run / "metrics.json", {})
    return {
        "ok": True,
        "run_id": run.name,
        "path": str(run),
        "name": summary.get("name"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "total_return": metrics.get("total_return", summary.get("total_return")),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "trade_count": metrics.get("trade_count", summary.get("trades")),
    }


def _report_summary(root: Path, run_id: str) -> Dict[str, Any]:
    run_dir, data = _load_report(root, run_id)
    trades = data.get("trades") or []
    valid_trades = [trade for trade in trades if not trade.get("reject_reason")]
    rejects = [trade for trade in trades if trade.get("reject_reason")]
    by_symbol = Counter(trade.get("symbol") for trade in valid_trades if trade.get("symbol"))
    top_symbols = [symbol for symbol, _ in by_symbol.most_common(10)]
    meta = {row["symbol"]: row for row in _meta_rows(top_symbols)} if top_symbols else {}
    top = []
    for symbol, count in by_symbol.most_common(10):
        norm = normalize_symbol(symbol)
        top.append({
            "symbol": norm,
            "name": meta.get(norm, {}).get("name"),
            "industry_name": meta.get(norm, {}).get("industry_name"),
            "trade_count": count,
        })
    return {
        "ok": True,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "summary": data.get("summary") or {},
        "metrics": data.get("metrics") or {},
        "daily_nav_count": len(data.get("daily_nav") or []),
        "trade_count": len(valid_trades),
        "reject_count": len(rejects),
        "reject_summary": dict(Counter(trade.get("reject_reason") for trade in rejects)),
        "top_traded_symbols": top,
        "artifact_paths": {path.stem: str(path) for path in sorted(run_dir.glob("*.json"))},
    }


def cmd_report(args) -> Dict[str, Any]:
    return _report_summary(Path(args.root).resolve(), args.run_id)


def cmd_symbol(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    run_dir, data = _load_report(root, args.run_id)
    symbol = normalize_symbol(args.symbol)
    trades = [trade for trade in data.get("trades", []) if normalize_symbol(trade.get("symbol", "")) == symbol and not trade.get("reject_reason")]
    closed = [pos for pos in data.get("closed_positions", []) if normalize_symbol(pos.get("symbol", "")) == symbol]
    returns = [pos.get("return") for pos in closed if isinstance(pos.get("return"), (int, float))]
    meta = _meta_rows([symbol])[0]
    return {
        "ok": True,
        "run_id": run_dir.name,
        "symbol": symbol,
        "name": meta.get("name"),
        "industry_name": meta.get("industry_name"),
        "trade_count": len(trades),
        "round_trips": len(closed),
        "avg_round_trip_return": sum(returns) / len(returns) if returns else None,
        "trades": trades[:20],
    }


def cmd_diagnose(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    report = _report_summary(root, args.run_id)
    data = cmd_data_status(argparse.Namespace(root=str(root), provider_uri=args.provider_uri))
    checks = [
        {"name": "data_coverage", "status": "pass" if data.get("ok") else "fail", "details": data},
        {"name": "rejected_orders", "status": "warn" if report.get("reject_count") else "pass", "count": report.get("reject_count")},
        {"name": "trade_count", "status": "warn" if not report.get("trade_count") else "pass", "count": report.get("trade_count")},
        {"name": "artifacts", "status": "pass" if report.get("artifact_paths") else "fail", "artifacts": sorted((report.get("artifact_paths") or {}).keys())},
    ]
    likely = []
    if report.get("reject_count"):
        likely.append("Rejected orders may affect returns; inspect reject_summary and execution rules.")
    if not data.get("ok"):
        likely.append("Qlib provider coverage is incomplete or missing.")
    if report.get("metrics", {}).get("total_return") is None:
        likely.append("Metrics are missing; recompute or inspect report artifacts.")
    return {"ok": True, "run_id": report.get("run_id"), "checks": checks, "likely_causes": likely}


def cmd_metrics_list(args) -> Dict[str, Any]:
    return {"ok": True, "metrics": METRIC_SPECS}


def cmd_metrics_compute(args) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    run_dir, data = _load_report(root, args.run_id)
    summary = data.get("summary") or {}
    init_cash = float(args.init_cash or summary.get("initial_cash") or 1_000_000)
    metrics = compute_metrics(data.get("daily_nav") or [], data.get("trades") or [], data.get("closed_positions") or [], init_cash)
    if args.include:
        metrics = {name: metrics.get(name) for name in args.include}
    return {"ok": True, "run_id": run_dir.name, "metrics": metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="QuantX repo root")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    data = sub.add_parser("data-status")
    data.add_argument("--provider-uri", default=DEFAULT_PROVIDER)
    data.set_defaults(func=cmd_data_status)

    data_update = sub.add_parser("data-update")
    data_update.add_argument("--provider-uri", default=DEFAULT_PROVIDER)
    data_update.add_argument("--raw-dir", default="data/raw/baostock")
    data_update.add_argument("--symbols", nargs="*")
    data_update.add_argument("--symbol-file")
    data_update.add_argument("--limit", type=int)
    data_update.add_argument("--end-date")
    data_update.add_argument("--overlap-days", type=int, default=40)
    data_update.add_argument("--tolerance", type=float, default=1e-4)
    data_update.add_argument("--full-refresh-start", default="2010-01-01")
    data_update.add_argument("--pause-seconds", type=float, default=0.5)
    data_update.add_argument("--socket-timeout", type=float, default=30.0)
    data_update.add_argument("--max-requests", type=int, default=45000)
    data_update.add_argument("--progress-every", type=int, default=1)
    data_update.add_argument("--dry-run", action="store_true")
    data_update.set_defaults(func=cmd_data_update)

    meta = sub.add_parser("meta")
    meta.add_argument("--symbols", nargs="+", required=True)
    meta.add_argument("--meta-uri", default="data/meta/quantx_meta.sqlite")
    meta.set_defaults(func=cmd_meta)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--symbol-limit", type=int)
    validate.set_defaults(func=cmd_validate_config)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--symbol-limit", type=int)
    run.add_argument("--output-dir", default="runs")
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    sub.add_parser("latest-run").set_defaults(func=cmd_latest_run)

    report = sub.add_parser("report")
    report.add_argument("--run-id", default="latest")
    report.set_defaults(func=cmd_report)

    symbol = sub.add_parser("symbol")
    symbol.add_argument("--run-id", default="latest")
    symbol.add_argument("--symbol", required=True)
    symbol.set_defaults(func=cmd_symbol)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--run-id", default="latest")
    diagnose.add_argument("--provider-uri", default=DEFAULT_PROVIDER)
    diagnose.set_defaults(func=cmd_diagnose)

    sub.add_parser("metrics-list").set_defaults(func=cmd_metrics_list)


    metrics = sub.add_parser("metrics-compute")
    metrics.add_argument("--run-id", default="latest")
    metrics.add_argument("--include", nargs="*")
    metrics.add_argument("--init-cash", type=float)
    metrics.set_defaults(func=cmd_metrics_compute)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(Path(args.root).expanduser().resolve())
    try:
        result = args.func(args)
        code = 0 if result.get("ok", False) else 1
    except Exception as exc:
        result = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)}
        code = 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
