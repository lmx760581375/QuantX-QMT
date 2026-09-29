"""从 BaoStock 全量拉取前复权日线并构建 QuantX Qlib provider。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quantx.core.data import (
    AdjustmentAwareIncrementalUpdater,
    BaoStockClient,
    BaostockToQlibConverter,
    DataSyncService,
    LocalDataRepository,
    SyncReport,
)
from quantx.core.data.manifest import write_data_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE = REPO_ROOT / "artifacts" / "reference" / "data_20260928" / "instruments_all.txt"
DEFAULT_SECURITY_MASTER = REPO_ROOT / "artifacts" / "reference" / "data_20260928" / "security_master.csv"


def _is_model_symbol(symbol: str) -> bool:
    return (
        symbol.startswith("SH6")
        or (symbol.startswith("SZ0") and not symbol.startswith("SZ399"))
        or (symbol.startswith("SZ3") and not symbol.startswith("SZ399"))
    )


def load_symbols(
    path: str | Path,
    *,
    limit: int | None = None,
    a_share_only: bool = True,
) -> list[str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing universe snapshot: {source}")
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, dtype=str)
        column = next((name for name in ("symbol", "ts_code", "code") if name in frame), None)
        if column is None:
            raise ValueError(f"Universe CSV has no symbol column: {source}")
        values = frame[column].dropna().astype(str).tolist()
    else:
        values = [line.split()[0] for line in source.read_text(encoding="utf-8").splitlines() if line.split()]
    normalized = {BaoStockClient._normalize_symbol(value) for value in values}
    symbols = sorted(
        symbol
        for symbol in normalized
        if not a_share_only or _is_model_symbol(symbol)
    )
    return symbols[: int(limit)] if limit is not None else symbols


def live_symbols(end: str, *, limit: int | None, pause_seconds: float, max_retries: int) -> list[str]:
    with BaoStockClient(pause_seconds=pause_seconds, max_retries=max_retries) as client:
        frame = client.query_all_stocks(end)
    if frame.empty or "code" not in frame:
        raise RuntimeError(f"BaoStock returned no stock universe for {end}")
    symbols = sorted({
        BaoStockClient._normalize_symbol(value)
        for value in frame["code"].dropna().astype(str)
        if _is_model_symbol(BaoStockClient._normalize_symbol(value))
    })
    return symbols[: int(limit)] if limit is not None else symbols


def build_historical_st_snapshot(raw_stock_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    raw_stock_dir = Path(raw_stock_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    parts = []
    for path in sorted(raw_stock_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in {"date", "is_st", "isST"},
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if frame.empty or "date" not in frame:
            continue
        column = "is_st" if "is_st" in frame else "isST" if "isST" in frame else None
        if column is None:
            continue
        flag = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(bool)
        selected = frame.loc[flag, ["date"]].copy()
        if selected.empty:
            continue
        selected["ts_code"] = path.stem
        selected["trade_date"] = pd.to_datetime(selected.pop("date"), errors="coerce")
        selected["is_st"] = 1
        parts.append(selected)
    snapshot = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["ts_code", "trade_date", "is_st"])
    )
    snapshot["trade_date"] = pd.to_datetime(snapshot["trade_date"], errors="coerce")
    snapshot = snapshot.dropna(subset=["trade_date"])
    snapshot["trade_date"] = snapshot["trade_date"].dt.strftime("%Y-%m-%d")
    snapshot = snapshot[["ts_code", "trade_date", "is_st"]]
    snapshot = snapshot.drop_duplicates(["ts_code", "trade_date"]).sort_values(
        ["trade_date", "ts_code"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(output, index=False, compression="zstd")
    return {
        "path": str(output),
        "rows": int(len(snapshot)),
        "symbols": int(snapshot["ts_code"].nunique()),
        "start": str(snapshot["trade_date"].min()) if not snapshot.empty else None,
        "end": str(snapshot["trade_date"].max()) if not snapshot.empty else None,
    }


def write_security_master(symbols: list[str], output_path: str | Path, *, snapshot_date: str) -> Path:
    output = Path(output_path).expanduser().resolve()
    frame = pd.DataFrame({"symbol": symbols})
    frame["name"] = ""
    frame["exchange"] = frame["symbol"].str[:2]
    frame["board"] = frame["symbol"].map(
        lambda symbol: (
            "star"
            if symbol.startswith("SH688")
            else "chinext"
            if symbol.startswith(("SZ300", "SZ301"))
            else "mainboard"
        )
    )
    frame["baostock_code"] = frame["symbol"].map(
        lambda symbol: f"{symbol[:2].lower()}.{symbol[2:]}"
    )
    frame["trade_status"] = 1
    frame["source"] = "quantx-reference-universe"
    frame["snapshot_date"] = str(snapshot_date)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def bootstrap_daily_data(
    *,
    data_root: str | Path,
    start: str,
    end: str,
    universe_path: str | Path = DEFAULT_UNIVERSE,
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER,
    live_universe: bool = False,
    workers: int = 1,
    pause_seconds: float = 0.5,
    max_retries: int = 3,
    socket_timeout: float = 30.0,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    raw_root = root / "raw" / "baostock"
    raw_stock_dir = raw_root / "stocks"
    provider_uri = root / "qlib_data_fixed"
    metadata_dir = root / "meta" / "snapshots"
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must not be later than end")
    symbols = (
        live_symbols(end, limit=limit, pause_seconds=pause_seconds, max_retries=max_retries)
        if live_universe and not dry_run
        else load_symbols(universe_path, limit=limit)
    )
    plan = {
        "source": "baostock",
        "adjustflag": "2",
        "start": str(start),
        "end": str(end),
        "symbols": len(symbols),
        "universe_path": str(Path(universe_path).expanduser().resolve()),
        "security_master_path": str(Path(security_master_path).expanduser().resolve()),
        "live_universe": bool(live_universe),
        "raw_root": str(raw_root),
        "provider_uri": str(provider_uri),
        "historical_st_path": str(metadata_dir / "historical_st_daily.parquet"),
        "resume": bool(resume),
    }
    if dry_run:
        return {"ok": True, "dry_run": True, **plan}
    existing = [
        path
        for path in (
            raw_stock_dir,
            provider_uri / "calendars" / "day.txt",
            provider_uri / "instruments" / "all.txt",
        )
        if path.exists()
    ]
    if existing and not force and not resume:
        raise FileExistsError(
            "Refusing to bootstrap into an existing data directory without --force: "
            + ", ".join(str(path) for path in existing)
        )

    repository = LocalDataRepository(str(raw_root))
    fetch_symbols = symbols
    if resume and not force:
        fetch_symbols = []
        for symbol in symbols:
            frame = repository.load_symbol(symbol)
            if frame.empty:
                fetch_symbols.append(symbol)
    sync = DataSyncService(
        BaoStockClient(pause_seconds=pause_seconds, max_retries=max_retries),
        repository,
        workers=max(1, int(workers)),
        pause_seconds=float(pause_seconds),
        max_retries=max(1, int(max_retries)),
        socket_timeout=float(socket_timeout),
    )
    def progress(report) -> None:
        done = report.synced_count + report.failed_count
        if done == 1 or done % 25 == 0 or done == report.total_symbols:
            print(
                json.dumps(
                    {
                        "event": "bootstrap_progress",
                        "done": done,
                        "total": report.total_symbols,
                        "synced": report.synced_count,
                        "failed": report.failed_count,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

    report = (
        sync.sync_full(
            fetch_symbols,
            str(start),
            str(end),
            progress_callback=progress,
        )
        if fetch_symbols
        else SyncReport(
            total_symbols=0,
            synced_count=0,
            start_time=str(start),
            end_time=str(end),
        )
    )
    missing = [symbol for symbol in symbols if repository.load_symbol(symbol).empty]
    if missing:
        raise RuntimeError(
            f"BaoStock bootstrap incomplete: {len(missing)} symbols have no local CSV; "
            f"examples={missing[:10]}"
        )

    converter = BaostockToQlibConverter(
        qlib_dir=str(provider_uri),
        csv_dir=str(raw_stock_dir),
    )
    converter.convert_all(str(start), str(end), symbols)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    security_source = Path(security_master_path).expanduser().resolve()
    if security_source.is_file() and not live_universe:
        shutil.copy2(security_source, metadata_dir / "security_master.csv")
    else:
        write_security_master(
            [symbol for symbol in symbols if _is_model_symbol(symbol)],
            metadata_dir / "security_master.csv",
            snapshot_date=str(end),
        )
    st_summary = build_historical_st_snapshot(
        raw_stock_dir,
        metadata_dir / "historical_st_daily.parquet",
    )
    manifest = write_data_manifest(root)
    return {
        "ok": True,
        **plan,
        "sync": asdict(report),
        "historical_st": st_summary,
        "manifest": manifest,
    }


def update_daily_data(
    *,
    data_root: str | Path,
    end: str,
    universe_path: str | Path = DEFAULT_UNIVERSE,
    live_universe: bool = False,
    pause_seconds: float = 0.5,
    max_retries: int = 3,
    socket_timeout: float = 30.0,
    overlap_days: int = 40,
    tolerance: float = 1.0e-4,
    max_requests: int = 45_000,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    raw_root = root / "raw" / "baostock"
    raw_stock_dir = raw_root / "stocks"
    provider_uri = root / "qlib_data_fixed"
    if not provider_uri.is_dir() or not raw_stock_dir.is_dir():
        raise FileNotFoundError("Incremental update requires an existing bootstrap data directory")
    symbols = (
        live_symbols(end, limit=limit, pause_seconds=pause_seconds, max_retries=max_retries)
        if live_universe and not dry_run
        else load_symbols(universe_path, limit=limit)
    )
    plan = {
        "source": "baostock",
        "mode": "adjustment_aware_incremental",
        "end": str(end),
        "symbols": len(symbols),
        "universe_path": str(Path(universe_path).expanduser().resolve()),
        "live_universe": bool(live_universe),
        "raw_root": str(raw_root),
        "provider_uri": str(provider_uri),
        "overlap_days": int(overlap_days),
        "tolerance": float(tolerance),
        "max_requests": int(max_requests),
    }
    if dry_run:
        return {"ok": True, "dry_run": True, **plan}

    repository = LocalDataRepository(str(raw_root))
    converter = BaostockToQlibConverter(
        qlib_dir=str(provider_uri),
        csv_dir=str(raw_stock_dir),
    )
    updater = AdjustmentAwareIncrementalUpdater(
        repository=repository,
        converter=converter,
        client_factory=lambda: BaoStockClient(
            pause_seconds=pause_seconds,
            max_retries=max_retries,
            socket_timeout=socket_timeout,
        ),
        overlap_days=overlap_days,
        tolerance=tolerance,
        full_refresh_start="2010-01-01",
        max_requests=max_requests,
    )
    report = updater.update(symbols=symbols, end=end, limit=limit, dry_run=False)
    if not report.ok:
        raise RuntimeError(
            f"Incremental update failed: failed={len(report.failed_symbols)}, "
            f"budget_exhausted={report.budget_exhausted}"
        )
    metadata_dir = root / "meta" / "snapshots"
    st_summary = build_historical_st_snapshot(
        raw_stock_dir,
        metadata_dir / "historical_st_daily.parquet",
    )
    manifest = write_data_manifest(root)
    return {
        "ok": True,
        **plan,
        "update": report.to_dict(),
        "historical_st": st_summary,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--universe-path", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--security-master-path", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--live-universe", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = bootstrap_daily_data(
        data_root=args.data_root,
        start=args.start,
        end=args.end,
        universe_path=args.universe_path,
        security_master_path=args.security_master_path,
        live_universe=args.live_universe,
        workers=args.workers,
        pause_seconds=args.pause_seconds,
        max_retries=args.max_retries,
        socket_timeout=args.socket_timeout,
        limit=args.limit,
        force=args.force,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
