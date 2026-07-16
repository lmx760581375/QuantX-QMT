"""Incremental market data update into QuantX Qlib data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

from quantx.core.data import (
    AdjustmentAwareIncrementalUpdater,
    BaoStockClient,
    BaostockToQlibConverter,
    LocalDataRepository,
    QMTClient,
)
from quantx.core.research.data_version import (
    DataVersionResolver,
    ProviderLock,
    write_data_version_manifest,
)


def _load_symbols(args) -> List[str] | None:
    symbols: List[str] = []
    if args.strategy_config:
        from quantx.tools.run_backtest import load_config, load_symbols

        symbols.extend(load_symbols(load_config(args.strategy_config)))
    if args.symbols:
        symbols.extend(args.symbols)
    if args.symbol_file:
        path = Path(args.symbol_file)
        symbols.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return symbols or None


def _resolve_turnover_start(
    provider_uri: str,
    *,
    explicit_start: str | None,
    full_refresh_start: str,
    end: str,
    overlap_days: int,
) -> str:
    if explicit_start:
        return explicit_start

    fallback = str(full_refresh_start)
    metadata_path = Path(provider_uri) / "metadata" / "turnover_features.json"
    if not metadata_path.exists():
        return fallback
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_end = metadata.get("end")
        if not previous_end:
            return fallback
        start_date = datetime.strptime(str(previous_end)[:10], "%Y-%m-%d").date() - timedelta(days=max(0, overlap_days))
        fallback_date = datetime.strptime(str(fallback)[:10], "%Y-%m-%d").date()
        end_date = datetime.strptime(str(end)[:10], "%Y-%m-%d").date()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback
    start_date = max(start_date, fallback_date)
    start_date = min(start_date, end_date)
    return start_date.strftime("%Y-%m-%d")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="qmt", choices=["qmt", "baostock"], help="Market data provider.")
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed", help="Qlib provider directory to update.")
    parser.add_argument("--raw-dir", help="Raw CSV repository root. Defaults to data/raw/qmt or data/raw/baostock.")
    parser.add_argument("--mode", default="incremental", choices=["incremental"], help="Only safe incremental mode is supported.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbols to update, e.g. SH600000 SZ000001.")
    parser.add_argument("--strategy-config", help="Load the target universe from a standard strategy YAML.")
    parser.add_argument("--symbol-file", help="Optional file with one symbol per line.")
    parser.add_argument("--limit", type=int, help="Limit symbol count for smoke tests.")
    parser.add_argument("--end-date", help="Update until this date, defaults to today.")
    parser.add_argument("--overlap-days", type=int, default=40, help="Local trading rows to refetch and compare.")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Price mismatch tolerance for adjusted history.")
    parser.add_argument("--full-refresh-start", default="2010-01-01", help="Fallback start date for new symbols.")
    parser.add_argument("--pause-seconds", type=float, help="Provider request pause seconds.")
    parser.add_argument("--socket-timeout", type=float, default=30.0, help="BaoStock socket timeout seconds to avoid hanging forever.")
    parser.add_argument("--max-requests", type=int, default=45000, help="Stop before exceeding the daily BaoStock logical request budget.")
    parser.add_argument("--qmt-dividend-type", default="front_ratio", help="QMT xtdata dividend_type for daily bars.")
    parser.add_argument(
        "--update-turnover-features",
        action="store_true",
        help="After daily bars are updated, write QMT Capital based turnover features into the Qlib provider. QMT source enables this by default.",
    )
    parser.add_argument(
        "--skip-turnover-features",
        action="store_true",
        help="Skip the automatic turnover feature refresh after a successful QMT data update.",
    )
    parser.add_argument(
        "--turnover-start",
        help="Explicit start date for turnover feature rebuild. Defaults to metadata end minus turnover overlap.",
    )
    parser.add_argument(
        "--turnover-overlap-days",
        type=int,
        default=90,
        help="Calendar days to recompute for turnover features when --turnover-start is omitted.",
    )
    parser.add_argument("--turnover-batch-size", type=int, default=200, help="QMT Capital batch size for turnover features.")
    parser.add_argument(
        "--turnover-download-financial",
        action="store_true",
        help="Call QMT download_financial_data2 before reading Capital for turnover features.",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress to stderr every N symbols.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and compare but do not write CSV or Qlib bins.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)
    update_turnover_features = bool(args.update_turnover_features or (args.source == "qmt" and not args.skip_turnover_features))

    raw_dir = args.raw_dir or ("data/raw/qmt" if args.source == "qmt" else "data/raw/baostock")
    pause_seconds = args.pause_seconds
    if pause_seconds is None:
        pause_seconds = 0.0 if args.source == "qmt" else 0.5

    repository = LocalDataRepository(raw_dir)
    converter = BaostockToQlibConverter(
        qlib_dir=args.provider_uri,
        csv_dir=str(Path(raw_dir) / "stocks"),
    )
    if args.source == "qmt":
        def client_factory():
            return QMTClient(
                dividend_type=args.qmt_dividend_type,
                pause_seconds=pause_seconds,
                max_retries=3,
            )
    else:
        def client_factory():
            return BaoStockClient(pause_seconds=pause_seconds, socket_timeout=args.socket_timeout)

    updater = AdjustmentAwareIncrementalUpdater(
        repository=repository,
        converter=converter,
        client_factory=client_factory,
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

    with ProviderLock(args.provider_uri, shared=bool(args.dry_run)):
        report = updater.update(
            symbols=_load_symbols(args),
            end=args.end_date,
            limit=args.limit,
            dry_run=args.dry_run,
            progress_callback=_progress if progress_every > 0 else None,
        )
        if report.ok and not args.dry_run:
            turnover_report = None
            if update_turnover_features:
                from quantx.tools.build_turnover_features import build_turnover_features

                turnover_end = args.end_date or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
                turnover_start = _resolve_turnover_start(
                    args.provider_uri,
                    explicit_start=args.turnover_start,
                    full_refresh_start=args.full_refresh_start,
                    end=turnover_end,
                    overlap_days=args.turnover_overlap_days,
                )
                turnover_report = build_turnover_features(
                    provider_uri=Path(args.provider_uri),
                    output_provider_uri=Path(args.provider_uri),
                    metadata_output=Path(args.provider_uri) / "metadata" / "turnover_features.json",
                    start=turnover_start,
                    end=turnover_end,
                    universe="all_a",
                    limit=args.limit,
                    batch_size=args.turnover_batch_size,
                    download=args.turnover_download_financial,
                    env_file=None,
                )
            sync_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("sync-%Y%m%dT%H%M%S%z")
            data_version = DataVersionResolver().resolve(
                args.provider_uri,
                source_sync_id=sync_id,
                adjustment_mode=args.qmt_dividend_type if args.source == "qmt" else "front_ratio",
            )
            write_data_version_manifest(
                args.provider_uri,
                data_version,
                sync_report=report.to_dict(),
            )
    data = report.to_dict()
    if report.ok and not args.dry_run:
        data["data_version"] = data_version.to_dict()
        if update_turnover_features:
            data["turnover_features"] = turnover_report
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(
            f"ok={data['ok']} total={data['total_symbols']} updated={len(data['updated_symbols'])} "
            f"full_refresh={len(data['full_refresh_symbols'])} failed={len(data['failed_symbols'])}"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
