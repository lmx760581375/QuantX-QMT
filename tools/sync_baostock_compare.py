#!/usr/bin/env python3
"""Sync BaoStock data into a separate comparison provider."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantx.core.data import BaoStockClient, BaostockToQlibConverter, LocalDataRepository


MAINBOARD_PREFIXES = ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")


def _mainboard_symbols(universe_path: Path, limit: int | None = None) -> list[str]:
    frame = pd.read_csv(universe_path, dtype={"code": str})
    symbols = sorted(str(code) for code in frame["code"] if str(code).startswith(MAINBOARD_PREFIXES))
    return symbols[:limit] if limit else symbols


def _coverage_ok(path: Path, start: str, end: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["date"])
    except Exception:
        return False
    if frame.empty:
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return dates.min() <= pd.Timestamp(start) and dates.max() >= pd.Timestamp(end)


def _write_status(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="2026-07-08")
    parser.add_argument("--raw-dir", default="data/raw/baostock_compare")
    parser.add_argument("--provider-uri", default="data/qlib_data_baostock_compare")
    parser.add_argument("--universe", default="data/raw/qmt/universe/all_stocks.csv")
    parser.add_argument("--status", default="runs/baostock_compare_sync_status.json")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = parser.parse_args(argv)

    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("baostock_compare_sync")

    raw_dir = Path(args.raw_dir)
    provider_uri = Path(args.provider_uri)
    status_path = Path(args.status)
    repo = LocalDataRepository(str(raw_dir))
    symbols = _mainboard_symbols(Path(args.universe), args.limit)
    repo.save_universe(pd.DataFrame({"code": symbols}))

    synced: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    started = pd.Timestamp.now()
    status = {
        "ok": False,
        "phase": "sync",
        "started_at": started.isoformat(),
        "finished_at": None,
        "raw_dir": str(raw_dir),
        "provider_uri": str(provider_uri),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "total": len(symbols),
        "done": 0,
        "synced": 0,
        "skipped": 0,
        "failed": 0,
        "latest": None,
        "failed_symbols": [],
    }
    _write_status(status_path, status)

    logger.info(
        "BaoStock compare sync start symbols=%s start=%s end=%s raw=%s provider=%s",
        len(symbols),
        args.start_date,
        args.end_date,
        raw_dir,
        provider_uri,
    )

    with BaoStockClient(pause_seconds=args.pause_seconds, max_retries=3, socket_timeout=args.socket_timeout) as client:
        for idx, symbol in enumerate(symbols, start=1):
            csv_path = raw_dir / "stocks" / f"{symbol}.csv"
            try:
                if args.skip_existing and _coverage_ok(csv_path, args.start_date, args.end_date):
                    skipped.append(symbol)
                else:
                    frame = client.query_history_k_data_with_retry(
                        BaoStockClient.to_baostock_format(symbol),
                        args.start_date,
                        args.end_date,
                    )
                    if frame.empty:
                        failed.append(symbol)
                    else:
                        repo.replace_symbol(symbol, frame)
                        synced.append(symbol)
            except Exception as exc:  # pragma: no cover - long-running provider job.
                failed.append(symbol)
                logger.warning("BaoStock sync failed symbol=%s error=%s", symbol, exc)

            status.update(
                {
                    "done": idx,
                    "synced": len(synced),
                    "skipped": len(skipped),
                    "failed": len(failed),
                    "latest": symbol,
                    "failed_symbols": failed[-50:],
                }
            )
            if idx == 1 or idx % max(1, args.progress_every) == 0 or idx == len(symbols):
                elapsed_min = (pd.Timestamp.now() - started).total_seconds() / 60.0
                logger.info(
                    "BaoStock sync progress %s/%s synced=%s skipped=%s failed=%s latest=%s elapsed=%.1fmin",
                    idx,
                    len(symbols),
                    len(synced),
                    len(skipped),
                    len(failed),
                    symbol,
                    elapsed_min,
                )
                _write_status(status_path, status)

    convert_symbols = sorted(set(synced) | set(skipped))
    status.update({"phase": "convert", "convert_symbols": len(convert_symbols)})
    _write_status(status_path, status)

    if convert_symbols:
        logger.info("BaoStock conversion start symbols=%s provider=%s", len(convert_symbols), provider_uri)
        converter = BaostockToQlibConverter(
            qlib_dir=str(provider_uri),
            csv_dir=str(raw_dir / "stocks"),
        )
        converter.convert_all(args.start_date, args.end_date, convert_symbols)
        logger.info("BaoStock conversion done provider=%s", provider_uri)

    status.update(
        {
            "ok": not failed,
            "phase": "done",
            "finished_at": pd.Timestamp.now().isoformat(),
            "elapsed_minutes": (pd.Timestamp.now() - started).total_seconds() / 60.0,
        }
    )
    _write_status(status_path, status)
    logger.info(
        "BaoStock compare sync done ok=%s synced=%s skipped=%s failed=%s elapsed=%.1fmin",
        not failed,
        len(synced),
        len(skipped),
        len(failed),
        status["elapsed_minutes"],
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
