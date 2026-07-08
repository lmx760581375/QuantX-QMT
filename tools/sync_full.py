#!/usr/bin/env python3
"""Full A-share QMT data sync into Qlib format."""

from __future__ import annotations

import logging
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantx.core.data import QMTConfig, QMTDataSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sync_full.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", help="Defaults to today.")
    parser.add_argument("--raw-dir", default="data/raw/qmt")
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-sync", action="store_true", help="Convert existing raw CSV files without downloading.")
    args = parser.parse_args(argv)

    start_time = time.time()
    config = QMTConfig(
        data_root=args.raw_dir,
        qlib_dir=args.provider_uri,
        pause_seconds=0.0,
        max_retries=3,
        auto_update=True,
        workers=args.workers,
    )
    source = QMTDataSource(config)

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = args.start_date
    end_date = args.end_date or today

    if args.skip_sync:
        stocks_dir = Path(args.raw_dir) / "stocks"
        symbols = sorted(path.stem for path in stocks_dir.glob("*.csv"))
        if not symbols:
            logger.error("No raw CSV files found under %s", stocks_dir)
            return
        logger.info("=== Convert only: %s raw CSV symbols, %s ~ %s ===", len(symbols), start_date, end_date)
        source.converter.convert_all(start_date, end_date, symbols)
        elapsed = time.time() - start_time
        logger.info("=== ALL DONE in %.1f minutes ===", elapsed / 60)
        logger.info("Stocks: %s", len(symbols))
        logger.info("Qlib data: %s", config.qlib_dir)
        return

    logger.info("=== Step 1: Fetching QMT stock list ===")
    symbols = source.get_stock_list(today)
    logger.info("A-share symbols from QMT: %s", len(symbols))
    if not symbols:
        logger.error("No A-share symbols found from QMT")
        return

    batch_size = args.batch_size
    total_batches = (len(symbols) + batch_size - 1) // batch_size
    total_synced = 0
    total_failed = 0

    logger.info("=== Step 2: Syncing %s symbols, %s ~ %s ===", len(symbols), start_date, end_date)
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]

        logger.info("Batch %s/%s: %s symbols", batch_idx + 1, total_batches, len(batch_symbols))
        report = source._sync_symbols(batch_symbols, start_date, end_date, replace=True)
        total_synced += report.synced_count
        total_failed += report.failed_count
        logger.info("Batch %s done: %s synced, %s failed", batch_idx + 1, report.synced_count, report.failed_count)

    logger.info("=== Step 3: Converting all synced symbols to Qlib format ===")
    source.converter.convert_all(start_date, end_date, symbols)

    elapsed = time.time() - start_time
    logger.info("=== ALL DONE in %.1f minutes ===", elapsed / 60)
    logger.info("Stocks: %s", total_synced)
    logger.info("Failed: %s", total_failed)
    logger.info("Qlib data: %s", config.qlib_dir)


if __name__ == "__main__":
    main()
