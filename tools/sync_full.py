#!/usr/bin/env python3
"""全量 A 股数据同步脚本（2010-2025）

拉取全量 A 股日线数据并转换为 Qlib 二进制格式。
预计耗时：2-4 小时（取决于网络和 CPU）。
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from quantx.core.data import (
    BaoStockClient,
    BaoStockConfig,
    BaoStockDataSource,
    LocalDataRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sync_full.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()

    config = BaoStockConfig(
        data_root="data/raw/baostock",
        qlib_dir="data/qlib_data",
        workers=8,
        pause_seconds=0.3,
        max_retries=3,
        auto_update=True,
    )

    source = BaoStockDataSource(config)

    # Step 1: 获取股票列表
    logger.info("=== Step 1: Fetching stock list ===")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    with BaoStockClient(pause_seconds=config.pause_seconds) as client:
        stocks_df = client.query_all_stocks(today)
        logger.info(f"Total stocks from BaoStock: {len(stocks_df)}")

        a_stocks = []
        for _, row in stocks_df.iterrows():
            code = row["code"]
            if code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3"):
                a_stocks.append(BaoStockClient._normalize_symbol(code))

        logger.info(f"A-share stocks: {len(a_stocks)}")
        source.repository.save_universe(stocks_df)

    if not a_stocks:
        logger.error("No A-share stocks found!")
        return

    # Step 2: 全量同步（分批处理，避免内存爆炸）
    start_date = "2010-01-01"
    end_date = today
    logger.info(f"=== Step 2: Syncing {len(a_stocks)} stocks, {start_date} ~ {end_date} ===")

    # 分批：每批 500 只
    batch_size = 500
    total_batches = (len(a_stocks) + batch_size - 1) // batch_size
    total_synced = 0
    total_failed = 0

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(a_stocks))
        batch_symbols = a_stocks[batch_start:batch_end]

        logger.info(f"  Batch {batch_idx + 1}/{total_batches}: {len(batch_symbols)} stocks")

        report = source.sync_service.sync_full(batch_symbols, start_date, end_date)
        total_synced += report.synced_count
        total_failed += report.failed_count

        logger.info(f"  Batch {batch_idx + 1} done: {report.synced_count} synced, {report.failed_count} failed")

    logger.info(f"Sync complete: {total_synced} synced, {total_failed} failed")

    # Step 3: 转换为 Qlib 格式
    logger.info("=== Step 3: Converting to Qlib format ===")
    source.converter.convert_all(start_date, end_date, a_stocks)

    elapsed = time.time() - start_time
    logger.info(f"=== ALL DONE in {elapsed/60:.1f} minutes ===")
    logger.info(f"  Stocks: {total_synced}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Qlib data: {config.qlib_dir}")


if __name__ == "__main__":
    main()