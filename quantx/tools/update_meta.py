"""Update QuantX security metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from quantx.core.data.meta import MetaStore, MetaUpdateService


def run_update(args) -> Dict[str, Any]:
    store = MetaStore(args.meta_uri)
    service = MetaUpdateService(store)
    results: List[Dict[str, Any]] = []

    if args.probe:
        results.append({"probe_akshare": service.probe_akshare()})
    if args.baostock_security_master:
        results.append(service.update_security_master_baostock(date=args.date).to_dict())
    if args.qmt_security_master:
        results.append(service.update_security_master_qmt(sectors=args.qmt_sector).to_dict())
    if args.security_master_csv:
        results.append(
            service.import_security_master_csv(
                args.security_master_csv,
                source=args.source,
                snapshot_date=args.date,
            ).to_dict()
        )
    if args.industry_csv:
        results.append(
            service.import_industry_csv(
                args.industry_csv,
                source=args.source,
                snapshot_date=args.date,
            ).to_dict()
        )
    if args.sector_csv:
        results.append(
            service.import_sector_csv(
                args.sector_csv,
                source=args.source,
                snapshot_date=args.date,
            ).to_dict()
        )

    if not results:
        raise ValueError("No update action specified")

    return {
        "meta_uri": str(Path(args.meta_uri)),
        "results": results,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-uri", default="data/meta/quantx_meta.sqlite")
    parser.add_argument("--source", default="csv")
    parser.add_argument("--date", help="Optional source date, e.g. 2026-06-25.")
    parser.add_argument("--probe", action="store_true", help="Probe online metadata sources without writing data.")
    parser.add_argument("--baostock-security-master", action="store_true", help="Fetch security master from BaoStock.")
    parser.add_argument("--qmt-security-master", action="store_true", help="Fetch current security master names from QMT/xqshare.")
    parser.add_argument("--qmt-sector", action="append", help="QMT sector to include; repeatable. Defaults to 沪深A股.")
    parser.add_argument("--security-master-csv", help="Import security master CSV.")
    parser.add_argument("--industry-csv", help="Import industry membership CSV.")
    parser.add_argument("--sector-csv", help="Import sector membership CSV.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_update(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for row in result["results"]:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
