#!/usr/bin/env python3
"""Query QuantX metadata cache for stock names and industries."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def normalize_symbol(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if "." in text:
        left, right = text.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{left.upper()}{right}"
        if right.upper() in {"SH", "SZ", "BJ"}:
            return f"{right.upper()}{left}"
    upper = text.upper()
    if upper.startswith(("SH", "SZ", "BJ")):
        return upper
    if len(upper) == 6 and upper.isdigit():
        if upper.startswith("6"):
            return f"SH{upper}"
        if upper.startswith(("0", "3")):
            return f"SZ{upper}"
        if upper.startswith(("4", "8", "9")):
            return f"BJ{upper}"
    return upper


def query_symbol(conn: sqlite3.Connection, symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    row = conn.execute(
        """
        SELECT sm.symbol, sm.name, sm.exchange, sm.board,
               im.industry_source, im.level, im.industry_code, im.industry_name
        FROM (SELECT ? AS symbol) s
        LEFT JOIN security_master sm ON s.symbol = sm.symbol
        LEFT JOIN industry_membership im
          ON s.symbol = im.symbol AND im.industry_source = 'eastmoney' AND im.level = 1
        """,
        (symbol,),
    ).fetchone()
    return {
        "symbol": symbol,
        "found": bool(row and (row[0] or row[7])),
        "name": row[1] if row else None,
        "exchange": row[2] if row else None,
        "board": row[3] if row else None,
        "industry_source": row[4] if row else None,
        "industry_level": row[5] if row else None,
        "industry_code": row[6] if row else None,
        "industry_name": row[7] if row else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QuantX repo root")
    parser.add_argument("--meta-uri", default="data/meta/quantx_meta.sqlite")
    parser.add_argument("--symbols", nargs="+", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    db = Path(args.meta_uri)
    if not db.is_absolute():
        db = root / db
    if not db.exists():
        print(json.dumps({"ok": False, "message": f"metadata db not found: {db}"}, ensure_ascii=False, indent=2))
        return 1
    with sqlite3.connect(db) as conn:
        rows = [query_symbol(conn, symbol) for symbol in args.symbols]
    print(json.dumps({"ok": True, "meta_uri": str(db), "symbols": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
