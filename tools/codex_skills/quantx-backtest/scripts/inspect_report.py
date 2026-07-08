#!/usr/bin/env python3
"""Inspect QuantX report artifacts and return a compact JSON summary."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def latest_run_id(root: Path) -> str | None:
    runs = root / "runs"
    if not runs.exists():
        return None
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def normalize_symbol(value: str) -> str:
    text = str(value).strip()
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


def load_meta(root: Path, symbols: list[str]) -> dict[str, dict]:
    db = root / "data" / "meta" / "quantx_meta.sqlite"
    if not db.exists() or not symbols:
        return {}
    out = {}
    with sqlite3.connect(db) as conn:
        for symbol in symbols:
            norm = normalize_symbol(symbol)
            row = conn.execute(
                """
                SELECT sm.name, im.industry_name
                FROM (SELECT ? AS symbol) s
                LEFT JOIN security_master sm ON s.symbol = sm.symbol
                LEFT JOIN industry_membership im
                  ON s.symbol = im.symbol AND im.industry_source = 'eastmoney' AND im.level = 1
                """,
                (norm,),
            ).fetchone()
            out[norm] = {"name": row[0] if row else None, "industry_name": row[1] if row else None}
    return out


def summarize(root: Path, run_id: str, symbol: str | None = None) -> dict:
    if run_id == "latest":
        run_id = latest_run_id(root) or "latest"
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        return {"ok": False, "message": f"run not found: {run_id}", "run_id": run_id}

    summary = read_json(run_dir / "summary.json", {})
    metrics = read_json(run_dir / "metrics.json", {})
    trades = read_json(run_dir / "trades.json", [])
    daily_nav = read_json(run_dir / "daily_nav.json", [])
    closed = read_json(run_dir / "closed_positions.json", [])

    valid_trades = [t for t in trades if not t.get("reject_reason")]
    rejects = [t for t in trades if t.get("reject_reason")]
    by_symbol = Counter(t.get("symbol") for t in valid_trades if t.get("symbol"))
    top_symbols = [sym for sym, _ in by_symbol.most_common(10)]
    meta = load_meta(root, top_symbols + ([symbol] if symbol else []))
    top_traded = [
        {
            "symbol": sym,
            "name": meta.get(normalize_symbol(sym), {}).get("name"),
            "industry_name": meta.get(normalize_symbol(sym), {}).get("industry_name"),
            "trade_count": count,
        }
        for sym, count in by_symbol.most_common(10)
    ]
    reject_summary = Counter(t.get("reject_reason") for t in rejects)

    result = {
        "ok": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "metrics": metrics,
        "daily_nav_count": len(daily_nav),
        "trade_count": len(valid_trades),
        "reject_count": len(rejects),
        "reject_summary": dict(reject_summary),
        "top_traded_symbols": top_traded,
        "artifacts": {p.stem: str(p) for p in sorted(run_dir.glob("*.json"))},
    }

    if symbol:
        norm = normalize_symbol(symbol)
        symbol_trades = [t for t in valid_trades if normalize_symbol(t.get("symbol", "")) == norm]
        symbol_closed = [p for p in closed if normalize_symbol(p.get("symbol", "")) == norm]
        returns = [p.get("return") for p in symbol_closed if isinstance(p.get("return"), (int, float))]
        result["symbol"] = {
            "symbol": norm,
            "name": meta.get(norm, {}).get("name"),
            "industry_name": meta.get(norm, {}).get("industry_name"),
            "trade_count": len(symbol_trades),
            "round_trips": len(symbol_closed),
            "avg_round_trip_return": sum(returns) / len(returns) if returns else None,
            "trades": symbol_trades[:20],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QuantX repo root")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--symbol")
    args = parser.parse_args()
    result = summarize(Path(args.root).expanduser().resolve(), args.run_id, args.symbol)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
