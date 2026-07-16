from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qmt_client import QMTClient, to_qmt_symbol


ROOT = REPO_ROOT / ".tmp/quantx-research/qmt-minute-coverage-probe-v1"
OUT = ROOT / "qmt_minute_coverage_probe.json"
CACHE = ROOT / "cache"
SYMBOLS = ("SH600000", "SZ000001", "SH600519", "SZ300750")
DATES = ("2021-01-04", "2022-01-04", "2023-01-04", "2024-01-04", "2025-01-02", "2026-01-05", "2026-07-01")
PERIODS = ("5m", "1m")


def qmt_date(day: str) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def cache_path(symbol: str, day: str, period: str) -> Path:
    return CACHE / period / f"{symbol}_{day}.csv"


def fetch(client: QMTClient, symbol: str, day: str, period: str) -> tuple[pd.DataFrame, str | None]:
    path = cache_path(symbol, day, period)
    if path.exists():
        return pd.read_csv(path), None
    path.parent.mkdir(parents=True, exist_ok=True)
    qsymbol = to_qmt_symbol(symbol)
    try:
        client.download_history_data(symbol, day, day, period=period)
        data = client.xtdata.get_market_data_ex(
            field_list=[],
            stock_list=[qsymbol],
            period=period,
            start_time=qmt_date(day),
            end_time=qmt_date(day),
            count=-1,
            dividend_type="none",
            fill_data=False,
        )
    except Exception as exc:  # provider failures are environment-specific.
        return pd.DataFrame(), repr(exc)
    raw = data.get(qsymbol) if isinstance(data, dict) else None
    frame = pd.DataFrame(raw).copy() if raw is not None and not raw.empty else pd.DataFrame()
    frame.to_csv(path, index=False)
    return frame, None


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "columns": []}
    out: dict[str, Any] = {"rows": int(len(frame)), "columns": list(map(str, frame.columns))}
    for col in ("open", "high", "low", "close", "volume", "amount", "preClose", "suspendFlag"):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            out[f"{col}_non_null"] = int(values.notna().sum())
            if col in ("volume", "amount"):
                out[f"{col}_positive"] = int((values > 0).sum())
    return out


def main() -> int:
    rows = []
    with QMTClient(dividend_type="none", fill_data=False, pause_seconds=0.0, max_retries=1) as client:
        for period in PERIODS:
            for day in DATES:
                for symbol in SYMBOLS:
                    frame, error = fetch(client, symbol, day, period)
                    item = {"period": period, "date": day, "symbol": symbol, "error": error, **summarize_frame(frame)}
                    rows.append(item)
                    print(json.dumps(item, ensure_ascii=False), flush=True)
    result = {
        "status": "complete",
        "experiment": "qmt_minute_coverage_probe_v1",
        "symbols": SYMBOLS,
        "dates": DATES,
        "periods": PERIODS,
        "rows": rows,
        "coverage_by_period_year": summarize_coverage(rows),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("sha256:" + hashlib.sha256(OUT.read_bytes()).hexdigest(), flush=True)
    return 0


def summarize_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = f"{row['period']}::{row['date'][:4]}"
        bucket = out.setdefault(key, {"checks": 0, "non_empty": 0, "avg_rows": 0.0, "errors": 0})
        bucket["checks"] += 1
        bucket["non_empty"] += int(row.get("rows", 0) > 0)
        bucket["avg_rows"] += float(row.get("rows", 0))
        bucket["errors"] += int(row.get("error") is not None)
    for bucket in out.values():
        if bucket["checks"]:
            bucket["avg_rows"] /= bucket["checks"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
