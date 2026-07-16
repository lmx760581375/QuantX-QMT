"""Probe whether QMT exposes intraday signal sources for short-term alpha research."""

from __future__ import annotations

import concurrent.futures as cf
import inspect
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qmt_client import QMTClient  # noqa: E402


OUT_FILE = Path(__file__).resolve().with_name("qmt_intraday_signal_sources.json")
STOCKS = ["600000.SH", "000001.SZ", "600519.SH", "002594.SZ", "601318.SH"]
KEYWORDS = [
    "tick",
    "full",
    "l2",
    "order",
    "transaction",
    "queue",
    "sector",
    "tabular",
    "finance",
    "north",
    "limit",
    "snapshot",
    "quote",
    "subscribe",
]
TABULAR_PERIODS = [
    "limitupperformance",
    "transactioncount1d",
    "transactioncount1m",
    "stocklistchange",
    "stoppricedata",
    "snapshotindex",
    "northfinancechange1d",
    "northfinancechange1m",
    "announcement",
]
THEME_KEYWORDS = ["人工智能", "机器人", "低空", "半导体", "芯片", "新能源", "概念", "行业"]


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return repr(value)[:300]
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, pd.DataFrame):
        out: dict[str, Any] = {"type": "DataFrame", "rows": int(len(value)), "columns": [str(c) for c in value.columns[:80]]}
        if len(value):
            out["sample"] = json.loads(value.head(2).reset_index().to_json(orient="records", force_ascii=False))
        return out
    if isinstance(value, dict):
        out = {"type": "dict", "len": len(value), "keys": [str(k) for k in list(value)[:30]]}
        out["sample"] = {str(k): compact(v, depth + 1) for k, v in list(value.items())[:5]}
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        return {"type": type(value).__name__, "len": len(seq), "sample": [compact(v, depth + 1) for v in seq[:10]]}
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": tuple(value.shape), "repr": repr(value[:2])[:500] if len(value) else ""}
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def call(fn: Callable[[], Any], timeout: float = 6.0) -> dict[str, Any]:
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return {"ok": True, "value": compact(future.result(timeout=timeout))}
        except cf.TimeoutError:
            return {"ok": False, "error_type": "TimeoutError", "error": f">{timeout}s"}
        except Exception as exc:
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc).split("\n")[0][:800],
                "traceback_tail": traceback.format_exc().splitlines()[-6:],
            }


def signature(obj: Any, name: str) -> str | None:
    if not hasattr(obj, name):
        return None
    try:
        return str(inspect.signature(getattr(obj, name)))
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {str(exc)[:120]}>"


def main() -> None:
    report: dict[str, Any] = {"created_at": datetime.now().isoformat(timespec="seconds"), "sample_stocks": STOCKS}
    with QMTClient(pause_seconds=0, max_retries=1) as client:
        xtdata = client.xtdata
        report["matching_methods"] = sorted(name for name in dir(xtdata) if any(k in name.lower() for k in KEYWORDS))
        interesting_methods = [
            "get_full_tick",
            "subscribe_quote",
            "unsubscribe_quote",
            "subscribe_whole_quote",
            "get_fullspeed_orderbook",
            "get_l2_quote",
            "get_l2_order",
            "get_l2_transaction",
            "get_l2_order_queue",
            "subscribe_l2thousand",
            "subscribe_l2thousand_queue",
            "get_transactioncount",
            "download_tabular_data",
            "get_tabular_data",
            "get_stock_list_in_sector",
            "download_sector_data",
        ]
        report["signatures"] = {name: signature(xtdata, name) for name in interesting_methods}

        calls: dict[str, Any] = {}
        calls["get_full_tick"] = call(lambda: xtdata.get_full_tick(STOCKS)) if hasattr(xtdata, "get_full_tick") else {"ok": False, "error": "missing"}
        calls["get_fullspeed_orderbook"] = (
            call(lambda: xtdata.get_fullspeed_orderbook(STOCKS[:1]))
            if hasattr(xtdata, "get_fullspeed_orderbook")
            else {"ok": False, "error": "missing"}
        )
        calls["get_transactioncount"] = (
            call(lambda: xtdata.get_transactioncount(STOCKS[:1]))
            if hasattr(xtdata, "get_transactioncount")
            else {"ok": False, "error": "missing"}
        )

        for name in ["get_l2_quote", "get_l2_order", "get_l2_transaction", "get_l2_order_queue"]:
            if hasattr(xtdata, name):
                fn = getattr(xtdata, name)
                calls[f"{name}(stock)"] = call(lambda fn=fn: fn(STOCKS[0]))
                calls[f"{name}([stock])"] = call(lambda fn=fn: fn(STOCKS[:1]))
            else:
                calls[name] = {"ok": False, "error": "missing"}

        calls["tabular_periods"] = {
            period: call(
                lambda p=period: xtdata.get_tabular_data([], STOCKS[:1], period=p, start_time="20260710", end_time="20260710", count=5),
                timeout=5.0,
            )
            for period in TABULAR_PERIODS
        }

        sector_list = call(lambda: xtdata.get_sector_list())
        calls["get_sector_list"] = sector_list
        raw_sectors = xtdata.get_sector_list() if sector_list.get("ok") else []
        theme_like = [sector for sector in raw_sectors if any(keyword in str(sector) for keyword in THEME_KEYWORDS)]
        calls["theme_like_sector_names"] = {"ok": True, "value": compact(theme_like)}
        sector_checks: dict[str, Any] = {}
        for sector in list(dict.fromkeys(theme_like[:10] + ["沪深A股", "科创板", "创业板"])):
            sector_checks[sector] = call(lambda s=sector: xtdata.get_stock_list_in_sector(s, real_timetag="20250102"), timeout=5.0)
        calls["sector_membership_checks"] = sector_checks
        report["calls"] = calls

    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
