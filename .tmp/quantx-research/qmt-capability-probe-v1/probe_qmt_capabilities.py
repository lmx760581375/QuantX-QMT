"""Probe current QMT/xtdata capabilities without touching production data.

The script is intentionally defensive: every provider call is isolated so one
missing permission or unsupported period does not hide the rest of the picture.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
import argparse
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qmt_client import QMTClient  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "qmt_capability_probe.json"

SAMPLE_STOCKS = ["600000.SH", "000001.SZ", "600519.SH", "002594.SZ", "601318.SH"]
DATE_WINDOWS = [
    ("20210101", "20210131"),
    ("20230101", "20230131"),
    ("20250101", "20250131"),
    ("20260101", "20260710"),
]
TABULAR_PERIODS = [
    "limitupperformance",
    "transactioncount1d",
    "transactioncount1m",
    "stocklistchange",
    "stoppricedata",
    "snapshotindex",
    "northfinancechange1d",
    "announcement",
]
INTERESTING_PERIODS = [
    "tick",
    "1m",
    "5m",
    "1d",
    "l2quote",
    "l2order",
    "l2transaction",
    "l2quoteaux",
    "l2orderqueue",
    "l2thousand",
    "transactioncount1m",
    "transactioncount1d",
    "limitupperformance",
    "stocklistchange",
    "stoppricedata",
    "snapshotindex",
    "northfinancechange1d",
    "announcement",
]
SECTOR_KEYWORDS = ["人工智能", "机器人", "低空经济", "半导体", "新能源", "芯片", "沪深A股"]
SECTOR_DATES = ["20210104", "20230103", "20250102", "20260710"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_json(value: Any, max_rows: int = 2) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [safe_json(item, max_rows=max_rows) for item in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): safe_json(v, max_rows=max_rows) for k, v in list(value.items())[:80]}
    if isinstance(value, pd.DataFrame):
        return summarize_frame(value, max_rows=max_rows)
    if hasattr(value, "tolist"):
        try:
            return safe_json(value.tolist(), max_rows=max_rows)
        except Exception:
            pass
    return repr(value)


def summarize_frame(frame: pd.DataFrame, max_rows: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "DataFrame",
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns[:80]],
    }
    if frame.index is not None:
        out["index_name"] = str(frame.index.name)
    if len(frame):
        out["sample"] = json.loads(frame.head(max_rows).reset_index().to_json(orient="records", force_ascii=False))
    return out


def summarize_array_like(value: Any) -> dict[str, Any]:
    try:
        length = len(value)
    except Exception:
        length = None
    out = {"type": type(value).__name__, "length": length}
    try:
        if length:
            out["sample"] = safe_json(value[:2])
    except Exception:
        out["repr"] = repr(value)[:800]
    return out


def call(name: str, func: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = func()
        return {
            "ok": True,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "value": safe_json(value),
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


def is_probable_trading_time() -> bool:
    t = datetime.now().time()
    return dtime(9, 15) <= t <= dtime(15, 5)


def compact_market_data_ex(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        out: dict[str, Any] = {"type": "dict", "keys": list(result.keys())[:30]}
        sample = {}
        for key, value in list(result.items())[:5]:
            if isinstance(value, pd.DataFrame):
                sample[str(key)] = summarize_frame(value)
            else:
                sample[str(key)] = summarize_array_like(value)
        out["sample"] = sample
        return out
    return {"value": safe_json(result)}


def probe_period_support(xtdata: Any, report: dict[str, Any]) -> None:
    periods_raw = call("get_period_list", lambda: xtdata.get_period_list())
    report["basic_interfaces"]["get_period_list"] = periods_raw
    periods = set(periods_raw.get("value") or []) if periods_raw.get("ok") else set()
    report["period_presence"] = {period: period in periods for period in INTERESTING_PERIODS}


def probe_basic_interfaces(xtdata: Any, report: dict[str, Any]) -> None:
    basic = report["basic_interfaces"]
    for name, func in [
        ("get_data_dir", lambda: xtdata.get_data_dir()),
        ("get_authorized_market_list", lambda: xtdata.get_authorized_market_list()),
        ("get_markets", lambda: xtdata.get_markets()),
        ("get_quote_server_status", lambda: xtdata.get_quote_server_status()),
        ("get_quote_server_config", lambda: xtdata.get_quote_server_config()),
        ("get_sector_list", lambda: xtdata.get_sector_list()),
    ]:
        basic[name] = call(name, func)


def choose_sector_names(sector_list_result: dict[str, Any]) -> list[str]:
    values = sector_list_result.get("value") if sector_list_result.get("ok") else []
    if not isinstance(values, list):
        return ["沪深A股"]
    chosen: list[str] = []
    for keyword in SECTOR_KEYWORDS:
        for sector in values:
            text = str(sector)
            if keyword in text and text not in chosen:
                chosen.append(text)
                break
    if "沪深A股" not in chosen:
        chosen.append("沪深A股")
    return chosen[:8]


def probe_sector_history(xtdata: Any, report: dict[str, Any]) -> None:
    section: dict[str, Any] = {"download_sector_data": None, "sectors": {}}
    if report.get("options", {}).get("allow_downloads"):
        section["download_sector_data"] = call("download_sector_data", lambda: xtdata.download_sector_data())
    else:
        section["download_sector_data"] = {"ok": None, "skipped": True, "reason": "quick_mode_no_downloads"}
    sectors = choose_sector_names(report["basic_interfaces"].get("get_sector_list", {}))
    for sector in sectors:
        item: dict[str, Any] = {"latest": None, "history": {}}
        latest_result = call(f"get_stock_list_in_sector:{sector}:latest", lambda s=sector: xtdata.get_stock_list_in_sector(s))
        item["latest"] = latest_result
        latest_set = set(latest_result.get("value") or []) if latest_result.get("ok") else set()
        for date in SECTOR_DATES:
            result = call(
                f"get_stock_list_in_sector:{sector}:{date}",
                lambda s=sector, d=date: xtdata.get_stock_list_in_sector(s, real_timetag=d),
            )
            hist_set = set(result.get("value") or []) if result.get("ok") else set()
            overlap = len(latest_set & hist_set) / max(1, len(latest_set | hist_set)) if latest_set or hist_set else None
            item["history"][date] = {
                **result,
                "count": len(hist_set) if result.get("ok") else 0,
                "jaccard_vs_latest": round(overlap, 4) if overlap is not None else None,
                "sample": sorted(hist_set)[:10],
            }
        section["sectors"][sector] = item
    report["sector_history"] = section


def probe_tabular_period(xtdata: Any, period: str, stock: str, start: str, end: str, allow_downloads: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"stock": stock, "start": start, "end": end}
    if allow_downloads:
        result["download"] = call(
            f"download_tabular_data:{period}:{stock}:{start}",
            lambda: xtdata.download_tabular_data([stock], period, start_time=start, end_time=end),
        )
    else:
        result["download"] = {"ok": None, "skipped": True, "reason": "quick_mode_no_downloads"}
    query = call(
        f"get_tabular_data:{period}:{stock}:{start}",
        lambda: xtdata.get_tabular_data([], [stock], period=period, start_time=start, end_time=end, count=10),
    )
    result["query"] = query
    value = query.get("value")
    if isinstance(value, dict) and value.get("type") == "DataFrame":
        result["rows"] = value.get("rows")
        result["columns"] = value.get("columns")
    return result


def probe_tabular_history(xtdata: Any, report: dict[str, Any]) -> None:
    section: dict[str, Any] = {}
    allow_downloads = bool(report.get("options", {}).get("allow_downloads"))
    for period in TABULAR_PERIODS:
        period_results = []
        for start, end in DATE_WINDOWS:
            found = None
            attempts = []
            stocks = SAMPLE_STOCKS[:3] if allow_downloads else SAMPLE_STOCKS[:1]
            for stock in stocks:
                item = probe_tabular_period(xtdata, period, stock, start, end, allow_downloads)
                attempts.append(item)
                if (item.get("rows") or 0) > 0:
                    found = item
                    break
            period_results.append({"window": [start, end], "found": found, "attempts": attempts})
        section[period] = period_results
    report["tabular_history"] = section


def probe_recent_kline_history(xtdata: Any, report: dict[str, Any]) -> None:
    section: dict[str, Any] = {}
    allow_downloads = bool(report.get("options", {}).get("allow_downloads"))
    for period in ["1m", "5m", "tick"]:
        rows = []
        for start, end in DATE_WINDOWS:
            item: dict[str, Any] = {"window": [start, end]}
            if allow_downloads:
                item["download"] = call(
                    f"download_history_data:{period}:{SAMPLE_STOCKS[0]}:{start}",
                    lambda p=period, s=SAMPLE_STOCKS[0], st=start, en=end: xtdata.download_history_data(s, p, start_time=st, end_time=en),
                )
            else:
                item["download"] = {"ok": None, "skipped": True, "reason": "quick_mode_no_downloads"}
            query = call(
                f"get_market_data_ex:{period}:{SAMPLE_STOCKS[0]}:{start}",
                lambda p=period, s=SAMPLE_STOCKS[0], st=start, en=end: compact_market_data_ex(
                    xtdata.get_market_data_ex([], [s], period=p, start_time=st, end_time=en, count=10, fill_data=False)
                ),
            )
            item["query"] = query
            rows.append(item)
        section[period] = rows
    report["recent_kline_history"] = section


def probe_realtime_interfaces(xtdata: Any, report: dict[str, Any]) -> None:
    section: dict[str, Any] = {"trading_time_probe": is_probable_trading_time(), "calls": {}}
    if not section["trading_time_probe"]:
        section["note"] = "Skipped realtime probes because local time is outside the regular A-share session window."
        report["realtime"] = section
        return
    section["calls"]["get_full_tick"] = call("get_full_tick", lambda: xtdata.get_full_tick([SAMPLE_STOCKS[0]]))
    section["calls"]["get_fullspeed_orderbook"] = call(
        "get_fullspeed_orderbook", lambda: xtdata.get_fullspeed_orderbook([SAMPLE_STOCKS[0]])
    )
    section["calls"]["get_transactioncount"] = call(
        "get_transactioncount", lambda: xtdata.get_transactioncount([SAMPLE_STOCKS[0]])
    )
    report["realtime"] = section


def classify_capabilities(report: dict[str, Any]) -> None:
    rows: dict[str, Any] = {}
    presence = report.get("period_presence", {})

    def any_positive_rows(period: str) -> bool:
        for window in report.get("tabular_history", {}).get(period, []):
            found = window.get("found")
            if found and (found.get("rows") or 0) > 0:
                return True
        return False

    for period in ["limitupperformance", "transactioncount1d", "stocklistchange", "stoppricedata", "snapshotindex", "northfinancechange1d"]:
        rows[period] = {
            "present_in_period_list": bool(presence.get(period)),
            "history_rows_found": any_positive_rows(period),
            "classification": "five_year_backtest_ready" if any_positive_rows(period) else "unavailable_or_permission_missing",
        }

    rows["minute_kline"] = {
        "present_in_period_list": bool(presence.get("1m") or presence.get("5m")),
        "classification": "recent_only_or_realtime",
        "note": "Use recent_kline_history rows to decide whether 2021-2025 is covered in this QMT environment.",
    }

    sector_scores = []
    for sector, item in report.get("sector_history", {}).get("sectors", {}).items():
        overlaps = [v.get("jaccard_vs_latest") for v in item.get("history", {}).values() if v.get("jaccard_vs_latest") is not None]
        sector_scores.append((sector, overlaps))
    rows["pit_sector_membership"] = {
        "classification": "probe_result_required",
        "note": "If historical counts differ from latest and Jaccard is materially below 1.0, QMT may provide usable PIT sector membership.",
        "sample_overlap": sector_scores[:5],
    }

    rows["level2_realtime"] = {
        "present_in_period_list": any(bool(presence.get(p)) for p in ["l2quote", "l2order", "l2transaction", "l2orderqueue", "l2thousand"]),
        "classification": "record_realtime_from_now",
        "note": "xtdata docs say Level2 has no historical storage and is cleared across trading days unless captured locally.",
    }
    report["capability_classification"] = rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe QMT/xtdata capabilities safely.")
    parser.add_argument(
        "--with-downloads",
        action="store_true",
        help="Actively call download_* APIs. Default quick mode only reads exposed/cached data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "created_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "sample_stocks": SAMPLE_STOCKS,
        "date_windows": DATE_WINDOWS,
        "options": {"allow_downloads": bool(args.with_downloads)},
        "basic_interfaces": {},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with QMTClient(fill_data=False, max_retries=1) as client:
            xtdata = client.xtdata
            probe_basic_interfaces(xtdata, report)
            probe_period_support(xtdata, report)
            probe_sector_history(xtdata, report)
            probe_tabular_history(xtdata, report)
            probe_recent_kline_history(xtdata, report)
            probe_realtime_interfaces(xtdata, report)
            classify_capabilities(report)
    except Exception as exc:
        report["fatal"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }

    report["finished_at"] = now_iso()
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(OUT_FILE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
