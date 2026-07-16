"""Sync complete available Tushare ``stock_st`` history into Git-tracked CSV assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import local
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/reference/security_state")
DEFAULT_CALENDAR = Path("data/qlib_data_fixed/calendars/day.txt")
DEFAULT_API_URL = "http://lianghua.nanyangqiankun.top"
DEFAULT_START = "2016-08-31"
RAW_COLUMNS = ["ts_code", "name", "trade_date", "type", "type_name"]
DAILY_COLUMNS = [*RAW_COLUMNS, "source_trade_date", "is_imputed"]
_thread_state = local()


@dataclass(frozen=True)
class FetchResult:
    trade_date: str
    frame: pd.DataFrame
    attempts: int
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calendar", default=str(DEFAULT_CALENDAR))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", required=True)
    parser.add_argument("--api-url", default=os.environ.get("TUSHARE_API_URL", DEFAULT_API_URL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=8.0)
    parser.add_argument("--checkpoint-days", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TS_TOKEN")
    if not token:
        raise SystemExit("Set TUSHARE_TOKEN (or TS_TOKEN); credentials are never read from repository files.")
    report = sync_st_history(
        output_dir=Path(args.output_dir),
        calendar_path=Path(args.calendar),
        start=args.start,
        end=args.end,
        token=token,
        api_url=args.api_url,
        workers=args.workers,
        retries=args.retries,
        request_timeout=args.request_timeout,
        checkpoint_days=args.checkpoint_days,
        refresh=args.refresh,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _format_report(report))
    return 0 if report["failed_dates"] == 0 else 1


def sync_st_history(
    *,
    output_dir: Path,
    calendar_path: Path,
    start: str,
    end: str,
    token: str,
    api_url: str = DEFAULT_API_URL,
    workers: int = 4,
    retries: int = 3,
    request_timeout: float = 8.0,
    checkpoint_days: int = 20,
    refresh: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_path = output_dir / "st_observed.csv"
    coverage_path = output_dir / "coverage.csv"
    daily_path = output_dir / "st_daily.csv"
    intervals_path = output_dir / "st_intervals.csv"
    metadata_path = output_dir / "metadata.json"

    calendar = load_calendar(calendar_path, start=start, end=end)
    observed = empty_observed() if refresh else read_csv(observed_path, RAW_COLUMNS)
    coverage = empty_coverage() if refresh else read_coverage(coverage_path)
    completed = set(coverage.loc[coverage["fetch_status"].isin(["observed", "empty"]), "trade_date"])
    pending = [date for date in calendar if date not in completed]
    run_started = time.perf_counter()
    print(
        f"sync_start calendar={len(calendar)} completed={len(completed)} pending={len(pending)} "
        f"workers={workers} timeout={request_timeout:.1f}s retries={retries}",
        flush=True,
    )

    batch_size = max(1, int(checkpoint_days))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        batch_started = time.perf_counter()
        print(
            f"batch_start index={offset // batch_size + 1} dates={batch[0]}..{batch[-1]} "
            f"size={len(batch)} progress={offset}/{len(pending)}",
            flush=True,
        )
        results = fetch_dates(
            batch,
            token=token,
            api_url=api_url,
            workers=max(1, int(workers)),
            retries=max(1, int(retries)),
            request_timeout=max(1.0, float(request_timeout)),
        )
        fetch_elapsed = time.perf_counter() - batch_started
        observed, coverage = merge_fetch_results(observed, coverage, results)
        write_started = time.perf_counter()
        write_csv(observed, observed_path, RAW_COLUMNS)
        write_csv(coverage, coverage_path, coverage_columns())
        write_elapsed = time.perf_counter() - write_started
        fetched = min(offset + len(batch), len(pending))
        elapsed = time.perf_counter() - run_started
        rate = fetched / elapsed if elapsed > 0 else 0.0
        remaining = len(pending) - fetched
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        observed_dates = sum(not result.frame.empty and not result.error for result in results)
        empty_dates = sum(result.frame.empty and not result.error for result in results)
        failed_dates = sum(bool(result.error) for result in results)
        print(
            f"batch_done fetched={fetched}/{len(pending)} dates={batch[0]}..{batch[-1]} "
            f"observed={observed_dates} empty={empty_dates} failed={failed_dates} "
            f"fetch_s={fetch_elapsed:.2f} write_s={write_elapsed:.2f} rate_days_s={rate:.2f} "
            f"eta={format_duration(eta_seconds)} observed_rows={len(observed)}",
            flush=True,
        )

    failed = coverage[coverage["fetch_status"] == "failed"]
    if not failed.empty:
        return build_report(output_dir, observed, coverage, pd.DataFrame(), pd.DataFrame())

    daily, effective_coverage = forward_fill_empty_dates(observed, coverage, calendar)
    intervals = build_st_intervals(daily, calendar)
    write_csv(effective_coverage, coverage_path, coverage_columns())
    write_csv(daily, daily_path, DAILY_COLUMNS)
    write_csv(intervals, intervals_path, interval_columns())
    metadata = build_metadata(
        start=start,
        end=end,
        api_url=api_url,
        calendar=calendar,
        observed=observed,
        coverage=effective_coverage,
        daily=daily,
        intervals=intervals,
        paths={
            "observed": observed_path,
            "coverage": coverage_path,
            "daily": daily_path,
            "intervals": intervals_path,
        },
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return build_report(output_dir, observed, effective_coverage, daily, intervals)


def load_calendar(path: Path, *, start: str, end: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Qlib trading calendar: {path}")
    dates = pd.DatetimeIndex(pd.to_datetime([line for line in path.read_text().splitlines() if line.strip()]))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if dates.empty:
        raise ValueError(f"No trading dates between {start} and {end}")
    return dates.strftime("%Y%m%d").tolist()


def fetch_dates(
    dates: list[str], *, token: str, api_url: str, workers: int, retries: int, request_timeout: float = 8.0
) -> list[FetchResult]:
    if not dates:
        return []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_one_date,
                date,
                token=token,
                api_url=api_url,
                retries=retries,
                request_timeout=request_timeout,
            ): date
            for date in dates
        }
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda row: row.trade_date)


def fetch_one_date(
    trade_date: str, *, token: str, api_url: str, retries: int, request_timeout: float = 8.0
) -> FetchResult:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            pro = tushare_client(token, api_url, request_timeout)
            frame = normalize_observed(pro.stock_st(trade_date=trade_date), requested_date=trade_date)
            return FetchResult(trade_date, frame, attempt)
        except Exception as exc:  # Provider/network failures must remain visible in coverage.
            last_error = exc
            print(
                f"request_retry trade_date={trade_date} attempt={attempt}/{retries} "
                f"error={type(exc).__name__}: {str(exc)[:240]}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))
    return FetchResult(trade_date, empty_observed(), retries, f"{type(last_error).__name__}: {last_error}")


def tushare_client(token: str, api_url: str, request_timeout: float):
    key = (token, api_url, float(request_timeout))
    if getattr(_thread_state, "key", None) != key:
        import tushare as ts

        pro = ts.pro_api(token, timeout=float(request_timeout))
        pro._DataApi__token = token
        pro._DataApi__http_url = api_url
        _thread_state.key = key
        _thread_state.pro = pro
    return _thread_state.pro


def normalize_observed(frame: pd.DataFrame, *, requested_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_observed()
    missing = [column for column in RAW_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"stock_st response lacks columns: {missing}")
    out = frame[RAW_COLUMNS].copy()
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    if set(out["trade_date"]) != {requested_date}:
        raise ValueError(f"stock_st returned dates {sorted(set(out['trade_date']))} for {requested_date}")
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    out["name"] = out["name"].astype(str)
    out["type"] = out["type"].astype(str)
    out["type_name"] = out["type_name"].astype(str)
    return out.drop_duplicates(["trade_date", "ts_code", "type"]).sort_values(["trade_date", "ts_code", "type"])


def merge_fetch_results(
    observed: pd.DataFrame, coverage: pd.DataFrame, results: list[FetchResult]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = {result.trade_date for result in results}
    observed = observed[~observed["trade_date"].isin(dates)]
    frames = [result.frame for result in results if not result.frame.empty]
    if frames:
        observed = pd.concat([observed, *frames], ignore_index=True)
    observed = observed.drop_duplicates(["trade_date", "ts_code", "type"]).sort_values(["trade_date", "ts_code", "type"])

    coverage = coverage[~coverage["trade_date"].isin(dates)]
    rows = []
    for result in results:
        rows.append({
            "trade_date": result.trade_date,
            "fetch_status": "failed" if result.error else ("empty" if result.frame.empty else "observed"),
            "observed_row_count": int(len(result.frame)),
            "effective_row_count": int(len(result.frame)),
            "source_trade_date": result.trade_date if not result.frame.empty else "",
            "is_imputed": False,
            "attempts": int(result.attempts),
            "error": result.error or "",
        })
    coverage = pd.concat([coverage, pd.DataFrame(rows)], ignore_index=True)
    return observed.reset_index(drop=True), coverage.sort_values("trade_date").reset_index(drop=True)


def forward_fill_empty_dates(
    observed: pd.DataFrame, coverage: pd.DataFrame, calendar: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_date = {date: group.copy() for date, group in observed.groupby("trade_date", sort=False)}
    coverage_by_date = coverage.set_index("trade_date").to_dict("index")
    daily_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    last_frame: pd.DataFrame | None = None
    last_source = ""
    for date in calendar:
        row = dict(coverage_by_date.get(date, {}))
        status = row.get("fetch_status")
        if status == "failed" or status is None:
            raise ValueError(f"Cannot fill ST state before successful fetch: {date}")
        source = by_date.get(date)
        is_imputed = False
        if source is not None and not source.empty:
            last_frame = source
            last_source = date
        elif last_frame is not None:
            source = last_frame
            is_imputed = True
        else:
            source = empty_observed()

        if not source.empty:
            effective = source.copy()
            effective["trade_date"] = date
            effective["source_trade_date"] = last_source
            effective["is_imputed"] = is_imputed
            daily_frames.append(effective[DAILY_COLUMNS])
        row.update({
            "trade_date": date,
            "effective_row_count": int(len(source)),
            "source_trade_date": last_source,
            "is_imputed": bool(is_imputed),
        })
        coverage_rows.append(row)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=DAILY_COLUMNS)
    return daily.sort_values(["trade_date", "ts_code", "type"]).reset_index(drop=True), pd.DataFrame(coverage_rows)[coverage_columns()]


def build_st_intervals(daily: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=interval_columns())
    position = {date: index for index, date in enumerate(calendar)}
    data = daily.copy()
    data["calendar_pos"] = data["trade_date"].map(position)
    records = []
    for (ts_code, st_type), group in data.groupby(["ts_code", "type"], sort=True):
        group = group.sort_values("calendar_pos").reset_index(drop=True)
        block = group["calendar_pos"].diff().ne(1).cumsum()
        for _, interval in group.groupby(block, sort=False):
            records.append({
                "ts_code": ts_code,
                "type": st_type,
                "type_name": interval["type_name"].mode().iloc[0],
                "start_date": interval.iloc[0]["trade_date"],
                "end_date": interval.iloc[-1]["trade_date"],
                "trading_days": int(len(interval)),
                "imputed_days": int(interval["is_imputed"].sum()),
                "names": "|".join(sorted(set(interval["name"].astype(str)))),
            })
    return pd.DataFrame(records, columns=interval_columns()).sort_values(["ts_code", "start_date", "type"]).reset_index(drop=True)


def build_metadata(*, start, end, api_url, calendar, observed, coverage, daily, intervals, paths) -> dict:
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source": "Tushare Pro stock_st",
        "api_url": api_url,
        "requested_start": start,
        "requested_end": end,
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
        "trading_days": len(calendar),
        "observed_dates": int((coverage["fetch_status"] == "observed").sum()),
        "empty_dates": int((coverage["fetch_status"] == "empty").sum()),
        "imputed_dates": int(coverage["is_imputed"].sum()),
        "precoverage_empty_dates": int(((coverage["fetch_status"] == "empty") & (coverage["source_trade_date"] == "")).sum()),
        "observed_rows": int(len(observed)),
        "effective_daily_rows": int(len(daily)),
        "interval_rows": int(len(intervals)),
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "notes": [
            "Empty API responses after the first observed date are forward-filled from the previous observed trading date.",
            "Rows before the first observed date remain empty and are never interpreted as confirmed non-ST states.",
            "source_trade_date and is_imputed preserve provenance for every effective daily row.",
            "Credentials are supplied only through TUSHARE_TOKEN/TS_TOKEN and are never persisted.",
        ],
    }


def build_report(output_dir, observed, coverage, daily, intervals) -> dict:
    return {
        "ok": bool(not (coverage.get("fetch_status", pd.Series(dtype=str)) == "failed").any()),
        "output_dir": str(output_dir),
        "coverage_dates": int(len(coverage)),
        "observed_dates": int((coverage.get("fetch_status", pd.Series(dtype=str)) == "observed").sum()),
        "empty_dates": int((coverage.get("fetch_status", pd.Series(dtype=str)) == "empty").sum()),
        "imputed_dates": int(coverage.get("is_imputed", pd.Series(dtype=bool)).fillna(False).sum()),
        "failed_dates": int((coverage.get("fetch_status", pd.Series(dtype=str)) == "failed").sum()),
        "observed_rows": int(len(observed)),
        "daily_rows": int(len(daily)),
        "interval_rows": int(len(intervals)),
    }


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame[columns]


def read_coverage(path: Path) -> pd.DataFrame:
    frame = read_csv(path, coverage_columns())
    if frame.empty:
        return empty_coverage()
    for column in ["observed_row_count", "effective_row_count", "attempts"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["is_imputed"] = frame["is_imputed"].astype(str).str.lower().isin(["true", "1"])
    return frame


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, columns].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_observed() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def empty_coverage() -> pd.DataFrame:
    return pd.DataFrame(columns=coverage_columns())


def coverage_columns() -> list[str]:
    return [
        "trade_date", "fetch_status", "observed_row_count", "effective_row_count",
        "source_trade_date", "is_imputed", "attempts", "error",
    ]


def interval_columns() -> list[str]:
    return ["ts_code", "type", "type_name", "start_date", "end_date", "trading_days", "imputed_days", "names"]


def _format_report(report: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in report.items())


def format_duration(seconds: float) -> str:
    if not pd.notna(seconds) or seconds == float("inf"):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
