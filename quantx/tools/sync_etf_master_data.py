"""Resumable ETF daily-bar sync driven by the local ETF master snapshot."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantx.core.data import BaostockToQlibConverter
from quantx.tools.sync_etf_data import _fetch_kline


DEFAULT_META_PATH = Path("data/meta/snapshots/etf_master.csv")
DEFAULT_RAW_DIR = Path("data/raw/eastmoney/etfs")
DEFAULT_PROVIDER_URI = Path("data/qlib_etf_data_wufu_tmp")
DEFAULT_START = "2016-01-04"
DEFAULT_END = "2026-07-07"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


@contextmanager
def _time_limit(seconds: float):
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"symbol timeout after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _logged_symbols(log_path: Path) -> tuple[set[str], set[str]]:
    done: set[str] = set()
    failed: set[str] = set()
    if not log_path.exists():
        return done, failed
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "symbol_done":
            symbol = str(row.get("symbol", "")).upper()
            if not symbol:
                continue
            if row.get("ok"):
                done.add(symbol)
                failed.discard(symbol)
            else:
                failed.add(symbol)
    return done, failed


def _load_target_symbols(args: argparse.Namespace) -> list[str]:
    meta = pd.read_csv(args.etf_meta_path)
    if "symbol" not in meta.columns:
        raise ValueError(f"missing symbol column in {args.etf_meta_path}")
    meta["symbol"] = meta["symbol"].astype(str).str.upper()
    if args.categories:
        meta = meta[meta["category"].astype(str).isin(args.categories)]
    if args.dynamic_theme_only:
        meta = meta[meta.get("is_dynamic_theme_candidate", False).astype(bool)]
    if "latest_amount" in meta.columns:
        meta["latest_amount"] = pd.to_numeric(meta["latest_amount"], errors="coerce").fillna(0.0)
        meta = meta.sort_values(["latest_amount", "symbol"], ascending=[False, True])
    else:
        meta = meta.sort_values("symbol")

    symbols = meta["symbol"].drop_duplicates().tolist()
    if not args.include_existing:
        existing = {path.stem.upper() for path in Path(args.raw_dir).glob("*.csv")}
        symbols = [symbol for symbol in symbols if symbol not in existing]
    if args.limit is not None:
        symbols = symbols[: int(args.limit)]
    return symbols


def _convert_batch(provider_uri: Path, raw_dir: Path, start: str, end: str, symbols: list[str], log_path: Path) -> None:
    if not symbols:
        return
    started = time.time()
    try:
        converter = BaostockToQlibConverter(qlib_dir=str(provider_uri), csv_dir=str(raw_dir))
        converter.convert_all(start, end, symbols)
        _append_jsonl(log_path, {
            "event": "convert_done",
            "ok": True,
            "symbols": symbols,
            "elapsed_seconds": round(time.time() - started, 2),
        })
    except Exception as exc:  # pragma: no cover - depends on filesystem/data state
        _append_jsonl(log_path, {
            "event": "convert_done",
            "ok": False,
            "symbols": symbols,
            "error": str(exc),
            "elapsed_seconds": round(time.time() - started, 2),
        })


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    provider_uri = Path(args.provider_uri)
    log_path = Path(args.log_path) if args.log_path else Path("artifacts/etf_sync") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_etf_master_sync.jsonl"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    provider_uri.mkdir(parents=True, exist_ok=True)

    symbols = _load_target_symbols(args)
    completed, logged_failed = _logged_symbols(log_path) if args.resume else (set(), set())
    skipped = completed | (logged_failed if args.skip_logged_failures else set())
    symbols = [symbol for symbol in symbols if symbol not in skipped]
    _append_jsonl(log_path, {
        "event": "sync_start",
        "requested": len(symbols),
        "meta_path": str(args.etf_meta_path),
        "raw_dir": str(raw_dir),
        "provider_uri": str(provider_uri),
        "start": args.start,
        "end": args.end,
        "source_order": args.source_order,
        "retries": args.retries,
        "pause_seconds": args.pause_seconds,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "skip_logged_failures": args.skip_logged_failures,
        "logged_failed_skipped": sorted(logged_failed) if args.skip_logged_failures else [],
    })

    fetched: list[str] = []
    failed: list[dict[str, Any]] = []
    convert_pending: list[str] = []
    started_all = time.time()
    for idx, symbol in enumerate(symbols, start=1):
        symbol_started = time.time()
        symbol_errors: list[str] = []
        ok = False
        used_source = ""
        rows = 0
        for attempt in range(1, int(args.retries) + 1):
            for source in args.source_order:
                try:
                    with _time_limit(float(args.symbol_timeout)):
                        frame, used_source = _fetch_kline(
                            symbol,
                            args.start,
                            args.end,
                            fqt=args.fqt,
                            timeout=args.timeout,
                            source=source,
                        )
                    if frame.empty:
                        symbol_errors.append(f"attempt={attempt} source={source}: empty")
                        continue
                    frame.to_csv(raw_dir / f"{symbol}.csv", index=False)
                    rows = int(len(frame))
                    ok = True
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    symbol_errors.append(f"attempt={attempt} source={source}: {exc}")
            if ok:
                break
            time.sleep(max(0.0, float(args.retry_sleep_seconds)) * attempt)

        event = {
            "event": "symbol_done",
            "ok": ok,
            "idx": idx,
            "total": len(symbols),
            "symbol": symbol,
            "source": used_source,
            "rows": rows,
            "elapsed_seconds": round(time.time() - symbol_started, 2),
            "errors": symbol_errors[-5:],
        }
        _append_jsonl(log_path, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if ok:
            fetched.append(symbol)
            convert_pending.append(symbol)
            if not args.no_convert and len(convert_pending) >= int(args.convert_batch_size):
                _convert_batch(provider_uri, raw_dir, args.start, args.end, convert_pending, log_path)
                convert_pending = []
        else:
            failed.append({"symbol": symbol, "errors": symbol_errors[-5:]})

        if idx < len(symbols):
            time.sleep(max(0.0, float(args.pause_seconds)))

    if not args.no_convert and convert_pending:
        _convert_batch(provider_uri, raw_dir, args.start, args.end, convert_pending, log_path)

    result = {
        "ok": not failed,
        "requested": len(symbols),
        "fetched": len(fetched),
        "failed_count": len(failed),
        "failed": failed,
        "log_path": str(log_path),
        "elapsed_seconds": round(time.time() - started_all, 2),
    }
    _append_jsonl(log_path, {"event": "sync_done", **result})
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf-meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--dynamic-theme-only", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-order", nargs="+", default=["auto", "sina"], choices=["auto", "eastmoney", "em", "sina"])
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--symbol-timeout", type=float, default=90.0)
    parser.add_argument("--fqt", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--convert-batch-size", type=int, default=50)
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--log-path")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--retry-logged-failures", dest="skip_logged_failures", action="store_false")
    parser.set_defaults(skip_logged_failures=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    return 0 if result["fetched"] or result["requested"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
