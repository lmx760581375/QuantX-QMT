"""Sync ETF daily bars into a qlib-style provider.

EastMoney is preferred because it provides amount and adjustment choices. On
machines where that endpoint is unstable, the script falls back to AkShare's
Sina ETF history endpoint, which has a simpler but stable OHLCV shape.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from quantx.core.data import BaostockToQlibConverter


DEFAULT_ETFS = [
    "SH510300",  # 沪深300ETF
    "SH510500",  # 中证500ETF
    "SH512100",  # 中证1000ETF
    "SZ159915",  # 创业板ETF
    "SH588080",  # 科创50ETF
    "SH510880",  # 红利ETF
    "SH512890",  # 红利低波ETF
    "SH512800",  # 银行ETF
    "SH512880",  # 证券ETF
    "SH512480",  # 半导体ETF
    "SZ159995",  # 芯片ETF
    "SH515050",  # 5GETF
    "SH512660",  # 军工ETF
    "SH512010",  # 医药ETF
    "SH512170",  # 医疗ETF
    "SH515790",  # 光伏ETF
    "SZ159928",  # 消费ETF
    "SH512400",  # 有色ETF
    "SH515220",  # 煤炭ETF
    "SH518880",  # 黄金ETF
    "SH511380",  # 转债ETF
    "SH511880",  # 银华日利
    "SH513100",  # 纳指ETF
    "SH513500",  # 标普500ETF
    "SH513030",  # 德国ETF
]


def _secid(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.startswith("SH"):
        return f"1.{symbol[2:]}"
    if symbol.startswith("SZ"):
        return f"0.{symbol[2:]}"
    raise ValueError(f"Unsupported ETF symbol: {symbol}")


def _fetch_eastmoney_kline(symbol: str, start: str, end: str, fqt: int, timeout: float) -> pd.DataFrame:
    params = {
        "secid": _secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": str(int(fqt)),
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }
    try:
        import requests

        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except ImportError:
        req = Request(url + "?" + urlencode(params), headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("rc") != 0:
        raise RuntimeError(f"EastMoney returned rc={payload.get('rc')} for {symbol}: {payload}")
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 11:
            continue
        date, open_, close, high, low, volume, amount, _amp, pct_chg, change, turnover = parts[:11]
        close_f = float(close)
        change_f = float(change)
        preclose = close_f - change_f
        rows.append({
            "date": date,
            "code": symbol,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": close_f,
            "preclose": preclose if preclose > 0 else close_f / (1 + float(pct_chg) / 100.0),
            "volume": float(volume),
            "amount": float(amount),
            "turn": float(turnover) if turnover not in {"", "-"} else 0.0,
            "tradestatus": 1,
            "pctChg": float(pct_chg),
            "isST": 0,
        })
    return pd.DataFrame(rows)


def _sina_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.startswith("SH"):
        return f"sh{symbol[2:]}"
    if symbol.startswith("SZ"):
        return f"sz{symbol[2:]}"
    raise ValueError(f"Unsupported ETF symbol: {symbol}")


def _fetch_sina_kline(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    frame = ak.fund_etf_hist_sina(symbol=_sina_symbol(symbol))
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date")
    for field in ["open", "high", "low", "close", "volume"]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df = _apply_continuity_adjustment(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    result = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "code": symbol,
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": close,
        "preclose": close.shift(1).fillna(close),
        "volume": volume,
        # Keep converter-generated vwap close to close regardless of whether
        # the source volume is shares or hands.
        "amount": close * volume * 100.0,
        "turn": 0.0,
        "tradestatus": 1,
        "pctChg": close.pct_change().fillna(0.0) * 100.0,
        "isST": 0,
    })
    return result.dropna(subset=["open", "high", "low", "close", "volume"])


def _apply_continuity_adjustment(df: pd.DataFrame, split_threshold: float = 0.35) -> pd.DataFrame:
    """Continuity-adjust Sina ETF prices around split/fund-share events.

    AkShare's Sina ETF endpoint returns raw prices. ETF share splits/fund
    conversions show up as huge one-day price jumps that are not tradable PnL.
    For research backtests we build a continuous price series by scaling all
    prior OHLC values whenever adjacent closes jump beyond ``split_threshold``.
    Normal ETF daily moves, including the 2024 broad-market limit-up days, are
    intentionally left untouched.
    """
    out = df.copy()
    close = out["close"].astype(float).to_numpy(copy=True)
    if len(close) < 2:
        return out

    factors = [1.0] * len(close)
    cumulative = 1.0
    for idx in range(len(close) - 1, 0, -1):
        cur = close[idx]
        prev = close[idx - 1]
        if prev > 0 and cur > 0:
            ratio = cur / prev
            if ratio < 1.0 - split_threshold or ratio > 1.0 + split_threshold:
                cumulative *= ratio
        factors[idx - 1] = cumulative

    factor_series = pd.Series(factors, index=out.index, dtype="float64")
    for field in ["open", "high", "low", "close"]:
        out[field] = out[field].astype(float) * factor_series
    return out


def _fetch_kline(symbol: str, start: str, end: str, fqt: int, timeout: float, source: str) -> tuple[pd.DataFrame, str]:
    source = str(source or "auto").lower()
    errors: list[str] = []
    if source in {"auto", "eastmoney", "em"}:
        try:
            frame = _fetch_eastmoney_kline(symbol, start, end, fqt=fqt, timeout=timeout)
            if not frame.empty:
                return frame, "eastmoney"
            errors.append("eastmoney empty")
        except Exception as exc:
            errors.append(f"eastmoney: {exc}")
            if source in {"eastmoney", "em"}:
                raise
    if source in {"auto", "sina"}:
        try:
            frame = _fetch_sina_kline(symbol, start, end)
            if not frame.empty:
                return frame, "sina"
            errors.append("sina empty")
        except Exception as exc:
            errors.append(f"sina: {exc}")
            if source == "sina":
                raise
    raise RuntimeError("; ".join(errors) or f"No data for {symbol}")


def _load_symbols(symbols: Iterable[str] | None, symbol_file: str | None) -> List[str]:
    result: List[str] = []
    if symbols:
        result.extend(str(sym).strip().upper() for sym in symbols if str(sym).strip())
    if symbol_file:
        path = Path(symbol_file)
        result.extend(line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return list(dict.fromkeys(result or DEFAULT_ETFS))


def sync_etfs(
    symbols: List[str],
    start: str,
    end: str,
    raw_dir: str | Path,
    provider_uri: str | Path,
    fqt: int = 1,
    pause_seconds: float = 0.2,
    timeout: float = 20.0,
    source: str = "auto",
    dry_run: bool = False,
) -> dict:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    fetched: List[str] = []
    failed: List[dict] = []
    row_counts: dict[str, int] = {}
    sources: dict[str, str] = {}
    for idx, symbol in enumerate(symbols, start=1):
        try:
            frame, used_source = _fetch_kline(symbol, start, end, fqt=fqt, timeout=timeout, source=source)
            if frame.empty:
                failed.append({"symbol": symbol, "error": "empty"})
            else:
                row_counts[symbol] = int(len(frame))
                sources[symbol] = used_source
                fetched.append(symbol)
                if not dry_run:
                    frame.to_csv(raw_path / f"{symbol}.csv", index=False)
        except Exception as exc:  # pragma: no cover - depends on remote data service
            failed.append({"symbol": symbol, "error": str(exc)})
        if idx < len(symbols):
            time.sleep(max(0.0, float(pause_seconds)))

    if fetched and not dry_run:
        converter = BaostockToQlibConverter(qlib_dir=str(provider_uri), csv_dir=str(raw_path))
        converter.convert_all(start, end, fetched)

    return {
        "ok": bool(fetched) and not failed,
        "provider_uri": str(provider_uri),
        "raw_dir": str(raw_path),
        "start": start,
        "end": end,
        "fqt": fqt,
        "requested": len(symbols),
        "fetched": fetched,
        "failed": failed,
        "row_counts": row_counts,
        "sources": sources,
        "dry_run": dry_run,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/qlib_etf_data")
    parser.add_argument("--raw-dir", default="data/raw/eastmoney/etfs")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--symbol-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fqt", type=int, default=1, choices=[0, 1, 2], help="EastMoney adjustment: 0 raw, 1 forward, 2 backward.")
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--source", default="auto", choices=["auto", "eastmoney", "em", "sina"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    symbols = _load_symbols(args.symbols, args.symbol_file)
    if args.limit is not None:
        symbols = symbols[: int(args.limit)]
    result = sync_etfs(
        symbols=symbols,
        start=args.start,
        end=args.end,
        raw_dir=args.raw_dir,
        provider_uri=args.provider_uri,
        fqt=args.fqt,
        pause_seconds=args.pause_seconds,
        timeout=args.timeout,
        source=args.source,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"ok={result['ok']} fetched={len(result['fetched'])}/{result['requested']} failed={len(result['failed'])}")
    return 0 if result["fetched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
