"""Build an incremental SH000001 market-regime asset from QMT daily bars."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.data.qmt_client import QMTClient


DEFAULT_OUTPUT = Path("data/derived/market_regime/sh000001_daily.parquet")
DEFAULT_SYMBOL = "SH000001"
DEFAULT_OVERLAP_DAYS = 120


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-output")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_name("metadata.json")
    report = build_market_regime_file(
        output=output,
        metadata_output=metadata_output,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        overlap_days=args.overlap_days,
        refresh=args.refresh,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} rows={report['rows']} start={report['start']} "
            f"end={report['end']} updated_rows={report['updated_rows']}"
        )
    return 0


def build_market_regime_file(
    *,
    output: Path,
    metadata_output: Path,
    symbol: str,
    start: str,
    end: str,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    refresh: bool = False,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(output, refresh=refresh)
    calc_start = choose_calc_start(existing, start=start, overlap_days=overlap_days)
    with QMTClient(max_retries=3) as client:
        bars = client.query_history_k_data_with_retry(
            symbol,
            calc_start,
            end,
            fields=["open", "high", "low", "close", "volume", "amount"],
            dividend_type="none",
            download_first=True,
        )
    if bars.empty:
        raise RuntimeError(f"QMT returned no daily bars for {symbol} between {calc_start} and {end}")

    partial = normalize_bars(bars)
    merged = merge_existing(existing, partial)
    merged = add_regime_indicators(merged)
    merged.to_parquet(output, index=False)

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    metadata = {
        "created_at": now.isoformat(),
        "source": "QMTClient/xqshare/xtdata",
        "symbol": symbol,
        "output": str(output),
        "start": str(merged["date"].min().date()),
        "end": str(merged["date"].max().date()),
        "calc_start": calc_start,
        "requested_start": start,
        "requested_end": end,
        "overlap_days": int(overlap_days),
        "regime_formula": "close > EMA(close, 60) and EMA(close, 20) > EMA(close, 60)",
        "kdj_formula": "RSV(9), K=SMA_TDX(RSV,3,1), D=SMA_TDX(K,3,1), J=3*K-2*D",
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "output": str(output),
        "metadata_output": str(metadata_output),
        "rows": int(len(merged)),
        "updated_rows": int(len(partial)),
        "start": metadata["start"],
        "end": metadata["end"],
        "calc_start": calc_start,
        "active_bull_days": int(merged["active_bull"].sum()),
    }


def read_existing(output: Path, *, refresh: bool) -> pd.DataFrame:
    if refresh or not output.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(output)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def choose_calc_start(existing: pd.DataFrame, *, start: str, overlap_days: int) -> str:
    if existing.empty:
        return pd.Timestamp(start).strftime("%Y-%m-%d")
    dates = pd.DatetimeIndex(existing["date"].sort_values().unique())
    position = max(0, len(dates) - max(int(overlap_days), 70))
    return pd.Timestamp(dates[position]).strftime("%Y-%m-%d")


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    keep = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pct_chg"]
    for column in keep:
        if column not in out:
            out[column] = np.nan
    for column in keep[2:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[keep].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def merge_existing(existing: pd.DataFrame, partial: pd.DataFrame) -> pd.DataFrame:
    raw_columns = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pct_chg"]
    if existing.empty:
        return partial[raw_columns].copy()
    cutoff = pd.Timestamp(partial["date"].min())
    prefix = existing.loc[existing["date"] < cutoff, raw_columns]
    return pd.concat([prefix, partial[raw_columns]], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def add_regime_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("date").reset_index(drop=True)
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema60"] = out["close"].ewm(span=60, adjust=False, min_periods=60).mean()
    low9 = out["low"].rolling(9, min_periods=9).min()
    high9 = out["high"].rolling(9, min_periods=9).max()
    spread = high9 - low9
    out["kdj_rsv"] = (out["close"] - low9).div(spread.where(spread.abs() > 1e-12)).mul(100.0)
    out["kdj_k"] = sma_tdx(out["kdj_rsv"], 3, 1)
    out["kdj_d"] = sma_tdx(out["kdj_k"], 3, 1)
    out["kdj_j"] = 3.0 * out["kdj_k"] - 2.0 * out["kdj_d"]
    out["active_bull"] = (out["close"] > out["ema60"]) & (out["ema20"] > out["ema60"])
    out["active_bull_kdj_low"] = out["active_bull"] & (out["kdj_j"] < 13.0)
    out["active_bull_age"] = consecutive_true_count(out["active_bull"])
    out["ema20_slope5"] = out["ema20"].pct_change(5, fill_method=None)
    out["close_to_ema60"] = out["close"] / out["ema60"] - 1.0
    return out


def sma_tdx(values: pd.Series, n: int, m: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    previous = np.nan
    for index, value in enumerate(numeric):
        if not np.isfinite(value):
            continue
        previous = value if not np.isfinite(previous) else (m * value + (n - m) * previous) / n
        result[index] = previous
    return pd.Series(result, index=values.index)


def consecutive_true_count(values: pd.Series) -> pd.Series:
    flags = values.fillna(False).astype(bool).to_numpy()
    result = np.zeros(len(flags), dtype=np.int32)
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        result[index] = run
    return pd.Series(result, index=values.index)


if __name__ == "__main__":
    raise SystemExit(main())
