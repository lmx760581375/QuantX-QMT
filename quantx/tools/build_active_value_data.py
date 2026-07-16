"""Build derived active-value market data from local Qlib daily bars."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.factor_runtime import ops as factor_ops
from quantx.tools.run_backtest import load_a_share_symbols


DEFAULT_OUTPUT = Path("data/derived/active_value/daily.parquet")
DEFAULT_0AMV_OUTPUT = Path("data/derived/0amv/daily.csv")
DEFAULT_META = Path("data/meta/quantx_meta.sqlite")
FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"]
ST_LIKE_WINDOW = 80
MARKET_BREADTH_WARMUP = 140
MIN_DAILY_COVERAGE = 0.98
AMV_MARKETS = {
    "0AMV_SH": ("上证0AMV", "sh_mainboard"),
    "0AMV_SZ": ("深证0AMV", "sz_mainboard"),
    "0AMV_KC": ("科创0AMV", "star"),
    "0AMV_CY": ("创业0AMV", "chinext"),
    "0AMV_ALL": ("全市场0AMV", "all"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed", help="Qlib provider directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output active-value daily parquet path.")
    parser.add_argument(
        "--0amv-output",
        dest="amv_output",
        default=str(DEFAULT_0AMV_OUTPUT),
        help="Output 0AMV daily CSV path.",
    )
    parser.add_argument("--metadata-output", help="Metadata JSON path. Defaults to output sibling metadata.json.")
    parser.add_argument("--meta-db", default=str(DEFAULT_META), help="Security metadata SQLite path for current ST/delist names.")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--universe", default="all_a", choices=["all_a", "all_mainboard"])
    parser.add_argument("--limit", type=int, help="Optional symbol limit for smoke tests.")
    parser.add_argument("--overlap-days", type=int, default=140, help="Trading rows to recompute before existing max date.")
    parser.add_argument("--min-listing-days", type=int, default=120)
    parser.add_argument("--refresh", action="store_true", help="Ignore existing derived data and rebuild from --start.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    args = parser.parse_args(argv)

    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_name("metadata.json")
    report = build_active_value_file(
        provider_uri=Path(args.provider_uri),
        output=output,
        metadata_output=metadata_output,
        amv_output=Path(args.amv_output),
        meta_db=Path(args.meta_db),
        start=args.start,
        end=args.end,
        universe=args.universe,
        limit=args.limit,
        overlap_days=args.overlap_days,
        min_listing_days=args.min_listing_days,
        refresh=args.refresh,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} rows={report['rows']} start={report['start']} end={report['end']} "
            f"updated_rows={report['updated_rows']} output={report['output']}"
        )
    return 0 if report["ok"] else 1


def build_active_value_file(
    *,
    provider_uri: Path,
    output: Path,
    metadata_output: Path,
    amv_output: Path | None = DEFAULT_0AMV_OUTPUT,
    meta_db: Path,
    start: str,
    end: str,
    universe: str = "all_a",
    limit: int | None = None,
    overlap_days: int = 140,
    min_listing_days: int = 120,
    refresh: bool = False,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    if amv_output is not None:
        amv_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(output, refresh=refresh)
    calendar = QlibBinReader(provider_uri).calendar(start, end)
    if len(calendar) == 0:
        raise ValueError(f"No trading calendar rows between {start} and {end}")
    merge_start = choose_calc_start(calendar, existing, start=start, overlap_days=overlap_days)
    calc_start = choose_calc_start(
        calendar,
        existing,
        start=start,
        overlap_days=overlap_days + MARKET_BREADTH_WARMUP,
    )
    symbols = load_a_share_symbols(provider_uri, calc_start, end, limit=limit, universe=universe)
    listed_dates = load_listed_dates(provider_uri, symbols)
    reader = QlibBinReader(provider_uri)
    quote = reader.features(symbols, FIELDS, calc_start, end)
    quote = normalize_quote(quote)
    risk_names = load_current_risk_names(meta_db)
    partial = compute_active_value_daily(
        quote,
        risk_names=risk_names,
        min_listing_days=min_listing_days,
        calendar=reader.calendar(None, end),
        listed_dates=listed_dates,
    )
    partial = partial[partial["date"] >= pd.Timestamp(merge_start)].reset_index(drop=True)
    merged = merge_existing(existing, partial)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    merged = finalize_active_value_daily(
        merged,
        mark_latest_incomplete=is_current_session_incomplete(merged, now=now),
    )
    merged.to_parquet(output, index=False)
    existing_amv = read_existing_0amv(amv_output, refresh=refresh) if amv_output is not None else pd.DataFrame()
    amv_calc_start = start if existing_amv.empty else calc_start
    amv_quote = quote
    if universe != "all_a" or amv_calc_start != calc_start:
        amv_symbols = load_a_share_symbols(provider_uri, amv_calc_start, end, limit=limit, universe="all_a")
        amv_quote = normalize_quote(reader.features(amv_symbols, FIELDS, amv_calc_start, end))
    amv_rows = merge_existing_0amv(existing_amv, compute_0amv_daily(amv_quote))
    if amv_output is not None:
        amv_rows.to_csv(amv_output, index=False, encoding="utf-8")
    updated_rows = int(len(partial))
    metadata = {
        "created_at": now.isoformat(),
        "provider_uri": str(provider_uri),
        "output": str(output),
        "0amv_output": str(amv_output) if amv_output is not None else None,
        "start": str(merged["date"].min().date()) if len(merged) else None,
        "end": str(merged["date"].max().date()) if len(merged) else None,
        "calc_start": calc_start,
        "merge_start": merge_start,
        "0amv_calc_start": amv_calc_start,
        "requested_start": start,
        "requested_end": end,
        "universe": universe,
        "symbols": len(symbols),
        "risk_names": len(risk_names),
        "overlap_days": int(overlap_days),
        "min_listing_days": int(min_listing_days),
        "primary_series": "active_core_amount",
        "notes": [
            "active_all_amount sums all basic-tradable A-share amount.",
            "active_core_amount sums mainboard, non-ST, non-ST-like, non-new, non-suspended amount.",
            "active_vwap_value uses volume*vwap for the same core universe and falls back to amount if vwap is missing.",
            "A current-day provider row is marked incomplete before 15:30 Asia/Shanghai; coverage checks apply on every date.",
            "Strong-up states require complete current/comparison dates and at least 98% of rolling-normal bar coverage.",
            "ret/ma/state columns are recomputed after merging existing and updated rows.",
            "Market breadth uses close > yellow and white > yellow, where white=EMA(EMA(close,10),10) and yellow=(MA14+MA28+MA57+MA114)/4.",
            "Incremental reads include 140 extra trading rows for market-breadth warmup and only replace rows from merge_start.",
            "0AMV CSV stores one virtual K-line instrument per market: SH, SZ, STAR, CHINEXT, ALL.",
        ],
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "output": str(output),
        "0amv_output": str(amv_output) if amv_output is not None else None,
        "metadata_output": str(metadata_output),
        "rows": int(len(merged)),
        "updated_rows": updated_rows,
        "start": metadata["start"],
        "end": metadata["end"],
        "calc_start": calc_start,
        "merge_start": merge_start,
        "0amv_calc_start": amv_calc_start,
        "symbols": len(symbols),
    }


def read_existing(output: Path, *, refresh: bool) -> pd.DataFrame:
    if refresh or not output.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(output)
    if "date" not in frame.columns:
        raise ValueError(f"Existing active-value file lacks date column: {output}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def read_existing_0amv(output: Path | None, *, refresh: bool) -> pd.DataFrame:
    if output is None or refresh or not output.exists():
        return pd.DataFrame()
    frame = pd.read_csv(output)
    if "date" not in frame.columns or "symbol" not in frame.columns:
        raise ValueError(f"Existing 0AMV file lacks date/symbol columns: {output}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def merge_existing_0amv(existing: pd.DataFrame, partial: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return partial.copy()
    if partial.empty:
        return existing.copy()
    cutoff = str(partial["date"].min())
    merged = pd.concat([existing[existing["date"] < cutoff], partial], ignore_index=True)
    return merged.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"]).reset_index(drop=True)


def choose_calc_start(calendar: pd.DatetimeIndex, existing: pd.DataFrame, *, start: str, overlap_days: int) -> str:
    if existing.empty:
        return pd.Timestamp(start).strftime("%Y-%m-%d")
    last = pd.Timestamp(existing["date"].max())
    pos = int(calendar.searchsorted(last, side="left"))
    start_pos = max(0, pos - max(int(overlap_days), ST_LIKE_WINDOW + 5))
    return pd.Timestamp(calendar[start_pos]).strftime("%Y-%m-%d")


def normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    if quote.empty:
        return pd.DataFrame(columns=["instrument", "datetime", *FIELDS])
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for col in FIELDS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan
    return frame


def compute_active_value_daily(
    quote: pd.DataFrame,
    *,
    risk_names: set[str],
    min_listing_days: int,
    calendar: pd.DatetimeIndex | None = None,
    listed_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if quote.empty:
        return pd.DataFrame(columns=base_output_columns())
    frame = quote.copy()
    grouped = frame.groupby("instrument", sort=False)
    frame["prev_close"] = grouped["$close"].shift(1)
    frame["listing_days"] = listing_days(frame, calendar=calendar, listed_dates=listed_dates)
    frame["is_mainboard"] = frame["instrument"].astype(str).map(is_mainboard_symbol)
    frame["is_current_risk_name"] = frame["instrument"].astype(str).isin(risk_names)
    frame["basic_tradable"] = (
        (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
        & (frame["$open"] > 0)
        & (frame["$close"] > 0)
    )
    abs_ret = grouped["$close"].pct_change(fill_method=None).abs()
    five_limit = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten_limit = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five_limit.groupby(frame["instrument"]).rolling(ST_LIKE_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True) >= 3
    ) & (
        ten_limit.groupby(frame["instrument"]).rolling(ST_LIKE_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True) == 0
    )
    frame["clean_universe"] = (
        frame["basic_tradable"].fillna(False)
        & frame["is_mainboard"].fillna(False)
        & (~frame["is_current_risk_name"].fillna(False))
        & (~frame["st_like_limit_history"].fillna(False))
        & (frame["listing_days"] >= int(min_listing_days))
    )
    frame = add_market_breadth_state(frame)
    frame["vwap_value"] = frame["$volume"] * frame["$vwap"]
    frame.loc[~np.isfinite(frame["vwap_value"]), "vwap_value"] = frame["$amount"]
    frame["active_all_component"] = frame["$amount"].where(frame["basic_tradable"], 0.0)
    frame["active_tradable_component"] = frame["$amount"].where(
        frame["basic_tradable"] & (~frame["is_current_risk_name"].fillna(False)) & (~frame["st_like_limit_history"].fillna(False)),
        0.0,
    )
    frame["active_mainboard_component"] = frame["$amount"].where(frame["basic_tradable"] & frame["is_mainboard"], 0.0)
    frame["active_core_component"] = frame["$amount"].where(frame["clean_universe"], 0.0)
    frame["active_vwap_component"] = frame["vwap_value"].where(frame["clean_universe"], 0.0)
    frame["right_side_amount_component"] = frame["$amount"].where(frame["right_side"], 0.0)
    frame["right_side_core_amount_component"] = frame["$amount"].where(frame["right_side_core"], 0.0)
    daily = frame.groupby("datetime", as_index=False).agg(
        active_all_amount=("active_all_component", "sum"),
        active_tradable_amount=("active_tradable_component", "sum"),
        active_mainboard_amount=("active_mainboard_component", "sum"),
        active_core_amount=("active_core_component", "sum"),
        active_vwap_value=("active_vwap_component", "sum"),
        symbol_count=("instrument", "nunique"),
        core_symbol_count=("clean_universe", "sum"),
        risk_symbol_count=("is_current_risk_name", "sum"),
        st_like_symbol_count=("st_like_limit_history", "sum"),
        basic_tradable_count=("basic_tradable", "sum"),
        right_side_count=("right_side", "sum"),
        right_side_core_count=("right_side_core", "sum"),
        right_side_kdj_low_count=("right_side_kdj_low", "sum"),
        right_side_core_kdj_low_count=("right_side_core_kdj_low", "sum"),
        right_side_amount=("right_side_amount_component", "sum"),
        right_side_core_amount=("right_side_core_amount_component", "sum"),
    )
    daily["right_side_ratio"] = daily["right_side_count"].div(daily["basic_tradable_count"].where(daily["basic_tradable_count"] > 0))
    daily["right_side_core_ratio"] = daily["right_side_core_count"].div(daily["core_symbol_count"].where(daily["core_symbol_count"] > 0))
    daily["right_side_kdj_low_ratio"] = daily["right_side_kdj_low_count"].div(daily["basic_tradable_count"].where(daily["basic_tradable_count"] > 0))
    daily["right_side_core_kdj_low_ratio"] = daily["right_side_core_kdj_low_count"].div(daily["core_symbol_count"].where(daily["core_symbol_count"] > 0))
    daily["right_side_amount_ratio"] = daily["right_side_amount"].div(daily["active_all_amount"].where(daily["active_all_amount"] > 0))
    daily["right_side_core_amount_ratio"] = daily["right_side_core_amount"].div(daily["active_core_amount"].where(daily["active_core_amount"] > 0))
    daily = daily.rename(columns={"datetime": "date"}).sort_values("date").reset_index(drop=True)
    return daily[base_output_columns()]


def add_market_breadth_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stock-level right-side and KDJ-low flags before daily aggregation."""
    out = frame
    size = len(out)
    short_trend = np.full(size, np.nan, dtype=np.float32)
    long_trend = np.full(size, np.nan, dtype=np.float32)
    kdj_j = np.full(size, np.nan, dtype=np.float32)
    high_values = out["$high"].to_numpy(dtype=np.float32)
    low_values = out["$low"].to_numpy(dtype=np.float32)
    close_values = out["$close"].to_numpy(dtype=np.float32)
    for positions in out.groupby("instrument", sort=False).indices.values():
        loc = np.asarray(positions, dtype=np.int64)
        close = pd.Series(close_values[loc], dtype="float32")
        ema10 = factor_ops.ema(close_values[loc], 10)
        short_trend[loc] = factor_ops.ema(ema10, 10)
        long_parts = [close.rolling(window, min_periods=window).mean() for window in (14, 28, 57, 114)]
        long_trend[loc] = pd.concat(long_parts, axis=1).mean(axis=1, skipna=False).to_numpy(dtype=np.float32)
        kdj_j[loc] = factor_ops.kdj_j(high_values[loc], low_values[loc], close_values[loc], 9)
    out["market_short_trend"] = short_trend
    out["market_long_trend"] = long_trend
    out["market_kdj_j"] = kdj_j
    valid = (
        out["basic_tradable"]
        & np.isfinite(out["market_short_trend"])
        & np.isfinite(out["market_long_trend"])
    )
    out["right_side"] = valid & (out["$close"] > out["market_long_trend"]) & (out["market_short_trend"] > out["market_long_trend"])
    out["right_side_core"] = out["right_side"] & out["clean_universe"]
    out["right_side_kdj_low"] = out["right_side"] & np.isfinite(out["market_kdj_j"]) & (out["market_kdj_j"] < 13.0)
    out["right_side_core_kdj_low"] = out["right_side_kdj_low"] & out["clean_universe"]
    return out


def compute_0amv_daily(quote: pd.DataFrame) -> pd.DataFrame:
    """Build virtual daily K-line rows for 0AMV market active-value series."""
    columns = [
        "date",
        "symbol",
        "name",
        "market",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "member_count",
        "source",
        "updated_at",
    ]
    if quote.empty:
        return pd.DataFrame(columns=columns)

    frame = quote.copy()
    frame["market"] = frame["instrument"].astype(str).map(amv_market_bucket)
    frame["basic_tradable"] = (
        (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
        & (frame["$open"] > 0)
        & (frame["$close"] > 0)
    )
    frame = frame[frame["basic_tradable"] & frame["market"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    market_rows = aggregate_0amv_market_rows(frame)
    all_rows = aggregate_0amv_market_rows(frame.assign(market="all"))
    rows = pd.concat([market_rows, all_rows], ignore_index=True).sort_values(["symbol", "date"])
    rows["open"] = rows.groupby("symbol")["close"].shift(1).fillna(rows["close"])
    rows["high"] = rows[["open", "close"]].max(axis=1)
    rows["low"] = rows[["open", "close"]].min(axis=1)
    rows["source"] = "quantx-qmt-derived"
    rows["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    return rows[columns].sort_values(["date", "symbol"]).reset_index(drop=True)


def aggregate_0amv_market_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.groupby(["datetime", "market"], as_index=False).agg(
        close=("$amount", "sum"),
        volume=("$volume", "sum"),
        member_count=("instrument", "nunique"),
    )
    reverse_market = {market: (symbol, name) for symbol, (name, market) in AMV_MARKETS.items()}
    rows["symbol"] = rows["market"].map(lambda market: reverse_market[str(market)][0])
    rows["name"] = rows["market"].map(lambda market: reverse_market[str(market)][1])
    rows["amount"] = rows["close"]
    rows = rows.rename(columns={"datetime": "date"})
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    return rows


def amv_market_bucket(symbol: str) -> str | None:
    value = str(symbol).upper()
    if value.startswith("SH688"):
        return "star"
    if value.startswith("SH"):
        return "sh_mainboard"
    if value.startswith("SZ300") or value.startswith("SZ301"):
        return "chinext"
    if value.startswith("SZ"):
        return "sz_mainboard"
    return None


def listing_days(
    frame: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | None,
    listed_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    if calendar is None or len(calendar) == 0:
        return frame.groupby("instrument", sort=False).cumcount()
    if listed_dates:
        first_dates = frame["instrument"].astype(str).map(listed_dates)
        first_dates = pd.to_datetime(first_dates).fillna(frame.groupby("instrument", sort=False)["datetime"].transform("min"))
    else:
        first_dates = frame.groupby("instrument", sort=False)["datetime"].transform("min")
    current_pos = calendar.searchsorted(pd.DatetimeIndex(frame["datetime"]), side="left")
    first_pos = calendar.searchsorted(pd.DatetimeIndex(first_dates), side="left")
    return pd.Series(current_pos - first_pos, index=frame.index)


def load_listed_dates(provider_uri: Path, symbols: Iterable[str]) -> dict[str, pd.Timestamp]:
    path = provider_uri / "instruments" / "all.txt"
    wanted = {str(symbol) for symbol in symbols}
    listed: dict[str, pd.Timestamp] = {}
    if not path.exists():
        return listed
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        symbol, listed_date = parts[:2]
        if symbol in wanted:
            listed[symbol] = pd.Timestamp(listed_date)
    return listed


def merge_existing(existing: pd.DataFrame, partial: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return partial.copy()
    if partial.empty:
        return existing.copy()
    cutoff = pd.Timestamp(partial["date"].min())
    merged = pd.concat([existing[existing["date"] < cutoff], partial], ignore_index=True)
    return merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def finalize_active_value_daily(
    frame: pd.DataFrame,
    *,
    mark_latest_incomplete: bool = False,
    min_daily_coverage: float = MIN_DAILY_COVERAGE,
) -> pd.DataFrame:
    out = frame.copy().sort_values("date").reset_index(drop=True)
    expected_count = out["symbol_count"].rolling(20, min_periods=1).max()
    out["bar_coverage_ratio"] = out["symbol_count"].div(expected_count).clip(upper=1.0)
    expected_tradable = out["basic_tradable_count"].rolling(20, min_periods=1).median()
    out["tradable_coverage_ratio"] = out["basic_tradable_count"].div(expected_tradable).clip(upper=1.0)
    out["is_complete_day"] = (
        out["bar_coverage_ratio"].ge(float(min_daily_coverage))
        & out["tradable_coverage_ratio"].ge(float(min_daily_coverage))
    )
    if mark_latest_incomplete and not out.empty:
        out.loc[out.index[-1], "is_complete_day"] = False

    for name in active_series_names():
        out[f"{name}_ret1"] = out[name].pct_change(fill_method=None)
        out[f"{name}_ret2"] = out[name].pct_change(2, fill_method=None)
        out[f"{name}_ma10"] = out[name].rolling(10, min_periods=5).mean()
        raw_state = (out[f"{name}_ret1"] >= 0.04) | (out[f"{name}_ret2"] >= 0.04)
        valid_ret1 = out["is_complete_day"] & out["is_complete_day"].shift(1, fill_value=False)
        valid_ret2 = valid_ret1 & out["is_complete_day"].shift(2, fill_value=False)
        out[f"{name}_strong_up_day_raw"] = raw_state
        out[f"{name}_strong_up_day"] = (
            (out[f"{name}_ret1"] >= 0.04) & valid_ret1
        ) | (
            (out[f"{name}_ret2"] >= 0.04) & valid_ret2
        )
        out[f"{name}_above_ma10"] = (out[name] >= out[f"{name}_ma10"]) & out["is_complete_day"]
    for name in ("right_side_ratio", "right_side_core_ratio", "right_side_amount_ratio", "right_side_core_amount_ratio"):
        if name not in out:
            continue
        out[f"{name}_ret1"] = out[name].pct_change(fill_method=None)
        out[f"{name}_ema20"] = out[name].ewm(span=20, adjust=False, min_periods=20).mean()
        out[f"{name}_ema60"] = out[name].ewm(span=60, adjust=False, min_periods=60).mean()
    breadth_name = "right_side_core_ratio"
    if breadth_name in out:
        out["breadth_bull"] = (
            (out[breadth_name] > out[f"{breadth_name}_ema60"])
            & (out[f"{breadth_name}_ema20"] > out[f"{breadth_name}_ema60"])
            & out["is_complete_day"]
        )
        out["breadth_bull_start"] = out["breadth_bull"] & (~out["breadth_bull"].shift(1, fill_value=False))
        out["breadth_bull_age"] = consecutive_true_count(out["breadth_bull"])
    return out


def consecutive_true_count(values: pd.Series) -> pd.Series:
    flags = values.fillna(False).astype(bool).to_numpy()
    result = np.zeros(len(flags), dtype=np.int32)
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        result[index] = run
    return pd.Series(result, index=values.index)


def is_current_session_incomplete(frame: pd.DataFrame, *, now: datetime) -> bool:
    if frame.empty:
        return False
    latest = pd.Timestamp(frame["date"].max()).date()
    local_now = now.astimezone(ZoneInfo("Asia/Shanghai"))
    market_data_ready = local_now.replace(hour=15, minute=30, second=0, microsecond=0)
    return latest == local_now.date() and local_now < market_data_ready


def load_current_risk_names(meta_db: Path) -> set[str]:
    if not meta_db.exists():
        return set()
    risk: set[str] = set()
    with sqlite3.connect(meta_db) as conn:
        try:
            rows = conn.execute("select symbol, name from security_master")
        except sqlite3.Error:
            return risk
        for symbol, name in rows:
            upper = str(name or "").upper()
            if "ST" in upper or "退" in upper:
                risk.add(normalize_symbol(str(symbol)))
    return risk


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value:
        code, exchange = value.split(".", 1)
        return f"{exchange}{code}"
    return value


def is_mainboard_symbol(symbol: str) -> bool:
    value = str(symbol).upper()
    return not (value.startswith("SH688") or value.startswith("SZ300") or value.startswith("SZ301"))


def active_series_names() -> list[str]:
    return [
        "active_all_amount",
        "active_tradable_amount",
        "active_mainboard_amount",
        "active_core_amount",
        "active_vwap_value",
    ]


def base_output_columns() -> list[str]:
    return [
        "date",
        "active_all_amount",
        "active_tradable_amount",
        "active_mainboard_amount",
        "active_core_amount",
        "active_vwap_value",
        "symbol_count",
        "core_symbol_count",
        "risk_symbol_count",
        "st_like_symbol_count",
        "basic_tradable_count",
        "right_side_count",
        "right_side_core_count",
        "right_side_kdj_low_count",
        "right_side_core_kdj_low_count",
        "right_side_amount",
        "right_side_core_amount",
        "right_side_ratio",
        "right_side_core_ratio",
        "right_side_kdj_low_ratio",
        "right_side_core_kdj_low_ratio",
        "right_side_amount_ratio",
        "right_side_core_amount_ratio",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
