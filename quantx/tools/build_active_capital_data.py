"""Build active-capital proxies from QMT historical capital and local Qlib bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.data.qmt_client import QMTClient, normalize_symbol, to_qmt_symbol
from quantx.tools.run_backtest import load_a_share_symbols


DEFAULT_OUTPUT = Path(".tmp/quantx-research/active-capital/daily.parquet")
DEFAULT_COMPONENTS_OUTPUT = Path(".tmp/quantx-research/active-capital/components.parquet")
CAPITAL_TABLE = "Capital"
BAR_FIELDS = ["$close", "$volume", "$amount"]
QLIB_CAPITAL_FIELDS = [
    "$close",
    "$volume",
    "$amount",
    "$turnover_rate",
    "$turnover_rate_circulating",
    "$turnover_rate_free_float",
    "$circ_mv",
    "$free_float_mv",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed", help="Qlib provider directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output aggregate daily parquet path.")
    parser.add_argument(
        "--components-output",
        default=None,
        help="Optional symbol-day component parquet path. Use this for diagnostics or later cross-sectional research.",
    )
    parser.add_argument("--metadata-output", help="Metadata JSON path. Defaults to output sibling metadata.json.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--universe", default="all_a", choices=["all_a", "all_mainboard"])
    parser.add_argument(
        "--source",
        default="qlib_features",
        choices=["qlib_features", "qmt_capital"],
        help="Use prebuilt Qlib turnover/capital features by default; qmt_capital reads QMT Capital directly.",
    )
    parser.add_argument("--limit", type=int, help="Optional symbol limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=200, help="QMT financial download/read batch size.")
    parser.add_argument("--refresh", action="store_true", help="Ignore existing aggregate output and rebuild requested window.")
    parser.add_argument("--download", action="store_true", help="Run QMT download_financial_data2 before reading Capital.")
    parser.add_argument("--env-file", help="Optional xqshare .env path.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_name("metadata.json")
    components_output = Path(args.components_output) if args.components_output else None
    report = build_active_capital_file(
        provider_uri=Path(args.provider_uri),
        output=output,
        metadata_output=metadata_output,
        components_output=components_output,
        start=args.start,
        end=args.end,
        universe=args.universe,
        source=args.source,
        limit=args.limit,
        batch_size=args.batch_size,
        refresh=args.refresh,
        download=args.download,
        env_file=args.env_file,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} rows={report['rows']} start={report['start']} end={report['end']} "
            f"symbols={report['symbols']} capital_symbols={report['capital_symbols']} output={report['output']}"
        )
    return 0 if report["ok"] else 1


def build_active_capital_file(
    *,
    provider_uri: Path,
    output: Path,
    metadata_output: Path,
    components_output: Path | None = None,
    start: str,
    end: str,
    universe: str = "all_a",
    source: str = "qlib_features",
    limit: int | None = None,
    batch_size: int = 200,
    refresh: bool = False,
    download: bool = False,
    env_file: str | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    if components_output is not None:
        components_output.parent.mkdir(parents=True, exist_ok=True)

    symbols = load_a_share_symbols(provider_uri, start, end, limit=limit, universe=universe)
    if not symbols:
        raise ValueError(f"No symbols for universe={universe} start={start} end={end}")

    if source == "qlib_features":
        bars = pd.DataFrame()
        capital = empty_capital_frame()
        components = load_components_from_qlib_features(provider_uri, symbols, start, end)
    else:
        bars = load_daily_bars(provider_uri, symbols, start, end)
        capital = load_qmt_capital(symbols, batch_size=batch_size, download=download, env_file=env_file)
        components = build_components(bars, capital)
    daily = aggregate_daily(components)
    daily = add_daily_states(daily)
    merged = merge_existing(read_existing(output, refresh=refresh), daily)

    merged.to_parquet(output, index=False)
    if components_output is not None:
        components.to_parquet(components_output, index=False)

    metadata = build_metadata(
        output=output,
        metadata_output=metadata_output,
        components_output=components_output,
        provider_uri=provider_uri,
        start=start,
        end=end,
        universe=universe,
        source=source,
        symbols=symbols,
        bars=bars,
        capital=capital,
        components=components,
        daily=merged,
        download=download,
    )
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "output": str(output),
        "components_output": str(components_output) if components_output is not None else None,
        "metadata_output": str(metadata_output),
        "rows": int(len(merged)),
        "updated_rows": int(len(daily)),
        "start": str(merged["date"].min().date()) if not merged.empty else None,
        "end": str(merged["date"].max().date()) if not merged.empty else None,
        "symbols": int(len(symbols)),
        "bar_symbols": int(bars["instrument"].nunique()) if not bars.empty else 0,
        "capital_symbols": int(capital["instrument"].nunique()) if not capital.empty else 0,
        "component_symbols": int(components["instrument"].nunique()) if not components.empty else 0,
        "component_rows": int(len(components)),
        "download": bool(download),
    }


def load_daily_bars(provider_uri: Path, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    frame = QlibBinReader(provider_uri).features(symbols, BAR_FIELDS, start, end).reset_index()
    if frame.empty:
        return pd.DataFrame(columns=["instrument", "datetime", *BAR_FIELDS])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in BAR_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["instrument"] = frame["instrument"].astype(str).map(normalize_symbol)
    return frame.sort_values(["instrument", "datetime"]).reset_index(drop=True)


def load_components_from_qlib_features(provider_uri: Path, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    frame = QlibBinReader(provider_uri).features(symbols, QLIB_CAPITAL_FIELDS, start, end).reset_index()
    columns = component_columns()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str).map(normalize_symbol)
    for column in QLIB_CAPITAL_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    out = frame.rename(
        columns={
            "datetime": "date",
            "$close": "close",
            "$volume": "volume",
            "$amount": "amount",
            "$turnover_rate_circulating": "turnover_circulating",
            "$turnover_rate_free_float": "turnover_free_float",
            "$circ_mv": "circ_mv",
            "$free_float_mv": "free_float_mv",
        }
    )
    out["volume_shares"] = out["volume"] * 100.0
    missing_free = out["turnover_free_float"].isna()
    out.loc[missing_free, "turnover_free_float"] = out.loc[missing_free, "$turnover_rate"]
    out["circulating_capital"] = out["circ_mv"].div(out["close"].where(out["close"] > 0))
    out["free_float_capital"] = out["free_float_mv"].div(out["close"].where(out["close"] > 0))
    out["total_capital"] = np.nan
    out["capital_date"] = pd.NaT
    out["announce_date"] = pd.NaT
    out["effective_date"] = pd.NaT
    out["active_circ_mv_proxy"] = out["circ_mv"] * out["turnover_circulating"]
    out["active_free_mv_proxy"] = out["free_float_mv"] * out["turnover_free_float"]
    out["market"] = out["instrument"].map(market_bucket)
    out["basic_tradable"] = (
        out["close"].gt(0)
        & out["volume"].gt(0)
        & out["amount"].gt(0)
        & out["turnover_free_float"].notna()
        & out["free_float_mv"].gt(0)
    )
    return out[columns].sort_values(["date", "instrument"]).reset_index(drop=True)


def load_qmt_capital(
    symbols: list[str], *, batch_size: int = 200, download: bool = False, env_file: str | None = None
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    qmt_symbols = [to_qmt_symbol(symbol) for symbol in symbols]
    with QMTClient(env_file=env_file) as client:
        xtdata = client.xtdata
        for batch in batched(qmt_symbols, max(1, int(batch_size))):
            if download:
                xtdata.download_financial_data2(stock_list=batch, table_list=[CAPITAL_TABLE])
            data = xtdata.get_financial_data(batch, [CAPITAL_TABLE])
            for qmt_symbol, tables in data.items():
                frame = tables.get(CAPITAL_TABLE) if isinstance(tables, dict) else None
                if frame is None or frame.empty:
                    continue
                normalized = normalize_capital_frame(frame, qmt_symbol)
                if not normalized.empty:
                    rows.append(normalized)
    if not rows:
        return empty_capital_frame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["instrument", "capital_date"], keep="last").sort_values(
        ["instrument", "capital_date", "announce_date"]
    ).reset_index(drop=True)


def normalize_capital_frame(frame: pd.DataFrame, qmt_symbol: str) -> pd.DataFrame:
    required = ["m_timetag", "total_capital", "circulating_capital", "freeFloatCapital"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return empty_capital_frame()
    out = frame.copy()
    out["instrument"] = normalize_symbol(qmt_symbol)
    out["capital_date"] = pd.to_datetime(out["m_timetag"].astype(str), format="%Y%m%d", errors="coerce")
    if "m_anntime" in out.columns:
        out["announce_date"] = pd.to_datetime(out["m_anntime"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        out["announce_date"] = pd.NaT
    out["effective_date"] = out[["capital_date", "announce_date"]].max(axis=1)
    out = out.rename(columns={"freeFloatCapital": "free_float_capital"})
    for column in ["total_capital", "circulating_capital", "free_float_capital"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["instrument", "capital_date", "circulating_capital"])
    out = out[out["circulating_capital"] > 0]
    out = out.dropna(subset=["effective_date"])
    columns = [
        "instrument",
        "capital_date",
        "announce_date",
        "effective_date",
        "total_capital",
        "circulating_capital",
        "free_float_capital",
    ]
    return out[columns].sort_values(["instrument", "effective_date", "capital_date", "announce_date"]).reset_index(drop=True)


def build_components(bars: pd.DataFrame, capital: pd.DataFrame) -> pd.DataFrame:
    columns = component_columns()
    if bars.empty or capital.empty:
        return pd.DataFrame(columns=columns)
    merged_parts: list[pd.DataFrame] = []
    for symbol, symbol_bars in bars.groupby("instrument", sort=False):
        symbol_capital = capital[capital["instrument"] == symbol]
        if symbol_capital.empty:
            continue
        merged = pd.merge_asof(
            symbol_bars.sort_values("datetime"),
            symbol_capital.sort_values("effective_date"),
            left_on="datetime",
            right_on="effective_date",
            by="instrument",
            direction="backward",
        )
        merged_parts.append(merged)
    if not merged_parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(merged_parts, ignore_index=True)
    out = out.rename(columns={"datetime": "date", "$close": "close", "$volume": "volume", "$amount": "amount"})
    out["volume_shares"] = out["volume"] * 100.0
    out["turnover_circulating"] = out["volume_shares"].div(out["circulating_capital"].where(out["circulating_capital"] > 0))
    out["turnover_free_float"] = out["volume_shares"].div(out["free_float_capital"].where(out["free_float_capital"] > 0))
    out["circ_mv"] = out["close"] * out["circulating_capital"]
    out["free_float_mv"] = out["close"] * out["free_float_capital"]
    out["active_circ_mv_proxy"] = out["circ_mv"] * out["turnover_circulating"]
    out["active_free_mv_proxy"] = out["free_float_mv"] * out["turnover_free_float"]
    out["market"] = out["instrument"].map(market_bucket)
    out["basic_tradable"] = (
        out["close"].gt(0)
        & out["volume"].gt(0)
        & out["amount"].gt(0)
        & out["circulating_capital"].gt(0)
    )
    return out[columns].sort_values(["date", "instrument"]).reset_index(drop=True)


def aggregate_daily(components: pd.DataFrame) -> pd.DataFrame:
    columns = daily_columns()
    if components.empty:
        return pd.DataFrame(columns=columns)
    frame = components[components["basic_tradable"].fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    all_rows = aggregate_group(frame, "all")
    market_rows = [aggregate_group(group, str(market)) for market, group in frame.groupby("market", sort=False)]
    return pd.concat([all_rows, *market_rows], ignore_index=True).sort_values(["date", "market"]).reset_index(drop=True)


def aggregate_group(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    grouped = frame.groupby("date", as_index=False).agg(
        active_circ_mv_proxy=("active_circ_mv_proxy", "sum"),
        active_free_mv_proxy=("active_free_mv_proxy", "sum"),
        circ_mv=("circ_mv", "sum"),
        free_float_mv=("free_float_mv", "sum"),
        amount=("amount", "sum"),
        volume_shares=("volume_shares", "sum"),
        symbol_count=("instrument", "nunique"),
        turnover_circulating_median=("turnover_circulating", "median"),
        turnover_free_float_median=("turnover_free_float", "median"),
    )
    grouped["market"] = market
    grouped["turnover_circulating_weighted"] = grouped["volume_shares"].div(grouped["circ_mv"].div(grouped["amount"].where(grouped["amount"] > 0)))
    grouped["active_circ_mv_proxy_ret1"] = grouped["active_circ_mv_proxy"].pct_change(fill_method=None)
    grouped["active_circ_mv_proxy_ret2"] = grouped["active_circ_mv_proxy"].pct_change(2, fill_method=None)
    grouped["active_free_mv_proxy_ret1"] = grouped["active_free_mv_proxy"].pct_change(fill_method=None)
    grouped["active_free_mv_proxy_ret2"] = grouped["active_free_mv_proxy"].pct_change(2, fill_method=None)
    return grouped[daily_columns()]


def add_daily_states(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy().sort_values(["market", "date"]).reset_index(drop=True)
    for value_col in ["active_circ_mv_proxy", "active_free_mv_proxy"]:
        grouped = out.groupby("market", sort=False)[value_col]
        out[f"{value_col}_ma10"] = grouped.transform(lambda series: series.rolling(10, min_periods=5).mean())
        out[f"{value_col}_above_ma10"] = out[value_col] >= out[f"{value_col}_ma10"]
        out[f"{value_col}_strong_up_day"] = (out[f"{value_col}_ret1"] >= 0.04) | (out[f"{value_col}_ret2"] >= 0.04)
    return out


def read_existing(path: Path, *, refresh: bool = False) -> pd.DataFrame:
    if refresh or not path.exists():
        return pd.DataFrame(columns=daily_columns())
    frame = pd.read_parquet(path)
    if "date" not in frame.columns or "market" not in frame.columns:
        raise ValueError(f"Existing active-capital file lacks date/market columns: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def merge_existing(existing: pd.DataFrame, partial: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return partial.copy()
    if partial.empty:
        return existing.copy()
    cutoff = pd.Timestamp(partial["date"].min())
    merged = pd.concat([existing[existing["date"] < cutoff], partial], ignore_index=True)
    return merged.drop_duplicates(["date", "market"], keep="last").sort_values(["date", "market"]).reset_index(drop=True)


def build_metadata(**kwargs) -> dict:
    output: Path = kwargs["output"]
    components_output: Path | None = kwargs["components_output"]
    capital: pd.DataFrame = kwargs["capital"]
    components: pd.DataFrame = kwargs["components"]
    daily: pd.DataFrame = kwargs["daily"]
    symbols: list[str] = kwargs["symbols"]
    return {
        "ok": True,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "provider_uri": str(kwargs["provider_uri"]),
        "start": kwargs["start"],
        "end": kwargs["end"],
        "universe": kwargs["universe"],
        "symbols": int(len(symbols)),
        "capital_symbols": int(capital["instrument"].nunique()) if not capital.empty else 0,
        "component_rows": int(len(components)),
        "daily_rows": int(len(daily)),
        "markets": sorted(map(str, daily["market"].dropna().unique())) if not daily.empty else [],
        "download_financial_data2": bool(kwargs["download"]),
        "source": kwargs["source"],
        "capital_fields": ["total_capital", "circulating_capital", "free_float_capital"],
        "bar_fields": BAR_FIELDS,
        "notes": [
            "QMT Capital table is matched backward by effective_date=max(m_timetag, m_anntime) to avoid announcement-date lookahead.",
            "Qlib volume is interpreted as lots; volume_shares = volume * 100.",
            "active_*_mv_proxy equals market value multiplied by the corresponding turnover rate.",
        ],
        "output": str(output),
        "output_sha256": file_sha256(output),
        "components_output": str(components_output) if components_output is not None else None,
        "components_sha256": file_sha256(components_output) if components_output is not None else None,
    }


def batched(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def market_bucket(symbol: str) -> str:
    value = str(symbol).upper()
    if value.startswith("SH688"):
        return "star"
    if value.startswith(("SZ300", "SZ301")):
        return "chinext"
    if value.startswith("BJ"):
        return "beijing"
    return "mainboard"


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def empty_capital_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "instrument",
            "capital_date",
            "announce_date",
            "effective_date",
            "total_capital",
            "circulating_capital",
            "free_float_capital",
        ]
    )


def component_columns() -> list[str]:
    return [
        "date",
        "instrument",
        "market",
        "close",
        "volume",
        "amount",
        "volume_shares",
        "capital_date",
        "announce_date",
        "effective_date",
        "total_capital",
        "circulating_capital",
        "free_float_capital",
        "turnover_circulating",
        "turnover_free_float",
        "circ_mv",
        "free_float_mv",
        "active_circ_mv_proxy",
        "active_free_mv_proxy",
        "basic_tradable",
    ]


def daily_columns() -> list[str]:
    return [
        "date",
        "market",
        "active_circ_mv_proxy",
        "active_free_mv_proxy",
        "circ_mv",
        "free_float_mv",
        "amount",
        "volume_shares",
        "symbol_count",
        "turnover_circulating_median",
        "turnover_free_float_median",
        "turnover_circulating_weighted",
        "active_circ_mv_proxy_ret1",
        "active_circ_mv_proxy_ret2",
        "active_free_mv_proxy_ret1",
        "active_free_mv_proxy_ret2",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
