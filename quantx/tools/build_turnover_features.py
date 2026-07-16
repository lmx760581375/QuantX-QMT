"""Build Qlib turnover and capital features from QMT Capital data."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.build_active_capital_data import build_components, load_daily_bars, load_qmt_capital
from quantx.tools.run_backtest import load_a_share_symbols


TURNOVER_FEATURES = [
    "turnover_rate",
    "turnover_rate_circulating",
    "turnover_rate_free_float",
    "circ_mv",
    "free_float_mv",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed", help="Source Qlib provider with daily bars.")
    parser.add_argument(
        "--output-provider-uri",
        help="Target provider to write features. Defaults to --provider-uri.",
    )
    parser.add_argument("--metadata-output", help="Metadata JSON path. Defaults to target provider metadata path.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--universe", default="all_a", choices=["all_a", "all_mainboard"])
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbols, e.g. SH600000 SZ000001.")
    parser.add_argument("--limit", type=int, help="Optional symbol limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=200, help="QMT financial read batch size.")
    parser.add_argument("--download", action="store_true", help="Run QMT download_financial_data2 before reading Capital.")
    parser.add_argument("--env-file", help="Optional xqshare .env path.")
    parser.add_argument("--fields", nargs="+", default=TURNOVER_FEATURES, help="Feature fields to write.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    provider_uri = Path(args.provider_uri)
    output_provider_uri = Path(args.output_provider_uri) if args.output_provider_uri else provider_uri
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_provider_uri / "metadata" / "turnover_features.json"
    )
    report = build_turnover_features(
        provider_uri=provider_uri,
        output_provider_uri=output_provider_uri,
        metadata_output=metadata_output,
        start=args.start,
        end=args.end,
        universe=args.universe,
        symbols=args.symbols,
        limit=args.limit,
        batch_size=args.batch_size,
        download=args.download,
        env_file=args.env_file,
        fields=args.fields,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} symbols={report['symbols']} written_symbols={report['written_symbols']} "
            f"rows={report['component_rows']} output={report['output_provider_uri']}"
        )
    return 0 if report["ok"] else 1


def build_turnover_features(
    *,
    provider_uri: Path,
    output_provider_uri: Path,
    metadata_output: Path,
    start: str,
    end: str,
    universe: str = "all_a",
    symbols: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 200,
    download: bool = False,
    env_file: str | None = None,
    fields: list[str] | None = None,
) -> dict:
    requested_fields = normalize_fields(fields or TURNOVER_FEATURES)
    symbol_list = [str(symbol) for symbol in symbols] if symbols else load_a_share_symbols(
        provider_uri, start, end, limit=limit, universe=universe
    )
    if not symbol_list:
        raise ValueError(f"No symbols for universe={universe} start={start} end={end}")

    ensure_provider_scaffold(provider_uri, output_provider_uri)
    bars = load_daily_bars(provider_uri, symbol_list, start, end)
    capital = load_qmt_capital(symbol_list, batch_size=batch_size, download=download, env_file=env_file)
    components = build_components(bars, capital)
    write_report = write_qlib_features(
        output_provider_uri,
        components,
        calendar=QlibBinReader(provider_uri).calendar(),
        fields=requested_fields,
    )
    metadata = {
        "ok": True,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "provider_uri": str(provider_uri),
        "output_provider_uri": str(output_provider_uri),
        "start": start,
        "end": end,
        "universe": universe,
        "symbols": int(len(symbol_list)),
        "bar_symbols": int(bars["instrument"].nunique()) if not bars.empty else 0,
        "capital_symbols": int(capital["instrument"].nunique()) if not capital.empty else 0,
        "component_rows": int(len(components)),
        "fields": requested_fields,
        "download_financial_data2": bool(download),
        "written_symbols": write_report["written_symbols"],
        "written_files": write_report["written_files"],
        "skipped_symbols": write_report["skipped_symbols"],
        "notes": [
            "turnover_rate is aligned to the free-float turnover convention: qlib volume * 100 / QMT Capital.freeFloatCapital.",
            "turnover_rate_circulating = qlib volume * 100 / QMT Capital.circulating_capital",
            "turnover_rate_free_float = qlib volume * 100 / QMT Capital.freeFloatCapital",
            "circ_mv and free_float_mv are close multiplied by the matched historical capital fields.",
        ],
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {**metadata, "metadata_output": str(metadata_output)}


def write_qlib_features(
    provider_uri: Path,
    components: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    fields: list[str],
) -> dict:
    if components.empty:
        return {"written_symbols": 0, "written_files": 0, "skipped_symbols": 0}
    features_dir = provider_uri / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    date_to_idx = {pd.Timestamp(date): idx for idx, date in enumerate(calendar)}
    field_map = {
        "turnover_rate": "turnover_free_float",
        "turnover_rate_circulating": "turnover_circulating",
        "turnover_rate_free_float": "turnover_free_float",
        "circ_mv": "circ_mv",
        "free_float_mv": "free_float_mv",
    }
    written_symbols = 0
    written_files = 0
    skipped_symbols = 0
    for symbol, frame in components.groupby("instrument", sort=False):
        symbol_written = False
        symbol_dir = features_dir / str(symbol).lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        date_indices = pd.to_datetime(frame["date"]).map(date_to_idx).to_numpy(dtype="float64")
        valid_dates = np.isfinite(date_indices)
        if not valid_dates.any():
            skipped_symbols += 1
            continue
        for feature in fields:
            source_column = field_map[feature]
            values = pd.to_numeric(frame[source_column], errors="coerce").to_numpy(dtype=np.float32)
            valid = valid_dates & np.isfinite(values)
            if not valid.any():
                continue
            indices = date_indices[valid].astype(np.int64)
            order = np.argsort(indices)
            sorted_indices = indices[order]
            sorted_values = values[valid][order]
            bin_path = symbol_dir / f"{feature}.day.bin"
            start_idx = int(sorted_indices[0])
            end_idx = int(sorted_indices[-1])
            existing = read_existing_feature(bin_path)
            if existing is not None:
                existing_start, existing_values = existing
                start_idx = min(start_idx, existing_start)
                end_idx = max(end_idx, existing_start + len(existing_values) - 1)
            aligned = np.full(end_idx - start_idx + 1, np.nan, dtype=np.float32)
            if existing is not None:
                existing_start, existing_values = existing
                offset = existing_start - start_idx
                aligned[offset : offset + len(existing_values)] = existing_values
            aligned[sorted_indices - start_idx] = sorted_values
            payload = np.hstack([np.array([start_idx], dtype=np.float32), aligned]).astype("<f")
            payload.tofile(str(bin_path))
            written_files += 1
            symbol_written = True
        if symbol_written:
            written_symbols += 1
        else:
            skipped_symbols += 1
    return {"written_symbols": written_symbols, "written_files": written_files, "skipped_symbols": skipped_symbols}


def read_existing_feature(path: Path) -> tuple[int, np.ndarray] | None:
    if not path.exists():
        return None
    raw = np.fromfile(str(path), dtype="<f4")
    if raw.size <= 1:
        return None
    return int(raw[0]), raw[1:].astype(np.float32, copy=False)


def ensure_provider_scaffold(source: Path, target: Path) -> None:
    (target / "features").mkdir(parents=True, exist_ok=True)
    for dirname in ["calendars", "instruments"]:
        source_dir = source / dirname
        target_dir = target / dirname
        if source_dir.resolve() == target_dir.resolve() or target_dir.exists():
            continue
        shutil.copytree(source_dir, target_dir)


def normalize_fields(fields: list[str]) -> list[str]:
    values = []
    allowed = set(TURNOVER_FEATURES)
    for field in fields:
        name = str(field).lower().lstrip("$")
        if name not in allowed:
            raise ValueError(f"Unsupported turnover feature field: {field}. Allowed: {sorted(allowed)}")
        if name not in values:
            values.append(name)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
