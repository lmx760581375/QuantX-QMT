"""Export audited active-value brick ML scores as a derived backtest asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_SOURCE = Path(".tmp/quantx-research/active-value-frontend-brick-v1/walk_forward_active_brick_scores.parquet")
DEFAULT_OUTPUT = Path("data/derived/active_value_brick_ml/ml_consensus_scores_2025_2026.parquet")
DEFAULT_SCORE_COL = "ml_consensus_score"
DEFAULT_START = "2025-01-02"
DEFAULT_END = "2026-07-15"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-output")
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-signal-days", type=int, default=1)
    parser.add_argument("--turnover-rank-low", type=float)
    parser.add_argument("--turnover-rank-high", type=float)
    parser.add_argument("--free-mv-rank-low", type=float)
    parser.add_argument("--free-mv-rank-high", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_name("metadata.json")
    report = export_score_asset(
        source=Path(args.source),
        output=output,
        metadata_output=metadata_output,
        score_col=str(args.score_col),
        start=args.start,
        end=args.end,
        min_rows=int(args.min_rows),
        min_signal_days=int(args.min_signal_days),
        turnover_rank_low=args.turnover_rank_low,
        turnover_rank_high=args.turnover_rank_high,
        free_mv_rank_low=args.free_mv_rank_low,
        free_mv_rank_high=args.free_mv_rank_high,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} rows={report['rows']} signal_days={report['signal_days']} "
            f"start={report['start']} end={report['end']} output={report['output']}"
        )
    return 0


def export_score_asset(
    *,
    source: Path,
    output: Path,
    metadata_output: Path,
    score_col: str = DEFAULT_SCORE_COL,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_rows: int = 1,
    min_signal_days: int = 1,
    turnover_rank_low: float | None = None,
    turnover_rank_high: float | None = None,
    free_mv_rank_low: float | None = None,
    free_mv_rank_high: float | None = None,
    dry_run: bool = False,
) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Source score parquet does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    source_frame = pd.read_parquet(source)
    filters = build_filters(
        turnover_rank_low=turnover_rank_low,
        turnover_rank_high=turnover_rank_high,
        free_mv_rank_low=free_mv_rank_low,
        free_mv_rank_high=free_mv_rank_high,
    )
    required = {"datetime", "instrument", score_col, *(row["column"] for row in filters)}
    missing = sorted(required - set(source_frame.columns))
    if missing:
        raise ValueError(f"Source score parquet lacks required columns: {missing}")

    asset = normalize_score_frame(source_frame, score_col=score_col, start=start, end=end, filters=filters)
    if len(asset) < int(min_rows):
        raise ValueError(f"Exported score asset has {len(asset)} rows, below min_rows={min_rows}")
    signal_days = int(asset["date"].nunique())
    if signal_days < int(min_signal_days):
        raise ValueError(
            f"Exported score asset has {signal_days} signal days, below min_signal_days={min_signal_days}"
        )

    if not dry_run:
        asset.to_parquet(output, index=False)
    metadata = build_metadata(
        source=source,
        output=output,
        source_frame=source_frame,
        asset=asset,
        score_col=score_col,
        start=start,
        end=end,
        min_rows=min_rows,
        min_signal_days=min_signal_days,
        filters=filters,
        dry_run=dry_run,
    )
    if not dry_run:
        metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "source": str(source),
        "output": str(output),
        "metadata_output": str(metadata_output),
        "rows": int(len(asset)),
        "source_rows": int(len(source_frame)),
        "signal_days": signal_days,
        "instruments": int(asset["instrument"].nunique()),
        "start": str(asset["date"].min().date()),
        "end": str(asset["date"].max().date()),
        "score_col": score_col,
        "filters": filters,
        "sha256": "dry-run" if dry_run else sha256_file(output),
    }


def normalize_score_frame(frame: pd.DataFrame, *, score_col: str, start: str, end: str, filters: list[dict] | None = None) -> pd.DataFrame:
    filter_columns = [row["column"] for row in (filters or [])]
    result = frame[["datetime", "instrument", score_col, *filter_columns]].copy()
    result = result.rename(columns={"datetime": "date"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str).str.strip().str.upper()
    result[score_col] = pd.to_numeric(result[score_col], errors="coerce")
    result = result[
        result["date"].notna()
        & result["instrument"].ne("")
        & result[score_col].notna()
    ].copy()
    for row in filters or []:
        values = pd.to_numeric(result[row["column"]], errors="coerce")
        result = result[values.ge(row["low"]) & values.le(row["high"])].copy()
    result = result[result["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if result.duplicated(["date", "instrument"]).any():
        duplicates = result.loc[result.duplicated(["date", "instrument"], keep=False), ["date", "instrument"]]
        sample = duplicates.head(10).to_dict("records")
        raise ValueError(f"Duplicate score rows for date/instrument: {sample}")
    output_columns = ["date", "instrument", score_col]
    return result.sort_values(["date", score_col, "instrument"], ascending=[True, False, True])[output_columns].reset_index(drop=True)


def build_metadata(
    *,
    source: Path,
    output: Path,
    source_frame: pd.DataFrame,
    asset: pd.DataFrame,
    score_col: str,
    start: str,
    end: str,
    min_rows: int,
    min_signal_days: int,
    filters: list[dict],
    dry_run: bool,
) -> dict:
    score = asset[score_col]
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "asset": "active_value_brick_ml_consensus_scores",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_rows": int(len(source_frame)),
        "source_columns": list(map(str, source_frame.columns)),
        "output": str(output),
        "output_rows": int(len(asset)),
        "output_sha256": "dry-run" if dry_run else sha256_file(output),
        "score_col": score_col,
        "output_columns": ["date", "instrument", score_col],
        "requested_start": start,
        "requested_end": end,
        "start": str(asset["date"].min().date()),
        "end": str(asset["date"].max().date()),
        "signal_days": int(asset["date"].nunique()),
        "instruments": int(asset["instrument"].nunique()),
        "min_rows": int(min_rows),
        "min_signal_days": int(min_signal_days),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
        "filters": filters,
        "dry_run": bool(dry_run),
        "definition": {
            "origin": "Audited walk-forward active-value brick ML scores.",
            "filter": describe_filters(score_col, filters),
            "intended_use": "external_score selector input for formal backtests and daily signal bridge; upstream score panel must be refreshed separately.",
        },
    }


def build_filters(
    *,
    turnover_rank_low: float | None,
    turnover_rank_high: float | None,
    free_mv_rank_low: float | None,
    free_mv_rank_high: float | None,
) -> list[dict]:
    filters: list[dict] = []
    if turnover_rank_low is not None or turnover_rank_high is not None:
        filters.append(make_between_filter("turnover_rate_rank_pct", turnover_rank_low, turnover_rank_high))
    if free_mv_rank_low is not None or free_mv_rank_high is not None:
        filters.append(make_between_filter("free_float_mv_rank_pct", free_mv_rank_low, free_mv_rank_high))
    return filters


def make_between_filter(column: str, low: float | None, high: float | None) -> dict:
    if low is None or high is None:
        raise ValueError(f"Both low and high are required for {column}")
    low_value = float(low)
    high_value = float(high)
    if low_value > high_value:
        raise ValueError(f"Invalid filter for {column}: low > high")
    return {"column": column, "low": low_value, "high": high_value}


def describe_filters(score_col: str, filters: list[dict]) -> str:
    parts = [f"Keep rows where {score_col} is non-null within requested date range"]
    parts.extend(f"{row['column']} between {row['low']:.2f} and {row['high']:.2f}" for row in filters)
    return "; ".join(parts) + "."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
