#!/usr/bin/env python3
"""Audit qlib provider coverage against strategy lookback windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantx.core.engine.context import compute_load_start


DEFAULT_CONFIGS = [
    "runs/baostock_qmt_align_configs/2020_champion_baostock_partial63.yaml",
    "runs/baostock_qmt_align_configs/2020_champion_baostock_partial63_2010.yaml",
    "runs/baostock_qmt_align_configs/2020_champion_qmt_fill_partial63.yaml",
    "runs/baostock_qmt_align_configs/2016_scale12_baostock_partial63.yaml",
    "runs/baostock_qmt_align_configs/2016_scale12_baostock_partial63_2010.yaml",
    "runs/baostock_qmt_align_configs/2016_scale12_qmt_fill_partial63.yaml",
]

DEFAULT_RUNS = [
    "runs/align63_2020_baostock",
    "runs/align63_2020_baostock_2010",
    "runs/align63_2020_qmt_fill",
    "runs/align63_2016_baostock",
    "runs/align63_2016_baostock_2010",
    "runs/align63_2016_qmt_fill",
]


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    path = provider_uri / "calendars" / "day.txt"
    if not path.exists():
        return pd.DatetimeIndex([])
    frame = pd.read_csv(path, header=None, names=["date"])
    return pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce").dropna()).sort_values()


def _instrument_count(provider_uri: Path) -> int:
    path = provider_uri / "instruments" / "all.txt"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def _resolve_end(end: str, reference_calendar: pd.DatetimeIndex) -> str | None:
    if not end:
        return None
    if str(end).lower() == "latest":
        if reference_calendar.empty:
            return None
        return reference_calendar[-1].strftime("%Y-%m-%d")
    return str(pd.Timestamp(end).strftime("%Y-%m-%d"))


def _missing_warmup_trading_days(reference_calendar: pd.DatetimeIndex, load_start: str | None, provider_min: str | None) -> int:
    if reference_calendar.empty or load_start is None or provider_min is None:
        return 0
    required = pd.Timestamp(load_start)
    provider_start = pd.Timestamp(provider_min)
    if provider_start <= required:
        return 0
    mask = (reference_calendar >= required) & (reference_calendar < provider_start)
    return int(mask.sum())


def _calendar_gap_days(load_start: str | None, provider_min: str | None) -> int:
    if load_start is None or provider_min is None:
        return 0
    return max(0, int((pd.Timestamp(provider_min) - pd.Timestamp(load_start)).days))


def audit_config(path: Path, reference_calendar: pd.DatetimeIndex) -> dict[str, Any]:
    config = _load_yaml(path)
    data_cfg = config.get("data") or {}
    provider_uri = Path(str(data_cfg.get("provider_uri", "")))
    calendar = _read_calendar(provider_uri)
    provider_min = calendar[0].strftime("%Y-%m-%d") if len(calendar) else None
    provider_max = calendar[-1].strftime("%Y-%m-%d") if len(calendar) else None
    start = str(data_cfg.get("start", ""))
    end = str(data_cfg.get("end", ""))
    resolved_end = _resolve_end(end, reference_calendar)
    look_back_days = int(data_cfg.get("look_back_days", 0) or 0)
    load_start = compute_load_start(start, look_back_days)
    missing = _missing_warmup_trading_days(reference_calendar, load_start, provider_min)
    gap_days = _calendar_gap_days(load_start, provider_min)
    has_end = provider_max is not None and resolved_end is not None and pd.Timestamp(provider_max) >= pd.Timestamp(resolved_end)
    has_warmup = gap_days <= 10 and provider_min is not None
    return {
        "config": str(path),
        "provider_uri": str(provider_uri),
        "start": start,
        "end": end,
        "resolved_end": resolved_end,
        "look_back_days": look_back_days,
        "required_load_start": load_start,
        "provider_calendar_start": provider_min,
        "provider_calendar_end": provider_max,
        "provider_calendar_days": int(len(calendar)),
        "provider_instruments": _instrument_count(provider_uri),
        "calendar_gap_days": gap_days,
        "missing_warmup_trading_days": missing,
        "coverage_ok": bool(has_warmup and has_end),
    }


def summarize_run(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    metrics_path = path / "metrics.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "run": str(path),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "symbols": summary.get("symbols"),
        "final_value": summary.get("final_value"),
        "total_return": summary.get("total_return"),
        "trades": summary.get("trades"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "avg_closed_return": metrics.get("avg_closed_return"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-provider", default="data/qlib_data_qmt_fill_mainboard_compare")
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--output", default="runs/baostock_qmt_alignment_audit.json")
    args = parser.parse_args(argv)

    reference_provider = Path(args.reference_provider)
    reference_calendar = _read_calendar(reference_provider)
    configs = [Path(item) for item in (args.configs or DEFAULT_CONFIGS) if Path(item).exists()]
    runs = [Path(item) for item in (args.runs or DEFAULT_RUNS) if Path(item).exists()]

    report = {
        "reference_provider": str(reference_provider),
        "reference_calendar_start": reference_calendar[0].strftime("%Y-%m-%d") if len(reference_calendar) else None,
        "reference_calendar_end": reference_calendar[-1].strftime("%Y-%m-%d") if len(reference_calendar) else None,
        "configs": [audit_config(path, reference_calendar) for path in configs],
        "runs": [item for item in (summarize_run(path) for path in runs) if item is not None],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bad = [item for item in report["configs"] if not item["coverage_ok"]]
    print(f"Wrote {output}")
    print(f"Configs audited: {len(report['configs'])}, coverage issues: {len(bad)}")
    for item in bad:
        print(
            "COVERAGE_ISSUE",
            item["config"],
            "provider_start=",
            item["provider_calendar_start"],
            "required_load_start=",
            item["required_load_start"],
            "calendar_gap_days=",
            item["calendar_gap_days"],
            "missing_warmup_days=",
            item["missing_warmup_trading_days"],
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
