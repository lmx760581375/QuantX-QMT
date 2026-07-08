"""Validate fast-exit contexts on recorded strategy candidate pools.

This tool expands evidence beyond actually filled positions by treating recorded
daily selection candidates as virtual trades. It is a diagnostic, not a
portfolio backtest: it ignores cash competition, fills, position sizing, and
ranking interactions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quantx.core.analysis.patterns.io import json_safe
from quantx.core.analysis.patterns.market_data import DailyBarLoader


@dataclass(frozen=True)
class FastExitRule:
    min_holding_days: int = 18
    max_peak_pnl: float = 0.02
    min_trough_pnl: float = -0.08
    first_3d_return: float = -0.01
    first_10d_return: float = -0.04

    @property
    def rule_id(self) -> str:
        return f"d{self.min_holding_days}_p{int(round(self.max_peak_pnl * 100))}_t{int(round(abs(self.min_trough_pnl) * 100))}"

    @property
    def yaml_when(self) -> str:
        return (
            f"holding_days > {self.min_holding_days} and pnl_pct < 0 and "
            f"hold_first_3d_return < {self.first_3d_return:g} and "
            f"hold_first_10d_return < {self.first_10d_return:g} and "
            f"peak_pnl_pct < {self.max_peak_pnl:g} and trough_pnl_pct < {self.min_trough_pnl:g}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = Path(args.run)
    output_dir = Path(args.output_dir) / (args.analysis_id or datetime.now().strftime("%Y%m%d_%H%M%S_candidate_pool_fast_exit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rule = FastExitRule(
        min_holding_days=args.min_holding_days,
        max_peak_pnl=args.max_peak_pnl,
        min_trough_pnl=args.min_trough_pnl,
        first_3d_return=args.first_3d_return,
        first_10d_return=args.first_10d_return,
    )
    candidates = _load_candidate_rows(run_dir, source=args.source, max_rows=args.max_rows)
    loader = DailyBarLoader(raw_data_dir=args.raw_data_dir, provider_uri=args.provider_uri, prefer_qlib=not args.prefer_csv)
    rows = validate_candidates(candidates, loader, rule, max_holding_days=args.max_holding_days)
    frame = pd.DataFrame(rows)
    samples_path = output_dir / "candidate_fast_exit_samples.csv"
    frame.to_csv(samples_path, index=False)
    summary = _summary(frame, candidates, rule, run_dir, args.source)
    summary["samples_path"] = str(samples_path)
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(json_safe({"ok": True, "output_dir": str(output_dir), "summary": summary}), ensure_ascii=False, indent=2))
    else:
        print(f"output_dir={output_dir}")
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


def validate_candidates(
    candidates: list[dict[str, Any]],
    loader: DailyBarLoader,
    rule: FastExitRule,
    *,
    max_holding_days: int = 45,
) -> list[dict[str, Any]]:
    symbols = sorted({str(row["symbol"]) for row in candidates if row.get("symbol")})
    if candidates:
        dates = pd.to_datetime([row["signal_date"] for row in candidates])
        loader.preload_symbols(symbols, start=(dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), end=(dates.max() + pd.Timedelta(days=max_holding_days * 3)).strftime("%Y-%m-%d"))
    rows: list[dict[str, Any]] = []
    for row in candidates:
        result = _simulate_candidate(row, loader, rule, max_holding_days=max_holding_days)
        if result:
            rows.append(result)
    return rows


def _simulate_candidate(
    candidate: dict[str, Any],
    loader: DailyBarLoader,
    rule: FastExitRule,
    *,
    max_holding_days: int,
) -> dict[str, Any] | None:
    symbol = str(candidate.get("symbol") or "")
    signal_date = candidate.get("signal_date") or candidate.get("date")
    if not symbol or not signal_date:
        return None
    bars = loader.load_symbol(symbol)
    if bars.empty:
        return None
    bars = bars.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(bars["date"]))
    signal_ts = pd.Timestamp(signal_date)
    entry_idx = int(dates.searchsorted(signal_ts, side="right"))
    if entry_idx >= len(bars):
        return None
    end_idx = min(len(bars) - 1, entry_idx + max_holding_days)
    path = bars.iloc[entry_idx : end_idx + 1].copy().reset_index(drop=True)
    if len(path) < 2:
        return None
    entry_price = float(path.iloc[0]["close"])
    if entry_price <= 0:
        return None

    first_3d = _close_return(path, entry_price, 3)
    first_10d = _close_return(path, entry_price, 10)
    peak_pnl = float(path.iloc[0]["high"] / entry_price - 1)
    trough_pnl = float(path.iloc[0]["low"] / entry_price - 1)
    baseline_exit = _exit_result(path.iloc[-1], len(path) - 1, "max_holding", entry_price)
    fast_exit = baseline_exit

    for offset, bar in list(path.iterrows())[1:]:
        close_pnl = float(bar["close"] / entry_price - 1)
        high_pnl = float(bar["high"] / entry_price - 1)
        low_pnl = float(bar["low"] / entry_price - 1)
        peak_pnl = max(peak_pnl, high_pnl)
        trough_pnl = min(trough_pnl, low_pnl)

        hard_exit = _hard_exit_reason(offset, close_pnl, peak_pnl)
        if hard_exit:
            baseline_exit = _exit_result(bar, offset, hard_exit, entry_price)
            fast_exit = baseline_exit
            break
        if _fast_exit_trigger(offset, close_pnl, peak_pnl, trough_pnl, first_3d, first_10d, rule):
            fast_exit = _exit_result(bar, offset, "fast_exit", entry_price)
            baseline_exit = _continue_baseline(path, offset + 1, entry_price, peak_pnl, max_holding_days)
            break
        soft_exit = _soft_exit_reason(offset, close_pnl, peak_pnl)
        if soft_exit:
            baseline_exit = _exit_result(bar, offset, soft_exit, entry_price)
            fast_exit = baseline_exit
            break

    return {
        "symbol": symbol,
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "entry_date": str(pd.Timestamp(path.iloc[0]["date"]).date()),
        "entry_price": entry_price,
        "score": candidate.get("score"),
        "baseline_exit_date": baseline_exit["exit_date"],
        "baseline_exit_reason": baseline_exit["exit_reason"],
        "baseline_holding_days": baseline_exit["holding_days"],
        "baseline_return": baseline_exit["return"],
        "fast_exit_date": fast_exit["exit_date"],
        "fast_exit_reason": fast_exit["exit_reason"],
        "fast_holding_days": fast_exit["holding_days"],
        "fast_return": fast_exit["return"],
        "fast_triggered": fast_exit["exit_reason"] == "fast_exit",
        "delta_return": fast_exit["return"] - baseline_exit["return"],
        "first_3d_return": first_3d,
        "first_10d_return": first_10d,
        "peak_pnl_pct": peak_pnl,
        "trough_pnl_pct": trough_pnl,
    }


def _continue_baseline(path: pd.DataFrame, start_offset: int, entry_price: float, peak_pnl: float, max_holding_days: int) -> dict[str, Any]:
    fallback = _exit_result(path.iloc[-1], len(path) - 1, "max_holding", entry_price)
    for offset, bar in list(path.iterrows())[start_offset:]:
        close_pnl = float(bar["close"] / entry_price - 1)
        peak_pnl = max(peak_pnl, float(bar["high"] / entry_price - 1))
        reason = _hard_exit_reason(offset, close_pnl, peak_pnl) or _soft_exit_reason(offset, close_pnl, peak_pnl)
        if reason:
            return _exit_result(bar, offset, reason, entry_price)
        if offset >= max_holding_days:
            return _exit_result(bar, offset, "max_holding", entry_price)
    return fallback


def _hard_exit_reason(offset: int, close_pnl: float, peak_pnl: float) -> str | None:
    if close_pnl > 0.210:
        return "take_profit_210permil"
    if close_pnl < -0.089:
        return "stop_loss_89permil"
    return None


def _soft_exit_reason(offset: int, close_pnl: float, peak_pnl: float) -> str | None:
    drawdown_from_peak = (1 + close_pnl) / (1 + peak_pnl) - 1 if peak_pnl > -1 else 0.0
    if offset >= 5 and peak_pnl > 0.15 and drawdown_from_peak < -0.10:
        return "trail_peak15_dd10"
    if offset > 25:
        return "time_stop_25d"
    return None


def _fast_exit_trigger(
    offset: int,
    close_pnl: float,
    peak_pnl: float,
    trough_pnl: float,
    first_3d: float,
    first_10d: float,
    rule: FastExitRule,
) -> bool:
    return bool(
        offset > rule.min_holding_days
        and close_pnl < 0
        and first_3d < rule.first_3d_return
        and first_10d < rule.first_10d_return
        and peak_pnl < rule.max_peak_pnl
        and trough_pnl < rule.min_trough_pnl
    )


def _exit_result(bar: pd.Series, offset: int, reason: str, entry_price: float) -> dict[str, Any]:
    close = float(bar["close"])
    return {
        "exit_date": str(pd.Timestamp(bar["date"]).date()),
        "exit_reason": reason,
        "holding_days": int(offset),
        "return": close / entry_price - 1,
    }


def _close_return(path: pd.DataFrame, entry_price: float, days: int) -> float:
    idx = min(days - 1, len(path) - 1)
    return float(path.iloc[idx]["close"] / entry_price - 1)


def _load_candidate_rows(run_dir: Path, *, source: str, max_rows: int | None) -> list[dict[str, Any]]:
    data = json.loads((run_dir / "daily_selection_candidates.json").read_text(encoding="utf-8"))
    key = "raw_candidates" if source == "raw" else "selected_candidates"
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in data:
        signal_date = item.get("date") or item.get("signal_date")
        for candidate in item.get(key) or []:
            symbol = candidate.get("symbol")
            if not symbol or not signal_date:
                continue
            dedupe_key = (str(symbol), str(signal_date))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append({"symbol": symbol, "signal_date": signal_date, "score": candidate.get("score")})
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def _summary(frame: pd.DataFrame, candidates: list[dict[str, Any]], rule: FastExitRule, run_dir: Path, source: str) -> dict[str, Any]:
    triggered = frame[frame["fast_triggered"].astype(bool)] if not frame.empty else pd.DataFrame()
    return {
        "run_dir": str(run_dir),
        "candidate_source": source,
        "candidate_rows": len(candidates),
        "simulated_rows": int(len(frame)),
        "unique_symbols": int(frame["symbol"].nunique()) if "symbol" in frame else 0,
        "rule_id": rule.rule_id,
        "yaml_when": rule.yaml_when,
        "baseline_avg_return": _mean(frame, "baseline_return"),
        "fast_avg_return": _mean(frame, "fast_return"),
        "avg_delta_return": _mean(frame, "delta_return"),
        "fast_trigger_count": int(len(triggered)),
        "fast_trigger_rate": float(len(triggered) / len(frame)) if len(frame) else None,
        "trigger_avg_baseline_return": _mean(triggered, "baseline_return"),
        "trigger_avg_fast_return": _mean(triggered, "fast_return"),
        "trigger_avg_delta_return": _mean(triggered, "delta_return"),
        "trigger_improved_count": int((triggered["delta_return"] > 0).sum()) if not triggered.empty else 0,
        "trigger_worsened_count": int((triggered["delta_return"] < 0).sum()) if not triggered.empty else 0,
        "trigger_improved_rate": float((triggered["delta_return"] > 0).mean()) if not triggered.empty else None,
        "trigger_unique_symbols": int(triggered["symbol"].nunique()) if "symbol" in triggered else 0,
        "trigger_exit_reasons": triggered["baseline_exit_reason"].value_counts().to_dict() if "baseline_exit_reason" in triggered else {},
    }


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing daily_selection_candidates.json")
    parser.add_argument("--output-dir", default="artifacts/candidate_pool_validation")
    parser.add_argument("--analysis-id")
    parser.add_argument("--source", choices=["selected", "raw"], default="selected")
    parser.add_argument("--raw-data-dir", default="data/raw/baostock")
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--prefer-csv", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-holding-days", type=int, default=45)
    parser.add_argument("--min-holding-days", type=int, default=18)
    parser.add_argument("--max-peak-pnl", type=float, default=0.02)
    parser.add_argument("--min-trough-pnl", type=float, default=-0.08)
    parser.add_argument("--first-3d-return", type=float, default=-0.01)
    parser.add_argument("--first-10d-return", type=float, default=-0.04)
    parser.add_argument("--json", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
