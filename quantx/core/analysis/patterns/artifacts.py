"""Load completed trades from QuantX run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from quantx.core.analysis.reporting import load_run_artifacts, match_closed_positions


def resolve_run_dir(run: str | Path, runs_root: str | Path = "runs") -> Path:
    """Resolve a run id, run path, or latest alias to a run directory."""
    run_path = Path(run)
    if str(run) == "latest":
        root = Path(runs_root)
        candidates = [path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
        if not candidates:
            raise FileNotFoundError("No run with summary.json found under runs/")
        return max(candidates, key=lambda path: path.stat().st_mtime)
    if run_path.is_dir():
        return run_path
    candidate = Path(runs_root) / str(run)
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Run not found: {run}")


def load_closed_trades(runs: Iterable[str | Path]) -> pd.DataFrame:
    """Load closed trades from one or more QuantX runs."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = resolve_run_dir(run)
        artifacts = load_run_artifacts(run_dir)
        run_id = run_dir.name
        summary = artifacts.get("summary") or {}
        explain = artifacts.get("explain") or {}
        config = explain.get("config") or {}
        strategy_name = summary.get("name") or config.get("name") or run_id
        sell_reason_by_exit = _sell_reason_by_exit(artifacts.get("trades") or [])
        closed_positions = artifacts.get("closed_positions") or []
        if not closed_positions:
            closed_positions = match_closed_positions(artifacts.get("trades") or [])
        for index, trade in enumerate(closed_positions):
            row = dict(trade)
            if "exit_reason" not in row:
                row["exit_reason"] = sell_reason_by_exit.get((row.get("symbol"), row.get("exit_date")), "")
            row["run_id"] = run_id
            row["strategy_name"] = strategy_name
            row["source_artifact"] = str(run_dir)
            row["sample_seq"] = index
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ["entry_price", "exit_price", "return", "net_pnl", "quantity", "holding_days"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["sample_id"] = frame.apply(_sample_id, axis=1)
    return frame.sort_values(["entry_date", "symbol", "run_id"]).reset_index(drop=True)


def _sample_id(row: pd.Series) -> str:
    entry = pd.Timestamp(row["entry_date"]).strftime("%Y%m%d")
    exit_date = pd.Timestamp(row["exit_date"]).strftime("%Y%m%d")
    return f"{row['run_id']}:{row['symbol']}:{entry}:{exit_date}:{int(row.get('sample_seq', 0))}"


def _sell_reason_by_exit(trades: Iterable[Dict[str, Any]]) -> Dict[tuple[Any, Any], str]:
    reasons: Dict[tuple[Any, Any], str] = {}
    for trade in trades:
        if str(trade.get("action") or "").upper() != "SELL":
            continue
        if trade.get("reject_reason"):
            continue
        key = (trade.get("symbol"), trade.get("date"))
        reasons[key] = str(trade.get("reason") or "")
    return reasons
