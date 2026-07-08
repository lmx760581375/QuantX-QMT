"""Approximate opportunity-cost diagnostics for delayed exits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from quantx.core.analysis.reporting import load_run_artifacts

from .artifacts import resolve_run_dir


def build_wait_opportunity_cost(windows: pd.DataFrame, runs: Iterable[str | Path], wait_days: int = 10) -> Dict[str, Any]:
    """Estimate signal and capacity pressure after failed exits.

    This is not a portfolio replay. It answers a narrower question: if failed
    trades were held longer, did the original run see fresh candidates and was
    the account already close to full during that waiting window?
    """
    if windows.empty:
        return {"ok": False, "reason": "empty windows", "by_exit_reason": []}
    required = {"run_id", "sample_id", "exit_date", "return"}
    if not required.issubset(windows.columns):
        return {"ok": False, "reason": "missing required window columns", "by_exit_reason": []}

    failed = windows[pd.to_numeric(windows["return"], errors="coerce") <= 0.0].copy()
    if failed.empty:
        return {"ok": False, "reason": "no failed trades", "by_exit_reason": []}

    context_by_run = _load_run_contexts(runs)
    if not context_by_run:
        return {"ok": False, "reason": "missing run opportunity artifacts", "by_exit_reason": []}

    rows: List[Dict[str, Any]] = []
    for _, trade in failed.iterrows():
        run_id = str(trade.get("run_id") or "")
        context = context_by_run.get(run_id)
        if not context:
            continue
        exit_date = pd.Timestamp(trade["exit_date"])
        start = exit_date + pd.Timedelta(days=1)
        end = exit_date + pd.Timedelta(days=max(wait_days * 2, wait_days + 5))
        candidate_window = _slice_by_date(context["candidates"], start, end).head(wait_days)
        nav_window = _slice_by_date(context["nav"], start, end).head(wait_days)
        if candidate_window.empty and nav_window.empty:
            continue
        selected_counts = _numeric_series(candidate_window, "selected_count")
        raw_counts = _numeric_series(candidate_window, "raw_candidate_count")
        position_counts = _numeric_series(nav_window, "position_count")
        rows.append({
            "sample_id": trade.get("sample_id"),
            "run_id": run_id,
            "exit_reason": str(trade.get("exit_reason") or ""),
            "return": float(trade.get("return") or 0.0),
            "wait_days_observed": int(max(len(candidate_window), len(nav_window))),
            "candidate_days": int((selected_counts > 0).sum()),
            "raw_candidate_days": int((raw_counts > 0).sum()),
            "selected_candidate_count": int(selected_counts.sum()),
            "raw_candidate_count": int(raw_counts.sum()),
            "avg_position_count": _mean(nav_window, "position_count"),
            "max_position_count": _max(nav_window, "position_count"),
            "near_full_days": int((position_counts >= context["near_full_threshold"]).sum()),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"ok": False, "reason": "no matched failed trades", "by_exit_reason": []}
    by_reason = [_opportunity_row(group, str(reason), wait_days) for reason, group in frame.groupby(frame["exit_reason"].fillna(""), dropna=False)]
    by_reason.sort(key=lambda row: int(row.get("sample_count") or 0), reverse=True)
    total = _opportunity_row(frame, "all_failed", wait_days)
    total.update({
        "ok": True,
        "mode": "post_exit_opportunity_cost_proxy",
        "wait_days": int(wait_days),
        "note": "Proxy only: counts original-run candidates and position_count after failed exits; it does not replay cash reuse, ranking, or fills.",
        "samples": frame.to_dict(orient="records"),
        "by_exit_reason": by_reason,
    })
    return total


def _load_run_contexts(runs: Iterable[str | Path]) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        run_dir = resolve_run_dir(run)
        artifacts = load_run_artifacts(run_dir)
        candidates = pd.DataFrame(artifacts.get("daily_selection_candidates") or artifacts.get("selection_candidates") or [])
        nav = pd.DataFrame(artifacts.get("daily_nav") or [])
        if candidates.empty and nav.empty:
            continue
        if "date" in candidates:
            candidates["date"] = pd.to_datetime(candidates["date"])
        if "date" in nav:
            nav["date"] = pd.to_datetime(nav["date"])
        max_position = _numeric_series(nav, "position_count").max() if not nav.empty else 0
        contexts[run_dir.name] = {
            "candidates": candidates.sort_values("date") if "date" in candidates else pd.DataFrame(),
            "nav": nav.sort_values("date") if "date" in nav else pd.DataFrame(),
            "near_full_threshold": max(float(max_position or 0) - 1.0, 1.0),
        }
    return contexts


def _slice_by_date(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "date" not in frame:
        return pd.DataFrame()
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()


def _opportunity_row(frame: pd.DataFrame, group: str, wait_days: int) -> Dict[str, Any]:
    observed = pd.to_numeric(frame["wait_days_observed"], errors="coerce")
    observed = observed.where(observed > 0)
    candidate_days = pd.to_numeric(frame["candidate_days"], errors="coerce").fillna(0)
    near_full_days = pd.to_numeric(frame["near_full_days"], errors="coerce").fillna(0)
    return {
        "group": group,
        "sample_count": int(len(frame)),
        "avg_return": _mean(frame, "return"),
        "any_candidate_rate": float((candidate_days > 0).mean()) if not frame.empty else None,
        "avg_candidate_days": _mean(frame, "candidate_days"),
        "avg_selected_candidate_count": _mean(frame, "selected_candidate_count"),
        "avg_raw_candidate_count": _mean(frame, "raw_candidate_count"),
        "avg_position_count": _mean(frame, "avg_position_count"),
        "avg_max_position_count": _mean(frame, "max_position_count"),
        "near_full_window_rate": float((near_full_days > 0).mean()) if not frame.empty else None,
        "avg_near_full_days": _mean(frame, "near_full_days"),
        "candidate_day_density": float((candidate_days / observed).mean()) if observed.notna().any() else None,
        "near_full_day_density": float((near_full_days / observed).mean()) if observed.notna().any() else None,
        "wait_days": int(wait_days),
    }


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").max()
    return float(value) if pd.notna(value) else None


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
