"""Adjustment-aware incremental data update for BaoStock CSV and Qlib bins."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd

from .baostock_client import BaoStockClient
from .converter import BaostockToQlibConverter
from .repository import LocalDataRepository

logger = logging.getLogger(__name__)

PRICE_COMPARE_COLUMNS = ["open", "high", "low", "close", "preclose"]


@dataclass
class SymbolUpdateResult:
    symbol: str
    status: str
    last_date: str | None = None
    overlap_start: str | None = None
    fetched_rows: int = 0
    rows_added: int = 0
    adjustment_changed: bool = False
    mismatch_columns: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IncrementalUpdateReport:
    ok: bool
    total_symbols: int
    updated_symbols: List[str] = field(default_factory=list)
    full_refresh_symbols: List[str] = field(default_factory=list)
    unchanged_symbols: List[str] = field(default_factory=list)
    skipped_symbols: List[str] = field(default_factory=list)
    failed_symbols: List[str] = field(default_factory=list)
    symbol_results: List[SymbolUpdateResult] = field(default_factory=list)
    converted: bool = False
    dry_run: bool = False
    request_count: int = 0
    max_requests: int | None = None
    budget_exhausted: bool = False
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbol_results"] = [row.to_dict() for row in self.symbol_results]
        return data


class AdjustmentAwareIncrementalUpdater:
    """Incrementally update BaoStock data while guarding forward-adjusted history.

    The updater fetches an overlap window before each stock's local last date.
    If adjusted prices in that overlap no longer match local CSV values, it
    refreshes the stock from its first local date and rewrites its Qlib bin
    files. Otherwise it only merges the overlap/new rows and rewrites that
    stock's Qlib bin files.
    """

    def __init__(
        self,
        repository: LocalDataRepository,
        converter: BaostockToQlibConverter,
        client_factory: Callable[[], Any] | None = None,
        overlap_days: int = 40,
        tolerance: float = 1e-4,
        full_refresh_start: str = "2010-01-01",
        max_requests: int | None = None,
    ):
        self.repository = repository
        self.converter = converter
        self.client_factory = client_factory or BaoStockClient
        self.overlap_days = int(overlap_days)
        self.tolerance = float(tolerance)
        self.full_refresh_start = full_refresh_start
        self.max_requests = int(max_requests) if max_requests is not None else None
        self._request_count = 0

    def update(
        self,
        symbols: Optional[Iterable[str]] = None,
        end: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        progress_callback: Callable[[IncrementalUpdateReport], None] | None = None,
    ) -> IncrementalUpdateReport:
        end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
        self._request_count = 0
        symbol_list = list(symbols) if symbols is not None else self.repository.get_stock_codes()
        if limit is not None:
            symbol_list = symbol_list[: int(limit)]

        report = IncrementalUpdateReport(
            ok=True,
            total_symbols=len(symbol_list),
            dry_run=dry_run,
            max_requests=self.max_requests,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

        if not symbol_list:
            report.ok = False
            report.finished_at = datetime.now().isoformat(timespec="seconds")
            return report

        try:
            with self.client_factory() as client:
                for idx, symbol in enumerate(symbol_list):
                    try:
                        result = self._update_symbol(client, symbol, end=end, dry_run=dry_run)
                    except RequestBudgetExceeded as exc:
                        report.budget_exhausted = True
                        result = SymbolUpdateResult(symbol=symbol, status="skipped_budget", message=str(exc))
                        self._append_result(report, result)
                        for remaining in symbol_list[idx + 1 :]:
                            self._append_result(
                                report,
                                SymbolUpdateResult(
                                    symbol=remaining,
                                    status="skipped_budget",
                                    message="request budget exhausted before symbol update",
                                ),
                            )
                        break
                    except Exception as exc:  # pragma: no cover - keeps batch update alive for unexpected provider errors.
                        logger.warning("Failed to update %s: %s", symbol, exc)
                        result = SymbolUpdateResult(symbol=symbol, status="failed", message=str(exc))
                    self._append_result(report, result)
                    if progress_callback is not None:
                        report.request_count = self._request_count
                        progress_callback(report)
        except Exception as exc:
            logger.warning("Failed to initialize data update client: %s", exc)
            for symbol in symbol_list:
                self._append_result(
                    report,
                    SymbolUpdateResult(symbol=symbol, status="failed", message=f"client error: {exc}"),
                )

        report.request_count = self._request_count
        if report.updated_symbols and not dry_run:
            self.converter.convert_incremental(report.updated_symbols)
            report.converted = True
        report.ok = not report.failed_symbols and not report.budget_exhausted
        report.finished_at = datetime.now().isoformat(timespec="seconds")
        return report

    def _append_result(self, report: IncrementalUpdateReport, result: SymbolUpdateResult) -> None:
        report.symbol_results.append(result)
        if result.status in {"incremental", "full_refresh"}:
            report.updated_symbols.append(result.symbol)
        if result.status == "full_refresh":
            report.full_refresh_symbols.append(result.symbol)
        if result.status == "unchanged":
            report.unchanged_symbols.append(result.symbol)
        if result.status == "skipped_budget":
            report.skipped_symbols.append(result.symbol)
        if result.status == "failed":
            report.failed_symbols.append(result.symbol)

    def _update_symbol(self, client, symbol: str, end: str, dry_run: bool) -> SymbolUpdateResult:
        local = self.repository.load_symbol(symbol)
        if local.empty:
            fetched = self._fetch(client, symbol, self.full_refresh_start, end)
            if fetched.empty:
                return SymbolUpdateResult(symbol=symbol, status="failed", message="no data fetched for new symbol")
            if not dry_run:
                self.repository.replace_symbol(symbol, fetched)
            return SymbolUpdateResult(
                symbol=symbol,
                status="full_refresh",
                last_date=None,
                overlap_start=self.full_refresh_start,
                fetched_rows=len(fetched),
                rows_added=len(fetched),
                adjustment_changed=False,
                message="new symbol full refresh",
            )

        local = _normalize_frame(local)
        first_date = pd.Timestamp(local["date"].min()).strftime("%Y-%m-%d")
        last_date = pd.Timestamp(local["date"].max()).strftime("%Y-%m-%d")
        overlap_start = _overlap_start(local, self.overlap_days)
        if pd.Timestamp(end) < pd.Timestamp(overlap_start):
            return SymbolUpdateResult(
                symbol=symbol,
                status="unchanged",
                last_date=last_date,
                overlap_start=overlap_start,
                fetched_rows=0,
                rows_added=0,
                message="end date is earlier than local overlap; no update needed",
            )
        fetched = self._fetch(client, symbol, overlap_start, end)
        if fetched.empty:
            return SymbolUpdateResult(
                symbol=symbol,
                status="failed",
                last_date=last_date,
                overlap_start=overlap_start,
                message="no data fetched for overlap window",
            )

        mismatch_columns = _mismatch_columns(local, fetched, overlap_start, last_date, self.tolerance)
        if mismatch_columns:
            refresh_start = first_date or self.full_refresh_start
            full = self._fetch(client, symbol, refresh_start, end)
            if full.empty:
                return SymbolUpdateResult(
                    symbol=symbol,
                    status="failed",
                    last_date=last_date,
                    overlap_start=overlap_start,
                    fetched_rows=len(fetched),
                    adjustment_changed=True,
                    mismatch_columns=mismatch_columns,
                    message="adjustment mismatch found but full refresh fetched no data",
                )
            if not dry_run:
                self.repository.replace_symbol(symbol, full)
            return SymbolUpdateResult(
                symbol=symbol,
                status="full_refresh",
                last_date=last_date,
                overlap_start=overlap_start,
                fetched_rows=len(full),
                rows_added=max(0, len(full) - len(local)),
                adjustment_changed=True,
                mismatch_columns=mismatch_columns,
                message="overlap adjusted prices changed; full symbol refreshed",
            )

        fetched = _normalize_frame(fetched)
        new_rows = fetched[pd.to_datetime(fetched["date"]) > pd.Timestamp(last_date)]
        if new_rows.empty:
            return SymbolUpdateResult(
                symbol=symbol,
                status="unchanged",
                last_date=last_date,
                overlap_start=overlap_start,
                fetched_rows=len(fetched),
                rows_added=0,
                message="no new rows and overlap aligned",
            )
        if not dry_run:
            self.repository.save_symbol(symbol, fetched)
        return SymbolUpdateResult(
            symbol=symbol,
            status="incremental",
            last_date=last_date,
            overlap_start=overlap_start,
            fetched_rows=len(fetched),
            rows_added=len(new_rows),
            adjustment_changed=False,
            message="overlap aligned; merged new rows",
        )

    def _fetch(self, client, symbol: str, start: str, end: str) -> pd.DataFrame:
        self._consume_request(symbol, start, end)
        baostock_symbol = BaoStockClient.to_baostock_format(symbol)
        frame = client.query_history_k_data_with_retry(baostock_symbol, start, end)
        return _normalize_frame(frame)

    def _consume_request(self, symbol: str, start: str, end: str) -> None:
        if self.max_requests is not None and self._request_count >= self.max_requests:
            raise RequestBudgetExceeded(
                f"request budget exhausted at {self._request_count}/{self.max_requests} before {symbol} {start}~{end}"
            )
        self._request_count += 1


class RequestBudgetExceeded(RuntimeError):
    """Raised when the configured BaoStock logical request budget is exhausted."""


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "date" not in out.columns:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in PRICE_COMPARE_COLUMNS + ["volume", "amount", "turnover", "pct_chg", "is_st"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def _overlap_start(local: pd.DataFrame, overlap_days: int) -> str:
    dates = pd.to_datetime(local["date"]).sort_values().drop_duplicates()
    if dates.empty:
        return "2010-01-01"
    idx = max(0, len(dates) - max(1, int(overlap_days)))
    return pd.Timestamp(dates.iloc[idx]).strftime("%Y-%m-%d")


def _mismatch_columns(
    local: pd.DataFrame,
    fetched: pd.DataFrame,
    overlap_start: str,
    last_date: str,
    tolerance: float,
) -> List[str]:
    local_norm = _normalize_frame(local)
    fetched_norm = _normalize_frame(fetched)
    local_slice = local_norm[
        (pd.to_datetime(local_norm["date"]) >= pd.Timestamp(overlap_start))
        & (pd.to_datetime(local_norm["date"]) <= pd.Timestamp(last_date))
    ]
    fetched_slice = fetched_norm[pd.to_datetime(fetched_norm["date"]) <= pd.Timestamp(last_date)]
    if local_slice.empty or fetched_slice.empty:
        return []
    merged = local_slice.merge(fetched_slice, on="date", suffixes=("_local", "_new"))
    if merged.empty:
        return []
    mismatches: List[str] = []
    for col in PRICE_COMPARE_COLUMNS:
        left = f"{col}_local"
        right = f"{col}_new"
        if left not in merged.columns or right not in merged.columns:
            continue
        diff = (pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")).abs()
        if bool((diff > tolerance).fillna(False).any()):
            mismatches.append(col)
    return mismatches
