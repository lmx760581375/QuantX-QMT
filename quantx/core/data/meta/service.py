"""Metadata update service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .sources import AkShareMetaSource, BaoStockMetaSource, MetaSourceError
from .store import MetaStore


@dataclass
class MetaUpdateResult:
    ok: bool
    source: str
    rows: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "source": self.source,
            "rows": self.rows,
            "message": self.message,
        }


class MetaUpdateService:
    def __init__(self, store: Optional[MetaStore] = None):
        self.store = store or MetaStore()

    def import_security_master_csv(
        self,
        path: str | Path,
        source: str = "csv",
        snapshot_date: str | None = None,
    ) -> MetaUpdateResult:
        frame = pd.read_csv(path, dtype=str)
        rows = self.store.upsert_security_master(frame, source=source, snapshot_date=snapshot_date)
        return MetaUpdateResult(ok=True, source=source, rows=rows, message=f"imported security master: {path}")

    def import_industry_csv(
        self,
        path: str | Path,
        source: str = "csv",
        snapshot_date: str | None = None,
    ) -> MetaUpdateResult:
        frame = pd.read_csv(path, dtype=str)
        rows = self.store.upsert_industry_membership(frame, source=source, snapshot_date=snapshot_date)
        return MetaUpdateResult(ok=True, source=source, rows=rows, message=f"imported industry membership: {path}")

    def import_sector_csv(
        self,
        path: str | Path,
        source: str = "csv",
        snapshot_date: str | None = None,
    ) -> MetaUpdateResult:
        frame = pd.read_csv(path, dtype=str)
        rows = self.store.upsert_sector_membership(frame, source=source, snapshot_date=snapshot_date)
        return MetaUpdateResult(ok=True, source=source, rows=rows, message=f"imported sector membership: {path}")

    def update_security_master_baostock(self, date: str | None = None) -> MetaUpdateResult:
        try:
            frame = BaoStockMetaSource().fetch_security_master(date)
            rows = self.store.upsert_security_master(frame, source="baostock")
            return MetaUpdateResult(ok=True, source="baostock", rows=rows, message="updated security master")
        except MetaSourceError as exc:
            return MetaUpdateResult(ok=False, source="baostock", message=str(exc))

    def probe_akshare(self) -> Dict[str, object]:
        source = AkShareMetaSource()
        result: Dict[str, object] = {}
        for name, func in {
            "industry_names": source.fetch_industry_names,
            "concept_names": source.fetch_concept_names,
        }.items():
            try:
                frame = func()
                result[name] = {
                    "ok": True,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                }
            except MetaSourceError as exc:
                result[name] = {"ok": False, "message": str(exc)}
        return result
