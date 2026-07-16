"""Metadata source adapters."""

from __future__ import annotations

import pandas as pd

from quantx.core.data.baostock_client import BaoStockClient
from quantx.core.data.meta.store import normalize_symbol
from quantx.core.data.qmt_client import QMTClient


class MetaSourceError(RuntimeError):
    """Raised when an online metadata source is unavailable."""


class BaoStockMetaSource:
    def fetch_security_master(self, date: str | None = None) -> pd.DataFrame:
        try:
            with BaoStockClient(pause_seconds=0.1) as client:
                frame = client.query_all_stocks(date)
        except Exception as exc:
            raise MetaSourceError(f"BaoStock security master fetch failed: {exc}") from exc
        if frame.empty:
            raise MetaSourceError("BaoStock security master fetch returned empty data")
        return pd.DataFrame({
            "symbol": frame["code"].map(normalize_symbol),
            "name": frame.get("code_name"),
            "list_date": frame.get("ipoDate"),
            "delist_date": frame.get("outDate"),
            "source": "baostock",
        })


class QMTMetaSource:
    def fetch_security_master(self, sectors: list[str] | None = None) -> pd.DataFrame:
        try:
            with QMTClient() as client:
                frame = client.query_security_master(sectors=sectors)
        except Exception as exc:
            raise MetaSourceError(f"QMT security master fetch failed: {exc}") from exc
        if frame.empty:
            raise MetaSourceError("QMT security master fetch returned empty data")
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame["source"] = "qmt"
        return frame


class AkShareMetaSource:
    def fetch_industry_names(self) -> pd.DataFrame:
        try:
            import akshare as ak

            return ak.stock_board_industry_name_em()
        except Exception as exc:
            raise MetaSourceError(f"AkShare industry list fetch failed: {exc}") from exc

    def fetch_concept_names(self) -> pd.DataFrame:
        try:
            import akshare as ak

            return ak.stock_board_concept_name_em()
        except Exception as exc:
            raise MetaSourceError(f"AkShare concept list fetch failed: {exc}") from exc

    def fetch_industry_members(self, industry_name: str) -> pd.DataFrame:
        try:
            import akshare as ak

            frame = ak.stock_board_industry_cons_em(symbol=industry_name)
        except Exception as exc:
            raise MetaSourceError(f"AkShare industry members fetch failed for {industry_name}: {exc}") from exc
        if frame.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "symbol": frame["代码"].map(normalize_symbol),
            "industry_name": industry_name,
            "industry_source": "eastmoney",
            "level": 1,
            "industry_code": industry_name,
        })

    def fetch_concept_members(self, concept_name: str) -> pd.DataFrame:
        try:
            import akshare as ak

            frame = ak.stock_board_concept_cons_em(symbol=concept_name)
        except Exception as exc:
            raise MetaSourceError(f"AkShare concept members fetch failed for {concept_name}: {exc}") from exc
        if frame.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "symbol": frame["代码"].map(normalize_symbol),
            "sector_type": "concept",
            "sector_name": concept_name,
            "sector_code": concept_name,
        })
