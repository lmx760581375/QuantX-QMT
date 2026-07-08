"""MetaStore tests."""

import pandas as pd

from quantx.core.data.meta import MetaStore
from quantx.core.data.meta.service import MetaUpdateService
from quantx.core.data.meta.store import normalize_symbol


def test_normalize_symbol_formats():
    assert normalize_symbol("sh.600000") == "SH600000"
    assert normalize_symbol("600000.SH") == "SH600000"
    assert normalize_symbol("000001") == "SZ000001"
    assert normalize_symbol("300750") == "SZ300750"


def test_meta_store_imports_security_and_industry_csv(tmp_path):
    meta_uri = tmp_path / "meta.sqlite"
    security_csv = tmp_path / "security.csv"
    industry_csv = tmp_path / "industry.csv"
    pd.DataFrame([
        {"symbol": "SH600000", "name": "浦发银行", "exchange": "SH", "board": "mainboard"},
        {"symbol": "SZ000001", "name": "平安银行", "exchange": "SZ", "board": "mainboard"},
    ]).to_csv(security_csv, index=False)
    pd.DataFrame([
        {"symbol": "SH600000", "industry_source": "eastmoney", "level": 1, "industry_code": "bank", "industry_name": "银行"},
        {"symbol": "SZ000001", "industry_source": "eastmoney", "level": 1, "industry_code": "bank", "industry_name": "银行"},
    ]).to_csv(industry_csv, index=False)

    store = MetaStore(meta_uri)
    service = MetaUpdateService(store)
    assert service.import_security_master_csv(security_csv).rows == 2
    assert service.import_industry_csv(industry_csv).rows == 2

    meta = store.get_symbol_meta(["SH600000"])

    assert meta.loc["SH600000", "name"] == "浦发银行"
    assert meta.loc["SH600000", "industry_name"] == "银行"


def test_meta_store_imports_sector_membership_and_symbol_concepts(tmp_path):
    store = MetaStore(tmp_path / "meta.sqlite")
    service = MetaUpdateService(store)
    sector_csv = tmp_path / "sector.csv"
    pd.DataFrame([
        {"symbol": "SH600000", "sector_type": "concept", "sector_code": "BK1", "sector_name": "中字头"},
        {"symbol": "SH600000", "sector_type": "concept", "sector_code": "BK2", "sector_name": "融资融券"},
        {"symbol": "SZ000001", "sector_type": "concept", "sector_code": "BK2", "sector_name": "融资融券"},
    ]).to_csv(sector_csv, index=False)

    assert service.import_sector_csv(sector_csv).rows == 3
    meta = store.get_symbol_meta(["SH600000"])
    sectors = store.get_sectors()

    assert meta.loc["SH600000", "concept_codes"] == "BK1|BK2"
    assert meta.loc["SH600000", "concept_names"] == "中字头|融资融券"
    assert set(sectors["sector_name"]) == {"中字头", "融资融券"}


def test_meta_store_returns_industry_only_symbols(tmp_path):
    store = MetaStore(tmp_path / "meta.sqlite")
    store.upsert_industry_membership(pd.DataFrame([
        {"symbol": "SZ920010", "industry_source": "eastmoney", "level": 1, "industry_code": "gas", "industry_name": "燃气"},
    ]))

    meta = store.get_symbol_meta(["SZ920010"])

    assert meta.loc["SZ920010", "name"] is None
    assert meta.loc["SZ920010", "industry_name"] == "燃气"


def test_meta_store_industry_summary(tmp_path):
    store = MetaStore(tmp_path / "meta.sqlite")
    store.upsert_security_master(pd.DataFrame([{"symbol": "SH600000", "name": "浦发银行"}]))
    store.upsert_industry_membership(pd.DataFrame([{"symbol": "SH600000", "industry_name": "银行"}]))

    industries = store.get_industries()

    assert industries.iloc[0]["industry_name"] == "银行"
    assert industries.iloc[0]["symbol_count"] == 1


def test_meta_csv_import_uses_row_snapshot_date(tmp_path):
    meta_uri = tmp_path / "meta.sqlite"
    security_csv = tmp_path / "security.csv"
    pd.DataFrame([{"symbol": "SH600000", "name": "浦发银行", "snapshot_date": "2026-06-25"}]).to_csv(
        security_csv,
        index=False,
    )

    store = MetaStore(meta_uri)
    service = MetaUpdateService(store)
    service.import_security_master_csv(security_csv, source="myquant")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT source, snapshot_date FROM security_master WHERE symbol = ?",
            ("SH600000",),
        ).fetchone()

    assert row == ("myquant", "2026-06-25")


def test_meta_csv_import_preserves_leading_zero_codes(tmp_path):
    security_csv = tmp_path / "security.csv"
    pd.DataFrame([{"code": "000001", "name": "平安银行"}]).to_csv(security_csv, index=False)

    store = MetaStore(tmp_path / "meta.sqlite")
    service = MetaUpdateService(store)
    service.import_security_master_csv(security_csv)

    meta = store.get_symbol_meta(["SZ000001"])

    assert meta.loc["SZ000001", "name"] == "平安银行"
