"""ETF metadata snapshot builder tests."""

from __future__ import annotations

import pandas as pd

from quantx.tools.sync_etf_meta import build_etf_master, classify_etf_name, normalize_fund_symbol


def test_normalize_fund_symbol_handles_common_market_formats():
    assert normalize_fund_symbol("sh512480") == "SH512480"
    assert normalize_fund_symbol("512480.SH") == "SH512480"
    assert normalize_fund_symbol("159995") == "SZ159995"
    assert normalize_fund_symbol("511880") == "SH511880"


def test_classify_etf_name_marks_sector_theme_candidates():
    chip = classify_etf_name("芯片ETF华夏")
    broad = classify_etf_name("沪深300ETF")
    bond = classify_etf_name("可转债ETF")
    hk = classify_etf_name("恒生科技ETF")

    assert chip["category"] == "sector_theme"
    assert chip["is_dynamic_theme_candidate"] is True
    assert broad["category"] == "broad_index"
    assert broad["is_dynamic_theme_candidate"] is False
    assert bond["category"] == "bond"
    assert hk["category"] == "cross_border"


def test_build_etf_master_from_akshare_like_frame():
    frame = pd.DataFrame([
        {"代码": "sh512480", "名称": "半导体ETF", "最新价": "1.2", "涨跌幅": "2.5", "成交量": "100", "成交额": "200"},
        {"代码": "sz159995", "名称": "芯片ETF华夏", "最新价": "1.5", "涨跌幅": "-1.2", "成交量": "300", "成交额": "400"},
        {"代码": "sh510300", "名称": "沪深300ETF", "最新价": "4.0", "涨跌幅": "0.1", "成交量": "500", "成交额": "600"},
    ])
    frame["fund_market_type"] = "ETF基金"

    master = build_etf_master(frame, snapshot_date="2026-07-08")

    assert list(master["symbol"]) == ["SH510300", "SH512480", "SZ159995"]
    row = master.set_index("symbol").loc["SH512480"]
    assert row["category"] == "sector_theme"
    assert bool(row["is_dynamic_theme_candidate"]) is True
    assert row["latest_amount"] == 200.0
    assert row["snapshot_date"] == "2026-07-08"
