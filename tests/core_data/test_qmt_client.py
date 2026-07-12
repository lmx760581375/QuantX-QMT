"""Tests for the xqshare-backed QMT client."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from quantx.core.data.qmt_client import QMTClient


def test_qmt_client_uses_remote_xtdata_and_closes_connection(monkeypatch, tmp_path):
    calls = {}

    class FakeXtData:
        def download_history_data(self, symbol, **kwargs):
            calls["download"] = (symbol, kwargs)

        def get_market_data_ex(self, **kwargs):
            calls["market_data"] = kwargs
            return {
                "000001.SZ": pd.DataFrame(
                    {
                        "time": [1782835200000],
                        "open": [10.05],
                        "high": [10.18],
                        "low": [9.99],
                        "close": [10.16],
                        "volume": [906890],
                        "amount": [915838549.0],
                    },
                    index=["20260701"],
                )
            }

    class FakeRemote:
        def __init__(self, **kwargs):
            calls["remote_kwargs"] = kwargs
            self.xtdata = FakeXtData()

        def close(self):
            calls["closed"] = True

    monkeypatch.setitem(sys.modules, "xqshare", SimpleNamespace(XtQuantRemote=FakeRemote))
    env_file = tmp_path / ".env"
    env_file.write_text("XQSHARE_REMOTE_HOST=192.168.0.117\n", encoding="utf-8")

    with QMTClient(env_file=env_file) as client:
        frame = client.query_history_k_data("SZ000001", "2026-07-01", "2026-07-11")

    assert calls["remote_kwargs"]["env_file"] == str(env_file)
    assert calls["download"] == (
        "000001.SZ",
        {"period": "1d", "start_time": "20260701", "end_time": "20260711"},
    )
    assert calls["market_data"]["stock_list"] == ["000001.SZ"]
    assert calls["market_data"]["dividend_type"] == "front_ratio"
    assert calls["closed"] is True
    assert frame.loc[0, "date"] == "2026-07-01"
    assert frame.loc[0, "code"] == "SZ000001"
    assert frame.loc[0, "close"] == 10.16


def test_qmt_client_reuses_remote_catalog_and_calendar(monkeypatch, tmp_path):
    calls = {}

    class FakeXtData:
        def get_stock_list_in_sector(self, sector):
            calls.setdefault("sectors", []).append(sector)
            return ["000001.SZ", "600000.SH", "INVALID"]

        def get_trading_dates(self, market, **kwargs):
            calls["calendar"] = (market, kwargs)
            return [20260701, 20260702]

    class FakeRemote:
        def __init__(self, **kwargs):
            self.xtdata = FakeXtData()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "xqshare", SimpleNamespace(XtQuantRemote=FakeRemote))

    with QMTClient(env_file=tmp_path / ".env") as client:
        symbols = client.get_stock_codes()
        dates = client.get_trading_dates("2026-07-01", "2026-07-02")

    assert symbols == ["SH600000", "SZ000001"]
    assert dates == ["2026-07-01", "2026-07-02"]
    assert calls["sectors"] == ["沪深A股"]
    assert calls["calendar"] == (
        "SH",
        {"start_time": "20260701", "end_time": "20260702"},
    )
