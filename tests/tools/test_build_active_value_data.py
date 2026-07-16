import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from quantx.tools.build_active_value_data import (
    compute_0amv_daily,
    compute_active_value_daily,
    finalize_active_value_daily,
    is_current_session_incomplete,
    merge_existing,
)


def test_compute_active_value_daily_builds_core_proxy_with_filters():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = []
    for i, date in enumerate(dates):
        rows.append(_row("SH600000", date, amount=100 + i, volume=10, vwap=10, close=10 + i * 0.1))
        rows.append(_row("SZ300001", date, amount=200 + i, volume=20, vwap=10, close=20 + i * 0.1))
        rows.append(_row("SH600001", date, amount=300 + i, volume=30, vwap=10, close=30 + i * 0.1))
    quote = pd.DataFrame(rows)

    daily = compute_active_value_daily(
        quote,
        risk_names={"SH600001"},
        min_listing_days=2,
        calendar=pd.DatetimeIndex(dates),
    )

    # All basic-tradable A shares are counted in the broad amount proxy.
    assert daily.loc[2, "active_all_amount"] == (102 + 202 + 302)
    # Core proxy is mainboard, non-risk-name, non-new after listing-day filter.
    assert daily.loc[0, "active_core_amount"] == 0
    assert daily.loc[1, "active_core_amount"] == 0
    assert daily.loc[2, "active_core_amount"] == 102
    assert daily.loc[2, "core_symbol_count"] == 1
    assert daily.loc[2, "risk_symbol_count"] == 1


def test_compute_active_value_daily_builds_right_side_market_breadth():
    dates = pd.date_range("2024-01-02", periods=130, freq="B")
    rows = []
    for i, date in enumerate(dates):
        rows.append(_row("SH600000", date, amount=100, volume=10, vwap=10, close=10 + i * 0.1))
        rows.append(_row("SZ000001", date, amount=100, volume=10, vwap=10, close=30 - i * 0.1))
    daily = compute_active_value_daily(
        pd.DataFrame(rows),
        risk_names=set(),
        min_listing_days=0,
        calendar=pd.DatetimeIndex(dates),
    )

    latest = daily.iloc[-1]
    assert latest["right_side_core_count"] == 1
    assert latest["right_side_core_ratio"] == pytest.approx(0.5)
    assert latest["right_side_core_amount_ratio"] == pytest.approx(0.5)


def test_finalize_active_value_daily_adds_returns_ma_and_state():
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=10, freq="B"),
        "active_all_amount": [100, 104, 108, 100, 101, 102, 103, 104, 105, 106],
        "active_tradable_amount": [100] * 10,
        "active_mainboard_amount": [100] * 10,
        "active_core_amount": [100, 104, 108, 100, 101, 102, 103, 104, 105, 106],
        "active_vwap_value": [100] * 10,
        "symbol_count": [1] * 10,
        "core_symbol_count": [1] * 10,
        "risk_symbol_count": [0] * 10,
        "st_like_symbol_count": [0] * 10,
        "basic_tradable_count": [1] * 10,
    })

    out = finalize_active_value_daily(frame)

    assert out.loc[1, "active_core_amount_ret1"] == pytest.approx(0.04)
    assert out.loc[2, "active_core_amount_ret2"] == pytest.approx(0.08)
    assert out.loc[1, "active_core_amount_strong_up_day"]
    assert out.loc[4, "active_core_amount_ma10"] == frame.loc[:4, "active_core_amount"].mean()
    assert "active_core_amount_above_ma10" in out.columns
    assert out["is_complete_day"].all()


def test_finalize_active_value_daily_adds_breadth_bull_state():
    rows = 100
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=rows, freq="B"),
        "active_all_amount": [100] * rows,
        "active_tradable_amount": [100] * rows,
        "active_mainboard_amount": [100] * rows,
        "active_core_amount": [100] * rows,
        "active_vwap_value": [100] * rows,
        "symbol_count": [10] * rows,
        "core_symbol_count": [10] * rows,
        "risk_symbol_count": [0] * rows,
        "st_like_symbol_count": [0] * rows,
        "basic_tradable_count": [10] * rows,
        "right_side_core_ratio": [0.1 + i * 0.005 for i in range(rows)],
    })

    out = finalize_active_value_daily(frame)

    assert out.iloc[-1]["breadth_bull"]
    assert out.iloc[-1]["breadth_bull_age"] > 1
    assert out["breadth_bull_start"].sum() >= 1


def test_finalize_active_value_daily_blocks_incomplete_latest_state():
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=3, freq="B"),
        "active_all_amount": [100, 100, 120],
        "active_tradable_amount": [100, 100, 120],
        "active_mainboard_amount": [100, 100, 120],
        "active_core_amount": [100, 100, 120],
        "active_vwap_value": [100, 100, 120],
        "symbol_count": [100, 100, 100],
        "core_symbol_count": [80, 80, 80],
        "risk_symbol_count": [0, 0, 0],
        "st_like_symbol_count": [0, 0, 0],
        "basic_tradable_count": [100, 100, 100],
    })

    out = finalize_active_value_daily(frame, mark_latest_incomplete=True)

    assert out.loc[2, "active_core_amount_strong_up_day_raw"]
    assert not out.loc[2, "is_complete_day"]
    assert not out.loc[2, "active_core_amount_strong_up_day"]


def test_current_session_is_only_incomplete_before_market_data_ready():
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-07-15")]})
    timezone = ZoneInfo("Asia/Shanghai")

    assert is_current_session_incomplete(frame, now=datetime(2026, 7, 15, 14, 30, tzinfo=timezone))
    assert not is_current_session_incomplete(frame, now=datetime(2026, 7, 15, 18, 0, tzinfo=timezone))
    assert not is_current_session_incomplete(frame, now=datetime(2026, 7, 16, 10, 0, tzinfo=timezone))


def test_merge_existing_replaces_overlap_dates():
    existing = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "active_all_amount": [1, 2, 3],
    })
    partial = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        "active_all_amount": [20, 30, 40],
    })

    merged = merge_existing(existing, partial)

    assert merged["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert merged["active_all_amount"].tolist() == [1, 20, 30, 40]


def test_compute_0amv_daily_builds_market_virtual_klines():
    date1 = pd.Timestamp("2024-01-02")
    date2 = pd.Timestamp("2024-01-03")
    quote = pd.DataFrame([
        _row("SH600000", date1, amount=100, volume=10, vwap=10, close=10),
        _row("SZ000001", date1, amount=200, volume=20, vwap=10, close=10),
        _row("SH688001", date1, amount=300, volume=30, vwap=10, close=10),
        _row("SZ300001", date1, amount=400, volume=40, vwap=10, close=10),
        _row("SH600000", date2, amount=110, volume=11, vwap=10, close=11),
        _row("SZ000001", date2, amount=210, volume=21, vwap=10, close=11),
        _row("SH688001", date2, amount=310, volume=31, vwap=10, close=11),
        _row("SZ300001", date2, amount=410, volume=41, vwap=10, close=11),
    ])

    rows = compute_0amv_daily(quote)

    latest = rows[rows["date"] == "2024-01-03"].set_index("symbol")
    assert latest.loc["0AMV_SH", "close"] == 110
    assert latest.loc["0AMV_SZ", "close"] == 210
    assert latest.loc["0AMV_KC", "close"] == 310
    assert latest.loc["0AMV_CY", "close"] == 410
    assert latest.loc["0AMV_ALL", "close"] == 1040
    assert latest.loc["0AMV_SH", "open"] == 100
    assert latest.loc["0AMV_ALL", "member_count"] == 4


def _row(symbol, date, *, amount, volume, vwap, close):
    return {
        "instrument": symbol,
        "datetime": pd.Timestamp(date),
        "$open": close,
        "$high": close * 1.01,
        "$low": close * 0.99,
        "$close": close,
        "$volume": volume,
        "$amount": amount,
        "$vwap": vwap,
    }
