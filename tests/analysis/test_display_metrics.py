"""Derived display metrics tests."""

import pytest

from quantx.core.analysis.display_metrics import annual_returns, build_next_session_guide


def test_annual_returns_groups_by_calendar_year():
    rows = annual_returns([
        {"date": "2021-01-04", "total_value": 100.0},
        {"date": "2021-12-31", "total_value": 120.0},
        {"date": "2022-01-04", "total_value": 120.0},
        {"date": "2022-12-30", "total_value": 90.0},
    ])

    assert [row["year"] for row in rows] == [2021, 2022]
    assert rows[0]["return"] == pytest.approx(0.2)
    assert rows[1]["return"] == pytest.approx(-0.25)


def test_build_next_session_guide_splits_new_and_held_candidates():
    guide = build_next_session_guide(
        {
            "date": "2026-07-03",
            "raw_candidate_count": 2,
            "selected_count": 2,
            "selected_candidates": [
                {"symbol": "SZ000001", "score": 1.2, "where": True},
                {"symbol": "SZ000002", "score": 1.1, "where": True},
            ],
        },
        [{"symbol": "SZ000002", "quantity": 100}],
        {
            "execution": {"deal_price": "close", "buy": {"sizing": "cash_equal", "lot_size": 100}},
            "rebalance": {"max_positions": 5},
        },
        latest_date="2026-07-03",
    )

    assert guide["signal_date"] == "2026-07-03"
    assert [row["symbol"] for row in guide["new_buy_candidates"]] == ["SZ000001"]
    assert [row["symbol"] for row in guide["already_held_candidates"]] == ["SZ000002"]
    assert "下一交易日按策略候选尝试新买入 1 只" in guide["steps"][0]
