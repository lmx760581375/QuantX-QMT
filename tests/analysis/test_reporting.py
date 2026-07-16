"""Run artifact reporting tests."""

from datetime import date

import pytest

from quantx.core.analysis.reporting import compute_metrics, match_closed_positions, write_run_artifacts
from quantx.core.analysis import load_run_artifacts
from quantx.core.analysis.reporting import RunReport


def test_compute_metrics_and_match_closed_positions():
    daily_nav = [
        {"date": "2021-01-04", "cash": 20.0, "total_value": 100.0, "drawdown": 0.0, "position_count": 1},
        {"date": "2021-01-05", "cash": 110.0, "total_value": 110.0, "drawdown": 0.0, "position_count": 0},
    ]
    trades = [
        {
            "date": "2021-01-04",
            "symbol": "SZ000001",
            "action": "BUY",
            "quantity": 10,
            "price": 10.0,
            "trade_value": 100.0,
            "total_cost": 0.0,
        },
        {
            "date": "2021-01-05",
            "symbol": "SZ000001",
            "action": "SELL",
            "quantity": 10,
            "price": 11.0,
            "trade_value": 110.0,
            "total_cost": 0.0,
        },
    ]

    closed = match_closed_positions(trades)
    metrics = compute_metrics(daily_nav, trades, closed, init_cash=100.0)

    assert closed[0]["net_pnl"] == 10.0
    assert metrics["total_return"] == pytest.approx(0.1)
    assert metrics["trade_count"] == 2
    assert metrics["avg_capital_utilization"] == pytest.approx(0.4)
    assert metrics["high_utilization_day_ratio"] == pytest.approx(0.5)
    assert metrics["zero_utilization_day_ratio"] == pytest.approx(0.5)


def test_compute_metrics_counts_executed_trades_and_rejections_separately():
    daily_nav = [
        {"date": "2021-01-04", "cash": 0.0, "total_value": 100.0, "drawdown": 0.0, "position_count": 1},
        {"date": "2021-01-05", "cash": 110.0, "total_value": 110.0, "drawdown": 0.0, "position_count": 0},
    ]
    trades = [
        {
            "date": "2021-01-04",
            "symbol": "SZ000001",
            "action": "BUY",
            "quantity": 10,
            "price": 10.0,
            "trade_value": 100.0,
            "total_cost": 1.0,
            "reject_reason": "",
        },
        {
            "date": "2021-01-05",
            "symbol": "SZ000001",
            "action": "SELL",
            "quantity": 10,
            "price": 11.0,
            "trade_value": 110.0,
            "total_cost": 1.5,
            "reject_reason": "",
        },
        {
            "date": "2021-01-05",
            "symbol": "SZ000002",
            "action": "BUY",
            "quantity": 10,
            "price": 9.0,
            "trade_value": 90.0,
            "total_cost": 99.0,
            "reject_reason": "suspended",
        },
    ]

    closed = match_closed_positions(trades)
    metrics = compute_metrics(daily_nav, trades, closed, init_cash=100.0)

    assert metrics["order_count"] == 3
    assert metrics["trade_count"] == 2
    assert metrics["reject_count"] == 1
    assert metrics["buy_count"] == 1
    assert metrics["sell_count"] == 1
    assert metrics["total_cost"] == pytest.approx(2.5)


def test_write_and_load_run_artifacts(tmp_path):
    report = RunReport(
        run_id="unit_run",
        summary={"run_id": "unit_run", "total_return": 0.1},
        metrics={"total_return": 0.1},
        daily_nav=[],
        trades=[],
        positions=[],
        closed_positions=[],
        daily_selection_candidates=[
            {
                "date": "2021-01-04",
                "raw_candidate_count": 1,
                "selected_count": 1,
                "selected_candidates": [{"symbol": "SZ000001", "score": 1.0, "where": True}],
            }
        ],
        explain={"strategy": {"formula_order": []}, "config": {"start": date(2021, 1, 4)}},
    )

    run_dir = write_run_artifacts(report, tmp_path)
    loaded = load_run_artifacts(run_dir)

    assert loaded["summary"]["run_id"] == "unit_run"
    assert loaded["explain"]["strategy"]["formula_order"] == []
    assert loaded["explain"]["config"]["start"] == "2021-01-04"
    assert loaded["daily_selection_candidates"][0]["date"] == "2021-01-04"
