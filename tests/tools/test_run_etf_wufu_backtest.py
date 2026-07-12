"""Wufu ETF rotation helper tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quantx.strategies.market_regime_rotation_model import (
    _affordable_quantity,
    _apply_choppy_confirmation,
    _apply_hold_hysteresis,
    _dynamic_theme_pool,
    _apply_dynamic_theme_gate,
    _execute_rebalance,
    _execution_price_table,
    _last_price,
    _load_theme_symbols_from_meta,
    _select_target,
    _trade_price,
    adjusted_corr,
    calculate_momentum_score,
    parse_args,
)


def test_affordable_quantity_includes_transaction_costs():
    cfg = SimpleNamespace(min_money=10.0, commission=0.001, min_cost=5.0)

    quantity = _affordable_quantity(cash=1000.0, price=10.0, cfg=cfg)

    assert quantity == 99
    assert quantity * 10.0 + max(5.0, quantity * 10.0 * 0.001) <= 1000.0


def test_trade_price_requires_current_bar_but_last_price_can_mark_to_previous_close():
    close = pd.DataFrame(
        {"SH513500": [1.39, None, 1.38]},
        index=pd.to_datetime(["2022-03-28", "2022-03-29", "2022-03-30"]),
    )

    assert _trade_price(close, pd.Timestamp("2022-03-29"), "SH513500") is None
    assert _last_price(close, pd.Timestamp("2022-03-29"), "SH513500") == 1.39


def test_adjusted_corr_handles_duplicate_free_price_table():
    prices = pd.DataFrame(
        {
            "A": [1.0, 1.1, 1.2, 1.3],
            "B": [2.0, 2.1, 2.3, 2.5],
        }
    )

    corr = adjusted_corr(prices)

    assert set(corr.index) == {"A", "B"}
    assert corr.loc["A", "A"] == 1.0


def test_weighted_momentum_score_reports_positive_trend_quality():
    score, annualized, r2 = calculate_momentum_score([1.0, 1.02, 1.04, 1.06, 1.08, 1.10], 5)

    assert score is not None and score > 0
    assert annualized is not None and annualized > 0
    assert r2 is not None and r2 > 0.99


def test_hold_hysteresis_keeps_current_top_rank_position():
    cfg = SimpleNamespace(hold_rank_top=3, switch_score_premium=0.0, hold_min_momentum_score=0.0)
    filtered = [
        {"etf": "SH513100", "momentum_score": 1.20},
        {"etf": "SH513400", "momentum_score": 1.05},
        {"etf": "SH518880", "momentum_score": 0.90},
    ]

    target, diag = _apply_hold_hysteresis("SH513100", "SH513400", filtered, cfg)

    assert target == "SH513400"
    assert diag["hold_hysteresis_blocked"] is True
    assert "hold_rank_top" in diag["hold_hysteresis_reasons"]


def test_choppy_confirmation_waits_before_switching_target():
    cfg = SimpleNamespace(choppy_confirm_days=2, defensive_etf="SH511880")
    state = {}

    first_target, first_diag = _apply_choppy_confirmation(
        "SH513100", "震荡期", "SH513400", cfg, state, defensive_available=True
    )
    second_target, second_diag = _apply_choppy_confirmation(
        "SH513100", "震荡期", "SH513400", cfg, state, defensive_available=True
    )

    assert first_target == "SH513400"
    assert first_diag["choppy_confirm_wait"] is True
    assert second_target == "SH513100"
    assert second_diag["choppy_confirmed"] is True


def test_execution_price_table_uses_open_only_for_next_open():
    close = pd.DataFrame({"SH513400": [1.20]}, index=pd.to_datetime(["2026-07-08"]))
    open_ = pd.DataFrame({"SH513400": [1.10]}, index=pd.to_datetime(["2026-07-08"]))

    assert _execution_price_table(SimpleNamespace(execution_mode="next_open"), close, open_).at[
        pd.Timestamp("2026-07-08"), "SH513400"
    ] == 1.10
    assert _execution_price_table(SimpleNamespace(execution_mode="next_close"), close, open_).at[
        pd.Timestamp("2026-07-08"), "SH513400"
    ] == 1.20


def test_execute_rebalance_records_prior_signal_reason_on_next_day_open():
    cfg = SimpleNamespace(min_money=10.0, commission=0.001, min_cost=1.0, slippage=0.0)
    open_prices = pd.DataFrame({"SH513400": [1.10]}, index=pd.to_datetime(["2026-07-08"]))
    trades = []

    position, cash = _execute_rebalance(
        pd.Timestamp("2026-07-08"),
        "SH513400",
        None,
        1000.0,
        open_prices,
        cfg,
        trades,
        reason="signal:2026-07-07",
    )

    assert position is not None
    assert position["entry_date"] == "2026-07-08"
    assert position["entry_price"] == 1.10
    assert cash < 1000.0
    assert trades[0]["reason"] == "signal:2026-07-07"


def _sample_bars(symbol: str, start: str, prices: list[float], amount: float = 100_000_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({
        "date": dates,
        "code": symbol,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000.0] * len(prices),
        "amount": [amount] * len(prices),
    })


def test_dynamic_theme_pool_selects_strong_liquid_theme_etfs():
    cfg = SimpleNamespace(
        dynamic_theme_top_n=2,
        dynamic_theme_min_score=0.0,
        dynamic_theme_liquidity_lookback=3,
        dynamic_theme_min_avg_amount=10_000_000.0,
        lookback_days=5,
    )
    bars = {
        "SH512480": _sample_bars("SH512480", "2026-01-01", [1, 1.01, 1.03, 1.06, 1.10, 1.15]),
        "SH515880": _sample_bars("SH515880", "2026-01-01", [1, 1.01, 1.015, 1.02, 1.03, 1.04]),
        "SH512880": _sample_bars("SH512880", "2026-01-01", [1, 1.05, 1.04, 1.03, 1.02, 1.01]),
    }

    pool, detail = _dynamic_theme_pool(pd.Timestamp("2026-01-08"), bars, cfg)

    assert pool == ["SH512480", "SH515880"]
    assert detail[0]["momentum_score"] > detail[1]["momentum_score"] > 0
    assert detail[0]["name"] is None


def test_load_theme_symbols_from_etf_meta_snapshot(tmp_path):
    meta_path = tmp_path / "etf_master.csv"
    pd.DataFrame([
        {
            "symbol": "SH512480",
            "name": "半导体ETF",
            "category": "sector_theme",
            "theme_group": "半导体",
            "is_dynamic_theme_candidate": "True",
        },
        {
            "symbol": "SH510300",
            "name": "沪深300ETF",
            "category": "broad_index",
            "theme_group": "沪深300",
            "is_dynamic_theme_candidate": "False",
        },
    ]).to_csv(meta_path, index=False)

    symbols, meta = _load_theme_symbols_from_meta(meta_path)

    assert symbols == ["SH512480"]
    assert meta["SH512480"]["name"] == "半导体ETF"


def test_dynamic_theme_gate_blocks_disallowed_regime_and_weak_premium():
    filtered = [
        {"etf": "SH513100", "momentum_score": 2.0, "dynamic_theme_pool": False},
        {"etf": "SH512480", "momentum_score": 2.1, "dynamic_theme_pool": True},
        {"etf": "SH515880", "momentum_score": 2.5, "dynamic_theme_pool": True},
    ]
    cfg = SimpleNamespace(dynamic_theme_regimes=["正常期"], dynamic_theme_score_premium=0.2)

    weak_gated, weak_diag = _apply_dynamic_theme_gate([item.copy() for item in filtered], "走弱期", cfg)
    normal_gated, normal_diag = _apply_dynamic_theme_gate([item.copy() for item in filtered], "正常期", cfg)

    assert [item["etf"] for item in weak_gated] == ["SH513100"]
    assert weak_diag["dynamic_theme_gate_blocked_reasons"]["regime_not_allowed"] == 2
    assert [item["etf"] for item in normal_gated] == ["SH513100", "SH515880"]
    assert normal_diag["dynamic_theme_gate_blocked_reasons"]["score_premium"] == 1


def test_parse_args_dynamic_theme_defaults_use_recommended_gate():
    cfg = parse_args(["--dynamic-theme-top-n", "1"])

    assert cfg.dynamic_theme_top_n == 1
    assert cfg.dynamic_theme_regimes == ["正常期", "震荡期"]
    assert cfg.dynamic_theme_score_premium == 0.05


def test_select_target_can_add_dynamic_theme_in_weak_regime():
    cfg = SimpleNamespace(
        dynamic_theme_top_n=1,
        dynamic_theme_min_score=0.0,
        dynamic_theme_liquidity_lookback=3,
        dynamic_theme_min_avg_amount=10_000_000.0,
        lookback_days=5,
        short_lookback=5,
        volume_lookback=3,
        min_score=0.0,
        max_score=1_000_000.0,
        score_threshold_ratio=0.9,
        r2_threshold=0.0,
        normal_r2_threshold=0.0,
        ma_lookback=3,
        ma_threshold=1.0,
        volume_threshold=99.0,
        intraday_volume_fraction=0.67,
        loss_threshold=0.0,
        corr_lookback=5,
        corr_hold_threshold=0.85,
        corr_hold_momentum_max=7.0,
        corr_overlay_threshold=0.88,
        corr_overlay_momentum_max=8.0,
        pick_target_max_padj=0.85,
        defensive_etf="SH511880",
        top_n=1,
        hold_rank_top=0,
        switch_score_premium=0.0,
        hold_min_momentum_score=0.0,
    )
    dates = pd.bdate_range("2026-01-01", periods=8)
    bars = {
        "SH513100": _sample_bars("SH513100", "2026-01-01", [1, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035]),
        "SH512480": _sample_bars("SH512480", "2026-01-01", [1, 1.008, 1.016, 1.024, 1.032, 1.040, 1.048, 1.056]),
        "SH511880": _sample_bars("SH511880", "2026-01-01", [1] * 8),
    }
    close = pd.DataFrame({symbol: df.set_index("date")["close"] for symbol, df in bars.items()}, index=dates)
    theme_meta = {"SH512480": {"name": "半导体ETF", "category": "sector_theme", "theme_group": "半导体"}}

    target, _top, diag = _select_target(
        dates[-1], "走弱期", bars, close, None, cfg, ["SH512480"], theme_meta
    )

    assert target == "SH512480"
    assert diag["dynamic_theme_count"] == 1
    assert diag["dynamic_theme_top"][0]["etf"] == "SH512480"
    assert diag["dynamic_theme_top"][0]["name"] == "半导体ETF"


def test_generated_strict_wufu_runs_trade_on_next_selection_day():
    run_ids = [
        "20260708_100527_etf_wufu_next_open_confirm3_corr_daily_2016_2026",
        "20260708_100530_etf_wufu_next_close_confirm3_corr_daily_2016_2026",
    ]
    root = Path("runs")
    if not all((root / run_id).exists() for run_id in run_ids):
        pytest.skip("strict Wufu run artifacts are not present in this workspace")

    for run_id in run_ids:
        run_dir = root / run_id
        selections = json.loads((run_dir / "selection_candidates.json").read_text(encoding="utf-8"))
        trades = json.loads((run_dir / "trades.json").read_text(encoding="utf-8"))
        selection_dates = [pd.Timestamp(row["date"]) for row in selections]
        next_selection_date = {selection_dates[idx]: selection_dates[idx + 1] for idx in range(len(selection_dates) - 1)}

        assert trades, run_id
        for trade in trades:
            reason = str(trade.get("reason", ""))
            assert reason.startswith("signal:"), (run_id, trade)
            signal_date = pd.Timestamp(reason.split(":", 1)[1])
            assert pd.Timestamp(trade["date"]) == next_selection_date[signal_date], (run_id, trade)
