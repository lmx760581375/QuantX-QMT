import pandas as pd
import pytest

from quantx.tools.build_market_regime_data import (
    add_regime_indicators,
    choose_calc_start,
    consecutive_true_count,
    sma_tdx,
)


def test_sma_tdx_uses_recursive_tdx_definition():
    values = pd.Series([10.0, 20.0, 30.0])

    result = sma_tdx(values, 3, 1)

    assert result.tolist() == pytest.approx([10.0, 13.3333333333, 18.8888888889])


def test_add_regime_indicators_builds_active_bull_and_kdj():
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    close = pd.Series(range(100), dtype=float) + 100.0
    frame = pd.DataFrame({
        "date": dates,
        "code": "SH000001",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "preclose": close.shift(1),
        "volume": 1.0,
        "amount": 1.0,
        "pct_chg": close.pct_change() * 100.0,
    })

    result = add_regime_indicators(frame)

    assert result.loc[99, "active_bull"]
    assert result.loc[99, "active_bull_age"] > 1
    assert pd.notna(result.loc[99, "kdj_j"])


def test_consecutive_true_count_resets_on_false():
    result = consecutive_true_count(pd.Series([False, True, True, False, True]))

    assert result.tolist() == [0, 1, 2, 0, 1]


def test_choose_calc_start_uses_overlap_tail():
    existing = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=200, freq="B")})

    result = choose_calc_start(existing, start="2012-01-01", overlap_days=120)

    assert result == existing.loc[80, "date"].strftime("%Y-%m-%d")
