"""Dynamic factor runtime tests."""

import numpy as np
import pandas as pd
import pytest

from quantx.core.factor_runtime import FactorRuntime, FormulaError, MarketPanel


def _panel():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]), ["SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [1.0, 2.0, 2.0, 1.0, 3.0, 4.0],
        "$open": [1.0, 2.2, 2.0, 0.9, 3.5, 3.5],
        "$high": [1.1, 2.1, 2.2, 1.2, 3.2, 4.2],
        "$low": [0.9, 1.9, 1.8, 0.8, 2.8, 3.8],
        "$volume": [10.0, 20.0, 30.0, 15.0, 25.0, 40.0],
    }, index=index)
    return MarketPanel.from_frame(frame)


def test_market_panel_from_frame():
    panel = _panel()
    assert panel.get("close").shape == (3, 2)
    np.testing.assert_allclose(panel.get("close")[0], [1.0, 2.0])


def test_formula_runtime_csmean_and_dependency_order():
    runtime = FactorRuntime(_panel())
    runtime.compute_formulas({
        "strong": "close > 1.5",
        "market_health": "CSMean(strong)",
    })

    np.testing.assert_allclose(runtime.get_matrix("market_health"), [0.5, 0.5, 1.0])
    assert runtime.explain("market_health")[0] == "market_health = CSMean(strong)"


def test_formula_runtime_broadcasts_daily_market_vector_in_boolean_formula():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "strong": "close > 1.5",
        "market_health": "CSRatio(strong)",
        "tradable": "strong and (market_health >= 1.0)",
    })

    np.testing.assert_array_equal(
        result["tradable"],
        np.array([[False, False], [False, False], [True, True]]),
    )


def test_formula_runtime_broadcasts_daily_market_vector_in_arithmetic_formula():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "market_close": "CSMean(close)",
        "relative_close": "close / market_close",
    })

    np.testing.assert_allclose(
        result["relative_close"],
        np.array([[1.0 / 1.5, 2.0 / 1.5], [2.0 / 1.5, 1.0 / 1.5], [3.0 / 3.5, 4.0 / 3.5]]),
    )


def test_formula_runtime_where_broadcasts_daily_market_vector():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "market_health": "CSRatio(close > 1.5)",
        "boost": "Where(market_health >= 1.0, close * 0.1, 0)",
    })

    np.testing.assert_allclose(
        result["boost"],
        np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.4]]),
    )


def test_formula_runtime_group_mean_and_rank_with_label_group():
    runtime = FactorRuntime(_panel())
    runtime.values["industry"] = np.array([0, 1], dtype=np.int32)
    result = runtime.compute_formulas({
        "industry_close": "GroupMean(close, industry)",
        "industry_rank": "GroupRank(industry_close, industry)",
    })

    np.testing.assert_allclose(result["industry_close"], runtime.values["close"])
    np.testing.assert_allclose(
        result["industry_rank"],
        np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 0.5]], dtype=np.float32),
    )


def test_formula_runtime_group_mean_with_multi_membership_concept_group():
    runtime = FactorRuntime(_panel())
    runtime.values["concept"] = np.array([
        [True, True, False],
        [False, True, True],
    ])
    result = runtime.compute_formulas({
        "concept_mean": "GroupMean(close, concept, agg='max')",
        "concept_rank": "GroupRank(close, concept, agg='max')",
    })

    np.testing.assert_allclose(
        result["concept_mean"],
        np.array([[1.5, 2.0], [2.0, 1.5], [3.5, 4.0]], dtype=np.float32),
    )
    assert result["concept_rank"].shape == (3, 2)
    assert np.isfinite(result["concept_rank"]).all()


def test_formula_runtime_market_breadth_right_side_and_kdj_low():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=3), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [10, 8, 12, 11, 9, 7, 13, 12, np.nan],
    }, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    result = runtime.compute_formulas({
        "short_trend": "Mean(close, 2)",
        "long_trend": "Mean(close, 3) - 0.5",
        "right": "MarketBreadth(close, short_trend, long_trend)",
        "j": "Where(close >= 11, 10, 30)",
        "right_kdj_low": "MarketBreadth(close, short_trend, long_trend, j, 13)",
        "tradable": "(close > 0) and (right >= 0.5)",
    })

    np.testing.assert_allclose(result["right"], [1.0, 2 / 3, 1.0], rtol=1e-6)
    np.testing.assert_allclose(result["right_kdj_low"], [1 / 3, 1 / 3, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(
        result["tradable"],
        np.array([
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]),
    )


def test_formula_runtime_market_concentration_metrics():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=2), ["A", "B", "C", "D"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [10, 20, 30, 40, 11, 18, 33, 44],
        "$amount": [100, 200, 300, 400, 100, 100, 600, 200],
        "$float_mv": [1, 1, 2, 6, 1, 1, 2, 6],
    }, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    runtime.values["industry"] = np.array([0, 0, 1, 2], dtype=np.int32)
    result = runtime.compute_formulas({
        "ret1": "close / Ref(close, 1) - 1",
        "total_amount": "MarketTotalAmount(amount)",
        "industry_cr2": "MarketIndustryCR(amount, industry, 2)",
        "stock_hhi": "MarketHHI(amount)",
        "csd": "MarketCSD(ret1, float_mv)",
    })

    np.testing.assert_allclose(result["total_amount"], [1000.0, 1000.0])
    np.testing.assert_allclose(result["industry_cr2"], [0.7, 0.8])
    np.testing.assert_allclose(result["stock_hhi"], [3000.0, 4200.0])
    assert np.isnan(result["csd"][0])

    returns = np.array([0.1, -0.1, 0.1, 0.1], dtype=np.float32)
    weights = np.array([1, 1, 2, 6], dtype=np.float32)
    weights = weights / weights.sum()
    mu = float(np.sum(weights * returns))
    expected_csd = float(np.sqrt(np.sum(weights * (returns - mu) ** 2)))
    np.testing.assert_allclose(result["csd"][1], expected_csd, rtol=1e-6)


def test_formula_runtime_rejects_unsafe_syntax():
    runtime = FactorRuntime(_panel())
    with pytest.raises(FormulaError):
        runtime.compute_formulas({"bad": "close.__class__"})
    with pytest.raises(FormulaError):
        runtime.compute_formulas({"bad": "Unknown(close)"})


def test_formula_runtime_params_and_cross_section():
    runtime = FactorRuntime(_panel())
    runtime.compute_formulas({"mask": "close > ${threshold}"}, params={"threshold": 2.5})
    section = runtime.get_cross_section(["mask"], "2021-01-06")
    assert section.loc["SZ000001", "mask"]
    assert section.loc["SZ000002", "mask"]


def test_formula_runtime_bool_words_and_function_operators():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "word_mask": "(close > 1.5) and not (high < 0)",
        "func_mask": "And(close > 1.5, Not(high < 0))",
        "or_mask": "Or(close < 1.5, close > 3.5)",
    })

    np.testing.assert_array_equal(result["word_mask"], result["func_mask"])
    np.testing.assert_array_equal(
        result["or_mask"],
        np.array([[True, False], [False, True], [False, True]]),
    )


def test_formula_runtime_rejects_bitwise_boolean_syntax():
    runtime = FactorRuntime(_panel())
    with pytest.raises(FormulaError, match="Use and/or/not"):
        runtime.compute_formulas({"bad": "(close > 1) & (high > 1)"})


def test_formula_runtime_clip_fillna_isna():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "lag": "Ref(close, 1)",
        "filled": "FillNa(lag, 50)",
        "clipped": "Clip(close - 2, 0, 1)",
        "forced_nan": "NaN()",
        "missing": "IsNa(lag)",
    })

    np.testing.assert_allclose(result["filled"][0], [50.0, 50.0])
    np.testing.assert_allclose(result["clipped"], [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    assert np.isnan(result["forced_nan"])
    np.testing.assert_array_equal(result["missing"][0], [True, True])


def test_ref_boolean_missing_values_are_false():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({"lagged": "Ref(close > 1.5, 1)"})

    assert result["lagged"].dtype == bool
    np.testing.assert_array_equal(
        result["lagged"],
        np.array([[False, False], [False, True], [True, False]]),
    )


def test_formula_runtime_expanding_quantile():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({"q50": "ExpandingQuantile(close, 0.5)"})

    np.testing.assert_allclose(
        result["q50"],
        np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 2.0]], dtype=np.float32),
    )


def test_formula_runtime_max_vol_not_bearish():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({"ok": "MaxVolNotBearish(volume, open, close, 2)"})

    np.testing.assert_array_equal(
        result["ok"],
        np.array([[True, False], [True, False], [True, True]]),
    )


def test_formula_runtime_prior_consecutive():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({"run": "PriorConsecutive(close > open)"})

    np.testing.assert_allclose(
        result["run"],
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_formula_runtime_brick_chart_shape_and_edges():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({"brick": "BrickChart(high, low, close, 2, 3, 3, 3, 1, 90, 100, 1, 1, 1)"})

    assert result["brick"].shape == (3, 2)
    np.testing.assert_allclose(result["brick"][0], [0.0, 0.0])
    assert np.isfinite(result["brick"]).all()


def test_formula_runtime_sum_and_kdjj():
    runtime = FactorRuntime(_panel())
    result = runtime.compute_formulas({
        "vol2": "Sum(volume, 2)",
        "j": "KDJJ(high, low, close, 9)",
    })

    np.testing.assert_allclose(
        result["vol2"],
        np.array([[10.0, 20.0], [40.0, 35.0], [55.0, 55.0]], dtype=np.float32),
    )
    assert result["j"].shape == (3, 2)
    assert np.isfinite(result["j"]).all()
    np.testing.assert_allclose(result["j"][0], [50.0, 50.0], rtol=1e-5)


def test_kdjj_accepts_history_shorter_than_window():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$high": [11.0, 12.0, 13.0],
        "$low": [9.0, 10.0, 11.0],
        "$close": [10.0, 11.0, 12.0],
    }, index=index)

    result = FactorRuntime(MarketPanel.from_frame(frame)).compute_formulas({
        "j": "KDJJ(high, low, close, 9)",
    })

    assert result["j"].shape == (3, 1)
    assert np.isfinite(result["j"]).all()


def test_formula_runtime_bbi_uptrend():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=5), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"$close": [1, 5, 2, 4, 3, 3, 4, 2, 5, 1]}, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    result = runtime.compute_formulas({"up": "BBIUptrend(close, 3, 5, 0.0)"})

    np.testing.assert_array_equal(
        result["up"],
        np.array([
            [False, False],
            [False, False],
            [True, False],
            [True, False],
            [True, False],
        ]),
    )


def test_formula_runtime_atr_and_supertrend():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=6), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$high": [10, 20, 11, 19, 12, 18, 13, 17, 14, 16, 15, 15],
        "$low": [9, 19, 10, 18, 11, 17, 12, 16, 13, 15, 14, 14],
        "$close": [9.5, 19.5, 10.5, 18.5, 11.5, 17.5, 12.5, 16.5, 13.5, 15.5, 14.5, 14.5],
    }, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    result = runtime.compute_formulas({
        "atr2": "ATR(high, low, close, 2)",
        "st_line": "SuperTrend(high, low, close, 3, 2.0)",
        "st_dir": "SuperTrend(high, low, close, 3, 2.0, 'direction')",
    })

    assert result["atr2"].shape == (6, 2)
    assert result["st_line"].shape == (6, 2)
    assert result["st_dir"].shape == (6, 2)
    assert np.isfinite(result["atr2"]).all()
    assert set(np.unique(result["st_dir"])) <= {-1.0, 1.0}


def test_formula_runtime_recent_stable_signal():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=6), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({
        "$close": [10, 10, 10.1, 11, 10.05, 12, 10.08, 13, 9.8, 12, 9.7, 12],
        "$open": [10] * 12,
        "$high": [10] * 12,
        "$low": [10] * 12,
        "$volume": [1] * 12,
    }, index=index)
    runtime = FactorRuntime(MarketPanel.from_frame(frame))
    result = runtime.compute_formulas({
        "signal": "close == 10",
        "stable": "RecentStableSignal(signal, close, 4, 0.03, 3)",
    })

    np.testing.assert_array_equal(
        result["stable"],
        np.array([
            [False, False],
            [False, False],
            [False, False],
            [True, False],
            [True, False],
            [False, False],
        ]),
    )
