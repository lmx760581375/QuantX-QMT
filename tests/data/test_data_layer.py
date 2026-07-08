"""数据层测试"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantx.core.data import (
    TradingCalendar,
    Adjuster,
    LocalDataRepository,
    DataValidator,
    DataCache,
    BaoStockClient,
)


# ============================================================
# TradingCalendar Tests
# ============================================================

class TestTradingCalendar:
    def test_create_from_dates(self):
        """从日期列表创建日历"""
        dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
        cal = TradingCalendar(dates=dates)
        assert len(cal) == 3

    def test_is_trading_day(self):
        cal = TradingCalendar(dates=["2020-01-02", "2020-01-03", "2020-01-06"])
        assert cal.is_trading_day("2020-01-02") is True
        assert cal.is_trading_day("2020-01-04") is False  # 周六

    def test_get_trading_days(self):
        cal = TradingCalendar(dates=["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        days = cal.get_trading_days("2020-01-02", "2020-01-06")
        assert days == ["2020-01-02", "2020-01-03", "2020-01-06"]

    def test_next_prev_trading_day(self):
        cal = TradingCalendar(dates=["2020-01-02", "2020-01-03", "2020-01-06"])
        assert cal.get_next_trading_day("2020-01-02") == "2020-01-03"
        assert cal.get_prev_trading_day("2020-01-06") == "2020-01-03"

    def test_count_trading_days(self):
        cal = TradingCalendar(dates=["2020-01-02", "2020-01-03", "2020-01-06"])
        assert cal.count_trading_days("2020-01-02", "2020-01-06") == 3


# ============================================================
# Adjuster Tests
# ============================================================

class TestAdjuster:
    def test_forward_adjust_equiv(self):
        """测试 equiv 模式复权"""
        # 模拟数据：10 天价格，第 5 天除权
        df = pd.DataFrame({
            "open": [10.0] * 10,
            "high": [11.0] * 10,
            "low": [9.0] * 10,
            "close": [10.5] * 10,
        })
        dividend_df = pd.DataFrame({
            "preclose": [10.0],
            "stockBonus": [0],  # 不送股
            "stockGift": [0],
            "allotNum": [0],
            "allotPrice": [0],
            "interest": [0.5],  # 每股分红 0.5 元
        })

        result = Adjuster.forward_adjust(df, dividend_df, mode="equiv")
        assert "adj_close" in result.columns
        # 复权后价格应小于原始价格（现金分红）
        assert (result["adj_close"] <= result["close"]).all()

    def test_no_adjustment_needed(self):
        """无除权除息时，复权后价格应等于原始价格"""
        df = pd.DataFrame({
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
        })
        dividend_df = pd.DataFrame({
            "preclose": [10.0],
            "stockBonus": [0],
            "stockGift": [0],
            "allotNum": [0],
            "allotPrice": [0],
            "interest": [0.0],
        })

        result = Adjuster.forward_adjust(df, dividend_df, mode="equiv")
        np.testing.assert_array_almost_equal(result["adj_close"], result["close"])


# ============================================================
# Repository Tests
# ============================================================

class TestLocalDataRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        return LocalDataRepository(str(tmp_path / "test_data"))

    def test_save_and_load(self, repo):
        df = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03"],
            "code": ["SH600519", "SH600519"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 9.5],
            "close": [10.5, 11.0],
            "preclose": [9.8, 10.5],
            "volume": [1000000, 1200000],
            "amount": [10500000, 12600000],
            "turnover": [0.01, 0.012],
            "pct_chg": [0.05, 0.048],
            "is_st": [0, 0],
        })
        repo.save_symbol("SH600519", df)
        loaded = repo.load_symbol("SH600519")
        assert len(loaded) == 2

    def test_upsert_dedup(self, repo):
        df1 = pd.DataFrame({
            "date": ["2020-01-02"],
            "code": ["SH600519"],
            "open": [10.0], "high": [11.0], "low": [9.0],
            "close": [10.5], "preclose": [9.8],
            "volume": [1000000], "amount": [10500000],
            "turnover": [0.01], "pct_chg": [0.05], "is_st": [0],
        })
        df2 = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03"],
            "code": ["SH600519", "SH600519"],
            "open": [10.2, 10.8], "high": [11.2, 11.8], "low": [9.2, 9.8],
            "close": [10.7, 11.3], "preclose": [10.0, 10.7],
            "volume": [1100000, 1300000], "amount": [11550000, 13780000],
            "turnover": [0.011, 0.013], "pct_chg": [0.07, 0.056], "is_st": [0, 0],
        })
        repo.save_symbol("SH600519", df1)
        repo.save_symbol("SH600519", df2)  # 第二次保存应覆盖 2020-01-02
        loaded = repo.load_symbol("SH600519")
        assert len(loaded) == 2  # 只有 2 行（去重后）
        # 2020-01-02 的数据应该是 df2 的（新数据覆盖旧数据）
        assert loaded[loaded["date"] == "2020-01-02"]["open"].values[0] == 10.2

    def test_last_date(self, repo):
        df = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "code": ["SH600519"] * 3,
            "open": [10.0] * 3, "high": [11.0] * 3, "low": [9.0] * 3,
            "close": [10.5] * 3, "preclose": [9.8] * 3,
            "volume": [1000000] * 3, "amount": [10500000] * 3,
            "turnover": [0.01] * 3, "pct_chg": [0.05] * 3, "is_st": [0] * 3,
        })
        repo.save_symbol("SH600519", df)
        assert repo.get_last_date("SH600519") == "2020-01-06"
        assert repo.get_first_date("SH600519") == "2020-01-02"


# ============================================================
# Validator Tests
# ============================================================

class TestDataValidator:
    def test_ohlc_consistency_valid(self):
        df = pd.DataFrame({
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 9.5],
            "close": [10.5, 11.0],
        })
        ok, msg = DataValidator.validate_ohlc_consistency(df)
        assert ok, msg

    def test_ohlc_consistency_invalid(self):
        df = pd.DataFrame({
            "open": [10.0, 10.5],
            "high": [11.0, 9.0],  # high < open on second row
            "low": [9.0, 9.5],
            "close": [10.5, 11.0],
        })
        ok, msg = DataValidator.validate_ohlc_consistency(df)
        assert not ok

    def test_negative_prices(self):
        df = pd.DataFrame({
            "open": [10.0, -1.0],
            "high": [11.0, 12.0],
            "low": [9.0, 8.0],
            "close": [10.5, 11.0],
        })
        ok, msg = DataValidator.validate_no_negative_prices(df)
        assert not ok

    def test_sudden_jumps(self):
        df = pd.DataFrame({
            "close": [10.0, 20.0],  # 100% jump
            "preclose": [9.0, 10.0],
        })
        ok, msg = DataValidator.validate_no_sudden_jumps(df, threshold=0.5)
        assert not ok


# ============================================================
# Cache Tests
# ============================================================

class TestDataCache:
    def test_save_and_load(self, tmp_path):
        cache = DataCache(str(tmp_path / "cache"))
        df = pd.DataFrame({"a": [1, 2, 3]})
        symbols = ["SH600519", "SZ000001"]

        cache.save(df, "2020-01-01", "2020-12-31", "forward", symbols)
        assert cache.exists("2020-01-01", "2020-12-31", "forward", symbols)

        loaded = cache.load("2020-01-01", "2020-12-31", "forward", symbols)
        assert loaded is not None
        assert len(loaded) == 3

    def test_invalidate(self, tmp_path):
        cache = DataCache(str(tmp_path / "cache"))
        df = pd.DataFrame({"a": [1]})
        symbols = ["SH600519"]

        cache.save(df, "2020-01-01", "2020-12-31", "forward", symbols)
        cache.invalidate("2020-01-01", "2020-12-31", "forward", symbols)
        assert not cache.exists("2020-01-01", "2020-12-31", "forward", symbols)


# ============================================================
# Symbol Normalization Tests
# ============================================================

class TestSymbolNormalization:
    def test_baostock_to_qlib(self):
        assert BaoStockClient._normalize_symbol("sh.600519") == "SH600519"
        assert BaoStockClient._normalize_symbol("sz.000001") == "SZ000001"
        assert BaoStockClient._normalize_symbol("sh.688001") == "SH688001"
        assert BaoStockClient._normalize_symbol("sz.300750") == "SZ300750"

    def test_qlib_to_baostock(self):
        assert BaoStockClient.to_baostock_format("SH600519") == "sh.600519"
        assert BaoStockClient.to_baostock_format("SZ000001") == "sz.000001"