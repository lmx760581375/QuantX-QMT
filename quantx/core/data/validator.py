"""数据校验

校验本地数据的完整性和正确性。
每个校验方法返回 (passed, message) 元组。
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """数据校验器"""

    @staticmethod
    def validate_ohlc_consistency(df: pd.DataFrame) -> Tuple[bool, str]:
        """验证 OHLC 价格关系：low <= min(open,close), high >= max(open,close)"""
        if df.empty:
            return True, "Empty DataFrame"

        errors = []
        # low <= open
        if "low" in df.columns and "open" in df.columns:
            bad = df["low"] > df["open"]
            if bad.any():
                errors.append(f"low > open: {bad.sum()} rows")

        # low <= close
        if "low" in df.columns and "close" in df.columns:
            bad = df["low"] > df["close"]
            if bad.any():
                errors.append(f"low > close: {bad.sum()} rows")

        # high >= open
        if "high" in df.columns and "open" in df.columns:
            bad = df["high"] < df["open"]
            if bad.any():
                errors.append(f"high < open: {bad.sum()} rows")

        # high >= close
        if "high" in df.columns and "close" in df.columns:
            bad = df["high"] < df["close"]
            if bad.any():
                errors.append(f"high < close: {bad.sum()} rows")

        if errors:
            return False, "; ".join(errors)
        return True, "OK"

    @staticmethod
    def validate_no_negative_prices(df: pd.DataFrame) -> Tuple[bool, str]:
        """验证价格非负"""
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                bad = df[col] < 0
                if bad.any():
                    return False, f"{col} has {bad.sum()} negative values"
        return True, "OK"

    @staticmethod
    def validate_volume_positive(df: pd.DataFrame) -> Tuple[bool, str]:
        """验证成交量非负"""
        if "volume" not in df.columns:
            return True, "No volume column"
        bad = df["volume"] < 0
        if bad.any():
            return False, f"volume has {bad.sum()} negative values"
        return True, "OK"

    @staticmethod
    def validate_no_sudden_jumps(
        df: pd.DataFrame, threshold: float = 0.5
    ) -> Tuple[bool, str]:
        """验证价格无异常跳变（单日涨跌幅 > 50% 视为异常，除权除息和涨停除外）

        注意：A 股涨跌停板最高 30%，超过 50% 的跳变通常是数据错误。
        """
        if "close" not in df.columns or "preclose" not in df.columns:
            return True, "No price columns"

        change = (df["close"] - df["preclose"]) / df["preclose"].replace(0, np.nan)
        bad = abs(change) > threshold
        if bad.any():
            bad_dates = df.loc[bad, "date"].tolist() if "date" in df.columns else []
            return False, f"{bad.sum()} abnormal jumps > {threshold:.0%}: {bad_dates[:5]}"
        return True, "OK"

    @staticmethod
    def validate_coverage(
        df: pd.DataFrame, expected_dates: List[str], min_ratio: float = 0.9
    ) -> Tuple[bool, str]:
        """验证数据覆盖范围"""
        if df.empty:
            return False, "Empty DataFrame"

        n_expected = len(expected_dates)
        if n_expected == 0:
            return True, "No expected dates"

        n_actual = len(df)
        coverage = n_actual / n_expected

        if coverage < min_ratio:
            return False, f"Coverage {coverage:.1%} < {min_ratio:.0%} ({n_actual}/{n_expected})"
        return True, f"Coverage {coverage:.1%} ({n_actual}/{n_expected})"

    @staticmethod
    def validate_missing_values(df: pd.DataFrame) -> Tuple[bool, str]:
        """验证关键字段的缺失值比例"""
        key_fields = ["open", "high", "low", "close", "volume"]
        issues = []
        for field in key_fields:
            if field in df.columns:
                missing = df[field].isna().mean()
                if missing > 0.1:
                    issues.append(f"{field}: {missing:.1%} missing")
        if issues:
            return False, "; ".join(issues)
        return True, "OK"

    @staticmethod
    def generate_report(
        df: pd.DataFrame, expected_dates: List[str]
    ) -> Dict:
        """生成完整校验报告"""
        checks = [
            ("OHLC 一致性", DataValidator.validate_ohlc_consistency(df)),
            ("价格非负", DataValidator.validate_no_negative_prices(df)),
            ("成交量非负", DataValidator.validate_volume_positive(df)),
            ("无异常跳变", DataValidator.validate_no_sudden_jumps(df)),
            ("覆盖范围", DataValidator.validate_coverage(df, expected_dates)),
            ("缺失值", DataValidator.validate_missing_values(df)),
        ]

        passed = sum(1 for _, (ok, _) in checks if ok)
        failed = len(checks) - passed

        return {
            "total_checks": len(checks),
            "passed": passed,
            "failed": failed,
            "details": [
                {"name": name, "passed": ok, "message": msg}
                for name, (ok, msg) in checks
            ],
        }