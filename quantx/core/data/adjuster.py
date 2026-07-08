"""复权处理

对股票价格进行前复权（forward adjustment）处理。
支持两种模式：
- origin: 仿射变换 a*price + b（精确，处理现金分红）
- equiv: 纯乘法调整 price * factor（简化，便于跨股票比较）

Phase 1 使用 BaoStock 的 adjustflag='2' 前复权数据，adjuster 作为备用模块。
Phase 2 接入其他数据源时启用。
"""

from typing import Tuple

import numpy as np
import pandas as pd


class Adjuster:
    """股票价格复权处理器"""

    @staticmethod
    def compute_affine_factors(dividend_df: pd.DataFrame) -> pd.DataFrame:
        """计算仿射变换因子 (a, b)

        Args:
            dividend_df: 除权除息数据，包含 interest(每股分红), stockBonus(送股),
                         stockGift(转增), allotNum(配股数), allotPrice(配股价)

        Returns:
            DataFrame with columns: a, b (affine transform factors)
        """
        factors = pd.DataFrame(index=dividend_df.index)
        factors["a"] = 1.0
        factors["b"] = 0.0

        for i in range(len(dividend_df)):
            row = dividend_df.iloc[i]
            # 送股 + 转增
            stock_ratio = (row.get("stockBonus", 0) + row.get("stockGift", 0)) / 10.0
            # 配股
            allot_ratio = row.get("allotNum", 0) / 10.0
            allot_price = row.get("allotPrice", 0)
            # 现金分红（每股）
            interest = row.get("interest", 0)

            # 前复权：a = (preclose - interest + allot_ratio * allot_price)
            #          / (preclose * (1 + stock_ratio + allot_ratio))
            # 这里的 a 和 b 是累积调整因子
            if i == 0:
                preclose = row.get("preclose", 1)
                if preclose and preclose > 0:
                    denom = preclose * (1 + stock_ratio + allot_ratio)
                    if denom > 0:
                        factors.iloc[i, 0] = (preclose - interest + allot_ratio * allot_price) / denom
                        factors.iloc[i, 1] = -interest * factors.iloc[i, 0]
            else:
                prev_a = factors.iloc[i - 1, 0]
                prev_b = factors.iloc[i - 1, 1]
                factors.iloc[i, 0] = prev_a * factors.iloc[i, 0] if i > 0 else factors.iloc[i, 0]
                factors.iloc[i, 1] = prev_a * factors.iloc[i, 1] + prev_b

        return factors

    @staticmethod
    def compute_multiplicative_factors(dividend_df: pd.DataFrame) -> np.ndarray:
        """计算纯乘法复权因子

        Args:
            dividend_df: 除权除息数据

        Returns:
            复权因子数组，每个元素是累积乘法因子
        """
        factors = np.ones(len(dividend_df))
        for i in range(len(dividend_df)):
            row = dividend_df.iloc[i]
            stock_ratio = (row.get("stockBonus", 0) + row.get("stockGift", 0)) / 10.0
            allot_ratio = row.get("allotNum", 0) / 10.0
            interest = row.get("interest", 0)
            preclose = row.get("preclose", 1)

            if preclose and preclose > 0:
                factor = (preclose - interest) / (preclose * (1 + stock_ratio + allot_ratio))
                if factor > 0:
                    factors[i] = factor

            if i > 0:
                factors[i] *= factors[i - 1]

        return factors

    @staticmethod
    def forward_adjust(
        df: pd.DataFrame, dividend_df: pd.DataFrame, mode: str = "origin"
    ) -> pd.DataFrame:
        """前复权处理

        Args:
            df: 原始 OHLCV 数据，按日期排序
            dividend_df: 除权除息数据
            mode: 复权模式 origin/equiv

        Returns:
            复权后的 DataFrame，新增 adj_open/adj_high/adj_low/adj_close 列
        """
        result = df.copy()

        if mode == "origin":
            affine_factors = Adjuster.compute_affine_factors(dividend_df)
            for col in ["open", "high", "low", "close"]:
                if col in result.columns:
                    result[f"adj_{col}"] = result[col] * affine_factors["a"].values[0] + affine_factors["b"].values[0]
        elif mode == "equiv":
            mult_factors = Adjuster.compute_multiplicative_factors(dividend_df)
            for col in ["open", "high", "low", "close"]:
                if col in result.columns:
                    result[f"adj_{col}"] = result[col] * mult_factors[0]
        else:
            raise ValueError(f"Unknown adjust mode: {mode}")

        return result

    @staticmethod
    def verify_consistency(
        df_origin: pd.DataFrame, df_equiv: pd.DataFrame, tolerance: float = 0.01
    ) -> Tuple[bool, str]:
        """验证两种复权模式的一致性

        Args:
            df_origin: origin 模式复权结果
            df_equiv: equiv 模式复权结果
            tolerance: 允许的最大偏差比例

        Returns:
            (passed, message)
        """
        for col in ["adj_open", "adj_high", "adj_low", "adj_close"]:
            if col in df_origin.columns and col in df_equiv.columns:
                ratio = df_origin[col] / df_equiv[col].replace(0, np.nan)
                if ratio.std() > tolerance:
                    return False, f"{col} ratio std {ratio.std():.4f} > {tolerance}"
        return True, "OK"