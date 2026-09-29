"""板别管理

判断股票所属交易板别，返回对应的涨跌停阈值。
"""

from typing import Optional


class BoardManager:
    """A 股板别管理

    板别规则：
    | 板别   | 代码前缀    | 涨跌停幅度 | 说明 |
    |--------|------------|-----------|------|
    | 主板   | SH60/SZ00  | ±10%      | 沪深主板 |
    | 创业板 | SZ30       | ±20%      | ChiNext |
    | 科创板 | SH68       | ±20%      | STAR Market |
    | 北交所 | BJ         | ±30%      | 北京证券交易所 |
    | ST 股票 | 含 ST      | ±5%       | 风险警示板 |
    """

    @staticmethod
    def get_board(symbol: str) -> str:
        """返回股票板别：mainboard / chi_next / star / beijing / st"""
        if symbol.startswith("SH68"):
            return "star"
        elif symbol.startswith("SZ30"):
            return "chinet"
        elif symbol.startswith("BJ"):
            return "beijing"
        elif symbol.startswith("SH") or symbol.startswith("SZ"):
            return "mainboard"
        return "unknown"

    @staticmethod
    def get_limit_up_rate(symbol: str, is_st: Optional[bool] = None) -> float:
        """返回涨停幅度"""
        if is_st:
            return 0.05
        board = BoardManager.get_board(symbol)
        if board == "star":
            return 0.20
        elif board == "chinet":
            return 0.20
        elif board == "beijing":
            return 0.30
        else:
            return 0.10

    @staticmethod
    def get_limit_down_rate(symbol: str, is_st: Optional[bool] = None) -> float:
        """返回跌停幅度（与涨停幅度相同，方向相反）"""
        return BoardManager.get_limit_up_rate(symbol, is_st)

    @staticmethod
    def is_star_market(symbol: str) -> bool:
        """判断是否为科创板"""
        return symbol.startswith("SH68")

    @staticmethod
    def is_chi_next(symbol: str) -> bool:
        """判断是否为创业板"""
        return symbol.startswith("SZ30")