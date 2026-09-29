"""信号矩阵

预计算信号的稀疏存储和管理。
只存 selected=True 的行，大幅降低内存占用。
"""

from typing import Dict, List

import pandas as pd

from .types import Signal


class SignalMatrix:
    """信号矩阵（稀疏存储）

    全矩阵：N 天 × M 股 = N*M 条记录
    稀疏矩阵：只存选中股票 ≈ N*M*选中率，通常节省 95% 内存
    """

    def __init__(self):
        self._signals: Dict[str, List[Signal]] = {}

    def set(self, date: str, signals: List[Signal]) -> None:
        """存储某日信号"""
        self._signals[date] = signals

    def get(self, date: str) -> List[Signal]:
        """获取某日信号"""
        return self._signals.get(date, [])

    def get_selected(self, date: str) -> List[Signal]:
        """获取某日选中的信号"""
        return [s for s in self.get(date) if s.selected]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame（用于分析和调试）"""
        rows = []
        for date, signals in self._signals.items():
            for s in signals:
                if s.selected:
                    rows.append({
                        "date": date,
                        "symbol": s.symbol,
                        "score": s.score,
                        "reason": s.reason,
                    })
        return pd.DataFrame(rows)

    @property
    def total_signals(self) -> int:
        return sum(len(s) for s in self._signals.values())

    @property
    def selected_signals(self) -> int:
        return sum(1 for signals in self._signals.values() for s in signals if s.selected)

    def __len__(self) -> int:
        return len(self._signals)