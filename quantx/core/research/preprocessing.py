"""Fold-local preprocessing fitted exclusively on training observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelWinsorizer:
    lower_quantile: float
    upper_quantile: float
    lower_value: float
    upper_value: float

    @classmethod
    def fit(
        cls,
        labels: pd.Series,
        quantiles: tuple[float, float],
    ) -> "LabelWinsorizer":
        lower_quantile, upper_quantile = quantiles
        numeric = pd.to_numeric(labels, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if numeric.empty:
            raise ValueError("Cannot fit label winsorization without finite training labels")
        lower_value, upper_value = numeric.quantile([lower_quantile, upper_quantile]).to_numpy()
        return cls(
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
            lower_value=float(lower_value),
            upper_value=float(upper_value),
        )

    def transform(self, frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        transformed = frame.copy()
        transformed[label_column] = pd.to_numeric(transformed[label_column], errors="coerce").clip(
            self.lower_value, self.upper_value
        )
        return transformed

    def describe(self) -> dict:
        return {"type": "label_winsorizer", **asdict(self)}
