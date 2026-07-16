"""Anchored chronological walk-forward splitting with label purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchFold:
    fold_id: str
    train: pd.DataFrame
    validation: pd.DataFrame
    prediction: pd.DataFrame
    training_information_cutoff: pd.Timestamp


class AnchoredWalkForwardSplitter:
    def __init__(
        self,
        *,
        first_prediction_year: int,
        last_prediction_year: int,
        validation_sessions: int = 252,
        embargo_sessions: int = 0,
    ):
        if first_prediction_year > last_prediction_year:
            raise ValueError("first_prediction_year cannot exceed last_prediction_year")
        if validation_sessions < 1 or embargo_sessions < 0:
            raise ValueError("validation_sessions must be positive and embargo_sessions non-negative")
        self.first_prediction_year = first_prediction_year
        self.last_prediction_year = last_prediction_year
        self.validation_sessions = validation_sessions
        self.embargo_sessions = embargo_sessions

    def split(self, frame: pd.DataFrame) -> tuple[ResearchFold, ...]:
        if "signal_time" not in frame.index.names:
            raise ValueError("Research frame index must include signal_time")
        dates = pd.DatetimeIndex(frame.index.get_level_values("signal_time"))
        label_end = pd.to_datetime(frame["label_end_session"], errors="coerce")
        folds = []
        for year in range(self.first_prediction_year, self.last_prediction_year + 1):
            prediction_mask = dates.year == year
            if not prediction_mask.any():
                continue
            prediction_start = dates[prediction_mask].min()
            historical_sessions = sorted(set(dates[dates < prediction_start]))
            if len(historical_sessions) <= self.validation_sessions:
                continue
            validation_dates = historical_sessions[-self.validation_sessions :]
            validation_start = pd.Timestamp(validation_dates[0])
            training_dates = historical_sessions[: -self.validation_sessions]
            if self.embargo_sessions:
                training_dates = training_dates[: -self.embargo_sessions]
            train_cutoff = pd.Timestamp(training_dates[-1])
            train_mask = dates.isin(training_dates) & (label_end <= validation_start)
            validation_mask = dates.isin(validation_dates) & (label_end <= prediction_start)
            folds.append(
                ResearchFold(
                    fold_id=f"fold-{year}",
                    train=frame[train_mask].copy(),
                    validation=frame[validation_mask].copy(),
                    prediction=frame[prediction_mask].copy(),
                    training_information_cutoff=train_cutoff,
                )
            )
        return tuple(folds)
