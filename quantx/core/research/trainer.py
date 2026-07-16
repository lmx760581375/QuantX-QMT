"""Trainer protocols and deterministic tabular baselines."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .artifacts import LightGBMModelArtifact, LinearModelArtifact, TorchModelArtifact


class Trainer(Protocol):
    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
        *,
        seed: int,
        validation: pd.DataFrame | None = None,
    ): ...

    def describe(self) -> dict: ...


class RidgeTrainer:
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("Ridge alpha cannot be negative")
        self.alpha = float(alpha)

    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
        *,
        seed: int,
        validation: pd.DataFrame | None = None,
    ) -> LinearModelArtifact:
        del seed, validation
        if train.empty:
            raise ValueError("Cannot train Ridge on an empty dataset")
        x = train.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
        y = pd.to_numeric(train[label_column], errors="coerce").to_numpy(dtype=float)
        valid_y = np.isfinite(y)
        x = x[valid_y]
        y = y[valid_y]
        if len(y) == 0:
            raise ValueError("Ridge training labels are all missing")
        medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        missing = ~np.isfinite(x)
        if missing.any():
            x[missing] = np.take(medians, np.where(missing)[1])
        means = x.mean(axis=0)
        scales = x.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        normalized = (x - means) / scales
        y_mean = float(y.mean())
        centered_y = y - y_mean
        penalty = np.eye(normalized.shape[1]) * self.alpha
        coefficients = np.linalg.solve(normalized.T @ normalized + penalty, normalized.T @ centered_y)
        return LinearModelArtifact(
            feature_names=tuple(feature_columns),
            coefficients=tuple(float(value) for value in coefficients),
            intercept=y_mean,
            medians=tuple(float(value) for value in medians),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
        )

    def describe(self) -> dict:
        return {"type": "ridge", "alpha": self.alpha, "format": "quantx_linear_json"}


class LightGBMTrainer:
    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
        *,
        seed: int,
        validation: pd.DataFrame | None = None,
    ) -> LightGBMModelArtifact:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm optional dependency is not installed") from exc
        if train.empty:
            raise ValueError("Cannot train LightGBM on an empty dataset")
        x_train, medians = _imputed_values(train, feature_columns)
        y_train = pd.to_numeric(train[label_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_train)
        params = {"random_state": seed, "verbosity": -1, **self.params}
        model = LGBMRegressor(**params)
        fit_kwargs = {}
        if validation is not None and not validation.empty:
            x_validation, _ = _imputed_values(validation, feature_columns, medians=medians)
            y_validation = pd.to_numeric(validation[label_column], errors="coerce").to_numpy(dtype=float)
            valid_validation = np.isfinite(y_validation)
            if valid_validation.any():
                fit_kwargs["eval_set"] = [(x_validation[valid_validation], y_validation[valid_validation])]
        model.fit(x_train[valid], y_train[valid], **fit_kwargs)
        return LightGBMModelArtifact(
            model=model,
            feature_names=feature_columns,
            medians=tuple(float(value) for value in medians),
        )

    def describe(self) -> dict:
        return {"type": "lightgbm", "params": self.params, "format": "lightgbm_text"}


def _imputed_values(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    if medians is None:
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    return values, medians


class TorchTrainer:
    def __init__(
        self,
        *,
        architecture: str = "mlp",
        lookback_sessions: int = 20,
        hidden_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.001,
    ):
        if architecture not in {"mlp", "lstm"}:
            raise ValueError("Torch architecture must be mlp or lstm")
        self.architecture = architecture
        self.lookback_sessions = int(lookback_sessions)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)

    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
        *,
        seed: int,
        validation: pd.DataFrame | None = None,
    ) -> TorchModelArtifact:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch optional dependency is not installed") from exc
        from .artifacts import _normalize_windows
        from .sequence import RawWindowFeatureBuilder

        torch.manual_seed(seed)
        np.random.seed(seed)
        builder = RawWindowFeatureBuilder(self.lookback_sessions)
        batch = builder.transform(train, feature_columns)
        numeric = train.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
        medians = numeric.median().fillna(0.0).to_numpy()
        means = numeric.mean().fillna(0.0).to_numpy()
        scales = numeric.std(ddof=0).fillna(1.0).to_numpy()
        scales = np.where(scales > 1e-12, scales, 1.0)
        x = _normalize_windows(batch.values, medians, means, scales)
        y = pd.to_numeric(train[label_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y) & batch.valid_mask.all(axis=1)
        if not valid.any():
            raise ValueError("Torch training fold has no complete causal windows")
        model = _build_torch_model(
            torch,
            architecture=self.architecture,
            lookback_sessions=self.lookback_sessions,
            feature_count=len(feature_columns),
            hidden_size=self.hidden_size,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        x_tensor = torch.tensor(x[valid], dtype=torch.float32)
        y_tensor = torch.tensor(y[valid], dtype=torch.float32).reshape(-1, 1)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        history_source = train if validation is None else pd.concat([train, validation]).sort_index()
        history = _tail_by_instrument(history_source, self.lookback_sessions - 1)
        return TorchModelArtifact(
            model=model,
            architecture=self.architecture,
            feature_names=feature_columns,
            lookback_sessions=self.lookback_sessions,
            medians=tuple(float(value) for value in medians),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            hidden_size=self.hidden_size,
            history=history,
        )

    def describe(self) -> dict:
        return {
            "type": "torch",
            "architecture": self.architecture,
            "lookback_sessions": self.lookback_sessions,
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "format": "torch_state_dict",
        }


def _build_torch_model(torch, *, architecture, lookback_sessions, feature_count, hidden_size):
    if architecture == "mlp":
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(lookback_sessions * feature_count, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    class LSTMRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(feature_count, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values):
            sequence, _ = self.lstm(values)
            return self.output(sequence[:, -1, :])

    return LSTMRegressor()


def _tail_by_instrument(frame: pd.DataFrame, sessions: int) -> pd.DataFrame:
    if sessions <= 0 or frame.empty:
        return frame.iloc[0:0]
    parts = []
    for instrument in frame.index.get_level_values("instrument").unique():
        part = frame.xs(instrument, level="instrument", drop_level=False).sort_index().tail(sessions)
        parts.append(part)
    return pd.concat(parts).sort_index() if parts else frame.iloc[0:0]
