"""Reward 模型共享的数据、特征与网络组件。"""

from __future__ import annotations

import ast
import math
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "workdirs" / "reward"
KRONOS_EXPERIMENT_ROOT = REPO_ROOT / "workdirs" / "feature_store"

from models.reward.feature_store.kronos_memmap_v1.config import (  # noqa: E402
    MARKET_FEATURE_FIELDS,
    RAW_OHLCV_FIELDS,
    RAW_STATE_FIELDS,
    RAW_TRADE_FIELDS,
    SEQ_LEN,
    DatasetPaths,
)
from models.reward.feature_store.kronos_memmap_v1.execution import ExecutionRulesV1  # noqa: E402
from models.reward.feature_store.kronos_memmap_v1.storage import read_json, write_json as write_json  # noqa: E402
from models.reward.feature_store.kronos_memmap_v1.training import (  # noqa: E402
    FeatureScaler,
    load_scaler as load_scaler,
    normalize,
    robust_stats,
    save_scaler as save_scaler,
)


DEFAULT_WTS_CONFIG = (
    REPO_ROOT
    / "configs"
    / "strategies"
    / "weak_to_strong.yaml"
)
DEFAULT_BASELINE_PATH = REPO_ROOT / "workdirs" / "market_baseline.parquet"
MAINBOARD_BOARDS = {"mainboard_sh", "mainboard_sz"}
SCORE_PROB_THRESHOLDS = (0.05, 0.08, 0.10, 0.15, 0.20)
TARGET_FILES = {
    "relative": "target_relative_path_float32.mmap",
    "abs": "target_abs_path_float32.mmap",
    "baseline": "target_baseline_path_float32.mmap",
}
CANDIDATE_FEATURE_FIELDS = (
    "wts_rank_pct",
    "wts_inv_rank",
    "wts_score_z",
    "wts_score_pct",
    "wts_selector_where",
    "wts_rank_in_topn_pct",
)
PATH_QUALITY_WEIGHTS = (0.50, 0.30, 0.20)
PATH_QUALITY_VOLATILITY_EPSILON = 1.0e-6


@dataclass(frozen=True)
class WTSPaths:
    root: Path = EXPERIMENT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def folds(self) -> Path:
        return self.data / "folds"

    @property
    def scalers(self) -> Path:
        return self.data / "scalers"

    @property
    def candidate_index(self) -> Path:
        return self.data / "candidate_index_v1.parquet"

    @property
    def manifest(self) -> Path:
        return self.data / "candidate_dataset_manifest.json"

    @property
    def target_meta(self) -> Path:
        return self.data / "target_paths_meta.json"

    def target_path(self, kind: str) -> Path:
        return self.data / TARGET_FILES[kind]


@dataclass(frozen=True)
class ModelConfig:
    lookback: int = SEQ_LEN
    horizon: int = 30
    d_stock: int = 48
    d_market: int = 32
    d_candidate: int = 0
    d_model: int = 64
    candidate_dim: int = 0
    latent_tokens: int = 8
    latent_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    generative_hidden_dim: int = 128
    diffusion_steps: int = 100
    flow_steps: int = 20
    inference_steps: int = 0
    generative_objective: str = "flow"
    denoiser_type: str = "mlp"
    dit_d_model: int = 384
    dit_layers: int = 6
    dit_heads: int = 8
    dit_mlp_ratio: float = 3.65
    dit_dropout: float = 0.1


@dataclass(frozen=True)
class SourceMemmaps:
    paths: DatasetPaths
    calendar: pd.DataFrame
    instruments: pd.DataFrame
    raw_meta: dict[str, Any]
    trade_meta: dict[str, Any]
    state_meta: dict[str, Any]
    market_meta: dict[str, Any]
    feature_manifest: dict[str, Any]
    raw_ohlcv: np.memmap
    raw_trade: np.memmap
    raw_state: np.memmap
    market: np.memmap


def source_memmaps(kronos_root: str | Path) -> SourceMemmaps:
    paths = DatasetPaths(Path(kronos_root).resolve())
    calendar_pickle = paths.calendar.with_suffix(".pkl")
    instruments_pickle = paths.instruments.with_suffix(".pkl")
    calendar = (
        pd.read_pickle(calendar_pickle)
        if calendar_pickle.is_file()
        else pd.read_parquet(paths.calendar)
    )
    instruments = (
        pd.read_pickle(instruments_pickle)
        if instruments_pickle.is_file()
        else pd.read_parquet(paths.instruments)
    )
    raw_meta = read_json(paths.raw_ohlcv_meta)
    trade_meta = read_json(paths.raw_trade_meta)
    state_meta = read_json(paths.raw_state_meta)
    market_meta = read_json(paths.market_meta)
    feature_manifest = read_json(paths.features_manifest)
    n_inst, n_dates, _ = (int(v) for v in raw_meta["shape"])
    raw_ohlcv = np.memmap(paths.raw_ohlcv, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
    raw_trade = np.memmap(paths.raw_trade, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)))
    raw_state = np.memmap(paths.raw_state, dtype=np.uint8, mode="r", shape=(n_inst, n_dates, len(RAW_STATE_FIELDS)))
    market = np.memmap(paths.market, dtype=np.float32, mode="r", shape=(n_dates, len(MARKET_FEATURE_FIELDS)))
    return SourceMemmaps(
        paths=paths,
        calendar=calendar,
        instruments=instruments,
        raw_meta=raw_meta,
        trade_meta=trade_meta,
        state_meta=state_meta,
        market_meta=market_meta,
        feature_manifest=feature_manifest,
        raw_ohlcv=raw_ohlcv,
        raw_trade=raw_trade,
        raw_state=raw_state,
        market=market,
    )


def load_shard_lookup(feature_manifest: dict[str, Any]) -> dict[int, tuple[int, int]]:
    lookup: dict[int, tuple[int, int]] = {}
    for shard in feature_manifest["shards"]:
        shard_id = int(shard["shard_id"])
        start = int(shard["instrument_start"])
        end = int(shard["instrument_end"])
        for inst_idx in range(start, end):
            lookup[inst_idx] = (shard_id, inst_idx - start)
    return lookup


def date_indexer(calendar: pd.DataFrame) -> dict[str, int]:
    return {str(row.date): int(row.date_idx) for row in calendar.itertuples(index=False)}


def model_config_from_size(model_size: str, *, objective: str = "flow") -> ModelConfig:
    if model_size == "ae_wide_08":
        return ModelConfig(
            d_stock=64,
            d_market=32,
            d_candidate=16,
            d_model=96,
            candidate_dim=len(CANDIDATE_FEATURE_FIELDS),
            latent_tokens=27,
            latent_dim=128,
            n_heads=4,
            n_layers=4,
            generative_hidden_dim=3200,
            generative_objective=objective,
        )
    if model_size == "50m":
        return ModelConfig(
            d_stock=128,
            d_market=96,
            d_candidate=32,
            d_model=224,
            candidate_dim=len(CANDIDATE_FEATURE_FIELDS),
            latent_tokens=4,
            latent_dim=64,
            n_heads=4,
            n_layers=5,
            generative_hidden_dim=3200,
            generative_objective=objective,
        )
    if model_size == "base":
        return ModelConfig(
            d_stock=64,
            d_market=48,
            d_model=96,
            latent_tokens=8,
            latent_dim=96,
            n_heads=4,
            n_layers=3,
            generative_hidden_dim=256,
            generative_objective=objective,
        )
    if model_size == "large":
        return ModelConfig(
            d_stock=80,
            d_market=64,
            d_model=128,
            latent_tokens=10,
            latent_dim=96,
            n_heads=4,
            n_layers=4,
            generative_hidden_dim=384,
            generative_objective=objective,
        )
    return ModelConfig(generative_objective=objective)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch runtime")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {value!r}; expected one of cpu, cuda, mps, auto")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch runtime")
    return torch.device(value)


def target_memmap(paths: WTSPaths, kind: str, *, mode: str = "r") -> np.memmap:
    meta = read_json(paths.target_meta)
    shape = tuple(int(v) for v in meta["shape"])
    return np.memmap(paths.target_path(kind), dtype=np.float32, mode=mode, shape=shape)


def future_path_metrics(
    future_returns: np.ndarray,
    *,
    horizon: int | None = None,
    volatility_epsilon: float = PATH_QUALITY_VOLATILITY_EPSILON,
) -> dict[str, np.ndarray]:
    """Compute endpoint and path-risk metrics from close-to-close cumulative returns.

    The experiment target stores ``close(T+1..T+H) / close(T+1) - 1``.
    Therefore its adjacent log differences are the observable close-to-close
    returns from T+2 through T+H; no unobserved entry-day return is invented.
    """

    values = np.asarray(future_returns, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"future_returns must be [samples,horizon], got shape={tuple(values.shape)}")
    use_horizon = int(values.shape[1] if horizon is None else horizon)
    if use_horizon < 3:
        raise ValueError("Path-risk metrics require a horizon of at least 3 close observations")
    if use_horizon > int(values.shape[1]):
        raise ValueError(f"horizon={use_horizon} exceeds future return width={values.shape[1]}")
    if float(volatility_epsilon) <= 0.0:
        raise ValueError("volatility_epsilon must be positive")

    path = values[:, :use_horizon]
    nav = 1.0 + path
    valid = np.all(np.isfinite(path) & (nav > 0.0), axis=1)
    safe_nav = np.where(np.isfinite(nav) & (nav > 0.0), nav, 1.0)
    log_nav = np.log(safe_nav)
    daily_log_returns = np.diff(log_nav, axis=1)
    daily_mean = daily_log_returns.mean(axis=1)
    daily_std = daily_log_returns.std(axis=1, ddof=1)
    downside_deviation = np.sqrt(np.square(np.minimum(daily_log_returns, 0.0)).mean(axis=1))
    annualizer = math.sqrt(252.0)
    path_sharpe = annualizer * daily_mean / np.maximum(daily_std, float(volatility_epsilon))
    path_sortino = annualizer * daily_mean / np.maximum(downside_deviation, float(volatility_epsilon))
    running_peak = np.maximum.accumulate(safe_nav, axis=1)
    max_drawdown = (safe_nav / running_peak - 1.0).min(axis=1)

    result = {name: np.full(len(path), np.nan, dtype=np.float32) for name in ("terminal_return", "path_sharpe", "path_sortino", "max_drawdown")}
    result["terminal_return"][valid] = path[valid, -1].astype(np.float32, copy=False)
    result["path_sharpe"][valid] = path_sharpe[valid].astype(np.float32, copy=False)
    result["path_sortino"][valid] = path_sortino[valid].astype(np.float32, copy=False)
    result["max_drawdown"][valid] = max_drawdown[valid].astype(np.float32, copy=False)
    return result


def future_path_metrics_from_target_rows(
    targets: np.ndarray,
    target_rows: np.ndarray,
    *,
    horizon: int,
    chunk_size: int = 131_072,
) -> dict[str, np.ndarray]:
    """Read target memmap rows in bounded chunks before computing path metrics."""

    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError(f"target_rows must be one-dimensional, got shape={tuple(rows.shape)}")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(horizon) > int(targets.shape[1]):
        raise ValueError(f"horizon={horizon} exceeds target width={targets.shape[1]}")
    result = {name: np.empty(len(rows), dtype=np.float32) for name in ("terminal_return", "path_sharpe", "path_sortino", "max_drawdown")}
    for start in range(0, len(rows), int(chunk_size)):
        end = min(start + int(chunk_size), len(rows))
        metrics = future_path_metrics(np.asarray(targets[rows[start:end], : int(horizon)], dtype=np.float32), horizon=int(horizon))
        for name, values in metrics.items():
            result[name][start:end] = values
    return result


def cross_sectional_percentile_rank(values: np.ndarray, signal_dates: np.ndarray | pd.Series) -> np.ndarray:
    """Average-tie percentile rank inside each signal-date cross-section."""

    numeric = np.asarray(values, dtype=np.float64)
    dates = np.asarray(signal_dates).astype(str)
    if numeric.ndim != 1 or dates.ndim != 1 or len(numeric) != len(dates):
        raise ValueError("values and signal_dates must be matching one-dimensional arrays")
    frame = pd.DataFrame({"signal_date": dates, "value": numeric})
    return frame.groupby("signal_date", sort=False)["value"].rank(method="average", pct=True).to_numpy(dtype=np.float32)


def path_quality_from_metrics(
    *,
    signal_dates: np.ndarray | pd.Series,
    terminal_return: np.ndarray,
    path_sharpe: np.ndarray,
    max_drawdown: np.ndarray,
    weights: tuple[float, float, float] = PATH_QUALITY_WEIGHTS,
) -> dict[str, np.ndarray]:
    """Build a same-date return, Sharpe, and drawdown quality label."""

    if len(weights) != 3:
        raise ValueError("quality weights must contain return, Sharpe, and drawdown values")
    normalized_weights = tuple(float(value) for value in weights)
    if any(not np.isfinite(value) or value < 0.0 for value in normalized_weights):
        raise ValueError("quality weights must be finite and non-negative")
    if not math.isclose(sum(normalized_weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("quality weights must sum to 1")

    return_percentile = cross_sectional_percentile_rank(terminal_return, signal_dates)
    sharpe_percentile = cross_sectional_percentile_rank(path_sharpe, signal_dates)
    # ``max_drawdown`` is stored in its conventional signed form (0 for no
    # drawdown, negative otherwise), so a larger value means less drawdown.
    drawdown_percentile = cross_sectional_percentile_rank(np.asarray(max_drawdown, dtype=np.float64), signal_dates)
    quality_score = (
        normalized_weights[0] * return_percentile
        + normalized_weights[1] * sharpe_percentile
        + normalized_weights[2] * drawdown_percentile
    ).astype(np.float32, copy=False)
    invalid = ~(np.isfinite(return_percentile) & np.isfinite(sharpe_percentile) & np.isfinite(drawdown_percentile))
    quality_score[invalid] = np.nan
    return {
        "return_percentile": return_percentile,
        "sharpe_percentile": sharpe_percentile,
        "drawdown_percentile": drawdown_percentile,
        "quality_score": quality_score,
    }


class WeakToStrongPathDataset(Dataset):
    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        scaler: FeatureScaler | None,
        target_kind: str | None = "relative",
        input_mode: str = "raw_relative",
        max_open_shards: int = 2,
    ):
        if target_kind not in {"relative", "abs", None}:
            raise ValueError("target_kind must be relative, abs, or None")
        if input_mode not in {"raw_relative", "relative_only"}:
            raise ValueError("input_mode must be raw_relative or relative_only")
        self.source = source_memmaps(kronos_root)
        self.paths = WTSPaths(Path(experiment_root).resolve())
        self.rows = rows.reset_index(drop=True)
        self.scaler = scaler
        self.target_kind = target_kind
        self.input_mode = input_mode
        self.max_open_shards = int(max_open_shards)
        self.targets = (
            target_memmap(
                self.paths,
                "relative" if target_kind == "relative" else "abs",
                mode="r",
            )
            if target_kind is not None
            else None
        )
        self._shards: OrderedDict[int, np.memmap] = OrderedDict()
        self._shard_meta = {int(item["shard_id"]): item for item in self.source.feature_manifest["shards"]}
        self._sample_read_seconds: deque[float] = deque(maxlen=100_000)
        self._shard_misses = 0
        self._shard_evictions = 0
        self.instrument_idx = self.rows["instrument_idx"].to_numpy(dtype=np.int64)
        self.shard_id = self.rows["shard_id"].to_numpy(dtype=np.int64)
        self.local_instrument_idx = self.rows["local_instrument_idx"].to_numpy(dtype=np.int64)
        self.date_idx = self.rows["date_idx"].to_numpy(dtype=np.int64)
        self.target_row = self.rows["target_row"].to_numpy(dtype=np.int64)
        self.candidate_features = build_candidate_features(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def get_feature_shard(self, shard_id: int) -> np.memmap:
        shard_id = int(shard_id)
        if shard_id in self._shards:
            mmap = self._shards.pop(shard_id)
            self._shards[shard_id] = mmap
            return mmap
        meta = self._shard_meta[shard_id]
        self._shard_misses += 1
        mmap = np.memmap(meta["path"], dtype=np.float16, mode="r", shape=tuple(int(v) for v in meta["shape"]))
        self._shards[shard_id] = mmap
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
            self._shard_evictions += 1
        return mmap

    def get_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inst = int(self.instrument_idx[index])
        shard_id = int(self.shard_id[index])
        local_inst = int(self.local_instrument_idx[index])
        t = int(self.date_idx[index])
        start = t - SEQ_LEN + 1
        if start < 0:
            raise IndexError(f"Sample date_idx={t} cannot provide seq_len={SEQ_LEN}")
        x_raw = np.concatenate(
            (
                self.source.raw_ohlcv[inst, start : t + 1, :],
                self.source.raw_trade[inst, start : t + 1, :],
            ),
            axis=-1,
        ).astype(np.float32)
        x_derived = np.asarray(self.get_feature_shard(shard_id)[local_inst, start : t + 1, :], dtype=np.float32)
        x_market = np.asarray(self.source.market[start : t + 1, :], dtype=np.float32)
        x_ts = np.concatenate((x_raw, x_derived), axis=-1)
        y = (
            np.asarray(self.targets[int(self.target_row[index])], dtype=np.float32)
            if self.targets is not None
            else np.empty(0, dtype=np.float32)
        )
        x_candidate = np.asarray(self.candidate_features[index], dtype=np.float32)
        return x_ts, x_market, x_candidate, y

    def __getitem__(self, index: int):
        started = time.perf_counter()
        x_ts, x_market, x_candidate, y = self.get_arrays(index)
        raw_dim = len(RAW_OHLCV_FIELDS) + len(RAW_TRADE_FIELDS)
        if self.scaler is not None:
            raw = normalize(
                x_ts[:, :raw_dim],
                self.scaler.raw_clip_low,
                self.scaler.raw_clip_high,
                self.scaler.raw_mean,
                self.scaler.raw_std,
            )
            derived = normalize(
                x_ts[:, raw_dim:],
                self.scaler.derived_clip_low,
                self.scaler.derived_clip_high,
                self.scaler.derived_mean,
                self.scaler.derived_std,
            )
            x_ts = np.concatenate((raw, derived), axis=-1)
            x_market = normalize(
                x_market,
                self.scaler.market_clip_low,
                self.scaler.market_clip_high,
                self.scaler.market_mean,
                self.scaler.market_std,
            )
        if self.input_mode == "relative_only":
            x_ts = x_ts[:, raw_dim:]
        self._sample_read_seconds.append(time.perf_counter() - started)
        return (
            torch.from_numpy(np.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            torch.from_numpy(np.nan_to_num(x_market, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            torch.from_numpy(np.nan_to_num(x_candidate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            torch.from_numpy(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
        )

    def reset_io_stats(self) -> None:
        self._shard_misses = 0
        self._shard_evictions = 0
        self._sample_read_seconds.clear()

    def io_stats(self) -> dict[str, float | int]:
        samples = np.asarray(self._sample_read_seconds, dtype=float)
        return {
            "shard_misses": int(self._shard_misses),
            "shard_evictions": int(self._shard_evictions),
            "sample_read_count": int(len(samples)),
            "sample_read_p50_ms": float(np.percentile(samples, 50) * 1000.0) if len(samples) else 0.0,
            "sample_read_p95_ms": float(np.percentile(samples, 95) * 1000.0) if len(samples) else 0.0,
        }


def fit_wts_feature_scaler(
    *,
    kronos_root: str | Path,
    experiment_root: str | Path,
    rows: pd.DataFrame,
    max_samples: int,
    seed: int,
    target_kind: str = "relative",
) -> FeatureScaler:
    if rows.empty:
        raise ValueError("Cannot fit scaler with empty rows")
    rng = np.random.default_rng(seed)
    fit_rows = rows
    if max_samples > 0 and len(rows) > max_samples:
        selected = rng.choice(np.arange(len(rows)), size=int(max_samples), replace=False)
        fit_rows = rows.iloc[np.sort(selected)].reset_index(drop=True)
    dataset = WeakToStrongPathDataset(
        kronos_root=kronos_root,
        experiment_root=experiment_root,
        rows=fit_rows,
        scaler=None,
        target_kind=target_kind,
    )
    raw_blocks: list[np.ndarray] = []
    derived_blocks: list[np.ndarray] = []
    market_date_idx: set[int] = set()
    raw_dim = len(RAW_OHLCV_FIELDS) + len(RAW_TRADE_FIELDS)
    for idx in range(len(dataset)):
        x_ts, _, _, _ = dataset.get_arrays(idx)
        raw_blocks.append(x_ts[:, :raw_dim])
        derived_blocks.append(x_ts[:, raw_dim:])
        t = int(dataset.date_idx[idx])
        market_date_idx.update(range(t - SEQ_LEN + 1, t + 1))
    raw_tokens = np.concatenate(raw_blocks, axis=0)
    derived_tokens = np.concatenate(derived_blocks, axis=0)
    market_idx = np.asarray(sorted(market_date_idx), dtype=int)
    market_tokens = np.asarray(dataset.source.market[market_idx], dtype=np.float32)
    raw_low, raw_high, raw_mean, raw_std = robust_stats(raw_tokens)
    derived_low, derived_high, derived_mean, derived_std = robust_stats(derived_tokens)
    market_low, market_high, market_mean, market_std = robust_stats(market_tokens)
    return FeatureScaler(
        raw_mean=raw_mean,
        raw_std=raw_std,
        raw_clip_low=raw_low,
        raw_clip_high=raw_high,
        derived_mean=derived_mean,
        derived_std=derived_std,
        derived_clip_low=derived_low,
        derived_clip_high=derived_high,
        market_mean=market_mean,
        market_std=market_std,
        market_clip_low=market_low,
        market_clip_high=market_high,
        fit_sample_count=int(len(fit_rows)),
        fit_token_count=int(len(raw_tokens)),
        fit_seed=int(seed),
        fit_start=str(rows["signal_date"].min()) if "signal_date" in rows else None,
        fit_end=str(rows["signal_date"].max()) if "signal_date" in rows else None,
    )


def build_candidate_features(rows: pd.DataFrame) -> np.ndarray:
    n = int(len(rows))
    if n == 0:
        return np.zeros((0, len(CANDIDATE_FEATURE_FIELDS)), dtype=np.float32)
    def numeric_column(name: str, default: float | np.ndarray) -> np.ndarray:
        values = rows[name] if name in rows else default
        series = pd.Series(values, index=rows.index) if np.ndim(values) else pd.Series(float(values), index=rows.index)
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)

    rank = np.nan_to_num(numeric_column("wts_rank", 1.0), nan=1.0, posinf=1.0, neginf=1.0)
    pool = np.clip(np.nan_to_num(numeric_column("wts_pool_size", 1.0), nan=1.0, posinf=1.0, neginf=1.0), 1.0, None)
    topn = np.clip(np.nan_to_num(numeric_column("topn", 1.0), nan=1.0, posinf=1.0, neginf=1.0), 1.0, None)
    rank_pct = np.nan_to_num(numeric_column("wts_rank_pct", rank / pool), nan=1.0, posinf=1.0, neginf=1.0)
    score_z = np.nan_to_num(numeric_column("wts_score_z", 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    score_pct = np.nan_to_num(
        numeric_column("wts_score_pct", 1.0 - (rank - 1.0) / np.maximum(pool - 1.0, 1.0)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if "wts_selector_where" in rows:
        selector_where = rows["wts_selector_where"].astype(float).to_numpy(dtype=np.float32)
    else:
        selector_where = np.ones(n, dtype=np.float32)
    values = np.column_stack(
        [
            np.clip(rank_pct, 0.0, 1.0),
            np.clip(1.0 / np.maximum(rank, 1.0), 0.0, 1.0),
            np.clip(score_z, -8.0, 8.0) / 8.0,
            np.clip(score_pct, 0.0, 1.0),
            np.clip(selector_where, 0.0, 1.0),
            np.clip(rank / np.maximum(topn, 1.0), 0.0, 1.0),
        ]
    )
    return values.astype(np.float32)


def formula_identifiers(expr: str) -> set[str]:
    tree = ast.parse(str(expr), mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def required_formula_subset(config: dict[str, Any]) -> dict[str, str]:
    formulas = {**dict(config.get("factors") or {}), **dict(config.get("signals") or {})}
    selector = dict(config.get("selector") or {})
    roots = [str(selector.get("where", "")), str(selector.get("score", "0"))]
    required: set[str] = set()
    stack: list[str] = []
    for expr in roots:
        stack.extend(name for name in formula_identifiers(expr) if name in formulas)
    while stack:
        name = stack.pop()
        if name in required:
            continue
        required.add(name)
        stack.extend(dep for dep in formula_identifiers(formulas[name]) if dep in formulas and dep not in required)
    return {name: str(formulas[name]) for name in formulas if name in required}


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_external_baseline(path: Path, dates: pd.DatetimeIndex) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not path.exists():
        return None, {"source": "missing_external", "path": str(path)}
    frame = pd.read_parquet(path)
    date_col = "date" if "date" in frame.columns else "datetime"
    close_col = "close" if "close" in frame.columns else "$close"
    series = pd.Series(pd.to_numeric(frame[close_col], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(frame[date_col]))
    values = series.reindex(dates).ffill().bfill().to_numpy(dtype=np.float32)
    return values, {"source": "external_parquet", "path": str(path)}


def equal_weight_baseline(close_tn: np.ndarray) -> np.ndarray:
    values = np.asarray(close_tn, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(np.isfinite(values) & (values > 0), values, np.nan), axis=1).astype(np.float32)


def execution_buyable(
    *,
    rules: ExecutionRulesV1,
    instrument: str,
    raw: np.ndarray,
    state: np.ndarray,
    entry_idx: int,
) -> tuple[bool, str]:
    if entry_idx <= 0 or entry_idx >= raw.shape[0]:
        return False, "entry_out_of_range"
    return rules.can_buy(
        instrument,
        open_=float(raw[entry_idx, RAW_OHLCV_FIELDS.index("open")]),
        high=float(raw[entry_idx, RAW_OHLCV_FIELDS.index("high")]),
        low=float(raw[entry_idx, RAW_OHLCV_FIELDS.index("low")]),
        close=float(raw[entry_idx, RAW_OHLCV_FIELDS.index("close")]),
        preclose=float(raw[entry_idx - 1, RAW_OHLCV_FIELDS.index("close")]),
        volume=float(raw[entry_idx, RAW_OHLCV_FIELDS.index("volume")]),
        tradestatus=int(state[entry_idx, RAW_STATE_FIELDS.index("tradestatus")]),
        is_st=int(state[entry_idx, RAW_STATE_FIELDS.index("is_st")]),
        has_raw_row=int(state[entry_idx, RAW_STATE_FIELDS.index("has_raw_row")]),
    )


def valid_history_mask(raw: np.ndarray, state: np.ndarray, *, min_rows: int = 55, window: int = SEQ_LEN) -> np.ndarray:
    open_ = raw[:, :, RAW_OHLCV_FIELDS.index("open")]
    close = raw[:, :, RAW_OHLCV_FIELDS.index("close")]
    volume = raw[:, :, RAW_OHLCV_FIELDS.index("volume")]
    tradestatus = state[:, :, RAW_STATE_FIELDS.index("tradestatus")]
    is_st = state[:, :, RAW_STATE_FIELDS.index("is_st")]
    has_raw = state[:, :, RAW_STATE_FIELDS.index("has_raw_row")]
    valid = (
        (has_raw == 1)
        & (tradestatus == 1)
        & (is_st == 0)
        & np.isfinite(open_)
        & (open_ > 0)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(volume)
        & (volume > 0)
    )
    csum = np.cumsum(valid.astype(np.int16), axis=1)
    out = csum.copy()
    out[:, window:] = csum[:, window:] - csum[:, :-window]
    out[:, : window - 1] = 0
    return out >= int(min_rows)


def build_future_paths(
    *,
    raw: np.ndarray,
    baseline_close: np.ndarray,
    date_idx: int,
    horizon: int,
    target_start_offset: int = 1,
    target_price_mode: str = "entry_open",
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    entry_idx = int(date_idx) + 1
    first_target_idx = int(date_idx) + int(target_start_offset)
    end = first_target_idx + int(horizon)
    if entry_idx <= 0 or first_target_idx <= 0 or end > raw.shape[0]:
        return None, None, None, "near_calendar_end"
    open_idx = RAW_OHLCV_FIELDS.index("open")
    close_idx = RAW_OHLCV_FIELDS.index("close")
    entry_open = float(raw[entry_idx, open_idx])
    close = raw[first_target_idx:end, RAW_OHLCV_FIELDS.index("close")].astype(np.float32)
    if len(close) != horizon or not np.all(np.isfinite(close) & (close > 0)):
        return None, None, None, "invalid_stock_future_path"
    if target_price_mode == "entry_open":
        price0 = entry_open
        base0 = float(baseline_close[int(date_idx)])
    elif target_price_mode == "close_to_close":
        price0 = float(raw[first_target_idx, close_idx])
        base0 = float(baseline_close[first_target_idx])
    else:
        raise ValueError(f"Unsupported target_price_mode: {target_price_mode}")
    if not np.isfinite(price0) or price0 <= 0:
        return None, None, None, "invalid_stock_future_path"
    base_future = baseline_close[first_target_idx:end].astype(np.float32)
    if not np.isfinite(base0) or base0 <= 0 or len(base_future) != horizon or not np.all(np.isfinite(base_future) & (base_future > 0)):
        return None, None, None, "invalid_baseline_future_path"
    y_abs = close / price0 - 1.0
    y_baseline = base_future / base0 - 1.0
    y_relative = y_abs - y_baseline
    if not (np.all(np.isfinite(y_abs)) and np.all(np.isfinite(y_baseline)) and np.all(np.isfinite(y_relative))):
        return None, None, None, "nonfinite_target"
    return y_abs.astype(np.float32), y_baseline.astype(np.float32), y_relative.astype(np.float32), "ok"


class TransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        stock_dim: int,
        market_dim: int,
        candidate_dim: int,
        lookback: int,
        d_stock: int,
        d_market: int,
        d_candidate: int,
        d_model: int,
        latent_tokens: int,
        latent_dim: int,
        n_heads: int,
        n_layers: int,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.latent_tokens = int(latent_tokens)
        self.latent_dim = int(latent_dim)
        self.candidate_dim = int(candidate_dim)
        self.stock_proj = nn.Linear(stock_dim, d_stock)
        self.market_proj = nn.Linear(market_dim, d_market)
        self.candidate_proj = nn.Linear(candidate_dim, d_candidate) if candidate_dim > 0 and d_candidate > 0 else None
        fusion_dim = d_stock + d_market + (d_candidate if self.candidate_proj is not None else 0)
        self.fusion = nn.Linear(fusion_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.compress = nn.Sequential(
            nn.Linear(lookback * d_model, latent_tokens * latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_tokens * latent_dim),
        )
        self.expand = nn.Sequential(
            nn.Linear(latent_tokens * latent_dim, lookback * d_model),
            nn.GELU(),
            nn.LayerNorm(lookback * d_model),
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=1)
        self.stock_out = nn.Linear(d_model, stock_dim)
        self.market_out = nn.Linear(d_model, market_dim)

    def encode(self, stock: torch.Tensor, market: torch.Tensor, candidate: torch.Tensor | None = None) -> torch.Tensor:
        parts = [self.stock_proj(stock), self.market_proj(market)]
        if self.candidate_proj is not None:
            if candidate is None:
                candidate = stock.new_zeros((stock.shape[0], self.candidate_dim))
            candidate_token = self.candidate_proj(candidate).unsqueeze(1).expand(-1, stock.shape[1], -1)
            parts.append(candidate_token)
        x = self.fusion(torch.cat(parts, dim=-1))
        h = self.encoder(x + self.pos)
        return self.compress(h.flatten(1)).view(stock.shape[0], self.latent_tokens, self.latent_dim)

    def forward(self, stock: torch.Tensor, market: torch.Tensor, candidate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(stock, market, candidate)
        h = self.expand(z.flatten(1)).view(stock.shape[0], self.lookback, -1)
        h = self.decoder(h + self.pos)
        return self.stock_out(h), self.market_out(h), z


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    args = t[:, None].float() * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb


class ConditionalDiffusionDenoiser(nn.Module):
    def __init__(self, *, target_dim: int, cond_dim: int, hidden_dim: int, diffusion_steps: int):
        super().__init__()
        self.step_emb = nn.Embedding(diffusion_steps, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(target_dim + cond_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, y_noisy: torch.Tensor, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if step.dtype in (torch.long, torch.int64, torch.int32):
            time_emb = self.step_emb(step.long())
        else:
            time_emb = self.time_mlp(sinusoidal_time_embedding(step.float(), self.step_emb.embedding_dim))
        return self.net(torch.cat([y_noisy, cond, time_emb], dim=-1))


class FluxPathDoubleStreamBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        mlp_hidden = int(round(d_model * mlp_ratio))
        self.target_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.target_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cond_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cond_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.target_qkv = nn.Linear(d_model, d_model * 3)
        self.cond_qkv = nn.Linear(d_model, d_model * 3)
        self.target_proj = nn.Linear(d_model, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.target_mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, d_model),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, d_model),
        )
        self.target_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))
        self.cond_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))
        self.dropout = nn.Dropout(dropout)
        self.n_heads = int(n_heads)

    def forward(self, target: torch.Tensor, cond: torch.Tensor, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_mod1, target_mod2 = split_modulation(self.target_mod(vec))
        cond_mod1, cond_mod2 = split_modulation(self.cond_mod(vec))
        target_q, target_k, target_v = split_qkv_heads(
            self.target_qkv(modulate(self.target_norm1(target), target_mod1[0], target_mod1[1])),
            self.n_heads,
        )
        cond_q, cond_k, cond_v = split_qkv_heads(
            self.cond_qkv(modulate(self.cond_norm1(cond), cond_mod1[0], cond_mod1[1])),
            self.n_heads,
        )
        q = torch.cat([cond_q, target_q], dim=2)
        k = torch.cat([cond_k, target_k], dim=2)
        v = torch.cat([cond_v, target_v], dim=2)
        attn = scaled_dot_product_attention(q, k, v)
        cond_attn, target_attn = attn[:, : cond.shape[1]], attn[:, cond.shape[1] :]
        target = target + self.dropout(torch.tanh(target_mod1[2]).unsqueeze(1) * self.target_proj(target_attn))
        target = target + self.dropout(
            torch.tanh(target_mod2[2]).unsqueeze(1)
            * self.target_mlp(modulate(self.target_norm2(target), target_mod2[0], target_mod2[1]))
        )
        cond = cond + self.dropout(torch.tanh(cond_mod1[2]).unsqueeze(1) * self.cond_proj(cond_attn))
        cond = cond + self.dropout(
            torch.tanh(cond_mod2[2]).unsqueeze(1) * self.cond_mlp(modulate(self.cond_norm2(cond), cond_mod2[0], cond_mod2[1]))
        )
        return target, cond


class FluxPathSingleStreamBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        mlp_hidden = int(round(d_model * mlp_ratio))
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, d_model),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 3))
        self.dropout = nn.Dropout(dropout)
        self.n_heads = int(n_heads)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(vec).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        q, k, v = split_qkv_heads(self.qkv(h), self.n_heads)
        attn = scaled_dot_product_attention(q, k, v)
        out = self.proj(attn) + self.mlp(h)
        return x + self.dropout(torch.tanh(gate).unsqueeze(1) * out)


class FluxPathFinalLayer(nn.Module):
    def __init__(self, *, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 2))
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(vec).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class ConditionalCrossAttentionDiTDenoiser(nn.Module):
    def __init__(
        self,
        *,
        target_dim: int,
        cond_dim: int,
        cond_tokens: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        diffusion_steps: int,
    ):
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"dit_d_model must be divisible by dit_heads: {d_model} vs {n_heads}")
        self.target_dim = int(target_dim)
        self.cond_tokens = int(cond_tokens)
        self.diffusion_steps = int(diffusion_steps)
        self.target_value_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.source_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(max(int(target_dim), int(cond_tokens)), d_model)
        cond_ids = torch.stack([torch.zeros(int(cond_tokens), dtype=torch.long), torch.arange(int(cond_tokens), dtype=torch.long)], dim=1)
        target_ids = torch.stack([torch.ones(int(target_dim), dtype=torch.long), torch.arange(int(target_dim), dtype=torch.long)], dim=1)
        self.register_buffer("cond_ids", cond_ids, persistent=False)
        self.register_buffer("target_ids", target_ids, persistent=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        double_layers = max(1, int(round(int(n_layers) * 0.75)))
        single_layers = max(1, int(n_layers) - double_layers)
        self.double_blocks = nn.ModuleList(
            [
                FluxPathDoubleStreamBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                FluxPathSingleStreamBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(single_layers)
            ]
        )
        self.final_layer = FluxPathFinalLayer(d_model=d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.source_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embed.weight, std=0.02)

    def forward(self, y_noisy: torch.Tensor, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 3:
            raise ValueError(f"cross_dit expects token condition [B,T,C], got shape={tuple(cond.shape)}")
        t = step.float()
        if step.dtype in (torch.long, torch.int64, torch.int32):
            t = t / max(float(self.diffusion_steps - 1), 1.0)
        vec = self.time_mlp(sinusoidal_time_embedding(t, self.source_embed.embedding_dim))
        target = self.target_value_proj(y_noisy.unsqueeze(-1)) + self.ids_to_embedding(self.target_ids[: y_noisy.shape[1]])
        cond = self.cond_proj(cond) + self.ids_to_embedding(self.cond_ids[: cond.shape[1]])
        for block in self.double_blocks:
            target, cond = block(target, cond, vec)
        joined = torch.cat([cond, target], dim=1)
        for block in self.single_blocks:
            joined = block(joined, vec)
        target = joined[:, cond.shape[1] :, :]
        return self.final_layer(target, vec).squeeze(-1)

    def ids_to_embedding(self, ids: torch.Tensor) -> torch.Tensor:
        return (self.source_embed(ids[:, 0]) + self.position_embed(ids[:, 1])).unsqueeze(0)


def split_modulation(values: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    chunks = values.chunk(6, dim=-1)
    return (chunks[0], chunks[1], chunks[2]), (chunks[3], chunks[4], chunks[5])


def split_qkv_heads(qkv: torch.Tensor, n_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, channels3 = qkv.shape
    channels = channels3 // 3
    head_dim = channels // int(n_heads)
    qkv = qkv.view(batch, tokens, 3, int(n_heads), head_dim).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], -1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        self.steps = int(steps)
        self.betas = torch.linspace(1e-4, 0.02, self.steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, y0: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[step].view(-1, 1)
        return torch.sqrt(a) * y0 + torch.sqrt(1.0 - a) * noise


def build_models(cfg: ModelConfig, *, stock_dim: int, market_dim: int) -> tuple[TransformerAutoEncoder, nn.Module]:
    ae = TransformerAutoEncoder(
        stock_dim=stock_dim,
        market_dim=market_dim,
        candidate_dim=cfg.candidate_dim,
        lookback=cfg.lookback,
        d_stock=cfg.d_stock,
        d_market=cfg.d_market,
        d_candidate=cfg.d_candidate,
        d_model=cfg.d_model,
        latent_tokens=cfg.latent_tokens,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )
    if cfg.denoiser_type == "cross_dit":
        diffusion = ConditionalCrossAttentionDiTDenoiser(
            target_dim=cfg.horizon,
            cond_dim=cfg.latent_dim,
            cond_tokens=cfg.latent_tokens,
            d_model=cfg.dit_d_model,
            n_layers=cfg.dit_layers,
            n_heads=cfg.dit_heads,
            mlp_ratio=cfg.dit_mlp_ratio,
            dropout=cfg.dit_dropout,
            diffusion_steps=cfg.diffusion_steps,
        )
    elif cfg.denoiser_type == "mlp":
        diffusion = ConditionalDiffusionDenoiser(
            target_dim=cfg.horizon,
            cond_dim=cfg.latent_tokens * cfg.latent_dim,
            hidden_dim=cfg.generative_hidden_dim,
            diffusion_steps=cfg.diffusion_steps,
        )
    else:
        raise ValueError(f"Unsupported denoiser_type: {cfg.denoiser_type}")
    return ae, diffusion


def encode_condition(
    ae: TransformerAutoEncoder,
    stock: torch.Tensor,
    market: torch.Tensor,
    candidate: torch.Tensor | None = None,
    cfg: ModelConfig | None = None,
) -> torch.Tensor:
    z = ae.encode(stock, market, candidate)
    if cfg is not None and cfg.denoiser_type == "cross_dit":
        return z
    return z.flatten(1)


def flow_matching_loss(model: nn.Module, y_target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    x0 = torch.randn_like(y_target)
    t = torch.rand((y_target.shape[0],), device=y_target.device)
    xt = (1.0 - t.view(-1, 1)) * x0 + t.view(-1, 1) * y_target
    velocity = y_target - x0
    return torch.nn.functional.smooth_l1_loss(model(xt, t, cond), velocity)


def ddpm_loss(model: nn.Module, schedule: DiffusionSchedule, y_target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    step = torch.randint(0, schedule.steps, (y_target.shape[0],), device=y_target.device)
    noise = torch.randn_like(y_target)
    y_noisy = schedule.q_sample(y_target, step, noise)
    return torch.nn.functional.smooth_l1_loss(model(y_noisy, step, cond), noise)


def effective_inference_steps(cfg: ModelConfig, override: int | None = None) -> int:
    if override is not None and override > 0:
        return int(override)
    if cfg.inference_steps > 0:
        return int(cfg.inference_steps)
    return int(cfg.flow_steps if cfg.generative_objective == "flow" else cfg.diffusion_steps)


def sample_flow(model: nn.Module, cond: torch.Tensor, *, horizon: int, steps: int) -> torch.Tensor:
    steps = max(1, int(steps))
    y = torch.randn((cond.shape[0], int(horizon)), device=cond.device)
    dt = 1.0 / steps
    for idx in range(steps):
        t = torch.full((cond.shape[0],), (idx + 0.5) / steps, device=cond.device)
        y = y + dt * model(y, t, cond)
    return y


def sample_ddpm(model: nn.Module, schedule: DiffusionSchedule, cond: torch.Tensor, *, horizon: int, inference_steps: int) -> torch.Tensor:
    inference_steps = max(1, min(int(inference_steps), schedule.steps))
    indices = np.linspace(schedule.steps - 1, 0, inference_steps).round().astype(int)
    y = torch.randn((cond.shape[0], int(horizon)), device=cond.device)
    for i, step_value in enumerate(indices):
        step = torch.full((cond.shape[0],), int(step_value), device=cond.device, dtype=torch.long)
        eps = model(y, step, cond)
        alpha_bar = schedule.alpha_bar[step].view(-1, 1)
        x0 = (y - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        if i == len(indices) - 1:
            y = x0
        else:
            next_step_value = int(indices[i + 1])
            next_step = torch.full((cond.shape[0],), next_step_value, device=cond.device, dtype=torch.long)
            next_alpha_bar = schedule.alpha_bar[next_step].view(-1, 1)
            y = torch.sqrt(next_alpha_bar) * x0 + torch.sqrt(1.0 - next_alpha_bar) * eps
    return y


@torch.no_grad()
def sample_prediction_batch(
    *,
    ae: nn.Module,
    diffusion: nn.Module,
    schedule: DiffusionSchedule,
    stock: torch.Tensor,
    market: torch.Tensor,
    candidate: torch.Tensor | None = None,
    cfg: ModelConfig,
    k_samples: int,
    inference_steps: int | None = None,
) -> dict[str, np.ndarray]:
    ae.eval()
    diffusion.eval()
    cond = encode_condition(ae, stock, market, candidate, cfg)
    samples = []
    steps = effective_inference_steps(cfg, inference_steps)
    for _ in range(int(k_samples)):
        if cfg.generative_objective == "flow":
            y = sample_flow(diffusion, cond, horizon=cfg.horizon, steps=steps)
        else:
            y = sample_ddpm(diffusion, schedule, cond, horizon=cfg.horizon, inference_steps=steps)
        samples.append(y.detach().cpu().numpy())
    arr = np.stack(samples, axis=1)
    drawdown = sample_path_drawdown(arr)
    return {
        "mean": arr.mean(axis=1),
        "std": arr.std(axis=1),
        "prob_pos": (arr > 0.0).mean(axis=1),
        "q10": np.quantile(arr, 0.1, axis=1),
        "q90": np.quantile(arr, 0.9, axis=1),
        "drawdown": drawdown.mean(axis=1),
    }


def sample_path_drawdown(samples: np.ndarray) -> np.ndarray:
    zero = np.zeros((*samples.shape[:2], 1), dtype=samples.dtype)
    path = np.concatenate([zero, samples], axis=2)
    peak = np.maximum.accumulate(path, axis=2)
    return (path - peak).min(axis=2)


def add_score_variants(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for suffix in ("3d", "5d", "7d", "10d", "15d", "20d", "30d"):
        if f"diff_mean_{suffix}" not in out:
            continue
        mean = out[f"diff_mean_{suffix}"]
        std = out[f"diff_std_{suffix}"].clip(lower=1e-6)
        out[f"score_mean_{suffix}"] = mean
        out[f"score_mean_over_std_{suffix}"] = mean / std
        out[f"score_mean_minus_0.5std_{suffix}"] = mean - 0.5 * std
        out[f"score_mean_plus_q10_{suffix}"] = mean + out[f"diff_q10_{suffix}"]
        out[f"score_prob_pos_{suffix}"] = out[f"diff_prob_pos_{suffix}"]
        out[f"score_q10_{suffix}"] = out[f"diff_q10_{suffix}"]
        out[f"score_q90_{suffix}"] = out[f"diff_q90_{suffix}"]
        for threshold in SCORE_PROB_THRESHOLDS:
            tag = threshold_tag(threshold)
            out[f"score_prob_rel_ge_{tag}_{suffix}"] = normal_survival((threshold - mean) / std)
    if {"diff_mean_20d", "diff_q90_30d", "diff_q10_30d", "diff_drawdown_30d"}.issubset(out.columns):
        out["score_tail_utility_30d"] = (
            out["diff_mean_20d"] + 0.5 * out["diff_q90_30d"] + out["diff_q10_30d"] + 0.5 * out["diff_drawdown_30d"]
        )
    return out


def normal_survival(z: pd.Series | np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    tensor = torch.from_numpy(values)
    return (0.5 * torch.special.erfc(tensor / math.sqrt(2.0))).numpy()


def threshold_tag(value: float) -> str:
    return f"{int(round(float(value) * 1000)):03d}"


def daily_ic_values(frame: pd.DataFrame, score: str, label: str, *, min_count: int = 5) -> list[dict[str, Any]]:
    rows = []
    for day, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < min_count or values[score].nunique() < 2 or values[label].nunique() < 2:
            continue
        rows.append(
            {
                "signal_time": day,
                "spearman": values[score].corr(values[label], method="spearman"),
                "pearson": values[score].corr(values[label], method="pearson"),
                "n": int(len(values)),
            }
        )
    return rows


def spread_summary(frame: pd.DataFrame, score: str, label: str, *, top_frac: float = 0.2) -> dict[str, Any]:
    spreads = []
    top_values = []
    bottom_values = []
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 5:
            continue
        n = max(1, int(len(values) * float(top_frac)))
        bottom = values.head(n)[label].mean()
        top = values.tail(n)[label].mean()
        spreads.append(top - bottom)
        top_values.append(top)
        bottom_values.append(bottom)
    arr = np.asarray(spreads, dtype=float)
    return {
        "score": score,
        "label": label,
        "spread_mean": safe_float(arr.mean()) if len(arr) else None,
        "spread_median": safe_float(np.median(arr)) if len(arr) else None,
        "spread_pos_ratio": safe_float((arr > 0).mean()) if len(arr) else None,
        "top_mean": safe_float(np.mean(top_values)) if top_values else None,
        "bottom_mean": safe_float(np.mean(bottom_values)) if bottom_values else None,
        "days": int(len(arr)),
    }


def rankic_summary(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    daily = daily_ic_values(frame, score, label)
    values = pd.Series([row["spearman"] for row in daily], dtype=float).dropna()
    return {
        "score": score,
        "label": label,
        "rankic_mean": safe_float(values.mean()),
        "rankic_median": safe_float(values.median()),
        "rankic_ir": safe_float(values.mean() / (values.std(ddof=1) + 1e-12)) if len(values) > 1 else None,
        "rankic_pos_ratio": safe_float((values > 0).mean()) if len(values) else None,
        "days": int(len(values)),
    }


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def state_dict_without_module(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model_state_flexible(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(clean)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
