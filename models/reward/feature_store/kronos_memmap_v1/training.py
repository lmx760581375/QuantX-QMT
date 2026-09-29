"""PyTorch dataset, sampler, scaler, and model for Kronos memmaps."""

from __future__ import annotations

import json
import math
import time
from hashlib import sha256
from collections import OrderedDict
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from .config import DERIVED_FEATURE_FIELDS, MARKET_FEATURE_FIELDS, RAW_OHLCV_FIELDS, RAW_TRADE_FIELDS, SEQ_LEN, DatasetPaths
from .storage import read_json


INPUT_MODES = ("raw_relative", "relative_only", "raw_relative_causal")


@dataclass(frozen=True)
class FeatureScaler:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    raw_clip_low: np.ndarray
    raw_clip_high: np.ndarray
    derived_mean: np.ndarray
    derived_std: np.ndarray
    derived_clip_low: np.ndarray
    derived_clip_high: np.ndarray
    market_mean: np.ndarray
    market_std: np.ndarray
    market_clip_low: np.ndarray
    market_clip_high: np.ndarray
    fit_sample_count: int
    fit_token_count: int
    fit_seed: int
    fit_start: str | None = None
    fit_end: str | None = None

    def to_json(self) -> dict:
        return {
            "raw_feature_mean": self.raw_mean.tolist(),
            "raw_feature_std": self.raw_std.tolist(),
            "raw_clip_low": self.raw_clip_low.tolist(),
            "raw_clip_high": self.raw_clip_high.tolist(),
            "derived_feature_mean": self.derived_mean.tolist(),
            "derived_feature_std": self.derived_std.tolist(),
            "derived_clip_low": self.derived_clip_low.tolist(),
            "derived_clip_high": self.derived_clip_high.tolist(),
            "market_feature_mean": self.market_mean.tolist(),
            "market_feature_std": self.market_std.tolist(),
            "market_clip_low": self.market_clip_low.tolist(),
            "market_clip_high": self.market_clip_high.tolist(),
            "fit_sample_count": int(self.fit_sample_count),
            "fit_token_count": int(self.fit_token_count),
            "fit_seed": int(self.fit_seed),
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FeatureScaler":
        return cls(
            raw_mean=np.asarray(payload["raw_feature_mean"], dtype=np.float32),
            raw_std=np.asarray(payload["raw_feature_std"], dtype=np.float32),
            raw_clip_low=np.asarray(payload["raw_clip_low"], dtype=np.float32),
            raw_clip_high=np.asarray(payload["raw_clip_high"], dtype=np.float32),
            derived_mean=np.asarray(payload["derived_feature_mean"], dtype=np.float32),
            derived_std=np.asarray(payload["derived_feature_std"], dtype=np.float32),
            derived_clip_low=np.asarray(payload["derived_clip_low"], dtype=np.float32),
            derived_clip_high=np.asarray(payload["derived_clip_high"], dtype=np.float32),
            market_mean=np.asarray(payload["market_feature_mean"], dtype=np.float32),
            market_std=np.asarray(payload["market_feature_std"], dtype=np.float32),
            market_clip_low=np.asarray(payload["market_clip_low"], dtype=np.float32),
            market_clip_high=np.asarray(payload["market_clip_high"], dtype=np.float32),
            fit_sample_count=int(payload["fit_sample_count"]),
            fit_token_count=int(payload["fit_token_count"]),
            fit_seed=int(payload["fit_seed"]),
            fit_start=str(payload["fit_start"]) if payload.get("fit_start") is not None else None,
            fit_end=str(payload["fit_end"]) if payload.get("fit_end") is not None else None,
        )


class KronosMemmapDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        rows: pd.DataFrame,
        *,
        scaler: FeatureScaler | None,
        input_mode: str = "raw_relative",
        max_open_shards: int = 2,
    ):
        self.paths = DatasetPaths(Path(root).resolve())
        self.rows = rows.reset_index(drop=True)
        self.scaler = scaler
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input_mode={input_mode}; expected one of {INPUT_MODES}")
        if input_mode == "raw_relative_causal" and scaler is None:
            raise ValueError("raw_relative_causal requires a fold FeatureScaler for change/factor normalization")
        self.input_mode = input_mode
        self.max_open_shards = int(max_open_shards)
        self.raw_meta = read_json(self.paths.raw_ohlcv_meta)
        self.trade_meta = read_json(self.paths.raw_trade_meta)
        self.market_meta = read_json(self.paths.market_meta)
        self.feature_manifest = read_json(self.paths.features_manifest)
        n_inst, n_dates, _ = (int(v) for v in self.raw_meta["shape"])
        self.raw_ohlcv = np.memmap(self.paths.raw_ohlcv, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_OHLCV_FIELDS)))
        self.raw_trade = np.memmap(self.paths.raw_trade, dtype=np.float32, mode="r", shape=(n_inst, n_dates, len(RAW_TRADE_FIELDS)))
        self.market = np.memmap(
            self.paths.market,
            dtype=np.float32,
            mode="r",
            shape=(n_dates, len(MARKET_FEATURE_FIELDS)),
        )
        self._shards: OrderedDict[int, np.memmap] = OrderedDict()
        self._shard_misses = 0
        self._shard_evictions = 0
        self._sample_read_seconds: deque[float] = deque(maxlen=100_000)
        self._shard_meta = {int(item["shard_id"]): item for item in self.feature_manifest["shards"]}
        self.instrument_idx = self.rows["instrument_idx"].to_numpy(dtype=np.int64)
        self.shard_id = self.rows["shard_id"].to_numpy(dtype=np.int64)
        self.local_instrument_idx = self.rows["local_instrument_idx"].to_numpy(dtype=np.int64)
        self.date_idx = self.rows["date_idx"].to_numpy(dtype=np.int64)
        self.label = self.rows["label"].to_numpy(dtype=np.int64)

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

    def get_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        inst = int(self.instrument_idx[index])
        shard_id = int(self.shard_id[index])
        local_inst = int(self.local_instrument_idx[index])
        t = int(self.date_idx[index])
        start = t - SEQ_LEN + 1
        if start < 0:
            raise IndexError(f"Sample date_idx={t} cannot provide seq_len={SEQ_LEN}")
        x_raw = np.concatenate(
            (
                self.raw_ohlcv[inst, start : t + 1, :],
                self.raw_trade[inst, start : t + 1, :],
            ),
            axis=-1,
        ).astype(np.float32)
        x_derived = np.asarray(self.get_feature_shard(shard_id)[local_inst, start : t + 1, :], dtype=np.float32)
        x_market = np.asarray(self.market[start : t + 1, :], dtype=np.float32)
        x_ts = np.concatenate((x_raw, x_derived), axis=-1)
        return x_ts, x_market, int(self.label[index])

    def __getitem__(self, index: int):
        started = time.perf_counter()
        x_ts, x_market, y = self.get_arrays(index)
        raw_dim = len(RAW_OHLCV_FIELDS) + len(RAW_TRADE_FIELDS)
        if self.scaler is not None:
            raw = x_ts[:, :raw_dim]
            derived = x_ts[:, raw_dim:]
            if self.input_mode == "raw_relative_causal":
                raw = causal_instance_normalize(
                    raw,
                    raw_dim=raw_dim,
                    global_low=self.scaler.raw_clip_low,
                    global_high=self.scaler.raw_clip_high,
                    global_mean=self.scaler.raw_mean,
                    global_std=self.scaler.raw_std,
                )
            else:
                raw = normalize(raw, self.scaler.raw_clip_low, self.scaler.raw_clip_high, self.scaler.raw_mean, self.scaler.raw_std)
            derived = normalize(
                derived,
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
        result = (torch.from_numpy(np.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)), torch.from_numpy(
            np.nan_to_num(x_market, nan=0.0, posinf=0.0, neginf=0.0)
        ), torch.tensor(y, dtype=torch.long))
        self._sample_read_seconds.append(time.perf_counter() - started)
        return result

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


def normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, low, high)
    return ((clipped - mean) / np.where(std <= 1e-12, 1.0, std)).astype(np.float32)


def causal_instance_normalize(
    values: np.ndarray,
    *,
    raw_dim: int,
    global_low: np.ndarray,
    global_high: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """Normalize price/liquidity fields using only the current <= t window.

    `change` and `factor` keep the fold-global normalization because their
    semantics are not absolute price/liquidity levels.
    """
    out = normalize(values, global_low, global_high, global_mean, global_std)
    original = np.asarray(values, dtype=np.float32)
    causal_columns = (0, 1, 2, 3, 4, 5, 8, 9)
    if raw_dim != out.shape[1]:
        raise ValueError(f"Expected raw_dim={raw_dim}, got raw feature width={out.shape[1]}")
    for column in causal_columns:
        series = original[:, column]
        finite = np.isfinite(series)
        if not finite.any():
            out[:, column] = 0.0
            continue
        mean = float(series[finite].mean())
        std = float(series[finite].std())
        out[:, column] = (series - mean) / (std if std > 1e-12 else 1.0)
    return out


def fit_feature_scaler(
    root: str | Path,
    rows: pd.DataFrame,
    *,
    max_samples: int = 4096,
    seed: int = 7,
) -> FeatureScaler:
    rng = np.random.default_rng(seed)
    if len(rows) == 0:
        raise ValueError("Cannot fit scaler with empty rows")
    if len(rows) > max_samples:
        selected_pos = rng.choice(np.arange(len(rows)), size=int(max_samples), replace=False)
        fit_rows = rows.iloc[np.sort(selected_pos)].reset_index(drop=True)
    else:
        fit_rows = rows.reset_index(drop=True)
    dataset = KronosMemmapDataset(root, fit_rows, scaler=None)
    raw_blocks: list[np.ndarray] = []
    derived_blocks: list[np.ndarray] = []
    market_date_idx: set[int] = set()
    raw_dim = len(RAW_OHLCV_FIELDS) + len(RAW_TRADE_FIELDS)
    for idx in range(len(dataset)):
        x_ts, _, _ = dataset.get_arrays(idx)
        raw_blocks.append(x_ts[:, :raw_dim])
        derived_blocks.append(x_ts[:, raw_dim:])
        t = int(dataset.date_idx[idx])
        market_date_idx.update(range(t - SEQ_LEN + 1, t + 1))
    raw_tokens = np.concatenate(raw_blocks, axis=0)
    derived_tokens = np.concatenate(derived_blocks, axis=0)
    paths = DatasetPaths(Path(root).resolve())
    market_meta = read_json(paths.market_meta)
    n_dates = int(market_meta["shape"][0])
    market = np.memmap(paths.market, dtype=np.float32, mode="r", shape=(n_dates, len(MARKET_FEATURE_FIELDS)))
    market_idx = np.asarray(sorted(day for day in market_date_idx if 0 <= day < n_dates), dtype=int)
    market_tokens = np.asarray(market[market_idx], dtype=np.float32)
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


def robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = values.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    low = np.nanpercentile(arr, 0.5, axis=0).astype(np.float32)
    high = np.nanpercentile(arr, 99.5, axis=0).astype(np.float32)
    clipped = np.clip(arr, low, high)
    mean = np.nanmean(clipped, axis=0).astype(np.float32)
    std = np.nanstd(clipped, axis=0).astype(np.float32)
    low = np.nan_to_num(low, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.where(std <= 1e-12, 1.0, std).astype(np.float32)
    return low, high, mean, std


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Single-temperature multiclass calibration fitted only on a held-out split."""

    temperature: float
    sample_count: int
    before: dict[str, float]
    after: dict[str, float]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64) / max(float(self.temperature), 1e-6)
        values -= values.max(axis=1, keepdims=True)
        exp = np.exp(values)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def to_json(self) -> dict:
        payload = {
            "kind": "multiclass_temperature_v1",
            "temperature": float(self.temperature),
            "sample_count": int(self.sample_count),
            "before": self.before,
            "after": self.after,
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        payload["sha256"] = "sha256:" + sha256(encoded).hexdigest()
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "TemperatureCalibrator":
        return cls(
            temperature=float(payload["temperature"]),
            sample_count=int(payload["sample_count"]),
            before={str(k): float(v) for k, v in payload["before"].items()},
            after={str(k): float(v) for k, v in payload["after"].items()},
        )


def fit_temperature_calibrator(logits: np.ndarray, labels: np.ndarray) -> TemperatureCalibrator:
    values = np.asarray(logits, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) != len(target) or len(target) == 0:
        raise ValueError("Temperature calibration expects non-empty [n, classes] logits and matching labels")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    logits_t = torch.from_numpy(values.astype(np.float64, copy=False))
    labels_t = torch.from_numpy(target)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_temperature.exp().clamp(0.05, 20.0), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    before_prob = softmax_numpy(values)
    after_prob = softmax_numpy(values / temperature)
    return TemperatureCalibrator(
        temperature=temperature,
        sample_count=int(len(target)),
        before=probability_metrics(before_prob, target),
        after=probability_metrics(after_prob, target),
    )


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    prob = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if len(prob) == 0:
        return {"nll": float("nan"), "brier": float("nan"), "ece": float("nan")}
    one_hot = np.eye(prob.shape[1], dtype=np.float64)[target]
    nll = float(-np.log(np.clip(prob[np.arange(len(target)), target], 1e-12, 1.0)).mean())
    brier = float(np.square(prob - one_hot).sum(axis=1).mean())
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == target).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1] if idx < bins - 1 else confidence <= edges[idx + 1])
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"nll": nll, "brier": brier, "ece": float(ece)}


def save_calibrator(path: Path, calibrator: TemperatureCalibrator) -> dict:
    payload = calibrator.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_calibrator(path: Path) -> TemperatureCalibrator:
    return TemperatureCalibrator.from_json(json.loads(path.read_text(encoding="utf-8")))


class BalancedBlockSampler(Sampler[int]):
    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        hold_multiplier: float = 2.0,
        block_size: int = 8192,
        seed: int = 7,
        mode: str = "legacy_buy_sell_hold",
        positive_label: int = 1,
        negative_label: int = 0,
        negative_multiplier: float = 1.0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ):
        self.rows = rows.reset_index(drop=True)
        self.hold_multiplier = float(hold_multiplier)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.mode = str(mode)
        self.positive_label = int(positive_label)
        self.negative_label = int(negative_label)
        self.negative_multiplier = float(negative_multiplier)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        if self.num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = self._selected_indices(rng)
        order_frame = self.rows.iloc[selected].assign(_pos=selected)
        order_frame = order_frame.sort_values(["shard_id", "local_instrument_idx", "date_idx", "_pos"])
        ordered = order_frame["_pos"].to_numpy(dtype=np.int64)
        if len(ordered) == 0:
            return iter(())
        blocks = [ordered[start : start + self.block_size] for start in range(0, len(ordered), self.block_size)]
        rng.shuffle(blocks)
        ordered = np.concatenate(blocks).astype(int)
        if self.num_replicas > 1:
            ordered = self._replica_slice(ordered)
        self.epoch += 1
        return iter(ordered.tolist())

    def __len__(self) -> int:
        length = self._global_length()
        if self.num_replicas == 1:
            return length
        if self.drop_last:
            return length // self.num_replicas
        return int(math.ceil(length / self.num_replicas))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_length(self) -> int:
        labels = self.rows["label"].to_numpy(dtype=np.int64)
        if self.mode == "binary_balanced":
            positive = int((labels == self.positive_label).sum())
            negative = int((labels == self.negative_label).sum())
            return positive + min(negative, int(math.ceil(max(1, positive) * self.negative_multiplier)))
        if self.mode == "shuffle":
            return int(len(labels))
        if self.mode != "legacy_buy_sell_hold":
            raise ValueError(f"Unsupported sampler mode: {self.mode}")
        buy_sell = int((labels != 1).sum())
        hold = int((labels == 1).sum())
        return buy_sell + min(hold, int(math.ceil(max(1, buy_sell) * self.hold_multiplier)))

    def _selected_indices(self, rng: np.random.Generator) -> np.ndarray:
        labels = self.rows["label"].to_numpy(dtype=np.int64)
        if self.mode == "binary_balanced":
            positive = np.flatnonzero(labels == self.positive_label)
            negative = np.flatnonzero(labels == self.negative_label)
            negative_target = min(len(negative), int(math.ceil(max(1, len(positive)) * self.negative_multiplier)))
            negative_selected = self._sample_stratified(negative, negative_target, rng)
            return np.concatenate((positive, negative_selected))
        if self.mode == "shuffle":
            return np.arange(len(self.rows), dtype=np.int64)
        if self.mode != "legacy_buy_sell_hold":
            raise ValueError(f"Unsupported sampler mode: {self.mode}")
        buy_sell = np.flatnonzero(labels != 1)
        hold = np.flatnonzero(labels == 1)
        hold_target = min(len(hold), int(math.ceil(max(1, len(buy_sell)) * self.hold_multiplier)))
        hold_selected = self._sample_stratified(hold, hold_target, rng)
        return np.concatenate((buy_sell, hold_selected))

    def _replica_slice(self, ordered: np.ndarray) -> np.ndarray:
        if len(ordered) == 0:
            return ordered
        if self.drop_last:
            total_size = (len(ordered) // self.num_replicas) * self.num_replicas
            ordered = ordered[:total_size]
        else:
            total_size = int(math.ceil(len(ordered) / self.num_replicas)) * self.num_replicas
            padding = total_size - len(ordered)
            if padding > 0:
                repeats = int(math.ceil(padding / len(ordered)))
                ordered = np.concatenate((ordered, np.tile(ordered, repeats)[:padding]))
        return ordered[self.rank:len(ordered):self.num_replicas]

    def _sample_hold(self, hold: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_stratified(hold, target, rng)

    def _sample_stratified(self, indices: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
        if len(indices) <= target:
            return indices
        strata = ("year", "board", "asset_bucket")
        if not set(strata).issubset(self.rows.columns):
            return rng.choice(indices, size=target, replace=False)
        keys = self.rows.iloc[indices].loc[:, list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        quota = np.zeros(len(unique), dtype=int)
        if target >= len(unique):
            quota[:] = 1
        remaining = target - int(quota.sum())
        available = counts - quota
        ideal = available.astype(float) * (remaining / max(1, available.sum()))
        quota += np.floor(ideal).astype(int)
        remaining = target - int(quota.sum())
        # Largest fractional remainders retain the total target while capping at availability.
        fractions = ideal - np.floor(ideal)
        for group in np.argsort(-fractions):
            if remaining <= 0:
                break
            if quota[group] < counts[group]:
                quota[group] += 1
                remaining -= 1
        selected = []
        for group in range(len(unique)):
            positions = indices[inverse == group]
            if quota[group]:
                selected.append(rng.choice(positions, size=quota[group], replace=False))
        return np.concatenate(selected).astype(np.int64) if selected else np.empty(0, dtype=np.int64)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q)
        k = apply_rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out(out)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    dim = x.shape[-1]
    device = x.device
    dtype = x.dtype
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    freqs = torch.einsum("s,d->sd", pos, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(d_model, ffn_dim)
        self.gate = nn.Linear(d_model, ffn_dim)
        self.down = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(torch.nn.functional.silu(self.gate(x)) * self.up(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class KronosTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        ts_feature_dim: int = 52,
        market_feature_dim: int = len(MARKET_FEATURE_FIELDS),
        use_market_input: bool = True,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 192,
        head_hidden: int = 128,
        dropout: float = 0.05,
        num_classes: int = 3,
    ):
        super().__init__()
        ts_dim = max(32, d_model * 2 // 3)
        market_dim = d_model - ts_dim
        self.use_market_input = bool(use_market_input)
        self.market_dim = market_dim
        self.ts_proj = nn.Sequential(nn.Linear(ts_feature_dim, ts_dim), RMSNorm(ts_dim))
        self.market_proj = nn.Sequential(nn.Linear(market_feature_dim, market_dim), RMSNorm(market_dim))
        self.fusion = nn.Sequential(nn.Linear(d_model, d_model), RMSNorm(d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, int(num_classes)),
        )

    def forward(self, x_ts: torch.Tensor, x_market: torch.Tensor) -> torch.Tensor:
        ts = self.ts_proj(x_ts)
        market = self.market_proj(x_market) if self.use_market_input else torch.zeros(
            (*ts.shape[:-1], self.market_dim), dtype=ts.dtype, device=ts.device
        )
        x = torch.cat((ts, market), dim=-1)
        x = self.fusion(x)
        for block in self.blocks:
            x = block(x)
        pooled = torch.cat((x[:, -1, :], x.mean(dim=1)), dim=-1)
        return self.head(pooled)


def save_scaler(path: Path, scaler: FeatureScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_json(), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_scaler(path: Path) -> FeatureScaler:
    return FeatureScaler.from_json(json.loads(path.read_text(encoding="utf-8")))
