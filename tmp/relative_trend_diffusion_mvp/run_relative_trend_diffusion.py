"""Relative-trend Transformer AE + conditional diffusion experiment.

This is an intentionally self-contained research script. It keeps code and
artifacts under ./tmp and does not modify QuantX core modules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader


MAINBOARD_PREFIXES = ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")


@dataclass(frozen=True)
class Config:
    provider_uri: str
    baseline_path: str
    output_root: str
    dataset_format: str
    start: str
    end: str
    train_end: str
    test_start: str
    val_ratio_after_train: float
    max_instruments: int
    sample_every: int
    max_samples: int
    lookback: int
    horizon: int
    seed: int
    batch_size: int
    num_workers: int
    shuffle_block_size: int
    ae_epochs: int
    diffusion_epochs: int
    lr: float
    generative_objective: str
    flow_steps: int
    inference_steps: int
    inference_steps_sweep: str
    denoiser_type: str
    generative_hidden_dim: int
    dit_d_model: int
    dit_layers: int
    dit_heads: int
    dit_mlp_ratio: float
    dit_dropout: float
    ae_recon_gate_nmae: float
    model_size: str
    d_model: int
    d_stock: int
    d_market: int
    latent_tokens: int
    latent_dim: int
    n_heads: int
    n_layers: int
    diffusion_steps: int
    k_samples: int
    log_every_batches: int


def main() -> int:
    args = parse_args()
    apply_model_size(args)
    cfg_values = vars(args).copy()
    command = str(cfg_values.pop("command"))
    cfg = Config(**cfg_values)
    output_root = Path(cfg.output_root)
    for subdir in ("data", "runs"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    if command in {"build", "all"}:
        build_dataset(cfg)
    if command in {"train", "all"}:
        train_models(cfg)
    if command in {"eval", "all"}:
        evaluate(cfg)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "train", "eval", "all"])
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--baseline-path", default="data/derived/market_regime/sh000001_daily.parquet")
    parser.add_argument("--output-root", default="tmp/relative_trend_diffusion_mvp")
    parser.add_argument("--dataset-format", choices=["npz", "sharded_memmap"], default="npz")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--test-start", default="")
    parser.add_argument("--val-ratio-after-train", type=float, default=0.15)
    parser.add_argument("--max-instruments", type=int, default=240)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-block-size", type=int, default=4096)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--diffusion-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generative-objective", choices=["ddpm", "flow"], default="ddpm")
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, default=0)
    parser.add_argument("--inference-steps-sweep", default="")
    parser.add_argument("--denoiser-type", choices=["mlp", "cross_dit"], default="mlp")
    parser.add_argument("--generative-hidden-dim", type=int, default=0)
    parser.add_argument("--dit-d-model", type=int, default=384)
    parser.add_argument("--dit-layers", type=int, default=6)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=3.65)
    parser.add_argument("--dit-dropout", type=float, default=0.1)
    parser.add_argument("--ae-recon-gate-nmae", type=float, default=0.02)
    parser.add_argument("--model-size", choices=["custom", "small", "base", "large"], default="custom")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-stock", type=int, default=48)
    parser.add_argument("--d-market", type=int, default=32)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--log-every-batches", type=int, default=20)
    return parser.parse_args()


def apply_model_size(args: argparse.Namespace) -> None:
    if args.model_size in {"custom", "small"}:
        return
    presets = {
        "base": {
            "d_model": 96,
            "d_stock": 64,
            "d_market": 48,
            "latent_tokens": 8,
            "latent_dim": 96,
            "n_heads": 4,
            "n_layers": 3,
            "generative_hidden_dim": 256,
        },
        "large": {
            "d_model": 128,
            "d_stock": 80,
            "d_market": 64,
            "latent_tokens": 10,
            "latent_dim": 96,
            "n_heads": 4,
            "n_layers": 4,
            "generative_hidden_dim": 384,
        },
    }
    for key, value in presets[args.model_size].items():
        setattr(args, key, value)


def build_dataset(cfg: Config) -> None:
    set_seed(cfg.seed)
    output_root = Path(cfg.output_root)
    provider_uri = Path(cfg.provider_uri)
    reader = QlibBinReader(provider_uri)
    calendar = reader.calendar(None, cfg.end)
    start_ts = pd.Timestamp(cfg.start)
    end_ts = pd.Timestamp(cfg.end)
    start_pos = int(calendar.searchsorted(start_ts, side="left"))
    load_start_pos = max(0, start_pos - cfg.lookback - 40)
    load_start = calendar[load_start_pos]

    symbols = load_symbols(provider_uri / "instruments" / "all.txt", cfg.max_instruments, start_ts, end_ts)
    fields = [
        "$open",
        "$high",
        "$low",
        "$close",
        "$volume",
        "$amount",
        "$vwap",
        "$turnover_rate",
    ]
    print(
        f"[build] symbols={len(symbols)} load_start={load_start.date()} start={start_ts.date()} end={end_ts.date()}",
        flush=True,
    )
    raw = reader.features(symbols, fields, load_start.strftime("%Y-%m-%d"), cfg.end)
    if raw.empty:
        raise RuntimeError("No qlib features were loaded")
    panels = frame_to_panels(raw, calendar[(calendar >= load_start) & (calendar <= end_ts)], symbols, fields)
    baseline = load_baseline(Path(cfg.baseline_path), panels["close"].index)

    stock_features, stock_names = build_stock_features(panels)
    market_features, market_names = build_market_features(panels, stock_features, stock_names)
    if cfg.dataset_format == "sharded_memmap":
        build_sharded_memmap_dataset(
            cfg=cfg,
            output_root=output_root,
            dates=panels["close"].index,
            symbols=list(panels["close"].columns),
            close=panels["close"].to_numpy(dtype=np.float32),
            baseline_close=baseline.to_numpy(dtype=np.float32),
            stock_features=stock_features,
            market_features=market_features,
            stock_names=stock_names,
            market_names=market_names,
        )
        return

    stock_features, market_features, target, meta = build_samples(
        cfg=cfg,
        dates=panels["close"].index,
        symbols=list(panels["close"].columns),
        close=panels["close"].to_numpy(dtype=np.float32),
        baseline_close=baseline.to_numpy(dtype=np.float32),
        stock_features=stock_features,
        market_features=market_features,
    )
    split = time_splits(meta["signal_time"], cfg)
    stock_features, market_features, norm = robust_normalize(stock_features, market_features, split["train"])

    out_path = output_root / "data" / "dataset_small.npz"
    np.savez_compressed(
        out_path,
        stock=stock_features.astype(np.float32),
        market=market_features.astype(np.float32),
        y=target.astype(np.float32),
        y_relative=target.astype(np.float32),
        y_abs=meta["y_abs"].astype(np.float32),
        y_baseline=meta["y_baseline"].astype(np.float32),
        signal_time=meta["signal_time"].astype("datetime64[D]"),
        instrument=meta["instrument"].astype("U16"),
        split_train=split["train"],
        split_val=split["val"],
        split_test=split["test"],
        stock_feature_names=np.asarray(stock_names, dtype="U64"),
        market_feature_names=np.asarray(market_names, dtype="U64"),
        stock_center=norm["stock_center"].astype(np.float32),
        stock_scale=norm["stock_scale"].astype(np.float32),
        market_center=norm["market_center"].astype(np.float32),
        market_scale=norm["market_scale"].astype(np.float32),
    )
    manifest = {
        "config": asdict(cfg),
        "dataset": str(out_path),
        "samples": int(len(target)),
        "stock_shape": list(stock_features.shape),
        "market_shape": list(market_features.shape),
        "target_shape": list(target.shape),
        "train_samples": int(split["train"].sum()),
        "val_samples": int(split["val"].sum()),
        "test_samples": int(split["test"].sum()),
        "stock_feature_names": stock_names,
        "market_feature_names": market_names,
        "target": "future 1..30d cumulative stock return minus cumulative baseline return from condition day",
        "target_fields": {
            "y": "alias of y_relative, kept for backward compatibility",
            "y_relative": "future stock cumulative return minus future baseline cumulative return",
            "y_abs": "future stock cumulative return from condition day",
            "y_baseline": "future baseline cumulative return from condition day",
        },
        "input_note": "stock and market sequences share 60 time tokens; embeddings are concatenated on channel dimension",
    }
    (output_root / "data" / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[build] wrote {out_path} samples={len(target)}", flush=True)


def load_symbols(path: Path, max_instruments: int, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    records: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        symbol = parts[0].upper()
        if not symbol.startswith(MAINBOARD_PREFIXES):
            continue
        listed = pd.Timestamp(parts[1])
        retired = pd.Timestamp(parts[2])
        if listed <= start and retired >= end:
            records.append((symbol, listed, retired))
    symbols = [symbol for symbol, _, _ in sorted(records)]
    if max_instruments > 0:
        return symbols[:max_instruments]
    return symbols


def frame_to_panels(
    raw: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    fields: list[str],
) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    frame = raw.copy()
    for field in fields:
        if field not in frame:
            continue
        name = field.lower().lstrip("$")
        wide = frame[field].unstack("instrument")
        wide = wide.reindex(index=dates, columns=symbols).astype("float32")
        panels[name] = wide
    required = {"open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required - set(panels))
    if missing:
        raise RuntimeError(f"Missing required fields from qlib data: {missing}")
    if "vwap" not in panels:
        panels["vwap"] = panels["close"].copy()
    if "turnover_rate" not in panels:
        panels["turnover_rate"] = panels["volume"].copy() * np.nan
    return panels


def load_baseline(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    if path.exists():
        frame = pd.read_parquet(path)
        date_col = "date" if "date" in frame else "datetime"
        close_col = "close" if "close" in frame else "$close"
        series = pd.Series(
            pd.to_numeric(frame[close_col], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(frame[date_col]),
        )
        return series.reindex(dates).ffill().astype("float32")
    raise FileNotFoundError(f"Missing baseline path: {path}")


def build_stock_features(panels: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    close = panels["close"].to_numpy(dtype=np.float32)
    open_ = panels["open"].to_numpy(dtype=np.float32)
    high = panels["high"].to_numpy(dtype=np.float32)
    low = panels["low"].to_numpy(dtype=np.float32)
    vwap = panels["vwap"].to_numpy(dtype=np.float32)
    volume = panels["volume"].to_numpy(dtype=np.float32)
    amount = panels["amount"].to_numpy(dtype=np.float32)
    turnover = panels["turnover_rate"].to_numpy(dtype=np.float32)
    prev_close = shift(close, 1)
    prev_volume = shift(volume, 1)
    prev_amount = shift(amount, 1)
    prev_turnover = shift(turnover, 1)

    ret1 = safe_ratio(close, prev_close) - 1.0
    features = [
        ret1,
        safe_ratio(open_, prev_close) - 1.0,
        safe_ratio(high, prev_close) - 1.0,
        safe_ratio(low, prev_close) - 1.0,
        safe_ratio(vwap, prev_close) - 1.0,
        log_change(volume, prev_volume),
        log_change(amount, prev_amount),
        safe_ratio(turnover, prev_turnover) - 1.0,
        safe_ratio(high, low) - 1.0,
        safe_ratio(close - low, high - low),
        safe_ratio(close, shift(close, 3)) - 1.0,
        safe_ratio(close, shift(close, 5)) - 1.0,
        safe_ratio(close, shift(close, 10)) - 1.0,
        safe_ratio(close, shift(close, 20)) - 1.0,
        safe_ratio(close, rolling_mean(close, 5)) - 1.0,
        safe_ratio(close, rolling_mean(close, 10)) - 1.0,
        safe_ratio(close, rolling_mean(close, 20)) - 1.0,
        rolling_std(ret1, 10),
        rolling_std(ret1, 20),
    ]
    names = [
        "close_ret_1d",
        "open_rel_prev_close",
        "high_rel_prev_close",
        "low_rel_prev_close",
        "vwap_rel_prev_close",
        "volume_log_chg_1d",
        "amount_log_chg_1d",
        "turnover_chg_1d",
        "amplitude_pct",
        "close_pos",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ma5_rel",
        "ma10_rel",
        "ma20_rel",
        "vol_10d",
        "vol_20d",
    ]
    arr = np.stack(features, axis=2)
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
    return arr, names


def build_market_features(
    panels: dict[str, pd.DataFrame],
    stock_features: np.ndarray,
    stock_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    close = panels["close"].to_numpy(dtype=np.float32)
    amount = panels["amount"].to_numpy(dtype=np.float32)
    ret1 = stock_features[:, :, stock_names.index("close_ret_1d")]
    ret3 = stock_features[:, :, stock_names.index("ret_3d")]
    ret5 = stock_features[:, :, stock_names.index("ret_5d")]
    close_pos = stock_features[:, :, stock_names.index("close_pos")]
    amount_chg = stock_features[:, :, stock_names.index("amount_log_chg_1d")]

    q70_ret = nanpercentile_2d(ret1, 70.0)
    q70_amount = nanpercentile_2d(amount_chg, 70.0)
    brick = (shift(ret3, 1) < 0.0) & (ret1 > 0.0) & (close_pos > 0.65) & (amount_chg > 0.0)
    weak_to_strong = (ret5 < 0.0) & (ret1 > q70_ret[:, None]) & (amount_chg > q70_amount[:, None])

    total_amount = np.nansum(np.where(np.isfinite(amount) & (amount > 0), amount, 0.0), axis=1)
    market_features = [
        nanmean(ret1),
        nanmedian(ret1),
        nanstd(ret1),
        nanmean(ret1 > 0.0),
        nanmean(ret1 >= 0.04),
        nanmean(ret1 <= -0.04),
        log_change_1d(total_amount),
        top_amount_share(amount, 10),
        top_amount_share(amount, 20),
        top_amount_share(amount, 50),
        top_mean(ret1, 20),
        bottom_mean(ret1, 20),
        top_mean(ret1, 20) - bottom_mean(ret1, 20),
        nanmean(brick.astype(float)),
        nanmean(weak_to_strong.astype(float)),
        nanmean(safe_ratio(close, rolling_mean(close, 20)) - 1.0),
        nanstd(safe_ratio(close, shift(close, 20)) - 1.0),
    ]
    names = [
        "market_ret_mean",
        "market_ret_median",
        "market_ret_std",
        "advancer_ratio",
        "strong4_ratio",
        "weak4_ratio",
        "amount_total_log_chg",
        "amount_top10_share",
        "amount_top20_share",
        "amount_top50_share",
        "ret1_top20_mean",
        "ret1_bottom20_mean",
        "ret1_top_bottom_spread",
        "brick_reversal_ratio",
        "weak_to_strong_ratio",
        "market_ma20_rel_mean",
        "market_ret20_std",
    ]
    arr = np.stack(market_features, axis=1).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, names


def build_samples(
    *,
    cfg: Config,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    close: np.ndarray,
    baseline_close: np.ndarray,
    stock_features: np.ndarray,
    market_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    stock_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    abs_targets: list[np.ndarray] = []
    baseline_targets: list[np.ndarray] = []
    signal_times: list[np.datetime64] = []
    instruments: list[str] = []

    for sample in iter_samples(
        cfg=cfg,
        dates=dates,
        symbols=symbols,
        close=close,
        baseline_close=baseline_close,
        stock_features=stock_features,
        market_features=market_features,
    ):
        stock_parts.append(sample["stock"])
        market_parts.append(sample["market"])
        targets.append(sample["y"])
        abs_targets.append(sample["y_abs"])
        baseline_targets.append(sample["y_baseline"])
        signal_times.append(sample["signal_time"])
        instruments.append(sample["instrument"])

    if not targets:
        raise RuntimeError("No training samples were generated")
    stock = np.stack(stock_parts).astype(np.float32)
    market = np.stack(market_parts).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    meta = {
        "signal_time": np.asarray(signal_times),
        "instrument": np.asarray(instruments),
        "y_abs": np.stack(abs_targets).astype(np.float32),
        "y_baseline": np.stack(baseline_targets).astype(np.float32),
    }
    return stock, market, y, meta


def iter_samples(
    *,
    cfg: Config,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    close: np.ndarray,
    baseline_close: np.ndarray,
    stock_features: np.ndarray,
    market_features: np.ndarray,
) -> Any:
    start_pos = int(dates.searchsorted(pd.Timestamp(cfg.start), side="left"))
    end_pos = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
    first_signal = max(start_pos, cfg.lookback)
    last_signal = min(end_pos - cfg.horizon, len(dates) - cfg.horizon - 1)
    rng = np.random.default_rng(cfg.seed)
    emitted = 0

    candidate_dates = list(range(first_signal, last_signal + 1, max(1, cfg.sample_every)))
    for t in candidate_dates:
        market_window = market_features[t - cfg.lookback + 1 : t + 1]
        if not finite_enough(market_window, 0.85):
            continue
        base0 = baseline_close[t]
        base_future = baseline_close[t + 1 : t + cfg.horizon + 1]
        if not np.isfinite(base0) or base0 <= 0 or not np.all(np.isfinite(base_future)):
            continue
        baseline_path = base_future / base0 - 1.0
        order = np.arange(len(symbols))
        rng.shuffle(order)
        for j in order:
            window = stock_features[t - cfg.lookback + 1 : t + 1, j, :]
            c0 = close[t, j]
            cf = close[t + 1 : t + cfg.horizon + 1, j]
            if not np.isfinite(c0) or c0 <= 0 or not np.all(np.isfinite(cf)):
                continue
            if not finite_enough(window, 0.85):
                continue
            stock_path = cf / c0 - 1.0
            y = stock_path - baseline_path
            if not np.all(np.isfinite(y)):
                continue
            emitted += 1
            yield {
                "stock": window.astype(np.float32),
                "market": market_window.astype(np.float32),
                "y": y.astype(np.float32),
                "y_abs": stock_path.astype(np.float32),
                "y_baseline": baseline_path.astype(np.float32),
                "signal_time": np.datetime64(dates[t].date()),
                "instrument": symbols[j],
            }
            if cfg.max_samples > 0 and emitted >= cfg.max_samples:
                break
        if cfg.max_samples > 0 and emitted >= cfg.max_samples:
            break
        if emitted and emitted % 50000 == 0:
            print(f"[build] samples={emitted} latest={dates[t].date()}", flush=True)


def build_sharded_memmap_dataset(
    *,
    cfg: Config,
    output_root: Path,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    close: np.ndarray,
    baseline_close: np.ndarray,
    stock_features: np.ndarray,
    market_features: np.ndarray,
    stock_names: list[str],
    market_names: list[str],
) -> None:
    import yaml

    split_for_date = make_split_classifier(cfg, dates)
    stock_features, market_features, norm = robust_normalize_panels_for_shards(stock_features, market_features, dates, split_for_date)
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for sample in iter_samples(
        cfg=cfg,
        dates=dates,
        symbols=symbols,
        close=close,
        baseline_close=baseline_close,
        stock_features=stock_features,
        market_features=market_features,
    ):
        split = split_for_date(pd.Timestamp(sample["signal_time"]))
        if split is None:
            continue
        key = shard_key(split, pd.Timestamp(sample["signal_time"]))
        counts[key] += 1
        total += 1
    if total == 0:
        raise RuntimeError("No sharded samples were generated")

    data_root = output_root / "data"
    shards_root = data_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    writers = create_shard_writers(
        shards_root=shards_root,
        counts=counts,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stock_dim=stock_features.shape[2],
        market_dim=market_features.shape[1],
    )
    positions = {key: 0 for key in counts}
    symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
    for sample in iter_samples(
        cfg=cfg,
        dates=dates,
        symbols=symbols,
        close=close,
        baseline_close=baseline_close,
        stock_features=stock_features,
        market_features=market_features,
    ):
        signal_time = pd.Timestamp(sample["signal_time"])
        split = split_for_date(signal_time)
        if split is None:
            continue
        key = shard_key(split, signal_time)
        pos = positions[key]
        writer = writers[key]
        writer["stock"][pos] = sample["stock"]
        writer["market"][pos] = sample["market"]
        writer["y"][pos] = sample["y"]
        writer["y_abs"][pos] = sample["y_abs"]
        writer["y_baseline"][pos] = sample["y_baseline"]
        writer["signal_time"][pos] = date_to_day_int(signal_time)
        writer["instrument_id"][pos] = symbol_to_id[str(sample["instrument"])]
        positions[key] += 1
        if sum(positions.values()) % 100000 == 0:
            print(f"[build][sharded] wrote={sum(positions.values())}/{total}", flush=True)
    for writer in writers.values():
        for value in writer.values():
            if hasattr(value, "flush"):
                value.flush()

    shards = []
    for key in sorted(counts):
        split, year = key.split("_")
        shard_dir = shards_root / key
        manifest = {
            "name": key,
            "split": split,
            "year": int(year),
            "samples": int(counts[key]),
            "path": str(shard_dir.relative_to(data_root)),
        }
        (shard_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shards.append(manifest)

    dataset = {
        "format": "sharded_memmap",
        "config": asdict(cfg),
        "lookback": int(cfg.lookback),
        "horizon": int(cfg.horizon),
        "stock_dim": int(stock_features.shape[2]),
        "market_dim": int(market_features.shape[1]),
        "dtype": "float32",
        "samples": int(total),
        "shards": shards,
        "splits": {split: [s["name"] for s in shards if s["split"] == split] for split in ("train", "val", "test")},
        "symbols": symbols,
        "stock_feature_names": stock_names,
        "market_feature_names": market_names,
        "normalization": {
            "stock_center": norm["stock_center"].astype(float).tolist(),
            "stock_scale": norm["stock_scale"].astype(float).tolist(),
            "market_center": norm["market_center"].astype(float).tolist(),
            "market_scale": norm["market_scale"].astype(float).tolist(),
        },
    }
    (data_root / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (data_root / "dataset_manifest.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[build][sharded] wrote shards={len(shards)} samples={total} root={data_root / 'dataset.yaml'}", flush=True)


def make_split_classifier(cfg: Config, dates: pd.DatetimeIndex) -> Any:
    start_pos = int(dates.searchsorted(pd.Timestamp(cfg.start), side="left"))
    end_pos = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
    first_signal = max(start_pos, cfg.lookback)
    last_signal = min(end_pos - cfg.horizon, len(dates) - cfg.horizon - 1)
    signal_dates = pd.DatetimeIndex(dates[first_signal : last_signal + 1 : max(1, cfg.sample_every)]).normalize()
    if len(signal_dates) < 5:
        raise RuntimeError("Not enough signal dates for sharded split")
    explicit = bool(cfg.train_end or cfg.test_start)
    val_end_ts: pd.Timestamp | None = None
    if explicit:
        if not cfg.train_end or not cfg.test_start:
            raise ValueError("--train-end and --test-start must be provided together")
        train_end_ts = pd.Timestamp(cfg.train_end).normalize()
        test_start_ts = pd.Timestamp(cfg.test_start).normalize()
        train_pool = signal_dates[signal_dates <= train_end_ts]
        if len(train_pool) < 5:
            raise RuntimeError("Not enough train dates before --train-end")
        val_ratio = min(max(float(cfg.val_ratio_after_train), 0.0), 0.5)
        val_dates = max(1, int(round(len(train_pool) * val_ratio))) if val_ratio > 0 else 1
        val_start = train_pool[-val_dates]
    else:
        unique = np.array(sorted(pd.unique(signal_dates)))
        train_end_ts = pd.Timestamp(unique[int(len(unique) * 0.70)]).normalize()
        val_end_ts = pd.Timestamp(unique[int(len(unique) * 0.85)]).normalize()
        val_start = train_end_ts + pd.Timedelta(days=1)
        test_start_ts = val_end_ts + pd.Timedelta(days=1)

    def classify(value: pd.Timestamp) -> str | None:
        day = pd.Timestamp(value).normalize()
        if explicit:
            if day < val_start:
                return "train"
            if day <= train_end_ts:
                return "val"
            if day >= test_start_ts:
                return "test"
            return None
        if day <= train_end_ts:
            return "train"
        if val_end_ts is not None and day <= val_end_ts:
            return "val"
        if day >= test_start_ts:
            return "test"
        return None

    return classify


def robust_normalize_panels_for_shards(
    stock: np.ndarray,
    market: np.ndarray,
    dates: pd.DatetimeIndex,
    split_for_date: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_days = np.asarray([split_for_date(pd.Timestamp(day)) == "train" for day in dates], dtype=bool)
    if not train_days.any():
        raise RuntimeError("No train panel dates for normalization")
    stock_center, stock_scale = robust_stats(stock[train_days], axes=(0, 1))
    market_center, market_scale = robust_stats(market[train_days], axes=(0,))
    stock = np.clip((np.nan_to_num(stock, nan=0.0, posinf=0.0, neginf=0.0) - stock_center) / stock_scale, -8.0, 8.0)
    market = np.clip((np.nan_to_num(market, nan=0.0, posinf=0.0, neginf=0.0) - market_center) / market_scale, -8.0, 8.0)
    return stock.astype(np.float32), market.astype(np.float32), {
        "stock_center": stock_center,
        "stock_scale": stock_scale,
        "market_center": market_center,
        "market_scale": market_scale,
    }


def shard_key(split: str, signal_time: pd.Timestamp) -> str:
    return f"{split}_{pd.Timestamp(signal_time).year}"


def date_to_day_int(value: pd.Timestamp) -> int:
    return int(np.datetime64(pd.Timestamp(value).date(), "D").astype("int64"))


def day_int_to_str(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[D]").astype(str)


def create_shard_writers(
    *,
    shards_root: Path,
    counts: dict[str, int],
    lookback: int,
    horizon: int,
    stock_dim: int,
    market_dim: int,
) -> dict[str, dict[str, Any]]:
    writers = {}
    for key, n in counts.items():
        shard_dir = shards_root / key
        shard_dir.mkdir(parents=True, exist_ok=True)
        writers[key] = {
            "stock": np.lib.format.open_memmap(shard_dir / "stock.npy", mode="w+", dtype="float32", shape=(n, lookback, stock_dim)),
            "market": np.lib.format.open_memmap(shard_dir / "market.npy", mode="w+", dtype="float32", shape=(n, lookback, market_dim)),
            "y": np.lib.format.open_memmap(shard_dir / "y.npy", mode="w+", dtype="float32", shape=(n, horizon)),
            "y_abs": np.lib.format.open_memmap(shard_dir / "y_abs.npy", mode="w+", dtype="float32", shape=(n, horizon)),
            "y_baseline": np.lib.format.open_memmap(shard_dir / "y_baseline.npy", mode="w+", dtype="float32", shape=(n, horizon)),
            "signal_time": np.lib.format.open_memmap(shard_dir / "signal_time.npy", mode="w+", dtype="int32", shape=(n,)),
            "instrument_id": np.lib.format.open_memmap(shard_dir / "instrument_id.npy", mode="w+", dtype="int32", shape=(n,)),
        }
    return writers


def time_splits(signal_time: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(signal_time)
    unique = np.array(sorted(pd.unique(dates)))
    if len(unique) < 5:
        raise RuntimeError("Not enough signal dates for time split")
    if cfg.train_end or cfg.test_start:
        if not cfg.train_end or not cfg.test_start:
            raise ValueError("--train-end and --test-start must be provided together")
        train_end_ts = pd.Timestamp(cfg.train_end)
        test_start_ts = pd.Timestamp(cfg.test_start)
        train_pool = unique[unique <= train_end_ts]
        if len(train_pool) < 5:
            raise RuntimeError("Not enough train dates before --train-end")
        val_ratio = min(max(float(cfg.val_ratio_after_train), 0.0), 0.5)
        val_dates = max(1, int(round(len(train_pool) * val_ratio))) if val_ratio > 0 else 1
        split_point = train_pool[-val_dates]
        train = dates < split_point
        val = (dates >= split_point) & (dates <= train_end_ts)
        test = dates >= test_start_ts
        if not np.asarray(train).any() or not np.asarray(val).any() or not np.asarray(test).any():
            raise RuntimeError(
                "Explicit time split produced an empty split: "
                f"train={int(np.asarray(train).sum())} val={int(np.asarray(val).sum())} test={int(np.asarray(test).sum())}"
            )
        return {"train": np.asarray(train), "val": np.asarray(val), "test": np.asarray(test)}
    train_end = unique[int(len(unique) * 0.70)]
    val_end = unique[int(len(unique) * 0.85)]
    train = dates <= train_end
    val = (dates > train_end) & (dates <= val_end)
    test = dates > val_end
    return {"train": np.asarray(train), "val": np.asarray(val), "test": np.asarray(test)}


def robust_normalize(
    stock: np.ndarray,
    market: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    stock_center, stock_scale = robust_stats(stock[train_mask], axes=(0, 1))
    market_center, market_scale = robust_stats(market[train_mask], axes=(0, 1))
    stock = np.clip((np.nan_to_num(stock, nan=0.0, posinf=0.0, neginf=0.0) - stock_center) / stock_scale, -8.0, 8.0)
    market = np.clip((np.nan_to_num(market, nan=0.0, posinf=0.0, neginf=0.0) - market_center) / market_scale, -8.0, 8.0)
    return stock.astype(np.float32), market.astype(np.float32), {
        "stock_center": stock_center,
        "stock_scale": stock_scale,
        "market_center": market_center,
        "market_scale": market_scale,
    }


def robust_stats(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=axes, keepdims=False).astype(np.float32)
    q25 = np.nanpercentile(values, 25, axis=axes).astype(np.float32)
    q75 = np.nanpercentile(values, 75, axis=axes).astype(np.float32)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-6), scale, 1.0).astype(np.float32)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    return center, scale


def load_sharded_manifest(root: Path) -> dict[str, Any]:
    import yaml

    manifest_path = root / "data" / "dataset.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing sharded dataset manifest: {manifest_path}")
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


class ShardedMemmapDataset:
    def __init__(self, root: Path, manifest: dict[str, Any], split: str, *, include_y: bool):
        self.root = root
        self.data_root = root / "data"
        self.manifest = manifest
        self.split = split
        self.include_y = include_y
        names = manifest.get("splits", {}).get(split, [])
        shard_by_name = {shard["name"]: shard for shard in manifest.get("shards", [])}
        self.shards = []
        self.offsets = [0]
        for name in names:
            shard = shard_by_name[name]
            path = self.data_root / shard["path"]
            item = {
                "name": name,
                "path": path,
                "stock": np.load(path / "stock.npy", mmap_mode="r"),
                "market": np.load(path / "market.npy", mmap_mode="r"),
                "signal_time": np.load(path / "signal_time.npy", mmap_mode="r"),
                "instrument_id": np.load(path / "instrument_id.npy", mmap_mode="r"),
            }
            if include_y:
                item["y"] = np.load(path / "y.npy", mmap_mode="r")
            self.shards.append(item)
            self.offsets.append(self.offsets[-1] + int(shard["samples"]))
        if self.offsets[-1] == 0:
            raise RuntimeError(f"No samples for split={split} in sharded dataset")

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Any:
        import torch

        shard_idx = bisect_right(self.offsets, int(index)) - 1
        local_idx = int(index) - self.offsets[shard_idx]
        shard = self.shards[shard_idx]
        stock = torch.from_numpy(np.array(shard["stock"][local_idx], dtype=np.float32, copy=True))
        market = torch.from_numpy(np.array(shard["market"][local_idx], dtype=np.float32, copy=True))
        if not self.include_y:
            return stock, market
        y = torch.from_numpy(np.array(shard["y"][local_idx], dtype=np.float32, copy=True))
        return stock, market, y

    def meta_frame(self) -> pd.DataFrame:
        symbols = list(self.manifest.get("symbols", []))
        frames = []
        for shard in self.shards:
            signal_time = day_int_to_str(np.asarray(shard["signal_time"]))
            instrument_id = np.asarray(shard["instrument_id"], dtype=np.int64)
            frames.append(
                pd.DataFrame(
                    {
                        "signal_time": signal_time,
                        "instrument": [symbols[i] if 0 <= i < len(symbols) else str(i) for i in instrument_id],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["signal_time", "instrument"])

    def column_array(self, name: str) -> np.ndarray:
        arrays = []
        for shard in self.shards:
            path = shard["path"] / f"{name}.npy"
            arrays.append(np.asarray(np.load(path, mmap_mode="r")))
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0,), dtype=np.float32)


class BlockShuffleSampler:
    def __init__(self, length: int, block_size: int, seed: int):
        self.length = int(length)
        self.block_size = max(1, int(block_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Any:
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = np.arange(0, self.length, self.block_size)
        rng.shuffle(starts)
        self.epoch += 1
        for start in starts:
            end = min(int(start) + self.block_size, self.length)
            for idx in range(int(start), end):
                yield idx

    def __len__(self) -> int:
        return self.length


def sharded_loader(dataset: Any, cfg: Config, *, train: bool) -> Any:
    from torch.utils.data import DataLoader

    sampler = BlockShuffleSampler(len(dataset), cfg.shuffle_block_size, cfg.seed) if train else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        drop_last=False,
        num_workers=cfg.num_workers,
    )


def should_log_batch(batch_idx: int, total_batches: int, cfg: Config) -> bool:
    interval = max(0, int(cfg.log_every_batches))
    return batch_idx == 1 or batch_idx == total_batches or (interval > 0 and batch_idx % interval == 0)


def log_batch_progress(
    *,
    tag: str,
    epoch: int,
    step: int,
    batch_idx: int,
    total_batches: int,
    seen: int,
    total_samples: int,
    losses: list[float],
    start_time: float,
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    samples_per_second = seen / elapsed
    print(
        f"[{tag}] epoch={epoch} step={step} batch={batch_idx}/{total_batches} "
        f"seen={seen}/{total_samples} loss_avg={np.mean(losses):.6f} "
        f"elapsed_s={elapsed:.1f} samples_per_s={samples_per_second:.1f}",
        flush=True,
    )


def train_models(cfg: Config) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if cfg.dataset_format == "sharded_memmap":
        train_models_sharded(cfg)
        return

    set_seed(cfg.seed)
    device = default_device()
    print(f"[train] device={device}", flush=True)
    data = np.load(Path(cfg.output_root) / "data" / "dataset_small.npz", allow_pickle=False)
    stock = data["stock"].astype(np.float32)
    market = data["market"].astype(np.float32)
    y = data["y"].astype(np.float32)
    train_mask = data["split_train"].astype(bool)
    val_mask = data["split_val"].astype(bool)
    check_compression(cfg, stock.shape[1], stock.shape[2], market.shape[2])

    ae = TransformerAutoEncoder(
        stock_dim=stock.shape[2],
        market_dim=market.shape[2],
        lookback=stock.shape[1],
        d_stock=cfg.d_stock,
        d_market=cfg.d_market,
        d_model=cfg.d_model,
        latent_tokens=cfg.latent_tokens,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(stock[train_mask]), torch.from_numpy(market[train_mask])),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_tensors = (torch.from_numpy(stock[val_mask]).to(device), torch.from_numpy(market[val_mask]).to(device))
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for epoch in range(1, cfg.ae_epochs + 1):
        ae.train()
        losses = []
        for xb_stock, xb_market in train_loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            noisy_stock = mask_tensor(xb_stock, 0.15)
            noisy_market = mask_tensor(xb_market, 0.15)
            recon_stock, recon_market, _ = ae(noisy_stock, noisy_market)
            loss = torch.nn.functional.smooth_l1_loss(recon_stock, xb_stock) + 0.7 * torch.nn.functional.smooth_l1_loss(
                recon_market, xb_market
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = eval_ae(ae, val_tensors)
        print(f"[train][ae] epoch={epoch} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}", flush=True)
    torch.save({"model": ae.state_dict(), "config": asdict(cfg)}, Path(cfg.output_root) / "runs" / "ae_checkpoint.pt")

    val_recon = ae_reconstruction_summary(
        ae=ae,
        stock=stock[val_mask],
        market=market[val_mask],
        batch_size=cfg.batch_size,
        device=device,
        masked=False,
        stock_names=[f"stock_{i}" for i in range(stock.shape[2])],
        market_names=[f"market_{i}" for i in range(market.shape[2])],
    )[0]
    val_nmae = val_recon.get("combined_nmae")
    print(
        f"[train][ae] val_recon combined_nmae={val_nmae:.6f} "
        f"stock_nmae={val_recon.get('stock_nmae'):.6f} market_nmae={val_recon.get('market_nmae'):.6f} "
        f"gate={cfg.ae_recon_gate_nmae:.6f}",
        flush=True,
    )
    if cfg.ae_recon_gate_nmae > 0 and (val_nmae is None or val_nmae > cfg.ae_recon_gate_nmae):
        raise RuntimeError(
            "AE reconstruction gate failed; diffusion/flow training is blocked. "
            f"val_combined_nmae={val_nmae} gate={cfg.ae_recon_gate_nmae}. "
            "Increase --ae-epochs, reduce compression, or set a looser --ae-recon-gate-nmae only for debugging."
        )

    for param in ae.parameters():
        param.requires_grad_(False)
    ae.eval()
    diffusion = make_denoiser(cfg, target_dim=y.shape[1]).to(device)
    print(
        f"[train][{cfg.denoiser_type}] params={count_parameters(diffusion):,} total_with_ae={count_parameters(ae) + count_parameters(diffusion):,}",
        flush=True,
    )
    schedule = DiffusionSchedule(cfg.diffusion_steps, device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(stock[train_mask]), torch.from_numpy(market[train_mask]), torch.from_numpy(y[train_mask])),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for epoch in range(1, cfg.diffusion_epochs + 1):
        diffusion.train()
        losses = []
        for xb_stock, xb_market, yb in train_loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                cond = encode_condition(ae, xb_stock, xb_market, cfg)
            if cfg.generative_objective == "flow":
                loss = flow_matching_loss(diffusion, yb, cond)
            else:
                step = torch.randint(0, cfg.diffusion_steps, (yb.shape[0],), device=device)
                noise = torch.randn_like(yb)
                y_noisy = schedule.q_sample(yb, step, noise)
                noise_hat = diffusion(y_noisy, step, cond)
                loss = torch.nn.functional.smooth_l1_loss(noise_hat, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if cfg.generative_objective == "flow":
            val_loss = eval_flow(ae, diffusion, stock[val_mask], market[val_mask], y[val_mask], cfg, device)
        else:
            val_loss = eval_diffusion(ae, diffusion, schedule, stock[val_mask], market[val_mask], y[val_mask], cfg, device)
        print(
            f"[train][{cfg.generative_objective}] epoch={epoch} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
    torch.save({"model": diffusion.state_dict(), "config": asdict(cfg)}, Path(cfg.output_root) / "runs" / "diffusion_checkpoint.pt")


def train_models_sharded(cfg: Config) -> None:
    import torch

    set_seed(cfg.seed)
    device = default_device()
    root = Path(cfg.output_root)
    manifest = load_sharded_manifest(root)
    print(f"[train][sharded] device={device} samples={manifest.get('samples')}", flush=True)
    check_compression(cfg, int(manifest["lookback"]), int(manifest["stock_dim"]), int(manifest["market_dim"]))

    ae = TransformerAutoEncoder(
        stock_dim=int(manifest["stock_dim"]),
        market_dim=int(manifest["market_dim"]),
        lookback=int(manifest["lookback"]),
        d_stock=cfg.d_stock,
        d_market=cfg.d_market,
        d_model=cfg.d_model,
        latent_tokens=cfg.latent_tokens,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    train_ae_ds = ShardedMemmapDataset(root, manifest, "train", include_y=False)
    val_ae_ds = ShardedMemmapDataset(root, manifest, "val", include_y=False)
    train_loader = sharded_loader(train_ae_ds, cfg, train=True)
    val_loader = sharded_loader(val_ae_ds, cfg, train=False)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=1e-4)
    global_step = 0
    for epoch in range(1, cfg.ae_epochs + 1):
        ae.train()
        losses = []
        epoch_start = time.time()
        seen = 0
        total_batches = len(train_loader)
        total_samples = len(train_ae_ds)
        for batch_idx, (xb_stock, xb_market) in enumerate(train_loader, start=1):
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            noisy_stock = mask_tensor(xb_stock, 0.15)
            noisy_market = mask_tensor(xb_market, 0.15)
            recon_stock, recon_market, _ = ae(noisy_stock, noisy_market)
            loss = torch.nn.functional.smooth_l1_loss(recon_stock, xb_stock) + 0.7 * torch.nn.functional.smooth_l1_loss(
                recon_market, xb_market
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
            global_step += 1
            seen += int(xb_stock.shape[0])
            losses.append(float(loss.detach().cpu()))
            if should_log_batch(batch_idx, total_batches, cfg):
                log_batch_progress(
                    tag="train][ae][sharded",
                    epoch=epoch,
                    step=global_step,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    seen=seen,
                    total_samples=total_samples,
                    losses=losses,
                    start_time=epoch_start,
                )
        val_loss = eval_ae_loader(ae, val_loader, device)
        print(f"[train][ae][sharded] epoch={epoch} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}", flush=True)
        torch.save({"model": ae.state_dict(), "config": asdict(cfg), "epoch": epoch}, root / "runs" / f"ae_checkpoint_epoch_{epoch}.pt")
    torch.save({"model": ae.state_dict(), "config": asdict(cfg)}, root / "runs" / "ae_checkpoint.pt")

    val_recon = ae_reconstruction_summary_loader(
        ae=ae,
        loader=val_loader,
        device=device,
        masked=False,
        stock_names=[str(x) for x in manifest["stock_feature_names"]],
        market_names=[str(x) for x in manifest["market_feature_names"]],
    )[0]
    val_nmae = val_recon.get("combined_nmae")
    print(
        f"[train][ae][sharded] val_recon combined_nmae={val_nmae:.6f} "
        f"stock_nmae={val_recon.get('stock_nmae'):.6f} market_nmae={val_recon.get('market_nmae'):.6f} "
        f"gate={cfg.ae_recon_gate_nmae:.6f}",
        flush=True,
    )
    if cfg.ae_recon_gate_nmae > 0 and (val_nmae is None or val_nmae > cfg.ae_recon_gate_nmae):
        raise RuntimeError(
            "AE reconstruction gate failed; diffusion/flow training is blocked. "
            f"val_combined_nmae={val_nmae} gate={cfg.ae_recon_gate_nmae}."
        )

    for param in ae.parameters():
        param.requires_grad_(False)
    ae.eval()
    diffusion = make_denoiser(cfg, target_dim=int(manifest["horizon"])).to(device)
    print(
        f"[train][{cfg.denoiser_type}][sharded] params={count_parameters(diffusion):,} "
        f"total_with_ae={count_parameters(ae) + count_parameters(diffusion):,}",
        flush=True,
    )
    schedule = DiffusionSchedule(cfg.diffusion_steps, device)
    train_diff_ds = ShardedMemmapDataset(root, manifest, "train", include_y=True)
    val_diff_ds = ShardedMemmapDataset(root, manifest, "val", include_y=True)
    train_loader = sharded_loader(train_diff_ds, cfg, train=True)
    val_loader = sharded_loader(val_diff_ds, cfg, train=False)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr, weight_decay=1e-4)
    global_step = 0
    for epoch in range(1, cfg.diffusion_epochs + 1):
        diffusion.train()
        losses = []
        epoch_start = time.time()
        seen = 0
        total_batches = len(train_loader)
        total_samples = len(train_diff_ds)
        for batch_idx, (xb_stock, xb_market, yb) in enumerate(train_loader, start=1):
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                cond = encode_condition(ae, xb_stock, xb_market, cfg)
            if cfg.generative_objective == "flow":
                loss = flow_matching_loss(diffusion, yb, cond)
            else:
                step = torch.randint(0, cfg.diffusion_steps, (yb.shape[0],), device=device)
                noise = torch.randn_like(yb)
                y_noisy = schedule.q_sample(yb, step, noise)
                noise_hat = diffusion(y_noisy, step, cond)
                loss = torch.nn.functional.smooth_l1_loss(noise_hat, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            global_step += 1
            seen += int(xb_stock.shape[0])
            losses.append(float(loss.detach().cpu()))
            if should_log_batch(batch_idx, total_batches, cfg):
                log_batch_progress(
                    tag=f"train][{cfg.generative_objective}][sharded",
                    epoch=epoch,
                    step=global_step,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    seen=seen,
                    total_samples=total_samples,
                    losses=losses,
                    start_time=epoch_start,
                )
        val_loss = eval_flow_loader(ae, diffusion, val_loader, cfg, device) if cfg.generative_objective == "flow" else eval_diffusion_loader(
            ae, diffusion, schedule, val_loader, cfg, device
        )
        print(
            f"[train][{cfg.generative_objective}][sharded] epoch={epoch} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
    torch.save({"model": diffusion.state_dict(), "config": asdict(cfg)}, root / "runs" / "diffusion_checkpoint.pt")


def evaluate(cfg: Config) -> None:
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if cfg.dataset_format == "sharded_memmap":
        evaluate_sharded(cfg)
        return

    set_seed(cfg.seed)
    device = default_device()
    print(f"[eval] device={device}", flush=True)
    root = Path(cfg.output_root)
    data = np.load(root / "data" / "dataset_small.npz", allow_pickle=False)
    stock = data["stock"].astype(np.float32)
    market = data["market"].astype(np.float32)
    y = data["y"].astype(np.float32)
    y_abs = data["y_abs"].astype(np.float32) if "y_abs" in data.files else y.copy()
    y_baseline = data["y_baseline"].astype(np.float32) if "y_baseline" in data.files else np.zeros_like(y)
    train_mask = data["split_train"].astype(bool)
    test_mask = data["split_test"].astype(bool)
    signal_time = pd.to_datetime(data["signal_time"].astype(str))
    instrument = data["instrument"].astype(str)

    ae = TransformerAutoEncoder(
        stock_dim=stock.shape[2],
        market_dim=market.shape[2],
        lookback=stock.shape[1],
        d_stock=cfg.d_stock,
        d_market=cfg.d_market,
        d_model=cfg.d_model,
        latent_tokens=cfg.latent_tokens,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    ae.load_state_dict(torch.load(root / "runs" / "ae_checkpoint.pt", map_location=device)["model"])
    ae.eval()
    diffusion = make_denoiser(cfg, target_dim=y.shape[1]).to(device)
    diffusion.load_state_dict(torch.load(root / "runs" / "diffusion_checkpoint.pt", map_location=device)["model"], strict=False)
    diffusion.eval()
    schedule = DiffusionSchedule(cfg.diffusion_steps, device)
    pred = sample_predictions(ae, diffusion, schedule, stock[test_mask], market[test_mask], cfg, device)

    test_idx = np.flatnonzero(test_mask)
    frame = pd.DataFrame(
        {
            "signal_time": signal_time[test_idx].strftime("%Y-%m-%d"),
            "instrument": instrument[test_idx],
            "y_rel_5d": y[test_mask, 4],
            "y_rel_10d": y[test_mask, 9],
            "y_rel_20d": y[test_mask, 19],
            "y_rel_30d": y[test_mask, 29],
            "y_5d": y[test_mask, 4],
            "y_10d": y[test_mask, 9],
            "y_20d": y[test_mask, 19],
            "y_30d": y[test_mask, 29],
            "y_abs_5d": y_abs[test_mask, 4],
            "y_abs_10d": y_abs[test_mask, 9],
            "y_abs_20d": y_abs[test_mask, 19],
            "y_abs_30d": y_abs[test_mask, 29],
            "y_baseline_5d": y_baseline[test_mask, 4],
            "y_baseline_10d": y_baseline[test_mask, 9],
            "y_baseline_20d": y_baseline[test_mask, 19],
            "y_baseline_30d": y_baseline[test_mask, 29],
            "diff_mean_5d": pred["mean"][:, 4],
            "diff_mean_10d": pred["mean"][:, 9],
            "diff_mean_20d": pred["mean"][:, 19],
            "diff_mean_30d": pred["mean"][:, 29],
            "diff_std_5d": pred["std"][:, 4],
            "diff_std_10d": pred["std"][:, 9],
            "diff_std_20d": pred["std"][:, 19],
            "diff_std_30d": pred["std"][:, 29],
            "diff_prob_pos_5d": pred["prob_pos"][:, 4],
            "diff_prob_pos_10d": pred["prob_pos"][:, 9],
            "diff_prob_pos_20d": pred["prob_pos"][:, 19],
            "diff_prob_pos_30d": pred["prob_pos"][:, 29],
            "diff_q10_5d": pred["q10"][:, 4],
            "diff_q10_10d": pred["q10"][:, 9],
            "diff_q10_20d": pred["q10"][:, 19],
            "diff_q10_30d": pred["q10"][:, 29],
            "diff_q90_5d": pred["q90"][:, 4],
            "diff_q90_10d": pred["q90"][:, 9],
            "diff_q90_20d": pred["q90"][:, 19],
            "diff_q90_30d": pred["q90"][:, 29],
        }
    )
    frame["diff_score_10d"] = frame["diff_mean_10d"] / (frame["diff_std_10d"] + 1e-6)
    frame["baseline_momentum_20d"] = stock[test_mask, -1, 13]

    flat_train = np.concatenate([stock[train_mask].reshape(train_mask.sum(), -1), market[train_mask].reshape(train_mask.sum(), -1)], axis=1)
    flat_test = np.concatenate([stock[test_mask].reshape(test_mask.sum(), -1), market[test_mask].reshape(test_mask.sum(), -1)], axis=1)
    ridge = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=10.0))
    ridge.fit(flat_train, y[train_mask, 9])
    frame["baseline_ridge_10d"] = ridge.predict(flat_test)

    frame.to_parquet(root / "runs" / "predictions.parquet", index=False)
    metrics = build_metrics(frame)
    metrics_frame = pd.DataFrame(metrics["rows"])
    metrics_frame.to_csv(root / "runs" / "metrics.csv", index=False)
    (root / "runs" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_diagnostics(root, frame)
    write_inference_step_sweep(root, ae, diffusion, schedule, stock[test_mask], market[test_mask], frame, cfg, device)
    write_ae_reconstruction_metrics(root, ae, stock, market, data, cfg, device)
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2), flush=True)


def evaluate_sharded(cfg: Config) -> None:
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    set_seed(cfg.seed)
    device = default_device()
    root = Path(cfg.output_root)
    manifest = load_sharded_manifest(root)
    print(f"[eval][sharded] device={device} samples={manifest.get('samples')}", flush=True)

    ae = TransformerAutoEncoder(
        stock_dim=int(manifest["stock_dim"]),
        market_dim=int(manifest["market_dim"]),
        lookback=int(manifest["lookback"]),
        d_stock=cfg.d_stock,
        d_market=cfg.d_market,
        d_model=cfg.d_model,
        latent_tokens=cfg.latent_tokens,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    ae.load_state_dict(torch.load(root / "runs" / "ae_checkpoint.pt", map_location=device)["model"])
    ae.eval()

    diffusion = make_denoiser(cfg, target_dim=int(manifest["horizon"])).to(device)
    diffusion.load_state_dict(torch.load(root / "runs" / "diffusion_checkpoint.pt", map_location=device)["model"], strict=False)
    diffusion.eval()
    schedule = DiffusionSchedule(cfg.diffusion_steps, device)

    test_ds = ShardedMemmapDataset(root, manifest, "test", include_y=True)
    test_loader = sharded_loader(test_ds, cfg, train=False)
    pred = sample_predictions_loader(ae, diffusion, schedule, test_loader, cfg, device)
    y = test_ds.column_array("y").astype(np.float32)
    y_abs = test_ds.column_array("y_abs").astype(np.float32)
    y_baseline = test_ds.column_array("y_baseline").astype(np.float32)
    frame = prediction_frame_from_arrays(test_ds.meta_frame(), y, y_abs, y_baseline, pred)
    frame["baseline_momentum_20d"] = last_stock_feature(test_ds, 13)

    train_ds = ShardedMemmapDataset(root, manifest, "train", include_y=True)
    ridge_cap = 50_000
    flat_train, ridge_y = flatten_sharded_features(train_ds, max_samples=ridge_cap)
    flat_test, _ = flatten_sharded_features(test_ds, max_samples=0)
    ridge = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=10.0))
    ridge.fit(flat_train, ridge_y[:, 9])
    frame["baseline_ridge_10d"] = ridge.predict(flat_test)
    print(f"[eval][sharded] ridge_train_samples={len(flat_train)} test_samples={len(flat_test)}", flush=True)

    frame.to_parquet(root / "runs" / "predictions.parquet", index=False)
    metrics = build_metrics(frame)
    pd.DataFrame(metrics["rows"]).to_csv(root / "runs" / "metrics.csv", index=False)
    (root / "runs" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_diagnostics(root, frame)
    write_inference_step_sweep_loader(root, ae, diffusion, schedule, test_ds, frame, cfg, device)
    write_ae_reconstruction_metrics_sharded(root, ae, manifest, cfg, device)
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2), flush=True)


def prediction_frame_from_arrays(
    meta: pd.DataFrame,
    y: np.ndarray,
    y_abs: np.ndarray,
    y_baseline: np.ndarray,
    pred: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = meta.copy()
    for idx, suffix in ((4, "5d"), (9, "10d"), (19, "20d"), (29, "30d")):
        frame[f"y_rel_{suffix}"] = y[:, idx]
        frame[f"y_{suffix}"] = y[:, idx]
        frame[f"y_abs_{suffix}"] = y_abs[:, idx]
        frame[f"y_baseline_{suffix}"] = y_baseline[:, idx]
        frame[f"diff_mean_{suffix}"] = pred["mean"][:, idx]
        frame[f"diff_std_{suffix}"] = pred["std"][:, idx]
        frame[f"diff_prob_pos_{suffix}"] = pred["prob_pos"][:, idx]
        frame[f"diff_q10_{suffix}"] = pred["q10"][:, idx]
        frame[f"diff_q90_{suffix}"] = pred["q90"][:, idx]
    frame["diff_score_10d"] = frame["diff_mean_10d"] / (frame["diff_std_10d"] + 1e-6)
    return frame


def last_stock_feature(dataset: Any, feature_idx: int) -> np.ndarray:
    arrays = []
    for shard in dataset.shards:
        stock = np.load(shard["path"] / "stock.npy", mmap_mode="r")
        arrays.append(np.asarray(stock[:, -1, feature_idx], dtype=np.float32))
    return np.concatenate(arrays, axis=0) if arrays else np.empty((0,), dtype=np.float32)


def flatten_sharded_features(dataset: Any, max_samples: int) -> tuple[np.ndarray, np.ndarray | None]:
    total = len(dataset)
    if max_samples and total > max_samples:
        indices = np.linspace(0, total - 1, max_samples, dtype=np.int64)
    else:
        indices = np.arange(total, dtype=np.int64)
    stock_dim = int(dataset.manifest["lookback"]) * int(dataset.manifest["stock_dim"])
    market_dim = int(dataset.manifest["lookback"]) * int(dataset.manifest["market_dim"])
    x = np.empty((len(indices), stock_dim + market_dim), dtype=np.float32)
    y = np.empty((len(indices), int(dataset.manifest["horizon"])), dtype=np.float32) if dataset.include_y else None
    for out_idx, src_idx in enumerate(indices):
        shard_idx = bisect_right(dataset.offsets, int(src_idx)) - 1
        local_idx = int(src_idx) - dataset.offsets[shard_idx]
        shard = dataset.shards[shard_idx]
        stock = np.asarray(shard["stock"][local_idx], dtype=np.float32).reshape(-1)
        market = np.asarray(shard["market"][local_idx], dtype=np.float32).reshape(-1)
        x[out_idx, :stock_dim] = stock
        x[out_idx, stock_dim:] = market
        if y is not None:
            y[out_idx] = np.asarray(np.load(shard["path"] / "y.npy", mmap_mode="r")[local_idx], dtype=np.float32)
    return x, y


def build_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    score_cols = ["diff_score_10d", "diff_mean_10d", "baseline_momentum_20d", "baseline_ridge_10d"]
    for score in score_cols:
        rows.extend(ic_rows(frame, score, "y_10d"))
        rows.append(top_bottom_row(frame, score, "y_10d"))
    rows.extend(uncertainty_rows(frame))
    summary = {
        "rows": int(len(frame)),
        "days": int(frame["signal_time"].nunique()),
        "diff_score_rankic_mean": metric_lookup(rows, "diff_score_10d", "daily_rankic_mean"),
        "diff_mean_rankic_mean": metric_lookup(rows, "diff_mean_10d", "daily_rankic_mean"),
        "momentum_rankic_mean": metric_lookup(rows, "baseline_momentum_20d", "daily_rankic_mean"),
        "ridge_rankic_mean": metric_lookup(rows, "baseline_ridge_10d", "daily_rankic_mean"),
        "diff_top_bottom_spread": metric_lookup(rows, "diff_score_10d", "top_bottom_spread"),
        "ridge_top_bottom_spread": metric_lookup(rows, "baseline_ridge_10d", "top_bottom_spread"),
    }
    return {"summary": summary, "rows": rows}


def ic_rows(frame: pd.DataFrame, score: str, label: str) -> list[dict[str, Any]]:
    daily = []
    for day, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].dropna()
        if len(values) < 5 or values[score].nunique() < 2 or values[label].nunique() < 2:
            continue
        daily.append(
            {
                "signal_time": day,
                "spearman": values[score].corr(values[label], method="spearman"),
                "pearson": values[score].corr(values[label], method="pearson"),
                "n": len(values),
            }
        )
    daily_frame = pd.DataFrame(daily)
    if daily_frame.empty:
        return [{"score": score, "metric": "daily_rankic_mean", "value": None}]
    rankic = pd.to_numeric(daily_frame["spearman"], errors="coerce").dropna()
    pearson = pd.to_numeric(daily_frame["pearson"], errors="coerce").dropna()
    return [
        {"score": score, "metric": "daily_rankic_mean", "value": safe_float(rankic.mean())},
        {"score": score, "metric": "daily_rankic_median", "value": safe_float(rankic.median())},
        {"score": score, "metric": "daily_rankic_ir", "value": safe_float(rankic.mean() / (rankic.std(ddof=1) + 1e-12))},
        {"score": score, "metric": "daily_rankic_pos_ratio", "value": safe_float((rankic > 0).mean())},
        {"score": score, "metric": "daily_pearson_mean", "value": safe_float(pearson.mean())},
        {"score": score, "metric": "ic_days", "value": int(len(rankic))},
    ]


def top_bottom_row(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    spreads = []
    top_values = []
    bottom_values = []
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        bottom = values.head(n)[label].mean()
        top = values.tail(n)[label].mean()
        spreads.append(top - bottom)
        top_values.append(top)
        bottom_values.append(bottom)
    return {
        "score": score,
        "metric": "top_bottom_spread",
        "value": safe_float(np.mean(spreads)) if spreads else None,
        "top_mean": safe_float(np.mean(top_values)) if top_values else None,
        "bottom_mean": safe_float(np.mean(bottom_values)) if bottom_values else None,
        "days": len(spreads),
    }


def uncertainty_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    values = frame.dropna(subset=["diff_std_10d", "diff_score_10d", "y_10d"]).copy()
    if values.empty:
        return out
    values["uncertainty_bucket"] = pd.qcut(values["diff_std_10d"], 3, labels=["low", "mid", "high"], duplicates="drop")
    for bucket, group in values.groupby("uncertainty_bucket", observed=True):
        ics = []
        for _, day in group.groupby("signal_time", sort=True):
            if len(day) >= 5 and day["diff_score_10d"].nunique() > 1 and day["y_10d"].nunique() > 1:
                ics.append(day["diff_score_10d"].corr(day["y_10d"], method="spearman"))
        out.append(
            {
                "score": "diff_score_10d",
                "metric": f"uncertainty_{bucket}_rankic_mean",
                "value": safe_float(np.nanmean(ics)) if ics else None,
                "rows": int(len(group)),
                "days": int(group["signal_time"].nunique()),
            }
        )
    return out


def add_score_variants(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    horizons = {"5d": 5, "10d": 10, "20d": 20, "30d": 30}
    for suffix in horizons:
        mean = out[f"diff_mean_{suffix}"]
        std = out[f"diff_std_{suffix}"].clip(lower=1e-6)
        out[f"score_mean_{suffix}"] = mean
        out[f"score_mean_over_std_{suffix}"] = mean / std
        out[f"score_mean_over_sqrtstd_{suffix}"] = mean / np.sqrt(std)
        out[f"score_prob_pos_{suffix}"] = out[f"diff_prob_pos_{suffix}"]
        out[f"score_q10_{suffix}"] = out[f"diff_q10_{suffix}"]
        out[f"score_q90_{suffix}"] = out[f"diff_q90_{suffix}"]
        out[f"score_qmid_{suffix}"] = (out[f"diff_q10_{suffix}"] + out[f"diff_q90_{suffix}"]) / 2.0
        for lam in (0.5, 1.0):
            out[f"score_mean_minus_{lam:g}std_{suffix}"] = mean - lam * std
        out[f"score_mean_plus_q10_{suffix}"] = mean + out[f"diff_q10_{suffix}"]
    return out


def write_diagnostics(root: Path, frame: pd.DataFrame) -> None:
    diag = root / "runs" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    scored = add_score_variants(frame)
    horizons = {"5d": "y_5d", "10d": "y_10d", "20d": "y_20d", "30d": "y_30d"}
    score_cols = [
        col
        for col in scored.columns
        if col.startswith("score_")
    ] + ["baseline_momentum_20d", "baseline_ridge_10d"]
    horizon_ic_rows: list[dict[str, Any]] = []
    horizon_spread_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for horizon, label in horizons.items():
        for score in score_cols:
            if score not in scored:
                continue
            daily = daily_ic_values(scored, score, label)
            for row in daily:
                daily_rows.append({"horizon": horizon, "score": score, **row})
            rankic = pd.Series([row["spearman"] for row in daily], dtype=float).dropna()
            pearson = pd.Series([row["pearson"] for row in daily], dtype=float).dropna()
            horizon_ic_rows.append(
                {
                    "horizon": horizon,
                    "score": score,
                    "rankic_mean": safe_float(rankic.mean()),
                    "rankic_median": safe_float(rankic.median()),
                    "rankic_ir": safe_float(rankic.mean() / (rankic.std(ddof=1) + 1e-12)) if len(rankic) > 1 else None,
                    "rankic_pos_ratio": safe_float((rankic > 0).mean()) if len(rankic) else None,
                    "pearson_mean": safe_float(pearson.mean()),
                    "days": int(len(rankic)),
                }
            )
            horizon_spread_rows.append(spread_summary(scored, score, label, horizon))
    pd.DataFrame(horizon_ic_rows).to_csv(diag / "horizon_ic.csv", index=False)
    pd.DataFrame(horizon_spread_rows).to_csv(diag / "horizon_spread.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(diag / "daily_ic.csv", index=False)
    pd.DataFrame(score_variant_rows(scored, horizons)).to_csv(diag / "score_variants_ic.csv", index=False)
    pd.DataFrame(bootstrap_rows(scored)).to_csv(diag / "bootstrap_significance.csv", index=False)
    pd.DataFrame(cost_stress_rows(scored)).to_csv(diag / "cost_stress.csv", index=False)
    pd.DataFrame(regime_rows(scored)).to_csv(diag / "regime_metrics.csv", index=False)
    pd.DataFrame(uncertainty_diagnostics(scored)).to_csv(diag / "uncertainty_buckets.csv", index=False)


def daily_ic_values(frame: pd.DataFrame, score: str, label: str) -> list[dict[str, Any]]:
    out = []
    for day, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 10 or values[score].nunique() < 2 or values[label].nunique() < 2:
            continue
        out.append(
            {
                "signal_time": day,
                "spearman": values[score].corr(values[label], method="spearman"),
                "pearson": values[score].corr(values[label], method="pearson"),
                "n": int(len(values)),
            }
        )
    return out


def spread_summary(frame: pd.DataFrame, score: str, label: str, horizon: str | None = None) -> dict[str, Any]:
    spreads = []
    top_values = []
    bottom_values = []
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        bottom = values.head(n)[label].mean()
        top = values.tail(n)[label].mean()
        spreads.append(top - bottom)
        top_values.append(top)
        bottom_values.append(bottom)
    row = {
        "score": score,
        "label": label,
        "spread_mean": safe_float(np.mean(spreads)) if spreads else None,
        "spread_median": safe_float(np.median(spreads)) if spreads else None,
        "spread_pos_ratio": safe_float((np.asarray(spreads) > 0).mean()) if spreads else None,
        "top_mean": safe_float(np.mean(top_values)) if top_values else None,
        "bottom_mean": safe_float(np.mean(bottom_values)) if bottom_values else None,
        "days": int(len(spreads)),
    }
    if horizon is not None:
        row["horizon"] = horizon
    return row


def score_variant_rows(frame: pd.DataFrame, horizons: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for horizon, label in horizons.items():
        score_cols = [col for col in frame.columns if col.startswith("score_") and col.endswith(f"_{horizon}")]
        for score in score_cols:
            daily = daily_ic_values(frame, score, label)
            rankic = pd.Series([row["spearman"] for row in daily], dtype=float).dropna()
            rows.append(
                {
                    "horizon": horizon,
                    "label": label,
                    "score": score,
                    "rankic_mean": safe_float(rankic.mean()),
                    "rankic_median": safe_float(rankic.median()),
                    "rankic_ir": safe_float(rankic.mean() / (rankic.std(ddof=1) + 1e-12)) if len(rankic) > 1 else None,
                    "rankic_pos_ratio": safe_float((rankic > 0).mean()) if len(rankic) else None,
                    "days": int(len(rankic)),
                }
            )
    return rows


def bootstrap_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260716)
    rows = []
    tests = [
        ("5d", "y_5d", "score_mean_over_std_5d"),
        ("5d", "y_5d", "score_mean_minus_1std_5d"),
        ("10d", "y_10d", "score_mean_over_std_10d"),
        ("10d", "y_10d", "score_mean_10d"),
        ("10d", "y_10d", "score_prob_pos_10d"),
        ("10d", "y_10d", "baseline_momentum_20d"),
        ("10d", "y_10d", "baseline_ridge_10d"),
    ]
    for horizon, label, score in tests:
        if score not in frame:
            continue
        ics = np.asarray([row["spearman"] for row in daily_ic_values(frame, score, label)], dtype=float)
        spreads = daily_spreads(frame, score, label)
        if len(ics) == 0 or len(spreads) == 0:
            continue
        idx = rng.integers(0, len(ics), size=(2000, len(ics)))
        boot_ic = ics[idx].mean(axis=1)
        boot_spread = spreads[idx].mean(axis=1)
        flips = rng.choice([-1, 1], size=(2000, len(ics)))
        rows.append(
            {
                "horizon": horizon,
                "score": score,
                "days": int(len(ics)),
                "ic_mean": safe_float(ics.mean()),
                "ic_ci05": safe_float(np.quantile(boot_ic, 0.05)),
                "ic_ci95": safe_float(np.quantile(boot_ic, 0.95)),
                "ic_signflip_p": safe_float((np.abs((flips * ics).mean(axis=1)) >= abs(ics.mean())).mean()),
                "spread_mean": safe_float(spreads.mean()),
                "spread_ci05": safe_float(np.quantile(boot_spread, 0.05)),
                "spread_ci95": safe_float(np.quantile(boot_spread, 0.95)),
                "spread_pos_ratio": safe_float((spreads > 0).mean()),
            }
        )
    return rows


def daily_spreads(frame: pd.DataFrame, score: str, label: str) -> np.ndarray:
    spreads = []
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        spreads.append(values.tail(n)[label].mean() - values.head(n)[label].mean())
    return np.asarray(spreads, dtype=float)


def cost_stress_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    tests = [
        ("5d", "relative", "y_5d", "score_mean_over_std_5d"),
        ("5d", "relative", "y_5d", "score_mean_minus_1std_5d"),
        ("5d", "absolute", "y_abs_5d", "score_mean_over_std_5d"),
        ("5d", "absolute", "y_abs_5d", "score_mean_minus_1std_5d"),
        ("10d", "relative", "y_10d", "score_mean_over_std_10d"),
        ("10d", "relative", "y_10d", "score_mean_10d"),
        ("10d", "relative", "y_10d", "baseline_momentum_20d"),
        ("10d", "relative", "y_10d", "baseline_ridge_10d"),
        ("10d", "absolute", "y_abs_10d", "score_mean_over_std_10d"),
        ("10d", "absolute", "y_abs_10d", "score_mean_10d"),
        ("10d", "absolute", "y_abs_10d", "baseline_momentum_20d"),
        ("10d", "absolute", "y_abs_10d", "baseline_ridge_10d"),
    ]
    for horizon, return_type, label, score in tests:
        if score not in frame or label not in frame:
            continue
        spreads = daily_spreads(frame, score, label)
        turnover = average_group_turnover(frame, score)
        top_returns = daily_top_returns(frame, score, label)
        top_turnover = average_top_turnover(frame, score)
        gross = float(np.mean(spreads)) if len(spreads) else float("nan")
        gross_top = float(np.mean(top_returns)) if len(top_returns) else float("nan")
        for cost_bps in (0, 5, 10, 20, 40):
            drag = 2.0 * turnover * cost_bps / 10000.0 if math.isfinite(turnover) else float("nan")
            top_drag = top_turnover * cost_bps / 10000.0 if math.isfinite(top_turnover) else float("nan")
            rows.append(
                {
                    "horizon": horizon,
                    "return_type": return_type,
                    "portfolio_type": "top_bottom_market_neutral",
                    "label": label,
                    "score": score,
                    "cost_bps_per_side": cost_bps,
                    "avg_group_turnover": safe_float(turnover),
                    "gross_spread": safe_float(gross),
                    "cost_drag": safe_float(drag),
                    "net_spread": safe_float(gross - drag),
                    "gross_top_return": None,
                    "net_top_return": None,
                    "days": int(len(spreads)),
                }
            )
            rows.append(
                {
                    "horizon": horizon,
                    "return_type": return_type,
                    "portfolio_type": "top_long_only",
                    "label": label,
                    "score": score,
                    "cost_bps_per_side": cost_bps,
                    "avg_group_turnover": safe_float(top_turnover),
                    "gross_spread": None,
                    "cost_drag": safe_float(top_drag),
                    "net_spread": None,
                    "gross_top_return": safe_float(gross_top),
                    "net_top_return": safe_float(gross_top - top_drag),
                    "days": int(len(top_returns)),
                }
            )
    return rows


def daily_top_returns(frame: pd.DataFrame, score: str, label: str) -> np.ndarray:
    returns = []
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[[score, label]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        returns.append(values.tail(n)[label].mean())
    return np.asarray(returns, dtype=float)


def average_group_turnover(frame: pd.DataFrame, score: str) -> float:
    turnovers = []
    prev_top: set[str] | None = None
    prev_bottom: set[str] | None = None
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[["instrument", score]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        bottom = set(values.head(n)["instrument"].astype(str))
        top = set(values.tail(n)["instrument"].astype(str))
        if prev_top is not None and prev_bottom is not None:
            top_turn = 1.0 - len(top & prev_top) / max(1, len(top))
            bottom_turn = 1.0 - len(bottom & prev_bottom) / max(1, len(bottom))
            turnovers.append((top_turn + bottom_turn) / 2.0)
        prev_top = top
        prev_bottom = bottom
    return float(np.mean(turnovers)) if turnovers else float("nan")


def average_top_turnover(frame: pd.DataFrame, score: str) -> float:
    turnovers = []
    prev_top: set[str] | None = None
    for _, group in frame.groupby("signal_time", sort=True):
        values = group[["instrument", score]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(score)
        if len(values) < 10:
            continue
        n = max(1, int(len(values) * 0.2))
        top = set(values.tail(n)["instrument"].astype(str))
        if prev_top is not None:
            turnovers.append(1.0 - len(top & prev_top) / max(1, len(top)))
        prev_top = top
    return float(np.mean(turnovers)) if turnovers else float("nan")


def regime_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    daily = frame.groupby("signal_time").agg(y10_mean=("y_10d", "mean")).reset_index()
    if daily["y10_mean"].nunique() < 3:
        return []
    daily["regime"] = pd.qcut(daily["y10_mean"], 3, labels=["weak_rel", "mid_rel", "strong_rel"], duplicates="drop")
    rows = []
    for regime, days in daily.groupby("regime", observed=True):
        sub = frame[frame["signal_time"].isin(days["signal_time"])]
        for score in ["score_mean_over_std_10d", "score_mean_10d", "baseline_momentum_20d", "baseline_ridge_10d"]:
            if score not in sub:
                continue
            ics = pd.Series([row["spearman"] for row in daily_ic_values(sub, score, "y_10d")], dtype=float).dropna()
            rows.append(
                {
                    "regime": str(regime),
                    "score": score,
                    "rankic_mean": safe_float(ics.mean()),
                    "days": int(len(ics)),
                    "day_y10_mean": safe_float(days["y10_mean"].mean()),
                }
            )
    return rows


def uncertainty_diagnostics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    values = frame.dropna(subset=["diff_std_10d", "score_mean_over_std_10d", "y_10d"]).copy()
    if values.empty:
        return rows
    values["uncertainty_bucket"] = pd.qcut(values["diff_std_10d"], 3, labels=["low", "mid", "high"], duplicates="drop")
    for bucket, group in values.groupby("uncertainty_bucket", observed=True):
        ics = pd.Series([row["spearman"] for row in daily_ic_values(group, "score_mean_over_std_10d", "y_10d")], dtype=float)
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "days": int(group["signal_time"].nunique()),
                "std_mean": safe_float(group["diff_std_10d"].mean()),
                "abs_y10_mean": safe_float(group["y_10d"].abs().mean()),
                "rankic_mean": safe_float(ics.mean()),
            }
        )
    return rows


def parse_inference_steps_sweep(cfg: Config) -> list[int]:
    if not cfg.inference_steps_sweep.strip():
        return []
    out = []
    for part in cfg.inference_steps_sweep.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0 and value not in out:
            out.append(value)
    return out


def write_inference_step_sweep(
    root: Path,
    ae: Any,
    diffusion: Any,
    schedule: Any,
    stock_test: np.ndarray,
    market_test: np.ndarray,
    base_frame: pd.DataFrame,
    cfg: Config,
    device: Any,
) -> None:
    steps = parse_inference_steps_sweep(cfg)
    if not steps:
        return
    rows = []
    for step_count in steps:
        pred = sample_predictions(
            ae,
            diffusion,
            schedule,
            stock_test,
            market_test,
            cfg,
            device,
            inference_steps_override=step_count,
        )
        frame = base_frame[[
            "signal_time",
            "instrument",
            "y_5d",
            "y_10d",
            "y_abs_5d",
            "y_abs_10d",
            "y_baseline_5d",
            "y_baseline_10d",
            "baseline_momentum_20d",
            "baseline_ridge_10d",
        ]].copy()
        frame["diff_mean_5d"] = pred["mean"][:, 4]
        frame["diff_mean_10d"] = pred["mean"][:, 9]
        frame["diff_std_5d"] = pred["std"][:, 4]
        frame["diff_std_10d"] = pred["std"][:, 9]
        frame["score_mean_over_std_5d"] = frame["diff_mean_5d"] / (frame["diff_std_5d"].clip(lower=1e-6))
        frame["score_mean_over_std_10d"] = frame["diff_mean_10d"] / (frame["diff_std_10d"].clip(lower=1e-6))
        for horizon, label, score in (
            ("5d", "y_5d", "score_mean_over_std_5d"),
            ("10d", "y_10d", "score_mean_over_std_10d"),
            ("5d", "y_abs_5d", "score_mean_over_std_5d"),
            ("10d", "y_abs_10d", "score_mean_over_std_10d"),
        ):
            daily = pd.Series([row["spearman"] for row in daily_ic_values(frame, score, label)], dtype=float).dropna()
            spread = daily_spreads(frame, score, label)
            rows.append(
                {
                    "inference_steps": int(step_count),
                    "horizon": horizon,
                    "label": label,
                    "score": score,
                    "rankic_mean": safe_float(daily.mean()),
                    "rankic_pos_ratio": safe_float((daily > 0).mean()) if len(daily) else None,
                    "spread_mean": safe_float(spread.mean()) if len(spread) else None,
                    "spread_pos_ratio": safe_float((spread > 0).mean()) if len(spread) else None,
                    "days": int(len(daily)),
                }
            )
    diag = root / "runs" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(diag / "inference_steps_sweep.csv", index=False)


def write_inference_step_sweep_loader(
    root: Path,
    ae: Any,
    diffusion: Any,
    schedule: Any,
    test_ds: Any,
    base_frame: pd.DataFrame,
    cfg: Config,
    device: Any,
) -> None:
    steps = parse_inference_steps_sweep(cfg)
    if not steps:
        return
    rows = []
    for step_count in steps:
        loader = sharded_loader(test_ds, cfg, train=False)
        pred = sample_predictions_loader(
            ae,
            diffusion,
            schedule,
            loader,
            cfg,
            device,
            inference_steps_override=step_count,
        )
        frame = base_frame[[
            "signal_time",
            "instrument",
            "y_5d",
            "y_10d",
            "y_abs_5d",
            "y_abs_10d",
            "y_baseline_5d",
            "y_baseline_10d",
            "baseline_momentum_20d",
            "baseline_ridge_10d",
        ]].copy()
        frame["diff_mean_5d"] = pred["mean"][:, 4]
        frame["diff_mean_10d"] = pred["mean"][:, 9]
        frame["diff_std_5d"] = pred["std"][:, 4]
        frame["diff_std_10d"] = pred["std"][:, 9]
        frame["score_mean_over_std_5d"] = frame["diff_mean_5d"] / frame["diff_std_5d"].clip(lower=1e-6)
        frame["score_mean_over_std_10d"] = frame["diff_mean_10d"] / frame["diff_std_10d"].clip(lower=1e-6)
        for horizon, label, score in (
            ("5d", "y_5d", "score_mean_over_std_5d"),
            ("10d", "y_10d", "score_mean_over_std_10d"),
            ("5d", "y_abs_5d", "score_mean_over_std_5d"),
            ("10d", "y_abs_10d", "score_mean_over_std_10d"),
        ):
            daily = pd.Series([row["spearman"] for row in daily_ic_values(frame, score, label)], dtype=float).dropna()
            spread = daily_spreads(frame, score, label)
            rows.append(
                {
                    "inference_steps": int(step_count),
                    "horizon": horizon,
                    "label": label,
                    "score": score,
                    "rankic_mean": safe_float(daily.mean()),
                    "rankic_pos_ratio": safe_float((daily > 0).mean()) if len(daily) else None,
                    "spread_mean": safe_float(spread.mean()) if len(spread) else None,
                    "spread_pos_ratio": safe_float((spread > 0).mean()) if len(spread) else None,
                    "days": int(len(daily)),
                }
            )
    diag = root / "runs" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(diag / "inference_steps_sweep.csv", index=False)


def write_ae_reconstruction_metrics(
    root: Path,
    ae: Any,
    stock: np.ndarray,
    market: np.ndarray,
    data: Any,
    cfg: Config,
    device: Any,
) -> None:
    rows = []
    feature_rows = []
    split_masks = {
        "train": data["split_train"].astype(bool),
        "val": data["split_val"].astype(bool),
        "test": data["split_test"].astype(bool),
    }
    stock_names = [str(x) for x in data["stock_feature_names"]]
    market_names = [str(x) for x in data["market_feature_names"]]
    for split_name, mask in split_masks.items():
        if not mask.any():
            continue
        for masked in (False, True):
            summary, features = ae_reconstruction_summary(
                ae=ae,
                stock=stock[mask],
                market=market[mask],
                batch_size=cfg.batch_size,
                device=device,
                masked=masked,
                stock_names=stock_names,
                market_names=market_names,
            )
            rows.append(
                {
                    "split": split_name,
                    "masked_input": masked,
                    "samples": int(mask.sum()),
                    "compression_ratio": safe_float((cfg.latent_tokens * cfg.latent_dim) / (stock.shape[1] * (stock.shape[2] + market.shape[2]))),
                    **summary,
                }
            )
            for feature in features:
                feature_rows.append({"split": split_name, "masked_input": masked, **feature})
    metrics = pd.DataFrame(rows)
    per_feature = pd.DataFrame(feature_rows)
    metrics.to_csv(root / "runs" / "ae_reconstruction_metrics.csv", index=False)
    per_feature.to_csv(root / "runs" / "ae_reconstruction_by_feature.csv", index=False)
    summary = {
        "compression_ratio": safe_float((cfg.latent_tokens * cfg.latent_dim) / (stock.shape[1] * (stock.shape[2] + market.shape[2]))),
        "rows": rows,
    }
    (root / "runs" / "ae_reconstruction_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_ae_reconstruction_metrics_sharded(root: Path, ae: Any, manifest: dict[str, Any], cfg: Config, device: Any) -> None:
    rows = []
    feature_rows = []
    stock_names = [str(x) for x in manifest["stock_feature_names"]]
    market_names = [str(x) for x in manifest["market_feature_names"]]
    for split_name in ("train", "val", "test"):
        if not manifest.get("splits", {}).get(split_name):
            continue
        dataset = ShardedMemmapDataset(root, manifest, split_name, include_y=False)
        for masked in (False, True):
            loader = sharded_loader(dataset, cfg, train=False)
            summary, features = ae_reconstruction_summary_loader(
                ae=ae,
                loader=loader,
                device=device,
                masked=masked,
                stock_names=stock_names,
                market_names=market_names,
            )
            rows.append(
                {
                    "split": split_name,
                    "masked_input": masked,
                    "samples": int(len(dataset)),
                    "compression_ratio": safe_float(
                        (cfg.latent_tokens * cfg.latent_dim)
                        / (int(manifest["lookback"]) * (int(manifest["stock_dim"]) + int(manifest["market_dim"])))
                    ),
                    **summary,
                }
            )
            for feature in features:
                feature_rows.append({"split": split_name, "masked_input": masked, **feature})
    pd.DataFrame(rows).to_csv(root / "runs" / "ae_reconstruction_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(root / "runs" / "ae_reconstruction_by_feature.csv", index=False)
    summary = {
        "compression_ratio": safe_float(
            (cfg.latent_tokens * cfg.latent_dim) / (int(manifest["lookback"]) * (int(manifest["stock_dim"]) + int(manifest["market_dim"])))
        ),
        "rows": rows,
    }
    (root / "runs" / "ae_reconstruction_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ae_reconstruction_summary(
    *,
    ae: Any,
    stock: np.ndarray,
    market: np.ndarray,
    batch_size: int,
    device: Any,
    masked: bool,
    stock_names: list[str],
    market_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.from_numpy(stock), torch.from_numpy(market)), batch_size=batch_size)
    stats = {
        "stock": init_recon_stats(len(stock_names)),
        "market": init_recon_stats(len(market_names)),
    }
    ae.eval()
    with torch.no_grad():
        for xb_stock, xb_market in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            in_stock = mask_tensor(xb_stock, 0.15) if masked else xb_stock
            in_market = mask_tensor(xb_market, 0.15) if masked else xb_market
            recon_stock, recon_market, _ = ae(in_stock, in_market)
            update_recon_stats(stats["stock"], xb_stock, recon_stock)
            update_recon_stats(stats["market"], xb_market, recon_market)
    stock_summary, stock_features = finalize_recon_stats(stats["stock"], "stock", stock_names)
    market_summary, market_features = finalize_recon_stats(stats["market"], "market", market_names)
    summary = {
        "stock_mse": stock_summary["mse"],
        "stock_mae": stock_summary["mae"],
        "stock_nmae": stock_summary["nmae"],
        "stock_corr_mean": stock_summary["corr_mean"],
        "market_mse": market_summary["mse"],
        "market_mae": market_summary["mae"],
        "market_nmae": market_summary["nmae"],
        "market_corr_mean": market_summary["corr_mean"],
        "combined_mse": safe_float(0.5 * (stock_summary["mse"] + market_summary["mse"])),
        "combined_mae": safe_float(0.5 * (stock_summary["mae"] + market_summary["mae"])),
        "combined_nmae": safe_float(
            (stock_summary["sum_abs_err"] + market_summary["sum_abs_err"])
            / max(stock_summary["sum_abs_x"] + market_summary["sum_abs_x"], 1e-6)
        ),
    }
    return summary, stock_features + market_features


def ae_reconstruction_summary_loader(
    *,
    ae: Any,
    loader: Any,
    device: Any,
    masked: bool,
    stock_names: list[str],
    market_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    stats = {
        "stock": init_recon_stats(len(stock_names)),
        "market": init_recon_stats(len(market_names)),
    }
    ae.eval()
    with torch.no_grad():
        for xb_stock, xb_market in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            in_stock = mask_tensor(xb_stock, 0.15) if masked else xb_stock
            in_market = mask_tensor(xb_market, 0.15) if masked else xb_market
            recon_stock, recon_market, _ = ae(in_stock, in_market)
            update_recon_stats(stats["stock"], xb_stock, recon_stock)
            update_recon_stats(stats["market"], xb_market, recon_market)
    stock_summary, stock_features = finalize_recon_stats(stats["stock"], "stock", stock_names)
    market_summary, market_features = finalize_recon_stats(stats["market"], "market", market_names)
    summary = {
        "stock_mse": stock_summary["mse"],
        "stock_mae": stock_summary["mae"],
        "stock_nmae": stock_summary["nmae"],
        "stock_corr_mean": stock_summary["corr_mean"],
        "market_mse": market_summary["mse"],
        "market_mae": market_summary["mae"],
        "market_nmae": market_summary["nmae"],
        "market_corr_mean": market_summary["corr_mean"],
        "combined_mse": safe_float(0.5 * (stock_summary["mse"] + market_summary["mse"])),
        "combined_mae": safe_float(0.5 * (stock_summary["mae"] + market_summary["mae"])),
        "combined_nmae": safe_float(
            (stock_summary["sum_abs_err"] + market_summary["sum_abs_err"])
            / max(stock_summary["sum_abs_x"] + market_summary["sum_abs_x"], 1e-6)
        ),
    }
    return summary, stock_features + market_features


def init_recon_stats(feature_dim: int) -> dict[str, np.ndarray | float]:
    return {
        "n": 0.0,
        "sum_abs": np.zeros(feature_dim, dtype=np.float64),
        "sum_abs_x": np.zeros(feature_dim, dtype=np.float64),
        "sum_sq_err": np.zeros(feature_dim, dtype=np.float64),
        "sum_x": np.zeros(feature_dim, dtype=np.float64),
        "sum_y": np.zeros(feature_dim, dtype=np.float64),
        "sum_x2": np.zeros(feature_dim, dtype=np.float64),
        "sum_y2": np.zeros(feature_dim, dtype=np.float64),
        "sum_xy": np.zeros(feature_dim, dtype=np.float64),
    }


def update_recon_stats(stats: dict[str, Any], target: Any, recon: Any) -> None:
    x = target.detach().cpu().numpy().reshape(-1, target.shape[-1]).astype(np.float64)
    y = recon.detach().cpu().numpy().reshape(-1, recon.shape[-1]).astype(np.float64)
    err = y - x
    stats["n"] += float(x.shape[0])
    stats["sum_abs"] += np.abs(err).sum(axis=0)
    stats["sum_abs_x"] += np.abs(x).sum(axis=0)
    stats["sum_sq_err"] += np.square(err).sum(axis=0)
    stats["sum_x"] += x.sum(axis=0)
    stats["sum_y"] += y.sum(axis=0)
    stats["sum_x2"] += np.square(x).sum(axis=0)
    stats["sum_y2"] += np.square(y).sum(axis=0)
    stats["sum_xy"] += (x * y).sum(axis=0)


def finalize_recon_stats(stats: dict[str, Any], group: str, feature_names: list[str]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    n = max(float(stats["n"]), 1.0)
    mae = stats["sum_abs"] / n
    mean_abs_x = stats["sum_abs_x"] / n
    nmae = mae / np.maximum(mean_abs_x, 1e-6)
    mse = stats["sum_sq_err"] / n
    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n
    cov = stats["sum_xy"] / n - mean_x * mean_y
    var_x = stats["sum_x2"] / n - np.square(mean_x)
    var_y = stats["sum_y2"] / n - np.square(mean_y)
    corr = cov / (np.sqrt(np.maximum(var_x, 0.0)) * np.sqrt(np.maximum(var_y, 0.0)) + 1e-12)
    corr = np.where(np.isfinite(corr), corr, np.nan)
    rows = []
    for idx, name in enumerate(feature_names):
        rows.append(
            {
                "group": group,
                "feature": name,
                "mae": safe_float(mae[idx]),
                "nmae": safe_float(nmae[idx]),
                "mse": safe_float(mse[idx]),
                "corr": safe_float(corr[idx]),
            }
        )
    summary = {
        "mae": safe_float(np.nanmean(mae)),
        "nmae": safe_float(np.nanmean(nmae)),
        "mse": safe_float(np.nanmean(mse)),
        "corr_mean": safe_float(np.nanmean(corr)),
        "sum_abs_err": float(np.nansum(stats["sum_abs"])),
        "sum_abs_x": float(np.nansum(stats["sum_abs_x"])),
    }
    return summary, rows


def metric_lookup(rows: list[dict[str, Any]], score: str, metric: str) -> Any:
    for row in rows:
        if row.get("score") == score and row.get("metric") == metric:
            return row.get("value")
    return None


def check_compression(cfg: Config, lookback: int, stock_dim: int, market_dim: int) -> None:
    raw_dim = lookback * (stock_dim + market_dim)
    latent_dim = cfg.latent_tokens * cfg.latent_dim
    if latent_dim >= raw_dim:
        raise ValueError(f"Latent is not compressed: latent={latent_dim} raw={raw_dim}")
    print(f"[train] compression latent={latent_dim} raw={raw_dim} ratio={latent_dim / raw_dim:.4f}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def shift(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float32)
    if n > 0:
        out[n:] = values[:-n]
    elif n < 0:
        out[:n] = values[-n:]
    else:
        out[:] = values
    return out


def log_change(values: np.ndarray, prev: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log1p(np.maximum(values, 0.0)) - np.log1p(np.maximum(prev, 0.0))
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def log_change_1d(values: np.ndarray) -> np.ndarray:
    return log_change(values.astype(np.float32), shift(values.astype(np.float32), 1))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    frame = pd.DataFrame(values)
    return frame.rolling(window, min_periods=max(2, window // 2)).mean().to_numpy(dtype=np.float32)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    frame = pd.DataFrame(values)
    return frame.rolling(window, min_periods=max(2, window // 2)).std().to_numpy(dtype=np.float32)


def nanmean(values: np.ndarray) -> np.ndarray:
    return np.nanmean(values, axis=1).astype(np.float32)


def nanmedian(values: np.ndarray) -> np.ndarray:
    return np.nanmedian(values, axis=1).astype(np.float32)


def nanstd(values: np.ndarray) -> np.ndarray:
    return np.nanstd(values, axis=1).astype(np.float32)


def nanpercentile_2d(values: np.ndarray, q: float) -> np.ndarray:
    return np.nanpercentile(values, q, axis=1).astype(np.float32)


def top_amount_share(amount: np.ndarray, n: int) -> np.ndarray:
    safe = np.where(np.isfinite(amount) & (amount > 0), amount, 0.0)
    total = safe.sum(axis=1)
    if amount.shape[1] <= n:
        top = safe.sum(axis=1)
    else:
        top = np.partition(safe, -n, axis=1)[:, -n:].sum(axis=1)
    return np.where(total > 0, top / total, np.nan).astype(np.float32)


def top_mean(values: np.ndarray, n: int) -> np.ndarray:
    out = []
    for row in values:
        row = row[np.isfinite(row)]
        if len(row) == 0:
            out.append(np.nan)
            continue
        k = min(n, len(row))
        out.append(float(np.partition(row, -k)[-k:].mean()))
    return np.asarray(out, dtype=np.float32)


def bottom_mean(values: np.ndarray, n: int) -> np.ndarray:
    out = []
    for row in values:
        row = row[np.isfinite(row)]
        if len(row) == 0:
            out.append(np.nan)
            continue
        k = min(n, len(row))
        out.append(float(np.partition(row, k - 1)[:k].mean()))
    return np.asarray(out, dtype=np.float32)


def finite_enough(values: np.ndarray, ratio: float) -> bool:
    return float(np.isfinite(values).mean()) >= ratio


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def mask_tensor(x: Any, p: float) -> Any:
    import torch

    mask = torch.rand_like(x) < p
    return x.masked_fill(mask, 0.0)


def default_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def encode_condition(ae: Any, stock: Any, market: Any, cfg: Config) -> Any:
    z = ae.encode(stock, market)
    if cfg.denoiser_type == "cross_dit":
        return z
    return z.flatten(1)


def make_denoiser(cfg: Config, target_dim: int) -> Any:
    if cfg.denoiser_type == "cross_dit":
        return ConditionalCrossAttentionDiTDenoiser(
            target_dim=target_dim,
            cond_dim=cfg.latent_dim,
            cond_tokens=cfg.latent_tokens,
            d_model=cfg.dit_d_model,
            n_layers=cfg.dit_layers,
            n_heads=cfg.dit_heads,
            mlp_ratio=cfg.dit_mlp_ratio,
            dropout=cfg.dit_dropout,
            diffusion_steps=cfg.diffusion_steps,
        )
    return ConditionalDiffusionDenoiser(
        target_dim=target_dim,
        cond_dim=cfg.latent_tokens * cfg.latent_dim,
        hidden_dim=generative_hidden_dim(cfg),
        diffusion_steps=cfg.diffusion_steps,
    )


def eval_ae(model: Any, tensors: tuple[Any, Any]) -> float:
    import torch

    if tensors[0].numel() == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        recon_stock, recon_market, _ = model(tensors[0], tensors[1])
        loss = torch.nn.functional.smooth_l1_loss(recon_stock, tensors[0]) + 0.7 * torch.nn.functional.smooth_l1_loss(
            recon_market, tensors[1]
        )
    return float(loss.detach().cpu())


def eval_ae_loader(model: Any, loader: Any, device: Any) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for xb_stock, xb_market in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            recon_stock, recon_market, _ = model(xb_stock, xb_market)
            loss = torch.nn.functional.smooth_l1_loss(recon_stock, xb_stock) + 0.7 * torch.nn.functional.smooth_l1_loss(
                recon_market, xb_market
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def eval_diffusion(
    ae: Any,
    diffusion: Any,
    schedule: Any,
    stock: np.ndarray,
    market: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    device: Any,
) -> float:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if len(y) == 0:
        return float("nan")
    loader = DataLoader(TensorDataset(torch.from_numpy(stock), torch.from_numpy(market), torch.from_numpy(y)), batch_size=cfg.batch_size)
    losses = []
    ae.eval()
    diffusion.eval()
    with torch.no_grad():
        for xb_stock, xb_market, yb in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            step = torch.randint(0, schedule.steps, (yb.shape[0],), device=device)
            noise = torch.randn_like(yb)
            y_noisy = schedule.q_sample(yb, step, noise)
            noise_hat = diffusion(y_noisy, step, cond)
            losses.append(float(torch.nn.functional.smooth_l1_loss(noise_hat, noise).detach().cpu()))
    return float(np.mean(losses))


def eval_diffusion_loader(
    ae: Any,
    diffusion: Any,
    schedule: Any,
    loader: Any,
    cfg: Config,
    device: Any,
) -> float:
    import torch

    losses = []
    ae.eval()
    diffusion.eval()
    with torch.no_grad():
        for xb_stock, xb_market, yb in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            step = torch.randint(0, schedule.steps, (yb.shape[0],), device=device)
            noise = torch.randn_like(yb)
            y_noisy = schedule.q_sample(yb, step, noise)
            noise_hat = diffusion(y_noisy, step, cond)
            losses.append(float(torch.nn.functional.smooth_l1_loss(noise_hat, noise).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def generative_hidden_dim(cfg: Config) -> int:
    return int(cfg.generative_hidden_dim) if cfg.generative_hidden_dim > 0 else max(128, cfg.d_model * 2)


def effective_inference_steps(cfg: Config, override: int | None = None) -> int:
    if override is not None and override > 0:
        return int(override)
    if cfg.inference_steps > 0:
        return int(cfg.inference_steps)
    return int(cfg.flow_steps if cfg.generative_objective == "flow" else cfg.diffusion_steps)


def flow_matching_loss(model: Any, y_target: Any, cond: Any) -> Any:
    import torch

    x0 = torch.randn_like(y_target)
    t = torch.rand((y_target.shape[0],), device=y_target.device)
    xt = (1.0 - t.view(-1, 1)) * x0 + t.view(-1, 1) * y_target
    velocity = y_target - x0
    velocity_hat = model(xt, t, cond)
    return torch.nn.functional.smooth_l1_loss(velocity_hat, velocity)


def eval_flow(
    ae: Any,
    model: Any,
    stock: np.ndarray,
    market: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    device: Any,
) -> float:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if len(y) == 0:
        return float("nan")
    loader = DataLoader(TensorDataset(torch.from_numpy(stock), torch.from_numpy(market), torch.from_numpy(y)), batch_size=cfg.batch_size)
    losses = []
    ae.eval()
    model.eval()
    with torch.no_grad():
        for xb_stock, xb_market, yb in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            losses.append(float(flow_matching_loss(model, yb, cond).detach().cpu()))
    return float(np.mean(losses))


def eval_flow_loader(ae: Any, model: Any, loader: Any, cfg: Config, device: Any) -> float:
    import torch

    losses = []
    ae.eval()
    model.eval()
    with torch.no_grad():
        for xb_stock, xb_market, yb in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            yb = yb.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            losses.append(float(flow_matching_loss(model, yb, cond).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def sample_predictions(
    ae: Any,
    diffusion: Any,
    schedule: Any,
    stock: np.ndarray,
    market: np.ndarray,
    cfg: Config,
    device: Any,
    inference_steps_override: int | None = None,
) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.from_numpy(stock), torch.from_numpy(market)), batch_size=cfg.batch_size)
    all_samples = []
    ae.eval()
    diffusion.eval()
    with torch.no_grad():
        for xb_stock, xb_market in loader:
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            batch_samples = []
            for _ in range(cfg.k_samples):
                if cfg.generative_objective == "flow":
                    y = sample_flow(
                        diffusion,
                        cond,
                        xb_stock.shape[0],
                        cfg.horizon,
                        effective_inference_steps(cfg, inference_steps_override),
                        device,
                    )
                else:
                    y = sample_ddpm(
                        diffusion,
                        schedule,
                        cond,
                        xb_stock.shape[0],
                        cfg.horizon,
                        effective_inference_steps(cfg, inference_steps_override),
                        device,
                    )
                batch_samples.append(y.detach().cpu().numpy())
            all_samples.append(np.stack(batch_samples, axis=1))
    samples = np.concatenate(all_samples, axis=0)
    return {
        "mean": samples.mean(axis=1),
        "std": samples.std(axis=1),
        "prob_pos": (samples > 0.0).mean(axis=1),
        "q10": np.quantile(samples, 0.1, axis=1),
        "q90": np.quantile(samples, 0.9, axis=1),
    }


def sample_predictions_loader(
    ae: Any,
    diffusion: Any,
    schedule: Any,
    loader: Any,
    cfg: Config,
    device: Any,
    inference_steps_override: int | None = None,
) -> dict[str, np.ndarray]:
    import torch

    all_samples = []
    ae.eval()
    diffusion.eval()
    with torch.no_grad():
        for batch in loader:
            xb_stock, xb_market = batch[0], batch[1]
            xb_stock = xb_stock.to(device)
            xb_market = xb_market.to(device)
            cond = encode_condition(ae, xb_stock, xb_market, cfg)
            batch_samples = []
            for _ in range(cfg.k_samples):
                if cfg.generative_objective == "flow":
                    y = sample_flow(
                        diffusion,
                        cond,
                        xb_stock.shape[0],
                        cfg.horizon,
                        effective_inference_steps(cfg, inference_steps_override),
                        device,
                    )
                else:
                    y = sample_ddpm(
                        diffusion,
                        schedule,
                        cond,
                        xb_stock.shape[0],
                        cfg.horizon,
                        effective_inference_steps(cfg, inference_steps_override),
                        device,
                    )
                batch_samples.append(y.detach().cpu().numpy())
            all_samples.append(np.stack(batch_samples, axis=1))
    samples = np.concatenate(all_samples, axis=0)
    return {
        "mean": samples.mean(axis=1),
        "std": samples.std(axis=1),
        "prob_pos": (samples > 0.0).mean(axis=1),
        "q10": np.quantile(samples, 0.1, axis=1),
        "q90": np.quantile(samples, 0.9, axis=1),
    }


def sample_flow(model: Any, cond: Any, batch_size: int, horizon: int, steps: int, device: Any) -> Any:
    import torch

    steps = max(1, int(steps))
    y = torch.randn((batch_size, horizon), device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_value = (i + 0.5) / steps
        t = torch.full((batch_size,), t_value, device=device)
        y = y + dt * model(y, t, cond)
    return y


def sample_ddpm(model: Any, schedule: Any, cond: Any, batch_size: int, horizon: int, inference_steps: int, device: Any) -> Any:
    import torch

    inference_steps = max(1, min(int(inference_steps), schedule.steps))
    indices = np.linspace(schedule.steps - 1, 0, inference_steps).round().astype(int)
    y = torch.randn((batch_size, horizon), device=device)
    for i, step_value in enumerate(indices):
        step = torch.full((batch_size,), int(step_value), device=device, dtype=torch.long)
        eps = model(y, step, cond)
        alpha_bar = schedule.alpha_bar[step].view(-1, 1)
        x0 = (y - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        if i == len(indices) - 1:
            y = x0
        else:
            next_step_value = int(indices[i + 1])
            next_alpha_bar = schedule.alpha_bar[torch.full((batch_size,), next_step_value, device=device, dtype=torch.long)].view(-1, 1)
            y = torch.sqrt(next_alpha_bar) * x0 + torch.sqrt(1.0 - next_alpha_bar) * eps
    return y


class DiffusionSchedule:
    def __init__(self, steps: int, device: Any):
        import torch

        self.steps = int(steps)
        self.betas = torch.linspace(1e-4, 0.02, self.steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, y0: Any, step: Any, noise: Any) -> Any:
        import torch

        a = self.alpha_bar[step].view(-1, 1)
        return torch.sqrt(a) * y0 + torch.sqrt(1.0 - a) * noise

    def p_sample(self, y: Any, step: Any, eps: Any) -> Any:
        import torch

        beta = self.betas[step].view(-1, 1)
        alpha = self.alphas[step].view(-1, 1)
        alpha_bar = self.alpha_bar[step].view(-1, 1)
        mean = (1.0 / torch.sqrt(alpha)) * (y - beta / torch.sqrt(1.0 - alpha_bar) * eps)
        noise = torch.randn_like(y)
        nonzero = (step > 0).float().view(-1, 1)
        return mean + nonzero * torch.sqrt(beta) * noise


class TransformerAutoEncoder(__import__("torch").nn.Module):
    def __init__(
        self,
        *,
        stock_dim: int,
        market_dim: int,
        lookback: int,
        d_stock: int,
        d_market: int,
        d_model: int,
        latent_tokens: int,
        latent_dim: int,
        n_heads: int,
        n_layers: int,
    ):
        import torch

        super().__init__()
        self.lookback = lookback
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.stock_proj = torch.nn.Linear(stock_dim, d_stock)
        self.market_proj = torch.nn.Linear(market_dim, d_market)
        self.fusion = torch.nn.Linear(d_stock + d_market, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(1, lookback, d_model))
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.compress = torch.nn.Sequential(
            torch.nn.Linear(lookback * d_model, latent_tokens * latent_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(latent_tokens * latent_dim),
        )
        self.expand = torch.nn.Sequential(
            torch.nn.Linear(latent_tokens * latent_dim, lookback * d_model),
            torch.nn.GELU(),
            torch.nn.LayerNorm(lookback * d_model),
        )
        dec_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = torch.nn.TransformerEncoder(dec_layer, num_layers=1)
        self.stock_out = torch.nn.Linear(d_model, stock_dim)
        self.market_out = torch.nn.Linear(d_model, market_dim)

    def encode(self, stock: Any, market: Any) -> Any:
        x = self.fusion(__import__("torch").cat([self.stock_proj(stock), self.market_proj(market)], dim=-1))
        h = self.encoder(x + self.pos)
        z = self.compress(h.flatten(1)).view(stock.shape[0], self.latent_tokens, self.latent_dim)
        return z

    def forward(self, stock: Any, market: Any) -> tuple[Any, Any, Any]:
        z = self.encode(stock, market)
        h = self.expand(z.flatten(1)).view(stock.shape[0], self.lookback, -1)
        h = self.decoder(h + self.pos)
        return self.stock_out(h), self.market_out(h), z


class ConditionalDiffusionDenoiser(__import__("torch").nn.Module):
    def __init__(self, *, target_dim: int, cond_dim: int, hidden_dim: int, diffusion_steps: int):
        import torch

        super().__init__()
        self.step_emb = torch.nn.Embedding(diffusion_steps, hidden_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(target_dim + cond_dim + hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, y_noisy: Any, step: Any, cond: Any) -> Any:
        import torch

        if step.dtype in (torch.long, torch.int64, torch.int32):
            time_emb = self.step_emb(step.long())
        else:
            time_emb = self.time_mlp(sinusoidal_time_embedding(step.float(), self.step_emb.embedding_dim))
        return self.net(torch.cat([y_noisy, cond, time_emb], dim=-1))


class FluxPathDoubleStreamBlock(__import__("torch").nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        import torch

        super().__init__()
        mlp_hidden = int(round(d_model * mlp_ratio))
        self.target_norm1 = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.target_norm2 = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cond_norm1 = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cond_norm2 = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.target_qkv = torch.nn.Linear(d_model, d_model * 3)
        self.cond_qkv = torch.nn.Linear(d_model, d_model * 3)
        self.target_proj = torch.nn.Linear(d_model, d_model)
        self.cond_proj = torch.nn.Linear(d_model, d_model)
        self.target_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, mlp_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden, d_model),
        )
        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, mlp_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden, d_model),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.target_mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(d_model, d_model * 6))
        self.cond_mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(d_model, d_model * 6))
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, target: Any, cond: Any, vec: Any) -> tuple[Any, Any]:
        import torch

        target_mod1, target_mod2 = split_modulation(self.target_mod(vec))
        cond_mod1, cond_mod2 = split_modulation(self.cond_mod(vec))

        target_q, target_k, target_v = split_qkv_heads(
            self.target_qkv(modulate(self.target_norm1(target), *target_mod1[:2])), self.n_heads
        )
        cond_q, cond_k, cond_v = split_qkv_heads(
            self.cond_qkv(modulate(self.cond_norm1(cond), *cond_mod1[:2])), self.n_heads
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


class FluxPathSingleStreamBlock(__import__("torch").nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        import torch

        super().__init__()
        mlp_hidden = int(round(d_model * mlp_ratio))
        self.norm = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = torch.nn.Linear(d_model, d_model * 3)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, mlp_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden, d_model),
        )
        self.mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(d_model, d_model * 3))
        self.dropout = torch.nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, x: Any, vec: Any) -> Any:
        import torch

        shift, scale, gate = self.mod(vec).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        q, k, v = split_qkv_heads(self.qkv(h), self.n_heads)
        attn = scaled_dot_product_attention(q, k, v)
        out = self.proj(attn) + self.mlp(h)
        return x + self.dropout(torch.tanh(gate).unsqueeze(1) * out)


class FluxPathFinalLayer(__import__("torch").nn.Module):
    def __init__(self, *, d_model: int):
        import torch

        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(d_model, d_model * 2))
        self.out = torch.nn.Linear(d_model, 1)

    def forward(self, x: Any, vec: Any) -> Any:
        shift, scale = self.mod(vec).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class ConditionalCrossAttentionDiTDenoiser(__import__("torch").nn.Module):
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
        import torch

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"dit_d_model must be divisible by dit_heads: {d_model} vs {n_heads}")
        self.target_dim = target_dim
        self.cond_tokens = cond_tokens
        self.diffusion_steps = diffusion_steps
        self.target_value_proj = torch.nn.Linear(1, d_model)
        self.cond_proj = torch.nn.Linear(cond_dim, d_model)
        self.source_embed = torch.nn.Embedding(2, d_model)
        self.position_embed = torch.nn.Embedding(max(target_dim, cond_tokens), d_model)
        cond_ids = torch.stack([torch.zeros(cond_tokens, dtype=torch.long), torch.arange(cond_tokens, dtype=torch.long)], dim=1)
        target_ids = torch.stack([torch.ones(target_dim, dtype=torch.long), torch.arange(target_dim, dtype=torch.long)], dim=1)
        self.register_buffer("cond_ids", cond_ids, persistent=False)
        self.register_buffer("target_ids", target_ids, persistent=False)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model * 4, d_model),
        )
        double_layers = max(1, int(round(n_layers * 0.75)))
        single_layers = max(1, n_layers - double_layers)
        self.double_blocks = torch.nn.ModuleList(
            [
                FluxPathDoubleStreamBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(double_layers)
            ]
        )
        self.single_blocks = torch.nn.ModuleList(
            [
                FluxPathSingleStreamBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(single_layers)
            ]
        )
        self.final_layer = FluxPathFinalLayer(d_model=d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        import torch

        torch.nn.init.trunc_normal_(self.source_embed.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.position_embed.weight, std=0.02)

    def forward(self, y_noisy: Any, step: Any, cond: Any) -> Any:
        if cond.ndim != 3:
            raise ValueError(f"cross_dit expects token condition [B,T,C], got shape={tuple(cond.shape)}")
        t = step.float()
        if step.dtype in (__import__("torch").long, __import__("torch").int64, __import__("torch").int32):
            t = t / max(float(self.diffusion_steps - 1), 1.0)
        vec = self.time_mlp(sinusoidal_time_embedding(t, self.source_embed.embedding_dim))
        target = self.target_value_proj(y_noisy.unsqueeze(-1)) + self.ids_to_embedding(self.target_ids[: y_noisy.shape[1]])
        cond = self.cond_proj(cond) + self.ids_to_embedding(self.cond_ids[: cond.shape[1]])
        for block in self.double_blocks:
            target, cond = block(target, cond, vec)
        joined = __import__("torch").cat([cond, target], dim=1)
        for block in self.single_blocks:
            joined = block(joined, vec)
        target = joined[:, cond.shape[1] :, :]
        return self.final_layer(target, vec).squeeze(-1)

    def ids_to_embedding(self, ids: Any) -> Any:
        return (self.source_embed(ids[:, 0]) + self.position_embed(ids[:, 1])).unsqueeze(0)


def split_modulation(values: Any) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    chunks = values.chunk(6, dim=-1)
    return (chunks[0], chunks[1], chunks[2]), (chunks[3], chunks[4], chunks[5])


def split_qkv_heads(qkv: Any, n_heads: int) -> tuple[Any, Any, Any]:
    import torch

    batch, tokens, channels3 = qkv.shape
    channels = channels3 // 3
    head_dim = channels // n_heads
    qkv = qkv.view(batch, tokens, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def scaled_dot_product_attention(q: Any, k: Any, v: Any) -> Any:
    import torch

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], -1)


def modulate(x: Any, shift: Any, scale: Any) -> Any:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_time_embedding(t: Any, dim: int) -> Any:
    import torch

    half = dim // 2
    if half <= 0:
        return t.view(-1, 1)
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device))
    args = t.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


if __name__ == "__main__":
    raise SystemExit(main())
