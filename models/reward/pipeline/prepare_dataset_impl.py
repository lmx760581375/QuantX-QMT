"""从统一特征存储构建 Reward 训练候选与未来路径标签。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (  # noqa: E402
    DEFAULT_BASELINE_PATH,
    DEFAULT_WTS_CONFIG,
    EXPERIMENT_ROOT,
    ExecutionRulesV1,
    MAINBOARD_BOARDS,
    RAW_OHLCV_FIELDS,
    WTSPaths,
    build_future_paths,
    date_indexer,
    equal_weight_baseline,
    execution_buyable,
    ensure_dirs,
    load_external_baseline,
    load_shard_lookup,
    load_yaml,
    required_formula_subset,
    source_memmaps,
    target_memmap,
    valid_history_mask,
    write_json,
)
from quantx.core.factor_runtime.panel import MarketPanel  # noqa: E402
from quantx.core.factor_runtime.runtime import FactorRuntime  # noqa: E402


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    prediction_start: str
    prediction_end: str
    horizon: int


OFFICIAL_FOLDS = (
    ("fold_2021", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("fold_2022", "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("fold_2023", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("fold_2024", "2016-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("fold_2025", "2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("forward_2026", "2016-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01", "2026-07-15"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--kronos-root", default=str(EXPERIMENT_ROOT.parent / "feature_store"))
    parser.add_argument("--strategy-config", default=str(DEFAULT_WTS_CONFIG))
    parser.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--candidate-topn", type=int, default=50)
    parser.add_argument(
        "--candidate-mode",
        default="strict_where",
        choices=["strict_where", "score_rank", "all_market"],
        help=(
            "strict_where ranks only selector.where hits; score_rank ranks all finite-score history-valid stocks and records selector.where as a feature column; "
            "all_market reuses the Kronos full sample index and does not apply weak-to-strong formulas."
        ),
    )
    parser.add_argument("--universe", default="mainboard", choices=["mainboard", "all_a"], help="Universe for all_market mode.")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--target-start-offset", type=int, default=1, help="First close offset from signal day T. 1 means T+1 close; 2 means T+2 close.")
    parser.add_argument("--target-price-mode", default="entry_open", choices=["entry_open", "close_to_close"])
    parser.add_argument("--trend-threshold", type=float, default=0.05, help="Final-return threshold for up/down/range trend labels.")
    parser.add_argument(
        "--trend-horizon",
        type=int,
        default=0,
        help="Endpoint used for up/down/range labels; 0 uses --horizon for backward compatibility.",
    )
    parser.add_argument("--balance-trend-train", action="store_true", help="Write an additional train_balanced split with up/down/range classes near 1/3 each.")
    parser.add_argument("--balance-seed", type=int, default=20260717)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--min-history-rows", type=int, default=55)
    parser.add_argument("--max-instruments", type=int, default=0, help="Smoke limit after mainboard filtering. 0 uses all.")
    parser.add_argument("--limit-days", type=int, default=0, help="Smoke limit for signal days. 0 uses all.")
    parser.add_argument("--target-chunk-size", type=int, default=200000, help="Rows per chunk when building all_market targets.")
    parser.add_argument("--keep-unbuyable", action="store_true", help="Keep unbuyable candidates for audit only; not recommended for training.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    paths = WTSPaths(root)
    ensure_dirs([paths.data, paths.folds, paths.reports, paths.runs])
    guard_outputs(paths, force=bool(args.force))
    payload = build_dataset(args, paths)
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def guard_outputs(paths: WTSPaths, *, force: bool) -> None:
    if force:
        return
    candidates = [paths.candidate_index, paths.target_meta, paths.manifest, *(paths.target_path(kind) for kind in ("relative", "abs", "baseline"))]
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing dataset artifacts without --force: " + ", ".join(existing))


def build_dataset(args: argparse.Namespace, paths: WTSPaths) -> dict[str, Any]:
    if str(args.candidate_mode) == "all_market":
        return build_all_market_dataset(args, paths)

    source = source_memmaps(args.kronos_root)
    config_path = Path(args.strategy_config).resolve()
    strategy = load_yaml(config_path)
    formulas = required_formula_subset(strategy)
    selector = dict(strategy.get("selector") or {})
    where_expr = str(selector.get("where", "buy_signal"))
    score_expr = str(selector.get("score", "score"))
    topn = int(args.candidate_topn) if int(args.candidate_topn) > 0 else int(selector.get("topk") or 50)

    calendar = source.calendar.copy()
    dates = pd.to_datetime(calendar["date"])
    instruments = source.instruments.copy()
    main = instruments["board"].isin(MAINBOARD_BOARDS) & (~instruments["is_index"].astype(bool))
    instruments = instruments.loc[main].sort_values("instrument_idx").reset_index(drop=True)
    if int(args.max_instruments) > 0:
        instruments = instruments.head(int(args.max_instruments)).copy()
    main_idx = instruments["instrument_idx"].to_numpy(dtype=np.int64)
    symbols = instruments["instrument"].astype(str).tolist()
    if len(main_idx) == 0:
        raise RuntimeError("No mainboard instruments available for candidate construction")

    panel = build_factor_panel(source, dates, main_idx, symbols)
    runtime = FactorRuntime(panel)
    runtime.compute_formulas({**formulas, "__selector_where": where_expr, "__selector_score": score_expr})
    where = np.asarray(runtime.values["__selector_where"], dtype=bool)
    score = np.asarray(runtime.values["__selector_score"], dtype=np.float32)
    if where.shape != score.shape:
        raise RuntimeError(f"where/score shape mismatch: {where.shape} vs {score.shape}")

    close = panel.fields["close"]
    baseline, baseline_meta = load_external_baseline(Path(args.baseline_path), dates)
    if baseline is None:
        baseline = equal_weight_baseline(close)
        baseline_meta = {
            **baseline_meta,
            "source": "mainboard_equal_weight_close",
            "reason": "external baseline file missing",
            "instrument_count": int(len(main_idx)),
        }

    history_ok = valid_history_mask(
        source.raw_ohlcv[main_idx],
        source.raw_state[main_idx],
        min_rows=int(args.min_history_rows),
        window=int(args.lookback),
    )
    shard_lookup = load_shard_lookup(source.feature_manifest)
    rules = ExecutionRulesV1()
    start_idx = int(dates.searchsorted(pd.Timestamp(args.start), side="left"))
    end_idx = int(dates.searchsorted(pd.Timestamp(args.end), side="right")) - 1
    first_signal = max(start_idx, int(args.lookback) - 1)
    last_signal = min(end_idx, len(dates) - int(args.target_start_offset) - int(args.horizon))
    signal_indices = list(range(first_signal, last_signal + 1))
    if int(args.limit_days) > 0:
        signal_indices = signal_indices[: int(args.limit_days)]

    rows: list[dict[str, Any]] = []
    y_abs_blocks: list[np.ndarray] = []
    y_baseline_blocks: list[np.ndarray] = []
    y_relative_blocks: list[np.ndarray] = []
    reason_counts: dict[str, int] = {}
    empty_pool_days = 0
    for day_no, t in enumerate(signal_indices, start=1):
        finite_score = np.isfinite(score[t])
        if args.candidate_mode == "score_rank":
            mask = finite_score & history_ok[:, t]
        else:
            mask = where[t] & finite_score & history_ok[:, t]
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            empty_pool_days += 1
            continue
        order = pool[np.argsort(-score[t, pool], kind="mergesort")]
        selected = order[:topn]
        pool_size = int(len(pool))
        pool_scores = score[t, pool].astype(np.float32)
        score_mean = float(np.nanmean(pool_scores)) if len(pool_scores) else 0.0
        score_std = float(np.nanstd(pool_scores)) if len(pool_scores) else 0.0
        if not np.isfinite(score_std) or score_std < 1e-6:
            score_std = 1.0
        buyable_rank = 0
        for rank0, local_pos in enumerate(selected):
            inst_idx = int(main_idx[local_pos])
            instrument = symbols[int(local_pos)]
            raw = source.raw_ohlcv[inst_idx]
            state = source.raw_state[inst_idx]
            ok, reason = execution_buyable(rules=rules, instrument=instrument, raw=raw, state=state, entry_idx=t + 1)
            if not ok and not bool(args.keep_unbuyable):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                continue
            y_abs, y_baseline, y_relative, target_reason = build_future_paths(
                raw=raw,
                baseline_close=baseline,
                date_idx=t,
                horizon=int(args.horizon),
                target_start_offset=int(args.target_start_offset),
                target_price_mode=str(args.target_price_mode),
            )
            if y_abs is None or y_baseline is None or y_relative is None:
                reason_counts[target_reason] = reason_counts.get(target_reason, 0) + 1
                continue
            shard_id, local_inst = shard_lookup[inst_idx]
            target_row = len(rows)
            if ok:
                buyable_rank += 1
            rows.append(
                {
                    "row_id": target_row,
                    "target_row": target_row,
                    "instrument_idx": inst_idx,
                    "shard_id": int(shard_id),
                    "local_instrument_idx": int(local_inst),
                    "date_idx": int(t),
                    "entry_date_idx": int(t + 1),
                    "instrument": instrument,
                    "signal_date": str(dates[t].date()),
                    "entry_date": str(dates[t + 1].date()),
                    "label_end_date": str(dates[t + int(args.target_start_offset) + int(args.horizon) - 1].date()),
                    "year": int(str(dates[t].date())[:4]),
                    "board": str(instruments.iloc[int(local_pos)]["board"]),
                    "asset_bucket": int(instruments.iloc[int(local_pos)]["asset_bucket"]),
                    "wts_rank": int(rank0 + 1),
                    "wts_rank_pct": float((rank0 + 1) / max(pool_size, 1)),
                    "wts_buyable_rank": int(buyable_rank) if ok else -1,
                    "wts_score": float(score[t, local_pos]),
                    "wts_score_z": float((float(score[t, local_pos]) - score_mean) / score_std),
                    "wts_score_pct": float(1.0 - rank0 / max(pool_size - 1, 1)),
                    "wts_selector_where": bool(where[t, local_pos]),
                    "wts_pool_size": pool_size,
                    "topn": int(topn),
                    "buyable": bool(ok),
                    "buy_block_reason": reason,
                }
            )
            y_abs_blocks.append(y_abs)
            y_baseline_blocks.append(y_baseline)
            y_relative_blocks.append(y_relative)
        if day_no % 250 == 0 or day_no == len(signal_indices):
            print(
                f"[build] days={day_no}/{len(signal_indices)} date={dates[t].date()} rows={len(rows)} skipped={sum(reason_counts.values())}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No weak-to-strong diffusion samples were generated")
    candidate = pd.DataFrame(rows)
    write_targets(
        paths,
        y_abs_blocks,
        y_baseline_blocks,
        y_relative_blocks,
        horizon=int(args.horizon),
        target_start_offset=int(args.target_start_offset),
        target_price_mode=str(args.target_price_mode),
    )
    candidate.to_parquet(paths.candidate_index, index=False, compression="zstd")
    fold_payload = write_all_folds(paths, candidate, horizon=int(args.horizon))
    manifest = {
        "kind": "weak_to_strong_diffusion_candidate_dataset_v1",
        "status": "built",
        "root": str(paths.root),
        "kronos_root": str(Path(args.kronos_root).resolve()),
        "strategy_config": str(config_path),
        "strategy_name": strategy.get("name"),
        "selector_where": where_expr,
        "selector_score": score_expr,
        "candidate_mode": str(args.candidate_mode),
        "required_formula_count": len(formulas),
        "candidate_topn": int(topn),
        "horizon": int(args.horizon),
        "target_start_offset": int(args.target_start_offset),
        "target_price_mode": str(args.target_price_mode),
        "trend_threshold": float(args.trend_threshold),
        "balance_trend_train": bool(args.balance_trend_train),
        "lookback": int(args.lookback),
        "start": str(args.start),
        "end": str(args.end),
        "mainboard_instruments": int(len(main_idx)),
        "signal_days": int(len(signal_indices)),
        "empty_pool_days": int(empty_pool_days),
        "samples": int(len(candidate)),
        "buyable_samples": int(candidate["buyable"].sum()),
        "skip_reason_counts": {key: int(value) for key, value in sorted(reason_counts.items())},
        "baseline": baseline_meta,
        "candidate_index": str(paths.candidate_index),
        "target_meta": str(paths.target_meta),
        "folds": fold_payload,
    }
    write_json(paths.manifest, manifest)
    return {"samples": int(len(candidate)), "candidate_index": str(paths.candidate_index), "manifest": str(paths.manifest)}


def resolve_trend_horizon(args: argparse.Namespace) -> int:
    target_horizon = int(args.horizon)
    trend_horizon = int(args.trend_horizon) if int(args.trend_horizon) > 0 else target_horizon
    if not 1 <= trend_horizon <= target_horizon:
        raise ValueError(
            f"--trend-horizon must be within [1, --horizon], received trend_horizon={trend_horizon}, horizon={target_horizon}"
        )
    return trend_horizon


def build_all_market_dataset(args: argparse.Namespace, paths: WTSPaths) -> dict[str, Any]:
    """Build full-market path targets without weak-to-strong candidate filtering.

    This mode deliberately keeps the downstream row schema compatible with
    WeakToStrongPathDataset while neutralising weak-to-strong rank features.
    """
    source = source_memmaps(args.kronos_root)
    trend_horizon = resolve_trend_horizon(args)
    dates = pd.to_datetime(source.calendar["date"])
    baseline, baseline_meta = load_external_baseline(Path(args.baseline_path), dates)
    if baseline is None:
        close = np.asarray(source.raw_ohlcv[:, :, RAW_OHLCV_FIELDS.index("close")], dtype=np.float32).T
        baseline = equal_weight_baseline(close)
        baseline_meta = {
            **baseline_meta,
            "source": "all_market_equal_weight_close",
            "reason": "external baseline file missing",
            "instrument_count": int(source.raw_ohlcv.shape[0]),
        }

    sample = load_all_market_sample_index(args, source, dates)
    target_ok = compute_all_market_target_validity(
        sample,
        source=source,
        baseline=baseline,
        horizon=int(args.horizon),
        target_start_offset=int(args.target_start_offset),
        target_price_mode=str(args.target_price_mode),
        chunk_size=int(args.target_chunk_size),
    )
    invalid_targets = int((~target_ok).sum())
    sample = sample.loc[target_ok].reset_index(drop=True)
    if sample.empty:
        raise RuntimeError("No all_market diffusion samples were generated")

    sample = finalize_all_market_rows(
        sample,
        dates=dates,
        horizon=int(args.horizon),
        target_start_offset=int(args.target_start_offset),
    )
    write_all_market_targets(
        paths,
        sample,
        source=source,
        baseline=baseline,
        horizon=int(args.horizon),
        target_start_offset=int(args.target_start_offset),
        target_price_mode=str(args.target_price_mode),
        chunk_size=int(args.target_chunk_size),
    )
    sample = annotate_trend_labels(
        paths,
        sample,
        horizon=trend_horizon,
        threshold=float(args.trend_threshold),
    )
    sample.to_parquet(paths.candidate_index, index=False, compression="zstd")
    fold_payload = write_all_market_folds(
        paths,
        sample,
        horizon=int(args.horizon),
        balance_trend_train=bool(args.balance_trend_train),
        balance_seed=int(args.balance_seed),
    )
    manifest = {
        "kind": "market_all_path_diffusion_dataset_v1",
        "status": "built",
        "root": str(paths.root),
        "kronos_root": str(Path(args.kronos_root).resolve()),
        "candidate_mode": "all_market",
        "universe": str(args.universe),
        "candidate_topn": 0,
        "horizon": int(args.horizon),
        "target_start_offset": int(args.target_start_offset),
        "target_price_mode": str(args.target_price_mode),
        "trend_horizon": int(trend_horizon),
        "trend_threshold": float(args.trend_threshold),
        "balance_trend_train": bool(args.balance_trend_train),
        "balance_seed": int(args.balance_seed),
        "lookback": int(args.lookback),
        "start": str(args.start),
        "end": str(args.end),
        "samples": int(len(sample)),
        "invalid_target_rows": invalid_targets,
        "signal_days": int(sample["signal_date"].astype(str).nunique()),
        "instrument_count": int(sample["instrument"].astype(str).nunique()),
        "baseline": baseline_meta,
        "candidate_index": str(paths.candidate_index),
        "target_meta": str(paths.target_meta),
        "folds": fold_payload,
        "neutral_candidate_features": True,
    }
    write_json(paths.manifest, manifest)
    return {"samples": int(len(sample)), "candidate_index": str(paths.candidate_index), "manifest": str(paths.manifest)}


def load_all_market_sample_index(args: argparse.Namespace, source, dates: pd.DatetimeIndex) -> pd.DataFrame:
    columns = [
        "row_id",
        "instrument_idx",
        "shard_id",
        "local_instrument_idx",
        "date_idx",
        "instrument",
        "signal_date",
        "year",
        "entry_feasible",
        "label_execution_invalid",
        "board",
        "asset_bucket",
    ]
    sample = pd.read_parquet(source.paths.sample_index, columns=columns)
    mask = sample["entry_feasible"].astype(bool) & (~sample["label_execution_invalid"].astype(bool))
    if str(args.universe) == "mainboard":
        mask &= sample["board"].isin(MAINBOARD_BOARDS)
    start = str(args.start)
    end = str(args.end)
    mask &= sample["signal_date"].astype(str).between(start, end)
    mask &= sample["date_idx"].astype(np.int64) >= int(args.lookback) - 1
    mask &= sample["date_idx"].astype(np.int64) <= len(dates) - int(args.target_start_offset) - int(args.horizon)
    if int(args.max_instruments) > 0:
        keep_inst = (
            source.instruments
            .sort_values("instrument_idx")
            .head(int(args.max_instruments))["instrument_idx"]
            .astype(int)
            .to_numpy()
        )
        mask &= sample["instrument_idx"].isin(keep_inst)
    sample = sample.loc[mask].copy()
    if int(args.limit_days) > 0 and not sample.empty:
        keep_dates = sorted(sample["signal_date"].astype(str).unique())[: int(args.limit_days)]
        sample = sample.loc[sample["signal_date"].astype(str).isin(keep_dates)].copy()
    sample = sample.sort_values(["date_idx", "instrument_idx"], kind="mergesort").reset_index(drop=True)
    return sample


def compute_all_market_target_validity(
    sample: pd.DataFrame,
    *,
    source,
    baseline: np.ndarray,
    horizon: int,
    target_start_offset: int,
    target_price_mode: str,
    chunk_size: int,
) -> np.ndarray:
    n = int(len(sample))
    ok = np.zeros(n, dtype=bool)
    open_idx = RAW_OHLCV_FIELDS.index("open")
    close_idx = RAW_OHLCV_FIELDS.index("close")
    inst_all = sample["instrument_idx"].to_numpy(dtype=np.int64)
    date_all = sample["date_idx"].to_numpy(dtype=np.int64)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        inst = inst_all[start:end]
        date_idx = date_all[start:end]
        entry_idx = date_idx + 1
        first_target_idx = date_idx + int(target_start_offset)
        entry_open = np.asarray(source.raw_ohlcv[inst, entry_idx, open_idx], dtype=np.float32)
        first_close = np.asarray(source.raw_ohlcv[inst, first_target_idx, close_idx], dtype=np.float32)
        if target_price_mode == "entry_open":
            price0 = entry_open
            base0 = np.asarray(baseline[date_idx], dtype=np.float32)
        elif target_price_mode == "close_to_close":
            price0 = first_close
            base0 = np.asarray(baseline[first_target_idx], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported target_price_mode: {target_price_mode}")
        chunk_ok = np.isfinite(price0) & (price0 > 0) & np.isfinite(base0) & (base0 > 0)
        for offset in range(int(horizon)):
            future_idx = date_idx + int(target_start_offset) + offset
            close = np.asarray(source.raw_ohlcv[inst, future_idx, close_idx], dtype=np.float32)
            base_future = np.asarray(baseline[future_idx], dtype=np.float32)
            chunk_ok &= np.isfinite(close) & (close > 0) & np.isfinite(base_future) & (base_future > 0)
        ok[start:end] = chunk_ok
        if end == n or end % (chunk_size * 10) == 0:
            print(f"[all_market] target_validity rows={end}/{n} valid={int(ok[:end].sum())}", flush=True)
    return ok


def finalize_all_market_rows(sample: pd.DataFrame, *, dates: pd.DatetimeIndex, horizon: int, target_start_offset: int) -> pd.DataFrame:
    out = sample.copy()
    n = int(len(out))
    date_idx = out["date_idx"].to_numpy(dtype=np.int64)
    out["row_id"] = np.arange(n, dtype=np.int64)
    out["target_row"] = out["row_id"]
    out["entry_date_idx"] = date_idx + 1
    out["entry_date"] = [str(dates[int(idx) + 1].date()) for idx in date_idx]
    out["target_start_offset"] = int(target_start_offset)
    out["target_start_date"] = [str(dates[int(idx) + int(target_start_offset)].date()) for idx in date_idx]
    out["label_end_date"] = [str(dates[int(idx) + int(target_start_offset) + int(horizon) - 1].date()) for idx in date_idx]
    out["year"] = out["signal_date"].astype(str).str.slice(0, 4).astype(int)
    out["wts_rank"] = 2
    out["wts_rank_pct"] = 0.5
    out["wts_buyable_rank"] = 2
    out["wts_score"] = 0.0
    out["wts_score_z"] = 0.0
    out["wts_score_pct"] = 0.5
    out["wts_selector_where"] = True
    out["wts_pool_size"] = 4
    out["topn"] = 4
    out["buyable"] = True
    out["buy_block_reason"] = ""
    return out


def write_all_market_targets(
    paths: WTSPaths,
    sample: pd.DataFrame,
    *,
    source,
    baseline: np.ndarray,
    horizon: int,
    target_start_offset: int,
    target_price_mode: str,
    chunk_size: int,
) -> None:
    n = int(len(sample))
    shape = (n, int(horizon))
    mmap_abs = np.memmap(paths.target_path("abs"), dtype=np.float32, mode="w+", shape=shape)
    mmap_base = np.memmap(paths.target_path("baseline"), dtype=np.float32, mode="w+", shape=shape)
    mmap_rel = np.memmap(paths.target_path("relative"), dtype=np.float32, mode="w+", shape=shape)
    open_idx = RAW_OHLCV_FIELDS.index("open")
    close_idx = RAW_OHLCV_FIELDS.index("close")
    inst_all = sample["instrument_idx"].to_numpy(dtype=np.int64)
    date_all = sample["date_idx"].to_numpy(dtype=np.int64)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        inst = inst_all[start:end]
        date_idx = date_all[start:end]
        entry_idx = date_idx + 1
        first_target_idx = date_idx + int(target_start_offset)
        entry_open = np.asarray(source.raw_ohlcv[inst, entry_idx, open_idx], dtype=np.float32)
        first_close = np.asarray(source.raw_ohlcv[inst, first_target_idx, close_idx], dtype=np.float32)
        if target_price_mode == "entry_open":
            price0 = entry_open
            base0 = np.asarray(baseline[date_idx], dtype=np.float32)
        elif target_price_mode == "close_to_close":
            price0 = first_close
            base0 = np.asarray(baseline[first_target_idx], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported target_price_mode: {target_price_mode}")
        abs_block = np.empty((end - start, int(horizon)), dtype=np.float32)
        base_block = np.empty_like(abs_block)
        for offset in range(int(horizon)):
            future_idx = date_idx + int(target_start_offset) + offset
            close = np.asarray(source.raw_ohlcv[inst, future_idx, close_idx], dtype=np.float32)
            base_future = np.asarray(baseline[future_idx], dtype=np.float32)
            abs_block[:, offset] = close / price0 - 1.0
            base_block[:, offset] = base_future / base0 - 1.0
        mmap_abs[start:end] = abs_block
        mmap_base[start:end] = base_block
        mmap_rel[start:end] = abs_block - base_block
        if end == n or end % (chunk_size * 10) == 0:
            print(f"[all_market] wrote_targets rows={end}/{n}", flush=True)
    mmap_abs.flush()
    mmap_base.flush()
    mmap_rel.flush()
    write_json(
        paths.target_meta,
        {
            "shape": [int(shape[0]), int(shape[1])],
            "dtype": "float32",
            "axes": ["sample", "future_session"],
            "targets": {
                "abs": str(paths.target_path("abs")),
                "baseline": str(paths.target_path("baseline")),
                "relative": str(paths.target_path("relative")),
            },
            "definition": {
                "abs": target_definition_abs(target_start_offset=int(target_start_offset), horizon=int(horizon), target_price_mode=str(target_price_mode)),
                "baseline": target_definition_baseline(target_start_offset=int(target_start_offset), horizon=int(horizon), target_price_mode=str(target_price_mode)),
                "relative": "abs - baseline",
            },
            "target_start_offset": int(target_start_offset),
            "target_price_mode": str(target_price_mode),
        },
    )


def target_definition_abs(*, target_start_offset: int, horizon: int, target_price_mode: str) -> str:
    start = int(target_start_offset)
    end = start + int(horizon) - 1
    if target_price_mode == "entry_open":
        denom = "open(T+1)"
    elif target_price_mode == "close_to_close":
        denom = f"close(T+{start})"
    else:
        raise ValueError(f"Unsupported target_price_mode: {target_price_mode}")
    return f"close(T+{start}..T+{end}) / {denom} - 1"


def target_definition_baseline(*, target_start_offset: int, horizon: int, target_price_mode: str) -> str:
    start = int(target_start_offset)
    end = start + int(horizon) - 1
    if target_price_mode == "entry_open":
        denom = "baseline_close(T)"
    elif target_price_mode == "close_to_close":
        denom = f"baseline_close(T+{start})"
    else:
        raise ValueError(f"Unsupported target_price_mode: {target_price_mode}")
    return f"baseline_close(T+{start}..T+{end}) / {denom} - 1"


def write_all_market_folds(
    paths: WTSPaths,
    sample: pd.DataFrame,
    *,
    horizon: int,
    balance_trend_train: bool = False,
    balance_seed: int = 20260717,
) -> dict[str, Any]:
    spec = FoldSpec(
        fold_id="pre2020_eval2020_2026",
        train_start="2010-01-01",
        train_end="2018-12-31",
        validation_start="2019-01-01",
        validation_end="2019-12-31",
        prediction_start="2020-01-01",
        prediction_end="2026-07-15",
        horizon=int(horizon),
    )
    payload = write_fold(
        paths,
        sample,
        spec,
        balance_trend_train=bool(balance_trend_train),
        balance_seed=int(balance_seed),
    )
    payload["mode"] = "all_market_pre2020_path_v1"
    write_json(paths.folds / f"{spec.fold_id}.json", payload)
    return {spec.fold_id: payload}


def build_factor_panel(source, dates: pd.DatetimeIndex, main_idx: np.ndarray, symbols: list[str]) -> MarketPanel:
    raw_fields = {}
    for field in ("open", "high", "low", "close", "volume", "vwap", "change"):
        idx = RAW_OHLCV_FIELDS.index(field)
        raw_fields[field] = np.asarray(source.raw_ohlcv[main_idx, :, idx], dtype=np.float32).T
    return MarketPanel(dates=pd.DatetimeIndex(dates), instruments=pd.Index(symbols), fields=raw_fields)


def write_targets(
    paths: WTSPaths,
    y_abs_blocks: list[np.ndarray],
    y_baseline_blocks: list[np.ndarray],
    y_relative_blocks: list[np.ndarray],
    *,
    horizon: int,
    target_start_offset: int = 1,
    target_price_mode: str = "entry_open",
) -> None:
    blocks = {
        "abs": np.stack(y_abs_blocks).astype(np.float32),
        "baseline": np.stack(y_baseline_blocks).astype(np.float32),
        "relative": np.stack(y_relative_blocks).astype(np.float32),
    }
    shape = blocks["abs"].shape
    if shape[1] != int(horizon):
        raise RuntimeError(f"Target horizon mismatch: {shape[1]} != {horizon}")
    for kind, values in blocks.items():
        mmap = np.memmap(paths.target_path(kind), dtype=np.float32, mode="w+", shape=shape)
        mmap[:] = values
        mmap.flush()
    write_json(
        paths.target_meta,
        {
            "shape": [int(shape[0]), int(shape[1])],
            "dtype": "float32",
            "axes": ["sample", "future_session"],
            "targets": {
                "abs": str(paths.target_path("abs")),
                "baseline": str(paths.target_path("baseline")),
                "relative": str(paths.target_path("relative")),
            },
            "definition": {
                "abs": target_definition_abs(target_start_offset=int(target_start_offset), horizon=int(horizon), target_price_mode=str(target_price_mode)),
                "baseline": target_definition_baseline(target_start_offset=int(target_start_offset), horizon=int(horizon), target_price_mode=str(target_price_mode)),
                "relative": "abs - baseline",
            },
            "target_start_offset": int(target_start_offset),
            "target_price_mode": str(target_price_mode),
        },
    )


def write_all_folds(paths: WTSPaths, sample: pd.DataFrame, *, horizon: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for fold_id, train_start, train_end, val_start, val_end, pred_start, pred_end in OFFICIAL_FOLDS:
        spec = FoldSpec(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            validation_start=val_start,
            validation_end=val_end,
            prediction_start=pred_start,
            prediction_end=pred_end,
            horizon=int(horizon),
        )
        outputs[fold_id] = write_fold(paths, sample, spec)
    return outputs


def annotate_trend_labels(paths: WTSPaths, sample: pd.DataFrame, *, horizon: int, threshold: float) -> pd.DataFrame:
    out = sample.copy()
    target_rows = out["target_row"].to_numpy(dtype=np.int64)
    final_abs = np.asarray(target_memmap(paths, "abs", mode="r")[target_rows, int(horizon) - 1], dtype=np.float32)
    labels = np.full(len(out), "range", dtype=object)
    labels[final_abs >= float(threshold)] = "up"
    labels[final_abs <= -float(threshold)] = "down"
    out["trend_final_abs_return"] = final_abs
    out["trend_threshold"] = float(threshold)
    out["trend_label"] = labels
    return out


def make_balanced_trend_train(rows: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "trend_label" not in rows.columns:
        raise ValueError("Cannot build train_balanced split: missing trend_label column")
    labels = ("up", "down", "range")
    pools = {label: rows.loc[rows["trend_label"].astype(str) == label].copy() for label in labels}
    counts = {label: int(len(pool)) for label, pool in pools.items()}
    if counts["up"] == 0 or counts["down"] == 0 or counts["range"] == 0:
        raise RuntimeError(f"Cannot balance trend split with empty class: {counts}")
    target = max(counts["up"], counts["down"])
    balanced_parts = [
        sample_class_rows(pools["up"], n=target, seed=seed + 11),
        sample_class_rows(pools["down"], n=target, seed=seed + 17),
        sample_range_rows_by_instrument(pools["range"], n=target, seed=seed + 23),
    ]
    balanced = pd.concat(balanced_parts, axis=0, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=int(seed) + 31).sort_values(
        ["shard_id", "local_instrument_idx", "date_idx"],
        kind="mergesort",
    )
    balanced = balanced.reset_index(drop=True)
    balanced_counts = {label: int((balanced["trend_label"].astype(str) == label).sum()) for label in labels}
    return balanced, {
        "source_counts": counts,
        "balanced_counts": balanced_counts,
        "target_per_class": int(target),
        "seed": int(seed),
    }


def sample_class_rows(rows: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    replace = len(rows) < int(n)
    return rows.sample(n=int(n), replace=replace, random_state=int(seed)).copy()


def sample_range_rows_by_instrument(rows: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    rows = rows.copy()
    if len(rows) <= int(n):
        return rows.sample(n=int(n), replace=len(rows) < int(n), random_state=int(seed)).copy()
    rng = np.random.default_rng(int(seed))
    groups = list(rows.groupby("instrument_idx", sort=False).indices.items())
    if not groups:
        return rows.head(0).copy()
    per_inst = max(1, int(np.ceil(int(n) / len(groups))))
    chosen: list[np.ndarray] = []
    chosen_mask = np.zeros(len(rows), dtype=bool)
    for _, positions in groups:
        pos = np.asarray(positions, dtype=np.int64)
        take = min(len(pos), per_inst)
        if take <= 0:
            continue
        selected = rng.choice(pos, size=take, replace=False)
        chosen.append(selected)
        chosen_mask[selected] = True
    selected_pos = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if len(selected_pos) > int(n):
        selected_pos = rng.choice(selected_pos, size=int(n), replace=False)
    elif len(selected_pos) < int(n):
        remaining = np.flatnonzero(~chosen_mask)
        need = int(n) - len(selected_pos)
        if len(remaining) > 0:
            fill = rng.choice(remaining, size=need, replace=len(remaining) < need)
            selected_pos = np.concatenate([selected_pos, fill])
        else:
            fill = rng.choice(selected_pos, size=need, replace=True)
            selected_pos = np.concatenate([selected_pos, fill])
    return rows.iloc[selected_pos].copy()


def write_fold(
    paths: WTSPaths,
    sample: pd.DataFrame,
    spec: FoldSpec,
    *,
    balance_trend_train: bool = False,
    balance_seed: int = 20260717,
) -> dict[str, Any]:
    signal = sample["signal_date"].astype(str)
    label_end = sample["label_end_date"].astype(str)
    val_dates = sorted(sample.loc[signal.between(spec.validation_start, spec.validation_end), "signal_date"].astype(str).unique())
    if len(val_dates) >= 4:
        cut = max(1, int(len(val_dates) * 0.70))
        select_start = val_dates[0]
        select_end = val_dates[cut - 1]
        calibration_start = val_dates[min(cut + int(spec.horizon), len(val_dates) - 1)]
        calibration_end = val_dates[-1]
    else:
        select_start, select_end = spec.validation_start, spec.validation_end
        calibration_start, calibration_end = spec.validation_end, spec.validation_end

    splits = {
        "train": sample.loc[signal.between(spec.train_start, spec.train_end) & (label_end < spec.validation_start)].copy(),
        "validation": sample.loc[signal.between(spec.validation_start, spec.validation_end) & (label_end < spec.prediction_start)].copy(),
        "validation_select": sample.loc[signal.between(select_start, select_end) & (label_end < calibration_start)].copy(),
        "validation_calibration": sample.loc[
            signal.between(calibration_start, calibration_end) & (label_end < spec.prediction_start)
        ].copy(),
        "prediction": sample.loc[signal.between(spec.prediction_start, spec.prediction_end) & (label_end <= spec.prediction_end)].copy(),
    }
    balance_meta = None
    if bool(balance_trend_train):
        splits["train_balanced"], balance_meta = make_balanced_trend_train(splits["train"], seed=int(balance_seed))
    fold_dir = paths.folds / spec.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}
    counts = {}
    for name, rows in splits.items():
        output = fold_dir / f"{name}.parquet"
        rows.to_parquet(output, index=False, compression="zstd")
        index_paths[name] = str(output)
        counts[name] = int(len(rows))
    payload = {
        **asdict(spec),
        "mode": "official_wts_path_v1",
        "validation_select_start": select_start,
        "validation_select_end": select_end,
        "validation_calibration_start": calibration_start,
        "validation_calibration_end": calibration_end,
        "counts": counts,
        "index_paths": index_paths,
    }
    if balance_meta is not None:
        payload["balance_trend_train"] = balance_meta
    write_json(paths.folds / f"{spec.fold_id}.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
