"""Export frozen-AE reward Transformer scores as a mandatory QuantX artifact.

DDP ranks own complete signal-date cross-sections, never row shards.  Every
successful run writes a validated parquet plus JSON manifest; offline capacity
metrics are diagnostic only and never replace a QuantX execution backtest.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifact import write_market_score_artifact  # noqa: E402
from .train_impl import (  # noqa: E402
    DEFAULT_KRONOS_ROOT,
    DEFAULT_ROOT,
    FrozenAELoopPreferenceTransformer,
    FrozenAERewardTransformer,
    MULTITASK_SCORE_HEADS,
    RawFeatureRewardTransformer,
    RawFeatureRewardTransformerConfig,
    RewardTransformerConfig,
    autocast_context,
    multitask_score_column,
    multitask_two_stage_score_column,
    multitask_two_stage_scores,
    reward_score_column,
    true_return_column,
)
from .runtime import barrier, broadcast_object, cleanup_distributed, init_distributed  # noqa: E402
from .common import (  # noqa: E402
    MARKET_FEATURE_FIELDS,
    ModelConfig,
    PATH_QUALITY_WEIGHTS,
    WTSPaths,
    WeakToStrongPathDataset,
    build_models,
    future_path_metrics,
    load_model_state_flexible,
    load_scaler,
    path_quality_from_metrics,
    read_json,
    set_seed,
)


FIXED_TOP_K = (1, 5, 10, 20)
DYNAMIC_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)
MULTITASK_CHECKPOINT_KIND = "wts_frozen_ae_multitask_reward_transformer_checkpoint_v1"
H7_ABSOLUTE_FULL_PAIR_CHECKPOINT_KIND = (
    "wts_frozen_ae_reward_transformer_h7_absolute_full_pair_v1"
)


def _multitask_inference_contract(checkpoint_args: dict[str, Any], rank_horizon: int) -> dict[str, Any]:
    """Read the immutable two-stage selection contract from a checkpoint."""

    required = (
        "multitask_return_top_n",
        "multitask_risk_sharpe_weight",
        "multitask_risk_drawdown_weight",
    )
    missing = [name for name in required if name not in checkpoint_args]
    if missing:
        raise RuntimeError(f"Multi-task checkpoint is missing selection-contract fields: {missing}")
    return_top_n = int(checkpoint_args["multitask_return_top_n"])
    risk_weights = (
        float(checkpoint_args["multitask_risk_sharpe_weight"]),
        float(checkpoint_args["multitask_risk_drawdown_weight"]),
    )
    if return_top_n < 5:
        raise RuntimeError("Multi-task checkpoint has multitask_return_top_n < 5")
    if any(not math.isfinite(value) or value < 0.0 for value in risk_weights):
        raise RuntimeError("Multi-task checkpoint has invalid risk rerank weights")
    if not math.isclose(sum(risk_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise RuntimeError("Multi-task checkpoint risk rerank weights must sum to 1")
    return {
        "model_score_columns": tuple(multitask_score_column(head, int(rank_horizon)) for head in MULTITASK_SCORE_HEADS),
        "two_stage_score_column": multitask_two_stage_score_column(int(rank_horizon)),
        "return_top_n": return_top_n,
        "risk_weights": risk_weights,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--kronos-root", default=str(DEFAULT_KRONOS_ROOT))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--rank-horizon",
        type=int,
        default=0,
        help="0 uses the checkpoint horizon. A nonzero value must match the checkpoint to prevent mislabeled scores.",
    )
    parser.add_argument("--ae-checkpoint", default="", help="Override the frozen AE path persisted in the reward checkpoint.")
    parser.add_argument("--fold-id", default="pre2020_eval2020_2026")
    parser.add_argument("--prediction-split", default="prediction")
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--max-dates", type=int, default=0, help="Diagnostic cap on complete signal-date cross-sections.")
    parser.add_argument("--stratified-max-rows", type=int, default=0, help="Diagnostic time-balanced row cap; not valid for capacity claims.")
    parser.add_argument(
        "--diagnostic-horizons",
        default="",
        help="Optional comma-separated future-path horizons for diagnostics, for example 5,10,15,20,30. Does not alter score parquet columns.",
    )
    parser.add_argument("--scaler-path", default="")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=32768)
    parser.add_argument("--batch-size-multiple", type=int, default=256)
    parser.add_argument("--auto-batch-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-gpu-memory-fraction", type=float, default=0.86)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", default="bf16", choices=["off", "bf16", "fp16"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ddp", default="auto", choices=["auto", "off"])
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-shards", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_diagnostic_horizons(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    try:
        horizons = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("--diagnostic-horizons must be a comma-separated integer list") from exc
    if not horizons or any(horizon < 3 for horizon in horizons):
        raise ValueError("--diagnostic-horizons must contain horizons >= 3")
    return horizons


def _path_metric_column(metric: str, horizon: int) -> str:
    return f"{metric}_{int(horizon)}d"


def _selected_dates(fold: dict, split_name: str, start: str, end: str) -> list[str]:
    paths = dict(fold.get("index_paths") or {})
    if split_name not in paths:
        raise KeyError(f"Prediction split {split_name!r} is unavailable; choices={sorted(paths)}")
    dates = pd.read_parquet(paths[split_name], columns=["signal_date"])["signal_date"]
    values = pd.to_datetime(dates, errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    selected = [date for date in sorted(set(values.tolist())) if str(start) <= date <= str(end)]
    if not selected:
        raise RuntimeError(f"No signal dates in {start} through {end}")
    return selected


def _read_rows(path: str | Path, dates: list[str]) -> pd.DataFrame:
    try:
        rows = pd.read_parquet(path, filters=[("signal_date", "in", dates)])
    except Exception:
        rows = pd.read_parquet(path)
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = rows.loc[rows["signal_date"].isin(set(dates))].sort_values(["signal_date", "instrument"]).reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("Assigned date partition contains no prediction rows")
    return rows


def _stratified_rows(rows: pd.DataFrame, all_dates: list[str], target_rows: int, seed: int = 20260720) -> pd.DataFrame:
    if int(target_rows) <= 0:
        return rows
    if int(target_rows) < len(all_dates):
        raise ValueError("--stratified-max-rows must retain at least one row per selected signal_date")
    base, remainder = divmod(int(target_rows), len(all_dates))
    quotas = {date: base + int(index < remainder) for index, date in enumerate(all_dates)}
    parts: list[pd.DataFrame] = []
    for date, group in rows.groupby("signal_date", sort=False):
        quota = quotas[str(date)]
        if len(group) <= quota:
            parts.append(group)
        else:
            date_seed = (seed + int(pd.Timestamp(date).strftime("%Y%m%d"))) % (2**32 - 1)
            parts.append(group.sample(n=quota, random_state=date_seed))
    return pd.concat(parts, ignore_index=True).sort_values(["signal_date", "instrument"]).reset_index(drop=True)


def _load_models(args: argparse.Namespace, device: torch.device):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = dict(checkpoint.get("args") or {})
    input_mode = str(saved_args.get("input_mode", "raw_relative"))
    kind = str(checkpoint.get("kind", ""))
    if kind == "wts_raw_reward_transformer_checkpoint_v1":
        required = ("reward_state", "raw_model_config")
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise RuntimeError(f"Raw reward checkpoint is missing required fields: {missing}")
        reward_config = RawFeatureRewardTransformerConfig(**dict(checkpoint["raw_model_config"]))
        reward = RawFeatureRewardTransformer(reward_config)
        load_model_state_flexible(reward, checkpoint["reward_state"])
        reward.to(device).eval()
        for parameter in reward.parameters():
            parameter.requires_grad_(False)
        return checkpoint_path, checkpoint, "raw", "transformer", None, None, reward_config, None, reward, input_mode
    frozen_kinds = {
        "wts_frozen_ae_reward_transformer_checkpoint_v1",
        "wts_frozen_ae_loop_preference_checkpoint_v1",
        MULTITASK_CHECKPOINT_KIND,
        H7_ABSOLUTE_FULL_PAIR_CHECKPOINT_KIND,
    }
    if kind not in frozen_kinds:
        raise RuntimeError(f"Unsupported frozen-AE reward checkpoint kind: {kind!r}")
    required = ("reward_state", "reward_config", "ae_model_config", "ae_checkpoint")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise RuntimeError(f"Frozen-AE reward checkpoint is missing required fields: {missing}")
    ae_config = ModelConfig(**dict(checkpoint["ae_model_config"]))
    reward_config = RewardTransformerConfig(**dict(checkpoint["reward_config"]))
    if kind == MULTITASK_CHECKPOINT_KIND and tuple(reward_config.score_heads) != MULTITASK_SCORE_HEADS:
        raise RuntimeError(
            "Multi-task checkpoint must expose exactly the return, Sharpe, and drawdown score heads; "
            f"got {reward_config.score_heads!r}"
        )
    stock_dim = 42 if input_mode == "relative_only" else 52
    ae, _ = build_models(ae_config, stock_dim=stock_dim, market_dim=len(MARKET_FEATURE_FIELDS))
    ae_checkpoint = Path(args.ae_checkpoint or checkpoint["ae_checkpoint"]).expanduser().resolve()
    if not ae_checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen AE checkpoint: {ae_checkpoint}")
    ae_state = torch.load(ae_checkpoint, map_location="cpu", weights_only=False)
    state = ae_state.get("ae_state") if isinstance(ae_state, dict) else None
    if state is None and isinstance(ae_state, dict):
        state = ae_state.get("model")
    if state is None:
        raise RuntimeError(f"AE checkpoint has no ae_state/model: {ae_checkpoint}")
    load_model_state_flexible(ae, state)
    architecture = "loop_preference" if kind == "wts_frozen_ae_loop_preference_checkpoint_v1" else "transformer"
    if architecture == "loop_preference":
        reward = FrozenAELoopPreferenceTransformer(
            reward_config,
            rollout_steps=int(saved_args.get("loop_steps", 0)),
            condition_update=str(saved_args.get("loop_condition_update", "")),
        )
    else:
        reward = FrozenAERewardTransformer(reward_config)
    load_model_state_flexible(reward, checkpoint["reward_state"])
    for module in (ae, reward):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return checkpoint_path, checkpoint, "frozen_ae", architecture, ae_checkpoint, ae_config, reward_config, ae, reward, input_mode


def _round_batch(value: float, multiple: int, maximum: int, minimum: int) -> int:
    rounded = int(math.floor(float(value) / max(1, multiple)) * max(1, multiple))
    return max(minimum, min(int(maximum), rounded))


@torch.no_grad()
def _predict_score(ae, reward, x_ts, x_market, x_candidate, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    x_ts = x_ts.to(device, dtype=torch.float32, non_blocking=True)
    x_market = x_market.to(device, dtype=torch.float32, non_blocking=True)
    x_candidate = x_candidate.to(device, dtype=torch.float32, non_blocking=True)
    with autocast_context(args, device):
        if ae is None:
            logits = reward(x_ts, x_market, x_candidate)
        else:
            condition = ae.encode(x_ts, x_market, x_candidate)
            logits = reward(condition)
    # Pairwise ranking learns logits.  Applying sigmoid in BF16 would collapse
    # high but distinct logits to 1.0, so the public [0, 1] score is formed in
    # FP32 after leaving the autocast region.
    return torch.sigmoid(logits.float()).clamp_(1.0e-6, 1.0 - 1.0e-6)


def _probe_batch_size(dataset, ae, reward, args: argparse.Namespace, device: torch.device) -> tuple[int, list[dict[str, Any]]]:
    requested = min(int(args.batch_size), int(args.max_batch_size), len(dataset))
    minimum = min(max(1, int(args.batch_size_multiple)), requested)
    if not bool(args.auto_batch_size) or device.type != "cuda":
        return requested, [{"batch_size": requested, "status": "manual_or_cpu"}]
    target = float(args.target_gpu_memory_fraction)
    if not 0.50 <= target <= 0.92:
        raise ValueError("--target-gpu-memory-fraction must be within [0.50, 0.92]")
    candidate = requested
    best = 0
    attempts: list[dict[str, Any]] = []
    for _ in range(6):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            items = [dataset[index] for index in range(candidate)]
            score = _predict_score(
                ae,
                reward,
                torch.stack([item[0] for item in items]),
                torch.stack([item[1] for item in items]),
                torch.stack([item[2] for item in items]),
                args,
                device,
            )
            peak = int(torch.cuda.max_memory_allocated(device))
            total = int(torch.cuda.get_device_properties(device).total_memory)
            fraction = float(peak / total)
            attempts.append({"batch_size": candidate, "status": "ok", "peak_memory_bytes": peak, "memory_fraction": fraction})
            del items, score
            torch.cuda.empty_cache()
            best = candidate
            proposed = _round_batch(
                candidate * target / max(fraction, 1.0e-6) * 0.985,
                int(args.batch_size_multiple),
                int(args.max_batch_size),
                minimum,
            )
            if proposed <= candidate or fraction >= target * 0.96 or candidate >= int(args.max_batch_size):
                break
            candidate = min(proposed, len(dataset))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            attempts.append({"batch_size": candidate, "status": "oom"})
            candidate = _round_batch(candidate * 0.70, int(args.batch_size_multiple), int(args.max_batch_size), minimum)
            if candidate >= best and best > 0:
                break
        if candidate < minimum:
            break
    if best <= 0:
        raise RuntimeError(f"Unable to infer even batch_size={minimum}; attempts={attempts}")
    return best, attempts


def _capacity_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    truth_column: str,
    event_threshold: float,
    metric_label: str = "return",
) -> dict[str, Any]:
    rankics: list[float] = []
    metrics: dict[str, dict[str, float]] = {}
    selections: dict[str, list[np.ndarray]] = {**{f"top{count}": [] for count in FIXED_TOP_K}, **{f"top{fraction * 100:g}pct": [] for fraction in DYNAMIC_FRACTIONS}}
    bases: dict[str, list[np.ndarray]] = {key: [] for key in selections}
    for _, group in frame.groupby("signal_date", sort=False):
        scores = group[score_column].to_numpy(dtype=np.float64)
        truth = group[truth_column].to_numpy(dtype=np.float64)
        if len(group) >= 2 and np.unique(scores).size > 1 and np.unique(truth).size > 1:
            value = pd.Series(scores).corr(pd.Series(truth), method="spearman")
            if value is not None and np.isfinite(value):
                rankics.append(float(value))
        order = np.argsort(-scores, kind="mergesort")
        for count in FIXED_TOP_K:
            key = f"top{count}"
            selected = truth[order[: min(count, len(order))]]
            selections[key].append(selected)
            bases[key].append(np.full(len(selected), truth.mean(), dtype=np.float64))
        for fraction in DYNAMIC_FRACTIONS:
            key = f"top{fraction * 100:g}pct"
            count = max(1, int(math.ceil(len(group) * fraction)))
            selected = truth[order[:count]]
            selections[key].append(selected)
            bases[key].append(np.full(len(selected), truth.mean(), dtype=np.float64))
    for key, selected_parts in selections.items():
        selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.float64)
        base = np.concatenate(bases[key]) if bases[key] else np.empty(0, dtype=np.float64)
        item: dict[str, float | int | None] = {
            "selected_rows": int(len(selected)),
            "positive_rate": float((selected > 0.0).mean()) if len(selected) else None,
        }
        if str(metric_label) == "return":
            item.update(
                {
                    "selected_avg_return": float(selected.mean()) if len(selected) else None,
                    "base_avg_return": float(base.mean()) if len(base) else None,
                    "excess_avg_return": float(selected.mean() - base.mean()) if len(selected) else None,
                    f"p_return_ge_{float(event_threshold) * 100:g}pct": float((selected >= float(event_threshold)).mean()) if len(selected) else None,
                }
            )
        else:
            item.update(
                {
                    f"selected_avg_{metric_label}": float(selected.mean()) if len(selected) else None,
                    f"base_avg_{metric_label}": float(base.mean()) if len(base) else None,
                    f"selection_advantage_{metric_label}": float(selected.mean() - base.mean()) if len(selected) else None,
                    f"p_{metric_label}_ge_{float(event_threshold) * 100:g}pct": float((selected >= float(event_threshold)).mean()) if len(selected) else None,
                }
            )
        metrics[key] = item
    return {
        "rows": int(len(frame)),
        "dates": int(frame["signal_date"].nunique()),
        "rankic_mean": float(np.mean(rankics)) if rankics else None,
        "rankic_median": float(np.median(rankics)) if rankics else None,
        "rankic_positive_ratio": float(np.mean(np.asarray(rankics) > 0.0)) if rankics else None,
        "score": {
            "column": score_column,
            "min": float(frame[score_column].min()),
            "max": float(frame[score_column].max()),
            "mean": float(frame[score_column].mean()),
        },
        "capacity": metrics,
    }


def _path_quality_diagnostics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    horizon: int,
    event_threshold: float,
) -> dict[str, Any]:
    terminal_column = _path_metric_column("terminal_return", horizon)
    sharpe_column = _path_metric_column("path_sharpe", horizon)
    sortino_column = _path_metric_column("path_sortino", horizon)
    drawdown_column = _path_metric_column("max_drawdown", horizon)
    quality_column = _path_metric_column("path_quality", horizon)
    drawdown_rank_column = _path_metric_column("signed_max_drawdown", horizon)
    audit = frame
    audit[drawdown_rank_column] = audit[drawdown_column]
    return {
        "horizon": int(horizon),
        "quality_score": _capacity_metrics(
            audit,
            score_column=score_column,
            truth_column=quality_column,
            event_threshold=0.0,
            metric_label="quality_score",
        ),
        "terminal_return": _capacity_metrics(
            audit,
            score_column=score_column,
            truth_column=terminal_column,
            event_threshold=event_threshold,
            metric_label="terminal_return",
        ),
        "path_sharpe": _capacity_metrics(
            audit,
            score_column=score_column,
            truth_column=sharpe_column,
            event_threshold=0.0,
            metric_label="path_sharpe",
        ),
        "path_sortino": _capacity_metrics(
            audit,
            score_column=score_column,
            truth_column=sortino_column,
            event_threshold=0.0,
            metric_label="path_sortino",
        ),
        "max_drawdown": {
            "contract": "signed maximum drawdown; larger is better because values are zero for no drawdown and negative otherwise",
            "metrics": _capacity_metrics(
                audit,
                score_column=score_column,
                truth_column=drawdown_rank_column,
                event_threshold=0.0,
                metric_label="signed_max_drawdown",
            ),
        },
    }


def _shard_paths(output: Path, rank: int) -> tuple[Path, Path, Path]:
    directory = output.parent / f"{output.stem}_ddp_shards"
    return directory / f"rank{rank:02d}.parquet", directory / f"rank{rank:02d}.metrics.parquet", directory / f"rank{rank:02d}.summary.json"


def _run_rank(
    args,
    ddp,
    rows,
    ae,
    reward,
    input_mode: str,
    rank_horizon: int,
    model_score_columns: tuple[str, ...],
    multitask: bool,
    truth_column: str,
    metric_horizons: tuple[int, ...],
    future_targets_available: bool,
    output: Path,
) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    saved_args = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False).get("args", {})
    scaler_path = (
        Path(args.scaler_path).expanduser().resolve()
        if args.scaler_path
        else Path(saved_args.get("scaler_path") or root / "data" / "scalers" / f"{args.fold_id}_{input_mode}_scaler.json").resolve()
    )
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    dataset = WeakToStrongPathDataset(
        kronos_root=args.kronos_root,
        experiment_root=root,
        rows=rows,
        scaler=load_scaler(scaler_path),
        target_kind="abs" if future_targets_available else None,
        input_mode=input_mode,
    )
    batch_size, probe_attempts = _probe_batch_size(dataset, ae, reward, args, ddp.device)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": ddp.device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    diagnostic_parts: dict[int, dict[str, list[np.ndarray]]] = {
        int(horizon): {name: [] for name in ("terminal_return", "path_sharpe", "path_sortino", "max_drawdown")}
        for horizon in metric_horizons
    }
    started = time.perf_counter()
    for step, (x_ts, x_market, x_candidate, target) in enumerate(loader, start=1):
        score = _predict_score(ae, reward, x_ts, x_market, x_candidate, args, ddp.device)
        scores.append(score.cpu().numpy())
        if future_targets_available:
            if int(rank_horizon) > int(target.shape[1]):
                raise ValueError(f"checkpoint rank_horizon={rank_horizon} exceeds target horizon={target.shape[1]}")
            targets.append(target[:, int(rank_horizon) - 1].float().numpy())
        if future_targets_available and diagnostic_parts:
            target_values = target.float().numpy()
            for horizon, parts in diagnostic_parts.items():
                metrics = future_path_metrics(target_values, horizon=int(horizon))
                for name, values in metrics.items():
                    parts[name].append(values)
        done = sum(len(item) for item in scores)
        if step == 1 or step % int(args.log_every_steps) == 0 or done == len(rows):
            event = {
                "event": "infer_progress",
                "rank": int(ddp.rank),
                "step": int(step),
                "rows_done": int(done),
                "rows_total": int(len(rows)),
                "batch_size": int(batch_size),
                "elapsed_sec": round(time.perf_counter() - started, 1),
            }
            if ddp.device.type == "cuda":
                event["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated(ddp.device) / (1024**3), 2)
            print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
    score_values = np.concatenate(scores).astype(np.float32, copy=False)
    truth_values = (
        np.concatenate(targets).astype(np.float32, copy=False)
        if targets
        else None
    )
    expected_shape = (len(rows), len(model_score_columns)) if multitask else (len(rows),)
    if tuple(score_values.shape) != expected_shape:
        raise RuntimeError(f"Inference rows do not align: scores={len(score_values)}, rows={len(rows)}")
    shard, metric_shard, summary_path = _shard_paths(output, ddp.rank)
    shard.parent.mkdir(parents=True, exist_ok=True)
    score_frame = rows.loc[:, ["signal_date", "instrument"]].copy()
    if multitask:
        for head_index, score_column in enumerate(model_score_columns):
            score_frame[score_column] = score_values[:, head_index]
    else:
        score_frame[model_score_columns[0]] = score_values
    score_frame.to_parquet(shard, index=False, compression="zstd")
    metric_frame = (
        score_frame.assign(**{truth_column: truth_values})
        if truth_values is not None
        else score_frame.copy()
    )
    for horizon, parts in diagnostic_parts.items():
        metric_values = {name: np.concatenate(values).astype(np.float32, copy=False) for name, values in parts.items()}
        if any(len(values) != len(rows) for values in metric_values.values()):
            raise RuntimeError(f"Diagnostic metric rows do not align for horizon={horizon}")
        quality = path_quality_from_metrics(
            signal_dates=rows["signal_date"].astype(str).to_numpy(),
            terminal_return=metric_values["terminal_return"],
            path_sharpe=metric_values["path_sharpe"],
            max_drawdown=metric_values["max_drawdown"],
        )
        for name, values in metric_values.items():
            metric_frame[_path_metric_column(name, int(horizon))] = values
        metric_frame[_path_metric_column("path_quality", int(horizon))] = quality["quality_score"]
    metric_frame.to_parquet(metric_shard, index=False, compression="zstd")
    peak = int(torch.cuda.max_memory_allocated(ddp.device)) if ddp.device.type == "cuda" else 0
    total = int(torch.cuda.get_device_properties(ddp.device).total_memory) if ddp.device.type == "cuda" else 0
    summary = {
        "rank": int(ddp.rank),
        "rank_horizon": int(rank_horizon),
        "metric_horizons": [int(horizon) for horizon in metric_horizons],
        "model_score_columns": list(model_score_columns),
        "rows": int(len(rows)),
        "dates": int(rows["signal_date"].nunique()),
        "batch_size": int(batch_size),
        "probe_attempts": probe_attempts,
        "peak_memory_bytes": peak,
        "peak_memory_fraction": float(peak / total) if total else None,
        "elapsed_sec": float(time.perf_counter() - started),
        "scaler_path": str(scaler_path),
        "shard_path": str(shard),
        "metric_shard_path": str(metric_shard),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    ddp = init_distributed(args)
    try:
        set_seed(20260720 + ddp.rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        root = Path(args.root).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Refusing to overwrite score artifact without --overwrite: {output}")
        fold = read_json(root / "data" / "folds" / f"{args.fold_id}.json")
        if ddp.is_main:
            selected_dates = _selected_dates(fold, str(args.prediction_split), str(args.start), str(args.end))
            if int(args.max_dates) > 0:
                selected_dates = selected_dates[: int(args.max_dates)]
        else:
            selected_dates = None
        selected_dates = broadcast_object(selected_dates, ddp)
        if not selected_dates:
            raise RuntimeError("rank0 did not publish selected signal dates")
        rank_dates = list(selected_dates)[ddp.rank :: ddp.world_size]
        split_path = fold["index_paths"][str(args.prediction_split)]
        rows = _read_rows(split_path, rank_dates)
        if int(args.stratified_max_rows) > 0:
            # A diagnostic cap is apportioned across global dates before the DDP
            # date partition, preserving time coverage without pretending it is a
            # full-cross-section capacity test.
            global_target = int(args.stratified_max_rows)
            global_base, global_remainder = divmod(global_target, len(selected_dates))
            local_target = sum(global_base + int(index < global_remainder) for index in range(ddp.rank, len(selected_dates), ddp.world_size))
            rows = _stratified_rows(rows, rank_dates, local_target)
        checkpoint_path, checkpoint, backbone, architecture, ae_path, ae_config, reward_config, ae, reward, input_mode = _load_models(
            args, ddp.device
        )
        checkpoint_args = dict(checkpoint.get("args") or {})
        checkpoint_horizon = int(checkpoint_args.get("rank_horizon", 7))
        requested_horizon = int(args.rank_horizon)
        rank_horizon = checkpoint_horizon if requested_horizon == 0 else requested_horizon
        if rank_horizon != checkpoint_horizon:
            raise ValueError(
                f"--rank-horizon={rank_horizon} disagrees with checkpoint rank_horizon={checkpoint_horizon}; "
                "refusing to export a mislabeled score artifact"
            )
        if rank_horizon < 1:
            raise ValueError("checkpoint rank_horizon must be positive")
        multitask = str(checkpoint.get("kind", "")) == MULTITASK_CHECKPOINT_KIND
        if multitask and str(checkpoint_args.get("label_mode", "")) != "multitask":
            raise RuntimeError("Multi-task checkpoint does not declare label_mode=multitask")
        multitask_contract = _multitask_inference_contract(checkpoint_args, rank_horizon) if multitask else None
        diagnostic_horizons = _parse_diagnostic_horizons(str(args.diagnostic_horizons))
        metric_horizons = tuple(sorted({*diagnostic_horizons, *( (rank_horizon,) if multitask else () )}))
        target_meta = read_json(WTSPaths(root).target_meta)
        future_targets_available = bool(target_meta.get("future_targets_available", True))
        target_width = int(target_meta["shape"][1])
        if not future_targets_available:
            metric_horizons = ()
        else:
            invalid_metric_horizons = [horizon for horizon in metric_horizons if int(horizon) > target_width]
            if invalid_metric_horizons:
                raise ValueError(
                    f"Metric horizons exceed target width={target_width}: {invalid_metric_horizons}"
                )
        if str(checkpoint.get("kind", "")) == H7_ABSOLUTE_FULL_PAIR_CHECKPOINT_KIND:
            trend_threshold = float(checkpoint_args.get("good_return_threshold", 0.10))
        else:
            trend_threshold = float(checkpoint_args.get("trend_threshold", 0.05))
        model_score_columns = (
            tuple(multitask_contract["model_score_columns"])
            if multitask
            else (reward_score_column(rank_horizon),)
        )
        primary_score_column = (
            str(multitask_contract["two_stage_score_column"]) if multitask else model_score_columns[0]
        )
        truth_column = true_return_column(rank_horizon)
        shard_dir = output.parent / f"{output.stem}_ddp_shards"
        if ddp.is_main:
            shard_dir.mkdir(parents=True, exist_ok=True)
        barrier(ddp)
        _run_rank(
            args,
            ddp,
            rows,
            ae,
            reward,
            input_mode,
            rank_horizon,
            model_score_columns,
            multitask,
            truth_column,
            metric_horizons,
            future_targets_available,
            output,
        )
        barrier(ddp)
        if ddp.is_main:
            summaries = []
            score_parts = []
            metric_parts = []
            metric_columns = ["signal_date", "instrument", *model_score_columns]
            if future_targets_available:
                metric_columns.append(truth_column)
            if multitask and future_targets_available:
                metric_columns.extend(
                    [
                        _path_metric_column("path_sharpe", rank_horizon),
                        _path_metric_column("max_drawdown", rank_horizon),
                    ]
                )
            for rank in range(ddp.world_size):
                shard, metric_shard, summary_path = _shard_paths(output, rank)
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                score_parts.append(pd.read_parquet(shard))
                metric_parts.append(pd.read_parquet(metric_shard, columns=metric_columns))
            score_frame = pd.concat(score_parts, ignore_index=True).sort_values(["signal_date", "instrument"]).reset_index(drop=True)
            metric_frame = pd.concat(metric_parts, ignore_index=True).sort_values(["signal_date", "instrument"]).reset_index(drop=True)
            artifact_score_columns = list(model_score_columns)
            if not future_targets_available:
                offline_metric_values: dict[str, Any] = {
                    "status": "unavailable_for_label_free_inference",
                }
            elif multitask:
                if not score_frame.loc[:, ["signal_date", "instrument"]].equals(metric_frame.loc[:, ["signal_date", "instrument"]]):
                    raise RuntimeError("Multi-task score and metric shards have mismatched keys")
                score_frame[primary_score_column] = multitask_two_stage_scores(
                    signal_dates=score_frame["signal_date"].to_numpy(),
                    return_scores=score_frame[model_score_columns[0]].to_numpy(),
                    sharpe_scores=score_frame[model_score_columns[1]].to_numpy(),
                    drawdown_scores=score_frame[model_score_columns[2]].to_numpy(),
                    return_top_n=int(multitask_contract["return_top_n"]),
                    risk_weights=tuple(multitask_contract["risk_weights"]),
                )
                metric_frame[primary_score_column] = score_frame[primary_score_column].to_numpy()
                artifact_score_columns.append(primary_score_column)
            full_cross_section = int(args.stratified_max_rows) <= 0
            if future_targets_available and multitask:
                offline_metric_values = {
                    "return_score": _capacity_metrics(
                        metric_frame,
                        score_column=model_score_columns[0],
                        truth_column=truth_column,
                        event_threshold=trend_threshold,
                    ),
                    "return_topn_risk_top5": _capacity_metrics(
                        metric_frame,
                        score_column=primary_score_column,
                        truth_column=truth_column,
                        event_threshold=trend_threshold,
                    ),
                    "sharpe_score": _capacity_metrics(
                        metric_frame,
                        score_column=model_score_columns[1],
                        truth_column=_path_metric_column("path_sharpe", rank_horizon),
                        event_threshold=0.0,
                        metric_label="path_sharpe",
                    ),
                    "drawdown_score": {
                        "contract": "signed maximum drawdown; larger is better because zero is no drawdown and negative values are worse",
                        "metrics": _capacity_metrics(
                            metric_frame,
                            score_column=model_score_columns[2],
                            truth_column=_path_metric_column("max_drawdown", rank_horizon),
                            event_threshold=0.0,
                            metric_label="signed_max_drawdown",
                        ),
                    },
                }
            elif future_targets_available:
                offline_metric_values = _capacity_metrics(
                    metric_frame,
                    score_column=primary_score_column,
                    truth_column=truth_column,
                    event_threshold=trend_threshold,
                )
            offline_metrics = {
                "kind": (
                    "wts_raw_reward_transformer_offline_metrics_v1"
                    if backbone == "raw"
                    else (
                        "wts_frozen_ae_loop_preference_offline_metrics_v1"
                        if architecture == "loop_preference"
                        else (
                            "wts_frozen_ae_multitask_reward_transformer_offline_metrics_v1"
                            if multitask
                            else "wts_frozen_ae_reward_transformer_offline_metrics_v1"
                        )
                    )
                ),
                "status": "offline_score_evaluation_only",
                "warning": (
                    "This is a time-stratified diagnostic, not a full-cross-section capacity evaluation or QuantX backtest."
                    if not full_cross_section
                    else "This is not a QuantX backtest. QuantX execution artifacts are required for formal trading results."
                ),
                "checkpoint": str(checkpoint_path),
                "reward_backbone": backbone,
                "reward_architecture": architecture,
                "ae_checkpoint": str(ae_path) if ae_path is not None else None,
                "full_selected_cross_sections": bool(full_cross_section),
                "metrics": offline_metric_values,
            }
            metrics_path = output.with_name(f"{output.stem}_full_metrics.json")
            metrics_path.write_text(json.dumps(offline_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            diagnostics_path: Path | None = None
            if diagnostic_horizons:
                horizon_metrics: dict[str, Any] = {}
                for horizon in diagnostic_horizons:
                    columns = [
                        "signal_date",
                        "instrument",
                        *model_score_columns,
                        _path_metric_column("terminal_return", horizon),
                        _path_metric_column("path_sharpe", horizon),
                        _path_metric_column("path_sortino", horizon),
                        _path_metric_column("max_drawdown", horizon),
                        _path_metric_column("path_quality", horizon),
                    ]
                    horizon_frame = pd.concat(
                        [pd.read_parquet(_shard_paths(output, rank)[1], columns=columns) for rank in range(ddp.world_size)],
                        ignore_index=True,
                    ).sort_values(["signal_date", "instrument"]).reset_index(drop=True)
                    diagnostic_score_columns = list(model_score_columns)
                    if multitask:
                        horizon_frame = horizon_frame.merge(
                            score_frame.loc[:, ["signal_date", "instrument", primary_score_column]],
                            on=["signal_date", "instrument"],
                            how="left",
                            validate="one_to_one",
                        )
                        diagnostic_score_columns.append(primary_score_column)
                    horizon_metrics[str(horizon)] = {
                        score_column: _path_quality_diagnostics(
                            horizon_frame,
                            score_column=score_column,
                            horizon=int(horizon),
                            event_threshold=trend_threshold,
                        )
                        for score_column in diagnostic_score_columns
                    }
                diagnostics = {
                    "kind": "wts_reward_transformer_path_quality_diagnostics_v1",
                    "status": "offline_score_evaluation_only",
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_label_mode": str(checkpoint_args.get("label_mode", "endpoint")),
                    "score_columns": diagnostic_score_columns,
                    "quality_definition": {
                        "formula": "0.50 * same_date_percentile(terminal_return) + 0.30 * same_date_percentile(path_sharpe) + 0.20 * same_date_percentile(signed_max_drawdown)",
                        "weights": {
                            "terminal_return": float(PATH_QUALITY_WEIGHTS[0]),
                            "path_sharpe": float(PATH_QUALITY_WEIGHTS[1]),
                            "signed_max_drawdown": float(PATH_QUALITY_WEIGHTS[2]),
                        },
                        "daily_return_contract": "adjacent log returns of close(T+1..T+H), therefore T+2 through T+H",
                    },
                    "horizons": horizon_metrics,
                }
                diagnostics_path = output.with_name(f"{output.stem}_path_quality_diagnostics.json")
                diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            model_metadata: dict[str, Any]
            if backbone == "raw":
                model_metadata = {"raw_model_config": asdict(reward_config)}
            else:
                if ae_path is None or ae_config is None:
                    raise RuntimeError("Frozen-AE inference metadata is incomplete")
                model_metadata = {
                    "ae_checkpoint": str(ae_path),
                    "ae_model_config": asdict(ae_config),
                    "reward_config": asdict(reward_config),
                }
            manifest = write_market_score_artifact(
                score_frame,
                output,
                score_columns=artifact_score_columns,
                overwrite=bool(args.overwrite),
                metadata={
                    "model_family": (
                        "raw_reward_transformer"
                        if backbone == "raw"
                        else (
                            "frozen_ae_loop_preference"
                            if architecture == "loop_preference"
                            else ("frozen_ae_multitask_reward_transformer" if multitask else "frozen_ae_reward_transformer")
                        )
                    ),
                    "checkpoint": str(checkpoint_path),
                    "reward_backbone": backbone,
                    "reward_architecture": architecture,
                    "loop_contract": (
                        {
                            "steps": int(checkpoint_args["loop_steps"]),
                            "condition_update": str(checkpoint_args["loop_condition_update"]),
                            "deep_supervision": bool(checkpoint_args.get("loop_deep_supervision", True)),
                        }
                        if architecture == "loop_preference"
                        else None
                    ),
                    "rank_horizon": int(rank_horizon),
                    "label_mode": (
                        "h7_absolute_full_pair"
                        if str(checkpoint.get("kind", "")) == H7_ABSOLUTE_FULL_PAIR_CHECKPOINT_KIND
                        else str(checkpoint_args.get("label_mode", "endpoint"))
                    ),
                    "trend_threshold": trend_threshold,
                    "fold_id": str(args.fold_id),
                    "prediction_split": str(args.prediction_split),
                    "signal_date_selection": {"start": str(args.start), "end": str(args.end), "dates": int(len(selected_dates))},
                    "row_selection": (
                        {"mode": "full_selected_cross_sections", "rows": int(len(score_frame))}
                        if full_cross_section
                        else {"mode": "signal_date_stratified_diagnostic", "requested_rows": int(args.stratified_max_rows), "rows": int(len(score_frame))}
                    ),
                    "distributed_inference": {"world_size": int(ddp.world_size), "ranks": summaries},
                    "score_contract": {
                        "column": primary_score_column,
                        "score_columns": artifact_score_columns,
                        "model_score_columns": list(model_score_columns),
                        "rank_horizon": int(rank_horizon),
                        "model_score_range": "[1e-6, 1 - 1e-6]",
                        "model_score_transform": "sigmoid(reward_logit)",
                        "two_stage_selection": (
                            {
                                "primary_score_column": primary_score_column,
                                "return_candidate_score_column": model_score_columns[0],
                                "return_top_n": int(multitask_contract["return_top_n"]),
                                "risk_score_columns": {
                                    "sharpe": model_score_columns[1],
                                    "drawdown": model_score_columns[2],
                                },
                                "risk_rank_weights": {
                                    "sharpe": float(multitask_contract["risk_weights"][0]),
                                    "drawdown": float(multitask_contract["risk_weights"][1]),
                                },
                                "contract": "per signal_date: return TopN, then weighted same-date Sharpe/drawdown rank rerank",
                            }
                            if multitask
                            else None
                        ),
                    },
                    "offline_metrics_path": str(metrics_path),
                    "path_quality_diagnostics_path": str(diagnostics_path) if diagnostics_path is not None else None,
                    "formal_status": "score_exported_quantx_not_run",
                    **model_metadata,
                },
            )
            if not bool(args.keep_shards):
                for item in summaries:
                    for key in ("shard_path", "metric_shard_path"):
                        Path(item[key]).unlink(missing_ok=True)
                for rank in range(ddp.world_size):
                    _shard_paths(output, rank)[2].unlink(missing_ok=True)
                shard_dir.rmdir()
            print(
                json.dumps({"ok": True, "output": str(output), "manifest": manifest, "offline_metrics": str(metrics_path)}, ensure_ascii=False),
                flush=True,
            )
        barrier(ddp)
        return 0
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    raise SystemExit(main())
