"""Train frozen-AE, same-date pairwise reward models.

The only supervised objective is a DPO-style pairwise logistic loss. Every
chosen/rejected pair is constructed within one signal_date. The default
contract balances dynamic endpoint-return labels; optional quality contracts
rank return, Sharpe, and drawdown, including bounded endpoint-gap experiments.
The exported public score is sigmoid(logit),
therefore in (0, 1). Besides the legacy one-pass Transformer, this entry
point supports a fixed-step loop preference model. Its only supervision is
still pairwise: every rollout state receives a Bradley-Terry loss and no
return-regression or flow-matching target is fabricated from the pair data.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .runtime import (  # noqa: E402
    DistributedContext,
    append_training_log,
    barrier,
    cleanup_distributed,
    count_parameters,
    create_tensorboard_writer,
    init_distributed,
    load_or_fit_scaler,
    make_scheduler,
    prepare_run_dir,
    unwrap_model,
)
from .execution_utility import ExecutionUtilitySameDatePairDataset  # noqa: E402
from .common import (  # noqa: E402
    CANDIDATE_FEATURE_FIELDS,
    EXPERIMENT_ROOT,
    MARKET_FEATURE_FIELDS,
    ModelConfig,
    SEQ_LEN,
    WTSPaths,
    WeakToStrongPathDataset,
    build_models,
    future_path_metrics,
    future_path_metrics_from_target_rows,
    load_model_state_flexible,
    model_config_from_size,
    path_quality_from_metrics,
    read_json,
    set_seed,
    state_dict_without_module,
    target_memmap,
    write_json,
)


DEFAULT_ROOT = EXPERIMENT_ROOT / "market_all_close2close_balanced_v2"
DEFAULT_KRONOS_ROOT = EXPERIMENT_ROOT.parent / "feature_store"
DEFAULT_AE = Path(__file__).resolve().parents[3] / "weights" / "reward_v1" / "ae_epoch020.pt"
TREND_LABELS = ("up", "range", "down")
MULTITASK_SCORE_HEADS = ("return", "sharpe", "drawdown")
MULTITASK_PAIR_CONTRACTS = ("independent_risk_pairs", "shared_directional")


def reward_score_column(rank_horizon: int) -> str:
    if int(rank_horizon) < 1:
        raise ValueError("rank_horizon must be positive")
    return f"reward_score_{int(rank_horizon)}d"


def true_return_column(rank_horizon: int) -> str:
    if int(rank_horizon) < 1:
        raise ValueError("rank_horizon must be positive")
    return f"true_return_{int(rank_horizon)}d"


def multitask_score_column(head: str, rank_horizon: int) -> str:
    if head not in MULTITASK_SCORE_HEADS:
        raise ValueError(f"Unsupported multi-task score head: {head!r}")
    if int(rank_horizon) < 1:
        raise ValueError("rank_horizon must be positive")
    return f"reward_{head}_score_{int(rank_horizon)}d"


def multitask_two_stage_score_column(rank_horizon: int) -> str:
    if int(rank_horizon) < 1:
        raise ValueError("rank_horizon must be positive")
    return f"reward_return_topn_risk_score_{int(rank_horizon)}d"


def is_multitask_label_mode(args: argparse.Namespace) -> bool:
    return str(args.label_mode) == "multitask"


def is_execution_utility_label_mode(args: argparse.Namespace) -> bool:
    """Whether pairs are ordered by fixed-horizon executable net utility."""

    return str(args.label_mode) == "residual_utility"


def multitask_pair_contract(args: argparse.Namespace) -> str:
    """Return the persisted multi-task pair contract with old-checkpoint compatibility."""
    raw_contract = getattr(args, "multitask_pair_contract", None)
    contract = "independent_risk_pairs" if raw_contract in (None, "") else str(raw_contract)
    if contract not in MULTITASK_PAIR_CONTRACTS:
        raise ValueError(
            f"Unsupported multi-task pair contract: {contract!r}; choices={list(MULTITASK_PAIR_CONTRACTS)}"
        )
    return contract


def uses_shared_directional_multitask(args: argparse.Namespace) -> bool:
    return is_multitask_label_mode(args) and multitask_pair_contract(args) == "shared_directional"


def multitask_directional_pair_targets(
    good_risk_percentiles: np.ndarray,
    bad_risk_percentiles: np.ndarray,
    min_percentile_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one return pair's independent return, Sharpe, and drawdown labels.

    The return-pair generator already establishes good return > bad return.
    Risk directions are intentionally independent: a higher-return stock can
    have either better or worse Sharpe/drawdown.  Weak percentile differences
    are masked so they cannot manufacture a preference target.
    """
    if not 0.0 < float(min_percentile_gap) <= 1.0:
        raise ValueError("min_percentile_gap must be inside (0, 1]")
    good = np.asarray(good_risk_percentiles, dtype=np.float32).reshape(-1)
    bad = np.asarray(bad_risk_percentiles, dtype=np.float32).reshape(-1)
    expected = len(MULTITASK_SCORE_HEADS) - 1
    if good.shape != (expected,) or bad.shape != (expected,):
        raise ValueError(
            f"Expected {expected} Sharpe/drawdown percentile values for each pair, got good={good.shape}, bad={bad.shape}"
        )
    directions = np.zeros(len(MULTITASK_SCORE_HEADS), dtype=np.int8)
    masks = np.zeros(len(MULTITASK_SCORE_HEADS), dtype=np.bool_)
    directions[0] = 1
    masks[0] = True
    for head_index, difference in enumerate(good - bad, start=1):
        if np.isfinite(difference) and abs(float(difference)) >= float(min_percentile_gap):
            directions[head_index] = 1 if float(difference) > 0.0 else -1
            masks[head_index] = True
    return directions, masks


def multitask_loss_weights(args: argparse.Namespace) -> tuple[float, float, float]:
    weights = (
        float(args.multitask_return_loss_weight),
        float(args.multitask_sharpe_loss_weight),
        float(args.multitask_drawdown_loss_weight),
    )
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("multi-task loss weights must be finite and strictly positive")
    return weights


def multitask_risk_weights(args: argparse.Namespace) -> tuple[float, float]:
    weights = (float(args.multitask_risk_sharpe_weight), float(args.multitask_risk_drawdown_weight))
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("multi-task risk weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("multi-task Sharpe and drawdown risk weights must sum to 1")
    return weights


def multitask_score_heads_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    return MULTITASK_SCORE_HEADS if is_multitask_label_mode(args) else ("return",)


def multitask_two_stage_scores(
    *,
    signal_dates: np.ndarray | pd.Series,
    return_scores: np.ndarray,
    sharpe_scores: np.ndarray,
    drawdown_scores: np.ndarray,
    return_top_n: int,
    risk_weights: tuple[float, float],
) -> np.ndarray:
    """Encode the deterministic return-TopN then risk-rerank selection contract.

    Candidate rows always outrank non-candidates.  Within a candidate set the
    score is the stable rank of the weighted same-date Sharpe/drawdown ranks.
    """

    if int(return_top_n) < 5:
        raise ValueError("multi-task return_top_n must be at least 5")
    sharpe_weight, drawdown_weight = (float(value) for value in risk_weights)
    if not math.isclose(sharpe_weight + drawdown_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("multi-task risk weights must sum to 1")
    dates = np.asarray(signal_dates).astype(str)
    values = [np.asarray(item, dtype=np.float64) for item in (return_scores, sharpe_scores, drawdown_scores)]
    if any(item.ndim != 1 or len(item) != len(dates) for item in values):
        raise ValueError("multi-task scores and signal_dates must be matching one-dimensional arrays")
    if not all(np.isfinite(item).all() for item in values):
        raise ValueError("multi-task scores must be finite before two-stage reranking")

    output = np.empty(len(dates), dtype=np.float32)
    positions_frame = pd.DataFrame({"signal_date": dates, "position": np.arange(len(dates), dtype=np.int64)})
    for _, group in positions_frame.groupby("signal_date", sort=False):
        positions = group["position"].to_numpy(dtype=np.int64)
        count = len(positions)
        return_order = np.argsort(-values[0][positions], kind="mergesort")
        return_rank = np.empty(count, dtype=np.int64)
        return_rank[return_order] = np.arange(count, dtype=np.int64)
        candidate_local = return_order[: min(int(return_top_n), count)]
        candidate_positions = positions[candidate_local]
        candidate_count = len(candidate_positions)
        sharpe_order = np.argsort(-values[1][candidate_positions], kind="mergesort")
        drawdown_order = np.argsort(-values[2][candidate_positions], kind="mergesort")
        sharpe_rank = np.empty(candidate_count, dtype=np.int64)
        drawdown_rank = np.empty(candidate_count, dtype=np.int64)
        sharpe_rank[sharpe_order] = np.arange(candidate_count, dtype=np.int64)
        drawdown_rank[drawdown_order] = np.arange(candidate_count, dtype=np.int64)
        denominator = max(candidate_count - 1, 1)
        risk = sharpe_weight * (1.0 - sharpe_rank / denominator) + drawdown_weight * (
            1.0 - drawdown_rank / denominator
        )
        risk_order = np.argsort(-risk, kind="mergesort")
        candidate_score = np.empty(candidate_count, dtype=np.float32)
        candidate_score[risk_order] = 1.0 - np.arange(candidate_count, dtype=np.float32) / max(candidate_count, 1)
        output[positions] = -1.0 - return_rank.astype(np.float32) / max(count, 1)
        output[candidate_positions] = candidate_score
    return output


def endpoint_trend_labels(returns: np.ndarray, threshold: float) -> np.ndarray:
    if float(threshold) <= 0.0:
        raise ValueError("trend_threshold must be positive")
    labels = np.full(len(returns), "range", dtype=object)
    labels[np.asarray(returns) >= float(threshold)] = "up"
    labels[np.asarray(returns) <= -float(threshold)] = "down"
    return labels


def quality_trend_labels(scores: np.ndarray, *, up_quantile: float, down_quantile: float) -> np.ndarray:
    if not 0.0 < float(down_quantile) < float(up_quantile) < 1.0:
        raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
    values = np.asarray(scores, dtype=np.float32)
    labels = np.full(len(values), "range", dtype=object)
    valid = np.isfinite(values)
    labels[valid & (values >= float(up_quantile))] = "up"
    labels[valid & (values <= float(down_quantile))] = "down"
    return labels


def gap_quality_weights(args: argparse.Namespace) -> tuple[float, float, float]:
    weights = (
        float(args.gap_quality_return_weight),
        float(args.gap_quality_sharpe_weight),
        float(args.gap_quality_drawdown_weight),
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("gap-quality weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("gap-quality return, Sharpe, and drawdown weights must sum to 1")
    return weights


def path_quality_weights_for_args(args: argparse.Namespace) -> tuple[float, float, float]:
    if str(args.label_mode) == "gap_quality":
        return gap_quality_weights(args)
    return (0.50, 0.30, 0.20)


def path_quality_formula(weights: tuple[float, float, float]) -> str:
    return (
        f"{weights[0]:.2f} * same_date_percentile(terminal_return) + "
        f"{weights[1]:.2f} * same_date_percentile(path_sharpe) + "
        f"{weights[2]:.2f} * same_date_percentile(signed_max_drawdown)"
    )


@dataclass(frozen=True)
class RewardTransformerConfig:
    cond_dim: int = 128
    cond_tokens: int = 27
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    score_heads: tuple[str, ...] = ("return",)


@dataclass(frozen=True)
class PairGroup:
    date: str
    date_code: int
    good_by_label: tuple[np.ndarray, np.ndarray, np.ndarray]
    raw_positions_sorted: np.ndarray
    raw_returns_sorted: np.ndarray


@dataclass(frozen=True)
class AllUpDay:
    """One date-local pool used to materialize independent good/bad pairs."""

    date: str
    date_code: int
    up_positions: np.ndarray
    up_bad_draw_counts: np.ndarray
    up_bad_draw_ends: np.ndarray
    range_positions: np.ndarray
    down_positions: np.ndarray
    raw_positions_sorted: np.ndarray
    raw_returns_sorted: np.ndarray
    raw_positions_by_position: np.ndarray
    raw_global_ranks_by_position: np.ndarray
    range_step: int
    down_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "inspect-pairs"], default="train", nargs="?")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--kronos-root", default=str(DEFAULT_KRONOS_ROOT))
    parser.add_argument("--run-name", default="market-all-pre2020-ae-reward-transformer20m")
    parser.add_argument("--fold-id", default="pre2020_eval2020_2026")
    parser.add_argument("--fold-path", default="")
    parser.add_argument("--good-train-split", default="train_balanced", help=argparse.SUPPRESS)
    parser.add_argument("--raw-train-split", default="train")
    parser.add_argument("--pair-train-split", default="train", help="Raw full-market split used for both good and bad rows.")
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--target-kind", default="abs", choices=["abs"])
    parser.add_argument("--input-mode", default="raw_relative", choices=["raw_relative", "relative_only"])
    parser.add_argument("--ae-model-size", default="ae_wide_08", choices=["ae_wide_08", "debug", "50m", "base", "large"])
    parser.add_argument("--ae-checkpoint", default=str(DEFAULT_AE))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pairs-per-batch", type=int, default=2048, help="Independent good/bad pairs per GPU step.")
    parser.add_argument("--bad-per-good", type=int, default=1, help="Independently sampled same-date worse bad stocks per good.")
    parser.add_argument(
        "--up-top1-bad-draws",
        type=int,
        default=1,
        help="Total same-date bad comparisons for the highest-return eligible up stock on each date.",
    )
    parser.add_argument(
        "--up-top10-bad-draws",
        type=int,
        default=1,
        help="Total same-date bad comparisons for eligible up ranks 2 through 10 on each date.",
    )
    parser.add_argument(
        "--head-hard-negative-top-rank",
        type=int,
        default=0,
        help="When positive, global ranks 1-10 use strictly-worse bads only from the same-date top-N (0 preserves legacy up-only fan-out).",
    )
    parser.add_argument(
        "--gap-free-top-rank",
        type=int,
        default=0,
        help="Global ranks at or above this rank use strictly-lower bads without --min-return-gap (0 preserves the legacy gap rule).",
    )
    parser.add_argument(
        "--class-balance-scope",
        default="date",
        choices=["date", "global"],
        help="Balance range/down per signal date (legacy) or globally while retaining every eligible up row.",
    )
    parser.add_argument(
        "--pair-mode",
        default="all_up",
        choices=["all_up", "tail_net"],
        help="all_up preserves the existing broad ranking contract; tail_net trains only TopK tail pairs.",
    )
    parser.add_argument(
        "--tail-positive-rank",
        type=int,
        default=20,
        help="tail_net: same-date realized-return rank ceiling for chosen stocks.",
    )
    parser.add_argument(
        "--tail-bad-min-rank",
        type=int,
        default=21,
        help="tail_net: inclusive same-date realized-return rank floor for rejected stocks.",
    )
    parser.add_argument(
        "--tail-bad-max-rank",
        type=int,
        default=200,
        help="tail_net: inclusive same-date realized-return rank ceiling for rejected stocks.",
    )
    parser.add_argument(
        "--tail-min-return",
        type=float,
        default=0.0,
        help="tail_net: optional chosen gross endpoint return floor; 0 preserves the pure rank-tail contract.",
    )
    parser.add_argument("--groups-per-batch", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--good-per-class", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--min-return-gap", type=float, default=0.005)
    parser.add_argument("--rank-horizon", type=int, default=7)
    parser.add_argument(
        "--label-mode",
        default="endpoint",
        choices=["endpoint", "path_quality", "gap_quality", "multitask", "residual_utility"],
        help=(
            "endpoint preserves the legacy terminal-return label; path_quality/gap_quality use a composite score; "
            "multitask trains separate return, Sharpe, and drawdown heads; residual_utility ranks only rows "
            "executable at the fixed entry and exit endpoints by net utility."
        ),
    )
    parser.add_argument(
        "--execution-utility-labels",
        default="",
        help=(
            "residual_utility: parquet keyed by row_id with eligible and net_utility columns, "
            "built by build_execution_utility_labels_v1.py."
        ),
    )
    parser.add_argument(
        "--min-quality-gap",
        type=float,
        default=0.15,
        help="path_quality: minimum same-date composite-quality gap between good and bad stocks.",
    )
    parser.add_argument(
        "--quality-up-quantile",
        type=float,
        default=2.0 / 3.0,
        help="path_quality: quality percentile at or above which a row is in the up bucket.",
    )
    parser.add_argument(
        "--quality-down-quantile",
        type=float,
        default=1.0 / 3.0,
        help="path_quality: quality percentile at or below which a row is in the down bucket.",
    )
    parser.add_argument(
        "--gap-return-min",
        type=float,
        default=0.0,
        help="gap_quality: inclusive absolute terminal-return-gap lower bound.",
    )
    parser.add_argument(
        "--gap-return-max",
        type=float,
        default=0.005,
        help="gap_quality: exclusive absolute terminal-return-gap upper bound.",
    )
    parser.add_argument(
        "--gap-quality-return-weight",
        type=float,
        default=0.20,
        help="gap_quality: same-date terminal-return percentile weight.",
    )
    parser.add_argument(
        "--gap-quality-sharpe-weight",
        type=float,
        default=0.50,
        help="gap_quality: same-date path-Sharpe percentile weight.",
    )
    parser.add_argument(
        "--gap-quality-drawdown-weight",
        type=float,
        default=0.30,
        help="gap_quality: same-date signed-max-drawdown percentile weight.",
    )
    parser.add_argument(
        "--gap-quality-min-difference",
        type=float,
        default=0.05,
        help="gap_quality: strict minimum composite-quality advantage for the chosen row.",
    )
    parser.add_argument(
        "--gap-quality-bad-options",
        type=int,
        default=4,
        help="gap_quality: deterministic same-date rejected alternatives retained per chosen row and rotated by epoch.",
    )
    parser.add_argument(
        "--multitask-pair-contract",
        default="independent_risk_pairs",
        choices=MULTITASK_PAIR_CONTRACTS,
        help=(
            "multitask: independent_risk_pairs preserves the legacy three-stream objective; "
            "shared_directional uses each return pair for all heads with independent Sharpe/drawdown directions."
        ),
    )
    parser.add_argument(
        "--multitask-risk-return-gap-max",
        type=float,
        default=0.005,
        help=(
            "multitask independent_risk_pairs: exclusive same-date absolute return-gap ceiling for "
            "Sharpe/drawdown pairs; unused by shared_directional."
        ),
    )
    parser.add_argument(
        "--multitask-risk-min-percentile-gap",
        type=float,
        default=0.05,
        help=(
            "multitask: strict same-date risk-percentile advantage for legacy risk pairs, or minimum absolute "
            "Sharpe/drawdown percentile difference retained as a shared_directional label."
        ),
    )
    parser.add_argument(
        "--multitask-risk-bad-options",
        type=int,
        default=4,
        help="multitask independent_risk_pairs: deterministic same-date risk alternatives retained per chosen row and rotated by epoch.",
    )
    parser.add_argument(
        "--multitask-return-loss-weight",
        type=float,
        default=1.0,
        help="multitask: relative Bradley-Terry weight for the strict-return head.",
    )
    parser.add_argument(
        "--multitask-sharpe-loss-weight",
        type=float,
        default=0.25,
        help="multitask: relative Bradley-Terry weight for the Sharpe head.",
    )
    parser.add_argument(
        "--multitask-drawdown-loss-weight",
        type=float,
        default=0.25,
        help="multitask: relative Bradley-Terry weight for the drawdown head.",
    )
    parser.add_argument(
        "--multitask-return-top-n",
        type=int,
        default=50,
        help="multitask: return-head candidate count before the Sharpe/drawdown rerank selects Top5.",
    )
    parser.add_argument(
        "--multitask-risk-sharpe-weight",
        type=float,
        default=0.50,
        help="multitask: within-TopN Sharpe-rank weight for the final risk rerank.",
    )
    parser.add_argument(
        "--multitask-risk-drawdown-weight",
        type=float,
        default=0.50,
        help="multitask: within-TopN signed-drawdown-rank weight for the final risk rerank.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.05,
        help="Absolute endpoint-return boundary for dynamic up/range/down pair labels.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument(
        "--max-open-feature-shards",
        type=int,
        default=16,
        help="Feature shard mmap cache per DataLoader worker for the all-up pair stream.",
    )
    parser.add_argument(
        "--train-shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle individual pairs. Disabled by default to preserve date-local feature I/O; labels/bads still change each epoch.",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--logit-center-weight",
        type=float,
        default=0.01,
        help="Penalty on the batch mean reward logit; removes the additive-score ambiguity of pairwise loss.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reward-d-model", type=int, default=512)
    parser.add_argument("--reward-layers", type=int, default=6)
    parser.add_argument("--reward-heads", type=int, default=8)
    parser.add_argument("--reward-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--reward-dropout", type=float, default=0.05)
    parser.add_argument(
        "--reward-backbone",
        choices=["frozen_ae", "raw"],
        default="frozen_ae",
        help="frozen_ae maps AE latents to a score; raw trains directly from normalized stock, market, and candidate features.",
    )
    parser.add_argument(
        "--reward-architecture",
        choices=["transformer", "loop_preference"],
        default="transformer",
        help="transformer preserves the legacy one-pass head; loop_preference unrolls fixed score updates.",
    )
    parser.add_argument(
        "--loop-steps",
        type=int,
        default=3,
        help="Fixed score rollout length for --reward-architecture=loop_preference.",
    )
    parser.add_argument(
        "--loop-condition-update",
        choices=["static", "loop"],
        default="loop",
        help="static resets per-block AE states each step; loop carries each block's state across score rollout steps.",
    )
    parser.add_argument(
        "--loop-deep-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply linearly increasing Bradley-Terry supervision to every rollout state; disabled means final-state-only supervision.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation micro-batches per optimizer update. One preserves the legacy behavior.",
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=0,
        help="Epoch at which cosine decay reaches --min-lr; zero means --epochs, preserving the legacy schedule.",
    )
    parser.add_argument("--min-lr", type=float, default=0.0, help="Learning-rate floor reached after cosine decay and held thereafter.")
    parser.add_argument("--max-scaler-samples", type=int, default=4096)
    parser.add_argument("--scaler-path", default="")
    parser.add_argument("--cross-sectional-eval-every", type=int, default=5)
    parser.add_argument(
        "--screen-epoch",
        type=int,
        default=0,
        help="Stop a run after this validation epoch when Top5 terminal-return excess does not exceed --screen-min-top5-excess; 0 disables screening.",
    )
    parser.add_argument(
        "--screen-min-top5-excess",
        type=float,
        default=0.0,
        help="Strict 2019 Top5 terminal-return selection-advantage threshold required at --screen-epoch.",
    )
    parser.add_argument(
        "--pair-validation-every",
        type=int,
        default=1,
        help="Run full independent-pair validation every N epochs; 0 disables it during DDP training.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-dates", type=int, default=0)
    parser.add_argument("--max-validation-dates", type=int, default=0)
    parser.add_argument("--amp-dtype", default="bf16", choices=["off", "bf16", "fp16"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ddp", default="auto", choices=["auto", "off"])
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-run-dir", default="")
    return parser.parse_args()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RewardTransformerBlock(nn.Module):
    """Bidirectional attention block backed by PyTorch Flash SDP when available."""

    def __init__(self, cfg: RewardTransformerConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError(f"d_model={cfg.d_model} must divide n_heads={cfg.n_heads}")
        self.norm1 = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        hidden = int(round(cfg.d_model * cfg.mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.n_heads = int(cfg.n_heads)
        self.head_dim = int(cfg.d_model // cfg.n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        batch, tokens, _ = h.shape
        qkv = self.qkv(h).view(batch, tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, tokens, -1)
        x = residual + self.dropout(self.proj(attention))
        return x + self.dropout(self.mlp(self.norm2(x)))


class FrozenAERewardTransformer(nn.Module):
    """Maps frozen AE latent tokens to one legacy or several task-specific preference logits."""

    def __init__(self, cfg: RewardTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.cond_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.cond_tokens + 1, cfg.d_model))
        self.blocks = nn.ModuleList([RewardTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.score_heads = tuple(str(name) for name in cfg.score_heads)
        if not self.score_heads or len(set(self.score_heads)) != len(self.score_heads):
            raise ValueError("RewardTransformerConfig.score_heads must contain unique names")
        if any(name not in MULTITASK_SCORE_HEADS for name in self.score_heads):
            raise ValueError(f"Unsupported reward score heads: {self.score_heads}")
        self.head = nn.Linear(cfg.d_model, len(self.score_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 3:
            raise ValueError(f"Expected condition [B,T,C], got {tuple(condition.shape)}")
        if condition.shape[1:] != (self.cfg.cond_tokens, self.cfg.cond_dim):
            raise ValueError(
                "AE latent shape does not match reward configuration: "
                f"condition={tuple(condition.shape[1:])}, expected={(self.cfg.cond_tokens, self.cfg.cond_dim)}"
            )
        x = self.input_proj(condition)
        cls = self.cls_token.expand(condition.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x[:, 0]))
        return logits.squeeze(-1) if len(self.score_heads) == 1 else logits

    def score(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(condition))


class FrozenAELoopPreferenceTransformer(nn.Module):
    """Fixed-step score refinement with per-block AE condition recurrence.

    ``score`` is the only public scalar state. Each Transformer block owns a
    separate AE-token state and writes it back for the next rollout iteration,
    mirroring TrajDiT's layer-aligned loop state without adding a second
    transformer or a synthetic regression target.
    """

    def __init__(self, cfg: RewardTransformerConfig, *, rollout_steps: int, condition_update: str):
        super().__init__()
        if int(rollout_steps) < 1:
            raise ValueError("rollout_steps must be positive")
        if condition_update not in {"static", "loop"}:
            raise ValueError(f"Unsupported loop condition update mode: {condition_update!r}")
        self.cfg = cfg
        self.rollout_steps = int(rollout_steps)
        self.condition_update = str(condition_update)
        self.input_proj = nn.Linear(cfg.cond_dim, cfg.d_model)
        self.context_position = nn.Parameter(torch.zeros(1, cfg.cond_tokens, cfg.d_model))
        self.score_in = nn.Linear(1, cfg.d_model)
        self.time_in = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([RewardTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.delta_head = nn.Linear(cfg.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.context_position, mean=0.0, std=0.02)
        nn.init.zeros_(self.delta_head.bias)

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim != 3:
            raise ValueError(f"Expected condition [B,T,C], got {tuple(condition.shape)}")
        if condition.shape[1:] != (self.cfg.cond_tokens, self.cfg.cond_dim):
            raise ValueError(
                "AE latent shape does not match reward configuration: "
                f"condition={tuple(condition.shape[1:])}, expected={(self.cfg.cond_tokens, self.cfg.cond_dim)}"
            )

    def _score_token(self, score: torch.Tensor, step: int) -> torch.Tensor:
        step_size = 1.0 / float(self.rollout_steps)
        time = score.new_empty((score.shape[0], 2))
        time[:, 0] = float(step) * step_size
        time[:, 1] = step_size
        return self.score_in(score.unsqueeze(-1)).unsqueeze(1) + self.time_in(time).unsqueeze(1)

    def forward_steps(self, condition: torch.Tensor) -> torch.Tensor:
        self._validate_condition(condition)
        context = self.input_proj(condition) + self.context_position
        initial_states = [context for _ in self.blocks]
        condition_states = initial_states
        score = condition.new_zeros(condition.shape[0])
        step_scores: list[torch.Tensor] = []
        step_size = 1.0 / float(self.rollout_steps)

        for step in range(self.rollout_steps):
            active_states = initial_states if self.condition_update == "static" else condition_states
            score_token = self._score_token(score, step)
            next_states: list[torch.Tensor] = []
            for block, block_condition in zip(self.blocks, active_states, strict=True):
                block_out = block(torch.cat((score_token, block_condition), dim=1))
                score_token = block_out[:, :1]
                next_states.append(block_out[:, 1:])
            delta = self.delta_head(self.norm(score_token[:, 0])).squeeze(-1)
            score = score + step_size * delta
            step_scores.append(score)
            if self.condition_update == "loop":
                condition_states = next_states
        return torch.stack(step_scores, dim=0)

    def forward(self, condition: torch.Tensor, *, return_step_logits: bool = False) -> torch.Tensor:
        step_logits = self.forward_steps(condition)
        return step_logits if return_step_logits else step_logits[-1]

    def score(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(condition))


@dataclass(frozen=True)
class RawFeatureRewardTransformerConfig:
    stock_dim: int
    market_dim: int
    candidate_dim: int
    lookback: int = 60
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.05


class RawFeatureRewardTransformer(nn.Module):
    """Scores normalized raw feature sequences directly, without an AE bottleneck."""

    def __init__(self, cfg: RawFeatureRewardTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.stock_dim + cfg.market_dim + cfg.candidate_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.lookback + 1, cfg.d_model))
        block_cfg = RewardTransformerConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        self.blocks = nn.ModuleList([RewardTransformerBlock(block_cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, stock: torch.Tensor, market: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        expected_stock = (self.cfg.lookback, self.cfg.stock_dim)
        expected_market = (self.cfg.lookback, self.cfg.market_dim)
        if stock.ndim != 3 or tuple(stock.shape[1:]) != expected_stock:
            raise ValueError(f"Expected stock [B,{expected_stock[0]},{expected_stock[1]}], got {tuple(stock.shape)}")
        if market.ndim != 3 or tuple(market.shape[1:]) != expected_market:
            raise ValueError(f"Expected market [B,{expected_market[0]},{expected_market[1]}], got {tuple(market.shape)}")
        if candidate.ndim != 2 or candidate.shape != (stock.shape[0], self.cfg.candidate_dim):
            raise ValueError(
                f"Expected candidate [B,{self.cfg.candidate_dim}], got {tuple(candidate.shape)} for batch={stock.shape[0]}"
            )
        candidate_tokens = candidate.unsqueeze(1).expand(-1, self.cfg.lookback, -1)
        x = self.input_proj(torch.cat((stock, market, candidate_tokens), dim=-1))
        cls = self.cls_token.expand(stock.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)

    def score(self, stock: torch.Tensor, market: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(stock, market, candidate))


class DateBalancedPairDataset(Dataset):
    """Fixed-date pair groups with strict same-date chosen/rejected sampling."""

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        good_rows: pd.DataFrame,
        raw_rows: pd.DataFrame,
        scaler,
        input_mode: str,
        rank_horizon: int,
        good_per_class: int,
        min_return_gap: float,
        seed: int,
    ):
        if int(rank_horizon) < 1:
            raise ValueError("--rank-horizon must be positive")
        if int(good_per_class) < 1:
            raise ValueError("--good-per-class must be positive")
        if float(min_return_gap) <= 0.0:
            raise ValueError("--min-return-gap must be positive")
        required = {"row_id", "signal_date", "trend_label", "target_row"}
        missing = required - set(good_rows.columns) | (required - set(raw_rows.columns))
        if missing:
            raise KeyError(f"Pair construction is missing required row columns: {sorted(missing)}")

        # Balancing can duplicate minority-class rows across the global split.  A
        # single cross-section must never contain duplicate chosen samples.
        original_good_rows = int(len(good_rows))
        good_rows = good_rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        raw_rows = raw_rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        self.good_base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=good_rows,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
        )
        self.raw_base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=raw_rows,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
        )
        self.good_per_class = int(good_per_class)
        self.min_return_gap = float(min_return_gap)
        self.rank_horizon_index = int(rank_horizon) - 1
        if self.rank_horizon_index >= int(self.good_base.targets.shape[1]):
            raise ValueError(
                f"--rank-horizon={rank_horizon} exceeds target horizon={self.good_base.targets.shape[1]}"
            )
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)

        good_returns = np.asarray(
            self.good_base.targets[self.good_base.target_row, self.rank_horizon_index], dtype=np.float32
        )
        raw_returns = np.asarray(
            self.raw_base.targets[self.raw_base.target_row, self.rank_horizon_index], dtype=np.float32
        )
        self._groups, group_stats = self._build_groups(good_returns, raw_returns)
        self.stats = {
            "good_rows_before_deduplicate": original_good_rows,
            "good_rows_after_deduplicate": int(len(good_rows)),
            "raw_rows_after_deduplicate": int(len(raw_rows)),
            "good_per_class": self.good_per_class,
            "group_size": self.group_size,
            "min_return_gap": self.min_return_gap,
            **group_stats,
        }
        if not self._groups:
            raise RuntimeError(
                "No eligible same-date pair groups. Check trend labels, horizon, or --good-per-class / --min-return-gap."
            )

    @property
    def group_size(self) -> int:
        return len(TREND_LABELS) * self.good_per_class

    def _build_groups(self, good_returns: np.ndarray, raw_returns: np.ndarray) -> tuple[list[PairGroup], dict[str, int]]:
        good_frame = self.good_base.rows.loc[:, ["signal_date", "trend_label"]].copy()
        good_frame["position"] = np.arange(len(good_frame), dtype=np.int64)
        good_frame["return"] = good_returns
        raw_frame = self.raw_base.rows.loc[:, ["signal_date"]].copy()
        raw_frame["position"] = np.arange(len(raw_frame), dtype=np.int64)
        raw_frame["return"] = raw_returns
        raw_by_date = {str(date): group for date, group in raw_frame.groupby("signal_date", sort=True)}
        groups: list[PairGroup] = []
        stats = {
            "dates_seen": 0,
            "dates_missing_raw_cross_section": 0,
            "dates_skipped_insufficient_good": 0,
            "dates_eligible": 0,
            "good_invalid_return": 0,
            "good_without_worse_bad": 0,
        }
        for date_code, (date, group) in enumerate(good_frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            date_text = str(date)
            raw_group = raw_by_date.get(date_text)
            if raw_group is None:
                stats["dates_missing_raw_cross_section"] += 1
                continue
            raw_valid = raw_group.loc[np.isfinite(raw_group["return"].to_numpy(dtype=np.float32))]
            if raw_valid.empty:
                stats["dates_missing_raw_cross_section"] += 1
                continue
            raw_order = np.argsort(raw_valid["return"].to_numpy(dtype=np.float32), kind="mergesort")
            raw_positions_sorted = raw_valid["position"].to_numpy(dtype=np.int64)[raw_order]
            raw_returns_sorted = raw_valid["return"].to_numpy(dtype=np.float32)[raw_order]
            label_positions: list[np.ndarray] = []
            complete = True
            for label in TREND_LABELS:
                candidates = group.loc[group["trend_label"].astype(str) == label]
                candidate_returns = candidates["return"].to_numpy(dtype=np.float32)
                valid = np.isfinite(candidate_returns)
                stats["good_invalid_return"] += int((~valid).sum())
                candidates = candidates.loc[valid]
                candidate_returns = candidate_returns[valid]
                # Prefix [0:count] is exactly the eligible uniform bad pool:
                # r_bad < r_good - min_return_gap. searchsorted(side='left')
                # preserves the requested strict inequality at the threshold.
                bad_counts = np.searchsorted(raw_returns_sorted, candidate_returns - self.min_return_gap, side="left")
                eligible = bad_counts > 0
                stats["good_without_worse_bad"] += int((~eligible).sum())
                positions = candidates["position"].to_numpy(dtype=np.int64)[eligible]
                if len(positions) < self.good_per_class:
                    complete = False
                    break
                label_positions.append(positions)
            if not complete:
                stats["dates_skipped_insufficient_good"] += 1
                continue
            groups.append(
                PairGroup(
                    date=date_text,
                    date_code=date_code,
                    good_by_label=(label_positions[0], label_positions[1], label_positions[2]),
                    raw_positions_sorted=raw_positions_sorted,
                    raw_returns_sorted=raw_returns_sorted,
                )
            )
        stats["dates_eligible"] = int(len(groups))
        return groups, stats

    def __len__(self) -> int:
        return len(self._groups)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __getitem__(self, index: int):
        group = self._groups[int(index)]
        seed = self.seed + int(self._epoch.value) * 1_000_003 + int(index) * 17_171
        generator = np.random.default_rng(seed)
        chosen_parts = [
            generator.choice(pool, size=self.good_per_class, replace=False).astype(np.int64, copy=False)
            for pool in group.good_by_label
        ]
        chosen_positions = np.concatenate(chosen_parts)
        chosen_returns = np.asarray(
            self.good_base.targets[self.good_base.target_row[chosen_positions], self.rank_horizon_index], dtype=np.float32
        )
        bad_counts = np.searchsorted(group.raw_returns_sorted, chosen_returns - self.min_return_gap, side="left")
        if not np.all(bad_counts > 0):
            raise RuntimeError(f"Pair group {group.date} lost a valid same-date bad candidate")
        bad_positions = np.asarray(
            [group.raw_positions_sorted[int(generator.integers(int(count)))] for count in bad_counts], dtype=np.int64
        )
        bad_returns = np.asarray(
            self.raw_base.targets[self.raw_base.target_row[bad_positions], self.rank_horizon_index], dtype=np.float32
        )
        if not bool(np.all(bad_returns < chosen_returns - self.min_return_gap)):
            raise RuntimeError(f"Pair group {group.date} violates bad_return < good_return - min_gap")
        if not bool(
            (self.good_base.rows.iloc[chosen_positions]["signal_date"].astype(str).to_numpy() == group.date).all()
            and (self.raw_base.rows.iloc[bad_positions]["signal_date"].astype(str).to_numpy() == group.date).all()
        ):
            raise RuntimeError(f"Pair group {group.date} attempted to cross a signal_date boundary")
        chosen_samples = [self.good_base[int(position)] for position in chosen_positions]
        bad_samples = [self.raw_base[int(position)] for position in bad_positions]
        labels = np.repeat(np.arange(len(TREND_LABELS), dtype=np.int64), self.good_per_class)
        date_codes = torch.full((self.group_size,), group.date_code, dtype=torch.int64)
        return (
            torch.stack([item[0] for item in chosen_samples]),
            torch.stack([item[1] for item in chosen_samples]),
            torch.stack([item[2] for item in chosen_samples]),
            torch.stack([item[0] for item in bad_samples]),
            torch.stack([item[1] for item in bad_samples]),
            torch.stack([item[2] for item in bad_samples]),
            torch.from_numpy(chosen_returns.copy()),
            torch.from_numpy(bad_returns.copy()),
            torch.from_numpy(labels),
            date_codes,
            date_codes.clone(),
        )


def pair_group_collate(items):
    if not items:
        raise RuntimeError("Cannot collate an empty pair batch")
    return tuple(torch.cat([item[field] for item in items], dim=0) for field in range(len(items[0])))


class AllUpSameDatePairDataset(Dataset):
    """Independent same-date pairs with all eligible daily up stocks as good.

    The legacy ``date`` class-balance scope cycles range/down within each date.
    The ``global`` scope keeps every eligible up row, then cycles same-date
    range/down endpoints globally to an exact 1:1:1 pair-label ratio.  Both
    paths retain same-date bad selection.
    """

    _MASK64 = (1 << 64) - 1

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        scaler,
        input_mode: str,
        rank_horizon: int,
        label_mode: str,
        trend_threshold: float,
        min_return_gap: float,
        min_quality_gap: float,
        quality_up_quantile: float,
        quality_down_quantile: float,
        bad_per_good: int,
        up_top1_bad_draws: int,
        up_top10_bad_draws: int,
        head_hard_negative_top_rank: int,
        gap_free_top_rank: int,
        class_balance_scope: str,
        max_open_feature_shards: int,
        seed: int,
    ):
        if int(rank_horizon) < 1:
            raise ValueError("--rank-horizon must be positive")
        if str(label_mode) not in {"endpoint", "path_quality"}:
            raise ValueError("--label-mode must be endpoint or path_quality")
        if str(label_mode) == "endpoint" and float(trend_threshold) <= 0.0:
            raise ValueError("--trend-threshold must be positive")
        if str(label_mode) == "endpoint" and float(min_return_gap) <= 0.0:
            raise ValueError("--min-return-gap must be positive")
        if str(label_mode) == "path_quality" and float(min_quality_gap) <= 0.0:
            raise ValueError("--min-quality-gap must be positive")
        if str(label_mode) == "path_quality" and not 0.0 < float(quality_down_quantile) < float(quality_up_quantile) < 1.0:
            raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
        if int(bad_per_good) < 1:
            raise ValueError("--bad-per-good must be positive")
        if int(up_top1_bad_draws) < 1 or int(up_top10_bad_draws) < 1:
            raise ValueError("--up-top1-bad-draws and --up-top10-bad-draws must be positive")
        if int(up_top1_bad_draws) < int(up_top10_bad_draws):
            raise ValueError("--up-top1-bad-draws must be greater than or equal to --up-top10-bad-draws")
        if (int(up_top1_bad_draws) > 1 or int(up_top10_bad_draws) > 1) and int(bad_per_good) != 1:
            raise ValueError("Head-up bad fan-out requires --bad-per-good=1 to keep draw counts unambiguous")
        if int(head_hard_negative_top_rank) not in (0,) and int(head_hard_negative_top_rank) < 10:
            raise ValueError("--head-hard-negative-top-rank must be 0 or at least 10")
        if int(gap_free_top_rank) < 0:
            raise ValueError("--gap-free-top-rank must be non-negative")
        if str(label_mode) == "path_quality" and (
            int(head_hard_negative_top_rank) != 0 or int(gap_free_top_rank) != 0
        ):
            raise ValueError("path_quality requires --head-hard-negative-top-rank=0 and --gap-free-top-rank=0")
        if str(class_balance_scope) not in {"date", "global"}:
            raise ValueError("--class-balance-scope must be 'date' or 'global'")
        if str(class_balance_scope) == "global" and (
            int(bad_per_good) != 1 or int(up_top1_bad_draws) != 1 or int(up_top10_bad_draws) != 1
        ):
            raise ValueError("--class-balance-scope=global requires --bad-per-good=1 and disabled head fan-out")
        if int(max_open_feature_shards) < 1:
            raise ValueError("--max-open-feature-shards must be positive")
        required = {"row_id", "signal_date", "target_row"}
        missing = required - set(rows.columns)
        if missing:
            raise KeyError(f"All-up pair construction is missing required row columns: {sorted(missing)}")

        original_rows = int(len(rows))
        rows = rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        self.base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=rows,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
            max_open_shards=int(max_open_feature_shards),
        )
        self.rank_horizon_index = int(rank_horizon) - 1
        if self.rank_horizon_index >= int(self.base.targets.shape[1]):
            raise ValueError(f"--rank-horizon={rank_horizon} exceeds target horizon={self.base.targets.shape[1]}")
        self.label_mode = str(label_mode)
        self.min_return_gap = float(min_return_gap)
        self.min_quality_gap = float(min_quality_gap)
        self.min_pair_gap = self.min_return_gap if self.label_mode == "endpoint" else self.min_quality_gap
        self.trend_threshold = float(trend_threshold)
        self.quality_up_quantile = float(quality_up_quantile)
        self.quality_down_quantile = float(quality_down_quantile)
        self.bad_per_good = int(bad_per_good)
        self.up_top1_bad_draws = int(up_top1_bad_draws)
        self.up_top10_bad_draws = int(up_top10_bad_draws)
        self.head_hard_negative_top_rank = int(head_hard_negative_top_rank)
        self.gap_free_top_rank = int(gap_free_top_rank)
        self.class_balance_scope = str(class_balance_scope)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)
        self.endpoint_returns = np.asarray(
            self.base.targets[self.base.target_row, self.rank_horizon_index], dtype=np.float32
        )
        self.quality_scores: np.ndarray | None = None
        self.pair_values = self.endpoint_returns
        if self.label_mode == "endpoint":
            self._trend_labels = endpoint_trend_labels(self.endpoint_returns, self.trend_threshold)
        else:
            path_metrics = future_path_metrics_from_target_rows(
                self.base.targets,
                self.base.target_row,
                horizon=int(rank_horizon),
            )
            quality = path_quality_from_metrics(
                signal_dates=self.base.rows["signal_date"].astype(str).to_numpy(),
                terminal_return=path_metrics["terminal_return"],
                path_sharpe=path_metrics["path_sharpe"],
                max_drawdown=path_metrics["max_drawdown"],
            )
            self.quality_scores = quality["quality_score"]
            self.pair_values = self.quality_scores
            self._trend_labels = quality_trend_labels(
                self.quality_scores,
                up_quantile=self.quality_up_quantile,
                down_quantile=self.quality_down_quantile,
            )
        self._days, global_endpoints, build_stats = self._build_days()
        if not self._days:
            raise RuntimeError("No days can form all-up same-date 1/3 pairs")
        up_pairs_per_epoch = int(sum(int(day.up_bad_draw_ends[-1]) for day in self._days))
        base_up_rows_per_epoch = int(sum(len(day.up_positions) for day in self._days))
        self._up_day_ends = np.cumsum(
            np.asarray([int(day.up_bad_draw_ends[-1]) for day in self._days], dtype=np.int64), dtype=np.int64
        )
        self._global_range_day_indices, self._global_range_positions = global_endpoints["range"]
        self._global_down_day_indices, self._global_down_positions = global_endpoints["down"]
        self._global_class_pair_count = up_pairs_per_epoch if self.class_balance_scope == "global" else 0
        if self.class_balance_scope == "global":
            if not len(self._global_range_positions) or not len(self._global_down_positions):
                raise RuntimeError("Global 1:1:1 balancing requires at least one eligible same-date range and down good")
            self._day_ends = self._up_day_ends
            pairs_per_epoch = int(3 * up_pairs_per_epoch)
            range_rows_sampled = up_pairs_per_epoch
            down_rows_sampled = up_pairs_per_epoch
        else:
            day_pair_counts = np.asarray(
                [
                    int(day.up_bad_draw_ends[-1])
                    + 2 * len(day.up_positions) * self.bad_per_good
                    for day in self._days
                ],
                dtype=np.int64,
            )
            self._day_ends = np.cumsum(day_pair_counts, dtype=np.int64)
            pairs_per_epoch = int(self._day_ends[-1])
            range_rows_sampled = base_up_rows_per_epoch
            down_rows_sampled = base_up_rows_per_epoch
        self.stats = {
            "source_rows_before_deduplicate": original_rows,
            "source_rows_after_deduplicate": int(len(rows)),
            "bad_per_good": self.bad_per_good,
            "up_top1_bad_draws": self.up_top1_bad_draws,
            "up_top10_bad_draws": self.up_top10_bad_draws,
            "up_head_bad_fanout_enabled": bool(self.up_top1_bad_draws > 1 or self.up_top10_bad_draws > 1),
            "head_hard_negative_top_rank": self.head_hard_negative_top_rank,
            "gap_free_top_rank": self.gap_free_top_rank,
            "class_balance_scope": self.class_balance_scope,
            "max_open_feature_shards": int(max_open_feature_shards),
            "rank_horizon": int(rank_horizon),
            "trend_threshold": self.trend_threshold,
            "label_mode": self.label_mode,
            "trend_label_source": "dynamic_endpoint_return" if self.label_mode == "endpoint" else "same_date_path_quality_percentile",
            "min_return_gap": self.min_return_gap,
            "min_quality_gap": self.min_quality_gap,
            "min_pair_gap": self.min_pair_gap,
            "quality_up_quantile": self.quality_up_quantile if self.label_mode == "path_quality" else None,
            "quality_down_quantile": self.quality_down_quantile if self.label_mode == "path_quality" else None,
            "days_with_pairs": int(len(self._days)),
            "eligible_up_rows_per_epoch": base_up_rows_per_epoch,
            "up_pairs_per_epoch": up_pairs_per_epoch,
            "up_extra_pairs_per_epoch": int(up_pairs_per_epoch - base_up_rows_per_epoch * self.bad_per_good),
            "range_rows_sampled_per_epoch": int(range_rows_sampled),
            "down_rows_sampled_per_epoch": int(down_rows_sampled),
            "range_good_candidates": int(len(self._global_range_positions)),
            "down_good_candidates": int(len(self._global_down_positions)),
            "pairs_per_epoch": int(pairs_per_epoch),
            "all_up_covered_each_epoch": int(build_stats["up_rows_skipped_missing_counter_class"]) == 0,
            "all_up_covered_on_trainable_dates_each_epoch": True,
            "sampling": (
                "all eligible daily up; global range/down cycle to exact 1:1:1 labels; independent same-date worse bad"
                if self.class_balance_scope == "global"
                else "all eligible daily up; equal same-date range/down cycle; independent same-date worse bad"
            ),
            **build_stats,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (int(value) + 0x9E3779B97F4A7C15) & AllUpSameDatePairDataset._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & AllUpSameDatePairDataset._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & AllUpSameDatePairDataset._MASK64
        return (value ^ (value >> 31)) & AllUpSameDatePairDataset._MASK64

    @staticmethod
    def _coprime_step(size: int, seed: int) -> int:
        if int(size) <= 1:
            return 1
        candidate = 1 + int(seed % (int(size) - 1))
        while math.gcd(candidate, int(size)) != 1:
            candidate = 1 + candidate % (int(size) - 1)
        return int(candidate)

    @staticmethod
    def _global_ranks_for_positions(
        positions: np.ndarray,
        raw_positions_by_position: np.ndarray,
        raw_global_ranks_by_position: np.ndarray,
    ) -> np.ndarray:
        lookup = np.searchsorted(raw_positions_by_position, positions)
        if np.any(lookup >= len(raw_positions_by_position)) or not np.array_equal(
            raw_positions_by_position[lookup], positions
        ):
            raise RuntimeError("A good/bad position is missing from its same-date ranked cross-section")
        return raw_global_ranks_by_position[lookup]

    def _label_positions(
        self,
        frame: pd.DataFrame,
        label: str,
        raw_returns_sorted: np.ndarray,
        raw_positions_by_position: np.ndarray,
        raw_global_ranks_by_position: np.ndarray,
        stats: dict[str, int],
    ) -> np.ndarray:
        candidates = frame.loc[frame["trend_label"].astype(str) == label]
        returns = candidates["return"].to_numpy(dtype=np.float32)
        valid = np.isfinite(returns)
        stats[f"{label}_invalid_return"] += int((~valid).sum())
        candidates = candidates.loc[valid]
        returns = returns[valid]
        positions = candidates["position"].to_numpy(dtype=np.int64)
        global_ranks = self._global_ranks_for_positions(
            positions, raw_positions_by_position, raw_global_ranks_by_position
        )
        gap_bad_counts = np.searchsorted(raw_returns_sorted, returns - self.min_pair_gap, side="left")
        strictly_lower_bad_counts = np.searchsorted(raw_returns_sorted, returns, side="left")
        if self.gap_free_top_rank > 0:
            bad_counts = np.where(global_ranks <= self.gap_free_top_rank, strictly_lower_bad_counts, gap_bad_counts)
        else:
            bad_counts = gap_bad_counts
        eligible = bad_counts > 0
        stats[f"{label}_without_worse_bad"] += int((~eligible).sum())
        return positions[eligible]

    def _global_rank(self, day: AllUpDay, position: int) -> int:
        return int(
            self._global_ranks_for_positions(
                np.asarray([position], dtype=np.int64),
                day.raw_positions_by_position,
                day.raw_global_ranks_by_position,
            )[0]
        )

    def _uses_top30_hard_pool(self, day: AllUpDay, label_id: int, position: int) -> bool:
        return bool(
            label_id == 0
            and self.head_hard_negative_top_rank > 0
            and (self.up_top1_bad_draws > 1 or self.up_top10_bad_draws > 1)
            and self._global_rank(day, position) <= 10
        )

    def _requires_min_return_gap(self, day: AllUpDay, label_id: int, position: int) -> bool:
        if self._uses_top30_hard_pool(day, label_id, position):
            return False
        return bool(self.gap_free_top_rank <= 0 or self._global_rank(day, position) > self.gap_free_top_rank)

    def _bad_positions(self, day: AllUpDay, label_id: int, position: int) -> np.ndarray:
        good_value = float(self.pair_values[position])
        global_rank = self._global_rank(day, position)
        if self._uses_top30_hard_pool(day, label_id, position):
            top_start = max(0, len(day.raw_positions_sorted) - self.head_hard_negative_top_rank)
            top_end = len(day.raw_positions_sorted) - global_rank
            candidate_positions = day.raw_positions_sorted[top_start:top_end]
            candidate_returns = day.raw_returns_sorted[top_start:top_end]
            # A rank tiebreak alone is insufficient: the rejected target must
            # still have a strictly lower realized return, only the 0.5% gap is removed.
            return candidate_positions[candidate_returns < good_value]
        bound = good_value - self.min_pair_gap if self._requires_min_return_gap(day, label_id, position) else good_value
        bad_count = int(np.searchsorted(day.raw_returns_sorted, bound, side="left"))
        return day.raw_positions_sorted[:bad_count]

    def _build_days(self) -> tuple[list[AllUpDay], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
        frame = self.base.rows.loc[:, ["signal_date"]].copy()
        frame["position"] = np.arange(len(frame), dtype=np.int64)
        frame["return"] = self.pair_values
        frame["trend_label"] = self._trend_labels
        days: list[AllUpDay] = []
        global_day_indices: dict[str, list[np.ndarray]] = {"range": [], "down": []}
        global_positions: dict[str, list[np.ndarray]] = {"range": [], "down": []}
        stats = {
            "dates_seen": 0,
            "dates_with_no_eligible_up": 0,
            "dates_skipped_no_range": 0,
            "dates_skipped_no_down": 0,
            "up_rows_skipped_missing_counter_class": 0,
            "up_invalid_return": 0,
            "range_invalid_return": 0,
            "down_invalid_return": 0,
            "up_without_worse_bad": 0,
            "range_without_worse_bad": 0,
            "down_without_worse_bad": 0,
            "range_repeat_draws_per_epoch": 0,
            "down_repeat_draws_per_epoch": 0,
            "up_bad_pool_repeat_draws_per_epoch": 0,
            "head_top30_rows_without_strictly_worse_bad": 0,
        }
        for date_code, (date, date_frame) in enumerate(frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            valid_raw = date_frame.loc[np.isfinite(date_frame["return"].to_numpy(dtype=np.float32))]
            if valid_raw.empty:
                stats["dates_with_no_eligible_up"] += 1
                continue
            order = np.argsort(valid_raw["return"].to_numpy(dtype=np.float32), kind="mergesort")
            raw_positions_sorted = valid_raw["position"].to_numpy(dtype=np.int64)[order]
            raw_returns_sorted = valid_raw["return"].to_numpy(dtype=np.float32)[order]
            raw_position_order = np.argsort(raw_positions_sorted, kind="mergesort")
            raw_positions_by_position = raw_positions_sorted[raw_position_order]
            raw_global_ranks_by_position = len(raw_positions_sorted) - raw_position_order
            up = self._label_positions(
                date_frame,
                "up",
                raw_returns_sorted,
                raw_positions_by_position,
                raw_global_ranks_by_position,
                stats,
            )
            if not len(up):
                stats["dates_with_no_eligible_up"] += 1
                continue
            ranging = self._label_positions(
                date_frame,
                "range",
                raw_returns_sorted,
                raw_positions_by_position,
                raw_global_ranks_by_position,
                stats,
            )
            if self.class_balance_scope == "date" and not len(ranging):
                stats["dates_skipped_no_range"] += 1
                stats["up_rows_skipped_missing_counter_class"] += int(len(up))
                continue
            down = self._label_positions(
                date_frame,
                "down",
                raw_returns_sorted,
                raw_positions_by_position,
                raw_global_ranks_by_position,
                stats,
            )
            if self.class_balance_scope == "date" and not len(down):
                stats["dates_skipped_no_down"] += 1
                stats["up_rows_skipped_missing_counter_class"] += int(len(up))
                continue
            head_fanout = self.up_top1_bad_draws > 1 or self.up_top10_bad_draws > 1
            if head_fanout and self.head_hard_negative_top_rank > 0:
                up_global_ranks = self._global_ranks_for_positions(
                    up, raw_positions_by_position, raw_global_ranks_by_position
                )
                head_rows = up_global_ranks <= 10
                head_pool_sizes = np.zeros(len(up), dtype=np.int64)
                for up_index in np.flatnonzero(head_rows):
                    good_rank = int(up_global_ranks[up_index])
                    top_start = max(0, len(raw_positions_sorted) - self.head_hard_negative_top_rank)
                    top_end = len(raw_positions_sorted) - good_rank
                    candidates = raw_returns_sorted[top_start:top_end]
                    head_pool_sizes[up_index] = int((candidates < self.pair_values[up[up_index]]).sum())
                usable = ~head_rows | (head_pool_sizes > 0)
                stats["head_top30_rows_without_strictly_worse_bad"] += int((~usable).sum())
                up = up[usable]
                up_global_ranks = up_global_ranks[usable]
                if not len(up):
                    stats["dates_with_no_eligible_up"] += 1
                    continue
                order = np.argsort(up_global_ranks, kind="mergesort")
                up = up[order]
                up_global_ranks = up_global_ranks[order]
                up_bad_draw_counts = np.full(len(up), self.bad_per_good, dtype=np.int64)
                up_bad_draw_counts[up_global_ranks == 1] = self.up_top1_bad_draws
                up_bad_draw_counts[(up_global_ranks >= 2) & (up_global_ranks <= 10)] = self.up_top10_bad_draws
            elif head_fanout:
                # Preserve the legacy behavior unless the explicit global Top-N
                # hard-negative pool is enabled.
                up_returns = self.pair_values[up]
                up = up[np.argsort(-up_returns, kind="mergesort")]
                up_bad_draw_counts = np.ones(len(up), dtype=np.int64)
                up_bad_draw_counts[0] = self.up_top1_bad_draws
                if len(up) > 1:
                    up_bad_draw_counts[1 : min(10, len(up))] = self.up_top10_bad_draws
            else:
                # Preserve the original scalar --bad-per-good contract exactly when
                # the rank fan-out feature is disabled.
                up_bad_draw_counts = np.full(len(up), self.bad_per_good, dtype=np.int64)
            up_bad_draw_ends = np.cumsum(up_bad_draw_counts, dtype=np.int64)
            up_count = int(len(up))
            if self.class_balance_scope == "date":
                stats["range_repeat_draws_per_epoch"] += max(0, up_count - len(ranging))
                stats["down_repeat_draws_per_epoch"] += max(0, up_count - len(down))
            date_seed = self._mix64(self.seed + int(date_code) * 0x9E3779B9)
            day = AllUpDay(
                date=str(date),
                date_code=int(date_code),
                up_positions=up,
                up_bad_draw_counts=up_bad_draw_counts,
                up_bad_draw_ends=up_bad_draw_ends,
                range_positions=ranging,
                down_positions=down,
                raw_positions_sorted=raw_positions_sorted,
                raw_returns_sorted=raw_returns_sorted,
                raw_positions_by_position=raw_positions_by_position,
                raw_global_ranks_by_position=raw_global_ranks_by_position,
                range_step=self._coprime_step(len(ranging), date_seed ^ 0xA5A5A5A5),
                down_step=self._coprime_step(len(down), date_seed ^ 0x5A5A5A5A),
            )
            up_bad_pool_sizes = np.asarray(
                [len(self._bad_positions(day, 0, int(position))) for position in day.up_positions], dtype=np.int64
            )
            stats["up_bad_pool_repeat_draws_per_epoch"] += int(
                np.maximum(day.up_bad_draw_counts - up_bad_pool_sizes, 0).sum()
            )
            days.append(day)
            if self.class_balance_scope == "global":
                current_day_index = int(len(days) - 1)
                if len(ranging):
                    global_day_indices["range"].append(np.full(len(ranging), current_day_index, dtype=np.int64))
                    global_positions["range"].append(ranging)
                if len(down):
                    global_day_indices["down"].append(np.full(len(down), current_day_index, dtype=np.int64))
                    global_positions["down"].append(down)
        endpoints = {
            label: (
                np.concatenate(global_day_indices[label]) if global_day_indices[label] else np.empty(0, dtype=np.int64),
                np.concatenate(global_positions[label]) if global_positions[label] else np.empty(0, dtype=np.int64),
            )
            for label in ("range", "down")
        }
        if self.class_balance_scope == "global":
            up_pairs = int(sum(int(day.up_bad_draw_ends[-1]) for day in days))
            stats["range_repeat_draws_per_epoch"] = max(0, up_pairs - int(len(endpoints["range"][1])))
            stats["down_repeat_draws_per_epoch"] = max(0, up_pairs - int(len(endpoints["down"][1])))
        return days, endpoints, stats

    def __len__(self) -> int:
        if self.class_balance_scope == "global":
            return int(self._global_class_pair_count * 3)
        return int(self._day_ends[-1])

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _cycle_position(self, day: AllUpDay, positions: np.ndarray, slot: int, label_id: int, step: int) -> int:
        size = int(len(positions))
        epoch_seed = self._mix64(
            self.seed + (int(self._epoch.value) + 1) * 0xD1B54A32D192ED03 + day.date_code * 0x94D049BB133111EB + label_id
        )
        start = int(epoch_seed % size)
        return int(positions[(start + int(slot) * int(step)) % size])

    def _cycle_global_endpoint(self, label_id: int, slot: int) -> tuple[AllUpDay, int]:
        if label_id == 1:
            day_indices = self._global_range_day_indices
            positions = self._global_range_positions
        elif label_id == 2:
            day_indices = self._global_down_day_indices
            positions = self._global_down_positions
        else:
            raise ValueError(f"Global class endpoint is only valid for range/down, got label_id={label_id}")
        size = int(len(positions))
        epoch_seed = self._mix64(self.seed + (int(self._epoch.value) + 1) * 0xD1B54A32D192ED03 + label_id)
        start = int(epoch_seed % size)
        # Global candidates are appended in signal-date order. A cyclic rotation
        # changes the epoch start without destroying feature-shard locality.
        endpoint_index = int((start + int(slot)) % size)
        return self._days[int(day_indices[endpoint_index])], int(positions[endpoint_index])

    def _resolve_pair(self, index: int) -> tuple[AllUpDay, int, int, int, int]:
        if self.class_balance_scope == "global":
            up_pair_count = int(self._global_class_pair_count)
            if int(index) < up_pair_count:
                day_index = int(np.searchsorted(self._up_day_ends, int(index), side="right"))
                if day_index >= len(self._days):
                    raise IndexError(index)
                day = self._days[day_index]
                previous_end = 0 if day_index == 0 else int(self._up_day_ends[day_index - 1])
                local_pair = int(index) - previous_end
                label_id = 0
                up_index = int(np.searchsorted(day.up_bad_draw_ends, local_pair, side="right"))
                previous_up_end = 0 if up_index == 0 else int(day.up_bad_draw_ends[up_index - 1])
                good_position = int(day.up_positions[up_index])
                bad_draw = int(local_pair - previous_up_end)
            elif int(index) < 2 * up_pair_count:
                label_id = 1
                bad_draw = int(index) - up_pair_count
                day, good_position = self._cycle_global_endpoint(label_id, bad_draw)
            elif int(index) < 3 * up_pair_count:
                label_id = 2
                bad_draw = int(index) - 2 * up_pair_count
                day, good_position = self._cycle_global_endpoint(label_id, bad_draw)
            else:
                raise IndexError(index)
        else:
            return self._resolve_date_balanced_pair(index)
        bad_candidates = self._bad_positions(day, label_id, good_position)
        if not len(bad_candidates):
            raise RuntimeError(f"Pair index={index} lost its prevalidated same-date worse bad candidate")
        choice_seed = self._mix64(
            self.seed
            + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9
            + int(day.date_code) * 0x94D049BB133111EB
            + int(good_position) * 0xD1B54A32D192ED03
        )
        bad_count = int(len(bad_candidates))
        offset = int(choice_seed % bad_count)
        step = self._coprime_step(bad_count, choice_seed ^ 0xA5A5A5A5)
        bad_position = int(bad_candidates[(offset + int(bad_draw) * step) % bad_count])
        return day, label_id, good_position, bad_position, bad_draw

    def _resolve_date_balanced_pair(self, index: int) -> tuple[AllUpDay, int, int, int, int]:
        day_index = int(np.searchsorted(self._day_ends, int(index), side="right"))
        if day_index >= len(self._days):
            raise IndexError(index)
        day = self._days[day_index]
        previous_end = 0 if day_index == 0 else int(self._day_ends[day_index - 1])
        local_pair = int(index) - previous_end
        up_pair_count = int(day.up_bad_draw_ends[-1])
        up_count = int(len(day.up_positions))
        if local_pair < up_pair_count:
            label_id = 0
            up_index = int(np.searchsorted(day.up_bad_draw_ends, local_pair, side="right"))
            previous_up_end = 0 if up_index == 0 else int(day.up_bad_draw_ends[up_index - 1])
            good_position = int(day.up_positions[up_index])
            bad_draw = int(local_pair - previous_up_end)
        else:
            remaining_pair = int(local_pair - up_pair_count)
            class_pair_count = up_count * self.bad_per_good
            if remaining_pair < class_pair_count:
                label_id = 1
                good_slot = remaining_pair // self.bad_per_good
                bad_draw = remaining_pair % self.bad_per_good
                good_position = self._cycle_position(day, day.range_positions, good_slot, label_id, day.range_step)
            elif remaining_pair < 2 * class_pair_count:
                label_id = 2
                good_slot = (remaining_pair - class_pair_count) // self.bad_per_good
                bad_draw = (remaining_pair - class_pair_count) % self.bad_per_good
                good_position = self._cycle_position(
                    day, day.down_positions, good_slot, label_id, day.down_step
                )
            else:
                raise RuntimeError(f"Pair index={index} exceeds its date-local range/down pair allocation")
        bad_candidates = self._bad_positions(day, label_id, good_position)
        if not len(bad_candidates):
            raise RuntimeError(f"Pair index={index} lost its prevalidated same-date worse bad candidate")
        choice_seed = self._mix64(
            self.seed
            + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9
            + int(day.date_code) * 0x94D049BB133111EB
            + int(good_position) * 0xD1B54A32D192ED03
        )
        # A date-local affine permutation gives a high-rank good distinct bads
        # until its eligible pool is exhausted, then cycles deterministically.
        bad_count = int(len(bad_candidates))
        offset = int(choice_seed % bad_count)
        step = self._coprime_step(bad_count, choice_seed ^ 0xA5A5A5A5)
        bad_position = int(bad_candidates[(offset + int(bad_draw) * step) % bad_count])
        return day, label_id, good_position, bad_position, bad_draw

    def pair_metadata(self, index: int) -> dict[str, Any]:
        day, label_id, good_position, bad_position, bad_draw = self._resolve_pair(index)
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        good_pair_value = float(self.pair_values[good_position])
        bad_pair_value = float(self.pair_values[bad_position])
        good_global_rank = self._global_rank(day, good_position)
        bad_global_rank = self._global_rank(day, bad_position)
        requires_min_return_gap = self._requires_min_return_gap(day, label_id, good_position)
        return {
            "signal_date": day.date,
            "date_code": int(day.date_code),
            "label": TREND_LABELS[label_id],
            "good_position": int(good_position),
            "bad_position": int(bad_position),
            "bad_draw": int(bad_draw),
            "good_return": good_return,
            "bad_return": bad_return,
            "good_pair_value": good_pair_value,
            "bad_pair_value": bad_pair_value,
            "quality_score_good": float(self.quality_scores[good_position]) if self.quality_scores is not None else None,
            "quality_score_bad": float(self.quality_scores[bad_position]) if self.quality_scores is not None else None,
            "good_global_rank": good_global_rank,
            "bad_global_rank": bad_global_rank,
            "uses_top30_hard_pool": self._uses_top30_hard_pool(day, label_id, good_position),
            "requires_min_return_gap": requires_min_return_gap,
            "min_pair_gap": self.min_pair_gap,
        }

    def __getitem__(self, index: int):
        day, label_id, good_position, bad_position, _ = self._resolve_pair(int(index))
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        good_pair_value = float(self.pair_values[good_position])
        bad_pair_value = float(self.pair_values[bad_position])
        requires_min_return_gap = self._requires_min_return_gap(day, label_id, good_position)
        if requires_min_return_gap and not bad_pair_value < good_pair_value - self.min_pair_gap:
            raise RuntimeError(f"Pair index={index} violates bad_pair_value < good_pair_value - min_pair_gap")
        if not requires_min_return_gap and not bad_pair_value < good_pair_value:
            raise RuntimeError(f"Pair index={index} violates bad_pair_value < good_pair_value without a minimum gap")
        good = self.base[good_position]
        bad = self.base[bad_position]
        return (
            good[0],
            good[1],
            good[2],
            bad[0],
            bad[1],
            bad[2],
            torch.tensor(good_return, dtype=torch.float32),
            torch.tensor(bad_return, dtype=torch.float32),
            torch.tensor(label_id, dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
        )


class GapBucketQualityPairDataset(Dataset):
    """Same-date bounded-return-gap pairs ordered by a weighted path-quality score."""

    _MASK64 = (1 << 64) - 1
    _GAP_BOUNDARY_EPSILON = 1.0e-7

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        scaler,
        input_mode: str,
        rank_horizon: int,
        gap_return_min: float,
        gap_return_max: float,
        quality_weights: tuple[float, float, float],
        min_quality_difference: float,
        bad_options: int,
        quality_up_quantile: float,
        quality_down_quantile: float,
        max_open_feature_shards: int,
        seed: int,
    ):
        if int(rank_horizon) < 3:
            raise ValueError("gap_quality requires --rank-horizon >= 3")
        if not 0.0 <= float(gap_return_min) < float(gap_return_max):
            raise ValueError("gap_quality requires 0 <= --gap-return-min < --gap-return-max")
        if float(min_quality_difference) <= 0.0:
            raise ValueError("--gap-quality-min-difference must be positive")
        if int(bad_options) < 1:
            raise ValueError("--gap-quality-bad-options must be positive")
        if not 0.0 < float(quality_down_quantile) < float(quality_up_quantile) < 1.0:
            raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
        if int(max_open_feature_shards) < 1:
            raise ValueError("--max-open-feature-shards must be positive")
        weights = tuple(float(value) for value in quality_weights)
        if len(weights) != 3 or any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("gap-quality weights must contain three finite non-negative values")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("gap-quality weights must sum to 1")
        required = {"row_id", "signal_date", "target_row"}
        missing = required - set(rows.columns)
        if missing:
            raise KeyError(f"Gap-quality pair construction is missing required row columns: {sorted(missing)}")

        original_rows = int(len(rows))
        rows = rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        self.base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=rows,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
            max_open_shards=int(max_open_feature_shards),
        )
        self.rank_horizon_index = int(rank_horizon) - 1
        if self.rank_horizon_index >= int(self.base.targets.shape[1]):
            raise ValueError(f"--rank-horizon={rank_horizon} exceeds target horizon={self.base.targets.shape[1]}")
        self.gap_return_min = float(gap_return_min)
        self.gap_return_max = float(gap_return_max)
        self.quality_weights = weights
        self.min_quality_difference = float(min_quality_difference)
        self.bad_option_count = int(bad_options)
        self.quality_up_quantile = float(quality_up_quantile)
        self.quality_down_quantile = float(quality_down_quantile)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)
        self.endpoint_returns = np.asarray(
            self.base.targets[self.base.target_row, self.rank_horizon_index], dtype=np.float32
        )
        path_metrics = future_path_metrics_from_target_rows(
            self.base.targets,
            self.base.target_row,
            horizon=int(rank_horizon),
        )
        quality = path_quality_from_metrics(
            signal_dates=self.base.rows["signal_date"].astype(str).to_numpy(),
            terminal_return=path_metrics["terminal_return"],
            path_sharpe=path_metrics["path_sharpe"],
            max_drawdown=path_metrics["max_drawdown"],
            weights=self.quality_weights,
        )
        self.quality_scores = quality["quality_score"]
        self._quality_labels = quality_trend_labels(
            self.quality_scores,
            up_quantile=self.quality_up_quantile,
            down_quantile=self.quality_down_quantile,
        )
        self.good_positions, self.bad_positions, self.labels, self.date_codes, build_stats = self._build_pairs()
        if not len(self.good_positions):
            raise RuntimeError("No eligible same-date bounded-gap quality pairs; relax the gap or quality difference")
        primary_gaps = np.abs(
            self.endpoint_returns[self.good_positions] - self.endpoint_returns[self.bad_positions[:, 0]]
        )
        primary_quality_gaps = self.quality_scores[self.good_positions] - self.quality_scores[self.bad_positions[:, 0]]
        self.stats = {
            "pair_mode": "bounded_gap_quality",
            "source_rows_before_deduplicate": original_rows,
            "source_rows_after_deduplicate": int(len(rows)),
            "rank_horizon": int(rank_horizon),
            "gap_return_min": self.gap_return_min,
            "gap_return_max": self.gap_return_max,
            "quality_weights": {
                "terminal_return": self.quality_weights[0],
                "path_sharpe": self.quality_weights[1],
                "max_drawdown": self.quality_weights[2],
            },
            "min_quality_difference": self.min_quality_difference,
            "bad_options_per_good": self.bad_option_count,
            "pairs_per_epoch": int(len(self.good_positions)),
            "good_class_counts": {
                label: int((self.labels == label_id).sum()) for label_id, label in enumerate(TREND_LABELS)
            },
            "primary_return_gap_mean": float(primary_gaps.mean()),
            "primary_return_gap_min": float(primary_gaps.min()),
            "primary_return_gap_max": float(primary_gaps.max()),
            "primary_quality_gap_mean": float(primary_quality_gaps.mean()),
            "sampling": "same-date bounded absolute terminal-return gap; higher weighted path quality is chosen; daily exact 1:1:1 quality-class coverage",
            **build_stats,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (int(value) + 0x9E3779B97F4A7C15) & GapBucketQualityPairDataset._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & GapBucketQualityPairDataset._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & GapBucketQualityPairDataset._MASK64
        return (value ^ (value >> 31)) & GapBucketQualityPairDataset._MASK64

    @staticmethod
    def return_gap_in_bucket(return_gap: float, *, gap_return_min: float, gap_return_max: float) -> bool:
        epsilon = GapBucketQualityPairDataset._GAP_BOUNDARY_EPSILON
        return bool(
            float(return_gap) >= max(0.0, float(gap_return_min) - epsilon)
            and float(return_gap) < float(gap_return_max) - epsilon
        )

    @staticmethod
    def sample_bad_options(
        returns: np.ndarray,
        quality: np.ndarray,
        *,
        gap_return_min: float,
        gap_return_max: float,
        min_quality_difference: float,
        bad_options: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample lower-quality alternatives inside an absolute terminal-return-gap bucket."""

        values = np.asarray(returns, dtype=np.float32)
        scores = np.asarray(quality, dtype=np.float32)
        if values.ndim != 1 or scores.ndim != 1 or len(values) != len(scores):
            raise ValueError("returns and quality must be matching one-dimensional arrays")
        if not 0.0 <= float(gap_return_min) < float(gap_return_max):
            raise ValueError("gap bounds must satisfy 0 <= min < max")
        if float(min_quality_difference) <= 0.0 or int(bad_options) < 1:
            raise ValueError("quality difference and bad option count must be positive")
        valid = np.isfinite(values) & np.isfinite(scores)
        valid_positions = np.flatnonzero(valid)
        if len(valid_positions) < 2:
            return np.empty(0, dtype=np.int64), np.empty((0, int(bad_options)), dtype=np.int64)
        valid_returns = values[valid_positions]
        valid_scores = scores[valid_positions]
        order = np.argsort(valid_returns, kind="mergesort")
        sorted_returns = valid_returns[order]
        epsilon = GapBucketQualityPairDataset._GAP_BOUNDARY_EPSILON
        effective_min = max(0.0, float(gap_return_min) - epsilon)
        effective_max = float(gap_return_max) - epsilon
        lower_start = np.searchsorted(sorted_returns, valid_returns - effective_max, side="right")
        lower_stop_side = "left" if effective_min == 0.0 else "right"
        lower_stop = np.searchsorted(sorted_returns, valid_returns - effective_min, side=lower_stop_side)
        upper_start = np.searchsorted(sorted_returns, valid_returns + effective_min, side="left")
        upper_stop = np.searchsorted(sorted_returns, valid_returns + effective_max, side="left")
        lower_count = np.maximum(lower_stop - lower_start, 0)
        upper_count = np.maximum(upper_stop - upper_start, 0)
        total_count = lower_count + upper_count
        candidate_rows = np.flatnonzero(total_count > 0)
        if not len(candidate_rows):
            return np.empty(0, dtype=np.int64), np.empty((0, int(bad_options)), dtype=np.int64)

        draw_count = max(64, int(bad_options) * 24)
        generator = np.random.default_rng(int(seed))
        offsets = (generator.random((len(candidate_rows), draw_count)) * total_count[candidate_rows, None]).astype(np.int64)
        lower_offsets = offsets < lower_count[candidate_rows, None]
        sorted_candidate_indices = np.where(
            lower_offsets,
            lower_start[candidate_rows, None] + offsets,
            upper_start[candidate_rows, None] + offsets - lower_count[candidate_rows, None],
        )
        candidate_rows_local = order[sorted_candidate_indices]
        valid_choice = valid_scores[candidate_rows_local] < (
            valid_scores[candidate_rows, None] - float(min_quality_difference)
        )
        has_choice = valid_choice.any(axis=1)
        if not bool(has_choice.any()):
            return np.empty(0, dtype=np.int64), np.empty((0, int(bad_options)), dtype=np.int64)
        candidate_rows = candidate_rows[has_choice]
        candidate_rows_local = candidate_rows_local[has_choice]
        valid_choice = valid_choice[has_choice]
        first_choice = valid_choice.argmax(axis=1)
        options = np.repeat(
            candidate_rows_local[np.arange(len(candidate_rows)), first_choice, None], int(bad_options), axis=1
        )
        valid_order = np.cumsum(valid_choice, axis=1)
        for option_index in range(int(bad_options)):
            matches = valid_choice & (valid_order == option_index + 1)
            has_option = matches.any(axis=1)
            if bool(has_option.any()):
                options[has_option, option_index] = candidate_rows_local[
                    np.flatnonzero(has_option), matches[has_option].argmax(axis=1)
                ]
        return valid_positions[candidate_rows], valid_positions[options]

    def _build_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        frame = self.base.rows.loc[:, ["signal_date"]].copy()
        frame["position"] = np.arange(len(frame), dtype=np.int64)
        good_parts: list[np.ndarray] = []
        bad_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        date_code_parts: list[np.ndarray] = []
        stats = {
            "dates_seen": 0,
            "dates_without_valid_path_metrics": 0,
            "dates_without_lower_quality_candidates": 0,
            "dates_skipped_missing_quality_class": 0,
            "anchors_with_lower_quality_candidates": 0,
        }
        for date_code, (_, group) in enumerate(frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            positions = group["position"].to_numpy(dtype=np.int64)
            valid = np.isfinite(self.endpoint_returns[positions]) & np.isfinite(self.quality_scores[positions])
            positions = positions[valid]
            if len(positions) < 2:
                stats["dates_without_valid_path_metrics"] += 1
                continue
            anchors, bad_options = self.sample_bad_options(
                self.endpoint_returns[positions],
                self.quality_scores[positions],
                gap_return_min=self.gap_return_min,
                gap_return_max=self.gap_return_max,
                min_quality_difference=self.min_quality_difference,
                bad_options=self.bad_option_count,
                seed=self._mix64(self.seed + int(date_code) * 0x9E3779B9),
            )
            if not len(anchors):
                stats["dates_without_lower_quality_candidates"] += 1
                continue
            anchors = positions[anchors]
            bad_options = positions[bad_options]
            stats["anchors_with_lower_quality_candidates"] += int(len(anchors))
            labels = self._quality_labels[anchors]
            class_indices = [np.flatnonzero(labels == label) for label in TREND_LABELS]
            class_count = min((len(indices) for indices in class_indices), default=0)
            if class_count < 1:
                stats["dates_skipped_missing_quality_class"] += 1
                continue
            generator = np.random.default_rng(self._mix64(self.seed + int(date_code) * 0xD1B54A32D192ED03))
            for label_id, candidates in enumerate(class_indices):
                selected = candidates
                if len(selected) > class_count:
                    selected = np.sort(generator.choice(selected, size=class_count, replace=False))
                good_parts.append(anchors[selected])
                bad_parts.append(bad_options[selected])
                label_parts.append(np.full(class_count, label_id, dtype=np.int64))
                date_code_parts.append(np.full(class_count, date_code, dtype=np.int64))
        if not good_parts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.bad_option_count), dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                stats,
            )
        return (
            np.concatenate(good_parts),
            np.concatenate(bad_parts),
            np.concatenate(label_parts),
            np.concatenate(date_code_parts),
            stats,
        )

    def __len__(self) -> int:
        return int(len(self.good_positions))

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _bad_position(self, index: int) -> tuple[int, int]:
        option = int(
            self._mix64(self.seed + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9 + int(index))
            % self.bad_option_count
        )
        return int(self.bad_positions[int(index), option]), option

    def pair_metadata(self, index: int) -> dict[str, Any]:
        index = int(index)
        good_position = int(self.good_positions[index])
        bad_position, option = self._bad_position(index)
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        return {
            "signal_date": str(self.base.rows.iloc[good_position]["signal_date"]),
            "date_code": int(self.date_codes[index]),
            "label": TREND_LABELS[int(self.labels[index])],
            "good_position": good_position,
            "bad_position": bad_position,
            "bad_option": option,
            "good_return": good_return,
            "bad_return": bad_return,
            "return_gap": abs(good_return - bad_return),
            "quality_score_good": float(self.quality_scores[good_position]),
            "quality_score_bad": float(self.quality_scores[bad_position]),
            "quality_gap": float(self.quality_scores[good_position] - self.quality_scores[bad_position]),
        }

    def __getitem__(self, index: int):
        index = int(index)
        good_position = int(self.good_positions[index])
        bad_position, _ = self._bad_position(index)
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        return_gap = abs(good_return - bad_return)
        quality_gap = float(self.quality_scores[good_position] - self.quality_scores[bad_position])
        if not self.return_gap_in_bucket(
            return_gap, gap_return_min=self.gap_return_min, gap_return_max=self.gap_return_max
        ):
            raise RuntimeError(f"Gap-quality pair index={index} violates the configured terminal-return bucket")
        if not quality_gap >= self.min_quality_difference:
            raise RuntimeError(f"Gap-quality pair index={index} violates the configured quality difference")
        good = self.base[good_position]
        bad = self.base[bad_position]
        date_code = int(self.date_codes[index])
        return (
            good[0],
            good[1],
            good[2],
            bad[0],
            bad[1],
            bad[2],
            torch.tensor(good_return, dtype=torch.float32),
            torch.tensor(bad_return, dtype=torch.float32),
            torch.tensor(int(self.labels[index]), dtype=torch.int64),
            torch.tensor(date_code, dtype=torch.int64),
            torch.tensor(date_code, dtype=torch.int64),
        )


@dataclass(frozen=True)
class MultiTaskRiskPairIndex:
    """Lightweight pair indices for one conditional risk-ranking task."""

    head: str
    metric_values: np.ndarray
    good_positions: np.ndarray
    bad_positions: np.ndarray
    labels: np.ndarray
    date_codes: np.ndarray
    stats: dict[str, Any]


class MultiTaskSameDatePairDataset(Dataset):
    """Build legacy independent or shared-directional multi-task pair labels.

    ``independent_risk_pairs`` preserves the existing three-stream objective.
    ``shared_directional`` emits one established return pair and independently
    supervises every output head from that same pair.
    """

    _MASK64 = (1 << 64) - 1

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        scaler,
        input_mode: str,
        rank_horizon: int,
        trend_threshold: float,
        min_return_gap: float,
        quality_up_quantile: float,
        quality_down_quantile: float,
        bad_per_good: int,
        up_top1_bad_draws: int,
        up_top10_bad_draws: int,
        head_hard_negative_top_rank: int,
        gap_free_top_rank: int,
        class_balance_scope: str,
        risk_return_gap_max: float,
        risk_min_percentile_gap: float,
        risk_bad_options: int,
        max_open_feature_shards: int,
        seed: int,
        pair_contract: str = "independent_risk_pairs",
    ):
        if int(rank_horizon) < 3:
            raise ValueError("multi-task pairs require --rank-horizon >= 3")
        if str(pair_contract) not in MULTITASK_PAIR_CONTRACTS:
            raise ValueError(
                f"Unsupported multi-task pair contract: {pair_contract!r}; choices={list(MULTITASK_PAIR_CONTRACTS)}"
            )
        if not 0.0 < float(risk_min_percentile_gap) <= 1.0:
            raise ValueError("--multitask-risk-min-percentile-gap must be inside (0, 1]")
        if str(pair_contract) == "independent_risk_pairs":
            if not 0.0 < float(risk_return_gap_max):
                raise ValueError("--multitask-risk-return-gap-max must be positive")
            if int(risk_bad_options) < 1:
                raise ValueError("--multitask-risk-bad-options must be positive")

        self.return_dataset = AllUpSameDatePairDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=rows,
            scaler=scaler,
            input_mode=input_mode,
            rank_horizon=int(rank_horizon),
            label_mode="endpoint",
            trend_threshold=float(trend_threshold),
            min_return_gap=float(min_return_gap),
            min_quality_gap=0.15,
            quality_up_quantile=float(quality_up_quantile),
            quality_down_quantile=float(quality_down_quantile),
            bad_per_good=int(bad_per_good),
            up_top1_bad_draws=int(up_top1_bad_draws),
            up_top10_bad_draws=int(up_top10_bad_draws),
            head_hard_negative_top_rank=int(head_hard_negative_top_rank),
            gap_free_top_rank=int(gap_free_top_rank),
            class_balance_scope=str(class_balance_scope),
            max_open_feature_shards=int(max_open_feature_shards),
            seed=int(seed),
        )
        self.base = self.return_dataset.base
        self.rank_horizon = int(rank_horizon)
        self.pair_contract = str(pair_contract)
        self.risk_return_gap_max = float(risk_return_gap_max)
        self.risk_min_percentile_gap = float(risk_min_percentile_gap)
        self.risk_bad_option_count = int(risk_bad_options)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)
        self.endpoint_returns = np.asarray(self.return_dataset.endpoint_returns, dtype=np.float32)
        path_metrics = future_path_metrics_from_target_rows(
            self.base.targets,
            self.base.target_row,
            horizon=self.rank_horizon,
        )
        component_ranks = path_quality_from_metrics(
            signal_dates=self.base.rows["signal_date"].astype(str).to_numpy(),
            terminal_return=path_metrics["terminal_return"],
            path_sharpe=path_metrics["path_sharpe"],
            max_drawdown=path_metrics["max_drawdown"],
            weights=(1.0, 0.0, 0.0),
        )
        self.risk_metric_percentiles = np.column_stack(
            (component_ranks["sharpe_percentile"], component_ranks["drawdown_percentile"])
        ).astype(np.float32, copy=False)
        if self.pair_contract == "independent_risk_pairs":
            self.risk_tasks = (
                self._build_risk_task("sharpe", self.risk_metric_percentiles[:, 0]),
                self._build_risk_task("drawdown", self.risk_metric_percentiles[:, 1]),
            )
            self._task_lengths = (len(self.return_dataset), *(len(task.good_positions) for task in self.risk_tasks))
            if any(length < 1 for length in self._task_lengths):
                raise RuntimeError("multi-task construction produced an empty return, Sharpe, or drawdown task")
            self.pairs_per_task_per_epoch = int(max(self._task_lengths))
            self.stats = {
                "pair_mode": "multitask_return_then_risk",
                "rank_horizon": self.rank_horizon,
                "heads": list(MULTITASK_SCORE_HEADS),
                "pairs_per_epoch": int(len(self)),
                "pairs_per_task_per_epoch": self.pairs_per_task_per_epoch,
                "return": {
                    "pairs_available": int(len(self.return_dataset)),
                    "contract": "same-date strict endpoint return pair; bad_return < good_return - min_return_gap",
                    "dataset": self.return_dataset.stats,
                },
                "sharpe": self._risk_task_stats(self.risk_tasks[0]),
                "drawdown": self._risk_task_stats(self.risk_tasks[1]),
                "risk_pair_contract": {
                    "same_date": True,
                    "absolute_return_gap_min": 0.0,
                    "absolute_return_gap_max_exclusive": self.risk_return_gap_max,
                    "minimum_metric_percentile_gap": self.risk_min_percentile_gap,
                    "bad_options_per_good": self.risk_bad_option_count,
                    "daily_class_balance": "exact 1:1:1 by task metric percentile",
                },
            }
        else:
            self.risk_tasks = ()
            self._task_lengths = (len(self.return_dataset),)
            self.pairs_per_task_per_epoch = int(len(self.return_dataset))
            self.stats = {
                "pair_mode": "multitask_shared_directional",
                "pair_contract": self.pair_contract,
                "rank_horizon": self.rank_horizon,
                "heads": list(MULTITASK_SCORE_HEADS),
                "pairs_per_epoch": int(len(self)),
                "pairs_per_task_per_epoch": self.pairs_per_task_per_epoch,
                "return": {
                    "pairs_available": int(len(self.return_dataset)),
                    "contract": "same-date endpoint return pair reused by all three heads",
                    "dataset": self.return_dataset.stats,
                },
                "shared_directional_contract": {
                    "same_date": True,
                    "return_direction": "good_return > bad_return",
                    "risk_directions": "independent sign of good-minus-bad daily percentile for Sharpe and signed max drawdown",
                    "minimum_absolute_risk_percentile_gap": self.risk_min_percentile_gap,
                    "risk_return_gap_max": None,
                    "risk_bad_options": None,
                    "masked_risk_labels": "absolute percentile difference below the configured minimum or non-finite",
                },
            }

    @staticmethod
    def _risk_task_stats(task: MultiTaskRiskPairIndex) -> dict[str, Any]:
        return {
            "pairs_available": int(len(task.good_positions)),
            "good_class_counts": {label: int((task.labels == index).sum()) for index, label in enumerate(TREND_LABELS)},
            **task.stats,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (int(value) + 0x9E3779B97F4A7C15) & MultiTaskSameDatePairDataset._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MultiTaskSameDatePairDataset._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MultiTaskSameDatePairDataset._MASK64
        return (value ^ (value >> 31)) & MultiTaskSameDatePairDataset._MASK64

    def _build_risk_task(self, head: str, metric_values: np.ndarray) -> MultiTaskRiskPairIndex:
        if head not in {"sharpe", "drawdown"}:
            raise ValueError(f"Unsupported multi-task risk head: {head!r}")
        values = np.asarray(metric_values, dtype=np.float32)
        if values.shape != self.endpoint_returns.shape:
            raise ValueError("risk metric values must align with endpoint returns")
        frame = self.base.rows.loc[:, ["signal_date"]].copy()
        frame["position"] = np.arange(len(frame), dtype=np.int64)
        good_parts: list[np.ndarray] = []
        bad_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        date_code_parts: list[np.ndarray] = []
        stats = {
            "head": head,
            "dates_seen": 0,
            "dates_without_valid_metrics": 0,
            "dates_without_metric_candidates": 0,
            "dates_skipped_missing_quality_class": 0,
            "anchors_with_metric_candidates": 0,
        }
        labels_all = quality_trend_labels(
            values,
            up_quantile=float(self.return_dataset.quality_up_quantile),
            down_quantile=float(self.return_dataset.quality_down_quantile),
        )
        for date_code, (_, group) in enumerate(frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            positions = group["position"].to_numpy(dtype=np.int64)
            valid = np.isfinite(self.endpoint_returns[positions]) & np.isfinite(values[positions])
            positions = positions[valid]
            if len(positions) < 2:
                stats["dates_without_valid_metrics"] += 1
                continue
            anchors, bad_options = GapBucketQualityPairDataset.sample_bad_options(
                self.endpoint_returns[positions],
                values[positions],
                gap_return_min=0.0,
                gap_return_max=self.risk_return_gap_max,
                min_quality_difference=self.risk_min_percentile_gap,
                bad_options=self.risk_bad_option_count,
                seed=self._mix64(self.seed + int(date_code) * 0xD1B54A32D192ED03 + (17 if head == "sharpe" else 31)),
            )
            if not len(anchors):
                stats["dates_without_metric_candidates"] += 1
                continue
            anchors = positions[anchors]
            bad_options = positions[bad_options]
            stats["anchors_with_metric_candidates"] += int(len(anchors))
            labels = labels_all[anchors]
            class_indices = [np.flatnonzero(labels == label) for label in TREND_LABELS]
            class_count = min((len(indices) for indices in class_indices), default=0)
            if class_count < 1:
                stats["dates_skipped_missing_quality_class"] += 1
                continue
            generator = np.random.default_rng(self._mix64(self.seed + int(date_code) * 0x94D049BB133111EB))
            for label_id, candidates in enumerate(class_indices):
                selected = candidates
                if len(selected) > class_count:
                    selected = np.sort(generator.choice(selected, size=class_count, replace=False))
                good_parts.append(anchors[selected])
                bad_parts.append(bad_options[selected])
                label_parts.append(np.full(class_count, label_id, dtype=np.int64))
                date_code_parts.append(np.full(class_count, date_code, dtype=np.int64))
        if not good_parts:
            return MultiTaskRiskPairIndex(
                head=head,
                metric_values=values,
                good_positions=np.empty(0, dtype=np.int64),
                bad_positions=np.empty((0, self.risk_bad_option_count), dtype=np.int64),
                labels=np.empty(0, dtype=np.int64),
                date_codes=np.empty(0, dtype=np.int64),
                stats=stats,
            )
        return MultiTaskRiskPairIndex(
            head=head,
            metric_values=values,
            good_positions=np.concatenate(good_parts),
            bad_positions=np.concatenate(bad_parts),
            labels=np.concatenate(label_parts),
            date_codes=np.concatenate(date_code_parts),
            stats=stats,
        )

    def __len__(self) -> int:
        if self.pair_contract == "shared_directional":
            return int(len(self.return_dataset))
        return int(len(MULTITASK_SCORE_HEADS) * self.pairs_per_task_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.return_dataset.set_epoch(epoch)
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _source_index(self, index: int, task_id: int) -> int:
        length = int(self._task_lengths[int(task_id)])
        cycle = int(index) // len(MULTITASK_SCORE_HEADS)
        offset = int(
            self._mix64(self.seed + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9 + int(task_id)) % length
        )
        return int((cycle + offset) % length)

    def _risk_bad_position(self, task: MultiTaskRiskPairIndex, source_index: int) -> tuple[int, int]:
        option = int(
            self._mix64(self.seed + (int(self._epoch.value) + 1) * 0x94D049BB133111EB + int(source_index))
            % self.risk_bad_option_count
        )
        return int(task.bad_positions[int(source_index), option]), option

    def _shared_directional_targets(self, good_position: int, bad_position: int) -> tuple[np.ndarray, np.ndarray]:
        return multitask_directional_pair_targets(
            self.risk_metric_percentiles[int(good_position)],
            self.risk_metric_percentiles[int(bad_position)],
            self.risk_min_percentile_gap,
        )

    def pair_metadata(self, index: int) -> dict[str, Any]:
        if self.pair_contract == "shared_directional":
            metadata = dict(self.return_dataset.pair_metadata(int(index)))
            good_position = int(metadata["good_position"])
            bad_position = int(metadata["bad_position"])
            directions, masks = self._shared_directional_targets(good_position, bad_position)
            metadata.update(
                {
                    "task": "shared_directional",
                    "pair_contract": self.pair_contract,
                    "head_directions": {
                        head: int(directions[head_index]) for head_index, head in enumerate(MULTITASK_SCORE_HEADS)
                    },
                    "head_masks": {
                        head: bool(masks[head_index]) for head_index, head in enumerate(MULTITASK_SCORE_HEADS)
                    },
                    "good_risk_percentiles": {
                        "sharpe": float(self.risk_metric_percentiles[good_position, 0]),
                        "drawdown": float(self.risk_metric_percentiles[good_position, 1]),
                    },
                    "bad_risk_percentiles": {
                        "sharpe": float(self.risk_metric_percentiles[bad_position, 0]),
                        "drawdown": float(self.risk_metric_percentiles[bad_position, 1]),
                    },
                }
            )
            return metadata
        task_id = int(index) % len(MULTITASK_SCORE_HEADS)
        source_index = self._source_index(int(index), task_id)
        head = MULTITASK_SCORE_HEADS[task_id]
        if task_id == 0:
            metadata = dict(self.return_dataset.pair_metadata(source_index))
            metadata["task"] = head
            metadata["task_id"] = task_id
            return metadata
        task = self.risk_tasks[task_id - 1]
        good_position = int(task.good_positions[source_index])
        bad_position, option = self._risk_bad_position(task, source_index)
        return {
            "task": head,
            "task_id": task_id,
            "signal_date": str(self.base.rows.iloc[good_position]["signal_date"]),
            "date_code": int(task.date_codes[source_index]),
            "label": TREND_LABELS[int(task.labels[source_index])],
            "good_position": good_position,
            "bad_position": bad_position,
            "bad_option": option,
            "good_return": float(self.endpoint_returns[good_position]),
            "bad_return": float(self.endpoint_returns[bad_position]),
            "return_gap": abs(float(self.endpoint_returns[good_position] - self.endpoint_returns[bad_position])),
            "good_metric_percentile": float(task.metric_values[good_position]),
            "bad_metric_percentile": float(task.metric_values[bad_position]),
            "metric_percentile_gap": float(task.metric_values[good_position] - task.metric_values[bad_position]),
        }

    def __getitem__(self, index: int):
        if self.pair_contract == "shared_directional":
            day, label_id, good_position, bad_position, _ = self.return_dataset._resolve_pair(int(index))
            good_return = float(self.endpoint_returns[good_position])
            bad_return = float(self.endpoint_returns[bad_position])
            if not good_return > bad_return:
                raise RuntimeError("shared-directional return pair lost its strict endpoint-return ordering")
            directions, masks = self._shared_directional_targets(good_position, bad_position)
            good = self.base[good_position]
            bad = self.base[bad_position]
            return (
                good[0],
                good[1],
                good[2],
                bad[0],
                bad[1],
                bad[2],
                torch.tensor(good_return, dtype=torch.float32),
                torch.tensor(bad_return, dtype=torch.float32),
                torch.tensor(label_id, dtype=torch.int64),
                torch.tensor(day.date_code, dtype=torch.int64),
                torch.tensor(day.date_code, dtype=torch.int64),
                torch.from_numpy(directions.copy()),
                torch.from_numpy(masks.copy()),
            )
        task_id = int(index) % len(MULTITASK_SCORE_HEADS)
        source_index = self._source_index(int(index), task_id)
        if task_id == 0:
            return (*self.return_dataset[source_index], torch.tensor(task_id, dtype=torch.int64))
        task = self.risk_tasks[task_id - 1]
        good_position = int(task.good_positions[source_index])
        bad_position, _ = self._risk_bad_position(task, source_index)
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        return_gap = abs(good_return - bad_return)
        metric_gap = float(task.metric_values[good_position] - task.metric_values[bad_position])
        if not 0.0 <= return_gap < self.risk_return_gap_max:
            raise RuntimeError(f"multi-task {task.head} pair violates the configured return similarity band")
        if not metric_gap >= self.risk_min_percentile_gap:
            raise RuntimeError(f"multi-task {task.head} pair violates the metric percentile gap")
        good = self.base[good_position]
        bad = self.base[bad_position]
        date_code = int(task.date_codes[source_index])
        return (
            good[0],
            good[1],
            good[2],
            bad[0],
            bad[1],
            bad[2],
            torch.tensor(good_return, dtype=torch.float32),
            torch.tensor(bad_return, dtype=torch.float32),
            torch.tensor(int(task.labels[source_index]), dtype=torch.int64),
            torch.tensor(date_code, dtype=torch.int64),
            torch.tensor(date_code, dtype=torch.int64),
            torch.tensor(task_id, dtype=torch.int64),
        )


@dataclass(frozen=True)
class TailNetDay:
    """Same-date TopK chosen rows and their eligible near-tail rejected pools."""

    date: str
    date_code: int
    good_positions: np.ndarray
    bad_positions_by_good: tuple[np.ndarray, ...]
    raw_positions_by_position: np.ndarray
    raw_global_ranks_by_position: np.ndarray


class CostAwareTailPairDataset(Dataset):
    """Pair TopK labels with lower-ranked same-date alternatives.

    The broad all-up objective intentionally learns a full cross-sectional
    ordering. This dataset is a separate experimental contract for the entry
    tail. An optional chosen-return floor is retained for controlled ablations;
    rejected rows always come from a bounded rank band below the chosen tail.
    """

    _MASK64 = (1 << 64) - 1

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        scaler,
        input_mode: str,
        rank_horizon: int,
        min_return_gap: float,
        bad_per_good: int,
        tail_positive_rank: int,
        tail_bad_min_rank: int,
        tail_bad_max_rank: int,
        tail_min_return: float,
        max_open_feature_shards: int,
        seed: int,
    ):
        if int(rank_horizon) < 1:
            raise ValueError("--rank-horizon must be positive")
        if float(min_return_gap) <= 0.0:
            raise ValueError("--min-return-gap must be positive")
        if int(bad_per_good) < 1:
            raise ValueError("--bad-per-good must be positive")
        if int(tail_positive_rank) < 1:
            raise ValueError("--tail-positive-rank must be positive")
        if int(tail_bad_min_rank) <= int(tail_positive_rank):
            raise ValueError("--tail-bad-min-rank must be greater than --tail-positive-rank")
        if int(tail_bad_max_rank) < int(tail_bad_min_rank):
            raise ValueError("--tail-bad-max-rank must be at least --tail-bad-min-rank")
        if float(tail_min_return) < 0.0:
            raise ValueError("--tail-min-return must be non-negative")
        if int(max_open_feature_shards) < 1:
            raise ValueError("--max-open-feature-shards must be positive")
        required = {"row_id", "signal_date", "target_row"}
        missing = required - set(rows.columns)
        if missing:
            raise KeyError(f"Tail pair construction is missing required row columns: {sorted(missing)}")

        original_rows = int(len(rows))
        rows = rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
        self.base = WeakToStrongPathDataset(
            kronos_root=kronos_root,
            experiment_root=experiment_root,
            rows=rows,
            scaler=scaler,
            target_kind="abs",
            input_mode=input_mode,
            max_open_shards=int(max_open_feature_shards),
        )
        self.rank_horizon_index = int(rank_horizon) - 1
        if self.rank_horizon_index >= int(self.base.targets.shape[1]):
            raise ValueError(f"--rank-horizon={rank_horizon} exceeds target horizon={self.base.targets.shape[1]}")
        self.min_return_gap = float(min_return_gap)
        self.bad_per_good = int(bad_per_good)
        self.tail_positive_rank = int(tail_positive_rank)
        self.tail_bad_min_rank = int(tail_bad_min_rank)
        self.tail_bad_max_rank = int(tail_bad_max_rank)
        self.tail_min_return = float(tail_min_return)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0)
        self.endpoint_returns = np.asarray(
            self.base.targets[self.base.target_row, self.rank_horizon_index], dtype=np.float32
        )
        self._days, build_stats = self._build_days()
        if not self._days:
            raise RuntimeError("No eligible same-date cost-aware tail pairs; relax the tail rank, return floor, or gap")
        self._day_ends = np.cumsum(
            np.asarray([len(day.good_positions) * self.bad_per_good for day in self._days], dtype=np.int64), dtype=np.int64
        )
        self.stats = {
            "pair_mode": "tail_net",
            "source_rows_before_deduplicate": original_rows,
            "source_rows_after_deduplicate": int(len(rows)),
            "rank_horizon": int(rank_horizon),
            "min_return_gap": self.min_return_gap,
            "tail_positive_rank": self.tail_positive_rank,
            "tail_bad_min_rank": self.tail_bad_min_rank,
            "tail_bad_max_rank": self.tail_bad_max_rank,
            "tail_min_return": self.tail_min_return,
            "bad_per_good": self.bad_per_good,
            "days_with_pairs": int(len(self._days)),
            "eligible_good_rows_per_epoch": int(sum(len(day.good_positions) for day in self._days)),
            "pairs_per_epoch": int(self._day_ends[-1]),
            "sampling": "same-date top-rank chosen versus lower-ranked bounded-tail rejected",
            **build_stats,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (int(value) + 0x9E3779B97F4A7C15) & CostAwareTailPairDataset._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & CostAwareTailPairDataset._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & CostAwareTailPairDataset._MASK64
        return (value ^ (value >> 31)) & CostAwareTailPairDataset._MASK64

    @staticmethod
    def _coprime_step(size: int, seed: int) -> int:
        if int(size) <= 1:
            return 1
        candidate = 1 + int(seed % (int(size) - 1))
        while math.gcd(candidate, int(size)) != 1:
            candidate = 1 + candidate % (int(size) - 1)
        return int(candidate)

    @staticmethod
    def _global_rank(day: TailNetDay, position: int) -> int:
        lookup = np.searchsorted(day.raw_positions_by_position, int(position))
        if lookup >= len(day.raw_positions_by_position) or int(day.raw_positions_by_position[lookup]) != int(position):
            raise RuntimeError("A tail pair position is missing from its same-date cross-section")
        return int(day.raw_global_ranks_by_position[lookup])

    def _build_days(self) -> tuple[list[TailNetDay], dict[str, int]]:
        frame = self.base.rows.loc[:, ["signal_date"]].copy()
        frame["position"] = np.arange(len(frame), dtype=np.int64)
        frame["return"] = self.endpoint_returns
        days: list[TailNetDay] = []
        stats = {
            "dates_seen": 0,
            "dates_without_finite_return": 0,
            "tail_rows_in_positive_rank_band": 0,
            "tail_rows_below_return_floor": 0,
            "tail_rows_without_rank_band_bad": 0,
            "tail_rows_without_gap_bad": 0,
        }
        for date_code, (date, group) in enumerate(frame.groupby("signal_date", sort=True)):
            stats["dates_seen"] += 1
            positions = group["position"].to_numpy(dtype=np.int64)
            returns = group["return"].to_numpy(dtype=np.float32)
            valid = np.isfinite(returns)
            if not bool(valid.any()):
                stats["dates_without_finite_return"] += 1
                continue
            valid_positions = positions[valid]
            valid_returns = returns[valid]
            ascending = np.argsort(valid_returns, kind="mergesort")
            raw_positions_sorted = valid_positions[ascending]
            raw_returns_sorted = valid_returns[ascending]
            raw_position_order = np.argsort(raw_positions_sorted, kind="mergesort")
            raw_positions_by_position = raw_positions_sorted[raw_position_order]
            raw_global_ranks_by_position = len(raw_positions_sorted) - raw_position_order
            descending_positions = raw_positions_sorted[::-1]
            descending_returns = raw_returns_sorted[::-1]
            candidate_good_positions = descending_positions[: self.tail_positive_rank]
            candidate_good_returns = descending_returns[: self.tail_positive_rank]
            stats["tail_rows_in_positive_rank_band"] += int(len(candidate_good_positions))
            good_floor_mask = candidate_good_returns >= self.tail_min_return
            stats["tail_rows_below_return_floor"] += int((~good_floor_mask).sum())
            candidate_good_positions = candidate_good_positions[good_floor_mask]
            candidate_good_returns = candidate_good_returns[good_floor_mask]

            bad_start = self.tail_bad_min_rank - 1
            bad_end = min(self.tail_bad_max_rank, len(descending_positions))
            rank_band_positions = descending_positions[bad_start:bad_end]
            rank_band_returns = descending_returns[bad_start:bad_end]
            if not len(rank_band_positions):
                stats["tail_rows_without_rank_band_bad"] += int(len(candidate_good_positions))
                continue
            eligible_good_positions: list[int] = []
            bad_positions_by_good: list[np.ndarray] = []
            for good_position, good_return in zip(candidate_good_positions, candidate_good_returns):
                bad_pool = rank_band_positions[rank_band_returns < good_return - self.min_return_gap]
                if not len(bad_pool):
                    stats["tail_rows_without_gap_bad"] += 1
                    continue
                eligible_good_positions.append(int(good_position))
                bad_positions_by_good.append(np.asarray(bad_pool, dtype=np.int64))
            if not eligible_good_positions:
                continue
            days.append(
                TailNetDay(
                    date=str(date),
                    date_code=int(date_code),
                    good_positions=np.asarray(eligible_good_positions, dtype=np.int64),
                    bad_positions_by_good=tuple(bad_positions_by_good),
                    raw_positions_by_position=raw_positions_by_position,
                    raw_global_ranks_by_position=raw_global_ranks_by_position,
                )
            )
        return days, stats

    def __len__(self) -> int:
        return int(self._day_ends[-1])

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _resolve_pair(self, index: int) -> tuple[TailNetDay, int, int, int]:
        day_index = int(np.searchsorted(self._day_ends, int(index), side="right"))
        if day_index >= len(self._days):
            raise IndexError(index)
        day = self._days[day_index]
        previous_end = 0 if day_index == 0 else int(self._day_ends[day_index - 1])
        local_pair = int(index) - previous_end
        good_slot = local_pair // self.bad_per_good
        bad_draw = local_pair % self.bad_per_good
        good_position = int(day.good_positions[good_slot])
        bad_pool = day.bad_positions_by_good[good_slot]
        choice_seed = self._mix64(
            self.seed
            + (int(self._epoch.value) + 1) * 0xBF58476D1CE4E5B9
            + int(day.date_code) * 0x94D049BB133111EB
            + int(good_position) * 0xD1B54A32D192ED03
        )
        offset = int(choice_seed % len(bad_pool))
        step = self._coprime_step(len(bad_pool), choice_seed ^ 0xA5A5A5A5)
        bad_position = int(bad_pool[(offset + int(bad_draw) * step) % len(bad_pool)])
        return day, good_position, bad_position, int(bad_draw)

    def pair_metadata(self, index: int) -> dict[str, Any]:
        day, good_position, bad_position, bad_draw = self._resolve_pair(index)
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        return {
            "signal_date": day.date,
            "date_code": int(day.date_code),
            "label": "tail_net",
            "good_position": good_position,
            "bad_position": bad_position,
            "bad_draw": bad_draw,
            "good_return": good_return,
            "bad_return": bad_return,
            "good_global_rank": self._global_rank(day, good_position),
            "bad_global_rank": self._global_rank(day, bad_position),
            "eligible_bad_pool": int(len(day.bad_positions_by_good[np.where(day.good_positions == good_position)[0][0]])),
            "requires_min_return_gap": True,
        }

    def __getitem__(self, index: int):
        day, good_position, bad_position, _ = self._resolve_pair(int(index))
        good_return = float(self.endpoint_returns[good_position])
        bad_return = float(self.endpoint_returns[bad_position])
        if good_return < self.tail_min_return:
            raise RuntimeError(f"Tail pair index={index} violates the chosen return floor")
        if not bad_return < good_return - self.min_return_gap:
            raise RuntimeError(f"Tail pair index={index} violates r_bad < r_good - min_return_gap")
        good = self.base[good_position]
        bad = self.base[bad_position]
        return (
            good[0],
            good[1],
            good[2],
            bad[0],
            bad[1],
            bad[2],
            torch.tensor(good_return, dtype=torch.float32),
            torch.tensor(bad_return, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
            torch.tensor(day.date_code, dtype=torch.int64),
        )


def build_same_date_pair_dataset(
    args: argparse.Namespace,
    *,
    root: Path,
    rows: pd.DataFrame,
    scaler,
    bad_per_good: int,
    up_top1_bad_draws: int,
    up_top10_bad_draws: int,
    head_hard_negative_top_rank: int,
    gap_free_top_rank: int,
    class_balance_scope: str,
    seed: int,
) -> Dataset:
    """Construct the selected pair contract without changing model or loss APIs."""
    if is_execution_utility_label_mode(args):
        labels_path = Path(str(args.execution_utility_labels)).expanduser().resolve()
        if not str(args.execution_utility_labels):
            raise ValueError("--label-mode=residual_utility requires --execution-utility-labels")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing execution utility label artifact: {labels_path}")
        labels = pd.read_parquet(labels_path)
        return ExecutionUtilitySameDatePairDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=rows,
            labels=labels,
            scaler=scaler,
            input_mode=args.input_mode,
            trend_threshold=float(args.trend_threshold),
            min_utility_gap=float(args.min_return_gap),
            bad_per_good=int(bad_per_good),
            max_open_feature_shards=int(args.max_open_feature_shards),
            seed=int(seed),
        )
    if is_multitask_label_mode(args):
        return MultiTaskSameDatePairDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=rows,
            scaler=scaler,
            input_mode=args.input_mode,
            rank_horizon=int(args.rank_horizon),
            trend_threshold=float(args.trend_threshold),
            min_return_gap=float(args.min_return_gap),
            quality_up_quantile=float(args.quality_up_quantile),
            quality_down_quantile=float(args.quality_down_quantile),
            bad_per_good=int(bad_per_good),
            up_top1_bad_draws=int(up_top1_bad_draws),
            up_top10_bad_draws=int(up_top10_bad_draws),
            head_hard_negative_top_rank=int(head_hard_negative_top_rank),
            gap_free_top_rank=int(gap_free_top_rank),
            class_balance_scope=str(class_balance_scope),
            risk_return_gap_max=float(args.multitask_risk_return_gap_max),
            risk_min_percentile_gap=float(args.multitask_risk_min_percentile_gap),
            risk_bad_options=int(args.multitask_risk_bad_options),
            max_open_feature_shards=int(args.max_open_feature_shards),
            seed=int(seed),
            pair_contract=multitask_pair_contract(args),
        )
    if str(args.label_mode) == "gap_quality":
        return GapBucketQualityPairDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=rows,
            scaler=scaler,
            input_mode=args.input_mode,
            rank_horizon=int(args.rank_horizon),
            gap_return_min=float(args.gap_return_min),
            gap_return_max=float(args.gap_return_max),
            quality_weights=gap_quality_weights(args),
            min_quality_difference=float(args.gap_quality_min_difference),
            bad_options=int(args.gap_quality_bad_options),
            quality_up_quantile=float(args.quality_up_quantile),
            quality_down_quantile=float(args.quality_down_quantile),
            max_open_feature_shards=int(args.max_open_feature_shards),
            seed=int(seed),
        )
    if str(args.pair_mode) == "tail_net":
        return CostAwareTailPairDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=rows,
            scaler=scaler,
            input_mode=args.input_mode,
            rank_horizon=int(args.rank_horizon),
            min_return_gap=float(args.min_return_gap),
            bad_per_good=int(bad_per_good),
            tail_positive_rank=int(args.tail_positive_rank),
            tail_bad_min_rank=int(args.tail_bad_min_rank),
            tail_bad_max_rank=int(args.tail_bad_max_rank),
            tail_min_return=float(args.tail_min_return),
            max_open_feature_shards=int(args.max_open_feature_shards),
            seed=int(seed),
        )
    return AllUpSameDatePairDataset(
        kronos_root=args.kronos_root,
        experiment_root=root,
        rows=rows,
        scaler=scaler,
        input_mode=args.input_mode,
        rank_horizon=int(args.rank_horizon),
        label_mode=str(args.label_mode),
        trend_threshold=float(args.trend_threshold),
        min_return_gap=float(args.min_return_gap),
        min_quality_gap=float(args.min_quality_gap),
        quality_up_quantile=float(args.quality_up_quantile),
        quality_down_quantile=float(args.quality_down_quantile),
        bad_per_good=int(bad_per_good),
        up_top1_bad_draws=int(up_top1_bad_draws),
        up_top10_bad_draws=int(up_top10_bad_draws),
        head_hard_negative_top_rank=int(head_hard_negative_top_rank),
        gap_free_top_rank=int(gap_free_top_rank),
        class_balance_scope=str(class_balance_scope),
        max_open_feature_shards=int(args.max_open_feature_shards),
        seed=int(seed),
    )


def autocast_context(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp_dtype == "off":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _limit_dates(rows: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if int(maximum) <= 0 or rows.empty:
        return rows
    dates = np.sort(rows["signal_date"].drop_duplicates().astype(str).to_numpy())[: int(maximum)]
    return rows.loc[rows["signal_date"].astype(str).isin(set(dates))].copy()


def read_split_rows(paths: WTSPaths, split: dict, name: str, max_dates: int = 0) -> pd.DataFrame:
    index_paths = dict(split.get("index_paths") or {})
    if name not in index_paths:
        raise KeyError(f"Split {name!r} is unavailable; choices={sorted(index_paths)}")
    return _limit_dates(pd.read_parquet(index_paths[name]), max_dates).reset_index(drop=True)


def validate_target_contract(paths: WTSPaths, rank_horizon: int) -> None:
    meta = read_json(paths.target_meta)
    if str(meta.get("target_price_mode")) != "close_to_close" or int(meta.get("target_start_offset", -1)) != 1:
        raise RuntimeError(
            "Reward pair labels require close(T+1..T+30) / close(T+1) - 1 targets; "
            f"received target_price_mode={meta.get('target_price_mode')!r}, "
            f"target_start_offset={meta.get('target_start_offset')!r}"
        )
    target = target_memmap(paths, "abs", mode="r")
    if int(rank_horizon) > int(target.shape[1]):
        raise ValueError(f"--rank-horizon={rank_horizon} exceeds target horizon={target.shape[1]}")


def load_frozen_ae(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, ModelConfig, Path]:
    model_cfg = model_config_from_size(str(args.ae_model_size), objective="flow")
    stock_dim = 42 if args.input_mode == "relative_only" else 52
    ae, _ = build_models(model_cfg, stock_dim=stock_dim, market_dim=len(MARKET_FEATURE_FIELDS))
    checkpoint_path = Path(args.ae_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing frozen AE checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ae_state") if isinstance(checkpoint, dict) else None
    if state is None and isinstance(checkpoint, dict):
        state = checkpoint.get("model")
    if state is None:
        raise RuntimeError(f"AE checkpoint does not contain ae_state or model: {checkpoint_path}")
    load_model_state_flexible(ae, state)
    ae = ae.to(device).eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    return ae, model_cfg, checkpoint_path


def _forward_reward(
    ae: nn.Module | None,
    model: nn.Module,
    x_ts: torch.Tensor,
    x_market: torch.Tensor,
    x_candidate: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    return_step_logits: bool = False,
) -> torch.Tensor:
    with autocast_context(args, device):
        if ae is None:
            if return_step_logits:
                raise ValueError("Step logits are only available for frozen-AE loop_preference models")
            return model(x_ts, x_market, x_candidate)
        with torch.no_grad():
            condition = ae.encode(x_ts, x_market, x_candidate)
        if return_step_logits:
            return model(condition, return_step_logits=True)
        return model(condition)


def _move_pair_batch(batch, device: torch.device):
    good_dates, bad_dates = batch[9], batch[10]
    if not torch.equal(good_dates, bad_dates):
        raise RuntimeError("Pair batch contains a cross-date chosen/rejected pair")
    if len(batch) not in (11, 12, 13):
        raise RuntimeError(f"Unsupported pair batch field count: {len(batch)}")
    fields = [value.to(device, dtype=torch.float32, non_blocking=True) for value in batch[:8]]
    labels = batch[8].to(device, dtype=torch.long, non_blocking=True)
    task_ids = batch[11].to(device, dtype=torch.long, non_blocking=True) if len(batch) == 12 else None
    directions = batch[11].to(device, dtype=torch.float32, non_blocking=True) if len(batch) == 13 else None
    masks = batch[12].to(device, dtype=torch.bool, non_blocking=True) if len(batch) == 13 else None
    return (*fields, labels, task_ids, directions, masks)


def pair_logits(
    ae,
    model,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    *,
    return_step_logits: bool = False,
):
    (
        good_ts,
        good_market,
        good_candidate,
        bad_ts,
        bad_market,
        bad_candidate,
        good_return,
        bad_return,
        labels,
        task_ids,
        directions,
        masks,
    ) = _move_pair_batch(batch, device)
    count = int(good_ts.shape[0])
    x_ts = torch.cat((good_ts, bad_ts), dim=0)
    x_market = torch.cat((good_market, bad_market), dim=0)
    x_candidate = torch.cat((good_candidate, bad_candidate), dim=0)
    logits = _forward_reward(
        ae,
        model,
        x_ts,
        x_market,
        x_candidate,
        args,
        device,
        return_step_logits=return_step_logits,
    )
    if return_step_logits:
        return logits[:, :count], logits[:, count:], good_return, bad_return, labels, task_ids, directions, masks
    return logits[:count], logits[count:], good_return, bad_return, labels, task_ids, directions, masks


def pairwise_loss(
    good_logits: torch.Tensor,
    bad_logits: torch.Tensor,
    temperature: float,
    logit_center_weight: float,
) -> torch.Tensor:
    """Bradley-Terry loss with a small anchor for its unidentifiable common offset."""
    good_logits = good_logits.float()
    bad_logits = bad_logits.float()
    rank_loss = F.softplus(-(good_logits - bad_logits) / float(temperature)).mean()
    center_loss = torch.cat((good_logits, bad_logits), dim=0).mean().square()
    return rank_loss + float(logit_center_weight) * center_loss


def multitask_pairwise_loss(
    good_logits: torch.Tensor,
    bad_logits: torch.Tensor,
    task_ids: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply each task's pairwise loss only to its own output head."""

    if task_ids is None:
        raise RuntimeError("multi-task training requires task IDs in every pair batch")
    if good_logits.ndim != 2 or bad_logits.shape != good_logits.shape:
        raise ValueError(
            "multi-task logits must have matching [batch, heads] shapes, "
            f"got good={tuple(good_logits.shape)}, bad={tuple(bad_logits.shape)}"
        )
    if good_logits.shape[1] != len(MULTITASK_SCORE_HEADS):
        raise ValueError(f"Expected {len(MULTITASK_SCORE_HEADS)} multi-task heads, got {good_logits.shape[1]}")
    if task_ids.ndim != 1 or len(task_ids) != len(good_logits):
        raise ValueError("multi-task IDs must be a one-dimensional vector aligned with the pair batch")
    if bool(((task_ids < 0) | (task_ids >= len(MULTITASK_SCORE_HEADS))).any()):
        raise ValueError("multi-task batch contains an unsupported task ID")
    selected_good = good_logits.gather(1, task_ids.unsqueeze(1)).squeeze(1).float()
    selected_bad = bad_logits.gather(1, task_ids.unsqueeze(1)).squeeze(1).float()
    task_weights = torch.tensor(multitask_loss_weights(args), dtype=selected_good.dtype, device=selected_good.device)
    sample_weights = task_weights[task_ids]
    rank_losses = F.softplus(-(selected_good - selected_bad) / float(args.temperature))
    rank_loss = (rank_losses * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-12)
    center_mean = ((selected_good + selected_bad) * sample_weights).sum() / (2.0 * sample_weights.sum().clamp_min(1.0e-12))
    return rank_loss + float(args.logit_center_weight) * center_mean.square(), selected_good, selected_bad


def shared_directional_multitask_pairwise_loss(
    good_logits: torch.Tensor,
    bad_logits: torch.Tensor,
    directions: torch.Tensor | None,
    masks: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply independent directional Bradley-Terry targets to every valid head.

    Each head is normalized by its own number of valid pair labels.  The
    return weight is not globally renormalized, so auxiliary risk loss cannot
    dilute the configured return-head gradient.
    """
    if directions is None or masks is None:
        raise RuntimeError("shared-directional multi-task training requires directions and masks in every pair batch")
    if good_logits.ndim != 2 or bad_logits.shape != good_logits.shape:
        raise ValueError(
            "multi-task logits must have matching [batch, heads] shapes, "
            f"got good={tuple(good_logits.shape)}, bad={tuple(bad_logits.shape)}"
        )
    if good_logits.shape[1] != len(MULTITASK_SCORE_HEADS):
        raise ValueError(f"Expected {len(MULTITASK_SCORE_HEADS)} multi-task heads, got {good_logits.shape[1]}")
    if directions.shape != good_logits.shape or masks.shape != good_logits.shape:
        raise ValueError(
            "shared-directional targets must match [batch, heads] logits, "
            f"got directions={tuple(directions.shape)}, masks={tuple(masks.shape)}, logits={tuple(good_logits.shape)}"
        )
    good_logits = good_logits.float()
    bad_logits = bad_logits.float()
    valid = masks.to(dtype=torch.bool)
    direction = directions.to(dtype=good_logits.dtype)
    signed_margins = direction * (good_logits - bad_logits)
    valid_weight = valid.to(dtype=good_logits.dtype)
    valid_count = valid_weight.sum(dim=0)
    denominator = valid_count.clamp_min(1.0)
    rank_loss_by_head = (F.softplus(-signed_margins / float(args.temperature)) * valid_weight).sum(dim=0) / denominator
    center_mean_by_head = (((good_logits + bad_logits) * 0.5 * valid_weight).sum(dim=0) / denominator)
    head_loss = rank_loss_by_head + float(args.logit_center_weight) * center_mean_by_head.square()
    head_active = (valid_count > 0.0).to(dtype=good_logits.dtype)
    task_weights = torch.tensor(multitask_loss_weights(args), dtype=good_logits.dtype, device=good_logits.device)
    return (head_loss * task_weights * head_active).sum(), signed_margins, valid


def _shared_directional_metric_totals(
    signed_margins: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = masks.to(dtype=torch.bool)
    margin = signed_margins.detach().to(dtype=torch.float64)
    counts = valid.sum(dim=0, dtype=torch.int64).to(dtype=torch.float64)
    correct = ((margin > 0.0) & valid).sum(dim=0, dtype=torch.int64).to(dtype=torch.float64)
    margin_sum = (margin * valid.to(dtype=margin.dtype)).sum(dim=0)
    return counts, correct, margin_sum


def attach_shared_directional_pair_metrics(
    metrics: dict[str, float | int],
    valid_counts: torch.Tensor,
    correct_counts: torch.Tensor,
    margin_sums: torch.Tensor,
    ddp: DistributedContext,
) -> dict[str, float | int]:
    values = torch.cat((valid_counts, correct_counts, margin_sums)).to(device=ddp.device, dtype=torch.float64)
    if ddp.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    head_count = len(MULTITASK_SCORE_HEADS)
    total_pairs = max(int(metrics["pair_count"]), 1)
    for head_index, head in enumerate(MULTITASK_SCORE_HEADS):
        count = float(values[head_index].item())
        correct = float(values[head_count + head_index].item())
        margin_sum = float(values[2 * head_count + head_index].item())
        metrics[f"{head}_supervised_pair_count"] = int(count)
        metrics[f"{head}_supervision_fraction"] = float(count / total_pairs)
        metrics[f"{head}_pair_accuracy"] = float(correct / count) if count else 0.0
        metrics[f"{head}_signed_margin_mean"] = float(margin_sum / count) if count else 0.0
    return metrics


def loop_pairwise_weights(steps: int, deep_supervision: bool, *, device=None, dtype=None) -> torch.Tensor:
    """Return normalized, final-step-heavy rollout supervision weights."""
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if not deep_supervision:
        weights = torch.zeros(int(steps), device=device, dtype=dtype)
        weights[-1] = 1.0
        return weights
    weights = torch.arange(1, int(steps) + 1, device=device, dtype=dtype)
    return weights / weights.sum()


def deep_pairwise_loss(
    good_step_logits: torch.Tensor,
    bad_step_logits: torch.Tensor,
    temperature: float,
    logit_center_weight: float,
    deep_supervision: bool,
) -> torch.Tensor:
    """Bradley-Terry supervision at each fixed rollout state.

    This is deliberately deep supervision rather than flow matching: preference
    pairs contain no per-sample continuous target velocity.
    """
    if good_step_logits.ndim != 2 or bad_step_logits.shape != good_step_logits.shape:
        raise ValueError(
            "Expected matching [steps,batch] good/bad rollout logits, got "
            f"good={tuple(good_step_logits.shape)}, bad={tuple(bad_step_logits.shape)}"
        )
    weights = loop_pairwise_weights(
        int(good_step_logits.shape[0]),
        deep_supervision,
        device=good_step_logits.device,
        dtype=good_step_logits.dtype,
    )
    losses = torch.stack(
        [
            pairwise_loss(good_step_logits[step], bad_step_logits[step], temperature, logit_center_weight)
            for step in range(int(good_step_logits.shape[0]))
        ]
    )
    return (losses * weights).sum()


def reduce_pair_metrics(
    loss_sum: float,
    pairs: int,
    correct: int,
    gap_sum: float,
    chosen_score_sum: float,
    rejected_score_sum: float,
    elapsed: float,
    ddp: DistributedContext,
    optimizer,
) -> dict[str, float | int]:
    values = torch.tensor(
        [loss_sum, float(pairs), float(correct), gap_sum, chosen_score_sum, rejected_score_sum, elapsed],
        dtype=torch.float64,
        device=ddp.device,
    )
    if ddp.enabled:
        dist.all_reduce(values[:6], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[6:], op=dist.ReduceOp.MAX)
    loss_sum, pair_count, correct, gap_sum, chosen_sum, rejected_sum, elapsed = (float(value) for value in values.cpu().tolist())
    return {
        "loss": float(loss_sum / pair_count) if pair_count else float("nan"),
        "pair_count": int(pair_count),
        "pair_accuracy": float(correct / pair_count) if pair_count else 0.0,
        "mean_return_gap": float(gap_sum / pair_count) if pair_count else 0.0,
        "chosen_score_mean": float(chosen_sum / pair_count) if pair_count else 0.0,
        "rejected_score_mean": float(rejected_sum / pair_count) if pair_count else 0.0,
        "pairs_per_second": float(pair_count / max(elapsed, 1e-9)),
        "lr_end": float(optimizer.param_groups[0]["lr"]),
    }


def attach_rollout_pair_metrics(
    metrics: dict[str, float | int],
    step_correct: list[int],
    step_margin_sum: list[float],
    ddp: DistributedContext,
) -> dict[str, float | int]:
    if not step_correct:
        return metrics
    values = torch.tensor(
        [*map(float, step_correct), *step_margin_sum], dtype=torch.float64, device=ddp.device
    )
    if ddp.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(int(metrics["pair_count"]), 1)
    steps = len(step_correct)
    for step in range(steps):
        metrics[f"step{step + 1}_pair_accuracy"] = float(values[step].item() / count)
        metrics[f"step{step + 1}_margin_mean"] = float(values[steps + step].item() / count)
    return metrics


def train_one_epoch(ae, model, loader, optimizer, scheduler, args, ddp, epoch: int, writer) -> dict[str, float | int]:
    if ae is not None:
        ae.eval()
    model.train()
    loss_sum = 0.0
    pairs = 0
    correct = 0
    gap_sum = 0.0
    chosen_score_sum = 0.0
    rejected_score_sum = 0.0
    uses_loop = str(args.reward_architecture) == "loop_preference"
    uses_multitask = is_multitask_label_mode(args)
    uses_shared_directional = uses_shared_directional_multitask(args)
    step_correct = [0 for _ in range(int(args.loop_steps))] if uses_loop else []
    step_margin_sum = [0.0 for _ in range(int(args.loop_steps))] if uses_loop else []
    shared_valid_counts = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=ddp.device)
    shared_correct_counts = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=ddp.device)
    shared_margin_sums = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=ddp.device)
    started = time.perf_counter()
    accumulation = int(args.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        group_start = ((step - 1) // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        is_update = step % accumulation == 0 or step == len(loader)
        sync_context = nullcontext() if is_update or not ddp.enabled else model.no_sync()
        shared_signed_margins = None
        shared_masks = None
        with sync_context:
            pair_output = pair_logits(ae, model, batch, args, ddp.device, return_step_logits=uses_loop)
            good_logits, bad_logits, good_return, bad_return, _, task_ids, directions, masks = pair_output
            if uses_loop:
                loss = deep_pairwise_loss(
                    good_logits,
                    bad_logits,
                    args.temperature,
                    args.logit_center_weight,
                    bool(args.loop_deep_supervision),
                )
                good_step_logits, bad_step_logits = good_logits, bad_logits
                metric_good_logits, metric_bad_logits = good_step_logits[-1], bad_step_logits[-1]
            elif uses_multitask:
                if uses_shared_directional:
                    loss, shared_signed_margins, shared_masks = shared_directional_multitask_pairwise_loss(
                        good_logits, bad_logits, directions, masks, args
                    )
                    metric_good_logits, metric_bad_logits = good_logits[:, 0], bad_logits[:, 0]
                else:
                    loss, metric_good_logits, metric_bad_logits = multitask_pairwise_loss(
                        good_logits, bad_logits, task_ids, args
                    )
            else:
                loss = pairwise_loss(good_logits, bad_logits, args.temperature, args.logit_center_weight)
                metric_good_logits, metric_bad_logits = good_logits, bad_logits
            (loss / group_size).backward()
        if is_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        count = int(metric_good_logits.shape[0])
        good_score = torch.sigmoid(metric_good_logits.detach().float())
        bad_score = torch.sigmoid(metric_bad_logits.detach().float())
        loss_sum += float(loss.detach()) * count
        pairs += count
        correct += int((metric_good_logits.detach() > metric_bad_logits.detach()).sum().item())
        gap_sum += float((good_return - bad_return).detach().sum())
        chosen_score_sum += float(good_score.sum())
        rejected_score_sum += float(bad_score.sum())
        if shared_signed_margins is not None and shared_masks is not None:
            valid_counts, correct_counts, margin_sums = _shared_directional_metric_totals(
                shared_signed_margins, shared_masks
            )
            shared_valid_counts += valid_counts
            shared_correct_counts += correct_counts
            shared_margin_sums += margin_sums
        if uses_loop:
            for rollout_step, (good_step, bad_step) in enumerate(zip(good_step_logits.detach(), bad_step_logits.detach(), strict=True)):
                step_correct[rollout_step] += int((good_step > bad_step).sum().item())
                step_margin_sum[rollout_step] += float((good_step - bad_step).float().sum())
        if int(args.log_every_steps) and step % int(args.log_every_steps) == 0 and ddp.is_main:
            elapsed = max(time.perf_counter() - started, 1e-9)
            running = {
                "event": "train_progress",
                "stage": "standalone_reward_pairwise",
                "epoch": int(epoch),
                "step": int(step),
                "steps_per_epoch": int(len(loader)),
                "loss_running": float(loss_sum / max(pairs, 1)),
                "pair_accuracy_running": float(correct / max(pairs, 1)),
                "pairs_per_second_running": float(pairs / elapsed),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            print(json.dumps(running, ensure_ascii=False, sort_keys=True), flush=True)
            append_training_log(args, ddp, running)
            if writer is not None:
                global_step = (int(epoch) - 1) * len(loader) + step
                for name in ("loss_running", "pair_accuracy_running", "pairs_per_second_running"):
                    writer.add_scalar(f"reward/{name}", float(running[name]), global_step)
                writer.flush()
    metrics = reduce_pair_metrics(
        loss_sum, pairs, correct, gap_sum, chosen_score_sum, rejected_score_sum, time.perf_counter() - started, ddp, optimizer
    )
    metrics = attach_rollout_pair_metrics(metrics, step_correct, step_margin_sum, ddp)
    if uses_shared_directional:
        metrics = attach_shared_directional_pair_metrics(
            metrics, shared_valid_counts, shared_correct_counts, shared_margin_sums, ddp
        )
    return metrics


@torch.no_grad()
def evaluate_pairs(ae, model, loader, args: argparse.Namespace, device: torch.device) -> dict[str, float | int]:
    if ae is not None:
        ae.eval()
    model.eval()
    loss_sum = 0.0
    pairs = 0
    correct = 0
    gap_sum = 0.0
    chosen_score_sum = 0.0
    rejected_score_sum = 0.0
    uses_loop = str(args.reward_architecture) == "loop_preference"
    uses_multitask = is_multitask_label_mode(args)
    uses_shared_directional = uses_shared_directional_multitask(args)
    step_correct = [0 for _ in range(int(args.loop_steps))] if uses_loop else []
    step_margin_sum = [0.0 for _ in range(int(args.loop_steps))] if uses_loop else []
    shared_valid_counts = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=device)
    shared_correct_counts = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=device)
    shared_margin_sums = torch.zeros(len(MULTITASK_SCORE_HEADS), dtype=torch.float64, device=device)
    class_counts = np.zeros(len(TREND_LABELS), dtype=np.int64)
    started = time.perf_counter()
    for batch in loader:
        good_logits, bad_logits, good_return, bad_return, labels, task_ids, directions, masks = pair_logits(
            ae, model, batch, args, device, return_step_logits=uses_loop
        )
        shared_signed_margins = None
        shared_masks = None
        if uses_loop:
            loss = deep_pairwise_loss(
                good_logits,
                bad_logits,
                args.temperature,
                args.logit_center_weight,
                bool(args.loop_deep_supervision),
            )
            good_step_logits, bad_step_logits = good_logits, bad_logits
            metric_good_logits, metric_bad_logits = good_step_logits[-1], bad_step_logits[-1]
        elif uses_multitask:
            if uses_shared_directional:
                loss, shared_signed_margins, shared_masks = shared_directional_multitask_pairwise_loss(
                    good_logits, bad_logits, directions, masks, args
                )
                metric_good_logits, metric_bad_logits = good_logits[:, 0], bad_logits[:, 0]
            else:
                loss, metric_good_logits, metric_bad_logits = multitask_pairwise_loss(
                    good_logits, bad_logits, task_ids, args
                )
        else:
            loss = pairwise_loss(good_logits, bad_logits, args.temperature, args.logit_center_weight)
            metric_good_logits, metric_bad_logits = good_logits, bad_logits
        count = int(metric_good_logits.shape[0])
        loss_sum += float(loss) * count
        pairs += count
        correct += int((metric_good_logits > metric_bad_logits).sum().item())
        gap_sum += float((good_return - bad_return).sum())
        chosen_score_sum += float(torch.sigmoid(metric_good_logits.float()).sum())
        rejected_score_sum += float(torch.sigmoid(metric_bad_logits.float()).sum())
        if shared_signed_margins is not None and shared_masks is not None:
            valid_counts, correct_counts, margin_sums = _shared_directional_metric_totals(
                shared_signed_margins, shared_masks
            )
            shared_valid_counts += valid_counts
            shared_correct_counts += correct_counts
            shared_margin_sums += margin_sums
        if uses_loop:
            for rollout_step, (good_step, bad_step) in enumerate(zip(good_step_logits, bad_step_logits, strict=True)):
                step_correct[rollout_step] += int((good_step > bad_step).sum().item())
                step_margin_sum[rollout_step] += float((good_step - bad_step).float().sum())
        class_counts += np.bincount(labels.cpu().numpy(), minlength=len(TREND_LABELS))
    metrics = reduce_pair_metrics(
        loss_sum,
        pairs,
        correct,
        gap_sum,
        chosen_score_sum,
        rejected_score_sum,
        time.perf_counter() - started,
        DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0, device=device, backend=None),
        type("Optimizer", (), {"param_groups": [{"lr": 0.0}]})(),
    )
    metrics = attach_rollout_pair_metrics(metrics, step_correct, step_margin_sum, DistributedContext(
        enabled=False, rank=0, world_size=1, local_rank=0, device=device, backend=None
    ))
    if uses_shared_directional:
        metrics = attach_shared_directional_pair_metrics(
            metrics,
            shared_valid_counts,
            shared_correct_counts,
            shared_margin_sums,
            DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0, device=device, backend=None),
        )
    metrics["good_class_counts"] = {label: int(class_counts[index]) for index, label in enumerate(TREND_LABELS)}
    return metrics


def _rankic_and_topk(
    frame: pd.DataFrame,
    *,
    score_column: str,
    truth_column: str,
    event_threshold: float,
) -> dict[str, Any]:
    rankics: list[float] = []
    top: dict[int, dict[str, float]] = {count: {"selected_sum": 0.0, "base_sum": 0.0, "selected": 0.0, "event": 0.0} for count in (1, 5, 10)}
    for _, group in frame.groupby("signal_date", sort=False):
        score = group[score_column].to_numpy(dtype=np.float64)
        truth = group[truth_column].to_numpy(dtype=np.float64)
        if len(group) >= 2 and np.unique(score).size > 1 and np.unique(truth).size > 1:
            rankic = pd.Series(score).corr(pd.Series(truth), method="spearman")
            if rankic is not None and np.isfinite(rankic):
                rankics.append(float(rankic))
        order = np.argsort(-score, kind="mergesort")
        for count, metrics in top.items():
            selected = truth[order[: min(count, len(order))]]
            metrics["selected_sum"] += float(selected.sum())
            metrics["base_sum"] += float(truth.mean()) * len(selected)
            metrics["selected"] += len(selected)
            metrics["event"] += int((selected >= float(event_threshold)).sum())
    output: dict[str, Any] = {
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
        "topk": {},
    }
    for count, values in top.items():
        selected = max(1.0, values["selected"])
        selected_avg = values["selected_sum"] / selected
        base_avg = values["base_sum"] / selected
        event_key = f"p_return_ge_{float(event_threshold) * 100:g}pct"
        output["topk"][f"top{count}"] = {
            "selected_rows": int(values["selected"]),
            "selected_avg_return": float(selected_avg),
            "base_avg_return": float(base_avg),
            "excess_avg_return": float(selected_avg - base_avg),
            event_key: float(values["event"] / selected),
        }
    return output


def _cross_sectional_metric_summary(
    frame: pd.DataFrame,
    *,
    score_column: str,
    value_column: str,
    higher_is_better: bool,
    event_threshold: float | None = None,
) -> dict[str, Any]:
    """Audit one path metric under the model's TopK selections."""

    direction = 1.0 if higher_is_better else -1.0
    rankics: list[float] = []
    top: dict[int, dict[str, float]] = {
        count: {"selected_sum": 0.0, "base_sum": 0.0, "selected": 0.0, "event": 0.0}
        for count in (1, 5, 10)
    }
    for _, group in frame.groupby("signal_date", sort=False):
        score = group[score_column].to_numpy(dtype=np.float64)
        value = group[value_column].to_numpy(dtype=np.float64)
        rank_value = direction * value
        if len(group) >= 2 and np.unique(score).size > 1 and np.unique(rank_value).size > 1:
            rankic = pd.Series(score).corr(pd.Series(rank_value), method="spearman")
            if rankic is not None and np.isfinite(rankic):
                rankics.append(float(rankic))
        order = np.argsort(-score, kind="mergesort")
        for count, totals in top.items():
            selected = value[order[: min(count, len(order))]]
            totals["selected_sum"] += float(selected.sum())
            totals["base_sum"] += float(value.mean()) * len(selected)
            totals["selected"] += len(selected)
            if event_threshold is not None:
                totals["event"] += int((selected >= float(event_threshold)).sum())
    result: dict[str, Any] = {
        "value_column": value_column,
        "higher_is_better": bool(higher_is_better),
        "rows": int(len(frame)),
        "dates": int(frame["signal_date"].nunique()),
        "rankic_mean": float(np.mean(rankics)) if rankics else None,
        "rankic_median": float(np.median(rankics)) if rankics else None,
        "rankic_positive_ratio": float(np.mean(np.asarray(rankics) > 0.0)) if rankics else None,
        "topk": {},
    }
    for count, totals in top.items():
        selected_count = max(1.0, totals["selected"])
        selected_avg = totals["selected_sum"] / selected_count
        base_avg = totals["base_sum"] / selected_count
        values: dict[str, float | int] = {
            "selected_rows": int(totals["selected"]),
            "selected_avg": float(selected_avg),
            "base_avg": float(base_avg),
            "selection_advantage": float(direction * (selected_avg - base_avg)),
        }
        if event_threshold is not None:
            values[f"p_value_ge_{float(event_threshold) * 100:g}pct"] = float(totals["event"] / selected_count)
        result["topk"][f"top{count}"] = values
    return result


def top5_terminal_return_selection_advantage(
    cross_sectional: dict[str, Any], *, label_mode: str
) -> float | None:
    if str(label_mode) == "endpoint":
        value = cross_sectional.get("topk", {}).get("top5", {}).get("excess_avg_return")
    elif str(label_mode) == "multitask":
        value = cross_sectional.get("return_score", {}).get("topk", {}).get("top5", {}).get("selection_advantage")
    else:
        value = cross_sectional.get("terminal_return", {}).get("topk", {}).get("top5", {}).get("selection_advantage")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def multitask_two_stage_terminal_return_selection_advantage(cross_sectional: dict[str, Any]) -> float | None:
    value = cross_sectional.get("return_topn_risk_top5", {}).get("topk", {}).get("top5", {}).get("selection_advantage")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


@torch.no_grad()
def evaluate_cross_sectional(ae, model, dataset, rows: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if ae is not None:
        ae.eval()
    model.eval()
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.eval_batch_size),
        "shuffle": False,
        "num_workers": int(args.eval_num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.eval_num_workers) > 0,
    }
    if int(args.eval_num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    path_parts: list[np.ndarray] = []
    offset = 0
    started = time.perf_counter()
    for x_ts, x_market, x_candidate, targets in loader:
        x_ts = x_ts.to(device, dtype=torch.float32, non_blocking=True)
        x_market = x_market.to(device, dtype=torch.float32, non_blocking=True)
        x_candidate = x_candidate.to(device, dtype=torch.float32, non_blocking=True)
        logits = _forward_reward(ae, model, x_ts, x_market, x_candidate, args, device)
        score_parts.append(torch.sigmoid(logits.float()).cpu().numpy())
        target_parts.append(targets[:, int(args.rank_horizon) - 1].float().numpy())
        if str(args.label_mode) not in {"endpoint", "residual_utility"}:
            path_parts.append(targets[:, : int(args.rank_horizon)].float().numpy())
        offset += int(len(logits))
    if offset != len(rows):
        raise RuntimeError(f"Cross-sectional validation row mismatch: predicted={offset}, expected={len(rows)}")
    score_values = np.concatenate(score_parts)
    frame = pd.DataFrame(
        {
            "signal_date": rows["signal_date"].astype(str).to_numpy(),
            "instrument": rows["instrument"].astype(str).to_numpy(),
            true_return_column(int(args.rank_horizon)): np.concatenate(target_parts),
        }
    )
    if is_multitask_label_mode(args):
        if score_values.ndim != 2 or score_values.shape[1] != len(MULTITASK_SCORE_HEADS):
            raise RuntimeError(
                "multi-task cross-sectional evaluation expected [rows, return/sharpe/drawdown] scores, "
                f"got {tuple(score_values.shape)}"
            )
        for head_index, head in enumerate(MULTITASK_SCORE_HEADS):
            frame[multitask_score_column(head, int(args.rank_horizon))] = score_values[:, head_index]
        if not path_parts:
            raise RuntimeError("multi-task cross-sectional evaluation did not collect target paths")
        path_metrics = future_path_metrics(np.concatenate(path_parts), horizon=int(args.rank_horizon))
        frame["path_sharpe"] = path_metrics["path_sharpe"]
        frame["max_drawdown"] = path_metrics["max_drawdown"]
        two_stage_column = multitask_two_stage_score_column(int(args.rank_horizon))
        frame[two_stage_column] = multitask_two_stage_scores(
            signal_dates=frame["signal_date"].to_numpy(),
            return_scores=frame[multitask_score_column("return", int(args.rank_horizon))].to_numpy(),
            sharpe_scores=frame[multitask_score_column("sharpe", int(args.rank_horizon))].to_numpy(),
            drawdown_scores=frame[multitask_score_column("drawdown", int(args.rank_horizon))].to_numpy(),
            return_top_n=int(args.multitask_return_top_n),
            risk_weights=multitask_risk_weights(args),
        )
        metrics = {
            "label_mode": "multitask",
            "return_score": _cross_sectional_metric_summary(
                frame,
                score_column=multitask_score_column("return", int(args.rank_horizon)),
                value_column=true_return_column(int(args.rank_horizon)),
                higher_is_better=True,
                event_threshold=float(args.trend_threshold),
            ),
            "sharpe_score": _cross_sectional_metric_summary(
                frame,
                score_column=multitask_score_column("sharpe", int(args.rank_horizon)),
                value_column="path_sharpe",
                higher_is_better=True,
            ),
            "drawdown_score": _cross_sectional_metric_summary(
                frame,
                score_column=multitask_score_column("drawdown", int(args.rank_horizon)),
                value_column="max_drawdown",
                higher_is_better=True,
            ),
            "return_topn_risk_top5": _cross_sectional_metric_summary(
                frame,
                score_column=two_stage_column,
                value_column=true_return_column(int(args.rank_horizon)),
                higher_is_better=True,
                event_threshold=float(args.trend_threshold),
            ),
            "two_stage_contract": {
                "return_top_n": int(args.multitask_return_top_n),
                "risk_weights": {"sharpe": float(multitask_risk_weights(args)[0]), "drawdown": float(multitask_risk_weights(args)[1])},
                "score_column": two_stage_column,
            },
        }
    elif str(args.label_mode) in {"endpoint", "residual_utility"}:
        frame[reward_score_column(int(args.rank_horizon))] = score_values
        metrics = _rankic_and_topk(
            frame,
            score_column=reward_score_column(int(args.rank_horizon)),
            truth_column=true_return_column(int(args.rank_horizon)),
            event_threshold=float(args.trend_threshold),
        )
        if is_execution_utility_label_mode(args):
            metrics["label_mode"] = "residual_utility"
            metrics["cross_sectional_truth"] = (
                "gross close-to-close return is diagnostic only; training pairs use executable net-utility labels"
            )
    else:
        if not path_parts:
            raise RuntimeError("quality-based cross-sectional evaluation did not collect target paths")
        frame[reward_score_column(int(args.rank_horizon))] = score_values
        path_metrics = future_path_metrics(np.concatenate(path_parts), horizon=int(args.rank_horizon))
        quality = path_quality_from_metrics(
            signal_dates=frame["signal_date"].to_numpy(),
            terminal_return=path_metrics["terminal_return"],
            path_sharpe=path_metrics["path_sharpe"],
            max_drawdown=path_metrics["max_drawdown"],
            weights=path_quality_weights_for_args(args),
        )
        frame["quality_score"] = quality["quality_score"]
        frame["path_sharpe"] = path_metrics["path_sharpe"]
        frame["path_sortino"] = path_metrics["path_sortino"]
        frame["max_drawdown"] = path_metrics["max_drawdown"]
        metrics = {
            "label_mode": str(args.label_mode),
            "quality_score": _cross_sectional_metric_summary(
                frame,
                score_column=reward_score_column(int(args.rank_horizon)),
                value_column="quality_score",
                higher_is_better=True,
            ),
            "terminal_return": _cross_sectional_metric_summary(
                frame,
                score_column=reward_score_column(int(args.rank_horizon)),
                value_column=true_return_column(int(args.rank_horizon)),
                higher_is_better=True,
                event_threshold=float(args.trend_threshold),
            ),
            "path_sharpe": _cross_sectional_metric_summary(
                frame,
                score_column=reward_score_column(int(args.rank_horizon)),
                value_column="path_sharpe",
                higher_is_better=True,
            ),
            "path_sortino": _cross_sectional_metric_summary(
                frame,
                score_column=reward_score_column(int(args.rank_horizon)),
                value_column="path_sortino",
                higher_is_better=True,
            ),
            "max_drawdown": _cross_sectional_metric_summary(
                frame,
                score_column=reward_score_column(int(args.rank_horizon)),
                value_column="max_drawdown",
                higher_is_better=True,
            ),
        }
    metrics["rank_horizon"] = int(args.rank_horizon)
    metrics["trend_threshold"] = float(args.trend_threshold)
    metrics["elapsed_sec"] = float(time.perf_counter() - started)
    return metrics


def reward_checkpoint_kind(backbone: str, architecture: str, *, multitask: bool = False) -> str:
    if multitask:
        if backbone != "frozen_ae" or architecture != "transformer":
            raise ValueError("multi-task reward training requires the frozen-AE one-pass Transformer")
        return "wts_frozen_ae_multitask_reward_transformer_checkpoint_v1"
    if backbone == "raw":
        if architecture != "transformer":
            raise ValueError("loop_preference is only defined for --reward-backbone=frozen_ae")
        return "wts_raw_reward_transformer_checkpoint_v1"
    if backbone != "frozen_ae":
        raise ValueError(f"Unknown reward backbone: {backbone}")
    if architecture == "transformer":
        return "wts_frozen_ae_reward_transformer_checkpoint_v1"
    if architecture == "loop_preference":
        return "wts_frozen_ae_loop_preference_checkpoint_v1"
    raise ValueError(f"Unknown reward architecture: {architecture}")


def _checkpoint_state(
    args: argparse.Namespace,
    *,
    backbone: str,
    architecture: str,
    ae_config: ModelConfig | None,
    ae_checkpoint: Path | None,
    reward_model: nn.Module,
    reward_config: RewardTransformerConfig | RawFeatureRewardTransformerConfig,
    scheduler_contract: dict[str, Any],
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "kind": reward_checkpoint_kind(backbone, architecture, multitask=is_multitask_label_mode(args)),
        "reward_backbone": backbone,
        "reward_architecture": architecture,
        "reward_state": state_dict_without_module(reward_model),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scheduler_contract": scheduler_contract,
        "args": vars(args),
    }
    if backbone == "raw":
        state["raw_model_config"] = asdict(reward_config)
    else:
        if ae_config is None or ae_checkpoint is None:
            raise RuntimeError("frozen_ae checkpoints require AE metadata")
        state["reward_config"] = asdict(reward_config)
        state["ae_model_config"] = asdict(ae_config)
        state["ae_checkpoint"] = str(ae_checkpoint)
    return state


def _pair_loader(dataset, sampler, pairs_per_batch: int, workers: int, cuda: bool):
    kwargs: dict[str, Any] = {
        "batch_size": int(pairs_per_batch),
        "sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": bool(cuda),
        "persistent_workers": int(workers) > 0,
    }
    if int(workers) > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


def train(args: argparse.Namespace, ddp: DistributedContext) -> None:
    if not bool(args.tensorboard):
        raise ValueError("TensorBoard is mandatory for formal training; remove --no-tensorboard")
    if bool(args.resume_checkpoint) != bool(args.resume_run_dir):
        raise ValueError("--resume-checkpoint and --resume-run-dir must be supplied together")
    if str(args.reward_architecture) == "loop_preference" and str(args.reward_backbone) != "frozen_ae":
        raise ValueError("--reward-architecture=loop_preference requires --reward-backbone=frozen_ae")
    if int(args.loop_steps) < 1:
        raise ValueError("--loop-steps must be positive")
    if (
        int(args.epochs) < 1
        or int(args.pairs_per_batch) < 1
        or int(args.bad_per_good) < 1
        or int(args.up_top1_bad_draws) < 1
        or int(args.up_top10_bad_draws) < 1
        or int(args.head_hard_negative_top_rank) < 0
        or int(args.gap_free_top_rank) < 0
        or int(args.rank_horizon) < 1
        or (str(args.label_mode) == "endpoint" and float(args.trend_threshold) <= 0.0)
    ):
        raise ValueError(
            "--epochs, --pairs-per-batch, --bad-per-good, --up-top1-bad-draws, --up-top10-bad-draws, "
            "--head-hard-negative-top-rank, --gap-free-top-rank, "
            "--rank-horizon, and --trend-threshold must be positive"
        )
    if int(args.up_top1_bad_draws) < int(args.up_top10_bad_draws):
        raise ValueError("--up-top1-bad-draws must be greater than or equal to --up-top10-bad-draws")
    if (int(args.up_top1_bad_draws) > 1 or int(args.up_top10_bad_draws) > 1) and int(args.bad_per_good) != 1:
        raise ValueError("Head-up bad fan-out requires --bad-per-good=1")
    if str(args.label_mode) == "path_quality":
        if str(args.pair_mode) != "all_up":
            raise ValueError("--label-mode=path_quality is currently defined for --pair-mode=all_up only")
        if int(args.rank_horizon) < 3:
            raise ValueError("--label-mode=path_quality requires --rank-horizon >= 3")
        if float(args.min_quality_gap) <= 0.0:
            raise ValueError("--min-quality-gap must be positive")
        if not 0.0 < float(args.quality_down_quantile) < float(args.quality_up_quantile) < 1.0:
            raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
        if int(args.head_hard_negative_top_rank) != 0 or int(args.gap_free_top_rank) != 0:
            raise ValueError("path_quality requires --head-hard-negative-top-rank=0 and --gap-free-top-rank=0")
    if str(args.label_mode) == "gap_quality":
        if str(args.pair_mode) != "all_up":
            raise ValueError("--label-mode=gap_quality is currently defined for --pair-mode=all_up only")
        if int(args.rank_horizon) < 3:
            raise ValueError("--label-mode=gap_quality requires --rank-horizon >= 3")
        if not 0.0 <= float(args.gap_return_min) < float(args.gap_return_max):
            raise ValueError("gap_quality requires 0 <= --gap-return-min < --gap-return-max")
        if float(args.gap_quality_min_difference) <= 0.0:
            raise ValueError("--gap-quality-min-difference must be positive")
        if int(args.gap_quality_bad_options) < 1:
            raise ValueError("--gap-quality-bad-options must be positive")
        if not 0.0 < float(args.quality_down_quantile) < float(args.quality_up_quantile) < 1.0:
            raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
        if int(args.bad_per_good) != 1 or int(args.head_hard_negative_top_rank) != 0 or int(args.gap_free_top_rank) != 0:
            raise ValueError("gap_quality requires --bad-per-good=1 with head fan-out and gap-free ranking disabled")
        gap_quality_weights(args)
    if is_execution_utility_label_mode(args):
        if str(args.pair_mode) != "all_up":
            raise ValueError("--label-mode=residual_utility is currently defined for --pair-mode=all_up only")
        if str(args.reward_backbone) != "frozen_ae" or str(args.reward_architecture) != "transformer":
            raise ValueError("--label-mode=residual_utility requires --reward-backbone=frozen_ae and --reward-architecture=transformer")
        if int(args.rank_horizon) < 1:
            raise ValueError("--label-mode=residual_utility requires --rank-horizon >= 1")
        if float(args.min_return_gap) <= 0.0:
            raise ValueError("residual_utility pairs require --min-return-gap to be positive")
        labels_path = Path(str(args.execution_utility_labels)).expanduser().resolve()
        if not str(args.execution_utility_labels) or not labels_path.is_file():
            raise FileNotFoundError("--label-mode=residual_utility requires an existing --execution-utility-labels artifact")
    if is_multitask_label_mode(args):
        if str(args.pair_mode) != "all_up":
            raise ValueError("--label-mode=multitask is currently defined for --pair-mode=all_up only")
        if str(args.reward_backbone) != "frozen_ae" or str(args.reward_architecture) != "transformer":
            raise ValueError("--label-mode=multitask requires --reward-backbone=frozen_ae and --reward-architecture=transformer")
        if int(args.rank_horizon) < 3:
            raise ValueError("--label-mode=multitask requires --rank-horizon >= 3")
        if float(args.min_return_gap) <= 0.0:
            raise ValueError("multi-task return pairs require --min-return-gap to be positive")
        if not 0.0 < float(args.multitask_risk_min_percentile_gap) <= 1.0:
            raise ValueError("--multitask-risk-min-percentile-gap must be inside (0, 1]")
        if multitask_pair_contract(args) == "independent_risk_pairs":
            if not 0.0 < float(args.multitask_risk_return_gap_max):
                raise ValueError("--multitask-risk-return-gap-max must be positive")
            if int(args.multitask_risk_bad_options) < 1:
                raise ValueError("--multitask-risk-bad-options must be positive")
        if int(args.multitask_return_top_n) < 5:
            raise ValueError("--multitask-return-top-n must be at least 5")
        if not 0.0 < float(args.quality_down_quantile) < float(args.quality_up_quantile) < 1.0:
            raise ValueError("quality down/up quantiles must satisfy 0 < down < up < 1")
        multitask_loss_weights(args)
        multitask_risk_weights(args)
    if str(args.pair_mode) == "tail_net":
        if int(args.tail_positive_rank) < 1:
            raise ValueError("--tail-positive-rank must be positive")
        if int(args.tail_bad_min_rank) <= int(args.tail_positive_rank):
            raise ValueError("--tail-bad-min-rank must be greater than --tail-positive-rank")
        if int(args.tail_bad_max_rank) < int(args.tail_bad_min_rank):
            raise ValueError("--tail-bad-max-rank must be at least --tail-bad-min-rank")
        if float(args.tail_min_return) < 0.0:
            raise ValueError("--tail-min-return must be non-negative")
    if int(args.screen_epoch) < 0 or int(args.screen_epoch) > int(args.epochs):
        raise ValueError("--screen-epoch must be zero or inside the configured epoch range")
    if int(args.screen_epoch) > 0 and (
        int(args.cross_sectional_eval_every) <= 0 or int(args.screen_epoch) % int(args.cross_sectional_eval_every) != 0
    ):
        raise ValueError("--screen-epoch must coincide with a cross-sectional validation epoch")
    root = Path(args.root).expanduser().resolve()
    paths = WTSPaths(root)
    validate_target_contract(paths, int(args.rank_horizon))
    fold_path = Path(args.fold_path).expanduser().resolve() if args.fold_path else paths.folds / f"{args.fold_id}.json"
    split = read_json(fold_path)
    pair_train_rows = read_split_rows(paths, split, str(args.pair_train_split), int(args.max_train_dates))
    validation_rows = read_split_rows(paths, split, str(args.validation_split), int(args.max_validation_dates))
    run_dir = prepare_run_dir(paths, args, ddp)
    args.training_log_path = str(run_dir / "logs" / "train_metrics.jsonl")
    writer, tensorboard = create_tensorboard_writer(run_dir, args, ddp)
    try:
        scaler = load_or_fit_scaler(args, paths, pair_train_rows, ddp)
        train_dataset = build_same_date_pair_dataset(
            args,
            root=root,
            rows=pair_train_rows,
            scaler=scaler,
            bad_per_good=int(args.bad_per_good),
            up_top1_bad_draws=int(args.up_top1_bad_draws),
            up_top10_bad_draws=int(args.up_top10_bad_draws),
            head_hard_negative_top_rank=int(args.head_hard_negative_top_rank),
            gap_free_top_rank=int(args.gap_free_top_rank),
            class_balance_scope=str(args.class_balance_scope),
            seed=20260720,
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ddp.world_size,
            rank=ddp.rank,
            shuffle=bool(args.train_shuffle),
            seed=20260720,
            drop_last=False,
        )
        train_loader = _pair_loader(
            train_dataset, train_sampler, int(args.pairs_per_batch), int(args.num_workers), ddp.device.type == "cuda"
        )
        val_pair_dataset = None
        val_pair_loader = None
        val_market_dataset = None
        if ddp.is_main:
            val_pair_dataset = build_same_date_pair_dataset(
                args,
                root=root,
                rows=validation_rows,
                scaler=scaler,
                bad_per_good=1,
                up_top1_bad_draws=1,
                up_top10_bad_draws=1,
                head_hard_negative_top_rank=0,
                gap_free_top_rank=0,
                class_balance_scope="date",
                seed=20260721,
            )
            val_pair_loader = _pair_loader(
                val_pair_dataset, None, int(args.pairs_per_batch), int(args.eval_num_workers), ddp.device.type == "cuda"
            )
            val_market_dataset = WeakToStrongPathDataset(
                kronos_root=args.kronos_root,
                experiment_root=root,
                rows=validation_rows,
                scaler=scaler,
                target_kind="abs",
                input_mode=args.input_mode,
            )
        backbone = str(args.reward_backbone)
        architecture = str(args.reward_architecture)
        ae: nn.Module | None = None
        ae_config: ModelConfig | None = None
        ae_checkpoint: Path | None = None
        stock_dim = 42 if args.input_mode == "relative_only" else 52
        if backbone == "frozen_ae":
            ae, ae_config, ae_checkpoint = load_frozen_ae(args, ddp.device)
            reward_config: RewardTransformerConfig | RawFeatureRewardTransformerConfig = RewardTransformerConfig(
                cond_dim=int(ae_config.latent_dim),
                cond_tokens=int(ae_config.latent_tokens),
                d_model=int(args.reward_d_model),
                n_layers=int(args.reward_layers),
                n_heads=int(args.reward_heads),
                mlp_ratio=float(args.reward_mlp_ratio),
                dropout=float(args.reward_dropout),
                score_heads=multitask_score_heads_for_args(args),
            )
            if architecture == "loop_preference":
                reward = FrozenAELoopPreferenceTransformer(
                    reward_config,
                    rollout_steps=int(args.loop_steps),
                    condition_update=str(args.loop_condition_update),
                ).to(ddp.device)
            else:
                reward = FrozenAERewardTransformer(reward_config).to(ddp.device)
        else:
            reward_config = RawFeatureRewardTransformerConfig(
                stock_dim=stock_dim,
                market_dim=len(MARKET_FEATURE_FIELDS),
                candidate_dim=len(CANDIDATE_FEATURE_FIELDS),
                lookback=SEQ_LEN,
                d_model=int(args.reward_d_model),
                n_layers=int(args.reward_layers),
                n_heads=int(args.reward_heads),
                mlp_ratio=float(args.reward_mlp_ratio),
                dropout=float(args.reward_dropout),
            )
            reward = RawFeatureRewardTransformer(reward_config).to(ddp.device)
        optimizer = torch.optim.AdamW(reward.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
        if int(args.grad_accum_steps) < 1:
            raise ValueError("--grad-accum-steps must be positive")
        updates_per_epoch = int(math.ceil(len(train_loader) / int(args.grad_accum_steps)))
        decay_epochs = int(args.lr_decay_epochs) or int(args.epochs)
        if not 1 <= decay_epochs <= int(args.epochs):
            raise ValueError("--lr-decay-epochs must be within [1, --epochs], or zero to use --epochs")
        if not 0.0 <= float(args.min_lr) <= float(args.lr):
            raise ValueError("--min-lr must be within [0, --lr]")
        scheduler_contract = {
            "base_lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "warmup_ratio": float(args.warmup_ratio),
            "total_epochs": int(args.epochs),
            "decay_epochs": decay_epochs,
            "micro_batches_per_epoch": int(len(train_loader)),
            "optimizer_updates_per_epoch": updates_per_epoch,
            "grad_accum_steps": int(args.grad_accum_steps),
        }
        scheduler = make_scheduler(
            optimizer,
            updates_per_epoch * int(args.epochs),
            float(args.warmup_ratio),
            min_lr=float(args.min_lr),
            decay_steps=updates_per_epoch * decay_epochs,
        )
        start_epoch = 1
        completed_steps = 0
        resume_path = str(args.resume_checkpoint).strip()
        if resume_path:
            state = torch.load(Path(resume_path).expanduser().resolve(), map_location="cpu", weights_only=False)
            expected_kind = reward_checkpoint_kind(backbone, architecture, multitask=is_multitask_label_mode(args))
            if str(state.get("kind")) != expected_kind:
                raise RuntimeError("Resume checkpoint backbone does not match requested architecture")
            config_key = "raw_model_config" if backbone == "raw" else "reward_config"
            if dict(state.get(config_key) or {}) != asdict(reward_config):
                raise RuntimeError("Resume checkpoint reward configuration does not match requested architecture")
            if is_multitask_label_mode(args):
                saved_args = dict(state.get("args") or {})
                saved_raw_pair_contract = saved_args.get("multitask_pair_contract")
                saved_pair_contract = (
                    "independent_risk_pairs"
                    if saved_raw_pair_contract in (None, "")
                    else str(saved_raw_pair_contract)
                )
                requested_pair_contract = multitask_pair_contract(args)
                if saved_pair_contract != requested_pair_contract:
                    raise RuntimeError(
                        "Resume checkpoint multi-task pair contract does not match; start a new run for a changed objective: "
                        f"checkpoint={saved_pair_contract}, requested={requested_pair_contract}"
                    )
            if architecture == "loop_preference":
                saved_args = dict(state.get("args") or {})
                expected_loop_contract = {
                    "loop_steps": int(args.loop_steps),
                    "loop_condition_update": str(args.loop_condition_update),
                    "loop_deep_supervision": bool(args.loop_deep_supervision),
                }
                actual_loop_contract = {key: saved_args.get(key) for key in expected_loop_contract}
                if actual_loop_contract != expected_loop_contract:
                    raise RuntimeError(
                        "Resume checkpoint loop contract does not match requested rollout configuration: "
                        f"checkpoint={actual_loop_contract}, requested={expected_loop_contract}"
                    )
            if backbone == "frozen_ae" and str(state.get("ae_checkpoint")) != str(ae_checkpoint):
                raise RuntimeError("Resume checkpoint refers to a different frozen AE checkpoint")
            if dict(state.get("scheduler_contract") or {}) != scheduler_contract:
                raise RuntimeError("Resume checkpoint scheduler contract does not match; start a new run for a changed schedule")
            load_model_state_flexible(reward, state["reward_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            start_epoch = int(state["epoch"]) + 1
            completed_steps = int(state.get("global_step", (start_epoch - 1) * updates_per_epoch))
            if start_epoch > int(args.epochs):
                raise RuntimeError(f"Resume checkpoint already completed requested epoch {start_epoch - 1}")
        model: nn.Module = reward
        if ddp.enabled:
            kwargs: dict[str, Any] = {}
            if ddp.device.type == "cuda":
                kwargs.update({"device_ids": [ddp.local_rank], "output_device": ddp.local_rank})
            model = DistributedDataParallel(reward, **kwargs)
        parameter_counts = {
            "frozen_ae": count_parameters(ae) if ae is not None else 0,
            "reward_trainable": count_parameters(reward),
            "total_registered": (count_parameters(ae) if ae is not None else 0) + count_parameters(reward),
        }
        if ddp.is_main:
            append_training_log(
                args,
                ddp,
                {
                    "event": "run_start",
                    "stage": "standalone_reward_pairwise",
                    "run_dir": str(run_dir),
                    "target_epochs": int(args.epochs),
                    "start_epoch": int(start_epoch),
                    "backbone": backbone,
                    "architecture": architecture,
                    "multitask_pair_contract": (
                        multitask_pair_contract(args) if is_multitask_label_mode(args) else None
                    ),
                    "loop_contract": (
                        {
                            "steps": int(args.loop_steps),
                            "condition_update": str(args.loop_condition_update),
                            "deep_supervision": bool(args.loop_deep_supervision),
                            "step_loss_weights": loop_pairwise_weights(
                                int(args.loop_steps), bool(args.loop_deep_supervision), dtype=torch.float32
                            ).tolist(),
                        }
                        if architecture == "loop_preference"
                        else None
                    ),
                    "micro_batches_per_epoch": int(len(train_loader)),
                    "optimizer_updates_per_epoch": updates_per_epoch,
                    "effective_pairs_per_optimizer_update": int(args.pairs_per_batch)
                    * int(args.grad_accum_steps)
                    * ddp.world_size,
                    "train_pair_dataset": train_dataset.stats,
                    "validation_pair_dataset": val_pair_dataset.stats if val_pair_dataset is not None else None,
                    "parameter_counts": parameter_counts,
                    "reward_model_config": asdict(reward_config),
                    "ae_checkpoint": str(ae_checkpoint) if ae_checkpoint is not None else None,
                    "scheduler_contract": scheduler_contract,
                    "tensorboard": tensorboard,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "model_parameters",
                        **parameter_counts,
                        "train_pairs_per_epoch": len(train_dataset),
                        "validation_pairs_per_epoch": len(val_pair_dataset) if val_pair_dataset is not None else 0,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        history: list[dict[str, Any]] = []
        run_status = "trained"
        checkpoint_dir = run_dir / "checkpoints"
        for epoch in range(start_epoch, int(args.epochs) + 1):
            train_sampler.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            metrics = train_one_epoch(ae, model, train_loader, optimizer, scheduler, args, ddp, epoch, writer)
            row: dict[str, Any] = {"stage": "standalone_reward_pairwise", "epoch": int(epoch), "train": metrics}
            screen_rejected = False
            if ddp.is_main:
                if val_pair_dataset is None or val_pair_loader is None or val_market_dataset is None:
                    raise RuntimeError("rank0 validation datasets were not initialized")
                should_pair_eval = int(args.pair_validation_every) > 0 and (
                    epoch % int(args.pair_validation_every) == 0 or epoch == int(args.epochs)
                )
                if should_pair_eval:
                    val_pair_dataset.set_epoch(epoch)
                    row["validation_pairs"] = evaluate_pairs(ae, unwrap_model(model), val_pair_loader, args, ddp.device)
                else:
                    row["validation_pairs"] = {"status": "skipped_by_pair_validation_every"}
                should_cross_eval = int(args.cross_sectional_eval_every) > 0 and (
                    epoch % int(args.cross_sectional_eval_every) == 0 or epoch == int(args.epochs)
                )
                if should_cross_eval:
                    row["validation_cross_sectional"] = evaluate_cross_sectional(
                        ae, unwrap_model(model), val_market_dataset, validation_rows, args, ddp.device
                    )
                completed_steps = epoch * updates_per_epoch
                state = _checkpoint_state(
                    args,
                    backbone=backbone,
                    architecture=architecture,
                    ae_config=ae_config,
                    ae_checkpoint=ae_checkpoint,
                    reward_model=unwrap_model(model),
                    reward_config=reward_config,
                    scheduler_contract=scheduler_contract,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=completed_steps,
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                epoch_checkpoint = checkpoint_dir / f"reward_epoch_{epoch:03d}.pt"
                torch.save(state, epoch_checkpoint)
                torch.save(state, run_dir / "last_checkpoint.pt")
                compact_keys = ("kind", "reward_backbone", "reward_state", "reward_config", "raw_model_config", "ae_model_config", "ae_checkpoint", "epoch", "args")
                torch.save({key: state[key] for key in compact_keys if key in state}, run_dir / "reward_checkpoint.pt")
                if int(args.screen_epoch) == int(epoch):
                    cross = row.get("validation_cross_sectional")
                    if not isinstance(cross, dict):
                        raise RuntimeError("screening requires cross-sectional validation at the configured epoch")
                    top5_excess = top5_terminal_return_selection_advantage(cross, label_mode=str(args.label_mode))
                    if top5_excess is None:
                        raise RuntimeError("screening could not read the validation Top5 terminal-return selection advantage")
                    if is_multitask_label_mode(args):
                        two_stage_excess = multitask_two_stage_terminal_return_selection_advantage(cross)
                        if two_stage_excess is None:
                            raise RuntimeError("multi-task screening could not read the return-TopN risk-Top5 selection advantage")
                        screen_rejected = not (
                            top5_excess > float(args.screen_min_top5_excess)
                            and two_stage_excess > float(args.screen_min_top5_excess)
                        )
                        row["screening"] = {
                            "epoch": int(epoch),
                            "metric": "validation_return_head_and_return_topn_risk_top5_selection_advantage",
                            "return_head_top5_value": top5_excess,
                            "two_stage_top5_value": two_stage_excess,
                            "minimum_exclusive": float(args.screen_min_top5_excess),
                            "decision": "rejected" if screen_rejected else "passed",
                        }
                    else:
                        screen_rejected = not top5_excess > float(args.screen_min_top5_excess)
                        row["screening"] = {
                            "epoch": int(epoch),
                            "metric": "validation_top5_terminal_return_selection_advantage",
                            "value": top5_excess,
                            "minimum_exclusive": float(args.screen_min_top5_excess),
                            "decision": "rejected" if screen_rejected else "passed",
                        }
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
                append_training_log(
                    args,
                    ddp,
                    {"event": "epoch_end", **row, "checkpoint": str(epoch_checkpoint), "last_checkpoint": str(run_dir / "last_checkpoint.pt")},
                )
                if writer is not None:
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"reward/train/{name}", float(value), epoch)
                    for name, value in row["validation_pairs"].items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"reward/validation_pair/{name}", float(value), epoch)
                    cross = row.get("validation_cross_sectional")
                    if isinstance(cross, dict):
                        if str(args.label_mode) in {"endpoint", "residual_utility"}:
                            for name in ("rankic_mean", "rankic_median", "rankic_positive_ratio"):
                                if isinstance(cross.get(name), (int, float)):
                                    writer.add_scalar(f"reward/validation_cross_sectional/{name}", float(cross[name]), epoch)
                            for top_name, top_metrics in cross.get("topk", {}).items():
                                event_key = f"p_return_ge_{float(args.trend_threshold) * 100:g}pct"
                                for name in ("selected_avg_return", "excess_avg_return", event_key):
                                    if isinstance(top_metrics.get(name), (int, float)):
                                        writer.add_scalar(
                                            f"reward/validation_cross_sectional/{top_name}_{name}", float(top_metrics[name]), epoch
                                        )
                        else:
                            for metric_name, metric_summary in cross.items():
                                if not isinstance(metric_summary, dict) or "rankic_mean" not in metric_summary:
                                    continue
                                for name in ("rankic_mean", "rankic_median", "rankic_positive_ratio"):
                                    if isinstance(metric_summary.get(name), (int, float)):
                                        writer.add_scalar(
                                            f"reward/validation_cross_sectional/{metric_name}_{name}",
                                            float(metric_summary[name]),
                                            epoch,
                                        )
                                top5 = metric_summary.get("topk", {}).get("top5", {})
                                if isinstance(top5.get("selection_advantage"), (int, float)):
                                    writer.add_scalar(
                                        f"reward/validation_cross_sectional/{metric_name}_top5_selection_advantage",
                                        float(top5["selection_advantage"]),
                                        epoch,
                                    )
                    writer.flush()
            if ddp.enabled:
                screen_flag = torch.tensor([int(screen_rejected)], device=ddp.device, dtype=torch.int64)
                dist.broadcast(screen_flag, src=0)
                screen_rejected = bool(screen_flag.item())
            history.append(row)
            barrier(ddp)
            if screen_rejected:
                run_status = "screened_out"
                break
        if ddp.is_main:
            if str(args.pair_mode) == "tail_net":
                pair_contract = (
                    "same signal_date; chosen rank <= tail_positive_rank and return >= tail_min_return; "
                    "rejected rank is inside tail_bad_min_rank..tail_bad_max_rank and "
                    "bad return < good return - min_return_gap"
                )
                good_distribution = "realized TopK chosen rows only with optional return floor; no range/down balancing"
            else:
                if is_multitask_label_mode(args):
                    if uses_shared_directional_multitask(args):
                        pair_contract = (
                            "same signal_date return pair is shared by all heads; return direction is good return > bad return; "
                            "Sharpe and signed-max-drawdown directions independently follow their daily percentile differences, "
                            "with abs difference below multitask_risk_min_percentile_gap masked"
                        )
                        good_distribution = (
                            "one endpoint-return pair stream; every retained pair supervises return and any sufficiently "
                            "separated Sharpe/drawdown labels"
                        )
                    else:
                        pair_contract = (
                            "return task: same signal_date with bad return < good return - min_return_gap; "
                            "Sharpe/drawdown tasks: same signal_date, abs endpoint return gap below "
                            "multitask_risk_return_gap_max, and chosen metric percentile exceeds rejected by "
                            "multitask_risk_min_percentile_gap"
                        )
                        good_distribution = "each task has independent same-date pairs; return preserves endpoint labels, risk tasks use daily exact 1:1:1 metric-percentile labels"
                elif is_execution_utility_label_mode(args):
                    pair_contract = (
                        "same signal_date; both endpoints are QuantX-executable; "
                        "bad net utility < good net utility - min_return_gap"
                    )
                    good_distribution = (
                        "all eligible daily net-utility up rows plus equal same-date range/down samples; "
                        "rows rejected at either endpoint never enter a pair"
                    )
                elif str(args.label_mode) == "path_quality":
                    pair_contract = (
                        "same signal_date; quality up/range/down percentile labels; "
                        "Q_bad < Q_good - min_quality_gap; raw endpoint return is audit-only"
                    )
                    good_distribution = (
                        "all eligible up plus globally cycled same-date range/down endpoints to exact 1:1:1 pair labels"
                        if str(args.class_balance_scope) == "global"
                        else "all eligible daily up plus equal same-date range/down base samples; "
                        "optional higher-return up bad-comparison fan-out changes pair-level weights"
                    )
                elif str(args.label_mode) == "gap_quality":
                    pair_contract = (
                        "same signal_date; abs(endpoint_return_good - endpoint_return_bad) inside "
                        f"[{float(args.gap_return_min):.6f}, {float(args.gap_return_max):.6f}); "
                        "Q_good >= Q_bad + gap_quality_min_difference"
                    )
                    good_distribution = "daily exact 1:1:1 weighted-quality up/range/down endpoints with rotated bounded-gap lower-quality bads"
                else:
                    pair_contract = "same signal_date; dynamic endpoint up/range/down labels; bad return < good return - min_return_gap"
                    good_distribution = (
                        "all eligible up plus globally cycled same-date range/down endpoints to exact 1:1:1 pair labels"
                        if str(args.class_balance_scope) == "global"
                        else "all eligible daily up plus equal same-date range/down base samples; "
                        "optional higher-return up bad-comparison fan-out changes pair-level weights"
                    )
            manifest = {
                "kind": (
                    "wts_raw_reward_transformer_train_manifest_v1"
                    if backbone == "raw"
                    else (
                        "wts_frozen_ae_multitask_reward_transformer_train_manifest_v1"
                        if is_multitask_label_mode(args)
                        else (
                            "wts_frozen_ae_loop_preference_train_manifest_v1"
                            if architecture == "loop_preference"
                            else "wts_frozen_ae_reward_transformer_train_manifest_v1"
                        )
                    )
                ),
                "status": run_status,
                "run_dir": str(run_dir),
                "root": str(root),
                "kronos_root": str(Path(args.kronos_root).resolve()),
                "args": vars(args),
                "split": split,
                "reward_backbone": backbone,
                "reward_architecture": architecture,
                "loop_contract": (
                    {
                        "steps": int(args.loop_steps),
                        "condition_update": str(args.loop_condition_update),
                        "deep_supervision": bool(args.loop_deep_supervision),
                        "step_loss_weights": loop_pairwise_weights(
                            int(args.loop_steps), bool(args.loop_deep_supervision), dtype=torch.float32
                        ).tolist(),
                    }
                    if architecture == "loop_preference"
                    else None
                ),
                "parameter_counts": parameter_counts,
                "scheduler_contract": scheduler_contract,
                "train_pair_dataset": train_dataset.stats,
                "validation_pair_dataset": val_pair_dataset.stats if val_pair_dataset is not None else None,
                "history": history,
                "score_contract": {
                "score_column": (
                    multitask_two_stage_score_column(int(args.rank_horizon))
                    if is_multitask_label_mode(args)
                    else reward_score_column(int(args.rank_horizon))
                ),
                    "score_columns": (
                        [
                            *(multitask_score_column(head, int(args.rank_horizon)) for head in MULTITASK_SCORE_HEADS),
                            multitask_two_stage_score_column(int(args.rank_horizon)),
                        ]
                        if is_multitask_label_mode(args)
                        else [reward_score_column(int(args.rank_horizon))]
                    ),
                    "rank_horizon": int(args.rank_horizon),
                    "label_mode": str(args.label_mode),
                    "trend_threshold": float(args.trend_threshold),
                    "path_quality": (
                        {
                            "formula": path_quality_formula(path_quality_weights_for_args(args)),
                            "min_quality_gap": float(args.min_quality_gap),
                            "up_quantile": float(args.quality_up_quantile),
                            "down_quantile": float(args.quality_down_quantile),
                        }
                        if str(args.label_mode) == "path_quality"
                        else None
                    ),
                    "gap_quality": (
                        {
                            "formula": path_quality_formula(gap_quality_weights(args)),
                            "return_gap_min": float(args.gap_return_min),
                            "return_gap_max": float(args.gap_return_max),
                            "min_quality_difference": float(args.gap_quality_min_difference),
                            "bad_options_per_good": int(args.gap_quality_bad_options),
                            "up_quantile": float(args.quality_up_quantile),
                            "down_quantile": float(args.quality_down_quantile),
                        }
                        if str(args.label_mode) == "gap_quality"
                        else None
                    ),
                    "multitask": (
                        {
                            "heads": list(MULTITASK_SCORE_HEADS),
                            "pair_contract": multitask_pair_contract(args),
                            "loss_weights": dict(zip(MULTITASK_SCORE_HEADS, multitask_loss_weights(args), strict=True)),
                            "return_pair_min_gap": float(args.min_return_gap),
                            "risk_return_gap_max_exclusive": (
                                float(args.multitask_risk_return_gap_max)
                                if multitask_pair_contract(args) == "independent_risk_pairs"
                                else None
                            ),
                            "risk_min_percentile_gap": float(args.multitask_risk_min_percentile_gap),
                            "risk_label_semantics": (
                                "chosen percentile exceeds rejected percentile by configured gap"
                                if multitask_pair_contract(args) == "independent_risk_pairs"
                                else "independent signed percentile direction with absolute gap below threshold masked"
                            ),
                            "return_top_n": int(args.multitask_return_top_n),
                            "risk_weights": dict(zip(("sharpe", "drawdown"), multitask_risk_weights(args), strict=True)),
                            "two_stage_score_column": multitask_two_stage_score_column(int(args.rank_horizon)),
                        }
                        if is_multitask_label_mode(args)
                        else None
                    ),
                    "range": "sigmoid(logit), strictly between 0 and 1",
                    "pair_mode": str(args.pair_mode),
                    "pair_contract": pair_contract,
                    "good_distribution": good_distribution,
                },
            }
            if backbone == "raw":
                manifest["raw_model_config"] = asdict(reward_config)
            else:
                if ae_config is None or ae_checkpoint is None:
                    raise RuntimeError("frozen_ae manifest requires AE metadata")
                manifest["ae_model_config"] = asdict(ae_config)
                manifest["ae_checkpoint"] = str(ae_checkpoint)
                manifest["reward_config"] = asdict(reward_config)
            write_json(run_dir / "train_manifest.json", manifest)
            append_training_log(
                args,
                ddp,
                {"event": "run_end", "stage": "standalone_reward_pairwise", "status": run_status, "run_dir": str(run_dir)},
            )
            print(json.dumps({"ok": True, "run_dir": str(run_dir), "last_checkpoint": str(run_dir / "last_checkpoint.pt")}, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def inspect_pairs(args: argparse.Namespace, ddp: DistributedContext) -> None:
    if not ddp.is_main:
        return
    root = Path(args.root).expanduser().resolve()
    paths = WTSPaths(root)
    validate_target_contract(paths, int(args.rank_horizon))
    split = read_json(Path(args.fold_path).expanduser().resolve() if args.fold_path else paths.folds / f"{args.fold_id}.json")
    pair_rows = read_split_rows(paths, split, str(args.pair_train_split), int(args.max_train_dates))
    scaler = load_or_fit_scaler(args, paths, pair_rows, ddp)
    dataset = build_same_date_pair_dataset(
        args,
        root=root,
        rows=pair_rows,
        scaler=scaler,
        bad_per_good=int(args.bad_per_good),
        up_top1_bad_draws=int(args.up_top1_bad_draws),
        up_top10_bad_draws=int(args.up_top10_bad_draws),
        head_hard_negative_top_rank=int(args.head_hard_negative_top_rank),
        gap_free_top_rank=int(args.gap_free_top_rank),
        class_balance_scope=str(args.class_balance_scope),
        seed=20260720,
    )
    sample_count = min(96, len(dataset))
    loader = DataLoader(dataset, batch_size=sample_count, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    sample_metadata = [dataset.pair_metadata(index) for index in range(sample_count)]
    if isinstance(dataset, ExecutionUtilitySameDatePairDataset):
        output = {
            "pair_mode": "execution_utility_all_up",
            "pairs_per_epoch": len(dataset),
            "stats": dataset.stats,
            "smoke_pair_rows": int(batch[0].shape[0]),
            "same_date": bool(torch.equal(batch[9], batch[10])),
            "strict_net_utility_gap": bool(torch.all(batch[7] < batch[6] - float(args.min_return_gap))),
            "sample_good_net_utility_mean": float(batch[6].float().mean()),
            "sample_bad_net_utility_mean": float(batch[7].float().mean()),
            "sample_bad_pool_mean": float(np.mean([metadata["eligible_bad_pool"] for metadata in sample_metadata])),
            "good_class_counts": {label: int((batch[8] == index).sum()) for index, label in enumerate(TREND_LABELS)},
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)
        return
    if isinstance(dataset, CostAwareTailPairDataset):
        output = {
            "pair_mode": "tail_net",
            "pairs_per_epoch": len(dataset),
            "stats": dataset.stats,
            "smoke_pair_rows": int(batch[0].shape[0]),
            "same_date": bool(torch.equal(batch[9], batch[10])),
            "strict_bad_lower": bool(torch.all(batch[7] < batch[6])),
            "chosen_return_floor_respected": bool(torch.all(batch[6] >= float(args.tail_min_return))),
            "gap_rule_respected": bool(torch.all(batch[7] < batch[6] - float(args.min_return_gap))),
            "chosen_rank_in_range": bool(
                all(1 <= metadata["good_global_rank"] <= int(args.tail_positive_rank) for metadata in sample_metadata)
            ),
            "rejected_rank_in_range": bool(
                all(
                    int(args.tail_bad_min_rank) <= metadata["bad_global_rank"] <= int(args.tail_bad_max_rank)
                    for metadata in sample_metadata
                )
            ),
            "sample_good_return_mean": float(batch[6].float().mean()),
            "sample_bad_return_mean": float(batch[7].float().mean()),
            "sample_good_rank_mean": float(np.mean([metadata["good_global_rank"] for metadata in sample_metadata])),
            "sample_bad_rank_mean": float(np.mean([metadata["bad_global_rank"] for metadata in sample_metadata])),
            "sample_bad_pool_mean": float(np.mean([metadata["eligible_bad_pool"] for metadata in sample_metadata])),
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)
        return
    if isinstance(dataset, GapBucketQualityPairDataset):
        output = {
            "pair_mode": "bounded_gap_quality",
            "pairs_per_epoch": len(dataset),
            "stats": dataset.stats,
            "smoke_pair_rows": int(batch[0].shape[0]),
            "same_date": bool(torch.equal(batch[9], batch[10])),
            "return_gap_bucket_respected": bool(
                all(
                    dataset.return_gap_in_bucket(
                        metadata["return_gap"],
                        gap_return_min=float(args.gap_return_min),
                        gap_return_max=float(args.gap_return_max),
                    )
                    for metadata in sample_metadata
                )
            ),
            "quality_gap_respected": bool(
                all(
                    metadata["quality_gap"] >= float(args.gap_quality_min_difference) for metadata in sample_metadata
                )
            ),
            "good_class_counts": {label: int((batch[8] == index).sum()) for index, label in enumerate(TREND_LABELS)},
            "sample_return_gap_mean": float(
                np.mean([metadata["return_gap"] for metadata in sample_metadata])
            ),
            "sample_quality_gap_mean": float(
                np.mean([metadata["quality_gap"] for metadata in sample_metadata])
            ),
            "sample_good_return_mean": float(batch[6].float().mean()),
            "sample_bad_return_mean": float(batch[7].float().mean()),
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)
        return
    if isinstance(dataset, MultiTaskSameDatePairDataset):
        if dataset.pair_contract == "shared_directional":
            directions = batch[11]
            masks = batch[12]
            output = {
                "pair_mode": "multitask_shared_directional",
                "pair_contract": dataset.pair_contract,
                "pairs_per_epoch": len(dataset),
                "stats": dataset.stats,
                "smoke_pair_rows": int(batch[0].shape[0]),
                "same_date": bool(torch.equal(batch[9], batch[10])),
                "return_pair_ordering_respected": bool(torch.all(batch[7] < batch[6])),
                "head_label_counts": {
                    head: {
                        "valid": int(masks[:, index].sum()),
                        "positive": int(((directions[:, index] > 0) & masks[:, index]).sum()),
                        "negative": int(((directions[:, index] < 0) & masks[:, index]).sum()),
                        "masked": int((~masks[:, index]).sum()),
                    }
                    for index, head in enumerate(MULTITASK_SCORE_HEADS)
                },
                "risk_label_gap_respected": all(
                    not metadata["head_masks"][head]
                    or abs(
                        metadata["good_risk_percentiles"][head] - metadata["bad_risk_percentiles"][head]
                    )
                    >= float(args.multitask_risk_min_percentile_gap)
                    for metadata in sample_metadata
                    for head in ("sharpe", "drawdown")
                ),
                "sample_risk_percentile_abs_gap_mean": {
                    head: float(
                        np.mean(
                            [
                                abs(
                                    metadata["good_risk_percentiles"][head]
                                    - metadata["bad_risk_percentiles"][head]
                                )
                                for metadata in sample_metadata
                            ]
                        )
                    )
                    for head in ("sharpe", "drawdown")
                },
            }
            print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)
            return
        task_metadata = {head: [] for head in MULTITASK_SCORE_HEADS}
        for metadata in sample_metadata:
            task_metadata[str(metadata["task"])].append(metadata)
        return_pairs = task_metadata["return"]
        risk_pairs = [*task_metadata["sharpe"], *task_metadata["drawdown"]]
        output = {
            "pair_mode": "multitask_return_then_risk",
            "pairs_per_epoch": len(dataset),
            "stats": dataset.stats,
            "smoke_pair_rows": int(batch[0].shape[0]),
            "same_date": bool(torch.equal(batch[9], batch[10])),
            "task_counts": {
                head: int((batch[11] == index).sum()) for index, head in enumerate(MULTITASK_SCORE_HEADS)
            },
            "return_pair_gap_respected": bool(
                return_pairs
                and all(
                    metadata["bad_return"] < metadata["good_return"] - float(args.min_return_gap)
                    for metadata in return_pairs
                )
            ),
            "risk_return_similarity_respected": bool(
                risk_pairs
                and all(0.0 <= metadata["return_gap"] < float(args.multitask_risk_return_gap_max) for metadata in risk_pairs)
            ),
            "risk_metric_orientation_respected": bool(
                risk_pairs
                and all(
                    metadata["metric_percentile_gap"] >= float(args.multitask_risk_min_percentile_gap)
                    for metadata in risk_pairs
                )
            ),
            "sample_return_gap_mean_by_task": {
                head: (
                    float(np.mean([abs(metadata["good_return"] - metadata["bad_return"]) for metadata in values]))
                    if values
                    else None
                )
                for head, values in task_metadata.items()
            },
            "sample_metric_percentile_gap_mean": {
                head: (
                    float(np.mean([metadata["metric_percentile_gap"] for metadata in values])) if values and head != "return" else None
                )
                for head, values in task_metadata.items()
            },
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)
        return
    daily_base_good_counts = {
        day.date: {
            "up": int(len(day.up_positions)),
            "range": int(len(day.range_positions)),
            "down": int(len(day.down_positions)),
        }
        for day in dataset._days
    }
    if dataset.class_balance_scope == "global":
        global_pair_counts = {
            "up": int(dataset.stats["up_pairs_per_epoch"]),
            "range": int(dataset.stats["range_rows_sampled_per_epoch"]),
            "down": int(dataset.stats["down_rows_sampled_per_epoch"]),
        }
        daily_pair_counts = {day.date: {"up": int(day.up_bad_draw_ends[-1])} for day in dataset._days}
        base_good_balanced = len(set(global_pair_counts.values())) == 1
        pair_balanced = base_good_balanced
    else:
        global_pair_counts = {
            label: int(sum(counts[label] for counts in daily_base_good_counts.values())) for label in TREND_LABELS
        }
        daily_pair_counts = {
            day.date: {
                "up": int(day.up_bad_draw_ends[-1]),
                "range": int(len(day.up_positions) * dataset.bad_per_good),
                "down": int(len(day.up_positions) * dataset.bad_per_good),
            }
            for day in dataset._days
        }
        base_good_balanced = len(set(global_pair_counts.values())) == 1
        pair_balanced = all(len(set(counts.values())) == 1 for counts in daily_pair_counts.values())
    fanout_preview = {
        day.date: {
            "top1_bad_draws": int(day.up_bad_draw_counts[0]),
            "top2_to_10_bad_draws": [int(value) for value in day.up_bad_draw_counts[1 : min(10, len(day.up_bad_draw_counts))]],
            "up_global_ranks": [dataset._global_rank(day, int(position)) for position in day.up_positions[:10]],
            "up_pairs": int(day.up_bad_draw_ends[-1]),
        }
        for day in dataset._days[:5]
    }
    first_day = dataset._days[0]
    first_top1_draws = int(first_day.up_bad_draw_counts[0])
    first_top1_bad_positions = [dataset.pair_metadata(index)["bad_position"] for index in range(first_top1_draws)]
    first_top1_bad_pool = len(dataset._bad_positions(first_day, 0, int(first_day.up_positions[0])))
    first_top1_metadata = dataset.pair_metadata(0)
    output = {
        "pairs_per_epoch": len(dataset),
        "stats": dataset.stats,
        "label_mode": dataset.label_mode,
        "smoke_pair_rows": int(batch[0].shape[0]),
        "same_date": bool(torch.equal(batch[9], batch[10])),
        "strict_bad_lower": bool(torch.all(batch[7] < batch[6])) if dataset.label_mode == "endpoint" else None,
        "raw_endpoint_bad_lower": bool(torch.all(batch[7] < batch[6])) if dataset.label_mode == "path_quality" else None,
        "gap_rule_respected": bool(
            all(
                metadata["bad_pair_value"] < metadata["good_pair_value"] - float(metadata["min_pair_gap"])
                if metadata["requires_min_return_gap"]
                else metadata["bad_pair_value"] < metadata["good_pair_value"]
                for metadata in sample_metadata
            )
        ),
        "first_batch_class_counts": {label: int((batch[8] == index).sum()) for index, label in enumerate(TREND_LABELS)},
        "base_good_rows_exact_1_over_3": bool(base_good_balanced),
        "pair_rows_exact_1_over_3": bool(pair_balanced),
        "global_pair_label_counts": global_pair_counts,
        "daily_base_good_counts_preview": dict(list(daily_base_good_counts.items())[:5]),
        "daily_pair_counts_preview": dict(list(daily_pair_counts.items())[:5]),
        "up_head_fanout_preview": fanout_preview,
        "first_day_top1_bad_diversity": {
            "draws": first_top1_draws,
            "unique_bad_positions": int(len(set(first_top1_bad_positions))),
            "eligible_bad_pool": first_top1_bad_pool,
            "good_global_rank": int(first_top1_metadata["good_global_rank"]),
            "bad_global_rank_min": int(min(dataset._global_rank(first_day, position) for position in first_top1_bad_positions)),
            "bad_global_rank_max": int(max(dataset._global_rank(first_day, position) for position in first_top1_bad_positions)),
            "uses_top30_hard_pool": bool(first_top1_metadata["uses_top30_hard_pool"]),
        },
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    ddp = init_distributed(args)
    try:
        set_seed(20260720 + ddp.rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if args.command == "inspect-pairs":
            inspect_pairs(args, ddp)
        else:
            train(args, ddp)
        barrier(ddp)
        return 0
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    raise SystemExit(main())
