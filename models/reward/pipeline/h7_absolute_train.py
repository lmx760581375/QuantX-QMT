"""Train the frozen-AE Reward Transformer with exact H7 good x bad pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .h7_absolute import (
    FullCartesianGoodBadDayDataset,
    H7AbsoluteLabelStore,
    balanced_day_shards,
    exact_cartesian_pair_loss,
)
from .train_impl import (
    FrozenAERewardTransformer,
    RewardTransformerConfig,
    _forward_reward,
    load_frozen_ae,
    read_split_rows,
)
from .runtime import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    count_parameters,
    init_distributed,
    make_scheduler,
    unwrap_model,
)
from .common import (
    EXPERIMENT_ROOT,
    REPO_ROOT,
    WTSPaths,
    WeakToStrongPathDataset,
    load_scaler,
    set_seed,
    state_dict_without_module,
    write_json,
)


DEFAULT_ROOT = EXPERIMENT_ROOT / "market_all_close2close_balanced_v2"
DEFAULT_KRONOS_ROOT = REPO_ROOT / "workdirs" / "feature_store"
DEFAULT_AE = REPO_ROOT / "weights" / "reward_v1" / "ae_epoch020.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--kronos-root", default=str(DEFAULT_KRONOS_ROOT))
    parser.add_argument("--labels", required=True)
    parser.add_argument("--run-name", default="h7-absolute-full-pair")
    parser.add_argument("--fold-id", default="pre2020_eval2020_2026")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--scaler-path", default="")
    parser.add_argument("--ae-model-size", default="ae_wide_08")
    parser.add_argument("--ae-checkpoint", default=str(DEFAULT_AE))
    parser.add_argument("--input-mode", default="raw_relative", choices=["raw_relative", "relative_only"])
    parser.add_argument("--rank-horizon", type=int, default=7)
    parser.add_argument("--good-return-threshold", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--reward-d-model", type=int, default=768)
    parser.add_argument("--reward-layers", type=int, default=7)
    parser.add_argument("--reward-heads", type=int, default=12)
    parser.add_argument("--reward-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--reward-dropout", type=float, default=0.05)
    parser.add_argument("--amp-dtype", default="bf16", choices=["off", "bf16", "fp16"])
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit-center-weight", type=float, default=0.01)
    parser.add_argument("--pair-block-size", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--dates-per-step", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-open-feature-shards", type=int, default=16)
    parser.add_argument("--max-train-dates", type=int, default=0)
    parser.add_argument("--max-validation-dates", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ddp", default="auto", choices=["auto", "off"])
    return parser.parse_args()


def _run_dir(paths: WTSPaths, args: argparse.Namespace, ddp) -> Path:
    value = None
    if ddp.is_main:
        value = paths.runs / (
            f"{args.run_name}-{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')}"
        )
        value.mkdir(parents=True, exist_ok=False)
    resolved = broadcast_object(str(value) if value is not None else None, ddp)
    if resolved is None:
        raise RuntimeError("rank0 did not broadcast run directory")
    return Path(resolved)


def _score_day(ae, model, batch, args, device: torch.device) -> torch.Tensor:
    x_ts, x_market, x_candidate = batch[:3]
    outputs = []
    for start in range(0, len(x_ts), int(args.score_batch_size)):
        stop = start + int(args.score_batch_size)
        outputs.append(
            _forward_reward(
                ae,
                model,
                x_ts[start:stop].to(device, dtype=torch.float32, non_blocking=True),
                x_market[start:stop].to(device, dtype=torch.float32, non_blocking=True),
                x_candidate[start:stop].to(device, dtype=torch.float32, non_blocking=True),
                args,
                device,
            )
        )
    return torch.cat(outputs)


def _dummy_loss(ae, model, dataset, args, device: torch.device) -> torch.Tensor:
    sample = dataset.base[0]
    logits = _forward_reward(
        ae,
        model,
        sample[0].unsqueeze(0).to(device),
        sample[1].unsqueeze(0).to(device),
        sample[2].unsqueeze(0).to(device),
        args,
        device,
    )
    return logits.float().sum() * 0.0


def train_epoch(dataset, day_indices, ae, model, optimizer, scheduler, args, ddp, epoch: int):
    rng = np.random.default_rng(20260928 + int(epoch) * 1009 + ddp.rank)
    order = list(day_indices)
    rng.shuffle(order)
    lengths = [len(value) for value in balanced_day_shards(
        [day.pair_count for day in dataset.days], ddp.world_size
    )]
    dates_per_step = max(1, int(args.dates_per_step))
    steps = max(math.ceil(length / dates_per_step) for length in lengths)
    gb_normalizer = dataset.pairs_per_epoch / max(1, steps * ddp.world_size)
    loss_sum = 0.0
    pairs = 0
    correct = 0
    margin_sum = 0.0
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        selected = order[
            step * dates_per_step : min((step + 1) * dates_per_step, len(order))
        ]
        if selected:
            loss = None
            for day_index in selected:
                batch = dataset[day_index]
                logits = _score_day(ae, model, batch, args, ddp.device)
                good_mask = batch[3].to(ddp.device)
                mean_loss, stats = exact_cartesian_pair_loss(
                    logits,
                    good_mask,
                    temperature=float(args.temperature),
                    logit_center_weight=float(args.logit_center_weight),
                    block_size=int(args.pair_block_size),
                )
                pair_count = int(stats["pairs"])
                weighted_loss = mean_loss * (pair_count / max(gb_normalizer, 1.0))
                loss = weighted_loss if loss is None else loss + weighted_loss
                loss_sum += float(mean_loss.detach()) * pair_count
                pairs += pair_count
                correct += int(stats["correct"])
                margin_sum += float(stats["margin_sum"])
            if loss is None:
                raise RuntimeError("Non-empty date pack did not produce a loss")
        else:
            loss = _dummy_loss(ae, model, dataset, args, ddp.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        scheduler.step()
    values = torch.tensor(
        [
            loss_sum,
            pairs,
            correct,
            margin_sum,
            time.perf_counter() - started,
        ],
        dtype=torch.float64,
        device=ddp.device,
    )
    if ddp.enabled:
        dist.all_reduce(values[:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[4:], op=dist.ReduceOp.MAX)
    (
        loss_sum,
        pairs,
        correct,
        margin_sum,
        elapsed,
    ) = values.cpu().tolist()
    return {
        "loss": loss_sum / max(pairs, 1),
        "pair_count": int(pairs),
        "pair_accuracy": correct / max(pairs, 1),
        "margin_mean": margin_sum / max(pairs, 1),
        "pairs_per_second": pairs / max(elapsed, 1e-9),
        "optimizer_steps": steps,
        "lr_end": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def evaluate_pairs(dataset, ae, model, args, device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    pairs = 0
    correct = 0
    for index in range(len(dataset)):
        batch = dataset[index]
        logits = _score_day(ae, model, batch, args, device)
        loss, stats = exact_cartesian_pair_loss(
            logits,
            batch[3].to(device),
            temperature=float(args.temperature),
            logit_center_weight=float(args.logit_center_weight),
            block_size=int(args.pair_block_size),
        )
        count = int(stats["pairs"])
        loss_sum += float(loss) * count
        pairs += count
        correct += int(stats["correct"])
    return {
        "loss": loss_sum / max(pairs, 1),
        "pair_count": pairs,
        "pair_accuracy": correct / max(pairs, 1),
    }


@torch.no_grad()
def evaluate_cross_sectional(rows, dataset, labels, ae, model, args, device):
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    scores = []
    for x_ts, x_market, x_candidate, _ in loader:
        logits = _forward_reward(
            ae,
            model,
            x_ts.to(device),
            x_market.to(device),
            x_candidate.to(device),
            args,
            device,
        )
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    good, returns = labels.arrays_for_rows(rows)
    frame = pd.DataFrame(
        {
            "signal_date": rows["signal_date"].astype(str).to_numpy(),
            "score": np.concatenate(scores),
            "return": returns,
            "good": good,
        }
    )
    result = {"rows": len(frame), "dates": int(frame.signal_date.nunique()), "topk": {}}
    rankics = []
    totals = {
        k: {"return": 0.0, "base": 0.0, "good": 0, "base_good": 0.0, "count": 0}
        for k in (1, 5, 10)
    }
    for _, day in frame.groupby("signal_date", sort=False):
        if day.score.nunique() > 1 and day["return"].nunique() > 1:
            rankics.append(float(day.score.corr(day["return"], method="spearman")))
        order = np.argsort(-day.score.to_numpy(), kind="mergesort")
        for k, values in totals.items():
            selected = day.iloc[order[: min(k, len(day))]]
            values["return"] += float(selected["return"].sum())
            values["base"] += float(day["return"].mean()) * len(selected)
            values["good"] += int(selected.good.sum())
            values["base_good"] += float(day.good.mean()) * len(selected)
            values["count"] += len(selected)
    result["rankic_mean"] = float(np.nanmean(rankics))
    for k, values in totals.items():
        count = max(1, values["count"])
        precision = values["good"] / count
        base_rate = values["base_good"] / count
        result["topk"][f"top{k}"] = {
            "selected_avg_return": values["return"] / count,
            "base_avg_return": values["base"] / count,
            "excess_avg_return": (values["return"] - values["base"]) / count,
            "good_precision": precision,
            "base_good_rate": base_rate,
            "good_lift": precision / base_rate if base_rate else None,
        }
    return result


def main() -> int:
    args = parse_args()
    if int(args.rank_horizon) != 7 or not math.isclose(
        float(args.good_return_threshold), 0.10
    ):
        raise ValueError("This v1 contract is fixed to H7 return >= 10%")
    ddp = init_distributed(args)
    try:
        set_seed(20260928 + ddp.rank)
        root = Path(args.root).expanduser().resolve()
        paths = WTSPaths(root)
        fold = json.loads((paths.folds / f"{args.fold_id}.json").read_text())
        train_rows = read_split_rows(paths, fold, args.train_split)
        val_rows = read_split_rows(paths, fold, args.validation_split)
        labels = H7AbsoluteLabelStore(args.labels)
        label_config = dict(labels.manifest.get("config") or {})
        if int(label_config.get("rank_horizon", -1)) != int(args.rank_horizon):
            raise RuntimeError("H7 label artifact rank_horizon does not match training config")
        if not math.isclose(
            float(label_config.get("good_return_threshold", float("nan"))),
            float(args.good_return_threshold),
        ):
            raise RuntimeError("H7 label artifact good_return_threshold does not match training config")
        scaler_path = (
            Path(args.scaler_path).expanduser().resolve()
            if args.scaler_path
            else paths.scalers / f"{args.fold_id}_{args.input_mode}_scaler.json"
        )
        scaler = load_scaler(scaler_path)
        train_data = FullCartesianGoodBadDayDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=train_rows,
            labels=labels,
            scaler=scaler,
            input_mode=args.input_mode,
            max_open_feature_shards=args.max_open_feature_shards,
        )
        val_pair_data = FullCartesianGoodBadDayDataset(
            kronos_root=args.kronos_root,
            experiment_root=root,
            rows=val_rows,
            labels=labels,
            scaler=scaler,
            input_mode=args.input_mode,
            max_open_feature_shards=args.max_open_feature_shards,
        )
        if args.max_train_dates:
            train_data.days = train_data.days[: int(args.max_train_dates)]
            train_data.pairs_per_epoch = sum(day.pair_count for day in train_data.days)
        if args.max_validation_dates:
            val_pair_data.days = val_pair_data.days[: int(args.max_validation_dates)]
            val_pair_data.pairs_per_epoch = sum(day.pair_count for day in val_pair_data.days)
            dates = {day.date for day in val_pair_data.days}
            val_rows = val_rows[val_rows.signal_date.astype(str).isin(dates)].reset_index(drop=True)
        shards = balanced_day_shards(
            [day.pair_count for day in train_data.days],
            ddp.world_size,
        )
        run_dir = _run_dir(paths, args, ddp)
        ae, ae_cfg, ae_path = load_frozen_ae(args, ddp.device)
        reward_cfg = RewardTransformerConfig(
            cond_dim=int(ae_cfg.latent_dim),
            cond_tokens=int(ae_cfg.latent_tokens),
            d_model=int(args.reward_d_model),
            n_layers=int(args.reward_layers),
            n_heads=int(args.reward_heads),
            mlp_ratio=float(args.reward_mlp_ratio),
            dropout=float(args.reward_dropout),
        )
        reward = FrozenAERewardTransformer(reward_cfg).to(ddp.device)
        model: nn.Module = reward
        if ddp.enabled:
            kwargs = (
                {"device_ids": [ddp.local_rank], "output_device": ddp.local_rank}
                if ddp.device.type == "cuda"
                else {}
            )
            model = DistributedDataParallel(reward, **kwargs)
        optimizer = torch.optim.AdamW(
            reward.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            betas=(0.9, 0.95),
        )
        if int(args.dates_per_step) < 1:
            raise ValueError("--dates-per-step must be positive")
        steps_per_epoch = max(
            math.ceil(len(shard) / int(args.dates_per_step)) for shard in shards
        )
        scheduler = make_scheduler(
            optimizer,
            steps_per_epoch * int(args.epochs),
            float(args.warmup_ratio),
            min_lr=float(args.min_lr),
        )
        if ddp.is_main:
            print(
                json.dumps(
                    {
                        "event": "run_start",
                        "run_dir": str(run_dir),
                        "train": train_data.stats,
                        "active_train_pairs": train_data.pairs_per_epoch,
                        "validation_pairs": val_pair_data.pairs_per_epoch,
                        "rank_pair_loads": [
                            sum(train_data.days[index].pair_count for index in shard)
                            for shard in shards
                        ],
                        "dates_per_step": int(args.dates_per_step),
                        "parameters": {
                            "frozen_ae": count_parameters(ae),
                            "reward": count_parameters(reward),
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        history = []
        val_market = None
        if ddp.is_main:
            val_market = WeakToStrongPathDataset(
                kronos_root=args.kronos_root,
                experiment_root=root,
                rows=val_rows,
                scaler=scaler,
                target_kind="abs",
                input_mode=args.input_mode,
                max_open_shards=args.max_open_feature_shards,
            )
        for epoch in range(1, int(args.epochs) + 1):
            metrics = train_epoch(
                train_data,
                shards[ddp.rank],
                ae,
                model,
                optimizer,
                scheduler,
                args,
                ddp,
                epoch,
            )
            row = {"epoch": epoch, "train": metrics}
            barrier(ddp)
            if ddp.is_main:
                row["validation_pairs"] = evaluate_pairs(
                    val_pair_data, ae, unwrap_model(model), args, ddp.device
                )
                row["validation_cross_sectional"] = evaluate_cross_sectional(
                    val_rows,
                    val_market,
                    labels,
                    ae,
                    unwrap_model(model),
                    args,
                    ddp.device,
                )
                checkpoint = {
                    "kind": "wts_frozen_ae_reward_transformer_h7_absolute_full_pair_v1",
                    "epoch": epoch,
                    "args": vars(args),
                    "ae_checkpoint": str(ae_path),
                    "ae_model_config": asdict(ae_cfg),
                    "reward_config": asdict(reward_cfg),
                    "reward_state": state_dict_without_module(unwrap_model(model)),
                }
                torch.save(checkpoint, run_dir / f"reward_epoch_{epoch:03d}.pt")
                torch.save(checkpoint, run_dir / "last_checkpoint.pt")
                print(json.dumps(row, ensure_ascii=False), flush=True)
            history.append(row)
            barrier(ddp)
        if ddp.is_main:
            write_json(
                run_dir / "train_manifest.json",
                {
                    "kind": "wts_h7_absolute_full_pair_train_manifest_v1",
                    "status": "trained",
                    "args": vars(args),
                    "label_manifest": str(Path(args.labels).resolve()),
                    "pair_contract": "every same-date good x bad pair exactly once per epoch",
                    "good_good_contract": "disabled",
                    "train_dataset": train_data.stats,
                    "active_train_pairs": train_data.pairs_per_epoch,
                    "validation_pairs": val_pair_data.pairs_per_epoch,
                    "history": history,
                },
            )
        return 0
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    raise SystemExit(main())
