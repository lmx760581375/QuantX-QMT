"""Reward 训练和推理共享的分布式运行时。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .common import (
    WTSPaths,
    fit_wts_feature_scaler,
    load_scaler,
    resolve_device,
    save_scaler,
)


class DistributedContext:
    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device,
        backend: str | None,
    ):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.device = device
        self.backend = backend

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def to_json(self) -> dict:
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "backend": self.backend,
        }


def init_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = args.ddp != "off" and world_size > 1
    if not enabled:
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=resolve_device(args.device),
            backend=None,
        )
    if args.device == "mps" or (
        args.device == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("MPS inference is single-process only; use --ddp off or omit torchrun")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cpu":
        backend = "gloo"
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=backend)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def cleanup_distributed(ddp: DistributedContext) -> None:
    if ddp.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ddp: DistributedContext) -> None:
    if not ddp.enabled:
        return
    if ddp.device.type == "cuda":
        dist.barrier(device_ids=[ddp.local_rank])
    else:
        dist.barrier()


def broadcast_object(value, ddp: DistributedContext):
    if not ddp.enabled:
        return value
    values = [value if ddp.is_main else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def prepare_run_dir(paths: WTSPaths, args: argparse.Namespace, ddp: DistributedContext) -> Path:
    run_dir = None
    if ddp.is_main:
        resume_run_dir = str(getattr(args, "resume_run_dir", "")).strip()
        if resume_run_dir:
            run_dir = Path(resume_run_dir).expanduser().resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        else:
            timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
            run_dir = paths.runs / f"{args.run_name}-{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_text = broadcast_object(str(run_dir) if run_dir is not None else None, ddp)
    if run_dir_text is None:
        raise RuntimeError("rank0 did not broadcast run_dir")
    resolved = Path(run_dir_text)
    log_path = resolved / "logs" / "train_metrics.jsonl" if ddp.is_main else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = broadcast_object(str(log_path) if log_path is not None else None, ddp)
    if log_text is None:
        raise RuntimeError("rank0 did not broadcast training log path")
    args.training_log_path = log_text
    return resolved


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def load_or_fit_scaler(
    args: argparse.Namespace,
    paths: WTSPaths,
    train_rows: pd.DataFrame,
    ddp: DistributedContext,
):
    paths.scalers.mkdir(parents=True, exist_ok=True)
    if str(getattr(args, "scaler_path", "")):
        scaler_path = Path(str(args.scaler_path)).expanduser().resolve()
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing --scaler-path: {scaler_path}")
        return load_scaler(scaler_path)
    scaler_path = paths.scalers / f"{args.fold_id}_{args.input_mode}_scaler.json"
    if ddp.is_main:
        if scaler_path.exists():
            load_scaler(scaler_path)
        else:
            scaler = fit_wts_feature_scaler(
                kronos_root=args.kronos_root,
                experiment_root=args.root,
                rows=train_rows,
                max_samples=int(args.max_scaler_samples),
                seed=20260717,
                target_kind=args.target_kind,
            )
            save_scaler(scaler_path, scaler)
    barrier(ddp)
    return load_scaler(scaler_path)


def create_tensorboard_writer(run_dir: Path, args: argparse.Namespace, ddp: DistributedContext):
    if not bool(args.tensorboard):
        raise RuntimeError("TensorBoard is mandatory for formal training")
    payload = None
    writer = None
    if ddp.is_main:
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from torch.utils.tensorboard import SummaryWriter

            log_dir = run_dir / "tensorboard"
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_text("config/args", json.dumps(vars(args), ensure_ascii=False, sort_keys=True, indent=2), 0)
            writer.flush()
            payload = {"status": "enabled", "log_dir": str(log_dir)}
        except Exception as exc:
            payload = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    payload = broadcast_object(payload, ddp)
    if payload is None or payload.get("status") != "enabled":
        raise RuntimeError(f"TensorBoard initialization failed: {payload}")
    return writer, payload


def _gpu_memory_metrics(ddp: DistributedContext) -> dict:
    if ddp.device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory": {"device": str(ddp.device), "available": False}}
    return {
        "gpu_memory": {
            "device": str(ddp.device),
            "allocated_bytes": int(torch.cuda.memory_allocated(ddp.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(ddp.device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(ddp.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(ddp.device)),
        }
    }


def append_training_log(args: argparse.Namespace, ddp: DistributedContext, event: dict) -> None:
    if not ddp.is_main:
        return
    path_text = str(getattr(args, "training_log_path", "")).strip()
    if not path_text:
        return
    payload = {
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        **event,
        **_gpu_memory_metrics(ddp),
    }
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    *,
    min_lr: float = 0.0,
    decay_steps: int | None = None,
):
    total_steps = max(1, int(total_steps))
    decay_steps = total_steps if decay_steps is None else int(decay_steps)
    if not 1 <= decay_steps <= total_steps:
        raise ValueError("decay_steps must be within [1, total_steps]")
    warmup_steps = min(decay_steps, max(0, int(round(decay_steps * float(warmup_ratio)))))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if float(min_lr) < 0.0 or any(float(min_lr) > base_lr for base_lr in base_lrs):
        raise ValueError("min_lr must be non-negative and no greater than every optimizer learning rate")

    def scale(step: int, floor: float) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return float(floor + (1.0 - floor) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step, floor=float(min_lr) / base_lr: scale(step, floor)
            for base_lr in base_lrs
        ],
    )
