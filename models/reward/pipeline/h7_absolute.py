"""H7 absolute-return labels and exact same-date good x bad supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import WTSPaths, WeakToStrongPathDataset, target_memmap, write_json


LABEL_KIND = "quantx_h7_absolute_full_pair_labels_v1"
LABEL_COLUMNS = ("is_good", "h7_return")


@dataclass(frozen=True)
class H7AbsoluteGoodBadConfig:
    rank_horizon: int = 7
    good_return_threshold: float = 0.10


@dataclass(frozen=True)
class FullPairDay:
    date: str
    date_code: int
    positions: np.ndarray
    good_mask: np.ndarray
    pair_count: int


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_h7_absolute_label_artifact(
    *,
    experiment_root: str | Path,
    output_dir: str | Path,
    config: H7AbsoluteGoodBadConfig,
    chunk_size: int = 500_000,
    force: bool = False,
) -> dict[str, Any]:
    paths = WTSPaths(Path(experiment_root).expanduser().resolve())
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels_float32.mmap"
    daily_path = output_dir / "daily_counts.parquet"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (labels_path, daily_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite H7 labels without force=True: "
            + ", ".join(map(str, existing))
        )

    target = target_memmap(paths, "abs", mode="r")
    horizon_index = int(config.rank_horizon) - 1
    if horizon_index < 0 or horizon_index >= int(target.shape[1]):
        raise ValueError(
            f"rank_horizon={config.rank_horizon} exceeds target shape={target.shape}"
        )
    parquet = pq.ParquetFile(paths.candidate_index)
    row_count = int(parquet.metadata.num_rows)
    labels = np.memmap(
        labels_path, dtype=np.float32, mode="w+", shape=(row_count, len(LABEL_COLUMNS))
    )
    offset = 0
    daily_parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(
        columns=["row_id", "target_row", "signal_date"],
        batch_size=max(1, int(chunk_size)),
    ):
        frame = batch.to_pandas()
        row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        expected = np.arange(offset, offset + len(frame), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise RuntimeError("H7 label artifact requires contiguous row_id order")
        returns = np.asarray(
            target[frame["target_row"].to_numpy(dtype=np.int64), horizon_index],
            dtype=np.float32,
        )
        valid = np.isfinite(returns)
        good = valid & (
            returns >= np.float32(config.good_return_threshold)
        )
        labels[offset : offset + len(frame), 0] = good.astype(np.float32)
        labels[offset : offset + len(frame), 1] = returns
        daily_parts.append(
            pd.DataFrame(
                {
                    "signal_date": frame["signal_date"].astype(str).to_numpy(),
                    "rows": valid.astype(np.int64),
                    "good": good.astype(np.int64),
                }
            ).groupby("signal_date", sort=False, as_index=False).sum()
        )
        offset += len(frame)
    if offset != row_count:
        raise RuntimeError(f"H7 label row count mismatch: wrote={offset}, expected={row_count}")
    labels.flush()
    daily = pd.concat(daily_parts, ignore_index=True).groupby(
        "signal_date", as_index=False
    ).sum()
    daily["bad"] = daily["rows"] - daily["good"]
    daily["pairs"] = daily["good"] * daily["bad"]
    daily.to_parquet(daily_path, index=False, compression="zstd")
    manifest = {
        "kind": LABEL_KIND,
        "row_count": row_count,
        "columns": list(LABEL_COLUMNS),
        "config": asdict(config),
        "source": {
            "experiment_root": str(paths.root),
            "candidate_index": str(paths.candidate_index),
            "target_meta": str(paths.target_meta),
        },
        "labels": {
            "path": labels_path.name,
            "dtype": "float32",
            "shape": [row_count, len(LABEL_COLUMNS)],
        },
        "daily_counts": {
            "path": daily_path.name,
            "dates": int(len(daily)),
            "good": int(daily["good"].sum()),
            "bad": int(daily["bad"].sum()),
            "pairs": int(daily["pairs"].sum()),
            "zero_pair_dates": int((daily["pairs"] == 0).sum()),
            "pair_mean": float(daily["pairs"].mean()),
            "pair_median": float(daily["pairs"].median()),
            "pair_max": int(daily["pairs"].max()),
        },
    }
    manifest["labels"]["sha256"] = _sha256(labels_path)
    manifest["daily_counts"]["sha256"] = _sha256(daily_path)
    write_json(manifest_path, manifest)
    return manifest


class H7AbsoluteLabelStore:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != LABEL_KIND:
            raise RuntimeError(f"Unexpected H7 label kind: {self.manifest.get('kind')!r}")
        meta = self.manifest["labels"]
        if tuple(self.manifest.get("columns") or ()) != LABEL_COLUMNS:
            raise RuntimeError("H7 label schema does not match the runtime contract")
        path = Path(meta["path"])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        self.row_count = int(self.manifest["row_count"])
        self.values = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(self.row_count, len(LABEL_COLUMNS)),
        )

    def arrays_for_rows(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        row_ids = rows["row_id"].to_numpy(dtype=np.int64)
        values = np.asarray(self.values[row_ids])
        return values[:, 0].astype(bool), values[:, 1].astype(np.float32)


class FullCartesianGoodBadDayDataset(Dataset):
    """Return each date's rows once; every good x bad pair is formed in loss."""

    def __init__(
        self,
        *,
        kronos_root: str | Path,
        experiment_root: str | Path,
        rows: pd.DataFrame,
        labels: H7AbsoluteLabelStore,
        scaler,
        input_mode: str,
        max_open_feature_shards: int,
    ):
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
        is_good, returns = labels.arrays_for_rows(rows)
        self.returns = returns
        self.days: list[FullPairDay] = []
        good_rows = 0
        bad_rows = 0
        zero_pair_dates = 0
        for date_code, (date, group) in enumerate(
            rows.groupby("signal_date", sort=True)
        ):
            positions = group.index.to_numpy(dtype=np.int64)
            valid = np.isfinite(returns[positions])
            positions = positions[valid]
            good_mask = is_good[positions]
            good_count = int(good_mask.sum())
            bad_count = int(len(good_mask) - good_count)
            if not good_count or not bad_count:
                zero_pair_dates += 1
                continue
            good_rows += good_count
            bad_rows += bad_count
            self.days.append(
                FullPairDay(
                    date=str(date),
                    date_code=int(date_code),
                    positions=positions,
                    good_mask=good_mask,
                    pair_count=good_count * bad_count,
                )
            )
        self.pairs_per_epoch = int(sum(day.pair_count for day in self.days))
        self.rows_per_epoch = int(sum(len(day.positions) for day in self.days))
        self.stats = {
            "label_mode": "h7_absolute_full_pair",
            "source_rows": int(len(rows)),
            "days_with_pairs": int(len(self.days)),
            "zero_pair_dates": int(zero_pair_dates),
            "good_rows": int(good_rows),
            "bad_rows": int(bad_rows),
            "rows_per_epoch": self.rows_per_epoch,
            "pairs_per_epoch": self.pairs_per_epoch,
            "pair_contract": "same signal_date; exact Cartesian product of every good and every bad",
        }

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, index: int):
        day = self.days[int(index)]
        samples = [self.base[int(position)] for position in day.positions]
        return (
            torch.stack([sample[0] for sample in samples]),
            torch.stack([sample[1] for sample in samples]),
            torch.stack([sample[2] for sample in samples]),
            torch.from_numpy(day.good_mask.copy()),
            torch.from_numpy(self.returns[day.positions].copy()),
            torch.tensor(day.date_code, dtype=torch.int64),
            torch.tensor(day.pair_count, dtype=torch.int64),
        )


def exact_cartesian_pair_loss(
    logits: torch.Tensor,
    good_mask: torch.Tensor,
    *,
    temperature: float,
    logit_center_weight: float = 0.0,
    block_size: int = 1024,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    logits = logits.float().reshape(-1)
    good_mask = good_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
    good = logits[good_mask]
    bad = logits[~good_mask]
    if not len(good) or not len(bad):
        raise ValueError("Exact Cartesian loss requires at least one good and one bad")
    loss_sum = logits.new_zeros(())
    correct = logits.new_zeros((), dtype=torch.int64)
    margin_sum = logits.new_zeros(())
    pair_count = int(len(good) * len(bad))
    for start in range(0, len(good), max(1, int(block_size))):
        margin = good[start : start + int(block_size), None] - bad[None, :]
        loss_sum = loss_sum + F.softplus(-margin / float(temperature)).sum()
        correct = correct + (margin.detach() > 0.0).sum()
        margin_sum = margin_sum + margin.detach().sum()
    loss = loss_sum / float(pair_count)
    if float(logit_center_weight):
        loss = loss + float(logit_center_weight) * logits.mean().square()
    return loss, {
        "pairs": pair_count,
        "correct": correct,
        "margin_sum": margin_sum,
        "good_count": int(len(good)),
        "bad_count": int(len(bad)),
    }


def balanced_day_shards(
    pair_counts: Iterable[int], world_size: int
) -> list[list[int]]:
    """Greedily assign whole dates to ranks while preserving exact global coverage."""
    if int(world_size) < 1:
        raise ValueError("world_size must be positive")
    counts = [int(value) for value in pair_counts]
    shards = [[] for _ in range(int(world_size))]
    loads = [0 for _ in range(int(world_size))]
    for index in sorted(range(len(counts)), key=lambda item: (-counts[item], item)):
        rank = min(range(int(world_size)), key=lambda item: (loads[item], item))
        shards[rank].append(index)
        loads[rank] += counts[index]
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "data" / "h7_absolute_good_bad_v1"
    )
    if args.command == "build":
        payload = build_h7_absolute_label_artifact(
            experiment_root=root,
            output_dir=output_dir,
            config=H7AbsoluteGoodBadConfig(),
            chunk_size=int(args.chunk_size),
            force=bool(args.force),
        )
    else:
        store = H7AbsoluteLabelStore(output_dir / "manifest.json")
        label_path = Path(store.manifest["labels"]["path"])
        if not label_path.is_absolute():
            label_path = store.manifest_path.parent / label_path
        daily_path = Path(store.manifest["daily_counts"]["path"])
        if not daily_path.is_absolute():
            daily_path = store.manifest_path.parent / daily_path
        payload = {
            "kind": store.manifest["kind"],
            "row_count": store.row_count,
            "labels_sha256_match": _sha256(label_path)
            == store.manifest["labels"].get("sha256"),
            "daily_counts_sha256_match": _sha256(daily_path)
            == store.manifest["daily_counts"].get("sha256"),
            "pairs": int(store.manifest["daily_counts"]["pairs"]),
        }
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
