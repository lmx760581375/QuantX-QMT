from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models.reward.pipeline import h7_absolute as module
from models.reward.pipeline.h7_absolute import (
    FullCartesianGoodBadDayDataset,
    H7AbsoluteGoodBadConfig,
    H7AbsoluteLabelStore,
    balanced_day_shards,
    build_h7_absolute_label_artifact,
    exact_cartesian_pair_loss,
)


def test_exact_cartesian_loss_matches_explicit_pairs_and_gradient():
    logits = torch.tensor([1.0, 0.4, -0.2, -0.7], requires_grad=True)
    good_mask = torch.tensor([True, True, False, False])
    loss, stats = exact_cartesian_pair_loss(
        logits,
        good_mask,
        temperature=0.7,
        logit_center_weight=0.03,
        block_size=1,
    )
    explicit_good = logits[good_mask].repeat_interleave(2)
    explicit_bad = logits[~good_mask].repeat(2)
    expected = torch.nn.functional.softplus(
        -(explicit_good - explicit_bad) / 0.7
    ).mean() + 0.03 * logits.mean().square()
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]

    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    assert torch.allclose(grad, expected_grad)
    assert stats["pairs"] == 4
    assert int(stats["correct"]) == 4


def test_balanced_day_shards_cover_every_date_once():
    counts = [100, 90, 50, 30, 20, 10]
    shards = balanced_day_shards(counts, 3)

    assert sorted(index for shard in shards for index in shard) == list(range(len(counts)))
    assert len({index for shard in shards for index in shard}) == len(counts)
    loads = [sum(counts[index] for index in shard) for shard in shards]
    assert max(loads) - min(loads) <= max(counts)


def test_label_builder_and_day_dataset_use_all_good_bad_pairs(monkeypatch, tmp_path):
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "row_id": np.arange(6),
            "target_row": np.arange(6),
            "signal_date": ["2019-01-02"] * 4 + ["2019-01-03"] * 2,
        }
    )
    rows.to_parquet(data / "candidate_index_v1.parquet", index=False)
    targets = np.zeros((6, 7), dtype=np.float32)
    targets[:, 6] = [0.12, 0.10, 0.09, -0.02, 0.20, -0.01]
    target_path = data / "target_abs_path_float32.mmap"
    target_mmap = np.memmap(target_path, dtype=np.float32, mode="w+", shape=targets.shape)
    target_mmap[:] = targets
    target_mmap.flush()
    (data / "target_paths_meta.json").write_text(
        '{"shape":[6,7],"targets":{"abs":"%s"}}' % target_path,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "target_memmap",
        lambda *_args, **_kwargs: np.memmap(
            target_path, dtype=np.float32, mode="r", shape=targets.shape
        ),
    )
    manifest = build_h7_absolute_label_artifact(
        experiment_root=root,
        output_dir=tmp_path / "labels",
        config=H7AbsoluteGoodBadConfig(),
    )

    assert manifest["daily_counts"]["good"] == 3
    assert manifest["daily_counts"]["bad"] == 3
    assert manifest["daily_counts"]["pairs"] == 5
    store = H7AbsoluteLabelStore(tmp_path / "labels" / "manifest.json")
    good, returns = store.arrays_for_rows(rows)
    assert good.tolist() == [True, True, False, False, True, False]
    assert returns[-1] == pytest.approx(-0.01)

    class FakeBase:
        def __init__(self, *, rows, **_):
            self.rows = rows.reset_index(drop=True)

        def __getitem__(self, index):
            value = torch.tensor([float(index)])
            return value, value, value, value

    monkeypatch.setattr(module, "WeakToStrongPathDataset", FakeBase)
    dataset = FullCartesianGoodBadDayDataset(
        kronos_root=tmp_path,
        experiment_root=root,
        rows=rows,
        labels=store,
        scaler=None,
        input_mode="raw_relative",
        max_open_feature_shards=1,
    )

    assert len(dataset) == 2
    assert dataset.pairs_per_epoch == 5
    assert dataset.stats["rows_per_epoch"] == 6
    assert dataset[0][-1].item() == 4
