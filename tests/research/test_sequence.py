"""Raw sequence windows and optional Torch adapter tests."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from quantx.core.research.sequence import RawWindowFeatureBuilder
from quantx.core.research.trainer import TorchTrainer


def sample_frame():
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=6), ["A", "B"]],
        names=["signal_time", "instrument"],
    )
    return pd.DataFrame(
        {
            "x1": np.arange(len(index), dtype=float),
            "x2": np.arange(len(index), dtype=float) * 2,
            "label": np.arange(len(index), dtype=float) / 10,
        },
        index=index,
    )


def test_raw_window_builder_never_reads_later_sessions():
    frame = sample_frame()
    builder = RawWindowFeatureBuilder(lookback_sessions=3)

    original = builder.transform(frame, ("x1", "x2"))
    changed = frame.copy()
    changed.loc[(pd.Timestamp("2021-01-11"), "A"), "x1"] = 1_000_000
    mutated = builder.transform(changed, ("x1", "x2"))

    target = (pd.Timestamp("2021-01-08"), "A")
    position = original.sample_index.index(target)
    np.testing.assert_array_equal(original.values[position], mutated.values[position])
    assert original.values.shape == (12, 3, 2)


def test_torch_trainer_has_clear_optional_dependency_error():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("Torch is installed in this environment")
    trainer = TorchTrainer(epochs=1, lookback_sessions=2)

    with pytest.raises(RuntimeError, match="torch"):
        trainer.fit(sample_frame(), ("x1", "x2"), "label", seed=7)
