"""Train-only preprocessing boundary tests."""

import pandas as pd
import pytest

from quantx.core.research.preprocessing import LabelWinsorizer


def test_label_winsorizer_uses_training_thresholds_for_other_splits():
    train = pd.DataFrame({"label": [-100.0, 0.0, 1.0, 2.0, 100.0]})
    holdout = pd.DataFrame({"label": [-1000.0, 1.5, 1000.0]})

    winsorizer = LabelWinsorizer.fit(train["label"], (0.2, 0.8))
    transformed = winsorizer.transform(holdout, "label")

    assert winsorizer.lower_value == pytest.approx(-20.0)
    assert winsorizer.upper_value == pytest.approx(21.6)
    assert transformed["label"].tolist() == pytest.approx([-20.0, 1.5, 21.6])
    assert holdout["label"].tolist() == [-1000.0, 1.5, 1000.0]
