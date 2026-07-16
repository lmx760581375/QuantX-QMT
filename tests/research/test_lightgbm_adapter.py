"""Optional LightGBM trainer dependency behavior."""

import importlib.util

import pandas as pd
import pytest

from quantx.core.research.trainer import LightGBMTrainer


def test_lightgbm_trainer_is_optional_with_clear_error_when_unavailable():
    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("LightGBM is installed in this environment")
    trainer = LightGBMTrainer(params={"n_estimators": 5})
    frame = pd.DataFrame({"x": [1.0, 2.0], "label": [0.1, 0.2]})

    with pytest.raises(RuntimeError, match="lightgbm"):
        trainer.fit(frame, ("x",), "label", seed=7)
