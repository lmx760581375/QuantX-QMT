"""Static feature read-window analysis tests."""

import pytest

from quantx.core.research.causality import FeatureCausalityValidator, ReadWindow
from quantx.core.research.specs import FeatureSpec


def test_causality_analyzes_formula_dependencies_and_rolling_windows():
    spec = FeatureSpec(
        name="daily_features",
        raw_fields=("open", "close"),
        expressions={
            "ret5": "close / Ref(close, 5) - 1",
            "ma20": "Mean(close, 20)",
            "distance": "close / ma20 - 1",
        },
        lookback_sessions=20,
    )

    windows = FeatureCausalityValidator().analyze(spec)

    assert windows["ret5"] == ReadWindow(-5, 0)
    assert windows["ma20"] == ReadWindow(-19, 0)
    assert windows["distance"] == ReadWindow(-19, 0)


def test_causality_rejects_future_ref():
    spec = FeatureSpec(
        name="leaking",
        raw_fields=("close",),
        expressions={"future": "Ref(close, -1)"},
        lookback_sessions=1,
    )

    with pytest.raises(ValueError, match="future data"):
        FeatureCausalityValidator().require_causal(spec)
