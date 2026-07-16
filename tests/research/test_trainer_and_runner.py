"""Deterministic trainer, artifact, and OOS research-runner tests."""

import numpy as np
import pandas as pd

from quantx.core.research.artifacts import LinearModelArtifact
from quantx.core.research.dataset import ResearchDataset
from quantx.core.research.runner import ResearchRunner
from quantx.core.research.split import AnchoredWalkForwardSplitter
from quantx.core.research.trainer import RidgeTrainer


def make_dataset():
    dates = pd.bdate_range("2019-01-02", "2022-12-30")
    rows = []
    for index, date in enumerate(dates):
        for instrument, offset in (("A", 0.0), ("B", 0.5)):
            x1 = np.sin(index / 20) + offset
            x2 = np.cos(index / 15) - offset
            rows.append(
                {
                    "signal_time": date,
                    "instrument": instrument,
                    "x1": x1,
                    "x2": x2,
                    "valid_features": True,
                    "theoretical_label": 0.03 * x1 - 0.02 * x2,
                    "label_entry_session": (date + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
                    "label_end_session": (date + pd.offsets.BDay(6)).strftime("%Y-%m-%d"),
                    "label_available": True,
                }
            )
    frame = pd.DataFrame(rows).set_index(["signal_time", "instrument"])
    return ResearchDataset(frame, ("x1", "x2"), "features-v1", "labels-v1", "universe-v1")


def test_ridge_research_runner_writes_reproducible_oos_predictions_and_safe_artifacts(tmp_path):
    dataset = make_dataset()
    splitter = AnchoredWalkForwardSplitter(
        first_prediction_year=2021,
        last_prediction_year=2022,
        validation_sessions=40,
        embargo_sessions=5,
    )

    first = ResearchRunner(
        trainer=RidgeTrainer(alpha=1.0),
        splitter=splitter,
        output_dir=tmp_path / "first",
        experiment_name="ridge-test",
        seed=7,
    ).run(dataset, data_version_id="data-v1")
    second = ResearchRunner(
        trainer=RidgeTrainer(alpha=1.0),
        splitter=splitter,
        output_dir=tmp_path / "second",
        experiment_name="ridge-test",
        seed=7,
    ).run(dataset, data_version_id="data-v1")

    assert first.prediction_checksum == second.prediction_checksum
    assert first.fold_count == 2
    assert first.prediction_count > 0
    assert first.metrics["rank_ic"] > 0.99
    assert first.prediction_path.exists()
    assert all(artifact.model_format == "quantx_linear_json" for artifact in first.artifacts)
    assert all(artifact.model_uri.endswith("model.json") for artifact in first.artifacts)
    loaded = LinearModelArtifact.load(first.artifacts[0].model_uri, first.artifacts[0].model_checksum)
    assert loaded.feature_names == ("x1", "x2")


def test_ridge_model_predictions_are_finite_with_missing_features():
    train = pd.DataFrame({"x1": [1.0, 2.0, np.nan, 4.0], "x2": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0]})
    model = RidgeTrainer(alpha=0.1).fit(train, ("x1", "x2"), "y", seed=7)

    predictions = model.predict(pd.DataFrame({"x1": [np.nan, 5.0], "x2": [2.0, np.nan]}))

    assert np.isfinite(predictions).all()
