"""Mutation tests proving future and OOS data cannot influence model fitting."""

import numpy as np
import pandas as pd

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.decision.predictions import PredictionStore
from quantx.core.research.dataset import DatasetBuilder, ResearchDataset
from quantx.core.research.runner import ResearchRunner
from quantx.core.research.specs import FeatureSpec, LabelSpec
from quantx.core.research.split import AnchoredWalkForwardSplitter
from quantx.core.research.trainer import RidgeTrainer
from quantx.core.research.universe import PointInTimeUniverseProvider


def _write_field(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0, *values], dtype="<f4").tofile(path)


def _provider(root):
    dates = pd.bdate_range("2021-01-04", periods=10)
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text(
        f"SZ000001\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n",
        encoding="utf-8",
    )
    opens = np.arange(10.0, 20.0)
    closes = opens + 0.5
    _write_field(root / "features" / "sz000001" / "open.day.bin", opens)
    _write_field(root / "features" / "sz000001" / "close.day.bin", closes)
    return dates, opens, closes


def _build_dataset(provider, start, end):
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider / "instruments" / "all.txt")
    return DatasetBuilder(QlibBinReader(provider), universe).build(
        FeatureSpec(
            name="causal",
            raw_fields=("open", "close"),
            expressions={"ret1": "close / Ref(close, 1) - 1"},
            lookback_sessions=1,
        ),
        LabelSpec(name="return_2d", horizon_sessions=2),
        start=start,
        end=end,
    )


def test_mutating_future_market_data_does_not_change_historical_features(tmp_path):
    provider = tmp_path / "provider"
    dates, opens, closes = _provider(provider)
    before = _build_dataset(provider, str(dates[0].date()), str(dates[-1].date()))

    changed_opens = opens.copy()
    changed_closes = closes.copy()
    changed_opens[7:] *= 10
    changed_closes[7:] *= 10
    _write_field(provider / "features" / "sz000001" / "open.day.bin", changed_opens)
    _write_field(provider / "features" / "sz000001" / "close.day.bin", changed_closes)
    after = _build_dataset(provider, str(dates[0].date()), str(dates[-1].date()))

    historical = before.frame.index.get_level_values("signal_time") <= dates[5]
    pd.testing.assert_frame_equal(
        before.frame.loc[historical, list(before.feature_columns)],
        after.frame.loc[historical, list(after.feature_columns)],
    )


def test_mutating_oos_labels_does_not_change_model_or_predictions(tmp_path):
    dates = pd.bdate_range("2019-01-02", "2021-12-31")
    rows = []
    for index, date in enumerate(dates):
        for instrument, offset in (("A", 0.0), ("B", 0.5)):
            feature = np.sin(index / 20) + offset
            rows.append(
                {
                    "signal_time": date,
                    "instrument": instrument,
                    "feature": feature,
                    "valid_features": True,
                    "theoretical_label": 0.05 * feature,
                    "label_end_session": (date + pd.offsets.BDay(2)).strftime("%Y-%m-%d"),
                }
            )
    frame = pd.DataFrame(rows).set_index(["signal_time", "instrument"])
    mutated = frame.copy()
    oos = mutated.index.get_level_values("signal_time").year == 2021
    mutated.loc[oos, "theoretical_label"] += 1000.0
    splitter = AnchoredWalkForwardSplitter(
        first_prediction_year=2021,
        last_prediction_year=2021,
        validation_sessions=20,
    )
    base_dataset = ResearchDataset(frame, ("feature",), "features-v1", "labels-v1", "universe-v1")
    mutated_dataset = ResearchDataset(mutated, ("feature",), "features-v1", "labels-v1", "universe-v1")

    first = ResearchRunner(
        trainer=RidgeTrainer(),
        splitter=splitter,
        output_dir=tmp_path / "first",
        experiment_name="leakage-test",
    ).run(base_dataset, data_version_id="data-v1")
    second = ResearchRunner(
        trainer=RidgeTrainer(),
        splitter=splitter,
        output_dir=tmp_path / "second",
        experiment_name="leakage-test",
    ).run(mutated_dataset, data_version_id="data-v1")

    assert [artifact.model_checksum for artifact in first.artifacts] == [
        artifact.model_checksum for artifact in second.artifacts
    ]
    assert first.prediction_checksum == second.prediction_checksum
    assert PredictionStore.load(first.prediction_path).records_for("2021-01-04") == (
        PredictionStore.load(second.prediction_path).records_for("2021-01-04")
    )


def test_t_plus_one_missing_open_does_not_remove_signal_time_sample(tmp_path):
    provider = tmp_path / "provider"
    dates, opens, _ = _provider(provider)
    opens[3] = np.nan
    _write_field(provider / "features" / "sz000001" / "open.day.bin", opens)

    dataset = _build_dataset(provider, str(dates[0].date()), str(dates[-1].date()))
    signal_row = dataset.frame.loc[(dates[2], "SZ000001")]

    assert bool(signal_row["valid_features"]) is True
    assert pd.isna(signal_row["theoretical_label"])
    assert bool(signal_row["label_available"]) is False
