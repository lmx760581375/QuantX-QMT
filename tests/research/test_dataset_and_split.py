"""Golden dataset, next-open labels, and purged walk-forward tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.research.dataset import DatasetBuilder
from quantx.core.research.specs import FeatureSpec, LabelSpec
from quantx.core.research.split import AnchoredWalkForwardSplitter
from quantx.core.research.universe import PointInTimeUniverseProvider


def write_field(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0, *values], dtype="<f4").tofile(path)


def make_provider(root: Path):
    dates = pd.bdate_range("2021-01-04", periods=8)
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text(
        f"SZ000001\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\nSH600000\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n",
        encoding="utf-8",
    )
    for symbol, base in (("sz000001", 10.0), ("sh600000", 20.0)):
        opens = [base + i for i in range(8)]
        closes = [value + 0.5 for value in opens]
        write_field(root / "features" / symbol / "open.day.bin", opens)
        write_field(root / "features" / symbol / "close.day.bin", closes)
    benchmark_opens = [100.0 + 2 * i for i in range(8)]
    write_field(root / "features" / "sh000905" / "open.day.bin", benchmark_opens)
    write_field(
        root / "features" / "sh000905" / "close.day.bin",
        [value + 0.5 for value in benchmark_opens],
    )
    return dates


def test_dataset_builder_uses_next_open_label_without_future_features(tmp_path):
    provider = tmp_path / "provider"
    dates = make_provider(provider)
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider / "instruments" / "all.txt")
    dataset = DatasetBuilder(QlibBinReader(provider), universe).build(
        FeatureSpec(
            name="features",
            raw_fields=("open", "close"),
            expressions={"ret1": "close / Ref(close, 1) - 1"},
            lookback_sessions=1,
        ),
        LabelSpec(name="ret2", horizon_sessions=2),
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
    )

    row = dataset.frame.loc[(dates[1], "SZ000001")]
    expected = (10.0 + 4) / (10.0 + 2) - 1
    assert row["theoretical_label"] == pytest.approx(expected)
    assert row["label_entry_session"] == dates[2].strftime("%Y-%m-%d")
    assert row["label_end_session"] == dates[4].strftime("%Y-%m-%d")
    assert dataset.feature_columns == ("open", "close", "ret1")
    assert dataset.frame.index.names == ["signal_time", "instrument"]


def test_walk_forward_purges_training_labels_crossing_prediction_boundary():
    frame = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(["2019-12-27", "2019-12-30", "2020-01-02", "2020-06-01", "2021-01-04"]),
            "instrument": ["A"] * 5,
            "label_end_session": [
                "2019-12-30",
                "2020-01-03",
                "2020-01-06",
                "2020-06-05",
                "2021-01-08",
            ],
            "feature": range(5),
            "theoretical_label": range(5),
        }
    ).set_index(["signal_time", "instrument"])
    splitter = AnchoredWalkForwardSplitter(
        first_prediction_year=2020,
        last_prediction_year=2021,
        validation_sessions=1,
        embargo_sessions=0,
    )

    folds = splitter.split(frame)

    assert len(folds) == 2
    first = folds[0]
    assert list(first.train.index.get_level_values("signal_time")) == [pd.Timestamp("2019-12-27")]
    assert set(first.prediction.index.get_level_values("signal_time").year) == {2020}


def test_dataset_builder_subtracts_matching_benchmark_next_open_return(tmp_path):
    provider = tmp_path / "provider"
    dates = make_provider(provider)
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider / "instruments" / "all.txt")

    dataset = DatasetBuilder(QlibBinReader(provider), universe).build(
        FeatureSpec(name="features", raw_fields=("open",), expressions={}, lookback_sessions=0),
        LabelSpec(
            name="excess_ret2",
            horizon_sessions=2,
            benchmark="SH000905",
            excess_return=True,
        ),
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
    )

    row = dataset.frame.loc[(dates[1], "SZ000001")]
    stock_return = (10.0 + 4) / (10.0 + 2) - 1
    benchmark_return = (100.0 + 2 * 4) / (100.0 + 2 * 2) - 1
    assert row["theoretical_label"] == pytest.approx(stock_return - benchmark_return)
    assert "SH000905" not in dataset.frame.index.get_level_values("instrument")


def test_excess_return_label_requires_benchmark():
    with pytest.raises(ValueError, match="benchmark"):
        LabelSpec(name="invalid", horizon_sessions=5, excess_return=True)


def test_cross_sectional_mean_benchmark_creates_daily_relative_labels(tmp_path):
    provider = tmp_path / "provider"
    dates = make_provider(provider)
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider / "instruments" / "all.txt")
    dataset = DatasetBuilder(QlibBinReader(provider), universe).build(
        FeatureSpec(name="features", raw_fields=("open",), expressions={}, lookback_sessions=0),
        LabelSpec(
            name="relative_ret2",
            horizon_sessions=2,
            benchmark="cross_sectional_mean",
            excess_return=True,
        ),
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
    )

    daily_means = dataset.frame["theoretical_label"].groupby(level="signal_time").mean()

    assert daily_means.dropna().abs().max() < 1e-12


def test_dataset_builder_injects_static_industry_and_concept_groups(tmp_path):
    provider = tmp_path / "provider"
    dates = make_provider(provider)
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider / "instruments" / "all.txt")
    industry_csv = tmp_path / "industry.csv"
    pd.DataFrame([
        {"symbol": "SZ000001", "industry_code": "bank"},
        {"symbol": "SH600000", "industry_code": "steel"},
    ]).to_csv(industry_csv, index=False)
    concept_csv = tmp_path / "sector.csv"
    pd.DataFrame([
        {"symbol": "SZ000001", "sector_type": "concept", "sector_code": "BK1"},
        {"symbol": "SZ000001", "sector_type": "concept", "sector_code": "BK2"},
        {"symbol": "SH600000", "sector_type": "concept", "sector_code": "BK2"},
    ]).to_csv(concept_csv, index=False)

    dataset = DatasetBuilder(QlibBinReader(provider), universe).build(
        FeatureSpec(
            name="group_features",
            raw_fields=("open", "close"),
            expressions={
                "ret1": "close / Ref(close, 1) - 1",
                "industry_ret1": "GroupMean(ret1, industry_l1)",
                "concept_ret1": "GroupMean(ret1, concept, agg='max')",
                "rel_industry_ret1": "ret1 - industry_ret1",
            },
            lookback_sessions=1,
            groups={
                "industry_l1": {"source": "meta.industry_l1", "path": str(industry_csv)},
                "concept": {"source": "meta.concept", "path": str(concept_csv)},
            },
        ),
        LabelSpec(name="ret2", horizon_sessions=2),
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
    )

    assert "industry_ret1" in dataset.feature_columns
    assert "concept_ret1" in dataset.feature_columns
    assert "rel_industry_ret1" in dataset.feature_columns
    assert np.isfinite(dataset.frame["industry_ret1"].dropna()).all()
    assert np.isfinite(dataset.frame["concept_ret1"].dropna()).all()
