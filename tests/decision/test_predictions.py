"""Frozen prediction storage and alpha adapter tests."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import DecisionClock, MarketTime, SessionPhase
from quantx.core.decision.predictions import PredictionContract, PredictionRecord, PredictionStore
from quantx.core.decision.precomputed import PrecomputedAlphaModel
from quantx.core.decision.types import FeatureBatch, MarketObservation


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day: int, hour: int, phase: SessionPhase) -> MarketTime:
    return MarketTime(
        session=f"2026-07-{day:02d}",
        phase=phase,
        timestamp=datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def prediction(instrument: str, score: float, *, fold_id: str = "fold-1") -> PredictionRecord:
    return PredictionRecord(
        signal_time=market_time(10, 15, SessionPhase.AFTER_CLOSE),
        instrument=instrument,
        score=score,
        prediction_horizon=5,
        artifact_id="ridge-v1",
        fold_id=fold_id,
        feature_schema_hash="schema-1",
        evaluation_tier="development_oos",
    )


def observation() -> MarketObservation:
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    return MarketObservation(
        clock=DecisionClock(
            observation_end=signal_time,
            data_available_at=signal_time,
            signal_time=signal_time,
            execute_not_before=market_time(13, 9, SessionPhase.OPEN),
        ),
        universe=("SZ000001", "SH600000"),
        features=FeatureBatch(
            as_of=signal_time,
            instruments=(),
            values=(),
            schema_id="schema-1",
            available_at=signal_time,
        ),
    )


def test_prediction_store_round_trips_and_rejects_overlapping_fold_predictions(tmp_path):
    store = PredictionStore([prediction("SZ000001", 0.8), prediction("SH600000", 0.4)])
    path = tmp_path / "predictions.json"

    checksum = store.write(path)
    loaded = PredictionStore.load(path, expected_checksum=checksum)

    assert loaded.records_for("2026-07-10", artifact_id="ridge-v1") == store.records_for(
        "2026-07-10", artifact_id="ridge-v1"
    )
    assert loaded.artifact_ids == ("ridge-v1",)

    with pytest.raises(ValueError, match="overlapping prediction"):
        store.add(prediction("SZ000001", 0.9, fold_id="fold-2"))


def test_prediction_store_streaming_writer_preserves_canonical_format(tmp_path):
    records = [prediction("SZ000001", 0.8), prediction("SH600000", 0.4)]
    store = PredictionStore(records)
    path = tmp_path / "predictions.json"

    checksum = store.write(path)
    expected = json.dumps(
        {
            "format": "quantx.predictions",
            "format_version": 1,
            "contracts": [],
            "records": [record.to_dict() for record in sorted(records, key=lambda row: row.instrument)],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"

    assert path.read_text() == expected
    assert checksum == PredictionStore._checksum(expected.encode("utf-8"))


def test_prediction_store_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "predictions.json"
    PredictionStore([prediction("SZ000001", 0.8)]).write(path)

    with pytest.raises(ValueError, match="checksum"):
        PredictionStore.load(path, expected_checksum="sha256:wrong")


def test_prediction_store_session_lookup_does_not_scan_all_records():
    store = PredictionStore([prediction("SZ000001", 0.8), prediction("SH600000", 0.4)])

    class FailingValues(dict):
        def values(self):
            raise AssertionError("records_for must use the session index")

    store._records = FailingValues(store._records)

    records = store.records_for("2026-07-10", artifact_id="ridge-v1")

    assert [record.instrument for record in records] == ["SZ000001", "SH600000"]


def test_prediction_contract_enforces_oos_range_training_cutoff_and_schema():
    contract = PredictionContract(
        artifact_id="ridge-v1",
        fold_id="fold-1",
        prediction_start="2026-07-01",
        prediction_end="2026-07-31",
        training_information_end="2026-06-30",
        feature_schema_hash="schema-1",
    )
    store = PredictionStore(contracts=[contract])
    store.add(prediction("SZ000001", 0.8))

    outside = PredictionRecord(
        signal_time=market_time(13, 15, SessionPhase.AFTER_CLOSE),
        instrument="SH600000",
        score=0.2,
        prediction_horizon=5,
        artifact_id="ridge-v1",
        fold_id="missing-fold",
        feature_schema_hash="schema-1",
    )
    with pytest.raises(ValueError, match="contract"):
        store.add(outside)

    wrong_schema = PredictionRecord(
        signal_time=market_time(13, 15, SessionPhase.AFTER_CLOSE),
        instrument="SH600000",
        score=0.2,
        prediction_horizon=5,
        artifact_id="ridge-v1",
        fold_id="fold-1",
        feature_schema_hash="wrong",
    )
    with pytest.raises(ValueError, match="schema"):
        store.add(wrong_schema)


def test_precomputed_alpha_uses_only_matching_signal_time_and_universe():
    model = PrecomputedAlphaModel(
        PredictionStore([prediction("SZ000001", 0.8), prediction("SH600000", 0.4)]),
        artifact_id="ridge-v1",
        feature_schema_hash="schema-1",
    )

    alpha = model.predict(observation())

    assert [value.instrument for value in alpha.values] == ["SZ000001", "SH600000"]
    assert [value.rank for value in alpha.values] == [1, 2]
    assert alpha.signal_time.session == "2026-07-10"
    assert alpha.earliest_execution_time.session == "2026-07-13"
    assert alpha.metadata["evaluation_tier"] == "development_oos"


def test_precomputed_alpha_rejects_schema_mismatch():
    model = PrecomputedAlphaModel(
        PredictionStore([prediction("SZ000001", 0.8)]),
        artifact_id="ridge-v1",
        feature_schema_hash="other-schema",
    )

    with pytest.raises(ValueError, match="feature schema"):
        model.predict(observation())
