"""Alpha model backed by immutable out-of-sample predictions."""

from __future__ import annotations

from typing import Any, Mapping

from .predictions import PredictionStore
from .types import AlphaSnapshot, AlphaValue, MarketObservation


class PrecomputedAlphaModel:
    def __init__(
        self,
        store: PredictionStore,
        *,
        artifact_id: str,
        feature_schema_hash: str,
    ):
        self.store = store
        self.artifact_id = artifact_id
        self.feature_schema_hash = feature_schema_hash

    def prepare(self, context: Any) -> None:
        del context

    def predict(self, observation: MarketObservation) -> AlphaSnapshot:
        if observation.features.schema_id != self.feature_schema_hash:
            raise ValueError(
                f"Prediction feature schema mismatch: expected {self.feature_schema_hash}, "
                f"got {observation.features.schema_id}"
            )
        records = self.store.records_for(observation.clock.signal_time.session, artifact_id=self.artifact_id)
        outside = {record.instrument for record in records} - set(observation.universe)
        if outside:
            raise ValueError(f"Predictions are outside the decision universe: {sorted(outside)}")
        for record in records:
            if record.signal_time != observation.clock.signal_time:
                raise ValueError("Prediction signal_time does not match MarketObservation")
            if record.feature_schema_hash != self.feature_schema_hash:
                raise ValueError("Prediction feature schema does not match model configuration")
        tiers = {record.evaluation_tier for record in records}
        if len(tiers) > 1:
            raise ValueError("One AlphaSnapshot cannot mix evaluation tiers")
        horizons = {record.prediction_horizon for record in records}
        if len(horizons) > 1:
            raise ValueError("One AlphaSnapshot cannot mix prediction horizons")
        values = tuple(
            AlphaValue(
                instrument=record.instrument,
                score=record.score,
                rank=record.rank or index,
                uncertainty=record.uncertainty,
            )
            for index, record in enumerate(records, start=1)
        )
        return AlphaSnapshot(
            signal_time=observation.clock.signal_time,
            earliest_execution_time=observation.clock.execute_not_before,
            values=values,
            source_id=f"{self.artifact_id}:{observation.clock.signal_time.session}",
            horizon_sessions=next(iter(horizons), None),
            metadata={
                "artifact_id": self.artifact_id,
                "feature_schema_hash": self.feature_schema_hash,
                "evaluation_tier": next(iter(tiers), "development_oos"),
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "type": "predictions",
            "artifact_id": self.artifact_id,
            "feature_schema_hash": self.feature_schema_hash,
            "prediction_count": len(self.store),
        }
