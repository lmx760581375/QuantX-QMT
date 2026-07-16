"""Walk-forward model training, OOS prediction, and aggregate evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.predictions import (
    EVALUATION_TIERS,
    PredictionContract,
    PredictionRecord,
    PredictionStore,
)

from .artifacts import ModelArtifact, write_artifact_manifest
from .dataset import ResearchDataset
from .evaluation import evaluate_predictions
from .fingerprint import compute_code_fingerprint
from .preprocessing import LabelWinsorizer


@dataclass(frozen=True)
class ResearchResult:
    experiment_id: str
    output_dir: Path
    prediction_path: Path
    prediction_checksum: str
    prediction_count: int
    fold_count: int
    artifacts: tuple[ModelArtifact, ...]
    metrics: dict


class ResearchRunner:
    def __init__(
        self,
        *,
        trainer,
        splitter,
        output_dir: str | Path,
        experiment_name: str,
        seed: int = 7,
        evaluation_tier: str = "development_oos",
    ):
        self.trainer = trainer
        self.splitter = splitter
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.seed = int(seed)
        if evaluation_tier not in EVALUATION_TIERS:
            raise ValueError(f"Unsupported evaluation tier: {evaluation_tier}")
        self.evaluation_tier = evaluation_tier

    def run(self, dataset: ResearchDataset, *, data_version_id: str) -> ResearchResult:
        experiment_id = self._experiment_id(dataset, data_version_id)
        root = self.output_dir / experiment_id
        root.mkdir(parents=True, exist_ok=True)
        folds = self.splitter.split(dataset.frame)
        if not folds:
            raise ValueError("Walk-forward splitter produced no folds")
        prediction_records = []
        prediction_contracts = []
        metric_rows = []
        artifacts = []
        fold_preprocessing = []
        for fold in folds:
            train = fold.train[fold.train["valid_features"].fillna(False) & fold.train["theoretical_label"].notna()]
            validation = fold.validation[
                fold.validation["valid_features"].fillna(False) & fold.validation["theoretical_label"].notna()
            ]
            if dataset.label_winsorize is not None:
                winsorizer = LabelWinsorizer.fit(train["theoretical_label"], dataset.label_winsorize)
                train = winsorizer.transform(train, "theoretical_label")
                validation = winsorizer.transform(validation, "theoretical_label")
                fold_preprocessing.append({"fold_id": fold.fold_id, "label": winsorizer.describe()})
            else:
                fold_preprocessing.append({"fold_id": fold.fold_id, "label": None})
            model = self.trainer.fit(
                train,
                dataset.feature_columns,
                "theoretical_label",
                seed=self.seed,
                validation=validation,
            )
            fold_dir = root / "folds" / fold.fold_id
            artifact = model.write(
                fold_dir / "model.json",
                artifact_id=f"{experiment_id}:{fold.fold_id}",
                feature_schema_hash=dataset.feature_spec_hash,
                data_version_id=data_version_id,
                fold_id=fold.fold_id,
            )
            artifacts.append(artifact)
            scores = model.predict(fold.prediction)
            prediction_sessions = pd.DatetimeIndex(fold.prediction.index.get_level_values("signal_time"))
            prediction_contracts.append(
                PredictionContract(
                    artifact_id=experiment_id,
                    fold_id=fold.fold_id,
                    prediction_start=prediction_sessions.min().strftime("%Y-%m-%d"),
                    prediction_end=prediction_sessions.max().strftime("%Y-%m-%d"),
                    training_information_end=fold.training_information_cutoff.strftime("%Y-%m-%d"),
                    feature_schema_hash=dataset.feature_spec_hash,
                )
            )
            fold_rows = fold.prediction[["theoretical_label"]].copy()
            fold_rows["score"] = scores
            fold_rows["fold_id"] = fold.fold_id
            metric_rows.append(fold_rows)
            for (signal_time, instrument), score in zip(fold.prediction.index, scores):
                prediction_records.append(
                    PredictionRecord(
                        signal_time=_after_close(pd.Timestamp(signal_time)),
                        instrument=str(instrument),
                        score=float(score),
                        prediction_horizon=dataset.label_horizon_sessions,
                        artifact_id=experiment_id,
                        fold_id=fold.fold_id,
                        feature_schema_hash=dataset.feature_spec_hash,
                        evaluation_tier=self.evaluation_tier,
                    )
                )
        store = PredictionStore(prediction_records, contracts=prediction_contracts)
        prediction_path = root / "oos_predictions.json"
        prediction_checksum = store.write(prediction_path)
        all_metrics = pd.concat(metric_rows).sort_index()
        metrics = evaluate_predictions(all_metrics)
        artifacts_tuple = tuple(artifacts)
        write_artifact_manifest(root / "model_artifacts.json", artifacts_tuple)
        report = {
            "experiment_id": experiment_id,
            "experiment_name": self.experiment_name,
            "data_version_id": data_version_id,
            "feature_schema_hash": dataset.feature_spec_hash,
            "label_spec_hash": dataset.label_spec_hash,
            "universe_version_id": dataset.universe_version_id,
            "evaluation_tier": self.evaluation_tier,
            "seed": self.seed,
            "trainer": self.trainer.describe(),
            "code_fingerprint": compute_code_fingerprint(Path.cwd()),
            "prediction_store": str(prediction_path),
            "prediction_checksum": prediction_checksum,
            "prediction_count": len(store),
            "folds": [fold.fold_id for fold in folds],
            "fold_preprocessing": fold_preprocessing,
            "metrics": metrics,
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }
        (root / "research_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return ResearchResult(
            experiment_id=experiment_id,
            output_dir=root,
            prediction_path=prediction_path,
            prediction_checksum=prediction_checksum,
            prediction_count=len(store),
            fold_count=len(folds),
            artifacts=artifacts_tuple,
            metrics=metrics,
        )

    def _experiment_id(self, dataset: ResearchDataset, data_version_id: str) -> str:
        identity = {
            "name": self.experiment_name,
            "data_version_id": data_version_id,
            "feature_spec_hash": dataset.feature_spec_hash,
            "label_spec_hash": dataset.label_spec_hash,
            "universe_version_id": dataset.universe_version_id,
            "trainer": self.trainer.describe(),
            "seed": self.seed,
            "evaluation_tier": self.evaluation_tier,
            "splitter": vars(self.splitter),
            "code_fingerprint": compute_code_fingerprint(Path.cwd()),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return f"{self.experiment_name}-{digest[:16]}"


def _after_close(value: pd.Timestamp) -> MarketTime:
    timestamp = value.to_pydatetime().replace(hour=15, minute=0, second=0, microsecond=0)
    timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return MarketTime(value.strftime("%Y-%m-%d"), SessionPhase.AFTER_CLOSE, timestamp)
