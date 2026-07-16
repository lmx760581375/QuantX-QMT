"""Model promotion, rollback, drift monitoring, and retraining decisions."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MODEL_ROLES = {"shadow", "challenger", "champion", "retired"}


@dataclass(frozen=True)
class PromotionCriteria:
    min_positive_fold_ratio: float = 0.5
    min_rank_ic: float = 0.0
    min_seed_count: int = 3
    min_cost_adjusted_return: float = 0.0
    min_baseline_improvement: float = 0.0
    max_single_stock_contribution_share: float = 0.25
    max_single_industry_contribution_share: float = 0.5
    min_live_shadow_sessions: int = 20


@dataclass(frozen=True)
class PromotionEvidence:
    evaluation_tier: str
    fold_metrics: Mapping[str, Mapping[str, float | None]]
    seed_count: int
    cost_adjusted_total_return: float
    adjacent_variant_returns: tuple[float, ...]
    baseline_total_return: float
    max_single_stock_contribution_share: float
    max_single_industry_contribution_share: float
    integrity_passed: bool
    artifacts_verifiable: bool
    experiment_lock_hash: str
    physical_snapshot_id: str
    live_shadow_sessions: int


@dataclass(frozen=True)
class PromotionGateDecision:
    passed: bool
    failures: tuple[str, ...]
    report_hash: str


class PromotionGate:
    def __init__(self, criteria: PromotionCriteria | None = None):
        self.criteria = criteria or PromotionCriteria()

    def evaluate(self, evidence: PromotionEvidence) -> PromotionGateDecision:
        failures = []
        if evidence.evaluation_tier != "sealed_holdout":
            failures.append("sealed_holdout_required")
        fold_values = [
            metrics.get("rank_ic") for metrics in evidence.fold_metrics.values() if metrics.get("rank_ic") is not None
        ]
        positive_fold_ratio = sum(float(value) > 0 for value in fold_values) / len(fold_values) if fold_values else 0.0
        if positive_fold_ratio <= self.criteria.min_positive_fold_ratio:
            failures.append("insufficient_positive_folds")
        aggregate_rank_ic = float(np.mean(fold_values)) if fold_values else float("-inf")
        if aggregate_rank_ic < self.criteria.min_rank_ic:
            failures.append("rank_ic_below_threshold")
        if evidence.seed_count < self.criteria.min_seed_count:
            failures.append("insufficient_seed_count")
        if evidence.cost_adjusted_total_return <= self.criteria.min_cost_adjusted_return:
            failures.append("cost_adjusted_return_below_threshold")
        if len(evidence.adjacent_variant_returns) < 2 or min(evidence.adjacent_variant_returns) <= 0:
            failures.append("adjacent_variant_instability")
        if (
            evidence.cost_adjusted_total_return - evidence.baseline_total_return
            < self.criteria.min_baseline_improvement
        ):
            failures.append("no_baseline_improvement")
        if evidence.max_single_stock_contribution_share > self.criteria.max_single_stock_contribution_share:
            failures.append("single_stock_concentration")
        if evidence.max_single_industry_contribution_share > self.criteria.max_single_industry_contribution_share:
            failures.append("industry_concentration")
        if not evidence.integrity_passed:
            failures.append("integrity_checks_failed")
        if not evidence.artifacts_verifiable:
            failures.append("artifacts_not_verifiable")
        if not evidence.experiment_lock_hash or not evidence.physical_snapshot_id:
            failures.append("reproduction_identity_missing")
        if evidence.live_shadow_sessions < self.criteria.min_live_shadow_sessions:
            failures.append("insufficient_live_shadow")
        report_hash = _promotion_report_hash(self.criteria, evidence, failures)
        return PromotionGateDecision(not failures, tuple(failures), report_hash)


@dataclass(frozen=True)
class ModelRegistration:
    strategy_id: str
    artifact_id: str
    artifact_checksum: str
    experiment_lock_hash: str
    data_version_id: str
    prediction_store: str
    role: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    seed_set: tuple[int, ...] = (7,)
    stochastic_trainer: bool = False
    promotion_gate_passed: bool = False
    promotion_gate_report_hash: str | None = None
    registered_at: str = field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())

    def __post_init__(self) -> None:
        if not all(
            (
                self.strategy_id,
                self.artifact_id,
                self.artifact_checksum,
                self.experiment_lock_hash,
                self.data_version_id,
            )
        ):
            raise ValueError("Model registration identity fields are required")
        if self.role not in MODEL_ROLES:
            raise ValueError(f"Unsupported model role: {self.role}")
        object.__setattr__(self, "metrics", dict(self.metrics))
        normalized_seeds = tuple(sorted(set(int(seed) for seed in self.seed_set)))
        if not normalized_seeds:
            raise ValueError("Model registration requires at least one seed")
        object.__setattr__(self, "seed_set", normalized_seeds)
        if self.promotion_gate_passed and not self.promotion_gate_report_hash:
            raise ValueError("Passed promotion gate requires a report hash")


class PromotionRegistry:
    def __init__(self, path: str | Path, *, min_stochastic_seeds: int = 3):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        if min_stochastic_seeds < 2:
            raise ValueError("min_stochastic_seeds must be at least 2")
        self.min_stochastic_seeds = int(min_stochastic_seeds)

    def register(self, registration: ModelRegistration) -> None:
        with self._locked_state() as state:
            registrations = state["registrations"]
            if any(
                row["strategy_id"] == registration.strategy_id and row["artifact_id"] == registration.artifact_id
                for row in registrations
            ):
                raise ValueError("Model artifact is already registered for this strategy")
            registrations.append(asdict(registration))
            self._history(state, "register", registration.strategy_id, registration.artifact_id)

    def promote(self, strategy_id: str, artifact_id: str, *, experiment_lock_hash: str) -> None:
        with self._locked_state() as state:
            target = self._find(state, strategy_id, artifact_id)
            if target["experiment_lock_hash"] != experiment_lock_hash:
                raise ValueError("Promotion ExperimentLock hash does not match the registered artifact")
            if not target.get("promotion_gate_passed", False) or not target.get("promotion_gate_report_hash"):
                raise ValueError("Model has not passed the promotion quality gate")
            if target.get("stochastic_trainer", False) and len(target.get("seed_set", ())) < self.min_stochastic_seeds:
                raise ValueError(f"Stochastic trainer promotion requires at least {self.min_stochastic_seeds} seeds")
            current_id = state["champions"].get(strategy_id)
            if current_id and current_id != artifact_id:
                current = self._find(state, strategy_id, current_id)
                current["role"] = "retired"
                state["previous_champions"].setdefault(strategy_id, []).append(current_id)
            target["role"] = "champion"
            state["champions"][strategy_id] = artifact_id
            self._history(state, "promote", strategy_id, artifact_id)

    def rollback(self, strategy_id: str) -> None:
        with self._locked_state() as state:
            previous = state["previous_champions"].get(strategy_id, [])
            if not previous:
                raise ValueError("No previous champion is available for rollback")
            current_id = state["champions"].get(strategy_id)
            if current_id:
                self._find(state, strategy_id, current_id)["role"] = "retired"
            restored_id = previous.pop()
            self._find(state, strategy_id, restored_id)["role"] = "champion"
            state["champions"][strategy_id] = restored_id
            self._history(state, "rollback", strategy_id, restored_id)

    def champion(self, strategy_id: str) -> ModelRegistration | None:
        state = self._read_state()
        artifact_id = state["champions"].get(strategy_id)
        return self._registration(self._find(state, strategy_id, artifact_id)) if artifact_id else None

    def challengers(self, strategy_id: str) -> tuple[ModelRegistration, ...]:
        state = self._read_state()
        return tuple(
            self._registration(row)
            for row in state["registrations"]
            if row["strategy_id"] == strategy_id and row["role"] == "challenger"
        )

    def _locked_state(self):
        return _RegistryTransaction(self)

    def _read_state(self) -> dict:
        if not self.path.exists():
            return _empty_state()
        return json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _find(state: dict, strategy_id: str, artifact_id: str) -> dict:
        for row in state["registrations"]:
            if row["strategy_id"] == strategy_id and row["artifact_id"] == artifact_id:
                return row
        raise ValueError(f"Unknown registered artifact: {strategy_id}/{artifact_id}")

    @staticmethod
    def _registration(row: dict) -> ModelRegistration:
        return ModelRegistration(**row)

    @staticmethod
    def _history(state: dict, action: str, strategy_id: str, artifact_id: str) -> None:
        state["history"].append(
            {
                "action": action,
                "strategy_id": strategy_id,
                "artifact_id": artifact_id,
                "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            }
        )


class _RegistryTransaction:
    def __init__(self, registry: PromotionRegistry):
        self.registry = registry
        self.handle = None
        self.state = None

    def __enter__(self):
        self.registry.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.registry.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.state = self.registry._read_state()
        return self.state

    def __exit__(self, exc_type, exc, traceback):
        del exc, traceback
        if exc_type is None:
            self.registry.path.write_text(json.dumps(self.state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _empty_state() -> dict:
    return {
        "format": "quantx.promotion-registry",
        "format_version": 1,
        "registrations": [],
        "champions": {},
        "previous_champions": {},
        "history": [],
    }


@dataclass(frozen=True)
class DriftReport:
    feature_psi: Mapping[str, float]
    max_psi: float
    sample_count: int


class DriftMonitor:
    def __init__(self, edges: Mapping[str, tuple[float, ...]], baseline_probabilities: Mapping[str, tuple[float, ...]]):
        self.edges = dict(edges)
        self.baseline_probabilities = dict(baseline_probabilities)

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: tuple[str, ...], *, bins: int = 10) -> "DriftMonitor":
        if bins < 2:
            raise ValueError("Drift monitor requires at least two bins")
        edges = {}
        probabilities = {}
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                raise ValueError(f"Drift baseline feature is empty: {feature}")
            inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
            feature_edges = np.concatenate(([-np.inf], inner, [np.inf]))
            counts, _ = np.histogram(values, bins=feature_edges)
            edges[feature] = tuple(float(value) for value in feature_edges)
            probabilities[feature] = tuple(float(value) for value in _probabilities(counts))
        return cls(edges, probabilities)

    def evaluate(self, frame: pd.DataFrame) -> DriftReport:
        feature_psi = {}
        for feature, raw_edges in self.edges.items():
            values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
            counts, _ = np.histogram(values, bins=np.asarray(raw_edges))
            actual = _probabilities(counts)
            expected = np.asarray(self.baseline_probabilities[feature])
            feature_psi[feature] = float(np.sum((actual - expected) * np.log(actual / expected)))
        return DriftReport(
            feature_psi=feature_psi,
            max_psi=max(feature_psi.values(), default=0.0),
            sample_count=len(frame),
        )


def _probabilities(counts) -> np.ndarray:
    values = np.asarray(counts, dtype=float) + 1e-6
    return values / values.sum()


def _promotion_report_hash(criteria, evidence, failures) -> str:
    payload = {
        "criteria": asdict(criteria),
        "evidence": asdict(evidence),
        "failures": list(failures),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class RetrainingDecision:
    required: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetrainingPolicy:
    max_sessions_between_training: int
    psi_threshold: float

    def should_retrain(self, *, sessions_since_training: int, drift: DriftReport) -> RetrainingDecision:
        reasons = []
        if sessions_since_training >= self.max_sessions_between_training:
            reasons.append("scheduled_interval")
        if drift.max_psi >= self.psi_threshold:
            reasons.append("feature_drift")
        return RetrainingDecision(required=bool(reasons), reasons=tuple(reasons))
