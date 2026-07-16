"""Champion/challenger promotion, rollback, drift, and retraining tests."""

import numpy as np
import pandas as pd
import pytest

from quantx.core.research.promotion import (
    DriftMonitor,
    ModelRegistration,
    PromotionCriteria,
    PromotionEvidence,
    PromotionGate,
    PromotionRegistry,
    RetrainingPolicy,
)


def registration(artifact_id, lock_hash, *, role="shadow"):
    return ModelRegistration(
        strategy_id="mainboard-alpha",
        artifact_id=artifact_id,
        artifact_checksum=f"sha256:{artifact_id}",
        experiment_lock_hash=lock_hash,
        data_version_id="data-v1",
        prediction_store=f"/tmp/{artifact_id}.json",
        role=role,
        metrics={"rank_ic": 0.05},
        promotion_gate_passed=True,
        promotion_gate_report_hash="sha256:gate",
    )


def test_promotion_registry_supports_champion_challenger_and_rollback(tmp_path):
    registry = PromotionRegistry(tmp_path / "registry.json")
    registry.register(registration("model-a", "lock-a"))
    registry.promote("mainboard-alpha", "model-a", experiment_lock_hash="lock-a")
    registry.register(registration("model-b", "lock-b", role="challenger"))

    assert registry.champion("mainboard-alpha").artifact_id == "model-a"
    assert registry.challengers("mainboard-alpha")[0].artifact_id == "model-b"

    registry.promote("mainboard-alpha", "model-b", experiment_lock_hash="lock-b")
    assert registry.champion("mainboard-alpha").artifact_id == "model-b"

    registry.rollback("mainboard-alpha")
    assert registry.champion("mainboard-alpha").artifact_id == "model-a"


def test_drift_monitor_and_retraining_policy_detect_material_shift():
    rng = np.random.default_rng(7)
    baseline = pd.DataFrame({"ret5": rng.normal(0, 1, 2000), "vol20": rng.normal(1, 0.2, 2000)})
    monitor = DriftMonitor.fit(baseline, ("ret5", "vol20"), bins=10)

    stable = monitor.evaluate(baseline.sample(1000, random_state=7))
    shifted = monitor.evaluate(pd.DataFrame({"ret5": rng.normal(3, 1, 1000), "vol20": rng.normal(2, 0.2, 1000)}))
    policy = RetrainingPolicy(max_sessions_between_training=20, psi_threshold=0.2)

    assert stable.max_psi < 0.2
    assert shifted.max_psi > 0.2
    assert policy.should_retrain(sessions_since_training=5, drift=stable).required is False
    decision = policy.should_retrain(sessions_since_training=5, drift=shifted)
    assert decision.required is True
    assert "feature_drift" in decision.reasons
    assert policy.should_retrain(sessions_since_training=21, drift=stable).required is True


def test_stochastic_model_requires_multiple_seeds_before_promotion(tmp_path):
    registry = PromotionRegistry(tmp_path / "registry.json", min_stochastic_seeds=3)
    candidate = ModelRegistration(
        strategy_id="mainboard-alpha",
        artifact_id="torch-one-seed",
        artifact_checksum="sha256:model",
        experiment_lock_hash="lock-one",
        data_version_id="data-v1",
        prediction_store="/tmp/predictions.json",
        role="challenger",
        seed_set=(7,),
        stochastic_trainer=True,
        promotion_gate_passed=True,
        promotion_gate_report_hash="sha256:gate",
    )
    registry.register(candidate)

    with pytest.raises(ValueError, match="at least 3 seeds"):
        registry.promote("mainboard-alpha", "torch-one-seed", experiment_lock_hash="lock-one")


def test_promotion_gate_requires_quality_stability_reproduction_and_shadow_evidence():
    evidence = PromotionEvidence(
        evaluation_tier="sealed_holdout",
        fold_metrics={"fold-2023": {"rank_ic": 0.03}, "fold-2024": {"rank_ic": 0.05}},
        seed_count=3,
        cost_adjusted_total_return=0.12,
        adjacent_variant_returns=(0.08, 0.10, 0.06),
        baseline_total_return=0.05,
        max_single_stock_contribution_share=0.15,
        max_single_industry_contribution_share=0.35,
        integrity_passed=True,
        artifacts_verifiable=True,
        experiment_lock_hash="sha256:lock",
        physical_snapshot_id="snapshot:v1",
        live_shadow_sessions=30,
    )

    decision = PromotionGate(PromotionCriteria(min_rank_ic=0.02, min_baseline_improvement=0.03)).evaluate(evidence)

    assert decision.passed is True
    assert decision.failures == ()
    assert decision.report_hash.startswith("sha256:")


def test_promotion_gate_reports_all_material_failures():
    evidence = PromotionEvidence(
        evaluation_tier="development_oos",
        fold_metrics={"fold-2024": {"rank_ic": -0.02}},
        seed_count=1,
        cost_adjusted_total_return=-0.1,
        adjacent_variant_returns=(-0.2,),
        baseline_total_return=0.0,
        max_single_stock_contribution_share=0.8,
        max_single_industry_contribution_share=0.9,
        integrity_passed=False,
        artifacts_verifiable=False,
        experiment_lock_hash="",
        physical_snapshot_id="",
        live_shadow_sessions=0,
    )

    decision = PromotionGate().evaluate(evidence)

    assert decision.passed is False
    assert "sealed_holdout_required" in decision.failures
    assert "insufficient_seed_count" in decision.failures
    assert "cost_adjusted_return_below_threshold" in decision.failures
    assert "insufficient_live_shadow" in decision.failures
