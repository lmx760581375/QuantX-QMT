"""Experiment lock, sealed holdout ledger, and live-shadow lifecycle tests."""

import pytest

from quantx.core.research.lifecycle import ExperimentLifecycle, ExperimentLock, LiveShadowRegistry


def lock():
    return ExperimentLock(
        research_generation=1,
        normalized_research_config_hash="research-config",
        normalized_backtest_config_hash="backtest-config",
        data_version_id="data-v1",
        physical_snapshot_id="snapshot-v1",
        universe_version_id="universe-v1",
        feature_schema_hash="features-v1",
        label_spec_hash="labels-v1",
        model_artifact_checksums=("sha256:model",),
        seed_set=(7,),
        portfolio_config_hash="portfolio-v1",
        risk_config_hash="risk-v1",
        order_planner_config_hash="planner-v1",
        code_fingerprint="code-v1",
        holdout_range=("2025-01-01", "2026-07-10"),
    )


def test_sealed_holdout_can_only_be_claimed_once_per_lock(tmp_path):
    lifecycle = ExperimentLifecycle(tmp_path)
    experiment_lock = lock()

    lock_hash = lifecycle.write_lock(experiment_lock)
    lifecycle.claim_sealed_holdout(lock_hash, result_uri="/tmp/result-1")

    with pytest.raises(ValueError, match="already been run"):
        lifecycle.claim_sealed_holdout(lock_hash, result_uri="/tmp/result-2")


def test_live_shadow_registry_tracks_frozen_artifact_without_mutating_it(tmp_path):
    registry = LiveShadowRegistry(tmp_path / "shadow.json")

    registry.register(
        strategy_id="ridge-top20",
        artifact_id="ridge-v1",
        artifact_checksum="sha256:model",
        prediction_store="/tmp/predictions.json",
    )

    state = registry.load()
    assert state["ridge-top20"]["artifact_id"] == "ridge-v1"
    assert state["ridge-top20"]["mode"] == "live_shadow"
