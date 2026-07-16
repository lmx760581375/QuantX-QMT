"""Feature schema, code fingerprint, and artifact-loader security tests."""

import subprocess

import pytest

from quantx.core.research.artifacts import ArtifactLoaderRegistry, LinearModelArtifact
from quantx.core.research.fingerprint import compute_code_fingerprint
from quantx.core.research.schema import FeatureSchema


def test_feature_schema_hash_is_stable_and_order_sensitive():
    first = FeatureSchema.create(
        names=("ret5", "vol20"),
        dtypes=("float32", "float32"),
        lookback_sessions=20,
        frequency="day",
        read_windows={"ret5": (-5, 0), "vol20": (-19, 0)},
    )
    same = FeatureSchema.create(
        names=("ret5", "vol20"),
        dtypes=("float32", "float32"),
        lookback_sessions=20,
        frequency="day",
        read_windows={"ret5": (-5, 0), "vol20": (-19, 0)},
    )
    reordered = FeatureSchema.create(
        names=("vol20", "ret5"),
        dtypes=("float32", "float32"),
        lookback_sessions=20,
        frequency="day",
        read_windows={"ret5": (-5, 0), "vol20": (-19, 0)},
    )

    assert first.schema_hash == same.schema_hash
    assert first.schema_hash != reordered.schema_hash
    assert first.max_lookahead_sessions == 0


def test_artifact_loader_enforces_checksum_format_and_trust(tmp_path):
    model = LinearModelArtifact(("x",), (1.0,), 0.0, (0.0,), (0.0,), (1.0,))
    artifact = model.write(
        tmp_path / "model.json",
        artifact_id="model-v1",
        feature_schema_hash="schema-v1",
        data_version_id="data-v1",
        fold_id="fold-1",
    )
    registry = ArtifactLoaderRegistry()

    loaded = registry.load(artifact)
    assert loaded.feature_names == ("x",)

    with pytest.raises(ValueError, match="trust"):
        registry.load(artifact.__class__(**{**artifact.__dict__, "trust_level": "network_untrusted"}))
    with pytest.raises(ValueError, match="checksum"):
        registry.load(artifact.__class__(**{**artifact.__dict__, "model_checksum": "sha256:wrong"}))


def test_code_fingerprint_changes_for_uncommitted_and_untracked_code(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    clean = compute_code_fingerprint(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    dirty = compute_code_fingerprint(tmp_path)
    (tmp_path / "new_model.py").write_text("NEW = True\n", encoding="utf-8")
    untracked = compute_code_fingerprint(tmp_path)

    assert clean != dirty
    assert dirty != untracked
