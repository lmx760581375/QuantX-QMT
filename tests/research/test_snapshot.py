"""Promotion physical snapshot tests."""

from quantx.core.research.data_version import DataVersionResolver
from quantx.core.research.snapshot import PhysicalSnapshotManager


def make_provider(root):
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features" / "sz000001").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2021-01-04\n2021-01-05\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SZ000001\t2021-01-04\t2021-01-05\n", encoding="utf-8")
    (root / "features" / "sz000001" / "close.day.bin").write_bytes(b"first")


def test_physical_snapshot_is_verifiable_after_source_changes(tmp_path):
    provider = tmp_path / "provider"
    make_provider(provider)
    version = DataVersionResolver().resolve(provider, source_sync_id="sync-1")
    manager = PhysicalSnapshotManager(tmp_path / "snapshots")

    snapshot = manager.create(provider, version)
    source_feature = provider / "features" / "sz000001" / "close.day.bin"
    source_feature.write_bytes(b"source-now-different")

    assert manager.verify(snapshot.snapshot_id) is True
    assert snapshot.data_version_id == version.version_id
    assert (snapshot.root / "features" / "sz000001" / "close.day.bin").read_bytes() == b"first"
