"""Logical data-version and provider-lock tests."""

from pathlib import Path

import pytest

from quantx.core.research.data_version import DataVersionResolver, ProviderLock


def make_provider(root: Path):
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features" / "sz000001").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2021-01-04\n2021-01-05\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SZ000001\t2021-01-04\t2021-01-05\n", encoding="utf-8")
    (root / "features" / "sz000001" / "close.day.bin").write_bytes(b"first")


def test_data_version_changes_when_provider_changes(tmp_path):
    provider = tmp_path / "provider"
    make_provider(provider)
    resolver = DataVersionResolver()

    first = resolver.resolve(provider, source_sync_id="sync-1")
    same = resolver.resolve(provider, source_sync_id="sync-1")
    (provider / "features" / "sz000001" / "close.day.bin").write_bytes(b"second-version")
    second = resolver.resolve(provider, source_sync_id="sync-2")

    assert first.version_id == same.version_id
    assert first.version_id != second.version_id
    assert first.provider_uri == str(provider.resolve())


def test_provider_read_lock_blocks_writer(tmp_path):
    provider = tmp_path / "provider"
    make_provider(provider)

    with ProviderLock(provider, shared=True):
        with pytest.raises(BlockingIOError):
            with ProviderLock(provider, shared=False, blocking=False):
                pass

    with ProviderLock(provider, shared=False, blocking=False):
        pass
