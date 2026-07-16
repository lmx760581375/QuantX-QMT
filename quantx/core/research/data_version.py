"""Logical Qlib provider versions and cooperative read/write locking."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import IO
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DataVersion:
    version_id: str
    provider_uri: str
    created_at: datetime
    calendar_hash: str
    instruments_hash: str
    feature_manifest_hash: str
    universe_version_id: str
    adjustment_mode: str
    corporate_action_version: str | None
    source_sync_id: str
    revision_policy: str
    physical_snapshot_id: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result


class ProviderLock:
    """Cooperative flock used by research readers and data-sync writers."""

    def __init__(self, provider_uri: str | Path, *, shared: bool, blocking: bool = True):
        self.provider_uri = Path(provider_uri)
        self.shared = shared
        self.blocking = blocking
        self._handle: IO[str] | None = None

    def __enter__(self) -> "ProviderLock":
        self.provider_uri.mkdir(parents=True, exist_ok=True)
        lock_path = self.provider_uri / ".quantx-provider.lock"
        self._handle = lock_path.open("a+", encoding="utf-8")
        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._handle.fileno(), operation)
        except OSError:
            self._handle.close()
            self._handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class DataVersionResolver:
    def resolve(
        self,
        provider_uri: str | Path,
        *,
        source_sync_id: str,
        adjustment_mode: str = "front_ratio",
        corporate_action_version: str | None = None,
        revision_policy: str = "revised_adjusted_history",
        full_feature_hash: bool = False,
    ) -> DataVersion:
        provider = Path(provider_uri).resolve()
        calendar_path = provider / "calendars" / "day.txt"
        instruments_path = provider / "instruments" / "all.txt"
        if not calendar_path.exists() or not instruments_path.exists():
            raise FileNotFoundError("Qlib provider requires calendars/day.txt and instruments/all.txt")
        calendar_hash = _file_hash(calendar_path)
        instruments_hash = _file_hash(instruments_path)
        feature_manifest_hash = self._feature_manifest(provider, full_hash=full_feature_hash)
        universe_version_id = f"universe:{instruments_hash.removeprefix('sha256:')[:16]}"
        identity = {
            "provider_uri": str(provider),
            "calendar_hash": calendar_hash,
            "instruments_hash": instruments_hash,
            "feature_manifest_hash": feature_manifest_hash,
            "universe_version_id": universe_version_id,
            "adjustment_mode": adjustment_mode,
            "corporate_action_version": corporate_action_version,
            "source_sync_id": source_sync_id,
            "revision_policy": revision_policy,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return DataVersion(
            version_id=f"qmt:{digest[:24]}",
            provider_uri=str(provider),
            created_at=datetime.now(ZoneInfo("Asia/Shanghai")),
            calendar_hash=calendar_hash,
            instruments_hash=instruments_hash,
            feature_manifest_hash=feature_manifest_hash,
            universe_version_id=universe_version_id,
            adjustment_mode=adjustment_mode,
            corporate_action_version=corporate_action_version,
            source_sync_id=source_sync_id,
            revision_policy=revision_policy,
        )

    @staticmethod
    def _feature_manifest(provider: Path, *, full_hash: bool) -> str:
        digest = hashlib.sha256()
        feature_root = provider / "features"
        for path in sorted(feature_root.rglob("*.bin")):
            relative = path.relative_to(provider).as_posix()
            stat = path.stat()
            digest.update(relative.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            if full_hash:
                digest.update(_file_hash(path).encode("ascii"))
            else:
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return f"sha256:{digest.hexdigest()}"


def write_data_version_manifest(
    provider_uri: str | Path,
    data_version: DataVersion,
    *,
    sync_report: dict | None = None,
) -> Path:
    manifest_dir = Path(provider_uri) / ".quantx"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "quantx.data-version",
        "format_version": 1,
        "data_version": data_version.to_dict(),
        "sync_report": dict(sync_report or {}),
    }
    content = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    version_path = manifest_dir / f"{data_version.version_id.replace(':', '-')}.json"
    version_path.write_text(content, encoding="utf-8")
    (manifest_dir / "latest.json").write_text(content, encoding="utf-8")
    return version_path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
