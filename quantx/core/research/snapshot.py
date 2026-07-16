"""Optional physical snapshots for promotion and exact reproduction."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .data_version import DataVersion, ProviderLock


@dataclass(frozen=True)
class PhysicalSnapshot:
    snapshot_id: str
    data_version_id: str
    root: Path
    content_hash: str
    created_at: str


class PhysicalSnapshotManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create(self, provider_uri: str | Path, data_version: DataVersion) -> PhysicalSnapshot:
        source = Path(provider_uri).resolve()
        with ProviderLock(source, shared=True):
            content_hash = _tree_hash(source)
            identity = hashlib.sha256(f"{data_version.version_id}|{content_hash}".encode("utf-8")).hexdigest()[:24]
            snapshot_id = f"snapshot:{identity}"
            destination = self.root / identity
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    source,
                    destination,
                    ignore=shutil.ignore_patterns(".quantx-provider.lock", ".quantx"),
                )
            snapshot = PhysicalSnapshot(
                snapshot_id=snapshot_id,
                data_version_id=data_version.version_id,
                root=destination,
                content_hash=content_hash,
                created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            )
            (destination / "snapshot-manifest.json").write_text(
                json.dumps(
                    {**asdict(snapshot), "root": str(destination)},
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return snapshot

    def verify(self, snapshot_id: str) -> bool:
        identity = snapshot_id.removeprefix("snapshot:")
        destination = self.root / identity
        manifest_path = destination / "snapshot-manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("snapshot_id") == snapshot_id and _tree_hash(destination) == manifest.get("content_hash")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {".quantx-provider.lock", "snapshot-manifest.json"}:
            continue
        if ".quantx" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
