"""Small storage helpers for memmap-based research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cheap_tree_fingerprint(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            count += 1
            total_size += int(stat.st_size)
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {
        "root": str(root),
        "file_count": int(count),
        "total_size": int(total_size),
        "fingerprint": "sha256:" + digest.hexdigest(),
    }


def memmap_meta(
    *,
    path: Path,
    shape: tuple[int, ...],
    dtype: str,
    axes: tuple[str, ...],
    fields: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "path": str(path),
        "shape": [int(v) for v in shape],
        "dtype": dtype,
        "axes": list(axes),
        "fields": list(fields),
        "created_at": now_shanghai(),
    }
    if path.exists():
        payload["file"] = file_fingerprint(path)
    if extra:
        payload.update(extra)
    return payload


def validate_meta_shape(meta: dict[str, Any], *, shape: tuple[int, ...], dtype: str) -> None:
    if tuple(int(v) for v in meta["shape"]) != tuple(shape):
        raise ValueError(f"Shape mismatch for {meta.get('path')}: {meta['shape']} != {shape}")
    if str(meta["dtype"]) != dtype:
        raise ValueError(f"Dtype mismatch for {meta.get('path')}: {meta['dtype']} != {dtype}")
