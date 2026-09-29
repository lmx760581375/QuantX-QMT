"""校验正式权重与冻结参考指标。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from quantx_reward.config import REPO_ROOT

DEFAULT_WEIGHT_BUNDLE = "reward_v1"
DEFAULT_ENVIRONMENT_REFERENCE = REPO_ROOT / "environment.reference.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_weight_manifest(bundle: str) -> tuple[Path, dict[str, Any]]:
    root = REPO_ROOT / "weights" / str(bundle)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing weight manifest: {manifest_path}")
    return root, json.loads(manifest_path.read_text(encoding="utf-8"))


def _asset_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: dict(manifest[key])
        for key in ("reward_checkpoint", "frozen_autoencoder", "preprocessing_scaler")
        if key in manifest
    }


def verify_weights(bundle: str = DEFAULT_WEIGHT_BUNDLE) -> dict[str, Any]:
    root, manifest = _load_weight_manifest(bundle)
    checks = {}
    for key, spec in _asset_specs(manifest).items():
        path = root / spec["path"]
        actual = sha256(path) if path.is_file() else None
        checks[key] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "ok": path.is_file() and actual == spec["sha256"],
        }
    return {
        "ok": all(item["ok"] for item in checks.values()),
        "bundle": str(bundle),
        "checks": checks,
    }


def restore_weights(
    *,
    force: bool = False,
    bundle: str = DEFAULT_WEIGHT_BUNDLE,
) -> dict[str, Any]:
    root, manifest = _load_weight_manifest(bundle)
    restored = {}
    for key, spec in _asset_specs(manifest).items():
        output = root / spec["path"]
        if output.is_file() and not force and sha256(output) == spec["sha256"]:
            restored[key] = {
                "path": str(output),
                "status": "verified" if key == "preprocessing_scaler" else "already_valid",
            }
            continue
        source_bundle = spec.get("source_bundle")
        chunk_glob = spec.get("chunk_glob")
        if source_bundle:
            restore_weights(bundle=str(source_bundle))
        elif chunk_glob:
            chunks = sorted(root.glob(str(chunk_glob)))
            if not chunks:
                raise FileNotFoundError(f"No weight chunks found for {key}: {chunk_glob}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("wb") as destination:
                for chunk in chunks:
                    with chunk.open("rb") as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            destination.write(block)
        elif not output.is_file():
            raise FileNotFoundError(f"Missing weight asset for {key}: {output}")
        actual = sha256(output)
        if actual != spec["sha256"]:
            if chunk_glob:
                output.unlink(missing_ok=True)
            raise RuntimeError(f"Restored weight hash mismatch for {key}: {actual}")
        restored[key] = {
            "path": str(output),
            "status": "restored" if chunk_glob else "verified",
            **({"chunks": len(chunks)} if chunk_glob else {}),
        }
    return {"ok": True, "bundle": str(bundle), "restored": restored}


def compare_metrics(actual_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    actual = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    fields = ("total_return", "annual_return", "max_drawdown", "sharpe", "trade_count")
    checks = {}
    for field in fields:
        actual_value = actual.get(field)
        reference_value = reference.get(field)
        if isinstance(actual_value, (int, float)) and isinstance(reference_value, (int, float)):
            ok = abs(float(actual_value) - float(reference_value)) <= 1.0e-12
        else:
            ok = actual_value == reference_value
        checks[field] = {
            "actual": actual_value,
            "reference": reference_value,
            "ok": ok,
        }
    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks}


def verify_environment(
    reference_path: str | Path = DEFAULT_ENVIRONMENT_REFERENCE,
    *,
    strict_gpu: bool = False,
) -> dict[str, Any]:
    reference_path = Path(reference_path).expanduser().resolve()
    expected = json.loads(reference_path.read_text(encoding="utf-8"))
    packages = {}
    for name, expected_version in dict(expected.get("packages") or {}).items():
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        packages[name] = {
            "expected": expected_version,
            "actual": actual_version,
            "ok": actual_version == expected_version,
        }
    python_check = {
        "expected": expected.get("python"),
        "actual": platform.python_version(),
        "ok": platform.python_version() == expected.get("python"),
    }
    runtime: dict[str, Any] = {"python": python_check, "packages": packages}
    if strict_gpu:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        runtime["cuda"] = {
            "expected": expected.get("cuda"),
            "actual": torch.version.cuda,
            "ok": torch.version.cuda == expected.get("cuda"),
        }
        runtime["gpu"] = {
            "expected": expected.get("reference_gpu"),
            "actual": gpu_name,
            "ok": gpu_name == expected.get("reference_gpu"),
        }
    checks = [python_check["ok"], *(item["ok"] for item in packages.values())]
    if strict_gpu:
        checks.extend([runtime["cuda"]["ok"], runtime["gpu"]["ok"]])
    return {
        "ok": all(checks),
        "reference": str(reference_path),
        "runtime": runtime,
    }
