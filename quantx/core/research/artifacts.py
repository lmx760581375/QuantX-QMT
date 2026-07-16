"""Auditable model artifact formats and checksum verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    method_type: str
    model_uri: str
    model_format: str
    model_checksum: str
    loader_name: str
    loader_version: str
    trust_level: str
    feature_schema_hash: str
    data_version_id: str
    fold_id: str
    created_at: str


@dataclass(frozen=True)
class LinearModelArtifact:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(dtype=float, copy=True)
        medians = np.asarray(self.medians)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(medians, np.where(missing)[1])
        normalized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        return normalized @ np.asarray(self.coefficients) + self.intercept

    def write(
        self,
        path: str | Path,
        *,
        artifact_id: str,
        feature_schema_hash: str,
        data_version_id: str,
        fold_id: str,
    ) -> ModelArtifact:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "quantx.linear", "format_version": 1, **asdict(self)}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        destination.write_text(content, encoding="utf-8")
        checksum = _checksum(destination.read_bytes())
        return ModelArtifact(
            artifact_id=artifact_id,
            method_type="ridge",
            model_uri=str(destination),
            model_format="quantx_linear_json",
            model_checksum=checksum,
            loader_name="quantx.linear",
            loader_version="1",
            trust_level="local_verified",
            feature_schema_hash=feature_schema_hash,
            data_version_id=data_version_id,
            fold_id=fold_id,
            created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )

    @classmethod
    def load(cls, path: str | Path, expected_checksum: str) -> "LinearModelArtifact":
        source = Path(path)
        content = source.read_bytes()
        actual = _checksum(content)
        if actual != expected_checksum:
            raise ValueError(f"Model artifact checksum mismatch: expected {expected_checksum}, got {actual}")
        payload = json.loads(content)
        if payload.get("format") != "quantx.linear" or payload.get("format_version") != 1:
            raise ValueError("Unsupported linear model artifact format")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            medians=tuple(float(value) for value in payload["medians"]),
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
        )


class LightGBMModelArtifact:
    def __init__(self, model, feature_names: tuple[str, ...], medians: tuple[float, ...]):
        self.model = model
        self.feature_names = feature_names
        self.medians = medians

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(np.asarray(self.medians), np.where(missing)[1])
        return np.asarray(self.model.predict(values), dtype=float)

    def write(
        self,
        path: str | Path,
        *,
        artifact_id: str,
        feature_schema_hash: str,
        data_version_id: str,
        fold_id: str,
    ) -> ModelArtifact:
        base = Path(path).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        model_path = base.with_suffix(".txt")
        preprocessing_path = base.with_name(f"{base.name}.preprocessing.json")
        booster = getattr(self.model, "booster_", self.model)
        booster.save_model(str(model_path))
        preprocessing_path.write_text(
            json.dumps(
                {"feature_names": self.feature_names, "medians": self.medians},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        checksum = _checksum(model_path.read_bytes() + preprocessing_path.read_bytes())
        return ModelArtifact(
            artifact_id=artifact_id,
            method_type="lightgbm",
            model_uri=str(model_path),
            model_format="lightgbm_text",
            model_checksum=checksum,
            loader_name="quantx.lightgbm",
            loader_version="1",
            trust_level="local_verified",
            feature_schema_hash=feature_schema_hash,
            data_version_id=data_version_id,
            fold_id=fold_id,
            created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )


class TorchModelArtifact:
    def __init__(
        self,
        *,
        model,
        architecture: str,
        feature_names: tuple[str, ...],
        lookback_sessions: int,
        medians: tuple[float, ...],
        means: tuple[float, ...],
        scales: tuple[float, ...],
        hidden_size: int,
        history: pd.DataFrame,
    ):
        self.model = model
        self.architecture = architecture
        self.feature_names = feature_names
        self.lookback_sessions = lookback_sessions
        self.medians = medians
        self.means = means
        self.scales = scales
        self.hidden_size = hidden_size
        self.history = history

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch optional dependency is not installed") from exc
        from .sequence import RawWindowFeatureBuilder

        batch = RawWindowFeatureBuilder(self.lookback_sessions).transform(
            frame,
            self.feature_names,
            history=self.history,
        )
        values = _normalize_windows(batch.values, self.medians, self.means, self.scales)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(values, dtype=torch.float32)).cpu().numpy().reshape(-1)

    def write(
        self,
        path: str | Path,
        *,
        artifact_id: str,
        feature_schema_hash: str,
        data_version_id: str,
        fold_id: str,
    ) -> ModelArtifact:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch optional dependency is not installed") from exc
        base = Path(path).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        model_path = base.with_suffix(".pt")
        manifest_path = base.with_name(f"{base.name}.manifest.json")
        torch.save(self.model.state_dict(), model_path)
        manifest = {
            "format": "quantx.torch_state_dict",
            "format_version": 1,
            "architecture": self.architecture,
            "feature_names": self.feature_names,
            "lookback_sessions": self.lookback_sessions,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "hidden_size": self.hidden_size,
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        checksum = _checksum(model_path.read_bytes() + manifest_path.read_bytes())
        return ModelArtifact(
            artifact_id=artifact_id,
            method_type=f"torch_{self.architecture}",
            model_uri=str(model_path),
            model_format="torch_state_dict",
            model_checksum=checksum,
            loader_name="quantx.torch",
            loader_version="1",
            trust_level="local_verified",
            feature_schema_hash=feature_schema_hash,
            data_version_id=data_version_id,
            fold_id=fold_id,
            created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )


def _normalize_windows(values, medians, means, scales):
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    if missing.any():
        feature_indices = np.where(missing)[2]
        result[missing] = np.take(np.asarray(medians), feature_indices)
    return (result - np.asarray(means)[None, None, :]) / np.asarray(scales)[None, None, :]


def write_artifact_manifest(path: str | Path, artifacts: tuple[ModelArtifact, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        [asdict(artifact) for artifact in artifacts],
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    )
    destination.write_text(content + "\n", encoding="utf-8")


class ArtifactLoaderRegistry:
    """Allowlisted artifact loading with mandatory local trust and checksums."""

    allowed_trust_levels = {"local_verified"}

    def load(self, artifact: ModelArtifact):
        if artifact.trust_level not in self.allowed_trust_levels:
            raise ValueError(f"Artifact trust level is not allowed: {artifact.trust_level}")
        if artifact.loader_name == "quantx.linear" and artifact.model_format == "quantx_linear_json":
            return LinearModelArtifact.load(artifact.model_uri, artifact.model_checksum)
        if artifact.loader_name == "quantx.lightgbm" and artifact.model_format == "lightgbm_text":
            return self._load_lightgbm(artifact)
        if artifact.loader_name == "quantx.torch" and artifact.model_format == "torch_state_dict":
            return self._load_torch(artifact)
        raise ValueError(f"Artifact loader/format is not allowlisted: {artifact.loader_name}/{artifact.model_format}")

    @staticmethod
    def _load_lightgbm(artifact: ModelArtifact):
        try:
            from lightgbm import Booster
        except ImportError as exc:
            raise RuntimeError("lightgbm optional dependency is not installed") from exc
        model_path = Path(artifact.model_uri)
        preprocessing_path = model_path.with_name(f"{model_path.stem}.preprocessing.json")
        actual = _checksum(model_path.read_bytes() + preprocessing_path.read_bytes())
        if actual != artifact.model_checksum:
            raise ValueError(f"Model artifact checksum mismatch: expected {artifact.model_checksum}, got {actual}")
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        return LightGBMModelArtifact(
            model=Booster(model_file=str(model_path)),
            feature_names=tuple(preprocessing["feature_names"]),
            medians=tuple(float(value) for value in preprocessing["medians"]),
        )

    @staticmethod
    def _load_torch(artifact: ModelArtifact):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch optional dependency is not installed") from exc
        from .trainer import _build_torch_model

        model_path = Path(artifact.model_uri)
        manifest_path = model_path.with_name(f"{model_path.stem}.manifest.json")
        actual = _checksum(model_path.read_bytes() + manifest_path.read_bytes())
        if actual != artifact.model_checksum:
            raise ValueError(f"Model artifact checksum mismatch: expected {artifact.model_checksum}, got {actual}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = _build_torch_model(
            torch,
            architecture=manifest["architecture"],
            lookback_sessions=int(manifest["lookback_sessions"]),
            feature_count=len(manifest["feature_names"]),
            hidden_size=int(manifest["hidden_size"]),
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return TorchModelArtifact(
            model=model,
            architecture=manifest["architecture"],
            feature_names=tuple(manifest["feature_names"]),
            lookback_sessions=int(manifest["lookback_sessions"]),
            medians=tuple(manifest["medians"]),
            means=tuple(manifest["means"]),
            scales=tuple(manifest["scales"]),
            hidden_size=int(manifest["hidden_size"]),
            history=pd.DataFrame(),
        )


def _checksum(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
