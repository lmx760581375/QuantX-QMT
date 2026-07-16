"""Safe, deterministic storage for out-of-sample model predictions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .clock import MarketTime, SessionPhase


PREDICTION_STORE_FORMAT_VERSION = 1
EVALUATION_TIERS = {"development_oos", "sealed_holdout", "live_shadow", "exploratory"}


@dataclass(frozen=True)
class PredictionContract:
    """Declares the out-of-sample boundary for one frozen model fold."""

    artifact_id: str
    fold_id: str
    prediction_start: str
    prediction_end: str
    training_information_end: str
    feature_schema_hash: str

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.fold_id or not self.feature_schema_hash:
            raise ValueError("Prediction contract identity fields are required")
        prediction_start = self._parse_session(self.prediction_start, "prediction_start")
        prediction_end = self._parse_session(self.prediction_end, "prediction_end")
        training_end = self._parse_session(self.training_information_end, "training_information_end")
        if prediction_start > prediction_end:
            raise ValueError("Prediction contract start cannot exceed end")
        if training_end >= prediction_start:
            raise ValueError("Training information must end before the prediction boundary")

    def accepts(self, record: "PredictionRecord") -> None:
        if record.feature_schema_hash != self.feature_schema_hash:
            raise ValueError(
                f"Prediction feature schema mismatch: expected {self.feature_schema_hash}, "
                f"got {record.feature_schema_hash}"
            )
        signal_session = self._parse_session(record.signal_session, "signal_session")
        prediction_start = self._parse_session(self.prediction_start, "prediction_start")
        prediction_end = self._parse_session(self.prediction_end, "prediction_end")
        if not prediction_start <= signal_session <= prediction_end:
            raise ValueError(
                f"Prediction signal session {record.signal_session} is outside contract range "
                f"[{self.prediction_start}, {self.prediction_end}]"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "PredictionContract":
        return cls(
            artifact_id=str(row["artifact_id"]),
            fold_id=str(row["fold_id"]),
            prediction_start=str(row["prediction_start"]),
            prediction_end=str(row["prediction_end"]),
            training_information_end=str(row["training_information_end"]),
            feature_schema_hash=str(row["feature_schema_hash"]),
        )

    @staticmethod
    def _parse_session(value: str, field_name: str):
        from datetime import date

        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{field_name} must use YYYY-MM-DD format")
        return parsed


@dataclass(frozen=True)
class PredictionRecord:
    signal_time: MarketTime
    instrument: str
    score: float
    prediction_horizon: int
    artifact_id: str
    fold_id: str
    feature_schema_hash: str
    evaluation_tier: str = "development_oos"
    rank: int | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if self.signal_time.phase != SessionPhase.AFTER_CLOSE:
            raise ValueError("Prediction signal_time must be AFTER_CLOSE")
        if not self.instrument or not self.artifact_id or not self.fold_id or not self.feature_schema_hash:
            raise ValueError("Prediction identity fields are required")
        if not math.isfinite(float(self.score)):
            raise ValueError("Prediction score must be finite")
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        if self.rank is not None and self.rank < 1:
            raise ValueError("Prediction rank must be positive")
        if self.uncertainty is not None and (not math.isfinite(float(self.uncertainty)) or float(self.uncertainty) < 0):
            raise ValueError("Prediction uncertainty must be finite and non-negative")
        if self.evaluation_tier not in EVALUATION_TIERS:
            raise ValueError(f"Unsupported evaluation tier: {self.evaluation_tier}")

    @property
    def signal_session(self) -> str:
        return self.signal_time.session

    def to_dict(self) -> dict:
        row = asdict(self)
        row["signal_time"] = {
            "session": self.signal_time.session,
            "phase": self.signal_time.phase.value,
            "timestamp": self.signal_time.timestamp.isoformat(),
        }
        return row

    @classmethod
    def from_dict(cls, row: dict) -> "PredictionRecord":
        from datetime import datetime
        from zoneinfo import ZoneInfo

        raw_time = dict(row["signal_time"])
        timestamp = datetime.fromisoformat(str(raw_time["timestamp"]))
        return cls(
            signal_time=MarketTime(
                session=str(raw_time["session"]),
                phase=SessionPhase(str(raw_time["phase"])),
                timestamp=timestamp.astimezone(ZoneInfo("Asia/Shanghai")),
            ),
            instrument=str(row["instrument"]),
            score=float(row["score"]),
            prediction_horizon=int(row["prediction_horizon"]),
            artifact_id=str(row["artifact_id"]),
            fold_id=str(row["fold_id"]),
            feature_schema_hash=str(row["feature_schema_hash"]),
            evaluation_tier=str(row.get("evaluation_tier", "development_oos")),
            rank=int(row["rank"]) if row.get("rank") is not None else None,
            uncertainty=float(row["uncertainty"]) if row.get("uncertainty") is not None else None,
        )


class PredictionStore:
    """In-memory index backed by canonical JSON, never pickle."""

    def __init__(
        self,
        records: Iterable[PredictionRecord] = (),
        *,
        contracts: Iterable[PredictionContract] = (),
    ):
        self._records: dict[tuple[str, str], PredictionRecord] = {}
        self._records_by_session: dict[str, list[PredictionRecord]] = {}
        self._records_by_session_artifact: dict[tuple[str, str], list[PredictionRecord]] = {}
        self._contracts: dict[tuple[str, str], PredictionContract] = {}
        for contract in contracts:
            key = (contract.artifact_id, contract.fold_id)
            if key in self._contracts:
                raise ValueError(f"Duplicate prediction contract: {contract.artifact_id}/{contract.fold_id}")
            self._contracts[key] = contract
        for record in records:
            self.add(record)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.artifact_id for record in self._records.values()}))

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(sorted({record.instrument for record in self._records.values()}))

    @property
    def contracts(self) -> tuple[PredictionContract, ...]:
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    @property
    def signal_sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._records_by_session))

    def add(self, record: PredictionRecord) -> None:
        if self._contracts:
            contract = self._contracts.get((record.artifact_id, record.fold_id))
            if contract is None:
                raise ValueError(f"No prediction contract for {record.artifact_id}/{record.fold_id}")
            contract.accepts(record)
        key = (record.signal_time.timestamp.isoformat(), record.instrument)
        if key in self._records:
            existing = self._records[key]
            raise ValueError(
                "overlapping prediction for "
                f"{record.signal_session}/{record.instrument}: {existing.artifact_id}/{existing.fold_id} "
                f"vs {record.artifact_id}/{record.fold_id}"
            )
        self._records[key] = record
        self._records_by_session.setdefault(record.signal_session, []).append(record)
        self._records_by_session_artifact.setdefault((record.signal_session, record.artifact_id), []).append(record)

    def records_for(self, signal_session: str, *, artifact_id: str | None = None) -> tuple[PredictionRecord, ...]:
        records = (
            self._records_by_session.get(signal_session, [])
            if artifact_id is None
            else self._records_by_session_artifact.get((signal_session, artifact_id), [])
        )
        return tuple(sorted(records, key=lambda record: (-record.score, record.instrument)))

    def top_instruments(self, *, artifact_id: str | None = None, top_k: int | None = None) -> tuple[str, ...]:
        selected: list[str] = []
        seen: set[str] = set()
        for session in self.signal_sessions:
            records = self.records_for(session, artifact_id=artifact_id)
            if top_k is not None:
                records = records[: int(top_k)]
            for record in records:
                if record.instrument not in seen:
                    selected.append(record.instrument)
                    seen.add(record.instrument)
        return tuple(selected)

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()

        def write_chunk(handle, value: str) -> None:
            content = value.encode("utf-8")
            handle.write(content)
            digest.update(content)

        def encode(value) -> str:
            return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

        with destination.open("wb") as handle:
            write_chunk(handle, '{"contracts":')
            write_chunk(handle, encode([contract.to_dict() for contract in self.contracts]))
            write_chunk(handle, ',"format":"quantx.predictions","format_version":')
            write_chunk(handle, str(PREDICTION_STORE_FORMAT_VERSION))
            write_chunk(handle, ',"records":[')
            for index, record in enumerate(self._iter_ordered_records()):
                if index:
                    write_chunk(handle, ",")
                write_chunk(handle, encode(record.to_dict()))
            write_chunk(handle, "]}\n")
        return f"sha256:{digest.hexdigest()}"

    @classmethod
    def load(cls, path: str | Path, *, expected_checksum: str | None = None) -> "PredictionStore":
        source = Path(path)
        content = source.read_bytes()
        actual_checksum = cls._checksum(content)
        if expected_checksum is not None and expected_checksum != actual_checksum:
            raise ValueError(f"PredictionStore checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
        payload = json.loads(content)
        if payload.get("format") != "quantx.predictions":
            raise ValueError("Unsupported prediction store format")
        if int(payload.get("format_version", -1)) != PREDICTION_STORE_FORMAT_VERSION:
            raise ValueError("Unsupported prediction store format version")
        return cls(
            (PredictionRecord.from_dict(row) for row in payload.get("records", [])),
            contracts=(PredictionContract.from_dict(row) for row in payload.get("contracts", [])),
        )

    def _iter_ordered_records(self):
        return iter(
            sorted(
                self._records.values(),
                key=lambda record: (record.signal_time.timestamp, record.instrument, record.artifact_id),
            )
        )

    @staticmethod
    def _checksum(content: bytes) -> str:
        return f"sha256:{hashlib.sha256(content).hexdigest()}"
