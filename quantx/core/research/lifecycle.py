"""Promotion lock, sealed holdout ledger, and live-shadow registration."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ExperimentLock:
    research_generation: int
    normalized_research_config_hash: str
    normalized_backtest_config_hash: str
    data_version_id: str
    physical_snapshot_id: str
    universe_version_id: str
    feature_schema_hash: str
    label_spec_hash: str
    model_artifact_checksums: tuple[str, ...]
    seed_set: tuple[int, ...]
    portfolio_config_hash: str
    risk_config_hash: str
    order_planner_config_hash: str
    code_fingerprint: str
    holdout_range: tuple[str, str]

    @property
    def lock_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class ExperimentLifecycle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_lock(self, experiment_lock: ExperimentLock) -> str:
        lock_hash = experiment_lock.lock_hash
        path = self.root / "locks" / f"{lock_hash.removeprefix('sha256:')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**asdict(experiment_lock), "lock_hash": lock_hash}
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return lock_hash

    def claim_sealed_holdout(self, lock_hash: str, *, result_uri: str) -> None:
        lock_path = self.root / "locks" / f"{lock_hash.removeprefix('sha256:')}.json"
        if not lock_path.exists():
            raise ValueError("ExperimentLock does not exist")
        ledger = self.root / "sealed_holdout_ledger.jsonl"
        ledger.touch(exist_ok=True)
        with ledger.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            rows = [json.loads(line) for line in handle if line.strip()]
            if any(row.get("lock_hash") == lock_hash for row in rows):
                raise ValueError("Sealed holdout has already been run for this ExperimentLock")
            handle.seek(0, 2)
            record = {
                "lock_hash": lock_hash,
                "result_uri": result_uri,
                "claimed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LiveShadowRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def register(
        self,
        *,
        strategy_id: str,
        artifact_id: str,
        artifact_checksum: str,
        prediction_store: str,
    ) -> None:
        state = self.load()
        state[strategy_id] = {
            "mode": "live_shadow",
            "artifact_id": artifact_id,
            "artifact_checksum": artifact_checksum,
            "prediction_store": prediction_store,
            "registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "online_update": False,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
