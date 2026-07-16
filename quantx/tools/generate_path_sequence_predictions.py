"""Generate PredictionStore files for path-sequence strategy tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.predictions import PredictionContract, PredictionRecord, PredictionStore


GRID_0878 = {"base": 0.30, "path_ev": 0.30, "path_spread": 0.20, "path_topq": 0.05, "exec_cls": 0.15}
ARTIFACT_ID = "path_sequence_top7_bridge_v1"
FEATURE_SCHEMA_HASH = "path_sequence_v2_with_aux_104_grid0878_top7"


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    panel_path = Path(args.panel)
    frame = pd.read_parquet(panel_path).replace([np.inf, -np.inf], np.nan)
    frame["session"] = frame["session"].astype(str)
    signal_dates = resolve_signal_dates(frame, args.signal_date)
    scored_days = [score_day(frame, signal_date, args, panel_path) for signal_date in signal_dates]
    records = [record for day in scored_days for record in records_for_day(day, str(day["session"].iloc[0]), args)]
    contracts = [contract(signal_dates[0], args)] if len(signal_dates) == 1 else []
    store = PredictionStore(records, contracts=contracts)
    output_path = Path(args.output)
    checksum = "dry-run"
    if not args.dry_run:
        checksum = store.write(output_path)
    manifest = {
        "artifact_id": args.artifact_id,
        "checksum": checksum,
        "feature_schema_hash": args.feature_schema_hash,
        "signal_date": signal_dates[-1],
        "signal_start": signal_dates[0],
        "signal_count": len(signal_dates),
        "output": str(output_path),
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "score_mode": args.score_mode,
        "topk_pool": int(args.topk_pool),
        "record_count": len(records),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "notes": [
            "This bridge requires an already materialized causal scored panel.",
            "DL mode fails if the requested DL score column is absent; it never silently retrains inside daily production.",
            "When signal-date=all, contracts are omitted because historical rows may come from multiple walk-forward folds.",
        ],
    }
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, **manifest, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2_with_aux.parquet")
    parser.add_argument("--signal-date", default="latest", help="YYYY-MM-DD, latest, or all from panel.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--score-mode", default="grid0878", choices=["grid0878", "dl_blend"])
    parser.add_argument("--dl-score-col", default="score_dl_pairwise_pct")
    parser.add_argument("--dl-alpha", type=float, default=0.03)
    parser.add_argument("--topk-pool", type=int, default=200)
    parser.add_argument("--min-records", type=int, default=20)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--artifact-id", default=ARTIFACT_ID)
    parser.add_argument("--feature-schema-hash", default=FEATURE_SCHEMA_HASH)
    parser.add_argument("--fold-id", default="live-bridge")
    parser.add_argument("--evaluation-tier", default="live_shadow")
    parser.add_argument("--training-information-end", help="Defaults to previous available panel session before signal date.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs but do not write files.")
    return parser.parse_args(argv)


def resolve_signal_dates(frame: pd.DataFrame, value: str) -> list[str]:
    mode = str(value).lower()
    sessions = sorted(str(v) for v in frame["session"].dropna().unique())
    if not sessions:
        raise ValueError("No sessions in panel")
    if mode == "latest":
        return [sessions[-1]]
    if mode == "all":
        return sessions
    return [str(value)]


def score_day(frame: pd.DataFrame, signal_date: str, args: argparse.Namespace, panel_path: Path) -> pd.DataFrame:
    day = frame.loc[frame["session"] == signal_date].copy()
    if day.empty:
        raise ValueError(f"No rows for signal date {signal_date} in {panel_path}")
    score_col = build_score(day, args)
    day = day.assign(_prediction_score=score_col).dropna(subset=["_prediction_score"])
    day = day.sort_values(["_prediction_score", "instrument"], ascending=[False, True]).head(int(args.topk_pool))
    if len(day) < int(args.min_records):
        raise ValueError(f"Only {len(day)} prediction rows for {signal_date}; expected at least {args.min_records}")
    return day


def build_score(day: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    grid = sum(float(weight) * one_dim(day, f"part_{part}") for part, weight in GRID_0878.items())
    if args.score_mode == "grid0878":
        return grid
    if args.dl_score_col not in day.columns:
        raise ValueError(
            f"DL score column {args.dl_score_col!r} is absent. Generate a model artifact/prediction panel first; "
            "daily production will not retrain implicitly."
        )
    dl = pd.to_numeric(day[args.dl_score_col], errors="coerce").fillna(0.5)
    alpha = float(args.dl_alpha)
    return (1.0 - alpha) * grid + alpha * dl


def one_dim(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise ValueError(f"Missing score part column: {col}")
    values = pd.to_numeric(frame[col], errors="coerce")
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=frame.index, dtype=float)
    ranked = values.rank(pct=True, method="first")
    return ranked.fillna(0.5)


def records_for_day(day: pd.DataFrame, signal_date: str, args: argparse.Namespace) -> list[PredictionRecord]:
    signal_time = MarketTime(
        session=signal_date,
        phase=SessionPhase.AFTER_CLOSE,
        timestamp=datetime.strptime(signal_date, "%Y-%m-%d").replace(hour=15, minute=0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    rows = day[["instrument", "_prediction_score"]].itertuples(index=False, name=None)
    return [
        PredictionRecord(
            signal_time=signal_time,
            instrument=str(instrument),
            score=float(score),
            prediction_horizon=int(args.prediction_horizon),
            artifact_id=str(args.artifact_id),
            fold_id=str(args.fold_id),
            feature_schema_hash=str(args.feature_schema_hash),
            evaluation_tier=str(args.evaluation_tier),
            rank=index,
        )
        for index, (instrument, score) in enumerate(rows, start=1)
    ]


def contract(signal_date: str, args: argparse.Namespace) -> PredictionContract:
    return PredictionContract(
        artifact_id=str(args.artifact_id),
        fold_id=str(args.fold_id),
        prediction_start=signal_date,
        prediction_end=signal_date,
        training_information_end=str(args.training_information_end or previous_session(args.panel, signal_date)),
        feature_schema_hash=str(args.feature_schema_hash),
    )


def previous_session(panel: str, signal_date: str) -> str:
    frame = pd.read_parquet(panel, columns=["session"])
    sessions = sorted(str(v) for v in frame["session"].dropna().unique() if str(v) < signal_date)
    if not sessions:
        raise ValueError("Cannot infer training_information_end before signal date")
    return sessions[-1]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
