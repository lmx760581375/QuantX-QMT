from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "qmt_intraday_path_event_world_model_cache_only_v1_summary.json"
CACHE = ROOT / "qmt_minute_cache_v1/5m"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

PERIOD = "5m"
INTERVAL = 5
POOL_SIZE = 500
FETCH_TOP = 50
TOP_KS = (10, 20)
MAX_TRAIN_SESSIONS_PER_YEAR = 8
MAX_VALID_SESSIONS = 10
MAX_FORWARD_SESSIONS = 24
MIN_TRAIN_SESSIONS = 18
MIN_VALID_SESSIONS = 6
MIN_FORWARD_SESSIONS = 10
MIN_TRAIN_ROWS = 700
MIN_VALID_ROWS = 200
RANDOM_TRIALS = 200
SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODEL = load_module("qmt_intraday_path_event_world_model_v1_for_cache_only", ROOT / "analyze_qmt_intraday_path_event_world_model_v1.py")
PATH_PROBE = MODEL.PATH_PROBE
CVR = MODEL.CVR
RT = MODEL.RT
POS = MODEL.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(symbol: str, day: str) -> Path:
    return CACHE / f"{symbol}_{day}.csv"


def cached_year_coverage() -> dict[str, Any]:
    files = sorted(CACHE.glob("*.csv"))
    by_year: dict[str, int] = {}
    nonempty_by_year: dict[str, int] = {}
    by_day: dict[str, int] = {}
    nonempty_by_day: dict[str, int] = {}
    empty_files = 0
    for path in files:
        day = path.stem.rsplit("_", 1)[-1]
        by_year[day[:4]] = by_year.get(day[:4], 0) + 1
        by_day[day] = by_day.get(day, 0) + 1
        if path.stat().st_size > 1:
            nonempty_by_year[day[:4]] = nonempty_by_year.get(day[:4], 0) + 1
            nonempty_by_day[day] = nonempty_by_day.get(day, 0) + 1
        else:
            empty_files += 1
    return {
        "files": len(files),
        "nonempty_files": len(files) - empty_files,
        "empty_files": empty_files,
        "by_year": dict(sorted(by_year.items())),
        "nonempty_by_year": dict(sorted(nonempty_by_year.items())),
        "days": len(by_day),
        "nonempty_days": len(nonempty_by_day),
        "first_days": sorted(by_day)[:10],
        "last_days": sorted(by_day)[-10:],
    }


def pick_sessions(market: dict[str, Any], start: str, end: str, max_sessions: int | None, by_year: bool = False) -> list[str]:
    sessions = CVR.sessions(market, start, end, INTERVAL)
    if max_sessions is None or len(sessions) <= max_sessions:
        return sessions
    if by_year:
        picked: list[str] = []
        by_year_sessions: dict[str, list[str]] = {}
        for day in sessions:
            by_year_sessions.setdefault(day[:4], []).append(day)
        for values in by_year_sessions.values():
            limit = min(MAX_TRAIN_SESSIONS_PER_YEAR, len(values))
            indices = np.linspace(0, len(values) - 1, limit).round().astype(int)
            picked.extend(values[int(i)] for i in indices)
        return sorted(set(picked))
    indices = np.linspace(0, len(sessions) - 1, max_sessions).round().astype(int)
    return sorted({sessions[int(i)] for i in indices})


def candidate_order(market: dict[str, Any], amount20_arr: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in cols:
        y = PATH_PROBE.future_return(market, idx, int(col))
        if y is None:
            continue
        labels.append(float(y))
        kept_cols.append(int(col))
    if len(labels) < max(TOP_KS):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    kept = np.asarray(kept_cols, dtype=int)
    scores = PATH_PROBE.CA52.candidate_scores(PATH_PROBE.CA52.feature_frame(market, idx, kept))["mid_trend_volume_not_extreme"]
    order = sorted(range(len(kept)), key=lambda j: (-float(scores[j]), str(market["symbols"][kept[j]])))[:FETCH_TOP]
    return kept[np.asarray(order, dtype=int)], scores[np.asarray(order, dtype=int)]


def build_split_cache_only(market: dict[str, Any], amount20_arr: np.ndarray, split_name: str, sessions: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fetch = {"requested": 0, "cached": 0, "available": 0, "missing": 0, "empty": 0, "errors": 0}
    for session in sessions:
        idx = market["date_index"][session]
        cols, base_scores = candidate_order(market, amount20_arr, idx)
        if len(cols) < max(TOP_KS):
            continue
        for rank_pos, (col, base_score) in enumerate(zip(cols, base_scores, strict=True), 1):
            symbol = str(market["symbols"][int(col)])
            y = PATH_PROBE.future_return(market, idx, int(col))
            if y is None:
                continue
            fetch["requested"] += 1
            path = cache_path(symbol, session)
            if not path.exists():
                fetch["missing"] += 1
                continue
            fetch["cached"] += 1
            if path.stat().st_size <= 1:
                fetch["empty"] += 1
                continue
            preclose = float(market["arrays"]["close"][idx - 1, int(col)])
            try:
                minute = pd.read_csv(path)
                day_frame = minute[minute["date"] == session] if not minute.empty and "date" in minute.columns else minute
                minute_features = PATH_PROBE.intraday_path_features(day_frame, preclose)
                if minute_features.get("minute_available", 0.0) > 0:
                    fetch["available"] += 1
                else:
                    fetch["empty"] += 1
            except Exception:
                fetch["errors"] += 1
                minute_features = {"minute_available": 0.0}
            row = {
                "split": split_name,
                "date": session,
                "symbol": symbol,
                "future_return": float(y),
                "base_score": float(base_score),
                "rank_in_candidate": float((rank_pos - 1) / max(1, FETCH_TOP - 1)),
            }
            row.update({key: float(minute_features.get(key, 0.0)) for key in MODEL.FEATURES if key not in row})
            rows.append(row)
    return {"name": split_name, "sessions": sessions, "rows": rows, "fetch": fetch}


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    rows = split["rows"]
    sessions_with_rows = sorted({row["date"] for row in rows})
    return {
        "planned_sessions": len(split["sessions"]),
        "sessions_with_rows": len(sessions_with_rows),
        "rows": len(rows),
        "available_rows": sum(1 for row in rows if float(row.get("minute_available", 0.0)) > 0),
        "sessions_with_rows_sample": sessions_with_rows[:5] + (["..."] if len(sessions_with_rows) > 10 else []) + sessions_with_rows[-5:],
        "fetch": split["fetch"],
    }


def evaluate_event_rules(splits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, weights in MODEL.EVENTS.items():
        row = {"model": name, "kind": "event_rule"}
        for split_name, split in splits.items():
            scores = MODEL.event_scores(split["rows"], weights)
            row[split_name] = MODEL.metric_from_scores(split["rows"], scores)
        row["valid_score"] = row["valid"]["top_returns"]["top20"] + 0.25 * row["valid"]["top_returns"]["top10"] + 0.005 * row["valid"]["mean_rank_ic"]
        rows.append(row)
    return sorted(rows, key=lambda row: row["valid_score"], reverse=True)


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    sessions = {
        "train": pick_sessions(market, TRAIN_START, TRAIN_END, None, by_year=True),
        "valid": pick_sessions(market, VALID_START, VALID_END, MAX_VALID_SESSIONS),
        "forward": pick_sessions(market, FWD_START, FWD_END, MAX_FORWARD_SESSIONS),
    }
    splits = {name: build_split_cache_only(market, amount20_arr, name, split_sessions) for name, split_sessions in sessions.items()}
    coverage = {name: summarize_split(split) for name, split in splits.items()}
    can_train = (
        coverage["train"]["sessions_with_rows"] >= MIN_TRAIN_SESSIONS
        and coverage["valid"]["sessions_with_rows"] >= MIN_VALID_SESSIONS
        and coverage["forward"]["sessions_with_rows"] >= MIN_FORWARD_SESSIONS
        and coverage["train"]["available_rows"] >= MIN_TRAIN_ROWS
        and coverage["valid"]["available_rows"] >= MIN_VALID_ROWS
    )
    leaderboard = evaluate_event_rules(splits)
    forward_random = MODEL.random_pool_summary(splits["forward"]["rows"]) if splits["forward"]["rows"] else {}
    verdict = "qmt_intraday_path_event_world_model_cache_only_insufficient_valid_coverage"
    if can_train:
        verdict = "qmt_intraday_path_event_world_model_cache_only_can_train_next"
    out = {
        "experiment": "qmt_intraday_path_event_world_model_cache_only_v1",
        "method": "cache_only_qmt_5m_path_event_coverage_and_event_rule_diagnostic_no_download",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "forward": [FWD_START, FWD_END],
            "period": PERIOD,
            "interval": INTERVAL,
            "pool_size": POOL_SIZE,
            "fetch_top": FETCH_TOP,
            "top_ks": TOP_KS,
            "random_trials": RANDOM_TRIALS,
            "seed": SEED,
        },
        "inputs_sha256": {"script": sha256(Path(__file__)), "world_model_script": sha256(ROOT / "analyze_qmt_intraday_path_event_world_model_v1.py")},
        "cache_coverage": cached_year_coverage(),
        "coverage": coverage,
        "can_train": can_train,
        "leaderboard": leaderboard,
        "selected": leaderboard[0] if leaderboard else {},
        "forward_random_top20_label": forward_random,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "cache_coverage": out["cache_coverage"], "coverage": coverage, "can_train": can_train, "selected": out["selected"], "forward_random": forward_random, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
