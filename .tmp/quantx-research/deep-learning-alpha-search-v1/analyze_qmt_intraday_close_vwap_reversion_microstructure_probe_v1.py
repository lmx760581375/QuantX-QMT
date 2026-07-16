from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qmt_client import QMTClient, to_qmt_symbol


ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "qmt_intraday_close_vwap_reversion_microstructure_probe_v1_summary.json"
CACHE = ROOT / "qmt_minute_cache_v1"

FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
PERIOD = "5m"
FETCH_POOL = 60
TOP_K = 10
INTERVALS = [3, 5]
RANDOM_TRIALS = 300
SEED = 20260714

VARIANTS = [
    "daily_close_vwap_neg",
    "qmt_tail_recovery",
    "qmt_vwap_absorption",
    "qmt_low_to_close_support",
    "qmt_late_volume_support",
    "qmt_intraday_composite",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CVR = load_module("close_vwap_reversion_replay_v1_for_qmt_probe", ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py")
RT = CVR.RT
POS = CVR.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qmt_date(day: str) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def cache_path(symbol: str, day: str, period: str) -> Path:
    return CACHE / period / f"{symbol}_{day}.csv"


def load_or_fetch_minute(client: QMTClient, symbol: str, day: str, period: str) -> pd.DataFrame:
    path = cache_path(symbol, day, period)
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qsymbol = to_qmt_symbol(symbol)
    client.download_history_data(symbol, day, day, period=period)
    data = client.xtdata.get_market_data_ex(
        field_list=[],
        stock_list=[qsymbol],
        period=period,
        start_time=qmt_date(day),
        end_time=qmt_date(day),
        count=-1,
        dividend_type="none",
        fill_data=False,
    )
    raw = data.get(qsymbol) if isinstance(data, dict) else None
    if raw is None or raw.empty:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame(raw).copy()
        if "time" in frame.columns:
            dt = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="ms")
            frame["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
            frame["date"] = dt.dt.strftime("%Y-%m-%d")
            frame["hhmm"] = dt.dt.strftime("%H:%M")
        else:
            index_text = pd.Series(frame.index).astype(str)
            frame["datetime"] = index_text
            frame["date"] = pd.to_datetime(index_text.str.slice(0, 8), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
            frame["hhmm"] = index_text.str.slice(8, 12).map(lambda text: f"{text[:2]}:{text[2:]}")
    frame.to_csv(path, index=False)
    return frame


def rank_map(rows: list[dict[str, Any]], field: str, ascending: bool = True) -> dict[str, float]:
    values = pd.Series({row["symbol"]: float(row.get(field, np.nan)) for row in rows}).replace([np.inf, -np.inf], np.nan)
    ranks = values.rank(pct=True, ascending=ascending, method="average").fillna(0.5)
    return {sym: float(value) for sym, value in ranks.items()}


def minute_microstructure(frame: pd.DataFrame, preclose: float) -> dict[str, float]:
    if frame.empty or not np.isfinite(preclose) or preclose <= 0:
        return {"minute_available": 0.0}
    frame = frame.sort_values("datetime")
    close = pd.to_numeric(frame.get("close"), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high"), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low"), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    amount = pd.to_numeric(frame.get("amount"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(frame)
    if n < 12 or not np.isfinite(close).any():
        return {"minute_available": 0.0}
    total_vol = float(np.nansum(volume))
    total_amt = float(np.nansum(amount))
    vwap = total_amt / total_vol / 100.0 if total_vol > 0 and total_amt > 0 else float(np.nanmean(close))
    last = float(close[-1])
    day_low = float(np.nanmin(low)) if np.isfinite(low).any() else np.nan
    day_high = float(np.nanmax(high)) if np.isfinite(high).any() else np.nan
    tail_n = min(6, n)
    first_n = min(12, n)
    pm_start = min(max(int(n * 0.5), 1), n - 1)
    first_low = float(np.nanmin(low[:first_n])) if np.isfinite(low[:first_n]).any() else np.nan
    pm_low = float(np.nanmin(low[pm_start:])) if np.isfinite(low[pm_start:]).any() else np.nan
    tail_vol = float(np.nansum(volume[-tail_n:]))
    prev_tail = float(close[-tail_n]) if np.isfinite(close[-tail_n]) else np.nan
    open_px = float(close[0]) if np.isfinite(close[0]) else np.nan
    return {
        "minute_available": 1.0,
        "intraday_vwap_support": float(last / vwap - 1.0) if np.isfinite(last) and np.isfinite(vwap) and vwap > 0 else 0.0,
        "tail_strength_30m": float(last / prev_tail - 1.0) if np.isfinite(last) and np.isfinite(prev_tail) and prev_tail > 0 else 0.0,
        "tail_volume_share_30m": float(tail_vol / total_vol) if total_vol > 0 else 0.0,
        "low_to_close_recovery": float(last / day_low - 1.0) if np.isfinite(last) and np.isfinite(day_low) and day_low > 0 else 0.0,
        "high_to_close_fade": float(last / day_high - 1.0) if np.isfinite(last) and np.isfinite(day_high) and day_high > 0 else 0.0,
        "morning_sell_pressure": float(first_low / preclose - 1.0) if np.isfinite(first_low) and preclose > 0 else 0.0,
        "pm_low_to_close_recovery": float(last / pm_low - 1.0) if np.isfinite(last) and np.isfinite(pm_low) and pm_low > 0 else 0.0,
        "open_to_close_intraday": float(last / open_px - 1.0) if np.isfinite(last) and np.isfinite(open_px) and open_px > 0 else 0.0,
        "close_return": float(last / preclose - 1.0) if np.isfinite(last) else 0.0,
    }


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = {
        "daily": rank_map(rows, "daily_score"),
        "vwap": rank_map(rows, "intraday_vwap_support"),
        "tail": rank_map(rows, "tail_strength_30m"),
        "tail_volume": rank_map(rows, "tail_volume_share_30m"),
        "low_recovery": rank_map(rows, "low_to_close_recovery"),
        "pm_recovery": rank_map(rows, "pm_low_to_close_recovery"),
        "fade_avoid": rank_map(rows, "high_to_close_fade"),
        "morning_drop": rank_map(rows, "morning_sell_pressure", ascending=False),
        "intraday_body": rank_map(rows, "open_to_close_intraday"),
    }
    out = []
    for row in rows:
        sym = row["symbol"]
        daily = ranks["daily"][sym]
        tail_recovery = 0.45 * daily + 0.25 * ranks["tail"][sym] + 0.15 * ranks["low_recovery"][sym] + 0.15 * ranks["vwap"][sym]
        vwap_absorption = 0.45 * daily + 0.25 * ranks["vwap"][sym] + 0.15 * ranks["fade_avoid"][sym] + 0.15 * ranks["intraday_body"][sym]
        low_to_close = 0.40 * daily + 0.25 * ranks["low_recovery"][sym] + 0.20 * ranks["pm_recovery"][sym] + 0.15 * ranks["morning_drop"][sym]
        late_volume = 0.40 * daily + 0.20 * ranks["tail_volume"][sym] + 0.20 * ranks["tail"][sym] + 0.20 * ranks["vwap"][sym]
        composite = 0.25 * daily + 0.20 * ranks["vwap"][sym] + 0.20 * ranks["tail"][sym] + 0.20 * ranks["low_recovery"][sym] + 0.15 * ranks["fade_avoid"][sym]
        enriched = dict(row)
        enriched.update(
            {
                "score_daily_close_vwap_neg": daily,
                "score_qmt_tail_recovery": tail_recovery,
                "score_qmt_vwap_absorption": vwap_absorption,
                "score_qmt_low_to_close_support": low_to_close,
                "score_qmt_late_volume_support": late_volume,
                "score_qmt_intraday_composite": composite,
            }
        )
        out.append(enriched)
    return out


def build_candidates(client: QMTClient, market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    selections = {variant: {} for variant in VARIANTS}
    pools: dict[str, list[str]] = {}
    fetch = {"requested": 0, "available": 0, "missing": 0}
    for session in CVR.sessions(market, FWD_START, FWD_END, interval):
        idx = market["date_index"][session]
        cols = RT.pool_for_session(market, amount20_arr, idx)
        if len(cols) < TOP_K:
            continue
        view = feats[idx - 19:idx + 1, cols, :].mean(axis=0)
        daily_scores = CVR.score_rows(view)["close_vwap_neg"]
        ordered = sorted(range(len(cols)), key=lambda j: (-float(daily_scores[j]), str(market["symbols"][cols[j]])))[:FETCH_POOL]
        rows = []
        for j in ordered:
            col = cols[j]
            sym = market["symbols"][col]
            preclose = float(market["arrays"]["close"][idx, col])
            fetch["requested"] += 1
            try:
                minute = load_or_fetch_minute(client, sym, session, PERIOD)
                day_frame = minute[minute["date"] == session] if not minute.empty and "date" in minute.columns else minute
                minute_feats = minute_microstructure(day_frame, preclose)
            except Exception:
                minute_feats = {"minute_available": 0.0}
            if minute_feats.get("minute_available", 0.0) > 0:
                fetch["available"] += 1
            else:
                fetch["missing"] += 1
            row = {"symbol": sym, "daily_score": float(daily_scores[j])}
            row.update(minute_feats)
            rows.append(row)
        if len(rows) < TOP_K:
            continue
        scored = score_rows(rows)
        pools[session] = [row["symbol"] for row in scored]
        for variant in VARIANTS:
            key = f"score_{variant}"
            ranked = sorted(scored, key=lambda row: (-float(row.get(key, 0.0)), str(row["symbol"])))[:TOP_K]
            selections[variant][session] = [row["symbol"] for row in ranked]
    return {"selections": selections, "pools": pools, "fetch": fetch}


def run_interval(client: QMTClient, market: dict[str, Any], feats: np.ndarray, amount20_arr: np.ndarray, interval: int) -> dict[str, Any]:
    built = build_candidates(client, market, feats, amount20_arr, interval)
    results = {variant: CVR.replay_interval(selection, market, FWD_START, FWD_END, interval)["summary"] for variant, selection in built["selections"].items()}
    rng = np.random.default_rng(SEED + interval)
    random_values = [CVR.replay_interval(CVR.random_selections(built["pools"], rng), market, FWD_START, FWD_END, interval)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = CVR.percentile_summary(random_values)
    baseline = results["daily_close_vwap_neg"]["final_multiple"]
    rows = []
    for variant, summary in results.items():
        rows.append(
            {
                "interval": interval,
                "variant": variant,
                "final_multiple": summary["final_multiple"],
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "remove_best_3": summary["remove_best_period_multiples"].get("remove_best_3", 0.0),
                "beats_random_p95": summary["final_multiple"] > random_summary["p95"],
                "beats_daily_baseline": summary["final_multiple"] > baseline,
                "score": math.log(max(summary["final_multiple"], 1e-9)) + 0.25 * math.log(max(summary["remove_best_period_multiples"].get("remove_best_3", 1e-9), 1e-9)) - abs(summary["max_drawdown"]),
            }
        )
    return {"results": results, "random_pool500_top10": random_summary, "leaderboard": sorted(rows, key=lambda row: -row["score"]), "fetch": built["fetch"], "sessions": len(built["pools"])}


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    client = QMTClient()
    intervals = {str(interval): run_interval(client, market, feats, amount20_arr, interval) for interval in INTERVALS}
    leaderboard = sorted([row for value in intervals.values() for row in value["leaderboard"]], key=lambda row: -row["score"])
    best = leaderboard[0]
    verdict = "qmt_intraday_close_vwap_microstructure_not_enough"
    if best["beats_random_p95"] and best["beats_daily_baseline"] and best["final_multiple"] > 1.25 and best["remove_best_3"] > 1.0:
        verdict = "qmt_intraday_close_vwap_microstructure_candidate_needs_dev_validation"
    out = {
        "experiment": "qmt_intraday_close_vwap_reversion_microstructure_probe_v1",
        "method": "qmt_5m_forward_probe_for_close_vwap_reversion_intraday_absorption_tail_recovery",
        "params": {"forward": [FWD_START, FWD_END], "period": PERIOD, "fetch_pool": FETCH_POOL, "top_k": TOP_K, "intervals": INTERVALS, "random_trials": RANDOM_TRIALS, "seed": SEED, "variants": VARIANTS},
        "inputs_sha256": {"script": sha256(Path(__file__)), "close_vwap_replay_script": sha256(ROOT / "analyze_cross_sectional_close_vwap_reversion_replay_v1.py"), "close_vwap_replay_summary": sha256(ROOT / "cross_sectional_close_vwap_reversion_replay_v1_summary.json") if (ROOT / "cross_sectional_close_vwap_reversion_replay_v1_summary.json").exists() else None},
        "intervals": intervals,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best": best, "leaderboard_top10": leaderboard[:10], "fetch": {k: v["fetch"] for k, v in intervals.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
