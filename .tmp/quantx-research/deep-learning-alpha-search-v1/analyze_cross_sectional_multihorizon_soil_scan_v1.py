from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "cross_sectional_multihorizon_soil_scan_v1_summary.json"
PROVIDER = REPO_ROOT / "data/qlib_data_fixed"

LOAD_START = "2020-01-01"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_K = 10
POOL_SIZE = 500
REBALANCE_INTERVAL = 5
RANDOM_TRIALS = 300
SEED = 20260714

VARIANTS = (
    "smooth_trend",
    "trend_acceleration",
    "vol_compression_breakout",
    "anti_crowded_momentum",
    "low_vol_uptrend",
    "liquidity_momentum",
    "mean_reversion_liquid",
    "hybrid_multihorizon_state",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_market() -> dict[str, Any]:
    universe = _build_universe(PROVIDER, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(PROVIDER)
    calendar = reader.calendar(LOAD_START, FWD_END)
    quotes = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$vwap"], LOAD_START, FWD_END)
    frames = {
        name: quotes[f"${name}"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
        for name in ("open", "high", "low", "close", "volume", "vwap")
    }
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
    return {
        "symbols": symbols,
        "dates": dates,
        "date_index": {day: idx for idx, day in enumerate(dates)},
        "symbol_index": {symbol: idx for idx, symbol in enumerate(symbols)},
        "frames": frames,
        "arrays": {name: frame.to_numpy(dtype=float, copy=True) for name, frame in frames.items()},
    }


def rank_cs(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    ranked = series.rank(pct=True, ascending=ascending, method="average")
    return ranked.fillna(0.5).to_numpy(dtype=float)


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (den > 0))


def past_ret(close: np.ndarray, idx: int, horizon: int) -> np.ndarray:
    return safe_div(close[idx], close[idx - horizon]) - 1.0


def feature_rows(market: dict[str, Any], day: str) -> list[dict[str, Any]]:
    idx = market["date_index"][day]
    arrays = market["arrays"]
    symbols = market["symbols"]
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]
    vwap = arrays["vwap"]
    amount = np.where(np.isfinite(vwap[idx]) & (vwap[idx] > 0), vwap[idx], close[idx]) * volume[idx]
    amount20 = np.nanmean(np.where(np.isfinite(vwap[idx - 19:idx + 1]) & (vwap[idx - 19:idx + 1] > 0), vwap[idx - 19:idx + 1], close[idx - 19:idx + 1]) * volume[idx - 19:idx + 1], axis=0)
    ret1 = past_ret(close, idx, 1)
    ret3 = past_ret(close, idx, 3)
    ret5 = past_ret(close, idx, 5)
    ret10 = past_ret(close, idx, 10)
    ret20 = past_ret(close, idx, 20)
    ret60 = past_ret(close, idx, 60)
    daily_ret20 = safe_div(close[idx - 19:idx + 1], close[idx - 20:idx]) - 1.0
    vol20 = np.nanstd(daily_ret20, axis=0)
    pos_ratio20 = np.nanmean(daily_ret20 > 0, axis=0)
    range20 = np.nanmean(safe_div(high[idx - 19:idx + 1], low[idx - 19:idx + 1]) - 1.0, axis=0)
    max20 = np.nanmax(close[idx - 19:idx + 1], axis=0)
    high20_prev = np.nanmax(high[idx - 20:idx], axis=0)
    drawdown20 = safe_div(close[idx], max20) - 1.0
    close_vs_vwap = safe_div(close[idx], vwap[idx]) - 1.0
    vol_ma20 = np.nanmean(volume[idx - 19:idx + 1], axis=0)
    vol_ratio20 = safe_div(volume[idx], vol_ma20)
    break20 = safe_div(close[idx], high20_prev) - 1.0
    abs_ret1 = np.abs(ret1)
    valid = (
        np.isfinite(open_[idx]) & (open_[idx] > 0)
        & np.isfinite(close[idx]) & (close[idx] > 0)
        & np.isfinite(volume[idx]) & (volume[idx] > 0)
        & np.isfinite(amount20) & (amount20 > 0)
        & np.isfinite(ret20)
        & np.isfinite(ret60)
    )
    liquidity_rank = rank_cs(amount20)
    pool_mask = valid & (liquidity_rank >= max(0.0, 1.0 - POOL_SIZE / max(1, len(symbols))))
    pool_idx = np.flatnonzero(pool_mask)
    if pool_idx.size > POOL_SIZE:
        pool_idx = np.asarray(sorted(pool_idx, key=lambda col: (-float(amount20[col]), str(symbols[col])))[:POOL_SIZE], dtype=int)
    ranks = {
        "ret1": rank_cs(ret1),
        "ret3": rank_cs(ret3),
        "ret5": rank_cs(ret5),
        "ret10": rank_cs(ret10),
        "ret20": rank_cs(ret20),
        "ret60": rank_cs(ret60),
        "low_ret1_abs": rank_cs(abs_ret1, ascending=False),
        "low_ret5": rank_cs(ret5, ascending=False),
        "low_ret20": rank_cs(ret20, ascending=False),
        "vol20_low": rank_cs(vol20, ascending=False),
        "range20_low": rank_cs(range20, ascending=False),
        "drawdown20": rank_cs(drawdown20),
        "pos_ratio20": rank_cs(pos_ratio20),
        "close_vs_vwap": rank_cs(close_vs_vwap),
        "vol_ratio20": rank_cs(vol_ratio20),
        "vol_ratio20_low": rank_cs(vol_ratio20, ascending=False),
        "break20": rank_cs(break20),
        "liquidity": rank_cs(amount20),
    }
    rows = []
    for col in pool_idx:
        smooth_trend = 0.25 * ranks["ret20"][col] + 0.20 * ranks["ret10"][col] + 0.15 * ranks["ret5"][col] + 0.15 * ranks["pos_ratio20"][col] + 0.15 * ranks["drawdown20"][col] + 0.10 * ranks["range20_low"][col]
        trend_accel = 0.25 * ranks["ret5"][col] + 0.20 * ranks["ret10"][col] + 0.15 * ranks["ret20"][col] + 0.15 * ranks["vol_ratio20"][col] + 0.15 * ranks["close_vs_vwap"][col] + 0.10 * ranks["break20"][col]
        vol_break = 0.25 * ranks["vol20_low"][col] + 0.20 * ranks["range20_low"][col] + 0.20 * ranks["ret5"][col] + 0.15 * ranks["close_vs_vwap"][col] + 0.10 * ranks["vol_ratio20"][col] + 0.10 * ranks["break20"][col]
        anti_crowded = 0.25 * ranks["ret20"][col] + 0.20 * ranks["ret10"][col] + 0.20 * ranks["low_ret1_abs"][col] + 0.20 * ranks["vol_ratio20_low"][col] + 0.15 * ranks["range20_low"][col]
        low_vol_up = 0.30 * ranks["vol20_low"][col] + 0.25 * ranks["ret20"][col] + 0.20 * ranks["drawdown20"][col] + 0.15 * ranks["pos_ratio20"][col] + 0.10 * ranks["liquidity"][col]
        liq_mom = 0.30 * ranks["liquidity"][col] + 0.25 * ranks["ret20"][col] + 0.20 * ranks["ret5"][col] + 0.15 * ranks["close_vs_vwap"][col] + 0.10 * ranks["vol_ratio20"][col]
        mean_rev = 0.25 * ranks["low_ret5"][col] + 0.20 * ranks["low_ret20"][col] + 0.20 * ranks["liquidity"][col] + 0.20 * ranks["vol20_low"][col] + 0.15 * ranks["low_ret1_abs"][col]
        hybrid = 0.25 * smooth_trend + 0.20 * anti_crowded + 0.20 * vol_break + 0.20 * low_vol_up + 0.15 * liq_mom
        rows.append(
            {
                "symbol": symbols[col],
                "score_smooth_trend": float(smooth_trend),
                "score_trend_acceleration": float(trend_accel),
                "score_vol_compression_breakout": float(vol_break),
                "score_anti_crowded_momentum": float(anti_crowded),
                "score_low_vol_uptrend": float(low_vol_up),
                "score_liquidity_momentum": float(liq_mom),
                "score_mean_reversion_liquid": float(mean_rev),
                "score_hybrid_multihorizon_state": float(hybrid),
                "amount20": float(amount20[col]) if np.isfinite(amount20[col]) else 0.0,
                "ret5": float(ret5[col]) if np.isfinite(ret5[col]) else 0.0,
                "ret20": float(ret20[col]) if np.isfinite(ret20[col]) else 0.0,
                "ret60": float(ret60[col]) if np.isfinite(ret60[col]) else 0.0,
            }
        )
    return rows


def tradable(sym: str, date_idx: int, signal_idx: int, market: dict[str, Any]) -> tuple[bool, str]:
    col = market["symbol_index"].get(sym)
    if col is None:
        return False, "missing_or_suspended"
    arrays = market["arrays"]
    open_price = float(arrays["open"][date_idx, col])
    volume = float(arrays["volume"][date_idx, col])
    preclose = float(arrays["close"][signal_idx, col])
    if not np.isfinite(open_price) or open_price <= 0 or not np.isfinite(volume) or volume <= 0:
        return False, "missing_or_suspended"
    if not np.isfinite(preclose) or preclose <= 0:
        return False, "missing_preclose"
    if abs(open_price / preclose - 1.0) > 0.095:
        return False, "price_jump_abs_gt_9p5"
    return True, "ok"


def annual_returns(curve: list[dict[str, Any]]) -> dict[str, float]:
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in curve:
        by_year.setdefault(row["date"][:4], []).append(row)
    return {year: rows[-1]["nav"] / rows[0]["nav_before"] - 1.0 for year, rows in sorted(by_year.items()) if rows[0]["nav_before"] > 0}


def summarize(curve: list[dict[str, Any]], trades: list[dict[str, Any]], rejects: dict[str, int]) -> dict[str, Any]:
    returns = [row["daily_return"] for row in curve]
    final = float(curve[-1]["nav"]) if curve else 1.0
    years = annual_returns(curve)
    n_years = max(1e-9, len(curve) / 242.0)
    annual_return = final ** (1.0 / n_years) - 1.0
    vol = pstdev(returns) * math.sqrt(252.0) if len(returns) > 1 else 0.0
    period_returns = sorted([float(row.get("period_return", 0.0)) for row in trades], reverse=True)
    remove = {}
    for n in (1, 3, 5, 10):
        kept = period_returns[n:]
        remove[f"remove_best_{n}"] = float(np.prod([1.0 + value for value in kept])) if kept else 1.0
    return {
        "final_multiple": final,
        "total_return": final - 1.0,
        "annual_return": annual_return,
        "annual_volatility": vol,
        "sharpe": annual_return / vol if vol > 0 else 0.0,
        "max_drawdown": min((row["drawdown"] for row in curve), default=0.0),
        "annual_returns": years,
        "all_years_positive": all(value > 0 for value in years.values()),
        "avg_position_count": mean(row["position_count"] for row in curve) if curve else 0.0,
        "max_position_count": max((row["position_count"] for row in curve), default=0),
        "trade_count": len(trades),
        "days": len(curve),
        "rejects": rejects,
        "remove_best_period_multiples": remove,
    }


def replay(selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    dates = market["dates"]
    arrays = market["arrays"]
    symbol_index = market["symbol_index"]
    active = sorted(day for day in selections if start <= day <= end)
    if not active:
        return {"summary": summarize([], [], {}), "trades_tail": []}
    start_idx = market["date_index"][active[0]] + 1
    end_idx = min(len(dates) - 1, market["date_index"][active[-1]] + 1)
    nav = 1.0
    peak = 1.0
    positions: dict[str, float] = {}
    last_rebalance_idx: int | None = None
    active_trade: int | None = None
    curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rejects = {"candidate_total": 0, "selected_total": 0, "missing_or_suspended": 0, "missing_preclose": 0, "price_jump_abs_gt_9p5": 0, "empty_rebalances": 0}
    for idx in range(start_idx, end_idx + 1):
        date = dates[idx]
        nav_before = nav
        if idx > start_idx and positions:
            daily_ret = 0.0
            for sym, weight in positions.items():
                col = symbol_index.get(sym)
                if col is None:
                    continue
                prev_open = float(arrays["open"][idx - 1, col])
                cur_open = float(arrays["open"][idx, col])
                if np.isfinite(prev_open) and np.isfinite(cur_open) and prev_open > 0 and cur_open > 0:
                    daily_ret += weight * (cur_open / prev_open - 1.0)
            nav *= 1.0 + daily_ret
        signal_date = dates[idx - 1]
        records = selections.get(signal_date)
        if records and (last_rebalance_idx is None or idx - last_rebalance_idx >= REBALANCE_INTERVAL):
            if active_trade is not None and trades[active_trade].get("end_nav") is None:
                trades[active_trade]["end_nav"] = nav
                start_nav = trades[active_trade]["start_nav"]
                trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
            picked = []
            for sym in records[:TOP_K]:
                rejects["candidate_total"] += 1
                ok, reason = tradable(sym, idx, idx - 1, market)
                if ok:
                    picked.append(sym)
                else:
                    rejects[reason] += 1
            target = {sym: 1.0 / len(picked) for sym in picked} if picked else {}
            if not target:
                rejects["empty_rebalances"] += 1
            rejects["selected_total"] += len(target)
            universe = set(target) | set(positions)
            buy = sum(max(0.0, target.get(sym, 0.0) - positions.get(sym, 0.0)) for sym in universe)
            sell = sum(max(0.0, positions.get(sym, 0.0) - target.get(sym, 0.0)) for sym in universe)
            nav *= max(0.0, 1.0 - buy * 0.00052 - sell * 0.00102)
            trades.append({"date": date, "signal_date": signal_date, "count": len(target), "start_nav": nav, "end_nav": None})
            active_trade = len(trades) - 1
            positions = target
            last_rebalance_idx = idx
        peak = max(peak, nav)
        curve.append({"date": date, "nav_before": nav_before, "nav": nav, "daily_return": nav / nav_before - 1.0 if nav_before > 0 else 0.0, "drawdown": nav / peak - 1.0, "position_count": len(positions)})
    if active_trade is not None and trades[active_trade].get("end_nav") is None:
        trades[active_trade]["end_nav"] = nav
        start_nav = trades[active_trade]["start_nav"]
        trades[active_trade]["period_return"] = nav / start_nav - 1.0 if start_nav > 0 else 0.0
    return {"summary": summarize(curve, trades, rejects), "trades_tail": trades[-5:]}


def random_selections(pools: dict[str, list[str]], rng: np.random.Generator) -> dict[str, list[str]]:
    return {session: [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=min(TOP_K, len(pool)), replace=False)] for session, pool in pools.items() if len(pool) >= TOP_K}


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def build_split(market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    sessions = [day for day in market["dates"] if start <= day <= end]
    due5 = sessions[::REBALANCE_INTERVAL]
    rows_by_session = {session: feature_rows(market, session) for session in due5 if market["date_index"][session] >= 80}
    selections: dict[str, dict[str, list[str]]] = {variant: {} for variant in VARIANTS}
    pools: dict[str, list[str]] = {}
    label_means: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    for session, rows in rows_by_session.items():
        pools[session] = [row["symbol"] for row in rows]
        signal_idx = market["date_index"][session]
        exit_idx = min(len(market["dates"]) - 1, signal_idx + REBALANCE_INTERVAL + 1)
        for variant in VARIANTS:
            key = f"score_{variant}"
            ranked = sorted(rows, key=lambda row: (-float(row.get(key, 0.0)), str(row["symbol"])))[:TOP_K]
            picks = [row["symbol"] for row in ranked]
            selections[variant][session] = picks
            rets = []
            for sym in picks:
                col = market["symbol_index"][sym]
                entry = market["arrays"]["open"][signal_idx + 1, col] if signal_idx + 1 < len(market["dates"]) else np.nan
                exit_ = market["arrays"]["open"][exit_idx, col]
                if np.isfinite(entry) and np.isfinite(exit_) and entry > 0 and exit_ > 0:
                    rets.append(float(exit_ / entry - 1.0))
            if rets:
                label_means[variant].append(float(np.mean(rets)))
    results = {variant: replay(sel, market, start, end)["summary"] for variant, sel in selections.items()}
    rng = np.random.default_rng(SEED + (1 if start.startswith("2026") else 0))
    random_values = [replay(random_selections(pools, rng), market, start, end)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = percentile_summary(random_values)
    best_variant = max(VARIANTS, key=lambda variant: results[variant]["final_multiple"])
    gates = {
        variant: {
            "beats_random_p95": results[variant]["final_multiple"] > random_summary["p95"],
            "beats_random_p99": results[variant]["final_multiple"] > random_summary["p99"],
            "all_years_positive": bool(results[variant]["all_years_positive"]),
            "avg_pos_gt_5": results[variant]["avg_position_count"] > 5,
            "remove_best_3_positive": results[variant]["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0,
        }
        for variant in VARIANTS
    }
    return {
        "sessions": len(rows_by_session),
        "pool_size_mean": mean(len(rows) for rows in rows_by_session.values()) if rows_by_session else 0.0,
        "results": results,
        "random_pool500_top10_strict": random_summary,
        "label_mean_5d_open_to_open": {variant: float(np.mean(values)) if values else 0.0 for variant, values in label_means.items()},
        "gates": gates,
        "best_variant": best_variant,
    }


def main() -> None:
    started = time.perf_counter()
    market = load_market()
    dev = build_split(market, DEV_START, DEV_END)
    forward = build_split(market, FWD_START, FWD_END)
    best = forward["best_variant"]
    verdict = "multihorizon_soil_not_found"
    if (
        dev["gates"][best]["beats_random_p95"]
        and dev["results"][best]["final_multiple"] >= 20.0
        and dev["gates"][best]["all_years_positive"]
        and forward["gates"][best]["beats_random_p95"]
        and forward["results"][best]["final_multiple"] > 1.0
        and forward["gates"][best]["remove_best_3_positive"]
    ):
        verdict = "multihorizon_soil_candidate_needs_modeling"
    out = {
        "experiment": "cross_sectional_multihorizon_soil_scan_v1",
        "method": "all_mainboard_pool500_daily_multihorizon_state_scores_top10_rebalance5_dev_forward_random_ablation",
        "params": {
            "load_start": LOAD_START,
            "dev_start": DEV_START,
            "dev_end": DEV_END,
            "forward_start": FWD_START,
            "forward_end": FWD_END,
            "top_k": TOP_K,
            "pool_size": POOL_SIZE,
            "rebalance_interval": REBALANCE_INTERVAL,
            "random_trials": RANDOM_TRIALS,
            "seed": SEED,
        },
        "inputs_sha256": {"script": sha256(Path(__file__))},
        "dev": dev,
        "forward": forward,
        "best_forward_variant": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_forward_variant": best, "dev_best": dev["best_variant"], "forward_best": forward["best_variant"], "dev_result": dev["results"][dev["best_variant"]], "forward_result": forward["results"][best], "forward_random": forward["random_pool500_top10_strict"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
