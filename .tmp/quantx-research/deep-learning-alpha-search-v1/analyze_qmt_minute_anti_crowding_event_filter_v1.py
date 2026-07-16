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

from quantx.core.data.qmt_client import QMTClient, to_qmt_symbol
from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "qmt_minute_anti_crowding_event_filter_v1_summary.json"
CACHE = ROOT / "qmt_minute_cache_v1"
PROVIDER = REPO_ROOT / "data/qlib_data_fixed"

START = "2026-01-01"
END = "2026-07-10"
LOAD_START = "2025-10-01"
PERIOD = "5m"
FETCH_POOL = 60
TOP_K = 10
REBALANCE_INTERVAL = 5
RANDOM_TRIALS = 300
SEED = 20260714

VARIANTS = (
    "daily_event_quality",
    "anti_late_touch",
    "anti_soft_limit",
    "anti_smooth_breakout",
    "anti_low_concentration",
    "anti_gap_safe_proxy",
    "anti_crowding_composite",
    "reverse_minute_event_quality",
)


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


def load_market() -> dict[str, Any]:
    universe = _build_universe(PROVIDER, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(PROVIDER)
    calendar = reader.calendar(LOAD_START, END)
    quotes = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$vwap"], LOAD_START, END)
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


def pct_rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    ranked = series.rank(pct=True, ascending=ascending, method="average")
    return ranked.fillna(0.5).to_numpy(dtype=float)


def daily_candidate_rows(market: dict[str, Any], day: str) -> list[dict[str, Any]]:
    idx = market["date_index"][day]
    arrays = market["arrays"]
    symbols = market["symbols"]
    prev_close = arrays["close"][idx - 1]
    open_ = arrays["open"][idx]
    high = arrays["high"][idx]
    low = arrays["low"][idx]
    close = arrays["close"][idx]
    vwap = arrays["vwap"][idx]
    volume = arrays["volume"][idx]
    vol_hist = arrays["volume"][max(0, idx - 20):idx]
    vol_ma20 = np.nanmean(vol_hist, axis=0)
    valid = np.isfinite(prev_close) & np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (prev_close > 0) & (open_ > 0) & (volume > 0)
    ret1 = np.divide(close, prev_close, out=np.full_like(close, np.nan), where=valid) - 1.0
    high_ret = np.divide(high, prev_close, out=np.full_like(high, np.nan), where=valid) - 1.0
    close_from_high = np.divide(close, high, out=np.full_like(close, np.nan), where=valid & (high > 0)) - 1.0
    close_vs_vwap = np.divide(close, vwap, out=np.full_like(close, np.nan), where=valid & (vwap > 0)) - 1.0
    intraday_body = np.divide(close, open_, out=np.full_like(close, np.nan), where=valid) - 1.0
    upper_shadow = np.divide(high, np.maximum(open_, close), out=np.full_like(high, np.nan), where=valid & (np.maximum(open_, close) > 0)) - 1.0
    vol_ratio = np.divide(volume, vol_ma20, out=np.full_like(volume, np.nan), where=np.isfinite(vol_ma20) & (vol_ma20 > 0))
    r_ret = pct_rank(ret1)
    r_high = pct_rank(high_ret)
    r_close_high = pct_rank(close_from_high)
    r_vwap = pct_rank(close_vs_vwap)
    r_body = pct_rank(intraday_body)
    r_upper_low = pct_rank(upper_shadow, ascending=False)
    r_vol = pct_rank(vol_ratio)
    event_score = 0.20 * r_ret + 0.20 * r_high + 0.20 * r_close_high + 0.15 * r_vwap + 0.10 * r_body + 0.10 * r_upper_low + 0.05 * r_vol
    candidate_idx = np.flatnonzero(valid)
    ordered = sorted(candidate_idx, key=lambda col: (-float(event_score[col]), str(symbols[col])))[:FETCH_POOL]
    rows = []
    for col in ordered:
        rows.append(
            {
                "symbol": symbols[col],
                "daily_event_quality": float(event_score[col]),
                "daily_ret1": float(ret1[col]) if np.isfinite(ret1[col]) else 0.0,
                "daily_high_ret": float(high_ret[col]) if np.isfinite(high_ret[col]) else 0.0,
                "daily_close_from_high": float(close_from_high[col]) if np.isfinite(close_from_high[col]) else 0.0,
                "daily_close_vs_vwap": float(close_vs_vwap[col]) if np.isfinite(close_vs_vwap[col]) else 0.0,
            }
        )
    return rows


def minute_features(frame: pd.DataFrame, preclose: float) -> dict[str, float]:
    if frame.empty or not np.isfinite(preclose) or preclose <= 0:
        return {"minute_available": 0.0}
    frame = frame.sort_values("datetime")
    close = pd.to_numeric(frame.get("close"), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high"), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low"), errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(frame.get("open"), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    amount = pd.to_numeric(frame.get("amount"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(frame)
    if n == 0:
        return {"minute_available": 0.0}
    limit_price = preclose * 1.095
    near_price = preclose * 1.085
    touch = np.isfinite(high) & (high >= limit_price)
    near = np.isfinite(high) & (high >= near_price)
    sealed = np.isfinite(close) & (close >= limit_price)
    break_board = touch & np.isfinite(close) & (close < near_price)
    touch_idx = np.flatnonzero(touch)
    re_seal = np.zeros(n, dtype=bool)
    if touch_idx.size:
        seen_break = False
        for i in range(int(touch_idx[0]), n):
            if break_board[i]:
                seen_break = True
            if seen_break and sealed[i]:
                re_seal[i] = True
    total_vol = float(np.nansum(volume))
    total_amt = float(np.nansum(amount))
    vwap = total_amt / total_vol / 100.0 if total_vol > 0 and total_amt > 0 else float(np.nanmean(close))
    last_close = float(close[-1]) if np.isfinite(close[-1]) else np.nan
    day_high = float(np.nanmax(high)) if np.isfinite(high).any() else np.nan
    day_low = float(np.nanmin(low)) if np.isfinite(low).any() else np.nan
    tail_n = min(6, n)
    first_touch = int(touch_idx[0]) if touch_idx.size else n
    max_break_drawdown = 0.0
    if touch_idx.size and np.isfinite(low[first_touch:]).any():
        max_break_drawdown = float(np.nanmin(low[first_touch:]) / limit_price - 1.0)
    top_count = max(1, int(math.ceil(n * 0.2)))
    return {
        "minute_available": 1.0,
        "limit_touch_flag": float(touch.any()),
        "first_touch_frac": float(first_touch / max(1, n - 1)) if touch.any() else 1.0,
        "touch_count": float(touch.sum()),
        "near_limit_bar_ratio": float(near.mean()),
        "seal_to_close_ratio": float(sealed[-tail_n:].mean()),
        "close_sealed_flag": float(bool(sealed[-1])),
        "break_board_count": float(break_board.sum()),
        "re_seal_count": float(re_seal.sum()),
        "afternoon_reseal_flag": float(bool(re_seal[int(n * 0.5):].any())),
        "max_break_drawdown": max_break_drawdown,
        "tail_strength_30m": float(last_close / close[-tail_n] - 1.0) if tail_n > 1 and np.isfinite(last_close) and np.isfinite(close[-tail_n]) and close[-tail_n] > 0 else 0.0,
        "tail_volume_share_30m": float(np.nansum(volume[-tail_n:]) / total_vol) if total_vol > 0 else 0.0,
        "intraday_vwap_support": float(last_close / vwap - 1.0) if np.isfinite(last_close) and np.isfinite(vwap) and vwap > 0 else 0.0,
        "high_to_close_fade": float(last_close / day_high - 1.0) if np.isfinite(last_close) and np.isfinite(day_high) and day_high > 0 else 0.0,
        "open_to_high_speed": float(day_high / open_[0] - 1.0) if np.isfinite(day_high) and np.isfinite(open_[0]) and open_[0] > 0 else 0.0,
        "volume_concentration_top20pct": float(np.nansum(np.sort(volume)[-top_count:]) / total_vol) if total_vol > 0 else 0.0,
        "close_return": float(last_close / preclose - 1.0) if np.isfinite(last_close) else 0.0,
        "low_return": float(day_low / preclose - 1.0) if np.isfinite(day_low) else 0.0,
        "high_return": float(day_high / preclose - 1.0) if np.isfinite(day_high) else 0.0,
    }


def rank_map(rows: list[dict[str, Any]], field: str, ascending: bool = True) -> dict[str, float]:
    values = pd.Series({row["symbol"]: float(row.get(field, np.nan)) for row in rows}).replace([np.inf, -np.inf], np.nan)
    ranks = values.rank(pct=True, ascending=ascending, method="average").fillna(0.5)
    return {sym: float(value) for sym, value in ranks.items()}


def add_minute_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = {
        "early_touch": rank_map(rows, "first_touch_frac", ascending=False),
        "late_touch": rank_map(rows, "first_touch_frac"),
        "low_touch_count": rank_map(rows, "touch_count", ascending=False),
        "low_near_ratio": rank_map(rows, "near_limit_bar_ratio", ascending=False),
        "low_seal_close": rank_map(rows, "seal_to_close_ratio", ascending=False),
        "not_close_sealed": rank_map(rows, "close_sealed_flag", ascending=False),
        "near_ratio": rank_map(rows, "near_limit_bar_ratio"),
        "seal_close": rank_map(rows, "seal_to_close_ratio"),
        "break_low": rank_map(rows, "break_board_count", ascending=False),
        "re_seal": rank_map(rows, "re_seal_count"),
        "break_drawdown": rank_map(rows, "max_break_drawdown"),
        "tail_strength": rank_map(rows, "tail_strength_30m"),
        "tail_volume": rank_map(rows, "tail_volume_share_30m"),
        "vwap": rank_map(rows, "intraday_vwap_support"),
        "fade_low": rank_map(rows, "high_to_close_fade"),
        "vol_conc": rank_map(rows, "volume_concentration_top20pct"),
        "low_vol_conc": rank_map(rows, "volume_concentration_top20pct", ascending=False),
        "low_open_to_high_speed": rank_map(rows, "open_to_high_speed", ascending=False),
        "moderate_close_return": rank_map(rows, "close_return", ascending=False),
    }
    out = []
    for row in rows:
        sym = row["symbol"]
        daily = float(row["daily_event_quality"])
        seal = 0.20 * ranks["early_touch"][sym] + 0.25 * ranks["near_ratio"][sym] + 0.25 * ranks["seal_close"][sym] + 0.15 * ranks["vwap"][sym] + 0.15 * ranks["tail_strength"][sym]
        reseal = 0.25 * ranks["re_seal"][sym] + 0.25 * ranks["break_drawdown"][sym] + 0.20 * ranks["tail_strength"][sym] + 0.15 * ranks["vwap"][sym] + 0.15 * ranks["tail_volume"][sym]
        tail = 0.35 * ranks["tail_strength"][sym] + 0.25 * ranks["tail_volume"][sym] + 0.20 * ranks["vwap"][sym] + 0.20 * ranks["fade_low"][sym]
        break_avoid = 0.35 * ranks["break_low"][sym] + 0.25 * ranks["break_drawdown"][sym] + 0.20 * ranks["fade_low"][sym] + 0.20 * ranks["vwap"][sym]
        minute_event = 0.35 * seal + 0.25 * reseal + 0.20 * tail + 0.20 * break_avoid
        anti_late_touch = 0.25 * ranks["late_touch"][sym] + 0.20 * ranks["low_near_ratio"][sym] + 0.20 * ranks["low_touch_count"][sym] + 0.20 * ranks["vwap"][sym] + 0.15 * ranks["tail_strength"][sym]
        anti_soft_limit = 0.25 * ranks["low_seal_close"][sym] + 0.20 * ranks["not_close_sealed"][sym] + 0.20 * ranks["low_near_ratio"][sym] + 0.20 * ranks["fade_low"][sym] + 0.15 * ranks["vwap"][sym]
        anti_smooth_breakout = 0.25 * ranks["vwap"][sym] + 0.20 * ranks["tail_strength"][sym] + 0.20 * ranks["low_open_to_high_speed"][sym] + 0.20 * ranks["low_vol_conc"][sym] + 0.15 * ranks["fade_low"][sym]
        anti_low_conc = 0.35 * ranks["low_vol_conc"][sym] + 0.25 * ranks["low_open_to_high_speed"][sym] + 0.20 * ranks["vwap"][sym] + 0.20 * ranks["tail_strength"][sym]
        anti_gap_safe = 0.30 * ranks["moderate_close_return"][sym] + 0.25 * ranks["low_near_ratio"][sym] + 0.20 * ranks["low_open_to_high_speed"][sym] + 0.15 * ranks["vwap"][sym] + 0.10 * ranks["tail_strength"][sym]
        anti_composite = 0.25 * anti_late_touch + 0.25 * anti_soft_limit + 0.20 * anti_smooth_breakout + 0.15 * anti_low_conc + 0.15 * anti_gap_safe
        enriched = dict(row)
        enriched.update(
            {
                "score_daily_event_quality": daily,
                "score_anti_late_touch": 0.50 * daily + 0.50 * anti_late_touch,
                "score_anti_soft_limit": 0.50 * daily + 0.50 * anti_soft_limit,
                "score_anti_smooth_breakout": 0.50 * daily + 0.50 * anti_smooth_breakout,
                "score_anti_low_concentration": 0.50 * daily + 0.50 * anti_low_conc,
                "score_anti_gap_safe_proxy": 0.50 * daily + 0.50 * anti_gap_safe,
                "score_anti_crowding_composite": 0.45 * daily + 0.55 * anti_composite,
                "score_reverse_minute_event_quality": 0.45 * daily + 0.55 * (1.0 - minute_event),
            }
        )
        out.append(enriched)
    return out


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


def replay(selections: dict[str, list[str]], market: dict[str, Any]) -> dict[str, Any]:
    dates = market["dates"]
    arrays = market["arrays"]
    symbol_index = market["symbol_index"]
    active = sorted(selections)
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


def main() -> None:
    started = time.perf_counter()
    market = load_market()
    sessions = [day for day in market["dates"] if START <= day <= END]
    due5 = sessions[::REBALANCE_INTERVAL]
    candidates_by_session: dict[str, list[dict[str, Any]]] = {}
    fetch = {"requested": 0, "available": 0, "empty": 0, "errors": 0}
    with QMTClient(max_retries=1, fill_data=False) as client:
        for session in due5:
            idx = market["date_index"][session]
            rows = daily_candidate_rows(market, session)
            enriched = []
            for row in rows:
                sym = row["symbol"]
                col = market["symbol_index"][sym]
                preclose = float(market["arrays"]["close"][idx - 1, col])
                try:
                    fetch["requested"] += 1
                    minute = load_or_fetch_minute(client, sym, session, PERIOD)
                    day_frame = minute[minute["date"] == session] if not minute.empty and "date" in minute.columns else minute
                    feats = minute_features(day_frame, preclose)
                    if feats.get("minute_available", 0.0) > 0:
                        fetch["available"] += 1
                    else:
                        fetch["empty"] += 1
                except Exception:
                    fetch["errors"] += 1
                    feats = {"minute_available": 0.0}
                merged = dict(row)
                merged.update(feats)
                enriched.append(merged)
            candidates_by_session[session] = add_minute_scores(enriched)
    selections: dict[str, dict[str, list[str]]] = {variant: {} for variant in VARIANTS}
    pools: dict[str, list[str]] = {}
    for session, rows in candidates_by_session.items():
        pools[session] = [row["symbol"] for row in rows]
        for variant in VARIANTS:
            key = f"score_{variant}"
            ranked = sorted(rows, key=lambda row: (-float(row.get(key, 0.0)), str(row["symbol"])))[:TOP_K]
            selections[variant][session] = [row["symbol"] for row in ranked]
    results = {variant: replay(sel, market)["summary"] for variant, sel in selections.items()}
    rng = np.random.default_rng(SEED)
    random_values = [replay(random_selections(pools, rng), market)["summary"]["final_multiple"] for _ in range(RANDOM_TRIALS)]
    random_summary = percentile_summary(random_values)
    best_anti = max((variant for variant in VARIANTS if variant != "daily_event_quality"), key=lambda variant: results[variant]["final_multiple"])
    gates = {
        variant: {
            "beats_random_p95": results[variant]["final_multiple"] > random_summary["p95"],
            "beats_random_p99": results[variant]["final_multiple"] > random_summary["p99"],
            "beats_daily_event_quality": results[variant]["final_multiple"] > results["daily_event_quality"]["final_multiple"],
            "remove_best_3_positive": results[variant]["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0,
        }
        for variant in VARIANTS
    }
    verdict = "anti_crowding_soil_not_found"
    if gates[best_anti]["beats_random_p95"] and gates[best_anti]["beats_daily_event_quality"] and gates[best_anti]["remove_best_3_positive"]:
        verdict = "anti_crowding_soil_candidate_needs_dev_validation"
    out = {
        "experiment": "qmt_minute_anti_crowding_event_filter_v1",
        "method": "qmt_5m_daily_event_pool_anti_crowding_late_touch_soft_limit_low_concentration_2026_soil_scan",
        "params": {"start": START, "end": END, "load_start": LOAD_START, "period": PERIOD, "fetch_pool": FETCH_POOL, "top_k": TOP_K, "rebalance_interval": REBALANCE_INTERVAL, "random_trials": RANDOM_TRIALS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__))},
        "minute_fetch": fetch,
        "sessions": len(candidates_by_session),
        "results": results,
        "random_pool_top10_strict": random_summary,
        "gates": gates,
        "best_anti_variant": best_anti,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_anti_variant": best_anti, "minute_fetch": fetch, "results": results, "random": random_summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
