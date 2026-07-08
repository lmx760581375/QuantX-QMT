"""Backtest a daily approximation of the JoinQuant Wufu ETF rotation strategy.

The original strategy trades around 13:10 and uses JoinQuant-specific ETF NAV,
minute volume, and dynamic pool helpers. This tool keeps the reproducible core:
large ETF pool, three market regimes, weighted-regression momentum, regime
filters, Top1 rotation, defensive ETF fallback, and adjusted-correlation guard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quantx.core.analysis.reporting import compute_metrics, match_closed_positions


DEFAULT_START = "2016-01-04"
DEFAULT_END = "2026-07-07"
DEFAULT_INIT_CASH = 1_000_000.0
DEFAULT_SOURCE_STRATEGY = "/home/users/mingxiao.li/.codex/attachments/7ea8b77b-7730-4834-8651-76aad502998b/pasted-text.txt"
DEFAULT_ETF_META_PATH = "data/meta/snapshots/etf_master.csv"

GLOBAL_POOL = [
    "SH518880", "SH501018", "SZ161226", "SZ159985", "SZ159980", "SH513310", "SZ159518",
    "SZ159509", "SH513100", "SH513520", "SH513500", "SZ159502", "SH513400", "SH513030",
    "SH513290", "SH520830", "SZ159529", "SZ164824", "SH513080", "SH513730", "SH511380",
    "SH511220", "SH510050", "SH563300", "SZ159928", "SH510300", "SH510500", "SH512100",
    "SZ159915", "SH513180", "SZ159920",
]

CHINA_POOL = [
    "SH513090", "SH513120", "SH513180", "SH513330", "SH513750", "SZ159892", "SH513190",
    "SZ159605", "SH513630", "SZ159323", "SH510900", "SH513920", "SH513970", "SH511380",
    "SH512050", "SH510500", "SZ159915", "SH510300", "SH512100", "SZ159949", "SH588080",
    "SZ159967", "SH588220", "SH563300", "SH510760", "SH588200", "SH515880", "SZ159981",
    "SH512880", "SH513350", "SZ159326", "SZ159516", "SZ159206", "SH512480", "SZ159363",
    "SZ159870", "SH512400", "SZ159755", "SH588170", "SZ159992", "SZ159995", "SH512890",
    "SH515220", "SZ159566", "SZ159819", "SH512800", "SH512690", "SH515050", "SH562500",
    "SH512170", "SH517520", "SZ159869", "SH512070", "SZ159611", "SH562800", "SH515120",
    "SH512010", "SH510880", "SH515790", "SH515980", "SH512660", "SZ159928", "SH512710",
    "SH560860", "SH515030", "SZ159766", "SZ159218", "SZ159852", "SH516160", "SH516150",
    "SZ159227", "SZ159583", "SH588790", "SZ159865", "SH512980", "SZ159851", "SH561360",
    "SH561980", "SH562590", "SH512200", "SZ159732", "SZ159667", "SH516510", "SZ159840",
    "SZ159998", "SZ159825", "SH512670", "SZ159883", "SH515210", "SH515400", "SZ159256",
    "SH561330", "SH515170", "SZ159638", "SH516520", "SH513360", "SH516190",
]

THEME_ETF_POOL = [
    "SH513090", "SH513120", "SH513750", "SZ159892", "SH513190", "SZ159323", "SH513970",
    "SH588200", "SH515880", "SZ159981", "SH512880", "SH513350", "SZ159326", "SZ159516",
    "SZ159206", "SH512480", "SZ159363", "SZ159870", "SH512400", "SZ159755", "SH588170",
    "SZ159992", "SZ159995", "SH515220", "SZ159566", "SZ159819", "SH512800", "SH512690",
    "SH515050", "SH562500", "SH512170", "SH517520", "SZ159869", "SH512070", "SZ159611",
    "SH562800", "SH515120", "SH512010", "SH515790", "SH515980", "SH512660", "SH512710",
    "SH560860", "SH515030", "SZ159766", "SZ159218", "SZ159852", "SH516160", "SH516150",
    "SZ159227", "SZ159583", "SH588790", "SZ159865", "SH512980", "SZ159851", "SH561360",
    "SH561980", "SH562590", "SH512200", "SZ159732", "SZ159667", "SH516510", "SZ159840",
    "SZ159998", "SZ159825", "SH512670", "SZ159883", "SH515210", "SH515400", "SZ159256",
    "SH561330", "SH515170", "SZ159638", "SH516520", "SH513360", "SH516190",
]

REGIME_PROXIES = {
    "沪深300": ["SH000300", "SH510300"],
    "深证综指": ["SZ399101", "SH510500"],
    "创业板": ["SZ399006", "SZ159915"],
    "中证A500": ["SH000510", "SH510300"],
    "中证1000": ["SH000852", "SH512100"],
    "国证2000": ["SZ399303", "SH563300", "SH512100"],
}


@dataclass
class Config:
    start: str
    end: str
    init_cash: float
    raw_dir: Path
    etf_meta_path: Path
    index_raw_dir: Path
    runs_dir: Path
    source_strategy: Path
    sync_missing: bool
    source: str
    pause_seconds: float
    commission: float
    slippage: float
    min_cost: float
    min_money: float
    lookback_days: int
    short_lookback: int
    min_score: float
    max_score: float
    score_threshold_ratio: float
    r2_threshold: float
    normal_r2_threshold: float
    ma_lookback: int
    ma_threshold: float
    volume_lookback: int
    volume_threshold: float
    intraday_volume_fraction: float
    loss_threshold: float
    corr_lookback: int
    corr_hold_threshold: float
    corr_hold_momentum_max: float
    corr_overlay_threshold: float
    corr_overlay_momentum_max: float
    pick_target_max_padj: float
    regime_confirm_days: int
    weak_below_ma20_min: int
    normal_above_ma10_min: int
    defensive_etf: str
    top_n: int
    hold_rank_top: int
    switch_score_premium: float
    hold_min_momentum_score: float
    choppy_confirm_days: int
    choppy_min_filtered_count: int
    execution_mode: str
    dynamic_theme_top_n: int
    dynamic_theme_min_score: float
    dynamic_theme_liquidity_lookback: int
    dynamic_theme_min_avg_amount: float
    dynamic_theme_regimes: list[str]
    dynamic_theme_score_premium: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item).upper() for item in items if str(item).strip()))


def _extract_symbols_from_source(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    symbols: list[str] = []
    for code, exch in re.findall(r"'([0-9]{6})\.(XSHG|XSHE)'", text):
        if code.startswith(("000", "399")):
            continue
        symbols.append(("SH" if exch == "XSHG" else "SZ") + code)
    return _dedupe(symbols)


def _all_pool_symbols(source_strategy: Path) -> list[str]:
    extracted = _extract_symbols_from_source(source_strategy)
    fallback = GLOBAL_POOL + CHINA_POOL + THEME_ETF_POOL + ["SH511880"]
    return _dedupe((extracted or fallback) + THEME_ETF_POOL)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _load_theme_symbols_from_meta(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not path.exists():
        return THEME_ETF_POOL[:], {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "symbol" not in df.columns:
        return THEME_ETF_POOL[:], {}
    if "is_dynamic_theme_candidate" in df.columns:
        df = df[df["is_dynamic_theme_candidate"].map(_truthy)]
    elif "category" in df.columns:
        df = df[df["category"] == "sector_theme"]
    symbols = _dedupe(df["symbol"].tolist())
    meta = {
        str(row["symbol"]).upper(): {str(k): row[k] for k in df.columns}
        for _, row in df.iterrows()
    }
    return symbols or THEME_ETF_POOL[:], meta


def _load_csv(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {"pct_chg": "pctChg", "is_st": "isST", "turnover": "turn"}
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        raise ValueError(f"missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df.get("volume", 0) * 100.0
    return df.dropna(subset=["open", "high", "low", "close"])


def _sync_missing_etfs(symbols: list[str], cfg: Config) -> dict[str, Any]:
    missing = [symbol for symbol in symbols if not (cfg.raw_dir / f"{symbol}.csv").exists()]
    if not missing or not cfg.sync_missing:
        return {"requested": len(symbols), "missing_before": missing, "sync_ran": False}
    cmd = [
        "conda", "run", "-n", "test", "python", "-m", "quantx.tools.sync_etf_data",
        "--source", cfg.source,
        "--start", cfg.start,
        "--end", cfg.end,
        "--provider-uri", "data/qlib_etf_data_wufu_tmp",
        "--raw-dir", str(cfg.raw_dir),
        "--pause-seconds", str(cfg.pause_seconds),
        "--symbols", *missing,
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout[proc.stdout.find("{"):])
    except Exception:
        payload = {"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    payload["returncode"] = proc.returncode
    payload["missing_before"] = missing
    return payload


def _load_bars(symbols: list[str], raw_dir: Path, start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    bars: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        try:
            df = _load_csv(path, start, end)
        except Exception:
            missing.append(symbol)
            continue
        if len(df) >= 30:
            bars[symbol] = df
        else:
            missing.append(symbol)
    return bars, missing


def _close_table(bars: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    series = {symbol: df.set_index("date")[field] for symbol, df in bars.items() if field in df.columns}
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def _choose_regime_sources(etf_close: pd.DataFrame, cfg: Config) -> dict[str, str]:
    index_bars: dict[str, pd.DataFrame] = {}
    for candidates in REGIME_PROXIES.values():
        for symbol in candidates:
            path = cfg.index_raw_dir / f"{symbol}.csv"
            if path.exists() and symbol not in index_bars:
                try:
                    df = _load_csv(path, cfg.start, cfg.end)
                    if len(df) >= 30:
                        index_bars[symbol] = df
                except Exception:
                    pass
    index_close = _close_table(index_bars)
    sources: dict[str, str] = {}
    for name, candidates in REGIME_PROXIES.items():
        for symbol in candidates:
            if symbol in index_close.columns and index_close[symbol].notna().sum() >= 30:
                sources[name] = symbol
                break
            if symbol in etf_close.columns and etf_close[symbol].notna().sum() >= 30:
                sources[name] = symbol
                break
    return sources


def _merged_close_for_regime(etf_close: pd.DataFrame, cfg: Config, sources: dict[str, str]) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for name, symbol in sources.items():
        if symbol in etf_close.columns:
            cols[name] = etf_close[symbol]
            continue
        path = cfg.index_raw_dir / f"{symbol}.csv"
        if path.exists():
            df = _load_csv(path, cfg.start, cfg.end)
            cols[name] = df.set_index("date")["close"]
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def calculate_momentum_score(prices: np.ndarray, lookback_days: int) -> tuple[float | None, float | None, float | None]:
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < lookback_days + 1:
        return None, None, None
    y = np.log(prices[-(lookback_days + 1):])
    x = np.arange(len(y), dtype=float)
    weights = np.linspace(1.0, 2.0, len(y))
    w = weights ** 2
    w_sum = float(np.sum(w))
    x_bar = float(np.sum(w * x) / w_sum)
    y_bar = float(np.sum(w * y) / w_sum)
    dx = x - x_bar
    dy = y - y_bar
    variance_x = float(np.sum(w * dx ** 2))
    if variance_x == 0:
        return 0.0, 0.0, 0.0
    slope = float(np.sum(w * dx * dy) / variance_x)
    intercept = y_bar - slope * x_bar
    annualized = math.exp(slope * 250.0) - 1.0
    y_pred = slope * x + intercept
    ss_res = float(np.sum(weights * (y - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return annualized * r2, annualized, r2


def laplace_filter(prices: np.ndarray, s: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-float(s))
    out = np.zeros(len(prices), dtype=float)
    out[0] = prices[0]
    for idx in range(1, len(prices)):
        out[idx] = alpha * prices[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def gaussian_filter_last_two(prices: np.ndarray, sigma: float = 1.2) -> tuple[float, float]:
    n = len(prices)
    if n < 2:
        return 0.0, 0.0
    idx = np.arange(n)
    weights = np.exp(-((idx + 1) ** 2) / (2.0 * sigma ** 2))[::-1]
    weights /= np.sum(weights)
    g1 = float(np.sum(prices * weights))
    prev = prices[:-1]
    idx2 = np.arange(n - 1)
    weights2 = np.exp(-((idx2 + 1) ** 2) / (2.0 * sigma ** 2))[::-1]
    weights2 /= np.sum(weights2)
    g2 = float(np.sum(prev * weights2))
    return g1, g2


def adjusted_corr(price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.dropna(thresh=max(2, int(len(price_df) * 0.7)), axis=1).ffill().dropna()
    if price_df.empty:
        return pd.DataFrame()
    if price_df.shape[1] == 1:
        code = price_df.columns[0]
        return pd.DataFrame([[1.0]], index=[code], columns=[code])
    returns = np.log(price_df / price_df.shift(1)).dropna()
    if returns.empty:
        return pd.DataFrame()
    base_corr = returns.corr()
    cum_returns = (price_df / price_df.iloc[0]) - 1.0
    vols = returns.std() * np.sqrt(252.0)
    curve_diff = np.mean(np.abs(cum_returns.values[:, :, None] - cum_returns.values[:, None, :]), axis=0)
    ret_factor = np.exp(-curve_diff)
    vol_arr = vols.values
    vol_min = np.minimum(vol_arr[:, None], vol_arr[None, :])
    vol_max = np.maximum(vol_arr[:, None], vol_arr[None, :])
    vol_max[vol_max == 0] = 1e-9
    adj = base_corr.values * ret_factor * (vol_min / vol_max)
    return pd.DataFrame(adj, index=price_df.columns, columns=price_df.columns)


def resolve_regime(date: pd.Timestamp, regime_close: pd.DataFrame, state: dict[str, Any], cfg: Config) -> str:
    history = regime_close.loc[:date].tail(max(cfg.ma_lookback, 20) + 1)
    below = 0
    above = 0
    ok = 0
    for col in history.columns:
        s = history[col].dropna()
        if len(s) < 20:
            continue
        cur = float(s.iloc[-1])
        if cur < float(s.tail(20).mean()):
            below += 1
        if cur > float(s.tail(cfg.ma_lookback).mean()):
            above += 1
        ok += 1
    if ok == 0:
        raw = "震荡期"
    else:
        weak_min = min(cfg.weak_below_ma20_min, max(1, math.ceil(ok * cfg.weak_below_ma20_min / 6.0)))
        normal_min = min(cfg.normal_above_ma10_min, max(1, math.ceil(ok * cfg.normal_above_ma10_min / 6.0)))
        if below >= weak_min:
            raw = "走弱期"
        elif above >= normal_min:
            raw = "正常期"
        else:
            raw = "震荡期"

    current = state.get("regime")
    if current is None:
        state.update({"regime": raw, "pending": None, "streak": 0, "last_change": date.strftime("%Y-%m-%d")})
        return raw
    if raw == current:
        state.update({"pending": None, "streak": 0})
        return current
    if cfg.regime_confirm_days <= 1:
        state.update({"regime": raw, "pending": None, "streak": 0, "last_change": date.strftime("%Y-%m-%d")})
        return raw
    if state.get("pending") == raw:
        state["streak"] = int(state.get("streak", 0)) + 1
    else:
        state["pending"] = raw
        state["streak"] = 1
    if int(state["streak"]) >= cfg.regime_confirm_days:
        state.update({"regime": raw, "pending": None, "streak": 0, "last_change": date.strftime("%Y-%m-%d")})
        return raw
    return current


def _metric_for_symbol(symbol: str, date: pd.Timestamp, df: pd.DataFrame, cfg: Config, regime: str) -> dict[str, Any] | None:
    hist = df[df["date"] <= date].tail(max(cfg.lookback_days, cfg.short_lookback, cfg.volume_lookback, 20) + 25)
    if len(hist) < max(cfg.lookback_days, cfg.short_lookback) + 1:
        return None
    prices = hist["close"].to_numpy(dtype=float)
    volumes = hist["volume"].to_numpy(dtype=float) if "volume" in hist else np.zeros(len(hist))
    if not np.all(np.isfinite(prices[-max(cfg.lookback_days, cfg.short_lookback):])):
        return None
    score, annualized, r2 = calculate_momentum_score(prices, cfg.lookback_days)
    short_score, _, _ = calculate_momentum_score(prices, cfg.short_lookback)
    if score is None or short_score is None or r2 is None:
        return None
    current = float(prices[-1])
    min_score = cfg.min_score
    max_score = cfg.max_score
    short_min = 0.0
    short_max = 6.0
    if regime == "震荡期":
        short_min = max(short_min, 0.0)
        short_max = min(short_max, 6.0)
    passed_momentum = min_score <= score <= max_score
    passed_short = short_min <= short_score <= short_max
    passed_dual = score > 0 and short_score > 0
    r2_threshold = cfg.normal_r2_threshold if regime == "正常期" else cfg.r2_threshold
    passed_r2 = r2 > r2_threshold

    ma = float(np.mean(prices[-cfg.ma_lookback:])) if len(prices) >= cfg.ma_lookback else math.nan
    passed_ma = bool(np.isfinite(ma) and current > ma * cfg.ma_threshold)
    volume_ratio = None
    if len(volumes) >= cfg.volume_lookback + 1 and np.nanmean(volumes[-cfg.volume_lookback - 1:-1]) > 0:
        intraday_volume = float(volumes[-1]) * cfg.intraday_volume_fraction
        volume_ratio = intraday_volume / float(np.nanmean(volumes[-cfg.volume_lookback - 1:-1]))
    passed_volume = volume_ratio is not None and volume_ratio < cfg.volume_threshold

    day_ratios: list[float] = []
    passed_loss = True
    if len(prices) >= 4:
        day_ratios = [float(prices[-1] / prices[-2]), float(prices[-2] / prices[-3]), float(prices[-3] / prices[-4])]
        passed_loss = min(day_ratios) >= cfg.loss_threshold

    laplace_s = 0.06 if regime == "正常期" else (0.12 if regime == "走弱期" else 0.05)
    laplace_min_slope = 0.0022 if regime == "正常期" else (0.001 if regime == "走弱期" else 0.002)
    passed_laplace = True
    laplace_slope = 0.0
    if len(prices) >= 10:
        lv = laplace_filter(prices, laplace_s)
        laplace_slope = float(lv[-1] - lv[-2])
        passed_laplace = current > float(lv[-1]) and laplace_slope > laplace_min_slope
    g1, g2 = gaussian_filter_last_two(prices, 1.2)
    gaussian_slope = (g1 - g2) / g2 if abs(g2) > 1e-12 else 0.0
    passed_gaussian = current > g1 and gaussian_slope > 0.0013

    filters = [passed_momentum, passed_short, passed_dual]
    if regime != "走弱期":
        filters.extend([passed_r2, passed_volume, passed_loss])
    else:
        filters.extend([passed_ma, passed_volume, passed_loss, passed_laplace])
    if regime == "正常期":
        filters.append(passed_laplace)
    if regime == "震荡期":
        filters.append(passed_gaussian)

    return {
        "etf": symbol,
        "momentum_score": float(score),
        "momentum_rank_score": float(score),
        "short_momentum_score": float(short_score),
        "annualized_returns": float(annualized or 0.0),
        "r_squared": float(r2),
        "current_price": current,
        "volume_ratio": None if volume_ratio is None else float(volume_ratio),
        "day_ratios": day_ratios,
        "ma_value": ma,
        "laplace_slope": laplace_slope,
        "gaussian_slope": float(gaussian_slope),
        "passed_all_filters": bool(all(filters)),
        "passed_momentum": bool(passed_momentum),
        "passed_r2": bool(passed_r2),
        "passed_ma": bool(passed_ma),
        "passed_volume": bool(passed_volume),
        "passed_loss": bool(passed_loss),
        "passed_laplace": bool(passed_laplace),
        "passed_gaussian": bool(passed_gaussian),
    }


def _dynamic_theme_pool(
    date: pd.Timestamp,
    bars: dict[str, pd.DataFrame],
    cfg: Config,
    theme_symbols: list[str] | None = None,
    theme_meta: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if cfg.dynamic_theme_top_n <= 0:
        return [], []
    rows: list[dict[str, Any]] = []
    theme_symbols = theme_symbols or THEME_ETF_POOL
    theme_meta = theme_meta or {}
    for symbol in theme_symbols:
        df = bars.get(symbol)
        if df is None:
            continue
        hist = df[df["date"] <= date].tail(max(cfg.lookback_days, cfg.dynamic_theme_liquidity_lookback) + 1)
        if len(hist) < max(cfg.lookback_days, cfg.dynamic_theme_liquidity_lookback) + 1:
            continue
        prices = hist["close"].to_numpy(dtype=float)
        score, annualized, r2 = calculate_momentum_score(prices, cfg.lookback_days)
        if score is None or score < cfg.dynamic_theme_min_score:
            continue
        amount = hist["amount"] if "amount" in hist.columns else hist["close"] * hist["volume"] * 100.0
        avg_amount = float(pd.to_numeric(amount.tail(cfg.dynamic_theme_liquidity_lookback), errors="coerce").mean())
        if not math.isfinite(avg_amount) or avg_amount < cfg.dynamic_theme_min_avg_amount:
            continue
        meta = theme_meta.get(symbol, {})
        rows.append({
            "etf": symbol,
            "name": meta.get("name"),
            "category": meta.get("category"),
            "theme_group": meta.get("theme_group"),
            "momentum_score": float(score),
            "annualized_returns": float(annualized or 0.0),
            "r_squared": float(r2 or 0.0),
            "avg_amount": avg_amount,
        })
    rows.sort(key=lambda item: (item["momentum_score"], item["avg_amount"]), reverse=True)
    selected = rows[: cfg.dynamic_theme_top_n]
    return [str(item["etf"]) for item in selected], selected


def _parse_regime_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["走弱期", "震荡期", "正常期"]
    if isinstance(value, str):
        parts = [item.strip() for item in re.split(r"[,，]", value) if item.strip()]
    else:
        parts = [str(item).strip() for item in value if str(item).strip()]
    if not parts or any(item.lower() == "all" for item in parts):
        return ["走弱期", "震荡期", "正常期"]
    return parts


def _apply_dynamic_theme_gate(
    filtered: list[dict[str, Any]],
    regime: str,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dynamic_items = [item for item in filtered if item.get("dynamic_theme_pool")]
    if not dynamic_items:
        return filtered, {"dynamic_theme_gate_blocked": 0}

    allowed_regimes = set(_parse_regime_list(getattr(cfg, "dynamic_theme_regimes", None)))
    score_premium = float(getattr(cfg, "dynamic_theme_score_premium", 0.0) or 0.0)
    base_items = [item for item in filtered if not item.get("dynamic_theme_pool")]
    base_best_score = max((float(item.get("momentum_score", float("-inf"))) for item in base_items), default=None)

    gated: list[dict[str, Any]] = []
    blocked = 0
    blocked_reasons: dict[str, int] = {}
    for item in filtered:
        if not item.get("dynamic_theme_pool"):
            gated.append(item)
            continue

        reasons: list[str] = []
        if regime not in allowed_regimes:
            reasons.append("regime_not_allowed")
        if base_best_score is not None and math.isfinite(base_best_score) and score_premium > 0:
            item_score = float(item.get("momentum_score", float("-inf")))
            required = base_best_score * (1.0 + score_premium)
            if item_score < required:
                reasons.append("score_premium")
                item["dynamic_theme_required_score"] = float(required)

        if reasons:
            blocked += 1
            item["dynamic_theme_gate_blocked"] = True
            item["dynamic_theme_gate_reasons"] = reasons
            for reason in reasons:
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
            continue
        gated.append(item)

    return gated, {
        "dynamic_theme_gate_allowed_regimes": sorted(allowed_regimes),
        "dynamic_theme_gate_score_premium": score_premium,
        "dynamic_theme_gate_blocked": blocked,
        "dynamic_theme_gate_blocked_reasons": blocked_reasons,
        "dynamic_theme_gate_base_best_score": base_best_score,
    }


def _select_target(
    date: pd.Timestamp,
    regime: str,
    bars: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    current_holding: str | None,
    cfg: Config,
    theme_symbols: list[str] | None = None,
    theme_meta: dict[str, dict[str, Any]] | None = None,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    pool = GLOBAL_POOL if regime == "走弱期" else _dedupe(GLOBAL_POOL + CHINA_POOL)
    if regime == "走弱期":
        # The pasted best profile disables the domestic dual pool in weak regime.
        pool = _dedupe(symbol for symbol in pool if symbol in bars)
    else:
        pool = _dedupe(symbol for symbol in pool if symbol in bars)
    dynamic_pool, dynamic_metrics = _dynamic_theme_pool(date, bars, cfg, theme_symbols, theme_meta)
    if dynamic_pool:
        pool = _dedupe([*pool, *dynamic_pool])
    metrics = []
    for symbol in pool:
        m = _metric_for_symbol(symbol, date, bars[symbol], cfg, regime)
        if m:
            m["dynamic_theme_pool"] = symbol in dynamic_pool
            metrics.append(m)
    metrics.sort(key=lambda item: item.get("momentum_rank_score", float("-inf")), reverse=True)
    filtered = [m for m in metrics if m.get("passed_all_filters")]
    filtered.sort(key=lambda item: item.get("momentum_score", float("-inf")), reverse=True)
    filtered_before_gate = len(filtered)
    filtered, gate_diag = _apply_dynamic_theme_gate(filtered, regime, cfg)
    top = filtered[:10]
    diag: dict[str, Any] = {
        "pool_size": len(pool),
        "metrics_count": len(metrics),
        "filtered_count": len(filtered),
        "filtered_before_dynamic_gate": filtered_before_gate,
        "dynamic_theme_count": len(dynamic_pool),
        "dynamic_theme_top": dynamic_metrics[:5],
        **gate_diag,
    }
    if regime == "震荡期" and cfg.choppy_min_filtered_count > 0 and len(filtered) < cfg.choppy_min_filtered_count:
        target = current_holding or (cfg.defensive_etf if cfg.defensive_etf in bars else None)
        return target, metrics[:20], {**diag, "mode": "choppy_defensive_low_breadth"}
    if not top:
        return cfg.defensive_etf if cfg.defensive_etf in bars else None, metrics[:20], {**diag, "mode": "defensive"}
    ref = top[min(cfg.top_n, len(top)) - 1]["momentum_rank_score"]
    ratio = 1.0 if regime == "走弱期" else cfg.score_threshold_ratio
    candidate_pool = [m for m in top if m["momentum_rank_score"] >= ref * ratio]
    if not candidate_pool:
        return cfg.defensive_etf if cfg.defensive_etf in bars else None, metrics[:20], {**diag, "mode": "defensive_empty_candidate"}

    target = candidate_pool[0]["etf"]
    skipped_high_corr: list[dict[str, Any]] = []
    pair_corr = None
    hold_metric = next((m for m in metrics if m["etf"] == current_holding), None)
    if current_holding and current_holding in close.columns:
        corr_codes = _dedupe([current_holding] + [m["etf"] for m in candidate_pool if m["etf"] in close.columns])
        corr_df = adjusted_corr(close.loc[:date, corr_codes].tail(cfg.corr_lookback))
        low_corr_pick = None
        for item in candidate_pool:
            symbol = item["etf"]
            if symbol == current_holding:
                low_corr_pick = item
                break
            p = None
            if not corr_df.empty and current_holding in corr_df.index and symbol in corr_df.columns:
                p = float(corr_df.loc[current_holding, symbol])
            if p is not None and p >= cfg.pick_target_max_padj:
                skipped_high_corr.append({"symbol": symbol, "p_adj": p})
                continue
            low_corr_pick = item
            break
        if low_corr_pick is not None:
            target = low_corr_pick["etf"]
        elif hold_metric is not None:
            target = current_holding
        if target != current_holding and target in close.columns:
            corr_df = adjusted_corr(close.loc[:date, [current_holding, target]].tail(cfg.corr_lookback))
            if not corr_df.empty and current_holding in corr_df.index and target in corr_df.columns:
                pair_corr = float(corr_df.loc[current_holding, target])
                hold_mom = hold_metric.get("momentum_score") if hold_metric else None
                block = False
                if pair_corr >= cfg.corr_overlay_threshold:
                    block = hold_mom is None or float(hold_mom) <= cfg.corr_overlay_momentum_max
                elif hold_mom is not None and pair_corr > cfg.corr_hold_threshold and float(hold_mom) <= cfg.corr_hold_momentum_max:
                    block = True
                if block:
                    target = current_holding
                    diag["corr_guard_blocked"] = True
    target, hold_diag = _apply_hold_hysteresis(target, current_holding, filtered, cfg)
    return target, metrics[:20], {
        **diag,
        **hold_diag,
        "mode": "momentum",
        "candidate_count": len(candidate_pool),
        "pair_corr": pair_corr,
        "skipped_high_corr": skipped_high_corr[:5],
    }


def _apply_hold_hysteresis(
    target: str | None,
    current_holding: str | None,
    filtered_metrics: list[dict[str, Any]],
    cfg: Config,
) -> tuple[str | None, dict[str, Any]]:
    if not current_holding or not target or target == current_holding:
        return target, {}
    rank_by_symbol = {str(item.get("etf")): idx + 1 for idx, item in enumerate(filtered_metrics)}
    metric_by_symbol = {str(item.get("etf")): item for item in filtered_metrics}
    hold_rank = rank_by_symbol.get(current_holding)
    hold_metric = metric_by_symbol.get(current_holding)
    target_metric = metric_by_symbol.get(target)
    hold_score = None if hold_metric is None else float(hold_metric.get("momentum_score", float("nan")))
    target_score = None if target_metric is None else float(target_metric.get("momentum_score", float("nan")))
    diag: dict[str, Any] = {
        "hold_rank": hold_rank,
        "hold_score": hold_score,
        "target_score": target_score,
    }
    reasons: list[str] = []

    if cfg.hold_rank_top > 0 and hold_rank is not None and hold_rank <= cfg.hold_rank_top:
        reasons.append("hold_rank_top")

    if (
        cfg.switch_score_premium > 0
        and hold_score is not None
        and target_score is not None
        and math.isfinite(hold_score)
        and math.isfinite(target_score)
        and hold_score >= cfg.hold_min_momentum_score
        and target_score <= hold_score * (1.0 + cfg.switch_score_premium)
    ):
        reasons.append("switch_score_premium")

    if reasons:
        diag["hold_hysteresis_blocked"] = True
        diag["hold_hysteresis_reasons"] = reasons
        return current_holding, diag
    return target, diag


def _apply_choppy_confirmation(
    target: str | None,
    regime: str,
    current_holding: str | None,
    cfg: Config,
    state: dict[str, Any],
    defensive_available: bool,
) -> tuple[str | None, dict[str, Any]]:
    if regime != "震荡期" or cfg.choppy_confirm_days <= 1 or target == current_holding:
        state.pop("choppy_pending", None)
        state.pop("choppy_streak", None)
        return target, {}

    pending = target or ""
    if state.get("choppy_pending") == pending:
        state["choppy_streak"] = int(state.get("choppy_streak", 0)) + 1
    else:
        state["choppy_pending"] = pending
        state["choppy_streak"] = 1

    streak = int(state["choppy_streak"])
    if streak >= cfg.choppy_confirm_days:
        return target, {"choppy_confirmed": True, "choppy_confirm_streak": streak}

    fallback = current_holding or (cfg.defensive_etf if defensive_available else None)
    return fallback, {
        "choppy_confirm_wait": True,
        "choppy_pending_target": target,
        "choppy_confirm_streak": streak,
        "choppy_confirm_required": cfg.choppy_confirm_days,
    }


def _trade_cost(value: float, cfg: Config) -> float:
    if value <= 0:
        return 0.0
    return max(float(cfg.min_cost), abs(value) * float(cfg.commission))


def _affordable_quantity(cash: float, price: float, cfg: Config) -> int:
    if cash <= cfg.min_money or price <= 0:
        return 0
    quantity = int(cash // (price * (1.0 + max(0.0, cfg.commission))))
    while quantity > 0:
        value = price * quantity
        if value + _trade_cost(value, cfg) <= cash:
            return quantity
        quantity -= 1
    return 0


def _last_price(close: pd.DataFrame, date: pd.Timestamp, symbol: str) -> float | None:
    if symbol not in close.columns:
        return None
    series = close.loc[:date, symbol].dropna()
    if series.empty:
        return None
    price = float(series.iloc[-1])
    return price if price > 0 and math.isfinite(price) else None


def _trade_price(close: pd.DataFrame, date: pd.Timestamp, symbol: str) -> float | None:
    if symbol not in close.columns or date not in close.index:
        return None
    price = close.at[date, symbol]
    if pd.isna(price):
        return None
    price = float(price)
    return price if price > 0 and math.isfinite(price) else None


def _execute_rebalance(
    date: pd.Timestamp,
    target: str | None,
    position: dict[str, Any] | None,
    cash: float,
    price_table: pd.DataFrame,
    cfg: Config,
    trades: list[dict[str, Any]],
    reason: str = "rebalance",
) -> tuple[dict[str, Any] | None, float]:
    if position and target != position["symbol"]:
        symbol = position["symbol"]
        raw_price = _trade_price(price_table, date, symbol)
        if raw_price is None:
            return position, cash
        price = raw_price * (1.0 - cfg.slippage)
        quantity = int(position["quantity"])
        value = price * quantity
        cost = _trade_cost(value, cfg)
        cash += value - cost
        trades.append({
            "date": date.strftime("%Y-%m-%d"), "symbol": symbol, "action": "SELL", "quantity": quantity,
            "price": price, "trade_value": value, "commission": cost, "slippage": 0.0,
            "stamp_tax": 0.0, "total_cost": cost, "cash_after": cash, "reason": reason,
        })
        position = None

    raw_buy_price = _trade_price(price_table, date, target) if target else None
    if target and position is None and raw_buy_price is not None and cash >= cfg.min_money:
        price = raw_buy_price * (1.0 + cfg.slippage)
        quantity = _affordable_quantity(cash, price, cfg)
        if quantity > 0:
            value = price * quantity
            cost = _trade_cost(value, cfg)
            if value + cost <= cash:
                cash -= value + cost
                position = {
                    "symbol": target,
                    "quantity": quantity,
                    "entry_date": date.strftime("%Y-%m-%d"),
                    "entry_price": price,
                    "avg_cost": (value + cost) / quantity,
                }
                trades.append({
                    "date": date.strftime("%Y-%m-%d"), "symbol": target, "action": "BUY", "quantity": quantity,
                    "price": price, "trade_value": value, "commission": cost, "slippage": 0.0,
                    "stamp_tax": 0.0, "total_cost": cost, "cash_after": cash, "reason": reason,
                })
    return position, cash


def _append_position_snapshot(
    positions: list[dict[str, Any]],
    position: dict[str, Any],
    date: pd.Timestamp,
    close: pd.DataFrame,
) -> None:
    symbol = str(position["symbol"])
    mark_price = _last_price(close, date, symbol)
    if mark_price is None:
        return
    quantity = int(position["quantity"])
    avg_cost = float(position.get("avg_cost") or position.get("entry_price") or 0.0)
    entry_date = pd.Timestamp(position.get("entry_date") or date)
    positions.append({
        "date": date.strftime("%Y-%m-%d"),
        "symbol": symbol,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "market_value": float(quantity * mark_price),
        "holding_days": int((date - entry_date).days),
    })


def _execution_price_table(cfg: Config, close: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    if cfg.execution_mode == "next_open":
        return open_
    return close


def _mark_to_market(
    date: pd.Timestamp,
    cash: float,
    position: dict[str, Any] | None,
    close: pd.DataFrame,
    cfg: Config,
    running_max: float,
    prev_value: float,
) -> tuple[dict[str, Any], float, float]:
    total = cash
    if position:
        mark_price = _last_price(close, date, position["symbol"])
        if mark_price is not None:
            total += float(position["quantity"]) * mark_price
    running_max = max(running_max, total)
    row = {
        "date": date.strftime("%Y-%m-%d"),
        "cash": float(cash),
        "total_value": float(total),
        "daily_return": float(total / prev_value - 1.0) if prev_value else 0.0,
        "cumulative_return": float(total / cfg.init_cash - 1.0),
        "drawdown": float(total / running_max - 1.0),
        "position_count": 1 if position else 0,
    }
    return row, running_max, total


def run_backtest(cfg: Config) -> dict[str, Any]:
    symbols = _all_pool_symbols(cfg.source_strategy)
    symbols = _dedupe(symbols + [cfg.defensive_etf])
    theme_symbols, theme_meta = _load_theme_symbols_from_meta(cfg.etf_meta_path)
    local_theme_symbols = [symbol for symbol in theme_symbols if (cfg.raw_dir / f"{symbol}.csv").exists()]
    symbols = _dedupe([*symbols, *local_theme_symbols])
    sync_result = _sync_missing_etfs(symbols, cfg)
    bars, missing_after = _load_bars(symbols, cfg.raw_dir, cfg.start, cfg.end)
    if not bars:
        raise RuntimeError("No ETF bars loaded")
    close = _close_table(bars)
    open_ = _close_table(bars, "open")
    dates = [d for d in close.index if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)]
    regime_sources = _choose_regime_sources(close, cfg)
    regime_close = _merged_close_for_regime(close, cfg, regime_sources)

    cash = float(cfg.init_cash)
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    daily_nav: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    regime_state: dict[str, Any] = {}
    choppy_confirm_state: dict[str, Any] = {}
    running_max = cfg.init_cash
    prev_value = cfg.init_cash
    pending_target: str | None = None
    pending_signal_date: pd.Timestamp | None = None

    for date in dates:
        if cfg.execution_mode in {"next_open", "next_close"} and pending_signal_date is not None:
            position, cash = _execute_rebalance(
                date,
                pending_target,
                position,
                cash,
                _execution_price_table(cfg, close, open_),
                cfg,
                trades,
                reason=f"signal:{pending_signal_date.strftime('%Y-%m-%d')}",
            )
            pending_target = None
            pending_signal_date = None

        if len(close.loc[:date]) < max(cfg.lookback_days, cfg.short_lookback, cfg.corr_lookback):
            nav_row, running_max, total = _mark_to_market(date, cash, position, close, cfg, running_max, prev_value)
            daily_nav.append(nav_row)
            if position:
                _append_position_snapshot(positions, position, date, close)
            prev_value = total
            continue

        regime = resolve_regime(date, regime_close, regime_state, cfg)
        current_holding = position["symbol"] if position else None
        target, top_metrics, diag = _select_target(
            date,
            regime,
            bars,
            close,
            current_holding,
            cfg,
            local_theme_symbols,
            theme_meta,
        )
        target, confirm_diag = _apply_choppy_confirmation(
            target,
            regime,
            current_holding,
            cfg,
            choppy_confirm_state,
            cfg.defensive_etf in bars,
        )
        diag.update(confirm_diag)
        selection.append({
            "date": date.strftime("%Y-%m-%d"),
            "regime": regime,
            "target": target,
            "execution_mode": cfg.execution_mode,
            **diag,
            "top": top_metrics[:5],
        })

        if cfg.execution_mode == "same_close":
            position, cash = _execute_rebalance(date, target, position, cash, close, cfg, trades)
        else:
            pending_target = target
            pending_signal_date = date

        nav_row, running_max, total = _mark_to_market(date, cash, position, close, cfg, running_max, prev_value)
        daily_nav.append(nav_row)
        if position:
            _append_position_snapshot(positions, position, date, close)
        prev_value = total

    closed = match_closed_positions(trades)
    metrics = compute_metrics(daily_nav, trades, closed, cfg.init_cash)
    confirm_suffix = f"_confirm{cfg.choppy_confirm_days}" if cfg.choppy_confirm_days > 1 else ""
    dynamic_suffix = f"_dyn_theme{cfg.dynamic_theme_top_n}" if cfg.dynamic_theme_top_n > 0 else ""
    run_name = f"etf_wufu_{cfg.execution_mode}{confirm_suffix}{dynamic_suffix}_corr_daily_2016_2026"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{run_name}"
    run_dir = cfg.runs_dir / run_id
    execution_desc = {
        "same_close": "same-day close signal and same-day close execution diagnostic",
        "next_open": "T close signal with T+1 open execution",
        "next_close": "T close signal with T+1 close execution",
    }[cfg.execution_mode]
    summary = {
        "run_id": run_id,
        "name": run_name,
        "description": f"Daily Wufu ETF rotation with adjusted-correlation guard; {execution_desc}",
        "execution_mode": cfg.execution_mode,
        "dynamic_theme": {
            "top_n": cfg.dynamic_theme_top_n,
            "min_score": cfg.dynamic_theme_min_score,
            "liquidity_lookback": cfg.dynamic_theme_liquidity_lookback,
            "min_avg_amount": cfg.dynamic_theme_min_avg_amount,
            "regimes": cfg.dynamic_theme_regimes,
            "score_premium": cfg.dynamic_theme_score_premium,
            "meta_path": str(cfg.etf_meta_path),
            "meta_candidates": len(theme_symbols),
            "local_candidates": len(local_theme_symbols),
        },
        "start_date": daily_nav[0]["date"] if daily_nav else cfg.start,
        "end_date": daily_nav[-1]["date"] if daily_nav else cfg.end,
        "symbols": len(bars),
        "missing_symbols": missing_after,
        "regime_sources": regime_sources,
        "sync": sync_result,
        "final_value": metrics.get("final_value"),
        "total_return": metrics.get("total_return"),
        "annual_return": metrics.get("annual_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "trades": metrics.get("trade_count"),
        "final_cash": cash,
        "final_positions": 1 if position else 0,
        "caveats": [
            "Signals are computed from daily bars; strict modes execute on the following trading day.",
            "same_close mode is kept only as an optimistic diagnostic and should not be treated as a daily-trading backtest.",
            "Minute accumulated volume is approximated by daily volume multiplied by intraday_volume_fraction.",
            "Premium/NAV filters are not applied because reliable daily NAV history is not available locally.",
            "Market regime uses local index data when present and ETF proxies otherwise.",
        ],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "daily_nav.json", daily_nav)
    _write_json(run_dir / "trades.json", trades)
    _write_json(run_dir / "positions.json", positions)
    _write_json(run_dir / "closed_positions.json", closed)
    _write_json(run_dir / "selection_candidates.json", selection)
    _write_json(run_dir / "explain.json", {
        "config": {
            "name": summary["name"],
            "description": summary["description"],
            "data": {"provider_uri": "data/qlib_etf_data_wufu"},
        }
    })
    _write_json(run_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()})
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": summary, "metrics": metrics}


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--init-cash", type=float, default=DEFAULT_INIT_CASH)
    parser.add_argument("--raw-dir", default="data/raw/eastmoney/etfs")
    parser.add_argument("--etf-meta-path", default=DEFAULT_ETF_META_PATH)
    parser.add_argument("--index-raw-dir", default="data/raw/baostock/stocks")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--source-strategy", default=DEFAULT_SOURCE_STRATEGY)
    parser.add_argument("--sync-missing", action="store_true")
    parser.add_argument("--source", default="sina", choices=["sina", "auto", "eastmoney", "em"])
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--min-money", type=float, default=10.0)
    parser.add_argument("--intraday-volume-fraction", type=float, default=0.67)
    parser.add_argument("--hold-rank-top", type=int, default=0)
    parser.add_argument("--switch-score-premium", type=float, default=0.0)
    parser.add_argument("--hold-min-momentum-score", type=float, default=0.0)
    parser.add_argument("--choppy-confirm-days", type=int, default=1)
    parser.add_argument("--choppy-min-filtered-count", type=int, default=0)
    parser.add_argument("--execution-mode", choices=["same_close", "next_open", "next_close"], default="next_open")
    parser.add_argument("--dynamic-theme-top-n", type=int, default=0)
    parser.add_argument("--dynamic-theme-min-score", type=float, default=0.0)
    parser.add_argument("--dynamic-theme-liquidity-lookback", type=int, default=3)
    parser.add_argument("--dynamic-theme-min-avg-amount", type=float, default=50_000_000.0)
    parser.add_argument(
        "--dynamic-theme-regimes",
        default="正常期,震荡期",
        help="Comma-separated regimes where dynamic theme ETFs may survive the gate; use all for no regime gate.",
    )
    parser.add_argument(
        "--dynamic-theme-score-premium",
        type=float,
        default=0.05,
        help="Required dynamic theme score premium over the best non-dynamic filtered candidate.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config(
        start=args.start, end=args.end, init_cash=args.init_cash, raw_dir=Path(args.raw_dir),
        etf_meta_path=Path(args.etf_meta_path),
        index_raw_dir=Path(args.index_raw_dir), runs_dir=Path(args.runs_dir), source_strategy=Path(args.source_strategy),
        sync_missing=args.sync_missing, source=args.source, pause_seconds=args.pause_seconds,
        commission=args.commission, slippage=args.slippage, min_cost=args.min_cost, min_money=args.min_money,
        lookback_days=25, short_lookback=21, min_score=0.0, max_score=5.0, score_threshold_ratio=0.9,
        r2_threshold=0.4, normal_r2_threshold=0.39, ma_lookback=10, ma_threshold=1.0001,
        volume_lookback=5, volume_threshold=1.9, intraday_volume_fraction=args.intraday_volume_fraction,
        loss_threshold=0.97, corr_lookback=60, corr_hold_threshold=0.85, corr_hold_momentum_max=7.0,
        corr_overlay_threshold=0.88, corr_overlay_momentum_max=8.0, pick_target_max_padj=0.85,
        regime_confirm_days=2, weak_below_ma20_min=4, normal_above_ma10_min=4,
        defensive_etf="SH511880", top_n=1,
        hold_rank_top=args.hold_rank_top,
        switch_score_premium=args.switch_score_premium,
        hold_min_momentum_score=args.hold_min_momentum_score,
        choppy_confirm_days=args.choppy_confirm_days,
        choppy_min_filtered_count=args.choppy_min_filtered_count,
        execution_mode=args.execution_mode,
        dynamic_theme_top_n=args.dynamic_theme_top_n,
        dynamic_theme_min_score=args.dynamic_theme_min_score,
        dynamic_theme_liquidity_lookback=args.dynamic_theme_liquidity_lookback,
        dynamic_theme_min_avg_amount=args.dynamic_theme_min_avg_amount,
        dynamic_theme_regimes=_parse_regime_list(args.dynamic_theme_regimes),
        dynamic_theme_score_premium=args.dynamic_theme_score_premium,
    )
    cfg._json = bool(args.json)  # type: ignore[attr-defined]
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    result = run_backtest(cfg)
    if getattr(cfg, "_json", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        summary = result["summary"]
        print(
            f"run_id={summary['run_id']} total_return={summary['total_return']:.4f} "
            f"annual_return={summary['annual_return']:.4f} max_drawdown={summary['max_drawdown']:.4f} "
            f"trades={summary['trades']} symbols={summary['symbols']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
