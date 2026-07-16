"""Evaluate cached QMT 5m intraday proxies before trying deeper models.

This is a temporary research script. It deliberately avoids production modules
except for filesystem conventions and uses only local cached 5m bars plus local
daily QMT CSVs, so the result is reproducible without a live trading session.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "cached_5m_signal_utility_summary.json"
OUT_MD = OUT_DIR / "cached_5m_signal_utility_report.md"

CACHE_DIRS = [
    REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1/qmt_minute_cache_v1/5m",
    REPO_ROOT / ".tmp/quantx-research/qmt-minute-coverage-probe-v1/cache/5m",
]
DAILY_DIR = REPO_ROOT / "data/raw/qmt/stocks"
FILENAME_RE = re.compile(r"^(?P<symbol>[A-Z]{2}\d{6})_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")

FEATURES_DAILY = [
    "daily_ret",
    "daily_high_ret",
    "daily_close_from_high",
    "daily_close_vs_vwap",
    "daily_volume_z20",
    "daily_amount_z20",
]
FEATURES_MINUTE = [
    "m_close_ret",
    "m_high_ret",
    "m_low_ret",
    "m_close_from_high",
    "m_vwap_support",
    "m_tail_ret_30m",
    "m_tail_volume_share_30m",
    "m_volume_concentration_top20pct",
    "m_first_high_frac",
    "m_last_bar_ret",
    "m_near_limit_ratio",
    "m_limit_touch_flag",
    "m_break_proxy_count",
]
LABELS = ["next_open_ret", "next_open_to_close_ret", "next_open_to_high_ret", "next_close_ret"]


@dataclass(frozen=True)
class MinuteFile:
    path: Path
    symbol: str
    date: str


def symbol_from_file_symbol(value: str) -> str:
    if value.startswith("SH") or value.startswith("SZ"):
        return value
    if value.endswith(".SH"):
        return "SH" + value[:6]
    if value.endswith(".SZ"):
        return "SZ" + value[:6]
    return value


def discover_minute_files() -> list[MinuteFile]:
    chosen: dict[tuple[str, str], Path] = {}
    for cache_dir in CACHE_DIRS:
        if not cache_dir.exists():
            continue
        for path in cache_dir.glob("*.csv"):
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            key = (match.group("symbol"), match.group("date"))
            chosen.setdefault(key, path)
    return [MinuteFile(path=path, symbol=symbol, date=date) for (symbol, date), path in sorted(chosen.items())]


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.stat().st_size <= 1:
            return pd.DataFrame()
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if frame.empty or len(frame.columns) <= 1:
        return pd.DataFrame()
    return frame


def load_daily(symbol: str) -> pd.DataFrame:
    path = DAILY_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    frame = frame.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def zscore_last(values: pd.Series, window: int = 20) -> float:
    hist = pd.to_numeric(values, errors="coerce").dropna().tail(window)
    if len(hist) < 5:
        return 0.0
    std = float(hist.std(ddof=0))
    if std <= 0 or not np.isfinite(std):
        return 0.0
    return float((hist.iloc[-1] - hist.mean()) / std)


def daily_features_and_labels(daily: pd.DataFrame, date: str) -> dict[str, float] | None:
    idxs = np.flatnonzero(daily["date"].astype(str).to_numpy() == date)
    if len(idxs) == 0:
        return None
    idx = int(idxs[0])
    if idx + 1 >= len(daily):
        return None
    row = daily.iloc[idx]
    nxt = daily.iloc[idx + 1]
    preclose = float(row.get("preclose", np.nan))
    close = float(row.get("close", np.nan))
    high = float(row.get("high", np.nan))
    amount = float(row.get("amount", np.nan))
    volume = float(row.get("volume", np.nan))
    if not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(close) or close <= 0:
        return None
    vwap = amount / volume / 100.0 if np.isfinite(amount) and np.isfinite(volume) and volume > 0 else close
    next_open = float(nxt.get("open", np.nan))
    next_high = float(nxt.get("high", np.nan))
    next_close = float(nxt.get("close", np.nan))
    if not all(np.isfinite(x) and x > 0 for x in [next_open, next_high, next_close]):
        return None
    hist = daily.iloc[: idx + 1]
    return {
        "preclose": preclose,
        "daily_ret": close / preclose - 1.0,
        "daily_high_ret": high / preclose - 1.0 if np.isfinite(high) and high > 0 else 0.0,
        "daily_close_from_high": close / high - 1.0 if np.isfinite(high) and high > 0 else 0.0,
        "daily_close_vs_vwap": close / vwap - 1.0 if np.isfinite(vwap) and vwap > 0 else 0.0,
        "daily_volume_z20": zscore_last(hist["volume"]),
        "daily_amount_z20": zscore_last(hist["amount"]),
        "next_open_ret": next_open / close - 1.0,
        "next_open_to_close_ret": next_close / next_open - 1.0,
        "next_open_to_high_ret": next_high / next_open - 1.0,
        "next_close_ret": next_close / close - 1.0,
    }


def minute_features(frame: pd.DataFrame, preclose: float) -> dict[str, float] | None:
    if frame.empty or not np.isfinite(preclose) or preclose <= 0:
        return None
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in frame.columns:
            return None
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if frame.empty:
        return None
    if "datetime" in frame.columns:
        frame = frame.sort_values("datetime")
    else:
        frame = frame.sort_values("time") if "time" in frame.columns else frame
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].fillna(0).to_numpy(dtype=float)
    amount = frame["amount"].fillna(0).to_numpy(dtype=float)
    n = len(frame)
    if n < 6:
        return None
    total_volume = float(np.nansum(volume))
    total_amount = float(np.nansum(amount))
    vwap = total_amount / total_volume / 100.0 if total_volume > 0 and total_amount > 0 else float(np.nanmean(close))
    day_high = float(np.nanmax(high))
    day_low = float(np.nanmin(low))
    last_close = float(close[-1])
    first_high_idx = int(np.nanargmax(high)) if np.isfinite(high).any() else n - 1
    tail_n = min(6, n)
    top_count = max(1, int(math.ceil(n * 0.2)))
    limit_price = preclose * 1.095
    near_price = preclose * 1.085
    near_limit = high >= near_price
    touch = high >= limit_price
    break_proxy = touch & (close < near_price)
    return {
        "minute_rows": float(n),
        "m_close_ret": last_close / preclose - 1.0,
        "m_high_ret": day_high / preclose - 1.0,
        "m_low_ret": day_low / preclose - 1.0,
        "m_close_from_high": last_close / day_high - 1.0 if day_high > 0 else 0.0,
        "m_vwap_support": last_close / vwap - 1.0 if np.isfinite(vwap) and vwap > 0 else 0.0,
        "m_tail_ret_30m": last_close / close[-tail_n] - 1.0 if close[-tail_n] > 0 else 0.0,
        "m_tail_volume_share_30m": float(np.nansum(volume[-tail_n:]) / total_volume) if total_volume > 0 else 0.0,
        "m_volume_concentration_top20pct": float(np.nansum(np.sort(volume)[-top_count:]) / total_volume) if total_volume > 0 else 0.0,
        "m_first_high_frac": first_high_idx / max(1, n - 1),
        "m_last_bar_ret": close[-1] / close[-2] - 1.0 if close[-2] > 0 else 0.0,
        "m_near_limit_ratio": float(np.nanmean(near_limit)),
        "m_limit_touch_flag": float(bool(np.any(touch))),
        "m_break_proxy_count": float(np.nansum(break_proxy)),
    }


def build_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    files = discover_minute_files()
    rows: list[dict[str, Any]] = []
    status = {
        "minute_files_discovered": len(files),
        "minute_empty_or_bad": 0,
        "daily_missing": 0,
        "label_missing": 0,
        "minute_feature_missing": 0,
        "rows_built": 0,
    }
    daily_cache: dict[str, pd.DataFrame] = {}
    for item in files:
        minute = safe_read_csv(item.path)
        if minute.empty:
            status["minute_empty_or_bad"] += 1
            continue
        daily = daily_cache.setdefault(item.symbol, load_daily(item.symbol))
        if daily.empty:
            status["daily_missing"] += 1
            continue
        daily_part = daily_features_and_labels(daily, item.date)
        if daily_part is None:
            status["label_missing"] += 1
            continue
        minute_part = minute_features(minute, daily_part["preclose"])
        if minute_part is None:
            status["minute_feature_missing"] += 1
            continue
        row = {"symbol": item.symbol, "date": item.date, "source_path": str(item.path)}
        row.update(daily_part)
        row.update(minute_part)
        rows.append(row)
    panel = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    status["rows_built"] = int(len(panel))
    status["unique_dates"] = int(panel["date"].nunique()) if not panel.empty else 0
    status["unique_symbols"] = int(panel["symbol"].nunique()) if not panel.empty else 0
    return panel, status


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def factor_stats(panel: pd.DataFrame, features: list[str], label: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature in features:
        pooled_ic = spearman(panel[feature], panel[label])
        daily_ics = []
        qspread = []
        for _, group in panel.groupby("date"):
            if len(group) < 5:
                continue
            ic = spearman(group[feature], group[label])
            if np.isfinite(ic):
                daily_ics.append(ic)
            ranked = group.sort_values(feature)
            k = max(1, len(ranked) // 5)
            low = float(ranked.head(k)[label].mean())
            high = float(ranked.tail(k)[label].mean())
            qspread.append(high - low)
        out[feature] = {
            "pooled_spearman": pooled_ic,
            "daily_ic_mean": float(np.nanmean(daily_ics)) if daily_ics else float("nan"),
            "daily_ic_tstat": float(np.nanmean(daily_ics) / (np.nanstd(daily_ics, ddof=1) / math.sqrt(len(daily_ics)))) if len(daily_ics) > 2 and np.nanstd(daily_ics, ddof=1) > 0 else 0.0,
            "daily_ic_n": len(daily_ics),
            "top_bottom_spread_mean": float(np.nanmean(qspread)) if qspread else float("nan"),
            "top_bottom_spread_hit_rate": float(np.mean(np.asarray(qspread) > 0)) if qspread else float("nan"),
        }
    return out


def chronological_model(panel: pd.DataFrame, features: list[str], label: str) -> dict[str, Any]:
    frame = panel[["date", "symbol", label] + features].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(frame) < 60 or frame["date"].nunique() < 6:
        return {"ok": False, "reason": "sample_too_small", "rows": int(len(frame)), "dates": int(frame["date"].nunique())}
    dates = sorted(frame["date"].unique())
    split = dates[int(len(dates) * 0.6)]
    train = frame[frame["date"] < split]
    test = frame[frame["date"] >= split]
    if len(train) < 30 or len(test) < 20:
        return {"ok": False, "reason": "train_test_too_small", "train_rows": int(len(train)), "test_rows": int(len(test))}
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_train = train[label].to_numpy(dtype=float)
    means = np.nanmean(x_train, axis=0)
    stds = np.nanstd(x_train, axis=0)
    stds[stds == 0] = 1.0
    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    y_mean = float(np.mean(y_train))
    y_center = y_train - y_mean
    alpha = 5.0
    beta = np.linalg.solve(x_train.T @ x_train + alpha * np.eye(x_train.shape[1]), x_train.T @ y_center)
    pred = x_test @ beta + y_mean
    scored = test[["date", "symbol", label]].copy()
    scored["pred"] = pred
    daily_spreads = []
    for _, group in scored.groupby("date"):
        if len(group) < 5:
            continue
        ranked = group.sort_values("pred")
        k = max(1, len(ranked) // 5)
        daily_spreads.append(float(ranked.tail(k)[label].mean() - ranked.head(k)[label].mean()))
    return {
        "ok": True,
        "label": label,
        "features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "split_date": str(split),
        "test_spearman": spearman(scored["pred"], scored[label]),
        "test_top_bottom_spread_mean": float(np.nanmean(daily_spreads)) if daily_spreads else float("nan"),
        "test_top_bottom_spread_hit_rate": float(np.mean(np.asarray(daily_spreads) > 0)) if daily_spreads else float("nan"),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    return value


def verdict(summary: dict[str, Any]) -> str:
    cov = summary["coverage"]
    if cov["rows_built"] < 100 or cov["unique_dates"] < 10:
        return "sample_too_sparse_to_validate_deep_learning"
    model = summary["models"].get("daily_plus_minute_next_open_to_high_ret", {})
    base = summary["models"].get("daily_only_next_open_to_high_ret", {})
    if model.get("ok") and base.get("ok"):
        improves = (model.get("test_spearman") or -9) > (base.get("test_spearman") or -9) + 0.03
        useful_spread = (model.get("test_top_bottom_spread_mean") or 0) > 0.003
        if improves and useful_spread:
            return "weak_positive_candidate_needs_live_orderbook_collection"
    return "no_reliable_cached_5m_edge_found"


def write_report(summary: dict[str, Any]) -> None:
    cov = summary["coverage"]
    lines = [
        "# Cached 5m Intraday Signal Utility",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Coverage",
        "",
        f"- minute files discovered: {cov['minute_files_discovered']}",
        f"- rows built: {cov['rows_built']}",
        f"- unique dates: {cov.get('unique_dates', 0)}",
        f"- unique symbols: {cov.get('unique_symbols', 0)}",
        f"- empty/bad minute files: {cov['minute_empty_or_bad']}",
        f"- missing labels/daily rows: {cov['label_missing'] + cov['daily_missing']}",
        "",
        "## Model Check",
        "",
    ]
    for name, result in summary["models"].items():
        if not result.get("ok"):
            lines.append(f"- {name}: not run ({result.get('reason')})")
            continue
        lines.append(
            f"- {name}: test_spearman={result['test_spearman']:.4f}, "
            f"top-bottom={result['test_top_bottom_spread_mean']:.4%}, "
            f"hit={result['test_top_bottom_spread_hit_rate']:.1%}, "
            f"test_rows={result['test_rows']}"
        )
    lines.extend(["", "## Strongest Single Factors", ""])
    for label, stats in summary["factor_stats"].items():
        ranked = sorted(
            stats.items(),
            key=lambda kv: abs(kv[1].get("daily_ic_mean") or 0),
            reverse=True,
        )[:8]
        lines.append(f"### {label}")
        for feature, item in ranked:
            lines.append(
                f"- {feature}: daily_ic={item['daily_ic_mean']:.4f}, "
                f"t={item['daily_ic_tstat']:.2f}, spread={item['top_bottom_spread_mean']:.4%}, "
                f"hit={item['top_bottom_spread_hit_rate']:.1%}"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(summary["interpretation"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel, coverage = build_panel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUT_DIR / "cached_5m_signal_panel.csv"
    panel.to_csv(panel_path, index=False)
    factor_features = FEATURES_DAILY + FEATURES_MINUTE
    stats = {label: factor_stats(panel, factor_features, label) for label in LABELS} if not panel.empty else {}
    models = {
        "daily_only_next_open_to_high_ret": chronological_model(panel, FEATURES_DAILY, "next_open_to_high_ret"),
        "daily_plus_minute_next_open_to_high_ret": chronological_model(panel, FEATURES_DAILY + FEATURES_MINUTE, "next_open_to_high_ret"),
        "daily_only_next_close_ret": chronological_model(panel, FEATURES_DAILY, "next_close_ret"),
        "daily_plus_minute_next_close_ret": chronological_model(panel, FEATURES_DAILY + FEATURES_MINUTE, "next_close_ret"),
    }
    summary: dict[str, Any] = {
        "experiment": "intraday_signal_dl_validation_cached_5m_v1",
        "data_scope": {
            "minute_cache_dirs": [str(path) for path in CACHE_DIRS],
            "daily_dir": str(DAILY_DIR),
            "note": "Uses cached 5m OHLCV only. No historical five-level orderbook, queue, Level2 orders, or true PIT theme data are available in this offline sample.",
        },
        "coverage": coverage,
        "panel_csv": str(panel_path),
        "factor_stats": stats,
        "models": models,
        "interpretation": [
            "The cached sample is useful as a smoke test for intraday path proxies, but it is not enough to validate a deep learning pipeline on orderbook-style signals.",
            "Any positive result here would still need live get_full_tick collection, because the cached 5m bars do not contain bid/ask volumes, seal queue changes, cancellations, or true money-flow direction.",
            "A negative or unstable result means the current cached 5m proxy layer should not be promoted into deep learning features without richer live data.",
        ],
    }
    summary["verdict"] = verdict(summary)
    OUT_JSON.write_text(json.dumps(sanitize(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(sanitize(summary))
    print(json.dumps({"summary": str(OUT_JSON), "report": str(OUT_MD), "verdict": summary["verdict"], "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
