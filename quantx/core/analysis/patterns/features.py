"""K-line bar and trade-window feature engineering."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PatternAnalysisConfig
from .market_data import DailyBarLoader


def build_trade_windows(
    trades: pd.DataFrame,
    loader: DailyBarLoader,
    config: PatternAnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per trade window and one row per daily bar in each window."""
    window_rows: List[Dict[str, object]] = []
    bar_frames: List[pd.DataFrame] = []
    preload_start, preload_end = _preload_range(trades, config)
    loader.preload_symbols(trades["symbol"].astype(str).unique(), start=preload_start, end=preload_end)

    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        bars = loader.load_symbol(symbol)
        if bars.empty:
            continue
        bars = add_bar_features(bars)
        dates = pd.DatetimeIndex(bars["date"])
        entry_idx = _date_index(dates, trade["entry_date"], side="left")
        exit_idx = _date_index(dates, trade["exit_date"], side="right")
        if entry_idx is None or exit_idx is None:
            continue
        start_idx = max(0, entry_idx - config.pre_n)
        end_idx = min(len(bars) - 1, exit_idx + config.post_n)
        pre_count = entry_idx - start_idx
        post_count = end_idx - exit_idx
        if pre_count < config.min_pre_bars or post_count < config.min_post_bars:
            continue

        sample_id = str(trade["sample_id"])
        window = bars.iloc[start_idx : end_idx + 1].copy()
        window["sample_id"] = sample_id
        window["run_id"] = trade.get("run_id")
        window["strategy_name"] = trade.get("strategy_name")
        window["entry_date"] = pd.Timestamp(trade["entry_date"])
        window["exit_date"] = pd.Timestamp(trade["exit_date"])
        window["offset_from_entry"] = np.arange(start_idx - entry_idx, end_idx - entry_idx + 1)
        window["offset_from_exit"] = np.arange(start_idx - exit_idx, end_idx - exit_idx + 1)
        window["is_pre_entry"] = window["offset_from_entry"] < 0
        window["is_holding"] = (window["offset_from_entry"] >= 0) & (window["offset_from_exit"] <= 0)
        window["is_post_exit"] = window["offset_from_exit"] > 0
        bar_frames.append(window)

        mfe, mae = _excursions(bars.iloc[entry_idx : exit_idx + 1], float(trade.get("entry_price") or bars.iloc[entry_idx]["close"]))
        window_rows.append({
            "sample_id": sample_id,
            "run_id": trade.get("run_id"),
            "strategy_name": trade.get("strategy_name"),
            "symbol": symbol,
            "entry_date": pd.Timestamp(trade["entry_date"]).strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(trade["exit_date"]).strftime("%Y-%m-%d"),
            "entry_price": float(trade.get("entry_price") or bars.iloc[entry_idx]["close"]),
            "exit_price": float(trade.get("exit_price") or bars.iloc[exit_idx]["close"]),
            "holding_days": int(trade.get("holding_days") or max(exit_idx - entry_idx, 0)),
            "exit_reason": str(trade.get("exit_reason") or ""),
            "window_start": bars.iloc[start_idx]["date"].strftime("%Y-%m-%d"),
            "window_end": bars.iloc[end_idx]["date"].strftime("%Y-%m-%d"),
            "pre_n": config.pre_n,
            "post_n": config.post_n,
            "pre_bars": int(pre_count),
            "post_bars": int(post_count),
            "return": float(trade.get("return") if pd.notna(trade.get("return")) else (bars.iloc[exit_idx]["close"] / bars.iloc[entry_idx]["close"] - 1)),
            "pnl": float(trade.get("net_pnl")) if pd.notna(trade.get("net_pnl")) else 0.0,
            "max_favorable_excursion": mfe,
            "max_adverse_excursion": mae,
            "source_artifact": trade.get("source_artifact"),
        })

    windows = pd.DataFrame(window_rows)
    bar_features = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame()
    return windows, bar_features


def add_bar_features(frame: pd.DataFrame) -> pd.DataFrame:
    bars = frame.copy().sort_values("date").reset_index(drop=True)
    prev_close = bars["close"].shift(1).replace(0, np.nan)
    price_range = (bars["high"] - bars["low"]).replace(0, np.nan)
    body = (bars["close"] - bars["open"]).abs()
    upper_shadow = bars["high"] - bars[["open", "close"]].max(axis=1)
    lower_shadow = bars[["open", "close"]].min(axis=1) - bars["low"]

    bars["return_pct"] = bars["close"].pct_change(fill_method=None).fillna(0.0)
    if "preclose" in bars.columns:
        bars["return_pct"] = np.where(bars["preclose"] > 0, bars["close"] / bars["preclose"] - 1, bars["return_pct"])
    bars["gap_pct"] = bars["open"] / prev_close - 1
    bars["range_pct"] = (bars["high"] - bars["low"]) / prev_close
    bars["body_pct"] = body / prev_close
    bars["body_to_range"] = (body / price_range).fillna(0.0)
    bars["upper_shadow_ratio"] = (upper_shadow / price_range).fillna(0.0)
    bars["lower_shadow_ratio"] = (lower_shadow / price_range).fillna(0.0)
    bars["close_position"] = ((bars["close"] - bars["low"]) / price_range).fillna(0.5)
    bars["is_up_day"] = bars["close"] > bars["open"]
    bars["is_down_day"] = bars["close"] < bars["open"]
    bars["is_doji"] = bars["body_to_range"] <= 0.15
    bars["is_long_upper_shadow"] = bars["upper_shadow_ratio"] >= 0.45
    bars["is_long_lower_shadow"] = bars["lower_shadow_ratio"] >= 0.45
    bars["is_long_body"] = bars["body_pct"] >= bars["body_pct"].rolling(20, min_periods=5).mean().fillna(bars["body_pct"].mean()) * 1.5
    bars["is_wide_range"] = bars["range_pct"] >= bars["range_pct"].rolling(20, min_periods=5).mean().fillna(bars["range_pct"].mean()) * 1.5

    bars["volume_ma5"] = bars["volume"].rolling(5, min_periods=1).mean()
    bars["volume_ma20"] = bars["volume"].rolling(20, min_periods=1).mean()
    bars["amount_ma20"] = bars["amount"].rolling(20, min_periods=1).mean()
    bars["volume_ratio_5"] = _safe_ratio(bars["volume"], bars["volume_ma5"])
    bars["volume_ratio_20"] = _safe_ratio(bars["volume"], bars["volume_ma20"])
    bars["amount_ratio_20"] = _safe_ratio(bars["amount"], bars["amount_ma20"])
    bars["volume_trend_5"] = bars["volume"].rolling(5, min_periods=3).apply(_slope, raw=True).fillna(0.0)
    bars["is_high_volume"] = bars["volume_ratio_20"] >= 2.0
    bars["is_volume_contracting"] = bars["volume_ratio_20"] <= 0.7

    for period in [5, 10, 20, 60]:
        ma = bars["close"].rolling(period, min_periods=1).mean()
        bars[f"ma{period}"] = ma
        bars[f"ma{period}_distance"] = _safe_ratio(bars["close"], ma) - 1
    bars["ma5_slope"] = bars["ma5"].rolling(5, min_periods=3).apply(_slope, raw=True).fillna(0.0)
    bars["ma20_slope"] = bars["ma20"].rolling(10, min_periods=5).apply(_slope, raw=True).fillna(0.0)
    high20 = bars["high"].shift(1).rolling(20, min_periods=1).max()
    high60 = bars["high"].shift(1).rolling(60, min_periods=1).max()
    bars["breakout_20d_high"] = bars["close"] > high20
    bars["distance_to_20d_high"] = _safe_ratio(bars["close"], high20) - 1
    bars["distance_to_60d_high"] = _safe_ratio(bars["close"], high60) - 1
    return bars.replace([np.inf, -np.inf], np.nan)


def build_window_features(windows: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if windows.empty or bars.empty:
        return pd.DataFrame()
    for _, window in windows.iterrows():
        sample_id = window["sample_id"]
        sample_bars = bars[bars["sample_id"] == sample_id].sort_values("offset_from_entry")
        row: Dict[str, object] = window.to_dict()
        row.update(_exit_reason_features(str(window.get("exit_reason") or "")))
        pre = sample_bars[sample_bars["offset_from_entry"] < 0]
        entry = sample_bars[sample_bars["offset_from_entry"] == 0]
        holding = sample_bars[sample_bars["is_holding"]]
        post = sample_bars[sample_bars["is_post_exit"]]
        for days in [5, 10, 20, 40]:
            segment = pre.tail(days)
            row.update(_segment_features(segment, f"pre_{days}"))
        if not entry.empty:
            row.update(_entry_features(entry.iloc[-1]))
        row.update(_review_features(holding, post, float(window.get("entry_price") or 0.0)))
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def train_feature_columns(frame: pd.DataFrame, feature_scope: str = "entry") -> List[str]:
    blocked_names = {
        "sample_id",
        "run_id",
        "strategy_name",
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "window_start",
        "window_end",
        "source_artifact",
        "pre_n",
        "pre_bars",
        "post_n",
        "post_bars",
        "pnl",
        "outcome_label",
        "binary_success",
        "behavior_label",
        "opportunity_label",
        "had_opportunity",
        "fade_label",
        "faded_after_peak",
        "efficient_capture",
        "is_failed_trade",
        "recoverable_failure",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "recovery_label",
    }
    blocked_prefixes = ("post_",)
    if feature_scope == "entry":
        allowed_prefixes = ("pre_", "entry_")
        allowed_names: set[str] = set()
    elif feature_scope == "trade_management":
        allowed_prefixes = ("pre_", "entry_", "hold_", "exit_reason_")
        allowed_names = {"holding_days", "return", "max_favorable_excursion", "max_adverse_excursion"}
    else:
        raise ValueError(f"Unsupported feature_scope: {feature_scope}")
    columns = []
    for column in frame.columns:
        if column in blocked_names:
            continue
        if column.startswith(blocked_prefixes):
            continue
        if column not in allowed_names and not column.startswith(allowed_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            columns.append(column)
    return columns


def _exit_reason_features(reason: str) -> Dict[str, bool]:
    normalized = reason.lower()
    return {
        "exit_reason_stop_loss": "stop_loss" in normalized,
        "exit_reason_time_stop": "time_stop" in normalized,
        "exit_reason_take_profit": "take_profit" in normalized,
        "exit_reason_trailing": "trail" in normalized or "trailing" in normalized,
        "exit_reason_strategy": "strategy" in normalized,
        "exit_reason_unknown": normalized == "",
    }


def _segment_features(segment: pd.DataFrame, prefix: str) -> Dict[str, float]:
    if segment.empty:
        return {
            f"{prefix}_return": np.nan,
            f"{prefix}_max_drawdown": np.nan,
            f"{prefix}_volatility": np.nan,
            f"{prefix}_range_mean": np.nan,
            f"{prefix}_up_day_ratio": np.nan,
            f"{prefix}_long_upper_count": 0,
            f"{prefix}_long_lower_count": 0,
            f"{prefix}_high_volume_count": 0,
            f"{prefix}_volume_trend": np.nan,
            f"{prefix}_volume_price_corr": np.nan,
            f"{prefix}_close_position_mean": np.nan,
            f"{prefix}_above_ma20_ratio": np.nan,
        }
    close = segment["close"].astype(float)
    cumulative = close / close.iloc[0]
    drawdown = cumulative / cumulative.cummax() - 1
    volume_price_corr = segment["return_pct"].corr(segment["volume_ratio_20"])
    return {
        f"{prefix}_return": float(close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] else 0.0,
        f"{prefix}_max_drawdown": float(drawdown.min()),
        f"{prefix}_volatility": float(segment["return_pct"].std(ddof=0)),
        f"{prefix}_range_mean": float(segment["range_pct"].mean()),
        f"{prefix}_up_day_ratio": float(segment["is_up_day"].mean()),
        f"{prefix}_long_upper_count": int(segment["is_long_upper_shadow"].sum()),
        f"{prefix}_long_lower_count": int(segment["is_long_lower_shadow"].sum()),
        f"{prefix}_high_volume_count": int(segment["is_high_volume"].sum()),
        f"{prefix}_volume_trend": float(segment["volume_trend_5"].mean()),
        f"{prefix}_volume_price_corr": float(volume_price_corr) if pd.notna(volume_price_corr) else 0.0,
        f"{prefix}_close_position_mean": float(segment["close_position"].mean()),
        f"{prefix}_above_ma20_ratio": float((segment["close"] > segment["ma20"]).mean()),
    }


def _entry_features(row: pd.Series) -> Dict[str, float | bool]:
    return {
        "entry_return_pct": float(row.get("return_pct", 0.0)),
        "entry_gap_pct": float(row.get("gap_pct", 0.0)),
        "entry_body_pct": float(row.get("body_pct", 0.0)),
        "entry_upper_shadow_ratio": float(row.get("upper_shadow_ratio", 0.0)),
        "entry_lower_shadow_ratio": float(row.get("lower_shadow_ratio", 0.0)),
        "entry_close_position": float(row.get("close_position", 0.5)),
        "entry_volume_ratio_20": float(row.get("volume_ratio_20", 0.0)),
        "entry_breakout_20d_high": bool(row.get("breakout_20d_high", False)),
        "entry_distance_to_ma20": float(row.get("ma20_distance", 0.0)),
        "entry_distance_to_60d_high": float(row.get("distance_to_60d_high", 0.0)),
    }


def _review_features(holding: pd.DataFrame, post: pd.DataFrame, entry_price: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not holding.empty and entry_price > 0:
        result["hold_max_return"] = float(holding["high"].max() / entry_price - 1)
        result["hold_min_return"] = float(holding["low"].min() / entry_price - 1)
        result["hold_first_3d_return"] = _forward_return(holding, entry_price, 3)
        result["hold_first_5d_return"] = _forward_return(holding, entry_price, 5)
        result["hold_first_10d_return"] = _forward_return(holding, entry_price, 10)
        high_returns = holding["high"] / entry_price - 1
        peak_idx = high_returns.idxmax()
        after_peak = holding.loc[peak_idx:]
        result["hold_drawdown_after_peak"] = float(after_peak["low"].min() / holding.loc[peak_idx, "high"] - 1) if not after_peak.empty else 0.0
    else:
        result.update({
            "hold_max_return": np.nan,
            "hold_min_return": np.nan,
            "hold_first_3d_return": np.nan,
            "hold_first_5d_return": np.nan,
            "hold_first_10d_return": np.nan,
            "hold_drawdown_after_peak": np.nan,
        })
    result["post_5_return"] = _period_return(post.head(5))
    result["post_10_return"] = _period_return(post.head(10))
    if not holding.empty and entry_price > 0:
        exit_close = float(holding.iloc[-1]["close"])
        exit_return = exit_close / entry_price - 1 if exit_close > 0 else np.nan
    else:
        exit_return = np.nan
    for days in [3, 5, 10, 15, 20]:
        segment = post.head(days)
        if segment.empty or entry_price <= 0:
            result[f"post_{days}d_best_high_from_entry"] = np.nan
            result[f"post_{days}d_best_close_from_entry"] = np.nan
            result[f"post_{days}d_close_from_entry"] = np.nan
            result[f"post_{days}d_best_high_improvement"] = np.nan
            result[f"post_{days}d_best_close_improvement"] = np.nan
            result[f"post_{days}d_close_improvement"] = np.nan
            result[f"post_{days}d_first_close_profit_day"] = np.nan
            result[f"post_{days}d_first_high_profit_day"] = np.nan
            result[f"post_{days}d_first_close_improve_2pct_day"] = np.nan
            continue
        best_high = float(segment["high"].max() / entry_price - 1)
        best_close = float(segment["close"].max() / entry_price - 1)
        close_ret = float(segment.iloc[-1]["close"] / entry_price - 1)
        result[f"post_{days}d_best_high_from_entry"] = best_high
        result[f"post_{days}d_best_close_from_entry"] = best_close
        result[f"post_{days}d_close_from_entry"] = close_ret
        result[f"post_{days}d_best_high_improvement"] = best_high - exit_return if pd.notna(exit_return) else np.nan
        result[f"post_{days}d_best_close_improvement"] = best_close - exit_return if pd.notna(exit_return) else np.nan
        result[f"post_{days}d_close_improvement"] = close_ret - exit_return if pd.notna(exit_return) else np.nan
        close_returns = segment["close"] / entry_price - 1
        high_returns = segment["high"] / entry_price - 1
        close_improvements = close_returns - exit_return if pd.notna(exit_return) else pd.Series(np.nan, index=segment.index)
        result[f"post_{days}d_first_close_profit_day"] = _first_match_day(close_returns > 0.0)
        result[f"post_{days}d_first_high_profit_day"] = _first_match_day(high_returns > 0.0)
        result[f"post_{days}d_first_close_improve_2pct_day"] = _first_match_day(close_improvements >= 0.02)
    return result


def _date_index(dates: pd.DatetimeIndex, value: object, side: str) -> int | None:
    ts = pd.Timestamp(value)
    if dates.empty:
        return None
    if side == "left":
        idx = dates.searchsorted(ts, side="left")
        return int(idx) if idx < len(dates) else None
    idx = dates.searchsorted(ts, side="right") - 1
    return int(idx) if idx >= 0 else None


def _preload_range(trades: pd.DataFrame, config: PatternAnalysisConfig) -> Tuple[str | None, str | None]:
    if trades.empty:
        return None, None
    start = pd.to_datetime(trades["entry_date"]).min() - pd.Timedelta(days=max(config.pre_n * 3, 90))
    end = pd.to_datetime(trades["exit_date"]).max() + pd.Timedelta(days=max(config.post_n * 3, 30))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _excursions(holding: pd.DataFrame, entry_price: float) -> Tuple[float, float]:
    if holding.empty or entry_price <= 0:
        return 0.0, 0.0
    return float(holding["high"].max() / entry_price - 1), float(holding["low"].min() / entry_price - 1)


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(0.0)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    first = values[0]
    if abs(first) < 1e-12:
        return 0.0
    return float((values[-1] - first) / abs(first))


def _forward_return(holding: pd.DataFrame, entry_price: float, days: int) -> float:
    if holding.empty or entry_price <= 0:
        return np.nan
    idx = min(days - 1, len(holding) - 1)
    return float(holding.iloc[idx]["close"] / entry_price - 1)


def _period_return(segment: pd.DataFrame) -> float:
    if len(segment) < 2:
        return np.nan
    return float(segment.iloc[-1]["close"] / segment.iloc[0]["close"] - 1)


def _first_match_day(mask: pd.Series) -> float:
    if mask.empty:
        return np.nan
    values = mask.fillna(False).astype(bool).to_numpy()
    matches = np.flatnonzero(values)
    if len(matches) == 0:
        return np.nan
    return float(matches[0] + 1)
