"""Formula operator registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from . import ops


@dataclass(frozen=True)
class Operator:
    name: str
    func: Callable
    kind: str = "elementwise"


class OperatorRegistry:
    def __init__(self):
        self._operators: Dict[str, Operator] = {}

    def register(self, name: str, func: Callable, kind: str = "elementwise") -> None:
        self._operators[name] = Operator(name=name, func=func, kind=kind)

    def get(self, name: str) -> Operator:
        if name not in self._operators:
            raise KeyError(f"Unknown formula operator: {name}")
        return self._operators[name]

    def has(self, name: str) -> bool:
        return name in self._operators

    def names(self):
        return sorted(self._operators)


DEFAULT_REGISTRY = OperatorRegistry()


def operator(name: str, kind: str = "elementwise"):
    def deco(func: Callable):
        DEFAULT_REGISTRY.register(name, func, kind)
        return func
    return deco


@operator("Ref", "time_series")
def _op_ref(x, n):
    return ops.ref(x, int(n))


@operator("Min", "time_series")
def _op_min(x, n):
    return ops.rolling_min(x, int(n))


@operator("Max", "time_series")
def _op_max(x, n):
    return ops.rolling_max(x, int(n))


@operator("Mean", "time_series")
def _op_mean(x, n):
    return ops.rolling_mean(x, int(n))


@operator("ATR", "time_series")
def _op_atr(high, low, close, n):
    return ops.atr(high, low, close, int(n))


@operator("SuperTrend", "time_series")
def _op_supertrend(high, low, close, n=10, multiplier=3.0, output="line"):
    return ops.supertrend(high, low, close, int(n), float(multiplier), str(output))


@operator("Sum", "time_series")
def _op_sum(x, n):
    return ops.rolling_sum(x, int(n))


@operator("RollingQuantile", "time_series")
def _op_rolling_quantile(x, n, q):
    return ops.rolling_quantile(x, int(n), float(q))


@operator("BBIUptrend", "time_series")
def _op_bbi_uptrend(bbi, min_window, max_window, q_threshold=0.0):
    return ops.bbi_uptrend(bbi, int(min_window), int(max_window), float(q_threshold))


@operator("ExpandingQuantile", "time_series")
def _op_expanding_quantile(x, q):
    return ops.expanding_quantile(x, float(q))


@operator("EMA", "time_series")
def _op_ema(x, span):
    return ops.ema(x, int(span))


@operator("SMA_TDX", "time_series")
def _op_sma_tdx(x, n, m=1):
    return ops.tdx_sma(x, int(n), int(m))


@operator("KDJJ", "time_series")
def _op_kdjj(high, low, close, n=9):
    return ops.kdj_j(high, low, close, int(n))


@operator("Cross", "time_series")
def _op_cross(a, b):
    return ops.cross(a, b)


@operator("Filter", "time_series")
def _op_filter(cond, n):
    return ops.filter_signal(cond, int(n))


@operator("Maximum")
def _op_maximum(a, b):
    return np.maximum(a, b)


@operator("Minimum")
def _op_minimum(a, b):
    return np.minimum(a, b)


@operator("Where")
def _op_where(cond, a, b):
    cond, a = _broadcast_pair(cond, a)
    cond, b = _broadcast_pair(cond, b)
    a, b = _broadcast_pair(a, b)
    return np.where(cond, a, b)


@operator("Clip")
def _op_clip(x, lower=None, upper=None):
    return np.clip(x, lower, upper)


@operator("FillNa")
def _op_fillna(x, value):
    arr = np.asarray(x)
    return np.where(np.isnan(arr), value, arr)


@operator("IsNa")
def _op_isna(x):
    return np.isnan(np.asarray(x))


@operator("NaN")
def _op_nan():
    return np.nan


@operator("Abs")
def _op_abs(x):
    return np.abs(x)


@operator("And")
def _op_and(*args):
    if not args:
        raise ValueError("And requires at least one argument")
    out = np.asarray(args[0], dtype=bool)
    for arg in args[1:]:
        out = np.logical_and(out, np.asarray(arg, dtype=bool))
    return out


@operator("Or")
def _op_or(*args):
    if not args:
        raise ValueError("Or requires at least one argument")
    out = np.asarray(args[0], dtype=bool)
    for arg in args[1:]:
        out = np.logical_or(out, np.asarray(arg, dtype=bool))
    return out


@operator("Not")
def _op_not(x):
    return np.logical_not(np.asarray(x, dtype=bool))


def _broadcast_pair(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.ndim == 2 and right_arr.ndim == 1 and left_arr.shape[0] == right_arr.shape[0]:
        return left, right_arr[:, None]
    if right_arr.ndim == 2 and left_arr.ndim == 1 and right_arr.shape[0] == left_arr.shape[0]:
        return left_arr[:, None], right
    return left, right


@operator("CSMean", "cross_section")
def _op_csmean(x):
    return ops.cs_mean(x)


@operator("CSCount", "cross_section")
def _op_cscount(x):
    return ops.cs_count(x)


@operator("CSRatio", "cross_section")
def _op_csratio(x):
    return ops.cs_mean(x)


@operator("CSRank", "cross_section")
def _op_csrank(x):
    return ops.cs_rank(x, pct=False)


@operator("CSPctRank", "cross_section")
def _op_cspctrank(x):
    return ops.cs_rank(x, pct=True)


@operator("TopK", "cross_section")
def _op_topk(x, k):
    return ops.topk(x, int(k))


@operator("GroupMean", "group")
def _op_group_mean(x, group, agg="mean"):
    return ops.group_mean(x, group, agg=str(agg))


@operator("GroupRatio", "group")
def _op_group_ratio(x, group, agg="mean"):
    return ops.group_mean(x, group, agg=str(agg))


@operator("GroupRank", "group")
def _op_group_rank(x, group=None, ascending=False, pct=True, agg="mean"):
    return ops.group_rank(x, group=group, ascending=bool(ascending), pct=bool(pct), agg=str(agg))


@operator("MaxVolNotBearish", "time_series")
def _op_max_vol_not_bearish(volume, open_, close, n):
    return ops.max_vol_not_bearish(volume, open_, close, int(n))


@operator("PriorConsecutive", "time_series")
def _op_prior_consecutive(cond):
    return ops.prior_consecutive(cond)


@operator("RecentStableSignal", "time_series")
def _op_recent_stable_signal(signal, close, lookback, max_range_pct, min_segment=3):
    return ops.recent_stable_signal(signal, close, int(lookback), float(max_range_pct), int(min_segment))


@operator("BrickChart", "time_series")
def _op_brick_chart(
    high,
    low,
    close,
    n=8,
    m1=3,
    m2=12,
    m3=12,
    t=8,
    shift1=92,
    shift2=114,
    sma_w1=1,
    sma_w2=1,
    sma_w3=1,
):
    return ops.brick_chart(
        high,
        low,
        close,
        int(n),
        int(m1),
        int(m2),
        int(m3),
        float(t),
        float(shift1),
        float(shift2),
        int(sma_w1),
        int(sma_w2),
        int(sma_w3),
    )


@operator("ShuijiaoMidline", "composite")
def _op_shuijiao_midline(close, high, low):
    h1 = np.maximum(high, ops.ref(close, 1))
    l1 = np.minimum(low, ops.ref(close, 1))
    p1 = h1 - l1
    resistance = l1 + p1 * 7.0 / 8.0
    support = l1 + p1 * 0.5 / 8.0
    return (support + resistance) / 2.0


@operator("ShuijiaoTrend", "composite")
def _op_shuijiao_trend(close, high, low, period=55):
    period = int(period)
    llv = ops.rolling_min(low, period)
    hhv = ops.rolling_max(high, period)
    rsv = (close - llv) / (hhv - llv + 1e-10) * 100.0
    sma1 = ops.tdx_sma(rsv, 5, 1)
    v11 = 3.0 * sma1 - 2.0 * ops.tdx_sma(sma1, 3, 1)
    return ops.ema(v11, 3)


@operator("ShuijiaoBuy", "composite")
def _op_shuijiao_buy(trend_line, close, midline):
    prev = ops.ref(trend_line, 1)
    bb1 = (prev < 11) & (prev > 6) & ops.cross(trend_line, 11)
    bb2 = (prev < 6) & (prev > 3) & ops.cross(trend_line, 6)
    bb3 = (prev < 3) & (prev > 1) & ops.cross(trend_line, 3)
    bb4 = (prev < 1) & (prev > 0) & ops.cross(trend_line, 1)
    bb5 = (prev < 0) & ops.cross(trend_line, 0)
    return (bb1 | bb2 | bb3 | bb4 | bb5) & (close < midline)


@operator("ShuijiaoReadySell", "composite")
def _op_shuijiao_ready_sell(trend_line, close, midline):
    return (trend_line > 89) & ops.filter_signal(trend_line > 89, 15) & (close > midline)
