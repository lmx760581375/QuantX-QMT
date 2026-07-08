"""Matrix operators used by the dynamic factor runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # Optional acceleration; tests and basic runtime work without it.
    import bottleneck as bn
except Exception:  # pragma: no cover - depends on local environment
    bn = None


def ensure_float(x):
    return np.asarray(x, dtype=np.float32)


def ref(x, n: int):
    arr = np.asarray(x)
    n = int(n)
    if n == 0:
        return arr.copy()
    if arr.dtype == bool:
        out = np.zeros(arr.shape, dtype=bool)
    else:
        out = np.full_like(arr, np.nan, dtype=np.float32)
    if n > 0:
        out[n:] = arr[:-n]
    else:
        out[:n] = arr[-n:]
    return out


def rolling_min(x, window: int):
    arr = ensure_float(x)
    if bn is not None:
        return bn.move_min(arr, window=int(window), min_count=1, axis=0).astype(np.float32)
    return _pandas_rolling(arr, int(window), "min")


def rolling_max(x, window: int):
    arr = ensure_float(x)
    if bn is not None:
        return bn.move_max(arr, window=int(window), min_count=1, axis=0).astype(np.float32)
    return _pandas_rolling(arr, int(window), "max")


def rolling_mean(x, window: int):
    arr = ensure_float(x)
    if bn is not None:
        return bn.move_mean(arr, window=int(window), min_count=1, axis=0).astype(np.float32)
    return _pandas_rolling(arr, int(window), "mean")


def atr(high, low, close, window: int):
    hi = ensure_float(high)
    lo = ensure_float(low)
    cls = ensure_float(close)
    prev_close = ref(cls, 1)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_close), np.abs(lo - prev_close)))
    first = np.isnan(prev_close)
    tr = np.where(first, hi - lo, tr).astype(np.float32)
    return rolling_mean(tr, int(window))


def supertrend(high, low, close, window: int = 10, multiplier: float = 3.0, output: str = "line"):
    """Standard ATR SuperTrend line or direction.

    Direction is ``1`` for uptrend and ``-1`` for downtrend. The line follows
    the lower ATR band in uptrends and the upper ATR band in downtrends.
    """
    hi = ensure_float(high)
    lo = ensure_float(low)
    cls = ensure_float(close)
    if cls.ndim == 1:
        hi = hi[:, None]
        lo = lo[:, None]
        cls = cls[:, None]
        squeeze = True
    else:
        squeeze = False

    n = int(window)
    if n <= 0:
        raise ValueError("window must be positive")
    m = np.float32(float(multiplier))
    atr_values = atr(hi, lo, cls, n)
    mid = (hi + lo) / np.float32(2.0)
    basic_upper = mid + m * atr_values
    basic_lower = mid - m * atr_values
    rows, cols = cls.shape
    final_upper = np.full((rows, cols), np.nan, dtype=np.float32)
    final_lower = np.full((rows, cols), np.nan, dtype=np.float32)
    line = np.full((rows, cols), np.nan, dtype=np.float32)
    direction = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        valid = np.isfinite(cls[i]) & np.isfinite(basic_upper[i]) & np.isfinite(basic_lower[i])
        if not valid.any():
            if i > 0:
                direction[i] = direction[i - 1]
            continue

        if i == 0:
            final_upper[i, valid] = basic_upper[i, valid]
            final_lower[i, valid] = basic_lower[i, valid]
            direction[i, valid] = np.where(cls[i, valid] >= mid[i, valid], 1.0, -1.0)
        else:
            prev_fu = final_upper[i - 1]
            prev_fl = final_lower[i - 1]
            prev_close = cls[i - 1]
            prev_dir = direction[i - 1]
            reset_upper = np.isnan(prev_fu) | (basic_upper[i] < prev_fu) | (prev_close > prev_fu)
            reset_lower = np.isnan(prev_fl) | (basic_lower[i] > prev_fl) | (prev_close < prev_fl)
            final_upper[i, valid] = np.where(reset_upper[valid], basic_upper[i, valid], prev_fu[valid])
            final_lower[i, valid] = np.where(reset_lower[valid], basic_lower[i, valid], prev_fl[valid])

            direction[i] = prev_dir
            unset = valid & (direction[i] == 0.0)
            direction[i, unset] = np.where(cls[i, unset] >= mid[i, unset], 1.0, -1.0)
            flip_up = valid & (cls[i] > final_upper[i - 1])
            flip_down = valid & (cls[i] < final_lower[i - 1])
            direction[i, flip_up] = 1.0
            direction[i, flip_down] = -1.0

        up = valid & (direction[i] >= 0.0)
        down = valid & (direction[i] < 0.0)
        line[i, up] = final_lower[i, up]
        line[i, down] = final_upper[i, down]

    output_name = str(output or "line").lower()
    if output_name in {"line", "trend", "st"}:
        out = line
    elif output_name in {"direction", "dir"}:
        out = direction
    elif output_name in {"upper", "up"}:
        out = final_upper
    elif output_name in {"lower", "dn", "down"}:
        out = final_lower
    else:
        raise ValueError("output must be one of: line, direction, upper, lower")
    return out[:, 0] if squeeze else out


def rolling_sum(x, window: int):
    arr = ensure_float(x)
    if bn is not None:
        return bn.move_sum(arr, window=int(window), min_count=1, axis=0).astype(np.float32)
    return _pandas_rolling(arr, int(window), "sum")


def rolling_quantile(x, window: int, q: float):
    arr = ensure_float(x)
    frame = pd.DataFrame(arr)
    return frame.rolling(int(window), min_periods=1).quantile(float(q)).to_numpy(dtype=np.float32)


def bbi_uptrend(bbi, min_window: int, max_window: int, q_threshold: float = 0.0):
    """StockTradebyZ-style BBI derivative uptrend search.

    The original selector describes ``q_threshold`` as the tolerated share of
    pullback days. Counting negative daily BBI differences keeps that meaning
    while avoiding hundreds of expensive rolling quantile passes.
    """
    arr = ensure_float(bbi)
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    else:
        squeeze = False

    min_window = int(min_window)
    max_window = int(max_window)
    q_threshold = float(q_threshold)
    if min_window < 2:
        raise ValueError("min_window must be at least 2")
    if max_window < min_window:
        raise ValueError("max_window must be greater than or equal to min_window")
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold must be in [0, 1]")

    def trailing_count(mask: np.ndarray, window: int) -> np.ndarray:
        csum = np.vstack([
            np.zeros((1, mask.shape[1]), dtype=np.int32),
            np.cumsum(mask.astype(np.int32), axis=0, dtype=np.int32),
        ])
        out_count = np.zeros(mask.shape, dtype=np.int32)
        out_count[window - 1 :] = csum[window:] - csum[:-window]
        return out_count

    rows, cols = arr.shape
    out = np.zeros((rows, cols), dtype=bool)
    diffs = arr - ref(arr, 1)
    negative = (~np.isnan(diffs)) & (diffs < 0.0)
    valid_diff = ~np.isnan(diffs)
    valid_price = ~np.isnan(arr)
    for window in range(min(max_window, rows), min_window - 1, -1):
        diff_window = window - 1
        neg_count = trailing_count(negative, diff_window)
        diff_count = trailing_count(valid_diff, diff_window)
        price_count = trailing_count(valid_price, window)
        allowed_negative = int(np.floor(float(q_threshold) * float(diff_window)))
        out |= (price_count >= window) & (diff_count >= diff_window) & (neg_count <= allowed_negative)
    return out[:, 0] if squeeze else out


def expanding_quantile(x, q: float):
    arr = ensure_float(x)
    frame = pd.DataFrame(arr)
    return frame.expanding(min_periods=1).quantile(float(q)).to_numpy(dtype=np.float32)


def _pandas_rolling(arr: np.ndarray, window: int, method: str):
    frame = pd.DataFrame(arr)
    return getattr(frame.rolling(window, min_periods=1), method)().to_numpy(dtype=np.float32)


def ema(x, span: int):
    arr = ensure_float(x)
    alpha = np.float32(2.0 / (float(span) + 1.0))
    return _ewm_alpha(arr, alpha)


def tdx_sma(x, n: int, m: int = 1):
    arr = ensure_float(x)
    alpha = np.float32(float(m) / float(n))
    return _ewm_alpha(arr, alpha)


def kdj_j(high, low, close, n: int = 9):
    """KDJ J line with K/D initialized to 50, matching myquant-clean Selector.py."""
    hi = ensure_float(high)
    lo = ensure_float(low)
    cls = ensure_float(close)
    if cls.ndim == 1:
        hi = hi[:, None]
        lo = lo[:, None]
        cls = cls[:, None]
        squeeze = True
    else:
        squeeze = False

    low_n = rolling_min(lo, int(n))
    high_n = rolling_max(hi, int(n))
    rsv = (cls - low_n) / (high_n - low_n + np.float32(1e-9)) * np.float32(100.0)
    rows, cols = rsv.shape
    k = np.full((rows, cols), np.nan, dtype=np.float32)
    d = np.full((rows, cols), np.nan, dtype=np.float32)
    prev_k = np.full(cols, np.float32(50.0), dtype=np.float32)
    prev_d = np.full(cols, np.float32(50.0), dtype=np.float32)
    for i in range(rows):
        cur = rsv[i]
        valid = ~np.isnan(cur)
        prev_k[valid] = np.float32(2.0 / 3.0) * prev_k[valid] + np.float32(1.0 / 3.0) * cur[valid]
        prev_d[valid] = np.float32(2.0 / 3.0) * prev_d[valid] + np.float32(1.0 / 3.0) * prev_k[valid]
        k[i, valid] = prev_k[valid]
        d[i, valid] = prev_d[valid]
    j = np.float32(3.0) * k - np.float32(2.0) * d
    return j[:, 0] if squeeze else j


def _ewm_alpha(arr: np.ndarray, alpha: np.float32):
    if arr.ndim == 1:
        arr2 = arr[:, None]
        squeeze = True
    else:
        arr2 = arr
        squeeze = False
    out = np.full(arr2.shape, np.nan, dtype=np.float32)
    prev = np.full(arr2.shape[1], np.nan, dtype=np.float32)
    for i in range(arr2.shape[0]):
        cur = arr2[i].astype(np.float32, copy=False)
        init = np.isnan(prev) & ~np.isnan(cur)
        prev[init] = cur[init]
        valid = ~np.isnan(cur) & ~init
        prev[valid] = alpha * cur[valid] + (1.0 - alpha) * prev[valid]
        out[i] = prev
    return out[:, 0] if squeeze else out


def cross(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    prev_b = ref(bb, 1) if bb.ndim else bb
    return (aa > bb) & (ref(aa, 1) <= prev_b)


def filter_signal(cond, n: int):
    arr = np.asarray(cond, dtype=bool)
    if arr.ndim == 1:
        arr2 = arr[:, None]
        squeeze = True
    else:
        arr2 = arr
        squeeze = False
    out = np.zeros_like(arr2, dtype=bool)
    ignore_until = np.full(arr2.shape[1], -1, dtype=np.int64)
    n = int(n)
    for i in range(arr2.shape[0]):
        fire = arr2[i] & (i > ignore_until)
        out[i, fire] = True
        ignore_until[fire] = i + n
    return out[:, 0] if squeeze else out


def cs_mean(x):
    arr = np.asarray(x)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr.astype(np.float32), axis=1).astype(np.float32)


def cs_count(x):
    arr = np.asarray(x)
    if arr.dtype == bool:
        return np.sum(arr, axis=1).astype(np.float32)
    return np.sum(~np.isnan(arr), axis=1).astype(np.float32)


def cs_rank(x, pct: bool = False):
    frame = pd.DataFrame(np.asarray(x, dtype=np.float32))
    ranked = frame.rank(axis=1, method="average", pct=pct)
    return ranked.to_numpy(dtype=np.float32)


def topk(x, k: int):
    arr = np.asarray(x, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=bool)
    k = int(k)
    if k <= 0:
        return out
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.flatnonzero(~np.isnan(row))
        if valid.size == 0:
            continue
        take = valid[np.argsort(row[valid])[-min(k, valid.size):]]
        out[i, take] = True
    return out


def group_mean(x, group, agg: str = "mean"):
    arr = ensure_float(x)
    raw_group = np.asarray(group)
    if arr.ndim != 2:
        raise ValueError("GroupMean expects x shape [date, instrument]")
    if raw_group.ndim == 2:
        return _membership_group_mean(arr, raw_group, agg=agg)
    codes = raw_group.astype(np.int32)
    if codes.ndim != 1 or codes.shape[0] != arr.shape[1]:
        raise ValueError("GroupMean expects group shape [instrument] or [instrument, group]")
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid_group = codes >= 0
    if not valid_group.any():
        return out
    group_count = int(np.nanmax(codes[valid_group])) + 1
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = valid_group & np.isfinite(row)
        if not valid.any():
            continue
        sums = np.bincount(codes[valid], weights=row[valid], minlength=group_count)
        counts = np.bincount(codes[valid], minlength=group_count)
        means = np.divide(sums, counts, out=np.full(group_count, np.nan), where=counts > 0)
        out[i, valid_group] = means[codes[valid_group]]
    return out.astype(np.float32)


def group_rank(x, group=None, ascending: bool = False, pct: bool = True, agg: str = "mean"):
    arr = ensure_float(x)
    if arr.ndim != 2:
        raise ValueError("GroupRank expects x shape [date, instrument]")
    if group is None:
        return cs_rank(arr if ascending else -arr, pct=pct)
    raw_group = np.asarray(group)
    if raw_group.ndim == 2:
        group_values = _membership_group_values(arr, raw_group)
        group_ranks = _rank_group_values(group_values, ascending=ascending, pct=pct)
        return _broadcast_membership_values(group_ranks, raw_group, agg=agg)
    codes = raw_group.astype(np.int32)
    if codes.ndim != 1 or codes.shape[0] != arr.shape[1]:
        raise ValueError("GroupRank expects group shape [instrument]")
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid_group = codes >= 0
    if not valid_group.any():
        return out
    group_count = int(np.nanmax(codes[valid_group])) + 1
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = valid_group & np.isfinite(row)
        if not valid.any():
            continue
        sums = np.bincount(codes[valid], weights=row[valid], minlength=group_count)
        counts = np.bincount(codes[valid], minlength=group_count)
        means = np.divide(sums, counts, out=np.full(group_count, np.nan), where=counts > 0)
        valid_codes = np.flatnonzero(np.isfinite(means))
        if valid_codes.size == 0:
            continue
        order_values = means[valid_codes] if ascending else -means[valid_codes]
        ranks = pd.Series(order_values).rank(method="average", pct=bool(pct)).to_numpy(dtype=np.float32)
        group_ranks = np.full(group_count, np.nan, dtype=np.float32)
        group_ranks[valid_codes] = ranks
        out[i, valid_group] = group_ranks[codes[valid_group]]
    return out


def _membership_group_mean(arr: np.ndarray, membership, agg: str = "mean") -> np.ndarray:
    group_values = _membership_group_values(arr, membership)
    return _broadcast_membership_values(group_values, membership, agg=agg)


def _membership_group_values(arr: np.ndarray, membership) -> np.ndarray:
    members = np.asarray(membership, dtype=bool)
    if members.ndim != 2 or members.shape[0] != arr.shape[1]:
        raise ValueError("membership group must have shape [instrument, group]")
    if members.shape[1] == 0:
        return np.full((arr.shape[0], 0), np.nan, dtype=np.float32)
    valid = np.isfinite(arr).astype(np.float32)
    values = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    weights = members.astype(np.float32)
    sums = values @ weights
    counts = valid @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    out[counts <= 0] = np.nan
    return out.astype(np.float32)


def _broadcast_membership_values(group_values: np.ndarray, membership, agg: str = "mean") -> np.ndarray:
    members = np.asarray(membership, dtype=bool)
    agg_name = str(agg or "mean").lower()
    rows, stock_count = group_values.shape[0], members.shape[0]
    out = np.full((rows, stock_count), np.nan, dtype=np.float32)
    for stock_idx in range(stock_count):
        group_idx = np.flatnonzero(members[stock_idx])
        if group_idx.size == 0:
            continue
        vals = group_values[:, group_idx]
        with np.errstate(invalid="ignore"):
            if agg_name == "mean":
                out[:, stock_idx] = np.nanmean(vals, axis=1)
            elif agg_name == "max":
                out[:, stock_idx] = np.nanmax(vals, axis=1)
            elif agg_name == "min":
                out[:, stock_idx] = np.nanmin(vals, axis=1)
            else:
                raise ValueError("membership agg must be one of: mean, max, min")
    return out.astype(np.float32)


def _rank_group_values(group_values: np.ndarray, ascending: bool = False, pct: bool = True) -> np.ndarray:
    if group_values.shape[1] == 0:
        return group_values.astype(np.float32)
    ranked = pd.DataFrame(group_values if ascending else -group_values).rank(axis=1, method="average", pct=pct)
    return ranked.to_numpy(dtype=np.float32)


def max_vol_not_bearish(volume, open_, close, window: int):
    """Return true when the max-volume day in the trailing window is not bearish."""
    vol = ensure_float(volume)
    opn = ensure_float(open_)
    cls = ensure_float(close)
    n = max(1, int(window))
    if vol.ndim == 1:
        vol = vol[:, None]
        opn = opn[:, None]
        cls = cls[:, None]
        squeeze = True
    else:
        squeeze = False

    rows, cols = vol.shape
    max_vol = np.full((rows, cols), np.nan, dtype=np.float32)
    out = np.zeros((rows, cols), dtype=bool)
    bullish = cls >= opn
    for lag in range(n):
        if lag >= rows:
            break
        candidate = vol[: rows - lag]
        target = max_vol[lag:]
        valid = ~np.isnan(candidate)
        better = valid & (np.isnan(target) | (candidate > target))
        target[better] = candidate[better]
        out_slice = out[lag:]
        bull_slice = bullish[: rows - lag]
        out_slice[better] = bull_slice[better]
    return out[:, 0] if squeeze else out


def prior_consecutive(cond):
    """Count consecutive true values ending at the previous row."""
    arr = np.asarray(cond, dtype=bool)
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    else:
        squeeze = False
    out = np.zeros(arr.shape, dtype=np.float32)
    for i in range(1, arr.shape[0]):
        out[i] = np.where(arr[i - 1], out[i - 1] + 1.0, 0.0)
    return out[:, 0] if squeeze else out


def recent_stable_signal(signal, close, lookback: int, max_range_pct: float, min_segment: int = 3):
    """True when a prior signal day exists and close stayed stable until yesterday.

    This matches StockTradebyZ SuperB1's search over t_m in the recent lookback:
    find a signal day before today where close[t_m: yesterday] has max/min - 1
    within max_range_pct.
    """
    sig = np.asarray(signal, dtype=bool)
    cls = ensure_float(close)
    if cls.ndim == 1:
        sig = sig[:, None]
        cls = cls[:, None]
        squeeze = True
    else:
        squeeze = False

    lookback = int(lookback)
    min_segment = int(min_segment)
    max_range_pct = float(max_range_pct)
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if min_segment < 1:
        raise ValueError("min_segment must be positive")

    rows, cols = cls.shape
    out = np.zeros((rows, cols), dtype=bool)
    for lag in range(1, lookback + 1):
        tm_signal = ref(sig, lag)
        segment_len = lag
        if segment_len < min_segment:
            continue
        prev_close = ref(cls, 1)
        high = rolling_max(prev_close, segment_len)
        low = rolling_min(prev_close, segment_len)
        stable = low > 0.0
        stable &= (high / (low + np.float32(1e-10)) - np.float32(1.0)) <= np.float32(max_range_pct)
        out |= tm_signal & stable
    return out[:, 0] if squeeze else out


def brick_chart(
    high,
    low,
    close,
    n: int = 8,
    m1: int = 3,
    m2: int = 12,
    m3: int = 12,
    t: float = 8.0,
    shift1: float = 92.0,
    shift2: float = 114.0,
    sma_w1: int = 1,
    sma_w2: int = 1,
    sma_w3: int = 1,
):
    """StockTradebyZ/TDX-style brick chart height, vectorized across instruments."""
    hi = ensure_float(high)
    lo = ensure_float(low)
    cls = ensure_float(close)
    if cls.ndim == 1:
        hi = hi[:, None]
        lo = lo[:, None]
        cls = cls[:, None]
        squeeze = True
    else:
        squeeze = False

    hhv = rolling_max(hi, int(n))
    llv = rolling_min(lo, int(n))
    rng = hhv - llv
    rng = np.where(rng == 0.0, 0.01, rng).astype(np.float32)

    a1 = np.float32(float(sma_w1) / float(m1))
    b1 = np.float32(1.0) - a1
    v1 = (hhv - cls) / rng * np.float32(100.0) - np.float32(shift1)
    var2a = np.empty_like(v1, dtype=np.float32)
    var2a[0] = v1[0] + np.float32(shift2)
    for i in range(1, cls.shape[0]):
        var2a[i] = a1 * v1[i] + b1 * (var2a[i - 1] - np.float32(shift2)) + np.float32(shift2)

    a2 = np.float32(float(sma_w2) / float(m2))
    b2 = np.float32(1.0) - a2
    a3 = np.float32(float(sma_w3) / float(m3))
    b3 = np.float32(1.0) - a3
    v3 = (cls - llv) / rng * np.float32(100.0)
    var4a = np.empty_like(v3, dtype=np.float32)
    var5a = np.empty_like(v3, dtype=np.float32)
    var4a[0] = v3[0]
    var5a[0] = v3[0] + np.float32(shift2)
    for i in range(1, cls.shape[0]):
        var4a[i] = a2 * v3[i] + b2 * var4a[i - 1]
        var5a[i] = a3 * var4a[i] + b3 * (var5a[i - 1] - np.float32(shift2)) + np.float32(shift2)

    diff = var5a - var2a
    raw = np.where(diff > float(t), diff - np.float32(t), np.float32(0.0)).astype(np.float32)
    brick = np.zeros_like(raw, dtype=np.float32)
    brick[1:] = raw[1:] - raw[:-1]
    return brick[:, 0] if squeeze else brick
