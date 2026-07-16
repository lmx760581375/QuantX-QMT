from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
G60_PATH = ROOT / "train_group_diffusion_world_model_v1.py"
INDUSTRY_CSV = REPO_ROOT / "data/meta/snapshots/industry_membership.csv"
OUT = ROOT / "coal_liquidity_cycle_probe_v1_summary.json"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G60 = load_module("group_diffusion_for_coal_probe", G60_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def roll_mean(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window, min_periods=max(5, window // 3)).mean().to_numpy(dtype=float)


def pct_rank_last(series: np.ndarray, lookback: int) -> float:
    clean = np.asarray(series, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 0.0
    tail = clean[-lookback:] if len(clean) > lookback else clean
    return float(np.mean(tail <= clean[-1]))


def symbol_to_keys(symbol: str) -> set[str]:
    raw = str(symbol).strip().upper()
    digits = "".join(ch for ch in raw if ch.isdigit())
    keys = {raw}
    if len(digits) == 6:
        suffix = "SH" if digits.startswith("6") else "SZ"
        keys.update({digits, f"{suffix}{digits}", f"{digits}.{suffix}"})
    return keys


def coal_columns(symbols: list[str]) -> tuple[list[int], list[dict[str, str]]]:
    lookup: dict[str, int] = {}
    for i, symbol in enumerate(symbols):
        for key in symbol_to_keys(symbol):
            lookup[key] = i
    frame = pd.read_csv(INDUSTRY_CSV, dtype=str)
    coal = frame[(frame["industry_code"] == "BK0437") | (frame["industry_name"].fillna("").str.contains("煤炭"))]
    cols = []
    members = []
    for _, row in coal.iterrows():
        symbol = str(row.get("symbol") or row.get("source_code"))
        col = None
        for key in symbol_to_keys(symbol):
            if key in lookup:
                col = lookup[key]
                break
        if col is not None:
            cols.append(col)
            members.append({"symbol": str(row.get("symbol") or ""), "name": str(row.get("name") or ""), "source_code": str(row.get("source_code") or "")})
    unique = sorted(set(cols))
    return unique, members


def main() -> None:
    market = G60.POS.load_market()
    dates = np.asarray(market["dates"])
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    vwap = arrays["vwap"].astype(float)
    amount = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close) * volume
    amount = np.where(np.isfinite(amount) & (amount > 0), amount, np.nan)
    coal_cols, members = coal_columns(list(market["symbols"]))
    coal_amount = np.nansum(amount[:, coal_cols], axis=1)
    market_amount = np.nansum(amount, axis=1)
    share = safe_div(coal_amount, market_amount)

    coal_close = np.nanmean(close[:, coal_cols], axis=1)
    amount20 = roll_mean(coal_amount, 20)
    amount60 = roll_mean(coal_amount, 60)
    amount120 = roll_mean(coal_amount, 120)
    share20 = roll_mean(share, 20)
    share120 = roll_mean(share, 120)
    close20 = roll_mean(coal_close, 20)
    close120 = roll_mean(coal_close, 120)

    member_amount20 = np.vstack([roll_mean(amount[:, col], 20) for col in coal_cols]).T
    member_amount120 = np.vstack([roll_mean(amount[:, col], 120) for col in coal_cols]).T
    member_liq_ratio = safe_div(member_amount20, member_amount120)
    breadth_liq_gt_12 = np.nanmean(member_liq_ratio > 1.2, axis=1)
    breadth_liq_gt_15 = np.nanmean(member_liq_ratio > 1.5, axis=1)

    ret20 = safe_div(coal_close, np.r_[np.full(20, np.nan), coal_close[:-20]]) - 1.0
    ret60 = safe_div(coal_close, np.r_[np.full(60, np.nan), coal_close[:-60]]) - 1.0

    last = len(dates) - 1
    snapshots = {}
    for label, offset in [("latest", 0), ("20d_ago", 20), ("60d_ago", 60), ("120d_ago", 120), ("240d_ago", 240)]:
        i = max(0, last - offset)
        snapshots[label] = {
            "date": str(dates[i]),
            "coal_amount20": float(amount20[i]) if np.isfinite(amount20[i]) else None,
            "coal_amount20_vs_120": float(amount20[i] / amount120[i]) if np.isfinite(amount20[i]) and np.isfinite(amount120[i]) and amount120[i] else None,
            "coal_market_share20": float(share20[i]) if np.isfinite(share20[i]) else None,
            "share20_vs_120": float(share20[i] / share120[i]) if np.isfinite(share20[i]) and np.isfinite(share120[i]) and share120[i] else None,
            "liq_breadth_gt_1p2": float(breadth_liq_gt_12[i]) if np.isfinite(breadth_liq_gt_12[i]) else None,
            "liq_breadth_gt_1p5": float(breadth_liq_gt_15[i]) if np.isfinite(breadth_liq_gt_15[i]) else None,
            "ret20": float(ret20[i]) if np.isfinite(ret20[i]) else None,
            "ret60": float(ret60[i]) if np.isfinite(ret60[i]) else None,
            "close_vs_ma120": float(coal_close[i] / close120[i] - 1.0) if np.isfinite(coal_close[i]) and np.isfinite(close120[i]) and close120[i] else None,
        }

    years = {}
    date_index = pd.to_datetime(dates)
    for year, idxs in pd.Series(np.arange(len(dates)), index=date_index).groupby(date_index.year):
        ix = idxs.to_numpy(dtype=int)
        years[str(year)] = {
            "sessions": int(len(ix)),
            "amount20_mean": float(np.nanmean(amount20[ix])),
            "market_share20_mean": float(np.nanmean(share20[ix])),
            "liq_breadth_gt_1p2_mean": float(np.nanmean(breadth_liq_gt_12[ix])),
            "ret_year_equal_weight_proxy": float(coal_close[ix[-1]] / coal_close[ix[0]] - 1.0) if np.isfinite(coal_close[ix[-1]]) and np.isfinite(coal_close[ix[0]]) and coal_close[ix[0]] else None,
        }

    latest_i = last
    verdict = "coal_liquidity_not_confirmed"
    if (
        np.isfinite(amount20[latest_i]) and np.isfinite(amount120[latest_i]) and amount120[latest_i] > 0
        and np.isfinite(share20[latest_i]) and np.isfinite(share120[latest_i]) and share120[latest_i] > 0
        and amount20[latest_i] / amount120[latest_i] > 1.15
        and share20[latest_i] / share120[latest_i] > 1.05
        and breadth_liq_gt_12[latest_i] > 0.35
    ):
        verdict = "coal_liquidity_cycle_reactivation_signal"

    out = {
        "experiment": "coal_liquidity_cycle_probe_v1",
        "scope": "daily OHLCV only, static coal industry membership BK0437, no news/fundamental data",
        "date_range": [str(dates[0]), str(dates[-1])],
        "coal_member_count": len(coal_cols),
        "members_preview": members[:40],
        "snapshots": snapshots,
        "latest_percentiles": {
            "amount20_pct_3y": pct_rank_last(amount20, 756),
            "market_share20_pct_3y": pct_rank_last(share20, 756),
            "liq_breadth_gt_1p2_pct_3y": pct_rank_last(breadth_liq_gt_12, 756),
            "ret60_pct_3y": pct_rank_last(ret60, 756),
        },
        "yearly": years,
        "verdict": verdict,
        "inputs_sha256": {"script": sha256(Path(__file__)), "industry_csv": sha256(INDUSTRY_CSV)},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "coal_member_count": len(coal_cols), "snapshots": snapshots, "latest_percentiles": out["latest_percentiles"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
