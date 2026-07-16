from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_backtest import load_a_share_symbols

ROOT = Path(".tmp/quantx-research/elastic-cross-section-v1")
PROVIDER = Path("data/qlib_data_fixed")
META = Path("data/meta/quantx_meta.sqlite")
START = "2016-01-01"
TRAIN_END = "2024-12-31"
OOS_START = "2025-01-02"
END = "2026-07-15"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    write_progress("start", {"start": START, "train_end": TRAIN_END, "oos_start": OOS_START, "end": END})
    symbols = load_a_share_symbols(PROVIDER, START, END, universe="all_a")
    reader = QlibBinReader(PROVIDER)
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount"], START, END)
    quote = normalize_quote(quote)
    risk_names = load_current_risk_names()
    write_progress("quote_loaded", {"symbols": len(symbols), "rows": int(len(quote)), "risk_names": len(risk_names)})
    panel = build_panel(quote, risk_names)
    write_progress("panel_built", {"rows": int(len(panel)), "candidate_union": int(panel["candidate_union"].sum())})
    candidates = add_labels(panel.loc[panel["candidate_union"]].copy(), panel)
    candidates.to_parquet(ROOT / "elastic_candidates_with_labels.parquet", index=False)
    report = analyze(candidates)
    (ROOT / "elastic_candidate_soil_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def write_progress(step: str, payload: dict) -> None:
    row = {"ts": datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(), "elapsed_epoch": time.time(), "step": step, **payload}
    with (ROOT / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"progress": step, **payload}, ensure_ascii=False, default=str), flush=True)


def normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    if list(quote.index.names) != ["instrument", "datetime"]:
        quote = quote.reorder_levels(["instrument", "datetime"])
    quote = quote.sort_index()
    for col in quote.columns:
        quote[col] = pd.to_numeric(quote[col], errors="coerce").astype("float32")
    return quote


def load_current_risk_names() -> set[str]:
    risk: set[str] = set()
    if not META.exists():
        return risk
    with sqlite3.connect(META) as conn:
        for symbol, name in conn.execute("select symbol, name from security_master"):
            upper = str(name or "").upper()
            if "ST" in upper or "退" in upper:
                risk.add(normalize_symbol(str(symbol)))
    return risk


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value:
        code, exchange = value.split(".", 1)
        return f"{exchange}{code}"
    return value


def build_panel(quote: pd.DataFrame, risk_names: set[str]) -> pd.DataFrame:
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    grouped = frame.groupby("instrument", sort=False, group_keys=False)
    prev_close = grouped["$close"].shift(1)
    for w in [1, 2, 3, 5, 10, 20, 60]:
        frame[f"ret{w}"] = grouped["$close"].pct_change(w, fill_method=None)
    for w in [5, 10, 20, 50, 100]:
        ma = grouped["$close"].transform(lambda s, window=w: s.rolling(window, min_periods=max(3, window // 3)).mean())
        frame[f"ma{w}"] = ma
        frame[f"ma{w}_rel"] = frame["$close"] / ma - 1.0
    frame["amount20"] = grouped["$amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol20"] = grouped["$volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["vol_ratio"] = frame["$volume"] / frame["vol20"]
    frame["amplitude_pct"] = (frame["$high"] - frame["$low"]) / prev_close
    frame["body_pct"] = (frame["$close"] - frame["$open"]) / prev_close
    high_low = (frame["$high"] - frame["$low"]).replace(0, np.nan)
    frame["close_pos"] = (frame["$close"] - frame["$low"]) / high_low
    frame["open_gap_pct"] = frame["$open"] / prev_close - 1.0
    frame["max_ret1_20"] = grouped["ret1"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    frame["max_amp_20"] = grouped["amplitude_pct"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    frame["max_amount_ratio_20"] = grouped["vol_ratio"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    frame["pullback_5"] = frame["$close"] / grouped["$close"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).max()) - 1.0
    frame["drawdown_20"] = frame["$close"] / grouped["$close"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max()) - 1.0
    frame["rebound_3"] = frame["$close"] / grouped["$close"].shift(3) - 1.0
    inst = frame["instrument"].astype(str)
    frame["is_star"] = inst.str.startswith("SH688").astype(float)
    frame["is_chinext"] = (inst.str.startswith("SZ300") | inst.str.startswith("SZ301")).astype(float)
    frame["is_mainboard"] = ((frame["is_star"] == 0) & (frame["is_chinext"] == 0)).astype(float)
    frame["is_current_risk_name"] = frame["instrument"].isin(risk_names)
    abs_ret = grouped["$close"].pct_change(fill_method=None).abs()
    five = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) >= 3
    ) & (ten.groupby(frame["instrument"]).rolling(80, min_periods=1).sum().reset_index(level=0, drop=True) == 0)
    one_price_up = (
        (frame["$high"].sub(frame["$low"]).abs() <= 1e-6)
        & (frame["$close"].div(prev_close) - 1.0 >= 0.045)
    )
    for col in ["ret1", "ret3", "ret5", "ret20", "amount20", "$amount", "vol_ratio", "amplitude_pct", "body_pct", "close_pos", "ma20_rel", "ma50_rel", "max_ret1_20", "drawdown_20", "pullback_5", "rebound_3"]:
        frame[col + "_rank_pct"] = frame.groupby("datetime")[col].rank(pct=True)
    tradable_base = (
        (frame["$volume"] > 0)
        & (frame["$amount"] > 0)
        & (~frame["is_current_risk_name"].fillna(False))
        & (~frame["st_like_limit_history"].fillna(False))
        & (~one_price_up.fillna(False))
    )
    proved_elastic = (
        (frame["max_ret1_20"] >= 0.07)
        | (frame["max_amp_20"] >= 0.10)
        | ((frame["ret20_rank_pct"] >= 0.75) & (frame["amount20_rank_pct"] >= 0.55))
    )
    not_overheated = (
        (frame["ret1_rank_pct"] <= 0.70)
        & (frame["close_pos_rank_pct"] <= 0.75)
        & (frame["body_pct_rank_pct"] <= 0.75)
        & (frame["ma50_rel_rank_pct"] <= 0.75)
        & (frame["amplitude_pct"] <= 0.14)
    )
    repair = (
        ((frame["drawdown_20"] <= -0.04) & (frame["drawdown_20"] >= -0.28) & (frame["rebound_3"] >= -0.03))
        | ((frame["pullback_5"] <= -0.02) & (frame["ret1"] > -0.03))
        | ((frame["ret5_rank_pct"] <= 0.50) & (frame["ret20_rank_pct"] >= 0.60))
    )
    liquid = frame["amount20_rank_pct"] >= 0.35
    frame["candidate_elastic_repair"] = tradable_base & proved_elastic & not_overheated & repair & liquid
    frame["candidate_low_chase_strength"] = tradable_base & proved_elastic & not_overheated & (frame["ret20_rank_pct"] >= 0.65) & liquid
    frame["candidate_nonmain_repair"] = frame["candidate_elastic_repair"] & (frame["is_mainboard"] == 0)
    frame["candidate_union"] = frame["candidate_elastic_repair"] | frame["candidate_low_chase_strength"] | frame["candidate_nonmain_repair"]
    return frame


def add_labels(candidates: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    panel_by_symbol = {symbol: day.reset_index(drop=True) for symbol, day in panel.groupby("instrument", sort=False)}
    panel_pos = panel.groupby("instrument", sort=False).cumcount()
    candidates = candidates.copy()
    candidates["__panel_pos"] = panel_pos.loc[candidates.index].to_numpy(dtype="int64")
    rows = []
    for record in candidates.to_dict("records"):
        hist = panel_by_symbol[str(record["instrument"])]
        pos = int(record.pop("__panel_pos"))
        if pos < 0 or pos >= len(hist):
            continue
        record.update(dynamic_exit_label(hist, pos))
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["datetime"])
    out["year"] = out["date"].dt.year
    return out


def dynamic_exit_label(hist: pd.DataFrame, pos: int) -> dict:
    buy_idx = pos + 1
    if buy_idx >= len(hist):
        return empty_label()
    preclose = float(hist.loc[pos, "$close"])
    buy_open = float(hist.loc[buy_idx, "$open"])
    if not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(buy_open) or buy_open <= 0:
        return empty_label()
    buy_price = buy_open * 1.003
    next_high = float(hist.loc[buy_idx, "$high"])
    next_low = float(hist.loc[buy_idx, "$low"])
    next_close = float(hist.loc[buy_idx, "$close"])
    next_volume = float(hist.loc[buy_idx, "$volume"])
    next_amount = float(hist.loc[buy_idx, "$amount"])
    one_price_up = abs(next_high - next_low) <= 1e-6 and abs(buy_open - next_close) <= 1e-6 and next_close / preclose - 1.0 >= 0.045
    untradable = one_price_up or next_volume <= 0 or next_amount <= 0 or buy_open / preclose - 1.0 >= 0.03 - 1e-12
    exit_idx = min(buy_idx + 6, len(hist) - 1)
    reason = "max_hold_6d"
    peak = buy_price
    for e in range(buy_idx + 1, min(buy_idx + 6, len(hist) - 1) + 1):
        decision = e - 1
        close_decision = float(hist.loc[decision, "$close"])
        if np.isfinite(close_decision):
            peak = max(peak, close_decision)
        holding_days = e - buy_idx
        pnl_close = close_decision / buy_price - 1.0
        drawdown = close_decision / peak - 1.0 if peak > 0 else 0.0
        if holding_days >= 1 and pnl_close <= -0.06:
            exit_idx = e
            reason = "stop_loss_6pct"
            break
        if holding_days >= 2 and peak / buy_price - 1.0 >= 0.10 and drawdown <= -0.05:
            exit_idx = e
            reason = "trail_peak"
            break
        if holding_days >= 3 and float(hist.loc[decision, "ret3"] or 0.0) < -0.06:
            exit_idx = e
            reason = "short_momentum_down"
            break
    sell_open = float(hist.loc[exit_idx, "$open"])
    ret = sell_open / buy_price - 1.0 if np.isfinite(sell_open) and sell_open > 0 else np.nan
    close_path = hist.loc[buy_idx:exit_idx, "$close"].astype(float)
    mfe = float(close_path.max() / buy_price - 1.0) if len(close_path) else np.nan
    mae = float(close_path.min() / buy_price - 1.0) if len(close_path) else np.nan
    if untradable and np.isfinite(ret):
        ret = min(ret, -0.20)
    return {
        "buy_date": hist.loc[buy_idx, "datetime"],
        "exit_date": hist.loc[exit_idx, "datetime"],
        "forward_return": ret,
        "mfe_close": mfe,
        "mae_close": mae,
        "exit_reason_label": reason,
        "entry_one_price_limit_up": bool(one_price_up),
        "entry_zero_volume_or_amount": bool(next_volume <= 0 or next_amount <= 0),
        "entry_open_gap_ge_3pct": bool(buy_open / preclose - 1.0 >= 0.03 - 1e-12),
    }


def empty_label() -> dict:
    return {
        "buy_date": pd.NaT,
        "exit_date": pd.NaT,
        "forward_return": np.nan,
        "mfe_close": np.nan,
        "mae_close": np.nan,
        "exit_reason_label": "missing",
        "entry_one_price_limit_up": False,
        "entry_zero_volume_or_amount": False,
        "entry_open_gap_ge_3pct": False,
    }


def analyze(candidates: pd.DataFrame) -> dict:
    if candidates.empty:
        return {"ok": False, "reason": "empty candidates"}
    dates = pd.to_datetime(candidates["date"])
    exits = pd.to_datetime(candidates["exit_date"])
    train = candidates[(dates <= pd.Timestamp(TRAIN_END)) & (exits <= pd.Timestamp(TRAIN_END))]
    oos = candidates[dates >= pd.Timestamp(OOS_START)]
    score_defs = {
        "low_chase_strength": ["ret20_rank_pct", "amount20_rank_pct", "vol_ratio_rank_pct", "max_ret1_20_rank_pct", "drawdown_20_rank_pct", "ret1_rank_pct", "close_pos_rank_pct", "ma50_rel_rank_pct"],
        "repair_liquid": ["drawdown_20_rank_pct", "ret20_rank_pct", "amount20_rank_pct", "rebound_3_rank_pct", "ret1_rank_pct", "amplitude_pct_rank_pct"],
        "nonmain_strength": ["is_star", "is_chinext", "ret20_rank_pct", "amount20_rank_pct", "drawdown_20_rank_pct", "close_pos_rank_pct"],
    }
    data = candidates.copy()
    data["score_low_chase_strength"] = data["ret20_rank_pct"] + data["amount20_rank_pct"] + data["vol_ratio_rank_pct"] + data["max_ret1_20_rank_pct"] - data["ret1_rank_pct"] - data["close_pos_rank_pct"] - data["ma50_rel_rank_pct"]
    data["score_repair_liquid"] = data["ret20_rank_pct"] + data["amount20_rank_pct"] + data["rebound_3_rank_pct"] - data["drawdown_20_rank_pct"] - data["ret1_rank_pct"] - data["amplitude_pct_rank_pct"]
    data["score_nonmain_strength"] = 0.5 * data["is_star"] + 0.3 * data["is_chinext"] + data["ret20_rank_pct"] + data["amount20_rank_pct"] - data["close_pos_rank_pct"]
    score_reports = {}
    for score_col in ["score_low_chase_strength", "score_repair_liquid", "score_nonmain_strength"]:
        score_reports[score_col] = {
            "train_top7": topk_stats(data[(data["date"] <= pd.Timestamp(TRAIN_END)) & (pd.to_datetime(data["exit_date"]) <= pd.Timestamp(TRAIN_END))], score_col, 7),
            "oos_top7": topk_stats(data[data["date"] >= pd.Timestamp(OOS_START)], score_col, 7),
            "oos_top10": topk_stats(data[data["date"] >= pd.Timestamp(OOS_START)], score_col, 10),
        }
    candidate_reports = {}
    for name in ["candidate_elastic_repair", "candidate_low_chase_strength", "candidate_nonmain_repair", "candidate_union"]:
        candidate_reports[name] = {"train": basic_stats(train[train[name]]), "oos": basic_stats(oos[oos[name]])}
    untradable = {
        "one_price_limit_up": safe_mean(candidates["entry_one_price_limit_up"]),
        "zero_volume_or_amount": safe_mean(candidates["entry_zero_volume_or_amount"]),
        "open_gap_ge_3pct": safe_mean(candidates["entry_open_gap_ge_3pct"]),
    }
    return {
        "ok": True,
        "rows": int(len(candidates)),
        "date_count": int(candidates["date"].nunique()),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "candidate_reports": candidate_reports,
        "score_reports": score_reports,
        "entry_untradable_ratios": untradable,
        "score_defs": score_defs,
    }


def basic_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "dates": int(df["date"].nunique()),
        "mean": safe_mean(df["forward_return"]),
        "median": float(pd.to_numeric(df["forward_return"], errors="coerce").median()),
        "positive_ratio": safe_mean(df["forward_return"] > 0),
        "yearly_mean": {str(k): safe_mean(v["forward_return"]) for k, v in df.groupby("year")},
    }


def topk_stats(df: pd.DataFrame, score_col: str, topk: int) -> dict:
    clean = df.dropna(subset=[score_col, "forward_return"]).copy()
    if clean.empty:
        return {"rows": 0}
    top = clean.sort_values(["datetime", score_col], ascending=[True, False]).groupby("datetime").head(topk)
    return basic_stats(top)


def safe_mean(values) -> float | None:
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.mean()) if len(series) else None


if __name__ == "__main__":
    raise SystemExit(main())
