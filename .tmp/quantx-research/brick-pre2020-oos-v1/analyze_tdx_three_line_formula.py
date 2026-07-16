from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_backtest import load_a_share_symbols


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
ALIGN_SCRIPT = ROOT / "analyze_raw_brick_alignment.py"
PROVIDER = Path("data/qlib_data_fixed")
OUT_JSON = ROOT / "tdx_three_line_formula_alignment.json"
OUT_ROWS = ROOT / "tdx_three_line_formula_rows.parquet"


def main() -> int:
    align = load_align_module()
    official = align.load_official_rows()
    official_dates = sorted(official["date"].dropna().unique().tolist())
    symbols = load_a_share_symbols(PROVIDER, "2025-01-01", str(official["date"].max()), universe="all")
    reader = QlibBinReader(PROVIDER)
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount"], "2025-01-01", str(official["date"].max()))
    quote = align.normalize_quote(quote)
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    frame["date"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")

    official_key = official[["date", "instrument"]].dropna().drop_duplicates().assign(official_candidate=True)
    all_rows = frame[frame["date"].isin(official_dates)].merge(official_key, on=["date", "instrument"], how="left")
    all_rows["official_candidate"] = all_rows["official_candidate"].fillna(False).astype(bool)

    reports: dict[str, Any] = {}
    sample_frames = []
    for n in range(2, 11):
        calc = compute_three_line(frame, n)
        calc = calc[calc["date"].isin(official_dates)].merge(official_key, on=["date", "instrument"], how="left")
        calc["official_candidate"] = calc["official_candidate"].fillna(False).astype(bool)
        calc["red_state_prev"] = calc.groupby("instrument", sort=False)["red_state"].shift(1).fillna(False)
        calc["green_state_prev"] = calc.groupby("instrument", sort=False)["green_state"].shift(1).fillna(False)
        calc["red_signal_prev"] = calc.groupby("instrument", sort=False)["red_signal"].shift(1).fillna(False)
        calc["green_signal_prev"] = calc.groupby("instrument", sort=False)["green_signal"].shift(1).fillna(False)
        calc["recent_green_signal_5"] = calc.groupby("instrument", sort=False)["green_signal"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).max().fillna(False).astype(bool)
        )
        calc["recent_green_state_5"] = calc.groupby("instrument", sort=False)["green_state"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).max().fillna(False).astype(bool)
        )
        calc["rule_red_signal"] = calc["red_signal"]
        calc["rule_red_state"] = calc["red_state"]
        calc["rule_red_signal_after_green_signal_prev"] = calc["red_signal"] & calc["green_signal_prev"]
        calc["rule_red_signal_after_green_state_prev"] = calc["red_signal"] & calc["green_state_prev"]
        calc["rule_red_state_flip_from_green"] = calc["red_state"] & calc["green_state_prev"]
        calc["rule_red_signal_after_recent_green_signal_5"] = calc["red_signal"] & calc["recent_green_signal_5"]
        calc["rule_red_signal_after_recent_green_state_5"] = calc["red_signal"] & calc["recent_green_state_5"]
        rule_cols = [c for c in calc.columns if c.startswith("rule_")]
        reports[str(n)] = {col: candidate_quality(calc, col) for col in rule_cols}
        if n == 3:
            sample_frames.append(calc)

    best = []
    for n, per_rule in reports.items():
        for rule, stats in per_rule.items():
            best.append({"n": int(n), "rule": rule, **stats})
    best_frame = pd.DataFrame(best).sort_values(["f1", "recall", "precision"], ascending=False)
    n3 = sample_frames[0] if sample_frames else pd.DataFrame()
    official_n3 = n3[n3["official_candidate"]].copy()
    report = {
        "formula": {
            "source": "用户提供的通达信三线翻红/翻绿公式",
            "tdx": "N:=3; A3:=REF(C,N); 突破:=C>HHV(A3,N); 破位:=C<LLV(A3,N); 三线翻红:BARSLAST(突破)<BARSLAST(破位) AND C>O; 三线翻绿:BARSLAST(突破)>BARSLAST(破位) AND C<O;",
            "translation": "HHV(REF(C,N),N) 取 t-N 到 t-2N+1 的 close 高点；BARSLAST 今日触发为 0，未触发用大数。",
        },
        "inputs": {
            "official_rows": int(len(official)),
            "official_dates": len(official_dates),
            "symbols": len(symbols),
            "all_rows_on_official_dates": int(len(all_rows)),
        },
        "best_rules_top30": best_frame.head(30).to_dict("records"),
        "n3_rules": reports.get("3", {}),
        "n3_official_candidate_state_ratios": state_ratios(official_n3),
        "n3_mismatch_examples": official_n3[
            ~official_n3["rule_red_signal_after_green_state_prev"].fillna(False)
        ][[
            "date", "instrument", "$open", "$close", "breakout", "breakdown", "barslast_breakout",
            "barslast_breakdown", "red_state", "green_state", "red_signal", "green_signal",
            "green_state_prev", "green_signal_prev", "recent_green_state_5", "recent_green_signal_5",
        ]].head(80).replace({np.nan: None}).to_dict("records"),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if not n3.empty:
        n3.to_parquet(OUT_ROWS, index=False)
    print(json.dumps({
        "wrote": str(OUT_JSON),
        "rows": int(len(best_frame)),
        "top10": best_frame.head(10).to_dict("records"),
        "n3": reports.get("3", {}),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def load_align_module():
    spec = importlib.util.spec_from_file_location("raw_brick_alignment_module", ALIGN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ALIGN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_three_line(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    out = frame[["datetime", "date", "instrument", "$open", "$close", "$volume", "$amount"]].copy()
    out["breakout"] = False
    out["breakdown"] = False
    out["barslast_breakout"] = np.nan
    out["barslast_breakdown"] = np.nan
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        close = frame.loc[loc, "$close"].to_numpy(dtype="float64")
        ref = np.full(len(close), np.nan, dtype="float64")
        ref[n:] = close[:-n]
        hhv = rolling_extreme(ref, n, "max")
        llv = rolling_extreme(ref, n, "min")
        breakout = close > hhv
        breakdown = close < llv
        out.loc[loc, "breakout"] = breakout
        out.loc[loc, "breakdown"] = breakdown
        out.loc[loc, "barslast_breakout"] = barslast(breakout)
        out.loc[loc, "barslast_breakdown"] = barslast(breakdown)
    out["red_state"] = out["barslast_breakout"] < out["barslast_breakdown"]
    out["green_state"] = out["barslast_breakout"] > out["barslast_breakdown"]
    out["red_signal"] = out["red_state"] & (out["$close"] > out["$open"])
    out["green_signal"] = out["green_state"] & (out["$close"] < out["$open"])
    return out


def rolling_extreme(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype="float64")
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        segment = segment[np.isfinite(segment)]
        if len(segment) == 0:
            continue
        out[i] = np.max(segment) if mode == "max" else np.min(segment)
    return out


def barslast(cond: np.ndarray) -> np.ndarray:
    out = np.full(len(cond), 1000000, dtype="int32")
    last = -1000000
    for i, flag in enumerate(cond):
        if bool(flag):
            last = i
        out[i] = i - last
    return out


def candidate_quality(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    pred = frame[col].fillna(False).astype(bool)
    truth = frame["official_candidate"].fillna(False).astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    counts = frame.loc[pred].groupby("date")["instrument"].nunique()
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": int(pred.sum()),
        "official_count": int(truth.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_pred_per_day": float(counts.mean()) if len(counts) else 0.0,
        "median_pred_per_day": float(counts.median()) if len(counts) else 0.0,
    }


def state_ratios(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"rows": 0}
    cols = [
        "red_state", "green_state", "red_signal", "green_signal", "green_state_prev", "green_signal_prev",
        "recent_green_state_5", "recent_green_signal_5", "rule_red_signal_after_green_signal_prev",
        "rule_red_signal_after_green_state_prev", "rule_red_signal_after_recent_green_state_5",
    ]
    return {"rows": int(len(frame)), **{col: float(frame[col].fillna(False).astype(bool).mean()) for col in cols}}


if __name__ == "__main__":
    raise SystemExit(main())
