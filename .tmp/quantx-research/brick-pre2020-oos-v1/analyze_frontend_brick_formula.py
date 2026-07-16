from __future__ import annotations

import argparse
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
RUN_DIR = ROOT / "renko_official_runs"
PROVIDER = Path("data/qlib_data_fixed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Align official K-line frontend brick formula with strategy-renko candidates.")
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", default="all")
    parser.add_argument("--artifact-suffix", default="frontend")
    args = parser.parse_args()

    suffix = f"_{args.artifact_suffix}" if args.artifact_suffix else ""
    out_report = ROOT / f"frontend_brick_alignment_report{suffix}.json"
    out_rows = ROOT / f"frontend_brick_alignment_rows{suffix}.parquet"
    out_candidates = ROOT / f"frontend_brick_candidate_daily{suffix}.parquet"

    official = load_official_rows()
    if official.empty:
        raise RuntimeError(f"no official rows loaded from {RUN_DIR}")
    end = args.end or str(official["date"].max())
    official_dates = sorted(official["date"].dropna().unique().tolist())

    symbols = load_a_share_symbols(Path(args.provider_uri), args.start, end, universe=args.universe)
    reader = QlibBinReader(Path(args.provider_uri))
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount"], args.start, end)
    quote = normalize_quote(quote)

    panel = build_frontend_brick_panel(quote)
    panel["date"] = pd.to_datetime(panel["datetime"]).dt.strftime("%Y-%m-%d")
    official_key = official[["date", "instrument"]].dropna().drop_duplicates().assign(official_candidate=True)
    merged_all = panel[panel["date"].isin(official_dates)].merge(official_key, on=["date", "instrument"], how="left")
    merged_all["official_candidate"] = merged_all["official_candidate"].fillna(False).astype(bool)
    add_candidate_rules(merged_all)

    official_merged = official.merge(panel, on=["date", "instrument"], how="left")
    official_merged["has_local_bar"] = official_merged["$close"].notna()
    official_merged["local_change_pct"] = official_merged["ret1"] * 100.0
    official_merged["close_gap_pct"] = official_merged["official_close"] / official_merged["$close"] - 1.0
    full = official_merged[official_merged["official_brick"].notna() & official_merged["has_local_bar"]].copy()

    candidate_cols = [
        "candidate_delta_gt0",
        "candidate_prev_down_today_up",
        "candidate_decline_prev5_today_up",
        "candidate_decline_incl5_today_up",
        "candidate_decline_prev5_range_gt0",
        "candidate_decline_prev5_brick_gt60",
        "candidate_decline_prev5_ret_gt4",
        "candidate_decline_prev5_ret_rank90_brick_rank60_amt_rank50",
    ]
    candidate_reports = {name: candidate_quality(merged_all, name) for name in candidate_cols}
    report = {
        "inputs": {
            "provider_uri": str(args.provider_uri),
            "start": args.start,
            "end": end,
            "universe": args.universe,
            "symbols": len(symbols),
            "official_rows": int(len(official)),
            "official_dates": len(official_dates),
            "panel_rows_on_official_dates": int(len(merged_all)),
        },
        "formula_source": {
            "origin": "touzikexue K-line frontend chunk useKLineBars-CyArTqP-.js::bn",
            "window": "rolling high/low over current bar and previous 3 bars",
            "var2": "SMA((HHV4-close)/(HHV4-LLV4)*100-90, 4, 1)",
            "var4": "SMA((close-LLV4)/(HHV4-LLV4)*100, 6, 1)",
            "var5": "SMA(var4, 6, 1)",
            "brick": "max(0, var5 - var2 - 4)",
        },
        "official_field_checks_full_factor_dates": official_checks(full),
        "official_vs_local_field_corr_full_factor_dates": correlation_report(full),
        "candidate_quality_all_official_dates": candidate_reports,
        "daily_candidate_quality_tail20": daily_candidate_quality(merged_all, candidate_cols)[-20:],
        "diagnostic_examples": build_examples(official_merged, merged_all),
        "notes": [
            "002485 官方 daily-bars 的 brick 可由前端公式复算，最近 200 根最大绝对误差约 0.0013。",
            "本脚本使用本地 qlib 全市场行情复算同一公式，用于区分 brick 指标口径和候选过滤口径。",
            "若字段相关性接近 1 但候选覆盖仍低，下一步应反推 strategy-renko 的硬过滤和市场门控。",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    official_merged.to_parquet(out_rows, index=False)
    cols = [
        "date", "instrument", "official_candidate", "$open", "$high", "$low", "$close", "$volume", "$amount",
        "ret1", "amount_rank_pct", "ret1_rank_pct", "brick_rank_pct", "frontend_brick", "frontend_prev",
        "frontend_delta", "frontend_delta_prev", "frontend_decline_sum_prev5", "frontend_decline_sum_incl5",
        "frontend_max_consec_down_prev5", "frontend_range_from_5d_min", *candidate_cols,
    ]
    merged_all[[c for c in cols if c in merged_all.columns]].to_parquet(out_candidates, index=False)
    print(json.dumps({
        "wrote": str(out_report),
        "official_rows": int(len(official)),
        "official_dates": len(official_dates),
        "symbols": len(symbols),
        "field_corr": report["official_vs_local_field_corr_full_factor_dates"],
        "candidate_reports": candidate_reports,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def load_official_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        date = str(payload.get("data_date"))
        for item in payload.get("items", []):
            instrument = norm_symbol(str(item.get("symbol")))
            if instrument is None:
                continue
            fv = item.get("factor_values") or {}
            rows.append({
                "date": date,
                "run_id": payload.get("id"),
                "instrument": instrument,
                "symbol_raw": str(item.get("symbol")),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "official_score": item.get("score"),
                "official_close": item.get("close"),
                "official_change_pct": item.get("latest_change_pct"),
                "official_brick": fv.get("brick"),
                "official_prev_brick": fv.get("prev_brick"),
                "official_brick_delta_1": fv.get("brick_delta_1"),
                "official_reversal_h": fv.get("reversal_h"),
                "official_brick_decline_sum_5d": fv.get("brick_decline_sum_5d"),
                "official_brick_range_from_5d_min": fv.get("brick_range_from_5d_min"),
                "official_brick_max_consec_down_5d": fv.get("brick_max_consec_down_5d"),
                "official_day_ret": fv.get("day_ret"),
                "official_ret_3d": fv.get("ret_3d"),
                "official_ret_5d": fv.get("ret_5d"),
                "official_open_gap_pct": fv.get("open_gap_pct"),
                "official_turnover_rate": fv.get("turnover_rate"),
            })
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col.startswith("official_") or col in {"rank", "run_id"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def norm_symbol(code: str) -> str | None:
    code = code.strip()
    if code.startswith("SH") or code.startswith("SZ") or code.startswith("BJ"):
        return code
    if code.startswith("6"):
        return "SH" + code
    if code.startswith(("0", "3")):
        return "SZ" + code
    if code.startswith(("8", "9")):
        return "BJ" + code
    return None


def normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    if list(quote.index.names) != ["instrument", "datetime"]:
        quote = quote.reorder_levels(["instrument", "datetime"])
    quote = quote.sort_index()
    for col in quote.columns:
        quote[col] = pd.to_numeric(quote[col], errors="coerce").astype("float32")
    return quote


def build_frontend_brick_panel(quote: pd.DataFrame) -> pd.DataFrame:
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    for col in ["frontend_brick", "frontend_var2_base", "frontend_var4", "frontend_var5_base"]:
        frame[col] = np.nan
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        high = frame.loc[loc, "$high"].to_numpy(dtype="float64")
        low = frame.loc[loc, "$low"].to_numpy(dtype="float64")
        close = frame.loc[loc, "$close"].to_numpy(dtype="float64")
        brick, var2, var4, var5 = frontend_brick(high, low, close)
        frame.loc[loc, "frontend_brick"] = brick
        frame.loc[loc, "frontend_var2_base"] = var2
        frame.loc[loc, "frontend_var4"] = var4
        frame.loc[loc, "frontend_var5_base"] = var5
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    frame["frontend_prev"] = grouped["frontend_brick"].shift(1)
    frame["frontend_delta"] = frame["frontend_brick"] - frame["frontend_prev"]
    frame["frontend_delta_prev"] = grouped["frontend_delta"].shift(1)
    down = (-frame["frontend_delta"].clip(upper=0)).fillna(0.0)
    frame["frontend_decline_sum_incl5"] = down.groupby(frame["instrument"]).transform(lambda s: s.rolling(5, min_periods=1).sum())
    frame["frontend_decline_sum_prev5"] = down.groupby(frame["instrument"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    frame["frontend_max_consec_down_prev5"] = rolling_max_consec_down(frame, "frontend_delta", shift=1)
    frame["frontend_range_from_5d_min"] = frame["frontend_brick"] - grouped["frontend_brick"].transform(lambda s: s.rolling(5, min_periods=1).min())
    frame["ret1"] = grouped["$close"].pct_change(fill_method=None)
    return frame


def frontend_brick(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    brick = np.zeros(n, dtype="float64")
    var2 = np.zeros(n, dtype="float64")
    var4 = np.zeros(n, dtype="float64")
    var5 = np.zeros(n, dtype="float64")
    prev_var2 = prev_var4 = prev_var5 = None
    for t in range(n):
        start = max(0, t - 3)
        hh = np.nanmax(high[start : t + 1])
        ll = np.nanmin(low[start : t + 1])
        rng = hh - ll
        if not np.isfinite(rng) or rng == 0:
            low_side = 0.0
            high_side = 0.0
        else:
            low_side = (hh - close[t]) / rng * 100.0 - 90.0
            high_side = (close[t] - ll) / rng * 100.0
        prev_var2 = sma_like(low_side, prev_var2, 4.0)
        prev_var4 = sma_like(high_side, prev_var4, 6.0)
        prev_var5 = sma_like(prev_var4, prev_var5, 6.0)
        var2[t] = prev_var2
        var4[t] = prev_var4
        var5[t] = prev_var5
        diff = prev_var5 - prev_var2
        brick[t] = diff - 4.0 if diff > 4.0 else 0.0
    return brick, var2, var4, var5


def sma_like(value: float, prev: float | None, period: float) -> float:
    return value if prev is None else (value + (period - 1.0) * prev) / period


def rolling_max_consec_down(frame: pd.DataFrame, col: str, *, shift: int) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float32")
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        delta = frame.loc[loc, col].to_numpy(dtype="float64")
        down = np.where(np.isfinite(delta), delta < 0, False)
        values = np.zeros(len(delta), dtype="float32")
        for i in range(len(delta)):
            end = i + 1 - shift
            start = max(0, end - 5)
            if end <= start:
                continue
            run = 0
            best = 0
            for flag in down[start:end]:
                if flag:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            values[i] = best
        out.loc[loc] = values
    return out


def add_candidate_rules(frame: pd.DataFrame) -> None:
    frame["candidate_delta_gt0"] = frame["frontend_delta"] > 0
    frame["candidate_prev_down_today_up"] = (frame["frontend_delta_prev"] < 0) & (frame["frontend_delta"] > 0)
    frame["candidate_decline_prev5_today_up"] = (frame["frontend_delta"] > 0) & (frame["frontend_decline_sum_prev5"] > 0)
    frame["candidate_decline_incl5_today_up"] = (frame["frontend_delta"] > 0) & (frame["frontend_decline_sum_incl5"] > 0)
    frame["candidate_decline_prev5_range_gt0"] = frame["candidate_decline_prev5_today_up"] & (frame["frontend_range_from_5d_min"] > 0)
    frame["candidate_decline_prev5_brick_gt60"] = frame["candidate_decline_prev5_today_up"] & (frame["frontend_brick"] >= 60)
    frame["candidate_decline_prev5_ret_gt4"] = frame["candidate_decline_prev5_today_up"] & (frame["ret1"] >= 0.04)
    frame["amount_rank_pct"] = frame.groupby("date")["$amount"].rank(pct=True)
    frame["ret1_rank_pct"] = frame.groupby("date")["ret1"].rank(pct=True)
    frame["brick_rank_pct"] = frame.groupby("date")["frontend_brick"].rank(pct=True)
    frame["candidate_decline_prev5_ret_rank90_brick_rank60_amt_rank50"] = (
        frame["candidate_decline_prev5_today_up"]
        & (frame["ret1_rank_pct"] >= 0.90)
        & (frame["brick_rank_pct"] >= 0.60)
        & (frame["amount_rank_pct"] >= 0.50)
    )


def correlation_report(full: pd.DataFrame) -> dict[str, float | None]:
    pairs = {
        "official_brick_vs_frontend_brick": ("official_brick", "frontend_brick"),
        "official_prev_brick_vs_frontend_prev": ("official_prev_brick", "frontend_prev"),
        "official_delta_vs_frontend_delta": ("official_brick_delta_1", "frontend_delta"),
        "official_reversal_h_vs_frontend_delta": ("official_reversal_h", "frontend_delta"),
        "official_decline_sum_vs_frontend_prev5": ("official_brick_decline_sum_5d", "frontend_decline_sum_prev5"),
        "official_decline_sum_vs_frontend_incl5": ("official_brick_decline_sum_5d", "frontend_decline_sum_incl5"),
        "official_range_vs_frontend_range": ("official_brick_range_from_5d_min", "frontend_range_from_5d_min"),
        "official_day_ret_vs_local_ret1": ("official_day_ret", "ret1"),
        "official_close_vs_local_close": ("official_close", "$close"),
    }
    out: dict[str, float | None] = {}
    for name, (left, right) in pairs.items():
        values = full[[left, right]].dropna()
        out[name] = float(values[left].corr(values[right])) if len(values) >= 3 else None
    return out


def official_checks(full: pd.DataFrame) -> dict[str, Any]:
    if full.empty:
        return {"rows": 0}
    return {
        "rows": int(len(full)),
        "dates": int(full["date"].nunique()),
        "delta_eq_brick_minus_prev_ratio": bool_mean(np.isclose(full["official_brick_delta_1"], full["official_brick"] - full["official_prev_brick"], atol=1e-6)),
        "delta_eq_reversal_h_ratio": bool_mean(np.isclose(full["official_brick_delta_1"], full["official_reversal_h"], atol=1e-6)),
        "official_delta_positive_ratio": float((full["official_brick_delta_1"] > 0).mean()),
        "official_decline_sum_positive_ratio": float((full["official_brick_decline_sum_5d"] > 0).mean()),
        "official_max_consec_down_ge1_ratio": float((full["official_brick_max_consec_down_5d"] >= 1).mean()),
        "local_frontend_delta_positive_ratio": float((full["frontend_delta"] > 0).mean()),
        "local_frontend_prev5_decline_positive_ratio": float((full["frontend_decline_sum_prev5"] > 0).mean()),
        "local_frontend_scale_summary": describe(full["frontend_brick"]),
        "official_brick_scale_summary": describe(full["official_brick"]),
        "close_gap_pct_summary": describe(full["close_gap_pct"]),
    }


def candidate_quality(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    pred = frame[col].fillna(False).astype(bool)
    truth = frame["official_candidate"].fillna(False).astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": int(pred.sum()),
        "official_count": int(truth.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_pred_per_day": float(frame.loc[pred].groupby("date")["instrument"].nunique().mean()) if pred.any() else 0.0,
    }


def daily_candidate_quality(frame: pd.DataFrame, candidate_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        row = {"date": date, "official_count": int(group["official_candidate"].sum())}
        for col in candidate_cols:
            stats = candidate_quality(group, col)
            row[f"{col}_pred"] = stats["pred_count"]
            row[f"{col}_recall"] = stats["recall"]
            row[f"{col}_precision"] = stats["precision"]
        rows.append(row)
    return rows


def build_examples(official_merged: pd.DataFrame, merged_all: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    cols = [
        "date", "instrument", "name", "rank", "official_score", "official_brick", "official_prev_brick",
        "official_brick_delta_1", "official_reversal_h", "official_brick_decline_sum_5d",
        "official_brick_max_consec_down_5d", "frontend_brick", "frontend_prev", "frontend_delta",
        "frontend_decline_sum_prev5", "frontend_decline_sum_incl5", "frontend_max_consec_down_prev5", "ret1", "$close",
    ]
    available = [c for c in cols if c in official_merged.columns]
    full = official_merged[official_merged["official_brick"].notna()].copy()
    misses = merged_all[merged_all["candidate_decline_prev5_today_up"] & ~merged_all["official_candidate"]]
    return {
        "official_full_factor_head30": full[available].head(30).replace({np.nan: None}).to_dict("records"),
        "local_pred_not_official_head30": misses[[c for c in ["date", "instrument", "frontend_brick", "frontend_delta", "frontend_decline_sum_prev5", "ret1", "$close", "$amount"] if c in misses.columns]].head(30).replace({np.nan: None}).to_dict("records"),
    }


def bool_mean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values)
    return float(np.nanmean(arr.astype(float))) if len(arr) else 0.0


def describe(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
