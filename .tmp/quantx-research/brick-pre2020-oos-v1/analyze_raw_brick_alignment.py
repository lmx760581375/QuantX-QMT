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
from quantx.core.factor_runtime import ops
from quantx.tools.run_backtest import load_a_share_symbols


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
RUN_DIR = ROOT / "renko_official_runs"
PROVIDER = Path("data/qlib_data_fixed")
OUT_REPORT = ROOT / "raw_brick_alignment_report.json"
OUT_ROWS = ROOT / "raw_brick_alignment_rows.parquet"
OUT_CANDIDATES = ROOT / "raw_brick_candidate_daily.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Align local raw brick formula with official strategy-renko candidates.")
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", default="all")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m1", type=int, default=3)
    parser.add_argument("--m2", type=int, default=12)
    parser.add_argument("--m3", type=int, default=12)
    parser.add_argument("--t", type=float, default=8.0)
    parser.add_argument("--shift1", type=float, default=92.0)
    parser.add_argument("--shift2", type=float, default=114.0)
    parser.add_argument("--artifact-suffix", default="")
    args = parser.parse_args()

    suffix = f"_{args.artifact_suffix}" if args.artifact_suffix else ""
    out_report = ROOT / f"raw_brick_alignment_report{suffix}.json"
    out_rows = ROOT / f"raw_brick_alignment_rows{suffix}.parquet"
    out_candidates = ROOT / f"raw_brick_candidate_daily{suffix}.parquet"
    brick_params = {
        "n": args.n,
        "m1": args.m1,
        "m2": args.m2,
        "m3": args.m3,
        "t": args.t,
        "shift1": args.shift1,
        "shift2": args.shift2,
    }

    official = load_official_rows()
    if official.empty:
        raise RuntimeError(f"no official rows loaded from {RUN_DIR}")
    end = args.end or str(official["date"].max())
    official_dates = sorted(official["date"].dropna().unique().tolist())
    symbols = load_a_share_symbols(Path(args.provider_uri), args.start, end, universe=args.universe)
    reader = QlibBinReader(Path(args.provider_uri))
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount"], args.start, end)
    quote = normalize_quote(quote)

    panel = build_raw_brick_panel(quote, brick_params)
    panel["date"] = pd.to_datetime(panel["datetime"]).dt.strftime("%Y-%m-%d")
    official_key = official[["date", "instrument"]].dropna().drop_duplicates().assign(official_candidate=True)
    merged_all = panel[panel["date"].isin(official_dates)].merge(official_key, on=["date", "instrument"], how="left")
    merged_all["official_candidate"] = merged_all["official_candidate"].fillna(False).astype(bool)
    merged_all["candidate_delta"] = merged_all["raw_delta"] > 0
    merged_all["candidate_reversal_prev5"] = (
        (merged_all["raw_delta"] > 0)
        & (merged_all["raw_decline_sum_prev5"] > 0)
        & (merged_all["raw_max_consec_down_prev5"] >= 1)
    )
    merged_all["candidate_reversal_incl5"] = (
        (merged_all["raw_delta"] > 0)
        & (merged_all["raw_decline_sum_incl5"] > 0)
        & (merged_all["raw_max_consec_down_incl5"] >= 1)
    )
    merged_all["candidate_prev_down_today_up"] = (merged_all["raw_delta"] > 0) & (merged_all["raw_delta_prev"] < 0)
    merged_all["candidate_current_brickchart_g2r"] = (merged_all["brick_delta"] > 0) & (merged_all["brick_delta_prev"] < 0)

    official_merged = official.merge(
        panel,
        on=["date", "instrument"],
        how="left",
        suffixes=("", "_local"),
    )
    official_merged["has_local_bar"] = official_merged["$close"].notna()
    official_merged["local_change_pct"] = official_merged["ret1"] * 100.0
    official_merged["close_gap_pct"] = official_merged["official_close"] / official_merged["$close"] - 1.0
    official_merged["official_delta_eq_local_raw_delta"] = np.isclose(
        official_merged["official_brick_delta_1"], official_merged["raw_delta"], rtol=1e-5, atol=1e-4
    )
    official_merged["official_reversal_eq_local_raw_delta"] = np.isclose(
        official_merged["official_reversal_h"], official_merged["raw_delta"], rtol=1e-5, atol=1e-4
    )

    full = official_merged[official_merged["official_brick"].notna() & official_merged["has_local_bar"]].copy()
    field_corr = correlation_report(full)
    official_field_checks = official_checks(full)
    candidate_reports = {
        name: candidate_quality(merged_all, name)
        for name in [
            "candidate_delta",
            "candidate_reversal_prev5",
            "candidate_reversal_incl5",
            "candidate_prev_down_today_up",
            "candidate_current_brickchart_g2r",
        ]
    }
    daily_reports = daily_candidate_quality(merged_all)
    formula_examples = build_formula_examples(official_merged)
    report = {
        "inputs": {
            "provider_uri": str(args.provider_uri),
            "start": args.start,
            "end": end,
            "universe": args.universe,
            "brick_params": brick_params,
            "symbols": len(symbols),
            "official_rows": int(len(official)),
            "official_dates": len(official_dates),
            "panel_rows_on_official_dates": int(len(merged_all)),
        },
        "core_observations": [
            "当前 QuantX BrickChart 返回 raw brick 高度的一阶差分；官方 factor_values.brick 更像 raw brick 高度本体。",
            "官方 2026-07-03 以后完整字段里 brick_delta_1、reversal_h、brick-prev_brick 完全一致。",
            "官方候选公式更接近 raw brick 过去下行后当日上拐，而不是本地 brick_delta 的 green_to_red 简化条件。",
            "候选池对齐优先级高于模型训练；候选池不对齐时，严格 OOS 模型收益没有解释意义。",
        ],
        "official_field_checks_full_factor_dates": official_field_checks,
        "official_vs_local_field_corr_full_factor_dates": field_corr,
        "candidate_quality_all_official_dates": candidate_reports,
        "daily_candidate_quality_tail20": daily_reports[-20:],
        "formula_examples": formula_examples,
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    official_merged.to_parquet(out_rows, index=False)
    merged_all[[
        "date", "instrument", "official_candidate", "$open", "$high", "$low", "$close", "$volume", "$amount",
        "ret1", "raw_brick", "raw_prev", "raw_delta", "raw_delta_prev", "raw_decline_sum_prev5",
        "raw_decline_sum_incl5", "raw_max_consec_down_prev5", "raw_max_consec_down_incl5",
        "candidate_delta", "candidate_reversal_prev5", "candidate_reversal_incl5",
        "candidate_prev_down_today_up", "candidate_current_brickchart_g2r",
    ]].to_parquet(out_candidates, index=False)
    print(json.dumps({
        "wrote": str(out_report),
        "official_rows": int(len(official)),
        "official_dates": len(official_dates),
        "symbols": len(symbols),
        "brick_params": brick_params,
        "best_candidate_f1": max((v["f1"] for v in candidate_reports.values()), default=0.0),
        "candidate_reports": candidate_reports,
        "field_corr": field_corr,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def load_official_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        date = str(payload.get("data_date"))
        for item in payload.get("items", []):
            symbol = norm_symbol(str(item.get("symbol")))
            if symbol is None:
                continue
            fv = item.get("factor_values") or {}
            rows.append({
                "date": date,
                "run_id": payload.get("id"),
                "instrument": symbol,
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
            })
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col.startswith("official_") or col in {"rank", "run_id"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def norm_symbol(code: str) -> str | None:
    code = code.strip()
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


def build_raw_brick_panel(quote: pd.DataFrame, params: dict[str, float | int] | None = None) -> pd.DataFrame:
    params = params or {"n": 8, "m1": 3, "m2": 12, "m3": 12, "t": 8.0, "shift1": 92.0, "shift2": 114.0}
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    frame["raw_brick"] = np.nan
    frame["brick_delta"] = np.nan
    for _, idx in frame.groupby("instrument", sort=False).groups.items():
        loc = np.asarray(idx)
        high = frame.loc[loc, "$high"].to_numpy(dtype="float32")
        low = frame.loc[loc, "$low"].to_numpy(dtype="float32")
        close = frame.loc[loc, "$close"].to_numpy(dtype="float32")
        raw = brick_chart_raw(
            high,
            low,
            close,
            int(params["n"]),
            int(params["m1"]),
            int(params["m2"]),
            int(params["m3"]),
            float(params["t"]),
            float(params["shift1"]),
            float(params["shift2"]),
            1,
            1,
            1,
        )
        frame.loc[loc, "raw_brick"] = raw
        delta = np.zeros_like(raw, dtype="float32")
        delta[1:] = raw[1:] - raw[:-1]
        frame.loc[loc, "brick_delta"] = delta
    grouped = frame.groupby("instrument", group_keys=False, sort=False)
    frame["raw_prev"] = grouped["raw_brick"].shift(1)
    frame["raw_delta"] = frame["raw_brick"] - frame["raw_prev"]
    frame["raw_delta_prev"] = grouped["raw_delta"].shift(1)
    frame["brick_delta_prev"] = grouped["brick_delta"].shift(1)
    down = (-frame["raw_delta"].clip(upper=0)).fillna(0.0)
    frame["raw_decline_sum_incl5"] = down.groupby(frame["instrument"]).transform(lambda s: s.rolling(5, min_periods=1).sum())
    frame["raw_decline_sum_prev5"] = down.groupby(frame["instrument"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    frame["raw_max_consec_down_incl5"] = rolling_max_consec_down(frame, "raw_delta", shift=0)
    frame["raw_max_consec_down_prev5"] = rolling_max_consec_down(frame, "raw_delta", shift=1)
    frame["raw_range_from_5d_min_incl"] = frame["raw_brick"] - grouped["raw_brick"].transform(lambda s: s.rolling(5, min_periods=1).min())
    frame["raw_range_from_5d_min_prev"] = frame["raw_brick"] - grouped["raw_brick"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())
    frame["ret1"] = grouped["$close"].pct_change(fill_method=None)
    return frame


def brick_chart_raw(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int,
    m1: int,
    m2: int,
    m3: int,
    t: float,
    shift1: float,
    shift2: float,
    sma_w1: int,
    sma_w2: int,
    sma_w3: int,
) -> np.ndarray:
    hi = np.asarray(high, dtype="float32")[:, None]
    lo = np.asarray(low, dtype="float32")[:, None]
    cls = np.asarray(close, dtype="float32")[:, None]
    hhv = ops.rolling_max(hi, int(n))
    llv = ops.rolling_min(lo, int(n))
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
    return raw[:, 0]


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
                values[i] = 0
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


def correlation_report(full: pd.DataFrame) -> dict[str, float | None]:
    pairs = {
        "official_brick_vs_raw_brick": ("official_brick", "raw_brick"),
        "official_prev_brick_vs_raw_prev": ("official_prev_brick", "raw_prev"),
        "official_delta_vs_raw_delta": ("official_brick_delta_1", "raw_delta"),
        "official_reversal_h_vs_raw_delta": ("official_reversal_h", "raw_delta"),
        "official_decline_sum_vs_local_prev5": ("official_brick_decline_sum_5d", "raw_decline_sum_prev5"),
        "official_decline_sum_vs_local_incl5": ("official_brick_decline_sum_5d", "raw_decline_sum_incl5"),
        "official_range_vs_local_range_incl": ("official_brick_range_from_5d_min", "raw_range_from_5d_min_incl"),
        "official_range_vs_local_range_prev": ("official_brick_range_from_5d_min", "raw_range_from_5d_min_prev"),
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
        "local_raw_delta_positive_ratio": float((full["raw_delta"] > 0).mean()),
        "local_raw_prev5_decline_positive_ratio": float((full["raw_decline_sum_prev5"] > 0).mean()),
        "local_raw_incl5_decline_positive_ratio": float((full["raw_decline_sum_incl5"] > 0).mean()),
        "local_raw_scale_summary": describe(full["raw_brick"]),
        "official_raw_scale_summary": describe(full["official_brick"]),
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


def daily_candidate_quality(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        row = {"date": date, "official_count": int(group["official_candidate"].sum())}
        for col in ["candidate_reversal_prev5", "candidate_reversal_incl5", "candidate_current_brickchart_g2r"]:
            stats = candidate_quality(group, col)
            row[f"{col}_pred"] = stats["pred_count"]
            row[f"{col}_recall"] = stats["recall"]
            row[f"{col}_precision"] = stats["precision"]
        rows.append(row)
    return rows


def build_formula_examples(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    cols = [
        "date", "instrument", "name", "rank", "official_score", "official_brick", "official_prev_brick",
        "official_brick_delta_1", "official_reversal_h", "official_brick_decline_sum_5d",
        "official_brick_max_consec_down_5d", "raw_brick", "raw_prev", "raw_delta",
        "raw_decline_sum_prev5", "raw_decline_sum_incl5", "raw_max_consec_down_prev5", "ret1", "$close",
    ]
    available = [c for c in cols if c in frame.columns]
    full = frame[frame["official_brick"].notna()].copy()
    no_bar = frame[~frame["has_local_bar"]].copy()
    return {
        "full_factor_head30": full[available].head(30).replace({np.nan: None}).to_dict("records"),
        "missing_local_bar_head20": no_bar[[c for c in available if c in no_bar.columns]].head(20).replace({np.nan: None}).to_dict("records"),
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
