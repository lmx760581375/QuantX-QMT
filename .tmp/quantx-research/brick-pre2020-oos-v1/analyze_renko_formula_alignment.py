from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
RUN_DIR = ROOT / "renko_official_runs"
BASE_SCRIPT = ROOT / "run_brick_pre2020_oos.py"
OUT = ROOT / "renko_formula_alignment_report.json"


def load_base():
    spec = importlib.util.spec_from_file_location("brick_base_for_renko_alignment", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def norm_symbol(code: str) -> str | None:
    code = str(code)
    if code.startswith("6"):
        return "SH" + code
    if code.startswith(("0", "3")):
        return "SZ" + code
    if code.startswith("920"):
        return "BJ" + code
    return None


def load_official_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        data_date = str(payload.get("data_date"))
        for item in payload.get("items", []):
            fv = item.get("factor_values") or {}
            symbol = norm_symbol(str(item.get("symbol")))
            row = {
                "date": data_date,
                "symbol_raw": str(item.get("symbol")),
                "instrument": symbol,
                "name": item.get("name"),
                "rank": item.get("rank"),
                "official_score": item.get("score"),
                "official_close": item.get("close"),
                "official_change_pct": item.get("latest_change_pct"),
                "official_brick": fv.get("brick"),
                "official_prev_brick": fv.get("prev_brick"),
                "official_brick_abs": fv.get("brick_abs"),
                "official_brick_delta_1": fv.get("brick_delta_1"),
                "official_reversal_h": fv.get("reversal_h"),
                "official_brick_decline_sum_5d": fv.get("brick_decline_sum_5d"),
                "official_day_ret": fv.get("day_ret"),
                "official_ret_3d": fv.get("ret_3d"),
                "official_ret_5d": fv.get("ret_5d"),
                "official_turnover_rate": fv.get("turnover_rate"),
                "official_candidate_count": fv.get("candidate_count"),
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col.startswith("official_") or col == "rank":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def main() -> int:
    base = load_base()
    official = load_official_rows()
    symbols = sorted(s for s in official["instrument"].dropna().unique() if not str(s).startswith("BJ"))
    start = "2025-01-01"
    end = str(official["date"].max())
    reader = base.QlibBinReader(Path("data/qlib_data_fixed"))
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"], start, end)
    quote = quote.reorder_levels(["datetime", "instrument"]).sort_index()
    panel = base.build_panel(quote, set())
    panel["date"] = pd.to_datetime(panel["datetime"]).dt.strftime("%Y-%m-%d")
    keep_cols = [
        "date", "instrument", "$open", "$high", "$low", "$close", "$volume", "$amount",
        "ret1", "ret3", "ret5", "brick", "brick_prev", "brick_growth", "brick_strength",
        "green_to_red", "prior_green_bars", "red_risk_7", "amplitude", "vol_ratio",
    ]
    merged = official.merge(panel[keep_cols], on=["date", "instrument"], how="left", suffixes=("", "_local"))
    merged["has_local_bar"] = merged["$close"].notna()
    merged["local_change_pct"] = merged["ret1"] * 100.0
    merged["official_close_gap_pct"] = merged["official_close"] / merged["$close"] - 1.0
    merged["official_vs_local_change_gap"] = merged["official_change_pct"] - merged["local_change_pct"]
    merged["official_delta_matches_reversal"] = np.isclose(
        merged["official_brick_delta_1"], merged["official_reversal_h"], rtol=1e-6, atol=1e-6
    )
    merged["official_delta_equals_brick_minus_prev"] = np.isclose(
        merged["official_brick_delta_1"], merged["official_brick"] - merged["official_prev_brick"], rtol=1e-6, atol=1e-4
    )
    merged["official_candidate_positive_delta"] = merged["official_brick_delta_1"] > 0
    merged["local_green_to_red"] = (merged["brick"] > 0) & (merged["brick_prev"] < 0)
    merged["local_red_current"] = merged["brick"] > 0

    corr_cols = [
        "official_brick", "official_prev_brick", "official_brick_delta_1", "official_reversal_h",
        "official_day_ret", "official_ret_3d", "official_ret_5d", "official_close",
        "brick", "brick_prev", "brick_strength", "ret1", "ret3", "ret5", "$close",
    ]
    corr = merged[corr_cols].corr(numeric_only=True).fillna(0.0)
    important_corr = {
        "official_brick_vs_local": float(corr.loc["official_brick", "brick"]) if "official_brick" in corr.index else 0.0,
        "official_delta_vs_local_brick": float(corr.loc["official_brick_delta_1", "brick"]) if "official_brick_delta_1" in corr.index else 0.0,
        "official_day_ret_vs_local_ret1": float(corr.loc["official_day_ret", "ret1"]) if "official_day_ret" in corr.index else 0.0,
        "official_close_vs_local_close": float(corr.loc["official_close", "$close"]) if "official_close" in corr.index else 0.0,
    }

    by_date = []
    for date, group in merged.groupby("date", sort=True):
        by_date.append({
            "date": date,
            "official_count": int(len(group)),
            "local_bar_count": int(group["has_local_bar"].sum()),
            "local_green_to_red_overlap": int(group["local_green_to_red"].sum()),
            "local_red_current_overlap": int(group["local_red_current"].sum()),
            "official_positive_delta_count": int(group["official_candidate_positive_delta"].sum()),
            "median_close_gap_pct": finite_median(group["official_close_gap_pct"]),
            "median_change_gap_pct": finite_median(group["official_vs_local_change_gap"]),
        })

    report = {
        "rows": int(len(merged)),
        "dates": int(merged["date"].nunique()),
        "official_files": len(list(RUN_DIR.glob("*.json"))),
        "local_bar_rows": int(merged["has_local_bar"].sum()),
        "formula_observations": [
            "官方 screener 的 factor_values.brick / prev_brick 多为正数强度，不等同于本地 QuantX BrickChart 的正负红绿值。",
            "官方 brick_delta_1 与 reversal_h 基本是同一字段，并且等于 official_brick - official_prev_brick。",
            "官方候选列表里包含 score 为负的低排名项，因此候选池不是按 score 截断，而是先按反转公式出池再由 v11 score 排序。",
            "本地 qlib/raw QMT 的 close/change 与官方 screener 在部分标的上不完全一致，公式对齐必须同时处理数据口径差异。",
        ],
        "important_corr": important_corr,
        "field_consistency": {
            "delta_matches_reversal_ratio": ratio(merged["official_delta_matches_reversal"]),
            "delta_equals_brick_minus_prev_ratio": ratio(merged["official_delta_equals_brick_minus_prev"]),
            "positive_delta_ratio": ratio(merged["official_candidate_positive_delta"]),
            "local_green_to_red_overlap_ratio": ratio(merged["local_green_to_red"]),
            "local_red_current_overlap_ratio": ratio(merged["local_red_current"]),
        },
        "close_gap_summary": describe(merged["official_close_gap_pct"]),
        "change_gap_summary": describe(merged["official_vs_local_change_gap"]),
        "by_date_tail20": by_date[-20:],
        "mismatch_examples": merged[
            merged["has_local_bar"] & (~merged["local_green_to_red"])
        ][[
            "date", "instrument", "name", "rank", "official_score", "official_brick", "official_prev_brick",
            "official_brick_delta_1", "official_close", "$close", "brick", "brick_prev", "ret1",
        ]].head(80).replace({np.nan: None}).to_dict("records"),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (ROOT / "renko_formula_alignment_rows.parquet").write_bytes(b"")
    merged.to_parquet(ROOT / "renko_formula_alignment_rows.parquet", index=False)
    print(json.dumps({
        "wrote": str(OUT),
        "rows": report["rows"],
        "dates": report["dates"],
        "delta_matches_reversal_ratio": report["field_consistency"]["delta_matches_reversal_ratio"],
        "delta_equals_brick_minus_prev_ratio": report["field_consistency"]["delta_equals_brick_minus_prev_ratio"],
        "local_green_to_red_overlap_ratio": report["field_consistency"]["local_green_to_red_overlap_ratio"],
        "important_corr": important_corr,
    }, ensure_ascii=False, indent=2))
    return 0


def ratio(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.astype(bool).mean()) if len(valid) else 0.0


def finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if len(values) else None


def describe(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "max_abs": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "max_abs": float(values.abs().max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
