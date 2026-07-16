from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
PANEL_PATH = ROOT / "panel_all_a_20130101_20260715.parquet"
OUT_ROWS = ROOT / "active_value_reversal_correlation_rows.parquet"
OUT_JSON = ROOT / "active_value_reversal_correlation.json"
OUT_MD = ROOT / "active_value_reversal_correlation.md"

TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-02")
HORIZONS = (1, 3, 5, 10)


@dataclass
class CorrelationResult:
    ok: bool
    input_panel: str
    output_rows: str
    rows: int
    summary: dict
    correlations: list[dict]
    quantiles: list[dict]
    notes: list[str]


def main() -> int:
    frame = load_panel(PANEL_PATH)
    rows = build_reversal_rows(frame)
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT_ROWS, index=False)

    summary = build_summary(rows)
    correlations = build_correlations(rows)
    quantiles = build_quantiles(rows)
    result = CorrelationResult(
        ok=True,
        input_panel=str(PANEL_PATH),
        output_rows=str(OUT_ROWS),
        rows=int(len(rows)),
        summary=summary,
        correlations=correlations,
        quantiles=quantiles,
        notes=[
            "active_value is the local proxy sum_all_a($amount), not the real 0AMV OHLC series.",
            "reversal_value is frontend_delta on the signal day.",
            "reversal_prev_down requires frontend_delta_prev < 0 and frontend_delta > 0.",
            "reversal_decline_prev5 requires at least one negative frontend_delta in the previous five trading days and frontend_delta > 0.",
            "Forward returns buy at T+1 open and mark to T+N close, without dynamic exits.",
        ],
    )
    OUT_JSON.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def load_panel(path: Path) -> pd.DataFrame:
    columns = [
        "instrument",
        "datetime",
        "$open",
        "$close",
        "$volume",
        "$amount",
        "active_value",
        "active_ret1",
        "active_ret2",
        "active_gate",
        "active_days_since_start",
        "frontend_brick",
        "frontend_delta",
        "frontend_delta_prev",
        "is_current_risk_name",
        "st_like_limit_history",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values(["instrument", "datetime"]).reset_index(drop=True)


def build_reversal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    grouped = df.groupby("instrument", group_keys=False, sort=False)
    next_open = grouped["$open"].shift(-1)
    for horizon in HORIZONS:
        future_close = grouped["$close"].shift(-int(horizon))
        df[f"fwd{horizon}"] = future_close / next_open - 1.0

    down = df["frontend_delta"].lt(0).fillna(False)
    df["prev5_down_count"] = down.groupby(df["instrument"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    df["reversal_value"] = df["frontend_delta"]
    df["reversal_ratio_prev_abs"] = df["frontend_delta"] / df["frontend_delta_prev"].abs().replace(0, np.nan)

    tradable = (
        (df["$volume"] > 0)
        & (df["$amount"] > 0)
        & (~df["is_current_risk_name"].fillna(False))
        & (~df["st_like_limit_history"].fillna(False))
    )
    positive_delta = df["frontend_delta"] > 0
    df["reversal_prev_down"] = (df["frontend_delta_prev"] < 0) & positive_delta
    df["reversal_decline_prev5"] = (df["prev5_down_count"] > 0) & positive_delta
    df["reversal_delta_positive"] = positive_delta
    df["active_up_day"] = df["active_ret1"] > 0
    df["active_strong_up_day"] = (df["active_ret1"] >= 0.04) | (df["active_ret2"] >= 0.04)
    df["active_above_ma10_wave"] = df["active_gate"].fillna(False)

    keep = tradable & positive_delta
    cols = [
        "datetime",
        "instrument",
        "active_value",
        "active_ret1",
        "active_ret2",
        "active_up_day",
        "active_strong_up_day",
        "active_above_ma10_wave",
        "active_days_since_start",
        "frontend_brick",
        "frontend_delta",
        "frontend_delta_prev",
        "prev5_down_count",
        "reversal_value",
        "reversal_ratio_prev_abs",
        "reversal_prev_down",
        "reversal_decline_prev5",
        "reversal_delta_positive",
        *[f"fwd{h}" for h in HORIZONS],
    ]
    rows = df.loc[keep, cols].copy()
    rows["date"] = rows["datetime"].dt.strftime("%Y-%m-%d")
    rows["year"] = rows["datetime"].dt.year
    rows["sample"] = np.where(rows["datetime"] >= OOS_START, "oos", np.where(rows["datetime"] <= TRAIN_END, "train", "gap"))
    return rows.replace([np.inf, -np.inf], np.nan)


def build_summary(rows: pd.DataFrame) -> dict:
    out = {"rows": int(len(rows)), "dates": int(rows["datetime"].nunique()) if len(rows) else 0}
    for active_name in ["all", "active_up_day", "active_strong_up_day", "active_above_ma10_wave"]:
        active_rows = rows if active_name == "all" else rows[rows[active_name].fillna(False)]
        out[active_name] = describe(active_rows)
    return out


def build_correlations(rows: pd.DataFrame) -> list[dict]:
    records = []
    active_filters = ["all", "active_up_day", "active_strong_up_day", "active_above_ma10_wave"]
    reversal_filters = ["reversal_delta_positive", "reversal_prev_down", "reversal_decline_prev5"]
    value_cols = ["reversal_value", "frontend_brick", "reversal_ratio_prev_abs"]
    samples = ["all", "train", "oos", "2025", "2026"]
    for active_filter in active_filters:
        active_rows = rows if active_filter == "all" else rows[rows[active_filter].fillna(False)]
        for reversal_filter in reversal_filters:
            reversal_rows = active_rows[active_rows[reversal_filter].fillna(False)]
            for sample in samples:
                sample_rows = filter_sample(reversal_rows, sample)
                for value_col in value_cols:
                    for horizon in HORIZONS:
                        target = f"fwd{horizon}"
                        records.append(corr_record(sample_rows, active_filter, reversal_filter, sample, value_col, target))
    return records


def build_quantiles(rows: pd.DataFrame) -> list[dict]:
    records = []
    active_filters = ["active_up_day", "active_strong_up_day", "active_above_ma10_wave"]
    reversal_filters = ["reversal_prev_down", "reversal_decline_prev5", "reversal_delta_positive"]
    samples = ["train", "oos", "2025", "2026"]
    for active_filter in active_filters:
        active_rows = rows[rows[active_filter].fillna(False)]
        for reversal_filter in reversal_filters:
            reversal_rows = active_rows[active_rows[reversal_filter].fillna(False)]
            for sample in samples:
                sample_rows = filter_sample(reversal_rows, sample)
                for horizon in HORIZONS:
                    records.extend(quantile_records(sample_rows, active_filter, reversal_filter, sample, "reversal_value", f"fwd{horizon}"))
    return records


def corr_record(frame: pd.DataFrame, active_filter: str, reversal_filter: str, sample: str, value_col: str, target: str) -> dict:
    data = frame[[value_col, target]].dropna()
    if len(data) < 20 or data[value_col].nunique() < 2:
        pearson = None
        spearman = None
    else:
        pearson = float(data[value_col].corr(data[target], method="pearson"))
        spearman = float(data[value_col].corr(data[target], method="spearman"))
    return {
        "active_filter": active_filter,
        "reversal_filter": reversal_filter,
        "sample": sample,
        "value": value_col,
        "target": target,
        "rows": int(len(data)),
        "pearson": pearson,
        "spearman": spearman,
        "target_mean": safe_mean(data[target]),
        "target_median": safe_median(data[target]),
        "target_win": safe_mean(data[target] > 0) if len(data) else None,
    }


def quantile_records(frame: pd.DataFrame, active_filter: str, reversal_filter: str, sample: str, value_col: str, target: str) -> list[dict]:
    data = frame[[value_col, target]].dropna().copy()
    if len(data) < 50 or data[value_col].nunique() < 5:
        return []
    try:
        data["bucket"] = pd.qcut(data[value_col], q=5, duplicates="drop")
    except ValueError:
        return []
    records = []
    for bucket, group in data.groupby("bucket", observed=True):
        records.append({
            "active_filter": active_filter,
            "reversal_filter": reversal_filter,
            "sample": sample,
            "value": value_col,
            "target": target,
            "bucket": str(bucket),
            "rows": int(len(group)),
            "value_mean": safe_mean(group[value_col]),
            "target_mean": safe_mean(group[target]),
            "target_median": safe_median(group[target]),
            "target_win": safe_mean(group[target] > 0),
        })
    return records


def filter_sample(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "all":
        return frame
    if sample in {"train", "oos"}:
        return frame[frame["sample"] == sample]
    if sample in {"2025", "2026"}:
        return frame[frame["year"] == int(sample)]
    return frame.iloc[0:0]


def describe(frame: pd.DataFrame) -> dict:
    out = {"rows": int(len(frame)), "dates": int(frame["datetime"].nunique()) if len(frame) else 0}
    for horizon in HORIZONS:
        col = f"fwd{horizon}"
        out[f"{col}_mean"] = safe_mean(frame.get(col))
        out[f"{col}_median"] = safe_median(frame.get(col))
        out[f"{col}_win"] = safe_mean(frame[col] > 0) if col in frame else None
    return out


def safe_mean(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.mean()) if len(series) else None


def safe_median(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.median()) if len(series) else None


def render_markdown(result: CorrelationResult) -> str:
    data = asdict(result)
    lines = [
        "# Active Value Reversal Correlation",
        "",
        f"Input panel: `{data['input_panel']}`",
        f"Output rows: `{data['output_rows']}`",
        f"Rows: {data['rows']}",
        "",
        "## Summary",
        "",
    ]
    for name, payload in data["summary"].items():
        if not isinstance(payload, dict):
            lines.append(f"- {name}: {payload}")
            continue
        lines.append(f"- {name}: rows={payload.get('rows')}, dates={payload.get('dates')}, fwd5_mean={fmt(payload.get('fwd5_mean'))}, fwd10_mean={fmt(payload.get('fwd10_mean'))}")
    lines.extend(["", "## Key Correlations", ""])
    key_rows = [
        row for row in data["correlations"]
        if row["value"] == "reversal_value"
        and row["target"] in {"fwd3", "fwd5", "fwd10"}
        and row["active_filter"] in {"active_up_day", "active_strong_up_day"}
        and row["reversal_filter"] in {"reversal_prev_down", "reversal_decline_prev5"}
        and row["sample"] in {"train", "oos", "2025", "2026"}
    ]
    for row in key_rows:
        lines.append(
            f"- {row['active_filter']} / {row['reversal_filter']} / {row['sample']} / {row['target']}: "
            f"rows={row['rows']}, pearson={fmt(row['pearson'])}, spearman={fmt(row['spearman'])}, mean={fmt(row['target_mean'])}, win={fmt(row['target_win'])}"
        )
    lines.extend(["", "## Quantile Check", ""])
    key_quantiles = [
        row for row in data["quantiles"]
        if row["target"] in {"fwd5", "fwd10"}
        and row["active_filter"] in {"active_up_day", "active_strong_up_day"}
        and row["reversal_filter"] == "reversal_prev_down"
        and row["sample"] in {"oos", "2025", "2026"}
    ]
    for row in key_quantiles:
        lines.append(
            f"- {row['active_filter']} / {row['sample']} / {row['target']} / {row['bucket']}: "
            f"rows={row['rows']}, value_mean={fmt(row['value_mean'])}, target_mean={fmt(row['target_mean'])}, win={fmt(row['target_win'])}"
        )
    lines.extend(["", "## Notes", ""])
    for note in data["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def fmt(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
