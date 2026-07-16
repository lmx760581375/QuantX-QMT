"""State/factor diagnostics for buyable non-ST f3 weak-to-strong candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUT = ROOT / "f3_buyable_state_factor_ic"
ROUND_C_SCRIPT = ROOT / "run_market_state_ranking_round_c.py"
ACTIVE_VALUE = Path("data/derived/active_value/daily.parquet")

ACTIVE_COLUMNS = (
    "active_all_amount_ret1",
    "active_all_amount_ret2",
    "active_all_amount_above_ma10",
    "active_all_amount_strong_up_day",
    "active_tradable_amount_ret1",
    "active_tradable_amount_ret2",
    "active_tradable_amount_above_ma10",
    "active_core_amount_ret1",
    "active_core_amount_ret2",
    "active_core_amount_above_ma10",
    "right_side_ratio",
    "right_side_core_ratio",
    "right_side_amount_ratio",
    "right_side_core_amount_ratio",
    "right_side_ratio_ret1",
    "right_side_core_ratio_ret1",
    "right_side_amount_ratio_ret1",
    "right_side_core_amount_ratio_ret1",
    "breadth_bull",
    "breadth_bull_start",
    "breadth_bull_age",
)

FACTOR_COLUMNS = (
    "score",
    "raw_rank_inv",
    "raw_candidate_count_inv",
    "amount_slope",
    "breadth_gap",
    "amplitude_inv",
    "rsv_short",
    "rsv_long",
    "liquidity_rank",
    "turnover_amount_rank",
    "turnover43_rank",
    *ACTIVE_COLUMNS,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    state_summary = summarize_states(data)
    factor_ic = summarize_factor_ic(data)
    factor_bins = summarize_factor_bins(data)
    state_summary.to_csv(OUT / "state_forward_returns.csv", index=False)
    factor_ic.to_csv(OUT / "state_factor_ic.csv", index=False)
    factor_bins.to_csv(OUT / "state_factor_bins.csv", index=False)
    write_report(data, state_summary, factor_ic, factor_bins)


def load_data() -> pd.DataFrame:
    round_c = load_round_c()
    candidates = round_c.load_enriched_candidates(ROOT).copy()
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"]).dt.strftime("%Y-%m-%d")
    rank = pd.to_numeric(candidates["raw_rank"], errors="coerce")
    good = candidates["regime"].eq("bull") | candidates["amount_state"].eq("hi") | candidates["breadth_state"].eq("hi")
    candidates["f3_selected"] = (rank <= np.where(good, 12, 8)) & rank.le(12)
    candidates["state3"] = candidates["regime"].astype(str) + "|" + candidates["amount_state"].astype(str) + "|" + candidates["breadth_state"].astype(str)

    active = pd.read_parquet(ACTIVE_VALUE).copy()
    active["signal_time"] = pd.to_datetime(active["date"]).dt.strftime("%Y-%m-%d")
    keep = ["signal_time", *[col for col in ACTIVE_COLUMNS if col in active.columns]]
    data = candidates.merge(active[keep], on="signal_time", how="left")
    for col in FACTOR_COLUMNS:
        if col not in data:
            data[col] = np.nan
        if col not in {"breadth_bull", "breadth_bull_start"}:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data[data["f3_selected"] & data["is_buyable_nonst"].fillna(False).astype(bool)].copy()


def load_round_c():
    spec = importlib.util.spec_from_file_location("round_c_helpers", ROUND_C_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {ROUND_C_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_states(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (3, 5, 7):
        label = f"ret_{horizon}d"
        usable = usable_label(data, horizon)
        for group_name, group_cols in (
            ("all", None),
            ("state3", ["state3"]),
            ("regime", ["regime"]),
            ("amount_state", ["amount_state"]),
            ("breadth_state", ["breadth_state"]),
            ("rank_bucket", ["rank_bucket"]),
        ):
            groups = [("all", usable)] if group_cols is None else usable.groupby(group_cols, dropna=False, sort=True)
            for group_value, group in groups:
                if len(group) < 20:
                    continue
                ret = pd.to_numeric(group[label], errors="coerce")
                rows.append({
                    "horizon": horizon,
                    "group_name": group_name,
                    "group_value": format_group_value(group_value),
                    "rows": int(len(group)),
                    "days": int(group["signal_time"].nunique()),
                    "ret_mean": safe_mean(ret),
                    "ret_median": safe_median(ret),
                    "win_rate": safe_mean((ret > 0).astype(float)),
                    "avg_raw_rank": safe_mean(pd.to_numeric(group["raw_rank"], errors="coerce")),
                })
    return pd.DataFrame(rows).sort_values(["horizon", "group_name", "ret_mean"], ascending=[True, True, False])


def summarize_factor_ic(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (3, 5, 7):
        label = f"ret_{horizon}d"
        usable = usable_label(data, horizon)
        for group_name, group_cols in (
            ("all", None),
            ("state3", ["state3"]),
            ("regime", ["regime"]),
            ("amount_state", ["amount_state"]),
            ("breadth_state", ["breadth_state"]),
        ):
            groups = [("all", usable)] if group_cols is None else usable.groupby(group_cols, dropna=False, sort=True)
            for group_value, group in groups:
                if len(group) < 30 or group["signal_time"].nunique() < 10:
                    continue
                for factor in FACTOR_COLUMNS:
                    if factor not in group:
                        continue
                    daily = daily_rank_ic(group, factor, label)
                    if daily.empty:
                        continue
                    spearman = pd.to_numeric(daily["spearman_ic"], errors="coerce").dropna()
                    rows.append({
                        "horizon": horizon,
                        "factor": factor,
                        "group_name": group_name,
                        "group_value": format_group_value(group_value),
                        "rows": int(len(group)),
                        "days": int(group["signal_time"].nunique()),
                        "ic_days": int(len(spearman)),
                        "spearman_mean": safe_mean(spearman),
                        "spearman_median": safe_median(spearman),
                        "spearman_ir": safe_ir(spearman),
                        "spearman_pos_ratio": safe_mean((spearman > 0).astype(float)),
                        "ret_mean": safe_mean(pd.to_numeric(group[label], errors="coerce")),
                        "score_abs": abs(safe_mean(spearman)) * np.log1p(len(spearman)),
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["horizon", "score_abs"], ascending=[True, False])
    return out


def summarize_factor_bins(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (3, 5, 7):
        label = f"ret_{horizon}d"
        usable = usable_label(data, horizon)
        for factor in FACTOR_COLUMNS:
            if factor not in usable:
                continue
            for state, group in usable.groupby("state3", dropna=False, sort=True):
                if len(group) < 40:
                    continue
                rows.extend(factor_bin_rows(group, factor, label, horizon, str(state)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["horizon", "edge_high_minus_low"], ascending=[True, False])
    return out


def usable_label(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    label = f"ret_{horizon}d"
    label_ok = f"label_available_{horizon}d"
    out = data[data[label_ok].fillna(False).astype(bool)].copy()
    out[label] = pd.to_numeric(out[label], errors="coerce")
    return out.dropna(subset=[label])


def daily_rank_ic(group: pd.DataFrame, factor: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_time, day in group.groupby("signal_time", sort=True):
        day = day[[factor, label]].copy()
        day[factor] = pd.to_numeric(day[factor], errors="coerce")
        day[label] = pd.to_numeric(day[label], errors="coerce")
        day = day.dropna()
        if len(day) < 3 or day[factor].nunique() < 2 or day[label].nunique() < 2:
            continue
        rows.append({
            "signal_time": signal_time,
            "spearman_ic": day[factor].corr(day[label], method="spearman"),
        })
    return pd.DataFrame(rows)


def factor_bin_rows(group: pd.DataFrame, factor: str, label: str, horizon: int, state: str) -> list[dict[str, Any]]:
    frame = group[["signal_time", factor, label]].copy()
    frame[factor] = pd.to_numeric(frame[factor], errors="coerce")
    frame[label] = pd.to_numeric(frame[label], errors="coerce")
    frame = frame.dropna()
    if len(frame) < 40 or frame[factor].nunique() < 4:
        return []
    try:
        frame["bucket"] = pd.qcut(frame[factor].rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
    except ValueError:
        return []
    bucketed = frame.groupby("bucket", observed=False)
    means = bucketed[label].mean()
    if "q1_low" not in means or "q4_high" not in means:
        return []
    edge = float(means["q4_high"] - means["q1_low"])
    rows = []
    for bucket, bucket_group in bucketed:
        ret = pd.to_numeric(bucket_group[label], errors="coerce")
        rows.append({
            "horizon": horizon,
            "state3": state,
            "factor": factor,
            "bucket": str(bucket),
            "rows": int(len(bucket_group)),
            "days": int(bucket_group["signal_time"].nunique()),
            "ret_mean": safe_mean(ret),
            "ret_median": safe_median(ret),
            "win_rate": safe_mean((ret > 0).astype(float)),
            "edge_high_minus_low": edge,
        })
    return rows


def write_report(data: pd.DataFrame, states: pd.DataFrame, ic: pd.DataFrame, bins: pd.DataFrame) -> None:
    lines = [
        "# F3 Buyable State Factor IC",
        "",
        f"- f3 buyable non-ST rows: {len(data)}",
        f"- signal days: {data['signal_time'].nunique()}",
        f"- states: {data['state3'].nunique()}",
        "",
        "## Best State Forward Returns",
        "",
    ]
    for _, row in states[(states["group_name"] == "state3") & (states["horizon"] == 5)].head(12).iterrows():
        lines.append(f"- {row['group_value']}: ret5={row['ret_mean']:.4f}, win={row['win_rate']:.4f}, rows={int(row['rows'])}, days={int(row['days'])}")
    lines.extend(["", "## Strongest State Factor IC", ""])
    top_ic = ic[(ic["horizon"] == 5) & (ic["group_name"] == "state3") & (ic["ic_days"] >= 10)].head(20)
    for _, row in top_ic.iterrows():
        lines.append(f"- {row['group_value']} / {row['factor']}: ic5={row['spearman_mean']:.4f}, ir={row['spearman_ir']:.4f}, pos={row['spearman_pos_ratio']:.4f}, days={int(row['ic_days'])}")
    lines.extend(["", "## Strongest Factor Bin Edges", ""])
    edges = bins[(bins["horizon"] == 5) & (bins["bucket"] == "q4_high")].sort_values("edge_high_minus_low", ascending=False).head(20)
    for _, row in edges.iterrows():
        lines.append(f"- {row['state3']} / {row['factor']}: q4_ret5={row['ret_mean']:.4f}, edge={row['edge_high_minus_low']:.4f}, rows={int(row['rows'])}")
    lines.extend(["", "## Artifacts", "", "- `state_forward_returns.csv`", "- `state_factor_ic.csv`", "- `state_factor_bins.csv`"])
    (OUT / "F3_BUYABLE_STATE_FACTOR_IC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_group_value(value: object) -> str:
    if isinstance(value, tuple):
        return "|".join(map(str, value))
    return str(value)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def safe_ir(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    std = values.std(ddof=0)
    if values.empty or not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(values.mean() / std)


if __name__ == "__main__":
    main()
