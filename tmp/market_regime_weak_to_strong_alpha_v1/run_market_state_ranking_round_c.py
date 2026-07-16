"""Market-state ranking Round C for weak-to-strong capacity alpha.

This temporary research script keeps all artifacts under ./tmp and uses the
formal QuantX backtest engine for validation. It enriches the original
weak-to-strong raw candidates with existing market-state/candidate features,
filters non-buyable/ST rows for IC diagnostics, then emits a small set of
external_score variants focused on low-empty capacity expansion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "market_state_ranking_round_c"
PROVIDER_URI = "data/qlib_data_fixed"
BACKTEST_START = "2016-01-04"

SOURCE_CONFIG = Path("configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t8.yaml")
RAW_CANDIDATES = ROOT / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet"
CANDIDATE_PANEL = ROOT / "data/candidate_panel.parquet"

REFERENCE_RUNS = {
    "current_top4_pos5": ROOT
    / "capacity_extension_round_a/reference_artifacts/current_precomputed_top4_pos5",
    "raw_top12_pos12": ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12",
    "core4_pos12": ROOT / "capacity_extension_round_b_core4/artifacts/core4_pos12",
}

RULE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "c1_core4_tail5_8_amount_hi_pos8",
        "topk": 8,
        "max_positions": 8,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 8,
        "tail_condition": "amount_hi",
    },
    {
        "variant": "c2_core4_tail5_12_amount_hi_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 12,
        "tail_condition": "amount_hi",
    },
    {
        "variant": "c3_core4_tail5_8_bull_amount_hi_pos8",
        "topk": 8,
        "max_positions": 8,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 8,
        "tail_condition": "bull_amount_hi",
    },
    {
        "variant": "c4_core4_tail5_12_bull_amount_hi_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 12,
        "tail_condition": "bull_amount_hi",
    },
    {
        "variant": "c5_dynamic_top4_8_amount_pos8",
        "topk": 8,
        "max_positions": 8,
        "mode": "dynamic_amount",
        "hi_rank": 8,
        "normal_rank": 4,
    },
    {
        "variant": "c6_dynamic_top4_12_amount_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "dynamic_amount",
        "hi_rank": 12,
        "normal_rank": 4,
    },
    {
        "variant": "c7_core4_tail5_8_amount_or_bull_pos8",
        "topk": 8,
        "max_positions": 8,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 8,
        "tail_condition": "amount_hi_or_bull",
    },
    {
        "variant": "c8_core4_tail5_12_amount_or_bull_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "core_tail",
        "tail_min": 5,
        "tail_max": 12,
        "tail_condition": "amount_hi_or_bull",
    },
    {
        "variant": "c9_core2_bear_lo_else_core4_tail12_amount_hi_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "bear_lo_risk",
    },
    {
        "variant": "c10_ic_weighted_core4_tail12_pos12",
        "topk": 12,
        "max_positions": 12,
        "mode": "ic_weighted",
    },
)

IC_FACTORS = (
    "score",
    "raw_rank_inv",
    "candidate_rank_inv",
    "tradable_rank_inv",
    "amount_slope",
    "breadth_gap",
    "raw_candidate_count_inv",
    "amplitude_inv",
    "rsv_short",
    "rsv_long",
    "position_rank",
    "position_strength",
    "repair_rank",
    "ret3_rank",
    "ret5_rank",
    "ret20_rank",
    "liquidity_rank",
    "turnover_amount_rank",
    "turnover20_rank",
    "turnover43_rank",
    "trend_bonus",
)

ML_FEATURES = (
    "score",
    "raw_rank_inv",
    "amount_slope",
    "breadth_gap",
    "raw_candidate_count_inv",
    "amplitude_inv",
    "rsv_short",
    "rsv_long",
    "position_rank",
    "position_strength",
    "repair_rank",
    "ret3_rank",
    "ret5_rank",
    "ret20_rank",
    "liquidity_rank",
    "turnover_amount_rank",
    "turnover20_rank",
    "turnover43_rank",
    "trend_bonus",
    "amount_hi_flag",
    "amount_lo_flag",
    "breadth_hi_flag",
    "breadth_lo_flag",
    "regime_bull",
    "regime_bear",
    "regime_range",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--build-only", action="store_true", help="Only generate diagnostics, signals, and configs.")
    parser.add_argument("--dry-run", action="store_true", help="Compile configs through run_backtest --dry-run.")
    parser.add_argument("--analyze-only", action="store_true", help="Only collect existing artifacts into reports.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip finished backtest artifacts.")
    parser.add_argument("--no-ml", action="store_true", help="Skip walk-forward LightGBM signal variants.")
    parser.add_argument(
        "--python-cmd",
        nargs="+",
        default=["conda", "run", "--no-capture-output", "-n", "test", "python"],
        help="Python command prefix used to invoke quantx.tools.run_backtest.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / OUTPUT_NAME
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        candidates = load_enriched_candidates(root)
        ic_summary, ic_daily = analyze_ic(candidates)
        ic_summary.to_csv(runs_dir / "round_c_ic_summary.csv", index=False)
        ic_daily.to_csv(runs_dir / "round_c_ic_daily.csv", index=False)

        variants = list(RULE_VARIANTS)
        signals = build_rule_signals(candidates, variants, signals_dir, runs_dir)
        if not args.no_ml:
            ml_variants, ml_signals = build_ml_signals(candidates, signals_dir, runs_dir)
            variants.extend(ml_variants)
            signals.extend(ml_signals)

        configs = build_configs(signals, configs_dir)
        manifest = {
            "experiment": OUTPUT_NAME,
            "source_config": str(SOURCE_CONFIG),
            "raw_candidates": str(root / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet"),
            "candidate_panel": str(root / "data/candidate_panel.parquet"),
            "backtest_start": BACKTEST_START,
            "execution_lag": 1,
            "deal_price": "close",
            "buyable_filter_for_ic": buyable_filter_description(),
            "ic_factors": list(IC_FACTORS),
            "ml_features": [] if args.no_ml else list(ML_FEATURES),
            "signals": signals,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_backtest_report(ic_summary, signals, reports_dir / "MARKET_STATE_RANKING_ROUND_C_PRE.md")

        if args.build_only:
            print(f"[round_c] build-only candidates={len(candidates)} signals={len(signals)} configs={len(configs)}", flush=True)
            return

        run_rows = run_configs(
            configs,
            artifacts_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            python_cmd=args.python_cmd,
        )
        pd.DataFrame(run_rows).to_csv(
            runs_dir / ("dry_run_summary.csv" if args.dry_run else "backtest_summary.csv"),
            index=False,
        )
        if args.dry_run:
            return

    collect_artifacts(artifacts_dir, runs_dir, reports_dir)


def load_enriched_candidates(root: Path) -> pd.DataFrame:
    raw = pd.read_parquet(root / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet")
    raw = raw.copy()
    raw["signal_time"] = pd.to_datetime(raw["signal_time"]).dt.strftime("%Y-%m-%d")
    raw["instrument"] = raw["instrument"].astype(str)
    raw = raw.sort_values(["signal_time", "raw_rank"]).reset_index(drop=True)

    panel = pd.read_parquet(root / "data/candidate_panel.parquet")
    panel = panel.copy()
    panel["signal_time"] = pd.to_datetime(panel["signal_time"]).dt.strftime("%Y-%m-%d")
    panel["instrument"] = panel["instrument"].astype(str)
    panel = prefer_core_panel_rows(panel)
    panel = add_point_in_time_states(panel)

    merged = raw.merge(panel, on=["signal_time", "instrument"], how="left", suffixes=("_raw", ""))
    merged["signal_time_dt"] = pd.to_datetime(merged["signal_time"])
    merged["year"] = merged["signal_time_dt"].dt.year.astype(int)
    merged["score"] = pd.to_numeric(merged["score_raw"], errors="coerce").fillna(pd.to_numeric(merged.get("score"), errors="coerce"))
    merged["raw_rank"] = pd.to_numeric(merged["raw_rank"], errors="coerce")
    merged["rank_bucket"] = pd.cut(
        merged["raw_rank"],
        bins=[0, 4, 8, 12, 9999],
        labels=["rank1_4", "rank5_8", "rank9_12", "rank13_plus"],
    ).astype(str)
    merged["amount_state"] = merged["amount_state"].fillna("unknown")
    merged["breadth_state"] = merged["breadth_state"].fillna("unknown")
    merged["regime"] = merged["regime"].fillna("unknown")
    merged = add_derived_factor_columns(merged)
    merged["is_buyable_nonst"] = buyable_mask(merged)
    return merged.sort_values(["signal_time", "raw_rank", "instrument"]).reset_index(drop=True)


def prefer_core_panel_rows(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    layer_priority = {"core": 0, "shape": 1, "mid": 2, "wide": 3}
    out["_layer_priority"] = out["layer"].map(layer_priority).fillna(9).astype(int)
    out["_rank_priority"] = pd.to_numeric(out.get("candidate_rank"), errors="coerce").fillna(999999)
    out = out.sort_values(["signal_time", "instrument", "_layer_priority", "_rank_priority"])
    out = out.drop_duplicates(["signal_time", "instrument"], keep="first")
    return out.drop(columns=["_layer_priority", "_rank_priority"])


def add_point_in_time_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    daily = (
        out.dropna(subset=["amount_slope", "breadth_gap"])
        .drop_duplicates("signal_time")[["signal_time", "amount_slope", "breadth_gap", "regime"]]
        .sort_values("signal_time")
        .reset_index(drop=True)
    )
    for column in ("amount_slope", "breadth_gap"):
        values = pd.to_numeric(daily[column], errors="coerce")
        daily[f"{column}_q25"] = values.expanding(min_periods=120).quantile(0.25).shift(1)
        daily[f"{column}_q75"] = values.expanding(min_periods=120).quantile(0.75).shift(1)
    daily["amount_state"] = state_from_quantiles(daily, "amount_slope")
    daily["breadth_state"] = state_from_quantiles(daily, "breadth_gap")
    daily["amount_hi_flag"] = (daily["amount_state"] == "hi").astype(float)
    daily["amount_lo_flag"] = (daily["amount_state"] == "lo").astype(float)
    daily["breadth_hi_flag"] = (daily["breadth_state"] == "hi").astype(float)
    daily["breadth_lo_flag"] = (daily["breadth_state"] == "lo").astype(float)
    state_cols = [
        "signal_time",
        "amount_state",
        "breadth_state",
        "amount_hi_flag",
        "amount_lo_flag",
        "breadth_hi_flag",
        "breadth_lo_flag",
    ]
    out = out.drop(columns=[col for col in state_cols if col != "signal_time" and col in out.columns], errors="ignore")
    return out.merge(daily[state_cols], on="signal_time", how="left")


def state_from_quantiles(daily: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(daily[column], errors="coerce")
    q25 = pd.to_numeric(daily[f"{column}_q25"], errors="coerce")
    q75 = pd.to_numeric(daily[f"{column}_q75"], errors="coerce")
    state = np.where(values >= q75, "hi", np.where(values <= q25, "lo", "mid"))
    state = pd.Series(state, index=daily.index, dtype="object")
    state[q25.isna() | q75.isna() | values.isna()] = "unknown"
    return state


def add_derived_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["raw_rank_inv"] = -pd.to_numeric(out["raw_rank"], errors="coerce")
    out["candidate_rank_inv"] = -pd.to_numeric(out.get("candidate_rank"), errors="coerce")
    out["tradable_rank_inv"] = -pd.to_numeric(out.get("tradable_rank"), errors="coerce")
    out["raw_candidate_count_inv"] = -pd.to_numeric(out.get("raw_candidate_count"), errors="coerce")
    out["amplitude_inv"] = -pd.to_numeric(out.get("amplitude"), errors="coerce")
    for source, target in (
        ("turnover_amount", "turnover_amount_rank"),
        ("turnover20", "turnover20_rank"),
        ("turnover43", "turnover43_rank"),
    ):
        values = pd.to_numeric(out.get(source), errors="coerce")
        out[target] = out.groupby("signal_time", sort=False)[source].transform(
            lambda s: pd.to_numeric(s, errors="coerce").rank(pct=True)
        ) if source in out else np.nan
        out.loc[values.isna(), target] = np.nan
    for regime in ("bull", "bear", "range"):
        out[f"regime_{regime}"] = (out["regime"] == regime).astype(float)
    for column in ("amount_hi_flag", "amount_lo_flag", "breadth_hi_flag", "breadth_lo_flag"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce").fillna(0.0)
    return out


def buyable_mask(frame: pd.DataFrame) -> pd.Series:
    def bool_col(name: str, default: bool) -> pd.Series:
        if name not in frame:
            return pd.Series(default, index=frame.index)
        return frame[name].fillna(default).astype(bool)

    return (
        bool_col("is_tradable", False)
        & bool_col("entry_has_bar", False)
        & ~bool_col("is_st_signal", True)
        & ~bool_col("is_st_entry", True)
        & ~bool_col("entry_zero_volume", True)
        & ~bool_col("entry_zero_amount", True)
        & ~bool_col("entry_one_price_limit_up", True)
    )


def buyable_filter_description() -> str:
    return (
        "is_tradable & entry_has_bar & not is_st_signal & not is_st_entry & "
        "not entry_zero_volume & not entry_zero_amount & not entry_one_price_limit_up"
    )


def analyze_ic(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    base = candidates[candidates["is_buyable_nonst"]].copy()
    group_specs = [
        ("all", None),
        ("regime", "regime"),
        ("amount_state", "amount_state"),
        ("breadth_state", "breadth_state"),
        ("rank_bucket", "rank_bucket"),
        ("regime_x_amount", ["regime", "amount_state"]),
        ("rank_x_amount", ["rank_bucket", "amount_state"]),
    ]
    for horizon in (3, 5, 7):
        label = f"ret_{horizon}d"
        label_ok = f"label_available_{horizon}d"
        usable = base[base.get(label_ok, True).fillna(False)].copy() if label_ok in base else base.copy()
        usable[label] = pd.to_numeric(usable[label], errors="coerce")
        usable = usable.dropna(subset=[label])
        for factor in IC_FACTORS:
            if factor not in usable:
                continue
            factor_frame = usable[["signal_time", label, factor, "regime", "amount_state", "breadth_state", "rank_bucket"]].copy()
            factor_frame[factor] = pd.to_numeric(factor_frame[factor], errors="coerce")
            factor_frame = factor_frame.dropna(subset=[factor, label])
            if factor_frame.empty:
                continue
            for group_name, group_cols in group_specs:
                grouped = [("all", factor_frame)] if group_cols is None else factor_frame.groupby(group_cols, dropna=False, sort=True)
                for group_value, group in grouped:
                    if len(group) < 30 or group["signal_time"].nunique() < 10:
                        continue
                    daily_ic = daily_rank_ic(group, factor, label)
                    if daily_ic.empty:
                        continue
                    for _, ic_row in daily_ic.iterrows():
                        daily_rows.append({
                            "horizon": horizon,
                            "factor": factor,
                            "group_name": group_name,
                            "group_value": format_group_value(group_value),
                            "signal_time": ic_row["signal_time"],
                            "spearman_ic": ic_row["spearman_ic"],
                            "pearson_ic": ic_row["pearson_ic"],
                            "n": ic_row["n"],
                        })
                    rows.append(summarize_ic_daily(horizon, factor, group_name, group_value, group, daily_ic, label))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["horizon", "group_name", "ic_mean_abs_rank"], ascending=[True, True, False])
    return summary, pd.DataFrame(daily_rows)


def daily_rank_ic(group: pd.DataFrame, factor: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_time, day in group.groupby("signal_time", sort=True):
        day = day[[factor, label]].dropna()
        if len(day) < 3 or day[factor].nunique() < 2 or day[label].nunique() < 2:
            continue
        rows.append({
            "signal_time": signal_time,
            "spearman_ic": day[factor].corr(day[label], method="spearman"),
            "pearson_ic": day[factor].corr(day[label], method="pearson"),
            "n": int(len(day)),
        })
    return pd.DataFrame(rows)


def summarize_ic_daily(
    horizon: int,
    factor: str,
    group_name: str,
    group_value: object,
    group: pd.DataFrame,
    daily_ic: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    spearman = pd.to_numeric(daily_ic["spearman_ic"], errors="coerce").dropna()
    pearson = pd.to_numeric(daily_ic["pearson_ic"], errors="coerce").dropna()
    return {
        "horizon": int(horizon),
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
        "pearson_mean": safe_mean(pearson),
        "ret_mean": safe_mean(pd.to_numeric(group[label], errors="coerce")),
        "ret_median": safe_median(pd.to_numeric(group[label], errors="coerce")),
        "win_rate": safe_mean((pd.to_numeric(group[label], errors="coerce") > 0).astype(float)),
        "ic_mean_abs_rank": abs(safe_mean(spearman)),
    }


def build_rule_signals(
    candidates: pd.DataFrame,
    variants: list[dict[str, Any]],
    signals_dir: Path,
    runs_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signal_stats: list[dict[str, Any]] = []
    for spec in variants:
        selected = select_rule_variant(candidates, spec).copy()
        selected["score"] = rule_score(selected, spec)
        selected = selected.dropna(subset=["score", "signal_time", "instrument"])
        selected = selected.sort_values(["signal_time", "score"], ascending=[True, False])
        path = signals_dir / f"{spec['variant']}.parquet"
        selected[["signal_time", "instrument", "score", "raw_rank"]].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        rows.append(manifest)
        signal_stats.append({**manifest, **variant_signal_stats(selected)})
    pd.DataFrame(signal_stats).to_csv(runs_dir / "round_c_signal_stats.csv", index=False)
    return rows


def select_rule_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rank = pd.to_numeric(candidates["raw_rank"], errors="coerce")
    core4 = rank <= 4
    mode = spec["mode"]
    if mode == "core_tail":
        tail = rank.between(int(spec["tail_min"]), int(spec["tail_max"])) & tail_condition(candidates, spec["tail_condition"])
        return candidates[core4 | tail].copy()
    if mode == "dynamic_amount":
        hi = candidates["amount_state"].eq("hi")
        selected = (hi & (rank <= int(spec["hi_rank"]))) | (~hi & (rank <= int(spec["normal_rank"])))
        return candidates[selected].copy()
    if mode == "bear_lo_risk":
        risk = candidates["regime"].eq("bear") & candidates["amount_state"].eq("lo")
        selected = (risk & (rank <= 2)) | (~risk & ((rank <= 4) | (rank.between(5, 12) & candidates["amount_state"].eq("hi"))))
        return candidates[selected].copy()
    if mode == "ic_weighted":
        return candidates[rank <= 12].copy()
    raise ValueError(f"Unknown rule mode: {mode}")


def tail_condition(candidates: pd.DataFrame, condition: str) -> pd.Series:
    amount_hi = candidates["amount_state"].eq("hi")
    bull = candidates["regime"].eq("bull")
    if condition == "amount_hi":
        return amount_hi
    if condition == "bull_amount_hi":
        return bull & amount_hi
    if condition == "amount_hi_or_bull":
        return amount_hi | bull
    raise ValueError(f"Unknown tail condition: {condition}")


def rule_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    base = pd.to_numeric(selected["score"], errors="coerce")
    if spec["mode"] != "ic_weighted":
        return base
    score = base.fillna(0.0)
    score = score + 0.20 * zscore(selected["position_rank"])
    score = score + 0.15 * zscore(selected["repair_rank"])
    score = score + 0.10 * zscore(selected["ret5_rank"])
    score = score + 0.10 * zscore(selected["liquidity_rank"])
    score = score - 0.15 * zscore(selected["amplitude"])
    score = score + 0.08 * pd.to_numeric(selected["amount_hi_flag"], errors="coerce").fillna(0.0)
    score = score + 0.05 * selected["regime"].eq("bull").astype(float)
    score = score - 0.02 * pd.to_numeric(selected["raw_rank"], errors="coerce").fillna(99.0)
    return score


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((values - mean) / std).fillna(0.0)


def build_ml_signals(candidates: pd.DataFrame, signals_dir: Path, runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import lightgbm as lgb
    except Exception as exc:
        print(f"[round_c] skip ML variants because LightGBM is unavailable: {exc}", flush=True)
        return [], []

    data = candidates.copy()
    data["ml_score"] = np.nan
    train_mask = data["is_buyable_nonst"] & data.get("label_available_5d", False).fillna(False)
    for column in ML_FEATURES:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ret_5d"] = pd.to_numeric(data["ret_5d"], errors="coerce")
    years = sorted(int(y) for y in data["year"].dropna().unique() if int(y) >= 2021)
    diag_rows: list[dict[str, Any]] = []
    for year in years:
        train = data[train_mask & (data["year"] < year)].copy()
        test = data[data["year"] == year].copy()
        if len(train) < 500 or test.empty:
            continue
        train_x = train[list(ML_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_y = train["ret_5d"].fillna(0.0)
        test_x = test[list(ML_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=20260715 + year,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        data.loc[test.index, "ml_score"] = pred
        valid_test = test.copy()
        valid_test["ml_score"] = pred
        valid_test = valid_test[valid_test["is_buyable_nonst"] & valid_test["ret_5d"].notna()]
        diag_rows.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "buyable_label_test_rows": int(len(valid_test)),
            "spearman_ic": safe_corr(valid_test["ml_score"], valid_test["ret_5d"], method="spearman"),
            "pearson_ic": safe_corr(valid_test["ml_score"], valid_test["ret_5d"], method="pearson"),
        })
    pd.DataFrame(diag_rows).to_csv(runs_dir / "round_c_ml_oos_diagnostics.csv", index=False)

    ml_specs = [
        {"variant": "ml_oos_top12_pos12", "topk": 12, "max_positions": 12, "mode": "ml_top12"},
        {"variant": "ml_oos_core4_tail12_amount_pos12", "topk": 12, "max_positions": 12, "mode": "ml_amount_tail"},
    ]
    manifests: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    rank = pd.to_numeric(data["raw_rank"], errors="coerce")
    for spec in ml_specs:
        if spec["mode"] == "ml_top12":
            selected = data[rank <= 12].copy()
        else:
            selected = data[(rank <= 4) | (rank.between(5, 12) & data["amount_state"].eq("hi"))].copy()
        fallback = pd.to_numeric(selected["score"], errors="coerce") - 0.02 * pd.to_numeric(selected["raw_rank"], errors="coerce")
        selected["score"] = pd.to_numeric(selected["ml_score"], errors="coerce").fillna(fallback)
        selected = selected.dropna(subset=["score", "signal_time", "instrument"])
        selected = selected.sort_values(["signal_time", "score"], ascending=[True, False])
        path = signals_dir / f"{spec['variant']}.parquet"
        selected[["signal_time", "instrument", "score", "raw_rank"]].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        manifest["ml_oos_first_year"] = 2021
        manifests.append(manifest)
        stats.append({**manifest, **variant_signal_stats(selected)})
    if stats:
        stats_path = runs_dir / "round_c_ml_signal_stats.csv"
        pd.DataFrame(stats).to_csv(stats_path, index=False)
    return ml_specs, manifests


def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
    signal_days = signals.groupby("signal_time", sort=True).size() if not signals.empty else pd.Series(dtype=int)
    topk = int(spec["topk"])
    return {
        "variant": str(spec["variant"]),
        "path": str(path),
        "topk": topk,
        "max_positions": int(spec["max_positions"]),
        "mode": str(spec["mode"]),
        "signal_rows": int(len(signals)),
        "signal_days": int(signal_days.size),
        "days_with_at_least_topk": int((signal_days >= topk).sum()) if not signal_days.empty else 0,
        "first_signal": str(signals["signal_time"].min()) if not signals.empty else None,
        "last_signal": str(signals["signal_time"].max()) if not signals.empty else None,
    }


def variant_signal_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"avg_daily_candidates": np.nan, "median_daily_candidates": np.nan, "max_daily_candidates": np.nan}
    counts = signals.groupby("signal_time").size()
    return {
        "avg_daily_candidates": float(counts.mean()),
        "median_daily_candidates": float(counts.median()),
        "max_daily_candidates": int(counts.max()),
    }


def build_configs(signals: list[dict[str, Any]], configs_dir: Path) -> list[Path]:
    source = read_yaml(SOURCE_CONFIG)
    paths: list[Path] = []
    for signal in signals:
        config = make_external_score_config(source, signal)
        path = configs_dir / f"{signal['variant']}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def make_external_score_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config["name"] = f"true_wts_{OUTPUT_NAME}_{signal['variant']}"
    config["title"] = "弱转强市场状态排序扩容 Round C"
    config["description"] = (
        "Tmp experiment: market-state/ranking capacity expansion for original weak-to-strong candidates. "
        "Preserves lag=1, next-session close execution, sizing, costs, and sell rules."
    )
    config["data"] = dict(source.get("data") or {})
    config["data"].update({
        "provider_uri": PROVIDER_URI,
        "universe": "external_score",
        "start": BACKTEST_START,
        "end": "latest",
    })
    config["selector"] = {
        "mode": "external_score",
        "path": signal["path"],
        "date_col": "signal_time",
        "instrument_col": "instrument",
        "score_col": "score",
        "sort": "score_desc",
        "topk": int(signal["topk"]),
        "lag": 1,
        "candidate_limit": max(20, int(signal["topk"])),
        "reason": f"true_wts_{OUTPUT_NAME}_{signal['variant']}",
    }
    config["rebalance"] = dict(source.get("rebalance") or {})
    config["rebalance"]["max_positions"] = int(signal["max_positions"])
    config["execution"] = dict(source.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["engine"] = dict(source.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "variant": signal["variant"],
        "mode": signal.get("mode"),
        "source_config": str(SOURCE_CONFIG),
        "raw_candidates": str(RAW_CANDIDATES),
        "candidate_panel": str(CANDIDATE_PANEL),
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal["signal_rows"]),
        "signal_days": int(signal["signal_days"]),
        "days_with_at_least_topk": int(signal["days_with_at_least_topk"]),
    }
    if "ml_oos_first_year" in signal:
        config["metadata"]["ml_oos_first_year"] = int(signal["ml_oos_first_year"])
    return config


def run_configs(
    configs: list[Path],
    artifacts_dir: Path,
    *,
    dry_run: bool,
    skip_existing: bool,
    python_cmd: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[round_c] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[round_c] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
        cmd = [*python_cmd, "-m", "quantx.tools.run_backtest", "--config", str(config_path), "--json"]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--output-dir", str(artifacts_dir), "--run-id", run_id])
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        row = parse_run_output(completed.stdout)
        row.update({
            "config": str(config_path),
            "run_id": run_id,
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        })
        rows.append(flatten_summary(row))
        if completed.returncode != 0:
            print(completed.stderr[-4000:], flush=True)
            raise RuntimeError(f"Backtest failed for {config_path}")
    return rows


def collect_artifacts(artifacts_dir: Path, runs_dir: Path, reports_dir: Path) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[round_c] no artifacts found under {artifacts_dir}", flush=True)
        return
    reference = load_reference_metrics()
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    crash_rows: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        run_id = artifact_dir.name
        summary = read_json(artifact_dir / "summary.json")
        metrics = read_json(artifact_dir / "metrics.json")
        config = read_yaml(artifact_dir / "config.yaml")
        metadata = dict(config.get("metadata") or {})
        nav = read_json_frame(artifact_dir / "daily_nav.json")
        row = {"run_id": run_id}
        for key in (
            "name",
            "symbols",
            "start_date",
            "end_date",
            "final_value",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "buy_count",
            "sell_count",
            "reject_count",
        ):
            row[key] = summary.get(key)
        for key in (
            "closed_position_count",
            "win_rate",
            "profit_factor",
            "avg_closed_return",
            "avg_holding_days",
            "avg_position_count",
            "max_position_count",
            "avg_capital_utilization",
            "zero_utilization_day_ratio",
        ):
            row[key] = metrics.get(key)
        row.update(metadata)
        row.update(position_stats(nav, prefix="full"))
        post2022 = nav[nav["date"].astype(str) >= "2022-01-01"] if not nav.empty and "date" in nav else nav
        row.update(position_stats(post2022, prefix="post2022"))
        row.update(window_return_stats(nav, "2022-03-31", "2022-05-05", prefix="crash20"))
        rows.append(row)
        yearly_rows.extend(summarize_yearly_nav(run_id, nav, metadata))
        crash_rows.append({"run_id": run_id, **window_return_stats(nav, "2022-03-31", "2022-05-05", prefix="crash20")})

    metrics_frame = pd.DataFrame(rows).sort_values(["max_positions", "topk", "variant"])
    metrics_frame.to_csv(runs_dir / "round_c_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "round_c_yearly_nav.csv", index=False)
    pd.DataFrame(crash_rows).to_csv(runs_dir / "round_c_crash_windows.csv", index=False)
    comparison = write_comparison(metrics_frame, reference, runs_dir / "round_c_comparison.csv")
    write_report(metrics_frame, comparison, reference, reports_dir / "MARKET_STATE_RANKING_ROUND_C.md")
    print(f"[round_c] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def load_reference_metrics() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, run_dir in REFERENCE_RUNS.items():
        summary = read_json(run_dir / "summary.json")
        metrics = read_json(run_dir / "metrics.json")
        row = {**summary, **metrics}
        nav = read_json_frame(run_dir / "daily_nav.json")
        row.update(position_stats(nav, prefix="full"))
        post2022 = nav[nav["date"].astype(str) >= "2022-01-01"] if not nav.empty and "date" in nav else nav
        row.update(position_stats(post2022, prefix="post2022"))
        row.update(window_return_stats(nav, "2022-03-31", "2022-05-05", prefix="crash20"))
        result[name] = row
    return result


def position_stats(nav: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if nav.empty or "position_count" not in nav:
        return {
            f"{prefix}_avg_position_count": np.nan,
            f"{prefix}_zero_position_ratio": np.nan,
            f"{prefix}_max_position_count": np.nan,
        }
    counts = pd.to_numeric(nav["position_count"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_avg_position_count": float(counts.mean()),
        f"{prefix}_zero_position_ratio": float((counts <= 0).mean()),
        f"{prefix}_max_position_count": float(counts.max()),
    }


def window_return_stats(nav: pd.DataFrame, start: str, end: str, *, prefix: str) -> dict[str, float]:
    if nav.empty or "date" not in nav or "daily_return" not in nav:
        return {f"{prefix}_return": np.nan, f"{prefix}_mdd": np.nan, f"{prefix}_avg_position_count": np.nan}
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].copy()
    if window.empty:
        return {f"{prefix}_return": np.nan, f"{prefix}_mdd": np.nan, f"{prefix}_avg_position_count": np.nan}
    returns = pd.to_numeric(window["daily_return"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(window["total_value"], errors="coerce") if "total_value" in window else (1.0 + returns).cumprod()
    drawdown = values / values.cummax() - 1.0
    return {
        f"{prefix}_return": compound_return(returns),
        f"{prefix}_mdd": float(drawdown.min()) if not drawdown.empty else np.nan,
        f"{prefix}_avg_position_count": float(pd.to_numeric(window.get("position_count"), errors="coerce").fillna(0.0).mean())
        if "position_count" in window
        else np.nan,
    }


def summarize_yearly_nav(run_id: str, nav: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        values = pd.to_numeric(group["total_value"], errors="coerce")
        returns = pd.to_numeric(group["daily_return"], errors="coerce").fillna(0.0)
        drawdown = values / values.cummax() - 1.0
        rows.append({
            "run_id": run_id,
            "variant": metadata.get("variant"),
            "topk": metadata.get("topk"),
            "max_positions": metadata.get("max_positions"),
            "year": int(year),
            "year_return": compound_return(returns),
            "year_mdd": float(drawdown.min()) if not drawdown.empty else np.nan,
            "avg_position_count": float(pd.to_numeric(group.get("position_count"), errors="coerce").fillna(0.0).mean())
            if "position_count" in group
            else np.nan,
            "zero_position_ratio": float((pd.to_numeric(group.get("position_count"), errors="coerce").fillna(0.0) <= 0).mean())
            if "position_count" in group
            else np.nan,
        })
    return rows


def write_comparison(metrics: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top4 = reference.get("current_top4_pos5", {})
    raw12 = reference.get("raw_top12_pos12", {})
    for _, row in metrics.iterrows():
        out = row.to_dict()
        for prefix, ref in (("vs_top4", top4), ("vs_raw12", raw12)):
            out[f"{prefix}_total_return_delta"] = safe_float(row.get("total_return")) - safe_float(ref.get("total_return"))
            out[f"{prefix}_mdd_delta"] = safe_float(row.get("max_drawdown")) - safe_float(ref.get("max_drawdown"))
            out[f"{prefix}_sharpe_delta"] = safe_float(row.get("sharpe")) - safe_float(ref.get("sharpe"))
            out[f"{prefix}_post2022_avg_position_delta"] = safe_float(row.get("post2022_avg_position_count")) - safe_float(
                ref.get("post2022_avg_position_count")
            )
            out[f"{prefix}_crash20_return_delta"] = safe_float(row.get("crash20_return")) - safe_float(ref.get("crash20_return"))
        rows.append(out)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(path, index=False)
    return comparison


def write_pre_backtest_report(ic_summary: pd.DataFrame, signals: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Market-State Ranking Round C Pre-Backtest",
        "",
        "范围：原弱转强 raw candidates；IC 过滤排除不可买、ST、入口 ST、无入口 bar、零量/零额、一字涨停。",
        "",
        "## Top 5D IC",
        "",
        "| group | value | factor | rows | days | ic_mean | ic_ir | pos_ratio | ret_mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not ic_summary.empty:
        top = ic_summary[ic_summary["horizon"] == 5].sort_values("ic_mean_abs_rank", ascending=False).head(25)
        for _, row in top.iterrows():
            lines.append(
                "| {group} | {value} | {factor} | {rows:.0f} | {days:.0f} | {ic:.4f} | {ir:.4f} | {pos:.4f} | {ret:.4f} |".format(
                    group=row.get("group_name"),
                    value=row.get("group_value"),
                    factor=row.get("factor"),
                    rows=safe_float(row.get("rows")),
                    days=safe_float(row.get("days")),
                    ic=safe_float(row.get("spearman_mean")),
                    ir=safe_float(row.get("spearman_ir")),
                    pos=safe_float(row.get("spearman_pos_ratio")),
                    ret=safe_float(row.get("ret_mean")),
                )
            )
    lines.extend([
        "",
        "## Signals",
        "",
        "| variant | topk | pos | rows | days | days>=topk | first | last |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for signal in signals:
        lines.append(
            "| {variant} | {topk} | {pos} | {rows} | {days} | {days_topk} | {first} | {last} |".format(
                variant=signal["variant"],
                topk=signal["topk"],
                pos=signal["max_positions"],
                rows=signal["signal_rows"],
                days=signal["signal_days"],
                days_topk=signal["days_with_at_least_topk"],
                first=signal["first_signal"],
                last=signal["last_signal"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(metrics: pd.DataFrame, comparison: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> None:
    lines = [
        "# Market-State Ranking Round C",
        "",
        "范围：原弱转强候选；保持 `selector.lag=1`，第二天 `close` 买入；卖出、成本、仓位规则沿用源策略。",
        "",
        "## Baselines",
        "",
        "| baseline | total_return | max_drawdown | sharpe | avg_pos | post2022_avg_pos | post2022_zero_pos | crash20_return |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in reference.items():
        lines.append(
            "| {name} | {ret:.4f} | {mdd:.4f} | {sharpe:.4f} | {avg:.4f} | {avg22:.4f} | {zero22:.4f} | {crash:.4f} |".format(
                name=name,
                ret=safe_float(row.get("total_return")),
                mdd=safe_float(row.get("max_drawdown")),
                sharpe=safe_float(row.get("sharpe")),
                avg=safe_float(row.get("full_avg_position_count")),
                avg22=safe_float(row.get("post2022_avg_position_count")),
                zero22=safe_float(row.get("post2022_zero_position_ratio")),
                crash=safe_float(row.get("crash20_return")),
            )
        )
    lines.extend([
        "",
        "## Results",
        "",
        "| variant | topk | pos | total_return | max_drawdown | sharpe | calmar | avg_pos | post2022_avg_pos | post2022_zero_pos | crash20_return | vs_top4_ret | vs_raw12_ret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in comparison.sort_values(["sharpe", "total_return"], ascending=[False, False]).iterrows():
        lines.append(
            "| {variant} | {topk:.0f} | {pos:.0f} | {ret:.4f} | {mdd:.4f} | {sharpe:.4f} | {calmar:.4f} | {avg:.4f} | {avg22:.4f} | {zero22:.4f} | {crash:.4f} | {dtop4:.4f} | {draw12:.4f} |".format(
                variant=row.get("variant"),
                topk=safe_float(row.get("topk")),
                pos=safe_float(row.get("max_positions")),
                ret=safe_float(row.get("total_return")),
                mdd=safe_float(row.get("max_drawdown")),
                sharpe=safe_float(row.get("sharpe")),
                calmar=safe_float(row.get("calmar")),
                avg=safe_float(row.get("full_avg_position_count")),
                avg22=safe_float(row.get("post2022_avg_position_count")),
                zero22=safe_float(row.get("post2022_zero_position_ratio")),
                crash=safe_float(row.get("crash20_return")),
                dtop4=safe_float(row.get("vs_top4_total_return_delta")),
                draw12=safe_float(row.get("vs_raw12_total_return_delta")),
            )
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `runs/round_c_ic_summary.csv`",
        "- `runs/round_c_signal_stats.csv`",
        "- `runs/round_c_metrics_full.csv`",
        "- `runs/round_c_comparison.csv`",
        "- `runs/round_c_yearly_nav.csv`",
        "- `runs/round_c_crash_windows.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_run_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {"raw_stdout": text[-2000:]}
    return json.loads(text[start:])


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (np.integer, np.floating)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


def compound_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float((1.0 + series.astype(float)).prod() - 1.0)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def safe_ir(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    std = values.std(ddof=0)
    if values.empty or not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return float(values.mean() / std)


def safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return np.nan
    return float(frame["left"].corr(frame["right"], method=method))


def format_group_value(value: object) -> str:
    if isinstance(value, tuple):
        return "|".join(str(part) for part in value)
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
