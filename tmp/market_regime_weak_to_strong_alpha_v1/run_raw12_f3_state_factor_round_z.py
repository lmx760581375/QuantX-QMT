"""Round Z state-factor ranking tests around the Round W/X winner.

This round keeps the f3 pool, max_positions=12, cash_equal sizing,
selector lag=1, next-day close execution, and original sell rules unchanged.
It uses the buyable non-ST f3 diagnostic to test a small set of localized
state/factor ordering rules on top of the w6 reference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_state_factor_round_z"
ROUND_Y_SCRIPT = ROOT / "run_raw12_f3_state_horizon_round_y.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ACTIVE_VALUE = Path("data/derived/active_value/daily.parquet")

ACTIVE_COLUMNS = (
    "active_all_amount_ret1",
    "active_all_amount_ret2",
    "active_tradable_amount_ret1",
    "active_tradable_amount_ret2",
    "active_core_amount_ret1",
    "active_core_amount_ret2",
    "right_side_ratio",
    "right_side_core_ratio",
    "right_side_amount_ratio",
    "right_side_core_amount_ratio",
    "breadth_bull_age",
)


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "z0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "z1_w6_reference", "description": "Round W/X winner reproduced with 5d ML.", "mode": "w6_reference", "topk": 12, "max_positions": 12},
    {
        "variant": "z2_w6_bull_hihi_hot_liquidity",
        "description": "w6 plus rank bull|hi|hi by liquidity rank only when active amount ret2 is hot.",
        "mode": "w6_state_factor",
        "rules": ("bull_hihi_hot_liquidity",),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z3_w6_range_hilo_cool_high_amp",
        "description": "w6 plus rank range|hi|lo by high amplitude only when active amount ret1 is cool.",
        "mode": "w6_state_factor",
        "rules": ("range_hilo_cool_high_amplitude",),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z4_w6_bull_midmid_turnover",
        "description": "w6 plus rank bull|mid|mid by turnover amount rank.",
        "mode": "w6_state_factor",
        "rules": ("bull_midmid_turnover",),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z5_w6_range_midmid_amplitude",
        "description": "w6 plus rank range|mid|mid by low amplitude.",
        "mode": "w6_state_factor",
        "rules": ("range_midmid_amplitude",),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z6_w6_bear_midmid_turnover",
        "description": "w6 plus rank bear|mid|mid by turnover amount rank.",
        "mode": "w6_state_factor",
        "rules": ("bear_midmid_turnover",),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z7_w6_combo_234",
        "description": "w6 plus the three strongest state-return rules: z2+z3+z4.",
        "mode": "w6_state_factor",
        "rules": ("bull_hihi_hot_liquidity", "range_hilo_cool_high_amplitude", "bull_midmid_turnover"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "z8_w6_combo_3456",
        "description": "w6 plus z3+z4+z5+z6 localized IC rules.",
        "mode": "w6_state_factor",
        "rules": ("range_hilo_cool_high_amplitude", "bull_midmid_turnover", "range_midmid_amplitude", "bear_midmid_turnover"),
        "topk": 12,
        "max_positions": 12,
    },
)


def main() -> None:
    round_y = load_round_y()
    patch_round_y(round_y)
    round_y.main()


def load_round_y():
    spec = importlib.util.spec_from_file_location("round_y_harness", ROUND_Y_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round Y harness from {ROUND_Y_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_round_y(round_y) -> None:
    round_y.OUTPUT_NAME = OUTPUT_NAME
    round_y.VARIANTS = VARIANTS
    round_y.variant_score = variant_score
    round_y.normalize_outputs = normalize_outputs

    round_w = round_y.load_round_w()
    round_s = round_w.load_round_s()
    original_prepare_data = round_s.prepare_data

    def prepare_data_with_active(candidates: pd.DataFrame) -> pd.DataFrame:
        return attach_active_value(original_prepare_data(candidates))

    round_s.prepare_data = prepare_data_with_active
    round_w.load_round_s = lambda: round_s
    round_y.load_round_w = lambda: round_w


def attach_active_value(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    active = pd.read_parquet(ACTIVE_VALUE).copy()
    active["signal_time"] = pd.to_datetime(active["date"]).dt.strftime("%Y-%m-%d")
    active = active.sort_values("signal_time").drop_duplicates("signal_time", keep="last")
    for column in ACTIVE_COLUMNS:
        if column in active.columns:
            active[f"{column}_pct252"] = prior_rolling_percentile(pd.to_numeric(active[column], errors="coerce"))
    keep = ["signal_time", *[col for col in ACTIVE_COLUMNS if col in active.columns], *[f"{col}_pct252" for col in ACTIVE_COLUMNS if f"{col}_pct252" in active.columns]]
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    return frame.merge(active[keep], on="signal_time", how="left")


def prior_rolling_percentile(values: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    out = np.full(len(values), np.nan, dtype=float)
    raw = values.to_numpy(dtype=float)
    for idx, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        history = values.iloc[max(0, idx - window):idx].dropna().to_numpy(dtype=float)
        if len(history) < min_periods:
            continue
        out[idx] = float((history <= value).mean())
    return pd.Series(out, index=values.index)


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    if mode == "f3_reference":
        return base_score

    score = w6_score(selected)
    if mode == "w6_reference":
        return score
    if mode != "w6_state_factor":
        raise ValueError(f"Unknown Round Z mode: {mode}")

    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule))
    return score.fillna(base_score)


def w6_score(selected: pd.DataFrame) -> pd.Series:
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    state_key = state_key_series(selected)
    bad_state = ~selected["f3_good_state"].fillna(False).astype(bool)
    use_ml = bad_state & ~state_key.isin(("range|mid|lo",))
    adjusted = ml_rank_score(selected, "ml_pred_5d")
    keep_mask = state_key.isin(("range|mid|mid", "bear|mid|mid"))
    adjusted = preserve_original_top(selected, adjusted, keep_mask=keep_mask, keep_top=1)
    return adjusted.where(use_ml, base_score).fillna(base_score)


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    state_key = state_key_series(selected)
    if rule == "bull_hihi_hot_liquidity":
        mask = state_key.eq("bull|hi|hi") & percentile(selected, "active_all_amount_ret2_pct252").ge(0.75)
        return score.where(~mask, factor_rank_score(selected, "liquidity_rank", ascending=False, priority=300000.0))
    if rule == "range_hilo_cool_high_amplitude":
        mask = state_key.eq("range|hi|lo") & percentile(selected, "active_all_amount_ret1_pct252").le(0.50)
        return score.where(~mask, factor_rank_score(selected, "amplitude_inv", ascending=True, priority=300000.0))
    if rule == "bull_midmid_turnover":
        mask = state_key.eq("bull|mid|mid")
        return score.where(~mask, factor_rank_score(selected, "turnover_amount_rank", ascending=False, priority=300000.0))
    if rule == "range_midmid_amplitude":
        mask = state_key.eq("range|mid|mid")
        return score.where(~mask, factor_rank_score(selected, "amplitude_inv", ascending=False, priority=300000.0))
    if rule == "bear_midmid_turnover":
        mask = state_key.eq("bear|mid|mid")
        return score.where(~mask, factor_rank_score(selected, "turnover_amount_rank", ascending=False, priority=300000.0))
    raise ValueError(f"Unknown Round Z rule: {rule}")


def factor_rank_score(selected: pd.DataFrame, factor: str, *, ascending: bool, priority: float) -> pd.Series:
    values = pd.to_numeric(selected[factor], errors="coerce") if factor in selected else pd.Series(np.nan, index=selected.index)
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    rank = values.groupby(selected["signal_time"]).rank(method="first", ascending=ascending)
    return (priority - rank).where(values.notna()) - base_rank * 1e-4


def ml_rank_score(selected: pd.DataFrame, pred_col: str) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    return (100000.0 - ml_rank).where(pred.notna()) - base_rank * 1e-4


def preserve_original_top(selected: pd.DataFrame, adjusted: pd.Series, *, keep_mask: pd.Series, keep_top: int) -> pd.Series:
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    preserved = keep_mask & base_rank.le(keep_top)
    preserved_score = 200000.0 - base_rank
    return adjusted.where(~preserved, preserved_score)


def state_key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_z_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_z_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_z_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_z_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_z_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_z_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_STATE_FACTOR_ROUND_Z_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_STATE_FACTOR_ROUND_Z.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_z_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"raw12": read_json(RAW12_BASE / "summary.json"), "f3": read_json(ROUND_F_BEST / "summary.json")}
    w6 = read_json(ROOT / "raw12_f3_state_combo_round_w/artifacts/w6_excl_range_mid_lo_keep_rmm_bmm_top1/summary.json")
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison["vs_w6_total_return_delta"] = comparison["total_return"] - float(w6.get("total_return", np.nan))
    comparison["vs_w6_mdd_delta"] = comparison["max_drawdown"] - float(w6.get("max_drawdown", np.nan))
    comparison["vs_w6_sharpe_delta"] = comparison["sharpe"] - float(w6.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_z_comparison.csv"
    signal_path = output_root / "runs/round_z_signal_stats.csv"
    diag_path = output_root / "runs/round_z_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 State Factor Round Z",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试状态触发的局部排序覆盖。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_w6_ret | vs_w6_mdd | vs_w6_sharpe | vs_f3_ret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_w6_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_w6_mdd_delta')):.4f} | {safe_float(row.get('vs_w6_sharpe_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_total_return_delta')):.4f} |"
        )
    if not diag.empty:
        lines.extend(["", "## OOS IC", "", "| horizon | avg_ic | avg_pearson | years |", "| ---: | ---: | ---: | ---: |"])
        for horizon, group in diag.groupby("horizon", sort=True):
            lines.append(
                f"| {int(horizon)} | {safe_float(group['spearman_ic'].mean()):.4f} | "
                f"{safe_float(group['pearson_ic'].mean()):.4f} | {int(group['year'].nunique())} |"
            )
    lines.extend(["", "## Artifacts", "", "- `runs/round_z_comparison.csv`", "- `runs/round_z_signal_stats.csv`", "- `runs/round_z_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_STATE_FACTOR_ROUND_Z.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
