"""Round AA focused range|hi|lo refinements around the Round W/Z anchor.

Round Z showed that broad state-factor overrides mostly damage the weak-to-strong
ordering. The only non-damaging move was a tiny improvement in range|hi|lo when
active amount ret1 was cool and high-amplitude candidates were promoted. This
round keeps the f3 pool and original execution/sell semantics unchanged, and
only tests narrower range|hi|lo variants.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_range_hilo_round_aa"
ROUND_Z_SCRIPT = ROOT / "run_raw12_f3_state_factor_round_z.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
W6_ARTIFACT = ROOT / "raw12_f3_state_combo_round_w/artifacts/w6_excl_range_mid_lo_keep_rmm_bmm_top1"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "aa0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "aa1_w6_reference", "description": "Round W/Z anchor reproduced.", "mode": "w6_reference", "topk": 12, "max_positions": 12},
    {"variant": "aa2_z3_reference", "description": "Round Z best: range|hi|lo high amplitude when active_all_ret1 <= p50.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all_ret1_p50",), "topk": 12, "max_positions": 12},
    {"variant": "aa3_hilo_amp_all", "description": "Rank all range|hi|lo days by high amplitude.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all",), "topk": 12, "max_positions": 12},
    {"variant": "aa4_hilo_amp_all_ret1_p25", "description": "Rank range|hi|lo by high amplitude only when active_all_ret1 <= p25.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all_ret1_p25",), "topk": 12, "max_positions": 12},
    {"variant": "aa5_hilo_amp_core_ret1_p50", "description": "Rank range|hi|lo by high amplitude only when active_core_ret1 <= p50.", "mode": "range_hilo_rule", "rules": ("hilo_amp_core_ret1_p50",), "topk": 12, "max_positions": 12},
    {"variant": "aa6_hilo_amp_tradable_ret1_p50", "description": "Rank range|hi|lo by high amplitude only when active_tradable_ret1 <= p50.", "mode": "range_hilo_rule", "rules": ("hilo_amp_tradable_ret1_p50",), "topk": 12, "max_positions": 12},
    {"variant": "aa7_hilo_amp_keep_top1_p50", "description": "aa2 but preserve original rank1 in range|hi|lo.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all_ret1_p50_keep1",), "topk": 12, "max_positions": 12},
    {"variant": "aa8_hilo_amp_keep_top2_p50", "description": "aa2 but preserve original top2 in range|hi|lo.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all_ret1_p50_keep2",), "topk": 12, "max_positions": 12},
    {"variant": "aa9_hilo_amp_keep_top1_all", "description": "Rank all range|hi|lo by high amplitude but preserve original rank1.", "mode": "range_hilo_rule", "rules": ("hilo_amp_all_keep1",), "topk": 12, "max_positions": 12},
    {"variant": "aa10_hilo_low_rsv_long_p50", "description": "Rank range|hi|lo by lower rsv_long only when active_all_ret1 <= p50.", "mode": "range_hilo_rule", "rules": ("hilo_low_rsv_long_p50",), "topk": 12, "max_positions": 12},
)


def main() -> None:
    round_z = load_round_z()
    round_z.OUTPUT_NAME = OUTPUT_NAME
    round_z.VARIANTS = VARIANTS
    round_z.variant_score = variant_score
    round_z.normalize_outputs = normalize_outputs
    round_z.main()


def load_round_z():
    spec = importlib.util.spec_from_file_location("round_z_harness", ROUND_Z_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round Z harness from {ROUND_Z_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    round_z = load_round_z_cached()
    mode = str(spec["mode"])
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    if mode == "f3_reference":
        return base_score
    score = round_z.w6_score(selected)
    if mode == "w6_reference":
        return score
    if mode != "range_hilo_rule":
        raise ValueError(f"Unknown Round AA mode: {mode}")
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule), round_z)
    return score.fillna(base_score)


_ROUND_Z_CACHE = None


def load_round_z_cached():
    global _ROUND_Z_CACHE
    if _ROUND_Z_CACHE is None:
        _ROUND_Z_CACHE = load_round_z()
    return _ROUND_Z_CACHE


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str, round_z) -> pd.Series:
    state = round_z.state_key_series(selected)
    mask = state.eq("range|hi|lo")
    keep_top = 0
    factor = "amplitude_inv"
    ascending = True
    if rule == "hilo_amp_all":
        pass
    elif rule == "hilo_amp_all_ret1_p50":
        mask &= pct(selected, "active_all_amount_ret1_pct252").le(0.50)
    elif rule == "hilo_amp_all_ret1_p25":
        mask &= pct(selected, "active_all_amount_ret1_pct252").le(0.25)
    elif rule == "hilo_amp_core_ret1_p50":
        mask &= pct(selected, "active_core_amount_ret1_pct252").le(0.50)
    elif rule == "hilo_amp_tradable_ret1_p50":
        mask &= pct(selected, "active_tradable_amount_ret1_pct252").le(0.50)
    elif rule == "hilo_amp_all_ret1_p50_keep1":
        mask &= pct(selected, "active_all_amount_ret1_pct252").le(0.50)
        keep_top = 1
    elif rule == "hilo_amp_all_ret1_p50_keep2":
        mask &= pct(selected, "active_all_amount_ret1_pct252").le(0.50)
        keep_top = 2
    elif rule == "hilo_amp_all_keep1":
        keep_top = 1
    elif rule == "hilo_low_rsv_long_p50":
        mask &= pct(selected, "active_all_amount_ret1_pct252").le(0.50)
        factor = "rsv_long"
        ascending = True
    else:
        raise ValueError(f"Unknown Round AA rule: {rule}")
    adjusted = round_z.factor_rank_score(selected, factor, ascending=ascending, priority=300000.0)
    if keep_top:
        base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
        adjusted = adjusted.where(~(mask & base_rank.le(keep_top)), 400000.0 - base_rank)
    return score.where(~mask, adjusted)


def pct(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_aa_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_aa_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_aa_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_aa_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_aa_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_aa_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_RANGE_HILO_ROUND_AA_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_RANGE_HILO_ROUND_AA.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_aa_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"raw12": read_json(RAW12_BASE / "summary.json"), "f3": read_json(ROUND_F_BEST / "summary.json"), "w6": read_json(W6_ARTIFACT / "summary.json")}
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_aa_comparison.csv"
    signal_path = output_root / "runs/round_aa_signal_stats.csv"
    diag_path = output_root / "runs/round_aa_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 Range Hi-Lo Round AA",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只细化 `range|hi|lo` 局部排序。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_aa_comparison.csv`", "- `runs/round_aa_signal_stats.csv`", "- `runs/round_aa_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_RANGE_HILO_ROUND_AA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
