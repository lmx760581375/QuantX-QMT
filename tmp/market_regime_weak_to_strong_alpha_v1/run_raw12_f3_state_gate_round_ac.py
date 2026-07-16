"""Round AC gated state-factor refinements around AA3/AB9.

Round AB found that applying high-amplitude ordering in range|lo|mid lifted
Sharpe but gave back AA3's small return edge. This round keeps the f3 pool,
max_positions=12, cash_equal sizing, selector lag=1, next-day close execution,
and original sell rules unchanged. It only gates that range|lo|mid override by
active-value market context.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_state_gate_round_ac"
ROUND_Z_SCRIPT = ROOT / "run_raw12_f3_state_factor_round_z.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
W6_ARTIFACT = ROOT / "raw12_f3_state_combo_round_w/artifacts/w6_excl_range_mid_lo_keep_rmm_bmm_top1"
AA3_ARTIFACT = ROOT / "raw12_f3_range_hilo_round_aa/artifacts/aa3_hilo_amp_all"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ac0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "ac1_w6_reference", "description": "Round W anchor reproduced.", "mode": "w6_reference", "topk": 12, "max_positions": 12},
    {"variant": "ac2_aa3_reference", "description": "Round AA3 anchor reproduced.", "mode": "aa3_reference", "topk": 12, "max_positions": 12},
    {"variant": "ac3_ab9_reference", "description": "AB9 reproduced: AA3 plus range|lo|mid high amplitude on all days.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp",), "topk": 12, "max_positions": 12},
    {"variant": "ac4_lomid_right_side_le50", "description": "AA3 plus range|lo|mid high amplitude only when right_side_ratio pct252 <= 0.50.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_side_le50",), "topk": 12, "max_positions": 12},
    {"variant": "ac5_lomid_right_amt_le50", "description": "AA3 plus range|lo|mid high amplitude only when right_side_amount_ratio pct252 <= 0.50.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_amt_le50",), "topk": 12, "max_positions": 12},
    {"variant": "ac6_lomid_right_side_le60", "description": "AA3 plus range|lo|mid high amplitude only when right_side_ratio pct252 <= 0.60.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_side_le60",), "topk": 12, "max_positions": 12},
    {"variant": "ac7_lomid_right_amt_le60", "description": "AA3 plus range|lo|mid high amplitude only when right_side_amount_ratio pct252 <= 0.60.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_amt_le60",), "topk": 12, "max_positions": 12},
    {"variant": "ac8_lomid_right_both_le60", "description": "AA3 plus range|lo|mid high amplitude when either right-side pct252 <= 0.60.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_both_le60",), "topk": 12, "max_positions": 12},
    {"variant": "ac9_lomid_amt_ret1_le60", "description": "AA3 plus range|lo|mid high amplitude when active amount ret1 pct252 <= 0.60.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_amt_ret1_le60",), "topk": 12, "max_positions": 12},
    {"variant": "ac10_lomid_right_amt_le50_breadth_le75", "description": "AA3 plus range|lo|mid high amplitude when right_side_amount <= 0.50 and breadth age <= 0.75.", "mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_amt_le50_breadth_le75",), "topk": 12, "max_positions": 12},
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


_ROUND_Z_CACHE = None


def load_round_z_cached():
    global _ROUND_Z_CACHE
    if _ROUND_Z_CACHE is None:
        _ROUND_Z_CACHE = load_round_z()
    return _ROUND_Z_CACHE


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    round_z = load_round_z_cached()
    mode = str(spec["mode"])
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    if mode == "f3_reference":
        return base_score

    w6 = round_z.w6_score(selected)
    if mode == "w6_reference":
        return w6

    score = apply_hilo_high_amp(selected, w6, round_z)
    if mode == "aa3_reference":
        return score.fillna(base_score)
    if mode != "rules":
        raise ValueError(f"Unknown Round AC mode: {mode}")
    if spec.get("start") != "aa3":
        score = w6
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule), round_z)
    return score.fillna(base_score)


def apply_hilo_high_amp(selected: pd.DataFrame, score: pd.Series, round_z) -> pd.Series:
    state = round_z.state_key_series(selected)
    adjusted = factor_rank_score(selected, "amplitude_inv", ascending=True, priority=300000.0)
    return score.where(~state.eq("range|hi|lo"), adjusted)


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str, round_z) -> pd.Series:
    state = round_z.state_key_series(selected)
    lomid = state.eq("range|lo|mid")
    adjusted = factor_rank_score(selected, "amplitude_inv", ascending=True, priority=300000.0)
    if rule == "lomid_high_amp":
        mask = lomid
    elif rule == "lomid_high_amp_right_side_le50":
        mask = lomid & percentile(selected, "right_side_ratio_pct252").le(0.50)
    elif rule == "lomid_high_amp_right_amt_le50":
        mask = lomid & percentile(selected, "right_side_amount_ratio_pct252").le(0.50)
    elif rule == "lomid_high_amp_right_side_le60":
        mask = lomid & percentile(selected, "right_side_ratio_pct252").le(0.60)
    elif rule == "lomid_high_amp_right_amt_le60":
        mask = lomid & percentile(selected, "right_side_amount_ratio_pct252").le(0.60)
    elif rule == "lomid_high_amp_right_both_le60":
        mask = lomid & (percentile(selected, "right_side_ratio_pct252").le(0.60) | percentile(selected, "right_side_amount_ratio_pct252").le(0.60))
    elif rule == "lomid_high_amp_amt_ret1_le60":
        mask = lomid & percentile(selected, "active_all_amount_ret1_pct252").le(0.60)
    elif rule == "lomid_high_amp_right_amt_le50_breadth_le75":
        mask = lomid & percentile(selected, "right_side_amount_ratio_pct252").le(0.50) & percentile(selected, "breadth_bull_age_pct252").le(0.75)
    else:
        raise ValueError(f"Unknown Round AC rule: {rule}")
    return score.where(~mask, adjusted)


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def factor_rank_score(selected: pd.DataFrame, factor: str, *, ascending: bool, priority: float) -> pd.Series:
    values = pd.to_numeric(selected[factor], errors="coerce") if factor in selected else pd.Series(np.nan, index=selected.index)
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    rank = values.groupby(selected["signal_time"]).rank(method="first", ascending=ascending)
    return (priority - rank).where(values.notna()) - base_rank * 1e-4


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_ac_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_ac_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_ac_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_ac_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_ac_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_ac_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_STATE_GATE_ROUND_AC_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_STATE_GATE_ROUND_AC.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_ac_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "raw12": read_json(RAW12_BASE / "summary.json"),
        "f3": read_json(ROUND_F_BEST / "summary.json"),
        "w6": read_json(W6_ARTIFACT / "summary.json"),
        "aa3": read_json(AA3_ARTIFACT / "summary.json"),
    }
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_ac_comparison.csv"
    signal_path = output_root / "runs/round_ac_signal_stats.csv"
    diag_path = output_root / "runs/round_ac_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 State Gate Round AC",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 AA3 上的 `range|lo|mid` 活跃市值门控排序。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_aa3_ret | vs_aa3_mdd | vs_aa3_sharpe | vs_w6_ret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_aa3_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_aa3_mdd_delta')):.4f} | {safe_float(row.get('vs_aa3_sharpe_delta')):.4f} | "
            f"{safe_float(row.get('vs_w6_total_return_delta')):.4f} |"
        )
    if not diag.empty:
        lines.extend(["", "## OOS IC", "", "| horizon | avg_ic | avg_pearson | years |", "| ---: | ---: | ---: | ---: |"])
        for horizon, group in diag.groupby("horizon", sort=True):
            lines.append(
                f"| {int(horizon)} | {safe_float(group['spearman_ic'].mean()):.4f} | "
                f"{safe_float(group['pearson_ic'].mean()):.4f} | {int(group['year'].nunique())} |"
            )
    lines.extend(["", "## Artifacts", "", "- `runs/round_ac_comparison.csv`", "- `runs/round_ac_signal_stats.csv`", "- `runs/round_ac_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_STATE_GATE_ROUND_AC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
