"""Round AG: AC5 plus market-gated bull|mid|hi low-amplitude ordering.

AF found that lower-amplitude ordering inside bull|mid|hi improves Sharpe but
loses total return when applied unconditionally. Realized attribution and IC
diagnostics suggest the effect is concentrated in active-value expansion
contexts. This round keeps the AC5 candidate pool, max_positions=12,
cash_equal sizing, selector lag=1, next-day close execution, and original sell
rules unchanged; it only gates the bull|mid|hi local ordering by market state.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ac5_bullmidhi_gate_round_ag"
ROUND_AC_SCRIPT = ROOT / "run_raw12_f3_state_gate_round_ac.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AA3_ARTIFACT = ROOT / "raw12_f3_range_hilo_round_aa/artifacts/aa3_hilo_amp_all"
W6_ARTIFACT = ROOT / "raw12_f3_state_combo_round_w/artifacts/w6_excl_range_mid_lo_keep_rmm_bmm_top1"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ag0_ac5_reference", "description": "Exact AC5 reference.", "mode": "ac5_reference", "topk": 12, "max_positions": 12},
    {"variant": "ag1_bmh_low_amp_active_amt1_gt50", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret1_pct252 > 0.50.", "mode": "ac5_rules", "rules": ("bmh_low_amp_active_amt1_gt50",), "topk": 12, "max_positions": 12},
    {"variant": "ag2_bmh_low_amp_active_amt1_gt75", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret1_pct252 > 0.75.", "mode": "ac5_rules", "rules": ("bmh_low_amp_active_amt1_gt75",), "topk": 12, "max_positions": 12},
    {"variant": "ag3_bmh_low_amp_active_amt2_gt50", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret2_pct252 > 0.50.", "mode": "ac5_rules", "rules": ("bmh_low_amp_active_amt2_gt50",), "topk": 12, "max_positions": 12},
    {"variant": "ag4_bmh_low_amp_active_amt2_gt75", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret2_pct252 > 0.75.", "mode": "ac5_rules", "rules": ("bmh_low_amp_active_amt2_gt75",), "topk": 12, "max_positions": 12},
    {"variant": "ag5_bmh_low_amp_right_amt_gt50", "description": "Bull|mid|hi lower-amplitude ordering when right_side_amount_ratio_pct252 > 0.50.", "mode": "ac5_rules", "rules": ("bmh_low_amp_right_amt_gt50",), "topk": 12, "max_positions": 12},
    {"variant": "ag6_bmh_low_amp_right_amt_le75", "description": "Bull|mid|hi lower-amplitude ordering when right_side_amount_ratio_pct252 <= 0.75.", "mode": "ac5_rules", "rules": ("bmh_low_amp_right_amt_le75",), "topk": 12, "max_positions": 12},
    {"variant": "ag7_bmh_low_amp_amt1_gt50_right_gt50", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret1 and right-side amount are both > 0.50.", "mode": "ac5_rules", "rules": ("bmh_low_amp_amt1_gt50_right_gt50",), "topk": 12, "max_positions": 12},
    {"variant": "ag8_bmh_low_amp_amt2_gt50_right_gt50", "description": "Bull|mid|hi lower-amplitude ordering when active_all_amount_ret2 and right-side amount are both > 0.50.", "mode": "ac5_rules", "rules": ("bmh_low_amp_amt2_gt50_right_gt50",), "topk": 12, "max_positions": 12},
)


_AC = None
_AC_VARIANT_SCORE = None


def main() -> None:
    global _AC_VARIANT_SCORE
    ac = load_ac()
    ac.OUTPUT_NAME = OUTPUT_NAME
    ac.VARIANTS = VARIANTS
    _AC_VARIANT_SCORE = ac.variant_score
    ac.variant_score = variant_score
    ac.normalize_outputs = normalize_outputs
    ac.main()


def load_ac():
    global _AC
    if _AC is not None:
        return _AC
    spec = importlib.util.spec_from_file_location("round_ac_harness", ROUND_AC_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round AC harness from {ROUND_AC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AC = module
    return module


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    if _AC_VARIANT_SCORE is None:
        raise RuntimeError("Round AC original variant_score has not been initialized")
    ac5 = _AC_VARIANT_SCORE(
        selected,
        {"mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_amt_le50",), "topk": 12, "max_positions": 12},
    )
    mode = str(spec["mode"])
    if mode == "ac5_reference":
        return ac5
    if mode != "ac5_rules":
        raise ValueError(f"Unknown Round AG mode: {mode}")
    score = ac5
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule))
    return score.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    state = state_key_series(selected)
    bull_midhi = state.eq("bull|mid|hi")
    amt1 = percentile(selected, "active_all_amount_ret1_pct252")
    amt2 = percentile(selected, "active_all_amount_ret2_pct252")
    right_amt = percentile(selected, "right_side_amount_ratio_pct252")

    if rule == "bmh_low_amp_active_amt1_gt50":
        mask = bull_midhi & amt1.gt(0.50)
    elif rule == "bmh_low_amp_active_amt1_gt75":
        mask = bull_midhi & amt1.gt(0.75)
    elif rule == "bmh_low_amp_active_amt2_gt50":
        mask = bull_midhi & amt2.gt(0.50)
    elif rule == "bmh_low_amp_active_amt2_gt75":
        mask = bull_midhi & amt2.gt(0.75)
    elif rule == "bmh_low_amp_right_amt_gt50":
        mask = bull_midhi & right_amt.gt(0.50)
    elif rule == "bmh_low_amp_right_amt_le75":
        mask = bull_midhi & right_amt.le(0.75)
    elif rule == "bmh_low_amp_amt1_gt50_right_gt50":
        mask = bull_midhi & amt1.gt(0.50) & right_amt.gt(0.50)
    elif rule == "bmh_low_amp_amt2_gt50_right_gt50":
        mask = bull_midhi & amt2.gt(0.50) & right_amt.gt(0.50)
    else:
        raise ValueError(f"Unknown Round AG rule: {rule}")
    return apply_factor(selected, score, mask, "amplitude_inv", ascending=False, priority=400000.0)


def apply_factor(
    selected: pd.DataFrame,
    score: pd.Series,
    mask: pd.Series,
    factor: str,
    *,
    ascending: bool,
    priority: float,
) -> pd.Series:
    adjusted = factor_rank_score(selected, factor, ascending=ascending, priority=priority)
    return score.where(~mask, adjusted)


def state_key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def factor_rank_score(selected: pd.DataFrame, factor: str, *, ascending: bool, priority: float) -> pd.Series:
    values = pd.to_numeric(selected[factor], errors="coerce") if factor in selected else pd.Series(np.nan, index=selected.index)
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    rank = values.groupby(selected["signal_time"]).rank(method="first", ascending=ascending)
    return (priority - rank).where(values.notna()) - base_rank * 1e-4


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_ag_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_ag_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_ag_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_ag_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_ag_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_ag_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AC5_BULLMIDHI_GATE_ROUND_AG_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AC5_BULLMIDHI_GATE_ROUND_AG.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_ag_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "ac5": read_json(AC5_ARTIFACT / "summary.json"),
        "aa3": read_json(AA3_ARTIFACT / "summary.json"),
        "w6": read_json(W6_ARTIFACT / "summary.json"),
    }
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_ag_comparison.csv"
    signal_path = output_root / "runs/round_ag_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 AC5 Bullmidhi Gate Round AG",
        "",
        "范围：保持 AC5 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 `bull|mid|hi` 低振幅排序的市场状态门控。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_ac5_ret | vs_ac5_mdd | vs_ac5_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_ac5_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_ac5_mdd_delta')):.4f} | {safe_float(row.get('vs_ac5_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_ag_comparison.csv`", "- `runs/round_ag_signal_stats.csv`", "- `runs/round_ag_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AC5_BULLMIDHI_GATE_ROUND_AG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
