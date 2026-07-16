"""Round AR: AQ4 boundary scan for path-sensitive bad-state tails.

Round AQ found the first formal improvement over AM2: pushing weak-state tail
candidates below the rest of the AM2 pool lifted return and Sharpe while keeping
the same top12 capacity. This round keeps that mechanism and scans the minimum
state/rank boundary needed for the effect.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_ar"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"
AK4_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_ak/artifacts/ak4_bull_hihi_ml5_low"
AM2_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_an/artifacts/an1_am2_reference"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ar0_am2_reference", "description": "AM2 reproduced.", "mode": "am2_reference", "topk": 12, "max_positions": 12},
    {"variant": "ar1_aq4_reference", "description": "AQ4 reproduced: preserve top3 and push weak-state tails lower.", "mode": "am2_rules", "rules": ("tail_bad_state_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar2_bad_tail_keep2", "description": "AQ4 with only AM2 top2 protected.", "mode": "am2_rules", "rules": ("tail_bad_state_bottom_keep2",), "topk": 12, "max_positions": 12},
    {"variant": "ar3_bad_tail_keep4", "description": "AQ4 with AM2 top4 protected.", "mode": "am2_rules", "rules": ("tail_bad_state_bottom_keep4",), "topk": 12, "max_positions": 12},
    {"variant": "ar4_bad_tail_range_only", "description": "AQ4 restricted to range weak states.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar5_bad_tail_drop_midmid", "description": "AQ4 without range|mid|mid in the bad-state set.", "mode": "am2_rules", "rules": ("tail_bad_no_midmid_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar6_bad_tail_low_breadth", "description": "AQ4 restricted to weak states with low breadth side.", "mode": "am2_rules", "rules": ("tail_bad_low_breadth_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar7_bad_tail_plus_bear_midlo", "description": "AQ4 plus bear|mid|lo in the bad-state set.", "mode": "am2_rules", "rules": ("tail_bad_plus_bear_midlo_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar8_bad_tail_cool_amount", "description": "AQ4 only when active amount ret2 percentile is not hot.", "mode": "am2_rules", "rules": ("tail_bad_cool_amount_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ar9_bad_tail_hot_amount", "description": "AQ4 only when active amount ret2 percentile is hot.", "mode": "am2_rules", "rules": ("tail_bad_hot_amount_bottom",), "topk": 12, "max_positions": 12},
)


_AI = None


def main() -> None:
    ai = load_ai()
    ai.OUTPUT_NAME = OUTPUT_NAME
    ai.VARIANTS = VARIANTS
    ai.variant_score = variant_score
    ai.apply_rule = apply_rule
    ai.normalize_outputs = normalize_outputs
    ai.main()


def load_ai():
    global _AI
    if _AI is not None:
        return _AI
    spec = importlib.util.spec_from_file_location("round_ai_harness", ROUND_AI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round AI harness from {ROUND_AI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AI = module
    return module


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    ai = load_ai()
    if ai._AC_VARIANT_SCORE is None:
        raise RuntimeError("Round AI original variant_score has not been initialized")
    ac5 = ai._AC_VARIANT_SCORE(
        selected,
        {"mode": "rules", "start": "aa3", "rules": ("lomid_high_amp_right_amt_le50",), "topk": 12, "max_positions": 12},
    )
    ag4 = apply_rule(selected, ac5, "bmh_low_amp_active_amt2_gt75")
    ak4 = apply_rule(selected, ag4, "bull_hihi_ml5_low")
    am2 = apply_rule(selected, ak4, "range_himid_ml3_low")
    mode = str(spec["mode"])
    if mode == "am2_reference":
        return am2.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))
    if mode != "am2_rules":
        raise ValueError(f"Unknown Round AR mode: {mode}")
    score = am2
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule))
    return score.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    ai = load_ai()
    state = ai.state_key_series(selected)
    bull_midhi = state.eq("bull|mid|hi")
    bull_hihi = state.eq("bull|hi|hi")
    bull_midmid = state.eq("bull|mid|mid")
    range_himid = state.eq("range|hi|mid")
    range_midhi = state.eq("range|mid|hi")
    range_midmid = state.eq("range|mid|mid")
    bear_midmid = state.eq("bear|mid|mid")
    bear_lolo = state.eq("bear|lo|lo")
    good_state = state.isin(("bull|mid|mid", "bull|hi|hi", "bear|mid|mid", "range|hi|mid", "bull|mid|hi"))
    bad_state = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid", "bear|lo|mid"))
    range_bad_state = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid"))
    bad_no_midmid_state = state.isin(("range|mid|lo", "range|lo|hi", "range|lo|mid", "bear|lo|mid"))
    bad_low_breadth_state = state.isin(("range|mid|lo", "range|lo|hi", "range|lo|mid", "bear|lo|mid"))
    bad_plus_bear_midlo_state = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid", "bear|lo|mid", "bear|mid|lo"))
    am2_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    tail = am2_rank.gt(3.0)
    tail_keep2 = am2_rank.gt(2.0)
    tail_keep4 = am2_rank.gt(4.0)
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "bull_hihi_ml5_low":
        return ai.apply_factor(selected, score, bull_hihi, "ml_pred_5d", ascending=True, priority=410000.0)
    if rule == "range_himid_ml3_low":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_3d", ascending=True, priority=420000.0)
    if rule == "bull_midmid_raw_high":
        return ai.apply_factor(selected, score, bull_midmid, "raw_rank_inv", ascending=False, priority=430000.0)
    if rule == "bull_midmid_low_amp":
        return ai.apply_factor(selected, score, bull_midmid, "amplitude_inv", ascending=False, priority=430000.0)
    if rule == "range_midhi_raw_high":
        return ai.apply_factor(selected, score, range_midhi, "raw_rank_inv", ascending=False, priority=430000.0)
    if rule == "range_midmid_ml357_high":
        return ai.apply_factor(selected, score, range_midmid, "ml_pred_357d", ascending=False, priority=430000.0)
    if rule == "bear_midmid_turnover_high":
        return ai.apply_factor(selected, score, bear_midmid, "turnover_amount_rank", ascending=False, priority=430000.0)
    if rule == "bear_lolo_low_amp":
        return ai.apply_factor(selected, score, bear_lolo, "amplitude_inv", ascending=False, priority=430000.0)
    if rule == "good_state_soft_plus":
        return score + good_state.astype(float) * 0.02
    if rule == "bad_state_soft_minus":
        return score - bad_state.astype(float) * 0.02
    if rule == "tail_bad_state_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_state), adjusted)
    if rule == "tail_bad_state_bottom_keep2":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail_keep2 & bad_state), adjusted)
    if rule == "tail_bad_state_bottom_keep4":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail_keep4 & bad_state), adjusted)
    if rule == "tail_range_bad_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & range_bad_state), adjusted)
    if rule == "tail_bad_no_midmid_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_no_midmid_state), adjusted)
    if rule == "tail_bad_low_breadth_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_low_breadth_state), adjusted)
    if rule == "tail_bad_plus_bear_midlo_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_plus_bear_midlo_state), adjusted)
    if rule == "tail_bad_cool_amount_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_state & amt2.le(0.75)), adjusted)
    if rule == "tail_bad_hot_amount_bottom":
        adjusted = tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & bad_state & amt2.gt(0.75)), adjusted)
    if rule == "tail_good_state_raw":
        adjusted = tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & good_state), adjusted)
    if rule == "tail_bull_midmid_raw":
        adjusted = tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & bull_midmid), adjusted)
    if rule == "tail_range_midmid_ml357":
        adjusted = tail_after_top_score(selected, score, "ml_pred_357d", ascending=False)
        return score.where(~(tail & range_midmid), adjusted)
    raise ValueError(f"Unknown Round AR rule: {rule}")


def factor_rank(selected: pd.DataFrame, factor: str, *, ascending: bool) -> pd.Series:
    values = pd.to_numeric(selected[factor], errors="coerce") if factor in selected else pd.Series(np.nan, index=selected.index)
    rank = values.groupby(selected["signal_time"]).rank(method="first", ascending=ascending)
    return rank.fillna(99.0)


def tail_after_top_score(selected: pd.DataFrame, score: pd.Series, factor: str, *, ascending: bool) -> pd.Series:
    rank = factor_rank(selected, factor, ascending=ascending)
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    score_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    third_score = score.where(score_rank.eq(3.0)).groupby(selected["signal_time"]).transform("max")
    fallback = score.groupby(selected["signal_time"]).transform("max") - 3.0
    anchor = third_score.fillna(fallback)
    return anchor - rank * 0.01 - base_rank * 1e-5


def tail_bottom_score(selected: pd.DataFrame, score: pd.Series, factor: str, *, ascending: bool) -> pd.Series:
    rank = factor_rank(selected, factor, ascending=ascending)
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    bottom = score.groupby(selected["signal_time"]).transform("min")
    return bottom - 0.5 - rank * 0.01 - base_rank * 1e-5


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_ar_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_ar_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_ar_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_ar_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_ar_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_ar_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AR_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AR.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_ar_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "ac5": read_json(AC5_ARTIFACT / "summary.json"),
        "ag4": read_json(AG4_ARTIFACT / "summary.json"),
        "ak4": read_json(AK4_ARTIFACT / "summary.json"),
        "am2": read_json(AM2_ARTIFACT / "summary.json"),
    }
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_ar_comparison.csv"
    signal_path = output_root / "runs/round_ar_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    top1_change = signals.set_index("variant")["top1_changed_ratio"].to_dict() if not signals.empty and "top1_changed_ratio" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AR",
        "",
        "范围：保持 AM2 候选集合、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只追加保守的状态内排序。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | top1_chg | vs_am2_ret | vs_am2_mdd | vs_am2_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(top1_change.get(variant)):.4f} | "
            f"{safe_float(row.get('vs_am2_total_return_delta')):.4f} | {safe_float(row.get('vs_am2_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_am2_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_ar_comparison.csv`", "- `runs/round_ar_signal_stats.csv`", "- `runs/round_ar_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
