"""Round AX: early-repair active-value rescues on top of AU2.

AW showed that a plain right-side active-value rescue can help in some years but
hurts badly in older breadth-bull conditions. This round keeps the AU2 top12
candidate set, lag=1, next-day close execution, cash_equal sizing,
max_positions=12, and original sell rules. It only tests observable filters for
when to rescue AQ4-demoted range weak tails.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_ax"
ROUND_AW_SCRIPT = ROOT / "run_raw12_f3_ag4_path_scan_round_aw.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ax0_am2_reference", "description": "AM2 reproduced.", "mode": "am2_reference", "topk": 12, "max_positions": 12},
    {"variant": "ax1_aq4_reference", "description": "AQ4 reproduced.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "ax2_au2_reference", "description": "AU2 reproduced.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom"), "topk": 12, "max_positions": 12},
    {
        "variant": "ax3_au2_hot_right_rescue_age_le50",
        "description": "AU2 plus rescue range weak tails when right-side amount is hot and breadth-bull age pct252 <= 0.50.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le50"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax4_au2_hot_right_rescue_age_le60",
        "description": "AU2 plus rescue range weak tails when right-side amount is hot and breadth-bull age pct252 <= 0.60.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le60"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax5_au2_hot_right_rescue_age_le75",
        "description": "AU2 plus rescue range weak tails when right-side amount is hot and breadth-bull age pct252 <= 0.75.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le75"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax6_au2_hot_right_rescue_not_old",
        "description": "AU2 plus rescue range weak tails when right-side amount is hot and breadth-bull age pct252 is not in the oldest quartile.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_not_old"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax7_au2_hot_right_rescue_early_or_unknown",
        "description": "AU2 plus rescue range weak tails when right-side amount is hot and breadth age is early or missing.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_early_or_unknown"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax8_au2_hot_right_rescue_age_le50_amt_hot",
        "description": "AU2 plus rescue only when right-side amount is hot, active amount ret2 is hot, and breadth-bull age pct252 <= 0.50.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le50_amt_hot"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax9_au2_hot_right_rescue_age_le50_right_trend",
        "description": "AU2 plus rescue when right-side amount is hot, above its slower context, and breadth-bull age pct252 <= 0.50.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le50_right_trend"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "ax10_au2_hot_right_rescue_age_le50_raw_tophalf",
        "description": "AU2 plus rescue only for right-hot early breadth tails whose raw rank is in the daily top half.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_age_le50_raw_tophalf"),
        "topk": 12,
        "max_positions": 12,
    },
)


_AW = None
_AW_APPLY_RULE = None


def main() -> None:
    aw = load_aw()
    aw.OUTPUT_NAME = OUTPUT_NAME
    aw.VARIANTS = VARIANTS
    aw.apply_rule = apply_rule
    aw.normalize_outputs = normalize_outputs
    aw.main()


def load_aw():
    global _AW, _AW_APPLY_RULE
    if _AW is not None:
        return _AW
    spec = importlib.util.spec_from_file_location("round_aw_harness", ROUND_AW_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round AW harness from {ROUND_AW_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AW_APPLY_RULE = module.apply_rule
    _AW = module
    return module


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    aw = load_aw()
    au = aw.load_au()
    asmod = au.load_as()
    state = asmod.load_ai().state_key_series(selected)
    score_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    tail = score_rank.gt(3.0)
    range_bad = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid"))

    right_hot = percentile(selected, "right_side_amount_ratio_pct252").ge(0.75)
    active_hot = percentile(selected, "active_all_amount_ret2_pct252").ge(0.75)
    age = percentile(selected, "breadth_bull_age_pct252")
    raw_rank = pd.to_numeric(selected["raw_rank"], errors="coerce") if "raw_rank" in selected else pd.Series(np.nan, index=selected.index)
    right_fast = percentile(selected, "right_side_amount_ratio_ema20_pct252")
    right_slow = percentile(selected, "right_side_amount_ratio_ema60_pct252")

    rescue = pd.Series(False, index=selected.index)
    if rule == "hot_right_rescue_age_le50":
        rescue = right_hot & age.le(0.50)
    elif rule == "hot_right_rescue_age_le60":
        rescue = right_hot & age.le(0.60)
    elif rule == "hot_right_rescue_age_le75":
        rescue = right_hot & age.le(0.75)
    elif rule == "hot_right_rescue_not_old":
        rescue = right_hot & age.le(0.75)
    elif rule == "hot_right_rescue_early_or_unknown":
        rescue = right_hot & (age.isna() | age.le(0.50))
    elif rule == "hot_right_rescue_age_le50_amt_hot":
        rescue = right_hot & active_hot & age.le(0.50)
    elif rule == "hot_right_rescue_age_le50_right_trend":
        rescue = right_hot & age.le(0.50) & right_fast.gt(right_slow)
    elif rule == "hot_right_rescue_age_le50_raw_tophalf":
        rescue = right_hot & age.le(0.50) & raw_rank.le(6.0)
    else:
        if _AW_APPLY_RULE is None:
            raise RuntimeError("Round AW original apply_rule has not been initialized")
        return _AW_APPLY_RULE(selected, score, rule)

    adjusted = asmod.tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
    return score.where(~(tail & range_bad & rescue), adjusted)


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_ax_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_ax_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_ax_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_ax_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_ax_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_ax_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AX_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AX.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    aw = load_aw()
    au = aw.load_au()
    asmod = au.load_as()
    path = output_root / "runs/round_ax_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "ac5": asmod.read_json(asmod.AC5_ARTIFACT / "summary.json"),
        "ag4": asmod.read_json(asmod.AG4_ARTIFACT / "summary.json"),
        "ak4": asmod.read_json(asmod.AK4_ARTIFACT / "summary.json"),
        "am2": asmod.read_json(asmod.AM2_ARTIFACT / "summary.json"),
    }
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    au2 = comparison.loc[comparison["variant"].eq("ax2_au2_reference")]
    if not au2.empty:
        ref = au2.iloc[0]
        comparison["vs_au2_total_return_delta"] = comparison["total_return"] - float(ref["total_return"])
        comparison["vs_au2_mdd_delta"] = comparison["max_drawdown"] - float(ref["max_drawdown"])
        comparison["vs_au2_sharpe_delta"] = comparison["sharpe"] - float(ref["sharpe"])
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_ax_comparison.csv"
    signal_path = output_root / "runs/round_ax_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    top1_change = signals.set_index("variant")["top1_changed_ratio"].to_dict() if not signals.empty and "top1_changed_ratio" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AX",
        "",
        "范围：保持 AM2/AQ4/AU2 候选集合、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试右侧活跃市值修复在广度年龄不过老时的排序 overlay。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | top1_chg | vs_au2_ret | vs_au2_mdd | vs_au2_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(top1_change.get(variant)):.4f} | "
            f"{safe_float(row.get('vs_au2_total_return_delta')):.4f} | {safe_float(row.get('vs_au2_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_au2_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_ax_comparison.csv`", "- `runs/round_ax_signal_stats.csv`", "- `runs/round_ax_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
