"""Round AW: active-value market overlays on top of AU2.

AU2 is the current best formal top12 weak-to-strong variant, but its edge is
small and path-sensitive. This round keeps the same candidate count, lag=1,
next-day close execution, cash_equal sizing, max_positions=12, and sell rules.
It only tests point-in-time active-value market context as a ranking overlay.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_aw"
ROUND_AU_SCRIPT = ROOT / "run_raw12_f3_ag4_path_scan_round_au.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "aw0_am2_reference", "description": "AM2 reproduced.", "mode": "am2_reference", "topk": 12, "max_positions": 12},
    {"variant": "aw1_aq4_reference", "description": "AQ4 reproduced.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "aw2_au2_reference", "description": "AU2 reproduced: AQ4 plus demote all bear|lo|lo.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom"), "topk": 12, "max_positions": 12},
    {
        "variant": "aw3_au2_hot_amount_rescue_range_tail",
        "description": "AU2 plus rescue AQ4-demoted range weak tails when active amount ret2 is hot.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_amount_rescue_range_tail"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw4_au2_hot_right_rescue_range_tail",
        "description": "AU2 plus rescue AQ4-demoted range weak tails when right-side amount is hot.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_right_rescue_range_tail"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw5_au2_hot_both_rescue_range_tail",
        "description": "AU2 plus rescue range weak tails only when amount and right-side amount are both hot.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_both_rescue_range_tail"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw6_au2_cold_amount_extra_bad_bottom",
        "description": "AU2 plus demote bear|lo|mid and bear|mid|lo tails only when active amount is cold.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "cold_amount_extra_bad_bottom"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw7_au2_hot_good_tail_raw_front",
        "description": "AU2 plus promote observable good-state tails in hot active-value regimes by raw rank.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_good_tail_raw_front"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw8_au2_hot_good_tail_turnover_front",
        "description": "AU2 plus promote observable good-state tails in hot active-value regimes by turnover rank.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom", "hot_good_tail_turnover_front"),
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "aw9_aq4_active_cold_bear_lolo_only",
        "description": "AQ4 plus demote bear|lo|lo only when active amount context is cold, a stress test against AU2.",
        "mode": "am2_rules",
        "rules": ("tail_range_bad_bottom", "cold_active_bear_lolo_bottom"),
        "topk": 12,
        "max_positions": 12,
    },
)


_AU = None
_AU_APPLY_RULE = None


def main() -> None:
    au = load_au()
    au.OUTPUT_NAME = OUTPUT_NAME
    au.VARIANTS = VARIANTS
    au.apply_rule = apply_rule
    au.normalize_outputs = normalize_outputs
    au.main()


def load_au():
    global _AU, _AU_APPLY_RULE
    if _AU is not None:
        return _AU
    spec = importlib.util.spec_from_file_location("round_au_harness", ROUND_AU_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round AU harness from {ROUND_AU_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AU_APPLY_RULE = module.apply_rule
    _AU = module
    return module


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    au = load_au()
    asmod = au.load_as()
    state = asmod.load_ai().state_key_series(selected)
    score_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    tail = score_rank.gt(3.0)

    range_bad = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid"))
    good_tail_state = state.isin(("bull|mid|mid", "bull|hi|hi", "bull|mid|hi", "range|hi|mid", "bear|mid|mid"))
    extra_bad_bear = state.isin(("bear|lo|mid", "bear|mid|lo"))
    bear_lolo = state.eq("bear|lo|lo")

    active_hot = percentile(selected, "active_all_amount_ret2_pct252").ge(0.75)
    active_cold = percentile(selected, "active_all_amount_ret2_pct252").le(0.25)
    right_hot = percentile(selected, "right_side_amount_ratio_pct252").ge(0.75)

    if rule == "hot_amount_rescue_range_tail":
        adjusted = asmod.tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & range_bad & active_hot), adjusted)
    if rule == "hot_right_rescue_range_tail":
        adjusted = asmod.tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & range_bad & right_hot), adjusted)
    if rule == "hot_both_rescue_range_tail":
        adjusted = asmod.tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & range_bad & active_hot & right_hot), adjusted)
    if rule == "cold_amount_extra_bad_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(tail & extra_bad_bear & active_cold), adjusted)
    if rule == "hot_good_tail_raw_front":
        adjusted = asmod.tail_after_top_score(selected, score, "raw_rank_inv", ascending=False)
        return score.where(~(tail & good_tail_state & (active_hot | right_hot)), adjusted)
    if rule == "hot_good_tail_turnover_front":
        adjusted = asmod.tail_after_top_score(selected, score, "turnover_amount_rank", ascending=False)
        return score.where(~(tail & good_tail_state & (active_hot | right_hot)), adjusted)
    if rule == "cold_active_bear_lolo_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(bear_lolo & active_cold), adjusted)
    if _AU_APPLY_RULE is None:
        raise RuntimeError("Round AU original apply_rule has not been initialized")
    return _AU_APPLY_RULE(selected, score, rule)


def percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_aw_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_aw_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_aw_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_aw_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_aw_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_aw_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AW_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AW.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    au = load_au()
    asmod = au.load_as()
    path = output_root / "runs/round_aw_comparison.csv"
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
    au2 = comparison.loc[comparison["variant"].eq("aw2_au2_reference")]
    if not au2.empty:
        ref = au2.iloc[0]
        comparison["vs_au2_total_return_delta"] = comparison["total_return"] - float(ref["total_return"])
        comparison["vs_au2_mdd_delta"] = comparison["max_drawdown"] - float(ref["max_drawdown"])
        comparison["vs_au2_sharpe_delta"] = comparison["sharpe"] - float(ref["sharpe"])
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_aw_comparison.csv"
    signal_path = output_root / "runs/round_aw_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "variant" in signals else {}
    top1_change = signals.set_index("variant")["top1_changed_ratio"].to_dict() if not signals.empty and "variant" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AW",
        "",
        "范围：保持 AM2/AQ4/AU2 候选集合、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试活跃市值市场状态下的排序 overlay。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_aw_comparison.csv`", "- `runs/round_aw_signal_stats.csv`", "- `runs/round_aw_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
