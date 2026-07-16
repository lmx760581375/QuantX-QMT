"""Round AT: AQ4 boundary amplitude ordering tests.

AQ4 is the current best top12 weak-to-strong variant. Boundary IC on the AQ4
buyable top12 pool showed the most stable 3/5/7 day signal is lower amplitude.
This round keeps AM2/AQ4 construction, max_positions=12, cash_equal sizing,
selector lag=1, next-day close execution, and original sell rules unchanged;
it only tests post-AQ4 ordering inside the preserved top12 candidate set.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_at"
ROUND_AS_SCRIPT = ROOT / "run_raw12_f3_ag4_path_scan_round_as.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "at0_am2_reference", "description": "AM2 reproduced.", "mode": "am2_reference", "topk": 12, "max_positions": 12},
    {"variant": "at1_aq4_reference", "description": "AQ4 reproduced: preserve top3 and push range weak-state tails lower.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "at2_aq4_tail_amp_low_all", "description": "AQ4 plus low-amplitude ordering for all ranks below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_all"), "topk": 12, "max_positions": 12},
    {"variant": "at3_aq4_tail_amp_low_range", "description": "AQ4 plus low-amplitude ordering for range-state ranks below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_range"), "topk": 12, "max_positions": 12},
    {"variant": "at4_aq4_tail_amp_low_bear_range", "description": "AQ4 plus low-amplitude ordering for bear/range ranks below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_bear_range"), "topk": 12, "max_positions": 12},
    {"variant": "at5_aq4_midtail_amp_low_all", "description": "AQ4 plus low-amplitude ordering only for ranks 4-9.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "midtail_amp_low_all"), "topk": 12, "max_positions": 12},
    {"variant": "at6_aq4_bad_tail_amp_low_bottom", "description": "AQ4 keeps bad tails at bottom but orders those bad tails by lower amplitude.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom_amp"), "topk": 12, "max_positions": 12},
    {"variant": "at7_aq4_bear_midmid_amp_low", "description": "AQ4 plus low-amplitude ordering for bear|mid|mid below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_bear_midmid"), "topk": 12, "max_positions": 12},
    {"variant": "at8_aq4_range_midhi_amp_low", "description": "AQ4 plus low-amplitude ordering for range|mid|hi below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_range_midhi"), "topk": 12, "max_positions": 12},
    {"variant": "at9_aq4_ic_state_amp_low", "description": "AQ4 plus low-amplitude ordering in broad IC-positive bear/range states below top3.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "tail_amp_low_ic_states"), "topk": 12, "max_positions": 12},
)


_AS = None
_AS_APPLY_RULE = None


def main() -> None:
    asmod = load_as()
    asmod.OUTPUT_NAME = OUTPUT_NAME
    asmod.VARIANTS = VARIANTS
    asmod.variant_score = variant_score
    asmod.apply_rule = apply_rule
    asmod.normalize_outputs = normalize_outputs
    asmod.main()


def load_as():
    global _AS, _AS_APPLY_RULE
    if _AS is not None:
        return _AS
    spec = importlib.util.spec_from_file_location("round_as_harness", ROUND_AS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round AS harness from {ROUND_AS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AS_APPLY_RULE = module.apply_rule
    _AS = module
    return module


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    asmod = load_as()
    ai = asmod.load_ai()
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
        raise ValueError(f"Unknown Round AT mode: {mode}")
    score = am2
    for rule in normalized_rules(spec.get("rules", ())):
        score = apply_rule(selected, score, rule)
    return score.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))


def normalized_rules(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    asmod = load_as()
    state = asmod.load_ai().state_key_series(selected)
    score_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    tail = score_rank.gt(3.0)
    midtail = score_rank.between(4.0, 9.0)

    range_state = state.str.startswith("range|")
    bear_or_range = state.str.startswith("range|") | state.str.startswith("bear|")
    range_bad_state = state.isin(("range|mid|mid", "range|mid|lo", "range|lo|hi", "range|lo|mid"))
    bear_midmid = state.eq("bear|mid|mid")
    range_midhi = state.eq("range|mid|hi")
    ic_states = state.isin(("bear|mid|mid", "range|mid|hi", "range|mid|mid", "range|hi|mid"))

    if rule == "tail_range_bad_bottom_amp":
        adjusted = asmod.tail_bottom_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & range_bad_state), adjusted)
    if rule == "tail_amp_low_all":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~tail, adjusted)
    if rule == "tail_amp_low_range":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & range_state), adjusted)
    if rule == "tail_amp_low_bear_range":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & bear_or_range), adjusted)
    if rule == "midtail_amp_low_all":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~midtail, adjusted)
    if rule == "tail_amp_low_bear_midmid":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & bear_midmid), adjusted)
    if rule == "tail_amp_low_range_midhi":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & range_midhi), adjusted)
    if rule == "tail_amp_low_ic_states":
        adjusted = asmod.tail_after_top_score(selected, score, "amplitude_inv", ascending=False)
        return score.where(~(tail & ic_states), adjusted)
    if _AS_APPLY_RULE is None:
        raise RuntimeError("Round AS original apply_rule has not been initialized")
    return _AS_APPLY_RULE(selected, score, rule)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_at_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_at_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_at_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_at_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_at_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_at_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AT_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AT.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    asmod = load_as()
    path = output_root / "runs/round_at_comparison.csv"
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
    aq4 = comparison.loc[comparison["variant"].eq("at1_aq4_reference")]
    if not aq4.empty:
        ref = aq4.iloc[0]
        comparison["vs_aq4_total_return_delta"] = comparison["total_return"] - float(ref["total_return"])
        comparison["vs_aq4_mdd_delta"] = comparison["max_drawdown"] - float(ref["max_drawdown"])
        comparison["vs_aq4_sharpe_delta"] = comparison["sharpe"] - float(ref["sharpe"])
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_at_comparison.csv"
    signal_path = output_root / "runs/round_at_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "variant" in signals else {}
    top1_change = signals.set_index("variant")["top1_changed_ratio"].to_dict() if not signals.empty and "top1_changed_ratio" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AT",
        "",
        "范围：保持 AM2/AQ4 候选集合、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 AQ4 top12 边界内低振幅排序。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | top1_chg | vs_aq4_ret | vs_aq4_mdd | vs_aq4_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(top1_change.get(variant)):.4f} | "
            f"{safe_float(row.get('vs_aq4_total_return_delta')):.4f} | {safe_float(row.get('vs_aq4_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_aq4_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_at_comparison.csv`", "- `runs/round_at_signal_stats.csv`", "- `runs/round_at_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
