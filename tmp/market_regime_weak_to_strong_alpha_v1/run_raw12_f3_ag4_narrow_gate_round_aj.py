"""Round AJ: narrower AG4 local gates after Round AI path diagnostics.

This round keeps the AG4 candidate pool and execution semantics unchanged. It
only tests very narrow local re-ordering gates around the one Round AI rule that
nearly matched AG4: lower amplitude within bear|lo|mid.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_narrow_gate_round_aj"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "aj0_ag4_reference", "description": "AG4 reproduced.", "mode": "ag4_reference", "topk": 12, "max_positions": 12},
    {"variant": "aj1_bear_lomid_low_amp", "description": "Round AI ai6 reproduced: bear|lo|mid ordered by lower amplitude.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp",), "topk": 12, "max_positions": 12},
    {"variant": "aj2_bear_lomid_low_amp_keep_top1", "description": "bear|lo|mid lower amplitude while preserving AG4 day rank1.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_keep_top1",), "topk": 12, "max_positions": 12},
    {"variant": "aj3_bear_lomid_low_amp_amt2_25_50", "description": "bear|lo|mid lower amplitude only when active amount ret2 pct is 25%-50%.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_25_50",), "topk": 12, "max_positions": 12},
    {"variant": "aj4_bear_lomid_low_amp_amt2_le50", "description": "bear|lo|mid lower amplitude only when active amount ret2 pct <= 50%.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_le50",), "topk": 12, "max_positions": 12},
    {"variant": "aj5_bear_lomid_low_amp_amt2_25_50_right_le25", "description": "25%-50% active amount gate plus right-side amount pct <= 25%.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_25_50_right_le25",), "topk": 12, "max_positions": 12},
    {"variant": "aj6_bear_lomid_low_amp_amt2_25_50_right_le50", "description": "25%-50% active amount gate plus right-side amount pct <= 50%.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_25_50_right_le50",), "topk": 12, "max_positions": 12},
    {"variant": "aj7_bear_lomid_low_amp_amt2_25_50_breadth_gt60", "description": "25%-50% active amount gate plus breadth bull age pct > 60%.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_25_50_breadth_gt60",), "topk": 12, "max_positions": 12},
    {"variant": "aj8_bear_lomid_low_amp_amt2_25_50_keep_top1", "description": "25%-50% active amount gate while preserving AG4 day rank1.", "mode": "ag4_rules", "rules": ("bear_lomid_low_amp_amt2_25_50_keep_top1",), "topk": 12, "max_positions": 12},
    {"variant": "aj9_range_midlo_ml5_amt2_gt50", "description": "range|mid|lo ml_pred_5d ordering only when active amount ret2 pct > 50%.", "mode": "ag4_rules", "rules": ("range_midlo_ml5_amt2_gt50",), "topk": 12, "max_positions": 12},
    {"variant": "aj10_range_midlo_ml5_amt2_le25", "description": "range|mid|lo ml_pred_5d ordering only when active amount ret2 pct <= 25%.", "mode": "ag4_rules", "rules": ("range_midlo_ml5_amt2_le25",), "topk": 12, "max_positions": 12},
)


_AI = None


def main() -> None:
    ai = load_ai()
    ai.OUTPUT_NAME = OUTPUT_NAME
    ai.VARIANTS = VARIANTS
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


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    ai = load_ai()
    state = ai.state_key_series(selected)
    bull_midhi = state.eq("bull|mid|hi")
    bear_lomid = state.eq("bear|lo|mid")
    range_midlo = state.eq("range|mid|lo")
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")
    right_amt = ai.percentile(selected, "right_side_amount_ratio_pct252")
    breadth_age = ai.percentile(selected, "breadth_bull_age_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "bear_lomid_low_amp":
        return ai.apply_factor(selected, score, bear_lomid, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_keep_top1":
        adjusted = ai.factor_rank_score(selected, "amplitude_inv", ascending=False, priority=410000.0)
        adjusted = ai.preserve_rank1(selected, score, adjusted, bear_lomid)
        return score.where(~bear_lomid, adjusted)
    if rule == "bear_lomid_low_amp_amt2_25_50":
        mask = bear_lomid & amt2.gt(0.25) & amt2.le(0.50)
        return ai.apply_factor(selected, score, mask, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_amt2_le50":
        return ai.apply_factor(selected, score, bear_lomid & amt2.le(0.50), "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_amt2_25_50_right_le25":
        mask = bear_lomid & amt2.gt(0.25) & amt2.le(0.50) & right_amt.le(0.25)
        return ai.apply_factor(selected, score, mask, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_amt2_25_50_right_le50":
        mask = bear_lomid & amt2.gt(0.25) & amt2.le(0.50) & right_amt.le(0.50)
        return ai.apply_factor(selected, score, mask, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_amt2_25_50_breadth_gt60":
        mask = bear_lomid & amt2.gt(0.25) & amt2.le(0.50) & breadth_age.gt(0.60)
        return ai.apply_factor(selected, score, mask, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bear_lomid_low_amp_amt2_25_50_keep_top1":
        mask = bear_lomid & amt2.gt(0.25) & amt2.le(0.50)
        adjusted = ai.factor_rank_score(selected, "amplitude_inv", ascending=False, priority=410000.0)
        adjusted = ai.preserve_rank1(selected, score, adjusted, mask)
        return score.where(~mask, adjusted)
    if rule == "range_midlo_ml5_amt2_gt50":
        return ai.apply_factor(selected, score, range_midlo & amt2.gt(0.50), "ml_pred_5d", ascending=False, priority=410000.0)
    if rule == "range_midlo_ml5_amt2_le25":
        return ai.apply_factor(selected, score, range_midlo & amt2.le(0.25), "ml_pred_5d", ascending=False, priority=410000.0)
    raise ValueError(f"Unknown Round AJ rule: {rule}")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_aj_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_aj_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_aj_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_aj_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_aj_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_aj_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_NARROW_GATE_ROUND_AJ_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_NARROW_GATE_ROUND_AJ.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_aj_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"ac5": read_json(AC5_ARTIFACT / "summary.json"), "ag4": read_json(AG4_ARTIFACT / "summary.json")}
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_aj_comparison.csv"
    signal_path = output_root / "runs/round_aj_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Narrow Gate Round AJ",
        "",
        "范围：保持 AG4 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试更窄的状态门控排序。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_ag4_ret | vs_ag4_mdd | vs_ag4_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_ag4_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_ag4_mdd_delta')):.4f} | {safe_float(row.get('vs_ag4_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_aj_comparison.csv`", "- `runs/round_aj_signal_stats.csv`", "- `runs/round_aj_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_NARROW_GATE_ROUND_AJ.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
