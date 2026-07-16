"""Round AL: protected AK4 refinements.

AK4 improved AG4 by using lower 5d ML ordering inside bull|hi|hi, but the
path attribution showed that a few extreme promotions drove both the gains and
the biggest new loss. This round keeps the AG4/AK4 pool and execution semantics
unchanged and tests narrow protections around that local ordering rule.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_al"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"
AK4_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_ak/artifacts/ak4_bull_hihi_ml5_low"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "al0_ak4_reference", "description": "AK4 reproduced: bull|hi|hi ordered by lower ml_pred_5d.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low",), "topk": 12, "max_positions": 12},
    {"variant": "al1_right_le90", "description": "AK4 rule only when right-side amount pct252 <= 0.90.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_right_le90",), "topk": 12, "max_positions": 12},
    {"variant": "al2_right_75_90", "description": "AK4 rule only when right-side amount pct252 is between 0.75 and 0.90.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_right_75_90",), "topk": 12, "max_positions": 12},
    {"variant": "al3_no_extreme_low_backrank", "description": "AK4 rule, but keep original order for extreme-low ml5 and raw_rank > 8.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_no_extreme_backrank",), "topk": 12, "max_positions": 12},
    {"variant": "al4_front5_only", "description": "AK4 rule only inside AG4 front five.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_front5_only",), "topk": 12, "max_positions": 12},
    {"variant": "al5_right_le90_no_extreme_backrank", "description": "Right <= 0.90 plus no extreme-low back-rank promotion.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_right_le90_no_extreme_backrank",), "topk": 12, "max_positions": 12},
    {"variant": "al6_right_75_90_no_extreme_backrank", "description": "Right 0.75-0.90 plus no extreme-low back-rank promotion.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_right_75_90_no_extreme_backrank",), "topk": 12, "max_positions": 12},
    {"variant": "al7_right_le90_front5_only", "description": "Right <= 0.90 and only AG4 front five are re-ordered.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low_right_le90_front5_only",), "topk": 12, "max_positions": 12},
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
    bull_hihi = state.eq("bull|hi|hi")
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")
    right_amt = ai.percentile(selected, "right_side_amount_ratio_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "bull_hihi_ml5_low":
        return protected_ml5_low(selected, score, bull_hihi)
    if rule == "bull_hihi_ml5_low_right_le90":
        return protected_ml5_low(selected, score, bull_hihi & right_amt.le(0.90))
    if rule == "bull_hihi_ml5_low_right_75_90":
        return protected_ml5_low(selected, score, bull_hihi & right_amt.between(0.75, 0.90, inclusive="both"))
    if rule == "bull_hihi_ml5_low_no_extreme_backrank":
        return protected_ml5_low(selected, score, bull_hihi, no_extreme_backrank=True)
    if rule == "bull_hihi_ml5_low_front5_only":
        return protected_ml5_low(selected, score, bull_hihi, front5_only=True)
    if rule == "bull_hihi_ml5_low_right_le90_no_extreme_backrank":
        return protected_ml5_low(selected, score, bull_hihi & right_amt.le(0.90), no_extreme_backrank=True)
    if rule == "bull_hihi_ml5_low_right_75_90_no_extreme_backrank":
        return protected_ml5_low(selected, score, bull_hihi & right_amt.between(0.75, 0.90, inclusive="both"), no_extreme_backrank=True)
    if rule == "bull_hihi_ml5_low_right_le90_front5_only":
        return protected_ml5_low(selected, score, bull_hihi & right_amt.le(0.90), front5_only=True)
    raise ValueError(f"Unknown Round AL rule: {rule}")


def protected_ml5_low(
    selected: pd.DataFrame,
    score: pd.Series,
    mask: pd.Series,
    *,
    no_extreme_backrank: bool = False,
    front5_only: bool = False,
) -> pd.Series:
    ai = load_ai()
    effective = mask.copy()
    if no_extreme_backrank:
        ml5 = ai.percentile(selected, "ml_pred_5d")
        raw_rank = ai.percentile(selected, "raw_rank")
        effective &= ~(ml5.le(-0.018) & raw_rank.gt(8.0))
    if front5_only:
        ag4_rank = score.groupby(selected["signal_time"]).rank(method="first", ascending=False)
        effective &= ag4_rank.le(5.0)
    adjusted = ai.factor_rank_score(selected, "ml_pred_5d", ascending=True, priority=410000.0)
    return score.where(~effective, adjusted)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_al_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_al_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_al_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_al_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_al_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_al_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AL_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AL.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_al_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"ac5": read_json(AC5_ARTIFACT / "summary.json"), "ag4": read_json(AG4_ARTIFACT / "summary.json"), "ak4": read_json(AK4_ARTIFACT / "summary.json")}
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_al_comparison.csv"
    signal_path = output_root / "runs/round_al_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "variant" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AL",
        "",
        "范围：保持 AG4/AK4 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 AK4 的 bull|hi|hi 局部排序保护。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_ak4_ret | vs_ak4_mdd | vs_ak4_sharpe | vs_ag4_ret | vs_ag4_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_ak4_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_ak4_mdd_delta')):.4f} | {safe_float(row.get('vs_ak4_sharpe_delta')):.4f} | "
            f"{safe_float(row.get('vs_ag4_total_return_delta')):.4f} | {safe_float(row.get('vs_ag4_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_al_comparison.csv`", "- `runs/round_al_signal_stats.csv`", "- `runs/round_al_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
