"""Round AM: additive local ranking rules on top of AK4.

Round AL showed that protecting AK4's bull|hi|hi low-ML promotion does not beat
AK4. This round keeps AK4 unchanged and tests a small set of extra local
state/factor orderings discovered from buyable non-ST candidate IC scans. The
rules only use signal-time fields; future returns are used only in the offline
scan report, not in the generated signals.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_am"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"
AK4_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_ak/artifacts/ak4_bull_hihi_ml5_low"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "am0_ak4_reference", "description": "AK4 reproduced: bull|hi|hi ordered by lower ml_pred_5d.", "mode": "ak4_reference", "topk": 12, "max_positions": 12},
    {"variant": "am1_range_himid_ml357_low", "description": "AK4 plus range|hi|mid ordered by lower ml_pred_357d.", "mode": "ak4_rules", "rules": ("range_himid_ml357_low",), "topk": 12, "max_positions": 12},
    {"variant": "am2_range_himid_ml3_low", "description": "AK4 plus range|hi|mid ordered by lower ml_pred_3d.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low",), "topk": 12, "max_positions": 12},
    {"variant": "am3_bear_lomid_turnover_high", "description": "AK4 plus bear|lo|mid ordered by higher turnover amount rank.", "mode": "ak4_rules", "rules": ("bear_lomid_turnover_high",), "topk": 12, "max_positions": 12},
    {"variant": "am4_range_himid_rsv_short_high", "description": "AK4 plus range|hi|mid ordered by higher rsv_short.", "mode": "ak4_rules", "rules": ("range_himid_rsv_short_high",), "topk": 12, "max_positions": 12},
    {"variant": "am5_range_midhi_raw_rank_high", "description": "AK4 plus range|mid|hi ordered by stronger original raw rank.", "mode": "ak4_rules", "rules": ("range_midhi_raw_rank_high",), "topk": 12, "max_positions": 12},
    {"variant": "am6_range_hilo_ml5_low", "description": "AK4 plus range|hi|lo ordered by lower ml_pred_5d.", "mode": "ak4_rules", "rules": ("range_hilo_ml5_low",), "topk": 12, "max_positions": 12},
    {"variant": "am7_range_lomid_rsv_short_high", "description": "AK4 plus range|lo|mid ordered by higher rsv_short.", "mode": "ak4_rules", "rules": ("range_lomid_rsv_short_high",), "topk": 12, "max_positions": 12},
    {"variant": "am8_combo_himid_ml357_bear_turnover", "description": "AK4 plus range|hi|mid low ml357 and bear|lo|mid high turnover.", "mode": "ak4_rules", "rules": ("range_himid_ml357_low", "bear_lomid_turnover_high"), "topk": 12, "max_positions": 12},
    {"variant": "am9_combo_himid_rsv_midhi_raw", "description": "AK4 plus range|hi|mid high rsv_short and range|mid|hi strong raw rank.", "mode": "ak4_rules", "rules": ("range_himid_rsv_short_high", "range_midhi_raw_rank_high"), "topk": 12, "max_positions": 12},
    {"variant": "am10_combo_core_scan", "description": "AK4 plus the compact IC scan bundle: himid low ml357, bear turnover, midhi raw rank.", "mode": "ak4_rules", "rules": ("range_himid_ml357_low", "bear_lomid_turnover_high", "range_midhi_raw_rank_high"), "topk": 12, "max_positions": 12},
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
    mode = str(spec["mode"])
    if mode == "ak4_reference":
        return ak4.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))
    if mode != "ak4_rules":
        raise ValueError(f"Unknown Round AM mode: {mode}")
    score = ak4
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule))
    return score.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    ai = load_ai()
    state = ai.state_key_series(selected)
    bull_midhi = state.eq("bull|mid|hi")
    bull_hihi = state.eq("bull|hi|hi")
    range_himid = state.eq("range|hi|mid")
    range_midhi = state.eq("range|mid|hi")
    range_hilo = state.eq("range|hi|lo")
    range_lomid = state.eq("range|lo|mid")
    bear_lomid = state.eq("bear|lo|mid")
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "bull_hihi_ml5_low":
        return ai.apply_factor(selected, score, bull_hihi, "ml_pred_5d", ascending=True, priority=410000.0)
    if rule == "range_himid_ml357_low":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_357d", ascending=True, priority=420000.0)
    if rule == "range_himid_ml3_low":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_3d", ascending=True, priority=420000.0)
    if rule == "bear_lomid_turnover_high":
        return ai.apply_factor(selected, score, bear_lomid, "turnover_amount_rank", ascending=False, priority=420000.0)
    if rule == "range_himid_rsv_short_high":
        return ai.apply_factor(selected, score, range_himid, "rsv_short", ascending=False, priority=420000.0)
    if rule == "range_midhi_raw_rank_high":
        return ai.apply_factor(selected, score, range_midhi, "raw_rank_inv", ascending=False, priority=420000.0)
    if rule == "range_hilo_ml5_low":
        return ai.apply_factor(selected, score, range_hilo, "ml_pred_5d", ascending=True, priority=420000.0)
    if rule == "range_lomid_rsv_short_high":
        return ai.apply_factor(selected, score, range_lomid, "rsv_short", ascending=False, priority=420000.0)
    raise ValueError(f"Unknown Round AM rule: {rule}")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_am_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_am_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_am_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_am_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_am_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_am_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AM_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AM.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_am_comparison.csv"
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
    comparison_path = output_root / "runs/round_am_comparison.csv"
    signal_path = output_root / "runs/round_am_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AM",
        "",
        "范围：保持 AK4 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只在 AK4 基础上追加局部状态排序。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_am_comparison.csv`", "- `runs/round_am_signal_stats.csv`", "- `runs/round_am_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
