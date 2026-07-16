"""Round AK: path-scan candidates on top of AG4.

Round AI/AJ showed that broad posterior IC re-ordering does not reliably lift
AG4. This round uses only non-leaking, signal-time factors that looked useful in
the AG4 candidate path scan, while keeping the AG4 pool and execution semantics
unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_ak"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ak0_ag4_reference", "description": "AG4 reproduced.", "mode": "ag4_reference", "topk": 12, "max_positions": 12},
    {"variant": "ak1_range_hilo_ml357_high", "description": "range|hi|lo ordered by higher ml_pred_357d.", "mode": "ag4_rules", "rules": ("range_hilo_ml357_high",), "topk": 12, "max_positions": 12},
    {"variant": "ak2_range_lohi_ml3_high", "description": "range|lo|hi ordered by higher ml_pred_3d.", "mode": "ag4_rules", "rules": ("range_lohi_ml3_high",), "topk": 12, "max_positions": 12},
    {"variant": "ak3_bull_midmid_ml7_low", "description": "bull|mid|mid ordered by lower ml_pred_7d.", "mode": "ag4_rules", "rules": ("bull_midmid_ml7_low",), "topk": 12, "max_positions": 12},
    {"variant": "ak4_bull_hihi_ml5_low", "description": "bull|hi|hi ordered by lower ml_pred_5d.", "mode": "ag4_rules", "rules": ("bull_hihi_ml5_low",), "topk": 12, "max_positions": 12},
    {"variant": "ak5_bull_himid_ml5_low", "description": "bull|hi|mid ordered by lower ml_pred_5d.", "mode": "ag4_rules", "rules": ("bull_himid_ml5_low",), "topk": 12, "max_positions": 12},
    {"variant": "ak6_bull_midmid_low_amp", "description": "bull|mid|mid ordered by lower amplitude.", "mode": "ag4_rules", "rules": ("bull_midmid_low_amp",), "topk": 12, "max_positions": 12},
    {"variant": "ak7_bull_midmid_raw_rank_high", "description": "bull|mid|mid ordered by stronger original raw rank.", "mode": "ag4_rules", "rules": ("bull_midmid_raw_rank_high",), "topk": 12, "max_positions": 12},
    {"variant": "ak8_combo_range_ml", "description": "Combine range|hi|lo ml357 and range|lo|hi ml3 ordering.", "mode": "ag4_rules", "rules": ("range_hilo_ml357_high", "range_lohi_ml3_high"), "topk": 12, "max_positions": 12},
    {"variant": "ak9_combo_bull_ml", "description": "Combine bull-state ML reversal rules.", "mode": "ag4_rules", "rules": ("bull_midmid_ml7_low", "bull_hihi_ml5_low", "bull_himid_ml5_low"), "topk": 12, "max_positions": 12},
    {"variant": "ak10_combo_range_bull_core", "description": "Combine strongest range and bull path-scan rules.", "mode": "ag4_rules", "rules": ("range_hilo_ml357_high", "range_lohi_ml3_high", "bull_midmid_ml7_low", "bull_hihi_ml5_low"), "topk": 12, "max_positions": 12},
    {"variant": "ak11_range_hilo_ml357_keep_top1", "description": "range|hi|lo ml357 ordering while preserving AG4 day rank1.", "mode": "ag4_rules", "rules": ("range_hilo_ml357_keep_top1",), "topk": 12, "max_positions": 12},
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
    range_hilo = state.eq("range|hi|lo")
    range_lohi = state.eq("range|lo|hi")
    bull_midmid = state.eq("bull|mid|mid")
    bull_hihi = state.eq("bull|hi|hi")
    bull_himid = state.eq("bull|hi|mid")
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "range_hilo_ml357_high":
        return ai.apply_factor(selected, score, range_hilo, "ml_pred_357d", ascending=False, priority=410000.0)
    if rule == "range_lohi_ml3_high":
        return ai.apply_factor(selected, score, range_lohi, "ml_pred_3d", ascending=False, priority=410000.0)
    if rule == "bull_midmid_ml7_low":
        return ai.apply_factor(selected, score, bull_midmid, "ml_pred_7d", ascending=True, priority=410000.0)
    if rule == "bull_hihi_ml5_low":
        return ai.apply_factor(selected, score, bull_hihi, "ml_pred_5d", ascending=True, priority=410000.0)
    if rule == "bull_himid_ml5_low":
        return ai.apply_factor(selected, score, bull_himid, "ml_pred_5d", ascending=True, priority=410000.0)
    if rule == "bull_midmid_low_amp":
        return ai.apply_factor(selected, score, bull_midmid, "amplitude_inv", ascending=False, priority=410000.0)
    if rule == "bull_midmid_raw_rank_high":
        return ai.apply_factor(selected, score, bull_midmid, "raw_rank_inv", ascending=False, priority=410000.0)
    if rule == "range_hilo_ml357_keep_top1":
        adjusted = ai.factor_rank_score(selected, "ml_pred_357d", ascending=False, priority=410000.0)
        adjusted = ai.preserve_rank1(selected, score, adjusted, range_hilo)
        return score.where(~range_hilo, adjusted)
    raise ValueError(f"Unknown Round AK rule: {rule}")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_ak_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_ak_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_ak_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_ak_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_ak_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_ak_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AK_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AK.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_ak_comparison.csv"
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
    comparison_path = output_root / "runs/round_ak_comparison.csv"
    signal_path = output_root / "runs/round_ak_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "variant" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AK",
        "",
        "范围：保持 AG4 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试路径扫描得到的非泄漏局部排序。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_ak_comparison.csv`", "- `runs/round_ak_signal_stats.csv`", "- `runs/round_ak_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
