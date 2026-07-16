"""Round AN: targeted return lift attempts on top of AK4/AM2.

Round AM found a small but real improvement from promoting range|hi|mid names by
lower 3-day walk-forward ML prediction. This round keeps the same AK4 candidate
pool, top12 capacity, close execution, cash_equal sizing, and original sells;
it only tests tighter priority/condition variants and a few additional local
state rules with strong buyable non-ST forward IC evidence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_an"
ROUND_AI_SCRIPT = ROOT / "run_raw12_f3_ag4_local_factor_round_ai.py"
AC5_ARTIFACT = ROOT / "raw12_f3_state_gate_round_ac/artifacts/ac5_lomid_right_amt_le50"
AG4_ARTIFACT = ROOT / "raw12_f3_ac5_bullmidhi_gate_round_ag/artifacts/ag4_bmh_low_amp_active_amt2_gt75"
AK4_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_ak/artifacts/ak4_bull_hihi_ml5_low"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "an0_ak4_reference", "description": "AK4 reproduced.", "mode": "ak4_reference", "topk": 12, "max_positions": 12},
    {"variant": "an1_am2_reference", "description": "AM2 reproduced: range|hi|mid ordered by lower ml_pred_3d at high priority.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p420",), "topk": 12, "max_positions": 12},
    {"variant": "an2_himid_ml3_low_p415", "description": "AM2 signal with softer priority, still above AK4 bull|hi|hi.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p415",), "topk": 12, "max_positions": 12},
    {"variant": "an3_himid_ml3_low_p409", "description": "AM2 signal below AK4 bull|hi|hi but above AG4 bull|mid|hi.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p409",), "topk": 12, "max_positions": 12},
    {"variant": "an4_himid_ml3_low_multi2", "description": "AM2 signal only when range|hi|mid has at least two names on the day.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_multi2",), "topk": 12, "max_positions": 12},
    {"variant": "an5_himid_ml3_low_keep_top1", "description": "AM2 signal while preserving AK4 day rank1.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_keep_top1",), "topk": 12, "max_positions": 12},
    {"variant": "an6_range_hilo_ml5_low", "description": "range|hi|lo ordered by lower ml_pred_5d.", "mode": "ak4_rules", "rules": ("range_hilo_ml5_low_p420",), "topk": 12, "max_positions": 12},
    {"variant": "an7_combo_himid_ml3_hilo_ml5", "description": "AM2 plus range|hi|lo lower ml_pred_5d.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p420", "range_hilo_ml5_low_p420"), "topk": 12, "max_positions": 12},
    {"variant": "an8_bull_unknown_turnover_high", "description": "bull|unknown|unknown ordered by higher turnover amount rank.", "mode": "ak4_rules", "rules": ("bull_unknown_turnover_high_p420",), "topk": 12, "max_positions": 12},
    {"variant": "an9_combo_himid_bull_unknown_turnover", "description": "AM2 plus bull|unknown|unknown higher turnover amount rank.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p420", "bull_unknown_turnover_high_p420"), "topk": 12, "max_positions": 12},
    {"variant": "an10_range_lomid_amp_inv_low", "description": "range|lo|mid ordered by lower amplitude_inv, following negative IC.", "mode": "ak4_rules", "rules": ("range_lomid_amp_inv_low_p420",), "topk": 12, "max_positions": 12},
    {"variant": "an11_bear_lomid_ml357_low", "description": "bear|lo|mid ordered by lower ml_pred_357d.", "mode": "ak4_rules", "rules": ("bear_lomid_ml357_low_p420",), "topk": 12, "max_positions": 12},
    {"variant": "an12_combo_core_return", "description": "AM2 plus the two strongest extra 5/7-day IC rules.", "mode": "ak4_rules", "rules": ("range_himid_ml3_low_p420", "range_hilo_ml5_low_p420", "bull_unknown_turnover_high_p420"), "topk": 12, "max_positions": 12},
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
        raise ValueError(f"Unknown Round AN mode: {mode}")
    score = ak4
    for rule in tuple(spec.get("rules", ())):
        score = apply_rule(selected, score, str(rule))
    return score.fillna(pd.to_numeric(selected["base_score"], errors="coerce"))


def apply_rule(selected: pd.DataFrame, score: pd.Series, rule: str) -> pd.Series:
    ai = load_ai()
    state = ai.state_key_series(selected)
    bull_midhi = state.eq("bull|mid|hi")
    bull_hihi = state.eq("bull|hi|hi")
    bull_unknown = state.eq("bull|unknown|unknown")
    range_himid = state.eq("range|hi|mid")
    range_hilo = state.eq("range|hi|lo")
    range_lomid = state.eq("range|lo|mid")
    bear_lomid = state.eq("bear|lo|mid")
    amt2 = ai.percentile(selected, "active_all_amount_ret2_pct252")

    if rule == "bmh_low_amp_active_amt2_gt75":
        return ai.apply_factor(selected, score, bull_midhi & amt2.gt(0.75), "amplitude_inv", ascending=False, priority=400000.0)
    if rule == "bull_hihi_ml5_low":
        return ai.apply_factor(selected, score, bull_hihi, "ml_pred_5d", ascending=True, priority=410000.0)
    if rule == "range_himid_ml3_low_p420":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_3d", ascending=True, priority=420000.0)
    if rule == "range_himid_ml3_low_p415":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_3d", ascending=True, priority=415000.0)
    if rule == "range_himid_ml3_low_p409":
        return ai.apply_factor(selected, score, range_himid, "ml_pred_3d", ascending=True, priority=409000.0)
    if rule == "range_himid_ml3_low_multi2":
        return ai.apply_factor(selected, score, range_himid & state_count(selected, state, "range|hi|mid").ge(2), "ml_pred_3d", ascending=True, priority=420000.0)
    if rule == "range_himid_ml3_low_keep_top1":
        adjusted = ai.factor_rank_score(selected, "ml_pred_3d", ascending=True, priority=420000.0)
        adjusted = ai.preserve_rank1(selected, score, adjusted, range_himid)
        return score.where(~range_himid, adjusted)
    if rule == "range_hilo_ml5_low_p420":
        return ai.apply_factor(selected, score, range_hilo, "ml_pred_5d", ascending=True, priority=420000.0)
    if rule == "bull_unknown_turnover_high_p420":
        return ai.apply_factor(selected, score, bull_unknown, "turnover_amount_rank", ascending=False, priority=420000.0)
    if rule == "range_lomid_amp_inv_low_p420":
        return ai.apply_factor(selected, score, range_lomid, "amplitude_inv", ascending=True, priority=420000.0)
    if rule == "bear_lomid_ml357_low_p420":
        return ai.apply_factor(selected, score, bear_lomid, "ml_pred_357d", ascending=True, priority=420000.0)
    raise ValueError(f"Unknown Round AN rule: {rule}")


def state_count(selected: pd.DataFrame, state: pd.Series, key: str) -> pd.Series:
    mask = state.eq(key)
    counts = mask.groupby(selected["signal_time"]).transform("sum")
    return pd.to_numeric(counts, errors="coerce").fillna(0.0)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_an_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_an_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_an_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_an_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_an_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_an_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AN_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AN.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_an_comparison.csv"
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
    comparison_path = output_root / "runs/round_an_comparison.csv"
    signal_path = output_root / "runs/round_an_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AN",
        "",
        "范围：保持 AK4 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 AM2 周边的收益增强排序。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_an_comparison.csv`", "- `runs/round_an_signal_stats.csv`", "- `runs/round_an_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
