"""Round AV: narrow bear|lo|lo ordering tests on top of AU2.

AU2 improved AQ4 only slightly by pushing the realized-PnL weak bear|lo|lo
state to the bottom. This round keeps the same top12 candidate set, lag=1,
next-day close execution, cash_equal sizing, and sell rules, then tests whether
the bear|lo|lo slice should be ordered by observable live factors instead of the
default raw-rank-inverse bottom order.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_ag4_path_scan_round_av"
ROUND_AU_SCRIPT = ROOT / "run_raw12_f3_ag4_path_scan_round_au.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "av0_am2_reference", "description": "AM2 reproduced.", "mode": "am2_reference", "topk": 12, "max_positions": 12},
    {"variant": "av1_aq4_reference", "description": "AQ4 reproduced.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom",), "topk": 12, "max_positions": 12},
    {"variant": "av2_au2_reference", "description": "AU2 reproduced: AQ4 plus demote all bear|lo|lo by raw-rank inverse.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_all_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av3_au2_bear_lolo_rsv_low_bottom", "description": "AU2 but order bear|lo|lo bottom slice by lower rsv_short.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_rsv_low_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av4_au2_bear_lolo_liq_high_bottom", "description": "AU2 but order bear|lo|lo bottom slice by higher liquidity_rank.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_liq_high_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av5_au2_bear_lolo_ret1_high_bottom", "description": "AU2 but order bear|lo|lo bottom slice by higher ret1_abs.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_ret1_high_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av6_au2_bear_lolo_raw_high_bottom", "description": "AU2 but order bear|lo|lo bottom slice by higher raw_rank.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_raw_high_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av7_aq4_bear_lolo_high_rsv_bottom", "description": "AQ4 plus demote only bear|lo|lo with high rsv_short.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_high_rsv_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av8_aq4_bear_lolo_low_liq_bottom", "description": "AQ4 plus demote only bear|lo|lo with low liquidity_rank.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_low_liq_bottom"), "topk": 12, "max_positions": 12},
    {"variant": "av9_aq4_bear_lolo_low_ret1_bottom", "description": "AQ4 plus demote only bear|lo|lo with low ret1_abs.", "mode": "am2_rules", "rules": ("tail_range_bad_bottom", "bear_lolo_low_ret1_bottom"), "topk": 12, "max_positions": 12},
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
    bear_lolo = state.eq("bear|lo|lo")

    if rule == "bear_lolo_rsv_low_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "rsv_short", ascending=True)
        return score.where(~bear_lolo, adjusted)
    if rule == "bear_lolo_liq_high_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "liquidity_rank", ascending=False)
        return score.where(~bear_lolo, adjusted)
    if rule == "bear_lolo_ret1_high_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "ret1_abs", ascending=False)
        return score.where(~bear_lolo, adjusted)
    if rule == "bear_lolo_raw_high_bottom":
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank", ascending=False)
        return score.where(~bear_lolo, adjusted)
    if rule == "bear_lolo_high_rsv_bottom":
        rsv_short = pd.to_numeric(selected["rsv_short"], errors="coerce") if "rsv_short" in selected else pd.Series(np.nan, index=selected.index)
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(bear_lolo & rsv_short.ge(70.0)), adjusted)
    if rule == "bear_lolo_low_liq_bottom":
        liquidity = pd.to_numeric(selected["liquidity_rank"], errors="coerce") if "liquidity_rank" in selected else pd.Series(np.nan, index=selected.index)
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(bear_lolo & liquidity.le(0.5)), adjusted)
    if rule == "bear_lolo_low_ret1_bottom":
        ret1_abs = pd.to_numeric(selected["ret1_abs"], errors="coerce") if "ret1_abs" in selected else pd.Series(np.nan, index=selected.index)
        cutoff = ret1_abs.groupby(selected["signal_time"]).transform("median")
        adjusted = asmod.tail_bottom_score(selected, score, "raw_rank_inv", ascending=True)
        return score.where(~(bear_lolo & ret1_abs.le(cutoff)), adjusted)
    if _AU_APPLY_RULE is None:
        raise RuntimeError("Round AU original apply_rule has not been initialized")
    return _AU_APPLY_RULE(selected, score, rule)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_av_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_av_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_av_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_av_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_av_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_av_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AV_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_AG4_PATH_SCAN_ROUND_AV.md"),
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
    path = output_root / "runs/round_av_comparison.csv"
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
    au2 = comparison.loc[comparison["variant"].eq("av2_au2_reference")]
    if not au2.empty:
        ref = au2.iloc[0]
        comparison["vs_au2_total_return_delta"] = comparison["total_return"] - float(ref["total_return"])
        comparison["vs_au2_mdd_delta"] = comparison["max_drawdown"] - float(ref["max_drawdown"])
        comparison["vs_au2_sharpe_delta"] = comparison["sharpe"] - float(ref["sharpe"])
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_av_comparison.csv"
    signal_path = output_root / "runs/round_av_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "variant" in signals else {}
    top1_change = signals.set_index("variant")["top1_changed_ratio"].to_dict() if not signals.empty and "variant" in signals else {}
    lines = [
        "# Raw12 F3 AG4 Path Scan Round AV",
        "",
        "范围：保持 AM2/AQ4/AU2 候选集合、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 `bear|lo|lo` 的局部排序。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_av_comparison.csv`", "- `runs/round_av_signal_stats.csv`", "- `runs/round_av_yearly_nav.csv`"])
    (output_root / "reports/RAW12_F3_AG4_PATH_SCAN_ROUND_AV.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
