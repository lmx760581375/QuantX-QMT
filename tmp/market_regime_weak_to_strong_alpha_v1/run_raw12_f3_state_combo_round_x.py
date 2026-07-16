"""Round X local ablation tests around the Round W best variant.

This round reuses the Round W scoring implementation and only changes the
variant set. It keeps the f3 pool, max_positions=12, cash_equal sizing,
selector lag=1, next-day close execution, and original sell rules unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_state_combo_round_x"
ROUND_W_SCRIPT = ROOT / "run_raw12_f3_state_combo_round_w.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "x0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "x1_bad_ml5_full", "description": "5d ML order in all non-good f3 states.", "mode": "bad_ml", "horizon": 5, "topk": 12, "max_positions": 12},
    {"variant": "x2_excl_range_mid_lo", "description": "Keep original order in range|mid|lo, 5d ML elsewhere in bad states.", "mode": "exclude_states", "horizon": 5, "exclude_states": ("range|mid|lo",), "topk": 12, "max_positions": 12},
    {"variant": "x3_rmm_bmm_keep_top1_no_excl", "description": "No state exclusion; preserve original rank1 in range|mid|mid and bear|mid|mid.", "mode": "state_keep_top", "horizon": 5, "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "x4_excl_rml_keep_rmm_top1", "description": "Exclude range|mid|lo and preserve original rank1 only in range|mid|mid.", "mode": "exclude_and_keep", "horizon": 5, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "x5_excl_rml_keep_bmm_top1", "description": "Exclude range|mid|lo and preserve original rank1 only in bear|mid|mid.", "mode": "exclude_and_keep", "horizon": 5, "exclude_states": ("range|mid|lo",), "keep_states": ("bear|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "x6_excl_rml_keep_rmm_bmm_top1", "description": "Round W best: exclude range|mid|lo and preserve original rank1 in range|mid|mid plus bear|mid|mid.", "mode": "exclude_and_keep", "horizon": 5, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "x7_excl_rml_keep_rmm_bmm_top2", "description": "Exclude range|mid|lo and preserve original top2 in range|mid|mid plus bear|mid|mid.", "mode": "exclude_and_keep", "horizon": 5, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 2, "topk": 12, "max_positions": 12},
)


def main() -> None:
    round_w = load_round_w()
    round_w.OUTPUT_NAME = OUTPUT_NAME
    round_w.VARIANTS = VARIANTS
    round_w.normalize_outputs = normalize_outputs
    round_w.main()


def load_round_w():
    spec = importlib.util.spec_from_file_location("round_w_harness", ROUND_W_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round W harness from {ROUND_W_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_x_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_x_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_x_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_x_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_x_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_x_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_STATE_COMBO_ROUND_X_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_STATE_COMBO_ROUND_X.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_x_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"raw12": read_json(RAW12_BASE / "summary.json"), "f3": read_json(ROUND_F_BEST / "summary.json")}
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_x_comparison.csv"
    signal_path = output_root / "runs/round_x_signal_stats.csv"
    diag_path = output_root / "runs/round_x_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 State Combo Round X",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只围绕 Round W 最佳规则做局部消融。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_f3_ret | vs_f3_mdd | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_f3_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_mdd_delta')):.4f} | {safe_float(row.get('vs_f3_sharpe_delta')):.4f} |"
        )
    if not diag.empty:
        lines.extend(["", "## OOS IC", "", "| horizon | avg_ic | avg_pearson | years |", "| ---: | ---: | ---: | ---: |"])
        for horizon, group in diag.groupby("horizon", sort=True):
            lines.append(
                f"| {int(horizon)} | {safe_float(group['spearman_ic'].mean()):.4f} | "
                f"{safe_float(group['pearson_ic'].mean()):.4f} | {int(group['year'].nunique())} |"
            )
    lines.extend(["", "## Artifacts", "", "- `runs/round_x_comparison.csv`", "- `runs/round_x_signal_stats.csv`", "- `runs/round_x_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_STATE_COMBO_ROUND_X.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
