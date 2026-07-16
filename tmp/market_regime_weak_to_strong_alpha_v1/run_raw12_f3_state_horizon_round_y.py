"""Round Y horizon and state-protection tests around the Round X winner.

This round keeps the f3 pool, max_positions=12, cash_equal sizing,
selector lag=1, next-day close execution, and original sell rules unchanged.
It tests whether the stronger 7d/357 OOS IC can improve the localized bad-state
ranking rule found in Round W/X.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_state_horizon_round_y"
ROUND_W_SCRIPT = ROOT / "run_raw12_f3_state_combo_round_w.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "y0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "y1_h5_w6_reference", "description": "Round W/X winner reproduced with 5d ML.", "mode": "exclude_and_keep", "horizon": 5, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y2_h7_w6_rule", "description": "Same localized rule as w6, but using 7d ML prediction.", "mode": "exclude_and_keep", "horizon": 7, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y3_h3_w6_rule", "description": "Same localized rule as w6, but using 3d ML prediction.", "mode": "exclude_and_keep", "horizon": 3, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y4_h357_w6_rule", "description": "Same localized rule as w6, but using 3/5/7d ensemble prediction.", "mode": "exclude_and_keep", "horizon": 357, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y5_h7_bad_ml_full", "description": "7d ML order in all non-good f3 states.", "mode": "bad_ml", "horizon": 7, "topk": 12, "max_positions": 12},
    {"variant": "y6_h7_excl_range_mid_lo", "description": "7d ML in bad states except range|mid|lo keeps original order.", "mode": "exclude_states", "horizon": 7, "exclude_states": ("range|mid|lo",), "topk": 12, "max_positions": 12},
    {"variant": "y7_h7_keep_rmm_bmm_top1_no_excl", "description": "7d ML in bad states, preserve original rank1 in range|mid|mid and bear|mid|mid.", "mode": "state_keep_top", "horizon": 7, "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y8_h7_excl_rml_keep_rmm_top1", "description": "7d ML: exclude range|mid|lo and preserve range|mid|mid rank1.", "mode": "exclude_and_keep", "horizon": 7, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y9_h7_excl_rml_keep_bmm_top1", "description": "7d ML: exclude range|mid|lo and preserve bear|mid|mid rank1.", "mode": "exclude_and_keep", "horizon": 7, "exclude_states": ("range|mid|lo",), "keep_states": ("bear|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "y10_h7_excl_rml_keep_rmm_bmm_top2", "description": "7d ML: exclude range|mid|lo and preserve rank top2 in range|mid|mid plus bear|mid|mid.", "mode": "exclude_and_keep", "horizon": 7, "exclude_states": ("range|mid|lo",), "keep_states": ("range|mid|mid", "bear|mid|mid"), "keep_top": 2, "topk": 12, "max_positions": 12},
)


def main() -> None:
    round_w = load_round_w()
    round_w.OUTPUT_NAME = OUTPUT_NAME
    round_w.VARIANTS = VARIANTS
    round_w.variant_score = variant_score
    round_w.normalize_outputs = normalize_outputs
    round_w.main()


def load_round_w():
    spec = importlib.util.spec_from_file_location("round_w_harness", ROUND_W_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round W harness from {ROUND_W_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    if mode == "f3_reference":
        return base_score

    state_key = state_key_series(selected)
    bad_state = ~selected["f3_good_state"].fillna(False).astype(bool)
    adjusted = ml_rank_score(selected, prediction_column(spec))

    if mode == "bad_ml":
        use_ml = bad_state
    elif mode == "exclude_states":
        use_ml = bad_state & ~state_key.isin(tuple(spec.get("exclude_states", ())))
    elif mode == "state_keep_top":
        use_ml = bad_state
        keep_mask = state_key.isin(tuple(spec.get("keep_states", ())))
        adjusted = preserve_original_top(selected, adjusted, keep_mask=keep_mask, keep_top=int(spec.get("keep_top", 1)))
    elif mode == "exclude_and_keep":
        use_ml = bad_state & ~state_key.isin(tuple(spec.get("exclude_states", ())))
        keep_mask = state_key.isin(tuple(spec.get("keep_states", ())))
        adjusted = preserve_original_top(selected, adjusted, keep_mask=keep_mask, keep_top=int(spec.get("keep_top", 1)))
    else:
        raise ValueError(f"Unknown Round Y mode: {mode}")
    return adjusted.where(use_ml, base_score).fillna(base_score)


def state_key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def prediction_column(spec: dict[str, Any]) -> str:
    horizon = int(spec.get("horizon", 5))
    if horizon == 357:
        return "ml_pred_357d"
    return f"ml_pred_{horizon}d"


def ml_rank_score(selected: pd.DataFrame, pred_col: str) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    return (100000.0 - ml_rank).where(pred.notna()) - base_rank * 1e-4


def preserve_original_top(selected: pd.DataFrame, adjusted: pd.Series, *, keep_mask: pd.Series, keep_top: int) -> pd.Series:
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    preserved = keep_mask & base_rank.le(keep_top)
    preserved_score = 200000.0 - base_rank
    return adjusted.where(~preserved, preserved_score)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_y_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_y_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_y_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_y_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_y_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_y_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_STATE_HORIZON_ROUND_Y_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_STATE_HORIZON_ROUND_Y.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_y_comparison.csv"
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
    comparison_path = output_root / "runs/round_y_comparison.csv"
    signal_path = output_root / "runs/round_y_signal_stats.csv"
    diag_path = output_root / "runs/round_y_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 State Horizon Round Y",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试 Round X 胜出局部排序规则的 horizon 与状态保护。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_y_comparison.csv`", "- `runs/round_y_signal_stats.csv`", "- `runs/round_y_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_STATE_HORIZON_ROUND_Y.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
