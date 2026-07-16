"""Round V state-aware blend ranking for f3 weak-to-strong.

This round keeps the f3 pool, max_positions=12, cash_equal sizing,
selector lag=1, next-day close execution, and original sell rules unchanged.
It only tests state-aware blends/protections around the Round T/U winner:
5d ML ordering in non-good f3 states.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_state_blend_round_v"
ROUND_S_SCRIPT = ROOT / "run_raw12_f3_rank_ml_round_s.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


CORE_STATES = ("range|lo|mid", "range|mid|mid", "bear|mid|mid")
PNL_CORE_STATES = ("range|lo|mid", "range|mid|mid", "bear|lo|lo")
NEG_FORWARD_STATES = ("range|mid|mid", "bear|lo|lo", "range|mid|lo", "bear|lo|mid")


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "v0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "v1_bad_ml5_full", "description": "Round T/U return winner: 5d ML order in all non-good f3 states.", "mode": "bad_ml", "horizon": 5, "topk": 12, "max_positions": 12},
    {"variant": "v2_ml5_core_states", "description": "5d ML order only in range|lo|mid, range|mid|mid, bear|mid|mid.", "mode": "include_states", "horizon": 5, "include_states": CORE_STATES, "topk": 12, "max_positions": 12},
    {"variant": "v3_ml5_pnl_core_states", "description": "5d ML order only in the states with positive realized replacement attribution.", "mode": "include_states", "horizon": 5, "include_states": PNL_CORE_STATES, "topk": 12, "max_positions": 12},
    {"variant": "v4_ml5_range_mid_mid_only", "description": "5d ML order only in range|mid|mid, which Round U showed should not be excluded bluntly.", "mode": "include_states", "horizon": 5, "include_states": ("range|mid|mid",), "topk": 12, "max_positions": 12},
    {"variant": "v5_bad_ml5_core_full_else_blend30", "description": "Full 5d ML in core states, 30% ML blend in other non-good states.", "mode": "state_blend", "horizon": 5, "full_states": CORE_STATES, "default_ml_weight": 0.30, "topk": 12, "max_positions": 12},
    {"variant": "v6_bad_ml5_pnl_core_full_else_blend30", "description": "Full 5d ML in realized-PnL core states, 30% ML blend elsewhere.", "mode": "state_blend", "horizon": 5, "full_states": PNL_CORE_STATES, "default_ml_weight": 0.30, "topk": 12, "max_positions": 12},
    {"variant": "v7_bad_ml5_negative_forward_blend30", "description": "Full 5d ML except forward-negative states use 30% ML blend.", "mode": "state_blend", "horizon": 5, "full_states": "all_bad_except_weighted", "weighted_states": NEG_FORWARD_STATES, "weighted_ml_weight": 0.30, "default_ml_weight": 1.00, "topk": 12, "max_positions": 12},
    {"variant": "v8_bad_ml5_range_mid_mid_blend50", "description": "Full 5d ML except range|mid|mid uses 50% original/ML blend.", "mode": "state_blend", "horizon": 5, "full_states": "all_bad_except_weighted", "weighted_states": ("range|mid|mid",), "weighted_ml_weight": 0.50, "default_ml_weight": 1.00, "topk": 12, "max_positions": 12},
    {"variant": "v9_bad_ml5_excl_range_mid_lo", "description": "Full 5d ML in non-good states except range|mid|lo keeps original order.", "mode": "exclude_states", "horizon": 5, "exclude_states": ("range|mid|lo",), "topk": 12, "max_positions": 12},
    {"variant": "v10_bad_ml5_excl_bear_lolo", "description": "Full 5d ML in non-good states except bear|lo|lo keeps original order.", "mode": "exclude_states", "horizon": 5, "exclude_states": ("bear|lo|lo",), "topk": 12, "max_positions": 12},
    {"variant": "v11_bad_ml5_bear_mid_mid_keep_top1", "description": "Full 5d ML, but preserve original rank1 only in bear|mid|mid.", "mode": "state_keep_top", "horizon": 5, "keep_states": ("bear|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "v12_bad_ml5_range_mid_mid_keep_top1", "description": "Full 5d ML, but preserve original rank1 only in range|mid|mid.", "mode": "state_keep_top", "horizon": 5, "keep_states": ("range|mid|mid",), "keep_top": 1, "topk": 12, "max_positions": 12},
)


def main() -> None:
    round_s = load_round_s()
    round_s.OUTPUT_NAME = OUTPUT_NAME
    round_s.VARIANTS = VARIANTS
    round_s.variant_score = variant_score
    round_s.normalize_outputs = normalize_outputs
    round_s.main()


def load_round_s():
    spec = importlib.util.spec_from_file_location("round_s_harness", ROUND_S_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round S harness from {ROUND_S_SCRIPT}")
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
    pred_col = prediction_column(spec)

    if mode == "bad_ml":
        adjusted = ml_rank_score(selected, pred_col)
        return adjusted.where(bad_state, base_score).fillna(base_score)
    if mode == "include_states":
        use_ml = state_key.isin(tuple(spec.get("include_states", ())))
        adjusted = ml_rank_score(selected, pred_col)
        return adjusted.where(use_ml, base_score).fillna(base_score)
    if mode == "exclude_states":
        use_ml = bad_state & ~state_key.isin(tuple(spec.get("exclude_states", ())))
        adjusted = ml_rank_score(selected, pred_col)
        return adjusted.where(use_ml, base_score).fillna(base_score)
    if mode == "state_blend":
        adjusted = state_blend_score(selected, pred_col, spec, state_key, bad_state)
        return adjusted.where(bad_state, base_score).fillna(base_score)
    if mode == "state_keep_top":
        adjusted = ml_rank_score(selected, pred_col)
        keep_mask = state_key.isin(tuple(spec.get("keep_states", ())))
        adjusted = preserve_original_top(selected, adjusted, keep_mask=keep_mask, keep_top=int(spec.get("keep_top", 1)))
        return adjusted.where(bad_state, base_score).fillna(base_score)
    raise ValueError(f"Unknown Round V mode: {mode}")


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


def blended_rank_score(selected: pd.DataFrame, pred_col: str, *, ml_weight: float) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    combo = (1.0 - ml_weight) * base_rank + ml_weight * ml_rank
    return (100000.0 - combo).where(pred.notna())


def state_blend_score(selected: pd.DataFrame, pred_col: str, spec: dict[str, Any], state_key: pd.Series, bad_state: pd.Series) -> pd.Series:
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    score = base_score.copy()
    default_weight = float(spec.get("default_ml_weight", 0.30))
    weighted_states = tuple(spec.get("weighted_states", ()))
    weighted_weight = float(spec.get("weighted_ml_weight", default_weight))
    full_states = spec.get("full_states", ())

    default_adjusted = rank_score_for_weight(selected, pred_col, default_weight)
    score = default_adjusted.where(bad_state, score)

    if full_states == "all_bad_except_weighted":
        full_mask = bad_state & ~state_key.isin(weighted_states)
    else:
        full_mask = bad_state & state_key.isin(tuple(full_states))
    full_adjusted = ml_rank_score(selected, pred_col)
    score = full_adjusted.where(full_mask, score)

    if weighted_states:
        weighted_adjusted = rank_score_for_weight(selected, pred_col, weighted_weight)
        weighted_mask = bad_state & state_key.isin(weighted_states)
        score = weighted_adjusted.where(weighted_mask, score)
    return score


def rank_score_for_weight(selected: pd.DataFrame, pred_col: str, weight: float) -> pd.Series:
    if weight >= 0.999:
        return ml_rank_score(selected, pred_col)
    if weight <= 0.001:
        return pd.to_numeric(selected["base_score"], errors="coerce")
    return blended_rank_score(selected, pred_col, ml_weight=weight)


def preserve_original_top(selected: pd.DataFrame, adjusted: pd.Series, *, keep_mask: pd.Series, keep_top: int) -> pd.Series:
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    preserved = keep_mask & base_rank.le(keep_top)
    preserved_score = 200000.0 - base_rank
    return adjusted.where(~preserved, preserved_score)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_v_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_v_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_v_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_v_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_v_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_v_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_STATE_BLEND_ROUND_V_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_STATE_BLEND_ROUND_V.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_v_comparison.csv"
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
    comparison_path = output_root / "runs/round_v_comparison.csv"
    signal_path = output_root / "runs/round_v_signal_stats.csv"
    diag_path = output_root / "runs/round_v_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 State Blend Round V",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试状态内 ML 排序混合和局部 top1 保护。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_v_comparison.csv`", "- `runs/round_v_signal_stats.csv`", "- `runs/round_v_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_STATE_BLEND_ROUND_V.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
