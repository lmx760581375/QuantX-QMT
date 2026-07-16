"""Round T bad-state-only ranking refinements for f3 weak-to-strong.

This script reuses Round S feature/prediction generation and the Round F
formal QuantX backtest harness. It keeps the f3 pool unchanged and tests only
small bad-state order changes, including preserving the original top ranks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_bad_state_rank_round_t"
ROUND_S_SCRIPT = ROOT / "run_raw12_f3_rank_ml_round_s.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "t0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "t1_bad_ml5_full", "description": "Round S winner: use 5d ML order only outside f3 good states.", "mode": "bad_ml", "horizon": 5, "topk": 12, "max_positions": 12},
    {"variant": "t2_bad_ml7_full", "description": "use 7d ML order only outside f3 good states.", "mode": "bad_ml", "horizon": 7, "topk": 12, "max_positions": 12},
    {"variant": "t3_bad_ml357_full", "description": "use 3/5/7d ensemble order only outside f3 good states.", "mode": "bad_ml", "horizon": 357, "topk": 12, "max_positions": 12},
    {"variant": "t4_bad_ml5_keep_top1", "description": "bad-state 5d ML reorder while preserving original rank1.", "mode": "bad_ml_keep", "horizon": 5, "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "t5_bad_ml5_keep_top2", "description": "bad-state 5d ML reorder while preserving original top2.", "mode": "bad_ml_keep", "horizon": 5, "keep_top": 2, "topk": 12, "max_positions": 12},
    {"variant": "t6_bad_ml7_keep_top1", "description": "bad-state 7d ML reorder while preserving original rank1.", "mode": "bad_ml_keep", "horizon": 7, "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "t7_bad_ml357_keep_top1", "description": "bad-state ensemble reorder while preserving original rank1.", "mode": "bad_ml_keep", "horizon": 357, "keep_top": 1, "topk": 12, "max_positions": 12},
    {"variant": "t8_bad_ml5_tail4_plus", "description": "bad-state 5d ML reorder only for original ranks 4+.", "mode": "bad_ml_tail", "horizon": 5, "tail_min": 4, "topk": 12, "max_positions": 12},
    {"variant": "t9_bad_ml5_blend30", "description": "bad-state order by 70% original rank and 30% 5d ML rank.", "mode": "bad_ml_blend", "horizon": 5, "ml_weight": 0.30, "topk": 12, "max_positions": 12},
    {"variant": "t10_bad_ml5_blend50", "description": "bad-state order by 50% original rank and 50% 5d ML rank.", "mode": "bad_ml_blend", "horizon": 5, "ml_weight": 0.50, "topk": 12, "max_positions": 12},
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
    bad_state = ~selected["f3_good_state"].fillna(False).astype(bool)
    pred_col = prediction_column(spec)
    ml_rank_score = bad_state_ml_rank_score(selected, pred_col)
    if mode == "bad_ml":
        return ml_rank_score.where(bad_state, base_score).fillna(base_score)
    if mode == "bad_ml_keep":
        keep_top = int(spec.get("keep_top", 1))
        adjusted = keep_original_top_score(selected, ml_rank_score, keep_top=keep_top)
        return adjusted.where(bad_state, base_score).fillna(base_score)
    if mode == "bad_ml_tail":
        tail_min = int(spec.get("tail_min", 4))
        adjusted = keep_original_top_score(selected, ml_rank_score, keep_top=tail_min - 1)
        return adjusted.where(bad_state, base_score).fillna(base_score)
    if mode == "bad_ml_blend":
        adjusted = blended_rank_score(selected, pred_col, ml_weight=float(spec.get("ml_weight", 0.3)))
        return adjusted.where(bad_state, base_score).fillna(base_score)
    raise ValueError(f"Unknown Round T mode: {mode}")


def prediction_column(spec: dict[str, Any]) -> str:
    horizon = int(spec.get("horizon", 5))
    if horizon == 357:
        return "ml_pred_357d"
    return f"ml_pred_{horizon}d"


def bad_state_ml_rank_score(selected: pd.DataFrame, pred_col: str) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    return (100000.0 - ml_rank).where(pred.notna()) - base_rank * 1e-4


def keep_original_top_score(selected: pd.DataFrame, ml_rank_score: pd.Series, *, keep_top: int) -> pd.Series:
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    preserved = base_rank <= keep_top
    preserved_score = 200000.0 - base_rank
    return ml_rank_score.where(~preserved, preserved_score)


def blended_rank_score(selected: pd.DataFrame, pred_col: str, *, ml_weight: float) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    combo = (1.0 - ml_weight) * base_rank + ml_weight * ml_rank
    return (100000.0 - combo).where(pred.notna())


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_s_signal_stats.csv", runs_dir / "round_t_signal_stats.csv"),
        (runs_dir / "round_s_ml_oos_diagnostics.csv", runs_dir / "round_t_ml_oos_diagnostics.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_t_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_t_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_t_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_t_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_BAD_STATE_RANK_ROUND_T_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_BAD_STATE_RANK_ROUND_T.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_t_comparison.csv"
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
    comparison_path = output_root / "runs/round_t_comparison.csv"
    signal_path = output_root / "runs/round_t_signal_stats.csv"
    diag_path = output_root / "runs/round_t_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 Bad-State Rank Round T",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只在非 f3 好状态里测试排序微调。",
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
    lines.extend(["", "## Artifacts", "", "- `runs/round_t_comparison.csv`", "- `runs/round_t_signal_stats.csv`", "- `runs/round_t_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_BAD_STATE_RANK_ROUND_T.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
