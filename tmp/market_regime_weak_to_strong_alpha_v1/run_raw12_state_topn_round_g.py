"""Round G state-aware TopN/ranking tests for weak-to-strong raw top12.

This wrapper reuses the Round F temporary research harness while replacing only
the variant definitions and signal selection logic. It keeps the formal QuantX
backtest path and original execution semantics: lag=1, next-session close,
cash_equal buys, original sell rules, and no candidate expansion beyond the
original score-ranked raw top12 pool.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_state_topn_round_g"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


STATE_TOPN_MAP = {
    "range|hi|lo": 4,
    "bull|mid|mid": 6,
    "bull|hi|hi": 8,
    "bear|lo|lo": 4,
    "bull|unknown|unknown": 6,
    "bear|mid|mid": 10,
    "bear|mid|lo": 6,
    "range|hi|mid": 4,
    "range|lo|hi": 4,
    "range|mid|hi": 4,
    "bear|lo|mid": 4,
    "range|mid|lo": 8,
    "range|unknown|unknown": 8,
}

WEAK_STATES = {
    "range|unknown|unknown",
    "range|mid|lo",
    "range|lo|hi",
    "bear|lo|mid",
}

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "g0_f3_reference",
        "description": "Round F best rule: good states top12, otherwise top8.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g1_f3_plus_range_weak_top6",
        "description": "F3 rule, with known weak range states cut further to top6.",
        "mode": "f3_plus_range_weak_top6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g2_range_weak_top6_else12",
        "description": "Only weak range/bear states cut to top6; all other states keep top12.",
        "mode": "weak_state_top6_else12",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g3_good_top12_else_top6",
        "description": "Good states top12, all other states top6.",
        "mode": "good_top12_else_top6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g4_state_topn_map",
        "description": "State3-specific TopN map from 3/5/7 day label analysis.",
        "mode": "state_topn_map",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g5_state_topn_map_floor6",
        "description": "State3-specific TopN map, but never below top6.",
        "mode": "state_topn_map_floor6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g6_weak_state_top4_else12",
        "description": "Known weak states cut to top4; all other states keep top12.",
        "mode": "weak_state_top4_else12",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g7_state_map_quality_reorder",
        "description": "State TopN map plus light bad-state quality reordering.",
        "mode": "state_topn_map_quality_reorder",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "g8_f3_quality_reorder",
        "description": "F3 selection plus light bad-state quality reordering.",
        "mode": "f3_quality_reorder",
        "topk": 12,
        "max_positions": 12,
    },
)


def main() -> None:
    round_f = load_round_f()
    patch_round_f(round_f)
    round_f.main()
    normalize_outputs(ROOT / OUTPUT_NAME)


def load_round_f():
    spec = importlib.util.spec_from_file_location("round_f_harness", ROUND_F_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round F harness from {ROUND_F_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_round_f(round_f) -> None:
    round_f.OUTPUT_NAME = OUTPUT_NAME
    round_f.VARIANTS = VARIANTS
    round_f.select_variant = select_variant
    round_f.score_variant = score_variant


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    state = state3(frame)
    good = good_state(frame)
    mode = str(spec["mode"])

    if mode == "f3_reference" or mode == "f3_quality_reorder":
        keep = (good & (rank <= 12)) | (~good & (rank <= 8))
    elif mode == "f3_plus_range_weak_top6":
        keep = (good & (rank <= 12)) | (~good & (rank <= 8))
        keep &= (~state.isin(WEAK_STATES)) | (rank <= 6)
    elif mode == "weak_state_top6_else12":
        keep = (~state.isin(WEAK_STATES) & (rank <= 12)) | (state.isin(WEAK_STATES) & (rank <= 6))
    elif mode == "good_top12_else_top6":
        keep = (good & (rank <= 12)) | (~good & (rank <= 6))
    elif mode == "state_topn_map" or mode == "state_topn_map_quality_reorder":
        topn = state.map(STATE_TOPN_MAP).fillna(12).astype(int)
        keep = rank <= topn
    elif mode == "state_topn_map_floor6":
        floor_map = {name: max(6, topn) for name, topn in STATE_TOPN_MAP.items()}
        topn = state.map(floor_map).fillna(12).astype(int)
        keep = rank <= topn
    elif mode == "weak_state_top4_else12":
        keep = (~state.isin(WEAK_STATES) & (rank <= 12)) | (state.isin(WEAK_STATES) & (rank <= 4))
    else:
        raise ValueError(f"Unknown Round G mode: {mode}")
    return frame[keep].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    if not mode.endswith("quality_reorder"):
        return pd.to_numeric(selected["base_score"], errors="coerce")

    rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    base = 1000.0 - 2.0 * rank
    state = state3(selected)
    bad = ~good_state(selected) | state.isin(WEAK_STATES)

    quality = pd.Series(0.0, index=selected.index)
    quality += 0.45 * day_z(selected, "amplitude_inv")
    quality -= 0.30 * day_z(selected, "amount")
    quality -= 0.25 * day_z(selected, "turnover43")
    quality -= 0.20 * day_z(selected, "base_score")
    quality += 0.12 * day_z(selected, "position_strength")
    quality += 0.10 * day_z(selected, "repair_rank")
    return base + quality.where(bad, 0.0)


def good_state(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")


def state3(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def day_z(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame["signal_time"]).transform("mean")
    std = values.groupby(frame["signal_time"]).transform(lambda x: x.std(ddof=0))
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_g_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_g_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_g_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_g_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_g_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_STATE_TOPN_ROUND_G_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_STATE_TOPN_ROUND_G.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_g_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    f3 = read_json(ROUND_F_BEST / "summary.json")
    if f3:
        comparison["vs_round_f_f3_total_return_delta"] = comparison["total_return"] - float(f3.get("total_return", np.nan))
        comparison["vs_round_f_f3_mdd_delta"] = comparison["max_drawdown"] - float(f3.get("max_drawdown", np.nan))
        comparison["vs_round_f_f3_sharpe_delta"] = comparison["sharpe"] - float(f3.get("sharpe", np.nan))
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_g_comparison.csv"
    signal_path = output_root / "runs/round_g_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path).sort_values(["total_return", "sharpe"], ascending=[False, False])
    raw = read_json(RAW12_BASE / "summary.json")
    f3 = read_json(ROUND_F_BEST / "summary.json")
    lines = [
        "# Raw12 State TopN Round G",
        "",
        "范围：仅在原始弱转强 score-ranked raw top12 内做状态 TopN / 轻排序；保持 `selector.lag=1`、第二天 `close` 买入、`cash_equal` 和原卖出规则。",
        "",
        "## Baselines",
        "",
        "| baseline | total_return | max_drawdown | sharpe |",
        "| --- | ---: | ---: | ---: |",
        baseline_row("raw_top12_pos12", raw),
        baseline_row("round_f_f3_good_top12_else_top8", f3),
        "",
        "## Results",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | vs_raw12_ret | vs_f3_ret | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row.get('variant')} | {flt(row.get('total_return')):.4f} | {flt(row.get('max_drawdown')):.4f} | "
            f"{flt(row.get('sharpe')):.4f} | {flt(row.get('full_avg_position_count')):.4f} | "
            f"{flt(row.get('vs_raw12_total_return_delta')):.4f} | {flt(row.get('vs_round_f_f3_total_return_delta')):.4f} | "
            f"{flt(row.get('vs_round_f_f3_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_g_comparison.csv`", "- `runs/round_g_signal_stats.csv`", "- `runs/round_g_yearly_nav.csv`", "- `runs/round_g_metrics_full.csv`"])
    if signal_path.exists():
        lines.extend(["", "## Signal Stats", ""])
        signal = pd.read_csv(signal_path)
        lines.append(signal[["variant", "signal_rows", "signal_days", "avg_daily_candidates", "max_daily_candidates"]].to_markdown(index=False))
    report_path = output_root / "reports/RAW12_STATE_TOPN_ROUND_G.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_row(name: str, row: dict[str, Any]) -> str:
    return f"| {name} | {flt(row.get('total_return')):.4f} | {flt(row.get('max_drawdown')):.4f} | {flt(row.get('sharpe')):.4f} |"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def flt(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__))
    main()
