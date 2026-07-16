"""Round Q realized-attribution gates for weak-to-strong f3.

Round P showed that forward-label IC micro-ranking did not improve the formal
account path. This round uses realized f3 attribution instead, and keeps the
strategy mechanics unchanged except for very narrow entry-state gates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_realized_gate_round_q"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "q0_f3_reference", "description": "Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "q1_skip_bear_lo_lo", "description": "Skip new entries in bear|lo|lo, the worst realized f3 state.", "mode": "skip_bear_lo_lo", "topk": 12, "max_positions": 12},
    {"variant": "q2_skip_bearlo_rangelomid", "description": "Skip new entries in bear|lo|lo and range|lo|mid.", "mode": "skip_bearlo_rangelomid", "topk": 12, "max_positions": 12},
    {"variant": "q3_cap_bear_lo_lo_top3", "description": "In bear|lo|lo keep only original top3; otherwise f3.", "mode": "cap_bear_lo_lo_top3", "topk": 12, "max_positions": 12},
    {"variant": "q4_cap_two_bad_states_top3", "description": "In bear|lo|lo and range|lo|mid keep only original top3; otherwise f3.", "mode": "cap_two_bad_states_top3", "topk": 12, "max_positions": 12},
    {"variant": "q5_skip_bad_state_rank1", "description": "Only skip rank1 in bear|lo|lo and range|lo|mid; keep original tail order.", "mode": "skip_bad_state_rank1", "topk": 12, "max_positions": 12},
    {"variant": "q6_skip_bear_lo_lo_rank1", "description": "Only skip rank1 in bear|lo|lo; keep original tail order.", "mode": "skip_bear_lo_lo_rank1", "topk": 12, "max_positions": 12},
    {"variant": "q7_skip_2025_bear_lo_lo_diag", "description": "Diagnostic only: skip bear|lo|lo entries in 2025.", "mode": "skip_2025_bear_lo_lo_diag", "topk": 12, "max_positions": 12},
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
    keep = rank <= f3_topn(frame)
    mode = str(spec["mode"])
    if mode == "f3_reference":
        return frame[keep].copy()
    if mode == "skip_bear_lo_lo":
        keep &= ~bear_lo_lo(frame)
    elif mode == "skip_bearlo_rangelomid":
        keep &= ~(bear_lo_lo(frame) | range_lo_mid(frame))
    elif mode == "cap_bear_lo_lo_top3":
        keep &= ~bear_lo_lo(frame) | rank.le(3)
    elif mode == "cap_two_bad_states_top3":
        keep &= ~(bear_lo_lo(frame) | range_lo_mid(frame)) | rank.le(3)
    elif mode == "skip_bad_state_rank1":
        keep &= ~((bear_lo_lo(frame) | range_lo_mid(frame)) & rank.eq(1))
    elif mode == "skip_bear_lo_lo_rank1":
        keep &= ~(bear_lo_lo(frame) & rank.eq(1))
    elif mode == "skip_2025_bear_lo_lo_diag":
        year = pd.to_datetime(frame["signal_time"]).dt.year
        keep &= ~(bear_lo_lo(frame) & year.eq(2025))
    else:
        raise ValueError(f"Unknown Round Q mode: {mode}")
    return frame[keep].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def f3_good(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(f3_good(frame), 12, 8), index=frame.index, dtype="int64")


def bear_lo_lo(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("bear") & frame["amount_state"].eq("lo") & frame["breadth_state"].eq("lo")


def range_lo_mid(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("range") & frame["amount_state"].eq("lo") & frame["breadth_state"].eq("mid")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_q_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_q_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_q_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_q_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_q_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_REALIZED_GATE_ROUND_Q_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_REALIZED_GATE_ROUND_Q.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_q_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "raw12": read_json(RAW12_BASE / "summary.json"),
        "f3": read_json(ROUND_F_BEST / "summary.json"),
    }
    for name, ref in refs.items():
        if not ref:
            continue
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_q_comparison.csv"
    signal_path = output_root / "runs/round_q_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    avg_candidates = signals.set_index("variant")["avg_daily_candidates"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_REALIZED_GATE_ROUND_Q.md"
    lines = [
        "# Raw12 Realized Gate Round Q",
        "",
        "范围：归因驱动的极窄入场状态 gate；保持原弱转强排序、原卖出规则、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和 `max_positions=12`。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | avg_candidates | signal_rows | vs_f3_ret | vs_f3_mdd | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(avg_candidates.get(variant)):.4f} | "
            f"{int(signal_rows.get(variant, 0))} | {safe_float(row.get('vs_f3_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_mdd_delta')):.4f} | {safe_float(row.get('vs_f3_sharpe_delta')):.4f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
