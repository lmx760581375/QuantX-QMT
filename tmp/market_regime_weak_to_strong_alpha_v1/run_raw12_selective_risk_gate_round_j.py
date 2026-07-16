"""Round J selective risk gates for weak-to-strong raw top12.

This round keeps the original weak-to-strong score order and cash-equal sizing.
It only tests sparse, point-in-time TopN reductions for market states that looked
bad in account-level attribution after Round F/G/I.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_selective_risk_gate_round_j"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


BAD_STATES_POST2025 = {
    "bear|lo|lo",
    "range|lo|mid",
    "range|lo|hi",
    "range|mid|mid",
    "range|mid|lo",
    "bear|mid|mid",
}


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "j0_f3_reference",
        "description": "Round F best: good states top12, otherwise top8.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j1_bear_lo_lo_top4",
        "description": "F3 plus bear|lo|lo days cut to top4.",
        "mode": "state_topn",
        "state_topn": {"bear|lo|lo": 4},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j2_bear_lo_lo_bad_slope_top4",
        "description": "F3 plus bear|lo|lo top4 only when amount_slope<0 or breadth_gap<0.",
        "mode": "bear_lo_lo_bad_slope_top4",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j3_bear_lo_lo_after2024_top4",
        "description": "Diagnostic: F3 plus bear|lo|lo top4 from 2025 onward.",
        "mode": "bear_lo_lo_after2024_top4",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j4_post2025_bad_states_top6",
        "description": "Diagnostic: F3 plus post-2025 bad state cluster cut to top6.",
        "mode": "post2025_bad_states_top6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j5_liquidity_deteriorating_top6",
        "description": "F3 plus amount_lo with amount_slope<0 and non-hi breadth cut to top6.",
        "mode": "liquidity_deteriorating_top6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j6_bear_low_market_top6",
        "description": "F3 plus bear states with amount_lo or breadth_lo cut to top6.",
        "mode": "bear_low_market_top6",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "j7_range_low_deteriorating_top6",
        "description": "F3 plus range amount_lo states cut to top6 when amount_slope<0 or breadth_gap<0.",
        "mode": "range_low_deteriorating_top6",
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
    topn = f3_topn(frame)
    state = state3(frame)
    mode = str(spec["mode"])

    if mode == "f3_reference":
        pass
    elif mode == "state_topn":
        mapped = state.map(spec.get("state_topn", {}))
        topn = topn.where(mapped.isna(), np.minimum(topn, mapped.fillna(topn).astype(int)))
    elif mode == "bear_lo_lo_bad_slope_top4":
        bad = state.eq("bear|lo|lo") & deteriorating_market(frame)
        topn = topn.where(~bad, np.minimum(topn, 4))
    elif mode == "bear_lo_lo_after2024_top4":
        bad = state.eq("bear|lo|lo") & (year(frame) >= 2025)
        topn = topn.where(~bad, np.minimum(topn, 4))
    elif mode == "post2025_bad_states_top6":
        bad = state.isin(BAD_STATES_POST2025) & (year(frame) >= 2025)
        topn = topn.where(~bad, np.minimum(topn, 6))
    elif mode == "liquidity_deteriorating_top6":
        bad = frame["amount_state"].eq("lo") & ~frame["breadth_state"].eq("hi") & (num(frame, "amount_slope") < 0)
        topn = topn.where(~bad, np.minimum(topn, 6))
    elif mode == "bear_low_market_top6":
        bad = frame["regime"].eq("bear") & (frame["amount_state"].eq("lo") | frame["breadth_state"].eq("lo"))
        topn = topn.where(~bad, np.minimum(topn, 6))
    elif mode == "range_low_deteriorating_top6":
        bad = frame["regime"].eq("range") & frame["amount_state"].eq("lo") & deteriorating_market(frame)
        topn = topn.where(~bad, np.minimum(topn, 6))
    else:
        raise ValueError(f"Unknown Round J mode: {mode}")

    return frame[rank <= topn].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    return pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")


def deteriorating_market(frame: pd.DataFrame) -> pd.Series:
    return (num(frame, "amount_slope") < 0) | (num(frame, "breadth_gap") < 0)


def state3(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def year(frame: pd.DataFrame) -> pd.Series:
    if "year" in frame.columns:
        return pd.to_numeric(frame["year"], errors="coerce").fillna(0).astype(int)
    return pd.to_datetime(frame["signal_time"], errors="coerce").dt.year.fillna(0).astype(int)


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_j_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_j_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_j_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_j_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_j_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_SELECTIVE_RISK_GATE_ROUND_J_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_SELECTIVE_RISK_GATE_ROUND_J.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_j_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    raw12 = read_json(RAW12_BASE / "summary.json")
    f3 = read_json(ROUND_F_BEST / "summary.json")
    if raw12:
        comparison["vs_raw12_total_return_delta"] = comparison["total_return"] - float(raw12.get("total_return", np.nan))
        comparison["vs_raw12_mdd_delta"] = comparison["max_drawdown"] - float(raw12.get("max_drawdown", np.nan))
        comparison["vs_raw12_sharpe_delta"] = comparison["sharpe"] - float(raw12.get("sharpe", np.nan))
    if f3:
        comparison["vs_round_f_f3_total_return_delta"] = comparison["total_return"] - float(f3.get("total_return", np.nan))
        comparison["vs_round_f_f3_mdd_delta"] = comparison["max_drawdown"] - float(f3.get("max_drawdown", np.nan))
        comparison["vs_round_f_f3_sharpe_delta"] = comparison["sharpe"] - float(f3.get("sharpe", np.nan))
    comparison = comparison.sort_values(["total_return", "sharpe"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    path = output_root / "runs/round_j_comparison.csv"
    signal_path = output_root / "runs/round_j_signal_stats.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    report = output_root / "reports/RAW12_SELECTIVE_RISK_GATE_ROUND_J.md"
    lines = [
        "# Raw12 Selective Risk Gate Round J",
        "",
        "范围：保留原弱转强排序、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和原卖出规则；只测试少数市场状态下的 TopN 风险门控。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | signal_rows | vs_raw12_ret | vs_f3_ret | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(row.get('vs_raw12_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_round_f_f3_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_round_f_f3_sharpe_delta')):.4f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
