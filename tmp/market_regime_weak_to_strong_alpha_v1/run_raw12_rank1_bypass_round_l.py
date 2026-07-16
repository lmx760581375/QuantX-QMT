"""Round L rank-1 bypass with fill inside weak-to-strong raw top12.

This round keeps original score order, cash-equal sizing, next-day close
execution, and original sell rules. It tests whether the main account-level
drag states can be improved by skipping rank1 and filling from the next raw
weak-to-strong candidate, without expanding beyond raw top12.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_rank1_bypass_round_l"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ACTIVE_VALUE = Path("data/derived/active_value/daily.parquet")
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "l0_f3_reference",
        "description": "Round F best: good states top12, otherwise top8.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l1_bear_lo_lo_rank1_fill9",
        "description": "F3 plus skip rank1 on bear|lo|lo and fill with rank9.",
        "mode": "skip_rank1_fill",
        "skip_states": {"bear|lo|lo"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l2_range_lo_mid_rank1_fill9",
        "description": "F3 plus skip rank1 on range|lo|mid and fill with rank9.",
        "mode": "skip_rank1_fill",
        "skip_states": {"range|lo|mid"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l3_bad2_rank1_fill9",
        "description": "F3 plus skip rank1 on bear|lo|lo or range|lo|mid and fill with rank9.",
        "mode": "skip_rank1_fill",
        "skip_states": {"bear|lo|lo", "range|lo|mid"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l4_bear_lo_lo_rank1_no_fill",
        "description": "Diagnostic: F3 plus skip rank1 on bear|lo|lo without filling.",
        "mode": "skip_rank1_no_fill",
        "skip_states": {"bear|lo|lo"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l5_bad2_rank1_no_fill",
        "description": "Diagnostic: F3 plus skip rank1 on bear|lo|lo or range|lo|mid without filling.",
        "mode": "skip_rank1_no_fill",
        "skip_states": {"bear|lo|lo", "range|lo|mid"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l6_bear_lo_lo_rank1_weak_active_fill9",
        "description": "F3 plus skip rank1 on bear|lo|lo only when active amount is below MA10; fill with rank9.",
        "mode": "skip_rank1_fill_active_weak",
        "skip_states": {"bear|lo|lo"},
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "l7_range_lo_mid_rank1_weak_active_fill9",
        "description": "F3 plus skip rank1 on range|lo|mid only when active amount is below MA10; fill with rank9.",
        "mode": "skip_rank1_fill_active_weak",
        "skip_states": {"range|lo|mid"},
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
    original_build_signals = round_f.build_signals
    round_f.OUTPUT_NAME = OUTPUT_NAME
    round_f.VARIANTS = VARIANTS
    round_f.select_variant = select_variant
    round_f.score_variant = score_variant

    def build_signals_with_active(candidates: pd.DataFrame, signals_dir: Path, runs_dir: Path):
        return original_build_signals(attach_active_value(candidates), signals_dir, runs_dir)

    round_f.build_signals = build_signals_with_active


def attach_active_value(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    active = pd.read_parquet(ACTIVE_VALUE).copy()
    active["signal_time"] = pd.to_datetime(active["date"]).dt.strftime("%Y-%m-%d")
    active = active.sort_values("signal_time").drop_duplicates("signal_time", keep="last")
    keep = ["signal_time", "active_core_amount_above_ma10", "active_tradable_amount_above_ma10"]
    keep = [column for column in keep if column in active.columns]
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    return frame.merge(active[keep], on="signal_time", how="left")


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    topn = f3_topn(frame)
    mode = str(spec["mode"])

    if mode == "f3_reference":
        keep = rank <= topn
    elif mode in {"skip_rank1_fill", "skip_rank1_no_fill", "skip_rank1_fill_active_weak"}:
        skip = state3(frame).isin(set(spec.get("skip_states", set()))) & rank.eq(1)
        if mode == "skip_rank1_fill_active_weak":
            skip &= active_amount_weak(frame)
        if mode != "skip_rank1_no_fill":
            skip &= has_fill_candidate(frame, topn)
        limit = topn.where(~skip_state_rows(frame, spec, mode), topn + 1 if mode != "skip_rank1_no_fill" else topn)
        keep = (rank <= limit) & ~skip
    else:
        raise ValueError(f"Unknown Round L mode: {mode}")
    return frame[keep].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    return pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")


def skip_state_rows(frame: pd.DataFrame, spec: dict[str, Any], mode: str) -> pd.Series:
    rows = state3(frame).isin(set(spec.get("skip_states", set())))
    if mode == "skip_rank1_fill_active_weak":
        rows &= active_amount_weak(frame)
    if mode != "skip_rank1_no_fill":
        rows &= has_fill_candidate(frame, f3_topn(frame))
    return rows


def has_fill_candidate(frame: pd.DataFrame, topn: pd.Series) -> pd.Series:
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    max_rank = rank.groupby(frame["signal_time"]).transform("max")
    return max_rank.ge(topn + 1)


def active_amount_weak(frame: pd.DataFrame) -> pd.Series:
    core = bool_col(frame, "active_core_amount_above_ma10")
    tradable = bool_col(frame, "active_tradable_amount_above_ma10")
    if not core.any() and not tradable.any():
        return pd.Series(False, index=frame.index)
    return ~(core | tradable)


def state3(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].astype(str) + "|" + frame["amount_state"].astype(str) + "|" + frame["breadth_state"].astype(str)


def bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_l_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_l_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_l_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_l_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_l_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_RANK1_BYPASS_ROUND_L_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_RANK1_BYPASS_ROUND_L.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_l_comparison.csv"
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
    path = output_root / "runs/round_l_comparison.csv"
    signal_path = output_root / "runs/round_l_signal_stats.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    avg_candidates = signals.set_index("variant")["avg_daily_candidates"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_RANK1_BYPASS_ROUND_L.md"
    lines = [
        "# Raw12 Rank1 Bypass Round L",
        "",
        "范围：保留原弱转强排序、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和原卖出规则；只在少数坏状态中跳过 rank1，并用原 raw Top12 内的下一名补位。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | avg_candidates | signal_rows | vs_raw12_ret | vs_f3_ret | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(avg_candidates.get(variant)):.4f} | "
            f"{int(signal_rows.get(variant, 0))} | {safe_float(row.get('vs_raw12_total_return_delta')):.4f} | "
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
