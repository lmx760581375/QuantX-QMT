"""Round P rank2-12 micro-selection tests for weak-to-strong f3.

Round O showed that adding to existing holdings damages the right-tail path.
This round keeps rank1, close execution, cash-equal sizing, original sell rules,
and max_positions=12. It only makes small rank2-12 ordering or bad-state fill
changes suggested by buyable/non-ST candidate IC analysis.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_tail_micro_round_p"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "p0_f3_reference", "description": "Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "p1_tail_low_amp_order", "description": "Keep f3 pool and rank1; order rank2-12 by lower amplitude.", "mode": "tail_low_amp_order", "topk": 12, "max_positions": 12},
    {"variant": "p2_bad_tail_reverse_order", "description": "Keep f3 pool and rank1; reverse rank2-12 order only in bad f3 states.", "mode": "bad_tail_reverse_order", "topk": 12, "max_positions": 12},
    {"variant": "p3_bad_tail_low_liq_reverse_order", "description": "Keep f3 pool and rank1; bad states prefer lower turnover plus lower original score.", "mode": "bad_tail_low_liq_reverse_order", "topk": 12, "max_positions": 12},
    {"variant": "p4_good_low_amp_bad_reverse", "description": "Good states prefer low-amplitude rank2-12; bad states reverse rank2-12.", "mode": "good_low_amp_bad_reverse", "topk": 12, "max_positions": 12},
    {"variant": "p5_bad_rank1_tail5_11", "description": "Good states top12; bad states rank1 plus rank5-11.", "mode": "bad_rank1_tail5_11", "topk": 12, "max_positions": 12},
    {"variant": "p6_bad_rank1_low_amp_tail", "description": "Good states top12; bad states rank1 plus seven lowest-amplitude rank2-12 names.", "mode": "bad_rank1_low_amp_tail", "topk": 12, "max_positions": 12},
    {"variant": "p7_bad_rank1_low_amount_tail", "description": "Good states top12; bad states rank1 plus seven lowest-amount rank2-12 names.", "mode": "bad_rank1_low_amount_tail", "topk": 12, "max_positions": 12},
    {"variant": "p8_bad_rank1_combo_tail", "description": "Good states top12; bad states rank1 plus seven names by low amplitude, low amount, and lower original score.", "mode": "bad_rank1_combo_tail", "topk": 12, "max_positions": 12},
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
    mode = str(spec["mode"])
    if mode in {"f3_reference", "tail_low_amp_order", "bad_tail_reverse_order", "bad_tail_low_liq_reverse_order", "good_low_amp_bad_reverse"}:
        return frame[pd.to_numeric(frame["base_rank"], errors="coerce") <= f3_topn(frame)].copy()
    if mode == "bad_rank1_tail5_11":
        return select_bad_rank1_plus_window(frame, start=5, end=11)
    if mode == "bad_rank1_low_amp_tail":
        return select_bad_rank1_plus_factor(frame, "amp_inv", tail_count=7)
    if mode == "bad_rank1_low_amount_tail":
        return select_bad_rank1_plus_factor(frame, "amount_inv", tail_count=7)
    if mode == "bad_rank1_combo_tail":
        scored = frame.copy()
        rank = pd.to_numeric(scored["base_rank"], errors="coerce")
        scored["combo_tail_score"] = day_z(scored, "amp_inv") + 0.75 * day_z(scored, "amount_inv") + 0.50 * day_z(scored, "base_rank")
        scored["_factor_for_select"] = scored["combo_tail_score"]
        return select_bad_rank1_plus_factor(scored, "_factor_for_select", tail_count=7)
    raise ValueError(f"Unknown Round P mode: {mode}")


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    base = pd.to_numeric(selected["base_score"], errors="coerce")
    rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    if mode == "f3_reference":
        return base

    rank1 = rank.eq(1)
    good = f3_good(selected)
    original_order = 100000.0 - rank
    amp_score = 10000.0 + day_z(selected, "amp_inv") - 0.01 * rank
    bad_reverse = 10000.0 + rank
    low_liq_reverse = 10000.0 + 0.80 * day_z(selected, "amount_inv") + 0.80 * day_z(selected, "turnover43_inv") + 0.50 * day_z(selected, "base_rank")

    tail_score = original_order.copy()
    if mode == "tail_low_amp_order":
        tail_score = amp_score
    elif mode == "bad_tail_reverse_order":
        tail_score = original_order.where(good, bad_reverse)
    elif mode == "bad_tail_low_liq_reverse_order":
        tail_score = original_order.where(good, low_liq_reverse)
    elif mode == "good_low_amp_bad_reverse":
        tail_score = amp_score.where(good, bad_reverse)
    elif mode in {"bad_rank1_tail5_11", "bad_rank1_low_amp_tail", "bad_rank1_low_amount_tail", "bad_rank1_combo_tail"}:
        tail_score = original_order
    else:
        raise ValueError(f"Unknown Round P mode: {mode}")
    return pd.Series(np.where(rank1, 200000.0, tail_score), index=selected.index)


def select_bad_rank1_plus_window(frame: pd.DataFrame, *, start: int, end: int) -> pd.DataFrame:
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    good = f3_good(frame)
    keep_good = good & rank.le(12)
    keep_bad = (~good) & (rank.eq(1) | rank.between(start, end))
    return frame[keep_good | keep_bad].copy()


def select_bad_rank1_plus_factor(frame: pd.DataFrame, factor: str, *, tail_count: int) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    good = f3_good(frame)
    out.append(frame[good & rank.le(12)].copy())
    bad = frame[~good & rank.le(12)].copy()
    if bad.empty:
        return pd.concat(out, ignore_index=True)
    for _, group in bad.groupby("signal_time", sort=False):
        group_rank = pd.to_numeric(group["base_rank"], errors="coerce")
        head = group[group_rank.eq(1)].copy()
        tail = group[group_rank.between(2, 12)].copy()
        if factor not in tail.columns:
            tail[factor] = day_z(tail, "base_rank")
        values = pd.to_numeric(tail[factor], errors="coerce").fillna(-np.inf)
        tail = tail.assign(_select_score=values, _select_rank=group_rank.loc[tail.index])
        tail = tail.sort_values(["_select_score", "_select_rank", "instrument"], ascending=[False, True, True]).head(tail_count)
        out.append(pd.concat([head, tail.drop(columns=["_select_score", "_select_rank"], errors="ignore")], ignore_index=True))
    return pd.concat(out, ignore_index=True)


def f3_good(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(f3_good(frame), 12, 8), index=frame.index, dtype="int64")


def day_z(frame: pd.DataFrame, column: str) -> pd.Series:
    values = factor_values(frame, column)
    mean = values.groupby(frame["signal_time"]).transform("mean")
    std = values.groupby(frame["signal_time"]).transform(lambda item: item.std(ddof=0))
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def factor_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column == "amp_inv":
        return -pd.to_numeric(frame.get("amplitude"), errors="coerce")
    if column == "amount_inv":
        return -pd.to_numeric(frame.get("amount"), errors="coerce")
    if column == "turnover43_inv":
        return -pd.to_numeric(frame.get("turnover43"), errors="coerce")
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_p_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_p_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_p_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_p_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_p_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_TAIL_MICRO_ROUND_P_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_TAIL_MICRO_ROUND_P.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_p_comparison.csv"
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
    comparison_path = output_root / "runs/round_p_comparison.csv"
    signal_path = output_root / "runs/round_p_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    avg_candidates = signals.set_index("variant")["avg_daily_candidates"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_TAIL_MICRO_ROUND_P.md"
    lines = [
        "# Raw12 Tail Micro Round P",
        "",
        "范围：保留 rank1、原卖出规则、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和 `max_positions=12`；只在 rank2-12 做低振幅/低成交/坏状态微调。",
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
