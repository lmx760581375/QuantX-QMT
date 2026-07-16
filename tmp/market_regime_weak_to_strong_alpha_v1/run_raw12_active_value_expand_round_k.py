"""Round K active-value expansion for weak-to-strong raw top12.

This round keeps the original weak-to-strong order, cash-equal sizing, next-day
close execution, and original sell rules. It only uses point-in-time active-value
market states to restore Top12 on days where Round-F f3 would otherwise use Top8.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_active_value_expand_round_k"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ACTIVE_VALUE = Path("data/derived/active_value/daily.parquet")
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


ROLLING_PCT_COLUMNS = (
    "active_core_amount_ret1",
    "active_tradable_amount_ret1",
    "right_side_amount_ratio",
    "right_side_core_amount_ratio",
    "right_side_ratio",
)


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "k0_f3_reference",
        "description": "Round F best: good states top12, otherwise top8.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k1_core_above_ma10_restore12",
        "description": "F3 plus restore top12 when active core amount is above MA10.",
        "mode": "restore_if",
        "condition": "active_core_above_ma10",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k2_tradable_above_ma10_restore12",
        "description": "F3 plus restore top12 when active tradable amount is above MA10.",
        "mode": "restore_if",
        "condition": "active_tradable_above_ma10",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k3_any_amount_above_ma10_restore12",
        "description": "F3 plus restore top12 when either active amount MA10 flag is positive.",
        "mode": "restore_if",
        "condition": "any_amount_above_ma10",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k4_core_ret1_q2_restore12",
        "description": "F3 plus restore top12 when active core amount ret1 is in its prior-252d q2 zone.",
        "mode": "restore_if",
        "condition": "active_core_ret1_q2",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k5_tradable_ret1_q2_restore12",
        "description": "F3 plus restore top12 when active tradable amount ret1 is in its prior-252d q2 zone.",
        "mode": "restore_if",
        "condition": "active_tradable_ret1_q2",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k6_right_side_amount_q3_restore12",
        "description": "F3 plus restore top12 when right-side amount ratio is in its prior-252d q3 zone.",
        "mode": "restore_if",
        "condition": "right_side_amount_q3",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k7_right_side_amount_ge_mid_restore12",
        "description": "F3 plus restore top12 when right-side amount ratio is above its prior-252d median.",
        "mode": "restore_if",
        "condition": "right_side_amount_ge_mid",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k8_amount_ret1_or_right_q3_restore12",
        "description": "F3 plus restore top12 on active tradable ret1 q2 or right-side amount q3.",
        "mode": "restore_if",
        "condition": "tradable_ret1_q2_or_right_amount_q3",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "k9_active_broad_restore12",
        "description": "F3 plus restore top12 on amount above MA10 or right-side amount above median.",
        "mode": "restore_if",
        "condition": "active_broad",
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
    for column in ROLLING_PCT_COLUMNS:
        if column in active.columns:
            active[f"{column}_pct252"] = prior_rolling_percentile(pd.to_numeric(active[column], errors="coerce"))
    keep = [
        "signal_time",
        "active_core_amount_above_ma10",
        "active_tradable_amount_above_ma10",
        "active_core_amount_strong_up_day",
        "breadth_bull",
        *[f"{column}_pct252" for column in ROLLING_PCT_COLUMNS if f"{column}_pct252" in active.columns],
    ]
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    return frame.merge(active[keep], on="signal_time", how="left")


def prior_rolling_percentile(values: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    out = np.full(len(values), np.nan, dtype=float)
    for idx, value in enumerate(values.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue
        start = max(0, idx - window)
        history = values.iloc[start:idx].dropna().to_numpy(dtype=float)
        if len(history) < min_periods:
            continue
        out[idx] = float((history <= value).mean())
    return pd.Series(out, index=values.index)


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    topn = f3_topn(frame)
    mode = str(spec["mode"])
    if mode == "f3_reference":
        pass
    elif mode == "restore_if":
        restore = condition_mask(frame, str(spec["condition"]))
        topn = topn.where(~restore, 12)
    else:
        raise ValueError(f"Unknown Round K mode: {mode}")
    return frame[rank <= topn].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    return pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")


def condition_mask(frame: pd.DataFrame, condition: str) -> pd.Series:
    core_above = bool_col(frame, "active_core_amount_above_ma10")
    tradable_above = bool_col(frame, "active_tradable_amount_above_ma10")
    core_ret1_pct = num(frame, "active_core_amount_ret1_pct252")
    tradable_ret1_pct = num(frame, "active_tradable_amount_ret1_pct252")
    right_amount_pct = num(frame, "right_side_amount_ratio_pct252")

    if condition == "active_core_above_ma10":
        return core_above
    if condition == "active_tradable_above_ma10":
        return tradable_above
    if condition == "any_amount_above_ma10":
        return core_above | tradable_above
    if condition == "active_core_ret1_q2":
        return core_ret1_pct.ge(0.25) & core_ret1_pct.lt(0.50)
    if condition == "active_tradable_ret1_q2":
        return tradable_ret1_pct.ge(0.25) & tradable_ret1_pct.lt(0.50)
    if condition == "right_side_amount_q3":
        return right_amount_pct.ge(0.50) & right_amount_pct.lt(0.75)
    if condition == "right_side_amount_ge_mid":
        return right_amount_pct.ge(0.50)
    if condition == "tradable_ret1_q2_or_right_amount_q3":
        return (tradable_ret1_pct.ge(0.25) & tradable_ret1_pct.lt(0.50)) | (right_amount_pct.ge(0.50) & right_amount_pct.lt(0.75))
    if condition == "active_broad":
        return core_above | tradable_above | right_amount_pct.ge(0.50)
    raise ValueError(f"Unknown restore condition: {condition}")


def bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_k_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_k_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_k_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_k_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_k_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_ACTIVE_VALUE_EXPAND_ROUND_K_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_ACTIVE_VALUE_EXPAND_ROUND_K.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_k_comparison.csv"
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
    path = output_root / "runs/round_k_comparison.csv"
    signal_path = output_root / "runs/round_k_signal_stats.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    avg_candidates = signals.set_index("variant")["avg_daily_candidates"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_ACTIVE_VALUE_EXPAND_ROUND_K.md"
    lines = [
        "# Raw12 Active-Value Expand Round K",
        "",
        "范围：保留原弱转强排序、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和原卖出规则；只用活跃市值状态在 f3 原本 Top8 的日子恢复 Top12。",
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
