"""Round H re-rank and rank-weight tests for weak-to-strong raw top12.

This experiment keeps the original weak-to-strong raw top12 candidate pool and
does not expand or shrink the pool. It tests whether account results improve by
changing buy priority and optionally tilting cash toward the higher ranked rows
inside QuantX's existing `rebalance.rank_weights` mechanism.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_rerank_weight_round_h"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


FRONT_125 = [1.55, 1.42, 1.30, 1.20, 1.10, 1.03, 0.97, 0.92, 0.86, 0.80, 0.74, 0.68]
FRONT_150 = [1.85, 1.65, 1.48, 1.32, 1.18, 1.06, 0.95, 0.86, 0.78, 0.70, 0.62, 0.55]


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "h0_raw12_equal_reference",
        "description": "Raw top12 with original score order and equal cash sizing.",
        "mode": "raw12_original",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "h1_reverse_rank_equal",
        "description": "Raw top12, lower original score ranks bought first; equal cash.",
        "mode": "reverse_rank",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "h2_low_amplitude_equal",
        "description": "Raw top12, lower same-day amplitude bought first; equal cash.",
        "mode": "low_amplitude",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "h3_reverse_rank_low_amp_equal",
        "description": "Raw top12, reverse original rank plus low-amplitude preference; equal cash.",
        "mode": "reverse_rank_low_amp",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "h4_reverse_rank_weight125",
        "description": "Reverse original rank order with mild front-loaded rank weights.",
        "mode": "reverse_rank",
        "topk": 12,
        "max_positions": 12,
        "rank_weights": FRONT_125,
    },
    {
        "variant": "h5_low_amplitude_weight125",
        "description": "Low-amplitude order with mild front-loaded rank weights.",
        "mode": "low_amplitude",
        "topk": 12,
        "max_positions": 12,
        "rank_weights": FRONT_125,
    },
    {
        "variant": "h6_reverse_rank_low_amp_weight125",
        "description": "Reverse-rank plus low-amplitude order with mild front-loaded rank weights.",
        "mode": "reverse_rank_low_amp",
        "topk": 12,
        "max_positions": 12,
        "rank_weights": FRONT_125,
    },
    {
        "variant": "h7_reverse_rank_low_amp_weight150",
        "description": "Reverse-rank plus low-amplitude order with stronger front-loaded rank weights.",
        "mode": "reverse_rank_low_amp",
        "topk": 12,
        "max_positions": 12,
        "rank_weights": FRONT_150,
    },
    {
        "variant": "h8_state_adaptive_rank_weight125",
        "description": "State-adaptive rank direction plus low-amplitude preference and mild rank weights.",
        "mode": "state_adaptive_rank",
        "topk": 12,
        "max_positions": 12,
        "rank_weights": FRONT_125,
    },
    {
        "variant": "h9_state_adaptive_rank_equal",
        "description": "State-adaptive rank direction plus low-amplitude preference; equal cash.",
        "mode": "state_adaptive_rank",
        "topk": 12,
        "max_positions": 12,
    },
)


LOW_RANK_GOOD_STATES = {
    "bull|hi|mid",
    "bull|mid|mid",
    "bear|mid|mid",
    "bull|mid|hi",
    "range|mid|mid",
}

HIGH_RANK_GOOD_STATES = {
    "range|hi|lo",
    "bear|lo|lo",
    "bear|unknown|unknown",
}


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
    original_make_config = round_f.make_config

    def make_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
        config = original_make_config(source, signal)
        spec = next((item for item in VARIANTS if item["variant"] == signal["variant"]), {})
        rank_weights = spec.get("rank_weights") or []
        config["title"] = "弱转强 raw top12 重排与资金倾斜 Round H"
        config["description"] = "Tmp experiment: keep original raw top12, change order and optional rank weights only."
        config["rebalance"] = dict(config.get("rebalance") or {})
        if rank_weights:
            config["rebalance"]["rank_weights"] = [float(value) for value in rank_weights]
        else:
            config["rebalance"].pop("rank_weights", None)
        metadata = dict(config.get("metadata") or {})
        metadata["rank_weights"] = rank_weights
        metadata["pool_policy"] = "keep_original_raw_top12"
        config["metadata"] = metadata
        return config

    round_f.make_config = make_config


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rank = pd.to_numeric(candidates["base_rank"], errors="coerce")
    return candidates[rank <= int(spec["topk"])].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    base_score = pd.to_numeric(selected["base_score"], errors="coerce")
    rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)

    if mode == "raw12_original":
        return base_score

    amp = day_z(selected, "amplitude_inv")
    if mode == "reverse_rank":
        return 1000.0 + rank
    if mode == "low_amplitude":
        return 1000.0 + 10.0 * amp - 0.01 * rank
    if mode == "reverse_rank_low_amp":
        return 1000.0 + rank + 0.35 * amp
    if mode == "state_adaptive_rank":
        state = state3(selected)
        rank_direction = pd.Series(0.0, index=selected.index)
        rank_direction = rank_direction.where(~state.isin(LOW_RANK_GOOD_STATES), rank)
        rank_direction = rank_direction.where(~state.isin(HIGH_RANK_GOOD_STATES), -rank)
        rank_direction = rank_direction.where(rank_direction != 0.0, 0.25 * rank)
        return 1000.0 + rank_direction + 0.30 * amp
    raise ValueError(f"Unknown Round H mode: {mode}")


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
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_h_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_h_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_h_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_h_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_h_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_RERANK_WEIGHT_ROUND_H_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_RERANK_WEIGHT_ROUND_H.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_h_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    raw12 = read_json(OUTPUT_ROOT_SAFE(RAW12_BASE) / "summary.json")
    f3 = read_json(OUTPUT_ROOT_SAFE(ROUND_F_BEST) / "summary.json")
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
    path = output_root / "runs/round_h_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    report = output_root / "reports/RAW12_RERANK_WEIGHT_ROUND_H.md"
    lines = [
        "# Raw12 Rerank Weight Round H",
        "",
        "范围：保留原弱转强 raw top12，不扩池、不缩池；只测试池内排序和 `rebalance.rank_weights` 资金倾斜。执行语义仍为 `lag=1`、第二天 `close` 买入、原卖出规则。",
        "",
        "## Results",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | vs_raw12_ret | vs_raw12_mdd | vs_raw12_sharpe | vs_f3_ret | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row.get('variant')} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(row.get('vs_raw12_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_raw12_mdd_delta')):.4f} | {safe_float(row.get('vs_raw12_sharpe_delta')):.4f} | "
            f"{safe_float(row.get('vs_round_f_f3_total_return_delta')):.4f} | {safe_float(row.get('vs_round_f_f3_sharpe_delta')):.4f} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `runs/round_h_comparison.csv`",
        "- `runs/round_h_signal_stats.csv`",
        "- `runs/round_h_yearly_nav.csv`",
        "- `runs/round_h_entry_weight_buckets.csv`",
        "- `runs/round_h_metrics_full.csv`",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def OUTPUT_ROOT_SAFE(path: Path) -> Path:
    return path


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
