"""Round I original-order rank-weight tests for weak-to-strong raw top12.

Round H showed that reordering the raw top12 pool hurts account-level returns.
This round keeps the original weak-to-strong score order intact and tests only
QuantX's existing `rebalance.rank_weights` on raw12 and the Round-F f3 TopN rule.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_original_weight_round_i"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


FRONT_115 = [1.32, 1.25, 1.18, 1.12, 1.06, 1.00, 0.95, 0.90, 0.85, 0.80, 0.76, 0.72]
FRONT_125 = [1.55, 1.42, 1.30, 1.20, 1.10, 1.03, 0.97, 0.92, 0.86, 0.80, 0.74, 0.68]
FRONT_150 = [1.85, 1.65, 1.48, 1.32, 1.18, 1.06, 0.95, 0.86, 0.78, 0.70, 0.62, 0.55]
BACK_125 = list(reversed(FRONT_125))


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "i0_raw12_equal_reference", "mode": "raw12", "topk": 12, "max_positions": 12},
    {"variant": "i1_raw12_front115", "mode": "raw12", "topk": 12, "max_positions": 12, "rank_weights": FRONT_115},
    {"variant": "i2_raw12_front125", "mode": "raw12", "topk": 12, "max_positions": 12, "rank_weights": FRONT_125},
    {"variant": "i3_raw12_front150", "mode": "raw12", "topk": 12, "max_positions": 12, "rank_weights": FRONT_150},
    {"variant": "i4_raw12_back125", "mode": "raw12", "topk": 12, "max_positions": 12, "rank_weights": BACK_125},
    {"variant": "i5_f3_equal_reference", "mode": "f3", "topk": 12, "max_positions": 12},
    {"variant": "i6_f3_front115", "mode": "f3", "topk": 12, "max_positions": 12, "rank_weights": FRONT_115},
    {"variant": "i7_f3_front125", "mode": "f3", "topk": 12, "max_positions": 12, "rank_weights": FRONT_125},
    {"variant": "i8_f3_front150", "mode": "f3", "topk": 12, "max_positions": 12, "rank_weights": FRONT_150},
    {"variant": "i9_f3_back125", "mode": "f3", "topk": 12, "max_positions": 12, "rank_weights": BACK_125},
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
    original_make_config = round_f.make_config

    def make_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
        config = original_make_config(source, signal)
        spec = next((item for item in VARIANTS if item["variant"] == signal["variant"]), {})
        rank_weights = spec.get("rank_weights") or []
        config["title"] = "弱转强原排序资金倾斜 Round I"
        config["description"] = "Tmp experiment: keep original score order; test rank_weights only."
        config["rebalance"] = dict(config.get("rebalance") or {})
        if rank_weights:
            config["rebalance"]["rank_weights"] = [float(value) for value in rank_weights]
        else:
            config["rebalance"].pop("rank_weights", None)
        metadata = dict(config.get("metadata") or {})
        metadata["rank_weights"] = rank_weights
        metadata["score_order_policy"] = "original_weak_to_strong_score"
        config["metadata"] = metadata
        return config

    round_f.make_config = make_config


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    mode = str(spec["mode"])
    if mode == "raw12":
        return frame
    if mode == "f3":
        good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
        return frame[(good & (rank <= 12)) | (~good & (rank <= 8))].copy()
    raise ValueError(f"Unknown Round I mode: {mode}")


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_i_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_i_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_i_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_i_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_i_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_ORIGINAL_WEIGHT_ROUND_I_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_ORIGINAL_WEIGHT_ROUND_I.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_i_comparison.csv"
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
    path = output_root / "runs/round_i_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    report = output_root / "reports/RAW12_ORIGINAL_WEIGHT_ROUND_I.md"
    lines = [
        "# Raw12 Original Weight Round I",
        "",
        "范围：保留原弱转强分数排序，只测试 `rebalance.rank_weights`；执行语义仍为 `lag=1`、第二天 `close` 买入、原卖出规则。",
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
