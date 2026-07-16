"""Round R slot/exposure tests for weak-to-strong f3.

This round keeps f3 signals, ordering, lag=1 close execution, and original sell
rules. It only tests existing QuantX allocation controls: cash_use_ratio,
slot_equal sizing, and static rank_weights.
"""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_slot_exposure_round_r"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


FRONT_110 = [1.10, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]
FRONT_115 = [1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82]
BACK_110 = list(reversed(FRONT_110))


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "r0_f3_cash_equal_098", "description": "Round F f3 reference.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "cash_equal", "cash_use_ratio": 0.98},
    {"variant": "r1_f3_cash_equal_100", "description": "f3 with cash_use_ratio=1.00.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "cash_equal", "cash_use_ratio": 1.00},
    {"variant": "r2_f3_slot_equal_098", "description": "f3 with slot_equal single-name cap.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "slot_equal", "cash_use_ratio": 0.98},
    {"variant": "r3_f3_slot_equal_100", "description": "f3 slot_equal with cash_use_ratio=1.00.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "slot_equal", "cash_use_ratio": 1.00},
    {"variant": "r4_f3_slot_front110", "description": "f3 slot_equal with mild front rank weights.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "slot_equal", "cash_use_ratio": 1.00, "rank_weights": FRONT_110},
    {"variant": "r5_f3_slot_front115", "description": "f3 slot_equal with stronger front rank weights.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "slot_equal", "cash_use_ratio": 1.00, "rank_weights": FRONT_115},
    {"variant": "r6_f3_slot_back110", "description": "f3 slot_equal with mild back rank weights.", "mode": "f3", "topk": 12, "max_positions": 12, "sizing": "slot_equal", "cash_use_ratio": 1.00, "rank_weights": BACK_110},
)

VARIANT_BY_NAME = {str(item["variant"]): item for item in VARIANTS}


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
    original_make_config = round_f.make_config
    round_f.OUTPUT_NAME = OUTPUT_NAME
    round_f.VARIANTS = VARIANTS
    round_f.select_variant = select_variant
    round_f.score_variant = score_variant
    round_f.make_config = lambda source, signal: make_config(original_make_config, source, signal)


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    return frame[rank <= f3_topn(frame)].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def f3_topn(frame: pd.DataFrame) -> pd.Series:
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    return pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")


def make_config(original_make_config, source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    config = original_make_config(source, signal)
    spec = VARIANT_BY_NAME[str(signal["variant"])]
    config = deepcopy(config)
    config["description"] = "Tmp experiment: f3 signal/order with slot and cash exposure controls only."
    config["rebalance"] = dict(config.get("rebalance") or {})
    rank_weights = spec.get("rank_weights") or []
    if rank_weights:
        config["rebalance"]["rank_weights"] = [float(value) for value in rank_weights]
    else:
        config["rebalance"].pop("rank_weights", None)
    config["execution"] = dict(config.get("execution") or {})
    config["execution"]["cash_use_ratio"] = float(spec.get("cash_use_ratio", 0.98))
    config["execution"]["buy"] = dict(config["execution"].get("buy") or {})
    config["execution"]["buy"]["sizing"] = str(spec.get("sizing", "cash_equal"))
    metadata = dict(config.get("metadata") or {})
    metadata.update(
        {
            "experiment": OUTPUT_NAME,
            "sizing": str(spec.get("sizing", "cash_equal")),
            "cash_use_ratio": float(spec.get("cash_use_ratio", 0.98)),
            "rank_weights": [float(value) for value in rank_weights],
        }
    )
    config["metadata"] = metadata
    return config


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_r_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_r_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_r_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_r_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_r_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_SLOT_EXPOSURE_ROUND_R_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_SLOT_EXPOSURE_ROUND_R.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_r_comparison.csv"
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
    comparison_path = output_root / "runs/round_r_comparison.csv"
    signal_path = output_root / "runs/round_r_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_SLOT_EXPOSURE_ROUND_R.md"
    lines = [
        "# Raw12 Slot Exposure Round R",
        "",
        "范围：保持 f3 信号、原排序、原卖出规则、`selector.lag=1`、第二天 `close` 买入；只测试现有 `cash_use_ratio`、`slot_equal` 和静态 `rank_weights`。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | signal_rows | vs_f3_ret | vs_f3_mdd | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(row.get('vs_f3_total_return_delta')):.4f} | {safe_float(row.get('vs_f3_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_sharpe_delta')):.4f} |"
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
