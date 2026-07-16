"""Round O add-existing tests for weak-to-strong f3.

This round keeps the f3 entry path, original weak-to-strong ordering, sell
rules, close execution, cash-equal sizing, and max_positions=12 unchanged. It
only tests whether idle cash should be reinvested into existing winners.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_add_existing_round_o"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "o0_f3_reference",
        "description": "Round F f3 reference; no add-existing rule.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "o1_add_profit_trend",
        "description": "Add to profitable holdings with positive medium/long trend.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days >= 3 and pnl_pct > 0.03 and close > zxdkx and zxdq > zxdkx and rsv_long > 45 and position_weight < 0.18",
    },
    {
        "variant": "o2_add_profit_strong",
        "description": "Add only to strong winners that have not pulled back much from peak.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days >= 5 and pnl_pct > 0.06 and peak_pnl_pct > 0.08 and drawdown_from_peak > -0.04 and close > zxdq and close > zxdkx and zxdq > zxdkx and position_weight < 0.20",
    },
    {
        "variant": "o3_add_breakout_pullback",
        "description": "Add to winners on short pullback while long trend remains intact.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days >= 5 and pnl_pct > 0.02 and peak_pnl_pct > 0.10 and drawdown_from_peak > -0.08 and rsv_short < 35 and close > zxdkx and position_weight < 0.18",
    },
    {
        "variant": "o4_add_account_ok",
        "description": "Add to trend winners only when account drawdown is controlled.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days >= 3 and pnl_pct > 0.04 and account_drawdown > -0.10 and close > zxdkx and zxdq > zxdkx and rsv_long > 50 and position_weight < 0.18",
    },
    {
        "variant": "o5_add_late_winner",
        "description": "Add to established right-tail winners later in the holding path.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days > 8 and pnl_pct > 0.08 and drawdown_from_peak > -0.06 and close > zxdq and close > zxdkx and position_weight < 0.22",
    },
    {
        "variant": "o6_add_small_winner",
        "description": "Add early to small winners with very shallow drawdown.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days > 3 and pnl_pct > 0.02 and peak_pnl_pct > 0.04 and drawdown_from_peak > -0.03 and close > zxdq and close > zxdkx and position_weight < 0.16",
    },
    {
        "variant": "o7_add_market_strong",
        "description": "Add to winners only in strong market state proxy from f3 entry days.",
        "mode": "f3_reference",
        "topk": 12,
        "max_positions": 12,
        "add_existing_when": "holding_days >= 3 and pnl_pct > 0.03 and account_drawdown > -0.08 and close > zxdq and close > zxdkx and rsv_long > 55 and rsv_short > 35 and position_weight < 0.18",
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
    original_make_config = round_f.make_config
    original_signal_manifest = round_f.signal_manifest
    round_f.OUTPUT_NAME = OUTPUT_NAME
    round_f.VARIANTS = VARIANTS
    round_f.select_variant = select_variant
    round_f.score_variant = score_variant
    round_f.signal_manifest = signal_manifest_with_add_existing(original_signal_manifest)
    round_f.make_config = make_config_with_add_existing(original_make_config)


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    topn = pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")
    return frame[rank <= topn].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def make_config_with_add_existing(original_make_config):
    def make_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
        config = original_make_config(source, signal)
        add_rule = signal.get("add_existing_when")
        buy_cfg = config.setdefault("execution", {}).setdefault("buy", {})
        if add_rule:
            buy_cfg["add_existing_when"] = str(add_rule)
        else:
            buy_cfg.pop("add_existing_when", None)
        config["title"] = "弱转强 raw top12 持仓加仓 Round O"
        config["description"] = "Tmp experiment: keep f3 entry unchanged and test add-existing reinvestment rules."
        metadata = config.setdefault("metadata", {})
        metadata["add_existing_when"] = str(add_rule) if add_rule else None
        metadata["sell_rules_unchanged_from_f3"] = True
        return config

    return make_config


def signal_manifest_with_add_existing(original_signal_manifest):
    def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
        manifest = original_signal_manifest(spec, path, signals)
        manifest["add_existing_when"] = spec.get("add_existing_when")
        return manifest

    return signal_manifest


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_o_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_o_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_o_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_o_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_o_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_ADD_EXISTING_ROUND_O_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_ADD_EXISTING_ROUND_O.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_o_comparison.csv"
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
    comparison_path = output_root / "runs/round_o_comparison.csv"
    signal_path = output_root / "runs/round_o_signal_stats.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_ADD_EXISTING_ROUND_O.md"
    lines = [
        "# Raw12 Add Existing Round O",
        "",
        "范围：保留 f3 信号、原弱转强排序、`selector.lag=1`、第二天 `close` 买入、原卖出规则、`cash_equal` 和 `max_positions=12`；只测试 `execution.buy.add_existing_when` 加仓规则。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | avg_hold | closed | signal_rows | vs_f3_ret | vs_f3_mdd | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(row.get('avg_holding_days')):.2f} | "
            f"{int(safe_float(row.get('closed_position_count')))} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(row.get('vs_f3_total_return_delta')):.4f} | {safe_float(row.get('vs_f3_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_sharpe_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_o_comparison.csv`", "- `runs/round_o_signal_stats.csv`", "- `runs/round_o_yearly_nav.csv`", "- `runs/round_o_entry_weight_buckets.csv`", "- `runs/round_o_metrics_full.csv`"])
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
