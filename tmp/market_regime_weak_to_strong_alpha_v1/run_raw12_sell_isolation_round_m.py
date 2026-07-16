"""Round M isolated sell-rule tests for the weak-to-strong f3 signal.

This round keeps the Round-F f3 signal, original weak-to-strong score order,
cash-equal sizing, next-day close execution, and max 12 positions. It changes
only one sell-rule family at a time so account-level effects are attributable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_sell_isolation_round_m"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "m0_f3_reference", "description": "Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "m1_stop_loss_075_only", "description": "F3 with only stop-loss tightened to -7.5%.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "stop_loss_075_only"},
    {"variant": "m2_stop_loss_105_only", "description": "F3 with only stop-loss loosened to -10.5%.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "stop_loss_105_only"},
    {"variant": "m3_time_stop_20_only", "description": "F3 with only time stop shortened to 20 trading days.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time_stop_20_only"},
    {"variant": "m4_time_stop_35_only", "description": "F3 with only time stop extended to 35 trading days.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time_stop_35_only"},
    {"variant": "m5_take_profit_180_only", "description": "F3 with only take-profit lowered to 18%." , "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "take_profit_180_only"},
    {"variant": "m6_take_profit_260_only", "description": "F3 with only take-profit raised to 26%." , "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "take_profit_260_only"},
    {"variant": "m7_trail_12_08_only", "description": "F3 with only trailing stop made earlier/tighter: peak 12%, drawdown -8%.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "trail_12_08_only"},
    {"variant": "m8_trail_18_12_only", "description": "F3 with only trailing stop loosened: peak 18%, drawdown -12%.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "trail_18_12_only"},
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
    round_f.sell_rules_for_profile = sell_rules_for_profile


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
    topn = pd.Series(np.where(good, 12, 8), index=frame.index, dtype="int64")
    return frame[rank <= topn].copy()


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return pd.to_numeric(selected["base_score"], errors="coerce")


def sell_rules_for_profile(rules: list[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
    if profile is None:
        return rules
    out: list[dict[str, Any]] = []
    for rule in rules:
        item = dict(rule)
        name = str(item.get("name", ""))
        when = str(item.get("when", ""))
        if profile == "stop_loss_075_only" and (name.startswith("stop_loss") or "pnl_pct < -0.089" in when):
            item.update({"name": "stop_loss_75permil", "when": "pnl_pct < -0.075"})
        elif profile == "stop_loss_105_only" and (name.startswith("stop_loss") or "pnl_pct < -0.089" in when):
            item.update({"name": "stop_loss_105permil", "when": "pnl_pct < -0.105"})
        elif profile == "time_stop_20_only" and name.startswith("time_stop"):
            item.update({"name": "time_stop_20d", "when": "holding_days > 20"})
        elif profile == "time_stop_35_only" and name.startswith("time_stop"):
            item.update({"name": "time_stop_35d", "when": "holding_days > 35"})
        elif profile == "take_profit_180_only" and name.startswith("take_profit"):
            item.update({"name": "take_profit_180permil", "when": "pnl_pct > 0.180"})
        elif profile == "take_profit_260_only" and name.startswith("take_profit"):
            item.update({"name": "take_profit_260permil", "when": "pnl_pct > 0.260"})
        elif profile == "trail_12_08_only" and name.startswith("trail_peak"):
            item.update({"name": "trail_peak12_dd08_exempt_zx_rsv45_rsvs25", "when": "holding_days >= 5 and peak_pnl_pct > 0.12 and drawdown_from_peak < -0.08 and not (close > zxdq and close > zxdkx and zxdq > zxdkx and rsv_long > 45 and rsv_short < 25)"})
        elif profile == "trail_18_12_only" and name.startswith("trail_peak"):
            item.update({"name": "trail_peak18_dd12_exempt_zx_rsv45_rsvs25", "when": "holding_days >= 5 and peak_pnl_pct > 0.18 and drawdown_from_peak < -0.12 and not (close > zxdq and close > zxdkx and zxdq > zxdkx and rsv_long > 45 and rsv_short < 25)"})
        elif profile not in {
            "stop_loss_075_only", "stop_loss_105_only", "time_stop_20_only", "time_stop_35_only",
            "take_profit_180_only", "take_profit_260_only", "trail_12_08_only", "trail_18_12_only",
        }:
            raise ValueError(f"Unknown sell_profile: {profile}")
        out.append(item)
    return out


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_m_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_m_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_m_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_m_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_m_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_SELL_ISOLATION_ROUND_M_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_SELL_ISOLATION_ROUND_M.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_m_comparison.csv"
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
    path = output_root / "runs/round_m_comparison.csv"
    signal_path = output_root / "runs/round_m_signal_stats.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_SELL_ISOLATION_ROUND_M.md"
    lines = [
        "# Raw12 Sell Isolation Round M",
        "",
        "范围：保留 f3 信号、原弱转强排序、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和 `max_positions=12`；每次只改一个卖出规则族。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | avg_hold | signal_rows | vs_raw12_ret | vs_f3_ret | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(row.get('avg_holding_days')):.2f} | "
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
