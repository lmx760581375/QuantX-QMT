"""Round N conditional sell-rule tests for weak-to-strong f3.

Round M showed global shorter time stops cut drawdown but killed too much right
tail. This round keeps the f3 entry path unchanged and inserts narrow early-exit
rules before the original 25-day time stop.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_conditional_sell_round_n"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
ROUND_M_BEST_DD = ROOT / "raw12_sell_isolation_round_m/artifacts/m3_time_stop_20_only"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "n0_f3_reference", "description": "Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "n1_time20_loser", "description": "Exit after 20d only if current pnl is negative.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time20_loser"},
    {"variant": "n2_time20_flat_low_peak", "description": "Exit after 20d if low peak and pnl below 2%." , "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time20_flat_low_peak"},
    {"variant": "n3_time20_weak_trend_flat", "description": "Exit after 20d if flat/losing and stock trend is weak.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time20_weak_trend_flat"},
    {"variant": "n4_time20_drawdown_low_profit", "description": "Exit after 20d if drawdown from peak is large and profit is modest.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time20_drawdown_low_profit"},
    {"variant": "n5_time20_weak_path_combo", "description": "Exit after 20d for flat/losing weak path with weak trend confirmation.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time20_weak_path_combo"},
    {"variant": "n6_account_dd_time20_loser", "description": "Exit losing positions after 20d only during account drawdown.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "account_dd_time20_loser"},
    {"variant": "n7_account_dd15_early_loser", "description": "Earlier losing-position exit during deep account drawdown.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "account_dd15_early_loser"},
    {"variant": "n8_time18_bad_path", "description": "Exit after 18d for very weak path only.", "mode": "f3_reference", "topk": 12, "max_positions": 12, "sell_profile": "time18_bad_path"},
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
    extra = extra_sell_rule(str(profile))
    out: list[dict[str, Any]] = []
    inserted = False
    for rule in rules:
        name = str(rule.get("name", ""))
        if not inserted and name.startswith("time_stop"):
            out.append(extra)
            inserted = True
        out.append(dict(rule))
    if not inserted:
        out.append(extra)
    return out


def extra_sell_rule(profile: str) -> dict[str, Any]:
    rules = {
        "time20_loser": {
            "name": "early_time20_loser",
            "when": "holding_days > 20 and pnl_pct < 0",
        },
        "time20_flat_low_peak": {
            "name": "early_time20_flat_low_peak",
            "when": "holding_days > 20 and pnl_pct < 0.02 and peak_pnl_pct < 0.08",
        },
        "time20_weak_trend_flat": {
            "name": "early_time20_weak_trend_flat",
            "when": "holding_days > 20 and pnl_pct < 0.02 and not (close > zxdkx and zxdq > zxdkx)",
        },
        "time20_drawdown_low_profit": {
            "name": "early_time20_drawdown_low_profit",
            "when": "holding_days > 20 and pnl_pct < 0.08 and drawdown_from_peak < -0.08",
        },
        "time20_weak_path_combo": {
            "name": "early_time20_weak_path_combo",
            "when": "holding_days > 20 and pnl_pct < 0.02 and peak_pnl_pct < 0.10 and not (close > zxdkx and zxdq > zxdkx and rsv_long > 45)",
        },
        "account_dd_time20_loser": {
            "name": "early_account_dd_time20_loser",
            "when": "holding_days > 20 and pnl_pct < 0 and account_drawdown < -0.12",
        },
        "account_dd15_early_loser": {
            "name": "early_account_dd15_loser",
            "when": "holding_days > 15 and pnl_pct < 0 and account_drawdown < -0.15 and peak_pnl_pct < 0.06",
        },
        "time18_bad_path": {
            "name": "early_time18_bad_path",
            "when": "holding_days > 18 and pnl_pct < -0.02 and peak_pnl_pct < 0.04 and trough_pnl_pct < -0.08",
        },
    }
    if profile not in rules:
        raise ValueError(f"Unknown sell_profile: {profile}")
    return {**rules[profile], "action": "sell_all"}


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_signal_stats.csv", runs_dir / "round_n_signal_stats.csv"),
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_n_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_n_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_n_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_n_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_CONDITIONAL_SELL_ROUND_N_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_CONDITIONAL_SELL_ROUND_N.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_n_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {
        "raw12": read_json(RAW12_BASE / "summary.json"),
        "f3": read_json(ROUND_F_BEST / "summary.json"),
        "time20": read_json(ROUND_M_BEST_DD / "summary.json"),
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
    path = output_root / "runs/round_n_comparison.csv"
    signal_path = output_root / "runs/round_n_signal_stats.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    report = output_root / "reports/RAW12_CONDITIONAL_SELL_ROUND_N.md"
    lines = [
        "# Raw12 Conditional Sell Round N",
        "",
        "范围：保留 f3 信号、原弱转强排序、`selector.lag=1`、第二天 `close` 买入、`cash_equal` 和 `max_positions=12`；只在原 25 日 time stop 前加入条件化提前退出。",
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
