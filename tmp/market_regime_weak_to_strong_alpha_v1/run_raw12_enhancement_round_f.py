"""Round F raw-top12 return enhancement for weak-to-strong alpha.

This temporary research script stops capacity expansion and only works inside
the original weak-to-strong raw candidate pool. It validates dynamic TopN,
light re-ranking, and sell-rule variants with the formal QuantX backtest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_enhancement_round_f"
PROVIDER_URI = "data/qlib_data_fixed"
BACKTEST_START = "2016-01-04"

SOURCE_CONFIG = ROOT / "capacity_extension_round_a_current/configs/raw_top12_pos12.yaml"
RAW_CANDIDATES = ROOT / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet"


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "f1_bad_breadth_top8_else12",
        "description": "raw top12 normally; bear+breadth_lo days keep top8.",
        "mode": "dynamic_topn",
        "bad_topn": 8,
        "bad_condition": "bear_breadth_lo",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "f2_bad_breadth_top6_else12",
        "description": "raw top12 normally; bear+breadth_lo days keep top6.",
        "mode": "dynamic_topn",
        "bad_topn": 6,
        "bad_condition": "bear_breadth_lo",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "f3_good_top12_else_top8",
        "description": "use top12 only in bull or amount_hi states; otherwise top8.",
        "mode": "good_top12_else_top8",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "f4_bear_or_amountlo_top6_else12",
        "description": "raw top12 normally; bear or amount_lo days keep top6.",
        "mode": "dynamic_topn",
        "bad_topn": 6,
        "bad_condition": "bear_or_amount_lo",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "f5_raw12_state_residual_reorder",
        "description": "raw top12 with light point-in-time state/factor residual re-ranking.",
        "mode": "state_residual_reorder",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "f6_raw12_profit_let_run",
        "description": "raw top12 with slower take-profit and looser trailing/time stop.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sell_profile": "profit_let_run",
    },
    {
        "variant": "f7_raw12_fast_cut_loss",
        "description": "raw top12 with tighter loss/time exits for Sharpe protection.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sell_profile": "fast_cut_loss",
    },
    {
        "variant": "f8_bad_breadth_top8_profit_let_run",
        "description": "bear+breadth_lo top8 plus slower take-profit/trailing exits.",
        "mode": "dynamic_topn",
        "bad_topn": 8,
        "bad_condition": "bear_breadth_lo",
        "topk": 12,
        "max_positions": 12,
        "sell_profile": "profit_let_run",
    },
    {
        "variant": "f9_state_residual_profit_let_run",
        "description": "state residual re-ranking plus slower take-profit/trailing exits.",
        "mode": "state_residual_reorder",
        "topk": 12,
        "max_positions": 12,
        "sell_profile": "profit_let_run",
    },
)


REFERENCE_RUNS = {
    "current_top4_pos5": ROOT / "capacity_extension_round_a/reference_artifacts/current_precomputed_top4_pos5",
    "raw_top12_pos12": ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--python-cmd",
        nargs="+",
        default=["conda", "run", "--no-capture-output", "-n", "test", "python"],
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / OUTPUT_NAME
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    round_c = load_module(root / "run_market_state_ranking_round_c.py", "round_c_helpers")
    round_d = load_module(root / "run_market_state_risk_round_d.py", "round_d_helpers")

    if not args.analyze_only:
        candidates = round_c.load_enriched_candidates(root)
        signals = build_signals(candidates, signals_dir, runs_dir)
        configs = build_configs(signals, configs_dir)
        manifest = {
            "experiment": OUTPUT_NAME,
            "source_config": str(SOURCE_CONFIG),
            "raw_candidates": str(RAW_CANDIDATES),
            "backtest_start": BACKTEST_START,
            "execution_lag": 1,
            "deal_price": "close",
            "sizing": "cash_equal",
            "variants": signals,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_report(signals, reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md")
        if args.build_only:
            print(f"[round_f] build-only signals={len(signals)} configs={len(configs)}", flush=True)
            return

        run_rows = run_configs(configs, artifacts_dir, dry_run=args.dry_run, skip_existing=args.skip_existing, python_cmd=args.python_cmd)
        pd.DataFrame(run_rows).to_csv(runs_dir / ("dry_run_summary.csv" if args.dry_run else "backtest_summary.csv"), index=False)
        if args.dry_run:
            return

    collect_artifacts(artifacts_dir, runs_dir, reports_dir, round_d)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_signals(candidates: pd.DataFrame, signals_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    candidates = candidates.copy()
    candidates["base_score"] = pd.to_numeric(candidates["score"], errors="coerce")
    candidates["base_rank"] = candidates.groupby("signal_time")["base_score"].rank(method="first", ascending=False)
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for spec in VARIANTS:
        selected = select_variant(candidates, spec).copy()
        selected["score"] = score_variant(selected, spec)
        selected = selected.dropna(subset=["signal_time", "instrument", "score"])
        selected = selected.sort_values(["signal_time", "score", "instrument"], ascending=[True, False, True])
        path = signals_dir / f"{spec['variant']}.parquet"
        selected[["signal_time", "instrument", "score", "raw_rank", "base_rank", "regime", "amount_state", "breadth_state"]].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        rows.append(manifest)
        stats.append({**manifest, **signal_stats(selected)})
    pd.DataFrame(stats).to_csv(runs_dir / "round_f_signal_stats.csv", index=False)
    return rows


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = candidates[pd.to_numeric(candidates["base_rank"], errors="coerce") <= int(spec["topk"])].copy()
    mode = str(spec["mode"])
    rank = pd.to_numeric(frame["base_rank"], errors="coerce")
    if mode in {"raw12", "state_residual_reorder"}:
        return frame
    if mode == "dynamic_topn":
        bad = bad_state_mask(frame, str(spec["bad_condition"]))
        keep = (~bad & (rank <= int(spec["topk"]))) | (bad & (rank <= int(spec["bad_topn"])))
        return frame[keep].copy()
    if mode == "good_top12_else_top8":
        good = frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")
        keep = (good & (rank <= 12)) | (~good & (rank <= 8))
        return frame[keep].copy()
    raise ValueError(f"Unknown mode: {mode}")


def bad_state_mask(frame: pd.DataFrame, condition: str) -> pd.Series:
    if condition == "bear_breadth_lo":
        return frame["regime"].eq("bear") & frame["breadth_state"].eq("lo")
    if condition == "bear_or_amount_lo":
        return frame["regime"].eq("bear") | frame["amount_state"].eq("lo")
    raise ValueError(f"Unknown bad_condition: {condition}")


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    base = pd.to_numeric(selected["base_score"], errors="coerce")
    if str(spec["mode"]) != "state_residual_reorder":
        return base
    base = 100000.0 - rank
    residual = pd.Series(0.0, index=selected.index)
    residual += 0.90 * selected["breadth_state"].eq("lo").astype(float)
    residual += 0.50 * selected["regime"].eq("bear").astype(float)
    residual += 0.35 * selected["amount_state"].eq("hi").astype(float)
    residual -= 0.30 * selected["amount_state"].eq("lo").astype(float)
    residual += 0.28 * zscore(selected.get("repair_rank"))
    residual += 0.22 * zscore(selected.get("position_strength"))
    residual -= 0.18 * zscore(selected.get("amplitude"))
    return base + residual


def zscore(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(0.0)
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return ((values - values.mean()) / std).fillna(0.0)


def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
    counts = signals.groupby("signal_time", sort=True).size() if not signals.empty else pd.Series(dtype=int)
    bad_days = signals[bad_state_mask(signals, "bear_breadth_lo")]["signal_time"].nunique() if {"regime", "breadth_state"}.issubset(signals.columns) else 0
    return {
        "variant": str(spec["variant"]),
        "description": str(spec.get("description", "")),
        "path": str(path),
        "mode": str(spec["mode"]),
        "topk": int(spec["topk"]),
        "max_positions": int(spec["max_positions"]),
        "sizing": "cash_equal",
        "sell_profile": spec.get("sell_profile"),
        "bad_condition": spec.get("bad_condition"),
        "bad_topn": spec.get("bad_topn"),
        "signal_rows": int(len(signals)),
        "signal_days": int(counts.size),
        "days_with_at_least_topk": int((counts >= int(spec["topk"])).sum()) if not counts.empty else 0,
        "bear_breadth_lo_signal_days": int(bad_days),
        "first_signal": str(signals["signal_time"].min()) if not signals.empty else None,
        "last_signal": str(signals["signal_time"].max()) if not signals.empty else None,
    }


def signal_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"avg_daily_candidates": np.nan, "median_daily_candidates": np.nan, "max_daily_candidates": np.nan}
    counts = signals.groupby("signal_time").size()
    return {
        "avg_daily_candidates": float(counts.mean()),
        "median_daily_candidates": float(counts.median()),
        "max_daily_candidates": int(counts.max()),
    }


def build_configs(signals: list[dict[str, Any]], configs_dir: Path) -> list[Path]:
    source = read_yaml(SOURCE_CONFIG)
    paths: list[Path] = []
    for signal in signals:
        config = make_config(source, signal)
        path = configs_dir / f"{signal['variant']}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def make_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(source)
    config["name"] = f"true_wts_{OUTPUT_NAME}_{signal['variant']}"
    config["title"] = "弱转强 raw top12 收益增强 Round F"
    config["description"] = "Tmp experiment: enhance returns inside original raw top12 candidates only."
    config["data"] = dict(config.get("data") or {})
    config["data"].update({"provider_uri": PROVIDER_URI, "universe": "external_score", "start": BACKTEST_START, "end": "latest"})
    config["selector"] = {
        "mode": "external_score",
        "path": signal["path"],
        "date_col": "signal_time",
        "instrument_col": "instrument",
        "score_col": "score",
        "sort": "score_desc",
        "topk": int(signal["topk"]),
        "lag": 1,
        "candidate_limit": max(20, int(signal["topk"])),
        "reason": f"true_wts_{OUTPUT_NAME}_{signal['variant']}",
    }
    config["rebalance"] = dict(config.get("rebalance") or {})
    config["rebalance"].update({"max_positions": int(signal["max_positions"]), "cash_use_ratio": 0.98, "buy_only_new_positions": True})
    config["rebalance"].pop("rank_weights", None)
    config["execution"] = dict(config.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["execution"]["buy"] = dict(config["execution"].get("buy") or {})
    config["execution"]["buy"].update({"sizing": "cash_equal", "lot_size": 100, "skip_if_holding": True, "skip_limit_up": True, "reuse_sell_cash": True})
    config["execution"]["sell_rules"] = sell_rules_for_profile(config["execution"].get("sell_rules") or [], signal.get("sell_profile"))
    config["engine"] = dict(config.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "variant": signal["variant"],
        "mode": signal.get("mode"),
        "source_config": str(SOURCE_CONFIG),
        "raw_candidates": str(RAW_CANDIDATES),
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "sizing": "cash_equal",
        "sell_profile": signal.get("sell_profile"),
        "bad_condition": signal.get("bad_condition"),
        "bad_topn": signal.get("bad_topn"),
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal["signal_rows"]),
        "signal_days": int(signal["signal_days"]),
        "days_with_at_least_topk": int(signal["days_with_at_least_topk"]),
    }
    return config


def sell_rules_for_profile(rules: list[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
    if profile is None:
        return rules
    out: list[dict[str, Any]] = []
    for rule in rules:
        item = dict(rule)
        name = str(item.get("name", ""))
        when = str(item.get("when", ""))
        if profile == "profit_let_run":
            if name.startswith("scale_12"):
                item.update({"name": "scale_18_keep90", "when": "pnl_pct > 0.180 and remaining_position_pct > 0.90", "position_pct": 0.90})
            elif name.startswith("take_profit"):
                item.update({"name": "take_profit_320permil", "when": "pnl_pct > 0.320"})
            elif name.startswith("trail_peak15"):
                item.update({"name": "trail_peak22_dd12_exempt_zx_rsv45_rsvs25", "when": "holding_days >= 5 and peak_pnl_pct > 0.22 and drawdown_from_peak < -0.12 and not (close > zxdq and close > zxdkx and zxdq > zxdkx and rsv_long > 45 and rsv_short < 25)"})
            elif name.startswith("time_stop"):
                item.update({"name": "time_stop_35d", "when": "holding_days > 35"})
        elif profile == "fast_cut_loss":
            if name.startswith("stop_loss") or "pnl_pct < -0.089" in when:
                item.update({"name": "stop_loss_75permil", "when": "pnl_pct < -0.075"})
            elif name.startswith("fast_exit"):
                item.update({"name": "fast_exit_weak_early_hold_narrow_12d", "when": "holding_days > 12 and pnl_pct < 0 and hold_first_3d_return < -0.01 and hold_first_10d_return < -0.035 and peak_pnl_pct < 0.02 and trough_pnl_pct < -0.07"})
            elif name.startswith("time_stop"):
                item.update({"name": "time_stop_20d", "when": "holding_days > 20"})
        else:
            raise ValueError(f"Unknown sell_profile: {profile}")
        out.append(item)
    return out


def run_configs(configs: list[Path], artifacts_dir: Path, *, dry_run: bool, skip_existing: bool, python_cmd: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[round_f] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[round_f] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
        cmd = [*python_cmd, "-m", "quantx.tools.run_backtest", "--config", str(config_path), "--json"]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--output-dir", str(artifacts_dir), "--run-id", run_id])
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        row = parse_run_output(completed.stdout)
        row.update({"config": str(config_path), "run_id": run_id, "returncode": int(completed.returncode), "stderr_tail": completed.stderr[-2000:] if completed.stderr else ""})
        rows.append(flatten(row))
        if completed.returncode != 0:
            print(completed.stderr[-4000:], flush=True)
            raise RuntimeError(f"Backtest failed for {config_path}")
    return rows


def collect_artifacts(artifacts_dir: Path, runs_dir: Path, reports_dir: Path, round_d) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[round_f] no artifacts found under {artifacts_dir}", flush=True)
        return
    reference = load_reference_metrics(round_d)
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        run_id = artifact_dir.name
        summary = read_json(artifact_dir / "summary.json")
        metrics = read_json(artifact_dir / "metrics.json")
        config = read_yaml(artifact_dir / "config.yaml")
        metadata = dict(config.get("metadata") or {})
        nav = read_json_frame(artifact_dir / "daily_nav.json")
        row = {"run_id": run_id}
        for key in ("name", "start_date", "end_date", "final_value", "total_return", "annual_return", "annual_volatility", "sharpe", "sortino", "calmar", "max_drawdown", "buy_count", "sell_count", "reject_count"):
            row[key] = summary.get(key)
        for key in ("closed_position_count", "win_rate", "profit_factor", "avg_closed_return", "avg_holding_days", "avg_position_count", "max_position_count", "avg_capital_utilization", "zero_utilization_day_ratio"):
            row[key] = metrics.get(key)
        row.update(metadata)
        row.update(round_d.position_stats(nav, "full"))
        post2022 = nav[pd.to_datetime(nav["date"]) >= pd.Timestamp("2022-01-01")] if not nav.empty and "date" in nav else nav
        row.update(round_d.position_stats(post2022, "post2022"))
        row.update(round_d.window_return_stats(nav, "2022-03-31", "2022-05-05", "crash20"))
        row.update(round_d.exposure_summary(artifact_dir, "full"))
        row.update(round_d.exposure_summary(artifact_dir, "y2022", start="2022-01-01", end="2022-12-31"))
        rows.append(row)
        yearly_rows.extend(round_d.summarize_yearly_nav(run_id, nav, metadata))
        exposure_rows.extend(round_d.exposure_buckets(artifact_dir, run_id))
    metrics_frame = pd.DataFrame(rows).sort_values(["variant"])
    metrics_frame.to_csv(runs_dir / "round_f_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "round_f_yearly_nav.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(runs_dir / "round_f_entry_weight_buckets.csv", index=False)
    comparison = write_comparison(metrics_frame, reference, runs_dir / "round_f_comparison.csv")
    write_report(comparison, reference, reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md")
    print(f"[round_f] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def load_reference_metrics(round_d) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, run_dir in REFERENCE_RUNS.items():
        summary = read_json(run_dir / "summary.json")
        metrics = read_json(run_dir / "metrics.json")
        row = {**summary, **metrics}
        nav = read_json_frame(run_dir / "daily_nav.json")
        row.update(round_d.position_stats(nav, "full"))
        post2022 = nav[pd.to_datetime(nav["date"]) >= pd.Timestamp("2022-01-01")] if not nav.empty and "date" in nav else nav
        row.update(round_d.position_stats(post2022, "post2022"))
        row.update(round_d.window_return_stats(nav, "2022-03-31", "2022-05-05", "crash20"))
        row.update(round_d.exposure_summary(run_dir, "full"))
        row.update(round_d.exposure_summary(run_dir, "y2022", start="2022-01-01", end="2022-12-31"))
        result[name] = row
    return result


def write_comparison(metrics: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw12 = reference.get("raw_top12_pos12", {})
    top4 = reference.get("current_top4_pos5", {})
    for _, row in metrics.iterrows():
        out = row.to_dict()
        for prefix, ref in (("vs_raw12", raw12), ("vs_top4", top4)):
            out[f"{prefix}_total_return_delta"] = safe_float(row.get("total_return")) - safe_float(ref.get("total_return"))
            out[f"{prefix}_mdd_delta"] = safe_float(row.get("max_drawdown")) - safe_float(ref.get("max_drawdown"))
            out[f"{prefix}_sharpe_delta"] = safe_float(row.get("sharpe")) - safe_float(ref.get("sharpe"))
            out[f"{prefix}_y2022_max_buy_weight_delta"] = safe_float(row.get("y2022_max_buy_weight")) - safe_float(ref.get("y2022_max_buy_weight"))
        rows.append(out)
    comparison = pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=[False, False])
    comparison.to_csv(path, index=False)
    return comparison


def write_pre_report(signals: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Raw12 Enhancement Round F Pre-Backtest", "", "只在原弱转强 raw top12 内做动态 TopN、重排序、卖出规则优化。", "", "| variant | mode | sell | rows | days | avg candidates | days>=12 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for signal in signals:
        lines.append(f"| {signal['variant']} | {signal['mode']} | {signal.get('sell_profile')} | {signal['signal_rows']} | {signal['signal_days']} | {signal.get('avg_daily_candidates', np.nan):.2f} | {signal['days_with_at_least_topk']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(comparison: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> None:
    lines = ["# Raw12 Enhancement Round F", "", "范围：不新增候选，只在原弱转强 raw top12 内做增强；保持 `selector.lag=1`、第二天 `close` 买入和 `cash_equal`。", "", "## Baselines", "", "| baseline | total_return | max_drawdown | sharpe | avg_pos | post2022_zero |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in reference.items():
        lines.append(f"| {name} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | {safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | {safe_float(row.get('post2022_zero_position_ratio')):.4f} |")
    lines.extend(["", "## Results", "", "| variant | ret | mdd | sharpe | avg_pos | crash20_ret | 2022 max_w | vs_raw12_ret | vs_raw12_mdd | vs_raw12_sharpe |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for _, row in comparison.iterrows():
        lines.append(f"| {row.get('variant')} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | {safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | {safe_float(row.get('crash20_return')):.4f} | {safe_float(row.get('y2022_max_buy_weight')):.4f} | {safe_float(row.get('vs_raw12_total_return_delta')):.4f} | {safe_float(row.get('vs_raw12_mdd_delta')):.4f} | {safe_float(row.get('vs_raw12_sharpe_delta')):.4f} |")
    lines.extend(["", "## Artifacts", "", "- `runs/round_f_comparison.csv`", "- `runs/round_f_signal_stats.csv`", "- `runs/round_f_yearly_nav.csv`", "- `runs/round_f_entry_weight_buckets.csv`", "- `runs/round_f_metrics_full.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_run_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {"raw_stdout": text[-2000:]}
    return json.loads(text[start:])


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
