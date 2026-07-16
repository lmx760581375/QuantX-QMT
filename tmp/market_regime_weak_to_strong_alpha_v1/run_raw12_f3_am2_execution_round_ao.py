"""Round AO execution and holding-path tests for AM2.

This round keeps the AM2 signal fixed: same candidate pool, ordering, top12,
selector lag=1, next-day close execution, cash-equal sizing, and max 12
positions. It tests only small QuantX execution/rebalance configuration changes
so any account-level movement is attributable to capital usage, rank weights,
or a single sell-rule family.
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
OUTPUT_NAME = "raw12_f3_am2_execution_round_ao"
SOURCE_CONFIG = ROOT / "raw12_f3_ag4_path_scan_round_an/configs/an1_am2_reference.yaml"
SOURCE_SIGNAL = ROOT / "raw12_f3_ag4_path_scan_round_an/signals/an1_am2_reference.parquet"
AK4_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_ak/artifacts/ak4_bull_hihi_ml5_low"
AM2_ARTIFACT = ROOT / "raw12_f3_ag4_path_scan_round_an/artifacts/an1_am2_reference"
ROUND_D_SCRIPT = ROOT / "run_market_state_risk_round_d.py"


FRONT_105 = [1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]
FRONT_115 = [1.32, 1.25, 1.18, 1.12, 1.06, 1.00, 0.95, 0.90, 0.85, 0.80, 0.76, 0.72]
BACK_105 = list(reversed(FRONT_105))


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "ao0_am2_reference", "description": "Exact AM2/AN1 reference."},
    {"variant": "ao1_cash100", "description": "AM2 with cash_use_ratio raised from 0.98 to 1.00.", "cash_use_ratio": 1.00},
    {"variant": "ao2_front105", "description": "AM2 with very mild front-loaded rank weights.", "rank_weights": FRONT_105},
    {"variant": "ao3_front115", "description": "AM2 with prior Round-I mild front-loaded rank weights.", "rank_weights": FRONT_115},
    {"variant": "ao4_back105", "description": "AM2 with very mild back-loaded rank weights.", "rank_weights": BACK_105},
    {"variant": "ao5_scale12_keep95", "description": "AM2 selling less at the first +12% scale-out.", "sell_profile": "scale12_keep95"},
    {"variant": "ao6_scale15_keep90", "description": "AM2 delaying the first scale-out to +15% and keeping 90%.", "sell_profile": "scale15_keep90"},
    {"variant": "ao7_take_profit240", "description": "AM2 with only take-profit lifted from +21% to +24%.", "sell_profile": "take_profit240"},
    {"variant": "ao8_trail18_dd12", "description": "AM2 with only trailing stop loosened to peak +18%, drawdown -12%.", "sell_profile": "trail18_dd12"},
    {"variant": "ao9_time_stop30", "description": "AM2 with only time stop extended from 25d to 30d.", "sell_profile": "time_stop30"},
    {"variant": "ao10_cash100_scale12_keep95", "description": "AM2 cash100 plus less first scale-out.", "cash_use_ratio": 1.00, "sell_profile": "scale12_keep95"},
)


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
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    for directory in (configs_dir, artifacts_dir, runs_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        source = read_yaml(SOURCE_CONFIG)
        signal_stats = inspect_signal(SOURCE_SIGNAL)
        configs = build_configs(source, signal_stats, configs_dir)
        manifest = {
            "experiment": OUTPUT_NAME,
            "source_config": str(SOURCE_CONFIG),
            "source_signal": str(SOURCE_SIGNAL),
            "execution_lag": 1,
            "deal_price": "close",
            "sizing": "cash_equal",
            "variants": [{**spec, **signal_stats} for spec in VARIANTS],
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_report(signal_stats, reports_dir / "RAW12_F3_AM2_EXECUTION_ROUND_AO_PRE.md")
        if args.build_only:
            print(f"[round_ao] build-only configs={len(configs)}", flush=True)
            return

        run_rows = run_configs(configs, artifacts_dir, dry_run=args.dry_run, skip_existing=args.skip_existing, python_cmd=args.python_cmd)
        pd.DataFrame(run_rows).to_csv(runs_dir / ("dry_run_summary.csv" if args.dry_run else "backtest_summary.csv"), index=False)
        if args.dry_run:
            return

    round_d = load_round_d()
    collect_artifacts(artifacts_dir, runs_dir, reports_dir, round_d)


def inspect_signal(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    counts = frame.groupby("signal_time").size() if not frame.empty else pd.Series(dtype=int)
    return {
        "signal_path": str(path),
        "signal_rows": int(len(frame)),
        "signal_days": int(counts.size),
        "avg_daily_candidates": float(counts.mean()) if not counts.empty else np.nan,
        "max_daily_candidates": int(counts.max()) if not counts.empty else 0,
        "first_signal": str(frame["signal_time"].min()) if not frame.empty else None,
        "last_signal": str(frame["signal_time"].max()) if not frame.empty else None,
    }


def build_configs(source: dict[str, Any], signal_stats: dict[str, Any], configs_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for spec in VARIANTS:
        config = make_config(source, spec, signal_stats)
        path = configs_dir / f"{spec['variant']}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def make_config(source: dict[str, Any], spec: dict[str, Any], signal_stats: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(source)
    variant = str(spec["variant"])
    config["name"] = f"true_wts_{OUTPUT_NAME}_{variant}"
    config["title"] = "弱转强 AM2 执行与持仓路径 Round AO"
    config["description"] = str(spec.get("description", ""))
    config["selector"] = dict(config.get("selector") or {})
    config["selector"].update({
        "path": str(SOURCE_SIGNAL),
        "date_col": "signal_time",
        "instrument_col": "instrument",
        "score_col": "score",
        "sort": "score_desc",
        "topk": 12,
        "lag": 1,
        "candidate_limit": 20,
        "reason": f"true_wts_{OUTPUT_NAME}_{variant}",
    })
    config["rebalance"] = dict(config.get("rebalance") or {})
    config["rebalance"].update({
        "type": "equal_weight",
        "max_positions": 12,
        "cash_use_ratio": float(spec.get("cash_use_ratio", 0.98)),
        "buy_only_new_positions": True,
    })
    rank_weights = spec.get("rank_weights") or []
    if rank_weights:
        config["rebalance"]["rank_weights"] = [float(value) for value in rank_weights]
    else:
        config["rebalance"].pop("rank_weights", None)
    config["execution"] = dict(config.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["execution"]["buy"] = dict(config["execution"].get("buy") or {})
    config["execution"]["buy"].update({
        "sizing": "cash_equal",
        "lot_size": 100,
        "skip_if_holding": True,
        "skip_limit_up": True,
        "reuse_sell_cash": True,
    })
    config["execution"]["sell_rules"] = sell_rules_for_profile(config["execution"].get("sell_rules") or [], spec.get("sell_profile"))
    config["engine"] = dict(config.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    metadata = dict(config.get("metadata") or {})
    metadata.update({
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "variant": variant,
        "source_config": str(SOURCE_CONFIG),
        "source_signal": str(SOURCE_SIGNAL),
        "topk": 12,
        "max_positions": 12,
        "sizing": "cash_equal",
        "cash_use_ratio": float(spec.get("cash_use_ratio", 0.98)),
        "rank_weights": rank_weights,
        "sell_profile": spec.get("sell_profile"),
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal_stats["signal_rows"]),
        "signal_days": int(signal_stats["signal_days"]),
        "max_daily_candidates": int(signal_stats["max_daily_candidates"]),
    })
    config["metadata"] = metadata
    return config


def sell_rules_for_profile(rules: list[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
    if profile is None:
        return rules
    out: list[dict[str, Any]] = []
    for rule in rules:
        item = dict(rule)
        name = str(item.get("name", ""))
        if profile == "scale12_keep95" and name.startswith("scale_12"):
            item.update({"name": "scale_12_keep95", "position_pct": 0.95})
        elif profile == "scale15_keep90" and name.startswith("scale_12"):
            item.update({"name": "scale_15_keep90", "when": "pnl_pct > 0.150 and remaining_position_pct > 0.90", "position_pct": 0.90})
        elif profile == "take_profit240" and name.startswith("take_profit"):
            item.update({"name": "take_profit_240permil", "when": "pnl_pct > 0.240"})
        elif profile == "trail18_dd12" and name.startswith("trail_peak"):
            item.update({"name": "trail_peak18_dd12_exempt_zx_rsv45_rsvs25", "when": "holding_days >= 5 and peak_pnl_pct > 0.18 and drawdown_from_peak < -0.12 and not (close > zxdq and close > zxdkx and zxdq > zxdkx and rsv_long > 45 and rsv_short < 25)"})
        elif profile == "time_stop30" and name.startswith("time_stop"):
            item.update({"name": "time_stop_30d", "when": "holding_days > 30"})
        elif profile not in {"scale12_keep95", "scale15_keep90", "take_profit240", "trail18_dd12", "time_stop30"}:
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
            print(f"[round_ao] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[round_ao] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
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
        print(f"[round_ao] no artifacts found under {artifacts_dir}", flush=True)
        return
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
    metrics_frame.to_csv(runs_dir / "round_ao_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "round_ao_yearly_nav.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(runs_dir / "round_ao_entry_weight_buckets.csv", index=False)
    comparison = write_comparison(metrics_frame, runs_dir / "round_ao_comparison.csv")
    write_report(comparison, reports_dir / "RAW12_F3_AM2_EXECUTION_ROUND_AO.md")
    print(f"[round_ao] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def write_comparison(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    refs = {"ak4": read_json(AK4_ARTIFACT / "summary.json"), "am2": read_json(AM2_ARTIFACT / "summary.json")}
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        out = row.to_dict()
        for name, ref in refs.items():
            out[f"vs_{name}_total_return_delta"] = safe_float(row.get("total_return")) - safe_float(ref.get("total_return"))
            out[f"vs_{name}_mdd_delta"] = safe_float(row.get("max_drawdown")) - safe_float(ref.get("max_drawdown"))
            out[f"vs_{name}_sharpe_delta"] = safe_float(row.get("sharpe")) - safe_float(ref.get("sharpe"))
        rows.append(out)
    comparison = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)
    return comparison


def write_pre_report(signal_stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# Raw12 F3 AM2 Execution Round AO Pre-Backtest",
        "",
        "固定 AM2/AN1 信号，只测试 QuantX 配置层的轻量执行、权重、卖出阈值变化。",
        "",
        f"Signal rows: {signal_stats['signal_rows']}; days: {signal_stats['signal_days']}; max daily candidates: {signal_stats['max_daily_candidates']}.",
        "",
        "| variant | cash | rank_weights | sell_profile | description |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for spec in VARIANTS:
        weights = "yes" if spec.get("rank_weights") else "no"
        lines.append(f"| {spec['variant']} | {float(spec.get('cash_use_ratio', 0.98)):.2f} | {weights} | {spec.get('sell_profile')} | {spec.get('description', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(comparison: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Raw12 F3 AM2 Execution Round AO",
        "",
        "范围：固定 AM2/AN1 信号、`topk=12`、`max_positions=12`、`selector.lag=1`、第二天 `close` 买入、`cash_equal`；只改现金使用率、rank_weights 或单项卖出阈值。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | avg_hold | vs_am2_ret | vs_am2_mdd | vs_am2_sharpe | vs_ak4_ret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row.get('variant')} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {safe_float(row.get('avg_holding_days')):.2f} | "
            f"{safe_float(row.get('vs_am2_total_return_delta')):.4f} | {safe_float(row.get('vs_am2_mdd_delta')):.4f} | "
            f"{safe_float(row.get('vs_am2_sharpe_delta')):.4f} | {safe_float(row.get('vs_ak4_total_return_delta')):.4f} |"
        )
    lines.extend(["", "## Artifacts", "", "- `runs/round_ao_comparison.csv`", "- `runs/round_ao_metrics_full.csv`", "- `runs/round_ao_yearly_nav.csv`", "- `runs/round_ao_entry_weight_buckets.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_round_d():
    spec = importlib.util.spec_from_file_location("round_d_helpers", ROUND_D_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Round D helpers from {ROUND_D_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
