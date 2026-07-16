"""Round D risk/sizing validation for weak-to-strong capacity alpha.

This temporary research script keeps the original weak-to-strong candidate
alpha fixed and validates whether sizing/risk overlays improve the formal
QuantX backtest. It reuses Round C helpers for loading candidates, running
configs, and collecting comparable metrics.
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
OUTPUT_NAME = "market_state_risk_round_d"
PROVIDER_URI = "data/qlib_data_fixed"
BACKTEST_START = "2016-01-04"

SOURCE_CONFIG = ROOT / "capacity_extension_round_a_current/configs/raw_top12_pos12.yaml"
ROUND_C_SCRIPT = ROOT / "run_market_state_ranking_round_c.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "d1_raw12_slot_equal",
        "description": "raw top12, max 12, slot-equal sizing to cap sparse-day single-name exposure.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
    },
    {
        "variant": "d2_raw12_slot_rank_tail_half",
        "description": "slot-equal with rank weights: ranks 1-4 full, ranks 5-12 half.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
        "rank_weights": [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "variant": "d3_raw12_slot_rank_tail_quarter",
        "description": "slot-equal with rank weights: ranks 1-4 full, ranks 5-12 quarter.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
        "rank_weights": [1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    },
    {
        "variant": "d4_raw12_slot_amount_residual",
        "description": "slot-equal plus mild residual re-score from amount_hi, low amplitude, repair/position features.",
        "mode": "amount_residual",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
    },
    {
        "variant": "d5_raw12_slot_badstate_top8",
        "description": "slot-equal, top12 normally; bear+amount_lo days keep top8 instead of going empty.",
        "mode": "badstate_top8",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
    },
    {
        "variant": "d6_raw12_cash_stop75",
        "description": "baseline cash-equal top12 with tighter 7.5% stop-loss.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "cash_equal",
        "stop_loss": -0.075,
    },
    {
        "variant": "d7_raw12_slot_stop75",
        "description": "slot-equal top12 with tighter 7.5% stop-loss.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
        "stop_loss": -0.075,
    },
    {
        "variant": "d8_raw12_slot_rank_frontloaded",
        "description": "slot-equal but front-load ranks 1-4 without allowing sparse-day all-in behavior.",
        "mode": "raw12",
        "topk": 12,
        "max_positions": 12,
        "sizing": "slot_equal",
        "rank_weights": [1.4, 1.2, 1.0, 0.9, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3],
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

    if not args.analyze_only:
        helpers = load_round_c_helpers(root)
        candidates = helpers.load_enriched_candidates(root)
        signals = build_signals(candidates, signals_dir, runs_dir)
        configs = build_configs(signals, configs_dir)
        manifest = {
            "experiment": OUTPUT_NAME,
            "source_config": str(SOURCE_CONFIG),
            "backtest_start": BACKTEST_START,
            "execution_lag": 1,
            "deal_price": "close",
            "variants": signals,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_report(signals, reports_dir / "MARKET_STATE_RISK_ROUND_D_PRE.md")
        if args.build_only:
            print(f"[round_d] build-only signals={len(signals)} configs={len(configs)}", flush=True)
            return

        run_rows = run_configs(
            configs,
            artifacts_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            python_cmd=args.python_cmd,
        )
        pd.DataFrame(run_rows).to_csv(
            runs_dir / ("dry_run_summary.csv" if args.dry_run else "backtest_summary.csv"),
            index=False,
        )
        if args.dry_run:
            return

    collect_artifacts(artifacts_dir, runs_dir, reports_dir)


def load_round_c_helpers(root: Path):
    script = root / "run_market_state_ranking_round_c.py"
    spec = importlib.util.spec_from_file_location("round_c_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_signals(candidates: pd.DataFrame, signals_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for spec in VARIANTS:
        selected = select_variant(candidates, spec).copy()
        selected["score"] = score_variant(selected, spec)
        selected = selected.dropna(subset=["signal_time", "instrument", "score"])
        selected = selected.sort_values(["signal_time", "score"], ascending=[True, False])
        path = signals_dir / f"{spec['variant']}.parquet"
        selected[["signal_time", "instrument", "score", "raw_rank"]].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        rows.append(manifest)
        stats.append({**manifest, **signal_stats(selected)})
    pd.DataFrame(stats).to_csv(runs_dir / "round_d_signal_stats.csv", index=False)
    return rows


def select_variant(candidates: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rank = pd.to_numeric(candidates["raw_rank"], errors="coerce")
    if spec["mode"] in {"raw12", "amount_residual"}:
        return candidates[rank <= int(spec["topk"])].copy()
    if spec["mode"] == "badstate_top8":
        bad_state = candidates["regime"].eq("bear") & candidates["amount_state"].eq("lo")
        selected = (bad_state & (rank <= 8)) | (~bad_state & (rank <= 12))
        return candidates[selected].copy()
    raise ValueError(f"Unknown mode: {spec['mode']}")


def score_variant(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    base = pd.to_numeric(selected["score"], errors="coerce")
    if spec["mode"] != "amount_residual":
        return base
    raw_rank = pd.to_numeric(selected["raw_rank"], errors="coerce").fillna(99.0)
    tail_scale = np.where(raw_rank <= 4, 0.25, 1.0)
    residual = 0.0
    residual += 0.080 * selected["amount_state"].eq("hi").astype(float)
    residual -= 0.045 * selected["amount_state"].eq("lo").astype(float)
    residual += 0.040 * selected["regime"].eq("bull").astype(float)
    residual -= 0.030 * selected["regime"].eq("bear").astype(float)
    residual += 0.030 * zscore(selected.get("repair_rank"))
    residual += 0.025 * zscore(selected.get("position_strength"))
    residual -= 0.030 * zscore(selected.get("amplitude"))
    residual -= 0.010 * raw_rank
    return base.fillna(0.0) + residual * tail_scale


def zscore(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return ((values - values.mean()) / std).fillna(0.0)


def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
    counts = signals.groupby("signal_time", sort=True).size() if not signals.empty else pd.Series(dtype=int)
    return {
        "variant": str(spec["variant"]),
        "description": str(spec.get("description", "")),
        "path": str(path),
        "mode": str(spec["mode"]),
        "topk": int(spec["topk"]),
        "max_positions": int(spec["max_positions"]),
        "sizing": str(spec["sizing"]),
        "rank_weights": spec.get("rank_weights"),
        "stop_loss": spec.get("stop_loss"),
        "signal_rows": int(len(signals)),
        "signal_days": int(counts.size),
        "days_with_at_least_topk": int((counts >= int(spec["topk"])).sum()) if not counts.empty else 0,
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
    config["title"] = "弱转强风险仓位扩容 Round D"
    config["description"] = "Tmp experiment: keep raw weak-to-strong top12 alpha fixed and validate sizing/risk overlays."
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
    config["rebalance"]["max_positions"] = int(signal["max_positions"])
    config["rebalance"]["cash_use_ratio"] = 0.98
    config["rebalance"]["buy_only_new_positions"] = True
    if signal.get("rank_weights"):
        config["rebalance"]["rank_weights"] = [float(x) for x in signal["rank_weights"]]
    else:
        config["rebalance"].pop("rank_weights", None)
    config["execution"] = dict(config.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["execution"]["buy"] = dict(config["execution"].get("buy") or {})
    config["execution"]["buy"].update({"sizing": signal["sizing"], "lot_size": 100, "skip_if_holding": True, "skip_limit_up": True, "reuse_sell_cash": True})
    if signal.get("stop_loss") is not None:
        config["execution"]["sell_rules"] = replace_stop_loss(config["execution"].get("sell_rules") or [], float(signal["stop_loss"]))
    config["engine"] = dict(config.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "variant": signal["variant"],
        "mode": signal.get("mode"),
        "source_config": str(SOURCE_CONFIG),
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "sizing": signal["sizing"],
        "rank_weights": signal.get("rank_weights"),
        "stop_loss": signal.get("stop_loss"),
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal["signal_rows"]),
        "signal_days": int(signal["signal_days"]),
        "days_with_at_least_topk": int(signal["days_with_at_least_topk"]),
    }
    return config


def replace_stop_loss(rules: list[dict[str, Any]], stop_loss: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    replaced = False
    for rule in rules:
        item = dict(rule)
        name = str(item.get("name", ""))
        when = str(item.get("when", ""))
        if name.startswith("stop_loss") or "pnl_pct < -0.089" in when:
            item["name"] = f"stop_loss_{abs(stop_loss) * 1000:.0f}permil"
            item["when"] = f"pnl_pct < {stop_loss:.3f}"
            item["action"] = "sell_all"
            replaced = True
        out.append(item)
    if not replaced:
        out.insert(0, {"name": f"stop_loss_{abs(stop_loss) * 1000:.0f}permil", "when": f"pnl_pct < {stop_loss:.3f}", "action": "sell_all"})
    return out


def run_configs(configs: list[Path], artifacts_dir: Path, *, dry_run: bool, skip_existing: bool, python_cmd: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[round_d] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[round_d] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
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


def collect_artifacts(artifacts_dir: Path, runs_dir: Path, reports_dir: Path) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[round_d] no artifacts found under {artifacts_dir}", flush=True)
        return
    reference = load_reference_metrics()
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
        row.update(position_stats(nav, "full"))
        post2022 = nav[pd.to_datetime(nav["date"]) >= pd.Timestamp("2022-01-01")] if not nav.empty and "date" in nav else nav
        row.update(position_stats(post2022, "post2022"))
        row.update(window_return_stats(nav, "2022-03-31", "2022-05-05", "crash20"))
        row.update(exposure_summary(artifact_dir, "full"))
        row.update(exposure_summary(artifact_dir, "y2022", start="2022-01-01", end="2022-12-31"))
        rows.append(row)
        yearly_rows.extend(summarize_yearly_nav(run_id, nav, metadata))
        exposure_rows.extend(exposure_buckets(artifact_dir, run_id))
    metrics_frame = pd.DataFrame(rows).sort_values(["variant"])
    metrics_frame.to_csv(runs_dir / "round_d_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "round_d_yearly_nav.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(runs_dir / "round_d_entry_weight_buckets.csv", index=False)
    comparison = write_comparison(metrics_frame, reference, runs_dir / "round_d_comparison.csv")
    write_report(comparison, reference, reports_dir / "MARKET_STATE_RISK_ROUND_D.md")
    print(f"[round_d] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def load_reference_metrics() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, run_dir in REFERENCE_RUNS.items():
        summary = read_json(run_dir / "summary.json")
        metrics = read_json(run_dir / "metrics.json")
        row = {**summary, **metrics}
        nav = read_json_frame(run_dir / "daily_nav.json")
        row.update(position_stats(nav, "full"))
        post2022 = nav[pd.to_datetime(nav["date"]) >= pd.Timestamp("2022-01-01")] if not nav.empty and "date" in nav else nav
        row.update(position_stats(post2022, "post2022"))
        row.update(window_return_stats(nav, "2022-03-31", "2022-05-05", "crash20"))
        row.update(exposure_summary(run_dir, "full"))
        row.update(exposure_summary(run_dir, "y2022", start="2022-01-01", end="2022-12-31"))
        result[name] = row
    return result


def position_stats(nav: pd.DataFrame, prefix: str) -> dict[str, float]:
    if nav.empty or "position_count" not in nav:
        return {f"{prefix}_avg_position_count": np.nan, f"{prefix}_zero_position_ratio": np.nan, f"{prefix}_max_position_count": np.nan}
    counts = pd.to_numeric(nav["position_count"], errors="coerce").fillna(0.0)
    return {f"{prefix}_avg_position_count": float(counts.mean()), f"{prefix}_zero_position_ratio": float((counts <= 0).mean()), f"{prefix}_max_position_count": float(counts.max())}


def window_return_stats(nav: pd.DataFrame, start: str, end: str, prefix: str) -> dict[str, float]:
    if nav.empty or "date" not in nav or "daily_return" not in nav:
        return {f"{prefix}_return": np.nan, f"{prefix}_mdd": np.nan, f"{prefix}_avg_position_count": np.nan}
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].copy()
    if window.empty:
        return {f"{prefix}_return": np.nan, f"{prefix}_mdd": np.nan, f"{prefix}_avg_position_count": np.nan}
    returns = pd.to_numeric(window["daily_return"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(window["total_value"], errors="coerce") if "total_value" in window else (1 + returns).cumprod()
    drawdown = values / values.cummax() - 1
    return {f"{prefix}_return": compound_return(returns), f"{prefix}_mdd": float(drawdown.min()), f"{prefix}_avg_position_count": float(pd.to_numeric(window.get("position_count"), errors="coerce").fillna(0.0).mean()) if "position_count" in window else np.nan}


def exposure_summary(run_dir: Path, prefix: str, start: str | None = None, end: str | None = None) -> dict[str, float]:
    trades = read_json_frame(run_dir / "trades.json")
    nav = read_json_frame(run_dir / "daily_nav.json")
    if trades.empty or nav.empty:
        return {f"{prefix}_buy_count": np.nan, f"{prefix}_max_buy_weight": np.nan, f"{prefix}_avg_buy_weight": np.nan, f"{prefix}_buy_weight_gt20_ratio": np.nan, f"{prefix}_buy_weight_gt50_ratio": np.nan}
    buys = trades[trades["action"].astype(str).str.upper() == "BUY"].copy()
    buys["date"] = pd.to_datetime(buys["date"])
    if start:
        buys = buys[buys["date"] >= pd.Timestamp(start)]
    if end:
        buys = buys[buys["date"] <= pd.Timestamp(end)]
    if buys.empty:
        return {f"{prefix}_buy_count": 0, f"{prefix}_max_buy_weight": np.nan, f"{prefix}_avg_buy_weight": np.nan, f"{prefix}_buy_weight_gt20_ratio": np.nan, f"{prefix}_buy_weight_gt50_ratio": np.nan}
    nav2 = nav.copy()
    nav2["date"] = pd.to_datetime(nav2["date"])
    buys = buys.merge(nav2[["date", "total_value"]], on="date", how="left")
    buys["buy_weight"] = pd.to_numeric(buys["trade_value"], errors="coerce") / pd.to_numeric(buys["total_value"], errors="coerce")
    weights = buys["buy_weight"].replace([np.inf, -np.inf], np.nan).dropna()
    return {f"{prefix}_buy_count": int(len(buys)), f"{prefix}_max_buy_weight": float(weights.max()) if not weights.empty else np.nan, f"{prefix}_avg_buy_weight": float(weights.mean()) if not weights.empty else np.nan, f"{prefix}_buy_weight_gt20_ratio": float((weights > 0.20).mean()) if not weights.empty else np.nan, f"{prefix}_buy_weight_gt50_ratio": float((weights > 0.50).mean()) if not weights.empty else np.nan}


def exposure_buckets(run_dir: Path, run_id: str) -> list[dict[str, Any]]:
    trades = read_json_frame(run_dir / "trades.json")
    nav = read_json_frame(run_dir / "daily_nav.json")
    if trades.empty or nav.empty:
        return []
    buys = trades[trades["action"].astype(str).str.upper() == "BUY"].copy()
    buys["date"] = pd.to_datetime(buys["date"])
    nav["date"] = pd.to_datetime(nav["date"])
    buys = buys.merge(nav[["date", "total_value"]], on="date", how="left")
    buys["buy_weight"] = pd.to_numeric(buys["trade_value"], errors="coerce") / pd.to_numeric(buys["total_value"], errors="coerce")
    buys["period"] = np.where(buys["date"].dt.year == 2022, "2022", "full_ex_2022")
    buys["bucket"] = pd.cut(buys["buy_weight"], [0, 0.05, 0.0834, 0.12, 0.20, 1.01], labels=["<=5%", "5-8.3%", "8.3-12%", "12-20%", ">20%"])
    rows: list[dict[str, Any]] = []
    for (period, bucket), group in buys.groupby(["period", "bucket"], observed=True):
        rows.append({"run_id": run_id, "period": period, "bucket": str(bucket), "buy_count": int(len(group)), "avg_buy_weight": float(group["buy_weight"].mean()), "max_buy_weight": float(group["buy_weight"].max())})
    return rows


def summarize_yearly_nav(run_id: str, nav: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        values = pd.to_numeric(group["total_value"], errors="coerce")
        returns = pd.to_numeric(group["daily_return"], errors="coerce").fillna(0.0)
        drawdown = values / values.cummax() - 1.0
        rows.append({"run_id": run_id, "variant": metadata.get("variant"), "year": int(year), "year_return": compound_return(returns), "year_mdd": float(drawdown.min()), "avg_position_count": float(pd.to_numeric(group.get("position_count"), errors="coerce").fillna(0.0).mean()), "zero_position_ratio": float((pd.to_numeric(group.get("position_count"), errors="coerce").fillna(0.0) <= 0).mean())})
    return rows


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
    comparison = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)
    return comparison


def write_pre_report(signals: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Market-State Risk Round D Pre-Backtest", "", "保持原弱转强 raw top12 alpha，验证 sizing/rank_weights/止损/轻状态降权。", "", "| variant | sizing | topk | pos | rows | days | days>=topk | stop | rank_weights |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for signal in signals:
        lines.append(f"| {signal['variant']} | {signal['sizing']} | {signal['topk']} | {signal['max_positions']} | {signal['signal_rows']} | {signal['signal_days']} | {signal['days_with_at_least_topk']} | {signal.get('stop_loss')} | {signal.get('rank_weights')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(comparison: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> None:
    lines = ["# Market-State Risk Round D", "", "范围：原弱转强候选；保持 `selector.lag=1`，第二天 `close` 买入；本轮只验证仓位和风险覆盖。", "", "## Baselines", "", "| baseline | total_return | max_drawdown | sharpe | avg_pos | post2022_zero | 2022 max buy weight |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in reference.items():
        lines.append(f"| {name} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | {safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | {safe_float(row.get('post2022_zero_position_ratio')):.4f} | {safe_float(row.get('y2022_max_buy_weight')):.4f} |")
    lines.extend(["", "## Results", "", "| variant | sizing | total_return | max_drawdown | sharpe | avg_pos | post2022_zero | crash20_ret | 2022 max_buy_w | 2022 gt20 | vs_raw12_ret | vs_raw12_mdd | vs_raw12_sharpe |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for _, row in comparison.iterrows():
        lines.append(f"| {row.get('variant')} | {row.get('sizing')} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | {safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | {safe_float(row.get('post2022_zero_position_ratio')):.4f} | {safe_float(row.get('crash20_return')):.4f} | {safe_float(row.get('y2022_max_buy_weight')):.4f} | {safe_float(row.get('y2022_buy_weight_gt20_ratio')):.4f} | {safe_float(row.get('vs_raw12_total_return_delta')):.4f} | {safe_float(row.get('vs_raw12_mdd_delta')):.4f} | {safe_float(row.get('vs_raw12_sharpe_delta')):.4f} |")
    lines.extend(["", "## Artifacts", "", "- `runs/round_d_comparison.csv`", "- `runs/round_d_yearly_nav.csv`", "- `runs/round_d_entry_weight_buckets.csv`", "- `runs/round_d_metrics_full.csv`"]) 
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


def compound_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float((1 + series.astype(float)).prod() - 1)


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
