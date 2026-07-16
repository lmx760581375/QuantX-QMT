"""Formal gate backtests for the real weak-to-strong strategy body.

This tmp-only experiment replays the original weak-to-strong daily candidates via
the formal external_score selector, then filters signal dates with the market
gate mined in the earlier rounds. It keeps the original lag=1, close execution,
position sizing, and sell rules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "true_wts_gate_backtest"
PROVIDER_URI = "data/qlib_data_fixed"
GATE_FEATURE = "p_bad_top20_max"
GATE_DIRECTION = "le"
QUANTILES = (0.30, 0.35, 0.40, 0.50)
BACKTEST_START = "2022-01-04"

BASE_STRATEGIES = {
    "fast_exit_top4_pos5": {
        "source_config": "configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t8.yaml",
        "source_run": "runs/weak_to_strong_xqshare_20260711/28_fast_exit_d18_p2_t8",
        "topk": 4,
        "max_positions": 5,
    },
    "original_top6_pos6": {
        "source_config": "configs/strategies/generated/weak_to_strong_score_capacity/weak_to_strong_original_shape_pos6_topk6_2016_2026_mainboard.yaml",
        "source_run": "runs/weak_to_strong_xqshare_20260711/68_weak_to_strong_original_shape_pos6_topk6_2016_2026_mainboard",
        "topk": 6,
        "max_positions": 6,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--build-only", action="store_true", help="Only generate signals/configs.")
    parser.add_argument("--dry-run", action="store_true", help="Compile configs through run_backtest --dry-run.")
    parser.add_argument("--analyze-only", action="store_true", help="Only collect existing artifacts.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing summary.json.")
    parser.add_argument(
        "--python-cmd",
        nargs="+",
        default=["conda", "run", "--no-capture-output", "-n", "test", "python"],
        help="Python command prefix used to invoke quantx.tools.run_backtest.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / OUTPUT_NAME
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        gate_sets = build_gate_sets(root / "runs" / "gate_daily_panel.csv")
        signal_manifest = build_signals(gate_sets, signals_dir)
        configs = build_configs(signal_manifest, configs_dir)
        manifest = {
            "gate_feature": GATE_FEATURE,
            "gate_direction": GATE_DIRECTION,
            "quantiles": list(QUANTILES),
            "backtest_start": BACKTEST_START,
            "strategies": BASE_STRATEGIES,
            "signals": signal_manifest,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.build_only:
            print(f"[true-wts-gate] build-only signals={len(signal_manifest)} configs={len(configs)}", flush=True)
            return

        summaries = run_configs(
            configs,
            artifacts_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            python_cmd=args.python_cmd,
        )
        pd.DataFrame(summaries).to_csv(
            runs_dir / ("dry_run_summary.csv" if args.dry_run else "backtest_summary.csv"),
            index=False,
        )
        if args.dry_run:
            return

    collect_artifacts(artifacts_dir, runs_dir)


def build_gate_sets(gate_path: Path) -> dict[str, dict[str, set[str]]]:
    daily = pd.read_csv(gate_path)
    daily["signal_time"] = pd.to_datetime(daily["signal_time"]).dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("signal_time").reset_index(drop=True)
    available_days = set(daily["signal_time"].astype(str))
    result: dict[str, dict[str, set[str]]] = {}
    for q in QUANTILES:
        pass_days: set[str] = set()
        for pred_year in range(2022, 2027):
            train = daily[daily["year"] < pred_year]
            test = daily[daily["year"] == pred_year]
            if train.empty or test.empty:
                continue
            threshold = float(train[GATE_FEATURE].quantile(q))
            if GATE_DIRECTION == "le":
                keep = test[GATE_FEATURE] <= threshold
            elif GATE_DIRECTION == "ge":
                keep = test[GATE_FEATURE] >= threshold
            else:
                raise ValueError(f"Unsupported gate direction: {GATE_DIRECTION}")
            pass_days.update(test.loc[keep, "signal_time"].astype(str).tolist())
        result[f"q{int(q * 100):02d}"] = {"available": available_days, "pass": pass_days}
    return result


def build_signals(gate_sets: dict[str, dict[str, set[str]]], signals_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_name, spec in BASE_STRATEGIES.items():
        candidates = load_candidate_signals(Path(spec["source_run"]) / "daily_selection_candidates.json")
        variants = [("baseline_replay", candidates)]
        for q_name, days in gate_sets.items():
            available_days = days["available"]
            pass_days = days["pass"]
            variants.append((f"gate_{q_name}_missing_pass", candidates[
                candidates["signal_time"].map(lambda day: day not in available_days or day in pass_days)
            ]))
            variants.append((f"gate_{q_name}_missing_reject", candidates[candidates["signal_time"].isin(pass_days)]))

        for variant_name, frame in variants:
            path = signals_dir / f"{strategy_name}_{variant_name}.parquet"
            frame.to_parquet(path, index=False)
            rows.append({
                "strategy": strategy_name,
                "variant": variant_name,
                "path": str(path),
                "topk": int(spec["topk"]),
                "max_positions": int(spec["max_positions"]),
                "source_config": str(spec["source_config"]),
                "source_run": str(spec["source_run"]),
                "signal_rows": int(len(frame)),
                "signal_days": int(frame["signal_time"].nunique()) if not frame.empty else 0,
                "first_signal": frame["signal_time"].min() if not frame.empty else None,
                "last_signal": frame["signal_time"].max() if not frame.empty else None,
            })
    return rows


def load_candidate_signals(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for day in data:
        signal_time = str(day.get("signal_date") or day.get("date"))
        for rank, candidate in enumerate(day.get("selected_candidates") or [], start=1):
            score = candidate.get("score")
            rows.append({
                "signal_time": signal_time,
                "instrument": str(candidate.get("symbol")),
                "score": float(score) if score is not None else float(-rank),
                "rank": rank,
            })
    frame = pd.DataFrame(rows, columns=["signal_time", "instrument", "score", "rank"])
    if frame.empty:
        return frame
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values(["signal_time", "rank"]).reset_index(drop=True)


def build_configs(signal_manifest: list[dict[str, Any]], configs_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for signal in signal_manifest:
        source = read_yaml(Path(signal["source_config"]))
        config = make_external_score_config(source, signal)
        path = configs_dir / f"{signal['strategy']}_{signal['variant']}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def make_external_score_config(source: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config["name"] = f"true_wts_{signal['strategy']}_{signal['variant']}"
    config["title"] = "原弱转强市场 gate 正式回测"
    config["description"] = (
        "Tmp experiment: replay original weak-to-strong selected candidates through external_score, "
        "then apply date-level market gate while preserving lag=1, close execution, and sell rules."
    )
    config["data"] = dict(source.get("data") or {})
    config["data"].update({
        "provider_uri": PROVIDER_URI,
        "universe": "external_score",
        "start": BACKTEST_START,
        "end": "latest",
    })
    config["selector"] = {
        "mode": "external_score",
        "path": signal["path"],
        "date_col": "signal_time",
        "instrument_col": "instrument",
        "score_col": "score",
        "sort": "score_desc",
        "topk": int(signal["topk"]),
        "lag": 1,
        "candidate_limit": int(signal["topk"]),
        "reason": f"true_wts_gate_{signal['strategy']}_{signal['variant']}",
    }
    config["rebalance"] = dict(source.get("rebalance") or {})
    config["rebalance"]["max_positions"] = int(signal["max_positions"])
    config["execution"] = dict(source.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["engine"] = dict(source.get("engine") or {})
    config["engine"].update({
        "deal_price": "close",
        "max_workers": 1,
    })
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "base_strategy": signal["strategy"],
        "variant": signal["variant"],
        "source_config": signal["source_config"],
        "source_run": signal["source_run"],
        "gate_feature": GATE_FEATURE if signal["variant"].startswith("gate_") else None,
        "gate_direction": GATE_DIRECTION if signal["variant"].startswith("gate_") else None,
        "gate_quantile": parse_quantile(signal["variant"]),
        "missing_gate_policy": parse_missing_policy(signal["variant"]),
        "signal_rows": signal["signal_rows"],
        "signal_days": signal["signal_days"],
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "execution_lag": 1,
        "deal_price": "close",
    }
    return config


def parse_quantile(variant: str) -> float | None:
    if not variant.startswith("gate_q"):
        return None
    return int(variant.split("_", 2)[1][1:]) / 100.0


def parse_missing_policy(variant: str) -> str | None:
    if variant.endswith("missing_pass"):
        return "pass"
    if variant.endswith("missing_reject"):
        return "reject"
    return None


def run_configs(
    configs: list[Path],
    artifacts_dir: Path,
    *,
    dry_run: bool,
    skip_existing: bool,
    python_cmd: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[true-wts-gate] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[true-wts-gate] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
        cmd = [*python_cmd, "-m", "quantx.tools.run_backtest", "--config", str(config_path), "--json"]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--output-dir", str(artifacts_dir), "--run-id", run_id])
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        row = parse_run_output(completed.stdout)
        row.update({
            "config": str(config_path),
            "run_id": run_id,
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        })
        rows.append(flatten_summary(row))
        if completed.returncode != 0:
            print(completed.stderr[-4000:], flush=True)
            raise RuntimeError(f"Backtest failed for {config_path}")
    return rows


def parse_run_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {"raw_stdout": text[-2000:]}
    return json.loads(text[start:])


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (np.integer, np.floating)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


def collect_artifacts(artifacts_dir: Path, runs_dir: Path) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[true-wts-gate] no artifacts found under {artifacts_dir}", flush=True)
        return
    rows = []
    yearly_rows = []
    for artifact_dir in artifact_dirs:
        run_id = artifact_dir.name
        summary = read_json(artifact_dir / "summary.json")
        metrics = read_json(artifact_dir / "metrics.json")
        config = read_yaml(artifact_dir / "config.yaml")
        metadata = dict(config.get("metadata") or {})
        nav = read_json_frame(artifact_dir / "daily_nav.json")
        row = {"run_id": run_id}
        for key in (
            "name",
            "symbols",
            "start_date",
            "end_date",
            "final_value",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "buy_count",
            "sell_count",
            "reject_count",
        ):
            row[key] = summary.get(key)
        for key in (
            "closed_position_count",
            "win_rate",
            "profit_factor",
            "avg_closed_return",
            "avg_holding_days",
            "avg_position_count",
            "max_position_count",
            "avg_capital_utilization",
            "zero_utilization_day_ratio",
        ):
            row[key] = metrics.get(key)
        row.update(metadata)
        rows.append(row)
        yearly_rows.extend(summarize_yearly_nav(run_id, nav, metadata))
    metrics_frame = pd.DataFrame(rows).sort_values(["base_strategy", "variant"])
    metrics_frame.to_csv(runs_dir / "metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "yearly_nav.csv", index=False)
    write_comparison(metrics_frame, runs_dir / "comparison_vs_baseline.csv")
    print(f"[true-wts-gate] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def summarize_yearly_nav(run_id: str, nav: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        values = group["total_value"].astype(float)
        returns = group["daily_return"].fillna(0.0).astype(float)
        drawdown = values / values.cummax() - 1.0
        row = {
            "run_id": run_id,
            "year": int(year),
            "year_return": compound_return(returns),
            "year_mdd": float(drawdown.min()) if not drawdown.empty else np.nan,
            "avg_position_count": float(group["position_count"].fillna(0.0).mean()) if "position_count" in group else np.nan,
        }
        row.update({
            "base_strategy": metadata.get("base_strategy"),
            "variant": metadata.get("variant"),
            "gate_quantile": metadata.get("gate_quantile"),
            "missing_gate_policy": metadata.get("missing_gate_policy"),
        })
        rows.append(row)
    return rows


def write_comparison(metrics: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for base_strategy, group in metrics.groupby("base_strategy", sort=True):
        baseline_rows = group[group["variant"] == "baseline_replay"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, row in group.iterrows():
            out = row.to_dict()
            out["baseline_total_return"] = baseline.get("total_return")
            out["excess_total_return"] = safe_float(row.get("total_return")) - safe_float(baseline.get("total_return"))
            out["mdd_delta"] = safe_float(row.get("max_drawdown")) - safe_float(baseline.get("max_drawdown"))
            out["profit_factor_delta"] = safe_float(row.get("profit_factor")) - safe_float(baseline.get("profit_factor"))
            out["sharpe_delta"] = safe_float(row.get("sharpe")) - safe_float(baseline.get("sharpe"))
            rows.append(out)
    pd.DataFrame(rows).to_csv(path, index=False)


def compound_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float((1.0 + series.astype(float)).prod() - 1.0)


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
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
