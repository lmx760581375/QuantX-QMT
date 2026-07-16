"""Capacity extension backtests for the real weak-to-strong candidate body.

Round A intentionally does not widen the weak-to-strong definition. It replays
the original raw daily candidates through external_score, preserves lag=1 and
close execution, then sweeps TopN/max_positions to map the natural capacity
curve.

Round B-A keeps only the original front-rank candidates (raw_rank <= 4) and
sweeps max_positions, isolating whether the capacity alpha comes from releasing
more room for the core weak-to-strong signal.
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
PROVIDER_URI = "data/qlib_data_fixed"
BACKTEST_START = "2016-01-04"

SOURCE_CONFIG = Path("configs/strategies/generated/weak_to_strong_fast_exit_grid/fast_exit_d18_p2_t8.yaml")
SOURCE_RUN = Path(
    "tmp/market_regime_weak_to_strong_alpha_v1/capacity_extension_round_a/"
    "reference_artifacts/current_precomputed_top6_pos6"
)
SOURCE_CANDIDATES = SOURCE_RUN / "daily_selection_candidates.json"

VARIANTS = (
    {"variant": "raw_top4_pos5", "topk": 4, "max_positions": 5, "baseline": "current_precomputed_top4_pos5"},
    {"variant": "raw_top5_pos5", "topk": 5, "max_positions": 5, "baseline": None},
    {"variant": "raw_top6_pos6", "topk": 6, "max_positions": 6, "baseline": "current_precomputed_top6_pos6"},
    {"variant": "raw_top8_pos8", "topk": 8, "max_positions": 8, "baseline": None},
    {"variant": "raw_top10_pos10", "topk": 10, "max_positions": 10, "baseline": None},
    {"variant": "raw_top12_pos12", "topk": 12, "max_positions": 12, "baseline": None},
)

EXPERIMENTS = {
    "round_a_current": {
        "output_name": "capacity_extension_round_a_current",
        "title": "原弱转强容量扩展 Round A",
        "report_title": "容量扩展 Round A",
        "description": (
            "Tmp experiment: replay original weak-to-strong raw candidates through external_score "
            "and sweep TopN/max_positions while preserving lag=1, close execution, sizing, and sell rules."
        ),
        "variants": VARIANTS,
        "max_raw_rank": None,
        "uses_raw_candidates": True,
        "signal_filename": "true_wts_raw_candidates.parquet",
    },
    "core_capacity_round_b_a": {
        "output_name": "capacity_extension_round_b_core4",
        "title": "原弱转强核心容量释放 Round B-A",
        "report_title": "核心容量释放 Round B-A",
        "description": (
            "Tmp experiment: keep only original weak-to-strong raw_rank <= 4 candidates, "
            "then sweep max_positions while preserving lag=1, close execution, sizing, and sell rules."
        ),
        "variants": (
            {"variant": "core4_pos5", "topk": 4, "max_positions": 5, "baseline": "current_precomputed_top4_pos5"},
            {"variant": "core4_pos6", "topk": 4, "max_positions": 6, "baseline": None},
            {"variant": "core4_pos8", "topk": 4, "max_positions": 8, "baseline": None},
            {"variant": "core4_pos10", "topk": 4, "max_positions": 10, "baseline": None},
            {"variant": "core4_pos12", "topk": 4, "max_positions": 12, "baseline": None},
        ),
        "max_raw_rank": 4,
        "uses_raw_candidates": False,
        "signal_filename": "true_wts_core4_candidates.parquet",
    },
}

REFERENCE_RUNS = {
    "current_precomputed_top4_pos5": Path(
        "tmp/market_regime_weak_to_strong_alpha_v1/capacity_extension_round_a/"
        "reference_artifacts/current_precomputed_top4_pos5"
    ),
    "current_precomputed_top6_pos6": Path(
        "tmp/market_regime_weak_to_strong_alpha_v1/capacity_extension_round_a/"
        "reference_artifacts/current_precomputed_top6_pos6"
    ),
    "historical_top4_pos5": Path("runs/weak_to_strong_xqshare_20260711/28_fast_exit_d18_p2_t8"),
    "historical_top6_pos6": Path(
        "runs/weak_to_strong_xqshare_20260711/68_weak_to_strong_original_shape_pos6_topk6_2016_2026_mainboard"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="round_a_current",
        help="Experiment variant to build/run.",
    )
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

    experiment = EXPERIMENTS[args.experiment]
    root = Path(args.root)
    output_root = root / str(experiment["output_name"])
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        raw_candidates = load_raw_candidate_signals(SOURCE_CANDIDATES)
        signals = filter_signals(raw_candidates, experiment)
        signal_path = signals_dir / str(experiment["signal_filename"])
        signals.to_parquet(signal_path, index=False)

        manifest_rows = build_signal_manifest(signals, signal_path, experiment)
        configs = build_configs(manifest_rows, configs_dir, experiment)
        manifest = {
            "experiment": args.experiment,
            "output_name": experiment["output_name"],
            "source_config": str(SOURCE_CONFIG),
            "source_run": str(SOURCE_RUN),
            "source_candidates": str(SOURCE_CANDIDATES),
            "signal_path": str(signal_path),
            "backtest_start": BACKTEST_START,
            "execution_lag": 1,
            "deal_price": "close",
            "max_raw_rank": experiment.get("max_raw_rank"),
            "variants": manifest_rows,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.build_only:
            print(f"[{args.experiment}] build-only signals={len(signals)} configs={len(configs)}", flush=True)
            return

        summaries = run_configs(
            configs,
            artifacts_dir,
            label=args.experiment,
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

    collect_artifacts(artifacts_dir, runs_dir, reports_dir, experiment)


def load_raw_candidate_signals(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for day in data:
        signal_time = str(day.get("signal_date") or day.get("date"))
        for rank, candidate in enumerate(day.get("raw_candidates") or [], start=1):
            score = candidate.get("score")
            rows.append({
                "signal_time": signal_time,
                "instrument": str(candidate.get("symbol")),
                "score": float(score) if score is not None else float(-rank),
                "raw_rank": int(rank),
            })
    frame = pd.DataFrame(rows, columns=["signal_time", "instrument", "score", "raw_rank"])
    if frame.empty:
        raise ValueError(f"No raw candidates found in {path}")
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values(["signal_time", "raw_rank"]).reset_index(drop=True)


def filter_signals(raw_candidates: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    max_raw_rank = experiment.get("max_raw_rank")
    if max_raw_rank is None:
        return raw_candidates.copy()
    filtered = raw_candidates[raw_candidates["raw_rank"] <= int(max_raw_rank)].copy()
    if filtered.empty:
        raise ValueError(f"No candidates remain after max_raw_rank={max_raw_rank}")
    return filtered.reset_index(drop=True)


def build_signal_manifest(
    raw_candidates: pd.DataFrame,
    signal_path: Path,
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signal_days = raw_candidates.groupby("signal_time", sort=True).size()
    for spec in experiment["variants"]:
        topk = int(spec["topk"])
        rows.append({
            "variant": str(spec["variant"]),
            "path": str(signal_path),
            "topk": topk,
            "max_positions": int(spec["max_positions"]),
            "baseline": spec["baseline"],
            "signal_rows": int(len(raw_candidates)),
            "signal_days": int(signal_days.size),
            "days_with_at_least_topk": int((signal_days >= topk).sum()),
            "first_signal": str(raw_candidates["signal_time"].min()),
            "last_signal": str(raw_candidates["signal_time"].max()),
            "max_raw_rank": experiment.get("max_raw_rank"),
            "output_name": experiment["output_name"],
            "report_title": experiment["report_title"],
        })
    return rows


def build_configs(manifest_rows: list[dict[str, Any]], configs_dir: Path, experiment: dict[str, Any]) -> list[Path]:
    source = read_yaml(SOURCE_CONFIG)
    paths: list[Path] = []
    for row in manifest_rows:
        config = make_external_score_config(source, row, experiment)
        path = configs_dir / f"{row['variant']}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def make_external_score_config(source: dict[str, Any], signal: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config["name"] = f"true_wts_{experiment['output_name']}_{signal['variant']}"
    config["title"] = str(experiment["title"])
    config["description"] = str(experiment["description"])
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
        "candidate_limit": max(20, int(signal["topk"])),
        "reason": f"true_wts_{experiment['output_name']}_{signal['variant']}",
    }
    config["rebalance"] = dict(source.get("rebalance") or {})
    config["rebalance"]["max_positions"] = int(signal["max_positions"])
    config["execution"] = dict(source.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["engine"] = dict(source.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": experiment["output_name"],
        "variant": signal["variant"],
        "baseline": signal.get("baseline"),
        "source_config": str(SOURCE_CONFIG),
        "source_run": str(SOURCE_RUN),
        "source_candidates": str(SOURCE_CANDIDATES),
        "uses_raw_candidates": bool(experiment.get("uses_raw_candidates")),
        "max_raw_rank": signal.get("max_raw_rank"),
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal["signal_rows"]),
        "signal_days": int(signal["signal_days"]),
        "days_with_at_least_topk": int(signal["days_with_at_least_topk"]),
    }
    return config


def run_configs(
    configs: list[Path],
    artifacts_dir: Path,
    *,
    label: str,
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
            print(f"[{label}] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[{label}] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
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


def collect_artifacts(
    artifacts_dir: Path,
    runs_dir: Path,
    reports_dir: Path,
    experiment: dict[str, Any],
) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[{experiment['output_name']}] no artifacts found under {artifacts_dir}", flush=True)
        return

    reference = load_reference_metrics()
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
        row.update(position_stats(nav, prefix="full"))
        row.update(position_stats(nav[nav["date"].astype(str) >= "2022-01-01"] if not nav.empty else nav, prefix="post2022"))
        rows.append(row)
        yearly_rows.extend(summarize_yearly_nav(run_id, nav, metadata))

    metrics_frame = pd.DataFrame(rows).sort_values(["topk", "max_positions", "variant"])
    metrics_frame.to_csv(runs_dir / "capacity_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "capacity_yearly_nav.csv", index=False)
    comparison = write_comparison(metrics_frame, reference, runs_dir / "capacity_comparison.csv")
    report_name = f"{str(experiment['output_name']).upper()}.md"
    write_report(metrics_frame, comparison, reference, reports_dir / report_name, experiment)
    print(f"[{experiment['output_name']}] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def load_reference_metrics() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, run_dir in REFERENCE_RUNS.items():
        summary = read_json(run_dir / "summary.json")
        metrics = read_json(run_dir / "metrics.json")
        row = {**summary, **metrics}
        nav = read_json_frame(run_dir / "daily_nav.json")
        row.update(position_stats(nav, prefix="full"))
        row.update(position_stats(nav[nav["date"].astype(str) >= "2022-01-01"] if not nav.empty else nav, prefix="post2022"))
        result[name] = row
    return result


def position_stats(nav: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if nav.empty or "position_count" not in nav:
        return {
            f"{prefix}_avg_position_count": np.nan,
            f"{prefix}_zero_position_ratio": np.nan,
            f"{prefix}_max_position_count": np.nan,
        }
    counts = pd.to_numeric(nav["position_count"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_avg_position_count": float(counts.mean()),
        f"{prefix}_zero_position_ratio": float((counts <= 0).mean()),
        f"{prefix}_max_position_count": float(counts.max()),
    }


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
        rows.append({
            "run_id": run_id,
            "variant": metadata.get("variant"),
            "topk": metadata.get("topk"),
            "max_positions": metadata.get("max_positions"),
            "year": int(year),
            "year_return": compound_return(returns),
            "year_mdd": float(drawdown.min()) if not drawdown.empty else np.nan,
            "avg_position_count": float(group["position_count"].fillna(0.0).mean()) if "position_count" in group else np.nan,
            "zero_position_ratio": float((group["position_count"].fillna(0.0) <= 0).mean()) if "position_count" in group else np.nan,
        })
    return rows


def write_comparison(metrics: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_ref = reference.get("current_precomputed_top4_pos5", {})
    for _, row in metrics.iterrows():
        out = row.to_dict()
        baseline_name = row.get("baseline") or "current_precomputed_top4_pos5"
        direct_ref = reference.get(str(baseline_name), {})
        for prefix, ref in (("vs_top4", primary_ref), ("vs_direct_baseline", direct_ref)):
            out[f"{prefix}_total_return_delta"] = safe_float(row.get("total_return")) - safe_float(ref.get("total_return"))
            out[f"{prefix}_mdd_delta"] = safe_float(row.get("max_drawdown")) - safe_float(ref.get("max_drawdown"))
            out[f"{prefix}_sharpe_delta"] = safe_float(row.get("sharpe")) - safe_float(ref.get("sharpe"))
            out[f"{prefix}_avg_position_delta"] = safe_float(row.get("full_avg_position_count")) - safe_float(ref.get("full_avg_position_count"))
            out[f"{prefix}_post2022_avg_position_delta"] = safe_float(row.get("post2022_avg_position_count")) - safe_float(ref.get("post2022_avg_position_count"))
        rows.append(out)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(path, index=False)
    return comparison


def write_report(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    reference: dict[str, dict[str, Any]],
    path: Path,
    experiment: dict[str, Any],
) -> None:
    lines = [
        f"# {experiment['report_title']}",
        "",
        "范围：回放原弱转强候选；保持 `selector.lag=1` 和 `execution.deal_price=close`。",
        "",
        f"候选过滤：`raw_rank <= {experiment['max_raw_rank']}`。" if experiment.get("max_raw_rank") else "候选过滤：无，使用完整 `raw_candidates`。",
        "",
        "## 原策略参考",
        "",
        "| baseline | total_return | max_drawdown | sharpe | avg_pos | post2022_avg_pos | post2022_zero_pos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in reference.items():
        lines.append(
            "| {name} | {ret:.4f} | {mdd:.4f} | {sharpe:.4f} | {avg:.4f} | {avg22:.4f} | {zero22:.4f} |".format(
                name=name,
                ret=safe_float(row.get("total_return")),
                mdd=safe_float(row.get("max_drawdown")),
                sharpe=safe_float(row.get("sharpe")),
                avg=safe_float(row.get("full_avg_position_count")),
                avg22=safe_float(row.get("post2022_avg_position_count")),
                zero22=safe_float(row.get("post2022_zero_position_ratio")),
            )
        )
    lines.extend([
        "",
        "## 实验结果",
        "",
        "| variant | topk | pos | total_return | max_drawdown | sharpe | profit_factor | avg_pos | post2022_avg_pos | post2022_zero_pos | vs_top4_ret_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in comparison.sort_values(["topk", "max_positions"]).iterrows():
        lines.append(
            "| {variant} | {topk:.0f} | {pos:.0f} | {ret:.4f} | {mdd:.4f} | {sharpe:.4f} | {pf:.4f} | {avg:.4f} | {avg22:.4f} | {zero22:.4f} | {delta:.4f} |".format(
                variant=row.get("variant"),
                topk=safe_float(row.get("topk")),
                pos=safe_float(row.get("max_positions")),
                ret=safe_float(row.get("total_return")),
                mdd=safe_float(row.get("max_drawdown")),
                sharpe=safe_float(row.get("sharpe")),
                pf=safe_float(row.get("profit_factor")),
                avg=safe_float(row.get("full_avg_position_count")),
                avg22=safe_float(row.get("post2022_avg_position_count")),
                zero22=safe_float(row.get("post2022_zero_position_ratio")),
                delta=safe_float(row.get("vs_top4_total_return_delta")),
            )
        )
    lines.extend([
        "",
        "## 产物",
        "",
        "- `runs/capacity_metrics_full.csv`",
        "- `runs/capacity_comparison.csv`",
        "- `runs/capacity_yearly_nav.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
