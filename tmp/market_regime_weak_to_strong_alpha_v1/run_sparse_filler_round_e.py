"""Round E sparse-day filler validation for weak-to-strong capacity alpha.

This temporary research script keeps the original weak-to-strong raw top12
signals as the priority alpha, then adds buyable/non-ST filler candidates only
on sparse raw-candidate days. Formal validation is still delegated to
``quantx.tools.run_backtest`` so the results stay comparable with Round A/C/D.
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
OUTPUT_NAME = "sparse_filler_round_e"
PROVIDER_URI = "data/qlib_data_fixed"
BACKTEST_START = "2016-01-04"

SOURCE_CONFIG = ROOT / "capacity_extension_round_a_current/configs/raw_top12_pos12.yaml"
RAW_CANDIDATES = ROOT / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet"
CANDIDATE_PANEL = ROOT / "data/candidate_panel.parquet"
ROUND_C_SCRIPT = ROOT / "run_market_state_ranking_round_c.py"
ROUND_D_SCRIPT = ROOT / "run_market_state_risk_round_d.py"


VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "e1_raw12_sparse1_shape4_mid4_cash",
        "description": "raw top12 priority; when raw_count<=1, fill with shape top4 and mid top4.",
        "sparse_max_raw_count": 1,
        "families": (("shape", 4), ("mid", 4)),
        "filler_condition": "all_sparse",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "e2_raw12_sparse2_shape4_mid4_cash",
        "description": "raw top12 priority; when raw_count<=2, fill with shape top4 and mid top4.",
        "sparse_max_raw_count": 2,
        "families": (("shape", 4), ("mid", 4)),
        "filler_condition": "all_sparse",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "e3_raw12_sparse2_shape4_wide4_cash",
        "description": "raw top12 priority; when raw_count<=2, fill with shape top4 and wide top4.",
        "sparse_max_raw_count": 2,
        "families": (("shape", 4), ("wide", 4)),
        "filler_condition": "all_sparse",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "e4_raw12_sparse2_shape8_mid8_cash",
        "description": "raw top12 priority; when raw_count<=2, fill with broader shape top8 and mid top8.",
        "sparse_max_raw_count": 2,
        "families": (("shape", 8), ("mid", 8)),
        "filler_condition": "all_sparse",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "e5_raw12_sparse2_shape4_mid4_no_bear_breadthlo_cash",
        "description": "same as e2, but disable filler on bear+breadth_lo sparse days.",
        "sparse_max_raw_count": 2,
        "families": (("shape", 4), ("mid", 4)),
        "filler_condition": "not_bear_breadth_lo",
        "topk": 12,
        "max_positions": 12,
    },
    {
        "variant": "e6_raw12_sparse2_shape4_mid4_only_bear_breadthlo_cash",
        "description": "raw top12 priority; add filler only on bear+breadth_lo sparse days.",
        "sparse_max_raw_count": 2,
        "families": (("shape", 4), ("mid", 4)),
        "filler_condition": "only_bear_breadth_lo",
        "topk": 12,
        "max_positions": 12,
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
        raw, panel = load_sources(root, round_c)
        signals = build_signals(raw, panel, signals_dir, runs_dir)
        configs = build_configs(signals, configs_dir)
        manifest = {
            "experiment": OUTPUT_NAME,
            "source_config": str(SOURCE_CONFIG),
            "raw_candidates": str(RAW_CANDIDATES),
            "candidate_panel": str(CANDIDATE_PANEL),
            "backtest_start": BACKTEST_START,
            "execution_lag": 1,
            "deal_price": "close",
            "sizing": "cash_equal",
            "variants": signals,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_report(signals, reports_dir / "SPARSE_FILLER_ROUND_E_PRE.md")
        if args.build_only:
            print(f"[round_e] build-only signals={len(signals)} configs={len(configs)}", flush=True)
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

    collect_artifacts(artifacts_dir, runs_dir, reports_dir, round_d)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sources(root: Path, round_c) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(root / "capacity_extension_round_a_current/signals/true_wts_raw_candidates.parquet")
    raw = raw.copy()
    raw["signal_time"] = pd.to_datetime(raw["signal_time"]).dt.strftime("%Y-%m-%d")
    raw["instrument"] = raw["instrument"].astype(str)
    raw["raw_rank"] = pd.to_numeric(raw["raw_rank"], errors="coerce")
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
    raw = raw.dropna(subset=["signal_time", "instrument", "raw_rank"])
    raw = raw.sort_values(["signal_time", "raw_rank", "instrument"]).reset_index(drop=True)

    panel = pd.read_parquet(root / "data/candidate_panel.parquet")
    panel = panel.copy()
    panel["signal_time"] = pd.to_datetime(panel["signal_time"]).dt.strftime("%Y-%m-%d")
    panel["instrument"] = panel["instrument"].astype(str)
    panel = round_c.prefer_core_panel_rows(panel)
    panel = round_c.add_point_in_time_states(panel)
    if "raw_rank" not in panel:
        panel["raw_rank"] = np.nan
    panel = round_c.add_derived_factor_columns(panel)
    panel["is_buyable_nonst"] = round_c.buyable_mask(panel)
    panel["candidate_rank"] = pd.to_numeric(panel.get("candidate_rank"), errors="coerce")
    panel["layer_score"] = pd.to_numeric(panel.get("layer_score"), errors="coerce")
    for column in ("amount_state", "breadth_state", "regime"):
        panel[column] = panel[column].fillna("unknown")
    return raw, panel.sort_values(["signal_time", "layer", "candidate_rank", "instrument"]).reset_index(drop=True)


def build_signals(raw: pd.DataFrame, panel: pd.DataFrame, signals_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for spec in VARIANTS:
        selected = select_variant(raw, panel, spec)
        selected = selected.dropna(subset=["signal_time", "instrument", "score"])
        selected = selected.sort_values(["signal_time", "score", "instrument"], ascending=[True, False, True])
        path = signals_dir / f"{spec['variant']}.parquet"
        output_cols = [
            "signal_time",
            "instrument",
            "score",
            "raw_rank",
            "signal_source",
            "filler_layer",
            "candidate_rank",
            "filler_condition",
            "raw_count",
            "regime",
            "amount_state",
            "breadth_state",
        ]
        selected[output_cols].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        rows.append(manifest)
        stats.append({**manifest, **signal_stats(selected)})
    pd.DataFrame(stats).to_csv(runs_dir / "round_e_signal_stats.csv", index=False)
    return rows


def select_variant(raw: pd.DataFrame, panel: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    topk = int(spec["topk"])
    raw_top = raw[pd.to_numeric(raw["raw_rank"], errors="coerce") <= topk].copy()
    raw_counts = raw_top.groupby("signal_time").size().rename("raw_count").reset_index()
    raw_top = raw_top.merge(raw_counts, on="signal_time", how="left")
    raw_top["signal_source"] = "raw"
    raw_top["filler_layer"] = ""
    raw_top["candidate_rank"] = np.nan
    raw_top["filler_condition"] = str(spec["filler_condition"])
    raw_top["regime"] = "unknown"
    raw_top["amount_state"] = "unknown"
    raw_top["breadth_state"] = "unknown"
    raw_top["score"] = 100000.0 - pd.to_numeric(raw_top["raw_rank"], errors="coerce").fillna(9999.0)

    raw_count_by_day = raw_top.groupby("signal_time").size()
    all_panel_days = pd.Index(panel["signal_time"].dropna().unique())
    sparse_days = all_panel_days[raw_count_by_day.reindex(all_panel_days, fill_value=0) <= int(spec["sparse_max_raw_count"])]

    filler = panel[panel["signal_time"].isin(sparse_days)].copy()
    filler = filler[filler["is_buyable_nonst"].fillna(False)].copy()
    filler = filler[condition_mask(filler, str(spec["filler_condition"]))].copy()
    filler = select_filler_families(filler, spec)

    if not filler.empty:
        raw_keys = set(zip(raw_top["signal_time"], raw_top["instrument"]))
        filler = filler[
            [key not in raw_keys for key in zip(filler["signal_time"], filler["instrument"])]
        ].copy()
        filler = filler.sort_values(["signal_time", "_family_priority", "candidate_rank", "layer_score"], ascending=[True, True, True, False])
        filler = filler.drop_duplicates(["signal_time", "instrument"], keep="first")
        day_raw_count = raw_count_by_day.reindex(filler["signal_time"], fill_value=0).to_numpy()
        filler["raw_count"] = day_raw_count.astype(int)
        filler["raw_rank"] = np.nan
        filler["signal_source"] = "filler"
        filler["filler_layer"] = filler["layer"].astype(str)
        filler["filler_condition"] = str(spec["filler_condition"])
        filler["score"] = filler_score(filler)
        filler = filler[
            [
                "signal_time",
                "instrument",
                "score",
                "raw_rank",
                "signal_source",
                "filler_layer",
                "candidate_rank",
                "filler_condition",
                "raw_count",
                "regime",
                "amount_state",
                "breadth_state",
            ]
        ]
    else:
        filler = pd.DataFrame(columns=[
            "signal_time",
            "instrument",
            "score",
            "raw_rank",
            "signal_source",
            "filler_layer",
            "candidate_rank",
            "filler_condition",
            "raw_count",
            "regime",
            "amount_state",
            "breadth_state",
        ])

    raw_top = raw_top[
        [
            "signal_time",
            "instrument",
            "score",
            "raw_rank",
            "signal_source",
            "filler_layer",
            "candidate_rank",
            "filler_condition",
            "raw_count",
            "regime",
            "amount_state",
            "breadth_state",
        ]
    ]
    return pd.concat([raw_top, filler], ignore_index=True)


def select_filler_families(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for family_priority, (layer, topn) in enumerate(spec["families"], start=1):
        part = frame[(frame["layer"].astype(str) == str(layer)) & (pd.to_numeric(frame["candidate_rank"], errors="coerce") <= int(topn))].copy()
        if part.empty:
            continue
        part["_family_priority"] = family_priority
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=list(frame.columns) + ["_family_priority"])
    return pd.concat(parts, ignore_index=True)


def condition_mask(frame: pd.DataFrame, condition: str) -> pd.Series:
    danger = frame["regime"].eq("bear") & frame["breadth_state"].eq("lo")
    if condition == "all_sparse":
        return pd.Series(True, index=frame.index)
    if condition == "not_bear_breadth_lo":
        return ~danger
    if condition == "only_bear_breadth_lo":
        return danger
    raise ValueError(f"Unknown filler_condition: {condition}")


def filler_score(frame: pd.DataFrame) -> pd.Series:
    family_priority = pd.to_numeric(frame["_family_priority"], errors="coerce").fillna(9.0)
    candidate_rank = pd.to_numeric(frame["candidate_rank"], errors="coerce").fillna(9999.0)
    layer_score = pd.to_numeric(frame.get("layer_score"), errors="coerce").fillna(0.0)
    return 90000.0 - family_priority * 1000.0 - candidate_rank + layer_score * 0.001


def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
    counts = signals.groupby("signal_time", sort=True).size() if not signals.empty else pd.Series(dtype=int)
    filler = signals[signals["signal_source"].eq("filler")]
    raw = signals[signals["signal_source"].eq("raw")]
    return {
        "variant": str(spec["variant"]),
        "description": str(spec.get("description", "")),
        "path": str(path),
        "mode": "raw_priority_sparse_filler",
        "topk": int(spec["topk"]),
        "max_positions": int(spec["max_positions"]),
        "sizing": "cash_equal",
        "sparse_max_raw_count": int(spec["sparse_max_raw_count"]),
        "families": json.dumps(spec["families"], ensure_ascii=False),
        "filler_condition": str(spec["filler_condition"]),
        "signal_rows": int(len(signals)),
        "signal_days": int(counts.size),
        "raw_rows": int(len(raw)),
        "raw_days": int(raw["signal_time"].nunique()),
        "filler_rows": int(len(filler)),
        "filler_days": int(filler["signal_time"].nunique()),
        "days_with_at_least_topk": int((counts >= int(spec["topk"])).sum()) if not counts.empty else 0,
        "first_signal": str(signals["signal_time"].min()) if not signals.empty else None,
        "last_signal": str(signals["signal_time"].max()) if not signals.empty else None,
    }


def signal_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {
            "avg_daily_candidates": np.nan,
            "median_daily_candidates": np.nan,
            "avg_daily_filler": np.nan,
            "median_daily_filler": np.nan,
        }
    counts = signals.groupby("signal_time").size()
    filler_counts = signals[signals["signal_source"].eq("filler")].groupby("signal_time").size()
    filler_counts = filler_counts.reindex(counts.index, fill_value=0)
    return {
        "avg_daily_candidates": float(counts.mean()),
        "median_daily_candidates": float(counts.median()),
        "max_daily_candidates": int(counts.max()),
        "avg_daily_filler": float(filler_counts.mean()),
        "median_daily_filler": float(filler_counts.median()),
        "max_daily_filler": int(filler_counts.max()),
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
    config["title"] = "弱转强稀疏日补票扩容 Round E"
    config["description"] = "Tmp experiment: raw top12 priority with sparse-day buyable/non-ST filler candidates."
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
    config["rebalance"].pop("rank_weights", None)
    config["execution"] = dict(config.get("execution") or {})
    config["execution"]["deal_price"] = "close"
    config["execution"]["buy"] = dict(config["execution"].get("buy") or {})
    config["execution"]["buy"].update({"sizing": "cash_equal", "lot_size": 100, "skip_if_holding": True, "skip_limit_up": True, "reuse_sell_cash": True})
    config["engine"] = dict(config.get("engine") or {})
    config["engine"].update({"deal_price": "close", "max_workers": 1})
    config["metadata"] = {
        "experiment_root": str(ROOT),
        "experiment": OUTPUT_NAME,
        "variant": signal["variant"],
        "mode": signal.get("mode"),
        "source_config": str(SOURCE_CONFIG),
        "raw_candidates": str(RAW_CANDIDATES),
        "candidate_panel": str(CANDIDATE_PANEL),
        "topk": int(signal["topk"]),
        "max_positions": int(signal["max_positions"]),
        "sizing": "cash_equal",
        "sparse_max_raw_count": int(signal["sparse_max_raw_count"]),
        "families": signal["families"],
        "filler_condition": signal["filler_condition"],
        "execution_lag": 1,
        "deal_price": "close",
        "signal_rows": int(signal["signal_rows"]),
        "signal_days": int(signal["signal_days"]),
        "raw_rows": int(signal["raw_rows"]),
        "raw_days": int(signal["raw_days"]),
        "filler_rows": int(signal["filler_rows"]),
        "filler_days": int(signal["filler_days"]),
        "days_with_at_least_topk": int(signal["days_with_at_least_topk"]),
    }
    return config


def run_configs(configs: list[Path], artifacts_dir: Path, *, dry_run: bool, skip_existing: bool, python_cmd: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[round_e] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue
        print(f"[round_e] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
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
        print(f"[round_e] no artifacts found under {artifacts_dir}", flush=True)
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
    metrics_frame.to_csv(runs_dir / "round_e_metrics_full.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(runs_dir / "round_e_yearly_nav.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(runs_dir / "round_e_entry_weight_buckets.csv", index=False)
    comparison = write_comparison(metrics_frame, reference, runs_dir / "round_e_comparison.csv")
    write_report(comparison, reference, reports_dir / "SPARSE_FILLER_ROUND_E.md")
    print(f"[round_e] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


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
            out[f"{prefix}_post2022_zero_delta"] = safe_float(row.get("post2022_zero_position_ratio")) - safe_float(ref.get("post2022_zero_position_ratio"))
            out[f"{prefix}_y2022_max_buy_weight_delta"] = safe_float(row.get("y2022_max_buy_weight")) - safe_float(ref.get("y2022_max_buy_weight"))
        rows.append(out)
    comparison = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)
    return comparison


def write_pre_report(signals: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Sparse Filler Round E Pre-Backtest",
        "",
        "保持原弱转强 raw top12 优先；只在 raw 候选稀疏日加入已过滤的可买入非 ST filler。",
        "",
        "| variant | sparse<= | families | condition | rows | days | raw_rows | filler_rows | filler_days | days>=topk |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for signal in signals:
        lines.append(
            f"| {signal['variant']} | {signal['sparse_max_raw_count']} | {signal['families']} | {signal['filler_condition']} | "
            f"{signal['signal_rows']} | {signal['signal_days']} | {signal['raw_rows']} | {signal['filler_rows']} | "
            f"{signal['filler_days']} | {signal['days_with_at_least_topk']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(comparison: pd.DataFrame, reference: dict[str, dict[str, Any]], path: Path) -> None:
    lines = [
        "# Sparse Filler Round E",
        "",
        "范围：原弱转强 raw top12 永远优先；filler 只在 raw 稀疏日补位；保持 `selector.lag=1`，第二天 `close` 买入，`cash_equal` sizing 和原卖出规则。",
        "",
        "## Baselines",
        "",
        "| baseline | total_return | max_drawdown | sharpe | avg_pos | post2022_zero | 2022 max buy weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in reference.items():
        lines.append(
            f"| {name} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('post2022_zero_position_ratio')):.4f} | {safe_float(row.get('y2022_max_buy_weight')):.4f} |"
        )
    lines.extend([
        "",
        "## Results",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | post2022_zero | 2022 max_w | 2022 gt20 | filler_rows | filler_days | vs_raw12_ret | vs_raw12_mdd | vs_raw12_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row.get('variant')} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('post2022_zero_position_ratio')):.4f} | {safe_float(row.get('y2022_max_buy_weight')):.4f} | "
            f"{safe_float(row.get('y2022_buy_weight_gt20_ratio')):.4f} | {int(safe_float(row.get('filler_rows')))} | "
            f"{int(safe_float(row.get('filler_days')))} | {safe_float(row.get('vs_raw12_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_raw12_mdd_delta')):.4f} | {safe_float(row.get('vs_raw12_sharpe_delta')):.4f} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `runs/round_e_comparison.csv`",
        "- `runs/round_e_signal_stats.csv`",
        "- `runs/round_e_yearly_nav.csv`",
        "- `runs/round_e_entry_weight_buckets.csv`",
        "- `runs/round_e_metrics_full.csv`",
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
