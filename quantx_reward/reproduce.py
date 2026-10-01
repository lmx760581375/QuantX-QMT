"""从日线数据开始复现正式 Reward score 与双 sleeve 回测。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from quantx.core.data.manifest import sha256_file, verify_data_manifest
from quantx_reward.config import (
    REPO_ROOT,
    args_mapping_to_cli,
    load_yaml,
    resolve_data_paths,
    resolve_repo_path,
)
from quantx_reward.verify import compare_metrics, restore_weights, verify_environment
from models.reward.launcher import prepare_inference_launch


DEFAULT_REFERENCE_MANIFEST = REPO_ROOT / "artifacts" / "reference" / "data_20260928" / "MANIFEST.json"
DEFAULT_REFERENCE_METRICS = REPO_ROOT / "artifacts" / "reference" / "quarterly_wts3_reward5" / "metrics.json"
DEFAULT_REFERENCE_SCORE = (
    REPO_ROOT
    / "artifacts"
    / "reference"
    / "reward_score_h7_absolute"
    / "merged_score_summary.json"
)
DEFAULT_REFERENCE_CALENDAR = REPO_ROOT / "artifacts" / "reference" / "data_20260928" / "calendar_day.txt"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _calendar_dates(calendar: Path, end: str) -> list[str]:
    dates = [
        value
        for value in calendar.read_text(encoding="utf-8").splitlines()
        if value and value <= str(end)
    ]
    if not dates:
        raise RuntimeError(f"Qlib calendar has no dates through {end}: {calendar}")
    return dates


def _signal_end_from_calendar(calendar: Path, as_of: str) -> str:
    dates = _calendar_dates(calendar, as_of)
    if len(dates) < 2:
        raise RuntimeError(f"Qlib calendar has no T+1-compatible signal date before {as_of}")
    return dates[-2]


def _overlap_start_from_calendar(
    calendar: Path,
    cutoff: str,
    overlap_trading_days: int,
) -> str:
    dates = _calendar_dates(calendar, cutoff)
    if int(overlap_trading_days) < 1:
        raise ValueError("overlap_trading_days must be positive")
    return dates[max(0, len(dates) - int(overlap_trading_days))]


def _distributed_module_command(
    module: str,
    module_args: dict[str, Any],
    *,
    nproc_per_node: int,
) -> list[str]:
    module_args, nproc_per_node = prepare_inference_launch(module, module_args, nproc_per_node)
    cli_args = args_mapping_to_cli(module_args)
    if int(nproc_per_node) <= 1:
        return [sys.executable, "-m", module, *cli_args]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        "-m",
        module,
        "--",
        *cli_args,
    ]


def _inference_command(
    *,
    feature_root: Path,
    reward_root: Path,
    fold_id: str,
    start: str,
    end: str,
    score_path: Path,
    nproc_per_node: int,
) -> list[str]:
    config = load_yaml("configs/model/reward_v1_infer.yaml")
    inference_args = dict(config["args"])
    inference_args["root"] = str(reward_root)
    inference_args["kronos_root"] = str(feature_root)
    inference_args["fold_id"] = str(fold_id)
    inference_args["start"] = str(start)
    inference_args["end"] = str(end)
    inference_args["output"] = str(score_path)
    inference_args["overwrite"] = True
    inference_args["no_keep_shards"] = True
    return _distributed_module_command(
        str(config["module"]),
        inference_args,
        nproc_per_node=nproc_per_node,
    )


def build_reproduction_plan(args) -> dict[str, Any]:
    data_paths = resolve_data_paths(args.data_root)
    feature_root = resolve_repo_path(args.feature_root)
    reward_root = resolve_repo_path(args.reward_root)
    inference_root = resolve_repo_path(args.inference_root)
    historical_score = resolve_repo_path(args.historical_score_path)
    incremental_score = resolve_repo_path(args.incremental_score_path)
    score_path = resolve_repo_path(args.score_path)
    provider_calendar = data_paths["provider_uri"] / "calendars" / "day.txt"
    planning_calendar = provider_calendar if provider_calendar.is_file() else DEFAULT_REFERENCE_CALENDAR
    signal_end = _signal_end_from_calendar(planning_calendar, str(args.as_of))
    overlap_start = (
        str(args.overlap_start)
        if args.overlap_start
        else _overlap_start_from_calendar(
            planning_calendar,
            str(args.historical_cutoff),
            int(args.overlap_trading_days),
        )
    )
    commands: dict[str, list[str]] = {}
    data_ready = data_paths["provider_uri"].is_dir() and data_paths["raw_stock_dir"].is_dir()
    if not args.skip_data_download and (not data_ready or args.resume_data):
        commands["bootstrap_data"] = [
            sys.executable,
            "-m",
            "quantx_reward.cli",
            "bootstrap-data",
            "--data-root",
            str(data_paths["data_root"]),
            "--start",
            str(args.data_start),
            "--end",
            str(args.as_of),
        ]
        if args.force:
            commands["bootstrap_data"].append("--force")
        if args.resume_data:
            commands["bootstrap_data"].append("--resume")
    if args.update_data and data_ready:
        commands["update_data"] = [
            sys.executable,
            "-m",
            "quantx_reward.cli",
            "update-data",
            "--data-root",
            str(data_paths["data_root"]),
            "--end",
            str(args.as_of),
        ]
    commands["prepare_features"] = [
        sys.executable,
        "-m",
        "models.reward.prepare_features",
        "--stage",
        "all",
        "--output-root",
        str(feature_root),
        "--provider-uri",
        str(data_paths["provider_uri"]),
        "--raw-stock-dir",
        str(data_paths["raw_stock_dir"]),
        "--start",
        str(args.data_start),
        "--end",
        str(args.as_of),
    ]
    data_config = load_yaml("configs/model/data_prepare.yaml")
    reward_args = dict(data_config["reward_dataset"])
    reward_args["root"] = str(reward_root)
    reward_args["kronos_root"] = str(feature_root)
    reward_args["start"] = str(args.data_start)
    reward_args["end"] = str(args.historical_data_end)
    commands["prepare_reward_dataset"] = [
        sys.executable,
        "-m",
        "models.reward.prepare_dataset",
        *args_mapping_to_cli(reward_args),
    ]
    commands["historical_infer"] = _inference_command(
        feature_root=feature_root,
        reward_root=reward_root,
        fold_id="pre2020_eval2020_2026",
        start=str(args.score_start),
        end=str(args.historical_cutoff),
        score_path=historical_score,
        nproc_per_node=int(args.nproc_per_node),
    )
    commands["prepare_inference_index"] = [
        sys.executable,
        "-m",
        "models.reward.pipeline.inference_index",
        "--kronos-root",
        str(feature_root),
        "--output-root",
        str(inference_root),
        "--start",
        overlap_start,
        "--end",
        signal_end,
    ]
    commands["incremental_infer"] = _inference_command(
        feature_root=feature_root,
        reward_root=inference_root,
        fold_id="latest_inference",
        start=overlap_start,
        end=signal_end,
        score_path=incremental_score,
        nproc_per_node=int(args.nproc_per_node),
    )
    commands["merge_scores"] = [
        sys.executable,
        "-m",
        "quantx_reward.cli",
        "merge-scores",
        "--historical",
        str(historical_score),
        "--incremental",
        str(incremental_score),
        "--output",
        str(score_path),
        "--cutoff-date",
        str(args.historical_cutoff),
        "--overwrite",
    ]
    commands["dual_sleeve"] = [
        sys.executable,
        "-m",
        "quantx_reward.cli",
        "dual-sleeve",
        "--data-root",
        str(data_paths["data_root"]),
        "--score-path",
        str(score_path),
        "--end",
        str(args.as_of),
        "--output-dir",
        str(resolve_repo_path(args.output_dir)),
        "--run-id",
        str(args.run_id),
    ]
    if args.force:
        commands["prepare_features"].append("--force")
        commands["prepare_reward_dataset"].append("--force")
        commands["prepare_inference_index"].append("--force")
    return {
        "data_root": str(data_paths["data_root"]),
        "provider_uri": str(data_paths["provider_uri"]),
        "feature_root": str(feature_root),
        "reward_root": str(reward_root),
        "inference_root": str(inference_root),
        "historical_score_path": str(historical_score),
        "incremental_score_path": str(incremental_score),
        "score_path": str(score_path),
        "as_of": str(args.as_of),
        "historical_data_end": str(args.historical_data_end),
        "historical_cutoff": str(args.historical_cutoff),
        "overlap_start": overlap_start,
        "signal_end": signal_end,
        "commands": commands,
    }


def reproduce_full(args) -> dict[str, Any]:
    plan = build_reproduction_plan(args)
    if args.dry_run:
        return {"ok": True, "dry_run": True, **plan}

    data_paths = resolve_data_paths(args.data_root)
    provider_uri = data_paths["provider_uri"]
    data_ready = provider_uri.is_dir() and data_paths["raw_stock_dir"].is_dir()
    if not data_ready or args.resume_data:
        if args.skip_data_download:
            raise FileNotFoundError(f"Missing Qlib provider or BaoStock CSV directory under {data_paths['data_root']}")
        _run(plan["commands"]["bootstrap_data"])
    elif args.update_data:
        _run(plan["commands"]["update_data"])

    data_validation = None
    if args.strict_data:
        data_validation = verify_data_manifest(
            data_paths["data_root"],
            resolve_repo_path(args.reference_manifest),
        )
        if not data_validation["ok"]:
            failed = [
                name
                for name, check in data_validation["checks"].items()
                if not check["ok"]
            ]
            raise RuntimeError(
                "Downloaded data does not match the frozen reference manifest; "
                f"failed checks={failed}. Refusing to label the result as an exact reproduction"
            )

    environment_validation = verify_environment(
        resolve_repo_path(args.environment_reference),
        strict_gpu=args.strict_environment,
    )
    if args.strict_environment and not environment_validation["ok"]:
        raise RuntimeError("Runtime environment does not match the frozen reference")

    restore_weights(bundle="reward_v1")
    if not args.skip_features:
        _run(plan["commands"]["prepare_features"])
    if not args.skip_dataset:
        _run(plan["commands"]["prepare_reward_dataset"])
    if not args.skip_historical_inference:
        _run(plan["commands"]["historical_infer"])
    if not args.skip_inference_index:
        _run(plan["commands"]["prepare_inference_index"])
    if not args.skip_incremental_inference:
        _run(plan["commands"]["incremental_infer"])
    if not args.skip_merge:
        _run(plan["commands"]["merge_scores"])

    score_path = Path(plan["score_path"])
    if not score_path.is_file():
        raise FileNotFoundError(f"Missing final score artifact: {score_path}")
    expected_score = json.loads(
        resolve_repo_path(args.reference_score).read_text(encoding="utf-8")
    )
    expected_score_hash = (expected_score.get("sha256") or {}).get("merged")
    actual_score_hash = sha256_file(score_path)
    score_validation = {
        "expected_sha256": expected_score_hash,
        "actual_sha256": actual_score_hash,
        "ok": expected_score_hash == actual_score_hash,
    }
    if args.strict_score and not score_validation["ok"]:
        raise RuntimeError(
            "Generated score does not match the frozen reference; "
            f"expected={expected_score_hash}, actual={actual_score_hash}. "
            "Refusing to run an exact-reproduction backtest"
        )
    _run(plan["commands"]["dual_sleeve"])

    result = {
        "ok": True,
        **plan,
        "data_validation": data_validation,
        "environment_validation": environment_validation,
        "score_validation": score_validation,
    }
    if str(args.as_of) == "2026-09-28":
        comparison = compare_metrics(
            resolve_repo_path(args.output_dir) / str(args.run_id) / "metrics.json",
            DEFAULT_REFERENCE_METRICS,
        )
        result["reference_comparison"] = comparison
        if args.strict_metrics and not comparison["ok"]:
            raise RuntimeError("Reproduced metrics do not exactly match the frozen reference")
    return result
