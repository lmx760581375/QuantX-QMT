"""QuantX Reward 统一命令行入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from quantx.core.data.manifest import verify_data_manifest, write_data_manifest
from quantx.tools.plot_sleeve_fusion import main as plot_sleeve_main
from quantx.tools.run_backtest import dry_run_config, run_config_with_artifacts
from quantx.tools.run_dual_sleeve_backtest import run_dual_sleeve
from quantx.tools.bootstrap_daily_data import bootstrap_daily_data, update_daily_data
from quantx_reward.config import REPO_ROOT, load_yaml, print_json, resolve_data_paths, resolve_repo_path
from quantx_reward.reproduce import reproduce_full
from quantx_reward.verify import (
    compare_metrics,
    restore_weights,
    verify_environment,
    verify_weights,
)
from models.reward.launcher import build_command
from models.reward.pipeline.artifact import merge_market_score_artifacts


STRATEGY_CONFIGS = {
    "weak-to-strong": REPO_ROOT / "configs" / "strategies" / "weak_to_strong.yaml",
    "weak-to-strong-pos3": REPO_ROOT / "configs" / "strategies" / "weak_to_strong_pos3.yaml",
    "reward-h7-runner": REPO_ROOT / "configs" / "strategies" / "reward_h7_runner.yaml",
    "reward-h15": REPO_ROOT / "configs" / "strategies" / "reward_h15.yaml",
    "reward-h2-event": REPO_ROOT / "configs" / "strategies" / "reward_h2_event.yaml",
    "reward-h2-event-baseline": REPO_ROOT / "configs" / "strategies" / "reward_h2_event_baseline.yaml",
}

STRATEGY_SCORE_PATHS = {
    "reward-h7-runner": "artifacts/scores/reward_v1.parquet",
    "reward-h15": "artifacts/scores/reward_multitask_epoch10.parquet",
    "reward-h2-event": "artifacts/scores/reward_h2_event_v1.parquet",
    "reward-h2-event-baseline": "artifacts/scores/reward_h2_event_v1.parquet",
}

DEFAULT_DATA_REFERENCE = REPO_ROOT / "artifacts" / "reference" / "data_20260928" / "MANIFEST.json"


def _run_subprocess(command: list[str]) -> int:
    return int(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)


def _resolve_latest(provider_uri: Path) -> str:
    calendar_path = provider_uri / "calendars" / "day.txt"
    dates = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not dates:
        raise ValueError(f"Qlib calendar is empty: {calendar_path}")
    return dates[-1]


def prepare_data(args: argparse.Namespace) -> int:
    paths = resolve_data_paths(args.data_root)
    for name in ("provider_uri", "raw_stock_dir"):
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    config = load_yaml(args.config)
    feature_cfg = dict(config["feature_store"])
    reward_cfg = dict(config["reward_dataset"])
    configured_end = str(feature_cfg.get("end") or "latest")
    end = args.end or (
        _resolve_latest(paths["provider_uri"])
        if configured_end.lower() in {"latest", "auto"}
        else configured_end
    )

    feature_command = [
        sys.executable,
        "-m",
        "models.reward.prepare_features",
        "--stage",
        "all",
        "--output-root",
        str(resolve_repo_path(feature_cfg["output_root"])),
        "--provider-uri",
        str(paths["provider_uri"]),
        "--raw-stock-dir",
        str(paths["raw_stock_dir"]),
        "--start",
        str(args.start or feature_cfg["start"]),
        "--end",
        end,
        "--shard-size",
        str(feature_cfg.get("shard_size", 512)),
    ]
    if args.limit_instruments:
        feature_command.extend(["--limit-instruments", str(args.limit_instruments)])
    if args.force:
        feature_command.append("--force")
    dataset_command = [
        sys.executable,
        "-m",
        "models.reward.prepare_dataset",
        "--root",
        str(resolve_repo_path(reward_cfg["root"])),
        "--kronos-root",
        str(resolve_repo_path(reward_cfg["kronos_root"])),
        "--start",
        str(args.start or reward_cfg["start"]),
        "--end",
        end,
        "--candidate-mode",
        str(reward_cfg["candidate_mode"]),
        "--universe",
        str(reward_cfg["universe"]),
        "--candidate-topn",
        str(reward_cfg["candidate_topn"]),
        "--horizon",
        str(reward_cfg["horizon"]),
        "--target-start-offset",
        str(reward_cfg["target_start_offset"]),
        "--target-price-mode",
        str(reward_cfg["target_price_mode"]),
        "--trend-horizon",
        str(reward_cfg["trend_horizon"]),
        "--trend-threshold",
        str(reward_cfg["trend_threshold"]),
        "--lookback",
        str(reward_cfg["lookback"]),
        "--min-history-rows",
        str(reward_cfg["min_history_rows"]),
        "--target-chunk-size",
        str(reward_cfg["target_chunk_size"]),
    ]
    if args.limit_instruments:
        dataset_command.extend(["--max-instruments", str(args.limit_instruments)])
    if args.limit_days:
        dataset_command.extend(["--limit-days", str(args.limit_days)])
    if args.force:
        dataset_command.append("--force")
    if args.dry_run:
        print_json({
            "ok": True,
            "feature_command": feature_command,
            "dataset_command": dataset_command,
        })
        return 0
    code = _run_subprocess(feature_command)
    if code != 0:
        return code
    return _run_subprocess(dataset_command)


def bootstrap_data(args: argparse.Namespace) -> int:
    result = bootstrap_daily_data(
        data_root=args.data_root,
        start=args.start,
        end=args.end,
        universe_path=resolve_repo_path(args.universe_path),
        security_master_path=resolve_repo_path(args.security_master_path),
        live_universe=args.live_universe,
        workers=args.workers,
        pause_seconds=args.pause_seconds,
        max_retries=args.max_retries,
        socket_timeout=args.socket_timeout,
        limit=args.limit,
        force=args.force,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print_json(result)
    return 0 if result["ok"] else 1


def update_data(args: argparse.Namespace) -> int:
    result = update_daily_data(
        data_root=args.data_root,
        end=args.end,
        universe_path=resolve_repo_path(args.universe_path),
        live_universe=args.live_universe,
        pause_seconds=args.pause_seconds,
        max_retries=args.max_retries,
        socket_timeout=args.socket_timeout,
        overlap_days=args.overlap_days,
        tolerance=args.tolerance,
        max_requests=args.max_requests,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print_json(result)
    return 0 if result["ok"] else 1


def data_manifest(args: argparse.Namespace) -> int:
    if args.action == "write":
        payload = write_data_manifest(args.data_root, args.output)
        print_json({"ok": True, "manifest": payload})
        return 0
    payload = verify_data_manifest(
        args.data_root,
        resolve_repo_path(args.reference),
    )
    print_json(payload)
    return 0 if payload["ok"] else 1


def environment(args: argparse.Namespace) -> int:
    payload = verify_environment(
        resolve_repo_path(args.reference),
        strict_gpu=args.strict_gpu,
    )
    print_json(payload)
    return 0 if payload["ok"] else 1


def reproduce_full_command(args: argparse.Namespace) -> int:
    payload = reproduce_full(args)
    print_json(payload)
    return 0 if payload["ok"] else 1


def launch_model(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    if not args.dry_run:
        restore_weights(bundle=str(config.get("weight_bundle", "reward_v1")))
    command = build_command(
        resolve_repo_path(args.config),
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        rdzv_endpoint=args.rdzv_endpoint,
        rdzv_id=args.rdzv_id,
    )
    if args.dry_run:
        print_json({"ok": True, "command": command})
        return 0
    return _run_subprocess(command)


def prepare_h7_labels(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "models.reward.pipeline.h7_absolute",
        "build",
        "--root",
        str(resolve_repo_path(args.root)),
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.output_dir:
        command.extend(["--output-dir", str(resolve_repo_path(args.output_dir))])
    if args.force:
        command.append("--force")
    if args.dry_run:
        print_json({"ok": True, "command": command})
        return 0
    return _run_subprocess(command)


def prepare_inference_index(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "models.reward.pipeline.inference_index",
        "--kronos-root",
        str(resolve_repo_path(args.kronos_root)),
        "--output-root",
        str(resolve_repo_path(args.output_root)),
        "--start",
        str(args.start),
        "--end",
        str(args.end),
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        print_json({"ok": True, "command": command})
        return 0
    return _run_subprocess(command)


def merge_scores(args: argparse.Namespace) -> int:
    if args.dry_run:
        print_json({
            "ok": True,
            "historical": str(resolve_repo_path(args.historical)),
            "incremental": str(resolve_repo_path(args.incremental)),
            "output": str(resolve_repo_path(args.output)),
            "cutoff_date": str(args.cutoff_date),
            "score_columns": args.score_columns,
        })
        return 0
    manifest = merge_market_score_artifacts(
        resolve_repo_path(args.historical),
        resolve_repo_path(args.incremental),
        resolve_repo_path(args.output),
        cutoff_date=args.cutoff_date,
        score_columns=args.score_columns,
        overwrite=args.overwrite,
    )
    print_json({"ok": True, **manifest})
    return 0


def backtest(args: argparse.Namespace) -> int:
    config = load_yaml(STRATEGY_CONFIGS[args.strategy])
    paths = resolve_data_paths(args.data_root)
    config["data"]["provider_uri"] = str(paths["provider_uri"])
    if args.start:
        config["data"]["start"] = args.start
    if args.end:
        config["data"]["end"] = args.end
    external_score = (config.get("selector") or {}).get("external_score")
    if external_score:
        configured_score = args.score_path or STRATEGY_SCORE_PATHS.get(args.strategy)
        if not configured_score:
            raise ValueError(f"--score-path is required for strategy {args.strategy}")
        score_path = resolve_repo_path(configured_score)
        config["selector"]["external_score"]["path"] = str(score_path)
    output_dir = resolve_repo_path(args.output_dir)
    with TemporaryDirectory() as tmp_dir:
        temp_config = Path(tmp_dir) / "strategy.yaml"
        temp_config.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        if args.dry_run:
            print_json(dry_run_config(temp_config, symbol_limit=args.symbol_limit))
            return 0
        summary = run_config_with_artifacts(
            temp_config,
            output_dir,
            symbol_limit=args.symbol_limit,
            run_id=args.run_id,
        )
    print_json({"ok": True, **summary})
    return 0


def dual_sleeve(args: argparse.Namespace) -> int:
    sleeve_config = load_yaml(args.config)
    paths = resolve_data_paths(args.data_root)
    provider_uri = paths["provider_uri"]
    configured_end = str(args.end or sleeve_config.get("end") or "latest")
    end = _resolve_latest(provider_uri) if configured_end.lower() in {"latest", "auto"} else configured_end
    wts_config = resolve_repo_path(sleeve_config["wts_config"])
    reward_config = resolve_repo_path(sleeve_config["reward_config"])
    score_path = resolve_repo_path(args.score_path or sleeve_config["score_path"])
    wts_weight = float(args.wts_weight if args.wts_weight is not None else sleeve_config["wts_weight"])
    initial_cash = float(
        args.initial_cash
        if args.initial_cash is not None
        else sleeve_config.get("initial_cash", 100_000_000)
    )
    schedule = str(args.schedule or sleeve_config["schedule"])
    start = str(args.start or sleeve_config.get("start") or "2020-01-02")
    if args.dry_run:
        checks = {}
        with TemporaryDirectory() as tmp_dir:
            for sleeve, config_path in (("wts", wts_config), ("reward", reward_config)):
                config = load_yaml(config_path)
                config["data"]["provider_uri"] = str(provider_uri)
                config["data"]["start"] = start
                config["data"]["end"] = end
                if sleeve == "reward":
                    config["selector"]["external_score"]["path"] = str(score_path)
                temp_config = Path(tmp_dir) / f"{sleeve}.yaml"
                temp_config.write_text(
                    yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                checks[sleeve] = dry_run_config(temp_config)
        payload = {
            "ok": True,
            "wts_config": str(wts_config),
            "reward_config": str(reward_config),
            "provider_uri": str(provider_uri),
            "score_path": str(score_path),
            "start": start,
            "end": end,
            "wts_weight": wts_weight,
            "reward_weight": 1.0 - wts_weight,
            "initial_cash": initial_cash,
            "schedule": schedule,
            "strategies": checks,
        }
        print_json(payload)
        return 0
    result = run_dual_sleeve(
        wts_config_path=wts_config,
        rm_config_path=reward_config,
        output_dir=resolve_repo_path(args.output_dir),
        run_id=args.run_id,
        wts_weight=wts_weight,
        schedule=schedule,
        initial_cash=initial_cash,
        start=start,
        end=end,
        provider_uri=str(provider_uri),
        rm_score_path=str(score_path),
    )
    print_json({"ok": True, **result})
    return 0


def reproduce(args: argparse.Namespace) -> int:
    output_dir = resolve_repo_path(args.output_dir)
    reward_run = args.reward_run_id
    wts_run = args.wts_run_id
    shared = {
        "data_root": args.data_root,
        "start": args.start,
        "end": args.end,
        "symbol_limit": args.symbol_limit,
        "output_dir": str(output_dir),
        "dry_run": False,
    }
    reward_args = argparse.Namespace(
        strategy="reward-h15",
        score_path=args.score_path,
        run_id=reward_run,
        **shared,
    )
    wts_args = argparse.Namespace(
        strategy=args.wts_strategy,
        score_path=args.score_path,
        run_id=wts_run,
        **shared,
    )
    code = backtest(reward_args)
    if code != 0:
        return code
    code = backtest(wts_args)
    if code != 0:
        return code
    return sleeve(argparse.Namespace(
        runs_root=str(output_dir),
        reward_run=reward_run,
        wts_run=wts_run,
        wts_weight=args.wts_weight,
        output_dir=args.figure_dir,
        reward_label="Reward H15 Top5",
        wts_label=(
            "Weak-to-Strong Top3"
            if args.wts_strategy == "weak-to-strong-pos3"
            else "Weak-to-Strong Top5"
        ),
    ))


def sleeve(args: argparse.Namespace) -> int:
    argv = [
        "--runs-root",
        str(resolve_repo_path(args.runs_root)),
        "--reward-run",
        args.reward_run,
        "--wts-run",
        args.wts_run,
        "--wts-weight",
        str(args.wts_weight),
        "--output-dir",
        str(resolve_repo_path(args.output_dir)),
    ]
    if getattr(args, "reward_label", None):
        argv.extend(["--reward-label", str(args.reward_label)])
    if getattr(args, "wts_label", None):
        argv.extend(["--wts-label", str(args.wts_label)])
    return int(plot_sleeve_main(argv))


def verify(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"weights": verify_weights()}
    dual_run = getattr(args, "dual_run", None)
    if dual_run:
        runs_root = resolve_repo_path(args.runs_root or "runs")
        payload["quarterly_wts3_reward5"] = compare_metrics(
            runs_root / dual_run / "metrics.json",
            REPO_ROOT / "artifacts" / "reference" / "quarterly_wts3_reward5" / "metrics.json",
        )
    if args.runs_root and not dual_run:
        runs_root = resolve_repo_path(args.runs_root)
        payload["reward_h15"] = compare_metrics(
            runs_root / args.reward_run / "metrics.json",
            REPO_ROOT / "artifacts" / "reference" / "reward_h15" / "metrics.json",
        )
        payload["weak_to_strong"] = compare_metrics(
            runs_root / args.wts_run / "metrics.json",
            REPO_ROOT / "artifacts" / "reference" / "weak_to_strong" / "metrics.json",
        )
    payload["ok"] = all(
        bool(section.get("ok"))
        for section in payload.values()
        if isinstance(section, dict)
    )
    print_json(payload)
    return 0 if payload["ok"] else 1


def weights(args: argparse.Namespace) -> int:
    payload = (
        restore_weights(force=args.force, bundle=args.bundle)
        if args.action == "restore"
        else verify_weights(bundle=args.bundle)
    )
    print_json(payload)
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="从同源数据构建 Reward 训练数据")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--config", default="configs/model/data_prepare.yaml")
    prepare.add_argument("--start")
    prepare.add_argument("--end")
    prepare.add_argument("--limit-instruments", type=int)
    prepare.add_argument("--limit-days", type=int)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(handler=prepare_data)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-data",
        help="从 BaoStock 全量拉取前复权日线并构建 Qlib provider",
    )
    bootstrap_parser.add_argument("--data-root", required=True)
    bootstrap_parser.add_argument("--start", default="2010-01-01")
    bootstrap_parser.add_argument("--end", required=True)
    bootstrap_parser.add_argument(
        "--universe-path",
        default="artifacts/reference/data_20260928/instruments_all.txt",
    )
    bootstrap_parser.add_argument(
        "--security-master-path",
        default="artifacts/reference/data_20260928/security_master.csv",
    )
    bootstrap_parser.add_argument("--live-universe", action="store_true")
    bootstrap_parser.add_argument("--workers", type=int, default=1)
    bootstrap_parser.add_argument("--pause-seconds", type=float, default=0.5)
    bootstrap_parser.add_argument("--max-retries", type=int, default=3)
    bootstrap_parser.add_argument("--socket-timeout", type=float, default=30.0)
    bootstrap_parser.add_argument("--limit", type=int)
    bootstrap_parser.add_argument("--force", action="store_true")
    bootstrap_parser.add_argument("--resume", action="store_true")
    bootstrap_parser.add_argument("--dry-run", action="store_true")
    bootstrap_parser.set_defaults(handler=bootstrap_data)

    update_parser = subparsers.add_parser(
        "update-data",
        help="增量更新 BaoStock CSV 与 Qlib provider，并检测前复权历史变化",
    )
    update_parser.add_argument("--data-root", required=True)
    update_parser.add_argument("--end", required=True)
    update_parser.add_argument(
        "--universe-path",
        default="artifacts/reference/data_20260928/instruments_all.txt",
    )
    update_parser.add_argument("--live-universe", action="store_true")
    update_parser.add_argument("--pause-seconds", type=float, default=0.5)
    update_parser.add_argument("--max-retries", type=int, default=3)
    update_parser.add_argument("--socket-timeout", type=float, default=30.0)
    update_parser.add_argument("--overlap-days", type=int, default=40)
    update_parser.add_argument("--tolerance", type=float, default=1.0e-4)
    update_parser.add_argument("--max-requests", type=int, default=45_000)
    update_parser.add_argument("--limit", type=int)
    update_parser.add_argument("--dry-run", action="store_true")
    update_parser.set_defaults(handler=update_data)

    manifest_parser = subparsers.add_parser(
        "data-manifest",
        help="生成或校验原始行情与 Qlib provider 的内容摘要",
    )
    manifest_parser.add_argument("action", choices=("write", "verify"))
    manifest_parser.add_argument("--data-root", required=True)
    manifest_parser.add_argument("--output")
    manifest_parser.add_argument(
        "--reference",
        default=str(DEFAULT_DATA_REFERENCE),
    )
    manifest_parser.set_defaults(handler=data_manifest)

    environment_parser = subparsers.add_parser(
        "environment",
        help="校验 Python、核心依赖以及可选 GPU 环境",
    )
    environment_parser.add_argument("--reference", default="environment.reference.json")
    environment_parser.add_argument("--strict-gpu", action="store_true")
    environment_parser.set_defaults(handler=environment)

    full_parser = subparsers.add_parser(
        "reproduce-full",
        help="从日线数据准备开始生成 score 并复现正式双 sleeve",
    )
    full_parser.add_argument("--data-root", required=True)
    full_parser.add_argument("--data-start", default="2010-01-01")
    full_parser.add_argument("--score-start", default="2020-01-02")
    full_parser.add_argument("--as-of", default="2026-09-28")
    full_parser.add_argument("--feature-root", default="workdirs/feature_store")
    full_parser.add_argument("--reward-root", default="workdirs/reward/market_all_close2close_balanced_v2")
    full_parser.add_argument("--inference-root", default="workdirs/reward/latest_inference")
    full_parser.add_argument("--historical-data-end", default="2026-07-15")
    full_parser.add_argument("--historical-cutoff", default="2026-06-02")
    full_parser.add_argument("--overlap-start")
    full_parser.add_argument("--overlap-trading-days", type=int, default=7)
    full_parser.add_argument(
        "--historical-score-path",
        default="artifacts/scores/reward_v1_historical.parquet",
    )
    full_parser.add_argument(
        "--incremental-score-path",
        default="artifacts/scores/reward_v1_incremental.parquet",
    )
    full_parser.add_argument("--score-path", default="artifacts/scores/reward_v1.parquet")
    full_parser.add_argument("--output-dir", default="runs")
    full_parser.add_argument("--run-id", default="quarterly_wts3_reward5_65_35")
    full_parser.add_argument("--nproc-per-node", type=int, default=8)
    full_parser.add_argument(
        "--reference-manifest",
        default="artifacts/reference/data_20260928/MANIFEST.json",
    )
    full_parser.add_argument(
        "--environment-reference",
        default="environment.reference.json",
    )
    full_parser.add_argument(
        "--reference-score",
        default="artifacts/reference/reward_score_h7_absolute/merged_score_summary.json",
    )
    full_parser.add_argument("--skip-data-download", action="store_true")
    full_parser.add_argument("--resume-data", action="store_true")
    full_parser.add_argument("--update-data", action="store_true")
    full_parser.add_argument("--skip-features", action="store_true")
    full_parser.add_argument("--skip-dataset", action="store_true")
    full_parser.add_argument("--skip-historical-inference", action="store_true")
    full_parser.add_argument("--skip-inference-index", action="store_true")
    full_parser.add_argument("--skip-incremental-inference", action="store_true")
    full_parser.add_argument("--skip-merge", action="store_true")
    full_parser.add_argument("--force", action="store_true")
    full_parser.add_argument(
        "--strict-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    full_parser.add_argument(
        "--strict-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    full_parser.add_argument(
        "--strict-score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    full_parser.add_argument("--strict-environment", action="store_true")
    full_parser.add_argument("--dry-run", action="store_true")
    full_parser.set_defaults(handler=reproduce_full_command)

    for name, default_config in (
        ("train", "configs/model/reward_v1_train.yaml"),
        ("infer", "configs/model/reward_v1_infer.yaml"),
    ):
        model_parser = subparsers.add_parser(name)
        model_parser.add_argument("--config", default=default_config)
        model_parser.add_argument("--nproc-per-node", type=int)
        model_parser.add_argument("--nnodes", type=int, default=1)
        model_parser.add_argument("--node-rank", type=int)
        model_parser.add_argument("--rdzv-endpoint")
        model_parser.add_argument("--rdzv-id")
        model_parser.add_argument("--dry-run", action="store_true")
        model_parser.set_defaults(handler=launch_model)

    labels_parser = subparsers.add_parser(
        "prepare-h7-labels",
        help="构建 H7 绝对收益 good/bad 标签工件",
    )
    labels_parser.add_argument("--root", default="workdirs/reward/market_all_close2close_balanced_v2")
    labels_parser.add_argument("--output-dir")
    labels_parser.add_argument("--chunk-size", type=int, default=500_000)
    labels_parser.add_argument("--force", action="store_true")
    labels_parser.add_argument("--dry-run", action="store_true")
    labels_parser.set_defaults(handler=prepare_h7_labels)

    index_parser = subparsers.add_parser(
        "prepare-inference-index",
        help="构建不依赖未来标签的全市场推理索引",
    )
    index_parser.add_argument("--kronos-root", default="workdirs/feature_store")
    index_parser.add_argument("--output-root", default="workdirs/reward/latest_inference")
    index_parser.add_argument("--start", required=True)
    index_parser.add_argument("--end", required=True)
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--dry-run", action="store_true")
    index_parser.set_defaults(handler=prepare_inference_index)

    merge_parser = subparsers.add_parser("merge-scores", help="按日期边界合并历史和增量 score")
    merge_parser.add_argument("--historical", required=True)
    merge_parser.add_argument("--incremental", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--cutoff-date", required=True)
    merge_parser.add_argument("--score-columns", default="reward_score_7d")
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.add_argument("--dry-run", action="store_true")
    merge_parser.set_defaults(handler=merge_scores)

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGY_CONFIGS))
    backtest_parser.add_argument("--data-root", required=True)
    backtest_parser.add_argument("--score-path")
    backtest_parser.add_argument("--start")
    backtest_parser.add_argument("--end")
    backtest_parser.add_argument("--symbol-limit", type=int)
    backtest_parser.add_argument("--output-dir", default="runs")
    backtest_parser.add_argument("--run-id")
    backtest_parser.add_argument("--dry-run", action="store_true")
    backtest_parser.set_defaults(handler=backtest)

    dual_parser = subparsers.add_parser("dual-sleeve", help="运行独立账户与现金再平衡的真实双 sleeve 回测")
    dual_parser.add_argument("--config", default="configs/sleeve/quarterly_wts3_reward5_65_35.yaml")
    dual_parser.add_argument("--data-root", required=True)
    dual_parser.add_argument("--score-path")
    dual_parser.add_argument("--start")
    dual_parser.add_argument("--end")
    dual_parser.add_argument("--wts-weight", type=float)
    dual_parser.add_argument("--initial-cash", type=float)
    dual_parser.add_argument("--schedule", choices=("none", "monthly", "quarterly"))
    dual_parser.add_argument("--output-dir", default="runs")
    dual_parser.add_argument("--run-id", required=True)
    dual_parser.add_argument("--dry-run", action="store_true")
    dual_parser.set_defaults(handler=dual_sleeve)

    reproduce_parser = subparsers.add_parser("reproduce", help="复跑历史 Reward H15 静态 sleeve")
    reproduce_parser.add_argument("--data-root", required=True)
    reproduce_parser.add_argument("--score-path", default="artifacts/scores/reward_multitask_epoch10.parquet")
    reproduce_parser.add_argument("--start", default="2020-01-02")
    reproduce_parser.add_argument("--end", default="2026-06-02")
    reproduce_parser.add_argument("--symbol-limit", type=int)
    reproduce_parser.add_argument("--output-dir", default="runs")
    reproduce_parser.add_argument("--reward-run-id", default="reward_h15_reference")
    reproduce_parser.add_argument("--wts-run-id", default="weak_to_strong_reference")
    reproduce_parser.add_argument("--wts-weight", type=float, default=0.5)
    reproduce_parser.add_argument(
        "--wts-strategy",
        choices=("weak-to-strong", "weak-to-strong-pos3"),
        default="weak-to-strong",
    )
    reproduce_parser.add_argument("--figure-dir", default="artifacts/sleeve/reward_h15_wts_50_50")
    reproduce_parser.set_defaults(handler=reproduce)

    sleeve_parser = subparsers.add_parser("sleeve")
    sleeve_parser.add_argument("--runs-root", default="runs")
    sleeve_parser.add_argument("--reward-run", required=True)
    sleeve_parser.add_argument("--wts-run", required=True)
    sleeve_parser.add_argument("--wts-weight", type=float, default=0.5)
    sleeve_parser.add_argument("--reward-label")
    sleeve_parser.add_argument("--wts-label")
    sleeve_parser.add_argument("--output-dir", default="artifacts/sleeve")
    sleeve_parser.set_defaults(handler=sleeve)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--runs-root")
    verify_parser.add_argument("--dual-run")
    verify_parser.add_argument("--reward-run", default="migration_reward_h15_validation")
    verify_parser.add_argument("--wts-run", default="migration_weak_to_strong_validation")
    verify_parser.set_defaults(handler=verify)

    weights_parser = subparsers.add_parser("weights")
    weights_parser.add_argument("action", choices=("restore", "verify"), nargs="?", default="restore")
    weights_parser.add_argument("--bundle", default="reward_v1")
    weights_parser.add_argument("--force", action="store_true")
    weights_parser.set_defaults(handler=weights)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
