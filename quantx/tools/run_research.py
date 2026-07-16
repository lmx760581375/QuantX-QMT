"""Run leakage-aware walk-forward research and an optional standard backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.research.data_version import DataVersionResolver, ProviderLock
from quantx.core.research.dataset import DatasetBuilder
from quantx.core.research.evaluation import aggregate_seed_metrics
from quantx.core.research.fingerprint import compute_code_fingerprint
from quantx.core.research.lifecycle import ExperimentLifecycle, ExperimentLock, LiveShadowRegistry
from quantx.core.research.runner import ResearchRunner
from quantx.core.research.snapshot import PhysicalSnapshotManager
from quantx.core.research.specs import FeatureSpec, LabelSpec
from quantx.core.research.split import AnchoredWalkForwardSplitter
from quantx.core.research.trainer import LightGBMTrainer, RidgeTrainer, TorchTrainer
from quantx.core.research.universe import PointInTimeUniverseProvider, UniverseAuditThresholds


def load_research_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Research config must be a YAML mapping")
    return config


def run_research_config(path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    config = load_research_config(path)
    data_cfg = dict(config.get("data") or {})
    provider_uri = Path(data_cfg["provider_uri"])
    output_cfg = dict(config.get("output") or {})
    output_root = Path(output_cfg.get("root") or f"/tmp/quantx-research/{config.get('name', 'experiment')}")
    if not _is_under_tmp(output_root) and not bool(output_cfg.get("allow_project_output", False)):
        raise ValueError("Exploratory research output must stay under /tmp")
    with ProviderLock(provider_uri, shared=True):
        version = DataVersionResolver().resolve(
            provider_uri,
            source_sync_id=str(data_cfg.get("source_sync_id", "manual-research")),
            adjustment_mode=str(data_cfg.get("adjustment_mode", "front_ratio")),
        )
        reader = QlibBinReader(provider_uri)
        universe = _build_universe(provider_uri, data_cfg)
        calendar = reader.calendar(str(data_cfg["start"]), str(data_cfg["end"]))
        integrity_mode = str(data_cfg.get("integrity_mode", "exploratory"))
        audit = _audit_universe(
            universe,
            calendar,
            data_cfg,
            reader=reader,
            formal=integrity_mode == "formal",
        )
        if not audit.passed and integrity_mode == "formal":
            raise ValueError(f"Point-in-time universe audit failed: {audit.failures}")
        feature_spec = _feature_spec(config)
        label_spec = _label_spec(config)
        dataset = DatasetBuilder(reader, universe).build(
            feature_spec,
            label_spec,
            start=str(data_cfg["start"]),
            end=str(data_cfg["end"]),
        )
        validation = dict(config.get("validation") or {})
        splitter = AnchoredWalkForwardSplitter(
            first_prediction_year=int(
                validation.get("first_prediction_year", validation.get("first_development_oos_year"))
            ),
            last_prediction_year=int(
                validation.get(
                    "last_prediction_year",
                    str(validation.get("development_oos_end", data_cfg["end"]))[:4],
                )
            ),
            validation_sessions=int(validation.get("validation_sessions", 252)),
            embargo_sessions=int(validation.get("embargo_sessions", label_spec.horizon_sessions)),
        )
        if dry_run:
            folds = splitter.split(dataset.frame)
            return {
                "dry_run": True,
                "data_version": version.to_dict(),
                "universe_audit": _audit_response(audit),
                "execution_audit": dataset.execution_audit,
                "dataset_rows": len(dataset.frame),
                "feature_columns": dataset.feature_columns,
                "feature_schema_hash": dataset.feature_spec_hash,
                "label_spec_hash": dataset.label_spec_hash,
                "folds": [
                    {
                        "fold_id": fold.fold_id,
                        "train_rows": len(fold.train),
                        "validation_rows": len(fold.validation),
                        "prediction_rows": len(fold.prediction),
                        "training_information_cutoff": fold.training_information_cutoff.strftime("%Y-%m-%d"),
                    }
                    for fold in folds
                ],
            }
        trainer = _trainer(config)
        seeds = _model_seeds(config)
        evaluation_tier = str(validation.get("evaluation_tier", "development_oos"))
        results = {
            seed: ResearchRunner(
                trainer=trainer,
                splitter=splitter,
                output_dir=output_root,
                experiment_name=str(config.get("name", "research")),
                seed=seed,
                evaluation_tier=evaluation_tier,
            ).run(dataset, data_version_id=version.version_id)
            for seed in seeds
        }
    result = results[seeds[0]]
    response = {
        "experiment_id": result.experiment_id,
        "output_dir": str(result.output_dir),
        "data_version": version.to_dict(),
        "universe_audit": _audit_response(audit),
        "execution_audit": dataset.execution_audit,
        "evaluation_tier": evaluation_tier,
        "prediction_store": str(result.prediction_path),
        "prediction_checksum": result.prediction_checksum,
        "prediction_count": result.prediction_count,
        "fold_count": result.fold_count,
        "metrics": result.metrics,
        "artifacts": [artifact.__dict__ for artifact in result.artifacts],
        "seed_set": seeds,
        "multi_seed_metrics": aggregate_seed_metrics(
            {seed: seed_result.metrics for seed, seed_result in results.items()}
        ),
        "seed_runs": {
            str(seed): {
                "experiment_id": seed_result.experiment_id,
                "output_dir": str(seed_result.output_dir),
                "prediction_store": str(seed_result.prediction_path),
                "prediction_checksum": seed_result.prediction_checksum,
                "prediction_count": seed_result.prediction_count,
                "fold_count": seed_result.fold_count,
                "metrics": seed_result.metrics,
                "artifacts": [artifact.__dict__ for artifact in seed_result.artifacts],
            }
            for seed, seed_result in results.items()
        },
    }
    if bool((config.get("backtest") or {}).get("enabled", False)):
        backtests = {
            str(seed): _run_standard_backtest(config, seed_result, dataset, output_root)
            for seed, seed_result in results.items()
        }
        response["backtest"] = backtests[str(seeds[0])]
        response["seed_backtests"] = backtests
    lifecycle_result = _apply_lifecycle(
        config,
        provider_uri=provider_uri,
        data_version=version,
        dataset=dataset,
        results=results,
        seeds=seeds,
        integrity_mode=integrity_mode,
    )
    if lifecycle_result is not None:
        response["lifecycle"] = lifecycle_result
    (result.output_dir / "run_result.json").write_text(
        json.dumps(response, ensure_ascii=True, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return response


def _audit_response(audit) -> dict:
    return {
        "passed": audit.passed,
        "failures": audit.failures,
        "symbol_count": audit.symbol_count,
        "delisted_symbol_count": audit.delisted_symbol_count,
        "member_without_quote_ratio": audit.member_without_quote_ratio,
        "st_status_declared": audit.st_status_declared,
        "suspension_status_declared": audit.suspension_status_declared,
        "historical_constituents_declared": audit.historical_constituents_declared,
    }


def _build_universe(provider_uri: Path, data_cfg: dict) -> PointInTimeUniverseProvider:
    universe = PointInTimeUniverseProvider.from_qlib_instruments(provider_uri / "instruments" / "all.txt")
    configured = data_cfg.get("universe", "all_a")
    if isinstance(configured, list):
        universe = universe.subset(configured)
    elif configured in {"all_mainboard", "pit_all_mainboard"}:
        universe = universe.subset(symbol for symbol in universe.all_symbols if _is_mainboard_stock(symbol))
    elif configured != "all_a":
        raise ValueError(f"Unsupported research universe: {configured}")
    sample_size = data_cfg.get("symbol_sample_size")
    if sample_size is not None:
        sample_size = int(sample_size)
        if sample_size < 1:
            raise ValueError("symbol_sample_size must be positive")
        symbols = list(universe.all_symbols)
        if sample_size < len(symbols):
            symbols = random.Random(int(data_cfg.get("symbol_sample_seed", 7))).sample(symbols, sample_size)
        universe = universe.subset(symbols)
    limit = data_cfg.get("symbol_limit")
    return universe.subset(universe.all_symbols[: int(limit)]) if limit else universe


def _is_mainboard_stock(symbol: str) -> bool:
    return str(symbol).startswith(("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003"))


def _audit_universe(universe, calendar, data_cfg, *, reader, formal: bool):
    audit_cfg = dict(data_cfg.get("universe_audit") or {})
    measure_quotes = formal or bool(audit_cfg.get("measure_quote_coverage", False))
    quoted_members = _quoted_members(reader, universe, calendar) if measure_quotes else None
    return universe.audit(
        calendar,
        UniverseAuditThresholds(
            max_missing_list_date_ratio=float(audit_cfg.get("max_missing_list_date_ratio", 0.001)),
            max_member_without_quote_ratio=float(audit_cfg.get("max_member_without_quote_ratio", 0.01)),
            require_delisted_history=bool(audit_cfg.get("require_delisted_history", False)),
            require_historical_constituents=bool(audit_cfg.get("require_historical_constituents", False)),
            max_unexplained_daily_count_jump=float(audit_cfg.get("max_unexplained_daily_count_jump", 0.05)),
            require_quote_coverage=formal or bool(audit_cfg.get("require_quote_coverage", False)),
            require_st_status=formal or bool(audit_cfg.get("require_st_status", False)),
            require_suspension_status=formal or bool(audit_cfg.get("require_suspension_status", False)),
        ),
        quoted_members_by_session=quoted_members,
        st_status_declared=bool(data_cfg.get("st_status_source")),
        suspension_status_declared=bool(data_cfg.get("suspension_status_source")),
        historical_constituents_declared=bool(data_cfg.get("historical_constituents_source")),
    )


def _quoted_members(reader, universe, calendar) -> dict[str, set[str]]:
    if len(calendar) == 0:
        return {}
    quotes = reader.features(
        universe.all_symbols,
        ["$open"],
        pd.Timestamp(calendar[0]).strftime("%Y-%m-%d"),
        pd.Timestamp(calendar[-1]).strftime("%Y-%m-%d"),
    )
    if quotes.empty or "$open" not in quotes:
        return {}
    valid = quotes[np.isfinite(quotes["$open"])]
    result: dict[str, set[str]] = {}
    for instrument, session in valid.index:
        result.setdefault(pd.Timestamp(session).strftime("%Y-%m-%d"), set()).add(str(instrument))
    return result


def _feature_spec(config: dict) -> FeatureSpec:
    features = dict(config.get("features") or {})
    return FeatureSpec(
        name=str(features.get("name", "features")),
        raw_fields=tuple(features.get("raw_fields") or ("open", "high", "low", "close", "volume")),
        expressions=dict(features.get("expressions") or {}),
        lookback_sessions=int(features.get("lookback_sessions", 120)),
        normalization=str(features.get("normalization", "none")),
        missing_policy=str(features.get("missing_policy", "preserve_with_mask")),
        require_causal=bool(features.get("require_causal", True)),
        groups=dict(features.get("groups") or config.get("groups") or {}),
    )


def _label_spec(config: dict) -> LabelSpec:
    label = dict(config.get("label") or {})
    horizon = int(label.get("horizon_sessions", 5))
    return LabelSpec(
        name=str(label.get("name", f"return_{horizon}d")),
        horizon_sessions=horizon,
        entry_price=str(label.get("entry_price", "next_open")),
        exit_price=str(label.get("exit_price", "future_open")),
        benchmark=label.get("benchmark"),
        excess_return=bool(label.get("excess_return", False)),
        winsorize=tuple(label["winsorize"]) if label.get("winsorize") is not None else None,
        missing_exit_policy=str(label.get("missing_exit_policy", "mark_missing")),
        execution_diagnostic=bool(label.get("execution_diagnostic", True)),
    )


def _trainer(config: dict):
    model = dict(config.get("model") or {})
    kind = str(model.get("type", "ridge")).lower()
    params = dict(model.get("params") or {})
    if kind == "ridge":
        return RidgeTrainer(alpha=float(params.get("alpha", 1.0)))
    if kind == "lightgbm":
        return LightGBMTrainer(params=params)
    if kind in {"torch", "pytorch", "mlp", "lstm"}:
        architecture = str(params.get("architecture", kind if kind in {"mlp", "lstm"} else "mlp"))
        return TorchTrainer(
            architecture=architecture,
            lookback_sessions=int(params.get("lookback_sessions", 20)),
            hidden_size=int(params.get("hidden_size", 32)),
            epochs=int(params.get("epochs", 20)),
            learning_rate=float(params.get("learning_rate", 0.001)),
        )
    raise ValueError(f"Unsupported research model: {kind}")


def _model_seeds(config: dict) -> tuple[int, ...]:
    model = dict(config.get("model") or {})
    configured = model.get("seeds")
    values = configured if configured is not None else [model.get("seed", 7)]
    seeds = tuple(int(value) for value in values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be a non-empty unique sequence")
    return seeds


def _apply_lifecycle(
    config: dict,
    *,
    provider_uri: Path,
    data_version,
    dataset,
    results: dict,
    seeds: tuple[int, ...],
    integrity_mode: str,
) -> dict | None:
    lifecycle_cfg = dict(config.get("lifecycle") or {})
    action = str(lifecycle_cfg.get("action", "none"))
    if action == "none":
        return None
    lifecycle_root = Path(lifecycle_cfg.get("root", "/tmp/quantx-research/lifecycle"))
    if not _is_under_tmp(lifecycle_root):
        raise ValueError("Research lifecycle artifacts must stay under /tmp")
    lifecycle = ExperimentLifecycle(lifecycle_root)
    first_result = results[seeds[0]]

    if action == "promotion_candidate":
        if integrity_mode != "formal":
            raise ValueError("Promotion candidates require data.integrity_mode=formal")
        snapshot_root = Path(lifecycle_cfg.get("snapshot_root", "/tmp/quantx-research/snapshots"))
        if not _is_under_tmp(snapshot_root):
            raise ValueError("Physical research snapshots must stay under /tmp")
        snapshot = PhysicalSnapshotManager(snapshot_root).create(provider_uri, data_version)
        holdout_range = tuple(lifecycle_cfg.get("holdout_range") or ())
        if len(holdout_range) != 2:
            raise ValueError("promotion_candidate requires lifecycle.holdout_range [start, end]")
        backtest = dict(config.get("backtest") or {})
        experiment_lock = ExperimentLock(
            research_generation=int(lifecycle_cfg.get("research_generation", 1)),
            normalized_research_config_hash=_stable_hash(
                {key: value for key, value in config.items() if key not in {"output", "lifecycle"}}
            ),
            normalized_backtest_config_hash=_stable_hash(backtest),
            data_version_id=data_version.version_id,
            physical_snapshot_id=snapshot.snapshot_id,
            universe_version_id=dataset.universe_version_id,
            feature_schema_hash=dataset.feature_spec_hash,
            label_spec_hash=dataset.label_spec_hash,
            model_artifact_checksums=tuple(
                sorted(artifact.model_checksum for result in results.values() for artifact in result.artifacts)
            ),
            seed_set=seeds,
            portfolio_config_hash=_stable_hash(backtest.get("portfolio") or {}),
            risk_config_hash=_stable_hash(backtest.get("risk") or {}),
            order_planner_config_hash=_stable_hash(backtest.get("order_planner") or {}),
            code_fingerprint=compute_code_fingerprint(Path(__file__).resolve().parents[2]),
            holdout_range=(str(holdout_range[0]), str(holdout_range[1])),
        )
        lock_hash = lifecycle.write_lock(experiment_lock)
        return {
            "action": action,
            "experiment_lock_hash": lock_hash,
            "physical_snapshot_id": snapshot.snapshot_id,
            "physical_snapshot_root": str(snapshot.root),
        }

    if action == "sealed_holdout":
        lock_hash = str(lifecycle_cfg.get("experiment_lock_hash", ""))
        if not lock_hash:
            raise ValueError("sealed_holdout requires lifecycle.experiment_lock_hash")
        evaluation_tier = str((config.get("validation") or {}).get("evaluation_tier", ""))
        if evaluation_tier != "sealed_holdout":
            raise ValueError("sealed_holdout action requires validation.evaluation_tier=sealed_holdout")
        lifecycle.claim_sealed_holdout(lock_hash, result_uri=str(first_result.output_dir))
        return {"action": action, "experiment_lock_hash": lock_hash, "claimed": True}

    if action == "live_shadow":
        strategy_id = str(lifecycle_cfg.get("strategy_id", ""))
        if not strategy_id:
            raise ValueError("live_shadow requires lifecycle.strategy_id")
        latest_artifact = first_result.artifacts[-1]
        registry_path = Path(lifecycle_cfg.get("shadow_registry", lifecycle_root / "live_shadow_registry.json"))
        LiveShadowRegistry(registry_path).register(
            strategy_id=strategy_id,
            artifact_id=latest_artifact.artifact_id,
            artifact_checksum=latest_artifact.model_checksum,
            prediction_store=str(first_result.prediction_path),
        )
        return {
            "action": action,
            "strategy_id": strategy_id,
            "artifact_id": latest_artifact.artifact_id,
            "registry": str(registry_path),
        }

    raise ValueError(f"Unsupported lifecycle action: {action}")


def _stable_hash(value) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _run_standard_backtest(config, result, dataset, output_root):
    from quantx.tools.run_backtest import run_config_with_artifacts

    data_cfg = dict(config.get("data") or {})
    backtest = dict(config.get("backtest") or {})
    research_symbols = sorted(set(str(symbol) for symbol in dataset.frame.index.get_level_values("instrument")))
    strategy_config = {
        "name": f"{config.get('name', 'research')}_backtest",
        "version": 1,
        "strategy": {
            "type": "decision_pipeline",
            "rebalance_interval_sessions": int(backtest.get("rebalance_interval_sessions", 1)),
            "alpha": {
                "type": "predictions",
                "path": str(result.prediction_path),
                "checksum": result.prediction_checksum,
                "artifact_id": result.experiment_id,
                "feature_schema_hash": dataset.feature_spec_hash,
            },
            "portfolio": dict(backtest.get("portfolio") or {"type": "topk_equal", "top_k": 20}),
            "risk": dict(backtest.get("risk") or {"type": "noop"}),
            "order_planner": dict(backtest.get("order_planner") or {"type": "standard"}),
        },
        "data": {
            "provider_uri": str(data_cfg["provider_uri"]),
            "universe": research_symbols,
            "start": str(backtest.get("start", data_cfg["start"])),
            "end": str(backtest.get("end", data_cfg["end"])),
            "look_back_days": int(backtest.get("look_back_days", 0)),
        },
        "execution": {"deal_price": "open"},
        "engine": dict(backtest.get("engine") or {}),
        "cost": dict(backtest.get("cost") or {}),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "backtest.yaml"
        config_path.write_text(yaml.safe_dump(strategy_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return run_config_with_artifacts(
            config_path,
            output_root / "backtests",
            run_id=f"{result.experiment_id}-backtest",
        )


def _is_under_tmp(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    temporary_roots = {
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    return any(resolved == root or root in resolved.parents for root in temporary_roots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_research_config(args.config, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        if result.get("dry_run"):
            print(f"dry_run: dataset_rows={result['dataset_rows']} folds={len(result['folds'])}")
            return 0
        print(f"experiment_id: {result['experiment_id']}")
        print(f"prediction_store: {result['prediction_store']}")
        print(f"rank_ic: {result['metrics'].get('rank_ic')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
