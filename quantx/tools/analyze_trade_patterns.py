"""Build trade pattern ML analysis artifacts from QuantX runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantx.core.analysis.patterns import LabelConfig, PatternAnalysisConfig, build_pattern_analysis, find_similar_samples
from quantx.core.analysis.patterns.io import json_safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="Run ids, run directories, or latest.")
    parser.add_argument("--analysis-dir", help="Existing pattern analysis directory for similarity lookup.")
    parser.add_argument("--similar-to", help="sample_id to query in an existing analysis directory.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of similar samples to return.")
    parser.add_argument("--same-cluster-only", action="store_true", help="Restrict similarity lookup to the target sample cluster.")
    parser.add_argument("--output-dir", default="artifacts/pattern_analysis")
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--raw-data-dir", default="data/raw/baostock")
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--prefer-csv", action="store_true", help="Disable Qlib and read local CSV data directly.")
    parser.add_argument("--pre-n", type=int, default=40)
    parser.add_argument("--post-n", type=int, default=15)
    parser.add_argument("--min-pre-bars", type=int, default=20)
    parser.add_argument("--min-post-bars", type=int, default=0)
    parser.add_argument("--success-return-threshold", type=float, default=0.05)
    parser.add_argument("--failure-return-threshold", type=float, default=0.0)
    parser.add_argument("--hard-loss-threshold", type=float, default=0.05)
    parser.add_argument("--max-adverse-threshold", type=float, default=0.08)
    parser.add_argument("--recover-improvement-threshold", type=float, default=0.02)
    parser.add_argument("--recover-profit-threshold", type=float, default=0.0)
    parser.add_argument("--models", nargs="*", default=["logistic_regression", "random_forest", "lightgbm"])
    parser.add_argument(
        "--feature-scope",
        choices=["entry", "trade_management"],
        default="entry",
        help="Feature set used by supervised models. trade_management includes holding-period state known at exit.",
    )
    parser.add_argument("--no-models", action="store_true")
    parser.add_argument("--no-clusters", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.similar_to:
        if not args.analysis_dir:
            raise SystemExit("--analysis-dir is required with --similar-to")
        payload = find_similar_samples(
            args.analysis_dir,
            args.similar_to,
            top_k=args.top_k,
            same_cluster_only=args.same_cluster_only,
        )
        if args.json:
            print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        else:
            _print_similarity(payload)
        return 0

    if not args.runs:
        raise SystemExit("--runs is required unless --similar-to is used")

    label_config = LabelConfig(
        success_return_threshold=args.success_return_threshold,
        failure_return_threshold=args.failure_return_threshold,
        hard_loss_threshold=args.hard_loss_threshold,
        max_adverse_threshold=args.max_adverse_threshold,
        recover_improvement_threshold=args.recover_improvement_threshold,
        recover_profit_threshold=args.recover_profit_threshold,
    )
    config = PatternAnalysisConfig(
        runs=[Path(run) for run in args.runs],
        output_dir=Path(args.output_dir),
        analysis_id=args.analysis_id,
        raw_data_dir=Path(args.raw_data_dir),
        provider_uri=Path(args.provider_uri),
        prefer_qlib=not args.prefer_csv,
        pre_n=args.pre_n,
        post_n=args.post_n,
        min_pre_bars=args.min_pre_bars,
        min_post_bars=args.min_post_bars,
        label_config=label_config,
        models=args.models,
        n_clusters=args.n_clusters,
        build_models=not args.no_models,
        build_clusters=not args.no_clusters,
        render_report=not args.no_report,
        feature_scope=args.feature_scope,
    )
    result = build_pattern_analysis(config)
    payload: dict[str, Any] = {
        "ok": result.sample_count > 0,
        "analysis_id": result.analysis_id,
        "output_dir": str(result.output_dir),
        "sample_count": result.sample_count,
        "trade_windows_path": str(result.trade_windows_path),
        "bar_features_path": str(result.bar_features_path),
        "window_features_path": str(result.window_features_path),
        "labels_path": str(result.labels_path),
        "cluster_results_path": str(result.cluster_results_path) if result.cluster_results_path else None,
        "report_path": str(result.report_path) if result.report_path else None,
        "summary": result.summary,
    }
    if args.json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"analysis_id={result.analysis_id}")
        print(f"output_dir={result.output_dir}")
        print(f"sample_count={result.sample_count}")
        if result.report_path:
            print(f"report={result.report_path}")
    return 0 if result.sample_count > 0 else 1


def _print_similarity(payload: dict[str, Any]) -> None:
    target = payload.get("target", {})
    print(f"target={target.get('sample_id')} {target.get('symbol')} {target.get('entry_date')}")
    print(f"feature_count={payload.get('feature_count')} top_k={payload.get('top_k')}")
    for item in payload.get("matches", []):
        print(
            f"distance={item.get('distance'):.4f} "
            f"sample_id={item.get('sample_id')} "
            f"symbol={item.get('symbol')} entry={item.get('entry_date')} "
            f"return={item.get('return')} label={item.get('outcome_label')} "
            f"cluster={item.get('cluster_id')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
