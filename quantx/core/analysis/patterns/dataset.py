"""End-to-end trade pattern analysis dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .artifacts import load_closed_trades
from .clustering import run_cluster_analysis
from .config import PatternAnalysisConfig
from .exit_rules import run_exit_rule_what_if
from .features import build_trade_windows, build_window_features
from .io import write_json, write_table
from .labels import build_labels
from .market_data import DailyBarLoader
from .opportunity_cost import build_wait_opportunity_cost
from .plotting import write_cluster_kline_figures
from .report import build_pattern_summary, write_report
from .supervised import run_supervised_analysis


@dataclass
class PatternAnalysisResult:
    """Paths and summary for a built pattern analysis artifact."""

    analysis_id: str
    output_dir: Path
    sample_count: int
    trade_windows_path: Path
    bar_features_path: Path
    window_features_path: Path
    labels_path: Path
    cluster_results_path: Path | None
    report_path: Path | None
    summary: Dict[str, Any]


def build_pattern_analysis(config: PatternAnalysisConfig) -> PatternAnalysisResult:
    """Build pattern-analysis tables, optional models/clusters, and a report."""
    analysis_id = config.analysis_id or datetime.now().strftime("%Y%m%d_%H%M%S_trade_patterns")
    output_dir = config.output_dir / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())

    trades = load_closed_trades(config.runs)
    if trades.empty:
        empty = pd.DataFrame()
        trade_windows_path = write_table(empty, output_dir / "trade_windows.parquet")
        bar_features_path = write_table(empty, output_dir / "bar_features.parquet")
        window_features_path = write_table(empty, output_dir / "window_features.parquet")
        labels_path = write_table(empty, output_dir / "labels.parquet")
        summary = {"ok": False, "reason": "no closed trades", "sample_count": 0}
        write_json(output_dir / "pattern_summary.json", summary)
        return PatternAnalysisResult(
            analysis_id=analysis_id,
            output_dir=output_dir,
            sample_count=0,
            trade_windows_path=trade_windows_path,
            bar_features_path=bar_features_path,
            window_features_path=window_features_path,
            labels_path=labels_path,
            cluster_results_path=None,
            report_path=None,
            summary=summary,
        )

    loader = DailyBarLoader(
        raw_data_dir=config.raw_data_dir,
        provider_uri=config.provider_uri,
        prefer_qlib=config.prefer_qlib,
    )
    windows, bars = build_trade_windows(trades, loader, config)
    window_features = build_window_features(windows, bars)
    labels = build_labels(window_features, config.label_config) if not window_features.empty else pd.DataFrame()

    model_results: Dict[str, Any] = {"ok": False, "reason": "disabled", "models": {}}
    if config.build_models and not window_features.empty and not labels.empty:
        model_results = run_supervised_analysis(
            window_features,
            labels,
            models=config.models,
            random_state=config.random_state,
            feature_scope=config.feature_scope,
        )

    cluster_rows = pd.DataFrame()
    cluster_results: Dict[str, Any] = {"ok": False, "reason": "disabled"}
    if config.build_clusters and not window_features.empty:
        cluster_rows, cluster_results = run_cluster_analysis(
            window_features,
            labels,
            n_clusters=config.n_clusters,
            random_state=config.random_state,
        )

    trade_windows_path = write_table(windows, output_dir / "trade_windows.parquet")
    bar_features_path = write_table(bars, output_dir / "bar_features.parquet")
    window_features_path = write_table(window_features, output_dir / "window_features.parquet")
    labels_path = write_table(labels, output_dir / "labels.parquet")
    cluster_results_path = write_table(cluster_rows, output_dir / "cluster_results.parquet") if not cluster_rows.empty else None

    figure_results: Dict[str, Any] = {}
    if config.render_report and not cluster_rows.empty and not bars.empty:
        figure_results = write_cluster_kline_figures(output_dir, bars, cluster_rows)

    exit_rule_what_if = run_exit_rule_what_if(windows, bars)
    wait_opportunity_cost = build_wait_opportunity_cost(windows, config.runs)
    summary = build_pattern_summary(
        window_features,
        labels,
        model_results,
        cluster_results,
        figure_results,
        exit_rule_what_if,
        wait_opportunity_cost,
    )
    report_path = write_report(output_dir, summary, cluster_rows) if config.render_report else None
    if not config.render_report:
        write_json(output_dir / "pattern_summary.json", summary)

    return PatternAnalysisResult(
        analysis_id=analysis_id,
        output_dir=output_dir,
        sample_count=int(len(windows)),
        trade_windows_path=trade_windows_path,
        bar_features_path=bar_features_path,
        window_features_path=window_features_path,
        labels_path=labels_path,
        cluster_results_path=cluster_results_path,
        report_path=report_path,
        summary=summary,
    )
