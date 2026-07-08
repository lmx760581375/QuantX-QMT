"""Trade pattern analysis dataset tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantx.core.analysis.patterns import LabelConfig, PatternAnalysisConfig, build_pattern_analysis, find_similar_samples
from quantx.core.analysis.patterns.features import train_feature_columns
from quantx.core.analysis.patterns.report import (
    build_pattern_summary,
    write_report,
    _failure_recovery_rule_search,
    _fast_exit_rule_candidates,
)
from quantx.core.analysis.patterns.supervised import run_supervised_analysis


def test_build_pattern_analysis_from_run_artifacts_and_csv(tmp_path):
    run_dir = tmp_path / "runs" / "unit_run"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "summary.json", {"run_id": "unit_run", "name": "unit_strategy"})
    _write_json(run_dir / "metrics.json", {})
    _write_json(run_dir / "daily_nav.json", [
        {"date": "2021-01-25", "total_value": 1000000.0, "position_count": 1},
        {"date": "2021-01-26", "total_value": 1001000.0, "position_count": 2},
        {"date": "2021-01-27", "total_value": 1002000.0, "position_count": 2},
        {"date": "2021-01-28", "total_value": 1003000.0, "position_count": 1},
        {"date": "2021-01-29", "total_value": 1004000.0, "position_count": 2},
    ])
    _write_json(run_dir / "trades.json", [])
    _write_json(run_dir / "positions.json", [])
    _write_json(run_dir / "closed_positions.json", [
        {
            "symbol": "SZ000001",
            "entry_date": "2021-01-15",
            "exit_date": "2021-01-22",
            "quantity": 100,
            "entry_price": 11.4,
            "exit_price": 12.4,
            "holding_days": 7,
            "return": 0.0877,
            "net_pnl": 100.0,
        },
        {
            "symbol": "SZ000002",
            "entry_date": "2021-01-15",
            "exit_date": "2021-01-22",
            "quantity": 100,
            "entry_price": 20.8,
            "exit_price": 19.7,
            "holding_days": 7,
            "return": -0.0529,
            "net_pnl": -110.0,
        },
    ])
    _write_json(run_dir / "selection_candidates.json", [])
    _write_json(run_dir / "daily_selection_candidates.json", [
        {"date": "2021-01-25", "raw_candidate_count": 2, "selected_count": 1},
        {"date": "2021-01-26", "raw_candidate_count": 0, "selected_count": 0},
        {"date": "2021-01-27", "raw_candidate_count": 1, "selected_count": 1},
        {"date": "2021-01-28", "raw_candidate_count": 0, "selected_count": 0},
        {"date": "2021-01-29", "raw_candidate_count": 3, "selected_count": 2},
    ])
    _write_json(run_dir / "explain.json", {"config": {"name": "unit_strategy"}})

    raw_dir = tmp_path / "data" / "raw" / "baostock"
    stocks = raw_dir / "stocks"
    stocks.mkdir(parents=True)
    _write_stock_csv(stocks / "SZ000001.csv", "SZ000001", start_price=10.0, drift=0.10)
    _write_stock_csv(stocks / "SZ000002.csv", "SZ000002", start_price=22.0, drift=-0.08)

    result = build_pattern_analysis(
        PatternAnalysisConfig(
            runs=[run_dir],
            output_dir=tmp_path / "artifacts" / "pattern_analysis",
            analysis_id="unit_analysis",
            raw_data_dir=raw_dir,
            prefer_qlib=False,
            pre_n=10,
            post_n=3,
            min_pre_bars=5,
            min_post_bars=1,
            build_models=False,
            build_clusters=True,
            n_clusters=2,
            label_config=LabelConfig(
                success_return_threshold=0.05,
                failure_return_threshold=0.0,
                hard_loss_threshold=0.20,
                max_adverse_threshold=0.20,
            ),
        )
    )

    assert result.sample_count == 2
    assert result.trade_windows_path.exists()
    assert result.bar_features_path.exists()
    assert result.window_features_path.exists()
    assert result.labels_path.exists()
    assert result.cluster_results_path is not None and result.cluster_results_path.exists()
    assert result.report_path is not None and result.report_path.exists()

    bars = pd.read_parquet(result.bar_features_path)
    labels = pd.read_parquet(result.labels_path)
    features = pd.read_parquet(result.window_features_path)

    assert {"offset_from_entry", "body_pct", "upper_shadow_ratio", "volume_ratio_20", "ma20_distance"}.issubset(bars.columns)
    assert set(labels["outcome_label"]) == {"success", "failure"}
    assert "entry_close_position" in features.columns
    assert "hold_first_5d_return" in features.columns
    assert "hold_first_10d_return" in features.columns
    assert "post_20d_best_high_from_entry" in features.columns
    assert "post_20d_first_close_profit_day" in features.columns
    assert "first_10d_return" in labels.columns
    assert "recoverable_failure" in labels.columns
    assert "post_20d_first_close_profit_day" in labels.columns
    assert "recovery_summary" in result.summary
    assert "failure_recovery_profiles" in result.summary
    assert "failure_recovery_rule_search" in result.summary
    assert "failure_wait_diagnostics" in result.summary
    assert result.summary["failure_wait_cost_profiles"]["ok"] is True
    assert result.summary["failure_wait_cost_profiles"]["failed_count"] == 1
    assert result.summary["wait_opportunity_cost"]["ok"] is True
    assert result.summary["wait_opportunity_cost"]["any_candidate_rate"] is not None
    assert result.summary["exit_review_summary"]["bins"]
    assert result.summary["exit_rule_what_if"]["ok"] is True
    assert result.summary["exit_rule_what_if"]["rules"]
    assert "hold_first_10d_return" not in train_feature_columns(features)
    trade_management_columns = train_feature_columns(features, feature_scope="trade_management")
    assert "hold_first_10d_return" in trade_management_columns
    assert "return" in trade_management_columns
    assert "max_favorable_excursion" in trade_management_columns
    assert "exit_reason_stop_loss" in trade_management_columns
    assert "post_20d_best_high_from_entry" not in trade_management_columns
    assert "post_20d_first_close_profit_day" not in trade_management_columns
    assert result.summary["cluster_results"]["ok"] is True
    assert result.summary["figure_results"]
    for figure_group in result.summary["figure_results"].values():
        for relative_path in figure_group.values():
            assert (result.output_dir / relative_path).exists()

    query = find_similar_samples(result.output_dir, str(features.iloc[0]["sample_id"]), top_k=1)
    assert query["feature_count"] > 0
    assert len(query["matches"]) == 1
    assert query["matches"][0]["sample_id"] != features.iloc[0]["sample_id"]


def test_supervised_analysis_reports_trade_management_targets():
    # Synthetic data here is only a pipeline smoke test. Strategy conclusions must
    # come from real run artifacts, not from these hand-shaped rows.
    features = pd.DataFrame([
        {
            "sample_id": f"s{i}",
            "entry_date": f"2021-01-{i + 1:02d}",
            "exit_reason": "time_stop_25d" if i % 2 == 0 else "stop_loss_89permil",
            "exit_reason_time_stop": i % 2 == 0,
            "exit_reason_stop_loss": i % 2 == 1,
            "return": 0.04 if i % 4 in {0, 1} else -0.03,
            "pre_5_return": i / 100,
            "pre_5_max_drawdown": -i / 200,
            "pre_10_return": 0.03 if i % 2 == 0 else 0.07,
            "pre_10_long_lower_count": 2 if i % 2 == 0 else 0,
            "pre_20_above_ma20_ratio": 0.95 if i % 2 == 0 else 0.70,
            "pre_20_max_drawdown": -0.04 if i % 2 == 0 else -0.08,
            "pre_40_above_ma20_ratio": 0.72 if i % 2 == 0 else 0.55,
            "entry_close_position": 0.3 + (i % 5) / 10,
            "entry_volume_ratio_20": 0.8 + (i % 7) / 10,
            "hold_first_3d_return": 0.01 if i % 2 == 0 else -0.03,
            "hold_first_5d_return": -0.03 if i % 3 == 0 else 0.02,
            "hold_first_10d_return": -0.02 if i % 2 == 0 else -0.06,
            "hold_drawdown_after_peak": -0.06 if i % 2 == 0 else -0.01,
            "max_favorable_excursion": 0.08 if i % 5 == 0 else 0.02,
            "max_adverse_excursion": -0.09 if i % 2 == 0 else -0.03,
            "post_20d_best_high_from_entry": 0.10,
        }
        for i in range(24)
    ])
    labels = pd.DataFrame([
        {
            "sample_id": f"s{i}",
            "exit_reason": "time_stop_25d" if i % 2 == 0 else "stop_loss_89permil",
            "outcome_label": "success" if i % 4 == 0 else "failure",
            "binary_success": 1 if i % 4 == 0 else 0,
            "had_opportunity": i % 3 != 0,
            "faded_after_peak": i % 3 == 1,
            "efficient_capture": i % 4 == 0,
            "is_failed_trade": i % 4 != 0,
            "recoverable_failure": (i % 4 != 0) and (i % 3 != 0),
            "loss_reducible_after_exit": (i % 4 != 0) and (i % 2 == 0),
            "close_turn_profitable_after_exit": (i % 4 != 0) and (i % 3 == 0),
        }
        for i in range(24)
    ])

    result = run_supervised_analysis(
        features,
        labels,
        models=["logistic_regression"],
        random_state=7,
        feature_scope="trade_management",
    )

    assert result["ok"] is True
    assert result["majority_baseline"]["label"] == "not_success"
    assert "hold_first_5d_return" in result["feature_columns"]
    assert "max_adverse_excursion" in result["feature_columns"]
    assert "post_20d_best_high_from_entry" not in result["feature_columns"]
    assert {
        "success",
        "opportunity",
        "fade_after_peak",
        "efficient_capture",
        "recoverable_failure",
        "loss_reducible",
        "close_turn_profitable",
    }.issubset(result["targets"])
    assert result["targets"]["opportunity"]["majority_baseline"]["label"] in {"opportunity", "no_opportunity"}
    assert result["targets"]["fade_after_peak"]["models"]["logistic_regression"]["ok"] is True
    assert "positive_rate" in result["targets"]["efficient_capture"]["models"]["logistic_regression"]["score_quantiles"][0]

    wait_opportunity_cost = {
        "ok": True,
        "samples": [
            {
                "sample_id": f"s{i}",
                "exit_reason": "time_stop_25d" if i % 2 == 0 else "stop_loss_89permil",
                "wait_days_observed": 10,
                "candidate_days": 4 if i % 2 == 0 else 1,
                "raw_candidate_days": 5 if i % 2 == 0 else 2,
                "selected_candidate_count": 6 if i % 2 == 0 else 1,
                "raw_candidate_count": 10 if i % 2 == 0 else 3,
                "avg_position_count": 4.0 if i % 2 == 0 else 2.0,
                "max_position_count": 5.0 if i % 2 == 0 else 3.0,
                "near_full_days": 4 if i % 2 == 0 else 0,
            }
            for i in range(24)
            if i % 4 != 0
        ],
    }

    summary = build_pattern_summary(
        features.assign(max_favorable_excursion=0.10, max_adverse_excursion=-0.04),
        labels.assign(
            **{
                "return": features["return"],
            },
            max_favorable_excursion=0.10,
            max_adverse_excursion=-0.04,
            behavior_label=labels["outcome_label"],
            opportunity_label=labels["had_opportunity"].map({True: "opportunity", False: "no_opportunity"}),
            fade_label=labels["faded_after_peak"].map({True: "fade_after_peak", False: "held_or_no_peak_fade"}),
            recovery_label=labels["recoverable_failure"].map({True: "recoverable_failure", False: "not_recoverable_failure"}),
            early_confirmed=labels["had_opportunity"],
            early_failed=~labels["had_opportunity"],
            recoverable_failure=labels["recoverable_failure"],
            loss_reducible_after_exit=labels["loss_reducible_after_exit"],
            close_turn_profitable_after_exit=labels["close_turn_profitable_after_exit"],
            high_turn_profitable_after_exit=labels["recoverable_failure"],
            post_20d_close_improvement=0.02,
            post_20d_best_close_improvement=0.03,
            post_20d_best_high_improvement=0.04,
            post_20d_first_close_profit_day=3.0,
            post_20d_first_high_profit_day=2.0,
            post_20d_first_close_improve_2pct_day=4.0,
            first_3d_return=0.01,
            first_5d_return=0.02,
            first_10d_return=0.03,
            drawdown_after_peak=-0.03,
        ),
        result,
        {"ok": False},
        {},
        {"ok": True, "rules": [{"rule_id": "demo", "avg_delta_return": 0.01}]},
        wait_opportunity_cost,
    )
    assert {"success", "opportunity", "fade_after_peak", "efficient_capture", "recoverable_failure", "loss_reducible", "close_turn_profitable"} == {
        row["target"] for row in summary["target_comparison"]
    }
    assert {"success", "opportunity", "fade_after_peak", "efficient_capture", "recoverable_failure", "loss_reducible", "close_turn_profitable"} == {
        row["target"] for row in summary["model_interpretation"]
    }
    assert summary["recovery_summary"]["recoverable_failure_rate"] is not None
    assert "recovery_feature_differences" in summary
    assert "recovery_feature_differences_by_target" in summary
    assert "close_turn_profitable_after_exit" in summary["recovery_feature_differences_by_target"]
    assert summary["failure_wait_cost_profiles"]["ok"] is True
    assert summary["failure_wait_cost_profiles"]["high_wait_cost_count"] > 0
    assert summary["failure_wait_cost_profiles"]["targets"]
    assert summary["failure_recovery_profiles"]["ok"] is True
    assert {
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
    }.issubset({row["target"] for row in summary["failure_recovery_profiles"]["targets"]})
    loss_profile = next(row for row in summary["failure_recovery_profiles"]["targets"] if row["target"] == "loss_reducible_after_exit")
    assert loss_profile["by_exit_reason"]
    assert loss_profile["top_differences"]
    assert summary["failure_recovery_rule_search"]["ok"] is True
    assert summary["failure_recovery_rule_search"]["targets"]
    assert any(target["rules"] for target in summary["failure_recovery_rule_search"]["targets"])
    assert summary["recovery_rule_candidates"]["ok"] is True
    assert summary["fast_exit_rule_candidates"]["ok"] is True
    assert "rules" in summary["fast_exit_rule_candidates"]
    assert summary["failure_wait_diagnostics"]["ok"] is True
    assert summary["failure_wait_diagnostics"]["close_profit_within_5d_rate"] is not None
    assert summary["failure_wait_diagnostics"]["by_exit_reason"]
    assert {row["rule_id"] for row in summary["recovery_rule_candidates"]["rules"]} >= {
        "time_stop_shallow_wait_profit",
        "stop_loss_rebound_loss_reduction",
        "deep_stop_loss_do_not_wait",
    }
    assert summary["exit_review_summary"]["opportunity_rate_8pct"] is not None
    assert summary["exit_rule_what_if"]["rules"][0]["rule_id"] == "demo"
    assert {row["bucket"] for row in summary["exit_review_summary"]["bins"]} == {
        "no_4pct_opportunity",
        "mfe_4_8pct",
        "mfe_8_12pct",
        "mfe_12_18pct",
        "mfe_18pct_plus",
    }
    report_path = write_report(Path("/tmp") / "quantx_test_trade_patterns_report", summary, pd.DataFrame())
    assert report_path.exists()


def test_failure_recovery_rule_search_runs_on_synthetic_failed_trades():
    # Synthetic data here only validates the rule-search plumbing. It is not
    # evidence that a sell-delay rule will improve a real portfolio backtest.
    features = []
    labels = []
    for i in range(24):
        is_stop_loss = i < 12
        sample_id = f"failed_{i}"
        features.append({
            "sample_id": sample_id,
            "exit_reason": "stop_loss_89permil" if is_stop_loss else "time_stop_25d",
            "exit_reason_stop_loss": is_stop_loss,
            "exit_reason_time_stop": not is_stop_loss,
            "return": -0.092 if is_stop_loss else -0.032,
            "max_adverse_excursion": -0.095 if is_stop_loss else -0.045,
            "max_favorable_excursion": 0.030 if is_stop_loss else 0.070,
            "pre_5_return": -0.010 if is_stop_loss else 0.030,
            "pre_10_return": 0.030 if is_stop_loss else 0.070,
            "pre_10_long_lower_count": 1 if is_stop_loss else 3,
            "pre_20_above_ma20_ratio": 1.0 if is_stop_loss else 0.9,
            "pre_40_above_ma20_ratio": 0.75 if is_stop_loss else 0.65,
            "pre_20_max_drawdown": -0.04 if is_stop_loss else -0.08,
            "entry_close_position": 0.45 if is_stop_loss else 0.65,
            "hold_first_3d_return": 0.010 if is_stop_loss else -0.010,
            "hold_first_10d_return": -0.020 if is_stop_loss else -0.050,
        })
        labels.append({
            "sample_id": sample_id,
            "is_failed_trade": True,
            "loss_reducible_after_exit": is_stop_loss or i in {12, 13},
            "close_turn_profitable_after_exit": (not is_stop_loss) or i in {0, 1},
            "high_turn_profitable_after_exit": True,
            "return": -0.092 if is_stop_loss else -0.032,
            "exit_reason": "stop_loss_89permil" if is_stop_loss else "time_stop_25d",
            "post_20d_best_close_improvement": 0.045 if is_stop_loss else 0.070,
            "post_20d_close_improvement": 0.010 if is_stop_loss else 0.035,
            "post_20d_first_close_profit_day": None if is_stop_loss else 4.0,
            "post_20d_first_close_improve_2pct_day": 3.0 if is_stop_loss else 8.0,
        })

    result = _failure_recovery_rule_search(pd.DataFrame(features), pd.DataFrame(labels))

    assert result["ok"] is True
    assert result["mode"] == "sell_time_rule_search_on_post_exit_labels"
    assert result["failed_count"] == 24
    by_target = {row["target"]: row for row in result["targets"]}
    assert {"loss_reducible_after_exit", "close_turn_profitable_after_exit"}.issubset(by_target)

    loss_rules = by_target["loss_reducible_after_exit"]["rules"]
    close_profit_rules = by_target["close_turn_profitable_after_exit"]["rules"]
    assert loss_rules
    assert close_profit_rules
    assert any("exit_reason_stop_loss" in rule["conditions"] for rule in loss_rules)
    assert any("exit_reason_time_stop" in rule["conditions"] for rule in close_profit_rules)
    assert all(rule["positive_rate"] > rule["baseline_rate"] for rule in loss_rules[:5])
    assert all(rule["lift"] >= 0.04 for rule in close_profit_rules[:5])


def test_fast_exit_rule_candidates_emit_runtime_yaml_rules():
    features = []
    labels = []
    for i in range(20):
        weak = i < 3
        sample_id = f"failed_{i}"
        features.append({
            "sample_id": sample_id,
            "holding_days": 26 if weak else 24,
            "return": -0.055 if weak else -0.025,
            "hold_first_3d_return": -0.025 if weak else 0.005,
            "hold_first_10d_return": -0.065 if weak else -0.010,
            "max_favorable_excursion": 0.012 if weak else 0.055,
            "max_adverse_excursion": -0.085 if weak else -0.040,
        })
        labels.append({
            "sample_id": sample_id,
            "is_failed_trade": True,
            "recoverable_failure": not weak,
            "loss_reducible_after_exit": not weak,
            "close_turn_profitable_after_exit": not weak,
            "high_turn_profitable_after_exit": not weak,
            "post_20d_close_improvement": 0.005 if weak else 0.035,
            "post_20d_best_close_improvement": 0.010 if weak else 0.060,
            "post_20d_first_close_profit_day": None if weak else 4.0,
            "post_20d_first_close_improve_2pct_day": None if weak else 3.0,
        })

    result = _fast_exit_rule_candidates(pd.DataFrame(features), pd.DataFrame(labels))

    assert result["ok"] is True
    assert result["mode"] == "runtime_fast_exit_context_search"
    assert result["rules"]
    top = result["rules"][0]
    assert top["sample_count"] == 3
    assert top["do_not_wait_score"] > 0
    assert "holding_days >" in top["yaml_when"]
    assert "peak_pnl_pct" in top["yaml_when"]
    assert "trough_pnl_pct" in top["yaml_when"]


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_stock_csv(path: Path, symbol: str, start_price: float, drift: float) -> None:
    dates = pd.bdate_range("2021-01-01", periods=30)
    rows = []
    previous = start_price
    for i, date in enumerate(dates):
        close = previous * (1 + drift / 30 + (0.01 if i == 10 and drift > 0 else 0.0))
        open_price = previous * (1 + drift / 60)
        high = max(open_price, close) * 1.02
        low = min(open_price, close) * 0.98
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "code": symbol.lower(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "preclose": previous,
            "volume": 1000000 + i * 10000,
            "amount": (1000000 + i * 10000) * close * 100,
            "turnover": 1.0,
            "pct_chg": (close / previous - 1) * 100,
            "is_st": 0,
        })
        previous = close
    pd.DataFrame(rows).to_csv(path, index=False)
