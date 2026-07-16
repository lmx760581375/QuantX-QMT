"""Cross-sectional OOS model evaluation tests."""

import pandas as pd
import pytest

from quantx.core.research.evaluation import aggregate_seed_metrics, evaluate_predictions


def test_evaluation_reports_daily_stability_coverage_and_quantile_spread():
    rows = []
    for session in pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02"]):
        for index, instrument in enumerate(("A", "B", "C", "D", "E")):
            rows.append(
                {
                    "signal_time": session,
                    "instrument": instrument,
                    "score": float(index),
                    "theoretical_label": float(index) / 100,
                    "fold_id": f"fold-{session.year}",
                }
            )
    frame = pd.DataFrame(rows).set_index(["signal_time", "instrument"])

    metrics = evaluate_predictions(frame)

    assert metrics["sample_count"] == 15
    assert metrics["prediction_coverage"] == 1.0
    assert metrics["label_coverage"] == 1.0
    assert metrics["daily_rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["evaluated_session_count"] == 3
    assert metrics["quantile_returns"]["top_minus_bottom"] == pytest.approx(0.04)
    assert set(metrics["by_fold"]) == {"fold-2024", "fold-2025"}
    assert set(metrics["by_year"]) == {"2024", "2025"}


def test_evaluation_keeps_missing_labels_in_coverage_denominator():
    frame = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(["2024-01-02"] * 3),
            "instrument": ["A", "B", "C"],
            "score": [1.0, 2.0, 3.0],
            "theoretical_label": [0.01, None, 0.03],
        }
    ).set_index(["signal_time", "instrument"])

    metrics = evaluate_predictions(frame)

    assert metrics["sample_count"] == 2
    assert metrics["label_coverage"] == pytest.approx(2 / 3)
    assert metrics["joint_coverage"] == pytest.approx(2 / 3)


def test_multi_seed_metrics_report_mean_dispersion_and_worst_case():
    aggregate = aggregate_seed_metrics(
        {
            7: {"rank_ic": 0.03, "nested": {}},
            11: {"rank_ic": 0.06, "nested": {}},
            19: {"rank_ic": 0.00, "nested": {}},
        }
    )

    rank_ic = aggregate["metrics"]["rank_ic"]
    assert aggregate["seed_count"] == 3
    assert rank_ic["mean"] == pytest.approx(0.03)
    assert rank_ic["std"] == pytest.approx(0.0244948974)
    assert rank_ic["worst"] == 0.0
