"""HTML report rendering for trade pattern analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .features import train_feature_columns
from .io import write_json


def build_pattern_summary(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    model_results: Dict[str, Any],
    cluster_results: Dict[str, Any],
    figure_results: Dict[str, Any] | None = None,
    exit_rule_what_if: Dict[str, Any] | None = None,
    wait_opportunity_cost: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    target_comparison = _target_comparison(model_results)
    management_summary = _management_summary(labels)
    if isinstance(model_results, dict) and model_results.get("feature_scope"):
        management_summary["feature_scope"] = model_results.get("feature_scope")
    return {
        "sample_count": int(len(windows)),
        "date_start": str(windows["entry_date"].min()) if not windows.empty else None,
        "date_end": str(windows["entry_date"].max()) if not windows.empty else None,
        "outcome_counts": labels["outcome_label"].value_counts(dropna=False).to_dict() if not labels.empty else {},
        "behavior_counts": labels["behavior_label"].value_counts(dropna=False).to_dict() if not labels.empty else {},
        "opportunity_counts": labels["opportunity_label"].value_counts(dropna=False).to_dict() if "opportunity_label" in labels else {},
        "fade_counts": labels["fade_label"].value_counts(dropna=False).to_dict() if "fade_label" in labels else {},
        "recovery_counts": labels["recovery_label"].value_counts(dropna=False).to_dict() if "recovery_label" in labels else {},
        "management_summary": management_summary,
        "recovery_summary": _recovery_summary(labels),
        "recovery_feature_differences": _recovery_feature_differences(windows, labels),
        "recovery_feature_differences_by_target": _recovery_feature_differences_by_target(windows, labels),
        "failure_recovery_profiles": _failure_recovery_profiles(windows, labels),
        "failure_recovery_rule_search": _failure_recovery_rule_search(windows, labels),
        "recovery_rule_candidates": _recovery_rule_candidates(windows, labels),
        "fast_exit_rule_candidates": _fast_exit_rule_candidates(windows, labels),
        "failure_wait_diagnostics": _failure_wait_diagnostics(windows, labels),
        "failure_wait_cost_profiles": _failure_wait_cost_profiles(windows, labels, wait_opportunity_cost),
        "exit_review_summary": _exit_review_summary(labels),
        "avg_return": float(windows["return"].mean()) if not windows.empty else None,
        "avg_mfe": float(windows["max_favorable_excursion"].mean()) if not windows.empty else None,
        "avg_mae": float(windows["max_adverse_excursion"].mean()) if not windows.empty else None,
        "model_results": model_results,
        "target_comparison": target_comparison,
        "model_interpretation": _model_interpretation(target_comparison, management_summary),
        "exit_rule_what_if": exit_rule_what_if or {},
        "wait_opportunity_cost": wait_opportunity_cost or {},
        "cluster_results": cluster_results,
        "figure_results": figure_results or {},
    }


def _management_summary(labels: pd.DataFrame) -> Dict[str, Any]:
    if labels.empty:
        return {}
    summary: Dict[str, Any] = {
        "sample_count": int(len(labels)),
        "early_confirm_rate": _bool_mean(labels, "early_confirmed"),
        "early_fail_rate": _bool_mean(labels, "early_failed"),
        "opportunity_rate": _bool_mean(labels, "had_opportunity"),
        "fade_after_peak_rate": _bool_mean(labels, "faded_after_peak"),
        "efficient_capture_rate": _bool_mean(labels, "efficient_capture"),
        "recoverable_failure_rate": _failed_bool_mean(labels, "recoverable_failure"),
        "loss_reducible_after_exit_rate": _failed_bool_mean(labels, "loss_reducible_after_exit"),
        "close_turn_profitable_after_exit_rate": _failed_bool_mean(labels, "close_turn_profitable_after_exit"),
        "high_turn_profitable_after_exit_rate": _failed_bool_mean(labels, "high_turn_profitable_after_exit"),
        "avg_first_3d_return": _numeric_mean(labels, "first_3d_return"),
        "avg_first_5d_return": _numeric_mean(labels, "first_5d_return"),
        "avg_first_10d_return": _numeric_mean(labels, "first_10d_return"),
        "avg_drawdown_after_peak": _numeric_mean(labels, "drawdown_after_peak"),
    }
    if "had_opportunity" in labels and "efficient_capture" in labels:
        opportunity = labels[labels["had_opportunity"].astype(bool)]
        summary["opportunity_capture_rate"] = _bool_mean(opportunity, "efficient_capture") if not opportunity.empty else None
        summary["missed_opportunity_rate"] = 1.0 - summary["opportunity_capture_rate"] if summary["opportunity_capture_rate"] is not None else None
    return summary


def _recovery_summary(labels: pd.DataFrame) -> Dict[str, Any]:
    if labels.empty or "recovery_label" not in labels:
        return {}
    frame = labels.copy()
    failed = frame[frame["recovery_label"] != "not_failed"].copy()
    if failed.empty:
        return {"failed_count": 0, "by_exit_reason": []}
    for column in [
        "return",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
        "post_20d_close_from_entry",
        "post_20d_best_close_improvement",
        "post_20d_best_high_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_high_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    by_reason = []
    if "exit_reason" in failed:
        for reason, group in failed.groupby(failed["exit_reason"].fillna(""), dropna=False):
            by_reason.append({
                "exit_reason": str(reason),
                "sample_count": int(len(group)),
                "avg_return": _numeric_mean(group, "return"),
                "avg_post_20d_close_improvement": _numeric_mean(group, "post_20d_close_improvement"),
                "avg_post_20d_best_close_improvement": _numeric_mean(group, "post_20d_best_close_improvement"),
                "avg_post_20d_best_high_improvement": _numeric_mean(group, "post_20d_best_high_improvement"),
                "avg_days_to_close_profit": _numeric_mean(group, "post_20d_first_close_profit_day"),
                "avg_days_to_loss_reduction": _numeric_mean(group, "post_20d_first_close_improve_2pct_day"),
                "recoverable_failure_rate": _bool_mean(group, "recoverable_failure"),
                "loss_reducible_rate": _bool_mean(group, "loss_reducible_after_exit"),
                "close_turn_profitable_rate": _bool_mean(group, "close_turn_profitable_after_exit"),
                "high_turn_profitable_rate": _bool_mean(group, "high_turn_profitable_after_exit"),
            })
        by_reason.sort(key=lambda row: int(row.get("sample_count") or 0), reverse=True)

    return {
        "failed_count": int(len(failed)),
        "avg_failed_return": _numeric_mean(failed, "return"),
        "recoverable_failure_rate": _bool_mean(failed, "recoverable_failure"),
        "loss_reducible_rate": _bool_mean(failed, "loss_reducible_after_exit"),
        "close_turn_profitable_rate": _bool_mean(failed, "close_turn_profitable_after_exit"),
        "high_turn_profitable_rate": _bool_mean(failed, "high_turn_profitable_after_exit"),
        "avg_post_20d_close_improvement": _numeric_mean(failed, "post_20d_close_improvement"),
        "avg_post_20d_best_close_improvement": _numeric_mean(failed, "post_20d_best_close_improvement"),
        "avg_post_20d_best_high_improvement": _numeric_mean(failed, "post_20d_best_high_improvement"),
        "avg_days_to_close_profit": _numeric_mean(failed, "post_20d_first_close_profit_day"),
        "avg_days_to_loss_reduction": _numeric_mean(failed, "post_20d_first_close_improve_2pct_day"),
        "by_exit_reason": by_reason,
    }


def _failure_wait_diagnostics(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Summarize whether delayed exits are likely actionable or only intraday-looking."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "by_exit_reason": []}
    required = {"is_failed_trade", "close_turn_profitable_after_exit", "high_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "by_exit_reason": []}
    label_columns = [
        "sample_id",
        "is_failed_trade",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "loss_reducible_after_exit",
        "return",
        "exit_reason",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
        "post_20d_close_from_entry",
        "post_20d_best_close_improvement",
        "post_20d_best_high_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_high_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]
    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if failed.empty:
        return {"ok": False, "reason": "no failed trades", "by_exit_reason": []}
    numeric_columns = [
        "return",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
        "post_20d_close_from_entry",
        "post_20d_best_close_improvement",
        "post_20d_best_high_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_high_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]
    for column in numeric_columns:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    by_reason = []
    if "exit_reason" in failed:
        for reason, group in failed.groupby(failed["exit_reason"].fillna(""), dropna=False):
            by_reason.append(_wait_diagnostic_row(group, str(reason)))
        by_reason.sort(key=lambda row: int(row.get("sample_count") or 0), reverse=True)

    total = _wait_diagnostic_row(failed, "all_failed")
    total.update({
        "ok": True,
        "mode": "post_exit_actionability_diagnostic",
        "note": "Diagnostic only: post-exit columns are future review labels, not live features. Use this to avoid rules that chase intraday-only or slow rebound opportunities.",
        "by_exit_reason": by_reason,
    })
    return total


def _wait_diagnostic_row(frame: pd.DataFrame, group: str) -> Dict[str, Any]:
    high_only = _safe_bool(frame, "high_turn_profitable_after_exit") & ~_safe_bool(frame, "close_turn_profitable_after_exit")
    return {
        "group": group,
        "sample_count": int(len(frame)),
        "avg_return": _numeric_mean(frame, "return"),
        "loss_reducible_rate": _bool_mean(frame, "loss_reducible_after_exit"),
        "close_turn_profitable_rate": _bool_mean(frame, "close_turn_profitable_after_exit"),
        "high_turn_profitable_rate": _bool_mean(frame, "high_turn_profitable_after_exit"),
        "high_only_profit_rate": float(high_only.mean()) if len(high_only) else None,
        "close_profit_within_5d_rate": _day_threshold_rate(frame, "post_20d_first_close_profit_day", 5),
        "close_profit_within_10d_rate": _day_threshold_rate(frame, "post_20d_first_close_profit_day", 10),
        "loss_reduction_within_5d_rate": _day_threshold_rate(frame, "post_20d_first_close_improve_2pct_day", 5),
        "loss_reduction_within_10d_rate": _day_threshold_rate(frame, "post_20d_first_close_improve_2pct_day", 10),
        "avg_days_to_close_profit": _numeric_mean(frame, "post_20d_first_close_profit_day"),
        "avg_days_to_intraday_profit": _numeric_mean(frame, "post_20d_first_high_profit_day"),
        "avg_days_to_loss_reduction": _numeric_mean(frame, "post_20d_first_close_improve_2pct_day"),
        "avg_best_close_improvement": _numeric_mean(frame, "post_20d_best_close_improvement"),
        "avg_final_close_improvement": _numeric_mean(frame, "post_20d_close_improvement"),
        "avg_best_high_improvement": _numeric_mean(frame, "post_20d_best_high_improvement"),
        "avg_intraday_vs_close_gap": _numeric_mean(_with_gap(frame), "intraday_vs_close_gap"),
    }


def _recovery_feature_differences(features: pd.DataFrame, labels: pd.DataFrame) -> list[Dict[str, Any]]:
    """Compare exit-time features between recoverable and non-recoverable failures."""
    return _failure_feature_differences(features, labels, "recoverable_failure", "recoverable", "not_recoverable")


def _recovery_feature_differences_by_target(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, list[Dict[str, Any]]]:
    if features.empty or labels.empty:
        return {}
    targets = {
        "recoverable_failure": ("recoverable", "not_recoverable"),
        "loss_reducible_after_exit": ("loss_reducible", "not_loss_reducible"),
        "close_turn_profitable_after_exit": ("close_turn_profitable", "not_close_turn_profitable"),
        "high_turn_profitable_after_exit": ("high_turn_profitable", "not_high_turn_profitable"),
    }
    result: Dict[str, list[Dict[str, Any]]] = {}
    for column, (positive_name, negative_name) in targets.items():
        rows = _failure_feature_differences(features, labels, column, positive_name, negative_name)
        if rows:
            result[column] = rows
    return result


def _failure_feature_differences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
    positive_name: str,
    negative_name: str,
) -> list[Dict[str, Any]]:
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return []
    required = {"is_failed_trade", target_column}
    if not required.issubset(labels.columns):
        return []
    label_columns = ["sample_id", "is_failed_trade", target_column]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]
    data = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if data.empty or data[target_column].nunique() < 2:
        return []
    columns = train_feature_columns(data, feature_scope="trade_management")
    rows: list[Dict[str, Any]] = []
    positive = data[data[target_column].fillna(False).astype(bool)]
    negative = data[~data[target_column].fillna(False).astype(bool)]
    if len(positive) < 5 or len(negative) < 5:
        return []
    for column in columns:
        positive_values = pd.to_numeric(positive[column], errors="coerce").dropna()
        negative_values = pd.to_numeric(negative[column], errors="coerce").dropna()
        if len(positive_values) < 5 or len(negative_values) < 5:
            continue
        positive_mean = float(positive_values.mean())
        negative_mean = float(negative_values.mean())
        diff = positive_mean - negative_mean
        pooled_std = float(pd.concat([positive_values, negative_values]).std(ddof=0))
        standardized = diff / pooled_std if pooled_std > 1e-12 else 0.0
        rows.append({
            "target": target_column,
            "feature": column,
            "positive_name": positive_name,
            "negative_name": negative_name,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "recoverable_mean": positive_mean,
            "not_recoverable_mean": negative_mean,
            "diff": diff,
            "standardized_diff": standardized,
            "positive_count": int(len(positive_values)),
            "negative_count": int(len(negative_values)),
            "recoverable_count": int(len(positive_values)),
            "not_recoverable_count": int(len(negative_values)),
        })
    rows.sort(key=lambda row: abs(float(row.get("standardized_diff") or 0.0)), reverse=True)
    return rows[:30]


def _failure_recovery_profiles(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Profile failed trades by recovery target and exit reason."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "targets": []}
    required = {"is_failed_trade", "loss_reducible_after_exit", "close_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "targets": []}

    label_columns = [
        "sample_id",
        "is_failed_trade",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "return",
        "exit_reason",
        "post_20d_best_close_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]

    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if failed.empty:
        return {"ok": False, "reason": "no failed trades", "targets": []}
    for column in failed.columns:
        if pd.api.types.is_numeric_dtype(failed[column]) or pd.api.types.is_bool_dtype(failed[column]):
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    targets = [
        ("loss_reducible_after_exit", "缩亏型失败单", "short_loss_reduction"),
        ("close_turn_profitable_after_exit", "收盘可回本型失败单", "wait_for_close_profit"),
        ("high_turn_profitable_after_exit", "盘中可回本型失败单", "intraday_profit_review"),
    ]
    rows = []
    for target, label, intended_use in targets:
        if target not in failed or failed[target].nunique(dropna=True) < 2:
            continue
        rows.append({
            "target": target,
            "label": label,
            "intended_use": intended_use,
            "sample_count": int(len(failed)),
            "positive_count": int(failed[target].fillna(False).astype(bool).sum()),
            "positive_rate": _bool_mean(failed, target),
            "overall": _failure_profile_row(failed, target, "all_failed"),
            "by_exit_reason": _failure_profile_by_exit_reason(failed, target),
            "top_differences": _failure_feature_differences(features, labels, target, "positive", "negative")[:12],
        })
    return {
        "ok": True,
        "mode": "failed_trade_recovery_profile",
        "note": "Profiles use sell-time known features for positive/negative failed-trade groups; post-exit fields remain labels for review, not train features.",
        "failed_count": int(len(failed)),
        "targets": rows,
    }


def _failure_wait_cost_profiles(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    wait_opportunity_cost: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Profile failed trades where waiting is expensive and recovery is weak."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "targets": []}
    if not isinstance(wait_opportunity_cost, dict) or not wait_opportunity_cost.get("ok"):
        return {"ok": False, "reason": "missing wait opportunity cost samples", "targets": []}
    wait_samples = pd.DataFrame(wait_opportunity_cost.get("samples") or [])
    if wait_samples.empty or "sample_id" not in wait_samples:
        return {"ok": False, "reason": "empty wait opportunity cost samples", "targets": []}
    required = {"is_failed_trade", "loss_reducible_after_exit", "close_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "targets": []}

    label_columns = [
        "sample_id",
        "is_failed_trade",
        "recoverable_failure",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "return",
        "exit_reason",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]

    wait_columns = [
        "sample_id",
        "wait_days_observed",
        "candidate_days",
        "raw_candidate_days",
        "selected_candidate_count",
        "raw_candidate_count",
        "avg_position_count",
        "max_position_count",
        "near_full_days",
        "exit_reason",
    ]
    wait_columns = [column for column in wait_columns if column in wait_samples.columns]
    wait_frame = wait_samples[wait_columns].copy()
    if "exit_reason" in wait_frame:
        wait_frame = wait_frame.rename(columns={"exit_reason": "wait_exit_reason"})
    data = data.merge(wait_frame, on="sample_id", how="inner")
    if "exit_reason" not in data and "wait_exit_reason" in data:
        data["exit_reason"] = data["wait_exit_reason"]
    elif "exit_reason" in data and "wait_exit_reason" in data:
        data["exit_reason"] = data["exit_reason"].fillna(data["wait_exit_reason"])

    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if failed.empty:
        return {"ok": False, "reason": "no matched failed trades", "targets": []}

    numeric_columns = set(train_feature_columns(failed, feature_scope="trade_management")) | {
        "return",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "holding_days",
        "wait_days_observed",
        "candidate_days",
        "raw_candidate_days",
        "selected_candidate_count",
        "raw_candidate_count",
        "avg_position_count",
        "max_position_count",
        "near_full_days",
    }
    for column in numeric_columns:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    failed["high_wait_cost"] = _high_wait_cost_mask(failed)
    recoverable = _safe_bool(failed, "recoverable_failure")
    loss_reducible = _safe_bool(failed, "loss_reducible_after_exit")
    close_profitable = _safe_bool(failed, "close_turn_profitable_after_exit")
    high_cost = _safe_bool(failed, "high_wait_cost")
    failed["high_cost_not_loss_reducible"] = high_cost & ~loss_reducible
    failed["high_cost_not_close_profitable"] = high_cost & ~close_profitable
    failed["high_cost_not_recoverable"] = high_cost & ~recoverable
    failed["high_cost_not_worth_waiting"] = high_cost & ~(loss_reducible | close_profitable)
    failed["low_cost_recoverable"] = ~high_cost & (recoverable | loss_reducible | close_profitable)

    targets = [
        ("high_cost_not_worth_waiting", "high cost and not worth waiting", "sell_faster_or_keep_original_exit"),
        ("high_cost_not_loss_reducible", "high cost and not loss reducible", "avoid_loss_reduction_wait"),
        ("high_cost_not_close_profitable", "high cost and not close profitable", "avoid_profit_wait"),
        ("high_cost_not_recoverable", "high cost and not recoverable", "avoid_delayed_exit"),
        ("low_cost_recoverable", "low cost and recoverable", "wait_candidate_review"),
    ]
    target_rows = []
    for target, label, intended_use in targets:
        if target not in failed or failed[target].nunique(dropna=True) < 2:
            continue
        target_rows.append({
            "target": target,
            "label": label,
            "intended_use": intended_use,
            "sample_count": int(len(failed)),
            "positive_count": int(_safe_bool(failed, target).sum()),
            "positive_rate": _bool_mean(failed, target),
            "overall": _wait_cost_profile_row(failed, target, "all_failed"),
            "by_exit_reason": _wait_cost_profile_by_exit_reason(failed, target),
            "top_differences": _wait_cost_feature_differences(failed, target)[:12],
        })

    return {
        "ok": True,
        "mode": "failed_trade_wait_cost_profile",
        "note": "Diagnostic only: combines post-exit opportunity-cost proxy with post-exit recovery labels. Use top profiles to design sell rules, then validate by full backtest.",
        "failed_count": int(len(failed)),
        "high_wait_cost_count": int(_safe_bool(failed, "high_wait_cost").sum()),
        "high_wait_cost_rate": _bool_mean(failed, "high_wait_cost"),
        "high_cost_definition": "candidate_days >= 3 or selected_candidate_count >= 5 or near_full_days >= 3 within the wait window",
        "targets": target_rows,
    }


def _high_wait_cost_mask(frame: pd.DataFrame) -> pd.Series:
    candidate_days = pd.to_numeric(frame.get("candidate_days", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    selected_count = pd.to_numeric(frame.get("selected_candidate_count", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    near_full_days = pd.to_numeric(frame.get("near_full_days", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    return (candidate_days >= 3) | (selected_count >= 5) | (near_full_days >= 3)


def _wait_cost_profile_by_exit_reason(frame: pd.DataFrame, target: str) -> list[Dict[str, Any]]:
    if "exit_reason" not in frame:
        return []
    rows = []
    for reason, group in frame.groupby(frame["exit_reason"].fillna(""), dropna=False):
        if len(group) < 5:
            continue
        rows.append(_wait_cost_profile_row(group, target, str(reason)))
    rows.sort(key=lambda row: int(row.get("positive_count") or 0), reverse=True)
    return rows


def _wait_cost_profile_row(frame: pd.DataFrame, target: str, group: str) -> Dict[str, Any]:
    positive_mask = _safe_bool(frame, target)
    positive = frame[positive_mask]
    negative = frame[~positive_mask]
    return {
        "group": group,
        "sample_count": int(len(frame)),
        "positive_count": int(len(positive)),
        "positive_rate": _bool_mean(frame, target),
        "positive_avg_return": _numeric_mean(positive, "return"),
        "negative_avg_return": _numeric_mean(negative, "return"),
        "positive_avg_mae": _numeric_mean(positive, "max_adverse_excursion"),
        "negative_avg_mae": _numeric_mean(negative, "max_adverse_excursion"),
        "positive_avg_mfe": _numeric_mean(positive, "max_favorable_excursion"),
        "negative_avg_mfe": _numeric_mean(negative, "max_favorable_excursion"),
        "positive_avg_candidate_days": _numeric_mean(positive, "candidate_days"),
        "negative_avg_candidate_days": _numeric_mean(negative, "candidate_days"),
        "positive_avg_selected_candidate_count": _numeric_mean(positive, "selected_candidate_count"),
        "negative_avg_selected_candidate_count": _numeric_mean(negative, "selected_candidate_count"),
        "positive_avg_near_full_days": _numeric_mean(positive, "near_full_days"),
        "negative_avg_near_full_days": _numeric_mean(negative, "near_full_days"),
        "positive_loss_reducible_rate": _bool_mean(positive, "loss_reducible_after_exit"),
        "positive_close_profitable_rate": _bool_mean(positive, "close_turn_profitable_after_exit"),
        "positive_recoverable_rate": _bool_mean(positive, "recoverable_failure"),
    }


def _wait_cost_feature_differences(frame: pd.DataFrame, target: str) -> list[Dict[str, Any]]:
    if frame.empty or target not in frame or frame[target].nunique(dropna=True) < 2:
        return []
    columns = train_feature_columns(frame, feature_scope="trade_management")
    rows: list[Dict[str, Any]] = []
    positive = frame[_safe_bool(frame, target)]
    negative = frame[~_safe_bool(frame, target)]
    if len(positive) < 5 or len(negative) < 5:
        return []
    for column in columns:
        positive_values = pd.to_numeric(positive[column], errors="coerce").dropna()
        negative_values = pd.to_numeric(negative[column], errors="coerce").dropna()
        if len(positive_values) < 5 or len(negative_values) < 5:
            continue
        positive_mean = float(positive_values.mean())
        negative_mean = float(negative_values.mean())
        diff = positive_mean - negative_mean
        pooled_std = float(pd.concat([positive_values, negative_values]).std(ddof=0))
        rows.append({
            "target": target,
            "feature": column,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "diff": diff,
            "standardized_diff": diff / pooled_std if pooled_std > 1e-12 else 0.0,
            "positive_count": int(len(positive_values)),
            "negative_count": int(len(negative_values)),
        })
    rows.sort(key=lambda row: abs(float(row.get("standardized_diff") or 0.0)), reverse=True)
    return rows


def _failure_profile_by_exit_reason(frame: pd.DataFrame, target: str) -> list[Dict[str, Any]]:
    if "exit_reason" not in frame:
        return []
    rows = []
    for reason, group in frame.groupby(frame["exit_reason"].fillna(""), dropna=False):
        if len(group) < 5:
            continue
        rows.append(_failure_profile_row(group, target, str(reason)))
    rows.sort(key=lambda row: int(row.get("positive_count") or 0), reverse=True)
    return rows


def _failure_profile_row(frame: pd.DataFrame, target: str, group: str) -> Dict[str, Any]:
    positive_mask = _safe_bool(frame, target)
    positive = frame[positive_mask]
    negative = frame[~positive_mask]
    return {
        "group": group,
        "sample_count": int(len(frame)),
        "positive_count": int(len(positive)),
        "positive_rate": _bool_mean(frame, target),
        "positive_avg_return": _numeric_mean(positive, "return"),
        "negative_avg_return": _numeric_mean(negative, "return"),
        "positive_avg_mae": _numeric_mean(positive, "max_adverse_excursion"),
        "negative_avg_mae": _numeric_mean(negative, "max_adverse_excursion"),
        "positive_avg_mfe": _numeric_mean(positive, "max_favorable_excursion"),
        "negative_avg_mfe": _numeric_mean(negative, "max_favorable_excursion"),
        "positive_avg_holding_days": _numeric_mean(positive, "holding_days"),
        "negative_avg_holding_days": _numeric_mean(negative, "holding_days"),
        "positive_avg_post_best_close_improvement": _numeric_mean(positive, "post_20d_best_close_improvement"),
        "negative_avg_post_best_close_improvement": _numeric_mean(negative, "post_20d_best_close_improvement"),
        "positive_avg_days_to_close_profit": _numeric_mean(positive, "post_20d_first_close_profit_day"),
        "positive_avg_days_to_loss_reduction": _numeric_mean(positive, "post_20d_first_close_improve_2pct_day"),
    }


def _failure_recovery_rule_search(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Search compact sell-time feature filters for recovery-prone failed trades."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "targets": []}
    required = {"is_failed_trade", "loss_reducible_after_exit", "close_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "targets": []}

    label_columns = [
        "sample_id",
        "is_failed_trade",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "return",
        "exit_reason",
        "post_20d_best_close_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]

    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if len(failed) < 12:
        return {"ok": False, "reason": "not enough failed trades", "targets": []}
    numeric_columns = set(train_feature_columns(failed, feature_scope="trade_management")) | {
        "return",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "holding_days",
        "post_20d_best_close_improvement",
        "post_20d_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_close_improve_2pct_day",
    }
    for column in numeric_columns:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    targets = [
        ("loss_reducible_after_exit", "loss_reduction"),
        ("close_turn_profitable_after_exit", "close_profit"),
    ]
    target_rows = []
    for target, intended_use in targets:
        if target not in failed or failed[target].nunique(dropna=True) < 2:
            continue
        target_rows.append({
            "target": target,
            "intended_use": intended_use,
            "baseline_rate": _bool_mean(failed, target),
            "failed_count": int(len(failed)),
            "rules": _search_target_rules(failed, target),
        })
    return {
        "ok": True,
        "mode": "sell_time_rule_search_on_post_exit_labels",
        "note": "Diagnostic only: rules use sell-time known features, but are scored with post-exit labels. Treat top rows as candidates for real portfolio backtests, not live rules.",
        "failed_count": int(len(failed)),
        "targets": target_rows,
    }


def _search_target_rules(frame: pd.DataFrame, target: str) -> list[Dict[str, Any]]:
    candidate_masks = _rule_search_candidates(frame)
    rows: list[Dict[str, Any]] = []
    baseline_rate = _bool_mean(frame, target) or 0.0
    failed_count = int(len(frame))
    seen: set[tuple[str, ...]] = set()
    for size in [1, 2, 3]:
        if size == 1:
            combinations = [tuple([candidate]) for candidate in candidate_masks]
        elif size == 2:
            combinations = [
                (candidate_masks[i], candidate_masks[j])
                for i in range(len(candidate_masks))
                for j in range(i + 1, len(candidate_masks))
            ]
        else:
            combinations = _seeded_three_way_combinations(candidate_masks)
        for combo in combinations:
            names = tuple(item["name"] for item in combo)
            if names in seen:
                continue
            seen.add(names)
            mask = pd.Series(True, index=frame.index)
            for item in combo:
                mask &= item["mask"].reindex(frame.index).fillna(False).astype(bool)
            row = _rule_search_row(frame, mask, target, names, baseline_rate, failed_count)
            if row is not None:
                rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("lift") or 0.0),
            float(row.get("positive_rate") or 0.0),
            int(row.get("positive_count") or 0),
        ),
        reverse=True,
    )
    return rows[:20]


def _seeded_three_way_combinations(candidates: list[Dict[str, Any]]) -> list[tuple[Dict[str, Any], ...]]:
    seed_prefixes = ("exit_reason_", "pre_", "hold_", "return_", "mae_")
    groups: dict[str, list[Dict[str, Any]]] = {prefix: [] for prefix in seed_prefixes}
    for candidate in candidates:
        name = str(candidate.get("name") or "")
        for prefix in seed_prefixes:
            if name.startswith(prefix):
                groups[prefix].append(candidate)
                break
    combos: list[tuple[Dict[str, Any], ...]] = []
    for exit_candidate in groups["exit_reason_"]:
        for structure_candidate in groups["pre_"][:16]:
            for state_candidate in [*groups["hold_"][:8], *groups["return_"][:8], *groups["mae_"][:8]]:
                combos.append((exit_candidate, structure_candidate, state_candidate))
    return combos


def _rule_search_candidates(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    for column in ["exit_reason_stop_loss", "exit_reason_time_stop"]:
        if column in frame:
            candidates.append({"name": column, "mask": _safe_bool(frame, column), "condition": column})
    thresholds = {
        "pre_40_above_ma20_ratio": [(">=", 0.60), (">=", 0.65), (">=", 0.70), (">=", 0.75)],
        "pre_20_above_ma20_ratio": [(">=", 0.80), (">=", 0.90), (">=", 1.00)],
        "pre_10_long_lower_count": [(">=", 1), (">=", 2), (">=", 3)],
        "pre_5_return": [("<=", 0.00), ("<=", 0.01), ("<=", 0.02)],
        "pre_10_return": [("<=", 0.04), ("<=", 0.06), ("<=", 0.08)],
        "pre_20_max_drawdown": [(">=", -0.06), (">=", -0.05), (">=", -0.04)],
        "hold_first_3d_return": [(">=", -0.02), (">=", -0.01), (">=", 0.00)],
        "hold_first_10d_return": [(">=", -0.04), (">=", -0.03), (">=", -0.02)],
        "return": [(">=", -0.11), (">=", -0.08), (">=", -0.05), ("<=", -0.08)],
        "max_adverse_excursion": [(">=", -0.11), (">=", -0.10), (">=", -0.09)],
        "max_favorable_excursion": [(">=", 0.04), (">=", 0.06), (">=", 0.08)],
    }
    aliases = {
        "return": "return",
        "max_adverse_excursion": "mae",
        "max_favorable_excursion": "mfe",
    }
    for column, specs in thresholds.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        for op, threshold in specs:
            if op == ">=":
                mask = values >= threshold
                label = f"{aliases.get(column, column)}_ge_{_threshold_id(threshold)}"
                condition = f"{column} >= {threshold:g}"
            else:
                mask = values <= threshold
                label = f"{aliases.get(column, column)}_le_{_threshold_id(threshold)}"
                condition = f"{column} <= {threshold:g}"
            candidates.append({"name": label, "mask": mask.fillna(False), "condition": condition})
    return candidates


def _threshold_id(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _rule_search_row(
    frame: pd.DataFrame,
    mask: pd.Series,
    target: str,
    condition_names: tuple[str, ...],
    baseline_rate: float,
    failed_count: int,
) -> Dict[str, Any] | None:
    mask = mask.reindex(frame.index).fillna(False).astype(bool)
    group = frame[mask]
    sample_count = int(len(group))
    min_sample_count = min(12, max(6, int(failed_count * 0.05)))
    if sample_count < min_sample_count:
        return None
    coverage = sample_count / failed_count if failed_count else 0.0
    if coverage > 0.75:
        return None
    positive_rate = _bool_mean(group, target)
    if positive_rate is None:
        return None
    lift = positive_rate - baseline_rate
    if lift < 0.04:
        return None
    return {
        "rule_id": "__and__".join(condition_names),
        "conditions": list(condition_names),
        "sample_count": sample_count,
        "coverage": coverage,
        "positive_count": int(_safe_bool(group, target).sum()),
        "positive_rate": positive_rate,
        "baseline_rate": baseline_rate,
        "lift": lift,
        "avg_return": _numeric_mean(group, "return"),
        "avg_mae": _numeric_mean(group, "max_adverse_excursion"),
        "avg_mfe": _numeric_mean(group, "max_favorable_excursion"),
        "avg_post_best_close_improvement": _numeric_mean(group, "post_20d_best_close_improvement"),
        "avg_days_to_loss_reduction": _numeric_mean(group, "post_20d_first_close_improve_2pct_day"),
        "avg_days_to_close_profit": _numeric_mean(group, "post_20d_first_close_profit_day"),
        "exit_reason_counts": group["exit_reason"].fillna("").value_counts().to_dict() if "exit_reason" in group else {},
    }


def _exit_review_summary(labels: pd.DataFrame) -> Dict[str, Any]:
    if labels.empty:
        return {}
    frame = labels.copy()
    for column in [
        "return",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "first_3d_return",
        "first_5d_return",
        "first_10d_return",
        "drawdown_after_peak",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
        "post_20d_close_from_entry",
        "post_20d_best_high_improvement",
        "post_20d_close_improvement",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    summary: Dict[str, Any] = {
        "sample_count": int(len(frame)),
        "avg_return": _numeric_mean(frame, "return"),
        "median_return": _numeric_median(frame, "return"),
        "avg_mfe": _numeric_mean(frame, "max_favorable_excursion"),
        "avg_mae": _numeric_mean(frame, "max_adverse_excursion"),
        "avg_first_3d_return": _numeric_mean(frame, "first_3d_return"),
        "avg_first_5d_return": _numeric_mean(frame, "first_5d_return"),
        "avg_first_10d_return": _numeric_mean(frame, "first_10d_return"),
        "avg_drawdown_after_peak": _numeric_mean(frame, "drawdown_after_peak"),
        "avg_post_20d_close_improvement": _numeric_mean(frame, "post_20d_close_improvement"),
        "avg_post_20d_best_high_improvement": _numeric_mean(frame, "post_20d_best_high_improvement"),
        "opportunity_rate_4pct": _threshold_rate(frame, "max_favorable_excursion", 0.04),
        "opportunity_rate_8pct": _threshold_rate(frame, "max_favorable_excursion", 0.08),
        "opportunity_rate_12pct": _threshold_rate(frame, "max_favorable_excursion", 0.12),
        "opportunity_rate_18pct": _threshold_rate(frame, "max_favorable_excursion", 0.18),
        "hard_loss_rate_5pct": _threshold_rate(frame, "max_adverse_excursion", -0.05, less_equal=True),
        "hard_loss_rate_8pct": _threshold_rate(frame, "max_adverse_excursion", -0.08, less_equal=True),
        "ended_negative_after_8pct_opportunity_rate": _conditional_rate(
            frame,
            condition=(frame["max_favorable_excursion"] >= 0.08) if "max_favorable_excursion" in frame else pd.Series(False, index=frame.index),
            success=(frame["return"] <= 0.0) if "return" in frame else pd.Series(False, index=frame.index),
        ),
        "bins": [],
    }
    if "max_favorable_excursion" not in frame:
        return summary
    bins = [
        ("no_4pct_opportunity", None, 0.04),
        ("mfe_4_8pct", 0.04, 0.08),
        ("mfe_8_12pct", 0.08, 0.12),
        ("mfe_12_18pct", 0.12, 0.18),
        ("mfe_18pct_plus", 0.18, None),
    ]
    for label, lower, upper in bins:
        mask = pd.Series(True, index=frame.index)
        if lower is not None:
            mask &= frame["max_favorable_excursion"] >= lower
        if upper is not None:
            mask &= frame["max_favorable_excursion"] < upper
        group = frame[mask]
        summary["bins"].append({
            "bucket": label,
            "sample_count": int(len(group)),
            "avg_return": _numeric_mean(group, "return"),
            "median_return": _numeric_median(group, "return"),
            "avg_mfe": _numeric_mean(group, "max_favorable_excursion"),
            "avg_mae": _numeric_mean(group, "max_adverse_excursion"),
            "avg_drawdown_after_peak": _numeric_mean(group, "drawdown_after_peak"),
            "success_rate": _bool_mean(group, "binary_success") if "binary_success" in group else None,
            "efficient_capture_rate": _bool_mean(group, "efficient_capture") if "efficient_capture" in group else None,
            "ended_negative_rate": _threshold_rate(group, "return", 0.0, less_equal=True),
        })
    return summary


def _fast_exit_rule_candidates(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Find runtime-known contexts for exiting weak failed trades faster."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "rules": []}
    required = {"is_failed_trade", "loss_reducible_after_exit", "close_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "rules": []}
    label_columns = [
        "sample_id",
        "is_failed_trade",
        "recoverable_failure",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "post_20d_close_improvement",
        "post_20d_best_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]

    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if failed.empty:
        return {"ok": False, "reason": "no failed trades", "rules": []}
    for column in [
        "return",
        "holding_days",
        "hold_first_3d_return",
        "hold_first_10d_return",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "post_20d_close_improvement",
        "post_20d_best_close_improvement",
        "post_20d_first_close_profit_day",
        "post_20d_first_close_improve_2pct_day",
    ]:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    rows = []
    for min_days in (16, 18, 20):
        for max_peak in (0.01, 0.02, 0.03):
            for min_trough in (-0.06, -0.07, -0.08):
                mask = (
                    (failed.get("holding_days", pd.Series(index=failed.index, dtype=float)) > min_days)
                    & (failed.get("return", pd.Series(index=failed.index, dtype=float)) < 0)
                    & (failed.get("hold_first_3d_return", pd.Series(index=failed.index, dtype=float)) < -0.01)
                    & (failed.get("hold_first_10d_return", pd.Series(index=failed.index, dtype=float)) < -0.04)
                    & (failed.get("max_favorable_excursion", pd.Series(index=failed.index, dtype=float)) < max_peak)
                    & (failed.get("max_adverse_excursion", pd.Series(index=failed.index, dtype=float)) < min_trough)
                )
                row = _fast_exit_rule_row(failed, mask, min_days=min_days, max_peak=max_peak, min_trough=min_trough)
                if row is not None:
                    rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("do_not_wait_score") or 0.0),
            -float(row.get("recoverable_failure_rate") or 0.0),
            float(row.get("avg_loss_severity") or 0.0),
        ),
        reverse=True,
    )
    return {
        "ok": True,
        "mode": "runtime_fast_exit_context_search",
        "note": "Candidate generator only: rules use runtime-known holding context and post-exit labels for diagnostics. Promote a rule only after a full portfolio backtest.",
        "failed_count": int(len(failed)),
        "rules": rows[:20],
    }


def _fast_exit_rule_row(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    min_days: int,
    max_peak: float,
    min_trough: float,
) -> Dict[str, Any] | None:
    mask = mask.reindex(frame.index).fillna(False).astype(bool)
    group = frame[mask]
    sample_count = int(len(group))
    if sample_count == 0:
        return None
    failed_count = int(len(frame))
    coverage = sample_count / failed_count if failed_count else 0.0
    if coverage > 0.20:
        return None
    recoverable_rate = _bool_mean(group, "recoverable_failure") or 0.0
    loss_reducible_rate = _bool_mean(group, "loss_reducible_after_exit") or 0.0
    close_profit_rate = _bool_mean(group, "close_turn_profitable_after_exit") or 0.0
    high_profit_rate = _bool_mean(group, "high_turn_profitable_after_exit") or 0.0
    do_not_wait_score = (1.0 - max(loss_reducible_rate, close_profit_rate)) * (1.0 - min(recoverable_rate, 1.0))
    return {
        "rule_id": f"fast_exit_d{min_days}_p{int(round(max_peak * 100))}_t{int(round(abs(min_trough) * 100))}",
        "description": f"holding_days>{min_days}, early weak, peak<{max_peak:.0%}, trough<{min_trough:.0%}",
        "yaml_when": (
            f"holding_days > {min_days} and pnl_pct < 0 and "
            "hold_first_3d_return < -0.01 and hold_first_10d_return < -0.04 and "
            f"peak_pnl_pct < {max_peak:g} and trough_pnl_pct < {min_trough:g}"
        ),
        "sample_count": sample_count,
        "coverage": coverage,
        "avg_return": _numeric_mean(group, "return"),
        "avg_loss_severity": abs(_numeric_mean(group, "return") or 0.0),
        "avg_mae": _numeric_mean(group, "max_adverse_excursion"),
        "avg_mfe": _numeric_mean(group, "max_favorable_excursion"),
        "recoverable_failure_rate": recoverable_rate,
        "loss_reducible_rate": loss_reducible_rate,
        "close_turn_profitable_rate": close_profit_rate,
        "high_turn_profitable_rate": high_profit_rate,
        "avg_post_close_improvement": _numeric_mean(group, "post_20d_close_improvement"),
        "avg_post_best_close_improvement": _numeric_mean(group, "post_20d_best_close_improvement"),
        "avg_days_to_close_profit": _numeric_mean(group, "post_20d_first_close_profit_day"),
        "avg_days_to_loss_reduction": _numeric_mean(group, "post_20d_first_close_improve_2pct_day"),
        "do_not_wait_score": do_not_wait_score,
    }


def _recovery_rule_candidates(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
    """Post-exit diagnostics for candidate delayed-exit filters."""
    if features.empty or labels.empty or "sample_id" not in features or "sample_id" not in labels:
        return {"ok": False, "reason": "empty features or labels", "rules": []}
    required = {"is_failed_trade", "loss_reducible_after_exit", "close_turn_profitable_after_exit", "high_turn_profitable_after_exit"}
    if not required.issubset(labels.columns):
        return {"ok": False, "reason": "missing recovery labels", "rules": []}
    label_columns = [
        "sample_id",
        "is_failed_trade",
        "recoverable_failure",
        "loss_reducible_after_exit",
        "close_turn_profitable_after_exit",
        "high_turn_profitable_after_exit",
        "post_20d_close_improvement",
        "post_20d_best_high_improvement",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
    ]
    label_columns = [column for column in label_columns if column in labels.columns]
    data = features.merge(labels[label_columns], on="sample_id", how="inner", suffixes=("", "_label"))
    for column in label_columns:
        if column == "sample_id":
            continue
        label_column = f"{column}_label"
        if label_column not in data:
            continue
        if column in data:
            data[column] = data[column].combine_first(data[label_column])
        else:
            data[column] = data[label_column]
    failed = data[data["is_failed_trade"].fillna(False).astype(bool)].copy()
    if failed.empty:
        return {"ok": False, "reason": "no failed trades", "rules": []}
    for column in [
        "return",
        "max_adverse_excursion",
        "hold_min_return",
        "hold_first_10d_return",
        "pre_40_above_ma20_ratio",
        "pre_10_long_lower_count",
        "post_20d_close_improvement",
        "post_20d_best_high_improvement",
        "post_20d_best_close_from_entry",
        "post_20d_best_high_from_entry",
    ]:
        if column in failed:
            failed[column] = pd.to_numeric(failed[column], errors="coerce")

    rules = [
        {
            "rule_id": "time_stop_shallow_wait_profit",
            "description": "time_stop, shallow loss, mild MAE, first 10d not broken",
            "intended_use": "wait_for_close_profit",
            "mask": _safe_bool(failed, "exit_reason_time_stop")
            & (failed.get("return", pd.Series(index=failed.index, dtype=float)) > -0.07)
            & (failed.get("max_adverse_excursion", pd.Series(index=failed.index, dtype=float)) > -0.10)
            & (failed.get("hold_first_10d_return", pd.Series(index=failed.index, dtype=float)) > -0.04),
        },
        {
            "rule_id": "time_stop_conservative_wait_profit",
            "description": "time_stop, loss within 5%, MAE within 9%",
            "intended_use": "wait_for_close_profit",
            "mask": _safe_bool(failed, "exit_reason_time_stop")
            & (failed.get("return", pd.Series(index=failed.index, dtype=float)) > -0.05)
            & (failed.get("max_adverse_excursion", pd.Series(index=failed.index, dtype=float)) > -0.09),
        },
        {
            "rule_id": "stop_loss_rebound_loss_reduction",
            "description": "stop_loss, prior MA20 structure intact, recent lower-shadow support",
            "intended_use": "loss_reduction",
            "mask": _safe_bool(failed, "exit_reason_stop_loss")
            & (failed.get("pre_40_above_ma20_ratio", pd.Series(index=failed.index, dtype=float)) >= 0.65)
            & (failed.get("pre_10_long_lower_count", pd.Series(index=failed.index, dtype=float)) >= 1),
        },
        {
            "rule_id": "deep_stop_loss_do_not_wait",
            "description": "deep stop_loss or weak prior structure, keep original risk control",
            "intended_use": "avoid_waiting",
            "mask": _safe_bool(failed, "exit_reason_stop_loss")
            & (
                (failed.get("return", pd.Series(index=failed.index, dtype=float)) <= -0.11)
                | (failed.get("max_adverse_excursion", pd.Series(index=failed.index, dtype=float)) <= -0.11)
                | (failed.get("pre_40_above_ma20_ratio", pd.Series(index=failed.index, dtype=float)) < 0.60)
            ),
        },
    ]
    rows = [_candidate_rule_row(failed, rule["mask"], rule["rule_id"], rule["description"], rule["intended_use"]) for rule in rules]
    return {
        "ok": True,
        "mode": "post_exit_label_diagnostic",
        "note": "Diagnostic only: uses post-exit labels to evaluate candidate filters; validate with full portfolio backtests before changing strategy rules.",
        "failed_count": int(len(failed)),
        "rules": rows,
    }


def _safe_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _candidate_rule_row(frame: pd.DataFrame, mask: pd.Series, rule_id: str, description: str, intended_use: str) -> Dict[str, Any]:
    mask = mask.reindex(frame.index).fillna(False).astype(bool)
    group = frame[mask]
    failed_count = int(len(frame))
    return {
        "rule_id": rule_id,
        "description": description,
        "intended_use": intended_use,
        "sample_count": int(len(group)),
        "coverage": float(len(group) / failed_count) if failed_count else None,
        "avg_return": _numeric_mean(group, "return"),
        "avg_mae": _numeric_mean(group, "max_adverse_excursion"),
        "recoverable_failure_rate": _bool_mean(group, "recoverable_failure"),
        "loss_reducible_rate": _bool_mean(group, "loss_reducible_after_exit"),
        "close_turn_profitable_rate": _bool_mean(group, "close_turn_profitable_after_exit"),
        "high_turn_profitable_rate": _bool_mean(group, "high_turn_profitable_after_exit"),
        "avg_post_20d_close_improvement": _numeric_mean(group, "post_20d_close_improvement"),
        "avg_post_20d_best_high_improvement": _numeric_mean(group, "post_20d_best_high_improvement"),
        "avg_post_20d_best_close_from_entry": _numeric_mean(group, "post_20d_best_close_from_entry"),
        "avg_post_20d_best_high_from_entry": _numeric_mean(group, "post_20d_best_high_from_entry"),
        "avg_days_to_close_profit": _numeric_mean(group, "post_20d_first_close_profit_day"),
        "avg_days_to_loss_reduction": _numeric_mean(group, "post_20d_first_close_improve_2pct_day"),
    }


def _bool_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    return float(frame[column].fillna(False).astype(bool).mean())


def _failed_bool_mean(frame: pd.DataFrame, column: str) -> float | None:
    if "recovery_label" not in frame:
        return None
    failed = frame[frame["recovery_label"] != "not_failed"]
    return _bool_mean(failed, column) if not failed.empty else None


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def _numeric_median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").median()
    return float(value) if pd.notna(value) else None


def _threshold_rate(frame: pd.DataFrame, column: str, threshold: float, less_equal: bool = False) -> float | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    mask = values <= threshold if less_equal else values >= threshold
    return float(mask.mean())


def _conditional_rate(frame: pd.DataFrame, condition: pd.Series, success: pd.Series) -> float | None:
    if frame.empty:
        return None
    condition = condition.reindex(frame.index).fillna(False).astype(bool)
    success = success.reindex(frame.index).fillna(False).astype(bool)
    denominator = int(condition.sum())
    if denominator == 0:
        return None
    return float((condition & success).sum() / denominator)


def _day_threshold_rate(frame: pd.DataFrame, column: str, max_day: int) -> float | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float((values.notna() & (values <= max_day)).mean())


def _with_gap(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "post_20d_best_high_improvement" in result and "post_20d_best_close_improvement" in result:
        high = pd.to_numeric(result["post_20d_best_high_improvement"], errors="coerce")
        close = pd.to_numeric(result["post_20d_best_close_improvement"], errors="coerce")
        result["intraday_vs_close_gap"] = high - close
    return result


def _target_comparison(model_results: Dict[str, Any]) -> list[Dict[str, Any]]:
    targets = model_results.get("targets") if isinstance(model_results, dict) else None
    if not isinstance(targets, dict):
        targets = {"success": model_results} if isinstance(model_results, dict) and model_results.get("models") else {}
    rows: list[Dict[str, Any]] = []
    for target_key, result in targets.items():
        if not isinstance(result, dict):
            continue
        baseline = result.get("majority_baseline") or {}
        models = result.get("models") or {}
        ok_models = {name: metrics for name, metrics in models.items() if isinstance(metrics, dict) and metrics.get("ok")}
        best_auc_name = None
        best_auc = None
        best_delta_name = None
        best_delta = None
        for name, metrics in ok_models.items():
            auc = metrics.get("auc")
            if auc is not None and (best_auc is None or float(auc) > best_auc):
                best_auc_name = name
                best_auc = float(auc)
            delta = metrics.get("accuracy_vs_baseline")
            if delta is not None and (best_delta is None or float(delta) > best_delta):
                best_delta_name = name
                best_delta = float(delta)
        rows.append({
            "target": target_key,
            "label": result.get("target") or target_key,
            "sample_count": result.get("sample_count"),
            "positive_rate": result.get("positive_rate"),
            "baseline_label": baseline.get("label"),
            "baseline_accuracy": baseline.get("accuracy"),
            "best_auc_model": best_auc_name,
            "best_auc": best_auc,
            "best_accuracy_delta_model": best_delta_name,
            "best_accuracy_vs_baseline": best_delta,
        })
    return rows


def _model_interpretation(target_rows: list[Dict[str, Any]], management: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows_by_target = {str(row.get("target")): row for row in target_rows}
    interpretations: list[Dict[str, Any]] = []
    for target in (
        "success",
        "opportunity",
        "fade_after_peak",
        "efficient_capture",
        "recoverable_failure",
        "loss_reducible",
        "close_turn_profitable",
    ):
        row = rows_by_target.get(target)
        if not row:
            continue
        auc = row.get("best_auc")
        delta = row.get("best_accuracy_vs_baseline")
        auc_value = float(auc) if auc is not None else None
        delta_value = float(delta) if delta is not None else None
        signal = _signal_strength(auc_value, delta_value)
        interpretations.append({
            "target": target,
            "signal_strength": signal,
            "best_auc": auc_value,
            "best_accuracy_vs_baseline": delta_value,
            "implication": _target_implication(target, signal, management),
        })
    return interpretations


def _signal_strength(auc: float | None, accuracy_delta: float | None) -> str:
    if auc is None and accuracy_delta is None:
        return "unknown"
    auc_value = auc if auc is not None else 0.5
    delta_value = accuracy_delta if accuracy_delta is not None else 0.0
    if auc_value >= 0.62 and delta_value >= 0.03:
        return "usable"
    if auc_value >= 0.56 or delta_value >= 0.05:
        return "weak_but_present"
    if auc_value <= 0.52 and delta_value <= 0.0:
        return "not_learned"
    return "weak"


def _target_implication(target: str, signal: str, management: Dict[str, Any]) -> str:
    feature_scope = str(management.get("feature_scope") or "")
    if target == "success":
        if feature_scope == "trade_management":
            return "sell_time_diagnostic_only_not_buy_filter"
        if signal in {"not_learned", "weak"}:
            return "buy_filter_not_supported"
        return "buy_filter_candidate"
    if target == "opportunity":
        if signal in {"usable", "weak_but_present"}:
            return "trade_management_candidate"
        return "opportunity_signal_weak"
    if target == "fade_after_peak":
        if signal in {"not_learned", "weak"} and (management.get("fade_after_peak_rate") or 0) > 0.5:
            return "exit_rules_more_important_than_pre_entry_filter"
        return "fade_filter_candidate"
    if target == "efficient_capture":
        if signal in {"not_learned", "weak"}:
            return "capture_needs_exit_logic"
        return "capture_filter_candidate"
    if target == "recoverable_failure":
        if signal in {"usable", "weak_but_present"}:
            return "delayed_exit_filter_candidate"
        return "recoverable_failure_signal_weak"
    if target == "loss_reducible":
        if signal in {"usable", "weak_but_present"}:
            return "loss_reduction_exit_candidate"
        return "loss_reduction_signal_weak"
    if target == "close_turn_profitable":
        if signal in {"usable", "weak_but_present"}:
            return "wait_for_close_profit_candidate"
        return "close_profit_signal_weak"
    return "review_required"


def write_report(output_dir: Path, summary: Dict[str, Any], clusters: pd.DataFrame) -> Path:
    """Write a compact HTML report for quick inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pattern_summary.json", summary)
    report_path = output_dir / "report.html"
    cluster_rows = summary.get("cluster_results", {}).get("clusters", []) if isinstance(summary.get("cluster_results"), dict) else []
    report_path.write_text(_html(summary, cluster_rows, clusters), encoding="utf-8")
    return report_path


def _html(summary: Dict[str, Any], cluster_rows: list[Dict[str, Any]], clusters: pd.DataFrame) -> str:
    rows = []
    for cluster in cluster_rows:
        figure_html = _cluster_figure_html(summary, cluster.get("cluster_id"))
        rows.append(
            "<tr>"
            f"<td>{cluster.get('cluster_id')}</td>"
            f"<td>{cluster.get('description')}</td>"
            f"<td>{cluster.get('sample_count')}</td>"
            f"<td>{_pct(cluster.get('success_rate'))}</td>"
            f"<td>{_pct(cluster.get('avg_return'))}</td>"
            f"<td>{_pct(cluster.get('avg_mae'))}</td>"
            f"<td>{figure_html}</td>"
            "</tr>"
        )
    model_rows = []
    baseline = summary.get("model_results", {}).get("majority_baseline", {}) if isinstance(summary.get("model_results"), dict) else {}
    for name, result in (summary.get("model_results", {}).get("models", {}) or {}).items():
        model_rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{result.get('ok')}</td>"
            f"<td>{_num(result.get('auc'))}</td>"
            f"<td>{_pct(result.get('accuracy'))}</td>"
            f"<td>{_pct(result.get('accuracy_vs_baseline'))}</td>"
            f"<td>{_num(result.get('precision'))}</td>"
            f"<td>{_num(result.get('recall'))}</td>"
            "</tr>"
        )
    target_rows = []
    for row in summary.get("target_comparison") or []:
        target_rows.append(
            "<tr>"
            f"<td>{row.get('target')}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('positive_rate'))}</td>"
            f"<td>{row.get('baseline_label') or ''}</td>"
            f"<td>{_pct(row.get('baseline_accuracy'))}</td>"
            f"<td>{row.get('best_auc_model') or ''}</td>"
            f"<td>{_num(row.get('best_auc'))}</td>"
            f"<td>{row.get('best_accuracy_delta_model') or ''}</td>"
            f"<td>{_pct(row.get('best_accuracy_vs_baseline'))}</td>"
            "</tr>"
        )
    interpretation_rows = []
    for row in summary.get("model_interpretation") or []:
        interpretation_rows.append(
            "<tr>"
            f"<td>{row.get('target')}</td>"
            f"<td>{row.get('signal_strength')}</td>"
            f"<td>{row.get('implication')}</td>"
            f"<td>{_num(row.get('best_auc'))}</td>"
            f"<td>{_pct(row.get('best_accuracy_vs_baseline'))}</td>"
            "</tr>"
        )
    exit_review = summary.get("exit_review_summary") or {}
    exit_rows = []
    for row in exit_review.get("bins") or []:
        exit_rows.append(
            "<tr>"
            f"<td>{row.get('bucket')}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('success_rate'))}</td>"
            f"<td>{_pct(row.get('efficient_capture_rate'))}</td>"
            f"<td>{_pct(row.get('ended_negative_rate'))}</td>"
            f"<td>{_pct(row.get('avg_drawdown_after_peak'))}</td>"
            "</tr>"
        )
    what_if = summary.get("exit_rule_what_if") or {}
    what_if_rows = []
    for row in what_if.get("rules") or []:
        what_if_rows.append(
            "<tr>"
            f"<td>{row.get('description') or row.get('rule_id')}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('avg_delta_return'))}</td>"
            f"<td>{_pct(row.get('improved_rate'))}</td>"
            f"<td>{_pct(row.get('success_rate'))}</td>"
            f"<td>{_pct(row.get('negative_rate'))}</td>"
            f"<td>{_num(row.get('avg_holding_days'))}</td>"
            "</tr>"
        )
    management = summary.get("management_summary") or {}
    recovery = summary.get("recovery_summary") or {}
    recovery_reason_rows = []
    for row in recovery.get("by_exit_reason") or []:
        recovery_reason_rows.append(
            "<tr>"
            f"<td>{row.get('exit_reason') or '(none)'}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('recoverable_failure_rate'))}</td>"
            f"<td>{_pct(row.get('loss_reducible_rate'))}</td>"
            f"<td>{_pct(row.get('close_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('high_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('avg_post_20d_close_improvement'))}</td>"
            f"<td>{_pct(row.get('avg_post_20d_best_high_improvement'))}</td>"
            "</tr>"
        )
    recovery_feature_rows = []
    for row in summary.get("recovery_feature_differences") or []:
        recovery_feature_rows.append(
            "<tr>"
            f"<td>{row.get('feature')}</td>"
            f"<td>{_num(row.get('recoverable_mean'))}</td>"
            f"<td>{_num(row.get('not_recoverable_mean'))}</td>"
            f"<td>{_num(row.get('diff'))}</td>"
            f"<td>{_num(row.get('standardized_diff'))}</td>"
            f"<td>{row.get('recoverable_count')}</td>"
            f"<td>{row.get('not_recoverable_count')}</td>"
            "</tr>"
        )
    recovery_target_feature_rows = []
    differences_by_target = summary.get("recovery_feature_differences_by_target") or {}
    for target, difference_rows in differences_by_target.items():
        for row in difference_rows[:12]:
            recovery_target_feature_rows.append(
                "<tr>"
                f"<td>{target}</td>"
                f"<td>{row.get('feature')}</td>"
                f"<td>{_num(row.get('positive_mean'))}</td>"
                f"<td>{_num(row.get('negative_mean'))}</td>"
                f"<td>{_num(row.get('diff'))}</td>"
                f"<td>{_num(row.get('standardized_diff'))}</td>"
                f"<td>{row.get('positive_count')}</td>"
                f"<td>{row.get('negative_count')}</td>"
                "</tr>"
            )
    recovery_profiles = summary.get("failure_recovery_profiles") or {}
    recovery_profile_rows = []
    recovery_profile_diff_rows = []
    for target in recovery_profiles.get("targets") or []:
        for row in target.get("by_exit_reason") or []:
            recovery_profile_rows.append(
                "<tr>"
                f"<td>{target.get('label') or target.get('target')}</td>"
                f"<td>{row.get('group') or '(none)'}</td>"
                f"<td>{row.get('sample_count')}</td>"
                f"<td>{row.get('positive_count')}</td>"
                f"<td>{_pct(row.get('positive_rate'))}</td>"
                f"<td>{_pct(row.get('positive_avg_return'))}</td>"
                f"<td>{_pct(row.get('negative_avg_return'))}</td>"
                f"<td>{_pct(row.get('positive_avg_mae'))}</td>"
                f"<td>{_pct(row.get('negative_avg_mae'))}</td>"
                f"<td>{_num(row.get('positive_avg_holding_days'))}</td>"
                f"<td>{_pct(row.get('positive_avg_post_best_close_improvement'))}</td>"
                f"<td>{_num(row.get('positive_avg_days_to_loss_reduction'))}</td>"
                "</tr>"
            )
        for row in target.get("top_differences") or []:
            recovery_profile_diff_rows.append(
                "<tr>"
                f"<td>{target.get('label') or target.get('target')}</td>"
                f"<td>{row.get('feature')}</td>"
                f"<td>{_num(row.get('positive_mean'))}</td>"
                f"<td>{_num(row.get('negative_mean'))}</td>"
                f"<td>{_num(row.get('diff'))}</td>"
                f"<td>{_num(row.get('standardized_diff'))}</td>"
                "</tr>"
            )
    recovery_rule_search = summary.get("failure_recovery_rule_search") or {}
    recovery_rule_search_rows = []
    for target in recovery_rule_search.get("targets") or []:
        for row in target.get("rules") or []:
            recovery_rule_search_rows.append(
                "<tr>"
                f"<td>{target.get('target')}</td>"
                f"<td>{' & '.join(row.get('conditions') or [])}</td>"
                f"<td>{row.get('sample_count')}</td>"
                f"<td>{_pct(row.get('coverage'))}</td>"
                f"<td>{row.get('positive_count')}</td>"
                f"<td>{_pct(row.get('positive_rate'))}</td>"
                f"<td>{_pct(row.get('baseline_rate'))}</td>"
                f"<td>{_pct(row.get('lift'))}</td>"
                f"<td>{_pct(row.get('avg_return'))}</td>"
                f"<td>{_pct(row.get('avg_mae'))}</td>"
                f"<td>{_pct(row.get('avg_post_best_close_improvement'))}</td>"
                f"<td>{_num(row.get('avg_days_to_loss_reduction'))}</td>"
                f"<td>{row.get('exit_reason_counts') or {}}</td>"
                "</tr>"
            )
    recovery_rule_rows = []
    for row in (summary.get("recovery_rule_candidates") or {}).get("rules") or []:
        recovery_rule_rows.append(
            "<tr>"
            f"<td>{row.get('rule_id')}</td>"
            f"<td>{row.get('intended_use')}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('coverage'))}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('avg_mae'))}</td>"
            f"<td>{_pct(row.get('recoverable_failure_rate'))}</td>"
            f"<td>{_pct(row.get('loss_reducible_rate'))}</td>"
            f"<td>{_pct(row.get('close_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('high_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('avg_post_20d_close_improvement'))}</td>"
            f"<td>{_pct(row.get('avg_post_20d_best_high_improvement'))}</td>"
            f"<td>{_num(row.get('avg_days_to_close_profit'))}</td>"
            f"<td>{_num(row.get('avg_days_to_loss_reduction'))}</td>"
            "</tr>"
        )
    fast_exit_candidates = summary.get("fast_exit_rule_candidates") or {}
    fast_exit_rows = []
    for row in fast_exit_candidates.get("rules") or []:
        fast_exit_rows.append(
            "<tr>"
            f"<td>{row.get('rule_id')}</td>"
            f"<td>{row.get('description')}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('coverage'))}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('avg_mfe'))}</td>"
            f"<td>{_pct(row.get('avg_mae'))}</td>"
            f"<td>{_pct(row.get('recoverable_failure_rate'))}</td>"
            f"<td>{_pct(row.get('loss_reducible_rate'))}</td>"
            f"<td>{_pct(row.get('close_turn_profitable_rate'))}</td>"
            f"<td>{_num(row.get('do_not_wait_score'))}</td>"
            f"<td><code>{row.get('yaml_when') or ''}</code></td>"
            "</tr>"
        )
    wait_diagnostics = summary.get("failure_wait_diagnostics") or {}
    wait_rows = []
    for row in wait_diagnostics.get("by_exit_reason") or []:
        wait_rows.append(
            "<tr>"
            f"<td>{row.get('group') or '(none)'}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('loss_reducible_rate'))}</td>"
            f"<td>{_pct(row.get('close_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('high_turn_profitable_rate'))}</td>"
            f"<td>{_pct(row.get('high_only_profit_rate'))}</td>"
            f"<td>{_pct(row.get('close_profit_within_5d_rate'))}</td>"
            f"<td>{_pct(row.get('loss_reduction_within_5d_rate'))}</td>"
            f"<td>{_num(row.get('avg_days_to_close_profit'))}</td>"
            f"<td>{_num(row.get('avg_days_to_loss_reduction'))}</td>"
            f"<td>{_pct(row.get('avg_intraday_vs_close_gap'))}</td>"
            "</tr>"
        )
    opportunity_cost = summary.get("wait_opportunity_cost") or {}
    opportunity_cost_rows = []
    for row in opportunity_cost.get("by_exit_reason") or []:
        opportunity_cost_rows.append(
            "<tr>"
            f"<td>{row.get('group') or '(none)'}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f"<td>{_pct(row.get('avg_return'))}</td>"
            f"<td>{_pct(row.get('any_candidate_rate'))}</td>"
            f"<td>{_num(row.get('avg_candidate_days'))}</td>"
            f"<td>{_num(row.get('avg_selected_candidate_count'))}</td>"
            f"<td>{_num(row.get('avg_position_count'))}</td>"
            f"<td>{_pct(row.get('near_full_window_rate'))}</td>"
            f"<td>{_num(row.get('avg_near_full_days'))}</td>"
            f"<td>{_pct(row.get('candidate_day_density'))}</td>"
            f"<td>{_pct(row.get('near_full_day_density'))}</td>"
            "</tr>"
        )
    wait_cost_profiles = summary.get("failure_wait_cost_profiles") or {}
    wait_cost_profile_rows = []
    wait_cost_profile_diff_rows = []
    for target in wait_cost_profiles.get("targets") or []:
        for row in target.get("by_exit_reason") or []:
            wait_cost_profile_rows.append(
                "<tr>"
                f"<td>{target.get('label') or target.get('target')}</td>"
                f"<td>{target.get('intended_use') or ''}</td>"
                f"<td>{row.get('group') or '(none)'}</td>"
                f"<td>{row.get('sample_count')}</td>"
                f"<td>{row.get('positive_count')}</td>"
                f"<td>{_pct(row.get('positive_rate'))}</td>"
                f"<td>{_pct(row.get('positive_avg_return'))}</td>"
                f"<td>{_pct(row.get('negative_avg_return'))}</td>"
                f"<td>{_num(row.get('positive_avg_candidate_days'))}</td>"
                f"<td>{_num(row.get('negative_avg_candidate_days'))}</td>"
                f"<td>{_num(row.get('positive_avg_selected_candidate_count'))}</td>"
                f"<td>{_num(row.get('positive_avg_near_full_days'))}</td>"
                f"<td>{_pct(row.get('positive_loss_reducible_rate'))}</td>"
                f"<td>{_pct(row.get('positive_close_profitable_rate'))}</td>"
                "</tr>"
            )
        for row in target.get("top_differences") or []:
            wait_cost_profile_diff_rows.append(
                "<tr>"
                f"<td>{target.get('label') or target.get('target')}</td>"
                f"<td>{row.get('feature')}</td>"
                f"<td>{_num(row.get('positive_mean'))}</td>"
                f"<td>{_num(row.get('negative_mean'))}</td>"
                f"<td>{_num(row.get('diff'))}</td>"
                f"<td>{_num(row.get('standardized_diff'))}</td>"
                f"<td>{row.get('positive_count')}</td>"
                f"<td>{row.get('negative_count')}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trade Pattern Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f3f6f8; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d7dde5; padding: 12px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Trade Pattern Analysis</h1>
  <div class="grid">
    <div class="metric">Samples<strong>{summary.get('sample_count')}</strong></div>
    <div class="metric">Avg Return<strong>{_pct(summary.get('avg_return'))}</strong></div>
    <div class="metric">Avg MFE<strong>{_pct(summary.get('avg_mfe'))}</strong></div>
    <div class="metric">Avg MAE<strong>{_pct(summary.get('avg_mae'))}</strong></div>
  </div>
  <h2>Outcome Counts</h2>
  <pre>{summary.get('outcome_counts')}</pre>
  <h2>Trade Management</h2>
  <div class="grid">
    <div class="metric">Early Confirm<strong>{_pct(management.get('early_confirm_rate'))}</strong></div>
    <div class="metric">Opportunity<strong>{_pct(management.get('opportunity_rate'))}</strong></div>
    <div class="metric">Fade After Peak<strong>{_pct(management.get('fade_after_peak_rate'))}</strong></div>
    <div class="metric">Opportunity Capture<strong>{_pct(management.get('opportunity_capture_rate'))}</strong></div>
  </div>
  <h3>Exit Review</h3>
  <div class="grid">
    <div class="metric">8% Opportunity<strong>{_pct(exit_review.get('opportunity_rate_8pct'))}</strong></div>
    <div class="metric">12% Opportunity<strong>{_pct(exit_review.get('opportunity_rate_12pct'))}</strong></div>
    <div class="metric">Hard Loss 8%<strong>{_pct(exit_review.get('hard_loss_rate_8pct'))}</strong></div>
    <div class="metric">8% Then Negative<strong>{_pct(exit_review.get('ended_negative_after_8pct_opportunity_rate'))}</strong></div>
  </div>
  <table><thead><tr><th>MFE Bucket</th><th>Samples</th><th>Avg Return</th><th>Success</th><th>Capture</th><th>Ended Negative</th><th>Peak Drawdown</th></tr></thead><tbody>{''.join(exit_rows)}</tbody></table>
  <h3>Exit Rule What-If</h3>
  <p>{what_if.get('note', '')}</p>
  <table><thead><tr><th>Rule</th><th>Samples</th><th>Avg Return</th><th>Avg Delta</th><th>Improved</th><th>Success</th><th>Negative</th><th>Avg Hold</th></tr></thead><tbody>{''.join(what_if_rows)}</tbody></table>
  <h3>Recoverable Failures</h3>
  <div class="grid">
    <div class="metric">Failed Samples<strong>{recovery.get('failed_count')}</strong></div>
    <div class="metric">Recoverable<strong>{_pct(recovery.get('recoverable_failure_rate'))}</strong></div>
    <div class="metric">Loss Reducible<strong>{_pct(recovery.get('loss_reducible_rate'))}</strong></div>
    <div class="metric">Close Turns Profit<strong>{_pct(recovery.get('close_turn_profitable_rate'))}</strong></div>
  </div>
  <table><thead><tr><th>Exit Reason</th><th>Failed Samples</th><th>Avg Return</th><th>Recoverable</th><th>Loss Reducible</th><th>Close Profit</th><th>High Profit</th><th>Avg Close Improve</th><th>Avg High Improve</th></tr></thead><tbody>{''.join(recovery_reason_rows)}</tbody></table>
  <h3>Recoverable Failure Feature Differences</h3>
  <table><thead><tr><th>Feature</th><th>Recoverable Mean</th><th>Not Recoverable Mean</th><th>Diff</th><th>Std Diff</th><th>Recoverable N</th><th>Not Recoverable N</th></tr></thead><tbody>{''.join(recovery_feature_rows)}</tbody></table>
  <h3>Recovery Target Feature Differences</h3>
  <table><thead><tr><th>Target</th><th>Feature</th><th>Positive Mean</th><th>Negative Mean</th><th>Diff</th><th>Std Diff</th><th>Positive N</th><th>Negative N</th></tr></thead><tbody>{''.join(recovery_target_feature_rows)}</tbody></table>
  <h3>Failure Recovery Profiles</h3>
  <p>{recovery_profiles.get('note', '')}</p>
  <table><thead><tr><th>Target</th><th>Exit Reason</th><th>Samples</th><th>Positive</th><th>Rate</th><th>Positive Avg Return</th><th>Negative Avg Return</th><th>Positive MAE</th><th>Negative MAE</th><th>Positive Hold</th><th>Positive Best Close Improve</th><th>Days To Reduce</th></tr></thead><tbody>{''.join(recovery_profile_rows)}</tbody></table>
  <table><thead><tr><th>Target</th><th>Feature</th><th>Positive Mean</th><th>Negative Mean</th><th>Diff</th><th>Std Diff</th></tr></thead><tbody>{''.join(recovery_profile_diff_rows)}</tbody></table>
  <h3>Failure Recovery Rule Search</h3>
  <p>{recovery_rule_search.get('note', '')}</p>
  <table><thead><tr><th>Target</th><th>Conditions</th><th>Samples</th><th>Coverage</th><th>Positive</th><th>Rate</th><th>Baseline</th><th>Lift</th><th>Avg Return</th><th>Avg MAE</th><th>Best Close Improve</th><th>Days To Reduce</th><th>Exit Reasons</th></tr></thead><tbody>{''.join(recovery_rule_search_rows)}</tbody></table>
  <h3>Recovery Rule Candidates</h3>
  <p>{(summary.get('recovery_rule_candidates') or {}).get('note', '')}</p>
  <table><thead><tr><th>Rule</th><th>Use</th><th>Samples</th><th>Coverage</th><th>Avg Return</th><th>Avg MAE</th><th>Recoverable</th><th>Loss Reducible</th><th>Close Profit</th><th>High Profit</th><th>Avg Close Improve</th><th>Avg High Improve</th><th>Days To Profit</th><th>Days To Reduce</th></tr></thead><tbody>{''.join(recovery_rule_rows)}</tbody></table>
  <h3>Fast Exit Rule Candidates</h3>
  <p>{fast_exit_candidates.get('note', '')}</p>
  <table><thead><tr><th>Rule</th><th>Description</th><th>Samples</th><th>Coverage</th><th>Avg Return</th><th>Avg MFE</th><th>Avg MAE</th><th>Recoverable</th><th>Loss Reducible</th><th>Close Profit</th><th>Do Not Wait Score</th><th>YAML When</th></tr></thead><tbody>{''.join(fast_exit_rows)}</tbody></table>
  <h3>Failure Wait Diagnostics</h3>
  <p>{wait_diagnostics.get('note', '')}</p>
  <div class="grid">
    <div class="metric">Close Profit 5D<strong>{_pct(wait_diagnostics.get('close_profit_within_5d_rate'))}</strong></div>
    <div class="metric">Loss Reduce 5D<strong>{_pct(wait_diagnostics.get('loss_reduction_within_5d_rate'))}</strong></div>
    <div class="metric">High Only Profit<strong>{_pct(wait_diagnostics.get('high_only_profit_rate'))}</strong></div>
    <div class="metric">Intraday/Close Gap<strong>{_pct(wait_diagnostics.get('avg_intraday_vs_close_gap'))}</strong></div>
  </div>
  <table><thead><tr><th>Exit Reason</th><th>Failed Samples</th><th>Avg Return</th><th>Loss Reducible</th><th>Close Profit</th><th>High Profit</th><th>High Only Profit</th><th>Close Profit 5D</th><th>Loss Reduce 5D</th><th>Days To Profit</th><th>Days To Reduce</th><th>Intraday/Close Gap</th></tr></thead><tbody>{''.join(wait_rows)}</tbody></table>
  <h3>Wait Opportunity Cost Proxy</h3>
  <p>{opportunity_cost.get('note', '')}</p>
  <div class="grid">
    <div class="metric">Any Candidate<strong>{_pct(opportunity_cost.get('any_candidate_rate'))}</strong></div>
    <div class="metric">Avg Candidate Days<strong>{_num(opportunity_cost.get('avg_candidate_days'))}</strong></div>
    <div class="metric">Near Full Window<strong>{_pct(opportunity_cost.get('near_full_window_rate'))}</strong></div>
    <div class="metric">Avg Near Full Days<strong>{_num(opportunity_cost.get('avg_near_full_days'))}</strong></div>
  </div>
  <table><thead><tr><th>Exit Reason</th><th>Failed Samples</th><th>Avg Return</th><th>Any Candidate</th><th>Avg Candidate Days</th><th>Avg Selected Candidates</th><th>Avg Position Count</th><th>Near Full Window</th><th>Avg Near Full Days</th><th>Candidate Density</th><th>Near Full Density</th></tr></thead><tbody>{''.join(opportunity_cost_rows)}</tbody></table>
  <h3>Failure Wait Cost Profiles</h3>
  <p>{wait_cost_profiles.get('note', '')}</p>
  <div class="grid">
    <div class="metric">High Wait Cost<strong>{_pct(wait_cost_profiles.get('high_wait_cost_rate'))}</strong></div>
    <div class="metric">High Cost Samples<strong>{wait_cost_profiles.get('high_wait_cost_count')}</strong></div>
    <div class="metric">Failed Matched<strong>{wait_cost_profiles.get('failed_count')}</strong></div>
    <div class="metric">Definition<strong>{wait_cost_profiles.get('high_cost_definition') or ''}</strong></div>
  </div>
  <table><thead><tr><th>Target</th><th>Use</th><th>Exit Reason</th><th>Samples</th><th>Positive</th><th>Rate</th><th>Positive Avg Return</th><th>Negative Avg Return</th><th>Positive Candidate Days</th><th>Negative Candidate Days</th><th>Positive Selected Candidates</th><th>Positive Near Full Days</th><th>Positive Loss Reducible</th><th>Positive Close Profit</th></tr></thead><tbody>{''.join(wait_cost_profile_rows)}</tbody></table>
  <table><thead><tr><th>Target</th><th>Feature</th><th>Positive Mean</th><th>Negative Mean</th><th>Diff</th><th>Std Diff</th><th>Positive N</th><th>Negative N</th></tr></thead><tbody>{''.join(wait_cost_profile_diff_rows)}</tbody></table>
  <h2>Models</h2>
  <h3>Interpretation</h3>
  <table><thead><tr><th>Target</th><th>Signal</th><th>Implication</th><th>Best AUC</th><th>Vs Baseline</th></tr></thead><tbody>{''.join(interpretation_rows)}</tbody></table>
  <h3>Target Comparison</h3>
  <table><thead><tr><th>Target</th><th>Samples</th><th>Positive Rate</th><th>Baseline</th><th>Baseline Acc</th><th>Best AUC Model</th><th>Best AUC</th><th>Best Acc Model</th><th>Vs Baseline</th></tr></thead><tbody>{''.join(target_rows)}</tbody></table>
  <p>Majority baseline: {baseline.get('label', '')} / accuracy {_pct(baseline.get('accuracy'))}</p>
  <table><thead><tr><th>Model</th><th>OK</th><th>AUC</th><th>Accuracy</th><th>Vs Baseline</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table>
  <h2>Clusters</h2>
  <table><thead><tr><th>Cluster</th><th>Description</th><th>Samples</th><th>Success Rate</th><th>Avg Return</th><th>Avg MAE</th><th>Figures</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body>
</html>
"""


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _cluster_figure_html(summary: Dict[str, Any], cluster_id: Any) -> str:
    if not cluster_id:
        return ""
    figures = (summary.get("figure_results") or {}).get(str(cluster_id), {})
    links = []
    if figures.get("representatives"):
        links.append(f"<a href=\"{figures['representatives']}\">代表图</a>")
    if figures.get("counter_examples"):
        links.append(f"<a href=\"{figures['counter_examples']}\">反例图</a>")
    return " | ".join(links)
