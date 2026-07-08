"""Supervised model analysis for trade pattern features."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .features import train_feature_columns


def run_supervised_analysis(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    models: Sequence[str],
    random_state: int = 42,
    feature_scope: str = "entry",
) -> Dict[str, Any]:
    """Train optional baseline models and return compact diagnostics."""
    if features.empty or labels.empty:
        return {"ok": False, "reason": "empty dataset", "models": {}}
    if importlib.util.find_spec("sklearn") is None:
        return {"ok": False, "reason": "scikit-learn is not installed", "models": {}}

    target_specs = [
        ("binary_success", "success", "success", "not_success", None),
        ("had_opportunity", "opportunity", "opportunity", "no_opportunity", None),
        ("faded_after_peak", "fade_after_peak", "fade_after_peak", "no_fade_after_peak", None),
        ("efficient_capture", "efficient_capture", "efficient_capture", "no_efficient_capture", None),
        ("recoverable_failure", "recoverable_failure", "recoverable_failure", "not_recoverable_failure", "is_failed_trade"),
        ("loss_reducible_after_exit", "loss_reducible", "loss_reducible", "not_loss_reducible", "is_failed_trade"),
        ("close_turn_profitable_after_exit", "close_turn_profitable", "close_turn_profitable", "not_close_turn_profitable", "is_failed_trade"),
    ]
    keep = ["sample_id", "outcome_label", *[column for column, _, _, _, _ in target_specs if column in labels.columns]]
    if "is_failed_trade" in labels.columns:
        keep.append("is_failed_trade")
    data = features.merge(labels[keep], on="sample_id", how="inner")
    columns = train_feature_columns(data, feature_scope=feature_scope)
    if data.empty or len(columns) == 0:
        return {"ok": False, "reason": "not enough labeled samples or train features", "models": {}, "feature_columns": columns}

    data = data.sort_values("entry_date")
    targets: Dict[str, Any] = {}
    for column, target_name, positive_label, negative_label, subset_column in target_specs:
        if column not in data.columns:
            continue
        target_frame = data
        if subset_column and subset_column in data.columns:
            target_frame = data[data[subset_column].fillna(False).astype(bool)].copy()
        targets[target_name] = _run_target_analysis(
            target_frame,
            columns,
            label_column=column,
            positive_label=positive_label,
            negative_label=negative_label,
            models=models,
            random_state=random_state,
        )
        targets[target_name]["feature_scope"] = feature_scope

    primary = targets.get("success") or {}
    if not primary.get("ok"):
        return {
            "ok": False,
            "reason": primary.get("reason", "not enough labeled samples or train features"),
            "models": {},
            "feature_columns": columns,
            "feature_scope": feature_scope,
            "targets": targets,
        }
    primary = dict(primary)
    primary["feature_scope"] = feature_scope
    primary["targets"] = targets
    return primary


def _run_target_analysis(
    data: pd.DataFrame,
    columns: List[str],
    label_column: str,
    positive_label: str,
    negative_label: str,
    models: Sequence[str],
    random_state: int,
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    target_data = data[data[label_column].notna()].copy()
    if target_data.empty:
        return {"ok": False, "reason": "empty target", "models": {}, "feature_columns": columns}
    target_data[label_column] = target_data[label_column].astype(int)
    if target_data[label_column].nunique() < 2:
        return {"ok": False, "reason": "target has one class", "models": {}, "feature_columns": columns, "sample_count": int(len(target_data))}

    split = max(1, int(len(target_data) * 0.75))
    if split >= len(target_data):
        split = len(target_data) - 1
    train = target_data.iloc[:split]
    test = target_data.iloc[split:]
    if train[label_column].nunique() < 2 or test.empty:
        return {"ok": False, "reason": "time split lacks both classes", "models": {}, "feature_columns": columns, "sample_count": int(len(target_data))}

    x_train = train[columns]
    y_train = train[label_column].astype(int)
    x_test = test[columns]
    y_test = test[label_column].astype(int)
    majority_label = int(y_train.mean() >= 0.5)
    majority_pred = np.full(len(y_test), majority_label)
    results: Dict[str, Any] = {
        "ok": True,
        "target": positive_label,
        "label_column": label_column,
        "sample_count": int(len(target_data)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "positive_rate": float(target_data[label_column].mean()),
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline": {
            "label": positive_label if majority_label == 1 else negative_label,
            "accuracy": float(accuracy_score(y_test, majority_pred)),
            "precision": float(precision_score(y_test, majority_pred, zero_division=0)),
            "recall": float(recall_score(y_test, majority_pred, zero_division=0)),
        },
        "feature_columns": columns,
        "models": {},
    }

    for model_name in models:
        model_key, estimator, reason = _build_estimator(model_name, random_state)
        if estimator is None:
            results["models"][model_key] = {"ok": False, "reason": reason or f"unknown model: {model_name}"}
            continue

        try:
            estimator.fit(x_train, y_train)
            proba = _predict_proba(estimator, x_test)
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "ok": True,
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None,
                "accuracy_vs_baseline": float(accuracy_score(y_test, pred) - results["majority_baseline"]["accuracy"]),
                "top_features": _feature_importance(estimator, columns),
                "score_quantiles": _score_quantiles(test, proba, label_column=label_column),
            }
            results["models"][model_key] = metrics
        except Exception as exc:
            results["models"][model_key] = {"ok": False, "reason": str(exc)}
    return results


def _build_estimator(model_name: str, random_state: int) -> tuple[str, Any | None, str | None]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model_key = model_name.lower()
    if model_key in {"logistic", "logistic_regression"}:
        return model_key, make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
        ), None
    if model_key in {"random_forest", "rf"}:
        return model_key, make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, min_samples_leaf=3),
        ), None
    if model_key == "lightgbm":
        if importlib.util.find_spec("lightgbm") is None:
            return model_key, None, "lightgbm is not installed"
        from lightgbm import LGBMClassifier

        return model_key, make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(random_state=random_state, class_weight="balanced", verbose=-1),
        ), None
    if model_key == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            return model_key, None, "xgboost is not installed"
        from xgboost import XGBClassifier

        return model_key, make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(random_state=random_state, eval_metric="logloss"),
        ), None
    return model_key, None, f"unknown model: {model_name}"


def _predict_proba(estimator: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    decision = estimator.decision_function(x_test)
    return 1 / (1 + np.exp(-decision))


def _feature_importance(estimator: Any, columns: List[str]) -> List[Dict[str, Any]]:
    model = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    if values is None:
        return []
    order = np.argsort(values)[::-1][:30]
    return [{"feature": columns[int(i)], "importance": float(values[int(i)])} for i in order]


def _score_quantiles(test: pd.DataFrame, proba: np.ndarray, label_column: str) -> List[Dict[str, Any]]:
    frame = test[["sample_id", "return", label_column]].copy()
    frame["score"] = proba
    unique_scores = frame["score"].nunique()
    bins = min(5, unique_scores)
    if bins < 2:
        return []
    frame["score_quantile"] = pd.qcut(frame["score"], q=bins, labels=False, duplicates="drop")
    rows = []
    for quantile, group in frame.groupby("score_quantile", dropna=True):
        rows.append({
            "quantile": int(quantile),
            "sample_count": int(len(group)),
            "avg_score": float(group["score"].mean()),
            "positive_rate": float(group[label_column].mean()),
            "avg_return": float(group["return"].mean()),
        })
    return rows
