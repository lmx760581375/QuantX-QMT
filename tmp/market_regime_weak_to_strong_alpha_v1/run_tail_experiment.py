"""Range-regime right-tail mining for the weak-to-strong candidate pool.

This is a second-stage exploratory script. It reuses the candidate panel created
by run_experiment.py and writes only under ./tmp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ID_COLUMNS = {
    "signal_time",
    "instrument",
    "layer",
    "regime",
    "entry_session",
    "raw_candidate_count",
    "candidate_rank",
    "tradable_rank",
}
DIAGNOSTIC_COLUMNS = {
    "is_signal_member",
    "is_entry_member",
    "is_st_signal",
    "is_st_entry",
    "entry_has_bar",
    "entry_zero_volume",
    "entry_zero_amount",
    "entry_one_price_limit_up",
    "is_tradable",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tmp/market_regime_weak_to_strong_alpha_v1/configs/experiment.yaml")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--first-year", type=int)
    parser.add_argument("--last-year", type=int)
    parser.add_argument("--validation-sessions", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.first_year is not None:
        cfg["validation"]["first_prediction_year"] = int(args.first_year)
    if args.last_year is not None:
        cfg["validation"]["last_prediction_year"] = int(args.last_year)
    if args.validation_sessions is not None:
        cfg["validation"]["validation_sessions"] = int(args.validation_sessions)

    output_root = Path(cfg["output_root"])
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    run_tail_experiment(cfg, args.horizon)


def run_tail_experiment(cfg: dict[str, Any], horizon: int) -> None:
    from lightgbm import LGBMClassifier

    output_root = Path(cfg["output_root"])
    label_col = f"ret_{horizon}d"
    avail_col = f"label_available_{horizon}d"
    frame = pd.read_parquet(output_root / "data" / "candidate_panel.parquet")
    data = frame[
        frame["is_tradable"]
        & (frame["regime"] == "range")
        & frame["layer"].isin(["mid", "wide"])
        & frame[avail_col]
        & frame[label_col].notna()
    ].copy()
    if data.empty:
        raise RuntimeError("No range-regime tradable rows for tail experiment")

    data["signal_time"] = pd.to_datetime(data["signal_time"])
    data["year"] = data["signal_time"].dt.year
    data["layer_code"] = data["layer"].map({"mid": 0, "wide": 1}).astype(float)
    data = assign_tail_labels(data, label_col)
    feature_columns = choose_feature_columns(data)
    params = dict(cfg["model"]["lightgbm"])
    params.update({"objective": "binary", "class_weight": "balanced"})
    seeds = [int(x) for x in cfg["validation"]["seeds"]]
    years = range(int(cfg["validation"]["first_prediction_year"]), int(cfg["validation"]["last_prediction_year"]) + 1)
    topn_values = [5, 10, 20, 50]
    penalties = [0.0, 0.5, 1.0, 1.5]

    prediction_parts = []
    summary_rows = []
    importance_rows = []
    min_train_rows = int(cfg["model"].get("min_train_rows", 500))

    for year in years:
        pred_mask = data["year"] == year
        if not pred_mask.any():
            continue
        pred_start = data.loc[pred_mask, "signal_time"].min()
        train_cutoff = cutoff_before_validation(data, pred_start, cfg)
        if pd.isna(train_cutoff):
            continue
        base_train = data[data["signal_time"] <= train_cutoff]
        prediction = data[pred_mask].copy()
        if len(base_train) < min_train_rows:
            continue
        for layer, pred_group in prediction.groupby("layer", sort=True):
            train_group = base_train[base_train["layer"] == layer]
            if len(train_group) < min_train_rows:
                continue
            for target_name, bad_name in (("tail_top10", "bad_bottom30"), ("tail_top20", "bad_loss3")):
                for seed in seeds:
                    tail_model, medians = fit_classifier(train_group, feature_columns, target_name, params, seed)
                    bad_model, _ = fit_classifier(train_group, feature_columns, bad_name, params, seed + 1000)
                    scored = score_pair(pred_group, feature_columns, medians, tail_model, bad_model)
                    scored["prediction_year"] = year
                    scored["seed"] = seed
                    scored["target_name"] = target_name
                    scored["bad_name"] = bad_name
                    prediction_parts.append(
                        scored[
                            [
                                "signal_time",
                                "instrument",
                                "layer",
                                label_col,
                                "tail_top10",
                                "tail_top20",
                                "bad_bottom30",
                                "bad_loss3",
                                "p_tail",
                                "p_bad",
                                "prediction_year",
                                "seed",
                                "target_name",
                                "bad_name",
                                "tradable_rank",
                            ]
                        ]
                    )
                    importance_rows.extend(feature_importance_rows(tail_model, feature_columns, year, seed, layer, target_name))
                    importance_rows.extend(feature_importance_rows(bad_model, feature_columns, year, seed, layer, bad_name))
                    for penalty in penalties:
                        scored["tail_score"] = scored["p_tail"] - penalty * scored["p_bad"]
                        summary_rows.extend(
                            evaluate_scored(scored, label_col, target_name, bad_name, penalty, year, seed, topn_values)
                        )
        print(f"[tail] year={year} done", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    baseline = build_baselines(data, label_col, topn_values)
    robustness = build_robustness(predictions, label_col, topn_values, penalties)

    summary.to_csv(output_root / "runs" / "tail_model_summary.csv", index=False)
    baseline.to_csv(output_root / "runs" / "tail_baseline_summary.csv", index=False)
    robustness.to_csv(output_root / "runs" / "tail_robustness.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_root / "runs" / "tail_feature_importance.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(output_root / "runs" / "tail_predictions.parquet", index=False)
    print(f"[tail] wrote summary rows={len(summary)} baseline rows={len(baseline)}", flush=True)


def assign_tail_labels(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = frame.groupby(["signal_time", "layer"])[label_col]
    out = frame.copy()
    out["q80"] = grouped.transform(lambda s: s.quantile(0.80))
    out["q90"] = grouped.transform(lambda s: s.quantile(0.90))
    out["q30"] = grouped.transform(lambda s: s.quantile(0.30))
    out["tail_top20"] = (out[label_col] >= out["q80"]).astype(int)
    out["tail_top10"] = (out[label_col] >= out["q90"]).astype(int)
    out["bad_bottom30"] = (out[label_col] <= out["q30"]).astype(int)
    out["bad_loss3"] = (out[label_col] <= -0.03).astype(int)
    return out


def choose_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = set(ID_COLUMNS) | set(DIAGNOSTIC_COLUMNS)
    excluded.update({col for col in frame.columns if col.startswith("ret_") or col.startswith("up_")})
    excluded.update({col for col in frame.columns if col.startswith("label_available_")})
    excluded.update({col for col in frame.columns if col.startswith("entry_")})
    excluded.update({"year", "q80", "q90", "q30", "tail_top10", "tail_top20", "bad_bottom30", "bad_loss3"})
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if (pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col])) and has_finite_value(
            frame[col]
        ):
            cols.append(col)
    preferred = ["layer_score", "candidate_rank", "raw_candidate_count", "layer_code"]
    return tuple(dict.fromkeys([*preferred, *cols]))


def has_finite_value(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).any())


def cutoff_before_validation(data: pd.DataFrame, pred_start: pd.Timestamp, cfg: dict[str, Any]) -> pd.Timestamp:
    sessions = pd.DatetimeIndex(sorted(data.loc[data["signal_time"] < pred_start, "signal_time"].unique()))
    if len(sessions) == 0:
        return pd.NaT
    reserve = int(cfg["validation"].get("validation_sessions", 252)) + int(cfg["validation"].get("embargo_sessions", 0))
    if len(sessions) <= reserve:
        return sessions[-1]
    return sessions[-reserve - 1]


def fit_classifier(frame: pd.DataFrame, feature_columns: tuple[str, ...], target_col: str, params: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier

    values = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    labels = frame[target_col].astype(int).to_numpy()
    medians = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    model = LGBMClassifier(random_state=seed, verbosity=-1, **params)
    model.fit(values, labels)
    return model, medians


def score_pair(frame: pd.DataFrame, feature_columns: tuple[str, ...], medians: np.ndarray, tail_model: Any, bad_model: Any) -> pd.DataFrame:
    values = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    out = frame.copy()
    out["p_tail"] = tail_model.predict_proba(values)[:, 1]
    out["p_bad"] = bad_model.predict_proba(values)[:, 1]
    return out


def evaluate_scored(
    frame: pd.DataFrame,
    label_col: str,
    target_name: str,
    bad_name: str,
    penalty: float,
    year: int,
    seed: int,
    topn_values: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for layer, group in frame.groupby("layer", sort=True):
        ordered = group.sort_values(["signal_time", "tail_score"], ascending=[True, False])
        for topn in topn_values:
            selected = ordered.groupby("signal_time", sort=False).head(topn)
            row = metric_row(selected, label_col, topn, layer, "tail_score")
            row.update({"target_name": target_name, "bad_name": bad_name, "penalty": penalty, "year": year, "seed": seed})
            rows.append(row)
    return rows


def build_baselines(data: pd.DataFrame, label_col: str, topn_values: list[int]) -> pd.DataFrame:
    rows = []
    for layer, group in data.groupby("layer", sort=True):
        for topn in topn_values:
            original = group[group["tradable_rank"] <= topn]
            rows.append({**metric_row(original, label_col, topn, layer, "original_rank"), "year": "all"})
            rows.append({**metric_row(random_daily_sample(group, topn), label_col, topn, layer, "random_matched"), "year": "all"})
        for year, year_group in group.groupby("year", sort=True):
            for topn in topn_values:
                original = year_group[year_group["tradable_rank"] <= topn]
                rows.append({**metric_row(original, label_col, topn, layer, "original_rank_year"), "year": int(year)})
                rows.append(
                    {
                        **metric_row(random_daily_sample(year_group, topn), label_col, topn, layer, "random_matched_year"),
                        "year": int(year),
                    }
                )
    return pd.DataFrame(rows)


def random_daily_sample(group: pd.DataFrame, topn: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260715 + topn)
    pieces = []
    for _, daily in group.groupby("signal_time", sort=True):
        n = min(topn, len(daily))
        if n > 0:
            pieces.append(daily.iloc[rng.choice(len(daily), size=n, replace=False)])
    return pd.concat(pieces, ignore_index=True) if pieces else group.iloc[0:0]


def build_robustness(predictions: pd.DataFrame, label_col: str, topn_values: list[int], penalties: list[float]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    strict = predictions[predictions["prediction_year"].between(2021, 2025)].copy()
    for (target_name, bad_name, seed, layer), group in strict.groupby(["target_name", "bad_name", "seed", "layer"], sort=True):
        for penalty in penalties:
            scored = group.copy()
            scored["tail_score"] = scored["p_tail"] - penalty * scored["p_bad"]
            ordered = scored.sort_values(["signal_time", "tail_score"], ascending=[True, False])
            for topn in topn_values:
                selected = ordered.groupby("signal_time", sort=False).head(topn).copy()
                if selected.empty:
                    continue
                drop_max = selected.drop(selected.groupby("prediction_year")[label_col].idxmax())
                q99 = selected.groupby("prediction_year")[label_col].transform(lambda s: s.quantile(0.99))
                q95 = selected.groupby("prediction_year")[label_col].transform(lambda s: s.quantile(0.95))
                rows.append(
                    {
                        "target_name": target_name,
                        "bad_name": bad_name,
                        "seed": seed,
                        "layer": layer,
                        "topn": topn,
                        "penalty": penalty,
                        "sample_count": int(len(selected)),
                        "mean_full": mean_or_none(selected[label_col]),
                        "mean_drop_year_max": mean_or_none(drop_max[label_col]),
                        "mean_drop_top1pct": mean_or_none(selected.loc[selected[label_col] <= q99, label_col]),
                        "mean_drop_top5pct": mean_or_none(selected.loc[selected[label_col] <= q95, label_col]),
                    }
                )
    return pd.DataFrame(rows)


def metric_row(frame: pd.DataFrame, label_col: str, topn: int, layer: str, method: str) -> dict[str, Any]:
    values = pd.to_numeric(frame[label_col], errors="coerce").dropna()
    daily = frame.dropna(subset=[label_col]).groupby("signal_time")[label_col].mean() if not frame.empty else pd.Series(dtype=float)
    return {
        "method": method,
        "layer": layer,
        "topn": topn,
        "sample_count": int(len(values)),
        "session_count": int(daily.size),
        "mean_return": mean_or_none(values),
        "median_return": quantile_or_none(values, 0.5),
        "win_rate": mean_or_none(values > 0),
        "tail_top10_rate": mean_or_none(frame.loc[values.index, "tail_top10"] == 1) if len(values) else None,
        "tail_top20_rate": mean_or_none(frame.loc[values.index, "tail_top20"] == 1) if len(values) else None,
        "bad_bottom30_rate": mean_or_none(frame.loc[values.index, "bad_bottom30"] == 1) if len(values) else None,
        "bad_loss3_rate": mean_or_none(frame.loc[values.index, "bad_loss3"] == 1) if len(values) else None,
        "daily_mean_return": mean_or_none(daily),
        "positive_day_rate": mean_or_none(daily > 0),
        "avg_daily_count": mean_or_none(frame.groupby("signal_time").size()) if not frame.empty else None,
    }


def feature_importance_rows(model: Any, features: tuple[str, ...], year: int, seed: int, layer: str, target: str) -> list[dict[str, Any]]:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    return [
        {
            "year": year,
            "seed": seed,
            "layer": layer,
            "target": target,
            "feature": feature,
            "gain": float(g),
            "split": int(s),
        }
        for feature, g, s in zip(features, gain, split)
    ]


def mean_or_none(values: Any) -> float | None:
    series = pd.Series(values).dropna()
    return None if series.empty else float(series.mean())


def quantile_or_none(values: Any, q: float) -> float | None:
    series = pd.Series(values).dropna()
    return None if series.empty else float(series.quantile(q))


if __name__ == "__main__":
    main()
