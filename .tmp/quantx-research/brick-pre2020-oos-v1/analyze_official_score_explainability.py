from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
RUN_DIR = ROOT / "renko_official_runs"
OUT_REPORT = ROOT / "official_score_explainability_report.json"
OUT_ROWS = ROOT / "official_score_explainability_rows.parquet"


def main() -> int:
    rows = load_rows()
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in {"date", "instrument", "symbol", "name", "industry", "score_model"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["score_rank_pct"] = df.groupby("date")["score"].rank(pct=True)
    df["rank_rank_pct"] = 1.0 - (df.groupby("date")["rank"].rank(pct=True) - 1.0 / df.groupby("date")["rank"].transform("count"))

    full_v11 = df[df["v11_score"].notna()].copy()
    no_v11 = df[df["v11_score"].isna()].copy()
    feature_cols = choose_features(df)

    report = {
        "inputs": {
            "run_dir": str(RUN_DIR),
            "rows": int(len(df)),
            "dates": int(df["date"].nunique()),
            "v11_score_rows": int(len(full_v11)),
            "v11_score_dates": int(full_v11["date"].nunique()),
            "no_v11_score_rows": int(len(no_v11)),
            "no_v11_score_dates": int(no_v11["date"].nunique()),
        },
        "direct_field_checks": direct_field_checks(df),
        "score_feature_correlations": score_feature_correlations(df, feature_cols),
        "model_explainability_all_dates": evaluate_models(df, feature_cols),
        "model_explainability_v11_dates": evaluate_models(full_v11, feature_cols),
        "model_explainability_pre_v11_dates": evaluate_models(no_v11, feature_cols),
        "top_feature_importance_v11_dates": feature_importance(full_v11, feature_cols),
        "top_feature_importance_all_dates": feature_importance(df, feature_cols),
        "date_summary": date_summary(df),
        "notes": [
            "后段公开候选直接暴露 factor_values.v11_score/v11_rank，且 score 与 v11_score 基本等价。",
            "前段公开候选没有 v11_score/brick/prev_brick 等完整字段，score 量纲也明显不同，不能和后段直接混训为同一公式。",
            "本报告只分析公开候选内部排序可解释性，不使用公开 Top5 交易明细。",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    df.to_parquet(OUT_ROWS, index=False)
    print(json.dumps({
        "wrote": str(OUT_REPORT),
        "inputs": report["inputs"],
        "direct_field_checks": report["direct_field_checks"],
        "v11_models": report["model_explainability_v11_dates"],
        "pre_v11_models": report["model_explainability_pre_v11_dates"],
        "top_v11_importance": report["top_feature_importance_v11_dates"][:12],
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        date = str(payload.get("data_date"))
        for item in payload.get("items", []):
            fv = item.get("factor_values") or {}
            row: dict[str, Any] = {
                "date": date,
                "run_id": payload.get("id"),
                "instrument": norm_symbol(str(item.get("symbol"))),
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "close": item.get("close"),
                "change_since_screen_pct": item.get("change_since_screen_pct"),
                "industry": item.get("industry"),
                "circ_shares": item.get("circ_shares"),
                "is_suspended": int(bool(item.get("is_suspended"))),
                "trend_short_top": item.get("trend_short"),
                "trend_long_top": item.get("trend_long"),
                "kdj_j_top": item.get("kdj_j"),
            }
            for key, value in fv.items():
                if isinstance(value, (int, float)) or value is None:
                    row[key] = value
                elif key == "score_model":
                    row[key] = value
            rows.append(row)
    return rows


def norm_symbol(code: str) -> str | None:
    if code.startswith("6"):
        return "SH" + code
    if code.startswith(("0", "3")):
        return "SZ" + code
    if code.startswith(("8", "9")):
        return "BJ" + code
    return code or None


def choose_features(df: pd.DataFrame) -> list[str]:
    excluded = {
        "run_id",
        "rank",
        "score",
        "score_rank_pct",
        "rank_rank_pct",
        "v11_score",
        "v11_rank",
        "v10_score",
        "v10_rank",
        "v6_score",
        "v6_rank",
        "candidate_count",
    }
    cols = []
    for col in df.columns:
        if col in excluded or col in {"date", "instrument", "symbol", "name", "industry", "score_model"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().mean() >= 0.20:
            cols.append(col)
    return cols


def direct_field_checks(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "v11_score" in df.columns:
        full = df[df["v11_score"].notna()].copy()
        if not full.empty:
            out["score_eq_v11_score_max_abs_diff"] = float((full["score"] - full["v11_score"]).abs().max())
            out["score_vs_v11_score_corr"] = corr(full["score"], full["v11_score"])
            out["rank_eq_v11_rank_ratio"] = float((full["rank"].round() == full["v11_rank"].round()).mean()) if "v11_rank" in full.columns else None
            out["v11_dates"] = sorted(full["date"].unique().tolist())
    out["score_summary_by_v11_presence"] = {
        "v11_present": describe(df.loc[df.get("v11_score", pd.Series(index=df.index)).notna(), "score"]),
        "v11_missing": describe(df.loc[df.get("v11_score", pd.Series(index=df.index)).isna(), "score"]),
    }
    return out


def score_feature_correlations(df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in feature_cols:
        values = df[["score", col]].dropna()
        if len(values) < 20:
            continue
        rows.append({
            "feature": col,
            "pearson": corr(values["score"], values[col]),
            "spearman": spear(values["score"], values[col]),
            "non_null": int(len(values)),
        })
    rows.sort(key=lambda x: abs(x["spearman"] or 0.0), reverse=True)
    return rows[:60]


def evaluate_models(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    if df.empty or df["date"].nunique() < 3 or len(df) < 100:
        return {"skipped": True, "reason": "too few rows or dates"}
    data = df.dropna(subset=["score"]).copy()
    dates = sorted(data["date"].unique().tolist())
    split = max(1, int(len(dates) * 0.65))
    train_dates = set(dates[:split])
    test_dates = set(dates[split:])
    train = data[data["date"].isin(train_dates)].copy()
    test = data[data["date"].isin(test_dates)].copy()
    if train.empty or test.empty:
        return {"skipped": True, "reason": "empty split"}
    cols = [c for c in feature_cols if train[c].notna().any() and test[c].notna().any()]
    models = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=320, min_samples_leaf=6, random_state=20260715, n_jobs=-1)),
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=420, min_samples_leaf=4, random_state=20260715, n_jobs=-1)),
    }
    out: dict[str, Any] = {"features": cols, "train_dates": sorted(train_dates), "test_dates": sorted(test_dates), "train_rows": int(len(train)), "test_rows": int(len(test))}
    for name, model in models.items():
        model.fit(train[cols], train["score"])
        pred = model.predict(test[cols])
        test_eval = test.copy()
        test_eval["pred"] = pred
        daily_rank = []
        top5_hits = []
        for _, g in test_eval.groupby("date"):
            if len(g) >= 3:
                daily_rank.append(spear(g["score"], g["pred"]))
            true_top = set(g.nsmallest(5, "rank")["instrument"])
            pred_top = set(g.nlargest(min(5, len(g)), "pred")["instrument"])
            top5_hits.append(len(true_top & pred_top) / max(1, len(true_top)))
        out[name] = {
            "r2": float(r2_score(test["score"], pred)),
            "mae": float(mean_absolute_error(test["score"], pred)),
            "pearson": corr(test["score"], pd.Series(pred, index=test.index)),
            "spearman_all": spear(test["score"], pd.Series(pred, index=test.index)),
            "mean_daily_spearman": float(np.nanmean(daily_rank)) if daily_rank else None,
            "mean_daily_top5_recall": float(np.nanmean(top5_hits)) if top5_hits else None,
        }
    return out


def feature_importance(df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty or len(df) < 100:
        return []
    data = df.dropna(subset=["score"]).copy()
    cols = [c for c in feature_cols if data[c].notna().any()]
    if not cols:
        return []
    model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=520, min_samples_leaf=3, random_state=20260715, n_jobs=-1))
    model.fit(data[cols], data["score"])
    reg = model.named_steps["extratreesregressor"]
    rows = [{"feature": col, "importance": float(imp)} for col, imp in zip(cols, reg.feature_importances_)]
    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows[:40]


def date_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for date, g in df.groupby("date", sort=True):
        rows.append({
            "date": date,
            "count": int(len(g)),
            "score_min": float(g["score"].min()),
            "score_max": float(g["score"].max()),
            "score_mean": float(g["score"].mean()),
            "has_v11_score": bool(g.get("v11_score", pd.Series(index=g.index)).notna().any()),
            "candidate_count_field": int(g["candidate_count"].dropna().iloc[0]) if "candidate_count" in g.columns and g["candidate_count"].notna().any() else None,
        })
    return rows


def corr(a: pd.Series, b: pd.Series) -> float | None:
    values = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[:, 0].corr(values.iloc[:, 1])) if len(values) >= 3 else None


def spear(a: pd.Series, b: pd.Series) -> float | None:
    values = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return None
    val = spearmanr(values.iloc[:, 0], values.iloc[:, 1], nan_policy="omit").correlation
    return float(val) if np.isfinite(val) else None


def describe(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
