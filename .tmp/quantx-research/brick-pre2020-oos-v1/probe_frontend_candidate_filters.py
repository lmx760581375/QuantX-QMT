from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
IN_CANDIDATES = ROOT / "frontend_brick_candidate_daily_frontend.parquet"
OUT_REPORT = ROOT / "frontend_candidate_hard_filter_probe.json"


def main() -> int:
    df = pd.read_parquet(IN_CANDIDATES)
    enrich_features(df)

    base_name = "candidate_prev_down_today_up"
    base = df[df[base_name].fillna(False)].copy()
    truth = base["official_candidate"].astype(bool)
    official_total = int(df["official_candidate"].sum())

    feature_cols = [
        "ret1",
        "ret1_rank_pct",
        "frontend_brick",
        "brick_rank_pct",
        "frontend_delta",
        "delta_rank_pct",
        "frontend_delta_prev_abs",
        "frontend_decline_sum_prev5",
        "decline_rank_pct",
        "frontend_range_from_5d_min",
        "range_rank_pct",
        "$amount",
        "amount_rank_pct",
        "turnover_proxy_rank_pct",
        "$close",
        "price_rank_pct",
        "amplitude_pct",
        "amplitude_rank_pct",
        "body_pct",
        "close_pos",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "open_gap_pct",
        "is_mainboard",
        "is_chinext",
        "is_star",
    ]
    feature_cols = [c for c in feature_cols if c in base.columns]

    single_feature = single_feature_probe(base, feature_cols, official_total)
    greedy_rules = greedy_rule_probe(base, feature_cols, official_total)
    combo_rules = combo_rule_probe(base, official_total)
    tree_report = decision_tree_probe(base, feature_cols, official_total)
    daily_summary = daily_quality(base, base_name)

    report = {
        "inputs": {
            "candidates": str(IN_CANDIDATES),
            "rows": int(len(df)),
            "dates": int(df["date"].nunique()),
            "official_total": official_total,
            "base_rule": base_name,
            "base_rows": int(len(base)),
            "base_official": int(truth.sum()),
            "base_quality": quality(df[base_name].astype(bool), df["official_candidate"].astype(bool), df),
        },
        "feature_auc_on_base": feature_auc(base, feature_cols),
        "single_feature_threshold_top60": single_feature[:60],
        "greedy_rules": greedy_rules,
        "combo_rules_top80": combo_rules[:80],
        "decision_tree": tree_report,
        "daily_base_tail20": daily_summary[-20:],
        "interpretation": [
            "base_rule 已覆盖几乎全部官方候选，但过宽；本报告只用于反推排序前硬过滤，不代表 alpha 结论。",
            "若一个规则 precision 高但 recall 很低，说明它可能是排序偏好而不是硬过滤；候选复现优先寻找 recall 足够高且每日规模接近官方的规则。",
            "所有特征均来自 T 日本地 OHLCV 与 frontend brick 递推，不使用公开 Top5 交易标签。",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT_REPORT),
        "base_quality": report["inputs"]["base_quality"],
        "top_single": single_feature[:10],
        "top_combo": combo_rules[:10],
        "tree_summary": tree_report["depth_reports"],
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def enrich_features(df: pd.DataFrame) -> None:
    prev_close = df["$close"] / (1.0 + df["ret1"].replace(0, np.nan))
    df["prev_close"] = prev_close
    df["amplitude_pct"] = (df["$high"] - df["$low"]) / prev_close
    df["body_pct"] = (df["$close"] - df["$open"]) / prev_close
    high_low = (df["$high"] - df["$low"]).replace(0, np.nan)
    df["close_pos"] = (df["$close"] - df["$low"]) / high_low
    df["upper_shadow_pct"] = (df["$high"] - df[["$open", "$close"]].max(axis=1)) / prev_close
    df["lower_shadow_pct"] = (df[["$open", "$close"]].min(axis=1) - df["$low"]) / prev_close
    df["open_gap_pct"] = df["$open"] / prev_close - 1.0
    df["frontend_delta_prev_abs"] = df["frontend_delta_prev"].abs()
    df["delta_rank_pct"] = df.groupby("date")["frontend_delta"].rank(pct=True)
    df["decline_rank_pct"] = df.groupby("date")["frontend_decline_sum_prev5"].rank(pct=True)
    df["range_rank_pct"] = df.groupby("date")["frontend_range_from_5d_min"].rank(pct=True)
    df["price_rank_pct"] = df.groupby("date")["$close"].rank(pct=True)
    df["amplitude_rank_pct"] = df.groupby("date")["amplitude_pct"].rank(pct=True)
    df["turnover_proxy_rank_pct"] = df.groupby("date")["$volume"].rank(pct=True)
    inst = df["instrument"].astype(str)
    df["is_star"] = inst.str.startswith("SH688").astype(float)
    df["is_chinext"] = inst.str.startswith("SZ300").astype(float)
    df["is_mainboard"] = (~inst.str.startswith("SH688") & ~inst.str.startswith("SZ300") & ~inst.str.startswith("BJ")).astype(float)


def feature_auc(base: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    y = base["official_candidate"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        values = pd.to_numeric(base[col], errors="coerce")
        mask = values.notna()
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        auc = roc_auc_score(y[mask], values[mask])
        rows.append({"feature": col, "auc_high_is_official": float(auc), "auc_abs": float(max(auc, 1 - auc))})
    rows.sort(key=lambda x: x["auc_abs"], reverse=True)
    return rows


def single_feature_probe(base: pd.DataFrame, feature_cols: list[str], official_total: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        values = pd.to_numeric(base[col], errors="coerce")
        qs = sorted(set([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]))
        for q in qs:
            thr = values.quantile(q)
            if not np.isfinite(thr):
                continue
            for op in [">=", "<="]:
                pred = values >= thr if op == ">=" else values <= thr
                stats = quality_on_base(pred.fillna(False), base, official_total)
                if stats["recall_total"] >= 0.35 or stats["precision"] >= 0.30:
                    rows.append({"rule": f"{col} {op} {thr:.8g}", "feature": col, "op": op, "threshold": float(thr), **stats})
    rows.sort(key=lambda x: (x["f1_total"], x["precision"]), reverse=True)
    return rows


def greedy_rule_probe(base: pd.DataFrame, feature_cols: list[str], official_total: int) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    current = pd.Series(True, index=base.index)
    used: set[str] = set()
    for step in range(1, 7):
        best: dict[str, Any] | None = None
        for col in feature_cols:
            if col in used:
                continue
            values = pd.to_numeric(base[col], errors="coerce")
            for q in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                thr = values[current].quantile(q)
                if not np.isfinite(thr):
                    continue
                for op in [">=", "<="]:
                    cond = values >= thr if op == ">=" else values <= thr
                    pred = current & cond.fillna(False)
                    stats = quality_on_base(pred, base, official_total)
                    if stats["recall_total"] < 0.35:
                        continue
                    cand = {"step": step, "add": f"{col} {op} {thr:.8g}", "feature": col, "op": op, "threshold": float(thr), **stats}
                    if best is None or (cand["f1_total"], cand["precision"]) > (best["f1_total"], best["precision"]):
                        best = cand
        if best is None:
            break
        values = pd.to_numeric(base[best["feature"]], errors="coerce")
        cond = values >= best["threshold"] if best["op"] == ">=" else values <= best["threshold"]
        current = current & cond.fillna(False)
        used.add(best["feature"])
        best["full_rule"] = " AND ".join([r["add"] for r in rules] + [best["add"]])
        rules.append(best)
    return rules


def combo_rule_probe(base: pd.DataFrame, official_total: int) -> list[dict[str, Any]]:
    specs = [
        ("ret1", ">=", [0.00, 0.02, 0.03, 0.04, 0.05, 0.06]),
        ("ret1_rank_pct", ">=", [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]),
        ("frontend_brick", ">=", [50, 60, 70, 80, 90, 100]),
        ("brick_rank_pct", ">=", [0.50, 0.60, 0.70, 0.80, 0.90]),
        ("frontend_delta", ">=", [0.5, 1, 2, 3, 5, 8]),
        ("delta_rank_pct", ">=", [0.50, 0.60, 0.70, 0.80, 0.90]),
        ("frontend_decline_sum_prev5", ">=", [1, 3, 5, 8, 12, 20]),
        ("amount_rank_pct", ">=", [0.20, 0.40, 0.50, 0.60, 0.70, 0.80]),
        ("amplitude_pct", ">=", [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]),
        ("close_pos", ">=", [0.40, 0.50, 0.60, 0.70, 0.80]),
        ("upper_shadow_pct", "<=", [0.005, 0.01, 0.015, 0.02, 0.03]),
        ("open_gap_pct", "<=", [0.00, 0.01, 0.02, 0.03, 0.05]),
    ]
    y = base["official_candidate"].astype(bool).to_numpy()
    n_dates = max(1, int(base["date"].nunique()))
    conds = []
    for col, op, thresholds in specs:
        if col not in base.columns:
            continue
        values = pd.to_numeric(base[col], errors="coerce")
        for thr in thresholds:
            pred = values >= thr if op == ">=" else values <= thr
            mask = pred.fillna(False).to_numpy(dtype=bool)
            stats = quality_np(mask, y, official_total, n_dates)
            if stats["recall_total"] >= 0.25 and stats["pred_count"] >= 500:
                conds.append((f"{col} {op} {thr}", mask))

    rows: list[dict[str, Any]] = []
    for size in [2, 3]:
        for combo in combinations(conds, size):
            pred = np.ones(len(base), dtype=bool)
            names = []
            for name, cond in combo:
                pred &= cond
                names.append(name)
            stats = quality_np(pred, y, official_total, n_dates)
            if stats["pred_count"] < 500 or stats["recall_total"] < 0.20:
                continue
            rows.append({"rule": " AND ".join(names), "conditions": names, **stats})
    rows.sort(key=lambda x: (x["f1_total"], x["precision"]), reverse=True)
    return rows


def decision_tree_probe(base: pd.DataFrame, feature_cols: list[str], official_total: int) -> dict[str, Any]:
    model_cols = [c for c in feature_cols if base[c].notna().any()]
    x = base[model_cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    y = base["official_candidate"].astype(int)
    depth_reports = []
    tree_text = {}
    for depth in [2, 3, 4, 5, 6]:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=80, class_weight="balanced", random_state=20260715)
        clf.fit(x, y)
        prob = clf.predict_proba(x)[:, 1]
        for q in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            thr = np.quantile(prob, q)
            stats = quality_on_base(pd.Series(prob >= thr, index=base.index), base, official_total)
            depth_reports.append({"depth": depth, "prob_quantile": q, "prob_threshold": float(thr), **stats})
        if depth in {3, 5}:
            tree_text[f"depth_{depth}"] = export_text(clf, feature_names=model_cols, max_depth=depth)
    depth_reports.sort(key=lambda x: (x["f1_total"], x["precision"]), reverse=True)
    return {"feature_columns": model_cols, "depth_reports": depth_reports[:30], "tree_text": tree_text}


def quality_on_base(pred: pd.Series, base: pd.DataFrame, official_total: int) -> dict[str, Any]:
    truth = base["official_candidate"].astype(bool)
    pred = pred.fillna(False).astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn_base = int((~pred & truth).sum())
    fn_total = official_total - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall_base = tp / int(truth.sum()) if int(truth.sum()) else 0.0
    recall_total = tp / official_total if official_total else 0.0
    f1_total = 2 * precision * recall_total / (precision + recall_total) if precision + recall_total else 0.0
    pred_count = int(pred.sum())
    n_dates = max(1, int(base["date"].nunique()))
    return {
        "tp": tp,
        "fp": fp,
        "fn_base": fn_base,
        "fn_total": fn_total,
        "pred_count": pred_count,
        "precision": precision,
        "recall_base": recall_base,
        "recall_total": recall_total,
        "f1_total": f1_total,
        "avg_pred_per_day": float(pred_count / n_dates),
    }


def quality_np(pred: np.ndarray, truth: np.ndarray, official_total: int, n_dates: int) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn_base = int((~pred & truth).sum())
    fn_total = official_total - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    base_total = int(truth.sum())
    recall_base = tp / base_total if base_total else 0.0
    recall_total = tp / official_total if official_total else 0.0
    f1_total = 2 * precision * recall_total / (precision + recall_total) if precision + recall_total else 0.0
    pred_count = int(pred.sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn_base": fn_base,
        "fn_total": fn_total,
        "pred_count": pred_count,
        "precision": precision,
        "recall_base": recall_base,
        "recall_total": recall_total,
        "f1_total": f1_total,
        "avg_pred_per_day": float(pred_count / max(1, n_dates)),
    }


def quality(pred: pd.Series, truth: pd.Series, frame: pd.DataFrame) -> dict[str, Any]:
    pred = pred.fillna(False).astype(bool)
    truth = truth.fillna(False).astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": int(pred.sum()),
        "official_count": int(truth.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_pred_per_day": float(frame.loc[pred].groupby("date")["instrument"].nunique().mean()) if pred.any() else 0.0,
    }


def daily_quality(base: pd.DataFrame, base_name: str) -> list[dict[str, Any]]:
    rows = []
    for date, group in base.groupby("date", sort=True):
        rows.append({
            "date": date,
            "base_pred": int(group[base_name].sum()),
            "official_in_base": int(group["official_candidate"].sum()),
            "base_precision": float(group["official_candidate"].mean()) if len(group) else 0.0,
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
