from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP148_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/analyze_qmt_dynamic_theme_cluster_ranker.py"
RANDOM_SEED = 20260714
TOPKS = (10, 20)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP148 = load_module("exp148_for_early_mainline_cluster_opportunity", EXP148_PATH)

CLUSTER_BASE_FEATURES = [
    "cluster_size", "cluster_anchor_score_mean", "cluster_corr_anchor_mean", "cluster_corr_anchor_std",
    "ret5_rank_mean", "ret20_rank_mean", "ret60_rank_mean", "amount_rank_mean", "amt_ratio20_rank_mean",
    "near_high20_rank_mean", "close_strength_rank_mean", "upper_wick_low_rank_mean", "vol20_low_rank_mean",
    "range20_low_rank_mean", "price_rank_mean", "cluster_breadth20_mean", "cluster_disp20_mean",
    "early_lag_share", "leader_share", "overheat_share", "repair_share", "wide_participation",
    "cluster_rel_ret20_mean", "cluster_rel_ret20_std", "cluster_rel_ret60_mean", "cluster_rel_ret60_std",
    "cluster_lag_repair_mean", "cluster_mid_follower_mean", "cluster_leader_quality_mean", "cluster_overheat_risk_mean",
    "mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--anchor-count", type=int, default=EXP148.ANCHOR_COUNT)
    parser.add_argument("--corr-window", type=int, default=EXP148.CORR_WINDOW)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP148.EXP130.EXP124.load_market(args)
    panel = EXP148.EXP130.build_panel(market, args)
    panel = EXP148.add_dynamic_cluster_features(panel, market, args.anchor_count, args.corr_window)
    stock_scored, stock_folds, stock_importances = EXP148.walk_forward(panel, args.min_train_rows)
    cluster_frame = build_cluster_frame(panel)
    cluster_scored, cluster_folds, cluster_importances = cluster_walk_forward(cluster_frame)
    scored = attach_cluster_scores(stock_scored, cluster_scored)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_early_mainline_cluster_opportunity_v1",
        "source_experiment": "qmt_dynamic_theme_cluster_ranker_v1",
        "panel_rows": int(len(panel)),
        "stock_scored_rows": int(len(stock_scored)),
        "cluster_rows": int(len(cluster_frame)),
        "cluster_scored_rows": int(len(cluster_scored)),
        "stock_folds": stock_folds,
        "cluster_folds": cluster_folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "cluster_oracle_diagnostics": cluster_oracle_diagnostics(cluster_frame),
        "cluster_prediction_summary": cluster_prediction_summary(cluster_scored),
        "stock_feature_importance_top30": EXP148.summarize_importance(stock_importances)[:30],
        "cluster_feature_importance_top50": summarize_importance(cluster_importances)[:50],
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "anchor_count": args.anchor_count,
            "corr_window": args.corr_window,
            "cluster_feature_count": len(CLUSTER_BASE_FEATURES),
            "causality": "Dynamic clusters and cluster opportunity features use completed daily bars through signal date T only. Cluster opportunity models train only on clusters from years strictly before the test year; 2026 uses 2021-2025. Stock scores are annual walk-forward OOS from Exp148. Future T+1/T+6 open returns are used only as historical training targets and OOS evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_cluster_frame(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["session_exec_rank"] = work.groupby("session")["exec_label5_open"].rank(pct=True, method="first")
    rows: list[dict[str, Any]] = []
    for (session, cluster_id), group in work.groupby(["session", "cluster_id"], sort=True):
        ranked_future = group.sort_values(["exec_label5_open", "instrument"], ascending=[False, True])
        top_n = ranked_future.head(min(5, len(ranked_future)))
        row: dict[str, Any] = {
            "session": str(session),
            "year": int(str(session)[:4]),
            "cluster_id": int(cluster_id),
            "cluster_size": int(len(group)),
            "target_top5_exec": safe_mean(top_n["exec_label5_open"]),
            "target_top10_share": safe_mean(group["session_exec_rank"] >= 0.90),
            "target_top5_share": safe_mean(group["session_exec_rank"] >= 0.95),
            "target_mean_exec": safe_mean(group["exec_label5_open"]),
            "target_best_exec": safe_mean([group["exec_label5_open"].max()]),
        }
        row.update(cluster_visible_features(group))
        rows.append(row)
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    session_mean = out.groupby("session")["target_top5_exec"].transform("mean")
    out["target_rel_top5_exec"] = out["target_top5_exec"] - session_mean
    out["target_is_opportunity"] = (out.groupby("session")["target_top5_exec"].rank(pct=True, method="first") >= 0.80).astype(int)
    return out


def cluster_visible_features(group: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    mean_cols = [
        "cluster_anchor_score", "cluster_corr_anchor", "ret5_rank", "ret20_rank", "ret60_rank",
        "amount_rank", "amt_ratio20_rank", "near_high20_rank", "close_strength_rank", "upper_wick_low_rank",
        "vol20_low_rank", "range20_low_rank", "price_rank", "cluster_breadth20", "cluster_disp20",
        "cluster_rel_ret20", "cluster_rel_ret60", "cluster_lag_repair", "cluster_mid_follower",
        "cluster_leader_quality", "cluster_overheat_risk",
    ]
    for col in mean_cols:
        row[f"{col}_mean"] = safe_mean(group[col])
    for col in ["cluster_corr_anchor", "cluster_rel_ret20", "cluster_rel_ret60"]:
        row[f"{col}_std"] = safe_std(group[col])
    row["early_lag_share"] = safe_mean((group["cluster_rank_ret20"] < 0.40) & (group["cluster_corr_rank"] > 0.45))
    row["leader_share"] = safe_mean((group["cluster_rank_ret20"] > 0.70) & (group["cluster_rank_near_high"] > 0.65))
    row["overheat_share"] = safe_mean(group["cluster_overheat_risk"] > 0.60)
    row["repair_share"] = safe_mean(group["cluster_lag_repair"] > 0.58)
    row["wide_participation"] = safe_mean(group["ret20_rank"] > 0.50)
    for col in ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20"]:
        row[col] = safe_mean(group[col])
    return row


def cluster_walk_forward(cluster_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = cluster_frame.copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if train.empty or test.empty:
            continue
        reg_abs = make_cluster_regressor(year, 0)
        reg_rel = make_cluster_regressor(year, 101)
        cls = make_cluster_classifier(year, 211)
        reg_abs.fit(train[CLUSTER_BASE_FEATURES], train["target_top5_exec"])
        reg_rel.fit(train[CLUSTER_BASE_FEATURES], train["target_rel_top5_exec"])
        cls.fit(train[CLUSTER_BASE_FEATURES], train["target_is_opportunity"])
        out = test.copy()
        out["score_cluster_abs"] = reg_abs.predict(out[CLUSTER_BASE_FEATURES])
        out["score_cluster_rel"] = reg_rel.predict(out[CLUSTER_BASE_FEATURES])
        out["score_cluster_cls"] = cls.predict_proba(out[CLUSTER_BASE_FEATURES])[:, 1]
        out["score_cluster_blend"] = (
            0.40 * out.groupby("session")["score_cluster_abs"].rank(pct=True, method="first")
            + 0.35 * out.groupby("session")["score_cluster_rel"].rank(pct=True, method="first")
            + 0.25 * out.groupby("session")["score_cluster_cls"].rank(pct=True, method="first")
        )
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_sessions": int(train["session"].nunique()),
            "test_sessions": int(test["session"].nunique()),
        })
        imp = {f"abs::{name}": float(value) for name, value in zip(CLUSTER_BASE_FEATURES, reg_abs.feature_importances_)}
        imp.update({f"rel::{name}": float(value) for name, value in zip(CLUSTER_BASE_FEATURES, reg_rel.feature_importances_)})
        imp.update({f"cls::{name}": float(value) for name, value in zip(CLUSTER_BASE_FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_cluster_regressor(year: int, offset: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=220, learning_rate=0.035, num_leaves=21, max_depth=5, min_child_samples=35,
        subsample=0.90, colsample_bytree=0.90, reg_alpha=1.0, reg_lambda=10.0,
        random_state=RANDOM_SEED + year + offset, n_jobs=4, verbosity=-1,
    )


def make_cluster_classifier(year: int, offset: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=180, learning_rate=0.035, num_leaves=21, max_depth=5, min_child_samples=35,
        subsample=0.90, colsample_bytree=0.90, reg_alpha=1.0, reg_lambda=10.0,
        random_state=RANDOM_SEED + year + offset, n_jobs=4, verbosity=-1,
    )


def attach_cluster_scores(stock_scored: pd.DataFrame, cluster_scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "session", "cluster_id", "score_cluster_abs", "score_cluster_rel", "score_cluster_cls",
        "score_cluster_blend", "target_top5_exec", "target_rel_top5_exec", "target_is_opportunity",
    ]
    out = stock_scored.merge(cluster_scored[cols], on=["session", "cluster_id"], how="left")
    for col in ["score_cluster_abs", "score_cluster_rel", "score_cluster_cls", "score_cluster_blend"]:
        out[col] = out[col].fillna(out.groupby("session")[col].transform("mean")).fillna(0.0)
    out["score_stock_rank_pct"] = out.groupby("session")["score_cluster_rank"].rank(pct=True, method="first")
    out["score_cluster_abs_pct"] = out.groupby("session")["score_cluster_abs"].rank(pct=True, method="first")
    out["score_cluster_rel_pct"] = out.groupby("session")["score_cluster_rel"].rank(pct=True, method="first")
    out["score_cluster_cls_pct"] = out.groupby("session")["score_cluster_cls"].rank(pct=True, method="first")
    out["score_cluster_blend_pct"] = out.groupby("session")["score_cluster_blend"].rank(pct=True, method="first")
    out["score_opp_abs_rank"] = 0.52 * out["score_stock_rank_pct"] + 0.48 * out["score_cluster_abs_pct"]
    out["score_opp_rel_rank"] = 0.52 * out["score_stock_rank_pct"] + 0.48 * out["score_cluster_rel_pct"]
    out["score_opp_cls_rank"] = 0.52 * out["score_stock_rank_pct"] + 0.48 * out["score_cluster_cls_pct"]
    out["score_early_mainline"] = 0.45 * out["score_stock_rank_pct"] + 0.55 * out["score_cluster_blend_pct"]
    out["score_lag_in_opportunity"] = (
        0.42 * out["score_cluster_blend_pct"]
        + 0.28 * out["score_stock_rank_pct"]
        + 0.18 * out["cluster_lag_repair"]
        + 0.12 * (1.0 - out["cluster_overheat_risk"])
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def variants() -> list[tuple[str, str]]:
    return [
        ("cluster_rank", "score_cluster_rank"),
        ("opp_abs_rank", "score_opp_abs_rank"),
        ("opp_rel_rank", "score_opp_rel_rank"),
        ("opp_cls_rank", "score_opp_cls_rank"),
        ("early_mainline", "score_early_mainline"),
        ("lag_in_opportunity", "score_lag_in_opportunity"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append(label_row(session, variant, topk, selected))
        for variant, selected in structured_selections(group).items():
            for topk in TOPKS:
                rows.append(label_row(session, variant, topk, selected.head(topk)))
    return pd.DataFrame(rows)


def label_row(session: str, variant: str, topk: int, selected: pd.DataFrame) -> dict[str, Any]:
    return {
        "session": str(session),
        "year": int(str(session)[:4]),
        "variant": variant,
        "topk": int(topk),
        "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
        "mean_raw5_open": safe_mean(selected["raw5_open"]),
        "entry_ok": safe_mean(selected["entry_ok"]),
        "exit_ok": safe_mean(selected["exit_ok"]),
        "avg_cluster_score": safe_mean(selected["score_cluster_blend_pct"]),
        "unique_clusters": int(selected["cluster_id"].nunique()) if not selected.empty else 0,
        "max_cluster_share": max_cluster_share(selected),
        "count": int(len(selected)),
    }


def structured_selections(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "top_clusters_stock_rank": select_top_clusters(group, "score_cluster_blend", "score_cluster_rank", 20, 4),
        "top_clusters_lag_repair": select_top_clusters(group, "score_cluster_blend", "score_lag_in_opportunity", 20, 4),
    }


def select_top_clusters(group: pd.DataFrame, cluster_score_col: str, stock_score_col: str, topk: int, per_cluster: int) -> pd.DataFrame:
    cluster_order = (
        group.groupby("cluster_id")[cluster_score_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .to_list()
    )
    parts = []
    for cid in cluster_order:
        part = group[group["cluster_id"] == cid].sort_values([stock_score_col, "instrument"], ascending=[False, True]).head(per_cluster)
        parts.append(part)
        if sum(len(p) for p in parts) >= topk:
            break
    return pd.concat(parts, ignore_index=True).head(topk) if parts else group.head(0)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "avg_cluster_score": float(group["avg_cluster_score"].mean()),
            "avg_unique_clusters": float(group["unique_clusters"].mean()),
            "avg_max_cluster_share": float(group["max_cluster_share"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
                }
                for year, yg in group.groupby("year")
            },
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in grouped.items()
            }
            out[f"{variant}::top{topk}"] = EXP148.EXP130.replay_open_selection(selections, market, horizon)
    for variant in ["top_clusters_stock_rank", "top_clusters_lag_repair"]:
        for topk in TOPKS:
            selections = {
                session: list(structured_selections(group)[variant].head(topk)["instrument"].astype(str))
                for session, group in grouped.items()
            }
            out[f"{variant}::top{topk}"] = EXP148.EXP130.replay_open_selection(selections, market, horizon)
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows: list[dict[str, Any]] = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
        rows.append({
            "key": key,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "ret2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "mean_raw5_open": float(label.get("mean_raw5_open", 0.0)),
            "avg_unique_clusters": float(label.get("avg_unique_clusters", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows


def cluster_oracle_diagnostics(cluster_frame: pd.DataFrame) -> dict[str, Any]:
    best = cluster_frame.sort_values(["session", "target_top5_exec"], ascending=[True, False]).groupby("session").head(5)
    return {
        "top5_clusters_mean_target_top5_exec": safe_mean(best["target_top5_exec"]),
        "top5_clusters_by_year": {
            str(int(year)): {
                "mean_target_top5_exec": safe_mean(group["target_top5_exec"]),
                "mean_target_top10_share": safe_mean(group["target_top10_share"]),
                "avg_cluster_size": safe_mean(group["cluster_size"]),
            }
            for year, group in best.groupby("year")
        },
        "all_clusters_mean_target_top5_exec": safe_mean(cluster_frame["target_top5_exec"]),
    }


def cluster_prediction_summary(cluster_scored: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for year, group in cluster_scored.groupby("year"):
        year_data: dict[str, Any] = {}
        for score_col in ["score_cluster_abs", "score_cluster_rel", "score_cluster_cls", "score_cluster_blend"]:
            corr = group[[score_col, "target_top5_exec"]].corr().iloc[0, 1]
            chosen = group.sort_values(["session", score_col], ascending=[True, False]).groupby("session").head(5)
            year_data[score_col] = {
                "corr_target_top5_exec": float(corr) if np.isfinite(corr) else 0.0,
                "chosen_mean_target_top5_exec": safe_mean(chosen["target_top5_exec"]),
                "chosen_mean_target_top10_share": safe_mean(chosen["target_top10_share"]),
            }
        out[str(int(year))] = year_data
    return out


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows]


def max_cluster_share(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    return float(selected["cluster_id"].value_counts().max() / len(selected))


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else 0.0


def safe_std(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanstd(arr))
    return val if np.isfinite(val) else 0.0


def classify_status(accounts: dict[str, Any]) -> str:
    for account in accounts.values():
        annual = account.get("annual_returns", {})
        if (
            account.get("final_multiple", 0.0) >= 8.7749
            and annual.get("2026", -1.0) >= 0.1918
            and account.get("all_years_positive", False)
            and account.get("avg_selected_count", 0.0) > 5
        ):
            return "candidate_needs_deeper_audit_before_merge"
    return "rejected_before_formal_candidate"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"clusters={row['avg_unique_clusters']:.2f} exec={row['mean_exec_label5_open']:+.6f}"
        )
    print("cluster_prediction_summary", json.dumps(result["cluster_prediction_summary"], ensure_ascii=False, sort_keys=True)[:1400])


if __name__ == "__main__":
    raise SystemExit(main())
