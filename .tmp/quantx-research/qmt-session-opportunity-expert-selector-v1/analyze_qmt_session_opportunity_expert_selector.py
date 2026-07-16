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
from lightgbm import LGBMRegressor


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP150_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-cluster-lag-repair-ranker-v1/analyze_qmt_cluster_lag_repair_ranker.py"
RANDOM_SEED = 20260714
TOPKS = (10, 20)
EXPERTS = [
    ("cluster_rank", "score_cluster_rank"),
    ("repair_rank", "score_repair_rank"),
    ("residual_reg", "score_residual_reg"),
    ("weighted_reg", "score_weighted_reg"),
    ("rank_repair_35", "score_rank_repair_35"),
    ("rank_residual_35", "score_rank_residual_35"),
    ("rank_weighted_35", "score_rank_weighted_35"),
    ("repair_stack", "score_repair_stack"),
]
SESSION_FEATURES = [
    "mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20",
]
PORTFOLIO_FEATURES = [
    "repair_prior_rank", "repair_overheat_penalty", "repair_lag_depth", "repair_confirmation",
    "cluster_rel_ret20", "cluster_rel_ret60", "cluster_rank_ret20", "cluster_rank_ret60",
    "cluster_rank_near_high", "cluster_corr_anchor", "ret20_rank", "ret60_rank", "amount_rank",
    "amt_ratio20_rank", "near_high20_rank", "vol20_low_rank", "range20_low_rank",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP150 = load_module("exp150_for_session_opportunity_selector", EXP150_PATH)


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
    parser.add_argument("--anchor-count", type=int, default=EXP150.EXP148.ANCHOR_COUNT)
    parser.add_argument("--corr-window", type=int, default=EXP150.EXP148.CORR_WINDOW)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP150.EXP148.EXP130.EXP124.load_market(args)
    panel = EXP150.EXP148.EXP130.build_panel(market, args)
    panel = EXP150.EXP148.add_dynamic_cluster_features(panel, market, args.anchor_count, args.corr_window)
    panel = EXP150.add_repair_training_features(panel)
    scored, stock_folds, stock_importances = EXP150.walk_forward(panel, args.min_train_rows)
    expert_frame = build_expert_frame(scored)
    selections, selector_folds, predictions = causal_expert_selections(scored, expert_frame)
    detail = evaluate_selections(scored, selections)
    accounts = replay_accounts(selections, market, args.horizon)
    oracle = oracle_diagnostics(scored, expert_frame, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_session_opportunity_expert_selector_v1",
        "source_experiments": [
            "qmt_dynamic_theme_cluster_ranker_v1",
            "qmt_cluster_lag_repair_ranker_v1",
        ],
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "expert_rows": int(len(expert_frame)),
        "stock_folds": stock_folds,
        "selector_folds": selector_folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "oracle_diagnostics": oracle,
        "selector_prediction_summary": summarize_predictions(predictions),
        "stock_feature_importance_top30": EXP150.summarize_importance(stock_importances)[:30],
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
            "experts": [name for name, _ in EXPERTS],
            "cold_start": "2022 has no prior OOS expert ledger, so selector variants default to cluster_rank for 2022. From 2023 onward each year trains only on completed prior OOS expert rows.",
            "causality": "Stock scores are annual walk-forward OOS. Session-expert opportunity models train only on expert outcomes from years strictly before the test year; 2026 uses 2022-2025 only. Expert prediction features use only signal-date T market, score, and portfolio-profile fields. Future returns are used only as prior-year training targets and OOS evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_expert_frame(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        year = int(str(session)[:4])
        session_base = {name: safe_mean(group[name]) for name in SESSION_FEATURES}
        for expert, score_col in EXPERTS:
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True]).copy()
            selected = ranked.head(20)
            score_rank = group.groupby("session")[score_col].rank(pct=True, method="first")
            row: dict[str, Any] = {
                "session": str(session),
                "year": year,
                "expert": expert,
                "target_exec_top20": safe_mean(selected["exec_label5_open"]),
                "target_raw_top20": safe_mean(selected["raw5_open"]),
                "target_exec_top10": safe_mean(ranked.head(10)["exec_label5_open"]),
                "score_top20_mean": safe_mean(selected[score_col]),
                "score_top20_min": safe_mean([selected[score_col].min()]),
                "score_top20_max": safe_mean([selected[score_col].max()]),
                "score_all_std": safe_mean([group[score_col].std()]),
                "score_top20_rank_mean": safe_mean(score_rank.loc[selected.index]),
                "unique_clusters": float(selected["cluster_id"].nunique()),
                "max_cluster_share": max_cluster_share(selected),
            }
            row.update(session_base)
            for feature in PORTFOLIO_FEATURES:
                row[f"sel_{feature}"] = safe_mean(selected[feature])
            for expert_name, _ in EXPERTS:
                row[f"expert_is_{expert_name}"] = 1.0 if expert == expert_name else 0.0
            rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def causal_expert_selections(scored: pd.DataFrame, expert_frame: pd.DataFrame) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]], pd.DataFrame]:
    variants = ["selector_top1", "selector_top2_merge", "selector_top3_merge", "selector_all_weighted"]
    selections: dict[str, dict[str, list[str]]] = {variant: {} for variant in variants}
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_cols = selector_feature_columns(expert_frame)
    scored_by_session = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for year in sorted(int(y) for y in expert_frame["year"].unique()):
        test_rows = expert_frame[expert_frame["year"] == year].copy()
        train_rows = expert_frame[expert_frame["year"] < year].copy()
        if train_rows.empty:
            test_rows["pred_exec_top20"] = 0.0
            test_rows.loc[test_rows["expert"] == "cluster_rank", "pred_exec_top20"] = 1.0
            mode = "cold_start_cluster_rank"
        else:
            model = make_selector_model(year)
            model.fit(train_rows[feature_cols], train_rows["target_exec_top20"])
            test_rows["pred_exec_top20"] = model.predict(test_rows[feature_cols])
            mode = "lgbm_prior_years"
        prediction_frames.append(test_rows)
        fold_rows.append({
            "year": int(year),
            "mode": mode,
            "train_rows": int(len(train_rows)),
            "test_rows": int(len(test_rows)),
            "train_sessions": int(train_rows["session"].nunique()) if not train_rows.empty else 0,
            "test_sessions": int(test_rows["session"].nunique()),
            "train_years": sorted(int(y) for y in train_rows["year"].unique()) if not train_rows.empty else [],
        })
        for session, pred_group in test_rows.groupby("session", sort=True):
            stock_group = scored_by_session[str(session)]
            ranked_experts = pred_group.sort_values(["pred_exec_top20", "expert"], ascending=[False, True])
            selections["selector_top1"][str(session)] = select_from_experts(stock_group, ranked_experts.head(1), 20, 25)
            selections["selector_top2_merge"][str(session)] = select_from_experts(stock_group, ranked_experts.head(2), 20, 35)
            selections["selector_top3_merge"][str(session)] = select_from_experts(stock_group, ranked_experts.head(3), 20, 35)
            selections["selector_all_weighted"][str(session)] = select_from_experts(stock_group, ranked_experts, 20, 25)
    return selections, fold_rows, pd.concat(prediction_frames, ignore_index=True)


def selector_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"session", "year", "expert", "target_exec_top20", "target_raw_top20", "target_exec_top10"}
    return [col for col in frame.columns if col not in exclude]


def make_selector_model(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=24,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=RANDOM_SEED + year,
        n_jobs=4,
        verbosity=-1,
    )


def select_from_experts(stock_group: pd.DataFrame, expert_rows: pd.DataFrame, topk: int, per_expert_pool: int) -> list[str]:
    pred_rank = expert_rows["pred_exec_top20"].rank(pct=True, method="first").to_dict()
    candidates: list[pd.DataFrame] = []
    for _, row in expert_rows.iterrows():
        expert = str(row["expert"])
        score_col = dict(EXPERTS)[expert]
        ranked = stock_group.sort_values([score_col, "instrument"], ascending=[False, True]).head(per_expert_pool).copy()
        ranked["expert"] = expert
        ranked["within_rank"] = ranked[score_col].rank(pct=True, method="first")
        ranked["expert_pred_rank"] = float(pred_rank.get(row.name, 0.5))
        ranked["combined_score"] = 0.62 * ranked["expert_pred_rank"] + 0.38 * ranked["within_rank"]
        candidates.append(ranked)
    if not candidates:
        return []
    all_candidates = pd.concat(candidates, ignore_index=True)
    best = all_candidates.sort_values(["combined_score", "instrument"], ascending=[False, True]).drop_duplicates("instrument", keep="first")
    return list(best.head(topk)["instrument"].astype(str))


def evaluate_selections(scored: pd.DataFrame, selections: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_session = {str(session): group.set_index("instrument") for session, group in scored.groupby("session", sort=True)}
    for variant, session_map in selections.items():
        for session, symbols in session_map.items():
            group = by_session[session].loc[[sym for sym in symbols if sym in by_session[session].index]]
            for topk in TOPKS:
                selected = group.head(topk)
                rows.append({
                    "session": session,
                    "year": int(session[:4]),
                    "variant": variant,
                    "topk": int(topk),
                    "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
                    "mean_raw5_open": safe_mean(selected["raw5_open"]),
                    "entry_ok": safe_mean(selected["entry_ok"]),
                    "exit_ok": safe_mean(selected["exit_ok"]),
                    "count": int(len(selected)),
                    "mean_repair_prior_rank": safe_mean(selected["repair_prior_rank"]),
                    "mean_cluster_rel_ret20": safe_mean(selected["cluster_rel_ret20"]),
                    "unique_clusters": int(selected["cluster_id"].nunique()) if not selected.empty else 0,
                    "max_cluster_share": max_cluster_share(selected.reset_index()) if not selected.empty else 0.0,
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "avg_count": float(group["count"].mean()),
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


def replay_accounts(selections: dict[str, dict[str, list[str]]], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, session_map in selections.items():
        for topk in TOPKS:
            topk_map = {session: symbols[:topk] for session, symbols in session_map.items()}
            out[f"{variant}::top{topk}"] = EXP150.EXP148.EXP130.replay_open_selection(topk_map, market, horizon)
    return out


def oracle_diagnostics(scored: pd.DataFrame, expert_frame: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    oracle_map: dict[str, list[str]] = {}
    by_session = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for session, group in expert_frame.groupby("session", sort=True):
        best = group.sort_values(["target_exec_top20", "expert"], ascending=[False, True]).iloc[0]
        score_col = dict(EXPERTS)[str(best["expert"])]
        oracle_map[str(session)] = list(by_session[str(session)].sort_values([score_col, "instrument"], ascending=[False, True]).head(20)["instrument"].astype(str))
    account = EXP150.EXP148.EXP130.replay_open_selection(oracle_map, market, horizon)
    return {
        "oracle_best_expert_top20_account": account,
        "note": "Non-causal diagnostic only. It uses same-session future labels to pick the best expert and is not a deployable selector.",
    }


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if predictions.empty:
        return out
    for year, group in predictions.groupby("year"):
        corr = group[["pred_exec_top20", "target_exec_top20"]].corr().iloc[0, 1]
        chosen = group.sort_values(["session", "pred_exec_top20", "expert"], ascending=[True, False, True]).groupby("session").head(1)
        out[str(int(year))] = {
            "rows": int(len(group)),
            "pred_target_corr": float(corr) if np.isfinite(corr) else 0.0,
            "chosen_mean_target_exec_top20": safe_mean(chosen["target_exec_top20"]),
            "all_mean_target_exec_top20": safe_mean(group["target_exec_top20"]),
            "chosen_expert_counts": {str(k): int(v) for k, v in chosen["expert"].value_counts().sort_index().items()},
        }
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
            "entry_ok": float(label.get("entry_ok", 0.0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows


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


def max_cluster_share(selected: pd.DataFrame) -> float:
    if selected.empty or "cluster_id" not in selected:
        return 0.0
    return float(selected["cluster_id"].value_counts().max() / len(selected))


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else 0.0


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f}"
        )
    print("selector_prediction_summary", json.dumps(result["selector_prediction_summary"], ensure_ascii=False, sort_keys=True)[:1200])


if __name__ == "__main__":
    raise SystemExit(main())
