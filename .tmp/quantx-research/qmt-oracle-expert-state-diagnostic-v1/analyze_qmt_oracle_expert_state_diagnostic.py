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
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP151_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-session-opportunity-expert-selector-v1/analyze_qmt_session_opportunity_expert_selector.py"
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


EXP151 = load_module("exp151_for_oracle_expert_state", EXP151_PATH)


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
    parser.add_argument("--anchor-count", type=int, default=EXP151.EXP150.EXP148.ANCHOR_COUNT)
    parser.add_argument("--corr-window", type=int, default=EXP151.EXP150.EXP148.CORR_WINDOW)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP151.EXP150.EXP148.EXP130.EXP124.load_market(args)
    panel = EXP151.EXP150.EXP148.EXP130.build_panel(market, args)
    panel = EXP151.EXP150.EXP148.add_dynamic_cluster_features(panel, market, args.anchor_count, args.corr_window)
    panel = EXP151.EXP150.add_repair_training_features(panel)
    scored, stock_folds, stock_importances = EXP151.EXP150.walk_forward(panel, args.min_train_rows)
    expert_frame = add_oracle_targets(EXP151.build_expert_frame(scored))
    selections, selector_folds, predictions = causal_relative_selectors(scored, expert_frame)
    detail = EXP151.evaluate_selections(scored, selections)
    accounts = EXP151.replay_accounts(selections, market, args.horizon)
    oracle = EXP151.oracle_diagnostics(scored, expert_frame, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_oracle_expert_state_diagnostic_v1",
        "source_experiments": [
            "qmt_session_opportunity_expert_selector_v1",
            "qmt_cluster_lag_repair_ranker_v1",
        ],
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "expert_rows": int(len(expert_frame)),
        "stock_folds": stock_folds,
        "selector_folds": selector_folds,
        "label_results": EXP151.summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": EXP151.build_leaderboard(accounts, detail),
        "oracle_diagnostics": oracle,
        "oracle_state_diagnostics": oracle_state_diagnostics(expert_frame),
        "selector_prediction_summary": summarize_predictions(predictions),
        "stock_feature_importance_top30": EXP151.EXP150.summarize_importance(stock_importances)[:30],
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
            "experts": [name for name, _ in EXP151.EXPERTS],
            "targets": ["relative_reg", "win_classifier", "session_ranker"],
            "cold_start": "2022 has no prior OOS expert ledger, so causal selector variants default to cluster_rank for 2022. From 2023 onward each year trains only on completed prior OOS expert rows.",
            "causality": "Stock scores are annual walk-forward OOS. Expert-state models train only on years strictly before the test year; 2026 uses 2022-2025. Features are signal-date T market, score, and portfolio-profile fields. Future returns are used only as prior-year targets and OOS evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_oracle_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["session_mean_target"] = out.groupby("session")["target_exec_top20"].transform("mean")
    out["target_rel_exec_top20"] = out["target_exec_top20"] - out["session_mean_target"]
    out["session_max_target"] = out.groupby("session")["target_exec_top20"].transform("max")
    out["is_best_expert"] = (out["target_exec_top20"] >= out["session_max_target"] - 1e-12).astype(int)
    rank = out.groupby("session")["target_exec_top20"].rank(pct=True, method="first")
    out["target_rank_label"] = np.floor(np.clip(rank.to_numpy(dtype=float), 0.0, 0.999999) * 5).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def causal_relative_selectors(scored: pd.DataFrame, expert_frame: pd.DataFrame) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]], pd.DataFrame]:
    variants = [
        "relreg_top1", "relreg_top2_merge", "relreg_top3_merge",
        "wincls_top1", "wincls_top2_merge", "ranker_top1", "ranker_top2_merge",
        "consensus_top2_merge",
    ]
    selections: dict[str, dict[str, list[str]]] = {variant: {} for variant in variants}
    folds: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_cols = selector_feature_columns(expert_frame)
    scored_by_session = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for year in sorted(int(y) for y in expert_frame["year"].unique()):
        train = expert_frame[expert_frame["year"] < year].copy()
        test = expert_frame[expert_frame["year"] == year].copy()
        if train.empty:
            test["pred_rel_reg"] = 0.0
            test["pred_win_prob"] = 0.0
            test["pred_ranker"] = 0.0
            test.loc[test["expert"] == "cluster_rank", ["pred_rel_reg", "pred_win_prob", "pred_ranker"]] = 1.0
            mode = "cold_start_cluster_rank"
        else:
            rel_reg = make_rel_reg(year)
            win_cls = make_win_cls(year)
            ranker = make_ranker(year)
            rel_reg.fit(train[feature_cols], train["target_rel_exec_top20"])
            win_cls.fit(train[feature_cols], train["is_best_expert"])
            sorted_train = train.sort_values(["session", "expert"])
            groups = sorted_train.groupby("session", sort=False).size().to_numpy(dtype=int)
            ranker.fit(sorted_train[feature_cols], sorted_train["target_rank_label"].astype(int), group=groups)
            test["pred_rel_reg"] = rel_reg.predict(test[feature_cols])
            test["pred_win_prob"] = win_cls.predict_proba(test[feature_cols])[:, 1]
            test["pred_ranker"] = ranker.predict(test[feature_cols])
            mode = "prior_year_relative_models"
        test["pred_consensus"] = (
            0.40 * test.groupby("session")["pred_rel_reg"].rank(pct=True, method="first")
            + 0.35 * test.groupby("session")["pred_win_prob"].rank(pct=True, method="first")
            + 0.25 * test.groupby("session")["pred_ranker"].rank(pct=True, method="first")
        )
        prediction_frames.append(test)
        folds.append({
            "year": int(year),
            "mode": mode,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_sessions": int(train["session"].nunique()) if not train.empty else 0,
            "test_sessions": int(test["session"].nunique()),
            "train_years": sorted(int(y) for y in train["year"].unique()) if not train.empty else [],
        })
        for session, pred_group in test.groupby("session", sort=True):
            stock_group = scored_by_session[str(session)]
            selections["relreg_top1"][str(session)] = select_with_score(stock_group, pred_group, "pred_rel_reg", 1)
            selections["relreg_top2_merge"][str(session)] = select_with_score(stock_group, pred_group, "pred_rel_reg", 2)
            selections["relreg_top3_merge"][str(session)] = select_with_score(stock_group, pred_group, "pred_rel_reg", 3)
            selections["wincls_top1"][str(session)] = select_with_score(stock_group, pred_group, "pred_win_prob", 1)
            selections["wincls_top2_merge"][str(session)] = select_with_score(stock_group, pred_group, "pred_win_prob", 2)
            selections["ranker_top1"][str(session)] = select_with_score(stock_group, pred_group, "pred_ranker", 1)
            selections["ranker_top2_merge"][str(session)] = select_with_score(stock_group, pred_group, "pred_ranker", 2)
            selections["consensus_top2_merge"][str(session)] = select_with_score(stock_group, pred_group, "pred_consensus", 2)
    return selections, folds, pd.concat(prediction_frames, ignore_index=True)


def selector_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        "session", "year", "expert", "target_exec_top20", "target_raw_top20", "target_exec_top10",
        "session_mean_target", "target_rel_exec_top20", "session_max_target", "is_best_expert", "target_rank_label",
    }
    return [col for col in frame.columns if col not in exclude]


def make_rel_reg(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=180, learning_rate=0.035, num_leaves=15, max_depth=4, min_child_samples=24,
        subsample=0.90, colsample_bytree=0.90, reg_alpha=1.0, reg_lambda=10.0,
        random_state=RANDOM_SEED + year, n_jobs=4, verbosity=-1,
    )


def make_win_cls(year: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=160, learning_rate=0.035, num_leaves=15, max_depth=4, min_child_samples=20,
        subsample=0.90, colsample_bytree=0.90, reg_alpha=1.0, reg_lambda=10.0,
        random_state=RANDOM_SEED + year + 101, n_jobs=4, verbosity=-1,
    )


def make_ranker(year: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank", n_estimators=150, learning_rate=0.035, num_leaves=15, max_depth=4,
        min_child_samples=16, subsample=0.90, colsample_bytree=0.90, reg_alpha=1.0, reg_lambda=10.0,
        random_state=RANDOM_SEED + year + 211, n_jobs=4, verbosity=-1,
    )


def select_with_score(stock_group: pd.DataFrame, pred_group: pd.DataFrame, score_col: str, expert_count: int) -> list[str]:
    work = pred_group.copy()
    work["pred_exec_top20"] = work[score_col]
    ranked = work.sort_values(["pred_exec_top20", "expert"], ascending=[False, True]).head(expert_count)
    return EXP151.select_from_experts(stock_group, ranked, 20, 35 if expert_count > 1 else 25)


def oracle_state_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    best = frame[frame["is_best_expert"] == 1].copy()
    out["best_expert_counts_by_year"] = {
        str(int(year)): {str(k): int(v) for k, v in group["expert"].value_counts().sort_index().items()}
        for year, group in best.groupby("year")
    }
    out["best_expert_counts_all"] = {str(k): int(v) for k, v in best["expert"].value_counts().sort_index().items()}
    feature_cols = selector_feature_columns(frame)
    contrast_rows = []
    for col in feature_cols:
        if col.startswith("expert_is_"):
            continue
        b = safe_mean(best[col])
        o = safe_mean(frame[frame["is_best_expert"] == 0][col])
        sd = float(frame[col].std()) if col in frame else 0.0
        contrast_rows.append({"feature": col, "best_mean": b, "other_mean": o, "diff": b - o, "std_diff": (b - o) / sd if sd > 1e-12 else 0.0})
    contrast_rows.sort(key=lambda row: abs(row["std_diff"]), reverse=True)
    out["best_vs_other_feature_contrast_top40"] = contrast_rows[:40]
    out["expert_profile"] = {
        str(expert): {
            "rows": int(len(group)),
            "best_rate": float(group["is_best_expert"].mean()),
            "mean_target_exec_top20": safe_mean(group["target_exec_top20"]),
            "mean_rel_target": safe_mean(group["target_rel_exec_top20"]),
        }
        for expert, group in frame.groupby("expert")
    }
    return out


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for year, group in predictions.groupby("year"):
        year_data: dict[str, Any] = {"rows": int(len(group))}
        for pred_col in ["pred_rel_reg", "pred_win_prob", "pred_ranker", "pred_consensus"]:
            corr = group[[pred_col, "target_rel_exec_top20"]].corr().iloc[0, 1]
            chosen = group.sort_values(["session", pred_col, "expert"], ascending=[True, False, True]).groupby("session").head(1)
            year_data[pred_col] = {
                "corr_to_rel_target": float(corr) if np.isfinite(corr) else 0.0,
                "chosen_mean_target_exec_top20": safe_mean(chosen["target_exec_top20"]),
                "chosen_mean_rel_target": safe_mean(chosen["target_rel_exec_top20"]),
                "chosen_best_hit_rate": safe_mean(chosen["is_best_expert"]),
                "chosen_expert_counts": {str(k): int(v) for k, v in chosen["expert"].value_counts().sort_index().items()},
            }
        out[str(int(year))] = year_data
    return out


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanmean(arr))
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
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f}"
        )
    print("oracle_counts", json.dumps(result["oracle_state_diagnostics"].get("best_expert_counts_all", {}), ensure_ascii=False, sort_keys=True))
    print("prediction_summary", json.dumps(result["selector_prediction_summary"], ensure_ascii=False, sort_keys=True)[:1400])


if __name__ == "__main__":
    raise SystemExit(main())
