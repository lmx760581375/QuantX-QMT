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
from lightgbm import LGBMRanker, LGBMRegressor


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP148_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/analyze_qmt_dynamic_theme_cluster_ranker.py"
TOPKS = (10, 20)
RANDOM_SEED = 20260714


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP148 = load_module("exp148_for_cluster_lag_repair_ranker", EXP148_PATH)
REPAIR_FEATURES = [
    "repair_lag_depth", "repair_confirmation", "repair_overheat_penalty",
    "repair_prior_raw", "repair_prior_rank", "cluster_residual_exec_label5_open",
]
FEATURES = list(EXP148.FEATURES) + REPAIR_FEATURES[:-1]


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
    panel = add_repair_training_features(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_cluster_lag_repair_ranker_v1",
        "source_experiments": [
            "qmt_dynamic_theme_cluster_ranker_v1",
            "qmt_cluster_rank_thickness_diagnostic_v1",
        ],
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "feature_importance": summarize_importance(importances),
        "repair_profile": summarize_repair_profile(scored),
        "references": {
            "exp148_cluster_rank_top10_multiple": 8.0478,
            "exp148_cluster_rank_top10_2026": 0.1763,
            "exp148_cluster_rank_top20_multiple": 3.9764,
            "exp148_cluster_rank_top20_2026": 0.0331,
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
        },
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
            "feature_count": len(FEATURES),
            "causality": "Dynamic clusters and repair features use completed daily bars through signal date T only. Each test year trains only on years strictly before that year; 2026 uses 2021-2025. Future T+1/T+6 open returns are used only as historical training targets and out-of-sample evaluation, never as test-year features or parameter selectors.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
            "interpretation_guardrail": "All blends are fixed before evaluation from Exp149 diagnostics; no 2026-specific routing, thresholds, TopK shrinkage, or hard gates are used.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_repair_training_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    lag_depth = (
        0.28 * (1.0 - out["cluster_rank_ret20"])
        + 0.24 * (1.0 - out["cluster_rank_ret60"])
        + 0.20 * (1.0 - out["cluster_rank_near_high"])
        + 0.16 * out["cluster_corr_rank"]
        + 0.12 * out["cluster_breadth20"]
    )
    confirmation = (
        0.28 * out["cluster_rank_close_strength"]
        + 0.24 * out["cluster_rank_amt_ratio"]
        + 0.20 * out["cluster_rank_upper_wick_low"]
        + 0.16 * out["cluster_corr_rank"]
        + 0.12 * out["cluster_amount_mean"]
    )
    overheat = (
        0.30 * out["cluster_rank_ret5"]
        + 0.24 * out["cluster_rank_ret20"]
        + 0.20 * out["cluster_rank_near_high"]
        + 0.16 * out["cluster_rank_amt_ratio"]
        + 0.10 * (1.0 - out["cluster_rank_upper_wick_low"])
    )
    raw = 0.45 * lag_depth + 0.35 * confirmation + 0.20 * out["cluster_anchor_score"] - 0.35 * overheat
    out["repair_lag_depth"] = lag_depth.astype(float)
    out["repair_confirmation"] = confirmation.astype(float)
    out["repair_overheat_penalty"] = overheat.astype(float)
    out["repair_prior_raw"] = raw.astype(float)
    out["repair_prior_rank"] = out.groupby("session")["repair_prior_raw"].rank(pct=True, method="first")
    out["cluster_residual_exec_label5_open"] = out["exec_label5_open"] - out.groupby("cluster_query")["exec_label5_open"].transform("mean")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        sorted_train = train.sort_values(["cluster_query", "instrument"])
        groups = sorted_train.groupby("cluster_query", sort=False).size().to_numpy(dtype=int)
        visible_weight = (0.65 + 1.55 * sorted_train["repair_prior_rank"].to_numpy(dtype=float)).clip(0.65, 2.20)

        base_ranker = make_ranker(year, offset=0, estimators=240)
        repair_ranker = make_ranker(year, offset=101, estimators=220)
        residual_reg = make_regressor(year, offset=211, estimators=280)
        weighted_reg = make_regressor(year, offset=307, estimators=280)

        base_ranker.fit(sorted_train[FEATURES], sorted_train["cluster_quintile"].astype(int), group=groups)
        repair_ranker.fit(
            sorted_train[FEATURES],
            sorted_train["cluster_quintile"].astype(int),
            group=groups,
            sample_weight=visible_weight,
        )
        residual_reg.fit(train[FEATURES], train["cluster_residual_exec_label5_open"])
        weighted_reg.fit(
            train[FEATURES],
            train["exec_label5_open"],
            sample_weight=(0.65 + 1.55 * train["repair_prior_rank"].to_numpy(dtype=float)).clip(0.65, 2.20),
        )

        out = test.copy()
        out["score_cluster_rank"] = base_ranker.predict(out[FEATURES])
        out["score_repair_rank"] = repair_ranker.predict(out[FEATURES])
        out["score_residual_reg"] = residual_reg.predict(out[FEATURES])
        out["score_weighted_reg"] = weighted_reg.predict(out[FEATURES])
        out["score_repair_prior"] = out["repair_prior_rank"]
        out["score_rank_residual_35"] = blend_ranks(out, [("score_cluster_rank", 0.65), ("score_residual_reg", 0.35)])
        out["score_rank_weighted_35"] = blend_ranks(out, [("score_cluster_rank", 0.65), ("score_weighted_reg", 0.35)])
        out["score_rank_repair_35"] = blend_ranks(out, [("score_cluster_rank", 0.65), ("score_repair_rank", 0.35)])
        out["score_repair_stack"] = blend_ranks(
            out,
            [("score_cluster_rank", 0.45), ("score_residual_reg", 0.25), ("score_weighted_reg", 0.20), ("score_repair_prior", 0.10)],
        )
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "cluster_queries_train": int(train["cluster_query"].nunique()),
            "cluster_queries_test": int(test["cluster_query"].nunique()),
            "mean_visible_repair_weight_train": float(np.mean(visible_weight)),
        })
        imp = {f"base_rank::{name}": float(value) for name, value in zip(FEATURES, base_ranker.feature_importances_)}
        imp.update({f"repair_rank::{name}": float(value) for name, value in zip(FEATURES, repair_ranker.feature_importances_)})
        imp.update({f"residual_reg::{name}": float(value) for name, value in zip(FEATURES, residual_reg.feature_importances_)})
        imp.update({f"weighted_reg::{name}": float(value) for name, value in zip(FEATURES, weighted_reg.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_ranker(year: int, offset: int, estimators: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank", n_estimators=estimators, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=260, subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=8.0,
        random_state=RANDOM_SEED + year + offset, n_jobs=6, verbosity=-1,
    )


def make_regressor(year: int, offset: int, estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=estimators, learning_rate=0.032, num_leaves=35, max_depth=6, min_child_samples=520,
        subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=8.0,
        random_state=RANDOM_SEED + year + offset, n_jobs=6, verbosity=-1,
    )


def blend_ranks(frame: pd.DataFrame, parts: list[tuple[str, float]]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in parts:
        score = score + weight * frame.groupby("session")[col].rank(pct=True, method="first")
    return score


def variants() -> list[tuple[str, str]]:
    return [
        ("cluster_rank", "score_cluster_rank"),
        ("repair_rank", "score_repair_rank"),
        ("residual_reg", "score_residual_reg"),
        ("weighted_reg", "score_weighted_reg"),
        ("repair_prior", "score_repair_prior"),
        ("rank_residual_35", "score_rank_residual_35"),
        ("rank_weighted_35", "score_rank_weighted_35"),
        ("rank_repair_35", "score_rank_repair_35"),
        ("repair_stack", "score_repair_stack"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": str(session),
                    "year": int(str(session)[:4]),
                    "variant": variant,
                    "topk": int(topk),
                    "mean_label5_open": safe_mean(selected["label5_open"]),
                    "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
                    "mean_raw5_open": safe_mean(selected["raw5_open"]),
                    "entry_ok": safe_mean(selected["entry_ok"]),
                    "exit_ok": safe_mean(selected["exit_ok"]),
                    "mean_repair_prior_rank": safe_mean(selected["repair_prior_rank"]),
                    "mean_repair_overheat_penalty": safe_mean(selected["repair_overheat_penalty"]),
                    "mean_cluster_rel_ret20": safe_mean(selected["cluster_rel_ret20"]),
                    "count": int(len(selected)),
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "mean_repair_prior_rank": float(group["mean_repair_prior_rank"].mean()),
            "mean_repair_overheat_penalty": float(group["mean_repair_overheat_penalty"].mean()),
            "mean_cluster_rel_ret20": float(group["mean_cluster_rel_ret20"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "mean_repair_prior_rank": float(yg["mean_repair_prior_rank"].mean()),
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
            "mean_repair_prior_rank": float(label.get("mean_repair_prior_rank", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:100]]


def summarize_repair_profile(scored: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if scored.empty:
        return out
    for bucket, group in scored.assign(repair_bucket=pd.qcut(scored["repair_prior_rank"], 5, labels=False, duplicates="drop")).groupby("repair_bucket"):
        out[str(int(bucket))] = {
            "rows": int(len(group)),
            "mean_exec_label5_open": safe_mean(group["exec_label5_open"]),
            "mean_raw5_open": safe_mean(group["raw5_open"]),
            "mean_repair_prior_rank": safe_mean(group["repair_prior_rank"]),
            "mean_cluster_rel_ret20": safe_mean(group["cluster_rel_ret20"]),
            "mean_overheat": safe_mean(group["repair_overheat_penalty"]),
        }
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
    for row in result["leaderboard"][:18]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f} "
            f"repair={row['mean_repair_prior_rank']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
