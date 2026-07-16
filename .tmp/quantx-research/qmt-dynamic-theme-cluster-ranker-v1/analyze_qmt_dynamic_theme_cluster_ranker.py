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
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"
TOPKS = (10, 20)
RANDOM_SEED = 20260714
ANCHOR_COUNT = 48
CORR_WINDOW = 60


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module("exp130_for_dynamic_theme_cluster_ranker", EXP130_PATH)
BASE_FEATURES = list(EXP130.FEATURES)
CLUSTER_FEATURES = [
    "cluster_size_rank", "cluster_anchor_score", "cluster_corr_anchor", "cluster_corr_rank", "cluster_ret20_mean",
    "cluster_ret60_mean", "cluster_amount_mean", "cluster_amt_ratio_mean", "cluster_near_high_mean",
    "cluster_breadth20", "cluster_disp20", "cluster_rel_ret5", "cluster_rel_ret20", "cluster_rel_ret60",
    "cluster_rel_amount", "cluster_rel_amt_ratio", "cluster_rel_near_high", "cluster_rank_ret5",
    "cluster_rank_ret20", "cluster_rank_ret60", "cluster_rank_amount", "cluster_rank_amt_ratio",
    "cluster_rank_near_high", "cluster_rank_close_strength", "cluster_rank_upper_wick_low",
    "cluster_lag_repair", "cluster_mid_follower", "cluster_leader_quality", "cluster_overheat_risk",
]
FEATURES = BASE_FEATURES + CLUSTER_FEATURES


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
    parser.add_argument("--anchor-count", type=int, default=ANCHOR_COUNT)
    parser.add_argument("--corr-window", type=int, default=CORR_WINDOW)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    panel = add_dynamic_cluster_features(panel, market, args.anchor_count, args.corr_window)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_dynamic_theme_cluster_ranker_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "feature_importance": summarize_importance(importances),
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp121_fixed_reg_top20_label_all": 0.008680,
            "exp147_proto_best_blend_top20_multiple": 2.6944,
            "exp147_proto_best_blend_top20_2026": 0.0298,
        },
        "config": {
            "source_experiment": str(EXP130_PATH.relative_to(REPO_ROOT)),
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "anchor_count": args.anchor_count,
            "corr_window": args.corr_window,
            "feature_count": len(FEATURES),
            "causality": "For each signal date T, anchors, return correlations, dynamic cluster assignment, cluster aggregates, and cluster-local ranks use completed daily bars through T only. Each test year trains only on rows from years strictly before that year; 2026 uses 2021-2025 only. T+1/T+6 open returns are used only as historical labels and out-of-sample evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_dynamic_cluster_features(panel: pd.DataFrame, market: dict[str, Any], anchor_count: int, corr_window: int) -> pd.DataFrame:
    if panel.empty:
        return panel
    close = market["frames"]["close"]
    ret1 = close.pct_change(fill_method=None)
    date_index = market["date_index"]
    symbol_index = {symbol: idx for idx, symbol in enumerate(market["symbols"])}
    frames: list[pd.DataFrame] = []
    for session, group in panel.groupby("session", sort=True):
        idx = date_index[str(session)]
        start = idx - corr_window + 1
        if start < 1:
            continue
        g = group.copy()
        cols = np.asarray([symbol_index[str(sym)] for sym in g["instrument"]], dtype=int)
        anchor_score = (
            0.30 * g["ret20_rank"].to_numpy(dtype=float)
            + 0.24 * g["ret60_rank"].to_numpy(dtype=float)
            + 0.20 * g["amount_rank"].to_numpy(dtype=float)
            + 0.16 * g["near_high20_rank"].to_numpy(dtype=float)
            + 0.10 * g["amt_ratio20_rank"].to_numpy(dtype=float)
        )
        anchor_pos = np.argsort(anchor_score, kind="mergesort")[-min(anchor_count, len(g)):]
        window = ret1.iloc[start:idx + 1].to_numpy(dtype=np.float32, copy=False)[:, cols]
        z = zscore_columns(window)
        anchor_z = z[:, anchor_pos]
        corr = (z.T @ anchor_z) / float(max(1, z.shape[0]))
        corr = np.nan_to_num(corr, nan=-1.0, posinf=1.0, neginf=-1.0)
        cluster_local = np.argmax(corr, axis=1).astype(np.int16)
        best_corr = corr[np.arange(len(g)), cluster_local]
        g["cluster_id"] = cluster_local.astype(int)
        g["cluster_corr_anchor"] = best_corr.astype(float)
        g["cluster_corr_rank"] = pd.Series(best_corr).rank(pct=True).to_numpy(dtype=float)
        g["cluster_anchor_score"] = anchor_score[anchor_pos[cluster_local]].astype(float)
        add_cluster_aggregates(g)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    out["cluster_query"] = out["session"].astype(str) + "#" + out["cluster_id"].astype(str)
    out["cluster_quintile"] = make_cluster_quintile(out)
    return out


def zscore_columns(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32, copy=True)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_cluster_aggregates(g: pd.DataFrame) -> None:
    grouped = g.groupby("cluster_id", sort=False)
    size = grouped["instrument"].transform("size").astype(float)
    g["cluster_size_rank"] = pd.Series(size).rank(pct=True).to_numpy(dtype=float)
    mean_cols = {
        "cluster_ret20_mean": "ret20_rank",
        "cluster_ret60_mean": "ret60_rank",
        "cluster_amount_mean": "amount_rank",
        "cluster_amt_ratio_mean": "amt_ratio20_rank",
        "cluster_near_high_mean": "near_high20_rank",
    }
    for out_col, src in mean_cols.items():
        g[out_col] = grouped[src].transform("mean").to_numpy(dtype=float)
    g["cluster_breadth20"] = grouped["ret20_rank"].transform(lambda s: float((s > 0.5).mean())).to_numpy(dtype=float)
    g["cluster_disp20"] = grouped["ret20_rank"].transform("std").fillna(0.0).to_numpy(dtype=float)
    g["cluster_rel_ret5"] = g["ret5_rank"] - grouped["ret5_rank"].transform("mean")
    g["cluster_rel_ret20"] = g["ret20_rank"] - g["cluster_ret20_mean"]
    g["cluster_rel_ret60"] = g["ret60_rank"] - g["cluster_ret60_mean"]
    g["cluster_rel_amount"] = g["amount_rank"] - g["cluster_amount_mean"]
    g["cluster_rel_amt_ratio"] = g["amt_ratio20_rank"] - g["cluster_amt_ratio_mean"]
    g["cluster_rel_near_high"] = g["near_high20_rank"] - g["cluster_near_high_mean"]
    for src, out_col in [
        ("ret5_rank", "cluster_rank_ret5"), ("ret20_rank", "cluster_rank_ret20"),
        ("ret60_rank", "cluster_rank_ret60"), ("amount_rank", "cluster_rank_amount"),
        ("amt_ratio20_rank", "cluster_rank_amt_ratio"), ("near_high20_rank", "cluster_rank_near_high"),
        ("close_strength_rank", "cluster_rank_close_strength"), ("upper_wick_low_rank", "cluster_rank_upper_wick_low"),
    ]:
        g[out_col] = grouped[src].rank(pct=True, method="first").to_numpy(dtype=float)
    g["cluster_lag_repair"] = (
        0.30 * g["cluster_corr_rank"]
        + 0.25 * (1.0 - g["cluster_rank_ret20"])
        + 0.18 * g["cluster_rank_ret5"]
        + 0.15 * g["cluster_rank_amt_ratio"]
        + 0.12 * g["cluster_rank_close_strength"]
    )
    g["cluster_mid_follower"] = (
        0.28 * g["cluster_corr_rank"]
        + 0.22 * g["cluster_rank_ret20"].clip(0.25, 0.80)
        + 0.20 * g["cluster_rank_near_high"]
        + 0.16 * g["cluster_rank_amount"]
        + 0.14 * g["cluster_rank_upper_wick_low"]
    )
    g["cluster_leader_quality"] = (
        0.26 * g["cluster_rank_ret20"]
        + 0.22 * g["cluster_rank_ret60"]
        + 0.18 * g["cluster_rank_amount"]
        + 0.18 * g["cluster_rank_near_high"]
        + 0.16 * g["cluster_rank_upper_wick_low"]
    )
    g["cluster_overheat_risk"] = (
        0.35 * g["cluster_rank_ret5"]
        + 0.25 * g["cluster_rank_amt_ratio"]
        + 0.20 * (1.0 - g["cluster_rank_upper_wick_low"])
        + 0.20 * g["cluster_rank_ret20"]
    )


def make_cluster_quintile(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("cluster_query")["exec_label5_open"].rank(pct=True, method="first")
    return np.floor(np.clip(ranks.to_numpy(dtype=float), 0.0, 0.999999) * 5).astype(int)


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year]
        test = work[work["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        reg = make_regressor(year)
        ranker = make_ranker(year)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        sorted_train = train.sort_values(["cluster_query", "instrument"])
        groups = sorted_train.groupby("cluster_query", sort=False).size().to_numpy(dtype=int)
        ranker.fit(sorted_train[FEATURES], sorted_train["cluster_quintile"].astype(int), group=groups)
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cluster_rank"] = ranker.predict(out[FEATURES])
        out["score_blend"] = 0.55 * rank_by_session(out, "score_reg") + 0.45 * rank_by_session(out, "score_cluster_rank")
        out["score_cluster_mid_follower"] = out["cluster_mid_follower"]
        out["score_cluster_lag_repair"] = out["cluster_lag_repair"]
        out["score_cluster_leader_quality"] = out["cluster_leader_quality"]
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "cluster_queries_train": int(train["cluster_query"].nunique()),
            "cluster_queries_test": int(test["cluster_query"].nunique()),
        })
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"rank::{name}": float(value) for name, value in zip(FEATURES, ranker.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_regressor(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=320, learning_rate=0.032, num_leaves=35, max_depth=6, min_child_samples=520,
        subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=8.0,
        random_state=RANDOM_SEED + year, n_jobs=6, verbosity=-1,
    )


def make_ranker(year: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank", n_estimators=240, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=260, subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=8.0,
        random_state=RANDOM_SEED + year + 101, n_jobs=6, verbosity=-1,
    )


def rank_by_session(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("session")[col].rank(pct=True, method="first")


def variants() -> list[tuple[str, str]]:
    return [
        ("reg", "score_reg"),
        ("cluster_rank", "score_cluster_rank"),
        ("blend", "score_blend"),
        ("cluster_mid_follower", "score_cluster_mid_follower"),
        ("cluster_lag_repair", "score_cluster_lag_repair"),
        ("cluster_leader_quality", "score_cluster_leader_quality"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": str(session), "year": int(str(session)[:4]), "variant": variant, "topk": int(topk),
                    "mean_label5_open": float(selected["label5_open"].mean()),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()), "exit_ok": float(selected["exit_ok"].mean()),
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
            out[f"{variant}::top{topk}"] = EXP130.replay_open_selection(selections, market, horizon)
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


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:80]]


def classify_status(accounts: dict[str, Any]) -> str:
    for account in accounts.values():
        annual = account.get("annual_returns", {})
        if account.get("final_multiple", 0.0) >= 8.7749 and annual.get("2026", -1.0) >= 0.1918 and account.get("all_years_positive", False) and account.get("avg_selected_count", 0.0) > 5:
            return "candidate_needs_deeper_audit_before_merge"
    return "rejected_before_formal_candidate"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"][:18]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"exec={row['mean_exec_label5_open']:+.6f} raw={row['mean_raw5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
