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
from sklearn.cluster import KMeans


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP129_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-supply-demand-rebalance-v1/analyze_qmt_supply_demand_rebalance.py"
RANDOM_SEED = 20260714
TOPKS = (10, 15, 20, 30)
COHORT_TOPS = (20, 30)
CLUSTER_COUNTS = (12, 24, 36)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP129 = load_module("exp129_for_right_tail_cohort_subspace", EXP129_PATH)
BASE_FEATURES = list(EXP129.FEATURES)
MARKET_FEATURES = ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20"]
STOCK_FEATURES = [name for name in BASE_FEATURES if name not in MARKET_FEATURES]
SUBSPACE_FEATURES = [
    "subspace_max", "subspace_top3", "subspace_weighted", "subspace_contrast",
    "subspace_margin", "subspace_stable", "subspace_neg_distance", "subspace_market_fit",
]
FEATURES = BASE_FEATURES + SUBSPACE_FEATURES


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
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP129.EXP124.load_market(args)
    panel = EXP129.build_panel(market, args)
    panel = add_targets(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    stress = stress_diagnostics(accounts)
    result = {
        "status": classify_status(accounts, stress),
        "experiment": "qmt_right_tail_cohort_subspace_generator_v1",
        "source_experiment": "qmt_supply_demand_rebalance_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail, stress),
        "stress_diagnostics": stress,
        "feature_importance": summarize_importance(importances),
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp134_leaf_mean_top20_multiple": 3.55,
            "exp134_leaf_mean_top20_2026": 0.0962,
            "exp148_cluster_rank_top20_multiple": 3.9764,
            "exp148_cluster_rank_top20_2026": 0.0331,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "cohort_tops": COHORT_TOPS,
            "cluster_counts": CLUSTER_COUNTS,
            "feature_count": len(FEATURES),
            "causality": "For each test year, winner cohort subspaces are built only from sessions in years strictly before that test year; 2026 uses 2021-2025 only. Test-day stock scores use completed daily bars through signal date T only. T+1/T+6 open data is used only as historical labels and OOS account replay.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static concept tables, or other message data.",
            "interpretation_guardrail": "This experiment does not predict high-opportunity days, does not use test-year winner cohorts, and does not reduce TopK or hard-filter tails. It evaluates whether historical winner-cohort subspace directions naturally thicken Top20.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    ranks = out.groupby("session")["exec_label5_open"].rank(pct=True, method="first")
    out["right_tail_target"] = (ranks >= 0.95).astype(int)
    return out


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
        library = build_cohort_library(train, year)
        out = add_subspace_scores(test, library)
        reg = make_regressor(year, 0, 280)
        cls = make_classifier(year, 101, 240)
        reg_aug = make_regressor(year, 211, 300)
        cls_aug = make_classifier(year, 307, 260)
        reg.fit(train[BASE_FEATURES], train["exec_label5_open"])
        cls.fit(train[BASE_FEATURES], train["right_tail_target"])
        train_aug = add_subspace_scores(train, library)
        reg_aug.fit(train_aug[FEATURES], train_aug["exec_label5_open"])
        cls_aug.fit(train_aug[FEATURES], train_aug["right_tail_target"])
        out["score_base_reg"] = reg.predict(out[BASE_FEATURES])
        out["score_base_right_tail"] = cls.predict_proba(out[BASE_FEATURES])[:, 1]
        out["score_aug_reg"] = reg_aug.predict(out[FEATURES])
        out["score_aug_right_tail"] = cls_aug.predict_proba(out[FEATURES])[:, 1]
        out["score_subspace_max"] = out["subspace_max"]
        out["score_subspace_weighted"] = out["subspace_weighted"]
        out["score_subspace_contrast"] = out["subspace_contrast"]
        out["score_subspace_margin"] = out["subspace_margin"]
        out["score_subspace_stable"] = out["subspace_stable"]
        out["score_subspace_blend"] = blend_ranks(
            out,
            [("score_subspace_max", 0.25), ("score_subspace_weighted", 0.25), ("score_subspace_contrast", 0.20), ("score_subspace_margin", 0.18), ("score_subspace_stable", 0.12)],
        )
        out["score_aug_blend"] = blend_ranks(
            out,
            [("score_aug_reg", 0.36), ("score_aug_right_tail", 0.24), ("score_subspace_blend", 0.22), ("score_base_reg", 0.18)],
        )
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "library_rows": int(library["centers"].shape[0]),
            "library_clusters": int(library["cluster_count"]),
            "library_mean_payoff": float(np.mean(library["payoff"])),
            "library_min_years_per_cluster": int(np.min(library["year_count"])),
        })
        imp = {f"base_reg::{name}": float(value) for name, value in zip(BASE_FEATURES, reg.feature_importances_)}
        imp.update({f"base_right_tail::{name}": float(value) for name, value in zip(BASE_FEATURES, cls.feature_importances_)})
        imp.update({f"aug_reg::{name}": float(value) for name, value in zip(FEATURES, reg_aug.feature_importances_)})
        imp.update({f"aug_right_tail::{name}": float(value) for name, value in zip(FEATURES, cls_aug.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def build_cohort_library(train: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    session_rows: list[dict[str, Any]] = []
    for session, group in train.groupby("session", sort=True):
        g = session_standardize(group)
        values = g[STOCK_FEATURES].to_numpy(dtype=np.float32)
        labels = g["exec_label5_open"].to_numpy(dtype=np.float32)
        for topn in COHORT_TOPS:
            if len(g) < topn * 2:
                continue
            order = np.argsort(labels, kind="mergesort")
            loser_idx = order[:topn]
            winner_idx = order[-topn:]
            winner_mean = values[winner_idx].mean(axis=0)
            loser_mean = values[loser_idx].mean(axis=0)
            pool_mean = values.mean(axis=0)
            contrast = winner_mean - loser_mean
            direction = winner_mean - pool_mean
            norm = np.linalg.norm(contrast)
            if not np.isfinite(norm) or norm <= 1e-6:
                continue
            session_rows.append({
                "session": str(session),
                "year": int(str(session)[:4]),
                "topn": int(topn),
                "center": winner_mean,
                "contrast": contrast / norm,
                "direction": direction,
                "loser": loser_mean,
                "payoff": float(labels[winner_idx].mean()),
                "market": group[MARKET_FEATURES].iloc[0].to_numpy(dtype=np.float32),
            })
    if not session_rows:
        raise RuntimeError("empty cohort library")
    centers = np.vstack([row["center"] for row in session_rows]).astype(np.float32)
    contrasts = np.vstack([row["contrast"] for row in session_rows]).astype(np.float32)
    directions = np.vstack([row["direction"] for row in session_rows]).astype(np.float32)
    losers = np.vstack([row["loser"] for row in session_rows]).astype(np.float32)
    market = np.vstack([row["market"] for row in session_rows]).astype(np.float32)
    payoff = np.asarray([row["payoff"] for row in session_rows], dtype=np.float32)
    years = np.asarray([row["year"] for row in session_rows], dtype=np.int32)
    cluster_count = choose_cluster_count(len(session_rows))
    km = KMeans(n_clusters=cluster_count, n_init=10, random_state=RANDOM_SEED + year)
    labels = km.fit_predict(np.hstack([centers, contrasts]).astype(np.float32))
    cluster_rows: list[dict[str, Any]] = []
    global_payoff = float(np.mean(payoff))
    for cluster in range(cluster_count):
        mask = labels == cluster
        if int(mask.sum()) == 0:
            continue
        count = int(mask.sum())
        shrink = min(count / 8.0, 1.0)
        c_payoff = shrink * float(payoff[mask].mean()) + (1.0 - shrink) * global_payoff
        year_count = len(set(int(y) for y in years[mask]))
        stability = min(year_count / 3.0, 1.0) * min(count / 12.0, 1.0)
        cluster_rows.append({
            "center": centers[mask].mean(axis=0),
            "contrast": normalize_vector(contrasts[mask].mean(axis=0)),
            "direction": directions[mask].mean(axis=0),
            "loser": losers[mask].mean(axis=0),
            "market": market[mask].mean(axis=0),
            "payoff": c_payoff,
            "count": count,
            "year_count": year_count,
            "stability": stability,
        })
    return {
        "centers": np.vstack([row["center"] for row in cluster_rows]).astype(np.float32),
        "contrasts": np.vstack([row["contrast"] for row in cluster_rows]).astype(np.float32),
        "directions": np.vstack([row["direction"] for row in cluster_rows]).astype(np.float32),
        "losers": np.vstack([row["loser"] for row in cluster_rows]).astype(np.float32),
        "market": np.vstack([row["market"] for row in cluster_rows]).astype(np.float32),
        "payoff": np.asarray([row["payoff"] for row in cluster_rows], dtype=np.float32),
        "count": np.asarray([row["count"] for row in cluster_rows], dtype=np.float32),
        "year_count": np.asarray([row["year_count"] for row in cluster_rows], dtype=np.int32),
        "stability": np.asarray([row["stability"] for row in cluster_rows], dtype=np.float32),
        "cluster_count": np.asarray(cluster_count, dtype=np.int32),
    }


def choose_cluster_count(n_rows: int) -> int:
    valid = [k for k in CLUSTER_COUNTS if k <= max(2, n_rows // 3)]
    return int(valid[-1] if valid else min(8, n_rows))


def normalize_vector(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x / norm).astype(np.float32)


def session_standardize(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    for col in STOCK_FEATURES:
        values = out[col].to_numpy(dtype=np.float32)
        mu = float(np.nanmean(values))
        sd = float(np.nanstd(values))
        if not np.isfinite(sd) or sd <= 1e-6:
            sd = 1.0
        out[col] = np.nan_to_num((values - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def add_subspace_scores(frame: pd.DataFrame, library: dict[str, np.ndarray]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for _session, group in frame.groupby("session", sort=False):
        g = session_standardize(group)
        x = g[STOCK_FEATURES].to_numpy(dtype=np.float32)
        market = g[MARKET_FEATURES].iloc[0].to_numpy(dtype=np.float32)
        scores = score_against_library(x, market, library)
        out = group.copy()
        for name, values in scores.items():
            out[name] = values.astype(float)
        blocks.append(out)
    return pd.concat(blocks, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def score_against_library(x: np.ndarray, market: np.ndarray, library: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    centers = library["centers"]
    contrasts = library["contrasts"]
    losers = library["losers"]
    cluster_market = library["market"]
    payoff = library["payoff"]
    stability = library["stability"]
    center_dist = squared_distance(x, centers)
    loser_dist = squared_distance(x, losers)
    contrast_proj = x @ contrasts.T
    market_dist = np.sum((cluster_market - market[None, :]) ** 2, axis=1)
    market_fit = np.exp(-market_dist / float(np.nanmedian(market_dist) + 1e-6))
    payoff_weight = rank01(payoff) * 0.65 + stability * 0.25 + market_fit * 0.10
    sim = -center_dist + 0.65 * contrast_proj + 0.35 * loser_dist
    weighted_sim = sim * payoff_weight[None, :]
    top3 = np.sort(weighted_sim, axis=1)[:, -min(3, weighted_sim.shape[1]):]
    best_idx = np.argmax(weighted_sim, axis=1)
    sorted_sim = np.sort(weighted_sim, axis=1)
    margin = sorted_sim[:, -1] - sorted_sim[:, -2] if weighted_sim.shape[1] > 1 else sorted_sim[:, -1]
    stable_sim = sim * stability[None, :]
    market_sim = sim * market_fit[None, :]
    return {
        "subspace_max": rank01(np.max(sim, axis=1)),
        "subspace_top3": rank01(np.mean(top3, axis=1)),
        "subspace_weighted": rank01(np.max(weighted_sim, axis=1)),
        "subspace_contrast": rank01(contrast_proj[np.arange(len(x)), best_idx]),
        "subspace_margin": rank01(margin),
        "subspace_stable": rank01(np.max(stable_sim, axis=1)),
        "subspace_neg_distance": rank01(-np.min(center_dist, axis=1)),
        "subspace_market_fit": rank01(np.max(market_sim, axis=1)),
    }


def squared_distance(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum(x * x, axis=1, keepdims=True) + np.sum(centers * centers, axis=1)[None, :] - 2.0 * x @ centers.T


def rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    return series.rank(pct=True, method="first").fillna(0.5).to_numpy(dtype=np.float32)


def make_regressor(year: int, offset: int, estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=estimators, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=520, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def make_classifier(year: int, offset: int, estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=estimators, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=500, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def blend_ranks(frame: pd.DataFrame, parts: list[tuple[str, float]]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in parts:
        score = score + weight * frame.groupby("session")[col].rank(pct=True, method="first")
    return score


def variants() -> list[tuple[str, str]]:
    return [
        ("subspace_max", "score_subspace_max"),
        ("subspace_weighted", "score_subspace_weighted"),
        ("subspace_contrast", "score_subspace_contrast"),
        ("subspace_margin", "score_subspace_margin"),
        ("subspace_stable", "score_subspace_stable"),
        ("subspace_blend", "score_subspace_blend"),
        ("base_reg", "score_base_reg"),
        ("aug_reg", "score_aug_reg"),
        ("aug_right_tail", "score_aug_right_tail"),
        ("aug_blend", "score_aug_blend"),
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
                    "mean_label5_open": float(selected["label5_open"].mean()),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
                    "exit_ok": float(selected["exit_ok"].mean()),
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
            account = EXP129.replay_open_selection(selections, market, horizon)
            account["period_curve"] = account_period_curve(selections, market, horizon)
            out[f"{variant}::top{topk}"] = account
    return out


def account_period_curve(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows: list[dict[str, Any]] = []
    nav = 1.0
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets: list[float] = []
        holds: list[int] = []
        for sym in candidates:
            col = symbol_index.get(sym)
            if col is None:
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry / preclose - 1.0 >= 0.095:
                continue
            sell_idx = exit_idx
            while sell_idx < min(len(dates), exit_idx + 6):
                sell_open = float(arr["open"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if np.isfinite(sell_open) and sell_open > 0 and np.isfinite(sell_preclose) and sell_preclose > 0 and np.isfinite(sell_vol) and sell_vol > 0 and sell_open / sell_preclose - 1.0 > -0.095:
                    break
                sell_idx += 1
            if sell_idx >= min(len(dates), exit_idx + 6):
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        period_return = float(np.mean(rets)) if rets else 0.0
        nav *= 1.0 + period_return
        rows.append({
            "session": str(session),
            "year": int(str(session)[:4]),
            "period_return": period_return,
            "selected_count": int(len(rets)),
            "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
            "nav": float(nav),
        })
    return rows


def stress_diagnostics(accounts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, account in accounts.items():
        curve = account.get("period_curve", [])
        returns = np.asarray([row["period_return"] for row in curve], dtype=float)
        if returns.size == 0:
            continue
        rows: dict[str, Any] = {}
        for n in (1, 3, 5):
            if returns.size <= n:
                rows[f"remove_best_{n}_final_multiple"] = 1.0
                continue
            keep = np.ones(returns.size, dtype=bool)
            keep[np.argsort(returns)[-n:]] = False
            rows[f"remove_best_{n}_final_multiple"] = float(np.prod(1.0 + returns[keep]))
        rows["best_period_return"] = float(np.max(returns))
        rows["worst_period_return"] = float(np.min(returns))
        rows["positive_period_ratio"] = float(np.mean(returns > 0))
        out[key] = rows
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame, stress: dict[str, Any]) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows: list[dict[str, Any]] = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
        stress_row = stress.get(key, {})
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
            "remove_best_3_final_multiple": float(stress_row.get("remove_best_3_final_multiple", np.nan)),
            "remove_best_5_final_multiple": float(stress_row.get("remove_best_5_final_multiple", np.nan)),
            "positive_period_ratio": float(stress_row.get("positive_period_ratio", np.nan)),
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


def classify_status(accounts: dict[str, Any], stress: dict[str, Any]) -> str:
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        robust_after_best3 = stress.get(key, {}).get("remove_best_3_final_multiple", 0.0)
        if (
            account.get("final_multiple", 0.0) >= 20.0
            and annual.get("2026", -1.0) > 0.0
            and account.get("all_years_positive", False)
            and account.get("avg_selected_count", 0.0) > 5.0
            and 4.5 <= account.get("avg_hold_days", 0.0) <= 11.0
            and robust_after_best3 >= 10.0
        ):
            return "candidate_needs_deeper_audit_before_merge"
    return "rejected_before_formal_candidate"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"][:20]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f} "
            f"rm_best3={row['remove_best_3_final_multiple']:.2f}x"
        )


if __name__ == "__main__":
    raise SystemExit(main())
