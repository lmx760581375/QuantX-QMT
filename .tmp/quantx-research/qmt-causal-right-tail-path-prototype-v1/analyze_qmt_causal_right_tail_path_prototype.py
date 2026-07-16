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
from sklearn.cluster import MiniBatchKMeans


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"
TOPKS = (10, 20)
RANDOM_SEED = 20260714
LAGS = (1, 2, 3, 5, 10)
PROTO_KS = (16, 32, 64)
TAIL_SPECS = ((10, 10), (20, 20), (40, 40))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module("exp130_for_causal_right_tail_path_prototype", EXP130_PATH)
BASE_FEATURES = list(EXP130.FEATURES)
PATH_BASES = ["ret1_rank", "ret5_rank", "ret20_rank", "amount_rank", "amt_ratio20_rank", "near_high20_rank", "close_strength_rank", "upper_wick_low_rank", "vol20_low_rank", "range20_low_rank"]
PATH_FEATURES = [f"{name}_lag{lag}" for lag in LAGS for name in PATH_BASES]
DELTA_FEATURES = [
    "ret5_rank_chg_1_5", "ret20_rank_chg_1_5", "amount_rank_chg_1_5", "near_high_rank_chg_1_5",
    "close_strength_chg_1_5", "upper_wick_low_chg_1_5", "ret5_rank_chg_3_10", "ret20_rank_chg_3_10",
    "amount_rank_chg_3_10", "amt_ratio_rank_chg_3_10",
]
FEATURES = BASE_FEATURES + PATH_FEATURES + DELTA_FEATURES


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
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    panel = add_lagged_path_features(panel, market)
    scored, folds, diagnostics = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": classify_status(accounts),
        "experiment": "qmt_causal_right_tail_path_prototype_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "prototype_diagnostics": diagnostics,
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp40_due5_top20_dev_label": 0.014170,
            "exp40_due5_top20_2026_label": 0.012520,
            "exp134_leaf_mean_top20_multiple": 3.55,
            "exp134_leaf_mean_top20_2026_return": 0.0962,
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
            "lags": LAGS,
            "prototype_ks": PROTO_KS,
            "tail_specs": TAIL_SPECS,
            "feature_count": len(FEATURES),
            "causality": "For signal date T, lagged path ranks use only completed daily bars through T and earlier lag days. For each test year, winner/loser prototypes and supervised prototype-regressor train only on rows from years strictly before that test year; 2026 uses 2021-2025 only. T+1/T+6 open returns are used only as historical labels or out-of-sample evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_lagged_path_features(panel: pd.DataFrame, market: dict[str, Any]) -> pd.DataFrame:
    if panel.empty:
        return panel
    f = market["frames"]
    dates = market["dates"]
    date_index = market["date_index"]
    symbols = list(market["symbols"])
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
    amount = f["amount"]
    ret1 = close.pct_change(fill_method=None)
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    amount20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio20 = amount / amount20
    vol20 = ret1.rolling(20, min_periods=10).std()
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    near_high20 = close / high20
    blocks: list[pd.DataFrame] = []
    for session, group in panel.groupby("session", sort=True):
        idx = date_index[str(session)]
        out = group.copy()
        cols = np.asarray([symbol_index[str(sym)] for sym in out["instrument"]], dtype=int)
        for lag in LAGS:
            day_idx = max(0, idx - lag)
            values = daily_rank_values(ret1, ret5, ret20, amount, amt_ratio20, near_high20, close_strength, upper_wick, vol20, range20, day_idx)
            for name, arr in values.items():
                out[f"{name}_lag{lag}"] = arr[cols]
        out["ret5_rank_chg_1_5"] = out["ret5_rank_lag1"] - out["ret5_rank_lag5"]
        out["ret20_rank_chg_1_5"] = out["ret20_rank_lag1"] - out["ret20_rank_lag5"]
        out["amount_rank_chg_1_5"] = out["amount_rank_lag1"] - out["amount_rank_lag5"]
        out["near_high_rank_chg_1_5"] = out["near_high20_rank_lag1"] - out["near_high20_rank_lag5"]
        out["close_strength_chg_1_5"] = out["close_strength_rank_lag1"] - out["close_strength_rank_lag5"]
        out["upper_wick_low_chg_1_5"] = out["upper_wick_low_rank_lag1"] - out["upper_wick_low_rank_lag5"]
        out["ret5_rank_chg_3_10"] = out["ret5_rank_lag3"] - out["ret5_rank_lag10"]
        out["ret20_rank_chg_3_10"] = out["ret20_rank_lag3"] - out["ret20_rank_lag10"]
        out["amount_rank_chg_3_10"] = out["amount_rank_lag3"] - out["amount_rank_lag10"]
        out["amt_ratio_rank_chg_3_10"] = out["amt_ratio20_rank_lag3"] - out["amt_ratio20_rank_lag10"]
        blocks.append(out)
    return pd.concat(blocks, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.5)


def daily_rank_values(ret1: pd.DataFrame, ret5: pd.DataFrame, ret20: pd.DataFrame, amount: pd.DataFrame, amt_ratio20: pd.DataFrame, near_high20: pd.DataFrame, close_strength: pd.DataFrame, upper_wick: pd.DataFrame, vol20: pd.DataFrame, range20: pd.DataFrame, idx: int) -> dict[str, np.ndarray]:
    return {
        "ret1_rank": EXP130.rank_series(ret1.iloc[idx].to_numpy(dtype=float)),
        "ret5_rank": EXP130.rank_series(ret5.iloc[idx].to_numpy(dtype=float)),
        "ret20_rank": EXP130.rank_series(ret20.iloc[idx].to_numpy(dtype=float)),
        "amount_rank": EXP130.rank_series(np.log1p(amount.iloc[idx].to_numpy(dtype=float))),
        "amt_ratio20_rank": EXP130.rank_series(amt_ratio20.iloc[idx].to_numpy(dtype=float)),
        "near_high20_rank": EXP130.rank_series(near_high20.iloc[idx].to_numpy(dtype=float)),
        "close_strength_rank": EXP130.rank_series(close_strength.iloc[idx].to_numpy(dtype=float)),
        "upper_wick_low_rank": EXP130.rank_series(upper_wick.iloc[idx].to_numpy(dtype=float), ascending=False),
        "vol20_low_rank": EXP130.rank_series(vol20.iloc[idx].to_numpy(dtype=float), ascending=False),
        "range20_low_rank": EXP130.rank_series(range20.iloc[idx].to_numpy(dtype=float), ascending=False),
    }


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year]
        test = work[work["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        out = test.copy()
        prototype_scorers = fit_prototype_scorers(train, year)
        proto_scores, proto_diag = apply_prototype_scorers(prototype_scorers, test)
        for name, values in proto_scores.items():
            out[f"score_{name}"] = values
        proto_cols = [col for col in out.columns if col.startswith("score_proto_")]
        reg_features = FEATURES + proto_cols
        reg = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=5,
            min_child_samples=480,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=8.0,
            random_state=RANDOM_SEED + year,
            n_jobs=6,
            verbosity=-1,
        )
        train_proto, _ = apply_prototype_scorers(prototype_scorers, train)
        train_work = train.copy()
        for name, values in train_proto.items():
            train_work[f"score_{name}"] = values
        reg.fit(train_work[reg_features], train_work["exec_label5_open"])
        out["score_path_reg"] = reg.predict(out[reg_features])
        best_proto = best_proto_col(train_work, proto_cols)
        out["score_proto_best_blend"] = 0.65 * rank_by_session(out, "score_path_reg") + 0.35 * rank_by_session(out, best_proto)
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_years": [int(y) for y in sorted(train["year"].unique())],
            "proto_columns": int(len(proto_cols)),
        })
        diagnostics.append({"year": int(year), "prototype_grid": proto_diag, "reg_top_features": top_reg_features(reg, reg_features, 25)})
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, diagnostics


def fit_prototype_scorers(train: pd.DataFrame, year: int) -> list[dict[str, Any]]:
    train = train.reset_index(drop=True)
    train_x = train[FEATURES].to_numpy(dtype=np.float32)
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    tx = ((train_x - mu) / sd).astype(np.float32)
    y = train["exec_label5_open"].to_numpy(dtype=np.float32)
    session_groups = list(train.groupby("session", sort=True).indices.values())
    scorers: list[dict[str, Any]] = []
    for winner_n, loser_n in TAIL_SPECS:
        win_idx, lose_idx = collect_session_tails(session_groups, y, winner_n, loser_n)
        for k in PROTO_KS:
            if len(win_idx) < k or len(lose_idx) < k:
                continue
            win_proto = fit_kmeans(tx[win_idx], k, RANDOM_SEED + year + k + winner_n)
            lose_proto = fit_kmeans(tx[lose_idx], k, RANDOM_SEED + year + 77 + k + loser_n)
            scorers.append({
                "name": f"proto_w{winner_n}_l{loser_n}_k{k}",
                "mu": mu,
                "sd": sd,
                "winner_proto": win_proto,
                "loser_proto": lose_proto,
                "winner_samples": int(len(win_idx)),
                "loser_samples": int(len(lose_idx)),
            })
    return scorers


def apply_prototype_scorers(scorers: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    scores: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for scorer in scorers:
        vx = ((x - scorer["mu"]) / scorer["sd"]).astype(np.float32)
        raw = nearest_proto_score(vx, scorer["winner_proto"], scorer["loser_proto"])
        scores[scorer["name"]] = raw
        diagnostics.append({
            "name": scorer["name"],
            "winner_samples": int(scorer["winner_samples"]),
            "loser_samples": int(scorer["loser_samples"]),
            "score_mean": float(np.mean(raw)),
            "score_std": float(np.std(raw)),
        })
    return scores, diagnostics


def collect_session_tails(session_groups: list[np.ndarray], y: np.ndarray, winner_n: int, loser_n: int) -> tuple[np.ndarray, np.ndarray]:
    winners: list[np.ndarray] = []
    losers: list[np.ndarray] = []
    for idx in session_groups:
        vals = y[idx]
        order = np.argsort(vals, kind="mergesort")
        losers.append(idx[order[: min(loser_n, len(order))]])
        winners.append(idx[order[-min(winner_n, len(order)):]])
    return np.concatenate(winners), np.concatenate(losers)


def fit_kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192, max_iter=80, n_init=3, random_state=seed, reassignment_ratio=0.01)
    km.fit(x)
    centers = km.cluster_centers_.astype(np.float32)
    norm = np.linalg.norm(centers, axis=1, keepdims=True)
    return centers / np.maximum(norm, 1e-8)


def nearest_proto_score(x: np.ndarray, winners: np.ndarray, losers: np.ndarray, chunk: int = 20000) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    nx = x / np.maximum(norm, 1e-8)
    out = np.empty(len(nx), dtype=np.float32)
    for start in range(0, len(nx), chunk):
        block = nx[start:start + chunk]
        win = np.max(block @ winners.T, axis=1)
        lose = np.max(block @ losers.T, axis=1)
        out[start:start + len(block)] = win - lose
    return out


def rank_by_session(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("session")[col].rank(pct=True, method="first")


def best_proto_col(frame: pd.DataFrame, cols: list[str]) -> str:
    if not cols:
        return "score_path_reg"
    means = [(col, float(frame.groupby("session")[col].rank(pct=True).corr(frame["exec_label5_open"]))) for col in cols]
    means.sort(key=lambda item: np.nan_to_num(item[1], nan=-999.0), reverse=True)
    return means[0][0]


def top_reg_features(model: LGBMRegressor, features: list[str], n: int) -> list[dict[str, Any]]:
    rows = [{"feature": name, "importance": float(value)} for name, value in zip(features, model.feature_importances_)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:n]


def variants(scored: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    out = [("path_reg", "score_path_reg"), ("proto_best_blend", "score_proto_best_blend")]
    if scored is not None:
        proto_cols = sorted(col for col in scored.columns if col.startswith("score_proto_") and col != "score_proto_best_blend")
        out.extend((col.replace("score_", ""), col) for col in proto_cols[:9])
    return out


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants(scored):
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
    for variant, score_col in variants(scored):
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
