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
EXP124_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-close-execution-formal-proxy-v1/analyze_qmt_close_execution_formal_proxy.py"


def load_exp124() -> Any:
    spec = importlib.util.spec_from_file_location("exp124", EXP124_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP124_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp124"] = module
    spec.loader.exec_module(module)
    return module


EXP124 = load_exp124()

FEATURES = [
    "ret5_rank", "ret20_rank", "ret60_rank", "amount_rank", "amt_ratio20_rank",
    "best_corr_rank", "best_beta_rank", "anchor_ret20_rank", "anchor_ret5_rank", "anchor_amount_rank",
    "cluster_size_rank", "cluster_breadth20_rank", "cluster_ret20_rank", "cluster_amt_ratio_rank",
    "lag_to_anchor_rank", "resid20_rank", "resid5_rank", "follower_lag_rank",
    "leader_in_cluster_rank", "cluster_leader_continue", "cluster_follower_catchup",
    "cluster_lag_repair", "cluster_quality", "mkt_ret20_median", "mkt_breadth20", "mkt_disp20",
]
TOPKS = (10, 15, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--anchor-count", type=int, default=40)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    market = EXP124.load_market(args)
    panel = build_panel(market, args)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "folds": folds,
        "feature_importance": summarize_importance(importances),
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "corr_window": args.corr_window,
            "anchor_count": args.anchor_count,
            "features": FEATURES,
            "causality": "For each signal T, dynamic clusters use only returns/amount through completed T. Anchors are selected from T-known leader scores. Labels and replay use pre-scheduled T+1 open to T+6 open. Execution-day data is used only for realization/rejection.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(result, "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest())
    return 0


def rank_series(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).rank(pct=True, ascending=ascending).fillna(0.5).to_numpy(dtype=float)


def corr_beta(stock_window: np.ndarray, anchor_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = stock_window.astype(float)
    y = anchor_window.astype(float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y, axis=0, keepdims=True)
    x_std = np.nanstd(x, axis=0) + 1e-12
    y_var = np.nanmean(y * y, axis=0) + 1e-12
    cov = np.nanmean(x[:, :, None] * y[:, None, :], axis=0)
    corr = cov / (x_std[:, None] * np.sqrt(y_var)[None, :])
    beta = cov / y_var[None, :]
    return corr, beta


def build_panel(market: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    f = market["frames"]
    dates = market["dates"]
    symbols = market["symbols"]
    arr = market["arrays"]
    close = f["close"]
    open_ = f["open"]
    amount = f["amount"]
    ret1 = close.pct_change(fill_method=None)
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amt_ratio20 = amount / amount.rolling(20, min_periods=10).mean()
    history = (open_.notna() & (open_ > 0)).cumsum()
    sampled = [day for day in dates if args.start <= day <= args.end]
    sampled = set(sampled[:: args.sample_step])
    symbols_arr = np.asarray(symbols, dtype=object)
    frames: list[pd.DataFrame] = []
    for day in dates:
        if day not in sampled:
            continue
        idx = market["date_index"][day]
        entry_idx = idx + 1
        exit_idx = entry_idx + args.horizon
        start_idx = idx - args.corr_window + 1
        if start_idx < 1 or idx < 130 or exit_idx >= len(dates):
            continue
        raw = arr["open"][exit_idx] / arr["open"][entry_idx] - 1.0
        finite = np.isfinite(raw)
        if finite.sum() < 500:
            continue
        r5 = ret5.iloc[idx].to_numpy(dtype=float)
        r20 = ret20.iloc[idx].to_numpy(dtype=float)
        r60 = ret60.iloc[idx].to_numpy(dtype=float)
        amt = amount.iloc[idx].to_numpy(dtype=float)
        amt_ratio = amt_ratio20.iloc[idx].to_numpy(dtype=float)
        anchor_score = rank_series(r20) + 0.35 * rank_series(np.log1p(amt)) + 0.25 * rank_series(r5) - 0.20 * rank_series(np.abs(r5), ascending=True)
        valid_anchor = np.isfinite(anchor_score) & np.isfinite(r20) & np.isfinite(amt) & (amt > 0)
        anchor_idx = np.asarray(sorted(np.flatnonzero(valid_anchor), key=lambda col: (-float(anchor_score[col]), str(symbols[col])))[: args.anchor_count], dtype=int)
        if len(anchor_idx) < 10:
            continue
        window = ret1.iloc[start_idx: idx + 1].to_numpy(dtype=float, copy=False)
        corr, beta = corr_beta(window, window[:, anchor_idx])
        best_anchor_pos = np.nanargmax(np.where(np.isfinite(corr), corr, -9.0), axis=1)
        best_anchor_idx = anchor_idx[best_anchor_pos]
        best_corr = corr[np.arange(corr.shape[0]), best_anchor_pos]
        best_beta = beta[np.arange(beta.shape[0]), best_anchor_pos]
        cluster_size = np.zeros(len(anchor_idx), dtype=float)
        cluster_breadth = np.zeros(len(anchor_idx), dtype=float)
        cluster_ret = np.zeros(len(anchor_idx), dtype=float)
        cluster_amt_ratio = np.zeros(len(anchor_idx), dtype=float)
        for pos in range(len(anchor_idx)):
            members = np.flatnonzero((best_anchor_pos == pos) & np.isfinite(best_corr) & (best_corr > 0.35))
            if len(members) == 0:
                cluster_size[pos] = 1.0
                cluster_breadth[pos] = 0.0
                cluster_ret[pos] = 0.0
                cluster_amt_ratio[pos] = 0.0
            else:
                cluster_size[pos] = float(len(members))
                cluster_breadth[pos] = float(np.nanmean(r20[members] > 0))
                cluster_ret[pos] = float(np.nanmedian(r20[members]))
                cluster_amt_ratio[pos] = float(np.nanmedian(amt_ratio[members]))
        anchor_ret20 = r20[best_anchor_idx]
        anchor_ret5 = r5[best_anchor_idx]
        anchor_amount = amt[best_anchor_idx]
        lag_to_anchor = anchor_ret20 - r20
        resid20 = r20 - best_beta * anchor_ret20
        resid5 = r5 - best_beta * anchor_ret5
        entry_gap = arr["open"][entry_idx] / arr["close"][idx] - 1.0
        exit_gap = arr["open"][exit_idx] / arr["close"][exit_idx - 1] - 1.0
        entry_ok = finite & np.isfinite(arr["volume"][entry_idx]) & (arr["volume"][entry_idx] > 0) & (entry_gap < 0.095) & (history.iloc[idx].to_numpy(dtype=float) >= 130) & (arr["is_st"][idx] < 0.5)
        exit_ok = finite & np.isfinite(arr["volume"][exit_idx]) & (arr["volume"][exit_idx] > 0) & (exit_gap > -0.095)
        universe_ok = finite & np.isfinite(arr["open"][idx]) & (arr["open"][idx] > 0) & (arr["is_st"][idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        if universe_ok.sum() < 500:
            continue
        bench = float(np.nanmean(raw[finite]))
        exec_label = np.where(entry_ok & exit_ok, raw - 0.00154, -0.08)
        data = {
            "ret5_rank": rank_series(r5),
            "ret20_rank": rank_series(r20),
            "ret60_rank": rank_series(r60),
            "amount_rank": rank_series(np.log1p(amt)),
            "amt_ratio20_rank": rank_series(amt_ratio),
            "best_corr_rank": rank_series(best_corr),
            "best_beta_rank": rank_series(best_beta),
            "anchor_ret20_rank": rank_series(anchor_ret20),
            "anchor_ret5_rank": rank_series(anchor_ret5),
            "anchor_amount_rank": rank_series(np.log1p(anchor_amount)),
            "cluster_size_rank": rank_series(cluster_size[best_anchor_pos]),
            "cluster_breadth20_rank": rank_series(cluster_breadth[best_anchor_pos]),
            "cluster_ret20_rank": rank_series(cluster_ret[best_anchor_pos]),
            "cluster_amt_ratio_rank": rank_series(cluster_amt_ratio[best_anchor_pos]),
            "lag_to_anchor_rank": rank_series(lag_to_anchor),
            "resid20_rank": rank_series(resid20),
            "resid5_rank": rank_series(resid5),
        }
        data["follower_lag_rank"] = 0.35 * data["best_corr_rank"] + 0.30 * data["lag_to_anchor_rank"] + 0.20 * data["cluster_breadth20_rank"] + 0.15 * data["amount_rank"]
        data["leader_in_cluster_rank"] = 0.30 * data["ret20_rank"] + 0.25 * data["anchor_ret20_rank"] + 0.20 * data["best_corr_rank"] + 0.15 * data["amount_rank"] + 0.10 * data["cluster_size_rank"]
        data["cluster_leader_continue"] = 0.30 * data["leader_in_cluster_rank"] + 0.25 * data["cluster_ret20_rank"] + 0.20 * data["cluster_breadth20_rank"] + 0.15 * data["amt_ratio20_rank"] + 0.10 * data["ret5_rank"]
        data["cluster_quality"] = 0.30 * data["cluster_breadth20_rank"] + 0.25 * data["cluster_size_rank"] + 0.20 * data["cluster_ret20_rank"] + 0.15 * data["cluster_amt_ratio_rank"] + 0.10 * data["anchor_amount_rank"]
        data["cluster_follower_catchup"] = 0.35 * data["follower_lag_rank"] + 0.25 * data["cluster_quality"] + 0.20 * data["resid5_rank"] + 0.20 * data["best_beta_rank"]
        data["cluster_lag_repair"] = 0.35 * data["lag_to_anchor_rank"] + 0.25 * data["cluster_quality"] + 0.20 * data["best_corr_rank"] + 0.20 * data["amt_ratio20_rank"]
        mkt_values = r20
        block: dict[str, Any] = {
            "session": np.full(int(universe_ok.sum()), day, dtype=object),
            "year": np.full(int(universe_ok.sum()), int(day[:4]), dtype=int),
            "instrument": symbols_arr[universe_ok],
            "raw5_open": raw[universe_ok],
            "label5_open": raw[universe_ok] - bench,
            "exec_label5_open": exec_label[universe_ok] - bench,
            "entry_ok": entry_ok[universe_ok].astype(float),
            "exit_ok": exit_ok[universe_ok].astype(float),
            "mkt_ret20_median": np.full(int(universe_ok.sum()), float(np.nanmedian(mkt_values))),
            "mkt_breadth20": np.full(int(universe_ok.sum()), float(np.nanmean(mkt_values > 0))),
            "mkt_disp20": np.full(int(universe_ok.sum()), float(np.nanstd(mkt_values))),
        }
        for name in FEATURES:
            if name.startswith("mkt_"):
                continue
            block[name] = np.asarray(data[name], dtype=float)[universe_ok]
        frames.append(pd.DataFrame(block).replace([np.inf, -np.inf], np.nan).fillna(0.5))
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel["top_exec"] = (panel.groupby("session")["exec_label5_open"].rank(pct=True, method="first") >= 0.95).astype(int)
    return panel


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, int]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds = []
    importances = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        reg = LGBMRegressor(n_estimators=300, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year), n_jobs=6, verbosity=-1)
        cls = LGBMClassifier(n_estimators=240, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year) + 23, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_cluster_follower"] = out["cluster_follower_catchup"]
        out["score_cluster_leader"] = out["cluster_leader_continue"]
        out["score_lag_repair"] = out["cluster_lag_repair"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [("reg", "score_reg"), ("cls", "score_cls"), ("blend", "score_blend"), ("cluster_follower", "score_cluster_follower"), ("cluster_leader", "score_cluster_leader"), ("lag_repair", "score_lag_repair")]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({"session": session, "year": int(session[:4]), "variant": variant, "topk": int(topk), "mean_label5_open": float(selected["label5_open"].mean()), "mean_exec_label5_open": float(selected["exec_label5_open"].mean()), "entry_ok": float(selected["entry_ok"].mean()), "count": int(len(selected))})
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {"mean_label5_open": float(group["mean_label5_open"].mean()), "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()), "entry_ok": float(group["entry_ok"].mean()), "by_year": {str(int(y)): {"mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()), "entry_ok": float(yg["entry_ok"].mean())} for y, yg in group.groupby("year")}}
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out = {}
    for variant, score_col in variants():
        for topk in (10, 20):
            selections = {session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"]) for session, group in scored.groupby("session", sort=True)}
            out[f"{variant}::top{topk}"] = replay_open_selection(selections, market, horizon)
    return out


def replay_open_selection(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows = []
    rejects = {"candidate_total": 0, "selected_total": 0, "entry_limit_up_or_missing": 0, "exit_limit_down_or_missing": 0, "empty_periods": 0}
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets = []
        holds = []
        for sym in candidates:
            rejects["candidate_total"] += 1
            col = symbol_index.get(sym)
            if col is None:
                rejects["entry_limit_up_or_missing"] += 1
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry / preclose - 1.0 >= 0.095:
                rejects["entry_limit_up_or_missing"] += 1
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
                rejects["exit_limit_down_or_missing"] += 1
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        if not rets:
            rejects["empty_periods"] += 1
            period_ret = 0.0
        else:
            period_ret = float(np.mean(rets))
        rejects["selected_total"] += len(rets)
        rows.append({"session": session, "year": int(session[:4]), "period_return": period_ret, "selected_count": len(rets), "avg_hold_days": float(np.mean(holds)) if holds else 0.0})
    if not rows:
        return {"final_multiple": 1.0, "total_return": 0.0, "rejects": rejects}
    nav = 1.0
    curve = []
    for row in rows:
        nav *= 1.0 + row["period_return"]
        curve.append({**row, "nav": nav})
    df = pd.DataFrame(rows)
    annual = {str(int(y)): float(np.prod(1.0 + g["period_return"].to_numpy(dtype=float)) - 1.0) for y, g in df.groupby("year")}
    navs = np.asarray([r["nav"] for r in curve], dtype=float)
    peak = np.maximum.accumulate(navs)
    return {"final_multiple": float(nav), "total_return": float(nav - 1.0), "annual_returns": annual, "all_years_positive": bool(all(v > 0 for v in annual.values())), "max_drawdown_period": float(np.min(navs / peak - 1.0)), "avg_selected_count": float(df["selected_count"].mean()), "avg_hold_days": float(df.loc[df["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (df["avg_hold_days"] > 0).any() else 0.0, "periods": int(len(rows)), "rejects": rejects}


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:60]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(checksum)
    rows = []
    for name, data in result["account_results"].items():
        rows.append((name, data.get("final_multiple", 1.0), data.get("annual_returns", {}).get("2026", np.nan), data.get("avg_selected_count", 0.0)))
    rows.sort(key=lambda item: item[1], reverse=True)
    for name, multiple, ret2026, avg_count in rows[:14]:
        print(f"{name}: final={multiple:.2f}x 2026={ret2026:+.4f} avg_count={avg_count:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
