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
    "ret1_rank", "ret3_rank", "ret5_rank", "ret10_rank", "ret20_rank", "ret60_rank",
    "ret5_moderate_rank", "ret20_smooth_rank", "ret20_over_60_rank", "ret5_over_20_rank",
    "vol20_low_rank", "range20_low_rank", "amount_rank", "amt_ratio5_rank", "amt_ratio20_rank",
    "amount_persist_rank", "close_vs_vwap_rank", "vwap_support5_rank", "close_strength_rank",
    "lower_wick_rank", "upper_wick_low_rank", "near_high20_rank", "drawdown20_rank",
    "break20_rank", "limit_touch20_low_rank", "limit_close5_low_rank", "gap_risk_low_rank",
    "quiet_accumulation_rank", "latent_right_tail_rank", "noncrowded_trend_rank",
    "mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20",
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
            "features": FEATURES,
            "causality": "Features use completed T daily data only. Signal is formed after T close. Execution and labels use pre-scheduled T+1 open to T+6 open. Entry/exit tradability checks use execution-day prices only for replay/label realization, not for candidate selection.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(result, "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest())
    return 0


def rank_frame(frame: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, ascending=ascending).fillna(0.5)


def build_panel(market: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    f = market["frames"]
    dates = market["dates"]
    symbols = market["symbols"]
    open_ = f["open"]
    high = f["high"]
    low = f["low"]
    close = f["close"]
    volume = f["volume"]
    vwap = f["vwap"]
    amount = f["amount"]
    is_st = f["is_st"]

    ret1 = close / close.shift(1) - 1.0
    ret3 = close / close.shift(3) - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret10 = close / close.shift(10) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    daily_ret = close.pct_change(fill_method=None)
    vol20 = daily_ret.rolling(20, min_periods=10).std()
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    prev_high20 = high.shift(1).rolling(20, min_periods=10).max()
    amount5 = amount.rolling(5, min_periods=3).mean()
    amount20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio5 = amount / amount5
    amt_ratio20 = amount / amount20
    amount_persist = (amount > amount20).rolling(10, min_periods=5).mean()
    close_vs_vwap = close / vwap - 1.0
    vwap_support5 = (close > vwap).rolling(5, min_periods=3).mean()
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    lower_wick = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    near_high20 = close / high20
    drawdown20 = close / high20 - 1.0
    break20 = close / prev_high20 - 1.0
    limit_touch20 = ((high / close.shift(1) - 1.0) >= 0.095).rolling(20, min_periods=5).sum()
    limit_close5 = ((close / close.shift(1) - 1.0) >= 0.095).rolling(5, min_periods=2).sum()
    gap_risk = (open_ / close.shift(1) - 1.0).abs().rolling(10, min_periods=5).max()

    ret5_moderate = -((ret5 - 0.035).abs())
    ret20_smooth = ret20 - 0.75 * vol20
    ranks = {
        "ret1_rank": rank_frame(ret1),
        "ret3_rank": rank_frame(ret3),
        "ret5_rank": rank_frame(ret5),
        "ret10_rank": rank_frame(ret10),
        "ret20_rank": rank_frame(ret20),
        "ret60_rank": rank_frame(ret60),
        "ret5_moderate_rank": rank_frame(ret5_moderate),
        "ret20_smooth_rank": rank_frame(ret20_smooth),
        "ret20_over_60_rank": rank_frame(ret20 - ret60 / 3.0),
        "ret5_over_20_rank": rank_frame(ret5 - ret20 / 4.0),
        "vol20_low_rank": rank_frame(vol20, ascending=False),
        "range20_low_rank": rank_frame(range20, ascending=False),
        "amount_rank": rank_frame(np.log1p(amount)),
        "amt_ratio5_rank": rank_frame(amt_ratio5),
        "amt_ratio20_rank": rank_frame(amt_ratio20),
        "amount_persist_rank": rank_frame(amount_persist),
        "close_vs_vwap_rank": rank_frame(close_vs_vwap),
        "vwap_support5_rank": rank_frame(vwap_support5),
        "close_strength_rank": rank_frame(close_strength),
        "lower_wick_rank": rank_frame(lower_wick),
        "upper_wick_low_rank": rank_frame(upper_wick, ascending=False),
        "near_high20_rank": rank_frame(near_high20),
        "drawdown20_rank": rank_frame(drawdown20),
        "break20_rank": rank_frame(break20),
        "limit_touch20_low_rank": rank_frame(limit_touch20, ascending=False),
        "limit_close5_low_rank": rank_frame(limit_close5, ascending=False),
        "gap_risk_low_rank": rank_frame(gap_risk, ascending=False),
    }
    ranks["quiet_accumulation_rank"] = 0.24 * ranks["ret20_smooth_rank"] + 0.18 * ranks["amount_persist_rank"] + 0.16 * ranks["vwap_support5_rank"] + 0.16 * ranks["vol20_low_rank"] + 0.14 * ranks["upper_wick_low_rank"] + 0.12 * ranks["limit_close5_low_rank"]
    ranks["latent_right_tail_rank"] = 0.22 * ranks["ret20_rank"] + 0.18 * ranks["ret5_moderate_rank"] + 0.16 * ranks["amt_ratio20_rank"] + 0.14 * ranks["close_vs_vwap_rank"] + 0.12 * ranks["near_high20_rank"] + 0.10 * ranks["limit_touch20_low_rank"] + 0.08 * ranks["gap_risk_low_rank"]
    ranks["noncrowded_trend_rank"] = 0.24 * ranks["ret20_over_60_rank"] + 0.18 * ranks["range20_low_rank"] + 0.16 * ranks["upper_wick_low_rank"] + 0.16 * ranks["limit_close5_low_rank"] + 0.14 * ranks["drawdown20_rank"] + 0.12 * ranks["amount_rank"]

    history = (open_.notna() & (open_ > 0)).cumsum()
    sampled = [day for day in dates if args.start <= day <= args.end]
    sampled = set(sampled[:: args.sample_step])
    arr = market["arrays"]
    symbols_arr = np.asarray(symbols, dtype=object)
    frames: list[pd.DataFrame] = []
    for day in dates:
        if day not in sampled:
            continue
        idx = market["date_index"][day]
        entry_idx = idx + 1
        exit_idx = entry_idx + args.horizon
        if idx < 130 or exit_idx >= len(dates):
            continue
        entry_open = arr["open"][entry_idx]
        exit_open = arr["open"][exit_idx]
        raw = exit_open / entry_open - 1.0
        finite = np.isfinite(raw)
        if finite.sum() < 500:
            continue
        entry_gap = entry_open / arr["close"][idx] - 1.0
        exit_gap = exit_open / arr["close"][exit_idx - 1] - 1.0
        entry_ok = finite & np.isfinite(arr["volume"][entry_idx]) & (arr["volume"][entry_idx] > 0) & (entry_gap < 0.095) & (history.iloc[idx].to_numpy(dtype=float) >= 130) & (arr["is_st"][idx] < 0.5)
        exit_ok = finite & np.isfinite(arr["volume"][exit_idx]) & (arr["volume"][exit_idx] > 0) & (exit_gap > -0.095)
        universe_ok = finite & np.isfinite(arr["open"][idx]) & (arr["open"][idx] > 0) & (arr["is_st"][idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        if universe_ok.sum() < 500:
            continue
        bench = float(np.nanmean(raw[finite]))
        exec_label = np.where(entry_ok & exit_ok, raw - 0.00154, -0.08)
        mkt_values = ret20.iloc[idx].to_numpy(dtype=float)
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
            "mkt_amount_disp20": np.full(int(universe_ok.sum()), float(np.nanstd(np.log1p(amount.iloc[idx].to_numpy(dtype=float))))),
            "mkt_near_high20": np.full(int(universe_ok.sum()), float(np.nanmean(near_high20.iloc[idx].to_numpy(dtype=float) > 0.95))),
        }
        for name, value in ranks.items():
            block[name] = value.iloc[idx].to_numpy(dtype=float)[universe_ok]
        frame = pd.DataFrame(block).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        frame["top_exec"] = (frame.groupby("session")["exec_label5_open"].rank(pct=True, method="first") >= 0.95).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, int]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds = []
    importances = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        reg = LGBMRegressor(n_estimators=320, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year), n_jobs=6, verbosity=-1)
        cls = LGBMClassifier(n_estimators=260, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year) + 19, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_latent_static"] = out["latent_right_tail_rank"]
        out["score_quiet_accumulation"] = out["quiet_accumulation_rank"]
        out["score_noncrowded_trend"] = out["noncrowded_trend_rank"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [
        ("reg", "score_reg"),
        ("cls", "score_cls"),
        ("blend", "score_blend"),
        ("latent_static", "score_latent_static"),
        ("quiet_accumulation", "score_quiet_accumulation"),
        ("noncrowded_trend", "score_noncrowded_trend"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({"session": session, "year": int(session[:4]), "variant": variant, "topk": int(topk), "mean_label5_open": float(selected["label5_open"].mean()), "mean_exec_label5_open": float(selected["exec_label5_open"].mean()), "entry_ok": float(selected["entry_ok"].mean()), "exit_ok": float(selected["exit_ok"].mean()), "count": int(len(selected))})
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {"mean_label5_open": float(group["mean_label5_open"].mean()), "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()), "entry_ok": float(group["entry_ok"].mean()), "exit_ok": float(group["exit_ok"].mean()), "by_year": {str(int(y)): {"mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()), "entry_ok": float(yg["entry_ok"].mean())} for y, yg in group.groupby("year")}}
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out = {}
    for variant, score_col in variants():
        for topk in (10, 20):
            selections = {}
            for session, group in scored.groupby("session", sort=True):
                selections[session] = list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"])
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
