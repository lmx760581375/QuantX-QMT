from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


TOPKS = (10, 15, 20, 30)
FEATURES = [
    "ret5_rank", "ret20_rank", "ret60_rank", "amount_rank", "amt_ratio20_rank",
    "corr60_rank", "beta60_rank", "resid20_rank", "laggard20_rank", "resid5_rank",
    "leader_continue", "follower_catchup", "theme_beta_amount", "leader_pullback",
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
    parser.add_argument("--leader-count", type=int, default=50)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel = build_panel(args)
    detail, folds, importances = walk_forward(panel, args.min_train_rows)
    result = summarize(detail)
    result["folds"] = folds
    result["feature_importance"] = summarize_importance(importances)
    result["config"] = {
        "provider": args.provider,
        "raw_dir": args.raw_dir,
        "start": args.start,
        "end": args.end,
        "load_start": args.load_start,
        "horizon": args.horizon,
        "sample_step": args.sample_step,
        "leader_count": args.leader_count,
        "corr_window": args.corr_window,
        "panel_rows": int(len(panel)),
        "sessions": int(panel["session"].nunique()) if not panel.empty else 0,
        "features": FEATURES,
        "causality": "Leader basket and all features use QMT daily data through completed T only. Labels use T+1 open to T+6 open. Annual folds train only on years strictly before evaluated year.",
        "scope": "QMT retrievable daily market data only; no static industry/concept membership, news, announcements, LHB, ETF flow, northbound, financing, or event/message data.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_panel(args: argparse.Namespace) -> pd.DataFrame:
    provider = Path(args.provider)
    universe = _build_universe(provider, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(provider)
    calendar = reader.calendar(args.load_start, args.end)
    quotes = reader.features(symbols, ["$open", "$close", "$volume", "$vwap"], args.load_start, args.end)
    opens = quotes["$open"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    closes = quotes["$close"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    volumes = quotes["$volume"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    raw = load_raw_fields(Path(args.raw_dir), symbols, calendar)
    amount = raw["amount"].where(raw["amount"] > 0, quotes["$vwap"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float) * volumes)
    is_st = raw["is_st"].fillna(0.0)
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
    date_index = {day: idx for idx, day in enumerate(dates)}
    valid_sessions = [day for day in dates if args.start <= day <= args.end]
    sampled = set(valid_sessions[:: args.sample_step])
    open_values = opens.to_numpy(dtype=float, copy=False)
    close = closes.astype(float)
    ret1 = close.pct_change(fill_method=None)
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amt_ratio20 = amount / amount.rolling(20, min_periods=10).mean()
    history = pd.DataFrame(np.cumsum(np.isfinite(open_values) & (open_values > 0), axis=0), index=calendar, columns=symbols)
    frames: list[pd.DataFrame] = []
    symbols_arr = np.asarray(symbols, dtype=object)
    for session in valid_sessions:
        if session not in sampled:
            continue
        idx = date_index[session]
        entry_idx = idx + 1
        exit_idx = entry_idx + args.horizon
        start_idx = idx - args.corr_window + 1
        if start_idx < 1 or exit_idx >= len(open_values):
            continue
        raw5 = open_values[exit_idx] / open_values[entry_idx] - 1.0
        finite = np.isfinite(raw5)
        if finite.sum() < 500:
            continue
        leader_score = ret20.iloc[idx].rank(pct=True) + 0.35 * np.log1p(amount.iloc[idx]).rank(pct=True) + 0.25 * ret5.iloc[idx].rank(pct=True)
        leader_cols = leader_score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(args.leader_count).index
        if len(leader_cols) < 10:
            continue
        window = ret1.iloc[start_idx : idx + 1]
        leader_series = window[leader_cols].mean(axis=1).to_numpy(dtype=float)
        stock_window = window.to_numpy(dtype=float, copy=False)
        corr, beta = corr_beta(stock_window, leader_series)
        leader_ret20 = float(ret20.iloc[idx][leader_cols].mean())
        resid20 = ret20.iloc[idx].to_numpy(dtype=float) - beta * leader_ret20
        resid5 = ret5.iloc[idx].to_numpy(dtype=float) - beta * float(ret5.iloc[idx][leader_cols].mean())
        data = {
            "ret5_rank": ret5.iloc[idx].rank(pct=True).to_numpy(dtype=float),
            "ret20_rank": ret20.iloc[idx].rank(pct=True).to_numpy(dtype=float),
            "ret60_rank": ret60.iloc[idx].rank(pct=True).to_numpy(dtype=float),
            "amount_rank": np.log1p(amount.iloc[idx]).rank(pct=True).to_numpy(dtype=float),
            "amt_ratio20_rank": amt_ratio20.iloc[idx].rank(pct=True).to_numpy(dtype=float),
            "corr60_rank": pd.Series(corr, index=symbols).rank(pct=True).to_numpy(dtype=float),
            "beta60_rank": pd.Series(beta, index=symbols).rank(pct=True).to_numpy(dtype=float),
            "resid20_rank": pd.Series(resid20, index=symbols).rank(pct=True).to_numpy(dtype=float),
            "laggard20_rank": pd.Series(-resid20, index=symbols).rank(pct=True).to_numpy(dtype=float),
            "resid5_rank": pd.Series(resid5, index=symbols).rank(pct=True).to_numpy(dtype=float),
        }
        data["leader_continue"] = 0.35 * data["ret20_rank"] + 0.25 * data["ret60_rank"] + 0.20 * data["amount_rank"] + 0.20 * data["corr60_rank"]
        data["follower_catchup"] = 0.35 * data["corr60_rank"] + 0.25 * data["laggard20_rank"] + 0.20 * data["amt_ratio20_rank"] + 0.20 * data["ret5_rank"]
        data["theme_beta_amount"] = 0.40 * data["beta60_rank"] + 0.30 * data["amount_rank"] + 0.30 * data["ret20_rank"]
        data["leader_pullback"] = 0.35 * data["corr60_rank"] + 0.30 * data["ret60_rank"] + 0.20 * data["laggard20_rank"] + 0.15 * data["amt_ratio20_rank"]
        hist_ok = history.iloc[idx].to_numpy(dtype=float) >= 130
        st_ok = (1.0 - is_st.iloc[idx]).to_numpy(dtype=float) > 0.5
        tradable = finite & hist_ok & st_ok
        if tradable.sum() < 500:
            continue
        bench = float(np.nanmean(raw5[finite]))
        block: dict[str, Any] = {
            "session": np.full(int(tradable.sum()), session, dtype=object),
            "year": np.full(int(tradable.sum()), int(session[:4]), dtype=int),
            "instrument": symbols_arr[tradable],
            "label5": raw5[tradable] - bench,
            "raw5": raw5[tradable],
        }
        for name in FEATURES:
            block[name] = np.asarray(data[name], dtype=float)[tradable]
        frame = pd.DataFrame(block).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        labels = frame["label5"]
        frame["label_quintile"] = pd.qcut(labels.rank(method="first"), 5, labels=False).astype(int).to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_raw_fields(raw_dir: Path, symbols: tuple[str, ...], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    amount = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    is_st = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"date", "amount", "is_st"})
        if df.empty or "date" not in df:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(calendar)
        if "amount" in df:
            amount[symbol] = pd.to_numeric(df["amount"], errors="coerce")
        if "is_st" in df:
            is_st[symbol] = pd.to_numeric(df["is_st"], errors="coerce")
    return {"amount": amount, "is_st": is_st.fillna(0.0)}


def corr_beta(stock_window: np.ndarray, leader: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = stock_window.copy()
    y = leader.astype(float)
    y = y - np.nanmean(y)
    x = x - np.nanmean(x, axis=0)
    cov = np.nanmean(x * y[:, None], axis=0)
    x_std = np.nanstd(x, axis=0)
    y_var = float(np.nanmean(y * y))
    y_std = float(np.sqrt(y_var))
    corr = cov / (x_std * y_std + 1e-12)
    beta = cov / (y_var + 1e-12)
    return corr, beta


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, int]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, int]] = []
    imps: list[dict[str, float]] = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        scored, imp = fit_predict(train, test, int(year))
        rows.append(evaluate(scored))
        imps.append(imp)
        fold = {"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())}
        folds.append(fold)
        print(json.dumps(fold, ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, imps


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    reg = LGBMRegressor(n_estimators=260, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=6, verbosity=-1)
    ranker = LGBMRanker(objective="lambdarank", n_estimators=220, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=400, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=5.0, random_state=seed + 17, n_jobs=6, verbosity=-1)
    reg.fit(train[FEATURES], train["label5"])
    sorted_train = train.sort_values(["session", "instrument"])
    groups = sorted_train.groupby("session", sort=False).size().to_numpy(dtype=int)
    ranker.fit(sorted_train[FEATURES], sorted_train["label_quintile"].astype(int), group=groups)
    out = test.copy()
    out["score_reg"] = reg.predict(out[FEATURES])
    out["score_rank"] = ranker.predict(out[FEATURES])
    out["score_blend"] = 0.5 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.5 * out.groupby("session")["score_rank"].rank(pct=True, method="first")
    imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
    imp.update({f"rank::{name}": float(value) for name, value in zip(FEATURES, ranker.feature_importances_)})
    return out, imp


def evaluate(scored: pd.DataFrame) -> pd.DataFrame:
    variants = [("reg", "score_reg"), ("rank", "score_rank"), ("blend", "score_blend"), ("leader_continue", "leader_continue"), ("follower_catchup", "follower_catchup"), ("theme_beta_amount", "theme_beta_amount"), ("leader_pullback", "leader_pullback")]
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants:
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({"session": session, "year": int(str(session)[:4]), "variant": variant, "topk": int(topk), "mean_label5": float(selected["label5"].mean()), "mean_raw5": float(selected["raw5"].mean()), "positive_label5": bool(selected["label5"].mean() > 0), "count": int(len(selected))})
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"daily_row_count": int(len(detail)), "results": {}}
    if detail.empty:
        return result
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        key = f"{variant}::top{int(topk)}"
        result["results"][key] = {"daily_count": int(len(group)), "mean_daily_label5": float(group["mean_label5"].mean()), "mean_daily_raw5": float(group["mean_raw5"].mean()), "positive_label5_ratio": float(group["positive_label5"].mean()), "avg_selected_count": float(group["count"].mean()), "by_year": {str(int(y)): {"mean_daily_label5": float(yg["mean_label5"].mean()), "mean_daily_raw5": float(yg["mean_raw5"].mean()), "positive_label5_ratio": float(yg["positive_label5"].mean())} for y, yg in group.groupby("year")}}
    return result


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, float | str]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:50]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    rows = []
    for key, value in result.get("results", {}).items():
        if key.endswith("::top20"):
            rows.append((float(value["mean_daily_label5"]), key, {y: round(d["mean_daily_label5"], 6) for y, d in value["by_year"].items()}))
    rows.sort(key=lambda item: item[0], reverse=True)
    print(json.dumps({"checksum": checksum, "daily_row_count": result.get("daily_row_count"), "top20_results": rows, "top_features": result.get("feature_importance", [])[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
