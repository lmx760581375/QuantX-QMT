from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


FEATURES = [
    "ret1_rank", "ret3_rank", "ret5_rank", "ret10_rank", "ret20_rank", "ret60_rank",
    "ret5_accel_rank", "ret20_over_60_rank", "vol20_low_rank", "range20_low_rank",
    "amount_rank", "amt_ratio20_rank", "close_vs_vwap_rank", "close_strength_rank",
    "lower_wick_rank", "upper_wick_low_rank", "near_high20_rank", "drawdown20_rank",
    "break20_rank", "limit_touch20_rank", "limit_close5_rank", "quiet_trend_rank",
    "tradable_momentum_rank", "pullback_strength_rank", "mkt_ret20_median", "mkt_breadth20",
    "mkt_disp20", "mkt_amount_disp20", "mkt_near_high20",
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
    market = load_market(args)
    panel = build_panel(market, args)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args)
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
            "causality": "Features use completed T daily data only. Signal is formed after T close. Execution is pre-scheduled at T+1 close, with limit/suspension rejection checked on execution day. Labels and account returns use T+1 close to T+6 close.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(result, "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest())
    return 0


def load_market(args: argparse.Namespace) -> dict[str, Any]:
    provider = Path(args.provider)
    universe = _build_universe(provider, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(provider)
    calendar = reader.calendar(args.load_start, args.end)
    quotes = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$vwap"], args.load_start, args.end)
    frames = {name: quotes[f"${name}"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float) for name in ("open", "high", "low", "close", "volume", "vwap")}
    raw = load_raw_fields(Path(args.raw_dir), symbols, calendar)
    amount = raw["amount"].where(raw["amount"] > 0, frames["vwap"] * frames["volume"])
    frames["amount"] = amount.astype(float)
    frames["is_st"] = raw["is_st"].fillna(0.0).astype(float)
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
    return {"symbols": symbols, "dates": dates, "date_index": {day: idx for idx, day in enumerate(dates)}, "frames": frames, "arrays": {k: v.to_numpy(dtype=float, copy=False) for k, v in frames.items()}}


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


def rank_frame(frame: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, ascending=ascending).fillna(0.5)


def build_panel(market: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    f = market["frames"]
    symbols = market["symbols"]
    dates = market["dates"]
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
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
    daily_ret20 = close.pct_change(fill_method=None).rolling(20, min_periods=10)
    vol20 = daily_ret20.std()
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    prev_high20 = high.shift(1).rolling(20, min_periods=10).max()
    amount20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio20 = amount / amount20
    close_vs_vwap = close / vwap - 1.0
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    lower_wick = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    near_high20 = close / high20
    drawdown20 = close / high20 - 1.0
    break20 = close / prev_high20 - 1.0
    limit_touch = ((high / close.shift(1) - 1.0) >= 0.095).rolling(20, min_periods=5).sum()
    limit_close = ((close / close.shift(1) - 1.0) >= 0.095).rolling(5, min_periods=2).sum()
    ranks = {
        "ret1_rank": rank_frame(ret1),
        "ret3_rank": rank_frame(ret3),
        "ret5_rank": rank_frame(ret5),
        "ret10_rank": rank_frame(ret10),
        "ret20_rank": rank_frame(ret20),
        "ret60_rank": rank_frame(ret60),
        "ret5_accel_rank": rank_frame(ret5 - ret20 / 4.0),
        "ret20_over_60_rank": rank_frame(ret20 - ret60 / 3.0),
        "vol20_low_rank": rank_frame(vol20, ascending=False),
        "range20_low_rank": rank_frame(range20, ascending=False),
        "amount_rank": rank_frame(np.log1p(amount)),
        "amt_ratio20_rank": rank_frame(amt_ratio20),
        "close_vs_vwap_rank": rank_frame(close_vs_vwap),
        "close_strength_rank": rank_frame(close_strength),
        "lower_wick_rank": rank_frame(lower_wick),
        "upper_wick_low_rank": rank_frame(upper_wick, ascending=False),
        "near_high20_rank": rank_frame(near_high20),
        "drawdown20_rank": rank_frame(drawdown20),
        "break20_rank": rank_frame(break20),
        "limit_touch20_rank": rank_frame(limit_touch),
        "limit_close5_rank": rank_frame(limit_close),
    }
    ranks["quiet_trend_rank"] = 0.30 * ranks["ret20_rank"] + 0.25 * ranks["vol20_low_rank"] + 0.20 * ranks["near_high20_rank"] + 0.15 * ranks["amount_rank"] + 0.10 * ranks["upper_wick_low_rank"]
    ranks["tradable_momentum_rank"] = 0.25 * ranks["ret5_rank"] + 0.20 * ranks["ret20_rank"] + 0.20 * ranks["close_vs_vwap_rank"] + 0.20 * ranks["amt_ratio20_rank"] + 0.15 * ranks["upper_wick_low_rank"]
    ranks["pullback_strength_rank"] = 0.30 * ranks["ret20_rank"] + 0.25 * rank_frame(-ret3) + 0.20 * ranks["near_high20_rank"] + 0.15 * ranks["lower_wick_rank"] + 0.10 * ranks["amount_rank"]
    history = (open_.notna() & (open_ > 0)).cumsum()
    sampled = [day for day in dates if args.start <= day <= args.end]
    sampled = set(sampled[:: args.sample_step])
    open_arr = market["arrays"]["open"]
    close_arr = market["arrays"]["close"]
    volume_arr = market["arrays"]["volume"]
    st_arr = market["arrays"]["is_st"]
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
        entry_close = close_arr[entry_idx]
        exit_close = close_arr[exit_idx]
        raw = exit_close / entry_close - 1.0
        finite = np.isfinite(raw)
        if finite.sum() < 500:
            continue
        entry_ret = entry_close / close_arr[idx] - 1.0
        exit_ret = exit_close / close_arr[exit_idx - 1] - 1.0
        entry_ok = finite & np.isfinite(volume_arr[entry_idx]) & (volume_arr[entry_idx] > 0) & (entry_ret < 0.095) & (history.iloc[idx].to_numpy(dtype=float) >= 130) & (st_arr[idx] < 0.5)
        exit_ok = finite & np.isfinite(volume_arr[exit_idx]) & (volume_arr[exit_idx] > 0) & (exit_ret > -0.095)
        tradable = finite & np.isfinite(open_arr[idx]) & (open_arr[idx] > 0) & (st_arr[idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        if tradable.sum() < 500:
            continue
        bench = float(np.nanmean(raw[finite]))
        exec_label = raw - 0.00154
        exec_label = np.where(entry_ok & exit_ok, exec_label, -0.08)
        mkt_ret20 = float(np.nanmedian(ret20.iloc[idx].to_numpy(dtype=float)))
        mkt_breadth20 = float(np.nanmean(ret20.iloc[idx].to_numpy(dtype=float) > 0))
        mkt_disp20 = float(np.nanstd(ret20.iloc[idx].to_numpy(dtype=float)))
        mkt_amount_disp20 = float(np.nanstd(np.log1p(amount.iloc[idx].to_numpy(dtype=float))))
        mkt_near_high20 = float(np.nanmean(near_high20.iloc[idx].to_numpy(dtype=float) > 0.95))
        block: dict[str, Any] = {
            "session": np.full(int(tradable.sum()), day, dtype=object),
            "year": np.full(int(tradable.sum()), int(day[:4]), dtype=int),
            "instrument": symbols_arr[tradable],
            "raw5_close": raw[tradable],
            "label5_close": raw[tradable] - bench,
            "exec_label5_close": exec_label[tradable] - bench,
            "entry_ok": entry_ok[tradable].astype(float),
            "exit_ok": exit_ok[tradable].astype(float),
            "mkt_ret20_median": np.full(int(tradable.sum()), mkt_ret20),
            "mkt_breadth20": np.full(int(tradable.sum()), mkt_breadth20),
            "mkt_disp20": np.full(int(tradable.sum()), mkt_disp20),
            "mkt_amount_disp20": np.full(int(tradable.sum()), mkt_amount_disp20),
            "mkt_near_high20": np.full(int(tradable.sum()), mkt_near_high20),
        }
        for name, value in ranks.items():
            block[name] = value.iloc[idx].to_numpy(dtype=float)[tradable]
        frame = pd.DataFrame(block).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        frame["top_exec"] = (frame.groupby("session")["exec_label5_close"].rank(pct=True, method="first") >= 0.95).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, int]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds = []
    imps = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        reg = LGBMRegressor(n_estimators=300, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year), n_jobs=6, verbosity=-1)
        cls = LGBMClassifier(n_estimators=240, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year) + 11, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_close"])
        cls.fit(train[FEATURES], train["top_exec"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_static_tradable_mom"] = out["tradable_momentum_rank"]
        out["score_static_quiet_trend"] = out["quiet_trend_rank"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        imps.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, imps


def variants() -> list[tuple[str, str]]:
    return [("reg", "score_reg"), ("cls", "score_cls"), ("blend", "score_blend"), ("static_tradable_mom", "score_static_tradable_mom"), ("static_quiet_trend", "score_static_quiet_trend")]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({"session": session, "year": int(session[:4]), "variant": variant, "topk": int(topk), "mean_label5_close": float(selected["label5_close"].mean()), "mean_exec_label5_close": float(selected["exec_label5_close"].mean()), "entry_ok": float(selected["entry_ok"].mean()), "exit_ok": float(selected["exit_ok"].mean()), "count": int(len(selected))})
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {"mean_label5_close": float(group["mean_label5_close"].mean()), "mean_exec_label5_close": float(group["mean_exec_label5_close"].mean()), "entry_ok": float(group["entry_ok"].mean()), "exit_ok": float(group["exit_ok"].mean()), "by_year": {str(int(y)): {"mean_exec_label5_close": float(yg["mean_exec_label5_close"].mean()), "entry_ok": float(yg["entry_ok"].mean())} for y, yg in group.groupby("year")}}
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for variant, score_col in variants():
        for topk in (10, 20):
            selections = {}
            for session, group in scored.groupby("session", sort=True):
                ranked = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)
                selections[session] = list(ranked["instrument"])
            out[f"{variant}::top{topk}"] = replay_selection(selections, market, args.horizon)
    return out


def replay_selection(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    symbols = market["symbols"]
    symbol_index = {s: i for i, s in enumerate(symbols)}
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
            entry_close = float(arr["close"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry_close) or entry_close <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry_close / preclose - 1.0 >= 0.095:
                rejects["entry_limit_up_or_missing"] += 1
                continue
            sell_idx = exit_idx
            while sell_idx < min(len(dates), exit_idx + 6):
                sell_close = float(arr["close"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if np.isfinite(sell_close) and sell_close > 0 and np.isfinite(sell_preclose) and sell_preclose > 0 and np.isfinite(sell_vol) and sell_vol > 0 and sell_close / sell_preclose - 1.0 > -0.095:
                    break
                sell_idx += 1
            if sell_idx >= min(len(dates), exit_idx + 6):
                rejects["exit_limit_down_or_missing"] += 1
                continue
            sell_close = float(arr["close"][sell_idx, col])
            rets.append(sell_close / entry_close - 1.0 - 0.00154)
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
    annual = {}
    for year, group in pd.DataFrame(rows).groupby("year"):
        annual[str(int(year))] = float(np.prod(1.0 + group["period_return"].to_numpy(dtype=float)) - 1.0)
    navs = np.asarray([r["nav"] for r in curve], dtype=float)
    peak = np.maximum.accumulate(navs)
    return {"final_multiple": float(nav), "total_return": float(nav - 1.0), "annual_returns": annual, "all_years_positive": bool(all(v > 0 for v in annual.values())), "max_drawdown_period": float(np.min(navs / peak - 1.0)), "avg_selected_count": float(np.mean([r["selected_count"] for r in rows])), "avg_hold_days": float(np.mean([r["avg_hold_days"] for r in rows if r["avg_hold_days"] > 0])) if any(r["avg_hold_days"] > 0 for r in rows) else 0.0, "periods": int(len(rows)), "rejects": rejects}


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
    account_rows = []
    for name, data in result["account_results"].items():
        account_rows.append((name, data.get("final_multiple", 1.0), data.get("annual_returns", {}).get("2026", np.nan), data.get("avg_selected_count", 0.0)))
    account_rows.sort(key=lambda item: item[1], reverse=True)
    for name, multiple, ret2026, avg_count in account_rows[:12]:
        print(f"{name}: final={multiple:.2f}x 2026={ret2026:+.4f} avg_count={avg_count:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
