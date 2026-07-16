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
    "amount_rank", "amt_ratio20_rank", "vol20_low_rank", "range20_low_rank",
    "near_high20_rank", "close_strength_rank", "upper_wick_low_rank", "price_rank",
    "mom_bucket_strength", "mom_bucket_breadth", "mom_bucket_amt", "mom_rel5_rank", "mom_lag5_rank",
    "liq_bucket_strength", "liq_bucket_breadth", "liq_bucket_amt", "liq_rel5_rank", "liq_lag5_rank",
    "vol_bucket_strength", "vol_bucket_breadth", "vol_bucket_amt", "vol_rel5_rank", "vol_lag5_rank",
    "price_bucket_strength", "price_bucket_breadth", "price_bucket_amt", "price_rel5_rank", "price_lag5_rank",
    "multi_style_strength", "style_lag_catchup", "style_leader_confirm", "style_rotation_quality",
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
            "causality": "For each after-close signal T, style buckets and all flow/rotation features use completed daily bars through T only. Labels and replay use pre-scheduled T+1 open entry and T+6 open target exit. Execution-day data is used only for realized fills/rejections.",
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


def bucket_ids(rank_values: np.ndarray, valid: np.ndarray, bucket_count: int = 5) -> np.ndarray:
    ids = np.full(len(rank_values), -1, dtype=int)
    raw = np.floor(np.clip(rank_values, 0.0, 0.999999) * bucket_count).astype(int)
    ids[valid & np.isfinite(rank_values)] = raw[valid & np.isfinite(rank_values)]
    return ids


def assign_bucket_features(ids: np.ndarray, valid: np.ndarray, ret5: np.ndarray, ret20: np.ndarray, amt_ratio: np.ndarray) -> dict[str, np.ndarray]:
    strength = np.full(len(ids), 0.5, dtype=float)
    breadth = np.full(len(ids), 0.5, dtype=float)
    amt = np.full(len(ids), 0.5, dtype=float)
    rel5 = np.full(len(ids), 0.0, dtype=float)
    lag5 = np.full(len(ids), 0.0, dtype=float)
    bucket_ret5 = np.zeros(5, dtype=float)
    bucket_strength = np.zeros(5, dtype=float)
    bucket_breadth = np.zeros(5, dtype=float)
    bucket_amt = np.zeros(5, dtype=float)
    for bucket in range(5):
        mask = valid & (ids == bucket)
        if int(mask.sum()) < 50:
            continue
        r5 = float(np.nanmedian(ret5[mask]))
        r20 = float(np.nanmedian(ret20[mask]))
        br = float(np.nanmean(ret20[mask] > 0))
        ar = float(np.nanmedian(amt_ratio[mask]))
        bucket_ret5[bucket] = r5
        bucket_strength[bucket] = r5 + 0.45 * r20 + 0.015 * (br - 0.5) + 0.006 * np.log1p(max(ar, 0.0))
        bucket_breadth[bucket] = br
        bucket_amt[bucket] = ar
    strength_rank = rank_series(bucket_strength)
    breadth_rank = rank_series(bucket_breadth)
    amt_rank = rank_series(bucket_amt)
    for bucket in range(5):
        mask = valid & (ids == bucket)
        strength[mask] = strength_rank[bucket]
        breadth[mask] = breadth_rank[bucket]
        amt[mask] = amt_rank[bucket]
        rel5[mask] = ret5[mask] - bucket_ret5[bucket]
        lag5[mask] = bucket_ret5[bucket] - ret5[mask]
    return {"strength": strength, "breadth": breadth, "amt": amt, "rel5_rank": rank_series(rel5), "lag5_rank": rank_series(lag5)}


def build_panel(market: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    f = market["frames"]
    dates = market["dates"]
    symbols = market["symbols"]
    arr = market["arrays"]
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
    amount = f["amount"]
    ret1 = close.pct_change(fill_method=None)
    ret3 = close / close.shift(3) - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret10 = close / close.shift(10) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amount20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio20 = amount / amount20
    vol20 = ret1.rolling(20, min_periods=10).std()
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    near_high20 = close / high20
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
        if idx < 130 or exit_idx >= len(dates):
            continue
        raw = arr["open"][exit_idx] / arr["open"][entry_idx] - 1.0
        finite = np.isfinite(raw)
        if finite.sum() < 500:
            continue
        r1 = ret1.iloc[idx].to_numpy(dtype=float)
        r3 = ret3.iloc[idx].to_numpy(dtype=float)
        r5 = ret5.iloc[idx].to_numpy(dtype=float)
        r10 = ret10.iloc[idx].to_numpy(dtype=float)
        r20 = ret20.iloc[idx].to_numpy(dtype=float)
        r60 = ret60.iloc[idx].to_numpy(dtype=float)
        amt = amount.iloc[idx].to_numpy(dtype=float)
        amt_ratio = amt_ratio20.iloc[idx].to_numpy(dtype=float)
        vol = vol20.iloc[idx].to_numpy(dtype=float)
        rng = range20.iloc[idx].to_numpy(dtype=float)
        near_high = near_high20.iloc[idx].to_numpy(dtype=float)
        c_strength = close_strength.iloc[idx].to_numpy(dtype=float)
        u_wick = upper_wick.iloc[idx].to_numpy(dtype=float)
        price = close.iloc[idx].to_numpy(dtype=float)
        universe_ok = finite & np.isfinite(arr["open"][idx]) & (arr["open"][idx] > 0) & (arr["is_st"][idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        if universe_ok.sum() < 500:
            continue
        ret1_rank = rank_series(r1)
        ret3_rank = rank_series(r3)
        ret5_rank = rank_series(r5)
        ret10_rank = rank_series(r10)
        ret20_rank = rank_series(r20)
        ret60_rank = rank_series(r60)
        amount_rank = rank_series(np.log1p(amt))
        amt_ratio_rank = rank_series(amt_ratio)
        vol_low_rank = rank_series(vol, ascending=False)
        range_low_rank = rank_series(rng, ascending=False)
        near_high_rank = rank_series(near_high)
        close_strength_rank = rank_series(c_strength)
        upper_wick_low_rank = rank_series(u_wick, ascending=False)
        price_rank = rank_series(price)
        mom = assign_bucket_features(bucket_ids(ret20_rank, universe_ok), universe_ok, r5, r20, amt_ratio)
        liq = assign_bucket_features(bucket_ids(amount_rank, universe_ok), universe_ok, r5, r20, amt_ratio)
        volf = assign_bucket_features(bucket_ids(rank_series(vol), universe_ok), universe_ok, r5, r20, amt_ratio)
        prc = assign_bucket_features(bucket_ids(price_rank, universe_ok), universe_ok, r5, r20, amt_ratio)
        data = {
            "ret1_rank": ret1_rank,
            "ret3_rank": ret3_rank,
            "ret5_rank": ret5_rank,
            "ret10_rank": ret10_rank,
            "ret20_rank": ret20_rank,
            "ret60_rank": ret60_rank,
            "amount_rank": amount_rank,
            "amt_ratio20_rank": amt_ratio_rank,
            "vol20_low_rank": vol_low_rank,
            "range20_low_rank": range_low_rank,
            "near_high20_rank": near_high_rank,
            "close_strength_rank": close_strength_rank,
            "upper_wick_low_rank": upper_wick_low_rank,
            "price_rank": price_rank,
            "mom_bucket_strength": mom["strength"],
            "mom_bucket_breadth": mom["breadth"],
            "mom_bucket_amt": mom["amt"],
            "mom_rel5_rank": mom["rel5_rank"],
            "mom_lag5_rank": mom["lag5_rank"],
            "liq_bucket_strength": liq["strength"],
            "liq_bucket_breadth": liq["breadth"],
            "liq_bucket_amt": liq["amt"],
            "liq_rel5_rank": liq["rel5_rank"],
            "liq_lag5_rank": liq["lag5_rank"],
            "vol_bucket_strength": volf["strength"],
            "vol_bucket_breadth": volf["breadth"],
            "vol_bucket_amt": volf["amt"],
            "vol_rel5_rank": volf["rel5_rank"],
            "vol_lag5_rank": volf["lag5_rank"],
            "price_bucket_strength": prc["strength"],
            "price_bucket_breadth": prc["breadth"],
            "price_bucket_amt": prc["amt"],
            "price_rel5_rank": prc["rel5_rank"],
            "price_lag5_rank": prc["lag5_rank"],
        }
        data["multi_style_strength"] = 0.30 * data["mom_bucket_strength"] + 0.25 * data["liq_bucket_strength"] + 0.20 * data["vol_bucket_strength"] + 0.15 * data["price_bucket_strength"] + 0.10 * data["amt_ratio20_rank"]
        data["style_lag_catchup"] = 0.30 * data["multi_style_strength"] + 0.20 * data["mom_lag5_rank"] + 0.18 * data["liq_lag5_rank"] + 0.14 * data["near_high20_rank"] + 0.10 * data["close_strength_rank"] + 0.08 * data["upper_wick_low_rank"]
        data["style_leader_confirm"] = 0.28 * data["multi_style_strength"] + 0.22 * data["mom_rel5_rank"] + 0.18 * data["ret20_rank"] + 0.14 * data["amount_rank"] + 0.10 * data["near_high20_rank"] + 0.08 * data["upper_wick_low_rank"]
        data["style_rotation_quality"] = 0.26 * data["multi_style_strength"] + 0.20 * data["style_lag_catchup"] + 0.18 * data["style_leader_confirm"] + 0.16 * data["vol20_low_rank"] + 0.12 * data["range20_low_rank"] + 0.08 * data["price_bucket_breadth"]
        entry_gap = arr["open"][entry_idx] / arr["close"][idx] - 1.0
        exit_gap = arr["open"][exit_idx] / arr["close"][exit_idx - 1] - 1.0
        entry_ok = finite & np.isfinite(arr["volume"][entry_idx]) & (arr["volume"][entry_idx] > 0) & (entry_gap < 0.095) & universe_ok
        exit_ok = finite & np.isfinite(arr["volume"][exit_idx]) & (arr["volume"][exit_idx] > 0) & (exit_gap > -0.095)
        bench = float(np.nanmean(raw[finite]))
        exec_label = np.where(entry_ok & exit_ok, raw - 0.00154, -0.08)
        n = int(universe_ok.sum())
        block: dict[str, Any] = {
            "session": np.full(n, day, dtype=object),
            "year": np.full(n, int(day[:4]), dtype=int),
            "instrument": symbols_arr[universe_ok],
            "raw5_open": raw[universe_ok],
            "label5_open": raw[universe_ok] - bench,
            "exec_label5_open": exec_label[universe_ok] - bench,
            "entry_ok": entry_ok[universe_ok].astype(float),
            "exit_ok": exit_ok[universe_ok].astype(float),
            "mkt_ret20_median": np.full(n, float(np.nanmedian(r20))),
            "mkt_breadth20": np.full(n, float(np.nanmean(r20 > 0))),
            "mkt_disp20": np.full(n, float(np.nanstd(r20))),
            "mkt_amount_disp20": np.full(n, float(np.nanstd(np.log1p(amt)))),
            "mkt_near_high20": np.full(n, float(np.nanmean(near_high > 0.95))),
        }
        for name in FEATURES:
            if name in block:
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
        cls = LGBMClassifier(n_estimators=240, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year) + 53, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_style_lag_catchup"] = out["style_lag_catchup"]
        out["score_style_leader_confirm"] = out["style_leader_confirm"]
        out["score_style_rotation_quality"] = out["style_rotation_quality"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [("reg", "score_reg"), ("cls", "score_cls"), ("blend", "score_blend"), ("style_lag_catchup", "score_style_lag_catchup"), ("style_leader_confirm", "score_style_leader_confirm"), ("style_rotation_quality", "score_style_rotation_quality")]


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
        period_ret = float(np.mean(rets)) if rets else 0.0
        if not rets:
            rejects["empty_periods"] += 1
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
