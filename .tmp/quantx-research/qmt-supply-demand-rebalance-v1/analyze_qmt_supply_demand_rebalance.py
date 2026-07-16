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
    "amount_rank", "amt_ratio5_rank", "amt_ratio20_rank", "amt_dryup5_rank",
    "close_strength_rank", "upper_wick_low_rank", "lower_wick_rank", "range20_low_rank",
    "near_high20_rank", "drawdown20_rank", "break20_rank", "vwap_support_rank",
    "spike_age_rank", "spike_amount_rank", "post_spike_return_rank", "post_spike_drawdown_rank",
    "post_spike_hold_rank", "post_spike_dryup_rank", "post_spike_range_low_rank",
    "down_volume_absorb_rank", "up_volume_confirm_rank", "pressure_release_rank",
    "supply_absorption", "dryup_reaccumulation", "reconfirm_breakout", "rebalance_quality",
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
            "causality": "For each after-close signal T, all supply-demand rebalance features use completed daily bars through T only. Labels and replay use pre-scheduled T+1 open entry and T+6 open target exit. Execution-day data is used only for realized fills/rejections.",
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
    vwap = f["vwap"]
    ret1 = close.pct_change(fill_method=None)
    ret3 = close / close.shift(3) - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret10 = close / close.shift(10) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amount5 = amount.rolling(5, min_periods=3).mean()
    amount20 = amount.rolling(20, min_periods=10).mean()
    amount60 = amount.rolling(60, min_periods=20).mean()
    amt_ratio5 = amount5 / amount20
    amt_ratio20 = amount / amount20
    high20 = high.rolling(20, min_periods=10).max()
    prev_high20 = high.shift(1).rolling(20, min_periods=10).max()
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    lower_wick = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    near_high20 = close / high20
    drawdown20 = close / high20 - 1.0
    break20 = close / prev_high20 - 1.0
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    vwap_support = close / vwap - 1.0
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
        amt5 = amount5.iloc[idx].to_numpy(dtype=float)
        amt20 = amount20.iloc[idx].to_numpy(dtype=float)
        amt60 = amount60.iloc[idx].to_numpy(dtype=float)
        c_strength = close_strength.iloc[idx].to_numpy(dtype=float)
        u_wick = upper_wick.iloc[idx].to_numpy(dtype=float)
        l_wick = lower_wick.iloc[idx].to_numpy(dtype=float)
        near_high = near_high20.iloc[idx].to_numpy(dtype=float)
        drawdown = drawdown20.iloc[idx].to_numpy(dtype=float)
        brk20 = break20.iloc[idx].to_numpy(dtype=float)
        rng20 = range20.iloc[idx].to_numpy(dtype=float)
        vwap_sup = vwap_support.iloc[idx].to_numpy(dtype=float)
        universe_ok = finite & np.isfinite(arr["open"][idx]) & (arr["open"][idx] > 0) & (arr["is_st"][idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        if universe_ok.sum() < 500:
            continue

        ret_window20 = ret1.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        amount_window20 = amount.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        close_window20 = close.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        low_window20 = low.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        high_window20 = high.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        strength_window20 = close_strength.iloc[idx - 19: idx + 1].to_numpy(dtype=float, copy=False)
        amount_clean = np.nan_to_num(amount_window20, nan=-1.0, posinf=-1.0, neginf=-1.0)
        spike_pos = np.argmax(amount_clean, axis=0)
        cols = np.arange(len(symbols))
        spike_amt = amount_window20[spike_pos, cols]
        spike_close = close_window20[spike_pos, cols]
        spike_ret = ret_window20[spike_pos, cols]
        days_since_spike = 19 - spike_pos
        after_mask = np.arange(20)[:, None] >= spike_pos[None, :]
        low_after_spike = np.nanmin(np.where(after_mask, low_window20, np.nan), axis=0)
        high_after_spike = np.nanmax(np.where(after_mask, high_window20, np.nan), axis=0)
        post_spike_return = arr["close"][idx] / spike_close - 1.0
        post_spike_drawdown = low_after_spike / spike_close - 1.0
        post_spike_hold = arr["close"][idx] / np.maximum(spike_close, 1e-12) - 1.0 + post_spike_drawdown
        post_spike_dryup = 1.0 - (amt5 / np.maximum(spike_amt, 1e-12))
        post_spike_range = high_after_spike / np.maximum(low_after_spike, 1e-12) - 1.0
        down_volume = np.nanmean(np.where(ret_window20 < 0, amount_window20 / np.maximum(amt20[None, :], 1e-12), np.nan), axis=0)
        up_volume = np.nanmean(np.where(ret_window20 > 0, amount_window20 / np.maximum(amt20[None, :], 1e-12), np.nan), axis=0)
        down_strength = np.nanmean(np.where(ret_window20 < 0, strength_window20, np.nan), axis=0)
        up_strength = np.nanmean(np.where(ret_window20 > 0, strength_window20, np.nan), axis=0)
        pressure_release = np.nanmax(np.where((ret_window20 < 0) | (strength_window20 < 0.45), amount_window20 / np.maximum(amt20[None, :], 1e-12), np.nan), axis=0)
        amt_dryup5 = 1.0 - amt5 / np.maximum(amt20, 1e-12)

        data = {
            "ret1_rank": rank_series(r1),
            "ret3_rank": rank_series(r3),
            "ret5_rank": rank_series(r5),
            "ret10_rank": rank_series(r10),
            "ret20_rank": rank_series(r20),
            "ret60_rank": rank_series(r60),
            "amount_rank": rank_series(np.log1p(amt)),
            "amt_ratio5_rank": rank_series(amt5 / np.maximum(amt20, 1e-12)),
            "amt_ratio20_rank": rank_series(amt / np.maximum(amt20, 1e-12)),
            "amt_dryup5_rank": rank_series(amt_dryup5),
            "close_strength_rank": rank_series(c_strength),
            "upper_wick_low_rank": rank_series(u_wick, ascending=False),
            "lower_wick_rank": rank_series(l_wick),
            "range20_low_rank": rank_series(rng20, ascending=False),
            "near_high20_rank": rank_series(near_high),
            "drawdown20_rank": rank_series(drawdown),
            "break20_rank": rank_series(brk20),
            "vwap_support_rank": rank_series(vwap_sup),
            "spike_age_rank": rank_series(days_since_spike, ascending=False),
            "spike_amount_rank": rank_series(spike_amt / np.maximum(amt60, 1e-12)),
            "post_spike_return_rank": rank_series(post_spike_return),
            "post_spike_drawdown_rank": rank_series(post_spike_drawdown),
            "post_spike_hold_rank": rank_series(post_spike_hold),
            "post_spike_dryup_rank": rank_series(post_spike_dryup),
            "post_spike_range_low_rank": rank_series(post_spike_range, ascending=False),
            "down_volume_absorb_rank": rank_series(down_strength - 0.15 * down_volume),
            "up_volume_confirm_rank": rank_series(up_strength + 0.20 * up_volume),
            "pressure_release_rank": rank_series(pressure_release),
        }
        data["supply_absorption"] = 0.22 * data["pressure_release_rank"] + 0.22 * data["down_volume_absorb_rank"] + 0.18 * data["post_spike_hold_rank"] + 0.14 * data["post_spike_drawdown_rank"] + 0.12 * data["upper_wick_low_rank"] + 0.12 * data["vwap_support_rank"]
        data["dryup_reaccumulation"] = 0.22 * data["post_spike_dryup_rank"] + 0.20 * data["range20_low_rank"] + 0.18 * data["near_high20_rank"] + 0.16 * data["ret20_rank"] + 0.12 * data["close_strength_rank"] + 0.12 * data["amount_rank"]
        data["reconfirm_breakout"] = 0.22 * data["break20_rank"] + 0.18 * data["amt_ratio20_rank"] + 0.18 * data["up_volume_confirm_rank"] + 0.16 * data["close_strength_rank"] + 0.14 * data["post_spike_hold_rank"] + 0.12 * data["upper_wick_low_rank"]
        data["rebalance_quality"] = 0.30 * data["supply_absorption"] + 0.25 * data["dryup_reaccumulation"] + 0.20 * data["reconfirm_breakout"] + 0.15 * data["ret60_rank"] + 0.10 * data["vwap_support_rank"]
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
        cls = LGBMClassifier(n_estimators=240, learning_rate=0.035, num_leaves=31, max_depth=5, min_child_samples=500, subsample=0.85, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=6.0, random_state=int(year) + 47, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_supply_absorption"] = out["supply_absorption"]
        out["score_dryup_reaccumulation"] = out["dryup_reaccumulation"]
        out["score_rebalance_quality"] = out["rebalance_quality"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [("reg", "score_reg"), ("cls", "score_cls"), ("blend", "score_blend"), ("supply_absorption", "score_supply_absorption"), ("dryup_reaccumulation", "score_dryup_reaccumulation"), ("rebalance_quality", "score_rebalance_quality")]


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
