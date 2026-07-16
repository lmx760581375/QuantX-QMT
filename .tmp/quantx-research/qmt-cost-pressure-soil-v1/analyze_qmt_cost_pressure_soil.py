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
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"
TOPKS = (10, 15, 20, 30)
RANDOM_SEED = 20260714


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("exp130_for_cost_pressure", EXP130_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP130_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp130_for_cost_pressure"] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()
BASE_FEATURES = list(EXP130.FEATURES)
CHIP_FEATURES = [
    "cost20_profit_rank", "cost60_profit_rank", "cost20_distance_rank", "cost60_distance_rank",
    "cost20_overhang_low_rank", "cost60_overhang_low_rank", "cost20_dense_rank", "cost60_dense_rank",
    "cost20_support_rank", "cost60_support_rank", "cost20_turnover_rank", "cost60_turnover_rank",
    "cost20_breakout_rank", "cost60_breakout_rank", "cost_repair_rank", "cost_squeeze_rank",
    "cost_accumulation_rank", "cost_pressure_quality", "cost_breakout_quality", "cost_repair_quality",
]
FEATURES = BASE_FEATURES + CHIP_FEATURES


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
    market = EXP130.EXP124.load_market(args)
    panel = build_panel(market, args)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_cost_pressure_soil_v1",
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
            "exp40_due5_top20_dev_label": 0.014170,
            "exp40_due5_top20_2026_label": 0.012520,
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
            "features": FEATURES,
            "chip_feature_windows": [20, 60],
            "causality": "Signal date T uses completed daily OHLCV/amount/VWAP through T only. Rolling cost distribution proxies are computed from dates <= T. Each test year is predicted by models trained only on rows with year < test year. Labels and replay use T+1 open entry and T+6 open target exit with execution rejection handled by the existing QMT open replay.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only. No news, announcements, LHB, ETF, northbound, financing, static industry/concept membership, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def rank_series(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).rank(pct=True, ascending=ascending).fillna(0.5).to_numpy(dtype=float)


def chip_metrics_for_day(close_hist: np.ndarray, amount_hist: np.ndarray, current_close: np.ndarray) -> dict[str, np.ndarray]:
    amount_clean = np.where(np.isfinite(amount_hist) & (amount_hist > 0), amount_hist, np.nan)
    total_amount = np.nansum(amount_clean, axis=0)
    valid = np.isfinite(current_close) & (current_close > 0) & np.isfinite(total_amount) & (total_amount > 0)
    weights = np.divide(amount_clean, total_amount[None, :], out=np.zeros_like(amount_clean, dtype=float), where=total_amount[None, :] > 0)
    rel = close_hist / current_close[None, :] - 1.0
    below = rel <= 0.0
    above = rel > 0.0
    near = np.abs(rel) <= 0.035
    support_band = (rel >= -0.08) & (rel <= 0.0)
    overhang_band = (rel > 0.0) & (rel <= 0.12)
    profit = np.nansum(np.where(below, weights, 0.0), axis=0)
    overhang = np.nansum(np.where(overhang_band, weights, 0.0), axis=0)
    dense = np.nansum(np.where(near, weights, 0.0), axis=0)
    support = np.nansum(np.where(support_band, weights, 0.0), axis=0)
    turnover = np.nanmean(np.where(np.isfinite(amount_clean), amount_clean, np.nan), axis=0)
    above_rel = np.where(above, rel, np.nan)
    below_rel = np.where(below, -rel, np.nan)
    nearest_overhang = np.nanmin(above_rel, axis=0)
    nearest_support = np.nanmin(below_rel, axis=0)
    cost_mean = np.nansum(weights * close_hist, axis=0)
    distance = current_close / cost_mean - 1.0
    nearest_overhang = np.where(np.isfinite(nearest_overhang), nearest_overhang, 1.0)
    nearest_support = np.where(np.isfinite(nearest_support), nearest_support, 1.0)
    return {
        "profit": np.where(valid, profit, np.nan),
        "overhang": np.where(valid, overhang, np.nan),
        "dense": np.where(valid, dense, np.nan),
        "support": np.where(valid, support, np.nan),
        "turnover": np.where(valid, turnover, np.nan),
        "distance": np.where(valid, distance, np.nan),
        "nearest_overhang": np.where(valid, nearest_overhang, np.nan),
        "nearest_support": np.where(valid, nearest_support, np.nan),
    }


def chip_features_for_day(market: dict[str, Any], idx: int, tradable: np.ndarray) -> dict[str, np.ndarray]:
    arr = market["arrays"]
    close = arr["close"]
    amount = arr["amount"]
    current_close = close[idx]
    metrics20 = chip_metrics_for_day(close[idx - 19: idx + 1], amount[idx - 19: idx + 1], current_close)
    metrics60 = chip_metrics_for_day(close[idx - 59: idx + 1], amount[idx - 59: idx + 1], current_close)
    profit20_rank = rank_series(metrics20["profit"])
    profit60_rank = rank_series(metrics60["profit"])
    distance20_rank = rank_series(metrics20["distance"])
    distance60_rank = rank_series(metrics60["distance"])
    overhang20_low = rank_series(metrics20["overhang"], ascending=False)
    overhang60_low = rank_series(metrics60["overhang"], ascending=False)
    dense20_rank = rank_series(metrics20["dense"])
    dense60_rank = rank_series(metrics60["dense"])
    support20_rank = rank_series(metrics20["support"])
    support60_rank = rank_series(metrics60["support"])
    turnover20_rank = rank_series(np.log1p(metrics20["turnover"]))
    turnover60_rank = rank_series(np.log1p(metrics60["turnover"]))
    breakout20_rank = rank_series(metrics20["nearest_overhang"], ascending=False)
    breakout60_rank = rank_series(metrics60["nearest_overhang"], ascending=False)
    repair_rank = rank_series(metrics20["profit"] - metrics60["profit"])
    squeeze_rank = rank_series(metrics20["dense"] + metrics20["support"] - metrics20["overhang"])
    accumulation_rank = rank_series(metrics60["support"] + metrics60["dense"] - np.abs(metrics60["distance"]))
    features = {
        "cost20_profit_rank": profit20_rank,
        "cost60_profit_rank": profit60_rank,
        "cost20_distance_rank": distance20_rank,
        "cost60_distance_rank": distance60_rank,
        "cost20_overhang_low_rank": overhang20_low,
        "cost60_overhang_low_rank": overhang60_low,
        "cost20_dense_rank": dense20_rank,
        "cost60_dense_rank": dense60_rank,
        "cost20_support_rank": support20_rank,
        "cost60_support_rank": support60_rank,
        "cost20_turnover_rank": turnover20_rank,
        "cost60_turnover_rank": turnover60_rank,
        "cost20_breakout_rank": breakout20_rank,
        "cost60_breakout_rank": breakout60_rank,
        "cost_repair_rank": repair_rank,
        "cost_squeeze_rank": squeeze_rank,
        "cost_accumulation_rank": accumulation_rank,
    }
    features["cost_pressure_quality"] = 0.24 * profit20_rank + 0.18 * profit60_rank + 0.18 * overhang20_low + 0.14 * support20_rank + 0.12 * dense20_rank + 0.08 * repair_rank + 0.06 * turnover20_rank
    features["cost_breakout_quality"] = 0.22 * breakout20_rank + 0.18 * breakout60_rank + 0.16 * profit20_rank + 0.14 * overhang20_low + 0.12 * distance20_rank + 0.10 * turnover20_rank + 0.08 * dense20_rank
    features["cost_repair_quality"] = 0.22 * repair_rank + 0.18 * support20_rank + 0.16 * accumulation_rank + 0.14 * squeeze_rank + 0.12 * profit60_rank + 0.10 * overhang60_low + 0.08 * turnover60_rank
    return {name: np.asarray(value, dtype=float)[tradable] for name, value in features.items()}


def build_panel(market: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    base = EXP130.build_panel(market, args)
    if base.empty:
        return base
    frames = []
    symbols_arr = np.asarray(market["symbols"], dtype=object)
    history = (market["frames"]["open"].notna() & (market["frames"]["open"] > 0)).cumsum()
    arr = market["arrays"]
    for session, group in base.groupby("session", sort=True):
        idx = market["date_index"][session]
        raw = arr["open"][idx + 1 + args.horizon] / arr["open"][idx + 1] - 1.0
        finite = np.isfinite(raw)
        tradable = finite & np.isfinite(arr["open"][idx]) & (arr["open"][idx] > 0) & (arr["is_st"][idx] < 0.5) & (history.iloc[idx].to_numpy(dtype=float) >= 130)
        chip = pd.DataFrame({"instrument": symbols_arr[tradable], **chip_features_for_day(market, idx, tradable)})
        merged = group.merge(chip, on="instrument", how="left")
        frames.append(merged.replace([np.inf, -np.inf], np.nan).fillna(0.5))
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel


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
        reg = LGBMRegressor(n_estimators=320, learning_rate=0.032, num_leaves=31, max_depth=5, min_child_samples=520, subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=7.0, random_state=RANDOM_SEED + year, n_jobs=6, verbosity=-1)
        cls = LGBMClassifier(n_estimators=260, learning_rate=0.032, num_leaves=31, max_depth=5, min_child_samples=520, subsample=0.85, colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=7.0, random_state=RANDOM_SEED + year + 17, n_jobs=6, verbosity=-1)
        chip_reg = LGBMRegressor(n_estimators=240, learning_rate=0.035, num_leaves=23, max_depth=4, min_child_samples=650, subsample=0.90, colsample_bytree=0.95, reg_alpha=1.5, reg_lambda=9.0, random_state=RANDOM_SEED + year + 31, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        chip_reg.fit(train[CHIP_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_near_high20"]], train["exec_label5_open"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_chip_reg"] = chip_reg.predict(out[CHIP_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_near_high20"]])
        out["score_blend"] = 0.55 * rank_by_session(out, "score_reg") + 0.30 * rank_by_session(out, "score_cls") + 0.15 * rank_by_session(out, "score_chip_reg")
        out["score_cost_pressure_quality"] = out["cost_pressure_quality"]
        out["score_cost_breakout_quality"] = out["cost_breakout_quality"]
        out["score_cost_repair_quality"] = out["cost_repair_quality"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        imp.update({f"chip_reg::{name}": float(value) for name, value in zip(CHIP_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20", "mkt_near_high20"], chip_reg.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def rank_by_session(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("session")[col].rank(pct=True, method="first")


def variants() -> list[tuple[str, str]]:
    return [
        ("reg", "score_reg"),
        ("cls", "score_cls"),
        ("chip_reg", "score_chip_reg"),
        ("blend", "score_blend"),
        ("cost_pressure_quality", "score_cost_pressure_quality"),
        ("cost_breakout_quality", "score_cost_breakout_quality"),
        ("cost_repair_quality", "score_cost_repair_quality"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": session,
                    "year": int(session[:4]),
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
        by_year = {}
        for year, yg in group.groupby("year"):
            by_year[str(int(year))] = {
                "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                "entry_ok": float(yg["entry_ok"].mean()),
                "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
            }
        out[f"{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": by_year,
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, score_col in variants():
        for topk in (10, 20):
            selections = {}
            for session, group in scored.groupby("session", sort=True):
                selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)
                selections[str(session)] = list(selected["instrument"].astype(str))
            out[f"{variant}::top{topk}"] = EXP130.replay_open_selection(selections, market, horizon)
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
        rows.append({
            "key": key,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "ret2026": float(annual.get("2026", 0.0)),
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "mean_raw5_open": float(label.get("mean_raw5_open", 0.0)),
        })
    rows.sort(key=lambda row: (row["final_multiple"], row["ret2026"]), reverse=True)
    return rows


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:80]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print("sha256:" + checksum)
    for row in result["leaderboard"][:16]:
        print(f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} hold={row['avg_hold_days']:.2f} label={row['mean_exec_label5_open']:+.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
