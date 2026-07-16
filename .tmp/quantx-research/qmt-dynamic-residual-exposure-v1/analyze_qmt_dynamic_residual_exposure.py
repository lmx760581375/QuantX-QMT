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
    spec = importlib.util.spec_from_file_location("exp130_for_dynamic_residual", EXP130_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP130_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp130_for_dynamic_residual"] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()
BASE_FEATURES = list(EXP130.FEATURES)
EXPOSURE_FEATURES = [
    "mkt_beta_rank", "mom_beta_rank", "lowvol_beta_rank", "liq_beta_rank", "price_beta_rank",
    "resid5_rank", "resid20_rank", "resid5_accel_rank", "resid_vol20_low_rank",
    "beta_instability_low_rank", "mom_beta_chg_rank", "liq_beta_chg_rank", "risk_neutral_mom_rank",
    "factor_aligned_resid_rank", "resid_repair_rank", "crowding_resid_quality", "exposure_breakout_quality",
]
FEATURES = BASE_FEATURES + EXPOSURE_FEATURES


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
        "experiment": "qmt_dynamic_residual_exposure_v1",
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
            "causality": "Signal date T uses completed daily OHLCV/amount/VWAP through T only. Dynamic factor portfolios and rolling stock exposures are estimated from returns through T. Each test year is trained only on rows with year < test year. Labels and replay use T+1 open entry and T+6 open target exit.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only. No news, announcements, LHB, ETF, northbound, financing, static industry/concept membership, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(result, hashlib.sha256(output.read_bytes()).hexdigest())
    return 0


def rank_series(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).rank(pct=True, ascending=ascending).fillna(0.5).to_numpy(dtype=float)


def factor_returns(ret_window: np.ndarray, ret20: np.ndarray, amount: np.ndarray, vol20: np.ndarray, price: np.ndarray) -> np.ndarray:
    valid = np.isfinite(ret20) & np.isfinite(amount) & np.isfinite(vol20) & np.isfinite(price)
    factors = [np.nanmean(ret_window, axis=1)]
    specs = [
        (ret20, True),
        (vol20, False),
        (amount, True),
        (price, False),
    ]
    for values, high in specs:
        score = pd.Series(values).rank(pct=True, ascending=True).to_numpy(dtype=float)
        if high:
            mask = valid & (score >= 0.80)
        else:
            mask = valid & (score <= 0.20)
        if int(mask.sum()) < 50:
            factors.append(np.nanmean(ret_window, axis=1))
        else:
            factors.append(np.nanmean(ret_window[:, mask], axis=1))
    return np.column_stack(factors)


def rolling_exposure_features(market: dict[str, Any], idx: int, tradable: np.ndarray) -> dict[str, np.ndarray]:
    f = market["frames"]
    close = f["close"]
    amount = f["amount"]
    ret1 = close.pct_change(fill_method=None)
    ret_window = ret1.iloc[idx - 59: idx + 1].to_numpy(dtype=float)
    ret20 = (close.iloc[idx] / close.shift(20).iloc[idx] - 1.0).to_numpy(dtype=float)
    ret5 = (close.iloc[idx] / close.shift(5).iloc[idx] - 1.0).to_numpy(dtype=float)
    amount_now = amount.iloc[idx].to_numpy(dtype=float)
    price = close.iloc[idx].to_numpy(dtype=float)
    vol20 = ret1.rolling(20, min_periods=10).std().iloc[idx].to_numpy(dtype=float)
    fac60 = factor_returns(ret_window, ret20, amount_now, vol20, price)
    betas60, resid60 = regress_betas(ret_window, fac60)
    betas20, resid20_path = regress_betas(ret_window[-20:], fac60[-20:])
    resid5 = np.nanmean(resid60[-5:], axis=0)
    resid20 = np.nanmean(resid60[-20:], axis=0)
    resid_vol20 = np.nanstd(resid60[-20:], axis=0)
    beta_chg = betas20 - betas60
    beta_instability = np.nanmean(np.abs(beta_chg), axis=0)
    mkt_beta, mom_beta, lowvol_beta, liq_beta, price_beta = betas60
    resid5_rank = rank_series(resid5)
    resid20_rank = rank_series(resid20)
    resid5_accel_rank = rank_series(resid5 - resid20 / 4.0)
    resid_repair_rank = rank_series(-resid20 + resid5)
    mom_beta_rank = rank_series(mom_beta)
    liq_beta_rank = rank_series(liq_beta)
    risk_neutral_mom = resid20_rank + 0.35 * resid5_accel_rank - 0.25 * rank_series(np.abs(mkt_beta), ascending=False)
    factor_aligned = 0.30 * mom_beta_rank + 0.25 * liq_beta_rank + 0.25 * resid20_rank + 0.20 * resid5_accel_rank
    crowding_resid = 0.30 * resid20_rank + 0.20 * resid5_accel_rank + 0.20 * rank_series(resid_vol20, ascending=False) + 0.15 * rank_series(beta_instability, ascending=False) + 0.15 * rank_series(-np.maximum(mom_beta, 0.0))
    exposure_breakout = 0.25 * rank_series(beta_chg[1]) + 0.20 * rank_series(beta_chg[3]) + 0.20 * resid5_accel_rank + 0.20 * resid20_rank + 0.15 * rank_series(resid_vol20, ascending=False)
    data = {
        "mkt_beta_rank": rank_series(mkt_beta),
        "mom_beta_rank": mom_beta_rank,
        "lowvol_beta_rank": rank_series(lowvol_beta),
        "liq_beta_rank": liq_beta_rank,
        "price_beta_rank": rank_series(price_beta),
        "resid5_rank": resid5_rank,
        "resid20_rank": resid20_rank,
        "resid5_accel_rank": resid5_accel_rank,
        "resid_vol20_low_rank": rank_series(resid_vol20, ascending=False),
        "beta_instability_low_rank": rank_series(beta_instability, ascending=False),
        "mom_beta_chg_rank": rank_series(beta_chg[1]),
        "liq_beta_chg_rank": rank_series(beta_chg[3]),
        "risk_neutral_mom_rank": rank_series(risk_neutral_mom),
        "factor_aligned_resid_rank": rank_series(factor_aligned),
        "resid_repair_rank": resid_repair_rank,
        "crowding_resid_quality": rank_series(crowding_resid),
        "exposure_breakout_quality": rank_series(exposure_breakout),
    }
    return {name: np.asarray(value, dtype=float)[tradable] for name, value in data.items()}


def regress_betas(ret_window: np.ndarray, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.nan_to_num(ret_window, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xtx = x.T @ x + np.eye(x.shape[1]) * 1e-5
    beta = np.linalg.solve(xtx, x.T @ y)
    resid = y - x @ beta
    return beta, resid


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
        exposure = pd.DataFrame({"instrument": symbols_arr[tradable], **rolling_exposure_features(market, idx, tradable)})
        frames.append(group.merge(exposure, on="instrument", how="left").replace([np.inf, -np.inf], np.nan).fillna(0.5))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        exp_reg = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=23, max_depth=4, min_child_samples=650, subsample=0.90, colsample_bytree=0.95, reg_alpha=1.5, reg_lambda=9.0, random_state=RANDOM_SEED + year + 31, n_jobs=6, verbosity=-1)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"])
        exp_reg.fit(train[EXPOSURE_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20"]], train["exec_label5_open"])
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = cls.predict_proba(out[FEATURES])[:, 1]
        out["score_exp_reg"] = exp_reg.predict(out[EXPOSURE_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20"]])
        out["score_blend"] = 0.55 * rank_by_session(out, "score_reg") + 0.25 * rank_by_session(out, "score_cls") + 0.20 * rank_by_session(out, "score_exp_reg")
        out["score_crowding_resid_quality"] = out["crowding_resid_quality"]
        out["score_exposure_breakout_quality"] = out["exposure_breakout_quality"]
        out["score_factor_aligned_resid_rank"] = out["factor_aligned_resid_rank"]
        rows.append(out)
        folds.append({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        imp.update({f"exp_reg::{name}": float(value) for name, value in zip(EXPOSURE_FEATURES + ["mkt_ret20_median", "mkt_breadth20", "mkt_disp20"], exp_reg.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def rank_by_session(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("session")[col].rank(pct=True, method="first")


def variants() -> list[tuple[str, str]]:
    return [
        ("reg", "score_reg"),
        ("cls", "score_cls"),
        ("exp_reg", "score_exp_reg"),
        ("blend", "score_blend"),
        ("crowding_resid_quality", "score_crowding_resid_quality"),
        ("exposure_breakout_quality", "score_exposure_breakout_quality"),
        ("factor_aligned_resid", "score_factor_aligned_resid_rank"),
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
