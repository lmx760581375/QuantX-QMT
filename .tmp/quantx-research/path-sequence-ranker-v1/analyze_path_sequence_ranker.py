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
from lightgbm import LGBMClassifier


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP124_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-close-execution-formal-proxy-v1/analyze_qmt_close_execution_formal_proxy.py"
EXP129_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-supply-demand-rebalance-v1/analyze_qmt_supply_demand_rebalance.py"
RANDOM_SEED = 20260714
TOPKS = (10, 15, 20, 30)
PATH_WINDOWS = (5, 20)
RPS_HORIZONS = (1, 3, 5, 10, 20, 60, 120)
VOL_PATHS = (5, 20)
COST = 0.00154


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP124 = load_module("exp124_for_path_sequence_rebuild", EXP124_PATH)
EXP129 = load_module("exp129_for_path_sequence_rebuild", EXP129_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--dev-top200", required=True)
    parser.add_argument("--forward-top200", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=30000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP124.load_market(args)
    top200 = load_top200(args.dev_top200, args.forward_top200)
    panel = build_panel(market, top200, args.horizon)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "rebuild_complete_not_checksum_exact",
        "experiment": "path_sequence_ranker_v1_rebuild",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "feature_importance": summarize_importance(importances),
        "references": {
            "exp40_dev_due5_path_topq_top20_label5": 0.014170,
            "exp40_2026_due5_path_topq_top20_label5": 0.012520,
            "exp40_dev_reb5_multiple": 8.7749,
            "exp40_2026_reb5_return": 0.1918,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "load_start": args.load_start,
            "end": args.end,
            "dev_top200": args.dev_top200,
            "forward_top200": args.forward_top200,
            "horizon": args.horizon,
            "features": FEATURES,
            "causality": "Base candidate scores are frozen after-close PredictionStore records for signal day T. Path features use completed QMT/qlib daily OHLCV/VWAP through T only. Labels use T+1 open to T+6 open and are never used in same-year training. Development folds train on years strictly before the test year; 2026 trains only on 2021-2025 candidate rows.",
            "known_difference_from_archived_exp40": "Original Exp40 scripts and base OOS prediction checksum are absent locally. This rebuild uses a re-created base 5d store with matching data/fold sizes but lower base IC, and omits static industry/concept path features unless later restored from the archived script.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def load_top200(dev_path: str, forward_path: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(dev_path), pd.read_parquet(forward_path)], ignore_index=True)
    frame["session"] = frame["session"].astype(str)
    frame["instrument"] = frame["instrument"].astype(str)
    frame["year"] = frame["session"].str.slice(0, 4).astype(int)
    frame["base_rank_pct"] = 1.0 - (frame["base_rank"].astype(float) - 1.0) / 199.0
    return frame.sort_values(["session", "base_rank", "instrument"]).reset_index(drop=True)


def rank_frame(frame: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True, ascending=ascending).fillna(0.5)


def finite_pct_change(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / den.replace(0, np.nan) - 1.0


def build_panel(market: dict[str, Any], top200: pd.DataFrame, horizon: int) -> pd.DataFrame:
    f = market["frames"]
    arr = market["arrays"]
    dates = market["dates"]
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
    volume = f["volume"]
    vwap = f["vwap"]
    amount = f["amount"]
    ret1 = close.pct_change(fill_method=None)
    returns = {h: finite_pct_change(close, close.shift(h)) for h in RPS_HORIZONS}
    rps = {f"rps{h}": rank_frame(returns[h]) for h in RPS_HORIZONS}
    volume_ratio = {w: volume / volume.rolling(w, min_periods=max(2, w // 2)).mean().replace(0, np.nan) for w in VOL_PATHS}
    vol_rank = {f"vol_ratio{w}_rank": rank_frame(volume_ratio[w]) for w in VOL_PATHS}
    vol20 = ret1.abs().rolling(20, min_periods=10).mean()
    vol60 = ret1.abs().rolling(60, min_periods=20).mean()
    vol_path_rank = {"vol20_rank": rank_frame(vol20), "vol20_low_rank": rank_frame(vol20, ascending=False), "vol60_rank": rank_frame(vol60)}
    ma_dist = {w: close / close.rolling(w, min_periods=max(2, w // 2)).mean().replace(0, np.nan) - 1.0 for w in (5, 20, 60)}
    ma_rank = {f"ma{w}_dist_rank": rank_frame(ma_dist[w]) for w in (5, 20, 60)}
    high20 = high.rolling(20, min_periods=10).max()
    high60 = high.rolling(60, min_periods=20).max()
    near_high20 = rank_frame(close / high20.replace(0, np.nan))
    near_high60 = rank_frame(close / high60.replace(0, np.nan))
    drawdown20 = rank_frame(close / high20.replace(0, np.nan) - 1.0)
    range20 = rank_frame((high / low.replace(0, np.nan) - 1.0).rolling(20, min_periods=10).mean(), ascending=False)
    close_strength = rank_frame((close - low) / (high - low).replace(0, np.nan))
    close_vwap = rank_frame(close / vwap.replace(0, np.nan) - 1.0)
    amount_rank = rank_frame(np.log1p(amount))
    limit_touch20 = ((high / close.shift(1) - 1.0) >= 0.095).rolling(20, min_periods=5).sum()
    near_limit20 = ((high / close.shift(1) - 1.0) >= 0.075).rolling(20, min_periods=5).sum()
    limit_close5 = ((close / close.shift(1) - 1.0) >= 0.095).rolling(5, min_periods=2).sum()
    feature_frames = {
        **rps,
        **vol_rank,
        **vol_path_rank,
        **ma_rank,
        "rps60_minus120": rps["rps60"] - rps["rps120"],
        "rps20_minus60": rps["rps20"] - rps["rps60"],
        "near_high20": near_high20,
        "near_high60": near_high60,
        "drawdown20": drawdown20,
        "range20_low": range20,
        "close_strength": close_strength,
        "close_vwap": close_vwap,
        "amount_rank": amount_rank,
        "limit_touch20_rank": rank_frame(limit_touch20),
        "near_limit20_rank": rank_frame(near_limit20),
        "limit_close5_rank": rank_frame(limit_close5),
    }
    feature_arrays = {name: frame.to_numpy(dtype=np.float32, copy=False) for name, frame in feature_frames.items()}
    history = (open_.notna() & (open_ > 0)).cumsum().to_numpy(dtype=float, copy=False)
    rows: list[pd.DataFrame] = []
    for session, candidates in top200.groupby("session", sort=True):
        idx = market["date_index"].get(session)
        if idx is None:
            continue
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if idx < 130 or exit_idx >= len(dates):
            continue
        cols = np.asarray([symbol_index.get(sym, -1) for sym in candidates["instrument"]], dtype=int)
        valid_col = cols >= 0
        candidates = candidates.iloc[np.flatnonzero(valid_col)].copy()
        cols = cols[valid_col]
        if len(cols) < max(TOPKS):
            continue
        raw = arr["open"][exit_idx] / arr["open"][entry_idx] - 1.0
        universe_ok = (
            np.isfinite(raw)
            & np.isfinite(arr["open"][idx])
            & (arr["open"][idx] > 0)
            & np.isfinite(arr["volume"][idx])
            & (arr["volume"][idx] > 0)
            & (arr["is_st"][idx] < 0.5)
            & (history[idx] >= 130)
        )
        if int(universe_ok.sum()) < 500:
            continue
        entry_gap = arr["open"][entry_idx] / arr["close"][idx] - 1.0
        exit_gap = arr["open"][exit_idx] / arr["close"][exit_idx - 1] - 1.0
        entry_ok = universe_ok & np.isfinite(arr["volume"][entry_idx]) & (arr["volume"][entry_idx] > 0) & (entry_gap < 0.095)
        exit_ok = np.isfinite(raw) & np.isfinite(arr["volume"][exit_idx]) & (arr["volume"][exit_idx] > 0) & (exit_gap > -0.095)
        bench = float(np.nanmean(raw[universe_ok]))
        exec_label = np.where(entry_ok & exit_ok, raw - COST, -0.08)
        block: dict[str, Any] = {
            "session": np.full(len(cols), session, dtype=object),
            "year": np.full(len(cols), int(session[:4]), dtype=int),
            "instrument": candidates["instrument"].to_numpy(dtype=object),
            "base_score": candidates["score"].to_numpy(dtype=float),
            "base_rank": candidates["base_rank"].to_numpy(dtype=float),
            "base_rank_pct": candidates["base_rank_pct"].to_numpy(dtype=float),
            "raw5_open": raw[cols],
            "label5_open": raw[cols] - bench,
            "exec_label5_open": exec_label[cols] - bench,
            "entry_ok": entry_ok[cols].astype(float),
            "exit_ok": exit_ok[cols].astype(float),
        }
        for name, values in path_feature_values(feature_arrays, idx).items():
            block[name] = values[cols]
        for name, value in market_features(returns, volume_ratio, idx, universe_ok).items():
            block[name] = np.full(len(cols), value, dtype=float)
        rows.append(pd.DataFrame(block).replace([np.inf, -np.inf], np.nan).fillna(0.5))
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return panel
    pct = panel.groupby("session")["label5_open"].rank(pct=True, method="first")
    panel["target_quintile"] = np.minimum(4, np.floor(pct.to_numpy(dtype=float) * 5.0).astype(int))
    panel["target_quintile"] = np.maximum(0, panel["target_quintile"])
    return panel


def path_feature_values(feature_arrays: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, arr in feature_arrays.items():
        out[name] = arr[idx]
        for window in PATH_WINDOWS:
            hist = arr[idx - window + 1: idx + 1]
            out[f"{name}_mean{window}"] = np.nanmean(hist, axis=0)
            out[f"{name}_min{window}"] = np.nanmin(hist, axis=0)
            out[f"{name}_max{window}"] = np.nanmax(hist, axis=0)
        out[f"{name}_slope20"] = np.nanmean(arr[idx - 4: idx + 1], axis=0) - np.nanmean(arr[idx - 19: idx - 14], axis=0)
    return out


def market_features(returns: dict[int, pd.DataFrame], volume_ratio: dict[int, pd.DataFrame], idx: int, ok: np.ndarray) -> dict[str, float]:
    ret20 = returns[20].iloc[idx].to_numpy(dtype=float)
    ret60 = returns[60].iloc[idx].to_numpy(dtype=float)
    vol20 = volume_ratio[20].iloc[idx].to_numpy(dtype=float)
    return {
        "market_ret20_median": float(np.nanmedian(ret20[ok])),
        "market_ret60_median": float(np.nanmedian(ret60[ok])),
        "market_breadth20": float(np.nanmean(ret20[ok] > 0)),
        "market_breadth60": float(np.nanmean(ret60[ok] > 0)),
        "market_ret20_disp": float(np.nanstd(ret20[ok])),
        "market_ret60_disp": float(np.nanstd(ret60[ok])),
        "market_vol_ratio20_median": float(np.nanmedian(vol20[ok])),
    }


FEATURES = [
    "base_score", "base_rank_pct",
    *[f"rps{h}" for h in RPS_HORIZONS],
    *[f"rps{h}_{stat}{w}" for h in RPS_HORIZONS for w in PATH_WINDOWS for stat in ("mean", "min", "max")],
    *[f"rps{h}_slope20" for h in RPS_HORIZONS],
    "vol_ratio5_rank", "vol_ratio20_rank", "vol_ratio20_rank_mean5", "vol_ratio20_rank_mean20", "vol_ratio20_rank_slope20",
    "vol20_rank", "vol20_rank_mean20", "vol20_low_rank", "vol20_low_rank_mean20", "vol60_rank", "vol60_rank_mean20",
    "ma5_dist_rank", "ma20_dist_rank", "ma60_dist_rank",
    "rps60_minus120", "rps60_minus120_mean20", "rps60_minus120_slope20", "rps20_minus60", "rps20_minus60_mean20",
    "near_high20", "near_high60", "drawdown20", "range20_low", "close_strength", "close_vwap", "amount_rank",
    "limit_touch20_rank", "near_limit20_rank", "limit_close5_rank",
    "market_ret20_median", "market_ret60_median", "market_breadth20", "market_breadth60", "market_ret20_disp", "market_ret60_disp", "market_vol_ratio20_median",
]


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in (2022, 2023, 2024, 2025, 2026):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        model = make_model(year)
        model.fit(train[FEATURES], train["target_quintile"])
        out = test.copy()
        probs = model.predict_proba(out[FEATURES])
        if probs.shape[1] < 5:
            full = np.zeros((len(out), 5), dtype=float)
            for cls_idx, cls_value in enumerate(model.classes_):
                full[:, int(cls_value)] = probs[:, cls_idx]
            probs = full
        out["score_path_ev"] = probs @ np.arange(5, dtype=float)
        out["score_path_topq"] = probs[:, 4]
        out["score_path_spread"] = probs[:, 4] - probs[:, 0]
        out["score_blend_topq_25"] = blend_ranks(out, [("score_path_topq", 0.75), ("base_score", 0.25)])
        out["score_base"] = out["base_score"]
        rows.append(out)
        folds.append({"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())})
        importances.append({name: float(value) for name, value in zip(FEATURES, model.feature_importances_)})
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_model(year: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=280, learning_rate=0.035,
        num_leaves=35, max_depth=6, min_child_samples=360, subsample=0.86,
        colsample_bytree=0.88, reg_alpha=1.0, reg_lambda=7.0,
        random_state=RANDOM_SEED + int(year), n_jobs=6, verbosity=-1,
    )


def blend_ranks(frame: pd.DataFrame, parts: list[tuple[str, float]]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in parts:
        score = score + weight * frame.groupby("session")[col].rank(pct=True, method="first")
    return score


def variants() -> list[tuple[str, str]]:
    return [
        ("base", "score_base"),
        ("path_ev", "score_path_ev"),
        ("path_topq", "score_path_topq"),
        ("path_spread", "score_path_spread"),
        ("blend_topq_25", "score_blend_topq_25"),
    ]


def due_sessions(scored: pd.DataFrame) -> set[str]:
    sessions = sorted(scored["session"].unique())
    return set(sessions[::5])


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    due = due_sessions(scored)
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for scope in ("all", "due5"):
            if scope == "due5" and session not in due:
                continue
            for variant, score_col in variants():
                ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
                for topk in TOPKS:
                    selected = ranked.head(topk)
                    rows.append({
                        "scope": scope, "session": session, "year": int(session[:4]), "variant": variant, "topk": int(topk),
                        "mean_label5_open": float(selected["label5_open"].mean()),
                        "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                        "mean_raw5_open": float(selected["raw5_open"].mean()),
                        "entry_ok": float(selected["entry_ok"].mean()),
                        "exit_ok": float(selected["exit_ok"].mean()),
                    })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (scope, variant, topk), group in detail.groupby(["scope", "variant", "topk"]):
        out[f"{scope}::pool200::{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {str(int(y)): {"mean_label5_open": float(yg["mean_label5_open"].mean()), "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean())} for y, yg in group.groupby("year")},
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    due = due_sessions(scored)
    work = scored[scored["session"].isin(due)]
    for variant, score_col in variants():
        for topk in (20,):
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in work.groupby("session", sort=True)
            }
            out[f"{variant}::pool200::top{topk}::reb5"] = EXP129.replay_open_selection(selections, market, horizon)
    return out


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:80]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(checksum)
    for key in [
        "due5::pool200::base::top20",
        "due5::pool200::path_topq::top20",
        "due5::pool200::blend_topq_25::top20",
        "all::pool200::path_topq::top20",
    ]:
        value = result["label_results"].get(key)
        if value:
            print(f"{key}: label={value['mean_label5_open']:+.6f} exec={value['mean_exec_label5_open']:+.6f}")
    for name, account in result["account_results"].items():
        print(f"{name}: final={account.get('final_multiple', 1.0):.4f} return={account.get('total_return', 0.0):+.4f} mdd={account.get('max_drawdown_period', 0.0):+.4f} avg_count={account.get('avg_selected_count', 0.0):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
