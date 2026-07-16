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


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP140_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-formal-loss-attribution-v1/analyze_qmt_formal_loss_attribution.py"
TOPKS = (10, 20)
BIN_COUNT = 4
SCHEMAS: dict[str, tuple[str, ...]] = {
    "trend_liq_vol": ("ret20_rank", "ret60_rank", "amount_rank", "range20_low_rank"),
    "rotation_position": ("style_rotation_quality", "ret20_rank", "near_high20_rank", "upper_wick_low_rank"),
    "catchup_quality": ("style_lag_catchup", "mom_lag5_rank", "near_high20_rank", "vol20_low_rank"),
    "market_style": ("mkt_breadth20", "mkt_near_high20", "ret20_rank", "amount_rank"),
    "risk_position": ("mkt_breadth20", "mkt_near_high20", "range20_low_rank", "near_high20_rank"),
}
FIXED_01_FEATURES = {
    "ret1_rank", "ret3_rank", "ret5_rank", "ret10_rank", "ret20_rank", "ret60_rank",
    "amount_rank", "amt_ratio20_rank", "vol20_low_rank", "range20_low_rank", "near_high20_rank",
    "close_strength_rank", "upper_wick_low_rank", "price_rank", "style_rotation_quality",
    "style_lag_catchup", "style_leader_confirm", "multi_style_strength", "mom_lag5_rank",
    "mom_rel5_rank", "liq_lag5_rank", "vol_lag5_rank", "mkt_breadth20", "mkt_near_high20",
}


def load_exp140() -> Any:
    spec = importlib.util.spec_from_file_location("exp140", EXP140_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP140_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp140"] = module
    spec.loader.exec_module(module)
    return module


EXP140 = load_exp140()
EXP130 = EXP140.EXP130


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
    panel = EXP130.build_panel(market, args)
    panel = EXP140.add_loss_labels(panel)
    scored, folds, cell_stats = walk_forward_cells(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_robust_cell_soil_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "cell_stats": cell_stats,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "config": {
            "source_experiment": str(EXP140_PATH.relative_to(REPO_ROOT)),
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "bin_count": BIN_COUNT,
            "schemas": SCHEMAS,
            "causality": "Each fold builds cell boundaries and target encodings from rows with year < test year only. Signal features are completed daily bars through T. T+1/T+6 open data are labels/evaluation only.",
            "scope": "QMT retrievable daily OHLCV/amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def walk_forward_cells(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    cell_stats: dict[str, Any] = {}
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year]
        test = work[work["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        out = test.copy()
        fold_stats: dict[str, Any] = {}
        robust_cols: list[str] = []
        for schema, features in SCHEMAS.items():
            train_key, test_key = make_cell_keys(train, test, features)
            maps, stats = fit_cell_maps(train, train_key)
            fold_stats[schema] = stats
            for suffix in ("mean", "robust", "topedge"):
                mapping = maps[suffix]
                col = f"score_{schema}_{suffix}"
                out[col] = pd.Series(test_key).map(mapping).fillna(maps[f"global_{suffix}"]).to_numpy(dtype=float)
                if suffix == "robust":
                    robust_cols.append(col)
        out["score_ensemble_robust"] = average_session_ranks(out, robust_cols)
        out["score_ensemble_mean"] = average_session_ranks(out, [f"score_{schema}_mean" for schema in SCHEMAS])
        out["score_ensemble_topedge"] = average_session_ranks(out, [f"score_{schema}_topedge" for schema in SCHEMAS])
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_exec_mean": float(train["exec_label5_open"].mean()),
            "train_entry_ok": float(train["entry_ok"].mean()),
        })
        cell_stats[str(year)] = fold_stats
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, cell_stats


def make_cell_keys(train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    train_bins: list[np.ndarray] = []
    test_bins: list[np.ndarray] = []
    for feature in features:
        tr = train[feature].to_numpy(dtype=float)
        te = test[feature].to_numpy(dtype=float)
        if feature in FIXED_01_FEATURES:
            train_bins.append(fixed_bins(tr))
            test_bins.append(fixed_bins(te))
        else:
            thresholds = np.nanquantile(tr[np.isfinite(tr)], np.linspace(0, 1, BIN_COUNT + 1)[1:-1])
            train_bins.append(np.searchsorted(thresholds, np.nan_to_num(tr, nan=np.nanmedian(tr)), side="right"))
            test_bins.append(np.searchsorted(thresholds, np.nan_to_num(te, nan=np.nanmedian(tr)), side="right"))
    return combine_bins(train_bins), combine_bins(test_bins)


def fixed_bins(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(values, nan=0.5, posinf=0.5, neginf=0.5)
    return np.floor(np.clip(clean, 0.0, 0.999999) * BIN_COUNT).astype(int)


def combine_bins(bins: list[np.ndarray]) -> np.ndarray:
    key = np.zeros(len(bins[0]), dtype=int)
    mult = 1
    for b in bins:
        key += b.astype(int) * mult
        mult *= BIN_COUNT
    return key


def fit_cell_maps(train: pd.DataFrame, key: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    work = pd.DataFrame({
        "key": key,
        "year": train["year"].to_numpy(dtype=int),
        "exec": train["exec_label5_open"].to_numpy(dtype=float),
        "raw": train["raw5_open"].to_numpy(dtype=float),
        "entry_ok": train["entry_ok"].to_numpy(dtype=float),
        "top_exec": train["top_exec"].to_numpy(dtype=float),
        "bad_exec": train["bad_exec"].to_numpy(dtype=float),
    })
    global_mean = float(work["exec"].mean())
    global_top = float(work["top_exec"].mean())
    global_bad = float(work["bad_exec"].mean())
    agg = work.groupby("key").agg(
        count=("exec", "size"),
        mean_exec=("exec", "mean"),
        mean_raw=("raw", "mean"),
        entry_ok=("entry_ok", "mean"),
        top_exec=("top_exec", "mean"),
        bad_exec=("bad_exec", "mean"),
    )
    annual = work.groupby(["key", "year"])["exec"].mean().reset_index()
    annual_stats = annual.groupby("key").agg(worst_year=("exec", "min"), year_std=("exec", "std"), years=("year", "nunique"))
    agg = agg.join(annual_stats, how="left").fillna({"year_std": 0.0, "years": 0})
    shrink = agg["count"] / (agg["count"] + 1200.0)
    robust_raw = 0.55 * agg["mean_exec"] + 0.35 * agg["worst_year"] - 0.10 * agg["year_std"].fillna(0.0)
    topedge_raw = agg["mean_exec"] + 0.015 * (agg["top_exec"] - global_top) - 0.010 * (agg["bad_exec"] - global_bad)
    maps = {
        "mean": (global_mean + shrink * (agg["mean_exec"] - global_mean)).to_dict(),
        "robust": (global_mean + shrink * (robust_raw - global_mean)).to_dict(),
        "topedge": (global_mean + shrink * (topedge_raw - global_mean)).to_dict(),
        "global_mean": global_mean,
        "global_robust": global_mean,
        "global_topedge": global_mean,
    }
    stats = {
        "cells": int(len(agg)),
        "usable_cells_count_ge_1200": int((agg["count"] >= 1200).sum()),
        "best_mean_exec": float(agg["mean_exec"].max()),
        "best_robust_score": float(robust_raw.max()),
        "median_count": float(agg["count"].median()),
        "global_mean_exec": global_mean,
    }
    return maps, stats


def average_session_ranks(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(0.0, index=frame.index, dtype=float)
    if not cols:
        return out
    for col in cols:
        out += frame.groupby("session")[col].rank(pct=True, method="first") / len(cols)
    return out


def variants() -> list[tuple[str, str]]:
    out = []
    for schema in SCHEMAS:
        out.append((f"{schema}_mean", f"score_{schema}_mean"))
        out.append((f"{schema}_robust", f"score_{schema}_robust"))
        out.append((f"{schema}_topedge", f"score_{schema}_topedge"))
    out.extend([
        ("ensemble_mean", "score_ensemble_mean"),
        ("ensemble_robust", "score_ensemble_robust"),
        ("ensemble_topedge", "score_ensemble_topedge"),
    ])
    return out


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": session,
                    "year": int(str(session)[:4]),
                    "variant": variant,
                    "topk": int(topk),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
                    "top_exec_rate": float(selected["top_exec"].mean()),
                    "bad_exec_rate": float(selected["bad_exec"].mean()),
                    "count": int(len(selected)),
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "top_exec_rate": float(group["top_exec_rate"].mean()),
            "bad_exec_rate": float(group["bad_exec_rate"].mean()),
            "by_year": {
                str(int(y)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "top_exec_rate": float(yg["top_exec_rate"].mean()),
                    "bad_exec_rate": float(yg["bad_exec_rate"].mean()),
                }
                for y, yg in group.groupby("year")
            },
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"])
                for session, group in scored.groupby("session", sort=True)
            }
            out[f"{variant}::top{topk}"] = EXP140.replay_open_selection(selections, market, horizon)
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows = []
    for name, account in accounts.items():
        label = label_map.get(name, {})
        annual = account.get("annual_returns", {})
        rows.append({
            "name": name,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "return_2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "mean_raw5_open": float(label.get("mean_raw5_open", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
            "top_exec_rate": float(label.get("top_exec_rate", 0.0)),
            "bad_exec_rate": float(label.get("bad_exec_rate", 0.0)),
        })
    rows.sort(key=lambda row: row["final_multiple"], reverse=True)
    return rows


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:16]:
        print(
            f"{row['name']}: final={row['final_multiple']:.2f}x 2026={row['return_2026']:+.4f} "
            f"exec={row['mean_exec_label5_open']:+.6f} raw={row['mean_raw5_open']:+.6f} "
            f"entry={row['entry_ok']:.3f} top={row['top_exec_rate']:.3f} bad={row['bad_exec_rate']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
