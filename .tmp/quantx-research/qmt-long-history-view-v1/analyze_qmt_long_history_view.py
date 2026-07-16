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
EXP140_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-formal-loss-attribution-v1/analyze_qmt_formal_loss_attribution.py"
FEATURES: list[str]
TOPKS = (10, 20)
TRAIN_STARTS = (2010, 2016, 2021)
RANDOM_SEED = 20260714


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
FEATURES = list(EXP130.FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True, help="panel start; use 2010-01-04 for long-history experiment")
    parser.add_argument("--eval-start", default="2022-01-01")
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
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    result: dict[str, Any] = {
        "status": "diagnostic_complete",
        "experiment": "qmt_long_history_view_v1",
        "panel_rows": int(len(panel)),
        "eval_start": args.eval_start,
        "train_starts": TRAIN_STARTS,
        "runs": {},
        "leaderboard": [],
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
            "features": FEATURES,
            "causality": "For each test year in 2022-2026, each model trains only on rows with train_start <= year < test_year. Features use completed daily bars through signal day T; T+1/T+6 open data are labels/evaluation only.",
            "scope": "QMT retrievable daily OHLCV/amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or message/event data.",
        },
    }
    leaderboard: list[dict[str, Any]] = []
    for train_start in TRAIN_STARTS:
        scored, folds, importances = walk_forward(panel, train_start, int(args.eval_start[:4]), args.min_train_rows)
        detail = evaluate_label(scored)
        accounts = replay_accounts(scored, market, args.horizon)
        run = {
            "scored_rows": int(len(scored)),
            "folds": folds,
            "label_results": summarize_detail(detail),
            "account_results": accounts,
            "feature_importance": summarize_importance(importances),
        }
        key = f"train_start_{train_start}"
        result["runs"][key] = run
        for row in build_leaderboard(accounts, detail):
            row["train_start"] = int(train_start)
            row["run"] = key
            leaderboard.append(row)
    leaderboard.sort(key=lambda row: row["final_multiple"], reverse=True)
    result["leaderboard"] = leaderboard
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def walk_forward(panel: pd.DataFrame, train_start: int, eval_start_year: int, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    for year in sorted(int(y) for y in panel["year"].unique() if int(y) >= eval_start_year):
        train = panel[(panel["year"] >= train_start) & (panel["year"] < year)]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        reg = make_regressor(year, train_start)
        cls = make_classifier(year, train_start)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"].astype(int))
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = positive_proba(cls, out[FEATURES])
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        rows.append(out)
        folds.append({
            "train_start": int(train_start),
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_years": [int(y) for y in sorted(train["year"].unique())],
            "train_exec_mean": float(train["exec_label5_open"].mean()),
            "train_entry_ok": float(train["entry_ok"].mean()),
        })
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"cls::{name}": float(value) for name, value in zip(FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_regressor(year: int, train_start: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=320,
        learning_rate=0.032,
        num_leaves=31,
        max_depth=5,
        min_child_samples=650,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=RANDOM_SEED + year * 10 + train_start,
        n_jobs=6,
        verbosity=-1,
    )


def make_classifier(year: int, train_start: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=650,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=RANDOM_SEED + year * 10 + train_start + 5,
        n_jobs=6,
        verbosity=-1,
    )


def positive_proba(model: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.full(len(x), float(model.classes_[0] == 1), dtype=float)
    pos = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos]


def variants() -> list[tuple[str, str]]:
    return [("reg", "score_reg"), ("cls", "score_cls"), ("blend", "score_blend")]


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
            "by_year": {
                str(int(y)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "top_exec_rate": float(yg["top_exec_rate"].mean()),
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
        })
    rows.sort(key=lambda row: row["final_multiple"], reverse=True)
    return rows


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:60]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:18]:
        print(
            f"{row['run']}::{row['name']}: final={row['final_multiple']:.2f}x "
            f"2026={row['return_2026']:+.4f} exec={row['mean_exec_label5_open']:+.6f} "
            f"raw={row['mean_raw5_open']:+.6f} entry={row['entry_ok']:.3f} top={row['top_exec_rate']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
