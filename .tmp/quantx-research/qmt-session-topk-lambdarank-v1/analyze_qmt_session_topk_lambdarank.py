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
from lightgbm import LGBMRanker


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"
TOPKS = (10, 15, 20, 30)
RANDOM_SEED = 20260714


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("exp130_for_session_rank", EXP130_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP130_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp130_for_session_rank"] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()
FEATURES = list(EXP130.FEATURES)


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
    panel = add_rank_labels(panel)
    scored, folds, importances = walk_forward_rankers(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_session_topk_lambdarank_v1",
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
            "rank_objectives": ["q5", "q10", "top5", "top10_focus"],
            "causality": "Features use completed daily bars through signal date T. Each test year uses only rows from years strictly before that test year. Ranker query groups are signal dates; labels use T+1 open to T+6 open only for completed training samples and for evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_rank_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    pct = out.groupby("session")["exec_label5_open"].rank(pct=True, method="first")
    out["label_q5"] = np.floor(np.clip(pct.to_numpy(dtype=float), 0.0, 0.999999) * 5).astype(int)
    out["label_q10"] = np.floor(np.clip(pct.to_numpy(dtype=float), 0.0, 0.999999) * 10).astype(int)
    out["label_top5"] = (pct >= 0.95).astype(int)
    out["label_top10_focus"] = 0
    out.loc[pct >= 0.90, "label_top10_focus"] = 4
    out.loc[(pct >= 0.80) & (pct < 0.90), "label_top10_focus"] = 3
    out.loc[(pct >= 0.60) & (pct < 0.80), "label_top10_focus"] = 2
    out.loc[(pct >= 0.40) & (pct < 0.60), "label_top10_focus"] = 1
    return out


def rank_objectives() -> list[tuple[str, str, list[int]]]:
    return [
        ("rank_q5", "label_q5", [0, 1, 3, 7, 15]),
        ("rank_q10", "label_q10", [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ("rank_top5", "label_top5", [0, 20]),
        ("rank_top10_focus", "label_top10_focus", [0, 1, 3, 8, 20]),
    ]


def walk_forward_rankers(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year].sort_values(["session", "instrument"])
        test = work[work["year"] == year].sort_values(["session", "instrument"])
        if len(train) < min_train_rows or test.empty:
            continue
        train_x = train[FEATURES]
        test_x = test[FEATURES]
        groups = train.groupby("session", sort=False).size().astype(int).tolist()
        out = test.copy()
        fold_imp: dict[str, float] = {}
        for objective_name, label_col, label_gain in rank_objectives():
            ranker = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=260,
                learning_rate=0.035,
                num_leaves=31,
                max_depth=5,
                min_child_samples=450,
                subsample=0.85,
                colsample_bytree=0.90,
                reg_alpha=1.0,
                reg_lambda=7.0,
                random_state=RANDOM_SEED + int(year) + len(objective_name),
                n_jobs=6,
                verbosity=-1,
                label_gain=label_gain,
            )
            ranker.fit(train_x, train[label_col].to_numpy(dtype=int), group=groups)
            out[f"score_{objective_name}"] = ranker.predict(test_x)
            fold_imp.update({f"{objective_name}::{name}": float(value) for name, value in zip(FEATURES, ranker.feature_importances_)})
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_groups": int(len(groups)),
        })
        importances.append(fold_imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [(name, f"score_{name}") for name, _, _ in rank_objectives()]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
                    "mean_label5_open": float(selected["label5_open"].mean()),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
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
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
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
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:16]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"exec_label={row['mean_exec_label5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
