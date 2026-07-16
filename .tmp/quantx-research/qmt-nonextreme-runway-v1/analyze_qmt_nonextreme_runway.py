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
FEATURES = list(EXP140.BASE_FEATURES)


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
    panel = add_runway_labels(panel, market)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_nonextreme_runway_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail),
        "feature_importance": EXP140.summarize_importance(importances),
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
            "causality": "Features use completed T daily bars only. Runway labels use pre-scheduled T+1 open entry and future open path only as historical/test labels. Each test year trains strictly on rows with year < test year.",
            "scope": "QMT retrievable daily OHLCV/amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_runway_labels(panel: pd.DataFrame, market: dict[str, Any]) -> pd.DataFrame:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    arr = market["arrays"]
    dates = market["dates"]
    out_blocks: list[pd.DataFrame] = []
    for session, group in panel.groupby("session", sort=True):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        cols = np.asarray([symbol_index.get(sym, -1) for sym in group["instrument"].astype(str)], dtype=int)
        valid_col = cols >= 0
        entry = np.full(len(group), np.nan, dtype=float)
        if entry_idx < len(dates):
            entry[valid_col] = arr["open"][entry_idx, cols[valid_col]]
        returns: dict[int, np.ndarray] = {}
        for hold in (2, 3, 5, 8):
            exit_idx = entry_idx + hold
            ret = np.full(len(group), np.nan, dtype=float)
            if exit_idx < len(dates):
                exit_open = arr["open"][exit_idx, cols[valid_col]]
                ret[valid_col] = exit_open / entry[valid_col] - 1.0
            returns[hold] = ret
        path = np.vstack([returns[2], returns[3], returns[5]])
        min_path = np.nanmin(path, axis=0)
        mean_path = np.nanmean(path, axis=0)
        runway_raw = 0.35 * returns[5] + 0.25 * returns[8] + 0.25 * mean_path + 0.15 * min_path
        entry_ok = group["entry_ok"].to_numpy(dtype=float) > 0.5
        exit_ok = group["exit_ok"].to_numpy(dtype=float) > 0.5
        runway_exec = np.where(entry_ok & exit_ok & np.isfinite(runway_raw), runway_raw - 0.00154, -0.08)
        block = group.copy()
        block["ret2_open"] = np.nan_to_num(returns[2], nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["ret3_open"] = np.nan_to_num(returns[3], nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["ret8_open"] = np.nan_to_num(returns[8], nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["runway_min_open"] = np.nan_to_num(min_path, nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["runway_mean_open"] = np.nan_to_num(mean_path, nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["runway_raw"] = np.nan_to_num(runway_raw, nan=-0.08, posinf=-0.08, neginf=-0.08)
        block["runway_exec"] = runway_exec
        block["smooth_winner"] = ((block["runway_exec"].rank(pct=True, method="first") >= 0.90) & (block["runway_min_open"] > -0.03) & (block["entry_ok"] > 0.5)).astype(int)
        block["runway_bad"] = (block["runway_exec"] <= block["runway_exec"].quantile(0.20)).astype(int)
        out_blocks.append(block)
    return pd.concat(out_blocks, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.5)


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    for year in sorted(int(y) for y in panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        runway_reg = EXP140.make_regressor(year, 21)
        exec_reg = EXP140.make_regressor(year, 22)
        smooth_cls = EXP140.make_classifier(year, 23)
        bad_cls = EXP140.make_classifier(year, 24)
        runway_reg.fit(train[FEATURES], train["runway_exec"])
        exec_reg.fit(train[FEATURES], train["exec_label5_open"])
        smooth_cls.fit(train[FEATURES], train["smooth_winner"].astype(int))
        bad_cls.fit(train[FEATURES], train["runway_bad"].astype(int))
        out = test.copy()
        out["score_runway"] = runway_reg.predict(out[FEATURES])
        out["score_exec"] = exec_reg.predict(out[FEATURES])
        out["score_smooth"] = EXP140.prob_positive(smooth_cls, out[FEATURES])
        out["score_bad"] = EXP140.prob_positive(bad_cls, out[FEATURES])
        out["score_runway_smooth"] = EXP140.blend_session_ranks(out, [("score_runway", 0.45), ("score_smooth", 0.35), ("score_bad", -0.20)])
        out["score_nonextreme"] = EXP140.blend_session_ranks(out, [("score_runway", 0.35), ("score_exec", 0.25), ("score_smooth", 0.25), ("score_bad", -0.15)])
        out["score_mid_strength"] = 0.28 * out["style_rotation_quality"] + 0.20 * out["style_lag_catchup"] + 0.16 * out["range20_low_rank"] + 0.14 * out["upper_wick_low_rank"] + 0.12 * out["ret20_rank"] + 0.10 * out["amt_ratio20_rank"]
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_smooth_rate": float(train["smooth_winner"].mean()),
            "train_runway_bad_rate": float(train["runway_bad"].mean()),
            "train_runway_exec_mean": float(train["runway_exec"].mean()),
        })
        imp: dict[str, float] = {}
        for prefix, model in [("runway", runway_reg), ("exec", exec_reg), ("smooth", smooth_cls), ("bad", bad_cls)]:
            imp.update({f"{prefix}::{name}": float(value) for name, value in zip(FEATURES, model.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def variants() -> list[tuple[str, str]]:
    return [
        ("runway", "score_runway"),
        ("exec", "score_exec"),
        ("smooth", "score_smooth"),
        ("runway_smooth", "score_runway_smooth"),
        ("nonextreme", "score_nonextreme"),
        ("mid_strength", "score_mid_strength"),
    ]


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
                    "mean_runway_exec": float(selected["runway_exec"].mean()),
                    "mean_runway_min": float(selected["runway_min_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
                    "smooth_winner_rate": float(selected["smooth_winner"].mean()),
                    "runway_bad_rate": float(selected["runway_bad"].mean()),
                    "count": int(len(selected)),
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_runway_exec": float(group["mean_runway_exec"].mean()),
            "mean_runway_min": float(group["mean_runway_min"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "smooth_winner_rate": float(group["smooth_winner_rate"].mean()),
            "runway_bad_rate": float(group["runway_bad_rate"].mean()),
            "by_year": {
                str(int(y)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_runway_exec": float(yg["mean_runway_exec"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "smooth_winner_rate": float(yg["smooth_winner_rate"].mean()),
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
            "mean_runway_exec": float(label.get("mean_runway_exec", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
            "smooth_winner_rate": float(label.get("smooth_winner_rate", 0.0)),
        })
    rows.sort(key=lambda row: row["final_multiple"], reverse=True)
    return rows


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:12]:
        print(
            f"{row['name']}: final={row['final_multiple']:.2f}x 2026={row['return_2026']:+.4f} "
            f"exec={row['mean_exec_label5_open']:+.6f} runway={row['mean_runway_exec']:+.6f} "
            f"entry={row['entry_ok']:.3f} smooth={row['smooth_winner_rate']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
