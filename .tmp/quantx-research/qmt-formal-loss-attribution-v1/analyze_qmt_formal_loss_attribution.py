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
TOPKS = (10, 20)
RANDOM_SEED = 20260714


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("exp130", EXP130_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP130_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp130"] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()
BASE_FEATURES = list(EXP130.FEATURES)


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
    panel = add_loss_labels(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    attribution = diagnose_loss_attribution(scored)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_formal_loss_attribution_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "loss_attribution": attribution,
        "leaderboard": build_leaderboard(accounts, detail),
        "feature_importance": summarize_importance(importances),
        "config": {
            "source_experiment": str(EXP130_PATH.relative_to(REPO_ROOT)),
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "features": BASE_FEATURES,
            "causality": "All model features are inherited from Exp130 and use completed daily bars through signal date T only. Raw/exec/failure labels use T+1 open to T+6 open only for training completed historical folds and for test-year evaluation. Each test year trains only on rows with year strictly smaller than the test year.",
            "scope": "QMT retrievable daily OHLCV/amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or other message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_loss_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["top_raw"] = (out.groupby("session")["raw5_open"].rank(pct=True, method="first") >= 0.95).astype(int)
    out["top_exec"] = (out.groupby("session")["exec_label5_open"].rank(pct=True, method="first") >= 0.95).astype(int)
    out["bad_exec"] = (out["exec_label5_open"] <= out.groupby("session")["exec_label5_open"].transform("quantile", 0.20)).astype(int)
    out["formal_loss"] = out["raw5_open"] - out["exec_label5_open"]
    out["entry_fail"] = (out["entry_ok"] < 0.5).astype(int)
    out["exit_fail"] = (out["exit_ok"] < 0.5).astype(int)
    out["raw_exec_inversion"] = ((out["top_raw"] == 1) & (out["top_exec"] == 0)).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


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
        raw_reg = make_regressor(year, 1)
        exec_reg = make_regressor(year, 2)
        loss_reg = make_regressor(year, 3)
        entry_cls = make_classifier(year, 4)
        top_exec_cls = make_classifier(year, 5)
        bad_exec_cls = make_classifier(year, 6)

        raw_reg.fit(train[BASE_FEATURES], train["raw5_open"])
        exec_reg.fit(train[BASE_FEATURES], train["exec_label5_open"])
        loss_reg.fit(train[BASE_FEATURES], train["formal_loss"])
        entry_cls.fit(train[BASE_FEATURES], train["entry_ok"].astype(int))
        top_exec_cls.fit(train[BASE_FEATURES], train["top_exec"].astype(int))
        bad_exec_cls.fit(train[BASE_FEATURES], train["bad_exec"].astype(int))

        out = test.copy()
        out["score_raw"] = raw_reg.predict(out[BASE_FEATURES])
        out["score_exec"] = exec_reg.predict(out[BASE_FEATURES])
        out["score_loss"] = loss_reg.predict(out[BASE_FEATURES])
        out["score_entry_ok"] = prob_positive(entry_cls, out[BASE_FEATURES])
        out["score_top_exec"] = prob_positive(top_exec_cls, out[BASE_FEATURES])
        out["score_bad_exec"] = prob_positive(bad_exec_cls, out[BASE_FEATURES])
        out["score_raw_minus_loss"] = out["score_raw"] - out["score_loss"]
        out["score_exec_tradeable"] = blend_session_ranks(out, [("score_exec", 0.45), ("score_top_exec", 0.25), ("score_entry_ok", 0.20), ("score_bad_exec", -0.10)])
        out["score_substitute"] = blend_session_ranks(out, [("score_raw", 0.35), ("score_exec", 0.30), ("score_entry_ok", 0.20), ("score_loss", -0.15)])
        out["score_formal_avoid"] = blend_session_ranks(out, [("score_entry_ok", 0.35), ("score_bad_exec", -0.30), ("score_loss", -0.20), ("score_exec", 0.15)])
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "train_entry_ok": float(train["entry_ok"].mean()),
            "train_top_exec_rate": float(train["top_exec"].mean()),
            "train_bad_exec_rate": float(train["bad_exec"].mean()),
        })
        imp: dict[str, float] = {}
        for prefix, model in [("raw", raw_reg), ("exec", exec_reg), ("loss", loss_reg), ("entry", entry_cls), ("top_exec", top_exec_cls), ("bad_exec", bad_exec_cls)]:
            imp.update({f"{prefix}::{name}": float(value) for name, value in zip(BASE_FEATURES, model.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    scored = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return scored, folds, importances


def make_regressor(year: int, offset: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=360,
        learning_rate=0.032,
        num_leaves=31,
        max_depth=5,
        min_child_samples=450,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=7.0,
        random_state=RANDOM_SEED + year * 10 + offset,
        n_jobs=6,
        verbosity=-1,
    )


def make_classifier(year: int, offset: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=450,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=7.0,
        random_state=RANDOM_SEED + year * 10 + offset,
        n_jobs=6,
        verbosity=-1,
    )


def prob_positive(model: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.full(len(x), float(model.classes_[0] == 1), dtype=float)
    pos = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos]


def blend_session_ranks(frame: pd.DataFrame, terms: list[tuple[str, float]]) -> pd.Series:
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for col, weight in terms:
        out += weight * frame.groupby("session")[col].rank(pct=True, method="first")
    return out


def variants() -> list[tuple[str, str]]:
    return [
        ("raw", "score_raw"),
        ("raw_minus_loss", "score_raw_minus_loss"),
        ("exec", "score_exec"),
        ("top_exec", "score_top_exec"),
        ("exec_tradeable", "score_exec_tradeable"),
        ("substitute", "score_substitute"),
        ("formal_avoid", "score_formal_avoid"),
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
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "mean_label5_open": float(selected["label5_open"].mean()),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_formal_loss": float(selected["formal_loss"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
                    "exit_ok": float(selected["exit_ok"].mean()),
                    "bad_exec_rate": float(selected["bad_exec"].mean()),
                    "top_raw_rate": float(selected["top_raw"].mean()),
                    "top_exec_rate": float(selected["top_exec"].mean()),
                    "count": int(len(selected)),
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_formal_loss": float(group["mean_formal_loss"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "bad_exec_rate": float(group["bad_exec_rate"].mean()),
            "top_raw_rate": float(group["top_raw_rate"].mean()),
            "top_exec_rate": float(group["top_exec_rate"].mean()),
            "by_year": {
                str(int(y)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "mean_formal_loss": float(yg["mean_formal_loss"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
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
            out[f"{variant}::top{topk}"] = replay_open_selection(selections, market, horizon)
    return out


def replay_open_selection(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows: list[dict[str, Any]] = []
    rejects = {"candidate_total": 0, "selected_total": 0, "entry_limit_up_or_missing": 0, "exit_limit_down_or_missing": 0, "empty_periods": 0}
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets: list[float] = []
        holds: list[int] = []
        for sym in candidates:
            rejects["candidate_total"] += 1
            col = symbol_index.get(sym)
            if col is None:
                rejects["entry_limit_up_or_missing"] += 1
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not is_entry_ok(entry, preclose, vol):
                rejects["entry_limit_up_or_missing"] += 1
                continue
            sell_idx = exit_idx
            sell_stop = min(len(dates), exit_idx + 6)
            while sell_idx < sell_stop:
                sell_open = float(arr["open"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if is_exit_ok(sell_open, sell_preclose, sell_vol):
                    break
                sell_idx += 1
            if sell_idx >= sell_stop:
                rejects["exit_limit_down_or_missing"] += 1
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        period_return = float(np.mean(rets)) if rets else 0.0
        if not rets:
            rejects["empty_periods"] += 1
        rejects["selected_total"] += len(rets)
        rows.append({
            "session": session,
            "year": int(str(session)[:4]),
            "period_return": period_return,
            "selected_count": int(len(rets)),
            "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
        })
    if not rows:
        return {"final_multiple": 1.0, "total_return": 0.0, "rejects": rejects}
    nav = 1.0
    navs = []
    for row in rows:
        nav *= 1.0 + row["period_return"]
        navs.append(nav)
    df = pd.DataFrame(rows)
    annual = {str(int(y)): float(np.prod(1.0 + g["period_return"].to_numpy(dtype=float)) - 1.0) for y, g in df.groupby("year")}
    nav_arr = np.asarray(navs, dtype=float)
    peak = np.maximum.accumulate(nav_arr)
    return {
        "periods": int(len(rows)),
        "final_multiple": float(nav),
        "total_return": float(nav - 1.0),
        "annual_returns": annual,
        "all_years_positive": bool(all(v > 0 for v in annual.values())),
        "max_drawdown_period": float(np.min(nav_arr / peak - 1.0)),
        "avg_selected_count": float(df["selected_count"].mean()),
        "avg_hold_days": float(df.loc[df["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (df["avg_hold_days"] > 0).any() else 0.0,
        "rejects": rejects,
    }


def is_entry_ok(entry: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(entry) and entry > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and entry / preclose - 1.0 < 0.095)


def is_exit_ok(exit_open: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(exit_open) and exit_open > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and exit_open / preclose - 1.0 > -0.095)


def diagnose_loss_attribution(scored: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, score_col in [("predicted_raw_tail", "score_raw"), ("predicted_exec_tail", "score_exec"), ("predicted_substitute", "score_substitute")]:
        rows = []
        for session, group in scored.groupby("session", sort=True):
            top = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(20)
            rows.append({
                "session": session,
                "year": int(str(session)[:4]),
                "raw": float(top["raw5_open"].mean()),
                "exec": float(top["exec_label5_open"].mean()),
                "loss": float(top["formal_loss"].mean()),
                "entry_ok": float(top["entry_ok"].mean()),
                "exit_ok": float(top["exit_ok"].mean()),
                "top_raw_rate": float(top["top_raw"].mean()),
                "top_exec_rate": float(top["top_exec"].mean()),
                "bad_exec_rate": float(top["bad_exec"].mean()),
            })
        df = pd.DataFrame(rows)
        out[variant] = {
            "mean_raw5_open": float(df["raw"].mean()),
            "mean_exec_label5_open": float(df["exec"].mean()),
            "mean_formal_loss": float(df["loss"].mean()),
            "entry_ok": float(df["entry_ok"].mean()),
            "exit_ok": float(df["exit_ok"].mean()),
            "top_raw_rate": float(df["top_raw_rate"].mean()),
            "top_exec_rate": float(df["top_exec_rate"].mean()),
            "bad_exec_rate": float(df["bad_exec_rate"].mean()),
            "by_year": {
                str(int(y)): {
                    "mean_raw5_open": float(yg["raw"].mean()),
                    "mean_exec_label5_open": float(yg["exec"].mean()),
                    "mean_formal_loss": float(yg["loss"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "bad_exec_rate": float(yg["bad_exec_rate"].mean()),
                }
                for y, yg in df.groupby("year")
            },
        }
    out["raw_vs_exec_rank_corr"] = rank_corr_by_year(scored, "score_raw", "score_exec")
    out["raw_vs_loss_rank_corr"] = rank_corr_by_year(scored, "score_raw", "score_loss")
    return out


def rank_corr_by_year(frame: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        session_corr = []
        for _, sg in group.groupby("session"):
            if len(sg) < 10:
                continue
            corr = sg[a].rank().corr(sg[b].rank())
            if np.isfinite(corr):
                session_corr.append(float(corr))
        out[str(int(year))] = float(np.mean(session_corr)) if session_corr else 0.0
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
            "total_return": float(account.get("total_return", 0.0)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "return_2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
            "bad_exec_rate": float(label.get("bad_exec_rate", 0.0)),
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
    return [{"feature": name, "mean_importance": value} for name, value in rows[:80]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:14]:
        print(
            f"{row['name']}: final={row['final_multiple']:.2f}x "
            f"2026={row['return_2026']:+.4f} exec={row['mean_exec_label5_open']:+.6f} "
            f"entry={row['entry_ok']:.3f} bad={row['bad_exec_rate']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
