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
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("exp130", EXP130_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP130_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp130"] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()
TOPKS = (10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    scored, folds = score_walk_forward(panel)
    labels = evaluate_labels(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_regime_conditioned_path_family_soil_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_labels(labels),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, labels),
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
            "stock_families": list(stock_score_specs().keys()),
            "state_families": list(state_specs().keys()),
            "causality": "Features and scores use completed daily bars through signal date T only. Year-specific regime thresholds are fit using sessions from years strictly before the test year. Labels and replay use pre-scheduled T+1 open entry and T+6 open target exit.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def stock_score_specs() -> dict[str, dict[str, float]]:
    return {
        "momentum_quality": {
            "ret20_rank": 0.24, "ret60_rank": 0.22, "near_high20_rank": 0.16,
            "amount_rank": 0.12, "close_strength_rank": 0.12, "upper_wick_low_rank": 0.08,
            "range20_low_rank": 0.06,
        },
        "quiet_trend": {
            "ret20_rank": 0.24, "ret60_rank": 0.16, "vol20_low_rank": 0.18,
            "range20_low_rank": 0.16, "near_high20_rank": 0.14, "upper_wick_low_rank": 0.12,
        },
        "pullback_in_uptrend": {
            "ret60_rank": 0.22, "ret20_rank": 0.18, "near_high20_rank": 0.16,
            "close_strength_rank": 0.14, "upper_wick_low_rank": 0.10, "range20_low_rank": 0.10,
            "ret3_rank": -0.10,
        },
        "liquid_right_tail": {
            "ret5_rank": 0.18, "ret20_rank": 0.18, "ret60_rank": 0.14,
            "amount_rank": 0.16, "amt_ratio20_rank": 0.12, "near_high20_rank": 0.12,
            "close_strength_rank": 0.10,
        },
        "anti_crowded_trend": {
            "ret20_rank": 0.22, "ret60_rank": 0.18, "near_high20_rank": 0.14,
            "upper_wick_low_rank": 0.12, "range20_low_rank": 0.12, "amount_rank": -0.12,
            "amt_ratio20_rank": -0.10,
        },
        "lag_catchup": {
            "multi_style_strength": 0.24, "mom_lag5_rank": 0.20, "liq_lag5_rank": 0.16,
            "near_high20_rank": 0.14, "close_strength_rank": 0.12, "upper_wick_low_rank": 0.08,
            "ret5_rank": -0.06,
        },
        "leader_confirm": {
            "multi_style_strength": 0.22, "mom_rel5_rank": 0.18, "ret20_rank": 0.18,
            "ret60_rank": 0.14, "amount_rank": 0.12, "near_high20_rank": 0.10,
            "upper_wick_low_rank": 0.06,
        },
        "low_price_elastic": {
            "price_rank": -0.22, "ret20_rank": 0.20, "ret60_rank": 0.16,
            "amount_rank": 0.14, "near_high20_rank": 0.12, "close_strength_rank": 0.10,
            "upper_wick_low_rank": 0.06,
        },
        "risk_resilient_trend": {
            "ret20_rank": 0.18, "ret60_rank": 0.16, "vol20_low_rank": 0.18,
            "range20_low_rank": 0.18, "near_high20_rank": 0.12, "amount_rank": 0.10,
            "upper_wick_low_rank": 0.08,
        },
        "short_term_reversal": {
            "ret1_rank": -0.18, "ret3_rank": -0.20, "ret20_rank": 0.16,
            "ret60_rank": 0.12, "close_strength_rank": 0.16, "range20_low_rank": 0.10,
            "amount_rank": 0.08,
        },
    }


def state_specs() -> dict[str, str]:
    return {
        "all": "all sessions",
        "high_disp": "mkt_disp20 >= q67",
        "low_disp": "mkt_disp20 <= q33",
        "high_amount_disp": "mkt_amount_disp20 >= q67",
        "high_breadth": "mkt_breadth20 >= q67",
        "low_breadth": "mkt_breadth20 <= q33",
        "positive_market": "mkt_ret20_median >= q67",
        "weak_market": "mkt_ret20_median <= q33",
        "near_high_market": "mkt_near_high20 >= q67",
        "high_disp_high_breadth": "mkt_disp20 >= q67 and mkt_breadth20 >= q67",
        "high_disp_weak_market": "mkt_disp20 >= q67 and mkt_ret20_median <= q33",
        "high_disp_positive_market": "mkt_disp20 >= q67 and mkt_ret20_median >= q67",
        "low_disp_high_breadth": "mkt_disp20 <= q33 and mkt_breadth20 >= q67",
        "risk_on_confirmed": "mkt_breadth20 >= q67 and mkt_ret20_median >= q67",
        "risk_off_divergent": "mkt_breadth20 <= q33 and mkt_disp20 >= q67",
    }


def score_walk_forward(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scored_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    panel = panel.copy()
    add_stock_scores(panel)
    session_state = panel.groupby(["session", "year"], as_index=False)[market_state_columns()].first()
    for year in sorted(int(y) for y in panel["year"].unique()):
        train_state = session_state[session_state["year"] < year]
        test = panel[panel["year"] == year].copy()
        if train_state.empty or test.empty:
            continue
        thresholds = fit_state_thresholds(train_state)
        state_flags = apply_state_flags(session_state[session_state["year"] == year], thresholds)
        test = test.merge(state_flags, on="session", how="left")
        for flag in state_specs():
            test[f"state_{flag}"] = test[f"state_{flag}"].fillna(False).astype(bool)
        scored_parts.append(test)
        folds.append({
            "year": int(year),
            "train_sessions": int(train_state["session"].nunique()),
            "test_rows": int(len(test)),
            "test_sessions": int(test["session"].nunique()),
            "state_active_sessions": {name: int(state_flags[f"state_{name}"].sum()) for name in state_specs()},
        })
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()), folds


def market_state_columns() -> list[str]:
    return ["mkt_disp20", "mkt_amount_disp20", "mkt_breadth20", "mkt_ret20_median", "mkt_near_high20"]


def add_stock_scores(panel: pd.DataFrame) -> None:
    for name, weights in stock_score_specs().items():
        score = np.zeros(len(panel), dtype=float)
        for feature, weight in weights.items():
            score += float(weight) * panel[feature].to_numpy(dtype=float)
        panel[f"score_{name}"] = score


def fit_state_thresholds(session_state: pd.DataFrame) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for col in market_state_columns():
        values = session_state[col].to_numpy(dtype=float)
        thresholds[col] = {"q33": float(np.nanquantile(values, 0.33)), "q67": float(np.nanquantile(values, 0.67))}
    return thresholds


def apply_state_flags(session_state: pd.DataFrame, thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = session_state[["session"]].copy()
    s = session_state
    q = thresholds
    out["state_all"] = True
    out["state_high_disp"] = s["mkt_disp20"] >= q["mkt_disp20"]["q67"]
    out["state_low_disp"] = s["mkt_disp20"] <= q["mkt_disp20"]["q33"]
    out["state_high_amount_disp"] = s["mkt_amount_disp20"] >= q["mkt_amount_disp20"]["q67"]
    out["state_high_breadth"] = s["mkt_breadth20"] >= q["mkt_breadth20"]["q67"]
    out["state_low_breadth"] = s["mkt_breadth20"] <= q["mkt_breadth20"]["q33"]
    out["state_positive_market"] = s["mkt_ret20_median"] >= q["mkt_ret20_median"]["q67"]
    out["state_weak_market"] = s["mkt_ret20_median"] <= q["mkt_ret20_median"]["q33"]
    out["state_near_high_market"] = s["mkt_near_high20"] >= q["mkt_near_high20"]["q67"]
    out["state_high_disp_high_breadth"] = out["state_high_disp"] & out["state_high_breadth"]
    out["state_high_disp_weak_market"] = out["state_high_disp"] & out["state_weak_market"]
    out["state_high_disp_positive_market"] = out["state_high_disp"] & out["state_positive_market"]
    out["state_low_disp_high_breadth"] = out["state_low_disp"] & out["state_high_breadth"]
    out["state_risk_on_confirmed"] = out["state_high_breadth"] & out["state_positive_market"]
    out["state_risk_off_divergent"] = out["state_low_breadth"] & out["state_high_disp"]
    return out


def evaluate_labels(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_name in state_specs():
        state_col = f"state_{state_name}"
        active = scored[scored[state_col]]
        if active.empty:
            continue
        for score_name in stock_score_specs():
            score_col = f"score_{score_name}"
            for session, group in active.groupby("session", sort=True):
                ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
                for topk in TOPKS:
                    selected = ranked.head(topk)
                    rows.append({
                        "state": state_name,
                        "score": score_name,
                        "topk": int(topk),
                        "session": session,
                        "year": int(session[:4]),
                        "mean_label5_open": float(selected["label5_open"].mean()),
                        "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                        "mean_raw5_open": float(selected["raw5_open"].mean()),
                        "entry_ok": float(selected["entry_ok"].mean()),
                        "count": int(len(selected)),
                    })
    return pd.DataFrame(rows)


def summarize_labels(labels: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if labels.empty:
        return out
    for (state_name, score_name, topk), group in labels.groupby(["state", "score", "topk"]):
        key = f"{state_name}::{score_name}::top{topk}"
        out[key] = {
            "active_sessions": int(group["session"].nunique()),
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "active_sessions": int(yg["session"].nunique()),
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
                }
                for year, yg in group.groupby("year")
            },
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sessions = sorted(scored["session"].unique())
    groups = {str(session): group for session, group in scored.groupby("session", sort=True)}
    top_cache: dict[tuple[str, int, str], list[str]] = {}
    for session, group in groups.items():
        for score_name in stock_score_specs():
            score_col = f"score_{score_name}"
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                top_cache[(session, topk, score_name)] = list(ranked.head(topk)["instrument"].astype(str))
    for state_name in state_specs():
        state_col = f"state_{state_name}"
        for score_name in stock_score_specs():
            for topk in TOPKS:
                selections: dict[str, list[str]] = {}
                for session in sessions:
                    group = groups[str(session)]
                    if group.empty or not bool(group[state_col].iloc[0]):
                        selections[str(session)] = []
                        continue
                    selections[str(session)] = top_cache[(str(session), topk, score_name)]
                account = EXP130.replay_open_selection(selections, market, horizon)
                active_sessions = sum(1 for names in selections.values() if names)
                account["active_sessions"] = int(active_sessions)
                account["active_ratio"] = float(active_sessions / max(len(selections), 1))
                out[f"{state_name}::{score_name}::top{topk}"] = account
    return out


def build_leaderboard(accounts: dict[str, Any], labels: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = summarize_labels(labels)
    rows = []
    for key, account in accounts.items():
        label = label_map.get(key, {})
        annual = account.get("annual_returns", {})
        rows.append({
            "key": key,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "total_return": float(account.get("total_return", 0.0)),
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "ret2026": float(annual.get("2026", 0.0)),
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "active_ratio": float(account.get("active_ratio", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "active_sessions": int(label.get("active_sessions", 0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows[:80]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:20]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x "
            f"2026={row['ret2026']:+.4f} min_year={row['min_annual_return']:+.4f} "
            f"avg_count={row['avg_selected_count']:.2f} active={row['active_ratio']:.2%} "
            f"exec_label={row['mean_exec_label5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
