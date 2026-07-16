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
EXP134_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-adaptive-neighbor-leaf-memory-v1/analyze_qmt_adaptive_neighbor_leaf_memory.py"
TOPK = 20
MIN_SLEEVE_WEIGHT = 0.03


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module("exp130_for_causal_sleeve_budget", EXP130_PATH)
EXP134 = load_module("exp134_for_causal_sleeve_budget", EXP134_PATH)


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
    style_scored, style_folds, style_importance = EXP130.walk_forward(panel, args.min_train_rows)
    memory_scored, memory_folds, memory_importance = EXP134.walk_forward_memory(panel, args.min_train_rows)
    scored = merge_scores(style_scored, memory_scored)
    add_composite_scores(scored)
    sleeve_defs = sleeves()
    detail = evaluate_label(scored, sleeve_defs)
    ledger = build_sleeve_ledger(scored, market, args.horizon, sleeve_defs)
    accounts = build_accounts(ledger, sleeve_defs)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_causal_sleeve_risk_budget_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "sleeves": [name for name, _ in sleeve_defs],
        "style_folds": style_folds,
        "memory_folds": memory_folds,
        "label_results": summarize_detail(detail),
        "ledger_summary": summarize_ledger(ledger),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts),
        "feature_importance_heads": {
            "style": EXP130.summarize_importance(style_importance)[:12],
            "leaf": EXP134.summarize_importance(memory_importance)[:12],
        },
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp40_due5_top20_dev_label": 0.014170,
            "exp40_due5_top20_2026_label": 0.012520,
        },
        "config": {
            "source_style_experiment": str(EXP130_PATH.relative_to(REPO_ROOT)),
            "source_leaf_experiment": str(EXP134_PATH.relative_to(REPO_ROOT)),
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topk_per_sleeve": TOPK,
            "min_sleeve_weight": MIN_SLEEVE_WEIGHT,
            "causality": "Each stock score is yearly out-of-sample: test year models train only on rows from years strictly before that test year. Account allocation at signal date T uses only sleeve ledger rows whose formal exit_session <= T. T+1 open and later open prices are used only for realized replay and completed historical ledger rows, never for same-day scoring or weight formation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only. No news, announcements, LHB, ETF, northbound, financing, or other message/event data are used.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def merge_scores(style_scored: pd.DataFrame, memory_scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["session", "instrument"]
    base_cols = keys + ["year", "raw5_open", "label5_open", "exec_label5_open", "entry_ok", "exit_ok"]
    style_cols = keys + ["score_reg", "score_cls", "score_blend", "score_style_lag_catchup", "score_style_leader_confirm", "score_style_rotation_quality"]
    memory_cols = keys + ["score_leaf_mean", "score_leaf_toprate", "score_leaf_blend", "score_memory_blend"]
    left = style_scored[base_cols + [c for c in style_cols if c not in base_cols]].copy()
    right = memory_scored[memory_cols].copy()
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("merged scored panel is empty")
    return merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_composite_scores(scored: pd.DataFrame) -> None:
    for out, col in {
        "r_style_reg": "score_reg",
        "r_style_blend": "score_blend",
        "r_style_lag": "score_style_lag_catchup",
        "r_style_leader": "score_style_leader_confirm",
        "r_leaf_mean": "score_leaf_mean",
        "r_leaf_blend": "score_leaf_blend",
        "r_memory_blend": "score_memory_blend",
    }.items():
        scored[out] = scored.groupby("session")[col].rank(pct=True, method="first")
    scored["score_leaf_reg_25"] = 0.75 * scored["r_style_reg"] + 0.25 * scored["r_leaf_mean"]
    scored["score_leaf_reg_50"] = 0.50 * scored["r_style_reg"] + 0.50 * scored["r_leaf_mean"]
    scored["score_leaf_blend_25"] = 0.75 * scored["r_style_blend"] + 0.25 * scored["r_leaf_mean"]
    scored["score_lag_leaf_35"] = 0.65 * scored["r_style_lag"] + 0.35 * scored["r_leaf_mean"]
    scored["score_leader_leaf_35"] = 0.65 * scored["r_style_leader"] + 0.35 * scored["r_leaf_mean"]


def sleeves() -> list[tuple[str, str]]:
    return [
        ("style_reg", "score_reg"),
        ("style_blend", "score_blend"),
        ("style_lag_catchup", "score_style_lag_catchup"),
        ("style_leader_confirm", "score_style_leader_confirm"),
        ("leaf_mean", "score_leaf_mean"),
        ("leaf_blend", "score_leaf_blend"),
        ("memory_blend", "score_memory_blend"),
        ("leaf_reg_25", "score_leaf_reg_25"),
        ("leaf_reg_50", "score_leaf_reg_50"),
        ("leaf_blend_25", "score_leaf_blend_25"),
        ("lag_leaf_35", "score_lag_leaf_35"),
        ("leader_leaf_35", "score_leader_leaf_35"),
    ]


def evaluate_label(scored: pd.DataFrame, sleeve_defs: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for sleeve, score_col in sleeve_defs:
            selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(TOPK)
            rows.append({
                "session": str(session),
                "year": int(str(session)[:4]),
                "sleeve": sleeve,
                "topk": TOPK,
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
    for sleeve, group in detail.groupby("sleeve"):
        out[f"{sleeve}::top{TOPK}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
                }
                for year, yg in group.groupby("year")
            },
        }
    return out


def build_sleeve_ledger(scored: pd.DataFrame, market: dict[str, Any], horizon: int, sleeve_defs: list[tuple[str, str]]) -> pd.DataFrame:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        session = str(session)
        idx = market["date_index"][session]
        entry_idx = idx + 1
        target_exit_idx = entry_idx + horizon
        if target_exit_idx >= len(dates):
            continue
        for sleeve, score_col in sleeve_defs:
            candidates = list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(TOPK)["instrument"].astype(str))
            rets: list[float] = []
            holds: list[int] = []
            filled_symbols: list[str] = []
            rejects = {"candidate_total": 0, "entry_reject": 0, "exit_reject": 0}
            exit_indices: list[int] = []
            for sym in candidates:
                rejects["candidate_total"] += 1
                col = symbol_index.get(sym)
                if col is None:
                    rejects["entry_reject"] += 1
                    continue
                entry = float(arr["open"][entry_idx, col])
                preclose = float(arr["close"][idx, col])
                vol = float(arr["volume"][entry_idx, col])
                if not is_buyable(entry, preclose, vol):
                    rejects["entry_reject"] += 1
                    continue
                sell_idx = target_exit_idx
                while sell_idx < min(len(dates), target_exit_idx + 6):
                    sell_open = float(arr["open"][sell_idx, col])
                    sell_preclose = float(arr["close"][sell_idx - 1, col])
                    sell_vol = float(arr["volume"][sell_idx, col])
                    if is_sellable(sell_open, sell_preclose, sell_vol):
                        break
                    sell_idx += 1
                if sell_idx >= min(len(dates), target_exit_idx + 6):
                    rejects["exit_reject"] += 1
                    continue
                exit_ = float(arr["open"][sell_idx, col])
                rets.append(exit_ / entry - 1.0 - 0.00154)
                holds.append(sell_idx - entry_idx)
                exit_indices.append(sell_idx)
                filled_symbols.append(sym)
            exit_idx = max(exit_indices) if exit_indices else target_exit_idx
            rows.append({
                "session": session,
                "entry_session": dates[entry_idx],
                "target_exit_session": dates[target_exit_idx],
                "exit_session": dates[exit_idx],
                "year": int(session[:4]),
                "sleeve": sleeve,
                "period_return": float(np.mean(rets)) if rets else 0.0,
                "selected_count": int(len(rets)),
                "candidate_count": int(len(candidates)),
                "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
                "entry_rejects": int(rejects["entry_reject"]),
                "exit_rejects": int(rejects["exit_reject"]),
                "reject_rate": float((rejects["entry_reject"] + rejects["exit_reject"]) / max(1, rejects["candidate_total"])),
                "filled_symbols": filled_symbols,
            })
    return pd.DataFrame(rows)


def is_buyable(entry: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(entry) and entry > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and entry / preclose - 1.0 < 0.095)


def is_sellable(open_price: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(open_price) and open_price > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and open_price / preclose - 1.0 > -0.095)


def summarize_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sleeve, group in ledger.groupby("sleeve"):
        out[sleeve] = {
            "periods": int(len(group)),
            "mean_period_return": float(group["period_return"].mean()),
            "mean_selected_count": float(group["selected_count"].mean()),
            "mean_reject_rate": float(group["reject_rate"].mean()),
            "mean_hold_days": float(group.loc[group["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (group["avg_hold_days"] > 0).any() else 0.0,
            "by_year_return_mean": {str(int(y)): float(yg["period_return"].mean()) for y, yg in group.groupby("year")},
        }
    return out


def build_accounts(ledger: pd.DataFrame, sleeve_defs: list[tuple[str, str]]) -> dict[str, Any]:
    sleeve_names = [name for name, _ in sleeve_defs]
    accounts: dict[str, Any] = {}
    for sleeve in sleeve_names:
        weights = {session: {name: (1.0 if name == sleeve else 0.0) for name in sleeve_names} for session in sorted(ledger["session"].unique())}
        accounts[f"single::{sleeve}"] = replay_weighted_ledger(ledger, weights)
    accounts["fixed_equal::all_sleeves"] = replay_weighted_ledger(ledger, fixed_equal_weights(ledger, sleeve_names))
    accounts["causal_online::ret60_ret120_drawdown_reject"] = replay_weighted_ledger(ledger, causal_online_weights(ledger, sleeve_names))
    accounts["causal_year_start::prior_completed_years"] = replay_weighted_ledger(ledger, causal_year_start_weights(ledger, sleeve_names))
    return accounts


def fixed_equal_weights(ledger: pd.DataFrame, sleeve_names: list[str]) -> dict[str, dict[str, float]]:
    w = 1.0 / len(sleeve_names)
    return {session: {name: w for name in sleeve_names} for session in sorted(ledger["session"].unique())}


def causal_online_weights(ledger: pd.DataFrame, sleeve_names: list[str]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for session in sorted(ledger["session"].unique()):
        history = ledger[ledger["exit_session"] <= session]
        weights[session] = score_to_weights(score_sleeves(history, sleeve_names, min_periods=12), sleeve_names)
    return weights


def causal_year_start_weights(ledger: pd.DataFrame, sleeve_names: list[str]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    year_cache: dict[int, dict[str, float]] = {}
    for session in sorted(ledger["session"].unique()):
        year = int(str(session)[:4])
        if year not in year_cache:
            history = ledger[ledger["exit_session"] < f"{year}-01-01"]
            year_cache[year] = score_to_weights(score_sleeves(history, sleeve_names, min_periods=20), sleeve_names)
        weights[session] = year_cache[year]
    return weights


def score_sleeves(history: pd.DataFrame, sleeve_names: list[str], min_periods: int) -> dict[str, float] | None:
    if history.empty:
        return None
    scores: dict[str, float] = {}
    enough = False
    for sleeve in sleeve_names:
        group = history[history["sleeve"] == sleeve].sort_values("session")
        if len(group) < min_periods:
            scores[sleeve] = 0.0
            continue
        enough = True
        recent60 = group.tail(60)
        recent120 = group.tail(120)
        returns = recent60["period_return"].to_numpy(dtype=float)
        nav = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(nav) if len(nav) else np.asarray([1.0])
        drawdown = float(np.min(nav / peak - 1.0)) if len(nav) else 0.0
        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) else 0.0
        reject = float(recent60["reject_rate"].mean())
        selected = float(recent60["selected_count"].mean())
        scores[sleeve] = (
            0.65 * float(recent60["period_return"].mean())
            + 0.35 * float(recent120["period_return"].mean())
            - 0.35 * downside_std
            + 0.08 * drawdown
            - 0.015 * reject
            + 0.0005 * min(selected, TOPK)
        )
    return scores if enough else None


def score_to_weights(scores: dict[str, float] | None, sleeve_names: list[str]) -> dict[str, float]:
    if scores is None:
        w = 1.0 / len(sleeve_names)
        return {name: w for name in sleeve_names}
    values = np.asarray([scores.get(name, 0.0) for name in sleeve_names], dtype=float)
    if not np.isfinite(values).all() or np.nanstd(values) < 1e-9:
        w = 1.0 / len(sleeve_names)
        return {name: w for name in sleeve_names}
    z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-9)
    raw = np.exp(np.clip(z, -2.0, 2.0))
    raw = raw / raw.sum()
    floor = MIN_SLEEVE_WEIGHT
    adjusted = floor + (1.0 - floor * len(sleeve_names)) * raw
    adjusted = adjusted / adjusted.sum()
    return {name: float(weight) for name, weight in zip(sleeve_names, adjusted)}


def replay_weighted_ledger(ledger: pd.DataFrame, weights_by_session: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for session in sorted(weights_by_session):
        group = ledger[ledger["session"] == session]
        weights = weights_by_session[session]
        period_return = 0.0
        weighted_selected = 0.0
        weighted_hold = 0.0
        positive_hold_weight = 0.0
        active_weights: dict[str, float] = {}
        distinct: set[str] = set()
        for _, row in group.iterrows():
            sleeve = str(row["sleeve"])
            weight = float(weights.get(sleeve, 0.0))
            if weight <= 0:
                continue
            period_return += weight * float(row["period_return"])
            weighted_selected += weight * float(row["selected_count"])
            if float(row["avg_hold_days"]) > 0:
                weighted_hold += weight * float(row["avg_hold_days"])
                positive_hold_weight += weight
            if weight > 1e-6:
                active_weights[sleeve] = weight
                distinct.update(str(sym) for sym in row["filled_symbols"])
        rows.append({
            "session": session,
            "year": int(str(session)[:4]),
            "period_return": float(period_return),
            "weighted_selected_count": float(weighted_selected),
            "distinct_selected_count": int(len(distinct)),
            "weighted_avg_hold_days": float(weighted_hold / positive_hold_weight) if positive_hold_weight > 0 else 0.0,
            "active_sleeves": active_weights,
        })
    return summarize_account_rows(rows)


def summarize_account_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"final_multiple": 1.0, "total_return": 0.0, "annual_returns": {}, "periods": 0}
    nav = 1.0
    curve: list[dict[str, Any]] = []
    for row in rows:
        nav *= 1.0 + float(row["period_return"])
        curve.append({**row, "nav": nav})
    df = pd.DataFrame(rows)
    annual = {str(int(y)): float(np.prod(1.0 + g["period_return"].to_numpy(dtype=float)) - 1.0) for y, g in df.groupby("year")}
    navs = np.asarray([row["nav"] for row in curve], dtype=float)
    peak = np.maximum.accumulate(navs)
    avg_hold = float(df.loc[df["weighted_avg_hold_days"] > 0, "weighted_avg_hold_days"].mean()) if (df["weighted_avg_hold_days"] > 0).any() else 0.0
    return {
        "final_multiple": float(nav),
        "total_return": float(nav - 1.0),
        "annual_returns": annual,
        "all_years_positive": bool(all(v > 0 for v in annual.values())) if annual else False,
        "max_drawdown_period": float(np.min(navs / peak - 1.0)),
        "avg_weighted_selected_count": float(df["weighted_selected_count"].mean()),
        "avg_distinct_selected_count": float(df["distinct_selected_count"].mean()),
        "avg_hold_days": avg_hold,
        "periods": int(len(rows)),
        "period_return_mean": float(df["period_return"].mean()),
        "period_return_std": float(df["period_return"].std(ddof=0)),
        "positive_period_ratio": float((df["period_return"] > 0).mean()),
        "weight_summary": summarize_weight_rows(rows),
    }


def summarize_weight_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket: dict[str, list[float]] = {}
    for row in rows:
        for sleeve, weight in row["active_sleeves"].items():
            bucket.setdefault(sleeve, []).append(float(weight))
    return {sleeve: {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))} for sleeve, values in sorted(bucket.items())}


def build_leaderboard(accounts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, account in accounts.items():
        annual = account.get("annual_returns", {})
        rows.append({
            "name": name,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "return_2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_distinct_selected_count": float(account.get("avg_distinct_selected_count", 0.0)),
            "avg_weighted_selected_count": float(account.get("avg_weighted_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "positive_period_ratio": float(account.get("positive_period_ratio", 0.0)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["return_2026"]), reverse=True)
    return rows


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:18]:
        print(
            f"{row['name']}: final={row['final_multiple']:.2f}x 2026={row['return_2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} mdd={row['max_drawdown_period']:+.4f} "
            f"distinct={row['avg_distinct_selected_count']:.2f} hold={row['avg_hold_days']:.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
