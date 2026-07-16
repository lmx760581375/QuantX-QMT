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
EXP129_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-supply-demand-rebalance-v1/analyze_qmt_supply_demand_rebalance.py"
RANDOM_SEED = 20260714
TOPKS = (10, 15, 20, 30)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP129 = load_module("exp129_for_executable_transition_ranker", EXP129_PATH)

TRANSITION_FEATURES = [
    "pressure_repair_prior", "non_extreme_strength", "executable_transition_prior",
    "crowding_risk_prior", "left_tail_risk_prior", "transition_minus_risk_prior",
    "trend_continuity_rank", "short_pullback_repair_rank", "supply_repair_balance",
    "quality_without_chase", "market_risk_adjusted_transition", "entry_fail_visible_prior",
]
FEATURES = list(EXP129.FEATURES) + TRANSITION_FEATURES


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
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP129.EXP124.load_market(args)
    panel = EXP129.build_panel(market, args)
    panel = add_transition_features(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    stress = stress_diagnostics(accounts)
    result = {
        "status": classify_status(accounts, stress),
        "experiment": "qmt_executable_transition_ranker_v1",
        "source_experiment": "qmt_supply_demand_rebalance_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail, stress),
        "stress_diagnostics": stress,
        "feature_importance": summarize_importance(importances),
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp148_cluster_rank_top10_multiple": 8.0478,
            "exp148_cluster_rank_top10_2026": 0.1763,
            "exp150_rank_repair_35_top10_multiple": 9.45,
            "exp150_rank_repair_35_top10_2026": 0.0883,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "feature_count": len(FEATURES),
            "causality": "All visible features use completed daily bars through signal date T only. Future T+1/T+6 open data is used only as training labels, entry/exit feasibility labels, and out-of-sample account replay. Each test year trains only on rows from years strictly before that year; 2026 uses 2021-2025 only.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static concept tables, or other message data.",
            "interpretation_guardrail": "This experiment does not shrink TopK to satisfy requirements and does not hard-filter the 6-10 or 11-20 tail. Crowding and executable-risk terms are model inputs or learned risk heads, then evaluated at Top10/15/20/30 under the same open replay.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_transition_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    pressure_repair = (
        0.20 * out["pressure_release_rank"]
        + 0.18 * out["post_spike_hold_rank"]
        + 0.16 * out["post_spike_drawdown_rank"]
        + 0.14 * out["vwap_support_rank"]
        + 0.12 * out["close_strength_rank"]
        + 0.10 * out["upper_wick_low_rank"]
        + 0.10 * out["range20_low_rank"]
    )
    trend_continuity = (
        0.22 * out["ret3_rank"]
        + 0.22 * out["ret5_rank"]
        + 0.20 * out["ret10_rank"]
        + 0.16 * out["ret20_rank"]
        + 0.12 * out["near_high20_rank"]
        + 0.08 * out["vwap_support_rank"]
    )
    short_pullback_repair = (
        0.24 * (1.0 - out["ret1_rank"])
        + 0.20 * out["ret5_rank"]
        + 0.18 * out["ret20_rank"]
        + 0.16 * out["lower_wick_rank"]
        + 0.12 * out["close_strength_rank"]
        + 0.10 * out["vwap_support_rank"]
    )
    crowding = (
        0.22 * out["ret1_rank"]
        + 0.20 * out["ret3_rank"]
        + 0.18 * out["amt_ratio20_rank"]
        + 0.16 * out["near_high20_rank"]
        + 0.14 * (1.0 - out["upper_wick_low_rank"])
        + 0.10 * out["break20_rank"]
    )
    left_tail = (
        0.20 * (1.0 - out["post_spike_drawdown_rank"])
        + 0.18 * (1.0 - out["drawdown20_rank"])
        + 0.16 * (1.0 - out["range20_low_rank"])
        + 0.14 * (1.0 - out["vwap_support_rank"])
        + 0.12 * (1.0 - out["close_strength_rank"])
        + 0.10 * (1.0 - out["down_volume_absorb_rank"])
        + 0.10 * (1.0 - out["mkt_breadth20"].clip(0.0, 1.0))
    )
    non_extreme = (
        0.24 * out["upper_wick_low_rank"]
        + 0.22 * (1.0 - np.abs(out["ret1_rank"] - 0.55).clip(0.0, 1.0))
        + 0.18 * (1.0 - out["amt_ratio20_rank"])
        + 0.14 * out["vwap_support_rank"]
        + 0.12 * out["close_strength_rank"]
        + 0.10 * out["range20_low_rank"]
    )
    entry_fail_visible = (
        0.30 * crowding
        + 0.22 * out["ret1_rank"]
        + 0.18 * out["amt_ratio20_rank"]
        + 0.16 * out["break20_rank"]
        + 0.14 * (1.0 - out["upper_wick_low_rank"])
    )
    executable_transition = (
        0.26 * pressure_repair
        + 0.22 * trend_continuity
        + 0.18 * short_pullback_repair
        + 0.16 * non_extreme
        + 0.12 * out["reconfirm_breakout"]
        + 0.06 * out["mkt_breadth20"].clip(0.0, 1.0)
    )
    supply_repair_balance = 0.46 * out["supply_absorption"] + 0.34 * out["reconfirm_breakout"] + 0.20 * pressure_repair
    quality_without_chase = 0.55 * executable_transition + 0.25 * non_extreme - 0.20 * crowding
    market_adjusted = executable_transition * (0.65 + 0.70 * out["mkt_breadth20"].clip(0.0, 1.0)) - 0.25 * out["mkt_disp20"].clip(0.0, 0.20)
    out["pressure_repair_prior"] = pressure_repair.astype(float)
    out["non_extreme_strength"] = non_extreme.astype(float)
    out["executable_transition_prior"] = executable_transition.astype(float)
    out["crowding_risk_prior"] = crowding.astype(float)
    out["left_tail_risk_prior"] = left_tail.astype(float)
    out["transition_minus_risk_prior"] = (executable_transition - 0.45 * crowding - 0.55 * left_tail).astype(float)
    out["trend_continuity_rank"] = trend_continuity.astype(float)
    out["short_pullback_repair_rank"] = short_pullback_repair.astype(float)
    out["supply_repair_balance"] = supply_repair_balance.astype(float)
    out["quality_without_chase"] = quality_without_chase.astype(float)
    out["market_risk_adjusted_transition"] = market_adjusted.astype(float)
    out["entry_fail_visible_prior"] = entry_fail_visible.astype(float)
    out["right_tail_target"] = (out.groupby("session")["exec_label5_open"].rank(pct=True, method="first") >= 0.95).astype(int)
    out["left_tail_target"] = (out.groupby("session")["exec_label5_open"].rank(pct=True, method="first") <= 0.20).astype(int)
    out["entry_fail_target"] = (out["entry_ok"] < 0.5).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        reg = make_regressor(year, 0, 320)
        tail_cls = make_classifier(year, 101, 260)
        left_cls = make_classifier(year, 211, 220)
        entry_cls = make_classifier(year, 307, 180)
        weighted_reg = make_regressor(year, 401, 280)
        visible_weight = (0.65 + 1.45 * train["executable_transition_prior"].to_numpy(dtype=float)).clip(0.65, 2.10)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        tail_cls.fit(train[FEATURES], train["right_tail_target"])
        left_cls.fit(train[FEATURES], train["left_tail_target"])
        entry_cls.fit(train[FEATURES], train["entry_fail_target"])
        weighted_reg.fit(train[FEATURES], train["exec_label5_open"], sample_weight=visible_weight)

        out = test.copy()
        out["score_return_reg"] = reg.predict(out[FEATURES])
        out["score_weighted_reg"] = weighted_reg.predict(out[FEATURES])
        out["score_right_tail"] = tail_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_left_tail_risk"] = left_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_entry_fail_risk"] = entry_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_transition_prior"] = out["executable_transition_prior"]
        out["score_quality_without_chase"] = out["quality_without_chase"]
        out["score_transition_minus_risk"] = out["transition_minus_risk_prior"]
        out["score_executable_blend"] = blend_ranks(
            out,
            [
                ("score_return_reg", 0.30), ("score_right_tail", 0.24), ("score_weighted_reg", 0.18),
                ("score_transition_prior", 0.14), ("score_left_tail_risk", -0.08), ("score_entry_fail_risk", -0.06),
            ],
        )
        out["score_risk_adjusted_reg"] = blend_ranks(
            out,
            [("score_return_reg", 0.44), ("score_weighted_reg", 0.26), ("score_left_tail_risk", -0.18), ("score_entry_fail_risk", -0.12)],
        )
        out["score_tail_transition"] = blend_ranks(
            out,
            [("score_right_tail", 0.42), ("score_transition_prior", 0.28), ("score_quality_without_chase", 0.18), ("score_left_tail_risk", -0.12)],
        )
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "right_tail_rate_train": float(train["right_tail_target"].mean()),
            "left_tail_rate_train": float(train["left_tail_target"].mean()),
            "entry_fail_rate_train": float(train["entry_fail_target"].mean()),
            "mean_visible_weight_train": float(np.mean(visible_weight)),
        })
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"weighted_reg::{name}": float(value) for name, value in zip(FEATURES, weighted_reg.feature_importances_)})
        imp.update({f"right_tail::{name}": float(value) for name, value in zip(FEATURES, tail_cls.feature_importances_)})
        imp.update({f"left_tail::{name}": float(value) for name, value in zip(FEATURES, left_cls.feature_importances_)})
        imp.update({f"entry_fail::{name}": float(value) for name, value in zip(FEATURES, entry_cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def make_regressor(year: int, offset: int, estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=estimators, learning_rate=0.032, num_leaves=35, max_depth=6,
        min_child_samples=520, subsample=0.85, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def make_classifier(year: int, offset: int, estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=estimators, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=500, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def blend_ranks(frame: pd.DataFrame, parts: list[tuple[str, float]]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in parts:
        rank = frame.groupby("session")[col].rank(pct=True, method="first")
        if weight >= 0:
            score = score + weight * rank
        else:
            score = score + abs(weight) * (1.0 - rank)
    return score


def variants() -> list[tuple[str, str]]:
    return [
        ("return_reg", "score_return_reg"),
        ("weighted_reg", "score_weighted_reg"),
        ("right_tail", "score_right_tail"),
        ("transition_prior", "score_transition_prior"),
        ("quality_without_chase", "score_quality_without_chase"),
        ("transition_minus_risk", "score_transition_minus_risk"),
        ("risk_adjusted_reg", "score_risk_adjusted_reg"),
        ("tail_transition", "score_tail_transition"),
        ("executable_blend", "score_executable_blend"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": str(session),
                    "year": int(str(session)[:4]),
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
        out[f"{variant}::top{topk}"] = {
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


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in grouped.items()
            }
            account = EXP129.replay_open_selection(selections, market, horizon)
            account["period_curve"] = account.get("period_curve", account_period_curve(selections, market, horizon))
            out[f"{variant}::top{topk}"] = account
    return out


def account_period_curve(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows: list[dict[str, Any]] = []
    nav = 1.0
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets: list[float] = []
        holds: list[int] = []
        for sym in candidates:
            col = symbol_index.get(sym)
            if col is None:
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry / preclose - 1.0 >= 0.095:
                continue
            sell_idx = exit_idx
            while sell_idx < min(len(dates), exit_idx + 6):
                sell_open = float(arr["open"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if np.isfinite(sell_open) and sell_open > 0 and np.isfinite(sell_preclose) and sell_preclose > 0 and np.isfinite(sell_vol) and sell_vol > 0 and sell_open / sell_preclose - 1.0 > -0.095:
                    break
                sell_idx += 1
            if sell_idx >= min(len(dates), exit_idx + 6):
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        period_return = float(np.mean(rets)) if rets else 0.0
        nav *= 1.0 + period_return
        rows.append({
            "session": str(session),
            "year": int(str(session)[:4]),
            "period_return": period_return,
            "selected_count": int(len(rets)),
            "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
            "nav": float(nav),
        })
    return rows


def stress_diagnostics(accounts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, account in accounts.items():
        curve = account.get("period_curve", [])
        returns = np.asarray([row["period_return"] for row in curve], dtype=float)
        if returns.size == 0:
            continue
        rows: dict[str, Any] = {}
        for n in (1, 3, 5):
            if returns.size <= n:
                rows[f"remove_best_{n}_final_multiple"] = 1.0
                continue
            keep = np.ones(returns.size, dtype=bool)
            keep[np.argsort(returns)[-n:]] = False
            rows[f"remove_best_{n}_final_multiple"] = float(np.prod(1.0 + returns[keep]))
        rows["best_period_return"] = float(np.max(returns))
        rows["worst_period_return"] = float(np.min(returns))
        rows["positive_period_ratio"] = float(np.mean(returns > 0))
        out[key] = rows
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame, stress: dict[str, Any]) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows: list[dict[str, Any]] = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
        stress_row = stress.get(key, {})
        rows.append({
            "key": key,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "ret2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "mean_raw5_open": float(label.get("mean_raw5_open", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
            "remove_best_3_final_multiple": float(stress_row.get("remove_best_3_final_multiple", np.nan)),
            "positive_period_ratio": float(stress_row.get("positive_period_ratio", np.nan)),
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
    return [{"feature": name, "mean_importance": value} for name, value in rows[:100]]


def classify_status(accounts: dict[str, Any], stress: dict[str, Any]) -> str:
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        robust_after_best3 = stress.get(key, {}).get("remove_best_3_final_multiple", 0.0)
        if (
            account.get("final_multiple", 0.0) >= 20.0
            and annual.get("2026", -1.0) > 0.0
            and account.get("all_years_positive", False)
            and account.get("avg_selected_count", 0.0) > 5.0
            and 4.5 <= account.get("avg_hold_days", 0.0) <= 11.0
            and robust_after_best3 >= 10.0
        ):
            return "candidate_needs_deeper_audit_before_merge"
    return "rejected_before_formal_candidate"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"][:18]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f} "
            f"rm_best3={row['remove_best_3_final_multiple']:.2f}x"
        )


if __name__ == "__main__":
    raise SystemExit(main())
