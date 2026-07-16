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


EXP129 = load_module("exp129_for_rank_flow_transition", EXP129_PATH)

BASE_FEATURES = list(EXP129.FEATURES)
FLOW_FEATURES = [
    "ret5_flow3", "ret5_flow5", "ret5_flow10", "ret5_flow20",
    "ret20_flow3", "ret20_flow5", "ret20_flow10", "ret20_flow20",
    "ret60_flow5", "ret60_flow10", "ret60_flow20",
    "amount_flow3", "amount_flow5", "amount_flow10",
    "amt_ratio_flow3", "amt_ratio_flow5", "amt_ratio_flow10",
    "near_high_flow5", "near_high_flow10", "close_strength_flow5",
    "range_low_flow5", "range_low_flow10", "vwap_support_flow5",
    "ret5_accel_5_10", "ret20_accel_5_20", "amount_accel_5_10",
    "ret5_up_count5", "ret20_up_count5", "ret20_up_count10", "amount_up_count5",
    "amount_leads_price", "price_climb_without_amount_overheat", "steady_rank_climb",
    "low_to_mid_transition", "mid_to_high_transition", "high_rank_persistence",
    "pullback_after_climb_repair", "flow_exhaustion_risk", "flow_exhaustion_avoid",
    "rank_flow_prior", "durable_transition_prior", "flow_minus_exhaustion_prior",
    "mkt_ret5_climb_breadth", "mkt_ret20_climb_breadth", "mkt_mid_to_high_share",
    "mkt_high_rank_fail_share", "mkt_amount_leads_share", "mkt_flow_dispersion",
]
FEATURES = BASE_FEATURES + FLOW_FEATURES


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
    warnings.filterwarnings("ignore", category=FutureWarning, message="The previous implementation of stack is deprecated.*")
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP129.EXP124.load_market(args)
    panel = EXP129.build_panel(market, args)
    panel = add_rank_flow_features(panel, market)
    panel = add_targets(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    stress = stress_diagnostics(accounts)
    result = {
        "status": classify_status(accounts, stress),
        "experiment": "qmt_rank_flow_transition_v1",
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
            "exp154_best_positive_top20_multiple": 1.55,
            "exp155_best_top20_multiple": 1.25,
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
            "flow_feature_count": len(FLOW_FEATURES),
            "causality": "All rank-flow features are computed from completed daily bars through signal date T only. Rank slopes, acceleration, climb counts, and market-flow breadth use T minus historical shifted ranks, never T+1 or later. Labels and replay use scheduled T+1 open entry and T+6 open target exit. Each test year trains only on years strictly before it; 2026 uses 2021-2025 only.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static concept tables, or other message data.",
            "interpretation_guardrail": "This experiment tests whether multi-day cross-sectional rank flow naturally thickens Top20. It does not shrink TopK, hard-filter weak tails, use 2026 labels for training, or route by post-hoc opportunity days.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def rank_frame(frame: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    ranked = frame.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True, ascending=ascending)
    return ranked.fillna(0.5)


def add_rank_flow_features(panel: pd.DataFrame, market: dict[str, Any]) -> pd.DataFrame:
    if panel.empty:
        return panel
    f = market["frames"]
    dates = market["dates"]
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
    amount = f["amount"]
    vwap = f["vwap"]

    ret1 = close.pct_change(fill_method=None)
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amount20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio20 = amount / amount20
    high20 = high.rolling(20, min_periods=10).max()
    range20 = ((high / low) - 1.0).rolling(20, min_periods=10).mean()
    close_strength = (close - low) / (high - low).replace(0, np.nan)
    vwap_support = close / vwap - 1.0

    ranks: dict[str, pd.DataFrame] = {
        "ret1": rank_frame(ret1),
        "ret5": rank_frame(ret5),
        "ret20": rank_frame(ret20),
        "ret60": rank_frame(ret60),
        "amount": rank_frame(np.log1p(amount)),
        "amt_ratio": rank_frame(amt_ratio20),
        "near_high": rank_frame(close / high20),
        "close_strength": rank_frame(close_strength),
        "range_low": rank_frame(range20, ascending=False),
        "vwap_support": rank_frame(vwap_support),
    }
    for frame in ranks.values():
        frame.index = dates

    feats: dict[str, pd.DataFrame] = {}
    for base in ("ret5", "ret20"):
        for lag in (3, 5, 10, 20):
            feats[f"{base}_flow{lag}"] = ranks[base] - ranks[base].shift(lag)
    for base in ("ret60",):
        for lag in (5, 10, 20):
            feats[f"{base}_flow{lag}"] = ranks[base] - ranks[base].shift(lag)
    for base in ("amount", "amt_ratio"):
        for lag in (3, 5, 10):
            feats[f"{base}_flow{lag}"] = ranks[base] - ranks[base].shift(lag)
    for base in ("near_high", "range_low"):
        for lag in (5, 10):
            feats[f"{base}_flow{lag}"] = ranks[base] - ranks[base].shift(lag)
    for base in ("close_strength", "vwap_support"):
        feats[f"{base}_flow5"] = ranks[base] - ranks[base].shift(5)

    feats["ret5_accel_5_10"] = (ranks["ret5"] - ranks["ret5"].shift(5)) - (ranks["ret5"].shift(5) - ranks["ret5"].shift(10))
    feats["ret20_accel_5_20"] = (ranks["ret20"] - ranks["ret20"].shift(5)) - (ranks["ret20"].shift(5) - ranks["ret20"].shift(20))
    feats["amount_accel_5_10"] = (ranks["amount"] - ranks["amount"].shift(5)) - (ranks["amount"].shift(5) - ranks["amount"].shift(10))
    feats["ret5_up_count5"] = (ranks["ret5"].diff() > 0).rolling(5, min_periods=3).mean()
    feats["ret20_up_count5"] = (ranks["ret20"].diff() > 0).rolling(5, min_periods=3).mean()
    feats["ret20_up_count10"] = (ranks["ret20"].diff() > 0).rolling(10, min_periods=5).mean()
    feats["amount_up_count5"] = (ranks["amount"].diff() > 0).rolling(5, min_periods=3).mean()

    amount_leads = (
        0.28 * feats["amount_flow5"]
        + 0.24 * feats["amt_ratio_flow5"]
        + 0.18 * (ranks["amount"] - ranks["ret5"])
        + 0.16 * (feats["amount_flow10"] - feats["ret5_flow10"])
        + 0.14 * feats["amount_accel_5_10"]
    )
    steady_climb = (
        0.20 * feats["ret20_flow10"]
        + 0.18 * feats["ret5_flow5"]
        + 0.16 * feats["ret20_up_count10"]
        + 0.14 * ranks["near_high"]
        + 0.12 * ranks["range_low"]
        + 0.10 * ranks["vwap_support"]
        + 0.10 * (1.0 - ranks["ret1"])
    )
    low_to_mid = pd.DataFrame(
        np.where((ranks["ret20"] >= 0.30) & (ranks["ret20"] <= 0.65), 1.0, 0.0),
        index=ranks["ret20"].index,
        columns=ranks["ret20"].columns,
    ) * (0.55 * positive(feats["ret20_flow10"]) + 0.45 * positive(feats["amount_flow5"]))
    mid_to_high = pd.DataFrame(
        np.where((ranks["ret20"] >= 0.50) & (ranks["ret20"] <= 0.88), 1.0, 0.0),
        index=ranks["ret20"].index,
        columns=ranks["ret20"].columns,
    ) * (0.42 * positive(feats["ret20_flow10"]) + 0.28 * positive(feats["ret5_flow5"]) + 0.30 * ranks["near_high"])
    high_rank_persistence = pd.DataFrame(
        np.where(ranks["ret20"] >= 0.78, 1.0, 0.0),
        index=ranks["ret20"].index,
        columns=ranks["ret20"].columns,
    ) * (0.45 * feats["ret20_up_count10"] + 0.30 * ranks["range_low"] + 0.25 * ranks["close_strength"])
    repair = (
        0.24 * feats["ret20_flow20"]
        + 0.20 * ranks["near_high"]
        + 0.18 * ranks["vwap_support"]
        + 0.16 * ranks["close_strength"]
        + 0.12 * (1.0 - ranks["ret5"])
        + 0.10 * ranks["range_low"]
    )
    exhaustion = (
        0.20 * ranks["ret1"]
        + 0.20 * ranks["ret5"]
        + 0.18 * ranks["amt_ratio"]
        + 0.16 * (1.0 - ranks["range_low"])
        + 0.14 * positive(feats["ret5_accel_5_10"])
        + 0.12 * positive(feats["amount_accel_5_10"])
    )
    price_climb_without_amount_overheat = (
        0.28 * feats["ret20_flow10"]
        + 0.24 * feats["ret5_flow5"]
        + 0.18 * ranks["near_high"]
        + 0.16 * ranks["range_low"]
        + 0.14 * (1.0 - ranks["amt_ratio"])
    )
    prior = 0.24 * steady_climb + 0.20 * mid_to_high + 0.18 * amount_leads + 0.16 * repair + 0.12 * high_rank_persistence + 0.10 * price_climb_without_amount_overheat
    durable = 0.34 * steady_climb + 0.24 * high_rank_persistence + 0.18 * repair + 0.14 * ranks["range_low"] + 0.10 * ranks["vwap_support"]

    feats["amount_leads_price"] = amount_leads
    feats["price_climb_without_amount_overheat"] = price_climb_without_amount_overheat
    feats["steady_rank_climb"] = steady_climb
    feats["low_to_mid_transition"] = low_to_mid
    feats["mid_to_high_transition"] = mid_to_high
    feats["high_rank_persistence"] = high_rank_persistence
    feats["pullback_after_climb_repair"] = repair
    feats["flow_exhaustion_risk"] = exhaustion
    feats["flow_exhaustion_avoid"] = 1.0 - exhaustion
    feats["rank_flow_prior"] = prior
    feats["durable_transition_prior"] = durable
    feats["flow_minus_exhaustion_prior"] = prior - 0.55 * exhaustion

    mkt = pd.DataFrame(index=dates)
    mkt["mkt_ret5_climb_breadth"] = (feats["ret5_flow5"] > 0.08).mean(axis=1)
    mkt["mkt_ret20_climb_breadth"] = (feats["ret20_flow10"] > 0.08).mean(axis=1)
    mkt["mkt_mid_to_high_share"] = ((ranks["ret20"] >= 0.50) & (ranks["ret20"] <= 0.88) & (feats["ret20_flow10"] > 0.08)).mean(axis=1)
    mkt["mkt_high_rank_fail_share"] = ((ranks["ret20"] >= 0.78) & (feats["ret5_flow5"] < -0.08)).mean(axis=1)
    mkt["mkt_amount_leads_share"] = ((ranks["amount"] - ranks["ret5"]) > 0.15).mean(axis=1)
    mkt["mkt_flow_dispersion"] = feats["ret20_flow10"].std(axis=1).fillna(0.0)

    out = panel.copy()
    indexer = pd.MultiIndex.from_frame(out[["session", "instrument"]].astype(str))
    sessions = sorted(out["session"].astype(str).unique())
    for name in FLOW_FEATURES:
        if name.startswith("mkt_"):
            out[name] = out["session"].astype(str).map(mkt[name]).astype(float)
            continue
        series = feats[name].reindex(index=sessions).stack(dropna=False)
        series.index = series.index.set_names(["session", "instrument"])
        out[name] = series.reindex(indexer).to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def positive(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.clip(lower=0.0)


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    ranks = out.groupby("session")["exec_label5_open"].rank(pct=True, method="first")
    out["right_tail_target"] = (ranks >= 0.95).astype(int)
    out["top_quintile_target"] = (ranks >= 0.80).astype(int)
    out["left_tail_target"] = (ranks <= 0.20).astype(int)
    return out


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
        weighted_reg = make_regressor(year, 101, 280)
        tail_cls = make_classifier(year, 211, 260)
        quintile_cls = make_classifier(year, 307, 240)
        left_cls = make_classifier(year, 401, 220)
        flow_weight = (0.70 + 1.20 * train["rank_flow_prior"].to_numpy(dtype=float)).clip(0.70, 2.20)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        weighted_reg.fit(train[FEATURES], train["exec_label5_open"], sample_weight=flow_weight)
        tail_cls.fit(train[FEATURES], train["right_tail_target"])
        quintile_cls.fit(train[FEATURES], train["top_quintile_target"])
        left_cls.fit(train[FEATURES], train["left_tail_target"])

        out = test.copy()
        out["score_return_reg"] = reg.predict(out[FEATURES])
        out["score_weighted_reg"] = weighted_reg.predict(out[FEATURES])
        out["score_right_tail"] = tail_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_top_quintile"] = quintile_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_left_tail_risk"] = left_cls.predict_proba(out[FEATURES])[:, 1]
        out["score_rank_flow_prior"] = out["rank_flow_prior"]
        out["score_amount_leads_price"] = out["amount_leads_price"]
        out["score_steady_rank_climb"] = out["steady_rank_climb"]
        out["score_mid_to_high_transition"] = out["mid_to_high_transition"]
        out["score_flow_minus_exhaustion"] = out["flow_minus_exhaustion_prior"]
        out["score_durable_transition"] = out["durable_transition_prior"]
        out["score_flow_blend"] = blend_ranks(
            out,
            [
                ("score_rank_flow_prior", 0.28), ("score_steady_rank_climb", 0.22),
                ("score_mid_to_high_transition", 0.18), ("score_amount_leads_price", 0.16),
                ("score_flow_minus_exhaustion", 0.16),
            ],
        )
        out["score_ml_flow_blend"] = blend_ranks(
            out,
            [
                ("score_return_reg", 0.28), ("score_right_tail", 0.22),
                ("score_top_quintile", 0.16), ("score_weighted_reg", 0.14),
                ("score_flow_blend", 0.14), ("score_left_tail_risk", -0.06),
            ],
        )
        out["score_flow_risk_adjusted_reg"] = blend_ranks(
            out,
            [
                ("score_return_reg", 0.38), ("score_weighted_reg", 0.24),
                ("score_durable_transition", 0.18), ("score_left_tail_risk", -0.12),
                ("flow_exhaustion_risk", -0.08),
            ],
        )
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "right_tail_rate_train": float(train["right_tail_target"].mean()),
            "top_quintile_rate_train": float(train["top_quintile_target"].mean()),
            "left_tail_rate_train": float(train["left_tail_target"].mean()),
            "mean_flow_weight_train": float(np.mean(flow_weight)),
        })
        imp = {f"reg::{name}": float(value) for name, value in zip(FEATURES, reg.feature_importances_)}
        imp.update({f"weighted_reg::{name}": float(value) for name, value in zip(FEATURES, weighted_reg.feature_importances_)})
        imp.update({f"right_tail::{name}": float(value) for name, value in zip(FEATURES, tail_cls.feature_importances_)})
        imp.update({f"top_quintile::{name}": float(value) for name, value in zip(FEATURES, quintile_cls.feature_importances_)})
        imp.update({f"left_tail::{name}": float(value) for name, value in zip(FEATURES, left_cls.feature_importances_)})
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
        ("rank_flow_prior", "score_rank_flow_prior"),
        ("amount_leads_price", "score_amount_leads_price"),
        ("steady_rank_climb", "score_steady_rank_climb"),
        ("mid_to_high_transition", "score_mid_to_high_transition"),
        ("flow_minus_exhaustion", "score_flow_minus_exhaustion"),
        ("flow_blend", "score_flow_blend"),
        ("return_reg", "score_return_reg"),
        ("weighted_reg", "score_weighted_reg"),
        ("right_tail", "score_right_tail"),
        ("top_quintile", "score_top_quintile"),
        ("flow_risk_adjusted_reg", "score_flow_risk_adjusted_reg"),
        ("ml_flow_blend", "score_ml_flow_blend"),
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
            account["period_curve"] = account_period_curve(selections, market, horizon)
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
