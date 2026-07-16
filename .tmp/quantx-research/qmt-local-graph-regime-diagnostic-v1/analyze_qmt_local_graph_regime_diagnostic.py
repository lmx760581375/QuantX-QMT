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
EXP145_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-local-relative-graph-ranker-v1/analyze_qmt_local_relative_graph_ranker.py"
RANDOM_SEED = 20260714
TOPK = 20
TARGET_VARIANTS = (
    "reg",
    "graph_blend_25",
    "graph_blend_50",
    "graph_quiet_catchup",
    "graph_follower",
    "graph_consensus",
)
MARKET_COLS = [
    "mkt_disp20",
    "mkt_amount_disp20",
    "mkt_breadth20",
    "mkt_ret20_median",
    "mkt_near_high20",
]
PORTFOLIO_COLS = [
    "graph_peer_shock_mean",
    "graph_consensus_strength",
    "graph_leader_risk_composite",
    "graph_follower_composite",
    "graph_quiet_catchup_composite",
    "graph_rel_trend_mean",
    "ret1_rank",
    "ret5_rank",
    "ret20_rank",
    "ret60_rank",
    "amt_ratio20_rank",
    "upper_wick_low_rank",
    "close_strength_rank",
    "near_high20_rank",
    "vol20_low_rank",
    "range20_low_rank",
    "multi_style_strength",
    "style_lag_catchup",
    "style_leader_confirm",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP145 = load_module("exp145_for_graph_regime_diagnostic", EXP145_PATH)


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
    parser.add_argument("--min-train-sessions", type=int, default=40)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP145.EXP130.EXP124.load_market(args)
    panel = EXP145.EXP130.build_panel(market, args)
    panel = EXP145.add_local_relative_graph_features(panel)
    scored, folds, importances = EXP145.walk_forward(panel, args.min_train_rows)
    long_sessions, wide_sessions = build_session_frames(scored, market, args.horizon)
    model_features = build_model_features(wide_sessions)
    model_diagnostics = walk_forward_session_models(wide_sessions, model_features, args.min_train_sessions)
    result = {
        "status": classify_status(long_sessions, model_diagnostics),
        "experiment": "qmt_local_graph_regime_diagnostic_v1",
        "source_experiment": "qmt_local_relative_graph_ranker_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "session_rows": int(len(wide_sessions)),
        "folds": folds,
        "source_feature_importance_top30": EXP145.summarize_importance(importances)[:30],
        "variant_summary": summarize_variants(long_sessions),
        "graph_vs_reg_summary": summarize_graph_vs_reg(wide_sessions),
        "bucket_diagnostics": bucket_diagnostics(wide_sessions),
        "regime_contrast": regime_contrast(wide_sessions),
        "session_model_diagnostics": model_diagnostics,
        "model_feature_count": int(len(model_features)),
        "model_features": model_features,
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp40_due5_top20_dev_label": 0.014170,
            "exp40_due5_top20_2026_label": 0.012520,
            "exp145_reg_top20_final_multiple": 2.24,
            "exp145_reg_top20_2026_return": 0.4595,
            "exp145_graph_blend_25_top20_2026_return": 0.6485,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topk": TOPK,
            "target_variants": TARGET_VARIANTS,
            "causality": "Signal date T uses only completed daily bars, cross-sectional ranks, local graph aggregates, and top-k portfolio aggregates available after T close. T+1/T+6 open returns are used only as historical labels or out-of-sample evaluation. Session models train on years strictly before the evaluated year, so 2026 uses 2021-2025 sessions only.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only. No news, announcements, LHB, ETF, northbound, margin financing, static industry/concept table, or message/event data.",
            "interpretation_guardrail": "Top-predicted-session lifts are diagnostics, not deployable hard gates and not counted as satisfying the strategy target.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_session_frames(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_map = {variant: score_col for variant, score_col in EXP145.variants() if variant in TARGET_VARIANTS}
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        group = group.copy()
        wide: dict[str, Any] = {"session": str(session), "year": int(str(session)[:4])}
        for col in MARKET_COLS:
            wide[col] = safe_mean(group[col])
        selected_by_variant: dict[str, pd.DataFrame] = {}
        for variant in TARGET_VARIANTS:
            score_col = score_map[variant]
            selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(TOPK)
            selected_by_variant[variant] = selected
            replay = replay_one_period(str(session), selected["instrument"].astype(str).tolist(), market, horizon)
            row = {
                "session": str(session),
                "year": int(str(session)[:4]),
                "variant": variant,
                "topk": TOPK,
                "period_return": replay["period_return"],
                "selected_count": replay["selected_count"],
                "avg_hold_days": replay["avg_hold_days"],
                "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
                "mean_raw5_open": safe_mean(selected["raw5_open"]),
                "mean_label5_open": safe_mean(selected["label5_open"]),
                "entry_ok": safe_mean(selected["entry_ok"]),
                "exit_ok": safe_mean(selected["exit_ok"]),
            }
            for col in MARKET_COLS + PORTFOLIO_COLS:
                row[col] = safe_mean(selected[col]) if col in selected else np.nan
            long_rows.append(row)
            prefix = clean_variant(variant)
            wide[f"{prefix}_period_return"] = replay["period_return"]
            wide[f"{prefix}_selected_count"] = replay["selected_count"]
            wide[f"{prefix}_avg_hold_days"] = replay["avg_hold_days"]
            wide[f"{prefix}_mean_exec_label5_open"] = row["mean_exec_label5_open"]
            wide[f"{prefix}_mean_raw5_open"] = row["mean_raw5_open"]
            wide[f"{prefix}_entry_ok"] = row["entry_ok"]
            for col in PORTFOLIO_COLS:
                wide[f"{prefix}_{col}"] = row[col]
        add_overlap_features(wide, selected_by_variant)
        if "graph_blend_25_period_return" in wide and "reg_period_return" in wide:
            wide["graph_blend_25_minus_reg_period_return"] = wide["graph_blend_25_period_return"] - wide["reg_period_return"]
            wide["graph_blend_25_positive"] = float(wide["graph_blend_25_period_return"] > 0.0)
            wide["graph_blend_25_beats_reg"] = float(wide["graph_blend_25_minus_reg_period_return"] > 0.0)
        wide_rows.append(wide)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def replay_one_period(session: str, candidates: list[str], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    idx = market["date_index"][session]
    entry_idx = idx + 1
    exit_idx = entry_idx + horizon
    if exit_idx >= len(dates):
        return {"period_return": 0.0, "selected_count": 0, "avg_hold_days": 0.0}
    rets: list[float] = []
    holds: list[int] = []
    for sym in candidates:
        col = symbol_index.get(sym)
        if col is None:
            continue
        entry = float(arr["open"][entry_idx, col])
        preclose = float(arr["close"][idx, col])
        volume = float(arr["volume"][entry_idx, col])
        if not tradable_entry(entry, preclose, volume):
            continue
        sell_idx = exit_idx
        while sell_idx < min(len(dates), exit_idx + 6):
            sell_open = float(arr["open"][sell_idx, col])
            sell_preclose = float(arr["close"][sell_idx - 1, col])
            sell_volume = float(arr["volume"][sell_idx, col])
            if tradable_exit(sell_open, sell_preclose, sell_volume):
                break
            sell_idx += 1
        if sell_idx >= min(len(dates), exit_idx + 6):
            continue
        exit_price = float(arr["open"][sell_idx, col])
        rets.append(exit_price / entry - 1.0 - 0.00154)
        holds.append(sell_idx - entry_idx)
    return {
        "period_return": float(np.mean(rets)) if rets else 0.0,
        "selected_count": int(len(rets)),
        "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
    }


def tradable_entry(entry: float, preclose: float, volume: float) -> bool:
    return bool(
        np.isfinite(entry)
        and entry > 0
        and np.isfinite(preclose)
        and preclose > 0
        and np.isfinite(volume)
        and volume > 0
        and entry / preclose - 1.0 < 0.095
    )


def tradable_exit(open_price: float, preclose: float, volume: float) -> bool:
    return bool(
        np.isfinite(open_price)
        and open_price > 0
        and np.isfinite(preclose)
        and preclose > 0
        and np.isfinite(volume)
        and volume > 0
        and open_price / preclose - 1.0 > -0.095
    )


def add_overlap_features(wide: dict[str, Any], selected_by_variant: dict[str, pd.DataFrame]) -> None:
    reg = set(selected_by_variant.get("reg", pd.DataFrame()).get("instrument", []))
    for variant, selected in selected_by_variant.items():
        if variant == "reg":
            continue
        names = set(selected.get("instrument", []))
        denom = max(1, min(len(reg), len(names)))
        wide[f"{clean_variant(variant)}_overlap_with_reg"] = float(len(reg & names) / denom)


def summarize_variants(long_sessions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, group in long_sessions.groupby("variant"):
        annual = {
            str(int(year)): float(np.prod(1.0 + yg["period_return"].to_numpy(dtype=float)) - 1.0)
            for year, yg in group.groupby("year")
        }
        out[f"{variant}::top{TOPK}"] = {
            "periods": int(len(group)),
            "final_multiple": float(np.prod(1.0 + group["period_return"].to_numpy(dtype=float))),
            "mean_period_return": float(group["period_return"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "positive_session_ratio": float((group["period_return"] > 0).mean()),
            "avg_selected_count": float(group["selected_count"].mean()),
            "avg_hold_days": float(group.loc[group["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (group["avg_hold_days"] > 0).any() else 0.0,
            "annual_returns": annual,
            "all_years_positive": bool(annual and all(v > 0 for v in annual.values())),
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
        }
    return out


def summarize_graph_vs_reg(wide: pd.DataFrame) -> dict[str, Any]:
    delta = wide["graph_blend_25_minus_reg_period_return"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "periods": int(len(wide)),
        "mean_delta_period_return": float(np.mean(delta)),
        "median_delta_period_return": float(np.median(delta)),
        "graph_beats_reg_ratio": float(np.mean(delta > 0.0)),
        "graph_positive_ratio": float(np.mean(wide["graph_blend_25_period_return"].to_numpy(dtype=float) > 0.0)),
        "reg_positive_ratio": float(np.mean(wide["reg_period_return"].to_numpy(dtype=float) > 0.0)),
        "by_year": {},
    }
    for year, group in wide.groupby("year"):
        yd = group["graph_blend_25_minus_reg_period_return"].to_numpy(dtype=float)
        out["by_year"][str(int(year))] = {
            "mean_delta_period_return": float(np.mean(yd)),
            "graph_beats_reg_ratio": float(np.mean(yd > 0.0)),
            "graph_mean_period_return": float(group["graph_blend_25_period_return"].mean()),
            "reg_mean_period_return": float(group["reg_period_return"].mean()),
            "graph_annual_return": float(np.prod(1.0 + group["graph_blend_25_period_return"].to_numpy(dtype=float)) - 1.0),
            "reg_annual_return": float(np.prod(1.0 + group["reg_period_return"].to_numpy(dtype=float)) - 1.0),
        }
    return out


def bucket_diagnostics(wide: pd.DataFrame) -> dict[str, Any]:
    features = MARKET_COLS + [
        "graph_blend_25_graph_peer_shock_mean",
        "graph_blend_25_graph_consensus_strength",
        "graph_blend_25_graph_leader_risk_composite",
        "graph_blend_25_graph_follower_composite",
        "graph_blend_25_graph_quiet_catchup_composite",
        "graph_blend_25_graph_rel_trend_mean",
        "graph_blend_25_ret20_rank",
        "graph_blend_25_amt_ratio20_rank",
        "graph_blend_25_upper_wick_low_rank",
        "graph_blend_25_near_high20_rank",
        "graph_blend_25_overlap_with_reg",
    ]
    out: dict[str, Any] = {}
    for col in features:
        if col not in wide or wide[col].nunique(dropna=True) < 3:
            continue
        bucket = quantile_bucket(wide[col], 3)
        rows: list[dict[str, Any]] = []
        for idx in sorted(bucket.dropna().unique()):
            mask = bucket == idx
            group = wide[mask]
            rows.append({
                "bucket": int(idx),
                "count": int(len(group)),
                "feature_mean": float(group[col].mean()),
                "graph_mean_period_return": float(group["graph_blend_25_period_return"].mean()),
                "reg_mean_period_return": float(group["reg_period_return"].mean()),
                "mean_delta_period_return": float(group["graph_blend_25_minus_reg_period_return"].mean()),
                "graph_positive_ratio": float((group["graph_blend_25_period_return"] > 0).mean()),
                "graph_beats_reg_ratio": float((group["graph_blend_25_minus_reg_period_return"] > 0).mean()),
            })
        out[col] = rows
    return out


def regime_contrast(wide: pd.DataFrame) -> dict[str, Any]:
    weak = wide[wide["year"].isin([2022, 2023])]
    strong = wide[wide["year"].isin([2025, 2026])]
    cols = MARKET_COLS + [
        "graph_blend_25_graph_peer_shock_mean",
        "graph_blend_25_graph_consensus_strength",
        "graph_blend_25_graph_leader_risk_composite",
        "graph_blend_25_graph_follower_composite",
        "graph_blend_25_graph_quiet_catchup_composite",
        "graph_blend_25_graph_rel_trend_mean",
        "graph_blend_25_ret1_rank",
        "graph_blend_25_ret5_rank",
        "graph_blend_25_ret20_rank",
        "graph_blend_25_amt_ratio20_rank",
        "graph_blend_25_upper_wick_low_rank",
        "graph_blend_25_close_strength_rank",
        "graph_blend_25_near_high20_rank",
        "graph_blend_25_overlap_with_reg",
    ]
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in wide:
            continue
        weak_mean = safe_mean(weak[col])
        strong_mean = safe_mean(strong[col])
        rows.append({
            "feature": col,
            "weak_2022_2023_mean": weak_mean,
            "strong_2025_2026_mean": strong_mean,
            "strong_minus_weak": strong_mean - weak_mean,
        })
    rows.sort(key=lambda r: abs(r["strong_minus_weak"]), reverse=True)
    return {
        "weak_years": [2022, 2023],
        "strong_years": [2025, 2026],
        "weak_graph_mean_period_return": safe_mean(weak["graph_blend_25_period_return"]),
        "strong_graph_mean_period_return": safe_mean(strong["graph_blend_25_period_return"]),
        "weak_delta_mean_period_return": safe_mean(weak["graph_blend_25_minus_reg_period_return"]),
        "strong_delta_mean_period_return": safe_mean(strong["graph_blend_25_minus_reg_period_return"]),
        "feature_contrast_top": rows[:25],
    }


def build_model_features(wide: pd.DataFrame) -> list[str]:
    prefixes = ["reg", "graph_blend_25"]
    features = [col for col in MARKET_COLS if col in wide]
    for prefix in prefixes:
        for col in PORTFOLIO_COLS:
            name = f"{prefix}_{col}"
            if name in wide:
                features.append(name)
    for name in [
        "graph_blend_25_overlap_with_reg",
        "graph_blend_50_overlap_with_reg",
        "graph_quiet_catchup_overlap_with_reg",
        "graph_follower_overlap_with_reg",
        "graph_consensus_overlap_with_reg",
    ]:
        if name in wide:
            features.append(name)
    return features


def walk_forward_session_models(wide: pd.DataFrame, features: list[str], min_train_sessions: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    importances: dict[str, list[float]] = {name: [] for name in features}
    for year in sorted(int(y) for y in wide["year"].unique()):
        train = wide[wide["year"] < year]
        test = wide[wide["year"] == year]
        if len(train) < min_train_sessions or test.empty:
            continue
        x_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        year_row: dict[str, Any] = {"year": int(year), "train_sessions": int(len(train)), "test_sessions": int(len(test))}
        cls_targets = {
            "graph_positive": "graph_blend_25_positive",
            "graph_beats_reg": "graph_blend_25_beats_reg",
        }
        for target_name, target_col in cls_targets.items():
            y_train = train[target_col].astype(int)
            y_test = test[target_col].astype(int).to_numpy(dtype=int)
            if y_train.nunique() < 2:
                year_row[f"{target_name}_skipped"] = True
                continue
            clf = make_session_classifier(year, target_name)
            clf.fit(x_train, y_train)
            pred = positive_proba(clf, x_test)
            year_row[f"{target_name}_auc"] = auc_score(y_test, pred)
            year_row[f"{target_name}_target_rate"] = float(np.mean(y_test))
            year_row[f"{target_name}_top_half_target_rate"] = top_fraction_mean(y_test.astype(float), pred, 0.5)
            if target_name == "graph_positive":
                year_row[f"{target_name}_top_half_graph_return"] = top_fraction_mean(test["graph_blend_25_period_return"].to_numpy(dtype=float), pred, 0.5)
                year_row[f"{target_name}_all_graph_return"] = float(test["graph_blend_25_period_return"].mean())
            else:
                year_row[f"{target_name}_top_half_delta"] = top_fraction_mean(test["graph_blend_25_minus_reg_period_return"].to_numpy(dtype=float), pred, 0.5)
                year_row[f"{target_name}_all_delta"] = float(test["graph_blend_25_minus_reg_period_return"].mean())
            for name, value in zip(features, clf.feature_importances_):
                importances[name].append(float(value))
        reg = make_session_regressor(year)
        y_delta = train["graph_blend_25_minus_reg_period_return"]
        reg.fit(x_train, y_delta)
        pred_delta = reg.predict(x_test)
        actual_delta = test["graph_blend_25_minus_reg_period_return"].to_numpy(dtype=float)
        year_row["delta_reg_spearman"] = spearman(actual_delta, pred_delta)
        year_row["delta_reg_top_half_delta"] = top_fraction_mean(actual_delta, pred_delta, 0.5)
        year_row["delta_reg_all_delta"] = float(np.mean(actual_delta))
        for name, value in zip(features, reg.feature_importances_):
            importances[name].append(float(value))
        rows.append(year_row)
    summary = summarize_model_rows(rows)
    importance_rows = [
        {"feature": name, "mean_importance": float(np.mean(values))}
        for name, values in importances.items()
        if values
    ]
    importance_rows.sort(key=lambda r: r["mean_importance"], reverse=True)
    return {"by_year": rows, "summary": summary, "feature_importance_top30": importance_rows[:30]}


def make_session_classifier(year: int, salt: str) -> LGBMClassifier:
    offset = sum(ord(ch) for ch in salt)
    return LGBMClassifier(
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=9,
        max_depth=3,
        min_child_samples=16,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=RANDOM_SEED + year + offset,
        n_jobs=4,
        verbosity=-1,
    )


def make_session_regressor(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=150,
        learning_rate=0.035,
        num_leaves=9,
        max_depth=3,
        min_child_samples=16,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=RANDOM_SEED + year + 997,
        n_jobs=4,
        verbosity=-1,
    )


def summarize_model_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"folds": int(len(rows))}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and key not in {"year", "train_sessions", "test_sessions"}})
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
            out[f"min_{key}"] = float(np.min(vals))
    return out


def classify_status(long_sessions: pd.DataFrame, model_diagnostics: dict[str, Any]) -> str:
    variant_summary = summarize_variants(long_sessions)
    graph = variant_summary.get(f"graph_blend_25::top{TOPK}", {})
    final_multiple = float(graph.get("final_multiple", 0.0))
    all_years_positive = bool(graph.get("all_years_positive", False))
    summary = model_diagnostics.get("summary", {})
    mean_auc = float(summary.get("mean_graph_beats_reg_auc", 0.0))
    min_auc = float(summary.get("min_graph_beats_reg_auc", 0.0))
    if final_multiple >= 8.7749 and all_years_positive and mean_auc >= 0.58 and min_auc >= 0.50:
        return "diagnostic_has_candidate_regime_signal_but_not_deployable_gate"
    return "rejected_as_graph_regime_not_causally_identified"


def positive_proba(model: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.full(len(x), float(model.classes_[0] == 1), dtype=float)
    pos = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos]


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(ar) == 0.0 or np.nanstd(br) == 0.0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def top_fraction_mean(values: np.ndarray, scores: np.ndarray, frac: float) -> float:
    values = np.asarray(values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = max(1, int(np.ceil(len(values) * frac)))
    order = np.argsort(scores)[::-1]
    return float(np.mean(values[order[:n]]))


def quantile_bucket(series: pd.Series, q: int) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else 0.0


def clean_variant(variant: str) -> str:
    return variant.replace("::", "_").replace("-", "_")


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for key, data in sorted(result["variant_summary"].items(), key=lambda item: item[1].get("final_multiple", 0.0), reverse=True):
        annual = data.get("annual_returns", {})
        print(
            f"{key}: final={data.get('final_multiple', 1.0):.2f}x "
            f"2026={annual.get('2026', float('nan')):+.4f} "
            f"min_year={data.get('min_annual_return', 0.0):+.4f} "
            f"mean_period={data.get('mean_period_return', 0.0):+.6f}"
        )
    print("graph_vs_reg", json.dumps(result["graph_vs_reg_summary"], ensure_ascii=False, sort_keys=True)[:1200])
    print("model_summary", json.dumps(result["session_model_diagnostics"].get("summary", {}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
