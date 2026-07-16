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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module("exp130_for_local_relative_graph", EXP130_PATH)
BASE_FEATURES = list(EXP130.FEATURES)

GRAPH_SCHEMAS: dict[str, list[str]] = {
    "trend": ["ret20_rank", "ret60_rank", "amount_rank"],
    "flow": ["amt_ratio20_rank", "close_strength_rank", "upper_wick_low_rank"],
    "position": ["near_high20_rank", "ret5_rank", "vol20_low_rank"],
    "style": ["multi_style_strength", "style_lag_catchup", "style_leader_confirm"],
}

PEER_VALUE_COLS = [
    "ret1_rank", "ret5_rank", "ret20_rank", "ret60_rank", "amount_rank", "amt_ratio20_rank",
    "near_high20_rank", "close_strength_rank", "upper_wick_low_rank", "multi_style_strength",
]

GRAPH_FEATURES = [
    f"{schema}_{name}"
    for schema in GRAPH_SCHEMAS
    for name in [
        "cell_size_rank", "peer_ret1", "peer_ret5", "peer_ret20", "peer_ret60", "peer_amount",
        "peer_amt_ratio", "peer_near_high", "peer_close_strength", "peer_upper_wick_low",
        "peer_style_strength", "rel_ret1", "rel_ret5", "rel_ret20", "rel_ret60",
        "lag_ret5", "lag_ret20", "lag_ret60", "peer_shock_ret1", "peer_shock_ret5",
        "peer_volume_shock", "peer_breakout", "peer_close_strong", "local_follower_setup",
        "local_leader_risk", "local_quiet_catchup", "local_consensus_strength",
    ]
]

FEATURES = BASE_FEATURES + GRAPH_FEATURES + [
    "graph_follower_composite", "graph_leader_risk_composite", "graph_quiet_catchup_composite",
    "graph_consensus_strength", "graph_peer_shock_mean", "graph_rel_trend_mean",
]


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
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    panel = add_local_relative_graph_features(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_local_relative_graph_ranker_v1",
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
            "topks": TOPKS,
            "graph_schemas": GRAPH_SCHEMAS,
            "feature_count": len(FEATURES),
            "causality": "For each signal date T, local graph cells and peer aggregates use only same-day cross-sectional ranks and completed daily bars through T. Labels and replay use T+1/T+6 open only for historical training labels and evaluation. Each test year trains only on rows from years strictly before that test year.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_local_relative_graph_features(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    frames: list[pd.DataFrame] = []
    for _, group in panel.groupby("session", sort=True):
        g = group.copy()
        for schema, cols in GRAPH_SCHEMAS.items():
            add_schema_features(g, schema, cols)
        add_composite_graph_features(g)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def add_schema_features(g: pd.DataFrame, schema: str, cols: list[str]) -> None:
    bins = []
    for col in cols:
        values = g[col].to_numpy(dtype=float)
        bins.append(np.floor(np.clip(values, 0.0, 0.999999) * 4).astype(np.int16))
    cell = np.zeros(len(g), dtype=np.int32)
    for b in bins:
        cell = cell * 4 + b
    temp = pd.DataFrame({"cell": cell}, index=g.index)
    for col in PEER_VALUE_COLS:
        temp[col] = g[col].to_numpy(dtype=float)
    temp["shock_ret1"] = (g["ret1_rank"].to_numpy(dtype=float) >= 0.90).astype(float)
    temp["shock_ret5"] = (g["ret5_rank"].to_numpy(dtype=float) >= 0.88).astype(float)
    temp["volume_shock"] = (g["amt_ratio20_rank"].to_numpy(dtype=float) >= 0.88).astype(float)
    temp["breakout"] = (g["near_high20_rank"].to_numpy(dtype=float) >= 0.86).astype(float)
    temp["close_strong"] = (g["close_strength_rank"].to_numpy(dtype=float) >= 0.70).astype(float)
    grouped = temp.groupby("cell", sort=False)
    cell_size = grouped["cell"].transform("size").astype(float)
    g[f"{schema}_cell_size_rank"] = pd.Series(cell_size).rank(pct=True).to_numpy(dtype=float)
    session_means = {col: float(np.nanmean(temp[col].to_numpy(dtype=float))) for col in temp.columns if col != "cell"}
    for col in PEER_VALUE_COLS:
        peer = grouped[col].transform("mean").to_numpy(dtype=float)
        peer = np.where(cell_size >= 20, peer, session_means[col])
        out_col = peer_feature_name(schema, col)
        g[out_col] = peer
    for flag in ["shock_ret1", "shock_ret5", "volume_shock", "breakout", "close_strong"]:
        peer = grouped[flag].transform("mean").to_numpy(dtype=float)
        peer = np.where(cell_size >= 20, peer, session_means[flag])
        g[f"{schema}_peer_{flag}"] = peer
    g[f"{schema}_rel_ret1"] = g["ret1_rank"] - g[f"{schema}_peer_ret1"]
    g[f"{schema}_rel_ret5"] = g["ret5_rank"] - g[f"{schema}_peer_ret5"]
    g[f"{schema}_rel_ret20"] = g["ret20_rank"] - g[f"{schema}_peer_ret20"]
    g[f"{schema}_rel_ret60"] = g["ret60_rank"] - g[f"{schema}_peer_ret60"]
    g[f"{schema}_lag_ret5"] = g[f"{schema}_peer_ret5"] - g["ret5_rank"]
    g[f"{schema}_lag_ret20"] = g[f"{schema}_peer_ret20"] - g["ret20_rank"]
    g[f"{schema}_lag_ret60"] = g[f"{schema}_peer_ret60"] - g["ret60_rank"]
    g[f"{schema}_local_follower_setup"] = (
        0.35 * g[f"{schema}_peer_shock_ret5"]
        + 0.25 * g[f"{schema}_peer_volume_shock"]
        + 0.20 * g[f"{schema}_lag_ret5"].clip(lower=0.0)
        + 0.20 * g["close_strength_rank"]
        - 0.20 * g["ret1_rank"]
    )
    g[f"{schema}_local_leader_risk"] = (
        0.35 * g["ret1_rank"]
        + 0.25 * g["ret5_rank"]
        + 0.20 * g[f"{schema}_peer_shock_ret1"]
        + 0.20 * (1.0 - g["upper_wick_low_rank"])
    )
    g[f"{schema}_local_quiet_catchup"] = (
        0.30 * g[f"{schema}_lag_ret20"].clip(lower=0.0)
        + 0.25 * g[f"{schema}_peer_breakout"]
        + 0.20 * g["vol20_low_rank"]
        + 0.15 * g["upper_wick_low_rank"]
        + 0.10 * g["close_strength_rank"]
    )
    g[f"{schema}_local_consensus_strength"] = (
        0.25 * g[f"{schema}_peer_ret20"]
        + 0.25 * g[f"{schema}_peer_ret60"]
        + 0.20 * g[f"{schema}_peer_amt_ratio"]
        + 0.15 * g[f"{schema}_peer_near_high"]
        + 0.15 * g[f"{schema}_peer_close_strength"]
    )


def peer_feature_name(schema: str, col: str) -> str:
    mapping = {
        "ret1_rank": "peer_ret1",
        "ret5_rank": "peer_ret5",
        "ret20_rank": "peer_ret20",
        "ret60_rank": "peer_ret60",
        "amount_rank": "peer_amount",
        "amt_ratio20_rank": "peer_amt_ratio",
        "near_high20_rank": "peer_near_high",
        "close_strength_rank": "peer_close_strength",
        "upper_wick_low_rank": "peer_upper_wick_low",
        "multi_style_strength": "peer_style_strength",
    }
    return f"{schema}_{mapping[col]}"


def add_composite_graph_features(g: pd.DataFrame) -> None:
    schemas = list(GRAPH_SCHEMAS)
    g["graph_follower_composite"] = np.mean([g[f"{schema}_local_follower_setup"] for schema in schemas], axis=0)
    g["graph_leader_risk_composite"] = np.mean([g[f"{schema}_local_leader_risk"] for schema in schemas], axis=0)
    g["graph_quiet_catchup_composite"] = np.mean([g[f"{schema}_local_quiet_catchup"] for schema in schemas], axis=0)
    g["graph_consensus_strength"] = np.mean([g[f"{schema}_local_consensus_strength"] for schema in schemas], axis=0)
    g["graph_peer_shock_mean"] = np.mean([g[f"{schema}_peer_shock_ret5"] for schema in schemas], axis=0)
    g["graph_rel_trend_mean"] = np.mean([g[f"{schema}_rel_ret20"] + g[f"{schema}_rel_ret60"] for schema in schemas], axis=0)


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
        reg = make_regressor(year)
        cls = make_classifier(year)
        reg.fit(train[FEATURES], train["exec_label5_open"])
        cls.fit(train[FEATURES], train["top_exec"].astype(int))
        out = test.copy()
        out["score_reg"] = reg.predict(out[FEATURES])
        out["score_cls"] = positive_proba(cls, out[FEATURES])
        out["score_blend"] = 0.55 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.45 * out.groupby("session")["score_cls"].rank(pct=True, method="first")
        out["score_graph_follower"] = out["graph_follower_composite"]
        out["score_graph_quiet_catchup"] = out["graph_quiet_catchup_composite"]
        out["score_graph_consensus"] = out["graph_consensus_strength"]
        out["score_graph_blend_25"] = 0.75 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.25 * out.groupby("session")["graph_follower_composite"].rank(pct=True, method="first")
        out["score_graph_blend_50"] = 0.50 * out.groupby("session")["score_reg"].rank(pct=True, method="first") + 0.50 * out.groupby("session")["graph_follower_composite"].rank(pct=True, method="first")
        rows.append(out)
        folds.append({
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


def make_regressor(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=360,
        learning_rate=0.032,
        num_leaves=39,
        max_depth=6,
        min_child_samples=520,
        subsample=0.85,
        colsample_bytree=0.88,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=RANDOM_SEED + year,
        n_jobs=6,
        verbosity=-1,
    )


def make_classifier(year: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=260,
        learning_rate=0.034,
        num_leaves=39,
        max_depth=6,
        min_child_samples=520,
        subsample=0.85,
        colsample_bytree=0.88,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=RANDOM_SEED + year + 71,
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
    return [
        ("reg", "score_reg"),
        ("cls", "score_cls"),
        ("blend", "score_blend"),
        ("graph_follower", "score_graph_follower"),
        ("graph_quiet_catchup", "score_graph_quiet_catchup"),
        ("graph_consensus", "score_graph_consensus"),
        ("graph_blend_25", "score_graph_blend_25"),
        ("graph_blend_50", "score_graph_blend_50"),
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
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {
                str(session): list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in scored.groupby("session", sort=True)
            }
            out[f"{variant}::top{topk}"] = EXP130.replay_open_selection(selections, market, horizon)
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows: list[dict[str, Any]] = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
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
    for row in result["leaderboard"][:18]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"exec={row['mean_exec_label5_open']:+.6f} raw={row['mean_raw5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
