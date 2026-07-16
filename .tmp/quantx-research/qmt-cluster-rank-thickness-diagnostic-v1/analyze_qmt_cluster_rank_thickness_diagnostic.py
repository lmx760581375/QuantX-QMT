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
EXP148_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-dynamic-theme-cluster-ranker-v1/analyze_qmt_dynamic_theme_cluster_ranker.py"
TOPKS = (10, 20)
SCORE_COL = "score_cluster_rank"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP148 = load_module("exp148_for_cluster_rank_thickness", EXP148_PATH)


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
    parser.add_argument("--anchor-count", type=int, default=EXP148.ANCHOR_COUNT)
    parser.add_argument("--corr-window", type=int, default=EXP148.CORR_WINDOW)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP148.EXP130.EXP124.load_market(args)
    panel = EXP148.EXP130.build_panel(market, args)
    panel = EXP148.add_dynamic_cluster_features(panel, market, args.anchor_count, args.corr_window)
    scored, folds, importances = EXP148.walk_forward(panel, args.min_train_rows)
    rank_slices = rank_slice_diagnostics(scored)
    concentration = concentration_diagnostics(scored)
    contrast = top10_tail_contrast(scored)
    counterfactuals = counterfactual_replays(scored, market, args.horizon)
    result = {
        "status": classify_status(counterfactuals),
        "experiment": "qmt_cluster_rank_thickness_diagnostic_v1",
        "source_experiment": "qmt_dynamic_theme_cluster_ranker_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "rank_slice_diagnostics": rank_slices,
        "concentration_diagnostics": concentration,
        "top10_vs_11_20_contrast": contrast,
        "counterfactual_account_results": counterfactuals,
        "source_feature_importance_top20": EXP148.summarize_importance(importances)[:20],
        "references": {
            "exp148_cluster_rank_top10_multiple": 8.0478,
            "exp148_cluster_rank_top10_2026": 0.1763,
            "exp148_cluster_rank_top20_multiple": 3.9764,
            "exp148_cluster_rank_top20_2026": 0.0331,
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "anchor_count": args.anchor_count,
            "corr_window": args.corr_window,
            "causality": "Scored panel is regenerated with Exp148 annual walk-forward. Diagnostics and counterfactual selections use only Exp148 out-of-sample scores and T-day cluster features. Future returns are used only for evaluation/replay, not for constructing scores or selecting parameters.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no static industry/concept table, news, announcements, LHB, ETF, northbound, financing, or event/message data.",
            "interpretation_guardrail": "Cluster caps and slice selections are diagnostics, not deployable hard gates and not counted as satisfying the strategy target.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def rank_slice_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    slices = [(1, 5), (6, 10), (11, 15), (16, 20), (1, 10), (11, 20)]
    for session, group in scored.groupby("session", sort=True):
        ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True]).reset_index(drop=True)
        ranked["rank_pos"] = np.arange(1, len(ranked) + 1)
        for lo, hi in slices:
            selected = ranked[(ranked["rank_pos"] >= lo) & (ranked["rank_pos"] <= hi)]
            rows.append({
                "session": str(session),
                "year": int(str(session)[:4]),
                "slice": f"{lo}_{hi}",
                "count": int(len(selected)),
                "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
                "mean_raw5_open": safe_mean(selected["raw5_open"]),
                "entry_ok": safe_mean(selected["entry_ok"]),
                "unique_clusters": int(selected["cluster_id"].nunique()),
                "max_cluster_share": max_cluster_share(selected),
            })
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for name, group in frame.groupby("slice"):
        out[name] = summarize_session_frame(group)
    return out


def concentration_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True])
        for topk in TOPKS:
            selected = ranked.head(topk)
            rows.append({
                "session": str(session),
                "year": int(str(session)[:4]),
                "topk": int(topk),
                "unique_clusters": int(selected["cluster_id"].nunique()),
                "max_cluster_share": max_cluster_share(selected),
                "mean_cluster_size_rank": safe_mean(selected["cluster_size_rank"]),
                "mean_cluster_corr_anchor": safe_mean(selected["cluster_corr_anchor"]),
                "mean_cluster_overheat_risk": safe_mean(selected["cluster_overheat_risk"]),
            })
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for topk, group in frame.groupby("topk"):
        out[f"top{int(topk)}"] = summarize_session_frame(group)
    return out


def top10_tail_contrast(scored: pd.DataFrame) -> dict[str, Any]:
    features = [
        "cluster_size_rank", "cluster_anchor_score", "cluster_corr_anchor", "cluster_ret20_mean",
        "cluster_ret60_mean", "cluster_amount_mean", "cluster_amt_ratio_mean", "cluster_near_high_mean",
        "cluster_breadth20", "cluster_disp20", "cluster_rel_ret5", "cluster_rel_ret20", "cluster_rel_ret60",
        "cluster_rel_amount", "cluster_rel_amt_ratio", "cluster_rel_near_high", "cluster_rank_ret5",
        "cluster_rank_ret20", "cluster_rank_ret60", "cluster_rank_amount", "cluster_rank_amt_ratio",
        "cluster_rank_near_high", "cluster_rank_close_strength", "cluster_rank_upper_wick_low",
        "cluster_lag_repair", "cluster_mid_follower", "cluster_leader_quality", "cluster_overheat_risk",
        "ret5_rank", "ret20_rank", "ret60_rank", "amount_rank", "amt_ratio20_rank", "near_high20_rank",
    ]
    top_rows: list[pd.DataFrame] = []
    tail_rows: list[pd.DataFrame] = []
    for _, group in scored.groupby("session", sort=True):
        ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True]).reset_index(drop=True)
        top_rows.append(ranked.iloc[:10])
        tail_rows.append(ranked.iloc[10:20])
    top = pd.concat(top_rows, ignore_index=True)
    tail = pd.concat(tail_rows, ignore_index=True)
    rows = []
    for feature in features:
        rows.append({
            "feature": feature,
            "top10_mean": safe_mean(top[feature]),
            "rank11_20_mean": safe_mean(tail[feature]),
            "tail_minus_top": safe_mean(tail[feature]) - safe_mean(top[feature]),
        })
    rows.sort(key=lambda r: abs(r["tail_minus_top"]), reverse=True)
    return {
        "top10_mean_exec_label5_open": safe_mean(top["exec_label5_open"]),
        "rank11_20_mean_exec_label5_open": safe_mean(tail["exec_label5_open"]),
        "top10_entry_ok": safe_mean(top["entry_ok"]),
        "rank11_20_entry_ok": safe_mean(tail["entry_ok"]),
        "feature_contrast_top30": rows[:30],
    }


def counterfactual_replays(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    selectors = {
        "plain_top10": lambda g: select_plain(g, 10),
        "plain_top20": lambda g: select_plain(g, 20),
        "cap1_top20": lambda g: select_cluster_cap(g, 20, 1),
        "cap2_top20": lambda g: select_cluster_cap(g, 20, 2),
        "cap3_top20": lambda g: select_cluster_cap(g, 20, 3),
        "top10_plus_cap2_fill": lambda g: select_top10_plus_cap_fill(g, 20, 2),
        "best_cluster_first_top20": lambda g: select_best_cluster_first(g, 20),
    }
    out: dict[str, Any] = {}
    grouped = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for name, selector in selectors.items():
        selections: dict[str, list[str]] = {}
        label_rows: list[dict[str, Any]] = []
        for session, group in grouped.items():
            selected = selector(group)
            selections[session] = list(selected["instrument"].astype(str))
            label_rows.append({
                "session": session,
                "year": int(session[:4]),
                "count": int(len(selected)),
                "mean_exec_label5_open": safe_mean(selected["exec_label5_open"]),
                "mean_raw5_open": safe_mean(selected["raw5_open"]),
                "unique_clusters": int(selected["cluster_id"].nunique()),
                "max_cluster_share": max_cluster_share(selected),
            })
        account = EXP148.EXP130.replay_open_selection(selections, market, horizon)
        out[name] = {"account": account, "label_summary": summarize_session_frame(pd.DataFrame(label_rows))}
    return out


def select_plain(group: pd.DataFrame, topk: int) -> pd.DataFrame:
    return group.sort_values([SCORE_COL, "instrument"], ascending=[False, True]).head(topk)


def select_cluster_cap(group: pd.DataFrame, topk: int, cap: int) -> pd.DataFrame:
    ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True])
    counts: dict[int, int] = {}
    rows = []
    for _, row in ranked.iterrows():
        cid = int(row["cluster_id"])
        if counts.get(cid, 0) >= cap:
            continue
        rows.append(row)
        counts[cid] = counts.get(cid, 0) + 1
        if len(rows) >= topk:
            break
    return pd.DataFrame(rows)


def select_top10_plus_cap_fill(group: pd.DataFrame, topk: int, cap: int) -> pd.DataFrame:
    ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True])
    head = ranked.head(10)
    counts = head["cluster_id"].astype(int).value_counts().to_dict()
    rows = [row for _, row in head.iterrows()]
    for _, row in ranked.iloc[10:].iterrows():
        cid = int(row["cluster_id"])
        if counts.get(cid, 0) >= cap:
            continue
        rows.append(row)
        counts[cid] = counts.get(cid, 0) + 1
        if len(rows) >= topk:
            break
    return pd.DataFrame(rows)


def select_best_cluster_first(group: pd.DataFrame, topk: int) -> pd.DataFrame:
    ranked = group.sort_values([SCORE_COL, "instrument"], ascending=[False, True]).copy()
    ranked["cluster_best"] = ranked.groupby("cluster_id")[SCORE_COL].transform("max")
    ranked["cluster_local_rank"] = ranked.groupby("cluster_id")[SCORE_COL].rank(ascending=False, method="first")
    return ranked.sort_values(["cluster_best", "cluster_local_rank", SCORE_COL, "instrument"], ascending=[False, True, False, True]).head(topk)


def summarize_session_frame(frame: pd.DataFrame) -> dict[str, Any]:
    out = {
        "sessions": int(len(frame)),
        "mean_exec_label5_open": float(frame["mean_exec_label5_open"].mean()) if "mean_exec_label5_open" in frame else np.nan,
        "mean_raw5_open": float(frame["mean_raw5_open"].mean()) if "mean_raw5_open" in frame else np.nan,
        "avg_count": float(frame["count"].mean()) if "count" in frame else np.nan,
        "avg_unique_clusters": float(frame["unique_clusters"].mean()) if "unique_clusters" in frame else np.nan,
        "avg_max_cluster_share": float(frame["max_cluster_share"].mean()) if "max_cluster_share" in frame else np.nan,
        "by_year": {},
    }
    if "year" in frame:
        for year, group in frame.groupby("year"):
            out["by_year"][str(int(year))] = {
                "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()) if "mean_exec_label5_open" in group else np.nan,
                "mean_raw5_open": float(group["mean_raw5_open"].mean()) if "mean_raw5_open" in group else np.nan,
                "avg_count": float(group["count"].mean()) if "count" in group else np.nan,
                "avg_unique_clusters": float(group["unique_clusters"].mean()) if "unique_clusters" in group else np.nan,
                "avg_max_cluster_share": float(group["max_cluster_share"].mean()) if "max_cluster_share" in group else np.nan,
            }
    return out


def max_cluster_share(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    return float(selected["cluster_id"].value_counts().max() / len(selected))


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else 0.0


def classify_status(counterfactuals: dict[str, Any]) -> str:
    best = max((data["account"].get("final_multiple", 1.0) for data in counterfactuals.values()), default=1.0)
    if best >= 8.7749:
        return "diagnostic_has_promising_counterfactual_but_not_deployable_gate"
    return "diagnostic_complete_top20_thickness_not_solved"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for name, data in result["counterfactual_account_results"].items():
        acc = data["account"]
        annual = acc.get("annual_returns", {})
        print(
            f"{name}: final={acc.get('final_multiple', 1.0):.2f}x 2026={annual.get('2026', float('nan')):+.4f} "
            f"min_year={(min(annual.values()) if annual else 0.0):+.4f} avg_count={acc.get('avg_selected_count', 0.0):.2f}"
        )
    print("slice_11_20", json.dumps(result["rank_slice_diagnostics"].get("11_20", {}), ensure_ascii=False, sort_keys=True)[:900])


if __name__ == "__main__":
    raise SystemExit(main())
