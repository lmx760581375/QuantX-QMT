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
TOPKS = (10, 20)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module("exp130_for_leaf_complement", EXP130_PATH)
EXP134 = load_module("exp134_for_leaf_complement", EXP134_PATH)


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
    style_scored, style_folds, style_importances = EXP130.walk_forward(panel, args.min_train_rows)
    memory_scored, memory_folds, memory_importances = EXP134.walk_forward_memory(panel, args.min_train_rows)
    scored = merge_scores(style_scored, memory_scored)
    add_composite_scores(scored)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    complement = complement_diagnostics(scored)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_leaf_mean_complement_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "style_folds": style_folds,
        "memory_folds": memory_folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "complement_diagnostics": complement,
        "leaderboard": build_leaderboard(accounts, detail),
        "feature_importance_heads": {
            "style": EXP130.summarize_importance(style_importances)[:12],
            "leaf": EXP134.summarize_importance(memory_importances)[:12],
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
            "causality": "All stock/session features use completed daily bars through signal date T. Style and leaf-memory models are trained yearly with rows from years strictly before the test year. Composite scores use only same-day OOS model scores. Future returns are used only for evaluation and diagnostics.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
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
    style_cols = keys + ["score_reg", "score_cls", "score_blend", "score_style_lag_catchup"]
    memory_cols = keys + ["score_leaf_mean", "score_leaf_blend", "score_memory_blend", "score_proto_mean"]
    left = style_scored[base_cols + [c for c in style_cols if c not in base_cols]].copy()
    right = memory_scored[memory_cols].copy()
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("merged scored panel is empty")
    return merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_composite_scores(scored: pd.DataFrame) -> None:
    rank_cols = {
        "r_style_reg": "score_reg",
        "r_style_blend": "score_blend",
        "r_leaf_mean": "score_leaf_mean",
        "r_leaf_blend": "score_leaf_blend",
        "r_memory_blend": "score_memory_blend",
        "r_style_lag": "score_style_lag_catchup",
    }
    for out, col in rank_cols.items():
        scored[out] = scored.groupby("session")[col].rank(pct=True, method="first")
    scored["score_leaf_reg_25"] = 0.75 * scored["r_style_reg"] + 0.25 * scored["r_leaf_mean"]
    scored["score_leaf_reg_50"] = 0.50 * scored["r_style_reg"] + 0.50 * scored["r_leaf_mean"]
    scored["score_leaf_blend_25"] = 0.75 * scored["r_style_blend"] + 0.25 * scored["r_leaf_mean"]
    scored["score_leaf_blend_50"] = 0.50 * scored["r_style_blend"] + 0.50 * scored["r_leaf_mean"]
    scored["score_leaf_reg_consensus"] = np.minimum(scored["r_style_reg"], scored["r_leaf_mean"])
    scored["score_leaf_reg_max"] = np.maximum(scored["r_style_reg"], scored["r_leaf_mean"])
    scored["score_leaf_lag_50"] = 0.50 * scored["r_style_lag"] + 0.50 * scored["r_leaf_mean"]


def variants() -> list[tuple[str, str]]:
    return [
        ("style_reg", "score_reg"),
        ("style_blend", "score_blend"),
        ("style_lag_catchup", "score_style_lag_catchup"),
        ("leaf_mean", "score_leaf_mean"),
        ("leaf_blend", "score_leaf_blend"),
        ("memory_blend", "score_memory_blend"),
        ("leaf_reg_25", "score_leaf_reg_25"),
        ("leaf_reg_50", "score_leaf_reg_50"),
        ("leaf_blend_25", "score_leaf_blend_25"),
        ("leaf_blend_50", "score_leaf_blend_50"),
        ("leaf_reg_consensus", "score_leaf_reg_consensus"),
        ("leaf_reg_max", "score_leaf_reg_max"),
        ("leaf_lag_50", "score_leaf_lag_50"),
    ]


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
        for topk in TOPKS:
            selections = {}
            for session, group in scored.groupby("session", sort=True):
                selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)
                selections[str(session)] = list(selected["instrument"].astype(str))
            out[f"{variant}::top{topk}"] = EXP130.replay_open_selection(selections, market, horizon)
    return out


def complement_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    pairs = [("style_reg", "score_reg"), ("style_blend", "score_blend"), ("style_lag_catchup", "score_style_lag_catchup")]
    out: dict[str, Any] = {}
    leaf_col = "score_leaf_mean"
    for name, score_col in pairs:
        rows = []
        for session, group in scored.groupby("session", sort=True):
            leaf = set(group.sort_values([leaf_col, "instrument"], ascending=[False, True]).head(20)["instrument"].astype(str))
            other = set(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(20)["instrument"].astype(str))
            leaf_ret = float(group[group["instrument"].astype(str).isin(leaf)]["exec_label5_open"].mean())
            other_ret = float(group[group["instrument"].astype(str).isin(other)]["exec_label5_open"].mean())
            union = leaf | other
            inter = leaf & other
            leaf_only = leaf - other
            other_only = other - leaf
            rows.append({
                "session": session,
                "year": int(str(session)[:4]),
                "overlap": len(inter),
                "jaccard": len(inter) / max(1, len(union)),
                "leaf_exec": leaf_ret,
                "other_exec": other_ret,
                "leaf_only_exec": subset_mean(group, leaf_only),
                "other_only_exec": subset_mean(group, other_only),
                "intersection_exec": subset_mean(group, inter),
                "union_top20_exec": union_top20_exec(group, leaf_col, score_col),
            })
        frame = pd.DataFrame(rows)
        out[f"leaf_vs_{name}"] = {
            "mean_overlap_top20": float(frame["overlap"].mean()),
            "mean_jaccard_top20": float(frame["jaccard"].mean()),
            "session_exec_corr": float(frame[["leaf_exec", "other_exec"]].corr().iloc[0, 1]),
            "leaf_beats_other_ratio": float((frame["leaf_exec"] > frame["other_exec"]).mean()),
            "other_negative_leaf_positive_ratio": float(((frame["other_exec"] < 0) & (frame["leaf_exec"] > 0)).mean()),
            "mean_leaf_only_exec": float(frame["leaf_only_exec"].mean()),
            "mean_other_only_exec": float(frame["other_only_exec"].mean()),
            "mean_intersection_exec": float(frame["intersection_exec"].mean()),
            "mean_union_top20_exec": float(frame["union_top20_exec"].mean()),
            "by_year": {
                str(int(year)): {
                    "mean_overlap_top20": float(yg["overlap"].mean()),
                    "session_exec_corr": float(yg[["leaf_exec", "other_exec"]].corr().iloc[0, 1]),
                    "leaf_beats_other_ratio": float((yg["leaf_exec"] > yg["other_exec"]).mean()),
                    "mean_leaf_exec": float(yg["leaf_exec"].mean()),
                    "mean_other_exec": float(yg["other_exec"].mean()),
                    "mean_union_top20_exec": float(yg["union_top20_exec"].mean()),
                }
                for year, yg in frame.groupby("year")
            },
        }
    return out


def subset_mean(group: pd.DataFrame, instruments: set[str]) -> float:
    if not instruments:
        return 0.0
    return float(group[group["instrument"].astype(str).isin(instruments)]["exec_label5_open"].mean())


def union_top20_exec(group: pd.DataFrame, leaf_col: str, other_col: str) -> float:
    temp = group.copy()
    temp["union_rank_score"] = 0.5 * temp[leaf_col].rank(pct=True, method="first") + 0.5 * temp[other_col].rank(pct=True, method="first")
    return float(temp.sort_values(["union_rank_score", "instrument"], ascending=[False, True]).head(20)["exec_label5_open"].mean())


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


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:20]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"exec_label={row['mean_exec_label5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
