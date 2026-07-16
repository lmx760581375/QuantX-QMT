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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
V1_PATH = REPO_ROOT / ".tmp/quantx-research/path-sequence-ranker-v1/analyze_path_sequence_ranker.py"
RANDOM_SEED = 20260715
TOPKS = (15, 20)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("path_sequence_ranker_v1_reuse", V1_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="data/qlib_data_fixed")
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--load-start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--dev-top200", default=".tmp/quantx-research/path-sequence-ranker-v1/base-top200/base_5d_dev_2021_2025_top200.parquet")
    parser.add_argument("--forward-top200", default=".tmp/quantx-research/path-sequence-ranker-v1/base-top200/base_5d_val63_2026_top200.parquet")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=30000)
    parser.add_argument("--cache", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2.parquet")
    parser.add_argument("--output", default=".tmp/quantx-research/path-sequence-ranker-v2/path_sequence_ranker_v2_summary.json")
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = V1.EXP124.load_market(args)
    scored, base_folds, base_importances = load_or_build_scored(args, market)
    scored, aux_folds, aux_importances = add_exec_aware_scores(scored, args.min_train_rows)
    scored = add_blend_scores(scored)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "v2_complete_rebuild_based_on_v1_scored_panel",
        "experiment": "path_sequence_ranker_v2_stable_exec_blends",
        "panel_rows": int(len(scored)),
        "base_folds": base_folds,
        "aux_folds": aux_folds,
        "label_results": summarize_detail(detail),
        "account_results": add_risk_summaries(accounts),
        "base_feature_importance": V1.summarize_importance(base_importances) if base_importances else [],
        "aux_feature_importance": summarize_importance(aux_importances),
        "gates": {
            "min_avg_selected_count": 18.0,
            "target_final_multiple": 8.7749,
            "baseline_rebuild_best_final_multiple": 8.168796954642778,
            "target_2026_return": 0.1918,
            "target_max_drawdown_not_worse_than": -0.3262,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "load_start": args.load_start,
            "end": args.end,
            "dev_top200": args.dev_top200,
            "forward_top200": args.forward_top200,
            "horizon": args.horizon,
            "cache": args.cache,
            "causality": "Reuses v1 PIT top200 PredictionStore records and path features through signal day T. Aux heads train walk-forward on years strictly before test year, using exec_label5_open labels only from prior years. Account replay selects after-close T and executes T+1 open with rebalance interval 5.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def load_or_build_scored(args: argparse.Namespace, market: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    cache = Path(args.cache)
    meta_path = cache.with_suffix(".summary.json")
    if cache.exists():
        scored = pd.read_parquet(cache)
        folds = json.loads(meta_path.read_text(encoding="utf-8")).get("base_folds", []) if meta_path.exists() else []
        return scored, folds, []
    top200 = V1.load_top200(args.dev_top200, args.forward_top200)
    panel = V1.build_panel(market, top200, args.horizon)
    scored, folds, importances = V1.walk_forward(panel, args.min_train_rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(cache, index=False)
    meta = {
        "cache": str(cache),
        "cache_checksum": "sha256:" + hashlib.sha256(cache.read_bytes()).hexdigest(),
        "rows": int(len(scored)),
        "base_folds": folds,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scored, folds, importances


def add_exec_aware_scores(scored: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = scored.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    work["target_exec_top20"] = (work.groupby("session")["exec_label5_open"].rank(pct=True, method="first") >= 0.90).astype(int)
    for year in (2022, 2023, 2024, 2025, 2026):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if test.empty:
            continue
        if len(train) < min_train_rows:
            out = test.copy()
            out["score_exec_reg"] = 0.0
            out["score_exec_cls"] = 0.0
            out["aux_available"] = 0.0
            rows.append(out)
            folds.append({"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique()), "aux_available": False})
            print(json.dumps({"aux_fold": folds[-1]}, ensure_ascii=False), flush=True)
            continue
        reg = make_exec_reg(year)
        cls = make_exec_cls(year)
        reg.fit(train[V1.FEATURES], train["exec_label5_open"])
        cls.fit(train[V1.FEATURES], train["target_exec_top20"])
        out = test.copy()
        out["score_exec_reg"] = reg.predict(out[V1.FEATURES])
        proba = cls.predict_proba(out[V1.FEATURES])
        out["score_exec_cls"] = proba[:, int(np.where(cls.classes_ == 1)[0][0])] if 1 in cls.classes_ else 0.0
        out["aux_available"] = 1.0
        rows.append(out)
        folds.append({"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique()), "aux_available": True})
        imp = {f"exec_reg::{name}": float(value) for name, value in zip(V1.FEATURES, reg.feature_importances_)}
        imp.update({f"exec_cls::{name}": float(value) for name, value in zip(V1.FEATURES, cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps({"aux_fold": folds[-1]}, ensure_ascii=False), flush=True)
    if not rows:
        raise RuntimeError("no aux folds produced")
    return pd.concat(rows, ignore_index=True), folds, importances


def make_exec_reg(year: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=260, learning_rate=0.035, num_leaves=31, max_depth=5,
        min_child_samples=460, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + int(year),
        n_jobs=6, verbosity=-1,
    )


def make_exec_cls(year: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=220, learning_rate=0.035, num_leaves=31, max_depth=5,
        min_child_samples=460, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + int(year) + 97,
        n_jobs=6, verbosity=-1,
    )


def rank_pct(frame: pd.DataFrame, col: str) -> pd.Series:
    ranked = frame.groupby("session")[col].rank(pct=True, method="first")
    nunique = frame.groupby("session")[col].transform("nunique")
    return ranked.where(nunique > 1, 0.5)


def add_blend_scores(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    for col in ("score_base", "score_path_ev", "score_path_topq", "score_path_spread", "score_exec_reg", "score_exec_cls"):
        out[f"{col}_pct"] = rank_pct(out, col)
    neutral_aux = out.get("aux_available", pd.Series(1.0, index=out.index)).astype(float) < 0.5
    out.loc[neutral_aux, ["score_exec_reg_pct", "score_exec_cls_pct"]] = 0.5
    out["score_blend_stable_1"] = 0.45 * out["score_path_topq_pct"] + 0.35 * out["score_path_spread_pct"] + 0.20 * out["score_base_pct"]
    out["score_blend_stable_2"] = 0.40 * out["score_path_topq_pct"] + 0.40 * out["score_path_ev_pct"] + 0.20 * out["score_base_pct"]
    out["score_blend_stable_3"] = 0.35 * out["score_path_topq_pct"] + 0.35 * out["score_path_spread_pct"] + 0.20 * out["score_path_ev_pct"] + 0.10 * out["score_base_pct"]
    out["score_rank_consensus"] = (out["score_path_topq_pct"] + out["score_path_spread_pct"] + out["score_path_ev_pct"] + out["score_base_pct"]) / 4.0
    out["score_exec_reg_blend_20"] = 0.80 * out["score_path_topq_pct"] + 0.20 * out["score_exec_reg_pct"]
    out["score_exec_cls_blend_20"] = 0.80 * out["score_path_topq_pct"] + 0.20 * out["score_exec_cls_pct"]
    out["score_stable_exec_cls_15"] = 0.85 * out["score_rank_consensus"] + 0.15 * out["score_exec_cls_pct"]
    out["score_drawdown_aware_consensus"] = (
        0.45 * out["score_rank_consensus"]
        + 0.20 * out["near_high20"]
        + 0.15 * out["near_high60"]
        + 0.10 * out["vol20_low_rank"]
        + 0.10 * out["range20_low"]
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def variants() -> list[tuple[str, str]]:
    return [
        ("base", "score_base"),
        ("path_ev", "score_path_ev"),
        ("path_topq", "score_path_topq"),
        ("path_spread", "score_path_spread"),
        ("blend_topq_25", "score_blend_topq_25"),
        ("blend_stable_1", "score_blend_stable_1"),
        ("blend_stable_2", "score_blend_stable_2"),
        ("blend_stable_3", "score_blend_stable_3"),
        ("rank_consensus", "score_rank_consensus"),
        ("drawdown_aware_consensus", "score_drawdown_aware_consensus"),
        ("exec_reg_blend_20", "score_exec_reg_blend_20"),
        ("exec_cls_blend_20", "score_exec_cls_blend_20"),
        ("stable_exec_cls_15", "score_stable_exec_cls_15"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    due = V1.due_sessions(scored)
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for scope in ("all", "due5"):
            if scope == "due5" and session not in due:
                continue
            for variant, score_col in variants():
                ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
                for topk in TOPKS:
                    selected = ranked.head(topk)
                    rows.append({
                        "scope": scope,
                        "session": session,
                        "year": int(session[:4]),
                        "variant": variant,
                        "topk": int(topk),
                        "mean_label5_open": float(selected["label5_open"].mean()),
                        "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                        "mean_raw5_open": float(selected["raw5_open"].mean()),
                        "entry_ok": float(selected["entry_ok"].mean()),
                        "exit_ok": float(selected["exit_ok"].mean()),
                    })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (scope, variant, topk), group in detail.groupby(["scope", "variant", "topk"]):
        out[f"{scope}::pool200::{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {str(int(y)): {"mean_label5_open": float(yg["mean_label5_open"].mean()), "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean())} for y, yg in group.groupby("year")},
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    due = V1.due_sessions(scored)
    work = scored[scored["session"].isin(due)]
    for variant, score_col in variants():
        for topk in (20,):
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in work.groupby("session", sort=True)
            }
            out[f"{variant}::pool200::top{topk}::reb5"] = V1.EXP129.replay_open_selection(selections, market, horizon)
    return out


def add_risk_summaries(accounts: dict[str, Any]) -> dict[str, Any]:
    for account in accounts.values():
        annual = account.get("annual_returns", {})
        values = [float(v) for v in annual.values()]
        if len(values) > 1:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            account["annual_return_mean"] = mean
            account["annual_return_std"] = std
            account["annual_sharpe_proxy"] = float(mean / std) if std > 0 else None
            account["min_annual_return"] = float(np.min(values))
            account["passes_core_gate"] = bool(
                account.get("avg_selected_count", 0.0) >= 18.0
                and account.get("final_multiple", 0.0) >= 8.168796954642778
                and float(annual.get("2026", -999.0)) >= 0.1918
                and all(v > 0 for v in values)
                and account.get("max_drawdown_period", -1.0) >= -0.3262
            )
    return accounts


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:80]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(checksum)
    keys = sorted(result["account_results"], key=lambda k: result["account_results"][k].get("final_multiple", 0.0), reverse=True)
    for key in keys:
        account = result["account_results"][key]
        print(
            f"{key}: final={account.get('final_multiple', 1.0):.4f} "
            f"2026={account.get('annual_returns', {}).get('2026', 0.0):+.4f} "
            f"mdd={account.get('max_drawdown_period', 0.0):+.4f} "
            f"avg_count={account.get('avg_selected_count', 0.0):.2f} "
            f"sharpe_proxy={account.get('annual_sharpe_proxy')} "
            f"pass={account.get('passes_core_gate')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
