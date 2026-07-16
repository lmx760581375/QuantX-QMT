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
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


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
FEATURES = list(EXP130.FEATURES)


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
    scored, folds, importances = walk_forward_memory(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_adaptive_neighbor_leaf_memory_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
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
            "features": FEATURES,
            "causality": "All stock/session features use completed daily bars through signal date T. For each test year, memory models use only rows from years strictly before that test year. Future returns are used only as labels for completed historical samples and for evaluation of the test year.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def walk_forward_memory(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year]
        test = work[work["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        train_x = train[FEATURES].to_numpy(dtype=np.float32)
        test_x = test[FEATURES].to_numpy(dtype=np.float32)
        y = train["exec_label5_open"].to_numpy(dtype=np.float32)
        top_y = train["top_exec"].to_numpy(dtype=int)

        leaf_reg = ExtraTreesRegressor(
            n_estimators=96,
            max_features=0.65,
            min_samples_leaf=350,
            bootstrap=True,
            random_state=RANDOM_SEED + year,
            n_jobs=6,
        )
        leaf_cls = ExtraTreesClassifier(
            n_estimators=96,
            max_features=0.65,
            min_samples_leaf=350,
            bootstrap=True,
            random_state=RANDOM_SEED + year + 11,
            n_jobs=6,
        )
        leaf_reg.fit(train_x, y)
        leaf_cls.fit(train_x, top_y)

        proto = fit_prototype_memory(train_x, y, top_y, year)
        proto_score = score_prototype_memory(proto, test_x)

        out = test.copy()
        out["score_leaf_mean"] = leaf_reg.predict(test_x)
        out["score_leaf_toprate"] = leaf_cls.predict_proba(test_x)[:, 1]
        out["score_leaf_blend"] = blend_ranks(out, "score_leaf_mean", "score_leaf_toprate", 0.55)
        out["score_proto_mean"] = proto_score["mean"]
        out["score_proto_tail"] = proto_score["tail"]
        out["score_proto_toprate"] = proto_score["toprate"]
        out["score_proto_blend"] = 0.45 * rank_by_session(out, "score_proto_mean") + 0.35 * rank_by_session(out, "score_proto_tail") + 0.20 * rank_by_session(out, "score_proto_toprate")
        out["score_memory_blend"] = 0.50 * rank_by_session(out, "score_leaf_blend") + 0.50 * rank_by_session(out, "score_proto_blend")
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "prototype_clusters": int(proto["centers"].shape[0]),
        })
        importances.append({f"leaf_reg::{name}": float(value) for name, value in zip(FEATURES, leaf_reg.feature_importances_)})
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def fit_prototype_memory(train_x: np.ndarray, y: np.ndarray, top_y: np.ndarray, year: int) -> dict[str, np.ndarray]:
    n_clusters = int(min(512, max(96, len(train_x) // 2200)))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=8192,
        max_iter=80,
        n_init=3,
        random_state=RANDOM_SEED + year + 101,
        reassignment_ratio=0.01,
    )
    labels = km.fit_predict(train_x)
    mean = np.zeros(n_clusters, dtype=np.float32)
    tail = np.zeros(n_clusters, dtype=np.float32)
    toprate = np.zeros(n_clusters, dtype=np.float32)
    count = np.zeros(n_clusters, dtype=np.int32)
    global_mean = float(np.mean(y))
    global_tail = float(np.quantile(y, 0.80))
    global_toprate = float(np.mean(top_y))
    for cluster in range(n_clusters):
        mask = labels == cluster
        c = int(mask.sum())
        count[cluster] = c
        if c < 80:
            mean[cluster] = global_mean
            tail[cluster] = global_tail
            toprate[cluster] = global_toprate
            continue
        vals = y[mask]
        mean[cluster] = float(np.mean(vals))
        tail[cluster] = float(np.quantile(vals, 0.80))
        toprate[cluster] = float(np.mean(top_y[mask]))
    shrink = np.minimum(count.astype(np.float32) / 600.0, 1.0)
    mean = shrink * mean + (1.0 - shrink) * global_mean
    tail = shrink * tail + (1.0 - shrink) * global_tail
    toprate = shrink * toprate + (1.0 - shrink) * global_toprate
    return {"centers": km.cluster_centers_.astype(np.float32), "mean": mean, "tail": tail, "toprate": toprate}


def score_prototype_memory(proto: dict[str, np.ndarray], test_x: np.ndarray) -> dict[str, np.ndarray]:
    centers = proto["centers"]
    labels = nearest_center(test_x, centers)
    return {"mean": proto["mean"][labels], "tail": proto["tail"][labels], "toprate": proto["toprate"][labels]}


def nearest_center(x: np.ndarray, centers: np.ndarray, chunk: int = 20000) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int32)
    center_norm = np.sum(centers * centers, axis=1)
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        dist = np.sum(block * block, axis=1, keepdims=True) + center_norm[None, :] - 2.0 * block @ centers.T
        out[start:start + len(block)] = np.argmin(dist, axis=1).astype(np.int32)
    return out


def rank_by_session(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("session")[col].rank(pct=True, method="first")


def blend_ranks(frame: pd.DataFrame, a: str, b: str, wa: float) -> pd.Series:
    return wa * rank_by_session(frame, a) + (1.0 - wa) * rank_by_session(frame, b)


def variants() -> list[tuple[str, str]]:
    return [
        ("leaf_mean", "score_leaf_mean"),
        ("leaf_toprate", "score_leaf_toprate"),
        ("leaf_blend", "score_leaf_blend"),
        ("proto_mean", "score_proto_mean"),
        ("proto_tail", "score_proto_tail"),
        ("proto_toprate", "score_proto_toprate"),
        ("proto_blend", "score_proto_blend"),
        ("memory_blend", "score_memory_blend"),
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
        out[f"{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(y)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
                }
                for y, yg in group.groupby("year")
            },
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {}
            for session, group in grouped.items():
                selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)
                selections[session] = list(selected["instrument"].astype(str))
            out[f"{variant}::top{topk}"] = EXP130.replay_open_selection(selections, market, horizon)
    return out


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


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:50]]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    for row in result["leaderboard"][:16]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"exec_label={row['mean_exec_label5_open']:+.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
