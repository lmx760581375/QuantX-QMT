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
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    scored, folds, importances = EXP130.walk_forward(panel, args.min_train_rows)
    leakage = diagnose_leakage(scored, market, args.horizon)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_label_to_account_leakage_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "leakage": leakage,
        "feature_importance_top20": EXP130.summarize_importance(importances)[:20],
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
            "causality": "Reuses Exp130: all features are computed from completed T daily bars only; signal is after T close; realized diagnostics use pre-scheduled T+1 open entry and T+6 open target exit. Exit delay checks use future data only to measure fill feasibility, not to form signals.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, or event/message data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def diagnose_leakage(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, score_col in EXP130.variants():
        for topk in TOPKS:
            key = f"{variant}::top{topk}"
            selections = select_by_session(scored, score_col, topk)
            period_rows = build_period_rows(scored, selections)
            replay_rows, rejects = replay_period_rows(selections, market, horizon)
            merged = period_rows.merge(replay_rows, on="session", how="left", suffixes=("", "_replay"))
            out[key] = summarize_leakage(merged, rejects)
    return out


def select_by_session(scored: pd.DataFrame, score_col: str, topk: int) -> dict[str, pd.DataFrame]:
    selections = {}
    for session, group in scored.groupby("session", sort=True):
        selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk).copy()
        selected["bench_raw5_open"] = selected["raw5_open"] - selected["label5_open"]
        selected["exec_raw5_open"] = selected["exec_label5_open"] + selected["bench_raw5_open"]
        selections[str(session)] = selected
    return selections


def build_period_rows(scored: pd.DataFrame, selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market_bench = scored.groupby("session", sort=True).agg(
        year=("year", "first"),
        universe_raw5_open=("raw5_open", "mean"),
        universe_exec_label5_open=("exec_label5_open", "mean"),
    )
    rows = []
    previous: set[str] | None = None
    for session, selected in selections.items():
        names = set(selected["instrument"].astype(str))
        overlap = np.nan if previous is None else len(names & previous) / max(len(names), 1)
        previous = names
        rows.append({
            "session": session,
            "year": int(selected["year"].iloc[0]),
            "selected_count_label": int(len(selected)),
            "mean_label_excess": float(selected["label5_open"].mean()),
            "mean_exec_excess": float(selected["exec_label5_open"].mean()),
            "mean_raw_open": float(selected["raw5_open"].mean()),
            "mean_exec_raw_proxy": float(selected["exec_raw5_open"].mean()),
            "entry_ok": float(selected["entry_ok"].mean()),
            "exit_ok": float(selected["exit_ok"].mean()),
            "overlap_prev": float(overlap) if np.isfinite(overlap) else None,
            "p10_exec_excess": float(selected["exec_label5_open"].quantile(0.10)),
            "p50_exec_excess": float(selected["exec_label5_open"].quantile(0.50)),
            "p90_exec_excess": float(selected["exec_label5_open"].quantile(0.90)),
            "worst_exec_excess": float(selected["exec_label5_open"].min()),
            "best_exec_excess": float(selected["exec_label5_open"].max()),
        })
    df = pd.DataFrame(rows)
    df = df.merge(market_bench.reset_index(), on=["session", "year"], how="left")
    return df


def replay_period_rows(selections: dict[str, pd.DataFrame], market: dict[str, Any], horizon: int) -> tuple[pd.DataFrame, dict[str, int]]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows = []
    rejects = {"candidate_total": 0, "selected_total": 0, "entry_limit_up_or_missing": 0, "exit_limit_down_or_missing": 0, "empty_periods": 0}
    for session, selected in selections.items():
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets = []
        holds = []
        for sym in selected["instrument"].astype(str):
            rejects["candidate_total"] += 1
            col = symbol_index.get(sym)
            if col is None:
                rejects["entry_limit_up_or_missing"] += 1
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not is_entry_ok(entry, preclose, vol):
                rejects["entry_limit_up_or_missing"] += 1
                continue
            sell_idx = exit_idx
            sell_stop = min(len(dates), exit_idx + 6)
            while sell_idx < sell_stop:
                sell_open = float(arr["open"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if is_exit_ok(sell_open, sell_preclose, sell_vol):
                    break
                sell_idx += 1
            if sell_idx >= sell_stop:
                rejects["exit_limit_down_or_missing"] += 1
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        period_return = float(np.mean(rets)) if rets else 0.0
        if not rets:
            rejects["empty_periods"] += 1
        rejects["selected_total"] += len(rets)
        rows.append({
            "session": session,
            "period_return": period_return,
            "selected_count_replay": int(len(rets)),
            "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
            "loss_ratio": float(np.mean(np.asarray(rets) < 0.0)) if rets else 0.0,
            "p10_realized_return": float(np.quantile(rets, 0.10)) if rets else 0.0,
            "p90_realized_return": float(np.quantile(rets, 0.90)) if rets else 0.0,
        })
    return pd.DataFrame(rows), rejects


def is_entry_ok(entry: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(entry) and entry > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and entry / preclose - 1.0 < 0.095)


def is_exit_ok(exit_open: float, preclose: float, volume: float) -> bool:
    return bool(np.isfinite(exit_open) and exit_open > 0 and np.isfinite(preclose) and preclose > 0 and np.isfinite(volume) and volume > 0 and exit_open / preclose - 1.0 > -0.095)


def summarize_leakage(df: pd.DataFrame, rejects: dict[str, int]) -> dict[str, Any]:
    nav = np.cumprod(1.0 + df["period_return"].fillna(0.0).to_numpy(dtype=float))
    peak = np.maximum.accumulate(nav) if len(nav) else np.asarray([1.0])
    annual = {}
    for year, group in df.groupby("year"):
        annual[str(int(year))] = {
            "account_return": float(np.prod(1.0 + group["period_return"].fillna(0.0).to_numpy(dtype=float)) - 1.0),
            "mean_exec_excess": float(group["mean_exec_excess"].mean()),
            "mean_exec_raw_proxy": float(group["mean_exec_raw_proxy"].mean()),
            "mean_period_return": float(group["period_return"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "avg_selected_count": float(group["selected_count_replay"].mean()),
        }
    account_final = float(nav[-1]) if len(nav) else 1.0
    mean_period = float(df["period_return"].mean()) if len(df) else 0.0
    arith_projection = float((1.0 + mean_period) ** len(df)) if len(df) else 1.0
    label_projection = float((1.0 + df["mean_exec_raw_proxy"].mean()) ** len(df)) if len(df) else 1.0
    hit = df["period_return"].to_numpy(dtype=float) > 0.0
    return {
        "periods": int(len(df)),
        "final_multiple": account_final,
        "total_return": float(account_final - 1.0),
        "max_drawdown_period": float(np.min(nav / peak - 1.0)) if len(nav) else 0.0,
        "all_years_positive": bool(all(v["account_return"] > 0 for v in annual.values())),
        "annual": annual,
        "mean_label_excess": float(df["mean_label_excess"].mean()),
        "mean_exec_excess": float(df["mean_exec_excess"].mean()),
        "mean_raw_open": float(df["mean_raw_open"].mean()),
        "mean_exec_raw_proxy": float(df["mean_exec_raw_proxy"].mean()),
        "mean_period_return": mean_period,
        "median_period_return": float(df["period_return"].median()),
        "period_hit_rate": float(np.mean(hit)) if len(hit) else 0.0,
        "period_p10": float(df["period_return"].quantile(0.10)),
        "period_p90": float(df["period_return"].quantile(0.90)),
        "entry_ok": float(df["entry_ok"].mean()),
        "exit_ok_label": float(df["exit_ok"].mean()),
        "avg_selected_count": float(df["selected_count_replay"].mean()),
        "avg_hold_days": float(df.loc[df["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (df["avg_hold_days"] > 0).any() else 0.0,
        "avg_overlap_prev": float(df["overlap_prev"].dropna().mean()) if df["overlap_prev"].notna().any() else 0.0,
        "label_to_period_gap": float(df["mean_exec_raw_proxy"].mean() - df["period_return"].mean()),
        "compounding_drag_vs_arith": float(arith_projection - account_final),
        "proxy_to_account_gap_multiple": float(label_projection - account_final),
        "rejects": rejects,
    }


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    rows = []
    for name, data in result["leakage"].items():
        annual = data.get("annual", {})
        rows.append((name, data["final_multiple"], annual.get("2026", {}).get("account_return", np.nan), data["mean_exec_excess"], data["label_to_period_gap"]))
    rows.sort(key=lambda item: item[1], reverse=True)
    for name, multiple, ret2026, mean_exec, gap in rows[:14]:
        print(f"{name}: final={multiple:.2f}x 2026={ret2026:+.4f} mean_exec_excess={mean_exec:+.6f} gap={gap:+.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
