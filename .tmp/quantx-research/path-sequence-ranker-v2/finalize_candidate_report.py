from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SEARCH_PATH = REPO_ROOT / ".tmp/quantx-research/path-sequence-ranker-v2/search_path_sequence_blends.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SEARCH = load_module("path_sequence_v2_search_for_report", SEARCH_PATH)
V1 = SEARCH.V1


CANDIDATES = {
    "grid_0932_best_dev": {"base": 0.30, "path_ev": 0.45, "path_spread": 0.05, "path_topq": 0.05, "exec_cls": 0.15},
    "grid_0878_balanced_2026": {"base": 0.30, "path_ev": 0.30, "path_spread": 0.20, "path_topq": 0.05, "exec_cls": 0.15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-cache", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2_with_aux.parquet")
    parser.add_argument("--provider", default="data/qlib_data_fixed")
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--load-start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", default=".tmp/quantx-research/path-sequence-ranker-v2/path_sequence_v2_candidate_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored = pd.read_parquet(args.prepared_cache).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    market_args = argparse.Namespace(provider=args.provider, raw_dir=args.raw_dir, load_start=args.load_start, end=args.end)
    market = V1.EXP124.load_market(market_args)
    reports = []
    for name, weights in CANDIDATES.items():
        reports.append(evaluate_candidate(name, weights, scored, market, args.horizon))
    result = {
        "status": "candidate_report_complete",
        "experiment": "path_sequence_ranker_v2_fixed_candidates",
        "candidates": reports,
        "references": {
            "documented_exp40_dev_final_2022_2025": 8.7749,
            "documented_exp40_2026_return": 0.1918,
            "documented_exp40_dev_max_drawdown": -0.3262,
            "rebuild_v1_best_full_blend_topq_25": 8.168796954642778,
        },
        "config": {
            "prepared_cache": args.prepared_cache,
            "prepared_cache_sha256": sha256(Path(args.prepared_cache)),
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "load_start": args.load_start,
            "end": args.end,
            "horizon": args.horizon,
            "rebalance": "due_sessions every 5 scored sessions, T after-close signal, T+1 open execution",
            "selection": "Top20 equal weight from pool200; no TopK shrink and no head-weight capital concentration.",
            "causality": "Base/path scores are walk-forward OOS. Exec classifier is walk-forward by year and neutral for 2022 due to unavailable prior scored history. Fixed candidate weights were chosen from 2022-2025 development screen; 2026 is forward report.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = sha256(output)
    print(checksum)
    for report in reports:
        print(json.dumps(summary_line(report), ensure_ascii=False, sort_keys=True))
    return 0


def evaluate_candidate(name: str, weights: dict[str, float], scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    due = V1.due_sessions(scored)
    work = scored[scored["session"].isin(due)].copy()
    score = np.zeros(len(work), dtype=np.float64)
    for part, weight in weights.items():
        if weight:
            score += weight * SEARCH.one_dim(work, f"part_{part}")
    work["candidate_score"] = score
    selections = {
        session: list(group.sort_values(["candidate_score", "instrument"], ascending=[False, True]).head(20)["instrument"].astype(str))
        for session, group in work.groupby("session", sort=True)
    }
    account = replay_with_curve(selections, market, horizon)
    annual = account["annual_returns"]
    dev_final = float(np.prod([1.0 + annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025)]))
    full_final = float(dev_final * (1.0 + annual.get("2026", 0.0)))
    return {
        "name": name,
        "weights": weights,
        "dev_final_2022_2025": dev_final,
        "full_final_2022_2026": full_final,
        **account,
        "passes_exp40_like_gate": bool(
            dev_final >= 8.7749
            and annual.get("2026", -999.0) >= 0.1918
            and account["avg_selected_count"] >= 18.0
            and account["max_drawdown_period"] >= -0.3262
            and all(value > 0 for value in annual.values())
        ),
    }


def replay_with_curve(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> dict[str, Any]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows = []
    rejects = {"candidate_total": 0, "selected_total": 0, "entry_limit_up_or_missing": 0, "exit_limit_down_or_missing": 0, "empty_periods": 0}
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets = []
        holds = []
        for sym in candidates:
            rejects["candidate_total"] += 1
            col = symbol_index.get(sym)
            if col is None:
                rejects["entry_limit_up_or_missing"] += 1
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry / preclose - 1.0 >= 0.095:
                rejects["entry_limit_up_or_missing"] += 1
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
                rejects["exit_limit_down_or_missing"] += 1
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - V1.COST)
            holds.append(sell_idx - entry_idx)
        period_ret = float(np.mean(rets)) if rets else 0.0
        if not rets:
            rejects["empty_periods"] += 1
        rejects["selected_total"] += len(rets)
        rows.append({"session": session, "year": int(session[:4]), "period_return": period_ret, "selected_count": len(rets), "avg_hold_days": float(np.mean(holds)) if holds else 0.0})
    nav = 1.0
    curve = []
    for row in rows:
        nav *= 1.0 + row["period_return"]
        curve.append({**row, "nav": nav})
    df = pd.DataFrame(rows)
    annual = {str(int(y)): float(np.prod(1.0 + g["period_return"].to_numpy(dtype=float)) - 1.0) for y, g in df.groupby("year")}
    navs = np.asarray([r["nav"] for r in curve], dtype=float)
    peak = np.maximum.accumulate(navs)
    period_returns = df["period_return"].to_numpy(dtype=float)
    period_mean = float(np.mean(period_returns))
    period_std = float(np.std(period_returns, ddof=1))
    annual_values = list(annual.values())
    annual_mean = float(np.mean(annual_values))
    annual_std = float(np.std(annual_values, ddof=1))
    return {
        "final_multiple": float(nav),
        "total_return": float(nav - 1.0),
        "annual_returns": annual,
        "all_years_positive": bool(all(v > 0 for v in annual.values())),
        "max_drawdown_period": float(np.min(navs / peak - 1.0)),
        "avg_selected_count": float(df["selected_count"].mean()),
        "avg_hold_days": float(df.loc[df["avg_hold_days"] > 0, "avg_hold_days"].mean()) if (df["avg_hold_days"] > 0).any() else 0.0,
        "periods": int(len(rows)),
        "period_mean_return": period_mean,
        "period_return_std": period_std,
        "period_sharpe_proxy": float(period_mean / period_std * np.sqrt(252.0 / 5.0)) if period_std > 0 else None,
        "annual_return_mean": annual_mean,
        "annual_return_std": annual_std,
        "annual_sharpe_proxy": float(annual_mean / annual_std) if annual_std > 0 else None,
        "worst_period_return": float(np.min(period_returns)),
        "best_period_return": float(np.max(period_returns)),
        "positive_period_ratio": float(np.mean(period_returns > 0)),
        "rejects": rejects,
        "curve_tail": curve[-5:],
    }


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def summary_line(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": report["name"],
        "dev_final_2022_2025": report["dev_final_2022_2025"],
        "return_2026": report["annual_returns"].get("2026"),
        "full_final_2022_2026": report["full_final_2022_2026"],
        "max_drawdown_period": report["max_drawdown_period"],
        "avg_selected_count": report["avg_selected_count"],
        "avg_hold_days": report["avg_hold_days"],
        "period_sharpe_proxy": report["period_sharpe_proxy"],
        "annual_sharpe_proxy": report["annual_sharpe_proxy"],
        "passes_exp40_like_gate": report["passes_exp40_like_gate"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
