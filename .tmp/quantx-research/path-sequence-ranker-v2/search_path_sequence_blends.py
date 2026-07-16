from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
V2_PATH = REPO_ROOT / ".tmp/quantx-research/path-sequence-ranker-v2/analyze_path_sequence_ranker_v2.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_module("path_sequence_ranker_v2_search_reuse", V2_PATH)
V1 = V2.V1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2.parquet")
    parser.add_argument("--prepared-cache", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2_with_aux.parquet")
    parser.add_argument("--provider", default="data/qlib_data_fixed")
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--load-start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--output", default=".tmp/quantx-research/path-sequence-ranker-v2/path_sequence_blend_grid_search.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored = load_prepared_scores(args)
    args_for_market = argparse.Namespace(provider=args.provider, raw_dir=args.raw_dir, load_start=args.load_start, end=args.end)
    market = V1.EXP124.load_market(args_for_market)
    candidates = fast_rank_candidates(scored)
    shortlist = select_shortlist(candidates, args.top_n)
    accounts = replay_shortlist(scored, market, args.horizon, shortlist)
    result = {
        "status": "grid_search_complete",
        "experiment": "path_sequence_ranker_v2_blend_grid_search",
        "candidate_count": len(candidates),
        "shortlist_count": len(shortlist),
        "fast_top": candidates[:50],
        "account_results": accounts,
        "config": {
            "cache": args.cache,
            "prepared_cache": args.prepared_cache,
            "score_parts": SCORE_PARTS,
            "weight_grid": "coarse integer tenths plus focused 0.05 grid around the base/path_ev/path_spread/exec_cls stable region, plus selected hand anchors",
            "selection_rule": "fast screen on 2022-2025 due5 proxy with all dev years positive, then formal replay for shortlist; 2026 is reported, not used for ranking shortlist except tie fields.",
            "causality": "All scores are already walk-forward OOS or neutral for unavailable years. Grid weights are selected on 2022-2025 proxy only; 2026 remains forward evaluation.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print(checksum)
    for row in sorted(accounts, key=lambda r: r["dev_final_2022_2025"], reverse=True)[:30]:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


SCORE_PARTS = ["base", "path_ev", "path_spread", "path_topq", "exec_cls"]


def prepare_scores(scored: pd.DataFrame) -> pd.DataFrame:
    out = V2.add_exec_aware_scores(scored, min_train_rows=30000)[0]
    out = V2.add_blend_scores(out)
    rename = {
        "score_base_pct": "part_base",
        "score_path_ev_pct": "part_path_ev",
        "score_path_spread_pct": "part_path_spread",
        "score_path_topq_pct": "part_path_topq",
        "score_exec_cls_pct": "part_exec_cls",
    }
    return out.rename(columns=rename)


def load_prepared_scores(args: argparse.Namespace) -> pd.DataFrame:
    prepared = Path(args.prepared_cache)
    if prepared.exists():
        return pd.read_parquet(prepared).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    scored = pd.read_parquet(args.cache).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    scored = prepare_scores(scored)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(prepared, index=False)
    summary = {
        "prepared_cache": str(prepared),
        "prepared_checksum": "sha256:" + hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "rows": int(len(scored)),
    }
    prepared.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scored


def grid_weights() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ints in product(range(11), repeat=len(SCORE_PARTS)):
        if sum(ints) != 10:
            continue
        weights = {part: value / 10.0 for part, value in zip(SCORE_PARTS, ints)}
        if weights["path_topq"] > 0.7 or weights["exec_cls"] > 0.4:
            continue
        if weights["path_ev"] + weights["path_spread"] + weights["path_topq"] < 0.5:
            continue
        rows.append(weights)
    anchors = [
        {"base": 0.10, "path_ev": 0.20, "path_spread": 0.35, "path_topq": 0.20, "exec_cls": 0.15},
        {"base": 0.15, "path_ev": 0.15, "path_spread": 0.40, "path_topq": 0.15, "exec_cls": 0.15},
        {"base": 0.10, "path_ev": 0.30, "path_spread": 0.30, "path_topq": 0.15, "exec_cls": 0.15},
        {"base": 0.20, "path_ev": 0.20, "path_spread": 0.30, "path_topq": 0.20, "exec_cls": 0.10},
    ]
    fine_values = [i / 20.0 for i in range(21)]
    for base in fine_values:
        if not 0.25 <= base <= 0.50:
            continue
        for path_ev in fine_values:
            if not 0.25 <= path_ev <= 0.50:
                continue
            for path_spread in fine_values:
                if not 0.0 <= path_spread <= 0.30:
                    continue
                for path_topq in fine_values:
                    if not 0.0 <= path_topq <= 0.25:
                        continue
                    exec_cls = round(1.0 - base - path_ev - path_spread - path_topq, 10)
                    if exec_cls < 0.0 or exec_cls > 0.25:
                        continue
                    if path_ev + path_spread + path_topq < 0.50:
                        continue
                    rows.append({"base": base, "path_ev": path_ev, "path_spread": path_spread, "path_topq": path_topq, "exec_cls": exec_cls})
    rows.extend(anchors)
    seen = set()
    unique: list[dict[str, float]] = []
    for row in rows:
        key = tuple((part, round(row[part], 4)) for part in SCORE_PARTS)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def fast_rank_candidates(scored: pd.DataFrame) -> list[dict[str, Any]]:
    due = V1.due_sessions(scored)
    work = scored[scored["session"].isin(due)].copy()
    results: list[dict[str, Any]] = []
    part_cols = {part: f"part_{part}" for part in SCORE_PARTS}
    for idx, weights in enumerate(grid_weights()):
        score = np.zeros(len(work), dtype=np.float64)
        for part, weight in weights.items():
            if weight:
                score += weight * one_dim(work, part_cols[part])
        tmp = work[["session", "year", "instrument", "raw5_open", "entry_ok", "exit_ok"]].copy()
        tmp["score"] = score
        selected = tmp.sort_values(["session", "score", "instrument"], ascending=[True, False, True]).groupby("session", sort=False).head(20)
        selected["proxy_return"] = np.where((selected["entry_ok"] > 0.5) & (selected["exit_ok"] > 0.5), selected["raw5_open"] - V1.COST, np.nan)
        period = selected.groupby(["session", "year"], sort=True)["proxy_return"].mean().fillna(0.0).reset_index()
        annual = {str(int(y)): float(np.prod(1.0 + g["proxy_return"].to_numpy(dtype=float)) - 1.0) for y, g in period.groupby("year")}
        dev_final = float(np.prod([1.0 + annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025)]))
        dev_min = min(annual.get(str(y), -999.0) for y in (2022, 2023, 2024, 2025))
        values = [annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025)]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        results.append({
            "id": f"grid_{idx:04d}",
            "weights": weights,
            "dev_final_proxy": dev_final,
            "dev_min_annual_proxy": float(dev_min),
            "dev_sharpe_proxy": float(mean / std) if std > 0 else None,
            "forward_2026_proxy": float(annual.get("2026", 0.0)),
            "annual_proxy": annual,
        })
    results.sort(key=lambda r: (r["dev_min_annual_proxy"] > 0.0, r["dev_final_proxy"], r.get("dev_sharpe_proxy") or -999.0), reverse=True)
    return results


def select_shortlist(candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    positive = [r for r in candidates if r["dev_min_annual_proxy"] > 0.0]
    pool = positive if positive else candidates
    by_final = sorted(pool, key=lambda r: r["dev_final_proxy"], reverse=True)[:top_n]
    by_sharpe = sorted(pool, key=lambda r: r.get("dev_sharpe_proxy") or -999.0, reverse=True)[: max(20, top_n // 4)]
    seen = set()
    out: list[dict[str, Any]] = []
    for row in by_final + by_sharpe:
        key = row["id"]
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def replay_shortlist(scored: pd.DataFrame, market: dict[str, Any], horizon: int, shortlist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    due = V1.due_sessions(scored)
    work = scored[scored["session"].isin(due)].copy()
    part_cols = {part: f"part_{part}" for part in SCORE_PARTS}
    rows: list[dict[str, Any]] = []
    for item in shortlist:
        score = np.zeros(len(work), dtype=np.float64)
        for part, weight in item["weights"].items():
            if weight:
                score += weight * one_dim(work, part_cols[part])
        tmp = work[["session", "instrument"]].copy()
        tmp["score"] = score
        selections = {
            session: list(group.sort_values(["score", "instrument"], ascending=[False, True]).head(20)["instrument"].astype(str))
            for session, group in tmp.groupby("session", sort=True)
        }
        account = V1.EXP129.replay_open_selection(selections, market, horizon)
        annual = {str(k): float(v) for k, v in account.get("annual_returns", {}).items()}
        dev_final = float(np.prod([1.0 + annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025)]))
        full_final = float(dev_final * (1.0 + annual.get("2026", 0.0)))
        values = [annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025, 2026)]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "id": item["id"],
            "weights": item["weights"],
            "dev_final_2022_2025": dev_final,
            "full_final_2022_2026": full_final,
            "return_2026": float(annual.get("2026", 0.0)),
            "annual_returns": annual,
            "all_years_positive": bool(all(v > 0 for v in values)),
            "min_annual_return": float(min(values)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "annual_sharpe_proxy": float(mean / std) if std > 0 else None,
            "rejects": account.get("rejects", {}),
            "fast_screen": item,
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["dev_final_2022_2025"], r["return_2026"], r.get("annual_sharpe_proxy") or -999.0), reverse=True)
    return rows


def one_dim(frame: pd.DataFrame, col: str) -> np.ndarray:
    values = frame[col]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return values.to_numpy(dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
