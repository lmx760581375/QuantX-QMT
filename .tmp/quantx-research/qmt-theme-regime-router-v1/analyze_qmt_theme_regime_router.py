from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


VARIANTS = ("reg", "blend", "follower_catchup", "leader_pullback", "leader_continue", "theme_beta_amount")
TOPK = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--leader-count", type=int, default=50)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--min-meta-sessions", type=int, default=80)
    parser.add_argument("--source-script", default=".tmp/quantx-research/qmt-endogenous-leader-follower-v1/analyze_qmt_endogenous_leader_follower.py")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_exp121(Path(args.source_script))
    panel = source.build_panel(args)
    scored = walk_forward_scores(source, panel, args.min_train_rows)
    sessions = sorted(scored["session"].unique()) if not scored.empty else []
    states, exit_dates = load_market_states(args, sessions)
    perf = build_session_performance(scored, exit_dates)
    routed = apply_causal_routers(perf, states, args.min_meta_sessions)
    result = summarize(routed, perf, states, panel, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def load_exp121(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("exp121", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walk_forward_scores(source: Any, panel: pd.DataFrame, min_train_rows: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        scored, _ = source.fit_predict(train, test, int(year))
        rows.append(scored)
        print(json.dumps({"year": int(year), "train_rows": int(len(train)), "test_rows": int(len(test)), "sessions": int(test["session"].nunique())}, ensure_ascii=False), flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_market_states(args: argparse.Namespace, sessions: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    provider = Path(args.provider)
    universe = _build_universe(provider, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(provider)
    calendar = reader.calendar(args.load_start, args.end)
    quotes = reader.features(symbols, ["$open", "$close", "$volume", "$vwap"], args.load_start, args.end)
    close = quotes["$close"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    volume = quotes["$volume"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    vwap = quotes["$vwap"].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float)
    amount = vwap * volume
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    amt_ratio20 = amount / amount.rolling(20, min_periods=10).mean()
    high20 = close.rolling(20, min_periods=10).max()
    date_to_idx = {pd.Timestamp(day).strftime("%Y-%m-%d"): idx for idx, day in enumerate(calendar)}
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
    rows: list[dict[str, Any]] = []
    exit_dates: dict[str, str] = {}
    for session in sessions:
        idx = date_to_idx[session]
        exit_idx = idx + 1 + args.horizon
        if exit_idx < len(dates):
            exit_dates[session] = dates[exit_idx]
        r5 = ret5.iloc[idx]
        r20 = ret20.iloc[idx]
        r60 = ret60.iloc[idx]
        amt = amount.iloc[idx]
        amt_ratio = amt_ratio20.iloc[idx]
        near_high = close.iloc[idx] / high20.iloc[idx]
        leader_score = r20.rank(pct=True) + 0.35 * np.log1p(amt).rank(pct=True) + 0.25 * r5.rank(pct=True)
        leaders = leader_score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(args.leader_count).index
        rows.append({
            "session": session,
            "year": int(session[:4]),
            "mkt_ret5_median": finite_median(r5),
            "mkt_ret20_median": finite_median(r20),
            "mkt_ret60_median": finite_median(r60),
            "mkt_breadth5": finite_mean(r5 > 0),
            "mkt_breadth20": finite_mean(r20 > 0),
            "mkt_disp5": finite_std(r5),
            "mkt_disp20": finite_std(r20),
            "mkt_amount_disp": finite_std(np.log1p(amt.replace([np.inf, -np.inf], np.nan))),
            "mkt_amt_ratio20_median": finite_median(amt_ratio),
            "mkt_near_high20": finite_mean(near_high > 0.95),
            "leader_ret20_mean": finite_mean(r20.loc[leaders]) if len(leaders) else 0.0,
            "leader_ret5_mean": finite_mean(r5.loc[leaders]) if len(leaders) else 0.0,
            "leader_amount_rank_mean": finite_mean(np.log1p(amt).rank(pct=True).loc[leaders]) if len(leaders) else 0.0,
            "leader_vs_mkt20": (finite_mean(r20.loc[leaders]) - finite_median(r20)) if len(leaders) else 0.0,
        })
    states = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return states, exit_dates


def finite_median(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0


def finite_std(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if np.isfinite(arr).any() else 0.0


def build_session_performance(scored: pd.DataFrame, exit_dates: dict[str, str]) -> pd.DataFrame:
    score_cols = {
        "reg": "score_reg",
        "blend": "score_blend",
        "follower_catchup": "follower_catchup",
        "leader_pullback": "leader_pullback",
        "leader_continue": "leader_continue",
        "theme_beta_amount": "theme_beta_amount",
    }
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in score_cols.items():
            selected = group.sort_values([score_col, "instrument"], ascending=[False, True]).head(TOPK)
            rows.append({
                "session": session,
                "exit_date": exit_dates.get(session, session),
                "year": int(str(session)[:4]),
                "variant": variant,
                "mean_label5": float(selected["label5"].mean()),
                "mean_raw5": float(selected["raw5"].mean()),
                "count": int(len(selected)),
            })
    return pd.DataFrame(rows)


def apply_causal_routers(perf: pd.DataFrame, states: pd.DataFrame, min_meta_sessions: int) -> pd.DataFrame:
    state_cols = [c for c in states.columns if c not in {"session", "year"}]
    state_by_session = states.set_index("session")
    rows: list[dict[str, Any]] = []
    sessions = sorted(perf["session"].unique())
    for session in sessions:
        current = perf[perf["session"] == session].set_index("variant")
        history = perf[perf["exit_date"] <= session]
        choices: dict[str, str] = {}
        if history.empty:
            choices["expanding_best"] = "reg"
            choices["rolling20_best"] = "reg"
            choices["ewma20_best"] = "reg"
            choices["meta_lgbm"] = "reg"
        else:
            choices["expanding_best"] = best_by_mean(history)
            last_sessions = sorted(history["session"].unique())[-20:]
            choices["rolling20_best"] = best_by_mean(history[history["session"].isin(last_sessions)])
            choices["ewma20_best"] = best_by_ewma(history, half_life=20.0)
            choices["meta_lgbm"] = meta_choice(history, state_by_session, state_cols, session, min_meta_sessions)
        choices["oracle_noncausal"] = str(current["mean_label5"].idxmax())
        for router, variant in choices.items():
            row = current.loc[variant]
            rows.append({
                "session": session,
                "year": int(str(session)[:4]),
                "router": router,
                "variant": variant,
                "mean_label5": float(row["mean_label5"]),
                "mean_raw5": float(row["mean_raw5"]),
                "count": int(row["count"]),
                "history_sessions": int(history["session"].nunique()) if not history.empty else 0,
            })
        for variant in VARIANTS:
            row = current.loc[variant]
            rows.append({
                "session": session,
                "year": int(str(session)[:4]),
                "router": f"fixed_{variant}",
                "variant": variant,
                "mean_label5": float(row["mean_label5"]),
                "mean_raw5": float(row["mean_raw5"]),
                "count": int(row["count"]),
                "history_sessions": int(history["session"].nunique()) if not history.empty else 0,
            })
    return pd.DataFrame(rows)


def best_by_mean(history: pd.DataFrame) -> str:
    means = history.groupby("variant")["mean_label5"].mean()
    return str(means.reindex(VARIANTS).fillna(-1e9).idxmax())


def best_by_ewma(history: pd.DataFrame, half_life: float) -> str:
    sessions = sorted(history["session"].unique())
    age = {session: len(sessions) - 1 - idx for idx, session in enumerate(sessions)}
    weights = history["session"].map(lambda s: 0.5 ** (age[s] / half_life)).astype(float)
    tmp = history.copy()
    tmp["weight"] = weights
    scores = tmp.groupby("variant").apply(lambda g: float(np.average(g["mean_label5"], weights=g["weight"])), include_groups=False)
    return str(scores.reindex(VARIANTS).fillna(-1e9).idxmax())


def meta_choice(history: pd.DataFrame, state_by_session: pd.DataFrame, state_cols: list[str], session: str, min_meta_sessions: int) -> str:
    if history["session"].nunique() < min_meta_sessions or session not in state_by_session.index:
        return best_by_mean(history)
    train = history.merge(state_by_session[state_cols], left_on="session", right_index=True, how="left").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = pd.get_dummies(train, columns=["variant"], prefix="variant")
    variant_cols = [f"variant_{v}" for v in VARIANTS]
    for col in variant_cols:
        if col not in train:
            train[col] = 0
    features = state_cols + variant_cols
    model = LGBMRegressor(n_estimators=80, learning_rate=0.05, num_leaves=15, max_depth=3, min_child_samples=20, reg_alpha=1.0, reg_lambda=5.0, random_state=int(session[:4]), n_jobs=2, verbosity=-1)
    model.fit(train[features], train["mean_label5"])
    current_rows = []
    base = state_by_session.loc[session, state_cols].to_dict()
    for variant in VARIANTS:
        row = dict(base)
        for col in variant_cols:
            row[col] = 1 if col == f"variant_{variant}" else 0
        row["variant"] = variant
        current_rows.append(row)
    current = pd.DataFrame(current_rows)
    pred = model.predict(current[features])
    return str(current.loc[int(np.argmax(pred)), "variant"])


def summarize(routed: pd.DataFrame, perf: pd.DataFrame, states: pd.DataFrame, panel: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    routers: dict[str, Any] = {}
    for router, group in routed.groupby("router"):
        routers[str(router)] = {
            "daily_count": int(len(group)),
            "mean_daily_label5": float(group["mean_label5"].mean()),
            "mean_daily_raw5": float(group["mean_raw5"].mean()),
            "positive_label5_ratio": float((group["mean_label5"] > 0).mean()),
            "avg_selected_count": float(group["count"].mean()),
            "variant_usage": {str(k): int(v) for k, v in group["variant"].value_counts().to_dict().items()},
            "by_year": {
                str(int(year)): {
                    "mean_daily_label5": float(yg["mean_label5"].mean()),
                    "mean_daily_raw5": float(yg["mean_raw5"].mean()),
                    "positive_label5_ratio": float((yg["mean_label5"] > 0).mean()),
                    "variant_usage": {str(k): int(v) for k, v in yg["variant"].value_counts().to_dict().items()},
                }
                for year, yg in group.groupby("year")
            },
        }
    return {
        "status": "diagnostic_complete",
        "routers": routers,
        "oracle_note": "oracle_noncausal is only an upper-bound diagnostic and is not a valid tradable selector.",
        "base_variant_performance": summarize_base_perf(perf),
        "state_rows": int(len(states)),
        "panel_rows": int(len(panel)),
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "leader_count": args.leader_count,
            "corr_window": args.corr_window,
            "min_meta_sessions": args.min_meta_sessions,
            "topk": TOPK,
            "variants": VARIANTS,
            "causality": "Base stock scores are annual walk-forward. Router choices at session T use only variant outcomes with exit_date <= T plus market states observable through completed T. oracle_noncausal is reported only as impossible upper bound.",
            "scope": "QMT retrievable daily OHLCV/VWAP data only; no static industry/concept table, news, announcement, LHB, ETF flow, northbound, financing, or event/message data.",
        },
    }


def summarize_base_perf(perf: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, group in perf.groupby("variant"):
        out[str(variant)] = {
            "mean_daily_label5": float(group["mean_label5"].mean()),
            "mean_daily_raw5": float(group["mean_raw5"].mean()),
            "by_year": {str(int(y)): float(yg["mean_label5"].mean()) for y, yg in group.groupby("year")},
        }
    return out


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(checksum)
    rows = []
    for router, data in result["routers"].items():
        by_year = data["by_year"]
        rows.append((router, data["mean_daily_label5"], by_year.get("2026", {}).get("mean_daily_label5", np.nan), data["variant_usage"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    for router, mean_all, mean_2026, usage in rows:
        print(f"{router}: all={mean_all:+.6f} 2026={mean_2026:+.6f} usage={usage}")


if __name__ == "__main__":
    raise SystemExit(main())
