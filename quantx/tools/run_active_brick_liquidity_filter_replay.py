from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from quantx.core.research import active_brick_replay as base


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
SCORES_PATH = ROOT / "walk_forward_active_brick_scores.parquet"
OUT_JSON = ROOT / "active_brick_liquidity_filter_replay.json"
OUT_MD = ROOT / "active_brick_liquidity_filter_replay.md"
OUT_TRADES = ROOT / "active_brick_liquidity_filter_replay_trades.parquet"
OUT_NAV = ROOT / "active_brick_liquidity_filter_replay_nav.parquet"
RANDOM_RUNS = 50

FILTER_SPECS = [
    {"name": "all_ml_base_consensus", "score_col": "ml_base_consensus_score", "rules": []},
    {"name": "turnover_060_100_ml_base_consensus", "score_col": "ml_base_consensus_score", "rules": [("turnover_rate_rank_pct", 0.60, 1.00)]},
    {"name": "turnover_040_080_momentum", "score_col": "momentum_score", "rules": [("turnover_rate_rank_pct", 0.40, 0.80)]},
    {"name": "turnover_020_060_mv_020_080_rule", "score_col": "rule_score", "rules": [("turnover_rate_rank_pct", 0.20, 0.60), ("free_float_mv_rank_pct", 0.20, 0.80)]},
    {"name": "turnover_020_060_active_proxy_020_080_rule", "score_col": "rule_score", "rules": [("turnover_rate_rank_pct", 0.20, 0.60), ("active_free_mv_proxy_rank_pct", 0.20, 0.80)]},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "Replay active brick liquidity filters.")
    parser.add_argument("--scores-path", default=str(SCORES_PATH))
    parser.add_argument("--json-output", default=str(OUT_JSON))
    parser.add_argument("--markdown-output", default=str(OUT_MD))
    parser.add_argument("--trades-output", default=str(OUT_TRADES))
    parser.add_argument("--nav-output", default=str(OUT_NAV))
    parser.add_argument("--random-runs", type=int, default=RANDOM_RUNS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report, trades, nav = run_replay(Path(args.scores_path), random_runs=args.random_runs)
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    trades_output = Path(args.trades_output)
    nav_output = Path(args.nav_output)
    for path in (json_output, markdown_output, trades_output, nav_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(trades_output, index=False)
    nav.to_parquet(nav_output, index=False)
    report["outputs"] = {
        "json": str(json_output),
        "markdown": str(markdown_output),
        "trades": str(trades_output),
        "nav": str(nav_output),
    }
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"ok=True configs={len(report['summaries'])} output={json_output}")
        for row in sorted(report["summaries"], key=lambda item: item.get("total_return") or -999, reverse=True):
            print(
                f"{row['config']} return={row.get('total_return'):.4f} "
                f"mdd={row.get('max_drawdown'):.4f} buys={row.get('buys')} sells={row.get('sells')}"
            )
    return 0


def run_replay(scores_path: Path, *, random_runs: int = RANDOM_RUNS) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    panel = base.load_panel(base.PANEL_PATH)
    panel = base.add_trade_state(panel)
    active = base.load_active_value_daily(base.ACTIVE_VALUE_PATH)
    panel = base.attach_active_value_daily(panel, active, "active_core_amount")
    panel = base.add_forward_returns(panel)
    context = base.build_replay_context(panel)

    scored = pd.read_parquet(scores_path)
    scored["datetime"] = pd.to_datetime(scored["datetime"])
    summaries = []
    trade_parts = []
    nav_parts = []
    for spec in FILTER_SPECS:
        signals = apply_filter(scored, spec).dropna(subset=[spec["score_col"]]).copy()
        config = {
            "name": f"liquidity_{spec['name']}_top3_risk_stop6",
            "topk": 3,
            "score_col": spec["score_col"],
            "mode": "risk",
            "hold_days": 10,
            "require_ret4": True,
            "force_market_exit": False,
            "candidate_pool": "brick",
            "stop_loss": 0.06,
        }
        replay = base.replay_portfolio(context, signals, config)
        summary = dict(replay["summary"])
        summary["filter_rules"] = spec["rules"]
        summary["input_rows"] = int(len(signals))
        summary["input_dates"] = int(signals["datetime"].nunique()) if not signals.empty else 0
        summaries.append(summary)
        if not replay["trades"].empty:
            trade_parts.append(replay["trades"].assign(config=config["name"]))
        if not replay["nav"].empty:
            nav_parts.append(replay["nav"].assign(config=config["name"]))
    random_report = build_random_baseline(context, scored, random_runs=random_runs)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    nav = pd.concat(nav_parts, ignore_index=True) if nav_parts else pd.DataFrame()
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scores_path": str(scores_path),
        "definition": {
            "entry": "signal T, buy T+1 open",
            "exit": "6% stop loss or max holding 10 sessions, using active_brick_replay risk mode",
            "topk": 3,
            "filters": FILTER_SPECS,
        },
        "summaries": summaries,
        "random_baseline": random_report,
        "liquidity_gate": evaluate_liquidity_gate(summaries, random_report),
    }, trades, nav


def evaluate_liquidity_gate(summaries: list[dict], random_report: dict) -> dict:
    name = "liquidity_turnover_020_060_mv_020_080_rule_top3_risk_stop6"
    summary = next((row for row in summaries if row.get("config") == name), {})
    yearly = summary.get("yearly_return") or {}
    total_return = summary.get("total_return")
    max_drawdown = summary.get("max_drawdown")
    buys = summary.get("buys")
    random_q95 = random_report.get("q95")
    checks = {
        "total_positive": value_gt(total_return, 0),
        "year_2025_positive": value_gt(yearly.get("2025"), 0),
        "year_2026_positive": value_gt(yearly.get("2026"), 0),
        "max_drawdown_above_minus_20pct": value_gt(max_drawdown, -0.20),
        "above_same_pool_random_q95": value_ge(total_return, random_q95),
        "sample_at_least_60_buys": value_ge(buys, 60),
    }
    return {
        "status": "OPEN" if all(checks.values()) else "CLOSED",
        "candidate": name,
        "checks": checks,
        "evidence": {
            "total_return": total_return,
            "yearly_return": yearly,
            "max_drawdown": max_drawdown,
            "buys": buys,
            "same_pool_random_q95": random_q95,
            "random_runs": random_report.get("runs"),
        },
    }


def value_gt(value, threshold: float) -> bool:
    return value is not None and float(value) > float(threshold)


def value_ge(value, threshold) -> bool:
    return value is not None and threshold is not None and float(value) >= float(threshold)


def build_random_baseline(context: dict, scored: pd.DataFrame, *, random_runs: int) -> dict:
    target = next(spec for spec in FILTER_SPECS if spec["name"] == "turnover_020_060_mv_020_080_rule")
    signals = apply_filter(scored, target).copy()
    strategy_config = f"liquidity_{target['name']}_top3_risk_stop6"
    summaries = []
    for seed in range(max(0, int(random_runs))):
        score_col = f"random_score_{seed}"
        signals[score_col] = seeded_random_score(signals, seed)
        config = {
            "name": f"liquidity_{target['name']}_random_{seed}_top3_risk_stop6",
            "topk": 3,
            "score_col": score_col,
            "mode": "risk",
            "hold_days": 10,
            "require_ret4": True,
            "force_market_exit": False,
            "candidate_pool": "brick",
            "stop_loss": 0.06,
        }
        summaries.append(base.replay_portfolio(context, signals, config)["summary"])
    returns = pd.Series([row.get("total_return") for row in summaries], dtype=float).dropna()
    strategy_summary = None
    if strategy_config:
        strategy_summary = strategy_config
    return {
        "filter": target["name"],
        "runs": int(len(returns)),
        "mean": float(returns.mean()) if len(returns) else None,
        "median": float(returns.median()) if len(returns) else None,
        "q05": float(returns.quantile(0.05)) if len(returns) else None,
        "q95": float(returns.quantile(0.95)) if len(returns) else None,
        "positive_ratio": float((returns > 0).mean()) if len(returns) else None,
        "max": float(returns.max()) if len(returns) else None,
        "min": float(returns.min()) if len(returns) else None,
        "strategy_config": strategy_summary,
    }


def seeded_random_score(frame: pd.DataFrame, seed: int) -> pd.Series:
    keys = frame[["datetime", "instrument"]].astype(str).copy()
    keys["seed"] = str(seed)
    hashed = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
    return (hashed / np.float64(np.iinfo(np.uint64).max)).astype(float)


def apply_filter(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, low, high in spec["rules"]:
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(float("nan"), index=frame.index)
        mask &= values.ge(low) & values.le(high)
    return frame[mask].copy()


def render_markdown(report: dict) -> str:
    lines = [
        "# Active Brick Liquidity Filter Replay",
        "",
        f"- Scores: `{report['scores_path']}`",
        f"- Entry: {report['definition']['entry']}",
        f"- Exit: {report['definition']['exit']}",
        "",
        "## Summaries",
        "",
    ]
    for row in sorted(report["summaries"], key=lambda item: item.get("total_return") or -999, reverse=True):
        lines.append(
            f"- `{row['config']}`: return={row.get('total_return'):.2%}, "
            f"mdd={row.get('max_drawdown'):.2%}, buys={row.get('buys')}, "
            f"sells={row.get('sells')}, input_dates={row.get('input_dates')}"
        )
    random_report = report.get("random_baseline") or {}
    if random_report.get("runs"):
        lines.extend([
            "",
            "## Random Baseline",
            "",
            f"- filter: `{random_report['filter']}`",
            f"- runs: {random_report['runs']}, mean={random_report['mean']:.2%}, q95={random_report['q95']:.2%}, max={random_report['max']:.2%}",
        ])
    gate = report.get("liquidity_gate") or {}
    if gate:
        lines.extend(["", "## Gate", "", f"- status: `{gate.get('status')}`"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
