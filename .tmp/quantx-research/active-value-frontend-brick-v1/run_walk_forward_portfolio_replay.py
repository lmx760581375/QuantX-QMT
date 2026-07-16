from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
BASE_SCRIPT = ROOT / "run_active_value_reversal_portfolio_research.py"
ML_SCRIPT = ROOT / "run_walk_forward_active_brick_ml.py"
SCORES_PATH = ROOT / "walk_forward_active_brick_scores.parquet"
OUT_JSON = ROOT / "walk_forward_portfolio_replay.json"
OUT_MD = ROOT / "walk_forward_portfolio_replay.md"
OUT_TRADES = ROOT / "walk_forward_portfolio_replay_trades.parquet"
OUT_NAV = ROOT / "walk_forward_portfolio_replay_nav.parquet"


def main() -> int:
    base = load_base_module()
    panel = base.load_panel(base.PANEL_PATH)
    panel = base.add_trade_state(panel)
    active = base.load_active_value_daily(base.ACTIVE_VALUE_PATH)
    panel = base.attach_active_value_daily(panel, active, "active_core_amount")
    panel = base.add_forward_returns(panel)
    context = base.build_replay_context(panel)
    market_state = build_exit_market_state()
    context["market_state_by_date"] = market_state.set_index("datetime")
    scored = pd.read_parquet(SCORES_PATH)
    scored["datetime"] = pd.to_datetime(scored["datetime"])
    methods = {
        "ml": "ml_score",
        "ml_rank": "ml_rank_score",
        "ml_base": "ml_base_score",
        "ml_base_rank": "ml_base_rank_score",
        "ml_consensus": "ml_consensus_score",
        "ml_base_consensus": "ml_base_consensus_score",
        "rule": "rule_score",
        "reversal": "reversal_value",
        "momentum": "momentum_score",
    }
    summaries = []
    trades = []
    navs = []
    replay_results = []
    for method, score_col in methods.items():
        method_scored = scored.dropna(subset=[score_col]).copy()
        if method in {"ml", "ml_base"}:
            method_scored = method_scored[method_scored[score_col] > 0]
        for mode, hold_days, topk in (
            ("fixed", 3, 3), ("fixed", 5, 3), ("fixed", 10, 3),
            ("fixed", 5, 5), ("fixed", 10, 5), ("fixed", 10, 10),
            ("brick", 10, 3),
        ):
            config = replay_config(method, score_col, mode, hold_days=hold_days, topk=topk)
            replay = base.replay_portfolio(context, method_scored, config)
            replay_results.append(replay)
            summaries.append(replay["summary"])
            if not replay["trades"].empty:
                trades.append(replay["trades"].assign(method=method, mode=mode, config=config["name"]))
            if not replay["nav"].empty:
                navs.append(replay["nav"].assign(method=method, mode=mode, config=config["name"]))

    consensus = scored.dropna(subset=["ml_consensus_score"]).copy()
    for config in risk_exit_configs():
        replay = base.replay_portfolio(context, consensus, config)
        replay_results.append(replay)
        summaries.append(replay["summary"])
        if not replay["trades"].empty:
            trades.append(replay["trades"].assign(method="ml_consensus", mode="risk", config=config["name"]))
        if not replay["nav"].empty:
            navs.append(replay["nav"].assign(method="ml_consensus", mode="risk", config=config["name"]))

    random_by_config = build_random_distributions(base, context, scored)
    trade_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    nav_frame = pd.concat(navs, ignore_index=True) if navs else pd.DataFrame()
    trade_frame.to_parquet(OUT_TRADES, index=False)
    nav_frame.to_parquet(OUT_NAV, index=False)
    report = {
        "definition": {
            "entry": "signal T, buy T+1 open",
            "exit": "fixed 3/5/10, brick, or predefined next-open risk exits; one-price limit-down delays exit",
            "capital": float(base.INITIAL_CASH),
            "tested_max_positions": [3, 5, 10],
            "portfolio_awareness": "sell first, then rank only when account slots are available; direct-return ML may leave slots in cash when score <= 0",
            "label_audit": "fwd labels use proportional fees; account replay uses actual lots and minimum commission, so small-position label differences are expected fee effects",
            "costs": {
                "buy_slippage": base.BUY_SLIPPAGE,
                "sell_slippage": base.SELL_SLIPPAGE,
                "commission": base.COMMISSION,
                "stamp_tax": base.STAMP_TAX,
                "minimum_fee": base.MIN_FEE,
                "lot_size": 100,
            },
        },
        "summaries": summaries,
        "random_distributions": summarize_random_distributions(random_by_config, summaries),
        "attribution": base.build_attribution(replay_results, panel),
        "risk_exit_definitions": risk_exit_configs(),
        "trades_path": str(OUT_TRADES),
        "nav_path": str(OUT_NAV),
    }
    report["production_gate"] = evaluate_production_gate(report)
    report["risk_exit_gate"] = evaluate_risk_exit_gate(report)
    report["portfolio_ablation"] = build_portfolio_ablation(nav_frame)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def replay_config(method: str, score_col: str, mode: str, *, hold_days: int, topk: int) -> dict:
    return {
        "name": f"walk_forward_{method}_top{topk}_{f'fixed{hold_days}' if mode == 'fixed' else 'brick_exit'}",
        "topk": topk,
        "score_col": score_col,
        "mode": mode,
        "hold_days": hold_days,
        "require_ret4": True,
        "force_market_exit": False,
        "candidate_pool": "brick",
    }


def risk_exit_configs() -> list[dict]:
    base = {
        "topk": 3,
        "score_col": "ml_consensus_score",
        "mode": "risk",
        "hold_days": 10,
        "require_ret4": True,
        "force_market_exit": False,
        "candidate_pool": "brick",
    }
    variants = [
        ("stop6", {"stop_loss": 0.06}),
        ("take12", {"take_profit": 0.12}),
        ("trail8_6", {"trailing_activation": 0.08, "trailing_drawdown": 0.06}),
        ("breadth", {"exit_breadth": True}),
        ("amv", {"exit_amv": True}),
        ("combined", {
            "stop_loss": 0.06,
            "trailing_activation": 0.08,
            "trailing_drawdown": 0.06,
            "exit_breadth": True,
            "exit_amv": True,
        }),
    ]
    return [{**base, **options, "name": f"walk_forward_ml_consensus_top3_risk_{name}"} for name, options in variants]


def build_exit_market_state() -> pd.DataFrame:
    module = load_module("walk_forward_ml_state", ML_SCRIPT)
    state = module.build_market_state().sort_values("datetime").copy()
    state["right_side_core_ratio_ema20"] = state["right_side_core_ratio"].ewm(
        span=20, adjust=False, min_periods=10
    ).mean()
    state["breadth_healthy"] = (
        state["right_side_core_ratio"].notna()
        & state["right_side_core_ratio_ema20"].notna()
        & (state["right_side_core_ratio"] >= state["right_side_core_ratio_ema20"])
    )
    return state[[
        "datetime", "right_side_core_ratio", "right_side_core_ratio_ema20",
        "breadth_healthy", "amv_quantile_wave", "amv_wave_age",
    ]]


def seeded_random_score(frame: pd.DataFrame, seed: int) -> pd.Series:
    keys = frame[["datetime", "instrument"]].astype(str).copy()
    keys["seed"] = str(seed)
    hashed = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
    return (hashed / np.float64(np.iinfo(np.uint64).max)).astype(float)


def build_random_distributions(base, context: dict, scored: pd.DataFrame) -> dict[str, list[dict]]:
    configs = {
        "top3_fixed10": replay_config("random", "random_score", "fixed", hold_days=10, topk=3),
        "top3_brick10": replay_config("random", "random_score", "brick", hold_days=10, topk=3),
        "top3_risk_stop6": {**risk_exit_configs()[0], "score_col": "random_score"},
    }
    out = {key: [] for key in configs}
    for seed in range(50):
        score_col = f"random_score_{seed}"
        scored[score_col] = seeded_random_score(scored, seed)
        for key, raw_config in configs.items():
            config = dict(raw_config)
            config["name"] = f"walk_forward_random_{seed}_{key}"
            config["score_col"] = score_col
            replay = base.replay_portfolio(
                context,
                scored,
                config,
            )
            out[key].append(replay["summary"])
    return out


def summarize_random_distributions(random_by_config: dict[str, list[dict]], strategy_summaries: list[dict]) -> dict:
    result = {}
    for key, random_summaries in random_by_config.items():
        returns = pd.Series([row.get("total_return") for row in random_summaries], dtype=float).dropna()
        _, mode_token = key.split("_", 1)
        target_mode = "brick" if mode_token.startswith("brick") else "fixed"
        if mode_token.startswith("risk"):
            target_mode = "risk"
        target_hold = 10
        target_topk = 3
        strategy_percentiles = {}
        for row in strategy_summaries:
            if row.get("mode") != target_mode or row.get("topk") != target_topk:
                continue
            if target_mode == "fixed" and f"fixed{target_hold}" not in row.get("config", ""):
                continue
            if target_mode == "risk" and not row.get("config", "").endswith(mode_token):
                continue
            if row.get("total_return") is not None:
                strategy_percentiles[row["config"]] = float((returns <= float(row["total_return"])).mean())
        result[key] = {
            "runs": int(len(returns)),
            "mean": float(returns.mean()),
            "median": float(returns.median()),
            "q05": float(returns.quantile(0.05)),
            "q95": float(returns.quantile(0.95)),
            "positive_ratio": float((returns > 0).mean()),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "strategy_percentiles": strategy_percentiles,
        }
    return result


def evaluate_production_gate(report: dict) -> dict:
    summaries = {row["config"]: row for row in report["summaries"]}
    attribution = {row["config"]: row for row in report["attribution"]}
    fixed_name = "walk_forward_ml_consensus_top3_fixed10"
    brick_name = "walk_forward_ml_consensus_top3_brick_exit"
    fixed = summaries.get(fixed_name, {})
    brick = summaries.get(brick_name, {})
    fixed_alpha = attribution.get(fixed_name, {}).get("alpha_annualized")
    brick_alpha = attribution.get(brick_name, {}).get("alpha_annualized")
    fixed_percentile = report["random_distributions"].get("top3_fixed10", {}).get("strategy_percentiles", {}).get(fixed_name)
    brick_percentile = report["random_distributions"].get("top3_brick10", {}).get("strategy_percentiles", {}).get(brick_name)
    yearly = fixed.get("yearly_return", {})
    checks = {
        "fixed_total_positive": value_gt(fixed.get("total_return"), 0),
        "brick_total_positive": value_gt(brick.get("total_return"), 0),
        "fixed_2025_positive": value_gt(yearly.get("2025"), 0),
        "fixed_2026_positive": value_gt(yearly.get("2026"), 0),
        "fixed_max_drawdown_above_minus_25pct": value_gt(fixed.get("max_drawdown"), -0.25),
        "fixed_factor_alpha_positive": value_gt(fixed_alpha, 0),
        "brick_factor_alpha_positive": value_gt(brick_alpha, 0),
        "fixed_above_random_q95": value_ge(fixed_percentile, 0.95),
        "brick_above_random_q95": value_ge(brick_percentile, 0.95),
    }
    evidence = {
        "fixed_total_return": fixed.get("total_return"),
        "brick_total_return": brick.get("total_return"),
        "fixed_yearly_return": yearly,
        "fixed_max_drawdown": fixed.get("max_drawdown"),
        "fixed_alpha_annualized": fixed_alpha,
        "brick_alpha_annualized": brick_alpha,
        "fixed_random_percentile": fixed_percentile,
        "brick_random_percentile": brick_percentile,
    }
    return {
        "status": "OPEN" if all(checks.values()) else "CLOSED",
        "candidate": "ml_consensus_top3",
        "checks": checks,
        "evidence": evidence,
    }


def evaluate_risk_exit_gate(report: dict) -> dict:
    summaries = {row["config"]: row for row in report["summaries"]}
    attribution = {row["config"]: row for row in report["attribution"]}
    name = "walk_forward_ml_consensus_top3_risk_stop6"
    summary = summaries.get(name, {})
    attr = attribution.get(name, {})
    exposure_alpha = (attr.get("exposure_matched") or {}).get("alpha_annualized")
    random_percentile = report["random_distributions"].get("top3_risk_stop6", {}).get("strategy_percentiles", {}).get(name)
    yearly = summary.get("yearly_return", {})
    checks = {
        "total_positive": value_gt(summary.get("total_return"), 0),
        "year_2025_positive": value_gt(yearly.get("2025"), 0),
        "year_2026_positive": value_gt(yearly.get("2026"), 0),
        "max_drawdown_above_minus_25pct": value_gt(summary.get("max_drawdown"), -0.25),
        "exposure_matched_alpha_positive": value_gt(exposure_alpha, 0),
        "above_same_exit_random_q95": value_ge(random_percentile, 0.95),
        "sample_at_least_60_buys": value_ge(summary.get("buys"), 60),
    }
    return {
        "status": "OPEN" if all(checks.values()) else "CLOSED",
        "candidate": "ml_consensus_top3_risk_stop6",
        "checks": checks,
        "evidence": {
            "total_return": summary.get("total_return"),
            "yearly_return": yearly,
            "max_drawdown": summary.get("max_drawdown"),
            "exposure_matched_alpha_annualized": exposure_alpha,
            "same_exit_random_percentile": random_percentile,
            "buys": summary.get("buys"),
            "sell_reasons": summary.get("sell_reasons"),
        },
    }


def build_portfolio_ablation(nav: pd.DataFrame) -> list[dict]:
    if nav.empty or "config" not in nav.columns:
        return []
    comparisons = [
        ("walk_forward_ml_top3_fixed10", "walk_forward_ml_base_top3_fixed10"),
        ("walk_forward_ml_rank_top3_fixed10", "walk_forward_ml_base_rank_top3_fixed10"),
        ("walk_forward_ml_consensus_top3_fixed10", "walk_forward_ml_base_consensus_top3_fixed10"),
    ]
    returns = {}
    for config, frame in nav.groupby("config", sort=False):
        values = frame.sort_values("date").set_index("date")["total_value"]
        returns[config] = values.pct_change(fill_method=None).rename(config)
    rows = []
    for full_name, base_name in comparisons:
        if full_name not in returns or base_name not in returns:
            continue
        pair = pd.concat([returns[full_name], returns[base_name]], axis=1).dropna()
        difference = pair[full_name] - pair[base_name]
        low, high, probability = moving_block_bootstrap(difference)
        rows.append({
            "full_config": full_name,
            "base_config": base_name,
            "days": int(len(difference)),
            "mean_daily_increment": float(difference.mean()),
            "annualized_arithmetic_increment": float(difference.mean() * 252),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "bootstrap_probability_positive": probability,
        })
    return rows


def moving_block_bootstrap(values: pd.Series, *, block_size: int = 10, samples: int = 5000) -> tuple[float, float, float]:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260715)
    starts = np.arange(len(data))
    blocks = int(np.ceil(len(data) / block_size))
    offsets = np.arange(block_size)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(starts, size=blocks, replace=True)
        positions = (selected[:, None] + offsets[None, :]) % len(data)
        means[index] = data[positions.ravel()[:len(data)]].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float((means > 0).mean())


def value_gt(value, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) > threshold


def value_ge(value, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) >= threshold


def load_base_module():
    return load_module("walk_forward_replay_base", BASE_SCRIPT)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_markdown(report: dict) -> str:
    lines = [
        "# Walk-Forward Portfolio Replay",
        "",
        f"- Entry: {report['definition']['entry']}",
        f"- Exit: {report['definition']['exit']}",
        f"- Costs: `{json.dumps(report['definition']['costs'], ensure_ascii=False)}`",
        "",
        "## Summaries",
        "",
    ]
    lines.extend(f"- `{json.dumps(row, ensure_ascii=False)}`" for row in report["summaries"])
    lines.extend(["", "## Random Distributions", ""])
    lines.append(f"- `{json.dumps(report['random_distributions'], ensure_ascii=False)}`")
    lines.extend(["", "## Production Gate", ""])
    lines.append(f"- `{json.dumps(report['production_gate'], ensure_ascii=False)}`")
    lines.extend(["", "## Risk Exit Gate", ""])
    lines.append(f"- `{json.dumps(report.get('risk_exit_gate'), ensure_ascii=False)}`")
    lines.extend(["", "## Portfolio Ablation", ""])
    lines.extend(f"- `{json.dumps(row, ensure_ascii=False)}`" for row in report["portfolio_ablation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
