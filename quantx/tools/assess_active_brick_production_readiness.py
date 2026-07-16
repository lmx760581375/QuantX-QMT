from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
DEFAULT_REPLAY = ROOT / "active_brick_liquidity_filter_replay.json"
DEFAULT_SYMBOL_CONCENTRATION = ROOT / "liquidity_rule_symbol_concentration.json"
DEFAULT_SH_ATTRIBUTION = ROOT / "liquidity_rule_sh_index_attribution.json"
DEFAULT_MOMENTUM_ATTRIBUTION = ROOT / "liquidity_rule_market_momentum_attribution.json"
DEFAULT_TRADES = ROOT / "active_brick_liquidity_filter_replay_trades.parquet"
DEFAULT_INDUSTRY_MEMBERSHIP = Path("data/meta/snapshots/industry_membership.csv")
DEFAULT_SECURITY_MASTER = Path("data/meta/snapshots/security_master.csv")
DEFAULT_CONCEPT_MEMBERSHIP = Path("data/meta/snapshots/sector_membership.csv")
DEFAULT_WALK_FORWARD = ROOT / "walk_forward_active_brick_ml.json"
DEFAULT_FORMAL_SUMMARY = (
    ROOT
    / "formal_backtest_liquidity_rule"
    / "active_value_brick_liquidity_rule_turnover20_60_mv20_80_stop6_2025_2026"
    / "summary.json"
)
DEFAULT_FORMAL_METRICS = DEFAULT_FORMAL_SUMMARY.with_name("metrics.json")
DEFAULT_SCORE_METADATA = Path("data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_metadata.json")
DEFAULT_SCORE_ASSET = Path("data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_scores_2025_2026.parquet")
DEFAULT_CONFIG = Path("configs/strategies/generated/active_value_brick_liquidity_rule_turnover20_60_mv20_80_stop6_2025_2026.yaml")
DEFAULT_PRODUCTION_PROFILE = Path("configs/production/daily_default.yaml")
OUT_JSON = ROOT / "active_brick_production_readiness.json"
OUT_MD = ROOT / "active_brick_production_readiness.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "Assess active brick production readiness.")
    parser.add_argument("--replay", default=str(DEFAULT_REPLAY))
    parser.add_argument("--formal-summary", default=str(DEFAULT_FORMAL_SUMMARY))
    parser.add_argument("--formal-metrics", default=str(DEFAULT_FORMAL_METRICS))
    parser.add_argument("--score-metadata", default=str(DEFAULT_SCORE_METADATA))
    parser.add_argument("--score-asset", default=str(DEFAULT_SCORE_ASSET))
    parser.add_argument("--strategy-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--production-profile", default=str(DEFAULT_PRODUCTION_PROFILE))
    parser.add_argument("--symbol-concentration", default=str(DEFAULT_SYMBOL_CONCENTRATION))
    parser.add_argument("--sh-attribution", default=str(DEFAULT_SH_ATTRIBUTION))
    parser.add_argument("--momentum-attribution", default=str(DEFAULT_MOMENTUM_ATTRIBUTION))
    parser.add_argument("--trades", default=str(DEFAULT_TRADES))
    parser.add_argument("--industry-membership", default=str(DEFAULT_INDUSTRY_MEMBERSHIP))
    parser.add_argument("--security-master", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--concept-membership", default=str(DEFAULT_CONCEPT_MEMBERSHIP))
    parser.add_argument("--walk-forward", default=str(DEFAULT_WALK_FORWARD))
    parser.add_argument("--json-output", default=str(OUT_JSON))
    parser.add_argument("--markdown-output", default=str(OUT_MD))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = assess(
        replay_path=Path(args.replay),
        formal_summary_path=Path(args.formal_summary),
        formal_metrics_path=Path(args.formal_metrics),
        score_metadata_path=Path(args.score_metadata),
        score_asset_path=Path(args.score_asset),
        strategy_config_path=Path(args.strategy_config),
        production_profile_path=Path(args.production_profile),
        symbol_concentration_path=Path(args.symbol_concentration),
        sh_attribution_path=Path(args.sh_attribution),
        momentum_attribution_path=Path(args.momentum_attribution),
        trades_path=Path(args.trades),
        industry_membership_path=Path(args.industry_membership),
        security_master_path=Path(args.security_master),
        concept_membership_path=Path(args.concept_membership),
        walk_forward_path=Path(args.walk_forward),
    )
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok=True research_gate={report['research_gate']['status']} "
            f"production_gate={report['production_gate']['status']} output={json_output}"
        )
        for blocker in report["production_gate"].get("blockers", []):
            print(f"blocker={blocker}")
    return 0


def assess(
    *,
    replay_path: Path,
    formal_summary_path: Path,
    formal_metrics_path: Path,
    score_metadata_path: Path,
    score_asset_path: Path,
    strategy_config_path: Path,
    production_profile_path: Path,
    symbol_concentration_path: Path,
    sh_attribution_path: Path,
    momentum_attribution_path: Path,
    trades_path: Path,
    industry_membership_path: Path,
    security_master_path: Path,
    concept_membership_path: Path,
    walk_forward_path: Path,
) -> dict:
    evidence = {
        "replay": read_json(replay_path),
        "formal_summary": read_json(formal_summary_path),
        "formal_metrics": read_json(formal_metrics_path),
        "score_metadata": read_json(score_metadata_path),
        "symbol_concentration": read_json(symbol_concentration_path),
        "sh_attribution": read_json(sh_attribution_path),
        "momentum_attribution": read_json(momentum_attribution_path),
        "walk_forward": read_json(walk_forward_path),
    }
    paths = {
        "replay": str(replay_path),
        "formal_summary": str(formal_summary_path),
        "formal_metrics": str(formal_metrics_path),
        "score_metadata": str(score_metadata_path),
        "score_asset": str(score_asset_path),
        "strategy_config": str(strategy_config_path),
        "production_profile": str(production_profile_path),
        "symbol_concentration": str(symbol_concentration_path),
        "sh_attribution": str(sh_attribution_path),
        "momentum_attribution": str(momentum_attribution_path),
        "trades": str(trades_path),
        "industry_membership": str(industry_membership_path),
        "security_master": str(security_master_path),
        "concept_membership": str(concept_membership_path),
        "walk_forward": str(walk_forward_path),
    }
    file_checks = build_file_checks(
        replay_path,
        formal_summary_path,
        formal_metrics_path,
        score_metadata_path,
        score_asset_path,
        strategy_config_path,
        production_profile_path,
        symbol_concentration_path,
        sh_attribution_path,
        momentum_attribution_path,
        trades_path,
        industry_membership_path,
        security_master_path,
        concept_membership_path,
        walk_forward_path,
    )
    score_asset_audit = audit_score_asset(score_asset_path)
    trade_tail = audit_trade_tail(trades_path)
    sector_audit = audit_sector_concentration(
        trades_path=trades_path,
        industry_membership_path=industry_membership_path,
        security_master_path=security_master_path,
        concept_membership_path=concept_membership_path,
    )
    walk_forward_audit = audit_walk_forward(evidence["walk_forward"])
    live_pipeline_audit = audit_live_score_pipeline(
        production_profile_path=production_profile_path,
        score_metadata=evidence["score_metadata"],
        score_asset_path=score_asset_path,
        strategy_config_path=strategy_config_path,
    )
    research_checks = build_research_checks(evidence, score_asset_audit, trade_tail, sector_audit, walk_forward_audit)
    production_checks = build_production_checks(evidence, score_asset_audit, trade_tail, sector_audit, walk_forward_audit, live_pipeline_audit)
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strategy": "active_value_brick_liquidity_rule_turnover20_60_mv20_80_stop6_2025_2026",
        "paths": paths,
        "file_checks": file_checks,
        "score_asset_audit": score_asset_audit,
        "trade_tail_audit": trade_tail,
        "sector_concentration_audit": sector_audit,
        "walk_forward_audit": walk_forward_audit,
        "live_pipeline_audit": live_pipeline_audit,
        "key_metrics": build_key_metrics(evidence, score_asset_audit, trade_tail, sector_audit, walk_forward_audit, live_pipeline_audit),
        "research_gate": summarize_gate(research_checks),
        "production_gate": summarize_gate(production_checks),
        "research_checks": research_checks,
        "production_checks": production_checks,
        "interpretation": [
            "当前证据支持继续把该组合作为研究候选推进。",
            "当前证据还不足以把它称为成熟生产策略，主要缺口是更长样本/OOS、生产日更预测链路和更严格的右尾依赖控制。",
        ],
    }


def build_file_checks(*paths: Path) -> list[dict]:
    return [
        {
            "name": path.name,
            "path": str(path),
            "status": "PASS" if path.exists() else "FAIL",
            "value": path.stat().st_size if path.exists() else None,
            "threshold": "file exists",
        }
        for path in paths
    ]


def build_research_checks(
    evidence: dict,
    score_asset_audit: dict,
    trade_tail: dict,
    sector_audit: dict,
    walk_forward_audit: dict,
) -> list[dict]:
    replay = evidence["replay"]
    random_baseline = replay.get("random_baseline") or {}
    random_strategy = find_summary(replay, random_baseline.get("strategy_config"))
    formal = evidence["formal_summary"]
    metrics = evidence["formal_metrics"]
    metadata = evidence["score_metadata"]
    concentration = evidence["symbol_concentration"]
    sh_attr = evidence["sh_attribution"]
    momentum_attr = evidence["momentum_attribution"]
    return [
        check("liquidity_gate_open", replay.get("liquidity_gate", {}).get("status") == "OPEN", replay.get("liquidity_gate", {}).get("status"), "OPEN"),
        check("formal_total_return_positive", number(formal.get("total_return")) > 0, formal.get("total_return"), "> 0"),
        check("formal_max_drawdown_controlled", number(formal.get("max_drawdown")) > -0.20, formal.get("max_drawdown"), "> -0.20"),
        check("formal_trades_enough", int_value(formal.get("buys")) >= 60 and int_value(formal.get("sells")) >= 60, {"buys": formal.get("buys"), "sells": formal.get("sells")}, ">= 60 buys and sells"),
        check("formal_no_rejects", int_value(metrics.get("reject_count")) == 0, metrics.get("reject_count"), "0"),
        check("same_pool_random_q95_beaten", number(random_strategy.get("total_return")) > number(random_baseline.get("q95")), {"strategy": random_strategy.get("total_return"), "q95": random_baseline.get("q95")}, "strategy > q95"),
        check("score_asset_rows_enough", metadata_rows(metadata) >= 500, metadata_rows(metadata), ">= 500"),
        check("score_asset_readable", score_asset_audit.get("status") == "PASS", score_asset_audit.get("status"), "PASS"),
        check("symbol_not_single_name_bet", int_value(concentration.get("max_trades_single_symbol")) <= 3, concentration.get("max_trades_single_symbol"), "<= 3"),
        check("symbol_positive_contribution_not_extreme", number(concentration.get("top5_share_of_positive_symbol_return")) <= 0.50, concentration.get("top5_share_of_positive_symbol_return"), "<= 0.50"),
        check("single_trade_tail_not_dominant", number(trade_tail.get("top1_share_of_total_sell_return_abs")) <= 0.50, trade_tail.get("top1_share_of_total_sell_return_abs"), "<= 0.50"),
        check("sh_beta_not_primary_driver", number(sh_attr.get("r2")) <= 0.10 and abs(number(sh_attr.get("beta_sh000001"))) <= 0.70, {"beta": sh_attr.get("beta_sh000001"), "r2": sh_attr.get("r2")}, "r2 <= 0.10 and abs(beta) <= 0.70"),
        check("momentum_factor_not_primary_driver", number(momentum_attr.get("r2")) <= 0.10, momentum_attr.get("r2"), "r2 <= 0.10"),
        check("sector_audit_available", sector_audit.get("status") == "PASS", sector_audit.get("status"), "PASS"),
        check("walk_forward_oos_available", walk_forward_audit.get("status") == "PASS", walk_forward_audit.get("status"), "PASS"),
    ]


def build_production_checks(
    evidence: dict,
    score_asset_audit: dict,
    trade_tail: dict,
    sector_audit: dict,
    walk_forward_audit: dict,
    live_pipeline_audit: dict,
) -> list[dict]:
    formal = evidence["formal_summary"]
    metrics = evidence["formal_metrics"]
    metadata = evidence["score_metadata"]
    start = pd.Timestamp(formal.get("start_date")) if formal.get("start_date") else None
    end = pd.Timestamp(formal.get("end_date")) if formal.get("end_date") else None
    years = float((end - start).days / 365.25) if start is not None and end is not None else 0.0
    return [
        check("formal_backtest_passes_research_bar", number(formal.get("total_return")) > 0 and number(formal.get("max_drawdown")) > -0.20, {"total_return": formal.get("total_return"), "max_drawdown": formal.get("max_drawdown")}, "return > 0 and mdd > -0.20"),
        check("score_asset_has_tradeable_depth", int_value(metadata.get("signal_days")) >= 100 and int_value(metadata.get("instruments")) >= 500, {"signal_days": metadata.get("signal_days"), "instruments": metadata.get("instruments")}, ">= 100 signal days and >= 500 instruments"),
        check("sample_long_enough", years >= 3.0, {"start": formal.get("start_date"), "end": formal.get("end_date"), "years": years}, ">= 3 years"),
        check("has_explicit_oos_or_rolling_holdout", walk_forward_audit.get("status") == "PASS", walk_forward_audit, "walk-forward folds with purge"),
        check("account_level_oos_long_enough", False, "label-level walk-forward exists; account-level OOS remains 2025-2026", "required before production"),
        check("active_brick_daily_score_bridge_ready", live_pipeline_audit.get("status") == "PASS", live_pipeline_audit, "production prediction job + strategy include + score asset metadata"),
        check("active_brick_score_panel_auto_refresh_ready", live_pipeline_audit.get("score_panel_auto_refresh") is True, live_pipeline_audit, "refresh job chains candidates -> scores -> external_score asset"),
        check("right_tail_stress_passed", number(trade_tail.get("drop_top5_total_return_proxy")) > 0, trade_tail.get("drop_top5_total_return_proxy"), "> 0 after removing top 5 sell returns"),
        check("industry_or_theme_concentration_audited", sector_audit.get("status") == "PASS", sector_audit.get("status"), "PASS"),
        check("industry_concentration_not_extreme", number(sector_audit.get("top_industry_trade_share"), 1.0) <= 0.20, sector_audit.get("top_industry_trade_share"), "<= 0.20"),
        check("board_concentration_recorded", bool(sector_audit.get("board_trade_share")), sector_audit.get("board_trade_share"), "non-empty"),
        check("data_freshness_recorded", bool(metadata.get("created_at")) and bool(formal.get("end_date")), {"metadata_created_at": metadata.get("created_at"), "backtest_end": formal.get("end_date")}, "created_at and end_date present"),
        check("score_asset_readable", score_asset_audit.get("status") == "PASS", score_asset_audit.get("status"), "PASS"),
    ]


def build_key_metrics(evidence: dict, score_asset_audit: dict, trade_tail: dict, sector_audit: dict, walk_forward_audit: dict, live_pipeline_audit: dict) -> dict:
    replay = evidence["replay"]
    formal = evidence["formal_summary"]
    metrics = evidence["formal_metrics"]
    random_baseline = replay.get("random_baseline") or {}
    concentration = evidence["symbol_concentration"]
    sh_attr = evidence["sh_attribution"]
    momentum_attr = evidence["momentum_attribution"]
    metadata = evidence["score_metadata"]
    return {
        "formal_total_return": formal.get("total_return"),
        "formal_annual_return": formal.get("annual_return"),
        "formal_max_drawdown": formal.get("max_drawdown"),
        "formal_buys": formal.get("buys"),
        "formal_sells": formal.get("sells"),
        "formal_rejects": metrics.get("reject_count"),
        "random_q95": random_baseline.get("q95"),
        "random_runs": random_baseline.get("runs"),
        "score_rows": metadata_rows(metadata),
        "score_signal_days": metadata.get("signal_days"),
        "score_instruments": metadata.get("instruments"),
        "unique_symbols_traded": concentration.get("unique_symbols"),
        "max_trades_single_symbol": concentration.get("max_trades_single_symbol"),
        "top5_symbol_positive_share": concentration.get("top5_share_of_positive_symbol_return"),
        "top1_trade_share_abs": trade_tail.get("top1_share_of_total_sell_return_abs"),
        "drop_top5_total_return_proxy": trade_tail.get("drop_top5_total_return_proxy"),
        "sh_beta": sh_attr.get("beta_sh000001"),
        "sh_r2": sh_attr.get("r2"),
        "momentum_r2": momentum_attr.get("r2"),
        "score_asset_rows_read": score_asset_audit.get("rows"),
        "industry_coverage": sector_audit.get("industry_coverage"),
        "top_industry": sector_audit.get("top_industry"),
        "top_industry_trade_share": sector_audit.get("top_industry_trade_share"),
        "board_trade_share": sector_audit.get("board_trade_share"),
        "top_concepts": sector_audit.get("top_concepts"),
        "walk_forward_folds": walk_forward_audit.get("folds"),
        "walk_forward_dates_scored": walk_forward_audit.get("dates_scored"),
        "walk_forward_purge": walk_forward_audit.get("purge"),
        "live_pipeline_status": live_pipeline_audit.get("status"),
        "live_prediction_job": live_pipeline_audit.get("prediction_job"),
        "live_strategy_included": live_pipeline_audit.get("strategy_included"),
        "live_score_asset_output": live_pipeline_audit.get("score_asset_output"),
    }


def audit_score_asset(path: Path) -> dict:
    if not path.exists():
        return {"status": "FAIL", "reason": "missing"}
    try:
        frame = pd.read_parquet(path, columns=["date", "instrument", "rule_score"])
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {"status": "FAIL", "reason": str(exc)}
    return {
        "status": "PASS",
        "rows": int(len(frame)),
        "dates": int(pd.to_datetime(frame["date"]).nunique()) if "date" in frame else None,
        "instruments": int(frame["instrument"].nunique()) if "instrument" in frame else None,
        "rule_score_non_null_ratio": float(frame["rule_score"].notna().mean()) if "rule_score" in frame and len(frame) else None,
    }


def audit_trade_tail(path: Path) -> dict:
    if not path.exists():
        return {"status": "FAIL", "reason": "missing"}
    try:
        trades = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {"status": "FAIL", "reason": str(exc)}
    target = "liquidity_turnover_020_060_mv_020_080_rule_top3_risk_stop6"
    if "config" in trades:
        trades = trades[trades["config"].eq(target)].copy()
    if "side" in trades:
        sells = trades[trades["side"].astype(str).str.lower().eq("sell")].copy()
    elif "action" in trades:
        sells = trades[trades["action"].astype(str).str.upper().eq("SELL")].copy()
    else:
        sells = trades.copy()
    ret_col = first_existing_column(sells, ["realized_return", "return", "pnl_pct", "ret", "return_pct"])
    if ret_col is None or sells.empty:
        return {"status": "FAIL", "reason": "sell return column missing", "rows": int(len(sells))}
    returns = pd.to_numeric(sells[ret_col], errors="coerce").dropna().sort_values(ascending=False)
    total_abs = float(returns.abs().sum()) if len(returns) else 0.0
    top1 = float(returns.iloc[0]) if len(returns) else None
    top5_sum = float(returns.head(5).sum()) if len(returns) else None
    return {
        "status": "PASS",
        "config": target,
        "sell_rows": int(len(returns)),
        "return_column": ret_col,
        "top1_sell_return": top1,
        "top5_sell_return_sum": top5_sum,
        "total_sell_return_sum": float(returns.sum()) if len(returns) else None,
        "top1_share_of_total_sell_return_abs": abs(top1) / total_abs if top1 is not None and total_abs else None,
        "drop_top5_total_return_proxy": float(returns.iloc[5:].sum()) if len(returns) > 5 else None,
        "worst5_sell_return_sum": float(returns.tail(5).sum()) if len(returns) else None,
    }


def audit_sector_concentration(
    *,
    trades_path: Path,
    industry_membership_path: Path,
    security_master_path: Path,
    concept_membership_path: Path,
) -> dict:
    if not trades_path.exists():
        return {"status": "FAIL", "reason": "trades missing"}
    metadata_paths = (industry_membership_path, security_master_path, concept_membership_path)
    missing = [str(path) for path in metadata_paths if not path.exists()]
    if missing:
        return {"status": "FAIL", "reason": "metadata missing", "missing": missing}
    try:
        trades = pd.read_parquet(trades_path)
        industry = pd.read_csv(industry_membership_path)
        security = pd.read_csv(security_master_path)
        concept = pd.read_csv(concept_membership_path)
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {"status": "FAIL", "reason": str(exc)}

    target = "liquidity_turnover_020_060_mv_020_080_rule_top3_risk_stop6"
    sells = trades.copy()
    if "config" in sells:
        sells = sells[sells["config"].eq(target)].copy()
    if "action" in sells:
        sells = sells[sells["action"].astype(str).str.upper().eq("SELL")].copy()
    elif "side" in sells:
        sells = sells[sells["side"].astype(str).str.lower().eq("sell")].copy()
    if sells.empty:
        return {"status": "FAIL", "reason": "no sell trades"}

    for frame in (industry, security, concept):
        frame["qx_symbol"] = frame["symbol"].map(normalize_symbol)
    industry_map = industry.drop_duplicates("qx_symbol")[["qx_symbol", "industry_name"]]
    security_map = security.drop_duplicates("qx_symbol")[["qx_symbol", "exchange", "board"]]
    enriched = sells.merge(industry_map, left_on="symbol", right_on="qx_symbol", how="left")
    enriched = enriched.merge(security_map, left_on="symbol", right_on="qx_symbol", how="left", suffixes=("", "_security"))
    ret_col = first_existing_column(enriched, ["realized_return", "return", "pnl_pct", "ret", "return_pct"])
    industry_summary = summarize_group(enriched, "industry_name", ret_col)
    board_summary = summarize_group(enriched, "board", ret_col)

    concept_pairs = sells[["symbol"]].drop_duplicates().merge(
        concept[["qx_symbol", "sector_name"]].dropna().drop_duplicates(),
        left_on="symbol",
        right_on="qx_symbol",
        how="left",
    )
    concept_counts = (
        concept_pairs.dropna(subset=["sector_name"])
        .groupby("sector_name")["symbol"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
    )
    industry_coverage = float(enriched["industry_name"].notna().mean()) if len(enriched) else 0.0
    board_coverage = float(enriched["board"].notna().mean()) if len(enriched) else 0.0
    concept_symbol_coverage = float(sells["symbol"].isin(set(concept["qx_symbol"])).mean()) if len(sells) else 0.0
    top_industry = industry_summary[0] if industry_summary else {}
    unique_symbols = int(sells["symbol"].nunique())
    return {
        "status": "PASS" if industry_coverage >= 0.80 and board_coverage >= 0.95 else "FAIL",
        "config": target,
        "sell_trades": int(len(sells)),
        "unique_symbols": unique_symbols,
        "industry_coverage": industry_coverage,
        "board_coverage": board_coverage,
        "concept_symbol_coverage": concept_symbol_coverage,
        "top_industry": top_industry.get("group"),
        "top_industry_trade_share": top_industry.get("trade_share"),
        "industry_summary": industry_summary[:15],
        "board_summary": board_summary,
        "board_trade_share": {row["group"]: row["trade_share"] for row in board_summary},
        "top_concepts": [
            {"concept": str(name), "unique_symbols": int(value), "symbol_share": float(value / unique_symbols)}
            for name, value in concept_counts.items()
        ],
    }


def normalize_symbol(value) -> str:
    text = str(value)
    if text.startswith(("SH", "SZ")):
        return text
    if text.endswith(".SH"):
        return "SH" + text[:6]
    if text.endswith(".SZ"):
        return "SZ" + text[:6]
    return text


def summarize_group(frame: pd.DataFrame, column: str, ret_col: str | None) -> list[dict]:
    if column not in frame:
        return []
    valid = frame.dropna(subset=[column]).copy()
    if valid.empty:
        return []
    rows = []
    for name, group in valid.groupby(column):
        row = {
            "group": str(name),
            "trades": int(len(group)),
            "unique_symbols": int(group["symbol"].nunique()) if "symbol" in group else None,
            "trade_share": float(len(group) / len(valid)),
        }
        if ret_col:
            row["return_sum"] = float(pd.to_numeric(group[ret_col], errors="coerce").sum())
        rows.append(row)
    return sorted(rows, key=lambda row: (row["trades"], row.get("return_sum") or 0.0), reverse=True)


def audit_walk_forward(report: dict) -> dict:
    if not report or report.get("_missing"):
        return {"status": "FAIL", "reason": "missing"}
    definition = report.get("definition") or {}
    folds = report.get("folds") or []
    purge = str(definition.get("purge") or "")
    dates_scored = int_value(report.get("dates_scored"))
    rows_scored = int_value(report.get("rows_scored"))
    months = [str(row.get("month")) for row in folds if row.get("month")]
    has_purge = "label_end" in purge and "prediction month" in purge
    fold_count = len(folds)
    return {
        "status": "PASS" if fold_count >= 12 and dates_scored >= 80 and has_purge else "FAIL",
        "folds": fold_count,
        "rows_scored": rows_scored,
        "dates_scored": dates_scored,
        "purge": purge,
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "has_purge": has_purge,
        "note": "这是标签层 walk-forward/OOS 证据，不等同于生产级账户 OOS 回测。",
    }


def audit_live_score_pipeline(
    *,
    production_profile_path: Path,
    score_metadata: dict,
    score_asset_path: Path,
    strategy_config_path: Path,
) -> dict:
    if not production_profile_path.exists():
        return {"status": "FAIL", "reason": "production profile missing"}
    try:
        profile = yaml.safe_load(production_profile_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive CLI report path
        return {"status": "FAIL", "reason": str(exc)}

    jobs = ((profile.get("predictions") or {}).get("jobs") or [])
    target_job = None
    for job in jobs:
        if str(job.get("name") or "") == "active_value_brick_liquidity_rule_scores":
            target_job = job
            break
    command = [str(part) for part in (target_job or {}).get("command") or []]
    strategy_includes = [str(item) for item in ((profile.get("strategies") or {}).get("include") or [])]

    command_text = " ".join(command)
    has_refresh_module = "quantx.tools.refresh_active_brick_liquidity_scores" in command
    output_arg = command_arg(command, "--score-asset-output") or command_arg(command, "--output")
    metadata_arg = command_arg(command, "--metadata-output")
    output_matches = True if output_arg is None and has_refresh_module else paths_equal(output_arg, str(score_asset_path))
    strategy_included = any(paths_equal(item, str(strategy_config_path)) for item in strategy_includes)
    filters = score_metadata.get("filters") or []
    filter_ok = has_filter(filters, "turnover_rate_rank_pct", 0.20, 0.60) and has_filter(filters, "free_float_mv_rank_pct", 0.20, 0.80)
    metadata_output = score_metadata.get("output")
    metadata_output_matches = paths_equal(metadata_output, str(score_asset_path))
    asset_exists = score_asset_path.exists()
    checks = {
        "prediction_job_present": target_job is not None,
        "refresh_module": has_refresh_module,
        "output_matches_score_asset": output_matches,
        "metadata_output_arg_present": bool(metadata_arg),
        "strategy_included": strategy_included,
        "metadata_output_matches_score_asset": metadata_output_matches,
        "liquidity_filters_recorded": filter_ok,
        "score_asset_exists": asset_exists,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "profile": str(production_profile_path),
        "prediction_job": "active_value_brick_liquidity_rule_scores" if target_job else None,
        "command": command,
        "command_text": command_text,
        "strategy_included": strategy_included,
        "strategy_config": str(strategy_config_path),
        "score_asset_output": str(score_asset_path),
        "score_panel_auto_refresh": has_refresh_module,
        "metadata_output": metadata_output,
        "metadata_rows": metadata_rows(score_metadata),
        "signal_days": score_metadata.get("signal_days"),
        "instruments": score_metadata.get("instruments"),
        "filters": filters,
        "checks": checks,
        "note": "这是生产日更刷新桥接：候选生成、walk-forward 打分、流动性 external_score 导出由同一个 prediction job 串联。",
    }


def command_arg(command: list[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def paths_equal(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return Path(left).as_posix().rstrip("/") == Path(right).as_posix().rstrip("/")


def has_filter(filters: list[dict], column: str, low: float, high: float) -> bool:
    for row in filters:
        if row.get("column") != column:
            continue
        if abs(number(row.get("low")) - low) < 1e-9 and abs(number(row.get("high")) - high) < 1e-9:
            return True
    return False


def metadata_rows(metadata: dict) -> int:
    return int_value(metadata.get("rows", metadata.get("output_rows")))


def first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def find_summary(replay: dict, config_name: str | None) -> dict:
    if not config_name:
        return {}
    for row in replay.get("summaries") or []:
        if row.get("config") == config_name:
            return row
    return {}


def summarize_gate(checks: list[dict]) -> dict:
    failed = [row["name"] for row in checks if row["status"] != "PASS"]
    return {
        "status": "OPEN" if not failed else "CLOSED",
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "blockers": failed,
    }


def check(name: str, passed: bool, value, threshold) -> dict:
    return {
        "name": name,
        "status": "PASS" if bool(passed) else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def render_markdown(report: dict) -> str:
    metrics = report["key_metrics"]
    lines = [
        "# Active Brick Production Readiness",
        "",
        f"- Strategy: `{report['strategy']}`",
        f"- Research gate: `{report['research_gate']['status']}` ({report['research_gate']['passed']}/{report['research_gate']['passed'] + report['research_gate']['failed']})",
        f"- Production gate: `{report['production_gate']['status']}` ({report['production_gate']['passed']}/{report['production_gate']['passed'] + report['production_gate']['failed']})",
        "",
        "## Key Metrics",
        "",
        f"- Formal return: {percent(metrics.get('formal_total_return'))}, annual={percent(metrics.get('formal_annual_return'))}, mdd={percent(metrics.get('formal_max_drawdown'))}",
        f"- Trades: buys={metrics.get('formal_buys')}, sells={metrics.get('formal_sells')}, rejects={metrics.get('formal_rejects')}",
        f"- Random baseline: runs={metrics.get('random_runs')}, q95={percent(metrics.get('random_q95'))}",
        f"- Score asset: rows={metrics.get('score_rows')}, signal_days={metrics.get('score_signal_days')}, instruments={metrics.get('score_instruments')}",
        f"- Concentration: unique_symbols={metrics.get('unique_symbols_traded')}, max_trades_single_symbol={metrics.get('max_trades_single_symbol')}, top5_positive_share={percent(metrics.get('top5_symbol_positive_share'))}",
        f"- Sector: industry_coverage={percent(metrics.get('industry_coverage'))}, top_industry={metrics.get('top_industry')}, top_industry_share={percent(metrics.get('top_industry_trade_share'))}, board_share={metrics.get('board_trade_share')}",
        f"- Live bridge: status={metrics.get('live_pipeline_status')}, prediction_job={metrics.get('live_prediction_job')}, strategy_included={metrics.get('live_strategy_included')}, score_asset={metrics.get('live_score_asset_output')}",
        f"- Attribution: sh_beta={format_number(metrics.get('sh_beta'))}, sh_r2={percent(metrics.get('sh_r2'))}, momentum_r2={percent(metrics.get('momentum_r2'))}",
        "",
        "## Production Blockers",
        "",
    ]
    for blocker in report["production_gate"].get("blockers", []):
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Research Checks", ""])
    lines.extend(render_checks(report["research_checks"]))
    lines.extend(["", "## Production Checks", ""])
    lines.extend(render_checks(report["production_checks"]))
    lines.append("")
    return "\n".join(lines)


def render_checks(checks: list[dict]) -> list[str]:
    return [f"- `{row['status']}` {row['name']}: value={row['value']}, threshold={row['threshold']}" for row in checks]


def percent(value) -> str:
    if value is None:
        return "NA"
    return f"{number(value):.2%}"


def format_number(value) -> str:
    if value is None:
        return "NA"
    return f"{number(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
