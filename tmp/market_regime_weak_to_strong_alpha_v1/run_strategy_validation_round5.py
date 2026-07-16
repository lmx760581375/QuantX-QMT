"""Round 5 strategy validation for the gated weak-to-strong alpha.

This script stays under ./tmp. It reuses the Round 2 tail-model predictions and
Round 3 daily gate, then validates sizing, holding period, and light sell-risk
overlays through the formal QuantX config backtest engine.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "strategy_validation_round5"
PROVIDER_URI = "data/qlib_data_fixed"
GATE_FEATURE = "p_bad_top20_max"
GATE_DIRECTION = "le"
QUANTILES = (0.30, 0.35)
TOPN = 20
HOLD_DAYS = (3, 5, 7)
INITIAL_CASH = 1_000_000.0

SIZING_VARIANTS = {
    "capital_slot_pos20": {
        "capacity_mode": "capital",
        "max_positions": 20,
        "sizing": "slot_equal",
    },
    "capital_cash_pos20": {
        "capacity_mode": "capital",
        "max_positions": 20,
        "sizing": "cash_equal",
    },
    "basket_cash_pos100": {
        "capacity_mode": "basket_overlap",
        "max_positions": 100,
        "sizing": "cash_equal",
    },
    "basket_slot_pos100": {
        "capacity_mode": "basket_overlap",
        "max_positions": 100,
        "sizing": "slot_equal",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--build-only", action="store_true", help="Only generate signals/configs.")
    parser.add_argument("--dry-run", action="store_true", help="Compile configs through run_backtest --dry-run.")
    parser.add_argument("--analyze-only", action="store_true", help="Only collect existing artifact JSON outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose artifact summary.json already exists.")
    parser.add_argument(
        "--python-cmd",
        nargs="+",
        default=["conda", "run", "--no-capture-output", "-n", "test", "python"],
        help="Python command prefix used to invoke quantx.tools.run_backtest.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / OUTPUT_NAME
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        signal_manifest = build_signals(root, signals_dir)
        variants = build_variants()
        configs = build_configs(signal_manifest, variants, configs_dir)
        manifest = {
            "gate_feature": GATE_FEATURE,
            "gate_direction": GATE_DIRECTION,
            "quantiles": list(QUANTILES),
            "topn": TOPN,
            "hold_days": list(HOLD_DAYS),
            "variants": variants,
            "signals": signal_manifest,
            "configs": [str(path) for path in configs],
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.build_only:
            print(f"[round5] build-only signals={len(signal_manifest)} configs={len(configs)}", flush=True)
            return

        summaries = run_configs(
            configs,
            artifacts_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            python_cmd=args.python_cmd,
        )
        summary_frame = pd.DataFrame(summaries)
        summary_path = runs_dir / ("round5_dry_run_summary.csv" if args.dry_run else "round5_backtest_summary.csv")
        summary_frame.to_csv(summary_path, index=False)
        print(f"[round5] wrote {summary_path} rows={len(summary_frame)}", flush=True)
        if args.dry_run:
            return

    collect_artifacts(artifacts_dir, runs_dir)


def build_signals(root: Path, signals_dir: Path) -> list[dict[str, Any]]:
    tail = pd.read_parquet(root / "runs" / "tail_predictions.parquet")
    tail = tail[
        (tail["layer"] == "wide")
        & (tail["target_name"] == "tail_top20")
        & (tail["bad_name"] == "bad_loss3")
    ].copy()
    tail["signal_time"] = pd.to_datetime(tail["signal_time"]).dt.strftime("%Y-%m-%d")
    pred = (
        tail.groupby(["signal_time", "instrument", "prediction_year"], as_index=False)
        .agg(p_tail=("p_tail", "mean"), p_bad=("p_bad", "mean"), tradable_rank=("tradable_rank", "first"))
    )
    pred["tail_score"] = pred["p_tail"] - 1.5 * pred["p_bad"]

    daily = pd.read_csv(root / "runs" / "gate_daily_panel.csv")
    daily["signal_time"] = pd.to_datetime(daily["signal_time"]).dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("signal_time").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for q in QUANTILES:
        gated_days = expanding_gated_days(daily, GATE_FEATURE, GATE_DIRECTION, q)
        selected = select_daily_topn(pred, gated_days, TOPN)
        name = f"range_wide_tail20_badloss3_p15_gate_{GATE_FEATURE}_{GATE_DIRECTION}_q{int(q * 100):02d}_top{TOPN}"
        path = signals_dir / f"{name}.parquet"
        selected.to_parquet(path, index=False)
        rows.append({
            "name": name,
            "path": str(path),
            "quantile": q,
            "topn": TOPN,
            "signal_rows": int(len(selected)),
            "signal_days": int(selected["signal_time"].nunique()) if not selected.empty else 0,
            "first_signal": selected["signal_time"].min() if not selected.empty else None,
            "last_signal": selected["signal_time"].max() if not selected.empty else None,
        })
    return rows


def expanding_gated_days(daily: pd.DataFrame, feature: str, direction: str, quantile: float) -> set[str]:
    keep_days: set[str] = set()
    for pred_year in range(2022, 2027):
        train = daily[daily["year"] < pred_year]
        test = daily[daily["year"] == pred_year]
        if train.empty or test.empty:
            continue
        threshold = float(train[feature].quantile(quantile))
        if direction == "le":
            keep = test[feature] <= threshold
        elif direction == "ge":
            keep = test[feature] >= threshold
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        keep_days.update(test.loc[keep, "signal_time"].astype(str).tolist())
    return keep_days


def select_daily_topn(pred: pd.DataFrame, gated_days: set[str], topn: int) -> pd.DataFrame:
    cols = ["signal_time", "instrument", "tail_score", "p_tail", "p_bad", "prediction_year", "tradable_rank"]
    frame = pred[pred["signal_time"].isin(gated_days)].copy()
    if frame.empty:
        return frame[cols]
    selected = (
        frame.sort_values(["signal_time", "tail_score"], ascending=[True, False])
        .groupby("signal_time", group_keys=False)
        .head(topn)
        .reset_index(drop=True)
    )
    return selected[cols]


def build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    core_sizing = ("capital_slot_pos20", "capital_cash_pos20", "basket_cash_pos100")
    for sizing_name in core_sizing:
        for hold in HOLD_DAYS:
            variants.append(make_variant(sizing_name, hold, "time"))
        variants.append(make_variant(sizing_name, 5, "risk"))
    variants.append(make_variant("basket_slot_pos100", 5, "time"))
    return variants


def make_variant(sizing_name: str, hold_days: int, exit_variant: str) -> dict[str, Any]:
    sizing = dict(SIZING_VARIANTS[sizing_name])
    variant = {
        "sizing_variant": sizing_name,
        "capacity_mode": sizing["capacity_mode"],
        "max_positions": int(sizing["max_positions"]),
        "sizing": sizing["sizing"],
        "hold_days": int(hold_days),
        "exit_variant": exit_variant,
    }
    variant["name"] = f"{sizing_name}_hold{hold_days}_{exit_variant}"
    return variant


def build_configs(signal_manifest: list[dict[str, Any]], variants: list[dict[str, Any]], configs_dir: Path) -> list[Path]:
    configs: list[Path] = []
    for signal in signal_manifest:
        for variant in variants:
            config = strategy_config(signal, variant)
            path = configs_dir / f"{signal['name']}_{variant['name']}.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            configs.append(path)
    return configs


def strategy_config(signal: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    name = f"{signal['name']}_{variant['name']}"
    return {
        "name": name,
        "title": "弱转强市场状态门控策略化复核 Round 5",
        "description": "Round 5 tmp experiment: q30/q35 top20, sizing/hold/exit validation for gated weak-to-strong alpha.",
        "version": 1,
        "data": {
            "provider_uri": PROVIDER_URI,
            "universe": "external_score",
            "start": "2021-01-04",
            "end": "latest",
            "look_back_days": 20,
        },
        "fields": {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "amount": "$amount",
            "vwap": "$vwap",
        },
        "selector": {
            "mode": "external_score",
            "path": signal["path"],
            "date_col": "signal_time",
            "instrument_col": "instrument",
            "score_col": "tail_score",
            "sort": "score_desc",
            "topk": int(signal["topn"]),
            "lag": 1,
            "candidate_limit": int(signal["topn"]),
            "reason": f"{signal['name']}_{variant['name']}",
        },
        "rebalance": {
            "type": "equal_weight",
            "max_positions": int(variant["max_positions"]),
            "cash_use_ratio": 0.98,
            "buy_only_new_positions": True,
        },
        "execution": {
            "deal_price": "open",
            "sell_rules": sell_rules(int(variant["hold_days"]), str(variant["exit_variant"])),
            "buy": {
                "sizing": variant["sizing"],
                "lot_size": 100,
                "skip_if_holding": True,
                "skip_limit_up": True,
                "reuse_sell_cash": True,
            },
        },
        "cost": {
            "commission_rate": 0.0005,
            "min_commission": 5.0,
            "stamp_tax_rate": 0.0001,
            "stamp_tax_on_buy": True,
            "transfer_fee_rate": 0.0,
            "slippage": 0.0,
        },
        "engine": {
            "init_cash": INITIAL_CASH,
            "validate_trading_rules": True,
            "deal_price": "open",
            "max_workers": 1,
            "error_policy": "fail_fast",
            "legacy_cost_price": True,
            "auto_adjust_buy_quantity": False,
        },
        "metadata": {
            "experiment_root": str(ROOT),
            "round": 5,
            "signal_name": signal["name"],
            "gate_feature": GATE_FEATURE,
            "gate_direction": GATE_DIRECTION,
            "gate_quantile": float(signal["quantile"]),
            "topn": int(signal["topn"]),
            **variant,
        },
    }


def sell_rules(hold_days: int, exit_variant: str) -> list[dict[str, str]]:
    time_rule = {"name": f"time_stop_{hold_days}d", "when": f"holding_days >= {hold_days}", "action": "sell_all"}
    if exit_variant == "time":
        return [time_rule]
    if exit_variant == "risk":
        return [
            {"name": "stop_loss_6pct", "when": "pnl_pct <= -0.06", "action": "sell_all"},
            {
                "name": "trail_peak10_dd6_after3d",
                "when": "holding_days >= 3 and peak_pnl_pct >= 0.10 and drawdown_from_peak <= -0.06",
                "action": "sell_all",
            },
            {"name": "weak_time_exit_3d_loss", "when": "holding_days >= 3 and pnl_pct < 0", "action": "sell_all"},
            time_rule,
        ]
    raise ValueError(f"Unsupported exit variant: {exit_variant}")


def run_configs(
    configs: list[Path],
    artifacts_dir: Path,
    *,
    dry_run: bool,
    skip_existing: bool,
    python_cmd: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        summary_path = artifacts_dir / run_id / "summary.json"
        if skip_existing and not dry_run and summary_path.exists():
            print(f"[round5] {idx}/{total} skip-existing {run_id}", flush=True)
            rows.append({"run_id": run_id, "config": str(config_path), "skipped": True, "returncode": 0})
            continue

        print(f"[round5] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
        cmd = [*python_cmd, "-m", "quantx.tools.run_backtest", "--config", str(config_path), "--json"]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--output-dir", str(artifacts_dir), "--run-id", run_id])
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        row = parse_run_output(completed.stdout)
        row.update({
            "config": str(config_path),
            "run_id": run_id,
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        })
        rows.append(flatten_summary(row))
        if completed.returncode != 0:
            print(completed.stderr[-2000:], flush=True)
            raise RuntimeError(f"Backtest failed for {config_path}")
    return rows


def parse_run_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {"raw_stdout": text[-2000:]}
    return json.loads(text[start:])


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (np.integer, np.floating)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


def collect_artifacts(artifacts_dir: Path, runs_dir: Path) -> None:
    artifact_dirs = sorted(path for path in artifacts_dir.iterdir() if (path / "summary.json").exists())
    if not artifact_dirs:
        print(f"[round5] no artifacts found under {artifacts_dir}", flush=True)
        return

    metrics_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []

    for artifact_dir in artifact_dirs:
        run_id = artifact_dir.name
        summary = read_json(artifact_dir / "summary.json")
        metrics = read_json(artifact_dir / "metrics.json")
        config = read_yaml(artifact_dir / "config.yaml")
        metadata = dict(config.get("metadata") or {})
        nav = read_json_frame(artifact_dir / "daily_nav.json")
        closed = read_json_frame(artifact_dir / "closed_positions.json")
        trades = read_json_frame(artifact_dir / "trades.json")

        metrics_rows.append(summarize_run(run_id, summary, metrics, metadata, trades))
        yearly = summarize_yearly_nav(run_id, nav, metadata)
        yearly_rows.extend(yearly)
        oos_rows.append(summarize_oos_forward(run_id, yearly, summary, metrics, metadata))
        robustness_rows.extend(daily_nav_robustness(run_id, nav, metadata))
        closed_rows.extend(summarize_closed_positions(run_id, closed, metadata))
        reject_rows.extend(summarize_reject_reasons(run_id, trades, metadata))

    write_csv(runs_dir / "round5_metrics_full.csv", metrics_rows)
    write_csv(runs_dir / "round5_yearly_nav.csv", yearly_rows)
    write_csv(runs_dir / "round5_oos_forward_summary.csv", sorted(oos_rows, key=oos_sort_key))
    write_csv(runs_dir / "round5_daily_nav_robustness.csv", robustness_rows)
    write_csv(runs_dir / "round5_closed_positions_by_year.csv", closed_rows)
    write_csv(runs_dir / "round5_reject_reasons.csv", reject_rows)
    print(f"[round5] collected artifacts={len(artifact_dirs)} into {runs_dir}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def summarize_run(
    run_id: str,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    trades: pd.DataFrame,
) -> dict[str, Any]:
    row = {"run_id": run_id}
    for key in (
        "name",
        "symbols",
        "start_date",
        "end_date",
        "run_dir",
        "final_value",
        "total_return",
        "annual_return",
        "annual_volatility",
        "downside_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "order_count",
        "trade_count",
        "reject_count",
        "buy_count",
        "sell_count",
    ):
        row[key] = summary.get(key)
    for key in (
        "closed_position_count",
        "win_rate",
        "profit_factor",
        "avg_closed_return",
        "avg_holding_days",
        "avg_position_count",
        "max_position_count",
        "avg_capital_utilization",
        "recent_30_capital_utilization",
        "recent_60_capital_utilization",
        "high_utilization_day_ratio",
        "zero_utilization_day_ratio",
        "total_cost",
    ):
        row[key] = metrics.get(key)
    row.update(metadata_fields(metadata))
    row["reject_count_from_trades"] = count_rejects(trades)
    return row


def metadata_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantile": metadata.get("gate_quantile"),
        "topn": metadata.get("topn"),
        "sizing_variant": metadata.get("sizing_variant"),
        "capacity_mode": metadata.get("capacity_mode"),
        "max_positions": metadata.get("max_positions"),
        "sizing": metadata.get("sizing"),
        "hold_days": metadata.get("hold_days"),
        "exit_variant": metadata.get("exit_variant"),
    }


def count_rejects(trades: pd.DataFrame) -> int:
    if trades.empty or "reject_reason" not in trades.columns:
        return 0
    return int(trades["reject_reason"].fillna("").astype(str).ne("").sum())


def summarize_yearly_nav(run_id: str, nav: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        returns = group["daily_return"].fillna(0.0).astype(float)
        total_values = group["total_value"].astype(float)
        peak = total_values.cummax()
        drawdown = total_values / peak - 1.0
        position_count = group.get("position_count", pd.Series(dtype=float)).fillna(0.0).astype(float)
        row = {
            "run_id": run_id,
            "year": int(year),
            "days": int(len(group)),
            "year_return": compound_return(returns),
            "year_mdd": float(drawdown.min()) if not drawdown.empty else np.nan,
            "avg_position_count": float(position_count.mean()) if not position_count.empty else np.nan,
            "zero_position_ratio": float((position_count <= 0).mean()) if not position_count.empty else np.nan,
        }
        row.update(metadata_fields(metadata))
        rows.append(row)
    return rows


def summarize_oos_forward(
    run_id: str,
    yearly_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    by_year = {int(row["year"]): row for row in yearly_rows}
    oos_returns = [float(by_year[year]["year_return"]) for year in range(2022, 2026) if year in by_year]
    row = {
        "run_id": run_id,
        "total_return": summary.get("total_return"),
        "annual_return": summary.get("annual_return"),
        "max_drawdown": summary.get("max_drawdown"),
        "sharpe": summary.get("sharpe"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "avg_closed_return": metrics.get("avg_closed_return"),
        "avg_capital_utilization": metrics.get("avg_capital_utilization"),
        "zero_utilization_day_ratio": metrics.get("zero_utilization_day_ratio"),
        "oos_2022_2025_return": compound_return(pd.Series(oos_returns)) if oos_returns else np.nan,
        "oos_worst_year": min(oos_returns) if oos_returns else np.nan,
        "oos_pos_years": int(sum(value > 0 for value in oos_returns)),
        "forward_2026_return": by_year.get(2026, {}).get("year_return", np.nan),
        "forward_2026_mdd": by_year.get(2026, {}).get("year_mdd", np.nan),
    }
    row.update(metadata_fields(metadata))
    return row


def daily_nav_robustness(run_id: str, nav: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if nav.empty or "daily_return" not in nav.columns:
        return []
    frame = nav.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["date"].dt.year >= 2022]
    returns = frame["daily_return"].fillna(0.0).astype(float).reset_index(drop=True)
    scenarios = {
        "full": returns,
        "drop_top1_day": drop_top_n(returns, 1),
        "drop_top3_days": drop_top_n(returns, 3),
        "drop_top1pct_days": drop_top_pct(returns, 0.01),
        "drop_top5pct_days": drop_top_pct(returns, 0.05),
    }
    rows: list[dict[str, Any]] = []
    for scenario, series in scenarios.items():
        row = {
            "run_id": run_id,
            "scenario": scenario,
            "days": int(len(series)),
            "mean_daily_return": float(series.mean()) if len(series) else np.nan,
            "compound_return": compound_return(series),
            "positive_day_rate": float((series > 0).mean()) if len(series) else np.nan,
        }
        row.update(metadata_fields(metadata))
        rows.append(row)
    return rows


def drop_top_n(series: pd.Series, n: int) -> pd.Series:
    if len(series) <= n:
        return series.iloc[0:0]
    return series.drop(series.nlargest(n).index).reset_index(drop=True)


def drop_top_pct(series: pd.Series, pct: float) -> pd.Series:
    count = int(math.ceil(len(series) * pct))
    return drop_top_n(series, count)


def summarize_closed_positions(run_id: str, closed: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if closed.empty:
        return []
    frame = closed.copy()
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["year"] = frame["exit_date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        returns = group["return"].astype(float)
        holding = group.get("holding_days", pd.Series(dtype=float)).astype(float)
        row = {
            "run_id": run_id,
            "year": int(year),
            "closed_position_count": int(len(group)),
            "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            "avg_return": float(returns.mean()) if len(returns) else np.nan,
            "median_return": float(returns.median()) if len(returns) else np.nan,
            "avg_holding_days": float(holding.mean()) if len(holding) else np.nan,
        }
        row.update(metadata_fields(metadata))
        rows.append(row)
    return rows


def summarize_reject_reasons(run_id: str, trades: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if trades.empty or "reject_reason" not in trades.columns:
        return []
    reasons = trades["reject_reason"].fillna("").astype(str)
    reasons = reasons[reasons.ne("")]
    rows: list[dict[str, Any]] = []
    for reason, count in reasons.value_counts().sort_values(ascending=False).items():
        row = {"run_id": run_id, "reject_reason": reason, "count": int(count)}
        row.update(metadata_fields(metadata))
        rows.append(row)
    return rows


def compound_return(series: pd.Series | list[float]) -> float:
    values = pd.Series(series, dtype=float).dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def oos_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    mdd = row.get("max_drawdown")
    ann = row.get("annual_return")
    forward = row.get("forward_2026_return")
    return (
        float(mdd) if pd.notna(mdd) else -999.0,
        float(ann) if pd.notna(ann) else -999.0,
        float(forward) if pd.notna(forward) else -999.0,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
