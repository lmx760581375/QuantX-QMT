"""Strategy-like backtest for the gated range/wide weak-to-strong alpha.

This script only writes under ./tmp. It converts the Round 2 tail-model output
plus the Round 3 daily gate into external_score signals, then runs the formal
QuantX config backtest engine with realistic execution validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
PROVIDER_URI = "data/qlib_data_fixed"
GATE_FEATURE = "p_bad_top20_max"
GATE_DIRECTION = "le"
QUANTILES = (0.25, 0.30, 0.35, 0.40)
TOPN_VALUES = (20, 50)
HOLD_DAYS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--build-only", action="store_true", help="Only generate signals and configs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate generated configs without running trades.")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / "strategy_backtest"
    signals_dir = output_root / "signals"
    configs_dir = output_root / "configs"
    artifacts_dir = output_root / "artifacts"
    runs_dir = output_root / "runs"
    for directory in (signals_dir, configs_dir, artifacts_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    signal_manifest = build_signals(root, signals_dir)
    configs = build_configs(signal_manifest, configs_dir)
    manifest = {
        "gate_feature": GATE_FEATURE,
        "gate_direction": GATE_DIRECTION,
        "quantiles": list(QUANTILES),
        "topn_values": list(TOPN_VALUES),
        "hold_days": HOLD_DAYS,
        "signals": signal_manifest,
        "configs": [str(path) for path in configs],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.build_only:
        print(f"[strategy] build-only signals={len(signal_manifest)} configs={len(configs)}", flush=True)
        return

    summaries = run_configs(configs, artifacts_dir, dry_run=args.dry_run)
    summary_frame = pd.DataFrame(summaries)
    summary_path = runs_dir / ("strategy_dry_run_summary.csv" if args.dry_run else "strategy_backtest_summary.csv")
    summary_frame.to_csv(summary_path, index=False)
    print(f"[strategy] wrote {summary_path} rows={len(summary_frame)}", flush=True)


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
        for topn in TOPN_VALUES:
            selected = select_daily_topn(pred, gated_days, topn)
            name = f"range_wide_tail20_badloss3_p15_gate_{GATE_FEATURE}_{GATE_DIRECTION}_q{int(q * 100):02d}_top{topn}"
            path = signals_dir / f"{name}.parquet"
            selected.to_parquet(path, index=False)
            rows.append({
                "name": name,
                "path": str(path),
                "quantile": q,
                "topn": topn,
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
    frame = pred[pred["signal_time"].isin(gated_days)].copy()
    if frame.empty:
        return frame[["signal_time", "instrument", "tail_score", "p_tail", "p_bad", "prediction_year", "tradable_rank"]]
    pieces = []
    for _, day in frame.groupby("signal_time", sort=True):
        pieces.append(day.sort_values("tail_score", ascending=False).head(topn))
    selected = pd.concat(pieces, ignore_index=True)
    return selected[["signal_time", "instrument", "tail_score", "p_tail", "p_bad", "prediction_year", "tradable_rank"]]


def build_configs(signal_manifest: list[dict[str, Any]], configs_dir: Path) -> list[Path]:
    configs: list[Path] = []
    for signal in signal_manifest:
        topn = int(signal["topn"])
        for capacity_mode, max_positions in (("capital", topn), ("basket_overlap", topn * HOLD_DAYS)):
            config = strategy_config(signal, capacity_mode, max_positions)
            path = configs_dir / f"{signal['name']}_{capacity_mode}_pos{max_positions}_hold{HOLD_DAYS}.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            configs.append(path)
    return configs


def strategy_config(signal: dict[str, Any], capacity_mode: str, max_positions: int) -> dict[str, Any]:
    name = f"{signal['name']}_{capacity_mode}_pos{max_positions}_hold{HOLD_DAYS}"
    return {
        "name": name,
        "title": "弱转强市场状态门控策略化回测",
        "description": "Round 4 tmp experiment: range+wide right-tail model, expanding p_bad gate, external_score selector, fixed 5-day exit.",
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
            "reason": f"{signal['name']}_{capacity_mode}",
        },
        "rebalance": {
            "type": "equal_weight",
            "max_positions": int(max_positions),
            "cash_use_ratio": 0.98,
            "buy_only_new_positions": True,
        },
        "execution": {
            "deal_price": "open",
            "sell_rules": [
                {"name": f"time_stop_{HOLD_DAYS}d", "when": f"holding_days >= {HOLD_DAYS}", "action": "sell_all"},
            ],
            "buy": {
                "sizing": "slot_equal",
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
            "init_cash": 1_000_000,
            "validate_trading_rules": True,
            "deal_price": "open",
            "max_workers": 1,
            "error_policy": "fail_fast",
            "legacy_cost_price": True,
            "auto_adjust_buy_quantity": False,
        },
        "metadata": {
            "experiment_root": str(ROOT),
            "signal_name": signal["name"],
            "gate_feature": GATE_FEATURE,
            "gate_direction": GATE_DIRECTION,
            "gate_quantile": float(signal["quantile"]),
            "topn": int(signal["topn"]),
            "capacity_mode": capacity_mode,
            "max_positions": int(max_positions),
            "hold_days": HOLD_DAYS,
        },
    }


def run_configs(configs: list[Path], artifacts_dir: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    for idx, config_path in enumerate(configs, start=1):
        run_id = config_path.stem
        print(f"[strategy] {idx}/{total} {'dry-run' if dry_run else 'backtest'} {run_id}", flush=True)
        cmd = [
            "python",
            "-m",
            "quantx.tools.run_backtest",
            "--config",
            str(config_path),
            "--json",
        ]
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


if __name__ == "__main__":
    main()
