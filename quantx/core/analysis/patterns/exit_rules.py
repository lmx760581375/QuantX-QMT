"""Exit-rule what-if diagnostics for trade pattern reviews."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


DEFAULT_EXIT_RULES: tuple[Dict[str, Any], ...] = (
    {"rule_id": "original_exit", "description": "原始卖出"},
    {"rule_id": "stop_loss_8pct", "description": "跌破 -8% 止损", "stop_loss": -0.08},
    {"rule_id": "stop_loss_5pct", "description": "跌破 -5% 止损", "stop_loss": -0.05},
    {"rule_id": "take_profit_12pct", "description": "达到 12% 止盈", "take_profit": 0.12},
    {"rule_id": "take_profit_18pct", "description": "达到 18% 止盈", "take_profit": 0.18},
    {
        "rule_id": "scale_half_12pct_keep_rest",
        "description": "12% 卖一半，其余按原始卖出",
        "partial_take_profit": 0.12,
        "partial_fraction": 0.5,
    },
    {
        "rule_id": "scale_half_12pct_trail_rest_8pct",
        "description": "12% 卖一半，其余峰值回撤 8% 保护",
        "partial_take_profit": 0.12,
        "partial_fraction": 0.5,
        "trail_start": 0.12,
        "trail_drawdown": 0.08,
    },
    {
        "rule_id": "scale_third_12pct_third_18pct_keep_rest",
        "description": "12% 卖三分之一，18% 再卖三分之一",
        "partial_take_profits": [
            {"threshold": 0.12, "fraction": 1 / 3},
            {"threshold": 0.18, "fraction": 1 / 3},
        ],
    },
    {
        "rule_id": "trail_8pct_after_12pct",
        "description": "12% 后峰值回撤 8% 退出",
        "trail_start": 0.12,
        "trail_drawdown": 0.08,
    },
    {
        "rule_id": "trail_10pct_after_18pct",
        "description": "18% 后峰值回撤 10% 退出",
        "trail_start": 0.18,
        "trail_drawdown": 0.10,
    },
    {"rule_id": "early_weak_5d", "description": "5日仍低于 -3% 退出", "early_day": 5, "early_return": -0.03},
    {
        "rule_id": "guard_early_weak_then_trail",
        "description": "5日弱退出 + 12% 后回撤保护",
        "early_day": 5,
        "early_return": -0.03,
        "trail_start": 0.12,
        "trail_drawdown": 0.08,
    },
)


def run_exit_rule_what_if(
    windows: pd.DataFrame,
    bars: pd.DataFrame,
    rules: Iterable[Dict[str, Any]] = DEFAULT_EXIT_RULES,
) -> Dict[str, Any]:
    """Simulate simple single-trade exit rules over observed holding bars.

    This is a trade-path diagnostic, not a portfolio backtest. It ignores cash
    reuse, ranking changes, fees, slippage, and next-order interactions.
    """
    if windows.empty or bars.empty:
        return {"ok": False, "reason": "empty windows or bars", "rules": []}
    rule_rows = []
    trades_by_sample = {str(row["sample_id"]): row for _, row in windows.iterrows()}
    for rule in rules:
        simulated = []
        for sample_id, trade in trades_by_sample.items():
            sample_bars = bars[(bars["sample_id"] == sample_id) & (bars["is_holding"])].sort_values("offset_from_entry")
            if sample_bars.empty:
                continue
            simulated.append(_simulate_trade(rule, trade, sample_bars))
        if not simulated:
            continue
        frame = pd.DataFrame(simulated)
        original_return = pd.to_numeric(frame["original_return"], errors="coerce")
        simulated_return = pd.to_numeric(frame["simulated_return"], errors="coerce")
        delta = simulated_return - original_return
        original_success = original_return >= 0.05
        simulated_success = simulated_return >= 0.05
        original_negative = original_return <= 0.0
        simulated_negative = simulated_return <= 0.0
        rule_rows.append({
            "rule_id": rule.get("rule_id"),
            "description": rule.get("description", rule.get("rule_id")),
            "sample_count": int(len(frame)),
            "avg_return": _mean(simulated_return),
            "median_return": _median(simulated_return),
            "avg_delta_return": _mean(delta),
            "median_delta_return": _median(delta),
            "improved_rate": float((delta > 0).mean()),
            "worsened_rate": float((delta < 0).mean()),
            "success_rate": float(simulated_success.mean()),
            "success_delta": float(simulated_success.mean() - original_success.mean()),
            "negative_rate": float(simulated_negative.mean()),
            "negative_delta": float(simulated_negative.mean() - original_negative.mean()),
            "avg_holding_days": _mean(pd.to_numeric(frame["simulated_holding_days"], errors="coerce")),
            "avg_holding_days_delta": _mean(
                pd.to_numeric(frame["simulated_holding_days"], errors="coerce")
                - pd.to_numeric(frame["original_holding_days"], errors="coerce")
            ),
            "exit_reason_counts": frame["exit_reason"].value_counts(dropna=False).to_dict(),
        })
    return {
        "ok": bool(rule_rows),
        "mode": "single_trade_path_what_if",
        "note": "Diagnostic only: ignores cash reuse, ranking changes, fees, slippage, and portfolio interactions.",
        "rules": sorted(rule_rows, key=lambda row: (row.get("avg_delta_return") or 0.0), reverse=True),
    }


def _simulate_trade(rule: Dict[str, Any], trade: pd.Series, holding: pd.DataFrame) -> Dict[str, Any]:
    entry_price = float(trade.get("entry_price") or 0.0)
    original_return = float(trade.get("return") or 0.0)
    original_holding_days = int(trade.get("holding_days") or max(len(holding) - 1, 0))
    if entry_price <= 0 or str(rule.get("rule_id")) == "original_exit":
        return _result(trade, original_return, original_holding_days, "original_exit")

    partial_steps = _partial_steps(rule)
    realized_return = 0.0
    remaining_fraction = 1.0
    peak_return = -np.inf
    exit_reason = "original_exit"
    simulated_return = original_return
    simulated_holding_days = original_holding_days
    for offset, (_, bar) in enumerate(holding.iterrows(), start=1):
        high_return = float(bar.get("high", entry_price)) / entry_price - 1
        low_return = float(bar.get("low", entry_price)) / entry_price - 1
        close_return = float(bar.get("close", entry_price)) / entry_price - 1
        peak_return = max(peak_return, high_return)

        for step in partial_steps:
            if step["filled"]:
                continue
            if high_return >= step["threshold"] and remaining_fraction > 0:
                exit_fraction = min(float(step["fraction"]), remaining_fraction)
                realized_return += exit_fraction * float(step["threshold"])
                remaining_fraction -= exit_fraction
                step["filled"] = True
                exit_reason = "partial_take_profit" if exit_reason == "original_exit" else f"{exit_reason}+partial_take_profit"
                if remaining_fraction <= 1e-9:
                    return _result(trade, realized_return, offset, "partial_take_profit")

        take_profit = rule.get("take_profit")
        if take_profit is not None and high_return >= float(take_profit):
            return _result(trade, realized_return + remaining_fraction * float(take_profit), offset, "take_profit")

        stop_loss = rule.get("stop_loss")
        if stop_loss is not None and low_return <= float(stop_loss):
            return _result(trade, realized_return + remaining_fraction * float(stop_loss), offset, "stop_loss")

        early_day = rule.get("early_day")
        early_return = rule.get("early_return")
        if early_day is not None and early_return is not None and offset >= int(early_day) and close_return <= float(early_return):
            return _result(trade, realized_return + remaining_fraction * close_return, offset, "early_weak")

        trail_start = rule.get("trail_start")
        trail_drawdown = rule.get("trail_drawdown")
        if trail_start is not None and trail_drawdown is not None and peak_return >= float(trail_start):
            trigger_return = peak_return - float(trail_drawdown)
            if low_return <= trigger_return:
                return _result(trade, realized_return + remaining_fraction * trigger_return, offset, "trail_drawdown")

    if remaining_fraction < 1.0:
        simulated_return = realized_return + remaining_fraction * original_return
        exit_reason = f"{exit_reason}+original_exit" if "original_exit" not in exit_reason else exit_reason
    return _result(trade, simulated_return, simulated_holding_days, exit_reason)


def _partial_steps(rule: Dict[str, Any]) -> list[Dict[str, Any]]:
    steps = []
    if rule.get("partial_take_profit") is not None:
        steps.append({
            "threshold": float(rule["partial_take_profit"]),
            "fraction": float(rule.get("partial_fraction", 0.5)),
            "filled": False,
        })
    for raw in rule.get("partial_take_profits") or []:
        threshold = raw.get("threshold")
        fraction = raw.get("fraction")
        if threshold is None or fraction is None:
            continue
        steps.append({"threshold": float(threshold), "fraction": float(fraction), "filled": False})
    return sorted(steps, key=lambda item: item["threshold"])


def _result(trade: pd.Series, simulated_return: float, holding_days: int, reason: str) -> Dict[str, Any]:
    return {
        "sample_id": trade.get("sample_id"),
        "symbol": trade.get("symbol"),
        "entry_date": trade.get("entry_date"),
        "exit_date": trade.get("exit_date"),
        "original_return": float(trade.get("return") or 0.0),
        "simulated_return": float(simulated_return),
        "original_holding_days": int(trade.get("holding_days") or holding_days),
        "simulated_holding_days": int(holding_days),
        "exit_reason": reason,
    }


def _mean(series: pd.Series) -> float | None:
    value = series.mean()
    return float(value) if pd.notna(value) else None


def _median(series: pd.Series) -> float | None:
    value = series.median()
    return float(value) if pd.notna(value) else None
