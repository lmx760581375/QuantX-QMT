"""Derived report fields shared by the web workspace and daily production mail."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def annual_returns(daily_nav: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute calendar-year returns from daily NAV rows."""
    if not daily_nav:
        return []
    frame = pd.DataFrame(daily_nav)
    if "date" not in frame or "total_value" not in frame:
        return []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["total_value"] = pd.to_numeric(frame["total_value"], errors="coerce")
    frame = frame.dropna(subset=["date", "total_value"]).sort_values("date")
    if frame.empty:
        return []

    rows: List[Dict[str, Any]] = []
    previous_value: float | None = None
    for year, group in frame.groupby(frame["date"].dt.year):
        group = group.sort_values("date")
        first_value = float(group["total_value"].iloc[0])
        start_value = previous_value if previous_value and previous_value > 0 else first_value
        end_value = float(group["total_value"].iloc[-1])
        if start_value <= 0:
            year_return = 0.0
        else:
            year_return = end_value / start_value - 1.0
        values = group["total_value"].astype(float)
        drawdown = values / values.cummax() - 1.0
        rows.append({
            "year": int(year),
            "start_date": group["date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
            "start_value": start_value,
            "end_value": end_value,
            "return": float(year_return),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "trading_days": int(len(group)),
        })
        previous_value = end_value
    return rows


def build_next_session_guide(
    next_candidates: Dict[str, Any] | None,
    positions: List[Dict[str, Any]] | None,
    config: Dict[str, Any] | None = None,
    latest_date: str | None = None,
) -> Dict[str, Any]:
    """Build a plain-language next-session operation guide from latest signal candidates."""
    next_candidates = dict(next_candidates or {})
    positions = [dict(row) for row in (positions or [])]
    config = config or {}
    execution = config.get("execution") or {}
    buy_cfg = execution.get("buy") or {}
    rebalance = config.get("rebalance") or {}
    selected = [dict(row) for row in next_candidates.get("selected_candidates") or []]
    held_symbols = {str(row.get("symbol")) for row in positions if row.get("symbol")}
    new_buys = [row for row in selected if str(row.get("symbol")) not in held_symbols]
    already_held = [row for row in selected if str(row.get("symbol")) in held_symbols]
    deal_price = str(execution.get("deal_price") or config.get("engine", {}).get("deal_price") or "close")
    sizing = str(buy_cfg.get("sizing") or "cash_equal")
    lot_size = int(buy_cfg.get("lot_size") or 100)
    skip_limit_up = bool(buy_cfg.get("skip_limit_up", True))
    skip_if_holding = bool(buy_cfg.get("skip_if_holding", True))

    steps: List[str] = []
    if new_buys:
        steps.append(
            f"下一交易日按策略候选尝试新买入 {len(new_buys)} 只，成交价口径为 {deal_price}，仓位方式为 {sizing}。"
        )
    else:
        steps.append("下一交易日没有新的公式买入候选，默认不新开仓。")
    if already_held:
        steps.append(f"候选中有 {len(already_held)} 只是当前持仓，按 skip_if_holding={skip_if_holding} 规则处理。")
    if positions:
        steps.append(f"当前模拟持仓 {len(positions)} 只，下一交易日继续按卖出规则检查止盈、止损、回撤和时间止损。")
    else:
        steps.append("当前模拟持仓为空，下一交易日只需关注新买入候选是否出现。")
    if skip_limit_up:
        steps.append("若候选在执行日触发涨停/price_jump 等交易校验，则跳过或记录拒单。")
    steps.append(f"下单数量按 lot_size={lot_size} 取整，最大持仓上限为 {rebalance.get('max_positions', '-')}。")

    return {
        "latest_date": latest_date,
        "signal_date": next_candidates.get("date") or next_candidates.get("signal_date"),
        "candidate_count": int(next_candidates.get("selected_count") or len(selected)),
        "raw_candidate_count": int(next_candidates.get("raw_candidate_count") or 0),
        "new_buy_candidates": new_buys,
        "already_held_candidates": already_held,
        "current_positions": positions,
        "deal_price": deal_price,
        "sizing": sizing,
        "lot_size": lot_size,
        "skip_limit_up": skip_limit_up,
        "skip_if_holding": skip_if_holding,
        "steps": steps,
    }
