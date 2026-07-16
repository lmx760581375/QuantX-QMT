from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
OFFICIAL_META = ROOT / "touzikexue_official_meta.json"
OFFICIAL_TRADES = ROOT / "touzikexue_official_trades.json"
OFFICIAL_DAILY = ROOT / "touzikexue_official_daily.json"
LOCAL_RESULT = ROOT / "result_train2024_oos.json"
LOCAL_RUN = ROOT / "runs/brick_train2024_forward_label_oos_top5_maxpos10"
OUT = ROOT / "official_alignment_report.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    official_meta = read_json(OFFICIAL_META)
    official_trades = read_json(OFFICIAL_TRADES)
    official_daily = read_json(OFFICIAL_DAILY)
    local_result = read_json(LOCAL_RESULT)
    local_trades = read_json(LOCAL_RUN / "trades.json")
    local_daily_nav = read_json(LOCAL_RUN / "daily_nav.json")
    local_positions = read_json(LOCAL_RUN / "positions.json")

    official_buy_by_signal = defaultdict(list)
    official_buy_by_buy_date = defaultdict(list)
    for trade in official_trades:
        row = {
            "symbol": str(trade.get("symbol")),
            "name": trade.get("name"),
            "rank_on_buy": trade.get("rank_on_buy"),
            "score_on_buy": trade.get("score_on_buy"),
            "buy_date": trade.get("buy_date"),
            "buy_pct": trade.get("buy_pct"),
            "return_pct": trade.get("total_return_pct"),
        }
        official_buy_by_signal[str(trade.get("signal_date"))].append(row)
        official_buy_by_buy_date[str(trade.get("buy_date"))].append(row)

    local_buys_by_date = defaultdict(list)
    for trade in local_trades:
        if trade.get("action") == "BUY" and not trade.get("reject_reason"):
            local_buys_by_date[str(trade.get("date"))].append({
                "symbol": str(trade.get("symbol")),
                "price": trade.get("price"),
                "value": trade.get("value"),
                "reason": trade.get("reason"),
            })

    wave_start_dates = [row["date"] for row in official_daily if row.get("wave_start_today")]
    active_wave_dates = [row["date"] for row in official_daily if row.get("in_active_wave")]
    official_buy_dates = sorted(official_buy_by_buy_date)
    local_buy_dates = sorted(local_buys_by_date)
    compare_dates = sorted(set(official_buy_dates) | set(local_buy_dates))
    per_date = []
    for date in compare_dates:
        official_symbols = [row["symbol"] for row in official_buy_by_buy_date.get(date, [])]
        local_symbols = [row["symbol"] for row in local_buys_by_date.get(date, [])]
        per_date.append({
            "date": date,
            "official_buy_count": len(official_symbols),
            "local_buy_count": len(local_symbols),
            "overlap_count": len(set(official_symbols) & set(local_symbols)),
            "official_symbols": official_symbols,
            "local_symbols": local_symbols,
        })

    official_nav = {row["date"]: row for row in official_daily}
    local_nav = {row["date"]: row for row in local_daily_nav}
    nav_overlap_dates = sorted(set(official_nav) & set(local_nav))
    nav_compare = []
    for date in nav_overlap_dates:
        off = official_nav[date]
        loc = local_nav[date]
        nav_compare.append({
            "date": date,
            "official_value": off.get("capital_after_close"),
            "local_value": loc.get("total_value"),
            "official_open_positions": off.get("open_positions"),
            "local_open_positions": count_local_positions(local_positions, date),
            "official_in_active_wave": off.get("in_active_wave"),
        })

    report = {
        "official_run_id": official_meta.get("run_id"),
        "official_params_core": {
            key: official_meta.get("params", {}).get(key)
            for key in [
                "top_n",
                "max_positions",
                "max_single_position_pct",
                "min_single_position_pct",
                "min_residual_buy_pct",
                "market_gate",
                "buy_execution",
                "model",
                "model_id",
                "selection_method",
                "score_on_buy_display",
            ]
        },
        "official_stats": official_meta.get("stats"),
        "local_summary": local_result.get("summary", {}).get("summary", local_result.get("summary")),
        "first_mismatch_diagnosis": [
            "官方策略使用 wave_only：只有活跃市值多头波段内新增买入；本地严格版没有按官方 wave_periods 门控。",
            "官方 max_positions 为 null，靠单仓上限 50%、最小拆仓 10%、剩余现金 5% 控制；本地严格版使用 max_positions=10 与 cash_equal。",
            "官方交易明细包含 rank_on_buy 和 score_on_buy，score 来自 live_model/v11_ridge_v10_brick_model.json；本地严格版是重新训练的 HGB forward-label score，不是同一模型。",
            "官方允许 ST 名称买入，交易中 ST 名称 40 笔；本地严格版把当前 ST/退市名作为审计/过滤风险处理，两者口径不同。",
            "官方收益从 2025-01-15 第一批买入就开始，而本地严格版日频买入远多于官方，说明账户日程和候选触发没有对齐。",
        ],
        "official_counts": {
            "trade_count": len(official_trades),
            "buy_days": len(official_buy_dates),
            "signal_days": len(official_buy_by_signal),
            "wave_start_days": len(wave_start_dates),
            "active_wave_days": len(active_wave_dates),
            "st_name_trades": sum("ST" in str(row.get("name", "")).upper() for row in official_trades),
            "exit_reasons": dict(Counter(row.get("last_sell_reason") for row in official_trades)),
        },
        "local_counts": {
            "buy_days": len(local_buy_dates),
            "effective_buys": sum(len(rows) for rows in local_buys_by_date.values()),
        },
        "wave_start_dates_2025_2026": [d for d in wave_start_dates if d >= "2025-01-01"],
        "buy_date_alignment_head40": per_date[:40],
        "nav_alignment_head40": nav_compare[:40],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "official_total_return_pct": official_meta.get("stats", {}).get("total_return_pct"),
        "local_total_return": report["local_summary"].get("total_return"),
        "official_trade_count": len(official_trades),
        "local_effective_buys": report["local_counts"]["effective_buys"],
        "official_buy_days": len(official_buy_dates),
        "local_buy_days": len(local_buy_dates),
    }, ensure_ascii=False, indent=2))
    return 0


def count_local_positions(rows: list[dict[str, Any]], date: str) -> int:
    return len({str(row.get("symbol")) for row in rows if str(row.get("date")) == date})


if __name__ == "__main__":
    raise SystemExit(main())
