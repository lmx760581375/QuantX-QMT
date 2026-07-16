"""Render daily production reports."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, List


def render_markdown(report: Dict[str, Any]) -> str:
    """Render a Chinese Markdown daily report."""
    lines: List[str] = []
    trade_date = report.get("trade_date")
    profile = report.get("profile")
    data = report.get("data_update") or {}
    strategies = report.get("strategies") or []
    totals = report.get("totals") or _totals(strategies)
    lines.append(f"# QuantX 每日策略信号 {trade_date}")
    lines.append("")
    lines.append("## 一、运行概览")
    lines.append("")
    lines.append(f"- Profile: `{profile}`")
    lines.append(f"- 数据日期: `{data.get('calendar_end') or trade_date}`")
    lines.append(f"- 数据状态: {'正常' if data.get('ok') else '异常'}")
    lines.append(f"- 策略数量: {len(strategies)}")
    lines.append(f"- 今日买入建议: {totals['buy_count']} 笔")
    lines.append(f"- 今日卖出建议: {totals['sell_count']} 笔")
    lines.append(f"- 当前模拟持仓: {totals['position_count']} 只")
    lines.append(f"- 拒单/过滤: {totals['reject_count']} 笔")
    lines.append(f"- 失败策略: {totals['failed_count']} 个")
    lines.append("")
    lines.append("## 二、重要提醒")
    lines.append("")
    lines.append("- 以下为 QuantX 策略模拟持仓和交易建议，不代表真实账户持仓。")
    lines.append("- 第一版日度生产通过完整回放生成最新状态，后续可优化为单日增量运行。")
    if not data.get("ok"):
        lines.append(f"- 数据校验异常: {data.get('message') or data}")
    lines.append("")

    lines.append("## 三、策略逐项摘要")
    for strategy in strategies:
        lines.extend(_strategy_markdown(strategy))

    lines.append("")
    lines.append("## 四、全部建议订单汇总")
    lines.append("")
    lines.extend(_orders_table_markdown("买入建议", strategies, "buys"))
    lines.append("")
    lines.extend(_orders_table_markdown("卖出建议", strategies, "sells"))
    lines.append("")
    lines.append("## 五、产物")
    lines.append("")
    lines.append(f"- 生成时间: `{report.get('generated_at')}`")
    return "\n".join(lines).rstrip() + "\n"


def render_html(report: Dict[str, Any]) -> str:
    """Render a standalone HTML daily report suitable for email clients."""
    trade_date = report.get("trade_date")
    profile = report.get("profile")
    data = report.get("data_update") or {}
    strategies = report.get("strategies") or []
    totals = report.get("totals") or _totals(strategies)
    strategy_html = "\n".join(_strategy_html(strategy) for strategy in strategies)
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 0; padding: 0; background: #f6f7f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; color: #1f2933; }
    .page { max-width: 960px; margin: 0 auto; padding: 24px 16px 40px; }
    .header { background: #ffffff; border: 1px solid #d8dee9; padding: 20px 22px; border-radius: 8px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    h2 { margin: 28px 0 12px; font-size: 19px; border-bottom: 1px solid #d8dee9; padding-bottom: 6px; }
    h3 { margin: 22px 0 10px; font-size: 16px; }
    .muted { color: #667085; }
    .grid { display: table; width: 100%; border-spacing: 10px; margin: 12px -10px; }
    .card { display: table-cell; background: #ffffff; border: 1px solid #d8dee9; border-radius: 8px; padding: 12px; vertical-align: top; min-width: 120px; }
    .metric { font-size: 20px; font-weight: 650; margin-top: 4px; }
    .strategy { background: #ffffff; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; margin: 16px 0; }
    .desc { color: #344054; }
    code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid #d8dee9; padding: 6px 8px; text-align: left; }
    th { background: #f8fafc; }
    .num { text-align: right; }
    .chart { width: 100%; max-width: 880px; overflow-x: auto; }
    .empty { color: #667085; font-style: italic; }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>QuantX 每日策略信号 """ + html.escape(str(trade_date)) + """</h1>
      <div class="muted">Profile: <code>""" + html.escape(str(profile)) + """</code> · 数据日期: <code>""" + html.escape(str(data.get("calendar_end") or trade_date)) + """</code> · 数据状态: """ + ("正常" if data.get("ok") else "异常") + """</div>
    </div>
    <div class="grid">
      """ + _metric_card("策略数量", len(strategies)) + _metric_card("今日买入", totals["buy_count"]) + _metric_card("今日卖出", totals["sell_count"]) + _metric_card("当前持仓", totals["position_count"]) + _metric_card("拒单/过滤", totals["reject_count"]) + """
    </div>
    <h2>策略详情</h2>
    """ + strategy_html + """
    <h2>产物</h2>
    <p>生成时间: <code>""" + html.escape(str(report.get("generated_at"))) + """</code></p>
  </div>
</body>
</html>
"""


def build_report_payload(
    profile_name: str,
    trade_date: str,
    run_dir: str,
    data_update: Dict[str, Any],
    strategies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    totals = _totals(strategies)
    return {
        "profile": profile_name,
        "trade_date": trade_date,
        "run_dir": run_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_update": data_update,
        "strategies": strategies,
        "totals": totals,
    }


def _totals(strategies: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "buy_count": sum(len(item.get("buys") or []) for item in strategies if item.get("ok")),
        "sell_count": sum(len(item.get("sells") or []) for item in strategies if item.get("ok")),
        "reject_count": sum(len(item.get("rejected_orders") or []) for item in strategies if item.get("ok")),
        "position_count": sum(len(item.get("positions") or []) for item in strategies if item.get("ok")),
        "failed_count": sum(1 for item in strategies if not item.get("ok")),
    }


def _strategy_markdown(strategy: Dict[str, Any]) -> List[str]:
    lines = ["", f"### {_strategy_title(strategy)}", ""]
    desc = strategy.get("description") or "未配置策略说明。"
    lines.append(f"- 策略说明: {desc}")
    if not strategy.get("ok"):
        lines.append(f"- 状态: 失败，{strategy.get('error_type')}: {strategy.get('message')}")
        return lines
    lines.append(f"- 最新交易日: `{strategy.get('latest_date')}`")
    lines.append(f"- 今日买入: {len(strategy.get('buys') or [])} 笔")
    lines.append(f"- 今日卖出: {len(strategy.get('sells') or [])} 笔")
    lines.append(f"- 当前模拟持仓: {len(strategy.get('positions') or [])} 只")
    lines.append(f"- 拒单/过滤: {len(strategy.get('rejected_orders') or [])} 笔")
    lines.extend(_annual_returns_markdown(strategy.get("annual_returns") or []))
    candidates = strategy.get("selection_candidates") or {}
    if candidates:
        lines.append(_candidate_summary_markdown(candidates))
    next_candidates = strategy.get("next_session_candidates") or {}
    if next_candidates:
        lines.append(
            f"- 下一交易日候选: 信号日 `{next_candidates.get('date')}`，"
            f"原始命中 {next_candidates.get('raw_candidate_count', 0)} 只，最终选股 {next_candidates.get('selected_count', 0)} 只"
        )
    lines.extend(_next_session_guide_markdown(strategy.get("next_session_guide") or {}))
    lines.extend(_items_markdown("建议买入", strategy.get("buys") or [], _trade_line))
    lines.extend(_items_markdown("建议卖出", strategy.get("sells") or [], _trade_line))
    if candidates:
        selected_title = "预测最终候选" if candidates.get("mode") == "frozen_predictions" else "公式最终选股"
        raw_title = "预测原始候选" if candidates.get("mode") == "frozen_predictions" else "公式原始候选"
        lines.extend(_items_markdown(selected_title, candidates.get("selected_candidates") or [], _candidate_line, limit=10))
        lines.extend(_items_markdown(raw_title, candidates.get("raw_candidates") or [], _candidate_line, limit=10))
    if next_candidates:
        lines.extend(_items_markdown("下一交易日公式最终选股", next_candidates.get("selected_candidates") or [], _candidate_line, limit=10))
    lines.extend(_items_markdown("继续持仓", strategy.get("positions") or [], _position_line, limit=12))
    lines.extend(_items_markdown("拒单/过滤", strategy.get("rejected_orders") or [], _reject_line))
    lines.extend(_items_markdown("最近五笔历史交易", strategy.get("recent_trades") or [], _trade_line, limit=5))
    lines.extend(_items_markdown("最近五笔历史拒单", strategy.get("recent_rejections") or [], _reject_line, limit=5))
    return lines


def _strategy_html(strategy: Dict[str, Any]) -> str:
    name = html.escape(_strategy_title(strategy))
    if not strategy.get("ok"):
        return (
            '<div class="strategy">'
            f"<h3>{name}</h3>"
            f"<p>状态: 失败，{html.escape(str(strategy.get('error_type')))}: {html.escape(str(strategy.get('message')))}</p>"
            "</div>"
        )
    summary = strategy.get("summary") or {}
    metrics = strategy.get("metrics") or {}
    candidates = strategy.get("selection_candidates") or {}
    next_candidates = strategy.get("next_session_candidates") or {}
    parts = [
        '<div class="strategy">',
        f"<h3>{name}</h3>",
        f"<p class=\"desc\">{html.escape(str(strategy.get('description') or '未配置策略说明。'))}</p>",
        '<div class="grid">',
        _metric_card("总收益", _pct(summary.get("total_return")), preformatted=True),
        _metric_card("最大回撤", _pct(metrics.get("max_drawdown", summary.get("max_drawdown"))), preformatted=True),
        _metric_card("Sharpe", _num(metrics.get("sharpe", summary.get("sharpe")), 3), preformatted=True),
        _metric_card("交易数", metrics.get("trade_count", summary.get("trades"))),
        "</div>",
        f"<p class=\"muted\">最新交易日: <code>{html.escape(str(strategy.get('latest_date')))}</code></p>",
        "<h3>年度收益</h3>",
        _html_table(["年份", "收益", "年内回撤", "年初权益", "年末权益", "交易日"], [
            _annual_return_cells(row) for row in (strategy.get("annual_returns") or [])
        ]),
        "<h3>下一交易日操作指南</h3>",
        _next_session_guide_html(strategy.get("next_session_guide") or {}),
        "<h3>收益曲线</h3>",
        f"<div class=\"chart\">{_equity_svg(strategy.get('equity_curve') or [])}</div>",
        f"<h3>{html.escape(_candidate_section_title(candidates))}</h3>",
        _candidate_summary_html(candidates),
        _html_table(["股票", "名称", "行业", "分数", "公式命中"], [
            _candidate_cells(row) for row in (candidates.get("selected_candidates") or [])[:10]
        ]),
        "<h3>下一交易日计划候选</h3>",
        _candidate_summary_html(next_candidates, daily_signal=True),
        _html_table(["股票", "名称", "行业", "分数", "公式命中"], [
            _candidate_cells(row) for row in (next_candidates.get("selected_candidates") or [])[:10]
        ]),
        "<h3>今日执行结果</h3>",
        _html_table(["方向", "股票", "名称", "行业", "价格", "数量", "原因"], [
            _trade_cells(row) for row in (strategy.get("buys") or []) + (strategy.get("sells") or [])
        ]),
        "<h3>今日拒单/过滤</h3>",
        _html_table(["方向", "股票", "名称", "行业", "价格", "数量", "原因"], [
            _trade_cells(row, reject=True) for row in (strategy.get("rejected_orders") or [])
        ]),
        "<h3>当前持仓</h3>",
        _html_table(["股票", "名称", "行业", "数量", "成本", "现价", "浮盈", "持仓天数"], [
            _position_cells(row) for row in (strategy.get("positions") or [])
        ]),
        "<h3>最近五笔历史交易</h3>",
        _html_table(["日期", "方向", "股票", "名称", "行业", "价格", "数量", "结果/原因"], [
            _history_trade_cells(row) for row in (strategy.get("recent_trades") or [])[-5:]
        ]),
        "<h3>最近五笔历史拒单</h3>",
        _html_table(["日期", "方向", "股票", "名称", "行业", "价格", "数量", "原因"], [
            _history_trade_cells(row, reject=True) for row in (strategy.get("recent_rejections") or [])[:5]
        ]),
        "</div>",
    ]
    return "".join(parts)


def _items_markdown(title: str, rows: List[Dict[str, Any]], formatter, limit: int = 20) -> List[str]:
    lines = ["", f"**{title}**"]
    if not rows:
        lines.append("")
        lines.append("- 无")
        return lines
    lines.append("")
    for idx, row in enumerate(rows[:limit], 1):
        lines.append(f"{idx}. {formatter(row)}")
    if len(rows) > limit:
        lines.append(f"- 还有 {len(rows) - limit} 条未展示，请查看 JSON artifact。")
    return lines


def _annual_returns_markdown(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["", "**年度收益**"]
    if not rows:
        lines.extend(["", "- 无"])
        return lines
    lines.extend([
        "",
        "| 年份 | 收益 | 年内回撤 | 年初权益 | 年末权益 | 交易日 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row.get('year')} | {_pct(row.get('return'))} | {_pct(row.get('max_drawdown'))} | "
            f"{_num(row.get('start_value'))} | {_num(row.get('end_value'))} | {int(row.get('trading_days') or 0)} |"
        )
    return lines


def _next_session_guide_markdown(guide: Dict[str, Any]) -> List[str]:
    lines = ["", "**下一交易日操作指南**", ""]
    if not guide:
        lines.append("- 无")
        return lines
    lines.append(
        f"- 信号日: `{guide.get('signal_date') or '-'}`；原始命中 {guide.get('raw_candidate_count', 0)} 只；"
        f"最终候选 {guide.get('candidate_count', 0)} 只；成交价口径 `{guide.get('deal_price') or '-'}`。"
    )
    for step in guide.get("steps") or []:
        lines.append(f"- {step}")
    lines.extend(_items_markdown("下一交易日新买候选", guide.get("new_buy_candidates") or [], _candidate_line, limit=10))
    return lines


def _orders_table_markdown(title: str, strategies: List[Dict[str, Any]], key: str) -> List[str]:
    rows = []
    for strategy in strategies:
        if not strategy.get("ok"):
            continue
        for order in strategy.get(key) or []:
            rows.append((_strategy_title(strategy), order))
    lines = [f"### {title}", ""]
    if not rows:
        lines.append("无。")
        return lines
    lines.append("| 策略 | 股票 | 名称 | 行业 | 方向 | 价格 | 数量 | 原因 |")
    lines.append("|---|---|---|---|---|---:|---:|---|")
    for strategy_name, row in rows:
        lines.append(
            f"| {strategy_name} | {row.get('symbol')} | {row.get('name') or ''} | "
            f"{row.get('industry_name') or ''} | {row.get('action')} | {_num(row.get('price'))} | "
            f"{int(row.get('quantity') or 0)} | {row.get('reason') or row.get('reject_reason') or ''} |"
        )
    return lines


def _strategy_title(strategy: Dict[str, Any]) -> str:
    for key in ("display_title", "strategy_title", "title"):
        value = str(strategy.get(key) or "").strip()
        if value:
            return value
    return str(strategy.get("strategy_name") or "unknown")


def _candidate_summary_markdown(candidates: Dict[str, Any]) -> str:
    if candidates.get("mode") == "frozen_predictions":
        suffix = ""
        if candidates.get("status") == "missing_prediction":
            suffix = f"；当前信号日无预测记录，最近可用预测日 `{candidates.get('latest_available_signal_date') or '-'}`"
        return (
            f"- 预测候选: 交易日 `{candidates.get('date')}` 查询信号日 `{candidates.get('signal_date')}`，"
            f"原始命中 {candidates.get('raw_candidate_count', 0)} 只，最终选股 {candidates.get('selected_count', 0)} 只{suffix}"
        )
    return (
        f"- 公式候选: 交易日 `{candidates.get('date')}` 使用信号日 `{candidates.get('signal_date')}`，"
        f"原始命中 {candidates.get('raw_candidate_count', 0)} 只，最终选股 {candidates.get('selected_count', 0)} 只"
    )


def _candidate_section_title(candidates: Dict[str, Any]) -> str:
    return "预测候选解释" if candidates.get("mode") == "frozen_predictions" else "公式选股解释"


def _trade_line(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('symbol')} {row.get('name') or ''} / {row.get('industry_name') or '-'}，"
        f"价格={_num(row.get('price'))}，数量={int(row.get('quantity') or 0)}，"
        f"原因={row.get('reason') or row.get('action') or '-'}"
    )


def _reject_line(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('symbol')} {row.get('name') or ''} / {row.get('industry_name') or '-'}，"
        f"方向={row.get('action')}，价格={_num(row.get('price'))}，数量={int(row.get('quantity') or 0)}，"
        f"拒绝原因={row.get('reject_reason') or '-'}"
    )


def _candidate_line(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('symbol')} {row.get('name') or ''} / {row.get('industry_name') or '-'}，"
        f"分数={_num(row.get('score'))}，公式命中={'是' if row.get('where') else '否'}"
    )


def _metric_card(label: str, value: Any, preformatted: bool = False) -> str:
    text = str(value) if preformatted else str(value)
    return (
        '<div class="card">'
        f'<div class="muted">{html.escape(str(label))}</div>'
        f'<div class="metric">{html.escape(text)}</div>'
        "</div>"
    )


def _html_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return '<p class="empty">无</p>'
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _trade_cells(row: Dict[str, Any], reject: bool = False) -> List[str]:
    return [
        str(row.get("action") or ""),
        str(row.get("symbol") or ""),
        str(row.get("name") or ""),
        str(row.get("industry_name") or ""),
        _num(row.get("price")),
        str(int(row.get("quantity") or 0)),
        str(row.get("reject_reason") if reject else (row.get("reason") or row.get("action") or "")),
    ]


def _annual_return_cells(row: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("year") or ""),
        _pct(row.get("return")),
        _pct(row.get("max_drawdown")),
        _num(row.get("start_value")),
        _num(row.get("end_value")),
        str(int(row.get("trading_days") or 0)),
    ]


def _next_session_guide_html(guide: Dict[str, Any]) -> str:
    if not guide:
        return '<p class="empty">无</p>'
    steps = "".join(f"<li>{html.escape(str(step))}</li>" for step in (guide.get("steps") or []))
    if not steps:
        steps = "<li>无操作建议</li>"
    candidates = _html_table(["股票", "名称", "行业", "分数", "公式命中"], [
        _candidate_cells(row) for row in (guide.get("new_buy_candidates") or [])[:10]
    ])
    return (
        '<p class="muted">'
        f"信号日: <code>{html.escape(str(guide.get('signal_date') or '-'))}</code> · "
        f"原始命中: <strong>{html.escape(str(guide.get('raw_candidate_count', 0)))}</strong> · "
        f"最终候选: <strong>{html.escape(str(guide.get('candidate_count', 0)))}</strong> · "
        f"成交价口径: <code>{html.escape(str(guide.get('deal_price') or '-'))}</code>"
        "</p>"
        f"<ul>{steps}</ul>"
        "<h3>下一交易日新买候选</h3>"
        f"{candidates}"
    )


def _candidate_summary_html(candidates: Dict[str, Any], daily_signal: bool = False) -> str:
    if not candidates:
        artifact = "daily_selection_candidates.json" if daily_signal else "selection_candidates.json"
        return f'<p class="empty">无候选池 artifact，请重新运行策略生成 {artifact}。</p>'
    if daily_signal:
        stale = "" if candidates.get("is_latest_signal_date", True) else " · 注意：当前展示为最近一次信号候选，不是最新交易日"
        date_text = (
            f"信号日: <code>{html.escape(str(candidates.get('date')))}</code> · "
        )
    else:
        stale = "" if candidates.get("is_latest_trade_date", True) else " · 注意：当前展示为最近一次候选，不是最新交易日"
        if candidates.get("status") == "missing_prediction":
            stale += f" · 当前信号日无预测记录，最近可用预测日 {candidates.get('latest_available_signal_date') or '-'}"
        date_text = (
            f"交易日: <code>{html.escape(str(candidates.get('date')))}</code> · "
            f"信号日: <code>{html.escape(str(candidates.get('signal_date')))}</code> · "
        )
    return (
        '<p class="muted">'
        f"{date_text}"
        f"原始命中: <strong>{html.escape(str(candidates.get('raw_candidate_count', 0)))}</strong> · "
        f"最终选股: <strong>{html.escape(str(candidates.get('selected_count', 0)))}</strong> · "
        f"排序: <code>{html.escape(str(candidates.get('sort')))}</code> · "
        f"topk: <code>{html.escape(str(candidates.get('topk')))}</code>"
        f"{html.escape(stale)}"
        "</p>"
    )


def _candidate_cells(row: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("symbol") or ""),
        str(row.get("name") or ""),
        str(row.get("industry_name") or ""),
        _num(row.get("score")),
        "是" if row.get("where") else "否",
    ]


def _history_trade_cells(row: Dict[str, Any], reject: bool = False) -> List[str]:
    reason = row.get("reject_reason") if reject or row.get("reject_reason") else (row.get("reason") or "成交")
    return [
        str(row.get("date") or ""),
        str(row.get("action") or ""),
        str(row.get("symbol") or ""),
        str(row.get("name") or ""),
        str(row.get("industry_name") or ""),
        _num(row.get("price")),
        str(int(row.get("quantity") or 0)),
        str(reason or ""),
    ]


def _position_cells(row: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("symbol") or ""),
        str(row.get("name") or ""),
        str(row.get("industry_name") or ""),
        str(int(row.get("quantity") or 0)),
        _num(row.get("avg_cost")),
        _num(row.get("current_price")),
        _pct(row.get("unrealized_return")),
        str(int(row.get("holding_days") or 0)),
    ]


def _equity_svg(curve: List[Dict[str, Any]]) -> str:
    if len(curve) < 2:
        return '<p class="empty">暂无足够净值数据</p>'
    values = []
    for row in curve:
        try:
            values.append(float(row.get("cumulative_return")))
        except Exception:
            values.append(float("nan"))
    finite = [v for v in values if v == v]
    if len(finite) < 2:
        return '<p class="empty">暂无足够净值数据</p>'
    min_v = min(finite)
    max_v = max(finite)
    if min_v == max_v:
        max_v = min_v + 1e-9
    width, height = 820, 260
    left, right, top, bottom = 58, 18, 18, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    pts = []
    denom = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        if value != value:
            continue
        x = left + plot_w * idx / denom
        y = top + plot_h * (1 - (value - min_v) / (max_v - min_v))
        pts.append(f"{x:.1f},{y:.1f}")
    start_date = curve[0].get("date") or ""
    end_date = curve[-1].get("date") or ""
    zero_y = top + plot_h * (1 - (0 - min_v) / (max_v - min_v))
    zero_line = ""
    if top <= zero_y <= top + plot_h:
        zero_line = f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#d0d5dd" stroke-dasharray="4 4" />'
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="收益曲线">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#98a2b3" />'
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#98a2b3" />'
        f'{zero_line}'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.4" points="{" ".join(pts)}" />'
        f'<text x="8" y="{top+4}" font-size="12" fill="#475467">{html.escape(_pct(max_v))}</text>'
        f'<text x="8" y="{top+plot_h}" font-size="12" fill="#475467">{html.escape(_pct(min_v))}</text>'
        f'<text x="{left}" y="{height-14}" font-size="12" fill="#475467">{html.escape(str(start_date))}</text>'
        f'<text x="{width-right-86}" y="{height-14}" font-size="12" fill="#475467">{html.escape(str(end_date))}</text>'
        "</svg>"
    )


def _position_line(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('symbol')} {row.get('name') or ''} / {row.get('industry_name') or '-'}，"
        f"数量={int(row.get('quantity') or 0)}，成本={_num(row.get('avg_cost'))}，"
        f"现价={_num(row.get('current_price'))}，浮盈={_pct(row.get('unrealized_return'))}，"
        f"持仓天数={int(row.get('holding_days') or 0)}"
    )


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _inline_code(text: str) -> str:
    parts = []
    in_code = False
    buf = []
    for ch in text:
        if ch == "`":
            escaped = html.escape("".join(buf))
            parts.append(f"<code>{escaped}</code>" if in_code else escaped)
            buf = []
            in_code = not in_code
        else:
            buf.append(ch)
    escaped = html.escape("".join(buf))
    parts.append(f"<code>{escaped}</code>" if in_code else escaped)
    return "".join(parts)


def _table_placeholder(line: str) -> str:
    cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
    tag = "th" if set(cell.replace("-", "").strip() for cell in cells) == {""} else "td"
    if tag == "th":
        return ""
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def _collapse_markdown_tables(lines: List[str]) -> List[str]:
    out: List[str] = []
    table: List[str] = []
    for line in lines:
        if line.startswith("<tr>"):
            table.append(line)
            continue
        if table:
            header, *body = table
            header = header.replace("<td>", "<th>").replace("</td>", "</th>")
            out.append("<table><thead>" + header + "</thead><tbody>" + "".join(body) + "</tbody></table>")
            table = []
        out.append(line)
    if table:
        header, *body = table
        header = header.replace("<td>", "<th>").replace("</td>", "</th>")
        out.append("<table><thead>" + header + "</thead><tbody>" + "".join(body) + "</tbody></table>")
    return [line for line in out if line]
