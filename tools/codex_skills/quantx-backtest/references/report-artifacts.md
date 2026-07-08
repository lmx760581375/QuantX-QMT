# Report Artifacts

Use this reference when reading QuantX backtest output.

## Artifact Files

Each run lives under `runs/<run_id>/`.

- `summary.json`: run-level summary.
- `metrics.json`: performance metrics.
- `daily_nav.json`: daily equity, return, and drawdown.
- `trades.json`: trade and rejection records.
- `positions.json`: position snapshots.
- `closed_positions.json`: completed round trips if available.
- `explain.json`: strategy/config explanation.
- `logs.txt`: run logs.

## Useful Fields

`summary.json`:

- `name`, `run_id`, `start_date`, `end_date`
- `initial_cash`, `final_value`, `total_return`
- `symbols`

`metrics.json`:

- `total_return`, `annual_return`, `max_drawdown`, `sharpe`
- `win_rate`, `profit_factor`, `trade_count`
- `avg_holding_days`, `total_cost`

`trades.json`:

- `date`, `symbol`, `action`, `price`, `quantity`
- `trade_value`, `total_cost`, `reject_reason`

`daily_nav.json`:

- `date`, `cash`, `market_value`, `total_value`
- `daily_return`, `drawdown`

## Probe Commands

```bash
python tools/codex_skills/quantx-backtest/scripts/latest_run.py --root .
python tools/codex_skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest
python tools/codex_skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest --symbol SH600137
```

## Agent Summary Checklist

When summarizing a run, include:

- run id and config name
- total return, final value, max drawdown, Sharpe
- trade count and rejected order count
- most traded symbols, enriched with names/industries if possible
- any metric gaps or artifact gaps
