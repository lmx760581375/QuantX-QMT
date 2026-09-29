# Report Artifacts

Each backtest writes `runs/<run_id>/`:

- `summary.json`: config, period, final value, basic results.
- `metrics.json`: total/annual return, max drawdown, Sharpe, cost, trades.
- `daily_nav.json`: daily equity and drawdown.
- `trades.json`: executed/rejected orders.
- `positions.json`: daily position snapshots.
- `closed_positions.json`: completed round trips.
- `daily_selection_candidates.json`: daily selected candidates.
- `config.yaml`: resolved strategy config.

For a report, always state:

- config and run ID;
- date range;
- total return, annual return, maximum drawdown, Sharpe;
- trade count and rejected orders;
- whether `quantx-reward verify --runs-root runs` matched frozen references.
