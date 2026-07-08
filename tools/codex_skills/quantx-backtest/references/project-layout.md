# QuantX Project Layout

Use this reference when you need to locate QuantX configs, data, tools, reports, or web workspace code.

## Roots

- Repo root: `/home/users/mingxiao.li/git/quantization/quantx`
- Python env: `conda run -n test python`
- Direct Python: `/home/users/mingxiao.li/anaconda3/envs/test/bin/python`
- Default Qlib provider: `data/qlib_data_fixed`
- Default report root: `runs/`

## Important Paths

- `configs/strategies/`: YAML strategy configs.
- `data/qlib_data_fixed/`: preferred Qlib binary data provider.
- `data/raw/baostock/stocks/`: BaoStock raw CSV repository used by incremental updates.
- `data/meta/snapshots/security_master.csv`: committed stock names snapshot.
- `data/meta/snapshots/industry_membership.csv`: committed industry membership snapshot.
- `data/meta/quantx_meta.sqlite`: generated metadata cache; do not treat as canonical source.
- `runs/<run_id>/`: backtest artifacts.
- `quantx/tools/run_backtest.py`: config-driven backtest CLI.
- `quantx/tools/sync_daily_data.py`: adjustment-aware incremental BaoStock to Qlib data update CLI.
- `quantx/tools/install_daily_data_cron.py`: idempotent 17:30 cron installer for data updates.
- `quantx/tools/run_daily_pipeline.py`: daily production pipeline CLI.
- `quantx/tools/update_meta.py`: metadata import/probe CLI.
- `quantx/server/app.py`: local FastAPI workspace.
- `quantx/server/static/workspace.html`: web UI.
- `quantx/core/strategy/config_strategy.py`: YAML strategy compiler/runtime adapter.
- `quantx/core/factor_runtime/`: formula runtime.
- `quantx/core/analysis/reporting.py`: report artifact writer/reader.

## Run Artifacts

- `summary.json`: run id, config name, period, final value, return, symbols.
- `metrics.json`: total return, annual return, max drawdown, Sharpe, win rate, trade count, costs.
- `daily_nav.json`: daily portfolio value and drawdown series.
- `trades.json`: executed and rejected trades.
- `positions.json`: daily/open position snapshots.
- `closed_positions.json`: closed position pairs when available.
- `explain.json`: config and strategy explanation.
- `logs.txt`: textual run logs.

## Current Known Baseline

`configs/strategies/shuijiao_legacy.yaml` is the current config-driven Shuijiao reference. Its strict-rule full-market run has recently produced about `0.6898` total return over the 2021-2025 setting. Older `83%` results are not strict same-setting baseline.
