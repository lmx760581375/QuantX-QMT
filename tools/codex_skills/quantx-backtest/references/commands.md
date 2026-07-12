# QuantX Commands

Use these commands from the QuantX repo root.

## Validate A Strategy Config

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --dry-run \
  --json
```

Run dry-run before full backtest whenever a config was generated or materially changed.

## Run A Backtest

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config configs/strategies/shuijiao_legacy.yaml \
  --output-dir runs \
  --json
```

Use `--symbol-limit N` for quick exploratory runs if supported by the runner or web task API.

## Import Metadata Snapshots

```bash
conda run -n test python -m quantx.tools.update_meta \
  --source myquant-strategy \
  --security-master-csv data/meta/snapshots/security_master.csv \
  --industry-csv data/meta/snapshots/industry_membership.csv \
  --json
```

The importer reads each CSV row's `snapshot_date` first. `--date` is only needed as a fallback or intentional override.

## Incrementally Update QMT Market Data

Use the adjustment-aware QMT/xqshare updater. Stocks, ETFs, and other explicitly requested exchange instruments are stored in the same raw repository and Qlib provider.

Safe smoke run:

```bash
conda run -n test python -m quantx.tools.sync_daily_data \
  --source qmt \
  --provider-uri data/qlib_data_fixed \
  --raw-dir data/raw/qmt \
  --limit 5 \
  --dry-run \
  --json
```

Production data update:

```bash
conda run -n test python -m quantx.tools.sync_daily_data \
  --source qmt \
  --provider-uri data/qlib_data_fixed \
  --raw-dir data/raw/qmt \
  --mode incremental \
  --json
```

Sync the named universe from a standard strategy config, including ETF rotation pools:

```bash
conda run -n test python -m quantx.tools.sync_daily_data \
  --source qmt \
  --strategy-config configs/strategies/generated/etf_wufu_qmt_next_open.yaml \
  --provider-uri data/qlib_data_fixed \
  --raw-dir data/raw/qmt \
  --json
```

Agent JSON wrapper:

```bash
conda run -n test python -m quantx.tools.agent_context data-update --dry-run --limit 5
```

Agent skill wrapper:

```bash
python tools/codex_skills/quantx-backtest/scripts/update_data.py --dry-run --limit 5
python tools/codex_skills/quantx-backtest/scripts/update_data.py
```

The CLI and Agent wrapper print one progress line per stock by default. The daily production profile `configs/production/daily_default.yaml` calls this updater from its `data.update_command`, streams those progress lines into the cron log, and uses a 6-hour timeout for full-market safety.

## Install 17:30 Daily Data Cron

Print the cron block:

```bash
conda run -n test python -m quantx.tools.install_daily_data_cron
```

Install or replace the marked current-user crontab block:

```bash
conda run -n test python -m quantx.tools.install_daily_data_cron --install
```

The installed cron runs `quantx.tools.run_daily_pipeline --stage data` at `17:30` and logs to `/tmp/quantx_daily_data.log`.

Agent skill wrapper:

```bash
python tools/codex_skills/quantx-backtest/scripts/install_data_cron.py
python tools/codex_skills/quantx-backtest/scripts/install_data_cron.py --install
```

By default the cron expression is daily at 17:30. Pass `--weekdays-only` if the user explicitly wants Monday-Friday only.

## Probe Online Metadata Sources

```bash
conda run -n test python -m quantx.tools.update_meta --probe --json
```

Online metadata endpoints have been unreliable on this machine. Do not depend on them for normal strategy research.

## Add An Existing Strategy To The Web Workspace

Register only the strategy config:

```bash
conda run -n test python -m quantx.tools.agent_context add-to-visualization \
  --config configs/strategies/generated/example.yaml
```

Register the strategy and an existing backtest report:

```bash
conda run -n test python -m quantx.tools.agent_context add-to-visualization \
  --config configs/strategies/generated/example.yaml \
  --run-id batch/example_run \
  --report-id best_example \
  --title 最强示例 \
  --description 示例策略说明
```

The command validates both paths and updates `configs/production/daily_default.yaml` idempotently. `--report-id` must be a single URL-safe path segment; `--run-id` may point to a nested directory under `runs/`.

## Start Web Workspace

```bash
setsid /home/users/mingxiao.li/anaconda3/envs/test/bin/python -m quantx.server.app \
  > /tmp/quantx_server.log 2>&1 < /dev/null &
```

URL: `http://127.0.0.1:8000`

## Smoke Web APIs

```bash
curl -s http://127.0.0.1:8000/api/reports
curl -s http://127.0.0.1:8000/api/meta/symbols/SH600000
```

## Skill Probe Scripts

```bash
python tools/codex_skills/quantx-backtest/scripts/quantx_skill_probe.py --root .
python tools/codex_skills/quantx-backtest/scripts/latest_run.py --root .
python tools/codex_skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest
python tools/codex_skills/quantx-backtest/scripts/check_meta.py --root . --symbols SH600000 SH600137
```

These scripts output JSON for Agent consumption and are safe read-only probes except `restart_server.sh`.
