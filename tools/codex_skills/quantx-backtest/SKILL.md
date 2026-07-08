---
name: quantx-backtest
description: Use when working with the QuantX project to write, validate, or modify YAML strategy configs; run dry-run or full backtests; incrementally update BaoStock/Qlib market data; inspect returns, metrics, trades, reports, and per-symbol K-line trade details; diagnose QuantX strategy/result differences such as Shuijiao alignment; update local metadata; operate the QuantX web workspace; or add agent-facing analysis metrics.
---

# QuantX Backtest

## Defaults

- Repo root: `/home/users/mingxiao.li/git/quantization/quantx`
- Python env: `conda run -n test python`
- Direct Python: `/home/users/mingxiao.li/anaconda3/envs/test/bin/python`
- Company pip mirror: `https://pypi.hobot.cc/simple`
- Default Qlib provider: `data/qlib_data_fixed`
- Default runs dir: `runs/`
- Metadata snapshots: `data/meta/snapshots/security_master.csv` and `data/meta/snapshots/industry_membership.csv`
- Web workspace URL: `http://127.0.0.1:8000`

Always work from the repo root unless the user explicitly points elsewhere.

## Task Routing

- Strategy YAML work: read `references/config-schema.md` and, for formulas, `references/factor-language.md`.
- Backtest commands: read `references/commands.md`.
- Report/trade analysis: read `references/report-artifacts.md`.
- Incremental stock data updates or daily production runs: read `references/commands.md` and `references/safety-rules.md`.
- Result mismatch or suspected bug: read `references/diagnostics.md`.
- Stock names, industries, or industry factors: read `references/industry-meta.md`.
- New metrics such as Sortino, Calmar, turnover, or industry exposure: read `references/metrics-extension.md`.
- Web workspace start/restart/check: read `references/server-workspace.md`.
- Data-changing, destructive, or long-running work: read `references/safety-rules.md`.

## Standard Workflow

1. `cd /home/users/mingxiao.li/git/quantization/quantx`.
2. Check the task type and read only the matching reference files.
3. Prefer stable JSON outputs. Use `scripts/` probes or QuantX CLI with `--json`.
4. For generated or materially edited configs, run a dry-run before full backtest.
5. When reporting results, include exact config path, run id, total return, max drawdown, trade count, and any material warnings.
6. Do not delete data or runs, reset git state, or overwrite user configs unless explicitly requested.

## Quick Commands

Probe the repo:

```bash
python tools/codex_skills/quantx-backtest/scripts/quantx_skill_probe.py --root .
```

Dry-run a config:

```bash
conda run -n test python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_legacy.yaml --dry-run --json
```

Run a full backtest:

```bash
conda run -n test python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_legacy.yaml --output-dir runs --json
```

Inspect latest run:

```bash
python tools/codex_skills/quantx-backtest/scripts/latest_run.py --root .
python tools/codex_skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest
```

Check metadata:

```bash
python tools/codex_skills/quantx-backtest/scripts/check_meta.py --root . --symbols SH600000 SH600137 SZ920010
```

Restart workspace:

```bash
bash tools/codex_skills/quantx-backtest/scripts/restart_server.sh /home/users/mingxiao.li/git/quantization/quantx
```

## Known Baselines

- Shuijiao YAML strict-rule run has recently produced about `0.6898` total return on the 2021-2025 full-market setting.
- Older `83%` Shuijiao results are not strict same-setting baseline; do not treat them as a hard target without diagnosing first-day wrap/execution differences.
- Online industry/name sources have been unreliable on this machine. Prefer committed metadata snapshots.

## Preferred Agent Interface

Prefer `quantx.tools.agent_context` over ad hoc probes when a task maps to one of its JSON commands:

```bash
conda run -n test python -m quantx.tools.agent_context status
conda run -n test python -m quantx.tools.agent_context data-status
conda run -n test python -m quantx.tools.agent_context data-update --dry-run --limit 5
conda run -n test python -m quantx.tools.agent_context meta --symbols SH600000 SH600137
conda run -n test python -m quantx.tools.agent_context validate-config --config <path>
conda run -n test python -m quantx.tools.agent_context run --config <path> --dry-run
conda run -n test python -m quantx.tools.agent_context report --run-id latest
conda run -n test python -m quantx.tools.agent_context symbol --run-id latest --symbol SH600137
conda run -n test python -m quantx.tools.agent_context metrics-compute --run-id latest --include sharpe sortino calmar
```

For data updates, prefer the bundled wrappers when an Agent needs a repeatable command:

```bash
python tools/codex_skills/quantx-backtest/scripts/update_data.py --dry-run --limit 5
python tools/codex_skills/quantx-backtest/scripts/update_data.py
python tools/codex_skills/quantx-backtest/scripts/install_data_cron.py --install
```
