# Commands

Run these from the repository root.

## Validate and Backtest

```bash
quantx-reward backtest \
  --strategy weak-to-strong \
  --data-root /path/to/data \
  --dry-run

quantx-reward backtest \
  --strategy reward-h15 \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet \
  --output-dir runs \
  --run-id reward_h15
```

Use `--symbol-limit N` only for a small smoke, not for a performance claim.

## Sleeve Reproduction

```bash
quantx-reward reproduce \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet
```

This runs independent Reward H15 and weak-to-strong sleeves, then writes cumulative-return, drawdown, monthly-return heatmap, and calendar-month positive-return charts.

## Reward Data, Training, and Inference

```bash
quantx-reward prepare-data --data-root /path/to/data

quantx-reward train --dry-run
quantx-reward infer --dry-run

quantx-reward train
quantx-reward infer
```

## Verify

```bash
quantx-reward weights restore
quantx-reward verify
quantx-reward verify --runs-root runs
```

## Bundled Scripts

```bash
python .codex/skills/quantx-backtest/scripts/quantx_skill_probe.py --root .
python .codex/skills/quantx-backtest/scripts/validate_config.py \
  --root . \
  --config configs/strategies/reward_h15.yaml \
  --data-root /path/to/data
python .codex/skills/quantx-backtest/scripts/latest_run.py --root .
python .codex/skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest
```
