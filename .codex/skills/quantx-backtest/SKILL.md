---
name: quantx-backtest
description: Use when working with this final QuantX Reward project: build same-source Reward training data, train or infer the frozen-AE Reward model, run the weak-to-strong or Reward H15 backtests, reproduce the static sleeve portfolio, or inspect and verify final backtest results.
---

# QuantX Reward Backtest

## Defaults

- Repo root: project root containing this skill.
- Python: `conda run -n test python`
- Company pip mirror: `https://pypi.hobot.cc/simple`
- Qlib provider: `<data-root>/qlib_data_fixed`
- Formal strategies: `configs/strategies/weak_to_strong.yaml` and `configs/strategies/reward_h15.yaml`
- Formal model configs: `configs/model/`
- Formal weights: `weights/reward_v1/chunks/`

## Routing

- YAML strategy changes: read `references/config-schema.md` and `references/factor-language.md`.
- Backtest, Reward data preparation, training, inference, and sleeve commands: read `references/commands.md`.
- Result analysis: read `references/report-artifacts.md`.
- Long-running or data-mutating work: read `references/safety-rules.md`.
- Repository structure and frozen baselines: read `references/project-layout.md`.

## Standard Workflow

1. Run `python .codex/skills/quantx-backtest/scripts/quantx_skill_probe.py --root .`.
2. Before model use, run `quantx-reward weights restore` then `quantx-reward verify`.
3. Dry-run changed YAML before a full backtest.
4. For reproducibility claims, run both strategies, `quantx-reward reproduce`, then `quantx-reward verify --runs-root runs`.
5. Report config path, run ID, return, max drawdown, Sharpe, trade count, and whether reference metrics matched.
6. Preserve external input data, `workdirs/`, `runs/`, and generated score artifacts unless explicitly asked otherwise.

## Fast Commands

```bash
python .codex/skills/quantx-backtest/scripts/quantx_skill_probe.py --root .
quantx-reward weights restore
quantx-reward verify

quantx-reward backtest --strategy weak-to-strong --data-root /path/to/data --dry-run
quantx-reward backtest --strategy reward-h15 --data-root /path/to/data --dry-run

quantx-reward reproduce \
  --data-root /path/to/data \
  --score-path artifacts/scores/reward_v1.parquet
```
