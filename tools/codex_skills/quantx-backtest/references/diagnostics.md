# Diagnostics

Use this reference when a user asks why a strategy result changed, why QuantX differs from a baseline, or whether the backtest has a bug.

## Fixed Diagnosis Order

1. Data coverage: provider path, calendar range, instrument count, feature files.
2. Adjustments: price/factor/volume consistency and qlib conversion assumptions.
3. Universe: full market vs mainboard, STAR/ChiNext/BJ inclusion, delisted/missing symbols.
4. Signal counts: daily buy/sell signal counts and candidate counts.
5. Ranking: topK factor direction and tie behavior.
6. Rebalance: max positions, equal weight, sell-before-buy sequencing.
7. Execution: open/close fill, limit-up buy rejection, limit-down sell rejection, suspension, fees, slippage, lot size.
8. Cash/accounting: available cash, all-in behavior, failed orders, residual cash.
9. Reporting: total return, daily NAV, drawdown, annualization calculation.
10. Baseline quality: check whether the old system had a bug or different convention.

## Shuijiao Notes

- Current strict YAML run has recently produced about `0.6898` total return.
- Older `83%` result is not strict same-setting baseline.
- Known difference area: first-day wrap/execution convention and limit-up buy behavior.

## Useful Commands

```bash
conda run -n test python -m quantx.tools.run_backtest --config <config> --dry-run --json
python tools/codex_skills/quantx-backtest/scripts/inspect_report.py --root . --run-id latest
python tools/codex_skills/quantx-backtest/scripts/latest_run.py --root .
```

## Agent Rules

- Do not call a difference a QuantX bug until the fixed diagnosis order has been checked.
- If it is a config or data convention difference, record the exact cause and affected metrics.
- If it is a QuantX bug, implement a narrow fix and add a regression test.
