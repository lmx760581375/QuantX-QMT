# Factor Language

Use this reference when writing QuantX formula expressions.

## Current Style

QuantX YAML formulas are intended to look like Qlib-style expressions while being evaluated by QuantX's formula runtime.

Typical field aliases:

```yaml
fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume
```

Typical formulas:

```yaml
signals:
  ma20: Mean(close, 20)
  std20: Std(close, 20)
  breakout: close > Max(Ref(high, 1), 20)
  momentum20: close / Ref(close, 20) - 1
```

## Operator Checklist

Before using an operator in a config, confirm it exists in `quantx/core/factor_runtime/` or current tests.

Common intended operators:

- `Ref(x, n)`
- `Mean(x, n)`
- `RollingQuantile(x, n, q)`
- `ExpandingQuantile(x, q)`
- `Std(x, n)`
- `Max(x, n)`
- `Min(x, n)`
- `Sum(x, n)`
- arithmetic: `+`, `-`, `*`, `/`
- comparisons: `>`, `>=`, `<`, `<=`, `==`, `!=`
- boolean combinations if supported by the parser/runtime
- `Clip(x, lower, upper)`: clamp values, useful for positive/negative return legs.
- `FillNa(x, value)`: replace missing values with a scalar.
- `IsNa(x)`: boolean missing-value mask.
- `NaN()`: scalar missing value, useful with `Where(...)` for formulas that need pandas-like `replace(0, nan)` behavior.
- `MaxVolNotBearish(volume, open, close, n)`: true when the max-volume day in the trailing window is not bearish. Useful for StockTradebyZ B1-style filters.
- `PriorConsecutive(cond)`: number of consecutive true rows ending at the previous trading day.
- `BrickChart(high, low, close, n, m1, m2, m3, t, shift1, shift2, sma_w1, sma_w2, sma_w3)`: StockTradebyZ/TDX-style brick height operator for brick-pattern strategies.

Planned or strategy-dependent operators must be verified before use:

- `Rank(x)`
- `If(cond, a, b)`
- `Return(close, n)`
- `GroupMean(x, group=...)`
- `GroupRank(x, group=...)`
- Exact weekly resample operators for weekly MA filters

## Agent Rules

- Keep formulas acyclic.
- Prefer `Ref(close, 1)` for previous-day data; do not use future data.
- Split complex formulas into named intermediate signals.
- If the user asks for a missing operator, implement/runtime-test the operator first or mark the config as design-only.
