# Config-Driven Strategy Schema

Use this reference when generating, reviewing, or editing QuantX YAML strategies.

## Minimal Example

```yaml
name: example_strategy
version: 1

data:
  provider_uri: data/qlib_data_fixed
  universe: all_a
  start: 2021-01-04
  end: 2025-10-17

fields:
  open: $open
  high: $high
  low: $low
  close: $close
  volume: $volume

signals:
  ma20: Mean(close, 20)
  momentum20: close / Ref(close, 20) - 1
  buy_signal: close > ma20
  sell_signal: close < ma20

selector:
  where: buy_signal
  rank_by: momentum20
  ascending: false

rebalance:
  type: equal_weight
  max_positions: 10

execution:
  sell_rules:
    - sell_signal
```

## Meaning

- `data.provider_uri`: Qlib provider path, usually `data/qlib_data_fixed`.
- `data.universe`: universe selector, usually `all_a` for full A-share coverage.
- `fields`: aliases for Qlib fields such as `$close`.
- `signals`: formula DAG. Later formulas can reference earlier aliases.
- `selector.where`: boolean candidate filter.
- `selector.rank_by`: factor used to sort candidates.
- `selector.ascending`: `false` means larger factor is better.
- `rebalance.max_positions`: maximum holdings.
- `execution.sell_rules`: boolean rules that trigger sell decisions.

## Agent Rules

- Generate new exploratory configs under `configs/strategies/generated/` unless the user asks to edit a specific file.
- Preserve the original config when changing an existing strategy.
- Run dry-run after any material config edit.
- Prefer clear intermediate formulas over a single huge expression.
- Do not use negative `Ref` or any future-looking expression.
- If a required operator is missing, say it is an implementation task instead of pretending the config can run.

## Common Failures

- Formula references an undefined alias.
- `where` is numeric instead of boolean.
- `rank_by` direction conflicts with the intended signal.
- `provider_uri` points to missing data.
- The config describes future industry/group features that are not implemented yet.
