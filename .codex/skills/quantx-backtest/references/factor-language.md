# Factor Language

The strategy runtime evaluates a dependency DAG over daily cross-sections.

Common fields: `open`, `high`, `low`, `close`, `volume`, `vwap`, `change`.

Common operators:

- `Mean(close, 20)`
- `EMA(close, 12)`
- `Ref(close, 1)`
- `Min(low, 5)`, `Max(close, 5)`
- `Abs(value)`
- `CSPctRank(value)`
- `BBIUptrend(bbi, 2, 120, 0.20)`

Rules:

- Use prior data only; negative offsets are invalid.
- Define intermediate factors explicitly.
- Later factors may reference earlier aliases.
- `selector.where` must evaluate to boolean.
- Formula names must be valid Python identifiers.
