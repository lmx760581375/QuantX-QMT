# Strategy YAML Schema

```yaml
name: example
version: 1

data:
  provider_uri: /path/to/data/qlib_data_fixed
  universe: all_a
  start: '2020-01-02'
  end: '2026-06-02'
  look_back_days: 60

fields:
  close: $close

signals:
  buy_signal: close > 0

selector:
  mode: precomputed
  where: buy_signal
  score: close
  sort: score_desc
  topk: 5
  lag: 1

rebalance:
  type: equal_weight
  max_positions: 5
  weight_scope: portfolio_target
  cash_use_ratio: 0.98
  buy_only_new_positions: true

execution:
  deal_price: close
  sell_rules:
    - name: time_stop_h15
      when: holding_days >= 15
      action: sell_all
```

Reward uses `selector.external_score` with `reward_return_topn_risk_score_7d`.

Rules:

- `selector.lag` must be at least 1.
- `sort` must be `score_desc`, `score_asc`, `input_order`, or `none`.
- Use `sell_all` or `sell_to_position_pct` for `sell_rules`.
- Preserve the two formal configs; place experiments under `configs/strategies/experiments/`.
- Dry-run every changed config before full market backtesting.
