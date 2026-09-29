# Project Layout

```text
quantx/                 # engine, formula runtime, YAML strategy runtime, reports
models/reward/          # data preparation, training, inference
configs/strategies/     # weak_to_strong.yaml, reward_h15.yaml
configs/model/          # data/training/inference configs
configs/sleeve/         # static sleeve config
weights/reward_v1/      # final weight chunks and manifest
artifacts/reference/    # frozen metrics, pictures, model manifests
workdirs/               # ignored feature store and reward dataset
runs/                   # ignored runtime backtests
quantx_reward/cli.py    # unified CLI
```

Frozen reference period: 2020-01-02 to 2026-06-02.

| Strategy | Return | Max DD | Sharpe |
| --- | ---: | ---: | ---: |
| Reward H15 | 373.94% | -43.96% | 0.872 |
| Weak-to-strong | 519.96% | -33.99% | 1.004 |
| Static 50/50 sleeve | 446.95% | -23.91% | 1.194 |
