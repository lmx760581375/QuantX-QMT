# Safety Rules

- Do not delete external data roots, `workdirs/`, `runs/`, or user configs without explicit instruction.
- Do not add full score parquet, feature memmaps, DDP shards, optimizer states, or old epoch checkpoints to Git.
- Do not overwrite `weights/reward_v1/chunks/`; restore `.pt` files with `quantx-reward weights restore`.
- Use the `test` conda environment and company pip mirror.
- State that full-market multi-year backtests and multi-GPU training are long-running before starting them.
- Do not claim a new training run matches v1.0.0 without formal reference comparison.
