# Fixed-Step Loop Preference Experiment

## Objective

Replace the one-pass frozen-AE Reward Transformer with a fixed-step loop
preference model. The model keeps the validated frozen-AE input contract and
same-date Bradley-Terry pairs. It does not introduce return regression,
synthetic utility labels, diffusion noise, or flow-matching targets.

For AE condition tokens `Z` and `K=3` fixed steps:

```text
s_0 = 0
C_i^0 = project(Z), i = 1..L

for k = 0..K-1:
    delta_s_k, C^(k+1) = LoopBlock(s_k, C^k, t_k, delta_t_k)
    s_(k+1) = s_k + delta_t_k * delta_s_k
```

Each Transformer block owns one condition-token state. In `loop` mode, a
block reads its own previous-step state and writes its own next-step state. In
`static` mode it always reads `C_i^0`; this is the required recurrence control.

The public score is `sigmoid(s_K)`. It remains a ranking score, not a
calibrated return forecast.

## Training Loss

Only preference supervision is valid for this dataset:

```text
L = sum_k w_k * BradleyTerry(s_good^k, s_bad^k)
    + center_weight * mean(logits_k)^2
```

The default deep-supervision weights for `K=3` are `[1, 2, 3] / 6`; the final
state remains the highest-weight objective. `--no-loop-deep-supervision`
selects final-state-only BT loss. This is deep supervision, not flow matching.

## Pre-Registered Controls

All three runs must use the same root, fold, AE checkpoint, pair contract,
model width/depth, optimizer schedule, effective batch, epoch budget, and
checkpoint grid.

| ID | Architecture | State update | Step loss |
| --- | --- | --- | --- |
| A | Legacy Reward Transformer | N/A | final BT |
| B | Loop Preference, K=3 | `static` | deep BT |
| C | Loop Preference, K=3 | `loop` | deep BT |
| D | Loop Preference, K=3 | `loop` | final BT only |

The structural result is `C - B`. `C - D` measures the value of deep
supervision. `C - A` is descriptive only because it changes both rollout and
recurrence.

Checkpoint selection must use the configured historical validation split only.
The previously inspected 2020-2026 period is diagnostic after the model and
checkpoint are frozen; it cannot choose K, update mode, deep-supervision mode,
or checkpoint.

## Mainboard Training Commands

The commands below preserve the existing mainboard data, frozen AE, 5d label,
and reward capacity. `488 x 4` keeps the baseline effective per-GPU pair batch
of 1,952 while reducing loop BPTT activation memory.

```bash
conda run -n test torchrun --nproc_per_node=8 \
  tmp/weak-to-strong-diffusion-v1/train_wts_ae_reward_transformer_ddp_v1.py train \
  --root tmp/weak-to-strong-diffusion-v1/market_mainboard_close2close_balanced_v1 \
  --fold-id pre2020_eval2020_2026 \
  --rank-horizon 5 \
  --reward-backbone frozen_ae \
  --reward-architecture loop_preference \
  --loop-steps 3 \
  --loop-condition-update loop \
  --loop-deep-supervision \
  --reward-d-model 768 --reward-layers 7 --reward-heads 12 --reward-mlp-ratio 4.0 \
  --epochs 20 --pairs-per-batch 488 --grad-accum-steps 4 \
  --lr 1.5e-4 --min-lr 1.5e-5 --lr-decay-epochs 20 \
  --run-name market-mainboard-pre2020-ae-loop-preference50m-k3-deep
```

Run B with `--loop-condition-update static`. Run D with
`--no-loop-deep-supervision`. Run A with
`--reward-architecture transformer` and the same model/optimizer contract.

## Score Export And QuantX

After selecting one checkpoint from historical validation, export all complete
mainboard cross-sections. The exporter emits a validated parquet and JSON
manifest with the same `reward_score_5d` interface consumed by QuantX.

```bash
conda run -n test torchrun --nproc_per_node=8 \
  tmp/weak-to-strong-diffusion-v1/infer_wts_ae_reward_transformer_scores_v1.py \
  --root tmp/weak-to-strong-diffusion-v1/market_mainboard_close2close_balanced_v1 \
  --checkpoint <frozen-checkpoint> \
  --rank-horizon 5 --start 2020-01-02 --end 2026-07-15 \
  --output <run-dir>/score_artifacts/reward_loop_k3_epochNNN_20200102_20260715.parquet
```

Create one generated QuantX YAML from the existing mainboard OOS baseline:

```text
quantx/configs/strategies/generated/
reward_transformer_mainboard_epoch020_ppo_pre2020_native_v18/
reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026_oos_baseline.yaml
```

Only the config identity fields and `selector.external_score.path` may change.
`universe`, `topk`, `lag`, execution price, costs, exit rules, period, and
score column must remain identical. Run a dry-run before the formal replay:

```bash
conda run -n test python -m quantx.tools.run_backtest \
  --config <generated-loop-yaml> --dry-run --json
conda run -n test python -m quantx.tools.run_backtest \
  --config <generated-loop-yaml> --json
```

The formal result must retain the exported artifact manifest, QuantX run
directory, Top5 result, and same-date Bottom5 directionality control.
