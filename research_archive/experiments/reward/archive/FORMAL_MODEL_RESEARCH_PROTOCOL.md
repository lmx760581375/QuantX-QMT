# QuantX Model Research Protocol

## Purpose

This protocol applies to every predictive model family in this research area,
including diffusion, DiT, token decoders, autoregressive models, and later
architectures. Its purpose is to prevent a model result from being described as
a backtest before it has passed QuantX execution, cost, position, and trading
constraint checks.

The mandatory lifecycle is:

```text
dataset/fold -> train checkpoints -> offline validation -> full market score artifact
-> score validation -> QuantX dry-run -> QuantX full backtest -> formal bundle
-> RESEARCH_LOG conclusion
```

No step may be skipped. A failure or missing artifact leaves the experiment in
the previous state; it does not create a provisional production conclusion.

## Status Vocabulary

Use exactly one status in logs, reports, and discussion:

| Status | Minimum evidence | Allowed claim |
| --- | --- | --- |
| `training` | checkpoint and epoch metrics | training progress only |
| `offline_validated` | held-out reconstruction/path/event metrics | predictive evaluation only |
| `score_exported_quantx_not_run` | validated full score parquet and manifest | score artifact is ready, not a backtest |
| `quantx_dry_run_passed` | QuantX config validation and dry-run JSON | executable strategy candidate |
| `formal_backtest_complete` | QuantX run artifacts plus formal bundle | formal backtest result |
| `invalidated` | explicit defect or leakage evidence | no performance claim |

`offline score replay`, cross-sectional top-k return aggregation, or an event
grid are not backtests. They must never be named `backtest` in `RESEARCH_LOG`.

## 1. Dataset and Fold Contract

Before training, create and record an immutable experiment identity:

```text
experiment_id: model-family + data-root + fold-id + model-size + date
data-root: absolute path
fold-id: train/validation/prediction date boundaries and row counts
target definition: close-to-close or other executable definition
entry/exit timing: signal date, T+1/T+N execution assumptions
universe: board, exclusions, survivorship policy, label validity policy
```

The training split, validation split, calibration split, and prediction split
must be date-disjoint. Score generation must use only the prediction features
available at the signal date. Any use of realized future price as a feature
invalidates the experiment.

## 2. Training Contract

Every training run must write:

```text
runs/<run-id>/
  train_manifest.json
  checkpoints/<model>_epoch_XXX.pt  # every epoch, no best-only retention
  tensorboard/events.out.tfevents.* # mandatory, TensorBoard must initialize successfully
  logs/train_metrics.jsonl          # mandatory rank-0 append-only audit log
  decoder/AE/tokenizer configs and parameter counts
```

`logs/train_metrics.jsonl` must contain a `run_start` record, periodic
`train_progress` records, an `epoch_end` record after each checkpoint, and a
`run_end` record. Progress records must include loss, learning rate, throughput,
and allocated/reserved/peak GPU memory. Epoch records must include train and
validation loss plus the applicable path metrics. Terminal stdout is not a
training log and must never be the only source of loss history.

TensorBoard and the JSONL log are mandatory for every formal training or resume.
`--no-tensorboard` is invalid for those commands, and TensorBoard import or
writer initialization failure must stop the run. Every epoch checkpoint must
include the model, optimizer, scheduler, epoch, and global-step state so a later
resume continues the original learning-rate schedule. A legacy weight-only
checkpoint may be resumed only with an explicit audit record stating that its
previous optimizer, scheduler, and step-level loss history cannot be recovered.

The run manifest must identify the frozen dependencies: AE checkpoint, tokenizer
directory, scaler, target transform, input feature mode, random seed, and DDP
world size. A downstream score exporter must load these values from the
checkpoint/manifest rather than reconstructing them by hand.

## 3. Offline Validation Contract

Validation is necessary but insufficient. Record at least:

```text
loss, ADE, FDE, MAE at 3/5/7/10/15/20/30d,
direction accuracy, event precision/recall/base rate,
cross-sectional bucket selected return, base return, and coverage.
```

All offline metric outputs must carry `status: offline_validated` or
`status: offline_score_evaluation_only`. Do not use them as a replacement for
transaction-level results.

## 4. Mandatory Full Market Score Artifact

Every model must implement or call a model-specific exporter that writes the
shared `quantx_market_score_artifact_v1` contract. The shared writer is
`market_score_artifact_v1.py`.

Required parquet columns:

```text
signal_date: YYYY-MM-DD
instrument: QuantX/Qlib-compatible stock identifier
one or more numeric score columns
```

Required sidecar manifest:

```text
kind/schema version, parquet path, row/date/instrument counts,
score coverage per score column, model checkpoint, model/fold/scaler/tokenizer,
date range, score cadence, DDP world size, actual GPU peak memory,
and status=score_exported_quantx_not_run.
```

Rules:

1. Full means every selected stock on every selected signal date. It never means
   a 50k sample unless the artifact is explicitly a smoke artifact.
2. A cadence strategy must write null scores outside its selected dates. QuantX
   `missing: drop` then preserves the intended rebalance cadence.
3. The exporter must reject duplicate `(signal_date, instrument)` keys.
4. The exporter must refuse to overwrite an existing production artifact unless
   an explicit overwrite flag is supplied.
5. Per-rank shards may be retained for audit/resume, but the merged artifact and
   manifest are mandatory.

## 5. GPU Utilization Policy

Inference and training should use the largest stable per-GPU batch, not an
arbitrary batch copied from a previous model.

For each new model/host combination:

1. Run a one-date smoke export.
2. Probe GPU memory with the real model and real sequence generation path.
3. Start from a conservative batch and estimate the next batch from observed
   peak allocated memory.
4. Keep the largest successful batch below an 86% allocated-memory target.
   This preserves operating headroom while avoiding the previous half-empty GPU
   behavior. Record the probe attempts and actual peak in the manifest.
5. If an OOM occurs, reduce the batch, record it, and continue. Do not silently
   fall back to CPU or a single GPU.

The close-token decoder exporter implements this via `--auto-batch-size`,
`--target-gpu-memory-fraction`, and `--max-batch-size`. DDP must shard whole
signal dates, not arbitrary rows, so each rank can calculate a valid daily
cross-section and the final parquet remains auditable.

## 6. QuantX Formal Backtest Gate

Score export alone is not a conclusion. For every intended score/horizon:

1. Create an isolated YAML under `quantx/configs/strategies/generated/`.
2. Point `selector.external_score.path` and `score_col` to the validated score
   artifact. Set `missing: drop` and `require_artifact_manifest: true`. QuantX
   then rejects a missing, malformed, or undeclared-score manifest during load.
3. Make execution explicit: signal lag, deal price, max positions, holding exit,
   costs, board universe, limit-up behavior, lot size, and initial capital.
4. Run QuantX dry-run. Fix all missing-score, date, or identifier errors before
   the full run.
5. Run the full backtest and retain `summary.json`, `metrics.json`,
   `daily_nav.json`, `trades.json`, `positions.json`, and the exact YAML.
6. Run `verify_formal_research_bundle_v1.py`. A result is formal only when it
   produces `formal_backtest_complete`.

For capacity tests, initial capital must be large enough to buy the declared
number of positions in 100-share lots. A 400-stock portfolio with 1M initial
cash is not a valid capacity evaluation.

## 7. Required RESEARCH_LOG Entry

Every completed experiment section must include:

```text
experiment id and purpose
dataset/fold boundaries and OOS dates
model/checkpoint/tokenizer/scaler and parameter count
score artifact path + manifest + coverage + GPU peak/batch size
offline metrics (clearly labelled offline)
QuantX YAML path + run id
total return, annual return, max drawdown, Sharpe, PF, cost, trade count
yearly returns, capacity, rejected order count, and material execution warnings
counterfactual/control results
conclusion, failure cases, and next action
```

The log must distinguish a hypothesis from a confirmed result. Any mismatch
between offline replay and QuantX must be explained before proposing a model
improvement.

## 8. Standard Commands

Decoder score export uses the 8-GPU launcher and an explicit output path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 \
${HOME}/anaconda3/envs/test/bin/torchrun --standalone --nproc_per_node=8 \
infer_wts_close_token_decoder_scores_v1.py -- \
  --root <dataset-root> --kronos-root <feature-root> --checkpoint <checkpoint> \
  --start 2020-01-02 --end 2026-06-01 --cadences 5,7,10 \
  --auto-batch-size --target-gpu-memory-fraction 0.86 --max-batch-size 16384 \
  --output <score.parquet>
```

QuantX must then be run from `${HOME}/git/quantization/quantx`:

```bash
conda run -n test python -m quantx.tools.run_backtest --config <yaml> --dry-run --json
conda run -n test python -m quantx.tools.run_backtest --config <yaml> --output-dir runs --json
```

Finally prove the complete bundle:

```bash
python verify_formal_research_bundle_v1.py \
  --score-artifact <score.parquet> --score-columns <score-column> \
  --quantx-run <quantx-run-dir> --config <yaml> --output <formal_bundle.json>
```

## 9. Incident Response

If a model was evaluated without a score artifact or QuantX run:

1. Mark prior result `offline_score_evaluation_only` in the research log.
2. Do not reconstruct stock scores from aggregate period returns; that information
   has been discarded.
3. Re-run inference from the checkpoint and frozen preprocessing artifacts.
4. Export, validate, dry-run, and run QuantX before making any return claim.
5. Record the root cause and the prevention mechanism in the next log entry.
