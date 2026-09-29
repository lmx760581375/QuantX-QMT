# PPO 动态仓位管理

本目录实现 Reward score 策略的 PPO 动态仓位控制器。它不训练选股模型，也不维护第二套回测账户。
所有订单、成交、成本、整手、T+1、涨跌停、停牌和日终净值均由 QuantX 执行。每个 PPO episode 同时运行
agent 和同 source 的原始 QuantX baseline 两个独立账户；两者共享只读行情，但不共享现金、持仓、成交或净值。

## 边界

- 目标策略固定为原始 Reward 每日 TopK 候选，PPO 不得购买 rank K 以后的股票。
- PPO 在执行日生成 `PositionPlan`：现有持仓的 `0/25/50/100%` 卖出比例、未持仓候选的买入开关、
  `0/25/50/75/100%` 现金部署比例和候选间相对权重。
- 完整卖出才释放槽位；部分卖出不释放槽位；买入是否成交与实际数量由 QuantX 决定。
- 目标 `mainboard / reward epoch020 / 5d / Top5 / H30+trail` 不在 `source_catalog.yaml` 的训练 sources 内。

## 结构

- `rl_data.py`：artifact manifest 校验、同日分位和市场/个股特征。
- `quantx_replay_env.py`：训练期逐日 QuantX replay，不包含自建账户或撮合代码。
- `ppo_model.py`、`ppo_policy.py`：掩码的多头 actor-critic、Categorical 与 Dirichlet 动作分布。
- `train_ppo_position_manager.py`：GAE + clipped PPO、JSONL、TensorBoard、checkpoint、normalizer 和 manifest。
- `evaluate_ppo_position_manager.py`：确定性 replay 审计，输出逐日动作、订单、成交和收益。
- `select_ppo_checkpoint.py`：仅按固定 validation 指标预注册地选择一个 checkpoint。
- `quantx_position_manager.py`：正式 QuantX 插件，加载冻结 artifact 并生成 `PositionPlan`。
- `render_ppo_formal_config.py`：从原始基线复制生成正式 PPO YAML，避免手工改变候选或成交合同。

## 训练

PPO 模型较小，GPU 只用于网络前后向；瓶颈是 QuantX 账户的逐日执行。训练使用单个 GPU policy
进程加多个 CPU QuantX worker：每个 worker 独立持有 agent/baseline 两套账户，主进程一次批量推理
多个 observation。只读行情、score 和候选缓存会在 CUDA 初始化前通过 Linux copy-on-write 共享；账户、
订单、成交、成本和 NAV 不共享。多卡 DDP 会复制 policy、QLib 上下文和账户轨迹，不会提高同一条采样链路吞吐。

```bash
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
CUDA_VISIBLE_DEVICES=0 ${HOME}/anaconda3/envs/test/bin/python \
  ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/rl_position_manager_v1/train_ppo_position_manager.py \
  --quantx-root ${HOME}/git/quantization/quantx \
  --provider-uri ${HOME}/git/quantization/quantx/data/qlib_data_fixed \
  --output-dir ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/rl_position_manager_v1/runs/ppo_v1 \
  --train-start 2020-01-02 --train-end 2023-12-29 \
  --validation-start 2024-01-02 --validation-end 2024-12-31 \
  --updates 100 --rollout-steps 8192 --episode-steps 126 --warmup-steps 30 \
  --minibatch-size 2048 --num-envs 16
```

每个 checkpoint 保存模型、optimizer、normalizer 和完整训练参数；默认写入 TensorBoard 与
`train_metrics.jsonl`。日志额外记录 `agent_steps_per_second`、`quantx_account_steps_per_second`、
`policy_inference_seconds` 和 `quantx_wait_seconds`，用于区分 GPU policy 与 QuantX 环境瓶颈。
`source_catalog.yaml` 是训练输入的唯一清单，新增 source 前必须校验其 parquet
manifest、`universe`、各 TopK 的 `baseline_configs`，并确认不包含目标 holdout。

每个 source-capacity 都绑定已有正式 QuantX YAML。训练 reward 不是绝对收益，而是：

```text
log(1 + agent_daily_return) - log(1 + baseline_daily_return)
- 0.001 * max(0, agent_turnover - baseline_turnover)
- 0.01  * 新增的超基线回撤
```

因此 PPO 学习的是在相同 source、日期、候选、成本与成交规则下，是否可以通过动态持仓控制提升净收益、
减少相对 baseline 的回撤与额外换手。训练及验证日志都记录双方对应字段；不能以 agent 单独收益替代比较。

## Replay 审计

```bash
CUDA_VISIBLE_DEVICES=0 ${HOME}/anaconda3/envs/test/bin/python \
  ${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/rl_position_manager_v1/evaluate_ppo_position_manager.py \
  --checkpoint <policy_last.pt> \
  --quantx-root ${HOME}/git/quantization/quantx \
  --provider-uri ${HOME}/git/quantization/quantx/data/qlib_data_fixed \
  --output-dir <replay_audit_dir> --start 2024-01-02 --end 2024-12-31 --device cuda
```

此步骤只用于 source 策略的离线审计，不能替代正式结论。

完成训练后，先固定 checkpoint，再生成同窗口的一对正式配置：

```bash
${HOME}/anaconda3/envs/test/bin/python "$RL_ROOT/select_ppo_checkpoint.py" \
  --run-dir <ppo_run_dir>

${HOME}/anaconda3/envs/test/bin/python "$RL_ROOT/render_ppo_formal_config.py" \
  --baseline "$BASELINE" --checkpoint <selected_checkpoint> \
  --start 2025-01-02 --end 2026-06-16 \
  --baseline-output <oos_baseline.yaml> --output <oos_ppo.yaml>
```

## 正式 QuantX 回测

```bash
RL_ROOT=${HOME}/git/quantization/QuantX-QMT-qmt-mac/tmp/weak-to-strong-diffusion-v1/rl_position_manager_v1
BASELINE=${HOME}/git/quantization/quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_top5_exit_policy_combo_v1/reward_transformer_mainboard_epoch020_5d_top5_h30_trail_p20_dd12_2020_2026.yaml

${HOME}/anaconda3/envs/test/bin/python "$RL_ROOT/render_ppo_formal_config.py" \
  --baseline "$BASELINE" --checkpoint <policy_last.pt> \
  --start 2025-01-02 --end 2026-06-16 \
  --output ${HOME}/git/quantization/quantx/configs/strategies/generated/reward_transformer_mainboard_epoch020_top5_ppo_v1.yaml

cd ${HOME}/git/quantization/quantx
PYTHONPATH=. ${HOME}/anaconda3/envs/test/bin/python -m quantx.tools.run_backtest \
  --config configs/strategies/generated/reward_transformer_mainboard_epoch020_top5_ppo_v1.yaml \
  --output-dir artifacts/backtest_runs --json
```

正式 run 会生成常规 QuantX artifacts，另额外保存 `rl_decisions.parquet`。该文件记录执行日、信号日、
TopK 候选、持仓、PPO 原始动作、计划卖出比例、候选权重和现金部署；PPO 的额外动作叠加在原策略
卖出规则之上，实际成交仍以 `trades.json` 为准。
不得使用 replay 指标替代正式 QuantX 回测结果。

也可运行 `run_post_training_pipeline.sh`。它只根据 2024 validation 选 checkpoint，再顺序执行 source replay
审计及 paired baseline/PPO OOS QuantX 回测；`--wait-pid` 可让它等待正在进行的训练完成。
