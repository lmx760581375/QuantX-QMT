# Minute + Daily Cross-Attention Reward H7 设计方案

## 1. 目标

在保持原日线 Reward H7 任务周期和主要 Reward Transformer 容量的前提下，将分钟 AE 作为日线表征的辅助上下文，验证分钟数据是否能为日线 H7 return / drawdown 预测提供增量信息。

本方案不再把 64 个分钟 token 与 27 个日线 token 直接拼接，也不使用 candidate 特征。

## 2. 输入和冻结 AE

### 日线分支

- 输入窗口：60 个交易日；
- 日线 stock 特征：52 维；
- 日线 market 特征：20 维；
- 使用现有日线 AE 产生 `27 × 128` token；
- candidate 不作为新 Reward 模型输入。

### 分钟分支

- 输入窗口：240 根 5 分钟 K 线；
- 使用现有分钟 AE 产生 `64 × 128` token；
- 分钟 AE 和日线 AE 均冻结，第一阶段只训练融合 Reward 模型。

## 3. 融合结构

日线 token 作为 Query，分钟 token 作为 Key/Value：

```text
daily_tokens  [B, 27, 128]  ── Query ─┐
                                      ├─ Cross-Attention
minute_tokens [B, 64, 128]  ── K/V ──┘

输出 fused_daily_tokens [B, 27, 128]
```

建议第一版使用单层 cross-attention，避免在第一轮实验中引入过多自由度：

```python
minute_context = cross_attn(
    query=daily_tokens,
    key=minute_tokens,
    value=minute_tokens,
)
fused_daily = daily_tokens + minute_context
fused_daily = layer_norm(fused_daily)
```

工程上应使用独立的 daily/minute 输入投影和 modality positional embedding，避免两个 AE 输出虽然维度相同但统计分布不同，导致 cross-attention 难以区分 token 语义。

## 4. 日线 Reward Transformer

cross-attention 输出仍然只有 27 个日线 token：

```text
[B, 27, 128]
    ↓ input projection
[B, 27, 768]
    ↓ 7 层 Transformer，12 heads
[B, 27, 768]
```

对齐原日线 Reward baseline 的主要容量：

- `d_model = 768`；
- `layers = 7`；
- `heads = 12`；
- `dropout = 0.05`；
- 不加入 CLS token；
- 不做 mean pooling 或 attention pooling。

## 5. 输出 head

不把 `27 × 768` 一步压到标量，而采用两阶段降维：

```text
[B, 27, 768]
    ↓ 逐 token Linear(768, 256) + GELU + LayerNorm
[B, 27, 256]
    ↓ flatten
[B, 6912]
    ↓ Linear(6912, 256) + GELU + Dropout
[B, 256]
    ├─ Linear(256, 1) → return_logit
    └─ Linear(256, 1) → drawdown_logit
```

第一层在 27 个日线 token 上共享参数，第二层负责跨时间位置整合。这样既保留各日线位置的信息，又避免 `20736 → 1` 的过猛压缩。

## 6. 监督和时间周期

必须对齐原日线 H7：

- signal date 为 T；
- 使用 T+1 作为交易起点；
- 使用原日线 close-to-close H7 标签；
- return 与 drawdown 使用同日 pairwise 监督；
- 不使用当前分钟 fusion 的 `tail_4d_48_240` 目标。

第一版只训练两个 head：

- return；
- drawdown。

建议初始 loss 权重：

```text
return:    1.0
drawdown:  0.25
```

权重应写入 checkpoint metadata，不能在导出脚本中硬编码。

## 7. Score 导出和回测

导出时保留：

- `reward_return_logit`；
- `reward_drawdown_logit`；
- 经过明确后处理的最终 score。

第一轮比较至少保留两种 score：

1. return-only；
2. Return Top50 后，在候选池内按 drawdown rank 重排。

回测需要和原日线 baseline 对齐：

- H7；
- 相同 TopK；
- 相同买入 lag；
- 相同 holding rule；
- 相同交易成本；
- 相同日期区间。

## 8. 方案 Review

### 已解决的问题

- 不再把 64 个分钟 token 直接送入最终 Reward head；
- 最终预测只基于 27 个日线位置；
- 不使用 CLS 或 pooled token；
- Reward Transformer 容量与日线 baseline 对齐；
- H7 监督周期与日线 baseline 对齐；
- return/drawdown 采用独立输出 head；
- head 使用渐进式降维，避免一次性强压缩。

### 仍需特别验证的问题

1. **cross-attention 是否破坏日线表征**

   必须保留 `minute_mask` 或 zero-minute ablation，确认模型不使用分钟信息时可以退化为 daily-only。

2. **分钟和日线 token 的尺度差异**

   两个 AE 的 token 统计分布可能不同。建议记录两分支 token 的均值、标准差和范数，并在 cross-attention 前做独立 LayerNorm。

3. **现有日线 AE 的 candidate 接口**

   原 all-market baseline 使用的是固定 neutral candidate，不是策略 selector 特征。新 Reward 模型不使用 candidate，但调用旧 AE 时必须保持输入契约一致；不能把 neutral candidate 与全零 candidate 混用而不记录。

4. **head 参数量和过拟合**

   `Linear(6912, 256)` 约 1.77M 参数，规模可接受，但需要 dropout 和 weight decay。若验证集明显过拟合，可将中间维度从 256 降到 128。

5. **cross-attention 的额外自由度**

   第一版只使用单层 cross-attention。只有在 daily-only 与 minute fusion 的对照结果稳定后，才考虑增加 cross-attention 层数。

### 结论

该方案作为第一轮公平实验是合理的。它将分钟数据限定为日线 token 的上下文，不改变最终时间位置数量，并且把预测周期、Reward 主干容量和输出 head 设计与日线 baseline 对齐。

实现顺序应为：

1. 先实现 daily-only 版本，验证 H7 和新 head；
2. 加入 cross-attention，但将 minute token 置零，验证结构退化一致性；
3. 加入真实 minute token，比较增量效果；
4. 最后再研究分钟 AE 压缩率和 cross-attention 层数。

本文件只记录设计方案，不修改训练代码，也不代表训练任务已经提交。

## 9. Cache 生产约定

如果训练需要生成新的分钟或日线训练 cache，必须先写入本机 `/home` 路径完成生产和校验，再将完整 cache 复制到 bucket。训练阶段直接从本机 cache 读取；不要在 cache 生产过程中直接随机写 bucket，否则会显著放大远端随机 I/O 延迟。

推荐流程：

```text
本机 /home/.../cache
    ↓ 生产、校验 manifest、抽样读取
bucket/.../cache
    ↓ 训练任务只读
```

复制完成后至少核对文件数量、manifest、总字节数和关键 mmap shape，确认 bucket 版本完整后再提交集群任务。

## 10. Pair 采样策略

训练不把数十亿条合法两两组合全部物化，也不把当前的少量固定 bad options 视为完整 pair 空间。训练数据定义为同日内所有满足约束的合法 pair，训练时使用固定预算在线采样：

- return：从同日满足 return gap 的完整 bad pool 中按 epoch seed 采样；
- drawdown：从同日满足 return gap 和 drawdown percentile gap 的完整候选空间中拒绝采样；
- 每个 epoch 保持固定 pair budget，return/drawdown 两类任务平衡；
- 不同 epoch 使用不同 seed，覆盖完整 pair 空间而不展开成十亿级数组。
