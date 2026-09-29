# 冻结 AE 设计与表征边界

## 为什么保留 AE

最终 Reward Transformer 不直接读取 60×52 的原始个股张量。它读取一个先在历史数据上训练、在 Reward 训练和推理期间完全冻结的表征：

```text
60×52 个股序列 + 60×20 市场序列 + 6 维候选上下文
    -> Transformer AutoEncoder
    -> 27×128 latent token
    -> Reward Transformer
```

这不是把 AE 当作未来收益预测器。AE 的职责是压缩历史状态；Reward head 的职责才是同日排序。

## 正式 AE 配置

冻结权重为 `weights/reward_v1/ae_epoch020.pt`，manifest 中 SHA-256 为：

```text
d4b1532b8124c70215d3a067820748dcd1026f794f3c4f53bf1467fc9a2867f1
```

正式配置来自 checkpoint：

| 参数 | 值 |
| --- | ---: |
| lookback | 60 |
| 输入个股维度 | 52 |
| 输入市场维度 | 20 |
| 候选维度 | 6 |
| 个股投影维度 `d_stock` | 64 |
| 市场投影维度 `d_market` | 32 |
| 候选投影维度 `d_candidate` | 16 |
| 融合维度 `d_model` | 96 |
| encoder 层数 | 4 |
| encoder heads | 4 |
| latent tokens | 27 |
| latent dim | 128 |
| 未来路径长度配置 | 30 |

历史原始日志记录该 AE 有 40,427,736 个冻结参数。当前正式仓库提供其结构和权重，但不把 AE 预训练当作本次 v1 的可调组件；Reward v1 固定使用该 epoch20 表征。

## 编码器

对 batch 中每个样本：

1. `stock_proj: Linear(52, 64)`，逐时间步投影个股输入；
2. `market_proj: Linear(20, 32)`，逐时间步投影市场输入；
3. `candidate_proj: Linear(6, 16)`，将候选向量投影并复制到所有 60 个时间步；
4. 拼接得到 112 维，再经 `fusion: Linear(112, 96)`；
5. 加可学习位置参数 `pos[1,60,96]`；
6. 经过 4 层 PyTorch `TransformerEncoderLayer`：
   - hidden 96；
   - 4 heads；
   - FFN 宽度 $96×4=384$；
   - GELU；
   - dropout 0.1；
   - `batch_first=True`。

得到的编码状态形状为 `B×60×96`。

## 压缩为 27×128 token

编码状态先 flatten：

```text
B×60×96 = B×5760
```

随后：

```text
Linear(5760, 27×128=3456)
-> GELU
-> LayerNorm(3456)
-> reshape(B, 27, 128)
```

所以每个 Reward 样本的条件表征是 27 个 128 维 token，而不是单一向量。Reward Transformer 可以在这些 token 上再次做双向 attention。

## 解码器与训练目标

AE 解码路径：

```text
27×128
-> Linear(3456, 5760)
-> GELU + LayerNorm
-> reshape(B,60,96)
-> 1 层 TransformerEncoder
-> stock_out: Linear(96,52)
-> market_out: Linear(96,20)
```

它重建：

- 标准化后的 60×52 个股序列；
- 标准化后的 60×20 市场序列。

候选 6 维上下文只参与编码，不有独立候选重建头。这样候选身份可辅助表征历史状态，但不会把“候选名次本身”作为被重建的主要目标。

## 冻结边界

Reward 训练加载 AE 后：

- 设置 `eval()`；
- 所有 AE 参数 `requires_grad_(False)`；
- Reward optimizer 只包含 Reward Transformer 参数；
- checkpoint 记录 AE 路径和完整 AE config。

因此 Reward 的正式收益不能解释为“Reward 训练同时重新调了表征”；它严格是在固定 `27×128` 条件空间上学习排序分数。

## 重要限制

1. 这份仓库的 v1 复现对象是“冻结 AE + Reward v1”，不是从零搜索 AE 结构。
2. `models/reward/pipeline/common.py` 仍保留历史生成模型组件，原因是要兼容已冻结 checkpoint 的 `ModelConfig`；正式 H7 Reward inference 不调用生成采样分支。
3. 若未来重训或替换 AE，必须创建新版本、重新进行时间切分和 Reward 训练；不能把新 AE 接到旧 Reward checkpoint 上。
