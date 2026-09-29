# Reward Model

当前正式模型是 H7 absolute good/bad full-pair epoch9。

核心定义、网络结构、训练边界与 score 合同见
[Reward Model 正式合同](system/reward-model-contract.md)。

运行入口：

```bash
quantx-reward prepare-h7-labels
quantx-reward train
quantx-reward infer
```

旧的三头多任务 epoch10 模型保留在 Git 历史和研究归档中，不再是 `reward_v1` 的正式权重。
