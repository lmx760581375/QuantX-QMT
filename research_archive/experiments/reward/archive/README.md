# 历史原始材料归档

本目录保存从旧研究工作区迁入的原始文本，以保留失败尝试、实验协议、路径依赖和当时的具体结论。

| 文件 | 角色 | SHA-256 |
| --- | --- | --- |
| `RESEARCH_LOG_2026-07.md` | 2026-07 的详细研究日志，包含 Diffusion、Decoder、Reward、PPO、市场状态与失败对照。 | `7fa1b2e5f37d789aab2aebe31a207493b196e2d69e91166e15d82f3e60ee3be9` |
| `FORMAL_MODEL_RESEARCH_PROTOCOL.md` | 正式模型研究和 QuantX 证据流程。 | `4abb878a05dafe7d1449ffebf941dfb9e006dce149744d72fb692fdeb027c897` |
| `LOOP_PREFERENCE_EXPERIMENT.md` | 固定步 Loop Preference 的预注册实验设计。 | `54ddde58002eb97cde4450cbc19ec04ec29ee7b6a88c796becc497d7c8beac91` |
| `weak_to_strong_pattern_diagnostics/` | 482 笔弱转强历史交易的 cluster、代表图、反例图和静态报告。 | 配置/总结哈希见目录内材料。 |

## 使用规则

- 这些文件是历史原文，不做后验改写。
- 它们包含旧仓库路径、临时脚本名、已拒绝模型和已作废结论。
- 当前执行面只认 `configs/`、`weights/reward_v1/`、`docs/system/`、`docs/strategies/` 和 `docs/decisions/`。
- 查阅历史结论时，先阅读 `docs/research/experiment-index.md` 与 `docs/research/methodology-corrections.md`，确认该结论是否仍有效。
- 弱转强图片的解释见 `docs/research/weak-to-strong-pattern-diagnostic.md`。

弱转强图像诊断的非图片原始材料哈希：

```text
config.json:          0ebfa13962dce617acf055e8ed97eef34c65f6b938153997f4becd388de7ad35
pattern_summary.json: f6e3d48276eccbb6bea234665b84a33a6603dcb530ec25b40390cbf2bfddac65
report.html:          65a453da8d9ba787c30dc03288b0813e709fea089e3afe67c99890b08fd13bb7
```
