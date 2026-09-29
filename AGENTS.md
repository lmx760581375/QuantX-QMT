# 开发约束

- 常用 Python 环境为 conda `test`。
- 安装依赖优先使用 `https://pypi.hobot.cc/simple`。
- 修改代码前先给出方案并取得确认。
- 优先最小改动、高内聚、低耦合，不增加无必要实体。
- 文档和说明优先使用中文。
- 新策略配置先 dry-run，再执行完整回测。
- 不提交 `data/`、`workdirs/`、`runs/`、score parquet 或训练中间 checkpoint。
- 只有 `weights/reward_v1/` 清单声明的正式推理权重分块允许进入 Git。
