"""正式 Reward 模型推理与 score artifact 导出入口。"""

from .pipeline.infer_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
