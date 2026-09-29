#!/usr/bin/env python3
"""Probe final QuantX Reward repository readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    result = {
        "ok": (root / "pyproject.toml").is_file(),
        "root": str(root),
        "has_final_cli": (root / "quantx_reward" / "cli.py").is_file(),
        "has_backtest": (root / "quantx" / "tools" / "run_backtest.py").is_file(),
        "has_reward_train": (root / "models" / "reward" / "train.py").is_file(),
        "has_reward_infer": (root / "models" / "reward" / "infer.py").is_file(),
        "has_weight_manifest": (root / "weights" / "reward_v1" / "MANIFEST.json").is_file(),
        "has_formal_strategies": (
            (root / "configs" / "strategies" / "weak_to_strong.yaml").is_file()
            and (root / "configs" / "strategies" / "reward_h15.yaml").is_file()
        ),
        "run_count": len([path for path in (root / "runs").glob("*") if path.is_dir()])
        if (root / "runs").is_dir()
        else 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
