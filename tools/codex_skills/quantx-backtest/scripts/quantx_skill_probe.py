#!/usr/bin/env python3
"""Probe a QuantX repository for agent-skill readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QuantX repo root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    configs = root / "configs" / "strategies"
    result = {
        "ok": root.exists() and (root / "pyproject.toml").exists(),
        "root": str(root),
        "has_pyproject": (root / "pyproject.toml").exists(),
        "has_run_backtest": (root / "quantx" / "tools" / "run_backtest.py").exists(),
        "has_update_meta": (root / "quantx" / "tools" / "update_meta.py").exists(),
        "has_agent_context": (root / "quantx" / "tools" / "agent_context.py").exists(),
        "has_qlib_data_fixed": (root / "data" / "qlib_data_fixed").exists(),
        "has_meta_snapshots": (
            (root / "data" / "meta" / "snapshots" / "security_master.csv").exists()
            and (root / "data" / "meta" / "snapshots" / "industry_membership.csv").exists()
        ),
        "config_count": len(list(configs.rglob("*.yaml"))) + len(list(configs.rglob("*.yml"))) if configs.exists() else 0,
        "run_count": len([p for p in runs.iterdir() if p.is_dir()]) if runs.exists() else 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
