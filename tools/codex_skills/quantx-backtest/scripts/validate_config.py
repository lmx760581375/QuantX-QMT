#!/usr/bin/env python3
"""Run QuantX config validation through run_backtest dry-run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QuantX repo root")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default="conda run -n test python")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    cmd = args.python.split() + [
        "-m",
        "quantx.tools.run_backtest",
        "--config",
        args.config,
        "--dry-run",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    payload = {
        "ok": proc.returncode == 0,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        payload["result"] = json.loads(proc.stdout)
    except Exception:
        payload["result"] = None
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
