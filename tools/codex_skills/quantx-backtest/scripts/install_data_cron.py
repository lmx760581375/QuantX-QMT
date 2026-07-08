#!/usr/bin/env python3
"""Print or install QuantX's 17:30 data update cron entry."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_ROOT = "/home/users/mingxiao.li/git/quantization/quantx"
DEFAULT_PYTHON = "/home/users/mingxiao.li/anaconda3/envs/test/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="QuantX repo root.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON, help="Python executable.")
    parser.add_argument("--profile", default="configs/production/daily_default.yaml")
    parser.add_argument("--log-file", default="/tmp/quantx_daily_data.log")
    parser.add_argument("--hour", type=int, default=17)
    parser.add_argument("--minute", type=int, default=30)
    parser.add_argument("--days", default="*", help="Cron day-of-week field; default '*' runs every day.")
    parser.add_argument("--weekdays-only", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    root = str(Path(args.root).expanduser().resolve())
    cmd = [
        args.python_bin,
        "-m",
        "quantx.tools.install_daily_data_cron",
        "--root",
        root,
        "--profile",
        args.profile,
        "--python-bin",
        args.python_bin,
        "--log-file",
        args.log_file,
        "--hour",
        str(args.hour),
        "--minute",
        str(args.minute),
        "--days",
        "1-5" if args.weekdays_only else args.days,
    ]
    if args.install:
        cmd.append("--install")
    return subprocess.run(cmd, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
