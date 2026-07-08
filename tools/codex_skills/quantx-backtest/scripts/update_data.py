#!/usr/bin/env python3
"""Run QuantX's agent-facing adjustment-aware data update CLI."""

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
    parser.add_argument("--provider-uri", default="data/qlib_data_fixed")
    parser.add_argument("--raw-dir", default="data/raw/baostock")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--symbol-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--end-date")
    parser.add_argument("--overlap-days", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--full-refresh-start", default="2010-01-01")
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    parser.add_argument("--max-requests", type=int, default=45000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = str(Path(args.root).expanduser().resolve())
    cmd = [
        args.python_bin,
        "-m",
        "quantx.tools.agent_context",
        "--root",
        root,
        "data-update",
        "--provider-uri",
        args.provider_uri,
        "--raw-dir",
        args.raw_dir,
        "--overlap-days",
        str(args.overlap_days),
        "--tolerance",
        str(args.tolerance),
        "--full-refresh-start",
        args.full_refresh_start,
        "--pause-seconds",
        str(args.pause_seconds),
        "--socket-timeout",
        str(args.socket_timeout),
        "--max-requests",
        str(args.max_requests),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.symbols:
        cmd.extend(["--symbols", *args.symbols])
    if args.symbol_file:
        cmd.extend(["--symbol-file", args.symbol_file])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
