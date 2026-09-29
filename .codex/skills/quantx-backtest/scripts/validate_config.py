#!/usr/bin/env python3
"""Validate a formal strategy YAML with the final QuantX runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Strategy config must be a mapping: {config_path}")
    data_root = Path(args.data_root).expanduser().resolve()
    provider = data_root if (data_root / "calendars" / "day.txt").is_file() else data_root / "qlib_data_fixed"
    config.setdefault("data", {})["provider_uri"] = str(provider)
    with tempfile.TemporaryDirectory() as directory:
        resolved = Path(directory) / "strategy.yaml"
        resolved.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "quantx.tools.run_backtest",
            "--config",
            str(resolved),
            "--dry-run",
            "--json",
        ]
        return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
