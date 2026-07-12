"""Compatibility entry point for the standard Wufu strategy YAML."""

from __future__ import annotations

import sys
from typing import List

from quantx.tools.run_backtest import main as run_standard_backtest


DEFAULT_CONFIG = "configs/strategies/generated/etf_wufu_qmt_next_open.yaml"


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in args:
        args = ["--config", DEFAULT_CONFIG, *args]
    return run_standard_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
