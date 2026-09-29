#!/usr/bin/env python3
"""Inspect final QuantX Reward run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    run_id = args.run_id
    if run_id == "latest":
        candidates = sorted(
            [path for path in runs.glob("*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if runs.is_dir() else []
        if not candidates:
            print(json.dumps({"ok": False, "message": "no runs found"}, ensure_ascii=False))
            return 1
        run_id = candidates[0].name
    run = runs / run_id
    if not run.is_dir():
        print(json.dumps({"ok": False, "message": f"run not found: {run_id}"}, ensure_ascii=False))
        return 1
    trades = read_json(run / "trades.json", [])
    rejected = [row for row in trades if row.get("reject_reason")]
    print(json.dumps({
        "ok": True,
        "run_id": run_id,
        "summary": read_json(run / "summary.json", {}),
        "metrics": read_json(run / "metrics.json", {}),
        "trade_count": len([row for row in trades if not row.get("reject_reason")]),
        "reject_count": len(rejected),
        "reject_reasons": dict(Counter(row.get("reject_reason") for row in rejected)),
        "artifacts": sorted(path.name for path in run.glob("*") if path.is_file()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
