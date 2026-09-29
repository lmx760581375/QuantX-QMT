#!/usr/bin/env python3
"""Return latest final-project run summary as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    candidates = sorted(
        [path for path in runs.glob("*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if runs.is_dir() else []
    if not candidates:
        print(json.dumps({"ok": False, "message": "no runs found"}, ensure_ascii=False))
        return 1
    run = candidates[0]
    summary = read_json(run / "summary.json")
    metrics = read_json(run / "metrics.json")
    print(json.dumps({
        "ok": True,
        "run_id": run.name,
        "path": str(run),
        "name": summary.get("name"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "trade_count": metrics.get("trade_count"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
