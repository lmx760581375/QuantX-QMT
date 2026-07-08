#!/usr/bin/env python3
"""Return the latest QuantX run as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_latest(root: Path) -> dict:
    runs = root / "runs"
    if not runs.exists():
        return {"ok": False, "message": "runs directory not found", "root": str(root)}
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    if not candidates:
        return {"ok": False, "message": "no runs found", "root": str(root)}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run = candidates[0]
    summary = _read_json(run / "summary.json") or {}
    metrics = _read_json(run / "metrics.json") or {}
    return {
        "ok": True,
        "run_id": run.name,
        "path": str(run),
        "name": summary.get("name"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "total_return": metrics.get("total_return", summary.get("total_return")),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "trade_count": metrics.get("trade_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QuantX repo root")
    args = parser.parse_args()
    print(json.dumps(find_latest(Path(args.root).expanduser().resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
