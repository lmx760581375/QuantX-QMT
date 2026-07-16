"""Refresh active-brick liquidity-rule score assets for daily production."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
DEFAULT_CANDIDATES = ROOT / "sh_index_bull_brick_candidates.parquet"
DEFAULT_CANDIDATE_JSON = ROOT / "sh_index_bull_brick_relationship.json"
DEFAULT_CANDIDATE_MD = ROOT / "sh_index_bull_brick_relationship.md"
DEFAULT_SCORES = ROOT / "walk_forward_active_brick_scores.parquet"
DEFAULT_SCORE_JSON = ROOT / "walk_forward_active_brick_ml.json"
DEFAULT_SCORE_MD = ROOT / "walk_forward_active_brick_ml.md"
DEFAULT_ASSET = Path("data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_scores_2025_2026.parquet")
DEFAULT_METADATA = Path("data/derived/active_value_brick_ml/liquidity_rule_turnover20_60_mv20_80_metadata.json")
DEFAULT_OOS_START = "2025-01-02"
DEFAULT_END = "2026-07-15"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-start", default=DEFAULT_OOS_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--candidates-output", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--candidate-json-output", default=str(DEFAULT_CANDIDATE_JSON))
    parser.add_argument("--candidate-markdown-output", default=str(DEFAULT_CANDIDATE_MD))
    parser.add_argument("--scores-output", default=str(DEFAULT_SCORES))
    parser.add_argument("--score-json-output", default=str(DEFAULT_SCORE_JSON))
    parser.add_argument("--score-markdown-output", default=str(DEFAULT_SCORE_MD))
    parser.add_argument("--score-asset-output", default=str(DEFAULT_ASSET))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--min-rows", type=int, default=900)
    parser.add_argument("--min-signal-days", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = refresh(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok={report['ok']} dry_run={report['dry_run']} "
            f"asset={report['outputs']['score_asset']} metadata={report['outputs']['metadata']}"
        )
    return 0


def refresh(args: argparse.Namespace) -> dict:
    commands = build_commands(args)
    outputs = {
        "candidates": str(Path(args.candidates_output)),
        "scores": str(Path(args.scores_output)),
        "score_asset": str(Path(args.score_asset_output)),
        "metadata": str(Path(args.metadata_output)),
    }
    if args.dry_run:
        checks = dry_run_checks(args)
        return {
            "ok": all(row["ok"] for row in checks),
            "dry_run": True,
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "commands": commands,
            "outputs": outputs,
            "checks": checks,
            "note": "Dry-run validates the production refresh plan and current artifacts without rebuilding the heavy scored panel.",
        }

    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
        result = {
            "command": command,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(result)
        if completed.returncode != 0:
            return {
                "ok": False,
                "dry_run": False,
                "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                "commands": commands,
                "outputs": outputs,
                "results": results,
            }
    return {
        "ok": True,
        "dry_run": False,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "commands": commands,
        "outputs": outputs,
        "results": results,
    }


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    py = sys.executable
    return [
        [
            py,
            "-m",
            "quantx.tools.build_active_brick_candidates",
            "--candidates-output",
            str(args.candidates_output),
            "--json-output",
            str(args.candidate_json_output),
            "--markdown-output",
            str(args.candidate_markdown_output),
            "--oos-start",
            str(args.oos_start),
            "--end",
            str(args.end),
            "--json",
        ],
        [
            py,
            "-m",
            "quantx.tools.train_active_brick_ml_scores",
            "--candidate-path",
            str(args.candidates_output),
            "--scores-output",
            str(args.scores_output),
            "--json-output",
            str(args.score_json_output),
            "--markdown-output",
            str(args.score_markdown_output),
            "--oos-start",
            str(args.oos_start),
            "--end",
            str(args.end),
            "--json",
        ],
        [
            py,
            "-m",
            "quantx.tools.export_active_brick_ml_scores",
            "--source",
            str(args.scores_output),
            "--output",
            str(args.score_asset_output),
            "--metadata-output",
            str(args.metadata_output),
            "--score-col",
            "rule_score",
            "--start",
            str(args.oos_start),
            "--end",
            str(args.end),
            "--turnover-rank-low",
            "0.20",
            "--turnover-rank-high",
            "0.60",
            "--free-mv-rank-low",
            "0.20",
            "--free-mv-rank-high",
            "0.80",
            "--min-rows",
            str(int(args.min_rows)),
            "--min-signal-days",
            str(int(args.min_signal_days)),
            "--json",
        ],
    ]


def dry_run_checks(args: argparse.Namespace) -> list[dict]:
    required_inputs = [
        Path("data/derived/market_regime/sh000001_daily.parquet"),
        Path("data/reference/security_state/st_daily.csv"),
        Path("data/derived/0amv/daily.csv"),
        Path(args.candidates_output),
        Path(args.scores_output),
        Path(args.score_asset_output),
        Path(args.metadata_output),
    ]
    checks = []
    for path in required_inputs:
        checks.append({"path": str(path), "ok": path.exists(), "size": path.stat().st_size if path.exists() else None})
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
