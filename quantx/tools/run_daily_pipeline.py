"""Run the QuantX daily production pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from quantx.production import DailyPipeline
from quantx.production.config import load_profile


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Path to production profile YAML.")
    parser.add_argument("--trade-date", help="Override trade date, defaults to provider latest date.")
    parser.add_argument("--stage", default="all", choices=["all", "data", "signals", "report", "mail"])
    parser.add_argument("--strategy", help="Only run strategies whose path contains this text.")
    parser.add_argument("--symbol-limit", type=int, help="Limit symbols for smoke/dry runs.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send email; still writes preview artifacts.")
    parser.add_argument("--force-data-update", action="store_true", help="Run update_command even in verify-only mode.")
    parser.add_argument("--skip-data-update", action="store_true", help="Verify local data only; do not run update_command.")
    parser.add_argument("--force-send", action="store_true", help="Bypass email idempotency check.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    profile = load_profile(args.profile, project_root=root)
    result = DailyPipeline(profile).run(
        stage=args.stage,
        trade_date=args.trade_date,
        dry_run=args.dry_run,
        force_send=args.force_send,
        force_data_update=args.force_data_update,
        skip_data_update=args.skip_data_update,
        strategy_filter=args.strategy,
        symbol_limit=args.symbol_limit,
    )
    data = result.to_dict()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"ok={data['ok']} profile={data['profile']} trade_date={data['trade_date']} run_dir={data['run_dir']}")
        if data.get("mail_status"):
            print(f"mail={data['mail_status']}")
        if data.get("errors"):
            print(f"errors={data['errors']}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
