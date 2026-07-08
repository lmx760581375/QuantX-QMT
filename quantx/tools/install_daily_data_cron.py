"""Install or print QuantX daily production cron entries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


BEGIN = "# BEGIN QUANTX DAILY DATA UPDATE"
END = "# END QUANTX DAILY DATA UPDATE"


def _markers(label: str | None = None) -> tuple[str, str]:
    if not label:
        return BEGIN, END
    clean = " ".join(str(label).upper().split())
    return f"# BEGIN QUANTX {clean}", f"# END QUANTX {clean}"


def cron_block(
    root: str,
    profile: str,
    python_bin: str,
    log_file: str,
    hour: int = 17,
    minute: int = 30,
    days: str = "*",
    stage: str = "data",
    extra_args: List[str] | None = None,
    label: str | None = None,
) -> str:
    begin, end = _markers(label)
    extra = " ".join(extra_args or [])
    command = (
        f"cd {root} && {python_bin} -m quantx.tools.run_daily_pipeline "
        f"--profile {profile} --stage {stage} {extra} --json >> {log_file} 2>&1"
    )
    command = " ".join(command.split())
    return "\n".join([begin, f"{minute} {hour} * * {days} {command}", end])


def install_block(block: str) -> str:
    block_lines = [line.strip() for line in block.splitlines() if line.strip()]
    if block_lines and block_lines[0].startswith("# BEGIN ") and block_lines[-1].startswith("# END "):
        begin = block_lines[0]
        end = block_lines[-1]
    else:
        begin = BEGIN
        end = END
    current = _current_crontab()
    lines = current.splitlines()
    cleaned: List[str] = []
    skipping = False
    for line in lines:
        if line.strip() == begin:
            skipping = True
            continue
        if line.strip() == end:
            skipping = False
            continue
        if not skipping:
            cleaned.append(line)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    new_text = "\n".join([*cleaned, block]).strip() + "\n"
    subprocess.run(["crontab", "-"], input=new_text, text=True, check=True)
    return new_text


def _current_crontab() -> str:
    proc = subprocess.run(["crontab", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()), help="QuantX repo root.")
    parser.add_argument("--profile", default="configs/production/daily_default.yaml")
    parser.add_argument("--python-bin", default="/home/users/mingxiao.li/anaconda3/envs/test/bin/python")
    parser.add_argument("--log-file", default="/tmp/quantx_daily_data.log")
    parser.add_argument("--hour", type=int, default=17)
    parser.add_argument("--minute", type=int, default=30)
    parser.add_argument("--days", default="*", help="Cron day-of-week field. Default '*' runs every day; use '1-5' for weekdays.")
    parser.add_argument("--stage", default="data", choices=["all", "data", "signals", "report", "mail"])
    parser.add_argument("--skip-data-update", action="store_true", help="Add --skip-data-update to run_daily_pipeline.")
    parser.add_argument("--force-send", action="store_true", help="Add --force-send to run_daily_pipeline.")
    parser.add_argument("--label", help="Cron marker label. Defaults to data update or nightly full run.")
    parser.add_argument("--weekdays-only", action="store_true", help="Shortcut for --days 1-5.")
    parser.add_argument("--install", action="store_true", help="Write the cron entry to current user's crontab.")
    args = parser.parse_args(argv)

    root = str(Path(args.root).resolve())
    extra_args = []
    if args.skip_data_update:
        extra_args.append("--skip-data-update")
    if args.force_send:
        extra_args.append("--force-send")
    label = args.label or ("DAILY FULL RUN" if args.stage == "all" else None)
    block = cron_block(
        root=root,
        profile=args.profile,
        python_bin=args.python_bin,
        log_file=args.log_file,
        hour=args.hour,
        minute=args.minute,
        days="1-5" if args.weekdays_only else args.days,
        stage=args.stage,
        extra_args=extra_args,
        label=label,
    )
    if args.install:
        install_block(block)
        print("installed")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
