"""Data update and verification helpers for daily production."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TextIO


def verify_provider(provider_uri: str | Path) -> Dict[str, Any]:
    """Return a lightweight health summary for a qlib-style provider."""
    base = Path(provider_uri)
    calendar_path = base / "calendars" / "day.txt"
    instruments_path = base / "instruments" / "all.txt"
    dates = _read_lines(calendar_path)
    instruments = _read_lines(instruments_path)
    feature_count = len(list((base / "features").glob("*/*.bin"))) if (base / "features").exists() else 0
    ok = bool(base.exists() and dates and instruments and feature_count)
    return {
        "ok": ok,
        "provider_uri": str(provider_uri),
        "exists": base.exists(),
        "calendar_start": dates[0] if dates else None,
        "calendar_end": dates[-1] if dates else None,
        "calendar_days": len(dates),
        "instrument_count": len(instruments),
        "feature_file_count": feature_count,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


def latest_trade_date(provider_uri: str | Path) -> str:
    dates = _read_lines(Path(provider_uri) / "calendars" / "day.txt")
    if not dates:
        raise FileNotFoundError(f"Missing or empty qlib calendar: {provider_uri}")
    return dates[-1]


class _TailBuffer:
    def __init__(self, limit: int = 8000):
        self.limit = int(limit)
        self._text = ""

    def append(self, text: str) -> None:
        self._text = (self._text + text)[-self.limit :]

    def text(self) -> str:
        return self._text


def run_update_command(
    command: List[str],
    cwd: str | Path | None = None,
    timeout_seconds: int = 21600,
    stream_output: bool = True,
) -> Dict[str, Any]:
    """Run an optional data update command, streaming progress while keeping output tails."""
    if not command:
        return {"ok": True, "skipped": True, "message": "no update_command configured"}
    command = _resolve_update_command(command)
    started = datetime.now()
    stdout_tail = _TailBuffer()
    stderr_tail = _TailBuffer()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_thread = threading.Thread(
        target=_pump_pipe,
        args=(proc.stdout, sys.stdout, stdout_tail, stream_output),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_pipe,
        args=(proc.stderr, sys.stderr, stderr_tail, stream_output),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        returncode = proc.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    finished = datetime.now()
    return {
        "ok": returncode == 0 and not timed_out,
        "skipped": False,
        "command": command,
        "returncode": returncode,
        "stdout": stdout_tail.text(),
        "stderr": stderr_tail.text(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "timeout_seconds": timeout_seconds,
        **({"error_type": "timeout", "message": f"data update command timed out after {timeout_seconds}s"} if timed_out else {}),
    }


def _resolve_update_command(command: List[str]) -> List[str]:
    resolved = list(command)
    if resolved and resolved[0] in {"python", "python3"}:
        resolved[0] = sys.executable
    return resolved


def _pump_pipe(pipe: TextIO | None, target: TextIO, tail: _TailBuffer, stream_output: bool) -> None:
    if pipe is None:
        return
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            tail.append(line)
            if stream_output:
                target.write(line)
                target.flush()
    finally:
        pipe.close()


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
