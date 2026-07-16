"""Code fingerprints that include committed, dirty, and untracked research code."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def compute_code_fingerprint(root: str | Path) -> str:
    repository = Path(root).resolve()
    commit = _git(repository, "rev-parse", "HEAD").strip()
    diff = _git(repository, "diff", "--binary", "HEAD")
    untracked = sorted(
        line for line in _git(repository, "ls-files", "--others", "--exclude-standard").splitlines() if line
    )
    digest = hashlib.sha256()
    digest.update(commit.encode("utf-8"))
    digest.update(diff.encode("utf-8"))
    for relative in untracked:
        path = repository / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout
