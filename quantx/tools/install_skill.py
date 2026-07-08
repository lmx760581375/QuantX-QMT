"""Install QuantX Codex skills into the local Codex skills directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILL = PROJECT_ROOT / "tools" / "codex_skills" / "quantx-backtest"


def default_target() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "skills" / "quantx-backtest"


def install_skill(source: Path = DEFAULT_SKILL, target: Path | None = None, force: bool = True) -> Dict[str, Any]:
    source = source.expanduser().resolve()
    target = (target or default_target()).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Skill source not found: {source}")
    if not (source / "SKILL.md").exists():
        raise FileNotFoundError(f"SKILL.md not found under: {source}")
    if target.exists():
        if not force:
            raise FileExistsError(f"Target already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    copied: List[str] = [str(path.relative_to(target)) for path in sorted(target.rglob("*")) if path.is_file()]
    return {
        "ok": True,
        "source": str(source),
        "target": str(target),
        "files": copied,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SKILL), help="Skill source directory")
    parser.add_argument("--target", default=str(default_target()), help="Install target directory")
    parser.add_argument("--no-force", action="store_true", help="Fail if target already exists")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = install_skill(Path(args.source), Path(args.target), force=not args.no_force)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Installed {result['source']} -> {result['target']}")
        print(f"Files: {len(result['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
