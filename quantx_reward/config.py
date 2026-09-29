"""项目路径、YAML 和命令参数辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

PATH_KEYS = {
    "root",
    "kronos_root",
    "checkpoint",
    "ae_checkpoint",
    "scaler_path",
    "output",
    "fold_path",
    "labels",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    return payload


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_data_paths(data_root: str | Path) -> dict[str, Path]:
    """支持传入 QuantX data 根目录，或直接传入 Qlib provider 目录。"""
    root = Path(data_root).expanduser().resolve()
    if (root / "calendars" / "day.txt").is_file():
        provider = root
        data_dir = root.parent
    else:
        provider = root / "qlib_data_fixed"
        data_dir = root
    raw_stock_dir = data_dir / "raw" / "baostock" / "stocks"
    security_master = data_dir / "meta" / "snapshots" / "security_master.csv"
    return {
        "data_root": data_dir,
        "provider_uri": provider,
        "raw_stock_dir": raw_stock_dir,
        "security_master": security_master,
    }


def args_mapping_to_cli(args: dict[str, Any], *, positional_keys: tuple[str, ...] = ()) -> list[str]:
    output: list[str] = []
    for key in positional_keys:
        value = args.get(key)
        if value is not None:
            output.append(str(value))
    for key, raw_value in args.items():
        if key in positional_keys or raw_value is None:
            continue
        option = f"--{key.replace('_', '-')}"
        value = raw_value
        if key in PATH_KEYS and isinstance(value, str) and value:
            value = str(resolve_repo_path(value))
        if isinstance(value, bool):
            if value:
                output.append(option)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        output.extend([option, str(value)])
    return output


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
