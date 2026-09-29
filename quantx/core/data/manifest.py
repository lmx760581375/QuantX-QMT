"""可复现日线数据目录的内容摘要与校验。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


DATA_MANIFEST_KIND = "quantx_daily_data_manifest_v1"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def files_digest(root: str | Path, files: Iterable[Path]) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    selected = sorted(
        {Path(path).resolve() for path in files if Path(path).is_file()},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in selected:
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def tree_digest(root: str | Path, patterns: Iterable[str]) -> dict[str, Any]:
    """对目录内文件内容和相对路径生成顺序稳定的聚合摘要。"""

    root = Path(root).expanduser().resolve()
    return files_digest(
        root,
        (
            path
            for pattern in patterns
            for path in root.glob(pattern)
        ),
    )


def _is_model_symbol(symbol: str) -> bool:
    return (
        symbol.startswith("SH6")
        or (symbol.startswith("SZ0") and not symbol.startswith("SZ399"))
        or (symbol.startswith("SZ3") and not symbol.startswith("SZ399"))
    )


def _coverage(provider_uri: Path) -> tuple[dict[str, Any], list[str]]:
    calendar_path = provider_uri / "calendars" / "day.txt"
    instruments_path = provider_uri / "instruments" / "all.txt"
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Missing Qlib calendar: {calendar_path}")
    if not instruments_path.is_file():
        raise FileNotFoundError(f"Missing Qlib instruments: {instruments_path}")
    dates = [
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    instrument_lines = [
        line.strip()
        for line in instruments_path.read_text(encoding="utf-8").splitlines()
        if line.split() and _is_model_symbol(line.split()[0])
    ]
    instruments = [line.split()[0] for line in instrument_lines]
    if not dates:
        raise ValueError(f"Qlib calendar is empty: {calendar_path}")
    canonical_instruments = ("\n".join(instrument_lines) + "\n").encode("utf-8")
    return {
        "start": dates[0],
        "end": dates[-1],
        "date_count": len(dates),
        "instrument_count": len(instruments),
        "calendar_sha256": sha256_file(calendar_path),
        "instruments_sha256": hashlib.sha256(canonical_instruments).hexdigest(),
        "scope": "stock symbols used by the Reward feature store",
    }, instruments


def build_data_manifest(
    data_root: str | Path,
    *,
    source: str = "baostock",
    adjustflag: str = "2",
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    provider_uri = data_root / "qlib_data_fixed"
    raw_root = data_root / "raw" / "baostock"
    historical_st = data_root / "meta" / "snapshots" / "historical_st_daily.parquet"
    coverage, instruments = _coverage(provider_uri)
    raw_files = [raw_root / "stocks" / f"{symbol}.csv" for symbol in instruments]
    qlib_files = [provider_uri / "calendars" / "day.txt"]
    for symbol in instruments:
        qlib_files.extend((provider_uri / "features" / symbol.lower()).glob("*.bin"))
    payload = {
        "kind": DATA_MANIFEST_KIND,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source": source,
        "adjustflag": str(adjustflag),
        "coverage": coverage,
        "raw_stock_tree": files_digest(raw_root, raw_files),
        "qlib_tree": files_digest(provider_uri, qlib_files),
        "historical_st": (
            {
                "path": str(historical_st),
                "size": historical_st.stat().st_size,
                "sha256": sha256_file(historical_st),
            }
            if historical_st.is_file()
            else None
        ),
    }
    return payload


def write_data_manifest(
    data_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else data_root / "MANIFEST.json"
    )
    payload = build_data_manifest(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def verify_data_manifest(
    data_root: str | Path,
    reference_path: str | Path,
) -> dict[str, Any]:
    reference_path = Path(reference_path).expanduser().resolve()
    expected = json.loads(reference_path.read_text(encoding="utf-8"))
    if expected.get("kind") != DATA_MANIFEST_KIND:
        raise ValueError(f"Unexpected data manifest kind: {expected.get('kind')!r}")
    actual = build_data_manifest(data_root)
    checks = {}
    for field in ("source", "adjustflag"):
        checks[field] = {
            "expected": expected.get(field),
            "actual": actual.get(field),
            "ok": expected.get(field) == actual.get(field),
        }
    for field in ("start", "end", "date_count", "instrument_count"):
        expected_value = (expected.get("coverage") or {}).get(field)
        actual_value = (actual.get("coverage") or {}).get(field)
        checks[f"coverage.{field}"] = {
            "expected": expected_value,
            "actual": actual_value,
            "ok": expected_value == actual_value,
        }
    for section, field in (
        ("coverage", "calendar_sha256"),
        ("coverage", "instruments_sha256"),
        ("raw_stock_tree", "sha256"),
        ("qlib_tree", "sha256"),
    ):
        expected_value = (expected.get(section) or {}).get(field)
        actual_value = (actual.get(section) or {}).get(field)
        checks[f"{section}.{field}"] = {
            "expected": expected_value,
            "actual": actual_value,
            "ok": expected_value == actual_value,
        }
    expected_st = (expected.get("historical_st") or {}).get("sha256")
    actual_st = (actual.get("historical_st") or {}).get("sha256")
    if expected_st is not None:
        checks["historical_st.sha256"] = {
            "expected": expected_st,
            "actual": actual_st,
            "ok": expected_st == actual_st,
        }
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "reference": str(reference_path),
        "checks": checks,
        "actual": actual,
    }
