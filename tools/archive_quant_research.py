"""创建可安全提交 Git 的 QuantX Codex 对话与研究文档只读归档。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


HOME = Path.home()
QUANTIZATION_ROOT = HOME / "git" / "quantization"
CODEX_ROOT = HOME / ".codex"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "research_archive"

SESSION_ROOTS = (
    CODEX_ROOT / "sessions",
    CODEX_ROOT / "archived_sessions",
)

DOCUMENT_ROOTS = (
    ("qmt", QUANTIZATION_ROOT / "QuantX-QMT-qmt-mac" / "docs"),
    ("quantx", QUANTIZATION_ROOT / "quantx" / "docs"),
    ("root", QUANTIZATION_ROOT / "docs"),
    ("reward", QUANTIZATION_ROOT / "QuantX-reward" / "docs"),
    ("legacy_myquant_clean", QUANTIZATION_ROOT / "myquant-clean"),
    ("legacy_myquant_main", QUANTIZATION_ROOT / "myquant-main"),
    ("legacy_myquant_rl", QUANTIZATION_ROOT / "myquant-rl"),
    ("legacy_myquant_strategy", QUANTIZATION_ROOT / "myquant-strategy"),
    ("legacy_myquant_strategy_baseline", QUANTIZATION_ROOT / "myquant-strategy-baseline"),
    ("legacy_myyquant_1116", QUANTIZATION_ROOT / "myyquant-1116"),
    ("legacy_new_quant", QUANTIZATION_ROOT / "new-quant"),
    ("legacy_oskhquant", QUANTIZATION_ROOT / "OSkhQuant"),
)

WEAK_RESEARCH_ROOT = (
    QUANTIZATION_ROOT
    / "QuantX-QMT-qmt-mac"
    / "tmp"
    / "weak-to-strong-diffusion-v1"
)

SECRET_KEYS = (
    "password",
    "passwd",
    "job_password",
    "api_key",
    "hobot_api_key",
    "access_token",
    "refresh_token",
    "client_secret",
    "secret_key",
    "encrypt_passwd",
    "passphrase",
)

SECRET_KEY_PATTERN = "|".join(re.escape(value) for value in SECRET_KEYS)
SECRET_NAME_PATTERN = rf"(?:[A-Za-z0-9]+_)*(?:{SECRET_KEY_PATTERN})"
QUOTED_JSON_SECRET = re.compile(
    rf'(?i)(["\']?(?:{SECRET_NAME_PATTERN})["\']?\s*[:=]\s*)(["\'])(.*?)(\2)'
)
UNQUOTED_SECRET = re.compile(
    rf"(?im)(\b(?:{SECRET_NAME_PATTERN})\b\s*[:=]\s*)([^\s,;\]\}}]+)"
)
BEARER_PATTERN = re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+|bearer\s+)[A-Za-z0-9._~+/=-]+")
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
URL_CREDENTIAL_PATTERN = re.compile(r"(https?://[^/\s:@]+:)[^@\s/]+@")
PRIVATE_IP_PATTERN = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
INTERNAL_HOST_PATTERN = re.compile(r"\b[A-Za-z0-9._-]+\.(?:hobot|hogpu)\.cc\b", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_path(path: str | Path) -> str:
    text = str(path)
    home = str(HOME)
    return text.replace(home, "${HOME}") if text.startswith(home) else text


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.expandtabs(4).rstrip() for line in text.split("\n")).rstrip() + "\n"


def redact_text(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    text = text.replace(str(HOME), "${HOME}")

    def replace_private_key(match: re.Match[str]) -> str:
        counts["private_key"] += 1
        return "<REDACTED:PRIVATE_KEY>"

    def replace_json_secret(match: re.Match[str]) -> str:
        counts["credential"] += 1
        return f"{match.group(1)}{match.group(2)}<REDACTED:CREDENTIAL>{match.group(4)}"

    def replace_unquoted_secret(match: re.Match[str]) -> str:
        if match.group(2).strip("\"'").startswith("<REDACTED:"):
            return match.group(0)
        counts["credential"] += 1
        return f"{match.group(1)}<REDACTED:CREDENTIAL>"

    def replace_bearer(match: re.Match[str]) -> str:
        counts["bearer_token"] += 1
        return f"{match.group(1)}<REDACTED:BEARER_TOKEN>"

    def replace_openai_key(_: re.Match[str]) -> str:
        counts["api_key"] += 1
        return "<REDACTED:API_KEY>"

    def replace_url_credential(match: re.Match[str]) -> str:
        counts["url_credential"] += 1
        return f"{match.group(1)}<REDACTED:CREDENTIAL>@"

    def replace_private_ip(_: re.Match[str]) -> str:
        counts["private_ip"] += 1
        return "<PRIVATE_IP>"

    def replace_internal_host(_: re.Match[str]) -> str:
        counts["internal_host"] += 1
        return "<INTERNAL_HOST>"

    text = PRIVATE_KEY_PATTERN.sub(replace_private_key, text)
    text = QUOTED_JSON_SECRET.sub(replace_json_secret, text)
    text = UNQUOTED_SECRET.sub(replace_unquoted_secret, text)
    text = BEARER_PATTERN.sub(replace_bearer, text)
    text = OPENAI_KEY_PATTERN.sub(replace_openai_key, text)
    text = URL_CREDENTIAL_PATTERN.sub(replace_url_credential, text)
    text = PRIVATE_IP_PATTERN.sub(replace_private_ip, text)
    text = INTERNAL_HOST_PATTERN.sub(replace_internal_host, text)
    return text, counts


def extract_message_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        for key in ("text", "input_text", "output_text"):
            value = item.get(key)
            if isinstance(value, str):
                texts.append(value)
                break
    return "\n".join(texts)


def read_quant_sessions() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sessions: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for root in SESSION_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            meta: dict[str, Any] | None = None
            messages: list[dict[str, Any]] = []
            with path.open(encoding="utf-8", errors="replace") as handle:
                for ordinal, line in enumerate(handle):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "session_meta":
                        meta = dict(event.get("payload") or {})
                    if event.get("type") != "response_item":
                        continue
                    payload = event.get("payload") or {}
                    if payload.get("type") != "message":
                        continue
                    role = str(payload.get("role") or "")
                    if role not in {"user", "assistant"}:
                        continue
                    text = extract_message_text(payload)
                    if not text:
                        continue
                    messages.append(
                        {
                            "timestamp": str(event.get("timestamp") or ""),
                            "ordinal": int(event.get("ordinal", ordinal)),
                            "role": role,
                            "text": text,
                        }
                    )
            cwd = str((meta or {}).get("cwd") or "")
            if not cwd.startswith(str(QUANTIZATION_ROOT)):
                continue
            session_id = str((meta or {}).get("id") or (meta or {}).get("session_id") or path.stem)
            session = sessions.setdefault(
                session_id,
                {
                    "id": session_id,
                    "cwd": cwd,
                    "started": str((meta or {}).get("timestamp") or ""),
                    "messages": [],
                    "sources": [],
                },
            )
            session["messages"].extend(messages)
            session["sources"].append(path)
            source_rows.append(
                {
                    "session_id": session_id,
                    "source_path": normalized_path(path),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "message_count": len(messages),
                }
            )
    return sessions, source_rows


def substantive_title(messages: list[dict[str, Any]], session_id: str) -> str:
    ignored_prefixes = (
        "<environment_context>",
        "# AGENTS.md",
        "<INSTRUCTIONS>",
        "# Files mentioned",
    )
    for message in messages:
        if message["role"] != "user":
            continue
        text = message["text"].strip()
        if not text or text.startswith(ignored_prefixes):
            continue
        first_line = re.sub(r"\s+", " ", text.splitlines()[0]).strip("# ").strip()
        if first_line:
            return first_line[:100]
    return f"QuantX Codex session {session_id}"


def render_conversations(
    sessions: dict[str, dict[str, Any]],
    output: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    total_redactions: Counter[str] = Counter()
    for session_id, session in sorted(
        sessions.items(),
        key=lambda item: (item[1]["started"], item[0]),
    ):
        unique_messages: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for message in sorted(
            session["messages"],
            key=lambda item: (item["timestamp"], item["ordinal"]),
        ):
            key = (message["timestamp"], message["role"], message["text"])
            if key in seen:
                continue
            seen.add(key)
            unique_messages.append(message)
        raw_title = substantive_title(unique_messages, session_id)
        title, title_redactions = redact_text(raw_title)
        started = session["started"] or (unique_messages[0]["timestamp"] if unique_messages else "")
        date = started[:10] or "unknown-date"
        relative = Path("conversations") / date[:4] / date[5:7] / f"{date}_{session_id}.md"
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        session_redactions: Counter[str] = Counter(title_redactions)
        lines = [
            f"# {title}",
            "",
            f"- Session ID: `{session_id}`",
            f"- Started: `{started}`",
            f"- Working directory: `{normalized_path(session['cwd'])}`",
            f"- Source fragments: `{len(session['sources'])}`",
            f"- Messages: `{len(unique_messages)}`",
            "",
            "> 这是经过脱敏的只读副本；不包含系统/开发者提示词和原始工具事件。",
            "",
        ]
        for message in unique_messages:
            clean, counts = redact_text(message["text"])
            session_redactions.update(counts)
            role = "用户" if message["role"] == "user" else "Codex"
            lines.extend(
                [
                    f"## {role} — {message['timestamp']}",
                    "",
                    clean.rstrip(),
                    "",
                ]
            )
        target.write_text(normalize_text("\n".join(lines)), encoding="utf-8")
        total_redactions.update(session_redactions)
        rows.append(
            {
                "session_id": session_id,
                "started": started,
                "cwd": normalized_path(session["cwd"]),
                "title": title,
                "messages": len(unique_messages),
                "source_fragments": len(session["sources"]),
                "redactions": sum(session_redactions.values()),
                "archive_path": relative.as_posix(),
                "sha256": sha256_file(target),
            }
        )
    return rows, total_redactions


def iter_document_candidates() -> Iterable[tuple[str, Path, Path]]:
    excluded_parts = {
        ".git",
        "__pycache__",
        "checkpoints",
        "tensorboard",
        "runs",
        "node_modules",
        ".venv",
        "workdirs",
    }
    excluded_names = {
        "agents.md",
        "authors.txt",
        "claude.md",
        "entry_points.txt",
        "license.md",
        "license.txt",
        "proxies.txt",
        "top_level.txt",
        "vendor.txt",
    }
    for alias, root in DOCUMENT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            if any("venv" in part.lower() or part.lower() == "site-packages" for part in path.parts):
                continue
            if path.name.lower() in excluded_names or path.name.lower().startswith("requirements"):
                continue
            yield alias, root, path
    if WEAK_RESEARCH_ROOT.exists():
        allowed = (
            WEAK_RESEARCH_ROOT / "RESEARCH_LOG.md",
            WEAK_RESEARCH_ROOT / "FORMAL_MODEL_RESEARCH_PROTOCOL.md",
            WEAK_RESEARCH_ROOT / "LOOP_PREFERENCE_EXPERIMENT.md",
            WEAK_RESEARCH_ROOT / "rl_position_manager_v1" / "README.md",
        )
        for path in allowed:
            if path.is_file():
                yield "reward_research", WEAK_RESEARCH_ROOT, path


def archive_documents(
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    canonical_by_hash: dict[str, str] = {}
    source_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    redactions: Counter[str] = Counter()
    for alias, root, path in iter_document_candidates():
        source_hash = sha256_file(path)
        canonical = canonical_by_hash.get(source_hash)
        duplicate = canonical is not None
        if canonical is None:
            relative_source = path.relative_to(root)
            relative = Path("experiments") / alias / relative_source
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text(encoding="utf-8", errors="replace")
            clean, counts = redact_text(text)
            redactions.update(counts)
            target.write_text(normalize_text(clean), encoding="utf-8")
            canonical = relative.as_posix()
            canonical_by_hash[source_hash] = canonical
            canonical_rows.append(
                {
                    "archive_path": canonical,
                    "source_sha256": source_hash,
                    "archive_sha256": sha256_file(target),
                    "size": target.stat().st_size,
                }
            )
        source_rows.append(
            {
                "source_path": normalized_path(path),
                "source_sha256": source_hash,
                "canonical_path": canonical,
                "duplicate": duplicate,
            }
        )
    return canonical_rows, source_rows, redactions


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_indexes(
    output: Path,
    conversations: list[dict[str, Any]],
    canonical_docs: list[dict[str, Any]],
    document_sources: list[dict[str, Any]],
    session_sources: list[dict[str, Any]],
    redactions: Counter[str],
) -> None:
    write_csv(
        output / "conversations" / "index.csv",
        conversations,
        [
            "session_id",
            "started",
            "cwd",
            "title",
            "messages",
            "source_fragments",
            "redactions",
            "archive_path",
            "sha256",
        ],
    )
    write_csv(
        output / "provenance" / "conversation_sources.csv",
        session_sources,
        ["session_id", "source_path", "sha256", "size", "message_count"],
    )
    write_csv(
        output / "provenance" / "document_sources.csv",
        document_sources,
        ["source_path", "source_sha256", "canonical_path", "duplicate"],
    )
    manifest = {
        "kind": "quantx_research_archive_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source_policy": {
            "conversation_filter": f"session_meta.cwd 以 {normalized_path(QUANTIZATION_ROOT)} 开头",
            "conversation_roles": ["user", "assistant"],
            "excluded": [
                "系统提示词",
                "开发者提示词",
                "工具调用",
                "工具原始输出",
                "auth.json",
                "非 quantization 会话",
            ],
            "source_files_modified": False,
        },
        "counts": {
            "conversations": len(conversations),
            "conversation_messages": sum(int(row["messages"]) for row in conversations),
            "conversation_source_fragments": len(session_sources),
            "canonical_documents": len(canonical_docs),
            "document_source_paths": len(document_sources),
            "document_duplicates": sum(bool(row["duplicate"]) for row in document_sources),
        },
        "redactions": dict(sorted(redactions.items())),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "provenance" / "redactions.json").write_text(
        json.dumps(
            {
                "counts": dict(sorted(redactions.items())),
                "original_values_retained": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    index_lines = [
        "# QuantX 研究归档",
        "",
        f"- 合并会话数：**{len(conversations)}**",
        f"- 用户/Codex 消息数：**{sum(int(row['messages']) for row in conversations)}**",
        f"- 去重后实验文档数：**{len(canonical_docs)}**",
        f"- 实验文档来源路径数：**{len(document_sources)}**",
        f"- 脱敏替换数：**{sum(redactions.values())}**",
        "",
        "## 对话索引",
        "",
        "| 开始时间 | 会话 | 消息数 | 标题 |",
        "| --- | --- | ---: | --- |",
    ]
    for row in conversations:
        safe_title = str(row["title"]).replace("|", "\\|")
        index_lines.append(
            f"| {row['started']} | [{row['session_id']}]({row['archive_path']}) "
            f"| {row['messages']} | {safe_title} |"
        )
    index_lines.extend(
        [
            "",
            "## 实验文档",
            "",
            "每个原始路径及其对应的归档副本见 `provenance/document_sources.csv`。",
        ]
    )
    (output / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        """# QuantX 研究归档

本目录是 QuantX 研究对话和实验文档的脱敏副本。

- 原始 Codex 会话文件始终只读，没有被移动、编辑或删除。
- 对话 Markdown 只保留用户与 Codex 的消息。
- 系统/开发者提示词、工具事件、凭据、私网地址和内部主机名已排除或脱敏。
- 实验文档按源文件 SHA-256 去重；所有原始路径仍记录在 `provenance/document_sources.csv`。
- manifest 是 `created_at` 时刻的快照；活跃 Codex 会话在归档后仍可能继续增长。
- 归档已按 Git 上传场景清理，但发布前仍建议检查 `provenance/redactions.json`。
""",
        encoding="utf-8",
    )


def scan_output(output: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    forbidden = (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+(?!<REDACTED)"),
    )
    assignment = re.compile(
        rf"(?im)\b(?:{SECRET_NAME_PATTERN})\b\s*[:=]\s*(?P<value>[^\n]+)"
    )
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden:
            if pattern.search(text):
                findings.append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "pattern": pattern.pattern,
                    }
                )
        for match in assignment.finditer(text):
            value = match.group("value").strip().strip("\"'")
            if value.startswith("<REDACTED:"):
                continue
            findings.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "pattern": "unredacted_secret_assignment",
                }
            )
    return findings


def build_archive(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Archive directory already exists: {output}")
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sessions, session_sources = read_quant_sessions()
        conversations, conversation_redactions = render_conversations(sessions, staging)
        canonical_docs, document_sources, document_redactions = archive_documents(staging)
        total_redactions = conversation_redactions + document_redactions
        write_indexes(
            staging,
            conversations,
            canonical_docs,
            document_sources,
            session_sources,
            total_redactions,
        )
        findings = scan_output(staging)
        if findings:
            raise RuntimeError(
                "Sensitive-data scan failed: "
                + json.dumps(findings[:20], ensure_ascii=False)
            )
        os.replace(staging, output)
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    manifest = build_archive(output)
    print(json.dumps({"ok": True, "output": str(output), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
