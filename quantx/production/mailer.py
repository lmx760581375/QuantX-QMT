"""Email notification support for QuantX daily production."""

from __future__ import annotations

import hashlib
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List

from .artifacts import read_json, write_json


def send_daily_email(
    subject: str,
    text: str,
    html: str,
    recipients: List[str],
    status_path: str | Path,
    dry_run: bool = False,
    force_send: bool = False,
    username: str = "",
    password: str = "",
    smtp_host: str = "",
    smtp_port: int | None = None,
) -> Dict[str, Any]:
    """Send or preview a daily email with idempotency tracking."""
    resolved_recipients = _recipients(recipients)
    content_hash = _content_hash(subject, text, html, resolved_recipients)
    status_path = Path(status_path)
    previous = read_json(status_path, default={}) or {}
    if previous.get("sent") and previous.get("content_hash") == content_hash and not force_send:
        result = {
            "ok": True,
            "sent": False,
            "skipped": True,
            "reason": "same content already sent",
            "content_hash": content_hash,
            "recipients": resolved_recipients,
            "subject": subject,
        }
        write_json(status_path, {**previous, **result, "checked_at": _now()})
        return result

    if dry_run:
        result = {
            "ok": True,
            "sent": False,
            "dry_run": True,
            "content_hash": content_hash,
            "recipients": resolved_recipients,
            "subject": subject,
            "checked_at": _now(),
        }
        write_json(status_path, result)
        return result

    username = username or os.environ.get("QUANTX_MAIL_USERNAME")
    password = password or os.environ.get("QUANTX_MAIL_PASSWORD")
    smtp_host = smtp_host or os.environ.get("QUANTX_MAIL_SMTP_HOST", "smtp.qq.com")
    smtp_port = int(smtp_port or os.environ.get("QUANTX_MAIL_SMTP_PORT", "465"))
    if not username or not password:
        result = {
            "ok": False,
            "sent": False,
            "error_type": "auth_missing",
            "message": "QUANTX_MAIL_USERNAME and QUANTX_MAIL_PASSWORD are required",
            "content_hash": content_hash,
            "recipients": resolved_recipients,
            "subject": subject,
        }
        write_json(status_path, result)
        return result
    if not resolved_recipients:
        result = {
            "ok": False,
            "sent": False,
            "error_type": "recipient_missing",
            "message": "No email recipients configured",
            "content_hash": content_hash,
            "subject": subject,
        }
        write_json(status_path, result)
        return result

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = ", ".join(resolved_recipients)
    message.set_content(text)
    message.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
        result = {
            "ok": True,
            "sent": True,
            "sent_at": _now(),
            "recipients": resolved_recipients,
            "subject": subject,
            "content_hash": content_hash,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
        }
    except smtplib.SMTPAuthenticationError as exc:
        result = _mail_error("auth_failed", exc, content_hash, resolved_recipients, subject, smtp_host)
    except TimeoutError as exc:
        result = _mail_error("network_timeout", exc, content_hash, resolved_recipients, subject, smtp_host)
    except smtplib.SMTPRecipientsRefused as exc:
        result = _mail_error("recipient_error", exc, content_hash, resolved_recipients, subject, smtp_host)
    except smtplib.SMTPException as exc:
        result = _mail_error("smtp_rejected", exc, content_hash, resolved_recipients, subject, smtp_host)
    except Exception as exc:
        result = _mail_error(type(exc).__name__, exc, content_hash, resolved_recipients, subject, smtp_host)
    write_json(status_path, result)
    return result


def render_subject(template: str, trade_date: str, profile: str) -> str:
    return template.replace("{{ trade_date }}", trade_date).replace("{{ profile }}", profile)


def _recipients(configured: List[str]) -> List[str]:
    env = os.environ.get("QUANTX_MAIL_TO", "")
    rows = [item.strip() for item in configured if item.strip()]
    rows.extend(item.strip() for item in env.split(",") if item.strip())
    return list(dict.fromkeys(rows))


def _content_hash(subject: str, text: str, html: str, recipients: List[str]) -> str:
    payload = "\n".join([subject, ",".join(recipients), text, html])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mail_error(
    error_type: str,
    exc: Exception,
    content_hash: str,
    recipients: List[str],
    subject: str,
    smtp_host: str,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "sent": False,
        "error_type": error_type,
        "message": str(exc),
        "content_hash": content_hash,
        "recipients": recipients,
        "subject": subject,
        "smtp_host": smtp_host,
        "failed_at": _now(),
    }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
