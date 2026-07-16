"""Configuration helpers for the QuantX daily production pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class ProductionPaths:
    root: Path
    daily_runs_dir: Path
    state_dir: Path


@dataclass(frozen=True)
class MailConfig:
    enabled: bool
    channel: str
    recipients: List[str]
    subject_template: str
    config_path: str = ""
    smtp_host: str = ""
    smtp_port: int | None = None
    username: str = ""
    password: str = ""
    attach_html: bool = True
    attach_json: bool = False
    send_empty_signal_report: bool = True


@dataclass(frozen=True)
class PortfolioStateConfig:
    mode: str = "simulated"
    state_dir: str = "daily_state/default"
    init_cash: float = 1_000_000
    carry_positions: bool = True
    apply_simulated_orders: bool = True
    execution_price_policy: str = "config_deal_price"


@dataclass(frozen=True)
class DataUpdateConfig:
    provider_uri: str = "data/qlib_data_fixed"
    update_mode: str = "local_verify_only"
    verify_after_update: bool = True
    update_command: List[str] = field(default_factory=list)
    timeout_seconds: int = 21600


@dataclass(frozen=True)
class PredictionJobConfig:
    name: str
    command: List[str]
    timeout_seconds: int = 21600
    required: bool = True


@dataclass(frozen=True)
class ProductionProfile:
    name: str
    timezone: str
    raw: Dict[str, Any]
    paths: ProductionPaths
    data: DataUpdateConfig
    prediction_jobs: List[PredictionJobConfig]
    strategies: List[str]
    mail: MailConfig
    portfolio: PortfolioStateConfig


def load_profile(path: str | Path, project_root: str | Path | None = None) -> ProductionProfile:
    """Load a production profile YAML."""
    profile_path = Path(path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    with profile_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Production profile must be a YAML mapping: {profile_path}")

    name = str(raw.get("name") or profile_path.stem)
    timezone = str(raw.get("timezone") or "Asia/Shanghai")
    data_raw = raw.get("data") or {}
    strategies_raw = raw.get("strategies") or {}
    predictions_raw = raw.get("predictions") or {}
    notification_raw = raw.get("notification") or {}
    portfolio_raw = raw.get("portfolio_state") or {}
    mail_file_raw = _load_mail_file(root, notification_raw.get("config_path") or notification_raw.get("mail_config"))

    include = strategies_raw.get("include") or []
    exclude = set(str(item) for item in (strategies_raw.get("exclude") or []))
    strategies = [str(item) for item in include if str(item) not in exclude]
    if not strategies:
        raise ValueError("Production profile requires strategies.include")

    daily_runs_dir = _resolve(root, raw.get("daily_runs_dir") or "daily_runs")
    state_dir = _resolve(root, portfolio_raw.get("state_dir") or "daily_state/default")

    return ProductionProfile(
        name=name,
        timezone=timezone,
        raw=raw,
        paths=ProductionPaths(root=root, daily_runs_dir=daily_runs_dir, state_dir=state_dir),
        data=DataUpdateConfig(
            provider_uri=str(data_raw.get("provider_uri") or "data/qlib_data_fixed"),
            update_mode=str(data_raw.get("update_mode") or "local_verify_only"),
            verify_after_update=bool(data_raw.get("verify_after_update", True)),
            update_command=[str(part) for part in (data_raw.get("update_command") or [])],
            timeout_seconds=int(data_raw.get("timeout_seconds") or 21600),
        ),
        prediction_jobs=_prediction_jobs(predictions_raw),
        strategies=strategies,
        mail=MailConfig(
            enabled=bool(notification_raw.get("enabled", True)),
            channel=str(notification_raw.get("channel") or mail_file_raw.get("channel") or "qq_email"),
            recipients=_mail_recipients(notification_raw.get("recipients") or [], mail_file_raw),
            subject_template=str(notification_raw.get("subject_template") or "QuantX 每日策略信号 {{ trade_date }}"),
            config_path=str(notification_raw.get("config_path") or notification_raw.get("mail_config") or ""),
            smtp_host=str(mail_file_raw.get("smtp_host") or ""),
            smtp_port=_optional_int(mail_file_raw.get("smtp_port")),
            username=str(_mail_secret(mail_file_raw, "username", "username_env") or ""),
            password=str(_mail_secret(mail_file_raw, "password", "password_env") or ""),
            attach_html=bool(notification_raw.get("attach_html", True)),
            attach_json=bool(notification_raw.get("attach_json", False)),
            send_empty_signal_report=bool(notification_raw.get("send_empty_signal_report", True)),
        ),
        portfolio=PortfolioStateConfig(
            mode=str(portfolio_raw.get("mode") or "simulated"),
            state_dir=str(portfolio_raw.get("state_dir") or "daily_state/default"),
            init_cash=float(portfolio_raw.get("init_cash", 1_000_000)),
            carry_positions=bool(portfolio_raw.get("carry_positions", True)),
            apply_simulated_orders=bool(portfolio_raw.get("apply_simulated_orders", True)),
            execution_price_policy=str(portfolio_raw.get("execution_price_policy") or "config_deal_price"),
        ),
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_mail_file(root: Path, value: str | Path | None) -> Dict[str, Any]:
    if not value:
        return {}
    path = _resolve(root, value)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _mail_secret(raw: Dict[str, Any], direct_key: str, env_key: str) -> Any:
    if direct_key in raw:
        return raw.get(direct_key)
    value = raw.get(env_key)
    if isinstance(value, str):
        import os

        return os.environ.get(value, value)
    return value


def _mail_recipients(configured: List[Any], raw: Dict[str, Any]) -> List[str]:
    rows = [str(item) for item in configured if str(item).strip()]
    file_value = raw.get("recipients")
    if file_value is None:
        file_value = _mail_secret(raw, "recipients", "recipients_env")
    if isinstance(file_value, str):
        rows.extend(item.strip() for item in file_value.split(",") if item.strip())
    elif isinstance(file_value, list):
        rows.extend(str(item).strip() for item in file_value if str(item).strip())
    return list(dict.fromkeys(rows))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _prediction_jobs(raw: Any) -> List[PredictionJobConfig]:
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return []
    jobs = raw.get("jobs") or []
    result: List[PredictionJobConfig] = []
    for index, item in enumerate(jobs, start=1):
        if not isinstance(item, dict):
            continue
        command = [str(part) for part in (item.get("command") or [])]
        if not command:
            continue
        result.append(PredictionJobConfig(
            name=str(item.get("name") or f"prediction_job_{index}"),
            command=command,
            timeout_seconds=int(item.get("timeout_seconds") or raw.get("timeout_seconds") or 21600),
            required=bool(item.get("required", True)),
        ))
    return result
