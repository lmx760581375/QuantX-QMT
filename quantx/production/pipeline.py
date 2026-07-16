"""Daily production pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from quantx.core.data.meta import MetaStore

from .artifacts import read_json, write_json, write_text
from .config import ProductionProfile
from .data_update import latest_trade_date, run_update_command, verify_provider
from .mailer import render_subject, send_daily_email
from .report_renderer import build_report_payload, render_html, render_markdown
from .strategy_runtime import DailyStrategyRuntime, StrategyRunner


VALID_STAGES = {"all", "data", "predictions", "signals", "report", "mail"}


@dataclass
class DailyPipelineResult:
    ok: bool
    profile: str
    trade_date: str
    run_dir: str
    stage: str
    data_update: Dict[str, Any] = field(default_factory=dict)
    prediction_jobs: List[Dict[str, Any]] = field(default_factory=list)
    strategy_count: int = 0
    mail_status: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "profile": self.profile,
            "trade_date": self.trade_date,
            "run_dir": self.run_dir,
            "stage": self.stage,
            "data_update": self.data_update,
            "prediction_jobs": self.prediction_jobs,
            "strategy_count": self.strategy_count,
            "mail_status": self.mail_status,
            "errors": self.errors,
        }


class DailyPipeline:
    """Run data verification, strategy daily updates, report rendering, and email."""

    def __init__(
        self,
        profile: ProductionProfile,
        meta_store: MetaStore | None = None,
        strategy_runner: StrategyRunner | None = None,
    ):
        self.profile = profile
        self.meta_store = meta_store or MetaStore(profile.paths.root / "data" / "meta" / "quantx_meta.sqlite")
        self.strategy_runner = strategy_runner
        self.logs: List[str] = []

    def run(
        self,
        stage: str = "all",
        trade_date: str | None = None,
        dry_run: bool = False,
        force_send: bool = False,
        force_data_update: bool = False,
        skip_data_update: bool = False,
        strategy_filter: str | None = None,
        symbol_limit: int | None = None,
    ) -> DailyPipelineResult:
        if stage not in VALID_STAGES:
            raise ValueError(f"Unsupported stage: {stage}")
        provider_uri = self._provider_uri()
        trade_date = trade_date or latest_trade_date(provider_uri)
        run_dir = self.profile.paths.daily_runs_dir / trade_date.replace("-", "") / self.profile.name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"daily pipeline started: stage={stage} trade_date={trade_date} dry_run={dry_run}")

        errors: List[Dict[str, Any]] = []
        data_update: Dict[str, Any] = read_json(run_dir / "data_update.json", default={}) or {}
        prediction_jobs: List[Dict[str, Any]] = read_json(run_dir / "prediction_jobs.json", default=[]) or []
        strategies: List[Dict[str, Any]] = read_json(run_dir / "strategy_signals.json", default=[]) or []
        mail_status: Dict[str, Any] = read_json(run_dir / "mail_status.json", default={}) or {}

        try:
            if stage in {"all", "data"}:
                data_update = self._run_data_stage(provider_uri, run_dir, force_data_update, dry_run, skip_data_update)
            elif not data_update:
                data_update = verify_provider(provider_uri)
                write_json(run_dir / "data_update.json", data_update)

            if stage in {"all", "predictions"}:
                prediction_jobs = self._run_prediction_stage(run_dir, trade_date, dry_run)

            if stage in {"all", "signals"}:
                strategies = self._run_signal_stage(run_dir, trade_date, strategy_filter, symbol_limit)

            if stage in {"all", "report"}:
                self._run_report_stage(run_dir, trade_date, data_update, strategies)

            if stage in {"all", "mail"}:
                if not (run_dir / "report.md").exists() or not (run_dir / "report.html").exists():
                    self._run_report_stage(run_dir, trade_date, data_update, strategies)
                mail_status = self._run_mail_stage(run_dir, trade_date, dry_run=dry_run, force_send=force_send)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            self._log(f"ERROR {error['error_type']}: {error['message']}")

        result = DailyPipelineResult(
            ok=not errors and data_update.get("ok", True) and (not mail_status or mail_status.get("ok", True)),
            profile=self.profile.name,
            trade_date=trade_date,
            run_dir=str(run_dir),
            stage=stage,
            data_update=data_update,
            prediction_jobs=prediction_jobs,
            strategy_count=len(strategies),
            mail_status=mail_status,
            errors=errors,
        )
        status = {
            **result.to_dict(),
            "started_or_updated_at": datetime.now().isoformat(timespec="seconds"),
            "logs": self.logs,
        }
        write_json(run_dir / "pipeline_status.json", status)
        write_text(run_dir / "logs.txt", "\n".join(self.logs) + "\n")
        return result

    def _run_prediction_stage(self, run_dir: Path, trade_date: str, dry_run: bool) -> List[Dict[str, Any]]:
        self._log("predictions stage started")
        results: List[Dict[str, Any]] = []
        if not self.profile.prediction_jobs:
            results = [{"ok": True, "skipped": True, "message": "no prediction jobs configured"}]
            write_json(run_dir / "prediction_jobs.json", results)
            self._log("predictions stage skipped: no jobs")
            return results
        for job in self.profile.prediction_jobs:
            command = _prepare_prediction_command(job.command, trade_date, run_dir, dry_run)
            self._log(f"prediction job started: {job.name}")
            result = run_update_command(
                command,
                cwd=self.profile.paths.root,
                timeout_seconds=job.timeout_seconds,
            )
            item = {
                "name": job.name,
                "required": job.required,
                **result,
            }
            results.append(item)
            self._log(f"prediction job finished: {job.name} ok={item.get('ok')}")
            if job.required and not item.get("ok"):
                write_json(run_dir / "prediction_jobs.json", results)
                raise RuntimeError(f"prediction job failed: {job.name}")
        write_json(run_dir / "prediction_jobs.json", results)
        self._log(f"predictions stage finished: jobs={len(results)}")
        return results

    def _run_data_stage(
        self,
        provider_uri: Path,
        run_dir: Path,
        force_data_update: bool,
        dry_run: bool,
        skip_data_update: bool = False,
    ) -> Dict[str, Any]:
        self._log("data stage started")
        command_result = {"ok": True, "skipped": True, "message": "local verify only"}
        if skip_data_update:
            command_result = {"ok": True, "skipped": True, "message": "data update skipped by request"}
        elif self.profile.data.update_command and (force_data_update or self.profile.data.update_mode != "local_verify_only"):
            command = list(self.profile.data.update_command)
            if dry_run and "--dry-run" not in command:
                command.append("--dry-run")
            command_result = run_update_command(
                command,
                cwd=self.profile.paths.root,
                timeout_seconds=self.profile.data.timeout_seconds,
            )
        verify = verify_provider(provider_uri)
        result = {**verify, "update_command": command_result}
        result["ok"] = bool(command_result.get("ok") and verify.get("ok"))
        write_json(run_dir / "data_update.json", result)
        self._log(f"data stage finished: ok={result['ok']} calendar_end={result.get('calendar_end')}")
        return result

    def _run_signal_stage(
        self,
        run_dir: Path,
        trade_date: str,
        strategy_filter: str | None,
        symbol_limit: int | None,
    ) -> List[Dict[str, Any]]:
        self._log("signals stage started")
        runtime = DailyStrategyRuntime(
            profile=self.profile,
            trade_date=trade_date,
            output_dir=run_dir,
            meta_store=self.meta_store,
            symbol_limit=symbol_limit,
            runner=self.strategy_runner,
        )
        strategies = runtime.run_all(strategy_filter=strategy_filter)
        write_json(run_dir / "strategy_signals.json", strategies)
        write_json(run_dir / "strategy_positions.json", _positions_by_strategy(strategies))
        write_json(run_dir / "suggested_orders.json", _orders_by_strategy(strategies))
        write_json(run_dir / "next_session_guides.json", _next_guides_by_strategy(strategies))
        write_json(run_dir / "rejected_orders.json", _rejections_by_strategy(strategies))
        self._log(f"signals stage finished: strategies={len(strategies)}")
        return strategies

    def _run_report_stage(
        self,
        run_dir: Path,
        trade_date: str,
        data_update: Dict[str, Any],
        strategies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._log("report stage started")
        payload = build_report_payload(
            profile_name=self.profile.name,
            trade_date=trade_date,
            run_dir=str(run_dir),
            data_update=data_update,
            strategies=strategies,
        )
        markdown = render_markdown(payload)
        html = render_html(payload)
        write_json(run_dir / "daily_report.json", payload)
        write_text(run_dir / "report.md", markdown)
        write_text(run_dir / "report.html", html)
        self._log("report stage finished")
        return payload

    def _run_mail_stage(self, run_dir: Path, trade_date: str, dry_run: bool, force_send: bool) -> Dict[str, Any]:
        self._log("mail stage started")
        if not self.profile.mail.enabled:
            result = {"ok": True, "sent": False, "skipped": True, "reason": "notification disabled"}
            write_json(run_dir / "mail_status.json", result)
            self._log("mail stage skipped: disabled")
            return result
        subject = render_subject(self.profile.mail.subject_template, trade_date, self.profile.name)
        text = (run_dir / "report.md").read_text(encoding="utf-8")
        html = (run_dir / "report.html").read_text(encoding="utf-8")
        result = send_daily_email(
            subject=subject,
            text=text,
            html=html,
            recipients=self.profile.mail.recipients,
            status_path=run_dir / "mail_status.json",
            dry_run=dry_run,
            force_send=force_send,
            username=self.profile.mail.username,
            password=self.profile.mail.password,
            smtp_host=self.profile.mail.smtp_host,
            smtp_port=self.profile.mail.smtp_port,
        )
        self._log(f"mail stage finished: ok={result.get('ok')} sent={result.get('sent')}")
        return result

    def _provider_uri(self) -> Path:
        provider = Path(self.profile.data.provider_uri)
        return provider if provider.is_absolute() else self.profile.paths.root / provider

    def _log(self, message: str) -> None:
        self.logs.append(f"{datetime.now().isoformat(timespec='seconds')} {message}")


def _orders_by_strategy(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        item.get("strategy_name", ""): {
            "buys": item.get("buys") or [],
            "sells": item.get("sells") or [],
        }
        for item in strategies
        if item.get("ok")
    }


def _positions_by_strategy(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        item.get("strategy_name", ""): item.get("positions") or []
        for item in strategies
        if item.get("ok")
    }


def _next_guides_by_strategy(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        item.get("strategy_name", ""): item.get("next_session_guide") or {}
        for item in strategies
        if item.get("ok")
    }


def _rejections_by_strategy(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        item.get("strategy_name", ""): item.get("rejected_orders") or []
        for item in strategies
        if item.get("ok")
    }


def _prepare_prediction_command(command: List[str], trade_date: str, run_dir: Path, dry_run: bool) -> List[str]:
    mapping = {
        "trade_date": trade_date,
        "trade_date_compact": trade_date.replace("-", ""),
        "run_dir": str(run_dir),
    }
    resolved = [str(part).format(**mapping) for part in command]
    if dry_run and "--dry-run" not in resolved:
        resolved.append("--dry-run")
    return resolved
