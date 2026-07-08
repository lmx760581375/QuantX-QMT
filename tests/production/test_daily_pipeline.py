"""Daily production pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from quantx.core.data.meta import MetaStore
from quantx.production import DailyPipeline
from quantx.production.config import load_profile


def test_daily_pipeline_generates_report_and_mail_preview(tmp_path):
    provider = _provider(tmp_path)
    config_path = tmp_path / "configs" / "strategies" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump({
            "name": "demo_strategy",
            "title": "测试战法",
            "description": "测试策略：上涨买入，下跌卖出。",
            "data": {
                "provider_uri": str(provider),
                "universe": ["SZ000001"],
                "start": "2021-01-04",
                "end": "2021-01-04",
            },
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 0"},
            "selector": {"where": "buy_signal"},
            "execution": {"deal_price": "close"},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    profile_path = tmp_path / "configs" / "production" / "daily.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump({
            "name": "unit",
            "daily_runs_dir": "daily_runs",
            "data": {"provider_uri": str(provider), "update_mode": "local_verify_only"},
            "strategies": {"include": [str(config_path.relative_to(tmp_path))]},
            "notification": {
                "enabled": True,
                "recipients": ["unit@qq.com"],
                "subject_template": "QuantX 每日策略信号 {{ trade_date }}",
            },
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    meta_store = MetaStore(tmp_path / "data" / "meta" / "quantx_meta.sqlite")
    meta_store.upsert_security_master(pd.DataFrame([{"symbol": "SZ000001", "name": "平安银行"}]))
    meta_store.upsert_industry_membership(pd.DataFrame([{"symbol": "SZ000001", "industry_name": "银行"}]))

    profile = load_profile(profile_path, project_root=tmp_path)
    pipeline = DailyPipeline(profile, meta_store=meta_store, strategy_runner=_fake_runner)

    result = pipeline.run(stage="all", dry_run=True)

    run_dir = Path(result.run_dir)
    assert result.ok is True
    assert result.trade_date == "2021-01-05"
    assert (run_dir / "strategy_signals.json").exists()
    assert (run_dir / "suggested_orders.json").exists()
    assert (run_dir / "next_session_guides.json").exists()
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "平安银行" in markdown
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "测试战法" in markdown
    assert "测试战法" in html
    assert "### demo_strategy" not in markdown
    assert "Run:" not in markdown
    assert "Run:" not in html
    assert "顶层 runs 目录" not in html
    assert "Daily run 目录" not in markdown
    assert "Daily run 目录" not in html
    assert "<svg" in html
    assert "收益曲线" in html
    assert "最近五笔历史交易" in html
    mail_status = json.loads((run_dir / "mail_status.json").read_text(encoding="utf-8"))
    assert mail_status["dry_run"] is True
    assert mail_status["sent"] is False

    signals = json.loads((run_dir / "strategy_signals.json").read_text(encoding="utf-8"))
    guides = json.loads((run_dir / "next_session_guides.json").read_text(encoding="utf-8"))
    assert signals[0]["strategy_title"] == "测试战法"
    assert signals[0]["buys"][0]["name"] == "平安银行"
    assert signals[0]["positions"][0]["industry_name"] == "银行"
    assert signals[0]["recent_trades"][0]["symbol"] == "SZ000001"
    assert signals[0]["annual_returns"][0]["year"] == 2021
    assert signals[0]["next_session_guide"]["new_buy_candidates"][0]["symbol"] == "SZ000002"
    assert guides["demo_strategy"]["new_buy_candidates"][0]["symbol"] == "SZ000002"
    assert "年度收益" in markdown
    assert "下一交易日操作指南" in html
    assert Path(signals[0]["run_dir"]).parent == tmp_path / "runs"
    assert (tmp_path / "runs" / signals[0]["run_id"] / "summary.json").exists()


def test_daily_pipeline_marks_result_failed_when_data_update_command_fails(tmp_path):
    provider = _provider(tmp_path)
    config_path = tmp_path / "configs" / "strategies" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump({
            "name": "demo_strategy",
            "data": {
                "provider_uri": str(provider),
                "universe": ["SZ000001"],
                "start": "2021-01-04",
                "end": "2021-01-04",
            },
            "fields": {"close": "$close"},
            "signals": {"buy_signal": "close > 0"},
            "selector": {"where": "buy_signal"},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    profile_path = tmp_path / "configs" / "production" / "daily.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump({
            "name": "unit",
            "daily_runs_dir": "daily_runs",
            "data": {
                "provider_uri": str(provider),
                "update_mode": "incremental",
                "timeout_seconds": 456,
                "update_command": [
                    "python",
                    "-c",
                    "import sys; print('data update failed'); sys.exit(3)",
                ],
            },
            "strategies": {"include": [str(config_path.relative_to(tmp_path))]},
            "notification": {"enabled": False},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile = load_profile(profile_path, project_root=tmp_path)
    pipeline = DailyPipeline(profile, strategy_runner=_fake_runner)

    result = pipeline.run(stage="data", dry_run=True)

    assert result.ok is False
    assert result.data_update["ok"] is False
    assert result.data_update["update_command"]["returncode"] == 3
    assert result.data_update["update_command"]["timeout_seconds"] == 456


def test_daily_pipeline_can_skip_data_update_command(tmp_path):
    provider = _provider(tmp_path)
    config_path = tmp_path / "configs" / "strategies" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("name: demo\n", encoding="utf-8")
    profile_path = tmp_path / "configs" / "production" / "daily.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump({
            "name": "unit",
            "daily_runs_dir": "daily_runs",
            "data": {
                "provider_uri": str(provider),
                "update_mode": "incremental",
                "update_command": ["python", "-c", "raise SystemExit(3)"],
            },
            "strategies": {"include": [str(config_path.relative_to(tmp_path))]},
            "notification": {"enabled": False},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile = load_profile(profile_path, project_root=tmp_path)
    pipeline = DailyPipeline(profile)

    result = pipeline.run(stage="data", skip_data_update=True)

    assert result.ok is True
    assert result.data_update["update_command"]["skipped"] is True
    assert result.data_update["update_command"]["message"] == "data update skipped by request"


def test_daily_profile_loads_mail_yaml(tmp_path):
    provider = _provider(tmp_path)
    config_path = tmp_path / "configs" / "strategies" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("name: demo\n", encoding="utf-8")
    mail_path = tmp_path / "configs" / "production" / "mail.yaml"
    mail_path.parent.mkdir(parents=True)
    mail_path.write_text(
        yaml.safe_dump({
            "channel": "qq_email",
            "smtp_host": "smtp.qq.com",
            "smtp_port": 465,
            "username_env": "unit@qq.com",
            "password_env": "auth-code",
            "recipients_env": "a@qq.com,b@qq.com",
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    profile_path = tmp_path / "configs" / "production" / "daily.yaml"
    profile_path.write_text(
        yaml.safe_dump({
            "name": "unit",
            "data": {"provider_uri": str(provider)},
            "strategies": {"include": [str(config_path.relative_to(tmp_path))]},
            "notification": {"config_path": str(mail_path.relative_to(tmp_path))},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile = load_profile(profile_path, project_root=tmp_path)

    assert profile.mail.smtp_host == "smtp.qq.com"
    assert profile.mail.smtp_port == 465
    assert profile.mail.username == "unit@qq.com"
    assert profile.mail.password == "auth-code"
    assert profile.mail.recipients == ["a@qq.com", "b@qq.com"]


def test_daily_profile_loads_data_update_timeout(tmp_path):
    provider = _provider(tmp_path)
    config_path = tmp_path / "configs" / "strategies" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("name: demo\n", encoding="utf-8")
    profile_path = tmp_path / "configs" / "production" / "daily.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump({
            "name": "unit",
            "data": {"provider_uri": str(provider), "timeout_seconds": 999},
            "strategies": {"include": [str(config_path.relative_to(tmp_path))]},
            "notification": {"enabled": False},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile = load_profile(profile_path, project_root=tmp_path)

    assert profile.data.timeout_seconds == 999


def _provider(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir(parents=True)
    (provider / "features" / "SZ000001").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2021-01-04\n2021-01-05\n", encoding="utf-8")
    (provider / "instruments" / "all.txt").write_text("SZ000001 2000-01-01 2099-12-31\n", encoding="utf-8")
    (provider / "features" / "SZ000001" / "close.bin").write_bytes(b"unit")
    return provider


def _fake_runner(config_path, output_dir, run_id, symbol_limit):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert config["data"]["end"] == "2021-01-05"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True)
    _write(run_dir / "summary.json", {
        "run_id": run_id,
        "name": config["name"],
        "description": config["description"],
        "start_date": "2021-01-04",
        "end_date": "2021-01-05",
        "total_return": 0.01,
        "final_value": 1010000,
    })
    _write(run_dir / "metrics.json", {"total_return": 0.01, "trade_count": 1})
    _write(run_dir / "daily_nav.json", [
        {"date": "2021-01-04", "total_value": 1000000, "drawdown": 0, "position_count": 0},
        {"date": "2021-01-05", "total_value": 1010000, "drawdown": 0, "position_count": 1},
    ])
    _write(run_dir / "trades.json", [
        {
            "date": "2021-01-05",
            "symbol": "SZ000001",
            "action": "BUY",
            "price": 10.0,
            "quantity": 100,
            "trade_value": 1000.0,
            "total_cost": 5.0,
            "reject_reason": "",
            "reason": "config_buy",
        }
    ])
    _write(run_dir / "positions.json", [
        {
            "date": "2021-01-05",
            "symbol": "SZ000001",
            "quantity": 100,
            "avg_cost": 10.0,
            "market_value": 1050.0,
            "weight": 0.1,
            "holding_days": 1,
        }
    ])
    _write(run_dir / "closed_positions.json", [])
    _write(run_dir / "daily_selection_candidates.json", [
        {
            "date": "2021-01-05",
            "raw_candidate_count": 1,
            "selected_count": 1,
            "selected_candidates": [{"symbol": "SZ000002", "score": 1.0, "where": True}],
        }
    ])
    _write(run_dir / "explain.json", {"config": config})
    (run_dir / "logs.txt").write_text("", encoding="utf-8")
    return {"run_dir": str(run_dir), "run_id": run_id}


def _write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
