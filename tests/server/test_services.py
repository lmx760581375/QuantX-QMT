"""Server service tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quantx.core.data.meta import MetaStore
from quantx.server import services


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_table(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_config_service_delete_config(tmp_path):
    service = services.ConfigService(root=tmp_path)
    service.write_config("demo.yaml", "name: demo\n")

    result = service.delete_config("demo.yaml")

    assert result == {"ok": True, "path": "demo.yaml"}
    assert not (tmp_path / "demo.yaml").exists()


def test_pattern_analysis_service_reads_analysis_and_similarity(tmp_path):
    root = tmp_path / "pattern_analysis"
    analysis = root / "unit_analysis"
    figures = analysis / "figures" / "clusters"
    figures.mkdir(parents=True)
    _write_json(analysis / "pattern_summary.json", {
        "sample_count": 2,
        "avg_return": 0.01,
        "outcome_counts": {"success": 1, "failure": 1},
    })
    _write_json(analysis / "config.json", {"runs": ["runs/unit_run"]})
    _write_table(analysis / "window_features.parquet", [
        {"sample_id": "s1", "symbol": "SZ000001", "entry_date": "2021-01-04", "exit_date": "2021-01-05", "return": 0.1, "max_favorable_excursion": 0.12, "max_adverse_excursion": -0.01, "pre_5_return": 0.02, "entry_close_position": 0.9},
        {"sample_id": "s2", "symbol": "SZ000002", "entry_date": "2021-01-04", "exit_date": "2021-01-05", "return": -0.1, "max_favorable_excursion": 0.01, "max_adverse_excursion": -0.12, "pre_5_return": 0.01, "entry_close_position": 0.85},
    ])
    _write_table(analysis / "labels.parquet", [
        {
            "sample_id": "s1",
            "outcome_label": "success",
            "behavior_label": "quick_confirm_success",
            "binary_success": 1,
            "early_confirmed": True,
            "early_failed": False,
            "opportunity_label": "opportunity",
            "had_opportunity": True,
            "fade_label": "held_or_no_peak_fade",
            "faded_after_peak": False,
            "efficient_capture": True,
            "first_3d_return": 0.05,
            "first_5d_return": 0.08,
            "drawdown_after_peak": -0.02,
        },
        {
            "sample_id": "s2",
            "outcome_label": "failure",
            "behavior_label": "buy_and_drop_failure",
            "binary_success": 0,
            "early_confirmed": False,
            "early_failed": True,
            "opportunity_label": "no_opportunity",
            "had_opportunity": False,
            "fade_label": "fade_after_peak",
            "faded_after_peak": True,
            "efficient_capture": False,
            "first_3d_return": -0.06,
            "first_5d_return": -0.09,
            "drawdown_after_peak": -0.12,
        },
    ])
    _write_table(analysis / "cluster_results.parquet", [
        {"sample_id": "s1", "cluster_id": "pattern_cluster_001", "return": 0.1, "max_favorable_excursion": 0.12, "max_adverse_excursion": -0.01, "binary_success": 1, "early_confirmed": True, "had_opportunity": True, "faded_after_peak": False, "efficient_capture": True},
        {"sample_id": "s2", "cluster_id": "pattern_cluster_001", "return": -0.1, "max_favorable_excursion": 0.01, "max_adverse_excursion": -0.12, "binary_success": 0, "early_confirmed": False, "had_opportunity": False, "faded_after_peak": True, "efficient_capture": False},
    ])
    figure = figures / "pattern_cluster_001_representatives.png"
    figure.write_bytes(b"png")
    service = services.PatternAnalysisService(root=root)

    analyses = service.list_analyses()
    detail = service.read_analysis("unit_analysis")
    samples = service.cluster_samples("unit_analysis", "pattern_cluster_001")
    similar = service.similar("unit_analysis", "s1", top_k=1)
    figure_path = service.figure_path("unit_analysis", "figures/clusters/pattern_cluster_001_representatives.png")

    assert analyses[0]["analysis_id"] == "unit_analysis"
    assert detail["summary"]["sample_count"] == 2
    assert detail["clusters"][0]["sample_count"] == 2
    assert detail["clusters"][0]["early_confirm_rate"] == pytest.approx(0.5)
    assert detail["clusters"][0]["opportunity_rate"] == pytest.approx(0.5)
    assert detail["clusters"][0]["fade_after_peak_rate"] == pytest.approx(0.5)
    assert samples["samples"][0]["sample_id"] == "s1"
    assert samples["samples"][0]["early_confirmed"] is True
    assert similar["matches"][0]["sample_id"] == "s2"
    assert similar["matches"][0]["faded_after_peak"] is True
    assert figure_path == figure


def test_pattern_analysis_service_build_resolves_run_paths(tmp_path, monkeypatch):
    captured = {}

    def fake_build_pattern_analysis(config):
        captured["runs"] = list(config.runs)
        return SimpleNamespace(
            sample_count=1,
            analysis_id="unit_patterns",
            output_dir=tmp_path / "pattern_analysis" / "unit_patterns",
            report_path=None,
            summary={"sample_count": 1},
        )

    monkeypatch.setattr(services, "build_pattern_analysis", fake_build_pattern_analysis)
    service = services.PatternAnalysisService(root=tmp_path / "pattern_analysis", runs_root=tmp_path / "runs")

    result = service.build(["runs/unit_run", "other_run"], analysis_id="unit_patterns")

    assert result["ok"] is True
    assert captured["runs"] == [
        (tmp_path / "runs" / "unit_run").resolve(),
        (tmp_path / "runs" / "other_run").resolve(),
    ]


def test_config_service_list_configs_includes_description(tmp_path):
    service = services.ConfigService(root=tmp_path)
    service.write_config("demo.yaml", "name: demo\ndescription: 策略说明\n")

    configs = service.list_configs()

    assert configs[0]["name"] == "demo"
    assert configs[0]["display_title"] == "demo"
    assert configs[0]["description"] == "策略说明"


def test_config_service_uses_explicit_short_title(tmp_path):
    service = services.ConfigService(root=tmp_path)
    service.write_config("demo.yaml", "name: demo\ntitle: 弱转强五仓\ndescription: 策略说明\n")

    configs = service.list_configs()

    assert configs[0]["display_title"] == "弱转强五仓"
    assert configs[0]["title"] == "弱转强五仓"


def test_config_service_filters_by_production_profile_include_order(tmp_path):
    root = tmp_path / "strategies"
    root.mkdir()
    first = root / "generated" / "weak.yaml"
    second = root / "shuijiao.yaml"
    ignored = root / "ignored.yaml"
    first.parent.mkdir()
    first.write_text("name: weak_strategy\ntitle: 弱转强代表\n", encoding="utf-8")
    second.write_text("name: shuijiao_strategy\ntitle: 睡觉策略\n", encoding="utf-8")
    ignored.write_text("name: ignored\ntitle: 忽略策略\n", encoding="utf-8")
    profile = tmp_path / "daily_default.yaml"
    profile.write_text(
        "strategies:\n"
        "  include:\n"
        f"    - {first}\n"
        f"    - {second}\n",
        encoding="utf-8",
    )
    service = services.ConfigService(root=root, production_profile=profile)

    configs = service.list_configs()

    assert [row["path"] for row in configs] == ["generated/weak.yaml", "shuijiao.yaml"]
    assert [row["display_title"] for row in configs] == ["弱转强代表", "睡觉策略"]


def test_report_service_enriches_description_and_position_summaries(tmp_path):
    run_dir = tmp_path / "unit_run"
    run_dir.mkdir()
    _write_json(run_dir / "summary.json", {
        "run_id": "unit_run",
        "name": "demo_strategy",
        "start_date": "2021-01-04",
        "end_date": "2021-01-06",
        "total_return": 0.1,
    })
    _write_json(run_dir / "metrics.json", {})
    _write_json(run_dir / "daily_nav.json", [
        {"date": "2021-01-04", "total_value": 1000.0},
        {"date": "2021-01-06", "total_value": 1100.0},
    ])
    _write_json(run_dir / "trades.json", [
        {"date": "2021-01-04", "symbol": "SZ000001", "action": "BUY", "price": 10.0, "quantity": 100, "reject_reason": ""},
        {"date": "2021-01-05", "symbol": "SZ000002", "action": "BUY", "price": 8.0, "quantity": 200, "reject_reason": ""},
        {"date": "2021-01-06", "symbol": "SZ000001", "action": "SELL", "price": 12.0, "quantity": 100, "reject_reason": ""},
    ])
    _write_json(run_dir / "positions.json", [
        {"date": "2021-01-06", "symbol": "SZ000002", "quantity": 200, "avg_cost": 8.0, "market_value": 1800.0, "holding_days": 2},
    ])
    _write_json(run_dir / "closed_positions.json", [
        {"symbol": "SZ000001", "entry_date": "2021-01-04", "exit_date": "2021-01-06", "quantity": 100, "entry_price": 10.0, "exit_price": 12.0, "net_pnl": 190.0, "return": 0.19},
        {"symbol": "SZ000003", "entry_date": "2021-01-04", "exit_date": "2021-01-06", "quantity": 100, "entry_price": 10.0, "exit_price": 9.0, "net_pnl": -110.0, "return": -0.11},
    ])
    _write_json(run_dir / "daily_selection_candidates.json", [
        {
            "date": "2021-01-06",
            "raw_candidate_count": 1,
            "selected_count": 1,
            "selected_candidates": [{"symbol": "SZ000004", "score": 1.0, "where": True}],
        }
    ])
    _write_json(run_dir / "explain.json", {"config": {"name": "demo_strategy", "data": {"provider_uri": "data/qlib_data_fixed"}}})
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo.yaml").write_text("name: demo_strategy\ndescription: 从当前配置补充的策略说明\n", encoding="utf-8")
    service = services.ReportService(root=tmp_path, meta_store=MetaStore(tmp_path / "meta.sqlite"), config_root=config_root)

    reports = service.list_reports()
    report = service.read_report("unit_run")

    assert reports[0]["description"] == "从当前配置补充的策略说明"
    assert reports[0]["display_title"] == "demo"
    assert report["summary"]["description"] == "从当前配置补充的策略说明"
    assert report["summary"]["display_title"] == "demo"
    assert report["display_title"] == "demo"
    assert report["best_closed_positions"]["gains"][0]["symbol"] == "SZ000001"
    assert report["best_closed_positions"]["losses"][0]["symbol"] == "SZ000003"
    assert report["current_positions"][0]["symbol"] == "SZ000002"
    assert report["current_positions"][0]["unrealized_pnl"] == pytest.approx(200.0)
    assert report["closed_position_summary"][0]["profit_label"] == "盈利"
    assert report["closed_position_summary"][1]["profit_label"] == "亏损"
    assert report["annual_returns"][0]["year"] == 2021
    assert report["next_session_guide"]["new_buy_candidates"][0]["symbol"] == "SZ000004"


def test_report_service_infers_bbi_position_title(tmp_path):
    run_dir = tmp_path / "full_bbi_short_long_base_pos5_topk4_close_strict"
    run_dir.mkdir()
    _write_json(run_dir / "summary.json", {
        "run_id": run_dir.name,
        "name": "stocktradebyz_bbi_short_long_base_pos5_topk4",
        "start_date": "2021-01-04",
        "end_date": "2021-01-06",
        "total_return": 0.1,
    })
    _write_json(run_dir / "metrics.json", {})
    _write_json(run_dir / "daily_nav.json", [])
    _write_json(run_dir / "trades.json", [])
    _write_json(run_dir / "positions.json", [])
    _write_json(run_dir / "closed_positions.json", [])
    _write_json(run_dir / "explain.json", {"config": {"name": "stocktradebyz_bbi_short_long_base_pos5_topk4"}})
    service = services.ReportService(root=tmp_path, meta_store=MetaStore(tmp_path / "meta.sqlite"))

    reports = service.list_reports()
    report = service.read_report(run_dir.name)

    assert reports[0]["display_title"] == "弱转强5仓"
    assert report["summary"]["display_title"] == "弱转强5仓"


def test_report_service_filters_reports_by_production_profile_and_uses_config_title(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for run_id, name in (
        ("weak_run", "weak_strategy"),
        ("shuijiao_run", "shuijiao_strategy"),
        ("ignored_run", "ignored_strategy"),
    ):
        run_dir = runs_root / run_id
        run_dir.mkdir()
        _write_json(run_dir / "summary.json", {
            "run_id": run_id,
            "name": name,
            "start_date": "2021-01-04",
            "end_date": "2021-01-06",
            "total_return": 0.1,
        })
        _write_json(run_dir / "metrics.json", {
            "total_return": 0.2 if name == "weak_strategy" else 0.1,
            "annual_return": 0.12,
            "max_drawdown": -0.08,
            "sharpe": 1.5,
            "win_rate": 0.58,
            "avg_holding_days": 15.0,
            "avg_position_count": 5.5,
            "trade_count": 20,
            "final_value": 1200000.0,
        })
        _write_json(run_dir / "daily_nav.json", [])
        _write_json(run_dir / "trades.json", [])
        _write_json(run_dir / "positions.json", [])
        _write_json(run_dir / "closed_positions.json", [])
        _write_json(run_dir / "explain.json", {"config": {"name": name}})
    config_root = tmp_path / "strategies"
    config_root.mkdir()
    weak_config = config_root / "weak.yaml"
    shuijiao_config = config_root / "shuijiao.yaml"
    weak_config.write_text("name: weak_strategy\ntitle: 弱转强代表\n", encoding="utf-8")
    shuijiao_config.write_text("name: shuijiao_strategy\ntitle: 睡觉策略\n", encoding="utf-8")
    profile = tmp_path / "daily_default.yaml"
    profile.write_text(
        "strategies:\n"
        "  include:\n"
        f"    - {weak_config}\n"
        f"    - {shuijiao_config}\n",
        encoding="utf-8",
    )
    service = services.ReportService(
        root=runs_root,
        meta_store=MetaStore(tmp_path / "meta.sqlite"),
        config_root=config_root,
        production_profile=profile,
    )

    reports = service.list_reports()
    weak = service.read_report("weak_run")

    assert {row["run_id"] for row in reports} == {"weak_run", "shuijiao_run"}
    assert {row["display_title"] for row in reports} == {"弱转强代表", "睡觉策略"}
    assert reports[0]["win_rate"] == pytest.approx(0.58)
    assert reports[0]["avg_holding_days"] == pytest.approx(15.0)
    assert reports[0]["avg_position_count"] == pytest.approx(5.5)
    assert reports[0]["trade_count"] == 20
    assert weak["summary"]["display_title"] == "弱转强代表"


def test_report_service_allows_profile_report_include_for_custom_run(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for run_id, name in (
        ("weak_run", "weak_strategy"),
        ("wufu_run", "etf_wufu_corr_daily_2016_2026"),
        ("ignored_run", "ignored_strategy"),
    ):
        run_dir = runs_root / run_id
        run_dir.mkdir()
        _write_json(run_dir / "summary.json", {
            "run_id": run_id,
            "name": name,
            "start_date": "2021-01-04",
            "end_date": "2021-01-06",
            "total_return": 0.1,
        })
        _write_json(run_dir / "metrics.json", {"total_return": 0.1, "trade_count": 3})
        _write_json(run_dir / "daily_nav.json", [])
        _write_json(run_dir / "trades.json", [])
        _write_json(run_dir / "positions.json", [])
        _write_json(run_dir / "closed_positions.json", [])
        _write_json(run_dir / "explain.json", {"config": {"name": name}})
    config_root = tmp_path / "strategies"
    config_root.mkdir()
    weak_config = config_root / "weak.yaml"
    weak_config.write_text("name: weak_strategy\ntitle: 弱转强代表\n", encoding="utf-8")
    profile = tmp_path / "daily_default.yaml"
    profile.write_text(
        "strategies:\n"
        "  include:\n"
        f"    - {weak_config}\n"
        "reports:\n"
        "  include:\n"
        "    - run_id: wufu_run\n"
        "      title: 五福ETF\n"
        "      description: 五福 ETF 全池复刻\n",
        encoding="utf-8",
    )
    service = services.ReportService(
        root=runs_root,
        meta_store=MetaStore(tmp_path / "meta.sqlite"),
        config_root=config_root,
        production_profile=profile,
    )

    reports = service.list_reports()
    wufu = service.read_report("wufu_run")

    assert {row["run_id"] for row in reports} == {"weak_run", "wufu_run"}
    assert {row["display_title"] for row in reports} == {"弱转强代表", "五福ETF"}
    assert wufu["summary"]["display_title"] == "五福ETF"
    assert wufu["summary"]["description"] == "五福 ETF 全池复刻"


def test_report_service_keeps_multiple_explicit_profile_reports_with_same_name(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for run_id, title, total_return in (
        ("wufu_base", "五福ETF", 0.1),
        ("wufu_confirm", "五福ETF确认", 0.2),
    ):
        run_dir = runs_root / run_id
        run_dir.mkdir()
        _write_json(run_dir / "summary.json", {
            "run_id": run_id,
            "name": "etf_wufu_corr_daily_2016_2026",
            "start_date": "2021-01-04",
            "end_date": "2021-01-06",
            "total_return": total_return,
        })
        _write_json(run_dir / "metrics.json", {"total_return": total_return, "trade_count": 3})
        _write_json(run_dir / "daily_nav.json", [])
        _write_json(run_dir / "trades.json", [])
        _write_json(run_dir / "positions.json", [])
        _write_json(run_dir / "closed_positions.json", [])
        _write_json(run_dir / "explain.json", {"config": {"name": "etf_wufu_corr_daily_2016_2026"}})
    profile = tmp_path / "daily_default.yaml"
    profile.write_text(
        "reports:\n"
        "  include:\n"
        "    - run_id: wufu_base\n"
        "      title: 五福ETF\n"
        "    - run_id: wufu_confirm\n"
        "      title: 五福ETF确认\n",
        encoding="utf-8",
    )
    service = services.ReportService(
        root=runs_root,
        meta_store=MetaStore(tmp_path / "meta.sqlite"),
        config_root=tmp_path / "strategies",
        production_profile=profile,
    )

    reports = service.list_reports()

    assert {row["run_id"] for row in reports} == {"wufu_base", "wufu_confirm"}
    assert {row["display_title"] for row in reports} == {"五福ETF", "五福ETF确认"}


def test_report_service_current_positions_uses_summary_end_date(tmp_path):
    run_dir = tmp_path / "unit_run"
    run_dir.mkdir()
    _write_json(run_dir / "summary.json", {
        "run_id": "unit_run",
        "name": "demo_strategy",
        "start_date": "2021-01-04",
        "end_date": "2021-01-06",
        "final_positions": 0,
    })
    _write_json(run_dir / "metrics.json", {})
    _write_json(run_dir / "daily_nav.json", [
        {"date": "2021-01-04", "position_count": 1},
        {"date": "2021-01-05", "position_count": 1},
        {"date": "2021-01-06", "position_count": 0},
    ])
    _write_json(run_dir / "trades.json", [
        {"date": "2021-01-04", "symbol": "SZ000001", "action": "BUY", "price": 10.0, "quantity": 100, "reject_reason": ""},
        {"date": "2021-01-06", "symbol": "SZ000001", "action": "SELL", "price": 11.0, "quantity": 100, "reject_reason": ""},
    ])
    _write_json(run_dir / "positions.json", [
        {"date": "2021-01-04", "symbol": "SZ000001", "quantity": 100, "avg_cost": 10.0, "market_value": 1000.0, "holding_days": 1},
        {"date": "2021-01-05", "symbol": "SZ000001", "quantity": 100, "avg_cost": 10.0, "market_value": 1050.0, "holding_days": 2},
    ])
    _write_json(run_dir / "closed_positions.json", [])
    _write_json(run_dir / "explain.json", {"config": {"name": "demo_strategy"}})
    service = services.ReportService(root=tmp_path, meta_store=MetaStore(tmp_path / "meta.sqlite"))

    report = service.read_report("unit_run")

    assert report["summary"]["end_date"] == "2021-01-06"
    assert report["current_positions"] == []


def test_report_service_read_artifact_accepts_dash_alias(tmp_path):
    run_dir = tmp_path / "unit_run"
    run_dir.mkdir()
    _write_json(run_dir / "summary.json", {"run_id": "unit_run"})
    _write_json(run_dir / "daily_nav.json", [{"date": "2021-01-04", "total_value": 1000.0}])
    service = services.ReportService(root=tmp_path, meta_store=MetaStore(tmp_path / "meta.sqlite"))

    assert service.read_artifact("unit_run", "daily-nav") == [{"date": "2021-01-04", "total_value": 1000.0}]


def test_report_service_symbol_detail_returns_bars_and_bs_points(tmp_path, monkeypatch):
    run_dir = tmp_path / "unit_run"
    run_dir.mkdir()
    _write_json(run_dir / "summary.json", {
        "run_id": "unit_run",
        "start_date": "2021-01-04",
        "end_date": "2021-01-06",
        "total_return": 0.1,
    })
    _write_json(run_dir / "metrics.json", {})
    _write_json(run_dir / "daily_nav.json", [])
    _write_json(run_dir / "positions.json", [])
    _write_json(run_dir / "closed_positions.json", [])
    _write_json(run_dir / "explain.json", {"config": {"data": {"provider_uri": "data/qlib_data_fixed"}}})
    _write_json(run_dir / "trades.json", [
        {"date": "2021-01-04", "symbol": "SZ000001", "action": "BUY", "price": 10.0, "quantity": 100, "trade_value": 1000.0, "reject_reason": ""},
        {"date": "2021-01-06", "symbol": "SZ000001", "action": "SELL", "price": 12.0, "quantity": 100, "trade_value": 1200.0, "reject_reason": ""},
    ])

    class FakeExchange:
        def __init__(self, provider_uri):
            self.provider_uri = provider_uri
            self.quote = None

        def load_quote_data(self, symbols, start, end):
            index = pd.MultiIndex.from_product(
                [pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]), symbols],
                names=["datetime", "instrument"],
            )
            self.quote = pd.DataFrame({
                "$open": [10.0, 11.0, 11.5],
                "$high": [10.5, 11.5, 12.5],
                "$low": [9.8, 10.8, 11.2],
                "$close": [10.2, 11.3, 12.0],
                "$volume": [1000, 1200, 1500],
                "$change": [0.02, 0.10, 0.06],
                "$vwap": [10.1, 11.1, 12.0],
            }, index=index)

    monkeypatch.setattr(services, "AStockExchange", FakeExchange)
    meta_store = MetaStore(tmp_path / "meta.sqlite")
    meta_store.upsert_security_master(pd.DataFrame([{"symbol": "SZ000001", "name": "平安银行"}]))
    meta_store.upsert_industry_membership(pd.DataFrame([{"symbol": "SZ000001", "industry_name": "银行"}]))
    service = services.ReportService(root=tmp_path, meta_store=meta_store)

    detail = service.read_symbol_detail("unit_run", "SZ000001")

    assert detail["symbol"] == "SZ000001"
    assert detail["name"] == "平安银行"
    assert detail["meta"]["industry_name"] == "银行"
    assert len(detail["bars"]) == 3
    assert [trade["action"] for trade in detail["trades"]] == ["BUY", "SELL"]
    assert detail["round_trips"][0]["return"] == pytest.approx(0.2)
