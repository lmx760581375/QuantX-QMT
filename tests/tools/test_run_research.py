"""Research CLI configuration integration tests."""

import numpy as np
import pandas as pd
import yaml

from quantx.tools.run_research import _build_universe, run_research_config


def write_field(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0, *values], dtype="<f4").tofile(path)


def test_run_research_config_builds_qmt_dataset_and_oos_predictions(tmp_path):
    provider = tmp_path / "provider"
    dates = pd.bdate_range("2019-01-02", "2022-12-30")
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    (provider / "instruments" / "all.txt").write_text(
        f"SZ000001\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\nSH600000\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n",
        encoding="utf-8",
    )
    sequence = np.arange(len(dates), dtype=float)
    for symbol, offset in (("sz000001", 0.0), ("sh600000", 5.0)):
        opens = 10.0 + offset + sequence * 0.01 + np.sin(sequence / 30)
        closes = opens + np.cos(sequence / 20) * 0.1
        write_field(provider / "features" / symbol / "open.day.bin", opens)
        write_field(provider / "features" / symbol / "close.day.bin", closes)
    config = {
        "name": "ridge_cli_test",
        "data": {
            "provider_uri": str(provider),
            "universe": ["SZ000001", "SH600000"],
            "start": "2019-01-02",
            "end": "2022-12-30",
            "integrity_mode": "exploratory",
        },
        "features": {
            "raw_fields": ["open", "close"],
            "expressions": {"ret1": "close / Ref(close, 1) - 1"},
            "lookback_sessions": 2,
        },
        "label": {"horizon_sessions": 5},
        "validation": {
            "first_prediction_year": 2021,
            "last_prediction_year": 2022,
            "validation_sessions": 40,
            "embargo_sessions": 5,
        },
        "model": {"type": "ridge", "seed": 7, "params": {"alpha": 1.0}},
        "output": {"root": str(tmp_path / "research"), "allow_project_output": True},
    }
    config_path = tmp_path / "research.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    dry_run = run_research_config(config_path, dry_run=True)
    result = run_research_config(config_path)

    assert dry_run["dry_run"] is True
    assert dry_run["dataset_rows"] > 0
    assert len(dry_run["folds"]) == 2
    assert dry_run["execution_audit"]["label_available_count"] > 0
    assert "entry_one_price_limit_up" in dry_run["execution_audit"]["gate_columns"]
    assert result["fold_count"] == 2
    assert result["prediction_count"] > 0
    assert result["metrics"]["rank_ic"] is not None
    assert result["universe_audit"]["passed"] is True
    assert result["execution_audit"]["contaminated_count"] >= 0


def test_run_research_execution_audit_flags_untradable_entries(tmp_path):
    provider = tmp_path / "provider"
    dates = pd.bdate_range("2019-01-02", "2022-12-30")
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    (provider / "instruments" / "all.txt").write_text(
        f"SZ000001\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\nSH600000\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n",
        encoding="utf-8",
    )
    sequence = np.arange(len(dates), dtype=float)
    for symbol, offset in (("sz000001", 0.0), ("sh600000", 5.0)):
        opens = 10.0 + offset + sequence * 0.01
        highs = opens + 0.1
        lows = opens - 0.1
        closes = opens + 0.02
        volume = np.full(len(dates), 1000.0)
        amount = volume * opens
        if symbol == "sz000001":
            opens[1] = highs[1] = lows[1] = closes[1] = closes[0] * 1.05
            volume[1] = 1000.0
            amount[1] = volume[1] * opens[1]
        if symbol == "sh600000":
            volume[1] = 0.0
            amount[1] = 0.0
        for field, values in {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "amount": amount,
        }.items():
            write_field(provider / "features" / symbol / f"{field}.day.bin", values)
    config = {
        "name": "audit_cli_test",
        "data": {
            "provider_uri": str(provider),
            "universe": ["SZ000001", "SH600000"],
            "start": "2019-01-02",
            "end": "2022-12-30",
            "integrity_mode": "exploratory",
        },
        "features": {"raw_fields": ["open", "close"], "lookback_sessions": 2},
        "label": {"horizon_sessions": 5, "execution_diagnostic": True},
        "validation": {
            "first_prediction_year": 2021,
            "last_prediction_year": 2022,
            "validation_sessions": 40,
            "embargo_sessions": 5,
        },
        "model": {"type": "ridge", "seed": 7, "params": {"alpha": 1.0}},
        "output": {"root": str(tmp_path / "research"), "allow_project_output": True},
    }
    config_path = tmp_path / "research.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    dry_run = run_research_config(config_path, dry_run=True)

    audit = dry_run["execution_audit"]
    assert audit["counts"]["one_price_limit_up"] >= 1
    assert audit["counts"]["zero_volume_or_amount"] >= 1
    assert audit["contaminated_count"] >= 2

    config["data"].update(
        {
            "integrity_mode": "formal",
            "st_status_source": "test-pit-st-source",
            "suspension_status_source": "test-pit-suspension-source",
        }
    )
    config["lifecycle"] = {
        "action": "promotion_candidate",
        "root": str(tmp_path / "lifecycle"),
        "snapshot_root": str(tmp_path / "snapshots"),
        "holdout_range": ["2023-01-01", "2023-12-31"],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    promoted = run_research_config(config_path)

    assert promoted["lifecycle"]["experiment_lock_hash"].startswith("sha256:")
    assert promoted["lifecycle"]["physical_snapshot_id"].startswith("snapshot:")


def test_mainboard_universe_excludes_funds_and_sampling_is_reproducible(tmp_path):
    provider = tmp_path / "provider"
    (provider / "instruments").mkdir(parents=True)
    (provider / "instruments" / "all.txt").write_text(
        "\n".join(
            [
                "SH510300\t2020-01-01\t2026-01-01",
                "SH600000\t2020-01-01\t2026-01-01",
                "SH601398\t2020-01-01\t2026-01-01",
                "SH688001\t2020-01-01\t2026-01-01",
                "SZ000001\t2020-01-01\t2026-01-01",
                "SZ002415\t2020-01-01\t2026-01-01",
                "SZ300750\t2020-01-01\t2026-01-01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "universe": "all_mainboard",
        "symbol_sample_size": 3,
        "symbol_sample_seed": 11,
    }

    first = _build_universe(provider, config)
    second = _build_universe(provider, config)

    assert first.all_symbols == second.all_symbols
    assert len(first.all_symbols) == 3
    assert all(not symbol.startswith(("SH510", "SH688", "SZ300")) for symbol in first.all_symbols)
