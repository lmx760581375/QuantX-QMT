from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantx.core.data.manifest import (
    verify_data_manifest,
    write_data_manifest,
)
from quantx.tools.bootstrap_daily_data import (
    bootstrap_daily_data,
    build_historical_st_snapshot,
    load_symbols,
)


def _write_qlib_bin(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0, *values], dtype="<f4").tofile(path)


def test_load_symbols_preserves_reference_provider_universe(tmp_path):
    path = tmp_path / "all.txt"
    path.write_text(
        "SH600000 2020-01-01 2026-09-28\n"
        "SZ399001 2020-01-01 2026-09-28\n",
        encoding="utf-8",
    )

    assert load_symbols(path) == ["SH600000"]
    assert load_symbols(path, a_share_only=False) == ["SH600000", "SZ399001"]


def test_bootstrap_dry_run_neither_uses_network_nor_writes_data(tmp_path):
    universe = tmp_path / "all.txt"
    universe.write_text("SH600000 2020-01-01 2026-09-28\n", encoding="utf-8")
    result = bootstrap_daily_data(
        data_root=tmp_path / "data",
        start="2020-01-01",
        end="2026-09-28",
        universe_path=universe,
        security_master_path=tmp_path / "missing-security.csv",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["symbols"] == 1
    assert not (tmp_path / "data").exists()


def test_bootstrap_refuses_existing_data_without_force_or_resume(tmp_path):
    universe = tmp_path / "all.txt"
    universe.write_text("SH600000 2020-01-01 2026-09-28\n", encoding="utf-8")
    (tmp_path / "data" / "raw" / "baostock" / "stocks").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="without --force"):
        bootstrap_daily_data(
            data_root=tmp_path / "data",
            start="2020-01-01",
            end="2026-09-28",
            universe_path=universe,
        )


def test_build_historical_st_snapshot_from_raw_csv(tmp_path):
    raw = tmp_path / "stocks"
    raw.mkdir()
    pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "is_st": [0, 1],
    }).to_csv(raw / "SH600000.csv", index=False)

    summary = build_historical_st_snapshot(raw, tmp_path / "historical_st.parquet")

    frame = pd.read_parquet(tmp_path / "historical_st.parquet")
    assert summary["rows"] == 1
    assert frame.to_dict(orient="records") == [{
        "ts_code": "SH600000",
        "trade_date": "2026-09-02",
        "is_st": 1,
    }]


def test_data_manifest_round_trip(tmp_path):
    data_root = tmp_path / "data"
    provider = data_root / "qlib_data_fixed"
    raw = data_root / "raw" / "baostock" / "stocks"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir(parents=True)
    raw.mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2026-09-01\n2026-09-02\n",
        encoding="utf-8",
    )
    (provider / "instruments" / "all.txt").write_text(
        "SH600000 2026-09-01 2026-09-02\n",
        encoding="utf-8",
    )
    _write_qlib_bin(provider / "features" / "sh600000" / "close.day.bin", [10.0, 11.0])
    pd.DataFrame({"date": ["2026-09-01"], "close": [10.0]}).to_csv(
        raw / "SH600000.csv",
        index=False,
    )

    manifest_path = data_root / "MANIFEST.json"
    written = write_data_manifest(data_root, manifest_path)
    verified = verify_data_manifest(data_root, manifest_path)

    assert written["coverage"]["end"] == "2026-09-02"
    assert verified["ok"] is True
    assert json.loads(manifest_path.read_text())["kind"] == "quantx_daily_data_manifest_v1"

    (raw / "SH600000.csv").write_text("date,close\n2026-09-01,11\n", encoding="utf-8")
    assert verify_data_manifest(data_root, manifest_path)["ok"] is False
