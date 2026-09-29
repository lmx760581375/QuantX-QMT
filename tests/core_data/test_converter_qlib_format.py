"""Qlib converter binary format regression tests."""

import numpy as np
import pandas as pd

from quantx.core.data.converter import BaostockToQlibConverter


def test_write_features_uses_qlib_start_index_layout(tmp_path):
    converter = BaostockToQlibConverter(qlib_dir=str(tmp_path))
    calendar = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    frame = pd.DataFrame({
        "date": ["2021-01-04", "2021-01-06"],
        "$close": [10.0, 12.0],
    })

    converter._write_features({"SZ000001": frame}, list(calendar))

    data = np.fromfile(tmp_path / "features" / "sz000001" / "close.day.bin", dtype="<f4")
    assert data[0] == 0.0
    np.testing.assert_allclose(data[[1, 3]], [10.0, 12.0])
    assert np.isnan(data[2])


def test_write_features_append_mode_rewrites_file_instead_of_pair_appending(tmp_path):
    converter = BaostockToQlibConverter(qlib_dir=str(tmp_path))
    calendar = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    first = pd.DataFrame({
        "date": ["2021-01-04"],
        "$close": [10.0],
    })
    updated = pd.DataFrame({
        "date": ["2021-01-04", "2021-01-05", "2021-01-06"],
        "$close": [10.0, 11.0, 12.0],
    })

    converter._write_features({"SZ000001": first}, list(calendar))
    converter._write_features({"SZ000001": updated}, list(calendar), mode="append")

    data = np.fromfile(tmp_path / "features" / "sz000001" / "close.day.bin", dtype="<f4")
    np.testing.assert_allclose(data, [0.0, 10.0, 11.0, 12.0])


def test_convert_incremental_updates_instruments_dates(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    qlib_dir = tmp_path / "qlib"
    converter = BaostockToQlibConverter(qlib_dir=str(qlib_dir), csv_dir=str(csv_dir))
    frame = pd.DataFrame({
        "date": ["2021-01-04", "2021-01-05"],
        "code": ["SZ000001", "SZ000001"],
        "open": [10.0, 11.0],
        "high": [10.5, 11.5],
        "low": [9.5, 10.5],
        "close": [10.0, 11.0],
        "preclose": [10.0, 10.0],
        "volume": [1000, 1200],
        "amount": [1000000, 1320000],
    })
    frame.to_csv(csv_dir / "SZ000001.csv", index=False)
    converter.convert_all("2021-01-04", "2021-01-04", ["SZ000001"])
    converter.convert_incremental(["SZ000001"])

    text = (qlib_dir / "instruments" / "all.txt").read_text(encoding="utf-8")
    assert "SZ000001\t2021-01-04\t2021-01-05" in text


def test_convert_incremental_preserves_existing_instrument_start_date(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "instruments").mkdir(parents=True)
    (qlib_dir / "calendars" / "day.txt").write_text(
        "2020-01-02\n2026-07-31\n2026-08-03\n",
        encoding="utf-8",
    )
    (qlib_dir / "instruments" / "all.txt").write_text(
        "SZ000801\t2010-01-04\t2026-07-30\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2026-07-31", "2026-08-03"],
            "code": ["SZ000801", "SZ000801"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "preclose": [9.9, 10.1],
            "volume": [1000, 1200],
            "amount": [1_000_000, 1_200_000],
        }
    ).to_csv(csv_dir / "SZ000801.csv", index=False)

    converter = BaostockToQlibConverter(
        qlib_dir=str(qlib_dir),
        csv_dir=str(csv_dir),
    )
    converter.convert_incremental(["SZ000801"])

    text = (qlib_dir / "instruments" / "all.txt").read_text(encoding="utf-8")
    assert "SZ000801\t2010-01-04\t2026-08-03" in text
