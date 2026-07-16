"""turnover feature builder tests."""

import numpy as np
import pandas as pd

from quantx.tools.build_turnover_features import write_qlib_features


def _components(dates, turnovers):
    close = np.full(len(dates), 10.0)
    return pd.DataFrame({
        "instrument": ["SH600000"] * len(dates),
        "date": pd.to_datetime(dates),
        "turnover_free_float": turnovers,
        "turnover_circulating": turnovers,
        "circ_mv": close * 1000,
        "free_float_mv": close * 800,
    })


def _read_values(path):
    raw = np.fromfile(str(path), dtype="<f4")
    return int(raw[0]), raw[1:]


def test_write_qlib_features_merges_incremental_window(tmp_path):
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-07-14", "2026-07-15", "2026-07-16"]))

    write_qlib_features(
        tmp_path,
        _components(["2026-07-14", "2026-07-15"], [0.01, 0.02]),
        calendar=calendar,
        fields=["turnover_rate"],
    )
    write_qlib_features(
        tmp_path,
        _components(["2026-07-15", "2026-07-16"], [0.22, 0.03]),
        calendar=calendar,
        fields=["turnover_rate"],
    )

    start_idx, values = _read_values(tmp_path / "features" / "sh600000" / "turnover_rate.day.bin")

    assert start_idx == 0
    np.testing.assert_allclose(values, np.array([0.01, 0.22, 0.03], dtype=np.float32))
