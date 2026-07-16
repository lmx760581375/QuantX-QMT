import pandas as pd

from quantx.tools.sync_st_history import (
    RAW_COLUMNS,
    FetchResult,
    build_st_intervals,
    empty_coverage,
    empty_observed,
    forward_fill_empty_dates,
    merge_fetch_results,
    normalize_observed,
)


def _observed(date: str, code: str = "000001.SZ", name: str = "ST测试") -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": code,
        "name": name,
        "trade_date": date,
        "type": "ST",
        "type_name": "风险警示板",
    }], columns=RAW_COLUMNS)


def test_normalize_observed_rejects_cross_date_response():
    frame = _observed("20240102")

    try:
        normalize_observed(frame, requested_date="20240103")
    except ValueError as exc:
        assert "returned dates" in str(exc)
    else:
        raise AssertionError("cross-date response must be rejected")


def test_empty_date_after_observation_is_forward_filled_with_provenance():
    calendar = ["20240102", "20240103", "20240104"]
    results = [
        FetchResult("20240102", empty_observed(), 1),
        FetchResult("20240103", _observed("20240103"), 1),
        FetchResult("20240104", empty_observed(), 2),
    ]
    observed, coverage = merge_fetch_results(empty_observed(), empty_coverage(), results)

    daily, effective_coverage = forward_fill_empty_dates(observed, coverage, calendar)

    assert daily["trade_date"].tolist() == ["20240103", "20240104"]
    assert daily["source_trade_date"].tolist() == ["20240103", "20240103"]
    assert daily["is_imputed"].tolist() == [False, True]
    first = effective_coverage.set_index("trade_date").loc["20240102"]
    assert first["effective_row_count"] == 0
    assert first["source_trade_date"] == ""
    filled = effective_coverage.set_index("trade_date").loc["20240104"]
    assert filled["effective_row_count"] == 1
    assert bool(filled["is_imputed"])


def test_intervals_merge_adjacent_imputed_trading_days():
    calendar = ["20240102", "20240103", "20240104", "20240105"]
    results = [
        FetchResult("20240102", _observed("20240102"), 1),
        FetchResult("20240103", empty_observed(), 1),
        FetchResult("20240104", _observed("20240104"), 1),
        FetchResult("20240105", empty_observed(), 1),
    ]
    observed, coverage = merge_fetch_results(empty_observed(), empty_coverage(), results)
    daily, _ = forward_fill_empty_dates(observed, coverage, calendar)

    intervals = build_st_intervals(daily, calendar)

    assert len(intervals) == 1
    assert intervals.loc[0, "start_date"] == "20240102"
    assert intervals.loc[0, "end_date"] == "20240105"
    assert intervals.loc[0, "trading_days"] == 4
    assert intervals.loc[0, "imputed_days"] == 2


def test_failed_fetch_is_not_treated_as_empty():
    observed, coverage = merge_fetch_results(
        empty_observed(),
        empty_coverage(),
        [FetchResult("20240102", empty_observed(), 3, "Timeout: provider unavailable")],
    )

    assert observed.empty
    assert coverage.loc[0, "fetch_status"] == "failed"
    try:
        forward_fill_empty_dates(observed, coverage, ["20240102"])
    except ValueError as exc:
        assert "successful fetch" in str(exc)
    else:
        raise AssertionError("failed fetch must block effective data construction")
