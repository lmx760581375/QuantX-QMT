import pandas as pd
import pytest

from quantx.tools.plot_sleeve_fusion import (
    build_static_sleeve_nav,
    calendar_month_win_rates,
    monthly_return_heatmap,
    monthly_returns,
)


def test_static_sleeve_nav_uses_initial_capital_weights_without_rebalancing():
    reward = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
        "total_value": [100.0, 110.0, 121.0],
    })
    wts = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
        "total_value": [100.0, 100.0, 100.0],
    })

    output = build_static_sleeve_nav(reward, wts, wts_weight=0.5)

    assert output["portfolio_normalized"].tolist() == pytest.approx([1.0, 1.05, 1.105])
    assert output["portfolio_drawdown"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_monthly_returns_use_previous_month_end_as_next_month_start():
    nav = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04", "2021-01-29", "2021-02-01", "2021-02-26"]),
        "portfolio_normalized": [1.0, 1.10, 1.10, 1.21],
    })

    output = monthly_returns(nav)

    assert output["monthly_return"].tolist() == pytest.approx([0.10, 0.10])


def test_calendar_month_win_rate_is_the_frequency_of_positive_monthly_returns():
    monthly = pd.DataFrame({
        "month": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2020-02-01"]),
        "monthly_return": [0.10, -0.05, 0.02, -0.01],
    })

    output = calendar_month_win_rates(monthly)

    january = output.loc[output["month_number"] == 1].iloc[0]
    february = output.loc[output["month_number"] == 2].iloc[0]
    assert january["observed_months"] == 3
    assert january["red_months"] == 2
    assert january["win_rate"] == pytest.approx(2 / 3)
    assert february["win_rate"] == 0.0


def test_monthly_return_heatmap_uses_year_rows_and_calendar_month_columns():
    monthly = pd.DataFrame({
        "month": pd.to_datetime(["2021-01-01", "2021-03-01", "2022-02-01"]),
        "monthly_return": [0.10, -0.05, 0.02],
    })

    output = monthly_return_heatmap(monthly)

    assert output.index.tolist() == [2021, 2022]
    assert output.columns.tolist() == list(range(1, 13))
    assert output.loc[2021, 1] == pytest.approx(0.10)
    assert output.loc[2021, 2] != output.loc[2021, 2]
    assert output.loc[2022, 2] == pytest.approx(0.02)
