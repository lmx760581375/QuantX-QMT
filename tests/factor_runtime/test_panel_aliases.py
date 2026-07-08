"""MarketPanel field alias tests."""

import pandas as pd
import pytest

from quantx.core.factor_runtime import MarketPanel


def test_market_panel_adds_config_field_aliases():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04", "2021-01-05"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"$close": [10.0, 11.0]}, index=index)

    panel = MarketPanel.from_frame(frame, aliases={"my_close": "$close"})

    assert "close" in panel.fields
    assert "my_close" in panel.fields
    assert panel.fields["my_close"].tolist() == [[10.0], [11.0]]


def test_market_panel_rejects_missing_alias_source():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["SZ000001"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"$close": [10.0]}, index=index)

    with pytest.raises(ValueError, match="missing source field"):
        MarketPanel.from_frame(frame, aliases={"my_open": "$open"})
