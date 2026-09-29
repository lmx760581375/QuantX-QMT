from quantx.core.engine.account import Account
from quantx.tools.run_dual_sleeve_backtest import (
    is_rebalance_date,
    requested_transfer,
    transfer_available_cash,
)


def test_rebalance_schedule_uses_first_session_of_new_period():
    assert not is_rebalance_date(None, "2021-01-04", "monthly")
    assert not is_rebalance_date("2021-01-04", "2021-01-05", "monthly")
    assert is_rebalance_date("2021-01-29", "2021-02-01", "monthly")
    assert not is_rebalance_date("2021-01-29", "2021-02-01", "quarterly")
    assert is_rebalance_date("2021-03-31", "2021-04-01", "quarterly")
    assert not is_rebalance_date("2021-03-31", "2021-04-01", "none")


def test_requested_transfer_moves_from_overweight_sleeve():
    assert requested_transfer(70.0, 30.0, wts_weight=0.65) == ("wts", "rm", 5.0)
    assert requested_transfer(55.0, 45.0, wts_weight=0.65) == ("rm", "wts", 10.0)


def test_transfer_is_cash_limited_and_preserves_combined_value():
    accounts = {
        "wts": Account(init_cash=2.0),
        "rm": Account(init_cash=5.0),
    }
    accounts["wts"]._last_total_value = 70.0
    accounts["wts"]._peak_total_value = 80.0
    accounts["rm"]._last_total_value = 30.0
    accounts["rm"]._peak_total_value = 35.0
    before = accounts["wts"].cash + accounts["rm"].cash

    transferred = transfer_available_cash(
        accounts,
        source="wts",
        target="rm",
        requested=5.0,
    )

    assert transferred == 2.0
    assert accounts["wts"].cash == 0.0
    assert accounts["rm"].cash == 7.0
    assert accounts["wts"].cash + accounts["rm"].cash == before
    assert accounts["wts"]._last_total_value == 68.0
    assert accounts["rm"]._last_total_value == 32.0


def test_account_rejects_external_cash_outflow_above_available_cash():
    account = Account(init_cash=10.0)

    try:
        account.apply_external_cash_flow(-11.0)
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected an excessive external cash outflow to fail")
