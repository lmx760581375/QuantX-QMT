"""Decision-plane contract tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.types import (
    AlphaSnapshot,
    AlphaValue,
    ExecutionContext,
    FinalTargetPortfolio,
    PortfolioState,
    RawTargetPortfolio,
    TradabilityState,
)
from quantx.core.strategy.base import AccountSnapshot


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day: int, hour: int, phase: SessionPhase) -> MarketTime:
    return MarketTime(
        session=f"2026-07-{day:02d}",
        phase=phase,
        timestamp=datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def test_market_time_requires_timezone_aware_shanghai_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketTime(
            session="2026-07-10",
            phase=SessionPhase.CLOSE,
            timestamp=datetime(2026, 7, 10, 15),
        )

    with pytest.raises(ValueError, match="Asia/Shanghai"):
        MarketTime(
            session="2026-07-10",
            phase=SessionPhase.CLOSE,
            timestamp=datetime(2026, 7, 10, 15, tzinfo=ZoneInfo("UTC")),
        )


def test_alpha_snapshot_requires_strictly_later_execution_and_unique_values():
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)

    snapshot = AlphaSnapshot(
        signal_time=signal_time,
        earliest_execution_time=execution_time,
        values=(AlphaValue("SZ000001", 1.2), AlphaValue("SH600000", 0.7)),
        source_id="model:fold-1",
    )
    assert snapshot.values[0].instrument == "SZ000001"

    with pytest.raises(ValueError, match="after signal_time"):
        AlphaSnapshot(
            signal_time=signal_time,
            earliest_execution_time=signal_time,
            values=(AlphaValue("SZ000001", 1.2),),
            source_id="invalid-time",
        )

    with pytest.raises(ValueError, match="duplicate"):
        AlphaSnapshot(
            signal_time=signal_time,
            earliest_execution_time=execution_time,
            values=(AlphaValue("SZ000001", 1.2), AlphaValue("SZ000001", 0.7)),
            source_id="duplicates",
        )


def test_target_portfolios_validate_weight_budget_and_universe():
    signal_time = market_time(10, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    raw = RawTargetPortfolio(
        decision_time=signal_time,
        earliest_execution_time=execution_time,
        weights={"SZ000001": 0.6, "SH600000": 0.3},
        cash_weight=0.1,
        source_signal_id="alpha-1",
    )
    raw.require_universe(("SZ000001", "SH600000"))

    with pytest.raises(ValueError, match="outside the decision universe"):
        raw.require_universe(("SZ000001",))

    with pytest.raises(ValueError, match="sum to 1"):
        RawTargetPortfolio(
            decision_time=signal_time,
            earliest_execution_time=execution_time,
            weights={"SZ000001": 0.6},
            cash_weight=0.1,
            source_signal_id="bad-budget",
        )

    final = FinalTargetPortfolio.from_raw(raw, applied_risk_rules=("single_name_cap",))
    assert final.applied_risk_rules == ("single_name_cap",)
    assert final.weights == raw.weights


def test_open_execution_context_exposes_only_execution_safe_market_fields():
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    account = AccountSnapshot(cash=100_000.0, total_value=120_000.0)
    context = ExecutionContext(
        clock=execution_time,
        open_prices={"SZ000001": 10.25},
        tradability={"SZ000001": TradabilityState(can_buy=True, can_sell=True)},
        account=account,
        positions={},
    )

    assert context.price("SZ000001") == 10.25
    assert not hasattr(context, "close_prices")
    assert not hasattr(context, "high_prices")
    assert not hasattr(context, "low_prices")
    with pytest.raises(KeyError, match="No execution price"):
        context.price("SH600000")

    with pytest.raises(ValueError, match="OPEN"):
        ExecutionContext(
            clock=market_time(13, 15, SessionPhase.CLOSE),
            open_prices={"SZ000001": 10.25},
            tradability={"SZ000001": TradabilityState(can_buy=True, can_sell=True)},
            account=account,
            positions={},
        )


def test_portfolio_state_is_separate_from_market_observation():
    state = PortfolioState(
        as_of=market_time(10, 15, SessionPhase.AFTER_CLOSE),
        account=AccountSnapshot(cash=100_000.0, total_value=100_000.0),
        positions={},
    )
    assert state.account.cash == 100_000.0
    assert not hasattr(state, "features")
