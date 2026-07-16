"""Pending rebalance queue tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.queue import PendingRebalanceQueue
from quantx.core.decision.types import FinalTargetPortfolio, PendingRebalance


SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_time(day: int, hour: int, phase: SessionPhase) -> MarketTime:
    return MarketTime(
        session=f"2026-07-{day:02d}",
        phase=phase,
        timestamp=datetime(2026, 7, day, hour, tzinfo=SHANGHAI),
    )


def pending(
    rebalance_id: str,
    weight: float = 1.0,
    *,
    decision_day: int = 10,
    strategy_instance_id: str = "model-a",
    expires_at=None,
) -> PendingRebalance:
    decision_time = market_time(decision_day, 15, SessionPhase.AFTER_CLOSE)
    execution_time = market_time(13, 9, SessionPhase.OPEN)
    target = FinalTargetPortfolio(
        decision_time=decision_time,
        earliest_execution_time=execution_time,
        weights={"SZ000001": weight} if weight else {},
        cash_weight=1.0 - weight,
        source_signal_id=f"signal:{rebalance_id}",
    )
    return PendingRebalance.from_target(
        target,
        rebalance_id=rebalance_id,
        account_id="research",
        strategy_instance_id=strategy_instance_id,
        expires_at=expires_at,
    )


def test_pending_queue_does_not_release_before_execution_time():
    queue = PendingRebalanceQueue()
    queue.put(pending("rebalance-1"))

    assert queue.pop_ready(market_time(10, 15, SessionPhase.AFTER_CLOSE)) == ()
    assert len(queue) == 1

    ready = queue.pop_ready(market_time(13, 9, SessionPhase.OPEN))
    assert tuple(item.rebalance_id for item in ready) == ("rebalance-1",)
    assert len(queue) == 0


def test_pending_queue_replaces_same_strategy_target_for_same_execution_event():
    queue = PendingRebalanceQueue()
    queue.put(pending("old", weight=0.8))
    queue.put(pending("new", weight=0.5))

    assert len(queue) == 1
    ready = queue.pop_ready(market_time(13, 9, SessionPhase.OPEN))
    assert ready[0].rebalance_id == "new"
    assert ready[0].target.weights["SZ000001"] == 0.5


def test_pending_queue_rejects_an_older_target_after_a_newer_target():
    queue = PendingRebalanceQueue()
    queue.put(pending("new", decision_day=12))

    with pytest.raises(ValueError, match="older pending"):
        queue.put(pending("old", decision_day=10))


def test_pending_queue_requires_cross_strategy_targets_to_be_aggregated_first():
    queue = PendingRebalanceQueue()
    queue.put(pending("model-a"))

    with pytest.raises(ValueError, match="PortfolioAggregator"):
        queue.put(pending("model-b", strategy_instance_id="model-b"))


def test_pending_queue_discards_expired_rebalances():
    queue = PendingRebalanceQueue()
    queue.put(pending("expired", expires_at=market_time(13, 9, SessionPhase.OPEN)))

    assert queue.pop_ready(market_time(13, 15, SessionPhase.CLOSE)) == ()
    assert len(queue) == 0
