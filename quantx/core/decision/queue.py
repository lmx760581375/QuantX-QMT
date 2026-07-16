"""Deterministic queue for decision targets waiting for an execution event."""

from __future__ import annotations

from collections.abc import Iterator

from .clock import MarketTime
from .types import PendingRebalance


class PendingRebalanceQueue:
    """Keep one complete target per account/strategy/execution event."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str, MarketTime], PendingRebalance] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[PendingRebalance]:
        return iter(self._ordered_items())

    def put(self, pending: PendingRebalance) -> PendingRebalance | None:
        key = self._key(pending)
        replaced = self._items.get(key)
        if replaced is not None and pending.created_at < replaced.created_at:
            raise ValueError("Cannot replace a newer target with an older pending rebalance")
        conflicting_strategy = next(
            (
                item
                for item in self._items.values()
                if item.account_id == pending.account_id
                and item.execute_not_before == pending.execute_not_before
                and item.strategy_instance_id != pending.strategy_instance_id
            ),
            None,
        )
        if conflicting_strategy is not None:
            raise ValueError(
                "Cross-strategy targets must be combined by PortfolioAggregator before entering the pending queue"
            )
        self._items[key] = pending
        return replaced

    def pop_ready(
        self,
        clock: MarketTime,
        *,
        account_id: str | None = None,
        strategy_instance_id: str | None = None,
    ) -> tuple[PendingRebalance, ...]:
        ready: list[PendingRebalance] = []
        remove: list[tuple[str, str, MarketTime]] = []
        for key, pending in self._items.items():
            if account_id is not None and pending.account_id != account_id:
                continue
            if strategy_instance_id is not None and pending.strategy_instance_id != strategy_instance_id:
                continue
            if pending.expires_at is not None and clock > pending.expires_at:
                remove.append(key)
                continue
            if pending.execute_not_before <= clock:
                ready.append(pending)
                remove.append(key)
        for key in remove:
            del self._items[key]
        return tuple(sorted(ready, key=self._sort_key))

    def clear(self) -> None:
        self._items.clear()

    def _ordered_items(self) -> tuple[PendingRebalance, ...]:
        return tuple(sorted(self._items.values(), key=self._sort_key))

    @staticmethod
    def _key(pending: PendingRebalance) -> tuple[str, str, MarketTime]:
        return pending.account_id, pending.strategy_instance_id, pending.execute_not_before

    @staticmethod
    def _sort_key(pending: PendingRebalance) -> tuple[object, str, str, str]:
        return (
            pending.execute_not_before._sort_key(),
            pending.account_id,
            pending.strategy_instance_id,
            pending.rebalance_id,
        )
