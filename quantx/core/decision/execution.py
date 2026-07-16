"""Order planning protocol for strict execution events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .types import ExecutionContext, PendingRebalance

if TYPE_CHECKING:
    from quantx.core.strategy.base import OrderList


class OrderPlanner(Protocol):
    def create_orders(self, execution: ExecutionContext, pending: PendingRebalance) -> OrderList: ...
