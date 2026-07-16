"""Alpha model protocol."""

from typing import Any, Mapping, Protocol

from .types import AlphaSnapshot, MarketObservation


class AlphaModel(Protocol):
    def prepare(self, context: Any) -> None: ...

    def predict(self, observation: MarketObservation) -> AlphaSnapshot: ...

    def describe(self) -> Mapping[str, Any]: ...
