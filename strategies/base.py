from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from coordinator.aggregation import ClientUpdate, StateDict


class AggregationStrategy(ABC):
    """Pluggable server-side aggregation rule.

    Implementations receive only *already validated* updates (contract
    shapes checked, finite values, within norm limits, non-stale). They are
    responsible purely for combining them into the next global model.
    """

    name: str = "base"

    @abstractmethod
    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        ...
