from __future__ import annotations

from typing import Sequence

from coordinator.aggregation import ClientUpdate, StateDict, weighted_average
from strategies.base import AggregationStrategy


class FedAvgStrategy(AggregationStrategy):
    """Weighted FedAvg (McMahan et al., AISTATS 2017):

        w_{t+1} = sum_k (n_k / sum_j n_j) * w_k
    """

    name = "fedavg"

    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        return weighted_average(updates)
