"""
Byzantine-robust aggregation strategies (bonus scope: "Robust aggregation
against malicious or poisoned clients").

Implementations follow the aggregation rules described in:

  - Yin, Chen, Ramchandran & Bartlett, "Byzantine-Robust Distributed
    Learning: Towards Optimal Statistical Rates" (ICML 2018) - coordinate-
    wise Trimmed-mean and Median.
  - Blanchard, El Mhamdi, Guerraoui & Stainer, "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017) - Krum
    and Multi-Krum.

These strategies deliberately ignore each client's reported `n_samples`
weight, matching the literature (weighting by a self-reported sample count
would itself be an easy attack surface for a malicious client trying to
dominate the aggregate). They instead rely on a configured Byzantine
tolerance `f`: the assumed maximum number of malicious/faulty clients among
the *selected* set for a round. Fedroom exposes this as
`aggregation.byzantine_f` in the room config.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from coordinator.aggregation import ClientUpdate, EmptyAggregationError, StateDict
from strategies.base import AggregationStrategy


def _stack_param(updates: Sequence[ClientUpdate], name: str) -> np.ndarray:
    return np.stack([u.state[name].astype(np.float64) for u in updates], axis=0)


def _flatten_client(update: ClientUpdate) -> np.ndarray:
    return np.concatenate([np.ravel(update.state[k]) for k in sorted(update.state.keys())])


class TrimmedMeanStrategy(AggregationStrategy):
    """Coordinate-wise trimmed mean: drop the `beta` largest and `beta`
    smallest values per coordinate across clients, then average the rest.
    """

    name = "trimmed_mean"

    def __init__(self, beta: int = 0):
        self.beta = beta

    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        if not updates:
            raise EmptyAggregationError("No valid updates to aggregate")
        k = len(updates)
        beta = min(self.beta, (k - 1) // 2) if k > 1 else 0
        result: StateDict = {}
        for name in updates[0].state.keys():
            stacked = _stack_param(updates, name)  # (K, *shape)
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[beta: k - beta] if beta > 0 else sorted_vals
            result[name] = np.mean(trimmed, axis=0).astype(updates[0].state[name].dtype)
        return result


class MedianStrategy(AggregationStrategy):
    """Coordinate-wise median across client updates."""

    name = "median"

    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        if not updates:
            raise EmptyAggregationError("No valid updates to aggregate")
        result: StateDict = {}
        for name in updates[0].state.keys():
            stacked = _stack_param(updates, name)
            result[name] = np.median(stacked, axis=0).astype(updates[0].state[name].dtype)
        return result


class KrumStrategy(AggregationStrategy):
    """Krum (single-vector) or Multi-Krum (average of the m best-scoring
    updates), selecting the update(s) closest to their (k - f - 2) nearest
    neighbors in flattened parameter space -- i.e. farthest from being an
    outlier, which is what a malicious client's poisoned update typically
    looks like.
    """

    name = "krum"

    def __init__(self, byzantine_f: int = 0, multi: bool = True):
        self.f = byzantine_f
        self.multi = multi

    def _scores(self, updates: Sequence[ClientUpdate]) -> List[float]:
        k = len(updates)
        flat = [_flatten_client(u) for u in updates]
        dists = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    dists[i, j] = float(np.sum((flat[i] - flat[j]) ** 2))
        n_neighbors = max(1, k - self.f - 2)
        scores = []
        for i in range(k):
            nearest = np.sort(dists[i])[1:n_neighbors + 1]  # exclude self (distance 0)
            scores.append(float(np.sum(nearest)))
        return scores

    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        if not updates:
            raise EmptyAggregationError("No valid updates to aggregate")
        if len(updates) <= 2 * self.f + 2:
            # Not enough clients to provide the Krum robustness guarantee;
            # fall back to plain averaging rather than failing the round.
            k = len(updates)
            result: StateDict = {}
            for name in updates[0].state.keys():
                result[name] = np.mean(_stack_param(updates, name), axis=0).astype(
                    updates[0].state[name].dtype
                )
            return result

        scores = self._scores(updates)
        order = np.argsort(scores)
        if not self.multi:
            best = updates[order[0]]
            return {k: v.copy() for k, v in best.state.items()}

        m = max(1, len(updates) - self.f)
        chosen = [updates[i] for i in order[:m]]
        result: StateDict = {}
        for name in chosen[0].state.keys():
            result[name] = np.mean(_stack_param(chosen, name), axis=0).astype(
                chosen[0].state[name].dtype
            )
        return result


def get_strategy(name: str, byzantine_f: int = 0) -> AggregationStrategy:
    """Strategy registry used by the coordinator to pick an aggregation rule
    from the room's `aggregation.strategy` config field.
    """
    from strategies.fedavg import FedAvgStrategy

    registry: Dict[str, AggregationStrategy] = {
        "fedavg": FedAvgStrategy(),
        "trimmed_mean": TrimmedMeanStrategy(beta=byzantine_f),
        "median": MedianStrategy(),
        "krum": KrumStrategy(byzantine_f=byzantine_f, multi=False),
        "multi_krum": KrumStrategy(byzantine_f=byzantine_f, multi=True),
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregation strategy '{name}'. Options: {sorted(registry)}")
    return registry[name]
