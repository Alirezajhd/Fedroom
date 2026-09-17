"""
Weighted FedAvg aggregation and update/contract validation.

This module is deliberately dependency-light (numpy + stdlib only) so it can
be exercised by fast, deterministic unit tests without spinning up the
coordinator service, a database, MinIO, or MLflow.

Math (per the FLaaS spec):

    w_{t+1} = sum_k ( n_k / sum_j n_j ) * w_k

for the set of *valid* client updates w_k collected in round t, weighted by
each client's reported local sample count n_k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

Tensor = np.ndarray
StateDict = Dict[str, Tensor]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ContractError(ValueError):
    """The update does not match the room's model contract."""


class NonFiniteUpdateError(ValueError):
    """The update contains NaN or +/-inf values."""


class OversizedUpdateError(ValueError):
    """The update exceeds the configured L2-norm / payload limit."""


class StaleUpdateError(ValueError):
    """The update was trained against an older-than-expected model version."""


class EmptyAggregationError(ValueError):
    """There is nothing valid to aggregate."""


# --------------------------------------------------------------------------- #
# Model contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelContract:
    """Defines the exact tensor names/shapes/dtype a valid update must have."""

    model_id: str
    framework: str
    param_shapes: Dict[str, Sequence[int]]
    max_update_norm: float | None = None  # None = unlimited

    def validate_shapes(self, state: StateDict) -> None:
        expected_keys = set(self.param_shapes)
        got_keys = set(state.keys())
        if expected_keys != got_keys:
            missing = expected_keys - got_keys
            extra = got_keys - expected_keys
            raise ContractError(
                f"Parameter set mismatch for contract '{self.model_id}'. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        for name, expected_shape in self.param_shapes.items():
            got_shape = tuple(state[name].shape)
            if tuple(expected_shape) != got_shape:
                raise ContractError(
                    f"Shape mismatch for '{name}': expected {tuple(expected_shape)}, "
                    f"got {got_shape}"
                )


def validate_finite(state: StateDict) -> None:
    for name, arr in state.items():
        if not np.all(np.isfinite(arr)):
            raise NonFiniteUpdateError(
                f"Tensor '{name}' contains NaN or infinite values"
            )


def compute_update_norm(state: StateDict) -> float:
    total = 0.0
    for arr in state.values():
        total += float(np.sum(np.square(arr.astype(np.float64))))
    return math.sqrt(total)


def validate_norm(state: StateDict, max_norm: float | None) -> float:
    norm = compute_update_norm(state)
    if max_norm is not None and norm > max_norm:
        raise OversizedUpdateError(
            f"Update L2 norm {norm:.4f} exceeds configured limit {max_norm:.4f}"
        )
    return norm


def validate_update(
    state: StateDict,
    contract: ModelContract,
) -> float:
    """Run the full validation pipeline. Returns the update's L2 norm.

    Raises ContractError / NonFiniteUpdateError / OversizedUpdateError.
    """
    contract.validate_shapes(state)
    validate_finite(state)
    return validate_norm(state, contract.max_update_norm)


# --------------------------------------------------------------------------- #
# Weighted FedAvg
# --------------------------------------------------------------------------- #
@dataclass
class ClientUpdate:
    client_id: str
    n_samples: int
    state: StateDict
    base_version: int


def weighted_average(updates: Sequence[ClientUpdate]) -> StateDict:
    """Combine validated client updates via weighted FedAvg.

    w_{t+1} = sum_k (n_k / sum_j n_j) * w_k
    """
    if not updates:
        raise EmptyAggregationError("No valid updates to aggregate")

    total_samples = sum(u.n_samples for u in updates)
    if total_samples <= 0:
        raise EmptyAggregationError("Total reported sample count is <= 0")

    # All updates are assumed to already share the same key set/shape (the
    # caller is expected to have called validate_update() on each first).
    param_names = updates[0].state.keys()
    aggregated: StateDict = {}
    for name in param_names:
        acc = np.zeros_like(updates[0].state[name], dtype=np.float64)
        for u in updates:
            weight = u.n_samples / total_samples
            acc += weight * u.state[name].astype(np.float64)
        aggregated[name] = acc.astype(updates[0].state[name].dtype)
    return aggregated


def flatten_state(state: StateDict) -> np.ndarray:
    """Utility used by tests/metrics: concatenate all tensors into one vector."""
    return np.concatenate([np.ravel(state[k]) for k in sorted(state.keys())])
