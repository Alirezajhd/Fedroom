"""
Failure-injection experiment (Part 3.5).

Exercises, against a live coordinator:
  1. A client timeout / dropout (client joins, is selected, never submits).
  2. A stale model-version submission.
  3. A non-finite (NaN) update.
  4. An update exceeding the configured norm limit.

Each scenario creates its own room so failures are isolated and the
evidence is unambiguous. Prints a pass/fail summary suitable for pasting
into the technical report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Allow running this script directly (`python experiments/inject_failures.py`)
# without the project root already being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.model import build_model, model_contract_shapes, torch_state_to_numpy
from coordinator.codec import array_to_b64, state_to_b64


def _create_room(url, room_id, quorum=0.5, timeout=8, max_norm=None, n=2):
    try:
        model = build_model()
        param_shapes = model_contract_shapes(model)
        initial_state = torch_state_to_numpy(model)
    except ImportError:
        initial_state = {"w": np.zeros((4,), dtype="float32")}
        param_shapes = {"w": [4]}
    payload = {
        "room_id": room_id,
        "model_contract": {
            "model_id": "fashion-cnn-v1",
            "framework": "pytorch",
            "param_shapes": param_shapes,
            "max_update_norm": max_norm,
        },
        "preprocessing_contract": "fmnist-normalize-v1",
        "aggregation": {
            "strategy": "fedavg",
            "min_available_clients": n,
            "min_fit_clients": n,
            "quorum": quorum,
            "round_timeout_seconds": timeout,
        },
        "target_rounds": 1,
        "initial_state_b64": state_to_b64(initial_state),
    }
    resp = requests.post(f"{url}/rooms", json=payload, timeout=30)
    resp.raise_for_status()
    return initial_state


def scenario_timeout_dropout(url: str):
    room_id = f"fail-timeout-{int(time.time())}"
    state = _create_room(url, room_id, quorum=0.5, timeout=3, n=2)
    for cid in ["good-client", "straggler-client"]:
        requests.post(
            f"{url}/rooms/{room_id}/join", json={"client_id": cid}, timeout=10
        )
    requests.post(f"{url}/rooms/{room_id}/start", json={"rounds": 1}, timeout=10)

    # Only the good client submits; the straggler never responds.
    requests.post(
        f"{url}/rooms/{room_id}/updates",
        json={
            "client_id": "good-client",
            "base_version": 0,
            "n_samples": 10,
            "state_b64": state_to_b64(state),
            "metrics": {},
        },
        timeout=10,
    )
    time.sleep(5)  # exceed round_timeout_seconds
    requests.post(f"{url}/rooms/{room_id}/tick", timeout=10)
    status = requests.get(f"{url}/rooms/{room_id}", timeout=10).json()
    dropped = status["clients"]["straggler-client"]["status"] == "dropped"
    aggregated = status["current_version"] == 1
    print(
        f"[timeout/dropout] straggler marked dropped={dropped}, round aggregated={aggregated}"
    )
    return dropped and aggregated


def scenario_stale_update(url: str):
    room_id = f"fail-stale-{int(time.time())}"
    state = _create_room(url, room_id, quorum=1.0, timeout=30, n=1)
    requests.post(f"{url}/rooms/{room_id}/join", json={"client_id": "c1"}, timeout=10)
    requests.post(f"{url}/rooms/{room_id}/start", json={"rounds": 1}, timeout=10)
    resp = requests.post(
        f"{url}/rooms/{room_id}/updates",
        json={
            "client_id": "c1",
            "base_version": 99,
            "n_samples": 10,
            "state_b64": state_to_b64(state),
            "metrics": {},
        },
        timeout=10,
    )
    rejected = resp.status_code == 422 and "version" in resp.json().get("reason", "")
    print(
        f"[stale update] rejected={rejected} (http {resp.status_code}: {resp.json()})"
    )
    return rejected


def scenario_nan_update(url: str):
    room_id = f"fail-nan-{int(time.time())}"
    state = _create_room(url, room_id, quorum=1.0, timeout=30, n=1)
    requests.post(f"{url}/rooms/{room_id}/join", json={"client_id": "c1"}, timeout=10)
    requests.post(f"{url}/rooms/{room_id}/start", json={"rounds": 1}, timeout=10)

    poisoned = {k: np.full_like(v, np.nan) for k, v in state.items()}
    resp = requests.post(
        f"{url}/rooms/{room_id}/updates",
        json={
            "client_id": "c1",
            "base_version": 0,
            "n_samples": 10,
            "state_b64": state_to_b64(poisoned),
            "metrics": {},
        },
        timeout=10,
    )
    rejected = resp.status_code == 422
    print(f"[NaN update] rejected={rejected} (http {resp.status_code}: {resp.json()})")
    return rejected


def scenario_oversized_update(url: str):
    room_id = f"fail-oversized-{int(time.time())}"
    state = _create_room(url, room_id, quorum=1.0, timeout=30, max_norm=1.0, n=1)
    requests.post(f"{url}/rooms/{room_id}/join", json={"client_id": "c1"}, timeout=10)
    requests.post(f"{url}/rooms/{room_id}/start", json={"rounds": 1}, timeout=10)

    huge = {k: v + 1000.0 for k, v in state.items()}
    resp = requests.post(
        f"{url}/rooms/{room_id}/updates",
        json={
            "client_id": "c1",
            "base_version": 0,
            "n_samples": 10,
            "state_b64": state_to_b64(huge),
            "metrics": {},
        },
        timeout=10,
    )
    rejected = resp.status_code == 422
    print(
        f"[oversized update] rejected={rejected} (http {resp.status_code}: {resp.json()})"
    )
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    results = {
        "timeout_dropout": scenario_timeout_dropout(url),
        "stale_update": scenario_stale_update(url),
        "nan_update": scenario_nan_update(url),
        "oversized_update": scenario_oversized_update(url),
    }
    print("\n=== Failure-injection summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
