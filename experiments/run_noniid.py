"""
Non-IID learning experiment (Part 3.3).

Runs the same room twice with different `partition_scheme` (iid vs
non_iid) and, optionally, a third run using a different aggregation
strategy (e.g. multi_krum), recording the round-by-round metrics history
the coordinator already tracks via MLflow so results are directly
comparable to the required "global task metric vs round" plot.

This script assumes a coordinator is already running and creates a fresh
room per configuration (so runs don't interfere with each other).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import requests

from client.agent import ClientAgent
from client.config import ClientConfig
from client.model import build_model, model_contract_shapes, torch_state_to_numpy
from coordinator.codec import state_to_b64


def create_room(url: str, room_id: str, strategy: str, n_clients: int, rounds: int, byzantine_f: int = 0):
    try:
        model = build_model()
        param_shapes = model_contract_shapes(model)
        initial_state = torch_state_to_numpy(model)
    except ImportError:
        initial_state = {"w": np.zeros((4,), dtype="float32")}
        param_shapes = {"w": [4]}

    payload = {
        "room_id": room_id,
        "model_contract": {"model_id": "fashion-cnn-v1", "framework": "pytorch",
                            "param_shapes": param_shapes, "max_update_norm": None},
        "preprocessing_contract": "fmnist-normalize-v1",
        "aggregation": {
            "strategy": strategy, "min_available_clients": n_clients, "min_fit_clients": n_clients,
            "quorum": 0.6, "round_timeout_seconds": 60, "byzantine_f": byzantine_f,
        },
        "target_rounds": rounds,
        "initial_state_b64": state_to_b64(initial_state),
    }
    resp = requests.post(f"{url.rstrip('/')}/rooms", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_condition(url: str, room_id: str, strategy: str, scheme: str, n_clients: int, rounds: int):
    create_room(url, room_id, strategy, n_clients, rounds)
    requests.post(f"{url.rstrip('/')}/rooms/{room_id}/start", json={"rounds": rounds}, timeout=10)

    agents = []
    for i in range(n_clients):
        cfg = ClientConfig(
            client_id=f"{room_id}-c{i}", coordinator_url=url, room_id=room_id,
            n_clients=n_clients, partition_scheme=scheme, samples_per_client=300,
        )
        agent = ClientAgent(cfg)
        agent.join()
        agents.append(agent)

    for r in range(rounds):
        for agent in agents:
            agent.train_once()
        # give the background finalizer a moment, then explicitly advance
        time.sleep(2.5)
        requests.post(f"{url.rstrip('/')}/rooms/{room_id}/next-round", timeout=10)

    time.sleep(2.5)
    status = requests.get(f"{url.rstrip('/')}/rooms/{room_id}", timeout=10).json()
    return status["metrics_history"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n-clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default="experiments/results/noniid.json")
    args = parser.parse_args()

    conditions = {
        "iid_fedavg": dict(strategy="fedavg", scheme="iid"),
        "noniid_fedavg": dict(strategy="fedavg", scheme="non_iid"),
        "noniid_multikrum": dict(strategy="multi_krum", scheme="non_iid"),
    }
    results = {}
    for name, cond in conditions.items():
        room_id = f"exp-{name}-{int(time.time())}"
        print(f"=== {name} (room={room_id}) ===")
        results[name] = run_condition(args.url, room_id, cond["strategy"], cond["scheme"],
                                       args.n_clients, args.rounds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
