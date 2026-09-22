"""
Scalability experiment (Part 3.4).

Spawns `n` simulated ClientAgent instances as Ray actors (or, if Ray isn't
available/initialized, as plain threads -- so this script also runs on a
laptop without a Ray cluster for quick local checks). Each simulated client
joins a room and completes `rounds` FedAvg rounds against a *real* running
coordinator, and we record wall-clock round duration and per-client timing
reported by the coordinator's MLflow-tracked round summaries.

IMPORTANT: the coordinator never auto-advances rounds on its own -- the
background finalizer only *aggregates* a round once it's ready; starting the
very first round and advancing to every subsequent one requires an explicit
`/rooms/{room}/start` / `/rooms/{room}/next-round` call (exactly like
`tui.cli train start` / `train advance`). This script owns that lifecycle
itself (creating one fresh, disposable room per client-count level) instead
of assuming some other process is already driving an existing room -- reusing
a shared room like `fashion-room` here would either race with whatever else
is using it, or simply never progress if nothing else is calling
start/next-round.

Usage:
    # local (no Ray cluster; falls back to threads):
    python experiments/run_scalability.py --url http://localhost:8000 \
        --levels 1,2,4,8

    # against a real Ray/KubeRay cluster:
    RAY_ADDRESS=ray://fedroom-ray-head-svc:10001 \
        python experiments/run_scalability.py --levels 1,2,4,8,20,50,100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

# Allow running this script directly (`python experiments/run_scalability.py`)
# without the project root already being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.agent import ClientAgent
from client.config import ClientConfig
from client.model import build_model, model_contract_shapes, torch_state_to_numpy
from coordinator.codec import state_to_b64


def _create_room(
    url: str, room_id: str, n_clients: int, round_timeout_seconds: float
) -> None:
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
            "max_update_norm": None,
        },
        "preprocessing_contract": "fmnist-normalize-v1",
        "aggregation": {
            "strategy": "fedavg",
            "min_available_clients": n_clients,
            "min_fit_clients": n_clients,
            "quorum": 0.6,
            "round_timeout_seconds": round_timeout_seconds,
        },
        "target_rounds": 10_000,  # this script drives rounds explicitly via --rounds
        "initial_state_b64": state_to_b64(initial_state),
    }
    resp = requests.post(f"{url.rstrip('/')}/rooms", json=payload, timeout=30)
    resp.raise_for_status()


def _join_client(
    client_id: str, url: str, room_id: str, n_clients: int, scheme: str
) -> ClientConfig:
    cfg = ClientConfig(
        client_id=client_id,
        coordinator_url=url,
        room_id=room_id,
        n_clients=n_clients,
        partition_scheme=scheme,
        samples_per_client=200,
        local_epochs=1,
    )
    ClientAgent(cfg).join()
    return cfg


def _train_once(cfg: ClientConfig, max_wait_seconds: float) -> dict | None:
    return ClientAgent(cfg).train_once(max_wait_seconds=max_wait_seconds)


def _try_ray():
    try:
        import ray

        if not ray.is_initialized():
            ray.init(
                address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True
            )
        return ray
    except Exception:
        return None


def _map(ray, fn, items, max_workers):
    """Run `fn` over `items` concurrently, via Ray if available, else threads."""
    if ray is not None:
        remote_fn = ray.remote(fn)
        return ray.get([remote_fn.remote(*item) for item in items]), "ray"
    with ThreadPoolExecutor(max_workers=max(1, min(32, max_workers))) as pool:
        futures = [pool.submit(fn, *item) for item in items]
        return [f.result() for f in futures], "threads"


def run_level(
    n_clients: int, url: str, rounds: int, scheme: str, round_timeout_seconds: float
) -> dict:
    room_id = f"scale-{n_clients}-{int(time.time())}"
    client_ids = [f"sim-{n_clients}-{i}" for i in range(n_clients)]
    ray = _try_ray()

    _create_room(url, room_id, n_clients, round_timeout_seconds)

    t0 = time.time()

    # 1. All clients must join BEFORE the round starts -- a client joining
    # mid-round only becomes eligible for the *next* round, and `/start`
    # requires `min_available_clients` already-eligible clients to succeed.
    cfgs, engine = _map(
        ray,
        _join_client,
        [(cid, url, room_id, n_clients, scheme) for cid in client_ids],
        n_clients,
    )

    resp = requests.post(
        f"{url.rstrip('/')}/rooms/{room_id}/start", json={"rounds": rounds}, timeout=10
    )
    resp.raise_for_status()

    per_client_results = {cid: [] for cid in client_ids}
    round_wall_seconds = []
    for r in range(rounds):
        r_t0 = time.time()
        results, _ = _map(
            ray,
            _train_once,
            [(cfg, round_timeout_seconds + 30.0) for cfg in cfgs],
            n_clients,
        )
        round_wall_seconds.append(time.time() - r_t0)
        for cid, res in zip(client_ids, results):
            if res is not None:
                per_client_results[cid].append(res)

        # give the background finalizer a moment, then explicitly advance
        time.sleep(2.5)
        requests.post(f"{url.rstrip('/')}/rooms/{room_id}/next-round", timeout=10)

    total_wall = time.time() - t0
    time.sleep(1.0)
    status = requests.get(f"{url.rstrip('/')}/rooms/{room_id}", timeout=10).json()
    return {
        "n_clients": n_clients,
        "engine": engine,
        "total_wall_seconds": total_wall,
        "round_wall_seconds": round_wall_seconds,
        "final_round": status["current_round"],
        "final_version": status["current_version"],
        "metrics_history": status["metrics_history"],
        "client_results": {
            cid: {"n_results": len(res)} for cid, res in per_client_results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("FEDROOM_COORDINATOR_URL", "http://localhost:8000"),
    )
    parser.add_argument("--levels", default=os.environ.get("SCALE_LEVELS", "1,2,4,8"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scheme", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--round-timeout-seconds", type=float, default=500.0)
    parser.add_argument("--out", default="experiments/results/scalability.json")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    all_results = []
    for n in levels:
        print(f"=== scalability level: {n} clients ===")
        r = run_level(n, args.url, args.rounds, args.scheme, args.round_timeout_seconds)
        print(
            json.dumps({k: v for k, v in r.items() if k != "client_results"}, indent=2)
        )
        all_results.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
