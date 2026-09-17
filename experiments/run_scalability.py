"""
Scalability experiment (Part 3.4).

Spawns `n` simulated ClientAgent instances as Ray actors (or, if Ray isn't
available/initialized, as plain threads -- so this script also runs on a
laptop without a Ray cluster for quick local checks). Each simulated client
joins a room and completes `rounds` FedAvg rounds against a *real* running
coordinator, and we record wall-clock round duration and per-client timing
reported by the coordinator's MLflow-tracked round summaries.

Usage:
    # local (no Ray cluster; falls back to threads):
    python experiments/run_scalability.py --url http://localhost:8000 \
        --room fashion-room --levels 1,2,4,8

    # against a real Ray/KubeRay cluster:
    RAY_ADDRESS=ray://fedroom-ray-head-svc:10001 \
        python experiments/run_scalability.py --levels 1,2,4,8,20,50,100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from client.agent import ClientAgent
from client.config import ClientConfig


def _run_one_client(client_id: str, url: str, room_id: str, n_clients: int, rounds: int, scheme: str) -> dict:
    cfg = ClientConfig(
        client_id=client_id,
        coordinator_url=url,
        room_id=room_id,
        n_clients=n_clients,
        partition_scheme=scheme,
        samples_per_client=200,
        local_epochs=1,
    )
    agent = ClientAgent(cfg)
    agent.join()
    t0 = time.time()
    results = agent.run_rounds(rounds)
    return {"client_id": client_id, "n_results": len(results), "wall_seconds": time.time() - t0}


def _try_ray():
    try:
        import ray

        if not ray.is_initialized():
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)
        return ray
    except Exception:
        return None


def run_level(n_clients: int, url: str, room_id: str, rounds: int, scheme: str) -> dict:
    client_ids = [f"sim-{n_clients}-{i}" for i in range(n_clients)]
    ray = _try_ray()
    t0 = time.time()

    if ray is not None:
        @ray.remote
        def _remote_run(cid):
            return _run_one_client(cid, url, room_id, n_clients, rounds, scheme)

        futures = [_remote_run.remote(cid) for cid in client_ids]
        results = ray.get(futures)
        engine = "ray"
    else:
        with ThreadPoolExecutor(max_workers=min(32, n_clients)) as pool:
            futures = [pool.submit(_run_one_client, cid, url, room_id, n_clients, rounds, scheme)
                       for cid in client_ids]
            results = [f.result() for f in futures]
        engine = "threads"

    total_wall = time.time() - t0
    status = requests.get(f"{url.rstrip('/')}/rooms/{room_id}", timeout=10).json()
    return {
        "n_clients": n_clients,
        "engine": engine,
        "total_wall_seconds": total_wall,
        "final_round": status["current_round"],
        "final_version": status["current_version"],
        "metrics_history": status["metrics_history"],
        "client_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("FEDROOM_COORDINATOR_URL", "http://localhost:8000"))
    parser.add_argument("--room", default=os.environ.get("FEDROOM_ROOM_ID", "fashion-room"))
    parser.add_argument("--levels", default=os.environ.get("SCALE_LEVELS", "1,2,4,8"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scheme", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--out", default="experiments/results/scalability.json")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    all_results = []
    for n in levels:
        print(f"=== scalability level: {n} clients ===")
        r = run_level(n, args.url, args.room, args.rounds, args.scheme)
        print(json.dumps({k: v for k, v in r.items() if k != "client_results"}, indent=2))
        all_results.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
