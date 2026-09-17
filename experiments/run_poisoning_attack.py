"""
Poisoning-attack experiment (bonus: "Robust aggregation against malicious
or poisoned clients").

Setup: `n_honest` honest clients each submit a small, bounded perturbation
of the current global model (representing a real local-training delta).
`n_malicious` clients instead submit an adversarial update scaled to a
large magnitude in a fixed, attacker-chosen direction -- a standard
"scaled/negated gradient" model-poisoning attack in the style of Blanchard
et al. (NeurIPS 2017) and the AGR-tailored attacks in Shejwalkar &
Houmansadr (NDSS 2021): the attacker does not need any special knowledge of
the aggregation rule to mount this version, only the ability to submit an
arbitrarily-scaled update.

We run the SAME attack against two rooms that differ only in
`aggregation.strategy`:

    - "fedavg"      (no robustness -- the assignment's baseline)
    - "multi_krum"  (bonus robust aggregation, byzantine_f = n_malicious)

and report, per round, how far the resulting global model moved from the
"honest-only" reference average -- i.e. how much damage the attacker did to
the model the honest clients were trying to train.

Usage:
    uvicorn coordinator.app:app --host 127.0.0.1 --port 8000 &
    python experiments/run_poisoning_attack.py --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import requests

from coordinator.codec import state_to_b64, b64_to_state


@dataclass
class RunResult:
    strategy: str
    per_round_damage: list  # ||global - honest_reference|| per round
    per_round_honest_ref_norm: list


def create_room(url: str, room_id: str, strategy: str, n_selected: int, byzantine_f: int,
                 dim: int = 64, quorum: float = 1.0, timeout: float = 30.0) -> dict:
    initial_state = {"w": np.zeros(dim, dtype="float64")}
    payload = {
        "room_id": room_id,
        "model_contract": {"model_id": "poison-demo", "framework": "numpy",
                            "param_shapes": {"w": [dim]}, "max_update_norm": None},
        "preprocessing_contract": "none",
        "aggregation": {
            "strategy": strategy, "min_available_clients": n_selected, "min_fit_clients": n_selected,
            "quorum": quorum, "round_timeout_seconds": timeout, "byzantine_f": byzantine_f,
        },
        "target_rounds": 100,
        "initial_state_b64": state_to_b64(initial_state),
    }
    r = requests.post(f"{url}/rooms", json=payload, timeout=30)
    r.raise_for_status()
    return initial_state


def run_condition(url: str, room_id: str, strategy: str, n_honest: int, n_malicious: int,
                   n_rounds: int, dim: int, seed: int, attack_scale: float) -> RunResult:
    n_selected = n_honest + n_malicious
    create_room(url, room_id, strategy, n_selected, byzantine_f=n_malicious, dim=dim)

    honest_ids = [f"honest-{i}" for i in range(n_honest)]
    malicious_ids = [f"malicious-{i}" for i in range(n_malicious)]
    for cid in honest_ids + malicious_ids:
        requests.post(f"{url}/rooms/{room_id}/join", json={"client_id": cid}, timeout=10)
    requests.post(f"{url}/rooms/{room_id}/start", json={"rounds": n_rounds}, timeout=10)

    rng = np.random.default_rng(seed)
    # Fixed, attacker-chosen malicious direction (does not depend on the
    # honest clients' updates -- an "agnostic" adversary in the taxonomy of
    # Shejwalkar & Houmansadr).
    attack_direction = rng.normal(size=dim)
    attack_direction /= np.linalg.norm(attack_direction)

    damages, honest_ref_norms = [], []

    for round_idx in range(n_rounds):
        status = requests.get(f"{url}/rooms/{room_id}", timeout=10).json()
        model_resp = requests.get(f"{url}/rooms/{room_id}/model", timeout=10).json()
        base_version = model_resp["version"]
        global_state = b64_to_state(model_resp["state_b64"])
        base_w = global_state["w"]

        honest_updates = []
        for cid in honest_ids:
            delta = rng.normal(scale=0.05, size=dim)  # small, bounded "local training" step
            update = {"w": base_w + delta}
            honest_updates.append(update["w"])
            requests.post(f"{url}/rooms/{room_id}/updates", json={
                "client_id": cid, "base_version": base_version, "n_samples": 100,
                "state_b64": state_to_b64(update), "metrics": {},
            }, timeout=10)

        for cid in malicious_ids:
            update = {"w": base_w + attack_scale * attack_direction}
            requests.post(f"{url}/rooms/{room_id}/updates", json={
                "client_id": cid, "base_version": base_version, "n_samples": 100,
                "state_b64": state_to_b64(update), "metrics": {},
            }, timeout=10)

        # Wait for the round to finalize (all selected clients have responded).
        for _ in range(20):
            time.sleep(0.15)
            requests.post(f"{url}/rooms/{room_id}/tick", timeout=10)
            s = requests.get(f"{url}/rooms/{room_id}", timeout=10).json()
            if s["current_version"] > base_version:
                break

        s = requests.get(f"{url}/rooms/{room_id}", timeout=10).json()
        ck = requests.get(f"{url}/rooms/{room_id}/checkpoints/{s['current_version']}", timeout=10).json()
        new_w = b64_to_state(ck["state_b64"])["w"]

        honest_reference = np.mean(honest_updates, axis=0)  # what FedAvg would give with NO attacker
        damage = float(np.linalg.norm(new_w - honest_reference))
        damages.append(damage)
        honest_ref_norms.append(float(np.linalg.norm(honest_reference)))

        requests.post(f"{url}/rooms/{room_id}/next-round", timeout=10)

    return RunResult(strategy=strategy, per_round_damage=damages, per_round_honest_ref_norm=honest_ref_norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-honest", type=int, default=6)
    parser.add_argument("--n-malicious", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--attack-scale", type=float, default=50.0)
    parser.add_argument("--out", default="experiments/results/poisoning.json")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    stamp = int(time.time())
    conditions = {
        "fedavg (no robustness)": ("fedavg", f"poison-fedavg-{stamp}"),
        "multi_krum (robust, byzantine_f=n_malicious)": ("multi_krum", f"poison-multikrum-{stamp}"),
    }

    results = {}
    for label, (strategy, room_id) in conditions.items():
        print(f"=== {label} ===")
        res = run_condition(url, room_id, strategy, args.n_honest, args.n_malicious,
                             args.rounds, args.dim, seed=0, attack_scale=args.attack_scale)
        results[label] = {
            "strategy": res.strategy,
            "per_round_damage": res.per_round_damage,
            "per_round_honest_ref_norm": res.per_round_honest_ref_norm,
            "mean_damage": float(np.mean(res.per_round_damage)),
        }
        print(f"  mean ||global - honest_reference|| over {args.rounds} rounds: "
              f"{results[label]['mean_damage']:.4f}")

    fedavg_damage = results["fedavg (no robustness)"]["mean_damage"]
    robust_damage = results["multi_krum (robust, byzantine_f=n_malicious)"]["mean_damage"]
    reduction_pct = 100.0 * (1 - robust_damage / fedavg_damage) if fedavg_damage > 0 else float("nan")

    summary = {
        "config": {
            "n_honest": args.n_honest, "n_malicious": args.n_malicious,
            "rounds": args.rounds, "dim": args.dim, "attack_scale": args.attack_scale,
        },
        "results": results,
        "damage_reduction_pct_multikrum_vs_fedavg": reduction_pct,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"plain FedAvg mean damage:    {fedavg_damage:.4f}")
    print(f"Multi-Krum mean damage:      {robust_damage:.4f}")
    print(f"Damage reduction:            {reduction_pct:.1f}%")
    print(f"Wrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = "experiments/results/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure()
        for label, res in results.items():
            rounds = list(range(1, len(res["per_round_damage"]) + 1))
            plt.plot(rounds, res["per_round_damage"], marker="o", label=label)
        plt.xlabel("Round")
        plt.ylabel("||global model - honest-only reference||  (attack damage)")
        plt.title(f"Model poisoning: {args.n_malicious}/{args.n_honest + args.n_malicious} "
                  f"malicious clients, attack_scale={args.attack_scale}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(plot_dir, "poisoning_attack_damage.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_path}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
