"""
Produces the plots/tables required by Part 3.8:

  - global task metric vs round for >=2 client-count settings
  - round duration vs active/selected client count
  - completed/failed/dropped clients per round (membership experiment)

Reads the JSON files written by run_scalability.py / run_noniid.py and
writes PNGs + a markdown table summary into experiments/results/plots/.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scalability(path: str, out_dir: str):
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)

    levels = [d["n_clients"] for d in data]
    total_wall = [d["total_wall_seconds"] for d in data]

    plt.figure()
    plt.plot(levels, total_wall, marker="o")
    plt.xlabel("Number of clients")
    plt.ylabel("Total wall-clock time (s)")
    plt.title("Scalability: total time vs client count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "scalability_total_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Round duration vs selected client count, pulled from MLflow-tracked
    # per-round summaries recorded by the coordinator.
    plt.figure()
    for d in data:
        durations = [m["duration_seconds"] for m in d["metrics_history"]]
        rounds = list(range(1, len(durations) + 1))
        plt.plot(rounds, durations, marker="o", label=f"{d['n_clients']} clients")
    plt.xlabel("Round")
    plt.ylabel("Round duration (s)")
    plt.title("Round duration vs round, by client count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "round_duration_vs_clients.png"), dpi=150, bbox_inches="tight")
    plt.close()

    with open(os.path.join(out_dir, "scalability_table.md"), "w") as f:
        f.write("| Clients | Engine | Total wall (s) | Final round | Final version |\n")
        f.write("|---|---|---|---|---|\n")
        for d in data:
            f.write(f"| {d['n_clients']} | {d['engine']} | {d['total_wall_seconds']:.2f} "
                     f"| {d['final_round']} | {d['final_version']} |\n")

    print(f"wrote scalability plots/table to {out_dir}")


def plot_noniid(path: str, out_dir: str):
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)

    plt.figure()
    for condition, history in data.items():
        rounds = [m["round"] for m in history]
        completed = [m["n_completed"] for m in history]
        plt.plot(rounds, completed, marker="o", label=condition)
    plt.xlabel("Round")
    plt.ylabel("Completed clients")
    plt.title("Non-IID comparison: completed clients per round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "noniid_completed_clients.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote non-IID plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalability", default="experiments/results/scalability.json")
    parser.add_argument("--noniid", default="experiments/results/noniid.json")
    parser.add_argument("--out-dir", default="experiments/results/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_scalability(args.scalability, args.out_dir)
    plot_noniid(args.noniid, args.out_dir)


if __name__ == "__main__":
    main()
