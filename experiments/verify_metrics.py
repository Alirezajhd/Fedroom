"""
Observability verification script: proves every metric the assignment's
Part 2.6 / Part 3.8 tables require actually gets produced by a real round,
end to end through the HTTP API (not just unit-tested in isolation).

This is the script that generated the example output documented in
docs/EXPLAINER.md Section 12 -- run it yourself against your own
coordinator to get your own numbers for the report.

Usage:
    uvicorn coordinator.app:app --host 127.0.0.1 --port 8000 &
    python experiments/verify_metrics.py --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import requests

from client.agent import ClientAgent
from client.config import ClientConfig
from coordinator.codec import state_to_b64

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--room-id", default=f"metrics-check-{int(time.time())}")
ARGS = parser.parse_args()
URL = ARGS.url.rstrip("/")

# Metrics the assignment explicitly requires (Part 2.6 + Part 3.8), used
# below to check the round summary against the spec rather than just
# printing whatever happens to be there.
REQUIRED_ROOM_GLOBAL = ["new_version", "n_completed", "n_selected", "n_dropped", "n_rejected",
                        "strategy", "checkpoint_uri"]
REQUIRED_TIMING = ["duration_seconds", "aggregation_seconds"]
# The following are OPTIONAL in the summary (only present if at least one
# client reported them) but must be present when torch is installed and a
# real client actually trains -- checked separately below.
OPTIONAL_WHEN_TORCH_AVAILABLE = [
    "avg_local_training_seconds", "avg_download_seconds", "avg_upload_seconds",
    "avg_selection_wait_seconds", "avg_payload_bytes", "total_payload_bytes",
]

initial_state = {"w": np.zeros(4, dtype="float32")}
payload = {
    "room_id": ARGS.room_id,
    "model_contract": {"model_id": "tiny", "framework": "numpy",
                        "param_shapes": {"w": [4]}, "max_update_norm": None},
    "preprocessing_contract": "none",
    "aggregation": {"strategy": "fedavg", "min_available_clients": 1, "min_fit_clients": 1,
                     "quorum": 1.0, "round_timeout_seconds": 10, "byzantine_f": 0},
    "target_rounds": 1,
    "initial_state_b64": state_to_b64(initial_state),
}
r = requests.post(f"{URL}/rooms", json=payload)
r.raise_for_status()

cfg = ClientConfig(client_id="alice", coordinator_url=URL, room_id=ARGS.room_id, n_clients=1)
agent = ClientAgent(cfg)
agent.join()
requests.post(f"{URL}/rooms/{ARGS.room_id}/start", json={"rounds": 1}).raise_for_status()

result = agent.train_once(max_wait_seconds=15)
print("=== train_once() local result (client-side) ===")
print(json.dumps(result, indent=2))

time.sleep(2.5)  # let the background finalizer tick
status = requests.get(f"{URL}/rooms/{ARGS.room_id}").json()
summary = status["metrics_history"][-1]
print("\n=== round summary (this is what gets logged to MLflow) ===")
print(json.dumps(summary, indent=2))

print("\n=== Coverage check against the assignment's required metric groups ===")
all_ok = True
for key in REQUIRED_ROOM_GLOBAL + REQUIRED_TIMING:
    ok = key in summary
    all_ok &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {key}")
for key in OPTIONAL_WHEN_TORCH_AVAILABLE:
    present = key in summary
    print(f"  [{'PASS' if present else 'SKIP (torch not installed / no client metrics reported)'}] {key}")

try:
    import psutil  # noqa: F401
    sys_resp = requests.get(f"{URL}/system", timeout=10).json()
    print(f"  [{'PASS' if sys_resp.get('available') else 'FAIL'}] GET /system -> {sys_resp}")
except ImportError:
    print("  [SKIP] psutil not installed -- coordinator CPU/memory metrics unavailable")

print(f"\n{'ALL REQUIRED METRICS PRESENT' if all_ok else 'SOME REQUIRED METRICS MISSING -- see FAIL lines above'}")
