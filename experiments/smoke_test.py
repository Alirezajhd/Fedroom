"""
End-to-end smoke/integration test against a LIVE coordinator.

Unlike tests/, which exercise RoomManager in-process with no network, this
script drives the real HTTP API end-to-end: room creation, dynamic
mid-round joins, weighted FedAvg aggregation, stale/NaN update rejection
(each in its own throwaway room, so a deliberately-bad update never costs a
real client its one submission slot for the round), timeout-driven dropout
with quorum-based aggregation, checkpoint publishing/versioning, checkpoint
loading, and a clean client leave.

Usage:
    uvicorn coordinator.app:app --host 127.0.0.1 --port 8000 &
    python experiments/smoke_test.py --url http://127.0.0.1:8000

Exits non-zero (via AssertionError) on the first failing check.
"""
import argparse
import sys
sys.modules['torch'] = None          # <--- Add this
import time
from pathlib import Path

import numpy as np
import requests

# Allow running this script directly (`python experiments/smoke_test.py`)
# without the project root already being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.agent import ClientAgent
from client.config import ClientConfig
from coordinator.codec import state_to_b64

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
ARGS = parser.parse_args()
URL = ARGS.url.rstrip("/")


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


# 1. Create room with a tiny synthetic contract (no torch needed)
initial_state = {"w": np.zeros(4, dtype="float32")}
payload = {
    "room_id": "smoke-room",
    "model_contract": {
        "model_id": "tiny",
        "framework": "numpy",
        "param_shapes": {"w": [4]},
        "max_update_norm": None,
    },
    "preprocessing_contract": "none",
    "aggregation": {
        "strategy": "fedavg",
        "min_available_clients": 1,
        "min_fit_clients": 1,
        "quorum": 0.5,
        "round_timeout_seconds": 8,
        "byzantine_f": 0,
    },
    "target_rounds": 3,
    "initial_state_b64": state_to_b64(initial_state),
}
r = requests.post(f"{URL}/rooms", json=payload)
r.raise_for_status()
check("room created at version 0", r.json()["version"] == 0)

# 2. Client A joins, room starts, round 1 begins
cfg_a = ClientConfig(
    client_id="alice", coordinator_url=URL, room_id="smoke-room", n_clients=3
)
agent_a = ClientAgent(cfg_a)
agent_a.join()
r = requests.post(f"{URL}/rooms/smoke-room/start", json={"rounds": 3})
r.raise_for_status()
status = r.json()
check(
    "round 1 started with alice selected",
    status["active_round"]["selected"] == ["alice"],
)

# 3. Client B joins WHILE round 1 is active -> eligible round 2, not round 1
r = requests.post(f"{URL}/rooms/smoke-room/join", json={"client_id": "bob"})
check(
    "bob eligible_from_round == 2 (mid-round join)",
    r.json()["eligible_from_round"] == 2,
)

# 4. Alice trains and submits (torch not installed -> numpy fallback path exercised)
result = agent_a.train_once()
check("alice's update accepted", result["status_code"] == 200)

time.sleep(0.3)
requests.post(f"{URL}/rooms/smoke-room/tick")
status = requests.get(f"{URL}/rooms/smoke-room").json()
check("round 1 aggregated -> version 1", status["current_version"] == 1)

# 5. Advance to round 2: both alice and bob should now be selected
r = requests.post(f"{URL}/rooms/smoke-room/next-round")
r.raise_for_status()
status = requests.get(f"{URL}/rooms/smoke-room").json()
check(
    "round 2 selects both alice and bob",
    set(status["active_round"]["selected"]) == {"alice", "bob"},
)

# 6. Stale-update and NaN-update rejection, exercised in an isolated
#    throwaway room so they can't side-effect alice/bob's state in
#    smoke-room (a client whose update is rejected is marked FAILED for
#    that round and cannot resubmit -- correct behavior, but we don't want
#    to spend alice's one submission slot on a deliberately bad update).
r = requests.post(f"{URL}/rooms", json={**payload, "room_id": "smoke-room-reject"})
r.raise_for_status()
requests.post(f"{URL}/rooms/smoke-room-reject/join", json={"client_id": "eve"})
requests.post(f"{URL}/rooms/smoke-room-reject/start", json={"rounds": 1})

r = requests.post(
    f"{URL}/rooms/smoke-room-reject/updates",
    json={
        "client_id": "eve",
        "base_version": 99,
        "n_samples": 5,
        "state_b64": state_to_b64(initial_state),
        "metrics": {},
    },
)
check(
    "stale update rejected with 422",
    r.status_code == 422 and "version" in r.json()["reason"],
)

r2 = requests.post(f"{URL}/rooms", json={**payload, "room_id": "smoke-room-reject-2"})
r2.raise_for_status()
requests.post(f"{URL}/rooms/smoke-room-reject-2/join", json={"client_id": "eve"})
requests.post(f"{URL}/rooms/smoke-room-reject-2/start", json={"rounds": 1})
poisoned = {"w": np.full(4, np.nan, dtype="float32")}
r = requests.post(
    f"{URL}/rooms/smoke-room-reject-2/updates",
    json={
        "client_id": "eve",
        "base_version": 0,
        "n_samples": 5,
        "state_b64": state_to_b64(poisoned),
        "metrics": {},
    },
)
check("NaN update rejected with 422", r.status_code == 422)

# 8. Bob drops (never submits); alice submits; wait for timeout -> quorum(0.5) met -> aggregates
result = agent_a.train_once(max_wait_seconds=15)
check(
    "alice's round-2 update accepted (bob still pending)",
    result is not None and result["status_code"] == 200,
)
time.sleep(9)
requests.post(f"{URL}/rooms/smoke-room/tick")
status = requests.get(f"{URL}/rooms/smoke-room").json()
check(
    "bob marked dropped after timeout", status["clients"]["bob"]["status"] == "dropped"
)
check(
    "round 2 aggregated despite bob's dropout -> version 2",
    status["current_version"] == 2,
)

# 9. Checkpoint list + inference
check("two checkpoints published", len(status["checkpoints"]) == 2)
ck = requests.get(f"{URL}/rooms/smoke-room/checkpoints/2").json()
check("checkpoint v2 loadable with correct version", ck["version"] == 2)

# 10. Client leave
r = requests.post(f"{URL}/rooms/smoke-room/leave", json={"client_id": "bob"})
r.raise_for_status()
check("bob left cleanly", r.json()["status"] == "left")

print("\nAll end-to-end smoke checks passed.")
