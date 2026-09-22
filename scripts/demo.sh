#!/usr/bin/env bash
# Scripted walkthrough of the acceptance demonstration sequence.
# Assumes `scripts/start.sh` has already been run and the coordinator is up.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${FEDROOM_URL:-http://localhost:8000}

echo "== 1. Create room =="
python -m tui.cli room create --config configs/rooms/fashion-room.yaml --url "$URL"

echo "== 2. Client A joins and trains alone for one round =="
# NOTE: `client join` only registers a client -- it does not train. `client
# train` is what actually polls for selection, downloads the model, trains
# locally, and submits. Skipping it is the #1 cause of a round never
# progressing (it just times out with zero responses and fails quorum).
python -m tui.cli client join fashion-room --config configs/clients/client-a.yaml --url "$URL"
python -m tui.cli train start fashion-room --rounds 1 --url "$URL"
python -m tui.cli client train fashion-room --config configs/clients/client-a.yaml --rounds 1 --url "$URL"

echo "== 3. Clients B and C join WHILE training is active, then train in round 2 =="
python -m tui.cli client join fashion-room --config configs/clients/client-b.yaml --url "$URL"
python -m tui.cli client join fashion-room --config configs/clients/client-c.yaml --url "$URL"
python -m tui.cli train advance fashion-room --url "$URL"
python -m tui.cli client train fashion-room --config configs/clients/client-b.yaml --rounds 1 --url "$URL"
python -m tui.cli client train fashion-room --config configs/clients/client-c.yaml --rounds 1 --url "$URL"

echo "== 4. Room status =="
python -m tui.cli room status fashion-room --url "$URL"

echo "== 5. Run inference against the latest checkpoint =="
python -m tui.cli infer fashion-room --config configs/clients/client-a.yaml --version latest --url "$URL"

echo "== 6. Failure injection evidence =="
python experiments/inject_failures.py --url "$URL"

echo "== 7. Scalability experiment (writes experiments/results/scalability.json) =="
python experiments/run_scalability.py --rounds 20 --url "$URL" --levels 1,2,4,8

echo "== 8. Non-IID experiment (writes experiments/results/noniid.json) =="
python experiments/run_noniid.py --rounds 20 --url "$URL"

echo "== 9. Poisoning Attack Defense (writes experiments/results/poisoning.json) =="
python experiments/run_poisoning_attack.py --rounds 8 --url "$URL"

echo "== 10. Render plots/tables from the artifacts above =="
python experiments/plot_results.py

echo "== 11. Print demo summary =="
python scripts/print_demo_summary.py

echo "Demo complete. See experiments/results/ for scalability/non-IID artifacts."
