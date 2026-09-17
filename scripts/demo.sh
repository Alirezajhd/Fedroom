#!/usr/bin/env bash
# Scripted walkthrough of the acceptance demonstration sequence.
# Assumes `scripts/start.sh` has already been run and the coordinator is up.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${FEDROOM_URL:-http://localhost:8000}

echo "== 1. Create room =="
python -m tui.cli room create --config configs/rooms/fashion-room.yaml --url "$URL"

echo "== 2. Client A joins and trains alone for one round =="
python -m tui.cli client join fashion-room --config configs/clients/client-a.yaml
python -m tui.cli train start fashion-room --rounds 5 --url "$URL"
python -c "
from client.agent import ClientAgent
from client.config import ClientConfig
cfg = ClientConfig.from_yaml('configs/clients/client-a.yaml')
agent = ClientAgent(cfg)
print(agent.train_once())
"

echo "== 3. Clients B and C join WHILE training is active =="
python -m tui.cli client join fashion-room --config configs/clients/client-b.yaml
python -m tui.cli client join fashion-room --config configs/clients/client-c.yaml
python -m tui.cli train advance fashion-room --url "$URL"

echo "== 4. Room status =="
python -m tui.cli room status fashion-room --url "$URL"

echo "== 5. Run inference against the latest checkpoint =="
python -m tui.cli infer fashion-room --config configs/clients/client-a.yaml --version latest

echo "== 6. Failure injection evidence =="
python experiments/inject_failures.py --url "$URL"

echo "Demo complete. See experiments/results/ for scalability/non-IID artifacts."
