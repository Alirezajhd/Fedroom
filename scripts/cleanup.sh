#!/usr/bin/env bash
# Stop and remove all Fedroom baseline resources (containers, networks, volumes).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Tearing down Fedroom Docker Compose stack (including volumes)..."
docker compose -f deployments/compose/docker-compose.yaml down -v --remove-orphans

echo "Removing local scratch data (checkpoints/db/raw data caches)..."
rm -rf data/checkpoints data/fedroom.db data/mlflow_fallback.jsonl data/raw

echo "Cleanup complete."
