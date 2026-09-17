#!/usr/bin/env bash
# Reproducible baseline startup: coordinator + MinIO + MLflow via Docker Compose.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example (edit before production use)."
fi

echo "Building and starting Fedroom baseline stack..."
docker compose -f deployments/compose/docker-compose.yaml --env-file .env up --build -d minio minio-init mlflow coordinator

echo "Waiting for coordinator health check..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/healthz > /dev/null; then
    echo "Coordinator is healthy."
    break
  fi
  sleep 2
done

echo ""
echo "Fedroom is up:"
echo "  Coordinator API:   http://localhost:8000  (docs at /docs)"
echo "  Dashboard:          http://localhost:8000/dashboard/"
echo "  MinIO console:      http://localhost:9001  (fedroomadmin / fedroomsecret)"
echo "  MLflow UI:           http://localhost:5000"
echo ""
echo "Next: create a room and run clients, e.g.:"
echo "  python -m tui.cli room create --config configs/rooms/fashion-room.yaml"
echo "  python -m tui.cli train start fashion-room --rounds 5"
echo "  docker compose -f deployments/compose/docker-compose.yaml up client-a client-b"
