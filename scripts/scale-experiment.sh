#!/usr/bin/env bash
# Deployed-client scalability sweep against a real Kubernetes cluster:
# re-applies deployments/kubernetes/04-client-job.yaml with completions/
# parallelism set to each requested level, waits for completion, and
# collects the resulting round-duration metrics via the scalability
# experiment script (run from outside the cluster, pointed at the
# coordinator's exposed/port-forwarded URL).
set -euo pipefail
cd "$(dirname "$0")/.."

LEVELS=${1:-"1 2 4 8"}
NAMESPACE=fedroom
JOB_MANIFEST=deployments/kubernetes/04-client-job.yaml
URL=${FEDROOM_URL:-http://localhost:8000}

mkdir -p experiments/results

for N in $LEVELS; do
  echo "=== Deploying $N client pod(s) ==="
  sed -e "s/completions: [0-9]*/completions: ${N}/" \
      -e "s/parallelism: [0-9]*/parallelism: ${N}/" \
      -e "s/value: \"[0-9]*\"   # keep in sync with \`completions\`/value: \"${N}\"   # keep in sync with \`completions\`/" \
      "$JOB_MANIFEST" | kubectl apply -n "$NAMESPACE" -f -

  kubectl wait --for=condition=complete job/fedroom-clients -n "$NAMESPACE" --timeout=600s || true
  kubectl logs -n "$NAMESPACE" job/fedroom-clients --all-containers=true --prefix=true || true
  kubectl delete job fedroom-clients -n "$NAMESPACE" --ignore-not-found

  echo "=== Snapshot of coordinator room status after N=${N} ==="
  curl -s "${URL}/rooms/fashion-room" | python3 -m json.tool | tee "experiments/results/k8s_scale_N${N}.json"
done
