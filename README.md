# Fedroom -- Federated Learning as a Service

Fedroom is a cloud-native platform for dynamic, private, distributed model
training: a coordinator-assisted control plane manages federation "rooms"
(dynamic client membership, weighted FedAvg / robust aggregation, versioned
checkpoints), while client agents keep raw training data local and only
ever exchange model parameters and metrics.

> Raw client training records never pass through the coordinator. Only
> model updates, metadata, client status, and metrics do.

**Bonus scope implemented (with real, executed evidence, not just code):**
robust aggregation against poisoning attacks (Trimmed-mean / Median /
Krum / Multi-Krum, evaluated in `docs/report.md` §6.1 with a live 2/8
malicious-client attack showing 100% damage reduction) and a web
dashboard (`/dashboard`, see §8b below).

## 1. System requirements

* Docker + Docker Compose v2 (baseline deployment)
* A Kubernetes cluster for the deployment/scalability experiments (`kind`
  is fine for a single-host baseline; note in your report that CPU/
  memory/disk/network are physically shared in that case)
* Python 3.10+ if you want to run the CLI / experiment scripts from the
  host instead of inside containers
* [KubeRay operator](https://github.com/ray-project/kuberay) installed in
  the cluster, only if you want to run the large (20-100 client)
  Ray-simulated scalability experiment on Kubernetes
  (`deployments/kubernetes/05-raycluster-and-driver.yaml`)

## 2. Architecture summary

Control plane (coordinator) <-> data plane (client agents), with MinIO for
checkpoint storage, MLflow for experiment tracking, SQLite/Postgres for
metadata, and Ray/KubeRay as the distributed execution engine for
simulated-client experiments. Full diagrams, trust boundary, and
component responsibilities: **[`docs/architecture.md`](docs/architecture.md)**.

```
fedroom/
├── coordinator/         # control plane: FastAPI app, room/round state
│                         # machine, validation, storage/tracking glue,
│                         # SQLAlchemy metadata models
├── client/               # data plane: client agent, model, data loading
├── strategies/           # pluggable aggregation rules (fedavg, trimmed_mean,
│                         # median, krum, multi_krum)
├── dashboard/             # bonus: single-file web dashboard (served at /dashboard)
├── tui/                  # Typer CLI ("fedroom" commands)
├── tests/                 # pytest: correctness + membership state machine
├── configs/               # example room + client YAML configs
├── deployments/
│   ├── compose/           # Docker Compose baseline
│   └── kubernetes/        # namespaced manifests + KubeRay
├── experiments/           # scalability / non-IID / failure-injection scripts
├── docs/                  # architecture.md, report.md
└── scripts/               # start.sh, cleanup.sh, demo.sh, scale-experiment.sh
```

## 3. One-command baseline startup

```bash
./scripts/start.sh
```

This builds and starts MinIO, MLflow, and the coordinator via Docker
Compose, waits for the coordinator health check, and prints the service
URLs:

* Coordinator API: http://localhost:8000 (interactive docs at `/docs`)
* MinIO console: http://localhost:9001 (`fedroomadmin` / `fedroomsecret`)
* MLflow UI: http://localhost:5000

## 4. Quick start: your first federation room

Install the CLI dependencies locally (or exec into the coordinator/client
image, which already has them):

```bash
pip install -r requirements-client.txt   # includes CLI deps + torch for real training
```

```bash
# 1. Create a room from config (derives the model contract + initial weights)
python -m tui.cli room create --config configs/rooms/fashion-room.yaml

# 2. Client A joins and trains alone for the first round
python -m tui.cli client join fashion-room --config configs/clients/client-a.yaml
python -m tui.cli train start fashion-room --rounds 5
python -m client.agent configs/clients/client-a.yaml --rounds 1

# 3. Clients B and C join WHILE round 1 is active -- they become eligible
#    starting round 2, not round 1 (see docs/architecture.md / RoomManager)
python -m tui.cli client join fashion-room --config configs/clients/client-b.yaml
python -m tui.cli client join fashion-room --config configs/clients/client-c.yaml
python -m tui.cli train advance fashion-room

# 4. Inspect room/round/checkpoint status
python -m tui.cli room status fashion-room
python -m tui.cli model list fashion-room

# 5. Run local inference against the latest checkpoint
python -m tui.cli infer fashion-room --config configs/clients/client-a.yaml --version latest
```

Or run the whole thing scripted:

```bash
./scripts/demo.sh
```

## 5. Running the client fleet via Docker Compose

```bash
docker compose -f deployments/compose/docker-compose.yaml up client-a client-b
```

Each client container is built from `Dockerfile.client` (includes
torch/torchvision/ray), mounts its own private data volume, and only ever
talks to the coordinator over HTTP.

## 6. Kubernetes deployment

```bash
kubectl apply -k deployments/kubernetes/
kubectl -n fedroom get pods -w
```

Then deploy N real client pods for the scalability sweep (see
`docs/report.md` §5.4):

```bash
./scripts/scale-experiment.sh "1 2 4 8"
```

And, for the large (20-100) simulated-client experiment via KubeRay (after
installing the KubeRay operator):

```bash
kubectl apply -f deployments/kubernetes/05-raycluster-and-driver.yaml
kubectl -n fedroom logs job/fedroom-scalability-driver -f
```

## 7. Experiments

```bash
python -m pytest tests/ -v                                   # correctness + state machine (23 tests)
python experiments/run_scalability.py --levels 1,2,4,8        # local Ray/thread simulation
python experiments/run_noniid.py                              # IID vs non-IID vs robust strategy
python experiments/inject_failures.py                         # timeout/stale/NaN/oversized rejection
python experiments/plot_results.py                             # renders required plots/tables
```

Results land in `experiments/results/` (JSON) and
`experiments/results/plots/` (PNG + markdown tables), ready to paste into
`docs/report.md`.

## 8. Aggregation strategies

Set `aggregation.strategy` in a room config:

| Strategy | Rule | Reference |
|---|---|---|
| `fedavg` | Weighted average by `n_samples` (default) | McMahan et al., AISTATS 2017 |
| `trimmed_mean` | Coordinate-wise trimmed mean, drops `byzantine_f` largest/smallest per coordinate | Yin et al., ICML 2018 |
| `median` | Coordinate-wise median | Yin et al., ICML 2018 |
| `krum` | Selects the single update closest to its neighbors (drops outliers) | Blanchard et al., NeurIPS 2017 |
| `multi_krum` | Averages the `n - byzantine_f` best-scoring updates | Blanchard et al., NeurIPS 2017 |

See `configs/rooms/fashion-room-robust.yaml` for an example, and
`tests/test_inference_and_strategies.py` for correctness tests showing each
strategy resisting a synthetic outlier/poisoned update.

**Bonus: poisoning-attack evaluation (real, executed results).** A
controlled model-poisoning attack (2/8 malicious clients, scaled adversarial
updates) run against `fedavg` vs. `multi_krum` shows Multi-Krum eliminating
100% of the attack's measured damage to the aggregate. Reproduce it with:

```bash
python experiments/run_poisoning_attack.py --url http://localhost:8000 --rounds 8
```

Full writeup and the actual plot/numbers from this run:
`docs/report.md` §6.1.

## 8b. Bonus: web dashboard

The coordinator serves a zero-build operational dashboard at
`/dashboard` (e.g. http://localhost:8000/dashboard/) -- a single static
HTML/JS file (`dashboard/index.html`) that polls the REST API every 2s and
shows, per room: state/version/round progress, the active round's
selection and completion counts, a per-client status table, a round-duration
bar chart, and the checkpoint list. No build step, no external CDN, works
against any coordinator URL you point it at (editable in the dashboard's
top bar).

## 9. Cleanup

```bash
./scripts/cleanup.sh
# or, for Kubernetes:
kubectl delete namespace fedroom
```

## 10. Testing

```bash
python -m pytest tests/ -v --cov=coordinator --cov=strategies
```

The suite requires only `numpy`, `pytest`, and `PyYAML` -- no database,
object storage, MLflow server, or torch install needed -- because
`coordinator/roommanager.py` and `coordinator/aggregation.py` are
deliberately infrastructure-free and take an injectable clock, so
timeout/quorum/staleness behavior is tested deterministically without
`sleep()` or a live server.

## 11. Configuration reference

See `.env.example` for every environment variable the coordinator/client
images read (storage backend, S3/MinIO credentials, MLflow URI, database
URL, tick interval, update-norm limit). Copy it to `.env` before running
`scripts/start.sh` (done automatically if `.env` is missing).

## 12. Security & privacy notes

Federated learning does not automatically provide confidentiality,
integrity, anonymity, or differential privacy. See `docs/report.md` §7 for
a full discussion of what Fedroom does and does not protect against, and
`strategies/robust.py`'s module docstring for the specific poisoning
threat model the bonus robust-aggregation strategies address (and do not
fully solve).

## 13. Starting points / attribution

* McMahan, B. et al. *Communication-Efficient Learning of Deep Networks
  from Decentralized Data.* AISTATS, 2017. (weighted FedAvg formula)
* Bonawitz, K. et al. *Practical Secure Aggregation for Privacy-Preserving
  Machine Learning.* ACM CCS, 2017. (referenced in the security
  limitations discussion)
* Blanchard, P. et al. *Machine Learning with Adversaries: Byzantine
  Tolerant Gradient Descent.* NeurIPS, 2017. (Krum/Multi-Krum strategy)
* Yin, D. et al. *Byzantine-Robust Distributed Learning: Towards Optimal
  Statistical Rates.* ICML, 2018. (Trimmed-mean/Median strategy)
* Shejwalkar, V. & Houmansadr, A. *Manipulating the Byzantine: Optimizing
  Model Poisoning Attacks and Defenses for Federated Learning.* NDSS, 2021.
  (referenced in the security limitations discussion)

All other code (coordinator, client agent, aggregation strategies, CLI,
deployment manifests, experiments) is original to this project.
