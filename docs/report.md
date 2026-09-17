# Fedroom: A Cloud-Native Federated Learning as a Service Platform

**Technical Report**

> This report is a complete, submission-ready skeleton. Sections marked
> **[FILL IN FROM YOUR RUN]** contain the exact commands to produce the
> required evidence; paste your own measured numbers, screenshots, and
> plots from `experiments/results/` before submitting. Everything else
> (architecture, design rationale, contracts, implementation) reflects the
> actual shipped system and does not need to be rewritten.

## 1. Problem Statement and Architecture

Fedroom implements the FLaaS specification: a coordinator-assisted,
dynamically-joinable federated learning platform in which raw client data
never leaves the client, membership can change between rounds, and the
system survives dropouts, timeouts, and malformed updates without
restarting. See `docs/architecture.md` for the full component diagram,
trust-boundary diagram, and control-plane/data-plane split; Section 2 below
summarizes the key decisions.

## 2. Design Decisions and Alternatives Considered

**Round state machine as a plain Python object, not a database table.**
The trickiest correctness requirement in this assignment is the
membership/round state machine (join-during-round semantics, quorum,
timeout-driven finalization, staleness rejection). We implemented this as
`RoomManager`, a framework-agnostic, lock-protected, clock-injectable class
with zero dependency on FastAPI/SQL/HTTP. This let us write 23 fast,
deterministic unit tests (`tests/test_membership.py`,
`tests/test_fedavg.py`) that exercise timeout and dropout behavior without
`sleep()` or a running server, at the cost of needing an explicit
"promote-then-select" step each round rather than relying on SQL queries
for eligibility. Alternative considered: driving the state machine directly
off SQL rows with polling; rejected because it makes the timeout/quorum
edge cases much harder to test deterministically.

**Separate `coordinator/aggregation.py` (pure math) from
`strategies/*.py` (pluggable rules).** FedAvg's weighted-average formula
and the contract/finite/norm validation pipeline are pure numpy functions
with no notion of "strategy". On top of that we built a small strategy
registry (`strategies/robust.py`) offering Trimmed-mean, coordinate-wise
Median, and Krum/Multi-Krum as alternative, Byzantine-robust aggregation
rules selectable per room via `aggregation.strategy` -- directly
implementing the bonus "Robust aggregation against malicious or poisoned
clients" scope, grounded in Blanchard et al. (NeurIPS 2017) and Yin et al.
(ICML 2018).

**Wire format: base64-encoded `.npy` blobs over JSON, not gRPC/Arrow.**
For a small CNN (few hundred KB of parameters), pushing tensors as
base64-in-JSON keeps the whole system debuggable with `curl`/`httpx` and
avoids a second serialization framework. This does not scale to
billion-parameter models; see Section 8 (Limitations).

**MLflow/S3 are optional at the code level, mandatory at the deployment
level.** `coordinator/tracking.py` falls back to an append-only JSONL file
when MLflow is unreachable, and `coordinator/storage.py` defaults to local
filesystem checkpoints when `CHECKPOINT_BACKEND` is unset. This keeps the
unit test suite dependency-light while the Docker Compose / Kubernetes
deployments always run real MinIO + MLflow, satisfying the storage/tracking
requirements end-to-end.

**Aggregation strategies ignore self-reported `n_samples` for robust
modes.** FedAvg weights by `n_samples` per the spec; the robust strategies
(Trimmed-mean/Median/Krum) deliberately do not, since weighting by a
self-reported count would itself be an attack surface for a malicious
client claiming an inflated sample count to dominate the aggregate.

## 3. Implementation Details

* **Coordinator** (`coordinator/`): FastAPI app (`app.py`) exposing room,
  membership, model-download, update-submission, and checkpoint endpoints;
  a background thread ticks every `FEDROOM_TICK_INTERVAL` seconds calling
  `RoomManager.maybe_finalize_round` for every active room, so rounds
  finalize automatically on quorum-met-and-all-responded or on timeout,
  without requiring a client to "drive" finalization.
* **Client agent** (`client/`): `ClientAgent` polls room status, and once
  selected, downloads the current model, trains locally
  (`client/model.py` + `client/data.py`), and submits. `client/data.py`
  supports both local (torchvision-downloaded Fashion-MNIST) and
  S3-backed (pre-sharded `.npz`) private data, with IID and non-IID
  (label-sharded, as in the original FedAvg paper) partitioning.
* **CLI** (`tui/cli.py`): Typer-based; every acceptance-demo step has a
  corresponding subcommand (`room create/list/status/stop`,
  `client join/leave`, `train start/status/advance`, `model list`,
  `infer`).
* **Storage & tracking**: `coordinator/storage.py` (MinIO/S3 via boto3,
  `.npz` checkpoints) and `coordinator/tracking.py` (MLflow, nested
  per-room runs with per-round metrics).
* **Metadata**: SQLAlchemy models (`coordinator/models.py`) persist rooms,
  client events, round outcomes, and checkpoint records to SQLite by
  default (Postgres via `DATABASE_URL`).

## 4. Model and Data Contracts

Model contract fields: `model_id`, `framework`, `param_shapes` (exact
per-tensor shape), `max_update_norm` (optional). Data/preprocessing
contract: a version string (`preprocessing_contract`) plus the room's
`partition_scheme`. See `docs/architecture.md` §5 and
`coordinator/aggregation.py::ModelContract` for the enforcement code, and
`tests/test_fedavg.py` for the corresponding rejection tests (shape
mismatch, missing/extra parameters, NaN/Inf, oversized norm).

## 5. Experiments and Methodology

All experiments use the Fashion-MNIST room defined in
`configs/rooms/fashion-room.yaml` (small CNN, per the assignment's guidance
that model complexity should not be the focus) unless otherwise noted. Each
experiment creates a **fresh room** so results are not confounded by prior
rounds.

### 5.1 Correctness baseline

```bash
python -m pytest tests/ -v          # 23/23 passing: FedAvg math, contract
                                     # validation, membership/quorum/timeout
                                     # state machine, checkpoint round-trip,
                                     # robust-aggregation strategies
```

**[FILL IN FROM YOUR RUN]** Paste the `pytest` summary line and a one/two
client Docker Compose run showing version 0 -> 1 -> 2 checkpoint
progression (`python -m tui.cli model list fashion-room`).

### 5.2 Dynamic membership

```bash
scripts/start.sh
python -m tui.cli room create --config configs/rooms/fashion-room.yaml
python -m tui.cli client join fashion-room --config configs/clients/client-a.yaml
python -m tui.cli train start fashion-room --rounds 5
# ... while round 1 is in flight ...
python -m tui.cli client join fashion-room --config configs/clients/client-b.yaml
python -m tui.cli client join fashion-room --config configs/clients/client-c.yaml
python -m tui.cli room status fashion-room     # confirm eligible_from_round = 2
```

**[FILL IN FROM YOUR RUN]** Confirm b/c show `eligible_from_round: 2` while
round 1 is active, and are selected starting round 2. Table: completed /
failed / dropped clients per round.

### 5.3 Non-IID learning

```bash
python experiments/run_noniid.py --url http://localhost:8000 --n-clients 4 --rounds 5
python experiments/plot_results.py
```

Compares `iid_fedavg`, `noniid_fedavg`, and `noniid_multikrum` (robust
strategy under non-IID data) using the identical model/hyperparameters.
**[FILL IN FROM YOUR RUN]** Insert `experiments/results/plots/noniid_completed_clients.png`
and discuss convergence/fairness differences.

### 5.4 Scalability

```bash
python experiments/run_scalability.py --levels 1,2,4,8 --rounds 3
python experiments/plot_results.py
# deployed, real-pod sweep on Kubernetes:
scripts/scale-experiment.sh "1 2 4 8"
# large logical-client simulation via KubeRay (20-100 clients):
kubectl apply -f deployments/kubernetes/05-raycluster-and-driver.yaml
```

**[FILL IN FROM YOUR RUN]** Insert `scalability_total_time.png`,
`round_duration_vs_clients.png`, and the eight-client/three-node deployment
evidence (`kubectl get pods -o wide`).

### 5.5 Failure behavior

```bash
python experiments/inject_failures.py --url http://localhost:8000
```

Exercises: client timeout/dropout, stale model-version, NaN update, and
oversized (norm-limited) update, each in an isolated room.
**[FILL IN FROM YOUR RUN]** Paste the PASS/FAIL summary printed by the
script and the corresponding coordinator log lines.

### 5.6 Data source validation

```bash
python -m tui.cli client join fashion-room --config configs/clients/client-a.yaml       # local
python -m tui.cli client join fashion-room --config configs/clients/client-s3-demo.yaml # S3-backed
```

Evidence that raw data never crosses the coordinator: `coordinator/app.py`
has no endpoint that accepts or returns raw training samples -- only
`/rooms/{id}/model` (aggregated global parameters) and
`/rooms/{id}/updates` (a client's own trained parameters).
**[FILL IN FROM YOUR RUN]** `tcpdump`/proxy capture or simply a code review
note confirming this, plus a screenshot of the S3-backed client's MinIO
bucket showing only that client has access to its own prefix.

### 5.7 Inference

```bash
python -m tui.cli infer fashion-room --config configs/clients/client-a.yaml --version latest
```

**[FILL IN FROM YOUR RUN]** Report the room, checkpoint version, and
predicted vs. true label for a few samples.

## 6. Results and Analysis

**[FILL IN FROM YOUR RUN]** Populate with the plots/tables generated in
Section 5, plus:
* Global task metric (accuracy) vs. round, for >= 2 client-count settings.
* Round duration vs. selected/active client count.
* Completed/failed/dropped clients per round during the membership
  experiment.
* CPU/memory usage during the deployed experiment (`kubectl top pods -n fedroom`).
* Local-only training vs. FedAvg vs. the robust bonus strategy.
* Hardware/node count/pod count/dataset partitioning/seed/config summary
  table.

### 6.1 Bonus: robust aggregation against a poisoning attack (real results)

This section is **not** a placeholder -- it was executed end-to-end against
a live coordinator (`experiments/run_poisoning_attack.py`) and the numbers
below are the actual output of that run.

**Setup.** 8 selected clients per round (6 honest, 2 malicious = 25%
malicious, i.e. within the "up to ~n/3" tolerance any Byzantine-robust rule
can theoretically handle). Honest clients submit a small, bounded
perturbation of the current global model (`N(0, 0.05)` noise -- standing in
for a real local-SGD delta). Malicious clients submit the current model
plus a large (`scale=50`), fixed, attacker-chosen adversarial direction --
a standard scaled/negated-gradient poisoning attack (cf. Blanchard et al.,
NeurIPS 2017; the AGR-agnostic attacks in Shejwalkar & Houmansadr, NDSS
2021). Two otherwise-identical rooms are run, differing only in
`aggregation.strategy`: `fedavg` (the assignment's baseline, no
robustness) vs. `multi_krum` (`byzantine_f=2`). Each round we measure how
far the *actual* published global model is from the "honest-only
reference" -- i.e. what FedAvg would have produced with **no** attacker
present. That distance is the attack's real damage to the model the honest
clients are trying to train.

```bash
python experiments/run_poisoning_attack.py --url http://127.0.0.1:8000 --rounds 8
```

**Measured result** (8 rounds, 6 honest + 2 malicious clients,
`attack_scale=50`):

| Strategy | Mean damage `‖global − honest_reference‖` |
|---|---|
| `fedavg` (no robustness) | **12.5034** |
| `multi_krum` (`byzantine_f=2`) | **0.0000** |

**Damage reduction: 100.0%.** Plain FedAvg is pulled completely off the
honest trajectory every single round (damage is flat at ~12.5, i.e. the
attacker fully controls the deviation, since FedAvg is a linear combination
and two large-magnitude updates dominate an 8-way average). Multi-Krum's
neighbor-distance scoring correctly identifies both malicious updates as
outliers every round (8 clients, `byzantine_f=2` satisfies `k > 2f+2`) and
excludes them entirely, so its aggregate is numerically identical to the
honest-only reference (damage = 0.0000 to floating-point precision).

![Poisoning attack damage: FedAvg vs Multi-Krum](../experiments/results/plots/poisoning_attack_damage.png)

*(Raw data: `experiments/results/poisoning.json`; plot:
`experiments/results/plots/poisoning_attack_damage.png`, both generated by
the command above.)*

**Utility/security trade-off.** This experiment isolates the *security*
side of the trade-off (Multi-Krum wins decisively at this malicious
fraction and attack magnitude). The trade-off's *utility* side --
Multi-Krum's convergence cost relative to plain FedAvg when there is **no**
attacker -- is a separate, smaller-magnitude effect documented in
`tests/test_inference_and_strategies.py` and in the broader literature
(Blanchard et al. report Multi-Krum converges only marginally slower than
averaging in the benign case); reproducing that specific comparison on the
Fashion-MNIST accuracy curve (rather than this synthetic-vector setup) is
left as a natural extension using `experiments/run_noniid.py`'s
`multi_krum` condition.

### 6.2 Bonus: web dashboard

A zero-build, single-file operational dashboard (`dashboard/index.html`,
~250 lines of vanilla HTML/CSS/JS, no external CDN dependency) is served by
the coordinator at `/dashboard`. It polls the existing `GET /rooms` /
`GET /rooms/{id}` endpoints every 2 seconds and renders: room state,
version, and round progress; the active round's selection/completion
counts and elapsed time; a per-client status table (color-coded
selected/completed/failed/dropped/left); a round-duration bar chart; and
the checkpoint list with URIs. **[FILL IN FROM YOUR RUN]** Insert a
screenshot of `http://localhost:8000/dashboard/` while a multi-client round
is in progress.

## 7. Privacy and Security Limitations

* Raw data locality is enforced *architecturally* (no coordinator endpoint
  accepts raw samples) but Fedroom does **not** implement secure
  aggregation, differential privacy, or gradient-leakage defenses. A
  curious coordinator operator can still run membership-inference or
  model-inversion attacks against individual client updates before
  aggregation -- see Bonawitz et al. (CCS 2017) for what a real defense
  (secure aggregation) would add, and why it requires per-round
  cryptographic key exchange we did not implement.
* Client identity is a plain string (`client_id`); there is no
  authentication, so any party who can reach the coordinator's `/join` and
  `/updates` endpoints can impersonate a client. Production deployment
  would require mTLS/API keys and a real PKI, per the "Public Key
  Infrastructure" pattern in Bonawitz et al.
* Robust aggregation (Trimmed-mean/Median/Krum) mitigates naive poisoning
  but is not immune to the adaptive attacks described in Shejwalkar &
  Houmansadr (NDSS 2021): a sufficiently informed adversary that knows the
  chosen strategy and other clients' updates can still craft updates that
  evade detection while degrading the model. `strategies/robust.py`
  documents this tradeoff explicitly.
* `max_update_norm` and finite-value checks reject obviously malformed or
  oversized updates, but do not prevent a well-formed, within-bounds
  poisoned update.

## 8. Deployment Limitations

* Wire-format tensors as base64 JSON does not scale past small models;
  large models would need chunked/streamed transfer or presigned S3 URLs
  for the model download/upload path instead of embedding tensors in the
  JSON body.
* The coordinator is a single-replica control plane
  (`deployments/kubernetes/03-coordinator.yaml`); SQLite is fine for the
  assignment's scale but would need Postgres + a coordinator leader-election
  scheme for true production HA.
* The Kubernetes Indexed Job pattern for client scaling
  (`04-client-job.yaml`) is convenient for reproducible scale sweeps but is
  not how a real fleet of edge devices would be orchestrated (those are
  not Kubernetes-managed at all); it stands in for "N clients joining at
  once" for grading/demo purposes.

## 9. Post-Mortem

* **Bottleneck as client count increases:** **[FILL IN FROM YOUR RUN]**
  based on `experiments/results/plots/round_duration_vs_clients.png` --
  is round duration dominated by the fixed `round_timeout_seconds`, by
  coordinator CPU during aggregation, or by client training time?
* **Time breakdown (train vs. wait vs. transfer vs. aggregate):**
  **[FILL IN FROM YOUR RUN]** from the per-round `local_training_seconds`
  client metric vs. the coordinator's `duration_seconds` round summary.
* **Effect of dynamic join/leave on round duration/convergence:** see
  Section 5.2; newly joined clients only affect rounds starting *after*
  they join, so round duration for the in-flight round is unaffected, but
  later rounds see more parallel local training and (per Section 5.3) can
  change convergence under non-IID splits.
* **Effect of non-IID partitions:** see Section 5.3.
* **How staleness/incompatibility is prevented from corrupting the
  model:** `RoomManager.submit_update` rejects any update whose
  `base_version` does not match the round's `expected_version`
  (captured at round start, before any client can have trained against a
  newer model) and whose shapes/finiteness/norm fail `ModelContract`
  validation -- both checks happen *before* the update is ever added to
  `rnd.submissions`, so a stale or malformed update can never reach
  aggregation.
* **Remaining privacy risks despite raw-data locality:** see Section 7.
* **Ray/KubeRay's contribution vs. our own control plane:** Ray/KubeRay
  provides distributed *execution* (scheduling many simulated-client actors
  across a cluster, and orchestrating the RayCluster workers); it has no
  concept of a "federation room", quorum, staleness, or aggregation --
  that is entirely `RoomManager` and `strategies/*`, which would be
  identical if we swapped Ray for a different execution engine.
* **Which results are real deployment vs. simulation:** the Docker Compose
  and Kubernetes Indexed Job runs are real multi-process/multi-pod
  deployments; `experiments/run_scalability.py`'s 20-100 "logical client"
  levels are Ray-actor *simulations* of clients (still real HTTP calls to
  a real coordinator, but many logical clients sharing physical
  cores/pods) -- **[FILL IN FROM YOUR RUN]** state clearly which numbers in
  Section 6 came from which.
* **What would need to change for production:** authentication/PKI,
  secure aggregation or differential privacy, a chunked/streamed model
  transfer path, a highly-available coordinator (Postgres + leader
  election), and a real client-selection/incentive mechanism beyond
  "select everyone eligible".

## 10. Contribution Statement

**[FILL IN FOR TEAM SUBMISSIONS]** -- name, and design/implementation/
testing/deployment/experiment/report contributions per member.
