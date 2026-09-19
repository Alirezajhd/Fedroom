"""
Fedroom client agent.

A client agent represents one data owner. It:
  1. joins a federation room,
  2. polls the coordinator until it has been selected for a round,
  3. downloads the current global model,
  4. trains locally on its own private partition (never leaves the process),
  5. submits only the resulting parameter update + sample count, and
  6. can separately run local inference against any published checkpoint.

This module has a CLI entry point (`python -m client.agent ...`) but is also
used directly by `experiments/run_scalability.py` (via Ray actors) and by
`tui/cli.py`.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np
import requests

from client.config import ClientConfig
from client.data import partition_for_client
from client.model import build_model, model_contract_shapes, numpy_state_to_torch, torch_state_to_numpy
from coordinator.codec import b64_to_state, state_to_b64

logger = logging.getLogger("fedroom.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ClientAgent:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    def _url(self, path: str) -> str:
        return f"{self.cfg.coordinator_url.rstrip('/')}{path}"

    def join(self) -> dict:
        resp = self.session.post(
            self._url(f"/rooms/{self.cfg.room_id}/join"),
            json={"client_id": self.cfg.client_id, "capabilities": self.cfg.capabilities},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("client %s joined room %s: %s", self.cfg.client_id, self.cfg.room_id, resp.json())
        return resp.json()

    def leave(self) -> dict:
        resp = self.session.post(
            self._url(f"/rooms/{self.cfg.room_id}/leave"),
            json={"client_id": self.cfg.client_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict:
        resp = self.session.get(self._url(f"/rooms/{self.cfg.room_id}"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    def _load_partition(self):
        return partition_for_client(
            client_id=self.cfg.client_id,
            n_clients=self.cfg.n_clients,
            scheme=self.cfg.partition_scheme,
            source=self.cfg.data_source,
            local_root=self.cfg.local_data_root,
            s3_bucket=self.cfg.s3_bucket,
            s3_key=self.cfg.s3_key,
            samples_per_client=self.cfg.samples_per_client,
        )

    def _train_locally(self, global_state: dict) -> tuple[dict, int, dict]:
        """Train on the local partition starting from `global_state`.

        Returns (updated_state, n_samples, local_metrics). `local_metrics`
        includes both this client's own post-training quality
        (train_loss/train_accuracy) AND its evaluation of the model it was
        JUST GIVEN, before training (pretrain_eval_loss/pretrain_eval_accuracy)
        on a held-out slice of its own partition -- this is the federated-
        evaluation signal the coordinator averages across clients to report
        an approximate "global task metric" without ever holding a labeled
        validation set itself (see RoomManager._summarize_client_metrics).
        """
        partition = self._load_partition()
        n_samples = len(partition.labels)

        # Hold out the last 20% of this client's own partition as a local
        # validation split -- used only for the pretrain-eval signal above,
        # never sent anywhere itself (only the resulting scalar loss/accuracy
        # is reported).
        n_val = max(1, int(0.2 * n_samples)) if n_samples >= 5 else 0
        n_train = n_samples - n_val
        train_images, val_images = partition.images[:n_train], partition.images[n_train:]
        train_labels, val_labels = partition.labels[:n_train], partition.labels[n_train:]

        try:
            import torch
            import torch.nn as nn

            model = build_model()
            numpy_state_to_torch(global_state, model=model)

            # --- pretrain evaluation of the model we were just given ---
            pretrain_eval_loss, pretrain_eval_accuracy = None, None
            if n_val > 0:
                model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(val_images)
                    yv = torch.from_numpy(val_labels)
                    logits = model(xv)
                    pretrain_eval_loss = float(nn.functional.cross_entropy(logits, yv).item())
                    pretrain_eval_accuracy = float((logits.argmax(1) == yv).float().mean().item())

            x = torch.from_numpy(train_images)
            y = torch.from_numpy(train_labels)
            opt = torch.optim.SGD(model.parameters(), lr=self.cfg.learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            model.train()
            total_loss, total_correct, total_seen = 0.0, 0, 0
            n = len(y)
            bs = max(1, self.cfg.batch_size)
            t0 = time.time()
            for _ in range(self.cfg.local_epochs):
                perm = torch.randperm(n) if n > 0 else torch.arange(0)
                for start in range(0, n, bs):
                    idx = perm[start:start + bs]
                    xb, yb = x[idx], y[idx]
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    total_loss += float(loss.item()) * len(idx)
                    total_correct += int((logits.argmax(1) == yb).sum().item())
                    total_seen += len(idx)
            elapsed = time.time() - t0

            metrics = {
                "train_loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
                "pretrain_eval_loss": pretrain_eval_loss,
                "pretrain_eval_accuracy": pretrain_eval_accuracy,
                "local_training_seconds": elapsed,
                "n_samples": n_samples,
            }
            return torch_state_to_numpy(model), n_samples, metrics
        except ImportError:
            # torch not installed: numpy-only fallback so the round can still
            # complete (useful for lightweight CI / scalability simulations
            # that only care about coordinator throughput, not model quality).
            # No meaningful loss/accuracy exists in this path -- reported as
            # None (never a fabricated number), which RoomManager's averaging
            # simply skips.
            t0 = time.time()
            noisy_state = {
                k: (v + np.random.default_rng(hash(self.cfg.client_id) % (2**32)).normal(
                    0, 1e-3, size=v.shape).astype(v.dtype))
                for k, v in global_state.items()
            }
            elapsed = time.time() - t0
            metrics = {"train_loss": None, "train_accuracy": None,
                       "pretrain_eval_loss": None, "pretrain_eval_accuracy": None,
                       "local_training_seconds": elapsed, "n_samples": n_samples,
                       "note": "torch not installed; numpy no-op fallback update used"}
            return noisy_state, n_samples, metrics

    # ------------------------------------------------------------------ #
    def train_once(self, wait_for_selection: bool = True, max_wait_seconds: float = 300.0) -> Optional[dict]:
        """Wait until this client is selected in the active round, train, and
        submit. Returns the submission response, or None if the room stopped
        or no round selected this client within `max_wait_seconds` (e.g. a
        round failed quorum before this client was ever selected/responded,
        and no new round has started yet) -- avoids polling forever.
        """
        deadline = time.time() + max_wait_seconds
        wait_t0 = time.time()
        while True:
            st = self.status()
            if st["state"] in ("stopped",):
                logger.info("room stopped; client %s exiting", self.cfg.client_id)
                return None
            active = st.get("active_round")
            client_state = st["clients"].get(self.cfg.client_id, {}).get("status")
            if active and client_state == "selected" and self.cfg.client_id in active["selected"]:
                break
            if not wait_for_selection:
                return None
            if time.time() >= deadline:
                logger.warning(
                    "client %s gave up waiting for selection after %.0fs "
                    "(no active round included this client -- has the round "
                    "failed quorum and not yet been restarted?)",
                    self.cfg.client_id, max_wait_seconds,
                )
                return None
            time.sleep(self.cfg.poll_interval_seconds)
        selection_wait_seconds = time.time() - wait_t0

        download_t0 = time.time()
        model_resp = self.session.get(self._url(f"/rooms/{self.cfg.room_id}/model"), timeout=30)
        model_resp.raise_for_status()
        download_seconds = time.time() - download_t0
        payload = model_resp.json()
        base_version = payload["version"]
        global_state = b64_to_state(payload["state_b64"])
        download_bytes = len(model_resp.content)

        updated_state, n_samples, metrics = self._train_locally(global_state)

        submit_body = {
            "client_id": self.cfg.client_id,
            "base_version": base_version,
            "n_samples": n_samples,
            "state_b64": state_to_b64(updated_state),
            "metrics": {
                **metrics,
                "selection_wait_seconds": selection_wait_seconds,
                "download_seconds": download_seconds,
                "download_bytes": download_bytes,
            },
        }
        # Upload/server-processing time cannot be known before the request
        # finishes sending (it would have to be included in the very body
        # being timed), so it is measured server-side instead -- see
        # coordinator/app.py::submit_update, which injects "upload_seconds"
        # into the metrics it hands to RoomManager. `payload_bytes` here is
        # what we DO know client-side before sending.
        submit_body["metrics"]["payload_bytes"] = len(json.dumps(submit_body).encode("utf-8"))

        client_side_t0 = time.time()
        submit_resp = self.session.post(
            self._url(f"/rooms/{self.cfg.room_id}/updates"),
            json=submit_body,
            timeout=30,
        )
        # Round-trip time as observed by the client (network + server
        # processing); logged locally for the caller/CLI, not sent to the
        # server (see note above).
        client_observed_upload_seconds = time.time() - client_side_t0
        result = {"status_code": submit_resp.status_code, "body": submit_resp.json(),
                  "client_observed_upload_seconds": client_observed_upload_seconds}
        logger.info("client %s submitted update: %s", self.cfg.client_id, result)
        return result

    def run_rounds(self, rounds: int) -> list[dict]:
        results = []
        for _ in range(rounds):
            r = self.train_once()
            if r is None:
                break
            results.append(r)
        return results

    # ------------------------------------------------------------------ #
    def infer(self, version: str = "latest", sample_index: int = 0) -> dict:
        """Load a global checkpoint and run local inference on one sample
        from this client's own partition. Reports the model version used.
        """
        if version == "latest":
            st = self.status()
            checkpoints = st["checkpoints"]
            if not checkpoints:
                raise RuntimeError("No checkpoints published yet for this room")
            version_int = checkpoints[-1]["version"]
        else:
            version_int = int(version)

        ck_resp = self.session.get(
            self._url(f"/rooms/{self.cfg.room_id}/checkpoints/{version_int}"), timeout=30
        )
        ck_resp.raise_for_status()
        payload = ck_resp.json()
        state = b64_to_state(payload["state_b64"])

        partition = self._load_partition()
        sample_image = partition.images[sample_index: sample_index + 1]
        true_label = int(partition.labels[sample_index])

        try:
            import torch

            model = build_model()
            numpy_state_to_torch(state, model=model)
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(sample_image))
                pred = int(logits.argmax(1).item())
        except ImportError:
            pred = -1  # torch unavailable; cannot run real inference

        self.session.post(
            self._url(f"/rooms/{self.cfg.room_id}/inference-log"),
            json={"client_id": self.cfg.client_id, "version": version_int,
                  "note": f"predicted={pred} true={true_label}"},
            timeout=10,
        )
        return {"version": version_int, "predicted": pred, "true_label": true_label}


def _main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Fedroom client agent")
    parser.add_argument("config", help="Path to client YAML config")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--mode", choices=["train", "infer", "join", "leave"], default="train")
    args = parser.parse_args()

    cfg = ClientConfig.from_yaml(args.config)

    # Kubernetes Indexed Job support: each pod in an indexed Job gets a
    # unique JOB_COMPLETION_INDEX, which we use to derive a unique,
    # reproducible client identity. This is how the scalability experiment
    # (1/2/4/8+ clients) is driven from a single Job manifest -- see
    # deployments/kubernetes/04-client-job.yaml.
    idx = os.environ.get("JOB_COMPLETION_INDEX")
    if idx is not None:
        cfg.client_id = f"{cfg.client_id}-{idx}"
    if os.environ.get("FEDROOM_COORDINATOR_URL"):
        cfg.coordinator_url = os.environ["FEDROOM_COORDINATOR_URL"]
    if os.environ.get("FEDROOM_ROOM_ID"):
        cfg.room_id = os.environ["FEDROOM_ROOM_ID"]
    if os.environ.get("FEDROOM_N_CLIENTS"):
        cfg.n_clients = int(os.environ["FEDROOM_N_CLIENTS"])

    agent = ClientAgent(cfg)

    if args.mode == "join":
        print(agent.join())
    elif args.mode == "leave":
        print(agent.leave())
    elif args.mode == "infer":
        agent.join()
        print(agent.infer())
    else:
        agent.join()
        print(agent.run_rounds(args.rounds))


if __name__ == "__main__":
    _main()
