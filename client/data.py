"""
Client-side data access.

Supports two source types, per the "local filesystem and S3-compatible data
sources" requirement:

  - "local": Fashion-MNIST/MNIST downloaded via torchvision to a local path.
  - "s3":    a pre-partitioned .npz shard (images/labels) pulled from a
             MinIO/S3-compatible bucket that only *this* client can access.

Raw samples never leave this module / this process: only the partition is
read here, and only gradients/parameters are ever returned to the caller.

If torchvision/torch or network access to the dataset mirror is unavailable
(e.g. in an offline CI sandbox), a small deterministic synthetic dataset is
generated instead so the rest of the pipeline (partitioning, training loop,
submission) can still be exercised end-to-end.

CACHING (important for correctness under concurrency, not just speed):
`partition_for_client` -- and therefore `_load_full_dataset` /
`_load_s3_shard` -- is called fresh every training round by every client.
Without caching, `_load_full_dataset` used to re-decode and re-stack all
60,000 Fashion-MNIST images via a per-sample Python loop on EVERY call. For
a single client this alone costs several seconds per round; run several
simulated clients concurrently (as `experiments/run_scalability.py` and
`experiments/run_noniid.py` do via a `ThreadPoolExecutor`) and this
CPU-bound, pure-Python, GIL-bound loop does not parallelize -- it roughly
serializes across threads, so wall-clock time per round scaled with the
number of concurrent clients (observed: ~1 client ~7-10s/round, ~2 clients
~34s/round, ~4 clients ~170s/round). Past whatever `round_timeout_seconds`
is configured, this made every client's submission arrive after the round
had already failed quorum (0 responses) -- "No round is currently
accepting updates" for every client, every round, at 4+ concurrent
clients. The fix below caches the loaded array (keyed by source) so the
expensive load happens at most once per process, and avoids the slow
per-sample loop entirely by reading torchvision's underlying tensor
directly instead of iterating `dataset[i]` 60,000 times.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Partition:
    images: np.ndarray  # (N, 1, 28, 28) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64


_dataset_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_dataset_cache_lock = threading.Lock()


def _synthetic_dataset(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, size=n)
    images = np.zeros((n, 1, 28, 28), dtype=np.float32)
    for i, lbl in enumerate(labels):
        # Deterministic, label-correlated pattern so a model can actually
        # learn something above chance -- useful for a fast smoke test.
        base = rng.normal(loc=lbl / 10.0, scale=0.05, size=(28, 28)).astype(np.float32)
        images[i, 0] = np.clip(base, 0.0, 1.0)
    return images, labels.astype(np.int64)


def _load_full_dataset_uncached(local_root: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import torch  # noqa: F401
        from torchvision import datasets

        # Downloads (if needed) and loads the dataset, but reads the
        # underlying (60000, 28, 28) uint8 tensor and (60000,) label tensor
        # directly -- NOT via `dataset[i]` in a Python loop, which was the
        # actual bottleneck (see module docstring). This produces the exact
        # same values as the previous `transforms.ToTensor()` per-sample
        # path (uint8 [0,255] -> float32 [0,1], with a channel dimension
        # added), just ~60,000x fewer Python-level operations.
        ds = datasets.FashionMNIST(root=local_root, train=True, download=True)
        images = (ds.data.numpy().astype(np.float32) / 255.0)[:, np.newaxis, :, :]
        labels = ds.targets.numpy().astype(np.int64)
        return images, labels
    except Exception:
        # Offline / no torchvision available: fall back to synthetic data so
        # the platform remains runnable end-to-end.
        return _synthetic_dataset(n=6000, seed=0)


def _load_full_dataset(local_root: str) -> Tuple[np.ndarray, np.ndarray]:
    cache_key = f"local:{os.path.abspath(local_root)}"
    cached = _dataset_cache.get(cache_key)
    if cached is not None:
        return cached
    with _dataset_cache_lock:
        # Re-check inside the lock: another thread may have finished
        # loading while we were waiting for it (avoids a "thundering herd"
        # of concurrent clients all paying the load cost on their first
        # round -- only one does, the rest wait briefly then hit the cache).
        cached = _dataset_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _load_full_dataset_uncached(local_root)
        _dataset_cache[cache_key] = result
        return result


def _load_s3_shard_uncached(bucket: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )
    obj = client.get_object(Bucket=bucket, Key=key)
    import io

    with np.load(io.BytesIO(obj["Body"].read())) as npz:
        return npz["images"].astype(np.float32), npz["labels"].astype(np.int64)


def _load_s3_shard(bucket: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    cache_key = f"s3:{bucket}/{key}"
    cached = _dataset_cache.get(cache_key)
    if cached is not None:
        return cached
    with _dataset_cache_lock:
        cached = _dataset_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _load_s3_shard_uncached(bucket, key)
        _dataset_cache[cache_key] = result
        return result


def partition_for_client(
    client_id: str,
    n_clients: int,
    scheme: str = "iid",
    source: str = "local",
    local_root: str = "./data/raw",
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    samples_per_client: Optional[int] = None,
) -> Partition:
    """Return this client's private local partition. Never returns other
    clients' data, and this function's output must never be sent to the
    coordinator -- only trained parameters are.
    """
    if source == "s3":
        if not (s3_bucket and s3_key):
            raise ValueError("source='s3' requires s3_bucket and s3_key")
        images, labels = _load_s3_shard(s3_bucket, s3_key)
    else:
        images, labels = _load_full_dataset(local_root)

    # Deterministic client index from client_id so re-joining a room yields
    # the same partition (useful for reproducibility across restarts).
    idx = int(hashlib.sha256(client_id.encode()).hexdigest(), 16) % max(n_clients, 1)

    if scheme == "non_iid":
        # Sort-and-shard by label (as in the original FedAvg paper): give
        # each client a small number of contiguous label shards.
        order = np.argsort(labels, kind="stable")
        images, labels = images[order], labels[order]
        shards_per_client = max(1, 2)
        n_shards = n_clients * shards_per_client
        shard_size = max(1, len(labels) // n_shards)
        shard_ids = list(
            range(idx * shards_per_client, idx * shards_per_client + shards_per_client)
        )
        picked_idx = []
        for s in shard_ids:
            start, end = s * shard_size, min((s + 1) * shard_size, len(labels))
            picked_idx.extend(range(start, end))
        picked_idx = np.array(picked_idx, dtype=int)
    else:
        # IID: deterministic pseudo-random shuffle, then contiguous split.
        rng = np.random.default_rng(42)
        order = rng.permutation(len(labels))
        split = np.array_split(order, n_clients)
        picked_idx = split[idx % len(split)]

    if samples_per_client is not None and len(picked_idx) > samples_per_client:
        picked_idx = picked_idx[:samples_per_client]

    return Partition(images=images[picked_idx], labels=labels[picked_idx])
