"""
Checkpoint storage backend.

Serializes a StateDict (name -> np.ndarray) to a single .npz blob and pushes
it to an S3-compatible bucket (MinIO in the baseline deployment, or real
AWS S3 / any compatible provider). A local-filesystem fallback is provided so
the platform still runs (e.g. for unit/integration tests) without object
storage configured -- this satisfies R1 (reproducible baseline) without
forcing every developer to run MinIO just to run the test suite.

Raw client training data is NEVER written through this module; only
aggregated global-model checkpoints and (client-side, opt-in) client
datasets flow through S3-compatible storage.
"""
from __future__ import annotations

import io
import os
from typing import Dict

import numpy as np

StateDict = Dict[str, np.ndarray]


def serialize_state(state: StateDict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **state)
    return buf.getvalue()


def deserialize_state(blob: bytes) -> StateDict:
    buf = io.BytesIO(blob)
    with np.load(buf) as data:
        return {k: data[k] for k in data.files}


class LocalCheckpointStore:
    """Filesystem-backed store. Used by default / in tests."""

    def __init__(self, base_dir: str = "./data/checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, room_id: str, version: int, state: StateDict) -> str:
        room_dir = os.path.join(self.base_dir, room_id)
        os.makedirs(room_dir, exist_ok=True)
        path = os.path.join(room_dir, f"v{version:05d}.npz")
        with open(path, "wb") as f:
            f.write(serialize_state(state))
        return f"file://{os.path.abspath(path)}"

    def load(self, uri: str) -> StateDict:
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        with open(path, "rb") as f:
            return deserialize_state(f.read())


class S3CheckpointStore:
    """MinIO / S3-compatible object storage backend (boto3)."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ):
        import boto3  # local import: keeps boto3 optional for local-only runs

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=access_key or os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=secret_key or os.environ.get("S3_SECRET_KEY"),
            region_name=region,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def save(self, room_id: str, version: int, state: StateDict) -> str:
        key = f"{room_id}/v{version:05d}.npz"
        blob = serialize_state(state)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=blob)
        return f"s3://{self.bucket}/{key}"

    def load(self, uri: str) -> StateDict:
        assert uri.startswith("s3://")
        _, _, rest = uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        obj = self.client.get_object(Bucket=bucket, Key=key)
        return deserialize_state(obj["Body"].read())


def build_checkpoint_store():
    """Factory driven by environment variables (see .env.example)."""
    backend = os.environ.get("CHECKPOINT_BACKEND", "local").lower()
    if backend == "s3":
        return S3CheckpointStore(
            bucket=os.environ.get("S3_CHECKPOINT_BUCKET", "flaas-models"),
        )
    return LocalCheckpointStore(base_dir=os.environ.get("LOCAL_CHECKPOINT_DIR", "./data/checkpoints"))
