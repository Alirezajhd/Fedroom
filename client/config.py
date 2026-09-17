from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ClientConfig:
    client_id: str
    coordinator_url: str = "http://localhost:8000"
    room_id: str = "fashion-room"
    data_source: str = "local"           # "local" | "s3"
    partition_scheme: str = "iid"        # "iid" | "non_iid"
    n_clients: int = 3
    samples_per_client: Optional[int] = None
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    local_data_root: str = "./data/raw"
    local_epochs: int = 1
    learning_rate: float = 0.01
    batch_size: int = 32
    poll_interval_seconds: float = 2.0
    capabilities: dict = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "ClientConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return ClientConfig(**raw)
