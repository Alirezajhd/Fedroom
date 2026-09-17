from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = os.environ.get("FEDROOM_HOST", "0.0.0.0")
    port: int = int(os.environ.get("FEDROOM_PORT", "8000"))
    tick_interval_seconds: float = float(os.environ.get("FEDROOM_TICK_INTERVAL", "2.0"))
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./data/fedroom.db")
    checkpoint_backend: str = os.environ.get("CHECKPOINT_BACKEND", "local")
    mlflow_tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    max_update_norm_default: float | None = (
        float(os.environ["FEDROOM_MAX_UPDATE_NORM"])
        if os.environ.get("FEDROOM_MAX_UPDATE_NORM")
        else None
    )


settings = Settings()
