"""
Experiment tracking wrapper around MLflow.

Falls back to an append-only local JSONL file when MLflow is not installed
or no tracking server is reachable, so the platform degrades gracefully
rather than crashing a training round just because observability
infrastructure is temporarily unavailable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


class _JsonlFallbackTracker:
    def __init__(self, path: str = "./data/mlflow_fallback.jsonl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._active_run = None

    def start_run(self, experiment: str, run_name: str) -> str:
        self._active_run = f"{experiment}:{run_name}:{time.time()}"
        self._write({"event": "start_run", "run": self._active_run, "experiment": experiment})
        return self._active_run

    def log_params(self, params: dict) -> None:
        self._write({"event": "params", "run": self._active_run, "data": params})

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        self._write({"event": "metrics", "run": self._active_run, "step": step, "data": metrics})

    def end_run(self) -> None:
        self._write({"event": "end_run", "run": self._active_run})
        self._active_run = None

    def _write(self, record: dict) -> None:
        record["ts"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class MlflowTracker:
    """Thin wrapper. Nested runs: one parent per room, one child per round."""

    def __init__(self, tracking_uri: Optional[str] = None):
        try:
            import mlflow  # noqa: F401
            self._mlflow = __import__("mlflow")
            if tracking_uri:
                self._mlflow.set_tracking_uri(tracking_uri)
            self._fallback = None
        except Exception:
            self._mlflow = None
            self._fallback = _JsonlFallbackTracker()

    def start_run(self, experiment: str, run_name: str):
        if self._mlflow is not None:
            try:
                self._mlflow.set_experiment(experiment)
                run = self._mlflow.start_run(run_name=run_name)
                return run.info.run_id
            except Exception:
                self._mlflow = None
                self._fallback = _JsonlFallbackTracker()
        return self._fallback.start_run(experiment, run_name)

    def log_params(self, params: dict) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
                return
            except Exception:
                pass
        self._fallback.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(metrics, step=step)
                return
            except Exception:
                pass
        self._fallback.log_metrics(metrics, step=step)

    def end_run(self) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
                return
            except Exception:
                pass
        if self._fallback is not None:
            self._fallback.end_run()


def build_tracker() -> MlflowTracker:
    return MlflowTracker(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
