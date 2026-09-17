"""Small helpers to move numpy state dicts over JSON/HTTP as base64 .npy blobs.

Kept separate from storage.py (which persists whole-model .npz checkpoints)
because this is used for the smaller, per-request wire format between the
coordinator and client agents.
"""
from __future__ import annotations

import base64
import io
from typing import Dict

import numpy as np

StateDict = Dict[str, np.ndarray]


def array_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_array(s: str) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    buf = io.BytesIO(raw)
    return np.load(buf, allow_pickle=False)


def state_to_b64(state: StateDict) -> Dict[str, str]:
    return {k: array_to_b64(v) for k, v in state.items()}


def b64_to_state(payload: Dict[str, str]) -> StateDict:
    return {k: b64_to_array(v) for k, v in payload.items()}
