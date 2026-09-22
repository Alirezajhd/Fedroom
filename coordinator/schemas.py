from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelContractIn(BaseModel):
    model_id: str
    framework: str
    param_shapes: Dict[str, List[int]]
    max_update_norm: Optional[float] = None


class AggregationConfigIn(BaseModel):
    strategy: str = "fedavg"
    min_available_clients: int = 1
    min_fit_clients: int = 1
    quorum: float = 1.0
    round_timeout_seconds: float = 500.0
    byzantine_f: int = 0


class RoomCreateRequest(BaseModel):
    room_id: str
    model_contract: ModelContractIn
    preprocessing_contract: str
    aggregation: AggregationConfigIn
    target_rounds: int = 1
    initial_state_b64: Dict[str, str] = Field(
        ..., description="param name -> base64-encoded .npy bytes for the initial global model"
    )


class JoinRequest(BaseModel):
    client_id: str
    capabilities: dict = {}


class LeaveRequest(BaseModel):
    client_id: str


class StartTrainingRequest(BaseModel):
    rounds: Optional[int] = None


class UpdateSubmitRequest(BaseModel):
    client_id: str
    base_version: int
    n_samples: int
    state_b64: Dict[str, str] = Field(..., description="param name -> base64-encoded .npy bytes")
    metrics: dict = {}


class InferenceLogRequest(BaseModel):
    client_id: str
    version: int
    note: str = ""
