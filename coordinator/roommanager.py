"""
RoomManager: the control-plane brain of Fedroom.

Deliberately decoupled from FastAPI/SQLAlchemy/MinIO/MLflow so that the
membership/round state machine (the trickiest part of the assignment) can be
unit-tested deterministically with an injected clock, in-memory, with no
network or database involved. `coordinator/app.py` wires this class to HTTP
endpoints and to the storage/tracking backends.
"""

from __future__ import annotations

import enum
import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from coordinator.aggregation import (
    ClientUpdate,
    ContractError,
    ModelContract,
    NonFiniteUpdateError,
    OversizedUpdateError,
    StaleUpdateError,
    StateDict,
    validate_update,
)
from strategies.robust import get_strategy


# --------------------------------------------------------------------------- #
# Enums / small value objects
# --------------------------------------------------------------------------- #
class ClientStatus(str, enum.Enum):
    JOINED = "joined"  # registered, not yet eligible for this round
    ELIGIBLE = "eligible"  # may be selected starting next round
    SELECTED = "selected"  # chosen for the in-flight round
    COMPLETED = "completed"  # submitted a valid update this round
    FAILED = "failed"  # submitted an invalid update
    DROPPED = "dropped"  # selected but did not respond before timeout
    LEFT = "left"  # explicitly left the room


class RoundStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED_QUORUM = "failed_quorum"


class RoomState(str, enum.Enum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class AggregationConfig:
    strategy: str = "fedavg"
    min_available_clients: int = 1
    min_fit_clients: int = 1
    quorum: float = 1.0  # fraction of *selected* clients required
    round_timeout_seconds: float = 120.0
    byzantine_f: int = 0  # assumed max malicious clients per round (robust strategies)


@dataclass
class ClientRecord:
    client_id: str
    capabilities: dict
    status: ClientStatus
    eligible_from_round: int
    joined_at: float


@dataclass
class RoundState:
    round_number: int
    expected_version: int
    selected: List[str]
    started_at: float
    timeout_seconds: float
    submissions: Dict[str, ClientUpdate] = field(default_factory=dict)
    rejections: Dict[str, str] = field(default_factory=dict)
    # Per-client metrics reported alongside each update: local_training_seconds,
    # download_seconds, upload_seconds, payload_bytes, train_loss,
    # train_accuracy, pretrain_eval_loss, pretrain_eval_accuracy (all optional
    # -- see client/agent.py). Kept here (not just in the SQL audit log) so
    # `maybe_finalize_round` can aggregate them into the round summary that
    # gets logged to MLflow, satisfying the assignment's "Timing" and
    # "Client" metric-group requirements.
    client_metrics: Dict[str, dict] = field(default_factory=dict)
    status: RoundStatus = RoundStatus.RUNNING


@dataclass
class CheckpointRecord:
    version: int
    uri: str
    n_clients: int
    created_at: float


@dataclass
class Room:
    room_id: str
    contract: ModelContract
    preprocessing_contract: str
    agg_config: AggregationConfig
    target_rounds: int
    state: RoomState = RoomState.CREATED
    current_round: int = 0
    current_version: int = 0
    global_state: Optional[StateDict] = None
    clients: Dict[str, ClientRecord] = field(default_factory=dict)
    active_round: Optional[RoundState] = None
    checkpoints: List[CheckpointRecord] = field(default_factory=list)
    audit_log: List[dict] = field(default_factory=list)
    metrics_history: List[dict] = field(default_factory=list)

    def log(self, event: str, **kwargs) -> None:
        self.audit_log.append({"ts": time.time(), "event": event, **kwargs})


# --------------------------------------------------------------------------- #
# RoomManager
# --------------------------------------------------------------------------- #
def _summarize_client_metrics(client_metrics: Dict[str, dict]) -> dict:
    """Average the optional per-client metrics reported alongside each
    update into round-level numbers for MLflow/the dashboard.

    Reported keys (all optional -- see client/agent.py):
      - selection_wait_seconds, local_training_seconds, download_seconds,
        upload_seconds: timing (upload_seconds is measured server-side --
        see coordinator/app.py::submit_update -- since a client cannot know
        its own upload duration before the request finishes sending)
      - payload_bytes: size of the update sent over the wire
      - train_loss, train_accuracy: this client's own post-training metrics
      - pretrain_eval_loss, pretrain_eval_accuracy: this client's evaluation
        of the CURRENT global model (before local training) on its own
        held-out split -- a federated-evaluation proxy for "global task
        metric", since the coordinator never holds a labeled validation set
        itself (that would mean it holds training-style data, which
        contradicts the platform's data-locality design). Averaged (mean,
        not weighted -- these are per-client quality signals, not
        contributions to the model) across responding clients, this
        approximates how well the *previous* global checkpoint generalizes.
    """
    numeric_keys = [
        "local_training_seconds",
        "download_seconds",
        "upload_seconds",
        "selection_wait_seconds",
        "train_loss",
        "train_accuracy",
        "pretrain_eval_loss",
        "pretrain_eval_accuracy",
    ]
    out: dict = {}
    for key in numeric_keys:
        values = [
            m[key]
            for m in client_metrics.values()
            if isinstance(m, dict) and m.get(key) is not None
        ]
        if values:
            out[f"avg_{key}"] = sum(values) / len(values)
    payload_values = [
        m["payload_bytes"]
        for m in client_metrics.values()
        if isinstance(m, dict) and m.get("payload_bytes") is not None
    ]
    if payload_values:
        out["total_payload_bytes"] = sum(payload_values)
        out["avg_payload_bytes"] = sum(payload_values) / len(payload_values)
    return out


class RoomManager:
    """Thread-safe, in-process registry of rooms.

    `on_checkpoint` / `on_metrics` are optional callbacks the caller (app.py)
    wires up to the storage and tracking backends, keeping this class free of
    infra dependencies.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        on_checkpoint: Optional[Callable[[Room, int, StateDict], str]] = None,
        on_metrics: Optional[Callable[[Room, dict], None]] = None,
    ):
        self._rooms: Dict[str, Room] = {}
        self._lock = threading.RLock()
        self._clock = clock
        self._on_checkpoint = on_checkpoint
        self._on_metrics = on_metrics

    # ---- room lifecycle -------------------------------------------------- #
    def create_room(
        self,
        room_id: str,
        contract: ModelContract,
        preprocessing_contract: str,
        agg_config: AggregationConfig,
        initial_state: StateDict,
        target_rounds: int = 1,
    ) -> Room:
        with self._lock:
            if room_id in self._rooms:
                raise ValueError(f"Room '{room_id}' already exists")
            room = Room(
                room_id=room_id,
                contract=contract,
                preprocessing_contract=preprocessing_contract,
                agg_config=agg_config,
                target_rounds=target_rounds,
                global_state=initial_state,
            )
            room.log("room_created")
            self._rooms[room_id] = room
            return room

    def list_rooms(self) -> List[Room]:
        with self._lock:
            return list(self._rooms.values())

    def get_room(self, room_id: str) -> Room:
        with self._lock:
            if room_id not in self._rooms:
                raise KeyError(f"Room '{room_id}' not found")
            return self._rooms[room_id]

    def start_room(self, room_id: str) -> Room:
        with self._lock:
            room = self.get_room(room_id)
            room.state = RoomState.ACTIVE
            room.log("room_started")
            return room

    def stop_room(self, room_id: str) -> Room:
        with self._lock:
            room = self.get_room(room_id)
            room.state = RoomState.STOPPED
            room.log("room_stopped")
            return room

    # ---- client membership ------------------------------------------------ #
    def join_client(
        self, room_id: str, client_id: str, capabilities: dict
    ) -> ClientRecord:
        """A client joining during round r becomes eligible no earlier than r+1.

        Idempotent for an already-known, still-participating client: a
        client re-announcing itself (a reconnecting agent, a CLI
        convenience call before training, retried network requests, etc.)
        must NOT be treated as brand new. Doing so used to reset the
        client's status/eligibility unconditionally -- including while it
        was SELECTED in an in-flight round -- which silently knocked it
        back out of that round's selection and left callers polling for a
        "selected" status that would never come back (the round would then
        time out with zero responses). Only a client who explicitly LEFT
        gets a fresh record on rejoin.
        """
        with self._lock:
            room = self.get_room(room_id)
            existing = room.clients.get(client_id)
            if existing is not None and existing.status != ClientStatus.LEFT:
                existing.capabilities = capabilities
                room.log(
                    "client_rejoined", client_id=client_id, status=existing.status.value
                )
                return existing

            # A client joining while round r is in flight becomes eligible no
            # earlier than r+1; joining with no round active makes it
            # eligible for the very next round to be started.
            if room.active_round is not None:
                eligible_from = room.active_round.round_number + 1
            else:
                eligible_from = room.current_round + 1
            record = ClientRecord(
                client_id=client_id,
                capabilities=capabilities,
                status=ClientStatus.ELIGIBLE
                if eligible_from <= room.current_round + 1
                else ClientStatus.JOINED,
                eligible_from_round=eligible_from,
                joined_at=self._clock(),
            )
            room.clients[client_id] = record
            room.log(
                "client_joined", client_id=client_id, eligible_from_round=eligible_from
            )
            return record

    def leave_client(self, room_id: str, client_id: str) -> None:
        with self._lock:
            room = self.get_room(room_id)
            if client_id not in room.clients:
                raise KeyError(
                    f"Client '{client_id}' is not a member of room '{room_id}'"
                )
            record = room.clients[client_id]
            was_selected = (
                room.active_round is not None
                and client_id in room.active_round.selected
                and client_id not in room.active_round.submissions
            )
            record.status = ClientStatus.LEFT
            room.log(
                "client_left", client_id=client_id, was_selected_unfinished=was_selected
            )
            # Dropout accounting happens lazily at finalize-time via timeout,
            # so we do not force-fail the round here; this keeps behavior
            # identical whether a client stops cleanly or simply stops
            # responding (crash), matching the "fails safely" requirement.

    # ---- round lifecycle --------------------------------------------------- #
    def start_round(self, room_id: str) -> RoundState:
        with self._lock:
            room = self.get_room(room_id)
            if room.state != RoomState.ACTIVE:
                raise ValueError(f"Room '{room_id}' is not active (state={room.state})")
            if (
                room.active_round is not None
                and room.active_round.status == RoundStatus.RUNNING
            ):
                raise ValueError("A round is already in progress")

            # Promote clients whose eligibility has kicked in. A client may
            # be re-selected across rounds regardless of its status from the
            # *previous* round (completed/failed/dropped are historical
            # outcomes, not a ban) -- only an explicit LEFT excludes it.
            eligible = []
            for c in room.clients.values():
                if c.status == ClientStatus.LEFT:
                    continue
                if c.eligible_from_round <= room.current_round + 1:
                    c.status = ClientStatus.ELIGIBLE
                    eligible.append(c.client_id)

            if len(eligible) < room.agg_config.min_available_clients:
                raise ValueError(
                    f"Only {len(eligible)} eligible client(s); "
                    f"need >= {room.agg_config.min_available_clients}"
                )

            selected = eligible  # simple strategy: select all eligible clients
            for cid in selected:
                room.clients[cid].status = ClientStatus.SELECTED

            round_number = room.current_round + 1
            round_state = RoundState(
                round_number=round_number,
                expected_version=room.current_version,
                selected=selected,
                started_at=self._clock(),
                timeout_seconds=room.agg_config.round_timeout_seconds,
            )
            room.active_round = round_state
            room.log("round_started", round=round_number, selected=selected)
            return round_state

    def submit_update(
        self,
        room_id: str,
        client_id: str,
        state: StateDict,
        n_samples: int,
        base_version: int,
        client_metrics: Optional[dict] = None,
    ) -> None:
        with self._lock:
            room = self.get_room(room_id)
            rnd = room.active_round
            if rnd is None or rnd.status != RoundStatus.RUNNING:
                raise ValueError("No round is currently accepting updates")
            if client_id not in rnd.selected:
                raise PermissionError(
                    f"Client '{client_id}' was not selected for this round"
                )
            if client_id in rnd.submissions or client_id in rnd.rejections:
                raise ValueError(f"Client '{client_id}' already submitted this round")

            try:
                if base_version != rnd.expected_version:
                    raise StaleUpdateError(
                        f"Update trained from version {base_version}, "
                        f"round expects version {rnd.expected_version}"
                    )
                validate_update(state, room.contract)
                if n_samples <= 0:
                    raise ValueError("n_samples must be > 0")
            except (
                ContractError,
                NonFiniteUpdateError,
                OversizedUpdateError,
                StaleUpdateError,
                ValueError,
            ) as exc:
                rnd.rejections[client_id] = str(exc)
                room.clients[client_id].status = ClientStatus.FAILED
                room.log("update_rejected", client_id=client_id, reason=str(exc))
                raise

            rnd.submissions[client_id] = ClientUpdate(
                client_id=client_id,
                n_samples=n_samples,
                state=state,
                base_version=base_version,
            )
            rnd.client_metrics[client_id] = client_metrics or {}
            room.clients[client_id].status = ClientStatus.COMPLETED
            room.log("update_accepted", client_id=client_id, n_samples=n_samples)

    def _quorum_met(self, room: Room, rnd: RoundState) -> bool:
        if not rnd.selected:
            return False
        responded_valid = len(rnd.submissions)
        return (responded_valid / len(rnd.selected)) >= room.agg_config.quorum

    def _all_responded(self, rnd: RoundState) -> bool:
        return len(rnd.submissions) + len(rnd.rejections) >= len(rnd.selected)

    def maybe_finalize_round(self, room_id: str) -> Optional[dict]:
        """Aggregate if all selected clients have responded, or if quorum is
        met and the round timeout has elapsed. Returns a summary dict when a
        finalize action occurred, else None.
        """
        with self._lock:
            room = self.get_room(room_id)
            rnd = room.active_round
            if rnd is None or rnd.status != RoundStatus.RUNNING:
                return None

            elapsed = self._clock() - rnd.started_at
            timed_out = elapsed >= rnd.timeout_seconds
            ready = self._all_responded(rnd) or timed_out
            if not ready:
                return None

            # Anyone selected but silent by now is a dropout.
            dropped = [
                cid
                for cid in rnd.selected
                if cid not in rnd.submissions and cid not in rnd.rejections
            ]
            for cid in dropped:
                room.clients[cid].status = ClientStatus.DROPPED
                room.log("client_dropped", client_id=cid, round=rnd.round_number)

            if not self._quorum_met(room, rnd):
                rnd.status = RoundStatus.FAILED_QUORUM
                room.active_round = None
                room.log(
                    "round_failed_quorum",
                    round=rnd.round_number,
                    responded=len(rnd.submissions),
                    selected=len(rnd.selected),
                )
                return {
                    "status": "failed_quorum",
                    "round": rnd.round_number,
                    "dropped": dropped,
                }

            rnd.status = RoundStatus.AGGREGATING
            strategy = get_strategy(
                room.agg_config.strategy, byzantine_f=room.agg_config.byzantine_f
            )
            agg_t0 = time.perf_counter()
            new_state = strategy.aggregate(list(rnd.submissions.values()))
            aggregation_seconds = time.perf_counter() - agg_t0
            new_version = room.current_version + 1
            room.global_state = new_state
            room.current_version = new_version
            room.current_round = rnd.round_number

            checkpoint_uri = None
            if self._on_checkpoint is not None:
                checkpoint_uri = self._on_checkpoint(room, new_version, new_state)
            room.checkpoints.append(
                CheckpointRecord(
                    version=new_version,
                    uri=checkpoint_uri or f"memory://{room.room_id}/v{new_version}",
                    n_clients=len(rnd.submissions),
                    created_at=self._clock(),
                )
            )

            summary = {
                "status": "aggregated",
                "round": rnd.round_number,
                "new_version": new_version,
                "strategy": room.agg_config.strategy,
                "checkpoint_uri": checkpoint_uri
                or f"memory://{room.room_id}/v{new_version}",
                "n_completed": len(rnd.submissions),
                "n_selected": len(rnd.selected),
                "n_dropped": len(dropped),
                "n_rejected": len(rnd.rejections),
                "duration_seconds": elapsed,
                "aggregation_seconds": aggregation_seconds,
                **_summarize_client_metrics(rnd.client_metrics),
            }
            if self._on_metrics is not None:
                self._on_metrics(room, summary)
            room.metrics_history.append(summary)

            rnd.status = RoundStatus.COMPLETED
            room.active_round = None
            room.log("round_completed", **summary)
            return summary

    # ---- read-only status ---------------------------------------------- #
    def status(self, room_id: str) -> dict:
        with self._lock:
            room = self.get_room(room_id)
            rnd = room.active_round
            return {
                "room_id": room.room_id,
                "state": room.state.value,
                "current_round": room.current_round,
                "current_version": room.current_version,
                "target_rounds": room.target_rounds,
                "clients": {
                    cid: {
                        "status": c.status.value,
                        "eligible_from_round": c.eligible_from_round,
                    }
                    for cid, c in room.clients.items()
                },
                "active_round": None
                if rnd is None
                else {
                    "round_number": rnd.round_number,
                    "expected_version": rnd.expected_version,
                    "selected": rnd.selected,
                    "completed": list(rnd.submissions.keys()),
                    "rejected": rnd.rejections,
                    "status": rnd.status.value,
                    "elapsed_seconds": self._clock() - rnd.started_at,
                },
                "checkpoints": [
                    {"version": ck.version, "uri": ck.uri, "n_clients": ck.n_clients}
                    for ck in room.checkpoints
                ],
                "metrics_history": room.metrics_history,
            }
