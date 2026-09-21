"""
Fedroom coordinator service (control plane).

Exposes the REST API the TUI/CLI and client agents talk to. Business logic
lives in RoomManager (coordinator/roommanager.py); this module is glue:
HTTP <-> RoomManager <-> storage/tracking backends <-> SQLAlchemy audit log.

Run with:  uvicorn coordinator.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from coordinator.aggregation import ModelContract
from coordinator.codec import b64_to_state, state_to_b64
from coordinator.config import settings
from coordinator.db import build_sessionmaker
from coordinator.models import AuditLogORM, CheckpointORM, ClientEventORM, RoomORM, RoundEventORM
from coordinator.roommanager import AggregationConfig, ClientStatus, Room, RoomManager, RoomState
from coordinator.schemas import (
    InferenceLogRequest,
    JoinRequest,
    LeaveRequest,
    RoomCreateRequest,
    StartTrainingRequest,
    UpdateSubmitRequest,
)
from coordinator.storage import build_checkpoint_store
from coordinator.tracking import build_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fedroom.coordinator")

app = FastAPI(title="Fedroom Coordinator", version="1.0.0")

SessionLocal = build_sessionmaker()
checkpoint_store = build_checkpoint_store()
tracker = build_tracker()
_mlflow_run_ids: dict[str, str] = {}


def _persist_checkpoint(room: Room, version: int, state) -> str:
    uri = checkpoint_store.save(room.room_id, version, state)
    with SessionLocal() as session:
        session.add(
            CheckpointORM(room_id=room.room_id, version=version, uri=uri, n_clients=0)
        )
        session.commit()
    return uri


def _log_metrics(room: Room, summary: dict) -> None:
    run_id = _mlflow_run_ids.get(room.room_id)
    if run_id is None:
        run_id = tracker.start_run(experiment=f"fedroom-{room.room_id}", run_name=room.room_id)
        _mlflow_run_ids[room.room_id] = run_id
        tracker.log_params(
            {
                "model_id": room.contract.model_id,
                "framework": room.contract.framework,
                "strategy": room.agg_config.strategy,
                "quorum": room.agg_config.quorum,
                "round_timeout_seconds": room.agg_config.round_timeout_seconds,
                "byzantine_f": room.agg_config.byzantine_f,
            }
        )

    # --- Room/global metric group -----------------------------------
    metrics = {
        "completed_clients": summary["n_completed"],
        "selected_clients": summary["n_selected"],
        "dropped_clients": summary["n_dropped"],
        "rejected_clients": summary["n_rejected"],
        "version": summary["new_version"],
        # "Global task metric": the coordinator never holds a labeled
        # validation set of its own (it would then be doing centralized
        # training, contradicting the platform's data-locality design), so
        # this is a federated-evaluation ESTIMATE: the mean, across
        # responding clients, of each client's own evaluation of the model
        # it was just handed (before local training) on a held-out slice of
        # its own data. See RoomManager._summarize_client_metrics and
        # docs/EXPLAINER.md Section 8 for the full explanation and caveats.
        **({"global_accuracy_estimate": summary["avg_pretrain_eval_accuracy"]}
           if "avg_pretrain_eval_accuracy" in summary else {}),
        **({"global_loss_estimate": summary["avg_pretrain_eval_loss"]}
           if "avg_pretrain_eval_loss" in summary else {}),
        **({"avg_client_train_accuracy": summary["avg_train_accuracy"]}
           if "avg_train_accuracy" in summary else {}),
        **({"avg_client_train_loss": summary["avg_train_loss"]}
           if "avg_train_loss" in summary else {}),

        # --- Timing metric group -----------------------------------
        "round_duration_seconds": summary["duration_seconds"],
        "aggregation_seconds": summary["aggregation_seconds"],
        **({"avg_client_selection_wait_seconds": summary["avg_selection_wait_seconds"]}
           if "avg_selection_wait_seconds" in summary else {}),
        **({"avg_client_local_training_seconds": summary["avg_local_training_seconds"]}
           if "avg_local_training_seconds" in summary else {}),
        **({"avg_client_download_seconds": summary["avg_download_seconds"]}
           if "avg_download_seconds" in summary else {}),
        **({"avg_client_upload_seconds": summary["avg_upload_seconds"]}
           if "avg_upload_seconds" in summary else {}),

        # --- System metric group: payload size / network bytes -----
        **({"avg_update_payload_bytes": summary["avg_payload_bytes"]}
           if "avg_payload_bytes" in summary else {}),
        **({"total_update_payload_bytes": summary["total_payload_bytes"]}
           if "total_payload_bytes" in summary else {}),
    }

    # --- System metric group: coordinator process CPU/memory ---------
    # This measures the coordinator's OWN process, which is the one metric
    # the coordinator can observe directly without any extra infrastructure.
    # It is a *complement* to, not a replacement for, real per-pod CPU/
    # memory captured via `kubectl top pods` in the Kubernetes deployment
    # (see scripts/scale-experiment.sh and docs/report.md Section 5.4) --
    # that captures client pods too, which this cannot.
    try:
            import psutil
            metrics["system_cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["system_memory_mb"] = psutil.virtual_memory().used / (1024 * 1024)
    except ImportError:
        pass

    tracker.log_metrics(metrics, step=summary["round"])
    # Lineage: strategy/version/checkpoint URI as run tags (MLflow tags are
    # not step-indexed, so the numeric history above carries "version" per
    # round; the checkpoint URI's full per-version history lives in the SQL
    # CheckpointORM table / the dashboard's checkpoint list, which is the
    # more appropriate place for a non-numeric per-version artifact record).
    try:
        import mlflow

        mlflow.set_tag("latest_checkpoint_uri", summary["checkpoint_uri"])
        mlflow.set_tag("latest_version", str(summary["new_version"]))
    except Exception:
        pass

    with SessionLocal() as session:
        session.add(
            RoundEventORM(
                room_id=room.room_id,
                round_number=summary["round"],
                outcome=summary["status"],
                summary=summary,
            )
        )
        session.commit()
    logger.info("room=%s round=%s summary=%s", room.room_id, summary["round"], summary)


manager = RoomManager(clock=time.time, on_checkpoint=_persist_checkpoint, on_metrics=_log_metrics)

_background_stop = threading.Event()


def _background_finalizer_loop():
    while not _background_stop.is_set():
        for room in manager.list_rooms():
            try:
                manager.maybe_finalize_round(room.room_id)
            except Exception:  # noqa: BLE001 - keep the loop alive no matter what
                logger.exception("finalizer error for room %s", room.room_id)
        time.sleep(settings.tick_interval_seconds)


@app.on_event("startup")
def _on_startup():
    t = threading.Thread(target=_background_finalizer_loop, daemon=True)
    t.start()
    logger.info("Fedroom coordinator started; background finalizer running every %.1fs",
                settings.tick_interval_seconds)


@app.on_event("shutdown")
def _on_shutdown():
    _background_stop.set()


# --------------------------------------------------------------------------- #
# Room lifecycle
# --------------------------------------------------------------------------- #
@app.post("/rooms")
def create_room(req: RoomCreateRequest):
    try:
        contract = ModelContract(
            model_id=req.model_contract.model_id,
            framework=req.model_contract.framework,
            param_shapes={k: tuple(v) for k, v in req.model_contract.param_shapes.items()},
            max_update_norm=req.model_contract.max_update_norm,
        )
        agg_config = AggregationConfig(**req.aggregation.dict())
        initial_state = b64_to_state(req.initial_state_b64)
        room = manager.create_room(
            room_id=req.room_id,
            contract=contract,
            preprocessing_contract=req.preprocessing_contract,
            agg_config=agg_config,
            initial_state=initial_state,
            target_rounds=req.target_rounds,
        )
        with SessionLocal() as session:
            session.merge(
                RoomORM(
                    room_id=room.room_id,
                    model_contract=req.model_contract.dict(),
                    preprocessing_contract=room.preprocessing_contract,
                    agg_config=req.aggregation.dict(),
                    target_rounds=room.target_rounds,
                    state=room.state.value,
                )
            )
            session.commit()
        return {"room_id": room.room_id, "state": room.state.value, "version": room.current_version}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/rooms")
def list_rooms():
    return [manager.status(r.room_id) for r in manager.list_rooms()]


@app.get("/rooms/{room_id}")
def room_status(room_id: str):
    try:
        return manager.status(room_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/rooms/{room_id}/start")
def start_room(room_id: str, req: StartTrainingRequest):
    try:
        room = manager.start_room(room_id)
        if req.rounds is not None:
            room.target_rounds = req.rounds
        manager.start_round(room_id)
        return manager.status(room_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rooms/{room_id}/stop")
def stop_room(room_id: str):
    try:
        manager.stop_room(room_id)
        return manager.status(room_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/rooms/{room_id}/next-round")
def next_round(room_id: str):
    """Advance to a new round once the previous one has finalized.

    The background finalizer already aggregates finished rounds; this
    endpoint lets an operator (or the demo script) explicitly kick off the
    next round rather than waiting for a polling client agent to do it.
    """
    try:
        room = manager.get_room(room_id)
        if room.current_round >= room.target_rounds:
            return {"status": "target_rounds_reached", **manager.status(room_id)}
        rnd = manager.start_round(room_id)
        return {"status": "round_started", "round": rnd.round_number}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Client membership
# --------------------------------------------------------------------------- #
@app.post("/rooms/{room_id}/join")
def join(room_id: str, req: JoinRequest):
    try:
        record = manager.join_client(room_id, req.client_id, req.capabilities)
        with SessionLocal() as session:
            session.add(
                ClientEventORM(room_id=room_id, client_id=req.client_id, event="joined",
                                detail={"eligible_from_round": record.eligible_from_round})
            )
            session.commit()
        return {"client_id": req.client_id, "status": record.status.value,
                "eligible_from_round": record.eligible_from_round}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/rooms/{room_id}/leave")
def leave(room_id: str, req: LeaveRequest):
    try:
        manager.leave_client(room_id, req.client_id)
        with SessionLocal() as session:
            session.add(ClientEventORM(room_id=room_id, client_id=req.client_id, event="left"))
            session.commit()
        return {"client_id": req.client_id, "status": "left"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Model download / update submission (data plane <-> control plane boundary)
# --------------------------------------------------------------------------- #
@app.get("/rooms/{room_id}/model")
def get_model(room_id: str, version: Optional[str] = "latest"):
    try:
        room = manager.get_room(room_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if room.global_state is None:
        raise HTTPException(status_code=404, detail="Room has no model state yet")
    return {
        "version": room.current_version,
        "contract": {
            "model_id": room.contract.model_id,
            "framework": room.contract.framework,
            "param_shapes": {k: list(v) for k, v in room.contract.param_shapes.items()},
        },
        "state_b64": state_to_b64(room.global_state),
        "expected_round": (room.active_round.round_number if room.active_round else room.current_round + 1),
        "selected": (room.active_round.selected if room.active_round else []),
    }


@app.post("/rooms/{room_id}/updates")
def submit_update(room_id: str, req: UpdateSubmitRequest):
    request_t0 = time.perf_counter()
    try:
        state = b64_to_state(req.state_b64)
        # Server-side request-handling latency (decode + validate) is
        # measured here rather than client-side, because the client cannot
        # know its own upload duration before the request finishes sending
        # -- see client/agent.py::train_once for the client-side rationale.
        # This is what RoomManager averages into "avg_upload_seconds" per
        # round.
        client_metrics = dict(req.metrics)
        client_metrics["upload_seconds"] = time.perf_counter() - request_t0
        manager.submit_update(
            room_id=room_id,
            client_id=req.client_id,
            state=state,
            n_samples=req.n_samples,
            base_version=req.base_version,
            client_metrics=client_metrics,
        )
        with SessionLocal() as session:
            session.add(
                ClientEventORM(room_id=room_id, client_id=req.client_id, event="update_accepted",
                                detail={"n_samples": req.n_samples, "metrics": client_metrics})
            )
            session.commit()
        return {"status": "accepted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        with SessionLocal() as session:
            session.add(
                ClientEventORM(room_id=room_id, client_id=req.client_id, event="update_rejected",
                                detail={"reason": str(exc)})
            )
            session.commit()
        return JSONResponse(status_code=422, content={"status": "rejected", "reason": str(exc)})


@app.post("/rooms/{room_id}/tick")
def tick(room_id: str):
    """Manually trigger a finalize-check (the background thread also does this)."""
    result = manager.maybe_finalize_round(room_id)
    return {"result": result, **manager.status(room_id)}


@app.get("/rooms/{room_id}/checkpoints/{version}")
def get_checkpoint(room_id: str, version: int):
    try:
        room = manager.get_room(room_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = [c for c in room.checkpoints if c.version == version]
    if not matches:
        raise HTTPException(status_code=404, detail=f"No checkpoint v{version} for room '{room_id}'")
    ck = matches[-1]
    state = checkpoint_store.load(ck.uri)
    return {"version": ck.version, "uri": ck.uri, "state_b64": state_to_b64(state)}


@app.post("/rooms/{room_id}/inference-log")
def log_inference(room_id: str, req: InferenceLogRequest):
    with SessionLocal() as session:
        session.add(
            AuditLogORM(room_id=room_id, event="inference",
                        detail={"client_id": req.client_id, "version": req.version, "note": req.note})
        )
        session.commit()
    return {"status": "logged"}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "rooms": len(manager.list_rooms())}


@app.get("/system")
def system_metrics():
    """Live system-wide CPU/memory for the dashboard."""
    try:
        import psutil
        
        # Calculate memory in MB (used memory / 1024 / 1024)
        mem = psutil.virtual_memory()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),       # System-wide CPU %
            "memory_mb": mem.used / (1024 * 1024),                 # System-wide RAM used
            "available": True,
        }
    except ImportError:
        return {"available": False, "reason": "psutil not installed"}


# --------------------------------------------------------------------------- #
# Operational dashboard (bonus: "Web dashboard")
# --------------------------------------------------------------------------- #
# A single static HTML/JS file that polls the REST API above -- no build
# step, no framework, no external CDN dependency. Served at /dashboard so it
# never shadows any API route. Missing the directory (e.g. a coordinator-only
# checkout) degrades gracefully instead of crashing the API.
_dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")
    logger.info("Dashboard mounted at /dashboard (serving %s)", _dashboard_dir)
else:
    logger.info("No dashboard/ directory found at %s; /dashboard not mounted", _dashboard_dir)
