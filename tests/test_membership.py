import numpy as np
import pytest

from coordinator.aggregation import ModelContract
from coordinator.roommanager import AggregationConfig, RoomManager


class FakeClock:
    """Injectable clock so round-timeout logic is deterministic in tests."""

    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_manager(clock=None, checkpoints=None, metrics=None):
    checkpoints = checkpoints if checkpoints is not None else []
    metrics = metrics if metrics is not None else []

    def on_checkpoint(room, version, state):
        uri = f"mem://{room.room_id}/v{version}"
        checkpoints.append((version, uri))
        return uri

    def on_metrics(room, summary):
        metrics.append(summary)

    return RoomManager(clock=clock or FakeClock(), on_checkpoint=on_checkpoint, on_metrics=on_metrics)


def make_contract():
    return ModelContract(model_id="tiny", framework="numpy", param_shapes={"w": (2,)})


def test_single_client_room_behaves_like_managed_local_training():
    """'A room with one active client behaves like managed local training
    with checkpoint versioning.'"""
    mgr = make_manager()
    contract = make_contract()
    mgr.create_room(
        "room1", contract, "preproc-v1", AggregationConfig(min_available_clients=1, min_fit_clients=1),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room1")
    mgr.join_client("room1", "alice", {})
    rnd = mgr.start_round("room1")
    assert rnd.selected == ["alice"]

    mgr.submit_update("room1", "alice", {"w": np.array([2.0, 4.0])}, n_samples=10, base_version=0)
    summary = mgr.maybe_finalize_round("room1")
    assert summary["status"] == "aggregated"
    assert summary["new_version"] == 1

    room = mgr.get_room("room1")
    np.testing.assert_allclose(room.global_state["w"], [2.0, 4.0])
    assert len(room.checkpoints) == 1


def test_multi_client_weighted_aggregation():
    mgr = make_manager()
    contract = make_contract()
    mgr.create_room(
        "room2", contract, "preproc-v1", AggregationConfig(min_available_clients=2, quorum=1.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room2")
    mgr.join_client("room2", "alice", {})
    mgr.join_client("room2", "bob", {})
    mgr.start_round("room2")

    mgr.submit_update("room2", "alice", {"w": np.array([1.0, 1.0])}, n_samples=100, base_version=0)
    mgr.submit_update("room2", "bob", {"w": np.array([5.0, 5.0])}, n_samples=300, base_version=0)
    summary = mgr.maybe_finalize_round("room2")

    assert summary["status"] == "aggregated"
    room = mgr.get_room("room2")
    np.testing.assert_allclose(room.global_state["w"], [4.0, 4.0])  # 0.25*[1,1] + 0.75*[5,5]


def test_client_joining_mid_round_is_eligible_next_round_not_current():
    clock = FakeClock()
    mgr = make_manager(clock=clock)
    contract = make_contract()
    mgr.create_room(
        "room3", contract, "p1", AggregationConfig(min_available_clients=1, quorum=1.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room3")
    mgr.join_client("room3", "alice", {})
    rnd = mgr.start_round("room3")  # round 1

    # A second client joins WHILE round 1 is active.
    rec = mgr.join_client("room3", "bob", {})
    assert rec.eligible_from_round == 2
    assert "bob" not in rnd.selected

    mgr.submit_update("room3", "alice", {"w": np.array([1.0, 1.0])}, n_samples=1, base_version=0)
    mgr.maybe_finalize_round("room3")

    rnd2 = mgr.start_round("room3")  # round 2
    assert set(rnd2.selected) == {"alice", "bob"}


def test_dropout_via_timeout_with_quorum_met_still_aggregates():
    clock = FakeClock()
    mgr = make_manager(clock=clock)
    contract = make_contract()
    mgr.create_room(
        "room4", contract, "p1",
        AggregationConfig(min_available_clients=2, quorum=0.5, round_timeout_seconds=30.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room4")
    mgr.join_client("room4", "alice", {})
    mgr.join_client("room4", "bob", {})
    mgr.start_round("room4")

    # Only alice responds. Bob is a straggler/dropout.
    mgr.submit_update("room4", "alice", {"w": np.array([2.0, 2.0])}, n_samples=5, base_version=0)

    # Not ready yet (bob hasn't responded, timeout hasn't elapsed).
    assert mgr.maybe_finalize_round("room4") is None

    clock.advance(31.0)  # exceed round_timeout_seconds
    summary = mgr.maybe_finalize_round("room4")
    assert summary["status"] == "aggregated"
    assert summary["n_dropped"] == 1
    assert mgr.get_room("room4").clients["bob"].status.value == "dropped"


def test_round_fails_quorum_when_too_many_drop():
    clock = FakeClock()
    mgr = make_manager(clock=clock)
    contract = make_contract()
    mgr.create_room(
        "room5", contract, "p1",
        AggregationConfig(min_available_clients=2, quorum=1.0, round_timeout_seconds=10.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room5")
    mgr.join_client("room5", "alice", {})
    mgr.join_client("room5", "bob", {})
    mgr.start_round("room5")
    mgr.submit_update("room5", "alice", {"w": np.array([1.0, 1.0])}, n_samples=1, base_version=0)

    clock.advance(11.0)
    summary = mgr.maybe_finalize_round("room5")
    assert summary["status"] == "failed_quorum"
    room = mgr.get_room("room5")
    assert room.current_version == 0  # unchanged
    assert room.active_round is None  # room can be re-tried


def test_stale_update_is_rejected():
    mgr = make_manager()
    contract = make_contract()
    mgr.create_room(
        "room6", contract, "p1", AggregationConfig(min_available_clients=1),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room6")
    mgr.join_client("room6", "alice", {})
    mgr.start_round("room6")

    with pytest.raises(ValueError, match="Update trained from version"):
        mgr.submit_update("room6", "alice", {"w": np.array([1.0, 1.0])}, n_samples=1, base_version=99)

    assert mgr.get_room("room6").clients["alice"].status.value == "failed"


def test_incompatible_shape_update_is_rejected_and_room_continues():
    clock = FakeClock()
    mgr = make_manager(clock=clock)
    contract = make_contract()
    mgr.create_room(
        "room7", contract, "p1",
        AggregationConfig(min_available_clients=2, quorum=0.5, round_timeout_seconds=5.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room7")
    mgr.join_client("room7", "alice", {})
    mgr.join_client("room7", "bob", {})
    mgr.start_round("room7")

    with pytest.raises(Exception):
        mgr.submit_update("room7", "alice", {"w": np.array([1.0, 1.0, 1.0])}, n_samples=5, base_version=0)

    # Bob is still fine and the room does not crash.
    mgr.submit_update("room7", "bob", {"w": np.array([3.0, 3.0])}, n_samples=5, base_version=0)
    summary = mgr.maybe_finalize_round("room7")
    assert summary["status"] == "aggregated"
    assert summary["n_rejected"] == 1
    np.testing.assert_allclose(mgr.get_room("room7").global_state["w"], [3.0, 3.0])


def test_client_leave_does_not_crash_active_round():
    clock = FakeClock()
    mgr = make_manager(clock=clock)
    contract = make_contract()
    mgr.create_room(
        "room8", contract, "p1",
        AggregationConfig(min_available_clients=2, quorum=0.5, round_timeout_seconds=5.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("room8")
    mgr.join_client("room8", "alice", {})
    mgr.join_client("room8", "bob", {})
    mgr.start_round("room8")

    mgr.leave_client("room8", "bob")  # bob leaves mid-round without submitting
    mgr.submit_update("room8", "alice", {"w": np.array([9.0, 9.0])}, n_samples=1, base_version=0)

    clock.advance(6.0)
    summary = mgr.maybe_finalize_round("room8")
    assert summary["status"] == "aggregated"
    assert mgr.get_room("room8").clients["bob"].status.value == "dropped"
