import numpy as np

from coordinator.aggregation import ModelContract
from coordinator.roommanager import AggregationConfig, RoomManager
from tests.test_membership import FakeClock, make_manager


def make_contract():
    return ModelContract(model_id="tiny", framework="numpy", param_shapes={"w": (2,)})


def test_round_summary_averages_client_metrics_and_skips_missing_values():
    """Mirrors the real client agent's payload shape (client/agent.py):
    some clients report full metrics, one is missing pretrain_eval fields
    (e.g. the torch-not-installed fallback path) -- the average must only
    be computed over clients that actually reported that field, and must
    not silently include None/NaN.
    """
    mgr = make_manager()
    contract = make_contract()
    mgr.create_room(
        "metrics-room", contract, "p1", AggregationConfig(min_available_clients=3, quorum=1.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("metrics-room")
    for cid in ["a", "b", "c"]:
        mgr.join_client("metrics-room", cid, {})
    mgr.start_round("metrics-room")

    mgr.submit_update("metrics-room", "a", {"w": np.array([1.0, 1.0])}, n_samples=10, base_version=0,
                       client_metrics={"local_training_seconds": 1.0, "train_accuracy": 0.8,
                                        "pretrain_eval_accuracy": 0.5, "payload_bytes": 1000})
    mgr.submit_update("metrics-room", "b", {"w": np.array([1.0, 1.0])}, n_samples=10, base_version=0,
                       client_metrics={"local_training_seconds": 3.0, "train_accuracy": 0.6,
                                        "pretrain_eval_accuracy": 0.7, "payload_bytes": 2000})
    # Client c: torch-not-installed fallback -- no train_accuracy/pretrain_eval at all.
    mgr.submit_update("metrics-room", "c", {"w": np.array([1.0, 1.0])}, n_samples=10, base_version=0,
                       client_metrics={"local_training_seconds": 0.1, "train_accuracy": None,
                                        "pretrain_eval_accuracy": None, "payload_bytes": 500})

    summary = mgr.maybe_finalize_round("metrics-room")

    assert summary["status"] == "aggregated"
    # Averaged only over clients that reported each field.
    assert summary["avg_local_training_seconds"] == pytest_approx((1.0 + 3.0 + 0.1) / 3)
    assert summary["avg_train_accuracy"] == pytest_approx((0.8 + 0.6) / 2)  # c excluded (None)
    assert summary["avg_pretrain_eval_accuracy"] == pytest_approx((0.5 + 0.7) / 2)
    assert summary["total_payload_bytes"] == 3500
    assert summary["avg_payload_bytes"] == pytest_approx(3500 / 3)
    # Real timing, not simulated -- just check it's a small positive number.
    assert summary["aggregation_seconds"] >= 0.0


def test_round_summary_has_no_metric_keys_when_nothing_reported():
    """A round where clients submit with no `client_metrics` at all (e.g. a
    bare API call, or an older client) must not crash -- the averaged keys
    simply don't appear rather than being 0.0/None/NaN.
    """
    mgr = make_manager()
    contract = make_contract()
    mgr.create_room(
        "bare-metrics-room", contract, "p1", AggregationConfig(min_available_clients=1, quorum=1.0),
        initial_state={"w": np.array([0.0, 0.0])},
    )
    mgr.start_room("bare-metrics-room")
    mgr.join_client("bare-metrics-room", "solo", {})
    mgr.start_round("bare-metrics-room")
    mgr.submit_update("bare-metrics-room", "solo", {"w": np.array([1.0, 1.0])}, n_samples=1, base_version=0)

    summary = mgr.maybe_finalize_round("bare-metrics-room")
    assert summary["status"] == "aggregated"
    assert "avg_train_accuracy" not in summary
    assert "avg_local_training_seconds" not in summary
    assert "aggregation_seconds" in summary  # always present -- measured by the server, not reported


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
