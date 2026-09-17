import numpy as np
import pytest

from coordinator.aggregation import ClientUpdate
from coordinator.storage import LocalCheckpointStore, deserialize_state, serialize_state
from strategies.robust import KrumStrategy, MedianStrategy, TrimmedMeanStrategy, get_strategy


def test_checkpoint_serialize_roundtrip():
    state = {"w": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([0.5])}
    blob = serialize_state(state)
    restored = deserialize_state(blob)
    np.testing.assert_allclose(restored["w"], state["w"])
    np.testing.assert_allclose(restored["b"], state["b"])


def test_local_checkpoint_store_save_and_load_reports_correct_version(tmp_path):
    store = LocalCheckpointStore(base_dir=str(tmp_path))
    state = {"w": np.array([1.0, 2.0, 3.0])}
    uri = store.save("roomX", version=7, state=state)
    assert "v00007" in uri
    loaded = store.load(uri)
    np.testing.assert_allclose(loaded["w"], state["w"])


def _updates(vectors, n_samples=None):
    n_samples = n_samples or [10] * len(vectors)
    return [
        ClientUpdate(client_id=f"c{i}", n_samples=n, base_version=0, state={"w": np.array(v, dtype=np.float64)})
        for i, (v, n) in enumerate(zip(vectors, n_samples))
    ]


def test_trimmed_mean_removes_extreme_outlier():
    # 4 honest clients around [1,1], 1 malicious client sending a huge vector.
    updates = _updates([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.05], [1000.0, -1000.0]])
    strat = TrimmedMeanStrategy(beta=1)
    result = strat.aggregate(updates)
    assert abs(result["w"][0] - 1.0) < 0.2
    assert abs(result["w"][1] - 1.0) < 0.2


def test_median_is_robust_to_single_outlier():
    updates = _updates([[1.0], [1.0], [1.0], [1.0], [9999.0]])
    result = MedianStrategy().aggregate(updates)
    assert result["w"][0] == pytest.approx(1.0)


def test_krum_selects_a_plausible_honest_update_not_the_outlier():
    updates = _updates([[1.0, 1.0], [1.05, 0.95], [0.95, 1.05], [500.0, -500.0], [1.02, 0.98]])
    strat = KrumStrategy(byzantine_f=1, multi=False)
    result = strat.aggregate(updates)
    # The Krum-selected vector should be near the honest cluster, not the outlier.
    assert np.linalg.norm(result["w"] - np.array([1.0, 1.0])) < 1.0


def test_multi_krum_falls_back_to_average_when_too_few_clients():
    updates = _updates([[1.0], [3.0]])
    strat = KrumStrategy(byzantine_f=1, multi=True)  # k=2 <= 2f+2=4 -> fallback
    result = strat.aggregate(updates)
    assert result["w"][0] == pytest.approx(2.0)


def test_strategy_registry_returns_expected_types():
    assert get_strategy("fedavg").name == "fedavg"
    assert get_strategy("trimmed_mean", byzantine_f=1).name == "trimmed_mean"
    assert get_strategy("median").name == "median"
    assert get_strategy("krum", byzantine_f=1).name == "krum"
    assert get_strategy("multi_krum", byzantine_f=1).name == "krum"
    with pytest.raises(ValueError):
        get_strategy("does_not_exist")
