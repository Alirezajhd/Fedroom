import numpy as np
import pytest

from coordinator.aggregation import (
    ClientUpdate,
    ContractError,
    ModelContract,
    NonFiniteUpdateError,
    OversizedUpdateError,
    validate_update,
    weighted_average,
)


def test_weighted_fedavg_known_values():
    """w_{t+1} = sum_k (n_k / sum_j n_j) * w_k, checked against hand-computed values."""
    u1 = ClientUpdate(
        client_id="a", n_samples=100, base_version=0,
        state={"w": np.array([1.0, 2.0, 3.0])},
    )
    u2 = ClientUpdate(
        client_id="b", n_samples=300, base_version=0,
        state={"w": np.array([5.0, 5.0, 5.0])},
    )
    result = weighted_average([u1, u2])
    # weight_a = 100/400 = 0.25, weight_b = 300/400 = 0.75
    expected = 0.25 * np.array([1.0, 2.0, 3.0]) + 0.75 * np.array([5.0, 5.0, 5.0])
    np.testing.assert_allclose(result["w"], expected)


def test_weighted_fedavg_equal_weights_is_plain_average():
    u1 = ClientUpdate(client_id="a", n_samples=10, base_version=0, state={"w": np.array([2.0, 4.0])})
    u2 = ClientUpdate(client_id="b", n_samples=10, base_version=0, state={"w": np.array([6.0, 8.0])})
    result = weighted_average([u1, u2])
    np.testing.assert_allclose(result["w"], np.array([4.0, 6.0]))


def test_weighted_fedavg_multi_tensor_and_dtype_preserved():
    contract_state = lambda a, b: {"w1": np.array(a, dtype=np.float32), "b1": np.array(b, dtype=np.float32)}
    u1 = ClientUpdate(client_id="a", n_samples=1, base_version=0, state=contract_state([1, 1], [1]))
    u2 = ClientUpdate(client_id="b", n_samples=3, base_version=0, state=contract_state([5, 5], [5]))
    result = weighted_average([u1, u2])
    np.testing.assert_allclose(result["w1"], [4.0, 4.0])
    np.testing.assert_allclose(result["b1"], [4.0])
    assert result["w1"].dtype == np.float32


def test_validate_update_rejects_shape_mismatch():
    contract = ModelContract(model_id="m", framework="pytorch", param_shapes={"w": (3,)})
    with pytest.raises(ContractError):
        validate_update({"w": np.array([1.0, 2.0])}, contract)


def test_validate_update_rejects_extra_or_missing_params():
    contract = ModelContract(model_id="m", framework="pytorch", param_shapes={"w": (2,), "b": (1,)})
    with pytest.raises(ContractError):
        validate_update({"w": np.array([1.0, 2.0])}, contract)  # missing "b"
    with pytest.raises(ContractError):
        validate_update(
            {"w": np.array([1.0, 2.0]), "b": np.array([1.0]), "extra": np.array([0.0])}, contract
        )


def test_validate_update_rejects_nan_and_inf():
    contract = ModelContract(model_id="m", framework="pytorch", param_shapes={"w": (2,)})
    with pytest.raises(NonFiniteUpdateError):
        validate_update({"w": np.array([1.0, float("nan")])}, contract)
    with pytest.raises(NonFiniteUpdateError):
        validate_update({"w": np.array([1.0, float("inf")])}, contract)


def test_validate_update_rejects_oversized_norm():
    contract = ModelContract(model_id="m", framework="pytorch", param_shapes={"w": (3,)}, max_update_norm=1.0)
    with pytest.raises(OversizedUpdateError):
        validate_update({"w": np.array([10.0, 10.0, 10.0])}, contract)


def test_validate_update_accepts_within_norm_limit():
    contract = ModelContract(model_id="m", framework="pytorch", param_shapes={"w": (2,)}, max_update_norm=5.0)
    norm = validate_update({"w": np.array([1.0, 1.0])}, contract)
    assert norm == pytest.approx(2 ** 0.5)
