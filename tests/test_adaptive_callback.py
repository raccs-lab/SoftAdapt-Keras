import numpy as np
import pytest
from keras import backend, ops
import softadapt


@pytest.fixture
def adaptive_loss_callback():
    components = ["loss1", "loss2"]
    weights = [0.5, 0.5]
    callback = softadapt.callbacks.AdaptiveLossCallback(
        components=components,
        weights=weights,
        frequency="epoch",
        beta=0.1,
        algorithm="base",
    )
    yield callback
    backend.clear_session()


def test_initialization(adaptive_loss_callback):
    callback = adaptive_loss_callback
    # assert callback.order == callback.components
    assert isinstance(callback.algorithm, softadapt.algorithms.SoftAdapt)
    assert np.array_equal(ops.convert_to_numpy(callback.weights[0]), 0.5)
    assert np.array_equal(ops.convert_to_numpy(callback.weights[1]), 0.5)
    assert callback.frequency == "epoch"
    assert len(callback.components_history) == len(callback.order)


def test_on_epoch_end_updates_weights(adaptive_loss_callback, monkeypatch):
    def get_component_weights(
        self, *loss_component_values: tuple, verbose: bool = True
    ):
        return np.array([0.6, 0.4])

    callback = adaptive_loss_callback
    logs = {"loss1": 0.2, "loss2": 0.3}
    callback.on_epoch_end(epoch=0, logs=logs)

    # Check if the component history is updated
    assert callback.components_history[0] == [0.2]
    assert callback.components_history[1] == [0.3]

    # Mock the get_component_weights method
    monkeypatch.setattr(
        callback.algorithm,
        "get_component_weights",
        get_component_weights,
    )
    callback.on_epoch_end(epoch=1, logs=logs)

    # Check if the weights have been updated
    assert np.allclose(ops.convert_to_numpy(callback.weights), [0.6, 0.4])


def test_true_epoch_end_updates_weights(adaptive_loss_callback):
    callback = adaptive_loss_callback
    logs = {"loss1": 0.2, "loss2": 0.3}
    callback.on_epoch_end(epoch=0, logs=logs)

    # Check if the component history is updated
    assert callback.components_history[0] == [0.2]
    assert callback.components_history[1] == [0.3]

    # Check if weights are updated
    callback.on_epoch_end(epoch=1, logs=logs)

    # Check if the weights have been updated
    assert np.allclose(ops.convert_to_numpy(callback.weights), [0.5, 0.5])


def test_on_epoch_end_clears_history(adaptive_loss_callback, monkeypatch):
    def get_component_weights(
        self, *loss_component_values: tuple, verbose: bool = True
    ):
        return np.array([0.6, 0.4])

    callback = adaptive_loss_callback
    logs = {"loss1": 0.2, "loss2": 0.3}
    callback.on_epoch_end(epoch=0, logs=logs)

    # Check if the component history is updated
    assert callback.components_history[0] == [0.2]
    assert callback.components_history[1] == [0.3]

    # Mock the get_component_weights method
    monkeypatch.setattr(
        callback.algorithm, "get_component_weights", get_component_weights
    )
    logs = {"loss1": 0.5, "loss2": 0.2}
    callback.on_epoch_end(epoch=1, logs=logs)

    # Check if the history has been cleared except for one
    assert callback.components_history[0] == [0.5]
    assert callback.components_history[1] == [0.2]
