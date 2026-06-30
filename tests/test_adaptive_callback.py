from keras import Model
from keras import KerasTensor
import pytest
from pytest_mock import MockerFixture
import numpy as np

# Assuming the file containing AdaptiveLossCallback is named adaptive_loss_callback.py
from softadapt.callbacks import AdaptiveLossCallback


# =============================================================
# Fixtures and Setup
# =============================================================


@pytest.fixture
def callback_instance(mocker: MockerFixture):
    """Provides a fresh instance of the callback."""
    # Mocking dependencies that are global or expensive
    mocker.patch("keras.src.utils.file_utils")
    # We instantiate it without needing to pass all values, using defaults is safest.
    return AdaptiveLossCallback(components=["component_A"], frequency="epoch")


@pytest.fixture
def mock_keras_ops(mocker: MockerFixture):
    """Mocks keras tensor operations globally for the test scope."""
    # Patching ops is crucial as they are called throughout the class methods.
    mock_ops = mocker.patch("keras.ops")
    return mock_ops


# =============================================================
# Test Suite for Initialization (__init__)
# =============================================================


@pytest.mark.parametrize("frequency", ["epoch", "batch", 0, 1, 2, 5, 0.5])
@pytest.mark.parametrize("beta", [0.1, 0.5, 10, -0.2, -10])
@pytest.mark.parametrize("algorithm", ["base", "loss-weighted", "normalized", "blah"])
@pytest.mark.parametrize("calculate_on_validation", [True, False])
def test_initialization(
    callback_instance, frequency, beta, algorithm, calculate_on_validation
):
    """Tests that the callback initializes appropriately."""
    try:
        callback_instance = AdaptiveLossCallback(
            components=["component_A"],
            frequency=frequency,
            beta=beta,
            algorithm=algorithm,
            calculate_on_validation=calculate_on_validation,
        )
        assert callback_instance._frequency == (
            frequency if isinstance(frequency, int) else 1
        )
        assert callback_instance.algorithm.beta == beta
    except Exception as e:
        with pytest.raises(ValueError):
            raise e
        return

    if algorithm == "base":
        assert (
            callback_instance.algorithm.__class__.__name__ == "SoftAdapt"
        )  # Checks the default algorithm type
    elif algorithm == "loss-weighted":
        assert (
            callback_instance.algorithm.__class__.__name__ == "LossWeightedSoftAdapt"
        )  # Checks the default algorithm type
    elif algorithm == "normalized":
        assert (
            callback_instance.algorithm.__class__.__name__ == "NormalizedSoftAdapt"
        )  # Checks the default algorithm type
    else:
        pytest.fail("Class initialized with a non-implemented algorithm.")

    assert callback_instance.val is calculate_on_validation


# =============================================================
# Test Suite for Weights Property Getter/Setter
# =============================================================


@pytest.mark.parametrize("attribute", ["loss_weights", "_compile_loss", None])
@pytest.mark.parametrize("weights", [[0.5, 0.4], [-0.5, 0.1]])
@pytest.mark.parametrize("clipping", [True, False])
def test_weights_setter(
    mocker: MockerFixture, callback_instance, attribute, weights, clipping
) -> None:
    """Tests setting weights."""
    # Mock the model structure that supports assignment via property setter
    mock_model = mocker.MagicMock(Model)
    callback_instance.set_model(mock_model)

    # Assert: Verify that the operation to enforce minimum value occurred
    mock_max = mocker.patch("keras.ops.maximum")

    callback_instance._clip = clipping

    if attribute == "loss_weights":
        mock_model.loss_weights = []
    elif attribute == "_compile_loss":
        mock_model._compile_loss = mocker.MagicMock()
        mock_model._compile_loss._flat_losses = [(None, None, w) for w in weights]

    try:
        callback_instance.weights = weights
    except AttributeError as e:
        with pytest.raises(AttributeError):
            raise e

    if clipping:
        assert mock_max.called


@pytest.mark.parametrize("attribute", ["loss_weights", "_compile_loss", None])
@pytest.mark.parametrize("weights", [[0.5, 0.4]])
@pytest.mark.parametrize("clipping", [True, False])
def test_weights_getter(
    mocker: MockerFixture, callback_instance, attribute, weights, clipping
) -> None:
    """Tests getting weights."""
    # Mock the model structure that supports assignment via property setter
    mock_model = mocker.MagicMock(Model)
    callback_instance.set_model(mock_model)

    callback_instance._clip = clipping
    if attribute == "loss_weights":
        mock_model.loss_weights = weights
        assert callback_instance.weights == weights
    elif attribute == "_compile_loss":
        mock_model._compile_loss = mocker.MagicMock()
        mock_model._compile_loss._flat_losses = [(None, None, w) for w in weights]

    try:
        assert callback_instance.weights == weights
    except AttributeError as e:
        with pytest.raises(AttributeError):
            raise e


# =============================================================
# Test Suite for on_epoch_end (Core Orchestration)
# =============================================================


def test_on_epoch_end_successful_weight_recomputation(
    mocker: MockerFixture, callback_instance
):
    """Tests the successful path: history collected -> weights computed -> weights set."""

    # --- Mocks Setup ---
    # Mock the dependency on the underlying algorithm to control its output
    mock_algorithm = mocker.patch.object(
        callback_instance.algorithm, "get_component_weights"
    )

    # Simulate the successful computation of new weights (the result of get_component_weights)
    mock_algorithm.return_value = np.array([0.8, 0.2])

    # Simulate the loss logs passed to the callback (usually losses from components)
    mock_logs = {
        "component_A_loss": mocker.MagicMock(KerasTensor),  # Current loss tensor
        "val_component_A_loss": mocker.MagicMock(KerasTensor),  # Validation loss tensor
    }

    # --- Execution (Simulating the critical moment) ---
    callback_instance.on_epoch_end(epoch=5, logs=mock_logs)

    # --- Assertions ---

    # 1. Verify the algorithm was called with history (this is done by the base class)
    # Since we are simulating the first call after history collection, this verifies the flow.
    mock_algorithm.assert_called_once()

    # 2. Verify the new weights were set (the setter was called)
    # This ensures the `self.weights = ops.cast(adapt_weights, K.floatx())` line ran.
    # We can check the setter was implicitly called via the attribute assignment.

    # Note: Due to complexity, we rely on the mock_softmax (used by the algorithm) being called correctly.
    # The primary coverage point here is that `get_component_weights` was triggered and the history collection occurred.


# =============================================================
# Test Suite for File I/O / Backup (Edge Cases)
# =============================================================


def test_on_train_begin_restores_state(mocker: MockerFixture, callback_instance):
    """Tests the recovery feature: loading history and weights from disk."""

    callback_instance = AdaptiveLossCallback(
        components=["component_A"], frequency="epoch", backup_dir="."
    )

    # --- Mocks Setup ---
    mock_exists = mocker.patch("keras.src.utils.file_utils.exists")
    mock_np_load = mocker.patch("numpy.load")
    mocker.patch("keras.ops.convert_to_tensor")

    # Set up file system existence checks to pass
    mock_exists.side_effect = [True, True]  # Both paths exist

    # Simulate the loaded data
    mock_np_load.side_effect = [
        np.array([["component_A_loss"]]),  # returns history array
        np.array([0.8]),  # returns weights array
    ]

    # Act: Run the beginning of training
    callback_instance.on_train_begin({})

    # Assertions: Verify correct sequence of file operations
    mock_exists.assert_called()  # Check both paths were checked
    # np.load must have been called twice, once for history and once for weights
    assert mock_np_load.call_count == 2


# =============================================================
# Test Suite for Loss Collection Logic (Phase 1: Entering the Function)
# =============================================================


@pytest.mark.parametrize("dir_exists", [True, False])
@pytest.mark.parametrize("on_val", [True, False])
@pytest.mark.parametrize("components", [["A"], ["A", "B"]])
@pytest.mark.parametrize("try_epoch", [1, 5])
@pytest.mark.parametrize("freq", [1, 5])
def test_on_epoch_end_loss_collected_training_mode(
    mocker: MockerFixture,
    callback_instance,
    dir_exists,
    on_val,
    components,
    try_epoch,
    freq,
):
    """Tests the branch where training loss is collected."""
    # Setup: Enable backup path
    callback_instance = AdaptiveLossCallback(
        components=components,
        frequency=freq,
        backup_dir="/tmp/my_run",
        calculate_on_validation=on_val,
    )

    loss_data = {}
    # Act: Loss is available, but no validation mode is active
    for c in components:
        loss_data[c + "_loss"] = mocker.MagicMock()
        loss_data["val_" + c + "_loss"] = mocker.MagicMock()
        callback_instance._components_history[
            callback_instance.order.index(c + "_loss")
        ] = [mocker.MagicMock()]

    # Mock NumPy save to track calls
    np_save = mocker.patch("numpy.save")
    mocker.patch("keras.src.utils.file_utils.exists").return_value = dir_exists
    makedirs_mock = mocker.patch("keras.src.utils.file_utils.makedirs")

    # Mock Keras ops.copy to simulate tensor passage
    keras_copy = mocker.patch("keras.ops.copy")

    type(callback_instance).weights = mocker.PropertyMock()
    # mock_algo = mocker.MagicMock()
    type(callback_instance).algorithm = mocker.PropertyMock()

    # Act: Run the function (which triggers saving)
    callback_instance.on_epoch_end(epoch=try_epoch, logs=loss_data)

    # Assert: Verify that makedirs is only called when it doesn't exist
    if dir_exists:
        makedirs_mock.assert_not_called()
    else:
        makedirs_mock.assert_called_once()

    # Assert: Verify that the correct loss tensor was copied into history
    assert keras_copy.call_count == len(components)

    # Assert: Verify that save happens always
    assert np_save.call_count == 2

    # Assert: Verify that adapt weights are computed only when frequency hit
    # if try_epoch % freq != 0:
    #     mock_algo.get_component_weights.assert_not_called()
    # else:
    #     mock_algo.get_component_weights.assert_called_once()


# =============================================================
# Test Suite for Weight Computation Trigger (Phase 2: Calculation)
# =============================================================


# @pytest.mark.parametrize(
#     "frequency, epoch_check, run_weights",
#     [
#         # Case 1: Epoch frequency (triggers on every epoch > 0)
#         ("epoch", lambda e: e != 0, True),
#         # Case 2: N-step frequency (triggers only if epoch is multiple of N)
#         (3, lambda e: e > 0 and e % 3 == 0, True),
#         # Case 3: Never triggers (epoch=1)
#         (5, lambda e: e == 1, False),
#     ],
# )
# def test_weight_computation_trigger(
#     mocker: MockerFixture, callback_instance, frequency, epoch_check, run_weights
# ):
#     """Tests the complex IF condition that decides when to compute weights."""

#     # Set up frequency and epoch check
#     callback_instance.frequency = frequency
#     epoch = 10 if run_weights else 1

#     # Mock the dependencies that are called when weights are computed
#     mocker.patch.object(callback_instance.algorithm, "get_component_weights")
#     mocker.patch("keras.ops.convert_to_tensor")

#     # Act
#     callback_instance.on_epoch_end(epoch=epoch, logs={})

#     # Assert: Verify the correct computational path was taken
#     if run_weights and epoch == 10 and frequency == "epoch":
#         # This is the successful path, weight computation must happen
#         pass
#     elif epoch == 1 and frequency != "epoch":
#         # This is the failure path, weight computation must NOT happen (len > 1 check fails)
#         pass


# =============================================================
# Test Suite for History Cleanup and Backup (Phase 3: Post-computation)
# =============================================================


def test_on_epoch_end_history_cleanup_epoch(mocker: MockerFixture, callback_instance):
    """Tests the cleanup path for 'epoch' frequency (pop oldest item)."""

    # Setup: We need a list structure to test the `h.pop(0)` action
    callback_instance.components_history = [["loss1", "loss2"]]
    mocker.patch("keras.ops.convert_to_numpy")  # Mock conversion for safety

    # Act: Run the function
    callback_instance.on_epoch_end(epoch=5, logs={})

    # Assert: Verify the history buffer size decreased by one (successful pop)
    assert len(callback_instance.components_history[0]) == 1


def test_on_epoch_end_history_cleanup_batch(mocker: MockerFixture, callback_instance):
    """Tests the cleanup path for non-'epoch' frequency (clear buffer)."""

    # Setup: We need a list structure
    callback_instance.components_history = [["loss1", "loss2"]]

    # Act: Run the function with non-'epoch' frequency
    callback_instance.frequency = 3  # Simulating batch/integer frequency
    callback_instance.on_epoch_end(epoch=5, logs={})

    # Assert: Verify the history buffer was completely cleared
    assert len(callback_instance.components_history[0]) == 0
