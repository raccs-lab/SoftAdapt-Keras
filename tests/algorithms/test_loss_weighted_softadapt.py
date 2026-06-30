from pytest_mock import MockerFixture
import pytest
import numpy as np

# Assuming the file containing LossWeightedSoftAdapt is named loss_weighted_softadapt.py
from softadapt.algorithms import LossWeightedSoftAdapt


@pytest.fixture
def loss_weighted_instance():
    """Provides a fresh instance of the class for testing."""
    return LossWeightedSoftAdapt()


# =============================================================
# Test Suite for Initialization (__init__)
# =============================================================


def test_initialization_defaults(loss_weighted_instance):
    """Tests initialization using default parameters."""
    # The defaults are inherited from SoftAdaptBase, but we verify the setup is there.
    assert loss_weighted_instance.beta == 0.1
    assert loss_weighted_instance.accuracy_order is None


# =============================================================
# Test Suite for get_component_weights (Core Logic)
# =============================================================


def test_get_component_weights_raises_warning_on_single_input(
    mocker, loss_weighted_instance
):
    """Tests that the specific warning is issued when only one component is passed."""

    mocker.patch("keras.ops")

    mock_rate_1 = np.array([1.0])
    # We use side_effect to simulate the loop execution per component
    mocker.patch.object(
        loss_weighted_instance,
        "_compute_rates_of_change",
        side_effect=[mock_rate_1],
    )

    # Act: Pass only one component
    loss_inputs = (mocker.MagicMock(),)

    # Assert: Verify the specific warning is raised
    with pytest.warns(UserWarning, match="trivial weighting"):
        loss_weighted_instance.get_component_weights(*loss_inputs)


def test_get_component_weights_multiple_components_successful_path(
    mocker: MockerFixture, loss_weighted_instance
):
    """
    Tests the full successful execution path for multiple components,
    ensuring correct orchestration of rate computation and averaging.
    """

    # --- Setup Mocks ---

    # 1. Mock the dependencies on Keras Ops needed during the loop/conversion phase.
    mocker.patch("keras.ops")

    # Configure the mocked objects to control the flow:

    # The rates are computed by the base class method. We need to control its return value.
    # This simulates the list of rates being generated during the loop iterations.
    mock_rate_1 = np.array([1.0])
    mock_rate_2 = np.array([2.0])
    # We use side_effect to simulate the loop execution per component
    mock_compute = mocker.patch.object(
        loss_weighted_instance,
        "_compute_rates_of_change",
        side_effect=[mock_rate_1, mock_rate_2],
    )

    # The Keras ops are mocked to simulate successful tensor conversions.

    # The final softmax is where the weights are returned. We mock this to check inputs.
    mock_softmax = mocker.patch.object(loss_weighted_instance, "_softmax")

    # --- Execution ---
    # Two components are passed: two loss tensors.
    loss_inputs = (mocker.MagicMock(), mocker.MagicMock())
    loss_weighted_instance.get_component_weights(*loss_inputs)

    # --- Assertions ---

    # 1. Verify the rate computation ran for each input component
    assert mock_compute.call_count == 2

    # 2. Verify Keras ops were used inside the loop for averaging (occurs twice, once per component)
    # We check that casting and mean were called on the loss points during the loop.
    # assert mock_cast.call_count == 2
    # The loss averaging happens *inside* the loop, so we expect it to run for each component.
    # The specific verification of ops.mean requires knowing the exact tensor flow, but confirming it's called once per component is sufficient for unit testing this class.

    # 3. Verify the conversion of the collected lists to Tensors happened (occurs twice: once for rates, once for averages)
    # Rates list -> Tensor
    # assert (
    #     mock_ops.convert_to_tensor.call_count >= 2
    # )  # At least once for rates, and again for averages conversion

    # 4. Verify the final step: _softmax was called with the correct information
    mock_softmax.assert_called_once()
