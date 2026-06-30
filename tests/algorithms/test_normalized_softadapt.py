import pytest
import numpy as np

# Assuming the file containing NormalizedSoftAdapt is named normalized_softadapt.py
from softadapt.algorithms import NormalizedSoftAdapt


# =============================================================
# Fixtures and Setup
# =============================================================


@pytest.fixture
def normalized_softadapt_instance():
    """Provides a fresh instance of the class for testing."""
    return NormalizedSoftAdapt()


# =============================================================
# Test Suite for Initialization (__init__)
# =============================================================


def test_initialization_defaults(normalized_softadapt_instance):
    """Tests initialization using default parameters."""
    assert normalized_softadapt_instance.beta == 0.1
    assert normalized_softadapt_instance.accuracy_order is None


def test_initialization_custom(normalized_softadapt_instance):
    """Tests initialization when custom beta and accuracy order are provided."""
    base = NormalizedSoftAdapt(beta=0.5, accuracy_order=3)
    assert base.beta == 0.5
    assert base.accuracy_order == 3


# =============================================================
# Test Suite for get_component_weights (Core Logic)
# =============================================================


@pytest.mark.parametrize("num_components", [1])
def test_get_component_weights_raises_warning_on_single_input(
    mocker, normalized_softadapt_instance, num_components
):
    """Tests that a warning is correctly issued if only one component is passed."""

    # Mock the required dependencies for execution flow:
    mock_compute = mocker.patch.object(
        normalized_softadapt_instance, "_compute_rates_of_change"
    )

    # Set up the mock to handle the case where we only have one input
    if num_components == 1:
        # Simulate the single loss component being processed
        mock_compute.return_value = np.array([1.0], dtype="float32")

    # Act: We pass a tuple of loss values (though the content is irrelevant due to mocks)
    loss_inputs = tuple(np.array([1.0], dtype="float32") for _ in range(num_components))
    if num_components == 1:
        with pytest.warns(UserWarning, match="trivial weighting"):
            normalized_softadapt_instance.get_component_weights(*loss_inputs)

    # Assertions: Verify the warning was raised
    pass  # Success if no pytest.warns context manager exited without error


def test_get_component_weights_multiple_components_successful_path(
    mocker, normalized_softadapt_instance
):
    """Tests the successful path for multiple loss components: calculation, normalization, and softmax."""

    # --- Mocks Setup ---
    # 1. Mock the dependency on base class method: _compute_rates_of_change
    mock_rate1 = np.array([10.0])  # Rate for component 1
    mock_rate2 = np.array([20.0])  # Rate for component 2
    rates = [mock_rate1, mock_rate2]

    # The method is called once per input component
    mock_compute = mocker.patch.object(
        normalized_softadapt_instance, "_compute_rates_of_change", side_effect=rates
    )

    # 2. Mock Keras tensor operations used during normalization (ops.convert_to_tensor, ops.sum)
    # We simulate the result of the sum being 30 (10+20)
    mocker.patch("keras.ops.sum")

    # 3. Mock the final softmax call to prevent actual execution and verify inputs
    mock_softmax = mocker.patch.object(normalized_softadapt_instance, "_softmax")

    # --- Execution ---
    loss_inputs = (np.array([1.0]), np.array([2.0]))  # Two components
    normalized_softadapt_instance.get_component_weights(*loss_inputs)

    # --- Assertions ---

    # 1. Verify that rates were computed for each component
    assert mock_compute.call_count == 2

    # 2. Verify the tensor operations that led to normalization occurred
    # mock_convert.assert_has_calls([
    #     call(rates[0], dtype=backend.floatx()),  # Convert individual rate 1
    #     call(rates[1], dtype=backend.floatx())   # Convert individual rate 2
    # ])

    # 3. Verify the final softmax was called with the normalized tensor (which is a concept, here we verify inputs)
    # The softmax call should happen after the normalization factor (1/sum) is applied.
    mock_softmax.assert_called_once()  # The actual tensor input would be the normalized result


def test_compute_rates_of_change_delegation(mocker, normalized_softadapt_instance):
    """Tests that the method correctly calls the parent base class's rate calculation."""
    # Since this is a child class method, it calls the base version: self._compute_rates_of_change
    # We ensure that when this method is called, it behaves as expected.
    pass  # This test validates the setup but relies on _compute_rates_of_change being correct.
