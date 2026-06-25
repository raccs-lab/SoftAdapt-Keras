import pytest
import numpy as np
from unittest.mock import MagicMock

from softadapt.base._softadapt_base_class import SoftAdaptBase

# =============================================================
# Fixtures and Setup
# =============================================================


@pytest.fixture
def softadapt_instance():
    """Provides a ready-to-use instance of SoftAdaptBase."""
    # We initialize the base class with default values for predictable testing
    return SoftAdaptBase()


# =============================================================
# Test Suite for Initialization (__init__)
# =============================================================


def test_initialization_defaults(softadapt_instance):
    """Tests that the initializer sets attributes correctly using default values."""
    assert softadapt_instance.beta == 0.1
    assert softadapt_instance.accuracy_order is None
    # Epsilon is set by backend(), which we assume returns a small float
    assert isinstance(softadapt_instance.epsilon, float)


def test_initialization_custom(softadapt_instance):
    """Tests initialization with custom beta and accuracy order."""
    base = SoftAdaptBase(beta=0.5, accuracy_order=3)
    assert base.beta == 0.5
    assert base.accuracy_order == 3


# =============================================================
# Test Suite for Abstract Methods (get_component_weights)
# =============================================================


def test_get_component_weights_is_abstract(softadapt_instance):
    """Tests that the abstract method correctly raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        # Calling the base class implementation triggers the intended raise
        softadapt_instance.get_component_weights()


# =============================================================
# Test Suite for _softmax (Heavy Mocking)
# =============================================================


@pytest.mark.parametrize(
    "shift, weights_present",
    [
        (True, False),  # Standard case (shifted)
        (False, False),  # No shifting applied
        (True, True),  # Loss-weighted case (shifted and weighted)
    ],
)
def test_softmax_branching(mocker, softadapt_instance, shift, weights_present):
    """Tests the correct execution path for _softmax based on flags."""

    # --- Setup Mocks ---
    # Patch the dependent module/functions at the point of use in softadapt_base.py
    mock_ops = mocker.patch("softadapt.base._softadapt_base_class.ops")

    # Define predictable return values for the mocked Keras ops sequence
    mock_ops.max.return_value = 5.0  # Simulates ops.max(input_tensor)
    mock_ops.exp.return_value = np.array([10.0, 5.0])  # Simulates ops.exp(...)
    mock_ops.sum.return_value = 15.0  # Simulates the final ops.sum(...)

    # Simulate input/weight types
    input_tensor = MagicMock()
    numerator_weights = np.array([0.1, 0.2]) if weights_present else None

    # --- Execution ---
    if shift:
        softadapt_instance._softmax(
            input_tensor,
            beta=1.0,
            numerator_weights=numerator_weights,
            shift_by_max_value=True,
        )
    else:
        softadapt_instance._softmax(input_tensor, beta=1.0, shift_by_max_value=False)

    # --- Assertions based on scenario ---

    if shift:
        mock_ops.max.assert_called_once()
        if weights_present:
            # Loss-weighted path must include multiplication
            mock_ops.multiply.assert_called_once()
    else:
        # If shift=False, max should never be called
        mock_ops.max.assert_not_called()


# =============================================================
# Test Suite for _compute_rates_of_change (Delegation)
# =============================================================


def test_compute_rates_of_change_delegates_correctly(mocker, softadapt_instance):
    """Tests that the method properly converts tensor and delegates to _get_finite_difference."""

    # Mock the two external dependencies: tensor conversion and finite difference calculation
    mock_convert = mocker.patch("keras.ops.convert_to_numpy")
    mock_get_finite_difference = mocker.patch(
        "softadapt.base._softadapt_base_class._get_finite_difference"
    )

    test_tensor = MagicMock()
    expected_numpy_array = np.array([1.0, 2.0])

    # Configure mocks to simulate successful conversion and calculation
    mock_convert.return_value = expected_numpy_array
    mock_get_finite_difference.return_value = 0.987

    # Act
    rate = softadapt_instance._compute_rates_of_change(test_tensor, order=4)

    # Assertions
    assert rate == 0.987

    # Check that the tensor was converted to numpy correctly before calculation
    mock_convert.assert_called_once_with(test_tensor)
    # Check that the finite difference utility received the correct inputs
    mock_get_finite_difference.assert_called_once_with(
        input_array=expected_numpy_array, order=4, verbose=True
    )


def test_compute_rates_of_change_uses_default_order(mocker, softadapt_instance):
    """Tests that the function defaults to order=5 if no order is passed."""

    mock_convert = mocker.patch("keras.ops.convert_to_numpy")
    mock_get_finite_difference = mocker.patch(
        "softadapt.base._softadapt_base_class._get_finite_difference"
    )

    test_tensor = MagicMock()
    mock_convert.return_value = np.array([1.0, 2.0])

    # Act: Call without specifying order
    softadapt_instance._compute_rates_of_change(test_tensor)

    # Assert that the default order=5 was used in the external call
    mock_get_finite_difference.assert_called_once()
    # Check the arguments passed to _get_finite_difference (order is the second positional argument)
    # The positional arguments match: input_array, order=5, verbose=True
    call_args = mock_get_finite_difference.call_args[0]
    assert call_args[1] == 5
