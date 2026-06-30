"""Unit testing for calculating finite difference approximations."""

from pytest_mock import MockerFixture
import numpy as np
import pytest
from softadapt.utilities._finite_difference import _get_finite_difference


@pytest.fixture(scope="module")
def rtol():
    return 1e-5


@pytest.mark.parametrize("order", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "loss_points",
    [
        (1, [0, 1, 2, 3, 4, 5]),
        (1, [-5, -4, -3, -2, -1, 0]),
        (2, [0, 2, 4, 6, 8, 10]),
        (2, [-10, -8, -6, -4, -2, 0]),
    ],
)
def test_n_order_positive_slope(mocker: MockerFixture, rtol, order, loss_points):
    mocker.patch("logging.info")
    mocker.patch("logging.error")
    approximation: float = _get_finite_difference(loss_points[1], order, verbose=True)
    assert np.isclose(loss_points[0], approximation, rtol=rtol)


def test_10_order_less_points(rtol):
    with pytest.raises(ValueError):
        _get_finite_difference([0, 1, 2, 3, 4, 5], 10)


# Negative slope test cases
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "loss_points",
    [(-3, [15, 12, 9, 6, 3, 0]), (-1, [5, 4, 3, 2, 1, 0]), (-4, [20, 16, 12, 8, 4, 0])],
)
def test_n_order_negative_slope(rtol, order, loss_points):
    approximation: float = _get_finite_difference(loss_points[1], order)
    assert np.isclose(loss_points[0], approximation, rtol=rtol)


# =============================================================
# Test Suite: Error Handling (Achieving Full Coverage of `raise` statements)
# =============================================================


def test_error_not_enough_points(rtol):
    """Tests the ValueError raised when array size is too small for the desired order."""
    # To trigger the error (order + 1 < len(array) condition failure),
    # we need array length <= order. e.g., order=5, len=3.
    input_array = np.arange(3)
    order = 5

    # Act & Assert: Expect the ValueError defined in the source code
    with pytest.raises(ValueError):
        _get_finite_difference(input_array, order=order)


def test_error_invalid_high_order_odd(rtol):
    """Tests the ValueError raised for accuracy orders > 5 and not even."""
    input_array = np.arange(10)
    order = 7  # Odd number > 5

    # Act & Assert: Expect the ValueError defined in the source code
    with pytest.raises(ValueError):
        _get_finite_difference(input_array, order=order)
