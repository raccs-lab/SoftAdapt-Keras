"""Unit testing for calculating finite difference approximations."""

import numpy as np
import pytest
from softadapt.utilities._finite_difference import _get_finite_difference


@pytest.fixture(scope="module")
def rtol():
    return 1e-5


# Positive slope test cases
def test_first_order_positive_slope(rtol):
    order = 2
    loss_points = [0, 1, 2, 3, 4, 5]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(1, approximation, rtol=rtol), (
        "Incorrect first order approximation for simple positive slope test case."
    )


def test_second_order_positive_slope(rtol):
    order = 2
    loss_points = [0, 1, 2, 3, 4, 5]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(1, approximation, rtol=rtol), (
        "Incorrect second order approximation for simple positive slope test case."
    )


def test_third_order_positive_slope(rtol):
    order = 3
    loss_points = [0, 2, 4, 6, 8, 10]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(2, approximation, rtol=rtol), (
        "Incorrect third order approximation for simple positive slope test case."
    )


def test_fourth_order_positive_slope(rtol):
    order = 4
    loss_points = [0, 2, 4, 6, 8, 10]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(2, approximation, rtol=rtol), (
        "Incorrect fourth order approximation for simple positive slope test case."
    )


def test_fifth_order_positive_slope(rtol):
    order = 5
    loss_points = [-5, -4, -3, -2, -1, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(1, approximation, rtol=rtol), (
        "Incorrect fifth order approximation for simple positive slope test case."
    )


def test_tenth_order_positive_slope(rtol):
    order = 10
    loss_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(1, approximation, rtol=rtol), (
        "Incorrect 10th order approximation for simple positive slope test case."
    )


# Negative slope test cases
def test_first_order_negative_slope(rtol):
    order = 1
    loss_points = [15, 12, 9, 6, 3, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(-3, approximation, rtol=rtol), (
        "Incorrect first order approximation for simple negative slope test case."
    )


def test_second_order_negative_slope(rtol):
    order = 2
    loss_points = [5, 4, 3, 2, 1, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(-1, approximation, rtol=rtol), (
        "Incorrect second order approximation for simple negative slope test case."
    )


def test_third_order_negative_slope(rtol):
    order = 3
    loss_points = [20, 16, 12, 8, 4, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(-4, approximation, rtol=rtol), (
        "Incorrect third order approximation for simple negative slope test case."
    )


def test_fourth_order_negative_slope(rtol):
    order = 4
    loss_points = [5, 4, 3, 2, 1, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(-1, approximation, rtol=rtol), (
        "Incorrect fourth order approximation for simple negative slope test case."
    )


def test_fifth_order_negative_slope(rtol):
    order = 5
    loss_points = [5, 4, 3, 2, 1, 0]
    approximation = _get_finite_difference(loss_points, order)
    assert np.isclose(-1, approximation, rtol=rtol), (
        "Incorrect fifth order approximation for simple negative slope test case."
    )
