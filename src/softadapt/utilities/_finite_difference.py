"""
Internal implementation of finite difference.
Copyright (C) 2025 Jacob Logas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging

import numpy as np
from findiff import coefficients

from softadapt.constants import (
    _FIFTH_ORDER,
    _FIFTH_ORDER_COEFFICIENTS,
    _FIRST_ORDER,
    _FIRST_ORDER_COEFFICIENTS,
    _THIRD_ORDER,
    _THIRD_ORDER_COEFFICIENTS,
)


def _get_finite_difference(
    input_array: np.ndarray, order: int | None = None, *, verbose: bool = True
) -> float:
    """Internal utility method for estimating rate of change.

    This function aims to approximate the rate of change for a loss function,
    which is used for the 'LossWeighted' and 'Normalized' variants of SoftAdapt.

    For even accuracy orders, we take advantage of the `findiff` package
    (https://findiff.readthedocs.io/en/latest/source/examples-basic.html).
    Accuracy orders of 1 (trivial), 3, and 5 are retrieved from an internal
    constants file. Due to the underlying mathematics of computing the
    coefficients, all accuracy orders higher than 5 must be an even number.

    Args:
        input_array: An array of floats containing loss evaluations at the
          previous 'n' points (as many points as the order) of the finite
          difference method.
        order: An integer indicating the order of the finite difference method
          we want to use. The function will use the length of the 'input_array'
          array if no values is provided.
        verbose: Whether we want the function to print out information about
          computations or not.

    Returns:
        A float which is the approximated rate of change between the loss
        points.

    Raises:
        ValueError: If the number of points in the `input_array` array is
          smaller than the order of accuracy we desire.
        Value Error: If the order of accuracy is higher than 5 and it is not an
          even number.
    """
    # First, we want to check the order and the number of loss points we are
    # given
    if order is None:
        order = len(input_array) - 1
        if verbose:
            log_msg: str = f"Interpreting finite difference order as {order} since no explicit order was specified."
            logging.info(msg=log_msg)
    elif order > len(input_array):
        error_msg = (
            "The order of finite difference computations can"
            "not be larger than the number of loss points. "
            "Please check the order argument or wait until "
            "enough points have been stored before calling the"
            " method."
        )
        logging.error(msg=error_msg)
        raise ValueError(error_msg)
    elif order + 1 < len(input_array):
        log_msg = (
            f"There are more points than 'order' + 1 ({order + 1}) "
            f"points (array contains {len(input_array)} values). Function"
            f"will use the last {order} elements of loss points for "
            "computations."
        )
        logging.info(msg=log_msg)
        input_array = input_array[(-1 * order - 1) :]

    order_is_even: bool = order % 2 == 0
    # Next, we want to retrieve the correct coefficients based on the order
    if order > _FIFTH_ORDER and not order_is_even:
        err_msg = "Accuracy orders larger than 5 must be even. Please check the arguments passed to the function."
        logging.error(msg=err_msg)
        raise ValueError(err_msg)

    if order_is_even:
        constants = coefficients(deriv=1, acc=order)["forward"]["coefficients"]

    elif order == _FIRST_ORDER:
        constants = _FIRST_ORDER_COEFFICIENTS
    elif order == _THIRD_ORDER:
        constants = _THIRD_ORDER_COEFFICIENTS
    else:
        constants = _FIFTH_ORDER_COEFFICIENTS

    pointwise_multiplication = [
        input_array[i] * constants[i] for i in range(len(constants))
    ]
    return np.sum(pointwise_multiplication)
