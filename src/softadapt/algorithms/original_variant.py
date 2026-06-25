"""
Implementation of the original variant of SoftAdapt.
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

import warnings

from keras import KerasTensor, backend, ops

from softadapt.base._softadapt_base_class import SoftAdaptBase


class SoftAdapt(SoftAdaptBase):
    """The original variant class.
        The original variant of SoftAdapt is described in section 3.1.1
        of our manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta (float): A float that is the 'beta' hyperparameter in the manuscript. If
            beta > 0, then softAdapt will pay more attention the worst performing
            loss component. If beta < 0, then SoftAdapt will assign higher weights
            to the better performing components. beta==0 is the trivial case and
            all loss components will have coefficient 1.

        accuracy_order (int | None): An integer indicating the accuracy order of the finite
            volume approximation of each loss component's slope.
    """

    def __init__(self, beta: float = 0.1, accuracy_order: int | None = None) -> None:
        super().__init__(beta=beta, accuracy_order=accuracy_order)

    def get_component_weights(
        self, *loss_component_values: tuple[KerasTensor], verbose: bool = True
    ) -> KerasTensor:
        """Class method for SoftAdapt weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
                loss component that have been stored for the past 'n' iterations
                or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
                functions should print out information and warning about
                computations.
        Returns:
                The computed weights for each loss components. For example, if there
                were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
                return tensor will be the weights (alpha_1, alpha_2, alpha_3,
                alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1:
            warnings.warn(
                UserWarning(
                    "You have only passed on the value of one loss component, which will result in trivial weighting."
                ),
                stacklevel=2,
            )

        rates_of_change: list[float] = [
            self._compute_rates_of_change(
                loss_points, self.accuracy_order, verbose=verbose
            )
            for loss_points in loss_component_values
        ]

        rates_of_change_t: KerasTensor = ops.convert_to_tensor(
            rates_of_change, dtype=backend.floatx()
        )
        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change_t, beta=self.beta)
