"""
Third level module import for constants.
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

from numpy import array

# All constants are for forward finite difference method.
_FIRST_ORDER = 1
_FIRST_ORDER_COEFFICIENTS = array((-1, 1))
_THIRD_ORDER = 3
_THIRD_ORDER_COEFFICIENTS = array((-11 / 6, 3, -3 / 2, 1 / 3))
_FIFTH_ORDER = 5
_FIFTH_ORDER_COEFFICIENTS = array((-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5))

_EPSILON = 1e-08
