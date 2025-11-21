"""
Top level module import for SoftAdapt.
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

from softadapt.callbacks import AdaptiveLossCallback as AdaptiveLossCallback

from softadapt import algorithms as algorithms
from softadapt.algorithms import (
    LossWeightedSoftAdapt as LossWeightedSoftAdapt,
    NormalizedSoftAdapt as NormalizedSoftAdapt,
    SoftAdapt as SoftAdapt,
)

from softadapt import constants as constants
from softadapt.utilities import _finite_difference as _finite_difference
