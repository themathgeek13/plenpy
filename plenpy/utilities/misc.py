# Copyright (C) 2018-2020  The Plenpy Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Module defining miscellaneous methods used in the plenpy package.

"""
from typing import List

import imageio
import numpy as np
from numpy.core.multiarray import ndarray

import plenpy.logg

logger = plenpy.logg.get_logger()

__all__ = ['get_avail_extensions', 'rot_matrix']

# Define a list of all allowed sensor image suffixes
# This corresponds to extensions that can be read by :func:`.imageio.imread()`.
__sensorImageSuffix = []

for format_tuple in imageio.formats:
    for ext in format_tuple._extensions:
        __sensorImageSuffix.append(ext.upper())
        __sensorImageSuffix.append(ext.lower())
# Convert list to set to remove duplicates,
# then back to list to sort alphabetically
__sensorImageSuffix = sorted(list(set(__sensorImageSuffix)))


def get_avail_extensions() -> List[str]:
    """List all available extensions/suffixes that can be read as images.

    Returns:
        List of all image extensions that can be read as sensor images.
        See also :func:`imageio.help()`.

    """
    return __sensorImageSuffix


def rot_matrix(angle: float,
               unit: str = 'radians') -> ndarray:
    """Get 2D rotation matrix using angle alpha.

    Args:
        angle: Angle to rotate in the specified unit.

        unit: Specify if alpha is passed in radians or degrees. Available
            are 'rad', 'radian', 'radians', 'deg', 'degree', 'degrees'.

            Default: 'radians'

    Returns:
        Rotation matrix.

    """
    if unit == 'rad' or unit == 'radian' or unit == 'radians':
        alpha = angle
    elif unit == 'deg' or unit == 'degree' or unit == 'degrees':
        # Multiply by pi/180
        alpha = 0.017453292519943295*angle
    else:
        raise ValueError("Specified unit is not 'degree/s' nor 'radian/s'")

    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
