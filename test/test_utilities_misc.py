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


"""Tests for plenpy.utilities.misc module.

"""

import numpy as np
from pytest import raises

import plenpy.logg
from plenpy import testing
from plenpy.utilities import misc

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


def test_get_avail_extensions():

    list = misc.get_avail_extensions()

    # Check that list contains basic extensions
    # In particular, check LFR, LFP plugins provided by utilities.lytro_reader
    assert '.png' in list
    assert '.jpg' in list
    assert '.lfr' in list
    assert '.lfp' in list

    return


def test_rot_matrix():

    # test error handling
    with raises(ValueError) as cm:
        r = misc.rot_matrix(angle=3, unit='nonsense')
    assert ("Specified unit is not 'degree/s' nor 'radian/s'") == str(cm.value)

    with raises(ValueError) as cm:
        r = misc.rot_matrix(angle=3, unit=1)
    assert ("Specified unit is not 'degree/s' nor 'radian/s'") == str(cm.value)

    # test values
    alpha_rad = 0
    r = misc.rot_matrix(angle=alpha_rad, unit='rad')
    assert np.array_equal(r, np.eye(2, 2, dtype=np.float64))

    alpha_rad = np.pi / 4
    alpha_deg = 45
    r_rad = misc.rot_matrix(angle=alpha_rad, unit='rad')
    r_deg = misc.rot_matrix(angle=alpha_deg, unit='deg')
    assert np.allclose(r_rad, [[1/np.sqrt(2), -1/np.sqrt(2)],
                               [1/np.sqrt(2), 1/np.sqrt(2)]])
    assert np.allclose(r_deg, [[1/np.sqrt(2), -1/np.sqrt(2)],
                               [1/np.sqrt(2), 1/np.sqrt(2)]])

    alpha_rad = np.pi/2
    alpha_deg = 90
    r_rad = misc.rot_matrix(angle=alpha_rad, unit='rad')
    r_deg = misc.rot_matrix(angle=alpha_deg, unit='deg')
    assert np.allclose(r_rad, [[0, -1], [1, 0]])
    assert np.allclose(r_deg, [[0, -1], [1, 0]])

    alpha_rad = np.pi
    alpha_deg = 180
    r_rad = misc.rot_matrix(angle=alpha_rad, unit='rad')
    r_deg = misc.rot_matrix(angle=alpha_deg, unit='deg')
    assert np.allclose(r_rad, -np.eye(2, 2, dtype=np.float64))
    assert np.allclose(r_deg, -np.eye(2, 2, dtype=np.float64))

    alpha_rad = 2*np.pi
    alpha_deg = 360
    r_rad = misc.rot_matrix(angle=alpha_rad, unit='rad')
    r_deg = misc.rot_matrix(angle=alpha_deg, unit='deg')
    assert np.allclose(r_rad, np.eye(2, 2, dtype=np.float64))
    assert np.allclose(r_deg, np.eye(2, 2, dtype=np.float64))

    return
