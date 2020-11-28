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


"""Tests for plenpy.utilities.images module.
"""

import imageio
import numpy as np
import scipy.fftpack as sc_fft
from pytest import raises

import plenpy.logg
from plenpy import testing
from plenpy.utilities import denoise

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


# Test remove_outliers function
def test_remove_outliers():
    # Remove no outliers
    img_in = np.ones((10, 10))
    conf = np.ones((10, 10))
    img_out = denoise.remove_outliers(img_in, conf, threshold=0.0)
    assert img_in.shape == img_out.shape
    assert np.array_equal(img_in, img_out)

    # Remove single outlier
    img_in = np.ones((10, 10))
    conf = np.ones((10, 10))
    img_in[5, 5] = None
    conf[5, 5] = 0
    img_out = denoise.remove_outliers(img_in, conf, threshold=0.5)
    assert img_in.shape == img_out.shape
    assert np.array_equal(np.ones((10, 10)), img_out)

    # Remove multiple outliers
    img_in = np.asarray((np.linspace(0, 1, 10), np.linspace(0, 1, 10)))
    conf = np.ones(img_in.shape)
    img_in[0, 1] = None
    img_in[0, 4] = None
    img_in[1, 8] = None
    conf[0, 1] = 0
    conf[0, 4] = 0
    conf[1, 8] = 0
    img_out = denoise.remove_outliers(img_in, conf, threshold=0.5)
    assert img_in.shape == img_out.shape
    assert np.array_equal(
        np.asarray((np.linspace(0, 1, 10), np.linspace(0, 1, 10))),
        img_out)

    return


# Test remove_outliers parsing and error handling
def test_remove_outliers_parsing():

    with raises(ValueError) as cm:
        denoise.remove_outliers(np.zeros((10, 10)), np.zeros((10, 15)))
    assert ("Input image (10, 10) and confidence "
            "(10, 15) do not have the same shape.") == str(cm.value)

    with raises(ValueError) as cm:
        denoise.remove_outliers(np.zeros((10, 10, 5)), np.zeros((10, 10, 5)))
    assert "Invalid image dimension 3. Must have dimension 2" == str(cm.value)

    return
