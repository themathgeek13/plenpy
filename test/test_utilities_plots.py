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


"""Tests for plenpy.utilities.plots module.

"""

import os

import imageio
import numpy as np
from pytest import raises
from skimage.color import rgb2gray

import plenpy.logg
from plenpy import testing
from plenpy.utilities import images
from plenpy.utilities import plots

logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

# Test utilities.plots submodule
# These tests are not really useful, as plotting cannot be testet.
# They are basic "the-functions-don't-crash-tests"

TEST_IMG_FILENAME = "images/crayons.jpeg"


def test_plot_fft():
    # Check Color
    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    img = imageio.imread(img_file)

    # Crop image to (256,512) to increase fft speed
    img = img[:256, :512, :]

    assert plots.plot_fft(img) is None

    # Check odd shape
    img_odd_x = img[:-1, :]
    assert plots.plot_fft(img_odd_x) is None
    img_odd_y = img[:, :-1]
    assert plots.plot_fft(img_odd_y) is None
    img_odd_xy = img[:-1, :-1]
    assert plots.plot_fft(img_odd_xy) is None

    # Check BW
    img = rgb2gray(img)
    assert plots.plot_fft(img) is None

    # Check BW squeeze
    img = np.squeeze(img)
    assert plots.plot_fft(img) is None

    # Check shape error
    with raises(ValueError) as cm:
        plots.plot_fft(np.ones((100, 100, 100, 3)))

    # Check options
    assert plots.plot_fft(img, implementation='scipy') is None
    assert plots.plot_fft(img, implementation='numpy') is None
    assert plots.plot_fft(img, window='hann') is None
    assert plots.plot_fft(img, interpolation=None) is None
    assert plots.plot_fft(img, cmap='viridis') is None
    assert plots.plot_fft(img, vmin=2.0, vmax=5) is None
    assert plots.plot_fft(img, rescale=False) is None
    assert plots.plot_fft(img, shift=False) is None
    assert plots.plot_fft(img, plt_show=False) is None

    # Test error handling
    with raises(ValueError) as cm:
        plots.plot_fft(img, implementation='nonsense')

    assert (
               "The implementation 'nonsense' is not one of the "
                "supported implementations, ['numpy', 'scipy', 'fftw']."
           ) == str(cm.value)

    return
