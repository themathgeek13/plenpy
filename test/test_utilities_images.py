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

import os
import imageio
import numpy as np
import scipy.fftpack as sc_fft
from pytest import raises

import plenpy.logg
from plenpy import testing
from plenpy.utilities import images
from plenpy.utilities import kernels


# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

# Test data
TEST_RGB_FILENAME = "images/balloons_RGB.png"
TEST_GREY_FILENAME = "images/balloons_GREY.png"


# Test normalization methods
def test_array_normalization_inplace():

    # Int Array
    a = np.asarray([[0, 4], [3, 0]])
    with raises(ValueError) as cm:
        images.normalize_in_place(a)
    assert "In place normalization only allowed for float arrays." == str(cm.value)

    # Float Array
    a = np.asarray([[0.0, 4.0], [3.0, 0.0]], dtype=np.float32)

    a_ref_area = np.asarray([[0, 4.0 / 7.0], [3.0 / 7.0, 0]], dtype=np.float32)
    a_ref_peak = np.asarray([[0, 1.0], [3.0 / 4.0, 0]], dtype=np.float32)
    a_ref_l2 = np.asarray([[0, 4.0 / 5.0], [3.0 / 5.0, 0]], dtype=np.float32)

    images.normalize_in_place(a, by='area')
    assert np.array_equal(a, a_ref_area)

    a = np.asarray([[0.0, 4.0], [3.0, 0.0]], dtype=np.float32)
    images.normalize_in_place(a, by='peak')
    assert np.array_equal(a, a_ref_peak)

    a = np.asarray([[0.0, 4.0], [3.0, 0.0]], dtype=np.float32)
    images.normalize_in_place(a, by='l2')
    assert np.array_equal(a, a_ref_l2)

    return


def test_array_normalization():
    a = np.asarray([[0, 4], [3, 0]], dtype=np.float32)
    a_ref_area = np.asarray([[0, 4.0 / 7.0], [3.0 / 7.0, 0]], dtype=np.float32)
    a_ref_peak = np.asarray([[0, 1.0], [3.0 / 4.0, 0]], dtype=np.float32)
    a_ref_l2 = np.asarray([[0, 4.0 / 5.0], [3.0 / 5.0, 0]], dtype=np.float32)

    assert np.array_equal(images.normalize(a, by='area'), a_ref_area)
    assert np.array_equal(images.normalize(a, by='peak'), a_ref_peak)
    assert np.array_equal(images.normalize(a, by='l2'), a_ref_l2)

    return


# Test normalization array parsing and error handling
def test_array_normalization_parsing():

    a = np.asarray([1, 2, 3])

    with raises(ValueError) as cm:
        array = images.normalize(a, "noValidMethod")
    assert ("Specified argument type is not one of the recognized methods: "
             "['area', 'peak', 'l2']") == str(cm.value)

    return


def test_crop_center():

    test_data_even = np.asarray([[0,  1,  2,  3,  4,  5],
                                [10, 11, 12, 13, 14, 15],
                                [20, 21, 22, 23, 24, 25],
                                [30, 31, 32, 33, 34, 35],
                                [40, 41, 42, 43, 44, 45],
                                [50, 51, 52, 53, 54, 55]])

    test_data_odd = np.asarray([[0,  1,  2,  3,  4,  5,  6],
                                [10, 11, 12, 13, 14, 15, 16],
                                [20, 21, 22, 23, 24, 25, 26],
                                [30, 31, 32, 33, 34, 35, 36],
                                [40, 41, 42, 43, 44, 45, 46],
                                [50, 51, 52, 53, 54, 55, 56],
                                [60, 61, 62, 63, 64, 65, 66]])

    # Test symmetrically cropped images
    # Odd image
    assert np.array_equal(images.crop_center(test_data_odd, 1),
                          33*np.ones((1, 1)))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_odd, 3)),
                          np.asarray([[22, 23, 24],
                                      [32, 33, 34],
                                      [42, 43, 44]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_odd, 5, 3)),
                          np.asarray([[12, 13, 14],
                                      [22, 23, 24],
                                      [32, 33, 34],
                                      [42, 43, 44],
                                      [52, 53, 54]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_odd, 3, 5)),
                          np.asarray([[21, 22, 23, 24, 25],
                                      [31, 32, 33, 34, 35],
                                      [41, 42, 43, 44, 45]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_odd, 5)),
                          np.asarray([[11, 12, 13, 14, 15],
                                      [21, 22, 23, 24, 25],
                                      [31, 32, 33, 34, 35],
                                      [41, 42, 43, 44, 45],
                                      [51, 52, 53, 54, 55]]))

    # Even image
    assert np.array_equal(np.squeeze(images.crop_center(test_data_even, 2)),
                          np.asarray([[22, 23],
                                      [32, 33]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_even, 2, 4)),
                          np.asarray([[21, 22, 23, 24],
                                      [31, 32, 33, 34]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_even, 4, 2)),
                          np.asarray([[12, 13],
                                      [22, 23],
                                      [32, 33],
                                      [42, 43]]))

    assert np.array_equal(np.squeeze(images.crop_center(test_data_even, 4)),
                          np.asarray([[11, 12, 13, 14],
                                      [21, 22, 23, 24],
                                      [31, 32, 33, 34],
                                      [41, 42, 43, 44]]))

    return


def test_crop_center_parsing():
    img = np.ones((120, 120, 3))

    with raises(ValueError) as cm:
        images.crop_center(img, 130, 5)

    assert (f"Crop dimension ({130}{5})larger than "
            f"input image shape {(120, 120, 3)}.") == str(cm.value)

    return


def test_shear():

    # Test inversion
    im = np.zeros((100, 200))
    im[:, 90:110] = 1

    for k in [-1, -0.325, 0.5, 0.78356, 1]:
        im_sheared = images.shear(im, k=k)
        im_sheared_back = images.shear(im_sheared, k=-k)

        # Test equality (interpolation errors accounted for)
        assert np.allclose(im, im_sheared_back, atol=0.3)

        # Test shape
        assert im.shape == im_sheared.shape

    # Test shearing
    im = np.tri(100, 100)
    im_sheared = images.shear(im, k=-1)

    # Assert: right half is zeros
    assert np.allclose(np.zeros((100, 49)), im_sheared[..., 51:])

    # Assert: upper left quarter is triangular
    assert np.allclose(np.fliplr(np.tri(51, 51)), im_sheared[:51, 0:51])

    # Assert: lower left quarter is ones
    assert np.allclose(np.ones((50, 51)), im_sheared[50:, 0:51])

    return


def test_overlay_images():

    img1 = (64*np.ones((128, 256, 3))).astype(np.uint8)
    img2 = (128*np.ones((128, 256, 3))).astype(np.uint8)

    r = 64.0/255.0
    g = 128.0/255.0
    b = 0.5*(64.0 + 128.0)/255.0

    res = images.overlay_images(img1, img2)

    assert np.allclose(res[:, :, 0],
                       r * np.ones((128, 256)))

    assert np.allclose(res[:, :, 1],
                       g * np.ones((128, 256)))

    assert np.allclose(res[:, :, 2],
                       b * np.ones((128, 256)))

    return


def test_fourier():

    # Test image
    img = np.random.rand(51, 71).astype(np.float32)

    # Numpy FFT
    fft_np = np.fft.fft2(img)
    fft = images.fourier(img, shift=False, implementation='numpy')
    assert np.array_equal(fft_np, fft)

    # Scipy FFT
    fft_sc = sc_fft.fft2(img)
    fft = images.fourier(img, shift=False, implementation='scipy')
    assert np.array_equal(fft_sc, fft)

    # Invalid name
    with raises(ValueError) as cm:
        images.fourier(img, shift=False, implementation='nonsense')

    assert "The implementation 'nonsense' is not one of the supported " \
           "implementations, ['numpy', 'scipy', 'fftw']." == str(cm.value)

    # Check greyscale conversion
    # Caution: Conversion is weighted using CIE standard
    img_color = img[..., np.newaxis].repeat(3, axis=-1)
    assert np.array_equal(img, img_color[..., 0])
    assert np.array_equal(img, img_color[..., 1])
    assert np.array_equal(img, img_color[..., 2])

    fft_from_color = images.fourier(img_color)
    fft_bw = images.fourier(img)
    assert np.allclose(fft_bw, fft_from_color, atol=0.01)

    # Test windowing
    n, m = img.shape
    kern = kernels.get_kernel('hann', size=n, size_y=m, normalize='peak').astype(img.dtype)
    img_windowed = np.multiply(kern, img)

    fft_windowed = images.fourier(img_windowed, window=None)
    fft = images.fourier(img, window='hann')
    assert np.array_equal(fft, fft_windowed)

    # Test shift
    img = np.random.rand(41, 51)
    fft = images.fourier(img, shift=False)
    fft_shift = images.fourier(img, shift=True)
    assert np.array_equal([41, 51], fft.shape)
    assert np.array_equal([41, 51], fft_shift.shape)

    img = np.random.rand(43, 51)
    fft = images.fourier(img, shift=False)
    fft_shift = images.fourier(img, shift=True)
    assert np.array_equal([43, 51], fft.shape)
    assert np.array_equal([43, 51], fft_shift.shape)

    img = np.random.rand(51, 43)
    fft = images.fourier(img, shift=False)
    fft_shift = images.fourier(img, shift=True)
    assert np.array_equal([51, 43], fft.shape)
    assert np.array_equal([51, 43], fft_shift.shape)

    # Test padding
    # First, symmetric input
    img = np.random.rand(50, 50)
    fft = images.fourier(img, shift=False, pad=25)
    assert np.array_equal([101, 101], fft.shape)

    # List input
    fft = images.fourier(img, shift=False, pad=[10, 15])
    assert np.array_equal([71, 81], fft.shape)

    # Nonsense input
    with raises(ValueError) as cm:
        fft = images.fourier(img, shift=False, pad=[10, 15, 9])

    assert "Option 'pad' must be int or list/ndarray of length 2" == str(cm.value)


def test_get_gradients():
    """This only tests for failure, not for correct gradient calculation..."""

    testing.needs_internet()
    im_file = testing.get_remote_file(TEST_GREY_FILENAME)
    im = imageio.imread(im_file)

    methods = ["scharr", "sobel", "dog", "gradient"]

    for method in methods:
        gx, gy = images.get_gradients(im, method=method)

    # Test kwargs options for DOG, LOG
    gx, gy  = images.get_gradients(im, method='dog', sigma=4)

    # Color input
    im_file = testing.get_remote_file(TEST_RGB_FILENAME)
    im = imageio.imread(im_file)

    with raises(ValueError) as cm:
        images.get_gradients(im, method='scharr')

    assert "Gradient calculation only works on 2D images right now. " \
           "For multi-dimensional arrays, use numpy's gradient() instead." == str(cm.value)

    return


def test_get_edges():
    """This only tests for failure, not for correct edge detection..."""

    testing.needs_internet()
    im_file = testing.get_remote_file(TEST_GREY_FILENAME)
    im = imageio.imread(im_file)

    methods = ["scharr", "sobel", "dog", "log", "gradient"]

    for method in methods:
        edge = images.get_edges(im, method=method)

    # Test kwargs options for DOG, LOG
    edge = images.get_edges(im, method='log', sigma=4)
    edge = images.get_edges(im, method='dog', sigma=4)

    # Color input
    im_file = testing.get_remote_file(TEST_RGB_FILENAME)
    im = imageio.imread(im_file)

    with raises(ValueError) as cm:
        images.get_gradients(im, method='scharr')

    assert "Gradient calculation only works on 2D images right now. " \
           "For multi-dimensional arrays, use numpy's gradient() instead." == str(cm.value)

    return
