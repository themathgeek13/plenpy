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


"""Tests for plenpy.utilities.kernels module.

"""

import numpy as np
from pytest import raises, approx

import plenpy.logg
from plenpy.utilities import kernels

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


def test_get_avail_names():

    list = kernels.get_avail_names()

    # Basic type check.
    # Check that get_kernel() can use all of the provided names.
    # Check that all kernel names are implemented

    for element in list:
        assert type(element) == str
        kernels.get_kernel(name=element, size=5, option=1)

    return


def test_kernel_disk():

    kernel = kernels.get_kernel("disk", size=15)
    disk_kernel = kernels.get_kernel_disk(size=15)

    assert np.array_equal(kernel, disk_kernel)
    assert kernel.shape == (15, 15)
    assert kernel.dtype == np.float64
    assert 1.0 == approx(kernel.sum())

    kernel = kernels.get_kernel("disk", size=15)
    disk_kernel = kernels.get_kernel_disk(size=15, radius=7)

    assert np.array_equal(kernel, disk_kernel)
    assert kernel.shape == (15, 15)
    assert kernel.dtype == np.float64
    assert 1.0 == approx(kernel.sum())

    kernel = kernels.get_kernel("disk", size=15)
    disk_kernel = kernels.get_kernel_disk(size=15, radius_y=7)

    assert np.array_equal(kernel, disk_kernel)
    assert kernel.shape == (15, 15)
    assert kernel.dtype == np.float64
    assert 1.0 == approx(kernel.sum())

    kernel = kernels.get_kernel("disk",
                                size=15, size_y=17,
                                option=3, option_y=5)
    disk_kernel = kernels.get_kernel_disk(size=15, size_y=17,
                                          radius=3, radius_y=5)

    assert np.array_equal(kernel, disk_kernel)
    assert kernel.shape, (15, 17)
    assert kernel.dtype, np.float64
    assert 1.0 == approx(kernel.sum())

    # test disk kernel values
    size_x = 51
    size_y = 61
    radius_x = 21
    radius_y = 29
    kernel = kernels.get_kernel_disk(size=size_x,
                                     radius=radius_x,
                                     size_y=size_y,
                                     radius_y=radius_y)

    # set all non zero values to 1.0 for easy comparison
    kernel = np.divide(kernel, kernel.max())

    mu_x = (size_x - 1)/2.0
    mu_y = (size_y - 1)/2.0

    x, y = np.meshgrid(np.arange(0, size_x, 1), np.arange(0, size_y, 1))
    mask = (x - mu_x)**2/radius_x**2 + (y - mu_y)**2/radius_y**2 <= 1

    # Check that all masked values are 1.0, and all unmask values are 0.0
    assert np.all(kernel[mask.T])
    assert not np.all(kernel[~mask.T])

    return


def test_kernel_gauss():

    kernel = kernels.get_kernel("gauss", size=15)
    gauss_kernel = kernels.get_kernel_gauss(size=15)
    assert np.array_equal(kernel, gauss_kernel)
    assert gauss_kernel.shape == (15, 15)
    assert gauss_kernel.dtype == np.float64
    assert 1.0 == approx(gauss_kernel.sum())

    # check symmetries
    assert kernels.is_symmetric(gauss_kernel, symmetry=None)

    kernel = kernels.get_kernel("gauss", size=14, size_y=15,
                                option=3, option_y=4)
    gauss_kernel = kernels.get_kernel_gauss(size=14, size_y=15,
                                            sigma=3, sigma_y=4)
    assert np.array_equal(kernel, gauss_kernel)
    assert gauss_kernel.shape == (14, 15)
    assert gauss_kernel.dtype == np.float64
    assert 1.0 == approx(gauss_kernel.sum())

    kernel = kernels.get_kernel_gauss(size=14, size_y=14,
                                      sigma=3, sigma_y=3)
    gauss_kernel = kernels.get_kernel_gauss(size=14, sigma=3)
    assert np.array_equal(kernel, gauss_kernel)
    assert gauss_kernel.shape == (14, 14)
    assert gauss_kernel.dtype == np.float64
    assert 1.0 == approx(gauss_kernel.sum())

    kernel = kernels.get_kernel_gauss(size=14, size_y=14,
                                      sigma=14.0/5.0, sigma_y=3)
    gauss_kernel = kernels.get_kernel_gauss(size=14, sigma_y=3)
    assert np.array_equal(kernel, gauss_kernel)
    assert gauss_kernel.shape == (14, 14)
    assert gauss_kernel.dtype == np.float64
    assert 1.0 == approx(gauss_kernel.sum())

    # Test kernel values
    kernel = kernels.get_kernel_gauss(1)
    assert kernel == 1.0

    kernel = kernels.get_kernel_gauss(2)
    assert np.array_equal(kernel, np.asarray([[0.25, 0.25], [0.25, 0.25]]))

    # test that kernel is of gauss form along main axes
    for size in [3, 7, 15, 25, 57, 333]:
        for sigma in [size/3, size/5, size/7]:
            # Shift gauss reference to array middle
            mu = (size - 1)/2.0

            sigma_x = sigma
            sigma_y = sigma

            # test asymmetric kernel
            img = kernels.get_kernel_gauss(size,
                                           sigma=sigma_x,
                                           sigma_y=sigma_y)

            # Create a gauss reference, normalized to 1
            x = np.arange(0, size, 1)
            gauss_ref = np.exp(-0.5*(((x - mu)/sigma_x)**2))

            # Get slice of 2D gauss kernel and normalize to one
            gauss_from_kern = img[:,(size-1)//2]
            gauss_from_kern = np.divide(gauss_from_kern,
                                        gauss_from_kern.max())

            # Check that all values are equivalent up to float16 precision

            assert np.array_equal(gauss_ref.astype(np.float16),
                                  gauss_from_kern.astype(np.float16))

            # Get slice along other axis, normalize to 1
            # Sigma in y directions is now larger
            gauss_ref = np.exp(-0.5 * (((x - mu)/sigma_y)**2))
            gauss_from_kern = img[(size - 1)//2, :]
            gauss_from_kern = np.divide(gauss_from_kern,
                                        gauss_from_kern.max())

            assert np.array_equal(gauss_ref.astype(np.float16),
                                  gauss_from_kern.astype(np.float16))

    return


def test_kernel_hann():

    kernel = kernels.get_kernel("hann", size=15)
    hann_kernel = kernels.get_kernel_hann(size=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (15, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    # check symmetries
    assert kernels.is_symmetric(hann_kernel, "axial")
    assert kernels.is_symmetric(hann_kernel, "rotational")

    kernel = kernels.get_kernel("hann", size=13, size_y=15)
    hann_kernel = kernels.get_kernel_hann(size=13, size_y=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (13, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    return


def test_kernel_hann_rotational():

    kernel = kernels.get_kernel("hann_rotational", size=15)
    hann_kernel = kernels.get_kernel_hann_rotational(size=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (15, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    # check symmetries
    assert kernels.is_symmetric(hann_kernel, "axial")
    assert kernels.is_symmetric(hann_kernel, "rotational")

    kernel = kernels.get_kernel("hann_rotational", size=13, size_y=15)
    hann_kernel = kernels.get_kernel_hann_rotational(size=13, size_y=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (13, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    kernel = kernels.get_kernel("hann_rotational", size=13, size_y=11)
    hann_kernel = kernels.get_kernel_hann_rotational(size=13, size_y=11)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (13, 11)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    return


def test_kernel_hamming():

    kernel = kernels.get_kernel("hamming", size=15)
    hann_kernel = kernels.get_kernel_hamming(size=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (15, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    # check symmetries
    assert kernels.is_symmetric(hann_kernel, "axial")
    assert kernels.is_symmetric(hann_kernel, "rotational")

    kernel = kernels.get_kernel("hamming", size=13, size_y=15)
    hann_kernel = kernels.get_kernel_hamming(size=13, size_y=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (13, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    return


def test_kernel_kaiser():

    kernel = kernels.get_kernel("kaiser", size=15)
    hann_kernel = kernels.get_kernel_kaiser(size=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (15, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    # check symmetries
    assert kernels.is_symmetric(hann_kernel, "axial")
    assert kernels.is_symmetric(hann_kernel, "rotational")

    kernel = kernels.get_kernel("kaiser", size=13, size_y=15)
    hann_kernel = kernels.get_kernel_kaiser(size=13, size_y=15)
    assert np.array_equal(kernel, hann_kernel)
    assert hann_kernel.shape == (13, 15)
    assert hann_kernel.dtype == np.float64
    assert 1.0 == approx(hann_kernel.sum())

    return


# Test get_kernel parsing and error handling
def test_get_kernel_parsing():

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel("noValidKernelName", 3)
    assert ("Specified argument name 'noValidKernelName' is not one of the "
            f"recognized kernels: {kernels.get_avail_names()}") == str(
        cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_gauss(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_disk(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hann(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hann_rotational(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hamming(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_kaiser(3.5)
    assert "Specified size 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_disk(3, 1, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_gauss(3, 1, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hann(3, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hann_rotational(3, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_hamming(3, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.get_kernel_kaiser(3, size_y=3.5)
    assert "Specified size_y 3.5 is not an integer." == str(cm.value)

    # Test a not implemented kernel name
    kernels.__availableKernelNames.append("notImplementedKernel")
    with raises(NotImplementedError) as cm:
        kernels.get_kernel(name="notImplementedKernel", size=5)
    assert ("The specified kernel name "
            "'notImplementedKernel' is not implemented.") == str(cm.value)

    return


# Test symmetry utility function
def test_symmetry():

    m = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]])

    assert not kernels.is_symmetric(m, symmetry="axial")
    assert not kernels.is_symmetric(m, symmetry="rotational")
    assert not kernels.is_symmetric(m, symmetry=None)

    m = np.array(
        [[1, 2, 1],
         [2, 3, 2],
         [1, 2, 1]])

    assert kernels.is_symmetric(m, symmetry="axial")
    assert kernels.is_symmetric(m, symmetry="rotational")
    assert kernels.is_symmetric(m, symmetry=None)

    m = np.array(
        [[1, 1, 1],
         [2, 3, 2],
         [1, 1, 1]])

    assert kernels.is_symmetric(m, symmetry="axial")
    assert not kernels.is_symmetric(m, symmetry="rotational")
    assert not kernels.is_symmetric(m, symmetry=None)

    m = np.array(
        [[1, 2, 3, 4, 1],
         [4, 1, 2, 1, 2],
         [3, 2, 5, 2, 3],
         [2, 1, 2, 1, 4],
         [1, 4, 3, 2, 1]])

    assert not kernels.is_symmetric(m, symmetry="axial")
    assert kernels.is_symmetric(m, symmetry="rotational")
    assert not kernels.is_symmetric(m, symmetry=None)

    m = np.array(
        [[1, 1],
         [1, 1]])

    assert kernels.is_symmetric(m, symmetry="axial")
    assert kernels.is_symmetric(m, symmetry="rotational")
    assert kernels.is_symmetric(m, symmetry=None)

    return


# Test is_symmetric parsing and error handling
def test_is_symmetric_parsing():

    m = np.array(
        [[1, 1],
         [1, 1]])

    with raises(ValueError) as cm:
        kernel = kernels.is_symmetric(m, symmetry="noValidSymmetry")
    assert ("Symmetry name 'noValidSymmetry' invalid. "
            "Please specify a valid symmetry name.") == str(cm.value)

    with raises(ValueError) as cm:
        kernel = kernels.is_symmetric(m, symmetry=0.3)
    assert ("Symmetry name '0.3' invalid. "
            "Please specify a valid symmetry name.") == str(cm.value)

    return
