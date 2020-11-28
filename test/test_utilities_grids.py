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


"""Tests for plenpy.utilities.grids module.

"""

import numpy as np
import os
from pytest import raises, approx
from unittest.mock import patch

import plenpy.logg
from plenpy import testing
from plenpy.utilities import grids

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


def test_grid_parameters():
    """Test the GridParameters class"""

    size = 10
    test_params = grids.GridParameters(size=size)

    assert np.array_equal(test_params.size, np.asarray([10, 10]))
    assert np.array_equal(test_params.offset, np.asarray([0, 0]))
    assert np.array_equal(test_params.spacing, np.asarray([14]))
    assert test_params.rotation == 0.0

    size = [10, 15]
    offset = [20, 25]
    spacing = [30, 35]
    rotation = -0.5
    test_params = grids.GridParameters(size=size, offset=offset,
                                       spacing=spacing, rotation=rotation)

    assert np.array_equal(test_params.size, np.asarray(size))
    assert np.array_equal(test_params.offset, np.asarray(offset))
    assert np.array_equal(test_params.spacing, np.asarray(spacing))
    assert test_params.rotation == -0.5

    size = np.asarray([10, 15])
    offset = np.asarray([20, 25])
    spacing = np.asarray([30, 35])
    rotation = -0.5
    crop = 3
    test_params = grids.GridParameters(size=size, offset=offset,
                                       spacing=spacing, rotation=rotation)

    assert np.array_equal(test_params.size, np.asarray(size))
    assert np.array_equal(test_params.offset, np.asarray(offset))
    assert np.array_equal(test_params.spacing, np.asarray(spacing))
    assert test_params.rotation == -0.5

    # Test equal operator
    test_params_1 = grids.GridParameters(size=size, offset=offset,
                                         spacing=spacing, rotation=rotation)

    test_params_2 = grids.GridParameters(size=size, offset=offset,
                                         spacing=spacing, rotation=rotation)

    assert test_params_1 == test_params_2

    return


def test_abstract_grid():

    # Create Mock Patch Abstract Base Classes
    abc_patch = patch.multiple(grids.AbstractGrid, __abstractmethods__=set())
    abc_patch.start()

    # Test error handling
    with raises(ValueError) as cm:
        test_grid = grids.AbstractGrid(grid_parameters='nonsense')

    assert ("The supplied grid_parameters are not an instance of"
            "the GridParameters class.") == str(cm.value)

    with raises(ValueError) as cm:
        test_grid = grids.AbstractGrid(grid_parameters=1)

    assert ("The supplied grid_parameters are not an instance of"
            "the GridParameters class.") == str(cm.value)

    # Test constructor
    size = np.asarray([100, 100])
    n = 10
    m = 20
    test_params = grids.GridParameters(size=size)

    test_grid = grids.AbstractGrid(test_params)

    assert test_grid.gridParameters == test_params
    # In Abstract clas, the grid points are not calculated
    assert test_grid.gridPoints is None
    assert test_grid.n is None
    assert test_grid.m is None
    assert test_grid.cropAmount is None

    # Test properties
    test_grid._n = n
    test_grid._m = m
    test_grid._cropAmount = 10
    assert test_grid.n == n
    assert test_grid.m == m
    assert test_grid.cropAmount == 10

    test_vector = np.asarray([[1,     2],
                              [11,    22],
                              [111,   222],
                              [1111,  2222],
                              [11111, 22222]])

    test_grid._gridPoints = test_vector

    assert np.array_equal(test_grid.gridPoints, test_vector)
    assert np.array_equal(test_grid.x, test_vector[:, 0])
    assert np.array_equal(test_grid.y, test_vector[:, 1])

    # Test equality operator
    test_grid_1 = grids.AbstractGrid(test_params)
    test_grid_2 = grids.AbstractGrid(test_params)

    assert test_grid_1 == test_grid_2

    abc_patch.stop()
    return


def test_abstract_grid_crop():

    # Create Mock Patch Abstract Base Classes
    abc_patch = patch.multiple(grids.AbstractGrid, __abstractmethods__=set())
    abc_patch.start()

    # Test constructor
    size = np.asarray([100, 100])
    n = 10
    m = 20
    crop = 10
    test_params = grids.GridParameters(size=size)

    test_grid = grids.AbstractGrid(test_params)

    test_vector = np.asarray([[9, 5],
                              [9, 9],
                              [9, 10],
                              [9, 50],
                              [9, 89],
                              [9, 90],
                              [10, 5],
                              [10, 9],
                              [10, 10],
                              [10, 50],
                              [10, 89],
                              [10, 90],
                              [50, 5],
                              [50, 9],
                              [50, 10],
                              [50, 50],
                              [50, 89],
                              [50, 90],
                              [89, 5],
                              [89, 9],
                              [89, 10],
                              [89, 50],
                              [89, 89],
                              [89, 90],
                              [90, 5],
                              [90, 9],
                              [90, 10],
                              [90, 50],
                              [90, 89],
                              [90, 90],
                              ])

    test_vector_crop = np.asarray([
                              [10, 10],
                              [10, 50],
                              [10, 89],
                              [50, 10],
                              [50, 50],
                              [50, 89],
                              [89, 10],
                              [89, 50],
                              [89, 89]
                              ])

    test_grid._gridPoints = test_vector
    test_grid._m = 6
    test_grid._n = 5

    # Pixels with 0 < x < 10 and 90 < x < 100 will be cropped
    test_grid.crop(crop=crop)

    assert test_grid.cropAmount == crop
    assert test_grid.m == 3
    assert test_grid.n == 3
    assert np.array_equal(test_grid.gridPoints, test_vector_crop)

    # Test case with different crop
    crop = 6

    test_vector_crop = np.asarray([
                              [9, 9],
                              [9, 10],
                              [9, 50],
                              [9, 89],
                              [9, 90],
                              [10, 9],
                              [10, 10],
                              [10, 50],
                              [10, 89],
                              [10, 90],
                              [50, 9],
                              [50, 10],
                              [50, 50],
                              [50, 89],
                              [50, 90],
                              [89, 9],
                              [89, 10],
                              [89, 50],
                              [89, 89],
                              [89, 90],
                              [90, 9],
                              [90, 10],
                              [90, 50],
                              [90, 89],
                              [90, 90],
                              ])

    test_grid._gridPoints = test_vector
    test_grid._m = 6
    test_grid._n = 5

    # Pixels with 0 < x < 10 and 90 < x < 100 will be cropped
    test_grid.crop(crop=crop)

    assert test_grid.cropAmount == crop
    assert test_grid.m == 5
    assert test_grid.n == 5
    assert np.array_equal(test_grid.gridPoints, test_vector_crop)

    # Test case with no effective cropping
    crop = 3
    test_grid._gridPoints = test_vector
    test_grid._m = 6
    test_grid._n = 5
    test_grid.crop(crop=crop)

    assert test_grid.cropAmount == crop
    assert test_grid.m == 6
    assert test_grid.n == 5
    assert np.array_equal(test_grid.gridPoints, test_vector)

    abc_patch.stop()
    return


def test_abstract_grid_add_noise_external():

    # Create Mock Patch Abstract Base Classes
    abc_patch = patch.multiple(grids.AbstractGrid, __abstractmethods__=set())
    abc_patch.start()

    # Init grid
    size = np.asarray([1000, 1000])
    test_params = grids.GridParameters(size=size)
    test_grid = grids.AbstractGrid(test_params)
    test_vector = 10000 * np.random.rand(1000000, 2)
    test_grid._gridPoints = test_vector.astype(dtype=np.float64)

    # Init noise
    mu = 0.0
    sigma = 0.1
    mean = mu * np.ones(2)
    cov = sigma ** 2 * np.eye(2, 2, dtype=np.float64)
    length = test_grid.gridPointsLength
    noise = np.random.multivariate_normal(mean=mean, cov=cov, size=length)

    # Add noise
    test_grid.add_noise_external(noise=noise)

    diff = test_grid.gridPoints - noise
    assert np.allclose(diff, test_vector)

    abc_patch.stop()
    return


def test_abstract_grid_add_noise_gaussian():

    # Create Mock Patch Abstract Base Classes
    abc_patch = patch.multiple(grids.AbstractGrid, __abstractmethods__=set())
    abc_patch.start()

    size = np.asarray([1000, 1000])
    mu = 0.0
    sigma = 0.1
    test_params = grids.GridParameters(size=size)

    test_grid = grids.AbstractGrid(test_params)
    test_vector = 10000*np.random.rand(1000000, 2)
    test_grid._gridPoints = test_vector.astype(dtype=np.float64)
    test_grid.add_noise_gaussian(mu=mu, sigma=sigma)

    diff = test_vector - test_grid.gridPoints
    assert np.mean(diff, axis=0)[0] == approx(mu, abs=1e-3)
    assert np.mean(diff, axis=0)[1] == approx(mu, abs=1e-3)

    assert np.std(diff, axis=0)[0] == approx(sigma, abs=1e-3)
    assert np.std(diff, axis=0)[1] == approx(sigma, abs=1e-3)

    abc_patch.stop()
    return


def test_hex_grid():

    # Test regular grid spacing
    size = [101, 201]
    spacing = 10
    test_params = grids.GridParameters(size=size, spacing=spacing)

    test_grid = grids.HexGrid(test_params)

    assert test_grid.gridParameters.spacing[0] == np.sqrt(3)*spacing
    assert test_grid.gridParameters.spacing[1] == spacing

    # Test grid points

    size = [101, 201]
    spacing = [10, 20]

    # expected number of points per dimension:
    n = 20
    m = 10

    test_params = grids.GridParameters(size=size, spacing=spacing)

    test_grid = grids.HexGrid(test_params)

    assert test_grid.n == n
    assert test_grid.m == m
    assert test_grid.gridPoints.shape[0] == n*m
    assert test_grid.gridPoints.shape[1] == 2

    # Naively reconstruct grid points (generic algorithm)
    for i in range(0, n):
        for j in range(0, m):

            x = 0.5 * spacing[0] * i
            y1 = spacing[1] * j
            y2 = spacing[1] * (j + 0.5)

            if i % 2 == 0:
                assert test_grid.gridPoints[i*m + j, 0] == x
                assert test_grid.gridPoints[i*m + j, 1] == y1

            else:
                assert test_grid.gridPoints[i*m + j, 0] == x
                assert test_grid.gridPoints[i*m + j, 1] == y2

    return


def test_rect_grid():

    # Test regular grid spacing
    size = [101, 201]
    spacing = 10
    test_params = grids.GridParameters(size=size, spacing=spacing)

    test_grid = grids.RectGrid(test_params)

    assert test_grid.gridParameters.spacing[0] == spacing
    assert test_grid.gridParameters.spacing[1] == spacing

    # Test grid points

    size = [101, 201]
    spacing = [10, 20]

    # expected number of points per dimension:
    n = 10
    m = 10

    test_params = grids.GridParameters(size=size, spacing=spacing)

    test_grid = grids.RectGrid(test_params)

    assert test_grid.n == n
    assert test_grid.m == m
    assert test_grid.gridPoints.shape[0] == n*m
    assert test_grid.gridPoints.shape[1] == 2

    # Naively reconstruct grid points (generic algorithm)
    for i in range(0, n):
        for j in range(0, m):

            x = spacing[0] * i
            y = spacing[1] * j

            assert test_grid.gridPoints[i*m + j, 0] == x
            assert test_grid.gridPoints[i*m + j, 1] == y

    return

def test_save_grid():

    # Test grid parameters
    size = [101, 201]
    spacing = 10
    rotation = 0.01
    offset = 2.5

    test_params = grids.GridParameters(size=size,
                                       spacing=spacing,
                                       rotation=rotation,
                                       offset=offset)

    # Create Grids
    hex_grid_tmp = grids.HexGrid(grid_parameters=test_params)
    rect_grid_tmp = grids.HexGrid(grid_parameters=test_params)

    hex_grid_tmp.crop(20)
    rect_grid_tmp.crop(20)

    # Save grids
    path_tmp = os.path.abspath('./')
    hex_path = os.path.join(path_tmp, 'hex_file.npz')
    rect_path = os.path.join(path_tmp, 'rect_file.npz')

    hex_grid_tmp.save(hex_path)
    rect_grid_tmp.save(rect_path)

    # Load grids
    hex_grid_loaded = grids.HexGrid.from_file(hex_path)
    rect_grid_loaded = grids.RectGrid.from_file(rect_path)

    assert hex_grid_tmp == hex_grid_loaded
    assert rect_grid_tmp == rect_grid_loaded

    os.remove(hex_path)
    os.remove(rect_path)

    return