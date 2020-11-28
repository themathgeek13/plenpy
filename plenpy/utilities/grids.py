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
Module defining :class:`GridParameters` and different grid classes.

The grid parameter class and the grid classes model regular and slightly
irregular grids.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any

import numpy as np
from numpy.core.multiarray import ndarray

import plenpy.logg
from plenpy.utilities.misc import rot_matrix

logger = plenpy.logg.get_logger()

__all__ = ['GridParameters', 'HexGrid', 'RectGrid']


class GridParameters(object):
    """Class to save grid parameters of a grid.

    The grid parameters are an abstract representation of a grid and store
    all necessary geometrical information of a grid, except its layout, i.e.
    hexagonal or rectangular for example.

    Attributes:
        _size: Size [x, y] of the grid in pixels.

        _spacing: Spacing vector [x_spacing, y_spacing] between grid points
            in pixels.

        _rotation: Rotation of the grid in radians.

        _offset: Offset vector [x_offset, y_offset] of the first grid point
            in pixels.

    """

    def __init__(self,
                 size: Union[int, List[int], ndarray],
                 spacing: Union[float, List[float], ndarray] = 14,
                 rotation: float = 0.0,
                 offset: Union[float, List[float], ndarray] = 0):
        """:class:`GridParameters` base class initialization.

        Args:
            size: Size of the grid in pixels. That is, the region
                [0, size_x] x [0, size_y] where the grid coordinates will
                reside. This does not correspond to the actual number of grid
                points. If ``size`` is of type ``int``` a quadratic size is
                assumed.

            spacing: Spacing vector [x_spacing, y_spacing] of the grid.
                If ``float`` is given, a symmetric spacing is assumed.

            rotation: Rotation of the grid in radians.

            offset: Offset vector [x_offset, y_offset] of the first grid point
                in pixels. If ``float`` is given, a symmetric offset is
                assumed. Default: [0, 0]

        """
        if isinstance(size, int) or isinstance(size, float):
            self._size = np.asarray([size])
        else:
            self._size = np.asarray(size)

        if self._size.shape[0] == 1:
            self._size = np.array([self._size[0], self._size[0]])

        if isinstance(offset, int) or isinstance(offset, float):
            self._offset = np.asarray([offset, offset])
        else:
            self._offset = np.asarray(offset)

        if isinstance(spacing, int) or isinstance(spacing, float):
            self._spacing = np.asarray([spacing])
        else:
            self._spacing = np.asarray(spacing)

        self._rotation = rotation

        return

    def __eq__(self, other):

        result = False

        if np.array_equal(self.size, other.size) and \
                np.array_equal(self.offset, other.offset) and \
                np.array_equal(self.spacing, other.spacing) and \
                np.array_equal(self.rotation, other.rotation):

            result = True

        return result

    @property
    def size(self):
        return self._size

    @property
    def offset(self):
        return self._offset

    @property
    def spacing(self):
        return self._spacing

    @property
    def rotation(self):
        return self._rotation


class AbstractGrid(ABC):
    """AbstractGrid base class.

    This is the base class for all actual grid class implementations.

    Attributes:
        _gridParameters (GridParameters): Parameters of the grid.

        _gridPoints (ndarray): Array containing all grid point coordinates.

        _n (int): Number of grid points in x-direction. When the grid is
            cropped, this might not be a 100% accurate.

        _m (int): Number of grid points in y-direction. When the grid is
            cropped, this might not be a 100% accurate.

        _cropAmount (float): Number of pixels have been cropped from the
            coordinates from the border of the grid. If no cropping is applied,
            the value is set to ``None``.

    """

    def __init__(self,
                 grid_parameters: GridParameters,
                 grid_points: Optional[ndarray] = None,
                 noise: Optional[ndarray] = None,
                 n: Optional[int] = None,
                 m: Optional[int] = None,
                 crop_amount: Optional[float] = None):
        """:class:`AbstractGrid` base class initialization.

        Args:
            grid_parameters: Grid parameters of the grid.

        """

        if not isinstance(grid_parameters, GridParameters):
            raise ValueError(
                "The supplied grid_parameters are not an instance of"
                "the GridParameters class.")

        self._gridParameters: GridParameters = grid_parameters
        self._gridPoints = grid_points
        self._noise = noise
        self._n = n
        self._m = m
        self._cropAmount = crop_amount

        return

    def __eq__(self, other):

        result = False

        if self._gridParameters == other._gridParameters and \
                np.array_equal(self._gridPoints, other._gridPoints) and\
                self._n == other._n and \
                self._m == other._m and \
                self._cropAmount == other._cropAmount:
            result = True

        return result

    @classmethod
    def from_file(cls, path: Any):

        data = np.load(path, allow_pickle=True)

        grid_parameters_size = data['grid_parameters_size']
        grid_parameters_spacing = data['grid_parameters_spacing']
        grid_parameters_rotation = data['grid_parameters_rotation']
        grid_parameters_offset = data['grid_parameters_offset']
        grid_parameters = GridParameters(size=grid_parameters_size,
                                         spacing=grid_parameters_spacing,
                                         rotation=grid_parameters_rotation,
                                         offset=grid_parameters_offset)

        grid_points = data['grid_points']
        noise = data['noise']
        m = data['m']
        n = data['n']
        crop_amount = data['crop_amount']

        return cls(grid_parameters=grid_parameters,
                   grid_points=grid_points,
                   noise=noise,
                   m=m,
                   n=n,
                   crop_amount=crop_amount)

    def save(self, path: Any):

        np.savez(path,
                 grid_parameters_size = self._gridParameters.size,
                 grid_parameters_spacing=self._gridParameters.spacing,
                 grid_parameters_rotation=self._gridParameters.rotation,
                 grid_parameters_offset=self._gridParameters.offset,
                 grid_points=self._gridPoints,
                 noise=self._noise,
                 n=self._n,
                 m=self._m,
                 crop_amount=self._cropAmount
                 )

        return

    def crop(self, crop: float):
        """Crop the grid points from all borders.

        CAUTION: Cropping might break the gridParameters, such as
        the offset value or number of grid points in each direction for
        strongly rotated grids. Use with skepsis.

        Args:
            crop: Pixels to crop.

        """

        indices = []
        x_max, y_max = self.gridParameters.size
        for k in range(0, self.gridPointsLength):
            if self.gridPoints[k][0] < crop or \
                    self.gridPoints[k][1] < crop or \
                    self.gridPoints[k][0] > (x_max - 1 - crop) or \
                    self.gridPoints[k][1] > (y_max - 1 - crop):
                indices.append(k)

        # delete elements that are outside the border
        self._gridPoints = np.delete(self._gridPoints, indices, axis=0)

        # Set crop member and recalculate number of points
        self._cropAmount = crop

        # Calulcate number of cropped points per dimension
        indices = np.asarray(indices)

        # First, go through y-dimension
        # Find indices with same y-coordinate
        seq = np.sort(indices % self.m)
        # Split into arrays with same index
        idx = np.split(seq, np.where(np.diff(seq) != 0)[0] + 1)
        # Iterate and find smallest list
        length_y = np.inf
        for item in idx:
            length_y = min(length_y, len(item))

        # For the x-dimension, find indices with same x-coordinate
        seq = np.sort(indices // self.m)
        # Split into arrays with same index
        idx = np.split(seq, np.where(np.diff(seq) != 0)[0] + 1)
        length_x = np.inf
        for item in idx:
            length_x = min(length_x, len(item))

        if length_y < self.n:
            self._n -= length_y

        if length_x < self.m:
            self._m -= length_x

        # Reset offset parameter
        self._gridParameters._offset = self.gridPoints[0]

        return

    def add_noise_external(self,
                           noise: ndarray):
        """Add a noise vector to every grid point.

        The noise is passed as a numpy array

        Args:
            noise: The noise vector to be added.
        """
        # Add noise to grid coordinates
        self._gridPoints += noise
        self._noise = noise

        return

    def add_noise_gaussian(self,
                  mu: float = 0.0,
                  sigma: float = 0.3):
        """Add a noise vector to every grid point.

        The noise is drawn from a bivariate normal distribution with mean
        ``mu`` and standard deviation ``sigma``, i.e. a
        covariance matrix sigma**2 * ID_2.

        Args:
            mu: Mean of the normal distribution

            sigma:  Standard deviation of the normal distribution

        """

        mean = mu * np.ones(2)
        cov = sigma**2 * np.eye(2, 2, dtype=np.float64)
        size = self._gridPoints.shape[0]

        # Create 2D random vectors draw from a bivariate normal distribution
        rand_vectors = np.random.multivariate_normal(mean=mean, cov=cov,
                                                     size=size)

        # Add the random vectors to the coordinates
        self._gridPoints += rand_vectors

        return

    @property
    def gridParameters(self):
        return self._gridParameters

    @property
    def gridPoints(self):
        return self._gridPoints

    @property
    def gridPointsLength(self):
        return self._gridPoints.shape[0]

    @property
    def x(self):
        return self._gridPoints[:, 0]

    @property
    def y(self):
        return self._gridPoints[:, 1]

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def cropAmount(self):
        return self._cropAmount

    @abstractmethod
    def _calc_grid_points(self):
        """Calculate the grid points.

        This must be implemented by every inherited class separately.
        Sets the ``_gridPoints`` attribute and should be called upon
        initialization.

        """
        pass

    @abstractmethod
    def _calc_grid_points_num(self):
        """Calculate the number of grid points in every direction.

        This must be implemented by every inherited class separately.
        Sets the ``_m`` and ``_n`` attribute and should be called upon
        initialization.

        """
        pass


class HexGrid(AbstractGrid):
    """Hexagonal grid.

    """

    def __init__(self, grid_parameters: GridParameters, **kwargs):
        """:class:`HexGrid` class initialization.

        Args:
            grid_parameters: Grid parameters of the grid.

        """

        super().__init__(grid_parameters, **kwargs)

        # For regular grid, calculate missing spacing parameter
        if self.gridParameters.spacing.shape[0] == 1:

            self._gridParameters._spacing = np.array(
                [np.sqrt(3) * self._gridParameters.spacing[0],
                 self._gridParameters.spacing[0]])

        # Calculate number of grid points in each direction
        if self._m is None or self._n is None:
            self._calc_grid_points_num()

        # Calculate grid points
        if self._gridPoints is None:
            self._calc_grid_points()

            # Add noise if set (for example after reading from a file)
            if self._noise is not None:
                self.add_noise_external(self._noise)

        return

    def _calc_grid_points_num(self):
        """Calculate the number of grid points of a hexagonal grid
            defined by ``grid_parameters``.

        """

        self._n = int(
            (self.gridParameters.size[0] - self.gridParameters.offset[0]) //
            (0.5 * self.gridParameters.spacing[0]))

        self._m = int(
            (self.gridParameters.size[1] - self.gridParameters.offset[1]) //
            self.gridParameters.spacing[1])

        return

    def _calc_grid_points(self):
        """
        Calculate the grid points of a hexagonal grid
        defined by ``grid_parameters``.

        """
        # Extract values so it's a bit easier on the eyes
        spacing = self.gridParameters.spacing
        offset = self.gridParameters.offset
        alpha = self.gridParameters.rotation
        n = self.n
        m = self.m

        # Calculate grid coordinate vectors
        # X coordinates in steps of 0.5*spacing_x
        x = offset[0] + 0.5*spacing[0]*np.arange(0, n, 1)

        # Y coordinates in steps of spacing_y
        # Two vectors that are offset by 0.5*spacing_y due to hexagonal grid
        y_0 = offset[1] + spacing[1] * np.arange(0, m, 1)
        y_1 = offset[1] + 0.5*spacing[1] + spacing[1]*np.arange(0, m, 1)

        # Init grid coordinate matrix (x_ij, y_ij)
        grid_coords = np.empty((n * m, 2))

        # Fill grid coordinates block-wise
        # All coordinates where i is even
        grid_coords[:, 1].reshape(n, m).T[:, 0::2].T[:] = y_0
        # All coordinates where i is odd
        grid_coords[:, 1].reshape(n, m).T[:, 1::2].T[:] = y_1

        grid_coords[:, 0].reshape(n, m).T[:, :] = x

        # Rotate the points by vector-wise matrix multiplication
        # Rotation matrix:
        if not alpha == 0:
            r = rot_matrix(alpha, unit='radians')
            grid_coords = (r @ (grid_coords.T)).T

        self._gridPoints = grid_coords.astype(dtype=np.float64)

        return


class RectGrid(AbstractGrid):
    """Rectangular grid.

    """

    def __init__(self, grid_parameters: GridParameters, **kwargs):
        """:class:`RectGrid` class initialization.

        Args:
            grid_parameters: Grid parameters of the grid.

        """
        super().__init__(grid_parameters, **kwargs)

        # For regular grid, calculate missing spacing parameter
        if self.gridParameters.spacing.shape[0] == 1:
            self._gridParameters._spacing = np.array(
                [self._gridParameters.spacing[0],
                 self._gridParameters.spacing[0]])

        # Calculate number of grid points in each direction
        if self._m is None or self._n is None:
            self._calc_grid_points_num()

        # Calculate grid points
        if self._gridPoints is None:
            self._calc_grid_points()

            # Add noise if set (for example after reading from a file)
            if self._noise is not None:
                self.add_noise_external(self._noise)

        return

    def _calc_grid_points_num(self):
        """Calculate the number of grid points of a rectangular grid
            defined by ``grid_parameters``.

        """
        self._n = int(
            (self.gridParameters.size[0] - self.gridParameters.offset[0]) //
            self.gridParameters.spacing[0])

        self._m = int(
            (self.gridParameters.size[1] - self.gridParameters.offset[1]) //
            self.gridParameters.spacing[1])

    def _calc_grid_points(self):
        """Calculate the grid points of a rectangular grid
        defined by ``grid_parameters``.

        """

        # Extract values so it's a bit easier on the eyes
        spacing = self.gridParameters.spacing
        offset = self.gridParameters.offset
        alpha = self.gridParameters.rotation
        n = self.n
        m = self.m

        # Calculate grid coordinate vectors
        # X coordinates in steps of spacing_x
        x = offset[0] + spacing[0] * np.arange(0, n, 1)

        # Y coordinates in steps of spacing_y
        y_0 = offset[1] + spacing[1] * np.arange(0, m, 1)

        # Init grid coordinate matrix (x_ij, y_ij)
        # Fill grid coordinates block-wise
        grid_coords = np.empty((n * m, 2))

        grid_coords[:, 1].reshape(n, m).T[:, :].T[:] = y_0
        grid_coords[:, 0].reshape(n, m).T[:, :] = x

        # Rotate the points by vector-wise matrix multiplication
        # Rotation matrix:
        if not alpha == 0:
            r = rot_matrix(alpha, unit='radians')
            grid_coords = (r @ (grid_coords.T)).T

        self._gridPoints = grid_coords.astype(dtype=np.float64)

        return
