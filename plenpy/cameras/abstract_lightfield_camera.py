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
Module defining the :class:`AbstractLightFieldCamera` base class.

This bass class, derived from the :class:`AbstractCamera` base class,
implements methods that all microlens based light field cameras share.
In particular, this class implements the estimation of an ideal grid from
(a collection of) whiteimages.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as nd_filters
import scipy.ndimage.filters as sc_filters
import skimage.transform as skt
from numpy.core.multiarray import ndarray
from scipy import ndimage
from scipy import spatial
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.measurements import find_objects
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from skimage import exposure
from skimage import filters as sk_filters
from skimage.feature import peak_local_max as sk_peak_local_max

import plenpy.logg
from plenpy.cameras.abstract_camera import AbstractCamera
from plenpy.lightfields import LightField
from plenpy.utilities import grids
from plenpy.utilities import images
from plenpy.utilities import kernels
from plenpy.utilities import misc
from plenpy.utilities.misc import get_avail_extensions

logger = plenpy.logg.get_logger()

__all__ = ['AbstractLightFieldCamera']


class AbstractLightFieldCamera(AbstractCamera, metaclass=ABCMeta):
    """AbstractLightFieldCamera base class.

    This is the base class that all micro lens based light field camera classes
    are derived from.
    This class is abstract and objects cannot be initialized directly from it.
    Implements the basic functions needed by a light field camera model.

    The abstract methods :func:`.AbstractCamera.calibrate()` and
    :func:`.AbstractCamera.decode_sensor_image()` are left undefined and
    have to be implemented by derived light field camera classes.

    The camera inherits all attributes from the :class:`AbstractCamera` class.
    Additional attributes are:

    Attributes:
        _microlensSize (float): The microlens diameter in pixels.

        _microlensRadius (float): The microlens radius in pixels, derived from
            the microlens diameter.

        _microlensFocalLength (float): The (ideal) microlens focal length in meters (optional).

        _calDataFilename (str): The name of the calibration data file placed
            in the ``Calibration`` subfolder of the camera. Default is:
            `cal_data.npz`

    """

    def __init__(self,
                 path: Any,
                 microlens_size: float,
                 grid_type: str,
                 ml_focal_length: Optional[float] = None):
        """:class:`AbstractLightFieldCamera` base class initialization.

        Args:
            path: System path to the main camera folder.

            microlens_size: Estimated size of the microlens diameter in pixels.
                For example:\n
                * Lytro Illum: 14
                * Lytro F01: 11

            grid_type: Grid type of the underlying microlens array.
                Either 'hex' or 'rect'.

            ml_focal_length: Focal length of the micro lenses in meters (optional).
        """

        # Call init from AbstractCamera base class
        super().__init__(path)

        self._microlensSize = microlens_size
        self._microlensRadius = int(self._microlensSize / 2.0)
        self._microlensFocalLength = ml_focal_length

        self._calibrationDB = None
        self._whiteImageDB = None

        self._calDataFilename = self.calibrationFolderPath / 'cal_data.npz'

        if grid_type.lower() == 'hex' or grid_type.lower() == 'rect':
            self._gridType: str = grid_type.lower()

        else:
            raise ValueError(f"Unknown grid type '{grid_type}'. "
                             f"Valid types are 'hex' and 'rect'.")

        # Overwrite type hint for light field cameras.
        self._decodedImages: Dict[int, LightField] = {}

        return

    def _save_cal_data(self, filename: Optional[str] = None):
        """Save the calibration data.

        The calibration data is saved as a compressed numpy array file
        in the camera's ``Calibration`` folder.

        """
        if filename is not None:
            filename = self.calibrationFolderPath / filename
            if not filename.suffix == ".npz":
                filename = filename.with_suffix('.npz')
            self._calDataFilename = filename

        logger.info(f"Saving calibration data to {self._calDataFilename}...")
        np.savez_compressed(self.path / self._calDataFilename,
                            calDB=self._calibrationDB,
                            wiDB=self._whiteImageDB)
        logger.info("...done.")

        return

    def _load_cal_data(self, filename: Optional[Any] = None):
        """Load the calibration data from the calibration data file.

        Args:
            filename: Calibration data filename.
                Specify only if it deviates from the default value.

        """
        if filename is not None:
            filename = self.calibrationFolderPath / filename
            if not filename.suffix  == ".npz":
                filename = filename.with_suffix('.npz')
            self._calDataFilename = filename

        logger.info("Loading calibration data....")
        loaded = np.load(self.path / self._calDataFilename, allow_pickle=True)
        self._calibrationDB = loaded['calDB']
        self._whiteImageDB = loaded['wiDB']
        logger.info("...done")

        return

    @abstractmethod
    def _get_wi(self, path: Path, process: bool = True):
        """Read the white image from path and perform preprocessing such as
        contrast stretching, normalization, etc., if desired.

        Args:
            path: Path to the white image, relative to Calibration folder.

            process: Whether to perform preprocessing such as contrast stretching.

        Returns:
            White image as float in range [0, 1].

        """
        # Needs to be implemented by actual camera.
        # Depends on file type and preprocessing, etc.
        pass

    def _create_wi_db(self):
        """Create the white image database and save to self._whiteImageDB.

        """
        logger.info(f"Creating white image database...")

        # Read all available images in Calibration folder
        white_image_paths = []
        for file in (self.path / self.calibrationFolderPath).iterdir():
            # Check that the file is a readable image file
            if file.suffix in get_avail_extensions() and not file.suffix == '.npz':
                # Append relative path
                white_image_paths.append(
                    (self.calibrationFolderPath / file).relative_to(self.path)                )

        white_image_paths = sorted(white_image_paths)

        # Iterate through WI and get meta database
        N = len(white_image_paths)
        for i in range(N):
            path = white_image_paths[i]

            # Get WI entry for database
            if i == 0:
                white_image_db = self._get_wi_db_entry(path)

            else:
                white_image_db = np.append(
                    white_image_db,
                    self._get_wi_db_entry(path))

        self._whiteImageDB = white_image_db
        logger.info("...done")

        return

    @abstractmethod
    def _get_wi_db_entry(self, path: Path) -> Optional[ndarray]:
        """Get a single entry for the white image database.

        The type of entry depends on the actual camera.
        Must contain a 'focal_length' entry.
        This could for example be focus and zoom settings of the white image
        or gamma settings etc.

        Args:
            path: path: Path to the white image, relative to Camera main folder.
                  Used as database identfication

        Returns:
            A ndarray (best a structured array) with the needed infos
            from the whiteimage. If no metadata is available for the camera,
            should return ``None``.

        """
        # Needs to be implemented by actual camera.
        pass

    @staticmethod
    def _get_calibration_db_entry(path: Path,
                                  ideal_grid_params: grids.GridParameters,
                                  align_grid_params: grids.GridParameters,
                                  align_transform: skt.AffineTransform) -> ndarray:
        """Get a single entry for the calibration database.

        Args:
            path: Path to the corresponding white image.

            ideal_grid_params: The ideal grid parameters.

            align_grid_params: The aligned grid parameters.

            align_transform: The transformation needed for alignment.

        Returns:
            A structured numpy array with the needed data.

        """
        # Create structured array for calibration database entry
        db_entry = np.array(
            [(path, ideal_grid_params, align_grid_params, align_transform)],
            dtype=[('path', Path),
                   ('ideal_grid_params', grids.GridParameters),
                   ('align_grid_params', grids.GridParameters),
                   ('align_transform', skt.AffineTransform)])

        return db_entry

    def _est_grid(self, method: str = 'own'):
        """Estimate regular grid(s) that best fits the white image(s).
        For every white image in the Calibration folder, a regular grid is
        estimated.

        Args:
            method: Method for grid estimation. Choices are:
                        - 'own': Proposed method [ref]
                        - 'dans': Dansereau et al.

        """
        logger.info("Estimating ideal grid(s) from white image(s)...")

        white_image_paths = self._whiteImageDB['path']

        # Iterate through WI and estimate grids
        N = len(white_image_paths)
        for i in range(N):
            logger.info(f"Processing whiteimage {i+1} of {N}")
            path = white_image_paths[i]

            wi = self._get_wi(path, process=True)

            # Estimate arguments
            if method == 'own':
                # Estimated magnification factor lambda
                if self._microlensFocalLength is None:
                    raise ValueError("Calibration method 'own' needs the "
                                     "microlens focal length to be set.")

                image_distance = self._whiteImageDB[self._whiteImageDB['path'] == path]['focal_length']
                lam_est = (image_distance + self._microlensFocalLength) / image_distance
                args = dict(lam_est=lam_est)

            elif method == 'dans':
                rot_guess = 0
                args = dict(rot_guess=rot_guess, spacing_guess=self._microlensSize)

            else:
                raise ValueError(f"Unknown method {method}. Must be either "
                                 f"'own' or 'dans'.")

            # Calculate ideal grid parameters
            ideal_grid_params = self._get_ideal_grid_params(
                wi=wi, method=method, args=args)

            # Calculate aligned grid parameters and align transformation
            res = self._get_align_transform(ideal_grid_params)
            align_grid_params, align_transform = res

            # Get entry for database
            if i == 0:
                calibration_db = self._get_calibration_db_entry(
                    path, ideal_grid_params, align_grid_params, align_transform)

            else:
                calibration_db = np.append(
                    calibration_db,
                    self._get_calibration_db_entry(
                        path, ideal_grid_params, align_grid_params, align_transform))

        # Set calibration database
        self._calibrationDB = calibration_db
        logger.info("...done")

        return

    def _get_ideal_grid_params(self,
                               method: str,
                               wi: ndarray,
                               args: dict) -> grids.GridParameters:
        """Estimate a regular grid from a white image using the specified method.

        Args:
            method: Method used for estimation. One of "prop", "dans".

            wi: White image used for estimation in range [0, 1]

            args: Dictionary of arguments passed to the respective method.

        Returns:
            Ideal grid parameters estimated from the white image.

        """
        # Use proposed method [ref]
        if method.lower() == "own":
            # Extract options
            try:
                lam_est = args['lam_est']
            except KeyError:
                raise KeyError("Method 'prop' needs argument 'lam_est'.")

            if self._gridType == "hex":
                logger.info("Estimating spacing and rotation...")
                res = self._est_grid_params_prop_hex(wi=wi)
                spacing_est, spacing_std, rotation_est, rotation_std = res
                logger.info("...done")

                # Calculate missing grid spacing in x-direction
                # Here, grid implementation is such that y-spacing twice as large
                # as in paper
                spacing_est = np.asarray([np.sqrt(3) * spacing_est, spacing_est])

                logger.info("Estimating offset...")
                offset_est = self._est_grid_offset_prop(
                    grid_type="hex", wi=wi, lam_est=lam_est,
                    spacing_est=spacing_est, rotation_est=rotation_est)
                logger.info("...done")

            elif self._gridType == "rect":
                raise NotImplementedError("Rectangular grid not implemented (yet).")

        # Use method by D. Dansereau et al.
        elif method.lower() == "dans":
            # Extract options
            try:
                rot_guess = args['rot_guess']
                spacing_guess = args['spacing_guess']
            except KeyError:
                raise KeyError(
                    "Method 'dans' needs argument 'spacing_guess' and 'rotation_guess'.")

            if self._gridType == "hex":
                # Estimate spacing and rotation
                logger.info("Estimating spacing and rotation...")
                spacing_est, rotation_est, ml_centers = self._est_grid_params_dans_hex(
                    wi=wi, spacing_guess=spacing_guess, rot_guess=rot_guess)
                logger.info("...done")

                # Estimate offset
                logger.info("Estimating offset...")
                offset_est = self._est_grid_offset_dans(
                    grid_type="hex", wi=wi, spacing_est=spacing_est,
                    rotation_est=rotation_est, ml_centers=ml_centers)
                logger.info("...done")

            elif self._gridType == "rect":
                raise NotImplementedError()

        # Log Results
        logger.info(f"Spacing est.:\t{spacing_est}")
        logger.info(f"Rotation est.:\t{rotation_est}")
        logger.info(f"Offset est.:\t{offset_est}")

        # Init grid parameters
        parameters_est = grids.GridParameters(
            size=wi.shape, spacing=spacing_est, rotation=rotation_est,
            offset=offset_est)

        # Return final estimated gridparameters
        return parameters_est

    @staticmethod
    def _get_align_transform(ideal_grid_params: grids.GridParameters) \
            -> Tuple[grids.GridParameters, skt.AffineTransform]:
        """Calculate transformation that is used to align the images.
        """

        # A little bit of caution is necessary to not confuse the row/col
        # indexing used by numpy and x/y indexing used by skimage transforms

        # Get rotation transformation
        alpha = ideal_grid_params.rotation
        t_rot = skt.AffineTransform(rotation=alpha)

        # Get scaling transformation. We always perform upscaling
        spacing = ideal_grid_params.spacing
        spacing_up = np.ceil(spacing)

        # Get next highest even number for spacing
        spacing_up = spacing_up + (spacing_up % 2)

        # Get upscaling factor
        scaling = spacing_up / spacing
        scaling_xy = np.flip(scaling)
        t_scale = skt.AffineTransform(scale=scaling_xy)

        # Calculate translation
        offset = ideal_grid_params.offset
        offset_xy = np.flip(offset)
        offset_transformed_xy = np.squeeze((t_rot + t_scale)(offset_xy))
        offset_transformed = np.flip(offset_transformed_xy)

        # Shift to edge of microlens
        translation = -(offset_transformed - int(spacing_up[1] / 2))
        translation_xy = np.flip(translation)

        t_trans = skt.AffineTransform(translation=translation_xy)

        # Compose transformations, first rotation then scaling and translation
        align_transform = t_rot + t_scale + t_trans

        # Calculate aligned ideal grid parameters

        # Calculate output shape (row/col indexing)
        size = np.array(ideal_grid_params.size) * scaling
        size -= translation
        size = np.ceil(size).astype(np.uint32)

        align_grid_params = grids.GridParameters(
            size=size, spacing=spacing_up, rotation=0)

        return align_grid_params, align_transform

    def _align_image(self, img: ndarray, wi_idx: int) -> ndarray:
        """Align to sensor image with the microlens array.

        Image is rotated, scaled and translated, so that all microlens
        centers fall onto pixel centers and rotation is compensated.
        Bicubic itnerpolation is performed by default.

        Args:
            img: Input sensor image to be aligned.

            wi_idx: Index of the white image with zoom and focus parameters
                    closest to input image.

        Returns:
            The aligned output image.

        """
        logger.info("Aligning sensor image...")

        align_transform = self._calibrationDB[wi_idx]['align_transform']
        align_grid_params = self._calibrationDB[wi_idx]['align_grid_params']

        img_out = skt.warp(img, align_transform.inverse,
                           output_shape=align_grid_params.size,
                           order=3, cval=0, mode='constant')
        logger.info("...done.")

        return np.clip(img_out, 0, 1)

    def _slice_image(self, img: ndarray, wi_idx: int) -> LightField:
        """Slice image into a light field.

        Args:
            img: Input raw image.

            wi_idx: Index of the white image with zoom and focus parameters
                    closest to input image.

        Returns:
            Decoded light field.

        """

        logger.info("Slicing image to light field...")
        size_x_orig, size_y_orig, num_ch = img.shape

        if self._gridType == 'hex':

            align_grid_params = self._calibrationDB[wi_idx]['align_grid_params']

            ml_spacing = align_grid_params.spacing.astype(np.uint16)
            ml_diameter = ml_spacing[1]
            ml_radius = int(ml_diameter / 2)

            u_max = int(ml_diameter)
            v_max = int(ml_diameter)

            size_x, size_y = align_grid_params.size

            # Conservatively Estimate subaperture size (caution with hex grid)
            s_max = 2 * int(size_x / ml_spacing[0]) - 3
            t_max = int(size_y / ml_spacing[1]) - 3

            # make s_max even
            s_max = s_max - s_max % 2

            x_max = int(s_max * 0.5*ml_spacing[0])
            y_max = int(t_max * ml_spacing[1])

            # Initialize light field data
            data = np.zeros((u_max, v_max, s_max, t_max, 3))

            for u in range(0, u_max):
                for v in range(0, v_max):
                    # Start, end and step for even indices
                    ax, ay = u, v
                    ex, ey = x_max, y_max
                    sx, sy = ml_spacing

                    data[u, v, 0::2, :, :] = img[ax:ex:sx, ay:ey:sy]

                    # Start and end for odd indices
                    ax, ay = (np.array([u, v]) + 0.5 * ml_spacing).astype(np.uint32)
                    ex, ey = (np.array([x_max, y_max]) + 0.5 * ml_spacing).astype(np.uint32)
                    data[u, v, 1::2, :, :] = img[ax:ex:sx, ay:ey:sy]

        elif self._gridType == 'rect':
            raise NotImplementedError(
                "Slicing for rect grid is not yet implemented "
                "(but it should be very easy to do so).")

        logger.info("...done.")
        return LightField(data)

    def _resample_lf(self, lf: LightField, method: str):
        """Resample from a hexagonally sampled to rectangular sampled image.

        Args:
            lf: Light field to resample.

            method: Resampling method. Available:
                - 'guided': Perform gradient guided interpolation (recommended)
                - 'bilinear' : Perform bilinear interpolation.
                - 'horizontal': Only use horizontal 1D-interpolation
                - 'vertical': Only use vertical 1D-interpolation


        Returns:

        """
        logger.info("Resampling light field to rectangular grid...")

        u_max, v_max, s_max, t_max, num_ch = lf.shape
        sqrt3 = np.sqrt(3)

        # Size after upscaling to quadratic pixels, always upscale
        s_scale = 2 / sqrt3
        t_max_res = int(np.ceil(t_max*2/sqrt3))

        data_res = np.zeros((u_max, v_max, s_max, t_max_res, num_ch))

        # Create interpolation kernels for horizontal, vertical and bilinear
        # interpolation. Account for hex grid spacing...
        kernel_bilinear = np.array([[0, 1 / sqrt3,     0],
                                    [1, 2 + 2 / sqrt3, 1],
                                    [0, 1 / sqrt3,     0]])[..., np.newaxis]

        kernel_bilinear *= (1 / (2 + 2 / sqrt3))  # Normalization

        kernel_horz = 0.5 * np.array([[0, 0, 0],
                                      [1, 2, 1],
                                      [0, 0, 0]])[..., np.newaxis]

        kernel_vert = 0.5 * np.array([[0, 1, 0],
                                      [0, 2, 0],
                                      [0, 1, 0]])[..., np.newaxis]

        kernel_smooth = np.array([[0, 1, 0],
                                  [1, 3, 1],
                                  [2, 4, 2],
                                  [1, 3, 1],
                                  [0, 1, 0]])

        for u in range(u_max):
            for v in range(v_max):
                # Create rect grid and fill with hex grid values
                # Leaves checkerboard-like structure
                tmp = np.zeros((s_max, 2 * t_max, num_ch))
                tmp[0::2, 0::2] = lf[u, v, 0::2, ...]
                tmp[1::2, 1::2] = lf[u, v, 1::2, ...]

                if method == 'bilinear':
                    # Interpolate missing values bilinearly
                    tmp = ndimage.filters.convolve(tmp, kernel_bilinear)

                elif method == 'horizontal':
                    # Interpolate missing values horizontally
                    tmp = ndimage.filters.convolve(tmp, kernel_horz)

                elif method == 'vertical':
                    # Interpolate missing values vertically
                    tmp = ndimage.filters.convolve(tmp, kernel_vert)

                elif method == 'guided':
                    # Perform gradient guided interpolation

                    # Calculate gradients of image, different spacing in x, y
                    # dx = sqrt3, dy = 1
                    grad = np.gradient(tmp, sqrt3, 1, axis=[0, 1])

                    # Remove color channel by norming and smooth
                    grad[0] = ndimage.filters.convolve(
                        np.linalg.norm(grad[0], axis=-1), kernel_smooth)
                    grad[1] = ndimage.filters.convolve(
                        np.linalg.norm(grad[1], axis=-1), kernel_smooth)

                    # Stack gradients for dx, dy and sqrt(dx**2 + dy**2)
                    # Gradients are stacked along last axis, (x, y, 3)
                    grad_stack = np.stack((np.abs(grad[0]),
                                           np.abs(grad[1]),
                                           np.hypot(grad[0], grad[1])), axis=-1)

                    # Decide which gradient is smallest, shape (x, y)
                    choose = np.argmin(grad_stack, axis=-1)

                    im_horz = ndimage.filters.convolve(tmp, kernel_horz)
                    im_vert = ndimage.filters.convolve(tmp, kernel_vert)
                    im_bilinear = ndimage.filters.convolve(tmp, kernel_bilinear)

                    tmp[choose == 0] = im_vert[choose == 0]
                    tmp[choose == 1] = im_horz[choose == 1]
                    tmp[choose == 2] = im_bilinear[choose == 2]

                # Rescale to square pixels
                data_res[u, v] = skt.resize(tmp, (s_max, t_max_res))

        # Normalize
        data_res /= data_res.max()

        logger.info("...done.")

        del tmp, kernel_smooth, kernel_vert, kernel_horz, kernel_bilinear

        return LightField(data_res)

    @staticmethod
    def _est_offset(grid_type: str,
                    ml_centers: ndarray,
                    size: ndarray,
                    spacing_est: ndarray,
                    rotation_est: float,
                    offset_init: Optional[ndarray] = None,
                    window: Optional[ndarray] = None):
        """Estimate a grid offset from estimated ml_centers of a white image.

        Args:
            ml_centers:
            size:
            spacing_est:
            rotation_est:
            offset_init:
            window:

        Returns:

        """

        if window is None:
            window = np.ones(size, dtype=np.uint8)

        GridClass = grids.HexGrid if grid_type == 'hex' else grids.RectGrid

        # Bild KDTree of detected centers
        ml_tree = cKDTree(ml_centers)

        # Initialize grid with estimated spacing as offset
        if offset_init is None:
            offset_init = np.asarray([spacing_est[1], spacing_est[1]])

        par_est = grids.GridParameters(size=size, offset=offset_init,
                                       spacing=spacing_est,
                                       rotation=rotation_est)

        grid_est = GridClass(par_est)

        grid_points = grid_est.gridPoints

        # Find offset of grid to ml center points
        offset = []
        for point in grid_points:
            _d, _idx = ml_tree.query(point)

            if _d < spacing_est[1]:
                weight = window[int(ml_centers[_idx][0]),
                                int(ml_centers[_idx][1])]

                offset.append(weight * (ml_centers[_idx] - point))

        return offset_init + np.median(np.asarray(offset), axis=0)

    @staticmethod
    def _get_fourier_peaks(wi_fft: np.ndarray,
                           constrained: bool,
                           exponent: float = 2.0) -> ndarray:
        """Find peaks in absolute value of a Fourier transform of an image.

        Args:
            wi_fft: Input, FFT of an image. Should be of uneven shape, so that
                    conversion to frequency units 1/px is correct.

            constrained: Whether to constrain the number of returned peaks.

            exponent: Exponent of the center of mass calculation around each peak.

        Returns:
            Peak coordinates (p_x, p_y) as Numpy array of shape
            (N, 2) in units 1/px.

        """

        size_x_fft, size_y_fft = wi_fft.shape

        # check that size is odd
        if size_x_fft % 2 == 0 or size_y_fft % 2 == 0:
            raise ValueError(f"Shape of FFT must be uneven! "
                             f"Found {size_x_fft, size_y_fft}.")

        # Find local maxima and values of FFT
        maxima = sk_peak_local_max(wi_fft, min_distance=20)
        max_val = wi_fft[maxima[:, 0], maxima[:, 1]]

        # Get largest maxima values, constrain to largest 100 if desired
        if constrained:
            try:
                maxima = maxima[np.argsort(max_val)][-100:]
            except IndexError:
                maxima = maxima[np.argsort(max_val)]

        else:
            maxima = maxima[np.argsort(max_val)]

        # Get maxima in frequency units 1/px
        max_x = (maxima[:, 0] / (size_x_fft - 1)) - 0.5
        max_y = (maxima[:, 1] / (size_y_fft - 1)) - 0.5

        # Sort maxima by frequency distance, and use only first N components
        # excluding first peak which is the (0, 0) component
        if constrained:
            try:
                max_sort_idx = np.argsort(max_x ** 2 + max_y ** 2)[1:100]

            except IndexError:
                max_sort_idx = np.argsort(max_x ** 2 + max_y ** 2)[1:]
        else:
            max_sort_idx = np.argsort(max_x ** 2 + max_y ** 2)

        peak_coords = []

        # Calculate CoM around every maximum to get subpixel precision
        # Convert dtype to double precision avoid overflow due to large exponents
        com_pad = 7
        for idx, idy in maxima[max_sort_idx]:
            x, y = center_of_mass(
                wi_fft[idx - com_pad:idx + com_pad + 1,
                idy - com_pad:idy + com_pad + 1].astype(np.float64) ** exponent
            ) - np.asarray([com_pad, com_pad])

            peak_coords.append([x + idx, y + idy])

        peak_coords = np.asarray(peak_coords) / np.asarray(
            [size_x_fft - 1, size_y_fft - 1])

        # Shift zero freq
        peak_coords -= 0.5

        return peak_coords

    @staticmethod
    def _calc_grid_params_helper(x, im, f_mat, hires) -> Union[Tuple[float, float, float, float], float]:
        """Helper function to calculate spacing and rotation from a white image
        using different hyperparameters.

        Args:
            x: Hyperparameter vector. x = [x_q, x_gamma, x_exponent], where
                - q is pixel value lower bound used for contrast stretching
                - gamma is gamma correction factor used for value stretching
                - exponent is the exponent used in the center of mass calculation
                Note that x is normalized in [0, 1]^3

            im: Input white image.

            f_mat: Pre-calculated basis frequency vectors.

            hires: Whether to use hires FFT (zero padding) and return full grid values.

            show: Whether to show intermediate results for debugging.

        Returns:
            spacing_std [if hires == False] or
            Tuple (spacing, spacing_std, rot, rot_std) [if hires == True]

        """
        size = min(im.shape)

        if hires is True:
            pad = 1450
            window = "hann_rotational"
            n_max = 3
        else:
            pad = None
            window = "hann_rotational"
            n_max = 4

        # Map from unit interval to parameters interval
        q = 3 * 10 ** (1 * x[0] - 2)  # range [5*10E-2, 5*10E-1]
        gamma = 10 ** (2 * x[1] - 4)  # range [10E-4, 10E-2]
        sigma = (0.3 * x[2] + 0.3)  # range [0.3, 0.6]
        exponent = 2.5

        # Get rotationally symmetric Gauss kernel
        gauss = kernels.get_kernel_gauss(im.shape[0], sigma * size,
                                         im.shape[1],
                                         normalize='peak').astype(im.dtype)

        im_loc = gauss*(np.asarray(exposure.rescale_intensity(im, in_range=(q, 0.95)))**gamma).astype(im.dtype)
        del gauss

        wi_fft = np.abs(
            images.fourier(im_loc, shift=True, window=window, pad=pad,
                           implementation='numpy'))

        peak_coords = ALFC._get_fourier_peaks(
            wi_fft, constrained=False, exponent=exponent)

        # create axis for more measurements
        f = f_mat[..., np.newaxis].copy()

        # Build KDTree of peaks
        peak_tree = cKDTree(peak_coords)

        # Now find the two peaks that correspond to
        # N*f_a and N*f_b for N=1, 2, 3,...
        points_detected = []
        for i in range(1, n_max + 1):

            idx_multi_f = []
            for f_tmp in f_mat:
                dist, idx = peak_tree.query(i * f_tmp, k=1)
                if dist < 0.005:
                    idx_multi_f.append(idx)

            # have to have found two points for two basis vectors
            if len(idx_multi_f) == 2:
                tmp = (f, peak_coords[idx_multi_f, :, np.newaxis])
                f = np.concatenate(tmp, axis=-1)

                for idx in idx_multi_f:
                    points_detected.append(peak_coords[idx])

        points_detected = np.asarray(points_detected)

        # delete first row, as they contain the "old" measurement
        f = f[..., 1:]

        # calculate mean distance of Nf and (N-1)f
        d_list_1 = [np.linalg.norm(f[0, :, 0])]
        d_list_2 = [np.linalg.norm(f[1, :, 0])]

        for i in range(1, f.shape[-1]):
            d_list_1.append(np.linalg.norm(f[0, :, i] - f[0, :, i - 1]))
            d_list_2.append(np.linalg.norm(f[1, :, i] - f[1, :, i - 1]))

        d_list_tot = d_list_1 + d_list_2
        spacing_tot = np.mean(2 / (np.sqrt(3) * np.asarray(d_list_tot)))
        spacing_tot_std = np.std(2 / (np.sqrt(3) * np.asarray(d_list_tot)))

        spacing_1 = np.mean(2 / (np.sqrt(3) * np.asarray(d_list_1)))
        spacing_1_std = np.std(2 / (np.sqrt(3) * np.asarray(d_list_1)))

        spacing_2 = np.mean(2 / (np.sqrt(3) * np.asarray(d_list_2)))
        spacing_2_std = np.std(2 / (np.sqrt(3) * np.asarray(d_list_2)))

        # Choose measurement with smallest standard deviation
        spacing_std_min = np.min(
            np.asarray([spacing_1_std, spacing_2_std, spacing_tot_std]))

        if spacing_1_std == spacing_std_min:
            spacing = spacing_1
            spacing_std = spacing_1_std

        elif spacing_2_std == spacing_std_min:
            spacing = spacing_2
            spacing_std = spacing_2_std

        elif spacing_tot_std == spacing_std_min:
            spacing = spacing_tot
            spacing_std = spacing_tot_std

        else:
            raise ValueError("Something went wrong ¯\\_(ツ)_/¯.")

        # calculate rotation by line fit (swap x, y for numerical stability)
        fit1 = np.polyfit(f[1, 0, :], f[1, 1, :], deg=1, full=True)
        fit2 = np.polyfit(f[0, 0, :], f[0, 1, :], deg=1, full=True)

        alpha = np.arctan(fit1[0][0])
        try:
            alpha_std = fit1[1][0]
        except IndexError:
            alpha_std = 0

        if hires:
            logger.info(
                f"q: {q}, gamma: {gamma}, exponent: {exponent}, sigma/size: {sigma }.")
            # return spacing, spacing_std, alpha, alpha_std
            return spacing, spacing_std, alpha, alpha_std
        else:
            return float(spacing_tot_std)

    @staticmethod
    def _est_grid_params_prop_hex(wi: ndarray) -> Tuple[
        float, float, float, float]:
        """Estimate the grid parameters of a hex grid in the Fourier domain.

        This algorithm follows the presentation in [PAPER].
        The white image is fourier transformed and peaks are detected
        corresponding to the frequencies of the spatial basis vectors.
        From these, spacing and rotation are estimtaed.

        Returns:
            spacing, alpha
            Spacing vector and rotation of the grid in radians.

        """

        im = wi.copy()
        n, m = im.shape

        # crop wi to uneven shape so that there is a central pixel
        if n % 2 == 0:
            im = im[:-1, :]
        if m % 2 == 0:
            im = im[:, :-1]
        n, m = im.shape

        # crop to square shape (window will crop it anyways)
        s = min(n, m)
        im_small = im[(n - s) // 2:s + (n - s) // 2,
                   (m - s) // 2:s + (m - s) // 2]
        n, m = im_small.shape

        # First, calculate an estimate of the frequency vectors
        wi_fft = np.abs(
            images.fourier(im, shift=True, window='hann_rotational',
                           pad=1450, implementation='numpy'))

        peak_coords = ALFC._get_fourier_peaks(wi_fft, constrained=True)

        # Find main frequency grid vectors
        # Only use peaks with positive f_x component and sort by descending f_y
        f_mat = peak_coords[0:6].copy()
        f_mat = f_mat[f_mat[:, 0] > 0, :]
        sort = np.argsort(f_mat[:, 1])
        f_mat = np.flipud(f_mat[sort, :])[0:2]

        # Now, optimize free parameters using differential evolution
        # Start with small population and iterations
        # if result is not good enough, increase size
        spacing_std = 1000
        counter = 0
        max_iter = 8
        pop_size = 4

        while spacing_std > 0.002 and counter < 2:
            logger.info("Optimizing hyperparameters...")
            logger.info(f"Maxiter: {max_iter}, Popsize: {pop_size}")
            res = differential_evolution(ALFC._calc_grid_params_helper,
                                         bounds=[(0, 1), (0, 1), (0, 1)],
                                         args=(im_small, f_mat, False),
                                         maxiter=max_iter,
                                         popsize=pop_size,
                                         mutation=(0.6, 1),
                                         recombination=0.6,
                                         atol=0.0025,
                                         tol=0.01,
                                         polish=False,
                                         updating='deferred',
                                         workers=-1)
            q_est, gamma_est, sigma_est = res['x']
            logger.info("...done")

            # calculate hi res result
            logger.info("Calculate HIRES result...")
            spacing, spacing_std, alpha, alpha_std = ALFC._calc_grid_params_helper(
                np.asarray([q_est, gamma_est, sigma_est]), im,
                f_mat, hires=True)
            logger.info("...done")

            counter += 1
            max_iter += 5
            pop_size += 5
            # end while

        return spacing, spacing_std, alpha, alpha_std

    @staticmethod
    def _est_grid_offset_prop(wi: ndarray,
                              lam_est: float,
                              spacing_est: ndarray,
                              rotation_est: float,
                              grid_type: str) -> float:
        """Estimate overall grid offset.

        Args:
            wi: White image

            lam_est: Estimated magnification factor lam = (F +f) / F

            show: Whether to show results for debugging.

        Returns:
            Estimated grid offset

        """
        # Get ml centers and a window in central image region for weighting
        ml_centers, window = ALFC._get_ml_centers_prop(wi=wi, lam_est=lam_est)

        if ml_centers == []:
            raise ValueError("Could not estimate microlens centers...")

        # Get first, rough estimate without weighting the central region
        offset_init = ALFC._est_offset(grid_type=grid_type,
                                       ml_centers=ml_centers,
                                       size=wi.shape,
                                       spacing_est=spacing_est,
                                       rotation_est=rotation_est)

        # Refine offset using higher weights in central region
        offset_refined = ALFC._est_offset(grid_type=grid_type,
                                          ml_centers=ml_centers,
                                          size=wi.shape,
                                          spacing_est=spacing_est,
                                          rotation_est=rotation_est,
                                          offset_init=offset_init,
                                          window=window)

        return offset_refined

    @staticmethod
    def _est_grid_params_dans_hex(wi: ndarray, spacing_guess: float,
                                  rot_guess: float):

        # Preprocess WI
        if spacing_guess == 15 or spacing_guess == 14:
            # use Matlab disk kernel to be directly comparable to toolbox
            conv_kernel = np.array([[0., 0., 0., 0.09814366, 0.39010312,
                                     0.49165412, 0.39010312, 0.09814366, 0.,
                                     0., 0.],
                                    [0., 0.00251692, 0.48357511, 0.97356963,
                                     1., 1., 1., 0.97356963, 0.48357511,
                                     0.00251692, 0.],
                                    [0., 0.48357511, 1., 1., 1., 1., 1., 1.,
                                     1., 0.48357511, 0.],
                                    [0.09814366, 0.97356963, 1., 1., 1., 1.,
                                     1., 1., 1., 0.97356963, 0.09814366],
                                    [0.39010312, 1., 1., 1., 1., 1., 1., 1.,
                                     1., 1., 0.39010312],
                                    [0.49165412, 1., 1., 1., 1., 1., 1., 1.,
                                     1., 1., 0.49165412],
                                    [0.39010312, 1., 1., 1., 1., 1., 1., 1.,
                                     1., 1., 0.39010312],
                                    [0.09814366, 0.97356963, 1., 1., 1., 1.,
                                     1., 1., 1., 0.97356963, 0.09814366],
                                    [0., 0.48357511, 1., 1., 1., 1., 1., 1.,
                                     1., 0.48357511, 0.],
                                    [0., 0.00251692, 0.48357511, 0.97356963,
                                     1., 1., 1., 0.97356963, 0.48357511,
                                     0.00251692, 0.],
                                    [0., 0., 0., 0.09814366, 0.39010312,
                                     0.49165412, 0.39010312, 0.09814366, 0.,
                                     0., 0.]])

        else:
            conv_kernel = kernels.get_kernel(name='disk',
                                             size=int(spacing_guess / 3))

        # Convolve WI
        wi_conv = ndimage.filters.convolve(wi, conv_kernel)

        # Calculate ML centers
        ml_centers = ALFC._get_ml_centers_dans(wi=wi_conv)

        # Crop ML centers
        ml_centers = ALFC._crop_ml_centers(
            ml_centers=ml_centers, size=wi.shape, crop=int(10 * spacing_guess))

        # Calculate spacing and rotation using the ML centers
        spacing_est, rotation_est = ALFC._get_grid_params_dans_hex(
            ml_centers=ml_centers, spacing_guess=spacing_guess,
            rot_guess=rot_guess, wi=wi)

        return spacing_est, rotation_est, ml_centers

    @staticmethod
    def _get_grid_params_dans_hex(ml_centers: ndarray,
                                  spacing_guess: float,
                                  rot_guess: float,
                                  wi: ndarray,
                                  show: bool = False) -> Tuple[ndarray, float]:
        """Grid parameter estimation as implemented by Dansereau in the
        MATLAB Light Field Toolbox v0.4.

        The grid is traversed horizontally and vertically, performing line fits
        to estimate the grid rotation. For this a prior spacing and rotation
        estimate is necessary.

        Spacing is calculated as the mean distance of grid neighbors in the
        corresponding grid direction.

        Args:
            ml_centers: Estimated micro lens centers to be used for grid estimate.

            spacing_guess: A prior estimation of the grid spacing in y-direction.

            rot_guess: A prior estimation of the grid rotation in radians.

            show: Boolean flag indicating whether to plot results for debugging.

            white_img: White image. Only used for plotting.

        Returns:
            Estimated spacing (2D array) and rotation (in radians) of the grid.
            If not estimate is possible, returns nans.

        """

        # If no centers have been found, return nans
        if not np.any(ml_centers):
            return np.asarray([np.nan, np.nan]), np.nan

        crop = int(10 * spacing_guess)

        # Build a KDTree for nearest neighbor search
        ml_tree = cKDTree(ml_centers)

        # Build estimated grid vectors (for hex grid)
        # First, for zero rotaion
        a = np.asarray([0, spacing_guess])
        b = np.asarray([spacing_guess * 0.5 * np.sqrt(3), 0.5 * spacing_guess])

        # Rotate by estimated rotation
        if not rot_guess == 0:
            rot_mat = misc.rot_matrix(0, unit='radians')
            a = rot_mat @ a
            b = rot_mat @ b

        # Get conservative estimate of number of MLs per dimension
        num_x = int(wi.shape[0] / b[0]) + 100
        num_y = int(wi.shape[1] / a[1]) + 100

        # Find first lenslet as starting point
        _d, _idx = ml_tree.query(np.asarray([crop, crop]))
        ml_start = ml_centers[_idx]

        # Move one column in, to be sure of grid layout
        _d, _idx = ml_tree.query(ml_start + b)
        ml_start = ml_centers[_idx]

        # Initialize measurements
        alpha = []
        spacing_x = []
        spacing_y = []

        # Distance threshold for nearest neighbor search
        max_dist = 0.9 * spacing_guess // 2

        # #####################
        # Traverse horizontally
        # #####################

        # Find ML basis for vertical lines
        ml_curr = ml_start
        ml_basis = [ml_start]
        i = 0

        while True:
            # If on top row, move down using vector b
            if i % 2 == 0:
                _d, _idx = ml_tree.query(ml_curr + b)

            # Else, move up using a - b
            else:
                _d, _idx = ml_tree.query(ml_curr + a - b)

            # If below threshold, add to basis, otherwise we have left the grid
            if _d < max_dist:
                ml_curr = ml_centers[_idx]
                ml_basis.append(ml_curr)
                i += 1

            else:
                ml_basis = np.asarray(ml_basis)
                break

        # Iterate through ml_basis and perform vertical search
        for ml in ml_basis:
            i = 0
            ml_curr = ml
            ml_vert = [ml_curr]

            while True:
                # To get to next ML move to down/right, one left
                _d, _idx = ml_tree.query(ml_curr + 2 * b - a)

                # If below threshold, add to basis, otherwise we have left the grid
                if _d < max_dist:
                    ml_curr = ml_centers[_idx]
                    ml_vert.append(ml_curr)
                    i += 1

                else:
                    ml_vert = np.asarray(ml_vert)
                    break

            # Perform vertical line fit
            slope, off = np.polyfit(ml_vert[:, 0], ml_vert[:, 1], deg=1)

            alpha.append(np.arctan(slope))
            spacing_x.append(np.nanmean(np.diff(ml_vert[:, 0])))

            # Show result for debug
            if show:
                fig, ax = plt.subplots()
                im = ax.imshow(wi)
                ax.plot(ml_centers[:, 1], ml_centers[:, 0], 'o', color='black')
                ax.plot(ml_basis[:, 1], ml_basis[:, 0], 'o', color='red')
                ax.plot(ml_vert[:, 1], ml_vert[:, 0], 'o', color='blue')
                plt.show()

        # #####################
        # Traverse vertically
        # #####################

        # Find ML basis for vertical lines
        ml_curr = ml_start
        ml_basis = [ml_start]
        i = 0

        while True:
            # If in first colulmn, move right/down using b
            if i % 2 == 0:
                _d, _idx = ml_tree.query(ml_curr + b)

            # Else, move left using b - a
            else:
                _d, _idx = ml_tree.query(ml_curr - a + b)

            # If below threshold, add to basis, otherwise we have left the grid
            if _d < max_dist:
                ml_curr = ml_centers[_idx]
                ml_basis.append(ml_curr)
                i += 1

            else:
                ml_basis = np.asarray(ml_basis)
                break

        # Iterate through ml_basis and perform vertical search
        for ml in ml_basis:
            i = 0
            ml_curr = ml
            ml_horz = [ml_curr]

            while True:
                # To get to next ML move to right
                _d, _idx = ml_tree.query(ml_curr + a)

                # If below threshold, add to basis, otherwise we have left the grid
                if _d < max_dist:
                    ml_curr = ml_centers[_idx]
                    ml_horz.append(ml_curr)
                    i += 1

                else:
                    ml_horz = np.asarray(ml_horz)
                    break

            # Perform vertical line fit (caution: coordinates are swapped for fit!)
            slope, off = np.polyfit(ml_horz[:, 1], ml_horz[:, 0], deg=1)

            alpha.append(- np.arctan(slope))
            spacing_y.append(np.nanmean(np.diff(ml_horz[:, 1])))

            # Show result for debug
            if show:
                fig, ax = plt.subplots()
                im = ax.imshow(wi)
                ax.plot(ml_centers[:, 1], ml_centers[:, 0], 'o', color='black')
                ax.plot(ml_basis[:, 1], ml_basis[:, 0], 'o', color='red')
                ax.plot(ml_horz[:, 1], ml_horz[:, 0], 'o', color='blue')
                plt.show()

        # Calculate final values from measurements
        alpha = np.nanmean(np.asarray(alpha))
        spacing_x = np.nanmean(spacing_x)
        spacing_y = np.nanmean(spacing_y)
        spacing = np.asarray([spacing_x, spacing_y])

        return spacing, float(alpha)

    @staticmethod
    def _est_grid_offset_dans(wi: ndarray,
                              ml_centers: ndarray,
                              spacing_est: ndarray,
                              rotation_est: float,
                              grid_type: str,
                              show: bool = False) -> float:
        """Estimate overall grid offset.

        Args:
            wi: White image

            show: Whether to show results for debugging.

        Returns:
            Estimated grid offset

        """

        # Get first, rough estimate without weighting the central region
        offset = ALFC._est_offset(
            grid_type=grid_type, ml_centers=ml_centers, size=wi.shape,
            spacing_est=spacing_est, rotation_est=rotation_est)

        return offset

    @staticmethod
    def _get_ml_centers_prop(wi: ndarray, lam_est: float) -> Tuple[ndarray, ndarray]:
        """Estimate ml centers in only central part of the white image

        Args:
            wi: White image.
            lam_est: Estimated magnification factor lambda.

        Returns:
            Microlens center coordinates as ndarray of shape (N, 2).

        """
        x, y = wi.shape

        # Estimate region where deviation from perspectively to orthogonally
        # projected centers is less than 0.5px
        s_max = int(np.floor(0.5 / (lam_est - 1)))
        width = 2 * s_max - 1  # make odd width so center exists

        # Create Gauss window that matches the central window
        window = kernels.get_kernel_gauss(wi.shape[0], width / 5,
                                          wi.shape[1], width / 5,
                                          normalize='peak')

        # Create mask to crop image center
        mask = np.ones(wi.shape, dtype=bool)

        # Unmask center of the image
        start_x = (x - width) // 2
        start_y = (y - width) // 2
        mask[start_x:start_x + width, start_y:start_y + width] = 0

        wi_center = wi.copy()
        wi_center[mask] = 0.0

        # Convolve white image with gauss
        conv_kernel = kernels.get_kernel(name='gauss', size=5)

        conv_img = nd_filters.convolve(wi_center, conv_kernel)
        del wi_center

        # Contrast stretch
        conv_img -= conv_img.min()
        conv_img /= conv_img.max()

        # Gamma stretch
        conv_img = conv_img ** 5

        # Adaptive local thresholding to get local maximum regions
        adaptive_thresh = sk_filters.threshold_local(conv_img, 17, method='gaussian')
        binary = conv_img**1.5 > adaptive_thresh

        # Cut off all values outside the binary mask
        conv_img[~binary] = 0.0

        ml_clusters = np.ma.masked_array(conv_img, conv_img < 0.1)

        # Find clusters, which correspond the the microlenses, enumerate them
        cluster, num_of_clusters = ndimage.label(conv_img)
        idx = np.arange(1, num_of_clusters + 1)

        # Iterate over clusters, if too small or too large, delete
        # Get localization of each cluster
        loc = find_objects(cluster)

        # iterate through clusters label
        idx_new = []
        for i in idx - 1:

            # Get image values of cluster
            loc_tmp = loc[i]
            I_meas = cluster[loc_tmp]
            tmp = (I_meas == i + 1)
            N = tmp.nonzero()[0].size
            e = tmp.shape[0] / tmp.shape[1]

            # If cluster is too small or too large or too eccentric, ignore
            if 0.5 <= e <= 1.5 and 3 * 3 <= N <= 13 * 13:
                idx_new.append(i + 1)

        idx = idx_new
        # Calculate the subpixel microlens_coordinates by calculating the
        # center of mass of each detected cluster.
        ml_centers = np.asarray(
            ndimage.measurements.center_of_mass(
                ml_clusters**2,
                labels=cluster,
                index=idx))

        return ml_centers, window

    @staticmethod
    def _get_ml_centers_dans(wi: ndarray) -> ndarray:
        """Calculat ML centers from WI,
        method proposed by Dansereau et al. in Matlab LightField Toolbox.

        Args:
            wi: White image used for estimate.

        Returns:
            Estimated ml center coordinates in shape (N, 2).

        """

        # Find local maxima of image
        data_max = sc_filters.maximum_filter(wi, 10)

        return np.asarray(np.where(wi == data_max)).T

    @staticmethod
    def _crop_ml_centers(ml_centers: ndarray, size: ndarray, crop: int):
        """Crop ml center coordinates by crop amount to get rid of
        edge defects"""

        cropped_centers = ml_centers.copy()

        # Crop the ML centers
        indices = []
        x_max, y_max = size

        for k in range(0, ml_centers.shape[0]):
            if ml_centers[k][0] < crop or \
                    ml_centers[k][1] < crop or \
                    ml_centers[k][0] > (x_max - 1 - crop) or \
                    ml_centers[k][1] > (y_max - 1 - crop):
                indices.append(k)

        # delete elements that are too small
        cropped_centers = np.delete(cropped_centers, indices, axis=0)

        return cropped_centers

    def get_ml_centers(self, wi_idx: int):

        GridClass = grids.HexGrid if self._gridType == 'hex' else grids.RectGrid
        ideal_grid = GridClass(self._calibrationDB[wi_idx]['ideal_grid_params'])

        return ideal_grid.gridPoints

    def show_microlens_centers(self,
                               wi_idx: int,
                               grid: bool = False,
                               plt_show: bool = True):
        """Show the calculated microlens centers on top of the whiteimage.

        Args:
            wi_idx: Index of the corresponding white image.

            grid: Whether to also show the estimated ideal grid on top

            plt_show: Whether :func:`.matplotlib.pyplot.show()`
                      is called at the end of the function

        """
        if not self._isCalibrated:
            raise RuntimeError("Please calibrate the Camera instance first.")

        ml_centers = self.get_ml_centers(wi_idx)
        wi = self._get_wi(self._whiteImageDB[wi_idx]['path'])

        # Plot points and grid
        fig, ax = plt.subplots()

        ax.plot(ml_centers[:, 1],
                ml_centers[:, 0],
                'o', color='blue', zorder=999)

        ax.set_aspect('equal', adjustable='box')

        # Plot whiteimage
        ax.imshow(wi, cmap='gray', alpha=1.0)

        if grid:

            # Calculate Delaunay triangulation of grid points
            tri = spatial.Delaunay(ml_centers)
            ax.plot(ml_centers[:, 1],
                    ml_centers[:, 0],
                    'o', color='red')
            ax.triplot(ml_centers[:, 1],
                       ml_centers[:, 0],
                       tri.simplices.copy())

        if plt_show is True:
            plt.show()

        return

    def show_decoded_image(self, num: int):
        """Show the decoded light field using
        :func:`.plenpy.lightfields.lightfield.LightField.show_interactive()` .

        Args:
            num: Number of the sensor image that the image is decoded from.

        """
        lightfield = self.get_decoded_image(num)
        lightfield.show_interactive()

        return


# Introduce alias
ALFC = AbstractLightFieldCamera
