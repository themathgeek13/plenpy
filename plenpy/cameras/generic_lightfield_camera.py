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
Module defining the :class:`GenericLightFieldCamera` class.

This camera class, derived from the :class:`AbstractLightFieldCamera` base
class, implements a generic light field camera, i.e. a light field camera
with very few generic information available. For example, a grid layout
does not need to be specified as the algorithms work with both hexagonal,
rectangular or any other grid.

"""

import gc
import joblib
import numpy as np
from matplotlib import tri as mtri
from scipy import interpolate
from skimage import img_as_float

import plenpy.logg
from plenpy import lightfields as lf
from plenpy.utilities import demosaic
from plenpy.utilities import images

logger = plenpy.logg.get_logger()

__all__ = ['GenericLightFieldCamera']


# TODO: Needs a redo. A lot has changed, yet this implementation has not...
# Relies on old methods of AbstractLightFieldCamera...
raise NotImplementedError


class GenericLightFieldCamera(AbstractLightFieldCamera):
    """Generic light field camera.

    This camera class can be used to calibrate and decode generic light
    field cameras. That is, microlens based cameras with either hexagonal
    or rectangular microlens array. The calibration is solely done via
    whiteimages, which have to be stored in the ``Calibration`` folder.
    The camera calculates decoded images as a
    :class:`.plenpy.lightfields.lightfield.LightField` object.

    The class does not add any attributes to the
    :class:`AbstractLightFieldCamera` base class.

    """

    def __init__(self, path: Any, microlens_size: float):
        """:class:`GenericLightFieldCamera` class initialization.

        See Also:
            The camera uses the :class:`AbstractLightFieldCamera` base class
            initialization.

        """
        super().__init__(path=path,
                         microlens_size=microlens_size)

        return

    def calibrate(self,
                  demosaic_method: Optional[str] = None,
                  pattern: Optional[str] = None,
                  force: bool = False):
        """Calibration of the LightFieldCamera.

        The basic calibration pipeline is:\n
        * Calculate an ideal white image
        * Calculate the microlens centers

        Args:
            demosaic_method : Method used to calculate the demosaiced image.
                If ``None`` is specified, no demosaicing is performed. For
                available methods
                see :func:`plenpy.utilities.demosaic.get_demosaiced()`.

            pattern:  Bayer filter pattern that the input image is modulated
                with. Only used if image is being demosaiced.

            force: If set to ``True``, force the recalibration,
                even if a calibration file is found.
                Overwrites the calibration file.

        """
        logger.info("Calibrating camera...")
        # If a calibration file is found, load it
        if (self.path / self._calDataFilename).is_file() and not force:
            logger.info(
                "Found calibration data in '{}'. "
                "Reading from calibration file.".format(
                    self.calibrationFolderPath))
            self._load_cal_data()

        else:
            self._calc_ideal_whiteimage(demosaic_method=demosaic_method,
                                        pattern=pattern)
            self._calc_microlens_centers()
            self._isCalibrated = True

            # If no calibration file is there, save it
            if not (self.path / self._calDataFilename).is_file() or force:
                logger.info("Saving compressed calibration data...")
                self._save_cal_data()

        self._isCalibrated = True
        logger.info("... done.")
        gc.collect()
        return

    def decode_sensor_image(self,
                            num: int,
                            demosaic_method: Optional[str] = None,
                            pattern: Optional[str] = None,
                            multithreading: int = 0):
        """Decode the specified sensor image.

        The decoding gives a
        :class:`.plenpy.lightfields.lightfield.LightField` object that is
        added the objects dictionary of decoded images.

        Args:
            num: Number of the sensor image that is to be decoded.

            demosaic_method : Method used to calculate the demosaiced image.
                If ``None`` is specified, no demosaicing is performed. For
                available methods and default value,
                see :func:`plenpy.utilities.demosaic.get_demosaiced()`.

            pattern:  Bayer filter pattern that the input image is modulated
                with. Only used if image is being demosaiced.

            multithreading: Specify number of parallel threads.
                If ``0`` is specified, no multithreading is applied. Caution:
                The multithreading implementation is very heavy on RAM.
                Default: 0

        """
        logger.info("Decoding sensor image number {}...".format(num))

        # Check multithreading option:
        if not isinstance(multithreading, int) or multithreading < 0:
            raise ValueError(
                f"Multithreading option {multithreading} is not an integer. "
                "Please specify a positive integer value.")

        # Get sensor image in shape (x, y, N)
        raw_img = self.get_sensor_image(num)

        if demosaic_method is not None:
            # Demosaic sensor image
            logger.info(
                f"Demosaicing sensor image using {pattern} pattern...")
            img = demosaic.get_demosaiced(np.squeeze(raw_img),
                                          pattern=pattern,
                                          method=demosaic_method)

            logger.info("...done.")

        else:
            img = raw_img

        # Convert to float
        img = img_as_float(img, raw_img.dtype)
        del raw_img

        # Calculate interpolation object
        interp = self._calc_interp_image(img)

        # Initialize light field data
        lf_s = int(img.shape[0] / self._microlensSize)
        lf_t = int(img.shape[1] / self._microlensSize)
        lf_u = self._microlensSize
        lf_v = self._microlensSize
        num_channels = interp.size

        # Garbage collection to free up memory
        gc.collect()
        del img

        lf_data = np.zeros((lf_u, lf_v, lf_s, lf_t, num_channels)).astype(
            np.float64)

        if multithreading == 0:
            # Calculate light field serially, no multithreading.
            for u in range(-self._microlensRadius, self._microlensRadius + 1):
                for v in range(-self._microlensRadius,
                               self._microlensRadius + 1):
                    logger.info(
                        "Calculating subaperture "
                        "#({},{}) of ({},{})...".format(
                            u + self._microlensRadius,
                            v + self._microlensRadius,
                            self._microlensSize,
                            self._microlensSize))

                    img = self._calc_subaperture(
                        interp=interp, x=u, y=v, method="linear")

                    lf_data[u + self._microlensRadius,
                            v + self._microlensRadius,
                            :, :, :] = img

                    # Garbage collection to free up memory
                    gc.collect()

                    logger.info("...done #({},{}).".format(
                        u + self._microlensRadius,
                        v + self._microlensRadius
                    ))

                    del img

        else:
            # Calculate subaperture clusters in parallel
            img_clusters = joblib.Parallel(
                n_jobs=multithreading)\
                (joblib.delayed(self._decode_cluster_parallel)(interp, x)
                 for x in range(-self._microlensRadius,
                                self._microlensRadius + 1))

            # copy clusters to LF data
            for i in range(0, len(img_clusters[0])):
                lf_data[i, :, :, :, :] = img_clusters[i]

            # delete clusters
            del img_clusters

        # Create LightField object from data
        decoded = lf.LightField(lf_data)
        del lf_data

        # Add decoded image to _decodedImages dictionary
        self._add_decoded_image(decoded, num)
        logger.info("...done.")
        return

    @staticmethod
    def _calc_interp_image(img: ndarray) -> ndarray:
        """Calculate an interpolation of the raw sensor image using the
        microlens center's Delaunay triangulation.

        Args:
            img: Input image to interpolate

        Returns:
            Array of :class:`scipy.interpolate.RectBivariateSpline`
            Inteprolation object of the scipy package.

        """
        logger.info("Calculating interpolation object...")

        raw_img = img.copy()
        num_x, num_y, num_ch = raw_img.shape

        data_points_x = range(0, num_x)
        data_points_y = range(0, num_y)

        # Create empty array of interpolated images
        interp_image = []
        for i in range(0, num_ch):
            interp_image.append(None)
        interp_image = np.array(interp_image)

        # Interpolate all color channels independently
        for i in range(0, num_ch):
            data = raw_img[:, :, i]

            # Save interpolation as array
            interp_image[i] = interpolate.RectBivariateSpline(
                data_points_x,
                data_points_y,
                data)
            del data

        logger.info("...done.")
        del raw_img
        return interp_image

    def _calc_subaperture(self,
                          interp: ndarray,
                          x: int,
                          y: int,
                          method: str = 'cubic') -> ndarray:
        """Calculate a subaperture view from the raw sensor data.

        Args:
            interp : ndarray of :class:`scipy.interpolate.RectBivariateSpline`

            x: Distance from the microlens center in x-direction in pixels.

            y: Distance from the microlens center in y-direction in pixels.

            method: Used interpolation method. Available methods are:
                'bilinear', 'cubic'. Default: 'cubic'.

        Returns:
            The calculated subaperture image.
        """

        interp_param_list = ["cubic", "linear"]

        num_channels = interp.size

        if method not in interp_param_list:
            raise ValueError(
                f"Specified method '{method}' is not one of the "
                "recognized methods: {interp_param_list}")

        logger.debug(
            "Calculating subaperture with x = {} and y = {}...".format(x, y))

        # Calculate interpolation coordinates by given distance x and y
        interp_coordinates = np.asarray(self._microlensCenters)
        interp_coordinates_x = interp_coordinates[:, 0] + x
        interp_coordinates_y = interp_coordinates[:, 1] + y

        # Do a Delaunay triangulation of the interpolation points
        logger.debug("Calculating Delaunay triangulation of "
                     "subaperture interpolation points....")
        triang = mtri.Triangulation(interp_coordinates_x, interp_coordinates_y)
        logger.debug("...done.")

        # Get size of the original image from interp object
        x_max = int(interp[0].tck[0].max() + 1)
        y_max = int(interp[0].tck[1].max() + 1)

        # Size of subaperture image
        s_max = int(x_max/self._microlensSize)
        t_max = int(y_max/self._microlensSize)

        # Rectangular grid for final image
        xi, yi = np.meshgrid(
            np.linspace(0, x_max, s_max),
            np.linspace(0, y_max, t_max))

        # Initialize final image
        subaperture_image = np.zeros((s_max, t_max, num_channels),
                                     dtype=np.float64)

        # Check if (x,y) coordinates are still inside microlens,
        # if not, return black image
        # if x**2 + y**2 > self._microlensRadius**2:
        #     return subaperture_image

        # Iterate over image channels and fill final image
        for i in range(0, num_channels):
            logger.debug("Interpolating subaperture image channel {}.".format(
                i))

            # Do a cubic interpolation on the triangular grid
            if method == "cubic":
                interp_geom = mtri.CubicTriInterpolator(
                    triang,
                    interp[i].ev(interp_coordinates_x,
                                 interp_coordinates_y),
                    kind='geom')

            # Alternatively, do a linear interpolation
            elif method == "linear":
                interp_geom = mtri.LinearTriInterpolator(
                    triang,
                    interp[i].ev(interp_coordinates_x,
                                 interp_coordinates_y))

            subaperture_image[:, :, i] = np.flipud(np.rot90(
                interp_geom(xi, yi)))

            logger.debug("...done.")
        del interp_geom
        logger.debug("...done.")
        return np.squeeze(np.nan_to_num(subaperture_image))

    def _decode_cluster_parallel(self,
                                 interp: ndarray,
                                 x: int) -> ndarray:
        """Decode a cluster of subaperture views.

        Args:
            interp: Array of RectBivariateSpline interpolation objects
                to calculate the subaperture view.

            x: Coordinate of the cluster in pixels. Here, x specifies the
                distance from the microlens centers in x-direction.

        Returns:
            The subaperture views of the cluster.

        """
        gc.collect()

        # Get size of the original image from interp object
        x_max = int(interp[0].tck[0].max() + 1)
        y_max = int(interp[0].tck[1].max() + 1)

        u_max = self._microlensSize
        v_max = self._microlensSize

        s_max = int(x_max / self._microlensSize)
        t_max = int(y_max / self._microlensSize)

        num_channels = interp.size

        logger.info("Calculating subaperture cluster "
                    f"#{x + self._microlensRadius + 1} of {u_max}...")

        # calculate one raw (u=const) of subaperture views
        res = np.zeros((v_max, s_max, t_max, num_channels))
        for y in range(-self._microlensRadius, self._microlensRadius + 1):
            img = self._calc_subaperture(
                interp=interp, x=x, y=y, method="linear")

            res[y + self._microlensRadius,
                :, :
                :] = images.get_standard_shape(img)
            gc.collect()
            del img

        logger.info(f"...done #{x + self._microlensRadius + 1} of {u_max}.")

        del interp
        return res

    def _get_wi_db_entry(self, path: str, metadata: Dict) -> Optional[ndarray]:
        """Get a entry for the white image database.

        For the generic camera, no metadata is available.

        Returns:
            None

        """

        return None
