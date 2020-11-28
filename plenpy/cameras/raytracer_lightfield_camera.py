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
Module defining the :class:`RayTracerLF` camera class.

This camera class, derived from the :class:`AbstractLightFieldCamera` base
class, implements a MLA based light field camera synthesized by the IIIT-RayTracer.

"""
import json
from pathlib import Path
from typing import Any, Optional

import gc
import imageio
import jsmin
import numpy as np
import skimage.color
from numpy.core import ndarray

import plenpy.logg
from plenpy.cameras.abstract_lightfield_camera import AbstractLightFieldCamera
from plenpy.lightfields import LightField

logger = plenpy.logg.get_logger()

__all__ = ['RayTracerLF']


class RayTracerLF(AbstractLightFieldCamera):
    """Raytracer implementation of MLA based LF camera.

    The class does not add any attributes to the
    :class:`AbstractLightFieldCamera` base class.

    """

    def __init__(self,
                 path: Any,
                 grid_type: str = 'hex',
                 ml_size: float = 14,
                 ml_focal_length: float = 40e-6,
                 format='PNG-FI'):

        # Call init from AbstractLightfieldCamera base class
        super().__init__(path,
                         microlens_size=ml_size,
                         grid_type=grid_type,
                         ml_focal_length=ml_focal_length)

        self._format = format

    def calibrate(self,
                  filename: Optional[Any] = None,
                  method: str = 'own',
                  force: bool = False):
        """Calibrate the RayTracer camera.

        See Also: :func:LytroIllum.calibrate()

        Args:
            filename: System path to calibration filename withoout extension.
                      If ``None``, the standard path is used.

            method: Method used for grid estimation.
                        - 'own': Own method, proposed in [ref]
                        - 'dans': Method by Dansereau et al.

            force: If ``True`` forces the recalibration, even if a calibration
                   file is found. This can be useful when recalibrating with
                   different parameters.
        """

        logger.info("Calibrating RayTracer camera...")

        if filename is not None:
            filename = self.calibrationFolderPath / filename
            if not filename.suffix == ".npz":
                filename = filename.with_suffix(".npz")
            self._calDataFilename = filename

        # If a calibration file is found, load it
        if (self.path / self._calDataFilename).is_file() and not force:
            logger.info(
                "Found calibration data in '{}'. "
                "Reading from calibration file.".format(
                    self.calibrationFolderPath))
            self._load_cal_data()

        else:
            # Read white images
            self._create_wi_db()

            # Calculate grid parameters and align transformation
            self._est_grid(method=method)

            # Save calibration data for next use
            self._save_cal_data()

        self._isCalibrated = True
        logger.info("... done.")
        return

    def decode_sensor_image(self,
                            num: int,
                            resample_method: str = 'guided'):
        """Decode the specified sensor image.

        The decoding yields a
        :class:`.plenpy.lightfields.lightfield.LightField` object that is
        added the objects dictionary of decoded images.

        Args:
            num: Number of the sensor image that is to be decoded.

            resample_method: Method used to resample the light field from a
                hex to rect grid. Available:
                - 'guided': Perform gradient guided interpolation (recommended)
                - 'bilinear' : Perform bilinear interpolation.
                - 'horizontal': Only use horizontal 1D-interpolation
                - 'vertical': Only use vertical 1D-interpolation

        """
        # Check if camera is calibrated. If not, calibrate it.
        if not self._isCalibrated:
            self.calibrate()

        # Get raw sensor image
        img = np.squeeze(self.get_sensor_image(num))

        # Load metadata (not included in imageio reader here...)
        metadata = self._get_metadata(self._sensorImages[num].imagePath)

        # Convert to float
        img = skimage.img_as_float32(img)

        # Get the white image corresponding to current zoom and focus setting
        focus_length = metadata['camera']['imageDistance']
        focus_distance = metadata['camera']['focusDistance']

        # First, get all WI with same/similar zoomstep
        diff_ = np.abs(self._whiteImageDB['focal_length'] - (focus_length))
        idx_ = np.argwhere(diff_ == diff_.min())

        # Now, find those with similar focus step
        if len(idx_) > 1:
            wi_select = self._whiteImageDB[idx_]
            diff_ = np.abs(wi_select['focus_distance'] - (focus_distance))
            idx_select_ = np.argmin(diff_)
            idx_ = np.argwhere(self._whiteImageDB == wi_select[idx_select_])

        wi_select_idx = np.squeeze(idx_)

        # Load corresponding white image
        wi_select = self._get_wi(self._whiteImageDB[wi_select_idx]['path'],
                                 process=False)

        # Devignetting, divide by selected white image
        img = np.clip(img / wi_select, 0, 1)

        # Might encounter division by zero...
        img[np.isnan(img)] = 0

        del wi_select
        gc.collect()

        # Aligning the sensor image
        img = self._align_image(img, wi_idx=wi_select_idx)

        # Slice image to light field
        lf = self._slice_image(img, wi_idx=wi_select_idx)

        del img
        gc.collect()

        # Only use central subapertures
        u, v, s, t, ch = lf.shape
        step = self._microlensRadius - 1
        u_mid = int(u / 2)
        v_mid = int(v / 2)
        data = lf[u_mid - step:u_mid + step + 1, v_mid - step:v_mid + step + 1]
        lf = LightField(data)

        if self._gridType == 'hex':
            # Resampling the light field to rect grid:
            lf = self._resample_lf(lf, method=resample_method)

        self._add_decoded_image(img=lf, num=num)

        return

    def _get_wi_db_entry(self, path: Path) -> ndarray:
        """Get the database entry for a white image.
        Entries include the focal_length and focus_distance setting of
        the white image to be used for decoding.

        Args:
            path: Path to the white image, relative to Camera main folder.
                  Used as database identfication

        Returns:
            A structured array containing the path, focal_length and focus_distance.

        """
        # Check that image exists
        img = self._get_wi(path, process=False)

        # Load metadata from JSON file
        metadata = self._get_metadata(path)

        focal_length = metadata["camera"]["imageDistance"]
        focus_distance = metadata["camera"]["focusDistance"]

        # Create structured array for white image table
        db_entry = np.array(
            [(path, focal_length, focus_distance)],
            dtype=[('path', Path), ('focal_length', 'f8'), ('focus_distance', 'f8')])

        return db_entry

    def _get_wi(self, path: Path, process: bool = True):
        """Read the white image from path and perform preprocessing such as
        contrast stretching, normalization, etc.

        Args:
            path: Path to the white image, relative to Calibration folder.

            process: Whether to perform preprocessing such as contras stretching.

        Returns:
            White image as float in range [0, 1].

        """
        wi = imageio.imread(self.path / path, format=self._format)

        # Convert to float, ranged 0, 1
        wi = skimage.img_as_float32(wi)

        if process:
            # Convert to greyscale
            wi = skimage.color.rgb2gray(wi)

            # Contrast stretch
            wi -= wi.min()
            try:
                wi /= wi[2500:-2500, 3500:-3500].max()
            except ValueError or KeyError:
                wi /= wi.max()
            wi = np.clip(wi, 0, 1)

        return wi

    def _get_metadata(self, path: Path):
        """Get the metadata from a raytracer image using its JSON file.

        Args:
            path: Path to the white image, relative to Calibration folder.

        """

        meta_path = self.path / path.with_suffix(".json")

        # Open JSON file
        # Remove possible comments which cause JSON parsing to fail
        with open(meta_path) as js_file:
            minified = jsmin.jsmin(js_file.read())

        return json.loads(minified)
