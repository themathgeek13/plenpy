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
Module defining the :class:`LytroIllum` camera class.

This camera class, derived from the :class:`AbstractLightFieldCamera` base
class, implements the Lytro Illum camera.

"""
from pathlib import Path
from typing import Optional, Any

import gc
import imageio
import numpy as np
from numpy.core import ndarray

import plenpy.logg
from plenpy.cameras.abstract_lightfield_camera import AbstractLightFieldCamera
from plenpy.lightfields import LightField
from plenpy.utilities import demosaic

logger = plenpy.logg.get_logger()

__all__ = ['LytroIllum']

# Use Lytro Illum lfr amd raw format from imageio plugins
imageio.formats.sort('lytro-illum-lfr')
imageio.formats.sort('lytro-illum-raw')


class LytroIllum(AbstractLightFieldCamera):
    """Lytro Illum light field camera.

    The class does not add any attributes to the
    :class:`AbstractLightFieldCamera` base class.

    """

    def __init__(self, path: Any, format='LYTRO-ILLUM-RAW'):
        """
        Args:
            path: Folder path of camera.
            format: The ``imageio`` format of the white images.
                    Default: LYTRO-ILLUM-RAW.
        """
        # Call init from AbstractLightfieldCamera base class
        super().__init__(path,
                         microlens_size=14,
                         grid_type='hex',
                         ml_focal_length=40e-6)
        self._format = format

    def calibrate(self,
                  filename: Optional[str] = None,
                  method: str = 'own',
                  force: bool = False):
        """Calibrate the Lytro Illum camera.

        The calibration estimates the microlens centers from the provided
        white images. Based on the chosen method, this will yield ml centers
        with (``det_method='own'``) or without (``det_method='dans'``)
        subpixel precision. Using the estimated ML centers, an ideal grid is
        estimated that best approximates the ML centers.
        This grid is used in the decoding of every light field
        (depending on the zoom and focus settings).

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

        logger.info("Calibrating Lytro Illum camera...")

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
                            demosaic_method: str = 'malvar2004',
                            resample_method: str = 'guided'):
        """Decode the specified sensor image.

        The decoding yields a
        :class:`.plenpy.lightfields.lightfield.LightField` object that is
        added the objects dictionary of decoded images.

        Args:
            num: Number of the sensor image that is to be decoded.

            demosaic_method : Method used to calculate the demosaiced image.
                If ``None`` is specified, no demosaicing is performed. For
                available methods and default value,
                see :func:`plenpy.utilities.demosaic.get_demosaiced()`.

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

        # Get raw sensor image and metadata corresponding to number num
        raw_img = np.squeeze(self.get_sensor_image(num))
        metadata = self.get_image_metadata(num)

        # Get the white image corresponding to current zoom and focus setting
        zoomstep = metadata['metadata']['devices']['lens']['zoomStep']
        focusstep = metadata['metadata']['devices']['lens']['focusStep']

        # First, get all WI with same/similar zoomstep
        diff_ = np.abs(self._whiteImageDB['zoomStep'] - (zoomstep))
        idx_ = np.argwhere(diff_ == diff_.min())

        # Now, find those with similar focus step
        if len(idx_) > 1:
            wi_select = self._whiteImageDB[idx_]
            diff_ = np.abs(wi_select['focusStep'] - (focusstep))
            idx_select_ = np.argmin(diff_)
            idx_ = np.argwhere(self._whiteImageDB == wi_select[idx_select_])

        wi_select_idx = np.squeeze(idx_)

        # Load corresponding white image
        wi_select = self._get_wi(self._whiteImageDB[wi_select_idx]['path'],
                                 process=False)

        # Get white and black level from metadata
        white_dict = metadata['metadata']['image']['pixelFormat']['white']
        black_dict = metadata['metadata']['image']['pixelFormat']['black']

        # Convert levels to float from 10 bit uint
        white = np.mean([white_dict[key] for key in white_dict]) / (2**10 - 1)
        black = np.mean([black_dict[key] for key in black_dict]) / (2**10 - 1)

        # Remove black and white levels
        raw_img = (raw_img - black) / (white - black)
        wi_select = (wi_select - black) / (white - black)

        # Devignetting, divide by selected white image
        raw_img = np.clip(raw_img / wi_select, 0, 1)

        # Might encounter division by zero...
        raw_img[np.isnan(raw_img)] = 0

        del wi_select, black, white, white_dict, black_dict
        gc.collect()

        # Demosaic raw image
        logger.info("Demosaicing sensor image...")
        img = demosaic.get_demosaiced(raw_img,
                                      pattern='GRBG',
                                      method=demosaic_method)
        logger.info("...done.")

        del raw_img
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

        # Resampling the light field:
        lf = self._resample_lf(lf, method=resample_method)

        self._add_decoded_image(img=lf, num=num)

        return

    def _get_wi_db_entry(self, path: Path) -> ndarray:
        """Get the database entry for a white image.
        Entries include the zoomStep and focusStep setting of
        the white image to be used for decoding.

        Args:
            path: Path to the white image, relative to Camera main folder.
                  Used as database identfication

            metadata: The metadata to the whiteimage

        Returns:
            A structured array containing the path, focal_length, zoomStep and
            focusStep setting of the white image.

        """
        img = self._get_wi(self.path / path, process=False)
        metadata = img.meta

        zoomstep = metadata['master']['picture']['frameArray'][0]['frame'][
            'metadata']['devices']['lens']['zoomStep']
        focusstep = metadata['master']['picture']['frameArray'][0]['frame'][
            'metadata']['devices']['lens']['focusStep']

        focal_length = self._get_focal_length(zoomstep)

        # Create structured array for white image table
        db_entry = np.array(
            [(path, focal_length, zoomstep, focusstep)],
            dtype=[('path', 'U9999'), ('focal_length', 'f8'), ('zoomStep', 'i4'), ('focusStep', 'i4')])

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
        wi = imageio.imread(self.path / path, format=self._format).copy()

        if process:
            # Contrast stretch
            wi -= wi.min()
            try:
                wi /= wi[2500:-2500, 3500:-3500].max()
            except ValueError or KeyError:
                wi /= wi.max()
            wi = np.clip(wi, 0, 1)

        return wi

    @staticmethod
    def _get_focal_length(focus_step: int):
        """Calculate focal length in meter from focus_step setting."""

        # TODO: Implement interpolation of some kind
        res = None
        if focus_step > 356:
            res = 0.03

        elif 64 <= focus_step <= 81:
            res = 0.047

        elif -563 <= focus_step <= -559:
            res = 0.117

        elif -1041 <= focus_step <= -1000:
            res = 0.249

        elif focus_step <= -1042:
            res = 0.250

        return res
