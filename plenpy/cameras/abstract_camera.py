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
Module defining the :class:`AbstractCamera` base class.

This is the base class that all implemented cameras are derived from.
The derived camera classes are defined in their respective modules.

"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, Optional, List, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray

import plenpy.logg
from plenpy.cameras.sensor_image import SensorImage
from plenpy.lightfields import LightField
from plenpy.utilities.misc import get_avail_extensions

logger = plenpy.logg.get_logger()

__all__ = ['AbstractCamera']


# Camera is the abstract base class for all implemented camera models
class AbstractCamera(ABC):
    """AbstractCamera base class.

    This is the base class that all actual camera classes are derived from.
    This class is abstract and objects cannot be initialized directly from it.
    Implements an API for all the basic functions needed by a camera model.

    The structure of a camera folder has to be: \n
    <camera-folder> \n
                ├── Images/ \n
                ├── Calibration/ \n
                └── Reference/      [optional]

    Attributes:
        path (Path): Absolute system path the main camera folder.

        imageFolderPath (Path): Relative path to image folder.

        calibrationFolderPath (Path): Relative path to calibration folder.

        referenceFolderPath (Path): Relative path to reference folder.

        _sensorImages (Dict[int, SensorImage]): Dict of raw sensor images.

        _decodedImages (Dict[int, Union[ndarray, LightField]]): Dict of
            decoded sensor images.

        _isCalibrated (bool): Flag indicating camera's calibration status.

    """

    def __init__(self, path: Any):
        """:class:`AbstractCamera` base class initialization.

        Args:
            path: System path to the main camera folder.

        """

        # Set the camera folder path as absolute path and add trailing "/"
        # if necessary.
        self.path: Path = Path(path).absolute()
        logger.info(f"Initializing camera at {self.path}.")

        # Check that necessary camera folder exists
        if not self.path.is_dir():
            raise ValueError(f"Camera folder {self.path} does "
                             "not exist. "
                             "Please create it before initializing a camera.")

        # Set image, calibration and reference folder (absolute) paths
        self.imageFolderPath: Path = Path("Images/")
        self.calibrationFolderPath: Path = Path("Calibration/")
        self.referenceFolderPath: str = Path("Reference/")

        # Check that necessary folders Images/ and Calibration/ exist.
        if not (self.path / self.imageFolderPath).is_dir():
            raise ValueError("'Images' subdirectory does not exist. "
                             "Please create it before initializing a camera.")

        if not (self.path / self.calibrationFolderPath).is_dir():
            raise ValueError("'Calibration' subdirectory does not exist. "
                             "Please create it before initializing a camera.")

        # Read all the sensor images stored in the image folder, as a
        # key value pair. Do not load them.
        logger.info(f"Reading sensor image paths from {self.imageFolderPath.absolute()}.")

        self._sensorImages: Dict[int, SensorImage] = {}

        files_found = []
        # Recursively find files with a readable image file extension
        for extension in get_avail_extensions():
            for file in (self.path / self.imageFolderPath).glob('**/*' + extension):
                files_found.append(file)

        # Sort outcome and copy
        files_found = sorted(files_found)
        for i in range(len(files_found)):
            self._sensorImages[i] = SensorImage(files_found[i])

        # At initialization, no images is decoded
        self._decodedImages: Dict[int, Union[ndarray, LightField]] = {}

        # At initialization, Camera is not calibrated
        self._isCalibrated: bool = False
        return

    def load_sensor_image(self,
                          num: Union[int, List[int], str],
                          format: Optional[str] = None):
        """Load a sensor image.

        Args:
            num: Specifies the number of the image to load.
                If a list is passed, multiple images are loaded.
                If "all" is specified, all images
                in the ``Images`` folder are loaded.

            format: The image format to be used to read the file.
                By default, imageio selects an appropriate one based
                on the filename and its contents.
                Use :func:`plenpy.utilities.misc.get_avail_extensions()`
                to get a list of available formats.

        """

        if num == "all":
            num = range(0, len(self._sensorImages))

        elif isinstance(num, int):
            num = [num]

        for no in num:
            try:
                self._sensorImages[no].load(format)
            except KeyError:
                raise ValueError(
                    f"There is no Sensor image with number '{no}'.")

        return

    def unload_sensor_image(self,
                            num: Union[int, List[int], str]):
        """Unload a sensor image.

        Args:
            num: Specifies the number of the image to load.
                If a list is passed, multiple images are loaded.
                If "all" is specified, all images
                in the ``Images`` folder are loaded.

        """

        if num == "all":
            num = range(0, len(self._sensorImages))

        elif isinstance(num, int):
            num = [num]

        for no in num:
            try:
                self._sensorImages[no].unload()
            except KeyError:
                raise ValueError(
                    f"There is no Sensor image with number '{no}'.")

        return

    def load_sensor_image_from_path(self,
                                    path: Any,
                                    format: Optional[str] = None):
        """Load a sensor image from a system path image.

        Args:
            path: System path to the image file.

            format: The format to be used to read the file.
                By default, imageio selects an appropriate one based
                on the filename and its contents.
                Use :func:`plenpy.utilities.misc.get_avail_extensions()`
                to get a list of available formats.

        """
        # Get highest number of sensor image
        num = len(self._sensorImages)

        # Add sensor image to _sensorImages dict and load it
        self._sensorImages[num] = SensorImage(Path(path))
        self._sensorImages[num].load(format)
        return

    def get_sensor_image(self, num: int) -> ndarray:
        """Get a raw sensor image.

        Args:
            num: Specifies the number of the image to load.

        Returns:
            Sensor image of number `num`.

        """

        try:
            return self._sensorImages[num].get_image()
        except KeyError:
            raise ValueError(f"There is no Sensor image with number '{num}'.")

    def get_image_metadata(self, num: int) -> dict:
        """Get the metadata of a raw sensor image.

        Args:
            num: Specifies the number of the image to load the metadata from.

        Returns:
            Metadata of the raw sensor image specified by ``num``.

        """
        try:
            return self._sensorImages[num].get_metadata()
        except KeyError:
            raise ValueError(f"There is no Sensor image with number '{num}'.")

    def list_sensor_images(self):
        """Print a list of sensor images.

         Prints all sensor images found in the camera's image subfolder.
         Denotes which images have been loaded.

        """
        print("Num: \t Path:\t Loaded:\t Decoded:\n")
        print("-------------------------------------------")
        for key, value in self._sensorImages.items():
            print(key, "\t",
                  value.imagePath.relative_to(self.path), "\t",
                  "Loaded" if value.is_loaded() else "Not Loaded", "\t",
                  "Decoded" if value.is_decoded() else "Not Decoded")
        print("-------------------------------------------\n")
        return

    def show_sensor_image(self, num: int):
        """Plot a raw sensor image.

        Plots the sensor image specified by ``num`` using ``matplotlib``.

        Args:
            num: Number of the sensor image to be plotted.

        """
        plt.figure()
        plt.imshow(np.squeeze(self._sensorImages[num].get_image()))
        plt.show()
        return

    def show_image_metadata(self, num: int):
        """Print the metadata of a raw sensor image.

        Prints the metadata of the sensor image specified by ``num`` on the
        standard output.

        Args:
            num: Number of the sensor image.

        """
        print(json.dumps(dict(self._sensorImages[num].get_metadata()),
                         indent=4,
                         sort_keys=True))
        return

    def _add_decoded_image(self,
                           img: Union[ndarray, LightField],
                           num: int):
        """Add decoded image to dictionary of decoded images.

        The dictionary of decoded images collects all decoded images.
        The function adds an image to that dictionary using the key
        which is specified by ``num``.

        Args:
            img : Decoded image to add.

            num : Number of the sensor image that the image is decoded from.

        """
        self._decodedImages[num] = img
        self._sensorImages[num].set_decoded(True)
        return

    def get_decoded_image(self, num: int) -> Union[ndarray, LightField]:
        """Get decoded image.

        Args:
            num: Number of the sensor image that the image is decoded from.

        Returns:
            Decoded image decoded from sensor image of number ``num``.
        """
        try:
            # Check if image has been decoded
            if not self._sensorImages[num].is_decoded():
                raise ValueError(f"Image number {num} is not decoded yet.")

            return self._decodedImages[num]

        except KeyError:
            raise ValueError(f"There is no Sensor image with number '{num}'.")

    # Define all the abstract class methods (virtual functions)
    @abstractmethod
    def calibrate(self):
        """Calibrate the camera.

        Calibrates the camera using the calibration metadata and/or
        the files provided in the Calibration/ subfolder.
        The implementation depends on the camera in use.

        """
        pass

    @abstractmethod
    def decode_sensor_image(self, num: int):
        """Decode a sensor image.

        The implementation depends on the camera in use.

        Args:
            num: Number of the sensor image that is to be decoded.

        """
        pass

    @abstractmethod
    def show_decoded_image(self, num: int):
        """Show a decoded sensor image.

        Shows the image decoded from sensor image ``num`` using ``matplotlib``
        This can be a regular RGB image as well as a hyperspectral image
        or a :class:`plenpy.lightfields.LightField` object.

        Args:
            num: Number of the sensor image that the image is decoded from.

        """
        pass
