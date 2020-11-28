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
Module defining the :class:`SensorImage` class.

"""

from pathlib import Path
from typing import Optional, Any

import gc
import imageio
import numpy as np
from numpy.core.multiarray import ndarray

__all__ = ['SensorImage']


class SensorImage(object):
    """Class representing a raw sensor image.

    The raw sensor images are the non decoded images that a camera takes.
    Usually, this is just a 2D greyscale image. The class stores the system
    path of the image and loads the image into an :class:`.numpy.ndarray` only
    when :func:'load()' is called explicitly, not upon initialization.
    This is useful as the camera's ``Images`` subfolder may contain many,
    ofter vary large, raw images that do not all have to be loaded into memory.

    Attributes:
        imagePath (str): System path to the raw image file.
        __image (ndarray): Loaded image as numpy array.
        __loaded (bool): Flag indicating whether image has been loaded.
        __decoded (bool): Flag indicating whether image has been decoded.
        __metadata (dict): = Image metadata.

    """

    def __init__(self, image_path: Any):
        """:class:`SensorImage` class initialization.

        Upon initialization, only the system path of the image is stored and
        the corresponding flags are set. The image data and metadata are
        not loaded into memory.

        Args:
            image_path: System path the the raw sensor image.
                Both relative and absolute values are possible.
        """
        self.imagePath: Path = Path(image_path)
        self.__image: ndarray = None
        self.__loaded: bool = False
        self.__decoded: bool = False
        self.__metadata: dict = None

    def load(self, format: Optional[str] = None):
        """
        Load the image specified by ``imagePath`` attribute.

        Args:
            format: The image format to be used to read the file.
                By default, imageio selects an appropriate one based
                on the filename and its contents.
                Use :func:`plenpy.utilities.misc.get_avail_extensions()`
                to get a list of available formats.

        """
        im = imageio.imread(self.imagePath, format)
        self.__metadata = im.meta
        self.__image = np.asarray(im)
        self.__loaded = True
        return

    def unload(self):
        """
        Unload the image specified by ``imagePath`` attribute.

        Args:
            format: The image format to be used to read the file.
                By default, imageio selects an appropriate one based
                on the filename and its contents.
                Use :func:`plenpy.utilities.misc.get_avail_extensions()`
                to get a list of available formats.

        """

        del self.__metadata
        del self.__image
        gc.collect()

        self.__loaded = False
        return

    def is_loaded(self) -> bool:
        """Get bool specifying whether the raw image is loaded.

        Returns:
            ``True`` if image is loaded, ``False`` otherwise.

        """
        return self.__loaded

    def is_decoded(self) -> bool:
        """Get bool specifying whether the raw image is decoded.

        Returns:
            ``True`` if image is decoded, ``False`` otherwise.

        """
        return self.__decoded

    def set_decoded(self, value: bool):
        """Specify whether the image has been decoded or not.

        Args:
            value: Flag whether the image has been decoded or not.

        """
        if isinstance(value, bool) is not True:
            raise ValueError(
                f"The passed value '{value}' is not of type Bool")

        self.__decoded = value
        return

    def get_image(self) -> ndarray:
        """Get the raw sensor image.

        Returns:
            The raw sensor image as numpy ndarray.

        """
        if not self.is_loaded():
            raise RuntimeError(
                "The image is not loaded yet. "
                "Load the image with 'load()' first.")

        return self.__image

    def get_metadata(self) -> dict:
        """Get the metadata of the raw sensor image.

        Returns:
            The metadata of the raw sensor image.

        """
        if not self.is_loaded():
            raise RuntimeError(
                "The image is not loaded yet. "
                "Load the image with 'load()' first.")

        return self.__metadata
