# Copyright (C) 2018  The Plenpy Authors
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


"""Tests for plenpy.cameras.abstract_lightfield_camera module.

"""

import imageio
from pathlib import Path
import numpy as np
from typing import Tuple
from uuid import UUID
from skimage.color import rgb2gray
from colour_demosaicing import mosaicing_CFA_Bayer

import plenpy.logg

from plenpy.utilities import demosaic
from plenpy import testing
from plenpy.cameras.rgb_camera import RgbCamera

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


# Global test names
TEST_IMG_FILENAME = "images/crayons.jpeg"

IMAGES_FOLDERPATH = Path("Images/")
CALIBRATION_FOLDERPATH = Path("Calibration/")

FILES = [IMAGES_FOLDERPATH / "img_1.png",
         IMAGES_FOLDERPATH / "img_2.png",
         IMAGES_FOLDERPATH / "img_3.png",
         CALIBRATION_FOLDERPATH / "img_cal_1.png",
         CALIBRATION_FOLDERPATH / "img_cal_2.png",
         CALIBRATION_FOLDERPATH / "img_cal_3.png"]

FOLDERS = [CALIBRATION_FOLDERPATH, IMAGES_FOLDERPATH]

PATTERN = 'RGGB'


def mosaic(img):
    return mosaicing_CFA_Bayer(img, pattern=PATTERN)


# Create a test camera
def create_test_camera() -> Tuple[Path, UUID]:
    """ Helper function to create a test camera folder structure with images.
    """
    testing.needs_internet()

    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Create folder structure
        for folder in FOLDERS:
            try:
                (path / folder).mkdir()
            except FileExistsError:
                pass

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imageio.imread(img_file)

        # Mosaic
        img = mosaic(img)

        # Save test image, rotated image and BW image
        imageio.imsave(path / FILES[0], img)
        imageio.imsave(path / FILES[1], np.flipud(img))
        imageio.imsave(path / FILES[2], (255*img).astype(np.uint8))
        imageio.imsave(path / FILES[3], img)
        imageio.imsave(path / FILES[4], np.flipud(img))
        imageio.imsave(path / FILES[5], (255*img).astype(np.uint8))

    except:
        delete_test_camera(path, id)

    return path, id


def delete_test_camera(path, id):
    """ Helper function to delete the test camera created by
    :func:`create_test_camera()`.
    """
    for file in FILES:
        try:
            (path / file).unlink()
        except FileNotFoundError:
            pass

    for folder in reversed(FOLDERS):
        try:
            (path / folder).rmdir()
        except FileNotFoundError:
            pass

    testing.remove_tmp_folder(id)

    return


def test_rgb_camera():
    testing.needs_internet()

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        test_camera = RgbCamera(path, bayer_pattern=PATTERN)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imageio.imread(img_file)
        img = mosaic(img)
        img = demosaic.get_demosaiced(img, pattern=PATTERN, method='malvar2004')
        img = np.clip(img, 0, 255) / 255.0

        assert test_camera._bayerPattern == PATTERN

        test_camera.load_sensor_image(0)
        test_camera.decode_sensor_image(0)
        tmp = test_camera.get_decoded_image(0)

        # Test that demosaiced image is close to original
        assert img.shape == tmp.shape
        assert np.allclose(img, tmp, atol=0.05)

    finally:
        # Always clean up
        delete_test_camera(path, id)

    return


def test_rgb_camera_show():

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        test_camera = RgbCamera(path, bayer_pattern=PATTERN)

        test_camera.load_sensor_image(0)
        test_camera.decode_sensor_image(0)

        assert test_camera.show_decoded_image(0) is None

    finally:
        # Always clean up
        delete_test_camera(path, id)

    return
