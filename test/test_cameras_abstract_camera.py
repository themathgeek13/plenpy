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


"""Tests for plenpy.cameras.abstract_camera module.

"""

import shutil
from pathlib import Path
from unittest.mock import patch
from typing import Tuple
from uuid import UUID

import numpy as np
from imageio import imread, imsave
from pytest import raises
from skimage.color import rgb2gray

import plenpy.logg
from plenpy import testing

from plenpy.cameras.abstract_camera import AbstractCamera

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

# Create Mock Patch Abstract Base Classes
ABC_PATCH = patch.multiple(AbstractCamera, __abstractmethods__=set())


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
        img = imread(img_file)

        # Save test image, rotated image and BW image
        imsave(path / FILES[0], img)
        imsave(path / FILES[1], np.flipud(img))
        imsave(path / FILES[2], (255*rgb2gray(img)).astype(np.uint8))
        imsave(path / FILES[3], img)
        imsave(path / FILES[4], np.flipud(img))
        imsave(path / FILES[5], (255*rgb2gray(img)).astype(np.uint8))

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


def test_abstract_camera_init():

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        # Check member variables
        assert test_camera.imageFolderPath == IMAGES_FOLDERPATH
        assert test_camera.path == path
        assert test_camera.calibrationFolderPath == CALIBRATION_FOLDERPATH
        assert test_camera.referenceFolderPath == Path("Reference/")

        assert len(test_camera._sensorImages) == 3
        assert len(test_camera._decodedImages) == 0
        assert test_camera._isCalibrated is False

        # Check error for noon existing camera folder
        with raises(ValueError) as cm:
            new_test_camera = AbstractCamera(Path("./nonsense/"))
        assert (
                   f"Camera folder {Path('./nonsense/').absolute()} does "
                   "not exist. "
                   "Please create it before initializing a camera.") == str(
            cm.value)

        # Check error if directory is missing
        shutil.rmtree(path / IMAGES_FOLDERPATH)

        with raises(ValueError) as cm:
            new_test_camera = AbstractCamera(path)
        assert (
            "'Images' subdirectory does not exist. "
            "Please create it before initializing a camera.") == str(cm.value)

        path2, id2 = create_test_camera()
        shutil.rmtree(path2 / CALIBRATION_FOLDERPATH)

        with raises(ValueError) as cm:
            new_test_camera = AbstractCamera(path2)
        assert (
            "'Calibration' subdirectory does not exist. "
            "Please create it before initializing a camera.") == str(cm.value)

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
        delete_test_camera(path2, id2)

    return


def test_abstract_camera_load_sensor_image():

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imread(img_file)

        test_camera.load_sensor_image(0)
        test_camera.load_sensor_image(1)

        # Check that images have been read correctly

        assert test_camera._sensorImages[0].is_loaded() is True
        assert test_camera._sensorImages[1].is_loaded() is True
        assert test_camera._sensorImages[2].is_loaded() is False

        assert np.array_equal(img,
                              test_camera.get_sensor_image(0))
        assert np.array_equal(np.flipud(img),
                              test_camera.get_sensor_image(1))

        with raises(ValueError) as cm:
            test_camera.load_sensor_image(3)
        assert (
            f"There is no Sensor image with number '3'.") == str(cm.value)

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_load_sensor_image_list():
    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imread(img_file)

        test_camera.load_sensor_image([0, 2])

        # Check that images have been read correctly

        assert test_camera._sensorImages[0].is_loaded() is True
        assert test_camera._sensorImages[1].is_loaded() is False
        assert test_camera._sensorImages[2].is_loaded() is True

        assert np.array_equal(img,
                              test_camera.get_sensor_image(0))
        assert np.array_equal((rgb2gray(img) * 255).astype(np.uint8),
                              test_camera.get_sensor_image(2))

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_load_sensor_image_all():
    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imread(img_file)

        test_camera.load_sensor_image("all")

        # Check that images have been read correctly

        assert test_camera._sensorImages[0].is_loaded() is True
        assert test_camera._sensorImages[1].is_loaded() is True
        assert test_camera._sensorImages[2].is_loaded() is True

        assert np.array_equal(img,
                              test_camera.get_sensor_image(0))
        assert np.array_equal(np.flipud(img),
                              test_camera.get_sensor_image(1))
        assert np.array_equal((rgb2gray(img) * 255).astype(np.uint8),
                              test_camera.get_sensor_image(2))

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_load_sensor_image_from_path():
    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imread(img_file)

        assert len(test_camera._sensorImages) == 3

        test_camera.load_sensor_image_from_path(path=img_file)

        assert len(test_camera._sensorImages) == 4

        # Check that images have been read correctly
        assert np.array_equal(img,
                              test_camera.get_sensor_image(3))

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_get_sensor_image():

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        with raises(RuntimeError) as cm:
            test_camera.get_sensor_image(0)
        # Error message is checked in sensor_image test
        assert cm.type is RuntimeError

        with raises(ValueError) as cm:
            test_camera.get_sensor_image(4)
        assert (
            f"There is no Sensor image with number '4'.") == str(cm.value)

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_get_metadata():

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        with raises(ValueError) as cm:
            test_camera.get_image_metadata(4)
        assert (
            f"There is no Sensor image with number '4'.") == str(cm.value)

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_decoded_image():
    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)

        img_file = testing.get_remote_file(TEST_IMG_FILENAME)
        img = imread(img_file)

        # Load image
        test_camera.load_sensor_image(0)
        assert test_camera._sensorImages[0].is_loaded() is True
        assert test_camera._sensorImages[0].is_decoded() is False

        # Decode image
        test_camera._add_decoded_image(img, 0)
        assert test_camera._sensorImages[0].is_decoded() is True
        assert np.array_equal(img, test_camera.get_decoded_image(0))

        with raises(ValueError) as cm:
            test_camera.get_decoded_image(1)
            assert f"Image number 1 is not decoded yet." == str(cm.value)

        with raises(ValueError) as cm:
            test_camera.get_decoded_image(4)
        assert (
            f"There is no Sensor image with number '4'.") == str(cm.value)

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return


def test_abstract_camera_rest(capsys):

    try:
        # Create camera folder structure
        path, id = create_test_camera()

        # Create a mock patch to test abstract base class
        ABC_PATCH.start()
        test_camera = AbstractCamera(path)
        test_camera.load_sensor_image(0)

        # Basic "no-fail-tests". Actual output is nasty to test
        assert test_camera.list_sensor_images() is None
        assert test_camera.show_sensor_image(0) is None
        assert test_camera.show_image_metadata(0) is None
        assert test_camera.show_decoded_image(0) is None
        assert test_camera.decode_sensor_image(0) is None
        assert test_camera.calibrate() is None

        ABC_PATCH.stop()

    finally:
        # Always clean up
        delete_test_camera(path, id)
    return
