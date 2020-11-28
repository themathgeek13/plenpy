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


"""Tests for plenpy.cameras.sensor_image module.

"""

import numpy as np
from imageio import imread
from pytest import raises

import plenpy.logg
from plenpy import testing
from plenpy.cameras.sensor_image import SensorImage

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

# Global test names
TEST_IMG_FILENAME = "images/crayons.jpeg"


def test_sensor_image_init():

    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    test_sensor_image = SensorImage(img_file)

    assert test_sensor_image.imagePath == img_file
    assert test_sensor_image._SensorImage__image is None
    assert test_sensor_image.is_loaded() is False
    assert test_sensor_image.is_decoded() is False
    assert test_sensor_image._SensorImage__metadata is None

    return


def test_sensor_image_load():

    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    img = imread(img_file)
    test_sensor_image = SensorImage(img_file)

    assert test_sensor_image.is_loaded() is False
    assert test_sensor_image.is_decoded() is False

    test_sensor_image.load()

    assert test_sensor_image.is_loaded() is True
    assert test_sensor_image.is_decoded() is False
    assert np.array_equal(test_sensor_image._SensorImage__image, img)
    assert test_sensor_image._SensorImage__metadata is not None

    return


def test_sensor_image_get_image():
    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    img = imread(img_file)
    test_sensor_image = SensorImage(img_file)

    with raises(RuntimeError) as cm:
        test_sensor_image.get_image()

    assert ("The image is not loaded yet. "
            "Load the image with 'load()' first.") == str(cm.value)

    test_sensor_image.load()
    assert np.array_equal(test_sensor_image.get_image(), img)

    return


def test_sensor_image_get_metadata():
    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    img = imread(img_file)
    test_sensor_image = SensorImage(img_file)

    with raises(RuntimeError) as cm:
        test_sensor_image.get_metadata()

    assert ("The image is not loaded yet. "
            "Load the image with 'load()' first.") == str(cm.value)

    test_sensor_image.load()
    assert np.array_equal(test_sensor_image.get_metadata(), img._meta)

    return


def test_sensor_image_set_decoded():
    img_file = testing.get_remote_file(TEST_IMG_FILENAME)
    test_sensor_image = SensorImage(img_file)

    with raises(ValueError) as cm:
        test_sensor_image.set_decoded("nonsense")

    assert "The passed value 'nonsense' is not of type Bool" == str(cm.value)

    with raises(ValueError) as cm:
        test_sensor_image.set_decoded(3)

    assert "The passed value '3' is not of type Bool" == str(cm.value)

    assert test_sensor_image.is_decoded() is False
    test_sensor_image.set_decoded(True)
    assert test_sensor_image.is_decoded() is True

    return
