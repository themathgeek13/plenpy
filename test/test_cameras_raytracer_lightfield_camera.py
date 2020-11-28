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

from pathlib import Path

import plenpy.logg
from plenpy import __name__ as APPNAME
from plenpy import testing
from plenpy.cameras.raytracer_lightfield_camera import RayTracerLF
from plenpy.lightfields import LightField

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

APP_FOLDER = testing.appdata_dir(APPNAME)

# Global test names
BASE_FOLDERPATH = APP_FOLDER.absolute()
CAMERA_FOLDERPATH = BASE_FOLDERPATH / "cameras/RayTracerSmall/"

FILES = [CAMERA_FOLDERPATH / "Calibration/cal_data.npz",
         CAMERA_FOLDERPATH / "Calibration/cal_own.npz",
         CAMERA_FOLDERPATH / "Calibration/cal_dans.npz"]


def create_test_camera():
    """Download the Raytracer test camera, if not already present."""
    testing.needs_internet()

    for ext in [".png", ".json"]:
        testing.get_remote_file("cameras/RayTracerSmall/Calibration/cal_img" + ext)
        testing.get_remote_file("cameras/RayTracerSmall/Images/example_img" + ext)

    return


def delete_test_camera():
    """ Helper function to delete temporary files.
    """
    for file in FILES:
        try:
            file.unlink()
        except FileNotFoundError:
            pass

    return


def test_lf_cam_init():
    create_test_camera()

    try:
        test_camera = RayTracerLF(CAMERA_FOLDERPATH,
                                  grid_type='hex',
                                  ml_size=14,
                                  ml_focal_length=40e-6)

        # Check member variables
        assert test_camera._microlensSize == 14
        assert test_camera._microlensRadius == 7
        assert test_camera._gridType == 'hex'
        assert test_camera._microlensFocalLength == 40e-6
        assert test_camera._calibrationDB is None
        assert test_camera._whiteImageDB is None
        assert test_camera._calDataFilename == Path('Calibration/cal_data.npz')

    finally:
        # Always clean up
        delete_test_camera()

    return


def test_camera_calibrate():
    """Perform tests on LytroIllum camera with cropped images
    to speed up testing.
    """
    create_test_camera()

    try:
        test_camera = RayTracerLF(CAMERA_FOLDERPATH,
                                  grid_type='hex',
                                  ml_size=14,
                                  ml_focal_length=40e-6,
                                  format='PNG-PIL')


        # Calibrate
        test_camera.calibrate(method='dans', force=True)

        # Load
        test_camera.load_sensor_image(0)
        test_camera.decode_sensor_image(0)
        lf = test_camera.get_decoded_image(0)

        test_camera.load_sensor_image(0)
        test_camera.decode_sensor_image(0)
        lf_dans = test_camera.get_decoded_image(0)

        assert lf.shape == (13, 13, 36, 37, 3)
        assert type(lf) == LightField

    finally:
        # Always clean up
        delete_test_camera()

    return
